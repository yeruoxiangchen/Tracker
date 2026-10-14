"""Same-tokenizer, frozen-teacher OPD; no decode/re-encode of sampled completions."""
import contextlib
import copy
import math
import types

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .model import CachedVideoQA, assistant_only_preprocess, load_projector, qwen_core
from .visual_bridge import VisualBridge


def disable_dropout(model):
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0.0


@torch.no_grad()
def sample_completions(model, features, question, max_new_tokens=96, prompt=None,
                       do_sample=True):
    if features.shape[0] != 1:
        raise ValueError("initial OPD implementation requires per-rank batch size 1")
    was_training = model.training
    model.eval()
    try:
        core, embeddings, attention, _, input_ids = model._embed(
            features, [question], [""], for_generation=True,
            prompts=[prompt] if prompt is not None else None)
        ids = core.generate(input_ids=input_ids, inputs_embeds=embeddings,
                            attention_mask=attention, do_sample=do_sample,
                            temperature=1.0, top_p=1.0, top_k=0, num_beams=1,
                            repetition_penalty=1.0, use_cache=True,
                            max_new_tokens=max_new_tokens,
                            eos_token_id=model.backbone.llm_tokenizer.im_end_id,
                            pad_token_id=model.backbone.llm_tokenizer.eod_id)
        completion = ids[0, input_ids.shape[1]:].detach()
        eos = (completion == model.backbone.llm_tokenizer.im_end_id).nonzero(as_tuple=False)
        finished = len(eos) > 0
        if finished:
            completion = completion[:int(eos[0].item()) + 1]
        return completion, finished
    finally:
        model.train(was_training)


def completion_logits(model, features, question, completion, prompt=None, answer_only=False):
    """Return positions predicting completion[0..n-1], including EOS but never padding."""
    if completion.ndim != 1 or completion.numel() == 0 or features.shape[0] != 1:
        raise ValueError("one nonempty completion per call is required")
    core, prefix, attention, _, _ = model._embed(
        features, [question], [""], for_generation=True,
        prompts=[prompt] if prompt is not None else None)
    prefix_length = prefix.shape[1]
    if prefix_length + completion.numel() > model.backbone.max_txt_len:
        raise ValueError("OPD completion exceeds max_txt_len")
    # The last target token is not needed as input to predict itself.
    token_embeddings = core.transformer.get_input_embeddings()(completion[:-1].unsqueeze(0))
    embeddings = torch.cat((prefix, token_embeddings), dim=1)
    mask = torch.cat((attention, torch.ones((1, completion.numel() - 1),
                      dtype=attention.dtype, device=attention.device)), dim=1)
    if answer_only:
        hidden = core.transformer(inputs_embeds=embeddings, attention_mask=mask,
                                  use_cache=False, return_dict=True)[0]
        return core.lm_head(hidden[0, prefix_length - 1:prefix_length - 1 + completion.numel()])
    output = core(inputs_embeds=embeddings, attention_mask=mask, use_cache=False, return_dict=True)
    return output.logits[0, prefix_length - 1:prefix_length - 1 + completion.numel()]


def _divergence(student, teacher, objective):
    log_s = F.log_softmax(student.float(), dim=-1)
    log_t = F.log_softmax(teacher.detach().float(), dim=-1)
    if objective == "forward_kl":
        return (log_t.exp() * (log_t - log_s)).sum()
    if objective != "jsd":
        raise ValueError("unknown distillation objective")
    log_m = torch.logaddexp(log_s, log_t) - math.log(2.0)
    return 0.5 * ((log_s.exp() * (log_s - log_m)).sum() +
                  (log_t.exp() * (log_t - log_m)).sum())


def distillation_loss(student, teacher, objective="jsd", chunk_tokens=8):
    if student.shape != teacher.shape or student.ndim != 2 or not student.shape[0]:
        raise ValueError("teacher/student answer logits must have identical nonempty [tokens,vocab] shape")
    if chunk_tokens <= 0:
        raise ValueError("chunk_tokens must be positive")
    total = student.new_zeros((), dtype=torch.float32)
    for start in range(0, student.shape[0], chunk_tokens):
        s = student[start:start + chunk_tokens]
        t = teacher[start:start + chunk_tokens]
        if torch.is_grad_enabled() and s.requires_grad:
            total = total + checkpoint(lambda x, y: _divergence(x, y, objective),
                                       s, t, use_reentrant=False)
        else:
            total = total + _divergence(s, t, objective)
    return total / student.shape[0]


class SharedFrozenTeacher:
    """One frozen Qwen base, separate policy/teacher adapters and visual branches.

    Teacher scoring must finish before the student forward/backward pair because
    gradient checkpointing recomputes using the currently active policy adapter.
    """
    def __init__(self, student, args):
        from pathlib import Path
        self.student = student
        llm = student.backbone.llm_model
        self.policy_names = {name for name, parameter in llm.named_parameters() if parameter.requires_grad}
        llm.load_adapter(args.teacher_adapter, adapter_name="opd_teacher", is_trainable=False)
        exact_precision = getattr(args, "precision_protocol", "legacy") == "fp32_trainables_v1"
        if exact_precision:
            # PEFT initially rounds adapters to the base dtype; reload after promotion,
            # just as build_text_model does for the policy adapter.
            from peft.utils.save_and_load import load_peft_weights, set_peft_model_state_dict
            for name, parameter in llm.named_parameters():
                if ".opd_teacher." in name:
                    parameter.data = parameter.data.float()
            set_peft_model_state_dict(llm, load_peft_weights(args.teacher_adapter, device="cpu"),
                                      adapter_name="opd_teacher")
        self.llm = llm
        branch = nn.Module()
        branch.llm_model = llm
        branch.llm_tokenizer = student.backbone.llm_tokenizer
        branch.replace_image_string = student.backbone.replace_image_string
        branch.replace_image_token_id = student.backbone.replace_image_token_id
        branch.max_txt_len = student.backbone.max_txt_len
        branch.preprocess = types.MethodType(assistant_only_preprocess, branch)
        branch.llm_proj = copy.deepcopy(student.backbone.llm_proj)
        load_projector(branch, args.teacher_projector)
        if student.feature_stage == "eva":
            visual_root = Path(args.visual_checkpoint_root or args.checkpoint_root)
            branch.visual_bridge = VisualBridge(
                args.bridge_init, visual_root / "ckpt/bert-base-uncased", rank=0,
                trainable=False, weights=args.teacher_qformer_adapter,
                compute_dtype=getattr(args, "dtype", "bf16") if exact_precision else "fp32")
            branch.visual_bridge = branch.visual_bridge.to(
                device=next(student.parameters()).device,
                dtype=torch.float32 if exact_precision else torch.bfloat16)
        for name, parameter in branch.named_parameters():
            if not name.startswith("llm_model."):
                parameter.requires_grad = False
        self.model = CachedVideoQA(branch, task_instruction=student.task_instruction,
                                   feature_stage=student.feature_stage)
        disable_dropout(student)
        disable_dropout(self.model)
        self.restore_policy()

    def restore_policy(self):
        self.llm.set_adapter("default")
        # PEFT set_adapter can toggle requires_grad: restore exactly the policy set.
        for name, parameter in self.llm.named_parameters():
            parameter.requires_grad = name in self.policy_names

    def validate_initial_copy(self):
        """Check exact weights at initial diagnosis, not after student training.

        BF16 logit maxima alone are not a weight-identity check: GPU kernels can
        differ at the last bits even on repeated calls to the same policy.
        """
        for attribute in ("llm_proj", "visual_bridge"):
            if not hasattr(self.student.backbone, attribute):
                continue
            original = getattr(self.student.backbone, attribute).state_dict()
            copied = getattr(self.model.backbone, attribute).state_dict()
            if original.keys() != copied.keys() or any(not torch.equal(p, copied[name]) for name, p in original.items()):
                raise RuntimeError("initial teacher differs in %s weights" % attribute)
        parameters = dict(self.llm.named_parameters())
        for name, parameter in parameters.items():
            if ".default." in name and not torch.equal(parameter, parameters[name.replace(".default.", ".opd_teacher.")]):
                raise RuntimeError("initial teacher differs in language adapter weights")
        return True

    @contextlib.contextmanager
    def active(self):
        was_training = self.student.training
        self.llm.set_adapter("opd_teacher")
        for parameter in self.llm.parameters():
            parameter.requires_grad = False
        self.model.eval()
        try:
            with torch.no_grad():
                yield self.model
        finally:
            self.restore_policy()
            self.student.train(was_training)


def average_gradients(parameters, world):
    """Fixed-order all-reduce handles locally truncated rollouts without skipped collectives."""
    if world <= 1:
        return
    for parameter in parameters:
        if parameter.grad is None:
            parameter.grad = torch.zeros_like(parameter)
        torch.distributed.all_reduce(parameter.grad)
        parameter.grad.div_(world)
