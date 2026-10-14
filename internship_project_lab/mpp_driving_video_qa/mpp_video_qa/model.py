import contextlib
import gc
import json
import sys
import types
from pathlib import Path

import torch
from torch import nn

from .core import SYSTEM_MESSAGE, TASK_INSTRUCTION, build_user_prompt


IGNORE_TOKEN_ID = -100
FEATURE_STAGE_PROJECTED = "projected"
FEATURE_STAGE_QFORMER = "qformer"
FEATURE_STAGE_EVA = "eva"
FEATURE_STAGES = (FEATURE_STAGE_PROJECTED, FEATURE_STAGE_QFORMER, FEATURE_STAGE_EVA)


def add_upstream(upstream_root):
    upstream_root = str(Path(upstream_root).resolve())
    if upstream_root not in sys.path:
        sys.path.insert(0, upstream_root)


def mpp_config(args):
    root = Path(args.checkpoint_root).resolve()
    visual_root_value = getattr(args, "visual_checkpoint_root", "")
    visual_root = Path(visual_root_value).resolve() if visual_root_value else root
    sharded = root / "mpp-qwen-next-sft/sharded"
    monolithic = root / "mpp-qwen-next-sft/unfreeze_llm_model.pth"
    finetuned = sharded if (sharded / "model.safetensors.index.json").is_file() else monolithic
    return {
        "vit_model": "eva_clip_g", "image_size": 224, "drop_path_rate": 0,
        "use_grad_checkpoint": False, "vit_precision": "fp16", "freeze_vit": True,
        "num_query_token": 32, "qformer_text_input": False, "freeze_qformer": True,
        "freeze_queries": True, "freeze_proj": True, "max_txt_len": args.max_txt_len,
        "llm_model": str(root / "ckpt/Qwen7B-chat"),
        "get_lora": False, "freeze_llm": True, "llm_device_map": "cpu",
        "enable_autocast": False, "load_pretrained": True, "load_finetuned": True,
        "pretrained": str(visual_root / "ckpt/blip2/blip2_pretrained_flant5xxl.pth"),
        "finetuned": str(finetuned),
    }


def assistant_only_preprocess(self, prompts, answers, image_len=32,
                              for_generation=False):
    tokenizer = self.llm_tokenizer
    newline = tokenizer("\n").input_ids
    roles = {
        "system": tokenizer("system").input_ids + newline,
        "user": tokenizer("user").input_ids + newline,
        "assistant": tokenizer("assistant").input_ids + newline,
    }
    input_rows, label_rows = [], []
    for prompt, answer in zip(prompts, answers):
        count = prompt.count("<ImageHere>")
        prompt = prompt.replace("<ImageHere>", self.replace_image_string * image_len)
        system_ids = [tokenizer.im_start_id] + roles["system"] + \
            tokenizer(SYSTEM_MESSAGE).input_ids + [tokenizer.im_end_id] + newline
        user_ids = [tokenizer.im_start_id] + roles["user"] + tokenizer(prompt).input_ids + \
            [tokenizer.im_end_id] + newline
        assistant_prefix = [tokenizer.im_start_id] + roles["assistant"]
        if for_generation:
            # Leave the assistant turn open.  Appending im_end here would tell
            # the model that an empty answer has already finished.
            input_ids = system_ids + user_ids + assistant_prefix
            labels = [IGNORE_TOKEN_ID] * len(input_ids)
        else:
            answer_ids = tokenizer(str(answer)).input_ids
            suffix = [tokenizer.im_end_id] + newline
            input_ids = system_ids + user_ids + assistant_prefix + answer_ids + suffix
            labels = ([IGNORE_TOKEN_ID] *
                      (len(system_ids) + len(user_ids) + len(assistant_prefix))
                      + answer_ids + [tokenizer.im_end_id]
                      + [IGNORE_TOKEN_ID] * len(newline))
        replacements = sum(x == self.replace_image_token_id for x in input_ids)
        if replacements != count * image_len:
            raise RuntimeError("visual placeholder tokenization mismatch")
        if len(input_ids) > self.max_txt_len:
            raise ValueError("sequence length %d exceeds max_txt_len=%d" %
                             (len(input_ids), self.max_txt_len))
        input_rows.append(input_ids)
        label_rows.append(labels)
    width = max(map(len, input_rows))
    for inputs, labels in zip(input_rows, label_rows):
        padding = width - len(inputs)
        inputs.extend([tokenizer.pad_token_id] * padding)
        labels.extend([IGNORE_TOKEN_ID] * padding)
    inputs = torch.tensor(input_rows, dtype=torch.long)
    labels = torch.tensor(label_rows, dtype=torch.long)
    return {"input_ids": inputs, "labels": labels,
            "attention_mask": inputs.ne(tokenizer.pad_token_id)}


class TextBackbone(nn.Module):
    def __init__(self, args):
        super().__init__()
        add_upstream(args.upstream_root)
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
        config = mpp_config(args)
        llm_path = config["llm_model"]
        self.llm_tokenizer = AutoTokenizer.from_pretrained(
            llm_path, use_fast=False, trust_remote_code=True,
            model_max_length=args.max_txt_len, padding_side="right")
        llm_config = AutoConfig.from_pretrained(llm_path, trust_remote_code=True)
        self.llm_model = AutoModelForCausalLM.from_pretrained(
            llm_path, config=llm_config, trust_remote_code=True, device_map="cpu")
        self.llm_tokenizer.pad_token_id = self.llm_tokenizer.eod_id
        self.replace_image_string = "<|extra_0|>"
        self.replace_image_token_id = self.llm_tokenizer(self.replace_image_string).input_ids[0]
        self.max_txt_len = args.max_txt_len
        self.llm_proj = nn.Linear(768, self.llm_model.config.hidden_size)
        for parameter in self.parameters():
            parameter.requires_grad = False

    def preprocess(self, prompts, answers, image_len=32, for_generation=False):
        return assistant_only_preprocess(
            self, prompts, answers, image_len, for_generation=for_generation)


def save_projector(backbone, path):
    """Save the MPP visual-to-language projection independently of PEFT."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "format": "mpp_llm_proj_v1",
        "in_features": backbone.llm_proj.in_features,
        "out_features": backbone.llm_proj.out_features,
        "state_dict": {
            key: value.detach().float().cpu()
            for key, value in backbone.llm_proj.state_dict().items()
        },
    }
    torch.save(state, path)


def load_projector(backbone, path):
    """Load a jointly trained projection while keeping the official shape contract."""
    payload = torch.load(path, map_location="cpu")
    if payload.get("format") != "mpp_llm_proj_v1":
        raise RuntimeError("unsupported projector checkpoint: %s" % path)
    expected = (backbone.llm_proj.in_features, backbone.llm_proj.out_features)
    actual = (payload.get("in_features"), payload.get("out_features"))
    if actual != expected:
        raise RuntimeError("projector shape mismatch: %s vs %s" % (actual, expected))
    backbone.llm_proj.load_state_dict(payload["state_dict"], strict=True)


def build_text_model(args, train, adapter="", projector=""):
    add_upstream(args.upstream_root)
    from lavis.models.base_model import BaseModel
    model = TextBackbone(args)
    BaseModel.load_checkpoint(model, mpp_config(args)["finetuned"])
    report = getattr(model, "_last_checkpoint_report", None)
    if not report or report.get("matched_key_count", 0) < 200:
        raise RuntimeError("published MPP checkpoint did not load correctly: %s" % report)
    feature_stage = getattr(args, "feature_stage", FEATURE_STAGE_PROJECTED)
    exact_precision = getattr(args, "precision_protocol", "legacy") == "fp32_trainables_v1"
    train_projector = bool(getattr(args, "train_projector", False))
    if feature_stage not in FEATURE_STAGES:
        raise ValueError("unknown feature stage: %s" % feature_stage)
    if train_projector and feature_stage == FEATURE_STAGE_PROJECTED:
        raise ValueError("--train-projector requires qformer or eva features")
    if getattr(args, "train_qformer", False) and feature_stage != FEATURE_STAGE_EVA:
        raise ValueError("--train-qformer requires --feature-stage eva")
    if feature_stage == FEATURE_STAGE_EVA:
        from .visual_bridge import VisualBridge
        visual_root = Path(getattr(args, "visual_checkpoint_root", "") or args.checkpoint_root)
        model.visual_bridge = VisualBridge(
            args.bridge_init, visual_root / "ckpt/bert-base-uncased",
            rank=getattr(args, "qformer_r", 8), alpha=getattr(args, "qformer_alpha", 16),
            trainable=train and getattr(args, "train_qformer", False),
            weights=getattr(args, "qformer_adapter", ""),
            mode=getattr(args, "qformer_mode", "auto"),
            compute_dtype=args.dtype if exact_precision else "fp32",
            gradient_checkpointing=getattr(args, "qformer_gradient_checkpointing", False))
        if model.visual_bridge.mode == "full" and not exact_precision:
            raise ValueError("full Q-Former weights require --precision-protocol fp32_trainables_v1")
    if projector:
        load_projector(model, projector)
    if feature_stage == FEATURE_STAGE_PROJECTED:
        del model.llm_proj
    else:
        for parameter in model.llm_proj.parameters():
            parameter.requires_grad = train and train_projector
    gc.collect()
    if exact_precision:
        # Do not round saved FP32 bridge/projector weights through BF16.
        model.llm_model.to(dtype=torch.bfloat16 if args.dtype == "bf16" else torch.float32)
    else:
        model = model.bfloat16() if args.dtype == "bf16" else model.float()
    expanded_adapter = bool(train and getattr(args, "expand_language_lora", False))
    if expanded_adapter:
        if not adapter:
            raise ValueError("language LoRA expansion requires a trained source adapter")
        from .lora_expansion import expand_qkv_adapter
        model.llm_model = expand_qkv_adapter(model.llm_model, adapter, args.lora_dropout)
    elif adapter:
        from peft import PeftModel
        model.llm_model = PeftModel.from_pretrained(model.llm_model, adapter,
                                                    is_trainable=train)
    elif train:
        from peft import LoraConfig, get_peft_model
        config = LoraConfig(
            target_modules=args.lora_target_modules, r=args.lora_r,
            lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
            bias="none", task_type="CAUSAL_LM")
        model.llm_model = get_peft_model(model.llm_model, config)
    if train and getattr(args, "gradient_checkpointing", False):
        # Qwen's decoder activations dominate the per-rank memory footprint.
        # This keeps the same forward function and loss while recomputing each
        # transformer block during backward instead of retaining activations.
        core = qwen_core(model.llm_model)
        core.gradient_checkpointing_enable()
        core.config.use_cache = False
    if train:
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.data = parameter.data.float()
    if exact_precision:
        for name, parameter in model.llm_model.named_parameters():
            if "lora_" in name:
                parameter.data = parameter.data.float()
        if adapter and not expanded_adapter:
            # PEFT constructs adapters at the base dtype. Reload after promotion.
            from peft.utils.save_and_load import load_peft_weights, set_peft_model_state_dict
            set_peft_model_state_dict(model.llm_model, load_peft_weights(adapter, device="cpu"),
                                      adapter_name="default")
    return model


def qwen_core(llm_model):
    # HF QWenLMHeadModel.base_model is its transformer (without lm_head).
    # Only unwrap actual wrapper children, stopping at the causal LM itself.
    from peft import PeftModel

    current = llm_model
    seen = set()
    while id(current) not in seen:
        seen.add(id(current))
        children = current._modules
        if "transformer" in children and "lm_head" in children:
            return current
        if isinstance(current, PeftModel):
            current = current.get_base_model()
            continue
        next_model = next((children[key] for key in ("module", "base_model", "model")
                           if key in children), None)
        if next_model is None:
            break
        current = next_model
    raise RuntimeError("could not resolve Qwen causal LM from %s (stopped at %s)" %
                       (type(llm_model).__name__, type(current).__name__))


class CachedVideoQA(nn.Module):
    def __init__(self, backbone, task_instruction=TASK_INSTRUCTION,
                 feature_stage=FEATURE_STAGE_PROJECTED):
        super().__init__()
        self.backbone = backbone
        self.task_instruction = task_instruction
        if feature_stage not in FEATURE_STAGES:
            raise ValueError("unknown feature stage: %s" % feature_stage)
        self.feature_stage = feature_stage

    def _project_visual_features(self, features, target_dtype):
        if self.feature_stage == FEATURE_STAGE_EVA:
            features = self.backbone.visual_bridge(features)
        if self.feature_stage == FEATURE_STAGE_PROJECTED:
            expected = qwen_core(self.backbone.llm_model).config.hidden_size
            if features.shape[-1] != expected:
                raise RuntimeError("projected cache hidden size is %d, expected %d" %
                                   (features.shape[-1], expected))
            return features.to(target_dtype)
        if not hasattr(self.backbone, "llm_proj"):
            raise RuntimeError("qformer cache requires an MPP llm_proj module")
        expected = self.backbone.llm_proj.in_features
        if features.shape[-1] != expected:
            raise RuntimeError("qformer cache hidden size is %d, expected %d" %
                               (features.shape[-1], expected))
        # Trainable projector parameters are kept in fp32 for AdamW stability.
        # Cast the small cached input to the projector dtype and return Qwen's
        # activation dtype without detaching, preserving projector gradients.
        with torch.autocast(device_type=features.device.type, enabled=False):
            projected = self.backbone.llm_proj(
                features.to(self.backbone.llm_proj.weight.dtype))
        return projected.to(target_dtype)

    def _embed(self, features, questions, answers, for_generation=False, prompts=None):
        batch, frames, _, hidden = features.shape
        query_tokens = 32 if self.feature_stage == FEATURE_STAGE_EVA else features.shape[2]
        prompts = prompts or [build_user_prompt(q, frames, self.task_instruction) for q in questions]
        encoded = self.backbone.preprocess(
            prompts, answers, query_tokens, for_generation=for_generation)
        input_ids = encoded["input_ids"].to(features.device)
        labels = encoded["labels"].to(features.device)
        attention = encoded["attention_mask"].to(features.device)
        core = qwen_core(self.backbone.llm_model)
        embeddings = core.transformer.get_input_embeddings()(input_ids).clone()
        mask = input_ids.eq(self.backbone.replace_image_token_id)
        if not torch.all(mask.sum(1).eq(frames * query_tokens)):
            raise RuntimeError("visual placeholder count does not match cached features")
        visual_features = self._project_visual_features(features, embeddings.dtype)
        embeddings[mask] = visual_features.reshape(-1, visual_features.shape[-1])
        # Qwen uses re-entrant torch checkpointing.  Because the token
        # embeddings and cached visual features are frozen, its input would
        # otherwise have requires_grad=False and checkpointing would silently
        # drop the LoRA parameter gradients inside the decoder blocks.
        if self.training and getattr(core.transformer, "gradient_checkpointing", False):
            embeddings.requires_grad_(True)
        return core, embeddings, attention, labels, input_ids

    def forward(self, features, questions, answers, prompts=None, answer_only_logits=False):
        if answer_only_logits:
            from types import SimpleNamespace
            return SimpleNamespace(loss=self.answer_only_ce(features, questions, answers, prompts=prompts))
        core, embeddings, attention, labels, _ = self._embed(
            features, questions, answers, prompts=prompts)
        return core(inputs_embeds=embeddings, attention_mask=attention,
                    labels=labels, use_cache=False, return_dict=True)

    def answer_only_ce(self, features, questions, answers, prompts=None):
        """Same causal CE, projecting only supervised positions to the full vocabulary."""
        from torch.nn import functional as F
        core, embeddings, attention, labels, _ = self._embed(features, questions, answers, prompts=prompts)
        hidden = core.transformer(inputs_embeds=embeddings, attention_mask=attention,
                                  use_cache=False, return_dict=True)[0]
        targets = labels[:, 1:]
        supervised = targets.ne(IGNORE_TOKEN_ID)
        logits = core.lm_head(hidden[:, :-1][supervised])
        return F.cross_entropy(logits, targets[supervised])

    @torch.no_grad()
    def generate(self, features, questions, max_new_tokens=96, prompts=None):
        dummy = [""] * len(questions)
        core, embeddings, attention, _, input_ids = self._embed(
            features, questions, dummy, for_generation=True, prompts=prompts)
        # transformers cannot initialize decoder-only generation from
        # inputs_embeds alone when Qwen has no bos_token_id.  Keep the real
        # prompt ids as generation bookkeeping while the first forward still
        # consumes the visual-token-replaced embeddings.
        outputs = core.generate(
            input_ids=input_ids, inputs_embeds=embeddings,
            attention_mask=attention,
            do_sample=False, top_p=1.0, top_k=50,
            max_new_tokens=max_new_tokens,
            eos_token_id=self.backbone.llm_tokenizer.im_end_id,
            pad_token_id=self.backbone.llm_tokenizer.eod_id,
        )
        # With explicit input_ids, generate returns prompt + completion.  Only
        # score/decode newly generated answer tokens.
        completions = outputs[:, input_ids.shape[1]:]
        return [self.backbone.llm_tokenizer.decode(row, skip_special_tokens=True).strip()
                for row in completions]


def build_full_visual_model(args, adapter="", projector=""):
    add_upstream(args.upstream_root)
    from lavis.models.minigpt4qwen_models.minigpt4qwen import Minigpt4Qwen
    model = Minigpt4Qwen.from_config(mpp_config(args))
    report = getattr(model, "_last_checkpoint_report", None)
    if not report or report.get("matched_key_count", 0) < 200:
        raise RuntimeError("published MPP checkpoint did not load correctly: %s" % report)
    model = model.bfloat16() if args.dtype == "bf16" else model.float()
    if adapter:
        from peft import PeftModel
        model.llm_model = PeftModel.from_pretrained(model.llm_model, adapter,
                                                    is_trainable=False)
    if projector:
        load_projector(model, projector)
    if getattr(args, "qformer_adapter", ""):
        from .visual_bridge import VisualBridge
        visual_root = Path(getattr(args, "visual_checkpoint_root", "") or args.checkpoint_root)
        bridge = VisualBridge(args.bridge_init, visual_root / "ckpt/bert-base-uncased",
                              trainable=False, weights=args.qformer_adapter)
        bridge = bridge.to(dtype=next(model.Qformer.parameters()).dtype)
        model.Qformer = bridge.Qformer
        model.query_tokens = bridge.query_tokens
    return model


def _load_mpp_projection(model, checkpoint_root):
    """Load only the MPP projection tensors needed by ``encode_image``.

    The published MPP checkpoint contains 259 Qwen tensors and two projection
    tensors.  Visual caching never executes Qwen, so reading all 15 GB of
    language weights before deleting the language model is unnecessary.
    """
    from safetensors import safe_open

    checkpoint_dir = Path(checkpoint_root).resolve() / "mpp-qwen-next-sft/sharded"
    index_path = checkpoint_dir / "model.safetensors.index.json"
    if not index_path.is_file():
        raise RuntimeError(
            "visual-only loading requires the sharded MPP checkpoint: %s" % index_path)
    with index_path.open("r", encoding="utf-8") as handle:
        weight_map = json.load(handle).get("weight_map", {})
    expected = ("llm_proj.weight", "llm_proj.bias")
    loaded = []
    with torch.no_grad():
        for key in expected:
            shard_name = weight_map.get(key)
            if not shard_name:
                raise RuntimeError("MPP checkpoint is missing %s" % key)
            shard_path = checkpoint_dir / shard_name
            with safe_open(str(shard_path), framework="pt", device="cpu") as shard:
                source = shard.get_tensor(key)
            target = model.state_dict()[key]
            if source.shape != target.shape:
                raise RuntimeError(
                    "MPP projection shape mismatch for %s: %s vs %s" %
                    (key, tuple(source.shape), tuple(target.shape)))
            target.copy_(source.to(dtype=target.dtype))
            loaded.append(key)
    model._last_checkpoint_report = {
        "path": str(index_path),
        "mode": "visual_cache_projection_only",
        "matched_key_count": len(loaded),
        "matched_keys": loaded,
        "skipped_language_tensor_count": sum(
            key.startswith("llm_model.") for key in weight_map),
    }


def build_visual_cache_model(args):
    """Build the exact MPP visual path without constructing Qwen-7B."""
    add_upstream(args.upstream_root)
    from lavis.models.minigpt4qwen_models.blip2 import Blip2Base, disabled_train

    if getattr(args, "feature_stage", "") == FEATURE_STAGE_EVA:
        class EVAOnly(Blip2Base):
            def __init__(self):
                super().__init__()
                self.visual_encoder, self.ln_vision = self.init_vision_encoder(
                    "eva_clip_g", 224, 0, False, "fp16")
                initial = torch.load(args.bridge_init, map_location="cpu")
                self.ln_vision.load_state_dict({key[len("ln_vision."):]: value
                    for key, value in initial["state_dict"].items() if key.startswith("ln_vision.")})
                for parameter in self.parameters():
                    parameter.requires_grad = False

            def encode_image(self, image, feature_stage=FEATURE_STAGE_EVA):
                return self.ln_vision(self.visual_encoder(image))

        model = EVAOnly()
        return (model.bfloat16() if args.dtype == "bf16" else model.float()).eval()

    class VisualCacheBackbone(Blip2Base):
        def __init__(self):
            super().__init__()
            self.visual_encoder, self.ln_vision = self.init_vision_encoder(
                "eva_clip_g", 224, 0, False, "fp16")
            self.Qformer, self.query_tokens = self.init_Qformer(
                32, self.visual_encoder.num_features)
            self.Qformer.bert.embeddings.word_embeddings = None
            self.Qformer.bert.embeddings.position_embeddings = None
            for layer in self.Qformer.bert.encoder.layer:
                layer.output = None
                layer.intermediate = None
            self.Qformer.cls = None
            self.llm_proj = nn.Linear(self.Qformer.config.hidden_size, 4096)
            self.qformer_text_input = False
            self.enable_autocast = False

            for module in (self.visual_encoder, self.ln_vision,
                           self.Qformer, self.llm_proj):
                for parameter in module.parameters():
                    parameter.requires_grad = False
                module.eval()
                module.train = disabled_train
            self.query_tokens.requires_grad = False

        def encode_image(self, image, feature_stage=FEATURE_STAGE_PROJECTED):
            autocast_context = (self.maybe_autocast()
                                if self.enable_autocast
                                else contextlib.nullcontext())
            with autocast_context:
                image_embeds = self.ln_vision(self.visual_encoder(image))
            image_atts = torch.ones(
                image_embeds.size()[:-1], dtype=torch.long,
                device=image_embeds.device)
            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
            qformer_features = query_output.last_hidden_state[:, :query_tokens.size(1), :]
            if feature_stage == FEATURE_STAGE_QFORMER:
                return qformer_features
            if feature_stage == FEATURE_STAGE_PROJECTED:
                return self.llm_proj(qformer_features)
            raise ValueError("unknown feature stage: %s" % feature_stage)

    config = mpp_config(args)
    model = VisualCacheBackbone()
    print("Loading visual BLIP2 checkpoint: %s" % config["pretrained"], flush=True)
    model.load_from_pretrained(config["pretrained"])
    _load_mpp_projection(model, args.checkpoint_root)
    report = model._last_checkpoint_report
    print(
        "Loaded MPP visual projection (%d tensors); skipped %d Qwen tensors."
        % (report["matched_key_count"], report["skipped_language_tensor_count"]),
        flush=True,
    )
    model = model.bfloat16() if args.dtype == "bf16" else model.float()
    return model.eval()
