"""Shared model/data identity and runtime for feedback OPD and its probe."""
import json
import hashlib
import math
import os
from pathlib import Path

import torch

from mpp_video_qa.core import TASK_INSTRUCTIONS, file_sha256
from mpp_video_qa.data import CachedQADataset
from mpp_video_qa.feedback_opd import (TEACHER_MODES, feedback_protocol, feedback_prompt,
                                     decode_completion, gate_reason)
from mpp_video_qa.model import CachedVideoQA, build_text_model
from mpp_video_qa.opd import SharedFrozenTeacher, disable_dropout, sample_completions
from scripts.evaluate_answers import load_judge, judge_score
from scripts.generate_answers import directory_fingerprint
from mpp_video_qa.opd_runtime import phase
from mpp_video_qa.lora_expansion import LANGUAGE_LORA_SCOPES, scope_identity

ROOT = Path(__file__).resolve().parents[1]


def add_runtime_arguments(parser):
    parser.add_argument("--data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--initial-dir", default=str(ROOT / "outputs/mpp_lingoqa_qformer_all_1epoch_v1"))
    parser.add_argument("--cache-dir", default="/tmp/zjr_mpp_fast/cache/lingoqa_5f_eva_v1")
    parser.add_argument("--upstream-root", default=str(ROOT.parent / "mpp_llava_video_workspace/third_party/MPP-LLaVA"))
    parser.add_argument("--checkpoint-root", default="/tmp/zjr_mpp_fast/checkpoints")
    parser.add_argument("--visual-checkpoint-root", default=str(ROOT.parent / "mpp_llava_video_workspace/checkpoints"))
    parser.add_argument("--bridge-init", default=str(ROOT / "checkpoints/visual_bridge_init.pt"))
    parser.add_argument("--task-instruction-version", choices=sorted(TASK_INSTRUCTIONS), default="legacy_v1")
    parser.add_argument("--max-txt-len", type=int, default=1536)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--judge-model", default="wayveai/Lingo-Judge")
    parser.add_argument("--judge-device", default="cpu", help="CPU avoids using training GPU memory")
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--teacher-mode", choices=TEACHER_MODES, default="reference_draft")
    parser.add_argument("--language-lora-scope", choices=LANGUAGE_LORA_SCOPES, default="qkv",
                        help="decoder_linear expands the trained QKV adapter for pure CE SFT")


def configure_model(args, train):
    initial = Path(args.initial_dir)
    args.init_adapter = str(initial / "final_adapter")
    args.projector = str(initial / "final_projector.pt")
    args.qformer_adapter = str(initial / "final_qformer_adapter.pt")
    args.teacher_adapter, args.teacher_projector = args.init_adapter, args.projector
    args.teacher_qformer_adapter = args.qformer_adapter
    args.feature_stage, args.qformer_mode = "eva", "full"
    args.precision_protocol, args.dtype = "fp32_trainables_v1", "bf16"
    args.train_qformer, args.train_projector = bool(train), bool(train)
    args.qformer_r, args.qformer_alpha = 0, 0
    args.gradient_checkpointing = bool(train)
    args.qformer_gradient_checkpointing = bool(train)
    args.lora_r, args.lora_alpha = 16, 32
    scope = getattr(args, "language_lora_scope", "qkv")
    args.lora_dropout = 0.0
    args.lora_target_modules = (["c_attn", "c_proj", "w1", "w2"]
                               if scope == "decoder_linear" else ["c_attn"])
    args.expand_language_lora = bool(train and scope == "decoder_linear" and
                                    not getattr(args, "resume_from", ""))
    return args


def runtime_identity(args):
    initial = Path(args.initial_dir)
    parent = json.loads((initial / "summary.json").read_text())
    config = json.loads((initial / "run_config.json").read_text())
    if not parent.get("completed") or parent["optimizer_steps"] != config["total_steps"]:
        raise ValueError("initial full-QFormer SFT stage is incomplete")
    if config.get("qformer_mode") != "full" or config.get("precision_protocol") != "fp32_trainables_v1":
        raise ValueError("initial stage must contain complete FP32 Q-Former weights")
    if config["task_instruction_version"] != args.task_instruction_version:
        raise ValueError("student task instruction must match its SFT initialization")
    ckpt = Path(args.checkpoint_root)
    manifest = ckpt / "mpp-qwen-next-sft/sharded/model.safetensors.index.json"
    if not manifest.exists():
        manifest = ckpt / "mpp-qwen-next-sft/unfreeze_llm_model.pth"
    mode = getattr(args, "teacher_mode", "reference_draft")
    result = {"feedback_protocol": feedback_protocol(mode), "data_sha256": file_sha256(args.data),
            "initialization": {"adapter": directory_fingerprint(args.init_adapter),
                               "projector": file_sha256(args.projector),
                               "qformer": file_sha256(args.qformer_adapter)},
            "base_checkpoint_sha256": file_sha256(manifest),
            "checkpoint_root": str(ckpt.resolve()),
            "bridge_init_sha256": file_sha256(args.bridge_init),
            "cache_dir": str(Path(args.cache_dir).resolve()),
            "precision_protocol": args.precision_protocol, "qformer_mode": args.qformer_mode,
            "dtype": args.dtype, "task_instruction_version": args.task_instruction_version,
            "max_txt_len": args.max_txt_len, "max_new_tokens": args.max_new_tokens,
            "judge_model": args.judge_model, "judge_device": args.judge_device,
            "seed": args.seed, "dropout": 0.0,
            "rollout_distribution": "temperature=1,top_p=1,top_k=0,dropout=0"}
    if mode != "reference_draft":
        result["teacher_mode"] = mode
    if getattr(args, "language_lora_scope", "qkv") != "qkv":
        result["language_lora"] = scope_identity(args.language_lora_scope)
    return result


def training_dataset(args):
    dataset = CachedQADataset(args.data, args.cache_dir, "eva", validate_full_frames=True)
    ids = [row["question_id"] for row in dataset.rows]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("training/probe question IDs must be nonempty and unique")
    if any(row.get("feedback_training_source") != "train" or len(row["frames"]) != 5 for row in dataset.rows):
        raise ValueError("use prepare_feedback_opd.py for original train-only five-frame QA")
    return dataset


def build_student(args, device, train):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; no CPU fallback for the MPP model")
    model = CachedVideoQA(build_text_model(args, train=train, adapter=args.init_adapter,
                                          projector=args.projector),
                          task_instruction=TASK_INSTRUCTIONS[args.task_instruction_version],
                          feature_stage="eva").to(device)
    disable_dropout(model)
    model.train(train)
    return model


class Judge:
    def __init__(self, args):
        requested = getattr(args, "runtime_judge_device", "") or args.judge_device
        self.model_name = args.judge_model
        self.device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0))) if requested == "local" else torch.device(requested)
        self.tokenizer, self.model = load_judge(args.judge_model, self.device)
        self.model.float()
        self.cpu_model = None
        self.boundary_margin = getattr(args, "judge_boundary_margin", .05)
        self.cpu_fallbacks = 0
        if self.device.type == "cuda" and requested != args.judge_device:
            path = Path(getattr(args, "judge_equivalence_report", ""))
            if not path.is_file():
                raise ValueError("GPU Judge requires the verify_judge report")
            report = json.loads(path.read_text())
            if (not report.get("passed") or report.get("judge_model") != args.judge_model or
                    report.get("scorer_sha256") != file_sha256(ROOT / "scripts/evaluate_answers.py") or
                    report.get("gpu_name") != torch.cuda.get_device_name(self.device) or
                    report.get("boundary_margin") != self.boundary_margin):
                raise ValueError("GPU Judge validation report differs from this runtime")

    def score(self, row, answer):
        value, correct = judge_score(self.tokenizer, self.model, row["question"], row["answers"], answer)
        if self.device.type == "cuda" and abs(value) <= self.boundary_margin:
            if self.cpu_model is None:
                # Preserve the official zero boundary where numerical changes could affect the gate.
                _, self.cpu_model = load_judge(self.model_name, torch.device("cpu"))
                self.cpu_model.float()
            value, correct = judge_score(self.tokenizer, self.cpu_model, row["question"], row["answers"], answer)
            self.cpu_fallbacks += 1
        return {"logit": value, "correct": correct}


@torch.no_grad()
def current_rollout(student, teacher, judge, features, row, max_new_tokens,
                    teacher_mode="reference_draft", gate_cache=None, timer=None):
    """All scoring/generation ends before any student autograd graph is built."""
    with phase(timer, "student_generate"):
        ids, finished = sample_completions(student, features, row["question"], max_new_tokens)
    text = decode_completion(student, ids)
    with phase(timer, "student_judge", getattr(judge, "device", "cpu")):
        student_score = judge.score(row, text)
    diagnostic = {"student_answer": text, "student_finished": finished,
                  "student_judge": student_score, "student_token_ids": ids.tolist(),
                  "teacher_correction": None, "teacher_finished": False, "teacher_judge": None,
                  "eligible": False, "teacher_mode": teacher_mode,
                  "teacher_cache_hit": False, "teacher_generation_called": False}
    prompt = None
    if not math.isfinite(student_score["logit"]):
        diagnostic["gate_reason"] = "nonfinite_judge_ce_only"
        return ids, prompt, diagnostic
    if finished and not student_score["correct"]:
        if gate_cache is not None and teacher_mode != "reference_only":
            raise ValueError("teacher gate cache requires reference_only")
        prompt = feedback_prompt(row, text, student.task_instruction, teacher_mode)
        diagnostic["teacher_prompt_sha256"] = hashlib.sha256(prompt.encode()).hexdigest()
        with phase(timer, "teacher_cache_lookup", "cpu"):
            key = gate_cache.key(row, prompt, max_new_tokens) if gate_cache else None
            cached = gate_cache.get(key) if gate_cache else None
        if cached is not None:
            correction_text, teacher_finished, teacher_score = cached["text"], cached["finished"], cached["score"]
            diagnostic["teacher_cache_hit"] = True
        else:
            try:
                with phase(timer, "teacher_generate"):
                    with teacher.active() as frozen:
                        correction, teacher_finished = sample_completions(
                            frozen, features, row["question"], max_new_tokens, prompt=prompt, do_sample=False)
                        correction_text = decode_completion(frozen, correction)
                diagnostic["teacher_generation_called"] = True
            except ValueError as error:
                if "exceeds max_txt_len" not in str(error):
                    raise
                diagnostic.update(gate_reason="teacher_context_too_long", context_error=str(error))
                return ids, None, diagnostic
            with phase(timer, "teacher_judge", getattr(judge, "device", "cpu")):
                teacher_score = judge.score(row, correction_text)
            if gate_cache and math.isfinite(teacher_score["logit"]):
                with phase(timer, "teacher_cache_write", "cpu"):
                    gate_cache.put(key, {"text": correction_text, "finished": teacher_finished, "score": teacher_score})
        diagnostic.update(teacher_correction=correction_text, teacher_finished=teacher_finished,
                          teacher_judge=teacher_score)
        if not math.isfinite(teacher_score["logit"]):
            diagnostic["gate_reason"] = "nonfinite_judge_ce_only"
            return ids, prompt, diagnostic
    diagnostic["gate_reason"] = gate_reason(finished, student_score["correct"],
                                             diagnostic["teacher_finished"],
                                             bool(diagnostic["teacher_judge"] and diagnostic["teacher_judge"]["correct"]))
    diagnostic["eligible"] = diagnostic["gate_reason"] == "eligible"
    return ids, prompt, diagnostic
