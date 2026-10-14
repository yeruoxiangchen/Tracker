"""Training-only reference feedback, rollout gating and token diagnostics.

Judge decisions and log-prob changes are diagnostic signals, not verified visual
fact labels. No automatic fact-token mask is inferred from them.
"""
import json
import math

import torch
from torch.nn import functional as F

from .core import build_user_prompt, TASK_INSTRUCTION

FEEDBACK_PROTOCOL = "full5_reference_draft_feedback_opd_v1"
TEACHER_MODES = ("reference_draft", "reference_only")


def feedback_protocol(teacher_mode):
    if teacher_mode not in TEACHER_MODES:
        raise ValueError("unknown feedback teacher mode: %s" % teacher_mode)
    return (FEEDBACK_PROTOCOL if teacher_mode == "reference_draft" else
            "full5_reference_only_feedback_opd_v2")


def feedback_prompt(row, student_answer, instruction=TASK_INSTRUCTION, teacher_mode="reference_draft"):
    feedback_protocol(teacher_mode)
    if row.get("feedback_training_source") != "train":
        raise ValueError("feedback references may only come from the training split")
    if len(row["frames"]) != 5 or not row["answers"]:
        raise ValueError("expected five chronological full frames and original references")
    references = [str(answer).strip() for answer in row["answers"]]
    if any(not answer for answer in references):
        raise ValueError("empty training reference")
    # JSON quoting keeps reference/draft text separate from the prompt structure.
    if teacher_mode == "reference_only":
        context = json.dumps({"reference_answers": references}, ensure_ascii=False)
        return (build_user_prompt(row["question"], 5, instruction) +
                "\nTraining-only feedback:\n" + context +
                "\nCheck actions, directions, objects and explanations against the five frames"
                " and the reference answers. Answer the original question as a complete,"
                " concise answer consistent with the references and visible scene."
                " Do not mention these training instructions or the reference answers.")
    context = json.dumps({"reference_answers": references,
                          "previous_student_answer": student_answer}, ensure_ascii=False)
    return (build_user_prompt(row["question"], 5, instruction) +
            "\nTraining-only feedback:\n" + context +
            "\nThe previous student answer was judged inconsistent with the references."
            " Check its statements against the five frames and the reference answers."
            " Correct unsupported actions, directions, objects or explanations."
            " Answer the original question again as a complete, concise answer."
            " Do not repeat the draft merely because it is provided."
            " Do not mention these training instructions or the reference answers.")


def decode_completion(model, ids):
    return model.backbone.llm_tokenizer.decode(ids.tolist(), skip_special_tokens=True).strip()


def gate_reason(student_finished, student_correct, teacher_finished=False, teacher_correct=False):
    if not student_finished:
        return "student_truncated"
    if student_correct:
        return "student_correct_ce_only"
    if not teacher_finished:
        return "teacher_truncated"
    if not teacher_correct:
        return "teacher_wrong_ce_only"
    return "eligible"


def gate_from_scores(student_finished, student_logit, teacher_finished, teacher_logit):
    if not math.isfinite(student_logit) or not math.isfinite(teacher_logit):
        return "nonfinite_judge_ce_only"
    return gate_reason(student_finished, student_logit > 0, teacher_finished, teacher_logit > 0)


@torch.no_grad()
def token_diagnostics(student_logits, plain_teacher_logits, feedback_logits, ids, tokenizer, topk=3):
    if not (student_logits.shape == plain_teacher_logits.shape == feedback_logits.shape):
        raise ValueError("all three scorers must predict the same original student tokens")
    if student_logits.ndim != 2 or len(ids) != len(student_logits):
        raise ValueError("token positions do not align")
    # Compute per position to avoid three full FP32 vocabulary distributions in VRAM.
    records = []
    for index, token_id in enumerate(ids.tolist()):
        distributions = [F.log_softmax(logits[index].float(), dim=-1)
                         for logits in (student_logits, plain_teacher_logits, feedback_logits)]
        s, plain, feedback = distributions
        mixture = torch.logaddexp(s, feedback) - math.log(2)
        jsd = .5 * ((s.exp() * (s - mixture)).sum() +
                    (feedback.exp() * (feedback - mixture)).sum())
        plain_mixture = torch.logaddexp(s, plain) - math.log(2)
        plain_jsd = .5 * ((s.exp() * (s - plain_mixture)).sum() +
                         (plain.exp() * (plain - plain_mixture)).sum())
        def candidates(log_probs):
            values, positions = log_probs.topk(min(topk, len(log_probs)))
            return [{"token_id": int(i), "token": tokenizer.decode([int(i)]), "log_prob": float(v)}
                    for v, i in zip(values.cpu(), positions.cpu())]
        records.append({"position": index, "token_id": token_id,
                        "token": tokenizer.decode([token_id]),
                        "student_log_prob": float(s[token_id]),
                        "plain_teacher_log_prob": float(plain[token_id]),
                        "feedback_teacher_log_prob": float(feedback[token_id]),
                        "feedback_minus_plain": float(feedback[token_id] - plain[token_id]),
                        "jsd": float(jsd), "plain_teacher_jsd": float(plain_jsd), "student_top": candidates(s),
                        "feedback_teacher_top": candidates(feedback)})
    return records


MATCHED_KEYS = ("data_sha256", "initialization", "base_checkpoint_sha256", "bridge_init_sha256",
                "cache_dir", "precision_protocol", "qformer_mode", "dtype", "task_instruction_version",
                "lr", "projector_lr", "qformer_lr", "ce_weight", "weight_decay", "warmup_ratio",
                "epochs", "world_size", "grad_accumulation", "total_steps", "seed",
                "gradient_checkpointing", "qformer_gradient_checkpointing", "dropout",
                "max_txt_len", "max_new_tokens", "feature_stage", "probe_report_sha256")


def validate_matched_configs(control, opd):
    if control.get("training_mode") != "sft_control" or opd.get("training_mode") != "feedback_opd":
        raise ValueError("expected SFT control and feedback OPD configurations")
    changed = [key for key in MATCHED_KEYS if key not in control or key not in opd or control[key] != opd[key]]
    # Older completed v1 runs do not contain these optional v2 fields.
    changed.extend(key for key in ("teacher_mode", "checkpoint_steps", "objective", "chunk_tokens")
                   if (key in control or key in opd) and control.get(key) != opd.get(key))
    if changed:
        raise ValueError("SFT and OPD budgets/initialization differ: %s" % changed)
