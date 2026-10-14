#!/usr/bin/env python3
"""One real-model mechanical update; never a teacher qualification experiment."""
import argparse
import json
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mpp_video_qa.feedback_opd import feedback_prompt, decode_completion
from mpp_video_qa.opd import SharedFrozenTeacher, sample_completions, completion_logits, distillation_loss
from scripts.feedback_opd_common import add_runtime_arguments, configure_model, runtime_identity, training_dataset, build_student
from scripts.train_feedback_opd import tracked_parameters, teacher_checks
from scripts.train_sft import seed_all, gpu_memory
from scripts.generate_answers import write_json_atomic


def main():
    parser = argparse.ArgumentParser()
    add_runtime_arguments(parser)
    args = configure_model(parser.parse_args(), train=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; no CPU model fallback")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "summary.json").exists():
        raise ValueError("smoke already completed; use a new directory")
    identity = runtime_identity(args)
    write_json_atomic(output / "run_config.json", identity)
    torch.set_num_threads(args.cpu_threads)
    seed_all(args.seed)
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    started = time.time()
    print(json.dumps({"status": "loading_model", "validation_only": True}), flush=True)
    student = build_student(args, device, train=True)
    teacher = SharedFrozenTeacher(student, args)
    teacher.validate_initial_copy()
    record = training_dataset(args)[0]
    row, features = record["row"], record["features"].unsqueeze(0).to(device)
    tracked = tracked_parameters(student)
    initial = {key: p.detach().float().cpu().clone() for key, (_, p) in tracked.items()}
    frozen_checks = teacher_checks(teacher)
    with torch.no_grad():
        ids, finished = sample_completions(student, features, row["question"], min(24, args.max_new_tokens))
        draft = decode_completion(student, ids)
        student.eval()
        _, student_prefix, student_mask, _, student_ids = student._embed(features, [row["question"]], [""], for_generation=True)
        plain_student = completion_logits(student, features, row["question"], ids)
        repeated_student = completion_logits(student, features, row["question"], ids)
        with teacher.active() as frozen:
            _, teacher_prefix, teacher_mask, _, teacher_ids = frozen._embed(features, [row["question"]], [""], for_generation=True)
            plain_teacher = completion_logits(frozen, features, row["question"], ids)
            targets = completion_logits(frozen, features, row["question"], ids,
                                        feedback_prompt(row, draft, student.task_instruction)).detach()
        difference = float((plain_student.float() - plain_teacher.float()).abs().max())
        visual_difference = float((student._project_visual_features(features, torch.bfloat16).float() -
                                  teacher.model._project_visual_features(features, torch.bfloat16).float()).abs().max())
        language_weights = dict(student.backbone.llm_model.named_parameters())
        adapter_difference = max(float((p - language_weights[name.replace('.default.', '.opd_teacher.')]).abs().max())
                                 for name, p in language_weights.items() if '.default.' in name)
        bridge_difference = max(float((p - teacher.model.backbone.visual_bridge.state_dict()[name]).abs().max())
                                for name, p in student.backbone.visual_bridge.state_dict().items())
        projector_difference = max(float((p - teacher.model.backbone.llm_proj.state_dict()[name]).abs().max())
                                   for name, p in student.backbone.llm_proj.state_dict().items())
        alignment = {"logits_max_abs_difference": difference, "visual_max_abs_difference": visual_difference,
                     "adapter_max_abs_difference": adapter_difference, "bridge_max_abs_difference": bridge_difference,
                     "projector_max_abs_difference": projector_difference,
                     "prefix_max_abs_difference": float((student_prefix-teacher_prefix).abs().max()),
                     "input_ids_equal": torch.equal(student_ids, teacher_ids),
                     "masks_equal": torch.equal(student_mask, teacher_mask),
                     "repeat_student_max_abs_difference": float((plain_student-repeated_student).abs().max()),
                     "plain_teacher_jsd": float(distillation_loss(plain_student, plain_teacher)),
                     "repeat_student_jsd": float(distillation_loss(plain_student, repeated_student))}
        write_json_atomic(output / "alignment.json", alignment)
        if (any(alignment[key] != 0 for key in ("visual_max_abs_difference", "adapter_max_abs_difference",
                "bridge_max_abs_difference", "projector_max_abs_difference", "prefix_max_abs_difference")) or
                not alignment["input_ids_equal"] or not alignment["masks_equal"]):
            raise RuntimeError("teacher initialization or input differs without feedback")
        del plain_student, plain_teacher, repeated_student, student_prefix, teacher_prefix
    student.train()
    parameters = [p for p in student.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=2.5e-6)
    # Deliberately exercise JSD even if this short sample is truncated. This is
    # mechanical validation, NOT a gated training example or correctness claim.
    logits = completion_logits(student, features, row["question"], ids)
    divergence = distillation_loss(logits, targets, "jsd", 8)
    jsd = float(divergence.detach())
    divergence.backward()
    markers = {"language": ".lora_B.default.", "projector": "llm_proj.", "qformer": "visual_bridge."}
    gradients = {group: sum(float(p.grad.detach().float().abs().sum())
                            for name, p in student.named_parameters() if marker in name and p.grad is not None)
                 for group, marker in markers.items()}
    del logits, divergence, targets
    ce = student(features, [row["question"]], [record["answer"]]).loss
    ce_value = float(ce.detach())
    ce.backward()
    del ce
    norm = torch.nn.utils.clip_grad_norm_(parameters, 1.)
    if not torch.isfinite(norm) or not all(value > 0 for value in gradients.values()):
        raise RuntimeError("nonfinite or missing student branch gradients")
    optimizer.step()
    deltas = {group: float((p.detach().float().cpu() - initial[group]).abs().max())
              for group, (_, p) in tracked.items()}
    teacher_deltas = {group: float((p.detach().float().cpu() - before).abs().max())
                      for group, (_, p, before) in frozen_checks.items()}
    if not all(value > 0 for value in deltas.values()) or any(teacher_deltas.values()):
        raise RuntimeError("student failed to update or teacher changed")
    summary = {"completed": True, "validation_only": True, "optimizer_steps": 1,
               "question_id": row["question_id"], "student_draft": draft,
               "student_finished": finished, "sampled_tokens": len(ids),
               "plain_teacher_vs_student_max_abs_logit_difference": difference,
               "initial_alignment": alignment,
               "jsd": jsd, "ce": ce_value, "jsd_branch_gradient_l1": gradients,
               "tracked_parameter_deltas": deltas, "teacher_tracked_parameter_deltas": teacher_deltas,
               "gpu_memory": gpu_memory(device), "elapsed_seconds": time.time() - started,
               "quality_interpretation": "Forced mechanical JSD update, no Judge gate. No teacher or student accuracy measured. No training weights saved."}
    write_json_atomic(output / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
