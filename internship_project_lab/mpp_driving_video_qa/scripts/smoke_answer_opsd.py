#!/usr/bin/env python3
"""Bounded real-MPP check, not a teacher-selection or accuracy experiment."""
import argparse
import json
from pathlib import Path
import sys
import time

import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mpp_video_qa.data import CachedQADataset
from mpp_video_qa.model import CachedVideoQA, build_text_model
from mpp_video_qa.observations import teacher_prompt
from mpp_video_qa.opd import SharedFrozenTeacher, sample_completions, completion_logits, distillation_loss
from mpp_video_qa.visual_bridge import add_bridge_arguments
from scripts.generate_answers import write_json_atomic
from scripts.train_sft import seed_all


def main():
    p = argparse.ArgumentParser()
    add_bridge_arguments(p)
    for name in ("data", "cache-dir", "upstream-root", "checkpoint-root", "adapter", "projector", "output-dir"):
        p.add_argument("--" + name, required=True)
    p.set_defaults(feature_stage="qformer", train_projector=True, max_txt_len=1024,
                   dtype="bf16", gradient_checkpointing=True, lora_r=16, lora_alpha=32,
                   lora_dropout=0, lora_target_modules=["c_attn"])
    args = p.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "summary.json").exists():
        raise ValueError("smoke output already exists")
    seed_all(42)
    started = time.time()
    dataset = CachedQADataset(args.data, args.cache_dir, "qformer")
    if not dataset.rows or dataset.rows[0].get("opd_training_source") != "train":
        raise ValueError("use the train-only prepared probe")
    model = CachedVideoQA(build_text_model(args, train=True, adapter=args.adapter,
                                          projector=args.projector), feature_stage="qformer").to("cuda")
    args.teacher_adapter, args.teacher_projector = args.adapter, args.projector
    teacher = SharedFrozenTeacher(model, args)
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    parameters = [p for _, p in named]
    frozen = {n: p.detach().cpu().clone() for n, p in teacher.model.named_parameters()
              if "opd_teacher" in n or n.startswith("backbone.llm_proj.")}
    tracked = {}
    for label, marker in (("language_lora", "lora_B.default"), ("projector", "llm_proj.weight")):
        n, parameter = next((n, p) for n, p in named if marker in n)
        tracked[label] = (parameter, parameter.detach().cpu().clone())
    optimizer = torch.optim.AdamW(parameters, lr=2e-6)
    model.train()
    logs = []
    # Two updates, each accumulated from two real student rollouts.
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        for micro in range(2):
            item = dataset[(2 * step + micro) % len(dataset)]
            row = item["row"]
            features = item["features"].unsqueeze(0).to("cuda")
            ids, finished = sample_completions(model, features, row["question"], 96)
            if not finished:
                raise RuntimeError("smoke rollout truncated; not replacing it with a reference")
            prompt = teacher_prompt(row["question"], features.shape[1], row["answers"], "answer_hint")
            with teacher.active() as fixed:
                target = completion_logits(fixed, features, row["question"], ids, prompt).detach().clone()
            logits = completion_logits(model, features, row["question"], ids)
            loss = distillation_loss(logits, target)
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite divergence")
            jsd = float(loss.detach())
            (loss / 2).backward()
            del target, logits, loss
            replay = model(features, [row["question"]], [row["answers"][0]]).loss
            ce = float(replay.detach())
            (0.2 * replay / 2).backward()
            del replay
            logs.append({"step": step + 1, "micro": micro + 1, "question_id": row["question_id"],
                         "tokens": ids.numel(), "jsd": jsd, "ce": ce})
            print(json.dumps(logs[-1]), flush=True)
        if any(p.grad is not None for n, p in teacher.model.named_parameters() if n in frozen):
            raise RuntimeError("frozen teacher has gradients")
        torch.nn.utils.clip_grad_norm_(parameters, 1)
        optimizer.step()
    updates = {name: float((parameter.detach().cpu() - before).abs().max())
               for name, (parameter, before) in tracked.items()}
    if not all(v > 0 for v in updates.values()):
        raise RuntimeError("student branch did not update")
    for n, parameter in teacher.model.named_parameters():
        if n in frozen and not torch.equal(parameter.detach().cpu(), frozen[n]):
            raise RuntimeError("teacher changed")
    summary = {"completed": True, "kind": "real_model_engineering_smoke_not_accuracy_evaluation",
               "optimizer_steps": 2, "gradient_accumulation": 2, "rollouts": logs,
               "teacher_unchanged": True, "student_branch_updates": updates,
               "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
               "elapsed_seconds": time.time() - started}
    write_json_atomic(output / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
