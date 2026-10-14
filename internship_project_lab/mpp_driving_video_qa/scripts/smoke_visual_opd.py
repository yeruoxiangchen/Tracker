#!/usr/bin/env python3
"""Bounded real-weight smoke: EVA/Q-Former parity, SFT update, sampled-token OPD, teacher freeze."""
import argparse
import gc
import json
from pathlib import Path
import sys
import time
import types

import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mpp_video_qa.core import load_jsonl
from mpp_video_qa.data import cache_file
from mpp_video_qa.model import build_visual_cache_model, build_text_model, CachedVideoQA
from mpp_video_qa.visual_bridge import add_bridge_arguments
from mpp_video_qa.observations import open_views, view_prompt
from mpp_video_qa.opd import (SharedFrozenTeacher, sample_completions, completion_logits,
                              distillation_loss, disable_dropout)
from scripts.generate_answers import write_json_atomic


def main():
    p = argparse.ArgumentParser()
    add_bridge_arguments(p)
    for name in ("data", "upstream-root", "checkpoint-root", "adapter", "projector", "output-dir"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--old-cache-dir", default="/tmp/zjr_mpp_fast/cache/lingoqa_5f_qformer_v1")
    p.set_defaults(feature_stage="eva", train_qformer=True, train_projector=True,
                   max_txt_len=1024, dtype="bf16", lora_r=16, lora_alpha=32,
                   lora_dropout=0.0, lora_target_modules=["c_attn"], gradient_checkpointing=True)
    args = p.parse_args()
    torch.manual_seed(42)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    row = load_jsonl(args.data)[0]
    started = time.time()
    encoder = build_visual_cache_model(args).to("cuda").eval()
    from lavis.processors.blip_processors import BlipImageEvalProcessor
    processor = BlipImageEvalProcessor(image_size=224)
    views = {}
    for name in ("full", "full_crop"):
        images = open_views(row["frames"], name)
        try:
            pixels = torch.stack([processor(image) for image in images]).to("cuda", dtype=torch.bfloat16)
            with torch.no_grad():
                views[name] = encoder.encode_image(pixels).detach().clone().unsqueeze(0)
        finally:
            for image in images:
                image.close()
    del encoder, pixels
    gc.collect()
    torch.cuda.empty_cache()
    print("Smoke: EVA observations ready", flush=True)
    model = CachedVideoQA(build_text_model(args, train=True, adapter=args.adapter, projector=args.projector),
                          feature_stage="eva").to("cuda").eval()
    disable_dropout(model)
    old = torch.load(cache_file(args.old_cache_dir, row["segment_id"]), map_location="cpu")
    with torch.no_grad():
        current = model.backbone.visual_bridge(views["full"]).squeeze(0).cpu()
    error = float((current.float() - old["features"].float()).abs().max())
    # BF16 reduction order can differ; record both absolute and relative numerical error.
    torch.testing.assert_close(current.float(), old["features"].float(), atol=0.04, rtol=0.03)
    named = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    snapshots = {}
    for label, selector in (("qwen", "llm_model."), ("projector", "llm_proj."),
                            ("queries", "visual_bridge.query_tokens"),
                            ("qformer", "visual_bridge.Qformer")):
        name, param = next((n, v) for n, v in named if selector in n and
                           (label not in ("qwen", "qformer") or "lora_B" in n))
        snapshots[label] = (name, param, param.detach().float().cpu().clone())
    optimizer = torch.optim.AdamW([p for _, p in named], lr=5e-6)
    model.train()
    losses = []
    # Actual gradient accumulation across two micros, one update.
    for _ in range(2):
        loss = model(views["full"], [row["question"]], [row["answers"][0]]).loss
        losses.append(float(loss.detach()))
        (loss / 2).backward()
    grads = {name: float(p.grad.float().norm()) if p.grad is not None else None
             for name, (_, p, _) in snapshots.items()}
    if not all(value is not None and value > 0 for value in grads.values()):
        raise RuntimeError("missing trainable-branch gradients: %s" % grads)
    torch.nn.utils.clip_grad_norm_([p for _, p in named], 1.0)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    updates = {name: float((p.detach().float().cpu() - before).abs().max())
               for name, (_, p, before) in snapshots.items()}
    if not all(delta > 0 for delta in updates.values()):
        raise RuntimeError("a branch did not update: %s" % updates)
    print(json.dumps({"sft_losses": losses, "gradients": grads, "updates": updates}), flush=True)
    args.teacher_adapter = args.adapter
    args.teacher_projector = args.projector
    args.teacher_qformer_adapter = ""
    teacher = SharedFrozenTeacher(model, args)
    frozen = {name: p.detach().cpu().clone() for name, p in teacher.model.named_parameters()
              if "opd_teacher" in name or name.startswith(("backbone.llm_proj.", "backbone.visual_bridge."))}
    completion, finished = sample_completions(model, views["full"], row["question"], 96)
    if not finished:
        raise RuntimeError("smoke rollout truncated; no fabricated EOS or reference substitute")
    with teacher.active() as fixed:
        teacher_logits = completion_logits(fixed, views["full_crop"], row["question"], completion,
                                            view_prompt(row["question"], 5, "full_crop")).detach().clone()
    student_logits = completion_logits(model, views["full"], row["question"], completion)
    loss = distillation_loss(student_logits, teacher_logits, chunk_tokens=8)
    jsd = float(loss.detach())
    loss.backward()
    torch.nn.utils.clip_grad_norm_([p for _, p in named], 1.0)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    if not torch.isfinite(loss):
        raise RuntimeError("nonfinite distillation")
    for name, param in teacher.model.named_parameters():
        if name in frozen and not torch.equal(param.detach().cpu(), frozen[name]):
            raise RuntimeError("frozen teacher parameter changed: %s" % name)
    # Round-trip the actual trained Q-Former adapter.
    model.backbone.visual_bridge.save(output / "qformer_adapter.pt")
    result = {"completed": True, "kind": "single_gpu_real_model_smoke_not_quality_evaluation",
              "question_id": row["question_id"], "eva_shape": list(views["full"].shape),
              "qformer_cache_max_abs_error": error, "sft_losses": losses,
              "branch_gradient_norms": grads, "branch_updates": updates,
              "rollout_tokens": completion.numel(), "rollout_finished": finished,
              "jsd": jsd, "teacher_unchanged": True,
              "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
              "elapsed_seconds": time.time() - started}
    write_json_atomic(output / "summary.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

