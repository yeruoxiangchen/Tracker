#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Any

os.environ.setdefault("SPCONV_ALGO", "native")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
os.environ.setdefault("TORCH_EXTENSIONS_DIR", "/tmp/torch_extensions")

import numpy as np
import torch
import torch.nn.functional as F

TRACKER_ROOT = Path(__file__).resolve().parents[1]
RECONVIAGEN_ROOT = TRACKER_ROOT / "ReconViaGen"
VGGT_ROOT = RECONVIAGEN_ROOT / "wheels" / "vggt"
for path in (TRACKER_ROOT, RECONVIAGEN_ROOT, VGGT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ar_ss_flow.local_pose_lifting_flow import (  # noqa: E402
    PoseLiftingCacheDataset,
    volume_from_sample,
)
from ar_ss_flow.object_gate_c2 import (  # noqa: E402
    SelfReferenceObjectGateTable,
    apply_object_gate_exact,
)
from ar_ss_flow.train_local_pose_lifting_ss_flow import build_model  # noqa: E402


C2_CHECKPOINT_VERSION = "ar_ss_flow.object_gated_pose_lifting_c2.v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a frozen-stock local pose-lifting residual at a fixed train-set "
            "matched object-gate scale. Object-specific gates are reserved for fresh C2 eval."
        )
    )
    parser.add_argument("--cache_manifest", required=True)
    parser.add_argument("--gate_report", required=True)
    parser.add_argument("--gate_samples", required=True)
    parser.add_argument("--gate_calibration", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--pretrained", default="Stable-X/trellis-vggt-v0-2")
    parser.add_argument("--indices", default="0-15")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_steps", type=int, default=50)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--log_every", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--adapter_hidden_dim", type=int, default=96)
    parser.add_argument("--residual_scale", type=float, default=1.0)
    parser.add_argument(
        "--train_gate_policy",
        choices=("matched_constant", "correct_per_object", "ungated"),
        default="matched_constant",
    )
    parser.add_argument("--amp_dtype", choices=("fp16", "bf16", "none"), default="bf16")
    parser.add_argument("--amp_init_scale", type=float, default=8192.0)
    parser.add_argument("--flow_weight", type=float, default=1.0)
    parser.add_argument("--gain_weight", type=float, default=0.10)
    parser.add_argument("--gain_margin", type=float, default=0.002)
    parser.add_argument("--delta_norm_weight", type=float, default=0.01)
    return parser.parse_args()


def trainable_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def parameters_finite(parameters: list[torch.nn.Parameter]) -> bool:
    return all(bool(torch.isfinite(parameter.detach()).all().item()) for parameter in parameters)


def gradients_finite(parameters: list[torch.nn.Parameter]) -> bool:
    return all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad.detach()).all().item())
        for parameter in parameters
    )


def save_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    step: int,
    args: argparse.Namespace,
    model_summary: dict[str, Any],
    gate_summary: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": C2_CHECKPOINT_VERSION,
            "step": int(step),
            "model_trainable_state": trainable_state(model),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "args": vars(args),
            "model_summary": model_summary,
            "gate_summary": gate_summary,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    if min(int(args.max_steps), int(args.save_every), int(args.log_every)) <= 0:
        raise ValueError("step arguments must be positive")
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("C2 training requires CUDA")
    torch.cuda.set_device(0 if device.index is None else int(device.index))
    random.seed(int(args.seed))
    np.random.seed(int(args.seed) % (2**32 - 1))
    torch.manual_seed(int(args.seed))
    torch.cuda.manual_seed_all(int(args.seed))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    dataset = PoseLiftingCacheDataset(args.cache_manifest, indices=args.indices)
    gate_table = SelfReferenceObjectGateTable.load(
        report_path=args.gate_report,
        samples_path=args.gate_samples,
        calibration_path=args.gate_calibration,
    )
    dataset_uids = [
    str(row.get("object_uid", row["uid"]))
    for row in dataset.rows
    ]

    eligible_gate_uids = [
        uid for uid in dataset_uids
        if uid in gate_table
    ]

    missing_gate_uids = sorted(
        {
            uid for uid in dataset_uids
            if uid not in gate_table
        }
    )

    if not eligible_gate_uids:
        raise RuntimeError(
            "none of the train cache objects have complete gate records"
        )

    # Per-object gate training genuinely requires every object to have a gate.
    if (
        args.train_gate_policy == "correct_per_object"
        and missing_gate_uids
    ):
        raise KeyError(
            "correct_per_object training requires complete gates for every "
            f"training object; missing={missing_gate_uids[:5]}"
        )

    # matched_constant only needs a train-set mean estimated from objects
    # with complete self-reference records. Objects lacking a complete
    # self-reference audit can still train the correct residual using this
    # fixed constant.
    matched_constant = gate_table.mean_gate(
        "correct",
        eligible_gate_uids,
    )

    if missing_gate_uids:
        print(
            "[C2 gate coverage] "
            f"dataset_objects={len(dataset_uids)} "
            f"gate_objects={len(eligible_gate_uids)} "
            f"missing_gate_objects={len(missing_gate_uids)} "
            f"policy={args.train_gate_policy} "
            f"matched_constant={matched_constant:.6f}",
            flush=True,
        )
        print(
            "[C2 gate coverage] missing object_uids="
            + ",".join(missing_gate_uids),
            flush=True,
        )

    flow_sampler, model, model_summary = build_model(
        args, device, dataset.visual_feature_dim
    )
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        betas=(0.9, 0.95),
        eps=1.0e-8,
    )
    scaler = torch.cuda.amp.GradScaler(
        enabled=args.amp_dtype == "fp16",
        init_scale=float(args.amp_init_scale),
    )
    gate_summary = {
    **gate_table.summary(),
    "train_gate_policy": str(args.train_gate_policy),
    "train_matched_constant_gate": float(matched_constant),
    "dataset_object_count": len(dataset_uids),
    "gate_eligible_object_count": len(eligible_gate_uids),
    "missing_gate_object_count": len(missing_gate_uids),
    "missing_gate_object_uids": missing_gate_uids,
    "matched_constant_estimation_policy": (
        "mean_correct_gate_over_complete_train_selfref_records"
    ),
    "residual_scale": float(args.residual_scale),
}
    model_summary = dict(model_summary)
    model_summary["stage"] = "C2 fixed-scale local residual training for object-gate audit"
    model_summary["gate"] = gate_summary
    print(json.dumps(model_summary, indent=2), flush=True)

    use_amp = args.amp_dtype != "none"
    amp_dtype = torch.float16 if args.amp_dtype == "fp16" else torch.bfloat16
    optimizer.zero_grad(set_to_none=True)
    history: list[dict[str, Any]] = []
    start_time = time.time()
    for step in range(1, int(args.max_steps) + 1):
        sample = dataset[(step - 1) % len(dataset)]
        uid = str(sample.get("object_uid", sample["uid"]))
        with torch.no_grad():
            volume, metadata, volume_stats = volume_from_sample(
                sample, device=device, mode="correct"
            )
            target = sample["target"].unsqueeze(0).to(device=device, dtype=torch.float32)
            condition = sample["stock_condition"].to(device=device)
            noise = torch.randn_like(target)
            t_value = float(torch.rand((), device=device).item() * 0.98 + 0.01)
            x_t, gt_velocity = flow_sampler._get_model_gt(target, t_value, noise)
            t_tensor = torch.full(
                (1,), 1000.0 * t_value, device=device, dtype=torch.float32
            )
            stock = model.stock_prediction(x_t, t_tensor, condition)
        if args.train_gate_policy == "matched_constant":
            train_gate = float(matched_constant)
        elif args.train_gate_policy == "correct_per_object":
            train_gate = gate_table.gate(uid, "correct")
        else:
            train_gate = 1.0

        with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
            raw_delta, adapter_stats = model.adapter(
                x_t,
                stock,
                t_tensor,
                volume,
                metadata,
                scale=float(args.residual_scale),
                physical_present=True,
            )
            prediction, applied_delta = apply_object_gate_exact(
                stock, raw_delta, train_gate
            )
            flow_loss = F.mse_loss(prediction.float(), gt_velocity.float())
            stock_loss = F.mse_loss(stock.float(), gt_velocity.float()).detach()
            stock_energy = stock.float().square().mean().detach().clamp_min(1.0e-6)
            relative_gain = (stock_loss - flow_loss) / stock_loss.clamp_min(1.0e-6)
            gain_loss = F.relu(
                prediction.new_tensor(float(args.gain_margin)) - relative_gain
            )
            delta_norm_loss = applied_delta.float().square().mean() / stock_energy
            loss = (
                float(args.flow_weight) * flow_loss
                + float(args.gain_weight) * gain_loss
                + float(args.delta_norm_weight) * delta_norm_loss
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        finite = (
            bool(torch.isfinite(loss.detach()).item())
            and gradients_finite(trainable)
        )
        if not finite:
            raise RuntimeError(f"non-finite C2 update at step={step}")
        clip = torch.nn.utils.clip_grad_norm_(trainable, float(args.grad_clip))
        if scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if not parameters_finite(trainable):
            raise RuntimeError(f"C2 optimizer produced non-finite parameters at step={step}")

        correct_detector_gate = (
            gate_table.gate(uid, "correct")
            if uid in gate_table
            else None
        )

        row = {
            "step": step,
            "uid": str(sample["uid"]),
            "object_uid": uid,
            "train_gate": float(train_gate),
            "correct_detector_gate": correct_detector_gate,
            "correct_detector_gate_available": bool(uid in gate_table),
            "loss": float(loss.detach().float().item()),
            "flow_loss": float(flow_loss.detach().float().item()),
            "stock_flow_loss": float(stock_loss.detach().float().item()),
            "relative_gain_vs_stock": float(relative_gain.detach().float().item()),
            "gain_loss": float(gain_loss.detach().float().item()),
            "delta_norm_loss": float(delta_norm_loss.detach().float().item()),
            "raw_delta_rms": float(adapter_stats["delta_rms"].detach().float().item()),
            "applied_delta_rms": float(
                applied_delta.detach().float().square().mean().sqrt().item()
            ),
            "supported_voxel_ratio": float(volume_stats["supported_voxel_ratio"]),
            "grad_clip_total": float(clip.detach().float().item()),
            "t": t_value,
            "elapsed_seconds": float(time.time() - start_time),
        }
        if step == 1 or step % int(args.log_every) == 0:
            history.append(row)
            print(f"[object_gate_c2_train] {json.dumps(row)}", flush=True)
        if step % int(args.save_every) == 0 or step == int(args.max_steps):
            save_checkpoint(
                output_dir / "checkpoints" / f"step_{step:06d}.pt",
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                step=step,
                args=args,
                model_summary=model_summary,
                gate_summary=gate_summary,
            )
            save_checkpoint(
                output_dir / "checkpoints" / "last.pt",
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                step=step,
                args=args,
                model_summary=model_summary,
                gate_summary=gate_summary,
            )

    report = {
        "stage": model_summary["stage"],
        "format": C2_CHECKPOINT_VERSION,
        "args": vars(args),
        "dataset_size": len(dataset),
        "gate_summary": gate_summary,
        "completed_step": int(args.max_steps),
        "history": history,
    }
    (output_dir / "train_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
