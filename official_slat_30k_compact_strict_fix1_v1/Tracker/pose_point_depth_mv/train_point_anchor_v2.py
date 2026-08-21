#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import time
from typing import Any

os.environ.setdefault("SPCONV_ALGO", "native")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
os.environ.setdefault("TORCH_EXTENSIONS_DIR", "/tmp/torch_extensions")

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ar_ss_flow.local_pose_lifting_flow import collate_one
from pose_point_depth_mv.point_anchor_v2 import (
    POINT_ANCHOR_CHECKPOINT_VERSION,
    POINT_CONTROL_NAMES,
    PointAnchorCacheDataset,
    PointAnchorProbe,
    make_null_point_evidence,
)
from pose_point_depth_mv.train_local_target_probe import (
    build_frozen_stock_flow,
    masked_mse,
)
from reconvggt_ar_adapter_a.train_pointpose_ss_lora import (
    finite_tree,
    gradients_finite,
    optimizer_state_finite,
    parameters_finite,
    sample_t,
    tensors_finite,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the point-only fixed-mask local-anchor v2 probe."
    )
    parser.add_argument("--point_cache_manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--pretrained", default="Stable-X/trellis-vggt-v0-2")
    parser.add_argument("--indices", default="0-15")
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--save_every", type=int, default=25)
    parser.add_argument("--log_every", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--amp_dtype", choices=("fp16", "bf16", "none"), default="bf16")
    parser.add_argument("--amp_init_scale", type=float, default=8192.0)
    parser.add_argument("--nonfinite_policy", choices=("error", "skip"), default="error")
    parser.add_argument("--max_nonfinite_attempts", type=int, default=0)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--physical_scale", type=float, default=1.0)
    parser.add_argument(
        "--t_schedule",
        choices=("uniform", "logit_normal", "high_t_mix"),
        default="uniform",
    )
    parser.add_argument("--local_target_weight", type=float, default=1.0)
    parser.add_argument("--flow_weight", type=float, default=0.10)
    parser.add_argument("--control_zero_weight", type=float, default=0.25)
    parser.add_argument("--gain_weight", type=float, default=0.10)
    parser.add_argument("--gain_margin", type=float, default=0.00005)
    parser.add_argument("--rank_weight", type=float, default=0.25)
    parser.add_argument("--rank_margin", type=float, default=0.00001)
    parser.add_argument("--delta_norm_weight", type=float, default=0.01)
    return parser.parse_args()


def gradient_group_norms(probe: PointAnchorProbe) -> dict[str, float]:
    groups = {
        "state_projection": "state_projection.",
        "evidence_projection": "evidence_projection.",
        "fusion": "fusion.",
        "output": "output.",
    }
    named = list(probe.named_parameters())
    return {
        label: sum(
            float(parameter.grad.detach().float().square().sum().item())
            for name, parameter in named
            if name.startswith(prefix) and parameter.grad is not None
        )
        ** 0.5
        for label, prefix in groups.items()
    }


def trainable_state_dict(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu()
        for name, parameter in module.named_parameters()
        if parameter.requires_grad
    }


def save_checkpoint(
    path: Path,
    *,
    probe: PointAnchorProbe,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    step: int,
    args: argparse.Namespace,
    model_summary: dict[str, Any],
) -> None:
    trainable = [parameter for parameter in probe.parameters() if parameter.requires_grad]
    if not parameters_finite(trainable):
        raise RuntimeError(f"refusing to save non-finite point probe: {path}")
    if not optimizer_state_finite(optimizer):
        raise RuntimeError(f"refusing to save non-finite optimizer: {path}")
    if not finite_tree(scaler.state_dict()):
        raise RuntimeError(f"refusing to save non-finite scaler: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": POINT_ANCHOR_CHECKPOINT_VERSION,
            "step": int(step),
            "model_trainable_state": trainable_state_dict(probe),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "args": vars(args),
            "model_summary": model_summary,
        },
        path,
    )


@torch.no_grad()
def stock_preservation_audit(
    probe: PointAnchorProbe,
    sample: dict[str, Any],
    device: torch.device,
) -> dict[str, float]:
    target = sample["target"].unsqueeze(0).to(device=device, dtype=torch.float32)
    x_t = torch.randn_like(target)
    stock = torch.randn_like(target)
    t = torch.full((1,), 500.0, device=device)
    evidence = sample["point_correct_evidence"].unsqueeze(0).to(device=device)
    disabled, _ = probe(x_t, stock, t, evidence, physical_present=False)
    null, _ = probe(x_t, stock, t, make_null_point_evidence(evidence))
    enabled, stats = probe(x_t, stock, t, evidence)
    report = {
        "physical_off_max_abs": float(disabled.abs().max().item()),
        "null_evidence_max_abs": float(null.abs().max().item()),
        "zero_init_enabled_max_abs": float(enabled.abs().max().item()),
        "non_anchor_max_abs": float(stats["neutral_abs_max"].item()),
    }
    if any(value != 0.0 for value in report.values()):
        raise RuntimeError(f"point-anchor stock preservation failed: {report}")
    return report


def main() -> None:
    args = parse_args()
    if min(args.max_steps, args.save_every, args.log_every, args.grad_accum) <= 0:
        raise ValueError("step/log/accumulation values must be positive")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise RuntimeError("point-anchor v2 is a single-GPU mechanism probe")
    if not torch.cuda.is_available():
        raise RuntimeError("point-anchor v2 training requires CUDA")
    device = torch.device("cuda")
    random.seed(int(args.seed))
    np.random.seed(int(args.seed) % (2**32 - 1))
    torch.manual_seed(int(args.seed))
    torch.cuda.manual_seed_all(int(args.seed))

    dataset = PointAnchorCacheDataset(args.point_cache_manifest, indices=args.indices)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=int(args.num_workers),
        collate_fn=collate_one,
        generator=torch.Generator().manual_seed(int(args.seed)),
    )
    sampler, stock_flow, flow_schema = build_frozen_stock_flow(
        args.pretrained, device
    )
    probe = PointAnchorProbe(rank=int(args.rank)).to(device)
    trainable = [parameter for parameter in probe.parameters() if parameter.requires_grad]
    if not trainable or any(parameter.requires_grad for parameter in stock_flow.parameters()):
        raise RuntimeError("point-anchor trainable parameter whitelist failed")
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
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    object_uids = sorted(
        {str(row.get("object_uid", row["uid"])) for row in dataset.rows}
    )
    model_summary = {
        "stage": "Point-only local-anchor v2 all-control training",
        "probe": probe.metadata(),
        "flow": flow_schema,
        "trainable_parameter_count": int(sum(parameter.numel() for parameter in trainable)),
        "trainable_parameter_names": [
            name for name, parameter in probe.named_parameters() if parameter.requires_grad
        ],
        "dataset_size": len(dataset),
        "unique_object_count": len(object_uids),
        "train_object_uids": object_uids,
        "control_names": list(POINT_CONTROL_NAMES),
        "all_controls_per_update": True,
        "fixed_correct_point_mask": True,
        "cache_config_hash": dataset.config_hash,
        "cache_manifest": str(dataset.manifest_path.resolve()),
        "source_manifest_sha256": json.loads(
            dataset.manifest_path.read_text(encoding="utf-8")
        )["source_manifest_sha256"],
        "stock_preservation": stock_preservation_audit(probe, dataset[0], device),
    }
    print(json.dumps(model_summary, indent=2), flush=True)

    use_amp = args.amp_dtype != "none"
    amp_dtype = torch.float16 if args.amp_dtype == "fp16" else torch.bfloat16
    global_step = 0
    micro_step = 0
    applied_updates = 0
    nonfinite_attempts = 0
    history: list[dict[str, Any]] = []
    last_row: dict[str, Any] | None = None
    start_time = time.time()
    optimizer.zero_grad(set_to_none=True)

    while global_step < int(args.max_steps):
        for sample in loader:
            if global_step >= int(args.max_steps):
                break
            with torch.no_grad():
                target = sample["target"].unsqueeze(0).to(
                    device=device, dtype=torch.float32
                )
                condition = sample["stock_condition"].to(device=device)
                correct_evidence = sample["point_correct_evidence"].unsqueeze(0).to(
                    device=device, dtype=torch.float32
                )
                controls = {
                    name: evidence.unsqueeze(0).to(device=device, dtype=torch.float32)
                    for name, evidence in sample["point_control_evidence"].items()
                }
                correct_mask = probe.active_mask(correct_evidence).to(device=device)
                if not all(
                    torch.equal(probe.active_mask(evidence), correct_mask)
                    for evidence in controls.values()
                ):
                    raise RuntimeError("point controls do not share the correct mask")
                noise = torch.randn_like(target)
                t_value = sample_t(str(args.t_schedule), device)
                x_t, gt_velocity = sampler._get_model_gt(target, t_value, noise)
                t_tensor = torch.full(
                    (1,), 1000.0 * t_value, device=device, dtype=torch.float32
                )
                with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
                    stock = stock_flow(x_t, t_tensor, condition)
                target_residual = gt_velocity.float() - stock.float()

            with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
                correct_delta, correct_stats = probe(
                    x_t,
                    stock,
                    t_tensor,
                    correct_evidence,
                    scale=float(args.physical_scale),
                    active_mask_override=correct_mask,
                )
                control_outputs = {
                    name: probe(
                        x_t,
                        stock,
                        t_tensor,
                        evidence,
                        scale=float(args.physical_scale),
                        active_mask_override=correct_mask,
                    )
                    for name, evidence in controls.items()
                }
                target_energy = masked_mse(
                    target_residual,
                    torch.zeros_like(target_residual),
                    correct_mask,
                ).detach().clamp_min(1.0e-6)
                local_target_loss = masked_mse(
                    correct_delta, target_residual, correct_mask
                ) / target_energy
                control_zero_losses = {
                    name: masked_mse(
                        delta,
                        torch.zeros_like(delta),
                        correct_mask,
                    )
                    / target_energy
                    for name, (delta, _) in control_outputs.items()
                }
                control_zero_loss = torch.stack(
                    tuple(control_zero_losses.values())
                ).mean()
                correct_flow_loss = F.mse_loss(
                    (stock + correct_delta).float(), gt_velocity.float()
                )
                control_flow_losses = {
                    name: F.mse_loss(
                        (stock + delta).float(), gt_velocity.float()
                    )
                    for name, (delta, _) in control_outputs.items()
                }
                stock_flow_loss = F.mse_loss(
                    stock.float(), gt_velocity.float()
                ).detach().clamp_min(1.0e-6)
                relative_gain = (
                    stock_flow_loss - correct_flow_loss
                ) / stock_flow_loss
                control_advantages = {
                    name: (loss - correct_flow_loss) / stock_flow_loss
                    for name, loss in control_flow_losses.items()
                }
                gain_loss = F.relu(
                    correct_flow_loss.new_tensor(float(args.gain_margin))
                    - relative_gain
                )
                rank_loss = torch.stack(
                    tuple(
                        F.relu(
                            correct_flow_loss.new_tensor(float(args.rank_margin))
                            - advantage
                        )
                        for advantage in control_advantages.values()
                    )
                ).mean()
                delta_norm_loss = masked_mse(
                    correct_delta,
                    torch.zeros_like(correct_delta),
                    correct_mask,
                ) / target_energy
                loss = (
                    float(args.local_target_weight) * local_target_loss
                    + float(args.flow_weight) * correct_flow_loss
                    + float(args.control_zero_weight) * control_zero_loss
                    + float(args.gain_weight) * gain_loss
                    + float(args.rank_weight) * rank_loss
                    + float(args.delta_norm_weight) * delta_norm_loss
                )
                scaled_loss = loss / float(args.grad_accum)
            scaler.scale(scaled_loss).backward()
            sync_step = ((micro_step + 1) % int(args.grad_accum)) == 0
            if sync_step:
                scaler.unscale_(optimizer)
                diagnostic_tensors = [
                    loss,
                    local_target_loss,
                    control_zero_loss,
                    correct_flow_loss,
                    stock_flow_loss,
                    relative_gain,
                    rank_loss,
                    correct_stats["delta_rms"],
                    correct_stats["neutral_abs_max"],
                    *control_flow_losses.values(),
                    *control_advantages.values(),
                ]
                forward_finite = tensors_finite(diagnostic_tensors)
                gradient_finite = gradients_finite(trainable)
                update_finite = forward_finite and gradient_finite
                scaler_before = float(scaler.get_scale()) if scaler.is_enabled() else None
                optimizer_step_applied = False
                clip_total_norm = None
                gradient_norms = gradient_group_norms(probe)
                if update_finite:
                    clip = torch.nn.utils.clip_grad_norm_(trainable, float(args.grad_clip))
                    clip_total_norm = float(clip.detach().float().item())
                    if scaler.is_enabled():
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer_step_applied = True
                    global_step += 1
                    applied_updates += 1
                    if not parameters_finite(trainable) or not optimizer_state_finite(optimizer):
                        raise RuntimeError("point-anchor optimizer produced non-finite state")
                else:
                    nonfinite_attempts += 1
                    if scaler.is_enabled():
                        scaler.update()
                scaler_after = float(scaler.get_scale()) if scaler.is_enabled() else None
                optimizer.zero_grad(set_to_none=True)
                last_row = {
                    "step": global_step,
                    "micro_step": micro_step + 1,
                    "uid": str(sample["uid"]),
                    "object_uid": str(sample.get("object_uid", sample["uid"])),
                    "loss": float(loss.detach().float().item()),
                    "local_target_loss": float(local_target_loss.detach().float().item()),
                    "control_zero_loss": float(control_zero_loss.detach().float().item()),
                    "correct_flow_loss": float(correct_flow_loss.detach().float().item()),
                    "stock_flow_loss": float(stock_flow_loss.detach().float().item()),
                    "relative_correct_gain": float(relative_gain.detach().float().item()),
                    "control_advantages": {
                        name: float(value.detach().float().item())
                        for name, value in control_advantages.items()
                    },
                    "correct_delta_rms": float(
                        correct_stats["delta_rms"].detach().float().item()
                    ),
                    "active_ratio": float(
                        correct_stats["active_ratio"].detach().float().item()
                    ),
                    "non_anchor_abs_max": float(
                        correct_stats["neutral_abs_max"].detach().float().item()
                    ),
                    "gradient_norms": gradient_norms,
                    "clip_total_norm": clip_total_norm,
                    "forward_finite": bool(forward_finite),
                    "gradient_finite": bool(gradient_finite),
                    "update_finite": bool(update_finite),
                    "optimizer_step_applied": bool(optimizer_step_applied),
                    "nonfinite_attempts": nonfinite_attempts,
                    "amp_dtype": args.amp_dtype,
                    "scaler_before": scaler_before,
                    "scaler_after": scaler_after,
                    "t": float(t_value),
                    "elapsed_seconds": float(time.time() - start_time),
                }
                if (
                    global_step == 1
                    or global_step % int(args.log_every) == 0
                    or not update_finite
                ):
                    history.append(last_row)
                    print(f"[point_anchor_train] {json.dumps(last_row)}", flush=True)
                if not update_finite and (
                    args.nonfinite_policy == "error"
                    or nonfinite_attempts > int(args.max_nonfinite_attempts)
                ):
                    raise RuntimeError(
                        f"non-finite point-anchor update attempt={nonfinite_attempts}"
                    )
                if optimizer_step_applied and (
                    global_step % int(args.save_every) == 0
                    or global_step == int(args.max_steps)
                ):
                    for filename in (f"step_{global_step:06d}.pt", "last.pt"):
                        save_checkpoint(
                            output_dir / "checkpoints" / filename,
                            probe=probe,
                            optimizer=optimizer,
                            scaler=scaler,
                            step=global_step,
                            args=args,
                            model_summary=model_summary,
                        )
            micro_step += 1

    report = {
        "stage": model_summary["stage"],
        "args": vars(args),
        "world_size": 1,
        "dataset_size": len(dataset),
        "unique_object_count": len(object_uids),
        "effective_batch_size": int(args.grad_accum),
        "start_global_step": 0,
        "applied_optimizer_updates": applied_updates,
        "completed_global_step": global_step,
        "nonfinite_attempts": nonfinite_attempts,
        "model": model_summary,
        "last_update": last_row,
        "history": history,
    }
    (output_dir / "train_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
