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

from ar_ss_flow.local_pose_lifting_flow import (
    PoseLiftingCacheDataset,
    collate_one,
)
from pose_point_depth_mv.train_local_target_probe import (
    build_frozen_stock_flow,
    masked_mse,
)
from pose_point_depth_mv.view_identity_lifting import (
    VIEW_IDENTITY_CHECKPOINT_VERSION,
    VIEW_IDENTITY_CONTROL_NAMES,
    ViewIdentityPoseDepthProbe,
    build_view_identity_evidence,
    make_null_view_identity_evidence,
    trainable_state_dict,
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
        description="Train same-voxel view-identity pose/depth SS probe."
    )
    parser.add_argument("--cache_manifest", required=True)
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
    parser.add_argument("--hidden_dim", type=int, default=96)
    parser.add_argument("--pair_dim", type=int, default=32)
    parser.add_argument("--min_views", type=int, default=2)
    parser.add_argument("--physical_scale", type=float, default=1.0)
    parser.add_argument(
        "--t_schedule",
        choices=("uniform", "logit_normal", "high_t_mix"),
        default="uniform",
    )
    parser.add_argument("--local_target_weight", type=float, default=1.0)
    parser.add_argument("--flow_weight", type=float, default=0.10)
    parser.add_argument("--corrupt_zero_weight", type=float, default=0.25)
    parser.add_argument("--gain_weight", type=float, default=0.10)
    parser.add_argument("--gain_margin", type=float, default=0.00005)
    parser.add_argument("--rank_weight", type=float, default=0.25)
    parser.add_argument("--rank_margin", type=float, default=0.00001)
    parser.add_argument("--delta_norm_weight", type=float, default=0.01)
    return parser.parse_args()


def gradient_group_norms(probe: ViewIdentityPoseDepthProbe) -> dict[str, float]:
    groups = {
        "state": ("state_projection.",),
        "visual": ("visual_encoder.",),
        "geometry": ("geometry_encoder.",),
        "view_attention": (
            "query_projection.",
            "key_projection.",
            "value_projection.",
        ),
        "pairwise": ("pair_projection.", "pair_logit_scale"),
        "fusion": ("fusion.",),
        "output": ("output.",),
    }
    named = list(probe.named_parameters())
    return {
        label: sum(
            float(parameter.grad.detach().float().square().sum().item())
            for name, parameter in named
            if any(name.startswith(prefix) for prefix in prefixes)
            and parameter.grad is not None
        )
        ** 0.5
        for label, prefixes in groups.items()
    }


def save_checkpoint(
    path: Path,
    *,
    probe: ViewIdentityPoseDepthProbe,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    step: int,
    args: argparse.Namespace,
    model_summary: dict[str, Any],
) -> None:
    trainable = [parameter for parameter in probe.parameters() if parameter.requires_grad]
    if not parameters_finite(trainable):
        raise RuntimeError(f"refusing to save non-finite view probe: {path}")
    if not optimizer_state_finite(optimizer):
        raise RuntimeError(f"refusing to save non-finite optimizer: {path}")
    if not finite_tree(scaler.state_dict()):
        raise RuntimeError(f"refusing to save non-finite scaler: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": VIEW_IDENTITY_CHECKPOINT_VERSION,
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
    probe: ViewIdentityPoseDepthProbe,
    sample: dict[str, Any],
    device: torch.device,
) -> dict[str, float]:
    evidence = build_view_identity_evidence(sample, device=device, mode="correct")
    null = make_null_view_identity_evidence(evidence)
    x_t = torch.randn(1, 8, 16, 16, 16, device=device)
    stock = torch.randn_like(x_t)
    t = torch.tensor([500.0], device=device)
    enabled, stats = probe(x_t, stock, t, evidence)
    disabled, _ = probe(x_t, stock, t, evidence, physical_present=False)
    null_delta, _ = probe(x_t, stock, t, null)
    report = {
        "zero_init_enabled_max_abs": float(enabled.float().abs().max().item()),
        "physical_off_max_abs": float(disabled.float().abs().max().item()),
        "null_evidence_max_abs": float(null_delta.float().abs().max().item()),
        "neutral_max_abs": float(stats["neutral_abs_max"].item()),
        "support_ratio": float(stats["support_ratio"].item()),
    }
    if any(report[name] != 0.0 for name in (
        "zero_init_enabled_max_abs",
        "physical_off_max_abs",
        "null_evidence_max_abs",
        "neutral_max_abs",
    )):
        raise RuntimeError(f"view-identity stock preservation failed: {report}")
    if report["support_ratio"] <= 0.0:
        raise RuntimeError("view-identity evidence has no multi-view support")
    return report


def main() -> None:
    args = parse_args()
    if min(args.max_steps, args.save_every, args.log_every, args.grad_accum) <= 0:
        raise ValueError("step/log/accumulation values must be positive")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise RuntimeError("view-identity mechanism probe currently requires one GPU")
    if not torch.cuda.is_available():
        raise RuntimeError("view-identity training requires CUDA")
    device = torch.device("cuda")
    random.seed(int(args.seed))
    np.random.seed(int(args.seed) % (2**32 - 1))
    torch.manual_seed(int(args.seed))
    torch.cuda.manual_seed_all(int(args.seed))

    dataset = PoseLiftingCacheDataset(args.cache_manifest, indices=args.indices)
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
    probe = ViewIdentityPoseDepthProbe(
        visual_channels=dataset.visual_feature_dim,
        hidden_dim=int(args.hidden_dim),
        pair_dim=int(args.pair_dim),
        min_views=int(args.min_views),
    ).to(device)
    trainable = [parameter for parameter in probe.parameters() if parameter.requires_grad]
    if not trainable or any(parameter.requires_grad for parameter in stock_flow.parameters()):
        raise RuntimeError("view-identity trainable parameter whitelist failed")
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
        "stage": "View-identity pose-guided visual lifting V1",
        "probe": probe.metadata(),
        "flow": flow_schema,
        "trainable_parameter_count": int(sum(p.numel() for p in trainable)),
        "trainable_parameter_names": [
            name for name, parameter in probe.named_parameters() if parameter.requires_grad
        ],
        "dataset_size": len(dataset),
        "unique_object_count": len(object_uids),
        "train_object_uids": object_uids,
        "control_names": list(VIEW_IDENTITY_CONTROL_NAMES),
        "all_controls_per_update": True,
        "fixed_correct_view_weight": True,
        "fixed_correct_support_gate": True,
        "cache_config_hash": dataset.config_hash,
        "cache_manifest": str(dataset.manifest_path.resolve()),
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
                correct_evidence = build_view_identity_evidence(
                    sample, device=device, mode="correct"
                )
                control_evidence = {
                    mode: build_view_identity_evidence(
                        sample, device=device, mode=mode
                    )
                    for mode in VIEW_IDENTITY_CONTROL_NAMES
                }
                correct_view_weight = correct_evidence["view_weight"].float()
                correct_support = probe.support_gate(
                    correct_evidence,
                    view_weight_override=correct_view_weight,
                )
                active_mask = correct_support.reshape(1, 1, 16, 16, 16)
                if not bool(active_mask.any().item()):
                    raise RuntimeError(f"uid={sample['uid']} has empty multi-view support")
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
                    view_weight_override=correct_view_weight,
                    support_gate_override=correct_support,
                )
                control_outputs = {
                    mode: probe(
                        x_t,
                        stock,
                        t_tensor,
                        evidence,
                        scale=float(args.physical_scale),
                        view_weight_override=correct_view_weight,
                        support_gate_override=correct_support,
                    )
                    for mode, evidence in control_evidence.items()
                }
                target_energy = masked_mse(
                    target_residual,
                    torch.zeros_like(target_residual),
                    active_mask,
                ).detach().clamp_min(1.0e-6)
                local_target_loss = masked_mse(
                    correct_delta, target_residual, active_mask
                ) / target_energy
                control_zero_losses = {
                    mode: masked_mse(
                        delta, torch.zeros_like(delta), active_mask
                    )
                    / target_energy
                    for mode, (delta, _) in control_outputs.items()
                }
                control_zero_loss = torch.stack(
                    tuple(control_zero_losses.values())
                ).mean()
                correct_flow_loss = F.mse_loss(
                    (stock + correct_delta).float(), gt_velocity.float()
                )
                control_flow_losses = {
                    mode: F.mse_loss(
                        (stock + delta).float(), gt_velocity.float()
                    )
                    for mode, (delta, _) in control_outputs.items()
                }
                stock_flow_loss = F.mse_loss(
                    stock.float(), gt_velocity.float()
                ).detach().clamp_min(1.0e-6)
                relative_gain = (
                    stock_flow_loss - correct_flow_loss
                ) / stock_flow_loss
                control_advantages = {
                    mode: (loss - correct_flow_loss) / stock_flow_loss
                    for mode, loss in control_flow_losses.items()
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
                    active_mask,
                ) / target_energy
                loss = (
                    float(args.local_target_weight) * local_target_loss
                    + float(args.flow_weight) * correct_flow_loss
                    + float(args.corrupt_zero_weight) * control_zero_loss
                    + float(args.gain_weight) * gain_loss
                    + float(args.rank_weight) * rank_loss
                    + float(args.delta_norm_weight) * delta_norm_loss
                )
                scaled_loss = loss / float(args.grad_accum)
            scaler.scale(scaled_loss).backward()
            sync_step = ((micro_step + 1) % int(args.grad_accum)) == 0
            if sync_step:
                scaler.unscale_(optimizer)
                diagnostics = [
                    loss,
                    local_target_loss,
                    control_zero_loss,
                    correct_flow_loss,
                    stock_flow_loss,
                    relative_gain,
                    rank_loss,
                    correct_stats["delta_rms"],
                    correct_stats["neutral_abs_max"],
                    correct_stats["pair_consensus"],
                    *control_flow_losses.values(),
                    *control_advantages.values(),
                ]
                forward_finite = tensors_finite(diagnostics)
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
                        raise RuntimeError("view-identity optimizer produced non-finite state")
                else:
                    nonfinite_attempts += 1
                    if scaler.is_enabled():
                        scaler.update()
                scaler_after = float(scaler.get_scale()) if scaler.is_enabled() else None
                optimizer.zero_grad(set_to_none=True)
                row = {
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
                        mode: float(value.detach().float().item())
                        for mode, value in control_advantages.items()
                    },
                    "delta_rms": float(correct_stats["delta_rms"].detach().float().item()),
                    "support_ratio": float(correct_stats["support_ratio"].detach().float().item()),
                    "pair_consensus": float(correct_stats["pair_consensus"].detach().float().item()),
                    "attention_entropy": float(correct_stats["attention_entropy"].detach().float().item()),
                    "gradient_norms": gradient_norms,
                    "clip_total_norm": clip_total_norm,
                    "forward_finite": bool(forward_finite),
                    "gradient_finite": bool(gradient_finite),
                    "update_finite": bool(update_finite),
                    "optimizer_step_applied": bool(optimizer_step_applied),
                    "nonfinite_attempts": nonfinite_attempts,
                    "scaler_before": scaler_before,
                    "scaler_after": scaler_after,
                    "t": float(t_value),
                    "elapsed_seconds": time.time() - start_time,
                }
                if (
                    global_step <= 1
                    or global_step % int(args.log_every) == 0
                    or not update_finite
                ):
                    history.append(row)
                    print(f"[view_identity_train] {json.dumps(row)}", flush=True)
                if not update_finite:
                    message = (
                        "non-finite view-identity update "
                        f"attempt={nonfinite_attempts} micro_step={micro_step + 1}"
                    )
                    if (
                        args.nonfinite_policy == "error"
                        or nonfinite_attempts > int(args.max_nonfinite_attempts)
                    ):
                        raise RuntimeError(message)
                if optimizer_step_applied and (
                    global_step % int(args.save_every) == 0
                    or global_step == int(args.max_steps)
                ):
                    checkpoint = output_dir / "checkpoints" / f"step_{global_step:06d}.pt"
                    save_checkpoint(
                        checkpoint,
                        probe=probe,
                        optimizer=optimizer,
                        scaler=scaler,
                        step=global_step,
                        args=args,
                        model_summary=model_summary,
                    )
                    save_checkpoint(
                        output_dir / "checkpoints" / "last.pt",
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
        "model_summary": model_summary,
        "start_global_step": 0,
        "completed_global_step": global_step,
        "applied_optimizer_updates": applied_updates,
        "nonfinite_attempts": nonfinite_attempts,
        "elapsed_seconds": time.time() - start_time,
        "history": history,
    }
    (output_dir / "train_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "completed_global_step": global_step,
        "applied_optimizer_updates": applied_updates,
        "nonfinite_attempts": nonfinite_attempts,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
