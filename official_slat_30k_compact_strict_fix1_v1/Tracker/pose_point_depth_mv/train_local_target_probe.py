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
from torch import nn
from torch.utils.data import DataLoader

TRACKER_ROOT = Path(__file__).resolve().parents[1]
RECONVIAGEN_ROOT = TRACKER_ROOT / "ReconViaGen"
VGGT_ROOT = RECONVIAGEN_ROOT / "wheels" / "vggt"
for path in (TRACKER_ROOT, RECONVIAGEN_ROOT, VGGT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from trellis.pipelines import TrellisVGGTTo3DPipeline  # noqa: E402

from ar_ss_flow.local_pose_lifting_flow import collate_one  # noqa: E402
from pose_point_depth_mv.local_target_probe import (  # noqa: E402
    PPD_LOCAL_TARGET_PROBE_VERSION,
    PPDLocalTargetProbe,
    PPDProbeEvidenceDataset,
    PROBE_CORRUPTIONS,
    make_null_evidence,
)
from reconvggt_ar_adapter_a.train_pointpose_ss_lora import (  # noqa: E402
    finite_tree,
    gradients_finite,
    install_unused_model_stubs,
    optimizer_state_finite,
    parameters_finite,
    sample_t,
    tensors_finite,
)


CHECKPOINT_VERSION = "pose_point_depth_mv.local_target_probe_checkpoint.v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the PPD-3A same-voxel low-rank learnability probe for "
            "the local target residual v_gt - v_stock."
        )
    )
    parser.add_argument("--probe_cache_manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--pretrained", default="Stable-X/trellis-vggt-v0-2")
    parser.add_argument("--indices", default="0-15")
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--save_every", type=int, default=25)
    parser.add_argument("--log_every", type=int, default=1)
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
    parser.add_argument("--corruption_modes", default=",".join(PROBE_CORRUPTIONS))
    parser.add_argument("--local_target_weight", type=float, default=1.0)
    parser.add_argument("--flow_weight", type=float, default=0.10)
    parser.add_argument("--corrupt_zero_weight", type=float, default=0.25)
    parser.add_argument("--gain_weight", type=float, default=0.10)
    parser.add_argument("--gain_margin", type=float, default=0.001)
    parser.add_argument("--rank_weight", type=float, default=0.10)
    parser.add_argument("--rank_margin", type=float, default=0.001)
    parser.add_argument("--delta_norm_weight", type=float, default=0.01)
    return parser.parse_args()


def build_frozen_stock_flow(
    pretrained: str, device: torch.device
) -> tuple[Any, nn.Module, dict[str, Any]]:
    install_unused_model_stubs()
    pipeline = TrellisVGGTTo3DPipeline.from_pretrained(pretrained)
    sampler = pipeline.sparse_structure_sampler
    flow = pipeline.models["sparse_structure_flow_model"].to(device).eval()
    for parameter in flow.parameters():
        parameter.requires_grad = False
    schema = {
        "resolution": int(flow.resolution),
        "in_channels": int(flow.in_channels),
        "out_channels": int(flow.out_channels),
        "trainable_parameters": int(
            sum(parameter.numel() for parameter in flow.parameters() if parameter.requires_grad)
        ),
    }
    if schema != {
        "resolution": 16,
        "in_channels": 8,
        "out_channels": 8,
        "trainable_parameters": 0,
    }:
        raise RuntimeError(f"unexpected frozen SS Flow schema: {schema}")
    del pipeline
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return sampler, flow, schema


def masked_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise ValueError("masked MSE prediction/target shapes differ")
    if mask.shape != (prediction.shape[0], 1, *prediction.shape[-3:]):
        raise ValueError(f"invalid masked MSE mask shape={tuple(mask.shape)}")
    denominator = mask.float().sum() * float(prediction.shape[1])
    if float(denominator.detach().item()) <= 0.0:
        raise ValueError("masked MSE received an empty active mask")
    return (
        (prediction.float() - target.float()).square() * mask.float()
    ).sum() / denominator


def gradient_group_norms(probe: PPDLocalTargetProbe) -> dict[str, float]:
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


def trainable_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu()
        for name, parameter in module.named_parameters()
        if parameter.requires_grad
    }


def save_checkpoint(
    path: Path,
    *,
    probe: PPDLocalTargetProbe,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    step: int,
    args: argparse.Namespace,
    model_summary: dict[str, Any],
) -> None:
    trainable = [parameter for parameter in probe.parameters() if parameter.requires_grad]
    if not parameters_finite(trainable):
        raise RuntimeError(f"refusing to save non-finite probe parameters: {path}")
    if not optimizer_state_finite(optimizer):
        raise RuntimeError(f"refusing to save non-finite optimizer state: {path}")
    if not finite_tree(scaler.state_dict()):
        raise RuntimeError(f"refusing to save non-finite scaler state: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": CHECKPOINT_VERSION,
            "probe_version": PPD_LOCAL_TARGET_PROBE_VERSION,
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
    probe: PPDLocalTargetProbe,
    sample: dict[str, Any],
    device: torch.device,
) -> dict[str, float]:
    target = sample["target"].unsqueeze(0).to(device=device, dtype=torch.float32)
    stock = torch.randn_like(target)
    x_t = torch.randn_like(target)
    t = torch.full((1,), 500.0, device=device)
    evidence = sample["ppd_correct_features"].unsqueeze(0).to(device=device)
    disabled, _ = probe(x_t, stock, t, evidence, physical_present=False)
    null_evidence = make_null_evidence(evidence)
    null, _ = probe(x_t, stock, t, null_evidence)
    enabled, stats = probe(x_t, stock, t, evidence)
    report = {
        "physical_off_delta_abs_max": float(disabled.abs().max().item()),
        "null_evidence_delta_abs_max": float(null.abs().max().item()),
        "zero_init_enabled_delta_abs_max": float(enabled.abs().max().item()),
        "neutral_abs_max": float(stats["neutral_abs_max"].item()),
    }
    if any(value != 0.0 for value in report.values()):
        raise RuntimeError(f"PPD-3A stock preservation audit failed: {report}")
    return report


def main() -> None:
    args = parse_args()
    modes = tuple(
        item.strip() for item in args.corruption_modes.split(",") if item.strip()
    )
    if not modes or any(mode not in PROBE_CORRUPTIONS for mode in modes):
        raise ValueError(f"invalid corruption modes={modes}")
    if min(args.max_steps, args.save_every, args.log_every, args.grad_accum) <= 0:
        raise ValueError("step/log/accumulation arguments must be positive")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise RuntimeError(
            "PPD-3A is an intentionally low-cost single-GPU learnability probe"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("PPD-3A Flow training requires CUDA")
    device = torch.device("cuda")
    random.seed(int(args.seed))
    np.random.seed(int(args.seed) % (2**32 - 1))
    torch.manual_seed(int(args.seed))
    torch.cuda.manual_seed_all(int(args.seed))

    dataset = PPDProbeEvidenceDataset(
        args.probe_cache_manifest, indices=args.indices, eligible_only=True
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=int(args.num_workers),
        collate_fn=collate_one,
        pin_memory=False,
        generator=torch.Generator().manual_seed(int(args.seed)),
    )
    sampler, stock_flow, flow_schema = build_frozen_stock_flow(
        args.pretrained, device
    )
    probe = PPDLocalTargetProbe(rank=int(args.rank)).to(device)
    trainable = [parameter for parameter in probe.parameters() if parameter.requires_grad]
    if not trainable or any(parameter.requires_grad for parameter in stock_flow.parameters()):
        raise RuntimeError("PPD-3A trainable parameter whitelist failed")
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        betas=(0.9, 0.95),
        eps=1.0e-8,
    )
    scaler = torch.cuda.amp.GradScaler(
        enabled=args.amp_dtype == "fp16", init_scale=float(args.amp_init_scale)
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    model_summary = {
        "stage": "PPD-3A local target residual learnability probe",
        "probe": probe.metadata(),
        "flow": flow_schema,
        "trainable_parameter_count": int(
            sum(parameter.numel() for parameter in trainable)
        ),
        "trainable_parameter_names": [
            name for name, parameter in probe.named_parameters() if parameter.requires_grad
        ],
        "dataset_size": len(dataset),
        "unique_object_count": len(
            {str(row.get("object_uid", row["uid"])) for row in dataset.rows}
        ),
        "corruption_modes": list(modes),
        "evidence_cache_config_hash": dataset.config_hash,
        "evidence_cache_manifest": str(dataset.manifest_path.resolve()),
        "old_c2_loaded": False,
        "stock_preservation": stock_preservation_audit(probe, dataset[0], device),
    }
    print(json.dumps(model_summary, indent=2), flush=True)

    use_amp = args.amp_dtype != "none"
    amp_dtype = torch.float16 if args.amp_dtype == "fp16" else torch.bfloat16
    global_step = 0
    applied_updates = 0
    micro_step = 0
    nonfinite_attempts = 0
    history: list[dict[str, Any]] = []
    last_row: dict[str, Any] | None = None
    wall_start = time.time()
    optimizer.zero_grad(set_to_none=True)
    while global_step < int(args.max_steps):
        for sample in loader:
            if global_step >= int(args.max_steps):
                break
            corruption_mode = random.choice(modes)
            with torch.no_grad():
                target = sample["target"].unsqueeze(0).to(
                    device=device, dtype=torch.float32
                )
                condition = sample["stock_condition"].to(device=device)
                correct_evidence = sample["ppd_correct_features"].unsqueeze(0).to(
                    device=device, dtype=torch.float32
                )
                corrupt_evidence = sample["ppd_corrupt_features"][
                    corruption_mode
                ].unsqueeze(0).to(device=device, dtype=torch.float32)
                noise = torch.randn_like(target)
                t_value = sample_t(str(args.t_schedule), device)
                x_t, gt_velocity = sampler._get_model_gt(target, t_value, noise)
                t_tensor = torch.full(
                    (1,), 1000.0 * t_value, device=device, dtype=torch.float32
                )
                with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
                    stock_prediction = stock_flow(x_t, t_tensor, condition)
                target_residual = gt_velocity.float() - stock_prediction.float()
            with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
                correct_delta, correct_stats = probe(
                    x_t,
                    stock_prediction,
                    t_tensor,
                    correct_evidence,
                    scale=float(args.physical_scale),
                )
                corrupt_delta, corrupt_stats = probe(
                    x_t,
                    stock_prediction,
                    t_tensor,
                    corrupt_evidence,
                    scale=float(args.physical_scale),
                )
                correct_prediction = stock_prediction + correct_delta
                corrupt_prediction = stock_prediction + corrupt_delta
                correct_mask = probe.active_mask(correct_evidence).to(device=device)
                corrupt_mask = probe.active_mask(corrupt_evidence).to(device=device)
                target_energy = masked_mse(
                    target_residual,
                    torch.zeros_like(target_residual),
                    correct_mask,
                ).detach().clamp_min(1.0e-6)
                local_target_loss = masked_mse(
                    correct_delta, target_residual, correct_mask
                ) / target_energy
                if bool(corrupt_mask.any().item()):
                    corrupt_zero_loss = masked_mse(
                        corrupt_delta,
                        torch.zeros_like(corrupt_delta),
                        corrupt_mask,
                    ) / target_energy
                else:
                    corrupt_zero_loss = corrupt_delta.float().sum() * 0.0
                correct_flow_loss = F.mse_loss(
                    correct_prediction.float(), gt_velocity.float()
                )
                corrupt_flow_loss = F.mse_loss(
                    corrupt_prediction.float(), gt_velocity.float()
                )
                stock_flow_loss = F.mse_loss(
                    stock_prediction.float(), gt_velocity.float()
                ).detach().clamp_min(1.0e-6)
                relative_gain = (
                    stock_flow_loss - correct_flow_loss
                ) / stock_flow_loss
                relative_correct_vs_corrupt = (
                    corrupt_flow_loss - correct_flow_loss
                ) / stock_flow_loss
                gain_loss = F.relu(
                    correct_flow_loss.new_tensor(float(args.gain_margin))
                    - relative_gain
                )
                rank_loss = F.relu(
                    correct_flow_loss.new_tensor(float(args.rank_margin))
                    - relative_correct_vs_corrupt
                )
                delta_norm_loss = masked_mse(
                    correct_delta,
                    torch.zeros_like(correct_delta),
                    correct_mask,
                ) / target_energy
                loss = (
                    float(args.local_target_weight) * local_target_loss
                    + float(args.flow_weight) * correct_flow_loss
                    + float(args.corrupt_zero_weight) * corrupt_zero_loss
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
                    corrupt_zero_loss,
                    correct_flow_loss,
                    corrupt_flow_loss,
                    stock_flow_loss,
                    relative_gain,
                    relative_correct_vs_corrupt,
                    correct_stats["delta_rms"],
                    corrupt_stats["delta_rms"],
                    correct_stats["neutral_abs_max"],
                    corrupt_stats["neutral_abs_max"],
                ]
                forward_finite = tensors_finite(diagnostics)
                gradient_finite = gradients_finite(trainable)
                update_finite = forward_finite and gradient_finite
                scaler_before = float(scaler.get_scale()) if scaler.is_enabled() else None
                optimizer_step_applied = False
                clip_total_norm = None
                grad_norms = gradient_group_norms(probe)
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
                    if not parameters_finite(trainable) or not optimizer_state_finite(
                        optimizer
                    ):
                        raise RuntimeError(
                            "optimizer step produced non-finite parameters/state"
                        )
                else:
                    nonfinite_attempts += 1
                    if scaler.is_enabled():
                        scaler.update()
                scaler_after = float(scaler.get_scale()) if scaler.is_enabled() else None
                optimizer.zero_grad(set_to_none=True)
                last_row = {
                    "step": int(global_step),
                    "micro_step": int(micro_step + 1),
                    "uid": str(sample["uid"]),
                    "object_uid": str(sample.get("object_uid", sample["uid"])),
                    "corruption_mode": corruption_mode,
                    "loss": float(loss.detach().float().item()),
                    "local_target_loss": float(local_target_loss.detach().float().item()),
                    "corrupt_zero_loss": float(corrupt_zero_loss.detach().float().item()),
                    "correct_flow_loss": float(correct_flow_loss.detach().float().item()),
                    "corrupt_flow_loss": float(corrupt_flow_loss.detach().float().item()),
                    "stock_flow_loss": float(stock_flow_loss.detach().float().item()),
                    "relative_correct_gain": float(relative_gain.detach().float().item()),
                    "relative_correct_vs_corrupt": float(
                        relative_correct_vs_corrupt.detach().float().item()
                    ),
                    "correct_delta_rms": float(
                        correct_stats["delta_rms"].detach().float().item()
                    ),
                    "corrupt_delta_rms": float(
                        corrupt_stats["delta_rms"].detach().float().item()
                    ),
                    "correct_active_ratio": float(
                        correct_stats["active_ratio"].detach().float().item()
                    ),
                    "correct_neutral_abs_max": float(
                        correct_stats["neutral_abs_max"].detach().float().item()
                    ),
                    "corrupt_neutral_abs_max": float(
                        corrupt_stats["neutral_abs_max"].detach().float().item()
                    ),
                    "gradient_norms": grad_norms,
                    "clip_total_norm": clip_total_norm,
                    "forward_finite": bool(forward_finite),
                    "gradient_finite": bool(gradient_finite),
                    "update_finite": bool(update_finite),
                    "optimizer_step_applied": bool(optimizer_step_applied),
                    "nonfinite_attempts": int(nonfinite_attempts),
                    "amp_dtype": args.amp_dtype,
                    "scaler_before": scaler_before,
                    "scaler_after": scaler_after,
                    "t": float(t_value),
                    "elapsed_seconds": float(time.time() - wall_start),
                }
                should_log = (
                    global_step == 1
                    or global_step % int(args.log_every) == 0
                    or not update_finite
                )
                if should_log:
                    history.append(last_row)
                    print(f"[ppd3a_train] {json.dumps(last_row)}", flush=True)
                if not update_finite and (
                    args.nonfinite_policy == "error"
                    or nonfinite_attempts > int(args.max_nonfinite_attempts)
                ):
                    raise RuntimeError(
                        f"non-finite update attempt={nonfinite_attempts} "
                        f"micro_step={micro_step + 1}"
                    )
                if optimizer_step_applied and (
                    global_step % int(args.save_every) == 0
                    or global_step == int(args.max_steps)
                ):
                    save_checkpoint(
                        output_dir / "checkpoints" / f"step_{global_step:06d}.pt",
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
        "world_size": 1,
        "dataset_size": len(dataset),
        "unique_object_count": model_summary["unique_object_count"],
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
