#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime
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
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

TRACKER_ROOT = Path(__file__).resolve().parents[1]
RECONVIAGEN_ROOT = TRACKER_ROOT / "ReconViaGen"
VGGT_ROOT = RECONVIAGEN_ROOT / "wheels" / "vggt"
for path in (TRACKER_ROOT, RECONVIAGEN_ROOT, VGGT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from trellis.pipelines import TrellisVGGTTo3DPipeline  # noqa: E402

from ar_ss_flow.local_pose_lifting_flow import (  # noqa: E402
    LOCAL_LIFTING_ADAPTER_VERSION,
    LocalPoseLiftingSSFlowModel,
    LocalPoseLiftingVelocityAdapter,
    PoseLiftingCacheDataset,
    collate_one,
    volume_from_sample,
)
from reconvggt_ar_adapter_a.pointpose_ss_condition import (  # noqa: E402
    load_partial_state,
    trainable_state_dict,
)
from reconvggt_ar_adapter_a.train_pointpose_ss_lora import (  # noqa: E402
    distributed_all_true,
    distributed_mean,
    finite_tree,
    gradients_finite,
    install_unused_model_stubs,
    optimizer_state_finite,
    parameters_finite,
    sample_t,
    tensors_finite,
)


CORRUPTION_MODES = ("pose_perturb", "pose_shuffle", "depth_corrupt")
HARD_CORRUPTION_MODES = ("pose_shuffle", "depth_corrupt")


def build_model(
    args: argparse.Namespace,
    device: torch.device,
    visual_feature_dim: int,
) -> tuple[Any, LocalPoseLiftingSSFlowModel, dict[str, Any]]:
    install_unused_model_stubs()
    pipeline = TrellisVGGTTo3DPipeline.from_pretrained(args.pretrained)
    sampler = pipeline.sparse_structure_sampler
    flow = pipeline.models["sparse_structure_flow_model"].to(device).eval()
    for parameter in flow.parameters():
        parameter.requires_grad = False
    if (
        int(flow.resolution) != 16
        or int(flow.in_channels) != 8
        or int(flow.out_channels) != 8
    ):
        raise RuntimeError(
            f"unexpected SS Flow schema: resolution={flow.resolution}, "
            f"in={flow.in_channels}, out={flow.out_channels}"
        )
    adapter = LocalPoseLiftingVelocityAdapter(
        visual_channels=int(visual_feature_dim),
        latent_channels=8,
        hidden_dim=int(args.adapter_hidden_dim),
    ).to(device)
    model = LocalPoseLiftingSSFlowModel(flow, adapter).to(device)
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    unexpected = [name for name in trainable if not name.startswith("adapter.")]
    stock_trainable = sum(
        parameter.numel() for parameter in flow.parameters() if parameter.requires_grad
    )
    if unexpected or stock_trainable or not trainable:
        raise RuntimeError(
            f"trainable whitelist failed: unexpected={unexpected}, stock={stock_trainable}"
        )
    summary = {
        "stage": "P3/P4 local 16^3 pose-lifting velocity residual",
        "adapter": adapter.metadata(),
        "trainable_parameter_names": trainable,
        "trainable_parameter_count": int(
            sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        ),
        "stock_flow_trainable_parameters": int(stock_trainable),
        "flow_lora_enabled": False,
        "bridge_loaded_or_trainable": False,
        "slat_enabled": False,
    }
    del pipeline
    gc.collect()
    torch.cuda.empty_cache()
    return sampler, model, summary


@torch.no_grad()
def stock_equivalence_audit(
    model: LocalPoseLiftingSSFlowModel,
    sample: dict[str, Any],
    device: torch.device,
    *,
    expect_zero_init: bool,
) -> dict[str, Any]:
    volume, metadata, _ = volume_from_sample(sample, device=device, mode="correct")
    condition = sample["stock_condition"].to(device=device)
    generator = torch.Generator(device=device).manual_seed(42014)
    x_t = torch.randn((1, 8, 16, 16, 16), generator=generator, device=device)
    t = torch.tensor([500.0], device=device)
    stock = model.stock_prediction(x_t, t, condition)
    disabled, _ = model.adapt_from_stock(
        x_t, t, stock, volume, metadata, physical_present=False
    )
    null, _ = model.adapt_from_stock(
        x_t, t, stock, torch.zeros_like(volume), torch.zeros_like(metadata)
    )
    enabled, _ = model.adapt_from_stock(x_t, t, stock, volume, metadata)
    report = {
        "physical_off_max_abs_diff": float((disabled - stock).abs().max().item()),
        "null_volume_max_abs_diff": float((null - stock).abs().max().item()),
        "enabled_max_abs_diff": float((enabled - stock).abs().max().item()),
        "expect_zero_init": bool(expect_zero_init),
    }
    if report["physical_off_max_abs_diff"] != 0.0 or report["null_volume_max_abs_diff"] != 0.0:
        raise RuntimeError(f"stock equivalence failed: {report}")
    if expect_zero_init and report["enabled_max_abs_diff"] != 0.0:
        raise RuntimeError(f"zero-init path differs from stock: {report}")
    return report


def gradient_group_norms(model: LocalPoseLiftingSSFlowModel) -> dict[str, float]:
    groups = {
        "state_projection": "adapter.state_projection.",
        "visual_projection": "adapter.visual_projection.",
        "metadata_projection": "adapter.metadata_projection.",
        "fusion": "adapter.fusion.",
        "output": "adapter.output.",
    }
    named = list(model.named_parameters())
    return {
        label: sum(
            float(parameter.grad.detach().float().square().sum().item())
            for name, parameter in named
            if name.startswith(prefix) and parameter.grad is not None
        )
        ** 0.5
        for label, prefix in groups.items()
    }


def validate_checkpoint(checkpoint: dict[str, Any], args: argparse.Namespace) -> None:
    saved = checkpoint.get("args", {})
    keys = (
        "pretrained",
        "adapter_hidden_dim",
        "amp_dtype",
    )
    mismatch = {
        key: {"checkpoint": saved.get(key), "current": getattr(args, key)}
        for key in keys
        if saved.get(key) != getattr(args, key)
    }
    if checkpoint.get("format") != LOCAL_LIFTING_ADAPTER_VERSION:
        mismatch["format"] = {
            "checkpoint": checkpoint.get("format"),
            "current": LOCAL_LIFTING_ADAPTER_VERSION,
        }
    if mismatch:
        raise RuntimeError(f"checkpoint configuration mismatch: {mismatch}")


def save_checkpoint(
    path: Path,
    *,
    model: LocalPoseLiftingSSFlowModel,
    optimizer: torch.optim.Optimizer,
    scaler,
    step: int,
    args: argparse.Namespace,
    summary: dict[str, Any],
) -> None:
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters_finite(trainable):
        raise RuntimeError(f"refusing to save non-finite model parameters: {path}")
    if not optimizer_state_finite(optimizer) or not finite_tree(scaler.state_dict()):
        raise RuntimeError(f"refusing to save non-finite optimizer/scaler state: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": LOCAL_LIFTING_ADAPTER_VERSION,
            "step": int(step),
            "model_trainable_state": trainable_state_dict(model),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "args": vars(args),
            "model_summary": summary,
        },
        path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a stock-preserving same-voxel 16^3 pose-lifting SS adapter."
    )
    parser.add_argument("--cache_manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--pretrained", default="Stable-X/trellis-vggt-v0-2")
    parser.add_argument("--indices", default="all")
    parser.add_argument("--resume", default="")
    parser.add_argument("--max_steps", type=int, default=50)
    parser.add_argument("--save_every", type=int, default=10)
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
    parser.add_argument("--adapter_hidden_dim", type=int, default=96)
    parser.add_argument("--physical_scale", type=float, default=1.0)
    parser.add_argument("--t_schedule", choices=("uniform", "logit_normal", "high_t_mix"), default="uniform")
    parser.add_argument("--corruption_modes", default=",".join(CORRUPTION_MODES))
    parser.add_argument("--flow_weight", type=float, default=1.0)
    parser.add_argument("--corrupt_stock_weight", type=float, default=0.10)
    parser.add_argument("--correct_gain_weight", type=float, default=0.10)
    parser.add_argument("--correct_gain_margin", type=float, default=0.005)
    parser.add_argument("--correct_corrupt_rank_weight", type=float, default=0.05)
    parser.add_argument("--correct_corrupt_margin", type=float, default=0.002)
    parser.add_argument("--delta_norm_weight", type=float, default=0.01)
    parser.add_argument("--perturb_flow_weight", type=float, default=0.25)
    parser.add_argument("--perturb_consistency_weight", type=float, default=0.05)
    parser.add_argument("--perturb_gain_weight", type=float, default=0.05)
    parser.add_argument("--perturb_gain_margin", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    modes = tuple(item.strip() for item in args.corruption_modes.split(",") if item.strip())
    if not modes or any(mode not in CORRUPTION_MODES for mode in modes):
        raise ValueError(f"invalid corruption modes: {modes}")
    if min(args.max_steps, args.save_every, args.log_every, args.grad_accum) <= 0:
        raise ValueError("steps/log/accumulation arguments must be positive")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        dist.init_process_group(backend="nccl", timeout=datetime.timedelta(hours=2))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    process_seed = int(args.seed) + rank * 100003
    random.seed(process_seed)
    np.random.seed(process_seed % (2**32 - 1))
    torch.manual_seed(process_seed)
    torch.cuda.manual_seed_all(process_seed)

    dataset = PoseLiftingCacheDataset(args.cache_manifest, indices=args.indices)
    sampler_ddp = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=int(args.seed),
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        sampler=sampler_ddp,
        num_workers=int(args.num_workers),
        collate_fn=collate_one,
        pin_memory=False,
    )
    flow_sampler, model, summary = build_model(
        args, device, dataset.visual_feature_dim
    )
    summary["dataset_size"] = len(dataset)
    summary["unique_object_count"] = len(
        {str(row.get("object_uid", row["uid"])) for row in dataset.rows}
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
        enabled=args.amp_dtype == "fp16", init_scale=float(args.amp_init_scale)
    )
    start_step = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        validate_checkpoint(checkpoint, args)
        load_partial_state(
            model,
            checkpoint["model_trainable_state"],
            require_all_trainable=True,
        )
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_step = int(checkpoint.get("step", 0))
    summary["stock_equivalence"] = stock_equivalence_audit(
        model, dataset[0], device, expect_zero_init=not bool(args.resume)
    )
    if rank == 0:
        print(json.dumps(summary, indent=2), flush=True)
    wrapped: nn.Module = model
    if world_size > 1:
        wrapped = DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            find_unused_parameters=False,
        )
    output_dir = Path(args.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=False)
    if world_size > 1:
        dist.barrier()

    use_amp = args.amp_dtype != "none"
    amp_dtype = torch.float16 if args.amp_dtype == "fp16" else torch.bfloat16
    global_step = start_step
    applied_updates = 0
    micro_step = start_step * int(args.grad_accum)
    nonfinite_attempts = 0
    history: list[dict[str, Any]] = []
    optimizer.zero_grad(set_to_none=True)
    epoch = 0
    wall_start = time.time()
    last_row: dict[str, Any] | None = None
    while global_step < int(args.max_steps):
        sampler_ddp.set_epoch(epoch)
        for batch in loader:
            if global_step >= int(args.max_steps):
                break
            corruption_mode = random.choice(modes)
            with torch.no_grad():
                correct_volume, correct_metadata, correct_volume_stats = volume_from_sample(
                    batch, device=device, mode="correct"
                )
                corrupt_volume, corrupt_metadata, corrupt_volume_stats = volume_from_sample(
                    batch, device=device, mode=corruption_mode
                )
                target = batch["target"].unsqueeze(0).to(
                    device=device, dtype=torch.float32
                )
                condition = batch["stock_condition"].to(device=device)
                noise = torch.randn_like(target)
                t_model = sample_t(str(args.t_schedule), device)
                x_t, gt_velocity = flow_sampler._get_model_gt(target, t_model, noise)
                t_tensor = torch.full(
                    (1,), 1000.0 * t_model, device=device, dtype=torch.float32
                )
                stock_prediction = model.stock_prediction(x_t, t_tensor, condition)
            sync_step = ((micro_step + 1) % int(args.grad_accum)) == 0
            sync_context = (
                wrapped.no_sync()
                if world_size > 1 and not sync_step
                else torch.enable_grad()
            )
            with sync_context:
                with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
                    (
                        prediction,
                        corrupt_prediction,
                        returned_stock,
                        correct_stats,
                        corrupt_stats,
                    ) = wrapped(
                        x_t,
                        t_tensor,
                        None,
                        correct_volume,
                        correct_metadata,
                        stock_velocity=stock_prediction,
                        corrupted_visual_volume=corrupt_volume,
                        corrupted_metadata=corrupt_metadata,
                        scale=float(args.physical_scale),
                    )
                    if corrupt_prediction is None or corrupt_stats is None:
                        raise RuntimeError("paired adapter forward did not return corruption output")
                    if returned_stock.data_ptr() != stock_prediction.data_ptr():
                        raise RuntimeError("paired adapter did not reuse the supplied stock velocity")
                    correct_flow_loss = F.mse_loss(
                        prediction.float(), gt_velocity.float()
                    )
                    stock_flow_loss = F.mse_loss(
                        stock_prediction.float(), gt_velocity.float()
                    ).detach()
                    corrupt_flow_loss = F.mse_loss(
                        corrupt_prediction.float(), gt_velocity.float()
                    )
                    stock_energy = stock_prediction.float().square().mean().detach().clamp_min(1.0e-6)
                    corrupt_stock_loss = F.mse_loss(
                        corrupt_prediction.float(), stock_prediction.float()
                    ) / stock_energy
                    relative_gain = (
                        stock_flow_loss - correct_flow_loss
                    ) / stock_flow_loss.clamp_min(1.0e-6)
                    gain_loss = F.relu(
                        prediction.new_tensor(float(args.correct_gain_margin))
                        - relative_gain
                    )
                    relative_correct_corrupt = (
                        corrupt_flow_loss - correct_flow_loss
                    ) / stock_flow_loss.clamp_min(1.0e-6)
                    rank_loss = F.relu(
                        prediction.new_tensor(float(args.correct_corrupt_margin))
                        - relative_correct_corrupt
                    )
                    delta_norm_loss = 0.5 * (
                        F.mse_loss(prediction.float(), stock_prediction.float())
                        + F.mse_loss(corrupt_prediction.float(), stock_prediction.float())
                    ) / stock_energy
                    perturb_consistency_loss = F.mse_loss(
                        corrupt_prediction.float(), prediction.float()
                    ) / stock_energy
                    perturb_relative_gain = (
                        stock_flow_loss - corrupt_flow_loss
                    ) / stock_flow_loss.clamp_min(1.0e-6)
                    perturb_gain_loss = F.relu(
                        prediction.new_tensor(float(args.perturb_gain_margin))
                        - perturb_relative_gain
                    )
                    if corruption_mode in HARD_CORRUPTION_MODES:
                        corruption_role = "hard_invalid_to_stock"
                        corruption_objective = (
                            float(args.corrupt_stock_weight) * corrupt_stock_loss
                            + float(args.correct_corrupt_rank_weight) * rank_loss
                        )
                    else:
                        corruption_role = "robust_perturbation_to_target"
                        corruption_objective = (
                            float(args.perturb_flow_weight) * corrupt_flow_loss
                            + float(args.perturb_consistency_weight)
                            * perturb_consistency_loss
                            + float(args.perturb_gain_weight) * perturb_gain_loss
                        )
                    loss = (
                        float(args.flow_weight) * correct_flow_loss
                        + float(args.correct_gain_weight) * gain_loss
                        + float(args.delta_norm_weight) * delta_norm_loss
                        + corruption_objective
                    )
                    scaled_loss = loss / float(args.grad_accum)
                scaler.scale(scaled_loss).backward()

            if sync_step:
                scaler.unscale_(optimizer)
                grad_norms = gradient_group_norms(model)
                diagnostics = (
                    loss,
                    correct_flow_loss,
                    stock_flow_loss,
                    corrupt_flow_loss,
                    corrupt_stock_loss,
                    gain_loss,
                    rank_loss,
                    delta_norm_loss,
                    perturb_consistency_loss,
                    perturb_gain_loss,
                    correct_stats["delta_rms"],
                    corrupt_stats["delta_rms"],
                )
                forward_finite = distributed_all_true(
                    tensors_finite(list(diagnostics)), device, world_size
                )
                gradient_finite = distributed_all_true(
                    gradients_finite(trainable), device, world_size
                )
                update_finite = forward_finite and gradient_finite
                scaler_before = float(scaler.get_scale()) if scaler.is_enabled() else None
                optimizer_step_applied = False
                clip_total_norm = None
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
                    post_finite = parameters_finite(trainable) and optimizer_state_finite(optimizer)
                    if not distributed_all_true(post_finite, device, world_size):
                        raise RuntimeError("optimizer step produced non-finite parameters/state")
                else:
                    nonfinite_attempts += 1
                    if scaler.is_enabled():
                        scaler.update()
                scaler_after = float(scaler.get_scale()) if scaler.is_enabled() else None
                optimizer.zero_grad(set_to_none=True)
                last_row = {
                    "step": int(global_step),
                    "micro_step": int(micro_step + 1),
                    "uid": str(batch["uid"]),
                    "object_uid": str(batch["object_uid"]),
                    "corruption_mode": corruption_mode,
                    "corruption_role": corruption_role,
                    "loss": distributed_mean(loss, world_size),
                    "correct_flow_loss": distributed_mean(correct_flow_loss, world_size),
                    "stock_flow_loss": distributed_mean(stock_flow_loss, world_size),
                    "corrupt_flow_loss": distributed_mean(corrupt_flow_loss, world_size),
                    "relative_correct_gain": distributed_mean(relative_gain, world_size),
                    "relative_correct_vs_corrupt": distributed_mean(relative_correct_corrupt, world_size),
                    "corrupt_stock_loss": distributed_mean(corrupt_stock_loss, world_size),
                    "gain_loss": distributed_mean(gain_loss, world_size),
                    "rank_loss": distributed_mean(rank_loss, world_size),
                    "delta_norm_loss": distributed_mean(delta_norm_loss, world_size),
                    "perturb_relative_gain": distributed_mean(
                        perturb_relative_gain, world_size
                    ),
                    "perturb_consistency_loss": distributed_mean(
                        perturb_consistency_loss, world_size
                    ),
                    "perturb_gain_loss": distributed_mean(
                        perturb_gain_loss, world_size
                    ),
                    "correct_delta_rms": distributed_mean(correct_stats["delta_rms"], world_size),
                    "corrupt_delta_rms": distributed_mean(corrupt_stats["delta_rms"], world_size),
                    "correct_supported_voxel_ratio": correct_volume_stats["supported_voxel_ratio"],
                    "corrupt_supported_voxel_ratio": corrupt_volume_stats["supported_voxel_ratio"],
                    "depth_consistency_enabled": correct_volume_stats["depth_consistency_enabled"],
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
                    "t": float(t_model),
                    "elapsed_seconds": float(time.time() - wall_start),
                }
                should_log = (
                    global_step == 1
                    or global_step % int(args.log_every) == 0
                    or not update_finite
                )
                if rank == 0 and should_log:
                    history.append(last_row)
                    print(f"[pose_lifting_train] {json.dumps(last_row)}", flush=True)
                if not update_finite and (
                    args.nonfinite_policy == "error"
                    or nonfinite_attempts > int(args.max_nonfinite_attempts)
                ):
                    raise RuntimeError(
                        f"non-finite update attempt={nonfinite_attempts} micro_step={micro_step + 1}"
                    )
                if optimizer_step_applied and rank == 0 and (
                    global_step % int(args.save_every) == 0
                    or global_step == int(args.max_steps)
                ):
                    save_checkpoint(
                        output_dir / "checkpoints" / f"step_{global_step:06d}.pt",
                        model=model,
                        optimizer=optimizer,
                        scaler=scaler,
                        step=global_step,
                        args=args,
                        summary=summary,
                    )
                    save_checkpoint(
                        output_dir / "checkpoints" / "last.pt",
                        model=model,
                        optimizer=optimizer,
                        scaler=scaler,
                        step=global_step,
                        args=args,
                        summary=summary,
                    )
            micro_step += 1
        epoch += 1

    if rank == 0:
        report = {
            "stage": summary["stage"],
            "args": vars(args),
            "world_size": world_size,
            "dataset_size": len(dataset),
            "unique_object_count": summary["unique_object_count"],
            "effective_batch_size": world_size * int(args.grad_accum),
            "start_global_step": start_step,
            "applied_optimizer_updates": applied_updates,
            "completed_global_step": global_step,
            "nonfinite_attempts": nonfinite_attempts,
            "model": summary,
            "last_update": last_row,
            "history": history,
        }
        (output_dir / "train_report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
