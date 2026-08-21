#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime
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

from reconvggt_ar_adapter_a.pointpose_ss_condition import (  # noqa: E402
    load_partial_state,
    trainable_state_dict,
)
from reconvggt_ar_adapter_a.sparse_anchor_flow import (  # noqa: E402
    SparseAnchorSSFlowModel,
    SparseAnchorVelocityAdapter,
    build_sparse_anchor_masks,
    make_null_physical_grid,
    shift_sparse_prior,
)
from reconvggt_ar_adapter_a.train_pointpose_ss_lora import (  # noqa: E402
    PointPoseCacheDataset,
    collate_one,
    distributed_all_true,
    distributed_mean,
    encode_frozen_features,
    finite_tree,
    gradients_finite,
    install_unused_model_stubs,
    optimizer_state_finite,
    parameters_finite,
    rgba_images,
    sample_t,
    tensors_finite,
)


def parse_shifts(text: str) -> list[tuple[int, int, int]]:
    shifts: list[tuple[int, int, int]] = []
    for item in str(text).split(";"):
        if not item.strip():
            continue
        shift = tuple(int(value.strip()) for value in item.split(","))
        if len(shift) != 3 or shift == (0, 0, 0):
            raise ValueError(f"invalid corruption shift: {item!r}")
        shifts.append(shift)
    if not shifts:
        raise ValueError("--corruption_shifts must contain at least one shift")
    return shifts


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    selected = values[mask.expand_as(values)]
    if selected.numel() == 0:
        raise RuntimeError("masked loss received an empty mask")
    return selected.float().mean()


def masked_mse(left: torch.Tensor, right: torch.Tensor, mask16: torch.Tensor) -> torch.Tensor:
    return masked_mean((left.float() - right.float()).square(), mask16)


def gradient_group_norms(model: SparseAnchorSSFlowModel) -> dict[str, float]:
    groups = {
        "physical_encoder": "adapter.physical_encoder.",
        "state_encoder": "adapter.state_encoder.",
        "time_mlp": "adapter.time_mlp.",
        "fusion": "adapter.fusion.",
        "output": "adapter.output.",
    }
    named = list(model.named_parameters())
    output: dict[str, float] = {}
    for label, prefix in groups.items():
        square = sum(
            float(parameter.grad.detach().float().square().sum().item())
            for name, parameter in named
            if name.startswith(prefix) and parameter.grad is not None
        )
        output[label] = square**0.5
    return output


def build_pipeline_and_model(
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[Any, SparseAnchorSSFlowModel, nn.Module, dict[str, Any]]:
    install_unused_model_stubs()
    pipeline = TrellisVGGTTo3DPipeline.from_pretrained(args.pretrained)
    pipeline._device = device
    pipeline.low_vram = False
    keep_models = {
        "image_cond_model",
        "sparse_structure_vggt_cond",
        "sparse_structure_flow_model",
        "sparse_structure_decoder",
    }
    for name in list(pipeline.models):
        if name not in keep_models:
            del pipeline.models[name]
    for name in ("slat_flow_model", "slat_vggt_cond"):
        if hasattr(pipeline, name):
            delattr(pipeline, name)
    for name in ("camera_head", "point_head", "depth_head", "track_head"):
        if hasattr(pipeline.VGGT_model, name):
            delattr(pipeline.VGGT_model, name)
    for module in pipeline.models.values():
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad = False
    pipeline.VGGT_model.to(device).eval()
    for parameter in pipeline.VGGT_model.parameters():
        parameter.requires_grad = False
    pipeline.models["image_cond_model"].to(device).eval()
    bridge = pipeline.models["sparse_structure_vggt_cond"].to(device).eval()
    flow = pipeline.models["sparse_structure_flow_model"].to(device).eval()
    decoder = pipeline.models["sparse_structure_decoder"].to(device).eval()
    for module in (bridge, flow, decoder):
        for parameter in module.parameters():
            parameter.requires_grad = False

    if int(flow.resolution) != 16 or int(flow.in_channels) != 8 or int(flow.out_channels) != 8:
        raise RuntimeError(
            f"unexpected SS Flow schema: resolution={flow.resolution}, "
            f"in={flow.in_channels}, out={flow.out_channels}"
        )
    adapter = SparseAnchorVelocityAdapter(
        latent_channels=int(flow.in_channels),
        hidden_dim=int(args.anchor_hidden_dim),
        prior_confidence_min=float(args.prior_confidence_min),
        anchor_radius_16=int(args.anchor_radius_16),
        outside_visible_min=float(args.outside_visible_min),
        outside_ratio_min=float(args.outside_ratio_min),
    ).to(device)
    model = SparseAnchorSSFlowModel(flow, adapter).to(device)
    model.stock_flow.eval()
    model.adapter.train()
    trainable_names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    unexpected = [name for name in trainable_names if not name.startswith("adapter.")]
    frozen_counts = {
        "vggt": int(sum(p.numel() for p in pipeline.VGGT_model.parameters() if p.requires_grad)),
        "image_encoder": int(sum(
            p.numel() for p in pipeline.models["image_cond_model"].parameters() if p.requires_grad
        )),
        "bridge": int(sum(p.numel() for p in bridge.parameters() if p.requires_grad)),
        "stock_flow": int(sum(p.numel() for p in flow.parameters() if p.requires_grad)),
        "decoder": int(sum(p.numel() for p in decoder.parameters() if p.requires_grad)),
    }
    if unexpected or any(frozen_counts.values()) or not trainable_names:
        raise RuntimeError(
            f"trainable whitelist failure: unexpected={unexpected}, frozen_counts={frozen_counts}"
        )
    summary = {
        "stage": "SS16 sparse-anchor velocity residual",
        "adapter": adapter.metadata(),
        "trainable_parameter_names": trainable_names,
        "trainable_parameter_count": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "frozen_trainable_counts": frozen_counts,
        "decoder_logits_differentiable": True,
        "flow_lora_enabled": False,
        "bridge_trainable": False,
        "slat_enabled": False,
    }
    return pipeline, model, decoder, summary


@torch.no_grad()
def build_stock_condition(
    pipeline,
    batch: dict[str, Any],
    device: torch.device,
) -> torch.Tensor:
    images = rgba_images(batch["image_paths"], batch["mask_paths"], pipeline)
    aggregated, image_cond = encode_frozen_features(pipeline, images)
    return pipeline.get_ss_cond(image_cond, aggregated, num_samples=1)["cond"].to(device)


@torch.no_grad()
def stock_equivalence_audit(
    pipeline,
    model: SparseAnchorSSFlowModel,
    batch: dict[str, Any],
    device: torch.device,
    *,
    expect_zero_init: bool,
) -> dict[str, Any]:
    cond = build_stock_condition(pipeline, batch, device)
    physical = batch["physical_grid"].unsqueeze(0).to(device=device, dtype=torch.float32)
    null = make_null_physical_grid(physical)
    generator = torch.Generator(device=device).manual_seed(42013)
    x_t = torch.randn((1, 8, 16, 16, 16), device=device, generator=generator)
    t = torch.tensor([500.0], device=device)
    stock = model.stock_prediction(x_t, t, cond)
    disabled, _ = model.adapt_from_stock(
        x_t, t, stock, physical, physical_present=False
    )
    null_output, _ = model.adapt_from_stock(x_t, t, stock, null)
    enabled, _ = model.adapt_from_stock(x_t, t, stock, physical)
    report = {
        "disabled_max_abs_diff": float((disabled - stock).abs().max().item()),
        "null_max_abs_diff": float((null_output - stock).abs().max().item()),
        "enabled_max_abs_diff": float((enabled - stock).abs().max().item()),
        "expect_zero_init": bool(expect_zero_init),
    }
    if report["disabled_max_abs_diff"] != 0.0 or report["null_max_abs_diff"] != 0.0:
        raise RuntimeError(f"stock equivalence audit failed: {report}")
    if expect_zero_init and report["enabled_max_abs_diff"] != 0.0:
        raise RuntimeError(f"zero-init enabled path differs from stock: {report}")
    return report


def validate_checkpoint(checkpoint: dict[str, Any], args: argparse.Namespace) -> None:
    saved = checkpoint.get("args", {})
    keys = (
        "pretrained",
        "anchor_hidden_dim",
        "prior_confidence_min",
        "anchor_radius_16",
        "outside_visible_min",
        "outside_ratio_min",
        "amp_dtype",
    )
    mismatch = {
        key: {"checkpoint": saved.get(key), "current": getattr(args, key)}
        for key in keys
        if saved.get(key) != getattr(args, key)
    }
    if mismatch:
        raise RuntimeError(f"checkpoint configuration mismatch: {mismatch}")


def save_checkpoint(
    path: Path,
    *,
    model: SparseAnchorSSFlowModel,
    optimizer: torch.optim.Optimizer,
    scaler,
    step: int,
    args: argparse.Namespace,
    model_summary: dict[str, Any],
) -> None:
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters_finite(trainable):
        raise RuntimeError(f"refusing to save non-finite model parameters: {path}")
    if not optimizer_state_finite(optimizer) or not finite_tree(scaler.state_dict()):
        raise RuntimeError(f"refusing to save non-finite optimizer/scaler state: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "reconvggt.sparse_anchor_ss_flow.v1",
            "step": int(step),
            "model_trainable_state": trainable_state_dict(model),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "args": vars(args),
            "model_summary": model_summary,
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a stock-preserving 16^3 sparse-anchor SS Flow velocity adapter."
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
    parser.add_argument("--amp_dtype", choices=["fp16", "bf16", "none"], default="bf16")
    parser.add_argument("--amp_init_scale", type=float, default=8192.0)
    parser.add_argument("--nonfinite_policy", choices=["error", "skip"], default="error")
    parser.add_argument("--max_nonfinite_attempts", type=int, default=0)
    parser.add_argument("--anchor_hidden_dim", type=int, default=96)
    parser.add_argument("--physical_scale", type=float, default=1.0)
    parser.add_argument("--prior_confidence_min", type=float, default=0.25)
    parser.add_argument("--prior_mask_support_min", type=float, default=0.0)
    parser.add_argument("--anchor_radius_16", type=int, default=1)
    parser.add_argument("--outside_visible_min", type=float, default=0.5)
    parser.add_argument("--outside_ratio_min", type=float, default=0.9)
    parser.add_argument("--negative_surface_margin_64", type=int, default=1)
    parser.add_argument("--corruption_shifts", default="1,0,0;-1,0,0;0,1,0;0,-1,0;0,0,1;0,0,-1")
    parser.add_argument("--t_schedule", choices=["uniform", "logit_normal", "high_t_mix"], default="uniform")
    parser.add_argument("--local_loss_every", type=int, default=1)
    parser.add_argument("--flow_weight", type=float, default=1.0)
    parser.add_argument("--positive_weight", type=float, default=0.05)
    parser.add_argument("--negative_weight", type=float, default=0.02)
    parser.add_argument("--corruption_rank_weight", type=float, default=0.05)
    parser.add_argument("--corruption_logit_margin", type=float, default=0.02)
    parser.add_argument("--neutral_preserve_weight", type=float, default=0.5)
    parser.add_argument("--delta_norm_weight", type=float, default=0.01)
    args = parser.parse_args()

    if args.max_steps <= 0 or args.save_every <= 0 or args.log_every <= 0:
        raise ValueError("max_steps, save_every and log_every must be positive")
    if args.grad_accum <= 0 or args.local_loss_every <= 0:
        raise ValueError("grad_accum and local_loss_every must be positive")
    if args.grad_clip <= 0 or args.amp_init_scale <= 0:
        raise ValueError("grad_clip and amp_init_scale must be positive")
    corruption_shifts = parse_shifts(args.corruption_shifts)

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

    dataset = PointPoseCacheDataset(args.cache_manifest, indices=args.indices)
    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=int(args.seed)
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        sampler=sampler,
        num_workers=int(args.num_workers),
        collate_fn=collate_one,
        pin_memory=True,
    )
    pipeline, model, decoder, model_summary = build_pipeline_and_model(args, device)
    model_summary["dataset_size"] = len(dataset)
    model_summary["unique_object_count"] = len(
        {str(row.get("object_uid", row["uid"])) for row in dataset.samples}
    )
    if rank == 0:
        print(json.dumps(model_summary, indent=2, ensure_ascii=False), flush=True)

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

    model_summary["stock_equivalence"] = stock_equivalence_audit(
        pipeline,
        model,
        dataset[0],
        device,
        expect_zero_init=not bool(args.resume),
    )
    wrapped: nn.Module = model
    if world_size > 1:
        wrapped = DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            find_unused_parameters=False,
        )

    use_amp = args.amp_dtype != "none"
    amp_dtype = torch.float16 if args.amp_dtype == "fp16" else torch.bfloat16
    decoder_dtype = next(decoder.parameters()).dtype
    output_dir = Path(args.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=False)
    if world_size > 1:
        dist.barrier()
    history: list[dict[str, Any]] = []
    global_step = start_step
    micro_step = global_step * int(args.grad_accum)
    applied_updates = 0
    nonfinite_attempts = 0
    optimizer.zero_grad(set_to_none=True)
    epoch = 0
    wall_start = time.time()

    while global_step < int(args.max_steps):
        sampler.set_epoch(epoch)
        for batch in loader:
            if global_step >= int(args.max_steps):
                break
            physical = batch["physical_grid"].unsqueeze(0).to(
                device=device, dtype=torch.float32, non_blocking=True
            )
            target = batch["target"].unsqueeze(0).to(
                device=device, dtype=torch.float32, non_blocking=True
            )
            target_coords = batch["target_coords"].to(device=device)
            masks = build_sparse_anchor_masks(
                physical,
                target_coords,
                prior_confidence_min=float(args.prior_confidence_min),
                prior_mask_support_min=float(args.prior_mask_support_min),
                anchor_radius_16=int(args.anchor_radius_16),
                outside_visible_min=float(args.outside_visible_min),
                outside_ratio_min=float(args.outside_ratio_min),
                negative_surface_margin_64=int(args.negative_surface_margin_64),
            )
            shift = random.choice(corruption_shifts)
            corrupted = shift_sparse_prior(physical, shift)
            cond = build_stock_condition(pipeline, batch, device)
            noise = torch.randn_like(target)
            t_model = sample_t(str(args.t_schedule), device)
            x_t, gt_velocity = pipeline.sparse_structure_sampler._get_model_gt(
                target, t_model, noise
            )
            t_tensor = torch.full(
                (1,), 1000.0 * t_model, device=device, dtype=torch.float32
            )
            local_step = (global_step % int(args.local_loss_every)) == 0
            sync_step = ((micro_step + 1) % int(args.grad_accum)) == 0
            sync_context = (
                wrapped.no_sync() if world_size > 1 and not sync_step else torch.enable_grad()
            )

            with sync_context:
                with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
                    prediction, corrupted_prediction, stock_prediction, stats, corrupted_stats = wrapped(
                        x_t,
                        t_tensor,
                        cond,
                        physical,
                        corrupted_physical_grid=corrupted,
                        physical_scale=float(args.physical_scale),
                    )
                    flow_loss = F.mse_loss(prediction.float(), gt_velocity.float())
                    neutral_loss = masked_mse(
                        prediction, stock_prediction, masks["neutral16"]
                    )
                    delta_norm_loss = F.mse_loss(
                        prediction.float(), stock_prediction.float()
                    )
                    positive_loss = prediction.new_zeros((), dtype=torch.float32)
                    negative_loss = prediction.new_zeros((), dtype=torch.float32)
                    rank_loss = prediction.new_zeros((), dtype=torch.float32)
                    positive_gain = prediction.new_zeros((), dtype=torch.float32)
                    corrupted_gain = prediction.new_zeros((), dtype=torch.float32)
                    outside_degradation = prediction.new_zeros((), dtype=torch.float32)
                    if local_step:
                        pred_x0 = pipeline.sparse_structure_sampler._pred_to_xstart(
                            x_t, t_model, prediction
                        )
                        corrupt_x0 = pipeline.sparse_structure_sampler._pred_to_xstart(
                            x_t, t_model, corrupted_prediction
                        )
                        with torch.no_grad():
                            stock_x0 = pipeline.sparse_structure_sampler._pred_to_xstart(
                                x_t, t_model, stock_prediction
                            )
                            stock_logits = decoder(stock_x0.to(dtype=decoder_dtype)).float()
                        logits = decoder(pred_x0.to(dtype=decoder_dtype)).float()
                        corrupt_logits = decoder(corrupt_x0.to(dtype=decoder_dtype)).float()
                        positive_loss = masked_mean(F.softplus(-logits), masks["positive64"])
                        negative_loss = masked_mean(F.softplus(logits), masks["negative64"])
                        correct_vs_corrupt = logits - corrupt_logits
                        rank_loss = masked_mean(
                            F.relu(float(args.corruption_logit_margin) - correct_vs_corrupt),
                            masks["positive64"],
                        )
                        positive_gain = masked_mean(
                            torch.sigmoid(logits) - torch.sigmoid(stock_logits),
                            masks["positive64"],
                        )
                        corrupted_gain = masked_mean(
                            torch.sigmoid(logits) - torch.sigmoid(corrupt_logits),
                            masks["positive64"],
                        )
                        outside_degradation = masked_mean(
                            torch.sigmoid(logits) - torch.sigmoid(stock_logits),
                            masks["negative64"],
                        )
                    loss = (
                        float(args.flow_weight) * flow_loss
                        + float(args.positive_weight) * positive_loss
                        + float(args.negative_weight) * negative_loss
                        + float(args.corruption_rank_weight) * rank_loss
                        + float(args.neutral_preserve_weight) * neutral_loss
                        + float(args.delta_norm_weight) * delta_norm_loss
                    )
                    scaled_loss = loss / float(args.grad_accum)
                scaler.scale(scaled_loss).backward()

            if sync_step:
                scaler.unscale_(optimizer)
                grad_norms = gradient_group_norms(model)
                diagnostic_values = [
                    loss,
                    flow_loss,
                    positive_loss,
                    negative_loss,
                    rank_loss,
                    neutral_loss,
                    delta_norm_loss,
                    stats["delta_rms"],
                    stats["delta_abs_max"],
                    stats["gate_ratio"],
                ]
                forward_finite = distributed_all_true(
                    tensors_finite(diagnostic_values), device, world_size
                )
                gradient_finite = distributed_all_true(
                    gradients_finite(trainable), device, world_size
                )
                update_finite = forward_finite and gradient_finite
                scaler_before = float(scaler.get_scale()) if scaler.is_enabled() else None
                clip_total_norm = None
                optimizer_step_applied = False
                if update_finite:
                    clip_tensor = torch.nn.utils.clip_grad_norm_(trainable, float(args.grad_clip))
                    clip_total_norm = float(clip_tensor.detach().float().item())
                    if scaler.is_enabled():
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer_step_applied = True
                    global_step += 1
                    applied_updates += 1
                    local_post_finite = parameters_finite(trainable) and optimizer_state_finite(optimizer)
                    if not distributed_all_true(local_post_finite, device, world_size):
                        raise RuntimeError("optimizer step produced non-finite parameters/state")
                else:
                    nonfinite_attempts += 1
                    if scaler.is_enabled():
                        scaler.update()
                scaler_after = float(scaler.get_scale()) if scaler.is_enabled() else None
                optimizer.zero_grad(set_to_none=True)

                should_log = global_step == 1 or global_step % int(args.log_every) == 0 or not update_finite
                row = {
                    "step": int(global_step),
                    "micro_step": int(micro_step + 1),
                    "uid": str(batch["uid"]),
                    "object_uid": str(batch["object_uid"]),
                    "corruption_shift": list(shift),
                    "loss": distributed_mean(loss, world_size),
                    "flow_loss": distributed_mean(flow_loss, world_size),
                    "positive_loss": distributed_mean(positive_loss, world_size),
                    "negative_loss": distributed_mean(negative_loss, world_size),
                    "corruption_rank_loss": distributed_mean(rank_loss, world_size),
                    "neutral_preserve_loss": distributed_mean(neutral_loss, world_size),
                    "delta_norm_loss": distributed_mean(delta_norm_loss, world_size),
                    "positive_probability_gain": distributed_mean(positive_gain, world_size),
                    "correct_minus_corrupted_probability": distributed_mean(corrupted_gain, world_size),
                    "outside_probability_degradation": distributed_mean(outside_degradation, world_size),
                    "delta_rms": distributed_mean(stats["delta_rms"], world_size),
                    "gate_ratio": distributed_mean(stats["gate_ratio"], world_size),
                    "anchor_gate_ratio": distributed_mean(stats["anchor_gate_ratio"], world_size),
                    "outside_gate_ratio": distributed_mean(stats["outside_gate_ratio"], world_size),
                    "corrupted_delta_rms": distributed_mean(corrupted_stats["delta_rms"], world_size),
                    "positive16_count": int(masks["positive16"].sum().item()),
                    "negative16_count": int(masks["negative16"].sum().item()),
                    "positive64_count": int(masks["positive64"].sum().item()),
                    "negative64_count": int(masks["negative64"].sum().item()),
                    "gradient_norms": grad_norms,
                    "clip_total_norm": clip_total_norm,
                    "forward_finite": bool(forward_finite),
                    "gradient_finite": bool(gradient_finite),
                    "update_finite": bool(update_finite),
                    "optimizer_step_applied": bool(optimizer_step_applied),
                    "nonfinite_attempts": int(nonfinite_attempts),
                    "amp_dtype": str(args.amp_dtype),
                    "scaler_before": scaler_before,
                    "scaler_after": scaler_after,
                    "t": float(t_model),
                    "local_loss_step": bool(local_step),
                    "elapsed_seconds": float(time.time() - wall_start),
                }
                if rank == 0 and should_log:
                    history.append(row)
                    print(f"[sparse_anchor_train] {json.dumps(row, ensure_ascii=False)}", flush=True)
                if not update_finite:
                    message = (
                        f"non-finite update attempt={nonfinite_attempts} "
                        f"micro_step={micro_step + 1}"
                    )
                    if (
                        args.nonfinite_policy == "error"
                        or nonfinite_attempts > int(args.max_nonfinite_attempts)
                    ):
                        raise RuntimeError(message)
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
                        model_summary=model_summary,
                    )
                    save_checkpoint(
                        output_dir / "checkpoints" / "last.pt",
                        model=model,
                        optimizer=optimizer,
                        scaler=scaler,
                        step=global_step,
                        args=args,
                        model_summary=model_summary,
                    )
            micro_step += 1
        epoch += 1

    if rank == 0:
        report = {
            "stage": "SS16 sparse-anchor velocity residual",
            "args": vars(args),
            "world_size": world_size,
            "per_rank_batch_size": 1,
            "effective_batch_size": world_size * int(args.grad_accum),
            "dataset_size": len(dataset),
            "unique_object_count": model_summary["unique_object_count"],
            "start_global_step": start_step,
            "applied_optimizer_updates": applied_updates,
            "completed_global_step": global_step,
            "nonfinite_attempts": nonfinite_attempts,
            "model": model_summary,
            "history": history,
        }
        (output_dir / "train_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
