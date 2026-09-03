#!/usr/bin/env python3
"""Train adapter-only native 16^3 every-block SS conditioning."""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import nullcontext
import datetime
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, Sampler

from ar_ss_flow.local_pose_lifting_flow import PoseLiftingCacheDataset, collate_one
from pose_aligned_reconstruction.direct_flow import lifting_cache_identity
from pose_aligned_reconstruction.native_3d_condition import (
    NATIVE_CONTROL_MODES,
    NATIVE_SS_DELTA_BOUND_SMOOTH,
    NATIVE_SS_FLOW_VERSION,
    NATIVE_SS_GUIDED_DELTA_POLICY,
    NATIVE_SS_SUPPORT_INTERVAL_CFG_ACTIVE,
    NATIVE_SS_TRAINING_SEMANTICS,
    block_projection_gradient_norms,
    build_native_ss_components,
    ensure_finite_trainable,
    drop_cached_native_projection,
    load_trainable_state_dict,
    native_ss_cfg_is_active,
    native_ss_timestep_sequence,
    normalize_native_cfg_interval,
    sha256_file,
    trainable_state_dict,
    validate_lifting_feature_metadata,
    validate_native_checkpoint,
)
from pose_aligned_reconstruction.native_ss_occupancy import (
    NATIVE_SS_OCCUPANCY_OBJECTIVE_VERSION,
    frozen_decoder_occupancy_objective,
    objective_scalars,
    target_occupancy_grid,
)
from pose_aligned_reconstruction.train_direct_flow import ObjectBalancedDistributedSampler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache_manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--pretrained", default="Stable-X/trellis-vggt-v0-2")
    parser.add_argument("--indices", default="all")
    parser.add_argument("--max_objects", type=int, default=0)
    parser.add_argument("--resume", default="")
    parser.add_argument("--max_steps", type=int, default=300)
    parser.add_argument("--save_every", type=int, default=50)
    parser.add_argument("--log_every", type=int, default=5)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--amp_dtype", choices=("bf16", "fp16", "none"), default="bf16")
    parser.add_argument("--amp_init_scale", type=float, default=8192.0)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--feature_source", choices=("dino", "vggt", "all"), default="dino")
    parser.add_argument("--condition_scale", type=float, default=1.0)
    parser.add_argument("--delta_norm_weight", type=float, default=1.0e-4)
    parser.add_argument(
        "--training_semantics",
        choices=(NATIVE_SS_TRAINING_SEMANTICS,),
        default=NATIVE_SS_TRAINING_SEMANTICS,
    )
    parser.add_argument(
        "--guided_delta_policy",
        choices=(NATIVE_SS_GUIDED_DELTA_POLICY,),
        default=NATIVE_SS_GUIDED_DELTA_POLICY,
    )
    parser.add_argument(
        "--support_interval_policy",
        choices=(NATIVE_SS_SUPPORT_INTERVAL_CFG_ACTIVE,),
        default=NATIVE_SS_SUPPORT_INTERVAL_CFG_ACTIVE,
    )
    parser.add_argument(
        "--delta_bound_mode",
        choices=(NATIVE_SS_DELTA_BOUND_SMOOTH,),
        default=NATIVE_SS_DELTA_BOUND_SMOOTH,
    )
    parser.add_argument("--train_cfg_strength", type=float, default=5.0)
    parser.add_argument("--train_cfg_interval", default="0.5,1.0")
    parser.add_argument("--native_steps", type=int, default=25)
    parser.add_argument("--rescale_t", type=float, default=3.0)
    parser.add_argument("--guidance_rescale", type=float, default=0.0)
    parser.add_argument("--delta_scale", type=float, default=1.0)
    parser.add_argument("--delta_rms_ratio_cap", type=float, default=0.10)
    parser.add_argument("--raw_delta_excess_weight", type=float, default=0.10)
    parser.add_argument("--occupancy_weight", type=float, default=0.0)
    parser.add_argument("--occupancy_every", type=int, default=8)
    parser.add_argument("--occupancy_false_negative_weight", type=float, default=1.0)
    parser.add_argument("--occupancy_false_positive_weight", type=float, default=0.25)
    parser.add_argument("--occupancy_stock_recall_rank_weight", type=float, default=0.5)
    parser.add_argument("--occupancy_false_negative_margin", type=float, default=0.0)
    parser.add_argument("--occupancy_stock_recall_margin", type=float, default=0.0)
    parser.add_argument("--control_mode", choices=NATIVE_CONTROL_MODES, default="pose_cyclic1")
    parser.add_argument("--control_probability", type=float, default=0.0)
    parser.add_argument("--control_rank_weight", type=float, default=0.0)
    parser.add_argument("--control_rank_margin", type=float, default=0.0)
    parser.add_argument("--sampling_mode", choices=("object_balanced", "sequence"), default="object_balanced")
    parser.add_argument(
        "--t_schedule", choices=("native_cfg_active",), default="native_cfg_active"
    )
    parser.add_argument("--gradient_checkpointing", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    args.train_cfg_interval = normalize_native_cfg_interval(
        float(item.strip())
        for item in str(args.train_cfg_interval).split(",")
        if item.strip()
    )
    if int(args.max_objects) < 0:
        raise ValueError("max_objects must be non-negative")
    positive = (
        "max_steps",
        "save_every",
        "log_every",
        "grad_accum",
        "lr",
        "grad_clip",
        "amp_init_scale",
        "hidden_dim",
        "condition_scale",
        "train_cfg_strength",
        "native_steps",
        "rescale_t",
        "delta_rms_ratio_cap",
        "occupancy_every",
    )
    bad = [name for name in positive if float(getattr(args, name)) <= 0]
    if bad:
        raise ValueError(f"arguments must be positive: {bad}")
    for name in (
        "weight_decay",
        "delta_norm_weight",
        "delta_scale",
        "raw_delta_excess_weight",
        "occupancy_weight",
        "occupancy_false_negative_weight",
        "occupancy_false_positive_weight",
        "occupancy_stock_recall_rank_weight",
        "occupancy_false_negative_margin",
        "occupancy_stock_recall_margin",
        "guidance_rescale",
        "control_probability",
        "control_rank_weight",
        "control_rank_margin",
    ):
        if not math.isfinite(float(getattr(args, name))) or float(getattr(args, name)) < 0:
            raise ValueError(f"{name} must be finite and non-negative")
    if float(args.control_probability) > 1.0:
        raise ValueError("control_probability must be <= 1")
    if int(args.grad_accum) % int(args.occupancy_every) != 0:
        raise ValueError("occupancy_every must divide grad_accum for auditable updates")
    occupancy_components = (
        float(args.occupancy_false_negative_weight),
        float(args.occupancy_false_positive_weight),
        float(args.occupancy_stock_recall_rank_weight),
    )
    if float(args.occupancy_weight) > 0.0 and not any(
        value > 0.0 for value in occupancy_components
    ):
        raise ValueError("positive occupancy_weight requires a non-zero component weight")
    if float(args.control_rank_weight) > 0 and float(args.control_probability) == 0:
        raise ValueError("control rank loss requires non-zero control_probability")
    if float(args.guidance_rescale) != 0.0:
        raise ValueError("post_cfg_bounded_v2 requires guidance_rescale=0")
    active = active_training_timesteps(args)
    if not active:
        raise ValueError("native schedule has no timestep inside train_cfg_interval")


def active_training_timesteps(args: argparse.Namespace) -> tuple[float, ...]:
    return tuple(
        value
        for value in native_ss_timestep_sequence(
            steps=int(args.native_steps), rescale_t=float(args.rescale_t)
        )[:-1]
        if native_ss_cfg_is_active(value, args.train_cfg_interval)
    )


def _distributed_true(value: bool, device: torch.device, world_size: int) -> bool:
    tensor = torch.tensor(int(value), device=device)
    if world_size > 1:
        dist.all_reduce(tensor, op=dist.ReduceOp.MIN)
    return bool(tensor.item())


def _checkpoint_fields(args: argparse.Namespace) -> tuple[str, ...]:
    return (
        "cache_manifest",
        "pretrained",
        "indices",
        "max_objects",
        "hidden_dim",
        "feature_source",
        "condition_scale",
        "sampling_mode",
        "t_schedule",
        "delta_norm_weight",
        "training_semantics",
        "guided_delta_policy",
        "support_interval_policy",
        "delta_bound_mode",
        "train_cfg_strength",
        "train_cfg_interval",
        "native_steps",
        "rescale_t",
        "guidance_rescale",
        "delta_scale",
        "delta_rms_ratio_cap",
        "raw_delta_excess_weight",
        "occupancy_weight",
        "occupancy_every",
        "occupancy_false_negative_weight",
        "occupancy_false_positive_weight",
        "occupancy_stock_recall_rank_weight",
        "occupancy_false_negative_margin",
        "occupancy_stock_recall_margin",
        "control_mode",
        "control_probability",
        "control_rank_weight",
        "control_rank_margin",
        "grad_accum",
    )


def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    step: int,
    micro_step: int,
    epoch: int,
    samples_into_epoch: int,
    args: argparse.Namespace,
    summary: dict[str, Any],
    history: list[dict[str, Any]],
) -> None:
    ensure_finite_trainable(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": NATIVE_SS_FLOW_VERSION,
            "step": int(step),
            "micro_step": int(micro_step),
            "epoch": int(epoch),
            "samples_into_epoch": int(samples_into_epoch),
            "model_trainable_state": trainable_state_dict(model),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "args": vars(args),
            "model_summary": summary,
            "history": history,
        },
        path,
    )


def validate_resume(
    checkpoint: dict[str, Any], args: argparse.Namespace, summary: dict[str, Any]
) -> None:
    validate_native_checkpoint(
        checkpoint, expected_format=NATIVE_SS_FLOW_VERSION, pretrained=args.pretrained
    )
    saved = checkpoint.get("args", {})
    mismatch = {
        name: (saved.get(name), getattr(args, name))
        for name in _checkpoint_fields(args)
        if str(saved.get(name)) != str(getattr(args, name))
    }
    if mismatch:
        raise ValueError(f"native SS resume protocol mismatch={mismatch}")
    for key in ("data_identity", "feature_identity", "block_count"):
        if checkpoint["model_summary"].get(key) != summary.get(key):
            raise ValueError(f"native SS resume {key} binding differs")


@torch.no_grad()
def stock_equivalence_audit(
    *,
    model: nn.Module,
    sampler: Any,
    sample: dict[str, Any],
    device: torch.device,
    fresh: bool,
    args: argparse.Namespace,
) -> dict[str, Any]:
    target = sample["target"].unsqueeze(0).to(device=device, dtype=torch.float32)
    condition = sample["stock_condition"].to(device=device)
    negative = torch.zeros_like(condition)
    t_value = torch.tensor(0.5, device=device)
    noise = torch.zeros_like(target)
    x_t, _ = sampler._get_model_gt(target, t_value, noise)
    t = torch.full((1,), 500.0, device=device)
    stock_positive = model.stock_prediction(x_t, t, condition)
    stock_negative = model.stock_prediction(x_t, t, negative)
    disabled, stock, _ = model.post_cfg_conditioned_prediction(
        x_t,
        t,
        condition,
        negative,
        None,
        stock_positive_velocity=stock_positive,
        stock_negative_velocity=stock_negative,
        cfg_strength=float(args.train_cfg_strength),
        cfg_active=True,
        support_active=True,
        condition_scale=0.0,
        delta_scale=float(args.delta_scale),
        delta_rms_ratio_cap=float(args.delta_rms_ratio_cap),
    )
    enabled, enabled_stock, stats = model.post_cfg_conditioned_prediction(
        x_t,
        t,
        condition,
        negative,
        sample,
        stock_positive_velocity=stock_positive,
        stock_negative_velocity=stock_negative,
        cfg_strength=float(args.train_cfg_strength),
        cfg_active=True,
        support_active=True,
        condition_scale=float(args.condition_scale),
        delta_scale=float(args.delta_scale),
        delta_rms_ratio_cap=float(args.delta_rms_ratio_cap),
    )
    report = {
        "disabled_max_abs": float((disabled - stock).abs().max().item()),
        "enabled_zero_init_max_abs": float((enabled - stock).abs().max().item()),
        "expect_zero_init": bool(fresh),
        "conditioned_block_count": int(stats["conditioned_block_count"].item()),
        "stock_reference_match_max_abs": float(
            (enabled_stock - stock).abs().max().item()
        ),
    }
    report["passed"] = bool(
        report["disabled_max_abs"] == 0.0
        and (not fresh or report["enabled_zero_init_max_abs"] == 0.0)
        and report["stock_reference_match_max_abs"] == 0.0
        and report["conditioned_block_count"] == len(model.flow_core.blocks)
    )
    if not report["passed"]:
        raise RuntimeError(f"native SS stock equivalence failed: {report}")
    return report


def main() -> None:
    args = parse_args()
    validate_args(args)
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        dist.init_process_group(backend="nccl", timeout=datetime.timedelta(hours=12))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    process_seed = int(args.seed) + rank * 100003
    random.seed(process_seed)
    np.random.seed(process_seed % (2**32 - 1))
    torch.manual_seed(process_seed)
    torch.cuda.manual_seed_all(process_seed)

    output_dir = Path(args.output_dir)
    if rank == 0:
        if args.resume:
            output_dir.mkdir(parents=True, exist_ok=True)
        else:
            output_dir.mkdir(parents=True, exist_ok=False)
    if world_size > 1:
        dist.barrier()
    dataset = PoseLiftingCacheDataset(args.cache_manifest, indices=args.indices)
    if int(args.max_objects) > 0:
        allowed_objects = set(
            sorted(
                {
                    str(row.get("object_uid", row["uid"]))
                    for row in dataset.rows
                }
            )[: int(args.max_objects)]
        )
        dataset.rows = [
            row
            for row in dataset.rows
            if str(row.get("object_uid", row["uid"])) in allowed_objects
        ]
        if not dataset.rows:
            raise RuntimeError("max_objects filtering selected no SS rows")
    feature_identity = validate_lifting_feature_metadata(
        visual_feature_dim=dataset.visual_feature_dim,
        feature_metadata=dataset.feature_metadata,
        feature_source=str(args.feature_source),
    )
    if args.sampling_mode == "object_balanced":
        distributed_sampler: Sampler[int] = ObjectBalancedDistributedSampler(
            dataset.rows,
            num_replicas=world_size,
            rank=rank,
            seed=int(args.seed),
        )
    else:
        distributed_sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=int(args.seed),
            drop_last=False,
        )
    loader = DataLoader(
        dataset,
        batch_size=1,
        sampler=distributed_sampler,
        num_workers=int(args.num_workers),
        collate_fn=collate_one,
        pin_memory=True,
    )
    sampler, model, decoder, model_summary, sampler_params = build_native_ss_components(
        pretrained=args.pretrained,
        hidden_dim=int(args.hidden_dim),
        feature_source=str(args.feature_source),
        gradient_checkpointing=bool(args.gradient_checkpointing),
        need_decoder=float(args.occupancy_weight) > 0.0,
        device=device,
    )
    if float(args.occupancy_weight) > 0.0 and decoder is None:
        raise RuntimeError("occupancy-aware Native SS training requires frozen decoder")
    data_identity = lifting_cache_identity(args.cache_manifest, rows=dataset.rows)
    model_summary.update(
        {
            "feature_identity": feature_identity,
            "data_identity": data_identity,
            "cache_manifest_sha256": sha256_file(args.cache_manifest),
            "dataset_size": len(dataset),
            "unique_object_count": len(
                {str(row.get("object_uid", row["uid"])) for row in dataset.rows}
            ),
            "sampler_params": sampler_params,
            "training": {
                "adapter_only": True,
                "training_semantics": str(args.training_semantics),
                "flow_matching_mse_on_deployed_post_cfg_velocity": 1.0,
                "delta_norm_weight": float(args.delta_norm_weight),
                "raw_delta_excess_weight": float(args.raw_delta_excess_weight),
                "frozen_decoder_occupancy": {
                    "version": NATIVE_SS_OCCUPANCY_OBJECTIVE_VERSION,
                    "weight": float(args.occupancy_weight),
                    "every_micro_steps": int(args.occupancy_every),
                    "frequency_compensation": int(args.occupancy_every),
                    "effective_weight_when_applied": float(args.occupancy_weight)
                    * int(args.occupancy_every),
                    "false_negative_weight": float(
                        args.occupancy_false_negative_weight
                    ),
                    "false_positive_weight": float(
                        args.occupancy_false_positive_weight
                    ),
                    "stock_recall_rank_weight": float(
                        args.occupancy_stock_recall_rank_weight
                    ),
                    "false_negative_margin": float(
                        args.occupancy_false_negative_margin
                    ),
                    "stock_recall_margin": float(
                        args.occupancy_stock_recall_margin
                    ),
                    "target": "repaired decoder-projected 64^3 target_coords",
                    "decoder_trainable": False,
                },
                "guided_delta_policy": str(args.guided_delta_policy),
                "delta_bound_mode": str(args.delta_bound_mode),
                "delta_scale": float(args.delta_scale),
                "delta_rms_ratio_cap": float(args.delta_rms_ratio_cap),
                "support_interval_policy": str(args.support_interval_policy),
                "deployment": {
                    "steps": int(args.native_steps),
                    "cfg_strength": float(args.train_cfg_strength),
                    "cfg_interval": list(args.train_cfg_interval),
                    "rescale_t": float(args.rescale_t),
                    "guidance_rescale": float(args.guidance_rescale),
                    "external_full_cfg_strength": 1.0,
                },
                "active_training_timesteps": list(active_training_timesteps(args)),
                "control_mode": str(args.control_mode),
                "control_probability": float(args.control_probability),
                "control_rank_weight": float(args.control_rank_weight),
                "control_rank_margin": float(args.control_rank_margin),
                "wrong_condition_role": "optional diagnostic/weak regularizer",
            },
        }
    )
    trainable = [value for value in model.parameters() if value.requires_grad]
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
    micro_step = 0
    start_epoch = 0
    resume_samples = 0
    history: list[dict[str, Any]] = []
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        validate_resume(checkpoint, args, model_summary)
        load_trainable_state_dict(model, checkpoint["model_trainable_state"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint.get("scaler", {}))
        start_step = int(checkpoint["step"])
        micro_step = int(checkpoint.get("micro_step", start_step * int(args.grad_accum)))
        start_epoch = int(checkpoint.get("epoch", 0))
        resume_samples = int(checkpoint.get("samples_into_epoch", 0))
        history = list(checkpoint.get("history", []))
    if start_step >= int(args.max_steps):
        raise ValueError("resume already reached max_steps")
    model_summary["stock_equivalence"] = stock_equivalence_audit(
        model=model,
        sampler=sampler,
        sample=dataset[0],
        device=device,
        fresh=not bool(args.resume),
        args=args,
    )
    if rank == 0:
        print(json.dumps(model_summary, indent=2), flush=True)

    wrapped: nn.Module = model
    if world_size > 1:
        wrapped = DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            find_unused_parameters=False,
        )
    model.train()
    if decoder is not None:
        decoder.eval()
    use_amp = args.amp_dtype != "none"
    amp_dtype = torch.float16 if args.amp_dtype == "fp16" else torch.bfloat16
    global_step = start_step
    epoch = start_epoch
    exposure: Counter[str] = Counter()
    wall_start = time.time()
    active_t_values = active_training_timesteps(args)
    optimizer.zero_grad(set_to_none=True)
    while global_step < int(args.max_steps):
        distributed_sampler.set_epoch(epoch)
        samples_into_epoch = 0
        for loader_position, sample in enumerate(loader):
            if epoch == start_epoch and loader_position < resume_samples:
                continue
            samples_into_epoch = loader_position + 1
            if global_step >= int(args.max_steps):
                break
            exposure[str(sample.get("object_uid", sample["uid"]))] += 1
            event_seed = process_seed * 1000003 + micro_step * 1013
            control_event = (
                float(args.control_probability) > 0
                and random.Random(event_seed + 41).random()
                < float(args.control_probability)
            )
            with torch.no_grad():
                target = sample["target"].unsqueeze(0).to(
                    device=device, dtype=torch.float32, non_blocking=True
                )
                condition = sample["stock_condition"].to(device=device, non_blocking=True)
                negative = torch.zeros_like(condition)
                generator = torch.Generator(device=device).manual_seed(event_seed + 17)
                noise = torch.randn(
                    target.shape,
                    generator=generator,
                    device=device,
                    dtype=target.dtype,
                )
                t_value_float = active_t_values[
                    random.Random(event_seed + 31).randrange(len(active_t_values))
                ]
                t_value = torch.tensor(t_value_float, device=device)
                x_t, gt_velocity = sampler._get_model_gt(target, t_value, noise)
                t = torch.full((1,), 1000.0 * t_value, device=device)
                stock_positive = model.stock_prediction(x_t, t, condition)
                stock_negative = model.stock_prediction(x_t, t, negative)
            sync_step = (micro_step + 1) % int(args.grad_accum) == 0
            sync_context = wrapped.no_sync() if world_size > 1 and not sync_step else nullcontext()
            with sync_context:
                with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                    prediction, stock, stats = wrapped(
                        x_t,
                        t,
                        condition,
                        sample,
                        stock_velocity=stock_positive,
                        negative_condition=negative,
                        stock_negative_velocity=stock_negative,
                        cfg_strength=float(args.train_cfg_strength),
                        cfg_active=True,
                        support_active=True,
                        condition_scale=float(args.condition_scale),
                        delta_scale=float(args.delta_scale),
                        delta_rms_ratio_cap=float(args.delta_rms_ratio_cap),
                    )
                    flow_loss = F.mse_loss(prediction.float(), gt_velocity.float())
                    stock_loss = F.mse_loss(stock.float(), gt_velocity.float()).detach()
                    stock_energy = stock.float().square().mean().detach().clamp_min(1.0e-6)
                    delta_norm = F.mse_loss(prediction.float(), stock.float()) / stock_energy
                    raw_delta_excess = F.relu(
                        stats["raw_flow_delta_ratio_per_batch"]
                        - float(args.delta_rms_ratio_cap)
                    ).square().mean()
                    control_loss = flow_loss.new_zeros(())
                    control_gain = flow_loss.new_zeros(())
                    if control_event:
                        control_context = (
                            nullcontext()
                            if float(args.control_rank_weight) > 0
                            else torch.no_grad()
                        )
                        with control_context:
                            control, _, _ = model.post_cfg_conditioned_prediction(
                                x_t,
                                t,
                                condition,
                                negative,
                                sample,
                                stock_positive_velocity=stock_positive,
                                stock_negative_velocity=stock_negative,
                                cfg_strength=float(args.train_cfg_strength),
                                cfg_active=True,
                                support_active=True,
                                condition_scale=float(args.condition_scale),
                                projection_mode=str(args.control_mode),
                                delta_scale=float(args.delta_scale),
                                delta_rms_ratio_cap=float(args.delta_rms_ratio_cap),
                            )
                        control_loss = F.mse_loss(control.float(), gt_velocity.float())
                        control_gain = control_loss - flow_loss
                        drop_cached_native_projection(
                            sample, mode=str(args.control_mode)
                        )
                    rank_loss = F.relu(
                        flow_loss.new_tensor(float(args.control_rank_margin)) - control_gain
                    ) if control_event else flow_loss.new_zeros(())
                    occupancy_applied = bool(
                        decoder is not None
                        and float(args.occupancy_weight) > 0.0
                        and (micro_step + 1) % int(args.occupancy_every) == 0
                    )
                    occupancy_terms = {
                        "false_negative_loss": flow_loss.new_zeros(()),
                        "false_positive_loss": flow_loss.new_zeros(()),
                        "stock_recall_rank_loss": flow_loss.new_zeros(()),
                    }
                    occupancy_loss = flow_loss.new_zeros(())
                    weighted_occupancy_loss = flow_loss.new_zeros(())
                    occupancy_metrics: dict[str, float] = {}
                    if occupancy_applied:
                        pred_x0 = sampler._pred_to_xstart(
                            x_t, t_value, prediction
                        )
                        stock_x0 = sampler._pred_to_xstart(x_t, t_value, stock)
                        decoder_dtype = next(decoder.parameters()).dtype
                        full_logits = decoder(
                            pred_x0.to(dtype=decoder_dtype)
                        ).float()
                        with torch.no_grad():
                            stock_logits = decoder(
                                stock_x0.to(dtype=decoder_dtype)
                            ).float()
                            occupancy_target = target_occupancy_grid(
                                sample["target_coords"], device=device
                            )
                        occupancy_terms = frozen_decoder_occupancy_objective(
                            full_logits,
                            occupancy_target,
                            stock_logits=stock_logits,
                            false_negative_margin=float(
                                args.occupancy_false_negative_margin
                            ),
                            stock_recall_margin=float(
                                args.occupancy_stock_recall_margin
                            ),
                        )
                        occupancy_loss = (
                            float(args.occupancy_false_negative_weight)
                            * occupancy_terms["false_negative_loss"]
                            + float(args.occupancy_false_positive_weight)
                            * occupancy_terms["false_positive_loss"]
                            + float(args.occupancy_stock_recall_rank_weight)
                            * occupancy_terms["stock_recall_rank_loss"]
                        )
                        weighted_occupancy_loss = (
                            float(args.occupancy_weight)
                            * int(args.occupancy_every)
                            * occupancy_loss
                        )
                        occupancy_metrics = objective_scalars(occupancy_terms)
                    loss = (
                        flow_loss
                        + float(args.delta_norm_weight) * delta_norm
                        + float(args.raw_delta_excess_weight) * raw_delta_excess
                        + weighted_occupancy_loss
                        + float(args.control_rank_weight) * rank_loss
                    )
                    scaled_loss = loss / float(args.grad_accum)
                scaler.scale(scaled_loss).backward()
            micro_step += 1
            if not sync_step:
                continue
            scaler.unscale_(optimizer)
            grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable, float(args.grad_clip)).item())
            block_grad_norms = block_projection_gradient_norms(model)
            finite = math.isfinite(grad_norm) and all(math.isfinite(value) for value in block_grad_norms)
            finite = finite and all(
                parameter.grad is None or bool(torch.isfinite(parameter.grad).all().item())
                for parameter in trainable
            )
            if not _distributed_true(finite, device, world_size):
                raise FloatingPointError("native SS gradient became non-finite")
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            row = {
                "step": global_step,
                "micro_step": micro_step,
                "uid": str(sample["uid"]),
                "object_uid": str(sample.get("object_uid", sample["uid"])),
                "t": float(t_value_float),
                "loss": float(loss.detach().item()),
                "flow_loss": float(flow_loss.detach().item()),
                "stock_loss": float(stock_loss.item()),
                "gain_vs_stock": float((stock_loss - flow_loss.detach()).item()),
                "delta_norm": float(delta_norm.detach().item()),
                "raw_delta_excess_loss": float(raw_delta_excess.detach().item()),
                "occupancy_applied": occupancy_applied,
                "occupancy_loss": float(occupancy_loss.detach().item()),
                "occupancy_frequency_compensation": (
                    int(args.occupancy_every) if occupancy_applied else 0
                ),
                "weighted_occupancy_loss": float(
                    weighted_occupancy_loss.detach().item()
                ),
                "occupancy": occupancy_metrics,
                "control_evaluated": bool(control_event),
                "control_loss": float(control_loss.detach().item()),
                "correct_over_control_advantage": float(control_gain.detach().item()),
                "grad_norm": grad_norm,
                "block_projection_grad_norm_min": min(block_grad_norms),
                "block_projection_grad_norm_max": max(block_grad_norms),
                "condition_token_rms": float(stats["condition_token_rms"].detach().item()),
                "flow_delta_rms": float(stats["flow_delta_rms"].detach().item()),
                "stock_velocity_rms": float(
                    stats["stock_velocity_rms"].detach().item()
                ),
                "raw_flow_delta_rms": float(
                    stats["raw_flow_delta_rms"].detach().item()
                ),
                "effective_flow_delta_rms": float(
                    stats["effective_flow_delta_rms"].detach().item()
                ),
                "raw_flow_delta_ratio": float(
                    stats["raw_flow_delta_ratio"].detach().item()
                ),
                "effective_flow_delta_ratio": float(
                    stats["effective_flow_delta_ratio"].detach().item()
                ),
                "delta_clip_scale": float(stats["delta_clip_scale"].detach().item()),
                "delta_clip_activated": bool(
                    stats["delta_clip_activated"].detach().item()
                ),
                "cfg_strength": float(args.train_cfg_strength),
                "cfg_active": True,
                "elapsed_seconds": time.time() - wall_start,
            }
            history.append(row)
            if rank == 0 and (global_step == 1 or global_step % int(args.log_every) == 0):
                print(f"[native_ss_train] {json.dumps(row)}", flush=True)
            if rank == 0 and global_step % int(args.save_every) == 0:
                save_checkpoint(
                    output_dir / "checkpoints" / f"step_{global_step:06d}.pt",
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    step=global_step,
                    micro_step=micro_step,
                    epoch=epoch,
                    samples_into_epoch=samples_into_epoch,
                    args=args,
                    summary=model_summary,
                    history=history,
                )
        epoch += 1
        resume_samples = 0
    if rank == 0:
        last_path = output_dir / "checkpoints" / f"step_{global_step:06d}.pt"
        if not last_path.is_file():
            save_checkpoint(
                last_path,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                step=global_step,
                micro_step=micro_step,
                epoch=epoch,
                samples_into_epoch=0,
                args=args,
                summary=model_summary,
                history=history,
            )
        report = {
            "format": NATIVE_SS_FLOW_VERSION,
            "completed": True,
            "step": global_step,
            "micro_step": micro_step,
            "model_summary": model_summary,
            "object_exposure": dict(sorted(exposure.items())),
            "history": history,
            "final_checkpoint": str(last_path),
            "finite": all(math.isfinite(float(row["loss"])) for row in history),
        }
        (output_dir / "report.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
