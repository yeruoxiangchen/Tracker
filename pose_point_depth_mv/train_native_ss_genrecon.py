#!/usr/bin/env python3
"""Train Native SS from Direct-SS LoRA and GenReCon flow-matching semantics."""

from __future__ import annotations

import argparse
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
from torch.utils.data import DataLoader, Subset

from ar_ss_flow.local_pose_lifting_flow import PoseLiftingCacheDataset, collate_one
from pose_point_depth_mv.native_ss_genrecon import (
    NATIVE_SS_GENRECON_CFG,
    NATIVE_SS_GENRECON_PROJECTION,
    NATIVE_SS_GENRECON_TRAINING,
    NATIVE_SS_GENRECON_VERSION,
    build_native_ss_genrecon_components,
    canonical_json_sha256,
    load_trainable_state_dict,
    optimizer_parameter_groups,
    sha256_file,
    trainable_state_dict,
    validate_genrecon_cache_contract,
    validate_native_ss_genrecon_checkpoint,
)
from pose_point_depth_mv.train_direct_flow import ObjectBalancedDistributedSampler
from reconvggt_ar_adapter_a.train_pointpose_ss_lora import (
    finite_tree,
    gradients_finite,
    optimizer_state_finite,
    parameters_finite,
)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache_manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--pretrained", default="Stable-X/trellis-vggt-v0-2")
    parser.add_argument("--indices", default="all")
    parser.add_argument("--max_objects", type=int, default=0)
    parser.add_argument("--resume", default="")
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument("--save_every", type=int, default=50)
    parser.add_argument("--log_every", type=int, default=5)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--condition_channels", type=int, default=1024)
    parser.add_argument("--new_lr", type=float, default=1.0e-4)
    parser.add_argument("--lora_lr", type=float, default=3.0e-5)
    parser.add_argument("--new_weight_decay", type=float, default=0.01)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.95)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--warmup_steps", type=int, default=-1)
    parser.add_argument("--warmup_ratio", type=float, default=0.02)
    parser.add_argument("--ema_decay", type=float, default=0.9995)
    parser.add_argument("--amp_dtype", choices=("bf16", "fp16", "none"), default="bf16")
    parser.add_argument("--amp_init_scale", type=float, default=8192.0)
    parser.add_argument("--p_uncond", type=float, default=0.1)
    parser.add_argument("--t_logit_mean", type=float, default=1.0)
    parser.add_argument("--t_logit_std", type=float, default=1.0)
    parser.add_argument("--min_condition_views", type=int, default=1)
    parser.add_argument("--max_condition_views", type=int, default=16)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    positive = (
        "max_steps",
        "save_every",
        "log_every",
        "grad_accum",
        "lora_rank",
        "lora_alpha",
        "condition_channels",
        "new_lr",
        "lora_lr",
        "grad_clip",
        "t_logit_std",
        "min_condition_views",
        "max_condition_views",
    )
    bad = [name for name in positive if float(getattr(args, name)) <= 0]
    if bad:
        raise ValueError(f"arguments must be positive: {bad}")
    if not 0.0 <= float(args.p_uncond) < 1.0:
        raise ValueError("p_uncond must be in [0,1)")
    if int(args.min_condition_views) > int(args.max_condition_views):
        raise ValueError("min_condition_views exceeds max_condition_views")
    if int(args.condition_channels) != 1024:
        raise ValueError("Native SS v2 binds GenReCon/DINO condition_channels=1024")
    if int(args.max_objects) < 0 or int(args.num_workers) < 0:
        raise ValueError("max_objects/num_workers must be non-negative")
    if int(args.warmup_steps) < -1:
        raise ValueError("warmup_steps must be -1 or non-negative")
    if not 0.0 <= float(args.warmup_ratio) <= 1.0:
        raise ValueError("warmup_ratio must be in [0,1]")
    if not 0.0 < float(args.ema_decay) < 1.0:
        raise ValueError("ema_decay must be in (0,1)")


class NativeSSGenreconTrainingForward(nn.Module):
    """Expose the adapted path through ``forward`` so DDP owns synchronization."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        condition: Any,
        sample: dict[str, Any] | None,
        *,
        view_indices: torch.Tensor | None,
        stock_velocity: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        prediction, stats = self.model.adapted_prediction(
            x,
            t,
            condition,
            sample,
            view_indices=view_indices,
            stock_velocity=stock_velocity,
        )
        if sample is None:
            # The unconditional branch bypasses the learned view aggregator.
            # Keep every trainable parameter in the DDP graph with exact-zero
            # gradients so conditional and unconditional ranks can coexist.
            anchor = prediction.new_zeros((), dtype=torch.float32)
            for parameter in self.model.aggregator.parameters():
                anchor = anchor + parameter.reshape(-1)[0].float() * 0.0
            prediction = prediction + anchor.to(dtype=prediction.dtype)
        return prediction, stats


def distributed_true(
    value: bool, *, device: torch.device, world_size: int
) -> bool:
    result = torch.tensor(int(value), device=device)
    if int(world_size) > 1:
        dist.all_reduce(result, op=dist.ReduceOp.MIN)
    return bool(result.item())


def distributed_means(
    values: list[float], *, device: torch.device, world_size: int
) -> list[float]:
    result = torch.tensor(values, device=device, dtype=torch.float64)
    if int(world_size) > 1:
        dist.all_reduce(result, op=dist.ReduceOp.SUM)
        result.div_(float(world_size))
    return [float(value) for value in result.cpu().tolist()]


def selected_dataset(
    dataset: PoseLiftingCacheDataset, max_objects: int
) -> tuple[Subset, list[dict[str, Any]], list[str]]:
    object_uids = sorted(
        {str(row.get("object_uid", row["uid"])) for row in dataset.rows}
    )
    if int(max_objects) > 0:
        object_uids = object_uids[: int(max_objects)]
    admitted = set(object_uids)
    indices = [
        index
        for index, row in enumerate(dataset.rows)
        if str(row.get("object_uid", row["uid"])) in admitted
    ]
    rows = [dataset.rows[index] for index in indices]
    if not rows:
        raise RuntimeError("Native SS training selected no samples")
    return Subset(dataset, indices), rows, object_uids


def sample_t(args: argparse.Namespace, device: torch.device) -> float:
    value = torch.sigmoid(
        torch.randn((), device=device) * float(args.t_logit_std)
        + float(args.t_logit_mean)
    )
    return float(value.item())


def sample_views(
    sample: dict[str, Any], args: argparse.Namespace, device: torch.device
) -> torch.Tensor:
    views = int(sample["visual_patch_features"].shape[0])
    upper = min(views, int(args.max_condition_views))
    lower = min(upper, int(args.min_condition_views))
    count = int(torch.randint(lower, upper + 1, (), device=device).item())
    return torch.randperm(views, device=device)[:count]


def gradient_norms(model: torch.nn.Module) -> dict[str, float]:
    groups = {"lora": 0.0, "condition": 0.0}
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        key = "lora" if "lora_" in name else "condition"
        groups[key] += float(parameter.grad.detach().float().square().sum().item())
    return {name: value**0.5 for name, value in groups.items()}


def resolve_warmup_steps(args: argparse.Namespace) -> int:
    if int(args.warmup_steps) >= 0:
        result = int(args.warmup_steps)
    else:
        result = int(math.ceil(float(args.warmup_ratio) * int(args.max_steps)))
    if result > int(args.max_steps):
        raise ValueError("warmup_steps exceeds max_steps")
    return result


def warmup_factor(update_step: int, warmup_steps: int) -> float:
    if int(update_step) <= 0:
        raise ValueError("update_step must be positive")
    if int(warmup_steps) <= 0:
        return 1.0
    return min(1.0, float(update_step) / float(warmup_steps))


def ema_ramp_decay(target_decay: float, update_step: int) -> float:
    if int(update_step) <= 0:
        raise ValueError("EMA update_step must be positive")
    ramp = float(1 + int(update_step)) / float(10 + int(update_step))
    return min(float(target_decay), ramp)


def trainable_named_parameters(model) -> dict[str, torch.nn.Parameter]:
    return {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def initialize_ema_state(model) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().float().clone()
        for name, parameter in trainable_named_parameters(model).items()
    }


@torch.no_grad()
def update_ema_state(
    ema_state: dict[str, torch.Tensor], model, *, decay: float
) -> None:
    named = trainable_named_parameters(model)
    if set(ema_state) != set(named):
        raise ValueError("EMA/trainable parameter names differ")
    for name, parameter in named.items():
        ema = ema_state[name]
        if ema.shape != parameter.shape or ema.device != parameter.device:
            raise ValueError(f"EMA tensor contract mismatch for {name}")
        ema.mul_(float(decay)).add_(parameter.detach().float(), alpha=1.0 - float(decay))


def load_ema_state(model, state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    named = trainable_named_parameters(model)
    if set(state) != set(named):
        raise ValueError("checkpoint EMA/trainable parameter names differ")
    return {
        name: value.to(device=named[name].device, dtype=torch.float32).clone()
        for name, value in state.items()
    }


def ema_state_finite(state: dict[str, torch.Tensor]) -> bool:
    return bool(state) and all(
        bool(torch.isfinite(value).all().item()) for value in state.values()
    )


def save_checkpoint(
    path: Path,
    *,
    model,
    optimizer: torch.optim.Optimizer,
    scaler,
    step: int,
    micro_step: int,
    args: argparse.Namespace,
    model_summary: dict[str, Any],
    data_identity: dict[str, Any],
    history: list[dict[str, Any]],
    ema_state: dict[str, torch.Tensor],
    ema_last_decay: float,
) -> None:
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters_finite(trainable):
        raise RuntimeError(f"refusing to save non-finite Native SS parameters: {path}")
    if not optimizer_state_finite(optimizer) or not finite_tree(scaler.state_dict()):
        raise RuntimeError(f"refusing to save non-finite optimizer/scaler: {path}")
    if not ema_state_finite(ema_state):
        raise RuntimeError(f"refusing to save non-finite Native SS EMA: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": NATIVE_SS_GENRECON_VERSION,
            "step": int(step),
            "micro_step": int(micro_step),
            "model_trainable_state": trainable_state_dict(model),
            "ema_trainable_state": {
                name: value.detach().cpu().clone() for name, value in ema_state.items()
            },
            "ema": {
                "target_decay": float(args.ema_decay),
                "ramp": "min(target_decay,(1+update)/(10+update))",
                "updates": int(step),
                "last_decay": float(ema_last_decay),
            },
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "args": vars(args),
            "model_summary": model_summary,
            "data_identity": data_identity,
            "history": history,
            "rng": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state(),
            },
        },
        path,
    )


@torch.no_grad()
def initial_stock_audit(model, sampler, sample, device) -> dict[str, Any]:
    target = sample["target"].to(device=device, dtype=torch.float32)[None]
    noise = torch.zeros_like(target)
    t_value = 0.5
    x_t, _ = sampler._get_model_gt(target, t_value, noise)
    t = torch.full((1,), 500.0, device=device)
    positive = sample["stock_condition"].to(device=device)
    negative = torch.zeros_like(positive)
    stock_positive = model.stock_prediction(x_t, t, positive)
    stock_negative = model.stock_prediction(x_t, t, negative)
    full_positive, positive_stats = model.adapted_prediction(
        x_t, t, positive, sample, stock_velocity=stock_positive
    )
    full_negative, negative_stats = model.adapted_prediction(
        x_t, t, negative, None, stock_velocity=stock_negative
    )
    report = {
        "conditional_max_abs": float(
            (full_positive.float() - stock_positive.float()).abs().amax().item()
        ),
        "unconditional_max_abs": float(
            (full_negative.float() - stock_negative.float()).abs().amax().item()
        ),
        "conditional_blocks": len(model.block_condition.projections),
        "conditional_supported_fraction": float(
            positive_stats["supported_fraction"].item()
        ),
        "unconditional_condition_present": float(
            negative_stats["condition_present"].item()
        ),
    }
    report["passed"] = bool(
        report["conditional_max_abs"] == 0.0
        and report["unconditional_max_abs"] == 0.0
        and report["conditional_blocks"] == len(model.flow_core.blocks)
        and report["conditional_supported_fraction"] > 0.0
        and report["unconditional_condition_present"] == 0.0
    )
    if not report["passed"]:
        raise RuntimeError(f"initial Native SS Stock audit failed: {report}")
    return report


def main() -> None:
    args = make_parser().parse_args()
    validate_args(args)
    resolved_warmup_steps = resolve_warmup_steps(args)
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
        output_dir.mkdir(parents=True, exist_ok=bool(args.resume))
    if world_size > 1:
        dist.barrier()

    dataset = PoseLiftingCacheDataset(args.cache_manifest, indices=args.indices)
    cache_contract = validate_genrecon_cache_contract(dataset)
    subset, selected_rows, object_uids = selected_dataset(dataset, int(args.max_objects))
    sampler_rows = selected_rows
    object_sampler = ObjectBalancedDistributedSampler(
        sampler_rows,
        num_replicas=world_size,
        rank=rank,
        seed=int(args.seed),
    )
    loader = DataLoader(
        subset,
        batch_size=1,
        sampler=object_sampler,
        num_workers=int(args.num_workers),
        collate_fn=collate_one,
        pin_memory=True,
    )
    data_identity = {
        "manifest": str(Path(args.cache_manifest).resolve()),
        "manifest_sha256": sha256_file(args.cache_manifest),
        "config_hash": dataset.config_hash,
        "sample_count": len(selected_rows),
        "object_count": len(object_uids),
        "object_uids": object_uids,
        "object_uid_hash": canonical_json_sha256(object_uids),
        "feature_contract": cache_contract,
    }
    model_sampler, model, _, model_summary, sampler_params = (
        build_native_ss_genrecon_components(
            pretrained=args.pretrained,
            lora_rank=int(args.lora_rank),
            lora_alpha=int(args.lora_alpha),
            condition_channels=int(args.condition_channels),
            gradient_checkpointing=bool(args.gradient_checkpointing),
            need_decoder=False,
            device=device,
        )
    )
    model_summary = {
        **model_summary,
        "data_identity": data_identity,
        "t_schedule": {
            "name": "logit_normal",
            "mean": float(args.t_logit_mean),
            "std": float(args.t_logit_std),
        },
        "p_uncond": float(args.p_uncond),
        "view_augmentation": {
            "random_subset": True,
            "min": int(args.min_condition_views),
            "max": int(args.max_condition_views),
        },
        "sampler_defaults": sampler_params,
        "optimization": {
            "warmup_steps": int(resolved_warmup_steps),
            "warmup_ratio": float(args.warmup_ratio),
            "ema_target_decay": float(args.ema_decay),
            "ema_ramp": "min(target_decay,(1+update)/(10+update))",
            "distributed": {
                "world_size": int(world_size),
                "per_rank_batch_size": 1,
                "per_rank_grad_accum": int(args.grad_accum),
                "global_effective_batch": int(world_size) * int(args.grad_accum),
            },
        },
    }
    initial_audit = initial_stock_audit(model, model_sampler, subset[0], device)
    training_forward = NativeSSGenreconTrainingForward(model)
    wrapped: nn.Module = training_forward
    if world_size > 1:
        wrapped = DistributedDataParallel(
            training_forward,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            find_unused_parameters=False,
        )
    optimizer = torch.optim.AdamW(
        optimizer_parameter_groups(
            model,
            new_lr=float(args.new_lr),
            lora_lr=float(args.lora_lr),
            new_weight_decay=float(args.new_weight_decay),
        ),
        betas=(float(args.adam_beta1), float(args.adam_beta2)),
        eps=1.0e-8,
    )
    for group in optimizer.param_groups:
        group["base_lr"] = float(group["lr"])
    use_amp = args.amp_dtype != "none"
    amp_dtype = torch.float16 if args.amp_dtype == "fp16" else torch.bfloat16
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=args.amp_dtype == "fp16",
        init_scale=float(args.amp_init_scale),
    )
    step = 0
    micro_step = 0
    history: list[dict[str, Any]] = []
    ema_state = initialize_ema_state(model)
    ema_last_decay = 0.0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        validate_native_ss_genrecon_checkpoint(checkpoint, pretrained=args.pretrained)
        if checkpoint.get("data_identity") != data_identity:
            raise RuntimeError("resume data identity differs")
        saved_args = checkpoint["args"]
        bound_fields = (
            "lora_rank",
            "lora_alpha",
            "condition_channels",
            "new_lr",
            "lora_lr",
            "new_weight_decay",
            "p_uncond",
            "t_logit_mean",
            "t_logit_std",
            "min_condition_views",
            "max_condition_views",
            "grad_accum",
            "max_steps",
            "warmup_steps",
            "warmup_ratio",
            "ema_decay",
        )
        mismatch = {
            key: (saved_args.get(key), getattr(args, key))
            for key in bound_fields
            if saved_args.get(key) != getattr(args, key)
        }
        if mismatch:
            raise ValueError(f"resume argument mismatch={mismatch}")
        saved_distributed = (
            checkpoint.get("model_summary", {})
            .get("optimization", {})
            .get("distributed", {})
        )
        if int(saved_distributed.get("world_size", 1)) != world_size:
            raise ValueError(
                "resume world_size differs: "
                f"{saved_distributed.get('world_size', 1)} != {world_size}"
            )
        load_trainable_state_dict(model, checkpoint["model_trainable_state"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint.get("scaler", {}))
        ema_state = load_ema_state(model, checkpoint["ema_trainable_state"])
        ema_last_decay = float(checkpoint.get("ema", {}).get("last_decay", 0.0))
        step = int(checkpoint["step"])
        micro_step = int(checkpoint.get("micro_step", step * int(args.grad_accum)))
        history = list(checkpoint.get("history", []))
        rng = checkpoint.get("rng", {})
        if rng:
            random.setstate(rng["python"])
            np.random.set_state(rng["numpy"])
            torch.set_rng_state(rng["torch"])
            torch.cuda.set_rng_state(rng["cuda"])

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer.zero_grad(set_to_none=True)
    model.train()
    epoch = 0
    started = time.time()
    accumulator: list[dict[str, float | bool]] = []
    while step < int(args.max_steps):
        object_sampler.set_epoch(epoch)
        for sample in loader:
            if step >= int(args.max_steps):
                break
            target = sample["target"].to(device=device, dtype=torch.float32)[None]
            positive = sample["stock_condition"].to(device=device)
            negative = torch.zeros_like(positive)
            noise = torch.randn_like(target)
            t_value = sample_t(args, device)
            x_t, gt_velocity = model_sampler._get_model_gt(target, t_value, noise)
            t = torch.full((1,), 1000.0 * t_value, device=device)
            unconditional = random.random() < float(args.p_uncond)
            condition = negative if unconditional else positive
            condition_sample = None if unconditional else sample
            view_indices = None if unconditional else sample_views(sample, args, device)
            with torch.no_grad():
                stock = model.stock_prediction(x_t, t, condition)
                stock_loss = F.mse_loss(stock.float(), gt_velocity.float())
            sync_step = (micro_step + 1) % int(args.grad_accum) == 0
            sync_context = (
                wrapped.no_sync() if world_size > 1 and not sync_step else nullcontext()
            )
            with sync_context:
                with torch.autocast(
                    device_type="cuda", dtype=amp_dtype, enabled=use_amp
                ):
                    prediction, stats = wrapped(
                        x_t,
                        t,
                        condition,
                        condition_sample,
                        view_indices=view_indices,
                        stock_velocity=stock,
                    )
                    flow_loss = F.mse_loss(prediction.float(), gt_velocity.float())
                    loss = flow_loss / float(args.grad_accum)
                scaler.scale(loss).backward()
            micro_step += 1
            accumulator.append(
                {
                    "flow_loss": float(flow_loss.detach().item()),
                    "stock_loss": float(stock_loss.detach().item()),
                    "gain": float((stock_loss - flow_loss.detach()).item()),
                    "t": float(t_value),
                    "unconditional": bool(unconditional),
                    "views": float(0 if view_indices is None else len(view_indices)),
                    "flow_delta_rms": float(stats["flow_delta_rms"].detach().item()),
                    "supported_fraction": float(stats["supported_fraction"].detach().item()),
                }
            )
            if not sync_step:
                continue
            scaler.unscale_(optimizer)
            if not distributed_true(
                gradients_finite(trainable), device=device, world_size=world_size
            ):
                raise RuntimeError("Native SS gradients became non-finite")
            grad_norm = float(
                torch.nn.utils.clip_grad_norm_(trainable, float(args.grad_clip)).item()
            )
            group_gradients = gradient_norms(model)
            next_step = step + 1
            lr_scale = warmup_factor(next_step, resolved_warmup_steps)
            for group in optimizer.param_groups:
                group["lr"] = float(group["base_lr"]) * lr_scale
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            optimizer_finite = parameters_finite(trainable) and optimizer_state_finite(
                optimizer
            )
            if not distributed_true(
                optimizer_finite, device=device, world_size=world_size
            ):
                raise RuntimeError("Native SS optimizer produced non-finite state")
            ema_last_decay = ema_ramp_decay(float(args.ema_decay), next_step)
            update_ema_state(ema_state, model, decay=ema_last_decay)
            step = next_step
            metric_means = distributed_means(
                [
                    float(np.mean([float(item["flow_loss"]) for item in accumulator])),
                    float(np.mean([float(item["stock_loss"]) for item in accumulator])),
                    float(np.mean([float(item["gain"]) for item in accumulator])),
                    float(np.mean([float(item["t"]) for item in accumulator])),
                    float(
                        np.mean(
                            [float(bool(item["unconditional"])) for item in accumulator]
                        )
                    ),
                    float(np.mean([float(item["views"]) for item in accumulator])),
                    float(
                        np.mean(
                            [float(item["flow_delta_rms"]) for item in accumulator]
                        )
                    ),
                    float(
                        np.mean(
                            [float(item["supported_fraction"]) for item in accumulator]
                        )
                    ),
                ],
                device=device,
                world_size=world_size,
            )
            row = {
                "step": step,
                "micro_step": micro_step,
                "global_micro_samples": step * int(args.grad_accum) * int(world_size),
                "flow_loss": metric_means[0],
                "stock_loss": metric_means[1],
                "gain": metric_means[2],
                "t_mean": metric_means[3],
                "unconditional_fraction": metric_means[4],
                "view_count_mean": metric_means[5],
                "flow_delta_rms": metric_means[6],
                "supported_fraction": metric_means[7],
                "gradient_norm_before_clip": grad_norm,
                "gradient_norms": group_gradients,
                "learning_rate_scale": lr_scale,
                "learning_rates": {
                    str(group.get("name", index)): float(group["lr"])
                    for index, group in enumerate(optimizer.param_groups)
                },
                "ema_decay": ema_last_decay,
            }
            history.append(row)
            accumulator.clear()
            if rank == 0 and (step == 1 or step % int(args.log_every) == 0):
                print(f"[native_ss_genrecon] {json.dumps(row, ensure_ascii=False)}", flush=True)
            if rank == 0 and (
                step % int(args.save_every) == 0 or step == int(args.max_steps)
            ):
                checkpoint_path = output_dir / "checkpoints" / f"step_{step:06d}.pt"
                save_checkpoint(
                    checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    step=step,
                    micro_step=micro_step,
                    args=args,
                    model_summary=model_summary,
                    data_identity=data_identity,
                    history=history,
                    ema_state=ema_state,
                    ema_last_decay=ema_last_decay,
                )
                save_checkpoint(
                    output_dir / "checkpoints" / "last.pt",
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    step=step,
                    micro_step=micro_step,
                    args=args,
                    model_summary=model_summary,
                    data_identity=data_identity,
                    history=history,
                    ema_state=ema_state,
                    ema_last_decay=ema_last_decay,
                )
        epoch += 1
    report_passed = distributed_true(
        bool(
            step == int(args.max_steps)
            and parameters_finite(trainable)
            and optimizer_state_finite(optimizer)
            and ema_state_finite(ema_state)
        ),
        device=device,
        world_size=world_size,
    )
    report = {
        "format": NATIVE_SS_GENRECON_VERSION,
        "completed": True,
        "passed": report_passed,
        "step": step,
        "micro_step": micro_step,
        "global_micro_samples": step * int(args.grad_accum) * int(world_size),
        "elapsed_seconds": time.time() - started,
        "model_summary": model_summary,
        "data_identity": data_identity,
        "initial_stock_audit": initial_audit,
        "history": history,
        "checkpoint": str(output_dir / "checkpoints" / "last.pt"),
        "evaluation_weights": "ema",
        "ema": {
            "target_decay": float(args.ema_decay),
            "last_decay": float(ema_last_decay),
            "updates": int(step),
        },
        "explicitly_absent": [
            "Direct-SLAT dependency",
            "post-CFG residual cap",
            "decoder occupancy loss",
            "CFG-active-only timestep training",
        ],
    }
    if rank == 0:
        (output_dir / "report.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        print(
            json.dumps({"passed": report["passed"], "output": str(output_dir)}),
            flush=True,
        )
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()
    raise SystemExit(0 if report_passed else 2)


if __name__ == "__main__":
    main()
