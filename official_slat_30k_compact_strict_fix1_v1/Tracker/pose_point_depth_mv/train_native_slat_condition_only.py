#!/usr/bin/env python3
"""Train LoRA-free Native-SLat using only posed-DINO 3D conditioning."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import datetime
import json
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
from torch.utils.data import DataLoader

from pose_point_depth_mv.evaluate_native_ss_stock_slat_mesh import load_ss_evidence
from pose_point_depth_mv.native_3d_condition import (
    NativeConditionSLatDataset,
    collate_native_one,
)
from pose_point_depth_mv.native_slat_condition_only import (
    NATIVE_SLAT_CONDITION_ONLY_VERSION,
    NativeSLatConditionOnlyFlow,
    build_native_slat_condition_only_components,
    canonical_json_sha256,
    condition_only_parameter_group,
    load_stock_slat_freeze,
    load_trainable_state_dict,
    sha256_file,
    trainable_state_dict,
    validate_native_slat_condition_only_checkpoint,
)
from pose_point_depth_mv.native_slat_decoder_geometry import (
    DECODER_GEOMETRY_LOSS_VERSION,
    decoder_geometry_objective,
    stock_relative_trust_loss,
)
from pose_point_depth_mv.train_direct_flow import ObjectBalancedDistributedSampler
from pose_point_depth_mv.train_direct_slat_flow import normalized_target, to_device_tree
from pose_point_depth_mv.train_native_slat_genrecon import (
    distributed_means,
    distributed_true,
    ema_ramp_decay,
    ema_state_finite,
    initialize_ema_state,
    load_ema_state,
    resolve_warmup_steps,
    sample_t,
    sample_view_indices,
    select_context_views,
    update_ema_state,
    upstream_binding,
    validate_decoder_audit,
    warmup_factor,
)
from reconvggt_ar_adapter_a.train_pointpose_ss_lora import (
    finite_tree,
    gradients_finite,
    optimizer_state_finite,
    parameters_finite,
)
from trellis.modules import sparse as sp


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache_manifest", required=True)
    parser.add_argument("--lifting_cache_manifest", required=True)
    parser.add_argument("--target_decoder_audit", required=True)
    parser.add_argument("--native_ss_report", required=True)
    parser.add_argument("--stock_slat_freeze", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--pretrained", default="Stable-X/trellis-vggt-v0-2")
    parser.add_argument("--indices", default="all")
    parser.add_argument("--resume", default="")
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--save_every", type=int, default=200)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--condition_channels", type=int, default=1024)
    parser.add_argument("--condition_lr", type=float, default=1.0e-4)
    parser.add_argument("--condition_weight_decay", type=float, default=0.01)
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
    parser.add_argument(
        "--t_schedule",
        choices=("logit_normal", "uniform", "mixed"),
        default="logit_normal",
    )
    parser.add_argument("--t_uniform_probability", type=float, default=0.75)
    parser.add_argument("--separate_t_rng", action="store_true")
    parser.add_argument("--max_objects", type=int, default=0)
    parser.add_argument("--decoder_geometry_weight", type=float, default=0.0)
    parser.add_argument("--stock_flow_trust_weight", type=float, default=0.0)
    parser.add_argument("--stock_flow_required_improvement", type=float, default=0.01)
    parser.add_argument("--stock_trust_weight", type=float, default=0.0)
    parser.add_argument("--stock_required_improvement", type=float, default=0.01)
    parser.add_argument("--geometry_event_probability", type=float, default=0.125)
    parser.add_argument("--geometry_t_max", type=float, default=0.5)
    parser.add_argument("--min_condition_views", type=int, default=1)
    parser.add_argument("--max_condition_views", type=int, default=16)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--verify_cache_hashes", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    positive = (
        "max_steps",
        "save_every",
        "log_every",
        "grad_accum",
        "condition_channels",
        "condition_lr",
        "grad_clip",
        "t_logit_std",
        "min_condition_views",
        "max_condition_views",
    )
    bad = [name for name in positive if float(getattr(args, name)) <= 0]
    if bad:
        raise ValueError(f"arguments must be positive: {bad}")
    if int(args.condition_channels) != 1024:
        raise ValueError("condition-only Native-SLat binds DINO channels=1024")
    if not 0.0 <= float(args.p_uncond) < 1.0:
        raise ValueError("p_uncond must be in [0,1)")
    if int(args.min_condition_views) > int(args.max_condition_views):
        raise ValueError("min_condition_views exceeds max_condition_views")
    if int(args.warmup_steps) < -1 or not 0.0 <= float(args.warmup_ratio) <= 1.0:
        raise ValueError("invalid warmup configuration")
    if not 0.0 < float(args.ema_decay) < 1.0:
        raise ValueError("ema_decay must be in (0,1)")
    if float(args.condition_weight_decay) < 0.0:
        raise ValueError("condition_weight_decay must be non-negative")
    if int(args.max_objects) < 0:
        raise ValueError("max_objects must be non-negative")
    probabilities = ("t_uniform_probability", "geometry_event_probability")
    if any(not 0.0 <= float(getattr(args, name)) <= 1.0 for name in probabilities):
        raise ValueError(f"probabilities must lie in [0,1]: {probabilities}")
    non_negative = (
        "decoder_geometry_weight",
        "stock_flow_trust_weight",
        "stock_flow_required_improvement",
        "stock_trust_weight",
        "stock_required_improvement",
    )
    if any(float(getattr(args, name)) < 0.0 for name in non_negative):
        raise ValueError(f"geometry loss weights must be non-negative: {non_negative}")
    if not 0.0 < float(args.geometry_t_max) <= 1.0:
        raise ValueError("geometry_t_max must lie in (0,1]")
    if float(args.stock_trust_weight) > 0.0 and float(
        args.decoder_geometry_weight
    ) <= 0.0:
        raise ValueError("Stock trust requires a positive decoder geometry weight")


def sample_condition_only_t(
    args: argparse.Namespace,
    device: torch.device,
    *,
    seed: int | None = None,
    rank: int = 0,
    micro_step: int = 0,
) -> float:
    generator = None
    if bool(getattr(args, "separate_t_rng", False)):
        if seed is None:
            raise ValueError("separate_t_rng requires an explicit seed")
        generator = torch.Generator(device=device)
        mixed_seed = (
            (int(seed) + 0x9E3779B9) * 0x85EBCA6B
            + (int(rank) + 1) * 0xC2B2AE35
            + (int(micro_step) + 1) * 0x27D4EB2F
        ) & 0x7FFFFFFFFFFFFFFF
        generator.manual_seed(mixed_seed)

    def uniform() -> float:
        return float(torch.rand((), device=device, generator=generator).item())

    schedule = str(args.t_schedule)
    if schedule == "uniform":
        return uniform()
    if schedule == "mixed" and uniform() < float(
        args.t_uniform_probability
    ):
        return uniform()
    if generator is None:
        return sample_t(args, device)
    value = torch.sigmoid(
        torch.randn((), device=device, generator=generator)
        * float(args.t_logit_std)
        + float(args.t_logit_mean)
    )
    return float(value.item())


def deterministic_geometry_event(
    *, seed: int, rank: int, micro_step: int, probability: float
) -> bool:
    """Sample an auxiliary event without perturbing the base training RNG."""

    if probability <= 0.0:
        return False
    if probability >= 1.0:
        return True
    # Integer mixing is stable across Python versions and resume boundaries.
    value = (
        (int(seed) + 0x9E3779B9) * 0x85EBCA6B
        + (int(rank) + 1) * 0xC2B2AE35
        + (int(micro_step) + 1) * 0x27D4EB2F
    ) & 0xFFFFFFFF
    return (value + 0.5) / float(2**32) < float(probability)


def distributed_timestep_bins(
    rows: list[dict[str, float | bool]],
    *,
    device: torch.device,
    world_size: int,
) -> list[dict[str, float]]:
    edges = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0000001)
    values = torch.zeros((5, 4), device=device, dtype=torch.float64)
    for row in rows:
        t_value = float(row["t"])
        index = min(4, max(0, int(t_value * 5.0)))
        values[index, 0] += 1.0
        values[index, 1] += float(row["flow_loss"])
        values[index, 2] += float(row["stock_loss"])
        values[index, 3] += float(row["gain"])
    if int(world_size) > 1:
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
    output = []
    for index in range(5):
        count = float(values[index, 0].item())
        output.append(
            {
                "lo": edges[index],
                "hi": edges[index + 1],
                "count": count,
                "flow_loss": float(values[index, 1].item() / max(count, 1.0)),
                "stock_loss": float(values[index, 2].item() / max(count, 1.0)),
                "gain": float(values[index, 3].item() / max(count, 1.0)),
            }
        )
    return output


def distributed_geometry_summary(
    rows: list[dict[str, float | bool]],
    *,
    device: torch.device,
    world_size: int,
) -> dict[str, float]:
    names = (
        "field_loss",
        "stock_field_loss",
        "trust_loss",
        "relative_field_delta",
        "sdf_loss",
        "sign_loss",
        "deform_loss",
        "topology_loss",
        "boundary_weight_mean",
    )
    selected = [row for row in rows if bool(row["geometry_event"])]
    values = torch.zeros((len(names) + 1,), device=device, dtype=torch.float64)
    values[0] = float(len(selected))
    for row in selected:
        for index, name in enumerate(names, start=1):
            values[index] += float(row[name])
    if int(world_size) > 1:
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
    count = float(values[0].item())
    result = {"event_count": count}
    result.update(
        {
            name: float(values[index].item() / max(count, 1.0))
            for index, name in enumerate(names, start=1)
        }
    )
    return result


class ConditionOnlyTrainingForward(nn.Module):
    def __init__(self, model: NativeSLatConditionOnlyFlow) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        x: Any,
        t: torch.Tensor,
        condition: Any,
        sample: dict[str, Any] | None,
        *,
        view_indices: torch.Tensor | None,
        stock_velocity: Any,
    ) -> tuple[Any, dict[str, torch.Tensor]]:
        prediction, stats = self.model.adapted_prediction(
            x,
            t,
            condition,
            sample,
            view_indices=view_indices,
            stock_velocity=stock_velocity,
        )
        # DDP must see the aggregator in unconditional micro-steps as well.
        if sample is None:
            anchor = prediction.feats.new_zeros((), dtype=torch.float32)
            for parameter in self.model.aggregator.parameters():
                anchor = anchor + parameter.reshape(-1)[0].float() * 0.0
            prediction = prediction.replace(
                prediction.feats + anchor.to(dtype=prediction.feats.dtype)
            )
        return prediction, stats


def group_gradient_norms(model: nn.Module) -> dict[str, float]:
    values = {"aggregator": 0.0, "block_condition": 0.0}
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        key = "aggregator" if name.startswith("aggregator.") else "block_condition"
        values[key] += float(parameter.grad.detach().float().square().sum().item())
    return {key: value**0.5 for key, value in values.items()}


def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    step: int,
    micro_step: int,
    args: argparse.Namespace,
    model_summary: dict[str, Any],
    data_identity: dict[str, Any],
    history: list[dict[str, Any]],
    ema_state: dict[str, torch.Tensor],
    ema_last_decay: float,
) -> None:
    trainable = [value for value in model.parameters() if value.requires_grad]
    if not parameters_finite(trainable) or not optimizer_state_finite(optimizer):
        raise RuntimeError(f"refusing to save non-finite condition-only state: {path}")
    if not finite_tree(scaler.state_dict()) or not ema_state_finite(ema_state):
        raise RuntimeError(f"refusing to save non-finite scaler/EMA: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": NATIVE_SLAT_CONDITION_ONLY_VERSION,
            "step": int(step),
            "micro_step": int(micro_step),
            "model_trainable_state": trainable_state_dict(model),
            "ema_trainable_state": {
                name: value.detach().cpu().clone() for name, value in ema_state.items()
            },
            "ema": {
                "target_decay": float(args.ema_decay),
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
def initial_stock_audit(
    model: NativeSLatConditionOnlyFlow,
    sampler: Any,
    sample: dict[str, Any],
    *,
    mean: torch.Tensor,
    std: torch.Tensor,
    device: torch.device,
) -> dict[str, Any]:
    target = normalized_target(sample, mean=mean, std=std, device=device)
    noise = sp.SparseTensor(feats=torch.zeros_like(target.feats), coords=target.coords)
    x_t, _ = sampler._get_model_gt(target, 0.5, noise)
    t = torch.full((1,), 500.0, device=device)
    positive = to_device_tree(sample["condition"]["cond"], device)
    negative = to_device_tree(sample["condition"]["neg_cond"], device)
    stock_positive = model.stock_prediction(x_t, t, positive)
    stock_negative = model.stock_prediction(x_t, t, negative)
    full_positive, positive_stats = model.adapted_prediction(
        x_t,
        t,
        positive,
        sample["lifting_sample"],
        stock_velocity=stock_positive,
    )
    full_negative, negative_stats = model.adapted_prediction(
        x_t, t, negative, None, stock_velocity=stock_negative
    )
    trainable_names = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    report = {
        "conditional_max_abs": float(
            (full_positive.feats.float() - stock_positive.feats.float()).abs().amax().item()
        ),
        "unconditional_max_abs": float(
            (full_negative.feats.float() - stock_negative.feats.float()).abs().amax().item()
        ),
        "conditional_blocks": len(model.block_condition.projections),
        "stock_block_count": len(model.flow_core.blocks),
        "supported_fraction": float(positive_stats["supported_fraction"].item()),
        "unconditional_condition_present": float(
            negative_stats["condition_present"].item()
        ),
        "trainable_names": trainable_names,
        "lora_parameter_count": sum(
            parameter.numel()
            for name, parameter in model.named_parameters()
            if "lora_" in name
        ),
        "view_fusion_parameter_count": sum(
            parameter.numel()
            for name, parameter in model.named_parameters()
            if "view_fusion" in name or "view_gate" in name
        ),
    }
    report["passed"] = bool(
        report["conditional_max_abs"] == 0.0
        and report["unconditional_max_abs"] == 0.0
        and report["conditional_blocks"] == report["stock_block_count"]
        and report["supported_fraction"] > 0.0
        and report["unconditional_condition_present"] == 0.0
        and report["lora_parameter_count"] == 0
        and report["view_fusion_parameter_count"] == 0
        and all(
            name.startswith("aggregator.") or name.startswith("block_condition.")
            for name in trainable_names
        )
    )
    if not report["passed"]:
        raise RuntimeError(f"initial condition-only Stock audit failed: {report}")
    return report


def main() -> None:
    args = make_parser().parse_args()
    validate_args(args)
    warmup_steps = resolve_warmup_steps(args)
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
    dataset = NativeConditionSLatDataset(
        args.cache_manifest,
        args.lifting_cache_manifest,
        indices=args.indices,
        verify_hashes=bool(args.verify_cache_hashes),
    )
    dataset.limit_objects(int(args.max_objects))
    if len(dataset) <= 0:
        raise RuntimeError("condition-only training selection is empty")
    if dataset.config.get("condition_arch") != "native_ss_genrecon_v2":
        raise RuntimeError("condition-only SLat requires native_ss_genrecon_v2 cache")
    _, ss_binding_all = load_ss_evidence(args.native_ss_report)
    ss_binding = upstream_binding(ss_binding_all)
    if dataset.config.get("native_ss_deployment") != ss_binding_all:
        raise RuntimeError("SLat cache and training bind different Native SS deployments")
    stock_freeze = load_stock_slat_freeze(args.stock_slat_freeze)
    decoder_audit = validate_decoder_audit(
        args.target_decoder_audit,
        cache_config=dataset.config,
        pretrained=args.pretrained,
    )
    object_uids = sorted({str(row["object_uid"]) for row in dataset.rows})
    data_identity = {
        "cache_manifest": str(Path(args.cache_manifest).resolve()),
        "cache_manifest_sha256": sha256_file(args.cache_manifest),
        "lifting_cache_manifest": str(Path(args.lifting_cache_manifest).resolve()),
        "lifting_cache_manifest_sha256": sha256_file(args.lifting_cache_manifest),
        "config_hash": dataset.config_hash,
        "sample_count": len(dataset),
        "object_count": len(object_uids),
        "object_uids": object_uids,
        "object_uid_hash": canonical_json_sha256(object_uids),
        "native_ss": ss_binding,
        "stock_slat_freeze_sha256": stock_freeze["freeze_sha256"],
        "target_decoder_audit": decoder_audit,
    }
    object_sampler = ObjectBalancedDistributedSampler(
        dataset.rows,
        num_replicas=world_size,
        rank=rank,
        seed=int(args.seed),
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        sampler=object_sampler,
        num_workers=int(args.num_workers),
        collate_fn=collate_native_one,
        pin_memory=True,
    )
    geometry_enabled = float(args.decoder_geometry_weight) > 0.0
    sampler, model, decoder, model_summary, sampler_defaults, normalization = (
        build_native_slat_condition_only_components(
            pretrained=args.pretrained,
            stock_slat_freeze=stock_freeze,
            upstream_native_ss=ss_binding,
            condition_channels=int(args.condition_channels),
            gradient_checkpointing=bool(args.gradient_checkpointing),
            need_decoder=geometry_enabled,
            device=device,
        )
    )
    if geometry_enabled:
        if decoder is None:
            raise RuntimeError("decoder-aware objective requires the frozen Mesh decoder")
        decoder.eval()
        decoder.use_checkpoint = bool(args.gradient_checkpointing)
        for block in decoder.blocks:
            block.use_checkpoint = bool(args.gradient_checkpointing)
    runtime_normalization = {
        key: [float(value) for value in values] for key, values in normalization.items()
    }
    if canonical_json_sha256(runtime_normalization) != dataset.slat_normalization_hash:
        raise RuntimeError("runtime SLat normalization differs from cache")
    mean = torch.tensor(runtime_normalization["mean"], device=device)[None]
    std = torch.tensor(runtime_normalization["std"], device=device)[None]
    model_summary = {
        **model_summary,
        "data_identity": data_identity,
        "target_decoder_audit": decoder_audit,
        "slat_normalization": runtime_normalization,
        "slat_normalization_hash": dataset.slat_normalization_hash,
        "t_schedule": {
            "name": str(args.t_schedule),
            "mean": float(args.t_logit_mean),
            "std": float(args.t_logit_std),
            "uniform_probability": float(args.t_uniform_probability),
            "rng": (
                "separate deterministic rank/micro-step generator"
                if args.separate_t_rng
                else "legacy global torch RNG"
            ),
        },
        "training_objective": {
            "version": DECODER_GEOMETRY_LOSS_VERSION,
            "flow_matching_weight": 1.0,
            "decoder_geometry_weight": float(args.decoder_geometry_weight),
            "stock_flow_trust_weight": float(args.stock_flow_trust_weight),
            "stock_flow_required_improvement": float(
                args.stock_flow_required_improvement
            ),
            "stock_trust_weight": float(args.stock_trust_weight),
            "stock_required_improvement": float(args.stock_required_improvement),
            "geometry_event_probability": float(args.geometry_event_probability),
            "geometry_t_max": float(args.geometry_t_max),
            "geometry_event_rng": "deterministic rank/micro-step hash; base RNG unchanged",
            "decoder_trainable": False,
            "decoder_target": (
                "continuous frozen Mesh-decoder SDF/deformation/topology fields; "
                "no FlexiCubes face extraction in the training graph"
            ),
            "stock_reference": "same x_t, timestep, condition and frozen Stock velocity",
            "stock_relative_denominator_floor": 1.0e-4,
        },
        "p_uncond": float(args.p_uncond),
        "view_augmentation": {
            "random_subset": True,
            "min": int(args.min_condition_views),
            "max": int(args.max_condition_views),
        },
        "sampler_defaults": sampler_defaults,
        "optimization": {
            "warmup_steps": warmup_steps,
            "ema_target_decay": float(args.ema_decay),
            "world_size": world_size,
            "global_effective_batch": world_size * int(args.grad_accum),
        },
        "training_coordinate_policy": (
            "true target SLat coordinates for flow matching; deployment evaluation "
            "uses frozen Native SS predicted coordinates"
        ),
    }
    initial_audit = initial_stock_audit(
        model, sampler, dataset[0], mean=mean, std=std, device=device
    )
    training_forward = ConditionOnlyTrainingForward(model)
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
        condition_only_parameter_group(
            model,
            lr=float(args.condition_lr),
            weight_decay=float(args.condition_weight_decay),
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
        validate_native_slat_condition_only_checkpoint(
            checkpoint,
            pretrained=args.pretrained,
            stock_slat_freeze=stock_freeze,
            upstream_native_ss=ss_binding,
        )
        if checkpoint.get("data_identity") != data_identity:
            raise RuntimeError("resume data identity differs")
        saved = checkpoint["args"]
        fields = (
            "condition_channels",
            "condition_lr",
            "condition_weight_decay",
            "p_uncond",
            "t_logit_mean",
            "t_logit_std",
            "t_schedule",
            "t_uniform_probability",
            "separate_t_rng",
            "max_objects",
            "decoder_geometry_weight",
            "stock_flow_trust_weight",
            "stock_flow_required_improvement",
            "stock_trust_weight",
            "stock_required_improvement",
            "geometry_event_probability",
            "geometry_t_max",
            "min_condition_views",
            "max_condition_views",
            "grad_accum",
            "max_steps",
            "warmup_steps",
            "warmup_ratio",
            "ema_decay",
        )
        legacy_defaults = {
            "t_schedule": "logit_normal",
            "t_uniform_probability": 0.75,
            "separate_t_rng": False,
            "max_objects": 0,
            "decoder_geometry_weight": 0.0,
            "stock_flow_trust_weight": 0.0,
            "stock_flow_required_improvement": 0.01,
            "stock_trust_weight": 0.0,
            "stock_required_improvement": 0.01,
            "geometry_event_probability": 0.125,
            "geometry_t_max": 0.5,
        }
        mismatch = {
            key: (saved.get(key, legacy_defaults.get(key)), getattr(args, key))
            for key in fields
            if saved.get(key, legacy_defaults.get(key)) != getattr(args, key)
        }
        if mismatch:
            raise ValueError(f"resume argument mismatch={mismatch}")
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

    trainable = [value for value in model.parameters() if value.requires_grad]
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
            target = normalized_target(sample, mean=mean, std=std, device=device)
            noise = sp.SparseTensor(feats=torch.randn_like(target.feats), coords=target.coords)
            t_value = sample_condition_only_t(
                args,
                device,
                seed=int(args.seed),
                rank=rank,
                micro_step=micro_step,
            )
            x_t, gt_velocity = sampler._get_model_gt(target, t_value, noise)
            t = torch.full((1,), 1000.0 * t_value, device=device)
            view_indices = sample_view_indices(sample, args, device)
            positive_all = to_device_tree(sample["condition"]["cond"], device)
            negative_all = to_device_tree(sample["condition"]["neg_cond"], device)
            positive = select_context_views(positive_all, view_indices)
            negative = select_context_views(negative_all, view_indices)
            unconditional = random.random() < float(args.p_uncond)
            condition = negative if unconditional else positive
            condition_sample = None if unconditional else sample["lifting_sample"]
            with torch.no_grad():
                stock = model.stock_prediction(x_t, t, condition)
                stock_loss = F.mse_loss(stock.feats.float(), gt_velocity.feats.float())
            geometry_event = bool(
                geometry_enabled
                and not unconditional
                and t_value <= float(args.geometry_t_max)
                and deterministic_geometry_event(
                    seed=int(args.seed),
                    rank=rank,
                    micro_step=micro_step,
                    probability=float(args.geometry_event_probability),
                )
            )
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
                        view_indices=None if unconditional else view_indices,
                        stock_velocity=stock,
                    )
                    flow_loss = F.mse_loss(
                        prediction.feats.float(), gt_velocity.feats.float()
                    )
                    flow_trust_loss, flow_relative_to_stock = (
                        stock_relative_trust_loss(
                            flow_loss,
                            stock_loss,
                            required_improvement=float(
                                args.stock_flow_required_improvement
                            ),
                        )
                    )
                    geometry_values: dict[str, torch.Tensor] = {}
                    if geometry_event:
                        if decoder is None:
                            raise AssertionError("geometry event without decoder")
                        geometry_values = decoder_geometry_objective(
                            decoder=decoder,
                            sampler=sampler,
                            x_t=x_t,
                            t_value=t_value,
                            full_velocity=prediction,
                            stock_velocity=stock,
                            target=target,
                            mean=mean,
                            std=std,
                            required_improvement=float(
                                args.stock_required_improvement
                            ),
                        )
                        total_loss = (
                            flow_loss
                            + float(args.stock_flow_trust_weight)
                            * flow_trust_loss
                            + float(args.decoder_geometry_weight)
                            * geometry_values["field_loss"]
                            + float(args.stock_trust_weight)
                            * geometry_values["trust_loss"]
                        )
                    else:
                        total_loss = (
                            flow_loss
                            + float(args.stock_flow_trust_weight)
                            * flow_trust_loss
                        )
                    loss = total_loss / float(args.grad_accum)
                scaler.scale(loss).backward()
            micro_step += 1
            accumulator.append(
                {
                    "flow_loss": float(flow_loss.detach().item()),
                    "total_loss": float(total_loss.detach().item()),
                    "stock_loss": float(stock_loss.detach().item()),
                    "stock_flow_trust_loss": float(
                        flow_trust_loss.detach().item()
                    ),
                    "flow_relative_to_stock": float(
                        flow_relative_to_stock.detach().item()
                    ),
                    "gain": float((stock_loss - flow_loss.detach()).item()),
                    "t": t_value,
                    "unconditional": unconditional,
                    "views": float(len(view_indices)),
                    "flow_delta_rms": float(stats["flow_delta_rms"].detach().item()),
                    "condition_rms": float(stats["condition_rms"].detach().item()),
                    "supported_fraction": float(stats["supported_fraction"].detach().item()),
                    "active_points": float(target.feats.shape[0]),
                    "geometry_event": geometry_event,
                    **{
                        name: float(
                            geometry_values[name].detach().float().item()
                        )
                        if geometry_event
                        else 0.0
                        for name in (
                            "field_loss",
                            "stock_field_loss",
                            "trust_loss",
                            "relative_field_delta",
                            "sdf_loss",
                            "sign_loss",
                            "deform_loss",
                            "topology_loss",
                            "boundary_weight_mean",
                        )
                    },
                }
            )
            if not sync_step:
                continue
            scaler.unscale_(optimizer)
            if not distributed_true(
                gradients_finite(trainable), device=device, world_size=world_size
            ):
                raise RuntimeError("condition-only gradients became non-finite")
            grad_norm = float(
                torch.nn.utils.clip_grad_norm_(trainable, float(args.grad_clip)).item()
            )
            group_gradients = group_gradient_norms(model)
            next_step = step + 1
            lr_scale = warmup_factor(next_step, warmup_steps)
            for group in optimizer.param_groups:
                group["lr"] = float(group["base_lr"]) * lr_scale
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            finite = parameters_finite(trainable) and optimizer_state_finite(optimizer)
            if not distributed_true(finite, device=device, world_size=world_size):
                raise RuntimeError("condition-only optimizer produced non-finite state")
            ema_last_decay = ema_ramp_decay(float(args.ema_decay), next_step)
            update_ema_state(ema_state, model, decay=ema_last_decay)
            step = next_step
            metric_names = (
                "flow_loss",
                "stock_loss",
                "gain",
                "t",
                "unconditional",
                "views",
                "flow_delta_rms",
                "condition_rms",
                "supported_fraction",
                "active_points",
                "total_loss",
                "stock_flow_trust_loss",
                "flow_relative_to_stock",
            )
            local_means = [
                float(
                    np.mean(
                        [
                            float(bool(row[name])) if name == "unconditional" else float(row[name])
                            for row in accumulator
                        ]
                    )
                )
                for name in metric_names
            ]
            metrics = distributed_means(
                local_means, device=device, world_size=world_size
            )
            timestep_bins = distributed_timestep_bins(
                accumulator, device=device, world_size=world_size
            )
            geometry_summary = distributed_geometry_summary(
                accumulator, device=device, world_size=world_size
            )
            row = {
                "step": step,
                "micro_step": micro_step,
                "global_micro_samples": step * int(args.grad_accum) * world_size,
                "flow_loss": metrics[0],
                "stock_loss": metrics[1],
                "gain": metrics[2],
                "t_mean": metrics[3],
                "unconditional_fraction": metrics[4],
                "view_count_mean": metrics[5],
                "flow_delta_rms": metrics[6],
                "condition_rms": metrics[7],
                "supported_fraction": metrics[8],
                "active_point_count_mean": metrics[9],
                "total_loss": metrics[10],
                "stock_flow_trust_loss": metrics[11],
                "flow_relative_to_stock": metrics[12],
                "timestep_bins": timestep_bins,
                "decoder_geometry": geometry_summary,
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
                print(f"[native_slat_condition_only] {json.dumps(row)}", flush=True)
            if rank == 0 and (
                step % int(args.save_every) == 0 or step == int(args.max_steps)
            ):
                path = output_dir / "checkpoints" / f"step_{step:06d}.pt"
                for destination in (path, output_dir / "checkpoints" / "last.pt"):
                    save_checkpoint(
                        destination,
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

    passed = distributed_true(
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
        "format": NATIVE_SLAT_CONDITION_ONLY_VERSION,
        "completed": True,
        "passed": passed,
        "step": step,
        "micro_step": micro_step,
        "elapsed_seconds": time.time() - started,
        "model_summary": model_summary,
        "data_identity": data_identity,
        "initial_stock_audit": initial_audit,
        "history": history,
        "checkpoint": str(output_dir / "checkpoints" / "last.pt"),
        "evaluation_weights": "ema",
        "ema": {
            "target_decay": float(args.ema_decay),
            "last_decay": ema_last_decay,
            "updates": step,
        },
        "explicitly_absent": model_summary["explicitly_absent"],
    }
    if rank == 0:
        (output_dir / "report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(json.dumps({"passed": passed, "output": str(output_dir)}), flush=True)
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()
    raise SystemExit(0 if passed else 2)


if __name__ == "__main__":
    main()
