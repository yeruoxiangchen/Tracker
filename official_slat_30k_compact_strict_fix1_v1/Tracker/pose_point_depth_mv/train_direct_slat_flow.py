#!/usr/bin/env python3
"""Train zero-init corrected-SS support + LoRA on the native SLAT Flow."""

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
from torch.utils.data import DataLoader, DistributedSampler, Sampler


TRACKER_ROOT = Path(__file__).resolve().parents[1]
RECONVIAGEN_ROOT = TRACKER_ROOT / "ReconViaGen"
VGGT_ROOT = RECONVIAGEN_ROOT / "wheels" / "vggt"
for path in (TRACKER_ROOT, RECONVIAGEN_ROOT, VGGT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from pose_point_depth_mv.direct_slat_data import (  # noqa: E402
    DirectSLatCacheDataset,
    collate_direct_slat_one,
    sha256_file,
)
from pose_point_depth_mv.direct_slat_flow import (  # noqa: E402
    DIRECT_SLAT_TRAINING_SEMANTICS_V2,
    DIRECT_SLAT_TRAINING_SEMANTICS_V3,
    DIRECT_SLAT_TRAINING_SEMANTICS_V4,
    DIRECT_SLAT_TRAINING_SEMANTICS_V5,
    DIRECT_SLAT_FLOW_VERSION,
    SLAT_DELTA_BOUND_HARD,
    SLAT_DELTA_BOUND_MODES,
    SLAT_DELTA_BOUND_SMOOTH,
    SLAT_GUIDED_DELTA_POLICIES,
    SLAT_GUIDED_DELTA_POLICY_LEGACY,
    SLAT_GUIDED_DELTA_POLICY_V2,
    SLAT_ROLLOUT_SUPERVISION_ALL_VISITED,
    SLAT_ROLLOUT_SUPERVISION_POLICIES,
    SLAT_ROLLOUT_SUPERVISION_TERMINAL_CONSTANT_MEMORY,
    SLAT_SUPPORT_INTERVAL_CFG_ACTIVE,
    SLAT_SUPPORT_INTERVAL_POLICIES,
    SLAT_RESIDUAL_COMBINATION_BRANCH_BUDGET,
    SLAT_RESIDUAL_COMBINATION_JOINT,
    SLAT_RESIDUAL_COMBINATION_POLICIES,
    build_direct_slat_components,
    canonical_json_sha256,
    cfg_interval_is_active,
    combine_sparse_cfg,
    correct_over_wrong_support_rank_loss,
    deterministic_probability_event,
    deterministic_probability_partition,
    deterministic_wrong_support_index,
    load_strict_trainable_state,
    native_flow_timestep_sequence,
    slat_target_cache_identity,
    stock_relative_residual_excess_loss,
    strict_trainable_state_dict,
    support_generator_identity,
    validate_trainable_whitelist,
)
from pose_point_depth_mv.train_direct_flow import (  # noqa: E402
    ObjectBalancedDistributedSampler,
)
from reconvggt_ar_adapter_a.train_pointpose_ss_lora import (  # noqa: E402
    finite_tree,
    gradients_finite,
    optimizer_state_finite,
    parameters_finite,
    sample_t,
)
from trellis.modules import sparse as sp  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache_manifest", required=True)
    parser.add_argument("--target_decoder_audit", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--pretrained", default="Stable-X/trellis-vggt-v0-2")
    parser.add_argument("--indices", default="all")
    parser.add_argument("--resume", default="")
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--save_every", type=int, default=100)
    parser.add_argument("--log_every", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1.0e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--amp_dtype", choices=("bf16", "fp16", "none"), default="bf16")
    parser.add_argument("--amp_init_scale", type=float, default=8192.0)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--adapter_hidden_dim", type=int, default=128)
    parser.add_argument("--max_slat_points", type=int, default=40960)
    parser.add_argument("--support_scale", type=float, default=1.0)
    parser.add_argument(
        "--slat_delta_scale",
        type=float,
        default=1.0,
        help="Scale applied to the Direct-SLAT residual after optional RMS clipping.",
    )
    parser.add_argument(
        "--slat_delta_rms_ratio_cap",
        type=float,
        default=-1.0,
        help=(
            "Per-sparse-batch RMS cap relative to stock velocity RMS; "
            "negative disables the cap and preserves legacy Full behavior."
        ),
    )
    parser.add_argument("--delta_norm_weight", type=float, default=1.0e-4)
    parser.add_argument(
        "--training_semantics",
        choices=(
            "legacy_v1",
            DIRECT_SLAT_TRAINING_SEMANTICS_V2,
            DIRECT_SLAT_TRAINING_SEMANTICS_V3,
            DIRECT_SLAT_TRAINING_SEMANTICS_V4,
            DIRECT_SLAT_TRAINING_SEMANTICS_V5,
        ),
        default="legacy_v1",
    )
    parser.add_argument(
        "--slat_guided_delta_policy",
        choices=SLAT_GUIDED_DELTA_POLICIES,
        default=SLAT_GUIDED_DELTA_POLICY_LEGACY,
    )
    parser.add_argument(
        "--slat_delta_bound_mode",
        choices=SLAT_DELTA_BOUND_MODES,
        default=SLAT_DELTA_BOUND_HARD,
    )
    parser.add_argument(
        "--support_interval_policy",
        choices=SLAT_SUPPORT_INTERVAL_POLICIES,
        default="all_steps_v1",
    )
    parser.add_argument(
        "--slat_residual_combination_policy",
        choices=SLAT_RESIDUAL_COMBINATION_POLICIES,
        default=SLAT_RESIDUAL_COMBINATION_JOINT,
    )
    parser.add_argument("--slat_lora_delta_scale", type=float, default=1.0)
    parser.add_argument(
        "--slat_lora_delta_rms_ratio_cap", type=float, default=-1.0
    )
    parser.add_argument("--slat_support_delta_scale", type=float, default=1.0)
    parser.add_argument(
        "--slat_support_delta_rms_ratio_cap", type=float, default=-1.0
    )
    parser.add_argument("--raw_delta_excess_weight", type=float, default=0.0)
    parser.add_argument("--wrong_support_rank_weight", type=float, default=0.0)
    parser.add_argument("--wrong_support_margin", type=float, default=0.0)
    parser.add_argument("--wrong_support_probability", type=float, default=0.0)
    parser.add_argument("--support_dropout_weight", type=float, default=0.0)
    parser.add_argument("--support_dropout_probability", type=float, default=0.0)
    parser.add_argument(
        "--wrong_support_stock_weight",
        type=float,
        default=0.0,
        help="Train wrong-support Full back toward the same native stock velocity.",
    )
    parser.add_argument(
        "--rollout_consistency_weight",
        type=float,
        default=0.0,
        help=(
            "Weight of detached-state velocity losses; V4/V5 apply it over "
            "the selected native-schedule rollout horizon."
        ),
    )
    parser.add_argument("--rollout_probability", type=float, default=0.0)
    parser.add_argument(
        "--rollout_step_size",
        type=float,
        default=0.0,
        help="Normalized native Euler step size for rollout_aligned_v3.",
    )
    parser.add_argument(
        "--rollout_horizons",
        default="1,2,4",
        help=(
            "Comma-separated native-schedule truncated rollout horizons "
            "for V4/V5."
        ),
    )
    parser.add_argument(
        "--rollout_supervision_policy",
        choices=SLAT_ROLLOUT_SUPERVISION_POLICIES,
        default=SLAT_ROLLOUT_SUPERVISION_ALL_VISITED,
        help=(
            "all_visited_v1 retains a backward graph at every visited state; "
            "terminal_only_constant_memory_v1 generates the same trajectory "
            "under no_grad and supervises only its terminal state."
        ),
    )
    parser.add_argument("--rollout_schedule_steps", type=int, default=25)
    parser.add_argument("--rollout_rescale_t", type=float, default=3.0)
    parser.add_argument(
        "--endpoint_x0_weight",
        type=float,
        default=0.0,
        help=(
            "Weight of the x0/target-SLAT proxy: final visited state in V4, "
            "all supervised visited states in V5 all_visited_v1, or only the "
            "terminal state in V5.1 constant-memory mode."
        ),
    )
    parser.add_argument(
        "--rollout_endpoint_rank_weight",
        type=float,
        default=0.0,
        help=(
            "Weight requiring supervised-state Full x0 proxy loss to beat the "
            "same-state native Stock proxy loss in V5/V5.1."
        ),
    )
    parser.add_argument(
        "--rollout_endpoint_rank_margin",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--train_cfg_strength",
        type=float,
        default=1.0,
        help="Internal CFG strength used by rollout_aligned_v3 teacher/rollout calls.",
    )
    parser.add_argument(
        "--train_cfg_interval",
        type=float,
        nargs=2,
        default=(0.5, 1.0),
        metavar=("LO", "HI"),
        help="Inclusive normalized-t interval using internal post-CFG composition.",
    )
    parser.add_argument(
        "--sampling_mode", choices=("object_balanced", "sequence"), default="object_balanced"
    )
    parser.add_argument(
        "--t_schedule",
        choices=("uniform", "logit_normal", "high_t_mix", "native_schedule"),
        default="uniform",
    )
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--verify_cache_hashes", action="store_true")
    return parser.parse_args()


def parse_rollout_horizons(value: str) -> tuple[int, ...]:
    try:
        horizons = tuple(
            sorted({int(item.strip()) for item in str(value).split(",") if item.strip()})
        )
    except ValueError as error:
        raise ValueError(
            f"rollout_horizons must be comma-separated integers: {value!r}"
        ) from error
    if not horizons or any(item <= 0 for item in horizons):
        raise ValueError("rollout_horizons must contain positive integers")
    return horizons


def validate_args(args: argparse.Namespace) -> None:
    finite_names = (
        "max_steps",
        "save_every",
        "log_every",
        "grad_accum",
        "grad_clip",
        "amp_init_scale",
        "lora_rank",
        "lora_alpha",
        "adapter_hidden_dim",
        "max_slat_points",
        "rollout_schedule_steps",
        "support_scale",
        "slat_delta_scale",
        "slat_delta_rms_ratio_cap",
        "slat_lora_delta_scale",
        "slat_lora_delta_rms_ratio_cap",
        "slat_support_delta_scale",
        "slat_support_delta_rms_ratio_cap",
        "delta_norm_weight",
        "raw_delta_excess_weight",
        "wrong_support_rank_weight",
        "wrong_support_margin",
        "wrong_support_probability",
        "support_dropout_weight",
        "support_dropout_probability",
        "wrong_support_stock_weight",
        "rollout_consistency_weight",
        "rollout_probability",
        "rollout_step_size",
        "rollout_schedule_steps",
        "rollout_rescale_t",
        "endpoint_x0_weight",
        "rollout_endpoint_rank_weight",
        "rollout_endpoint_rank_margin",
        "train_cfg_strength",
    )
    interval = tuple(float(value) for value in args.train_cfg_interval)
    if len(interval) != 2 or not all(math.isfinite(value) for value in interval):
        raise ValueError("train_cfg_interval must contain two finite values")
    nonfinite = [
        name for name in finite_names if not math.isfinite(float(getattr(args, name)))
    ]
    if nonfinite:
        raise ValueError(f"non-finite training arguments={nonfinite}")
    for name in (
        "max_steps",
        "save_every",
        "log_every",
        "grad_accum",
        "grad_clip",
        "amp_init_scale",
        "lora_rank",
        "lora_alpha",
        "adapter_hidden_dim",
        "max_slat_points",
    ):
        if float(getattr(args, name)) <= 0:
            raise ValueError(f"{name} must be positive")
    for name in (
        "delta_norm_weight",
        "raw_delta_excess_weight",
        "wrong_support_rank_weight",
        "wrong_support_margin",
        "support_dropout_weight",
        "wrong_support_stock_weight",
        "rollout_consistency_weight",
        "rollout_step_size",
        "endpoint_x0_weight",
        "rollout_endpoint_rank_weight",
        "rollout_endpoint_rank_margin",
        "train_cfg_strength",
    ):
        if float(getattr(args, name)) < 0:
            raise ValueError(f"{name} must be non-negative")
    for name in (
        "wrong_support_probability",
        "support_dropout_probability",
        "rollout_probability",
    ):
        if not 0.0 <= float(getattr(args, name)) <= 1.0:
            raise ValueError(f"{name} must be within [0, 1]")
    if float(args.support_scale) <= 0:
        raise ValueError("training requires support_scale > 0")
    if float(args.slat_delta_scale) <= 0:
        raise ValueError("training requires slat_delta_scale > 0")
    if (
        float(args.slat_delta_rms_ratio_cap) < 0
        and float(args.slat_delta_rms_ratio_cap) != -1.0
    ):
        raise ValueError(
            "slat_delta_rms_ratio_cap must be -1 (disabled) or non-negative"
        )
    for name in (
        "slat_lora_delta_rms_ratio_cap",
        "slat_support_delta_rms_ratio_cap",
    ):
        if float(getattr(args, name)) < 0 and float(getattr(args, name)) != -1.0:
            raise ValueError(f"{name} must be -1 (disabled) or non-negative")
    for name in ("slat_lora_delta_scale", "slat_support_delta_scale"):
        if float(getattr(args, name)) <= 0:
            raise ValueError(f"{name} must be positive")
    if not 0.0 <= interval[0] <= interval[1] <= 1.0:
        raise ValueError("train_cfg_interval must satisfy 0 <= LO <= HI <= 1")
    horizons = parse_rollout_horizons(args.rollout_horizons)
    if int(args.rollout_schedule_steps) <= 0:
        raise ValueError("rollout_schedule_steps must be positive")
    if float(args.rollout_rescale_t) <= 0.0:
        raise ValueError("rollout_rescale_t must be positive")
    if horizons[-1] > int(args.rollout_schedule_steps):
        raise ValueError("rollout horizon exceeds rollout_schedule_steps")
    if (
        args.training_semantics != DIRECT_SLAT_TRAINING_SEMANTICS_V5
        and args.rollout_supervision_policy
        != SLAT_ROLLOUT_SUPERVISION_ALL_VISITED
    ):
        raise ValueError(
            "terminal-only constant-memory rollout supervision is versioned "
            "for branch_budget_rollout_v5 only"
        )
    if args.training_semantics == DIRECT_SLAT_TRAINING_SEMANTICS_V2:
        required_positive = (
            "raw_delta_excess_weight",
            "wrong_support_rank_weight",
            "wrong_support_margin",
            "wrong_support_probability",
            "support_dropout_weight",
            "support_dropout_probability",
        )
        invalid = [name for name in required_positive if float(getattr(args, name)) <= 0]
        if invalid:
            raise ValueError(
                f"{DIRECT_SLAT_TRAINING_SEMANTICS_V2} requires positive {invalid}"
            )
        if float(args.slat_delta_rms_ratio_cap) <= 0:
            raise ValueError(
                f"{DIRECT_SLAT_TRAINING_SEMANTICS_V2} requires a positive residual cap"
            )
        if args.slat_guided_delta_policy != SLAT_GUIDED_DELTA_POLICY_V2:
            raise ValueError(
                f"{DIRECT_SLAT_TRAINING_SEMANTICS_V2} requires post_cfg_v2 rollout"
            )
    if args.training_semantics == DIRECT_SLAT_TRAINING_SEMANTICS_V3:
        required_positive = (
            "raw_delta_excess_weight",
            "wrong_support_probability",
            "wrong_support_stock_weight",
            "support_dropout_weight",
            "support_dropout_probability",
            "rollout_consistency_weight",
            "rollout_probability",
            "rollout_step_size",
        )
        invalid = [name for name in required_positive if float(getattr(args, name)) <= 0]
        if invalid:
            raise ValueError(
                f"{DIRECT_SLAT_TRAINING_SEMANTICS_V3} requires positive {invalid}"
            )
        if float(args.slat_delta_rms_ratio_cap) <= 0:
            raise ValueError(
                f"{DIRECT_SLAT_TRAINING_SEMANTICS_V3} requires a positive residual cap"
            )
        if args.slat_guided_delta_policy != SLAT_GUIDED_DELTA_POLICY_V2:
            raise ValueError(
                f"{DIRECT_SLAT_TRAINING_SEMANTICS_V3} requires post_cfg_v2 rollout"
            )
        if float(args.train_cfg_strength) <= 1.0:
            raise ValueError(
                f"{DIRECT_SLAT_TRAINING_SEMANTICS_V3} requires train_cfg_strength > 1"
            )
        auxiliary_probability = sum(
            float(getattr(args, name))
            for name in (
                "wrong_support_probability",
                "support_dropout_probability",
                "rollout_probability",
            )
        )
        if auxiliary_probability > 1.0 + 1.0e-12:
            raise ValueError(
                f"{DIRECT_SLAT_TRAINING_SEMANTICS_V3} auxiliary probabilities "
                "must sum to at most 1"
            )
    if args.training_semantics == DIRECT_SLAT_TRAINING_SEMANTICS_V4:
        required_positive = (
            "raw_delta_excess_weight",
            "wrong_support_probability",
            "wrong_support_stock_weight",
            "support_dropout_weight",
            "support_dropout_probability",
            "rollout_consistency_weight",
            "rollout_probability",
            "endpoint_x0_weight",
        )
        invalid = [
            name for name in required_positive if float(getattr(args, name)) <= 0
        ]
        if invalid:
            raise ValueError(
                f"{DIRECT_SLAT_TRAINING_SEMANTICS_V4} requires positive {invalid}"
            )
        if float(args.slat_delta_rms_ratio_cap) <= 0:
            raise ValueError(
                f"{DIRECT_SLAT_TRAINING_SEMANTICS_V4} requires a positive residual cap"
            )
        if args.slat_guided_delta_policy != SLAT_GUIDED_DELTA_POLICY_V2:
            raise ValueError(
                f"{DIRECT_SLAT_TRAINING_SEMANTICS_V4} requires post_cfg_v2 rollout"
            )
        if args.slat_delta_bound_mode != SLAT_DELTA_BOUND_SMOOTH:
            raise ValueError(
                f"{DIRECT_SLAT_TRAINING_SEMANTICS_V4} requires smooth_rms_v2"
            )
        if args.support_interval_policy != SLAT_SUPPORT_INTERVAL_CFG_ACTIVE:
            raise ValueError(
                f"{DIRECT_SLAT_TRAINING_SEMANTICS_V4} requires cfg_active_only_v1"
            )
        if args.t_schedule != "native_schedule":
            raise ValueError(
                f"{DIRECT_SLAT_TRAINING_SEMANTICS_V4} requires native_schedule"
            )
        if float(args.rollout_step_size) != 0.0:
            raise ValueError(
                f"{DIRECT_SLAT_TRAINING_SEMANTICS_V4} uses schedule horizons, "
                "so rollout_step_size must be 0"
            )
        if float(args.train_cfg_strength) <= 1.0:
            raise ValueError(
                f"{DIRECT_SLAT_TRAINING_SEMANTICS_V4} requires train_cfg_strength > 1"
            )
        auxiliary_probability = sum(
            float(getattr(args, name))
            for name in (
                "wrong_support_probability",
                "support_dropout_probability",
                "rollout_probability",
            )
        )
        if auxiliary_probability > 1.0 + 1.0e-12:
            raise ValueError(
                f"{DIRECT_SLAT_TRAINING_SEMANTICS_V4} auxiliary probabilities "
                "must sum to at most 1"
            )
    if args.training_semantics == DIRECT_SLAT_TRAINING_SEMANTICS_V5:
        required_positive = (
            "raw_delta_excess_weight",
            "wrong_support_probability",
            "wrong_support_stock_weight",
            "support_dropout_weight",
            "support_dropout_probability",
            "rollout_consistency_weight",
            "rollout_probability",
            "endpoint_x0_weight",
            "rollout_endpoint_rank_weight",
        )
        invalid = [
            name for name in required_positive if float(getattr(args, name)) <= 0
        ]
        if invalid:
            raise ValueError(
                f"{DIRECT_SLAT_TRAINING_SEMANTICS_V5} requires positive {invalid}"
            )
        if float(args.slat_delta_rms_ratio_cap) <= 0:
            raise ValueError(
                f"{DIRECT_SLAT_TRAINING_SEMANTICS_V5} requires a positive total cap"
            )
        if (
            float(args.slat_lora_delta_rms_ratio_cap) <= 0
            or float(args.slat_support_delta_rms_ratio_cap) <= 0
        ):
            raise ValueError(
                f"{DIRECT_SLAT_TRAINING_SEMANTICS_V5} requires positive branch caps"
            )
        if (
            float(args.slat_lora_delta_rms_ratio_cap)
            + float(args.slat_support_delta_rms_ratio_cap)
            > float(args.slat_delta_rms_ratio_cap) + 1.0e-12
        ):
            raise ValueError("V5 LoRA/support branch caps exceed the total cap")
        if (
            args.slat_residual_combination_policy
            != SLAT_RESIDUAL_COMBINATION_BRANCH_BUDGET
        ):
            raise ValueError(
                f"{DIRECT_SLAT_TRAINING_SEMANTICS_V5} requires "
                f"{SLAT_RESIDUAL_COMBINATION_BRANCH_BUDGET}"
            )
        if args.slat_guided_delta_policy != SLAT_GUIDED_DELTA_POLICY_V2:
            raise ValueError(
                f"{DIRECT_SLAT_TRAINING_SEMANTICS_V5} requires post_cfg_v2"
            )
        if args.slat_delta_bound_mode != SLAT_DELTA_BOUND_SMOOTH:
            raise ValueError(
                f"{DIRECT_SLAT_TRAINING_SEMANTICS_V5} requires smooth_rms_v2"
            )
        if args.support_interval_policy != SLAT_SUPPORT_INTERVAL_CFG_ACTIVE:
            raise ValueError(
                f"{DIRECT_SLAT_TRAINING_SEMANTICS_V5} requires cfg_active_only_v1"
            )
        if args.t_schedule != "native_schedule":
            raise ValueError(
                f"{DIRECT_SLAT_TRAINING_SEMANTICS_V5} requires native_schedule"
            )
        if float(args.rollout_step_size) != 0.0:
            raise ValueError("V5 native-schedule rollout_step_size must be 0")
        if float(args.train_cfg_strength) <= 1.0:
            raise ValueError(
                f"{DIRECT_SLAT_TRAINING_SEMANTICS_V5} requires train_cfg_strength > 1"
            )
        if max(horizons) < 4:
            raise ValueError("V5 requires a rollout horizon of at least 4 steps")
        auxiliary_probability = sum(
            float(getattr(args, name))
            for name in (
                "wrong_support_probability",
                "support_dropout_probability",
                "rollout_probability",
            )
        )
        if auxiliary_probability > 1.0 + 1.0e-12:
            raise ValueError(
                f"{DIRECT_SLAT_TRAINING_SEMANTICS_V5} auxiliary probabilities "
                "must sum to at most 1"
            )


def to_device_tree(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device=device, non_blocking=True)
    if isinstance(value, dict):
        return {key: to_device_tree(child, device) for key, child in value.items()}
    if isinstance(value, list):
        return [to_device_tree(child, device) for child in value]
    if isinstance(value, tuple):
        return tuple(to_device_tree(child, device) for child in value)
    return value


def normalized_target(
    sample: dict[str, Any],
    *,
    mean: torch.Tensor,
    std: torch.Tensor,
    device: torch.device,
    max_points: int = 0,
    selection_seed: int = 0,
) -> Any:
    feats_cpu = sample["target_feats"]
    coords_cpu = sample["target_coords"]
    if int(max_points) > 0 and int(feats_cpu.shape[0]) > int(max_points):
        generator = torch.Generator().manual_seed(int(selection_seed))
        selected = torch.randperm(
            int(feats_cpu.shape[0]), generator=generator
        )[: int(max_points)]
        feats_cpu = feats_cpu[selected]
        coords_cpu = coords_cpu[selected]
    feats = feats_cpu.to(device=device, dtype=torch.float32)
    coords = coords_cpu.to(device=device)
    target = sp.SparseTensor(feats=feats, coords=coords)
    return (target - mean) / std


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
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters_finite(trainable):
        raise RuntimeError(f"refusing to save non-finite model: {path}")
    if not optimizer_state_finite(optimizer) or not finite_tree(scaler.state_dict()):
        raise RuntimeError(f"refusing to save non-finite optimizer/scaler: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": DIRECT_SLAT_FLOW_VERSION,
            "step": int(step),
            "micro_step": int(micro_step),
            "epoch": int(epoch),
            "samples_into_epoch": int(samples_into_epoch),
            "model_trainable_state": strict_trainable_state_dict(model),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "args": vars(args),
            "model_summary": summary,
            "history": history,
        },
        path,
    )


def validate_resume(
    checkpoint: dict[str, Any],
    args: argparse.Namespace,
    summary: dict[str, Any],
) -> None:
    if checkpoint.get("format") != DIRECT_SLAT_FLOW_VERSION:
        raise ValueError(f"unexpected resume format={checkpoint.get('format')!r}")
    saved = checkpoint.get("args", {})
    fields = (
        "cache_manifest",
        "target_decoder_audit",
        "pretrained",
        "lora_rank",
        "lora_alpha",
        "adapter_hidden_dim",
        "max_slat_points",
        "support_scale",
        "slat_delta_scale",
        "slat_delta_rms_ratio_cap",
        "delta_norm_weight",
        "training_semantics",
        "slat_guided_delta_policy",
        "slat_delta_bound_mode",
        "support_interval_policy",
        "slat_residual_combination_policy",
        "slat_lora_delta_scale",
        "slat_lora_delta_rms_ratio_cap",
        "slat_support_delta_scale",
        "slat_support_delta_rms_ratio_cap",
        "raw_delta_excess_weight",
        "wrong_support_rank_weight",
        "wrong_support_margin",
        "wrong_support_probability",
        "support_dropout_weight",
        "support_dropout_probability",
        "wrong_support_stock_weight",
        "rollout_consistency_weight",
        "rollout_probability",
        "rollout_step_size",
        "rollout_horizons",
        "rollout_supervision_policy",
        "rollout_schedule_steps",
        "rollout_rescale_t",
        "endpoint_x0_weight",
        "rollout_endpoint_rank_weight",
        "rollout_endpoint_rank_margin",
        "train_cfg_strength",
        "train_cfg_interval",
        "sampling_mode",
        "t_schedule",
        "seed",
        "lr",
        "weight_decay",
        "grad_accum",
        "grad_clip",
        "amp_dtype",
    )
    legacy_defaults = {
        "slat_delta_scale": 1.0,
        "slat_delta_rms_ratio_cap": -1.0,
        "training_semantics": "legacy_v1",
        "slat_guided_delta_policy": SLAT_GUIDED_DELTA_POLICY_LEGACY,
        "slat_delta_bound_mode": SLAT_DELTA_BOUND_HARD,
        "support_interval_policy": "all_steps_v1",
        "slat_residual_combination_policy": SLAT_RESIDUAL_COMBINATION_JOINT,
        "slat_lora_delta_scale": 1.0,
        "slat_lora_delta_rms_ratio_cap": -1.0,
        "slat_support_delta_scale": 1.0,
        "slat_support_delta_rms_ratio_cap": -1.0,
        "raw_delta_excess_weight": 0.0,
        "wrong_support_rank_weight": 0.0,
        "wrong_support_margin": 0.0,
        "wrong_support_probability": 0.0,
        "support_dropout_weight": 0.0,
        "support_dropout_probability": 0.0,
        "wrong_support_stock_weight": 0.0,
        "rollout_consistency_weight": 0.0,
        "rollout_probability": 0.0,
        "rollout_step_size": 0.0,
        "rollout_horizons": "1,2,4",
        "rollout_supervision_policy": SLAT_ROLLOUT_SUPERVISION_ALL_VISITED,
        "rollout_schedule_steps": 25,
        "rollout_rescale_t": 3.0,
        "endpoint_x0_weight": 0.0,
        "rollout_endpoint_rank_weight": 0.0,
        "rollout_endpoint_rank_margin": 0.0,
        "train_cfg_strength": 1.0,
        "train_cfg_interval": (0.5, 1.0),
    }
    mismatch = {
        name: (saved.get(name, legacy_defaults.get(name)), getattr(args, name))
        for name in fields
        if str(saved.get(name, legacy_defaults.get(name))) != str(getattr(args, name))
    }
    if mismatch:
        raise ValueError(f"resume protocol mismatch={mismatch}")
    for name in (
        "cache_identity",
        "slat_normalization",
        "support_adapter",
        "support_generator",
        "target_decoder_audit",
        "world_size",
    ):
        if checkpoint.get("model_summary", {}).get(name) != summary.get(name):
            raise ValueError(f"resume {name} binding differs from current runtime")


@torch.no_grad()
def stock_equivalence_audit(
    *,
    model: nn.Module,
    sampler: Any,
    sample: dict[str, Any],
    mean: torch.Tensor,
    std: torch.Tensor,
    device: torch.device,
    expect_zero_init: bool,
    max_slat_points: int,
) -> dict[str, Any]:
    target = normalized_target(
        sample,
        mean=mean,
        std=std,
        device=device,
        max_points=int(max_slat_points),
        selection_seed=22072026,
    )
    generator = torch.Generator(device=device).manual_seed(22072026)
    noise = sp.SparseTensor(
        feats=torch.randn(
            target.feats.shape,
            generator=generator,
            device=device,
            dtype=target.feats.dtype,
        ),
        coords=target.coords,
    )
    x_t, _ = sampler._get_model_gt(target, 0.75, noise)
    t = torch.full((1,), 750.0, device=device, dtype=torch.float32)
    condition = to_device_tree(sample["condition"]["cond"], device)
    support = (
        sample["corrected_ss"].to(device=device),
        sample["occupancy_logits64"].to(device=device),
        sample["physical_tokens16"].to(device=device),
    )
    stock = model.stock_prediction(x_t, t, condition)
    disabled, _ = model.conditioned_prediction(
        x_t,
        t,
        condition,
        corrected_ss=None,
        occupancy_logits64=None,
        physical_tokens16=None,
        stock_velocity=stock,
    )
    zero_scale, _ = model.conditioned_prediction(
        x_t,
        t,
        condition,
        corrected_ss=support[0],
        occupancy_logits64=support[1],
        physical_tokens16=support[2],
        stock_velocity=stock,
        support_scale=0.0,
    )
    enabled, _ = model.conditioned_prediction(
        x_t,
        t,
        condition,
        corrected_ss=support[0],
        occupancy_logits64=support[1],
        physical_tokens16=support[2],
        stock_velocity=stock,
    )
    report = {
        "support_none_max_abs": float((disabled.feats - stock.feats).abs().max().item()),
        "support_scale_zero_max_abs": float(
            (zero_scale.feats - stock.feats).abs().max().item()
        ),
        "zero_init_enabled_max_abs": float(
            (enabled.feats - stock.feats).abs().max().item()
        ),
        "expect_zero_init": bool(expect_zero_init),
    }
    report["passed"] = (
        report["support_none_max_abs"] == 0.0
        and report["support_scale_zero_max_abs"] == 0.0
        and (
            not expect_zero_init or report["zero_init_enabled_max_abs"] == 0.0
        )
    )
    if not report["passed"]:
        raise RuntimeError(f"direct SLAT stock equivalence failed: {report}")
    return report


@torch.no_grad()
def post_cfg_stock_equivalence_audit(
    *,
    model: nn.Module,
    sampler: Any,
    sample: dict[str, Any],
    mean: torch.Tensor,
    std: torch.Tensor,
    device: torch.device,
    cfg_strength: float,
    cfg_interval: tuple[float, float],
    slat_delta_scale: float,
    slat_delta_rms_ratio_cap: float,
    slat_delta_bound_mode: str,
    slat_residual_combination_policy: str,
    slat_lora_delta_scale: float,
    slat_lora_delta_rms_ratio_cap: float,
    slat_support_delta_scale: float,
    slat_support_delta_rms_ratio_cap: float,
    expect_zero_init: bool,
    max_slat_points: int,
) -> dict[str, Any]:
    """Audit the v3 value path against its deployed post-CFG stock reference."""

    target = normalized_target(
        sample,
        mean=mean,
        std=std,
        device=device,
        max_points=int(max_slat_points),
        selection_seed=23072026,
    )
    generator = torch.Generator(device=device).manual_seed(23072026)
    noise = sp.SparseTensor(
        feats=torch.randn(
            target.feats.shape,
            generator=generator,
            device=device,
            dtype=target.feats.dtype,
        ),
        coords=target.coords,
    )
    t_value = 0.75
    x_t, _ = sampler._get_model_gt(target, t_value, noise)
    t = torch.full((1,), 1000.0 * t_value, device=device, dtype=torch.float32)
    positive = to_device_tree(sample["condition"]["cond"], device)
    negative = to_device_tree(sample["condition"]["neg_cond"], device)
    support = (
        sample["corrected_ss"].to(device=device),
        sample["occupancy_logits64"].to(device=device),
        sample["physical_tokens16"].to(device=device),
    )
    active = cfg_interval_is_active(t_value, cfg_interval)
    stock_positive = model.stock_prediction(x_t, t, positive)
    stock_negative = model.stock_prediction(x_t, t, negative) if active else None
    stock_reference = (
        combine_sparse_cfg(
            stock_positive,
            stock_negative,
            cfg_strength=float(cfg_strength),
        )
        if stock_negative is not None
        else stock_positive
    )
    disabled, disabled_reference, _ = model.post_cfg_conditioned_prediction(
        x_t,
        t,
        positive,
        negative,
        corrected_ss=None,
        occupancy_logits64=None,
        physical_tokens16=None,
        stock_positive_velocity=stock_positive,
        stock_negative_velocity=stock_negative,
        cfg_strength=float(cfg_strength),
        cfg_active=active,
        support_scale=1.0,
        slat_delta_scale=float(slat_delta_scale),
        slat_delta_rms_ratio_cap=float(slat_delta_rms_ratio_cap),
        slat_delta_bound_mode=str(slat_delta_bound_mode),
        slat_residual_combination_policy=str(
            slat_residual_combination_policy
        ),
        slat_lora_delta_scale=float(slat_lora_delta_scale),
        slat_lora_delta_rms_ratio_cap=float(
            slat_lora_delta_rms_ratio_cap
        ),
        slat_support_delta_scale=float(slat_support_delta_scale),
        slat_support_delta_rms_ratio_cap=float(
            slat_support_delta_rms_ratio_cap
        ),
    )
    enabled, enabled_reference, _ = model.post_cfg_conditioned_prediction(
        x_t,
        t,
        positive,
        negative,
        corrected_ss=support[0],
        occupancy_logits64=support[1],
        physical_tokens16=support[2],
        stock_positive_velocity=stock_positive,
        stock_negative_velocity=stock_negative,
        cfg_strength=float(cfg_strength),
        cfg_active=active,
        support_scale=1.0,
        slat_delta_scale=float(slat_delta_scale),
        slat_delta_rms_ratio_cap=float(slat_delta_rms_ratio_cap),
        slat_delta_bound_mode=str(slat_delta_bound_mode),
        slat_residual_combination_policy=str(
            slat_residual_combination_policy
        ),
        slat_lora_delta_scale=float(slat_lora_delta_scale),
        slat_lora_delta_rms_ratio_cap=float(
            slat_lora_delta_rms_ratio_cap
        ),
        slat_support_delta_scale=float(slat_support_delta_scale),
        slat_support_delta_rms_ratio_cap=float(
            slat_support_delta_rms_ratio_cap
        ),
    )
    report = {
        "cfg_strength": float(cfg_strength),
        "cfg_interval": list(cfg_interval),
        "cfg_active": bool(active),
        "disabled_reference_max_abs": float(
            (disabled_reference.feats - stock_reference.feats).abs().max().item()
        ),
        "enabled_reference_max_abs": float(
            (enabled_reference.feats - stock_reference.feats).abs().max().item()
        ),
        "support_disabled_max_abs": float(
            (disabled.feats - stock_reference.feats).abs().max().item()
        ),
        "zero_init_enabled_max_abs": float(
            (enabled.feats - stock_reference.feats).abs().max().item()
        ),
        "expect_zero_init": bool(expect_zero_init),
    }
    report["passed"] = (
        report["disabled_reference_max_abs"] == 0.0
        and report["enabled_reference_max_abs"] == 0.0
        and report["support_disabled_max_abs"] == 0.0
        and (
            not expect_zero_init
            or report["zero_init_enabled_max_abs"] == 0.0
        )
    )
    if not report["passed"]:
        raise RuntimeError(f"Direct-SLAT post-CFG stock equivalence failed: {report}")
    return report


def gradient_group_norms(model: nn.Module) -> dict[str, float]:
    square_sums = {"support_adapter": 0.0, "lora": 0.0}
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        if name.startswith("support_adapter."):
            group = "support_adapter"
        elif "lora_" in name:
            group = "lora"
        else:
            continue
        square_sums[group] += float(parameter.grad.detach().float().square().sum().item())
    return {name: math.sqrt(value) for name, value in square_sums.items()}


def main() -> None:
    args = parse_args()
    validate_args(args)
    rollout_aligned_v3 = (
        args.training_semantics == DIRECT_SLAT_TRAINING_SEMANTICS_V3
    )
    rollout_endpoint_v4 = (
        args.training_semantics == DIRECT_SLAT_TRAINING_SEMANTICS_V4
    )
    branch_budget_v5 = (
        args.training_semantics == DIRECT_SLAT_TRAINING_SEMANTICS_V5
    )
    rollout_endpoint_native = rollout_endpoint_v4 or branch_budget_v5
    rollout_aligned = rollout_aligned_v3 or rollout_endpoint_native
    rollout_horizons = parse_rollout_horizons(args.rollout_horizons)
    native_schedule = (
        native_flow_timestep_sequence(
            steps=int(args.rollout_schedule_steps),
            rescale_t=float(args.rollout_rescale_t),
        )
        if rollout_endpoint_native
        else ()
    )
    train_cfg_interval = tuple(float(value) for value in args.train_cfg_interval)
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        dist.init_process_group(
            backend="nccl", timeout=datetime.timedelta(hours=12)
        )
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
    dataset = DirectSLatCacheDataset(
        args.cache_manifest,
        indices=args.indices,
        verify_hashes=bool(args.verify_cache_hashes),
    )
    if dataset.config.get("pretrained") != args.pretrained:
        raise RuntimeError(
            "cache native-condition pretrained binding differs from training runtime"
        )
    decoder_audit_path = Path(args.target_decoder_audit).resolve()
    decoder_audit = json.loads(decoder_audit_path.read_text(encoding="utf-8"))
    if decoder_audit.get("format") != (
        "pose_point_depth_mv.direct_slat_target_decoder_audit.v1"
    ):
        raise RuntimeError("unsupported target-decoder audit format")
    if decoder_audit.get("passed") is not True:
        raise RuntimeError("frozen native target-decoder audit did not pass")
    if str(decoder_audit.get("pretrained")) != str(args.pretrained):
        raise RuntimeError("target-decoder audit pretrained binding differs")
    if int(decoder_audit.get("summary", {}).get("object_count", 0)) < 32:
        raise RuntimeError("target-decoder audit must cover at least 32 objects")
    runtime_support_generator = support_generator_identity(dataset.config)
    if decoder_audit.get("support_generator") != runtime_support_generator:
        raise RuntimeError(
            "target-decoder audit and training cache use different frozen "
            "SS-support generation protocols"
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
        collate_fn=collate_direct_slat_one,
        pin_memory=True,
    )
    sampler, model, _, normalization, model_summary = build_direct_slat_components(
        pretrained=args.pretrained,
        adapter_hidden_dim=int(args.adapter_hidden_dim),
        lora_rank=int(args.lora_rank),
        lora_alpha=int(args.lora_alpha),
        gradient_checkpointing=bool(args.gradient_checkpointing),
        device=device,
    )
    runtime_normalization = {
        key: [float(item) for item in value] for key, value in normalization.items()
    }
    if canonical_json_sha256(runtime_normalization) != dataset.slat_normalization_hash:
        raise RuntimeError("runtime SLAT normalization differs from materialized cache")
    cache_identity = slat_target_cache_identity(
        args.cache_manifest, rows=dataset.rows
    )
    model_summary.update(
        {
            "cache_identity": cache_identity,
            "support_generator": runtime_support_generator,
            "target_decoder_audit": {
                "path": str(decoder_audit_path),
                "sha256": sha256_file(decoder_audit_path),
                "format": decoder_audit.get("format"),
                "summary": decoder_audit.get("summary"),
                "thresholds": decoder_audit.get("thresholds"),
            },
            "dataset_size": len(dataset),
            "unique_object_count": len(
                {str(row["object_uid"]) for row in dataset.rows}
            ),
            "slat_normalization": runtime_normalization,
            "slat_normalization_hash": dataset.slat_normalization_hash,
            "losses": {
                "flow_matching_mse": 1.0,
                "delta_norm": float(args.delta_norm_weight),
                "raw_delta_excess": float(args.raw_delta_excess_weight),
                "wrong_support_rank": float(args.wrong_support_rank_weight),
                "wrong_support_margin": float(args.wrong_support_margin),
                "support_dropout_stock_consistency": float(
                    args.support_dropout_weight
                ),
                "wrong_support_stock_reversion": float(
                    args.wrong_support_stock_weight
                ),
                "detached_rollout_velocity": float(
                    args.rollout_consistency_weight
                ),
                "endpoint_x0_target_slat": float(args.endpoint_x0_weight),
                "rollout_endpoint_full_over_stock_rank": float(
                    args.rollout_endpoint_rank_weight
                ),
                "rollout_endpoint_rank_margin": float(
                    args.rollout_endpoint_rank_margin
                ),
            },
            "training_semantics": {
                "version": str(args.training_semantics),
                "wrong_support_probability": float(args.wrong_support_probability),
                "support_dropout_probability": float(
                    args.support_dropout_probability
                ),
                "rollout_probability": float(args.rollout_probability),
                "rollout_step_size": float(args.rollout_step_size),
                "rollout_horizons": list(rollout_horizons),
                "rollout_supervision_policy": str(
                    args.rollout_supervision_policy
                ),
                "rollout_schedule_steps": int(args.rollout_schedule_steps),
                "rollout_rescale_t": float(args.rollout_rescale_t),
                "endpoint_x0_weight": float(args.endpoint_x0_weight),
                "rollout_endpoint_rank_weight": float(
                    args.rollout_endpoint_rank_weight
                ),
                "rollout_endpoint_rank_margin": float(
                    args.rollout_endpoint_rank_margin
                ),
                "train_cfg_strength": float(args.train_cfg_strength),
                "train_cfg_interval": [
                    float(value) for value in args.train_cfg_interval
                ],
                "wrong_support_identity": (
                    "object-disjoint; same support seed preferred; deterministic per micro-step"
                ),
                "wrong_support_counterfactual": (
                    "rollout-aligned semantics regress wrong-support Full to the same "
                    "post-CFG native stock reference"
                ),
                "support_dropout_counterfactual": (
                    "LoRA-only prediction is trained toward exact native stock; deployed "
                    "support-absent path remains an exact stock bypass"
                ),
                "rollout_state": (
                    "v3: one fixed Euler step; v4/v5: truncated native schedule, "
                    "each visited state detached before the next call; V5 "
                    "all_visited_v1 ranks every visited state, while V5.1 "
                    "terminal_only_constant_memory_v1 generates the identical "
                    "trajectory without intermediate graphs and ranks only the "
                    "selected terminal state"
                ),
                "auxiliary_event_partition": (
                    "rollout-aligned semantics sample wrong-support, support-dropout, "
                    "or rollout as mutually exclusive graphs to bound peak memory"
                ),
                "support_interval_policy": str(args.support_interval_policy),
            },
            "slat_delta_policy": {
                **model_summary["slat_delta_policy"],
                "training": {
                    "scale": float(args.slat_delta_scale),
                    "rms_ratio_cap": (
                        None
                        if float(args.slat_delta_rms_ratio_cap) < 0
                        else float(args.slat_delta_rms_ratio_cap)
                    ),
                    "per_sparse_batch": True,
                    "guided_delta_policy": str(args.slat_guided_delta_policy),
                    "bound_mode": str(args.slat_delta_bound_mode),
                    "support_interval_policy": str(
                        args.support_interval_policy
                    ),
                    "residual_combination_policy": str(
                        args.slat_residual_combination_policy
                    ),
                    "lora_branch": {
                        "scale": float(args.slat_lora_delta_scale),
                        "rms_ratio_cap": (
                            None
                            if float(args.slat_lora_delta_rms_ratio_cap) < 0
                            else float(args.slat_lora_delta_rms_ratio_cap)
                        ),
                    },
                    "support_increment_branch": {
                        "identity": "joint_full_minus_lora_only",
                        "scale": float(args.slat_support_delta_scale),
                        "rms_ratio_cap": (
                            None
                            if float(args.slat_support_delta_rms_ratio_cap) < 0
                            else float(args.slat_support_delta_rms_ratio_cap)
                        ),
                    },
                    "teacher_forward_scope": (
                        "positive-condition residual for legacy/v2; "
                        "v3/v4/v5 train the deployed post-CFG velocity"
                    ),
                    "legacy_default_exact": (
                        float(args.slat_delta_scale) == 1.0
                        and float(args.slat_delta_rms_ratio_cap) < 0
                    ),
                },
            },
            "sampling": {"mode": args.sampling_mode, "t_schedule": args.t_schedule},
            "max_slat_points": int(args.max_slat_points),
            "world_size": world_size,
            "resume_semantics": (
                "optimizer-boundary data cursor; noise/t/target subset derive "
                "deterministically from rank and micro_step; CUDA kernels are not "
                "claimed bit-exact"
            ),
        }
    )
    mean = torch.tensor(runtime_normalization["mean"], device=device)[None]
    std = torch.tensor(runtime_normalization["std"], device=device)[None]
    if bool(torch.any(std <= 0).item()):
        raise RuntimeError("SLAT normalization std must be positive")
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
    start_step = 0
    micro_step = 0
    resume_epoch = 0
    resume_samples_into_epoch = 0
    history: list[dict[str, Any]] = []
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        validate_resume(checkpoint, args, model_summary)
        load_strict_trainable_state(model, checkpoint["model_trainable_state"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint.get("scaler", {}))
        start_step = int(checkpoint.get("step", 0))
        micro_step = int(
            checkpoint.get("micro_step", start_step * int(args.grad_accum))
        )
        resume_epoch = int(checkpoint.get("epoch", 0))
        resume_samples_into_epoch = int(
            checkpoint.get("samples_into_epoch", 0)
        )
        history = list(checkpoint.get("history", []))
    if start_step >= int(args.max_steps):
        raise ValueError("resume already reached max_steps")

    model.eval()
    model_summary["stock_equivalence"] = stock_equivalence_audit(
        model=model,
        sampler=sampler,
        sample=dataset[0],
        mean=mean,
        std=std,
        device=device,
        expect_zero_init=not bool(args.resume),
        max_slat_points=int(args.max_slat_points),
    )
    if rollout_aligned:
        model_summary["post_cfg_stock_equivalence"] = (
            post_cfg_stock_equivalence_audit(
                model=model,
                sampler=sampler,
                sample=dataset[0],
                mean=mean,
                std=std,
                device=device,
                cfg_strength=float(args.train_cfg_strength),
                cfg_interval=train_cfg_interval,
                slat_delta_scale=float(args.slat_delta_scale),
                slat_delta_rms_ratio_cap=float(
                    args.slat_delta_rms_ratio_cap
                ),
                slat_delta_bound_mode=str(args.slat_delta_bound_mode),
                slat_residual_combination_policy=str(
                    args.slat_residual_combination_policy
                ),
                slat_lora_delta_scale=float(args.slat_lora_delta_scale),
                slat_lora_delta_rms_ratio_cap=float(
                    args.slat_lora_delta_rms_ratio_cap
                ),
                slat_support_delta_scale=float(args.slat_support_delta_scale),
                slat_support_delta_rms_ratio_cap=float(
                    args.slat_support_delta_rms_ratio_cap
                ),
                expect_zero_init=not bool(args.resume),
                max_slat_points=int(args.max_slat_points),
            )
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
    use_amp = args.amp_dtype != "none"
    amp_dtype = torch.float16 if args.amp_dtype == "fp16" else torch.bfloat16
    global_step = start_step
    epoch = resume_epoch
    wall_start = time.time()
    object_exposure: Counter[str] = Counter()
    mechanism_counts: Counter[str] = Counter()
    mechanism_sums: Counter[str] = Counter()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats(device)
    while global_step < int(args.max_steps):
        distributed_sampler.set_epoch(epoch)
        samples_into_epoch = 0
        for loader_position, sample in enumerate(loader):
            if epoch == resume_epoch and loader_position < resume_samples_into_epoch:
                continue
            samples_into_epoch = loader_position + 1
            if global_step >= int(args.max_steps):
                break
            object_exposure[str(sample["object_uid"])] += 1
            random_seed = process_seed * 1000003 + micro_step * 1013
            with torch.no_grad():
                if rollout_aligned:
                    auxiliary_event = deterministic_probability_partition(
                        seed=random_seed + 41,
                        probabilities=(
                            float(args.wrong_support_probability),
                            float(args.support_dropout_probability),
                            float(args.rollout_probability),
                        ),
                    )
                    wrong_support_event = auxiliary_event == 0
                    support_dropout_event = auxiliary_event == 1
                    rollout_event = auxiliary_event == 2
                else:
                    wrong_support_event = deterministic_probability_event(
                        seed=random_seed + 41,
                        probability=float(args.wrong_support_probability),
                    )
                    support_dropout_event = deterministic_probability_event(
                        seed=random_seed + 43,
                        probability=float(args.support_dropout_probability),
                    )
                    rollout_event = False

                target = normalized_target(
                    sample,
                    mean=mean,
                    std=std,
                    device=device,
                    max_points=int(args.max_slat_points),
                    selection_seed=random_seed + 11,
                )
                noise_generator = torch.Generator(device=device).manual_seed(
                    random_seed + 17
                )
                noise = sp.SparseTensor(
                    feats=torch.randn(
                        target.feats.shape,
                        generator=noise_generator,
                        device=device,
                        dtype=target.feats.dtype,
                    ),
                    coords=target.coords,
                )
                rollout_previous_t_values: tuple[float, ...] = ()
                if rollout_endpoint_native:
                    requested_horizon = (
                        random.Random(random_seed + 53).choice(
                            rollout_horizons
                        )
                        if rollout_event
                        else 0
                    )
                    eligible_indices = []
                    for schedule_index, schedule_t in enumerate(
                        native_schedule[:-1]
                    ):
                        if not cfg_interval_is_active(
                            schedule_t, train_cfg_interval
                        ):
                            continue
                        if requested_horizon:
                            end = schedule_index + requested_horizon
                            if end >= len(native_schedule):
                                continue
                            visited = native_schedule[
                                schedule_index + 1 : end + 1
                            ]
                            if not all(
                                cfg_interval_is_active(
                                    value, train_cfg_interval
                                )
                                for value in visited
                            ):
                                continue
                        eligible_indices.append(schedule_index)
                    if not eligible_indices:
                        raise RuntimeError(
                            "V4 native schedule has no timestep compatible "
                            "with the configured CFG interval/horizon"
                        )
                    schedule_index = random.Random(random_seed + 59).choice(
                        eligible_indices
                    )
                    t_value = torch.tensor(
                        native_schedule[schedule_index],
                        device=device,
                        dtype=torch.float32,
                    )
                    if requested_horizon:
                        rollout_previous_t_values = tuple(
                            native_schedule[
                                schedule_index + 1 :
                                schedule_index + requested_horizon + 1
                            ]
                        )
                else:
                    with torch.random.fork_rng(devices=[local_rank]):
                        torch.cuda.manual_seed(random_seed + 31)
                        t_value = sample_t(str(args.t_schedule), device)
                x_t, gt_velocity = sampler._get_model_gt(target, t_value, noise)
                t_tensor = torch.full(
                    (1,), 1000.0 * t_value, device=device, dtype=torch.float32
                )
                condition = to_device_tree(sample["condition"]["cond"], device)
                negative_condition = (
                    to_device_tree(sample["condition"]["neg_cond"], device)
                    if rollout_aligned
                    else None
                )
                corrected_ss = sample["corrected_ss"].to(device=device, non_blocking=True)
                occupancy = sample["occupancy_logits64"].to(
                    device=device, non_blocking=True
                )
                physical = sample["physical_tokens16"].to(
                    device=device, non_blocking=True
                )
                post_cfg_active = (
                    rollout_aligned
                    and cfg_interval_is_active(t_value, train_cfg_interval)
                )
                rollout_t_value = max(
                    0.0, float(t_value) - float(args.rollout_step_size)
                )
                rollout_cfg_active = (
                    rollout_event
                    and cfg_interval_is_active(
                        rollout_t_value,
                        train_cfg_interval,
                    )
                )
                rollout_cfg_active_values = tuple(
                    cfg_interval_is_active(value, train_cfg_interval)
                    for value in rollout_previous_t_values
                )
                rollout_support_active_values = tuple(
                    (
                        True
                        if args.support_interval_policy
                        != SLAT_SUPPORT_INTERVAL_CFG_ACTIVE
                        else active
                    )
                    for active in rollout_cfg_active_values
                )
                wrong_sample = None
                wrong_support_index = -1
                wrong_support = None
                if wrong_support_event:
                    wrong_support_index = deterministic_wrong_support_index(
                        dataset.rows,
                        correct_object_uid=str(sample["object_uid"]),
                        support_seed=int(sample["support_seed"]),
                        selection_seed=random_seed + 47,
                    )
                    wrong_sample = dataset[wrong_support_index]
                    if str(wrong_sample["object_uid"]) == str(sample["object_uid"]):
                        raise RuntimeError("wrong-support selector returned the correct object")
                    wrong_support = (
                        wrong_sample["corrected_ss"].to(device=device),
                        wrong_sample["occupancy_logits64"].to(device=device),
                        wrong_sample["physical_tokens16"].to(device=device),
                    )
            sync_step = ((micro_step + 1) % int(args.grad_accum)) == 0
            sync_context = (
                wrapped.no_sync()
                if world_size > 1 and not sync_step
                else nullcontext()
            )
            with sync_context:
                with torch.autocast(
                    device_type="cuda", dtype=amp_dtype, enabled=use_amp
                ):
                    with torch.no_grad():
                        stock_positive = model.stock_prediction(
                            x_t, t_tensor, condition
                        )
                        stock_negative = (
                            model.stock_prediction(
                                x_t, t_tensor, negative_condition
                            )
                            if post_cfg_active
                            else None
                        )
                        stock = (
                            combine_sparse_cfg(
                                stock_positive,
                                stock_negative,
                                cfg_strength=float(args.train_cfg_strength),
                            )
                            if stock_negative is not None
                            else stock_positive
                        )
                    prediction, stats, wrong_prediction, lora_without_support = wrapped(
                        x_t,
                        t_tensor,
                        condition,
                        corrected_ss=corrected_ss,
                        occupancy_logits64=occupancy,
                        physical_tokens16=physical,
                        stock_velocity=stock_positive,
                        support_scale=float(args.support_scale),
                        slat_delta_scale=float(args.slat_delta_scale),
                        slat_delta_rms_ratio_cap=(
                            None
                            if float(args.slat_delta_rms_ratio_cap) < 0
                            else float(args.slat_delta_rms_ratio_cap)
                        ),
                        slat_delta_bound_mode=str(
                            args.slat_delta_bound_mode
                        ),
                        slat_residual_combination_policy=str(
                            args.slat_residual_combination_policy
                        ),
                        slat_lora_delta_scale=float(
                            args.slat_lora_delta_scale
                        ),
                        slat_lora_delta_rms_ratio_cap=float(
                            args.slat_lora_delta_rms_ratio_cap
                        ),
                        slat_support_delta_scale=float(
                            args.slat_support_delta_scale
                        ),
                        slat_support_delta_rms_ratio_cap=float(
                            args.slat_support_delta_rms_ratio_cap
                        ),
                        support_active=(
                            True
                            if args.support_interval_policy
                            != SLAT_SUPPORT_INTERVAL_CFG_ACTIVE
                            else bool(post_cfg_active)
                        ),
                        wrong_corrected_ss=(
                            wrong_support[0]
                            if wrong_support_event and wrong_support is not None
                            else None
                        ),
                        wrong_occupancy_logits64=(
                            wrong_support[1]
                            if wrong_support_event and wrong_support is not None
                            else None
                        ),
                        wrong_physical_tokens16=(
                            wrong_support[2]
                            if wrong_support_event and wrong_support is not None
                            else None
                        ),
                        compute_lora_without_support=bool(support_dropout_event),
                        negative_condition=negative_condition,
                        stock_negative_velocity=stock_negative,
                        post_cfg_strength=float(args.train_cfg_strength),
                        post_cfg_active=bool(post_cfg_active),
                        rollout_step_size=(
                            float(args.rollout_step_size)
                            if rollout_event
                            else 0.0
                        ),
                        rollout_cfg_active=bool(rollout_cfg_active),
                        rollout_previous_t_values=rollout_previous_t_values,
                        rollout_cfg_active_values=rollout_cfg_active_values,
                        rollout_support_active_values=(
                            rollout_support_active_values
                        ),
                        rollout_supervision_policy=str(
                            args.rollout_supervision_policy
                        ),
                    )
                    stock_reference = stats["_stock_reference"]
                    if not torch.equal(stock_reference.feats, stock.feats):
                        raise RuntimeError(
                            "training loop/model post-CFG stock references differ"
                        )
                    flow_loss = F.mse_loss(
                        prediction.feats.float(), gt_velocity.feats.float()
                    )
                    stock_loss = F.mse_loss(
                        stock.feats.float(), gt_velocity.feats.float()
                    )
                    delta = prediction.feats.float() - stock.feats.float()
                    delta_norm = delta.square().mean()
                    zero_loss = flow_loss.new_zeros(())
                    if float(args.slat_delta_rms_ratio_cap) >= 0:
                        (
                            raw_delta_excess_loss,
                            raw_delta_ratio,
                            raw_delta_excess,
                        ) = stock_relative_residual_excess_loss(
                            stock_rms_per_batch=stats[
                                "stock_velocity_rms_per_batch"
                            ],
                            raw_delta_rms_per_batch=stats[
                                "raw_flow_delta_rms_per_batch"
                            ],
                            ratio_cap=float(args.slat_delta_rms_ratio_cap),
                        )
                    else:
                        raw_delta_excess_loss = zero_loss
                        raw_delta_ratio = zero_loss.reshape(1)
                        raw_delta_excess = zero_loss.reshape(1)

                    wrong_support_loss = zero_loss
                    support_rank_loss = zero_loss
                    correct_over_wrong_advantage = zero_loss
                    wrong_support_stock_loss = zero_loss
                    if wrong_support_event:
                        if wrong_sample is None:
                            raise AssertionError("wrong-support event lacks a sample")
                        if wrong_prediction is None:
                            raise AssertionError("wrong-support forward produced no prediction")
                        wrong_support_loss = F.mse_loss(
                            wrong_prediction.feats.float(), gt_velocity.feats.float()
                        )
                        (
                            support_rank_loss,
                            correct_over_wrong_advantage,
                        ) = correct_over_wrong_support_rank_loss(
                            correct_loss=flow_loss,
                            wrong_loss=wrong_support_loss,
                            margin=float(args.wrong_support_margin),
                        )
                        wrong_support_stock_loss = F.mse_loss(
                            wrong_prediction.feats.float(),
                            stock.feats.float(),
                        )

                    support_dropout_loss = zero_loss
                    if support_dropout_event:
                        if lora_without_support is None:
                            raise AssertionError(
                                "support-dropout forward produced no counterfactual"
                            )
                        support_dropout_loss = F.mse_loss(
                            lora_without_support.feats.float(), stock.feats.float()
                        )

                    rollout_loss = zero_loss
                    rollout_stock_loss = zero_loss
                    endpoint_x0_loss = zero_loss
                    endpoint_stock_x0_loss = zero_loss
                    rollout_endpoint_rank_loss = zero_loss
                    rollout_endpoint_advantage = zero_loss
                    if rollout_event:
                        if rollout_endpoint_native:
                            rollout_predictions = stats.get(
                                "_rollout_predictions"
                            )
                            rollout_stock_references = stats.get(
                                "_rollout_stock_references"
                            )
                            rollout_states = stats.get("_rollout_states")
                            rollout_t_values = stats.get("_rollout_t_values")
                            if not (
                                rollout_predictions
                                and rollout_stock_references
                                and rollout_states
                                and rollout_t_values
                            ):
                                raise AssertionError(
                                    "native-schedule rollout produced no trace"
                                )
                            if (
                                args.rollout_supervision_policy
                                == SLAT_ROLLOUT_SUPERVISION_TERMINAL_CONSTANT_MEMORY
                                and not (
                                    len(rollout_predictions)
                                    == len(rollout_stock_references)
                                    == len(rollout_states)
                                    == len(rollout_t_values)
                                    == 1
                                )
                            ):
                                raise AssertionError(
                                    "constant-memory rollout must expose exactly "
                                    "one supervised terminal state"
                                )
                            rollout_loss = torch.stack(
                                [
                                    F.mse_loss(
                                        item.feats.float(),
                                        gt_velocity.feats.float(),
                                    )
                                    for item in rollout_predictions
                                ]
                            ).mean()
                            rollout_stock_loss = torch.stack(
                                [
                                    F.mse_loss(
                                        item.feats.float(),
                                        gt_velocity.feats.float(),
                                    )
                                    for item in rollout_stock_references
                                ]
                            ).mean()
                            if branch_budget_v5:
                                full_endpoint_losses = []
                                stock_endpoint_losses = []
                                for state, full_item, stock_item, item_t in zip(
                                    rollout_states,
                                    rollout_predictions,
                                    rollout_stock_references,
                                    rollout_t_values,
                                ):
                                    full_endpoint_losses.append(
                                        F.mse_loss(
                                            state.feats.float()
                                            - float(item_t)
                                            * full_item.feats.float(),
                                            target.feats.float(),
                                        )
                                    )
                                    stock_endpoint_losses.append(
                                        F.mse_loss(
                                            state.feats.float()
                                            - float(item_t)
                                            * stock_item.feats.float(),
                                            target.feats.float(),
                                        )
                                    )
                                endpoint_x0_loss = torch.stack(
                                    full_endpoint_losses
                                ).mean()
                                endpoint_stock_x0_loss = torch.stack(
                                    stock_endpoint_losses
                                ).mean()
                                rollout_endpoint_advantage = (
                                    endpoint_stock_x0_loss - endpoint_x0_loss
                                )
                                rollout_endpoint_rank_loss = torch.relu(
                                    endpoint_x0_loss
                                    - endpoint_stock_x0_loss
                                    + float(args.rollout_endpoint_rank_margin)
                                )
                            else:
                                final_state = rollout_states[-1]
                                final_prediction = rollout_predictions[-1]
                                final_t = float(rollout_t_values[-1])
                                endpoint_x0 = (
                                    final_state.feats.float()
                                    - final_t * final_prediction.feats.float()
                                )
                                endpoint_x0_loss = F.mse_loss(
                                    endpoint_x0,
                                    target.feats.float(),
                                )
                        else:
                            rollout_prediction = stats.get(
                                "_rollout_prediction"
                            )
                            rollout_stock_reference = stats.get(
                                "_rollout_stock_reference"
                            )
                            if (
                                rollout_prediction is None
                                or rollout_stock_reference is None
                            ):
                                raise AssertionError(
                                    "rollout event produced no detached-state prediction"
                                )
                            rollout_loss = F.mse_loss(
                                rollout_prediction.feats.float(),
                                gt_velocity.feats.float(),
                            )
                            rollout_stock_loss = F.mse_loss(
                                rollout_stock_reference.feats.float(),
                                gt_velocity.feats.float(),
                            )

                    loss = (
                        flow_loss
                        + float(args.delta_norm_weight) * delta_norm
                        + float(args.raw_delta_excess_weight)
                        * raw_delta_excess_loss
                        + float(args.wrong_support_rank_weight) * support_rank_loss
                        + float(args.wrong_support_stock_weight)
                        * wrong_support_stock_loss
                        + float(args.support_dropout_weight) * support_dropout_loss
                        + float(args.rollout_consistency_weight) * rollout_loss
                        + float(args.endpoint_x0_weight) * endpoint_x0_loss
                        + float(args.rollout_endpoint_rank_weight)
                        * rollout_endpoint_rank_loss
                    )
                    scaled_loss = loss / int(args.grad_accum)
                scaler.scale(scaled_loss).backward()
            mechanism_counts["micro_steps"] += 1
            mechanism_counts["wrong_support_events"] += int(wrong_support_event)
            mechanism_counts["support_dropout_events"] += int(support_dropout_event)
            mechanism_counts["rollout_events"] += int(rollout_event)
            mechanism_counts["post_cfg_teacher_events"] += int(post_cfg_active)
            mechanism_counts["rank_passes"] += int(
                wrong_support_event
                and float(correct_over_wrong_advantage.detach().item())
                >= float(args.wrong_support_margin)
            )
            mechanism_sums["raw_delta_excess_loss"] += float(
                raw_delta_excess_loss.detach().item()
            )
            mechanism_sums["raw_delta_ratio_max"] += float(
                raw_delta_ratio.detach().amax().item()
            )
            if wrong_support_event:
                mechanism_sums["correct_over_wrong_advantage"] += float(
                    correct_over_wrong_advantage.detach().item()
                )
                mechanism_sums["support_rank_loss"] += float(
                    support_rank_loss.detach().item()
                )
                mechanism_sums["wrong_support_stock_loss"] += float(
                    wrong_support_stock_loss.detach().item()
                )
            if support_dropout_event:
                mechanism_sums["support_dropout_loss"] += float(
                    support_dropout_loss.detach().item()
                )
            if rollout_event:
                mechanism_sums["rollout_loss"] += float(
                    rollout_loss.detach().item()
                )
                mechanism_sums["rollout_gain_vs_stock"] += float(
                    (rollout_stock_loss - rollout_loss).detach().item()
                )
                mechanism_sums["endpoint_x0_loss"] += float(
                    endpoint_x0_loss.detach().item()
                )
                mechanism_sums["endpoint_stock_x0_loss"] += float(
                    endpoint_stock_x0_loss.detach().item()
                )
                mechanism_sums["rollout_endpoint_rank_loss"] += float(
                    rollout_endpoint_rank_loss.detach().item()
                )
                mechanism_sums["rollout_endpoint_advantage"] += float(
                    rollout_endpoint_advantage.detach().item()
                )
                mechanism_counts["rollout_endpoint_rank_passes"] += int(
                    branch_budget_v5
                    and float(rollout_endpoint_rank_loss.detach().item()) == 0.0
                )
                mechanism_sums["rollout_horizon"] += float(
                    len(rollout_previous_t_values)
                    if rollout_endpoint_native
                    else 1
                )
                mechanism_sums["rollout_supervised_state_count"] += float(
                    stats.get(
                        "rollout_supervised_state_count",
                        rollout_loss.new_tensor(1.0),
                    )
                    .detach()
                    .item()
                )
            micro_step += 1
            if not sync_step:
                continue
            if not gradients_finite(trainable):
                raise FloatingPointError("non-finite direct SLAT gradients")
            scaler.unscale_(optimizer)
            group_grad_norms = gradient_group_norms(model)
            grad_norm = float(
                torch.nn.utils.clip_grad_norm_(trainable, float(args.grad_clip)).item()
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            row = {
                "step": global_step,
                "micro_step": micro_step,
                "uid": str(sample["uid"]),
                "object_uid": str(sample["object_uid"]),
                "support_seed": int(sample["support_seed"]),
                "t": float(t_value),
                "loss": float(loss.detach().item()),
                "flow_loss": float(flow_loss.detach().item()),
                "stock_loss": float(stock_loss.detach().item()),
                "gain_vs_stock": float((stock_loss - flow_loss).detach().item()),
                "delta_norm": float(delta_norm.detach().item()),
                "raw_delta_excess_loss": float(
                    raw_delta_excess_loss.detach().item()
                ),
                "raw_delta_ratio_max": float(raw_delta_ratio.detach().amax().item()),
                "raw_delta_excess_max": float(
                    raw_delta_excess.detach().amax().item()
                ),
                "wrong_support_evaluated": bool(wrong_support_event),
                "wrong_support_uid": (
                    str(wrong_sample["uid"]) if wrong_sample is not None else ""
                ),
                "wrong_support_object_uid": (
                    str(wrong_sample["object_uid"])
                    if wrong_sample is not None
                    else ""
                ),
                "wrong_support_loss": float(wrong_support_loss.detach().item()),
                "wrong_support_stock_loss": float(
                    wrong_support_stock_loss.detach().item()
                ),
                "correct_over_wrong_advantage": float(
                    correct_over_wrong_advantage.detach().item()
                ),
                "support_rank_loss": float(support_rank_loss.detach().item()),
                "support_dropout_evaluated": bool(support_dropout_event),
                "support_dropout_loss": float(support_dropout_loss.detach().item()),
                "post_cfg_teacher": bool(post_cfg_active),
                "applied_cfg_strength": float(
                    stats.get(
                        "applied_cfg_strength",
                        flow_loss.new_tensor(1.0),
                    ).detach().item()
                ),
                "rollout_evaluated": bool(rollout_event),
                "rollout_t": (
                    float(stats["rollout_t"].detach().item())
                    if rollout_event
                    else None
                ),
                "rollout_step_size": (
                    float(stats["rollout_step_size"].detach().item())
                    if rollout_event
                    else 0.0
                ),
                "rollout_horizon": (
                    int(len(rollout_previous_t_values))
                    if rollout_endpoint_native and rollout_event
                    else (1 if rollout_event else 0)
                ),
                "rollout_supervised_state_count": (
                    int(
                        stats["rollout_supervised_state_count"]
                        .detach()
                        .item()
                    )
                    if rollout_endpoint_native and rollout_event
                    else (1 if rollout_event else 0)
                ),
                "rollout_loss": float(rollout_loss.detach().item()),
                "rollout_stock_loss": float(rollout_stock_loss.detach().item()),
                "endpoint_x0_loss": float(endpoint_x0_loss.detach().item()),
                "endpoint_stock_x0_loss": float(
                    endpoint_stock_x0_loss.detach().item()
                ),
                "rollout_endpoint_rank_loss": float(
                    rollout_endpoint_rank_loss.detach().item()
                ),
                "rollout_endpoint_advantage": float(
                    rollout_endpoint_advantage.detach().item()
                ),
                "rollout_gain_vs_stock": float(
                    (rollout_stock_loss - rollout_loss).detach().item()
                ),
                "support_token_rms": float(stats["support_token_rms"].detach().item()),
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
                "delta_clip_scale": float(
                    stats["delta_clip_scale"].detach().item()
                ),
                "delta_clip_activated": bool(
                    stats["delta_clip_activated"].detach().item() > 0.5
                ),
                "slat_delta_bound_mode": str(args.slat_delta_bound_mode),
                "slat_residual_combination_policy": str(
                    args.slat_residual_combination_policy
                ),
                "lora_branch_effective_delta_rms": float(
                    stats.get(
                        "lora_branch_effective_flow_delta_rms",
                        flow_loss.new_zeros(()),
                    ).detach().item()
                ),
                "support_branch_effective_delta_rms": float(
                    stats.get(
                        "support_branch_effective_flow_delta_rms",
                        flow_loss.new_zeros(()),
                    ).detach().item()
                ),
                "branch_delta_cosine": float(
                    stats.get(
                        "branch_delta_cosine",
                        flow_loss.new_zeros(()),
                    ).detach().item()
                ),
                "support_interval_policy": str(
                    args.support_interval_policy
                ),
                "raw_flow_delta_abs_max": float(
                    stats["raw_flow_delta_abs_max"].detach().item()
                ),
                "effective_flow_delta_abs_max": float(
                    stats["effective_flow_delta_abs_max"].detach().item()
                ),
                "grad_norm": grad_norm,
                "support_adapter_grad_norm": group_grad_norms["support_adapter"],
                "lora_grad_norm": group_grad_norms["lora"],
                "cuda_peak_allocated_mib": (
                    float(torch.cuda.max_memory_allocated(device)) / (1024.0**2)
                ),
                "cuda_peak_reserved_mib": (
                    float(torch.cuda.max_memory_reserved(device)) / (1024.0**2)
                ),
                "elapsed_seconds": time.time() - wall_start,
            }
            history.append(row)
            if rank == 0 and (
                global_step == 1 or global_step % int(args.log_every) == 0
            ):
                print(f"[direct_slat_train] {json.dumps(row)}", flush=True)
            if rank == 0 and (
                global_step % int(args.save_every) == 0
                or global_step == int(args.max_steps)
            ):
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
                save_checkpoint(
                    output_dir / "checkpoints" / "last.pt",
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
            torch.cuda.reset_peak_memory_stats(device)
        epoch += 1
        resume_samples_into_epoch = 0
    model.eval()
    post_training_stock_equivalence = stock_equivalence_audit(
        model=model,
        sampler=sampler,
        sample=dataset[0],
        mean=mean,
        std=std,
        device=device,
        expect_zero_init=False,
        max_slat_points=int(args.max_slat_points),
    )
    post_training_post_cfg_stock_equivalence = None
    if rollout_aligned:
        post_training_post_cfg_stock_equivalence = (
            post_cfg_stock_equivalence_audit(
                model=model,
                sampler=sampler,
                sample=dataset[0],
                mean=mean,
                std=std,
                device=device,
                cfg_strength=float(args.train_cfg_strength),
                cfg_interval=train_cfg_interval,
                slat_delta_scale=float(args.slat_delta_scale),
                slat_delta_rms_ratio_cap=float(
                    args.slat_delta_rms_ratio_cap
                ),
                slat_delta_bound_mode=str(args.slat_delta_bound_mode),
                slat_residual_combination_policy=str(
                    args.slat_residual_combination_policy
                ),
                slat_lora_delta_scale=float(args.slat_lora_delta_scale),
                slat_lora_delta_rms_ratio_cap=float(
                    args.slat_lora_delta_rms_ratio_cap
                ),
                slat_support_delta_scale=float(args.slat_support_delta_scale),
                slat_support_delta_rms_ratio_cap=float(
                    args.slat_support_delta_rms_ratio_cap
                ),
                expect_zero_init=False,
                max_slat_points=int(args.max_slat_points),
            )
        )
    local_audit = {
        "counts": dict(mechanism_counts),
        "sums": dict(mechanism_sums),
        "object_exposure": dict(object_exposure),
    }
    gathered_audits = [local_audit]
    if world_size > 1:
        gathered_audits = [None for _ in range(world_size)]
        dist.all_gather_object(gathered_audits, local_audit)
    global_counts: Counter[str] = Counter()
    global_sums: Counter[str] = Counter()
    global_object_exposure: Counter[str] = Counter()
    for audit in gathered_audits:
        if audit is None:
            raise RuntimeError("missing distributed Direct-SLAT training audit")
        global_counts.update(audit["counts"])
        global_sums.update(audit["sums"])
        global_object_exposure.update(audit["object_exposure"])
    wrong_events = int(global_counts["wrong_support_events"])
    dropout_events = int(global_counts["support_dropout_events"])
    rollout_events = int(global_counts["rollout_events"])
    total_micro_steps = int(global_counts["micro_steps"])
    mechanism_training_audit = {
        "counts": dict(global_counts),
        "means": {
            "raw_delta_excess_loss": (
                float(global_sums["raw_delta_excess_loss"]) / total_micro_steps
                if total_micro_steps
                else 0.0
            ),
            "raw_delta_ratio_max": (
                float(global_sums["raw_delta_ratio_max"]) / total_micro_steps
                if total_micro_steps
                else 0.0
            ),
            "correct_over_wrong_advantage": (
                float(global_sums["correct_over_wrong_advantage"]) / wrong_events
                if wrong_events
                else 0.0
            ),
            "support_rank_loss": (
                float(global_sums["support_rank_loss"]) / wrong_events
                if wrong_events
                else 0.0
            ),
            "support_dropout_loss": (
                float(global_sums["support_dropout_loss"]) / dropout_events
                if dropout_events
                else 0.0
            ),
            "wrong_support_stock_loss": (
                float(global_sums["wrong_support_stock_loss"]) / wrong_events
                if wrong_events
                else 0.0
            ),
            "rollout_loss": (
                float(global_sums["rollout_loss"]) / rollout_events
                if rollout_events
                else 0.0
            ),
            "rollout_gain_vs_stock": (
                float(global_sums["rollout_gain_vs_stock"]) / rollout_events
                if rollout_events
                else 0.0
            ),
            "endpoint_x0_loss": (
                float(global_sums["endpoint_x0_loss"]) / rollout_events
                if rollout_events
                else 0.0
            ),
            "endpoint_stock_x0_loss": (
                float(global_sums["endpoint_stock_x0_loss"]) / rollout_events
                if rollout_events
                else 0.0
            ),
            "rollout_endpoint_rank_loss": (
                float(global_sums["rollout_endpoint_rank_loss"])
                / rollout_events
                if rollout_events
                else 0.0
            ),
            "rollout_endpoint_advantage": (
                float(global_sums["rollout_endpoint_advantage"])
                / rollout_events
                if rollout_events
                else 0.0
            ),
            "rollout_horizon": (
                float(global_sums["rollout_horizon"]) / rollout_events
                if rollout_events
                else 0.0
            ),
            "rollout_supervised_state_count": (
                float(global_sums["rollout_supervised_state_count"])
                / rollout_events
                if rollout_events
                else 0.0
            ),
        },
        "correct_over_wrong_margin": float(args.wrong_support_margin),
        "rank_pass_rate": (
            float(global_counts["rank_passes"]) / wrong_events
            if wrong_events
            else 0.0
        ),
        "rollout_endpoint_rank_pass_rate": (
            float(global_counts["rollout_endpoint_rank_passes"])
            / rollout_events
            if rollout_events and branch_budget_v5
            else 0.0
        ),
    }
    if rank == 0:
        report = {
            "format": DIRECT_SLAT_FLOW_VERSION,
            "completed": True,
            "step": global_step,
            "micro_step": micro_step,
            "world_size": world_size,
            "model_summary": model_summary,
            "object_exposure": {
                "unique_objects_seen": len(global_object_exposure),
                "per_object": dict(sorted(global_object_exposure.items())),
            },
            "post_training_stock_equivalence": post_training_stock_equivalence,
            "post_training_post_cfg_stock_equivalence": (
                post_training_post_cfg_stock_equivalence
            ),
            "mechanism_training_audit": mechanism_training_audit,
            "history": history,
        }
        (output_dir / "report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
