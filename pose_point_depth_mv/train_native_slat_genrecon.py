#!/usr/bin/env python3
"""Train Native-SLAT v2 or v3 on a materialized Native-SS support cache."""

from __future__ import annotations

import argparse
import copy
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
from torch.utils.data import DataLoader

from pose_point_depth_mv.direct_slat_flow import support_generator_identity
from pose_point_depth_mv.evaluate_native_ss_stock_slat_mesh import load_ss_evidence
from pose_point_depth_mv.native_3d_condition import (
    NativeConditionSLatDataset,
    collate_native_one,
)
from pose_point_depth_mv.native_slat_genrecon import (
    NATIVE_SLAT_GENRECON_VERSION,
    NativeSLatGenreconFlow,
    build_native_slat_genrecon_components,
    canonical_json_sha256,
    load_stock_slat_freeze,
    load_trainable_state_dict,
    optimizer_parameter_groups,
    sha256_file,
    trainable_state_dict,
    validate_native_slat_genrecon_checkpoint,
)
from pose_point_depth_mv.native_slat_genrecon_v2 import (
    NATIVE_SLAT_GENRECON_V2_VERSION,
    build_native_slat_genrecon_v2_components,
    load_trainable_state_dict as load_v2_trainable_state_dict,
    optimizer_parameter_groups as v2_optimizer_parameter_groups,
    trainable_state_dict as v2_trainable_state_dict,
    validate_native_slat_genrecon_v2_checkpoint,
)
from pose_point_depth_mv.train_direct_flow import ObjectBalancedDistributedSampler
from pose_point_depth_mv.train_direct_slat_flow import normalized_target, to_device_tree
from reconvggt_ar_adapter_a.train_pointpose_ss_lora import (
    finite_tree,
    gradients_finite,
    optimizer_state_finite,
    parameters_finite,
)
from trellis.modules import sparse as sp


STOCK_CONTEXT_VIEW_MODES = ("all", "first")
APPROVED_RESUME_DATA_PATH_FIELDS = (
    ("cache_manifest",),
    ("lifting_cache_manifest",),
    ("native_ss", "report"),
    ("native_ss", "checkpoint"),
    ("target_decoder_audit", "path"),
)
APPROVED_RELOCATED_PATH = "<APPROVED_RELOCATED_PATH>"
APPROVED_NATIVE_SS_DEPLOYMENT_PATH_FIELDS = ("report", "checkpoint")


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture", choices=("v2", "v3"), default="v3")
    parser.add_argument("--cache_manifest", required=True)
    parser.add_argument("--lifting_cache_manifest", required=True)
    parser.add_argument("--target_decoder_audit", required=True)
    parser.add_argument("--native_ss_report", required=True)
    parser.add_argument("--stock_slat_freeze", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--pretrained", default="Stable-X/trellis-vggt-v0-2")
    parser.add_argument("--indices", default="all")
    parser.add_argument("--resume", default="")
    parser.add_argument(
        "--init_checkpoint",
        default="",
        help="initialize a new data identity from raw/EMA trainable weights",
    )
    parser.add_argument("--init_weights", choices=("raw", "ema"), default="ema")
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument(
        "--run_until_step",
        type=int,
        default=0,
        help=(
            "Operational stage boundary in (0, max_steps]; 0 runs to max_steps. "
            "This is deliberately excluded from checkpoint identity so a staged "
            "run can resume without changing the 2000-step training contract."
        ),
    )
    parser.add_argument("--save_every", type=int, default=200)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--condition_channels", type=int, default=1024)
    parser.add_argument("--view_fusion_hidden_dim", type=int, default=64)
    parser.add_argument("--geometry_logit_scale_init", type=float, default=1.0)
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
    parser.add_argument(
        "--t_schedule",
        choices=("logit_normal", "uniform"),
        default="logit_normal",
        help=(
            "Flow-matching timestep distribution. uniform is exposed for the "
            "official-SLat target-domain control and is checkpoint-bound."
        ),
    )
    parser.add_argument("--min_condition_views", type=int, default=1)
    parser.add_argument("--max_condition_views", type=int, default=16)
    parser.add_argument(
        "--stock_context_views",
        choices=STOCK_CONTEXT_VIEW_MODES,
        default="all",
        help=(
            "Views passed to the native Stock SLat cross-attention path. The posed-DINO "
            "3D condition always keeps every randomly selected view."
        ),
    )
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--verify_cache_hashes", action="store_true")
    parser.add_argument(
        "--allow_resume_max_steps_extension",
        action="store_true",
        help=(
            "Explicitly allow a resume checkpoint to extend max_steps. Only a "
            "strict increase is accepted; all other training identity fields remain "
            "strictly checked."
        ),
    )
    parser.add_argument(
        "--allow_resume_topology_change",
        action="store_true",
        help=(
            "Explicitly allow world_size/grad_accum to change on resume, but only "
            "when the global effective batch is unchanged. This preserves optimizer "
            "semantics but does not claim an identical per-rank sample/RNG stream."
        ),
    )
    parser.add_argument(
        "--allow_resume_data_path_relocation",
        action="store_true",
        help=(
            "Explicitly allow only the approved data-identity path strings to "
            "change on resume. Saved and current paths must resolve strictly to "
            "the same existing filesystem objects; every non-path identity field "
            "remains exact."
        ),
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.resume and args.init_checkpoint:
        raise ValueError("--resume and --init_checkpoint are mutually exclusive")
    positive = (
        "max_steps",
        "save_every",
        "log_every",
        "grad_accum",
        "lora_rank",
        "lora_alpha",
        "condition_channels",
        "view_fusion_hidden_dim",
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
    if int(args.condition_channels) != 1024:
        raise ValueError("Native-SLAT v3 binds DINO/GenReCon condition_channels=1024")
    if args.architecture == "v3" and not 0.0 <= float(
        args.geometry_logit_scale_init
    ) <= 4.0:
        raise ValueError("geometry_logit_scale_init must be in [0,4]")
    if not 0.0 <= float(args.p_uncond) < 1.0:
        raise ValueError("p_uncond must be in [0,1)")
    if int(args.min_condition_views) > int(args.max_condition_views):
        raise ValueError("min_condition_views exceeds max_condition_views")
    if int(args.run_until_step) < 0 or int(args.run_until_step) > int(args.max_steps):
        raise ValueError("run_until_step must be 0 or lie in [1,max_steps]")
    if args.architecture != "v2" and args.stock_context_views != "all":
        raise ValueError("first-view Stock context is frozen to the v2 2x2 experiment")
    if int(args.warmup_steps) < -1 or not 0.0 <= float(args.warmup_ratio) <= 1.0:
        raise ValueError("invalid warmup configuration")
    if not 0.0 < float(args.ema_decay) < 1.0:
        raise ValueError("ema_decay must be in (0,1)")
    if (
        args.allow_resume_max_steps_extension
        or args.allow_resume_topology_change
        or args.allow_resume_data_path_relocation
    ) and not args.resume:
        raise ValueError("resume contract override flags require --resume")


def architecture_checkpoint_format(args: argparse.Namespace) -> str:
    return (
        NATIVE_SLAT_GENRECON_V2_VERSION
        if args.architecture == "v2"
        else NATIVE_SLAT_GENRECON_VERSION
    )


def checkpoint_args(args: argparse.Namespace) -> dict[str, Any]:
    payload = dict(vars(args))
    # A stage boundary controls only how long this process invocation runs.  It
    # must not change the checkpoint/training identity: a step-400 pause must
    # resume with the original max_steps=2000 schedule, optimizer and RNG.
    payload.pop("run_until_step", None)
    payload.pop("allow_resume_max_steps_extension", None)
    payload.pop("allow_resume_topology_change", None)
    payload.pop("allow_resume_data_path_relocation", None)
    if args.architecture == "v2":
        # These fields do not exist in the released v2 architecture and must
        # not be serialized into a checkpoint that the strict v2 validator
        # would otherwise correctly reject.
        payload.pop("view_fusion_hidden_dim", None)
        payload.pop("geometry_logit_scale_init", None)
    return payload


def _nested_identity_value(identity: dict[str, Any], field: tuple[str, ...]) -> Any:
    value: Any = identity
    for key in field:
        if not isinstance(value, dict) or key not in value:
            raise RuntimeError(
                "resume data identity is missing approved path field: "
                + ".".join(field)
            )
        value = value[key]
    return value


def _replace_nested_identity_value(
    identity: dict[str, Any], field: tuple[str, ...], value: Any
) -> None:
    parent: Any = identity
    for key in field[:-1]:
        parent = parent[key]
    parent[field[-1]] = value


def validate_native_ss_deployment(
    frozen_deployment: Any,
    runtime_deployment: Any,
    *,
    allow_path_relocation: bool,
) -> dict[str, Any]:
    """Validate the full cache/runtime Native-SS deployment without weakening it."""

    error_message = "SLAT cache and training bind different Native SS deployments"
    if frozen_deployment == runtime_deployment:
        return {
            "path_relocated": False,
            "relocations": {},
            "all_non_path_fields_exact": True,
        }
    if not allow_path_relocation:
        raise RuntimeError(error_message)
    if not isinstance(frozen_deployment, dict) or not isinstance(
        runtime_deployment, dict
    ):
        raise RuntimeError(error_message)

    frozen_normalized = copy.deepcopy(frozen_deployment)
    runtime_normalized = copy.deepcopy(runtime_deployment)
    relocations: dict[str, dict[str, str]] = {}
    for field in APPROVED_NATIVE_SS_DEPLOYMENT_PATH_FIELDS:
        if field not in frozen_deployment or field not in runtime_deployment:
            raise RuntimeError(error_message)
        frozen_value = frozen_deployment[field]
        runtime_value = runtime_deployment[field]
        if not isinstance(frozen_value, str) or not isinstance(runtime_value, str):
            raise RuntimeError(error_message)
        try:
            frozen_resolved = Path(frozen_value).expanduser().resolve(strict=True)
            runtime_resolved = Path(runtime_value).expanduser().resolve(strict=True)
        except (FileNotFoundError, OSError, RuntimeError) as error:
            raise RuntimeError(
                "Native SS deployment path relocation requires existing resolvable "
                f"files: {field}"
            ) from error
        if frozen_resolved != runtime_resolved:
            raise RuntimeError(
                "Native SS deployment path relocation points to different files: "
                f"{field}"
            )
        frozen_normalized[field] = APPROVED_RELOCATED_PATH
        runtime_normalized[field] = APPROVED_RELOCATED_PATH
        if frozen_value != runtime_value:
            relocations[field] = {
                "frozen": frozen_value,
                "runtime": runtime_value,
                "resolved": str(runtime_resolved),
            }

    if frozen_normalized != runtime_normalized:
        raise RuntimeError(
            "SLAT cache and training bind different Native SS deployments outside "
            "approved path relocation"
        )
    return {
        "path_relocated": bool(relocations),
        "relocations": relocations,
        "all_non_path_fields_exact": True,
    }


def validate_resume_data_identity(
    saved_identity: Any,
    current_identity: Any,
    *,
    allow_path_relocation: bool,
) -> dict[str, Any]:
    """Allow only audited path relocation while preserving all content identity."""

    if saved_identity == current_identity:
        return {
            "applied": False,
            "approved_fields": [],
            "all_non_path_fields_exact": True,
        }
    if not allow_path_relocation:
        raise RuntimeError("resume data identity differs")
    if not isinstance(saved_identity, dict) or not isinstance(current_identity, dict):
        raise RuntimeError(
            "resume data identity differs outside approved path relocation"
        )

    saved_normalized = copy.deepcopy(saved_identity)
    current_normalized = copy.deepcopy(current_identity)
    relocated: list[dict[str, str]] = []
    for field in APPROVED_RESUME_DATA_PATH_FIELDS:
        saved_value = _nested_identity_value(saved_identity, field)
        current_value = _nested_identity_value(current_identity, field)
        if not isinstance(saved_value, str) or not isinstance(current_value, str):
            raise RuntimeError(
                "resume data identity differs outside approved path relocation"
            )
        try:
            saved_resolved = Path(saved_value).expanduser().resolve(strict=True)
            current_resolved = Path(current_value).expanduser().resolve(strict=True)
        except (FileNotFoundError, OSError, RuntimeError) as error:
            raise RuntimeError(
                "resume data path relocation requires existing resolvable files: "
                + ".".join(field)
            ) from error
        if saved_resolved != current_resolved:
            raise RuntimeError(
                "resume data path relocation points to different files: "
                + ".".join(field)
            )
        _replace_nested_identity_value(
            saved_normalized, field, APPROVED_RELOCATED_PATH
        )
        _replace_nested_identity_value(
            current_normalized, field, APPROVED_RELOCATED_PATH
        )
        if saved_value != current_value:
            relocated.append(
                {
                    "field": ".".join(field),
                    "saved": saved_value,
                    "current": current_value,
                    "resolved": str(current_resolved),
                }
            )

    if saved_normalized != current_normalized:
        raise RuntimeError(
            "resume data identity differs outside approved path relocation"
        )
    return {
        "applied": bool(relocated),
        "approved_fields": relocated,
        "all_non_path_fields_exact": True,
    }


def validate_resume_training_contract(
    checkpoint: dict[str, Any],
    args: argparse.Namespace,
    *,
    world_size: int,
) -> dict[str, Any]:
    """Validate and describe the two explicitly supported resume transitions."""

    saved = checkpoint["args"]
    saved_max_steps = int(saved["max_steps"])
    current_max_steps = int(args.max_steps)
    checkpoint_step = int(checkpoint["step"])
    max_steps_extended = current_max_steps != saved_max_steps
    if max_steps_extended:
        if not bool(args.allow_resume_max_steps_extension):
            raise ValueError(
                "resume max_steps mismatch requires "
                "--allow_resume_max_steps_extension"
            )
        if current_max_steps <= saved_max_steps:
            raise ValueError(
                "resume max_steps override must be a strict extension: "
                f"saved={saved_max_steps} current={current_max_steps}"
            )
        if checkpoint_step >= current_max_steps:
            raise ValueError(
                "extended max_steps must exceed checkpoint step: "
                f"checkpoint={checkpoint_step} current={current_max_steps}"
            )

    saved_grad_accum = int(saved["grad_accum"])
    current_grad_accum = int(args.grad_accum)
    saved_optimization = dict(
        checkpoint.get("model_summary", {}).get("optimization", {})
    )
    saved_world_size = int(saved_optimization.get("world_size", world_size))
    saved_global_batch = int(
        saved_optimization.get(
            "global_effective_batch", saved_world_size * saved_grad_accum
        )
    )
    current_global_batch = int(world_size) * current_grad_accum
    topology_changed = (
        saved_world_size != int(world_size)
        or saved_grad_accum != current_grad_accum
    )
    if topology_changed:
        if not bool(args.allow_resume_topology_change):
            raise ValueError(
                "resume world_size/grad_accum mismatch requires "
                "--allow_resume_topology_change"
            )
        if saved_global_batch != current_global_batch:
            raise ValueError(
                "resume topology change must preserve global effective batch: "
                f"saved={saved_global_batch} current={current_global_batch}"
            )

    return {
        "checkpoint_step": checkpoint_step,
        "max_steps_extended": bool(max_steps_extended),
        "saved_max_steps": saved_max_steps,
        "current_max_steps": current_max_steps,
        "topology_changed": bool(topology_changed),
        "saved_world_size": saved_world_size,
        "current_world_size": int(world_size),
        "saved_grad_accum": saved_grad_accum,
        "current_grad_accum": current_grad_accum,
        "saved_global_effective_batch": saved_global_batch,
        "current_global_effective_batch": current_global_batch,
        "optimizer_inherited": True,
        "ema_inherited": True,
        "per_rank_rng_stream_exact": not topology_changed,
    }


def resolved_run_until_step(args: argparse.Namespace) -> int:
    value = int(args.run_until_step)
    return int(args.max_steps) if value == 0 else value


def upstream_binding(value: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "report",
        "report_sha256",
        "checkpoint",
        "checkpoint_sha256",
        "checkpoint_step",
        "weights",
        "cfg_strength",
        "steps",
        "cfg_interval",
        "guidance_rescale",
        "rescale_t",
    )
    return {key: value[key] for key in keys}


def validate_decoder_audit(
    path: str | Path,
    *,
    cache_config: dict[str, Any],
    pretrained: str,
) -> dict[str, Any]:
    report_path = Path(path).resolve()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("format") != "pose_point_depth_mv.direct_slat_target_decoder_audit.v1":
        raise RuntimeError("unsupported target-decoder audit format")
    if report.get("passed") is not True:
        raise RuntimeError("target-decoder audit did not pass")
    if str(report.get("pretrained")) != str(pretrained):
        raise RuntimeError("target-decoder audit pretrained binding differs")
    if int(report.get("summary", {}).get("object_count", 0)) < 32:
        raise RuntimeError("target-decoder audit must cover at least 32 objects")
    # The decoder audit may intentionally be frozen on the object-disjoint val
    # cache.  Its Native-SS deployment and true-target source must match train,
    # while val is allowed to materialize additional noise seeds.
    audit_support = report.get("support_generator", {}).get("fields")
    train_support = support_generator_identity(cache_config).get("fields")
    if not isinstance(audit_support, dict) or not isinstance(train_support, dict):
        raise RuntimeError("target-decoder audit/cache support identity is malformed")
    audit_seeds = {int(value) for value in audit_support.get("ss_seeds", [])}
    train_seeds = {int(value) for value in train_support.get("ss_seeds", [])}
    audit_invariant = dict(audit_support)
    train_invariant = dict(train_support)
    audit_invariant.pop("ss_seeds", None)
    train_invariant.pop("ss_seeds", None)
    if audit_invariant != train_invariant or not train_seeds.issubset(audit_seeds):
        raise RuntimeError(
            "target-decoder audit/cache Native-SS or target protocol differs"
        )
    return {
        "path": str(report_path),
        "sha256": sha256_file(report_path),
        "format": report["format"],
        "summary": report["summary"],
        "thresholds": report["thresholds"],
    }


def sample_t(args: argparse.Namespace, device: torch.device) -> float:
    if str(getattr(args, "t_schedule", "logit_normal")) == "uniform":
        return float(torch.rand((), device=device).item())
    value = torch.sigmoid(
        torch.randn((), device=device) * float(args.t_logit_std)
        + float(args.t_logit_mean)
    )
    return float(value.item())


def sample_view_indices(
    sample: dict[str, Any], args: argparse.Namespace, device: torch.device
) -> torch.Tensor:
    count_all = int(sample["lifting_sample"]["visual_patch_features"].shape[0])
    upper = min(count_all, int(args.max_condition_views))
    lower = min(upper, int(args.min_condition_views))
    count = int(torch.randint(lower, upper + 1, (), device=device).item())
    return torch.randperm(count_all, device=device)[:count]


def select_context_views(values: Any, indices: torch.Tensor) -> Any:
    if not isinstance(values, list):
        return values
    selected = [values[int(index)] for index in indices.detach().cpu().tolist()]
    if not selected:
        raise ValueError("Native-SLAT condition selected no context views")
    return selected


def select_stock_context_views(values: Any, mode: str) -> Any:
    """Restrict only the non-spatial Stock cross-attention context.

    The separately projected posed-DINO path continues to receive all selected
    views through ``view_indices``.  Keeping this operation free of randomness
    preserves the matched seed/data order between the all-view and first-view
    training arms.
    """

    mode = str(mode)
    if mode not in STOCK_CONTEXT_VIEW_MODES:
        raise ValueError(f"unsupported Stock context view mode={mode!r}")
    if mode == "all":
        return values
    if not isinstance(values, list) or not values:
        raise ValueError("first-view Stock context requires a non-empty per-view list")
    return [values[0]]


def checkpoint_stock_context_views(checkpoint: dict[str, Any]) -> str:
    """Resolve the context policy, treating pre-experiment checkpoints as all-view."""

    saved = checkpoint.get("args", {})
    if not isinstance(saved, dict):
        raise ValueError("SLat checkpoint arguments are malformed")
    mode = str(saved.get("stock_context_views", "all"))
    if mode not in STOCK_CONTEXT_VIEW_MODES:
        raise ValueError(f"SLat checkpoint Stock context mode is invalid: {mode!r}")
    return mode


class NativeSLatTrainingForward(nn.Module):
    def __init__(self, model: NativeSLatGenreconFlow) -> None:
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
        if sample is None:
            anchor = prediction.feats.new_zeros((), dtype=torch.float32)
            for parameter in self.model.aggregator.parameters():
                anchor = anchor + parameter.reshape(-1)[0].float() * 0.0
            prediction = prediction.replace(
                prediction.feats + anchor.to(dtype=prediction.feats.dtype)
            )
        return prediction, stats


def distributed_true(value: bool, *, device: torch.device, world_size: int) -> bool:
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


def resolve_warmup_steps(args: argparse.Namespace) -> int:
    result = (
        int(args.warmup_steps)
        if int(args.warmup_steps) >= 0
        else int(math.ceil(float(args.warmup_ratio) * int(args.max_steps)))
    )
    if result > int(args.max_steps):
        raise ValueError("warmup_steps exceeds max_steps")
    return result


def warmup_factor(update_step: int, warmup_steps: int) -> float:
    return 1.0 if int(warmup_steps) <= 0 else min(1.0, update_step / warmup_steps)


def ema_ramp_decay(target_decay: float, update_step: int) -> float:
    return min(float(target_decay), float(1 + update_step) / float(10 + update_step))


def trainable_named_parameters(model: nn.Module) -> dict[str, nn.Parameter]:
    return {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def initialize_ema_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().float().clone()
        for name, parameter in trainable_named_parameters(model).items()
    }


@torch.no_grad()
def update_ema_state(
    ema_state: dict[str, torch.Tensor], model: nn.Module, *, decay: float
) -> None:
    named = trainable_named_parameters(model)
    if set(ema_state) != set(named):
        raise ValueError("EMA/trainable names differ")
    for name, parameter in named.items():
        ema_state[name].mul_(decay).add_(
            parameter.detach().float(), alpha=1.0 - decay
        )


def load_ema_state(
    model: nn.Module, state: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    named = trainable_named_parameters(model)
    if set(state) != set(named):
        raise ValueError("checkpoint EMA/trainable names differ")
    return {
        name: value.to(device=named[name].device, dtype=torch.float32).clone()
        for name, value in state.items()
    }


def ema_state_finite(state: dict[str, torch.Tensor]) -> bool:
    return bool(state) and all(
        bool(torch.isfinite(value).all().item()) for value in state.values()
    )


def gradient_norms(model: nn.Module) -> dict[str, float]:
    values = {"lora": 0.0, "condition": 0.0, "view_fusion": 0.0}
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            key = (
                "lora"
                if "lora_" in name
                else "view_fusion"
                if name.startswith("view_fusion.")
                else "condition"
            )
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
        raise RuntimeError(f"refusing to save non-finite Native-SLAT state: {path}")
    if not finite_tree(scaler.state_dict()) or not ema_state_finite(ema_state):
        raise RuntimeError(f"refusing to save non-finite scaler/EMA: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    state_builder = (
        v2_trainable_state_dict
        if args.architecture == "v2"
        else trainable_state_dict
    )
    payload = {
        "format": architecture_checkpoint_format(args),
        "step": int(step),
        "micro_step": int(micro_step),
        "model_trainable_state": state_builder(model),
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
        "args": checkpoint_args(args),
        "model_summary": model_summary,
        "data_identity": data_identity,
        "history": history,
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state(),
        },
    }
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


@torch.no_grad()
def initial_stock_audit(
    model: nn.Module,
    sampler: Any,
    sample: dict[str, Any],
    *,
    architecture: str,
    stock_context_views: str,
    mean: torch.Tensor,
    std: torch.Tensor,
    device: torch.device,
) -> dict[str, Any]:
    target = normalized_target(sample, mean=mean, std=std, device=device)
    noise = sp.SparseTensor(
        feats=torch.zeros_like(target.feats), coords=target.coords
    )
    x_t, _ = sampler._get_model_gt(target, 0.5, noise)
    t = torch.full((1,), 500.0, device=device)
    positive = select_stock_context_views(
        to_device_tree(sample["condition"]["cond"], device), stock_context_views
    )
    negative = select_stock_context_views(
        to_device_tree(sample["condition"]["neg_cond"], device), stock_context_views
    )
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
    report = {
        "conditional_max_abs": float(
            (full_positive.feats.float() - stock_positive.feats.float()).abs().amax().item()
        ),
        "unconditional_max_abs": float(
            (full_negative.feats.float() - stock_negative.feats.float()).abs().amax().item()
        ),
        "conditional_blocks": len(model.block_condition.projections),
        "supported_fraction": float(positive_stats["supported_fraction"].item()),
        "unconditional_condition_present": float(
            negative_stats["condition_present"].item()
        ),
        "stock_context_views": str(stock_context_views),
    }
    if architecture == "v3":
        report.update(
            {
                "view_fusion_blocks": model.view_fusion.block_count,
                "view_fusion_gate_max_abs": float(
                    model.view_fusion.transition_gate_raw.detach()
                    .abs()
                    .amax()
                    .item()
                ),
            }
        )
    report["passed"] = bool(
        report["conditional_max_abs"] == 0.0
        and report["unconditional_max_abs"] == 0.0
        and report["conditional_blocks"] == len(model.flow_core.blocks)
        and report["supported_fraction"] > 0.0
        and report["unconditional_condition_present"] == 0.0
        and (
            architecture == "v2"
            or (
                report["view_fusion_blocks"] == len(model.flow_core.blocks)
                and report["view_fusion_gate_max_abs"] == 0.0
            )
        )
    )
    if not report["passed"]:
        raise RuntimeError(f"initial Native-SLAT Stock audit failed: {report}")
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
    if dataset.config.get("condition_arch") != "native_ss_genrecon_v2":
        raise RuntimeError(
            "Native-SLAT v2/v3 requires the native_ss_genrecon_v2 cache contract"
        )
    ss_payload, ss_binding_all = load_ss_evidence(args.native_ss_report)
    del ss_payload
    validate_native_ss_deployment(
        dataset.config.get("native_ss_deployment"),
        ss_binding_all,
        allow_path_relocation=bool(
            args.resume and args.allow_resume_data_path_relocation
        ),
    )
    ss_binding = upstream_binding(ss_binding_all)
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
    if args.architecture == "v2":
        sampler, model, _, model_summary, sampler_defaults, normalization = (
            build_native_slat_genrecon_v2_components(
                pretrained=args.pretrained,
                stock_slat_freeze=stock_freeze,
                upstream_native_ss=ss_binding,
                lora_rank=int(args.lora_rank),
                lora_alpha=int(args.lora_alpha),
                condition_channels=int(args.condition_channels),
                gradient_checkpointing=bool(args.gradient_checkpointing),
                need_decoder=False,
                device=device,
            )
        )
    else:
        sampler, model, _, model_summary, sampler_defaults, normalization = (
            build_native_slat_genrecon_components(
                pretrained=args.pretrained,
                stock_slat_freeze=stock_freeze,
                upstream_native_ss=ss_binding,
                lora_rank=int(args.lora_rank),
                lora_alpha=int(args.lora_alpha),
                condition_channels=int(args.condition_channels),
                view_fusion_hidden_dim=int(args.view_fusion_hidden_dim),
                geometry_logit_scale_init=float(args.geometry_logit_scale_init),
                gradient_checkpointing=bool(args.gradient_checkpointing),
                need_decoder=False,
                device=device,
            )
        )
    runtime_normalization = {
        key: [float(value) for value in values]
        for key, values in normalization.items()
    }
    if canonical_json_sha256(runtime_normalization) != dataset.slat_normalization_hash:
        raise RuntimeError("runtime SLAT normalization differs from cache")
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
        },
        "p_uncond": float(args.p_uncond),
        "view_augmentation": {
            "random_subset": True,
            "min": int(args.min_condition_views),
            "max": int(args.max_condition_views),
        },
        "stock_cross_attention_context": {
            "views": str(args.stock_context_views),
            "first_definition": "first element of the randomly selected view subset",
            "posed_dino_3d_views": "all randomly selected views",
            "controlled_variable": "non-spatial Stock SLat cross-attention only",
        },
        "sampler_defaults": sampler_defaults,
        "optimization": {
            "warmup_steps": warmup_steps,
            "ema_target_decay": float(args.ema_decay),
            "world_size": world_size,
            "global_effective_batch": world_size * int(args.grad_accum),
        },
        "training_coordinate_policy": (
            "true target SLAT coordinates for supervised flow matching; deployment "
            "is always evaluated on frozen Native SS predicted coordinates"
        ),
    }
    initial_audit = initial_stock_audit(
        model,
        sampler,
        dataset[0],
        architecture=str(args.architecture),
        stock_context_views=str(args.stock_context_views),
        mean=mean,
        std=std,
        device=device,
    )
    training_forward = NativeSLatTrainingForward(model)
    wrapped: nn.Module = training_forward
    if world_size > 1:
        wrapped = DistributedDataParallel(
            training_forward,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            find_unused_parameters=False,
        )
    parameter_group_builder = (
        v2_optimizer_parameter_groups
        if args.architecture == "v2"
        else optimizer_parameter_groups
    )
    optimizer = torch.optim.AdamW(
        parameter_group_builder(
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
        checkpoint_upstream_native_ss = checkpoint.get("data_identity", {}).get(
            "native_ss"
        )
        if not isinstance(checkpoint_upstream_native_ss, dict):
            raise ValueError("resume checkpoint lacks Native SS data identity")
        if args.architecture == "v2":
            validate_native_slat_genrecon_v2_checkpoint(
                checkpoint,
                pretrained=args.pretrained,
                stock_slat_freeze=stock_freeze,
                upstream_native_ss=checkpoint_upstream_native_ss,
            )
        else:
            validate_native_slat_genrecon_checkpoint(
                checkpoint,
                pretrained=args.pretrained,
                stock_slat_freeze=stock_freeze,
                upstream_native_ss=checkpoint_upstream_native_ss,
            )
        data_identity_transition = validate_resume_data_identity(
            checkpoint.get("data_identity"),
            data_identity,
            allow_path_relocation=bool(args.allow_resume_data_path_relocation),
        )
        resume_transition = validate_resume_training_contract(
            checkpoint,
            args,
            world_size=world_size,
        )
        resume_transition["data_path_relocation"] = data_identity_transition
        saved = checkpoint["args"]
        fields = [
            "lora_rank",
            "lora_alpha",
            "condition_channels",
            "new_lr",
            "lora_lr",
            "new_weight_decay",
            "p_uncond",
            "t_logit_mean",
            "t_logit_std",
            "t_schedule",
            "min_condition_views",
            "max_condition_views",
            "stock_context_views",
            "grad_accum",
            "max_steps",
            "warmup_steps",
            "warmup_ratio",
            "ema_decay",
        ]
        if args.architecture == "v3":
            fields.extend(("view_fusion_hidden_dim", "geometry_logit_scale_init"))
        def saved_argument(key: str):
            if key == "stock_context_views":
                return saved.get(key, "all")
            if key == "t_schedule":
                # Checkpoints created before this explicit control used the
                # logit-normal implementation unconditionally.
                return saved.get(key, "logit_normal")
            return saved.get(key)

        permitted_mismatch = set()
        if resume_transition["max_steps_extended"]:
            permitted_mismatch.add("max_steps")
        if resume_transition["topology_changed"]:
            permitted_mismatch.add("grad_accum")
        mismatch = {
            key: (saved_argument(key), getattr(args, key))
            for key in fields
            if key not in permitted_mismatch
            and saved_argument(key) != getattr(args, key)
        }
        if mismatch:
            raise ValueError(f"resume argument mismatch={mismatch}")
        state_loader = (
            load_v2_trainable_state_dict
            if args.architecture == "v2"
            else load_trainable_state_dict
        )
        state_loader(model, checkpoint["model_trainable_state"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint.get("scaler", {}))
        ema_state = load_ema_state(model, checkpoint["ema_trainable_state"])
        ema_last_decay = float(checkpoint.get("ema", {}).get("last_decay", 0.0))
        step = int(checkpoint["step"])
        micro_step = (
            step * int(args.grad_accum)
            if resume_transition["topology_changed"]
            else int(checkpoint.get("micro_step", step * int(args.grad_accum)))
        )
        history = list(checkpoint.get("history", []))
        rng = checkpoint.get("rng", {})
        if rng and not resume_transition["topology_changed"]:
            random.setstate(rng["python"])
            np.random.set_state(rng["numpy"])
            torch.set_rng_state(rng["torch"])
            torch.cuda.set_rng_state(rng["cuda"])
        elif resume_transition["topology_changed"]:
            boundary_seed = int(args.seed) + rank * 100003 + step * 1000003
            random.seed(boundary_seed)
            np.random.seed(boundary_seed % (2**32 - 1))
            torch.manual_seed(boundary_seed)
            torch.cuda.manual_seed_all(boundary_seed)
        previous_transitions = list(
            checkpoint.get("model_summary", {}).get("resume_transitions", [])
        )
        model_summary["resume_transitions"] = [
            *previous_transitions,
            resume_transition,
        ]
    elif args.init_checkpoint:
        source_path = Path(args.init_checkpoint).expanduser().resolve()
        checkpoint = torch.load(source_path, map_location="cpu")
        if args.architecture == "v2":
            source_upstream = dict(
                checkpoint.get("model_summary", {}).get("upstream_native_ss", {})
            )
            validate_native_slat_genrecon_v2_checkpoint(
                checkpoint,
                pretrained=args.pretrained,
                stock_slat_freeze=stock_freeze,
                upstream_native_ss=source_upstream,
            )
            state_loader = load_v2_trainable_state_dict
        else:
            source_upstream = dict(
                checkpoint.get("model_summary", {}).get("upstream_native_ss", {})
            )
            validate_native_slat_genrecon_checkpoint(
                checkpoint,
                pretrained=args.pretrained,
                stock_slat_freeze=stock_freeze,
                upstream_native_ss=source_upstream,
            )
            state_loader = load_trainable_state_dict
        state_key = (
            "ema_trainable_state"
            if str(args.init_weights) == "ema"
            else "model_trainable_state"
        )
        state_loader(model, checkpoint[state_key])
        ema_state = initialize_ema_state(model)
        model_summary["initialization"] = {
            "mode": "new_data_identity",
            "checkpoint": str(source_path),
            "checkpoint_sha256": sha256_file(source_path),
            "checkpoint_step": int(checkpoint["step"]),
            "weights": str(args.init_weights),
            "source_upstream_native_ss": source_upstream,
            "destination_upstream_native_ss": ss_binding,
            "optimizer_inherited": False,
            "ema_reinitialized_from_selected_weights": True,
        }

    trainable = [value for value in model.parameters() if value.requires_grad]
    optimizer.zero_grad(set_to_none=True)
    model.train()
    epoch = 0
    started = time.time()
    accumulator: list[dict[str, float | bool]] = []
    run_until_step = resolved_run_until_step(args)
    if step > run_until_step:
        raise ValueError(
            f"resume checkpoint step={step} exceeds run_until_step={run_until_step}"
        )
    while step < run_until_step:
        object_sampler.set_epoch(epoch)
        for sample in loader:
            if step >= run_until_step:
                break
            target = normalized_target(sample, mean=mean, std=std, device=device)
            noise = sp.SparseTensor(
                feats=torch.randn_like(target.feats), coords=target.coords
            )
            t_value = sample_t(args, device)
            x_t, gt_velocity = sampler._get_model_gt(target, t_value, noise)
            t = torch.full((1,), 1000.0 * t_value, device=device)
            view_indices = sample_view_indices(sample, args, device)
            positive_all = to_device_tree(sample["condition"]["cond"], device)
            negative_all = to_device_tree(sample["condition"]["neg_cond"], device)
            positive = select_context_views(positive_all, view_indices)
            negative = select_context_views(negative_all, view_indices)
            positive_stock = select_stock_context_views(
                positive, str(args.stock_context_views)
            )
            negative_stock = select_stock_context_views(
                negative, str(args.stock_context_views)
            )
            unconditional = random.random() < float(args.p_uncond)
            condition = negative_stock if unconditional else positive_stock
            condition_sample = None if unconditional else sample["lifting_sample"]
            with torch.no_grad():
                stock = model.stock_prediction(x_t, t, condition)
                stock_loss = F.mse_loss(stock.feats.float(), gt_velocity.feats.float())
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
                    loss = flow_loss / float(args.grad_accum)
                scaler.scale(loss).backward()
            micro_step += 1
            accumulator.append(
                {
                    "flow_loss": float(flow_loss.detach().item()),
                    "stock_loss": float(stock_loss.detach().item()),
                    "gain": float((stock_loss - flow_loss.detach()).item()),
                    "t": t_value,
                    "unconditional": unconditional,
                    "views": float(len(view_indices)),
                    "flow_delta_rms": float(stats["flow_delta_rms"].detach().item()),
                    "supported_fraction": float(
                        stats["supported_fraction"].detach().item()
                    ),
                    "fusion_gate_mean": float(
                        stats.get(
                            "fusion_gate_mean",
                            stats["supported_fraction"].new_zeros(()),
                        )
                        .detach()
                        .item()
                    ),
                    "effective_view_weight_deviation": float(
                        stats.get(
                            "effective_view_weight_deviation",
                            stats["supported_fraction"].new_zeros(()),
                        )
                        .detach()
                        .item()
                    ),
                    "target_view_weight_entropy": float(
                        stats.get(
                            "target_view_weight_entropy",
                            stats["supported_fraction"].new_zeros(()),
                        )
                        .detach()
                        .item()
                    ),
                    "active_points": float(target.feats.shape[0]),
                }
            )
            if not sync_step:
                continue
            scaler.unscale_(optimizer)
            if not distributed_true(
                gradients_finite(trainable), device=device, world_size=world_size
            ):
                raise RuntimeError("Native-SLAT gradients became non-finite")
            grad_norm = float(
                torch.nn.utils.clip_grad_norm_(trainable, float(args.grad_clip)).item()
            )
            group_gradients = gradient_norms(model)
            next_step = step + 1
            lr_scale = warmup_factor(next_step, warmup_steps)
            for group in optimizer.param_groups:
                group["lr"] = float(group["base_lr"]) * lr_scale
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            finite = parameters_finite(trainable) and optimizer_state_finite(optimizer)
            if not distributed_true(finite, device=device, world_size=world_size):
                raise RuntimeError("Native-SLAT optimizer produced non-finite state")
            ema_last_decay = ema_ramp_decay(float(args.ema_decay), next_step)
            update_ema_state(ema_state, model, decay=ema_last_decay)
            step = next_step
            metrics = distributed_means(
                [
                    float(np.mean([float(row["flow_loss"]) for row in accumulator])),
                    float(np.mean([float(row["stock_loss"]) for row in accumulator])),
                    float(np.mean([float(row["gain"]) for row in accumulator])),
                    float(np.mean([float(row["t"]) for row in accumulator])),
                    float(np.mean([float(bool(row["unconditional"])) for row in accumulator])),
                    float(np.mean([float(row["views"]) for row in accumulator])),
                    float(np.mean([float(row["flow_delta_rms"]) for row in accumulator])),
                    float(np.mean([float(row["supported_fraction"]) for row in accumulator])),
                    float(np.mean([float(row["active_points"]) for row in accumulator])),
                    float(np.mean([float(row["fusion_gate_mean"]) for row in accumulator])),
                    float(
                        np.mean(
                            [
                                float(row["effective_view_weight_deviation"])
                                for row in accumulator
                            ]
                        )
                    ),
                    float(
                        np.mean(
                            [float(row["target_view_weight_entropy"]) for row in accumulator]
                        )
                    ),
                ],
                device=device,
                world_size=world_size,
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
                "supported_fraction": metrics[7],
                "active_point_count_mean": metrics[8],
                "fusion_gate_mean": metrics[9],
                "effective_view_weight_deviation": metrics[10],
                "target_view_weight_entropy": metrics[11],
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
                print(f"[native_slat_genrecon] {json.dumps(row)}", flush=True)
            if rank == 0 and (
                step % int(args.save_every) == 0 or step == run_until_step
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

    stage_passed = distributed_true(
        bool(
            step == run_until_step
            and parameters_finite(trainable)
            and optimizer_state_finite(optimizer)
            and ema_state_finite(ema_state)
        ),
        device=device,
        world_size=world_size,
    )
    completed = step == int(args.max_steps)
    report = {
        "format": architecture_checkpoint_format(args),
        "completed": completed,
        "stage_complete": stage_passed,
        "passed": bool(stage_passed and completed),
        "step": step,
        "run_until_step": run_until_step,
        "max_steps": int(args.max_steps),
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
        "explicitly_absent": [
            "Direct-SLAT residual bound",
            "wrong-support ranking",
            "teacher-forced rollout auxiliary",
            "trainable Mesh decoder",
            "condition scale",
        ],
    }
    if rank == 0:
        report_path = (
            output_dir / "report.json"
            if completed
            else output_dir / f"stage_report_step_{step:06d}.json"
        )
        report_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "completed": completed,
                    "stage_complete": stage_passed,
                    "step": step,
                    "max_steps": int(args.max_steps),
                    "report": str(report_path),
                    "output": str(output_dir),
                }
            ),
            flush=True,
        )
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()
    raise SystemExit(0 if stage_passed else 2)


if __name__ == "__main__":
    main()
