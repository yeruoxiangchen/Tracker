#!/usr/bin/env python3
"""Stock-preserving SS-supported SLAT Flow components.

The native SLAT image condition remains untouched.  A frozen, corrected SS
rollout supplies spatial evidence that is gathered after the native SLAT stem
has reduced 64^3 sparse coordinates to 32^3.  The gathered evidence is mapped
to the SLAT model width by a zero-initialised adapter and injected immediately
before transformer block 0.  Only that adapter and LoRA weights on the native
SLAT Flow are trainable.
"""

from __future__ import annotations

from collections import Counter
from contextlib import nullcontext
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F
from torch import nn

from reconvggt_ar_adapter_a.pointpose_ss_condition import lora_disabled


DIRECT_SLAT_FLOW_VERSION = "pose_point_depth_mv.direct_slat_flow.v2"
DIRECT_SLAT_CACHE_VERSION = "pose_point_depth_mv.direct_slat_cache.v1"
SLAT_SUPPORT_MAPPING_VERSION = "slat32_to_ss16_occ64_xmajor.v1"
SLAT_DELTA_POLICY_VERSION = "stock_anchored_per_sparse_batch_rms.v1"
SLAT_GUIDED_DELTA_POLICY_LEGACY = "positive_branch_v1"
SLAT_GUIDED_DELTA_POLICY_V2 = "post_cfg_v2"
SLAT_GUIDED_DELTA_POLICIES = (
    SLAT_GUIDED_DELTA_POLICY_LEGACY,
    SLAT_GUIDED_DELTA_POLICY_V2,
)
SLAT_DELTA_BOUND_HARD = "hard_clip_v1"
SLAT_DELTA_BOUND_SMOOTH = "smooth_rms_v2"
SLAT_DELTA_BOUND_MODES = (
    SLAT_DELTA_BOUND_HARD,
    SLAT_DELTA_BOUND_SMOOTH,
)
SLAT_SUPPORT_INTERVAL_ALL = "all_steps_v1"
SLAT_SUPPORT_INTERVAL_CFG_ACTIVE = "cfg_active_only_v1"
SLAT_SUPPORT_INTERVAL_POLICIES = (
    SLAT_SUPPORT_INTERVAL_ALL,
    SLAT_SUPPORT_INTERVAL_CFG_ACTIVE,
)
SLAT_ROLLOUT_COMPONENT_FULL = "full"
SLAT_ROLLOUT_COMPONENT_LORA_ONLY = "lora_only"
SLAT_ROLLOUT_COMPONENT_ADAPTER_ONLY = "adapter_only"
SLAT_ROLLOUT_COMPONENTS = (
    SLAT_ROLLOUT_COMPONENT_FULL,
    SLAT_ROLLOUT_COMPONENT_LORA_ONLY,
    SLAT_ROLLOUT_COMPONENT_ADAPTER_ONLY,
)
SLAT_RESIDUAL_COMBINATION_JOINT = "joint_total_v1"
SLAT_RESIDUAL_COMBINATION_BRANCH_BUDGET = "lora_support_budget_v2"
SLAT_RESIDUAL_COMBINATION_POLICIES = (
    SLAT_RESIDUAL_COMBINATION_JOINT,
    SLAT_RESIDUAL_COMBINATION_BRANCH_BUDGET,
)
DIRECT_SLAT_TRAINING_SEMANTICS_V2 = "bounded_mechanism_v2"
DIRECT_SLAT_TRAINING_SEMANTICS_V3 = "rollout_aligned_v3"
DIRECT_SLAT_TRAINING_SEMANTICS_V4 = "rollout_endpoint_v4"
DIRECT_SLAT_TRAINING_SEMANTICS_V5 = "branch_budget_rollout_v5"
SLAT_ROLLOUT_SUPERVISION_ALL_VISITED = "all_visited_v1"
SLAT_ROLLOUT_SUPERVISION_TERMINAL_CONSTANT_MEMORY = (
    "terminal_only_constant_memory_v1"
)
SLAT_ROLLOUT_SUPERVISION_POLICIES = (
    SLAT_ROLLOUT_SUPERVISION_ALL_VISITED,
    SLAT_ROLLOUT_SUPERVISION_TERMINAL_CONSTANT_MEMORY,
)
SUPPORT_RUNTIME_FIELDS = (
    "pretrained",
    "ss_flow_checkpoint_sha256",
    "expected_ss_step",
    "correspondence_checkpoint_sha256",
    "n3_report_sha256",
    "ss_seeds",
    "ss_steps",
    "cfg_strength",
    "guidance_rescale",
    "rescale_t",
    "physical_scale",
    "condition_replay_max_abs",
    "min_frame_iou",
    "amp_dtype",
    "mapping_version",
)


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def resolve_slat_guided_delta_policy(
    saved_args: dict[str, Any],
    override: str | None = None,
) -> str:
    """Resolve rollout policy while preserving legacy checkpoint behavior."""

    saved_policy = str(
        saved_args.get(
            "slat_guided_delta_policy",
            SLAT_GUIDED_DELTA_POLICY_LEGACY,
        )
    )
    if saved_policy not in SLAT_GUIDED_DELTA_POLICIES:
        raise ValueError(f"unsupported saved SLAT guided-delta policy={saved_policy!r}")
    resolved = saved_policy if override is None else str(override)
    if resolved not in SLAT_GUIDED_DELTA_POLICIES:
        raise ValueError(f"unsupported SLAT guided-delta policy={resolved!r}")
    if (
        saved_args.get("training_semantics")
        in {
            DIRECT_SLAT_TRAINING_SEMANTICS_V2,
            DIRECT_SLAT_TRAINING_SEMANTICS_V3,
            DIRECT_SLAT_TRAINING_SEMANTICS_V4,
            DIRECT_SLAT_TRAINING_SEMANTICS_V5,
        }
        and resolved != saved_policy
    ):
        raise ValueError(
            "versioned Direct-SLAT checkpoint rollout policy is immutable: "
            f"saved={saved_policy!r}, requested={resolved!r}"
        )
    return resolved


def resolve_slat_delta_bound_mode(
    saved_args: dict[str, Any],
    override: str | None = None,
) -> str:
    """Resolve the residual map without changing frozen v2/v3 semantics."""

    saved_mode = str(
        saved_args.get("slat_delta_bound_mode", SLAT_DELTA_BOUND_HARD)
    )
    if saved_mode not in SLAT_DELTA_BOUND_MODES:
        raise ValueError(f"unsupported saved SLAT delta bound mode={saved_mode!r}")
    resolved = saved_mode if override is None else str(override)
    if resolved not in SLAT_DELTA_BOUND_MODES:
        raise ValueError(f"unsupported SLAT delta bound mode={resolved!r}")
    if (
        saved_args.get("training_semantics")
        in {
            DIRECT_SLAT_TRAINING_SEMANTICS_V2,
            DIRECT_SLAT_TRAINING_SEMANTICS_V3,
            DIRECT_SLAT_TRAINING_SEMANTICS_V4,
            DIRECT_SLAT_TRAINING_SEMANTICS_V5,
        }
        and resolved != saved_mode
    ):
        raise ValueError(
            "versioned Direct-SLAT checkpoint bound mode is immutable: "
            f"saved={saved_mode!r}, requested={resolved!r}"
        )
    return resolved


def resolve_slat_support_interval_policy(
    saved_args: dict[str, Any],
    override: str | None = None,
) -> str:
    """Resolve whether Full support is active outside native CFG timesteps."""

    saved_policy = str(
        saved_args.get(
            "support_interval_policy",
            SLAT_SUPPORT_INTERVAL_ALL,
        )
    )
    if saved_policy not in SLAT_SUPPORT_INTERVAL_POLICIES:
        raise ValueError(
            f"unsupported saved SLAT support interval policy={saved_policy!r}"
        )
    resolved = saved_policy if override is None else str(override)
    if resolved not in SLAT_SUPPORT_INTERVAL_POLICIES:
        raise ValueError(
            f"unsupported SLAT support interval policy={resolved!r}"
        )
    if (
        saved_args.get("training_semantics")
        in {
            DIRECT_SLAT_TRAINING_SEMANTICS_V2,
            DIRECT_SLAT_TRAINING_SEMANTICS_V3,
            DIRECT_SLAT_TRAINING_SEMANTICS_V4,
            DIRECT_SLAT_TRAINING_SEMANTICS_V5,
        }
        and resolved != saved_policy
    ):
        raise ValueError(
            "versioned Direct-SLAT checkpoint support interval is immutable: "
            f"saved={saved_policy!r}, requested={resolved!r}"
        )
    return resolved


def resolve_slat_residual_combination_policy(
    saved_args: dict[str, Any],
    override: str | None = None,
) -> str:
    """Resolve joint versus separately budgeted LoRA/support residuals."""

    saved_policy = str(
        saved_args.get(
            "slat_residual_combination_policy",
            SLAT_RESIDUAL_COMBINATION_JOINT,
        )
    )
    if saved_policy not in SLAT_RESIDUAL_COMBINATION_POLICIES:
        raise ValueError(
            "unsupported saved SLAT residual combination policy="
            f"{saved_policy!r}"
        )
    resolved = saved_policy if override is None else str(override)
    if resolved not in SLAT_RESIDUAL_COMBINATION_POLICIES:
        raise ValueError(
            f"unsupported SLAT residual combination policy={resolved!r}"
        )
    if (
        saved_args.get("training_semantics")
        == DIRECT_SLAT_TRAINING_SEMANTICS_V5
        and resolved != saved_policy
    ):
        raise ValueError(
            "versioned Direct-SLAT V5 residual combination is immutable: "
            f"saved={saved_policy!r}, requested={resolved!r}"
        )
    return resolved


def support_generator_identity(config: dict[str, Any]) -> dict[str, Any]:
    """Return the frozen SS-support fields that must match across splits."""

    fields = (*SUPPORT_RUNTIME_FIELDS, "target_source")
    missing = [name for name in fields if name not in config]
    if missing:
        raise ValueError(f"support-generator config lacks fields={missing}")
    values = {name: config[name] for name in fields}
    return {"fields": values, "hash": canonical_json_sha256(values)}


def support_runtime_identity(config: dict[str, Any]) -> dict[str, Any]:
    """Return path-independent runtime fields shared by train and holdout caches.

    ``target_source`` identifies the object set used to build local GT SLAT
    targets.  A genuinely unseen holdout must have a different target-source
    run config, while the frozen SS generator and native SLAT runtime must
    remain identical.  Target-family compatibility is audited separately by
    the blind protocol freezer.
    """

    missing = [name for name in SUPPORT_RUNTIME_FIELDS if name not in config]
    if missing:
        raise ValueError(f"support-runtime config lacks fields={missing}")
    values = {name: config[name] for name in SUPPORT_RUNTIME_FIELDS}
    return {"fields": values, "hash": canonical_json_sha256(values)}


def legacy_support_runtime_identity(identity: dict[str, Any]) -> dict[str, Any]:
    """Project a v2 checkpoint's legacy support identity onto runtime fields."""

    fields = identity.get("fields")
    if not isinstance(fields, dict):
        raise ValueError("legacy support identity has no fields dictionary")
    return support_runtime_identity(fields)


def _require_shape(value: torch.Tensor, shape: tuple[int | None, ...], label: str) -> None:
    if value.ndim != len(shape):
        raise ValueError(f"{label} rank={value.ndim} != {len(shape)}")
    for axis, expected in enumerate(shape):
        if expected is not None and int(value.shape[axis]) != int(expected):
            raise ValueError(
                f"{label} shape={tuple(value.shape)} has axis {axis} != {expected}"
            )


def validate_sparse_target_alignment(
    coords: torch.Tensor,
    feats: torch.Tensor,
    *,
    resolution: int = 64,
    channels: int = 8,
    require_single_batch: bool = False,
) -> dict[str, int]:
    """Validate a SLAT target without sorting or silently deduplicating it."""

    _require_shape(coords, (None, 4), "target coords")
    _require_shape(feats, (int(coords.shape[0]), channels), "target feats")
    if coords.dtype not in (
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    ):
        raise TypeError(f"target coords must be integer, got {coords.dtype}")
    if not bool(torch.isfinite(feats.float()).all().item()):
        raise ValueError("target feats contain non-finite values")
    if int(coords.shape[0]) <= 0:
        raise ValueError("target support is empty")
    batch = coords[:, 0].long()
    xyz = coords[:, 1:].long()
    if int(batch.min().item()) < 0:
        raise ValueError("target coords contain a negative batch index")
    if require_single_batch and not bool(torch.all(batch == 0).item()):
        raise ValueError("single-sample target coords must use batch index 0")
    if int(xyz.min().item()) < 0 or int(xyz.max().item()) >= int(resolution):
        raise ValueError(
            f"target coords leave [0,{int(resolution) - 1}]"
        )
    keys = (
        batch * int(resolution) ** 3
        + xyz[:, 0] * int(resolution) ** 2
        + xyz[:, 1] * int(resolution)
        + xyz[:, 2]
    )
    if int(torch.unique(keys).numel()) != int(keys.numel()):
        raise ValueError("target coords contain duplicates")
    return {
        "point_count": int(coords.shape[0]),
        "batch_count": int(batch.max().item()) + 1,
        "resolution": int(resolution),
        "channels": int(channels),
    }


def gather_slat_support_evidence(
    coords32: torch.Tensor,
    corrected_ss: torch.Tensor,
    occupancy_logits64: torch.Tensor,
    physical_tokens16: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Gather corrected SS evidence for post-stem SLAT coordinates.

    Coordinate convention is always ``[batch, x, y, z]``.  The native SLAT
    stem maps 64^3 coordinates to 32^3.  Each 32-grid coordinate therefore
    owns a 2x2x2 block of 64-grid occupancy logits and maps by integer division
    by two to one 16-grid SS/physical token.  Physical tokens are flattened in
    x-major order: ``x * 16 * 16 + y * 16 + z``.
    """

    _require_shape(coords32, (None, 4), "SLAT stem coords")
    _require_shape(corrected_ss, (None, 8, 16, 16, 16), "corrected SS")
    if occupancy_logits64.ndim == 4:
        occupancy_logits64 = occupancy_logits64.unsqueeze(1)
    _require_shape(
        occupancy_logits64,
        (int(corrected_ss.shape[0]), 1, 64, 64, 64),
        "occupancy logits",
    )
    _require_shape(
        physical_tokens16,
        (int(corrected_ss.shape[0]), 16**3, None),
        "physical tokens",
    )
    if coords32.dtype not in (
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    ):
        raise TypeError(f"SLAT stem coords must be integer, got {coords32.dtype}")
    if int(coords32.shape[0]) <= 0:
        raise ValueError("SLAT stem support is empty")
    batch = coords32[:, 0].long()
    xyz32 = coords32[:, 1:].long()
    batch_count = int(corrected_ss.shape[0])
    if int(batch.min().item()) < 0 or int(batch.max().item()) >= batch_count:
        raise ValueError("SLAT stem coords contain an invalid batch index")
    if int(xyz32.min().item()) < 0 or int(xyz32.max().item()) >= 32:
        raise ValueError("SLAT stem coords leave [0,31]")
    if not all(
        bool(torch.isfinite(value.float()).all().item())
        for value in (corrected_ss, occupancy_logits64, physical_tokens16)
    ):
        raise ValueError("support evidence contains non-finite values")

    xyz16 = torch.div(xyz32, 2, rounding_mode="floor")
    x16, y16, z16 = xyz16.unbind(dim=1)
    ss = corrected_ss[batch, :, x16, y16, z16]
    flat16 = x16 * 16 * 16 + y16 * 16 + z16
    physical = physical_tokens16[batch, flat16]

    base64 = xyz32 * 2
    offsets = torch.tensor(
        [
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 1],
            [1, 0, 0],
            [1, 0, 1],
            [1, 1, 0],
            [1, 1, 1],
        ],
        device=coords32.device,
        dtype=torch.long,
    )
    children = base64[:, None, :] + offsets[None]
    expanded_batch = batch[:, None].expand(-1, 8)
    child_logits = occupancy_logits64[
        expanded_batch,
        0,
        children[:, :, 0],
        children[:, :, 1],
        children[:, :, 2],
    ]
    occupancy = torch.stack(
        [child_logits.mean(dim=1), child_logits.amax(dim=1)], dim=1
    )
    evidence = torch.cat(
        [ss.float(), occupancy.float(), physical.float()], dim=1
    )
    audit = {
        "batch": batch,
        "xyz32": xyz32,
        "xyz16": xyz16,
        "flat16": flat16,
        "children64": children,
        "occupancy_mean_max": occupancy,
    }
    return evidence, audit


class SS2SLATSupportAdapter(nn.Module):
    """Zero-initialised corrected-SS support adapter for SLAT block 0."""

    def __init__(
        self,
        *,
        physical_channels: int = 1024,
        hidden_dim: int = 128,
        flow_channels: int = 1024,
    ) -> None:
        super().__init__()
        if min(physical_channels, hidden_dim, flow_channels) <= 0:
            raise ValueError("adapter channel counts must be positive")
        self.physical_channels = int(physical_channels)
        self.hidden_dim = int(hidden_dim)
        self.flow_channels = int(flow_channels)
        self.ss_proj = nn.Sequential(
            nn.LayerNorm(8), nn.Linear(8, hidden_dim), nn.SiLU()
        )
        self.occupancy_proj = nn.Sequential(
            # Keep the absolute logit level.  Centering two values with
            # LayerNorm would collapse mean/max evidence to almost a sign bit.
            nn.Linear(2, hidden_dim), nn.SiLU()
        )
        self.physical_proj = nn.Sequential(
            nn.LayerNorm(physical_channels),
            nn.Linear(physical_channels, hidden_dim),
            nn.SiLU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.output = nn.Linear(hidden_dim, flow_channels)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def metadata(self) -> dict[str, Any]:
        return {
            "mapping_version": SLAT_SUPPORT_MAPPING_VERSION,
            "physical_channels": self.physical_channels,
            "hidden_dim": self.hidden_dim,
            "flow_channels": self.flow_channels,
            "input_evidence": [
                "corrected_ss_latent_8",
                "occupancy_child_logits_mean_max_2",
                f"frozen_ss_physical_token_{self.physical_channels}",
            ],
            "output_zero_initialized": True,
            "injection": "native SLAT stem+APE output before transformer block 0",
        }

    def forward(self, evidence: torch.Tensor) -> torch.Tensor:
        expected = 8 + 2 + self.physical_channels
        _require_shape(evidence, (None, expected), "SLAT support evidence")
        ss = evidence[:, :8]
        occupancy = evidence[:, 8:10]
        physical = evidence[:, 10:]
        hidden = torch.cat(
            [
                self.ss_proj(ss),
                self.occupancy_proj(occupancy),
                self.physical_proj(physical),
            ],
            dim=1,
        )
        return self.output(self.fusion(hidden))


def _flow_core(flow: nn.Module) -> nn.Module:
    base = getattr(flow, "base_model", None)
    core = getattr(base, "model", None)
    return core if isinstance(core, nn.Module) else flow


def straight_through_exact_reference(
    value: torch.Tensor, reference: torch.Tensor
) -> torch.Tensor:
    """Use the exact reference value while preserving ``value`` gradients."""

    if value.shape != reference.shape:
        raise ValueError(
            f"straight-through shapes differ: {tuple(value.shape)} != "
            f"{tuple(reference.shape)}"
        )
    return reference.detach() + (value - value.detach())


def normalize_slat_delta_policy(
    *,
    delta_scale: float,
    delta_rms_ratio_cap: float | None,
) -> tuple[float, float | None]:
    """Validate and normalize the stock-anchored Direct-SLAT delta policy.

    A negative ratio cap is the CLI/checkpoint sentinel for a disabled cap.
    ``delta_scale=1`` with a disabled cap is the legacy Direct-SLAT behavior.
    """

    scale = float(delta_scale)
    if not math.isfinite(scale) or scale < 0.0:
        raise ValueError("SLAT delta scale must be finite and non-negative")
    cap = (
        None
        if delta_rms_ratio_cap is None or float(delta_rms_ratio_cap) < 0.0
        else float(delta_rms_ratio_cap)
    )
    if cap is not None and (not math.isfinite(cap) or cap < 0.0):
        raise ValueError(
            "SLAT delta RMS ratio cap must be finite and non-negative or disabled"
        )
    return scale, cap


def _zero_safe_sqrt(value: torch.Tensor, *, eps: float) -> torch.Tensor:
    """Return exact sqrt values with a finite derivative at an exact zero."""

    epsilon = float(eps)
    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("zero-safe sqrt epsilon must be finite and positive")
    positive = value > 0
    safe_input = torch.where(
        positive,
        value,
        torch.full_like(value, epsilon),
    )
    return torch.where(positive, safe_input.sqrt(), torch.zeros_like(value))


def bounded_slat_flow_delta(
    stock_feats: torch.Tensor,
    raw_full_feats: torch.Tensor,
    coords: torch.Tensor,
    *,
    delta_scale: float = 1.0,
    delta_rms_ratio_cap: float | None = None,
    bound_mode: str = SLAT_DELTA_BOUND_HARD,
    eps: float = 1.0e-12,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Bound ``raw_full-stock`` independently for every sparse batch item.

    ``hard_clip_v1`` is the frozen v2/v3 behavior. ``smooth_rms_v2`` uses
    ``allowed / sqrt(raw_rms**2 + allowed**2)`` and keeps its radial gradient;
    it therefore approaches the stock-relative radius smoothly rather than
    spending most training steps on a detached hard boundary.

    The legacy policy (scale 1, cap disabled) returns ``raw_full_feats``
    directly, preserving the previous value/dtype path. Statistics are
    accumulated in FP32. Per-batch tensors make the policy auditable when
    future callers use sparse batches larger than one.
    """

    _require_shape(stock_feats, (None, None), "stock SLAT velocity")
    _require_shape(raw_full_feats, (None, None), "raw Full SLAT velocity")
    if stock_feats.shape != raw_full_feats.shape:
        raise ValueError(
            "stock/raw Full feature shapes differ: "
            f"{tuple(stock_feats.shape)} != {tuple(raw_full_feats.shape)}"
        )
    _require_shape(coords, (int(stock_feats.shape[0]), 4), "SLAT delta coords")
    if int(stock_feats.shape[0]) <= 0 or int(stock_feats.shape[1]) <= 0:
        raise ValueError("SLAT delta features must be non-empty rank-2 tensors")
    scale, cap = normalize_slat_delta_policy(
        delta_scale=delta_scale,
        delta_rms_ratio_cap=delta_rms_ratio_cap,
    )
    mode = str(bound_mode)
    if mode not in SLAT_DELTA_BOUND_MODES:
        raise ValueError(f"unsupported SLAT delta bound mode={mode!r}")

    batch = coords[:, 0].long().to(device=stock_feats.device)
    if int(batch.min().item()) < 0:
        raise ValueError("SLAT delta coords contain a negative batch index")
    batch_ids, inverse = torch.unique(batch, sorted=True, return_inverse=True)
    batch_count = int(batch_ids.numel())
    channel_count = int(stock_feats.shape[1])
    point_counts = torch.bincount(inverse, minlength=batch_count).to(
        device=stock_feats.device, dtype=torch.float32
    )
    element_counts = point_counts * float(channel_count)

    stock32 = stock_feats.float()
    raw_delta = raw_full_feats.float() - stock32
    stock_square_sum = stock32.square().sum(dim=1).new_zeros(batch_count)
    delta_square_sum = raw_delta.square().sum(dim=1).new_zeros(batch_count)
    stock_square_sum.index_add_(0, inverse, stock32.square().sum(dim=1))
    delta_square_sum.index_add_(0, inverse, raw_delta.square().sum(dim=1))
    stock_rms_per_batch = _zero_safe_sqrt(
        stock_square_sum / element_counts,
        eps=eps,
    )
    raw_delta_rms_per_batch = _zero_safe_sqrt(
        delta_square_sum / element_counts,
        eps=eps,
    )

    clip_scale_per_batch = raw_delta_rms_per_batch.new_ones(batch_count)
    saturation_per_batch = torch.zeros(
        batch_count, device=raw_delta.device, dtype=torch.bool
    )
    if cap is not None:
        allowed = float(cap) * stock_rms_per_batch.clamp_min(float(eps))
        saturation_per_batch = raw_delta_rms_per_batch > allowed
        if mode == SLAT_DELTA_BOUND_HARD:
            ratio = allowed / raw_delta_rms_per_batch.clamp_min(float(eps))
            clip_scale_per_batch = torch.where(
                raw_delta_rms_per_batch > float(eps),
                torch.minimum(torch.ones_like(ratio), ratio),
                torch.ones_like(ratio),
            )
        else:
            denominator = torch.sqrt(
                raw_delta_rms_per_batch.square() + allowed.square()
            ).clamp_min(float(eps))
            smooth_scale = allowed / denominator
            clip_scale_per_batch = torch.where(
                raw_delta_rms_per_batch > float(eps),
                smooth_scale,
                torch.ones_like(smooth_scale),
            )
    # Preserve the frozen v2/v3 straight-through hard clip exactly. The V4
    # smooth map intentionally retains the gradient through its scale.
    scale_for_forward = (
        clip_scale_per_batch.detach()
        if mode == SLAT_DELTA_BOUND_HARD
        else clip_scale_per_batch
    )
    effective_scale_per_batch = scale_for_forward * float(scale)

    legacy_policy = scale == 1.0 and cap is None
    if legacy_policy:
        effective_feats = raw_full_feats
        effective_delta = raw_delta
    elif scale == 0.0:
        effective_feats = stock_feats
        effective_delta = raw_delta * 0.0
    else:
        effective_delta = (
            raw_delta * effective_scale_per_batch[inverse, None]
        )
        effective_feats = (
            stock32 + effective_delta
        ).to(dtype=raw_full_feats.dtype)

    clip_activated_per_batch = saturation_per_batch
    effective_square_sum = effective_delta.square().sum(dim=1).new_zeros(batch_count)
    effective_square_sum.index_add_(
        0,
        inverse,
        effective_delta.square().sum(dim=1),
    )
    stats = {
        "stock_velocity_rms": _zero_safe_sqrt(
            stock32.square().mean(), eps=eps
        ),
        "raw_flow_delta_rms": _zero_safe_sqrt(
            raw_delta.square().mean(), eps=eps
        ),
        "effective_flow_delta_rms": _zero_safe_sqrt(
            effective_delta.square().mean(), eps=eps
        ),
        "delta_clip_scale": clip_scale_per_batch.amin(),
        "delta_clip_scale_mean": clip_scale_per_batch.mean(),
        "delta_clip_activated": clip_activated_per_batch.any().to(torch.float32),
        "raw_flow_delta_abs_max": raw_delta.abs().amax(),
        "effective_flow_delta_abs_max": effective_delta.abs().amax(),
        "delta_scale": raw_delta.new_tensor(scale),
        "delta_rms_ratio_cap": raw_delta.new_tensor(
            -1.0 if cap is None else cap
        ),
        "delta_bound_mode_id": raw_delta.new_tensor(
            float(SLAT_DELTA_BOUND_MODES.index(mode))
        ),
        "batch_ids": batch_ids,
        "stock_velocity_rms_per_batch": stock_rms_per_batch,
        "raw_flow_delta_rms_per_batch": raw_delta_rms_per_batch,
        "effective_flow_delta_rms_per_batch": _zero_safe_sqrt(
            effective_square_sum / element_counts,
            eps=eps,
        ),
        "delta_clip_scale_per_batch": clip_scale_per_batch,
        "delta_clip_activated_per_batch": clip_activated_per_batch.to(
            torch.float32
        ),
    }
    return effective_feats, stats


def combine_slat_lora_support_budgets(
    stock_feats: torch.Tensor,
    raw_lora_feats: torch.Tensor,
    raw_joint_feats: torch.Tensor,
    coords: torch.Tensor,
    *,
    lora_delta_scale: float,
    lora_delta_rms_ratio_cap: float,
    support_delta_scale: float,
    support_delta_rms_ratio_cap: float,
    total_delta_scale: float,
    total_delta_rms_ratio_cap: float | None,
    bound_mode: str,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compose separately bounded LoRA and support-increment residuals.

    The LoRA branch is ``raw_lora - stock``.  The support branch is the
    incremental joint effect ``raw_joint - raw_lora``; this prevents a generic
    LoRA correction from silently consuming the entire Full trust region.
    Both branch budgets are stock-RMS-relative and a final total budget remains
    in place as the deployment safety bound.
    """

    if stock_feats.shape != raw_lora_feats.shape or stock_feats.shape != raw_joint_feats.shape:
        raise ValueError("stock/LoRA/joint SLAT feature shapes differ")
    lora_effective, lora_stats = bounded_slat_flow_delta(
        stock_feats,
        raw_lora_feats,
        coords,
        delta_scale=float(lora_delta_scale),
        delta_rms_ratio_cap=float(lora_delta_rms_ratio_cap),
        bound_mode=str(bound_mode),
    )
    raw_support_target = (
        stock_feats.float()
        + raw_joint_feats.float()
        - raw_lora_feats.float()
    ).to(dtype=raw_joint_feats.dtype)
    support_effective, support_stats = bounded_slat_flow_delta(
        stock_feats,
        raw_support_target,
        coords,
        delta_scale=float(support_delta_scale),
        delta_rms_ratio_cap=float(support_delta_rms_ratio_cap),
        bound_mode=str(bound_mode),
    )
    combined_raw = (
        stock_feats.float()
        + (lora_effective.float() - stock_feats.float())
        + (support_effective.float() - stock_feats.float())
    ).to(dtype=raw_joint_feats.dtype)
    effective, total_stats = bounded_slat_flow_delta(
        stock_feats,
        combined_raw,
        coords,
        delta_scale=float(total_delta_scale),
        delta_rms_ratio_cap=total_delta_rms_ratio_cap,
        bound_mode=str(bound_mode),
    )
    lora_delta = lora_effective.float() - stock_feats.float()
    support_delta = support_effective.float() - stock_feats.float()
    denominator = (
        lora_delta.square().sum().sqrt()
        * support_delta.square().sum().sqrt()
    ).clamp_min(1.0e-12)
    branch_cosine = (lora_delta * support_delta).sum() / denominator
    stats = dict(total_stats)
    for prefix, branch_stats in (
        ("lora_branch", lora_stats),
        ("support_branch", support_stats),
    ):
        for name in (
            "raw_flow_delta_rms",
            "effective_flow_delta_rms",
            "delta_clip_scale",
            "delta_clip_scale_mean",
            "delta_clip_activated",
            "raw_flow_delta_abs_max",
            "effective_flow_delta_abs_max",
        ):
            stats[f"{prefix}_{name}"] = branch_stats[name]
    stats["branch_delta_cosine"] = branch_cosine
    stats["residual_combination_policy_id"] = stock_feats.new_tensor(
        float(
            SLAT_RESIDUAL_COMBINATION_POLICIES.index(
                SLAT_RESIDUAL_COMBINATION_BRANCH_BUDGET
            )
        ),
        dtype=torch.float32,
    )
    return effective, stats


def stock_relative_residual_excess_loss(
    *,
    stock_rms_per_batch: torch.Tensor,
    raw_delta_rms_per_batch: torch.Tensor,
    ratio_cap: float,
    eps: float = 1.0e-12,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Penalize raw residual RMS that exceeds a stock-relative trust region."""

    if stock_rms_per_batch.shape != raw_delta_rms_per_batch.shape:
        raise ValueError("stock/raw residual RMS batch shapes differ")
    cap = float(ratio_cap)
    if not math.isfinite(cap) or cap < 0.0:
        raise ValueError("raw residual excess ratio cap must be finite and non-negative")
    ratio = raw_delta_rms_per_batch / stock_rms_per_batch.clamp_min(float(eps))
    excess = torch.relu(ratio - cap)
    return excess.square().mean(), ratio, excess


def correct_over_wrong_support_rank_loss(
    *,
    correct_loss: torch.Tensor,
    wrong_loss: torch.Tensor,
    margin: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Require correct support to beat object-disjoint support by ``margin``."""

    if correct_loss.ndim != 0 or wrong_loss.ndim != 0:
        raise ValueError("support ranking losses must be scalar tensors")
    margin_value = float(margin)
    if not math.isfinite(margin_value) or margin_value < 0.0:
        raise ValueError("correct-over-wrong support margin must be non-negative")
    advantage = wrong_loss - correct_loss
    return torch.relu(correct_loss - wrong_loss + margin_value), advantage


def deterministic_wrong_support_index(
    rows: list[dict[str, Any]],
    *,
    correct_object_uid: str,
    support_seed: int,
    selection_seed: int,
) -> int:
    """Select a deterministic object-disjoint support row, preferring the same seed."""

    if not rows:
        raise ValueError("cannot select wrong support from an empty row set")
    correct_uid = str(correct_object_uid)
    candidates = [
        index
        for index, row in enumerate(rows)
        if str(row.get("object_uid", "")) != correct_uid
        and int(row.get("support_seed", -1)) == int(support_seed)
    ]
    if not candidates:
        candidates = [
            index
            for index, row in enumerate(rows)
            if str(row.get("object_uid", "")) != correct_uid
        ]
    if not candidates:
        raise ValueError("wrong-support selection requires at least two objects")
    return candidates[int(selection_seed) % len(candidates)]


def deterministic_probability_event(*, seed: int, probability: float) -> bool:
    """Return a stable Bernoulli event without mutating global RNG state."""

    value = float(probability)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("probability must be finite and within [0, 1]")
    if value == 0.0:
        return False
    if value == 1.0:
        return True
    generator = torch.Generator().manual_seed(int(seed) % (2**63 - 1))
    return bool(torch.rand((), generator=generator).item() < value)


def deterministic_probability_partition(
    *,
    seed: int,
    probabilities: Iterable[float],
) -> int | None:
    """Choose at most one deterministic auxiliary event.

    The unassigned probability mass returns ``None``.  v3 uses this partition
    to keep wrong-support, support-dropout, and rollout graphs mutually
    exclusive, bounding peak memory on 24-GB GPUs.
    """

    values = tuple(float(value) for value in probabilities)
    if not values:
        raise ValueError("probability partition cannot be empty")
    if any(
        not math.isfinite(value) or not 0.0 <= value <= 1.0
        for value in values
    ):
        raise ValueError("partition probabilities must be finite within [0, 1]")
    total = sum(values)
    if total > 1.0 + 1.0e-12:
        raise ValueError("partition probabilities must sum to at most 1")
    generator = torch.Generator().manual_seed(int(seed) % (2**63 - 1))
    draw = float(torch.rand((), generator=generator).item())
    cumulative = 0.0
    for index, probability in enumerate(values):
        cumulative += probability
        if draw < cumulative:
            return index
    return None


def _combine_sparse_cfg(
    positive: Any,
    negative: Any,
    *,
    cfg_strength: float,
) -> Any:
    if positive.feats.shape != negative.feats.shape or not torch.equal(
        positive.coords, negative.coords
    ):
        raise ValueError("positive/negative SLAT predictions differ in sparse identity")
    strength = float(cfg_strength)
    return positive.replace(
        strength * positive.feats + (1.0 - strength) * negative.feats
    )


def combine_sparse_cfg(
    positive: Any,
    negative: Any,
    *,
    cfg_strength: float,
) -> Any:
    """Public, differentiable sparse CFG composition used by train and eval."""

    strength = float(cfg_strength)
    if not math.isfinite(strength) or strength < 0.0:
        raise ValueError("CFG strength must be finite and non-negative")
    return _combine_sparse_cfg(
        positive,
        negative,
        cfg_strength=strength,
    )


def normalize_cfg_interval(cfg_interval: Iterable[float]) -> tuple[float, float]:
    interval = tuple(float(value) for value in cfg_interval)
    if (
        len(interval) != 2
        or not all(math.isfinite(value) for value in interval)
        or not 0.0 <= interval[0] <= interval[1] <= 1.0
    ):
        raise ValueError("CFG interval must satisfy finite 0 <= lo <= hi <= 1")
    return interval


def cfg_interval_is_active(
    t_value: float,
    cfg_interval: Iterable[float],
) -> bool:
    """Match the native sampler's inclusive guidance-interval semantics."""

    value = float(t_value)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("normalized flow timestep must be finite within [0, 1]")
    lo, hi = normalize_cfg_interval(cfg_interval)
    return lo <= value <= hi


def native_flow_timestep_sequence(
    *,
    steps: int,
    rescale_t: float = 3.0,
) -> tuple[float, ...]:
    """Return the native FlowEuler schedule, including both endpoints.

    This mirrors the TRELLIS/ReconViaGen sampler transform
    ``r*t / (1 + (r-1)*t)`` and is kept here so V4 training visits exactly
    the same discrete timesteps as deployment.
    """

    count = int(steps)
    rescale = float(rescale_t)
    if count <= 0:
        raise ValueError("native flow schedule steps must be positive")
    if not math.isfinite(rescale) or rescale <= 0.0:
        raise ValueError("native flow rescale_t must be finite and positive")
    base = torch.linspace(1.0, 0.0, count + 1, dtype=torch.float64)
    transformed = rescale * base / (1.0 + (rescale - 1.0) * base)
    return tuple(float(value) for value in transformed.tolist())


def detached_sparse_euler_to_t(
    x_t: Any,
    velocity: Any,
    *,
    t_value: float,
    previous_t: float,
) -> tuple[Any, float]:
    """Take one detached native Euler transition to an explicit schedule time."""

    value = float(t_value)
    previous = float(previous_t)
    if not all(math.isfinite(item) for item in (value, previous)):
        raise ValueError("Euler timesteps must be finite")
    if not 0.0 <= previous < value <= 1.0:
        raise ValueError(
            "Euler transition requires 0 <= previous_t < t_value <= 1"
        )
    if x_t.feats.shape != velocity.feats.shape or not torch.equal(
        x_t.coords, velocity.coords
    ):
        raise ValueError("Euler state/velocity differ in sparse identity")
    applied = value - previous
    state = x_t.replace((x_t.feats - applied * velocity.feats).detach())
    return state, applied


def detached_sparse_euler_step(
    x_t: Any,
    velocity: Any,
    *,
    t_value: float,
    step_size: float,
) -> tuple[Any, float, float]:
    """Take one sampler-aligned detached Euler step toward a smaller timestep.

    ReconViaGen's native FlowEuler sampler uses
    ``x_prev = x_t - (t - t_prev) * velocity``.  This helper deliberately
    detaches the resulting state so the second model call trains on its own
    visited state without backpropagating through an unrolled solver.
    """

    value = float(t_value)
    requested = float(step_size)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("Euler t_value must be finite within [0, 1]")
    if not math.isfinite(requested) or requested <= 0.0:
        raise ValueError("Euler step_size must be finite and positive")
    applied = min(requested, value)
    if applied <= 0.0:
        raise ValueError("Euler rollout cannot advance from t=0")
    previous_t = value - applied
    previous, applied = detached_sparse_euler_to_t(
        x_t,
        velocity,
        t_value=value,
        previous_t=previous_t,
    )
    return previous, previous_t, applied


def lora_output_layers_are_exact_zero(flow: nn.Module) -> bool:
    """Return true only for PEFT's initial zero-output LoRA state."""

    outputs = [
        parameter
        for name, parameter in flow.named_parameters()
        if ".lora_B." in name
    ]
    if not outputs:
        raise RuntimeError("direct SLAT Flow contains no LoRA-B output layers")
    return all(bool(torch.count_nonzero(parameter.detach()).item() == 0) for parameter in outputs)


class DirectSupportSLATFlowModel(nn.Module):
    """Native SLAT Flow + LoRA with an exact native-stock bypass."""

    def __init__(self, lora_flow: nn.Module, support_adapter: SS2SLATSupportAdapter):
        super().__init__()
        self.flow = lora_flow
        self.support_adapter = support_adapter
        core = self.flow_core
        expected = {
            "resolution": 64,
            "in_channels": 8,
            "out_channels": 8,
            "patch_size": 2,
            "model_channels": support_adapter.flow_channels,
        }
        actual = {name: int(getattr(core, name)) for name in expected}
        if actual != expected:
            raise ValueError(f"unsupported native SLAT schema: {actual} != {expected}")

    @property
    def flow_core(self) -> nn.Module:
        return _flow_core(self.flow)

    def _core_forward(
        self,
        x: Any,
        t: torch.Tensor,
        cond: torch.Tensor | list[torch.Tensor],
        *,
        support: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None,
        support_scale: float,
    ) -> tuple[Any, dict[str, torch.Tensor]]:
        core = self.flow_core
        h = core.input_layer(x).type(core.dtype)
        t_emb = core.t_embedder(t)
        if core.share_mod:
            t_emb = core.adaLN_modulation(t_emb)
        t_emb = t_emb.type(core.dtype)
        if isinstance(cond, list):
            cond = [value.type(core.dtype) for value in cond]
        else:
            cond = cond.type(core.dtype)

        skips: list[torch.Tensor] = []
        for block in core.input_blocks:
            h = block(h, t_emb)
            skips.append(h.feats)
        if core.pe_mode == "ape":
            h = h + core.pos_embedder(h.coords[:, 1:]).type(core.dtype)

        zero = h.feats.new_zeros((), dtype=torch.float32)
        stats = {
            "support_token_rms": zero,
            "support_token_abs_max": zero,
            "support_point_count": zero,
            "support_present": zero,
        }
        if support is not None and float(support_scale) != 0.0:
            corrected_ss, occupancy_logits64, physical_tokens16 = support
            evidence, _ = gather_slat_support_evidence(
                h.coords,
                corrected_ss,
                occupancy_logits64,
                physical_tokens16,
            )
            tokens = self.support_adapter(evidence).to(dtype=h.dtype)
            tokens = tokens * float(support_scale)
            if tokens.shape != h.feats.shape:
                raise ValueError(
                    f"support tokens {tuple(tokens.shape)} != stem {tuple(h.feats.shape)}"
                )
            h = h.replace(h.feats + tokens)
            stats = {
                "support_token_rms": tokens.float().square().mean().sqrt(),
                "support_token_abs_max": tokens.float().abs().amax(),
                "support_point_count": tokens.new_tensor(
                    float(tokens.shape[0]), dtype=torch.float32
                ),
                "support_present": tokens.new_tensor(1.0, dtype=torch.float32),
            }

        for block in core.blocks:
            h = block(h, t_emb, cond)
        for block, skip in zip(core.out_blocks, reversed(skips)):
            if core.use_skip_connection:
                h = block(h.replace(torch.cat([h.feats, skip], dim=1)), t_emb)
            else:
                h = block(h, t_emb)
        h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))
        h = core.out_layer(h.type(x.dtype))
        return h, stats

    @torch.no_grad()
    def stock_prediction(self, x: Any, t: torch.Tensor, cond: Any) -> Any:
        with lora_disabled(self.flow):
            return self.flow(x, t, cond)

    def conditioned_prediction(
        self,
        x: Any,
        t: torch.Tensor,
        cond: Any,
        *,
        corrected_ss: torch.Tensor | None,
        occupancy_logits64: torch.Tensor | None,
        physical_tokens16: torch.Tensor | None,
        stock_velocity: Any | None = None,
        support_scale: float = 1.0,
        slat_delta_scale: float = 1.0,
        slat_delta_rms_ratio_cap: float | None = None,
        slat_delta_bound_mode: str = SLAT_DELTA_BOUND_HARD,
    ) -> tuple[Any, dict[str, torch.Tensor]]:
        support_values = (corrected_ss, occupancy_logits64, physical_tokens16)
        supplied = sum(torch.is_tensor(value) for value in support_values)
        if supplied not in (0, 3):
            raise ValueError("all three support tensors must be supplied together")
        support_present = supplied == 3
        if not support_present or float(support_scale) == 0.0:
            stock = (
                self.stock_prediction(x, t, cond)
                if stock_velocity is None
                else stock_velocity
            )
            zero = stock.feats.new_zeros((), dtype=torch.float32)
            return stock, {
                "support_token_rms": zero,
                "support_token_abs_max": zero,
                "support_point_count": zero,
                "support_present": zero,
                "flow_delta_rms": zero,
                "flow_delta_abs_max": zero,
                "stock_velocity_rms": stock.feats.float().square().mean().sqrt(),
                "raw_flow_delta_rms": zero,
                "effective_flow_delta_rms": zero,
                "delta_clip_scale": zero.new_tensor(1.0),
                "delta_clip_scale_mean": zero.new_tensor(1.0),
                "delta_clip_activated": zero,
                "raw_flow_delta_abs_max": zero,
                "effective_flow_delta_abs_max": zero,
                "delta_scale": zero.new_tensor(float(slat_delta_scale)),
                "delta_rms_ratio_cap": zero.new_tensor(
                    -1.0
                    if slat_delta_rms_ratio_cap is None
                    else float(slat_delta_rms_ratio_cap)
                ),
                "delta_bound_mode_id": zero.new_tensor(
                    float(
                        SLAT_DELTA_BOUND_MODES.index(
                            str(slat_delta_bound_mode)
                        )
                    )
                ),
            }
        stock = (
            self.stock_prediction(x, t, cond)
            if stock_velocity is None
            else stock_velocity
        )
        prediction, stats = self._core_forward(
            x,
            t,
            cond,
            support=(corrected_ss, occupancy_logits64, physical_tokens16),
            support_scale=float(support_scale),
        )
        zero_init_anchor = (
            float(stats["support_token_abs_max"].detach().item()) == 0.0
            and lora_output_layers_are_exact_zero(self.flow)
        )
        if zero_init_anchor:
            prediction = prediction.replace(
                straight_through_exact_reference(
                    prediction.feats,
                    stock.feats,
                )
            )
        effective_feats, delta_stats = bounded_slat_flow_delta(
            stock.feats,
            prediction.feats,
            prediction.coords,
            delta_scale=float(slat_delta_scale),
            delta_rms_ratio_cap=slat_delta_rms_ratio_cap,
            bound_mode=str(slat_delta_bound_mode),
        )
        if effective_feats is not prediction.feats:
            prediction = prediction.replace(effective_feats)
        stats = dict(stats)
        stats.update(delta_stats)
        stats.update({
            # Backward-compatible names now describe the returned/effective
            # Full prediction.  Raw values have explicit names above.
            "flow_delta_rms": delta_stats["effective_flow_delta_rms"],
            "flow_delta_abs_max": delta_stats["effective_flow_delta_abs_max"],
            "zero_init_stock_anchor": prediction.feats.new_tensor(
                float(zero_init_anchor), dtype=torch.float32
            ),
        })
        return prediction, stats

    def lora_only_prediction(self, x: Any, t: torch.Tensor, cond: Any) -> Any:
        return self.flow(x, t, cond)

    def post_cfg_conditioned_prediction(
        self,
        x: Any,
        t: torch.Tensor,
        positive_condition: Any,
        negative_condition: Any,
        *,
        corrected_ss: torch.Tensor | None,
        occupancy_logits64: torch.Tensor | None,
        physical_tokens16: torch.Tensor | None,
        stock_positive_velocity: Any | None = None,
        stock_negative_velocity: Any | None = None,
        cfg_strength: float,
        cfg_active: bool,
        support_scale: float = 1.0,
        slat_delta_scale: float = 1.0,
        slat_delta_rms_ratio_cap: float | None = None,
        slat_delta_bound_mode: str = SLAT_DELTA_BOUND_HARD,
        slat_residual_combination_policy: str = SLAT_RESIDUAL_COMBINATION_JOINT,
        slat_lora_delta_scale: float = 1.0,
        slat_lora_delta_rms_ratio_cap: float = 0.0,
        slat_support_delta_scale: float = 1.0,
        slat_support_delta_rms_ratio_cap: float = 0.0,
        support_active: bool = True,
        rollout_component: str = SLAT_ROLLOUT_COMPONENT_FULL,
    ) -> tuple[Any, Any, dict[str, torch.Tensor]]:
        """Train/evaluate the exact deployed post-CFG Direct-SLAT prediction.

        The negative Full branch intentionally remains frozen native stock,
        exactly as :class:`PostCFGSupportSLATRolloutFlow`.  The trust region is
        applied only after positive/negative CFG composition.
        """

        strength = float(cfg_strength)
        if not math.isfinite(strength) or strength < 0.0:
            raise ValueError("post-CFG strength must be finite and non-negative")
        component = str(rollout_component)
        if component not in SLAT_ROLLOUT_COMPONENTS:
            raise ValueError(f"unsupported SLAT rollout component={component!r}")
        combination_policy = str(slat_residual_combination_policy)
        if combination_policy not in SLAT_RESIDUAL_COMBINATION_POLICIES:
            raise ValueError(
                "unsupported SLAT residual combination policy="
                f"{combination_policy!r}"
            )
        if (
            combination_policy == SLAT_RESIDUAL_COMBINATION_BRANCH_BUDGET
            and (
                float(slat_lora_delta_rms_ratio_cap) <= 0.0
                or float(slat_support_delta_rms_ratio_cap) <= 0.0
            )
        ):
            raise ValueError("branch-budget SLAT requires positive branch caps")
        support_values = (
            corrected_ss,
            occupancy_logits64,
            physical_tokens16,
        )
        supplied_support = sum(
            torch.is_tensor(value) for value in support_values
        )
        if supplied_support not in (0, 3):
            raise ValueError(
                "all three post-CFG support tensors must be supplied together"
            )
        support_enabled = (
            bool(support_active)
            and supplied_support == 3
            and float(support_scale) != 0.0
        )
        stock_positive = (
            self.stock_prediction(x, t, positive_condition)
            if stock_positive_velocity is None
            else stock_positive_velocity
        )
        if bool(cfg_active) and strength != 1.0:
            stock_negative = (
                self.stock_prediction(x, t, negative_condition)
                if stock_negative_velocity is None
                else stock_negative_velocity
            )
            stock_reference = combine_sparse_cfg(
                stock_positive,
                stock_negative,
                cfg_strength=strength,
            )
            applied_strength = strength
        else:
            stock_reference = stock_positive
            applied_strength = 1.0

        if not support_enabled:
            prediction, inactive_stats = self.conditioned_prediction(
                x,
                t,
                positive_condition,
                corrected_ss=None,
                occupancy_logits64=None,
                physical_tokens16=None,
                stock_velocity=stock_reference,
                support_scale=0.0,
                slat_delta_scale=float(slat_delta_scale),
                slat_delta_rms_ratio_cap=slat_delta_rms_ratio_cap,
                slat_delta_bound_mode=str(slat_delta_bound_mode),
            )
            inactive_stats = dict(inactive_stats)
            inactive_stats.update(
                {
                    "cfg_active": prediction.feats.new_tensor(
                        float(bool(cfg_active)), dtype=torch.float32
                    ),
                    "support_active": prediction.feats.new_tensor(
                        0.0, dtype=torch.float32
                    ),
                    "applied_cfg_strength": prediction.feats.new_tensor(
                        float(applied_strength), dtype=torch.float32
                    ),
                    "positive_raw_flow_delta_rms": prediction.feats.new_zeros(
                        (), dtype=torch.float32
                    ),
                    "positive_raw_flow_delta_abs_max": prediction.feats.new_zeros(
                        (), dtype=torch.float32
                    ),
                    "residual_combination_policy_id": prediction.feats.new_tensor(
                        float(
                            SLAT_RESIDUAL_COMBINATION_POLICIES.index(
                                combination_policy
                            )
                        ),
                        dtype=torch.float32,
                    ),
                }
            )
            return prediction, stock_reference, inactive_stats

        raw_lora_positive = None
        if component == SLAT_ROLLOUT_COMPONENT_FULL:
            raw_positive, positive_stats = self.conditioned_prediction(
                x,
                t,
                positive_condition,
                corrected_ss=corrected_ss,
                occupancy_logits64=occupancy_logits64,
                physical_tokens16=physical_tokens16,
                stock_velocity=stock_positive,
                support_scale=float(support_scale),
                slat_delta_scale=1.0,
                slat_delta_rms_ratio_cap=None,
                slat_delta_bound_mode=str(slat_delta_bound_mode),
            )
            if combination_policy == SLAT_RESIDUAL_COMBINATION_BRANCH_BUDGET:
                raw_lora_positive = self.lora_only_prediction(
                    x,
                    t,
                    positive_condition,
                )
        elif component == SLAT_ROLLOUT_COMPONENT_LORA_ONLY:
            raw_positive = self.lora_only_prediction(
                x,
                t,
                positive_condition,
            )
            _, positive_stats = bounded_slat_flow_delta(
                stock_positive.feats,
                raw_positive.feats,
                raw_positive.coords,
                delta_scale=1.0,
                delta_rms_ratio_cap=None,
                bound_mode=str(slat_delta_bound_mode),
            )
        else:
            if not all(
                torch.is_tensor(value)
                for value in (
                    corrected_ss,
                    occupancy_logits64,
                    physical_tokens16,
                )
            ):
                raise ValueError("adapter-only rollout requires all support tensors")
            raw_positive = self.adapter_only_prediction(
                x,
                t,
                positive_condition,
                corrected_ss=corrected_ss,
                occupancy_logits64=occupancy_logits64,
                physical_tokens16=physical_tokens16,
                support_scale=float(support_scale),
            )
            _, positive_stats = bounded_slat_flow_delta(
                stock_positive.feats,
                raw_positive.feats,
                raw_positive.coords,
                delta_scale=1.0,
                delta_rms_ratio_cap=None,
                bound_mode=str(slat_delta_bound_mode),
            )
        if bool(cfg_active) and strength != 1.0:
            raw_guided = combine_sparse_cfg(
                raw_positive,
                stock_negative,
                cfg_strength=strength,
            )
            raw_lora_guided = (
                combine_sparse_cfg(
                    raw_lora_positive,
                    stock_negative,
                    cfg_strength=strength,
                )
                if raw_lora_positive is not None
                else None
            )
        else:
            raw_guided = raw_positive
            raw_lora_guided = raw_lora_positive

        positive_zero_init_anchor = positive_stats.get(
            "zero_init_stock_anchor"
        )
        post_cfg_zero_init_anchor = (
            component == SLAT_ROLLOUT_COMPONENT_FULL
            and positive_zero_init_anchor is not None
            and float(positive_zero_init_anchor.detach().item()) > 0.5
            and lora_output_layers_are_exact_zero(self.flow)
        )
        if post_cfg_zero_init_anchor:
            raw_guided = raw_guided.replace(
                straight_through_exact_reference(
                    raw_guided.feats,
                    stock_reference.feats,
                )
            )
            if raw_lora_guided is not None:
                raw_lora_guided = raw_lora_guided.replace(
                    straight_through_exact_reference(
                        raw_lora_guided.feats,
                        stock_reference.feats,
                    )
                )

        if (
            component == SLAT_ROLLOUT_COMPONENT_FULL
            and combination_policy
            == SLAT_RESIDUAL_COMBINATION_BRANCH_BUDGET
        ):
            if raw_lora_guided is None:
                raise AssertionError("branch-budget Full lacks LoRA-only velocity")
            effective_feats, guided_stats = combine_slat_lora_support_budgets(
                stock_reference.feats,
                raw_lora_guided.feats,
                raw_guided.feats,
                raw_guided.coords,
                lora_delta_scale=float(slat_lora_delta_scale),
                lora_delta_rms_ratio_cap=float(
                    slat_lora_delta_rms_ratio_cap
                ),
                support_delta_scale=float(slat_support_delta_scale),
                support_delta_rms_ratio_cap=float(
                    slat_support_delta_rms_ratio_cap
                ),
                total_delta_scale=float(slat_delta_scale),
                total_delta_rms_ratio_cap=slat_delta_rms_ratio_cap,
                bound_mode=str(slat_delta_bound_mode),
            )
        else:
            effective_feats, guided_stats = bounded_slat_flow_delta(
                stock_reference.feats,
                raw_guided.feats,
                raw_guided.coords,
                delta_scale=float(slat_delta_scale),
                delta_rms_ratio_cap=slat_delta_rms_ratio_cap,
                bound_mode=str(slat_delta_bound_mode),
            )
            guided_stats["residual_combination_policy_id"] = (
                stock_reference.feats.new_tensor(
                    float(
                        SLAT_RESIDUAL_COMBINATION_POLICIES.index(
                            combination_policy
                        )
                    ),
                    dtype=torch.float32,
                )
            )
        prediction = (
            raw_guided
            if effective_feats is raw_guided.feats
            else raw_guided.replace(effective_feats)
        )
        stats = dict(positive_stats)
        stats.update(
            {
                "positive_raw_flow_delta_rms": positive_stats[
                    "raw_flow_delta_rms"
                ],
                "positive_raw_flow_delta_abs_max": positive_stats[
                    "raw_flow_delta_abs_max"
                ],
            }
        )
        stats.update(guided_stats)
        if raw_lora_guided is not None:
            # V5 already computes this branch to split LoRA and support
            # residual budgets.  Keep the tensor-bearing result private so
            # support-dropout training can reuse the same graph.
            stats["_raw_lora_guided"] = raw_lora_guided
        stats.update(
            {
                "flow_delta_rms": guided_stats["effective_flow_delta_rms"],
                "flow_delta_abs_max": guided_stats[
                    "effective_flow_delta_abs_max"
                ],
                "cfg_active": prediction.feats.new_tensor(
                    float(bool(cfg_active)), dtype=torch.float32
                ),
                "support_active": prediction.feats.new_tensor(
                    1.0, dtype=torch.float32
                ),
                "applied_cfg_strength": prediction.feats.new_tensor(
                    float(applied_strength), dtype=torch.float32
                ),
                "rollout_component_id": prediction.feats.new_tensor(
                    float(SLAT_ROLLOUT_COMPONENTS.index(component)),
                    dtype=torch.float32,
                ),
                "post_cfg_zero_init_stock_anchor": (
                    prediction.feats.new_tensor(
                        float(post_cfg_zero_init_anchor),
                        dtype=torch.float32,
                    )
                ),
            }
        )
        return prediction, stock_reference, stats

    def adapter_only_prediction(
        self,
        x: Any,
        t: torch.Tensor,
        cond: Any,
        *,
        corrected_ss: torch.Tensor,
        occupancy_logits64: torch.Tensor,
        physical_tokens16: torch.Tensor,
        support_scale: float = 1.0,
    ) -> Any:
        with lora_disabled(self.flow):
            prediction, _ = self._core_forward(
                x,
                t,
                cond,
                support=(corrected_ss, occupancy_logits64, physical_tokens16),
                support_scale=float(support_scale),
            )
        return prediction

    def forward(
        self,
        x: Any,
        t: torch.Tensor,
        cond: Any,
        *,
        corrected_ss: torch.Tensor | None,
        occupancy_logits64: torch.Tensor | None,
        physical_tokens16: torch.Tensor | None,
        stock_velocity: Any | None = None,
        support_scale: float = 1.0,
        slat_delta_scale: float = 1.0,
        slat_delta_rms_ratio_cap: float | None = None,
        slat_delta_bound_mode: str = SLAT_DELTA_BOUND_HARD,
        slat_residual_combination_policy: str = SLAT_RESIDUAL_COMBINATION_JOINT,
        slat_lora_delta_scale: float = 1.0,
        slat_lora_delta_rms_ratio_cap: float = 0.0,
        slat_support_delta_scale: float = 1.0,
        slat_support_delta_rms_ratio_cap: float = 0.0,
        support_active: bool = True,
        wrong_corrected_ss: torch.Tensor | None = None,
        wrong_occupancy_logits64: torch.Tensor | None = None,
        wrong_physical_tokens16: torch.Tensor | None = None,
        compute_lora_without_support: bool = False,
        negative_condition: Any | None = None,
        stock_negative_velocity: Any | None = None,
        post_cfg_strength: float = 1.0,
        post_cfg_active: bool = False,
        rollout_step_size: float = 0.0,
        rollout_cfg_active: bool = False,
        rollout_previous_t_values: Iterable[float] = (),
        rollout_cfg_active_values: Iterable[bool] = (),
        rollout_support_active_values: Iterable[bool] = (),
        rollout_supervision_policy: str = SLAT_ROLLOUT_SUPERVISION_ALL_VISITED,
    ) -> tuple[Any, dict[str, torch.Tensor], Any | None, Any | None]:
        post_cfg = negative_condition is not None
        if post_cfg:
            prediction, stock_reference, stats = (
                self.post_cfg_conditioned_prediction(
                    x,
                    t,
                    cond,
                    negative_condition,
                    corrected_ss=corrected_ss,
                    occupancy_logits64=occupancy_logits64,
                    physical_tokens16=physical_tokens16,
                    stock_positive_velocity=stock_velocity,
                    stock_negative_velocity=stock_negative_velocity,
                    cfg_strength=float(post_cfg_strength),
                    cfg_active=bool(post_cfg_active),
                    support_scale=support_scale,
                    slat_delta_scale=slat_delta_scale,
                    slat_delta_rms_ratio_cap=slat_delta_rms_ratio_cap,
                    slat_delta_bound_mode=slat_delta_bound_mode,
                    slat_residual_combination_policy=(
                        slat_residual_combination_policy
                    ),
                    slat_lora_delta_scale=slat_lora_delta_scale,
                    slat_lora_delta_rms_ratio_cap=(
                        slat_lora_delta_rms_ratio_cap
                    ),
                    slat_support_delta_scale=slat_support_delta_scale,
                    slat_support_delta_rms_ratio_cap=(
                        slat_support_delta_rms_ratio_cap
                    ),
                    support_active=bool(support_active),
                )
            )
        else:
            prediction, stats = self.conditioned_prediction(
                x,
                t,
                cond,
                corrected_ss=corrected_ss,
                occupancy_logits64=occupancy_logits64,
                physical_tokens16=physical_tokens16,
                stock_velocity=stock_velocity,
                support_scale=support_scale,
                slat_delta_scale=slat_delta_scale,
                slat_delta_rms_ratio_cap=slat_delta_rms_ratio_cap,
                slat_delta_bound_mode=slat_delta_bound_mode,
            )
            stock_reference = (
                self.stock_prediction(x, t, cond)
                if stock_velocity is None
                else stock_velocity
            )
        stats = dict(stats)
        # Tensor-bearing private entries are consumed by the training loop and
        # are never serialized as scalar audit statistics.
        stats["_stock_reference"] = stock_reference
        wrong_values = (
            wrong_corrected_ss,
            wrong_occupancy_logits64,
            wrong_physical_tokens16,
        )
        wrong_prediction = None
        if any(value is not None for value in wrong_values):
            if not all(torch.is_tensor(value) for value in wrong_values):
                raise ValueError("all three wrong-support tensors must be supplied together")
            if post_cfg:
                wrong_prediction, wrong_stock_reference, _ = (
                    self.post_cfg_conditioned_prediction(
                        x,
                        t,
                        cond,
                        negative_condition,
                        corrected_ss=wrong_corrected_ss,
                        occupancy_logits64=wrong_occupancy_logits64,
                        physical_tokens16=wrong_physical_tokens16,
                        stock_positive_velocity=stock_velocity,
                        stock_negative_velocity=stock_negative_velocity,
                        cfg_strength=float(post_cfg_strength),
                        cfg_active=bool(post_cfg_active),
                        support_scale=support_scale,
                        slat_delta_scale=slat_delta_scale,
                        slat_delta_rms_ratio_cap=slat_delta_rms_ratio_cap,
                        slat_delta_bound_mode=slat_delta_bound_mode,
                        slat_residual_combination_policy=(
                            slat_residual_combination_policy
                        ),
                        slat_lora_delta_scale=slat_lora_delta_scale,
                        slat_lora_delta_rms_ratio_cap=(
                            slat_lora_delta_rms_ratio_cap
                        ),
                        slat_support_delta_scale=slat_support_delta_scale,
                        slat_support_delta_rms_ratio_cap=(
                            slat_support_delta_rms_ratio_cap
                        ),
                        support_active=bool(support_active),
                    )
                )
                if not torch.equal(
                    wrong_stock_reference.feats, stock_reference.feats
                ):
                    raise RuntimeError(
                        "correct/wrong post-CFG stock references differ"
                    )
            else:
                wrong_prediction, _ = self.conditioned_prediction(
                    x,
                    t,
                    cond,
                    corrected_ss=wrong_corrected_ss,
                    occupancy_logits64=wrong_occupancy_logits64,
                    physical_tokens16=wrong_physical_tokens16,
                    stock_velocity=stock_velocity,
                    support_scale=support_scale,
                    slat_delta_scale=slat_delta_scale,
                    slat_delta_rms_ratio_cap=slat_delta_rms_ratio_cap,
                    slat_delta_bound_mode=slat_delta_bound_mode,
                )
        lora_without_support = None
        if bool(compute_lora_without_support):
            cached_lora_guided = stats.get("_raw_lora_guided")
            if cached_lora_guided is not None:
                lora_without_support = cached_lora_guided
            else:
                lora_positive = self.lora_only_prediction(x, t, cond)
                if (
                    post_cfg
                    and bool(post_cfg_active)
                    and float(post_cfg_strength) != 1.0
                ):
                    stock_negative = (
                        self.stock_prediction(x, t, negative_condition)
                        if stock_negative_velocity is None
                        else stock_negative_velocity
                    )
                    lora_without_support = combine_sparse_cfg(
                        lora_positive,
                        stock_negative,
                        cfg_strength=float(post_cfg_strength),
                    )
                else:
                    lora_without_support = lora_positive

        previous_t_values = tuple(float(value) for value in rollout_previous_t_values)
        cfg_active_values = tuple(bool(value) for value in rollout_cfg_active_values)
        support_active_values = tuple(
            bool(value) for value in rollout_support_active_values
        )
        if previous_t_values:
            if not post_cfg:
                raise ValueError("schedule rollout training requires post-CFG mode")
            if rollout_supervision_policy not in SLAT_ROLLOUT_SUPERVISION_POLICIES:
                raise ValueError(
                    "unsupported rollout supervision policy="
                    f"{rollout_supervision_policy!r}"
                )
            if len(cfg_active_values) != len(previous_t_values):
                raise ValueError("rollout CFG flags do not match rollout horizon")
            if len(support_active_values) != len(previous_t_values):
                raise ValueError("rollout support flags do not match rollout horizon")
            normalized_t = t.detach().float().reshape(-1) / 1000.0
            if not torch.allclose(normalized_t, normalized_t[:1]):
                raise ValueError("schedule rollout requires one shared timestep")
            current_t = float(normalized_t[0].item())
            current_x = x
            current_velocity = prediction
            rollout_predictions = []
            rollout_stocks = []
            rollout_states = []
            rollout_stats_rows = []
            supervised_t_values = []
            applied_steps = []
            outer_grad_enabled = torch.is_grad_enabled()
            rollout_rows = tuple(
                zip(
                    previous_t_values,
                    cfg_active_values,
                    support_active_values,
                )
            )
            for rollout_index, (previous_t, cfg_flag, support_flag) in enumerate(
                rollout_rows
            ):
                current_x, applied_step = detached_sparse_euler_to_t(
                    current_x,
                    current_velocity,
                    t_value=current_t,
                    previous_t=previous_t,
                )
                rollout_t = torch.full_like(t, 1000.0 * previous_t)
                retain_supervision_graph = (
                    rollout_supervision_policy
                    == SLAT_ROLLOUT_SUPERVISION_ALL_VISITED
                    or rollout_index == len(rollout_rows) - 1
                )
                # V5.1 keeps the numerical 2/4/8-step trajectory unchanged,
                # while intermediate state-generation calls are detached and
                # graph-free.  Only the selected terminal state retains the
                # large joint Full + LoRA-only sparse Transformer graph.
                with torch.set_grad_enabled(
                    outer_grad_enabled and retain_supervision_graph
                ):
                    current_velocity, current_stock, current_stats = (
                        self.post_cfg_conditioned_prediction(
                            current_x,
                            rollout_t,
                            cond,
                            negative_condition,
                            corrected_ss=corrected_ss,
                            occupancy_logits64=occupancy_logits64,
                            physical_tokens16=physical_tokens16,
                            cfg_strength=float(post_cfg_strength),
                            cfg_active=cfg_flag,
                            support_scale=support_scale,
                            slat_delta_scale=slat_delta_scale,
                            slat_delta_rms_ratio_cap=slat_delta_rms_ratio_cap,
                            slat_delta_bound_mode=slat_delta_bound_mode,
                            slat_residual_combination_policy=(
                                slat_residual_combination_policy
                            ),
                            slat_lora_delta_scale=slat_lora_delta_scale,
                            slat_lora_delta_rms_ratio_cap=(
                                slat_lora_delta_rms_ratio_cap
                            ),
                            slat_support_delta_scale=slat_support_delta_scale,
                            slat_support_delta_rms_ratio_cap=(
                                slat_support_delta_rms_ratio_cap
                            ),
                            support_active=support_flag,
                        )
                    )
                if retain_supervision_graph:
                    rollout_states.append(current_x)
                    rollout_predictions.append(current_velocity)
                    rollout_stocks.append(current_stock)
                    rollout_stats_rows.append(current_stats)
                    supervised_t_values.append(previous_t)
                applied_steps.append(applied_step)
                current_t = previous_t
            stats["_rollout_predictions"] = rollout_predictions
            stats["_rollout_stock_references"] = rollout_stocks
            stats["_rollout_states"] = rollout_states
            stats["_rollout_t_values"] = tuple(supervised_t_values)
            stats["_rollout_generated_t_values"] = previous_t_values
            stats["rollout_horizon"] = prediction.feats.new_tensor(
                len(previous_t_values), dtype=torch.float32
            )
            stats["rollout_supervised_state_count"] = prediction.feats.new_tensor(
                len(supervised_t_values), dtype=torch.float32
            )
            stats["rollout_t"] = prediction.feats.new_tensor(
                previous_t_values[-1], dtype=torch.float32
            )
            stats["rollout_step_size"] = prediction.feats.new_tensor(
                sum(applied_steps), dtype=torch.float32
            )
            stats["rollout_flow_delta_rms"] = rollout_stats_rows[-1][
                "effective_flow_delta_rms"
            ]
            stats["rollout_delta_clip_scale"] = rollout_stats_rows[-1][
                "delta_clip_scale"
            ]

        requested_rollout_step = float(rollout_step_size)
        if requested_rollout_step > 0.0:
            if previous_t_values:
                raise ValueError(
                    "fixed-step and schedule rollout cannot be enabled together"
                )
            if not post_cfg:
                raise ValueError("detached rollout training requires post-CFG mode")
            if t.numel() <= 0:
                raise ValueError("detached rollout received an empty timestep")
            normalized_t = t.detach().float().reshape(-1) / 1000.0
            if not torch.allclose(normalized_t, normalized_t[:1]):
                raise ValueError("detached rollout requires one shared timestep")
            rollout_x, rollout_t_value, applied_step = detached_sparse_euler_step(
                x,
                prediction,
                t_value=float(normalized_t[0].item()),
                step_size=requested_rollout_step,
            )
            rollout_t = torch.full_like(t, 1000.0 * rollout_t_value)
            rollout_prediction, rollout_stock, rollout_stats = (
                self.post_cfg_conditioned_prediction(
                    rollout_x,
                    rollout_t,
                    cond,
                    negative_condition,
                    corrected_ss=corrected_ss,
                    occupancy_logits64=occupancy_logits64,
                    physical_tokens16=physical_tokens16,
                    cfg_strength=float(post_cfg_strength),
                    cfg_active=bool(rollout_cfg_active),
                    support_scale=support_scale,
                    slat_delta_scale=slat_delta_scale,
                    slat_delta_rms_ratio_cap=slat_delta_rms_ratio_cap,
                    slat_delta_bound_mode=slat_delta_bound_mode,
                    slat_residual_combination_policy=(
                        slat_residual_combination_policy
                    ),
                    slat_lora_delta_scale=slat_lora_delta_scale,
                    slat_lora_delta_rms_ratio_cap=(
                        slat_lora_delta_rms_ratio_cap
                    ),
                    slat_support_delta_scale=slat_support_delta_scale,
                    slat_support_delta_rms_ratio_cap=(
                        slat_support_delta_rms_ratio_cap
                    ),
                    support_active=bool(support_active),
                )
            )
            stats["_rollout_prediction"] = rollout_prediction
            stats["_rollout_stock_reference"] = rollout_stock
            stats["rollout_t"] = prediction.feats.new_tensor(
                rollout_t_value, dtype=torch.float32
            )
            stats["rollout_step_size"] = prediction.feats.new_tensor(
                applied_step, dtype=torch.float32
            )
            stats["rollout_flow_delta_rms"] = rollout_stats[
                "effective_flow_delta_rms"
            ]
            stats["rollout_delta_clip_scale"] = rollout_stats[
                "delta_clip_scale"
            ]
        return prediction, stats, wrong_prediction, lora_without_support


class NativeStockSLATFlow(nn.Module):
    def __init__(self, model: DirectSupportSLATFlowModel) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: Any, t: torch.Tensor, condition: Any) -> Any:
        return self.model.stock_prediction(x, t, condition)


class PositiveSupportSLATRolloutFlow(nn.Module):
    """Use SS support/LoRA only on the positive CFG condition object."""

    def __init__(
        self,
        model: DirectSupportSLATFlowModel,
        positive_condition: Any,
        support: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        *,
        support_scale: float,
        slat_delta_scale: float = 1.0,
        slat_delta_rms_ratio_cap: float | None = None,
        slat_delta_bound_mode: str = SLAT_DELTA_BOUND_HARD,
        support_interval_policy: str = SLAT_SUPPORT_INTERVAL_ALL,
    ) -> None:
        super().__init__()
        self.model = model
        self.positive_condition = positive_condition
        self.support = support
        self.support_scale = float(support_scale)
        self.slat_delta_bound_mode = str(slat_delta_bound_mode)
        if self.slat_delta_bound_mode not in SLAT_DELTA_BOUND_MODES:
            raise ValueError(
                f"unsupported SLAT delta bound mode={self.slat_delta_bound_mode!r}"
            )
        self.support_interval_policy = str(support_interval_policy)
        if self.support_interval_policy not in SLAT_SUPPORT_INTERVAL_POLICIES:
            raise ValueError(
                "unsupported SLAT support interval policy="
                f"{self.support_interval_policy!r}"
            )
        (
            self.slat_delta_scale,
            self.slat_delta_rms_ratio_cap,
        ) = normalize_slat_delta_policy(
            delta_scale=slat_delta_scale,
            delta_rms_ratio_cap=slat_delta_rms_ratio_cap,
        )
        self.positive_calls = 0
        self.negative_calls = 0
        self.positive_stats: list[dict[str, float]] = []

    def _is_positive(self, condition: Any) -> bool:
        if condition is self.positive_condition:
            return True
        if isinstance(condition, list) and isinstance(self.positive_condition, list):
            if len(condition) != len(self.positive_condition):
                return False
            return all(
                left is right
                or (
                    torch.is_tensor(left)
                    and torch.is_tensor(right)
                    and left.shape == right.shape
                    and left.data_ptr() == right.data_ptr()
                )
                for left, right in zip(condition, self.positive_condition)
            )
        return False

    def forward(self, x: Any, t: torch.Tensor, condition: Any) -> Any:
        if not self._is_positive(condition):
            self.negative_calls += 1
            return self.model.stock_prediction(x, t, condition)
        self.positive_calls += 1
        prediction, stats = self.model.conditioned_prediction(
            x,
            t,
            condition,
            corrected_ss=self.support[0],
            occupancy_logits64=self.support[1],
            physical_tokens16=self.support[2],
            support_scale=self.support_scale,
            slat_delta_scale=self.slat_delta_scale,
            slat_delta_rms_ratio_cap=self.slat_delta_rms_ratio_cap,
            slat_delta_bound_mode=self.slat_delta_bound_mode,
        )
        scalar_names = (
            "stock_velocity_rms",
            "raw_flow_delta_rms",
            "effective_flow_delta_rms",
            "delta_clip_scale",
            "delta_clip_scale_mean",
            "delta_clip_activated",
            "raw_flow_delta_abs_max",
            "effective_flow_delta_abs_max",
        )
        self.positive_stats.append(
            {
                name: float(stats[name].detach().float().item())
                for name in scalar_names
            }
        )
        return prediction

    def stats_summary(self) -> dict[str, Any]:
        """Return JSON-safe positive-CFG bounded-delta rollout statistics."""

        result: dict[str, Any] = {
            "policy_version": SLAT_DELTA_POLICY_VERSION,
            "slat_delta_scale": self.slat_delta_scale,
            "slat_delta_rms_ratio_cap": self.slat_delta_rms_ratio_cap,
            "positive_calls": self.positive_calls,
            "negative_calls": self.negative_calls,
            "delta_clip_activated_calls": int(
                sum(row["delta_clip_activated"] > 0.5 for row in self.positive_stats)
            ),
        }
        if not self.positive_stats:
            result["metrics"] = {}
            return result
        result["metrics"] = {
            name: {
                "mean": float(
                    sum(row[name] for row in self.positive_stats)
                    / len(self.positive_stats)
                ),
                "min": float(min(row[name] for row in self.positive_stats)),
                "max": float(max(row[name] for row in self.positive_stats)),
            }
            for name in self.positive_stats[0]
        }
        return result


class PostCFGSupportSLATRolloutFlow(nn.Module):
    """Bound the Direct-SLAT residual after positive/negative CFG composition.

    Call the native sampler with external ``cfg_strength=1``.  This wrapper
    reconstructs the frozen stock-guided and raw Direct-SLAT-guided velocities
    internally, then applies the stock-relative trust region to their final
    guided difference.
    """

    def __init__(
        self,
        model: DirectSupportSLATFlowModel,
        positive_condition: Any,
        negative_condition: Any,
        support: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        *,
        cfg_strength: float,
        cfg_interval: Iterable[float],
        support_scale: float,
        slat_delta_scale: float = 1.0,
        slat_delta_rms_ratio_cap: float | None = None,
        slat_delta_bound_mode: str = SLAT_DELTA_BOUND_HARD,
        slat_residual_combination_policy: str = SLAT_RESIDUAL_COMBINATION_JOINT,
        slat_lora_delta_scale: float = 1.0,
        slat_lora_delta_rms_ratio_cap: float = 0.0,
        slat_support_delta_scale: float = 1.0,
        slat_support_delta_rms_ratio_cap: float = 0.0,
        support_interval_policy: str = SLAT_SUPPORT_INTERVAL_ALL,
        rollout_component: str = SLAT_ROLLOUT_COMPONENT_FULL,
    ) -> None:
        super().__init__()
        interval = tuple(float(value) for value in cfg_interval)
        if len(interval) != 2 or not 0.0 <= interval[0] <= interval[1] <= 1.0:
            raise ValueError("post-CFG SLAT interval must satisfy 0 <= lo <= hi <= 1")
        if not math.isfinite(float(cfg_strength)) or float(cfg_strength) < 0.0:
            raise ValueError("post-CFG SLAT strength must be finite and non-negative")
        self.model = model
        self.positive_condition = positive_condition
        self.negative_condition = negative_condition
        self.support = support
        self.cfg_strength = float(cfg_strength)
        self.cfg_interval = interval
        self.support_scale = float(support_scale)
        self.rollout_component = str(rollout_component)
        if self.rollout_component not in SLAT_ROLLOUT_COMPONENTS:
            raise ValueError(
                f"unsupported SLAT rollout component={self.rollout_component!r}"
            )
        self.slat_delta_bound_mode = str(slat_delta_bound_mode)
        if self.slat_delta_bound_mode not in SLAT_DELTA_BOUND_MODES:
            raise ValueError(
                f"unsupported SLAT delta bound mode={self.slat_delta_bound_mode!r}"
            )
        self.slat_residual_combination_policy = str(
            slat_residual_combination_policy
        )
        if (
            self.slat_residual_combination_policy
            not in SLAT_RESIDUAL_COMBINATION_POLICIES
        ):
            raise ValueError(
                "unsupported SLAT residual combination policy="
                f"{self.slat_residual_combination_policy!r}"
            )
        self.slat_lora_delta_scale = float(slat_lora_delta_scale)
        self.slat_lora_delta_rms_ratio_cap = float(
            slat_lora_delta_rms_ratio_cap
        )
        self.slat_support_delta_scale = float(slat_support_delta_scale)
        self.slat_support_delta_rms_ratio_cap = float(
            slat_support_delta_rms_ratio_cap
        )
        self.support_interval_policy = str(support_interval_policy)
        if self.support_interval_policy not in SLAT_SUPPORT_INTERVAL_POLICIES:
            raise ValueError(
                "unsupported SLAT support interval policy="
                f"{self.support_interval_policy!r}"
            )
        (
            self.slat_delta_scale,
            self.slat_delta_rms_ratio_cap,
        ) = normalize_slat_delta_policy(
            delta_scale=slat_delta_scale,
            delta_rms_ratio_cap=slat_delta_rms_ratio_cap,
        )
        self.positive_calls = 0
        self.negative_calls = 0
        self.guided_stats: list[dict[str, float]] = []

    def forward(self, x: Any, t: torch.Tensor, condition: Any) -> Any:
        del condition
        if t.numel() <= 0:
            raise ValueError("post-CFG SLAT received an empty timestep")
        t_values = t.detach().float().reshape(-1) / 1000.0
        if not torch.allclose(t_values, t_values[:1]):
            raise ValueError("post-CFG SLAT currently requires one shared timestep")
        t_value = float(t_values[0].item())
        cfg_active = self.cfg_interval[0] <= t_value <= self.cfg_interval[1]

        support_active = (
            True
            if self.support_interval_policy == SLAT_SUPPORT_INTERVAL_ALL
            else cfg_active
        )
        prediction, _, guided_stats = self.model.post_cfg_conditioned_prediction(
            x,
            t,
            self.positive_condition,
            self.negative_condition,
            corrected_ss=self.support[0],
            occupancy_logits64=self.support[1],
            physical_tokens16=self.support[2],
            cfg_strength=self.cfg_strength,
            cfg_active=cfg_active,
            support_scale=self.support_scale,
            slat_delta_scale=self.slat_delta_scale,
            slat_delta_rms_ratio_cap=self.slat_delta_rms_ratio_cap,
            slat_delta_bound_mode=self.slat_delta_bound_mode,
            slat_residual_combination_policy=(
                self.slat_residual_combination_policy
            ),
            slat_lora_delta_scale=self.slat_lora_delta_scale,
            slat_lora_delta_rms_ratio_cap=self.slat_lora_delta_rms_ratio_cap,
            slat_support_delta_scale=self.slat_support_delta_scale,
            slat_support_delta_rms_ratio_cap=(
                self.slat_support_delta_rms_ratio_cap
            ),
            support_active=support_active,
            rollout_component=self.rollout_component,
        )
        self.positive_calls += int(support_active)
        self.negative_calls += int(cfg_active and self.cfg_strength != 1.0)
        applied_strength = (
            self.cfg_strength
            if cfg_active and self.cfg_strength != 1.0
            else 1.0
        )
        row = {
            "t": t_value,
            "cfg_active": float(cfg_active),
            "support_active": float(support_active),
            "applied_cfg_strength": float(applied_strength),
            "stock_guided_velocity_rms": float(
                guided_stats["stock_velocity_rms"].detach().item()
            ),
            "raw_guided_delta_rms": float(
                guided_stats["raw_flow_delta_rms"].detach().item()
            ),
            "effective_guided_delta_rms": float(
                guided_stats["effective_flow_delta_rms"].detach().item()
            ),
            "guided_delta_clip_scale": float(
                guided_stats["delta_clip_scale"].detach().item()
            ),
            "guided_delta_clip_activated": float(
                guided_stats["delta_clip_activated"].detach().item()
            ),
            "raw_guided_delta_abs_max": float(
                guided_stats["raw_flow_delta_abs_max"].detach().item()
            ),
            "effective_guided_delta_abs_max": float(
                guided_stats["effective_flow_delta_abs_max"].detach().item()
            ),
            "positive_raw_delta_rms": float(
                guided_stats["positive_raw_flow_delta_rms"].detach().item()
            ),
        }
        self.guided_stats.append(row)
        return prediction

    def stats_summary(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "policy_version": SLAT_GUIDED_DELTA_POLICY_V2,
            "slat_delta_scale": self.slat_delta_scale,
            "slat_delta_rms_ratio_cap": self.slat_delta_rms_ratio_cap,
            "slat_delta_bound_mode": self.slat_delta_bound_mode,
            "slat_residual_combination_policy": (
                self.slat_residual_combination_policy
            ),
            "slat_lora_delta_scale": self.slat_lora_delta_scale,
            "slat_lora_delta_rms_ratio_cap": (
                self.slat_lora_delta_rms_ratio_cap
            ),
            "slat_support_delta_scale": self.slat_support_delta_scale,
            "slat_support_delta_rms_ratio_cap": (
                self.slat_support_delta_rms_ratio_cap
            ),
            "support_interval_policy": self.support_interval_policy,
            "rollout_component": self.rollout_component,
            "cfg_strength": self.cfg_strength,
            "cfg_interval": list(self.cfg_interval),
            "positive_calls": self.positive_calls,
            "negative_calls": self.negative_calls,
            "delta_clip_activated_calls": int(
                sum(
                    row["guided_delta_clip_activated"] > 0.5
                    for row in self.guided_stats
                )
            ),
            "by_timestep": self.guided_stats,
        }
        if not self.guided_stats:
            result["metrics"] = {}
            return result
        metric_names = tuple(
            name
            for name in self.guided_stats[0]
            if name not in {"t", "cfg_active"}
        )
        result["metrics"] = {
            name: {
                "mean": float(
                    sum(row[name] for row in self.guided_stats)
                    / len(self.guided_stats)
                ),
                "min": float(min(row[name] for row in self.guided_stats)),
                "max": float(max(row[name] for row in self.guided_stats)),
            }
            for name in metric_names
        }
        return result


def validate_trainable_whitelist(model: nn.Module) -> dict[str, Any]:
    trainable = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    unexpected = [
        name
        for name in trainable
        if not name.startswith("support_adapter.") and "lora_" not in name
    ]
    if unexpected or not trainable:
        raise RuntimeError(f"direct SLAT trainable whitelist failed: {unexpected}")
    core = _flow_core(getattr(model, "flow", model))
    block_count = len(core.blocks)
    lora_modules = sorted(
        name
        for name, module in getattr(model, "flow", model).named_modules()
        if hasattr(module, "lora_A") or hasattr(module, "lora_B")
    )
    covered = sorted(
        {
            int(part)
            for name in lora_modules
            for position, part in enumerate(name.split("."))
            if part.isdigit()
            and position > 0
            and name.split(".")[position - 1] == "blocks"
        }
    )
    if not lora_modules or covered != list(range(block_count)):
        raise RuntimeError(
            f"SLAT LoRA coverage failed: modules={len(lora_modules)} covered={covered}"
        )
    return {
        "trainable_names": trainable,
        "trainable_parameter_count": int(
            sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        ),
        "lora_module_count": len(lora_modules),
        "lora_target_counts": dict(
            sorted(Counter(name.rsplit(".", 1)[-1] for name in lora_modules).items())
        ),
        "flow_block_count": block_count,
        "covered_flow_blocks": covered,
    }


def strict_trainable_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    validate_trainable_whitelist(model)
    return {
        name: parameter.detach().cpu()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def load_strict_trainable_state(
    model: nn.Module, state: dict[str, torch.Tensor]
) -> None:
    expected = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    missing = sorted(set(expected) - set(state))
    extra = sorted(set(state) - set(expected))
    bad_shape = {
        name: (tuple(state[name].shape), tuple(expected[name].shape))
        for name in sorted(set(state) & set(expected))
        if tuple(state[name].shape) != tuple(expected[name].shape)
    }
    if missing or extra or bad_shape:
        raise RuntimeError(
            "direct SLAT trainable checkpoint mismatch: "
            f"missing={missing[:8]} extra={extra[:8]} bad_shape={bad_shape}"
        )
    with torch.no_grad():
        for name, parameter in expected.items():
            parameter.copy_(state[name].to(device=parameter.device, dtype=parameter.dtype))


def slat_target_cache_identity(
    manifest: str | Path,
    *,
    rows: Iterable[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    path = Path(manifest).resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format") != DIRECT_SLAT_CACHE_VERSION:
        raise ValueError(f"unexpected SLAT cache format={payload.get('format')!r}")
    selected = list(payload.get("samples", []) if rows is None else rows)
    if not selected:
        raise ValueError("SLAT cache identity received no rows")
    uids = [str(row.get("uid", "")) for row in selected]
    objects = [str(row.get("object_uid", "")) for row in selected]
    sources = [str(row.get("source_glb", "")) for row in selected]
    if not all(uids) or not all(objects):
        raise ValueError("SLAT cache contains an empty uid/object_uid")
    return {
        "manifest": str(path),
        "manifest_sha256": _sha256_file(path),
        "format": payload["format"],
        "config_hash": str(payload.get("config_hash", "")),
        "normalization_hash": str(payload.get("slat_normalization_hash", "")),
        "sample_count": len(selected),
        "object_count": len(set(objects)),
        "uid_hash": canonical_json_sha256(uids),
        "object_uid_hash": canonical_json_sha256(sorted(set(objects))),
        "source_glb_hash": canonical_json_sha256(sorted(set(sources))),
    }


def assert_disjoint_object_splits(
    train_identity_rows: Iterable[dict[str, Any]],
    eval_identity_rows: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    train_rows = list(train_identity_rows)
    eval_rows = list(eval_identity_rows)
    train_objects = {str(row.get("object_uid", "")) for row in train_rows}
    eval_objects = {str(row.get("object_uid", "")) for row in eval_rows}
    train_glbs = {str(row.get("source_glb", "")) for row in train_rows if row.get("source_glb")}
    eval_glbs = {str(row.get("source_glb", "")) for row in eval_rows if row.get("source_glb")}
    object_overlap = sorted(train_objects & eval_objects)
    glb_overlap = sorted(train_glbs & eval_glbs)
    if object_overlap or glb_overlap:
        raise RuntimeError(
            "SLAT train/eval leakage: "
            f"object_overlap={object_overlap[:8]} source_glb_overlap={glb_overlap[:8]}"
        )
    return {
        "train_object_count": len(train_objects),
        "eval_object_count": len(eval_objects),
        "object_overlap_count": 0,
        "source_glb_overlap_count": 0,
    }


def build_direct_slat_components(
    *,
    pretrained: str,
    adapter_hidden_dim: int,
    lora_rank: int,
    lora_alpha: int,
    gradient_checkpointing: bool,
    device: torch.device,
    retain_pipeline: bool = False,
) -> tuple[Any, ...]:
    """Load frozen native SLAT components and attach the trainable delta path."""

    from pose_aligned_reconstruction.train_direct_flow import install_unused_model_stubs
    from trellis.pipelines import TrellisVGGTTo3DPipeline

    install_unused_model_stubs()
    pipeline = TrellisVGGTTo3DPipeline.from_pretrained(pretrained)
    pipeline._device = device
    pipeline.low_vram = False
    flow = pipeline.models["slat_flow_model"].to(device).eval()
    for parameter in flow.parameters():
        parameter.requires_grad_(False)
    flow.use_checkpoint = bool(gradient_checkpointing)
    for block in flow.blocks:
        block.use_checkpoint = bool(gradient_checkpointing)

    from peft import LoraConfig, get_peft_model

    flow = get_peft_model(
        flow,
        LoraConfig(
            r=int(lora_rank),
            lora_alpha=int(lora_alpha),
            lora_dropout=0.0,
            bias="none",
            target_modules=["to_q", "to_kv", "to_out", "to_qkv"],
        ),
    )
    flow.train()
    core = _flow_core(flow)
    adapter = SS2SLATSupportAdapter(
        physical_channels=1024,
        hidden_dim=int(adapter_hidden_dim),
        flow_channels=int(core.model_channels),
    ).to(device)
    model = DirectSupportSLATFlowModel(flow, adapter).to(device)
    whitelist = validate_trainable_whitelist(model)
    summary = {
        "stage": "frozen-step900-SS-supported native SLAT Flow",
        "format": DIRECT_SLAT_FLOW_VERSION,
        "support_adapter": adapter.metadata(),
        "stock_fallback": (
            "support missing/scale=0 disables both support injection and all SLAT LoRA"
        ),
        "zero_init_stock_anchor": (
            "when adapter tokens and every LoRA-B are exactly zero, forward values "
            "are straight-through anchored to exact stock while gradients remain live"
        ),
        "slat_delta_policy": {
            "version": SLAT_DELTA_POLICY_VERSION,
            "default": {
                "slat_delta_scale": 1.0,
                "slat_delta_rms_ratio_cap": "disabled",
            },
            "scope": (
                "legacy positive-branch bound plus versioned post-CFG full-guided "
                "vs stock-guided bound; independently per sparse batch"
            ),
            "legacy_default_exact": True,
        },
        "native_slat_condition": "frozen pretrained slat_vggt_cond output",
        "frozen": [
            "step000900 corrected SS Flow and physical encoder",
            "native SLAT Flow base",
            "native slat_vggt_cond",
            "native SLAT mesh decoder",
        ],
        "trainable_whitelist": ["support_adapter.*", "flow.*.lora_[AB].*"],
        "lora": {
            "rank": int(lora_rank),
            "alpha": int(lora_alpha),
            **{
                key: value
                for key, value in whitelist.items()
                if key
                in {
                    "lora_module_count",
                    "lora_target_counts",
                    "flow_block_count",
                    "covered_flow_blocks",
                }
            },
        },
        "trainable_parameter_count": whitelist["trainable_parameter_count"],
    }
    result = (
        pipeline.slat_sampler,
        model,
        dict(pipeline.slat_sampler_params),
        dict(pipeline.slat_normalization),
        summary,
    )
    if retain_pipeline:
        return (*result, pipeline)
    del pipeline
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result
