#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

os.environ.setdefault("SPCONV_ALGO", "native")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
os.environ.setdefault("TORCH_EXTENSIONS_DIR", "/tmp/torch_extensions")

TRACKER_ROOT = Path(__file__).resolve().parents[1]
RECONVIAGEN_ROOT = TRACKER_ROOT / "ReconViaGen"
VGGT_WHEEL_ROOT = RECONVIAGEN_ROOT / "wheels" / "vggt"
for path in (TRACKER_ROOT, RECONVIAGEN_ROOT, VGGT_WHEEL_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from trellis.pipelines import TrellisVGGTTo3DPipeline  # noqa: E402

from reconvggt_ar_adapter_a.eval_b4_delta_prior_alignment import (  # noqa: E402
    _load_prior_manifest_sample,
    _set_compare,
    _xyz_set,
)
from reconvggt_ar_adapter_a.inspect_and_sanity import force_eval, load_images, normalize_image_cond  # noqa: E402
from reconvggt_ar_adapter_a.projection_token_features import parse_ar_pose_file, select_pose_records  # noqa: E402
from reconvggt_ar_adapter_a.run_b3_adapter_injection_smoke import _component_stats, _load_images_with_masks  # noqa: E402
from reconvggt_ar_adapter_a.run_b5_ss_cond_residual_smoke import coords_np, delta_report, summarize_tensor  # noqa: E402
from reconvggt_ar_adapter_a.run_b54_physical_ssgrid_smoke import (  # noqa: E402
    PHYSICAL_FEATURE_NAMES,
    _apply_physical_frame_scope,
    _build_physical_features,
    _stats,
    _token_grid_mapping_audit,
    candidate_quality_row,
    sample_sparse_structure_fixed_noise,
    upsample_ss_tokens_to_sparse,
)
from reconvggt_ar_adapter_a.train_b5_ss_cond_residual_adapter import install_dreamsim_stub, set_frozen_eval  # noqa: E402
from trellis_point_prior_mv.sparse_coord_tools import sparse_diagnostic_metrics  # noqa: E402


FEATURE_NAMES = list(PHYSICAL_FEATURE_NAMES)
FEATURE_INDEX = {name: idx for idx, name in enumerate(FEATURE_NAMES)}
TRAIN_FEATURE_NAMES = [
    "mask_support_ratio",
    "outside_visible_ratio",
    "prior_score",
    "prior_within_radius",
    "any_mask_hit",
    "visual_hull_inside",
    "support_view_fraction",
    "visible_view_fraction",
    "point_distance_over_radius",
    "confident_positive",
    "reliable_outside",
    "surface_contrast",
]
TRAIN_FEATURE_INDEX = [FEATURE_INDEX[name] for name in TRAIN_FEATURE_NAMES]


@dataclass
class PreparedSession:
    name: str
    split: str
    image_names: list[str]
    mask_summaries: Any
    cond_base: torch.Tensor
    neg_cond: torch.Tensor
    physical_features: torch.Tensor
    physical_summary: dict[str, Any]
    loss_masks: dict[str, torch.Tensor]
    prior_sample: dict[str, Any] | None
    prior_coords: np.ndarray | None
    prior_summary: dict[str, Any] | None


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    return str(obj)


def _parse_t_values(spec: str) -> list[float]:
    vals = [float(x.strip()) for x in str(spec).split(",") if x.strip()]
    if not vals:
        raise ValueError("empty --t_values")
    return vals


def _parse_scales(spec: str) -> list[float]:
    vals = [float(x.strip()) for x in str(spec).split(",") if x.strip()]
    if not vals:
        raise ValueError("empty --eval_runtime_scales")
    return vals


def _scale_name(scale: float) -> str:
    prefix = "p" if scale >= 0 else "m"
    body = f"{abs(float(scale)):.6f}".rstrip("0").rstrip(".")
    return f"{prefix}{body.replace('.', 'p')}"


def _session_args(args: argparse.Namespace, spec: dict[str, Any]) -> argparse.Namespace:
    out = copy.copy(args)
    out.image_dir = spec["image_dir"]
    out.pose_file = spec["pose_file"]
    out.mask_dir = spec.get("mask_dir", "")
    out.colmap_sparse_dir = spec.get("colmap_sparse_dir", "")
    out.points3d_txt = spec.get("points3d_txt", "")
    out.point_prior_npz = spec.get("point_prior_npz", "")
    return out


def _upsample_16_to_64(x: torch.Tensor) -> torch.Tensor:
    return upsample_ss_tokens_to_sparse(x, ss_grid_side=16, sparse_resolution=64)


def _safe_weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return (values * weights).sum() / weights.sum().clamp_min(1.0e-6)


def _build_loss_masks(features_np: np.ndarray, device: torch.device, args: argparse.Namespace) -> dict[str, torch.Tensor]:
    features = torch.from_numpy(features_np).to(device=device, dtype=torch.float32)
    if args.loss_mask_mode == "exclusive_surface":
        pos16 = features[:, FEATURE_INDEX["confident_positive"]].clamp(0.0, 1.0)
        neg16 = features[:, FEATURE_INDEX["reliable_outside"]].clamp(0.0, 1.0)
        neg16 = neg16 * (1.0 - pos16)
        active16 = torch.maximum(pos16, neg16)
    elif args.loss_mask_mode == "legacy":
        support_gate = features[:, FEATURE_INDEX["support_gate"]].clamp(0.0, 1.0)
        outside_gate = features[:, FEATURE_INDEX["outside_gate"]].clamp(0.0, 1.0)
        prior_within = features[:, FEATURE_INDEX["prior_within_radius"]].clamp(0.0, 1.0)
        any_mask_hit = features[:, FEATURE_INDEX["any_mask_hit"]].clamp(0.0, 1.0)
        visual_hull = features[:, FEATURE_INDEX["visual_hull_gate"]].clamp(0.0, 1.0)
        prior_score = features[:, FEATURE_INDEX["prior_score"]].clamp(0.0, 1.0)
        pos16 = torch.maximum(support_gate, prior_within * torch.maximum(any_mask_hit, visual_hull))
        if bool(args.use_prior_score_positive):
            pos16 = torch.maximum(pos16, prior_score * visual_hull)
        neg16 = outside_gate * (1.0 - 0.5 * visual_hull)
        active16 = torch.maximum(
            torch.maximum(pos16, neg16),
            visual_hull * float(args.visual_hull_active_weight),
        )
    else:
        raise ValueError(f"Unsupported loss_mask_mode={args.loss_mask_mode!r}")
    neutral16 = (1.0 - active16.clamp(0.0, 1.0)).clamp(0.0, 1.0)
    overlap16 = torch.minimum(pos16, neg16)
    overlap_count = int(((pos16 > 0) & (neg16 > 0)).sum().item())
    pos_count = int((pos16 > 0).sum().item())
    neg_count = int((neg16 > 0).sum().item())
    if args.loss_mask_mode == "exclusive_surface" and overlap_count != 0:
        raise RuntimeError(f"exclusive_surface labels overlap at {overlap_count} SS tokens")
    if bool(args.require_nonempty_surface_labels) and (pos_count == 0 or neg_count == 0):
        raise RuntimeError(
            f"surface labels must be non-empty: positive={pos_count} negative={neg_count}"
        )

    pos64 = _upsample_16_to_64(pos16)
    neg64 = _upsample_16_to_64(neg16)
    neutral64 = _upsample_16_to_64(neutral16)
    active_token = active16.reshape(1, -1, 1)
    model_features = features[:, TRAIN_FEATURE_INDEX].clone()
    distance_feature_idx = TRAIN_FEATURE_NAMES.index("point_distance_over_radius")
    model_features[:, distance_feature_idx] = (
        model_features[:, distance_feature_idx]
        / max(float(args.physical_distance_clip), 1.0e-6)
    ).clamp(0.0, 1.0)
    return {
        "features": features,
        "model_features": model_features,
        "pos16": pos16,
        "neg16": neg16,
        "neutral16": neutral16,
        "active16": active16,
        "pos64": pos64,
        "neg64": neg64,
        "neutral64": neutral64,
        "active_token": active_token,
        "summary": {
            "pos16": _stats(pos16.detach().cpu().numpy()),
            "neg16": _stats(neg16.detach().cpu().numpy()),
            "neutral16": _stats(neutral16.detach().cpu().numpy()),
            "active16": _stats(active16.detach().cpu().numpy()),
            "loss_mask_mode": str(args.loss_mask_mode),
            "pos_nonzero_ratio": float((pos16 > 0).float().mean().item()),
            "neg_nonzero_ratio": float((neg16 > 0).float().mean().item()),
            "neutral_nonzero_ratio": float((neutral16 > 0).float().mean().item()),
            "positive_count": pos_count,
            "negative_count": neg_count,
            "neutral_count": int((neutral16 > 0).sum().item()),
            "positive_negative_overlap_count": overlap_count,
            "positive_negative_overlap_ratio": float(((pos16 > 0) & (neg16 > 0)).float().mean().item()),
            "positive_negative_overlap_mass": float(overlap16.sum().item()),
            "model_feature_names": TRAIN_FEATURE_NAMES,
            "model_feature_stats": {
                name: _stats(model_features[:, idx].detach().cpu().numpy())
                for idx, name in enumerate(TRAIN_FEATURE_NAMES)
            },
        },
    }


class PhysicalSSCondResidualAdapter(nn.Module):
    """Zero-init SS-condition residual adapter driven by physical SS-grid features."""

    def __init__(self, channels: int = 1024, feature_dim: int = len(TRAIN_FEATURE_NAMES), hidden_dim: int = 256) -> None:
        super().__init__()
        self.channels = int(channels)
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.cond_norm = nn.LayerNorm(self.channels)
        self.phys_norm = nn.LayerNorm(self.feature_dim)
        self.cond_proj = nn.Linear(self.channels, self.hidden_dim)
        self.phys_proj = nn.Linear(self.feature_dim, self.hidden_dim)
        self.fuse = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.channels),
        )
        last = self.fuse[-1]
        assert isinstance(last, nn.Linear)
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def forward(self, cond_base: torch.Tensor, physical_features: torch.Tensor, active_gate: torch.Tensor) -> torch.Tensor:
        if cond_base.ndim != 3:
            raise ValueError(f"cond_base must be [B,T,C], got {tuple(cond_base.shape)}")
        if physical_features.ndim != 2:
            raise ValueError(f"physical_features must be [T,F], got {tuple(physical_features.shape)}")
        b, t, _ = cond_base.shape
        if physical_features.shape[0] != t:
            raise ValueError(f"Token mismatch: cond={tuple(cond_base.shape)}, features={tuple(physical_features.shape)}")
        cond_in = torch.nan_to_num(cond_base.float(), nan=0.0, posinf=0.0, neginf=0.0).clamp(-100.0, 100.0)
        phys = torch.nan_to_num(physical_features.float(), nan=0.0, posinf=0.0, neginf=0.0)
        phys = phys.clamp(-100.0, 100.0).unsqueeze(0).expand(b, -1, -1)
        cond_h = torch.nan_to_num(self.cond_proj(self.cond_norm(cond_in)), nan=0.0, posinf=0.0, neginf=0.0)
        phys_h = torch.nan_to_num(self.phys_proj(self.phys_norm(phys)), nan=0.0, posinf=0.0, neginf=0.0)
        hidden = torch.cat((cond_h, phys_h), dim=-1)
        for layer in self.fuse:
            hidden = layer(hidden)
            hidden = torch.nan_to_num(hidden, nan=0.0, posinf=0.0, neginf=0.0).clamp(-100.0, 100.0)
        delta = hidden
        return (delta * active_gate.float()).to(dtype=torch.float32)

    def metadata(self) -> dict[str, Any]:
        return {
            "type": self.__class__.__name__,
            "channels": self.channels,
            "feature_dim": self.feature_dim,
            "hidden_dim": self.hidden_dim,
            "zero_init_last_layer": True,
            "input": "cond_base + physical_ssgrid_features + active_gate",
        }


class LowRankPhysicalSSCondResidualAdapter(nn.Module):
    """Zero-output low-rank SS-condition residual adapter.

    It predicts a few scalar gates from physical SS-grid features and multiplies
    them by learned channel bases. This avoids directly regressing a dense
    1024-channel residual from a large MLP.
    """

    def __init__(
        self,
        channels: int = 1024,
        feature_dim: int = len(TRAIN_FEATURE_NAMES),
        hidden_dim: int = 128,
        rank: int = 4,
        basis_init_std: float = 0.02,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.rank = int(rank)
        if self.rank <= 0:
            raise ValueError(f"rank must be positive, got {rank}")
        self.gate_mlp = nn.Sequential(
            nn.Linear(self.feature_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.rank),
        )
        last = self.gate_mlp[-1]
        assert isinstance(last, nn.Linear)
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
        self.basis = nn.Parameter(torch.randn(self.rank, self.channels, dtype=torch.float32) * float(basis_init_std))

    def gate_scores(self, physical_features: torch.Tensor, batch: int) -> torch.Tensor:
        if physical_features.ndim != 2:
            raise ValueError(f"physical_features must be [T,F], got {tuple(physical_features.shape)}")
        phys = torch.nan_to_num(physical_features.float(), nan=0.0, posinf=0.0, neginf=0.0).clamp(-100.0, 100.0)
        hidden = phys.unsqueeze(0).expand(int(batch), -1, -1)
        gates = self.gate_mlp(hidden)
        return torch.nan_to_num(gates, nan=0.0, posinf=0.0, neginf=0.0).clamp(-10.0, 10.0)

    def forward(self, cond_base: torch.Tensor, physical_features: torch.Tensor, active_gate: torch.Tensor) -> torch.Tensor:
        if cond_base.ndim != 3:
            raise ValueError(f"cond_base must be [B,T,C], got {tuple(cond_base.shape)}")
        b, t, c = cond_base.shape
        if c != self.channels:
            raise ValueError(f"Channel mismatch: cond={c}, adapter={self.channels}")
        if physical_features.shape[0] != t:
            raise ValueError(f"Token mismatch: cond={tuple(cond_base.shape)}, features={tuple(physical_features.shape)}")
        gates = self.gate_scores(physical_features, batch=b)
        basis = torch.nan_to_num(self.basis.float(), nan=0.0, posinf=0.0, neginf=0.0)
        basis = F.normalize(basis, dim=-1)
        delta = torch.einsum("btr,rc->btc", gates, basis)
        return (delta * active_gate.float()).to(dtype=torch.float32)

    def metadata(self) -> dict[str, Any]:
        return {
            "type": self.__class__.__name__,
            "channels": self.channels,
            "feature_dim": self.feature_dim,
            "hidden_dim": self.hidden_dim,
            "rank": self.rank,
            "zero_init_gate": True,
            "model_feature_names": TRAIN_FEATURE_NAMES,
            "feature_normalization": "bounded ratios/binaries; point_distance_over_radius divided by physical_distance_clip; no LayerNorm",
            "input": "physical_ssgrid_features -> scalar gates -> learned low-rank channel basis",
        }


def _prepare_session(
    *,
    pipeline: TrellisVGGTTo3DPipeline,
    args: argparse.Namespace,
    spec: dict[str, Any],
    device: torch.device,
) -> PreparedSession:
    sargs = _session_args(args, spec)
    name = str(spec["name"])
    split = str(spec.get("split", "train"))
    print(f"[B5.5] preparing session name={name} split={split}", flush=True)

    if args.mask_mode == "apply":
        if not sargs.mask_dir:
            raise ValueError(f"Session {name} requires mask_dir when --mask_mode=apply")
        images, image_names, mask_summaries = _load_images_with_masks(
            Path(sargs.image_dir),
            mask_dir=Path(sargs.mask_dir),
            max_views=int(args.max_views),
            mask_background=args.mask_background,
        )
    else:
        images, image_names = load_images(
            Path(sargs.image_dir),
            max_views=int(args.max_views),
            preprocess=args.preprocess,
            pipeline=pipeline,
        )
        mask_summaries = None

    pose_records_all = parse_ar_pose_file(
        sargs.pose_file,
        default_intrinsics=(args.default_fx, args.default_fy, args.default_cx, args.default_cy),
        default_image_size=(args.default_image_width, args.default_image_height),
    )
    _ = select_pose_records(image_names, pose_records_all)

    with torch.no_grad(), torch.cuda.amp.autocast(enabled=(device.type == "cuda"), dtype=getattr(pipeline, "VGGT_dtype", torch.float16)):
        aggregated_tokens_list, _ = pipeline.vggt_feat(images)
        raw_image_cond = pipeline.encode_image(images)
    b, n, _, _ = aggregated_tokens_list[0].shape
    image_cond = normalize_image_cond(raw_image_cond, batch=b, views=n)
    with torch.no_grad():
        ss_cond_base = pipeline.get_ss_cond(image_cond[:, :, int(args.patch_start_idx) :], aggregated_tokens_list, int(args.num_samples))
    cond_base = ss_cond_base["cond"].detach()
    neg_cond = ss_cond_base["neg_cond"].detach()

    prior_manifest = str(spec.get("prior_manifest", "") or "")
    if not prior_manifest:
        raise ValueError(f"Session {name} requires prior_manifest for B5.5 physical proxy")
    prior_sample, prior_coords, prior_summary = _load_prior_manifest_sample(
        Path(prior_manifest),
        str(spec.get("prior_uid", "") or ""),
    )
    feature_sample, feature_scope_summary = _apply_physical_frame_scope(
        prior_sample,
        scope=str(args.physical_frame_scope),
        selected_image_names=image_names,
    )
    eval_sample, eval_scope_summary = _apply_physical_frame_scope(
        prior_sample,
        scope=str(args.evaluation_frame_scope),
        selected_image_names=image_names,
    )
    physical_np, physical_summary = _build_physical_features(
        sample=feature_sample,
        prior_coords=prior_coords,
        args=args,
    )
    physical_summary["feature_frame_scope"] = feature_scope_summary
    physical_summary["evaluation_frame_scope"] = eval_scope_summary
    loss_masks = _build_loss_masks(physical_np, cond_base.device, args)

    del aggregated_tokens_list, raw_image_cond, image_cond
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return PreparedSession(
        name=name,
        split=split,
        image_names=image_names,
        mask_summaries=mask_summaries,
        cond_base=cond_base,
        neg_cond=neg_cond,
        physical_features=loss_masks["model_features"],
        physical_summary={**physical_summary, "loss_masks": loss_masks["summary"]},
        loss_masks=loss_masks,
        prior_sample=eval_sample,
        prior_coords=prior_coords,
        prior_summary={
            **prior_summary,
            "feature_frame_scope": feature_scope_summary,
            "evaluation_frame_scope": eval_scope_summary,
        },
    )


def _one_step_logits(
    *,
    flow,
    decoder,
    sampler,
    x_t: torch.Tensor,
    t_model: float,
    cond: torch.Tensor,
    neg_cond: torch.Tensor,
    cfg_mode: str,
    cfg_strength: float,
    cfg_interval: tuple[float, float],
    guidance_rescale: float,
    autocast_enabled: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if cfg_mode == "unguided":
        effective_cfg_strength = 1.0
        effective_guidance_rescale = 0.0
    elif cfg_mode == "sampler":
        effective_cfg_strength = float(cfg_strength)
        effective_guidance_rescale = float(guidance_rescale)
    else:
        raise ValueError(f"Unsupported proxy_cfg_mode={cfg_mode!r}")
    with torch.cuda.amp.autocast(enabled=autocast_enabled):
        pred_v = sampler._inference_model(
            flow,
            x_t,
            float(t_model),
            cond,
            neg_cond=neg_cond,
            cfg_strength=effective_cfg_strength,
            cfg_interval=tuple(float(v) for v in cfg_interval),
            guidance_rescale=effective_guidance_rescale,
        )
        x0 = sampler._pred_to_xstart(x_t, float(t_model), pred_v)
        logits = decoder(x0)
    return logits.float(), x0.float()


def _rescale_proxy_time(t_raw: float, rescale_t: float) -> float:
    t_raw = float(t_raw)
    rescale_t = float(rescale_t)
    return float(rescale_t * t_raw / (1.0 + (rescale_t - 1.0) * t_raw))


@torch.no_grad()
def _stock_trajectory_cache(
    *,
    flow,
    sampler,
    session: PreparedSession,
    initial_noise: torch.Tensor,
    requested_t_values: list[float],
    args: argparse.Namespace,
) -> dict[float, dict[str, Any]]:
    steps = int(args.proxy_trajectory_steps)
    if steps <= 0:
        raise ValueError(f"proxy_trajectory_steps must be positive, got {steps}")
    raw_t_seq = np.linspace(1.0, 0.0, steps + 1, dtype=np.float64)
    model_t_seq = np.asarray(
        [_rescale_proxy_time(float(t), float(args.proxy_rescale_t)) for t in raw_t_seq],
        dtype=np.float64,
    )
    target_indices = {
        float(requested): int(np.argmin(np.abs(raw_t_seq - float(requested))))
        for requested in requested_t_values
    }
    needed_indices = set(target_indices.values())
    state = initial_noise.detach().clone()
    states: dict[int, torch.Tensor] = {}
    if 0 in needed_indices:
        states[0] = state.detach().clone()
    cfg_interval = (float(args.ss_cfg_interval_min), float(args.ss_cfg_interval_max))
    for idx in range(steps):
        t_model = float(model_t_seq[idx])
        t_prev_model = float(model_t_seq[idx + 1])
        with torch.cuda.amp.autocast(enabled=(state.device.type == "cuda")):
            pred_v = sampler._inference_model(
                flow,
                state,
                t_model,
                session.cond_base.float(),
                neg_cond=session.neg_cond,
                cfg_strength=float(args.ss_cfg_strength),
                cfg_interval=cfg_interval,
                guidance_rescale=float(args.ss_guidance_rescale),
            )
        state = state + (t_prev_model - t_model) * pred_v
        state_idx = idx + 1
        if state_idx in needed_indices:
            states[state_idx] = state.detach().clone()
    cache: dict[float, dict[str, Any]] = {}
    for requested, state_idx in target_indices.items():
        cache[requested] = {
            "x_t": states[state_idx],
            "requested_t_raw": requested,
            "snapped_t_raw": float(raw_t_seq[state_idx]),
            "t_model": float(model_t_seq[state_idx]),
            "trajectory_state_index": int(state_idx),
        }
    return cache


def _smoothness_loss(delta: torch.Tensor) -> torch.Tensor:
    # Use squared energy rather than sqrt energy. The residual adapters are
    # zero-init; sqrt-style norms have undefined/unstable gradients at zero.
    energy = (delta.float() * delta.float()).mean(dim=-1).reshape(delta.shape[0], 16, 16, 16)
    loss = energy.new_tensor(0.0)
    loss = loss + (energy[:, 1:, :, :] - energy[:, :-1, :, :]).pow(2).mean()
    loss = loss + (energy[:, :, 1:, :] - energy[:, :, :-1, :]).pow(2).mean()
    loss = loss + (energy[:, :, :, 1:] - energy[:, :, :, :-1]).pow(2).mean()
    return loss / 3.0


def _delta_norm_stats(delta: torch.Tensor, cond_base: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    cond_power = (cond_base.float() * cond_base.float()).mean().clamp_min(1.0e-12)
    delta_power = (delta.float() * delta.float()).mean()
    delta_norm_loss = delta_power / cond_power
    delta_norm_ratio = torch.sqrt(delta_norm_loss.detach().clamp_min(0.0))
    return delta_norm_loss, delta_norm_ratio


def _train_step_loss(
    *,
    adapter: PhysicalSSCondResidualAdapter,
    session: PreparedSession,
    base_logits: torch.Tensor,
    flow,
    decoder,
    sampler,
    x_t: torch.Tensor,
    t: float,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, Any]]:
    delta = adapter(session.cond_base, session.physical_features, session.loss_masks["active_token"])
    if float(args.delta_clip_abs) > 0:
        delta = delta.clamp(min=-float(args.delta_clip_abs), max=float(args.delta_clip_abs))
    cond_adapt = session.cond_base.float() + float(args.train_runtime_scale) * delta
    logits_adapt, _ = _one_step_logits(
        flow=flow,
        decoder=decoder,
        sampler=sampler,
        x_t=x_t,
        t_model=float(t),
        cond=cond_adapt,
        neg_cond=session.neg_cond,
        cfg_mode=str(args.proxy_cfg_mode),
        cfg_strength=float(args.ss_cfg_strength),
        cfg_interval=(float(args.ss_cfg_interval_min), float(args.ss_cfg_interval_max)),
        guidance_rescale=float(args.ss_guidance_rescale),
        autocast_enabled=(session.cond_base.device.type == "cuda"),
    )
    dlogits = logits_adapt - base_logits.detach()
    pos_w = session.loss_masks["pos64"]
    neg_w = session.loss_masks["neg64"]
    neutral_w = session.loss_masks["neutral64"]
    pos_loss = _safe_weighted_mean(F.relu(float(args.margin_pos) - dlogits).pow(2), pos_w)
    neg_loss = _safe_weighted_mean(F.relu(dlogits + float(args.margin_neg)).pow(2), neg_w)
    preserve_loss = _safe_weighted_mean(dlogits.pow(2), neutral_w)
    delta_norm_loss, delta_norm = _delta_norm_stats(delta, session.cond_base)
    smooth_loss = _smoothness_loss(delta)
    loss = (
        float(args.pos_weight) * pos_loss
        + float(args.neg_weight) * neg_loss
        + float(args.preserve_weight) * preserve_loss
        + float(args.delta_norm_weight) * delta_norm_loss
        + float(args.smooth_weight) * smooth_loss
    )
    stats = {
        "loss": float(loss.detach().cpu().item()),
        "pos_loss": float(pos_loss.detach().cpu().item()),
        "neg_loss": float(neg_loss.detach().cpu().item()),
        "preserve_loss": float(preserve_loss.detach().cpu().item()),
        "delta_norm_loss": float(delta_norm_loss.detach().cpu().item()),
        "smooth_loss": float(smooth_loss.detach().cpu().item()),
        "delta_norm_ratio": float(delta_norm.detach().cpu().item()),
        "dlogits_pos_mean": float(_safe_weighted_mean(dlogits.detach(), pos_w).cpu().item()),
        "dlogits_neg_mean": float(_safe_weighted_mean(dlogits.detach(), neg_w).cpu().item()),
        "dlogits_neutral_abs_mean": float(_safe_weighted_mean(dlogits.detach().abs(), neutral_w).cpu().item()),
    }
    return loss, stats


def _train_cond_basis_loss(
    *,
    adapter: PhysicalSSCondResidualAdapter,
    session: PreparedSession,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, Any]]:
    delta = adapter(session.cond_base, session.physical_features, session.loss_masks["active_token"])
    if float(args.delta_clip_abs) > 0:
        delta = delta.clamp(min=-float(args.delta_clip_abs), max=float(args.delta_clip_abs))

    channels = int(delta.shape[-1])
    basis_src = torch.nan_to_num(session.cond_base.float().detach(), nan=0.0, posinf=0.0, neginf=0.0)
    basis = F.normalize(basis_src, dim=-1)
    score = (delta.float() * basis).sum(dim=-1) / math.sqrt(float(channels))
    pos_w = session.loss_masks["pos16"].reshape(1, -1)
    neg_w = session.loss_masks["neg16"].reshape(1, -1)
    neutral_w = session.loss_masks["neutral16"].reshape(1, -1)

    pos_loss = _safe_weighted_mean((score - float(args.cond_basis_pos_target)).pow(2), pos_w)
    neg_loss = _safe_weighted_mean((score + float(args.cond_basis_neg_target)).pow(2), neg_w)
    preserve_loss = _safe_weighted_mean(score.pow(2), neutral_w)
    delta_norm_loss, delta_norm = _delta_norm_stats(delta, session.cond_base)
    smooth_loss = _smoothness_loss(delta)
    loss = (
        float(args.pos_weight) * pos_loss
        + float(args.neg_weight) * neg_loss
        + float(args.preserve_weight) * preserve_loss
        + float(args.delta_norm_weight) * delta_norm_loss
        + float(args.smooth_weight) * smooth_loss
    )
    stats = {
        "loss": float(loss.detach().cpu().item()),
        "pos_loss": float(pos_loss.detach().cpu().item()),
        "neg_loss": float(neg_loss.detach().cpu().item()),
        "preserve_loss": float(preserve_loss.detach().cpu().item()),
        "delta_norm_loss": float(delta_norm_loss.detach().cpu().item()),
        "smooth_loss": float(smooth_loss.detach().cpu().item()),
        "delta_norm_ratio": float(delta_norm.detach().cpu().item()),
        "dlogits_pos_mean": float(_safe_weighted_mean(score.detach(), pos_w).cpu().item()),
        "dlogits_neg_mean": float(_safe_weighted_mean(score.detach(), neg_w).cpu().item()),
        "dlogits_neutral_abs_mean": float(_safe_weighted_mean(score.detach().abs(), neutral_w).cpu().item()),
    }
    return loss, stats


def _train_lowrank_gate_loss(
    *,
    adapter: LowRankPhysicalSSCondResidualAdapter,
    session: PreparedSession,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, Any]]:
    gates = adapter.gate_scores(session.physical_features, batch=int(session.cond_base.shape[0]))
    delta = adapter(session.cond_base, session.physical_features, session.loss_masks["active_token"])
    if float(args.delta_clip_abs) > 0:
        delta = delta.clamp(min=-float(args.delta_clip_abs), max=float(args.delta_clip_abs))

    gate_score = gates.mean(dim=-1)
    pos_w = session.loss_masks["pos16"].reshape(1, -1)
    neg_w = session.loss_masks["neg16"].reshape(1, -1)
    neutral_w = session.loss_masks["neutral16"].reshape(1, -1)

    pos_loss = _safe_weighted_mean((gate_score - float(args.gate_pos_target)).pow(2), pos_w)
    neg_loss = _safe_weighted_mean((gate_score + float(args.gate_neg_target)).pow(2), neg_w)
    preserve_loss = _safe_weighted_mean(gate_score.pow(2), neutral_w)
    gate_l2_loss = (gates.float() * gates.float()).mean()
    delta_norm_loss, delta_norm = _delta_norm_stats(delta, session.cond_base)
    smooth_loss = _smoothness_loss(delta)
    loss = (
        float(args.pos_weight) * pos_loss
        + float(args.neg_weight) * neg_loss
        + float(args.preserve_weight) * preserve_loss
        + float(args.gate_l2_weight) * gate_l2_loss
        + float(args.delta_norm_weight) * delta_norm_loss
        + float(args.smooth_weight) * smooth_loss
    )
    stats = {
        "loss": float(loss.detach().cpu().item()),
        "pos_loss": float(pos_loss.detach().cpu().item()),
        "neg_loss": float(neg_loss.detach().cpu().item()),
        "preserve_loss": float(preserve_loss.detach().cpu().item()),
        "gate_l2_loss": float(gate_l2_loss.detach().cpu().item()),
        "delta_norm_loss": float(delta_norm_loss.detach().cpu().item()),
        "smooth_loss": float(smooth_loss.detach().cpu().item()),
        "delta_norm_ratio": float(delta_norm.detach().cpu().item()),
        "dlogits_pos_mean": float(_safe_weighted_mean(gate_score.detach(), pos_w).cpu().item()),
        "dlogits_neg_mean": float(_safe_weighted_mean(gate_score.detach(), neg_w).cpu().item()),
        "dlogits_neutral_abs_mean": float(_safe_weighted_mean(gate_score.detach().abs(), neutral_w).cpu().item()),
        "gate_abs_mean": float(gates.detach().abs().mean().cpu().item()),
    }
    return loss, stats


def _sample_sparse(
    *,
    pipeline: TrellisVGGTTo3DPipeline,
    flow,
    args: argparse.Namespace,
    session: PreparedSession,
    cond: torch.Tensor,
    noise: torch.Tensor,
    out_dir: Path,
) -> tuple[np.ndarray, dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    params = {
        "steps": int(args.ss_steps),
        "cfg_strength": float(args.ss_cfg_strength),
        "cfg_interval": [0.6, 1.0],
        "guidance_rescale": float(args.ss_guidance_rescale),
        "rescale_t": float(args.ss_rescale_t),
    }
    ss_cond = {"cond": cond.to(dtype=session.cond_base.dtype), "neg_cond": session.neg_cond}
    coords, _ = sample_sparse_structure_fixed_noise(
        pipeline=pipeline,
        cond=ss_cond,
        noise=noise,
        sampler_params=params,
    )
    coords_array = coords_np(coords)
    np.savez_compressed(out_dir / "coords.npz", coords=coords_array)
    prior_alignment = sparse_diagnostic_metrics(
        "b55_sparse",
        coords_array,
        session.prior_coords,
        session.prior_sample,
        prior_radius=float(args.prior_radius),
        min_support_views=int(args.projection_min_support_views),
        min_support_ratio=float(args.projection_min_support_ratio),
        visual_hull_min_visible_views=int(args.visual_hull_min_visible_views),
        visual_hull_min_support_ratio=float(args.visual_hull_min_support_ratio),
        grid_resolution=int(args.sparse_resolution),
        mask_threshold=int(args.mask_threshold),
    )
    return coords_array, {
        "output_dir": str(out_dir),
        "sparse": _component_stats(coords_array),
        "prior_alignment": prior_alignment,
        "sampling_backend": "native_pipeline_supplied_noise",
    }


def _evaluate(
    *,
    pipeline: TrellisVGGTTo3DPipeline,
    flow,
    adapter: PhysicalSSCondResidualAdapter,
    args: argparse.Namespace,
    sessions: list[PreparedSession],
    output_dir: Path,
) -> list[dict[str, Any]]:
    adapter.eval()
    reports: list[dict[str, Any]] = []
    scales = _parse_scales(args.eval_runtime_scales)
    for idx, session in enumerate(sessions):
        torch.manual_seed(int(args.seed) + 1000 + idx)
        noise = torch.randn(
            int(args.num_samples),
            int(flow.in_channels),
            int(flow.resolution),
            int(flow.resolution),
            int(flow.resolution),
            device=session.cond_base.device,
        )
        with torch.no_grad():
            delta = adapter(session.cond_base, session.physical_features, session.loss_masks["active_token"])
        coords_by_name: dict[str, np.ndarray] = {}
        evals: dict[str, Any] = {}
        base_coords, base_report = _sample_sparse(
            pipeline=pipeline,
            flow=flow,
            args=args,
            session=session,
            cond=session.cond_base,
            noise=noise,
            out_dir=output_dir / "eval" / session.name / "baseline",
        )
        fixed_noise_repeat = None
        if bool(args.verify_fixed_noise_repeat):
            repeat_coords, _ = _sample_sparse(
                pipeline=pipeline,
                flow=flow,
                args=args,
                session=session,
                cond=session.cond_base,
                noise=noise,
                out_dir=output_dir / "eval" / session.name / "baseline_repeat",
            )
            fixed_noise_repeat = {
                "passed": bool(np.array_equal(base_coords, repeat_coords)),
                "baseline_coord_count": int(base_coords.shape[0]),
                "repeat_coord_count": int(repeat_coords.shape[0]),
            }
            if not fixed_noise_repeat["passed"]:
                raise RuntimeError(
                    f"fixed-noise baseline repeat failed for session={session.name}: {fixed_noise_repeat}"
                )
        coords_by_name["baseline"] = base_coords
        evals["baseline"] = base_report
        for scale in scales:
            name = f"adapter_a{_scale_name(scale)}"
            cand_coords, cand_report = _sample_sparse(
                pipeline=pipeline,
                flow=flow,
                args=args,
                session=session,
                cond=session.cond_base.float() + float(scale) * delta.float(),
                noise=noise,
                out_dir=output_dir / "eval" / session.name / name,
            )
            coords_by_name[name] = cand_coords
            cand_report["delta_vs_baseline"] = delta_report(
                baseline_coords=base_coords,
                candidate_coords=cand_coords,
                prior_sample=session.prior_sample,
                prior_coords=session.prior_coords,
                prior_radius=float(args.prior_radius),
                projection_min_support_views=int(args.projection_min_support_views),
                projection_min_support_ratio=float(args.projection_min_support_ratio),
                visual_hull_min_visible_views=int(args.visual_hull_min_visible_views),
                visual_hull_min_support_ratio=float(args.visual_hull_min_support_ratio),
                mask_threshold=int(args.mask_threshold),
            )
            cand_report["set_compare_vs_baseline"] = _set_compare(_xyz_set(base_coords), _xyz_set(cand_coords))
            evals[name] = cand_report
        reports.append(
            {
                "name": session.name,
                "split": session.split,
                "image_names": session.image_names,
                "mask_summaries": session.mask_summaries,
                "physical_features": session.physical_summary,
                "prior_summary": session.prior_summary,
                "cond_base": summarize_tensor(session.cond_base),
                "adapter_delta": summarize_tensor(delta),
                "fixed_noise_repeat": fixed_noise_repeat,
                "eval": evals,
            }
        )
    return reports


def _aggregate_eval(session_reports: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    names = sorted({name for sess in session_reports for name in sess["eval"] if name != "baseline"})
    rows: list[dict[str, Any]] = []
    for name in names:
        vals = []
        for sess in session_reports:
            row = sess["eval"].get(name) or {}
            vals.append(
                candidate_quality_row(
                    session_name=sess["name"],
                    split=sess["split"],
                    baseline=sess["eval"]["baseline"],
                    candidate=row,
                    args=args,
                )
            )
        mask_vals = [float(v["mask"]) for v in vals if v["mask"] is not None]
        outside_vals = [float(v["outside"]) for v in vals if v["outside"] is not None]
        prior_vals = [float(v["prior"]) for v in vals if v["prior"] is not None]
        iou_vals = [float(v["iou"]) for v in vals if v["iou"] is not None]
        absolute_outside_deltas = [
            float(v["absolute_outside_delta"])
            for v in vals
            if v["absolute_outside_delta"] is not None
        ]
        strict_count = int(sum(bool(v["passed"]) for v in vals))
        rows.append(
            {
                "candidate": name,
                "session_count": len(vals),
                "strict_direction_session_count": strict_count,
                "mean_mask": float(np.mean(mask_vals)) if mask_vals else None,
                "min_mask": float(np.min(mask_vals)) if mask_vals else None,
                "mean_outside": float(np.mean(outside_vals)) if outside_vals else None,
                "max_outside": float(np.max(outside_vals)) if outside_vals else None,
                "mean_prior": float(np.mean(prior_vals)) if prior_vals else None,
                "min_prior": float(np.min(prior_vals)) if prior_vals else None,
                "mean_iou": float(np.mean(iou_vals)) if iou_vals else None,
                "mean_changed_ratio": float(np.mean([v["changed_ratio"] for v in vals])) if vals else None,
                "max_changed_ratio": float(np.max([v["changed_ratio"] for v in vals])) if vals else None,
                "mean_absolute_outside_delta": float(np.mean(absolute_outside_deltas)) if absolute_outside_deltas else None,
                "max_absolute_outside_delta": float(np.max(absolute_outside_deltas)) if absolute_outside_deltas else None,
                "max_component_count_delta": int(np.max([v["component_count_delta"] for v in vals])) if vals else None,
                "min_coord_count_ratio": float(np.min([v["coord_count_ratio"] for v in vals if v["coord_count_ratio"] is not None]))
                if any(v["coord_count_ratio"] is not None for v in vals) else None,
                "max_coord_count_ratio": float(np.max([v["coord_count_ratio"] for v in vals if v["coord_count_ratio"] is not None]))
                if any(v["coord_count_ratio"] is not None for v in vals) else None,
                "all_sessions_passed": bool(vals) and all(bool(v["passed"]) for v in vals),
                "per_session": vals,
            }
        )
    rows.sort(
        key=lambda r: (
            int(r["strict_direction_session_count"]),
            float(r["mean_mask"] or -1.0),
            -float(r["max_outside"] or 1.0),
            float(r["mean_prior"] or -1.0),
        ),
        reverse=True,
    )
    return rows


def _write_md(path: Path, report: dict[str, Any], command: str) -> None:
    proxy_mode = str(report["args"]["proxy_loss_mode"])
    if proxy_mode == "decoder_logits":
        proxy_scope = "loss traverses one frozen sparse-flow step and the frozen sparse decoder logits"
    elif proxy_mode == "cond_basis":
        proxy_scope = "loss acts on a deterministic condition-channel projection; flow/decoder are eval-only"
    else:
        proxy_scope = "loss acts directly on low-rank scalar gates; flow/decoder are eval-only"
    lines = [
        "# B5.5 Physical Proxy SS-Condition Residual Adapter",
        "",
        "## Command",
        "",
        "```bash",
        command.strip(),
        "```",
        "",
        "## Scope",
        "",
        "```text",
        "freeze ReconViaGen VGGT / sparse_structure_vggt_cond / sparse flow / sparse decoder",
        "train only B5.5 SS-condition residual adapter",
        f"proxy_loss_mode={proxy_mode}: {proxy_scope}",
        f"proxy_state_mode={report['args']['proxy_state_mode']}",
        f"proxy_cfg_mode={report['args']['proxy_cfg_mode']}",
        f"proxy_rescale_t={report['args']['proxy_rescale_t']}",
        "decoded sparse coords are used only for eval diagnostics",
        "SLAT is not touched",
        "```",
        "",
        "## Train",
        "",
        "```text",
    ]
    for row in report["rows"]:
        if row.get("skipped_nonfinite") or row.get("skipped_nonfinite_grad"):
            lines.append(
                f"step={row['step']} t={row.get('t')} skipped="
                f"{'loss' if row.get('skipped_nonfinite') else 'grad'}"
            )
            continue
        lines.append(
            f"step={row['step']} loss={row['loss']:.6g} pos={row['pos_loss']:.6g} "
            f"neg={row['neg_loss']:.6g} preserve={row['preserve_loss']:.6g} "
            f"delta_norm={row['delta_norm_ratio']:.6g} "
            f"dpos={row['dlogits_pos_mean']:.6g} dneg={row['dlogits_neg_mean']:.6g}"
        )
    lines.extend(["```", "", "## Aggregate Eval", "", "```text"])
    for row in report["aggregate"]:
        lines.append(
            f"{row['candidate']}: strict={row['strict_direction_session_count']}/{row['session_count']} "
            f"mean_mask={row['mean_mask']} min_mask={row['min_mask']} "
            f"mean_outside={row['mean_outside']} max_outside={row['max_outside']} "
            f"mean_prior={row['mean_prior']} min_prior={row['min_prior']} "
            f"mean_iou={row['mean_iou']} max_changed={row['max_changed_ratio']} "
            f"max_abs_outside_delta={row['max_absolute_outside_delta']} "
            f"max_component_delta={row['max_component_count_delta']} "
            f"coord_ratio=[{row['min_coord_count_ratio']},{row['max_coord_count_ratio']}]"
        )
    lines.extend(["```", "", "## Per Session", "", "```text"])
    for sess in report["sessions"]:
        lines.append(f"{sess['name']} split={sess['split']}")
        lines.append(f"  physical_sanity={sess['physical_features']['sanity']}")
        for name, eval_row in sess["eval"].items():
            if name == "baseline":
                lines.append(f"  baseline: sparse={eval_row.get('sparse')}")
                continue
            direction = ((eval_row.get("delta_vs_baseline") or {}).get("direction_summary") or {})
            cmp_row = eval_row.get("set_compare_vs_baseline") or {}
            lines.append(
                f"  {name}: iou={cmp_row.get('iou')} comp={(eval_row.get('sparse') or {}).get('component_count')} "
                f"mask={direction.get('added_minus_removed_projection_any_mask_hit_ratio')} "
                f"outside={direction.get('added_minus_removed_visible_outside_mask_event_ratio')} "
                f"prior={direction.get('added_minus_removed_within_prior_radius_ratio')} "
                f"absolute_outside={(eval_row.get('prior_alignment') or {}).get('b55_sparse_visible_outside_mask_event_ratio')}"
            )
        lines.append("")
    lines.extend(["```", "", "## Judgment", "", report["judgment"]])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="B5.5 physical proxy SS condition residual adapter.")
    parser.add_argument("--sessions_json", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--pretrained", default="Stable-X/trellis-vggt-v0-2")
    parser.add_argument("--max_views", type=int, default=7)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--low_vram", action="store_true")
    parser.add_argument("--preprocess", action="store_true")
    parser.add_argument("--mask_mode", choices=["none", "apply"], default="apply")
    parser.add_argument("--mask_background", choices=["black", "white"], default="black")
    parser.add_argument("--mask_threshold", type=int, default=127)
    parser.add_argument("--patch_start_idx", type=int, default=5)
    parser.add_argument("--default_fx", type=float, default=485.845947)
    parser.add_argument("--default_fy", type=float, default=485.744232)
    parser.add_argument("--default_cx", type=float, default=322.973236)
    parser.add_argument("--default_cy", type=float, default=237.599487)
    parser.add_argument("--default_image_width", type=int, default=640)
    parser.add_argument("--default_image_height", type=int, default=480)
    parser.add_argument("--load_dreamsim", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--ss_steps", type=int, default=12)
    parser.add_argument("--ss_cfg_strength", type=float, default=7.5)
    parser.add_argument("--ss_cfg_interval_min", type=float, default=0.6)
    parser.add_argument("--ss_cfg_interval_max", type=float, default=1.0)
    parser.add_argument("--ss_guidance_rescale", type=float, default=0.5)
    parser.add_argument("--ss_rescale_t", type=float, default=3.0)
    parser.add_argument("--ss_grid_side", type=int, default=16)
    parser.add_argument("--sparse_resolution", type=int, default=64)
    parser.add_argument("--prior_radius", type=float, default=4.0)
    parser.add_argument("--physical_frame_scope", choices=["selected", "fullscan"], default="selected")
    parser.add_argument("--evaluation_frame_scope", choices=["selected", "fullscan"], default="fullscan")
    parser.add_argument("--physical_distance_clip", type=float, default=8.0)
    parser.add_argument("--physical_vh_min_visible_views", type=int, default=1)
    parser.add_argument("--physical_vh_min_support_ratio", type=float, default=0.5)
    parser.add_argument("--positive_min_visible_views", type=int, default=1)
    parser.add_argument("--positive_min_support_ratio", type=float, default=0.5)
    parser.add_argument("--negative_min_visible_views", type=int, default=3)
    parser.add_argument("--negative_max_support_ratio", type=float, default=0.1)
    parser.add_argument("--negative_min_outside_ratio", type=float, default=0.9)
    parser.add_argument("--negative_prior_radius_multiplier", type=float, default=1.0)
    parser.add_argument("--projection_min_support_views", type=int, default=1)
    parser.add_argument("--projection_min_support_ratio", type=float, default=0.0)
    parser.add_argument("--visual_hull_min_visible_views", type=int, default=1)
    parser.add_argument("--visual_hull_min_support_ratio", type=float, default=0.0)
    parser.add_argument("--use_prior_score_positive", action="store_true")
    parser.add_argument("--visual_hull_active_weight", type=float, default=0.25)
    parser.add_argument("--loss_mask_mode", choices=["exclusive_surface", "legacy"], default="exclusive_surface")
    parser.add_argument(
        "--require_nonempty_surface_labels",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--candidate_min_changed_count", type=int, default=100)
    parser.add_argument("--candidate_min_changed_ratio", type=float, default=0.005)
    parser.add_argument("--candidate_max_changed_ratio", type=float, default=0.10)
    parser.add_argument("--candidate_min_set_iou", type=float, default=0.90)
    parser.add_argument("--candidate_max_absolute_outside_ratio", type=float, default=1.0)
    parser.add_argument("--candidate_max_component_increase", type=int, default=1)
    parser.add_argument("--candidate_min_coord_count_ratio", type=float, default=0.90)
    parser.add_argument("--candidate_max_coord_count_ratio", type=float, default=1.10)
    parser.add_argument(
        "--candidate_require_absolute_outside_nonincrease",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--adapter_type", choices=["mlp", "lowrank_basis"], default="mlp")
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--lowrank_rank", type=int, default=4)
    parser.add_argument("--lowrank_basis_init_std", type=float, default=0.02)
    parser.add_argument("--proxy_loss_mode", choices=["decoder_logits", "cond_basis", "lowrank_gate"], default="decoder_logits")
    parser.add_argument("--proxy_state_mode", choices=["random_xt", "stock_trajectory"], default="random_xt")
    parser.add_argument("--proxy_cfg_mode", choices=["unguided", "sampler"], default="unguided")
    parser.add_argument("--proxy_rescale_t", type=float, default=1.0)
    parser.add_argument("--proxy_trajectory_steps", type=int, default=12)
    parser.add_argument("--max_steps", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--t_values", default="0.5,0.75,0.9")
    parser.add_argument("--train_runtime_scale", type=float, default=1.0)
    parser.add_argument("--eval_runtime_scales", default="0.25,0.5,1.0")
    parser.add_argument("--margin_pos", type=float, default=0.003)
    parser.add_argument("--margin_neg", type=float, default=0.001)
    parser.add_argument("--cond_basis_pos_target", type=float, default=0.01)
    parser.add_argument("--cond_basis_neg_target", type=float, default=0.005)
    parser.add_argument("--gate_pos_target", type=float, default=0.02)
    parser.add_argument("--gate_neg_target", type=float, default=0.01)
    parser.add_argument("--pos_weight", type=float, default=1.0)
    parser.add_argument("--neg_weight", type=float, default=2.0)
    parser.add_argument("--preserve_weight", type=float, default=0.1)
    parser.add_argument("--gate_l2_weight", type=float, default=0.01)
    parser.add_argument("--delta_norm_weight", type=float, default=0.02)
    parser.add_argument("--smooth_weight", type=float, default=0.01)
    parser.add_argument("--delta_clip_abs", type=float, default=0.0)
    parser.add_argument("--grad_clip_norm", type=float, default=0.0)
    parser.add_argument("--backward_loss_scale", type=float, default=1.0)
    parser.add_argument("--nan_to_num_grads", action="store_true")
    parser.add_argument("--save_every", type=int, default=20)
    parser.add_argument("--log_every", type=int, default=5)
    parser.add_argument(
        "--verify_fixed_noise_repeat",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()

    startup_mapping_audit = _token_grid_mapping_audit(
        int(args.ss_grid_side),
        int(args.sparse_resolution),
    )
    if not startup_mapping_audit["passed"]:
        raise RuntimeError(f"SS token mapping/round-trip audit failed before pipeline load: {startup_mapping_audit}")

    output_dir = Path(args.output_dir)
    ckpt_dir = output_dir / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    sessions_spec = json.loads(Path(args.sessions_json).read_text(encoding="utf-8"))
    if not isinstance(sessions_spec, list) or not sessions_spec:
        raise ValueError("--sessions_json must contain a non-empty list")

    if not args.load_dreamsim:
        install_dreamsim_stub()
    device = torch.device(args.device if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    print(f"[B5.5] loading pipeline pretrained={args.pretrained} device={device}", flush=True)
    pipeline = TrellisVGGTTo3DPipeline.from_pretrained(args.pretrained)
    pipeline._device = device
    pipeline.low_vram = bool(args.low_vram)
    force_eval(pipeline)
    if hasattr(pipeline, "birefnet_model") and not pipeline.low_vram:
        pipeline.birefnet_model.to(device)
    if not pipeline.low_vram:
        for model in pipeline.models.values():
            model.to(device)
        pipeline.VGGT_model.to(device)
    force_eval(pipeline)

    torch.manual_seed(int(args.seed))
    sessions = [_prepare_session(pipeline=pipeline, args=args, spec=spec, device=device) for spec in sessions_spec]
    train_sessions = [s for s in sessions if s.split == "train"]
    if not train_sessions:
        raise ValueError("At least one session must have split='train'")

    flow = pipeline.models["sparse_structure_flow_model"].to(train_sessions[0].cond_base.device).eval()
    decoder = pipeline.models["sparse_structure_decoder"].to(train_sessions[0].cond_base.device).eval()
    set_frozen_eval(flow)
    set_frozen_eval(decoder)
    sampler = pipeline.sparse_structure_sampler

    if args.proxy_loss_mode == "lowrank_gate" and args.adapter_type != "lowrank_basis":
        raise ValueError("--proxy_loss_mode=lowrank_gate requires --adapter_type=lowrank_basis")
    if args.proxy_state_mode == "stock_trajectory" and args.proxy_cfg_mode != "sampler":
        raise ValueError("--proxy_state_mode=stock_trajectory requires --proxy_cfg_mode=sampler")
    if args.adapter_type == "lowrank_basis":
        adapter = LowRankPhysicalSSCondResidualAdapter(
            channels=int(train_sessions[0].cond_base.shape[-1]),
            feature_dim=len(TRAIN_FEATURE_NAMES),
            hidden_dim=int(args.hidden_dim),
            rank=int(args.lowrank_rank),
            basis_init_std=float(args.lowrank_basis_init_std),
        ).to(train_sessions[0].cond_base.device)
    else:
        adapter = PhysicalSSCondResidualAdapter(
            channels=int(train_sessions[0].cond_base.shape[-1]),
            feature_dim=len(TRAIN_FEATURE_NAMES),
            hidden_dim=int(args.hidden_dim),
        ).to(train_sessions[0].cond_base.device)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=float(args.lr), weight_decay=0.0)
    t_values = _parse_t_values(args.t_values)

    proxy_cache: dict[tuple[str, float], dict[str, Any]] = {}
    proxy_cache_summary: list[dict[str, Any]] = []
    if args.proxy_loss_mode == "decoder_logits":
        for sidx, session in enumerate(train_sessions):
            torch.manual_seed(int(args.seed) + 17 * sidx)
            initial_noise = torch.randn(
                int(args.num_samples),
                int(flow.in_channels),
                int(flow.resolution),
                int(flow.resolution),
                int(flow.resolution),
                device=session.cond_base.device,
                dtype=torch.float32,
            )
            if args.proxy_state_mode == "stock_trajectory":
                state_cache = _stock_trajectory_cache(
                    flow=flow,
                    sampler=sampler,
                    session=session,
                    initial_noise=initial_noise,
                    requested_t_values=t_values,
                    args=args,
                )
            else:
                state_cache = {
                    float(t): {
                        "x_t": initial_noise.detach().clone(),
                        "requested_t_raw": float(t),
                        "snapped_t_raw": float(t),
                        "t_model": _rescale_proxy_time(float(t), float(args.proxy_rescale_t)),
                        "trajectory_state_index": None,
                    }
                    for t in t_values
                }
            for t in t_values:
                state_row = state_cache[float(t)]
                with torch.no_grad():
                    base_logits, _ = _one_step_logits(
                        flow=flow,
                        decoder=decoder,
                        sampler=sampler,
                        x_t=state_row["x_t"],
                        t_model=float(state_row["t_model"]),
                        cond=session.cond_base.float(),
                        neg_cond=session.neg_cond,
                        cfg_mode=str(args.proxy_cfg_mode),
                        cfg_strength=float(args.ss_cfg_strength),
                        cfg_interval=(float(args.ss_cfg_interval_min), float(args.ss_cfg_interval_max)),
                        guidance_rescale=float(args.ss_guidance_rescale),
                        autocast_enabled=(session.cond_base.device.type == "cuda"),
                    )
                cache_row = {**state_row, "base_logits": base_logits.detach()}
                proxy_cache[(session.name, float(t))] = cache_row
                proxy_cache_summary.append(
                    {
                        "session": session.name,
                        "requested_t_raw": float(t),
                        "snapped_t_raw": float(state_row["snapped_t_raw"]),
                        "t_model": float(state_row["t_model"]),
                        "trajectory_state_index": state_row["trajectory_state_index"],
                        "state_mode": str(args.proxy_state_mode),
                        "cfg_mode": str(args.proxy_cfg_mode),
                    }
                )

    rows: list[dict[str, Any]] = []
    adapter.train()
    for step in range(1, int(args.max_steps) + 1):
        optimizer.zero_grad(set_to_none=True)
        total_loss = torch.zeros((), device=train_sessions[0].cond_base.device, dtype=torch.float32)
        row_sessions: list[dict[str, Any]] = []
        t = float(t_values[(step - 1) % len(t_values)])
        for session in train_sessions:
            if args.proxy_loss_mode == "decoder_logits":
                cache = proxy_cache[(session.name, t)]
                loss, stats = _train_step_loss(
                    adapter=adapter,
                    session=session,
                    base_logits=cache["base_logits"],
                    flow=flow,
                    decoder=decoder,
                    sampler=sampler,
                    x_t=cache["x_t"],
                    t=float(cache["t_model"]),
                    args=args,
                )
                stats["requested_t_raw"] = float(t)
                stats["snapped_t_raw"] = float(cache["snapped_t_raw"])
                stats["t_model"] = float(cache["t_model"])
            elif args.proxy_loss_mode == "cond_basis":
                loss, stats = _train_cond_basis_loss(adapter=adapter, session=session, args=args)
            else:
                if not isinstance(adapter, LowRankPhysicalSSCondResidualAdapter):
                    raise TypeError("lowrank_gate loss requires LowRankPhysicalSSCondResidualAdapter")
                loss, stats = _train_lowrank_gate_loss(adapter=adapter, session=session, args=args)
            total_loss = total_loss + loss
            row_sessions.append({"session": session.name, **stats})
        total_loss = total_loss / float(len(train_sessions))
        if not torch.isfinite(total_loss):
            optimizer.zero_grad(set_to_none=True)
            row = {
                "step": int(step),
                "t": float(t),
                "proxy_t_model": float(np.mean([x.get("t_model", t) for x in row_sessions])),
                "loss": float("nan"),
                "skipped_nonfinite": True,
                "sessions": row_sessions,
            }
            rows.append(row)
            print(f"[B5.5][warn] step={step} t={t:.2f} skipped non-finite loss", flush=True)
            continue
        backward_loss = total_loss * float(args.backward_loss_scale)
        backward_loss.backward()
        sanitized_grad_nonfinite = 0
        if args.nan_to_num_grads:
            for param in adapter.parameters():
                if param.grad is None:
                    continue
                nonfinite = ~torch.isfinite(param.grad)
                sanitized_grad_nonfinite += int(nonfinite.sum().detach().cpu().item())
                if bool(nonfinite.any().item()):
                    param.grad.data = torch.nan_to_num(param.grad.data, nan=0.0, posinf=0.0, neginf=0.0)
        grad_norm = None
        if float(args.grad_clip_norm) > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                adapter.parameters(),
                max_norm=float(args.grad_clip_norm),
                error_if_nonfinite=False,
            )
            if not torch.isfinite(grad_norm):
                optimizer.zero_grad(set_to_none=True)
                row = {
                    "step": int(step),
                    "t": float(t),
                    "loss": float(total_loss.detach().cpu().item()),
                    "skipped_nonfinite_grad": True,
                    "grad_norm": float("nan"),
                    "sessions": row_sessions,
                }
                rows.append(row)
                print(f"[B5.5][warn] step={step} t={t:.2f} skipped non-finite grad", flush=True)
                continue
        optimizer.step()

        if step == 1 or step % int(args.log_every) == 0 or step == int(args.max_steps):
            row = {
                "step": int(step),
                "t": float(t),
                "proxy_t_model": float(np.mean([x.get("t_model", t) for x in row_sessions])),
                "loss": float(total_loss.detach().cpu().item()),
                "pos_loss": float(np.mean([x["pos_loss"] for x in row_sessions])),
                "neg_loss": float(np.mean([x["neg_loss"] for x in row_sessions])),
                "preserve_loss": float(np.mean([x["preserve_loss"] for x in row_sessions])),
                "delta_norm_ratio": float(np.mean([x["delta_norm_ratio"] for x in row_sessions])),
                "dlogits_pos_mean": float(np.mean([x["dlogits_pos_mean"] for x in row_sessions])),
                "dlogits_neg_mean": float(np.mean([x["dlogits_neg_mean"] for x in row_sessions])),
                "dlogits_neutral_abs_mean": float(np.mean([x["dlogits_neutral_abs_mean"] for x in row_sessions])),
                "sessions": row_sessions,
            }
            if grad_norm is not None:
                row["grad_norm"] = float(grad_norm.detach().cpu().item())
            if args.nan_to_num_grads:
                row["sanitized_grad_nonfinite"] = int(sanitized_grad_nonfinite)
            rows.append(row)
            extra = ""
            if "grad_norm" in row:
                extra += f" grad={row['grad_norm']:.6g}"
            if "sanitized_grad_nonfinite" in row:
                extra += f" sanitized_grad={row['sanitized_grad_nonfinite']}"
            print(
                f"[B5.5] step={step} t={t:.2f} loss={row['loss']:.6g} "
                f"pos={row['pos_loss']:.6g} neg={row['neg_loss']:.6g} "
                f"pres={row['preserve_loss']:.6g} delta_norm={row['delta_norm_ratio']:.6g} "
                f"dpos={row['dlogits_pos_mean']:.6g} dneg={row['dlogits_neg_mean']:.6g}"
                + extra,
                flush=True,
            )
        if step % int(args.save_every) == 0 or step == int(args.max_steps):
            torch.save(
                {
                    "state_dict": adapter.state_dict(),
                    "metadata": adapter.metadata(),
                    "args": vars(args),
                    "rows": rows,
                },
                ckpt_dir / f"adapter_step_{step:06d}.ckpt",
            )

    final_ckpt = ckpt_dir / "last.ckpt"
    torch.save(
        {
            "state_dict": adapter.state_dict(),
            "metadata": adapter.metadata(),
            "args": vars(args),
            "rows": rows,
        },
        final_ckpt,
    )

    session_reports = _evaluate(pipeline=pipeline, flow=flow, adapter=adapter, args=args, sessions=sessions, output_dir=output_dir)
    aggregate = _aggregate_eval(session_reports, args)
    strict_any = any(int(row["strict_direction_session_count"]) == len(session_reports) for row in aggregate)
    judgment = (
        "B5.5 physical proxy found an eval scale that passes direction, changed-ratio, set-IoU, and absolute-outside checks on all train/validation sessions."
        if strict_any
        else "B5.5 physical proxy did not find a strict all-session pass; do not increase steps or capacity. Recheck physical labels and the flow-gradient audit first."
    )
    command = " ".join(sys.argv)
    report = {
        "args": vars(args),
        "command": command,
        "output_dir": str(output_dir),
        "checkpoint": str(final_ckpt),
        "adapter": adapter.metadata(),
        "startup_mapping_audit": startup_mapping_audit,
        "rows": rows,
        "proxy_cache_summary": proxy_cache_summary,
        "sessions": session_reports,
        "aggregate": aggregate,
        "judgment": judgment,
        "scope": "B5.5 SS-only physical proxy residual adapter; frozen ReconViaGen bridge, sparse flow, and decoder; SLAT not touched. Validation sessions participate in candidate selection and are not final holdout.",
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    _write_md(output_dir / "report.md", report, command)
    print(f"[B5.5] wrote {output_dir / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
