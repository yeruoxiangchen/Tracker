#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

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
from reconvggt_ar_adapter_a.train_b5_ss_cond_residual_adapter import install_dreamsim_stub, set_frozen_eval  # noqa: E402
from trellis_point_prior_mv.sparse_coord_tools import (  # noqa: E402
    coords_xyz,
    projection_support_counts,
    sparse_diagnostic_metrics,
)


PHYSICAL_FEATURE_NAMES = [
    "support_views",
    "visible_views",
    "mask_support_ratio",
    "outside_visible_ratio",
    "point_distance",
    "prior_score",
    "prior_within_radius",
    "any_mask_hit",
    "visual_hull_inside",
    "support_gate",
    "outside_gate",
    "contrast_gate",
    "visual_hull_gate",
    "prior_gate",
    "support_view_fraction",
    "visible_view_fraction",
    "point_distance_over_radius",
    "confident_positive",
    "reliable_outside",
    "surface_contrast",
]


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


def _parse_list(spec: str) -> list[str]:
    return [x.strip() for x in str(spec).split(",") if x.strip()]


def _parse_float_list(spec: str) -> list[float]:
    vals = [float(x.strip()) for x in str(spec).split(",") if x.strip()]
    if not vals:
        raise ValueError("empty float list")
    return vals


def _parse_gate_formulas(spec: str) -> list[tuple[str, dict[str, float]]]:
    formulas: list[tuple[str, dict[str, float]]] = []
    for item in [x.strip() for x in str(spec or "").split(";") if x.strip()]:
        if ":" not in item:
            raise ValueError(f"Formula must be name:gate=weight,... got {item!r}")
        name, body = item.split(":", 1)
        weights: dict[str, float] = {}
        for part in [x.strip() for x in body.split(",") if x.strip()]:
            if "=" not in part:
                raise ValueError(f"Formula term must be gate=weight got {part!r} in {item!r}")
            gate_name, value = part.split("=", 1)
            weights[gate_name.strip()] = float(value)
        if not weights:
            raise ValueError(f"Formula {name!r} has no terms")
        formulas.append((name.strip(), weights))
    return formulas


def _scale_name(scale: float) -> str:
    prefix = "p" if scale >= 0 else "m"
    body = f"{abs(float(scale)):.6f}".rstrip("0").rstrip(".")
    return f"{prefix}{body.replace('.', 'p')}"


def _ss_grid_coords64(ss_grid_side: int = 16, sparse_resolution: int = 64) -> np.ndarray:
    if sparse_resolution % ss_grid_side != 0:
        raise ValueError(f"sparse_resolution={sparse_resolution} must divide ss_grid_side={ss_grid_side}")
    step = sparse_resolution // ss_grid_side
    vals = np.arange(ss_grid_side, dtype=np.int32) * step + step // 2
    xx, yy, zz = np.meshgrid(vals, vals, vals, indexing="ij")
    # TRELLIS flattens dense [x,y,z] volumes with z as the fastest axis.
    # Keep the physical feature row order identical to the 4096 SS tokens.
    xyz = np.stack((xx, yy, zz), axis=-1).reshape(-1, 3).astype(np.int32)
    batch = np.zeros((xyz.shape[0], 1), dtype=np.int32)
    return np.concatenate((batch, xyz), axis=1)


def _xyz16_to_token_index(x: int, y: int, z: int, side: int) -> int:
    if not (0 <= x < side and 0 <= y < side and 0 <= z < side):
        raise ValueError(f"xyz16 out of bounds: {(x, y, z)} side={side}")
    return int(x) * int(side) * int(side) + int(y) * int(side) + int(z)


def upsample_ss_tokens_to_sparse(
    values: torch.Tensor,
    *,
    ss_grid_side: int = 16,
    sparse_resolution: int = 64,
) -> torch.Tensor:
    side = int(ss_grid_side)
    resolution = int(sparse_resolution)
    if resolution % side != 0:
        raise ValueError(f"sparse_resolution={resolution} must divide ss_grid_side={side}")
    if values.numel() != side ** 3:
        raise ValueError(f"Expected {side ** 3} SS tokens, got {values.numel()}")
    step = resolution // side
    dense = values.reshape(1, 1, side, side, side)
    return (
        dense.repeat_interleave(step, dim=2)
        .repeat_interleave(step, dim=3)
        .repeat_interleave(step, dim=4)
        .contiguous()
    )


def _single_token_roundtrip_audit(ss_grid_side: int, sparse_resolution: int) -> dict[str, Any]:
    side = int(ss_grid_side)
    resolution = int(sparse_resolution)
    if resolution % side != 0:
        raise ValueError(f"sparse_resolution={resolution} must divide ss_grid_side={side}")
    step = resolution // side
    token_index = 1
    coarse = torch.zeros(side ** 3, dtype=torch.float32)
    coarse[token_index] = 1.0
    fine_t = upsample_ss_tokens_to_sparse(
        coarse,
        ss_grid_side=side,
        sparse_resolution=resolution,
    )[0, 0]
    fine = fine_t.cpu().numpy()
    nonzero = np.argwhere(fine > 0.5)
    downsampled_t = fine_t.reshape(side, step, side, step, side, step).amax(dim=(1, 3, 5))
    recovered = torch.nonzero(downsampled_t.reshape(-1) > 0.5, as_tuple=False).reshape(-1).tolist()
    expected_min = [0, 0, step]
    expected_max = [step - 1, step - 1, 2 * step - 1]
    actual_min = nonzero.min(axis=0).astype(int).tolist() if nonzero.size else None
    actual_max = nonzero.max(axis=0).astype(int).tolist() if nonzero.size else None
    expected_count = step ** 3
    passed = bool(
        nonzero.shape[0] == expected_count
        and actual_min == expected_min
        and actual_max == expected_max
        and recovered == [token_index]
        and float(downsampled_t.reshape(-1)[token_index].item()) == 1.0
    )
    return {
        "passed": passed,
        "token_index": token_index,
        "fine_nonzero_count": int(nonzero.shape[0]),
        "expected_fine_nonzero_count": int(expected_count),
        "actual_fine_xyz_min": actual_min,
        "actual_fine_xyz_max": actual_max,
        "expected_fine_xyz_min": expected_min,
        "expected_fine_xyz_max": expected_max,
        "recovered_token_indices": recovered,
    }


def _token_grid_mapping_audit(ss_grid_side: int = 16, sparse_resolution: int = 64) -> dict[str, Any]:
    coords = _ss_grid_coords64(ss_grid_side, sparse_resolution)
    indices = [0, 1, int(ss_grid_side), int(ss_grid_side) ** 2]
    rows = []
    for index in indices:
        x = index // (int(ss_grid_side) ** 2)
        rem = index % (int(ss_grid_side) ** 2)
        y = rem // int(ss_grid_side)
        z = rem % int(ss_grid_side)
        rows.append(
            {
                "token_index": int(index),
                "expected_xyz16": [int(x), int(y), int(z)],
                "inverse_token_index": _xyz16_to_token_index(x, y, z, int(ss_grid_side)),
                "actual_xyz64_center": [int(v) for v in coords[index, -3:].tolist()],
            }
        )
    step = int(sparse_resolution) // int(ss_grid_side)
    sentinel_passed = all(
        row["inverse_token_index"] == row["token_index"]
        and row["actual_xyz64_center"]
        == [int(v) * step + step // 2 for v in row["expected_xyz16"]]
        for row in rows
    )
    roundtrip = _single_token_roundtrip_audit(int(ss_grid_side), int(sparse_resolution))
    return {
        "passed": bool(sentinel_passed and roundtrip["passed"]),
        "sentinel_passed": bool(sentinel_passed),
        "rows": rows,
        "single_token_roundtrip": roundtrip,
    }


def _apply_physical_frame_scope(
    sample: dict[str, Any],
    *,
    scope: str,
    selected_image_names: Sequence[str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    frames = list(sample.get("frames") or [])
    if scope == "fullscan":
        return sample, {
            "scope": scope,
            "input_frame_count": len(frames),
            "output_frame_count": len(frames),
            "selected_image_names": [],
        }
    if scope != "selected":
        raise ValueError(f"Unsupported physical_frame_scope={scope!r}")
    if not selected_image_names:
        raise ValueError("physical_frame_scope=selected requires selected_image_names")
    selected = {Path(name).name for name in selected_image_names}
    scoped_frames = [frame for frame in frames if Path(str(frame.get("name", ""))).name in selected]
    if not scoped_frames:
        raise ValueError(
            "physical_frame_scope=selected matched no manifest frames: "
            f"selected={sorted(selected)} manifest={[frame.get('name') for frame in frames]}"
        )
    scoped = copy.deepcopy(sample)
    scoped["frames"] = scoped_frames
    return scoped, {
        "scope": scope,
        "input_frame_count": len(frames),
        "output_frame_count": len(scoped_frames),
        "selected_image_names": sorted(selected),
        "matched_frame_names": [str(frame.get("name", "")) for frame in scoped_frames],
    }


def _nearest_dist(coords: np.ndarray, prior_coords: np.ndarray) -> np.ndarray:
    xyz = coords_xyz(coords).astype(np.float32)
    prior = coords_xyz(prior_coords).astype(np.float32)
    if xyz.shape[0] == 0 or prior.shape[0] == 0:
        return np.full((xyz.shape[0],), 1.0e6, dtype=np.float32)
    try:
        from scipy.spatial import cKDTree

        dist = cKDTree(prior).query(xyz, k=1)[0]
    except Exception:
        diff = xyz[:, None, :] - prior[None, :, :]
        dist = np.sqrt(np.min(np.sum(diff * diff, axis=-1), axis=1))
    return np.asarray(dist, dtype=np.float32)


def _stats(values: np.ndarray) -> dict[str, float | int]:
    x = np.asarray(values, dtype=np.float32).reshape(-1)
    if x.size == 0:
        return {"count": 0, "mean": 0.0, "std": 0.0, "min": 0.0, "p25": 0.0, "p50": 0.0, "p75": 0.0, "max": 0.0}
    return {
        "count": int(x.size),
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "min": float(np.min(x)),
        "p25": float(np.percentile(x, 25)),
        "p50": float(np.percentile(x, 50)),
        "p75": float(np.percentile(x, 75)),
        "max": float(np.max(x)),
    }


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    if a.size < 2 or b.size != a.size or float(np.std(a)) < 1.0e-8 or float(np.std(b)) < 1.0e-8:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _build_physical_features(
    *,
    sample: dict[str, Any],
    prior_coords: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, Any]]:
    mapping_audit = _token_grid_mapping_audit(int(args.ss_grid_side), int(args.sparse_resolution))
    if not mapping_audit["passed"]:
        raise RuntimeError(f"SS token mapping/round-trip audit failed: {mapping_audit}")
    coords64 = _ss_grid_coords64(int(args.ss_grid_side), int(args.sparse_resolution))
    dist = _nearest_dist(coords64, prior_coords)
    support, visible, matched = projection_support_counts(
        sample,
        coords64,
        grid_resolution=int(args.sparse_resolution),
        mask_threshold=int(args.mask_threshold),
    )
    support = support.astype(np.float32)
    visible = visible.astype(np.float32)
    support_ratio = np.zeros_like(support, dtype=np.float32)
    np.divide(support, np.maximum(visible, 1.0), out=support_ratio, where=visible > 0)
    outside_ratio = np.zeros_like(support, dtype=np.float32)
    np.divide(np.maximum(visible - support, 0.0), np.maximum(visible, 1.0), out=outside_ratio, where=visible > 0)
    prior_score = np.exp(-dist / max(float(args.prior_radius), 1.0e-6)).astype(np.float32)
    prior_within = (dist <= float(args.prior_radius)).astype(np.float32)
    mask_hit = (support > 0).astype(np.float32)
    visible_flag = (visible > 0).astype(np.float32)
    visual_hull_inside = (
        (visible >= float(args.physical_vh_min_visible_views))
        & (support_ratio >= float(args.physical_vh_min_support_ratio))
    ).astype(np.float32)
    support_gate = (support_ratio * prior_score).astype(np.float32)
    outside_gate = (outside_ratio * visible_flag).astype(np.float32)
    contrast_gate = (support_gate - outside_gate).astype(np.float32)
    vh_gate = visual_hull_inside.astype(np.float32)
    prior_gate = prior_score.astype(np.float32)
    matched_denom = float(max(int(matched), 1))
    support_view_fraction = (support / matched_denom).astype(np.float32)
    visible_view_fraction = (visible / matched_denom).astype(np.float32)
    point_distance_over_radius = np.clip(
        dist / max(float(args.prior_radius), 1.0e-6),
        0.0,
        float(getattr(args, "physical_distance_clip", 8.0)),
    ).astype(np.float32)

    pos_min_visible = int(getattr(args, "positive_min_visible_views", 1))
    pos_min_support = float(getattr(args, "positive_min_support_ratio", 0.5))
    neg_min_visible = int(getattr(args, "negative_min_visible_views", 3))
    neg_max_support = float(getattr(args, "negative_max_support_ratio", 0.1))
    neg_min_outside = float(getattr(args, "negative_min_outside_ratio", 0.9))
    neg_radius_multiplier = float(getattr(args, "negative_prior_radius_multiplier", 1.0))
    confident_positive = (
        (dist <= float(args.prior_radius))
        & (visible >= float(pos_min_visible))
        & (support_ratio >= pos_min_support)
    )
    reliable_outside = (
        (dist > float(args.prior_radius) * neg_radius_multiplier)
        & (visible >= float(neg_min_visible))
        & (support_ratio <= neg_max_support)
        & (outside_ratio >= neg_min_outside)
    )
    # Defensive exclusivity: reliable outside evidence never overrides an
    # observed near-prior surface cell.
    reliable_outside &= ~confident_positive
    confident_positive = confident_positive.astype(np.float32)
    reliable_outside = reliable_outside.astype(np.float32)
    overlap_count = int(((confident_positive > 0) & (reliable_outside > 0)).sum())
    positive_count = int((confident_positive > 0).sum())
    negative_count = int((reliable_outside > 0).sum())
    if overlap_count != 0:
        raise RuntimeError(f"physical surface labels overlap at {overlap_count} SS tokens")
    if bool(getattr(args, "require_nonempty_surface_labels", True)) and (
        positive_count == 0 or negative_count == 0
    ):
        raise RuntimeError(
            f"physical surface labels must be non-empty: positive={positive_count} negative={negative_count}"
        )
    surface_contrast = (confident_positive - reliable_outside).astype(np.float32)
    features = np.stack(
        (
            support,
            visible,
            support_ratio,
            outside_ratio,
            dist,
            prior_score,
            prior_within,
            mask_hit,
            visual_hull_inside,
            support_gate,
            outside_gate,
            contrast_gate,
            vh_gate,
            prior_gate,
            support_view_fraction,
            visible_view_fraction,
            point_distance_over_radius,
            confident_positive,
            reliable_outside,
            surface_contrast,
        ),
        axis=-1,
    )
    names = PHYSICAL_FEATURE_NAMES
    if features.shape[-1] != len(names):
        raise AssertionError(f"Physical feature mismatch: {features.shape[-1]} != {len(names)}")
    summary = {
        "grid": {
            "ss_grid_side": int(args.ss_grid_side),
            "sparse_resolution": int(args.sparse_resolution),
            "token_count": int(coords64.shape[0]),
            "token_mapping_audit": mapping_audit,
        },
        "projection_matched_frame_count": int(matched),
        "feature_names": names,
        "feature_stats": {name: _stats(features[:, i]) for i, name in enumerate(names)},
        "feature_corr": {
            "support_gate_vs_mask_hit": _safe_corr(support_gate, mask_hit),
            "support_gate_vs_prior_within": _safe_corr(support_gate, prior_within),
            "outside_gate_vs_mask_hit": _safe_corr(outside_gate, mask_hit),
            "outside_gate_vs_visual_hull_inside": _safe_corr(outside_gate, visual_hull_inside),
            "contrast_gate_vs_visual_hull_inside": _safe_corr(contrast_gate, visual_hull_inside),
            "prior_score_vs_prior_within": _safe_corr(prior_score, prior_within),
        },
        "sanity": {
            "support_gate_nonzero_ratio": float((support_gate > 0).mean()),
            "outside_gate_nonzero_ratio": float((outside_gate > 0).mean()),
            "visual_hull_inside_ratio": float(visual_hull_inside.mean()),
            "prior_within_radius_ratio": float(prior_within.mean()),
            "mask_hit_ratio": float(mask_hit.mean()),
            "confident_positive_ratio": float(confident_positive.mean()),
            "reliable_outside_ratio": float(reliable_outside.mean()),
            "positive_negative_overlap_ratio": float(
                ((confident_positive > 0) & (reliable_outside > 0)).mean()
            ),
        },
        "surface_label_counts": {
            "positive": positive_count,
            "negative": negative_count,
            "neutral": int(((confident_positive == 0) & (reliable_outside == 0)).sum()),
            "overlap": overlap_count,
        },
        "surface_label_feature_stats": {
            label: {
                "point_distance": _stats(dist[mask]),
                "support_ratio": _stats(support_ratio[mask]),
                "outside_ratio": _stats(outside_ratio[mask]),
                "visible_views": _stats(visible[mask]),
            }
            for label, mask in {
                "positive": confident_positive > 0,
                "negative": reliable_outside > 0,
                "neutral": (confident_positive == 0) & (reliable_outside == 0),
            }.items()
        },
        "surface_label_thresholds": {
            "positive_min_visible_views": pos_min_visible,
            "positive_min_support_ratio": pos_min_support,
            "negative_min_visible_views": neg_min_visible,
            "negative_max_support_ratio": neg_max_support,
            "negative_min_outside_ratio": neg_min_outside,
            "negative_prior_radius_multiplier": neg_radius_multiplier,
        },
    }
    return features.astype(np.float32), summary


def _standardize_gate(gate: np.ndarray) -> np.ndarray:
    g = np.asarray(gate, dtype=np.float32).reshape(-1)
    g = g - np.mean(g)
    std = float(np.std(g))
    if std < 1.0e-6:
        return np.zeros_like(g, dtype=np.float32)
    return (g / std).astype(np.float32)


def _normalize_candidate_gate(gate: np.ndarray, mode: str) -> np.ndarray:
    raw = np.asarray(gate, dtype=np.float32).reshape(-1)
    if mode == "none":
        return raw.copy()
    if mode == "active_rms":
        active = np.abs(raw) > 0
        out = np.zeros_like(raw)
        if np.any(active):
            rms = float(np.sqrt(np.mean(raw[active] * raw[active])))
            out[active] = raw[active] / max(rms, 1.0e-6)
        return out
    if mode == "standardize":
        return _standardize_gate(raw)
    raise ValueError(f"Unsupported candidate gate normalization: {mode}")


def _basis(cond_base: torch.Tensor, mode: str) -> torch.Tensor:
    x = cond_base.detach().float()
    if mode == "cond_unit":
        y = x / torch.sqrt((x * x).mean(dim=-1, keepdim=True)).clamp_min(1.0e-6)
    elif mode == "cond_centered_unit":
        y = x - x.mean(dim=1, keepdim=True)
        y = y / torch.sqrt((y * y).mean(dim=-1, keepdim=True)).clamp_min(1.0e-6)
    else:
        raise ValueError(f"Unsupported basis mode: {mode}")
    cond_rms = torch.sqrt((x * x).mean(dim=(1, 2), keepdim=True)).clamp_min(1.0e-6)
    return (y * cond_rms).to(dtype=cond_base.dtype)


def _compose_gate(
    features: np.ndarray,
    weights: dict[str, float],
    gate_index: dict[str, int],
    *,
    normalization: str,
    allow_neutral_leakage: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    gate = np.zeros((features.shape[0],), dtype=np.float32)
    pieces: dict[str, Any] = {}
    for name, weight in weights.items():
        if name not in gate_index:
            raise KeyError(f"Unknown gate {name}; choices={sorted(gate_index)}")
        raw = features[:, gate_index[name]]
        gate += float(weight) * raw
        pieces[name] = {
            "weight": float(weight),
            "raw_stats": _stats(raw),
        }
    raw_gate = gate.copy()
    gate = _normalize_candidate_gate(raw_gate, normalization)
    neutral = np.abs(raw_gate) <= 1.0e-12
    neutral_leakage_max = float(np.max(np.abs(gate[neutral]))) if np.any(neutral) else 0.0
    neutral_leakage_count = int((np.abs(gate[neutral]) > 1.0e-8).sum()) if np.any(neutral) else 0
    if neutral_leakage_count and not allow_neutral_leakage:
        raise RuntimeError(
            f"candidate normalization={normalization!r} changed {neutral_leakage_count} neutral tokens; "
            "use none/active_rms or explicitly allow leakage for a diagnostic only"
        )
    return gate.astype(np.float32), {
        "weights": dict(weights),
        "pieces": pieces,
        "normalization": normalization,
        "raw_composed_stats": _stats(raw_gate),
        "composed_stats": _stats(gate),
        "raw_neutral_count": int(neutral.sum()),
        "neutral_leakage_count": neutral_leakage_count,
        "neutral_leakage_max_abs": neutral_leakage_max,
    }


def _session_args(args: argparse.Namespace, spec: dict[str, Any]) -> argparse.Namespace:
    out = copy.copy(args)
    out.image_dir = spec["image_dir"]
    out.pose_file = spec["pose_file"]
    out.mask_dir = spec.get("mask_dir", "")
    out.colmap_sparse_dir = spec.get("colmap_sparse_dir", "")
    out.points3d_txt = spec.get("points3d_txt", "")
    out.point_prior_npz = spec.get("point_prior_npz", "")
    return out


def _prepare_cond(
    *,
    pipeline: TrellisVGGTTo3DPipeline,
    args: argparse.Namespace,
    spec: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    sargs = _session_args(args, spec)
    if args.mask_mode == "apply":
        images, image_names, mask_summaries = _load_images_with_masks(
            Path(sargs.image_dir),
            mask_dir=Path(sargs.mask_dir),
            max_views=int(args.max_views),
            mask_background=args.mask_background,
        )
    else:
        images, image_names = load_images(Path(sargs.image_dir), max_views=int(args.max_views), preprocess=args.preprocess, pipeline=pipeline)
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
    return {
        "image_names": image_names,
        "mask_summaries": mask_summaries,
        "cond_base": ss_cond_base["cond"].detach(),
        "neg_cond": ss_cond_base["neg_cond"].detach(),
    }


def _candidate_delta(
    *,
    cond_base: torch.Tensor,
    features: np.ndarray,
    gate_name: str,
    gate_weights: dict[str, float],
    basis_mode: str,
    scale: float,
    gate_index: dict[str, int],
    gate_normalization: str,
    allow_neutral_leakage: bool,
) -> tuple[torch.Tensor, dict[str, Any]]:
    gate, gate_meta = _compose_gate(
        features,
        gate_weights,
        gate_index,
        normalization=gate_normalization,
        allow_neutral_leakage=allow_neutral_leakage,
    )
    gate_t = torch.from_numpy(gate).to(device=cond_base.device, dtype=torch.float32).reshape(1, -1, 1)
    basis = _basis(cond_base, basis_mode).float()
    delta = basis * gate_t * float(scale)
    return delta.to(dtype=cond_base.dtype), {
        "gate_name": gate_name,
        "gate_weights": dict(gate_weights),
        "gate_meta": gate_meta,
        "basis_mode": basis_mode,
        "scale": float(scale),
        "gate_stats": _stats(gate),
        "delta": summarize_tensor(delta),
    }


@torch.no_grad()
def sample_sparse_structure_fixed_noise(
    *,
    pipeline: TrellisVGGTTo3DPipeline,
    cond: dict[str, torch.Tensor],
    noise: torch.Tensor,
    sampler_params: dict[str, Any],
) -> tuple[torch.Tensor, None]:
    """Sample through ReconViaGen while explicitly supplying the initial noise.

    The configured ``FlowEulerGuidanceIntervalSampler`` preserves this tensor.
    Callers additionally repeat the baseline without resetting RNG and require
    identical coordinates, which catches a future sampler that redraws noise.
    """
    coords = pipeline.sample_sparse_structure(
        cond,
        num_samples=int(noise.shape[0]),
        sampler_params=sampler_params,
        noise=noise.detach().clone(),
    )
    return coords, None


def _evaluate_candidate(
    *,
    pipeline: TrellisVGGTTo3DPipeline,
    args: argparse.Namespace,
    cond_base: torch.Tensor,
    neg_cond: torch.Tensor,
    delta: torch.Tensor,
    ss_noise: torch.Tensor,
    flow,
    out_dir: Path,
    prior_sample: dict[str, Any] | None,
    prior_coords: np.ndarray | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    params = {
        "steps": int(args.ss_steps),
        "cfg_strength": float(args.ss_cfg_strength),
        "cfg_interval": [0.6, 1.0],
        "guidance_rescale": float(args.ss_guidance_rescale),
        "rescale_t": float(args.ss_rescale_t),
    }
    ss_cond = {"cond": (cond_base + delta).to(dtype=cond_base.dtype), "neg_cond": neg_cond}
    coords, _ = sample_sparse_structure_fixed_noise(
        pipeline=pipeline,
        cond=ss_cond,
        noise=ss_noise,
        sampler_params=params,
    )
    coords_array = coords_np(coords)
    np.savez_compressed(out_dir / "coords.npz", coords=coords_array)
    prior_alignment = None
    if prior_sample is not None and prior_coords is not None:
        prior_alignment = sparse_diagnostic_metrics(
            "b54_sparse",
            coords_array,
            prior_coords,
            prior_sample,
            prior_radius=float(args.prior_radius),
            min_support_views=int(args.projection_min_support_views),
            min_support_ratio=float(args.projection_min_support_ratio),
            visual_hull_min_visible_views=int(args.visual_hull_min_visible_views),
            visual_hull_min_support_ratio=float(args.visual_hull_min_support_ratio),
            grid_resolution=int(args.sparse_resolution),
            mask_threshold=int(args.mask_threshold),
        )
    return coords_array, {
        "sparse": _component_stats(coords_array),
        "prior_alignment": prior_alignment,
        "output_dir": str(out_dir),
        "sampling_backend": "native_pipeline_supplied_noise",
    }


def _metric_by_suffix(metrics: dict[str, Any] | None, suffix: str) -> float | None:
    if not metrics:
        return None
    matches = [value for key, value in metrics.items() if str(key).endswith(suffix)]
    if len(matches) > 1:
        raise ValueError(f"Ambiguous metric suffix {suffix!r}: found {len(matches)} matches")
    if not matches or matches[0] is None:
        return None
    return float(matches[0])


def candidate_quality_row(
    *,
    session_name: str,
    split: str,
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    delta = candidate.get("delta_vs_baseline") or {}
    direction = delta.get("direction_summary") or {}
    set_compare = candidate.get("set_compare_vs_baseline") or delta.get("set_compare") or {}
    baseline_count = int(set_compare.get("baseline_count") or 0)
    added_count = int(set_compare.get("added_count") or 0)
    removed_count = int(set_compare.get("removed_count") or 0)
    changed_count = added_count + removed_count
    changed_ratio = float(changed_count) / float(max(baseline_count, 1))
    min_changed_count = max(
        int(args.candidate_min_changed_count),
        int(math.ceil(float(args.candidate_min_changed_ratio) * float(baseline_count))),
    )
    baseline_outside = _metric_by_suffix(
        baseline.get("prior_alignment"), "_visible_outside_mask_event_ratio"
    )
    candidate_outside = _metric_by_suffix(
        candidate.get("prior_alignment"), "_visible_outside_mask_event_ratio"
    )
    absolute_outside_delta = (
        candidate_outside - baseline_outside
        if candidate_outside is not None and baseline_outside is not None
        else None
    )
    mask_direction = direction.get("added_minus_removed_projection_any_mask_hit_ratio")
    outside_direction = direction.get("added_minus_removed_visible_outside_mask_event_ratio")
    prior_direction = direction.get("added_minus_removed_within_prior_radius_ratio")
    direction_passed = (
        mask_direction is not None
        and outside_direction is not None
        and prior_direction is not None
        and float(mask_direction) >= 0.0
        and float(outside_direction) <= 0.0
        and float(prior_direction) >= 0.0
    )
    changed_ratio_passed = (
        changed_count >= min_changed_count
        and changed_ratio <= float(args.candidate_max_changed_ratio)
    )
    set_iou = float(set_compare.get("iou")) if set_compare.get("iou") is not None else None
    set_iou_passed = set_iou is not None and set_iou >= float(args.candidate_min_set_iou)
    baseline_sparse = baseline.get("sparse") or {}
    candidate_sparse = candidate.get("sparse") or {}
    baseline_component_count = int(baseline_sparse.get("component_count") or 0)
    candidate_component_count = int(candidate_sparse.get("component_count") or 0)
    component_count_delta = candidate_component_count - baseline_component_count
    baseline_coord_count = int(baseline_sparse.get("coord_count") or 0)
    candidate_coord_count = int(candidate_sparse.get("coord_count") or 0)
    coord_count_ratio = (
        float(candidate_coord_count) / float(baseline_coord_count)
        if baseline_coord_count > 0
        else None
    )
    structure_passed = bool(
        baseline_coord_count > 0
        and coord_count_ratio is not None
        and float(args.candidate_min_coord_count_ratio)
        <= coord_count_ratio
        <= float(args.candidate_max_coord_count_ratio)
        and component_count_delta <= int(args.candidate_max_component_increase)
    )
    absolute_outside_passed = (
        candidate_outside is not None
        and candidate_outside <= float(args.candidate_max_absolute_outside_ratio)
        and (
            not bool(args.candidate_require_absolute_outside_nonincrease)
            or absolute_outside_delta is not None
            and absolute_outside_delta <= 0.0
        )
    )
    passed = bool(
        direction_passed
        and changed_ratio_passed
        and set_iou_passed
        and absolute_outside_passed
        and structure_passed
    )
    return {
        "session": session_name,
        "split": split,
        "mask": mask_direction,
        "outside": outside_direction,
        "prior": prior_direction,
        "iou": set_iou,
        "baseline_count": baseline_count,
        "added_count": added_count,
        "removed_count": removed_count,
        "changed_count": changed_count,
        "changed_ratio": changed_ratio,
        "min_changed_count_required": min_changed_count,
        "baseline_absolute_outside": baseline_outside,
        "candidate_absolute_outside": candidate_outside,
        "absolute_outside_delta": absolute_outside_delta,
        "direction_passed": bool(direction_passed),
        "changed_ratio_passed": bool(changed_ratio_passed),
        "set_iou_passed": bool(set_iou_passed),
        "absolute_outside_passed": bool(absolute_outside_passed),
        "baseline_component_count": baseline_component_count,
        "candidate_component_count": candidate_component_count,
        "component_count_delta": component_count_delta,
        "baseline_coord_count": baseline_coord_count,
        "candidate_coord_count": candidate_coord_count,
        "coord_count_ratio": coord_count_ratio,
        "structure_passed": structure_passed,
        "passed": passed,
        "component_count": candidate_component_count,
    }


def _write_md(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# B5.4 Physical SS-Grid Feature Smoke",
        "",
        "## Scope",
        "",
        "```text",
        "No training.",
        "SS condition residual candidate search only.",
        "Spatial gates come from physical SS-grid features: mask/projection/prior/visual-hull support.",
        "Channel basis comes from normalized cond_base, not random channel direction.",
        "```",
        "",
        "## Aggregate",
        "",
        "```text",
    ]
    for row in report["aggregate"]:
        lines.append(
            f"{row['candidate']}: strict={row['strict_pass_session_count']}/{row['session_count']} "
            f"direction={row['good_direction_session_count']}/{row['session_count']} "
            f"mean_mask={row.get('mean_mask_direction')} mean_outside={row.get('mean_outside_direction')} "
            f"mean_prior={row.get('mean_prior_direction')} min_mask={row.get('min_mask_direction')} "
            f"max_outside={row.get('max_outside_direction')} max_changed={row.get('max_changed_ratio')} "
            f"max_absolute_outside_delta={row.get('max_absolute_outside_delta')} "
            f"max_component_delta={row.get('max_component_count_delta')}"
        )
    lines.extend(["```", "", "## Per Session", "", "```text"])
    for sess in report["sessions"]:
        lines.append(f"{sess['name']} split={sess['split']}")
        lines.append(f"  physical_sanity={sess['physical_features']['sanity']}")
        for cand in sess["candidates"]:
            direction = (cand.get("delta_vs_baseline") or {}).get("direction_summary") or {}
            lines.append(
                f"  {cand['name']}: mask={direction.get('added_minus_removed_projection_any_mask_hit_ratio')} "
                f"outside={direction.get('added_minus_removed_visible_outside_mask_event_ratio')} "
                f"prior={direction.get('added_minus_removed_within_prior_radius_ratio')} "
                f"iou={(cand.get('delta_vs_baseline') or {}).get('set_compare', {}).get('iou')} "
                f"absolute_outside={_metric_by_suffix(cand.get('prior_alignment'), '_visible_outside_mask_event_ratio')} "
                f"comp={cand['sparse'].get('component_count')}"
            )
        lines.append("")
    lines.extend(["```", "", "## Judgment", "", report["judgment"]])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="B5.4 physical SS-grid features and residual candidate search.")
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
    parser.add_argument(
        "--require_nonempty_surface_labels",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--projection_min_support_views", type=int, default=1)
    parser.add_argument("--projection_min_support_ratio", type=float, default=0.0)
    parser.add_argument("--visual_hull_min_visible_views", type=int, default=1)
    parser.add_argument("--visual_hull_min_support_ratio", type=float, default=0.0)
    parser.add_argument("--candidate_gates", default="surface_contrast")
    parser.add_argument(
        "--candidate_formulas",
        default="",
        help="Optional semicolon-separated formulas: name:gate=weight,gate2=weight. If set, candidate_gates is ignored.",
    )
    parser.add_argument("--candidate_scales", default="-0.01,-0.005,-0.0025,0.0025,0.005,0.01")
    parser.add_argument("--basis_modes", default="cond_unit")
    parser.add_argument(
        "--candidate_gate_normalization",
        choices=["none", "active_rms", "standardize"],
        default="none",
    )
    parser.add_argument("--candidate_allow_neutral_leakage", action="store_true")
    parser.add_argument(
        "--verify_fixed_noise_repeat",
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
    args = parser.parse_args()

    startup_mapping_audit = _token_grid_mapping_audit(
        int(args.ss_grid_side),
        int(args.sparse_resolution),
    )
    if not startup_mapping_audit["passed"]:
        raise RuntimeError(f"SS token mapping/round-trip audit failed before pipeline load: {startup_mapping_audit}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sessions_spec = json.loads(Path(args.sessions_json).read_text(encoding="utf-8"))
    if not args.load_dreamsim:
        install_dreamsim_stub()
    device = torch.device(args.device if torch.cuda.is_available() and args.device != "cpu" else "cpu")

    print(f"[B5.4] loading pipeline pretrained={args.pretrained} device={device}", flush=True)
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
    flow = pipeline.models["sparse_structure_flow_model"].to(device).eval()
    set_frozen_eval(flow)

    scales = _parse_float_list(args.candidate_scales)
    basis_modes = _parse_list(args.basis_modes)
    feature_names = PHYSICAL_FEATURE_NAMES
    gate_index = {name: idx for idx, name in enumerate(feature_names)}
    formula_specs = _parse_gate_formulas(args.candidate_formulas)
    if formula_specs:
        candidate_specs = formula_specs
    else:
        candidate_specs = [(gate_name, {gate_name: 1.0}) for gate_name in _parse_list(args.candidate_gates)]
    for formula_name, weights in candidate_specs:
        for gate_name in weights:
            if gate_name not in gate_index:
                raise KeyError(f"Unknown gate {gate_name!r} in formula {formula_name!r}; choices={sorted(gate_index)}")

    session_reports: list[dict[str, Any]] = []
    for sess_idx, spec in enumerate(sessions_spec):
        name = str(spec["name"])
        split = str(spec.get("split", "train"))
        print(f"[B5.4] session={name} split={split}", flush=True)
        cond_pack = _prepare_cond(pipeline=pipeline, args=args, spec=spec, device=device)
        prior_sample, prior_coords, prior_summary = _load_prior_manifest_sample(Path(spec["prior_manifest"]), str(spec.get("prior_uid", "") or ""))
        feature_sample, feature_scope_summary = _apply_physical_frame_scope(
            prior_sample,
            scope=str(args.physical_frame_scope),
            selected_image_names=cond_pack["image_names"],
        )
        eval_sample, eval_scope_summary = _apply_physical_frame_scope(
            prior_sample,
            scope=str(args.evaluation_frame_scope),
            selected_image_names=cond_pack["image_names"],
        )
        physical_features, physical_summary = _build_physical_features(
            sample=feature_sample,
            prior_coords=prior_coords,
            args=args,
        )
        physical_summary["feature_frame_scope"] = feature_scope_summary
        physical_summary["evaluation_frame_scope"] = eval_scope_summary

        torch.manual_seed(int(args.seed) + sess_idx)
        ss_noise = torch.randn(
            int(args.num_samples),
            int(flow.in_channels),
            int(flow.resolution),
            int(flow.resolution),
            int(flow.resolution),
            device=cond_pack["cond_base"].device,
        )
        baseline_delta = torch.zeros_like(cond_pack["cond_base"])
        baseline_coords, baseline_report = _evaluate_candidate(
            pipeline=pipeline,
            args=args,
            cond_base=cond_pack["cond_base"],
            neg_cond=cond_pack["neg_cond"],
            delta=baseline_delta,
            ss_noise=ss_noise,
            flow=flow,
            out_dir=output_dir / "eval" / name / "baseline",
            prior_sample=eval_sample,
            prior_coords=prior_coords,
        )
        fixed_noise_repeat = None
        if bool(args.verify_fixed_noise_repeat):
            repeat_coords, _ = _evaluate_candidate(
                pipeline=pipeline,
                args=args,
                cond_base=cond_pack["cond_base"],
                neg_cond=cond_pack["neg_cond"],
                delta=baseline_delta,
                ss_noise=ss_noise,
                flow=flow,
                out_dir=output_dir / "eval" / name / "baseline_repeat",
                prior_sample=eval_sample,
                prior_coords=prior_coords,
            )
            fixed_noise_repeat = {
                "passed": bool(np.array_equal(baseline_coords, repeat_coords)),
                "baseline_coord_count": int(baseline_coords.shape[0]),
                "repeat_coord_count": int(repeat_coords.shape[0]),
            }
            if not fixed_noise_repeat["passed"]:
                raise RuntimeError(f"fixed-noise baseline repeat failed for session={name}: {fixed_noise_repeat}")
        candidates: list[dict[str, Any]] = [
            {
                "name": "baseline",
                "gate": "none",
                "basis_mode": "none",
                "scale": 0.0,
                **baseline_report,
                "delta_vs_baseline": None,
            }
        ]
        for basis_mode in basis_modes:
            for gate_name, gate_weights in candidate_specs:
                for scale in scales:
                    cand_name = f"{basis_mode}_{gate_name}_{_scale_name(scale)}"
                    print(f"[B5.4] sampling session={name} candidate={cand_name}", flush=True)
                    delta, meta = _candidate_delta(
                        cond_base=cond_pack["cond_base"],
                        features=physical_features,
                        gate_name=gate_name,
                        gate_weights=gate_weights,
                        basis_mode=basis_mode,
                        scale=float(scale),
                        gate_index=gate_index,
                        gate_normalization=str(args.candidate_gate_normalization),
                        allow_neutral_leakage=bool(args.candidate_allow_neutral_leakage),
                    )
                    coords, cand_report = _evaluate_candidate(
                        pipeline=pipeline,
                        args=args,
                        cond_base=cond_pack["cond_base"],
                        neg_cond=cond_pack["neg_cond"],
                        delta=delta,
                        ss_noise=ss_noise,
                        flow=flow,
                        out_dir=output_dir / "eval" / name / cand_name,
                        prior_sample=eval_sample,
                        prior_coords=prior_coords,
                    )
                    cand_report.update(
                        {
                            "name": cand_name,
                            "gate": gate_name,
                            "basis_mode": basis_mode,
                            "scale": float(scale),
                            "delta_meta": meta,
                            "delta_vs_baseline": delta_report(
                                baseline_coords=baseline_coords,
                                candidate_coords=coords,
                                prior_sample=eval_sample,
                                prior_coords=prior_coords,
                                prior_radius=float(args.prior_radius),
                                projection_min_support_views=int(args.projection_min_support_views),
                                projection_min_support_ratio=float(args.projection_min_support_ratio),
                                visual_hull_min_visible_views=int(args.visual_hull_min_visible_views),
                                visual_hull_min_support_ratio=float(args.visual_hull_min_support_ratio),
                                mask_threshold=int(args.mask_threshold),
                            ),
                        }
                    )
                    candidates.append(cand_report)
                    (Path(cand_report["output_dir"]) / "report.json").write_text(
                        json.dumps(cand_report, indent=2, ensure_ascii=False, default=_json_default),
                        encoding="utf-8",
                    )
        session_reports.append(
            {
                "name": name,
                "split": split,
                "image_names": cond_pack["image_names"],
                "mask_summaries": cond_pack["mask_summaries"],
                "prior_summary": {
                    **prior_summary,
                    "feature_frame_scope": feature_scope_summary,
                    "evaluation_frame_scope": eval_scope_summary,
                },
                "physical_features": physical_summary,
                "cond_base": summarize_tensor(cond_pack["cond_base"]),
                "fixed_noise_repeat": fixed_noise_repeat,
                "candidates": candidates,
            }
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()

    aggregate: list[dict[str, Any]] = []
    names = sorted({cand["name"] for sess in session_reports for cand in sess["candidates"] if cand["name"] != "baseline"})
    for cand_name in names:
        rows = []
        for sess in session_reports:
            found = next((c for c in sess["candidates"] if c["name"] == cand_name), None)
            if found is None:
                continue
            baseline = next(c for c in sess["candidates"] if c["name"] == "baseline")
            rows.append(
                candidate_quality_row(
                    session_name=sess["name"],
                    split=sess["split"],
                    baseline=baseline,
                    candidate=found,
                    args=args,
                )
            )
        def vals(key: str) -> list[float]:
            return [float(r[key]) for r in rows if r.get(key) is not None]
        masks, outsides, priors, ious = vals("mask"), vals("outside"), vals("prior"), vals("iou")
        aggregate.append(
            {
                "candidate": cand_name,
                "session_count": len(rows),
                "mean_mask_direction": float(np.mean(masks)) if masks else None,
                "min_mask_direction": float(np.min(masks)) if masks else None,
                "mean_outside_direction": float(np.mean(outsides)) if outsides else None,
                "max_outside_direction": float(np.max(outsides)) if outsides else None,
                "mean_prior_direction": float(np.mean(priors)) if priors else None,
                "min_prior_direction": float(np.min(priors)) if priors else None,
                "mean_iou": float(np.mean(ious)) if ious else None,
                "mean_changed_ratio": float(np.mean([r["changed_ratio"] for r in rows])) if rows else None,
                "max_changed_ratio": float(np.max([r["changed_ratio"] for r in rows])) if rows else None,
                "mean_absolute_outside_delta": float(
                    np.mean([r["absolute_outside_delta"] for r in rows if r["absolute_outside_delta"] is not None])
                ) if any(r["absolute_outside_delta"] is not None for r in rows) else None,
                "max_absolute_outside_delta": float(
                    np.max([r["absolute_outside_delta"] for r in rows if r["absolute_outside_delta"] is not None])
                ) if any(r["absolute_outside_delta"] is not None for r in rows) else None,
                "max_component_count_delta": int(np.max([r["component_count_delta"] for r in rows])) if rows else None,
                "min_coord_count_ratio": float(np.min([r["coord_count_ratio"] for r in rows if r["coord_count_ratio"] is not None]))
                if any(r["coord_count_ratio"] is not None for r in rows) else None,
                "max_coord_count_ratio": float(np.max([r["coord_count_ratio"] for r in rows if r["coord_count_ratio"] is not None]))
                if any(r["coord_count_ratio"] is not None for r in rows) else None,
                "good_direction_session_count": int(sum(bool(r["direction_passed"]) for r in rows)),
                "strict_pass_session_count": int(sum(bool(r["passed"]) for r in rows)),
                "all_sessions_passed": bool(rows) and all(bool(r["passed"]) for r in rows),
                "per_session": rows,
            }
        )
    aggregate.sort(
        key=lambda r: (
            int(r.get("strict_pass_session_count") or 0),
            int(r.get("good_direction_session_count") or 0),
            float(r.get("mean_mask_direction") or -999.0),
            -float(r.get("max_outside_direction") or 999.0),
        ),
        reverse=True,
    )
    top = aggregate[0] if aggregate else {}
    judgment = (
        "B5.9 found a candidate that passes every train/validation session for this seed. Multi-seed aggregation is still required before B5.10."
        if top and bool(top.get("all_sessions_passed"))
        else "B5.9 produced candidates but none passed every strict check on every train/validation session for this seed."
        if top
        else "B5.4 produced no candidates."
    )
    if top and not bool(top.get("all_sessions_passed")):
        judgment += (
            " No candidate passes direction, changed-ratio, set-IoU, absolute-outside, component, and coord-count checks "
            "on every train/validation session, so do not train the downstream adapter yet."
        )
    report = {
        "args": vars(args),
        "scope": "B5.4 physical SS-grid encoding and eval-time residual candidate search; no training.",
        "startup_mapping_audit": startup_mapping_audit,
        "feature_names": feature_names,
        "sessions": session_reports,
        "aggregate": aggregate,
        "judgment": judgment,
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    _write_md(output_dir / "report.md", report)
    print(f"[B5.4] wrote {output_dir / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
