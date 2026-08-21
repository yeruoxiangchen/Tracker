from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Literal

import numpy as np
import torch
import torch.nn.functional as F

from trellis_point_prior_mv.common import apply_grid_transform, coords_to_points


LIFTING_CACHE_VERSION = "ar_ss_flow.pose_lifting_cache.v1"
LIFTING_VOLUME_VERSION = "ar_ss_flow.pose_lifting_volume.v2"
LIFTING_METADATA_NAMES = (
    "weighted_support",
    "mask_support_fraction",
    "visible_fraction",
    "depth_consistency",
    "depth_confidence",
    "depth_residual_normalized",
    "prior_occupancy",
    "prior_confidence",
    "prior_distance",
    "x",
    "y",
    "z",
)
SUPPORT_METADATA_INDEX = LIFTING_METADATA_NAMES.index("weighted_support")


def schema_hash(values: tuple[str, ...] = LIFTING_METADATA_NAMES) -> str:
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def canonical_voxel_points(side: int, grid_transform: str) -> np.ndarray:
    if side <= 0:
        raise ValueError(f"side must be positive, got {side}")
    axis = np.arange(side, dtype=np.float32)
    xyz = np.stack(
        np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1
    ).reshape(-1, 3)
    points = (xyz + 0.5) / float(side) - 0.5
    return apply_grid_transform(points, str(grid_transform)).astype(np.float32)


def scale_intrinsics(
    intrinsics: np.ndarray,
    *,
    source_width: int,
    source_height: int,
    target_width: int,
    target_height: int,
) -> np.ndarray:
    output = np.asarray(intrinsics, dtype=np.float32).copy()
    if output.ndim != 3 or output.shape[1:] != (3, 3):
        raise ValueError(f"intrinsics must be [V,3,3], got {output.shape}")
    output[:, 0, :] *= float(target_width) / float(source_width)
    output[:, 1, :] *= float(target_height) / float(source_height)
    output[:, 2, :] = np.asarray((0.0, 0.0, 1.0), dtype=np.float32)
    return output


def _project_points(
    points: np.ndarray,
    intrinsic: np.ndarray,
    extrinsic: np.ndarray,
    *,
    extrinsics_type: str,
    camera_forward_sign: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=np.float32)
    points_h = np.concatenate(
        (points, np.ones((len(points), 1), dtype=np.float32)), axis=1
    )
    if extrinsics_type == "c2w":
        w2c = np.linalg.inv(np.asarray(extrinsic, dtype=np.float64)).astype(np.float32)
    elif extrinsics_type == "w2c":
        w2c = np.asarray(extrinsic, dtype=np.float32)
    else:
        raise ValueError(f"unsupported extrinsics_type={extrinsics_type!r}")
    camera = (w2c @ points_h.T).T[:, :3]
    depth = camera[:, 2] * float(camera_forward_sign)
    safe = np.maximum(depth, 1.0e-6)
    u = float(intrinsic[0, 0]) * camera[:, 0] / safe + float(intrinsic[0, 2])
    v = float(intrinsic[1, 1]) * camera[:, 1] / safe + float(intrinsic[1, 2])
    return u.astype(np.float32), v.astype(np.float32), depth.astype(np.float32)


def _bilinear_numpy(image: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    height, width = image.shape[-2:]
    u = np.clip(np.asarray(u, dtype=np.float32), 0.0, float(width - 1))
    v = np.clip(np.asarray(v, dtype=np.float32), 0.0, float(height - 1))
    x0 = np.floor(u).astype(np.int64)
    y0 = np.floor(v).astype(np.int64)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    wx = u - x0
    wy = v - y0
    return (
        image[y0, x0] * (1.0 - wx) * (1.0 - wy)
        + image[y0, x1] * wx * (1.0 - wy)
        + image[y1, x0] * (1.0 - wx) * wy
        + image[y1, x1] * wx * wy
    ).astype(np.float32)


def collect_vggt_depth_matches(
    *,
    predicted_depth: np.ndarray,
    depth_confidence: np.ndarray,
    prior_coords: np.ndarray,
    prior_confidence: np.ndarray | None,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    grid_transform: str,
    extrinsics_type: str,
    camera_forward_sign: float,
    masks: np.ndarray | None = None,
    mask_threshold: float = 0.5,
    zbuffer_cell_size: int = 0,
    point_view_visibility: np.ndarray | None = None,
    object_to_world: np.ndarray | None = None,
) -> dict[str, Any]:
    """Collect deployable sparse-point/VGGT-depth correspondences.

    ``zbuffer_cell_size`` removes points hidden behind a nearer sparse point in
    the same image cell.  It is deliberately optional so existing cache builds
    retain their original behavior unless a stricter protocol enables it.
    """

    depth = np.asarray(predicted_depth, dtype=np.float32)
    confidence = np.asarray(depth_confidence, dtype=np.float32)
    coords = np.asarray(prior_coords)
    sparse_confidence = (
        np.ones(len(coords), dtype=np.float32)
        if prior_confidence is None
        else np.asarray(prior_confidence, dtype=np.float32)
    )
    k_all = np.asarray(intrinsics, dtype=np.float32)
    t_all = np.asarray(extrinsics, dtype=np.float32)
    mask_all = None if masks is None else np.asarray(masks, dtype=np.float32)
    visibility = (
        None
        if point_view_visibility is None
        else np.asarray(point_view_visibility, dtype=bool)
    )
    if depth.ndim != 3 or confidence.shape != depth.shape:
        raise ValueError(
            f"depth/confidence must be aligned [V,H,W], got {depth.shape}/{confidence.shape}"
        )
    if k_all.shape != (len(depth), 3, 3) or t_all.shape != (len(depth), 4, 4):
        raise ValueError("depth/K/T view count mismatch")
    if mask_all is not None and mask_all.shape != depth.shape:
        raise ValueError("masks must align with predicted depth")
    if coords.ndim != 2 or coords.shape[1] not in (3, 4):
        raise ValueError(f"prior_coords must be [N,3/4], got {coords.shape}")
    if sparse_confidence.ndim != 1 or len(sparse_confidence) != len(coords):
        raise ValueError("prior confidence must align with prior coordinates")
    if visibility is not None and visibility.shape != (len(depth), len(coords)):
        raise ValueError(
            "point_view_visibility must have shape "
            f"{(len(depth), len(coords))}, got {visibility.shape}"
        )
    if coords.size and ((coords[:, -3:] < 0).any() or (coords[:, -3:] >= 64).any()):
        raise ValueError("prior coordinates outside [0,63]")
    if int(zbuffer_cell_size) < 0:
        raise ValueError("zbuffer_cell_size must be nonnegative")
    if not 0.0 <= float(mask_threshold) <= 1.0:
        raise ValueError("mask_threshold must be in [0,1]")

    points = apply_grid_transform(
        coords_to_points(coords, 64), str(grid_transform)
    ).astype(np.float32)
    if object_to_world is not None:
        transform = np.asarray(object_to_world, dtype=np.float32)
        if transform.shape != (4, 4) or not np.isfinite(transform).all():
            raise ValueError("object_to_world must be a finite [4,4] matrix")
        points = points @ transform[:3, :3].T + transform[:3, 3]

    predicted_rows: list[np.ndarray] = []
    target_rows: list[np.ndarray] = []
    confidence_rows: list[np.ndarray] = []
    view_rows: list[np.ndarray] = []
    point_rows: list[np.ndarray] = []
    per_view: list[dict[str, Any]] = []
    height, width = depth.shape[-2:]
    cell_size = int(zbuffer_cell_size)
    cells_per_row = int(math.ceil(float(width) / max(float(cell_size), 1.0)))

    for view_index in range(len(depth)):
        u, v, camera_depth = _project_points(
            points,
            k_all[view_index],
            t_all[view_index],
            extrinsics_type=extrinsics_type,
            camera_forward_sign=camera_forward_sign,
        )
        valid = (
            (camera_depth > 1.0e-5)
            & (u >= 0.0)
            & (u <= float(width - 1))
            & (v >= 0.0)
            & (v <= float(height - 1))
        )
        if visibility is not None:
            valid &= visibility[view_index]
        ids = np.nonzero(valid)[0]
        projected_count = int(len(ids))
        sampled_depth = _bilinear_numpy(depth[view_index], u[ids], v[ids])
        sampled_conf = _bilinear_numpy(confidence[view_index], u[ids], v[ids])
        sampled_mask = (
            np.ones(len(ids), dtype=np.float32)
            if mask_all is None
            else _bilinear_numpy(mask_all[view_index], u[ids], v[ids])
        )
        finite = (
            np.isfinite(sampled_depth)
            & np.isfinite(sampled_conf)
            & np.isfinite(sampled_mask)
            & (sampled_depth > 1.0e-6)
            & (sampled_mask >= float(mask_threshold))
            & np.isfinite(camera_depth[ids])
            & np.isfinite(sparse_confidence[ids])
        )
        ids = ids[finite]
        sampled_depth = sampled_depth[finite]
        sampled_conf = sampled_conf[finite]
        masked_match_count = int(len(ids))
        if cell_size > 0 and len(ids):
            cell_x = np.floor(u[ids] / float(cell_size)).astype(np.int64)
            cell_y = np.floor(v[ids] / float(cell_size)).astype(np.int64)
            cell_id = cell_y * cells_per_row + cell_x
            depth_order = np.argsort(camera_depth[ids], kind="stable")
            _, first = np.unique(cell_id[depth_order], return_index=True)
            keep = depth_order[np.sort(first)]
            ids = ids[keep]
            sampled_depth = sampled_depth[keep]
            sampled_conf = sampled_conf[keep]
        camera_values = camera_depth[ids]
        sparse_values = sparse_confidence[ids]
        if len(sampled_depth):
            positive_conf = sampled_conf[sampled_conf > 0]
            conf_scale = (
                float(np.median(positive_conf)) if len(positive_conf) else 1.0
            )
            normalized_conf = np.clip(
                sampled_conf / max(conf_scale, 1.0e-6), 0.05, 4.0
            )
            normalized_conf *= np.clip(sparse_values, 0.05, 1.0)
            predicted_rows.append(sampled_depth)
            target_rows.append(camera_values)
            confidence_rows.append(normalized_conf)
            view_rows.append(np.full(len(sampled_depth), view_index, dtype=np.int32))
            point_rows.append(ids.astype(np.int64, copy=False))
        per_view.append(
            {
                "view_index": view_index,
                "projected_count": projected_count,
                "masked_match_count": masked_match_count,
                "match_count": int(len(sampled_depth)),
            }
        )

    if predicted_rows:
        predicted_all = np.concatenate(predicted_rows)
        target_all = np.concatenate(target_rows)
        confidence_all = np.concatenate(confidence_rows)
        view_all = np.concatenate(view_rows)
        point_all = np.concatenate(point_rows)
    else:
        predicted_all = np.empty((0,), dtype=np.float32)
        target_all = np.empty((0,), dtype=np.float32)
        confidence_all = np.empty((0,), dtype=np.float32)
        view_all = np.empty((0,), dtype=np.int32)
        point_all = np.empty((0,), dtype=np.int64)
    return {
        "predicted": predicted_all,
        "target": target_all,
        "confidence": confidence_all,
        "view_index": view_all,
        "point_index": point_all,
        "per_view": per_view,
        "match_count": int(len(predicted_all)),
    }


def _weighted_lstsq(
    design: np.ndarray,
    target: np.ndarray,
    base_weight: np.ndarray,
    *,
    iterations: int = 12,
) -> np.ndarray:
    weight = np.maximum(np.asarray(base_weight, dtype=np.float64), 1.0e-6)
    design64 = np.asarray(design, dtype=np.float64)
    target64 = np.asarray(target, dtype=np.float64)
    solution = np.linalg.lstsq(
        design64 * np.sqrt(weight[:, None]),
        target64 * np.sqrt(weight),
        rcond=None,
    )[0]
    for _ in range(iterations):
        residual = target64 - design64 @ solution
        median = np.median(residual)
        mad = np.median(np.abs(residual - median))
        scale = max(1.4826 * float(mad), 1.0e-5)
        normalized = np.abs(residual) / (1.345 * scale)
        huber = np.ones_like(normalized)
        large = normalized > 1.0
        huber[large] = 1.0 / normalized[large]
        robust_weight = weight * huber
        solution = np.linalg.lstsq(
            design64 * np.sqrt(robust_weight[:, None]),
            target64 * np.sqrt(robust_weight),
            rcond=None,
        )[0]
    return solution.astype(np.float64)


def _fit_depth_model(
    predicted: np.ndarray,
    target: np.ndarray,
    confidence: np.ndarray,
    *,
    affine: bool,
) -> dict[str, Any]:
    predicted = np.asarray(predicted, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    confidence = np.asarray(confidence, dtype=np.float64)
    if affine:
        design = np.stack((predicted, np.ones_like(predicted)), axis=1)
    else:
        design = predicted[:, None]
    solution = _weighted_lstsq(design, target, confidence)
    scale = float(solution[0])
    shift = float(solution[1]) if affine else 0.0
    calibrated = scale * predicted + shift
    residual = np.abs(calibrated - target)
    return {
        "model": "affine" if affine else "scale_only",
        "scale": scale,
        "shift": shift,
        "median_abs_residual": float(np.median(residual)),
        "p90_abs_residual": float(np.quantile(residual, 0.9)),
        "mean_abs_residual": float(np.mean(residual)),
        "positive_scale": bool(np.isfinite(scale) and scale > 0.0),
    }


def calibrate_vggt_depth(
    *,
    predicted_depth: np.ndarray,
    depth_confidence: np.ndarray,
    prior_coords: np.ndarray,
    prior_confidence: np.ndarray | None = None,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    grid_transform: str,
    extrinsics_type: str,
    camera_forward_sign: float,
    min_matches: int = 8,
    affine_improvement_ratio: float = 0.90,
    masks: np.ndarray | None = None,
    mask_threshold: float = 0.5,
    zbuffer_cell_size: int = 0,
    point_view_visibility: np.ndarray | None = None,
    object_to_world: np.ndarray | None = None,
    force_scale_only: bool = False,
) -> dict[str, Any]:
    """Align VGGT depth to camera depth using only deployable sparse points."""

    matches = collect_vggt_depth_matches(
        predicted_depth=predicted_depth,
        depth_confidence=depth_confidence,
        prior_coords=prior_coords,
        prior_confidence=prior_confidence,
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        grid_transform=grid_transform,
        extrinsics_type=extrinsics_type,
        camera_forward_sign=camera_forward_sign,
        masks=masks,
        mask_threshold=mask_threshold,
        zbuffer_cell_size=zbuffer_cell_size,
        point_view_visibility=point_view_visibility,
        object_to_world=object_to_world,
    )
    match_count = int(matches["match_count"])
    per_view = matches["per_view"]
    report: dict[str, Any] = {
        "enabled": False,
        "fallback": "mask_visual_hull_only",
        "match_count": match_count,
        "min_matches": int(min_matches),
        "per_view": per_view,
        "scale_only": None,
        "affine": None,
        "selected_model": None,
        "scale": None,
        "shift": None,
        "median_abs_residual": None,
        "p90_abs_residual": None,
    }
    if match_count < int(min_matches):
        report["failure_reason"] = "insufficient_sparse_depth_matches"
        return report

    predicted_all = matches["predicted"]
    target_all = matches["target"]
    confidence_all = matches["confidence"]
    view_all = matches["view_index"]
    scale_fit = _fit_depth_model(
        predicted_all, target_all, confidence_all, affine=False
    )
    affine_fit = _fit_depth_model(
        predicted_all, target_all, confidence_all, affine=True
    )
    report["scale_only"] = scale_fit
    report["affine"] = affine_fit
    if bool(force_scale_only) and not scale_fit["positive_scale"]:
        report["failure_reason"] = "invalid_scale_only_depth_fit"
        return report
    if not scale_fit["positive_scale"] and not affine_fit["positive_scale"]:
        report["failure_reason"] = "nonpositive_or_nonfinite_depth_scale"
        return report
    selected = scale_fit if scale_fit["positive_scale"] else affine_fit
    if (
        not bool(force_scale_only)
        and affine_fit["positive_scale"]
        and match_count >= max(int(min_matches), 12)
        and affine_fit["median_abs_residual"]
        < float(affine_improvement_ratio) * scale_fit["median_abs_residual"]
    ):
        selected = affine_fit
    report.update(
        {
            "enabled": True,
            "fallback": None,
            "selected_model": selected["model"],
            "scale": selected["scale"],
            "shift": selected["shift"],
            "median_abs_residual": selected["median_abs_residual"],
            "p90_abs_residual": selected["p90_abs_residual"],
        }
    )
    calibrated = selected["scale"] * predicted_all + selected["shift"]
    residual = np.abs(calibrated - target_all)
    for row in per_view:
        selected_view = view_all == row["view_index"]
        if selected_view.any():
            row["median_abs_residual"] = float(np.median(residual[selected_view]))
            row["p90_abs_residual"] = float(np.quantile(residual[selected_view], 0.9))
        else:
            row["median_abs_residual"] = None
            row["p90_abs_residual"] = None
    return report


def evaluate_vggt_depth_calibration(
    *,
    calibration: dict[str, Any],
    predicted_depth: np.ndarray,
    depth_confidence: np.ndarray,
    prior_coords: np.ndarray,
    prior_confidence: np.ndarray | None,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    grid_transform: str,
    extrinsics_type: str,
    camera_forward_sign: float,
    masks: np.ndarray | None = None,
    mask_threshold: float = 0.5,
    zbuffer_cell_size: int = 0,
    point_view_visibility: np.ndarray | None = None,
    object_to_world: np.ndarray | None = None,
    tolerance: float = 0.15,
) -> dict[str, Any]:
    """Evaluate a frozen calibration without fitting on the evaluation points."""

    matches = collect_vggt_depth_matches(
        predicted_depth=predicted_depth,
        depth_confidence=depth_confidence,
        prior_coords=prior_coords,
        prior_confidence=prior_confidence,
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        grid_transform=grid_transform,
        extrinsics_type=extrinsics_type,
        camera_forward_sign=camera_forward_sign,
        masks=masks,
        mask_threshold=mask_threshold,
        zbuffer_cell_size=zbuffer_cell_size,
        point_view_visibility=point_view_visibility,
        object_to_world=object_to_world,
    )
    report: dict[str, Any] = {
        "match_count": int(matches["match_count"]),
        "per_view": matches["per_view"],
        "median_abs_residual": None,
        "p90_abs_residual": None,
        "mean_abs_residual": None,
        "within_tolerance_ratio": 0.0,
        "point_behind_surface_ratio": 0.0,
        "point_in_front_of_surface_ratio": 0.0,
    }
    if not calibration.get("enabled", False) or not matches["match_count"]:
        return report
    scale = float(calibration["scale"])
    shift = float(calibration["shift"])
    aligned = scale * matches["predicted"] + shift
    signed = matches["target"] - aligned
    residual = np.abs(signed)
    threshold = max(float(tolerance), 1.0e-6)
    report.update(
        {
            "median_abs_residual": float(np.median(residual)),
            "p90_abs_residual": float(np.quantile(residual, 0.9)),
            "mean_abs_residual": float(np.mean(residual)),
            "within_tolerance_ratio": float(np.mean(residual <= threshold)),
            "point_behind_surface_ratio": float(np.mean(signed > threshold)),
            "point_in_front_of_surface_ratio": float(np.mean(signed < -threshold)),
        }
    )
    for row in report["per_view"]:
        selected = matches["view_index"] == row["view_index"]
        if selected.any():
            current = residual[selected]
            row["median_abs_residual"] = float(np.median(current))
            row["p90_abs_residual"] = float(np.quantile(current, 0.9))
        else:
            row["median_abs_residual"] = None
            row["p90_abs_residual"] = None
    return report


def perturb_extrinsics(
    extrinsics: torch.Tensor,
    *,
    mode: Literal["correct", "pose_perturb", "pose_shuffle"],
    extrinsics_type: str,
    rotation_degrees: float = 3.0,
    translation: float = 0.02,
) -> torch.Tensor:
    if mode == "correct":
        return extrinsics
    if mode == "pose_shuffle":
        if extrinsics.shape[0] < 2:
            return extrinsics
        return torch.roll(extrinsics, shifts=1, dims=0)
    if mode != "pose_perturb":
        raise ValueError(f"unsupported pose mode={mode!r}")
    if extrinsics_type not in {"c2w", "w2c"}:
        raise ValueError(f"unsupported extrinsics_type={extrinsics_type!r}")
    c2w = extrinsics if extrinsics_type == "c2w" else torch.linalg.inv(extrinsics)
    output = c2w.clone()
    angle = math.radians(float(rotation_degrees))
    for view_index in range(int(c2w.shape[0])):
        sign = -1.0 if view_index % 2 else 1.0
        cosine = math.cos(sign * angle)
        sine = math.sin(sign * angle)
        rotation = torch.tensor(
            ((cosine, 0.0, sine), (0.0, 1.0, 0.0), (-sine, 0.0, cosine)),
            device=extrinsics.device,
            dtype=extrinsics.dtype,
        )
        output[view_index, :3, :3] = c2w[view_index, :3, :3] @ rotation
        local_shift = torch.tensor(
            (sign * float(translation), 0.5 * sign * float(translation), 0.0),
            device=extrinsics.device,
            dtype=extrinsics.dtype,
        )
        output[view_index, :3, 3] = (
            c2w[view_index, :3, 3]
            + c2w[view_index, :3, :3] @ local_shift
        )
    return output if extrinsics_type == "c2w" else torch.linalg.inv(output)


def build_projection_geometry(
    *,
    intrinsics: torch.Tensor,
    extrinsics: torch.Tensor,
    grid_transform: str,
    extrinsics_type: str,
    camera_forward_sign: float,
    image_height: int,
    image_width: int,
    patch_grid_side: int,
    volume_side: int = 16,
    object_to_world: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    device = intrinsics.device
    dtype = torch.float32
    points = torch.from_numpy(
        canonical_voxel_points(volume_side, grid_transform)
    ).to(device=device, dtype=dtype)
    if object_to_world is not None:
        transform = object_to_world.to(device=device, dtype=dtype)
        if transform.shape != (4, 4):
            raise ValueError(
                f"object_to_world must be [4,4], got {tuple(transform.shape)}"
            )
        points = (
            points @ transform[:3, :3].transpose(0, 1)
            + transform[:3, 3]
        )
    views = int(intrinsics.shape[0])
    points_h = torch.cat(
        (points, torch.ones((len(points), 1), device=device, dtype=dtype)), dim=1
    )
    w2c = torch.linalg.inv(extrinsics.float()) if extrinsics_type == "c2w" else extrinsics.float()
    camera = torch.einsum("vij,nj->vni", w2c, points_h)[..., :3]
    depth = camera[..., 2] * float(camera_forward_sign)
    safe = depth.clamp_min(1.0e-6)
    u = intrinsics[:, None, 0, 0].float() * camera[..., 0] / safe + intrinsics[:, None, 0, 2].float()
    v = intrinsics[:, None, 1, 1].float() * camera[..., 1] / safe + intrinsics[:, None, 1, 2].float()
    valid = (
        (depth > 1.0e-5)
        & (u >= 0.0)
        & (u <= float(image_width - 1))
        & (v >= 0.0)
        & (v <= float(image_height - 1))
    )
    image_grid = torch.stack(
        (
            2.0 * u / max(float(image_width - 1), 1.0) - 1.0,
            2.0 * v / max(float(image_height - 1), 1.0) - 1.0,
        ),
        dim=-1,
    )
    patch_u = (u + 0.5) / (float(image_width) / float(patch_grid_side)) - 0.5
    patch_v = (v + 0.5) / (float(image_height) / float(patch_grid_side)) - 0.5
    patch_grid = torch.stack(
        (
            2.0 * patch_u / max(float(patch_grid_side - 1), 1.0) - 1.0,
            2.0 * patch_v / max(float(patch_grid_side - 1), 1.0) - 1.0,
        ),
        dim=-1,
    )
    if image_grid.shape != (views, volume_side**3, 2):
        raise RuntimeError(f"unexpected projection grid shape {tuple(image_grid.shape)}")
    return {
        "image_grid": image_grid,
        "patch_grid": patch_grid,
        "camera_depth": depth,
        "valid": valid,
    }


def projection_roundtrip_audit(
    *,
    intrinsics: torch.Tensor,
    extrinsics: torch.Tensor,
    grid_transform: str,
    extrinsics_type: str,
    camera_forward_sign: float,
    image_height: int,
    image_width: int,
    patch_grid_side: int,
    volume_side: int = 16,
    object_to_world: torch.Tensor | None = None,
) -> dict[str, float | int]:
    geometry = build_projection_geometry(
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        grid_transform=grid_transform,
        extrinsics_type=extrinsics_type,
        camera_forward_sign=camera_forward_sign,
        image_height=image_height,
        image_width=image_width,
        patch_grid_side=patch_grid_side,
        volume_side=volume_side,
        object_to_world=object_to_world,
    )
    device = intrinsics.device
    points = torch.from_numpy(
        canonical_voxel_points(volume_side, grid_transform)
    ).to(device=device, dtype=torch.float32)
    if object_to_world is not None:
        transform = object_to_world.to(device=device, dtype=torch.float32)
        points = (
            points @ transform[:3, :3].transpose(0, 1)
            + transform[:3, 3]
        )
    grid = geometry["image_grid"].float()
    u = (grid[..., 0] + 1.0) * 0.5 * max(float(image_width - 1), 1.0)
    v = (grid[..., 1] + 1.0) * 0.5 * max(float(image_height - 1), 1.0)
    depth = geometry["camera_depth"].float()
    camera_z = depth / float(camera_forward_sign)
    camera_x = (u - intrinsics[:, None, 0, 2].float()) / intrinsics[
        :, None, 0, 0
    ].float() * depth
    camera_y = (v - intrinsics[:, None, 1, 2].float()) / intrinsics[
        :, None, 1, 1
    ].float() * depth
    camera = torch.stack((camera_x, camera_y, camera_z), dim=-1)
    camera_h = torch.cat(
        (
            camera,
            torch.ones((*camera.shape[:-1], 1), device=device, dtype=torch.float32),
        ),
        dim=-1,
    )
    c2w = extrinsics.float() if extrinsics_type == "c2w" else torch.linalg.inv(extrinsics.float())
    recovered = torch.einsum("vij,vnj->vni", c2w, camera_h)[..., :3]
    error = torch.linalg.vector_norm(recovered - points[None], dim=-1)
    valid_error = error[geometry["valid"]]
    if not valid_error.numel():
        return {"valid_projection_count": 0, "mean_error": float("inf"), "max_error": float("inf")}
    return {
        "valid_projection_count": int(valid_error.numel()),
        "mean_error": float(valid_error.mean().item()),
        "max_error": float(valid_error.max().item()),
    }


def _sample_maps(values: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    sampled = F.grid_sample(
        values,
        grid[:, :, None, :],
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    return sampled[..., 0]


def _prior_volume_features(
    prior_coords: torch.Tensor,
    prior_confidence: torch.Tensor,
    *,
    device: torch.device,
    volume_side: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    occupancy = torch.zeros((1, volume_side, volume_side, volume_side), device=device)
    confidence = torch.zeros_like(occupancy)
    xyz64 = prior_coords[..., -3:].to(device=device, dtype=torch.long)
    valid = ((xyz64 >= 0) & (xyz64 < 64)).all(dim=1)
    xyz64 = xyz64[valid]
    conf = prior_confidence.to(device=device, dtype=torch.float32)[valid]
    if xyz64.numel():
        xyz16 = torch.div(xyz64 * volume_side, 64, rounding_mode="floor").clamp(0, volume_side - 1)
        flat_index = (
            xyz16[:, 0] * volume_side * volume_side
            + xyz16[:, 1] * volume_side
            + xyz16[:, 2]
        )
        occupancy.view(-1)[flat_index] = 1.0
        confidence.view(-1).scatter_reduce_(
            0, flat_index, conf, reduce="amax", include_self=True
        )
        axis = torch.arange(volume_side, device=device, dtype=torch.float32)
        centers = torch.stack(
            torch.meshgrid(axis, axis, axis, indexing="ij"), dim=-1
        ).reshape(-1, 3)
        distance = torch.cdist(centers, xyz16.float()).amin(dim=1)
        distance = distance / max(math.sqrt(3.0) * float(volume_side - 1), 1.0)
        distance = distance.reshape(1, volume_side, volume_side, volume_side)
    else:
        distance = torch.ones_like(occupancy)
    return occupancy, confidence, distance.clamp(0.0, 1.0)


def build_lifting_volume(
    *,
    visual_patch_features: torch.Tensor,
    predicted_depth: torch.Tensor,
    depth_confidence: torch.Tensor,
    masks: torch.Tensor,
    intrinsics: torch.Tensor,
    extrinsics: torch.Tensor,
    prior_coords: torch.Tensor,
    prior_confidence: torch.Tensor,
    calibration: dict[str, Any],
    grid_transform: str,
    extrinsics_type: str,
    camera_forward_sign: float,
    pose_mode: Literal["correct", "pose_perturb", "pose_shuffle", "depth_corrupt"] = "correct",
    cached_correct_geometry: dict[str, torch.Tensor] | None = None,
    volume_side: int = 16,
    depth_corruption_scale: float = 1.15,
    compute_cross_view_metrics: bool = False,
    object_to_world: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Lift one reusable visual cache into a paired 16^3 volume."""

    if visual_patch_features.ndim != 3:
        raise ValueError(
            "visual_patch_features must be [V,P,C], "
            f"got {tuple(visual_patch_features.shape)}"
        )
    views, patch_count, channels = visual_patch_features.shape
    patch_side = int(round(math.sqrt(int(patch_count))))
    if patch_side * patch_side != int(patch_count):
        raise ValueError(f"patch count is not square: {patch_count}")
    if predicted_depth.ndim != 3 or depth_confidence.shape != predicted_depth.shape:
        raise ValueError("depth/confidence must be [V,H,W] and aligned")
    if masks.shape != predicted_depth.shape or int(masks.shape[0]) != int(views):
        raise ValueError("mask/depth/visual view count mismatch")
    device = visual_patch_features.device
    image_height, image_width = map(int, predicted_depth.shape[-2:])
    geometry_mode = "correct" if pose_mode == "depth_corrupt" else pose_mode
    use_cached = geometry_mode == "correct" and cached_correct_geometry is not None
    if use_cached:
        geometry = {
            key: value.to(device=device)
            for key, value in cached_correct_geometry.items()
        }
    else:
        variant_extrinsics = perturb_extrinsics(
            extrinsics,
            mode=geometry_mode,
            extrinsics_type=extrinsics_type,
        )
        geometry = build_projection_geometry(
            intrinsics=intrinsics,
            extrinsics=variant_extrinsics,
            grid_transform=grid_transform,
            extrinsics_type=extrinsics_type,
            camera_forward_sign=camera_forward_sign,
            image_height=image_height,
            image_width=image_width,
            patch_grid_side=patch_side,
            volume_side=volume_side,
            object_to_world=object_to_world,
        )
    patch_maps = visual_patch_features.permute(0, 2, 1).reshape(
        views, channels, patch_side, patch_side
    )
    sampled_visual = _sample_maps(patch_maps.float(), geometry["patch_grid"].float())
    depth_maps = predicted_depth[:, None].float()
    if pose_mode == "depth_corrupt":
        depth_maps = torch.flip(
            torch.roll(
                depth_maps,
                shifts=(max(1, image_height // 6), max(1, image_width // 5)),
                dims=(-2, -1),
            ),
            dims=(-1,),
        )
        view_scale = torch.linspace(
            0.80,
            1.20,
            int(depth_maps.shape[0]),
            device=depth_maps.device,
            dtype=depth_maps.dtype,
        ).reshape(-1, 1, 1, 1)
        depth_maps = depth_maps * view_scale * float(depth_corruption_scale)
    sampled_depth = _sample_maps(depth_maps, geometry["image_grid"].float())[:, 0]
    sampled_confidence = _sample_maps(
        depth_confidence[:, None].float(), geometry["image_grid"].float()
    )[:, 0]
    sampled_mask = _sample_maps(
        masks[:, None].float(), geometry["image_grid"].float()
    )[:, 0].clamp(0.0, 1.0)
    valid = geometry["valid"].float()
    visible = valid
    mask_weight = sampled_mask * valid
    positive_conf = sampled_confidence[sampled_confidence > 0]
    conf_scale = (
        positive_conf.median().clamp_min(1.0e-6)
        if positive_conf.numel()
        else torch.tensor(1.0, device=device)
    )
    confidence_weight = (sampled_confidence / conf_scale).clamp(0.0, 1.0)

    depth_enabled = bool(calibration.get("enabled", False))
    if depth_enabled:
        scale = float(calibration["scale"])
        shift = float(calibration["shift"])
        aligned_depth = sampled_depth * scale + shift
        residual = (aligned_depth - geometry["camera_depth"].float()).abs()
        tolerance = max(
            float(calibration.get("p90_abs_residual") or 0.0),
            float(calibration.get("minimum_depth_tolerance", 0.02)),
        )
        depth_weight = torch.exp(-0.5 * (residual / tolerance).square()) * valid
        normalized_residual = (residual / tolerance).clamp(0.0, 4.0) * 0.25
        depth_consistency_metadata = depth_weight
    else:
        depth_weight = valid
        normalized_residual = torch.zeros_like(valid)
        depth_consistency_metadata = torch.zeros_like(valid)

    weight = mask_weight * confidence_weight * depth_weight
    weight_sum = weight.sum(dim=0).clamp_min(1.0e-6)
    volume = (sampled_visual * weight[:, None]).sum(dim=0) / weight_sum[None]
    supported = (weight.sum(dim=0) > 1.0e-6).float()
    volume = volume * supported[None]

    occupancy, prior_conf, prior_distance = _prior_volume_features(
        prior_coords,
        prior_confidence,
        device=device,
        volume_side=volume_side,
    )
    axis = (torch.arange(volume_side, device=device, dtype=torch.float32) + 0.5)
    axis = axis / float(volume_side) * 2.0 - 1.0
    xyz = torch.stack(
        torch.meshgrid(axis, axis, axis, indexing="ij"), dim=0
    ).reshape(3, -1)
    view_denominator = max(float(views), 1.0)
    metadata_flat = torch.cat(
        (
            (weight.sum(dim=0) / view_denominator).clamp(0.0, 1.0)[None],
            (mask_weight.sum(dim=0) / view_denominator).clamp(0.0, 1.0)[None],
            (visible.sum(dim=0) / view_denominator).clamp(0.0, 1.0)[None],
            (
                (depth_consistency_metadata * valid).sum(dim=0)
                / valid.sum(dim=0).clamp_min(1.0)
            )[None],
            ((confidence_weight * valid).sum(dim=0) / valid.sum(dim=0).clamp_min(1.0))[None],
            ((normalized_residual * valid).sum(dim=0) / valid.sum(dim=0).clamp_min(1.0))[None],
            occupancy.reshape(1, -1),
            prior_conf.reshape(1, -1),
            prior_distance.reshape(1, -1),
            xyz,
        ),
        dim=0,
    )
    volume = volume.reshape(channels, volume_side, volume_side, volume_side)
    metadata = metadata_flat.reshape(
        len(LIFTING_METADATA_NAMES), volume_side, volume_side, volume_side
    )
    if not bool(torch.isfinite(volume).all().item()) or not bool(
        torch.isfinite(metadata).all().item()
    ):
        raise RuntimeError("lifting produced non-finite values")
    stats = {
        "pose_mode": pose_mode,
        "depth_consistency_enabled": depth_enabled,
        "depth_fallback": calibration.get("fallback"),
        "supported_voxel_ratio": float(supported.mean().item()),
        "mean_weighted_support": float(metadata[SUPPORT_METADATA_INDEX].mean().item()),
        "visual_rms": float(volume.float().square().mean().sqrt().item()),
    }
    if compute_cross_view_metrics:
        multi_view = weight.gt(1.0e-6).sum(dim=0) >= 2
        if bool(multi_view.any().item()):
            residual_energy = (
                (sampled_visual - volume.reshape(channels, -1)[None])
                .square()
                .mean(dim=1)
            )
            weighted_variance = (
                (residual_energy * weight).sum(dim=0) / weight_sum
            )[multi_view]
            normalized = F.normalize(sampled_visual, dim=1, eps=1.0e-6)
            mean_direction = (
                (normalized * weight[:, None]).sum(dim=0) / weight_sum[None]
            )
            cosine_consistency = mean_direction.square().sum(dim=0).sqrt()[multi_view]
            stats.update(
                {
                    "cross_view_voxel_count": int(multi_view.sum().item()),
                    "cross_view_weighted_variance": float(
                        weighted_variance.mean().item()
                    ),
                    "cross_view_cosine_consistency": float(
                        cosine_consistency.mean().item()
                    ),
                }
            )
        else:
            stats.update(
                {
                    "cross_view_voxel_count": 0,
                    "cross_view_weighted_variance": None,
                    "cross_view_cosine_consistency": None,
                }
            )
    return volume, metadata, stats


def cache_config_hash(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
