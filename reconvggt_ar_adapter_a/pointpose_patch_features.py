from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image
import torch

from trellis_point_prior_mv.common import apply_grid_transform, coords_to_points


PROJECTED_PATCH_FEATURE_NAMES = (
    "projected_occupancy",
    "projected_log_count",
    "projected_confidence_max",
    "projected_confidence_mean",
    "projected_nearest_depth",
    "projected_depth_mean",
    "projected_depth_std",
    "projected_mask_hit_ratio",
    "outside_projected_ratio",
    "mask_patch_fraction",
    "ray_direction_x",
    "ray_direction_y",
    "ray_direction_z",
    "camera_origin_direction_x",
    "camera_origin_direction_y",
    "camera_origin_direction_z",
    "camera_radius",
    "patch_u",
    "patch_v",
    "view_position",
)
PROJECTED_PATCH_EVIDENCE_COUNT = PROJECTED_PATCH_FEATURE_NAMES.index(
    "ray_direction_x"
)
PROJECTED_PATCH_FEATURE_VERSION = "reconvggt.pointpose_projected_patch.v1"


def projected_patch_feature_schema_hash() -> str:
    payload = "\n".join(PROJECTED_PATCH_FEATURE_NAMES).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def make_null_projected_patch_features(features: torch.Tensor) -> torch.Tensor:
    """Remove point/mask evidence while retaining patch rays and camera pose."""

    if features.ndim != 3 or features.shape[-1] != len(PROJECTED_PATCH_FEATURE_NAMES):
        raise ValueError(
            "projected patch features must be [B,L,F], "
            f"got {tuple(features.shape)}"
        )
    null = torch.zeros_like(features)
    null[..., PROJECTED_PATCH_EVIDENCE_COUNT:] = features[
        ..., PROJECTED_PATCH_EVIDENCE_COUNT:
    ]
    return null


def infer_patch_grid_side(aggregated_tokens: Sequence[torch.Tensor]) -> int:
    if not aggregated_tokens:
        raise ValueError("aggregated token list is empty")
    token = aggregated_tokens[0]
    if token.ndim != 4 or token.shape[2] <= 5:
        raise ValueError(f"unexpected aggregated token shape {tuple(token.shape)}")
    patch_count = int(token.shape[2]) - 5
    side = int(round(math.sqrt(patch_count)))
    if side * side != patch_count:
        raise ValueError(f"VGGT patch count is not square: {patch_count}")
    return side


def _load_masks(mask_paths: Sequence[str | Path]) -> list[np.ndarray]:
    masks: list[np.ndarray] = []
    for path in mask_paths:
        with Image.open(path) as image:
            mask = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
        if mask.ndim != 2 or not np.isfinite(mask).all():
            raise ValueError(f"invalid mask {path}: shape={mask.shape}")
        masks.append(mask)
    if not masks:
        raise ValueError("projected patch features require at least one mask")
    return masks


def _patch_mask_fraction(mask: np.ndarray, side: int) -> np.ndarray:
    image = Image.fromarray(mask.astype(np.float32), mode="F")
    resized = image.resize((side, side), Image.Resampling.BOX)
    result = np.asarray(resized, dtype=np.float32)
    return np.clip(result, 0.0, 1.0)


def _camera_geometry(
    intrinsic: np.ndarray,
    extrinsic: np.ndarray,
    *,
    width: int,
    height: int,
    side: int,
    extrinsics_type: str,
    camera_forward_sign: float,
    view_index: int,
    view_count: int,
) -> np.ndarray:
    c2w = extrinsic if extrinsics_type == "c2w" else np.linalg.inv(extrinsic)
    rotation = c2w[:3, :3].astype(np.float32)
    origin = c2w[:3, 3].astype(np.float32)
    radius = max(float(np.linalg.norm(origin)), 1.0e-6)
    origin_direction = origin / radius

    py, px = np.meshgrid(
        np.arange(side, dtype=np.float32),
        np.arange(side, dtype=np.float32),
        indexing="ij",
    )
    u = (px + 0.5) * float(width) / float(side) - 0.5
    v = (py + 0.5) * float(height) / float(side) - 0.5
    x = (u - float(intrinsic[0, 2])) / float(intrinsic[0, 0])
    y = (v - float(intrinsic[1, 2])) / float(intrinsic[1, 1])
    z = np.full_like(x, float(camera_forward_sign))
    rays_camera = np.stack((x, y, z), axis=-1).reshape(-1, 3)
    rays_world = rays_camera @ rotation.T
    rays_world /= np.maximum(
        np.linalg.norm(rays_world, axis=1, keepdims=True), 1.0e-6
    )

    patch_u = ((px + 0.5) / float(side) * 2.0 - 1.0).reshape(-1)
    patch_v = ((py + 0.5) / float(side) * 2.0 - 1.0).reshape(-1)
    view_position = (
        0.0
        if view_count <= 1
        else float(view_index) / float(view_count - 1) * 2.0 - 1.0
    )
    geometry = np.concatenate(
        (
            rays_world,
            np.broadcast_to(origin_direction, (side * side, 3)),
            np.full((side * side, 1), min(radius / 4.0, 2.0), dtype=np.float32),
            patch_u[:, None],
            patch_v[:, None],
            np.full((side * side, 1), view_position, dtype=np.float32),
        ),
        axis=1,
    )
    return geometry.astype(np.float32)


def build_projected_patch_features(
    *,
    prior_coords: np.ndarray | torch.Tensor,
    prior_conf: np.ndarray | torch.Tensor,
    intrinsics: np.ndarray | torch.Tensor,
    extrinsics: np.ndarray | torch.Tensor,
    mask_paths: Sequence[str | Path],
    grid_transform: str,
    extrinsics_type: str,
    camera_forward_sign: float,
    patch_grid_side: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Project sparse canonical points into the exact view-major patch layout."""

    coords = np.asarray(
        prior_coords.detach().cpu().numpy() if torch.is_tensor(prior_coords) else prior_coords
    )
    confidence = np.asarray(
        prior_conf.detach().cpu().numpy() if torch.is_tensor(prior_conf) else prior_conf,
        dtype=np.float32,
    )
    k_all = np.asarray(
        intrinsics.detach().cpu().numpy() if torch.is_tensor(intrinsics) else intrinsics,
        dtype=np.float32,
    )
    t_all = np.asarray(
        extrinsics.detach().cpu().numpy() if torch.is_tensor(extrinsics) else extrinsics,
        dtype=np.float32,
    )
    masks = _load_masks(mask_paths)
    side = int(patch_grid_side)
    if side <= 0:
        raise ValueError(f"patch_grid_side must be positive, got {side}")
    if coords.ndim != 2 or coords.shape[1] not in (3, 4):
        raise ValueError(f"prior_coords must be [N,3/4], got {coords.shape}")
    if confidence.ndim != 1 or len(confidence) != len(coords):
        raise ValueError(
            f"prior_conf mismatch: coords={coords.shape}, conf={confidence.shape}"
        )
    view_count = len(masks)
    if k_all.shape != (view_count, 3, 3) or t_all.shape != (view_count, 4, 4):
        raise ValueError(
            "K/T/view mismatch: "
            f"views={view_count}, K={k_all.shape}, T={t_all.shape}"
        )
    if not (
        np.isfinite(coords).all()
        and np.isfinite(confidence).all()
        and np.isfinite(k_all).all()
        and np.isfinite(t_all).all()
    ):
        raise ValueError("projected patch input contains non-finite values")
    xyz = coords[:, -3:]
    if xyz.size and ((xyz < 0).any() or (xyz >= 64).any()):
        raise ValueError("prior coordinates must be inside the 64^3 canonical grid")

    points = apply_grid_transform(coords_to_points(coords, 64), str(grid_transform))
    points_h = np.concatenate(
        (points.astype(np.float32), np.ones((len(points), 1), dtype=np.float32)),
        axis=1,
    )
    rows: list[np.ndarray] = []
    projected_per_view: list[int] = []
    occupied_per_view: list[int] = []

    for view_index, (mask, intrinsic, extrinsic) in enumerate(
        zip(masks, k_all, t_all)
    ):
        height, width = mask.shape
        w2c = np.linalg.inv(extrinsic) if extrinsics_type == "c2w" else extrinsic
        cam = (w2c @ points_h.T).T[:, :3]
        depth = cam[:, 2] * float(camera_forward_sign)
        valid_depth = depth > 1.0e-5
        safe_depth = np.maximum(depth, 1.0e-5)
        u = intrinsic[0, 0] * (cam[:, 0] / safe_depth) + intrinsic[0, 2]
        v = intrinsic[1, 1] * (cam[:, 1] / safe_depth) + intrinsic[1, 2]
        valid = (
            valid_depth
            & (u >= 0.0)
            & (u < float(width))
            & (v >= 0.0)
            & (v < float(height))
        )
        ids = np.nonzero(valid)[0]
        patch_count = side * side
        count = np.zeros(patch_count, dtype=np.float32)
        confidence_sum = np.zeros_like(count)
        confidence_max = np.zeros_like(count)
        depth_sum = np.zeros_like(count)
        depth_sq_sum = np.zeros_like(count)
        nearest_depth = np.ones_like(count)
        mask_hit_sum = np.zeros_like(count)

        if len(ids):
            px = np.floor(u[ids] / float(width) * side).astype(np.int32)
            py = np.floor(v[ids] / float(height) * side).astype(np.int32)
            px = np.clip(px, 0, side - 1)
            py = np.clip(py, 0, side - 1)
            patch_index = py * side + px
            median_depth = max(float(np.median(depth[ids])), 1.0e-5)
            normalized_depth = np.clip(depth[ids] / median_depth, 0.0, 2.0) * 0.5
            ui = np.clip(np.rint(u[ids]).astype(np.int32), 0, width - 1)
            vi = np.clip(np.rint(v[ids]).astype(np.int32), 0, height - 1)
            hits = mask[vi, ui]
            np.add.at(count, patch_index, 1.0)
            np.add.at(confidence_sum, patch_index, confidence[ids])
            np.maximum.at(confidence_max, patch_index, confidence[ids])
            np.add.at(depth_sum, patch_index, normalized_depth)
            np.add.at(depth_sq_sum, patch_index, normalized_depth**2)
            np.minimum.at(nearest_depth, patch_index, normalized_depth)
            np.add.at(mask_hit_sum, patch_index, hits)

        occupied = count > 0
        count_safe = np.maximum(count, 1.0)
        confidence_mean = confidence_sum / count_safe
        depth_mean = depth_sum / count_safe
        depth_var = np.maximum(depth_sq_sum / count_safe - depth_mean**2, 0.0)
        mask_hit_ratio = mask_hit_sum / count_safe
        nearest_depth[~occupied] = 0.0
        log_count = np.log1p(count) / max(float(np.log1p(max(count.max(), 1.0))), 1.0e-6)
        mask_fraction = _patch_mask_fraction(mask, side).reshape(-1)
        evidence = np.stack(
            (
                occupied.astype(np.float32),
                log_count.astype(np.float32),
                confidence_max,
                confidence_mean,
                nearest_depth,
                depth_mean,
                np.sqrt(depth_var),
                mask_hit_ratio,
                np.where(occupied, 1.0 - mask_hit_ratio, 0.0),
                mask_fraction,
            ),
            axis=1,
        )
        geometry = _camera_geometry(
            intrinsic,
            extrinsic,
            width=width,
            height=height,
            side=side,
            extrinsics_type=str(extrinsics_type),
            camera_forward_sign=float(camera_forward_sign),
            view_index=view_index,
            view_count=view_count,
        )
        rows.append(np.concatenate((evidence, geometry), axis=1).astype(np.float32))
        projected_per_view.append(int(len(ids)))
        occupied_per_view.append(int(occupied.sum()))

    features = np.stack(rows, axis=0)
    expected = (view_count, side * side, len(PROJECTED_PATCH_FEATURE_NAMES))
    if features.shape != expected:
        raise RuntimeError(f"unexpected projected feature shape {features.shape}, expected {expected}")
    if not np.isfinite(features).all():
        raise RuntimeError("projected patch features contain non-finite values")
    report = {
        "version": PROJECTED_PATCH_FEATURE_VERSION,
        "feature_names": list(PROJECTED_PATCH_FEATURE_NAMES),
        "feature_schema_hash": projected_patch_feature_schema_hash(),
        "view_count": int(view_count),
        "patch_grid_side": int(side),
        "patch_count_per_view": int(side * side),
        "projected_point_count_per_view": projected_per_view,
        "occupied_patch_count_per_view": occupied_per_view,
        "feature_min": float(features.min()),
        "feature_max": float(features.max()),
    }
    return torch.from_numpy(features.reshape(1, view_count * side * side, -1)), report
