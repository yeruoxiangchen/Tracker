from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class VisualHullStats:
    num_coords: int
    support_min: int
    support_max: int
    support_mean: float
    visible_min: int
    visible_max: int
    visible_mean: float
    score_min: float
    score_max: float
    score_mean: float
    min_visible_views: int
    min_support_views: int
    min_support_ratio: float
    surface_only: bool

    def to_dict(self) -> dict:
        return {
            "num_coords": self.num_coords,
            "support_min": self.support_min,
            "support_max": self.support_max,
            "support_mean": self.support_mean,
            "visible_min": self.visible_min,
            "visible_max": self.visible_max,
            "visible_mean": self.visible_mean,
            "score_min": self.score_min,
            "score_max": self.score_max,
            "score_mean": self.score_mean,
            "min_visible_views": self.min_visible_views,
            "min_support_views": self.min_support_views,
            "min_support_ratio": self.min_support_ratio,
            "surface_only": self.surface_only,
        }


def voxel_centers(resolution: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    coords_1d = torch.arange(resolution, device=device, dtype=torch.long)
    gx, gy, gz = torch.meshgrid(coords_1d, coords_1d, coords_1d, indexing="ij")
    coords = torch.stack([gx, gy, gz], dim=-1).reshape(-1, 3)
    centers = (coords.to(dtype) + 0.5) / float(resolution) - 0.5
    return centers, coords


def _prepare_masks(masks: torch.Tensor) -> torch.Tensor:
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    elif masks.ndim == 5 and masks.shape[0] == 1 and masks.shape[2] == 1:
        masks = masks[0, :, 0]
    elif masks.ndim != 3:
        raise ValueError(f"masks should be [V,H,W], [V,1,H,W], or [1,V,1,H,W], got {tuple(masks.shape)}")
    return masks.float().clamp(0.0, 1.0)


def compute_visual_hull_score_grid(
    masks: torch.Tensor,
    intrinsics: torch.Tensor,
    extrinsics: torch.Tensor,
    *,
    extrinsics_are_c2w: bool = True,
    resolution: int = 64,
    mask_threshold: float = 0.5,
    min_visible_views: int = 1,
    chunk_size: int = 32768,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project canonical voxel centers into masks and return support ratio grids.

    Spatial dimensions are ordered as [x, y, z], matching target_coords in the
    ar_pose_trellis datasets and the sparse structure tensor indexing used by
    train_ss_ar_pose.py.
    """
    masks = _prepare_masks(masks)
    if intrinsics.ndim != 3 or intrinsics.shape[-2:] != (3, 3):
        raise ValueError(f"intrinsics should be [V,3,3], got {tuple(intrinsics.shape)}")
    if extrinsics.ndim != 3 or extrinsics.shape[-2:] != (4, 4):
        raise ValueError(f"extrinsics should be [V,4,4], got {tuple(extrinsics.shape)}")
    if masks.shape[0] != intrinsics.shape[0] or masks.shape[0] != extrinsics.shape[0]:
        raise ValueError(
            "view count mismatch: "
            f"masks={masks.shape[0]} intrinsics={intrinsics.shape[0]} extrinsics={extrinsics.shape[0]}"
        )

    device = masks.device
    dtype = torch.float32
    masks = masks.to(device=device, dtype=dtype)
    intrinsics = intrinsics.to(device=device, dtype=dtype)
    extrinsics = extrinsics.to(device=device, dtype=dtype)
    w2c = torch.linalg.inv(extrinsics) if extrinsics_are_c2w else extrinsics

    height, width = int(masks.shape[-2]), int(masks.shape[-1])
    centers, _ = voxel_centers(resolution, device=device, dtype=dtype)
    num_points = centers.shape[0]
    support = torch.zeros(num_points, device=device, dtype=torch.int16)
    visible = torch.zeros(num_points, device=device, dtype=torch.int16)

    ones = torch.ones((min(chunk_size, num_points), 1), device=device, dtype=dtype)
    for start in range(0, num_points, chunk_size):
        end = min(start + chunk_size, num_points)
        pts = centers[start:end]
        if ones.shape[0] != pts.shape[0]:
            ones_chunk = torch.ones((pts.shape[0], 1), device=device, dtype=dtype)
        else:
            ones_chunk = ones
        pts_h = torch.cat([pts, ones_chunk], dim=1)

        support_chunk = torch.zeros(pts.shape[0], device=device, dtype=torch.int16)
        visible_chunk = torch.zeros(pts.shape[0], device=device, dtype=torch.int16)
        for view_idx in range(masks.shape[0]):
            cam = (w2c[view_idx] @ pts_h.T).T[:, :3]
            depth = cam[:, 2]
            valid_depth = depth > 1e-6
            z = depth.clamp_min(1e-6)
            k = intrinsics[view_idx]
            u = k[0, 0] * (cam[:, 0] / z) + k[0, 2]
            v = k[1, 1] * (cam[:, 1] / z) + k[1, 2]
            ui = torch.round(u).long()
            vi = torch.round(v).long()
            in_image = valid_depth & (ui >= 0) & (ui < width) & (vi >= 0) & (vi < height)
            visible_chunk += in_image.to(torch.int16)
            if in_image.any():
                valid_ids = torch.where(in_image)[0]
                mask_values = masks[view_idx, vi[valid_ids], ui[valid_ids]]
                hit = mask_values > float(mask_threshold)
                support_chunk[valid_ids] += hit.to(torch.int16)

        support[start:end] = support_chunk
        visible[start:end] = visible_chunk

    visible_f = visible.float()
    score = support.float() / visible_f.clamp_min(1.0)
    score = torch.where(visible >= int(min_visible_views), score, torch.zeros_like(score))
    return (
        score.reshape(resolution, resolution, resolution),
        support.reshape(resolution, resolution, resolution),
        visible.reshape(resolution, resolution, resolution),
    )


def occupancy_to_surface_coords(occupancy: torch.Tensor) -> torch.Tensor:
    occ = occupancy.bool()
    if occ.ndim != 3:
        raise ValueError(f"occupancy should be [R,R,R], got {tuple(occ.shape)}")

    interior = occ.clone()
    interior[0, :, :] = False
    interior[-1, :, :] = False
    interior[:, 0, :] = False
    interior[:, -1, :] = False
    interior[:, :, 0] = False
    interior[:, :, -1] = False
    interior[1:, :, :] &= occ[:-1, :, :]
    interior[:-1, :, :] &= occ[1:, :, :]
    interior[:, 1:, :] &= occ[:, :-1, :]
    interior[:, :-1, :] &= occ[:, 1:, :]
    interior[:, :, 1:] &= occ[:, :, :-1]
    interior[:, :, :-1] &= occ[:, :, 1:]

    surface = occ & ~interior
    return torch.nonzero(surface, as_tuple=False).int()


def visual_hull_coords(
    masks: torch.Tensor,
    intrinsics: torch.Tensor,
    extrinsics: torch.Tensor,
    *,
    extrinsics_are_c2w: bool = True,
    resolution: int = 64,
    mask_threshold: float = 0.5,
    min_visible_views: int = 1,
    min_support_views: int = 2,
    min_support_ratio: float = 0.6,
    surface_only: bool = True,
    chunk_size: int = 32768,
) -> tuple[torch.Tensor, VisualHullStats]:
    score, support, visible = compute_visual_hull_score_grid(
        masks,
        intrinsics,
        extrinsics,
        extrinsics_are_c2w=extrinsics_are_c2w,
        resolution=resolution,
        mask_threshold=mask_threshold,
        min_visible_views=min_visible_views,
        chunk_size=chunk_size,
    )
    occupied = (
        (visible >= int(min_visible_views))
        & (support >= int(min_support_views))
        & (score >= float(min_support_ratio))
    )
    coords = occupancy_to_surface_coords(occupied) if surface_only else torch.nonzero(occupied, as_tuple=False).int()

    stats = VisualHullStats(
        num_coords=int(coords.shape[0]),
        support_min=int(support.min().item()),
        support_max=int(support.max().item()),
        support_mean=float(support.float().mean().item()),
        visible_min=int(visible.min().item()),
        visible_max=int(visible.max().item()),
        visible_mean=float(visible.float().mean().item()),
        score_min=float(score.min().item()),
        score_max=float(score.max().item()),
        score_mean=float(score.mean().item()),
        min_visible_views=int(min_visible_views),
        min_support_views=int(min_support_views),
        min_support_ratio=float(min_support_ratio),
        surface_only=bool(surface_only),
    )
    return coords, stats


def visual_hull_logit_bias(
    masks: torch.Tensor,
    intrinsics: torch.Tensor,
    extrinsics: torch.Tensor,
    *,
    extrinsics_are_c2w: bool = True,
    resolution: int = 64,
    mask_threshold: float = 0.5,
    min_visible_views: int = 1,
    weight: float = 0.0,
    chunk_size: int = 32768,
) -> tuple[torch.Tensor, dict]:
    score, support, visible = compute_visual_hull_score_grid(
        masks,
        intrinsics,
        extrinsics,
        extrinsics_are_c2w=extrinsics_are_c2w,
        resolution=resolution,
        mask_threshold=mask_threshold,
        min_visible_views=min_visible_views,
        chunk_size=chunk_size,
    )
    bias = (2.0 * score - 1.0) * float(weight)
    stats = {
        "visual_hull_prior_weight": float(weight),
        "visual_hull_score_min": float(score.min().item()),
        "visual_hull_score_max": float(score.max().item()),
        "visual_hull_score_mean": float(score.mean().item()),
        "visual_hull_support_min": int(support.min().item()),
        "visual_hull_support_max": int(support.max().item()),
        "visual_hull_support_mean": float(support.float().mean().item()),
        "visual_hull_visible_min": int(visible.min().item()),
        "visual_hull_visible_max": int(visible.max().item()),
        "visual_hull_visible_mean": float(visible.float().mean().item()),
    }
    return bias.reshape(1, 1, resolution, resolution, resolution), stats
