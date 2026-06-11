from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


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


@dataclass
class ObjectVolumeEstimate:
    object_to_world: torch.Tensor
    center_world: torch.Tensor
    extent_world: float
    aabb_min_world: torch.Tensor
    aabb_max_world: torch.Tensor
    initial_center_world: torch.Tensor
    initial_extent_world: float
    occupied_count: int
    occupied_ratio: float
    valid_ray_count: int
    fallback: bool
    support_min: int
    support_max: int
    support_mean: float
    visible_min: int
    visible_max: int
    visible_mean: float
    score_min: float
    score_max: float
    score_mean: float

    def to_dict(self) -> dict:
        return {
            "mode": "visual_hull_auto_volume",
            "object_to_world": self.object_to_world.detach().cpu().tolist(),
            "center_world": self.center_world.detach().cpu().tolist(),
            "extent_world": self.extent_world,
            "aabb_min_world": self.aabb_min_world.detach().cpu().tolist(),
            "aabb_max_world": self.aabb_max_world.detach().cpu().tolist(),
            "initial_center_world": self.initial_center_world.detach().cpu().tolist(),
            "initial_extent_world": self.initial_extent_world,
            "occupied_count": self.occupied_count,
            "occupied_ratio": self.occupied_ratio,
            "valid_ray_count": self.valid_ray_count,
            "fallback": self.fallback,
            "support_min": self.support_min,
            "support_max": self.support_max,
            "support_mean": self.support_mean,
            "visible_min": self.visible_min,
            "visible_max": self.visible_max,
            "visible_mean": self.visible_mean,
            "score_min": self.score_min,
            "score_max": self.score_max,
            "score_mean": self.score_mean,
        }


def voxel_centers(resolution: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    coords_1d = torch.arange(resolution, device=device, dtype=torch.long)
    gx, gy, gz = torch.meshgrid(coords_1d, coords_1d, coords_1d, indexing="ij")
    coords = torch.stack([gx, gy, gz], dim=-1).reshape(-1, 3)
    centers = (coords.to(dtype) + 0.5) / float(resolution) - 0.5
    return centers, coords


def pixal3d_grid_points(resolution: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    coords_1d = torch.arange(resolution, device=device, dtype=torch.long)
    gx, gy, gz = torch.meshgrid(coords_1d, coords_1d, coords_1d, indexing="ij")
    coords = torch.stack([gx, gy, gz], dim=-1).reshape(-1, 3)
    one_dim = torch.linspace(-1.0, 1.0, resolution, device=device, dtype=dtype)
    x, y, z = torch.meshgrid(one_dim, one_dim, one_dim, indexing="ij")
    points = torch.stack((x, y, z), dim=-1)
    rotation = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
        device=device,
        dtype=dtype,
    )
    points = torch.matmul(points, rotation.T).reshape(-1, 3) / 2.0
    return points, coords


def coords_to_centers(coords: torch.Tensor, resolution: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    xyz = coords[:, 1:] if coords.shape[1] == 4 else coords
    return (xyz.to(dtype) + 0.5) / float(resolution) - 0.5


def resize_masks(masks: Optional[torch.Tensor], size: int, device: torch.device) -> Optional[torch.Tensor]:
    if masks is None:
        return None
    if masks.ndim == 3:
        masks = masks[:, None]
    if masks.ndim != 4 or masks.shape[1] != 1:
        raise ValueError(f"masks should be [V,H,W] or [V,1,H,W], got {tuple(masks.shape)}")
    return F.interpolate(masks.float(), size=(size, size), mode="nearest").to(device)


def scale_intrinsics_to_square(
    intrinsics: torch.Tensor,
    source_sizes: list[tuple[int, int]],
    image_size: int,
    device: torch.device,
) -> torch.Tensor:
    if intrinsics.ndim != 3 or intrinsics.shape[-2:] != (3, 3):
        raise ValueError(f"intrinsics should be [V,3,3], got {tuple(intrinsics.shape)}")
    if len(source_sizes) != intrinsics.shape[0]:
        raise ValueError("source_sizes length should match view count")
    scaled = intrinsics.to(device=device, dtype=torch.float32).clone()
    for i, (width, height) in enumerate(source_sizes):
        sx = float(image_size) / float(width)
        sy = float(image_size) / float(height)
        scaled[i, 0, 0] *= sx
        scaled[i, 0, 2] *= sx
        scaled[i, 1, 1] *= sy
        scaled[i, 1, 2] *= sy
    return scaled


def resolve_object_to_world(
    object_to_world: Optional[torch.Tensor],
    world_to_object: Optional[torch.Tensor],
    device: torch.device,
) -> Optional[torch.Tensor]:
    if object_to_world is not None and world_to_object is not None:
        raise ValueError("Provide only one of object_to_world or world_to_object")
    if object_to_world is not None:
        return object_to_world.to(device=device, dtype=torch.float32)
    if world_to_object is not None:
        return torch.linalg.inv(world_to_object.to(device=device, dtype=torch.float32))
    return None


def _c2w_from_extrinsics(extrinsics: torch.Tensor, extrinsics_are_c2w: bool) -> torch.Tensor:
    extrinsics = extrinsics.float()
    return extrinsics if extrinsics_are_c2w else torch.linalg.inv(extrinsics)


def mask_centroid_pixels(masks: torch.Tensor, mask_threshold: float = 0.5) -> tuple[torch.Tensor, torch.Tensor]:
    masks = _prepare_masks(masks)
    device = masks.device
    view_count, height, width = masks.shape
    centroids = torch.zeros((view_count, 2), device=device, dtype=torch.float32)
    valid = torch.zeros((view_count,), device=device, dtype=torch.bool)
    for view_idx in range(view_count):
        ys, xs = torch.where(masks[view_idx] > float(mask_threshold))
        if xs.numel() == 0:
            continue
        centroids[view_idx, 0] = xs.float().mean()
        centroids[view_idx, 1] = ys.float().mean()
        valid[view_idx] = True
    return centroids, valid


def rays_from_pixels(
    pixels: torch.Tensor,
    intrinsics: torch.Tensor,
    extrinsics: torch.Tensor,
    *,
    extrinsics_are_c2w: bool = True,
    camera_forward_sign: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = intrinsics.device
    dtype = torch.float32
    pixels = pixels.to(device=device, dtype=dtype)
    intrinsics = intrinsics.to(device=device, dtype=dtype)
    c2w = _c2w_from_extrinsics(extrinsics.to(device=device, dtype=dtype), extrinsics_are_c2w)
    ones = torch.ones((pixels.shape[0], 1), device=device, dtype=dtype)
    pix_h = torch.cat([pixels, ones], dim=1)
    inv_k = torch.linalg.inv(intrinsics)
    dirs_cam = torch.bmm(inv_k, pix_h[:, :, None]).squeeze(-1)
    dirs_cam[:, 2] = float(camera_forward_sign)
    dirs_world = torch.bmm(c2w[:, :3, :3], dirs_cam[:, :, None]).squeeze(-1)
    dirs_world = F.normalize(dirs_world, dim=1)
    origins_world = c2w[:, :3, 3]
    return origins_world, dirs_world


def closest_point_to_rays(
    origins: torch.Tensor,
    directions: torch.Tensor,
    *,
    min_rays: int = 2,
) -> tuple[torch.Tensor, bool]:
    device = origins.device
    dtype = torch.float32
    if origins.shape[0] < int(min_rays):
        if origins.numel() == 0:
            return torch.zeros((3,), device=device, dtype=dtype), True
        return origins.mean(dim=0).to(dtype), True

    directions = F.normalize(directions.to(dtype), dim=1)
    origins = origins.to(dtype)
    eye = torch.eye(3, device=device, dtype=dtype)
    proj = eye[None] - directions[:, :, None] * directions[:, None, :]
    a = proj.sum(dim=0)
    b = torch.bmm(proj, origins[:, :, None]).squeeze(-1).sum(dim=0)
    a = a + torch.eye(3, device=device, dtype=dtype) * 1e-5
    try:
        center = torch.linalg.solve(a, b)
        return center, False
    except RuntimeError:
        return origins.mean(dim=0), True


def default_extent_from_cameras(
    camera_centers: torch.Tensor,
    center_world: torch.Tensor,
    *,
    initial_extent_ratio: float = 0.6,
    min_extent: float = 0.05,
) -> float:
    if camera_centers.numel() == 0:
        return float(min_extent)
    distances = torch.linalg.norm(camera_centers - center_world[None], dim=1)
    distances = distances[torch.isfinite(distances) & (distances > 1e-6)]
    if distances.numel() == 0:
        return float(min_extent)
    return max(float(distances.median().item()) * float(initial_extent_ratio), float(min_extent))


def world_cube_points(
    center_world: torch.Tensor,
    extent_world: float,
    resolution: int,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    coords_1d = torch.arange(resolution, device=device, dtype=torch.long)
    gx, gy, gz = torch.meshgrid(coords_1d, coords_1d, coords_1d, indexing="ij")
    coords = torch.stack([gx, gy, gz], dim=-1).reshape(-1, 3)
    local = (coords.to(dtype) + 0.5) / float(resolution) - 0.5
    points = center_world.to(device=device, dtype=dtype)[None] + local * float(extent_world)
    return points, coords


def compute_visual_hull_score_points(
    points_world: torch.Tensor,
    masks: torch.Tensor,
    intrinsics: torch.Tensor,
    extrinsics: torch.Tensor,
    *,
    extrinsics_are_c2w: bool = True,
    camera_forward_sign: float = 1.0,
    mask_threshold: float = 0.5,
    min_visible_views: int = 1,
    chunk_size: int = 32768,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    masks = _prepare_masks(masks)
    device = masks.device
    points_world = points_world.to(device=device, dtype=torch.float32)
    intrinsics = intrinsics.to(device=device, dtype=torch.float32)
    extrinsics = extrinsics.to(device=device, dtype=torch.float32)
    support = torch.zeros((points_world.shape[0],), device=device, dtype=torch.int16)
    visible = torch.zeros((points_world.shape[0],), device=device, dtype=torch.int16)
    height, width = int(masks.shape[-2]), int(masks.shape[-1])

    for start in range(0, points_world.shape[0], chunk_size):
        end = min(start + chunk_size, points_world.shape[0])
        points_2d, _, valid_depth = project_points_multi_view(
            points_world[start:end],
            intrinsics,
            extrinsics,
            extrinsics_are_c2w=extrinsics_are_c2w,
            camera_forward_sign=camera_forward_sign,
            object_to_world=None,
        )
        u = points_2d[..., 0]
        v = points_2d[..., 1]
        ui = torch.round(u).long()
        vi = torch.round(v).long()
        in_image = valid_depth & (ui >= 0) & (ui < width) & (vi >= 0) & (vi < height)
        support_chunk = torch.zeros((end - start,), device=device, dtype=torch.int16)
        visible_chunk = in_image.sum(dim=0).to(torch.int16)
        for view_idx in range(masks.shape[0]):
            ids = torch.where(in_image[view_idx])[0]
            if ids.numel() == 0:
                continue
            hits = masks[view_idx, vi[view_idx, ids], ui[view_idx, ids]] > float(mask_threshold)
            support_chunk[ids] += hits.to(torch.int16)
        support[start:end] = support_chunk
        visible[start:end] = visible_chunk

    score = support.float() / visible.float().clamp_min(1.0)
    score = torch.where(visible >= int(min_visible_views), score, torch.zeros_like(score))
    return score, support, visible


def estimate_object_volume_from_visual_hull(
    masks: torch.Tensor,
    intrinsics: torch.Tensor,
    extrinsics: torch.Tensor,
    *,
    extrinsics_are_c2w: bool = True,
    camera_forward_sign: float = 1.0,
    mask_threshold: float = 0.5,
    resolution: int = 48,
    min_visible_views: int = 1,
    min_support_views: int = 2,
    min_support_ratio: float = 0.6,
    initial_extent_ratio: float = 0.6,
    padding: float = 1.25,
    min_extent: float = 0.05,
    refine_steps: int = 2,
    chunk_size: int = 32768,
) -> ObjectVolumeEstimate:
    """Estimate a temporary world-space object volume from masks and camera poses.

    The returned transform maps Pixal3D's canonical cube [-0.5, 0.5]^3 into
    the estimated visual-hull AABB. It is an internal projection aid, not a
    final model-to-world pose.
    """
    masks = _prepare_masks(masks).float()
    device = masks.device
    intrinsics = intrinsics.to(device=device, dtype=torch.float32)
    extrinsics = extrinsics.to(device=device, dtype=torch.float32)
    c2w = _c2w_from_extrinsics(extrinsics, extrinsics_are_c2w)

    centroids, valid_mask = mask_centroid_pixels(masks, mask_threshold)
    valid_ids = torch.where(valid_mask)[0]
    if valid_ids.numel() > 0:
        origins, dirs = rays_from_pixels(
            centroids[valid_ids],
            intrinsics[valid_ids],
            extrinsics[valid_ids],
            extrinsics_are_c2w=extrinsics_are_c2w,
            camera_forward_sign=camera_forward_sign,
        )
        initial_center, ray_fallback = closest_point_to_rays(origins, dirs)
    else:
        origins = c2w[:, :3, 3]
        forward = c2w[:, :3, 2] * float(camera_forward_sign)
        initial_center = (origins + forward).mean(dim=0)
        ray_fallback = True

    initial_extent = default_extent_from_cameras(
        c2w[:, :3, 3],
        initial_center,
        initial_extent_ratio=initial_extent_ratio,
        min_extent=min_extent,
    )

    center = initial_center
    extent = initial_extent
    occupied = None
    points_world = None
    score = support = visible = None
    fallback = bool(ray_fallback)

    for _ in range(max(int(refine_steps), 1)):
        points_world, _ = world_cube_points(
            center,
            extent,
            int(resolution),
            device=device,
            dtype=torch.float32,
        )
        score, support, visible = compute_visual_hull_score_points(
            points_world,
            masks,
            intrinsics,
            extrinsics,
            extrinsics_are_c2w=extrinsics_are_c2w,
            camera_forward_sign=camera_forward_sign,
            mask_threshold=mask_threshold,
            min_visible_views=min_visible_views,
            chunk_size=chunk_size,
        )
        occupied = (visible >= int(min_visible_views)) & (support >= int(min_support_views)) & (score >= float(min_support_ratio))
        if not occupied.any():
            fallback = True
            break
        occ_points = points_world[occupied]
        aabb_min = occ_points.min(dim=0).values
        aabb_max = occ_points.max(dim=0).values
        center = (aabb_min + aabb_max) * 0.5
        extent = max(float((aabb_max - aabb_min).max().item()) * float(padding), float(min_extent))

    if occupied is None or points_world is None or score is None or support is None or visible is None or not occupied.any():
        aabb_min = initial_center - initial_extent * 0.5
        aabb_max = initial_center + initial_extent * 0.5
        center = initial_center
        extent = initial_extent
        occupied_count = 0
        occupied_ratio = 0.0
        support_min = support_max = 0
        support_mean = 0.0
        visible_min = visible_max = 0
        visible_mean = 0.0
        score_min = score_max = score_mean = 0.0
    else:
        occ_points = points_world[occupied]
        aabb_min = occ_points.min(dim=0).values
        aabb_max = occ_points.max(dim=0).values
        occupied_count = int(occupied.sum().item())
        occupied_ratio = float(occupied.float().mean().item())
        support_min = int(support.min().item())
        support_max = int(support.max().item())
        support_mean = float(support.float().mean().item())
        visible_min = int(visible.min().item())
        visible_max = int(visible.max().item())
        visible_mean = float(visible.float().mean().item())
        score_min = float(score.min().item())
        score_max = float(score.max().item())
        score_mean = float(score.mean().item())

    object_to_world = torch.eye(4, device=device, dtype=torch.float32)
    object_to_world[:3, :3] *= float(extent)
    object_to_world[:3, 3] = center

    return ObjectVolumeEstimate(
        object_to_world=object_to_world,
        center_world=center,
        extent_world=float(extent),
        aabb_min_world=aabb_min,
        aabb_max_world=aabb_max,
        initial_center_world=initial_center,
        initial_extent_world=float(initial_extent),
        occupied_count=occupied_count,
        occupied_ratio=occupied_ratio,
        valid_ray_count=int(valid_ids.numel()),
        fallback=bool(fallback),
        support_min=support_min,
        support_max=support_max,
        support_mean=support_mean,
        visible_min=visible_min,
        visible_max=visible_max,
        visible_mean=visible_mean,
        score_min=score_min,
        score_max=score_max,
        score_mean=score_mean,
    )


def project_points_multi_view(
    points_obj: torch.Tensor,
    intrinsics: torch.Tensor,
    extrinsics: torch.Tensor,
    *,
    extrinsics_are_c2w: bool = True,
    camera_forward_sign: float = 1.0,
    object_to_world: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = points_obj.device
    dtype = torch.float32
    points_obj = points_obj.to(device=device, dtype=dtype)
    intrinsics = intrinsics.to(device=device, dtype=dtype)
    extrinsics = extrinsics.to(device=device, dtype=dtype)
    w2c = torch.linalg.inv(extrinsics) if extrinsics_are_c2w else extrinsics

    ones = torch.ones((points_obj.shape[0], 1), device=device, dtype=dtype)
    points_h = torch.cat([points_obj, ones], dim=1)
    if object_to_world is not None:
        points_h = (object_to_world.to(device=device, dtype=dtype) @ points_h.T).T

    points_2d, depths, valid = [], [], []
    sign = float(camera_forward_sign)
    for view_idx in range(w2c.shape[0]):
        cam = (w2c[view_idx] @ points_h.T).T[:, :3]
        depth = sign * cam[:, 2]
        z = depth.clamp_min(1e-6)
        k = intrinsics[view_idx]
        u = k[0, 0] * (cam[:, 0] / z) + k[0, 2]
        v = k[1, 1] * (cam[:, 1] / z) + k[1, 2]
        points_2d.append(torch.stack([u, v], dim=-1))
        depths.append(depth)
        valid.append(depth > 1e-6)
    return torch.stack(points_2d, dim=0), torch.stack(depths, dim=0), torch.stack(valid, dim=0)


def sample_features_multi_view(
    feature_map: torch.Tensor,
    points_2d: torch.Tensor,
    valid_depth: torch.Tensor,
    *,
    coordinate_size: int,
    masks: Optional[torch.Tensor] = None,
    mask_threshold: float = 0.5,
    depths: Optional[torch.Tensor] = None,
    front_depth_maps: Optional[torch.Tensor] = None,
    visibility_depth_tolerance: float = 0.03,
    visibility_weight_min: float = 0.05,
    empty_policy: str = "zero",
    fallback_weight: float = 1.0,
    support_confidence_power: float = 1.0,
) -> tuple[torch.Tensor, dict]:
    if feature_map.ndim == 4 and feature_map.shape[1] < feature_map.shape[-1]:
        feature_map = feature_map.permute(0, 3, 1, 2)
    view_count, _, _, _ = feature_map.shape
    if points_2d.shape[0] != view_count or valid_depth.shape[0] != view_count:
        raise ValueError("feature, point, and valid-depth view counts do not match")

    u = points_2d[..., 0]
    v = points_2d[..., 1]
    in_image = valid_depth & (u >= 0) & (u < coordinate_size) & (v >= 0) & (v < coordinate_size)
    valid = in_image

    if masks is not None:
        masks = resize_masks(masks, coordinate_size, feature_map.device)
        mask_grid = torch.stack(
            [(u + 0.5) / float(coordinate_size) * 2.0 - 1.0, (v + 0.5) / float(coordinate_size) * 2.0 - 1.0],
            dim=-1,
        ).view(view_count, -1, 1, 2)
        mask_values = F.grid_sample(masks, mask_grid, mode="bilinear", align_corners=False, padding_mode="zeros")
        mask_values = mask_values.squeeze(1).squeeze(-1)
        valid = in_image & (mask_values > float(mask_threshold))
    else:
        mask_values = None

    grid = torch.stack(
        [(u + 0.5) / float(coordinate_size) * 2.0 - 1.0, (v + 0.5) / float(coordinate_size) * 2.0 - 1.0],
        dim=-1,
    ).view(view_count, -1, 1, 2)
    sampled = F.grid_sample(feature_map, grid, mode="bilinear", align_corners=False, padding_mode="border")
    sampled = sampled.squeeze(-1).permute(0, 2, 1)

    weights = valid.float()
    visibility_stats = None
    if front_depth_maps is not None:
        if depths is None:
            raise ValueError("depths is required when front_depth_maps is provided")
        if front_depth_maps.ndim == 3:
            front_depth_maps = front_depth_maps[:, None]
        if front_depth_maps.ndim != 4 or front_depth_maps.shape[1] != 1:
            raise ValueError(f"front_depth_maps should be [V,H,W] or [V,1,H,W], got {tuple(front_depth_maps.shape)}")
        if front_depth_maps.shape[0] != view_count:
            raise ValueError("front_depth_maps view count does not match feature map")
        if front_depth_maps.shape[-2:] != (coordinate_size, coordinate_size):
            front_depth_maps = F.interpolate(
                front_depth_maps.float(),
                size=(coordinate_size, coordinate_size),
                mode="nearest",
            )
        front_depth_maps = front_depth_maps.to(device=feature_map.device, dtype=torch.float32)
        finite_maps = torch.isfinite(front_depth_maps) & (front_depth_maps > 0)
        front_depth_for_sample = torch.where(finite_maps, front_depth_maps, torch.zeros_like(front_depth_maps))
        front_depth = F.grid_sample(
            front_depth_for_sample,
            grid,
            mode="nearest",
            align_corners=False,
            padding_mode="zeros",
        ).squeeze(1).squeeze(-1)
        front_finite = F.grid_sample(
            finite_maps.float(),
            grid,
            mode="nearest",
            align_corners=False,
            padding_mode="zeros",
        ).squeeze(1).squeeze(-1) > 0.5
        delta = (depths.to(feature_map.device, dtype=torch.float32) - front_depth).abs()
        tol = max(float(visibility_depth_tolerance), 1e-6)
        visibility_weight = (1.0 - delta / tol).clamp_min(0.0)
        visibility_weight = torch.where(front_finite, visibility_weight, torch.zeros_like(visibility_weight))
        visibility_weight = torch.where(visibility_weight >= float(visibility_weight_min), visibility_weight, torch.zeros_like(visibility_weight))
        weights = weights * visibility_weight
        visibility_stats = {
            "enabled": True,
            "depth_tolerance": tol,
            "weight_min": float(visibility_weight_min),
            "front_depth_coverage": float(finite_maps.float().mean().item()),
            "visible_weight_mean": float(visibility_weight.mean().item()) if visibility_weight.numel() else 0.0,
            "visible_weight_nonzero_ratio": float((visibility_weight > 0).float().mean().item()) if visibility_weight.numel() else 0.0,
        }
    else:
        visibility_stats = {"enabled": False}

    raw_support = weights.sum(dim=0)
    effective_support = raw_support
    weighted_agg = (sampled * weights[..., None]).sum(dim=0) / raw_support[:, None].clamp_min(1.0)
    fallback_points = torch.zeros_like(raw_support, dtype=torch.bool)
    fallback_support = torch.zeros_like(raw_support)
    support_confidence = (raw_support / float(max(view_count, 1))).clamp(0.0, 1.0)

    if empty_policy == "zero":
        aggregated = weighted_agg
    elif empty_policy in {"visible", "border", "soft"}:
        if empty_policy == "border":
            fallback_weights = torch.ones_like(weights)
        else:
            fallback_weights = in_image.float()
            no_in_image = fallback_weights.sum(dim=0) <= 0
            if no_in_image.any():
                fallback_weights = torch.where(no_in_image[None], torch.ones_like(fallback_weights), fallback_weights)
        fallback_support = fallback_weights.sum(dim=0)
        fallback_agg = (sampled * fallback_weights[..., None]).sum(dim=0) / fallback_support[:, None].clamp_min(1.0)
        fallback_points = raw_support <= 0
        effective_support = torch.where(fallback_points, fallback_support, raw_support)
        if empty_policy == "soft":
            confidence = support_confidence.clamp(0.0, 1.0)
            power = max(float(support_confidence_power), 1e-6)
            confidence = confidence.pow(power)[:, None]
            fallback_mix = float(fallback_weight)
            aggregated = confidence * weighted_agg + (1.0 - confidence) * fallback_mix * fallback_agg
        else:
            aggregated = torch.where(fallback_points[:, None], float(fallback_weight) * fallback_agg, weighted_agg)
    else:
        raise ValueError(f"Unknown empty_policy: {empty_policy}")
    stats = {
        "num_points": int(points_2d.shape[1]),
        "num_views": int(view_count),
        "empty_policy": empty_policy,
        "fallback_weight": float(fallback_weight),
        "support_confidence_power": float(support_confidence_power),
        "support_min": float(effective_support.min().item()) if effective_support.numel() else 0.0,
        "support_max": float(effective_support.max().item()) if effective_support.numel() else 0.0,
        "support_mean": float(effective_support.mean().item()) if effective_support.numel() else 0.0,
        "zero_support": int((effective_support <= 0).sum().item()) if effective_support.numel() else 0,
        "raw_support_min": float(raw_support.min().item()) if raw_support.numel() else 0.0,
        "raw_support_max": float(raw_support.max().item()) if raw_support.numel() else 0.0,
        "raw_support_mean": float(raw_support.mean().item()) if raw_support.numel() else 0.0,
        "raw_zero_support": int((raw_support <= 0).sum().item()) if raw_support.numel() else 0,
        "fallback_points": int(fallback_points.sum().item()) if fallback_points.numel() else 0,
        "support_confidence_mean": float(support_confidence.mean().item()) if support_confidence.numel() else 0.0,
    }
    if mask_values is not None:
        stats["mask_mean"] = float(mask_values.mean().item())
    stats["visibility"] = visibility_stats
    return aggregated.unsqueeze(0), stats


def geometry_features_from_projection(
    points_obj: torch.Tensor,
    points_2d: torch.Tensor,
    depths: torch.Tensor,
    valid_depth: torch.Tensor,
    *,
    coordinate_size: int,
    masks: Optional[torch.Tensor] = None,
    mask_threshold: float = 0.5,
    front_depth_maps: Optional[torch.Tensor] = None,
    visibility_depth_tolerance: float = 0.03,
    visibility_weight_min: float = 0.05,
    min_visible_views: int = 1,
    min_support_views: int = 2,
    min_support_ratio: float = 0.6,
) -> tuple[torch.Tensor, dict]:
    """Build explicit per-voxel geometry features aligned with projected DINO features.

    Returned features are [N, 17]:
    support/visible fractions, mask statistics, visual-hull inside/surface
    indicators, front-depth visibility, support entropy, zero-support flag, and
    normalized xyz.
    """
    device = points_2d.device
    view_count, num_points = int(points_2d.shape[0]), int(points_2d.shape[1])
    u = points_2d[..., 0]
    v = points_2d[..., 1]
    in_image = valid_depth & (u >= 0) & (u < coordinate_size) & (v >= 0) & (v < coordinate_size)

    if masks is not None:
        masks = resize_masks(masks, coordinate_size, device)
        mask_grid = torch.stack(
            [(u + 0.5) / float(coordinate_size) * 2.0 - 1.0, (v + 0.5) / float(coordinate_size) * 2.0 - 1.0],
            dim=-1,
        ).view(view_count, -1, 1, 2)
        mask_values = F.grid_sample(masks, mask_grid, mode="bilinear", align_corners=False, padding_mode="zeros")
        mask_values = mask_values.squeeze(1).squeeze(-1).clamp(0.0, 1.0)
    else:
        mask_values = in_image.float()

    visible = in_image.float()
    visible_count = visible.sum(dim=0)
    mask_values_visible = torch.where(in_image, mask_values, torch.zeros_like(mask_values))
    support_count = (in_image & (mask_values > float(mask_threshold))).float().sum(dim=0)
    support_ratio = support_count / visible_count.clamp_min(1.0)
    support_fraction = support_count / float(max(view_count, 1))
    visible_fraction = visible_count / float(max(view_count, 1))
    mask_prob_mean = mask_values_visible.sum(dim=0) / visible_count.clamp_min(1.0)
    mask_prob_var = torch.where(
        visible_count > 0,
        (((mask_values - mask_prob_mean[None]) ** 2) * visible).sum(dim=0) / visible_count.clamp_min(1.0),
        torch.zeros_like(mask_prob_mean),
    )
    mask_prob_std = torch.sqrt(mask_prob_var.clamp_min(0.0))
    mask_prob_max = torch.where(in_image, mask_values, torch.full_like(mask_values, -1.0)).max(dim=0).values.clamp_min(0.0)
    support_binary = (in_image & (mask_values > float(mask_threshold))).float()
    support_prob = support_binary / support_count.clamp_min(1.0)[None]
    support_entropy = -(support_prob.clamp_min(1e-8) * support_prob.clamp_min(1e-8).log()).sum(dim=0)
    if view_count > 1:
        support_entropy = support_entropy / torch.log(torch.tensor(float(view_count), device=device, dtype=torch.float32))
    support_entropy = torch.where(support_count > 0, support_entropy, torch.zeros_like(support_entropy))

    occupied = (
        (visible_count >= int(min_visible_views))
        & (support_count >= int(min_support_views))
        & (support_ratio >= float(min_support_ratio))
    )
    surface = torch.zeros_like(occupied, dtype=torch.bool)
    resolution = int(round(float(num_points) ** (1.0 / 3.0)))
    if resolution > 0 and resolution ** 3 == num_points:
        surface_coords = occupancy_to_surface_coords(occupied.reshape(resolution, resolution, resolution))
        if surface_coords.numel() > 0:
            flat = surface_coords[:, 0] * resolution * resolution + surface_coords[:, 1] * resolution + surface_coords[:, 2]
            surface[flat.long()] = True

    front_visibility_ratio = torch.zeros((num_points,), device=device, dtype=torch.float32)
    front_visibility_fraction = torch.zeros((num_points,), device=device, dtype=torch.float32)
    front_visibility_max = torch.zeros((num_points,), device=device, dtype=torch.float32)
    front_visibility_nonzero = 0.0
    if front_depth_maps is not None:
        if front_depth_maps.ndim == 3:
            front_depth_maps = front_depth_maps[:, None]
        if front_depth_maps.shape[-2:] != (coordinate_size, coordinate_size):
            front_depth_maps = F.interpolate(
                front_depth_maps.float(),
                size=(coordinate_size, coordinate_size),
                mode="nearest",
            )
        front_depth_maps = front_depth_maps.to(device=device, dtype=torch.float32)
        finite_maps = torch.isfinite(front_depth_maps) & (front_depth_maps > 0)
        front_depth_for_sample = torch.where(finite_maps, front_depth_maps, torch.zeros_like(front_depth_maps))
        grid = torch.stack(
            [(u + 0.5) / float(coordinate_size) * 2.0 - 1.0, (v + 0.5) / float(coordinate_size) * 2.0 - 1.0],
            dim=-1,
        ).view(view_count, -1, 1, 2)
        front_depth = F.grid_sample(
            front_depth_for_sample,
            grid,
            mode="nearest",
            align_corners=False,
            padding_mode="zeros",
        ).squeeze(1).squeeze(-1)
        front_finite = F.grid_sample(
            finite_maps.float(),
            grid,
            mode="nearest",
            align_corners=False,
            padding_mode="zeros",
        ).squeeze(1).squeeze(-1) > 0.5
        delta = (depths.to(device=device, dtype=torch.float32) - front_depth).abs()
        tol = max(float(visibility_depth_tolerance), 1e-6)
        visibility_weight = (1.0 - delta / tol).clamp_min(0.0)
        visibility_weight = torch.where(front_finite, visibility_weight, torch.zeros_like(visibility_weight))
        visibility_weight = torch.where(
            visibility_weight >= float(visibility_weight_min),
            visibility_weight,
            torch.zeros_like(visibility_weight),
        )
        visibility_weight = visibility_weight * (mask_values > float(mask_threshold)).float() * visible
        front_visibility_ratio = visibility_weight.sum(dim=0) / support_count.clamp_min(1.0)
        front_visibility_fraction = (visibility_weight > 0).float().sum(dim=0) / float(max(view_count, 1))
        front_visibility_max = visibility_weight.max(dim=0).values
        front_visibility_nonzero = float((visibility_weight > 0).float().mean().item()) if visibility_weight.numel() else 0.0

    xyz = (points_obj.to(device=device, dtype=torch.float32) * 2.0).clamp(-1.0, 1.0)
    zero_support = support_count <= 0
    surface_score = surface.float() * support_ratio
    features = torch.cat(
        [
            support_ratio[:, None],
            support_fraction[:, None],
            visible_fraction[:, None],
            mask_prob_mean[:, None],
            mask_prob_max[:, None],
            mask_prob_std[:, None],
            occupied.float()[:, None],
            surface.float()[:, None],
            surface_score[:, None],
            front_visibility_ratio[:, None],
            front_visibility_fraction[:, None],
            front_visibility_max[:, None],
            support_entropy[:, None],
            zero_support.float()[:, None],
            xyz,
        ],
        dim=1,
    )
    stats = {
        "enabled": True,
        "feature_dim": int(features.shape[1]),
        "feature_names": [
            "support_ratio",
            "support_fraction",
            "visible_fraction",
            "mask_prob_mean",
            "mask_prob_max",
            "mask_prob_std",
            "visual_hull_inside",
            "visual_hull_surface",
            "surface_score",
            "front_visibility_ratio",
            "front_visibility_fraction",
            "front_visibility_max",
            "support_entropy",
            "zero_support",
            "x",
            "y",
            "z",
        ],
        "support_ratio_mean": float(support_ratio.mean().item()) if support_ratio.numel() else 0.0,
        "support_fraction_mean": float(support_fraction.mean().item()) if support_fraction.numel() else 0.0,
        "visible_fraction_mean": float(visible_fraction.mean().item()) if visible_fraction.numel() else 0.0,
        "inside_ratio": float(occupied.float().mean().item()) if occupied.numel() else 0.0,
        "surface_ratio": float(surface.float().mean().item()) if surface.numel() else 0.0,
        "surface_score_mean": float(surface_score.mean().item()) if surface_score.numel() else 0.0,
        "front_visibility_ratio_mean": float(front_visibility_ratio.mean().item()) if front_visibility_ratio.numel() else 0.0,
        "front_visibility_fraction_mean": float(front_visibility_fraction.mean().item()) if front_visibility_fraction.numel() else 0.0,
        "support_entropy_mean": float(support_entropy.mean().item()) if support_entropy.numel() else 0.0,
        "zero_support_ratio": float(zero_support.float().mean().item()) if zero_support.numel() else 0.0,
        "front_visibility_nonzero_ratio": front_visibility_nonzero,
    }
    return features, stats


def _prepare_masks(masks: torch.Tensor) -> torch.Tensor:
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    elif masks.ndim != 3:
        raise ValueError(f"masks should be [V,H,W] or [V,1,H,W], got {tuple(masks.shape)}")
    return masks.float().clamp(0.0, 1.0)


def compute_visual_hull_score_grid(
    masks: torch.Tensor,
    intrinsics: torch.Tensor,
    extrinsics: torch.Tensor,
    *,
    extrinsics_are_c2w: bool = True,
    camera_forward_sign: float = 1.0,
    object_to_world: Optional[torch.Tensor] = None,
    resolution: int = 32,
    mask_threshold: float = 0.5,
    min_visible_views: int = 1,
    chunk_size: int = 32768,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    masks = _prepare_masks(masks)
    device = masks.device
    centers, _ = pixal3d_grid_points(resolution, device=device, dtype=torch.float32)
    support = torch.zeros((centers.shape[0],), device=device, dtype=torch.int16)
    visible = torch.zeros((centers.shape[0],), device=device, dtype=torch.int16)
    height, width = int(masks.shape[-2]), int(masks.shape[-1])

    for start in range(0, centers.shape[0], chunk_size):
        end = min(start + chunk_size, centers.shape[0])
        points_2d, _, valid_depth = project_points_multi_view(
            centers[start:end],
            intrinsics,
            extrinsics,
            extrinsics_are_c2w=extrinsics_are_c2w,
            camera_forward_sign=camera_forward_sign,
            object_to_world=object_to_world,
        )
        u = points_2d[..., 0]
        v = points_2d[..., 1]
        ui = torch.round(u).long()
        vi = torch.round(v).long()
        in_image = valid_depth & (ui >= 0) & (ui < width) & (vi >= 0) & (vi < height)
        support_chunk = torch.zeros((end - start,), device=device, dtype=torch.int16)
        visible_chunk = in_image.sum(dim=0).to(torch.int16)
        for view_idx in range(masks.shape[0]):
            ids = torch.where(in_image[view_idx])[0]
            if ids.numel() == 0:
                continue
            hits = masks[view_idx, vi[view_idx, ids], ui[view_idx, ids]] > float(mask_threshold)
            support_chunk[ids] += hits.to(torch.int16)
        support[start:end] = support_chunk
        visible[start:end] = visible_chunk

    score = support.float() / visible.float().clamp_min(1.0)
    score = torch.where(visible >= int(min_visible_views), score, torch.zeros_like(score))
    return score.reshape(resolution, resolution, resolution), support.reshape(resolution, resolution, resolution), visible.reshape(resolution, resolution, resolution)


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
    return torch.nonzero(occ & ~interior, as_tuple=False).int()


def visual_hull_coords(
    masks: torch.Tensor,
    intrinsics: torch.Tensor,
    extrinsics: torch.Tensor,
    *,
    extrinsics_are_c2w: bool = True,
    camera_forward_sign: float = 1.0,
    object_to_world: Optional[torch.Tensor] = None,
    resolution: int = 32,
    mask_threshold: float = 0.5,
    min_visible_views: int = 1,
    min_support_views: int = 2,
    min_support_ratio: float = 0.6,
    surface_only: bool = True,
) -> tuple[torch.Tensor, VisualHullStats]:
    score, support, visible = compute_visual_hull_score_grid(
        masks,
        intrinsics,
        extrinsics,
        extrinsics_are_c2w=extrinsics_are_c2w,
        camera_forward_sign=camera_forward_sign,
        object_to_world=object_to_world,
        resolution=resolution,
        mask_threshold=mask_threshold,
        min_visible_views=min_visible_views,
    )
    occupied = (visible >= int(min_visible_views)) & (support >= int(min_support_views)) & (score >= float(min_support_ratio))
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


def _scatter_min_depth(
    pixel_indices: torch.Tensor,
    depths: torch.Tensor,
    num_pixels: int,
    device: torch.device,
) -> torch.Tensor:
    front = torch.full((num_pixels,), float("inf"), device=device, dtype=torch.float32)
    if pixel_indices.numel() == 0:
        return front
    pixel_indices = pixel_indices.long()
    depths = depths.float()
    try:
        front.scatter_reduce_(0, pixel_indices, depths, reduce="amin", include_self=True)
        return front
    except AttributeError:
        order = torch.argsort(pixel_indices)
        sorted_idx = pixel_indices[order]
        sorted_depths = depths[order]
        unique_idx, counts = torch.unique_consecutive(sorted_idx, return_counts=True)
        start = 0
        mins = []
        for count in counts.tolist():
            end = start + count
            mins.append(sorted_depths[start:end].min())
            start = end
        if mins:
            front[unique_idx] = torch.stack(mins)
        return front


def dilate_min_depth_maps(depth_maps: torch.Tensor, radius: int) -> torch.Tensor:
    if int(radius) <= 0:
        return depth_maps
    if depth_maps.ndim == 3:
        depth_maps = depth_maps[:, None]
    valid = torch.isfinite(depth_maps) & (depth_maps > 0)
    large = torch.full_like(depth_maps, 1.0e8)
    x = torch.where(valid, depth_maps, large)
    kernel = int(radius) * 2 + 1
    x = -F.max_pool2d(-x, kernel_size=kernel, stride=1, padding=int(radius))
    return torch.where(x < 5.0e7, x, torch.full_like(x, float("inf")))[:, 0]


def visual_hull_front_depth_maps(
    masks: torch.Tensor,
    intrinsics: torch.Tensor,
    extrinsics: torch.Tensor,
    *,
    extrinsics_are_c2w: bool = True,
    camera_forward_sign: float = 1.0,
    object_to_world: Optional[torch.Tensor] = None,
    resolution: int = 48,
    coordinate_size: int = 512,
    mask_threshold: float = 0.5,
    min_visible_views: int = 1,
    min_support_views: int = 2,
    min_support_ratio: float = 0.6,
    dilation_radius: int = 3,
) -> tuple[torch.Tensor, dict]:
    """Approximate per-view z-buffer from the temporary visual hull.

    Depth maps are in the same camera-depth convention as project_points_multi_view().
    They are only used to decide which views can see a query point.
    """
    masks = resize_masks(masks, coordinate_size, masks.device)
    intrinsics = intrinsics.to(device=masks.device, dtype=torch.float32)
    extrinsics = extrinsics.to(device=masks.device, dtype=torch.float32)
    surface_coords, surface_stats = visual_hull_coords(
        masks,
        intrinsics,
        extrinsics,
        extrinsics_are_c2w=extrinsics_are_c2w,
        camera_forward_sign=camera_forward_sign,
        object_to_world=object_to_world,
        resolution=resolution,
        mask_threshold=mask_threshold,
        min_visible_views=min_visible_views,
        min_support_views=min_support_views,
        min_support_ratio=min_support_ratio,
        surface_only=True,
    )
    view_count = int(masks.shape[0])
    front_maps = torch.full(
        (view_count, coordinate_size, coordinate_size),
        float("inf"),
        device=masks.device,
        dtype=torch.float32,
    )
    if surface_coords.numel() == 0:
        return front_maps, {
            "enabled": True,
            "surface_coords": 0,
            "finite_ratio_before_dilation": 0.0,
            "finite_ratio_after_dilation": 0.0,
            "dilation_radius": int(dilation_radius),
            "surface_stats": surface_stats.to_dict(),
        }

    points_obj, _ = pixal3d_grid_points(resolution, device=masks.device, dtype=torch.float32)
    coords = surface_coords.long()
    flat_ids = coords[:, 0] * resolution * resolution + coords[:, 1] * resolution + coords[:, 2]
    surface_points = points_obj[flat_ids]
    points_2d, depths, valid_depth = project_points_multi_view(
        surface_points,
        intrinsics,
        extrinsics,
        extrinsics_are_c2w=extrinsics_are_c2w,
        camera_forward_sign=camera_forward_sign,
        object_to_world=object_to_world,
    )

    u = torch.round(points_2d[..., 0]).long()
    v = torch.round(points_2d[..., 1]).long()
    for view_idx in range(view_count):
        valid = (
            valid_depth[view_idx]
            & (u[view_idx] >= 0)
            & (u[view_idx] < coordinate_size)
            & (v[view_idx] >= 0)
            & (v[view_idx] < coordinate_size)
        )
        if not valid.any():
            continue
        pixel_ids = v[view_idx, valid] * coordinate_size + u[view_idx, valid]
        front_maps[view_idx] = _scatter_min_depth(
            pixel_ids,
            depths[view_idx, valid],
            coordinate_size * coordinate_size,
            masks.device,
        ).view(coordinate_size, coordinate_size)

    finite_before = torch.isfinite(front_maps) & (front_maps > 0)
    front_maps = dilate_min_depth_maps(front_maps, dilation_radius)
    finite_after = torch.isfinite(front_maps) & (front_maps > 0)
    stats = {
        "enabled": True,
        "surface_coords": int(surface_coords.shape[0]),
        "finite_ratio_before_dilation": float(finite_before.float().mean().item()),
        "finite_ratio_after_dilation": float(finite_after.float().mean().item()),
        "dilation_radius": int(dilation_radius),
        "surface_stats": surface_stats.to_dict(),
    }
    return front_maps, stats
