from __future__ import annotations

import math
import sys
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

TRACKER_ROOT = Path(__file__).resolve().parents[1]
if str(TRACKER_ROOT) not in sys.path:
    sys.path.insert(0, str(TRACKER_ROOT))

from trellis_point_prior_mv.build_real_slam_prior_manifest import (  # noqa: E402
    denormalize_coords_to_points,
    parse_colmap_cameras,
    parse_colmap_images,
)


def coords_xyz(coords: np.ndarray) -> np.ndarray:
    coords = np.asarray(coords, dtype=np.int32)
    if coords.size == 0:
        return np.zeros((0, 3), dtype=np.int32)
    xyz = coords[:, -3:].astype(np.int32, copy=False)
    valid = ((xyz >= 0) & (xyz < 64)).all(axis=1)
    xyz = xyz[valid]
    if xyz.shape[0] > 1:
        xyz = np.unique(xyz, axis=0)
    return xyz


def coords_with_batch(xyz: np.ndarray) -> np.ndarray:
    xyz = coords_xyz(xyz)
    batch = np.zeros((xyz.shape[0], 1), dtype=np.int32)
    return np.concatenate([batch, xyz.astype(np.int32)], axis=1)


def component_labels(coords: np.ndarray) -> tuple[np.ndarray, list[int]]:
    xyz = coords_xyz(coords)
    n = int(xyz.shape[0])
    labels = np.full((n,), -1, dtype=np.int32)
    if n == 0:
        return labels, []
    index = {tuple(p.tolist()): i for i, p in enumerate(xyz)}
    offsets = (
        (1, 0, 0),
        (-1, 0, 0),
        (0, 1, 0),
        (0, -1, 0),
        (0, 0, 1),
        (0, 0, -1),
    )
    sizes: list[int] = []
    label = 0
    for start in range(n):
        if labels[start] >= 0:
            continue
        q: deque[int] = deque([start])
        labels[start] = label
        size = 0
        while q:
            cur = q.popleft()
            size += 1
            x, y, z = xyz[cur].tolist()
            for dx, dy, dz in offsets:
                nb = (x + dx, y + dy, z + dz)
                j = index.get(nb)
                if j is not None and labels[j] < 0:
                    labels[j] = label
                    q.append(j)
        sizes.append(size)
        label += 1
    return labels, sizes


def component_metrics(prefix: str, coords: np.ndarray) -> dict[str, float | int]:
    xyz = coords_xyz(coords)
    labels, sizes = component_labels(xyz)
    count = int(xyz.shape[0])
    if count == 0:
        return {
            f"{prefix}_coord_count": 0,
            f"{prefix}_component_count": 0,
            f"{prefix}_largest_component_count": 0,
            f"{prefix}_largest_component_ratio": 0.0,
            f"{prefix}_small_component_count_lt16": 0,
            f"{prefix}_small_component_coord_ratio_lt16": 0.0,
            f"{prefix}_small_component_count_lt64": 0,
            f"{prefix}_small_component_coord_ratio_lt64": 0.0,
        }
    sorted_sizes = sorted((int(x) for x in sizes), reverse=True)
    small16 = [x for x in sorted_sizes if x < 16]
    small64 = [x for x in sorted_sizes if x < 64]
    return {
        f"{prefix}_coord_count": count,
        f"{prefix}_component_count": int(len(sorted_sizes)),
        f"{prefix}_largest_component_count": int(sorted_sizes[0]),
        f"{prefix}_largest_component_ratio": float(sorted_sizes[0] / max(count, 1)),
        f"{prefix}_small_component_count_lt16": int(len(small16)),
        f"{prefix}_small_component_coord_ratio_lt16": float(sum(small16) / max(count, 1)),
        f"{prefix}_small_component_count_lt64": int(len(small64)),
        f"{prefix}_small_component_coord_ratio_lt64": float(sum(small64) / max(count, 1)),
        f"{prefix}_top10_component_counts": ",".join(str(x) for x in sorted_sizes[:10]),
    }


def keep_largest_component(coords: np.ndarray) -> np.ndarray:
    xyz = coords_xyz(coords)
    labels, sizes = component_labels(xyz)
    if xyz.shape[0] == 0 or not sizes:
        return coords_with_batch(xyz)
    largest = int(np.argmax(np.asarray(sizes, dtype=np.int64)))
    return coords_with_batch(xyz[labels == largest])


def keep_min_component_size(coords: np.ndarray, min_size: int) -> np.ndarray:
    xyz = coords_xyz(coords)
    if min_size <= 1 or xyz.shape[0] == 0:
        return coords_with_batch(xyz)
    labels, sizes = component_labels(xyz)
    keep_labels = {i for i, size in enumerate(sizes) if size >= int(min_size)}
    keep = np.asarray([int(label) in keep_labels for label in labels], dtype=bool)
    return coords_with_batch(xyz[keep])


def nearest_prior_metrics(prefix: str, coords: np.ndarray, prior_coords: np.ndarray, radius: float) -> dict[str, float | int]:
    xyz = coords_xyz(coords).astype(np.float32)
    prior = coords_xyz(prior_coords).astype(np.float32)
    if xyz.shape[0] == 0 or prior.shape[0] == 0:
        return {
            f"{prefix}_prior_distance_enabled": 0,
            f"{prefix}_prior_radius": float(radius),
            f"{prefix}_within_prior_radius_ratio": 0.0,
        }
    try:
        from scipy.spatial import cKDTree

        dist = cKDTree(prior).query(xyz, k=1)[0]
    except Exception:
        diff = xyz[:, None, :] - prior[None, :, :]
        dist = np.sqrt(np.min(np.sum(diff * diff, axis=-1), axis=1))
    return {
        f"{prefix}_prior_distance_enabled": 1,
        f"{prefix}_prior_radius": float(radius),
        f"{prefix}_prior_distance_mean": float(np.mean(dist)),
        f"{prefix}_prior_distance_median": float(np.median(dist)),
        f"{prefix}_prior_distance_p90": float(np.percentile(dist, 90)),
        f"{prefix}_within_prior_radius_ratio": float((dist <= float(radius)).mean()),
    }


def keep_near_prior(coords: np.ndarray, prior_coords: np.ndarray, radius: float) -> np.ndarray:
    xyz = coords_xyz(coords).astype(np.float32)
    prior = coords_xyz(prior_coords).astype(np.float32)
    if xyz.shape[0] == 0 or prior.shape[0] == 0:
        return coords_with_batch(xyz.astype(np.int32))
    try:
        from scipy.spatial import cKDTree

        dist = cKDTree(prior).query(xyz, k=1)[0]
    except Exception:
        diff = xyz[:, None, :] - prior[None, :, :]
        dist = np.sqrt(np.min(np.sum(diff * diff, axis=-1), axis=1))
    return coords_with_batch(xyz[dist <= float(radius)].astype(np.int32))


def projection_support_counts(
    sample: dict[str, Any],
    coords: np.ndarray,
    *,
    grid_resolution: int = 64,
    mask_threshold: int = 127,
) -> tuple[np.ndarray, np.ndarray, int]:
    xyz = coords_xyz(coords)
    if xyz.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32), 0
    dataset_dir = Path(sample["dataset_root"])
    sparse_dir = dataset_dir / str(sample.get("sparse_subdir", "sparse/0"))
    cameras = parse_colmap_cameras(sparse_dir / "cameras.txt")
    images = parse_colmap_images(sparse_dir / "images.txt")
    if not cameras or not images:
        return np.zeros((xyz.shape[0],), dtype=np.float32), np.zeros((xyz.shape[0],), dtype=np.float32), 0
    points = denormalize_coords_to_points(xyz, sample.get("normalization") or {}, int(grid_resolution))
    support = np.zeros((xyz.shape[0],), dtype=np.float32)
    in_image = np.zeros((xyz.shape[0],), dtype=np.float32)
    matched = 0
    for frame in sample.get("frames") or []:
        meta = images.get(frame.get("name", "")) or images.get(Path(frame.get("name", "")).name)
        if meta is None:
            continue
        camera = cameras.get(int(meta["camera_id"]))
        if camera is None:
            continue
        mask = Image.open(frame["mask"]).convert("L")
        size = (int(camera["width"]), int(camera["height"]))
        if mask.size != size:
            mask = mask.resize(size, Image.Resampling.NEAREST)
        mask_np = np.asarray(mask)
        xyz_cam = (meta["R"] @ points.astype(np.float64).T).T + meta["tvec"][None, :]
        z = xyz_cam[:, 2]
        valid = z > 1e-6
        u = camera["fx"] * (xyz_cam[:, 0] / np.maximum(z, 1e-6)) + camera["cx"]
        v = camera["fy"] * (xyz_cam[:, 1] / np.maximum(z, 1e-6)) + camera["cy"]
        ui = np.rint(u).astype(np.int64)
        vi = np.rint(v).astype(np.int64)
        inside = valid & (ui >= 0) & (ui < size[0]) & (vi >= 0) & (vi < size[1])
        inside_ids = np.where(inside)[0]
        if inside_ids.size:
            in_image[inside_ids] += 1.0
            xi = np.clip(ui[inside_ids], 0, mask_np.shape[1] - 1)
            yi = np.clip(vi[inside_ids], 0, mask_np.shape[0] - 1)
            hit = mask_np[yi, xi] > int(mask_threshold)
            support[inside_ids[hit]] += 1.0
        matched += 1
    return support, in_image, int(matched)


def projection_support_metrics(
    prefix: str,
    sample: dict[str, Any],
    coords: np.ndarray,
    *,
    grid_resolution: int = 64,
    mask_threshold: int = 127,
    min_support_views: int = 1,
    min_support_ratio: float = 0.0,
    visual_hull_min_visible_views: int = 1,
    visual_hull_min_support_ratio: float = 0.0,
) -> dict[str, float | int]:
    support, in_image, matched = projection_support_counts(
        sample,
        coords,
        grid_resolution=grid_resolution,
        mask_threshold=mask_threshold,
    )
    count = int(support.shape[0])
    required = max(int(min_support_views), int(math.ceil(float(min_support_ratio) * max(matched, 1))))
    visible = in_image > 0
    visible_count = int(visible.sum()) if count else 0
    outside_visible = np.maximum(in_image - support, 0.0)
    visible_event_count = float(in_image.sum()) if count else 0.0
    visible_outside_event_count = float(outside_visible.sum()) if count else 0.0
    support_over_visible = np.zeros_like(support, dtype=np.float32)
    if count:
        np.divide(support, np.maximum(in_image, 1e-6), out=support_over_visible, where=in_image > 0)
    vh_visible_req = max(int(visual_hull_min_visible_views), 1)
    vh_ratio_req = float(visual_hull_min_support_ratio)
    visual_hull_keep = visible & (in_image >= float(vh_visible_req)) & (support_over_visible >= vh_ratio_req)
    return {
        f"{prefix}_projection_enabled": int(matched > 0),
        f"{prefix}_projection_matched_frame_count": int(matched),
        f"{prefix}_projection_required_support": int(required),
        f"{prefix}_projection_any_in_image_ratio": float((in_image > 0).mean()) if count else 0.0,
        f"{prefix}_projection_any_mask_hit_ratio": float((support > 0).mean()) if count else 0.0,
        f"{prefix}_projection_support_mean": float(support.mean()) if count else 0.0,
        f"{prefix}_projection_support_median": float(np.median(support)) if count else 0.0,
        f"{prefix}_projection_keep_ratio": float((support >= required).mean()) if count else 0.0,
        f"{prefix}_projection_kept_count": int((support >= required).sum()) if count else 0,
        f"{prefix}_projection_visible_mean": float(in_image.mean()) if count else 0.0,
        f"{prefix}_projection_visible_median": float(np.median(in_image)) if count else 0.0,
        f"{prefix}_visible_point_count": int(visible_count),
        f"{prefix}_visible_point_ratio": float(visible_count / max(count, 1)),
        f"{prefix}_visible_support_ratio_mean": float(support_over_visible[visible].mean()) if visible_count else 0.0,
        f"{prefix}_visible_support_ratio_median": float(np.median(support_over_visible[visible])) if visible_count else 0.0,
        f"{prefix}_visible_outside_mask_event_ratio": (
            float(visible_outside_event_count / max(visible_event_count, 1.0)) if visible_event_count > 0 else 0.0
        ),
        f"{prefix}_visible_outside_mask_point_mean": (
            float((outside_visible[visible] / np.maximum(in_image[visible], 1e-6)).mean()) if visible_count else 0.0
        ),
        f"{prefix}_visual_hull_min_visible_views": int(vh_visible_req),
        f"{prefix}_visual_hull_min_support_ratio": float(vh_ratio_req),
        f"{prefix}_visual_hull_keep_ratio": float(visual_hull_keep.mean()) if count else 0.0,
        f"{prefix}_visual_hull_kept_count": int(visual_hull_keep.sum()) if count else 0,
    }


def keep_projection_supported(
    sample: dict[str, Any],
    coords: np.ndarray,
    *,
    grid_resolution: int = 64,
    mask_threshold: int = 127,
    min_support_views: int = 1,
    min_support_ratio: float = 0.0,
    visual_hull_min_visible_views: int = 1,
    visual_hull_min_support_ratio: float = 0.0,
) -> np.ndarray:
    xyz = coords_xyz(coords)
    support, _in_image, matched = projection_support_counts(
        sample,
        xyz,
        grid_resolution=grid_resolution,
        mask_threshold=mask_threshold,
    )
    if xyz.shape[0] == 0 or matched == 0:
        return coords_with_batch(xyz)
    required = max(int(min_support_views), int(math.ceil(float(min_support_ratio) * max(matched, 1))))
    keep = support >= required
    if int(visual_hull_min_visible_views) > 1 or float(visual_hull_min_support_ratio) > 0:
        visible = _in_image > 0
        support_over_visible = np.zeros_like(support, dtype=np.float32)
        np.divide(support, np.maximum(_in_image, 1e-6), out=support_over_visible, where=_in_image > 0)
        keep = keep & visible & (_in_image >= float(max(int(visual_hull_min_visible_views), 1))) & (
            support_over_visible >= float(visual_hull_min_support_ratio)
        )
    return coords_with_batch(xyz[keep])


def filter_sparse_coords(
    coords: np.ndarray,
    prior_coords: np.ndarray,
    sample: dict[str, Any],
    *,
    filter_spec: str,
    prior_radius: float = 4.0,
    min_component_size: int = 0,
    min_support_views: int = 1,
    min_support_ratio: float = 0.0,
    visual_hull_min_visible_views: int = 1,
    visual_hull_min_support_ratio: float = 0.0,
    grid_resolution: int = 64,
    mask_threshold: int = 127,
) -> tuple[np.ndarray, dict[str, float | int | str]]:
    spec = [part.strip() for part in str(filter_spec or "none").split(",") if part.strip()]
    if not spec:
        spec = ["none"]
    out = coords_with_batch(coords)
    metrics: dict[str, float | int | str] = {
        "sparse_filter_spec": ",".join(spec),
        "sparse_filter_input_count": int(coords_xyz(coords).shape[0]),
    }
    for part in spec:
        before = int(coords_xyz(out).shape[0])
        if part in {"none", "raw"}:
            after_coords = out
        elif part in {"largest", "largest_component"}:
            after_coords = keep_largest_component(out)
        elif part in {"min_component", "min_component_size"}:
            after_coords = keep_min_component_size(out, int(min_component_size))
        elif part in {"prior", "prior_radius", "near_prior"}:
            after_coords = keep_near_prior(out, prior_coords, float(prior_radius))
        elif part in {"projection", "projection_support", "mask_support"}:
            after_coords = keep_projection_supported(
                sample,
                out,
                grid_resolution=int(grid_resolution),
                mask_threshold=int(mask_threshold),
                min_support_views=int(min_support_views),
                min_support_ratio=float(min_support_ratio),
                visual_hull_min_visible_views=int(visual_hull_min_visible_views),
                visual_hull_min_support_ratio=float(visual_hull_min_support_ratio),
            )
        else:
            raise ValueError(f"unsupported sparse filter step {part!r} in {filter_spec!r}")
        after = int(coords_xyz(after_coords).shape[0])
        metrics[f"sparse_filter_{part}_before"] = before
        metrics[f"sparse_filter_{part}_after"] = after
        metrics[f"sparse_filter_{part}_keep_ratio"] = float(after / max(before, 1))
        out = after_coords
    metrics["sparse_filter_output_count"] = int(coords_xyz(out).shape[0])
    metrics["sparse_filter_total_keep_ratio"] = float(metrics["sparse_filter_output_count"] / max(int(metrics["sparse_filter_input_count"]), 1))
    return out, metrics


def sparse_diagnostic_metrics(
    prefix: str,
    coords: np.ndarray,
    prior_coords: np.ndarray,
    sample: dict[str, Any],
    *,
    prior_radius: float = 4.0,
    min_support_views: int = 1,
    min_support_ratio: float = 0.0,
    visual_hull_min_visible_views: int = 1,
    visual_hull_min_support_ratio: float = 0.0,
    grid_resolution: int = 64,
    mask_threshold: int = 127,
) -> dict[str, float | int | str]:
    out: dict[str, float | int | str] = {}
    out.update(component_metrics(prefix, coords))
    out.update(nearest_prior_metrics(prefix, coords, prior_coords, prior_radius))
    out.update(
        projection_support_metrics(
            prefix,
            sample,
            coords,
            grid_resolution=grid_resolution,
            mask_threshold=mask_threshold,
            min_support_views=min_support_views,
            min_support_ratio=min_support_ratio,
            visual_hull_min_visible_views=visual_hull_min_visible_views,
            visual_hull_min_support_ratio=visual_hull_min_support_ratio,
        )
    )
    return out
