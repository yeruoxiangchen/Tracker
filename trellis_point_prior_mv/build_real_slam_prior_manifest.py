#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

TRACKER_ROOT = Path(__file__).resolve().parents[1]
if str(TRACKER_ROOT) not in sys.path:
    sys.path.insert(0, str(TRACKER_ROOT))

from trellis_point_prior_mv.common import write_json  # noqa: E402


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
    q = np.asarray(qvec, dtype=np.float64)
    q = q / max(np.linalg.norm(q), 1e-12)
    w, x, y, z = q.tolist()
    return np.asarray(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
            [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
            [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
        ],
        dtype=np.float64,
    )


def parse_colmap_cameras(path: Path) -> dict[int, dict]:
    cameras: dict[int, dict] = {}
    if not path.exists():
        return cameras
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        camera_id = int(parts[0])
        model = parts[1]
        width, height = int(parts[2]), int(parts[3])
        params = [float(x) for x in parts[4:]]
        if model == "PINHOLE":
            fx, fy, cx, cy = params[:4]
        elif model == "SIMPLE_PINHOLE":
            fx = fy = params[0]
            cx, cy = params[1:3]
        elif model == "SIMPLE_RADIAL":
            fx = fy = params[0]
            cx, cy = params[1:3]
        else:
            raise ValueError(f"unsupported COLMAP camera model {model!r} in {path}")
        cameras[camera_id] = {
            "camera_id": camera_id,
            "model": model,
            "width": width,
            "height": height,
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
        }
    return cameras


def parse_colmap_images(path: Path) -> dict[str, dict]:
    images: dict[str, dict] = {}
    if not path.exists():
        return images
    raw = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    i = 0
    while i < len(raw):
        line = raw[i].strip()
        i += 1
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 10:
            continue
        image_id = int(parts[0])
        qvec = np.asarray([float(x) for x in parts[1:5]], dtype=np.float64)
        tvec = np.asarray([float(x) for x in parts[5:8]], dtype=np.float64)
        camera_id = int(parts[8])
        name = " ".join(parts[9:])
        images[name] = {
            "image_id": image_id,
            "camera_id": camera_id,
            "qvec": qvec,
            "tvec": tvec,
            "R": qvec_to_rotmat(qvec),
        }
        # Skip POINTS2D line.
        if i < len(raw):
            i += 1
    return images


def parse_colmap_points(path: Path) -> np.ndarray:
    if not path.exists():
        return np.zeros((0, 3), dtype=np.float32)
    pts = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        pts.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.asarray(pts, dtype=np.float32)


def camera_center_from_meta(meta: dict) -> np.ndarray:
    return -np.asarray(meta["R"], dtype=np.float64).T @ np.asarray(meta["tvec"], dtype=np.float64).reshape(3)


def pose_diverse_frame_indices(frames: list[dict], metas: dict[str, dict], count: int, seed: int, randomized: bool) -> list[int]:
    if count <= 0 or len(frames) <= count:
        return list(range(len(frames)))
    centers = []
    valid_ids = []
    for idx, frame in enumerate(frames):
        meta = metas.get(frame["name"]) or metas.get(Path(frame["name"]).name)
        if meta is None:
            continue
        centers.append(camera_center_from_meta(meta))
        valid_ids.append(idx)
    if len(valid_ids) < count:
        return list(range(min(count, len(frames))))
    centers_np = np.stack(centers, axis=0).astype(np.float64)
    center = np.median(centers_np, axis=0, keepdims=True)
    rel = centers_np - center
    norm = np.linalg.norm(rel, axis=1, keepdims=True)
    if float(norm.max()) < 1e-8:
        rel = np.arange(len(valid_ids), dtype=np.float64).reshape(-1, 1)
        norm = np.maximum(np.linalg.norm(rel, axis=1, keepdims=True), 1.0)
    feat = rel / np.maximum(norm, 1e-8)
    rng = np.random.default_rng(int(seed))
    first_local = int(rng.integers(0, len(valid_ids))) if randomized else 0
    selected_local = [first_local]
    min_dist = np.sum((feat - feat[first_local]) ** 2, axis=1)
    min_dist[first_local] = -1.0
    while len(selected_local) < count:
        if randomized:
            order = np.argsort(-min_dist)
            topn = max(1, min(len(order), max(3, int(math.ceil(count * 0.5)))))
            pool = order[:topn]
            weights = np.maximum(min_dist[pool], 0.0)
            weights = weights / weights.sum() if float(weights.sum()) > 0 else None
            nxt = int(rng.choice(pool, p=weights))
        else:
            nxt = int(np.argmax(min_dist))
        if nxt in selected_local or min_dist[nxt] < 0:
            break
        selected_local.append(nxt)
        dist = np.sum((feat - feat[nxt]) ** 2, axis=1)
        min_dist = np.minimum(min_dist, dist)
        min_dist[selected_local] = -1.0
    return sorted(valid_ids[i] for i in selected_local)


def select_frames(
    frames: list[dict],
    max_frames: int,
    frame_select: str,
    frame_stride: int,
    frame_select_seed: int,
    metas: dict[str, dict] | None = None,
) -> list[dict]:
    if max_frames <= 0 or len(frames) <= max_frames:
        return frames
    if frame_select == "uniform":
        ids = np.linspace(0, len(frames) - 1, int(max_frames))
        keep = sorted({int(round(x)) for x in ids})
        return [frames[i] for i in keep][: int(max_frames)]
    if frame_select == "stride":
        return frames[:: max(int(frame_stride), 1)][: int(max_frames)]
    if frame_select == "random":
        rng = np.random.default_rng(int(frame_select_seed))
        keep = sorted(rng.choice(len(frames), size=int(max_frames), replace=False).astype(int).tolist())
        return [frames[i] for i in keep]
    if frame_select == "random_uniform":
        rng = np.random.default_rng(int(frame_select_seed))
        edges = np.linspace(0, len(frames), int(max_frames) + 1)
        keep = []
        for start, end in zip(edges[:-1], edges[1:]):
            lo = int(math.floor(start))
            hi = max(lo + 1, int(math.ceil(end)))
            hi = min(hi, len(frames))
            keep.append(int(rng.integers(lo, hi)))
        keep = sorted(set(keep))
        if len(keep) < int(max_frames):
            rest = [i for i in range(len(frames)) if i not in keep]
            extra = rng.choice(rest, size=min(int(max_frames) - len(keep), len(rest)), replace=False).astype(int).tolist()
            keep = sorted(keep + extra)
        return [frames[i] for i in keep[: int(max_frames)]]
    if frame_select == "pose_farthest" and metas is not None:
        keep = pose_diverse_frame_indices(frames, metas, int(max_frames), int(frame_select_seed), randomized=False)
        return [frames[i] for i in keep][: int(max_frames)]
    if frame_select == "pose_random_farthest" and metas is not None:
        keep = pose_diverse_frame_indices(frames, metas, int(max_frames), int(frame_select_seed), randomized=True)
        return [frames[i] for i in keep][: int(max_frames)]
    return frames[: int(max_frames)]


def find_frame_pairs(
    dataset_dir: Path,
    max_frames: int,
    frame_select: str = "first",
    frame_stride: int = 1,
    frame_select_seed: int = 42,
    sparse_subdir: str = "sparse/0",
) -> list[dict]:
    image_dir = dataset_dir / "images"
    if not image_dir.exists():
        image_dir = dataset_dir / "rgb"
    mask_dir = dataset_dir / "masks"
    if not image_dir.exists() or not mask_dir.exists():
        raise FileNotFoundError(f"missing images/rgb or masks dir under {dataset_dir}")
    image_paths = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    frames = []
    for image_path in image_paths:
        mask_path = mask_dir / f"{image_path.stem}.png"
        if not mask_path.exists():
            continue
        frames.append(
            {
                "image": str(image_path),
                "mask": str(mask_path),
                "name": image_path.name,
                "stem": image_path.stem,
            }
        )
    metas = parse_colmap_images(dataset_dir / sparse_subdir / "images.txt") if str(frame_select).startswith("pose_") else None
    return select_frames(frames, int(max_frames), frame_select, int(frame_stride), int(frame_select_seed), metas)


def find_reference_model(dataset_dir: Path) -> Path | None:
    model_dir = dataset_dir / "models"
    if not model_dir.exists():
        return None
    norm = sorted(model_dir.glob("*_norm.obj"))
    if norm:
        return norm[0]
    objs = sorted(model_dir.glob("*.obj"))
    return objs[0] if objs else None


def normalize_points_to_coords(
    points: np.ndarray,
    confidence: np.ndarray | None = None,
    *,
    bbox_points: np.ndarray | None = None,
    bbox_scale: float,
    grid_resolution: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    points = np.asarray(points, dtype=np.float32)
    if confidence is not None:
        confidence = np.asarray(confidence, dtype=np.float32).reshape(-1)
        if confidence.shape[0] != points.shape[0]:
            confidence = None
    if points.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.int32), np.zeros((0,), dtype=np.float32), {
            "center": [0.0, 0.0, 0.0],
            "scale": 1.0,
            "bbox_min": [0.0, 0.0, 0.0],
            "bbox_max": [0.0, 0.0, 0.0],
            "input_point_count": 0,
            "valid_input_point_count": 0,
            "bbox_point_count": 0,
        }
    bbox = np.asarray(bbox_points, dtype=np.float32) if bbox_points is not None else points
    if bbox.shape[0] == 0:
        bbox = points
    pmin = bbox.min(axis=0)
    pmax = bbox.max(axis=0)
    center = (pmin + pmax) * 0.5
    extent = pmax - pmin
    scale = float(max(float(extent.max()) * float(bbox_scale), 1e-6))
    normalized = (points - center[None, :]) / scale
    coords = np.floor((normalized + 0.5) * float(grid_resolution)).astype(np.int32)
    valid = ((coords >= 0) & (coords < grid_resolution)).all(axis=1)
    coords = coords[valid]
    point_conf = confidence[valid] if confidence is not None else np.ones((coords.shape[0],), dtype=np.float32)
    if coords.shape[0] > 0:
        coords, inverse = np.unique(coords, axis=0, return_inverse=True)
        conf_sum = np.zeros((coords.shape[0],), dtype=np.float64)
        conf_count = np.zeros((coords.shape[0],), dtype=np.float64)
        np.add.at(conf_sum, inverse, point_conf.astype(np.float64))
        np.add.at(conf_count, inverse, 1.0)
        coord_conf = (conf_sum / np.maximum(conf_count, 1.0)).astype(np.float32)
    else:
        coord_conf = np.zeros((0,), dtype=np.float32)
    return coords.astype(np.int32), coord_conf, {
        "center": [float(x) for x in center.tolist()],
        "scale": scale,
        "bbox_min": [float(x) for x in pmin.tolist()],
        "bbox_max": [float(x) for x in pmax.tolist()],
        "input_point_count": int(points.shape[0]),
        "valid_input_point_count": int(valid.sum()),
        "bbox_point_count": int(bbox.shape[0]),
    }


def denormalize_coords_to_points(coords: np.ndarray, norm: dict, grid_resolution: int) -> np.ndarray:
    coords = np.asarray(coords, dtype=np.float32)
    if coords.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float32)
    xyz = coords[:, -3:] if coords.shape[1] >= 3 else coords.reshape(-1, 3)
    normalized = (xyz + 0.5) / float(grid_resolution) - 0.5
    center = np.asarray(norm.get("center", [0.0, 0.0, 0.0]), dtype=np.float32).reshape(1, 3)
    scale = float(norm.get("scale", 1.0))
    return (normalized * scale + center).astype(np.float32)


def project_mask_filter_colmap_points(dataset_dir: Path, frames: list[dict], points: np.ndarray, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, dict]:
    sparse_dir = dataset_dir / args.sparse_subdir
    cameras = parse_colmap_cameras(sparse_dir / "cameras.txt")
    images = parse_colmap_images(sparse_dir / "images.txt")
    if points.shape[0] == 0 or not cameras or not images:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.float32), {
            "colmap_point_count": int(points.shape[0]),
            "support_filtered_count": 0,
            "matched_frame_count": 0,
        }

    support = np.zeros((points.shape[0],), dtype=np.float32)
    matched = 0
    for frame in frames:
        meta = images.get(frame["name"]) or images.get(Path(frame["name"]).name)
        if meta is None:
            continue
        camera = cameras.get(int(meta["camera_id"]))
        if camera is None:
            continue
        mask = Image.open(frame["mask"]).convert("L")
        if mask.size != (int(camera["width"]), int(camera["height"])):
            mask = mask.resize((int(camera["width"]), int(camera["height"])), Image.Resampling.NEAREST)
        mask_np = np.asarray(mask)
        xyz_cam = (meta["R"] @ points.astype(np.float64).T).T + meta["tvec"][None, :]
        z = xyz_cam[:, 2]
        valid = z > 1e-6
        u = camera["fx"] * (xyz_cam[:, 0] / np.maximum(z, 1e-6)) + camera["cx"]
        v = camera["fy"] * (xyz_cam[:, 1] / np.maximum(z, 1e-6)) + camera["cy"]
        ui = np.rint(u).astype(np.int64)
        vi = np.rint(v).astype(np.int64)
        inside = valid & (ui >= 0) & (ui < int(camera["width"])) & (vi >= 0) & (vi < int(camera["height"]))
        ids = np.where(inside)[0]
        if ids.size:
            xi = np.clip(ui[ids], 0, mask_np.shape[1] - 1)
            yi = np.clip(vi[ids], 0, mask_np.shape[0] - 1)
            keep = mask_np[yi, xi] > int(args.mask_threshold)
            support[ids[keep]] += 1.0
        matched += 1
    min_support = max(float(args.min_support_views), math.ceil(float(args.min_support_ratio) * max(matched, 1)))
    keep = support >= min_support
    selected = points[keep]
    conf = support[keep] / max(float(matched), 1.0)
    return selected.astype(np.float32), conf.astype(np.float32), {
        "colmap_point_count": int(points.shape[0]),
        "support_filtered_count": int(selected.shape[0]),
        "matched_frame_count": int(matched),
        "min_support_required": float(min_support),
        "support_mean": float(support[keep].mean()) if selected.shape[0] else 0.0,
    }


def sample_model_surface(model_path: Path, point_count: int, seed: int) -> tuple[np.ndarray, np.ndarray, dict]:
    try:
        import trimesh
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"trimesh is required for model_surface prior: {exc}") from exc
    mesh = trimesh.load(model_path, force="mesh", process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(g for g in mesh.geometry.values()))
    rng = np.random.default_rng(seed)
    count = max(int(point_count) * 4, int(point_count))
    try:
        points, _ = trimesh.sample.sample_surface(mesh, count)
    except Exception:
        verts = np.asarray(mesh.vertices, dtype=np.float32)
        replace = verts.shape[0] < count
        ids = rng.choice(verts.shape[0], size=count, replace=replace)
        points = verts[ids]
    if points.shape[0] > point_count:
        ids = rng.choice(points.shape[0], size=point_count, replace=False)
        points = points[np.sort(ids)]
    conf = np.ones((points.shape[0],), dtype=np.float32)
    return np.asarray(points, dtype=np.float32), conf, {
        "model_surface_count": int(points.shape[0]),
        "model_path": str(model_path),
    }


def load_model_bbox_points(model_path: Path) -> tuple[np.ndarray, dict]:
    try:
        import trimesh
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"trimesh is required for model_bbox normalization: {exc}") from exc
    mesh = trimesh.load(model_path, force="mesh", process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(g for g in mesh.geometry.values()))
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    if vertices.shape[0] == 0:
        raise ValueError(f"reference model has no vertices: {model_path}")
    return vertices, {
        "model_bbox_vertex_count": int(vertices.shape[0]),
        "model_bbox_path": str(model_path),
    }


def choose_normalization_bbox(
    selected_points: np.ndarray,
    model_path: Path | None,
    args: argparse.Namespace,
) -> tuple[np.ndarray, str, dict]:
    source = str(args.normalization_source)
    if source == "auto":
        source = "model_bbox" if model_path is not None else "prior_bbox"
    if source == "model_bbox":
        if model_path is None:
            raise FileNotFoundError("--normalization_source model_bbox requires a reference model")
        bbox_points, stats = load_model_bbox_points(model_path)
        return bbox_points, source, stats
    if source == "prior_bbox":
        return selected_points, source, {"prior_bbox_point_count": int(selected_points.shape[0])}
    raise ValueError(f"unsupported normalization_source={source!r}")


def subsample_coords(coords: np.ndarray, conf: np.ndarray, point_count: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if coords.shape[0] <= point_count:
        return coords, conf
    rng = np.random.default_rng(seed)
    weights = conf.astype(np.float64)
    weights = weights / weights.sum() if weights.sum() > 0 else None
    ids = rng.choice(coords.shape[0], size=int(point_count), replace=False, p=weights)
    ids = np.sort(ids)
    return coords[ids], conf[ids]


def projection_diagnostics_for_coords(
    dataset_dir: Path,
    frames: list[dict],
    coords: np.ndarray,
    norm: dict,
    args: argparse.Namespace,
) -> dict[str, float | int]:
    sparse_dir = dataset_dir / args.sparse_subdir
    cameras = parse_colmap_cameras(sparse_dir / "cameras.txt")
    images = parse_colmap_images(sparse_dir / "images.txt")
    points = denormalize_coords_to_points(coords, norm, int(args.grid_resolution))
    base: dict[str, float | int] = {
        "projection_enabled": 0,
        "projection_point_count": int(points.shape[0]),
        "projection_matched_frame_count": 0,
    }
    if points.shape[0] == 0 or not cameras or not images:
        return base

    support = np.zeros((points.shape[0],), dtype=np.float32)
    in_image_seen = np.zeros((points.shape[0],), dtype=np.float32)
    in_image_ratios: list[float] = []
    mask_hit_over_points: list[float] = []
    mask_hit_over_inside: list[float] = []
    matched = 0
    for frame in frames:
        meta = images.get(frame["name"]) or images.get(Path(frame["name"]).name)
        if meta is None:
            continue
        camera = cameras.get(int(meta["camera_id"]))
        if camera is None:
            continue
        mask = Image.open(frame["mask"]).convert("L")
        if mask.size != (int(camera["width"]), int(camera["height"])):
            mask = mask.resize((int(camera["width"]), int(camera["height"])), Image.Resampling.NEAREST)
        mask_np = np.asarray(mask)
        xyz_cam = (meta["R"] @ points.astype(np.float64).T).T + meta["tvec"][None, :]
        z = xyz_cam[:, 2]
        valid = z > 1e-6
        u = camera["fx"] * (xyz_cam[:, 0] / np.maximum(z, 1e-6)) + camera["cx"]
        v = camera["fy"] * (xyz_cam[:, 1] / np.maximum(z, 1e-6)) + camera["cy"]
        ui = np.rint(u).astype(np.int64)
        vi = np.rint(v).astype(np.int64)
        inside = valid & (ui >= 0) & (ui < int(camera["width"])) & (vi >= 0) & (vi < int(camera["height"]))
        inside_ids = np.where(inside)[0]
        hit_ids = np.zeros((0,), dtype=np.int64)
        if inside_ids.size:
            xi = np.clip(ui[inside_ids], 0, mask_np.shape[1] - 1)
            yi = np.clip(vi[inside_ids], 0, mask_np.shape[0] - 1)
            hit = mask_np[yi, xi] > int(args.mask_threshold)
            hit_ids = inside_ids[hit]
            support[hit_ids] += 1.0
            in_image_seen[inside_ids] += 1.0
        in_image_ratios.append(float(inside_ids.size) / float(points.shape[0]))
        mask_hit_over_points.append(float(hit_ids.size) / float(points.shape[0]))
        mask_hit_over_inside.append(float(hit_ids.size) / float(max(inside_ids.size, 1)))
        matched += 1

    if matched == 0:
        return base
    return {
        "projection_enabled": 1,
        "projection_point_count": int(points.shape[0]),
        "projection_matched_frame_count": int(matched),
        "projection_in_image_ratio_mean": float(np.mean(in_image_ratios)),
        "projection_mask_hit_over_points_mean": float(np.mean(mask_hit_over_points)),
        "projection_mask_hit_over_inside_mean": float(np.mean(mask_hit_over_inside)),
        "projection_any_in_image_ratio": float((in_image_seen > 0).mean()),
        "projection_any_mask_hit_ratio": float((support > 0).mean()),
        "projection_support_mean": float(support.mean()),
        "projection_support_median": float(np.median(support)),
    }


def build_one(dataset_dir: Path, out_dir: Path, args: argparse.Namespace, out_index: int) -> tuple[dict, dict]:
    uid = dataset_dir.name
    frames = find_frame_pairs(
        dataset_dir,
        max_frames=args.max_frames,
        frame_select=args.frame_select,
        frame_stride=args.frame_stride,
        frame_select_seed=args.frame_select_seed,
        sparse_subdir=args.sparse_subdir,
    )
    if not frames:
        raise ValueError(f"no image/mask pairs found in {dataset_dir}")
    model_path = find_reference_model(dataset_dir)
    sparse_dir = dataset_dir / args.sparse_subdir
    colmap_points = parse_colmap_points(sparse_dir / "points3D.txt")
    selected_points = np.zeros((0, 3), dtype=np.float32)
    selected_conf = np.zeros((0,), dtype=np.float32)
    source_used = args.prior_source
    stats: dict[str, Any] = {}
    fallback_used = False
    fallback_reason = ""

    if args.prior_source in {"auto", "colmap_points"}:
        selected_points, selected_conf, stats = project_mask_filter_colmap_points(dataset_dir, frames, colmap_points, args)
        source_used = "colmap_points"
        if selected_points.shape[0] < int(args.min_prior_points):
            fallback_reason = (
                f"COLMAP/SLAM supported points {selected_points.shape[0]} "
                f"< min_prior_points {int(args.min_prior_points)}"
            )
            if args.prior_source == "colmap_points":
                raise ValueError(f"{fallback_reason}; dataset={dataset_dir}")
            if not bool(args.allow_model_fallback):
                raise ValueError(
                    f"{fallback_reason}; dataset={dataset_dir}. "
                    "Use --allow_model_fallback for smoke only, or --prior_source model_surface explicitly."
                )

    if args.prior_source == "model_surface" or (
        args.prior_source == "auto" and selected_points.shape[0] < int(args.min_prior_points)
    ):
        if model_path is None:
            raise FileNotFoundError(f"no reference model found for fallback model_surface: {dataset_dir}")
        fallback_used = args.prior_source == "auto"
        selected_points, selected_conf, model_stats = sample_model_surface(model_path, int(args.point_count), int(args.seed) + out_index * 1009)
        stats.update(model_stats)
        source_used = "model_surface"

    bbox_points, normalization_source, norm_stats = choose_normalization_bbox(selected_points, model_path, args)
    stats.update(norm_stats)
    coords, coord_conf, norm = normalize_points_to_coords(
        selected_points,
        selected_conf,
        bbox_points=bbox_points,
        bbox_scale=args.bbox_scale,
        grid_resolution=args.grid_resolution,
    )
    norm["source"] = normalization_source
    if coords.shape[0] == 0:
        raise ValueError(f"prior coords empty for {dataset_dir}")
    coords, coord_conf = subsample_coords(coords, coord_conf, int(args.point_count), int(args.seed) + out_index * 3571)
    if source_used == "colmap_points" and coords.shape[0] < int(args.min_prior_points):
        raise ValueError(
            f"voxelized prior coords {coords.shape[0]} < min_prior_points {int(args.min_prior_points)}; "
            f"source_points={selected_points.shape[0]}; dataset={dataset_dir}"
        )
    projection_stats = projection_diagnostics_for_coords(dataset_dir, frames, coords, norm, args)
    stats.update(projection_stats)

    prior_dir = out_dir / "priors"
    prior_dir.mkdir(parents=True, exist_ok=True)
    prior_path = prior_dir / f"{uid}.npz"
    np.savez_compressed(
        prior_path,
        prior_coords=coords.astype(np.int32),
        prior_conf=coord_conf.astype(np.float32),
        source_points=np.asarray(selected_points, dtype=np.float32),
        source_conf=np.asarray(selected_conf, dtype=np.float32),
        normalization_center=np.asarray(norm["center"], dtype=np.float32),
        normalization_scale=np.asarray([norm["scale"]], dtype=np.float32),
        normalization_bbox_min=np.asarray(norm["bbox_min"], dtype=np.float32),
        normalization_bbox_max=np.asarray(norm["bbox_max"], dtype=np.float32),
    )
    sample = {
        "uid": uid,
        "dataset_root": str(dataset_dir),
        "sparse_subdir": str(args.sparse_subdir),
        "prior_npz": str(prior_path.relative_to(out_dir)),
        "prior_source": source_used,
        "reference_model": str(model_path) if model_path is not None else None,
        "frames": frames,
        "normalization": norm,
        "fallback_used": bool(fallback_used),
        "fallback_reason": fallback_reason,
        "projection_diagnostics": projection_stats,
        "prior_point_count": int(coords.shape[0]),
        "source_point_count": int(selected_points.shape[0]),
    }
    row = {
        "uid": uid,
        "dataset_root": str(dataset_dir),
        "sparse_subdir": str(args.sparse_subdir),
        "prior_source": source_used,
        "frame_count": len(frames),
        "prior_point_count": int(coords.shape[0]),
        "source_point_count": int(selected_points.shape[0]),
        "reference_model": str(model_path) if model_path is not None else "",
        "fallback_used": int(bool(fallback_used)),
        "fallback_reason": fallback_reason,
        "normalization_source": normalization_source,
        "normalization_scale": float(norm["scale"]),
        **stats,
    }
    return sample, row


def build(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_dirs = [Path(p) for p in args.datasets]
    samples = []
    rows = []
    for i, dataset_dir in enumerate(dataset_dirs):
        sample, row = build_one(dataset_dir, output_dir, args, i)
        samples.append(sample)
        rows.append(row)
        print(
            f"[real_slam_prior] {dataset_dir.name} source={row['prior_source']} "
            f"norm={row['normalization_source']} fallback={row['fallback_used']} "
            f"frames={row['frame_count']} prior={row['prior_point_count']} source_points={row['source_point_count']} "
            f"proj_mask_any={row.get('projection_any_mask_hit_ratio', 0.0):.3f}",
            flush=True,
        )
    manifest = {
        "format": "trellis_real_slam_prior_v2",
        "output_dir": str(output_dir),
        "prior_root": str(output_dir),
        "samples": samples,
        "build_args": vars(args),
        "summary": {
            "num_samples": len(samples),
            "prior_point_mean": float(np.mean([r["prior_point_count"] for r in rows])) if rows else 0.0,
        },
    }
    write_json(output_dir / "manifest.json", manifest)
    write_json(output_dir / "build_report.json", {"rows": rows, "summary": manifest["summary"]})
    write_csv(output_dir / "build_report.csv", rows)
    print(f"[real_slam_prior] wrote {output_dir / 'manifest.json'}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build real CoarseModel/COLMAP or model-surface point priors for TRELLIS Stage2.")
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--prior_source", choices=["auto", "colmap_points", "model_surface"], default="auto")
    parser.add_argument("--sparse_subdir", default="sparse/0")
    parser.add_argument("--allow_model_fallback", action="store_true")
    parser.add_argument("--normalization_source", choices=["auto", "prior_bbox", "model_bbox"], default="auto")
    parser.add_argument("--max_frames", type=int, default=18)
    parser.add_argument(
        "--frame_select",
        choices=["first", "uniform", "stride", "random", "random_uniform", "pose_farthest", "pose_random_farthest"],
        default="first",
    )
    parser.add_argument("--frame_stride", type=int, default=1)
    parser.add_argument("--frame_select_seed", type=int, default=42)
    parser.add_argument("--point_count", type=int, default=1500)
    parser.add_argument("--min_prior_points", type=int, default=200)
    parser.add_argument("--grid_resolution", type=int, default=64)
    parser.add_argument("--bbox_scale", type=float, default=1.15)
    parser.add_argument("--min_support_views", type=float, default=1.0)
    parser.add_argument("--min_support_ratio", type=float, default=0.10)
    parser.add_argument("--mask_threshold", type=int, default=127)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    try:
        build(parse_args())
    except (ValueError, FileNotFoundError) as exc:
        raise SystemExit(f"[real_slam_prior][ERROR] {exc}") from None


if __name__ == "__main__":
    main()
