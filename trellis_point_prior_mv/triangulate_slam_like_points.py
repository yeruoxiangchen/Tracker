#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

TRACKER_ROOT = Path(__file__).resolve().parents[1]
if str(TRACKER_ROOT) not in sys.path:
    sys.path.insert(0, str(TRACKER_ROOT))

from trellis_point_prior_mv.build_real_slam_prior_manifest import (  # noqa: E402
    parse_colmap_cameras,
    parse_colmap_images,
)
from trellis_point_prior_mv.common import write_json  # noqa: E402


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def image_dir_for_dataset(dataset_dir: Path) -> Path:
    for name in ("images", "rgb"):
        path = dataset_dir / name
        if path.is_dir():
            return path
    raise FileNotFoundError(f"missing images/rgb dir under {dataset_dir}")


def mask_path_for_image(dataset_dir: Path, image_name: str) -> Path | None:
    mask_dir = dataset_dir / "masks"
    stem = Path(image_name).stem
    for suffix in (".png", ".jpg", ".jpeg"):
        path = mask_dir / f"{stem}{suffix}"
        if path.exists():
            return path
    return None


def load_frame_data(dataset_dir: Path, sparse_dir: Path, args: argparse.Namespace) -> list[dict]:
    cameras = parse_colmap_cameras(sparse_dir / "cameras.txt")
    images = parse_colmap_images(sparse_dir / "images.txt")
    image_dir = image_dir_for_dataset(dataset_dir)
    frames: list[dict] = []
    for name, meta in sorted(images.items(), key=lambda kv: int(kv[1]["image_id"])):
        image_path = image_dir / Path(name).name
        mask_path = mask_path_for_image(dataset_dir, Path(name).name)
        if not image_path.exists() or mask_path is None or not mask_path.exists():
            continue
        camera = cameras.get(int(meta["camera_id"]))
        if camera is None:
            continue
        frames.append({"name": Path(name).name, "image": image_path, "mask": mask_path, "meta": meta, "camera": camera})
    if args.max_frames > 0:
        frames = frames[: int(args.max_frames)]
    if len(frames) < 2:
        raise ValueError(f"need at least two image/mask frames with COLMAP poses: {dataset_dir}")
    return frames


def projection_matrix(frame: dict) -> np.ndarray:
    camera = frame["camera"]
    meta = frame["meta"]
    k = np.asarray(
        [
            [camera["fx"], 0.0, camera["cx"]],
            [0.0, camera["fy"], camera["cy"]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    rt = np.concatenate([meta["R"], meta["tvec"].reshape(3, 1)], axis=1)
    return k @ rt


def camera_center(frame: dict) -> np.ndarray:
    meta = frame["meta"]
    return -meta["R"].T @ meta["tvec"]


def project_points(points: np.ndarray, frame: dict) -> tuple[np.ndarray, np.ndarray]:
    camera = frame["camera"]
    meta = frame["meta"]
    xyz_cam = (meta["R"] @ points.astype(np.float64).T).T + meta["tvec"][None, :]
    z = xyz_cam[:, 2]
    valid = z > 1e-6
    u = camera["fx"] * (xyz_cam[:, 0] / np.maximum(z, 1e-6)) + camera["cx"]
    v = camera["fy"] * (xyz_cam[:, 1] / np.maximum(z, 1e-6)) + camera["cy"]
    uv = np.stack([u, v], axis=1)
    inside = (
        valid
        & (u >= 0)
        & (u < int(camera["width"]))
        & (v >= 0)
        & (v < int(camera["height"]))
    )
    return uv.astype(np.float32), inside


def reprojection_error(points: np.ndarray, frame: dict, observed_uv: np.ndarray) -> np.ndarray:
    uv, inside = project_points(points, frame)
    err = np.linalg.norm(uv - observed_uv.astype(np.float32), axis=1)
    err[~inside] = np.inf
    return err


def read_image_and_mask(frame: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    image = cv2.imread(str(frame["image"]), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(frame["image"])
    mask = Image.open(frame["mask"]).convert("L")
    if mask.size != (image.shape[1], image.shape[0]):
        mask = mask.resize((image.shape[1], image.shape[0]), Image.Resampling.NEAREST)
    mask_np = np.asarray(mask)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image, gray, mask_np


def detect_features(frames: list[dict], args: argparse.Namespace) -> None:
    sift = cv2.SIFT_create(nfeatures=int(args.max_features))
    for frame in frames:
        image, gray, mask_np = read_image_and_mask(frame)
        feature_mask = None
        if args.feature_mask_mode == "mask":
            feature_mask = (mask_np > int(args.mask_threshold)).astype(np.uint8) * 255
        keypoints, desc = sift.detectAndCompute(gray, feature_mask)
        if desc is None or not keypoints:
            pts = np.zeros((0, 2), dtype=np.float32)
            desc = np.zeros((0, 128), dtype=np.float32)
        else:
            pts = np.asarray([kp.pt for kp in keypoints], dtype=np.float32)
            desc = np.asarray(desc, dtype=np.float32)
        frame["image_bgr"] = image
        frame["mask_np"] = mask_np
        frame["keypoints_xy"] = pts
        frame["descriptors"] = desc


def pair_indices(num_frames: int, args: argparse.Namespace) -> list[tuple[int, int]]:
    pairs = []
    if args.matcher == "exhaustive":
        for i in range(num_frames):
            for j in range(i + 1, num_frames):
                if args.max_pair_gap > 0 and j - i > int(args.max_pair_gap):
                    continue
                pairs.append((i, j))
    elif args.matcher == "sequential":
        window = max(int(args.max_pair_gap), 1)
        for i in range(num_frames):
            for j in range(i + 1, min(num_frames, i + 1 + window)):
                pairs.append((i, j))
    else:
        raise ValueError(f"unsupported matcher={args.matcher}")
    return pairs


def mask_values_at(mask: np.ndarray, pts: np.ndarray) -> np.ndarray:
    if pts.shape[0] == 0:
        return np.zeros((0,), dtype=bool)
    x = np.rint(pts[:, 0]).astype(np.int64)
    y = np.rint(pts[:, 1]).astype(np.int64)
    inside = (x >= 0) & (x < mask.shape[1]) & (y >= 0) & (y < mask.shape[0])
    out = np.zeros((pts.shape[0],), dtype=bool)
    ids = np.where(inside)[0]
    out[ids] = mask[y[ids], x[ids]] > 0
    return out


def triangulation_angles(points: np.ndarray, frame_a: dict, frame_b: dict) -> np.ndarray:
    ca = camera_center(frame_a).reshape(1, 3)
    cb = camera_center(frame_b).reshape(1, 3)
    va = points - ca
    vb = points - cb
    va = va / np.maximum(np.linalg.norm(va, axis=1, keepdims=True), 1e-9)
    vb = vb / np.maximum(np.linalg.norm(vb, axis=1, keepdims=True), 1e-9)
    dots = np.clip(np.sum(va * vb, axis=1), -1.0, 1.0)
    return np.degrees(np.arccos(dots))


def triangulate_pair(frame_a: dict, frame_b: dict, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    desc_a = frame_a["descriptors"]
    desc_b = frame_b["descriptors"]
    if desc_a.shape[0] < 2 or desc_b.shape[0] < 2:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    raw = matcher.knnMatch(desc_a, desc_b, k=2)
    matches = []
    for item in raw:
        if len(item) < 2:
            continue
        m, n = item
        if m.distance < float(args.ratio_test) * n.distance:
            matches.append(m)
    if len(matches) < int(args.min_pair_matches):
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    pts_a = frame_a["keypoints_xy"][[m.queryIdx for m in matches]]
    pts_b = frame_b["keypoints_xy"][[m.trainIdx for m in matches]]
    if args.require_pair_mask_hit:
        in_mask = mask_values_at(frame_a["mask_np"], pts_a) & mask_values_at(frame_b["mask_np"], pts_b)
        pts_a = pts_a[in_mask]
        pts_b = pts_b[in_mask]
    if pts_a.shape[0] < int(args.min_pair_matches):
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    ph = cv2.triangulatePoints(projection_matrix(frame_a), projection_matrix(frame_b), pts_a.T, pts_b.T)
    points = (ph[:3] / np.maximum(ph[3:4], 1e-12)).T.astype(np.float32)
    finite = np.isfinite(points).all(axis=1)
    err_a = reprojection_error(points, frame_a, pts_a)
    err_b = reprojection_error(points, frame_b, pts_b)
    angles = triangulation_angles(points.astype(np.float64), frame_a, frame_b)
    valid = (
        finite
        & (err_a <= float(args.max_reproj_error))
        & (err_b <= float(args.max_reproj_error))
        & (angles >= float(args.min_triangulation_angle_deg))
    )
    points = points[valid]
    errors = ((err_a[valid] + err_b[valid]) * 0.5).astype(np.float32)
    return points, errors


def support_filter_and_color(points: np.ndarray, errors: np.ndarray, frames: list[dict], args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    if points.shape[0] == 0:
        return points, errors, np.zeros((0, 3), dtype=np.uint8), {
            "support_filtered_count": 0,
            "mask_support_mean": 0.0,
            "mask_support_median": 0.0,
        }
    support = np.zeros((points.shape[0],), dtype=np.float32)
    in_image = np.zeros((points.shape[0],), dtype=np.float32)
    colors = np.zeros((points.shape[0], 3), dtype=np.uint8)
    has_color = np.zeros((points.shape[0],), dtype=bool)
    for frame in frames:
        uv, inside = project_points(points, frame)
        ids = np.where(inside)[0]
        if ids.size == 0:
            continue
        xi = np.clip(np.rint(uv[ids, 0]).astype(np.int64), 0, frame["mask_np"].shape[1] - 1)
        yi = np.clip(np.rint(uv[ids, 1]).astype(np.int64), 0, frame["mask_np"].shape[0] - 1)
        hit = frame["mask_np"][yi, xi] > int(args.mask_threshold)
        in_image[ids] += 1.0
        hit_ids = ids[hit]
        support[hit_ids] += 1.0
        color_ids = hit_ids[~has_color[hit_ids]]
        if color_ids.size:
            cu = np.clip(np.rint(uv[color_ids, 0]).astype(np.int64), 0, frame["image_bgr"].shape[1] - 1)
            cv = np.clip(np.rint(uv[color_ids, 1]).astype(np.int64), 0, frame["image_bgr"].shape[0] - 1)
            bgr = frame["image_bgr"][cv, cu]
            colors[color_ids] = bgr[:, ::-1]
            has_color[color_ids] = True

    min_support = max(float(args.min_support_views), math.ceil(float(args.min_support_ratio) * len(frames)))
    keep = support >= min_support
    points = points[keep]
    errors = errors[keep]
    colors = colors[keep]
    kept_support = support[keep]
    return points, errors, colors, {
        "support_filtered_count": int(points.shape[0]),
        "min_support_required": float(min_support),
        "mask_support_mean": float(kept_support.mean()) if points.shape[0] else 0.0,
        "mask_support_median": float(np.median(kept_support)) if points.shape[0] else 0.0,
        "any_in_image_ratio": float((in_image > 0).mean()) if in_image.shape[0] else 0.0,
    }


def merge_points(points: np.ndarray, errors: np.ndarray, colors: np.ndarray, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if points.shape[0] == 0:
        return points, errors, colors
    order = np.argsort(errors)
    points = points[order]
    errors = errors[order]
    colors = colors[order]
    if float(args.merge_voxel_size) > 0:
        keys = np.floor(points / float(args.merge_voxel_size)).astype(np.int64)
        _, unique_ids = np.unique(keys, axis=0, return_index=True)
        keep = np.sort(unique_ids)
        points = points[keep]
        errors = errors[keep]
        colors = colors[keep]
    if args.max_output_points > 0 and points.shape[0] > int(args.max_output_points):
        points = points[: int(args.max_output_points)]
        errors = errors[: int(args.max_output_points)]
        colors = colors[: int(args.max_output_points)]
    return points, errors, colors


def copy_sparse_pose_files(input_sparse: Path, output_sparse: Path) -> None:
    output_sparse.mkdir(parents=True, exist_ok=True)
    for name in ("cameras.txt", "images.txt", "phone_pose_meta.json"):
        src = input_sparse / name
        if src.exists():
            shutil.copy2(src, output_sparse / name)


def write_points3d(path: Path, points: np.ndarray, errors: np.ndarray, colors: np.ndarray) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        f.write(f"# Number of points: {points.shape[0]}, mean track length: 0\n")
        for idx, (p, err, rgb) in enumerate(zip(points, errors, colors), start=1):
            r, g, b = [int(x) for x in rgb.tolist()]
            f.write(f"{idx} {p[0]:.9f} {p[1]:.9f} {p[2]:.9f} {r} {g} {b} {float(err):.6f}\n")


def process_dataset(dataset_dir: Path, args: argparse.Namespace) -> dict:
    input_sparse = (dataset_dir / args.input_sparse_subdir).resolve()
    output_sparse = (dataset_dir / args.output_sparse_subdir).resolve()
    if input_sparse == output_sparse:
        raise ValueError(
            f"refusing to overwrite input sparse dir: input_sparse={input_sparse}, output_sparse={output_sparse}. "
            "Use --output_sparse_subdir such as sparse_slam_eval/0."
        )
    if output_sparse.exists() and not args.overwrite:
        points_path = output_sparse / "points3D.txt"
        if points_path.exists():
            raise FileExistsError(f"output exists; pass --overwrite to rebuild: {output_sparse}")
    if output_sparse.exists() and args.overwrite:
        shutil.rmtree(output_sparse)

    frames = load_frame_data(dataset_dir, input_sparse, args)
    detect_features(frames, args)
    pairs = pair_indices(len(frames), args)
    all_points = []
    all_errors = []
    pair_rows = []
    for pair_idx, (i, j) in enumerate(pairs):
        pts, err = triangulate_pair(frames[i], frames[j], args)
        if pts.shape[0]:
            all_points.append(pts)
            all_errors.append(err)
        pair_rows.append({
            "pair_index": pair_idx,
            "image_a": frames[i]["name"],
            "image_b": frames[j]["name"],
            "triangulated_count": int(pts.shape[0]),
        })

    if all_points:
        points = np.concatenate(all_points, axis=0)
        errors = np.concatenate(all_errors, axis=0)
    else:
        points = np.zeros((0, 3), dtype=np.float32)
        errors = np.zeros((0,), dtype=np.float32)

    raw_count = int(points.shape[0])
    points, errors, colors, support_stats = support_filter_and_color(points, errors, frames, args)
    supported_count = int(points.shape[0])
    points, errors, colors = merge_points(points, errors, colors, args)

    if points.shape[0] < int(args.min_output_points):
        raise ValueError(
            f"{dataset_dir}: generated {points.shape[0]} object SLAM-like points "
            f"< min_output_points {int(args.min_output_points)}"
        )

    copy_sparse_pose_files(input_sparse, output_sparse)
    write_points3d(output_sparse / "points3D.txt", points, errors, colors)
    meta = {
        "source": "opencv_sift_fixed_pose_triangulation",
        "input_sparse": str(input_sparse),
        "output_sparse": str(output_sparse),
        "args": vars(args),
        "frame_count": len(frames),
        "pair_count": len(pairs),
        "raw_triangulated_count": raw_count,
        "support_filtered_count": supported_count,
        "final_point_count": int(points.shape[0]),
        **support_stats,
    }
    write_json(output_sparse / "slam_like_points_meta.json", meta)
    write_csv(output_sparse / "pair_report.csv", pair_rows)
    return {
        "dataset": str(dataset_dir),
        "output_sparse": str(output_sparse),
        "frame_count": len(frames),
        "pair_count": len(pairs),
        "raw_triangulated_count": raw_count,
        "support_filtered_count": supported_count,
        "final_point_count": int(points.shape[0]),
        **support_stats,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate SLAM-like object sparse points from fixed COLMAP/phone poses.")
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--input_sparse_subdir", default="sparse/0")
    parser.add_argument("--output_sparse_subdir", default="sparse_slam/0")
    parser.add_argument("--max_frames", type=int, default=18)
    parser.add_argument("--max_features", type=int, default=4096)
    parser.add_argument("--feature_mask_mode", choices=["none", "mask"], default="none")
    parser.add_argument("--matcher", choices=["exhaustive", "sequential"], default="exhaustive")
    parser.add_argument("--max_pair_gap", type=int, default=0)
    parser.add_argument("--ratio_test", type=float, default=0.75)
    parser.add_argument("--min_pair_matches", type=int, default=12)
    parser.add_argument("--require_pair_mask_hit", action="store_true", default=True)
    parser.add_argument("--allow_pair_outside_mask", dest="require_pair_mask_hit", action="store_false")
    parser.add_argument("--max_reproj_error", type=float, default=4.0)
    parser.add_argument("--min_triangulation_angle_deg", type=float, default=1.0)
    parser.add_argument("--min_support_views", type=float, default=2.0)
    parser.add_argument("--min_support_ratio", type=float, default=0.10)
    parser.add_argument("--mask_threshold", type=int, default=127)
    parser.add_argument("--merge_voxel_size", type=float, default=0.002)
    parser.add_argument("--max_output_points", type=int, default=50000)
    parser.add_argument("--min_output_points", type=int, default=50)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--output_report", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    try:
        for dataset in args.datasets:
            row = process_dataset(Path(dataset), args)
            rows.append(row)
            print(
                f"[slam_like_points] {Path(dataset).name} frames={row['frame_count']} "
                f"raw={row['raw_triangulated_count']} support={row['support_filtered_count']} "
                f"final={row['final_point_count']} -> {row['output_sparse']}",
                flush=True,
            )
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        raise SystemExit(f"[slam_like_points][ERROR] {exc}") from None
    if args.output_report:
        out = Path(args.output_report)
        write_json(out, {"rows": rows, "args": vars(args)})
        write_csv(out.with_suffix(".csv"), rows)


if __name__ == "__main__":
    main()
