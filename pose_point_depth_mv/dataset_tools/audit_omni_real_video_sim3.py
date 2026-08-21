#!/usr/bin/env python3
"""Diagnose proper-Sim(3) alignment from masked COLMAP points to Scan.obj."""

from __future__ import annotations

import argparse
import itertools
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

from pose_point_depth_mv.dataset_tools.prepare_omni_real_video_cache import (
    data_lines,
    parse_points,
    sha256_file,
    write_json,
    write_npz,
)


REPORT_FORMAT = "pose_point_depth_mv.omni_real_video_sample_sim3.v1"
OBJECT_FORMAT = "pose_point_depth_mv.omni_real_video_object_sim3.v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--sample_report", required=True)
    parser.add_argument("--contract_report", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--object_id",
        action="append",
        help="Optional development subset; omit for the frozen full sample.",
    )
    parser.add_argument("--mask_threshold", type=int, default=127)
    parser.add_argument("--min_mask_observations", type=int, default=2)
    parser.add_argument("--min_mask_ratio", type=float, default=0.60)
    parser.add_argument("--min_object_points", type=int, default=100)
    parser.add_argument("--max_source_points", type=int, default=10000)
    parser.add_argument("--max_scan_vertices", type=int, default=100000)
    parser.add_argument("--icp_iterations", type=int, default=20)
    parser.add_argument("--icp_trim_fraction", type=float, default=0.80)
    parser.add_argument("--max_median_normalized", type=float, default=0.03)
    parser.add_argument("--max_p90_normalized", type=float, default=0.08)
    parser.add_argument("--min_inlier_rate_3pct", type=float, default=0.70)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def evenly_spaced_indices(count: int, limit: int) -> np.ndarray:
    if count <= limit:
        return np.arange(count, dtype=np.int64)
    return np.linspace(0, count - 1, limit, dtype=np.int64)


def load_masked_colmap_points(
    object_root: Path,
    *,
    mask_threshold: int,
    min_observations: int,
    min_ratio: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    standard = object_root / "standard"
    sparse = standard / "sparse" / "0_txt"
    masks_dir = standard / "matting"
    points = parse_points(sparse / "points3D.txt")
    point_ids = points["point_id"]
    id_to_index = {int(point_id): index for index, point_id in enumerate(point_ids)}
    foreground = np.zeros(len(point_ids), dtype=np.int32)
    observed = np.zeros(len(point_ids), dtype=np.int32)

    lines = data_lines(sparse / "images.txt")
    if len(lines) % 2:
        raise RuntimeError(f"odd COLMAP images data-line count: {sparse / 'images.txt'}")
    resolved_frames = 0
    for pose_line, observations_line in zip(lines[0::2], lines[1::2]):
        pose_fields = pose_line.split()
        if len(pose_fields) < 10:
            raise RuntimeError(f"invalid image pose row: {pose_line[:200]}")
        frame_name = Path(" ".join(pose_fields[9:])).name
        mask_path = masks_dir / frame_name
        if not mask_path.is_file():
            continue
        mask = np.asarray(Image.open(mask_path).convert("L"), dtype=np.uint8)
        height, width = mask.shape
        fields = observations_line.split()
        if len(fields) % 3:
            raise RuntimeError(f"invalid POINTS2D row for {frame_name}")
        resolved_frames += 1
        for offset in range(0, len(fields), 3):
            point_id = int(fields[offset + 2])
            point_index = id_to_index.get(point_id)
            if point_index is None:
                continue
            x = int(round(float(fields[offset])))
            y = int(round(float(fields[offset + 1])))
            if x < 0 or x >= width or y < 0 or y >= height:
                continue
            observed[point_index] += 1
            foreground[point_index] += int(mask[y, x] >= mask_threshold)

    ratio = foreground.astype(np.float64) / np.maximum(observed, 1)
    keep = (foreground >= min_observations) & (ratio >= min_ratio)
    selected = points["xyz"][keep]
    if len(selected) == 0:
        raise RuntimeError(f"no mask-supported sparse points: {object_root}")
    return selected, {
        "all_sparse_point_count": len(point_ids),
        "resolved_mask_frame_count": resolved_frames,
        "mask_supported_point_count": len(selected),
        "mask_supported_fraction": len(selected) / max(len(point_ids), 1),
        "foreground_observation_median": float(np.median(foreground[keep])),
        "foreground_ratio_median": float(np.median(ratio[keep])),
    }


def scan_vertex_pass(path: Path, collect_stride: int | None = None) -> tuple[Any, ...]:
    count = 0
    low = np.full(3, np.inf, dtype=np.float64)
    high = np.full(3, -np.inf, dtype=np.float64)
    vertices = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.startswith("v "):
                continue
            fields = line.split()
            if len(fields) < 4:
                continue
            vertex = np.asarray([float(value) for value in fields[1:4]], dtype=np.float64)
            low = np.minimum(low, vertex)
            high = np.maximum(high, vertex)
            if collect_stride is not None and count % collect_stride == 0:
                vertices.append(vertex)
            count += 1
    if count == 0:
        raise RuntimeError(f"Scan.obj contains no vertices: {path}")
    if collect_stride is None:
        return count, low, high
    return np.asarray(vertices, dtype=np.float64), count, low, high


def load_scan_vertices(path: Path, maximum: int) -> tuple[np.ndarray, dict[str, Any]]:
    count, low, high = scan_vertex_pass(path)
    stride = max(1, int(math.ceil(count / maximum)))
    vertices, second_count, second_low, second_high = scan_vertex_pass(path, stride)
    if count != second_count or not np.allclose(low, second_low) or not np.allclose(high, second_high):
        raise RuntimeError(f"Scan.obj changed while reading: {path}")
    diagonal = float(np.linalg.norm(high - low))
    if not math.isfinite(diagonal) or diagonal <= 0:
        raise RuntimeError(f"invalid Scan.obj bounds: {path}")
    return vertices, {
        "scan_vertex_count": count,
        "sampled_scan_vertex_count": len(vertices),
        "scan_bounds_min": low.tolist(),
        "scan_bounds_max": high.tolist(),
        "scan_bounds_diagonal": diagonal,
        "vertex_stride": stride,
    }


def robust_pca(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    low = np.quantile(points, 0.02, axis=0)
    high = np.quantile(points, 0.98, axis=0)
    keep = np.all((points >= low) & (points <= high), axis=1)
    core = points[keep]
    if len(core) < 16:
        core = points
    center = np.mean(core, axis=0)
    centered = core - center
    covariance = centered.T @ centered / max(len(centered) - 1, 1)
    _, basis = np.linalg.eigh(covariance)
    basis = basis[:, ::-1]
    if np.linalg.det(basis) < 0:
        basis[:, -1] *= -1.0
    radius = float(np.sqrt(np.mean(np.sum(centered * centered, axis=1))))
    return center, basis, max(radius, 1.0e-12)


def proper_axis_maps() -> list[np.ndarray]:
    output = []
    for permutation in itertools.permutations(range(3)):
        base = np.eye(3, dtype=np.float64)[:, permutation]
        for signs in itertools.product((-1.0, 1.0), repeat=3):
            candidate = base @ np.diag(signs)
            if np.linalg.det(candidate) > 0.5:
                output.append(candidate)
    if len(output) != 24:
        raise RuntimeError(f"expected 24 proper axis maps, got {len(output)}")
    return output


def umeyama_proper(source: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    source_mean = np.mean(source, axis=0)
    target_mean = np.mean(target, axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    covariance = target_centered.T @ source_centered / len(source)
    u_matrix, singular, vt_matrix = np.linalg.svd(covariance)
    correction = np.eye(3, dtype=np.float64)
    if np.linalg.det(u_matrix @ vt_matrix) < 0:
        correction[-1, -1] = -1.0
    rotation = u_matrix @ correction @ vt_matrix
    variance = float(np.mean(np.sum(source_centered * source_centered, axis=1)))
    scale = float(np.sum(singular * np.diag(correction)) / max(variance, 1.0e-12))
    translation = target_mean - scale * (rotation @ source_mean)
    if scale <= 0 or np.linalg.det(rotation) <= 0:
        raise RuntimeError("Umeyama produced an invalid proper similarity")
    return scale, rotation, translation


def apply_similarity(
    points: np.ndarray, scale: float, rotation: np.ndarray, translation: np.ndarray
) -> np.ndarray:
    return scale * (points @ rotation.T) + translation


def fit_candidate(
    source: np.ndarray,
    target: np.ndarray,
    tree: cKDTree,
    *,
    source_center: np.ndarray,
    target_center: np.ndarray,
    source_radius: float,
    target_radius: float,
    source_basis: np.ndarray,
    target_basis: np.ndarray,
    axis_map: np.ndarray,
    iterations: int,
    trim_fraction: float,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    rotation = target_basis @ axis_map @ source_basis.T
    scale = target_radius / source_radius
    initial_scale = scale
    translation = target_center - scale * (rotation @ source_center)
    distances = np.empty(len(source), dtype=np.float64)
    for _ in range(iterations):
        transformed = apply_similarity(source, scale, rotation, translation)
        distances, indices = tree.query(transformed, workers=-1)
        keep_count = max(16, int(math.ceil(len(source) * trim_fraction)))
        keep = np.argpartition(distances, keep_count - 1)[:keep_count]
        next_scale, next_rotation, next_translation = umeyama_proper(
            source[keep], target[indices[keep]]
        )
        if not 0.4 * initial_scale <= next_scale <= 2.5 * initial_scale:
            break
        delta = abs(next_scale - scale) / max(scale, 1.0e-12)
        scale, rotation, translation = next_scale, next_rotation, next_translation
        if delta < 1.0e-6:
            break
    transformed = apply_similarity(source, scale, rotation, translation)
    distances, _ = tree.query(transformed, workers=-1)
    keep_count = max(16, int(math.ceil(len(source) * trim_fraction)))
    keep = np.argpartition(distances, keep_count - 1)[:keep_count]
    score = float(np.sqrt(np.mean(np.square(distances[keep]))))
    return scale, rotation, translation, distances


def align_points(
    source: np.ndarray,
    target: np.ndarray,
    *,
    iterations: int,
    trim_fraction: float,
) -> dict[str, Any]:
    source_center, source_basis, source_radius = robust_pca(source)
    target_center, target_basis, target_radius = robust_pca(target)
    tree = cKDTree(target)
    candidates = []
    for index, axis_map in enumerate(proper_axis_maps()):
        scale, rotation, translation, distances = fit_candidate(
            source,
            target,
            tree,
            source_center=source_center,
            target_center=target_center,
            source_radius=source_radius,
            target_radius=target_radius,
            source_basis=source_basis,
            target_basis=target_basis,
            axis_map=axis_map,
            iterations=iterations,
            trim_fraction=trim_fraction,
        )
        keep_count = max(16, int(math.ceil(len(source) * trim_fraction)))
        keep = np.argpartition(distances, keep_count - 1)[:keep_count]
        candidates.append(
            {
                "index": index,
                "scale": scale,
                "rotation": rotation,
                "translation": translation,
                "distances": distances,
                "trimmed_rmse": float(np.sqrt(np.mean(np.square(distances[keep])))),
            }
        )
    best = min(candidates, key=lambda row: row["trimmed_rmse"])
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = best["scale"] * best["rotation"]
    transform[:3, 3] = best["translation"]
    transformed = apply_similarity(
        source, best["scale"], best["rotation"], best["translation"]
    )
    return {
        "candidate_count": len(candidates),
        "selected_candidate": best["index"],
        "scale": best["scale"],
        "rotation_determinant": float(np.linalg.det(best["rotation"])),
        "T_COLMAP_W_to_Scan": transform,
        "transformed_source": transformed,
        "distances": best["distances"],
        "trimmed_rmse": best["trimmed_rmse"],
        "runner_up_trimmed_rmse": sorted(row["trimmed_rmse"] for row in candidates)[1],
    }


def write_colored_ply(path: Path, scan: np.ndarray, source: np.ndarray) -> None:
    scan_keep = scan[evenly_spaced_indices(len(scan), min(len(scan), 50000))]
    source_keep = source[evenly_spaced_indices(len(source), min(len(source), 10000))]
    vertices = np.concatenate([scan_keep, source_keep], axis=0)
    colors = np.concatenate(
        [
            np.tile(np.asarray([[170, 170, 170]], dtype=np.uint8), (len(scan_keep), 1)),
            np.tile(np.asarray([[220, 35, 35]], dtype=np.uint8), (len(source_keep), 1)),
        ],
        axis=0,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write(f"element vertex {len(vertices)}\n")
        handle.write("property float x\nproperty float y\nproperty float z\n")
        handle.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        handle.write("end_header\n")
        for vertex, color in zip(vertices, colors):
            handle.write(
                f"{vertex[0]:.9g} {vertex[1]:.9g} {vertex[2]:.9g} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )


def write_alignment_preview(
    path: Path,
    scan: np.ndarray,
    source: np.ndarray,
    *,
    title: str = (
        "Gray: Scan vertices; red: mask-supported COLMAP points after proper Sim(3)"
    ),
) -> None:
    scan_plot = scan[evenly_spaced_indices(len(scan), min(len(scan), 12000))]
    source_plot = source[evenly_spaced_indices(len(source), min(len(source), 4000))]
    pairs = ((0, 1, "X", "Y"), (0, 2, "X", "Z"), (1, 2, "Y", "Z"))
    figure, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=140)
    for axis, (first, second, first_name, second_name) in zip(axes, pairs):
        axis.scatter(scan_plot[:, first], scan_plot[:, second], s=0.35, c="#9b9b9b", alpha=0.30)
        axis.scatter(source_plot[:, first], source_plot[:, second], s=2.0, c="#d62728", alpha=0.80)
        axis.set_xlabel(first_name)
        axis.set_ylabel(second_name)
        axis.set_aspect("equal", adjustable="box")
        axis.grid(True, alpha=0.15)
    figure.suptitle(title)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path)
    plt.close(figure)


def audit_object(row: dict[str, Any], output_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    object_id = row["object_id"]
    object_root = Path(row["object_root"])
    scan_obj = Path(row["scan_obj"])
    object_output = output_dir / object_id
    object_output.mkdir(parents=True, exist_ok=True)
    source, source_stats = load_masked_colmap_points(
        object_root,
        mask_threshold=args.mask_threshold,
        min_observations=args.min_mask_observations,
        min_ratio=args.min_mask_ratio,
    )
    source_indices = evenly_spaced_indices(len(source), args.max_source_points)
    fit_source = source[source_indices]
    scan, scan_stats = load_scan_vertices(scan_obj, args.max_scan_vertices)
    aligned = align_points(
        fit_source,
        scan,
        iterations=args.icp_iterations,
        trim_fraction=args.icp_trim_fraction,
    )
    diagonal = scan_stats["scan_bounds_diagonal"]
    distances = aligned["distances"]
    median_normalized = float(np.median(distances) / diagonal)
    p90_normalized = float(np.quantile(distances, 0.90) / diagonal)
    inlier_rate = float(np.mean(distances <= 0.03 * diagonal))
    ambiguity_ratio = float(
        aligned["runner_up_trimmed_rmse"] / max(aligned["trimmed_rmse"], 1.0e-12)
    )
    checks = {
        "minimum_object_points": source_stats["mask_supported_point_count"] >= args.min_object_points,
        "proper_rotation": abs(aligned["rotation_determinant"] - 1.0) <= 1.0e-5,
        "median_normalized": median_normalized <= args.max_median_normalized,
        "p90_normalized": p90_normalized <= args.max_p90_normalized,
        "inlier_rate_3pct": inlier_rate >= args.min_inlier_rate_3pct,
    }
    cache_path = object_output / "masked_colmap_points_and_sim3.npz"
    write_npz(
        cache_path,
        P_W_mask_supported=source,
        P_Scan_aligned=aligned["transformed_source"],
        T_COLMAP_W_to_Scan=aligned["T_COLMAP_W_to_Scan"],
        nearest_scan_distance=distances,
    )
    ply_path = object_output / "alignment_scan_gray_colmap_red.ply"
    preview_path = object_output / "alignment_xyz.png"
    write_colored_ply(ply_path, scan, aligned["transformed_source"])
    write_alignment_preview(preview_path, scan, aligned["transformed_source"])
    payload = {
        "format": OBJECT_FORMAT,
        "category": row["category"],
        "object_id": object_id,
        "object_root": str(object_root),
        "scan_obj": str(scan_obj),
        "scan_obj_sha256": sha256_file(scan_obj),
        "source_stats": source_stats,
        "scan_stats": scan_stats,
        "fit_source_point_count": len(fit_source),
        "candidate_count": aligned["candidate_count"],
        "selected_candidate": aligned["selected_candidate"],
        "similarity_scale": aligned["scale"],
        "rotation_determinant": aligned["rotation_determinant"],
        "T_COLMAP_W_to_Scan": aligned["T_COLMAP_W_to_Scan"].tolist(),
        "trimmed_rmse": aligned["trimmed_rmse"],
        "runner_up_trimmed_rmse": aligned["runner_up_trimmed_rmse"],
        "runner_up_to_best_ratio": ambiguity_ratio,
        "median_normalized": median_normalized,
        "p90_normalized": p90_normalized,
        "inlier_rate_3pct": inlier_rate,
        "checks": checks,
        "automatic_passed": all(checks.values()),
        "cache_npz": str(cache_path),
        "alignment_ply": str(ply_path),
        "alignment_preview": str(preview_path),
        "scope_guard": (
            "GT-Scan-assisted sampled alignment diagnostic. It validates data-frame "
            "compatibility for supervised training; it is not a deployable AR pose estimator."
        ),
    }
    write_json(object_output / "report.json", payload)
    return payload


def main() -> None:
    args = parse_args()
    if not 0 < args.min_mask_ratio <= 1 or not 0 < args.icp_trim_fraction <= 1:
        raise ValueError("invalid ratio threshold")
    sample_report_path = Path(args.sample_report).expanduser().resolve()
    contract_report_path = Path(args.contract_report).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    sample = json.loads(sample_report_path.read_text(encoding="utf-8"))
    contract = json.loads(contract_report_path.read_text(encoding="utf-8"))
    if sample.get("automatic_passed") is not True or contract.get("passed") is not True:
        raise RuntimeError("sample/contract report has not passed")
    if Path(contract["sample_report"]).resolve() != sample_report_path:
        raise RuntimeError("contract report is bound to a different sample report")
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_rows = sample["samples"]
    if args.object_id:
        requested = set(args.object_id)
        selected_rows = [row for row in selected_rows if row["object_id"] in requested]
        if {row["object_id"] for row in selected_rows} != requested:
            raise RuntimeError("requested object_id is absent from the frozen sample")
    rows = []
    for index, row in enumerate(selected_rows, start=1):
        print(
            f"[omni_sim3] {index}/{len(selected_rows)} object={row['object_id']}",
            flush=True,
        )
        result = audit_object(row, output_dir, args)
        rows.append(result)
        print(
            f"[omni_sim3] object={row['object_id']} "
            f"median={result['median_normalized']:.6f} "
            f"p90={result['p90_normalized']:.6f} "
            f"passed={result['automatic_passed']}",
            flush=True,
        )
    payload = {
        "format": REPORT_FORMAT,
        "created_at_utc": utc_now(),
        "sample_report": str(sample_report_path),
        "sample_report_sha256": sha256_file(sample_report_path),
        "contract_report": str(contract_report_path),
        "contract_report_sha256": sha256_file(contract_report_path),
        "sample_count": len(rows),
        "development_object_filter": args.object_id,
        "automatic_pass_count": sum(row["automatic_passed"] for row in rows),
        "automatic_passed": all(row["automatic_passed"] for row in rows),
        "manual_preview_required": True,
        "manual_passed": False,
        "thresholds": {
            "min_object_points": args.min_object_points,
            "max_median_normalized": args.max_median_normalized,
            "max_p90_normalized": args.max_p90_normalized,
            "min_inlier_rate_3pct": args.min_inlier_rate_3pct,
        },
        "objects": rows,
        "scope_guard": (
            "Fixed six-object sampled proper-Sim(3) diagnostic only. No training "
            "manifest is emitted; inspect the six alignment previews before acceptance."
        ),
    }
    write_json(output_dir / "sim3_report.json", payload)
    print(
        json.dumps(
            {
                "automatic_passed": payload["automatic_passed"],
                "sample_count": payload["sample_count"],
                "automatic_pass_count": payload["automatic_pass_count"],
                "report": str(output_dir / "sim3_report.json"),
            },
            indent=2,
        )
    )
    if not payload["automatic_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
