#!/usr/bin/env python3
"""Coarsely align Omni Scan.obj meshes to mask-supported COLMAP points.

This is a training-data preprocessing tool.  It does not estimate a deployable
object pose and it does not modify Native SS/SLat or their checkpoints.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
import shutil
from typing import Any

import numpy as np

from pose_point_depth_mv.dataset_tools.audit_omni_real_video_sim3 import (
    align_points,
    evenly_spaced_indices,
    load_masked_colmap_points,
    load_scan_vertices,
    write_alignment_preview,
    write_colored_ply,
)
from pose_point_depth_mv.dataset_tools.prepare_omni_real_video_cache import (
    RAW_CACHE_FORMAT,
    sha256_file,
    utc_now,
    write_json,
    write_npz,
)


OBJECT_FORMAT = "pose_point_depth_mv.omni_real_mesh_coarse_alignment_object.v1"
MANIFEST_FORMAT = "pose_point_depth_mv.omni_real_mesh_coarse_alignment.v1"
MARKER_FORMAT = "pose_point_depth_mv.omni_real_mesh_coarse_alignment_marker.v1"


@dataclass(frozen=True)
class AlignmentConfig:
    mask_threshold: int = 127
    min_mask_observations: int = 2
    min_mask_ratio: float = 0.60
    min_object_points: int = 100
    max_source_points: int = 10000
    max_scan_vertices: int = 100000
    icp_iterations: int = 20
    icp_trim_fraction: float = 0.80
    max_median_normalized: float = 0.03
    max_p90_normalized: float = 0.08
    min_inlier_rate_3pct: float = 0.70

    def validate(self) -> None:
        if not 0 <= int(self.mask_threshold) <= 255:
            raise ValueError("mask_threshold must be in [0,255]")
        if int(self.min_mask_observations) <= 0 or int(self.min_object_points) < 16:
            raise ValueError("point/observation thresholds must be positive")
        if not 0.0 < float(self.min_mask_ratio) <= 1.0:
            raise ValueError("min_mask_ratio must be in (0,1]")
        if int(self.max_source_points) < 16 or int(self.max_scan_vertices) < 16:
            raise ValueError("point sampling limits must be at least sixteen")
        if int(self.icp_iterations) <= 0:
            raise ValueError("icp_iterations must be positive")
        if not 0.0 < float(self.icp_trim_fraction) <= 1.0:
            raise ValueError("icp_trim_fraction must be in (0,1]")
        if float(self.max_median_normalized) <= 0 or float(self.max_p90_normalized) <= 0:
            raise ValueError("normalized residual limits must be positive")
        if not 0.0 <= float(self.min_inlier_rate_3pct) <= 1.0:
            raise ValueError("min_inlier_rate_3pct must be in [0,1]")


def canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def apply_transform(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64)
    matrix = np.asarray(transform, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("points must be [N,3]")
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("transform must be a finite [4,4] matrix")
    return values @ matrix[:3, :3].T + matrix[:3, 3]


def invert_similarity(transform: np.ndarray) -> np.ndarray:
    matrix = np.asarray(transform, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("similarity must be a finite [4,4] matrix")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1.0e-10):
        raise ValueError("similarity has an invalid homogeneous row")
    linear = matrix[:3, :3]
    gram = linear.T @ linear
    scale_squared = float(np.trace(gram) / 3.0)
    if scale_squared <= 0.0 or not np.allclose(
        gram, np.eye(3) * scale_squared, rtol=1.0e-6, atol=1.0e-10
    ):
        raise ValueError("transform is not an isotropic similarity")
    rotation = linear / math.sqrt(scale_squared)
    if float(np.linalg.det(rotation)) <= 0.0:
        raise ValueError("similarity must contain a proper rotation")
    return np.linalg.inv(matrix)


def _transformed_obj_line(
    line: str,
    *,
    transform: np.ndarray,
    normal_matrix: np.ndarray,
) -> tuple[str, str | None]:
    stripped = line.lstrip()
    indent = line[: len(line) - len(stripped)]
    if stripped.startswith("v "):
        fields = stripped.split()
        if len(fields) < 4:
            raise ValueError(f"invalid OBJ vertex line: {line[:160]!r}")
        point = np.asarray([[float(value) for value in fields[1:4]]], dtype=np.float64)
        output = apply_transform(point, transform)[0]
        suffix = "" if len(fields) == 4 else " " + " ".join(fields[4:])
        return (
            f"{indent}v {output[0]:.12g} {output[1]:.12g} {output[2]:.12g}{suffix}\n",
            "vertex",
        )
    if stripped.startswith("vn "):
        fields = stripped.split()
        if len(fields) < 4:
            raise ValueError(f"invalid OBJ normal line: {line[:160]!r}")
        normal = normal_matrix @ np.asarray(
            [float(value) for value in fields[1:4]], dtype=np.float64
        )
        length = float(np.linalg.norm(normal))
        if not math.isfinite(length) or length <= 1.0e-12:
            raise ValueError("OBJ contains a zero or non-finite normal")
        normal /= length
        return (
            f"{indent}vn {normal[0]:.12g} {normal[1]:.12g} {normal[2]:.12g}\n",
            "normal",
        )
    return line, None


def transform_obj_geometry(
    source: Path,
    destination: Path,
    transform: np.ndarray,
    *,
    strip_materials: bool = True,
) -> dict[str, int | bool]:
    """Transform OBJ vertices/normals while preserving topology and texture indices."""

    source = Path(source)
    destination = Path(destination)
    matrix = np.asarray(transform, dtype=np.float64)
    inverse = invert_similarity(matrix)
    normal_matrix = inverse[:3, :3].T
    destination.parent.mkdir(parents=True, exist_ok=True)
    vertex_count = 0
    normal_count = 0
    stripped_material_lines = 0
    with source.open("r", encoding="utf-8", errors="strict") as input_handle, destination.open(
        "w", encoding="utf-8"
    ) as output_handle:
        for line in input_handle:
            stripped = line.lstrip()
            if strip_materials and (
                stripped.startswith("mtllib ") or stripped.startswith("usemtl ")
            ):
                stripped_material_lines += 1
                continue
            output, kind = _transformed_obj_line(
                line, transform=matrix, normal_matrix=normal_matrix
            )
            output_handle.write(output)
            vertex_count += int(kind == "vertex")
            normal_count += int(kind == "normal")
    if vertex_count <= 0:
        raise RuntimeError(f"OBJ contains no vertices: {source}")
    return {
        "vertex_count": vertex_count,
        "normal_count": normal_count,
        "materials_stripped": bool(strip_materials),
        "stripped_material_line_count": stripped_material_lines,
    }


def coarse_align_mesh(
    *,
    scan_obj: Path,
    world_points: np.ndarray,
    output_dir: Path,
    config: AlignmentConfig,
) -> dict[str, Any]:
    """Fit COLMAP-world points to Scan and export Scan geometry in COLMAP world."""

    config.validate()
    scan_obj = Path(scan_obj).resolve()
    output_dir = Path(output_dir)
    points = np.asarray(world_points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 16:
        raise ValueError("at least sixteen finite COLMAP points [N,3] are required")
    if not np.isfinite(points).all():
        raise ValueError("COLMAP points contain non-finite values")

    fit_indices = evenly_spaced_indices(len(points), int(config.max_source_points))
    fit_points = points[fit_indices]
    scan_sample, scan_stats = load_scan_vertices(
        scan_obj, int(config.max_scan_vertices)
    )
    if len(scan_sample) < 16:
        raise ValueError("at least sixteen sampled Scan vertices are required")
    aligned = align_points(
        fit_points,
        scan_sample,
        iterations=int(config.icp_iterations),
        trim_fraction=float(config.icp_trim_fraction),
    )
    world_to_scan = np.asarray(aligned["T_COLMAP_W_to_Scan"], dtype=np.float64)
    scan_to_world = invert_similarity(world_to_scan)
    scan_in_world = apply_transform(scan_sample, scan_to_world)

    roundtrip = scan_to_world @ world_to_scan
    roundtrip_error = float(np.max(np.abs(roundtrip - np.eye(4))))
    diagonal = float(scan_stats["scan_bounds_diagonal"])
    distances = np.asarray(aligned["distances"], dtype=np.float64)
    median_normalized = float(np.median(distances) / diagonal)
    p90_normalized = float(np.quantile(distances, 0.90) / diagonal)
    inlier_rate = float(np.mean(distances <= 0.03 * diagonal))
    ambiguity_ratio = float(
        aligned["runner_up_trimmed_rmse"]
        / max(float(aligned["trimmed_rmse"]), 1.0e-12)
    )

    aligned_obj = output_dir / "Scan_in_COLMAP_W.obj"
    obj_stats = transform_obj_geometry(
        scan_obj, aligned_obj, scan_to_world, strip_materials=True
    )
    cache_path = output_dir / "coarse_alignment.npz"
    write_npz(
        cache_path,
        P_W_mask_supported=points,
        P_Scan_from_W=aligned["transformed_source"],
        P_W_from_Scan=scan_in_world,
        T_COLMAP_W_to_Scan=world_to_scan,
        T_Scan_to_COLMAP_W=scan_to_world,
        nearest_scan_distance=distances,
    )
    ply_path = output_dir / "alignment_world_scan_gray_points_red.ply"
    preview_path = output_dir / "alignment_world_xyz.png"
    write_colored_ply(ply_path, scan_in_world, points)
    write_alignment_preview(
        preview_path,
        scan_in_world,
        points,
        title="COLMAP world: gray aligned Scan; red mask-supported COLMAP points",
    )

    checks = {
        "minimum_object_points": len(points) >= int(config.min_object_points),
        "proper_rotation": abs(float(aligned["rotation_determinant"]) - 1.0)
        <= 1.0e-5,
        "similarity_inverse_roundtrip": roundtrip_error <= 1.0e-8,
        "mesh_vertex_count_preserved": int(obj_stats["vertex_count"])
        == int(scan_stats["scan_vertex_count"]),
        "median_normalized": median_normalized
        <= float(config.max_median_normalized),
        "p90_normalized": p90_normalized <= float(config.max_p90_normalized),
        "inlier_rate_3pct": inlier_rate >= float(config.min_inlier_rate_3pct),
    }
    return {
        "fit_source_point_count": len(fit_points),
        "scan_stats": scan_stats,
        "obj_stats": obj_stats,
        "candidate_count": int(aligned["candidate_count"]),
        "selected_candidate": int(aligned["selected_candidate"]),
        "similarity_scale_COLMAP_W_to_Scan": float(aligned["scale"]),
        "rotation_determinant": float(aligned["rotation_determinant"]),
        "T_COLMAP_W_to_Scan": world_to_scan.tolist(),
        "T_Scan_to_COLMAP_W": scan_to_world.tolist(),
        "inverse_roundtrip_max_abs": roundtrip_error,
        "trimmed_rmse_scan_units": float(aligned["trimmed_rmse"]),
        "runner_up_trimmed_rmse_scan_units": float(
            aligned["runner_up_trimmed_rmse"]
        ),
        "runner_up_to_best_ratio": ambiguity_ratio,
        "median_normalized": median_normalized,
        "p90_normalized": p90_normalized,
        "inlier_rate_3pct": inlier_rate,
        "checks": checks,
        "automatic_passed": all(checks.values()),
        "aligned_mesh": str(aligned_obj.resolve()),
        "cache_npz": str(cache_path.resolve()),
        "alignment_ply": str(ply_path.resolve()),
        "alignment_preview": str(preview_path.resolve()),
    }


def _object_key(row: dict[str, Any]) -> str:
    return f"{row['category']}:{row['object_id']}"


def select_rows(
    rows: list[dict[str, Any]], requested: list[str] | None
) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda row: (_object_key(row), str(row["object_root"])))
    if not requested:
        return ordered
    by_key = {_object_key(row): row for row in ordered}
    by_object: dict[str, list[dict[str, Any]]] = {}
    for row in ordered:
        by_object.setdefault(str(row["object_id"]), []).append(row)
    selected = []
    for value in requested:
        if value in by_key:
            row = by_key[value]
        else:
            matches = by_object.get(value, [])
            if len(matches) != 1:
                raise ValueError(
                    f"object selector {value!r} matched {len(matches)} rows; "
                    "use category:object_id"
                )
            row = matches[0]
        if row not in selected:
            selected.append(row)
    return selected


def _load_reusable(
    destination: Path,
    *,
    config_hash: str,
    scan_sha256: str,
    raw_cache_sha256: str,
) -> dict[str, Any] | None:
    marker_path = destination / "_COARSE_ALIGNMENT_COMPLETE.json"
    report_path = destination / "report.json"
    if not marker_path.is_file() or not report_path.is_file():
        return None
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if (
        marker.get("format") != MARKER_FORMAT
        or marker.get("config_hash") != config_hash
        or marker.get("scan_obj_sha256") != scan_sha256
        or marker.get("raw_cache_sha256") != raw_cache_sha256
    ):
        raise RuntimeError(f"stale coarse-alignment output: {destination}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("format") != OBJECT_FORMAT:
        raise RuntimeError(f"invalid reusable report: {report_path}")
    required_artifacts = (
        "aligned_mesh",
        "cache_npz",
        "alignment_ply",
        "alignment_preview",
    )
    missing = [
        key
        for key in required_artifacts
        if not isinstance(report.get(key), str) or not Path(report[key]).is_file()
    ]
    if missing:
        raise RuntimeError(
            f"coarse-alignment marker has missing artifacts={missing}: {destination}"
        )
    return report


def align_object_row(
    row: dict[str, Any],
    *,
    output_dir: Path,
    config: AlignmentConfig,
    config_hash: str,
    resume_partial: bool = False,
) -> tuple[dict[str, Any], bool]:
    category = str(row["category"])
    object_id = str(row["object_id"])
    object_root = Path(row["object_root"]).resolve()
    scan_obj = Path(row["scan_obj"]).resolve()
    raw_cache = Path(row["cache_npz"]).resolve()
    scan_hash = sha256_file(scan_obj)
    raw_cache_hash = sha256_file(raw_cache)
    destination = output_dir / "objects" / category / object_id
    reusable = _load_reusable(
        destination,
        config_hash=config_hash,
        scan_sha256=scan_hash,
        raw_cache_sha256=raw_cache_hash,
    )
    if reusable is not None:
        return reusable, True
    if destination.exists():
        raise RuntimeError(f"partial coarse-alignment output exists: {destination}")

    category_root = destination.parent
    category_root.mkdir(parents=True, exist_ok=True)
    staging = category_root / f".{object_id}.aligning"
    if staging.exists():
        if not resume_partial:
            raise RuntimeError(f"partial coarse-alignment staging exists: {staging}")
        shutil.rmtree(staging)
    staging.mkdir()
    points, source_stats = load_masked_colmap_points(
        object_root,
        mask_threshold=int(config.mask_threshold),
        min_observations=int(config.min_mask_observations),
        min_ratio=float(config.min_mask_ratio),
    )
    result = coarse_align_mesh(
        scan_obj=scan_obj,
        world_points=points,
        output_dir=staging,
        config=config,
    )
    # Paths are recorded for the immutable final directory, not the staging path.
    for key in ("aligned_mesh", "cache_npz", "alignment_ply", "alignment_preview"):
        result[key] = str((destination / Path(result[key]).name).resolve())
    report = {
        "format": OBJECT_FORMAT,
        "created_at_utc": utc_now(),
        "category": category,
        "object_id": object_id,
        "object_key": _object_key(row),
        "object_root": str(object_root),
        "scan_obj": str(scan_obj),
        "scan_obj_sha256": scan_hash,
        "raw_cache": str(raw_cache),
        "raw_cache_sha256": raw_cache_hash,
        "source_stats": source_stats,
        "config": asdict(config),
        "config_hash": config_hash,
        **result,
        "scope_guard": (
            "GT-Scan-assisted coarse alignment for real-data training cache only. "
            "The aligned mesh is in raw COLMAP world coordinates; no Native v2 "
            "Full model, checkpoint, or runtime pose contract is modified."
        ),
    }
    write_json(staging / "report.json", report)
    write_json(
        staging / "_COARSE_ALIGNMENT_COMPLETE.json",
        {
            "format": MARKER_FORMAT,
            "completed_at_utc": utc_now(),
            "object_key": report["object_key"],
            "config_hash": config_hash,
            "scan_obj_sha256": scan_hash,
            "raw_cache_sha256": raw_cache_hash,
            "automatic_passed": report["automatic_passed"],
        },
    )
    staging.replace(destination)
    return report, False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--raw_cache_report", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--object",
        action="append",
        help="Optional object_id or category:object_id selector; repeat as needed.",
    )
    parser.add_argument("--allow_failures", action="store_true")
    parser.add_argument("--resume", action="store_true")
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
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = AlignmentConfig(
        mask_threshold=args.mask_threshold,
        min_mask_observations=args.min_mask_observations,
        min_mask_ratio=args.min_mask_ratio,
        min_object_points=args.min_object_points,
        max_source_points=args.max_source_points,
        max_scan_vertices=args.max_scan_vertices,
        icp_iterations=args.icp_iterations,
        icp_trim_fraction=args.icp_trim_fraction,
        max_median_normalized=args.max_median_normalized,
        max_p90_normalized=args.max_p90_normalized,
        min_inlier_rate_3pct=args.min_inlier_rate_3pct,
    )
    config.validate()
    config_hash = canonical_json_sha256(asdict(config))
    raw_report_path = Path(args.raw_cache_report).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    raw = json.loads(raw_report_path.read_text(encoding="utf-8"))
    if raw.get("format") != RAW_CACHE_FORMAT or raw.get("passed") is not True:
        raise RuntimeError(f"raw cache report is not eligible: {raw_report_path}")
    rows = select_rows(list(raw["objects"]), args.object)
    if not rows:
        raise RuntimeError("coarse alignment selected no objects")

    output_dir.mkdir(parents=True, exist_ok=True)
    reports: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    reused: list[str] = []
    for index, row in enumerate(rows, start=1):
        key = _object_key(row)
        print(f"[mesh_coarse_align] {index}/{len(rows)} object={key}", flush=True)
        try:
            report, was_reused = align_object_row(
                row,
                output_dir=output_dir,
                config=config,
                config_hash=config_hash,
                resume_partial=bool(args.resume),
            )
            reports.append(report)
            if was_reused:
                reused.append(key)
            print(
                f"[mesh_coarse_align] object={key} "
                f"median={report['median_normalized']:.6f} "
                f"p90={report['p90_normalized']:.6f} "
                f"passed={report['automatic_passed']} reused={was_reused}",
                flush=True,
            )
        except Exception as error:
            failures.append({"object_key": key, "error": repr(error)})
            print(f"[mesh_coarse_align] FAILED object={key}: {error!r}", flush=True)
            if not args.allow_failures:
                raise

    manifest = {
        "format": MANIFEST_FORMAT,
        "created_at_utc": utc_now(),
        "raw_cache_report": str(raw_report_path),
        "raw_cache_report_sha256": sha256_file(raw_report_path),
        "config": asdict(config),
        "config_hash": config_hash,
        "requested_objects": args.object,
        "selected_object_count": len(rows),
        "completed_object_count": len(reports),
        "automatic_pass_count": sum(
            int(report["automatic_passed"]) for report in reports
        ),
        "reused_objects": reused,
        "objects": reports,
        "failures": failures,
        "training_ready": False,
        "scope_guard": (
            "Mesh coarse-alignment derivative only. It emits Scan-to-COLMAP "
            "transforms and aligned geometry, but not a v2 Full training manifest."
        ),
    }
    manifest["passed"] = bool(reports) and not failures and all(
        report["automatic_passed"] for report in reports
    )
    manifest_path = output_dir / "coarse_alignment_manifest.json"
    write_json(manifest_path, manifest)
    print(
        json.dumps(
            {
                "passed": manifest["passed"],
                "selected_object_count": manifest["selected_object_count"],
                "completed_object_count": manifest["completed_object_count"],
                "automatic_pass_count": manifest["automatic_pass_count"],
                "failure_count": len(failures),
                "training_ready": manifest["training_ready"],
                "manifest": str(manifest_path),
            },
            indent=2,
        )
    )
    if not manifest["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
