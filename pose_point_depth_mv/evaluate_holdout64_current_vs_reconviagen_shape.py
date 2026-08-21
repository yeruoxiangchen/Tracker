#!/usr/bin/env python3
"""Post-hoc normalized-shape comparison on the consumed Omni Holdout64."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.spatial import cKDTree
import trimesh

from pose_point_depth_mv.dataset_tools.prepare_omni_real_label_cache import (
    MANIFEST_FORMAT as LABEL_MANIFEST_FORMAT,
)
from pose_point_depth_mv.dataset_tools.prepare_omni_real_video_cache import utc_now
from pose_point_depth_mv.evaluate_omni_real_mesh_benchmark import load_mesh, summarize
from pose_point_depth_mv.mesh_benchmark_metrics import deterministic_surface_sample
from pose_point_depth_mv.omni_real_benchmark_common import (
    atomic_json,
    canonical_sha256,
    index_objects,
    load_json,
    sha256_file,
    validate_bound_file,
)


REPORT_FORMAT = (
    "pose_point_depth_mv.holdout64_current_vs_reconviagen_normalized_shape.v1"
)
RECORD_FORMAT = (
    "pose_point_depth_mv.holdout64_current_vs_reconviagen_shape_record.v1"
)
RUN_CONFIG_FORMAT = (
    "pose_point_depth_mv.holdout64_current_vs_reconviagen_shape_config.v1"
)
CURRENT_MANIFEST_FORMAT = (
    "pose_point_depth_mv.omni_real_native_no_vggt_mixed_inference_manifest.v1"
)
RECONVIAGEN_MANIFEST_FORMAT = (
    "pose_point_depth_mv.omni_real_reconviagen_inference_manifest.v1"
)
CURRENT_RECORD_METHOD = "native_no_vggt_mixed"
RECONVIAGEN_RECORD_METHOD = "reconviagen_original"
EXPECTED_OBJECTS = 64
EXPECTED_SEED = 42
METHODS = ("current_point_mask", "reconviagen_original")
METRIC_FIELDS = (
    "pred_to_gt_mean",
    "gt_to_pred_mean",
    "chamfer_l1",
    "chamfer_l2",
    "fscore_0p01",
    "fscore_0p02",
    "normal_consistency",
)
LOWER_IS_BETTER = {
    "pred_to_gt_mean",
    "gt_to_pred_mean",
    "chamfer_l1",
    "chamfer_l2",
}

# MeshExtractResult.to_trimesh(transform_pose=True) maps decoder coordinates
# (x, y, z) to the reference/view convention (x, z, -y). ReconViaGen Holdout64
# meshes were exported with transform_pose=False, so this fixed proper rotation
# is part of the model-coordinate contract rather than a GT-fitted alignment.
RECONVIAGEN_DECODER_TO_REFERENCE = np.asarray(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


def _validate_axis_transform() -> None:
    rotation = RECONVIAGEN_DECODER_TO_REFERENCE[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-12):
        raise RuntimeError("ReconViaGen decoder axis transform is not rigid")
    if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1.0e-12):
        raise RuntimeError("ReconViaGen decoder axis transform is not proper")


def normalize_mesh_bbox(mesh: trimesh.Trimesh) -> tuple[trimesh.Trimesh, dict[str, Any]]:
    """Center at the AABB midpoint and scale the longest AABB side to one."""

    normalized = mesh.copy()
    vertices = np.asarray(normalized.vertices, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices):
        raise ValueError("cannot normalize an empty Mesh")
    if not np.isfinite(vertices).all():
        raise ValueError("cannot normalize a non-finite Mesh")
    minimum = vertices.min(axis=0)
    maximum = vertices.max(axis=0)
    center = 0.5 * (minimum + maximum)
    extent = maximum - minimum
    scale = float(np.max(extent))
    if not math.isfinite(scale) or scale <= 1.0e-12:
        raise ValueError("cannot normalize a zero-extent Mesh")
    normalized.vertices = (vertices - center[None]) / scale
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] /= scale
    transform[:3, 3] = -center / scale
    normalized_vertices = np.asarray(normalized.vertices, dtype=np.float64)
    normalized_extent = np.ptp(normalized_vertices, axis=0)
    return normalized, {
        "center_before": center.tolist(),
        "extent_before": extent.tolist(),
        "uniform_scale_divisor": scale,
        "mesh_to_normalized": transform.tolist(),
        "bounds_min_after": normalized_vertices.min(axis=0).tolist(),
        "bounds_max_after": normalized_vertices.max(axis=0).tolist(),
        "max_extent_after": float(np.max(normalized_extent)),
    }


def surface_metrics_from_samples(
    predicted: tuple[np.ndarray, np.ndarray],
    target: tuple[np.ndarray, np.ndarray],
    *,
    thresholds: Iterable[float],
    workers: int,
) -> dict[str, float]:
    pred_points, pred_normals = predicted
    target_points, target_normals = target
    target_tree = cKDTree(target_points)
    pred_tree = cKDTree(pred_points)
    pred_distance, pred_index = target_tree.query(
        pred_points, k=1, workers=int(workers)
    )
    target_distance, target_index = pred_tree.query(
        target_points, k=1, workers=int(workers)
    )
    output = {
        "pred_to_gt_mean": float(np.mean(pred_distance)),
        "gt_to_pred_mean": float(np.mean(target_distance)),
        "chamfer_l1": float(
            0.5 * (np.mean(pred_distance) + np.mean(target_distance))
        ),
        "chamfer_l2": float(
            0.5 * (np.mean(pred_distance**2) + np.mean(target_distance**2))
        ),
        "normal_consistency": float(
            0.5
            * (
                np.mean(
                    np.abs(
                        np.sum(
                            pred_normals * target_normals[pred_index], axis=1
                        )
                    )
                )
                + np.mean(
                    np.abs(
                        np.sum(
                            target_normals * pred_normals[target_index], axis=1
                        )
                    )
                )
            )
        ),
    }
    for threshold in thresholds:
        value = float(threshold)
        key = str(value).replace(".", "p")
        precision = float(np.mean(pred_distance < value))
        recall = float(np.mean(target_distance < value))
        output[f"precision_{key}"] = precision
        output[f"recall_{key}"] = recall
        output[f"fscore_{key}"] = (
            0.0
            if precision + recall <= 1.0e-12
            else float(2.0 * precision * recall / (precision + recall))
        )
    return output


def _records_by_key(
    manifest: dict[str, Any],
    *,
    manifest_format: str,
    record_method: str,
    label: str,
) -> dict[str, dict[str, Any]]:
    if (
        manifest.get("format") != manifest_format
        or manifest.get("method") != record_method
        or manifest.get("seeds") != [EXPECTED_SEED]
        or manifest.get("passed") is not True
    ):
        raise RuntimeError(f"{label} inference manifest contract differs")
    rows: dict[str, dict[str, Any]] = {}
    for row in manifest.get("objects", []):
        key = str(row.get("object_key", ""))
        if (
            not key
            or key in rows
            or row.get("method") != record_method
            or int(row.get("seed", -1)) != EXPECTED_SEED
            or row.get("passed") is not True
        ):
            raise RuntimeError(f"invalid or duplicate {label} inference record: {key}")
        rows[key] = row
    if (
        len(rows) != EXPECTED_OBJECTS
        or int(manifest.get("object_count", -1)) != EXPECTED_OBJECTS
        or int(manifest.get("record_count", -1)) != EXPECTED_OBJECTS
    ):
        raise RuntimeError(f"{label} does not provide exact Holdout64 coverage")
    return rows


def _validate_runtime_binding(
    manifest: dict[str, Any], *, reference_sha256: str, label: str
) -> dict[str, str]:
    declared = str(manifest.get("runtime_input_manifest_sha256", ""))
    path = validate_bound_file(
        manifest.get("runtime_input_manifest", ""), declared, label=f"{label} runtime"
    )
    if declared != reference_sha256:
        raise RuntimeError(f"{label} does not bind the GT reference runtime")
    return {"path": str(path), "sha256": declared}


def paired_comparison(records: list[dict[str, Any]]) -> dict[str, Any]:
    indexed = {
        method: {
            str(row["object_key"]): row
            for row in records
            if row["method"] == method
        }
        for method in METHODS
    }
    if set(indexed[METHODS[0]]) != set(indexed[METHODS[1]]):
        raise RuntimeError("current and ReconViaGen metric coverage differs")
    metrics: dict[str, Any] = {}
    for field in METRIC_FIELDS:
        values = []
        per_object = {}
        for key in sorted(indexed[METHODS[0]]):
            current = float(indexed[METHODS[0]][key][field])
            recon = float(indexed[METHODS[1]][key][field])
            delta = recon - current if field in LOWER_IS_BETTER else current - recon
            values.append(delta)
            per_object[key] = delta
        array = np.asarray(values, dtype=np.float64)
        metrics[f"{field}_current_improvement"] = {
            **summarize(values),
            "positive_rate": float(np.mean(array > 0.0)),
            "nonnegative_rate": float(np.mean(array >= 0.0)),
            "current_win_count": int(np.sum(array > 0.0)),
            "tie_count": int(np.sum(array == 0.0)),
            "reconviagen_win_count": int(np.sum(array < 0.0)),
            "per_object": per_object,
        }
    return {
        "left": METHODS[0],
        "right": METHODS[1],
        "positive_definition": "positive means current Point+Mask is better",
        "metrics": metrics,
    }


def _method_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        **{
            field: summarize([float(row[field]) for row in records])
            for field in METRIC_FIELDS
        },
        "record_count": len(records),
    }


def _record_identity(
    *,
    object_key_value: str,
    target_sha256: str,
    current_sha256: str,
    reconviagen_sha256: str,
    surface_seed: int,
    run_config_sha256: str,
) -> dict[str, Any]:
    return {
        "format": RECORD_FORMAT,
        "object_key": object_key_value,
        "seed": EXPECTED_SEED,
        "target_mesh_sha256": target_sha256,
        "current_mesh_sha256": current_sha256,
        "reconviagen_mesh_sha256": reconviagen_sha256,
        "surface_seed": int(surface_seed),
        "run_config_sha256": run_config_sha256,
    }


def _load_reusable_record(
    path: Path, *, identity: dict[str, Any], resume: bool
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    if not resume:
        raise RuntimeError(f"metric record exists; pass --resume: {path}")
    record = load_json(path)
    mismatch = {
        key: (record.get(key), value)
        for key, value in identity.items()
        if record.get(key) != value
    }
    if mismatch or record.get("passed") is not True:
        raise RuntimeError(f"stale normalized-shape record={mismatch}: {path}")
    return record


def _surface_seed(position: int) -> int:
    return EXPECTED_SEED * 1009 + int(position) * 9173


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label_manifest", required=True)
    parser.add_argument("--current_manifest", required=True)
    parser.add_argument("--reconviagen_manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--surface_samples", type=int, default=20000)
    parser.add_argument("--workers", type=int, default=-1)
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> None:
    args = make_parser().parse_args()
    if int(args.surface_samples) <= 0:
        raise ValueError("surface_samples must be positive")
    if int(args.workers) == 0 or int(args.workers) < -1:
        raise ValueError("workers must be -1 or a positive integer")
    _validate_axis_transform()

    label_path = Path(args.label_manifest).expanduser().resolve()
    current_path = Path(args.current_manifest).expanduser().resolve()
    recon_path = Path(args.reconviagen_manifest).expanduser().resolve()
    labels = load_json(label_path)
    current_manifest = load_json(current_path)
    recon_manifest = load_json(recon_path)
    if labels.get("format") != LABEL_MANIFEST_FORMAT or labels.get("passed") is not True:
        raise RuntimeError("Holdout64 runtime-O GT label manifest did not pass")
    label_rows = index_objects(labels.get("objects", []), label="Holdout64 labels")
    if (
        len(label_rows) != EXPECTED_OBJECTS
        or int(labels.get("selected_object_count", -1)) != EXPECTED_OBJECTS
        or int(labels.get("completed_object_count", -1)) != EXPECTED_OBJECTS
    ):
        raise RuntimeError("GT labels do not provide exact Holdout64 coverage")
    current_rows = _records_by_key(
        current_manifest,
        manifest_format=CURRENT_MANIFEST_FORMAT,
        record_method=CURRENT_RECORD_METHOD,
        label="current Point+Mask",
    )
    recon_rows = _records_by_key(
        recon_manifest,
        manifest_format=RECONVIAGEN_MANIFEST_FORMAT,
        record_method=RECONVIAGEN_RECORD_METHOD,
        label="ReconViaGen",
    )
    if set(label_rows) != set(current_rows) or set(label_rows) != set(recon_rows):
        raise RuntimeError("GT/current/ReconViaGen Holdout64 object sets differ")

    runtime_sha256 = str(labels.get("runtime_input_manifest_sha256", ""))
    reference_runtime = validate_bound_file(
        labels.get("runtime_input_manifest", ""),
        runtime_sha256,
        label="GT reference runtime",
    )
    runtime_bindings = {
        "gt": {"path": str(reference_runtime), "sha256": runtime_sha256},
        "current_point_mask": _validate_runtime_binding(
            current_manifest,
            reference_sha256=runtime_sha256,
            label="current Point+Mask",
        ),
        "reconviagen_original": _validate_runtime_binding(
            recon_manifest,
            reference_sha256=runtime_sha256,
            label="ReconViaGen",
        ),
    }

    thresholds = [0.01, 0.02]
    run_config = {
        "format": RUN_CONFIG_FORMAT,
        "protocol_scope": "post_hoc_consumed_holdout64_normalized_shape_only",
        "formal": False,
        "holdout64_consumed": True,
        "label_manifest": {"path": str(label_path), "sha256": sha256_file(label_path)},
        "current_manifest": {
            "path": str(current_path),
            "sha256": sha256_file(current_path),
        },
        "reconviagen_manifest": {
            "path": str(recon_path),
            "sha256": sha256_file(recon_path),
        },
        "runtime_bindings": runtime_bindings,
        "object_count": EXPECTED_OBJECTS,
        "seed": EXPECTED_SEED,
        "surface_samples": int(args.surface_samples),
        "surface_thresholds_in_unit_longest_extent": thresholds,
        "normalization": {
            "applied_independently_to_each_mesh": True,
            "center": "axis_aligned_bounding_box_midpoint",
            "scale": "divide_by_longest_axis_aligned_bounding_box_extent",
            "target_fit_or_optimization": False,
        },
        "reconviagen_fixed_axis_transform": {
            "mapping": "decoder (x,y,z) -> reference (x,z,-y)",
            "matrix": RECONVIAGEN_DECODER_TO_REFERENCE.tolist(),
            "proper_rotation": True,
            "gt_fitted": False,
        },
        "forbidden_alignment": [
            "per-object rotation fit",
            "ICP",
            "reflection",
            "anisotropic scale",
            "best-of-transform selection",
        ],
        "implementation": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
    }
    run_config_sha256 = canonical_sha256(run_config)
    run_config["run_config_sha256"] = run_config_sha256

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "run_config.json"
    if config_path.is_file():
        if load_json(config_path) != run_config:
            raise RuntimeError(f"normalized-shape run config changed: {config_path}")
    elif any(output_dir.iterdir()):
        raise RuntimeError(f"unbound output directory is not empty: {output_dir}")
    else:
        atomic_json(config_path, run_config)

    report_path = output_dir / "report.json"
    if report_path.is_file():
        report = load_json(report_path)
        if (
            report.get("format") != REPORT_FORMAT
            or report.get("run_config_sha256") != run_config_sha256
            or report.get("passed") is not True
        ):
            raise RuntimeError(f"stale final normalized-shape report: {report_path}")
        print(json.dumps({
            "passed": True,
            "reused": True,
            "report": str(report_path),
            "summary": report["summary"],
            "paired_comparison": report["paired_comparison"],
        }, indent=2))
        return

    records: list[dict[str, Any]] = []
    ordered_keys = sorted(label_rows)
    for position, key in enumerate(ordered_keys):
        label_row = label_rows[key]
        current_row = current_rows[key]
        recon_row = recon_rows[key]
        target_path = validate_bound_file(
            label_row["mesh_o"], label_row["mesh_o_sha256"], label=f"GT {key}"
        )
        current_mesh_path = validate_bound_file(
            current_row["mesh"], current_row["mesh_sha256"], label=f"current {key}"
        )
        recon_mesh_path = validate_bound_file(
            recon_row["mesh"], recon_row["mesh_sha256"], label=f"ReconViaGen {key}"
        )
        record_path = (
            output_dir
            / "records"
            / str(label_row["category"])
            / f"{label_row['object_id']}.json"
        )
        surface_seed = _surface_seed(position)
        identity = _record_identity(
            object_key_value=key,
            target_sha256=str(label_row["mesh_o_sha256"]),
            current_sha256=str(current_row["mesh_sha256"]),
            reconviagen_sha256=str(recon_row["mesh_sha256"]),
            surface_seed=surface_seed,
            run_config_sha256=run_config_sha256,
        )
        reusable = _load_reusable_record(
            record_path, identity=identity, resume=bool(args.resume)
        )
        if reusable is not None:
            records.extend(reusable["methods"])
            print(f"[shape64:reuse] {position + 1}/64 object={key}", flush=True)
            continue

        target_mesh, target_normalization = normalize_mesh_bbox(load_mesh(target_path))
        current_mesh, current_normalization = normalize_mesh_bbox(
            load_mesh(current_mesh_path)
        )
        recon_mesh = load_mesh(recon_mesh_path)
        recon_mesh.apply_transform(RECONVIAGEN_DECODER_TO_REFERENCE)
        recon_mesh, recon_normalization = normalize_mesh_bbox(recon_mesh)

        target_samples = deterministic_surface_sample(
            target_mesh, int(args.surface_samples), surface_seed
        )
        method_rows = []
        for method, mesh, mesh_path, mesh_sha256, normalization in (
            (
                METHODS[0],
                current_mesh,
                current_mesh_path,
                current_row["mesh_sha256"],
                current_normalization,
            ),
            (
                METHODS[1],
                recon_mesh,
                recon_mesh_path,
                recon_row["mesh_sha256"],
                recon_normalization,
            ),
        ):
            prediction_samples = deterministic_surface_sample(
                mesh, int(args.surface_samples), surface_seed
            )
            metrics = surface_metrics_from_samples(
                prediction_samples,
                target_samples,
                thresholds=thresholds,
                workers=int(args.workers),
            )
            method_rows.append({
                "method": method,
                "object_key": key,
                "seed": EXPECTED_SEED,
                "source_mesh": str(mesh_path),
                "source_mesh_sha256": str(mesh_sha256),
                "target_mesh": str(target_path),
                "target_mesh_sha256": str(label_row["mesh_o_sha256"]),
                "surface_seed": surface_seed,
                "normalization": normalization,
                **{field: float(metrics[field]) for field in METRIC_FIELDS},
                "precision_0p01": float(metrics["precision_0p01"]),
                "recall_0p01": float(metrics["recall_0p01"]),
                "precision_0p02": float(metrics["precision_0p02"]),
                "recall_0p02": float(metrics["recall_0p02"]),
            })
        record = {
            **identity,
            "created_at_utc": utc_now(),
            "target_normalization": target_normalization,
            "alignment_quality_passed": label_row.get(
                "alignment_quality_passed", True
            ) is True,
            "methods": method_rows,
            "passed": True,
        }
        atomic_json(record_path, record)
        records.extend(method_rows)
        print(f"[shape64] {position + 1}/64 object={key}", flush=True)
        del target_mesh, current_mesh, recon_mesh, target_samples

    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        by_method[str(row["method"])].append(row)
    summary = {method: _method_summary(by_method[method]) for method in METHODS}
    comparison = paired_comparison(records)
    report = {
        "format": REPORT_FORMAT,
        "created_at_utc": utc_now(),
        "formal": False,
        "post_hoc": True,
        "holdout64_consumed": True,
        "protocol_scope": "post_hoc_consumed_holdout64_normalized_shape_only",
        "interpretation_scope": (
            "Normalized object geometry only; no runtime-O/world placement claim"
        ),
        "run_config": str(config_path),
        "run_config_sha256": run_config_sha256,
        "object_count": EXPECTED_OBJECTS,
        "record_count": len(records),
        "surface_samples": int(args.surface_samples),
        "surface_thresholds_in_unit_longest_extent": thresholds,
        "summary": summary,
        "paired_comparison": comparison,
        "records": records,
        "passed": (
            len(records) == EXPECTED_OBJECTS * len(METHODS)
            and all(len(by_method[method]) == EXPECTED_OBJECTS for method in METHODS)
        ),
    }
    atomic_json(report_path, report)
    print(json.dumps({
        "passed": report["passed"],
        "formal": report["formal"],
        "post_hoc": report["post_hoc"],
        "objects": report["object_count"],
        "records": report["record_count"],
        "summary": summary,
        "paired_comparison": comparison,
        "report": str(report_path),
    }, indent=2))
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
