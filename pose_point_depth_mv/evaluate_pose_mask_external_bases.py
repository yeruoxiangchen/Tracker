#!/usr/bin/env python3
"""Re-score Pose+Mask and frozen Benchmark32 bases with identical surface samples."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from pose_point_depth_mv.dataset_tools.prepare_omni_real_video_cache import utc_now
from pose_point_depth_mv.evaluate_omni_real_mesh_benchmark import (
    STRUCTURE_FIELDS,
    SURFACE_FIELDS,
    load_mesh,
    summarize,
)
from pose_point_depth_mv.evaluate_omni_real_no_vggt_final import (
    REPORT_FORMAT as FIVEWAY_REPORT_FORMAT,
)
from pose_point_depth_mv.evaluate_pose_mask_pointcloud_ablation import (
    METHODS as ABLATION_METHODS,
    REPORT_FORMAT as ABLATION_REPORT_FORMAT,
)
from pose_point_depth_mv.mesh_benchmark_metrics import (
    deterministic_surface_sample,
    mesh_structure_metrics,
)
from pose_point_depth_mv.omni_real_benchmark_common import (
    atomic_json,
    load_json,
    sha256_file,
    validate_bound_file,
)


REPORT_FORMAT = "pose_point_depth_mv.pose_mask_external_base_comparison.v1"
EXTERNAL_METHODS = (
    "reconviagen_original",
    "pixal3d_official",
    "real_adapted_native_v2_full",
    "synthetic_parent_native_v2_full",
)
METHODS = (*ABLATION_METHODS, *EXTERNAL_METHODS)


def _target_surface_cache(
    target: Any, *, count: int, seed: int
) -> tuple[np.ndarray, np.ndarray, cKDTree]:
    points, normals = deterministic_surface_sample(target, count, seed)
    return points, normals, cKDTree(points)


def _surface_metrics_with_cached_target(
    predicted: Any,
    *,
    count: int,
    seed: int,
    target_points: np.ndarray,
    target_normals: np.ndarray,
    target_tree: cKDTree,
) -> dict[str, float]:
    pred_points, pred_normals = deterministic_surface_sample(predicted, count, seed)
    pred_tree = cKDTree(pred_points)
    pred_distance, pred_index = target_tree.query(pred_points, k=1, workers=-1)
    target_distance, target_index = pred_tree.query(target_points, k=1, workers=-1)
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
    for threshold in (0.01, 0.02):
        key = str(float(threshold)).replace(".", "p")
        precision = float(np.mean(pred_distance < threshold))
        recall = float(np.mean(target_distance < threshold))
        output[f"precision_{key}"] = precision
        output[f"recall_{key}"] = recall
        output[f"fscore_{key}"] = (
            0.0
            if precision + recall <= 1.0e-12
            else float(2.0 * precision * recall / (precision + recall))
        )
    return output


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    output = {
        field: summarize([float(row[field]) for row in rows])
        for field in (*SURFACE_FIELDS, *STRUCTURE_FIELDS)
    }
    output["mesh_success_rate"] = float(
        np.mean([float(row["mesh_success"]) for row in rows])
    )
    output["record_count"] = len(rows)
    return output


def paired_method_comparison(
    records: list[dict[str, Any]], *, left: str, right: str
) -> dict[str, Any]:
    left_rows = {
        (str(row["object_key"]), int(row["seed"])): row
        for row in records
        if row["method"] == left
    }
    right_rows = {
        (str(row["object_key"]), int(row["seed"])): row
        for row in records
        if row["method"] == right
    }
    if not left_rows or set(left_rows) != set(right_rows):
        raise RuntimeError(f"paired coverage differs: {left}/{right}")
    metrics: dict[str, Any] = {}
    for field in SURFACE_FIELDS:
        sign = -1.0 if field.startswith("chamfer") else 1.0
        values = np.asarray(
            [
                sign
                * (
                    float(left_rows[pair][field])
                    - float(right_rows[pair][field])
                )
                for pair in sorted(left_rows)
            ],
            dtype=np.float64,
        )
        metrics[f"{field}_left_improvement"] = {
            **summarize(values.tolist()),
            "left_win_rate": float(np.mean(values > 0.0)),
            "left_nonnegative_rate": float(np.mean(values >= 0.0)),
            "left_win_count": int(np.sum(values > 0.0)),
            "tie_count": int(np.sum(values == 0.0)),
            "right_win_count": int(np.sum(values < 0.0)),
        }
    return {
        "left": left,
        "right": right,
        "positive_definition": f"positive means {left} is better than {right}",
        "metrics": metrics,
    }


def _index_records(
    records: list[dict[str, Any]], *, methods: tuple[str, ...], label: str
) -> dict[tuple[str, str, int], dict[str, Any]]:
    indexed: dict[tuple[str, str, int], dict[str, Any]] = {}
    allowed = set(methods)
    for row in records:
        method = str(row.get("method"))
        if method not in allowed:
            continue
        identity = (method, str(row["object_key"]), int(row["seed"]))
        if identity in indexed:
            raise RuntimeError(f"duplicate {label} record={identity}")
        indexed[identity] = row
    return indexed


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ablation_report", action="append", required=True)
    parser.add_argument("--fiveway_report", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--surface_samples", type=int, default=20000)
    parser.add_argument("--expected_objects", type=int, default=32)
    return parser


def main() -> None:
    args = make_parser().parse_args()
    if int(args.surface_samples) <= 0 or int(args.expected_objects) <= 0:
        raise ValueError("surface_samples and expected_objects must be positive")

    fiveway_path = Path(args.fiveway_report).expanduser().resolve()
    fiveway = load_json(fiveway_path)
    if (
        fiveway.get("format") != FIVEWAY_REPORT_FORMAT
        or fiveway.get("passed") is not True
        or fiveway.get("formal") is not False
        or fiveway.get("protocol_scope") != "development_benchmark32"
        or int(fiveway.get("object_count", -1)) != 32
    ):
        raise RuntimeError(f"five-way Benchmark32 report did not pass: {fiveway_path}")
    label_path = validate_bound_file(
        fiveway["label_manifest"],
        fiveway["label_manifest_sha256"],
        label="five-way label manifest",
    )
    for method in EXTERNAL_METHODS:
        binding = fiveway.get("method_manifests", {}).get(method, {})
        validate_bound_file(
            binding.get("path", ""),
            binding.get("sha256", ""),
            label=f"five-way {method} manifest",
        )
    fiveway_rows = _index_records(
        fiveway.get("records", []), methods=EXTERNAL_METHODS, label="five-way"
    )

    ablation_paths = [Path(value).expanduser().resolve() for value in args.ablation_report]
    ablation_reports: list[tuple[Path, dict[str, Any]]] = []
    ablation_rows: dict[tuple[str, str, int], dict[str, Any]] = {}
    surface_seed_by_pair: dict[tuple[str, int], int] = {}
    target_by_pair: dict[tuple[str, int], dict[str, str]] = {}
    object_keys: set[str] = set()
    for path in ablation_paths:
        report = load_json(path)
        binding = report.get("bindings", {}).get("label_manifest", {})
        if (
            report.get("format") != ABLATION_REPORT_FORMAT
            or report.get("passed") is not True
            or report.get("formal") is not False
            or int(report.get("surface_samples", -1)) != int(args.surface_samples)
            or Path(str(binding.get("path", ""))).resolve() != label_path
            or str(binding.get("sha256")) != str(fiveway["label_manifest_sha256"])
        ):
            raise RuntimeError(f"ablation report protocol binding failed: {path}")
        selected = [str(value) for value in report.get("selected_object_keys", [])]
        overlap = object_keys.intersection(selected)
        if overlap:
            raise RuntimeError(f"ablation reports overlap: {sorted(overlap)}")
        object_keys.update(selected)
        indexed = _index_records(
            report.get("records", []), methods=ABLATION_METHODS, label=str(path)
        )
        expected = {
            (method, key, 42) for method in ABLATION_METHODS for key in selected
        }
        if set(indexed) != expected:
            raise RuntimeError(f"ablation report lacks a complete 2 x N product: {path}")
        for key in selected:
            pair = (key, 42)
            point = indexed[("point_mask", key, 42)]
            pose = indexed[("pose_mask", key, 42)]
            if (
                int(point["surface_seed"]) != int(pose["surface_seed"])
                or point["target_mesh"] != pose["target_mesh"]
            ):
                raise RuntimeError(f"A/B surface binding differs: {pair}")
            surface_seed_by_pair[pair] = int(pose["surface_seed"])
            target_by_pair[pair] = dict(pose["target_mesh"])
        ablation_rows.update(indexed)
        ablation_reports.append((path, report))

    if len(object_keys) != int(args.expected_objects):
        raise RuntimeError(
            f"expected {int(args.expected_objects)} unique objects, got {len(object_keys)}"
        )

    records: list[dict[str, Any]] = []
    for position, key in enumerate(sorted(object_keys), start=1):
        pair = (key, 42)
        target_binding = target_by_pair[pair]
        target_path = validate_bound_file(
            target_binding["path"], target_binding["sha256"], label=f"target {key}"
        )
        target = load_mesh(target_path)
        sample_seed = surface_seed_by_pair[pair]
        target_points, target_normals, target_tree = _target_surface_cache(
            target, count=int(args.surface_samples), seed=sample_seed
        )
        for method in METHODS:
            source = (
                ablation_rows[(method, key, 42)]
                if method in ABLATION_METHODS
                else fiveway_rows[(method, key, 42)]
            )
            source_target = source.get("target_mesh", target_binding)
            if (
                Path(str(source_target.get("path", ""))).resolve() != target_path
                or str(source_target.get("sha256")) != str(target_binding["sha256"])
            ):
                raise RuntimeError(f"target identity differs for {method}/{key}")
            mesh_path = validate_bound_file(
                source["mesh"], source["mesh_sha256"], label=f"{method} {key}"
            )
            mesh = load_mesh(mesh_path)
            structure = mesh_structure_metrics(mesh)
            surface = _surface_metrics_with_cached_target(
                mesh,
                count=int(args.surface_samples),
                seed=sample_seed,
                target_points=target_points,
                target_normals=target_normals,
                target_tree=target_tree,
            )
            records.append(
                {
                    "method": method,
                    "object_key": key,
                    "seed": 42,
                    "mesh": str(mesh_path),
                    "mesh_sha256": str(source["mesh_sha256"]),
                    "target_mesh": {
                        "path": str(target_path),
                        "sha256": target_binding["sha256"],
                    },
                    "surface_seed": sample_seed,
                    "mesh_success": bool(structure["mesh_success"]),
                    **{field: float(surface[field]) for field in SURFACE_FIELDS},
                    **{field: float(structure[field]) for field in STRUCTURE_FIELDS},
                }
            )
        print(
            f"[pose_mask_external_bases] {position}/{len(object_keys)} object={key}",
            flush=True,
        )

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[str(row["method"])].append(row)
    comparisons = {
        method: paired_method_comparison(
            records, left="pose_mask", right=method
        )
        for method in METHODS
        if method != "pose_mask"
    }
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "format": REPORT_FORMAT,
        "created_at_utc": utc_now(),
        "passed": True,
        "formal": False,
        "protocol_scope": "Benchmark32 shared-surface-seed point-cloud-removal evaluation",
        "question": (
            "whether Pose+Mask retains its advantage over frozen external bases after "
            "removing point-cloud input"
        ),
        "object_count": len(object_keys),
        "record_count": len(records),
        "methods": list(METHODS),
        "selected_object_keys": sorted(object_keys),
        "surface_samples": int(args.surface_samples),
        "surface_thresholds": [0.01, 0.02],
        "surface_seed_policy": (
            "reuse each paired Pose+Mask ablation surface_seed for every method"
        ),
        "coordinate_policy": (
            "all meshes are already expressed in frozen Benchmark32 reference O; "
            "no GT ICP or per-method fit"
        ),
        "bindings": {
            "ablation_reports": [
                {"path": str(path), "sha256": sha256_file(path)}
                for path, _ in ablation_reports
            ],
            "fiveway_report": {
                "path": str(fiveway_path),
                "sha256": sha256_file(fiveway_path),
            },
            "label_manifest": {
                "path": str(label_path),
                "sha256": str(fiveway["label_manifest_sha256"]),
            },
        },
        "summary": {method: _summary(grouped[method]) for method in METHODS},
        "pose_mask_paired_comparisons": comparisons,
        "records": records,
        "interpretation_guard": (
            "Benchmark32 is a development benchmark. A protocol pass is not a formal "
            "holdout pass and does not imply every object or metric wins."
        ),
    }
    report_path = output_dir / "report.json"
    atomic_json(report_path, report)
    print(
        {
            "passed": True,
            "formal": False,
            "objects": len(object_keys),
            "comparisons": comparisons,
            "report": str(report_path),
        }
    )


if __name__ == "__main__":
    main()
