#!/usr/bin/env python3
"""Evaluate original ReconViaGen, Direct stock and Direct Full on matched inputs.

This evaluator consumes:

* one ``export_direct_slat_mesh_pairs`` report containing Direct stock/full;
* one original-ReconViaGen report for every Direct ``(uid, joint_seed)`` record.

The comparison is intentionally stricter than the older Direct-SLAT reports:

* the original ReconViaGen report must bind the exact RGB/mask files used by the
  Direct source-lifting cache;
* all three decoder meshes receive the fixed vendored
  ``transform_pose=True`` axis conversion before comparison to the normalized
  source GLB;
* no per-object scale, reflection, bbox normalization or GT fit is used in the
  primary ``canonical_pose`` result;
* directed surface distance, quantiles, outlier fractions, precision, recall
  and F-score are retained instead of reducing everything to mean Chamfer.

An optional proper-SE(3) ICP result can be generated as a shape-only diagnostic.
It is never the primary AR/world-coordinate result.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.spatial import cKDTree
import torch
import trimesh

from .bunny_review.common import (
    atomic_json,
    atomic_text,
    binding,
    canonical_sha256,
    code_bindings,
    sha256_file,
    validate_binding,
)
from .bunny_review.finalize import load_mesh
from .eval_direct_slat_flow import bootstrap_mean_ci, summarize
from .export_direct_flow_mesh_pairs import (
    deterministic_surface_sample,
    mesh_structure_metrics,
)
from .render_direct_slat_fourway import (
    LATENT_DECODER_TO_REFERENCE,
    affine_audit,
    rigid_icp,
)


FORMAT = "pose_point_depth_mv.reconviagen_stock_full_mesh_accuracy.v1"
RECON_REPORT_FORMAT = "pose_point_depth_mv.reconviagen_stock_from_direct_cache.v1"
DIRECT_REPORT_FORMATS = frozenset(
    {
        "pose_point_depth_mv.direct_slat_mesh_exploratory.v1",
        "pose_point_depth_mv.direct_slat_mesh_exploratory.v2",
        "pose_point_depth_mv.direct_slat_matched_mesh_blind_report.v1",
    }
)
METHOD_IDS = (
    "reconviagen_original",
    "direct_stock",
    "direct_full",
)
COMPARISONS = (
    ("full_minus_reconviagen", "direct_full", "reconviagen_original"),
    ("stock_minus_reconviagen", "direct_stock", "reconviagen_original"),
    ("full_minus_stock", "direct_full", "direct_stock"),
)
LOWER_IS_BETTER_PREFIXES = (
    "pred_to_gt_",
    "gt_to_pred_",
    "chamfer_",
)
HIGHER_IS_BETTER_PREFIXES = (
    "precision_",
    "recall_",
    "fscore_",
)
HIGHER_IS_BETTER_EXACT = frozenset(
    {
        "normal_consistency",
        "largest_component_ratio",
    }
)
ABSOLUTE_STRUCTURE_KEYS = (
    "largest_component_ratio",
    "component_count",
    "small_component_vertex_ratio_lt100",
    "boundary_edge_count",
    "boundary_total_length",
    "nonmanifold_edge_count",
    "is_watertight",
    "is_winding_consistent",
)


def parse_csv(value: str, cast) -> list[Any]:
    result = [cast(item.strip()) for item in str(value).split(",") if item.strip()]
    if not result or len(result) != len(set(result)):
        raise ValueError("CSV values must be non-empty and unique")
    return result


def resolve_from(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def load_json(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).resolve()
    value = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root is not an object: {resolved}")
    return value


def numeric(value: Any) -> bool:
    return isinstance(value, (bool, int, float, np.number))


def binding_equal(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return (
        Path(str(left.get("path", ""))).resolve()
        == Path(str(right.get("path", ""))).resolve()
        and str(left.get("sha256", "")) == str(right.get("sha256", ""))
        and (
            "bytes" not in left
            or "bytes" not in right
            or int(left["bytes"]) == int(right["bytes"])
        )
    )


def assert_binding_equal(
    actual: dict[str, Any],
    expected: dict[str, Any],
    *,
    label: str,
) -> None:
    validate_binding(actual, label=f"{label}.actual")
    validate_binding(expected, label=f"{label}.expected")
    if not binding_equal(actual, expected):
        raise RuntimeError(
            f"{label} differs: actual={actual!r}, expected={expected!r}"
        )


def mesh_in_reference_frame(path: str | Path) -> trimesh.Trimesh:
    mesh = load_mesh(Path(path).resolve())
    mesh.apply_transform(LATENT_DECODER_TO_REFERENCE)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    if not len(vertices) or not np.isfinite(vertices).all() or not len(mesh.faces):
        raise RuntimeError(f"empty/non-finite Mesh after axis conversion: {path}")
    return mesh


def load_target(metadata: dict[str, Any]) -> tuple[trimesh.Trimesh, dict[str, Any]]:
    source_path = Path(str(metadata["source_glb"])).resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if sha256_file(source_path) != str(metadata["source_glb_sha256"]):
        raise RuntimeError(f"target source GLB changed: {source_path}")
    center = np.asarray(metadata["normalize_center"], dtype=np.float64)
    scale = float(metadata["normalize_scale"])
    margin = float(metadata["canonical_margin"])
    if center.shape != (3,) or not np.isfinite(center).all():
        raise ValueError("target normalization center is invalid")
    if not math.isfinite(scale) or scale <= 0 or not math.isfinite(margin):
        raise ValueError("target normalization scale/margin is invalid")
    target = load_mesh(source_path)
    target.vertices = (
        (np.asarray(target.vertices, dtype=np.float64) - center[None])
        / scale
        * margin
    )
    return target, {
        "source_glb": binding(source_path),
        "normalize_center": center.tolist(),
        "normalize_scale": scale,
        "canonical_margin": margin,
        "frame": "normalized source-GLB/view frame",
    }


def distance_summary(
    distance: np.ndarray,
    *,
    prefix: str,
    thresholds: Iterable[float],
) -> dict[str, float]:
    value = np.asarray(distance, dtype=np.float64)
    if value.ndim != 1 or not len(value) or not np.isfinite(value).all():
        raise ValueError(f"{prefix} distance array is invalid")
    output = {
        f"{prefix}_mean": float(np.mean(value)),
        f"{prefix}_median": float(np.median(value)),
        f"{prefix}_p90": float(np.quantile(value, 0.90)),
        f"{prefix}_p95": float(np.quantile(value, 0.95)),
        f"{prefix}_p99": float(np.quantile(value, 0.99)),
        f"{prefix}_max": float(np.max(value)),
    }
    for threshold in thresholds:
        key = str(float(threshold)).replace(".", "p")
        output[f"{prefix}_outlier_ratio_{key}"] = float(
            np.mean(value >= float(threshold))
        )
    return output


def extended_surface_metrics(
    predicted: trimesh.Trimesh,
    target: trimesh.Trimesh,
    *,
    count: int,
    seed: int,
    thresholds: Iterable[float],
) -> dict[str, float]:
    """Compute deterministic directed and symmetric surface metrics."""

    thresholds = tuple(float(value) for value in thresholds)
    pred_points, pred_normals = deterministic_surface_sample(
        predicted, int(count), int(seed)
    )
    target_points, target_normals = deterministic_surface_sample(
        target, int(count), int(seed)
    )
    target_tree = cKDTree(target_points)
    pred_tree = cKDTree(pred_points)
    pred_distance, pred_index = target_tree.query(
        pred_points, k=1, workers=-1
    )
    target_distance, target_index = pred_tree.query(
        target_points, k=1, workers=-1
    )
    output = {
        **distance_summary(
            pred_distance,
            prefix="pred_to_gt",
            thresholds=thresholds,
        ),
        **distance_summary(
            target_distance,
            prefix="gt_to_pred",
            thresholds=thresholds,
        ),
        "chamfer_l1": float(
            0.5 * (np.mean(pred_distance) + np.mean(target_distance))
        ),
        "chamfer_l2": float(
            0.5
            * (
                np.mean(np.square(pred_distance))
                + np.mean(np.square(target_distance))
            )
        ),
        "normal_consistency": float(
            0.5
            * (
                np.mean(
                    np.abs(
                        np.sum(
                            pred_normals * target_normals[pred_index],
                            axis=1,
                        )
                    )
                )
                + np.mean(
                    np.abs(
                        np.sum(
                            target_normals * pred_normals[target_index],
                            axis=1,
                        )
                    )
                )
            )
        ),
    }
    for threshold in thresholds:
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


def metric_delta(lhs: float, rhs: float, metric: str) -> float:
    """Return a signed delta where positive always means ``lhs`` is better."""

    if metric.startswith(LOWER_IS_BETTER_PREFIXES):
        return float(rhs - lhs)
    if metric.startswith(HIGHER_IS_BETTER_PREFIXES) or metric in HIGHER_IS_BETTER_EXACT:
        return float(lhs - rhs)
    raise KeyError(f"metric direction is not registered: {metric}")


def metric_distribution(
    values: list[float],
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array) or not np.isfinite(array).all():
        raise ValueError("metric distribution is empty or non-finite")
    return {
        "count": int(len(array)),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
        "bootstrap_mean_95_ci": bootstrap_mean_ci(
            [float(value) for value in array],
            samples=int(bootstrap_samples),
            seed=int(seed),
        ),
    }


def resolve_recon_reports(args: argparse.Namespace) -> dict[tuple[str, int], Path]:
    paths = [Path(value).resolve() for value in args.recon_report]
    for root_value in args.recon_root:
        root = Path(root_value).resolve()
        if not root.exists():
            raise FileNotFoundError(root)
        paths.extend(sorted(root.rglob("report.json")))
    unique_paths = sorted(set(paths))
    if not unique_paths:
        raise ValueError("provide at least one --recon_report or --recon_root")
    index: dict[tuple[str, int], Path] = {}
    for path in unique_paths:
        report = load_json(path)
        if (
            report.get("format") != RECON_REPORT_FORMAT
            or report.get("complete") is not True
        ):
            # A broad root can contain unrelated reports; explicit paths cannot.
            if path in paths[: len(args.recon_report)]:
                raise RuntimeError(f"unsupported ReconViaGen report: {path}")
            continue
        run_config = dict(report["run_config"])
        key = (str(run_config["uid"]), int(run_config["seed"]))
        if key in index:
            raise RuntimeError(
                f"duplicate ReconViaGen reports for uid/seed={key}: "
                f"{index[key]} and {path}"
            )
        index[key] = path
    if not index:
        raise RuntimeError("no matched ReconViaGen reports were found")
    return index


def indexed_samples(
    payload: dict[str, Any],
    *,
    seed_key: str | None,
) -> dict[Any, dict[str, Any]]:
    result: dict[Any, dict[str, Any]] = {}
    for row in payload.get("samples", []):
        key = (
            (str(row["uid"]), int(row[seed_key]))
            if seed_key is not None
            else str(row["uid"])
        )
        if key in result:
            raise RuntimeError(f"duplicate manifest sample key={key!r}")
        result[key] = row
    return result


def expected_input_bindings(
    source_manifest_path: Path,
    source_row: dict[str, Any],
) -> dict[str, Any]:
    cache_path = resolve_from(
        source_manifest_path.parent,
        str(source_row["cache_file"]),
    )
    if not cache_path.is_file():
        raise FileNotFoundError(cache_path)
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    if not isinstance(cache, dict) or str(cache.get("uid")) != str(source_row["uid"]):
        raise RuntimeError(f"source lifting cache identity mismatch: {cache_path}")
    images = [binding(Path(value).resolve()) for value in cache.get("image_paths", [])]
    masks = [binding(Path(value).resolve()) for value in cache.get("mask_paths", [])]
    if not images or len(images) != len(masks):
        raise RuntimeError(f"invalid image/mask bindings in {cache_path}")
    view_ids_value = cache.get("view_ids")
    if torch.is_tensor(view_ids_value):
        view_ids = [int(value) for value in view_ids_value.cpu().tolist()]
    else:
        view_ids = [int(value) for value in view_ids_value]
    if len(view_ids) != len(images):
        raise RuntimeError(f"view ID count differs from image count in {cache_path}")
    return {
        "source_cache": binding(cache_path),
        "input_images": images,
        "input_masks": masks,
        "view_ids": view_ids,
    }


def validate_same_input(
    *,
    uid: str,
    object_uid: str,
    seed: int,
    direct_manifest: dict[str, Any],
    direct_row: dict[str, Any],
    source_manifest_path: Path,
    source_row: dict[str, Any],
    recon_report_path: Path,
    recon_report: dict[str, Any],
) -> dict[str, Any]:
    if str(direct_row["uid"]) != uid or str(direct_row["object_uid"]) != object_uid:
        raise RuntimeError("Direct cache sample identity differs from Mesh record")
    if int(direct_row["support_seed"]) != int(seed):
        raise RuntimeError("Direct cache support seed differs from Mesh record")
    if str(source_row["uid"]) != uid or str(source_row["object_uid"]) != object_uid:
        raise RuntimeError("source lifting sample identity differs from Mesh record")
    if int(direct_row.get("view_count", direct_row.get("views", -1))) != int(
        source_row["view_count"]
    ):
        raise RuntimeError("Direct and source-lifting view counts differ")

    expected = expected_input_bindings(source_manifest_path, source_row)
    run_config = dict(recon_report["run_config"])
    if str(run_config["uid"]) != uid or str(run_config["object_uid"]) != object_uid:
        raise RuntimeError("ReconViaGen report identity differs from Direct record")
    if int(run_config["seed"]) != int(seed):
        raise RuntimeError("ReconViaGen integer seed differs from Direct joint seed")

    assert_binding_equal(
        run_config["source_lifting_manifest"],
        binding(source_manifest_path),
        label="source_lifting_manifest",
    )
    assert_binding_equal(
        run_config["source_cache"],
        expected["source_cache"],
        label="source_cache",
    )
    actual_images = list(run_config["input_images"])
    actual_masks = list(run_config["input_masks"])
    if len(actual_images) != len(expected["input_images"]) or len(actual_masks) != len(
        expected["input_masks"]
    ):
        raise RuntimeError("ReconViaGen input image/mask count differs")
    for position, (actual, wanted) in enumerate(
        zip(actual_images, expected["input_images"])
    ):
        assert_binding_equal(actual, wanted, label=f"input_images[{position}]")
    for position, (actual, wanted) in enumerate(
        zip(actual_masks, expected["input_masks"])
    ):
        assert_binding_equal(actual, wanted, label=f"input_masks[{position}]")
    if [int(value) for value in run_config["view_ids"]] != expected["view_ids"]:
        raise RuntimeError("ReconViaGen view IDs differ from Direct source input")

    return {
        "passed": True,
        "uid": uid,
        "object_uid": object_uid,
        "integer_seed": int(seed),
        "view_count": len(expected["view_ids"]),
        "view_ids": expected["view_ids"],
        "source_lifting_manifest": binding(source_manifest_path),
        "source_cache": expected["source_cache"],
        "input_images": expected["input_images"],
        "input_masks": expected["input_masks"],
        "direct_condition_binding": {
            "condition_file": str(direct_row["condition_file"]),
            "condition_file_sha256": str(direct_row["condition_file_sha256"]),
            "source_lifting_manifest_sha256": str(
                direct_manifest["source_lifting_manifest_sha256"]
            ),
        },
        "reconviagen_report": binding(recon_report_path),
        "seed_scope": (
            "same integer seed ID only; original ReconViaGen and Direct use "
            "different SS/SLAT state spaces, so cross-architecture latent noise "
            "is not claimed bit-exact"
        ),
    }


def aggregate_results(
    records: list[dict[str, Any]],
    *,
    surface_keys: list[str],
    bootstrap_samples: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Aggregate seeds within object before computing object-level statistics."""

    by_object: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        by_object[str(row["object_uid"])].append(row)
    object_rows: list[dict[str, Any]] = []
    for object_uid, values in sorted(by_object.items()):
        method_rows: dict[str, Any] = {}
        for method_id in METHOD_IDS:
            method_rows[method_id] = {
                key: float(
                    np.mean(
                        [
                            float(row["methods"][method_id]["surface"][key])
                            for row in values
                        ]
                    )
                )
                for key in surface_keys
            }
            for key in ABSOLUTE_STRUCTURE_KEYS:
                structure_values = [
                    row["methods"][method_id]["structure"].get(key)
                    for row in values
                ]
                if all(numeric(value) for value in structure_values):
                    method_rows[method_id][key] = float(
                        np.mean([float(value) for value in structure_values])
                    )
        comparison_rows: dict[str, Any] = {}
        for comparison_id, _, _ in COMPARISONS:
            metric_names = values[0]["comparisons"][comparison_id]["metrics"]
            comparison_rows[comparison_id] = {
                key: float(
                    np.mean(
                        [
                            float(
                                row["comparisons"][comparison_id]["metrics"][key]
                            )
                            for row in values
                        ]
                    )
                )
                for key in metric_names
            }
        object_rows.append(
            {
                "object_uid": object_uid,
                "seed_count": len(values),
                "methods": method_rows,
                "comparisons": comparison_rows,
            }
        )

    method_summary: dict[str, Any] = {}
    for method_position, method_id in enumerate(METHOD_IDS):
        keys = list(object_rows[0]["methods"][method_id])
        method_summary[method_id] = {
            key: metric_distribution(
                [
                    float(row["methods"][method_id][key])
                    for row in object_rows
                ],
                bootstrap_samples=int(bootstrap_samples),
                seed=51000 + method_position * 1000 + key_position,
            )
            for key_position, key in enumerate(keys)
        }

    comparison_summary: dict[str, Any] = {}
    for comparison_position, (comparison_id, lhs, rhs) in enumerate(COMPARISONS):
        keys = list(object_rows[0]["comparisons"][comparison_id])
        comparison_summary[comparison_id] = {
            "lhs": lhs,
            "rhs": rhs,
            "positive_means_lhs_better": True,
            "metrics": {
                key: summarize(
                    [
                        float(row["comparisons"][comparison_id][key])
                        for row in object_rows
                    ],
                    bootstrap_samples=int(bootstrap_samples),
                    seed=61000 + comparison_position * 1000 + key_position,
                )
                for key_position, key in enumerate(keys)
            },
        }
    return object_rows, method_summary, comparison_summary


def make_summary_text(report: dict[str, Any]) -> str:
    lines = [
        "ReconViaGen original / Direct stock / Direct Full Mesh accuracy",
        "================================================================",
        f"formal: {str(report['formal']).lower()}",
        f"objects: {report['object_count']}",
        f"records: {report['record_count']}",
        f"seeds: {report['joint_seeds']}",
        "",
    ]
    for mode_id, mode in report["modes"].items():
        lines.append(f"[{mode_id}] {mode['description']}")
        for method_id in METHOD_IDS:
            metrics = mode["method_summary"][method_id]
            lines.append(
                f"{method_id}: "
                f"chamfer={metrics['chamfer_l1']['mean']:.8f} "
                f"pred_p95={metrics['pred_to_gt_p95']['mean']:.8f} "
                f"gt_p95={metrics['gt_to_pred_p95']['mean']:.8f} "
                f"f@0.02={metrics['fscore_0p02']['mean']:.8f} "
                f"normal={metrics['normal_consistency']['mean']:.8f}"
            )
        for comparison_id, _, _ in COMPARISONS:
            comparison = mode["comparison_summary"][comparison_id]
            metrics = comparison["metrics"]
            lines.append(
                f"{comparison_id}: "
                f"chamfer={metrics['chamfer_l1']['mean']:+.8f} "
                f"pred_p95={metrics['pred_to_gt_p95']['mean']:+.8f} "
                f"gt_p95={metrics['gt_to_pred_p95']['mean']:+.8f} "
                f"f@0.02={metrics['fscore_0p02']['mean']:+.8f} "
                f"normal={metrics['normal_consistency']['mean']:+.8f}"
            )
        lines.append("")
    lines.extend(
        [
            "Positive comparison deltas always mean the left-hand method is better.",
            "canonical_pose is primary: fixed decoder axis conversion, no GT fit.",
            "rigid_aligned, when present, is GT-assisted shape-only SE(3) diagnosis.",
            "Original ReconViaGen consumes the same RGBA files but not cached camera extrinsics.",
            "",
        ]
    )
    return "\n".join(lines)


def run(args: argparse.Namespace) -> None:
    thresholds = parse_csv(args.thresholds, float)
    alignment_modes = parse_csv(args.alignment_modes, str)
    allowed_modes = {"canonical_pose", "rigid_aligned"}
    if not set(alignment_modes).issubset(allowed_modes):
        raise ValueError(
            f"unsupported alignment modes={alignment_modes}; allowed={sorted(allowed_modes)}"
        )
    if alignment_modes[0] != "canonical_pose":
        raise ValueError("canonical_pose must be the first and primary alignment mode")
    if min(
        int(args.surface_samples),
        int(args.bootstrap_samples),
        int(args.alignment_samples),
        int(args.candidate_samples),
    ) <= 0:
        raise ValueError("sample counts must be positive")

    direct_report_path = args.direct_report.resolve()
    direct_report = load_json(direct_report_path)
    if direct_report.get("format") not in DIRECT_REPORT_FORMATS:
        raise RuntimeError(f"unsupported Direct report: {direct_report_path}")
    direct_run_config_path = direct_report_path.parent / "run_config.json"
    direct_run_config = load_json(direct_run_config_path)
    if direct_run_config.get("format") != "pose_point_depth_mv.direct_slat_mesh_run.v3":
        raise RuntimeError(f"unsupported Direct run config: {direct_run_config_path}")
    direct_manifest_path = Path(direct_run_config["cache_manifest"]).resolve()
    if sha256_file(direct_manifest_path) != str(
        direct_run_config["cache_manifest_sha256"]
    ):
        raise RuntimeError("Direct cache manifest changed after Mesh export")
    direct_manifest = load_json(direct_manifest_path)
    source_manifest_path = Path(
        direct_manifest["source_lifting_manifest"]
    ).resolve()
    if sha256_file(source_manifest_path) != str(
        direct_manifest["source_lifting_manifest_sha256"]
    ):
        raise RuntimeError("source lifting manifest changed after Direct cache build")
    source_manifest = load_json(source_manifest_path)
    direct_samples = indexed_samples(direct_manifest, seed_key="support_seed")
    source_samples = indexed_samples(source_manifest, seed_key=None)

    direct_records = list(direct_report.get("records", []))
    if not direct_records:
        raise RuntimeError("Direct report contains no Mesh records")
    required_branches = {"stock", "full"}
    if not required_branches.issubset(
        set(direct_report.get("comparison_branches", ()))
    ):
        raise RuntimeError("Direct report does not contain stock and full branches")
    expected_keys = {
        (str(row["uid"]), int(row["joint_seed"])) for row in direct_records
    }
    if len(expected_keys) != len(direct_records):
        raise RuntimeError("Direct report contains duplicate uid/joint-seed records")
    recon_index = resolve_recon_reports(args)
    missing = sorted(expected_keys - set(recon_index))
    if missing:
        raise RuntimeError(
            "missing original ReconViaGen reports for Direct uid/seeds: "
            f"{missing[:10]}{' ...' if len(missing) > 10 else ''}"
        )

    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(
            f"immutable output exists; use a new directory: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    axis_audit = affine_audit(LATENT_DECODER_TO_REFERENCE)
    records_by_mode: dict[str, list[dict[str, Any]]] = {
        mode: [] for mode in alignment_modes
    }
    input_audits: list[dict[str, Any]] = []

    for record_position, direct_record in enumerate(direct_records):
        uid = str(direct_record["uid"])
        object_uid = str(direct_record["object_uid"])
        seed = int(direct_record["joint_seed"])
        pair_id = str(direct_record["pair_id"])
        direct_row = direct_samples.get((uid, seed))
        source_row = source_samples.get(uid)
        if direct_row is None or source_row is None:
            raise RuntimeError(f"input manifest row missing for uid/seed={(uid, seed)}")
        recon_report_path = recon_index[(uid, seed)]
        recon_report = load_json(recon_report_path)
        if (
            recon_report.get("format") != RECON_REPORT_FORMAT
            or recon_report.get("complete") is not True
        ):
            raise RuntimeError(f"invalid original ReconViaGen report: {recon_report_path}")
        input_audit = validate_same_input(
            uid=uid,
            object_uid=object_uid,
            seed=seed,
            direct_manifest=direct_manifest,
            direct_row=direct_row,
            source_manifest_path=source_manifest_path,
            source_row=source_row,
            recon_report_path=recon_report_path,
            recon_report=recon_report,
        )
        input_audits.append(input_audit)

        target, target_audit = load_target(dict(direct_record["target"]))
        pair_root = direct_report_path.parent / "mesh_pairs" / pair_id
        method_paths = {
            "reconviagen_original": validate_binding(
                recon_report["mesh_canonical"],
                label=f"{pair_id}.reconviagen_original",
            ),
            "direct_stock": (pair_root / "stock" / "mesh_canonical.obj").resolve(),
            "direct_full": (pair_root / "full" / "mesh_canonical.obj").resolve(),
        }
        for method_id, path in method_paths.items():
            if not path.is_file():
                raise FileNotFoundError(f"{pair_id}.{method_id}: {path}")
        canonical_meshes = {
            method_id: mesh_in_reference_frame(path)
            for method_id, path in method_paths.items()
        }

        for mode_position, mode_id in enumerate(alignment_modes):
            method_rows: dict[str, Any] = {}
            for method_position, method_id in enumerate(METHOD_IDS):
                mesh = canonical_meshes[method_id]
                if mode_id == "canonical_pose":
                    evaluated = mesh
                    alignment = {
                        **affine_audit(np.eye(4, dtype=np.float64)),
                        "gt_assisted": False,
                        "scale_applied": False,
                        "policy": (
                            "fixed vendored decoder-to-reference axis conversion; "
                            "no per-object GT fitting"
                        ),
                    }
                else:
                    evaluated, alignment = rigid_icp(
                        mesh,
                        target,
                        seed=(
                            int(args.alignment_seed)
                            + record_position * 1009
                            + method_position * 53
                        ),
                        candidate_samples=int(args.candidate_samples),
                        final_samples=int(args.alignment_samples),
                        candidate_iterations=int(args.candidate_iterations),
                        final_iterations=int(args.final_iterations),
                    )
                    alignment = {
                        **alignment,
                        "gt_assisted": True,
                        "policy": (
                            "proper SE(3) ICP shape-only diagnostic; scale and "
                            "reflection forbidden"
                        ),
                    }
                metric_seed = (
                    int(args.metric_seed)
                    + seed * 1009
                    + record_position * 9173
                    + mode_position * 100003
                )
                surface = extended_surface_metrics(
                    evaluated,
                    target,
                    count=int(args.surface_samples),
                    seed=metric_seed,
                    thresholds=thresholds,
                )
                method_rows[method_id] = {
                    "mesh": binding(method_paths[method_id]),
                    "source_frame": "decoder transform_pose=False",
                    "source_to_reference": axis_audit,
                    "alignment": alignment,
                    "surface": surface,
                    "structure": mesh_structure_metrics(evaluated),
                }

            comparison_rows: dict[str, Any] = {}
            comparison_metric_names = [
                *method_rows["direct_full"]["surface"].keys(),
                "largest_component_ratio",
            ]
            for comparison_id, lhs, rhs in COMPARISONS:
                lhs_values = {
                    **method_rows[lhs]["surface"],
                    "largest_component_ratio": method_rows[lhs]["structure"][
                        "largest_component_ratio"
                    ],
                }
                rhs_values = {
                    **method_rows[rhs]["surface"],
                    "largest_component_ratio": method_rows[rhs]["structure"][
                        "largest_component_ratio"
                    ],
                }
                comparison_rows[comparison_id] = {
                    "lhs": lhs,
                    "rhs": rhs,
                    "positive_means_lhs_better": True,
                    "metrics": {
                        key: metric_delta(
                            float(lhs_values[key]),
                            float(rhs_values[key]),
                            key,
                        )
                        for key in comparison_metric_names
                    },
                }
            records_by_mode[mode_id].append(
                {
                    "pair_id": pair_id,
                    "uid": uid,
                    "object_uid": object_uid,
                    "joint_seed": seed,
                    "view_count": input_audit["view_count"],
                    "target": target_audit,
                    "methods": method_rows,
                    "comparisons": comparison_rows,
                }
            )

    surface_keys = list(
        records_by_mode["canonical_pose"][0]["methods"]["direct_full"][
            "surface"
        ]
    )
    modes: dict[str, Any] = {}
    for mode_id in alignment_modes:
        object_rows, method_summary, comparison_summary = aggregate_results(
            records_by_mode[mode_id],
            surface_keys=surface_keys,
            bootstrap_samples=int(args.bootstrap_samples),
        )
        modes[mode_id] = {
            "primary": mode_id == "canonical_pose",
            "gt_assisted": mode_id != "canonical_pose",
            "description": (
                "end-to-end canonical/source frame; fixed axis conversion only"
                if mode_id == "canonical_pose"
                else (
                    "shape-only proper-SE(3) ICP; no scale/reflection; not an "
                    "AR/world-pose result"
                )
            ),
            "records": records_by_mode[mode_id],
            "object_rows": object_rows,
            "method_summary": method_summary,
            "comparison_summary": comparison_summary,
        }

    run_config = {
        "direct_report": binding(direct_report_path),
        "direct_run_config": binding(direct_run_config_path),
        "direct_cache_manifest": binding(direct_manifest_path),
        "source_lifting_manifest": binding(source_manifest_path),
        "reconviagen_reports": [
            binding(recon_index[key]) for key in sorted(expected_keys)
        ],
        "surface_samples": int(args.surface_samples),
        "thresholds": thresholds,
        "bootstrap_samples": int(args.bootstrap_samples),
        "metric_seed": int(args.metric_seed),
        "alignment_modes": alignment_modes,
        "rigid_alignment": {
            "alignment_seed": int(args.alignment_seed),
            "candidate_samples": int(args.candidate_samples),
            "alignment_samples": int(args.alignment_samples),
            "candidate_iterations": int(args.candidate_iterations),
            "final_iterations": int(args.final_iterations),
        },
        "decoder_to_reference": axis_audit,
        "config_sha256": "",
    }
    run_config["config_sha256"] = canonical_sha256(
        {key: value for key, value in run_config.items() if key != "config_sha256"}
    )
    report = {
        "format": FORMAT,
        "complete": True,
        "formal": False,
        "purpose": (
            "matched-input exploratory Mesh accuracy comparison of original "
            "ReconViaGen, Direct corrected-SS/native-SLAT stock and Direct Full"
        ),
        "object_count": len(
            {str(row["object_uid"]) for row in direct_records}
        ),
        "record_count": len(direct_records),
        "joint_seeds": sorted(
            {int(row["joint_seed"]) for row in direct_records}
        ),
        "same_input_audit": {
            "passed": len(input_audits) == len(direct_records),
            "records": input_audits,
            "scope": (
                "all methods bind the same RGB/mask files and view IDs; original "
                "ReconViaGen does not consume the cached camera extrinsics"
            ),
        },
        "run_config": run_config,
        "modes": modes,
        "scope_guard": (
            "exploratory matched-input comparison; canonical_pose is the only "
            "end-to-end AR/world-frame result. Cross-architecture same latent "
            "noise is neither possible nor claimed."
        ),
        "code_bindings": code_bindings(
            {
                "evaluator": Path(__file__).resolve(),
                "direct_exporter": (
                    Path(__file__).resolve().parent
                    / "export_direct_slat_mesh_pairs.py"
                ),
                "reconviagen_runner": (
                    Path(__file__).resolve().parent
                    / "infer_reconviagen_stock_from_direct_cache.py"
                ),
                "axis_contract": (
                    Path(__file__).resolve().parent
                    / "render_direct_slat_fourway.py"
                ),
            }
        ),
    }
    report["report_sha256"] = canonical_sha256(report)
    atomic_json(output_dir / "report.json", report)
    summary = make_summary_text(report)
    atomic_text(output_dir / "summary.txt", summary + "\n")
    print(summary)
    print(f"report: {output_dir / 'report.json'}")


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direct_report", type=Path, required=True)
    parser.add_argument(
        "--recon_report",
        action="append",
        default=[],
        help="Repeat once per original ReconViaGen uid/seed report.",
    )
    parser.add_argument(
        "--recon_root",
        action="append",
        default=[],
        help="Recursively discover original ReconViaGen report.json files.",
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--surface_samples", type=int, default=100000)
    parser.add_argument("--thresholds", default="0.01,0.02,0.05")
    parser.add_argument("--bootstrap_samples", type=int, default=10000)
    parser.add_argument("--metric_seed", type=int, default=20260730)
    parser.add_argument(
        "--alignment_modes",
        default="canonical_pose",
        help="canonical_pose, optionally followed by rigid_aligned.",
    )
    parser.add_argument("--alignment_seed", type=int, default=20260730)
    parser.add_argument("--candidate_samples", type=int, default=1000)
    parser.add_argument("--alignment_samples", type=int, default=4000)
    parser.add_argument("--candidate_iterations", type=int, default=8)
    parser.add_argument("--final_iterations", type=int, default=30)
    return parser


def main() -> None:
    run(make_parser().parse_args())


if __name__ == "__main__":
    main()
