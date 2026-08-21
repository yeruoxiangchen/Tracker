#!/usr/bin/env python3
"""Strict two-endpoint pure-shape evaluation on consumed Omni Holdout64.

The compared endpoints are:

* ProObjaverse official no-VGGT Native-SS step30000 + Native-SLat step30000;
* the released ReconViaGen VGGT -> Stock SS -> Stock SLat pipeline.

Both inference manifests must bind the exact frozen pose+mask runtime-O input.
Point clouds are forbidden by the runtime contract.  Geometry scoring first
centres every target/prediction at its own AABB midpoint and divides it by its
own longest AABB extent.  Each normalized prediction is then aligned to the
normalized target with a proper isotropic Sim(3).  Reflection and anisotropic
scale are forbidden.  Consequently the report is shape-only and cannot support
world-pose, metric-scale, or AR-placement claims.

The evaluator is split into prepare/worker/merge commands so the large Omni GT
meshes can be processed in parallel and resumed without weakening file/hash
bindings.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.spatial import cKDTree

from pose_point_depth_mv.compare_pixal3d_singleview_smoke import (
    load_mesh,
    similarity_icp,
)
from pose_point_depth_mv.dataset_tools.prepare_omni_real_label_cache import (
    MANIFEST_FORMAT as LABEL_MANIFEST_FORMAT,
)
from pose_point_depth_mv.dataset_tools.prepare_omni_real_runtime_inputs import (
    MANIFEST_FORMAT as RUNTIME_MANIFEST_FORMAT,
)
from pose_point_depth_mv.evaluate_holdout64_current_vs_reconviagen_shape import (
    RECONVIAGEN_DECODER_TO_REFERENCE,
    normalize_mesh_bbox,
)
from pose_point_depth_mv.evaluate_stock2_full2_reconviagen_sim3_shape import (
    audit_proper_sim3,
)
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
    "pose_point_depth_mv.omni_pose_mask_ss30k_slat30k_vs_recon_shape.v1"
)
CONFIG_FORMAT = REPORT_FORMAT + ".config"
RECORD_FORMAT = REPORT_FORMAT + ".record"
WORKER_FORMAT = REPORT_FORMAT + ".worker"
POSE_MASK_FRONTEND_FORMAT = "pose_point_depth_mv.pose_mask_input_frontend.v1"
DINO_INPUT_MANIFEST_FORMAT = (
    "pose_point_depth_mv.omni_real_dino_only_model_input_manifest.v1"
)
CURRENT_MANIFEST_FORMAT = (
    "pose_point_depth_mv.real_proobjaverse_official_ss_slat_inference_manifest.v1"
)
RECON_MANIFEST_FORMAT = (
    "pose_point_depth_mv.omni_real_reconviagen_inference_manifest.v1"
)
CURRENT_METHOD = "proobjaverse_official_native_ss_trained_slat"
RECON_METHOD = "reconviagen_original"
METHODS = ("ss30k_slat30k", RECON_METHOD)
METRICS = (
    "pred_to_gt_mean",
    "gt_to_pred_mean",
    "chamfer_l1",
    "chamfer_l2",
    "fscore_0p01",
    "fscore_0p02",
    "fscore_0p05",
    "normal_consistency",
)
LOWER_IS_BETTER = {
    "pred_to_gt_mean",
    "gt_to_pred_mean",
    "chamfer_l1",
    "chamfer_l2",
}
EXPECTED_SS_SAMPLING = {
    "checkpoint_step": 30000,
    "weights": "ema",
    "cfg_strength": 5.0,
    "steps": 25,
    "cfg_interval": [0.5, 1.0],
    "guidance_rescale": 0.0,
    "rescale_t": 3.0,
    "amp_dtype": "bf16",
    "false_checks": [],
}
RUNTIME_MANIFEST_FORMATS = {
    "pose_point_depth_mv.omni_real_runtime_input_manifest.v2",
    RUNTIME_MANIFEST_FORMAT,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def file_binding(path: str | Path, *, expected_sha256: str = "") -> dict[str, str]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    digest = sha256_file(resolved)
    if expected_sha256 and digest != str(expected_sha256):
        raise RuntimeError(f"file SHA256 changed: {resolved}")
    return {"path": str(resolved), "sha256": digest}


def frozen_mesh_binding(
    row: dict[str, Any], path_key: str, hash_key: str, *, label: str
) -> dict[str, str]:
    path = Path(str(row.get(path_key, ""))).expanduser().resolve()
    digest = str(row.get(hash_key, ""))
    if not path.is_file() or not digest:
        raise RuntimeError(f"{label} lacks a frozen mesh path/hash")
    return {"path": str(path), "sha256": digest}


def validate_config(config: dict[str, Any]) -> None:
    saved = str(config.get("config_sha256", ""))
    body = dict(config)
    body.pop("config_sha256", None)
    if (
        config.get("format") != CONFIG_FORMAT
        or not saved
        or canonical_sha256(body) != saved
    ):
        raise RuntimeError("evaluation config identity differs")


def _validate_pose_mask_runtime(
    path: Path, *, expected_objects: int
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], str]:
    payload = load_json(path)
    rows = index_objects(payload.get("objects", []), label="pose-mask runtime-O")
    build = dict(payload.get("build_config") or {})
    if (
        payload.get("format") not in RUNTIME_MANIFEST_FORMATS
        or payload.get("passed") is not True
        or int(payload.get("selected_object_count", -1)) != int(expected_objects)
        or int(payload.get("completed_object_count", -1)) != int(expected_objects)
        or len(rows) != int(expected_objects)
        or build.get("input_frontend_format") != POSE_MASK_FRONTEND_FORMAT
        or build.get("point_cloud_consumed") is not False
        or build.get("gt_consumed") is not False
        or build.get("old_mesh_consumed") is not False
        or build.get("metric_or_ranking_consumed") is not False
        or build.get("view_selection")
        != "exactly_reuse_reference_runtime_selected_views"
    ):
        raise RuntimeError("frozen pose+mask runtime-O contract differs")
    for key, row in rows.items():
        visible = dict(row.get("external_visible_equivalence") or {})
        stats = dict(row.get("runtime_frame_stats") or {})
        if (
            row.get("passed") is not True
            or row.get("input_frontend_format") != POSE_MASK_FRONTEND_FORMAT
            or row.get("point_cloud_consumed") is not False
            or row.get("gt_consumed") is not False
            or row.get("old_mesh_consumed") is not False
            or row.get("metric_or_ranking_consumed") is not False
            or stats.get("point_cloud_consumed") is not False
            or int(row.get("selected_view_count", -1)) != 8
            or not visible
            or not all(value is True for value in visible.values())
        ):
            raise RuntimeError(f"invalid pose+mask runtime row: {key}")
    return payload, rows, sha256_file(path)


def _validate_model_input(
    path: Path,
    *,
    runtime_sha256: str,
    expected_objects: int,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], str]:
    payload = load_json(path)
    rows = index_objects(payload.get("objects", []), label="pose-mask DINO inputs")
    runtime = validate_bound_file(
        payload.get("runtime_input_manifest", ""),
        str(payload.get("runtime_input_manifest_sha256", "")),
        label="pose-mask DINO runtime",
    )
    if (
        payload.get("format") != DINO_INPUT_MANIFEST_FORMAT
        or payload.get("passed") is not True
        or int(payload.get("selected_object_count", -1)) != int(expected_objects)
        or int(payload.get("completed_object_count", -1)) != int(expected_objects)
        or len(rows) != int(expected_objects)
        or payload.get("vggt_model_loaded") is not False
        or payload.get("vggt_model_executed") is not False
        or str(payload.get("runtime_input_manifest_sha256", ""))
        != str(runtime_sha256)
    ):
        raise RuntimeError("pose+mask DINO-only input contract differs")
    if sha256_file(runtime) != str(runtime_sha256):
        raise RuntimeError("pose+mask DINO runtime binding changed")
    for key, row in rows.items():
        if (
            row.get("passed") is not True
            or row.get("target_or_mesh_consumed") is not False
        ):
            raise RuntimeError(f"invalid DINO-only model input row: {key}")
    return payload, rows, sha256_file(path)


def _validate_current_manifest(
    path: Path,
    *,
    runtime_sha256: str,
    model_input_sha256: str,
    expected_objects: int,
    seed: int,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    payload = load_json(path)
    expected = {
        "format": CURRENT_MANIFEST_FORMAT,
        "method": CURRENT_METHOD,
        "seeds": [int(seed)],
        "object_count": int(expected_objects),
        "record_count": int(expected_objects),
        "output_frame": "runtime-O",
        "vggt_model_loaded": False,
        "vggt_model_executed": False,
        "target_or_metric_consumed": False,
        "passed": True,
    }
    mismatch = {
        key: (payload.get(key), value)
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if mismatch:
        raise RuntimeError(f"SS30K+SLat30K manifest contract differs: {mismatch}")
    if (
        str(payload.get("runtime_input_manifest_sha256", ""))
        != str(runtime_sha256)
        or str(payload.get("model_input_manifest_sha256", ""))
        != str(model_input_sha256)
    ):
        raise RuntimeError("SS30K+SLat30K input binding differs")
    validate_bound_file(
        payload.get("runtime_input_manifest", ""),
        str(runtime_sha256),
        label="SS30K+SLat30K runtime",
    )
    validate_bound_file(
        payload.get("model_input_manifest", ""),
        str(model_input_sha256),
        label="SS30K+SLat30K DINO input",
    )
    ss = dict(payload.get("native_ss_deployment") or {})
    ss_mismatch = {
        key: (ss.get(key), value)
        for key, value in EXPECTED_SS_SAMPLING.items()
        if ss.get(key) != value
    }
    if ss_mismatch:
        raise RuntimeError(f"Native-SS step30000 deployment differs: {ss_mismatch}")
    file_binding(ss.get("report", ""), expected_sha256=str(ss.get("report_sha256", "")))
    file_binding(
        ss.get("checkpoint", ""),
        expected_sha256=str(ss.get("checkpoint_sha256", "")),
    )
    slat = dict(payload.get("native_slat_deployment") or {})
    if (
        int(slat.get("checkpoint_step", -1)) != 30000
        or slat.get("weights") != "ema"
    ):
        raise RuntimeError("Native-SLat step30000 deployment differs")
    file_binding(
        slat.get("checkpoint", ""),
        expected_sha256=str(slat.get("checkpoint_sha256", "")),
    )
    bridge = dict(payload.get("cross_deployment_bridge") or {})
    if (
        bridge.get("passed") is not True
        or bridge.get("runtime_integrity_passed") is not True
        or bridge.get("native_ss_science_passed") is not True
        or bridge.get("route")
        != "posed_dino_official_native_ss_step30000_native_slat_step30000"
    ):
        raise RuntimeError("SS30K/SLat30K held-out deployment evidence differs")
    file_binding(
        bridge.get("path", ""), expected_sha256=str(bridge.get("sha256", ""))
    )
    file_binding(
        payload.get("stock_slat_freeze", ""),
        expected_sha256=str(payload.get("stock_slat_freeze_sha256", "")),
    )
    rows: dict[str, dict[str, Any]] = {}
    for row in payload.get("objects", []):
        key = str(row.get("object_key", ""))
        if (
            not key
            or key in rows
            or row.get("method") != CURRENT_METHOD
            or int(row.get("seed", -1)) != int(seed)
            or int(row.get("native_slat_checkpoint_step", -1)) != 30000
            or row.get("native_slat_weights") != "ema"
            or row.get("output_frame") != "runtime-O"
            or row.get("target_or_metric_consumed") is not False
            or row.get("vggt_model_loaded") is not False
            or row.get("vggt_model_executed") is not False
            or row.get("passed") is not True
        ):
            raise RuntimeError(f"invalid SS30K+SLat30K inference row: {key!r}")
        rows[key] = row
    if len(rows) != int(expected_objects):
        raise RuntimeError("SS30K+SLat30K object coverage differs")
    return payload, rows


def _validate_recon_manifest(
    path: Path,
    *,
    runtime_sha256: str,
    expected_objects: int,
    seed: int,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    payload = load_json(path)
    expected = {
        "format": RECON_MANIFEST_FORMAT,
        "method": RECON_METHOD,
        "seeds": [int(seed)],
        "object_count": int(expected_objects),
        "record_count": int(expected_objects),
        "target_or_metric_consumed": False,
        "passed": True,
    }
    mismatch = {
        key: (payload.get(key), value)
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if mismatch:
        raise RuntimeError(f"ReconViaGen manifest contract differs: {mismatch}")
    if str(payload.get("runtime_input_manifest_sha256", "")) != str(runtime_sha256):
        raise RuntimeError("ReconViaGen does not bind the pose+mask runtime input")
    validate_bound_file(
        payload.get("runtime_input_manifest", ""),
        str(runtime_sha256),
        label="ReconViaGen pose-mask runtime",
    )
    rows: dict[str, dict[str, Any]] = {}
    for row in payload.get("objects", []):
        key = str(row.get("object_key", ""))
        if (
            not key
            or key in rows
            or row.get("method") != RECON_METHOD
            or int(row.get("seed", -1)) != int(seed)
            or row.get("explicit_runtime_pose_condition") is not False
            or row.get("target_or_metric_consumed") is not False
            or row.get("passed") is not True
        ):
            raise RuntimeError(f"invalid ReconViaGen inference row: {key!r}")
        rows[key] = row
    if len(rows) != int(expected_objects):
        raise RuntimeError("ReconViaGen object coverage differs")
    return payload, rows


def _validate_labels(
    path: Path, *, expected_objects: int
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    payload = load_json(path)
    rows = index_objects(payload.get("objects", []), label="Omni GT labels")
    if (
        payload.get("format") != LABEL_MANIFEST_FORMAT
        or payload.get("passed") is not True
        or int(payload.get("selected_object_count", -1)) != int(expected_objects)
        or int(payload.get("completed_object_count", -1)) != int(expected_objects)
        or len(rows) != int(expected_objects)
    ):
        raise RuntimeError("Omni GT label manifest contract differs")
    validate_bound_file(
        payload.get("runtime_input_manifest", ""),
        str(payload.get("runtime_input_manifest_sha256", "")),
        label="Omni GT reference runtime",
    )
    for key, row in rows.items():
        if (
            row.get("passed") is not True
            or row.get("condition_identity_unchanged_by_gt_binding") is not True
            or row.get("gt_fields_exported_to_model_condition")
        ):
            raise RuntimeError(f"invalid Omni GT label row: {key}")
    return payload, rows


def surface_metrics_from_meshes(
    predicted: Any,
    target: Any,
    *,
    count: int,
    seed: int,
    thresholds: Iterable[float],
) -> dict[str, float]:
    pred_points, pred_normals = deterministic_surface_sample(predicted, count, seed)
    target_points, target_normals = deterministic_surface_sample(target, count, seed)
    target_tree = cKDTree(target_points)
    pred_tree = cKDTree(pred_points)
    pred_distance, pred_index = target_tree.query(pred_points, k=1, workers=1)
    target_distance, target_index = pred_tree.query(target_points, k=1, workers=1)
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
                        np.sum(pred_normals * target_normals[pred_index], axis=1)
                    )
                )
                + np.mean(
                    np.abs(
                        np.sum(target_normals * pred_normals[target_index], axis=1)
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


def numeric_summary(
    values: Iterable[float], *, bootstrap_samples: int, seed: int
) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=np.float64)
    if not len(array) or not np.isfinite(array).all():
        raise ValueError("summary requires finite values")
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, len(array), size=(int(bootstrap_samples), len(array)))
    means = array[indices].mean(axis=1)
    return {
        "count": int(len(array)),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "min": float(array.min()),
        "max": float(array.max()),
        "positive_rate": float(np.mean(array > 0.0)),
        "nonnegative_rate": float(np.mean(array >= 0.0)),
        "bootstrap_mean_95_ci": [
            float(np.quantile(means, 0.025)),
            float(np.quantile(means, 0.975)),
        ],
    }


def summarize_population(
    records: list[dict[str, Any]], *, bootstrap_samples: int, seed: int
) -> dict[str, Any]:
    methods = {
        method: {
            metric: numeric_summary(
                [float(row["methods"][method]["metrics"][metric]) for row in records],
                bootstrap_samples=int(bootstrap_samples),
                seed=int(seed) + method_position * 1000 + metric_position,
            )
            for metric_position, metric in enumerate(METRICS)
        }
        for method_position, method in enumerate(METHODS)
    }
    paired: dict[str, Any] = {}
    for metric_position, metric in enumerate(METRICS):
        values = []
        per_object = {}
        for row in records:
            candidate = float(row["methods"][METHODS[0]]["metrics"][metric])
            baseline = float(row["methods"][METHODS[1]]["metrics"][metric])
            delta = (
                baseline - candidate
                if metric in LOWER_IS_BETTER
                else candidate - baseline
            )
            values.append(delta)
            per_object[str(row["object_key"])] = float(delta)
        summary = numeric_summary(
            values,
            bootstrap_samples=int(bootstrap_samples),
            seed=int(seed) + 10000 + metric_position,
        )
        summary.update(
            {
                "candidate_win_count": int(sum(value > 0.0 for value in values)),
                "tie_count": int(sum(value == 0.0 for value in values)),
                "baseline_win_count": int(sum(value < 0.0 for value in values)),
                "per_object": per_object,
            }
        )
        paired[metric] = summary
    return {
        "object_count": len(records),
        "methods": methods,
        "ss30k_slat30k_vs_reconviagen": {
            "candidate": METHODS[0],
            "baseline": METHODS[1],
            "positive_definition": "positive means SS30K+SLat30K is better",
            "metrics": paired,
        },
    }


def summary_text(report: dict[str, Any]) -> str:
    population = report["summary"]["all64"]
    current = population["methods"][METHODS[0]]
    recon = population["methods"][METHODS[1]]
    paired = population["ss30k_slat30k_vs_reconviagen"]["metrics"]
    lines = [
        "Omni Holdout64 pose-mask: SS30K+SLat30K vs strict ReconViaGen",
        "================================================================",
        f"passed: {str(bool(report['passed'])).lower()}",
        "formal: false (consumed Holdout64; retrospective external-domain test)",
        f"objects: {report['object_count']}  seed: {report['seed']}",
        "runtime-O: pose+mask only; point cloud forbidden",
        "shape protocol: own-AABB center/longest-edge normalization, then proper isotropic Sim(3)",
        "reflection/anisotropic scale: forbidden",
        "",
        "[absolute object means]",
        f"SS30K+SLat30K: chamfer_l1={current['chamfer_l1']['mean']:.8f} "
        f"f@0.02={current['fscore_0p02']['mean']:.8f} "
        f"normal={current['normal_consistency']['mean']:.8f}",
        f"ReconViaGen:     chamfer_l1={recon['chamfer_l1']['mean']:.8f} "
        f"f@0.02={recon['fscore_0p02']['mean']:.8f} "
        f"normal={recon['normal_consistency']['mean']:.8f}",
        "",
        "[paired improvement; positive = SS30K+SLat30K better]",
    ]
    for metric in ("chamfer_l1", "fscore_0p02", "normal_consistency"):
        row = paired[metric]
        lines.append(
            f"{metric}: mean={row['mean']:+.8f} median={row['median']:+.8f} "
            f"win={row['positive_rate']:.4f} CI95={row['bootstrap_mean_95_ci']}"
        )
    if "reliable_labels" in report["summary"]:
        reliable = report["summary"]["reliable_labels"]
        lines.extend(
            (
                "",
                f"reliable-label sensitivity population: {reliable['object_count']}",
            )
        )
    lines.extend(
        (
            "",
            "Interpretation guard: this report compares only normalized geometry shape.",
            "It cannot support world-pose, metric-scale, runtime-O placement, or formal",
            "untouched-test claims. passed=true means protocol/artifact completeness.",
        )
    )
    return "\n".join(lines) + "\n"


def cmd_prepare(args: argparse.Namespace) -> None:
    positive = (
        args.expected_objects,
        args.worker_count,
        args.candidate_samples,
        args.alignment_samples,
        args.candidate_iterations,
        args.final_iterations,
        args.surface_samples,
        args.bootstrap_samples,
    )
    if any(int(value) <= 0 for value in positive):
        raise ValueError("all counts and iteration arguments must be positive")
    runtime_path = Path(args.pose_mask_runtime_manifest).expanduser().resolve()
    model_input_path = Path(args.pose_mask_model_input_manifest).expanduser().resolve()
    label_path = Path(args.label_manifest).expanduser().resolve()
    current_path = Path(args.current_manifest).expanduser().resolve()
    recon_path = Path(args.reconviagen_manifest).expanduser().resolve()

    runtime, runtime_rows, runtime_sha = _validate_pose_mask_runtime(
        runtime_path, expected_objects=int(args.expected_objects)
    )
    _, model_rows, model_sha = _validate_model_input(
        model_input_path,
        runtime_sha256=runtime_sha,
        expected_objects=int(args.expected_objects),
    )
    current, current_rows = _validate_current_manifest(
        current_path,
        runtime_sha256=runtime_sha,
        model_input_sha256=model_sha,
        expected_objects=int(args.expected_objects),
        seed=int(args.seed),
    )
    _, recon_rows = _validate_recon_manifest(
        recon_path,
        runtime_sha256=runtime_sha,
        expected_objects=int(args.expected_objects),
        seed=int(args.seed),
    )
    labels, label_rows = _validate_labels(
        label_path, expected_objects=int(args.expected_objects)
    )
    object_sets = tuple(
        set(rows) for rows in (runtime_rows, model_rows, current_rows, recon_rows, label_rows)
    )
    if any(values != object_sets[0] for values in object_sets[1:]):
        raise RuntimeError("pose-mask/current/ReconViaGen/GT object sets differ")

    cases = []
    for key in sorted(label_rows):
        label = label_rows[key]
        pose_row = runtime_rows[key]
        reference_path = validate_bound_file(
            pose_row.get("reference_runtime_report", ""),
            str(pose_row.get("reference_runtime_report_sha256", "")),
            label=f"pose-mask reference runtime {key}",
        )
        if str(Path(reference_path).parent.name) != str(label["object_id"]):
            raise RuntimeError(f"pose-mask/GT reference object differs: {key}")
        cases.append(
            {
                "object_key": key,
                "category": str(label["category"]),
                "object_id": str(label["object_id"]),
                "alignment_quality_passed": bool(
                    label.get("alignment_quality_passed")
                ),
                "target": frozen_mesh_binding(
                    label, "mesh_o", "mesh_o_sha256", label=f"GT/{key}"
                ),
                "methods": {
                    METHODS[0]: frozen_mesh_binding(
                        current_rows[key],
                        "mesh",
                        "mesh_sha256",
                        label=f"SS30K+SLat30K/{key}",
                    ),
                    METHODS[1]: frozen_mesh_binding(
                        recon_rows[key],
                        "mesh",
                        "mesh_sha256",
                        label=f"ReconViaGen/{key}",
                    ),
                },
            }
        )

    config = {
        "format": CONFIG_FORMAT,
        "formal": False,
        "post_hoc": True,
        "holdout64_consumed": True,
        "protocol_scope": "consumed_omni_holdout64_pose_mask_pure_shape_only",
        "pose_mask_runtime_manifest": file_binding(runtime_path),
        "pose_mask_model_input_manifest": file_binding(model_input_path),
        "label_manifest": file_binding(label_path),
        "current_manifest": file_binding(current_path),
        "reconviagen_manifest": file_binding(recon_path),
        "target_reference_runtime_manifest": file_binding(
            labels["runtime_input_manifest"],
            expected_sha256=str(labels["runtime_input_manifest_sha256"]),
        ),
        "runtime_contract": {
            "input_frontend_format": POSE_MASK_FRONTEND_FORMAT,
            "point_cloud_consumed": False,
            "selected_views": 8,
            "same_pose_mask_runtime_binding_for_both_endpoints": True,
            "reconviagen_explicit_pose_condition": False,
            "target_mesh_role": "metrics only; never consumed by either endpoint",
        },
        "current_deployment": {
            "method": "official ProObjaverse no-VGGT SS30K + SLat30K",
            "native_ss": dict(current["native_ss_deployment"]),
            "native_slat": dict(current["native_slat_deployment"]),
            "cross_deployment_bridge": dict(current["cross_deployment_bridge"]),
        },
        "object_count": int(args.expected_objects),
        "seed": int(args.seed),
        "worker_count": int(args.worker_count),
        "methods": list(METHODS),
        "normalization": {
            "all_meshes_center": "own axis-aligned bounding-box midpoint",
            "all_meshes_scale": "divide by own longest AABB extent",
            "reconviagen_fixed_decoder_axis": "(x,y,z) -> (x,z,-y)",
            "reconviagen_fixed_axis_is_gt_fitted": False,
        },
        "shape_alignment": {
            "policy": "GT-assisted proper isotropic Sim(3) after per-mesh normalization",
            "proper_cube_initializations": 24,
            "reflection": False,
            "anisotropic_scale": False,
            "candidate_samples": int(args.candidate_samples),
            "alignment_samples": int(args.alignment_samples),
            "candidate_iterations": int(args.candidate_iterations),
            "final_iterations": int(args.final_iterations),
        },
        "evaluation": {
            "surface_samples": int(args.surface_samples),
            "thresholds_in_unit_longest_extent": [0.01, 0.02, 0.05],
            "bootstrap_samples": int(args.bootstrap_samples),
            "metric_seed": int(args.metric_seed),
            "paired_resampling_unit": "object",
        },
        "cases": cases,
        "implementation": file_binding(Path(__file__).resolve()),
    }
    config["config_sha256"] = canonical_sha256(config)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "run_config.json"
    if config_path.is_file():
        if load_json(config_path) != config:
            raise RuntimeError(f"existing evaluation config differs: {config_path}")
    elif any(output_dir.iterdir()):
        raise RuntimeError(f"unbound evaluation output is not empty: {output_dir}")
    else:
        atomic_json(config_path, config)
    print(
        json.dumps(
            {
                "passed": True,
                "objects": len(cases),
                "workers": int(args.worker_count),
                "pose_mask_runtime_sha256": runtime_sha,
                "config": str(config_path),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


def _record_identity(config: dict[str, Any], case: dict[str, Any]) -> dict[str, Any]:
    return {
        "format": RECORD_FORMAT,
        "config_sha256": str(config["config_sha256"]),
        "object_key": str(case["object_key"]),
        "target_mesh_sha256": str(case["target"]["sha256"]),
        "method_mesh_sha256": {
            method: str(case["methods"][method]["sha256"]) for method in METHODS
        },
    }


def cmd_worker(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir).expanduser().resolve()
    config = load_json(output_dir / "run_config.json")
    validate_config(config)
    worker_id = int(args.worker_id)
    worker_count = int(args.worker_count)
    if (
        worker_count != int(config["worker_count"])
        or not 0 <= worker_id < worker_count
    ):
        raise RuntimeError("worker identity differs from run config")
    cases = list(config["cases"])
    begin = len(cases) * worker_id // worker_count
    end = len(cases) * (worker_id + 1) // worker_count
    selected = cases[begin:end]
    worker_dir = output_dir / f"worker_{worker_id:02d}_of_{worker_count:02d}"
    worker_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for local_position, case in enumerate(selected):
        global_position = begin + local_position
        name_hash = hashlib.sha256(str(case["object_key"]).encode("utf-8")).hexdigest()[:16]
        record_path = worker_dir / "records" / f"{global_position:03d}_{name_hash}.json"
        identity = _record_identity(config, case)
        if record_path.is_file():
            if not args.resume:
                raise RuntimeError(f"record exists; pass --resume: {record_path}")
            record = load_json(record_path)
            mismatch = {
                key: (record.get(key), value)
                for key, value in identity.items()
                if record.get(key) != value
            }
            if mismatch or record.get("passed") is not True:
                raise RuntimeError(f"stale shape record={mismatch}: {record_path}")
            records.append(record)
            print(
                f"[omni_shape:reuse:w{worker_id}] {local_position + 1}/{len(selected)} "
                f"object={case['object_key']}",
                flush=True,
            )
            continue

        target_binding = file_binding(
            case["target"]["path"], expected_sha256=case["target"]["sha256"]
        )
        target_raw = load_mesh(target_binding["path"])
        target, target_normalization = normalize_mesh_bbox(target_raw)
        alignment_seed = int(config["evaluation"]["metric_seed"]) + global_position * 100003
        metric_seed = alignment_seed + 50_000_000
        method_rows = {}
        for method in METHODS:
            binding = file_binding(
                case["methods"][method]["path"],
                expected_sha256=case["methods"][method]["sha256"],
            )
            mesh = load_mesh(binding["path"])
            fixed_axis_applied = method == RECON_METHOD
            if fixed_axis_applied:
                mesh.apply_transform(RECONVIAGEN_DECODER_TO_REFERENCE)
            normalized, normalization = normalize_mesh_bbox(mesh)
            aligned, alignment = similarity_icp(
                normalized,
                target,
                seed=alignment_seed,
                candidate_samples=int(config["shape_alignment"]["candidate_samples"]),
                final_samples=int(config["shape_alignment"]["alignment_samples"]),
                candidate_iterations=int(
                    config["shape_alignment"]["candidate_iterations"]
                ),
                final_iterations=int(config["shape_alignment"]["final_iterations"]),
            )
            alignment_audit = audit_proper_sim3(alignment)
            metrics = surface_metrics_from_meshes(
                aligned,
                target,
                count=int(config["evaluation"]["surface_samples"]),
                seed=metric_seed,
                thresholds=config["evaluation"]["thresholds_in_unit_longest_extent"],
            )
            method_rows[method] = {
                "mesh": binding,
                "reconviagen_fixed_decoder_axis_applied": fixed_axis_applied,
                "normalization": normalization,
                "alignment": alignment_audit,
                "metrics": {metric: float(metrics[metric]) for metric in METRICS},
            }
            del mesh, normalized, aligned

        record = {
            **identity,
            "created_at_utc": utc_now(),
            "global_position": int(global_position),
            "category": str(case["category"]),
            "object_id": str(case["object_id"]),
            "alignment_quality_passed": bool(case["alignment_quality_passed"]),
            "target_mesh": target_binding,
            "target_normalization": target_normalization,
            "alignment_seed": int(alignment_seed),
            "metric_seed": int(metric_seed),
            "methods": method_rows,
            "passed": True,
        }
        atomic_json(record_path, record)
        records.append(record)
        print(
            f"[omni_shape:w{worker_id}] {local_position + 1}/{len(selected)} "
            f"object={case['object_key']}",
            flush=True,
        )
        del target_raw, target

    worker_report = {
        "format": WORKER_FORMAT,
        "created_at_utc": utc_now(),
        "passed": len(records) == len(selected),
        "config_sha256": str(config["config_sha256"]),
        "worker_id": worker_id,
        "worker_count": worker_count,
        "object_begin": begin,
        "object_end": end,
        "object_count": len(records),
        "object_keys": [str(row["object_key"]) for row in records],
    }
    atomic_json(worker_dir / "report.json", worker_report)
    print(json.dumps(worker_report, indent=2, ensure_ascii=False))
    if not worker_report["passed"]:
        raise SystemExit(2)


def cmd_merge(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir).expanduser().resolve()
    config_path = output_dir / "run_config.json"
    config = load_json(config_path)
    validate_config(config)
    report_path = output_dir / "report.json"
    if report_path.is_file():
        report = load_json(report_path)
        if (
            report.get("format") != REPORT_FORMAT
            or report.get("config_sha256") != config["config_sha256"]
            or report.get("passed") is not True
        ):
            raise RuntimeError("existing merged report contract differs")
        print(summary_text(report), end="")
        print(json.dumps({"passed": True, "reused": True, "report": str(report_path)}))
        return

    records = []
    seen = set()
    worker_count = int(config["worker_count"])
    cases = list(config["cases"])
    for worker_id in range(worker_count):
        begin = len(cases) * worker_id // worker_count
        end = len(cases) * (worker_id + 1) // worker_count
        worker_dir = output_dir / f"worker_{worker_id:02d}_of_{worker_count:02d}"
        worker_report = load_json(worker_dir / "report.json")
        if (
            worker_report.get("format") != WORKER_FORMAT
            or worker_report.get("passed") is not True
            or worker_report.get("config_sha256") != config["config_sha256"]
            or int(worker_report.get("worker_id", -1)) != worker_id
            or int(worker_report.get("worker_count", -1)) != worker_count
            or int(worker_report.get("object_begin", -1)) != begin
            or int(worker_report.get("object_end", -1)) != end
        ):
            raise RuntimeError(f"worker report contract differs: worker={worker_id}")
        record_paths = sorted((worker_dir / "records").glob("*.json"))
        if len(record_paths) != end - begin:
            raise RuntimeError(
                f"worker {worker_id} record count differs: {len(record_paths)} != {end-begin}"
            )
        for position, record_path in zip(range(begin, end), record_paths):
            record = load_json(record_path)
            identity = _record_identity(config, cases[position])
            if (
                any(record.get(key) != value for key, value in identity.items())
                or int(record.get("global_position", -1)) != position
                or record.get("passed") is not True
                or record["object_key"] in seen
            ):
                raise RuntimeError(f"invalid or duplicate record: {record_path}")
            seen.add(str(record["object_key"]))
            records.append(record)
    expected_keys = {str(case["object_key"]) for case in cases}
    if seen != expected_keys or len(records) != len(cases):
        raise RuntimeError("merged object coverage differs")
    records.sort(key=lambda row: int(row["global_position"]))

    bootstrap = int(config["evaluation"]["bootstrap_samples"])
    metric_seed = int(config["evaluation"]["metric_seed"])
    summary = {
        "all64": summarize_population(
            records, bootstrap_samples=bootstrap, seed=metric_seed
        )
    }
    reliable = [row for row in records if row["alignment_quality_passed"]]
    if len(reliable) != len(records):
        summary["reliable_labels"] = summarize_population(
            reliable, bootstrap_samples=bootstrap, seed=metric_seed + 100000
        )
    report = {
        "format": REPORT_FORMAT,
        "created_at_utc": utc_now(),
        "passed": True,
        "formal": False,
        "post_hoc": True,
        "holdout64_consumed": True,
        "protocol_scope": config["protocol_scope"],
        "interpretation_scope": (
            "normalized geometry shape only; no world pose, metric scale, "
            "runtime-O placement, or AR claim"
        ),
        "config": str(config_path),
        "config_sha256": str(config["config_sha256"]),
        "seed": int(config["seed"]),
        "object_count": len(records),
        "record_count": len(records) * len(METHODS),
        "methods": list(METHODS),
        "runtime_contract": config["runtime_contract"],
        "normalization": config["normalization"],
        "shape_alignment": config["shape_alignment"],
        "summary": summary,
        "records": records,
    }
    atomic_json(report_path, report)
    text = summary_text(report)
    atomic_text(output_dir / "summary.txt", text)
    print(text, end="")
    print(
        json.dumps(
            {
                "passed": True,
                "objects": len(records),
                "records": report["record_count"],
                "report": str(report_path),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--pose_mask_runtime_manifest", required=True)
    prepare.add_argument("--pose_mask_model_input_manifest", required=True)
    prepare.add_argument("--label_manifest", required=True)
    prepare.add_argument("--current_manifest", required=True)
    prepare.add_argument("--reconviagen_manifest", required=True)
    prepare.add_argument("--output_dir", required=True)
    prepare.add_argument("--expected_objects", type=int, default=64)
    prepare.add_argument("--seed", type=int, default=42)
    prepare.add_argument("--worker_count", type=int, default=4)
    prepare.add_argument("--candidate_samples", type=int, default=2000)
    prepare.add_argument("--alignment_samples", type=int, default=10000)
    prepare.add_argument("--candidate_iterations", type=int, default=12)
    prepare.add_argument("--final_iterations", type=int, default=50)
    prepare.add_argument("--surface_samples", type=int, default=20000)
    prepare.add_argument("--bootstrap_samples", type=int, default=10000)
    prepare.add_argument("--metric_seed", type=int, default=20260819)
    prepare.set_defaults(handler=cmd_prepare)

    worker = subparsers.add_parser("worker")
    worker.add_argument("--output_dir", required=True)
    worker.add_argument("--worker_id", required=True, type=int)
    worker.add_argument("--worker_count", default=4, type=int)
    worker.add_argument("--resume", action="store_true")
    worker.set_defaults(handler=cmd_worker)

    merge = subparsers.add_parser("merge")
    merge.add_argument("--output_dir", required=True)
    merge.set_defaults(handler=cmd_merge)
    return parser


def main() -> None:
    args = make_parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
