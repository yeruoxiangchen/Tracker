#!/usr/bin/env python3
"""Canonical Direct-SLAT / official-final Pixal3D utility evaluation."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from pose_point_depth_mv.compare_pixal3d_singleview_smoke import (
    OFFICIAL_GEOMETRY_EXPORT,
    OFFICIAL_POSTPROCESS,
    atomic_json,
    atomic_text,
    binding,
    canonical_sha256,
    load_mesh,
    pixal3d_mesh_path,
    pixal3d_result_path,
    sha256_file,
    surface_metrics,
    validate_official_inference_result,
    validate_protocol,
)


FORMAT = "pose_point_depth_mv.direct_slat_pixal3d_utility.v2"
TRANSFORM_FORMAT = "pose_point_depth_mv.pixal3d_canonical_transform.v2"
DIRECT_REPORT_FORMATS = frozenset(
    {
        "pose_point_depth_mv.direct_slat_mesh_exploratory.v1",
        "pose_point_depth_mv.direct_slat_mesh_exploratory.v2",
    }
)
METHODS = ("stock", "full", "pixal3d_native")
METRICS = (
    "chamfer_l1",
    "pred_to_gt_mean",
    "gt_to_pred_mean",
    "fscore_0p02",
    "normal_consistency",
    "precision_0p02",
    "recall_0p02",
)
OFFICIAL_EXPORT_ROTATION = np.asarray(
    [
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)
O_VOXEL_TO_GLB = np.asarray(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)
OFFICIAL_FINAL_EXPORT_FROM_DECODED = (
    OFFICIAL_EXPORT_ROTATION @ O_VOXEL_TO_GLB
)
LOWER_IS_BETTER = {
    "chamfer_l1",
    "pred_to_gt_mean",
    "gt_to_pred_mean",
}
CANONICAL_MARGIN_ATOL = 1.0e-6


def parse_unique_csv(value: str, cast) -> list[Any]:
    values = [cast(item.strip()) for item in str(value).split(",") if item.strip()]
    if not values or len(values) != len(set(values)):
        raise ValueError("CSV values must be non-empty and unique")
    return values


def fixed_float_value(
    values: Iterable[float],
    *,
    label: str,
    atol: float = CANONICAL_MARGIN_ATOL,
) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    if not array.size or not np.isfinite(array).all():
        raise ValueError(f"{label} requires finite non-empty values")
    reference = float(array[0])
    if not np.allclose(array, reference, rtol=0.0, atol=float(atol)):
        raise RuntimeError(f"{label} is not fixed: {array.tolist()}")
    return reference


def numeric_summary(
    values: Iterable[float],
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=np.float64)
    if not array.size or not np.isfinite(array).all():
        raise ValueError("numeric summary requires finite non-empty values")
    rng = np.random.default_rng(int(seed))
    bootstrap = np.empty(int(bootstrap_samples), dtype=np.float64)
    for index in range(int(bootstrap_samples)):
        bootstrap[index] = float(
            rng.choice(array, size=array.size, replace=True).mean()
        )
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "min": float(array.min()),
        "max": float(array.max()),
        "positive_rate": float(np.mean(array > 0.0)),
        "bootstrap_mean_95_ci": [
            float(np.quantile(bootstrap, 0.025)),
            float(np.quantile(bootstrap, 0.975)),
        ],
    }


def scan_mesh_geometry(path: Path) -> dict[str, Any]:
    mesh = load_mesh(path)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    minimum = np.min(vertices, axis=0)
    maximum = np.max(vertices, axis=0)
    vertex_count = int(len(mesh.vertices))
    face_count = int(len(mesh.faces))
    finite = bool(np.isfinite(vertices).all())
    success = bool(vertex_count > 0 and face_count > 0 and finite)
    return {
        "mesh_success": success,
        "vertex_count": vertex_count,
        "face_count": face_count,
        "vertices_finite": finite,
        "bbox_min": minimum.tolist() if vertex_count else [],
        "bbox_max": maximum.tolist() if vertex_count else [],
        "bbox_extent": (maximum - minimum).tolist() if vertex_count else [],
        "bbox_diag": (
            float(np.linalg.norm(maximum - minimum)) if vertex_count else 0.0
        ),
        "max_abs_coordinate": (
            float(np.max(np.abs(np.concatenate((minimum, maximum)))))
            if vertex_count
            else math.inf
        ),
        "watertight": bool(mesh.is_watertight),
        "winding_consistent": bool(mesh.is_winding_consistent),
        "component_count": int(mesh.body_count),
    }


def load_pixal_result(protocol: dict[str, Any], case: dict[str, Any]) -> dict[str, Any]:
    mesh_path = pixal3d_mesh_path(
        Path(protocol["_protocol_path"]), str(case["case_id"])
    )
    result_path = pixal3d_result_path(
        Path(protocol["_protocol_path"]), str(case["case_id"])
    )
    if not mesh_path.is_file() or not result_path.is_file():
        raise FileNotFoundError(f"missing Pixal3D output for {case['case_id']}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    validate_official_inference_result(
        result,
        protocol=protocol,
        case=case,
        mesh_path=mesh_path,
    )
    return result


def common_pixal_runtime(results: list[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        raise ValueError("no Pixal3D inference results")
    fields = (
        "model_path",
        "model_snapshot",
        "original_pixal3d_code",
        "pixal3d_pipeline_code",
        "naf_loader_code",
        "naf",
        "background_removal",
        "camera_mode",
        "pipeline_type",
        "sampling_steps",
        "geometry_export",
        "postprocess",
        "postprocess_code",
    )
    runtime = {key: results[0][key] for key in fields}
    for result in results[1:]:
        for key in fields:
            if result.get(key) != runtime[key]:
                raise RuntimeError(f"Pixal3D runtime changed across cases: {key}")
    return runtime


def pixal_training_view_transform(c2w: Iterable[Iterable[float]]) -> np.ndarray:
    """Reproduce Pixal3D data_toolkit.utils.transform_mesh metadata transform."""

    c2w_orig = np.asarray(c2w, dtype=np.float64)
    if c2w_orig.shape != (4, 4) or not np.isfinite(c2w_orig).all():
        raise ValueError("selected c2w must be a finite 4x4 matrix")
    radius = float(np.linalg.norm(c2w_orig[:3, 3]))
    if radius <= 1.0e-12:
        raise ValueError("selected c2w has a degenerate camera radius")
    yaw = -0.5 * math.pi
    pitch = 0.0
    eye = np.asarray(
        [
            radius * math.cos(yaw) * math.cos(pitch),
            radius * math.sin(yaw) * math.cos(pitch),
            radius * math.sin(pitch),
        ],
        dtype=np.float64,
    )
    forward = -eye / np.linalg.norm(eye)
    up_global = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
    right = np.cross(forward, up_global)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    c2w_new = np.eye(4, dtype=np.float64)
    c2w_new[:3, 0] = right
    c2w_new[:3, 1] = up
    c2w_new[:3, 2] = -forward
    c2w_new[:3, 3] = eye
    rz90 = np.asarray(
        [
            [0.0, -1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    c2w_new = c2w_new @ rz90
    r_init = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    r_back = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    r_ply = r_back.copy()
    t_cam = c2w_new @ np.linalg.inv(c2w_orig) @ r_ply
    transform = r_back @ t_cam @ r_init
    if not np.isfinite(transform).all():
        raise RuntimeError("Pixal3D metadata transform is non-finite")
    return transform


def case_canonical_transform(
    case: dict[str, Any],
    transform_spec: dict[str, Any],
) -> np.ndarray:
    frame = case.get("selected_frame")
    if not isinstance(frame, dict) or frame.get("extrinsics_type") != "c2w":
        raise RuntimeError(
            f"case={case.get('case_id')} lacks frozen selected c2w metadata"
        )
    view_transform = pixal_training_view_transform(frame["extrinsic"])
    u, _, vh = np.linalg.svd(view_transform[:3, :3])
    view_rotation = u @ vh
    if np.linalg.det(view_rotation) <= 0.0:
        raise RuntimeError(
            f"case={case.get('case_id')} metadata rotation is not proper"
        )
    view_rotation_matrix = np.eye(4, dtype=np.float64)
    view_rotation_matrix[:3, :3] = view_rotation
    scale = float(transform_spec["derivation"]["uniform_scale"])
    scale_matrix = np.eye(4, dtype=np.float64)
    scale_matrix[:3, :3] *= scale
    # Official final inference first applies o_voxel's GLB axis conversion,
    # then applies inference.py's output rotation. Undo their composition,
    # then invert only the view rotation used by Pixal3D data.
    # Camera translation controls projection distance and must not translate
    # either object-centered canonical mesh.
    output = (
        scale_matrix
        @ np.linalg.inv(view_rotation_matrix)
        @ np.linalg.inv(OFFICIAL_FINAL_EXPORT_FROM_DECODED)
    )
    if not np.allclose(
        output[:3, 3],
        np.zeros(3, dtype=np.float64),
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise RuntimeError("canonical metadata transform introduced translation")
    return output


def command_freeze_transform(args: argparse.Namespace) -> None:
    protocol_path = args.protocol.resolve()
    protocol = validate_protocol(protocol_path)
    protocol["_protocol_path"] = str(protocol_path)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite frozen transform: {output}")
    if len(protocol["cases"]) < int(args.min_audit_objects):
        raise RuntimeError(
            f"canonical audit has {len(protocol['cases'])} cases; "
            f"needs at least {args.min_audit_objects}"
        )
    object_count = len({str(case["object_uid"]) for case in protocol["cases"]})
    if object_count < int(args.min_audit_objects):
        raise RuntimeError(
            f"canonical audit has {object_count} objects; "
            f"needs at least {args.min_audit_objects}"
        )
    direct_margin = fixed_float_value(
        (
        float(case["target_metadata"]["canonical_margin"])
        for case in protocol["cases"]
        ),
        label="Direct canonical margin",
    )
    if not math.isclose(
        direct_margin,
        float(args.expected_direct_margin),
        rel_tol=0.0,
        abs_tol=CANONICAL_MARGIN_ATOL,
    ):
        raise RuntimeError(
            f"Direct canonical margin={direct_margin} differs from expected "
            f"{args.expected_direct_margin}"
        )

    results = []
    audits = []
    limit = float(args.native_aabb_half_extent) + float(args.aabb_tolerance)
    for case in protocol["cases"]:
        result = load_pixal_result(protocol, case)
        mesh_path = Path(result["mesh"]).resolve()
        geometry = scan_mesh_geometry(mesh_path)
        if not geometry["mesh_success"]:
            raise RuntimeError(f"Pixal3D Mesh failed canonical audit: {case['case_id']}")
        if float(geometry["max_abs_coordinate"]) > limit:
            raise RuntimeError(
                f"Pixal3D Mesh exceeds frozen native AABB for {case['case_id']}: "
                f"{geometry['max_abs_coordinate']} > {limit}"
            )
        results.append(result)
        audits.append(
            {
                "case_id": case["case_id"],
                "object_uid": case["object_uid"],
                "joint_seed": int(case["pixal3d_seed"]),
                "mesh": binding(mesh_path),
                "geometry": geometry,
                "case_canonical_matrix": case_canonical_transform(
                    case,
                    {
                        "derivation": {
                            "uniform_scale": direct_margin
                            / (2.0 * float(args.native_aabb_half_extent))
                        }
                    },
                ).tolist(),
            }
        )
    runtime = common_pixal_runtime(results)
    if runtime["geometry_export"] != OFFICIAL_GEOMETRY_EXPORT:
        raise RuntimeError("Pixal3D output is not the official final GLB")
    if runtime["postprocess"] != OFFICIAL_POSTPROCESS:
        raise RuntimeError("Pixal3D official postprocess configuration changed")

    scale = direct_margin / (2.0 * float(args.native_aabb_half_extent))
    pixal_transform_source = (
        Path(__file__).resolve().parents[1]
        / "Pixal3D"
        / "data_toolkit"
        / "utils.py"
    )
    body = {
        "format": TRANSFORM_FORMAT,
        "score_independent": True,
        "per_object_adaptation": False,
        "source_frame": "Pixal3D official postprocessed final GLB frame",
        "target_frame": "Direct-SLAT canonical latent frame",
        "transform_rule": (
            "inverse_pixal_official_final_export_and_training_view_rotation_v2"
        ),
        "derivation": {
            "native_training_aabb": [
                [-float(args.native_aabb_half_extent)] * 3,
                [float(args.native_aabb_half_extent)] * 3,
            ],
            "direct_canonical_margin": direct_margin,
            "uniform_scale": scale,
            "translation": (
                "zero; selected camera position is projection metadata and "
                "is never applied to object-centered canonical geometry"
            ),
            "rotation": (
                "undo o_voxel's GLB axis conversion composed with the bound "
                "official inference.py output rotation, then invert the proper "
                "Pixal3D training view rotation computed from the frozen "
                "selected c2w; this rule never reads predicted or GT geometry"
            ),
            "o_voxel_to_glb_matrix": O_VOXEL_TO_GLB.tolist(),
            "inference_output_rotation_matrix": (
                OFFICIAL_EXPORT_ROTATION.tolist()
            ),
            "official_final_export_from_decoded_matrix": (
                OFFICIAL_FINAL_EXPORT_FROM_DECODED.tolist()
            ),
            "forbidden": [
                "GT-score fitting",
                "per-object bbox normalization",
                "ICP",
                "autoframe",
                "best-of-transform",
            ],
        },
        "pixal3d_runtime": runtime,
        "pixal3d_view_transform_source": binding(pixal_transform_source),
        "audit_protocol": binding(protocol_path),
        "audit_protocol_sha256": protocol["protocol_sha256"],
        "audit_object_count": object_count,
        "audit_case_count": len(audits),
        "native_aabb_tolerance": float(args.aabb_tolerance),
        "audits": audits,
        "code": binding(Path(__file__).resolve()),
    }
    body["transform_sha256"] = canonical_sha256(body)
    atomic_json(output, body)
    print(json.dumps({"status": "passed", "output": str(output), **body}, indent=2))


def validate_transform(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("format") != TRANSFORM_FORMAT:
        raise ValueError(f"unexpected transform format={value.get('format')!r}")
    expected = str(value.get("transform_sha256", ""))
    body = dict(value)
    body.pop("transform_sha256", None)
    if canonical_sha256(body) != expected:
        raise RuntimeError("canonical transform SHA mismatch")
    if value.get("score_independent") is not True:
        raise RuntimeError("canonical transform is not score independent")
    if value.get("per_object_adaptation") is not False:
        raise RuntimeError("per-object canonical adaptation is forbidden")
    if value.get("transform_rule") != (
        "inverse_pixal_official_final_export_and_training_view_rotation_v2"
    ):
        raise RuntimeError("unexpected canonical transform rule")
    return value


def direct_record_map(report: dict[str, Any]) -> dict[tuple[str, int], dict[str, Any]]:
    if report.get("format") not in DIRECT_REPORT_FORMATS:
        raise ValueError(f"unexpected Direct report format={report.get('format')!r}")
    if report.get("same_coordinates") != "both branches use frozen corrected-SS coords":
        raise RuntimeError("Direct report lacks corrected-coordinate binding")
    if report.get("same_noise") != "coordinate-keyed SLAT initial noise is bit-identical":
        raise RuntimeError("Direct report lacks same-noise binding")
    output = {}
    for row in report.get("records", []):
        key = (str(row["pair_id"]), int(row["joint_seed"]))
        if key in output:
            raise RuntimeError(f"duplicate Direct record: {key}")
        if row.get("same_initial_noise") is not True:
            raise RuntimeError(f"Direct record lacks same-noise flag: {key}")
        output[key] = row
    return output


def basic_structure(mesh) -> dict[str, Any]:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces)
    finite = bool(np.isfinite(vertices).all())
    success = bool(len(vertices) > 0 and len(faces) > 0 and finite)
    extent = np.ptp(vertices, axis=0) if len(vertices) else np.zeros(3)
    return {
        "mesh_success": success,
        "vertex_count": int(len(vertices)),
        "face_count": int(len(faces)),
        "vertices_finite": finite,
        "bbox_extent": [float(value) for value in extent],
        "bbox_diag": float(np.linalg.norm(extent)),
    }


def aggregate_utility(
    records: list[dict[str, Any]],
    *,
    expected_seeds: list[int],
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    by_object: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        by_object[str(row["object_uid"])].append(row)

    object_rows = []
    for object_uid, rows in sorted(by_object.items()):
        actual_seeds = sorted(int(row["joint_seed"]) for row in rows)
        if actual_seeds != sorted(expected_seeds):
            raise RuntimeError(
                f"object={object_uid} seed coverage differs: "
                f"actual={actual_seeds} expected={sorted(expected_seeds)}"
            )
        method_values: dict[str, Any] = {}
        for method in METHODS:
            successful = [
                row["methods"][method]
                for row in rows
                if row["methods"][method].get("success") is True
            ]
            method_values[method] = {
                "complete": len(successful) == len(expected_seeds),
                "successful_seed_count": len(successful),
                "metrics": (
                    {
                        metric: float(
                            np.mean(
                                [float(value["surface"][metric]) for value in successful]
                            )
                        )
                        for metric in METRICS
                    }
                    if len(successful) == len(expected_seeds)
                    else {}
                ),
            }
        object_rows.append(
            {
                "object_uid": object_uid,
                "seed_count": len(rows),
                "methods": method_values,
            }
        )

    method_summaries = {}
    for method in METHODS:
        complete = [
            row for row in object_rows if row["methods"][method]["complete"]
        ]
        method_summaries[method] = {
            "expected_object_count": len(object_rows),
            "complete_object_count": len(complete),
            "mesh_success_rate": float(len(complete) / max(len(object_rows), 1)),
            "metrics": {
                metric: {
                    "count": len(complete),
                    "mean": float(
                        np.mean(
                            [
                                row["methods"][method]["metrics"][metric]
                                for row in complete
                            ]
                        )
                    ),
                    "median": float(
                        np.median(
                            [
                                row["methods"][method]["metrics"][metric]
                                for row in complete
                            ]
                        )
                    ),
                }
                for metric in METRICS
            }
            if complete
            else {},
        }

    comparisons = {}
    for comparison_index, (name, lhs, rhs) in enumerate(
        (
            ("u1_full_minus_stock", "full", "stock"),
            ("u2_full_minus_pixal3d_native", "full", "pixal3d_native"),
        )
    ):
        complete = [
            row
            for row in object_rows
            if row["methods"][lhs]["complete"] and row["methods"][rhs]["complete"]
        ]
        deltas: dict[str, list[float]] = {metric: [] for metric in METRICS}
        paired_rows = []
        for row in complete:
            output = {}
            for metric in METRICS:
                left = float(row["methods"][lhs]["metrics"][metric])
                right = float(row["methods"][rhs]["metrics"][metric])
                delta = right - left if metric in LOWER_IS_BETTER else left - right
                deltas[metric].append(delta)
                output[metric] = delta
            paired_rows.append({"object_uid": row["object_uid"], "deltas": output})
        failures = [
            row["object_uid"]
            for row in object_rows
            if not (row["methods"][lhs]["complete"] and row["methods"][rhs]["complete"])
        ]
        comparisons[name] = {
            "lhs": lhs,
            "rhs": rhs,
            "positive_means_lhs_better": True,
            "expected_object_count": len(object_rows),
            "complete_object_count": len(complete),
            "failed_object_count": len(failures),
            "failed_object_uids": failures,
            "metrics": {
                metric: numeric_summary(
                    values,
                    bootstrap_samples=bootstrap_samples,
                    seed=seed + comparison_index * 100 + metric_index,
                )
                for metric_index, (metric, values) in enumerate(deltas.items())
            }
            if complete
            else {},
            "object_rows": paired_rows,
        }
    return {
        "object_count": len(object_rows),
        "object_rows": object_rows,
        "methods": method_summaries,
        "comparisons": comparisons,
    }


def utility_decision(
    summary: dict[str, Any],
    *,
    profile: str,
) -> dict[str, Any]:
    minimum_objects = {
        "report_only": 1,
        "smoke": 6,
        "exploratory": 32,
        "confirmatory": 64,
    }[profile]
    object_count = int(summary["object_count"])

    def comparison_checks(name: str, require_ci: bool) -> dict[str, bool]:
        row = summary["comparisons"][name]
        complete = (
            int(row["complete_object_count"]) == object_count
            and int(row["failed_object_count"]) == 0
        )
        metrics = row.get("metrics", {})
        direction = all(
            metric in metrics
            and float(metrics[metric]["mean"]) > 0.0
            and float(metrics[metric]["median"]) > 0.0
            and float(metrics[metric]["positive_rate"]) > 0.5
            for metric in ("chamfer_l1", "fscore_0p02")
        )
        ci = all(
            metric in metrics
            and float(metrics[metric]["bootstrap_mean_95_ci"][0]) > 0.0
            for metric in ("chamfer_l1", "fscore_0p02")
        )
        return {
            "complete_coverage": complete,
            "chamfer_and_fscore_mean_median_positive_with_majority_wins": direction,
            "chamfer_and_fscore_bootstrap_lower_positive": ci,
            "passed": complete and direction and (ci if require_ci else True),
        }

    require_ci = profile == "confirmatory"
    u1 = comparison_checks("u1_full_minus_stock", require_ci)
    u2 = comparison_checks("u2_full_minus_pixal3d_native", require_ci)
    enough_objects = object_count >= minimum_objects
    if profile == "report_only":
        passed = enough_objects
        next_step = False
    elif profile == "smoke":
        # Six-object Pixal3D direction is diagnostic. U1 rollout stability is
        # the preregistered condition for spending compute on n=32.
        passed = enough_objects and u1["passed"]
        next_step = passed
    else:
        passed = enough_objects and u1["passed"] and u2["passed"]
        next_step = passed
    return {
        "profile": profile,
        "minimum_object_count": minimum_objects,
        "object_count_sufficient": enough_objects,
        "u1_full_gt_stock": u1,
        "u2_full_gt_pixal3d_native": u2,
        "passed": passed,
        "continue_to_next_stage": next_step,
        "long_training_unlocked": profile == "confirmatory" and passed,
        "smoke_u2_is_diagnostic_only": profile == "smoke",
    }


def command_evaluate(args: argparse.Namespace) -> None:
    protocol_path = args.protocol.resolve()
    protocol = validate_protocol(protocol_path)
    protocol["_protocol_path"] = str(protocol_path)
    transform_path = args.pixal_transform.resolve()
    transform = validate_transform(transform_path)
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite utility output: {output_dir}")
    output_dir.mkdir(parents=True)

    direct_binding = protocol["bindings"]["current_report"]
    direct_path = Path(direct_binding["path"]).resolve()
    if sha256_file(direct_path) != direct_binding["sha256"]:
        raise RuntimeError("frozen Direct report changed")
    direct_report = json.loads(direct_path.read_text(encoding="utf-8"))
    direct_rows = direct_record_map(direct_report)
    case_keys = {
        (str(case["pair_id"]), int(case["current_seed"]))
        for case in protocol["cases"]
    }
    if case_keys != set(direct_rows):
        raise RuntimeError(
            "utility protocol must exactly cover the Direct report records: "
            f"missing={sorted(set(direct_rows) - case_keys)} "
            f"unexpected={sorted(case_keys - set(direct_rows))}"
        )
    object_count = len({str(case["object_uid"]) for case in protocol["cases"]})
    expected_seeds = parse_unique_csv(args.expected_seeds, int)
    if int(args.expected_objects) > 0 and object_count != int(args.expected_objects):
        raise RuntimeError(
            f"protocol object_count={object_count}, expected={args.expected_objects}"
        )
    if {
        int(case["current_seed"]) for case in protocol["cases"]
    } != set(expected_seeds):
        raise RuntimeError("protocol seed set differs from --expected_seeds")
    freshness_audit = None
    if args.exclude_protocol is not None:
        excluded_path = args.exclude_protocol.resolve()
        excluded = validate_protocol(excluded_path)
        current_objects = {str(case["object_uid"]) for case in protocol["cases"]}
        excluded_objects = {
            str(case["object_uid"]) for case in excluded["cases"]
        }
        current_sources = {
            str(case["target_metadata"]["source_glb_sha256"])
            for case in protocol["cases"]
        }
        excluded_sources = {
            str(case["target_metadata"]["source_glb_sha256"])
            for case in excluded["cases"]
        }
        object_overlap = sorted(current_objects & excluded_objects)
        source_overlap = sorted(current_sources & excluded_sources)
        freshness_audit = {
            "excluded_protocol": binding(excluded_path),
            "excluded_protocol_sha256": excluded["protocol_sha256"],
            "object_overlap_count": len(object_overlap),
            "source_glb_overlap_count": len(source_overlap),
            "object_overlap": object_overlap,
            "source_glb_overlap": source_overlap,
            "passed": not object_overlap and not source_overlap,
        }
        if not freshness_audit["passed"]:
            raise RuntimeError(
                "fresh utility holdout overlaps excluded exploratory protocol"
            )
    elif args.decision_profile == "confirmatory":
        raise RuntimeError("confirmatory profile requires --exclude_protocol")
    protocol_margin = fixed_float_value(
        (
            float(case["target_metadata"]["canonical_margin"])
            for case in protocol["cases"]
        ),
        label="protocol canonical margin",
    )
    expected_margin = float(transform["derivation"]["direct_canonical_margin"])
    if not math.isclose(
        protocol_margin,
        expected_margin,
        rel_tol=0.0,
        abs_tol=CANONICAL_MARGIN_ATOL,
    ):
        raise RuntimeError("protocol canonical margin differs from frozen transform")

    records = []
    runtime_expected = transform["pixal3d_runtime"]
    for position, case in enumerate(protocol["cases"], start=1):
        key = (str(case["pair_id"]), int(case["current_seed"]))
        direct = direct_rows[key]
        if (
            str(direct["uid"]) != str(case["uid"])
            or str(direct["object_uid"]) != str(case["object_uid"])
            or int(direct["dataset_index"]) != int(case["dataset_index"])
        ):
            raise RuntimeError(f"Direct/protocol identity mismatch for {case['case_id']}")
        if int(case["current_seed"]) != int(case["pixal3d_seed"]):
            raise RuntimeError(
                f"cross-architecture integer seed differs for {case['case_id']}"
            )
        target_path = Path(case["target_mesh"]["path"]).resolve()
        if sha256_file(target_path) != case["target_mesh"]["sha256"]:
            raise RuntimeError(f"target binding changed for {case['case_id']}")
        direct_root = direct_path.parent
        pair_root = direct_root / "mesh_pairs" / str(direct["pair_id"])
        paths = {
            "stock": pair_root / "stock" / "mesh_canonical.obj",
            "full": pair_root / "full" / "mesh_canonical.obj",
            "pixal3d_native": pixal3d_mesh_path(protocol_path, case["case_id"]),
        }
        if Path(case["current_mesh"]["path"]).resolve() != paths["full"].resolve():
            raise RuntimeError(f"Full mesh path changed for {case['case_id']}")

        target = load_mesh(target_path)
        methods = {}
        pixal_result = None
        try:
            pixal_result = load_pixal_result(protocol, case)
            runtime = common_pixal_runtime([pixal_result])
            for field, expected in runtime_expected.items():
                if runtime.get(field) != expected:
                    raise RuntimeError(
                        f"Pixal3D runtime differs from canonical audit: {field}"
                    )
        except Exception as error:
            methods["pixal3d_native"] = {
                "success": False,
                "error": f"{type(error).__name__}: {error}",
            }
        metric_seed = int(args.seed) + position * 1009
        for method in METHODS:
            if method in methods:
                continue
            path = paths[method].resolve()
            try:
                if not path.is_file():
                    raise FileNotFoundError(path)
                mesh = load_mesh(path)
                if method == "pixal3d_native":
                    canonical_matrix = case_canonical_transform(case, transform)
                    mesh.apply_transform(canonical_matrix)
                else:
                    canonical_matrix = np.eye(4, dtype=np.float64)
                structure = basic_structure(mesh)
                if not structure["mesh_success"]:
                    raise RuntimeError("empty or non-finite Mesh")
                surface = surface_metrics(
                    mesh,
                    target,
                    count=int(args.surface_samples),
                    seed=metric_seed,
                    thresholds=(0.01, 0.02, 0.05),
                )
                methods[method] = {
                    "success": True,
                    "mesh": binding(path),
                    "canonical_transform": (
                        {
                            "spec": binding(transform_path),
                            "case_matrix": canonical_matrix.tolist(),
                            "metadata_only": True,
                        }
                        if method == "pixal3d_native"
                        else "identity"
                    ),
                    "structure": structure,
                    "surface": surface,
                }
            except Exception as error:
                methods[method] = {
                    "success": False,
                    "mesh_path": str(path),
                    "error": f"{type(error).__name__}: {error}",
                }
        records.append(
            {
                "case_id": case["case_id"],
                "uid": case["uid"],
                "object_uid": case["object_uid"],
                "view_count": int(case["view_count"]),
                "pair_id": case["pair_id"],
                "joint_seed": int(case["current_seed"]),
                "same_corrected_ss_coordinates_stock_vs_full": True,
                "same_initial_slat_noise_stock_vs_full": True,
                "same_integer_seed_id_full_vs_pixal3d": (
                    int(case["current_seed"]) == int(case["pixal3d_seed"])
                ),
                "same_latent_noise_full_vs_pixal3d": False,
                "target_mesh": case["target_mesh"],
                "methods": methods,
                "pixal3d_inference": pixal_result,
            }
        )
        print(
            f"[utility_eval] {position}/{len(protocol['cases'])} {case['case_id']}",
            flush=True,
        )

    summary = aggregate_utility(
        records,
        expected_seeds=expected_seeds,
        bootstrap_samples=int(args.bootstrap_samples),
        seed=int(args.seed),
    )
    decision = utility_decision(summary, profile=args.decision_profile)
    report = {
        "format": FORMAT,
        "formal": args.decision_profile == "confirmatory",
        "stage": args.decision_profile,
        "passed": bool(decision["passed"]),
        "protocol": binding(protocol_path),
        "protocol_sha256": protocol["protocol_sha256"],
        "direct_report": direct_binding,
        "pixal3d_canonical_transform": binding(transform_path),
        "evaluation": {
            "surface_samples": int(args.surface_samples),
            "bootstrap_samples": int(args.bootstrap_samples),
            "seed": int(args.seed),
            "object_weighting": "average seeds per object before summary/bootstrap",
            "surface_scorer": binding(Path(__file__).resolve()),
            "coordinate_policy": (
                "stock/full remain in Direct canonical frame; Pixal3D receives "
                "one frozen score-independent transform; no per-object bbox "
                "normalization, ICP, autoframe, or best transform"
            ),
            "cross_architecture_noise_claim": (
                "same object, frozen input view, integer seed ID, target and "
                "scorer; latent noise is not claimed identical"
            ),
        },
        "freshness_audit": freshness_audit,
        "summary": summary,
        "decision": decision,
        "records": records,
        "guardrail": (
            "U1 and U2 are separate. Overall Base superiority requires both. "
            "Only a fresh >=64-object confirmatory PASS unlocks long training."
        ),
    }
    atomic_json(output_dir / "report.json", report)

    csv_path = output_dir / "object_metrics.csv"
    temporary = csv_path.with_name(f".{csv_path.name}.tmp-{os.getpid()}")
    fields = ["comparison", "object_uid", *METRICS]
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for name, comparison in summary["comparisons"].items():
            for row in comparison["object_rows"]:
                writer.writerow(
                    {
                        "comparison": name,
                        "object_uid": row["object_uid"],
                        **row["deltas"],
                    }
                )
    os.replace(temporary, csv_path)

    lines = [
        "Direct-SLAT / native Pixal3D canonical utility evaluation",
        "=========================================================",
        f"profile: {args.decision_profile}",
        f"objects: {summary['object_count']}",
    ]
    for name, row in summary["comparisons"].items():
        lines.append("")
        lines.append(
            f"{name}: complete={row['complete_object_count']}/"
            f"{row['expected_object_count']}"
        )
        for metric in ("chamfer_l1", "fscore_0p02", "normal_consistency"):
            value = row.get("metrics", {}).get(metric)
            if value:
                lines.append(
                    f"  {metric}: mean={value['mean']:+.8f} "
                    f"median={value['median']:+.8f} "
                    f"win={value['positive_rate']:.6f} "
                    f"CI={value['bootstrap_mean_95_ci']}"
                )
    lines.extend(
        [
            "",
            f"PASS: {decision['passed']}",
            f"continue: {decision['continue_to_next_stage']}",
            f"long training unlocked: {decision['long_training_unlocked']}",
            "",
            report["guardrail"],
        ]
    )
    atomic_text(output_dir / "summary.txt", "\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    if not decision["passed"]:
        raise SystemExit(2)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    freeze = subparsers.add_parser(
        "freeze-transform",
        help="freeze one score-independent native-Pixal3D canonical transform",
    )
    freeze.add_argument("--protocol", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)
    freeze.add_argument("--min_audit_objects", type=int, default=6)
    freeze.add_argument("--native_aabb_half_extent", type=float, default=0.5)
    freeze.add_argument("--aabb_tolerance", type=float, default=0.02)
    freeze.add_argument("--expected_direct_margin", type=float, default=0.9)
    freeze.set_defaults(handler=command_freeze_transform)

    evaluate = subparsers.add_parser(
        "evaluate",
        help="score stock, Full and native Pixal3D without per-object alignment",
    )
    evaluate.add_argument("--protocol", type=Path, required=True)
    evaluate.add_argument("--pixal_transform", type=Path, required=True)
    evaluate.add_argument("--output_dir", type=Path, required=True)
    evaluate.add_argument("--expected_objects", type=int, default=0)
    evaluate.add_argument("--expected_seeds", default="42")
    evaluate.add_argument(
        "--exclude_protocol",
        type=Path,
        default=None,
        help=(
            "Require zero object_uid and source-GLB overlap with this frozen "
            "protocol. Mandatory for confirmatory evaluation."
        ),
    )
    evaluate.add_argument("--surface_samples", type=int, default=20000)
    evaluate.add_argument("--bootstrap_samples", type=int, default=5000)
    evaluate.add_argument("--seed", type=int, default=20260728)
    evaluate.add_argument(
        "--decision_profile",
        choices=("report_only", "smoke", "exploratory", "confirmatory"),
        default="report_only",
    )
    evaluate.set_defaults(handler=command_evaluate)
    return parser


def main() -> None:
    args = make_parser().parse_args()
    if min(
        int(getattr(args, "surface_samples", 1)),
        int(getattr(args, "bootstrap_samples", 1)),
    ) <= 0:
        raise ValueError("sample counts must be positive")
    args.handler(args)


if __name__ == "__main__":
    main()
