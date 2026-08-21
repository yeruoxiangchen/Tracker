#!/usr/bin/env python3
"""Evaluate synthetic1k No-VGGT beside current No-VGGT and ReconViaGen."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pose_point_depth_mv.evaluate_objaverse16_no_vggt import (
    aggregate,
    atomic_json,
    export_obj_atomic,
    parse_float_csv,
    stable_metric_seed,
)
from pose_point_depth_mv.evaluate_objaverse16_reconviagen import (
    LOWER_IS_BETTER,
    REPORT_FORMAT as O9_REPORT_FORMAT,
    decoder_to_source_mesh,
    load_json,
    summarize,
)
from pose_point_depth_mv.export_direct_flow_mesh_pairs import load_canonical_gt
from pose_point_depth_mv.freeze_objaverse16_test import PROTOCOL_FORMAT, sha256_file
from pose_point_depth_mv.infer_objaverse16_no_vggt_synthetic1k import (
    MANIFEST_FORMAT as SYNTHETIC1K_MANIFEST_FORMAT,
    METHOD as SYNTHETIC1K_INFERENCE_METHOD,
)
from pose_point_depth_mv.mesh_benchmark_metrics import (
    mesh_structure_metrics,
    surface_metrics,
)
from pose_point_depth_mv.render_direct_slat_fourway import (
    LATENT_DECODER_TO_REFERENCE,
)


REPORT_FORMAT = "pose_point_depth_mv.objaverse16_synthetic1k_threeway_evaluation.v1"
RECORD_FORMAT = "pose_point_depth_mv.objaverse16_synthetic1k_metric_record.v1"
SYNTHETIC1K_METHOD = "synthetic1k_no_vggt"
CURRENT_METHOD = "current_no_vggt"
RECON_METHOD = "reconviagen_original"
METHODS = (SYNTHETIC1K_METHOD, CURRENT_METHOD, RECON_METHOD)


def paired_comparison(
    left: dict[str, dict[str, Any]],
    right: dict[str, dict[str, Any]],
    *,
    left_method: str,
    right_method: str,
) -> dict[str, Any]:
    if set(left) != set(right) or not left:
        raise RuntimeError("paired method object sets differ")
    first_uid = next(iter(left))
    metric_names = sorted(set(left[first_uid]["surface"]))
    if set(right[first_uid]["surface"]) != set(metric_names):
        raise RuntimeError("paired method metric sets differ")
    metrics: dict[str, Any] = {}
    for name in metric_names:
        deltas = []
        for object_uid in sorted(left):
            left_value = float(left[object_uid]["surface"][name])
            right_value = float(right[object_uid]["surface"][name])
            delta = (
                right_value - left_value
                if name in LOWER_IS_BETTER
                else left_value - right_value
            )
            deltas.append(delta)
        metrics[f"{name}_left_improvement"] = summarize(deltas)
    chamfer_deltas = {
        object_uid: (
            float(right[object_uid]["surface"]["chamfer_l1"])
            - float(left[object_uid]["surface"]["chamfer_l1"])
        )
        for object_uid in left
    }
    return {
        "left": left_method,
        "right": right_method,
        "positive_definition": f"positive means {left_method} is better",
        "metric_deltas": metrics,
        "chamfer_l1_wins": {
            left_method: sum(value > 0.0 for value in chamfer_deltas.values()),
            right_method: sum(value < 0.0 for value in chamfer_deltas.values()),
            "ties": sum(value == 0.0 for value in chamfer_deltas.values()),
        },
        "per_object_chamfer_l1_left_improvement": chamfer_deltas,
    }


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection_manifest", required=True)
    parser.add_argument("--synthetic1k_inference_manifest", required=True)
    parser.add_argument("--existing_o9_report", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--surface_samples", type=int, default=20000)
    parser.add_argument(
        "--fscore_thresholds", type=parse_float_csv, default=[0.01, 0.02, 0.05]
    )
    parser.add_argument("--resume", action="store_true")
    return parser


def _index_unique(
    rows: list[dict[str, Any]], key: str, *, label: str
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        value = str(row[key])
        if value in result:
            raise RuntimeError(f"duplicate {label} object={value}")
        result[value] = row
    return result


def _validate_synthetic1k(
    path: Path, *, selection_path: Path, selection_sha: str
) -> dict[str, Any]:
    manifest = load_json(path)
    expected = {
        "format": SYNTHETIC1K_MANIFEST_FORMAT,
        "method": SYNTHETIC1K_INFERENCE_METHOD,
        "protocol_scope": "frozen_objaverse_test16",
        "formal": False,
        "object_count": 16,
        "record_count": 16,
        "seeds": [42],
        "training_object_disjoint": True,
        "source_mesh_disjoint": True,
        "vggt_model_loaded": False,
        "vggt_model_executed": False,
        "point_cloud_tensor_present": False,
        "point_cloud_consumed": False,
        "target_or_metric_consumed": False,
        "passed": True,
    }
    mismatch = {
        key: (manifest.get(key), value)
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    model_path = Path(str(manifest.get("model_input_manifest", ""))).resolve()
    if not model_path.is_file():
        mismatch["model_input_manifest"] = (str(model_path), "existing file")
    else:
        model = load_json(model_path)
        if model.get("selection_manifest_sha256") != selection_sha:
            mismatch["model_input_selection_sha256"] = (
                model.get("selection_manifest_sha256"),
                selection_sha,
            )
        if Path(str(model.get("selection_manifest", ""))).resolve() != selection_path:
            mismatch["model_input_selection"] = (
                model.get("selection_manifest"),
                str(selection_path),
            )
    if mismatch:
        raise RuntimeError(f"synthetic1k inference contract differs: {mismatch}")
    return manifest


def _validate_o9(
    path: Path,
    *,
    selection_sha: str,
    surface_samples_count: int,
    thresholds: list[float],
) -> dict[str, Any]:
    report = load_json(path)
    expected = {
        "format": O9_REPORT_FORMAT,
        "passed": True,
        "formal": False,
        "methods": [CURRENT_METHOD, RECON_METHOD],
        "object_count": 16,
        "record_count": 32,
        "pair_count": 16,
        "seeds": [42],
        "selection_manifest_sha256": selection_sha,
        "surface_samples": int(surface_samples_count),
        "fscore_thresholds": list(thresholds),
        "training_object_disjoint": True,
        "source_mesh_disjoint": True,
    }
    mismatch = {
        key: (report.get(key), value)
        for key, value in expected.items()
        if report.get(key) != value
    }
    coordinate = dict(report.get("coordinate_evaluation", {}))
    if (
        coordinate.get("decoder_to_source_axis_transform")
        != LATENT_DECODER_TO_REFERENCE.tolist()
        or coordinate.get("applied_identically_to_both_methods") is not True
    ):
        mismatch["coordinate_evaluation"] = (
            coordinate,
            "fixed decoder-to-source transform applied to both methods",
        )
    if mismatch:
        raise RuntimeError(f"existing O9 contract differs: {mismatch}")
    return report


def main() -> None:
    args = make_parser().parse_args()
    if int(args.surface_samples) <= 0:
        raise ValueError("surface_samples must be positive")
    selection_path = Path(args.selection_manifest).expanduser().resolve()
    synthetic_path = (
        Path(args.synthetic1k_inference_manifest).expanduser().resolve()
    )
    o9_path = Path(args.existing_o9_report).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    selection = load_json(selection_path)
    protocol = dict(selection.get("objaverse16_protocol", {}))
    if (
        protocol.get("format") != PROTOCOL_FORMAT
        or protocol.get("passed") is not True
        or protocol.get("object_count") != 16
        or protocol.get("training_object_disjoint") is not True
        or protocol.get("source_mesh_disjoint") is not True
    ):
        raise RuntimeError("selection is not the passed disjoint Objaverse test16")
    selection_sha = sha256_file(selection_path)
    synthetic = _validate_synthetic1k(
        synthetic_path,
        selection_path=selection_path,
        selection_sha=selection_sha,
    )
    o9 = _validate_o9(
        o9_path,
        selection_sha=selection_sha,
        surface_samples_count=int(args.surface_samples),
        thresholds=list(args.fscore_thresholds),
    )

    selected = list(selection["samples"])
    selected_objects = {str(row["object_uid"]) for row in selected}
    synthetic_by_object = _index_unique(
        list(synthetic["objects"]), "object_id", label="synthetic1k"
    )
    baseline_records = [
        dict(record)
        for record in o9["records"]
        if record.get("method") in (CURRENT_METHOD, RECON_METHOD)
    ]
    baseline_by_method = {
        method: _index_unique(
            [row for row in baseline_records if row["method"] == method],
            "object_uid",
            label=method,
        )
        for method in (CURRENT_METHOD, RECON_METHOD)
    }
    if set(synthetic_by_object) != selected_objects or any(
        set(rows) != selected_objects for rows in baseline_by_method.values()
    ):
        raise RuntimeError("selection/synthetic1k/O9 object sets differ")

    output_dir.mkdir(parents=True, exist_ok=True)
    synthetic_records: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for position, selected_row in enumerate(selected, start=1):
        uid = str(selected_row["uid"])
        object_uid = str(selected_row["object_uid"])
        source_group = str(selected_row["source_group"])
        view_count = int(
            selected_row["objaverse16_selection"]["expected_point_prior_view_count"]
        )
        prediction = synthetic_by_object[object_uid]
        if int(prediction["seed"]) != 42:
            raise RuntimeError(f"synthetic1k seed differs for uid={uid}")
        target_mesh, target_metadata = load_canonical_gt(
            {
                "ss_latent": selected_row["ss_latent"],
                "source_glb": selected_row["source_glb"],
            }
        )
        target_structure = mesh_structure_metrics(target_mesh)
        target_path = output_dir / "targets" / object_uid / "mesh_source_canonical.obj"
        if not target_path.is_file():
            export_obj_atomic(target_mesh, target_path)
        metric_seed = stable_metric_seed(uid, 42)
        prediction_path = Path(str(prediction["mesh"])).resolve()
        prediction_sha = sha256_file(prediction_path)
        if prediction_sha != str(prediction["mesh_sha256"]):
            raise RuntimeError(f"synthetic1k prediction SHA differs: {prediction_path}")
        identity = {
            "format": RECORD_FORMAT,
            "method": SYNTHETIC1K_METHOD,
            "uid": uid,
            "object_uid": object_uid,
            "seed": 42,
            "prediction_sha256": prediction_sha,
            "target_source_sha256": str(target_metadata["source_glb_sha256"]),
            "metric_seed": int(metric_seed),
            "surface_samples": int(args.surface_samples),
            "fscore_thresholds": list(args.fscore_thresholds),
            "decoder_to_source_axis_transform": (
                LATENT_DECODER_TO_REFERENCE.tolist()
            ),
        }
        record_path = (
            output_dir / "records" / object_uid / "synthetic1k_no_vggt_seed42.json"
        )
        if record_path.is_file():
            if not args.resume:
                raise FileExistsError(record_path)
            record = load_json(record_path)
            if (
                record.get("passed") is not True
                or any(record.get(key) != value for key, value in identity.items())
            ):
                raise RuntimeError(f"stale synthetic1k metric record: {record_path}")
            synthetic_records.append(record)
        else:
            try:
                predicted_mesh = decoder_to_source_mesh(prediction_path)
                structure = mesh_structure_metrics(predicted_mesh)
                if not structure["mesh_success"]:
                    raise RuntimeError("synthetic1k prediction is empty after axis fix")
                surface = surface_metrics(
                    predicted_mesh,
                    target_mesh,
                    count=int(args.surface_samples),
                    seed=metric_seed,
                    thresholds=args.fscore_thresholds,
                )
                record = {
                    **identity,
                    "created_at_utc": datetime.now(timezone.utc).isoformat(),
                    "source_group": source_group,
                    "view_count": view_count,
                    "prediction": str(prediction_path),
                    "prediction_frame_before_evaluation": (
                        "latent decoder canonical; transform_pose=False"
                    ),
                    "prediction_frame_during_evaluation": "normalized source-GLB frame",
                    "target": target_metadata,
                    "target_canonical_mesh": str(target_path),
                    "alignment": "fixed proper axis transform only; no ICP/scale/GT fit",
                    "surface": surface,
                    "structure": structure,
                    "target_structure": target_structure,
                    "passed": True,
                }
                atomic_json(record_path, record)
                synthetic_records.append(record)
            except Exception as error:
                failures.append(
                    {"uid": uid, "object_uid": object_uid, "error": repr(error)}
                )
                raise

        for method in (CURRENT_METHOD, RECON_METHOD):
            baseline = baseline_by_method[method][object_uid]
            expected_baseline = {
                "uid": uid,
                "seed": 42,
                "source_group": source_group,
                "view_count": view_count,
                "metric_seed": metric_seed,
                "surface_samples": int(args.surface_samples),
                "fscore_thresholds": list(args.fscore_thresholds),
                "target_source_sha256": str(target_metadata["source_glb_sha256"]),
                "decoder_to_source_axis_transform": (
                    LATENT_DECODER_TO_REFERENCE.tolist()
                ),
                "passed": True,
            }
            mismatch = {
                key: (baseline.get(key), value)
                for key, value in expected_baseline.items()
                if baseline.get(key) != value
            }
            baseline_prediction = Path(str(baseline.get("prediction", ""))).resolve()
            if baseline.get("prediction_sha256") != sha256_file(baseline_prediction):
                mismatch["prediction_sha256"] = (
                    baseline.get("prediction_sha256"),
                    "current file SHA256",
                )
            if mismatch:
                raise RuntimeError(
                    f"existing O9 record differs for {method}/{object_uid}: {mismatch}"
                )
        print(
            f"[objaverse16_synthetic1k_eval] {position}/16 uid={uid} views={view_count}",
            flush=True,
        )

    records = synthetic_records + baseline_records
    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_source: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    by_views: dict[int, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    record_index: dict[str, dict[str, dict[str, Any]]] = {
        method: {} for method in METHODS
    }
    for record in records:
        method = str(record["method"])
        by_method[method].append(record)
        by_source[str(record["source_group"])][method].append(record)
        by_views[int(record["view_count"])][method].append(record)
        record_index[method][str(record["object_uid"])] = record

    comparisons = {
        "synthetic1k_vs_current": paired_comparison(
            record_index[SYNTHETIC1K_METHOD],
            record_index[CURRENT_METHOD],
            left_method=SYNTHETIC1K_METHOD,
            right_method=CURRENT_METHOD,
        ),
        "synthetic1k_vs_reconviagen": paired_comparison(
            record_index[SYNTHETIC1K_METHOD],
            record_index[RECON_METHOD],
            left_method=SYNTHETIC1K_METHOD,
            right_method=RECON_METHOD,
        ),
    }
    report = {
        "format": REPORT_FORMAT,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "passed": len(synthetic_records) == 16 and len(records) == 48 and not failures,
        "formal": False,
        "protocol_scope": "frozen_objaverse_test16_synthetic1k_addendum",
        "methods": list(METHODS),
        "object_count": len(selected_objects),
        "record_count": len(records),
        "pair_count_per_comparison": 16,
        "seeds": [42],
        "selection_manifest": str(selection_path),
        "selection_manifest_sha256": selection_sha,
        "synthetic1k_inference_manifest": str(synthetic_path),
        "synthetic1k_inference_manifest_sha256": sha256_file(synthetic_path),
        "existing_o9_report": str(o9_path),
        "existing_o9_report_sha256": sha256_file(o9_path),
        "surface_samples": int(args.surface_samples),
        "fscore_thresholds": list(args.fscore_thresholds),
        "coordinate_evaluation": {
            "decoder_to_source_axis_transform": LATENT_DECODER_TO_REFERENCE.tolist(),
            "mapping": "(x,y,z) -> (x,z,-y)",
            "applied_identically_to_all_methods": True,
            "alignment": "fixed proper axis transform only; no ICP/scale/GT fit",
        },
        "method_contracts": {
            SYNTHETIC1K_METHOD: {
                "training": "reviewed1k synthetic train868; no real-video images",
                "vggt_model_executed": False,
                "point_cloud_tensor_consumed": False,
                "target_or_metric_consumed": False,
                "explicit_camera_pose_consumed": True,
            },
            CURRENT_METHOD: o9["method_contracts"][CURRENT_METHOD],
            RECON_METHOD: o9["method_contracts"][RECON_METHOD],
        },
        "training_object_disjoint": True,
        "source_mesh_disjoint": True,
        "summary": {method: aggregate(by_method[method]) for method in METHODS},
        "summary_by_source_group": {
            source: {method: aggregate(rows[method]) for method in METHODS}
            for source, rows in sorted(by_source.items())
        },
        "summary_by_view_count": {
            str(views): {method: aggregate(rows[method]) for method in METHODS}
            for views, rows in sorted(by_views.items())
        },
        "paired_comparisons": comparisons,
        "synthetic1k_records": synthetic_records,
        "baseline_records_reused_from_o9": baseline_records,
        "failures": failures,
        "limitations": [
            "This is a 16-object diagnostic subset, not a formal benchmark.",
            "Inputs are synthetic Objaverse renders with known masks and cameras.",
            "The SS-stage matched-stock result is separate; this report compares decoded full Meshes.",
            "Current and ReconViaGen metric records are reused unchanged from the bound O9 report.",
        ],
    }
    report_path = output_dir / "report.json"
    atomic_json(report_path, report)
    print(
        {
            "passed": report["passed"],
            "formal": report["formal"],
            "objects": report["object_count"],
            "records": report["record_count"],
            "report": str(report_path),
        },
        flush=True,
    )
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
