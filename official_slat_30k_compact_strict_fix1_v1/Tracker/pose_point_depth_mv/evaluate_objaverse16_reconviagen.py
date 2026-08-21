#!/usr/bin/env python3
"""Jointly evaluate current No-VGGT and stock ReconViaGen on Objaverse16."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from pose_point_depth_mv.evaluate_objaverse16_no_vggt import (
    aggregate,
    export_obj_atomic,
    load_mesh,
    parse_float_csv,
    stable_metric_seed,
)
from pose_point_depth_mv.export_direct_flow_mesh_pairs import load_canonical_gt
from pose_point_depth_mv.freeze_objaverse16_test import PROTOCOL_FORMAT, sha256_file
from pose_point_depth_mv.infer_objaverse16_no_vggt_mixed import (
    MANIFEST_FORMAT as CURRENT_MANIFEST_FORMAT,
)
from pose_point_depth_mv.infer_objaverse16_reconviagen import (
    MANIFEST_FORMAT as RECON_MANIFEST_FORMAT,
)
from pose_point_depth_mv.mesh_benchmark_metrics import (
    mesh_structure_metrics,
    surface_metrics,
)
from pose_point_depth_mv.render_direct_slat_fourway import (
    LATENT_DECODER_TO_REFERENCE,
)


REPORT_FORMAT = "pose_point_depth_mv.objaverse16_reconviagen_joint_evaluation.v1"
RECORD_FORMAT = "pose_point_depth_mv.objaverse16_reconviagen_metric_record.v1"
METHODS = ("current_no_vggt", "reconviagen_original")
LOWER_IS_BETTER = {
    "pred_to_gt_mean",
    "gt_to_pred_mean",
    "chamfer_l1",
    "chamfer_l2",
}


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root is not an object: {path}")
    return value


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def summarize(values: Iterable[float]) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=np.float64)
    if not len(array) or not np.isfinite(array).all():
        raise ValueError("metric distribution is empty or non-finite")
    return {
        "count": int(len(array)),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "min": float(array.min()),
        "max": float(array.max()),
        "positive_rate": float(np.mean(array > 0.0)),
        "nonnegative_rate": float(np.mean(array >= 0.0)),
    }


def decoder_to_source_mesh(path: Path):
    mesh = load_mesh(path)
    mesh.apply_transform(LATENT_DECODER_TO_REFERENCE)
    return mesh


def paired_comparison(
    current: dict[str, dict[str, Any]],
    recon: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if set(current) != set(recon) or not current:
        raise RuntimeError("paired method object sets differ")
    metric_names = sorted(set(current[next(iter(current))]["surface"]))
    metrics: dict[str, Any] = {}
    for name in metric_names:
        deltas = []
        for object_uid in sorted(current):
            left = float(current[object_uid]["surface"][name])
            right = float(recon[object_uid]["surface"][name])
            delta = right - left if name in LOWER_IS_BETTER else left - right
            deltas.append(delta)
        metrics[f"{name}_current_improvement"] = summarize(deltas)
    chamfer_wins = {
        object_uid: (
            float(recon[object_uid]["surface"]["chamfer_l1"])
            - float(current[object_uid]["surface"]["chamfer_l1"])
        )
        for object_uid in current
    }
    return {
        "left": "current_no_vggt",
        "right": "reconviagen_original",
        "positive_definition": "positive means current No-VGGT is better",
        "metric_deltas": metrics,
        "chamfer_l1_wins": {
            "current_no_vggt": sum(value > 0.0 for value in chamfer_wins.values()),
            "reconviagen_original": sum(value < 0.0 for value in chamfer_wins.values()),
            "ties": sum(value == 0.0 for value in chamfer_wins.values()),
        },
        "per_object_chamfer_l1_current_improvement": chamfer_wins,
    }


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection_manifest", required=True)
    parser.add_argument("--current_inference_manifest", required=True)
    parser.add_argument("--reconviagen_inference_manifest", required=True)
    parser.add_argument("--legacy_o7_report")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--surface_samples", type=int, default=20000)
    parser.add_argument(
        "--fscore_thresholds", type=parse_float_csv, default=[0.01, 0.02, 0.05]
    )
    parser.add_argument("--resume", action="store_true")
    return parser


def _validate_current(path: Path, selection_sha: str) -> dict[str, Any]:
    manifest = load_json(path)
    expected = {
        "format": CURRENT_MANIFEST_FORMAT,
        "protocol_scope": "frozen_objaverse_test16",
        "formal": False,
        "object_count": 16,
        "record_count": 16,
        "seeds": [42],
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
    model_input = Path(str(manifest.get("model_input_manifest", ""))).resolve()
    if not model_input.is_file():
        mismatch["model_input_manifest"] = (str(model_input), "existing file")
    else:
        model = load_json(model_input)
        if model.get("selection_manifest_sha256") != selection_sha:
            mismatch["model_input_selection_sha256"] = (
                model.get("selection_manifest_sha256"),
                selection_sha,
            )
    if mismatch:
        raise RuntimeError(f"current inference contract differs: {mismatch}")
    return manifest


def _validate_recon(path: Path, selection_sha: str) -> dict[str, Any]:
    manifest = load_json(path)
    expected = {
        "format": RECON_MANIFEST_FORMAT,
        "protocol_scope": "frozen_objaverse_test16",
        "formal": False,
        "method": "reconviagen_original",
        "selection_manifest_sha256": selection_sha,
        "object_count": 16,
        "record_count": 16,
        "seeds": [42],
        "explicit_camera_pose_consumed": False,
        "point_cloud_tensor_consumed": False,
        "target_or_metric_consumed": False,
        "vggt_model_loaded": True,
        "vggt_model_executed": True,
        "training_object_disjoint": True,
        "source_mesh_disjoint": True,
        "passed": True,
    }
    mismatch = {
        key: (manifest.get(key), value)
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if mismatch:
        raise RuntimeError(f"ReconViaGen inference contract differs: {mismatch}")
    return manifest


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


def _record_identity(
    *,
    method: str,
    uid: str,
    object_uid: str,
    prediction_sha256: str,
    target_sha256: str,
    metric_seed: int,
    surface_samples: int,
    thresholds: list[float],
) -> dict[str, Any]:
    return {
        "format": RECORD_FORMAT,
        "method": method,
        "uid": uid,
        "object_uid": object_uid,
        "seed": 42,
        "prediction_sha256": prediction_sha256,
        "target_source_sha256": target_sha256,
        "metric_seed": int(metric_seed),
        "surface_samples": int(surface_samples),
        "fscore_thresholds": list(thresholds),
        "decoder_to_source_axis_transform": LATENT_DECODER_TO_REFERENCE.tolist(),
    }


def main() -> None:
    args = make_parser().parse_args()
    if int(args.surface_samples) <= 0:
        raise ValueError("surface_samples must be positive")
    selection_path = Path(args.selection_manifest).expanduser().resolve()
    current_path = Path(args.current_inference_manifest).expanduser().resolve()
    recon_path = Path(args.reconviagen_inference_manifest).expanduser().resolve()
    legacy_o7_path = (
        Path(args.legacy_o7_report).expanduser().resolve()
        if args.legacy_o7_report
        else None
    )
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
    current_manifest = _validate_current(current_path, selection_sha)
    recon_manifest = _validate_recon(recon_path, selection_sha)

    if legacy_o7_path is not None:
        legacy_o7 = load_json(legacy_o7_path)
        if (
            legacy_o7.get("format")
            != "pose_point_depth_mv.objaverse16_no_vggt_mesh_evaluation.v1"
            or legacy_o7.get("passed") is not True
            or legacy_o7.get("object_count") != 16
        ):
            raise RuntimeError("legacy O7 report is not the completed Objaverse16 result")

    selected = list(selection["samples"])
    selected_objects = {str(row["object_uid"]) for row in selected}
    current_by_object = _index_unique(
        list(current_manifest["objects"]), "object_id", label="current"
    )
    recon_by_object = _index_unique(
        list(recon_manifest["objects"]), "object_uid", label="ReconViaGen"
    )
    if set(current_by_object) != selected_objects or set(recon_by_object) != selected_objects:
        raise RuntimeError("selection/current/ReconViaGen object sets differ")

    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for position, selected_row in enumerate(selected, start=1):
        uid = str(selected_row["uid"])
        object_uid = str(selected_row["object_uid"])
        source_group = str(selected_row["source_group"])
        view_count = int(
            selected_row["objaverse16_selection"]["expected_point_prior_view_count"]
        )
        current_row = current_by_object[object_uid]
        recon_row = recon_by_object[object_uid]
        if (
            int(current_row["seed"]) != 42
            or int(recon_row["seed"]) != 42
            or str(recon_row["uid"]) != uid
            or int(recon_row["view_count"]) != view_count
            or str(recon_row["source_group"]) != source_group
        ):
            raise RuntimeError(f"paired identity/view contract differs for uid={uid}")

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

        method_rows = {
            "current_no_vggt": current_row,
            "reconviagen_original": recon_row,
        }
        for method, prediction_row in method_rows.items():
            prediction_path = Path(str(prediction_row["mesh"])).resolve()
            prediction_sha = sha256_file(prediction_path)
            if prediction_sha != str(prediction_row["mesh_sha256"]):
                raise RuntimeError(f"prediction SHA differs: {prediction_path}")
            identity = _record_identity(
                method=method,
                uid=uid,
                object_uid=object_uid,
                prediction_sha256=prediction_sha,
                target_sha256=str(target_metadata["source_glb_sha256"]),
                metric_seed=metric_seed,
                surface_samples=int(args.surface_samples),
                thresholds=list(args.fscore_thresholds),
            )
            record_path = output_dir / "records" / object_uid / f"{method}_seed42.json"
            if record_path.is_file():
                if not args.resume:
                    raise FileExistsError(record_path)
                record = load_json(record_path)
                if (
                    record.get("passed") is not True
                    or any(record.get(key) != value for key, value in identity.items())
                ):
                    raise RuntimeError(f"stale joint evaluation record: {record_path}")
                records.append(record)
                continue
            try:
                predicted_mesh = decoder_to_source_mesh(prediction_path)
                structure = mesh_structure_metrics(predicted_mesh)
                if not structure["mesh_success"]:
                    raise RuntimeError("prediction is empty or non-finite after axis fix")
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
                records.append(record)
            except Exception as error:
                failures.append(
                    {
                        "uid": uid,
                        "object_uid": object_uid,
                        "method": method,
                        "error": repr(error),
                    }
                )
                raise
        print(
            f"[objaverse16_joint_eval] {position}/16 uid={uid} views={view_count}",
            flush=True,
        )

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

    comparison = paired_comparison(
        record_index["current_no_vggt"],
        record_index["reconviagen_original"],
    )
    report = {
        "format": REPORT_FORMAT,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "passed": len(records) == 32 and not failures,
        "formal": False,
        "protocol_scope": "frozen_objaverse_test16_reconviagen_addendum",
        "methods": list(METHODS),
        "object_count": len(selected_objects),
        "record_count": len(records),
        "pair_count": len(comparison["per_object_chamfer_l1_current_improvement"]),
        "seeds": [42],
        "selection_manifest": str(selection_path),
        "selection_manifest_sha256": selection_sha,
        "current_inference_manifest": str(current_path),
        "current_inference_manifest_sha256": sha256_file(current_path),
        "reconviagen_inference_manifest": str(recon_path),
        "reconviagen_inference_manifest_sha256": sha256_file(recon_path),
        "surface_samples": int(args.surface_samples),
        "fscore_thresholds": list(args.fscore_thresholds),
        "coordinate_evaluation": {
            "decoder_to_source_axis_transform": LATENT_DECODER_TO_REFERENCE.tolist(),
            "mapping": "(x,y,z) -> (x,z,-y)",
            "applied_identically_to_both_methods": True,
            "alignment": "fixed proper axis transform only; no ICP/scale/GT fit",
        },
        "legacy_o7_superseded": {
            "path": str(legacy_o7_path) if legacy_o7_path is not None else None,
            "sha256": (
                sha256_file(legacy_o7_path) if legacy_o7_path is not None else None
            ),
            "reason": (
                "legacy O7 compared transform_pose=False decoder coordinates directly "
                "to source-GLB coordinates without the fixed decoder axis transform"
            ),
            "old_report_preserved": legacy_o7_path is not None,
        },
        "method_contracts": {
            "current_no_vggt": {
                "vggt_model_executed": False,
                "point_cloud_tensor_consumed": False,
                "target_or_metric_consumed": False,
                "explicit_camera_pose_consumed": True,
            },
            "reconviagen_original": {
                "vggt_model_executed": True,
                "point_cloud_tensor_consumed": False,
                "target_or_metric_consumed": False,
                "explicit_camera_pose_consumed": False,
                "input": "same frozen RGB/mask views; official 1.10 crop/518 resize",
            },
        },
        "training_object_disjoint": True,
        "source_mesh_disjoint": True,
        "summary": {method: aggregate(by_method[method]) for method in METHODS},
        "summary_by_source_group": {
            source: {
                method: aggregate(method_rows[method]) for method in METHODS
            }
            for source, method_rows in sorted(by_source.items())
        },
        "summary_by_view_count": {
            str(views): {
                method: aggregate(method_rows[method]) for method in METHODS
            }
            for views, method_rows in sorted(by_views.items())
        },
        "paired_comparison": comparison,
        "records": records,
        "failures": failures,
        "limitations": [
            "This is a 16-object diagnostic subset, not a formal benchmark.",
            "Inputs are synthetic Objaverse renders with known masks.",
            "ReconViaGen infers multi-view geometry with VGGT and does not consume explicit cameras.",
            "The upstream frozen view set was prepared using repaired target occupancy.",
        ],
    }
    report_path = output_dir / "report.json"
    atomic_json(report_path, report)
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "formal": report["formal"],
                "objects": report["object_count"],
                "records": report["record_count"],
                "pairs": report["pair_count"],
                "report": str(report_path),
            },
            indent=2,
        )
    )
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
