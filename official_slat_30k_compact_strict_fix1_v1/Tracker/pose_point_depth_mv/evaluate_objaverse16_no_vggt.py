#!/usr/bin/env python3
"""Evaluate frozen Objaverse16 predictions against canonical source Meshes."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import trimesh

from pose_point_depth_mv.export_direct_flow_mesh_pairs import load_canonical_gt
from pose_point_depth_mv.freeze_objaverse16_test import PROTOCOL_FORMAT, sha256_file
from pose_point_depth_mv.infer_objaverse16_no_vggt_mixed import (
    MANIFEST_FORMAT as INFERENCE_MANIFEST_FORMAT,
)
from pose_point_depth_mv.mesh_benchmark_metrics import (
    mesh_structure_metrics,
    surface_metrics,
)
from pose_point_depth_mv.prepare_objaverse16_no_vggt_model_inputs import (
    MANIFEST_FORMAT as MODEL_INPUT_MANIFEST_FORMAT,
)
from pose_point_depth_mv.render_direct_slat_fourway import (
    LATENT_DECODER_TO_REFERENCE,
)


REPORT_FORMAT = "pose_point_depth_mv.objaverse16_no_vggt_mesh_evaluation.v2"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def parse_float_csv(value: str) -> list[float]:
    result = [float(item.strip()) for item in str(value).split(",") if item.strip()]
    if not result or len(result) != len(set(result)) or min(result) <= 0.0:
        raise argparse.ArgumentTypeError("thresholds must be unique positive floats")
    return result


def stable_metric_seed(uid: str, seed: int) -> int:
    digest = hashlib.sha256(f"objaverse16\0{uid}\0{int(seed)}".encode("utf-8"))
    return int.from_bytes(digest.digest()[:4], "big")


def load_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, force="scene", process=False)
    if isinstance(loaded, trimesh.Scene):
        parts = [
            item
            for item in loaded.dump(concatenate=False)
            if isinstance(item, trimesh.Trimesh) and len(item.vertices) and len(item.faces)
        ]
        if not parts:
            raise ValueError(f"Mesh scene is empty: {path}")
        return trimesh.util.concatenate(parts)
    if isinstance(loaded, trimesh.Trimesh) and len(loaded.vertices) and len(loaded.faces):
        return loaded
    raise ValueError(f"Mesh is empty or unsupported: {path}")


def export_obj_atomic(mesh: trimesh.Trimesh, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    mesh.export(temporary, file_type="obj")
    os.replace(temporary, path)


def summarize(values: Iterable[float]) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=np.float64)
    finite = array[np.isfinite(array)]
    if not len(finite):
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "count": int(len(finite)),
        "mean": float(finite.mean()),
        "median": float(np.median(finite)),
        "min": float(finite.min()),
        "max": float(finite.max()),
    }


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    surface_names = sorted(
        {
            name
            for record in records
            for name, value in record["surface"].items()
            if isinstance(value, (int, float))
        }
    )
    structure_names = (
        "largest_component_ratio",
        "component_count",
        "boundary_edge_count",
        "nonmanifold_edge_count",
        "bbox_diag",
    )
    return {
        "surface": {
            name: summarize(float(record["surface"][name]) for record in records)
            for name in surface_names
        },
        "structure": {
            name: summarize(
                float(record["structure"][name])
                for record in records
                if name in record["structure"]
            )
            for name in structure_names
        },
        "mesh_success_rate": float(
            np.mean([bool(record["structure"]["mesh_success"]) for record in records])
        ),
        "record_count": len(records),
        "object_count": len({record["object_uid"] for record in records}),
    }


def record_identity(
    *,
    uid: str,
    object_uid: str,
    seed: int,
    prediction_sha256: str,
    latent_sha256: str,
    target_source_sha256: str,
    surface_samples: int,
    thresholds: list[float],
) -> dict[str, Any]:
    return {
        "uid": uid,
        "object_uid": object_uid,
        "seed": int(seed),
        "prediction_sha256": prediction_sha256,
        "latent_sha256": latent_sha256,
        "target_source_sha256": target_source_sha256,
        "surface_samples": int(surface_samples),
        "fscore_thresholds": list(thresholds),
    }


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection_manifest", required=True)
    parser.add_argument("--lifting_manifest", required=True)
    parser.add_argument("--inference_manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--surface_samples", type=int, default=20000)
    parser.add_argument(
        "--fscore_thresholds", type=parse_float_csv, default=[0.01, 0.02, 0.05]
    )
    parser.add_argument("--no_export_targets", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> None:
    args = make_parser().parse_args()
    if int(args.surface_samples) <= 0:
        raise ValueError("surface_samples must be positive")
    selection_path = Path(args.selection_manifest).expanduser().resolve()
    lifting_path = Path(args.lifting_manifest).expanduser().resolve()
    inference_path = Path(args.inference_manifest).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    selection = load_json(selection_path)
    lifting = load_json(lifting_path)
    inference = load_json(inference_path)
    protocol = dict(selection.get("objaverse16_protocol", {}))
    if (
        protocol.get("format") != PROTOCOL_FORMAT
        or protocol.get("passed") is not True
        or protocol.get("object_count") != 16
    ):
        raise RuntimeError("selection is not a passed frozen Objaverse test16")
    inference_guards = {
        "format": INFERENCE_MANIFEST_FORMAT,
        "protocol_scope": "frozen_objaverse_test16",
        "formal": False,
        "vggt_model_loaded": False,
        "vggt_model_executed": False,
        "point_cloud_tensor_present": False,
        "point_cloud_consumed": False,
        "target_or_metric_consumed": False,
        "passed": True,
    }
    mismatch = {
        key: (inference.get(key), expected)
        for key, expected in inference_guards.items()
        if inference.get(key) != expected
    }
    if mismatch:
        raise RuntimeError(f"inference scope/identity guard differs: {mismatch}")
    if int(inference.get("object_count", -1)) != 16:
        raise RuntimeError("inference did not contain exactly 16 objects")

    model_input_path = Path(inference["model_input_manifest"]).resolve()
    model_inputs = load_json(model_input_path)
    if (
        model_inputs.get("format") != MODEL_INPUT_MANIFEST_FORMAT
        or model_inputs.get("selection_manifest_sha256") != sha256_file(selection_path)
        or model_inputs.get("lifting_manifest_sha256") != sha256_file(lifting_path)
        or model_inputs.get("target_or_mesh_consumed") is not False
    ):
        raise RuntimeError("model-input manifest is not bound to this target-free test")
    input_by_object = {
        str(row["object_uid"]): row for row in model_inputs.get("objects", [])
    }
    lifting_by_uid = {str(row["uid"]): row for row in lifting.get("samples", [])}
    predictions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in inference.get("objects", []):
        predictions[str(row["object_id"])].append(row)
    selected = list(selection["samples"])
    selected_objects = {str(row["object_uid"]) for row in selected}
    if set(predictions) != selected_objects or set(input_by_object) != selected_objects:
        raise RuntimeError("selection/model-input/inference object sets differ")
    expected_seeds = sorted(int(value) for value in inference["seeds"])
    if not expected_seeds:
        raise RuntimeError("inference manifest has no seeds")

    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    actual_view_histogram: dict[int, int] = defaultdict(int)
    for position, selected_row in enumerate(selected, start=1):
        uid = str(selected_row["uid"])
        object_uid = str(selected_row["object_uid"])
        input_row = input_by_object[object_uid]
        if str(input_row["uid"]) != uid:
            raise RuntimeError(f"object={object_uid} selected/model-input uid differs")
        expected_views = int(
            selected_row["objaverse16_selection"]["expected_point_prior_view_count"]
        )
        actual_views = int(input_row["view_count"])
        if actual_views != expected_views:
            raise RuntimeError(
                f"uid={uid} point-prior view replay differs: {actual_views}/{expected_views}"
            )
        actual_view_histogram[actual_views] += 1
        lifting_row = lifting_by_uid.get(uid)
        if lifting_row is None or str(lifting_row.get("object_uid")) != object_uid:
            raise RuntimeError(f"uid={uid} lifting target identity differs")
        latent_path = Path(str(lifting_row["ss_latent"])).resolve()
        if latent_path != Path(str(selected_row["ss_latent"])).resolve():
            raise RuntimeError(f"uid={uid} selection/lifting SS latent differs")
        latent_sha = sha256_file(latent_path)
        target_mesh, target_metadata = load_canonical_gt(
            {
                "ss_latent": str(latent_path),
                "source_glb": selected_row["source_glb"],
            }
        )
        target_structure = mesh_structure_metrics(target_mesh)
        target_path = output_dir / "targets" / object_uid / "mesh_canonical.obj"
        if not args.no_export_targets:
            export_obj_atomic(target_mesh, target_path)
        prediction_rows = sorted(predictions[object_uid], key=lambda row: int(row["seed"]))
        if [int(row["seed"]) for row in prediction_rows] != expected_seeds:
            raise RuntimeError(f"object={object_uid} inference seed set differs")
        for prediction in prediction_rows:
            seed = int(prediction["seed"])
            record_path = output_dir / "records" / object_uid / f"seed_{seed}.json"
            prediction_path = Path(prediction["mesh"]).resolve()
            prediction_sha = sha256_file(prediction_path)
            if prediction_sha != str(prediction["mesh_sha256"]):
                raise RuntimeError(f"prediction Mesh SHA differs: {prediction_path}")
            identity = record_identity(
                uid=uid,
                object_uid=object_uid,
                seed=seed,
                prediction_sha256=prediction_sha,
                latent_sha256=latent_sha,
                target_source_sha256=str(target_metadata["source_glb_sha256"]),
                surface_samples=int(args.surface_samples),
                thresholds=list(args.fscore_thresholds),
            )
            if record_path.is_file():
                if not args.resume:
                    raise FileExistsError(record_path)
                record = load_json(record_path)
                if (
                    record.get("format") != REPORT_FORMAT
                    or record.get("passed") is not True
                    or any(record.get(key) != value for key, value in identity.items())
                ):
                    raise RuntimeError(f"stale evaluation record: {record_path}")
                records.append(record)
                continue
            try:
                predicted_mesh = load_mesh(prediction_path)
                # transform_pose=False exports decoder coordinates. Objaverse
                # source GLBs use the view/source convention, so apply the
                # fixed proper axis conversion used by the established stock
                # ReconViaGen evaluator before measuring either method.
                predicted_mesh.apply_transform(LATENT_DECODER_TO_REFERENCE)
                structure = mesh_structure_metrics(predicted_mesh)
                if not structure["mesh_success"]:
                    raise RuntimeError("predicted Mesh is empty or non-finite")
                metric_seed = stable_metric_seed(uid, seed)
                surface = surface_metrics(
                    predicted_mesh,
                    target_mesh,
                    count=int(args.surface_samples),
                    seed=metric_seed,
                    thresholds=args.fscore_thresholds,
                )
                record = {
                    "format": REPORT_FORMAT,
                    "created_at_utc": datetime.now(timezone.utc).isoformat(),
                    **identity,
                    "source_group": str(selected_row["source_group"]),
                    "view_count": actual_views,
                    "prediction": str(prediction_path),
                    "target": target_metadata,
                    "target_canonical_mesh": (
                        str(target_path) if not args.no_export_targets else None
                    ),
                    "metric_seed": metric_seed,
                    "alignment": "fixed proper axis transform only; no ICP/scale/GT fit",
                    "decoder_to_source_axis_transform": (
                        LATENT_DECODER_TO_REFERENCE.tolist()
                    ),
                    "coordinate_frame": "normalized source-GLB frame",
                    "surface": surface,
                    "structure": structure,
                    "target_structure": target_structure,
                    "passed": True,
                }
                atomic_json(record_path, record)
                records.append(record)
            except Exception as error:
                failures.append(
                    {"uid": uid, "object_uid": object_uid, "seed": str(seed), "error": repr(error)}
                )
                raise
        print(
            f"[objaverse16_eval] {position}/{len(selected)} uid={uid} "
            f"views={actual_views}",
            flush=True,
        )

    expected_records = 16 * len(expected_seeds)
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_views: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_group[str(record["source_group"])].append(record)
        by_views[int(record["view_count"])].append(record)
    report = {
        "format": REPORT_FORMAT,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "passed": len(records) == expected_records and not failures,
        "formal": False,
        "protocol_scope": "frozen_objaverse_test16",
        "method": "native_no_vggt_mixed_objaverse16",
        "methods": ["native_no_vggt_mixed"],
        "external_baselines_included": False,
        "object_count": len({record["object_uid"] for record in records}),
        "record_count": len(records),
        "seeds": expected_seeds,
        "selection_manifest": str(selection_path),
        "selection_manifest_sha256": sha256_file(selection_path),
        "lifting_manifest": str(lifting_path),
        "lifting_manifest_sha256": sha256_file(lifting_path),
        "model_input_manifest": str(model_input_path),
        "model_input_manifest_sha256": sha256_file(model_input_path),
        "inference_manifest": str(inference_path),
        "inference_manifest_sha256": sha256_file(inference_path),
        "training_object_disjoint": protocol["training_object_disjoint"],
        "source_mesh_disjoint": protocol["source_mesh_disjoint"],
        "target_or_metric_consumed_during_inference": False,
        "point_cloud_tensor_consumed_during_inference": False,
        "vggt_executed_during_inference": False,
        "alignment": "fixed proper axis transform only; no ICP/scale/GT fit",
        "coordinate_evaluation": {
            "decoder_to_source_axis_transform": LATENT_DECODER_TO_REFERENCE.tolist(),
            "mapping": "(x,y,z) -> (x,z,-y)",
            "reason": (
                "transform_pose=False Meshes are in decoder coordinates while "
                "canonical Objaverse source GLBs are in the source/view convention"
            ),
        },
        "surface_samples": int(args.surface_samples),
        "fscore_thresholds": list(args.fscore_thresholds),
        "actual_view_histogram": {
            str(key): value for key, value in sorted(actual_view_histogram.items())
        },
        "summary": aggregate(records),
        "summary_by_source_group": {
            key: aggregate(value) for key, value in sorted(by_group.items())
        },
        "summary_by_view_count": {
            str(key): aggregate(value) for key, value in sorted(by_views.items())
        },
        "records": records,
        "failures": failures,
        "limitations": [
            "This is a 16-object diagnostic subset, not the formal Holdout64.",
            "Inputs are synthetic Objaverse renders with known canonical cameras.",
            "The upstream point-prior builder used repaired target occupancy to choose posed views, but point coordinates are absent from and not consumed by model inference.",
            "No ReconViaGen or Pixal3D baseline is included in this first Objaverse16 protocol.",
            "This v2 report supersedes v1, which omitted the fixed decoder-to-source axis transform.",
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
                "view_histogram": report["actual_view_histogram"],
                "report": str(report_path),
            },
            indent=2,
        ),
        flush=True,
    )
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
