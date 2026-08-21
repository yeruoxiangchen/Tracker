#!/usr/bin/env python3
"""Pair a training-overlap Native model with ReconViaGen original geometrically."""

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

from pose_point_depth_mv.evaluate_objaverse16_no_vggt import (
    aggregate,
    export_obj_atomic,
    load_mesh,
    parse_float_csv,
)
from pose_point_depth_mv.export_direct_flow_mesh_pairs import load_canonical_gt
from pose_point_depth_mv.infer_objaverse16_reconviagen import (
    MANIFEST_FORMAT as RECON_MANIFEST_FORMAT,
)
from pose_point_depth_mv.infer_objaverse_training_overlap_native import (
    MANIFEST_FORMAT as NATIVE_MANIFEST_FORMAT,
    MODEL_LABELS,
)
from pose_point_depth_mv.mesh_benchmark_metrics import (
    mesh_structure_metrics,
    surface_metrics,
)
from pose_point_depth_mv.objaverse2k_slat_pipeline import (
    resolve_native_objaverse_normalization_bindings,
)
from pose_point_depth_mv.omni_real_benchmark_common import sha256_file
from pose_point_depth_mv.render_direct_slat_fourway import (
    LATENT_DECODER_TO_REFERENCE,
)
from pose_point_depth_mv.training_overlap_objaverse import (
    TRAINING_OVERLAP_SCOPE,
    expected_view_count,
    validate_selection,
)


REPORT_FORMAT = "pose_point_depth_mv.objaverse_training_overlap_evaluation.v1"
RECORD_FORMAT = "pose_point_depth_mv.objaverse_training_overlap_metric_record.v1"
LOWER_IS_BETTER = {
    "pred_to_gt_mean",
    "gt_to_pred_mean",
    "chamfer_l1",
    "chamfer_l2",
}


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"JSON root is not an object: {path}")
    return payload


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def stable_metric_seed(uid: str, seed: int) -> int:
    digest = hashlib.sha256(
        f"objaverse_training_overlap\0{uid}\0{int(seed)}".encode()
    ).digest()
    return int.from_bytes(digest[:4], "big")


def summarize(values: Iterable[float]) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=np.float64)
    if not len(array) or not np.isfinite(array).all():
        raise ValueError("paired metric distribution is empty or non-finite")
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
    native: dict[str, dict[str, Any]],
    recon: dict[str, dict[str, Any]],
    *,
    native_label: str,
) -> dict[str, Any]:
    if not native or set(native) != set(recon):
        raise RuntimeError("paired Native/ReconViaGen object sets differ")
    metric_names = sorted(native[next(iter(native))]["surface"])
    distributions: dict[str, Any] = {}
    for name in metric_names:
        deltas = []
        for object_uid in sorted(native):
            left = float(native[object_uid]["surface"][name])
            right = float(recon[object_uid]["surface"][name])
            deltas.append(right - left if name in LOWER_IS_BETTER else left - right)
        distributions[f"{name}_{native_label}_improvement"] = summarize(deltas)
    chamfer = {
        object_uid: float(recon[object_uid]["surface"]["chamfer_l1"])
        - float(native[object_uid]["surface"]["chamfer_l1"])
        for object_uid in native
    }
    return {
        "left": native_label,
        "right": "reconviagen_original",
        "positive_definition": f"positive means {native_label} is better",
        "metric_deltas": distributions,
        "chamfer_l1_wins": {
            native_label: sum(value > 0.0 for value in chamfer.values()),
            "reconviagen_original": sum(value < 0.0 for value in chamfer.values()),
            "ties": sum(value == 0.0 for value in chamfer.values()),
        },
        f"per_object_chamfer_l1_{native_label}_improvement": chamfer,
    }


def _load_worker_manifests(
    paths: list[str],
    *,
    expected_format: str,
    expected_method: str,
    selection_sha: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not paths:
        raise ValueError(f"no {expected_method} worker manifests were provided")
    manifests: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    seen_workers: set[tuple[int, int]] = set()
    seen_objects: set[str] = set()
    for value in paths:
        path = Path(value).expanduser().resolve()
        manifest = load_json(path)
        required = {
            "format": expected_format,
            "method": expected_method,
            "protocol_scope": TRAINING_OVERLAP_SCOPE,
            "formal": False,
            "training_overlap": True,
            "training_object_disjoint": False,
            "source_mesh_disjoint": False,
            "passed": True,
        }
        mismatch = {
            key: (manifest.get(key), expected)
            for key, expected in required.items()
            if manifest.get(key) != expected
        }
        if mismatch:
            raise RuntimeError(f"{expected_method} worker contract differs: {mismatch}")
        worker = (int(manifest["worker_index"]), int(manifest["num_workers"]))
        if worker in seen_workers:
            raise RuntimeError(f"duplicate {expected_method} worker shard={worker}")
        seen_workers.add(worker)
        if expected_method == "reconviagen_original":
            if manifest.get("selection_manifest_sha256") != selection_sha:
                raise RuntimeError("ReconViaGen worker selection binding differs")
            object_key = "object_uid"
        else:
            lineage = dict(manifest.get("training_lineage", {}))
            if lineage.get("selection_manifest_sha256") != selection_sha:
                raise RuntimeError("Native worker selection binding differs")
            object_key = "object_id"
        for row in manifest.get("objects", []):
            object_uid = str(row[object_key])
            if object_uid in seen_objects:
                raise RuntimeError(f"duplicate {expected_method} object={object_uid}")
            seen_objects.add(object_uid)
            records.append(row)
        manifests.append(
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "worker_index": worker[0],
                "num_workers": worker[1],
                "object_count": int(manifest["object_count"]),
            }
        )
    counts = {workers for _, workers in seen_workers}
    if len(counts) != 1 or len(seen_workers) != next(iter(counts)):
        raise RuntimeError(f"{expected_method} worker shard set is incomplete")
    return manifests, records


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection_manifest", required=True)
    parser.add_argument("--native_label", choices=MODEL_LABELS, required=True)
    parser.add_argument("--native_manifest", action="append", required=True)
    parser.add_argument("--reconviagen_manifest", action="append", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--surface_samples", type=int, default=20000)
    parser.add_argument(
        "--fscore_thresholds", type=parse_float_csv, default=[0.01, 0.02, 0.05]
    )
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> None:
    args = make_parser().parse_args()
    if int(args.surface_samples) <= 0:
        raise ValueError("surface_samples must be positive")
    selection_path = Path(args.selection_manifest).expanduser().resolve()
    selection = load_json(selection_path)
    contract = validate_selection(selection)
    expected_source = (
        "objaverse2k_train" if args.native_label == "objaverse2k_slat" else "mixed_objaverse_train"
    )
    if contract.source_scope != expected_source:
        raise RuntimeError("Native label and selection source scope differ")
    selection_sha = sha256_file(selection_path)
    native_bindings, native_rows = _load_worker_manifests(
        args.native_manifest,
        expected_format=NATIVE_MANIFEST_FORMAT,
        expected_method=str(args.native_label),
        selection_sha=selection_sha,
    )
    recon_bindings, recon_rows = _load_worker_manifests(
        args.reconviagen_manifest,
        expected_format=RECON_MANIFEST_FORMAT,
        expected_method="reconviagen_original",
        selection_sha=selection_sha,
    )
    native_by_object = {str(row["object_id"]): row for row in native_rows}
    recon_by_object = {str(row["object_uid"]): row for row in recon_rows}
    selected = list(selection["samples"])
    selected_objects = {str(row["object_uid"]) for row in selected}
    if (
        set(native_by_object) != selected_objects
        or set(recon_by_object) != selected_objects
        or len(native_by_object) != contract.object_count
    ):
        raise RuntimeError("selection/Native/ReconViaGen object sets differ")

    source_slat_path = Path(
        str(selection["training_overlap_protocol"]["source_slat_manifest"])
    ).resolve()
    source_slat = load_json(source_slat_path)
    normalization_bindings = resolve_native_objaverse_normalization_bindings(
        source_slat_path, source_slat, selected
    )
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for position, selected_row in enumerate(selected, start=1):
        uid = str(selected_row["uid"])
        object_uid = str(selected_row["object_uid"])
        view_count = expected_view_count(selected_row)
        native_row = native_by_object[object_uid]
        recon_row = recon_by_object[object_uid]
        if (
            int(native_row["seed"]) != 42
            or int(recon_row["seed"]) != 42
            or str(recon_row["uid"]) != uid
            or int(recon_row["view_count"]) != view_count
        ):
            raise RuntimeError(f"paired identity/view contract differs: {uid}")
        latent_path = Path(str(selected_row["ss_latent"])).resolve()
        target_mesh, target_metadata = load_canonical_gt(
            selected_row,
            canonical_margin_binding=normalization_bindings.get(str(latent_path)),
        )
        target_structure = mesh_structure_metrics(target_mesh)
        target_path = output_dir / "targets" / object_uid / "mesh_source_canonical.obj"
        if not target_path.is_file():
            export_obj_atomic(target_mesh, target_path)
        metric_seed = stable_metric_seed(uid, 42)
        for method, prediction_row in (
            (str(args.native_label), native_row),
            ("reconviagen_original", recon_row),
        ):
            prediction_path = Path(str(prediction_row["mesh"])).resolve()
            prediction_sha = sha256_file(prediction_path)
            if prediction_sha != str(prediction_row["mesh_sha256"]):
                raise RuntimeError(f"prediction hash changed: {prediction_path}")
            identity = {
                "format": RECORD_FORMAT,
                "method": method,
                "uid": uid,
                "object_uid": object_uid,
                "seed": 42,
                "prediction_sha256": prediction_sha,
                "target_source_sha256": str(target_metadata["source_glb_sha256"]),
                "metric_seed": metric_seed,
                "surface_samples": int(args.surface_samples),
                "fscore_thresholds": list(args.fscore_thresholds),
                "decoder_to_source_axis_transform": LATENT_DECODER_TO_REFERENCE.tolist(),
            }
            record_path = output_dir / "records" / object_uid / f"{method}_seed42.json"
            if record_path.is_file():
                if not args.resume:
                    raise FileExistsError(record_path)
                record = load_json(record_path)
                if record.get("passed") is not True or any(
                    record.get(key) != value for key, value in identity.items()
                ):
                    raise RuntimeError(f"stale evaluation record: {record_path}")
                records.append(record)
                continue
            predicted_mesh = decoder_to_source_mesh(prediction_path)
            structure = mesh_structure_metrics(predicted_mesh)
            if not structure["mesh_success"]:
                raise RuntimeError(f"empty prediction after fixed axis mapping: {prediction_path}")
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
                "source_scope": contract.source_scope,
                "view_count": view_count,
                "prediction": str(prediction_path),
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
        print(f"[training_overlap_eval] {position}/{len(selected)} uid={uid}", flush=True)

    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    index: dict[str, dict[str, dict[str, Any]]] = {
        str(args.native_label): {},
        "reconviagen_original": {},
    }
    for record in records:
        method = str(record["method"])
        by_method[method].append(record)
        index[method][str(record["object_uid"])] = record
    comparison = paired_comparison(
        index[str(args.native_label)],
        index["reconviagen_original"],
        native_label=str(args.native_label),
    )
    report = {
        "format": REPORT_FORMAT,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "passed": len(records) == contract.object_count * 2,
        "formal": False,
        "training_overlap": True,
        "protocol_scope": TRAINING_OVERLAP_SCOPE,
        "source_scope": contract.source_scope,
        "methods": [str(args.native_label), "reconviagen_original"],
        "object_count": contract.object_count,
        "record_count": len(records),
        "pair_count": len(index[str(args.native_label)]),
        "seeds": [42],
        "selection_manifest": str(selection_path),
        "selection_manifest_sha256": selection_sha,
        "native_worker_manifests": native_bindings,
        "reconviagen_worker_manifests": recon_bindings,
        "surface_samples": int(args.surface_samples),
        "fscore_thresholds": list(args.fscore_thresholds),
        "coordinate_evaluation": {
            "decoder_to_source_axis_transform": LATENT_DECODER_TO_REFERENCE.tolist(),
            "mapping": "(x,y,z) -> (x,z,-y)",
            "applied_identically_to_both_methods": True,
            "alignment": "fixed proper axis transform only; no ICP/scale/GT fit",
        },
        "training_object_disjoint": False,
        "source_mesh_disjoint": False,
        "method_contracts": {
            str(args.native_label): {
                "native_ss": "shared Mixed Native SS step2000 EMA",
                "explicit_camera_pose_consumed": True,
                "vggt_model_executed": False,
                "point_cloud_tensor_consumed": False,
                "target_or_metric_consumed_during_inference": False,
            },
            "reconviagen_original": {
                "official_pipeline": "VGGT + Stock SS + Stock SLat + Stock decoder",
                "explicit_camera_pose_consumed": False,
                "vggt_model_executed": True,
                "point_cloud_tensor_consumed": False,
                "target_or_metric_consumed_during_inference": False,
                "input": "same selected RGB/mask views; official 1.10 crop/518 resize",
            },
        },
        "summary": {
            method: aggregate(by_method[method])
            for method in (str(args.native_label), "reconviagen_original")
        },
        "paired_comparison": comparison,
        "records": records,
        "limitations": [
            "All selected objects overlap the evaluated Native model training set.",
            "This diagnoses fitting or memorization and is not a generalization benchmark.",
            "The two source scopes are separate object sets and must not be pooled.",
            "Inputs are synthetic Objaverse renders with known masks.",
            "This is a full-system comparison, not an isolated SLat-module ablation.",
        ],
    }
    report_path = output_dir / "report.json"
    atomic_json(report_path, report)
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "formal": False,
                "training_overlap": True,
                "objects": report["object_count"],
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
