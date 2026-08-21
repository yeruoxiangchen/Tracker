#!/usr/bin/env python3
"""Strict-coordinate Objaverse2K dev64 comparison against ReconViaGen original."""

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
from pose_point_depth_mv.mesh_benchmark_metrics import (
    mesh_structure_metrics,
    surface_metrics,
)
from pose_point_depth_mv.objaverse2k_slat_pipeline import (
    resolve_native_objaverse_normalization_bindings,
)
from pose_point_depth_mv.omni_real_benchmark_common import sha256_file
from pose_point_depth_mv.render_direct_slat_fourway import LATENT_DECODER_TO_REFERENCE
from pose_point_depth_mv.training_overlap_objaverse import (
    OBJAVERSE2K_DEV64_SCOPE,
    expected_view_count,
    validate_selection,
)


REPORT_FORMAT = "pose_point_depth_mv.objaverse2k_dev64_vs_reconviagen_strict.v1"
RECORD_FORMAT = "pose_point_depth_mv.objaverse2k_dev64_strict_metric_record.v1"
METHODS = ("objaverse2k_slat", "reconviagen_original")
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
    digest = hashlib.sha256(f"objaverse2k_dev64\0{uid}\0{seed}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


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


def bootstrap_mean_ci(values: list[float], *, samples: int, seed: int) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, len(array), size=(int(samples), len(array)))
    means = array[indices].mean(axis=1)
    return [float(value) for value in np.quantile(means, [0.025, 0.975])]


def decoder_to_source_mesh(path: Path):
    mesh = load_mesh(path)
    mesh.apply_transform(LATENT_DECODER_TO_REFERENCE)
    return mesh


def load_recon_manifests(
    paths: list[str], *, selection_sha: str, expected_seeds: list[int]
) -> tuple[list[dict[str, Any]], dict[tuple[str, int], dict[str, Any]]]:
    if len(paths) != 8:
        raise ValueError("exactly eight ReconViaGen worker manifests are required")
    bindings = []
    records: dict[tuple[str, int], dict[str, Any]] = {}
    workers: set[tuple[int, int]] = set()
    for value in paths:
        path = Path(value).expanduser().resolve()
        manifest = load_json(path)
        required = {
            "format": RECON_MANIFEST_FORMAT,
            "passed": True,
            "formal": False,
            "protocol_scope": OBJAVERSE2K_DEV64_SCOPE,
            "training_overlap": False,
            "training_object_disjoint": True,
            "source_mesh_disjoint": True,
            "method": "reconviagen_original",
            "selection_manifest_sha256": selection_sha,
            "seeds": expected_seeds,
        }
        mismatch = {
            key: (manifest.get(key), expected)
            for key, expected in required.items()
            if manifest.get(key) != expected
        }
        if mismatch:
            raise RuntimeError(f"ReconViaGen dev64 worker contract differs: {mismatch}")
        worker = (int(manifest["worker_index"]), int(manifest["num_workers"]))
        if worker in workers:
            raise RuntimeError(f"duplicate ReconViaGen worker={worker}")
        workers.add(worker)
        for row in manifest.get("objects", []):
            key = (str(row["object_uid"]), int(row["seed"]))
            if key in records:
                raise RuntimeError(f"duplicate ReconViaGen result={key}")
            mesh_path = Path(row["mesh"]).resolve()
            if sha256_file(mesh_path) != row.get("mesh_sha256"):
                raise RuntimeError(f"ReconViaGen mesh changed: {mesh_path}")
            records[key] = row
        bindings.append(
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "worker_index": worker[0],
                "num_workers": worker[1],
                "object_count": int(manifest["object_count"]),
                "record_count": int(manifest["record_count"]),
            }
        )
    if workers != {(index, 8) for index in range(8)}:
        raise RuntimeError("ReconViaGen dev64 worker set is incomplete")
    return sorted(bindings, key=lambda row: row["worker_index"]), records


def object_means(
    records: list[dict[str, Any]], *, objects: list[str]
) -> dict[str, dict[str, dict[str, float]]]:
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = {
        method: defaultdict(list) for method in METHODS
    }
    for row in records:
        grouped[str(row["method"])][str(row["object_uid"])].append(row)
    output: dict[str, dict[str, dict[str, float]]] = {method: {} for method in METHODS}
    for method in METHODS:
        if set(grouped[method]) != set(objects):
            raise RuntimeError(f"{method} object set differs")
        for object_uid in objects:
            rows = grouped[method][object_uid]
            if sorted(int(row["seed"]) for row in rows) != [42, 43, 44]:
                raise RuntimeError(f"{method} seed set differs for {object_uid}")
            names = sorted(rows[0]["surface"])
            output[method][object_uid] = {
                name: float(np.mean([float(row["surface"][name]) for row in rows]))
                for name in names
            }
    return output


def paired_comparison(
    means: dict[str, dict[str, dict[str, float]]],
    *,
    bootstrap_samples: int,
) -> dict[str, Any]:
    left = means["objaverse2k_slat"]
    right = means["reconviagen_original"]
    objects = sorted(left)
    if set(left) != set(right):
        raise RuntimeError("paired object sets differ")
    deltas_by_metric: dict[str, list[float]] = defaultdict(list)
    per_object: dict[str, dict[str, float]] = {}
    for object_uid in objects:
        per_object[object_uid] = {}
        for metric in sorted(left[object_uid]):
            delta = (
                right[object_uid][metric] - left[object_uid][metric]
                if metric in LOWER_IS_BETTER
                else left[object_uid][metric] - right[object_uid][metric]
            )
            deltas_by_metric[metric].append(float(delta))
            per_object[object_uid][metric] = float(delta)
    metric_deltas = {}
    for position, (metric, values) in enumerate(sorted(deltas_by_metric.items())):
        metric_deltas[f"{metric}_objaverse2k_slat_improvement"] = {
            **summarize(values),
            "object_bootstrap_95ci": bootstrap_mean_ci(
                values,
                samples=int(bootstrap_samples),
                seed=20260813 + position,
            ),
        }
    chamfer = deltas_by_metric["chamfer_l1"]
    return {
        "left": "objaverse2k_slat",
        "right": "reconviagen_original",
        "unit_of_analysis": "64 objects; each object metric is the mean of seeds 42/43/44",
        "positive_definition": "positive means objaverse2k_slat is better",
        "metric_deltas": metric_deltas,
        "chamfer_l1_wins": {
            "objaverse2k_slat": sum(value > 0.0 for value in chamfer),
            "reconviagen_original": sum(value < 0.0 for value in chamfer),
            "ties": sum(value == 0.0 for value in chamfer),
        },
        "per_object_improvements": per_object,
    }


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection_manifest", required=True)
    parser.add_argument("--reconviagen_manifest", action="append", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--surface_samples", type=int, default=20000)
    parser.add_argument(
        "--fscore_thresholds", type=parse_float_csv, default=[0.01, 0.02, 0.05]
    )
    parser.add_argument("--bootstrap_samples", type=int, default=10000)
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> None:
    args = make_parser().parse_args()
    if int(args.surface_samples) <= 0 or int(args.bootstrap_samples) <= 0:
        raise ValueError("surface_samples/bootstrap_samples must be positive")
    selection_path = Path(args.selection_manifest).expanduser().resolve()
    selection = load_json(selection_path)
    contract = validate_selection(selection)
    if (
        contract.scope != OBJAVERSE2K_DEV64_SCOPE
        or contract.training_object_disjoint is not True
        or contract.source_mesh_disjoint is not True
    ):
        raise RuntimeError("selection is not object-disjoint Objaverse2K dev64")
    protocol = selection["objaverse2k_dev64_protocol"]
    seeds = [int(value) for value in protocol["joint_seeds"]]
    if seeds != [42, 43, 44]:
        raise RuntimeError("Objaverse2K dev64 seed contract differs")
    selection_sha = sha256_file(selection_path)
    recon_bindings, recon_rows = load_recon_manifests(
        args.reconviagen_manifest,
        selection_sha=selection_sha,
        expected_seeds=seeds,
    )
    samples = list(selection["samples"])
    objects = [str(row["object_uid"]) for row in samples]
    if set(recon_rows) != {(object_uid, seed) for object_uid in objects for seed in seeds}:
        raise RuntimeError("selection and ReconViaGen result matrix differ")
    slat_path = Path(protocol["source_slat_manifest"]).resolve()
    if sha256_file(slat_path) != protocol["source_slat_manifest_sha256"]:
        raise RuntimeError("dev64 SLat manifest changed")
    normalization_bindings = resolve_native_objaverse_normalization_bindings(
        slat_path, load_json(slat_path), samples
    )
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for position, sample in enumerate(samples, start=1):
        uid = str(sample["uid"])
        object_uid = str(sample["object_uid"])
        view_count = expected_view_count(sample)
        latent_path = Path(sample["ss_latent"]).resolve()
        target_mesh, target_metadata = load_canonical_gt(
            sample,
            canonical_margin_binding=normalization_bindings.get(str(latent_path)),
        )
        target_structure = mesh_structure_metrics(target_mesh)
        target_path = output_dir / "targets" / object_uid / "mesh_source_canonical.obj"
        if not target_path.is_file():
            export_obj_atomic(target_mesh, target_path)
        for seed in seeds:
            native = sample["native_objaverse2k_meshes"][str(seed)]
            recon = recon_rows[(object_uid, seed)]
            if (
                str(recon["uid"]) != uid
                or int(recon["view_count"]) != view_count
                or int(recon["seed"]) != seed
            ):
                raise RuntimeError(f"ReconViaGen paired identity differs: {uid} seed={seed}")
            for method, prediction_path_value, expected_sha in (
                ("objaverse2k_slat", native["mesh"], native["mesh_sha256"]),
                ("reconviagen_original", recon["mesh"], recon["mesh_sha256"]),
            ):
                prediction_path = Path(prediction_path_value).resolve()
                prediction_sha = sha256_file(prediction_path)
                if prediction_sha != expected_sha:
                    raise RuntimeError(f"prediction changed: {prediction_path}")
                metric_seed = stable_metric_seed(uid, seed)
                identity = {
                    "format": RECORD_FORMAT,
                    "method": method,
                    "uid": uid,
                    "object_uid": object_uid,
                    "seed": seed,
                    "prediction_sha256": prediction_sha,
                    "target_source_sha256": str(target_metadata["source_glb_sha256"]),
                    "metric_seed": metric_seed,
                    "surface_samples": int(args.surface_samples),
                    "fscore_thresholds": list(args.fscore_thresholds),
                    "decoder_to_source_axis_transform": LATENT_DECODER_TO_REFERENCE.tolist(),
                }
                record_path = (
                    output_dir / "records" / object_uid / f"{method}_seed{seed}.json"
                )
                if record_path.is_file():
                    if not args.resume:
                        raise FileExistsError(record_path)
                    record = load_json(record_path)
                    if record.get("passed") is not True or any(
                        record.get(key) != value for key, value in identity.items()
                    ):
                        raise RuntimeError(f"stale metric record: {record_path}")
                    records.append(record)
                    continue
                predicted_mesh = decoder_to_source_mesh(prediction_path)
                structure = mesh_structure_metrics(predicted_mesh)
                if not structure["mesh_success"]:
                    raise RuntimeError(f"empty prediction: {prediction_path}")
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
        print(f"[objaverse2k_dev64_strict] {position}/64 uid={uid}", flush=True)

    if len(records) != 64 * 3 * 2:
        raise RuntimeError("strict dev64 metric matrix is incomplete")
    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_method[str(record["method"])].append(record)
    means = object_means(records, objects=objects)
    object_mean_summary = {
        method: {
            metric: summarize(
                means[method][object_uid][metric] for object_uid in objects
            )
            for metric in sorted(next(iter(means[method].values())))
        }
        for method in METHODS
    }
    comparison = paired_comparison(means, bootstrap_samples=int(args.bootstrap_samples))
    report = {
        "format": REPORT_FORMAT,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "passed": True,
        "formal": False,
        "post_selection_development_diagnostic": True,
        "training_overlap": False,
        "training_object_disjoint": True,
        "source_mesh_disjoint": True,
        "protocol_scope": OBJAVERSE2K_DEV64_SCOPE,
        "methods": list(METHODS),
        "object_count": 64,
        "seed_count": 3,
        "record_count": len(records),
        "pair_count": 64,
        "seeds": seeds,
        "selection_manifest": str(selection_path),
        "selection_manifest_sha256": selection_sha,
        "native_binding": protocol["native_binding"],
        "reconviagen_worker_manifests": recon_bindings,
        "surface_samples": int(args.surface_samples),
        "fscore_thresholds": list(args.fscore_thresholds),
        "coordinate_evaluation": {
            "decoder_to_source_axis_transform": LATENT_DECODER_TO_REFERENCE.tolist(),
            "mapping": "(x,y,z) -> (x,z,-y)",
            "applied_identically_to_both_methods": True,
            "alignment": "fixed proper axis transform only; no ICP/scale/GT fit",
        },
        "method_contracts": {
            "objaverse2k_slat": {
                "native_ss": "Mixed Native SS step2000 EMA frozen upstream",
                "native_slat": "Objaverse2K step2000 EMA",
                "explicit_camera_pose_consumed": True,
                "vggt_model_executed": False,
                "target_or_metric_consumed_during_inference": False,
            },
            "reconviagen_original": {
                "official_pipeline": "VGGT + Stock SS + Stock SLat + Stock decoder",
                "explicit_camera_pose_consumed": False,
                "vggt_model_executed": True,
                "target_or_metric_consumed_during_inference": False,
                "input": "same frozen dev64 RGB/mask views; official 1.10 crop/518 resize",
            },
        },
        "seed_level_summary": {method: aggregate(by_method[method]) for method in METHODS},
        "object_mean_summary": object_mean_summary,
        "paired_comparison": comparison,
        "records": records,
        "limitations": [
            "dev64 was already used for Objaverse2K checkpoint diagnostics; formal=false.",
            "This is object-disjoint from Objaverse2K training, but not an untouched final test.",
            "Inputs are synthetic Objaverse renders with known masks.",
            "This is a full-system comparison, not an isolated SLat-module ablation.",
        ],
    }
    report_path = output_dir / "report.json"
    atomic_json(report_path, report)
    print(
        json.dumps(
            {
                "passed": True,
                "formal": False,
                "training_object_disjoint": True,
                "objects": 64,
                "pairs": 64,
                "seeds": seeds,
                "report": str(report_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
