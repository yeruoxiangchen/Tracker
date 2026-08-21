#!/usr/bin/env python3
"""Score strict ReconViaGen on the frozen Dora-Bench current-valid 299 subset."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from manual_mesh_reconstruction.common import (
    atomic_json,
    canonical_sha256,
    load_json,
    sha256_file,
)
from pose_point_depth_mv.evaluate_omni200_ss30k_slat30k import (
    _gt_mesh,
    _normalize_points,
    _one_mesh,
    _summary,
)
from pose_point_depth_mv.mesh_benchmark_metrics import deterministic_surface_sample


SUBSET_FORMAT = "reconviagen.dorabench_dora299_current_valid_subset.v1"
INFERENCE_FORMAT = "reconviagen.dorabench_dora299_strict_reconviagen_inference_aggregate.v1"
OBJECT_FORMAT = "reconviagen.dorabench_dora299_strict_reconviagen_metric.v2"
WORKER_FORMAT = "reconviagen.dorabench_dora299_strict_reconviagen_metric_worker.v2"
AGGREGATE_FORMAT = "reconviagen.dorabench_dora299_strict_reconviagen_metric_aggregate.v2"

# The Stock SS support, Stock SLat and frozen Mesh decoder all live in the
# training sparse-latent/model-O frame.  ``decoded.to_trimesh(False)`` exports
# that frame directly.  TRELLIS' optional ``transform_pose=True`` rotation is
# only a presentation/export conversion (z-up to y-up); applying it during a
# model-O benchmark rotates the prediction a second time.
DECODER_TO_MODEL_O = np.eye(4, dtype=np.float64)


def _subset(path: Path) -> dict[str, Any]:
    payload = load_json(path)
    saved = str(payload.get("subset_identity", ""))
    identity = dict(payload)
    identity.pop("subset_identity", None)
    rows = list(payload.get("objects") or [])
    if (
        payload.get("format") != SUBSET_FORMAT
        or payload.get("passed") is not True
        or len(rows) != 299
        or len({str(row["uid"]) for row in rows}) != 299
        or not saved
        or canonical_sha256(identity) != saved
    ):
        raise RuntimeError(f"Dora299 subset identity differs: {path}")
    return payload


def _inference(path: Path, subset: dict[str, Any], seed: int) -> dict[str, Any]:
    payload = load_json(path)
    rows = list(payload.get("objects") or [])
    expected = [f"{row['category']}:{row['uid']}" for row in subset["objects"]]
    observed = [str(row["object_key"]) for row in rows]
    if (
        payload.get("format") != INFERENCE_FORMAT
        or payload.get("passed") is not True
        or payload.get("subset_identity") != subset["subset_identity"]
        or len(rows) != 299
        or len(set(observed)) != 299
        or observed != expected
        or any(int(row.get("seed", -1)) != int(seed) for row in rows)
    ):
        raise RuntimeError(f"strict ReconViaGen inference matrix differs: {path}")
    return payload


def _metrics(predicted, target, *, count: int, seed: int, radius: float):
    predicted = predicted.copy()
    predicted.apply_transform(DECODER_TO_MODEL_O)
    predicted_points, _ = deterministic_surface_sample(predicted, count, seed)
    target_points, _ = deterministic_surface_sample(target, count, seed + 1)
    predicted_points, predicted_norm = _normalize_points(predicted_points)
    target_points, target_norm = _normalize_points(target_points)
    target_tree = cKDTree(target_points)
    predicted_tree = cKDTree(predicted_points)
    pred_to_gt = target_tree.query(predicted_points, k=1, workers=1)[0]
    gt_to_pred = predicted_tree.query(target_points, k=1, workers=1)[0]
    precision = float(np.mean(pred_to_gt < float(radius)))
    recall = float(np.mean(gt_to_pred < float(radius)))
    fscore = (
        0.0
        if precision + recall <= 1.0e-12
        else float(2.0 * precision * recall / (precision + recall))
    )
    chamfer = float(0.5 * (pred_to_gt.mean() + gt_to_pred.mean()))
    return {
        "chamfer_distance": chamfer,
        "chamfer_distance_symmetric_sum": float(2.0 * chamfer),
        "pred_to_gt_mean": float(pred_to_gt.mean()),
        "gt_to_pred_mean": float(gt_to_pred.mean()),
        "fscore": fscore,
        "precision": precision,
        "recall": recall,
    }, {"predicted": predicted_norm, "target": target_norm}


def worker(args: argparse.Namespace) -> None:
    subset_path = Path(args.subset_manifest).expanduser().resolve(strict=True)
    inference_path = Path(args.inference_aggregate).expanduser().resolve(strict=True)
    subset = _subset(subset_path)
    inference = _inference(inference_path, subset, int(args.seed))
    source_by_uid = {str(row["uid"]): row for row in subset["objects"]}
    records = [
        row
        for index, row in enumerate(inference["objects"])
        if index % int(args.worker_count) == int(args.worker_index)
    ]
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    subset_sha = sha256_file(subset_path)
    inference_sha = sha256_file(inference_path)
    results = []
    for position, record in enumerate(records, 1):
        uid = str(record["object_id"])
        source = source_by_uid[uid]
        mesh_path = Path(record["mesh"]).resolve(strict=True)
        destination = output / "objects" / uid / "metric.json"
        if destination.is_file():
            cached = load_json(destination)
            if (
                cached.get("format") == OBJECT_FORMAT
                and cached.get("passed") is True
                and cached.get("predicted_mesh_sha256") == sha256_file(mesh_path)
                and cached.get("subset_manifest_sha256") == subset_sha
                and cached.get("inference_aggregate_sha256") == inference_sha
                and int(cached.get("surface_points", -1)) == int(args.surface_points)
                and float(cached.get("fscore_radius", -1.0)) == float(args.fscore_radius)
            ):
                results.append(cached)
                print(f"[dora299:recon-metric] {position}/{len(records)} reused uid={uid}", flush=True)
                continue
            raise RuntimeError(f"stale Dora299 metric artifact: {destination}")
        predicted = _one_mesh(mesh_path)
        target = _gt_mesh(source)
        uid_seed = int.from_bytes(hashlib.sha256(uid.encode("utf-8")).digest()[:8], "big")
        metric_seed = (int(args.seed) * 1_000_003 + uid_seed) % (2**63 - 1)
        metrics, normalization = _metrics(
            predicted,
            target,
            count=int(args.surface_points),
            seed=metric_seed,
            radius=float(args.fscore_radius),
        )
        result = {
            "format": OBJECT_FORMAT,
            "passed": True,
            "uid": uid,
            "category": str(source["category"]),
            "seed": int(args.seed),
            "subset_manifest_sha256": subset_sha,
            "inference_aggregate_sha256": inference_sha,
            "predicted_mesh": str(mesh_path),
            "predicted_mesh_sha256": sha256_file(mesh_path),
            "gt_mesh": str(Path(source["source_mesh"]).resolve(strict=True)),
            "gt_scan_tree_sha256": source["source_scan_tree_sha256"],
            "surface_points": int(args.surface_points),
            "fscore_radius": float(args.fscore_radius),
            "metric_seed": metric_seed,
            "normalization": normalization,
            "metrics": metrics,
            "alignment": (
                "decoder-native equals sparse-latent/model-O (identity), then "
                "independent AABB center/max-extent normalization; no GT fit/ICP"
            ),
        }
        atomic_json(destination, result)
        results.append(result)
        print(
            f"[dora299:recon-metric] {position}/{len(records)} uid={uid} "
            f"CD={metrics['chamfer_distance']:.8f} F={metrics['fscore']:.8f}",
            flush=True,
        )
    report = {
        "format": WORKER_FORMAT,
        "passed": len(results) == len(records),
        "worker_index": int(args.worker_index),
        "worker_count": int(args.worker_count),
        "object_count": len(results),
        "subset_manifest_sha256": subset_sha,
        "inference_aggregate_sha256": inference_sha,
        "objects": results,
    }
    atomic_json(output / "metrics_report.json", report)


def aggregate(args: argparse.Namespace) -> None:
    subset_path = Path(args.subset_manifest).expanduser().resolve(strict=True)
    inference_path = Path(args.inference_aggregate).expanduser().resolve(strict=True)
    current_path = Path(args.current_report).expanduser().resolve(strict=True)
    subset = _subset(subset_path)
    _inference(inference_path, subset, int(args.seed))
    subset_sha = sha256_file(subset_path)
    inference_sha = sha256_file(inference_path)
    root = Path(args.workers_root).expanduser().resolve(strict=True)
    reports = sorted(root.glob("worker_*/metrics_report.json"))
    if len(reports) != int(args.expected_workers):
        raise RuntimeError(f"Dora299 metric worker count differs: {len(reports)}")
    rows = []
    for path in reports:
        report = load_json(path)
        if (
            report.get("format") != WORKER_FORMAT
            or report.get("passed") is not True
            or report.get("subset_manifest_sha256") != subset_sha
            or report.get("inference_aggregate_sha256") != inference_sha
        ):
            raise RuntimeError(f"Dora299 metric worker identity differs: {path}")
        rows.extend(report["objects"])
    expected = {str(row["uid"]) for row in subset["objects"]}
    observed = [str(row["uid"]) for row in rows]
    if len(rows) != 299 or len(set(observed)) != 299 or set(observed) != expected:
        raise RuntimeError("Dora299 metric workers do not exactly cover the frozen subset")
    rows.sort(key=lambda row: str(row["uid"]))
    by_level = {}
    for level in sorted({str(row["category"]) for row in rows}):
        selected = [row for row in rows if row["category"] == level]
        by_level[level] = {
            "object_count": len(selected),
            "chamfer_distance": _summary([float(row["metrics"]["chamfer_distance"]) for row in selected]),
            "fscore": _summary([float(row["metrics"]["fscore"]) for row in selected]),
        }
    current = load_json(current_path)
    if (
        current.get("format")
        != "reconviagen.dorabench_dora300_ss30k_slat30k_metric_aggregate_failure_aware.v1"
        or current.get("passed") is not True
        or int(current.get("requested_object_count", -1)) != 300
        or int(current.get("surface_metric_object_count", -1)) != 299
        or int(current.get("registered_model_output_failure_count", -1)) != 1
        or current.get("benchmark_manifest_sha256")
        != subset["parent_benchmark_manifest"]["sha256"]
    ):
        raise RuntimeError("bound SS30K+SLat30K Dora299 report identity differs")
    current_by_uid = {str(row["uid"]): row for row in current["objects"]}
    if set(current_by_uid) != expected:
        raise RuntimeError("current-model metric object set differs from Dora299")
    strict_by_uid = {str(row["uid"]): row for row in rows}
    cd_improvement = [
        float(strict_by_uid[uid]["metrics"]["chamfer_distance"])
        - float(current_by_uid[uid]["metrics"]["chamfer_distance"])
        for uid in sorted(expected)
    ]
    fscore_delta = [
        float(current_by_uid[uid]["metrics"]["fscore"])
        - float(strict_by_uid[uid]["metrics"]["fscore"])
        for uid in sorted(expected)
    ]
    report = {
        "format": AGGREGATE_FORMAT,
        "passed": True,
        "method": "strict original ReconViaGen: VGGT -> Stock SS -> Stock SLat -> Stock Mesh decoder",
        "subset_manifest": str(subset_path),
        "subset_manifest_sha256": subset_sha,
        "subset_identity": subset["subset_identity"],
        "excluded_current_failure": subset["excluded_current_failure"],
        "inference_aggregate": str(inference_path),
        "inference_aggregate_sha256": inference_sha,
        "object_count": 299,
        "complexity_level_count": len(by_level),
        "surface_points_per_mesh": int(args.surface_points),
        "fscore_radius": float(args.fscore_radius),
        "chamfer_distance": _summary([float(row["metrics"]["chamfer_distance"]) for row in rows]),
        "fscore": _summary([float(row["metrics"]["fscore"]) for row in rows]),
        "by_complexity_level": by_level,
        "objects": rows,
        "paired_current_ss30k_slat30k": {
            "report": str(current_path),
            "report_sha256": sha256_file(current_path),
            "chamfer_improvement_current_over_reconviagen": {
                **_summary(cd_improvement),
                "positive_rate": float(np.mean(np.asarray(cd_improvement) > 0.0)),
            },
            "fscore_delta_current_minus_reconviagen": {
                **_summary(fscore_delta),
                "positive_rate": float(np.mean(np.asarray(fscore_delta) > 0.0)),
            },
        },
        "coordinate_contract": {
            "strict_inference_mesh": (
                "decoder-native sparse-latent/model-O; transform_pose=False"
            ),
            "decoder_to_model_o": DECODER_TO_MODEL_O.tolist(),
            "decoder_to_model_o_rule": "identity:(x,y,z)->(x,y,z)",
            "trellis_transform_pose_true_consumed": False,
            "gt_fitted": False,
            "normalization": "independent AABB center/max-extent to [-1,1]",
            "forbidden": ["per-object rotation fit", "ICP", "reflection", "best-of-transform"],
        },
        "metric_stage_loaded_model": False,
    }
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "report.json", report)
    lines = [
        "Dora-Bench current-valid 299 strict ReconViaGen evaluation",
        "=========================================================",
        "objects: 299",
        f"Chamfer Distance: mean={report['chamfer_distance']['mean']:.8f} median={report['chamfer_distance']['median']:.8f}",
        f"F-score@0.1: mean={report['fscore']['mean']:.8f} median={report['fscore']['median']:.8f}",
        f"Current-over-Recon CD improvement: mean={report['paired_current_ss30k_slat30k']['chamfer_improvement_current_over_reconviagen']['mean']:+.8f}",
        f"Current-minus-Recon F-score: mean={report['paired_current_ss30k_slat30k']['fscore_delta_current_minus_reconviagen']['mean']:+.8f}",
        "Alignment: decoder-native/model-O identity; no GT fit/ICP",
        f"report: {output / 'report.json'}",
    ]
    (output / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("worker")
    p.add_argument("--subset_manifest", required=True)
    p.add_argument("--inference_aggregate", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--worker_index", type=int, required=True)
    p.add_argument("--worker_count", type=int, required=True)
    p.add_argument("--surface_points", type=int, default=100000)
    p.add_argument("--fscore_radius", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=worker)
    p = sub.add_parser("aggregate")
    p.add_argument("--subset_manifest", required=True)
    p.add_argument("--inference_aggregate", required=True)
    p.add_argument("--current_report", required=True)
    p.add_argument("--workers_root", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--expected_workers", type=int, required=True)
    p.add_argument("--surface_points", type=int, default=100000)
    p.add_argument("--fscore_radius", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=aggregate)
    return parser


def main() -> None:
    args = make_parser().parse_args()
    if int(args.surface_points) <= 0 or float(args.fscore_radius) <= 0.0:
        raise ValueError("metric sampling/radius must be positive")
    args.func(args)


if __name__ == "__main__":
    main()
