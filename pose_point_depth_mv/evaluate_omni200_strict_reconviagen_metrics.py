#!/usr/bin/env python3
"""Score frozen strict ReconViaGen Omni200 meshes with the registered metrics."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from manual_mesh_reconstruction.common import atomic_json, load_json, sha256_file
from pose_point_depth_mv.evaluate_omni200_ss30k_slat30k import (
    _benchmark,
    _gt_mesh,
    _normalize_points,
    _one_mesh,
    _summary,
)
from pose_point_depth_mv.mesh_benchmark_metrics import deterministic_surface_sample


INFERENCE_FORMAT = (
    "reconviagen.omniobject3d_omni200_strict_reconviagen_inference_aggregate.v1"
)
OBJECT_FORMAT = "reconviagen.omniobject3d_strict_reconviagen_metric.v1"
WORKER_FORMAT = "reconviagen.omniobject3d_strict_reconviagen_metric_worker.v1"
AGGREGATE_FORMAT = "reconviagen.omniobject3d_strict_reconviagen_metric_aggregate.v1"


def _inference(path: Path, benchmark: dict[str, Any], seed: int) -> dict[str, Any]:
    payload = load_json(path)
    if payload.get("format") != INFERENCE_FORMAT or payload.get("passed") is not True:
        raise RuntimeError(f"strict ReconViaGen inference aggregate did not pass: {path}")
    rows = list(payload.get("objects") or [])
    expected = {f"{row['category']}:{row['uid']}" for row in benchmark["objects"]}
    keys = [str(row["object_key"]) for row in rows]
    if (
        len(rows) != 200
        or len(set(keys)) != 200
        or set(keys) != expected
        or any(int(row.get("seed", -1)) != int(seed) for row in rows)
    ):
        raise RuntimeError("strict ReconViaGen inference object/seed matrix differs")
    return payload


def _metrics(predicted, target, *, count: int, seed: int, radius: float):
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
    symmetric_mean = float(0.5 * (pred_to_gt.mean() + gt_to_pred.mean()))
    return {
        "chamfer_distance": symmetric_mean,
        "chamfer_distance_symmetric_sum": float(2.0 * symmetric_mean),
        "pred_to_gt_mean": float(pred_to_gt.mean()),
        "gt_to_pred_mean": float(gt_to_pred.mean()),
        "fscore": fscore,
        "precision": precision,
        "recall": recall,
    }, {"predicted": predicted_norm, "target": target_norm}


def worker(args: argparse.Namespace) -> None:
    benchmark_path = Path(args.benchmark_manifest).expanduser().resolve(strict=True)
    inference_path = Path(args.inference_aggregate).expanduser().resolve(strict=True)
    benchmark = _benchmark(benchmark_path, expected_objects=200)
    inference = _inference(inference_path, benchmark, int(args.seed))
    benchmark_by_uid = {str(row["uid"]): row for row in benchmark["objects"]}
    records = [
        row
        for index, row in enumerate(inference["objects"])
        if index % int(args.worker_count) == int(args.worker_index)
    ]
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    benchmark_sha = sha256_file(benchmark_path)
    inference_sha = sha256_file(inference_path)
    results = []
    for position, record in enumerate(records, 1):
        uid = str(record["object_id"])
        source = benchmark_by_uid[uid]
        mesh_path = Path(record["mesh"]).resolve(strict=True)
        destination = output / "objects" / uid / "metric.json"
        if destination.is_file():
            cached = load_json(destination)
            if (
                cached.get("format") == OBJECT_FORMAT
                and cached.get("passed") is True
                and cached.get("predicted_mesh_sha256") == sha256_file(mesh_path)
                and cached.get("benchmark_manifest_sha256") == benchmark_sha
                and cached.get("inference_aggregate_sha256") == inference_sha
                and int(cached.get("surface_points", -1)) == int(args.surface_points)
                and float(cached.get("fscore_radius", -1.0)) == float(args.fscore_radius)
            ):
                results.append(cached)
                print(f"[omni200:recon-metric] {position}/{len(records)} reused uid={uid}", flush=True)
                continue
            raise RuntimeError(f"stale strict ReconViaGen metric: {destination}")
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
            "category": source["category"],
            "seed": int(args.seed),
            "benchmark_manifest_sha256": benchmark_sha,
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
            "alignment": "none; independent AABB center/max-extent normalization only",
        }
        atomic_json(destination, result)
        results.append(result)
        print(
            f"[omni200:recon-metric] {position}/{len(records)} uid={uid} "
            f"CD={metrics['chamfer_distance']:.8f} F={metrics['fscore']:.8f}",
            flush=True,
        )
    report = {
        "format": WORKER_FORMAT,
        "passed": len(results) == len(records),
        "worker_index": int(args.worker_index),
        "worker_count": int(args.worker_count),
        "object_count": len(results),
        "benchmark_manifest_sha256": benchmark_sha,
        "inference_aggregate_sha256": inference_sha,
        "objects": results,
    }
    atomic_json(output / "metrics_report.json", report)


def aggregate(args: argparse.Namespace) -> None:
    benchmark_path = Path(args.benchmark_manifest).expanduser().resolve(strict=True)
    inference_path = Path(args.inference_aggregate).expanduser().resolve(strict=True)
    current_path = Path(args.current_report).expanduser().resolve(strict=True)
    benchmark = _benchmark(benchmark_path, expected_objects=200)
    _inference(inference_path, benchmark, int(args.seed))
    root = Path(args.workers_root).expanduser().resolve(strict=True)
    reports = sorted(root.glob("worker_*/metrics_report.json"))
    if len(reports) != int(args.expected_workers):
        raise RuntimeError(f"strict metric worker count differs: {len(reports)}")
    benchmark_sha = sha256_file(benchmark_path)
    inference_sha = sha256_file(inference_path)
    rows = []
    for path in reports:
        report = load_json(path)
        if (
            report.get("format") != WORKER_FORMAT
            or report.get("passed") is not True
            or report.get("benchmark_manifest_sha256") != benchmark_sha
            or report.get("inference_aggregate_sha256") != inference_sha
        ):
            raise RuntimeError(f"strict metric worker identity differs: {path}")
        rows.extend(report["objects"])
    expected = {str(row["uid"]) for row in benchmark["objects"]}
    uids = [str(row["uid"]) for row in rows]
    if len(rows) != 200 or len(set(uids)) != 200 or set(uids) != expected:
        raise RuntimeError("strict metric workers do not exactly cover Omni200")
    rows.sort(key=lambda row: str(row["uid"]))
    by_category = {}
    for category in sorted({str(row["category"]) for row in rows}):
        selected = [row for row in rows if row["category"] == category]
        by_category[category] = {
            "object_count": len(selected),
            "chamfer_distance": _summary([float(row["metrics"]["chamfer_distance"]) for row in selected]),
            "fscore": _summary([float(row["metrics"]["fscore"]) for row in selected]),
        }
    current = load_json(current_path)
    if (
        current.get("passed") is not True
        or current.get("object_count") != 200
        or current.get("benchmark_manifest_sha256") != benchmark_sha
    ):
        raise RuntimeError("bound SS30K+SLat30K report identity differs")
    current_by_uid = {str(row["uid"]): row for row in current["objects"]}
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
        "benchmark_manifest": str(benchmark_path),
        "benchmark_manifest_sha256": benchmark_sha,
        "inference_aggregate": str(inference_path),
        "inference_aggregate_sha256": inference_sha,
        "object_count": 200,
        "category_count": len(by_category),
        "surface_points_per_mesh": int(args.surface_points),
        "fscore_radius": float(args.fscore_radius),
        "chamfer_distance": _summary([float(row["metrics"]["chamfer_distance"]) for row in rows]),
        "fscore": _summary([float(row["metrics"]["fscore"]) for row in rows]),
        "by_category": by_category,
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
        "alignment": "none; independent AABB center/max-extent normalization only; no rotation/ICP alignment",
        "metric_stage_loaded_model": False,
    }
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "report.json", report)
    lines = [
        "OmniObject3D 200-object strict ReconViaGen evaluation",
        "====================================================",
        "objects: 200 categories: 20",
        f"Chamfer Distance: mean={report['chamfer_distance']['mean']:.8f} median={report['chamfer_distance']['median']:.8f}",
        f"F-score@0.1: mean={report['fscore']['mean']:.8f} median={report['fscore']['median']:.8f}",
        f"Current-over-Recon CD improvement: mean={report['paired_current_ss30k_slat30k']['chamfer_improvement_current_over_reconviagen']['mean']:+.8f}",
        f"Current-minus-Recon F-score: mean={report['paired_current_ss30k_slat30k']['fscore_delta_current_minus_reconviagen']['mean']:+.8f}",
        "Alignment: independent AABB normalization only; no rotation/ICP alignment",
        f"report: {output / 'report.json'}",
    ]
    (output / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("worker")
    p.add_argument("--benchmark_manifest", required=True)
    p.add_argument("--inference_aggregate", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--worker_index", type=int, required=True)
    p.add_argument("--worker_count", type=int, required=True)
    p.add_argument("--surface_points", type=int, default=100000)
    p.add_argument("--fscore_radius", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=worker)
    p = sub.add_parser("aggregate")
    p.add_argument("--benchmark_manifest", required=True)
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
