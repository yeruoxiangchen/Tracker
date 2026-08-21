#!/usr/bin/env python3
"""Aggregate paired Stock-vs-Full Objaverse2K training-overlap workers."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from pose_point_depth_mv.eval_direct_slat_flow import summarize
from pose_point_depth_mv.evaluate_objaverse2k_no_vggt_slat import (
    STRUCTURE_METRICS,
    SURFACE_METRICS,
    load_json,
    paired_improvement,
    stable_seed,
    validate_worker_reports,
)
from pose_point_depth_mv.omni_real_benchmark_common import sha256_file


REPORT_FORMAT = "pose_point_depth_mv.objaverse2k_train_stock_full_diagnostic.v1"
def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def branch_summary(records: list[dict[str, Any]], branch: str) -> dict[str, Any]:
    return {
        "surface": {
            metric: {
                "count": len(records),
                "mean": float(np.mean([row["branches"][branch]["surface"][metric] for row in records])),
                "median": float(np.median([row["branches"][branch]["surface"][metric] for row in records])),
            }
            for metric in SURFACE_METRICS
        },
        "structure": {
            metric: {
                "count": len(records),
                "mean": float(np.mean([row["branches"][branch]["structure"][metric] for row in records])),
                "median": float(np.median([row["branches"][branch]["structure"][metric] for row in records])),
            }
            for metric in STRUCTURE_METRICS
        },
    }


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker_reports", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--expected_workers", type=int, default=8)
    parser.add_argument("--expected_objects", type=int, default=64)
    parser.add_argument("--bootstrap_samples", type=int, default=10000)
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> None:
    args = make_parser().parse_args()
    paths = [value.strip() for value in args.worker_reports.split(",") if value.strip()]
    reports, indexed = validate_worker_reports(
        paths,
        model_label="objaverse2k",
        expected_workers=int(args.expected_workers),
    )
    config = reports[0]["run_config"]
    required = {
        "training_overlap": True,
        "training_object_disjoint": False,
        "source_mesh_disjoint": False,
        "joint_seeds": [42],
        "expected_objects": int(args.expected_objects),
        "num_workers": int(args.expected_workers),
    }
    mismatch = {
        key: (config.get(key), expected)
        for key, expected in required.items()
        if config.get(key) != expected
    }
    coordinate = dict(config.get("coordinate_evaluation", {}))
    if coordinate.get("alignment") != "fixed proper axis transform only; no ICP/scale/GT fit":
        mismatch["coordinate_evaluation"] = (coordinate, "strict fixed-axis")
    if mismatch:
        raise RuntimeError(f"training-overlap worker protocol differs: {mismatch}")
    if len(indexed) != int(args.expected_objects):
        raise RuntimeError("training-overlap record count differs")

    records = [indexed[key] for key in sorted(indexed)]
    if any(
        row.get("same_native_ss_coordinates") is not True
        or row.get("same_initial_noise") is not True
        for row in records
    ):
        raise RuntimeError("Stock/Full pairing did not preserve coordinates and noise")
    improvements = [
        {
            "object_uid": str(row["identity"]["object_uid"]),
            "uid": str(row["identity"]["uid"]),
            "support_seed": int(row["identity"]["support_seed"]),
            "metrics": paired_improvement(row["branches"]["full"], row["branches"]["stock"]),
        }
        for row in records
    ]
    metrics = {}
    for metric in (*SURFACE_METRICS, *STRUCTURE_METRICS):
        values = [float(row["metrics"][metric]) for row in improvements]
        metrics[metric] = summarize(
            values,
            bootstrap_samples=int(args.bootstrap_samples),
            seed=stable_seed("objaverse2k_train_stock_full", metric),
        )
    chamfer_gate = (
        metrics["chamfer_l1"]["bootstrap_mean_95_ci"][0] > 0.0
        and metrics["chamfer_l1"]["positive_rate"] > 0.5
    )
    fscore_gate = metrics["fscore_0p02"]["bootstrap_mean_95_ci"][0] > 0.0
    fit_advantage_established = bool(chamfer_gate and fscore_gate)
    report = {
        "format": REPORT_FORMAT,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "passed": True,
        "formal": False,
        "training_overlap": True,
        "training_object_disjoint": False,
        "source_mesh_disjoint": False,
        "protocol_scope": (
            f"Objaverse2K train{int(args.expected_objects)} Stock-vs-Full "
            "fitting diagnostic"
        ),
        "object_count": len(records),
        "record_count": len(records),
        "seed_count": 1,
        "seeds": [42],
        "unit_of_analysis": (
            f"{int(args.expected_objects)} training objects; one frozen support seed per object"
        ),
        "positive_definition": "positive means Objaverse2K Full is better than frozen Stock",
        "same_native_ss_coordinates": True,
        "same_initial_noise": True,
        "same_frozen_decoder": True,
        "coordinate_evaluation": coordinate,
        "checkpoint": {
            "path": config["checkpoint"],
            "sha256": config["checkpoint_sha256"],
            "step": config["checkpoint_step"],
            "weights": config["weights"],
        },
        "cache_manifest": {
            "path": config["cache_manifest"],
            "sha256": config["cache_manifest_sha256"],
        },
        "worker_reports": [
            {"path": str(Path(path).resolve()), "sha256": sha256_file(Path(path).resolve())}
            for path in paths
        ],
        "branches": {
            "stock": branch_summary(records, "stock"),
            "objaverse2k_full": branch_summary(records, "full"),
        },
        "paired_improvement": metrics,
        "chamfer_l1_wins": {
            "objaverse2k_full": sum(row["metrics"]["chamfer_l1"] > 0.0 for row in improvements),
            "stock": sum(row["metrics"]["chamfer_l1"] < 0.0 for row in improvements),
            "ties": sum(row["metrics"]["chamfer_l1"] == 0.0 for row in improvements),
        },
        "decision": {
            "fit_advantage_established": fit_advantage_established,
            "required_gates": {
                "chamfer_l1_object_bootstrap_95ci_lower_gt_zero": bool(
                    metrics["chamfer_l1"]["bootstrap_mean_95_ci"][0] > 0.0
                ),
                "chamfer_l1_object_win_rate_gt_half": bool(
                    metrics["chamfer_l1"]["positive_rate"] > 0.5
                ),
                "fscore_0p02_object_bootstrap_95ci_lower_gt_zero": bool(
                    metrics["fscore_0p02"]["bootstrap_mean_95_ci"][0] > 0.0
                ),
            },
            "interpretation": (
                "Current Objaverse2K SLat establishes a training-object geometry "
                "advantage over its frozen Stock baseline. Compare with dev64 to "
                "diagnose overfitting/generalization."
                if fit_advantage_established
                else "Current Objaverse2K SLat does not establish a training-object "
                "geometry advantage over its frozen Stock baseline under this "
                "single-seed fitting diagnostic."
            ),
        },
        "objects": improvements,
        "limitations": [
            f"All {int(args.expected_objects)} objects overlap Objaverse2K SLat training; "
            "this measures fitting, not generalization.",
            "The existing training cache provides only support seed 42, so this is a single-seed diagnostic.",
            "Stock and Full share coordinates, initial noise and decoder; only the trained SLat path is enabled in Full.",
        ],
    }
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists() and not args.resume:
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True, exist_ok=bool(args.resume))
    atomic_json(output_dir / "report.json", report)
    chamfer = metrics["chamfer_l1"]
    fscore = metrics["fscore_0p02"]
    normal = metrics["normal_consistency"]
    lines = [
        f"Objaverse2K train{int(args.expected_objects)} Full vs frozen Stock",
        "=" * 43,
        "passed: true",
        "formal: false",
        "training_overlap: true",
        f"objects: {int(args.expected_objects)}, seeds: [42]",
        "same SS coordinates / initial noise / decoder: true",
        "alignment: fixed axis only; no ICP/scale/GT fit",
        f"Chamfer improvement: {chamfer['mean']:+.8f}, median={chamfer['median']:+.8f}, win={chamfer['positive_rate']:.4f}, CI={chamfer['bootstrap_mean_95_ci']}",
        f"F@0.02 improvement: {fscore['mean']:+.8f}, CI={fscore['bootstrap_mean_95_ci']}",
        f"Normal improvement: {normal['mean']:+.8f}, CI={normal['bootstrap_mean_95_ci']}",
        f"Fit advantage established: {str(fit_advantage_established).lower()}",
        "Positive means Objaverse2K Full is better than Stock.",
    ]
    (output_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
