#!/usr/bin/env python3
"""Summarize the three-arm Native-SLat objective-v2 Train16 ablation."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Any

import numpy as np

from pose_point_depth_mv.eval_direct_slat_flow import summarize


EXPECTED_REPORT = (
    "pose_point_depth_mv.native_slat_condition_only_no_vggt_objective_v2_mesh.v1"
)
METRICS = (
    "chamfer_l1_improvement",
    "fscore_0p02_delta",
    "normal_consistency_delta",
    "largest_component_ratio_delta",
)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logit_report", required=True)
    parser.add_argument("--uniform_report", required=True)
    parser.add_argument("--geometry_report", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected_objects", type=int, default=16)
    parser.add_argument("--bootstrap_samples", type=int, default=5000)
    return parser


def _load(path: str, expected_objects: int) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if value.get("format") != EXPECTED_REPORT or value.get("passed") is not True:
        raise ValueError(f"invalid objective-v2 Mesh report: {path}")
    if int(value.get("object_count", -1)) != int(expected_objects):
        raise ValueError(f"unexpected object count in {path}")
    if int(value.get("record_count", -1)) != int(expected_objects):
        raise ValueError(f"Train16 ablation expects one seed per object: {path}")
    return value


def _identity(row: dict[str, Any]) -> tuple[str, str, int]:
    return str(row["object_uid"]), str(row["uid"]), int(row["seed"])


def _full_values(row: dict[str, Any]) -> dict[str, float]:
    surface = row["branches"]["full"]["surface"]
    structure = row["branches"]["full"]["structure"]
    return {
        "chamfer_l1": float(surface["chamfer_l1"]),
        "fscore_0p02": float(surface["fscore_0p02"]),
        "normal_consistency": float(surface["normal_consistency"]),
        "largest_component_ratio": float(structure["largest_component_ratio"]),
    }


def _paired_improvement(
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    left_rows = {_identity(row): row for row in left["records"]}
    right_rows = {_identity(row): row for row in right["records"]}
    if set(left_rows) != set(right_rows):
        raise ValueError("ablation reports do not contain the same object/uid/seed matrix")
    object_values: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for identity in sorted(left_rows):
        left_value = _full_values(left_rows[identity])
        right_value = _full_values(right_rows[identity])
        object_uid = identity[0]
        object_values[object_uid]["chamfer_l1_improvement"].append(
            right_value["chamfer_l1"] - left_value["chamfer_l1"]
        )
        for output_name, field in (
            ("fscore_0p02_delta", "fscore_0p02"),
            ("normal_consistency_delta", "normal_consistency"),
            ("largest_component_ratio_delta", "largest_component_ratio"),
        ):
            object_values[object_uid][output_name].append(
                left_value[field] - right_value[field]
            )
    summaries = {}
    for index, metric in enumerate(METRICS):
        values = [
            float(np.mean(by_metric[metric]))
            for _, by_metric in sorted(object_values.items())
        ]
        summaries[metric] = summarize(
            values,
            bootstrap_samples=int(bootstrap_samples),
            seed=int(seed) + index,
        )
    return summaries


def _stock_gate(report: dict[str, Any]) -> dict[str, Any]:
    chamfer = report["summary"]["chamfer_l1_improvement"]
    fscore = report["summary"]["fscore_0p02_delta"]
    checks = {
        "chamfer_mean_gt_zero": float(chamfer["mean"]) > 0.0,
        "chamfer_median_gt_zero": float(chamfer["median"]) > 0.0,
        "chamfer_win_rate_ge_0p625": float(chamfer["positive_rate"]) >= 0.625,
        "fscore_mean_nonnegative": float(fscore["mean"]) >= 0.0,
    }
    return {"passed": all(checks.values()), "checks": checks}


def _paired_gate(summary: dict[str, Any]) -> dict[str, Any]:
    chamfer = summary["chamfer_l1_improvement"]
    fscore = summary["fscore_0p02_delta"]
    checks = {
        "chamfer_mean_gt_zero": float(chamfer["mean"]) > 0.0,
        "chamfer_median_gt_zero": float(chamfer["median"]) > 0.0,
        "chamfer_win_rate_ge_0p625": float(chamfer["positive_rate"]) >= 0.625,
        "fscore_mean_nonnegative": float(fscore["mean"]) >= 0.0,
    }
    return {"passed": all(checks.values()), "checks": checks}


def main() -> None:
    args = make_parser().parse_args()
    if int(args.expected_objects) <= 0 or int(args.bootstrap_samples) <= 0:
        raise ValueError("expected_objects/bootstrap_samples must be positive")
    reports = {
        "logit_flow_only": _load(args.logit_report, args.expected_objects),
        "uniform_flow_only": _load(args.uniform_report, args.expected_objects),
        "uniform_geometry_trust": _load(args.geometry_report, args.expected_objects),
    }
    uniform_vs_logit = _paired_improvement(
        reports["uniform_flow_only"],
        reports["logit_flow_only"],
        bootstrap_samples=args.bootstrap_samples,
        seed=20260813,
    )
    geometry_vs_uniform = _paired_improvement(
        reports["uniform_geometry_trust"],
        reports["uniform_flow_only"],
        bootstrap_samples=args.bootstrap_samples,
        seed=20260823,
    )
    stock_gates = {name: _stock_gate(report) for name, report in reports.items()}
    paired_gates = {
        "uniform_flow_only_vs_logit_flow_only": _paired_gate(uniform_vs_logit),
        "uniform_geometry_trust_vs_uniform_flow_only": _paired_gate(
            geometry_vs_uniform
        ),
    }
    decision = {
        "uniform_schedule_advantage": paired_gates[
            "uniform_flow_only_vs_logit_flow_only"
        ]["passed"],
        "geometry_objective_stock_fit_gate": stock_gates[
            "uniform_geometry_trust"
        ]["passed"],
        "geometry_objective_increment_gate": paired_gates[
            "uniform_geometry_trust_vs_uniform_flow_only"
        ]["passed"],
    }
    decision["proceed_to_predicted_support_rollout"] = bool(
        decision["geometry_objective_stock_fit_gate"]
        and decision["geometry_objective_increment_gate"]
    )
    output = {
        "format": "pose_point_depth_mv.native_slat_objective_v2_train16_ablation.v1",
        "passed": True,
        "formal": False,
        "object_count": int(args.expected_objects),
        "reports": {
            name: {
                "path": str(Path(path).resolve()),
                "checkpoint": reports[name]["run_config"]["checkpoint"],
                "summary_vs_own_matched_stock": reports[name]["summary"],
            }
            for name, path in (
                ("logit_flow_only", args.logit_report),
                ("uniform_flow_only", args.uniform_report),
                ("uniform_geometry_trust", args.geometry_report),
            )
        },
        "paired_full_mesh_comparisons": {
            "uniform_flow_only_vs_logit_flow_only": uniform_vs_logit,
            "uniform_geometry_trust_vs_uniform_flow_only": geometry_vs_uniform,
        },
        "stock_gates": stock_gates,
        "paired_gates": paired_gates,
        "decision": decision,
        "scope_guard": (
            "training-overlap Train16 development gate only; no generalization or "
            "formal science claim. Predicted-support rollout is admitted only when "
            "both geometry-vs-Stock and geometry-vs-uniform gates pass."
        ),
    }
    destination = Path(args.output).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(destination), **decision}, indent=2))


if __name__ == "__main__":
    main()
