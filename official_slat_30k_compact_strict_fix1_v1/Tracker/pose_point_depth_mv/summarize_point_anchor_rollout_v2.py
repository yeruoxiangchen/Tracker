#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pose_point_depth_mv.point_anchor_v2 import POINT_CONTROL_NAMES


ROLLOUT_VERSION = "pose_point_depth_mv.point_anchor_rollout.v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Strict multi-training-seed summary for Point-anchor V2 rollout."
    )
    parser.add_argument("--report_dirs", nargs="+", required=True)
    parser.add_argument("--expected_training_seeds", default="42,43,44")
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def load_report(path: str) -> dict[str, Any]:
    report_path = Path(path)
    if report_path.is_dir():
        report_path = report_path / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("format") != ROLLOUT_VERSION:
        raise ValueError(f"unexpected rollout report format: {report_path}")
    report["_report_path"] = str(report_path.resolve())
    return report


def protocol(report: dict[str, Any]) -> dict[str, Any]:
    args = report["args"]
    return {
        "cache_config_hash": report["cache_config_hash"],
        "checkpoint_step": report["checkpoint_step"],
        "split_name": report["split_name"],
        "sample_count": report["sample_count"],
        "object_count": report["object_count"],
        "eval_object_uid_hash": report["eval_object_uid_hash"],
        "noise_seeds": report["noise_seeds"],
        "steps": args["steps"],
        "cfg_strength": args["cfg_strength"],
        "cfg_interval": args["cfg_interval"],
        "rescale_t": args["rescale_t"],
        "guidance_rescale": args["guidance_rescale"],
        "physical_scale": args["physical_scale"],
        "probe": report["probe"],
        "flow": report["flow"],
        "integration": report["integration"],
        "controls": list(POINT_CONTROL_NAMES),
        "decision_thresholds": {
            key: args[key]
            for key in (
                "min_object_win_rate",
                "min_positive_seed_count",
                "max_global_iou_mean_degradation",
                "max_global_precision_mean_degradation",
                "max_outside_iou_mean_degradation",
                "max_component_count_mean_increase",
                "max_largest_component_ratio_mean_degradation",
            )
        },
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Point-only Local-anchor V2 Rollout Multi-seed Summary",
        "",
        f"- Decision: **{'PASS' if report['passed'] else 'FAIL'}**",
        "",
        "## Checks",
        "",
    ]
    lines.extend(f"- `{key}`: `{value}`" for key, value in report["checks"].items())
    lines.extend(["", "## Training Seeds", ""])
    for seed, row in report["seed_results"].items():
        lines.append(
            f"- seed `{seed}`: passed=`{row['passed']}`, "
            f"local IoU mean=`{row['correct_local_iou_mean']:.8g}`, "
            f"win=`{row['correct_local_iou_win']:.4f}`"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    expected = [
        int(item.strip())
        for item in str(args.expected_training_seeds).split(",")
        if item.strip()
    ]
    if not expected or len(expected) != len(set(expected)):
        raise ValueError("expected training seeds must be unique")
    reports = [load_report(path) for path in args.report_dirs]
    by_seed = {int(report["training_seed"]): report for report in reports}
    if len(by_seed) != len(reports):
        raise ValueError("duplicate training seed rollout report")
    protocol_reference = protocol(reports[0])
    protocol_consistent = all(
        protocol(report) == protocol_reference for report in reports[1:]
    )
    checks = {
        "expected_training_seeds_present": sorted(by_seed) == sorted(expected),
        "protocol_consistent": protocol_consistent,
        "all_reports_pass": all(bool(report["passed"]) for report in reports),
        "all_stock_rollouts_bit_exact": all(
            bool(report["decision"]["checks"]["native_stock_rollout_bit_exact"])
            for report in reports
        ),
        "all_correct_local_iou_ci_lows_positive": all(
            float(
                report["comparisons"]["correct_vs_stock"]["local_iou"][
                    "object_bootstrap_95_ci"
                ][0]
            )
            > 0.0
            for report in reports
        ),
        "all_controls_local_iou_ci_lows_positive": all(
            float(
                report["comparisons"]["correct_vs_controls"][control][
                    "local_iou"
                ]["object_bootstrap_95_ci"][0]
            )
            > 0.0
            for report in reports
            for control in POINT_CONTROL_NAMES
        ),
    }
    seed_results = {
        str(seed): {
            "passed": bool(report["passed"]),
            "correct_local_iou_mean": float(
                report["comparisons"]["correct_vs_stock"]["local_iou"][
                    "object"
                ]["mean"]
            ),
            "correct_local_iou_median": float(
                report["comparisons"]["correct_vs_stock"]["local_iou"][
                    "object"
                ]["median"]
            ),
            "correct_local_iou_win": float(
                report["comparisons"]["correct_vs_stock"]["local_iou"][
                    "object_win_rate"
                ]
            ),
            "correct_global_iou_mean": float(
                report["comparisons"]["correct_vs_stock"]["global_iou"][
                    "object"
                ]["mean"]
            ),
            "report": report["_report_path"],
        }
        for seed, report in sorted(by_seed.items())
    }
    summary = {
        "stage": "Point-only local-anchor v2 rollout multi-training-seed summary",
        "passed": all(checks.values()),
        "checks": checks,
        "expected_training_seeds": expected,
        "protocol": protocol_reference,
        "seed_results": seed_results,
    }
    (output_dir / "report.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    write_markdown(summary, output_dir / "report.md")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
