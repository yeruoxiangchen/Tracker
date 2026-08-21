#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, median
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize held-out correspondence reports across training seeds."
    )
    parser.add_argument("--report_dirs", nargs="+", required=True)
    parser.add_argument("--expected_train_seeds", default="42,43,44")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--require_all_pass", action="store_true")
    return parser.parse_args()


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": mean(values),
        "median": median(values),
        "min": min(values),
        "max": max(values),
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    expected = [int(item.strip()) for item in args.expected_train_seeds.split(",") if item.strip()]
    reports: list[dict[str, Any]] = []
    for directory in args.report_dirs:
        path = Path(directory) / "report.json"
        report = json.loads(path.read_text(encoding="utf-8"))
        checkpoint_path = Path(report["protocol"]["checkpoint"])
        checkpoint = __import__("torch").load(checkpoint_path, map_location="cpu")
        train_seed = int(checkpoint.get("args", {}).get("seed", -1))
        reports.append(
            {
                "train_seed": train_seed,
                "path": str(path),
                "passed": bool(report.get("passed", False)),
                "highest_view_count": int(report["highest_view_count"]),
                "required_checks": report["required_checks"],
                "protocol": report["protocol"],
            }
        )
    seeds = sorted(row["train_seed"] for row in reports)
    protocol_fields = (
        "cache_manifest",
        "indices",
        "negative_modes",
        "required_modes",
        "requested_view_counts",
        "available_view_counts",
        "neighborhood_radius",
        "min_source_views",
    )
    protocol_consistent = True
    if reports:
        reference = reports[0]["protocol"]
        protocol_consistent = all(
            all(row["protocol"].get(key) == reference.get(key) for key in protocol_fields)
            for row in reports[1:]
        )
    mode_summary: dict[str, Any] = {}
    if reports:
        modes = list(reports[0]["required_checks"])
        for mode in modes:
            wins = [
                float(row["required_checks"][mode]["metrics"]["object_win_rate"])
                for row in reports
            ]
            advantages = [
                float(
                    row["required_checks"][mode]["metrics"]["advantage"]["mean"]
                )
                for row in reports
            ]
            visual_residuals = [
                float(
                    row["required_checks"][mode]["metrics"]
                    ["visual_over_geometry_advantage"]["mean"]
                )
                for row in reports
            ]
            mode_summary[mode] = {
                "pass_count": sum(
                    bool(row["required_checks"][mode]["passed"]) for row in reports
                ),
                "object_win_rate": summarize(wins),
                "advantage_mean": summarize(advantages),
                "visual_over_geometry_advantage": summarize(visual_residuals),
            }
    all_pass = all(row["passed"] for row in reports)
    passed = (
        seeds == sorted(expected)
        and protocol_consistent
        and (all_pass if args.require_all_pass else sum(row["passed"] for row in reports) >= 2)
    )
    report = {
        "stage": "C1 held-out correspondence multiseed summary",
        "passed": passed,
        "expected_train_seeds": expected,
        "observed_train_seeds": seeds,
        "protocol_consistent": protocol_consistent,
        "all_reports_passed": all_pass,
        "reports": reports,
        "mode_summary": mode_summary,
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
