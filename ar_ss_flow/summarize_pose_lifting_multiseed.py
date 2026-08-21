#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
import sys
from typing import Any

import torch

TRACKER_ROOT = Path(__file__).resolve().parents[1]
if str(TRACKER_ROOT) not in sys.path:
    sys.path.insert(0, str(TRACKER_ROOT))

CORRUPTION_MODES = ("pose_perturb", "pose_shuffle", "depth_corrupt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize P4 reports across training seeds.")
    parser.add_argument("--report_dirs", nargs="+", required=True)
    parser.add_argument("--expected_train_seeds", default="42,43,44")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--fail_on_error", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    expected = {int(item) for item in args.expected_train_seeds.split(",") if item.strip()}
    reports: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for directory in args.report_dirs:
        path = Path(directory) / "report.json"
        if not path.exists():
            raise FileNotFoundError(path)
        report = json.loads(path.read_text(encoding="utf-8"))
        train_seed = int(report.get("train_seed", -1))
        if train_seed < 0:
            checkpoint = Path(report["checkpoint"])
            checkpoint_payload = torch.load(checkpoint, map_location="cpu")
            train_seed = int(checkpoint_payload.get("args", {}).get("seed", -1))
        row = {
            "train_seed": train_seed,
            "report": str(path),
            "passed": bool(report.get("passed", False)),
            "correct_gain_mean": float(report["correct_gain_vs_stock"]["mean"]),
            "correct_gain_median": float(report["correct_gain_vs_stock"]["median"]),
            "positive_t_count": int(report["positive_t_count"]),
            "object_win_rates": report["object_win_rates_correct_vs_corruption"],
            "null_max_abs_diff": float(report["null_max_abs_diff"]),
        }
        rows.append(row)
        reports.append(report)
    seeds = {row["train_seed"] for row in rows}
    duplicate = len(seeds) != len(rows)
    protocol_reference = reports[0].get("protocol") if reports else None
    protocol_consistent = bool(protocol_reference) and all(
        report.get("protocol") == protocol_reference for report in reports
    )
    checks = {
        "expected_seeds_present": seeds == expected,
        "no_duplicate_train_seed": not duplicate,
        "every_seed_passed": all(row["passed"] for row in rows),
        "gain_sign_consistent": all(
            row["correct_gain_mean"] > 0.0 and row["correct_gain_median"] > 0.0
            for row in rows
        ),
        "null_bit_exact_every_seed": all(row["null_max_abs_diff"] == 0.0 for row in rows),
        "corruption_direction_consistent": all(
            all(
                report["comparisons"][mode]["correct_gain_difference"]["mean"] > 0.0
                for mode in CORRUPTION_MODES
            )
            for report in reports
        ),
        "evaluation_protocol_identical": protocol_consistent,
    }
    report = {
        "passed": all(checks.values()),
        "checks": checks,
        "expected_train_seeds": sorted(expected),
        "observed_train_seeds": sorted(seeds),
        "protocol": protocol_reference,
        "mean_correct_gain_across_train_seeds": mean(
            row["correct_gain_mean"] for row in rows
        ),
        "runs": sorted(rows, key=lambda row: row["train_seed"]),
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = [
        "# P4 Multi-seed Summary",
        "",
        f"- passed: `{report['passed']}`",
        f"- expected / observed seeds: `{sorted(expected)} / {sorted(seeds)}`",
        f"- mean correct gain: `{report['mean_correct_gain_across_train_seeds']:.6g}`",
        "",
        "| train seed | run pass | correct gain mean | median | positive t | null max diff |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["runs"]:
        lines.append(
            f"| {row['train_seed']} | {row['passed']} | {row['correct_gain_mean']:.6g} | "
            f"{row['correct_gain_median']:.6g} | {row['positive_t_count']} | "
            f"{row['null_max_abs_diff']:.6g} |"
        )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if args.fail_on_error and not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
