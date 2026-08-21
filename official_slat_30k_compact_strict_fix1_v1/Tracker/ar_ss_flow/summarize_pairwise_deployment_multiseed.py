#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, median
from typing import Any

MODES = ("pose_cyclic1", "pose_cyclic2", "pose_reverse")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize C1.5 visual-only deployment calibration across seeds."
    )
    parser.add_argument("--runs_root", required=True)
    parser.add_argument(
        "--run_pattern", default="corr_pairwise_v3_s200_seed{seed}"
    )
    parser.add_argument(
        "--calibration_relpath",
        default="deployment_calibration_v1/report.json",
    )
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_tau_low_spread", type=float, default=0.10)
    parser.add_argument("--max_tau_high_spread", type=float, default=0.10)
    parser.add_argument("--min_mean_auc", type=float, default=0.70)
    parser.add_argument("--min_mean_object_win_rate", type=float, default=0.80)
    parser.add_argument("--fail_on_error", action="store_true")
    return parser.parse_args()


def load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    if not seeds:
        raise ValueError("seeds must be non-empty")

    reports: dict[int, dict[str, Any]] = {}
    root = Path(args.runs_root)
    for seed in seeds:
        run = root / args.run_pattern.format(seed=seed)
        reports[seed] = load(run / args.calibration_relpath)

    tau_low = [float(report["calibration"]["tau_low"]) for report in reports.values()]
    tau_high = [float(report["calibration"]["tau_high"]) for report in reports.values()]
    tau_low_spread = max(tau_low) - min(tau_low)
    tau_high_spread = max(tau_high) - min(tau_high)

    seed_rows: dict[str, Any] = {}
    for seed, report in reports.items():
        seed_rows[str(seed)] = {
            "passed": bool(report.get("passed", False)),
            "tau_low": float(report["calibration"]["tau_low"]),
            "tau_high": float(report["calibration"]["tau_high"]),
            "per_mode_threshold_spread": float(
                report.get("per_mode_threshold_spread", 0.0)
            ),
        }

    mode_summary: dict[str, Any] = {}
    mode_checks: dict[str, bool] = {}
    for mode in MODES:
        aucs = [
            float(report["fresh_per_mode"][mode]["voxel_metrics"]["auc_correct_vs_wrong"])
            for report in reports.values()
        ]
        shuffle_aucs = [
            float(report["fresh_per_mode"][mode]["voxel_metrics"]["auc_correct_vs_shuffle"])
            for report in reports.values()
        ]
        object_wins = [
            float(
                report["fresh_per_mode"][mode]["object_metrics"][
                    "deployment_correct_greater_wrong_rate"
                ]
            )
            for report in reports.values()
        ]
        selectivities = [
            float(
                report["fresh_per_mode"][mode]["voxel_metrics"][
                    "correct_minus_wrong_gate_coverage"
                ]
            )
            for report in reports.values()
        ]
        pass_count = sum(
            bool(report["fresh_per_mode"][mode]["passed"])
            for report in reports.values()
        )
        row = {
            "seed_pass_count": pass_count,
            "seed_count": len(seeds),
            "mean_auc_correct_vs_wrong": mean(aucs),
            "min_auc_correct_vs_wrong": min(aucs),
            "mean_auc_correct_vs_shuffle": mean(shuffle_aucs),
            "mean_deployment_object_win_rate": mean(object_wins),
            "min_deployment_object_win_rate": min(object_wins),
            "mean_gate_selectivity": mean(selectivities),
        }
        mode_pass = (
            pass_count == len(seeds)
            and row["mean_auc_correct_vs_wrong"] >= float(args.min_mean_auc)
            and row["mean_deployment_object_win_rate"]
            >= float(args.min_mean_object_win_rate)
        )
        row["passed"] = mode_pass
        mode_summary[mode] = row
        mode_checks[mode] = mode_pass

    global_checks = {
        "all_seed_reports_passed": all(
            bool(report.get("passed", False)) for report in reports.values()
        ),
        "tau_low_stable": tau_low_spread <= float(args.max_tau_low_spread),
        "tau_high_stable": tau_high_spread <= float(args.max_tau_high_spread),
        "all_modes_passed": all(mode_checks.values()),
    }
    passed = all(global_checks.values())

    report = {
        "stage": "C1.5 visual-only deployment calibration multi-seed summary",
        "passed": passed,
        "args": vars(args),
        "seeds": seeds,
        "seed_rows": seed_rows,
        "tau_low": {
            "values": tau_low,
            "mean": mean(tau_low),
            "median": median(tau_low),
            "spread": tau_low_spread,
        },
        "tau_high": {
            "values": tau_high,
            "mean": mean(tau_high),
            "median": median(tau_high),
            "spread": tau_high_spread,
        },
        "mode_summary": mode_summary,
        "global_checks": global_checks,
        "deployment_recommendation": {
            "checkpoint_policy": "use seed42 checkpoint for the first C2 mechanism gate",
            "threshold_policy": "use seed42 train-selected tau_low/tau_high; multi-seed median is diagnostic only",
            "reference_tau_low_median": median(tau_low),
            "reference_tau_high_median": median(tau_high),
        },
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    print("=" * 100)
    print(report["stage"])
    for seed in seeds:
        row = seed_rows[str(seed)]
        print(
            f"seed={seed} passed={row['passed']} "
            f"tau_low={row['tau_low']:.6f} tau_high={row['tau_high']:.6f}"
        )
    print(
        f"tau_low_spread={tau_low_spread:.6f} "
        f"tau_high_spread={tau_high_spread:.6f}"
    )
    for mode, row in mode_summary.items():
        print(
            f"{mode}: passed={row['passed']} "
            f"auc={row['mean_auc_correct_vs_wrong']:.4f} "
            f"object_win={row['mean_deployment_object_win_rate']:.4f} "
            f"selectivity={row['mean_gate_selectivity']:+.4f}"
        )
    print("global_checks:", global_checks)
    print("passed:", passed)
    print("report:", output_dir / "report.json")

    if args.fail_on_error and not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
