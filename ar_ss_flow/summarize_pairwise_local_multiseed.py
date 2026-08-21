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
        description="Summarize C1.5-v2 local pairwise confidence calibration across seeds."
    )
    parser.add_argument("--runs_root", required=True)
    parser.add_argument("--run_pattern", default="corr_pairwise_v3_s200_seed{seed}")
    parser.add_argument(
        "--calibration_relpath",
        default="deployment_local_calibration_v2/report.json",
    )
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_tau_low_spread", type=float, default=0.10)
    parser.add_argument("--max_tau_high_spread", type=float, default=0.10)
    parser.add_argument("--min_mean_auc", type=float, default=0.70)
    parser.add_argument("--min_mean_object_win_rate", type=float, default=0.80)
    parser.add_argument("--min_mean_auc_gain_vs_raw", type=float, default=0.03)
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

    root = Path(args.runs_root)
    reports: dict[int, dict[str, Any]] = {}
    for seed in seeds:
        run = root / args.run_pattern.format(seed=seed)
        reports[seed] = load(run / args.calibration_relpath)

    methods = [str(report["selected_method"]) for report in reports.values()]
    tau_low = [float(report["calibration"]["tau_low"]) for report in reports.values()]
    tau_high = [float(report["calibration"]["tau_high"]) for report in reports.values()]
    tau_low_spread = max(tau_low) - min(tau_low)
    tau_high_spread = max(tau_high) - min(tau_high)

    seed_rows: dict[str, Any] = {}
    for seed, report in reports.items():
        seed_rows[str(seed)] = {
            "passed": bool(report.get("passed", False)),
            "selected_method": str(report["selected_method"]),
            "tau_low": float(report["calibration"]["tau_low"]),
            "tau_high": float(report["calibration"]["tau_high"]),
            "fresh_mean_auc_gain_vs_raw": float(
                report["selected_vs_raw"]["fresh_mean_auc_correct_vs_wrong_gain"]
            ),
        }

    mode_summary: dict[str, Any] = {}
    mode_checks: dict[str, bool] = {}
    for mode in MODES:
        selected_aucs: list[float] = []
        raw_aucs: list[float] = []
        object_wins: list[float] = []
        selectivities: list[float] = []
        mode_passes: list[bool] = []
        for report in reports.values():
            selected = str(report["selected_method"])
            selected_row = report["fresh_method_comparison"][selected]["per_mode"][mode]
            raw_row = report["fresh_method_comparison"]["raw"]["per_mode"][mode]
            selected_aucs.append(
                float(selected_row["voxel_metrics"]["auc_correct_vs_wrong"])
            )
            raw_aucs.append(float(raw_row["voxel_metrics"]["auc_correct_vs_wrong"]))
            object_wins.append(
                float(
                    selected_row["object_metrics"][
                        "deployment_correct_greater_wrong_rate"
                    ]
                )
            )
            selectivities.append(
                float(
                    selected_row["voxel_metrics"][
                        "correct_minus_wrong_gate_coverage"
                    ]
                )
            )
            mode_passes.append(bool(selected_row["passed"]))
        auc_gains = [a - b for a, b in zip(selected_aucs, raw_aucs)]
        row = {
            "seed_pass_count": sum(mode_passes),
            "seed_count": len(seeds),
            "mean_selected_auc": mean(selected_aucs),
            "min_selected_auc": min(selected_aucs),
            "mean_raw_auc": mean(raw_aucs),
            "mean_auc_gain_vs_raw": mean(auc_gains),
            "min_auc_gain_vs_raw": min(auc_gains),
            "mean_deployment_object_win_rate": mean(object_wins),
            "min_deployment_object_win_rate": min(object_wins),
            "mean_gate_selectivity": mean(selectivities),
        }
        row["passed"] = (
            all(mode_passes)
            and row["mean_selected_auc"] >= float(args.min_mean_auc)
            and row["mean_auc_gain_vs_raw"] >= float(args.min_mean_auc_gain_vs_raw)
            and row["mean_deployment_object_win_rate"]
            >= float(args.min_mean_object_win_rate)
        )
        mode_summary[mode] = row
        mode_checks[mode] = bool(row["passed"])

    global_checks = {
        "all_seed_reports_passed": all(
            bool(report.get("passed", False)) for report in reports.values()
        ),
        "same_selected_method": len(set(methods)) == 1,
        "selected_method_is_local": all(method != "raw" for method in methods),
        "tau_low_stable": tau_low_spread <= float(args.max_tau_low_spread),
        "tau_high_stable": tau_high_spread <= float(args.max_tau_high_spread),
        "all_modes_passed": all(mode_checks.values()),
    }
    passed = all(global_checks.values())

    report = {
        "stage": "C1.5-v2 local visual-only deployment calibration multi-seed summary",
        "passed": passed,
        "args": vars(args),
        "seeds": seeds,
        "seed_rows": seed_rows,
        "selected_methods": methods,
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
            "calibration_policy": "use seed42 train-selected local method and thresholds",
            "multi_seed_median_is_diagnostic_only": True,
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
            f"seed={seed} passed={row['passed']} method={row['selected_method']} "
            f"tau_low={row['tau_low']:.6f} tau_high={row['tau_high']:.6f} "
            f"auc_gain={row['fresh_mean_auc_gain_vs_raw']:+.4f}"
        )
    print(
        f"tau_low_spread={tau_low_spread:.6f} "
        f"tau_high_spread={tau_high_spread:.6f}"
    )
    for mode, row in mode_summary.items():
        print(
            f"{mode}: passed={row['passed']} auc={row['mean_selected_auc']:.4f} "
            f"gain={row['mean_auc_gain_vs_raw']:+.4f} "
            f"object_win={row['mean_deployment_object_win_rate']:.4f}"
        )
    print("global_checks:", global_checks)
    print("passed:", passed)
    print("report:", output_dir / "report.json")

    if args.fail_on_error and not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
