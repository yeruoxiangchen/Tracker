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
        description="Summarize C1.6 object-level scalar gate calibration across seeds."
    )
    parser.add_argument("--runs_root", required=True)
    parser.add_argument("--run_pattern", default="corr_pairwise_v3_s200_seed{seed}")
    parser.add_argument(
        "--calibration_relpath",
        default="deployment_object_scalar_calibration_v5/report.json",
    )
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_tau_low_spread", type=float, default=0.05)
    parser.add_argument("--max_tau_high_spread", type=float, default=0.05)
    parser.add_argument("--min_mean_auc", type=float, default=0.85)
    parser.add_argument("--min_min_score_win_rate", type=float, default=0.80)
    parser.add_argument("--min_mean_gate_gap", type=float, default=0.10)
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
    paths: dict[int, str] = {}
    root = Path(args.runs_root)
    for seed in seeds:
        path = root / args.run_pattern.format(seed=seed) / args.calibration_relpath
        reports[seed] = load(path)
        paths[seed] = str(path)

    statistic = reports[seeds[0]]["selected_statistic"]
    same_statistic = all(report["selected_statistic"] == statistic for report in reports.values())
    tau_low = [float(report["calibration"]["tau_low"]) for report in reports.values()]
    tau_high = [float(report["calibration"]["tau_high"]) for report in reports.values()]
    tau_low_spread = max(tau_low) - min(tau_low)
    tau_high_spread = max(tau_high) - min(tau_high)

    seed_rows = {
        str(seed): {
            "passed": bool(reports[seed]["passed"]),
            "selection_mode": reports[seed]["selection_mode"],
            "tau_low": float(reports[seed]["calibration"]["tau_low"]),
            "tau_high": float(reports[seed]["calibration"]["tau_high"]),
            "selected_train_score": float(reports[seed]["selected_train_score"]),
        }
        for seed in seeds
    }

    per_mode: dict[str, Any] = {}
    mode_passes: list[bool] = []
    for mode in MODES:
        auc_wrong = []
        auc_shuffle = []
        score_win_wrong = []
        score_win_shuffle = []
        gate_gap_wrong = []
        gate_gap_shuffle = []
        correct_gate = []
        for seed in seeds:
            dep = reports[seed]["fresh_per_mode"][mode]["deployment"]
            gate = dep["gate"]
            auc_wrong.append(float(dep["auc_correct_vs_wrong"]))
            auc_shuffle.append(float(dep["auc_correct_vs_shuffle"]))
            score_win_wrong.append(float(dep["correct_greater_wrong_rate"]))
            score_win_shuffle.append(float(dep["correct_greater_shuffle_rate"]))
            gate_gap_wrong.append(float(gate["correct_minus_wrong"]["mean"]))
            gate_gap_shuffle.append(float(gate["correct_minus_shuffle"]["mean"]))
            correct_gate.append(float(gate["correct"]["mean"]))
        checks = {
            "all_seed_modes_passed": all(
                bool(reports[seed]["fresh_per_mode"][mode]["passed"])
                for seed in seeds
            ),
            "mean_auc_correct_vs_wrong": mean(auc_wrong) >= float(args.min_mean_auc),
            "mean_auc_correct_vs_shuffle": mean(auc_shuffle) >= float(args.min_mean_auc),
            "min_score_win_correct_vs_wrong": min(score_win_wrong) >= float(args.min_min_score_win_rate),
            "min_score_win_correct_vs_shuffle": min(score_win_shuffle) >= float(args.min_min_score_win_rate),
            "mean_gate_gap_correct_vs_wrong": mean(gate_gap_wrong) >= float(args.min_mean_gate_gap),
            "mean_gate_gap_correct_vs_shuffle": mean(gate_gap_shuffle) >= float(args.min_mean_gate_gap),
        }
        passed = all(checks.values())
        mode_passes.append(passed)
        per_mode[mode] = {
            "passed": passed,
            "checks": checks,
            "mean_auc_correct_vs_wrong": mean(auc_wrong),
            "min_auc_correct_vs_wrong": min(auc_wrong),
            "mean_auc_correct_vs_shuffle": mean(auc_shuffle),
            "min_auc_correct_vs_shuffle": min(auc_shuffle),
            "mean_score_win_correct_vs_wrong": mean(score_win_wrong),
            "min_score_win_correct_vs_wrong": min(score_win_wrong),
            "mean_score_win_correct_vs_shuffle": mean(score_win_shuffle),
            "min_score_win_correct_vs_shuffle": min(score_win_shuffle),
            "mean_gate_gap_correct_vs_wrong": mean(gate_gap_wrong),
            "mean_gate_gap_correct_vs_shuffle": mean(gate_gap_shuffle),
            "mean_correct_gate": mean(correct_gate),
        }

    global_checks = {
        "same_frozen_statistic": same_statistic,
        "all_seed_reports_passed": all(bool(report["passed"]) for report in reports.values()),
        "tau_low_stable": tau_low_spread <= float(args.max_tau_low_spread),
        "tau_high_stable": tau_high_spread <= float(args.max_tau_high_spread),
        "all_modes_passed": all(mode_passes),
    }
    passed = all(global_checks.values())
    report = {
        "stage": "C1.6 object-level scalar gate multi-seed summary",
        "passed": passed,
        "args": vars(args),
        "seeds": seeds,
        "report_paths": paths,
        "reference_statistic": statistic,
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
        "per_mode": per_mode,
        "global_checks": global_checks,
        "c2_recommendation": (
            "Use the seed42 checkpoint, seed42 statistic, and seed42 tau_low/tau_high for a small object-gated C2 mechanism audit."
            if passed
            else "Do not enter formal object-gated C2; keep correspondence diagnostic-only or redesign calibration."
        ),
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("=" * 100)
    print(report["stage"])
    print("reference_statistic:", statistic)
    for seed in seeds:
        row = seed_rows[str(seed)]
        print(
            f"seed={seed} passed={row['passed']} selection={row['selection_mode']} "
            f"tau_low={row['tau_low']:.6f} tau_high={row['tau_high']:.6f}"
        )
    print(f"tau_low_spread={tau_low_spread:.6f} tau_high_spread={tau_high_spread:.6f}")
    for mode, row in per_mode.items():
        print(
            f"{mode}: passed={row['passed']} "
            f"auc_wrong={row['mean_auc_correct_vs_wrong']:.4f} "
            f"auc_shuffle={row['mean_auc_correct_vs_shuffle']:.4f} "
            f"win_wrong_min={row['min_score_win_correct_vs_wrong']:.4f} "
            f"win_shuffle_min={row['min_score_win_correct_vs_shuffle']:.4f} "
            f"gate_gap_wrong={row['mean_gate_gap_correct_vs_wrong']:+.4f} "
            f"gate_gap_shuffle={row['mean_gate_gap_correct_vs_shuffle']:+.4f}"
        )
    print("global_checks:", global_checks)
    print("passed:", passed)
    print("report:", output_dir / "report.json")
    if args.fail_on_error and not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
