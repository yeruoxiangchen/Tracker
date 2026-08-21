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
        description="Summarize self-referenced object scalar calibration across seeds."
    )
    parser.add_argument("--runs_root", required=True)
    parser.add_argument("--run_pattern", default="corr_pairwise_v3_s200_seed{seed}")
    parser.add_argument(
        "--calibration_relpath",
        default="deployment_object_selfref_calibration_v6/report.json",
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

    root = Path(args.runs_root)
    reports = {
        seed: load(root / args.run_pattern.format(seed=seed) / args.calibration_relpath)
        for seed in seeds
    }
    paths = {
        str(seed): str(root / args.run_pattern.format(seed=seed) / args.calibration_relpath)
        for seed in seeds
    }
    reference_config = reports[seeds[0]]["selected_config"]
    same_config = all(report["selected_config"] == reference_config for report in reports.values())
    tau_low = [float(report["calibration"]["tau_low"]) for report in reports.values()]
    tau_high = [float(report["calibration"]["tau_high"]) for report in reports.values()]

    per_mode: dict[str, Any] = {}
    mode_passes: list[bool] = []
    for mode in MODES:
        auc_wrong: list[float] = []
        auc_shuffle: list[float] = []
        score_win_wrong: list[float] = []
        score_win_shuffle: list[float] = []
        gate_gap_wrong: list[float] = []
        gate_gap_shuffle: list[float] = []
        for seed in seeds:
            row = reports[seed]["fresh_per_mode"][mode]
            dep = row["deployment"]
            gate = dep["gate"]
            auc_wrong.append(float(dep["auc_correct_vs_wrong"]))
            auc_shuffle.append(float(dep["auc_correct_vs_shuffle"]))
            score_win_wrong.append(float(dep["correct_greater_wrong_rate"]))
            score_win_shuffle.append(float(dep["correct_greater_shuffle_rate"]))
            gate_gap_wrong.append(float(gate["correct_minus_wrong"]["mean"]))
            gate_gap_shuffle.append(float(gate["correct_minus_shuffle"]["mean"]))
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
        }

    tau_low_spread = max(tau_low) - min(tau_low)
    tau_high_spread = max(tau_high) - min(tau_high)
    global_checks = {
        "same_frozen_config": same_config,
        "all_seed_reports_passed": all(bool(report["passed"]) for report in reports.values()),
        "tau_low_stable": tau_low_spread <= float(args.max_tau_low_spread),
        "tau_high_stable": tau_high_spread <= float(args.max_tau_high_spread),
        "all_modes_passed": all(mode_passes),
    }
    passed = all(global_checks.values())
    report = {
        "stage": "C1.6 self-referenced object scalar multi-seed summary",
        "passed": passed,
        "args": vars(args),
        "seeds": seeds,
        "report_paths": paths,
        "reference_config": reference_config,
        "seed_passes": {str(seed): bool(reports[seed]["passed"]) for seed in seeds},
        "tau_low": {"values": tau_low, "mean": mean(tau_low), "median": median(tau_low), "spread": tau_low_spread},
        "tau_high": {"values": tau_high, "mean": mean(tau_high), "median": median(tau_high), "spread": tau_high_spread},
        "per_mode": per_mode,
        "global_checks": global_checks,
        "c2_recommendation": (
            "Use the frozen self-reference config for a small object-gated C2 mechanism audit."
            if passed
            else "Do not enter formal object-gated C2. Keep correspondence diagnostic-only or redesign confidence training."
        ),
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("=" * 100)
    print(report["stage"])
    print("reference_config:", reference_config)
    for mode in MODES:
        row = per_mode[mode]
        print(
            f"{mode}: passed={row['passed']} "
            f"auc_wrong={row['mean_auc_correct_vs_wrong']:.4f} "
            f"auc_shuffle={row['mean_auc_correct_vs_shuffle']:.4f} "
            f"min_win_wrong={row['min_score_win_correct_vs_wrong']:.4f} "
            f"min_win_shuffle={row['min_score_win_correct_vs_shuffle']:.4f} "
            f"gap_wrong={row['mean_gate_gap_correct_vs_wrong']:+.4f} "
            f"gap_shuffle={row['mean_gate_gap_correct_vs_shuffle']:+.4f}"
        )
    print("global_checks:", global_checks)
    print("passed:", passed)
    print("report:", output_dir / "report.json")
    if args.fail_on_error and not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
