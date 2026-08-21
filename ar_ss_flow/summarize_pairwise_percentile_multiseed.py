#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

MODES = ("pose_cyclic1", "pose_cyclic2", "pose_reverse")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize C1.5-v3 percentile gates across seeds.")
    parser.add_argument("--runs_root", required=True)
    parser.add_argument("--run_pattern", default="corr_pairwise_v3_s200_seed{seed}")
    parser.add_argument(
        "--calibration_relpath",
        default="deployment_percentile_calibration_v3/report.json",
    )
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_fraction_spread", type=float, default=0.10)
    parser.add_argument("--min_mean_deployment_object_win_rate", type=float, default=0.80)
    parser.add_argument("--min_mean_gain_vs_random", type=float, default=0.001)
    parser.add_argument("--min_mean_object_win_vs_random", type=float, default=0.65)
    parser.add_argument("--min_mean_gain_vs_wrong_selector", type=float, default=0.0)
    parser.add_argument("--min_mean_shuffle_gain_vs_shuffle_selector", type=float, default=0.0)
    parser.add_argument("--fail_on_error", action="store_true")
    return parser.parse_args()


def load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    args = parse_args()
    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    if not seeds:
        raise ValueError("at least one seed is required")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)

    reports: dict[int, dict[str, Any]] = {}
    paths: dict[int, str] = {}
    for seed in seeds:
        path = (
            Path(args.runs_root)
            / args.run_pattern.format(seed=seed)
            / args.calibration_relpath
        )
        if not path.is_file():
            raise FileNotFoundError(path)
        reports[seed] = load(path)
        paths[seed] = str(path)

    fractions = [float(reports[seed]["selected_fraction"]) for seed in seeds]
    fraction_spread = max(fractions) - min(fractions)
    seed_passes = {str(seed): bool(reports[seed]["passed"]) for seed in seeds}

    per_mode: dict[str, Any] = {}
    mode_passes: list[bool] = []
    for mode in MODES:
        deployment_win = []
        gain_random = []
        win_random = []
        gain_wrong_selector = []
        shuffle_gain_selector = []
        selected_advantage = []
        for seed in seeds:
            row = reports[seed]["fresh_selected_result"]["per_mode"][mode]
            wrong = row["wrong_pose_label"]
            shuffle = row["shuffle_label"]
            deployment_win.append(
                float(row["deployment_metrics"]["correct_greater_wrong_rate"])
            )
            gain_random.append(float(wrong["correct_minus_random"]["mean"]))
            win_random.append(float(wrong["correct_greater_random_object_rate"]))
            gain_wrong_selector.append(
                float(wrong["correct_minus_wrong_selector"]["mean"])
            )
            shuffle_gain_selector.append(
                float(shuffle["correct_minus_shuffle_selector"]["mean"])
            )
            selected_advantage.append(
                float(wrong["correct_selector_advantage"]["mean"])
            )
        criteria = {
            "mean_deployment_object_win_rate": (
                float(np.mean(deployment_win))
                >= float(args.min_mean_deployment_object_win_rate)
            ),
            "mean_selected_advantage_positive": float(np.mean(selected_advantage)) > 0.0,
            "mean_gain_vs_random": (
                float(np.mean(gain_random)) >= float(args.min_mean_gain_vs_random)
            ),
            "mean_object_win_vs_random": (
                float(np.mean(win_random))
                >= float(args.min_mean_object_win_vs_random)
            ),
            "mean_gain_vs_wrong_selector": (
                float(np.mean(gain_wrong_selector))
                > float(args.min_mean_gain_vs_wrong_selector)
            ),
            "mean_shuffle_gain_vs_shuffle_selector": (
                float(np.mean(shuffle_gain_selector))
                > float(args.min_mean_shuffle_gain_vs_shuffle_selector)
            ),
        }
        passed = all(criteria.values())
        mode_passes.append(passed)
        per_mode[mode] = {
            "passed": passed,
            "criteria": criteria,
            "mean_deployment_object_win_rate": float(np.mean(deployment_win)),
            "mean_selected_advantage": float(np.mean(selected_advantage)),
            "mean_gain_vs_random": float(np.mean(gain_random)),
            "mean_object_win_vs_random": float(np.mean(win_random)),
            "mean_gain_vs_wrong_selector": float(np.mean(gain_wrong_selector)),
            "mean_shuffle_gain_vs_shuffle_selector": float(
                np.mean(shuffle_gain_selector)
            ),
            "by_seed": {
                str(seed): {
                    "selected_fraction": float(reports[seed]["selected_fraction"]),
                    "passed": bool(
                        reports[seed]["fresh_selected_result"]["per_mode"][mode][
                            "passed"
                        ]
                    ),
                }
                for seed in seeds
            },
        }

    global_checks = {
        "all_seed_reports_passed": all(seed_passes.values()),
        "fraction_spread": fraction_spread <= float(args.max_fraction_spread),
        "all_modes_passed": all(mode_passes),
    }
    passed = all(global_checks.values())
    report = {
        "stage": "C1.5-v3 percentile gate multi-seed summary",
        "passed": passed,
        "args": vars(args),
        "seeds": seeds,
        "report_paths": paths,
        "seed_passes": seed_passes,
        "selected_fractions": {str(seed): fractions[index] for index, seed in enumerate(seeds)},
        "fraction_spread": fraction_spread,
        "per_mode": per_mode,
        "global_checks": global_checks,
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    print("=" * 100)
    print(report["stage"])
    print("selected_fractions:", report["selected_fractions"])
    print(f"fraction_spread={fraction_spread:.4f}")
    for mode in MODES:
        row = per_mode[mode]
        print(
            f"{mode}: passed={row['passed']} "
            f"obj_win={row['mean_deployment_object_win_rate']:.4f} "
            f"selected_adv={row['mean_selected_advantage']:+.6f} "
            f"vs_random={row['mean_gain_vs_random']:+.6f} "
            f"win_random={row['mean_object_win_vs_random']:.4f} "
            f"vs_wrong={row['mean_gain_vs_wrong_selector']:+.6f} "
            f"shuffle_vs_shuffle={row['mean_shuffle_gain_vs_shuffle_selector']:+.6f}"
        )
    print("global_checks:", global_checks)
    print("passed:", passed)
    print("report:", output_dir / "report.json")

    if args.fail_on_error and not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
