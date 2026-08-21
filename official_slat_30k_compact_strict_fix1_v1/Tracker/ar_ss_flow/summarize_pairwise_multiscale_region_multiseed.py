#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

MODES = ("pose_cyclic1", "pose_cyclic2", "pose_reverse")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize frozen C1.6 multi-scale region gate results across seeds."
    )
    parser.add_argument("--runs_root", required=True)
    parser.add_argument("--run_pattern", default="corr_pairwise_v3_s200_seed{seed}")
    parser.add_argument(
        "--calibration_relpath",
        default="deployment_multiscale_region_calibration_v4/report.json",
    )
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--min_mean_gain_vs_object", type=float, default=0.001)
    parser.add_argument("--min_mean_win_vs_object", type=float, default=0.60)
    parser.add_argument("--min_mean_gain_vs_wrong_gate", type=float, default=0.0)
    parser.add_argument("--min_mean_win_vs_wrong_gate", type=float, default=0.60)
    parser.add_argument("--min_mean_spatial_correlation", type=float, default=0.05)
    parser.add_argument("--fail_on_error", action="store_true")
    return parser.parse_args()


def load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    if not seeds:
        raise ValueError("at least one seed is required")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)

    reports: dict[int, dict[str, Any]] = {}
    paths: dict[int, str] = {}
    for seed in seeds:
        path = Path(args.runs_root) / args.run_pattern.format(seed=seed) / args.calibration_relpath
        reports[seed] = load(path)
        paths[seed] = str(path)

    reference = reports[seeds[0]]["selected_candidate"]
    same_candidate = all(reports[seed]["selected_candidate"] == reference for seed in seeds)
    seed_passes = {str(seed): bool(reports[seed]["passed"]) for seed in seeds}

    per_mode: dict[str, Any] = {}
    mode_passes: list[bool] = []
    for mode in MODES:
        wrong_gain_object = []
        wrong_win_object = []
        wrong_gain_wrong_gate = []
        wrong_win_wrong_gate = []
        shuffle_gain_object = []
        shuffle_gain_shuffle_gate = []
        wrong_corr = []
        shuffle_corr = []
        deployment_win = []
        for seed in seeds:
            row = reports[seed]["fresh_selected_result"]["per_mode"][mode]
            dep = row["deployment"]
            probe = row["probe"]
            deployment_win.append(float(dep["correct_greater_wrong_object_rate"]))
            wrong_gain_object.append(float(probe["wrong_gain_vs_object"]["mean"]))
            wrong_win_object.append(float(probe["wrong_gain_vs_object_object_win_rate"]))
            wrong_gain_wrong_gate.append(float(probe["wrong_gain_vs_wrong_gate"]["mean"]))
            wrong_win_wrong_gate.append(float(probe["wrong_gain_vs_wrong_gate_object_win_rate"]))
            shuffle_gain_object.append(float(probe["shuffle_gain_vs_object"]["mean"]))
            shuffle_gain_shuffle_gate.append(float(probe["shuffle_gain_vs_shuffle_gate"]["mean"]))
            wrong_corr.append(float(probe["wrong_spatial_rank_correlation"]["mean"]))
            shuffle_corr.append(float(probe["shuffle_spatial_rank_correlation"]["mean"]))
        criteria = {
            "mean_wrong_gain_vs_object": float(np.mean(wrong_gain_object)) >= float(args.min_mean_gain_vs_object),
            "mean_wrong_win_vs_object": float(np.mean(wrong_win_object)) >= float(args.min_mean_win_vs_object),
            "mean_wrong_gain_vs_wrong_gate": float(np.mean(wrong_gain_wrong_gate)) > float(args.min_mean_gain_vs_wrong_gate),
            "mean_wrong_win_vs_wrong_gate": float(np.mean(wrong_win_wrong_gate)) >= float(args.min_mean_win_vs_wrong_gate),
            "mean_shuffle_gain_vs_object": float(np.mean(shuffle_gain_object)) >= float(args.min_mean_gain_vs_object),
            "mean_shuffle_gain_vs_shuffle_gate": float(np.mean(shuffle_gain_shuffle_gate)) > 0.0,
            "mean_wrong_spatial_correlation": float(np.mean(wrong_corr)) >= float(args.min_mean_spatial_correlation),
            "mean_shuffle_spatial_correlation": float(np.mean(shuffle_corr)) >= float(args.min_mean_spatial_correlation),
        }
        passed = all(criteria.values())
        mode_passes.append(passed)
        per_mode[mode] = {
            "passed": passed,
            "criteria": criteria,
            "mean_deployment_object_win": float(np.mean(deployment_win)),
            "mean_wrong_gain_vs_object": float(np.mean(wrong_gain_object)),
            "mean_wrong_win_vs_object": float(np.mean(wrong_win_object)),
            "mean_wrong_gain_vs_wrong_gate": float(np.mean(wrong_gain_wrong_gate)),
            "mean_wrong_win_vs_wrong_gate": float(np.mean(wrong_win_wrong_gate)),
            "mean_shuffle_gain_vs_object": float(np.mean(shuffle_gain_object)),
            "mean_shuffle_gain_vs_shuffle_gate": float(np.mean(shuffle_gain_shuffle_gate)),
            "mean_wrong_spatial_correlation": float(np.mean(wrong_corr)),
            "mean_shuffle_spatial_correlation": float(np.mean(shuffle_corr)),
            "by_seed": {
                str(seed): {
                    "passed": bool(reports[seed]["fresh_selected_result"]["per_mode"][mode]["passed"]),
                    "selection_mode": reports[seed]["selection_mode"],
                }
                for seed in seeds
            },
        }

    global_checks = {
        "same_frozen_candidate": same_candidate,
        "all_seed_reports_passed": all(seed_passes.values()),
        "all_modes_passed": all(mode_passes),
    }
    passed = all(global_checks.values())
    report = {
        "stage": "C1.6 multi-scale region gate multi-seed summary",
        "passed": passed,
        "args": vars(args),
        "seeds": seeds,
        "report_paths": paths,
        "reference_candidate": reference,
        "seed_passes": seed_passes,
        "per_mode": per_mode,
        "global_checks": global_checks,
        "c2_recommendation": (
            "Use the seed42 checkpoint and the frozen seed42 region candidate for a small C2 flow mechanism gate."
            if passed
            else "Do not enter region-gated C2. Fall back to object-scalar calibration or keep correspondence diagnostic-only."
        ),
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("=" * 100)
    print(report["stage"])
    print("reference_candidate:", reference)
    for mode in MODES:
        row = per_mode[mode]
        print(
            f"{mode}: passed={row['passed']} "
            f"vs_object={row['mean_wrong_gain_vs_object']:+.6f} "
            f"win_object={row['mean_wrong_win_vs_object']:.4f} "
            f"vs_wrong_gate={row['mean_wrong_gain_vs_wrong_gate']:+.6f} "
            f"win_wrong={row['mean_wrong_win_vs_wrong_gate']:.4f} "
            f"shuffle_vs_object={row['mean_shuffle_gain_vs_object']:+.6f} "
            f"corr={row['mean_wrong_spatial_correlation']:+.4f}"
        )
    print("global_checks:", global_checks)
    print("passed:", passed)
    print("report:", output_dir / "report.json")
    if args.fail_on_error and not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
