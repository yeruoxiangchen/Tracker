#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any

MODES = ("pose_cyclic1", "pose_cyclic2", "pose_reverse")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize C1 pairwise v3 multi-seed reports.")
    parser.add_argument("--runs_root", required=True)
    parser.add_argument("--run_pattern", default="corr_pairwise_v3_s200_seed{seed}")
    parser.add_argument("--eval_relpath", default="eval_fresh48_sources2_3/report.json")
    parser.add_argument("--train_relpath", default="train_report.json")
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--source_view_count", type=int, default=3)
    parser.add_argument("--fail_on_error", action="store_true")
    return parser.parse_args()


def load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    seeds = tuple(int(item.strip()) for item in args.seeds.split(",") if item.strip())
    if not seeds:
        raise ValueError("no seeds")
    source_count = int(args.source_view_count)
    runs_root = Path(args.runs_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)

    seed_rows: list[dict[str, Any]] = []
    for seed in seeds:
        run = runs_root / args.run_pattern.format(seed=seed)
        train = load(run / args.train_relpath)
        evaluation = load(run / args.eval_relpath)
        summary = evaluation["summary"][str(source_count)]
        modes: dict[str, Any] = {}
        passed_modes = 0
        for mode in MODES:
            row = summary[mode]
            metrics = {
                "objects": int(row["object_count"]),
                "reproj_win": float(row["object_win_rate"]),
                "reproj_mean": float(row["advantage"]["mean"]),
                "reproj_median": float(row["advantage"]["median"]),
                "pairwise_win": float(row["pairwise_confidence_win_rate"]),
                "pairwise_mean": float(row["pairwise_confidence_advantage"]["mean"]),
                "pairwise_median": float(row["pairwise_confidence_advantage"]["median"]),
            }
            mode_pass = (
                metrics["reproj_win"] >= 0.65
                and metrics["reproj_mean"] > 0.0
                and metrics["reproj_median"] > 0.0
                and metrics["pairwise_win"] >= 0.80
                and metrics["pairwise_mean"] > 0.0
                and metrics["pairwise_median"] > 0.0
            )
            metrics["passed"] = mode_pass
            passed_modes += int(mode_pass)
            modes[mode] = metrics

        average_reproj_win = mean(modes[mode]["reproj_win"] for mode in MODES)
        seed_rows.append(
            {
                "seed": seed,
                "run": str(run),
                "train_finite": bool(train.get("finite", False)),
                "completed_steps": int(train.get("completed_steps", 0)),
                "evaluation_passed": bool(evaluation.get("passed", False)),
                "passed_mode_count": passed_modes,
                "average_reproj_win": average_reproj_win,
                "modes": modes,
            }
        )

    mode_summary: dict[str, Any] = {}
    for mode in MODES:
        rows = [row["modes"][mode] for row in seed_rows]
        mode_summary[mode] = {
            "mean_reproj_win": mean(row["reproj_win"] for row in rows),
            "mean_reproj_advantage": mean(row["reproj_mean"] for row in rows),
            "mean_reproj_median": mean(row["reproj_median"] for row in rows),
            "mean_pairwise_win": mean(row["pairwise_win"] for row in rows),
            "mean_pairwise_advantage": mean(row["pairwise_mean"] for row in rows),
            "mean_pairwise_median": mean(row["pairwise_median"] for row in rows),
        }

    seed_gate = all(
        row["train_finite"]
        and row["completed_steps"] == 200
        and row["passed_mode_count"] >= 2
        and row["average_reproj_win"] >= 0.50
        for row in seed_rows
    )
    mode_gate = all(
        row["mean_reproj_win"] >= 0.65
        and row["mean_reproj_advantage"] > 0.0
        and row["mean_reproj_median"] > 0.0
        and row["mean_pairwise_win"] >= 0.80
        and row["mean_pairwise_advantage"] > 0.0
        and row["mean_pairwise_median"] > 0.0
        for row in mode_summary.values()
    )
    passed = seed_gate and mode_gate

    report = {
        "stage": "C1 pairwise v3 multi-seed summary",
        "passed": passed,
        "seeds": list(seeds),
        "source_view_count": source_count,
        "checks": {
            "every_seed_finite_s200_at_least_two_modes_and_not_below_random": seed_gate,
            "every_mode_three_seed_average_passes": mode_gate,
        },
        "seed_reports": seed_rows,
        "mode_summary": mode_summary,
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    print("=" * 100)
    print("C1 pairwise v3 multi-seed summary")
    for row in seed_rows:
        print(
            f"seed={row['seed']} finite={row['train_finite']} "
            f"steps={row['completed_steps']} modes={row['passed_mode_count']}/3 "
            f"avg_reproj_win={row['average_reproj_win']:.4f}"
        )
    print("mode_summary:")
    for mode, row in mode_summary.items():
        print(
            f"  {mode}: reproj_win={row['mean_reproj_win']:.4f} "
            f"reproj_adv={row['mean_reproj_advantage']:+.6f} "
            f"pairwise_win={row['mean_pairwise_win']:.4f} "
            f"pairwise_adv={row['mean_pairwise_advantage']:+.6f}"
        )
    print("passed:", passed)
    print("report:", output_dir / "report.json")

    if args.fail_on_error and not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
