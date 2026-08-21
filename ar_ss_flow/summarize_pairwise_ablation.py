#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any

MODES = ("pose_cyclic1", "pose_cyclic2", "pose_reverse")
ABLATIONS = ("visual_zero", "visual_shuffle", "geometry_pair_off", "uniform_pairwise")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize C1 pairwise v3 ablations.")
    parser.add_argument("--full_report", required=True)
    parser.add_argument("--ablation_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--source_view_count", type=int, default=3)
    parser.add_argument("--fail_on_error", action="store_true")
    return parser.parse_args()


def load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def mode_metrics(report: dict[str, Any], source_count: int) -> dict[str, dict[str, float]]:
    summary = report["summary"][str(source_count)]
    result: dict[str, dict[str, float]] = {}
    for mode in MODES:
        row = summary[mode]
        result[mode] = {
            "objects": float(row["object_count"]),
            "reproj_win": float(row["object_win_rate"]),
            "reproj_mean": float(row["advantage"]["mean"]),
            "reproj_median": float(row["advantage"]["median"]),
            "pairwise_win": float(row.get("pairwise_confidence_win_rate", 0.0)),
            "pairwise_mean": float(
                row.get("pairwise_confidence_advantage", {}).get("mean", 0.0)
            ),
            "pairwise_median": float(
                row.get("pairwise_confidence_advantage", {}).get("median", 0.0)
            ),
        }
    return result


def aggregate(rows: dict[str, dict[str, float]]) -> dict[str, float]:
    keys = (
        "reproj_win",
        "reproj_mean",
        "reproj_median",
        "pairwise_win",
        "pairwise_mean",
        "pairwise_median",
    )
    return {key: mean(row[key] for row in rows.values()) for key in keys}


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    source_count = int(args.source_view_count)

    full_report = load(Path(args.full_report))
    reports: dict[str, dict[str, Any]] = {"full": full_report}
    root = Path(args.ablation_root)
    for name in ABLATIONS:
        reports[name] = load(root / name / "report.json")

    per_ablation: dict[str, Any] = {}
    for name, report in reports.items():
        metrics = mode_metrics(report, source_count)
        per_ablation[name] = {
            "modes": metrics,
            "average": aggregate(metrics),
        }

    full = per_ablation["full"]["average"]
    zero = per_ablation["visual_zero"]["average"]
    shuffle = per_ablation["visual_shuffle"]["average"]
    geometry_off = per_ablation["geometry_pair_off"]["average"]
    uniform = per_ablation["uniform_pairwise"]["average"]

    full_pass = all(
        per_ablation["full"]["modes"][mode]["reproj_win"] >= 0.65
        and per_ablation["full"]["modes"][mode]["reproj_mean"] > 0.0
        and per_ablation["full"]["modes"][mode]["reproj_median"] > 0.0
        and per_ablation["full"]["modes"][mode]["pairwise_win"] >= 0.80
        for mode in MODES
    )

    def meaningful_visual_drop(ablation: dict[str, float]) -> bool:
        reproj_drop = (
            full["reproj_win"] - ablation["reproj_win"] >= 0.10
            or full["reproj_mean"] - ablation["reproj_mean"] >= 0.004
        )
        pairwise_drop = (
            full["pairwise_win"] - ablation["pairwise_win"] >= 0.10
            or full["pairwise_mean"] - ablation["pairwise_mean"] >= 0.005
        )
        return reproj_drop and pairwise_drop

    visual_zero_drop = meaningful_visual_drop(zero)
    visual_shuffle_drop = meaningful_visual_drop(shuffle)
    geometry_pair_off_signal = (
        geometry_off["reproj_win"] >= 0.55
        and geometry_off["reproj_mean"] > 0.0
        and geometry_off["pairwise_win"] >= 0.70
        and geometry_off["pairwise_mean"] > 0.0
    )
    pairwise_gate_value = (
        full["reproj_win"] - uniform["reproj_win"] >= 0.05
        or full["reproj_mean"] - uniform["reproj_mean"] >= 0.003
    )

    checks = {
        "full_gate_passed": full_pass,
        "visual_zero_causes_meaningful_drop": visual_zero_drop,
        "visual_shuffle_causes_meaningful_drop": visual_shuffle_drop,
        "visual_only_pairwise_signal_survives_geometry_pair_off": geometry_pair_off_signal,
        "learned_pairwise_gate_beats_uniform_pairwise": pairwise_gate_value,
    }
    passed = all(checks.values())

    report = {
        "stage": "C1 pairwise v3 visual/geometry ablation summary",
        "passed": passed,
        "source_view_count": source_count,
        "checks": checks,
        "thresholds": {
            "visual_drop_reproj_win_or_mean": [0.10, 0.004],
            "visual_drop_pairwise_win_or_mean": [0.10, 0.005],
            "geometry_pair_off_min_reproj_win": 0.55,
            "geometry_pair_off_min_pairwise_win": 0.70,
            "uniform_gap_reproj_win_or_mean": [0.05, 0.003],
        },
        "per_ablation": per_ablation,
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    print("=" * 100)
    print("C1 pairwise v3 ablation summary")
    print("source_view_count:", source_count)
    for name in ("full", *ABLATIONS):
        row = per_ablation[name]["average"]
        print(
            f"{name:18s} "
            f"reproj_win={row['reproj_win']:.4f} "
            f"reproj_mean={row['reproj_mean']:+.6f} "
            f"pairwise_win={row['pairwise_win']:.4f} "
            f"pairwise_mean={row['pairwise_mean']:+.6f}"
        )
    print("checks:")
    for name, value in checks.items():
        print(f"  {name}: {value}")
    print("passed:", passed)
    print("report:", output_dir / "report.json")

    if args.fail_on_error and not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
