#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any

from pose_point_depth_mv.point_anchor_v2 import POINT_CONTROL_NAMES


def parse_csv_ints(value: str) -> list[int]:
    output = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not output:
        raise ValueError("expected at least one training seed")
    return output


def load_report(path: str) -> dict[str, Any]:
    report_path = Path(path)
    if report_path.is_dir():
        report_path = report_path / "report.json"
    if not report_path.is_file():
        raise FileNotFoundError(report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["_path"] = str(report_path.resolve())
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize three-seed point-anchor v2 train/fresh reports."
    )
    parser.add_argument("--train_reports", nargs="+", required=True)
    parser.add_argument("--fresh_reports", nargs="+", required=True)
    parser.add_argument("--expected_seeds", default="42,43,44")
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def protocol_signature(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "cache_config_hash": report.get("cache_config_hash"),
        "checkpoint_step": report.get("checkpoint_step"),
        "noise_seeds": report.get("noise_seeds"),
        "t_values": report.get("t_values"),
        "physical_scale": report.get("physical_scale"),
        "probe": report.get("probe"),
        "flow": report.get("flow"),
        "training_config": report.get("training_config"),
        "controls": list(report.get("controls", {})),
    }


def main() -> None:
    args = parse_args()
    expected_seeds = parse_csv_ints(args.expected_seeds)
    train = [load_report(path) for path in args.train_reports]
    fresh = [load_report(path) for path in args.fresh_reports]
    if len(train) != len(expected_seeds) or len(fresh) != len(expected_seeds):
        raise ValueError("train/fresh report counts must match expected seeds")
    by_seed: dict[int, dict[str, dict[str, Any]]] = {}
    for report in train + fresh:
        seed = int(report.get("training_seed", -1))
        split = str(report.get("split_name"))
        if seed not in expected_seeds or split not in {"train16", "fresh48"}:
            raise ValueError(f"unexpected report seed/split: {seed}/{split}")
        if split in by_seed.setdefault(seed, {}):
            raise ValueError(f"duplicate report seed/split: {seed}/{split}")
        by_seed[seed][split] = report
    if sorted(by_seed) != sorted(expected_seeds) or any(
        set(rows) != {"train16", "fresh48"} for rows in by_seed.values()
    ):
        raise ValueError("missing point-anchor seed/split reports")

    signatures = [protocol_signature(report) for report in train + fresh]
    # Eval object hashes differ by split, so they are deliberately outside this signature.
    if any(signature != signatures[0] for signature in signatures[1:]):
        raise RuntimeError("point-anchor reports do not share one protocol")
    train_hashes = {report["eval_object_uid_hash"] for report in train}
    fresh_hashes = {report["eval_object_uid_hash"] for report in fresh}
    if len(train_hashes) != 1 or len(fresh_hashes) != 1:
        raise RuntimeError("point-anchor object sets differ across seeds")

    seed_rows: dict[str, Any] = {}
    for seed in expected_seeds:
        train_report = by_seed[seed]["train16"]
        fresh_report = by_seed[seed]["fresh48"]
        seed_rows[str(seed)] = {
            "train_passed": bool(train_report["passed"]),
            "fresh_passed": bool(fresh_report["passed"]),
            "paired_passed": bool(train_report["passed"] and fresh_report["passed"]),
            "train_correct_mean": float(
                train_report["correct_vs_stock"]["object"]["mean"]
            ),
            "fresh_correct_mean": float(
                fresh_report["correct_vs_stock"]["object"]["mean"]
            ),
            "fresh_correct_win": float(
                fresh_report["correct_vs_stock"]["object_win_rate"]
            ),
            "fresh_controls": {
                name: {
                    "mean": float(
                        fresh_report["controls"][name]["correct_advantage_flow"][
                            "object"
                        ]["mean"]
                    ),
                    "median": float(
                        fresh_report["controls"][name]["correct_advantage_flow"][
                            "object"
                        ]["median"]
                    ),
                    "win": float(
                        fresh_report["controls"][name]["correct_advantage_flow"][
                            "object_win_rate"
                        ]
                    ),
                    "ci_low": float(
                        fresh_report["controls"][name]["correct_advantage_flow"][
                            "object_bootstrap_95_ci"
                        ][0]
                    ),
                }
                for name in POINT_CONTROL_NAMES
            },
            "train_report": train_report["_path"],
            "fresh_report": fresh_report["_path"],
        }

    checks = {
        "all_train_reports_pass": all(report["passed"] for report in train),
        "all_fresh_reports_pass": all(report["passed"] for report in fresh),
        "all_seeds_paired_pass": all(
            row["paired_passed"] for row in seed_rows.values()
        ),
        "fresh_correct_mean_positive_all_seeds": all(
            row["fresh_correct_mean"] > 0.0 for row in seed_rows.values()
        ),
        "fresh_all_control_ci_lows_positive_all_seeds": all(
            control["ci_low"] > 0.0
            for row in seed_rows.values()
            for control in row["fresh_controls"].values()
        ),
    }
    report = {
        "stage": "Point-only local-anchor v2 three-seed summary",
        "passed": all(checks.values()),
        "checks": checks,
        "expected_seeds": expected_seeds,
        "protocol": signatures[0],
        "train_object_uid_hash": next(iter(train_hashes)),
        "fresh_object_uid_hash": next(iter(fresh_hashes)),
        "seed_results": seed_rows,
        "fresh_correct_mean_across_seeds": mean(
            row["fresh_correct_mean"] for row in seed_rows.values()
        ),
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    lines = [
        "# Point-only Local-anchor V2 Multi-seed Summary",
        "",
        f"- Decision: **{'PASS' if report['passed'] else 'FAIL'}**",
        "",
        "## Checks",
        "",
    ]
    lines.extend(f"- `{name}`: `{value}`" for name, value in checks.items())
    lines.extend(["", "## Seeds", "", "```json", json.dumps(seed_rows, indent=2), "```"])
    (output_dir / "report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
