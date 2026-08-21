#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pose_point_depth_mv.view_identity_lifting import (
    VIEW_IDENTITY_CONTROL_NAMES,
    protocol_hash,
)


def parse_csv_ints(value: str) -> list[int]:
    result = [int(item.strip()) for item in str(value).split(",") if item.strip()]
    if not result:
        raise ValueError("expected at least one seed")
    return result


def load_report(path: str) -> dict[str, Any]:
    report_path = Path(path)
    if report_path.is_dir():
        report_path = report_path / "report.json"
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    payload["_report_path"] = str(report_path.resolve())
    return payload


def protocol(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "cache_config_hash": report.get("cache_config_hash"),
        "checkpoint_step": report.get("checkpoint_step"),
        "noise_seeds": report.get("noise_seeds"),
        "t_values": report.get("t_values"),
        "physical_scale": report.get("physical_scale"),
        "mask_protocol": report.get("mask_protocol"),
        "training_config": report.get("training_config"),
        "control_names": list(report.get("controls", {})),
        "probe": report.get("probe"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Strict multi-seed summary for view-identity lifting."
    )
    parser.add_argument("--train_reports", nargs="+", required=True)
    parser.add_argument("--fresh_reports", nargs="+", required=True)
    parser.add_argument("--expected_seeds", default="42,43,44")
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    expected = parse_csv_ints(args.expected_seeds)
    train_reports = [load_report(path) for path in args.train_reports]
    fresh_reports = [load_report(path) for path in args.fresh_reports]
    if len(train_reports) != len(expected) or len(fresh_reports) != len(expected):
        raise ValueError("train/fresh report counts must match expected seeds")

    train_by_seed = {int(row["training_seed"]): row for row in train_reports}
    fresh_by_seed = {int(row["training_seed"]): row for row in fresh_reports}
    if sorted(train_by_seed) != sorted(expected) or sorted(fresh_by_seed) != sorted(expected):
        raise RuntimeError("training seed sets do not match expected seeds")

    protocol_rows = [protocol(row) for row in (*train_reports, *fresh_reports)]
    protocol_hashes = {protocol_hash(row) for row in protocol_rows}
    # Split-specific object hashes differ by design, while every other protocol
    # field must be bit-identical across reports.
    protocols_match = len(protocol_hashes) == 1
    train_object_hashes = {
        str(row.get("eval_object_uid_hash")) for row in train_reports
    }
    fresh_object_hashes = {
        str(row.get("eval_object_uid_hash")) for row in fresh_reports
    }
    seed_results: dict[str, Any] = {}
    for seed in expected:
        train = train_by_seed[seed]
        fresh = fresh_by_seed[seed]
        seed_results[str(seed)] = {
            "train_passed": bool(train.get("passed")),
            "fresh_passed": bool(fresh.get("passed")),
            "train_report": train["_report_path"],
            "fresh_report": fresh["_report_path"],
            "fresh_correct_mean": float(
                fresh["correct_vs_stock"]["object"]["mean"]
            ),
            "fresh_correct_median": float(
                fresh["correct_vs_stock"]["object"]["median"]
            ),
            "fresh_correct_win_rate": float(
                fresh["correct_vs_stock"]["object_win_rate"]
            ),
            "fresh_control_advantages": {
                mode: {
                    "mean": float(
                        fresh["controls"][mode]["correct_advantage_flow"]
                        ["object"]["mean"]
                    ),
                    "win_rate": float(
                        fresh["controls"][mode]["correct_advantage_flow"]
                        ["object_win_rate"]
                    ),
                }
                for mode in VIEW_IDENTITY_CONTROL_NAMES
            },
        }

    checks = {
        "protocols_match": protocols_match,
        "split_labels_match": all(
            row.get("split_name") == "train16" for row in train_reports
        ) and all(row.get("split_name") == "fresh48" for row in fresh_reports),
        "train_object_sets_match": len(train_object_hashes) == 1,
        "fresh_object_sets_match": len(fresh_object_hashes) == 1,
        "train_and_fresh_are_disjoint": train_object_hashes != fresh_object_hashes,
        "all_train_pass": all(
            bool(seed_results[str(seed)]["train_passed"]) for seed in expected
        ),
        "all_fresh_pass": all(
            bool(seed_results[str(seed)]["fresh_passed"]) for seed in expected
        ),
    }
    report = {
        "stage": "View-identity pose-guided lifting multi-seed summary",
        "passed": all(checks.values()),
        "expected_seeds": expected,
        "checks": checks,
        "protocol_hash": next(iter(protocol_hashes)) if protocols_match else None,
        "train_object_uid_hash": (
            next(iter(train_object_hashes)) if len(train_object_hashes) == 1 else None
        ),
        "fresh_object_uid_hash": (
            next(iter(fresh_object_hashes)) if len(fresh_object_hashes) == 1 else None
        ),
        "seed_results": seed_results,
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    lines = [
        "# View-identity Pose-guided Lifting Multi-seed Summary",
        "",
        f"Decision: **{'PASS' if report['passed'] else 'FAIL'}**",
        "",
        "```json",
        json.dumps(report, indent=2),
        "```",
    ]
    (output_dir / "report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
