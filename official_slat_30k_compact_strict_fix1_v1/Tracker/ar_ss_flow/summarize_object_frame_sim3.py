#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize P0.5 object-frame Sim(3) audits across transform seeds."
    )
    parser.add_argument("--report_dirs", nargs="+", required=True)
    parser.add_argument("--expected_seeds", default="42,43,44")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--fail_on_error", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    expected = {
        int(item) for item in str(args.expected_seeds).split(",") if item.strip()
    }
    reports: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for directory in args.report_dirs:
        path = Path(directory) / "report.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        report = json.loads(path.read_text(encoding="utf-8"))
        seed = int(report["args"]["seed"])
        reports.append(report)
        rows.append(
            {
                "seed": seed,
                "report": str(path),
                "passed": bool(report["passed"]),
                "mechanism_passed": bool(report["mechanism_passed"]),
                "estimated_frame_passed": bool(report["estimated_frame_passed"]),
                "oracle_mse_max": float(report["oracle_normalized_mse"]["max"]),
                "no_recovery_mse_mean": float(
                    report["no_recovery_normalized_mse"]["mean"]
                ),
                "point_pca_mse_mean": float(
                    report["point_pca_normalized_mse"]["mean"]
                ),
                "estimator_improvement_ratio": float(
                    report["estimator_improvement_ratio"]
                ),
                "rotation_error_median": float(
                    report["point_pca_rotation_error_degrees"]["median"]
                ),
            }
        )
    observed = {row["seed"] for row in rows}
    reference = reports[0]
    protocol_keys = (
        "cache_manifest",
        "indices",
        "max_samples",
        "expected_canonical_extent",
        "oracle_max_normalized_mse",
        "min_no_recovery_mse",
        "min_no_recovery_detection_ratio",
        "min_estimator_improvement_ratio",
        "max_estimator_center_error_over_scale",
        "max_estimator_scale_log_error",
    )
    protocol_consistent = all(
        report.get("format") == reference.get("format")
        and report.get("lifting_volume_version")
        == reference.get("lifting_volume_version")
        and all(
            report.get("args", {}).get(key) == reference.get("args", {}).get(key)
            for key in protocol_keys
        )
        and [item["uid"] for item in report["samples"]]
        == [item["uid"] for item in reference["samples"]]
        for report in reports
    )
    checks = {
        "expected_seeds_present": observed == expected,
        "no_duplicate_seeds": len(observed) == len(rows),
        "protocol_identical": protocol_consistent,
        "mechanism_passes_every_seed": all(
            row["mechanism_passed"] for row in rows
        ),
        "estimated_frame_passes_every_seed": all(
            row["estimated_frame_passed"] for row in rows
        ),
    }
    result = {
        "passed": all(checks.values()),
        "checks": checks,
        "expected_seeds": sorted(expected),
        "observed_seeds": sorted(observed),
        "mean_oracle_mse_max": mean(row["oracle_mse_max"] for row in rows),
        "mean_no_recovery_mse": mean(
            row["no_recovery_mse_mean"] for row in rows
        ),
        "mean_point_pca_mse": mean(row["point_pca_mse_mean"] for row in rows),
        "mean_estimator_improvement_ratio": mean(
            row["estimator_improvement_ratio"] for row in rows
        ),
        "runs": sorted(rows, key=lambda row: row["seed"]),
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "report.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    lines = [
        "# P0.5 Object-frame Multi-seed Summary",
        "",
        f"- passed: `{result['passed']}`",
        f"- checks: `{checks}`",
        "",
        "| seed | mechanism | point-PCA | oracle max MSE | no recovery MSE | PCA MSE | rotation error |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["runs"]:
        lines.append(
            f"| {row['seed']} | {row['mechanism_passed']} | "
            f"{row['estimated_frame_passed']} | {row['oracle_mse_max']:.6g} | "
            f"{row['no_recovery_mse_mean']:.6g} | "
            f"{row['point_pca_mse_mean']:.6g} | "
            f"{row['rotation_error_median']:.6g} |"
        )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if args.fail_on_error and not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
