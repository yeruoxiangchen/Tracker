#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean
from typing import Any

import torch


FORMAT = "ar_ss_flow.pair_feature_local_velocity.v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Protocol-checked multi-seed summary for C3 pair-feature SS evals."
    )
    parser.add_argument("--report_dirs", nargs="+", required=True)
    parser.add_argument("--expected_train_seeds", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--decision_profile",
        choices=("report_only", "benchmark_relaxed", "mechanism_strict"),
        default="report_only",
    )
    parser.add_argument("--fail_on_error", action="store_true")
    return parser.parse_args()


def load_report(path: Path) -> dict[str, Any]:
    report_path = path / "report.json" if path.is_dir() else path
    if not report_path.is_file():
        raise FileNotFoundError(report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("format") != FORMAT:
        raise RuntimeError(f"unexpected report format in {report_path}")
    report["_report_path"] = str(report_path.resolve())
    return report


def manifest_hash(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def training_protocol(checkpoint: dict[str, Any]) -> dict[str, Any]:
    args = checkpoint.get("args", {})
    keys = (
        "pretrained",
        "adapter_hidden_dim",
        "physical_scale",
        "residual_t_min",
        "residual_t_ramp",
        "train_t_min",
        "train_t_max",
        "confidence_floor",
        "neighborhood_radius",
        "min_source_views",
        "negative_modes",
        "flow_weight",
        "wrong_stock_weight",
        "correct_gain_weight",
        "correct_gain_margin",
        "correct_wrong_rank_weight",
        "correct_wrong_margin",
        "delta_norm_weight",
    )
    return {key: args.get(key) for key in keys}


def evaluation_protocol(report: dict[str, Any]) -> dict[str, Any]:
    args = report["args"]
    return {
        "format": report["format"],
        "cache_manifest_hash": manifest_hash(args["cache_manifest"]),
        "sample_uids": report["sample_uids"],
        "noise_seeds": report["noise_seeds"],
        "t_values": report["t_values"],
        "inactive_t_probe": report.get("inactive_t_probe"),
        "negative_modes": args["negative_modes"],
        "physical_scale": report["physical_scale"],
        "amp_dtype": args["amp_dtype"],
        "flow_checkpoint_step": report["flow_checkpoint_step"],
        "correspondence_checkpoint_step": report["correspondence_checkpoint_step"],
        "adapter": report["model_summary"]["adapter"],
        "benchmark_min_object_win": args["benchmark_min_object_win"],
        "benchmark_min_positive_t": args["benchmark_min_positive_t"],
        "mechanism_min_object_win": args["mechanism_min_object_win"],
        "mechanism_min_control_win": args["mechanism_min_control_win"],
        "mechanism_min_positive_t": args["mechanism_min_positive_t"],
        "max_corrupt_abs_gain": args["max_corrupt_abs_gain"],
    }


def main() -> None:
    args = parse_args()
    expected = sorted(
        int(item.strip())
        for item in args.expected_train_seeds.split(",")
        if item.strip()
    )
    reports = [load_report(Path(item)) for item in args.report_dirs]
    rows: list[dict[str, Any]] = []
    reference_eval = None
    reference_train = None
    protocol_matches = True
    for report in reports:
        checkpoint_path = Path(report["args"]["flow_checkpoint"])
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        train_seed = int(checkpoint.get("args", {}).get("seed", -1))
        eval_protocol = evaluation_protocol(report)
        train_protocol = training_protocol(checkpoint)
        if reference_eval is None:
            reference_eval = eval_protocol
            reference_train = train_protocol
        else:
            protocol_matches &= eval_protocol == reference_eval
            protocol_matches &= train_protocol == reference_train
        correct = report["branch_summary"]["correct"]
        rows.append(
            {
                "train_seed": train_seed,
                "report": report["_report_path"],
                "benchmark_relaxed_passed": bool(
                    report["benchmark_relaxed"]["passed"]
                ),
                "mechanism_strict_passed": bool(
                    report["mechanism_strict"]["passed"]
                ),
                "correct_gain_mean": float(correct["gain_vs_stock"]["mean"]),
                "correct_gain_median": float(correct["gain_vs_stock"]["median"]),
                "correct_stock_object_win": float(
                    correct["stock_object_win_rate"]
                ),
                "positive_t": int(report["positive_t"]),
            }
        )

    observed = sorted(row["train_seed"] for row in rows)
    unique_seeds = len(observed) == len(set(observed))
    seed_set_matches = observed == expected
    benchmark_all = all(row["benchmark_relaxed_passed"] for row in rows)
    mechanism_all = all(row["mechanism_strict_passed"] for row in rows)
    common_checks = {
        "report_count_matches": len(rows) == len(expected),
        "train_seeds_unique": unique_seeds,
        "expected_train_seeds": seed_set_matches,
        "protocols_identical": protocol_matches,
    }
    protocol_passed = all(common_checks.values())
    selected_passed = {
        "report_only": protocol_passed,
        "benchmark_relaxed": protocol_passed and benchmark_all,
        "mechanism_strict": protocol_passed and mechanism_all,
    }[args.decision_profile]
    summary = {
        "correct_gain_mean_across_train_seeds": mean(
            row["correct_gain_mean"] for row in rows
        ),
        "correct_gain_median_across_train_seeds": mean(
            row["correct_gain_median"] for row in rows
        ),
        "correct_stock_object_win_mean": mean(
            row["correct_stock_object_win"] for row in rows
        ),
        "positive_t_mean": mean(row["positive_t"] for row in rows),
    }
    output = {
        "stage": "C3 pair-feature SS multi-seed summary",
        "format": FORMAT,
        "decision_profile": args.decision_profile,
        "passed": selected_passed,
        "protocol_passed": protocol_passed,
        "benchmark_relaxed_passed_all_seeds": benchmark_all,
        "mechanism_strict_passed_all_seeds": mechanism_all,
        "checks": common_checks,
        "expected_train_seeds": expected,
        "observed_train_seeds": observed,
        "seed_reports": sorted(rows, key=lambda row: row["train_seed"]),
        "summary": summary,
        "evaluation_protocol": reference_eval,
        "training_protocol": reference_train,
        "claim_limits": {
            "benchmark_relaxed": (
                "Only supports entry into same-noise decoded rollout. It is a "
                "teacher-forced objective result, not a final generation claim."
            ),
            "mechanism_strict": (
                "Required before attributing improvement to correct pose/view-pair "
                "correspondence rather than a generic residual."
            ),
        },
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "report.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    markdown = [
        "# C3 Pair-feature SS Multi-seed Summary",
        "",
        f"- Protocol: {'PASS' if protocol_passed else 'FAIL'}",
        f"- Relaxed ReconViaGen benchmark: {'PASS' if benchmark_all else 'FAIL'}",
        f"- Strict pose/correspondence mechanism: {'PASS' if mechanism_all else 'FAIL'}",
        f"- Selected decision: {'PASS' if selected_passed else 'FAIL'}",
        "",
        "| seed | relaxed | strict | gain mean | gain median | object win | positive t |",
        "|---:|:---:|:---:|---:|---:|---:|---:|",
    ]
    for row in sorted(rows, key=lambda item: item["train_seed"]):
        markdown.append(
            f"| {row['train_seed']} | {row['benchmark_relaxed_passed']} | "
            f"{row['mechanism_strict_passed']} | {row['correct_gain_mean']:+.6g} | "
            f"{row['correct_gain_median']:+.6g} | "
            f"{row['correct_stock_object_win']:.3f} | {row['positive_t']} |"
        )
    (output_dir / "report.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))
    if args.fail_on_error and not selected_passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
