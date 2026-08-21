#!/usr/bin/env python3
"""Aggregate disjoint GPU shards from official Native-SS evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from pose_point_depth_mv.eval_direct_flow import (
    bootstrap_mean_ci,
    positive_rate,
    summarize,
)
from pose_point_depth_mv.evaluate_native_ss_genrecon import aggregate_records
from pose_point_depth_mv.native_ss_genrecon import canonical_json_sha256, sha256_file
from pose_point_depth_mv.proobjaverse_official_ss import (
    OFFICIAL_SS_EVAL,
    OFFICIAL_SS_EVAL_AGGREGATE,
    validate_official_ss_domain_contract,
)


def csv_paths(text: str) -> list[Path]:
    paths = [
        Path(item.strip()).expanduser().resolve()
        for item in text.split(",")
        if item.strip()
    ]
    if not paths or len(paths) != len(set(paths)):
        raise ValueError("shard_reports must be non-empty and unique")
    return paths


def validate_record_matrix(
    records: list[dict[str, Any]],
    *,
    object_uids: list[str],
    seeds: list[int],
    label: str,
) -> None:
    """Require exactly one record for every frozen object/seed pair."""

    expected = {
        (str(object_uid), int(seed))
        for object_uid in object_uids
        for seed in seeds
    }
    observed = [
        (str(row.get("object_uid", "")), int(row.get("seed", -1)))
        for row in records
    ]
    if (
        len(records) != len(expected)
        or len(observed) != len(set(observed))
        or set(observed) != expected
    ):
        missing = sorted(expected - set(observed))[:8]
        unexpected = sorted(set(observed) - expected)[:8]
        raise RuntimeError(
            f"{label} record matrix differs: records={len(records)} "
            f"expected={len(expected)} missing={missing} unexpected={unexpected}"
        )


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard_reports", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--expected_objects", type=int, required=True)
    parser.add_argument("--bootstrap_samples", type=int, default=5000)
    parser.add_argument("--min_iou_gain_mean", type=float, default=0.0)
    parser.add_argument("--min_iou_win_rate", type=float, default=0.5)
    parser.add_argument("--min_recall_gain_mean", type=float, default=0.0)
    parser.add_argument("--min_latent_mse_gain_mean", type=float, default=0.0)
    parser.add_argument("--min_count_ratio", type=float, default=0.85)
    parser.add_argument("--max_count_ratio", type=float, default=1.20)
    parser.add_argument("--min_pose_control_iou_advantage", type=float, default=0.0)
    return parser


def main() -> None:
    args = make_parser().parse_args()
    paths = csv_paths(args.shard_reports)
    reports: list[dict[str, Any]] = []
    for path in paths:
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("format") != OFFICIAL_SS_EVAL:
            raise ValueError(f"unexpected shard format: {path}")
        reports.append(value)
    first = reports[0]
    binding_keys = (
        "checkpoint",
        "cache_manifest_sha256",
        "checkpoint_sha256",
        "checkpoint_step",
        "joint_seeds",
        "steps",
        "cfg_interval",
        "guidance_rescale",
        "rescale_t",
        "amp_dtype",
        "weights",
        "condition_scale_policy",
        "post_cfg_cap",
    )
    reference = {key: first["protocol"].get(key) for key in binding_keys}
    calibration_sha = str(first.get("calibration_sha256", ""))
    calibrated_parameters = dict(first.get("calibrated_parameters", {}))
    cache_feature_contract = first["protocol"].get("cache_feature_contract")
    if not isinstance(cache_feature_contract, dict):
        raise RuntimeError("shard protocol lacks the official SS feature contract")
    official_ss_targets = cache_feature_contract.get("official_ss_targets")
    if not isinstance(official_ss_targets, dict):
        raise RuntimeError("shard protocol lacks the official SS target binding")
    official_ss_domain = official_ss_targets.get("domain_contract")
    if not isinstance(official_ss_domain, dict):
        raise RuntimeError("shard protocol lacks the official SS domain contract")
    validate_official_ss_domain_contract(official_ss_domain)
    correct_records: list[dict[str, Any]] = []
    control_records: list[dict[str, Any]] = []
    object_uids: list[str] = []
    for path, report in zip(paths, reports):
        observed = {key: report["protocol"].get(key) for key in binding_keys}
        if (
            observed != reference
            or str(report.get("calibration_sha256", "")) != calibration_sha
            or dict(report.get("calibrated_parameters", {}))
            != calibrated_parameters
            or report["protocol"].get("cache_feature_contract")
            != cache_feature_contract
        ):
            raise RuntimeError(f"shard protocol differs: {path}")
        shard_uids = [str(value) for value in report["protocol"]["object_uids"]]
        if (
            not shard_uids
            or len(shard_uids) != len(set(shard_uids))
            or canonical_json_sha256(sorted(shard_uids))
            != str(report["protocol"].get("object_uid_hash", ""))
        ):
            raise RuntimeError(f"shard object identity differs: {path}")
        if set(shard_uids).intersection(object_uids):
            raise RuntimeError(f"shard objects overlap: {path}")
        object_uids.extend(shard_uids)
        correct_records.extend(report["correct"]["records"])
        control_records.extend(report["pose_cyclic_control"]["records"])
    if (
        len(object_uids) != int(args.expected_objects)
        or len(set(object_uids)) != int(args.expected_objects)
    ):
        raise RuntimeError(
            f"expected {args.expected_objects} unique objects, observed "
            f"records={len(object_uids)} unique={len(set(object_uids))}"
        )
    seeds = [int(value) for value in reference["joint_seeds"]]
    if not seeds or len(seeds) != len(set(seeds)):
        raise RuntimeError("aggregate protocol seeds are empty or duplicated")
    if (
        set(calibrated_parameters)
        != {"cfg_strength", "condition_scale_policy", "post_cfg_cap"}
        or calibrated_parameters["condition_scale_policy"]
        != "learned_projection_only"
        or calibrated_parameters["post_cfg_cap"] is not False
        or reference["condition_scale_policy"] != "learned_projection_only"
        or reference["post_cfg_cap"] is not False
    ):
        raise RuntimeError("aggregate Native SS sampling semantics differ")
    validate_record_matrix(
        correct_records,
        object_uids=object_uids,
        seeds=seeds,
        label="correct",
    )
    validate_record_matrix(
        control_records,
        object_uids=object_uids,
        seeds=seeds,
        label="pose control",
    )
    correct_rows, summaries, count_summary = aggregate_records(
        correct_records,
        bootstrap_samples=int(args.bootstrap_samples),
        seed=93001,
    )
    control_by_key = {
        (str(row["object_uid"]), int(row["seed"])): row for row in control_records
    }
    pose_by_object: dict[str, list[float]] = {}
    for row in correct_records:
        key = (str(row["object_uid"]), int(row["seed"]))
        other = control_by_key[key]
        pose_by_object.setdefault(key[0], []).append(
            float(row["full"]["iou"] - other["full"]["iou"])
        )
    pose_values = [float(np.mean(values)) for _, values in sorted(pose_by_object.items())]
    pose_summary = {
        **summarize(pose_values),
        "positive_rate": positive_rate(pose_values),
        "bootstrap_mean_95_ci": bootstrap_mean_ci(
            pose_values, samples=int(args.bootstrap_samples), seed=93002
        ),
    }
    count_mean = float(count_summary["full_stock_count_ratio"]["mean"])
    checks = {
        "correct_record_matrix_exact": True,
        "pose_control_record_matrix_exact": True,
        "iou_gain_mean": float(summaries["iou_gain"]["mean"]) >= float(args.min_iou_gain_mean),
        "iou_object_win_rate": float(summaries["iou_gain"]["positive_rate"]) >= float(args.min_iou_win_rate),
        "recall_gain_mean": float(summaries["recall_gain"]["mean"]) >= float(args.min_recall_gain_mean),
        "latent_mse_gain_mean": float(summaries["latent_mse_gain"]["mean"]) >= float(args.min_latent_mse_gain_mean),
        "count_ratio_lower": count_mean >= float(args.min_count_ratio),
        "count_ratio_upper": count_mean <= float(args.max_count_ratio),
        "stock_baseline_nonempty": int(count_summary["stock_empty_record_count"]) == 0,
        "pose_control_iou_advantage": float(pose_summary["mean"]) > float(args.min_pose_control_iou_advantage),
        "disabled_stock_equivalence": all(
            bool(report["correct"].get("disabled_stock_equivalence", {}).get("passed"))
            for report in reports
        ),
    }
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "format": OFFICIAL_SS_EVAL_AGGREGATE,
        "passed": all(checks.values()),
        "formal": False,
        "object_count": len(set(object_uids)),
        "object_uids": sorted(set(object_uids)),
        "object_uid_hash": canonical_json_sha256(sorted(set(object_uids))),
        "record_count": len(correct_records),
        "protocol": reference,
        "official_ss_domain_contract": official_ss_domain,
        "calibration_sha256": calibration_sha,
        "calibrated_parameters": calibrated_parameters,
        "shard_reports": [str(path) for path in paths],
        "shard_report_sha256": [sha256_file(path) for path in paths],
        "correct": {
            "summary": summaries,
            "count_summary": count_summary,
            "object_rows": correct_rows,
        },
        "correct_over_pose_control_iou": pose_summary,
        "checks": checks,
        "deployment": {
            "checkpoint": str(reference["checkpoint"]),
            "checkpoint_sha256": str(reference["checkpoint_sha256"]),
            "checkpoint_step": int(reference["checkpoint_step"]),
            "weights": str(reference["weights"]),
            "cfg_strength": float(calibrated_parameters["cfg_strength"]),
            "steps": int(reference["steps"]),
            "cfg_interval": [float(value) for value in reference["cfg_interval"]],
            "guidance_rescale": float(reference["guidance_rescale"]),
            "rescale_t": float(reference["rescale_t"]),
            "amp_dtype": str(reference["amp_dtype"]),
        },
        "scope_guard": (
            "held-out official Dev48 Native-SS occupancy diagnosis; Stock-SLat/Mesh "
            "transfer remains a separate downstream gate"
        ),
    }
    report["report_sha256"] = canonical_json_sha256(report)
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    lines = [
        "ProObjaverse official Native-SS held-out aggregate",
        "=" * 52,
        f"objects: {report['object_count']} records: {report['record_count']}",
    ]
    for name, row in summaries.items():
        lines.append(
            f"{name}: mean={row['mean']:+.8f} median={row['median']:+.8f} "
            f"win={row['positive_rate']:.6f} CI={row['bootstrap_mean_95_ci']}"
        )
    lines.extend((f"pose_control: {pose_summary}", f"checks: {checks}", f"PASS: {report['passed']}"))
    (output / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)
    raise SystemExit(0 if report["passed"] else 3)


if __name__ == "__main__":
    main()
