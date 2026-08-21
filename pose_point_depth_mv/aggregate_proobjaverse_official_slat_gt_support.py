#!/usr/bin/env python3
"""Aggregate disjoint official GT-support SLat Mesh evaluation shards."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

from pose_point_depth_mv.eval_direct_slat_flow import summarize
from pose_point_depth_mv.evaluate_proobjaverse_official_slat_gt_support import (
    REPORT_FORMAT,
)
from pose_point_depth_mv.proobjaverse_official_slat_protocol import (
    canonical_sha256,
    sha256_file,
)


AGGREGATE_FORMAT = (
    "pose_point_depth_mv.proobjaverse_official_slat_gt_support_mesh_aggregate.v1"
)
METRICS = (
    "chamfer_l1_improvement",
    "fscore_0p02_delta",
    "normal_consistency_delta",
    "largest_component_ratio_delta",
)
IDENTITY_FIELDS = (
    "format",
    "arm",
    "checkpoint",
    "checkpoint_sha256",
    "checkpoint_step",
    "weights",
    "baseline",
    "target",
    "official_protocol_sha256",
    "official_split",
    "training_overlap",
    "same_coordinates",
    "same_initial_noise",
    "native_ss_executed",
    "native_ss_role",
    "sampling",
    "seeds",
    "expected_object_count_before_sharding",
)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard_reports", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--expected_objects", type=int, default=64)
    parser.add_argument("--bootstrap_samples", type=int, default=5000)
    return parser


def _identity(config: dict[str, Any]) -> dict[str, Any]:
    return {key: config.get(key) for key in IDENTITY_FIELDS}


def main() -> None:
    args = make_parser().parse_args()
    paths = [
        Path(item).expanduser().resolve()
        for item in str(args.shard_reports).split(",")
        if item.strip()
    ]
    if len(paths) < 2 or len(paths) != len(set(paths)):
        raise ValueError("shard_reports must contain at least two unique paths")
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    if any(report.get("format") != REPORT_FORMAT for report in reports):
        raise RuntimeError("a shard report has the wrong format")
    if any(report.get("passed") is not True for report in reports):
        raise RuntimeError("all shard reports must have passed=true")
    identity = _identity(reports[0]["run_config"])
    if any(_identity(report["run_config"]) != identity for report in reports[1:]):
        raise RuntimeError("shard run identities differ")

    ordered = sorted(
        zip(reports, paths),
        key=lambda item: int(item[0]["run_config"]["object_start"]),
    )
    cursor = 0
    object_uids: list[str] = []
    object_row_map: dict[str, dict[str, Any]] = {}
    records: list[dict[str, Any]] = []
    shard_bindings = []
    membership_audits: list[dict[str, Any]] = []
    for report, path in ordered:
        config = report["run_config"]
        start = int(config["object_start"])
        end = int(config["object_end"])
        if start != cursor or end <= start:
            raise RuntimeError(
                f"shards do not form a contiguous partition: expected start={cursor}, "
                f"got [{start}:{end}]"
            )
        if int(report["object_count"]) != end - start:
            raise RuntimeError("shard object_count differs from its object range")
        cursor = end
        object_uids.extend(str(uid) for uid in config["object_uids"])
        for row in report["object_rows"]:
            uid = str(row["object_uid"])
            if uid in object_row_map:
                raise RuntimeError(f"duplicate aggregate object row: {uid}")
            object_row_map[uid] = row
        records.extend(report["records"])
        membership = report.get("checkpoint_evaluation_membership")
        if membership is not None:
            if not isinstance(membership, dict) or membership.get("passed") is not True:
                raise RuntimeError("shard checkpoint/evaluation membership is invalid")
            if int(membership.get("evaluation_object_count", -1)) != end - start:
                raise RuntimeError("shard membership object count differs")
            membership_audits.append(dict(membership))
        shard_bindings.append(
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "object_start": start,
                "object_end": end,
            }
        )
    expected = int(args.expected_objects)
    if cursor != expected or len(object_uids) != expected:
        raise RuntimeError(
            f"aggregate object coverage differs: end={cursor} uids={len(object_uids)} "
            f"expected={expected}"
        )
    if len(set(object_uids)) != expected:
        raise RuntimeError("aggregate object_uids are not unique")
    if set(object_row_map) != set(object_uids):
        raise RuntimeError("aggregate object_rows do not match the frozen object set")
    if membership_audits and len(membership_audits) != len(reports):
        raise RuntimeError("only some GT-support shards contain membership audits")
    aggregate_membership = None
    if membership_audits:
        invariant_keys = (
            "checkpoint_protocol_sha256",
            "evaluation_protocol_sha256",
            "protocol_relation",
            "expected_membership",
            "checkpoint_training_object_count",
            "checkpoint_training_uid_sha256",
        )
        first = membership_audits[0]
        if any(
            any(row.get(key) != first.get(key) for key in invariant_keys)
            for row in membership_audits[1:]
        ):
            raise RuntimeError("GT-support shard membership identities differ")
        overlap_count = sum(
            int(row["training_overlap_count"]) for row in membership_audits
        )
        aggregate_membership = {
            "version": "pose_point_depth_mv.slat_checkpoint_evaluation_membership_aggregate.v1",
            **{key: first[key] for key in invariant_keys},
            "evaluation_object_count": expected,
            "evaluation_uid_sha256": canonical_sha256(object_uids),
            "training_overlap_count": overlap_count,
            "training_overlap_rate": overlap_count / expected,
            "all_evaluation_objects_in_checkpoint_training": overlap_count == expected,
            "all_evaluation_objects_disjoint_from_checkpoint_training": overlap_count == 0,
            "passed": True,
        }
    object_rows = [object_row_map[uid] for uid in object_uids]
    seeds = [int(seed) for seed in identity["seeds"]]
    counts = Counter(str(row["object_uid"]) for row in records)
    if set(counts) != set(object_uids) or any(counts[uid] != len(seeds) for uid in object_uids):
        raise RuntimeError("aggregate records do not contain every object x seed pair")

    summary = {
        name: summarize(
            [float(row[name]) for row in object_rows],
            bootstrap_samples=int(args.bootstrap_samples),
            seed=20260813 + index,
        )
        for index, name in enumerate(METRICS)
    }
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    run_config = dict(identity)
    run_config.update(
        {
            "aggregate": True,
            "object_start": 0,
            "object_end": expected,
            "object_uids": object_uids,
            "shard_reports": shard_bindings,
        }
    )
    report = {
        "format": AGGREGATE_FORMAT,
        "passed": True,
        "formal": False,
        "run_config": run_config,
        "model_summary": reports[0]["model_summary"],
        "object_count": expected,
        "record_count": len(records),
        "summary": summary,
        "object_rows": object_rows,
        "records": records,
        "checkpoint_evaluation_membership": aggregate_membership,
        "scope_guard": reports[0]["scope_guard"],
    }
    report["report_sha256"] = canonical_sha256(report)
    (output / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    lines = [
        "ProObjaverse official GT-support SLat Mesh aggregate",
        "=" * 58,
        f"split: {identity['official_split']}",
        f"objects: {expected} records: {len(records)} weights: {identity['weights']}",
    ]
    for name in METRICS:
        row = summary[name]
        lines.append(
            f"{name}: mean={row['mean']:+.8f} median={row['median']:+.8f} "
            f"win={row['positive_rate']:.6f} "
            f"CI={row['bootstrap_mean_95_ci']}"
        )
    lines.append("")
    lines.append(str(report["scope_guard"]))
    (output / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
