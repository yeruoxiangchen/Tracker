#!/usr/bin/env python3
"""Freeze a raw-cache subset that meets deployable runtime-input prerequisites."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pose_point_depth_mv.dataset_tools.prepare_omni_real_video_cache import (
    RAW_CACHE_FORMAT,
    sha256_file,
    utc_now,
    write_json,
)


ELIGIBILITY_FORMAT = "pose_point_depth_mv.omni_real_raw_cache_eligibility.v1"


def _object_key(row: dict[str, Any]) -> str:
    return f"{row['category']}:{row['object_id']}"


def freeze_eligible_raw_cache(
    source_path: Path,
    output_path: Path,
    *,
    expected_source_objects: int,
    min_eligible_objects: int,
    min_registered_pairs: int,
    min_sparse_points: int,
) -> dict[str, Any]:
    source_path = source_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    source_hash = sha256_file(source_path)
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if source.get("format") != RAW_CACHE_FORMAT or source.get("passed") is not True:
        raise RuntimeError(f"source raw cache is not eligible: {source_path}")
    rows = list(source.get("objects", []))
    if len(rows) != int(expected_source_objects):
        raise RuntimeError(
            f"source object count={len(rows)} != {int(expected_source_objects)}"
        )
    keys = [_object_key(row) for row in rows]
    if len(keys) != len(set(keys)):
        raise RuntimeError("source raw cache contains duplicate object identities")

    policy = {
        "min_registered_pairs": int(min_registered_pairs),
        "min_sparse_points": int(min_sparse_points),
    }
    accepted = []
    excluded = []
    for row in rows:
        reasons = []
        registered = int(row.get("registered_pair_count", -1))
        sparse = int(row.get("sparse_point_count", -1))
        if registered < policy["min_registered_pairs"]:
            reasons.append(
                f"registered_pair_count={registered}<{policy['min_registered_pairs']}"
            )
        if sparse < policy["min_sparse_points"]:
            reasons.append(f"sparse_point_count={sparse}<{policy['min_sparse_points']}")
        if reasons:
            excluded.append(
                {
                    "object_key": _object_key(row),
                    "registered_pair_count": registered,
                    "sparse_point_count": sparse,
                    "reasons": reasons,
                }
            )
        else:
            accepted.append(row)

    passed = len(accepted) >= int(min_eligible_objects)
    payload = {
        **source,
        "created_at_utc": utc_now(),
        "source_raw_cache_report": str(source_path),
        "source_raw_cache_report_sha256": source_hash,
        "source_object_count": len(rows),
        "category_count": len({str(row["category"]) for row in accepted}),
        "object_count": len(accepted),
        "objects": accepted,
        "eligibility": {
            "format": ELIGIBILITY_FORMAT,
            "policy": policy,
            "minimum_eligible_object_count": int(min_eligible_objects),
            "eligible_object_count": len(accepted),
            "excluded_object_count": len(excluded),
            "excluded": excluded,
            "passed": passed,
        },
        "alignment_passed": False,
        "training_ready": False,
        "scope_guard": (
            "Frozen raw-cache eligibility subset only. Objects below registered-view "
            "or sparse-point prerequisites are excluded before runtime-O and mesh "
            "alignment; no model input or training target is emitted."
        ),
        "passed": passed,
    }

    if output_path.is_file():
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        bindings_match = (
            existing.get("format") == RAW_CACHE_FORMAT
            and existing.get("source_raw_cache_report_sha256") == source_hash
            and existing.get("eligibility", {}).get("policy") == policy
            and existing.get("eligibility", {}).get("minimum_eligible_object_count")
            == int(min_eligible_objects)
        )
        if not bindings_match:
            raise RuntimeError(f"existing eligibility subset binding differs: {output_path}")
        return existing
    write_json(output_path, payload)
    return payload


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_raw_cache_report", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected_source_objects", type=int, default=500)
    parser.add_argument("--min_eligible_objects", type=int, default=450)
    parser.add_argument("--min_registered_pairs", type=int, default=8)
    parser.add_argument("--min_sparse_points", type=int, default=100)
    return parser


def main() -> None:
    args = make_parser().parse_args()
    if min(
        int(args.expected_source_objects),
        int(args.min_eligible_objects),
        int(args.min_registered_pairs),
        int(args.min_sparse_points),
    ) <= 0:
        raise ValueError("eligibility counts must be positive")
    if int(args.min_eligible_objects) > int(args.expected_source_objects):
        raise ValueError("min_eligible_objects exceeds expected_source_objects")
    payload = freeze_eligible_raw_cache(
        Path(args.source_raw_cache_report),
        Path(args.output),
        expected_source_objects=int(args.expected_source_objects),
        min_eligible_objects=int(args.min_eligible_objects),
        min_registered_pairs=int(args.min_registered_pairs),
        min_sparse_points=int(args.min_sparse_points),
    )
    eligibility = payload["eligibility"]
    print(
        json.dumps(
            {
                "passed": payload["passed"],
                "source_object_count": payload["source_object_count"],
                "eligible_object_count": eligibility["eligible_object_count"],
                "excluded_object_count": eligibility["excluded_object_count"],
                "excluded": eligibility["excluded"],
                "output": str(Path(args.output).expanduser().resolve()),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    if payload["passed"] is not True:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
