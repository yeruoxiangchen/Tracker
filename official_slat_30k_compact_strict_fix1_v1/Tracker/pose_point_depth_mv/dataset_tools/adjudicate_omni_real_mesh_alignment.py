#!/usr/bin/env python3
"""Freeze a robust label-alignment gate from the development benchmark.

The geometric transform is not recomputed. The original untrimmed p90 is kept
as a diagnostic, while the hard gate matches the 80%-trimmed ICP objective:
the median must be within 3% of the Scan diagonal and at least 60% of the
mask-supported COLMAP points must be within 3%.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pose_point_depth_mv.dataset_tools.align_omni_real_mesh_to_colmap import (
    MANIFEST_FORMAT as SOURCE_MANIFEST_FORMAT,
)
from pose_point_depth_mv.dataset_tools.prepare_omni_real_video_cache import utc_now
from pose_point_depth_mv.omni_real_benchmark_common import (
    atomic_json,
    load_json,
    sha256_file,
)


MANIFEST_FORMAT = "pose_point_depth_mv.omni_real_mesh_alignment_adjudicated.v2"
POLICY = {
    "name": "trimmed_icp_majority_support.v1",
    "max_median_normalized": 0.03,
    "min_inlier_rate_3pct": 0.60,
    "p90_role": "diagnostic_only",
    "rationale": (
        "The fitted objective already trims 20% of source points. Requiring an "
        "untrimmed p90 hard limit contradicts that robust objective when sparse "
        "COLMAP/mask support contains background or support-surface tail outliers."
    ),
    "freeze_scope": (
        "calibrated on reusable development benchmark32 before model metrics; "
        "must be reused unchanged for train500 and untouched holdout64"
    ),
}


def adjudicate_row(row: dict[str, Any]) -> dict[str, Any]:
    invariant_names = (
        "minimum_object_points",
        "proper_rotation",
        "similarity_inverse_roundtrip",
        "mesh_vertex_count_preserved",
    )
    original_checks = dict(row.get("checks", {}))
    invariant = {name: original_checks.get(name) is True for name in invariant_names}
    robust_checks = {
        **invariant,
        "median_normalized": float(row["median_normalized"])
        <= float(POLICY["max_median_normalized"]),
        "inlier_rate_3pct": float(row["inlier_rate_3pct"])
        >= float(POLICY["min_inlier_rate_3pct"]),
    }
    artifacts = {}
    for name in ("aligned_mesh", "cache_npz", "alignment_ply", "alignment_preview"):
        path = Path(row[name]).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        artifacts[name] = {"path": str(path), "sha256": sha256_file(path)}
    output = dict(row)
    output.update(
        {
            "source_automatic_passed": bool(row.get("automatic_passed")),
            "source_checks": original_checks,
            "alignment_quality_policy": POLICY,
            "alignment_quality_checks": robust_checks,
            "p90_normalized_diagnostic": float(row["p90_normalized"]),
            "artifacts": artifacts,
            "automatic_passed": all(robust_checks.values()),
        }
    )
    return output


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_alignment_manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected_objects", type=int, default=32)
    return parser


def main() -> None:
    args = make_parser().parse_args()
    source_path = Path(args.source_alignment_manifest).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    source = load_json(source_path)
    if source.get("format") != SOURCE_MANIFEST_FORMAT:
        raise RuntimeError(f"unexpected source alignment format: {source_path}")
    rows = list(source.get("objects", []))
    if (
        len(rows) != int(args.expected_objects)
        or int(source.get("completed_object_count", -1)) != len(rows)
        or source.get("failures")
    ):
        raise RuntimeError("source alignment is incomplete; adjudication is forbidden")
    adjudicated = [adjudicate_row(row) for row in rows]
    passed_count = sum(int(row["automatic_passed"]) for row in adjudicated)
    payload = {
        "format": MANIFEST_FORMAT,
        "created_at_utc": utc_now(),
        "source_alignment_manifest": str(source_path),
        "source_alignment_manifest_sha256": sha256_file(source_path),
        "raw_cache_report": source["raw_cache_report"],
        "raw_cache_report_sha256": source["raw_cache_report_sha256"],
        "alignment_quality_policy": POLICY,
        "selected_object_count": len(rows),
        "completed_object_count": len(rows),
        "source_automatic_pass_count": int(source.get("automatic_pass_count", 0)),
        "automatic_pass_count": passed_count,
        "objects": adjudicated,
        "failures": [],
        "training_ready": False,
        "scope_guard": (
            "Development label-front-end adjudication only. No transform is "
            "refit and no model output or metric is read. The policy is frozen "
            "before train500/holdout64 use."
        ),
        "passed": passed_count == len(rows),
    }
    if output_path.is_file():
        existing = load_json(output_path)
        comparable = dict(existing)
        comparable.pop("created_at_utc", None)
        expected = dict(payload)
        expected.pop("created_at_utc", None)
        if comparable != expected:
            raise RuntimeError(f"existing alignment adjudication differs: {output_path}")
    else:
        atomic_json(output_path, payload)
    print(json.dumps({
        "passed": payload["passed"],
        "source_automatic_pass_count": payload["source_automatic_pass_count"],
        "automatic_pass_count": passed_count,
        "object_count": len(rows),
        "policy": POLICY,
        "output": str(output_path),
    }, indent=2, ensure_ascii=False))
    if not payload["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
