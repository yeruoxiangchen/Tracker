#!/usr/bin/env python3
"""Explicitly unblind a completed formal Holdout64 Pose+Mask addendum report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pose_point_depth_mv.dataset_tools.prepare_omni_real_video_cache import utc_now
from pose_point_depth_mv.evaluate_holdout64_pose_mask_blind_addendum import (
    EXPECTED_OBJECTS,
    METHOD_SPECS,
    REPORT_FORMAT,
)
from pose_point_depth_mv.freeze_holdout64_pose_mask_blind_protocol import (
    validate_protocol_contract,
)
from pose_point_depth_mv.omni_real_benchmark_common import (
    atomic_json,
    load_json,
    sha256_file,
)


MARKER_FORMAT = "pose_point_depth_mv.holdout64_pose_mask_unblinding.v1"


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--blind_protocol_contract", required=True)
    parser.add_argument("--marker", required=True)
    return parser


def main() -> None:
    args = make_parser().parse_args()
    report_path = Path(args.report).expanduser().resolve()
    protocol_path = Path(args.blind_protocol_contract).expanduser().resolve()
    marker_path = Path(args.marker).expanduser().resolve()
    protocol = validate_protocol_contract(protocol_path)
    report = load_json(report_path)
    if (
        report.get("format") != REPORT_FORMAT
        or report.get("passed") is not True
        or report.get("formal") is not True
        or int(report.get("object_count", -1)) != EXPECTED_OBJECTS
        or int(report.get("record_count", -1))
        != EXPECTED_OBJECTS * len(METHOD_SPECS)
        or report.get("unblinding_required") is not True
        or report.get("blind_protocol_contract", {}).get("sha256")
        != sha256_file(protocol_path)
        or report.get("blind_protocol_contract", {}).get("payload_sha256")
        != protocol["payload_sha256"]
    ):
        raise RuntimeError("formal blind addendum report contract did not pass")
    marker = {
        "format": MARKER_FORMAT,
        "unblinded_at_utc": utc_now(),
        "report": str(report_path),
        "report_sha256": sha256_file(report_path),
        "blind_protocol_contract": str(protocol_path),
        "blind_protocol_contract_sha256": sha256_file(protocol_path),
        "protocol_passed": True,
        "passed": True,
    }
    if marker_path.exists():
        existing = load_json(marker_path)
        immutable = {
            key: value for key, value in marker.items() if key != "unblinded_at_utc"
        }
        if any(existing.get(key) != value for key, value in immutable.items()):
            raise RuntimeError("existing unblind marker binds a different report")
    else:
        atomic_json(marker_path, marker)

    print("Holdout64 Pose+Mask blind addendum: UNBLINDED")
    print(f"protocol_passed: {report['passed']}")
    print(f"objects: {report['object_count']}; records: {report['record_count']}")
    print("passed only means protocol and mesh completeness; it is not a quality win.")
    print("\nAll64 summaries:")
    print(json.dumps(report["summary"], indent=2, ensure_ascii=False))
    print("\nPose+Mask paired comparisons (positive means Pose+Mask is better):")
    print(
        json.dumps(
            report["pose_mask_paired_comparisons"], indent=2, ensure_ascii=False
        )
    )
    print("\nLabel-quality subgroup disclosures:")
    print(json.dumps(report["label_quality_subgroups"], indent=2, ensure_ascii=False))
    print(f"\nunblind_marker: {marker_path}")


if __name__ == "__main__":
    main()
