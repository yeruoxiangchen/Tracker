#!/usr/bin/env python3
"""Shared guards for official-target Native-SLat development training."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pose_point_depth_mv.audit_proobjaverse_official_slat_decoder import REPORT_FORMAT
from pose_point_depth_mv.proobjaverse_official_slat_protocol import load_json, sha256_file


def validate_official_decoder_audit(
    path: str | Path,
    *,
    cache_config: dict[str, Any],
    pretrained: str,
) -> dict[str, Any]:
    report_path = Path(path).expanduser().resolve()
    report = load_json(report_path)
    if report.get("format") != REPORT_FORMAT or report.get("passed") is not True:
        raise RuntimeError("official target decoder audit is missing or did not pass")
    if str(report.get("pretrained")) != str(pretrained):
        raise RuntimeError("official decoder audit pretrained binding differs")
    target = dict(cache_config.get("target_source", {}))
    if target.get("kind") != "official Stable-X/ProObjaverse-300K lh-slats":
        raise RuntimeError("training cache is not an official SLat target cache")
    if str(target.get("protocol_sha256")) != str(report.get("protocol_sha256")):
        raise RuntimeError("official decoder audit/training protocol differs")
    if target.get("support_policy") != "official_gt_slat_coordinates":
        raise RuntimeError("official training cache does not use GT support")
    summary = dict(report.get("summary", {}))
    if int(summary.get("object_count", 0)) < 32:
        raise RuntimeError("official decoder audit must cover 32 disjoint objects")
    if float(summary.get("mesh_success_rate", 0.0)) != 1.0:
        raise RuntimeError("official SLat labels do not all decode successfully")
    return {
        "path": str(report_path),
        "sha256": sha256_file(report_path),
        "format": report["format"],
        "protocol_sha256": report["protocol_sha256"],
        "summary": summary,
        "target_definition": report["run_config"]["target_definition"],
    }


__all__ = ["validate_official_decoder_audit"]
