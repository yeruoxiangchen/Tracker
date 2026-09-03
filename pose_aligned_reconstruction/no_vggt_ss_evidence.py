#!/usr/bin/env python3
"""Freeze and validate explicitly non-formal no-VGGT SS deployments."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from pose_aligned_reconstruction import evaluate_native_ss_stock_slat_mesh as _formal
from pose_aligned_reconstruction.native_ss_genrecon_no_vggt import (
    NATIVE_SS_NO_VGGT_CALIBRATION,
    NATIVE_SS_NO_VGGT_EVAL,
)
from pose_aligned_reconstruction.omni_real_benchmark_common import sha256_file


NO_VGGT_EXPLORATORY_SS_DEPLOYMENT = (
    "pose_point_depth_mv.native_ss_no_vggt_exploratory_deployment.v1"
)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def freeze_exploratory_ss_deployment(
    calibration_path: str | Path,
    output_path: str | Path,
    *,
    cfg_strength: float,
    expected_objects: int,
    diagnostic_scope: str,
) -> dict[str, Any]:
    source_path = Path(calibration_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    calibration = json.loads(source_path.read_text(encoding="utf-8"))
    if calibration.get("format") != NATIVE_SS_NO_VGGT_CALIBRATION:
        raise ValueError("source is not a no-VGGT SS calibration")
    candidates = [
        row
        for row in calibration.get("candidates", [])
        if float(row.get("cfg_strength", float("nan"))) == float(cfg_strength)
    ]
    if len(candidates) != 1:
        raise ValueError(f"expected exactly one CFG={cfg_strength:g} candidate")
    candidate = candidates[0]
    if int(candidate.get("object_count", -1)) != int(expected_objects):
        raise RuntimeError("diagnostic candidate object count differs")
    checks = dict(candidate.get("checks", {}))
    if not checks or all(value is True for value in checks.values()):
        raise RuntimeError(
            "exploratory deployment freezer is only for a non-passing diagnostic"
        )
    protocol = dict(calibration.get("protocol", {}))
    checkpoint = Path(str(protocol.get("checkpoint", ""))).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if sha256_file(checkpoint) != str(protocol.get("checkpoint_sha256", "")):
        raise RuntimeError("diagnostic checkpoint hash differs")
    payload = {
        "format": NO_VGGT_EXPLORATORY_SS_DEPLOYMENT,
        "formal": False,
        "passed": False,
        "exploratory_slat_allowed": True,
        "diagnostic_scope": str(diagnostic_scope),
        "source_calibration": str(source_path),
        "source_calibration_sha256": sha256_file(source_path),
        "protocol": protocol,
        "calibrated_parameters": {
            "cfg_strength": float(cfg_strength),
            "condition_scale_policy": "learned_projection_only",
            "post_cfg_cap": False,
        },
        "quality_checks": checks,
        "failed_quality_checks": sorted(
            str(name) for name, value in checks.items() if value is not True
        ),
        "source_candidate_summary": candidate.get("summary"),
        "source_candidate_count_summary": candidate.get("count_summary"),
        "scope_guard": (
            "exploratory Native-SLat support/training only; this artifact is not "
            "a passed SS calibration, final holdout, formal gate, or model claim"
        ),
    }
    if output.exists():
        existing = json.loads(output.read_text(encoding="utf-8"))
        if existing != payload:
            raise RuntimeError(f"refusing to overwrite changed deployment: {output}")
    else:
        _atomic_json(output, payload)
    return payload


def _exploratory_binding(
    report_path: Path, payload: dict[str, Any]
) -> dict[str, Any]:
    if payload.get("formal") is not False or payload.get("passed") is not False:
        raise RuntimeError("exploratory deployment must remain formal=false, passed=false")
    if payload.get("exploratory_slat_allowed") is not True:
        raise RuntimeError("exploratory SLat use was not explicitly frozen")
    source_path = Path(str(payload.get("source_calibration", ""))).resolve()
    if not source_path.is_file() or sha256_file(source_path) != str(
        payload.get("source_calibration_sha256", "")
    ):
        raise RuntimeError("exploratory deployment source calibration differs")
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if source.get("format") != NATIVE_SS_NO_VGGT_CALIBRATION:
        raise RuntimeError("exploratory deployment source format differs")
    protocol = payload.get("protocol")
    calibrated = payload.get("calibrated_parameters")
    if not isinstance(protocol, dict) or not isinstance(calibrated, dict):
        raise ValueError("exploratory deployment binding is malformed")
    if protocol != source.get("protocol"):
        raise RuntimeError("exploratory deployment protocol differs from calibration")
    required = {
        "weights": "ema",
        "condition_scale_policy": "learned_projection_only",
        "post_cfg_cap": False,
    }
    mismatch = {
        name: (protocol.get(name), expected)
        for name, expected in required.items()
        if protocol.get(name) != expected
    }
    if mismatch:
        raise RuntimeError(f"exploratory SS deployment differs: {mismatch}")
    cfg_strength = float(calibrated.get("cfg_strength", 0.0))
    if not np.isfinite(cfg_strength) or cfg_strength <= 0.0:
        raise ValueError("exploratory deployment has invalid CFG")
    candidates = [
        row
        for row in source.get("candidates", [])
        if float(row.get("cfg_strength", float("nan"))) == cfg_strength
    ]
    if len(candidates) != 1 or dict(candidates[0].get("checks", {})) != dict(
        payload.get("quality_checks", {})
    ):
        raise RuntimeError("exploratory deployment candidate binding differs")
    checkpoint = Path(str(protocol.get("checkpoint", ""))).resolve()
    if not checkpoint.is_file() or sha256_file(checkpoint) != str(
        protocol.get("checkpoint_sha256", "")
    ):
        raise RuntimeError("exploratory deployment checkpoint differs")
    return {
        "report": str(report_path),
        "report_sha256": sha256_file(report_path),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": str(protocol["checkpoint_sha256"]),
        "checkpoint_step": int(protocol["checkpoint_step"]),
        "weights": str(protocol["weights"]),
        "cfg_strength": cfg_strength,
        "steps": int(protocol["steps"]),
        "cfg_interval": [float(value) for value in protocol["cfg_interval"]],
        "guidance_rescale": float(protocol["guidance_rescale"]),
        "rescale_t": float(protocol["rescale_t"]),
        "amp_dtype": str(protocol["amp_dtype"]),
        "formal": False,
        "exploratory": True,
        "diagnostic_scope": str(payload.get("diagnostic_scope", "")),
        "failed_quality_checks": list(payload.get("failed_quality_checks", [])),
        "source_calibration": str(source_path),
        "source_calibration_sha256": str(payload["source_calibration_sha256"]),
        "scope_guard": str(payload.get("scope_guard", "")),
    }


def load_no_vggt_ss_evidence(
    path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    report_path = Path(path).expanduser().resolve()
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    if payload.get("format") == NO_VGGT_EXPLORATORY_SS_DEPLOYMENT:
        return payload, _exploratory_binding(report_path, payload)
    original_format = _formal.NATIVE_SS_GENRECON_EVAL
    _formal.NATIVE_SS_GENRECON_EVAL = NATIVE_SS_NO_VGGT_EVAL
    try:
        return _formal.load_ss_evidence(report_path)
    finally:
        _formal.NATIVE_SS_GENRECON_EVAL = original_format


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cfg_strength", type=float, required=True)
    parser.add_argument("--expected_objects", type=int, default=16)
    parser.add_argument("--diagnostic_scope", required=True)
    args = parser.parse_args()
    payload = freeze_exploratory_ss_deployment(
        args.calibration,
        args.output,
        cfg_strength=float(args.cfg_strength),
        expected_objects=int(args.expected_objects),
        diagnostic_scope=str(args.diagnostic_scope),
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
