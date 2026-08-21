#!/usr/bin/env python3
"""Evaluate no-VGGT objective-v2 Native-SLat against frozen Stock-SLat."""

from __future__ import annotations

from pose_point_depth_mv import evaluate_native_slat_condition_only_mesh as _evaluator
from pose_point_depth_mv.native_slat_condition_only_objective_v2 import (
    build_native_slat_condition_only_objective_v2_components,
    validate_native_slat_condition_only_objective_v2_checkpoint,
)
from pose_point_depth_mv.no_vggt_ss_evidence import load_no_vggt_ss_evidence


REPORT_VERSION = (
    "pose_point_depth_mv.native_slat_condition_only_no_vggt_objective_v2_mesh.v1"
)


def main() -> None:
    _evaluator.REPORT_VERSION = REPORT_VERSION
    _evaluator.build_native_slat_condition_only_components = (
        build_native_slat_condition_only_objective_v2_components
    )
    _evaluator.validate_native_slat_condition_only_checkpoint = (
        validate_native_slat_condition_only_objective_v2_checkpoint
    )
    _evaluator.load_ss_evidence = load_no_vggt_ss_evidence
    base_upstream_binding = _evaluator.upstream_binding

    def no_vggt_upstream_binding(value):
        binding = base_upstream_binding(value)
        if value.get("exploratory") is True:
            binding.update(
                {
                    key: value[key]
                    for key in (
                        "formal",
                        "exploratory",
                        "diagnostic_scope",
                        "failed_quality_checks",
                        "source_calibration",
                        "source_calibration_sha256",
                        "scope_guard",
                    )
                    if key in value
                }
            )
        return binding

    _evaluator.upstream_binding = no_vggt_upstream_binding
    _evaluator.main()


if __name__ == "__main__":
    main()
