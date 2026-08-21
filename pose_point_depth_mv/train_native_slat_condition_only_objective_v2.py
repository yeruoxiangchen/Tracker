#!/usr/bin/env python3
"""Train versioned no-VGGT Native-SLat timestep/geometry objective v2."""

from __future__ import annotations

import sys

from ar_ss_flow.local_pose_lifting_flow import PoseLiftingCacheDataset
from pose_point_depth_mv import train_native_slat_condition_only as _trainer
from pose_point_depth_mv.dino_only_condition import validate_dino_only_lifting_contract
from pose_point_depth_mv.native_slat_condition_only_objective_v2 import (
    NATIVE_SLAT_CONDITION_ONLY_OBJECTIVE_V2_VERSION,
    build_native_slat_condition_only_objective_v2_components,
    validate_native_slat_condition_only_objective_v2_checkpoint,
)
from pose_point_depth_mv.no_vggt_ss_evidence import load_no_vggt_ss_evidence


def _argument(name: str) -> str | None:
    try:
        return sys.argv[sys.argv.index(name) + 1]
    except (ValueError, IndexError):
        prefix = f"{name}="
        return next(
            (value[len(prefix) :] for value in sys.argv if value.startswith(prefix)),
            None,
        )


def main() -> None:
    help_requested = "--help" in sys.argv or "-h" in sys.argv
    lifting_manifest = _argument("--lifting_cache_manifest")
    if not lifting_manifest and not help_requested:
        raise ValueError("--lifting_cache_manifest is required")
    if lifting_manifest:
        lifting = PoseLiftingCacheDataset(lifting_manifest, indices="all")
        validate_dino_only_lifting_contract(lifting)

    _trainer.NATIVE_SLAT_CONDITION_ONLY_VERSION = (
        NATIVE_SLAT_CONDITION_ONLY_OBJECTIVE_V2_VERSION
    )
    _trainer.build_native_slat_condition_only_components = (
        build_native_slat_condition_only_objective_v2_components
    )
    _trainer.validate_native_slat_condition_only_checkpoint = (
        validate_native_slat_condition_only_objective_v2_checkpoint
    )
    _trainer.load_ss_evidence = load_no_vggt_ss_evidence
    base_upstream_binding = _trainer.upstream_binding

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

    _trainer.upstream_binding = no_vggt_upstream_binding
    _trainer.main()


if __name__ == "__main__":
    main()
