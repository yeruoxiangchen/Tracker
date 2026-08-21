#!/usr/bin/env python3
"""Train no-VGGT Native-SLat with the unchanged v2 optimization loop."""

from __future__ import annotations

import sys

from ar_ss_flow.local_pose_lifting_flow import PoseLiftingCacheDataset
from pose_point_depth_mv import evaluate_native_ss_stock_slat_mesh as _ss_evidence
from pose_point_depth_mv import train_native_slat_genrecon as _v2_train
from pose_point_depth_mv.dino_only_condition import (
    validate_dino_only_lifting_contract,
)
from pose_point_depth_mv.native_slat_genrecon_no_vggt import (
    NATIVE_SLAT_NO_VGGT_VERSION,
    build_native_slat_no_vggt_components,
    validate_native_slat_no_vggt_checkpoint,
)
from pose_point_depth_mv.native_ss_genrecon_no_vggt import NATIVE_SS_NO_VGGT_EVAL
from pose_point_depth_mv.no_vggt_ss_evidence import load_no_vggt_ss_evidence


_base_upstream_binding = _v2_train.upstream_binding


def _argument(name: str) -> str | None:
    try:
        return sys.argv[sys.argv.index(name) + 1]
    except (ValueError, IndexError):
        prefix = f"{name}="
        return next((value[len(prefix) :] for value in sys.argv if value.startswith(prefix)), None)


def no_vggt_upstream_binding(value):
    """Project a full no-VGGT deployment into the historical training identity."""

    binding = _base_upstream_binding(value)
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
            }
        )
    return binding


def main() -> None:
    help_requested = "--help" in sys.argv or "-h" in sys.argv
    architecture = _argument("--architecture")
    if architecture is None:
        sys.argv.extend(("--architecture", "v2"))
    elif architecture != "v2":
        raise ValueError("no-VGGT v1 freezes the Native-SLat v2 architecture")
    lifting_manifest = _argument("--lifting_cache_manifest")
    if not lifting_manifest and not help_requested:
        raise ValueError("--lifting_cache_manifest is required")
    if lifting_manifest:
        lifting = PoseLiftingCacheDataset(lifting_manifest, indices="all")
        validate_dino_only_lifting_contract(lifting)

    _v2_train.NATIVE_SLAT_GENRECON_V2_VERSION = NATIVE_SLAT_NO_VGGT_VERSION
    _v2_train.validate_native_slat_genrecon_v2_checkpoint = (
        validate_native_slat_no_vggt_checkpoint
    )
    _v2_train.build_native_slat_genrecon_v2_components = (
        build_native_slat_no_vggt_components
    )
    _ss_evidence.NATIVE_SS_GENRECON_EVAL = NATIVE_SS_NO_VGGT_EVAL
    _v2_train.load_ss_evidence = load_no_vggt_ss_evidence
    _v2_train.upstream_binding = no_vggt_upstream_binding
    _v2_train.main()


if __name__ == "__main__":
    main()
