#!/usr/bin/env python3
"""Train no-VGGT Native-SLat with the unchanged v2 optimization loop."""

from __future__ import annotations

import json
from pathlib import Path
import sys

from ar_ss_flow.local_pose_lifting_flow import PoseLiftingCacheDataset
from pose_aligned_reconstruction import evaluate_native_ss_stock_slat_mesh as _ss_evidence
from pose_aligned_reconstruction import train_native_slat_genrecon as _v2_train
from pose_aligned_reconstruction.dino_only_condition import (
    validate_dino_only_lifting_contract,
)
from pose_aligned_reconstruction.native_slat_genrecon_no_vggt import (
    NATIVE_SLAT_NO_VGGT_VERSION,
    build_native_slat_no_vggt_components,
    validate_native_slat_no_vggt_checkpoint,
)
from pose_aligned_reconstruction.native_ss_genrecon_no_vggt import NATIVE_SS_NO_VGGT_EVAL
from pose_aligned_reconstruction.no_vggt_ss_evidence import load_no_vggt_ss_evidence
from pose_aligned_reconstruction.proobjaverse_official_slat_compact import (
    COMPACT_LIFTING_MANIFEST_FORMAT,
    CompactNativeConditionBackend,
    is_compact_manifest_pair,
)


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


def validate_no_vggt_lifting_preflight(
    cache_manifest: str | None, lifting_manifest: str
):
    """Validate legacy or compact lifting data before the shared trainer starts.

    Legacy manifests intentionally retain the historical
    ``PoseLiftingCacheDataset`` path.  A compact lifting manifest is instead
    validated through the exact paired compact backend, whose ``lifting_view``
    exposes the same immutable DINO-only contract without pretending that the
    compact manifest is a legacy pose-lifting manifest.
    """

    lifting_path = Path(lifting_manifest).expanduser().resolve()
    payload = json.loads(lifting_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {lifting_path}")
    if payload.get("format") != COMPACT_LIFTING_MANIFEST_FORMAT:
        lifting = PoseLiftingCacheDataset(lifting_manifest, indices="all")
        return validate_dino_only_lifting_contract(lifting)
    if not cache_manifest:
        raise ValueError(
            "--cache_manifest is required with a compact lifting manifest"
        )
    if not is_compact_manifest_pair(cache_manifest, lifting_manifest):
        raise ValueError("compact SLat/lifting manifests must be a matched pair")
    compact = CompactNativeConditionBackend(
        cache_manifest, lifting_manifest, indices="all"
    )
    return validate_dino_only_lifting_contract(compact.lifting_view)


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
        validate_no_vggt_lifting_preflight(
            _argument("--cache_manifest"), lifting_manifest
        )

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
