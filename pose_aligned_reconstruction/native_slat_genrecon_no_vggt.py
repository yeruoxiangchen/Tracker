#!/usr/bin/env python3
"""No-VGGT Native-SLat protocol on the unchanged Native-SLat v2 model."""

from __future__ import annotations

from typing import Any

from pose_aligned_reconstruction.dino_only_condition import DINO_ONLY_CONTEXT_VERSION
from pose_aligned_reconstruction.native_slat_genrecon_v2 import (
    NATIVE_SLAT_GENRECON_V2_VERSION,
    build_native_slat_genrecon_v2_components as _build_v2_components,
    validate_native_slat_genrecon_v2_checkpoint as _validate_v2_checkpoint,
)
from pose_aligned_reconstruction.native_slat_genrecon_v2 import *  # noqa: F401,F403,E402


NATIVE_SLAT_NO_VGGT_VERSION = (
    "pose_point_depth_mv.native_slat_genrecon_no_vggt.v1"
)
NO_VGGT_SLAT_CONTRACT = {
    "version": DINO_ONLY_CONTEXT_VERSION,
    "architecture": "unchanged_native_slat_genrecon_v2",
    "stock_context": "per_view_raw_dino_patch_tokens",
    "spatial_condition": "posed_multiview_dino_frustum_lifting",
    "pose": True,
    "point_cloud_role": "runtime_O_canonicalization_and_pose_frame",
    "point_cloud_as_direct_flow_token": False,
    "vggt_features": False,
    "vggt_depth": False,
    "vggt_model_executed": False,
    "checkpoint_migration": "all_trainable_parameter_names_and_shapes_unchanged",
}


def build_native_slat_no_vggt_components(**kwargs: Any):
    from trellis import pipelines

    original = pipelines.TrellisVGGTTo3DPipeline
    pipelines.TrellisVGGTTo3DPipeline = pipelines.TrellisImageTo3DPipeline
    try:
        sampler, model, decoder, summary, defaults, normalization = (
            _build_v2_components(**kwargs)
        )
    finally:
        pipelines.TrellisVGGTTo3DPipeline = original
    summary = {
        **summary,
        "protocol_version": NATIVE_SLAT_NO_VGGT_VERSION,
        "input_context_contract": NO_VGGT_SLAT_CONTRACT,
        "vggt_model_executed": False,
    }
    return sampler, model, decoder, summary, defaults, normalization


def validate_native_slat_no_vggt_checkpoint(
    checkpoint: dict[str, Any],
    *,
    pretrained: str,
    stock_slat_freeze: dict[str, Any],
    upstream_native_ss: dict[str, Any],
    allow_v2_parent: bool = True,
) -> None:
    checkpoint_format = checkpoint.get("format")
    if checkpoint_format == NATIVE_SLAT_GENRECON_V2_VERSION:
        if not allow_v2_parent:
            raise ValueError("v2 Full SLat checkpoint is initialization-only")
        _validate_v2_checkpoint(
            checkpoint,
            pretrained=pretrained,
            stock_slat_freeze=stock_slat_freeze,
            upstream_native_ss=upstream_native_ss,
        )
        return
    if checkpoint_format != NATIVE_SLAT_NO_VGGT_VERSION:
        raise ValueError(f"unexpected no-VGGT Native-SLat format={checkpoint_format!r}")
    compatibility = dict(checkpoint)
    compatibility["format"] = NATIVE_SLAT_GENRECON_V2_VERSION
    _validate_v2_checkpoint(
        compatibility,
        pretrained=pretrained,
        stock_slat_freeze=stock_slat_freeze,
        upstream_native_ss=upstream_native_ss,
    )
    summary = dict(checkpoint.get("model_summary", {}))
    if summary.get("input_context_contract") != NO_VGGT_SLAT_CONTRACT:
        raise ValueError("no-VGGT Native-SLat input contract differs")


__all__ = [
    "NATIVE_SLAT_NO_VGGT_VERSION",
    "NO_VGGT_SLAT_CONTRACT",
    "build_native_slat_no_vggt_components",
    "validate_native_slat_no_vggt_checkpoint",
]
