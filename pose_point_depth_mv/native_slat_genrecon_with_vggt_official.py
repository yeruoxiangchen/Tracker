#!/usr/bin/env python3
"""Official Train2000 with-VGGT protocol on the unchanged v2 SLat model."""

from __future__ import annotations

from typing import Any

from pose_point_depth_mv.native_slat_genrecon_v2 import (
    NATIVE_SLAT_GENRECON_V2_VERSION,
    build_native_slat_genrecon_v2_components as _build_v2_components,
    validate_native_slat_genrecon_v2_checkpoint as _validate_v2_checkpoint,
)
from pose_point_depth_mv.native_slat_genrecon_v2 import *  # noqa: F401,F403,E402
from pose_point_depth_mv.proobjaverse_official_slat_with_vggt_cache import (
    WITH_VGGT_CONTEXT_VERSION,
)


NATIVE_SLAT_OFFICIAL_WITH_VGGT_VERSION = (
    "pose_point_depth_mv.native_slat_genrecon_official_with_vggt.v1"
)
OFFICIAL_WITH_VGGT_SLAT_CONTRACT = {
    "version": WITH_VGGT_CONTEXT_VERSION,
    "architecture": "unchanged_native_slat_genrecon_v2",
    "stock_floor": "V0",
    "stock_context": "native_reconviagen_vggt_plus_dinov2_slat_vggt_cond",
    "native_dino_sequence": "full_cls_register_patch_sequence",
    "spatial_condition": "posed_multiview_dino_frustum_lifting",
    "posed_dino_known_K_T": True,
    "vggt_camera_consumed": False,
    "vggt_depth_consumed": False,
    "native_ss_uses_vggt": False,
    "point_cloud_role": "runtime_O_canonicalization_and_pose_frame",
    "point_cloud_as_direct_flow_token": False,
    "step0_reference": "V0_native_reconviagen_stock_slat_with_slat_vggt_cond",
    "fresh_initialization_only": True,
}


def build_native_slat_official_with_vggt_components(**kwargs: Any):
    # The native context was materialized by the sidecar builder.  Training
    # needs the Stock SLat flow/sampler (and optional decoder), not a second
    # per-rank copy of VGGT.  Use the same lightweight construction path as the
    # frozen no-VGGT run; this changes only startup memory, never the SLat
    # forward or its externally supplied context.
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
        "format": NATIVE_SLAT_OFFICIAL_WITH_VGGT_VERSION,
        "protocol_version": NATIVE_SLAT_OFFICIAL_WITH_VGGT_VERSION,
        "input_context_contract": OFFICIAL_WITH_VGGT_SLAT_CONTRACT,
        "stock_floor": "V0",
        "step0_reference": (
            "native Stock SLat with the identical cached slat_vggt_cond"
        ),
        "vggt_model_executed_during_cache_build": True,
        "vggt_model_executed_during_training": False,
        "training_component_loader": (
            "legacy Stock pipeline construction with cached native context; "
            "no per-rank VGGT materialization"
        ),
        "vggt_camera_consumed": False,
        "native_ss_changed": False,
    }
    return sampler, model, decoder, summary, defaults, normalization


def validate_native_slat_official_with_vggt_checkpoint(
    checkpoint: dict[str, Any],
    *,
    pretrained: str,
    stock_slat_freeze: dict[str, Any],
    upstream_native_ss: dict[str, Any],
) -> None:
    if checkpoint.get("format") != NATIVE_SLAT_OFFICIAL_WITH_VGGT_VERSION:
        raise ValueError(
            "with-VGGT official training can resume only its own checkpoint "
            f"format, got={checkpoint.get('format')!r}"
        )
    compatibility = dict(checkpoint)
    compatibility["format"] = NATIVE_SLAT_GENRECON_V2_VERSION
    compatibility["model_summary"] = dict(checkpoint.get("model_summary", {}))
    compatibility["model_summary"]["format"] = NATIVE_SLAT_GENRECON_V2_VERSION
    _validate_v2_checkpoint(
        compatibility,
        pretrained=pretrained,
        stock_slat_freeze=stock_slat_freeze,
        upstream_native_ss=upstream_native_ss,
    )
    summary = dict(checkpoint.get("model_summary", {}))
    if summary.get("protocol_version") != NATIVE_SLAT_OFFICIAL_WITH_VGGT_VERSION:
        raise ValueError("with-VGGT official checkpoint protocol version differs")
    if summary.get("input_context_contract") != OFFICIAL_WITH_VGGT_SLAT_CONTRACT:
        raise ValueError("with-VGGT official checkpoint input contract differs")
    if summary.get("stock_floor") != "V0":
        raise ValueError("with-VGGT official checkpoint does not bind V0 Stock floor")
    data_identity = checkpoint.get("data_identity")
    if not isinstance(data_identity, dict):
        raise ValueError("with-VGGT official checkpoint lacks data identity")


__all__ = [
    "NATIVE_SLAT_OFFICIAL_WITH_VGGT_VERSION",
    "OFFICIAL_WITH_VGGT_SLAT_CONTRACT",
    "build_native_slat_official_with_vggt_components",
    "validate_native_slat_official_with_vggt_checkpoint",
]
