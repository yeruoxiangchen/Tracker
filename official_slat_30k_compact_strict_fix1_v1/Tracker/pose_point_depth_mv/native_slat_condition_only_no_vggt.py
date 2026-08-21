#!/usr/bin/env python3
"""Strict DINO-only protocol for the LoRA-free Native-SLat model."""

from __future__ import annotations

from typing import Any

from pose_point_depth_mv.dino_only_condition import DINO_ONLY_CONTEXT_VERSION
from pose_point_depth_mv.native_slat_condition_only import (
    NATIVE_SLAT_CONDITION_ONLY_VERSION,
    build_native_slat_condition_only_components as _build_condition_only_components,
    validate_native_slat_condition_only_checkpoint as _validate_condition_only_checkpoint,
)
from pose_point_depth_mv.native_slat_condition_only import *  # noqa: F401,F403,E402


NATIVE_SLAT_CONDITION_ONLY_NO_VGGT_VERSION = (
    "pose_point_depth_mv.native_slat_condition_only_no_vggt.v1"
)
NO_VGGT_CONDITION_ONLY_CONTRACT = {
    "version": DINO_ONLY_CONTEXT_VERSION,
    "architecture": "native_slat_condition_only",
    "stock_context": "per_view_raw_dino_patch_tokens",
    "spatial_condition": "posed_multiview_dino_frustum_lifting",
    "pose": True,
    "point_cloud_role": "runtime_O_canonicalization_and_pose_frame",
    "point_cloud_as_direct_flow_token": False,
    "vggt_features": False,
    "vggt_depth": False,
    "vggt_model_executed": False,
    "flow_lora": False,
}


def build_native_slat_condition_only_no_vggt_components(**kwargs: Any):
    """Build the unchanged condition-only model without loading a VGGT pipeline."""

    from trellis import pipelines

    original = pipelines.TrellisVGGTTo3DPipeline
    pipelines.TrellisVGGTTo3DPipeline = pipelines.TrellisImageTo3DPipeline
    try:
        sampler, model, decoder, summary, defaults, normalization = (
            _build_condition_only_components(**kwargs)
        )
    finally:
        pipelines.TrellisVGGTTo3DPipeline = original
    summary = {
        **summary,
        "protocol_version": NATIVE_SLAT_CONDITION_ONLY_NO_VGGT_VERSION,
        "input_context_contract": NO_VGGT_CONDITION_ONLY_CONTRACT,
        "vggt_model_executed": False,
    }
    return sampler, model, decoder, summary, defaults, normalization


def validate_native_slat_condition_only_no_vggt_checkpoint(
    checkpoint: dict[str, Any],
    *,
    pretrained: str,
    stock_slat_freeze: dict[str, Any],
    upstream_native_ss: dict[str, Any],
) -> None:
    if checkpoint.get("format") != NATIVE_SLAT_CONDITION_ONLY_NO_VGGT_VERSION:
        raise ValueError(
            "unexpected no-VGGT condition-only format="
            f"{checkpoint.get('format')!r}"
        )
    compatibility = dict(checkpoint)
    compatibility["format"] = NATIVE_SLAT_CONDITION_ONLY_VERSION
    _validate_condition_only_checkpoint(
        compatibility,
        pretrained=pretrained,
        stock_slat_freeze=stock_slat_freeze,
        upstream_native_ss=upstream_native_ss,
    )
    summary = dict(checkpoint.get("model_summary", {}))
    if summary.get("input_context_contract") != NO_VGGT_CONDITION_ONLY_CONTRACT:
        raise ValueError("no-VGGT condition-only input contract differs")
    if summary.get("vggt_model_executed") is not False:
        raise ValueError("no-VGGT condition-only checkpoint executed VGGT")


__all__ = [
    "NATIVE_SLAT_CONDITION_ONLY_NO_VGGT_VERSION",
    "NO_VGGT_CONDITION_ONLY_CONTRACT",
    "build_native_slat_condition_only_no_vggt_components",
    "validate_native_slat_condition_only_no_vggt_checkpoint",
]
