#!/usr/bin/env python3
"""Versioned no-VGGT Native-SLat objective-v2 protocol.

This keeps the condition-only architecture unchanged.  The new version binds
the timestep schedule and optional frozen-decoder/Stock-trust objectives so a
schedule control checkpoint cannot be confused with the older flow-only run.
"""

from __future__ import annotations

from typing import Any

from pose_point_depth_mv.native_slat_condition_only import (
    NATIVE_SLAT_CONDITION_ONLY_VERSION,
)
from pose_point_depth_mv.native_slat_condition_only_no_vggt import (
    NATIVE_SLAT_CONDITION_ONLY_NO_VGGT_VERSION,
    NO_VGGT_CONDITION_ONLY_CONTRACT,
    build_native_slat_condition_only_no_vggt_components,
    validate_native_slat_condition_only_no_vggt_checkpoint,
)
from pose_point_depth_mv.native_slat_decoder_geometry import (
    DECODER_GEOMETRY_LOSS_VERSION,
)


NATIVE_SLAT_CONDITION_ONLY_OBJECTIVE_V2_VERSION = (
    "pose_point_depth_mv.native_slat_condition_only_no_vggt_objective_v2.v1"
)


def build_native_slat_condition_only_objective_v2_components(**kwargs: Any):
    sampler, model, decoder, summary, defaults, normalization = (
        build_native_slat_condition_only_no_vggt_components(**kwargs)
    )
    summary = {
        **summary,
        "format": NATIVE_SLAT_CONDITION_ONLY_OBJECTIVE_V2_VERSION,
        "protocol_version": NATIVE_SLAT_CONDITION_ONLY_OBJECTIVE_V2_VERSION,
        "objective_family": DECODER_GEOMETRY_LOSS_VERSION,
    }
    return sampler, model, decoder, summary, defaults, normalization


def validate_native_slat_condition_only_objective_v2_checkpoint(
    checkpoint: dict[str, Any],
    *,
    pretrained: str,
    stock_slat_freeze: dict[str, Any],
    upstream_native_ss: dict[str, Any],
) -> None:
    if checkpoint.get("format") != NATIVE_SLAT_CONDITION_ONLY_OBJECTIVE_V2_VERSION:
        raise ValueError(
            "unexpected objective-v2 condition-only format="
            f"{checkpoint.get('format')!r}"
        )
    compatibility = dict(checkpoint)
    compatibility["format"] = NATIVE_SLAT_CONDITION_ONLY_NO_VGGT_VERSION
    compatibility_summary = dict(checkpoint.get("model_summary", {}))
    compatibility_summary.update(
        {
            "format": NATIVE_SLAT_CONDITION_ONLY_VERSION,
            "protocol_version": NATIVE_SLAT_CONDITION_ONLY_NO_VGGT_VERSION,
            "input_context_contract": NO_VGGT_CONDITION_ONLY_CONTRACT,
            "vggt_model_executed": False,
        }
    )
    compatibility["model_summary"] = compatibility_summary
    validate_native_slat_condition_only_no_vggt_checkpoint(
        compatibility,
        pretrained=pretrained,
        stock_slat_freeze=stock_slat_freeze,
        upstream_native_ss=upstream_native_ss,
    )

    args = checkpoint.get("args")
    summary = checkpoint.get("model_summary")
    if not isinstance(args, dict) or not isinstance(summary, dict):
        raise ValueError("objective-v2 checkpoint lacks args/model_summary")
    objective = summary.get("training_objective")
    schedule = summary.get("t_schedule")
    if not isinstance(objective, dict) or not isinstance(schedule, dict):
        raise ValueError("objective-v2 bindings are missing")
    if objective.get("version") != DECODER_GEOMETRY_LOSS_VERSION:
        raise ValueError("objective-v2 decoder geometry version differs")
    expected_objective = {
        "decoder_geometry_weight": float(args.get("decoder_geometry_weight", -1.0)),
        "stock_flow_trust_weight": float(
            args.get("stock_flow_trust_weight", -1.0)
        ),
        "stock_flow_required_improvement": float(
            args.get("stock_flow_required_improvement", -1.0)
        ),
        "stock_trust_weight": float(args.get("stock_trust_weight", -1.0)),
        "stock_required_improvement": float(
            args.get("stock_required_improvement", -1.0)
        ),
        "geometry_event_probability": float(
            args.get("geometry_event_probability", -1.0)
        ),
        "geometry_t_max": float(args.get("geometry_t_max", -1.0)),
    }
    for key, expected in expected_objective.items():
        if float(objective.get(key, float("nan"))) != expected:
            raise ValueError(f"objective-v2 binding differs for {key}")
    if schedule.get("name") != args.get("t_schedule"):
        raise ValueError("objective-v2 timestep schedule differs")
    expected_rng = (
        "separate deterministic rank/micro-step generator"
        if args.get("separate_t_rng") is True
        else "legacy global torch RNG"
    )
    if schedule.get("rng") != expected_rng:
        raise ValueError("objective-v2 timestep RNG differs")
    if float(schedule.get("uniform_probability", -1.0)) != float(
        args.get("t_uniform_probability", -2.0)
    ):
        raise ValueError("objective-v2 mixed-schedule probability differs")
    if summary.get("input_context_contract") != NO_VGGT_CONDITION_ONLY_CONTRACT:
        raise ValueError("objective-v2 no-VGGT input contract differs")


__all__ = [
    "NATIVE_SLAT_CONDITION_ONLY_OBJECTIVE_V2_VERSION",
    "build_native_slat_condition_only_objective_v2_components",
    "validate_native_slat_condition_only_objective_v2_checkpoint",
]
