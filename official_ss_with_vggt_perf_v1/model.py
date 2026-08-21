"""Official with-VGGT Native-SS protocol on the unchanged v2 flow."""

from __future__ import annotations

from typing import Any

from official_ss_with_vggt_perf_v1.cache import MODEL_CONTEXT_CONTRACT
from pose_point_depth_mv.native_ss_genrecon import (
    NATIVE_SS_GENRECON_VERSION,
    build_native_ss_genrecon_components as _build_v2_components,
    validate_native_ss_genrecon_checkpoint as _validate_v2_checkpoint,
)
from pose_point_depth_mv.native_ss_genrecon import *  # noqa: F401,F403,E402


VERSION = "official_ss_with_vggt_perf_v1.native_ss_genrecon.v2"
CALIBRATION_FORMAT = "official_ss_with_vggt_perf_v1.calibration.v1"
EVAL_FORMAT = "official_ss_with_vggt_perf_v1.eval.v1"
EVAL_AGGREGATE_FORMAT = "official_ss_with_vggt_perf_v1.eval_aggregate.v1"

STEP0_STRAIGHT_THROUGH_CONTRACT = {
    "version": "official_ss_with_vggt_perf_v1.step0_exact_reference.v1",
    "forward": "reference.detach() + (value - value.detach())",
    "forward_reference_exact": True,
    "gradient_to_adapted_value": True,
    "gradient_to_stock_reference": False,
    "active_only_while_condition_projection_and_lora_B_are_exact_zero": True,
    "mathematical_training_semantics_changed": False,
}


def _exact_forward_straight_through(value, reference):
    """Return Stock exactly while retaining only the adapted-value gradient.

    The historical ``value + (reference - value).detach()`` expression has the
    same real-number algebra, but cancellation can leave a one-ULP forward
    residual.  Subtracting a tensor from its own detached value creates exact
    zero before adding the reference, without changing the backward contract.
    """

    if value.shape != reference.shape:
        raise ValueError("straight-through reference shape mismatch")
    return reference.detach() + (value - value.detach())


def _install_exact_step0_reference() -> None:
    # Dedicated-entrypoint runtime override only.  The immutable shared
    # no-VGGT implementation and all historical checkpoints remain untouched.
    from pose_point_depth_mv import native_ss_genrecon as native_v2

    native_v2._straight_through_reference = _exact_forward_straight_through


def build_components(**kwargs: Any):
    """Build Stock SS Flow without materializing VGGT in every train rank.

    The exact native context was materialized by the sidecar builder.  The
    pipeline is needed here only for the frozen flow/sampler/optional decoder.
    Substituting the lightweight loader changes startup memory, not weights or
    the externally supplied cross-attention condition.
    """

    from trellis import pipelines

    _install_exact_step0_reference()

    original = pipelines.TrellisVGGTTo3DPipeline
    pipelines.TrellisVGGTTo3DPipeline = pipelines.TrellisImageTo3DPipeline
    try:
        sampler, model, decoder, summary, defaults = _build_v2_components(**kwargs)
    finally:
        pipelines.TrellisVGGTTo3DPipeline = original
    summary = {
        **summary,
        "format": VERSION,
        "protocol_version": VERSION,
        "input_context_contract": MODEL_CONTEXT_CONTRACT,
        "stock_floor": "VSS0",
        "step0_reference": (
            "native ReconViaGen Stock SS with identical cached ss_vggt_cond"
        ),
        "fresh_initialization_only": True,
        "vggt_model_executed_during_cache_build": True,
        "vggt_model_executed_during_training": False,
        "vggt_camera_consumed": False,
        "known_pose_dino_branch_unchanged": True,
        "official_ss_target_unchanged": True,
        "step0_straight_through_contract": STEP0_STRAIGHT_THROUGH_CONTRACT,
        "runtime_input_policy": {
            "ddp_device_ids": None,
            "caller_places_model_inputs": True,
            "complete_lifting_sample_stays_cpu_until_projection": True,
            "vggt_context_transferred_explicitly_once": True,
            "scientific_math_changed": False,
        },
    }
    return sampler, model, decoder, summary, defaults


def validate_checkpoint(
    checkpoint: dict[str, Any], *, pretrained: str
) -> None:
    if checkpoint.get("format") != VERSION:
        raise ValueError(
            "official with-VGGT SS accepts only its own checkpoint format; "
            f"got={checkpoint.get('format')!r}"
        )
    compatibility = dict(checkpoint)
    compatibility["format"] = NATIVE_SS_GENRECON_VERSION
    _validate_v2_checkpoint(compatibility, pretrained=pretrained)
    summary = checkpoint.get("model_summary")
    if not isinstance(summary, dict):
        raise ValueError("official with-VGGT SS checkpoint lacks model summary")
    required = {
        "protocol_version": VERSION,
        "input_context_contract": MODEL_CONTEXT_CONTRACT,
        "stock_floor": "VSS0",
        "fresh_initialization_only": True,
        "vggt_model_executed_during_training": False,
        "vggt_camera_consumed": False,
        "known_pose_dino_branch_unchanged": True,
        "official_ss_target_unchanged": True,
        "step0_straight_through_contract": STEP0_STRAIGHT_THROUGH_CONTRACT,
        "runtime_input_policy": {
            "ddp_device_ids": None,
            "caller_places_model_inputs": True,
            "complete_lifting_sample_stays_cpu_until_projection": True,
            "vggt_context_transferred_explicitly_once": True,
            "scientific_math_changed": False,
        },
    }
    mismatch = {
        key: (summary.get(key), value)
        for key, value in required.items()
        if summary.get(key) != value
    }
    if mismatch:
        raise ValueError(f"official with-VGGT SS checkpoint contract differs={mismatch}")
    feature = checkpoint.get("data_identity", {}).get("feature_contract", {})
    if not isinstance(feature, dict) or not isinstance(
        feature.get("with_vggt_ss"), dict
    ):
        raise ValueError("official with-VGGT SS checkpoint lacks sidecar identity")
    if feature["with_vggt_ss"].get("model_context") != MODEL_CONTEXT_CONTRACT:
        raise ValueError("official with-VGGT SS checkpoint model context differs")


__all__ = [
    "CALIBRATION_FORMAT",
    "EVAL_FORMAT",
    "EVAL_AGGREGATE_FORMAT",
    "STEP0_STRAIGHT_THROUGH_CONTRACT",
    "VERSION",
    "build_components",
    "validate_checkpoint",
]
