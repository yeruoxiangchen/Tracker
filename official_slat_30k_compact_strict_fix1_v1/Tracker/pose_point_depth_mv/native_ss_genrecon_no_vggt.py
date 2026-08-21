#!/usr/bin/env python3
"""No-VGGT Native SS protocol built on the unchanged v2 Flow architecture."""

from __future__ import annotations

from typing import Any, Iterable

from pose_point_depth_mv.dino_only_condition import (
    DINO_ONLY_CONTEXT_VERSION,
    validate_dino_only_lifting_contract,
)
from pose_point_depth_mv.native_ss_genrecon import (
    NATIVE_SS_GENRECON_VERSION,
    build_native_ss_genrecon_components as _build_v2_components,
    validate_genrecon_cache_contract as _validate_v2_cache_contract,
    validate_native_ss_genrecon_checkpoint as _validate_v2_checkpoint,
)
from pose_point_depth_mv.native_ss_genrecon import *  # noqa: F401,F403,E402


NATIVE_SS_NO_VGGT_VERSION = "pose_point_depth_mv.native_ss_genrecon_no_vggt.v1"
NATIVE_SS_NO_VGGT_CALIBRATION = (
    "pose_point_depth_mv.native_ss_no_vggt_calibration.v1"
)
NATIVE_SS_NO_VGGT_EVAL = "pose_point_depth_mv.native_ss_no_vggt_eval.v1"
NO_VGGT_MODEL_CONTRACT = {
    "version": DINO_ONLY_CONTEXT_VERSION,
    "architecture": "unchanged_native_ss_genrecon_v2",
    "stock_context": "deterministic_raw_dino_patch_tokens",
    "spatial_condition": "posed_multiview_dino_frustum_lifting",
    "pose": True,
    "point_cloud_role": "runtime_O_canonicalization_and_pose_frame",
    "point_cloud_as_direct_flow_token": False,
    "vggt_features": False,
    "vggt_depth": False,
    "vggt_model_executed": False,
    "checkpoint_migration": "all_trainable_parameter_names_and_shapes_unchanged",
}


def select_manifest_order_object_indices(
    rows: Iterable[dict[str, Any]], *, start: int = 0, end: int = 0
) -> list[int]:
    """Slice unique objects in the order frozen by the cache manifest."""

    ordered: list[int] = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        object_uid = str(row.get("object_uid", row["uid"]))
        if object_uid not in seen:
            seen.add(object_uid)
            ordered.append(index)
    stop = len(ordered) if int(end) <= 0 else int(end)
    selected = ordered[int(start) : stop]
    if not selected:
        raise ValueError(f"object slice [{start}:{end}] selected no objects")
    return selected


def validate_no_vggt_cache_contract(
    dataset: Any, *, training_config_hash: str | None = None
) -> dict[str, Any]:
    base = _validate_v2_cache_contract(dataset)
    no_vggt = validate_dino_only_lifting_contract(
        dataset, training_config_hash=training_config_hash
    )
    return {**base, "no_vggt": no_vggt, "model_context": NO_VGGT_MODEL_CONTRACT}


def validate_no_vggt_evaluation_cache_contract(
    dataset: Any, *, training_identity: dict[str, Any]
) -> dict[str, Any]:
    """Bind an evaluation cache to a direct or mixed training identity.

    A mixed manifest has its own aggregate config hash, so a held-out
    single-domain cache cannot equal that hash.  It must instead match one
    and only one component hash and the complete frozen component contract.
    """

    if not isinstance(training_identity, dict):
        raise TypeError("no-VGGT evaluation requires checkpoint data_identity")
    training_config_hash = str(training_identity.get("config_hash", ""))
    if not training_config_hash:
        raise RuntimeError("checkpoint data identity lacks config_hash")

    dataset_config_hash = str(getattr(dataset, "config_hash", ""))
    if dataset_config_hash == training_config_hash:
        contract = validate_no_vggt_cache_contract(
            dataset, training_config_hash=training_config_hash
        )
        contract["evaluation_training_binding"] = {
            "mode": "direct_training_config",
            "training_config_hash": training_config_hash,
            "evaluation_config_hash": dataset_config_hash,
        }
        return contract

    feature_contract = training_identity.get("feature_contract")
    if not isinstance(feature_contract, dict):
        raise RuntimeError(
            "DINO-only evaluation cache config differs from training and the "
            "checkpoint has no mixed-domain feature contract"
        )
    if str(feature_contract.get("config_hash", "")) != training_config_hash:
        raise RuntimeError("checkpoint feature contract config hash differs")
    mixed_domains = feature_contract.get("mixed_domains")
    if not isinstance(mixed_domains, dict) or not mixed_domains:
        raise RuntimeError(
            "DINO-only evaluation cache config differs from direct training config"
        )

    matches = [
        (str(name), binding)
        for name, binding in mixed_domains.items()
        if isinstance(binding, dict)
        and str(binding.get("config_hash", "")) == dataset_config_hash
    ]
    if len(matches) != 1:
        raise RuntimeError(
            "DINO-only evaluation cache must match exactly one frozen mixed "
            f"component: config_hash={dataset_config_hash!r}, matches={len(matches)}"
        )

    domain, binding = matches[0]
    observed = validate_dino_only_lifting_contract(dataset)
    frozen = binding.get("contract")
    if not isinstance(frozen, dict) or observed != frozen:
        raise RuntimeError(
            f"DINO-only evaluation contract differs from frozen {domain} component"
        )
    base = _validate_v2_cache_contract(dataset)
    return {
        **base,
        "no_vggt": observed,
        "model_context": NO_VGGT_MODEL_CONTRACT,
        "evaluation_training_binding": {
            "mode": "mixed_domain_component",
            "domain": domain,
            "training_config_hash": training_config_hash,
            "evaluation_config_hash": dataset_config_hash,
        },
    }


def build_native_ss_no_vggt_components(**kwargs: Any):
    from trellis import pipelines

    original = pipelines.TrellisVGGTTo3DPipeline
    pipelines.TrellisVGGTTo3DPipeline = pipelines.TrellisImageTo3DPipeline
    try:
        sampler, model, decoder, summary, defaults = _build_v2_components(**kwargs)
    finally:
        pipelines.TrellisVGGTTo3DPipeline = original
    summary = {
        **summary,
        "protocol_version": NATIVE_SS_NO_VGGT_VERSION,
        "input_context_contract": NO_VGGT_MODEL_CONTRACT,
        "vggt_model_executed": False,
    }
    return sampler, model, decoder, summary, defaults


def validate_native_ss_no_vggt_checkpoint(
    checkpoint: dict[str, Any], *, pretrained: str, allow_v2_parent: bool = True
) -> None:
    checkpoint_format = checkpoint.get("format")
    if checkpoint_format == NATIVE_SS_GENRECON_VERSION:
        if not allow_v2_parent:
            raise ValueError("v2 Full checkpoint is initialization-only for no-VGGT")
        _validate_v2_checkpoint(checkpoint, pretrained=pretrained)
        return
    if checkpoint_format != NATIVE_SS_NO_VGGT_VERSION:
        raise ValueError(f"unexpected no-VGGT Native SS format={checkpoint_format!r}")
    compatibility = dict(checkpoint)
    compatibility["format"] = NATIVE_SS_GENRECON_VERSION
    _validate_v2_checkpoint(compatibility, pretrained=pretrained)
    summary = dict(checkpoint.get("model_summary", {}))
    if summary.get("input_context_contract") != NO_VGGT_MODEL_CONTRACT:
        raise ValueError("no-VGGT Native SS input contract differs")
    identity_contract = (
        dict(checkpoint.get("data_identity", {}))
        .get("feature_contract", {})
        .get("no_vggt", {})
    )
    if identity_contract.get("vggt_feature_dim") != 0:
        raise ValueError("no-VGGT Native SS data identity still contains VGGT")


__all__ = [
    "NATIVE_SS_NO_VGGT_CALIBRATION",
    "NATIVE_SS_NO_VGGT_EVAL",
    "NATIVE_SS_NO_VGGT_VERSION",
    "NO_VGGT_MODEL_CONTRACT",
    "build_native_ss_no_vggt_components",
    "select_manifest_order_object_indices",
    "validate_native_ss_no_vggt_checkpoint",
    "validate_no_vggt_cache_contract",
    "validate_no_vggt_evaluation_cache_contract",
]
