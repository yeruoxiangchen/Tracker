#!/usr/bin/env python3
"""Calibrate/evaluate no-VGGT Native SS on DINO-only lifting caches."""

from __future__ import annotations

from pose_point_depth_mv import evaluate_native_ss_genrecon as _v2_eval
from pose_point_depth_mv.native_ss_genrecon_no_vggt import (
    NATIVE_SS_NO_VGGT_CALIBRATION,
    NATIVE_SS_NO_VGGT_EVAL,
    build_native_ss_no_vggt_components,
    select_manifest_order_object_indices,
    validate_native_ss_no_vggt_checkpoint,
    validate_no_vggt_evaluation_cache_contract,
)


def main() -> None:
    runtime: dict[str, object] = {}
    original_load_runtime = _v2_eval.load_runtime

    def load_runtime_with_training_identity(*args, **kwargs):
        loaded = original_load_runtime(*args, **kwargs)
        runtime["training_identity"] = dict(loaded[0].get("data_identity", {}))
        return loaded

    def validate_evaluation_cache(dataset, *, training_config_hash=None):
        training_identity = runtime.get("training_identity")
        if not isinstance(training_identity, dict):
            raise RuntimeError("checkpoint training identity was not loaded")
        if training_config_hash is not None and str(training_config_hash) != str(
            training_identity.get("config_hash", "")
        ):
            raise RuntimeError("evaluation training config hash argument differs")
        return validate_no_vggt_evaluation_cache_contract(
            dataset, training_identity=training_identity
        )

    _v2_eval.NATIVE_SS_GENRECON_CALIBRATION = NATIVE_SS_NO_VGGT_CALIBRATION
    _v2_eval.NATIVE_SS_GENRECON_EVAL = NATIVE_SS_NO_VGGT_EVAL
    _v2_eval.load_runtime = load_runtime_with_training_identity
    _v2_eval.validate_genrecon_cache_contract = validate_evaluation_cache
    _v2_eval.validate_native_ss_genrecon_checkpoint = (
        validate_native_ss_no_vggt_checkpoint
    )
    _v2_eval.build_native_ss_genrecon_components = (
        build_native_ss_no_vggt_components
    )
    # The source-balanced val64 manifest freezes checkpoint/CFG/final phases by
    # row order.  UID sorting would silently destroy those phase assignments.
    _v2_eval.select_object_indices = select_manifest_order_object_indices
    _v2_eval.main()


if __name__ == "__main__":
    main()
