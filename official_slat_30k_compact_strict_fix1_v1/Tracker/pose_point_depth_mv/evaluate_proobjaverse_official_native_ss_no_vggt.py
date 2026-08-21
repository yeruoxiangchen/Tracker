#!/usr/bin/env python3
"""Calibrate/evaluate retrained Native SS on held-out official targets."""

from __future__ import annotations

from pose_point_depth_mv import evaluate_native_ss_genrecon as _evaluator
from pose_point_depth_mv.native_ss_genrecon_no_vggt import (
    build_native_ss_no_vggt_components,
    select_manifest_order_object_indices,
    validate_native_ss_no_vggt_checkpoint,
)
from pose_point_depth_mv.proobjaverse_official_ss import (
    OFFICIAL_SS_CALIBRATION,
    OFFICIAL_SS_EVAL,
    validate_official_ss_evaluation_cache_contract,
)


def main() -> None:
    runtime: dict[str, object] = {}
    original_load_runtime = _evaluator.load_runtime

    def load_runtime_with_identity(*args, **kwargs):
        loaded = original_load_runtime(*args, **kwargs)
        runtime["training_identity"] = dict(loaded[0].get("data_identity", {}))
        return loaded

    def validate_evaluation_cache(dataset, *, training_config_hash=None):
        identity = runtime.get("training_identity")
        if not isinstance(identity, dict):
            raise RuntimeError("checkpoint training identity was not loaded")
        if training_config_hash is not None and str(training_config_hash) != str(
            identity.get("config_hash", "")
        ):
            raise RuntimeError("evaluation training hash argument differs")
        return validate_official_ss_evaluation_cache_contract(
            dataset, training_identity=identity
        )

    _evaluator.NATIVE_SS_GENRECON_CALIBRATION = OFFICIAL_SS_CALIBRATION
    _evaluator.NATIVE_SS_GENRECON_EVAL = OFFICIAL_SS_EVAL
    _evaluator.load_runtime = load_runtime_with_identity
    _evaluator.validate_genrecon_cache_contract = validate_evaluation_cache
    _evaluator.validate_native_ss_genrecon_checkpoint = (
        validate_native_ss_no_vggt_checkpoint
    )
    _evaluator.build_native_ss_genrecon_components = (
        build_native_ss_no_vggt_components
    )
    _evaluator.select_object_indices = select_manifest_order_object_indices
    _evaluator.main()


if __name__ == "__main__":
    main()
