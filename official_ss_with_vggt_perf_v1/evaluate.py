"""Calibrate/evaluate official with-VGGT Native-SS on paired held-out cache."""

from __future__ import annotations

from official_ss_with_vggt_perf_v1.cache import (
    WithVGGTOfficialSSDataset,
    validate_official_ss_with_vggt_evaluation_cache_contract,
)
from official_ss_with_vggt_perf_v1.model import (
    CALIBRATION_FORMAT,
    EVAL_FORMAT,
    build_components,
    validate_checkpoint,
)
from pose_point_depth_mv import evaluate_native_ss_genrecon as _evaluator
from pose_point_depth_mv.native_ss_genrecon_no_vggt import (
    select_manifest_order_object_indices,
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
            raise RuntimeError("with-VGGT SS checkpoint identity was not loaded")
        if training_config_hash is not None and str(training_config_hash) != str(
            identity.get("config_hash", "")
        ):
            raise RuntimeError("with-VGGT SS evaluation training hash differs")
        return validate_official_ss_with_vggt_evaluation_cache_contract(
            dataset, training_identity=identity
        )

    _evaluator.PoseLiftingCacheDataset = WithVGGTOfficialSSDataset
    _evaluator.NATIVE_SS_GENRECON_CALIBRATION = CALIBRATION_FORMAT
    _evaluator.NATIVE_SS_GENRECON_EVAL = EVAL_FORMAT
    _evaluator.load_runtime = load_runtime_with_identity
    _evaluator.validate_genrecon_cache_contract = validate_evaluation_cache
    _evaluator.validate_native_ss_genrecon_checkpoint = validate_checkpoint
    _evaluator.build_native_ss_genrecon_components = build_components
    _evaluator.select_object_indices = select_manifest_order_object_indices
    _evaluator.main()


if __name__ == "__main__":
    main()

