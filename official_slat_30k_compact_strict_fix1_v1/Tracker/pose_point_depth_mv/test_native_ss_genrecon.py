from __future__ import annotations

import inspect
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image
import torch
from torch import nn

from ar_ss_flow.shared_object_preprocessing import (
    prepare_shared_object_views,
    shared_preprocessing_contract,
    transform_intrinsics,
)
from pose_point_depth_mv.evaluate_native_ss_genrecon import (
    aggregate_records,
    candidate_checks,
)
from pose_point_depth_mv.native_ss_genrecon import (
    NATIVE_SS_GENRECON_CFG,
    NATIVE_SS_GENRECON_PROJECTION,
    NATIVE_SS_GENRECON_TRAINING,
    NATIVE_SS_GENRECON_VERSION,
    GenreconViewAggregator,
    NativeSSCalibratedCFGFlow,
    NativeSSGenreconFlow,
    project_frustum_dino,
    require_disjoint_object_uids,
    select_dino_features,
    select_object_indices,
    validate_genrecon_cache_contract,
    validate_native_ss_genrecon_checkpoint,
)
from pose_point_depth_mv.train_native_ss_genrecon import (
    ema_ramp_decay,
    initialize_ema_state,
    update_ema_state,
    warmup_factor,
)


def synthetic_sample() -> dict:
    views = 2
    c2w = torch.eye(4).repeat(views, 1, 1)
    c2w[:, 2, 3] = -2.0
    return {
        "visual_patch_features": torch.cat(
            (
                torch.zeros((views, 4, 2048)),
                torch.stack(
                    (
                        torch.ones((4, 1024)),
                        torch.full((4, 1024), 3.0),
                    )
                ),
            ),
            dim=-1,
        ),
        "intrinsics": torch.tensor(
            [[[2.0, 0.0, 1.5], [0.0, 2.0, 1.5], [0.0, 0.0, 1.0]]]
        ).repeat(views, 1, 1),
        "extrinsics": c2w,
        "predicted_depth": torch.ones((views, 4, 4)),
        "grid_transform": "pixal3d_rotation",
        "extrinsics_type": "c2w",
        "camera_forward_sign": 1.0,
    }


def test_selects_only_trailing_dino_channels() -> None:
    features = synthetic_sample()["visual_patch_features"]
    selected = select_dino_features(features)
    assert selected.shape == (2, 4, 1024)
    assert float(selected[0].mean()) == 1.0
    assert float(selected[1].mean()) == 3.0


def test_frustum_projection_and_aggregation_are_view_permutation_invariant() -> None:
    sample = synthetic_sample()
    projected, valid, stats = project_frustum_dino(
        sample, device=torch.device("cpu")
    )
    assert projected.shape == (2, 4096, 1024)
    assert valid.shape == (2, 4096)
    assert float(stats["supported_fraction"]) > 0.0
    aggregator = GenreconViewAggregator(1024)
    output, _ = aggregator(projected, valid)
    reversed_output, _ = aggregator(projected.flip(0), valid.flip(0))
    torch.testing.assert_close(output, reversed_output, rtol=0.0, atol=1.0e-6)


class DummyAdaptedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def stock_prediction(self, x, _t, _condition):
        return x

    def adapted_prediction(
        self,
        x,
        _t,
        _condition,
        sample,
        *,
        projection_mode,
    ):
        self.calls.append(sample is not None)
        increment = 1.0 if sample is not None else 2.0
        prediction = x + increment
        return prediction, {"flow_delta_rms": x.new_tensor(abs(increment))}


def test_standard_cfg_wrapper_adapts_both_branches_and_stock_bypass() -> None:
    model = DummyAdaptedModel()
    positive = torch.ones((1, 2, 3))
    negative = torch.zeros_like(positive)
    x = torch.zeros((1, 1))
    t = torch.ones((1,))
    enabled = NativeSSCalibratedCFGFlow(
        model, positive, {}, enabled=True
    )
    torch.testing.assert_close(enabled(x, t, positive), torch.full_like(x, 1.0))
    torch.testing.assert_close(enabled(x, t, negative), torch.full_like(x, 2.0))
    assert enabled.positive_calls == 1
    assert enabled.negative_calls == 1
    assert model.calls == [True, False]
    assert enabled.summary()["condition_scale_policy"] == "learned_projection_only"
    assert enabled.summary()["post_cfg_cap"] is False
    disabled = NativeSSCalibratedCFGFlow(
        model, positive, {}, enabled=False
    )
    torch.testing.assert_close(disabled(x, t, positive), x, rtol=0.0, atol=0.0)
    torch.testing.assert_close(disabled(x, t, negative), x, rtol=0.0, atol=0.0)


class UnconditionalProbeFlow(NativeSSGenreconFlow):
    def __init__(self) -> None:
        nn.Module.__init__(self)
        self.condition_channels = 3
        self.seen_condition = None

    def _adapted_core_forward(
        self,
        x,
        _t,
        _condition,
        condition_3d,
    ):
        self.seen_condition = condition_3d
        return x

    def stock_prediction(self, x, _t, _condition):
        return x

    def _lora_outputs_exact_zero(self) -> bool:
        return False

    @property
    def block_condition(self):
        return type("ProjectionProbe", (), {"exact_zero": lambda self: False})()


def test_unconditional_branch_uses_zero_3d_condition_not_missing_projection() -> None:
    model = UnconditionalProbeFlow()
    x = torch.ones((1, 1, 2, 2, 2))
    prediction, stats = model.adapted_prediction(
        x,
        torch.ones((1,)),
        torch.zeros((1, 2, 3)),
        None,
    )
    torch.testing.assert_close(prediction, x)
    assert model.seen_condition is not None
    assert tuple(model.seen_condition.shape) == (1, 4096, 3)
    assert int(torch.count_nonzero(model.seen_condition)) == 0
    assert float(stats["condition_present"]) == 0.0


def valid_checkpoint() -> dict:
    return {
        "format": NATIVE_SS_GENRECON_VERSION,
        "step": 1,
        "model_summary": {
            "pretrained": "test",
            "training_semantics": NATIVE_SS_GENRECON_TRAINING,
            "cfg_semantics": NATIVE_SS_GENRECON_CFG,
            "projection": NATIVE_SS_GENRECON_PROJECTION,
            "condition_scale_policy": "learned_projection_only",
            "post_cfg_cap": False,
            "direct_slat_dependency": False,
        },
        "args": {},
        "model_trainable_state": {"x": torch.ones(())},
        "ema_trainable_state": {"x": torch.ones(())},
        "ema": {"target_decay": 0.9995, "updates": 1},
    }


def test_checkpoint_rejects_slat_era_contracts() -> None:
    for forbidden in (
        "delta_rms_ratio_cap",
        "guided_delta_policy",
        "delta_bound_mode",
        "occupancy_weight",
        "raw_delta_excess_weight",
        "condition_scale",
    ):
        checkpoint = valid_checkpoint()
        checkpoint["args"][forbidden] = 0
        try:
            validate_native_ss_genrecon_checkpoint(checkpoint, pretrained="test")
        except ValueError as error:
            assert "forbidden SLAT-era" in str(error)
        else:
            raise AssertionError(f"forbidden checkpoint field was accepted: {forbidden}")


def test_checkpoint_accepts_new_contract() -> None:
    validate_native_ss_genrecon_checkpoint(valid_checkpoint(), pretrained="test")


def test_object_selection_is_uid_sorted_and_sequence_independent() -> None:
    rows = [
        {"uid": "b1", "object_uid": "b"},
        {"uid": "a1", "object_uid": "a"},
        {"uid": "a2", "object_uid": "a"},
        {"uid": "c1", "object_uid": "c"},
    ]
    assert select_object_indices(rows, start=0, end=2) == [1, 0]
    assert select_object_indices(rows, start=2, end=3) == [3]


def test_calibration_gate_requires_all_metrics_and_count_bounds() -> None:
    args = type(
        "Args",
        (),
        {
            "min_iou_gain_mean": 0.0,
            "min_recall_gain_mean": 0.0,
            "min_latent_mse_gain_mean": 0.0,
            "min_count_ratio": 0.85,
            "max_count_ratio": 1.20,
        },
    )()
    candidate = {
        "summary": {
            "iou_gain": {"mean": 0.01},
            "recall_gain": {"mean": 0.02},
            "latent_mse_gain": {"mean": 0.03},
        },
        "count_summary": {"full_stock_count_ratio": {"mean": 1.05}},
    }
    assert all(candidate_checks(candidate, args).values())
    candidate["summary"]["latent_mse_gain"]["mean"] = -0.01
    assert candidate_checks(candidate, args)["latent_mse_gain_mean"] is False


def test_empty_stock_record_is_reported_without_discarding_other_metrics() -> None:
    records = [
        {
            "object_uid": "a",
            "stock_count": 0,
            "full_stock_count_ratio": None,
            "full_minus_stock_count": 3,
            "iou_gain": 0.1,
            "precision_gain": 0.2,
            "recall_gain": 0.3,
            "latent_mse_gain": 0.4,
        },
        {
            "object_uid": "a",
            "stock_count": 2,
            "full_stock_count_ratio": 1.5,
            "full_minus_stock_count": 1,
            "iou_gain": 0.2,
            "precision_gain": 0.3,
            "recall_gain": 0.4,
            "latent_mse_gain": 0.5,
        },
        {
            "object_uid": "b",
            "stock_count": 4,
            "full_stock_count_ratio": 0.5,
            "full_minus_stock_count": -2,
            "iou_gain": -0.1,
            "precision_gain": -0.2,
            "recall_gain": -0.3,
            "latent_mse_gain": -0.4,
        },
    ]
    object_rows, summaries, count_summary = aggregate_records(
        records, bootstrap_samples=10, seed=1
    )
    by_uid = {row["object_uid"]: row for row in object_rows}
    assert by_uid["a"]["full_stock_count_ratio"] == 1.5
    assert by_uid["a"]["full_stock_count_ratio_defined_seed_count"] == 1
    assert by_uid["a"]["stock_empty_seed_count"] == 1
    assert summaries["iou_gain"]["count"] == 2
    assert count_summary["full_stock_count_ratio"]["count"] == 2
    assert count_summary["stock_empty_record_count"] == 1
    assert count_summary["stock_empty_record_rate"] == 1 / 3

    args = type(
        "Args",
        (),
        {
            "min_iou_gain_mean": -1.0,
            "min_recall_gain_mean": -1.0,
            "min_latent_mse_gain_mean": -1.0,
            "min_count_ratio": 0.1,
            "max_count_ratio": 2.0,
        },
    )()
    candidate = {
        "summary": summaries,
        "count_summary": count_summary,
        "stock_empty_record_count": 1,
    }
    assert candidate_checks(candidate, args)["stock_baseline_nonempty"] is False


def test_new_runtime_sources_do_not_import_direct_slat() -> None:
    root = Path(__file__).parent
    for name in (
        "native_ss_genrecon.py",
        "train_native_ss_genrecon.py",
        "evaluate_native_ss_genrecon.py",
    ):
        text = (root / name).read_text(encoding="utf-8").lower()
        assert "from pose_point_depth_mv.direct_slat" not in text
        assert "import pose_point_depth_mv.direct_slat" not in text


def test_calibration_source_reuses_stock_baselines() -> None:
    source = (Path(__file__).parent / "evaluate_native_ss_genrecon.py").read_text(
        encoding="utf-8"
    )
    assert "baseline_cache.get(cache_key)" in source
    assert "cache_key = (int(index), int(seed), float(cfg_strength))" in source
    assert '"latent": stock_latent.detach().cpu()' in source
    assert "baseline_cache=baseline_cache" in source


def test_cache_contract_binds_dino_schema_and_training_config() -> None:
    dataset = type(
        "Dataset",
        (),
        {
            "visual_feature_dim": 3072,
            "feature_metadata": {
                "dino_feature_dim": 1024,
                "patch_count": 1369,
                "patch_start_idx": 5,
            },
            "config_hash": "same",
            "config": {
                "geometric_preprocessing": shared_preprocessing_contract(
                    resolution=518,
                    foreground_margin=1.1,
                    alpha_threshold=0.8,
                )
            },
        },
    )()
    contract = validate_genrecon_cache_contract(
        dataset, training_config_hash="same"
    )
    assert contract["patch_side"] == 37
    try:
        validate_genrecon_cache_contract(dataset, training_config_hash="different")
    except RuntimeError as error:
        assert "differs from training" in str(error)
    else:
        raise AssertionError("evaluation accepted a different lifting cache config")


def test_calibration_objects_must_be_disjoint_from_training() -> None:
    require_disjoint_object_uids(["val_a", "val_b"], ["train_a", "train_b"])
    try:
        require_disjoint_object_uids(["val_a", "shared"], ["shared", "train_b"])
    except RuntimeError as error:
        assert "overlaps training objects" in str(error)
    else:
        raise AssertionError("calibration accepted a training object")


def test_shared_preprocessing_updates_intrinsics_and_retains_foreground() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        rgb = np.zeros((8, 12, 3), dtype=np.uint8)
        rgb[1:7, 0:3] = np.asarray((200, 100, 50), dtype=np.uint8)
        mask = np.zeros((8, 12), dtype=np.uint8)
        mask[1:7, 0:3] = 255
        image_path = root / "image.png"
        mask_path = root / "mask.png"
        Image.fromarray(rgb, mode="RGB").save(image_path)
        Image.fromarray(mask, mode="L").save(mask_path)
        prepared = prepare_shared_object_views(
            [str(image_path)],
            [str(mask_path)],
            resolution=16,
            foreground_margin=1.1,
            alpha_threshold=0.8,
        )
    assert prepared.images[0].size == (16, 16)
    assert prepared.masks.shape == (1, 16, 16)
    assert prepared.foreground_retained_fractions == [1.0]
    assert prepared.crop_boxes[0][0] < 0
    source_k = np.asarray(
        [[[8.0, 0.0, 2.0], [0.0, 8.0, 4.0], [0.0, 0.0, 1.0]]],
        dtype=np.float32,
    )
    transformed = transform_intrinsics(
        source_k, prepared.source_to_feature_affines
    )
    np.testing.assert_allclose(
        transformed,
        prepared.source_to_feature_affines @ source_k,
        rtol=0.0,
        atol=1.0e-6,
    )
    record = prepared.geometry_record()
    assert len(record["contract_hash"]) == 64
    assert len(record["geometry_hash"]) == 64


def test_cache_contract_rejects_legacy_split_preprocessing() -> None:
    dataset = type(
        "Dataset",
        (),
        {
            "visual_feature_dim": 3072,
            "feature_metadata": {
                "dino_feature_dim": 1024,
                "patch_count": 1369,
                "patch_start_idx": 5,
            },
            "config_hash": "legacy",
            "config": {},
        },
    )()
    try:
        validate_genrecon_cache_contract(dataset)
    except ValueError as error:
        assert "cache feature contract mismatch" in str(error)
    else:
        raise AssertionError("Native SS accepted a legacy split-preprocessing cache")


def test_formal_native_ss_api_has_no_external_condition_scale() -> None:
    assert "condition_scale" not in inspect.signature(
        NativeSSGenreconFlow.adapted_prediction
    ).parameters
    assert "condition_scale" not in inspect.signature(
        NativeSSCalibratedCFGFlow.__init__
    ).parameters
    evaluation_source = (
        Path(__file__).parent / "evaluate_native_ss_genrecon.py"
    ).read_text(encoding="utf-8")
    assert "candidate_condition_scales" not in evaluation_source


def test_warmup_and_ema_ramp_are_short_run_safe() -> None:
    assert warmup_factor(1, 10) == 0.1
    assert warmup_factor(10, 10) == 1.0
    assert warmup_factor(20, 10) == 1.0
    assert ema_ramp_decay(0.9995, 1) < ema_ramp_decay(0.9995, 200)
    assert ema_ramp_decay(0.9995, 200) < 0.9995


def test_ema_tracks_trainable_parameters_without_gradients() -> None:
    model = nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight.zero_()
    ema = initialize_ema_state(model)
    with torch.no_grad():
        model.weight.fill_(2.0)
    update_ema_state(ema, model, decay=0.5)
    torch.testing.assert_close(ema["weight"], torch.ones_like(ema["weight"]))
    assert ema["weight"].requires_grad is False
