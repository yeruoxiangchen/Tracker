from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from pose_aligned_reconstruction.audit_native_3d_condition_g0 import cuda_device_index
from pose_aligned_reconstruction.native_3d_condition import (
    EveryBlockConditionProjector,
    NativeEveryBlockSSFlowModel,
    NativeViewAggregator,
    bounded_native_ss_flow_delta,
    combine_dense_cfg,
    dense_native_coords,
    native_ss_cfg_is_active,
    native_ss_timestep_sequence,
    project_native_features,
)
from reconvggt_ar_adapter_a.train_pointpose_ss_lora import sample_t


def test_cuda_device_index_defaults_to_visible_device_zero() -> None:
    assert cuda_device_index(torch.device("cuda")) == 0
    assert cuda_device_index(torch.device("cuda:3")) == 3


def test_native_training_time_sampler_returns_loggable_float() -> None:
    for schedule in ("uniform", "logit_normal", "high_t_mix"):
        value = sample_t(schedule, torch.device("cpu"))
        assert isinstance(value, float)
        assert 0.0 <= float(value) <= 1.0


def synthetic_sample(*, views: int = 3, channels: int = 3072) -> dict:
    patch_side = 4
    height = width = 64
    visual = torch.arange(
        views * patch_side * patch_side * channels, dtype=torch.float32
    ).reshape(views, patch_side * patch_side, channels)
    visual = (visual.remainder(1009) / 1009.0).to(torch.float16)
    intrinsics = torch.tensor(
        [[48.0, 0.0, 31.5], [0.0, 48.0, 31.5], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    ).repeat(views, 1, 1)
    extrinsics = torch.eye(4, dtype=torch.float32).repeat(views, 1, 1)
    extrinsics[:, 2, 3] = -2.0
    if views > 1:
        extrinsics[:, 0, 3] = torch.linspace(-0.2, 0.2, views)
    return {
        "uid": "synthetic",
        "visual_patch_features": visual,
        "predicted_depth": torch.full((views, height, width), 2.0),
        "depth_confidence": torch.ones((views, height, width)),
        "masks": torch.ones((views, height, width)),
        "intrinsics": intrinsics,
        "extrinsics": extrinsics,
        "depth_calibration": {"enabled": False},
        "grid_transform": "identity",
        "extrinsics_type": "c2w",
        "camera_forward_sign": 1.0,
    }


def permute_views(sample: dict, permutation: torch.Tensor) -> dict:
    result = dict(sample)
    for name in (
        "visual_patch_features",
        "predicted_depth",
        "depth_confidence",
        "masks",
        "intrinsics",
        "extrinsics",
    ):
        result[name] = sample[name].index_select(0, permutation)
    return result


def test_dense_native_coords_are_x_major() -> None:
    coords = dense_native_coords(3)
    assert coords.shape == (27, 4)
    assert coords[:5].tolist() == [
        [0, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 2],
        [0, 0, 1, 0],
        [0, 0, 1, 1],
    ]
    assert coords[-1].tolist() == [0, 2, 2, 2]


def test_arbitrary_sparse_32_projection_preserves_coords() -> None:
    coords = torch.tensor(
        [[0, 0, 0, 0], [0, 7, 13, 21], [0, 31, 31, 31]], dtype=torch.int32
    )
    evidence = project_native_features(
        synthetic_sample(), coords, resolution=32, feature_source="dino"
    )
    assert torch.equal(evidence["coords"], coords)
    assert evidence["projected_visual"].shape == (3, 3, 1024)
    assert evidence["per_view_geometry"].shape == (3, 3, 10)
    assert evidence["base_weight"].shape == (3, 3)
    assert torch.isfinite(evidence["projected_visual"]).all()


def test_view_permutation_invariance() -> None:
    torch.manual_seed(7)
    sample = synthetic_sample()
    coords = torch.tensor(
        [[0, 8, 8, 8], [0, 16, 16, 16], [0, 24, 12, 20]], dtype=torch.int32
    )
    original = project_native_features(sample, coords, resolution=32)
    permutation = torch.tensor([2, 0, 1])
    permuted = project_native_features(
        permute_views(sample, permutation), coords, resolution=32
    )
    aggregator = NativeViewAggregator(visual_channels=1024, hidden_dim=12)
    left, _ = aggregator(
        original["projected_visual"],
        original["per_view_geometry"],
        original["base_weight"],
    )
    right, _ = aggregator(
        permuted["projected_visual"],
        permuted["per_view_geometry"],
        permuted["base_weight"],
    )
    assert torch.allclose(left, right, atol=1.0e-6, rtol=1.0e-6)


def test_pose_corruption_changes_projected_evidence() -> None:
    sample = synthetic_sample()
    coords = torch.tensor(
        [[0, 4, 10, 12], [0, 14, 20, 24], [0, 27, 11, 18]], dtype=torch.int32
    )
    correct = project_native_features(sample, coords, resolution=32, mode="correct")
    corrupt = project_native_features(
        sample, coords, resolution=32, mode="pose_cyclic1"
    )
    difference = (
        correct["projected_visual"] - corrupt["projected_visual"]
    ).abs().max()
    assert float(difference.item()) > 1.0e-5


def test_every_block_projection_is_independent_and_zero_initialized() -> None:
    projector = EveryBlockConditionProjector(
        hidden_dim=7, flow_channels=11, block_count=4
    )
    value = torch.randn(5, 7)
    assert projector.all_outputs_exact_zero()
    outputs = [projector(index, value) for index in range(4)]
    assert all(torch.count_nonzero(output) == 0 for output in outputs)
    assert len({id(module) for module in projector.projections}) == 4


class MockBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.linear = nn.Linear(channels, channels, bias=False)
        nn.init.eye_(self.linear.weight)

    def forward(
        self, h: torch.Tensor, t_emb: torch.Tensor, context: torch.Tensor
    ) -> torch.Tensor:
        return h + 0.05 * self.linear(h)


class MockSSFlow(nn.Module):
    resolution = 16
    in_channels = 8
    out_channels = 8
    patch_size = 1
    model_channels = 12
    share_mod = False
    dtype = torch.float32

    def __init__(self) -> None:
        super().__init__()
        self.input_layer = nn.Linear(8, self.model_channels)
        self.register_buffer("pos_emb", torch.zeros(16**3, self.model_channels))
        self.t_embedder = nn.Linear(1, self.model_channels)
        self.blocks = nn.ModuleList([MockBlock(self.model_channels) for _ in range(3)])
        self.out_layer = nn.Linear(self.model_channels, 8)

    def _run(
        self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor
    ) -> torch.Tensor:
        h = x.flatten(2).transpose(1, 2)
        h = self.input_layer(h) + self.pos_emb[None]
        t_emb = self.t_embedder(t.reshape(-1, 1))
        for block in self.blocks:
            h = block(h, t_emb, cond)
        h = torch.nn.functional.layer_norm(h, h.shape[-1:])
        h = self.out_layer(h)
        return h.transpose(1, 2).reshape(1, 8, 16, 16, 16)

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor
    ) -> torch.Tensor:
        return self._run(x, t, cond)


def test_ss_zero_init_is_exact_stock_and_all_blocks_receive_gradients() -> None:
    torch.manual_seed(13)
    flow = MockSSFlow()
    model = NativeEveryBlockSSFlowModel(
        flow, visual_channels=1024, hidden_dim=8, feature_source="dino"
    )
    for parameter in flow.parameters():
        parameter.requires_grad_(False)
    sample = synthetic_sample(views=2)
    x = torch.randn(1, 8, 16, 16, 16)
    t = torch.tensor([500.0])
    condition = torch.zeros(1, 1, 1)
    stock = model.stock_prediction(x, t, condition)
    prediction, stats = model.conditioned_prediction(
        x, t, condition, sample, stock_velocity=stock
    )
    assert torch.equal(prediction, stock)
    assert int(stats["conditioned_block_count"].item()) == 3
    prediction.square().mean().backward()
    for projection in model.block_condition.projections:
        assert projection.weight.grad is not None
        assert torch.isfinite(projection.weight.grad).all()
        assert float(projection.weight.grad.abs().sum().item()) > 0.0


def test_ss_post_cfg_zero_init_is_exact_stock_with_live_gradients() -> None:
    torch.manual_seed(17)
    flow = MockSSFlow()
    model = NativeEveryBlockSSFlowModel(
        flow, visual_channels=1024, hidden_dim=8, feature_source="dino"
    )
    for parameter in flow.parameters():
        parameter.requires_grad_(False)
    sample = synthetic_sample(views=2)
    x = torch.randn(1, 8, 16, 16, 16)
    t = torch.tensor([800.0])
    positive = torch.zeros(1, 1, 1)
    negative = torch.ones(1, 1, 1)
    stock_positive = model.stock_prediction(x, t, positive)
    stock_negative = model.stock_prediction(x, t, negative)
    prediction, stock, stats = model.post_cfg_conditioned_prediction(
        x,
        t,
        positive,
        negative,
        sample,
        stock_positive_velocity=stock_positive,
        stock_negative_velocity=stock_negative,
        cfg_strength=5.0,
        cfg_active=True,
        support_active=True,
    )
    assert torch.equal(prediction, stock)
    assert int(stats["conditioned_block_count"].item()) == 3
    prediction.square().mean().backward()
    for projection in model.block_condition.projections:
        assert projection.weight.grad is not None
        assert float(projection.weight.grad.abs().sum().item()) > 0.0


def test_ss_post_cfg_composes_residual_once_and_inactive_is_exact_stock() -> None:
    torch.manual_seed(19)
    model = NativeEveryBlockSSFlowModel(
        MockSSFlow(), visual_channels=1024, hidden_dim=8, feature_source="dino"
    )
    for projection in model.block_condition.projections:
        nn.init.constant_(projection.bias, 0.01)
    sample = synthetic_sample(views=2)
    x = torch.randn(1, 8, 16, 16, 16)
    t = torch.tensor([800.0])
    positive = torch.zeros(1, 1, 1)
    negative = torch.ones(1, 1, 1)
    stock_positive = model.stock_prediction(x, t, positive)
    stock_negative = model.stock_prediction(x, t, negative)
    raw_positive, _ = model.conditioned_prediction(
        x,
        t,
        positive,
        sample,
        stock_velocity=stock_positive,
    )
    expected_stock = combine_dense_cfg(
        stock_positive, stock_negative, cfg_strength=5.0
    )
    expected_raw = combine_dense_cfg(raw_positive, stock_negative, cfg_strength=5.0)
    expected, _ = bounded_native_ss_flow_delta(
        expected_stock, expected_raw, delta_rms_ratio_cap=0.10
    )
    prediction, stock, stats = model.post_cfg_conditioned_prediction(
        x,
        t,
        positive,
        negative,
        sample,
        stock_positive_velocity=stock_positive,
        stock_negative_velocity=stock_negative,
        cfg_strength=5.0,
        cfg_active=True,
        support_active=True,
        delta_rms_ratio_cap=0.10,
    )
    assert torch.equal(stock, expected_stock)
    assert torch.allclose(prediction, expected, atol=1.0e-6, rtol=1.0e-6)
    positive_delta_rms = (raw_positive.float() - stock_positive.float()).square().mean().sqrt()
    assert torch.allclose(
        stats["raw_flow_delta_rms"],
        5.0 * positive_delta_rms,
        atol=1.0e-6,
        rtol=1.0e-5,
    )
    inactive, inactive_stock, inactive_stats = model.post_cfg_conditioned_prediction(
        x,
        torch.tensor([200.0]),
        positive,
        negative,
        sample,
        stock_positive_velocity=stock_positive,
        cfg_strength=5.0,
        cfg_active=False,
        support_active=False,
    )
    assert torch.equal(inactive_stock, stock_positive)
    assert torch.equal(inactive, stock_positive)
    assert float(inactive_stats["effective_flow_delta_rms"].item()) == 0.0


def test_native_ss_smooth_bound_and_deployment_schedule() -> None:
    stock = torch.ones(2, 3, 4, 4)
    raw = stock + torch.tensor([1.0, 4.0]).reshape(2, 1, 1, 1)
    effective, stats = bounded_native_ss_flow_delta(
        stock, raw, delta_rms_ratio_cap=0.10
    )
    assert effective.shape == stock.shape
    assert bool((stats["effective_flow_delta_ratio_per_batch"] <= 0.10).all())
    assert bool((stats["delta_clip_scale_per_batch"] < 1.0).all())
    schedule = native_ss_timestep_sequence(steps=25, rescale_t=3.0)
    assert len(schedule) == 26
    assert schedule[0] == 1.0 and schedule[-1] == 0.0
    assert all(left > right for left, right in zip(schedule, schedule[1:]))
    active = [value for value in schedule[:-1] if native_ss_cfg_is_active(value, (0.5, 1.0))]
    assert active and active[0] == 1.0
    assert all(0.5 <= value <= 1.0 for value in active)


def test_slat_metadata_rejects_ss16_image_condition_dependency() -> None:
    source = Path(__file__).with_name("native_3d_condition.py").read_text(
        encoding="utf-8"
    )
    assert '"uses_ss16_as_image_condition": False' in source
    assert "resolution=32" in source
