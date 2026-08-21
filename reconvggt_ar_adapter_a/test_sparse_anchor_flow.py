from __future__ import annotations

import unittest

import torch
from torch import nn

from reconvggt_ar_adapter_a.pointpose_ss_condition import PHYSICAL_FEATURE_NAMES
from reconvggt_ar_adapter_a.sparse_anchor_flow import (
    FEATURE_INDEX,
    SparseAnchorSSFlowModel,
    SparseAnchorVelocityAdapter,
    build_sparse_anchor_masks,
    make_mask_only_physical_grid,
    make_point_only_physical_grid,
    shift_sparse_prior,
)
from reconvggt_ar_adapter_a.stock_preserving_pointpose_bridge import (
    make_null_physical_grid,
)


def physical_grid() -> torch.Tensor:
    grid = torch.zeros(1, len(PHYSICAL_FEATURE_NAMES), 16, 16, 16)
    axis = (torch.arange(16, dtype=torch.float32) + 0.5) / 8.0 - 1.0
    xx, yy, zz = torch.meshgrid(axis, axis, axis, indexing="ij")
    grid[0, FEATURE_INDEX["x"]] = xx
    grid[0, FEATURE_INDEX["y"]] = yy
    grid[0, FEATURE_INDEX["z"]] = zz
    grid[0, FEATURE_INDEX["prior_occupancy"], 5, 6, 7] = 1.0
    grid[0, FEATURE_INDEX["prior_confidence"], 5, 6, 7] = 0.9
    grid[0, FEATURE_INDEX["prior_log_count"], 5, 6, 7] = 1.0
    grid[0, FEATURE_INDEX["mask_support_fraction"], 5, 6, 7] = 1.0
    grid[0, FEATURE_INDEX["visible_fraction"]] = 1.0
    grid[0, FEATURE_INDEX["outside_visible_ratio"], :2] = 1.0
    return grid


class FakeFlow(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_parameter("weight", nn.Parameter(torch.tensor(0.25)))

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        del t, cond
        return x * self.weight


class SparseAnchorFlowTests(unittest.TestCase):
    def test_point_mask_and_null_views(self) -> None:
        grid = physical_grid()
        null = make_null_physical_grid(grid)
        mask_only = make_mask_only_physical_grid(grid)
        point_only = make_point_only_physical_grid(grid)
        self.assertEqual(int(torch.count_nonzero(null[:, :11]).item()), 0)
        self.assertEqual(float(mask_only[0, FEATURE_INDEX["prior_occupancy"]].sum()), 0.0)
        self.assertGreater(float(mask_only[0, FEATURE_INDEX["visible_fraction"]].sum()), 0.0)
        self.assertGreater(float(point_only[0, FEATURE_INDEX["prior_occupancy"]].sum()), 0.0)
        self.assertEqual(float(point_only[0, FEATURE_INDEX["visible_fraction"]].sum()), 0.0)
        self.assertTrue(torch.equal(null[:, 11:], grid[:, 11:]))

    def test_nonwrapping_prior_shift(self) -> None:
        grid = physical_grid()
        shifted = shift_sparse_prior(grid, (2, -1, 1))
        self.assertEqual(
            float(shifted[0, FEATURE_INDEX["prior_occupancy"], 7, 5, 8]), 1.0
        )
        self.assertEqual(float(shifted[0, FEATURE_INDEX["prior_occupancy"]].sum()), 1.0)
        self.assertEqual(
            float(shifted[0, FEATURE_INDEX["prior_distance"], 7, 5, 8]), 0.0
        )
        self.assertTrue(torch.equal(
            shifted[:, FEATURE_INDEX["outside_visible_ratio"]],
            grid[:, FEATURE_INDEX["outside_visible_ratio"]],
        ))

    def test_masks_are_nonempty_and_exclusive(self) -> None:
        grid = physical_grid()
        coords = torch.tensor(
            [[20, 24, 28], [21, 24, 28], [40, 40, 40]], dtype=torch.long
        )
        masks = build_sparse_anchor_masks(
            grid,
            coords,
            prior_confidence_min=0.25,
            anchor_radius_16=1,
            outside_ratio_min=0.9,
        )
        self.assertGreater(int(masks["positive16"].sum()), 0)
        self.assertGreater(int(masks["negative16"].sum()), 0)
        self.assertGreater(int(masks["positive64"].sum()), 0)
        self.assertGreater(int(masks["negative64"].sum()), 0)
        self.assertFalse(bool((masks["positive64"] & masks["negative64"]).any()))
        self.assertTrue(bool((masks["positive16"] | masks["negative16"] | masks["neutral16"]).all()))

    def test_zero_init_and_stock_routes(self) -> None:
        grid = physical_grid()
        flow = FakeFlow()
        adapter = SparseAnchorVelocityAdapter(hidden_dim=16)
        model = SparseAnchorSSFlowModel(flow, adapter)
        x = torch.randn(1, 8, 16, 16, 16)
        t = torch.tensor([500.0])
        cond = torch.randn(1, 4, 8)
        adapted, stock, _ = model(x, t, cond, grid)
        self.assertTrue(torch.equal(adapted, stock))
        disabled, _, _ = model(x, t, cond, grid, physical_present=False)
        self.assertTrue(torch.equal(disabled, stock))
        null, _, _ = model(x, t, cond, make_null_physical_grid(grid))
        self.assertTrue(torch.equal(null, stock))
        self.assertFalse(flow.weight.requires_grad)

    def test_gradient_starts_at_output_then_reaches_encoders(self) -> None:
        torch.manual_seed(7)
        grid = physical_grid()
        adapter = SparseAnchorVelocityAdapter(hidden_dim=16)
        optimizer = torch.optim.AdamW(adapter.parameters(), lr=1.0e-3)
        x = torch.randn(1, 8, 16, 16, 16)
        stock = torch.randn_like(x)
        t = torch.tensor([400.0])

        delta, _ = adapter(x, stock, t, grid)
        delta.sum().backward()
        output_grad = sum(
            float(parameter.grad.abs().sum())
            for parameter in adapter.output.parameters()
            if parameter.grad is not None
        )
        encoder_grad = sum(
            float(parameter.grad.abs().sum())
            for parameter in adapter.physical_encoder.parameters()
            if parameter.grad is not None
        )
        self.assertGreater(output_grad, 0.0)
        self.assertEqual(encoder_grad, 0.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        delta, _ = adapter(x, stock, t, grid)
        delta.square().mean().backward()
        encoder_grad = sum(
            float(parameter.grad.abs().sum())
            for parameter in adapter.physical_encoder.parameters()
            if parameter.grad is not None
        )
        self.assertGreater(encoder_grad, 0.0)


if __name__ == "__main__":
    unittest.main()

