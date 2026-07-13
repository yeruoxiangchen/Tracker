#!/usr/bin/env python3
from __future__ import annotations

import unittest

import torch

from reconvggt_ar_adapter_a.pointpose_ss_condition import (
    PointPoseConditionNet,
    load_partial_state,
    trainable_state_dict,
)


class PointPoseConditionNetTest(unittest.TestCase):
    def make_model(self) -> PointPoseConditionNet:
        return PointPoseConditionNet(cond_dim=32, hidden_dim=16, num_heads=4)

    def test_zero_init_is_exact_and_prefix_shapes_are_stable(self) -> None:
        torch.manual_seed(7)
        model = self.make_model().eval()
        cond = torch.randn(2, 11, 32)
        grid = torch.randn(2, 14, 16, 16, 16)
        fused, stats = model(cond, grid, scale=1.0)
        self.assertTrue(torch.equal(fused, cond))
        self.assertEqual(stats["physical_token_shape"], (2, 512, 32))
        self.assertEqual(float(stats["delta_abs_max"]), 0.0)

    def test_batch_and_scale_validation(self) -> None:
        model = self.make_model()
        cond = torch.randn(2, 5, 32)
        with self.assertRaisesRegex(ValueError, "batch mismatch"):
            model(cond, torch.randn(1, 14, 16, 16, 16))
        with self.assertRaisesRegex(ValueError, "cannot broadcast"):
            model(cond, torch.randn(2, 14, 16, 16, 16), scale=torch.ones(3))
        fused, _ = model(cond, torch.randn(2, 14, 16, 16, 16), scale=torch.ones(2))
        self.assertEqual(tuple(fused.shape), tuple(cond.shape))

    def test_gradients_start_at_output_and_reach_encoder_after_update(self) -> None:
        torch.manual_seed(11)
        model = self.make_model().train()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        cond = torch.randn(1, 7, 32)
        grid = torch.randn(1, 14, 16, 16, 16)
        fused, _ = model(cond, grid)
        fused.square().mean().backward()
        self.assertGreater(float(model.output_proj.weight.grad.abs().sum()), 0.0)
        first_grad = model.grid_encoder[0].weight.grad
        self.assertTrue(first_grad is None or float(first_grad.abs().sum()) == 0.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        fused, _ = model(cond, grid)
        fused.square().mean().backward()
        self.assertGreater(float(model.grid_encoder[0].weight.grad.abs().sum()), 0.0)

    def test_strict_resume_and_allowed_extension(self) -> None:
        model = self.make_model()
        state = trainable_state_dict(model)
        result = load_partial_state(model, state, require_all_trainable=True)
        self.assertFalse(result["missing_trainable"])
        incomplete = dict(state)
        incomplete.pop("output_proj.bias")
        with self.assertRaisesRegex(RuntimeError, "missing trainable"):
            load_partial_state(model, incomplete, require_all_trainable=True)
        result = load_partial_state(
            model,
            incomplete,
            require_all_trainable=True,
            allowed_missing_prefixes=("output_proj.",),
        )
        self.assertEqual(result["allowed_missing_trainable"], ["output_proj.bias"])


if __name__ == "__main__":
    unittest.main()
