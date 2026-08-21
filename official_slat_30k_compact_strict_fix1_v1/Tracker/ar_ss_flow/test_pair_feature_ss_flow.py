from __future__ import annotations

import unittest

import torch
from torch import nn

from ar_ss_flow.correspondence_lifting import CORRESPONDENCE_METADATA_NAMES
from ar_ss_flow.pair_feature_ss_flow import (
    LocalPairFeatureVelocityAdapter,
    PairFeatureSSFlowModel,
    PositiveConditionRolloutFlow,
)


class _FakeFlow(nn.Module):
    def forward(
        self, x_t: torch.Tensor, t: torch.Tensor, condition: torch.Tensor
    ) -> torch.Tensor:
        return 0.25 * x_t + condition.mean() * 0.0 + t.reshape(-1, 1, 1, 1, 1) * 0.0


class PairFeatureSSFlowTest(unittest.TestCase):
    def make_inputs(self):
        torch.manual_seed(9)
        x_t = torch.randn(1, 8, 16, 16, 16)
        stock = torch.randn_like(x_t)
        visual = torch.randn(1, 8, 16, 16, 16)
        metadata = torch.zeros(
            1, len(CORRESPONDENCE_METADATA_NAMES), 16, 16, 16
        )
        metadata[:, 0] = 1.0
        pairs = torch.randn(1, 3, 20, 16, 16, 16)
        pair_valid = torch.ones(1, 3, 1, 16, 16, 16, dtype=torch.bool)
        return x_t, stock, visual, metadata, pairs, pair_valid

    def make_adapter(self):
        return LocalPairFeatureVelocityAdapter(
            visual_channels=8,
            pair_feature_dim=20,
            hidden_dim=12,
            residual_t_min=0.5,
            residual_t_ramp=0.1,
        )

    def test_zero_init_null_off_and_time_window(self) -> None:
        adapter = self.make_adapter()
        x_t, stock, visual, metadata, pairs, pair_valid = self.make_inputs()
        high_t = torch.tensor([750.0])
        zero, _ = adapter(
            x_t, stock, high_t, visual, metadata, pairs, pair_valid
        )
        self.assertEqual(float(zero.abs().max().item()), 0.0)

        with torch.no_grad():
            adapter.output.weight.normal_(mean=0.0, std=0.02)
            adapter.output.bias.fill_(0.01)
        enabled, _ = adapter(
            x_t, stock, high_t, visual, metadata, pairs, pair_valid
        )
        self.assertGreater(float(enabled.abs().max().item()), 0.0)
        disabled, _ = adapter(
            x_t,
            stock,
            high_t,
            visual,
            metadata,
            pairs,
            pair_valid,
            physical_present=False,
        )
        null, _ = adapter(
            x_t,
            stock,
            high_t,
            torch.zeros_like(visual),
            torch.zeros_like(metadata),
            torch.zeros_like(pairs),
            torch.zeros_like(pair_valid),
        )
        low_t, _ = adapter(
            x_t,
            stock,
            torch.tensor([400.0]),
            visual,
            metadata,
            pairs,
            pair_valid,
        )
        self.assertEqual(float(disabled.abs().max().item()), 0.0)
        self.assertEqual(float(null.abs().max().item()), 0.0)
        self.assertEqual(float(low_t.abs().max().item()), 0.0)

    def test_pair_order_invariant_but_spatial_alignment_sensitive(self) -> None:
        adapter = self.make_adapter()
        with torch.no_grad():
            adapter.output.weight.normal_(mean=0.0, std=0.02)
            adapter.output.bias.fill_(0.01)
        x_t, stock, visual, metadata, pairs, pair_valid = self.make_inputs()
        t = torch.tensor([750.0])
        baseline, _ = adapter(
            x_t, stock, t, visual, metadata, pairs, pair_valid
        )
        order = torch.tensor([2, 0, 1])
        reordered, _ = adapter(
            x_t,
            stock,
            t,
            visual,
            metadata,
            pairs[:, order],
            pair_valid[:, order],
        )
        shifted, _ = adapter(
            x_t,
            stock,
            t,
            visual,
            metadata,
            torch.roll(pairs, shifts=3, dims=-1),
            torch.roll(pair_valid, shifts=3, dims=-1),
        )
        self.assertTrue(torch.allclose(baseline, reordered, atol=1.0e-6, rtol=1.0e-5))
        self.assertGreater(float((baseline - shifted).abs().max().item()), 1.0e-7)

    def test_rollout_wrapper_only_adapts_positive_condition(self) -> None:
        adapter = self.make_adapter()
        with torch.no_grad():
            adapter.output.bias.fill_(0.1)
        model = PairFeatureSSFlowModel(_FakeFlow(), adapter)
        x_t, _stock, visual, metadata, pairs, pair_valid = self.make_inputs()
        positive = torch.randn(1, 4, 6)
        negative = torch.zeros_like(positive)
        wrapper = PositiveConditionRolloutFlow(
            model,
            positive,
            (visual, metadata, pairs, pair_valid),
            scale=1.0,
        )
        t = torch.tensor([750.0])
        positive_out = wrapper(x_t, t, positive)
        negative_out = wrapper(x_t, t, negative)
        stock = model.stock_flow(x_t, t, positive)
        self.assertGreater(float((positive_out - stock).abs().max().item()), 0.0)
        self.assertTrue(torch.equal(negative_out, model.stock_flow(x_t, t, negative)))
        self.assertEqual(wrapper.positive_calls, 1)
        self.assertEqual(wrapper.negative_calls, 1)


if __name__ == "__main__":
    unittest.main()
