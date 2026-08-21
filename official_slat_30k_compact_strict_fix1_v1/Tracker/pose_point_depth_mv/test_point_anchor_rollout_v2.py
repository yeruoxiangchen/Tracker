from __future__ import annotations

import unittest

import numpy as np
import torch
from torch import nn

from pose_point_depth_mv.eval_point_anchor_rollout_v2 import (
    anchor_cell_hit_rate,
    guided_stock_velocity,
    overlap_metrics,
    rollout_point_branches,
    timestep_pairs,
)
from pose_point_depth_mv.point_anchor_v2 import (
    ACTIVE_INDEX,
    POINT_CONTROL_NAMES,
    PointAnchorProbe,
)


class FakeSampler:
    @staticmethod
    def _pred_to_xstart(x_t, t, prediction):
        return x_t - float(t) * prediction

    @staticmethod
    def _xstart_to_pred(x_t, t, x0):
        return (x_t - x0) / float(t)


class ConditionFlow(nn.Module):
    def forward(self, x_t, t, condition):
        value = condition[:, 0, 0].reshape(-1, 1, 1, 1, 1)
        return torch.zeros_like(x_t) + value


class PointAnchorRolloutTests(unittest.TestCase):
    def test_timestep_pairs_endpoints(self) -> None:
        pairs = timestep_pairs(5, 3.0)
        self.assertEqual(len(pairs), 5)
        self.assertEqual(pairs[0][0], 1.0)
        self.assertEqual(pairs[-1][1], 0.0)
        self.assertTrue(all(left > right for left, right in pairs))

    def test_cfg_interval_matches_expected_strength(self) -> None:
        x_t = torch.zeros(1, 1, 2, 2, 2)
        condition = torch.ones(1, 1, 1)
        negative = torch.zeros_like(condition)
        _, _, inside = guided_stock_velocity(
            FakeSampler(),
            ConditionFlow(),
            x_t,
            0.75,
            condition,
            negative,
            cfg_strength=5.0,
            cfg_interval=(0.5, 1.0),
            guidance_rescale=0.0,
        )
        _, _, outside = guided_stock_velocity(
            FakeSampler(),
            ConditionFlow(),
            x_t,
            0.25,
            condition,
            negative,
            cfg_strength=5.0,
            cfg_interval=(0.5, 1.0),
            guidance_rescale=0.0,
        )
        self.assertTrue(torch.equal(inside, torch.full_like(x_t, 5.0)))
        self.assertTrue(torch.equal(outside, torch.ones_like(x_t)))

    def test_overlap_and_anchor_cell_metrics(self) -> None:
        prediction = np.zeros((64, 64, 64), dtype=np.bool_)
        target = np.zeros_like(prediction)
        mask = np.zeros_like(prediction)
        prediction[1, 1, 1] = True
        prediction[20, 20, 20] = True
        target[1, 1, 1] = True
        target[2, 2, 2] = True
        mask[:4, :4, :4] = True
        metrics = overlap_metrics(prediction, target, mask)
        self.assertEqual(metrics["intersection"], 1)
        self.assertAlmostEqual(metrics["iou"], 0.5)
        anchor = np.zeros((16, 16, 16), dtype=np.bool_)
        anchor[0, 0, 0] = True
        self.assertEqual(anchor_cell_hit_rate(prediction, anchor), 1.0)

    def test_direct_rollout_delta_is_zero_outside_fixed_mask(self) -> None:
        rank = 2
        probe = PointAnchorProbe(rank=rank)
        with torch.no_grad():
            probe.output.weight.fill_(0.1)
            probe.evidence_projection.weight.fill_(0.1)
            probe.state_projection.weight.fill_(0.1)
            probe.fusion[0].weight.fill_(0.1)
        branch_count = 1 + len(POINT_CONTROL_NAMES)
        evidence = torch.zeros(branch_count, 8, 16, 16, 16)
        evidence[:, ACTIVE_INDEX, 3, 4, 5] = 1.0
        mask = evidence[:1, ACTIVE_INDEX : ACTIVE_INDEX + 1]
        samples, stats = rollout_point_branches(
            FakeSampler(),
            ConditionFlow(),
            probe,
            torch.zeros(1, 8, 16, 16, 16),
            torch.zeros(1, 1, 1),
            torch.zeros(1, 1, 1),
            evidence,
            mask,
            steps=2,
            cfg_strength=1.0,
            cfg_interval=(0.5, 1.0),
            rescale_t=1.0,
            guidance_rescale=0.0,
            physical_scale=1.0,
        )
        self.assertEqual(tuple(samples.shape), (branch_count, 8, 16, 16, 16))
        self.assertEqual(stats["direct_neutral_delta_abs_max"], 0.0)


if __name__ == "__main__":
    unittest.main()
