from __future__ import annotations

import unittest

import torch

from pose_point_depth_mv.c1_occupancy import (
    MonotoneOccupancyCalibrator,
    average_precision,
    balanced_binary_loss,
    c1_policy_scores,
    permute_within_active,
    policy_metrics,
    roc_auc,
    target_occupancy_masks,
    target_mapping_audit,
)


class C1OccupancyTests(unittest.TestCase):
    def test_target_64_to_16_mapping_and_neighborhood(self) -> None:
        coords = torch.tensor([[0, 0, 0], [3, 3, 3], [4, 8, 12], [63, 63, 63]])
        masks = target_occupancy_masks(coords)
        exact = masks["exact"]
        self.assertEqual(int(exact.sum()), 3)
        self.assertTrue(bool(exact[0, 0, 0]))
        self.assertTrue(bool(exact[1, 2, 3]))
        self.assertTrue(bool(exact[15, 15, 15]))
        self.assertTrue(torch.all(masks["surface_r1"][exact]))
        self.assertGreater(int(masks["surface_r1"].sum()), int(exact.sum()))
        audit = target_mapping_audit(coords)
        self.assertTrue(audit["passed"])
        self.assertEqual(audit["axis_order"], "[x,y,z]")
        self.assertEqual(
            audit["canonical_source"],
            "pose_point_depth_mv.view_identity_lifting._canonical_xyz",
        )
        self.assertEqual(len(audit["exact_target_mask_sha256"]), 64)
        same_count_other_positions = target_mapping_audit(
            torch.tensor([[8, 8, 8], [12, 16, 20], [60, 60, 60]])
        )
        self.assertNotEqual(
            audit["exact_target_mask_sha256"],
            same_count_other_positions["exact_target_mask_sha256"],
        )

    def test_active_permutation_preserves_support_and_histogram(self) -> None:
        active = torch.zeros((16, 16, 16), dtype=torch.bool)
        active.reshape(-1)[:100] = True
        score = torch.zeros_like(active, dtype=torch.float32)
        score.reshape(-1)[:100] = torch.arange(100, dtype=torch.float32)
        first = permute_within_active(score, active, uid="sample", repeat=0)
        second = permute_within_active(score, active, uid="sample", repeat=0)
        other = permute_within_active(score, active, uid="sample", repeat=1)
        self.assertTrue(torch.equal(first, second))
        self.assertFalse(torch.equal(first, other))
        self.assertTrue(torch.equal(torch.sort(first[active]).values, torch.sort(score[active]).values))
        self.assertEqual(float(first[~active].abs().max()), 0.0)

    def test_policy_metrics_detect_perfect_ranking(self) -> None:
        active = torch.ones(8, dtype=torch.bool)
        target = torch.tensor([1, 1, 0, 0, 0, 0, 0, 0], dtype=torch.bool)
        score = torch.tensor([1.0, 0.9, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0])
        metrics = policy_metrics(score, target, active)
        self.assertAlmostEqual(float(metrics["average_precision_active"]), 1.0)
        self.assertAlmostEqual(float(metrics["roc_auc_active"]), 1.0)
        self.assertGreater(float(metrics["weighted_target_rate"]), 0.8)
        self.assertAlmostEqual(average_precision(score, target), 1.0)
        self.assertAlmostEqual(roc_auc(score, target), 1.0)

    def test_average_precision_is_tie_aware(self) -> None:
        scores = torch.zeros(8)
        first = torch.tensor([1, 1, 0, 0, 0, 0, 0, 0], dtype=torch.bool)
        permuted = first[torch.tensor([2, 0, 4, 1, 6, 3, 7, 5])]
        self.assertAlmostEqual(average_precision(scores, first), 0.25)
        self.assertAlmostEqual(average_precision(scores, permuted), 0.25)

    def test_real_corruption_candidate_uses_corrupted_branch_score(self) -> None:
        shape = (16, 16, 16)
        active = torch.zeros(shape, dtype=torch.bool)
        active.reshape(-1)[:2] = True
        reliability = active.float()
        correct = torch.zeros(shape)
        correct.reshape(-1)[:2] = torch.tensor([3.0, 1.0])
        pose = torch.zeros(shape)
        pose.reshape(-1)[:2] = torch.tensor([1.0, 4.0])
        depth = torch.zeros(shape)
        depth.reshape(-1)[:2] = torch.tensor([2.0, 0.0])
        visual = torch.zeros(shape)
        visual.reshape(-1)[:2] = torch.tensor([0.0, 0.0])
        payload = {
            "active_mask": active,
            "gate_mask": active.clone(),
            "hard_admitted_soft_weight": reliability.clone(),
            "continuous_soft_weight": reliability.clone() * 0.1,
            "correct_score": correct,
            "training_control_margins": {
                "pose_cyclic1": correct - pose,
                "depth_view_cyclic1": correct - depth,
                "visual_view_cyclic1": correct - visual,
            },
            "heldout_margins": {},
            "hard_admitted_soft_weight_protocol": {
                "temperature": 0.25,
                "reliability_power": 1.0,
            },
            "continuous_soft_weight_protocol": {
                "temperature": 0.25,
                "reliability_power": 1.0,
                "max_scale": 0.1,
            },
            "audit_maps": {"raw_reliability": reliability},
        }
        policies = c1_policy_scores(payload)
        pose_weight = policies["corruption_hard_admitted_pose_cyclic1"].reshape(-1)
        self.assertEqual(float(pose_weight[0]), 0.0)
        self.assertGreater(float(pose_weight[1]), 0.0)
        self.assertEqual(float(pose_weight[2:].abs().max()), 0.0)

    def test_calibrator_is_monotone_and_finite(self) -> None:
        model = MonotoneOccupancyCalibrator(include_reliability=True)
        reliability = torch.full((3,), 0.5)
        logits = model(torch.tensor([0.0, 0.5, 1.0]), reliability)
        self.assertTrue(bool(torch.isfinite(logits).all()))
        self.assertTrue(bool((logits[1:] > logits[:-1]).all()))
        loss = balanced_binary_loss(logits, torch.tensor([0.0, 1.0, 1.0]))
        loss.backward()
        self.assertTrue(
            all(
                parameter.grad is not None
                and bool(torch.isfinite(parameter.grad).all())
                for parameter in model.parameters()
            )
        )

    def test_nested_calibrator_parameter_sets(self) -> None:
        m0 = MonotoneOccupancyCalibrator(
            include_score=False, include_reliability=False
        )
        m1 = MonotoneOccupancyCalibrator(
            include_score=False, include_reliability=True
        )
        m2 = MonotoneOccupancyCalibrator(
            include_score=True, include_reliability=True
        )
        self.assertEqual(set(dict(m0.named_parameters())), {"bias"})
        self.assertEqual(
            set(dict(m1.named_parameters())), {"bias", "reliability_weight_raw"}
        )
        self.assertEqual(
            set(dict(m2.named_parameters())),
            {"bias", "score_weight_raw", "reliability_weight_raw"},
        )


if __name__ == "__main__":
    unittest.main()
