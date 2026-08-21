from __future__ import annotations

import unittest

import torch

from pose_point_depth_mv.c1_direct_occupancy import (
    initialize_nested_models,
    nested_base_initialization_equal,
    occupancy_metrics,
)
from pose_point_depth_mv.c1_matched_budget import (
    MATCHED_CONTROLS,
    deterministic_rank_order,
    histogram_match_to_ranking,
    matched_budget_metrics,
    matched_candidate_weights,
)


def fake_payload() -> dict:
    shape = (16, 16, 16)
    active = torch.zeros(shape, dtype=torch.bool)
    active.reshape(-1)[:20] = True
    reliability = torch.zeros(shape)
    reliability.reshape(-1)[:20] = torch.linspace(0.05, 1.0, 20)
    correct = torch.zeros(shape)
    correct.reshape(-1)[:20] = torch.linspace(-1.0, 2.0, 20)
    controls = {}
    for offset, name in enumerate(
        ("pose_cyclic1", "depth_view_cyclic1", "visual_view_cyclic1")
    ):
        values = torch.zeros(shape)
        values.reshape(-1)[:20] = torch.roll(
            correct.reshape(-1)[:20], shifts=offset + 1
        )
        controls[name] = values
    hard_margin = correct - torch.stack(list(controls.values())).amax(0)
    hard_weight = (
        torch.tanh(hard_margin.clamp_min(0.0) / 0.25) * reliability
    ).masked_fill(~active, 0.0)
    continuous = (
        0.1 * torch.sigmoid(hard_margin / 0.25) * reliability
    ).masked_fill(~active, 0.0)
    return {
        "active_mask": active,
        "gate_mask": active & hard_margin.gt(0.0),
        "hard_admitted_soft_weight": hard_weight,
        "continuous_soft_weight": continuous,
        "correct_score": correct,
        "training_control_margins": {
            name: correct - value for name, value in controls.items()
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


class C1MatchedDirectTests(unittest.TestCase):
    def test_histogram_matching_is_exact_and_target_independent(self) -> None:
        active = torch.tensor([1, 1, 1, 1, 0], dtype=torch.bool)
        reference = torch.tensor([0.0, 0.2, 0.7, 1.0, 0.0])
        ranking = torch.tensor([4.0, 3.0, 2.0, 1.0, 100.0])
        output = histogram_match_to_ranking(
            reference, ranking, active, uid="u", name="control"
        )
        self.assertTrue(
            torch.equal(
                torch.sort(output[active]).values,
                torch.sort(reference[active]).values,
            )
        )
        self.assertEqual(float(output[~active].abs().max()), 0.0)
        self.assertAlmostEqual(float(output.sum()), float(reference.sum()), places=6)
        first = deterministic_rank_order(ranking, active, uid="u", name="same")
        second = deterministic_rank_order(ranking, active, uid="u", name="same")
        self.assertTrue(torch.equal(first, second))

    def test_all_real_controls_receive_the_correct_histogram(self) -> None:
        payload = fake_payload()
        matched, invariants = matched_candidate_weights(
            payload, policy="hard_admitted", uid="sample"
        )
        self.assertEqual(set(matched), {"correct", *MATCHED_CONTROLS})
        for row in invariants["candidates"].values():
            self.assertTrue(row["histogram_equal"])
            self.assertTrue(row["support_equal"])
            self.assertTrue(row["inactive_zero"])
            self.assertLessEqual(row["mass_abs_diff"], 1.0e-6)

    def test_budget_metrics_use_fixed_counts(self) -> None:
        active = torch.ones(100, dtype=torch.bool)
        target = torch.zeros(100, dtype=torch.bool)
        target[:10] = True
        weight = torch.arange(100, 0, -1, dtype=torch.float32)
        semantics = torch.ones(100, dtype=torch.long)
        metrics = matched_budget_metrics(
            weight,
            target,
            active,
            semantics,
            uid="sample",
            name="correct",
            fractions=(0.05, 0.10, 0.20),
        )
        self.assertEqual(metrics["budgets"]["top_05"]["count"], 5)
        self.assertEqual(metrics["budgets"]["top_10"]["count"], 10)
        self.assertEqual(metrics["budgets"]["top_20"]["count"], 20)
        self.assertEqual(metrics["budgets"]["top_10"]["target_rate"], 1.0)

    def test_nested_models_start_from_identical_base(self) -> None:
        models = initialize_nested_models(input_dim=13, hidden_dim=16, seed=42)
        self.assertTrue(nested_base_initialization_equal(models))
        base = torch.randn(8, 13)
        corr = torch.randn(8, 1)
        m1 = models["M1_view_geometry"](base)
        m2 = models["M2_plus_correspondence"](base, corr)
        self.assertTrue(torch.equal(m1, m2))
        loss = m2.square().mean()
        loss.backward()
        self.assertIsNotNone(
            models["M2_plus_correspondence"].correspondence[-1].weight.grad
        )

    def test_occupancy_metrics_are_finite(self) -> None:
        metrics = occupancy_metrics(
            torch.tensor([4.0, 3.0, -2.0, -3.0]),
            torch.tensor([1.0, 1.0, 0.0, 0.0]),
        )
        self.assertLess(metrics["balanced_bce"], 0.1)
        self.assertEqual(metrics["average_precision"], 1.0)
        self.assertEqual(metrics["roc_auc"], 1.0)


if __name__ == "__main__":
    unittest.main()
