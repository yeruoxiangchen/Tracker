from __future__ import annotations

import unittest
from types import SimpleNamespace

from ar_ss_flow.build_pose_lifting_cache import select_object_balanced_samples
from pose_point_depth_mv.eval_direct_flow import (
    object_balanced_rows,
    paired_rollout_delta_values,
)
from pose_point_depth_mv.train_direct_flow import (
    ObjectBalancedDistributedSampler,
    resolve_training_profile,
)


class DirectFlowStatisticsTests(unittest.TestCase):
    def test_cache_selection_uses_unique_objects(self) -> None:
        rows = [
            {"uid": "A0", "object_uid": "A"},
            {"uid": "A1", "object_uid": "A"},
            {"uid": "B0", "object_uid": "B"},
            {"uid": "C0", "object_uid": "C"},
        ]
        selected = select_object_balanced_samples(
            rows,
            max_objects=3,
            sequences_per_object=1,
            seed=42,
        )
        self.assertEqual(len(selected), 3)
        self.assertEqual(len({row["object_uid"] for row in selected}), 3)

    def test_performance_profile_disables_corruption_losses(self) -> None:
        args = SimpleNamespace(
            training_profile="performance",
            corruption_modes=None,
            wrong_stock_weight=None,
            correct_gain_weight=None,
            rank_weight=None,
        )
        modes = resolve_training_profile(args)
        self.assertEqual(modes, ())
        self.assertEqual(args.wrong_stock_weight, 0.0)
        self.assertEqual(args.correct_gain_weight, 0.0)
        self.assertEqual(args.rank_weight, 0.0)

    def test_rollout_pairs_by_uid_before_object_average(self) -> None:
        rows = [
            {"uid": "A0", "object_uid": "A", "seed": 1, "branch": "stock", "iou": 0.1},
            {"uid": "A1", "object_uid": "A", "seed": 1, "branch": "stock", "iou": 0.9},
            {"uid": "B0", "object_uid": "B", "seed": 1, "branch": "stock", "iou": 0.4},
            {"uid": "A0", "object_uid": "A", "seed": 1, "branch": "correct", "iou": 0.2},
            {"uid": "A1", "object_uid": "A", "seed": 1, "branch": "correct", "iou": 0.7},
            {"uid": "B0", "object_uid": "B", "seed": 1, "branch": "correct", "iou": 0.5},
        ]
        object_values, paired_values = paired_rollout_delta_values(
            rows,
            branch="correct",
            metric="iou",
        )
        self.assertEqual(len(paired_values), 3)
        self.assertAlmostEqual(paired_values[0], 0.1)
        self.assertAlmostEqual(paired_values[1], -0.2)
        self.assertAlmostEqual(object_values[0], -0.05)
        self.assertAlmostEqual(object_values[1], 0.1)

    def test_teacher_rows_are_object_balanced(self) -> None:
        records = [
            {"object_uid": "A", "branch": "correct", "gain_vs_stock": 1.0, "correct_advantage": 0.0},
            {"object_uid": "A", "branch": "correct", "gain_vs_stock": 0.8, "correct_advantage": 0.0},
            {"object_uid": "B", "branch": "correct", "gain_vs_stock": -0.5, "correct_advantage": 0.0},
        ]
        rows = object_balanced_rows(records)
        self.assertEqual(len(rows), 2)
        by_object = {row["object_uid"]: row for row in rows}
        self.assertAlmostEqual(by_object["A"]["gain_vs_stock"], 0.9)
        self.assertAlmostEqual(by_object["B"]["gain_vs_stock"], -0.5)

    def test_object_sampler_draws_one_sequence_per_object(self) -> None:
        rows = [
            {"uid": "A0", "object_uid": "A"},
            {"uid": "A1", "object_uid": "A"},
            {"uid": "B0", "object_uid": "B"},
            {"uid": "C0", "object_uid": "C"},
        ]
        sampler = ObjectBalancedDistributedSampler(
            rows,
            num_replicas=1,
            rank=0,
            seed=42,
        )
        indices = list(iter(sampler))
        self.assertEqual(len(indices), 3)
        objects = [rows[index]["object_uid"] for index in indices]
        self.assertEqual(set(objects), {"A", "B", "C"})


if __name__ == "__main__":
    unittest.main()
