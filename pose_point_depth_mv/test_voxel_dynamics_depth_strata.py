from __future__ import annotations

import unittest

import torch

from pose_point_depth_mv.audit_voxel_dynamics_depth_strata import (
    flip_audit,
    stratified_audit,
)


def fake_map(uid: str, margin: torch.Tensor, *, step: int) -> dict[str, object]:
    active = torch.ones_like(margin, dtype=torch.bool)
    controls = ["pose_cyclic1", "depth_view_cyclic1", "visual_view_cyclic1"]
    return {
        "uid": uid,
        "checkpoint_step": step,
        "views": 4,
        "active_mask": active,
        "hard_margin": margin,
        "hardest_control_index": torch.tensor([0, 1, 1, 2]),
        "training_controls": controls,
        "training_control_margins": {
            "pose_cyclic1": margin + 0.4,
            "depth_view_cyclic1": margin,
            "visual_view_cyclic1": margin + 0.2,
        },
        "depth_calibration_median_abs_residual": 0.1,
        "depth_calibration_p90_abs_residual": 0.3,
        "audit_maps": {
            "raw_reliability": torch.tensor([0.1, 0.3, 0.7, 0.9]),
            "depth_confidence": torch.tensor([0.2, 0.4, 0.6, 0.8]),
            "depth_semantic_label": torch.tensor([1, 2, 3, 4], dtype=torch.int8),
        },
    }


class VoxelDynamicsDepthStrataTests(unittest.TestCase):
    def test_flip_audit_tracks_new_and_lost_positive_voxels(self) -> None:
        before = {"a": fake_map("a", torch.tensor([-1.0, 1.0, 2.0, -2.0]), step=100)}
        after = {"a": fake_map("a", torch.tensor([0.5, -0.5, 3.0, -3.0]), step=200)}
        report = flip_audit(before, after)["aggregate"]
        self.assertAlmostEqual(
            float(report["negative_to_positive"]["fraction_active"]["mean"]), 0.25
        )
        self.assertAlmostEqual(
            float(report["positive_to_negative"]["fraction_active"]["mean"]), 0.25
        )
        self.assertAlmostEqual(
            float(report["original_positive_margin_delta_mean"]["mean"]), -0.25
        )
        self.assertAlmostEqual(
            float(report["original_negative_margin_delta_mean"]["mean"]), 0.25
        )

    def test_depth_strata_are_object_balanced_and_complete(self) -> None:
        maps = {
            f"u{index}": fake_map(
                f"u{index}",
                torch.tensor([-0.2, 0.1, 0.4, 0.8]) + 0.1 * index,
                step=100,
            )
            for index in range(4)
        }
        for index, payload in enumerate(maps.values()):
            payload["views"] = (2, 4, 8, 8)[index]
            payload["depth_calibration_median_abs_residual"] = 0.1 * (index + 1)
            payload["depth_calibration_p90_abs_residual"] = 0.2 * (index + 1)
        report = stratified_audit(maps)
        reliability = report["dimensions"]["reliability_quartile"]
        self.assertEqual(set(reliability), {"Q1", "Q2", "Q3", "Q4"})
        self.assertEqual(
            set(report["dimensions"]["depth_semantic"]),
            {"surface", "free_space", "occluded", "boundary"},
        )
        self.assertEqual(report["dimensions"]["view_count"]["8"]["object_count"], 2)


if __name__ == "__main__":
    unittest.main()
