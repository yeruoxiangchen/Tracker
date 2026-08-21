from __future__ import annotations

import sys
import types
import unittest

import numpy as np
import torch

# The real Tracker checkout provides trellis_point_prior_mv.  A tiny import stub
# keeps these pure gate tests runnable in source-only packaging environments.
try:
    import trellis_point_prior_mv.common  # type: ignore  # noqa: F401
except ModuleNotFoundError:
    package = types.ModuleType("trellis_point_prior_mv")
    common = types.ModuleType("trellis_point_prior_mv.common")
    common.apply_grid_transform = lambda points, _name: points
    common.coords_to_points = lambda coords, _side: coords[..., -3:]
    package.common = common
    sys.modules["trellis_point_prior_mv"] = package
    sys.modules["trellis_point_prior_mv.common"] = common

from pose_point_depth_mv.geometry import (
    EVIDENCE_NAMES,
    build_evidence,
    deterministic_point_split,
    match_applied_delta_rms,
    match_gated_delta_rms,
    mean_match_gate,
)
from ar_ss_flow.pose_lifting import collect_vggt_depth_matches


class PosePointDepthMVTests(unittest.TestCase):
    def test_evidence_names_are_unique(self) -> None:
        self.assertEqual(len(EVIDENCE_NAMES), len(set(EVIDENCE_NAMES)))
        for required in (
            "surface_support",
            "free_space_support",
            "occluded_support",
            "positive_label",
            "negative_label",
            "neutral_label",
        ):
            self.assertIn(required, EVIDENCE_NAMES)

    def test_mean_match_gate(self) -> None:
        candidate = torch.linspace(0.0, 1.0, 4096).reshape(1, 16, 16, 16)
        reference = torch.full_like(candidate, 0.37)
        matched = mean_match_gate(candidate, reference)
        self.assertTrue(bool(((matched >= 0.0) & (matched <= 1.0)).all()))
        self.assertAlmostEqual(float(matched.mean()), 0.37, places=5)

    def test_mean_match_zero_and_one(self) -> None:
        candidate = torch.rand(1, 16, 16, 16)
        zero = mean_match_gate(candidate, torch.zeros_like(candidate))
        one = mean_match_gate(candidate, torch.ones_like(candidate))
        self.assertEqual(int(torch.count_nonzero(zero)), 0)
        self.assertTrue(bool((one == 1.0).all()))

    def test_deterministic_crossfit_split_is_disjoint_and_complete(self) -> None:
        coords = torch.arange(36, dtype=torch.int64).reshape(12, 3)
        fit_a, heldout_a = deterministic_point_split(
            coords,
            uid="object-sequence-0",
            fit_fraction=0.5,
            split_seed=42,
            minimum_points_per_split=4,
        )
        fit_b, heldout_b = deterministic_point_split(
            coords,
            uid="object-sequence-0",
            fit_fraction=0.5,
            split_seed=42,
            minimum_points_per_split=4,
        )
        self.assertTrue(torch.equal(fit_a, fit_b))
        self.assertTrue(torch.equal(heldout_a, heldout_b))
        self.assertEqual(set(fit_a.tolist()).intersection(heldout_a.tolist()), set())
        self.assertEqual(
            set(fit_a.tolist()).union(heldout_a.tolist()), set(range(len(coords)))
        )

    def test_applied_delta_rms_matching(self) -> None:
        candidate = torch.linspace(0.1, 1.0, 128).reshape(1, 2, 4, 4, 4)
        reference = torch.full_like(candidate, 0.25)
        matched, scale = match_applied_delta_rms(candidate, reference)
        self.assertGreater(scale, 0.0)
        self.assertAlmostEqual(
            float(matched.float().square().mean().sqrt()),
            float(reference.float().square().mean().sqrt()),
            places=6,
        )

    def test_gated_energy_match_allows_large_scalar_without_amplification(self) -> None:
        raw = torch.ones((1, 1, 4, 4, 4), dtype=torch.float32)
        reference = raw * 0.60
        candidate_gate = torch.full((1, 1, 4, 4, 4), 0.04)
        delta, effective_gate, scale, attainable, error = match_gated_delta_rms(
            raw,
            candidate_gate,
            reference,
            maximum_scale=1000.0,
        )
        self.assertTrue(attainable)
        self.assertGreater(scale, 10.0)
        self.assertLessEqual(float(effective_gate.max()), 1.0)
        self.assertAlmostEqual(float(effective_gate.mean()), 0.60, places=5)
        self.assertAlmostEqual(float(delta.square().mean().sqrt()), 0.60, places=5)
        self.assertLess(error, 1.0e-6)

    def test_gated_energy_match_reports_unattainable_support(self) -> None:
        raw = torch.zeros((1, 1, 2, 2, 2), dtype=torch.float32)
        raw[..., 0, 0, 0] = 1.0
        reference = raw.clone()
        candidate_gate = torch.zeros_like(raw)
        candidate_gate[..., 1, 1, 1] = 1.0
        delta, effective_gate, _, attainable, error = match_gated_delta_rms(
            raw,
            candidate_gate,
            reference,
            maximum_scale=1000.0,
        )
        self.assertFalse(attainable)
        self.assertEqual(int(torch.count_nonzero(delta)), 0)
        self.assertLessEqual(float(effective_gate.max()), 1.0)
        self.assertGreater(error, 0.99)

    def test_projection_without_observation_evidence_has_zero_gate(self) -> None:
        views = 2
        side = 16
        intrinsics = torch.tensor(
            [[[12.0, 0.0, 7.5], [0.0, 12.0, 7.5], [0.0, 0.0, 1.0]]]
        ).repeat(views, 1, 1)
        extrinsics = torch.eye(4).repeat(views, 1, 1)
        extrinsics[:, 2, 3] = -2.0
        sample = {
            "uid": "synthetic-null-evidence",
            "predicted_depth": torch.full((views, side, side), 2.0),
            "depth_confidence": torch.ones((views, side, side)),
            "masks": torch.zeros((views, side, side)),
            "intrinsics": intrinsics,
            "extrinsics": extrinsics,
            "prior_coords": torch.empty((0, 3), dtype=torch.int64),
            "prior_confidence": torch.empty((0,), dtype=torch.float32),
            "grid_transform": "identity",
            "extrinsics_type": "c2w",
            "camera_forward_sign": 1.0,
            "depth_calibration": {},
        }
        calibration = {
            "enabled": True,
            "quality_passed": True,
            "quality_weight": 1.0,
            "scale": 1.0,
            "shift": 0.0,
            "p90_abs_residual": 0.02,
            "match_count": 8,
            "heldout": {"match_count": 8},
        }
        evidence = build_evidence(
            sample,
            device=torch.device("cpu"),
            calibration_override=calibration,
            volume_side=4,
            minimum_surface_views=1,
            minimum_free_views=1,
            recalibrate_each_hypothesis=False,
        )
        self.assertEqual(int(torch.count_nonzero(evidence.local_gate)), 0)
        self.assertEqual(evidence.stats["mask_zero_gate_mean"], 0.0)
        self.assertEqual(evidence.stats["neutral_gate_mean"], 0.0)
        self.assertEqual(evidence.stats["negative_gate_mean"], 0.0)

    def test_depth_matches_respect_mask_and_sparse_zbuffer(self) -> None:
        depth = np.full((1, 16, 16), 2.0, dtype=np.float32)
        confidence = np.ones_like(depth)
        intrinsics = np.asarray(
            [[[12.0, 0.0, 7.5], [0.0, 12.0, 7.5], [0.0, 0.0, 1.0]]],
            dtype=np.float32,
        )
        extrinsics = np.eye(4, dtype=np.float32)[None]
        extrinsics[:, 2, 3] = -2.0
        coords = np.asarray(((32, 32, 16), (32, 32, 48)), dtype=np.int64)
        kwargs = {
            "predicted_depth": depth,
            "depth_confidence": confidence,
            "prior_coords": coords,
            "prior_confidence": np.ones(2, dtype=np.float32),
            "intrinsics": intrinsics,
            "extrinsics": extrinsics,
            "grid_transform": "identity",
            "extrinsics_type": "c2w",
            "camera_forward_sign": 1.0,
            "zbuffer_cell_size": 16,
        }
        visible = collect_vggt_depth_matches(
            **kwargs, masks=np.ones_like(depth)
        )
        masked = collect_vggt_depth_matches(
            **kwargs, masks=np.zeros_like(depth)
        )
        self.assertEqual(visible["match_count"], 1)
        self.assertEqual(masked["match_count"], 0)


if __name__ == "__main__":
    unittest.main()
