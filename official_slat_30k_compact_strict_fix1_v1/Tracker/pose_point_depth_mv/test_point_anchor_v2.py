from __future__ import annotations

import unittest

import torch

from pose_point_depth_mv.point_anchor_v2 import (
    ACTIVE_INDEX,
    CONFIDENCE_INDEX,
    DISTANCE_INDEX,
    OCCUPANCY_INDEX,
    POINT_CONTROL_NAMES,
    PointAnchorProbe,
    build_point_evidence,
    deterministic_subsample_points,
    make_constant_evidence,
    make_drop_evidence,
    make_null_point_evidence,
    match_cross_object_points,
    point_volume,
    transform_points,
    validate_points,
)


class PointAnchorV2Tests(unittest.TestCase):
    @staticmethod
    def points() -> tuple[torch.Tensor, torch.Tensor]:
        coords = torch.tensor(
            [[2, 2, 2], [2, 2, 6], [2, 6, 2], [6, 2, 2], [30, 34, 38]],
            dtype=torch.long,
        )
        confidence = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9])
        return coords, confidence

    @staticmethod
    def model_inputs() -> tuple[torch.Tensor, ...]:
        generator = torch.Generator().manual_seed(42)
        x_t = torch.randn((1, 8, 16, 16, 16), generator=generator)
        stock = torch.randn((1, 8, 16, 16, 16), generator=generator)
        t = torch.tensor([500.0])
        coords, confidence = PointAnchorV2Tests.points()
        evidence = build_point_evidence(coords, confidence).unsqueeze(0)
        return x_t, stock, t, evidence

    def test_token_axis_mapping(self) -> None:
        coords = torch.tensor(
            [[2, 2, 2], [2, 2, 6], [2, 6, 2], [6, 2, 2]],
            dtype=torch.long,
        )
        confidence = torch.ones(4)
        occupancy, _, _ = point_volume(coords, confidence)
        expected = {(0, 0, 0), (0, 0, 1), (0, 1, 0), (1, 0, 0)}
        observed = {tuple(row.tolist()) for row in torch.nonzero(occupancy)}
        self.assertEqual(observed, expected)

    def test_cross_object_matches_count_and_confidence_multiset(self) -> None:
        correct_xyz, correct_confidence = self.points()
        candidate_xyz = torch.tensor(
            [[10 + index, 20 + index, 30 + index] for index in range(8)]
        )
        candidate_confidence = torch.linspace(0.05, 0.95, 8)
        xyz, confidence, report = match_cross_object_points(
            correct_xyz,
            correct_confidence,
            candidate_xyz,
            candidate_confidence,
            uid="correct",
            candidate_uid="candidate",
            seed=42,
        )
        self.assertEqual(len(xyz), len(correct_xyz))
        self.assertTrue(
            torch.equal(
                torch.sort(confidence).values,
                torch.sort(correct_confidence).values,
            )
        )
        self.assertEqual(report["confidence_multiset_max_abs_diff"], 0.0)
        self.assertFalse(report["used_coordinate_replacement"])

    def test_point_validation_rejects_silent_quantization_and_clipping(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-integral"):
            validate_points(
                torch.tensor([[1.5, 2.0, 3.0]]),
                torch.tensor([0.5]),
                uid="fractional",
            )
        with self.assertRaisesRegex(ValueError, r"outside \[0,1\]"):
            validate_points(
                torch.tensor([[1, 2, 3]]),
                torch.tensor([1.1]),
                uid="confidence",
            )

    def test_point_subsample_is_deterministic_and_has_no_replacement(self) -> None:
        xyz = torch.arange(30).reshape(10, 3)
        confidence = torch.linspace(0.0, 0.9, 10)
        first = deterministic_subsample_points(
            xyz, confidence, 6, uid="sample", seed=42
        )
        second = deterministic_subsample_points(
            xyz, confidence, 6, uid="sample", seed=42
        )
        self.assertTrue(torch.equal(first[0], second[0]))
        self.assertTrue(torch.equal(first[1], second[1]))
        self.assertEqual(len(torch.unique(first[0], dim=0)), 6)

    def test_controls_share_correct_mask_and_change_point_content(self) -> None:
        coords, confidence = self.points()
        correct = build_point_evidence(coords, confidence)
        active = correct[ACTIVE_INDEX]
        controls = {
            "point_reflect": build_point_evidence(
                transform_points(coords, "point_reflect"),
                confidence,
                reference_active_mask=active,
            ),
            "point_axis_cycle": build_point_evidence(
                transform_points(coords, "point_axis_cycle"),
                confidence,
                reference_active_mask=active,
            ),
            "point_spatial_roll": build_point_evidence(
                transform_points(coords, "point_spatial_roll"),
                confidence,
                reference_active_mask=active,
            ),
            "point_drop": make_drop_evidence(correct),
            "constant_prior": make_constant_evidence(correct),
        }
        for evidence in controls.values():
            self.assertTrue(torch.equal(evidence[ACTIVE_INDEX], active))
            self.assertGreater(float((evidence[:3] - correct[:3]).abs().sum()), 0.0)

    def test_drop_is_active_mask_only(self) -> None:
        coords, confidence = self.points()
        correct = build_point_evidence(coords, confidence)
        dropped = make_drop_evidence(correct)
        self.assertEqual(int(torch.count_nonzero(dropped[OCCUPANCY_INDEX])), 0)
        self.assertEqual(int(torch.count_nonzero(dropped[CONFIDENCE_INDEX])), 0)
        self.assertTrue(torch.equal(dropped[DISTANCE_INDEX], torch.ones((16, 16, 16))))
        self.assertTrue(torch.equal(dropped[ACTIVE_INDEX], correct[ACTIVE_INDEX]))

    def test_zero_init_and_non_anchor_exact_zero(self) -> None:
        probe = PointAnchorProbe(rank=4)
        delta, stats = probe(*self.model_inputs())
        self.assertEqual(int(torch.count_nonzero(delta)), 0)
        self.assertEqual(float(stats["neutral_abs_max"]), 0.0)
        with torch.no_grad():
            probe.output.weight.normal_(mean=0.0, std=0.1)
        delta, stats = probe(*self.model_inputs())
        evidence = self.model_inputs()[-1]
        active = probe.active_mask(evidence).bool().expand_as(delta)
        self.assertGreater(int(torch.count_nonzero(delta[active])), 0)
        self.assertEqual(int(torch.count_nonzero(delta[~active])), 0)
        self.assertEqual(float(stats["neutral_abs_max"]), 0.0)

    def test_physical_off_and_null_are_exact_after_randomization(self) -> None:
        probe = PointAnchorProbe(rank=4)
        with torch.no_grad():
            probe.output.weight.normal_(mean=0.0, std=0.1)
        x_t, stock, t, evidence = self.model_inputs()
        disabled, _ = probe(
            x_t, stock, t, evidence, physical_present=False
        )
        null, _ = probe(
            x_t, stock, t, make_null_point_evidence(evidence)
        )
        self.assertEqual(int(torch.count_nonzero(disabled)), 0)
        self.assertEqual(int(torch.count_nonzero(null)), 0)

    def test_schema_contains_all_controls(self) -> None:
        metadata = PointAnchorProbe(rank=4).metadata()
        self.assertEqual(tuple(metadata["controls"]), POINT_CONTROL_NAMES)
        self.assertFalse(metadata["uses_pose_depth"])
        self.assertFalse(metadata["uses_flow_lora"])


if __name__ == "__main__":
    unittest.main()
