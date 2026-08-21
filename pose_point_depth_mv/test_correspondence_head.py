from __future__ import annotations

import unittest

import torch

from pose_point_depth_mv.correspondence_head import (
    CONTINUOUS_SOFT_WEIGHT_VERSION,
    HARD_ADMITTED_SOFT_WEIGHT_VERSION,
    ViewCorrespondenceHead,
    continuous_voxel_gate_weight,
    correct_voxel_reliability_weight,
    load_correspondence_head_state,
    parse_control_names,
    soft_voxel_gate_confidence,
    trainable_state_dict,
    voxel_control_ranking_loss,
    voxel_self_calibration,
)
from pose_point_depth_mv.audit_neighborhood_core_shell import partition_support
from pose_point_depth_mv.prepare_c1_neighborhood_gates import valid_map
from pose_point_depth_mv.view_identity_lifting import (
    VIEW_IDENTITY_EVIDENCE_VERSION,
    VIEW_IDENTITY_GEOMETRY_NAMES,
    apply_symmetric_spatial_tolerance,
    view_identity_schema_hash,
)


def fake_evidence(
    *, views: int = 4, voxels: int = 37, visual_channels: int = 12
) -> dict[str, object]:
    generator = torch.Generator().manual_seed(19)
    return {
        "format": VIEW_IDENTITY_EVIDENCE_VERSION,
        "schema_hash": view_identity_schema_hash(),
        "mode": "correct",
        "views": views,
        "volume_side": 16,
        "sampled_visual": torch.randn(
            views, voxels, visual_channels, generator=generator
        ),
        "geometry": torch.randn(
            views,
            voxels,
            len(VIEW_IDENTITY_GEOMETRY_NAMES),
            generator=generator,
        ),
        "view_weight": torch.rand(views, voxels, generator=generator) + 0.2,
        "canonical_xyz": torch.zeros(voxels, 3),
        "depth_enabled": True,
    }


class CorrespondenceHeadTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(23)
        self.head = ViewCorrespondenceHead(
            visual_channels=12,
            hidden_dim=16,
            pair_hidden_dim=20,
            min_views=2,
        )
        self.evidence = fake_evidence()

    def test_forward_shapes_and_finite(self) -> None:
        result = self.head(self.evidence)
        self.assertEqual(result["voxel_score"].shape, (37,))
        self.assertEqual(result["active_mask"].shape, (37,))
        self.assertTrue(bool(torch.isfinite(result["sample_score"])))
        self.assertGreater(float(result["support_ratio"]), 0.0)

    def test_common_view_permutation_is_invariant(self) -> None:
        baseline = self.head(self.evidence)["sample_score"]
        permutation = torch.tensor([2, 0, 3, 1])
        changed = dict(self.evidence)
        for key in ("sampled_visual", "geometry", "view_weight"):
            changed[key] = self.evidence[key].index_select(0, permutation)
        candidate = self.head(changed)["sample_score"]
        self.assertTrue(torch.allclose(baseline, candidate, atol=1.0e-6, rtol=1.0e-6))

    def test_fixed_weight_override_controls_support(self) -> None:
        fixed = self.evidence["view_weight"].clone()
        changed = dict(self.evidence)
        changed["view_weight"] = torch.zeros_like(fixed)
        baseline = self.head(self.evidence, view_weight_override=fixed)
        candidate = self.head(changed, view_weight_override=fixed)
        self.assertTrue(torch.equal(baseline["active_mask"], candidate["active_mask"]))
        self.assertTrue(torch.equal(baseline["sample_score"], candidate["sample_score"]))

    def test_gaussian3_tolerance_is_symmetric_and_expands_fixed_support(self) -> None:
        evidence = fake_evidence(views=2, voxels=4**3, visual_channels=12)
        fixed = torch.zeros(2, 4**3)
        fixed[:, 1 * 16 + 1 * 4 + 1] = 1.0
        evidence["view_weight"] = fixed
        evidence["signed_normalized_depth_residual"] = torch.randn(2, 4**3)
        changed = dict(evidence)
        changed["sampled_visual"] = evidence["sampled_visual"] + 3.0
        correct, correct_weight = apply_symmetric_spatial_tolerance(
            evidence, fixed_correct_weight=fixed, mode="gaussian3"
        )
        control, control_weight = apply_symmetric_spatial_tolerance(
            changed, fixed_correct_weight=fixed, mode="gaussian3"
        )
        self.assertTrue(torch.equal(correct_weight, control_weight))
        self.assertGreater(int(correct_weight.gt(0).sum()), int(fixed.gt(0).sum()))
        self.assertEqual(
            correct["signed_normalized_depth_residual"].shape,
            fixed.shape,
        )
        self.assertFalse(torch.equal(correct["sampled_visual"], control["sampled_visual"]))
        correct_result = self.head(correct, view_weight_override=correct_weight)
        control_result = self.head(control, view_weight_override=control_weight)
        self.assertTrue(
            torch.equal(correct_result["active_mask"], control_result["active_mask"])
        )

    def test_exact_spatial_tolerance_is_identity(self) -> None:
        result, weight = apply_symmetric_spatial_tolerance(
            self.evidence,
            fixed_correct_weight=self.evidence["view_weight"],
            mode="exact",
        )
        self.assertIs(result, self.evidence)
        self.assertTrue(torch.equal(weight, self.evidence["view_weight"]))

    def test_core_shell_partition_is_disjoint_and_complete(self) -> None:
        exact = torch.tensor([True, True, False, False])
        neighborhood = torch.tensor([True, True, True, False])
        parts = partition_support(exact, neighborhood)
        self.assertTrue(
            torch.equal(parts["core"], torch.tensor([True, True, False, False]))
        )
        self.assertTrue(
            torch.equal(parts["shell"], torch.tensor([False, False, True, False]))
        )
        self.assertFalse(bool(parts["lost"].any().item()))
        self.assertFalse(bool((parts["core"] & parts["shell"]).any().item()))

    def test_soft_voxel_gate_is_bounded_and_inactive_zero(self) -> None:
        confidence = soft_voxel_gate_confidence(
            torch.tensor([1.0, -1.0, 2.0, 0.0]),
            torch.tensor([1.0, 1.0, 0.25, 1.0]),
            torch.tensor([True, True, False, True]),
            temperature=0.5,
        )
        self.assertGreater(float(confidence[0]), 0.0)
        self.assertEqual(float(confidence[1]), 0.0)
        self.assertEqual(float(confidence[2]), 0.0)
        self.assertEqual(float(confidence[3]), 0.0)
        self.assertTrue(bool(((confidence >= 0.0) & (confidence <= 1.0)).all()))

    def test_continuous_weight_keeps_negative_active_margin_at_low_amplitude(self) -> None:
        weight = continuous_voxel_gate_weight(
            torch.tensor([1.0, -1.0, 2.0]),
            torch.tensor([1.0, 1.0, 1.0]),
            torch.tensor([True, True, False]),
            temperature=0.5,
            max_scale=0.1,
        )
        self.assertGreater(float(weight[0]), float(weight[1]))
        self.assertGreater(float(weight[1]), 0.0)
        self.assertEqual(float(weight[2]), 0.0)
        self.assertLessEqual(float(weight.max()), 0.1)

    def test_c1_map_validation_rejects_hard_admitted_weight_outside_gate(self) -> None:
        active = torch.zeros(16, 16, 16, dtype=torch.bool)
        active[2, 2, 2] = True
        active[3, 3, 3] = True
        hard = torch.zeros(16, 16, 16)
        hard[2, 2, 2] = 1.0
        hard[3, 3, 3] = -1.0
        gate = torch.zeros_like(active)
        gate[2, 2, 2] = True
        admitted = torch.zeros_like(hard)
        admitted[2, 2, 2] = 0.75
        continuous = torch.zeros_like(hard)
        continuous[2, 2, 2] = 0.08
        continuous[3, 3, 3] = 0.02
        payload = {
            "active_mask": active,
            "gate_mask": gate,
            "hard_margin": hard,
            "hard_admitted_soft_weight": admitted,
            "continuous_soft_weight": continuous,
            "hard_admitted_soft_weight_protocol": {
                "version": HARD_ADMITTED_SOFT_WEIGHT_VERSION,
            },
            "continuous_soft_weight_protocol": {
                "version": CONTINUOUS_SOFT_WEIGHT_VERSION,
                "max_scale": 0.1,
            },
        }
        passed, checks = valid_map(payload, threshold=0.0)
        self.assertTrue(passed, checks)
        self.assertTrue(checks["continuous_positive_outside_hard_gate"])
        payload["hard_admitted_soft_weight"] = admitted.clone()
        payload["hard_admitted_soft_weight"][0, 0, 0] = 0.2
        passed, checks = valid_map(payload, threshold=0.0)
        self.assertFalse(passed)
        self.assertFalse(checks["inactive_hard_admitted_zero"])

    def test_all_paths_receive_gradients(self) -> None:
        result = self.head(self.evidence)
        result["sample_score"].square().backward()
        for module in (
            self.head.visual_encoder,
            self.head.geometry_encoder,
            self.head.joint_encoder,
            self.head.pair_encoder,
            self.head.pair_score,
        ):
            total = sum(
                float(parameter.grad.abs().sum())
                for parameter in module.parameters()
                if parameter.grad is not None
            )
            self.assertGreater(total, 0.0)

    def test_state_roundtrip_is_strict(self) -> None:
        state = trainable_state_dict(self.head)
        clone = ViewCorrespondenceHead(
            visual_channels=12,
            hidden_dim=16,
            pair_hidden_dim=20,
            min_views=2,
        )
        load_correspondence_head_state(clone, state)
        self.assertTrue(
            torch.equal(
                self.head(self.evidence)["sample_score"],
                clone(self.evidence)["sample_score"],
            )
        )

    def test_control_parser_rejects_duplicates(self) -> None:
        self.assertEqual(parse_control_names("a,b"), ("a", "b"))
        with self.assertRaises(ValueError):
            parse_control_names("a,a")

    def test_voxel_selfcal_is_local_and_uses_hardest_control(self) -> None:
        active = torch.tensor([True, True, False, True])
        correct = {
            "voxel_score": torch.tensor([3.0, 2.0, 100.0, 1.0]),
            "active_mask": active,
        }
        controls = {
            "first": {
                "voxel_score": torch.tensor([1.0, 3.0, -100.0, 0.0]),
                "active_mask": active,
            },
            "second": {
                "voxel_score": torch.tensor([2.0, 1.0, -200.0, 1.5]),
                "active_mask": active,
            },
        }
        result = voxel_self_calibration(correct, controls)
        self.assertTrue(
            torch.equal(result["raw_hard_margin"], torch.tensor([1.0, -1.0, 200.0, -0.5]))
        )
        self.assertTrue(
            torch.equal(result["hard_margin"], torch.tensor([1.0, -1.0, 0.0, -0.5]))
        )
        self.assertTrue(
            torch.equal(result["hard_control_index"], torch.tensor([1, 0, -1, 1]))
        )
        self.assertTrue(
            torch.equal(result["gate_mask"], torch.tensor([True, False, False, False]))
        )
        self.assertAlmostEqual(float(result["gate_fraction_of_active"]), 1.0 / 3.0)

    def test_voxel_selfcal_rejects_support_changes(self) -> None:
        baseline = self.head(self.evidence)
        changed = dict(baseline)
        changed["active_mask"] = baseline["active_mask"].clone()
        changed["active_mask"][0] = ~changed["active_mask"][0]
        with self.assertRaises(ValueError):
            voxel_self_calibration(baseline, {"changed": changed})

    def test_voxel_control_ranking_matches_exact_hard_margin(self) -> None:
        correct = torch.tensor([1.0, 1.0, 10.0], requires_grad=True)
        controls = torch.tensor(
            (
                (0.9, 0.4, -10.0),
                (0.2, 0.8, -20.0),
                (0.0, 0.7, -30.0),
            ),
            requires_grad=True,
        )
        active = torch.tensor([True, True, False])
        weight = torch.tensor([1.0, 3.0, 100.0])
        result = voxel_control_ranking_loss(
            correct,
            controls,
            active,
            margin=0.25,
            temperature=0.1,
            hard_weight=0.5,
            voxel_weight=weight,
        )
        self.assertTrue(
            torch.allclose(
                result["exact_hard_margin"], torch.tensor([0.1, 0.2]), atol=1.0e-6
            )
        )
        self.assertAlmostEqual(float(result["hard_margin_mean"]), 0.175, places=6)
        self.assertEqual(float(result["hard_positive_ratio"]), 1.0)
        self.assertTrue(bool(torch.isfinite(result["loss"])))

    def test_voxel_control_ranking_trains_all_controls_only_on_active(self) -> None:
        correct = torch.tensor([1.0, 1.0, 10.0], requires_grad=True)
        controls = torch.tensor(
            (
                (0.9, 0.4, -10.0),
                (0.2, 0.8, -20.0),
                (0.0, 0.7, -30.0),
            ),
            requires_grad=True,
        )
        result = voxel_control_ranking_loss(
            correct,
            controls,
            torch.tensor([True, True, False]),
            margin=0.25,
            temperature=0.1,
            hard_weight=0.5,
        )
        result["loss"].backward()
        self.assertTrue(bool((controls.grad[:, :2].abs().sum(dim=1) > 0).all()))
        self.assertEqual(float(controls.grad[:, 2].abs().max()), 0.0)
        self.assertEqual(float(correct.grad[2].abs()), 0.0)

        with self.assertRaises(ValueError):
            voxel_control_ranking_loss(
                correct.detach(),
                controls.detach(),
                torch.tensor([True, True, False]),
                margin=0.25,
                temperature=0.0,
                hard_weight=0.5,
            )

    def test_correct_reliability_is_detached_shared_weight(self) -> None:
        evidence = fake_evidence(views=4, voxels=3)
        geometry = torch.zeros_like(evidence["geometry"])
        names = list(VIEW_IDENTITY_GEOMETRY_NAMES)
        for name in (
            "valid",
            "mask_weight",
            "depth_confidence_weight",
            "depth_consistency_weight",
        ):
            geometry[..., names.index(name)] = 1.0
        geometry[:, 1, names.index("depth_confidence_weight")] = 0.16
        geometry[:, 1, names.index("depth_consistency_weight")] = 0.25
        evidence["geometry"] = geometry.requires_grad_(True)
        evidence["view_weight"] = torch.tensor(
            (
                (0.9, 0.2, 0.9),
                (0.9, 0.2, 0.0),
                (0.9, 0.0, 0.0),
                (0.9, 0.0, 0.0),
            ),
            requires_grad=True,
        )
        active = torch.tensor([True, True, False])
        result = correct_voxel_reliability_weight(
            evidence,
            active,
            min_views=2,
            floor=0.1,
            power=1.0,
        )
        self.assertFalse(result["weight"].requires_grad)
        self.assertEqual(float(result["weight"][2]), 0.0)
        self.assertGreater(float(result["weight"][0]), float(result["weight"][1]))
        self.assertGreaterEqual(float(result["weight"][1]), 0.1)
        self.assertLessEqual(float(result["weight"].max()), 1.0)
        self.assertGreater(float(result["effective_fraction"]), 0.0)
        self.assertLessEqual(float(result["effective_fraction"]), 1.0)

    def test_correct_reliability_rejects_invalid_protocol(self) -> None:
        active = torch.ones(37, dtype=torch.bool)
        with self.assertRaises(ValueError):
            correct_voxel_reliability_weight(
                self.evidence,
                active,
                min_views=2,
                floor=1.0,
                power=1.0,
            )
        with self.assertRaises(ValueError):
            correct_voxel_reliability_weight(
                self.evidence,
                active,
                min_views=2,
                floor=0.1,
                power=0.0,
            )


if __name__ == "__main__":
    unittest.main()
