from __future__ import annotations

import unittest

import torch

from pose_point_depth_mv.view_identity_lifting import (
    VIEW_IDENTITY_ABLATION_MODES,
    VIEW_IDENTITY_CONTROL_NAMES,
    VIEW_IDENTITY_EVIDENCE_VERSION,
    VIEW_IDENTITY_GEOMETRY_NAMES,
    ViewIdentityPoseDepthProbe,
    make_null_view_identity_evidence,
    spatially_misalign_view_evidence,
    view_identity_schema_hash,
)
from pose_point_depth_mv.eval_view_identity_rollout import rollout_view_branches


class ZeroFlow(torch.nn.Module):
    def forward(
        self, x_t: torch.Tensor, t: torch.Tensor, condition: torch.Tensor
    ) -> torch.Tensor:
        del t, condition
        return torch.zeros_like(x_t)


class UnusedSampler:
    pass


def fake_evidence(
    *, views: int = 3, visual_channels: int = 12
) -> dict[str, torch.Tensor | str | int | bool]:
    generator = torch.Generator().manual_seed(7)
    return {
        "format": VIEW_IDENTITY_EVIDENCE_VERSION,
        "schema_hash": view_identity_schema_hash(),
        "mode": "correct",
        "views": views,
        "volume_side": 16,
        "sampled_visual": torch.randn(
            views, 16**3, visual_channels, generator=generator
        ),
        "geometry": torch.randn(
            views,
            16**3,
            len(VIEW_IDENTITY_GEOMETRY_NAMES),
            generator=generator,
        ),
        "view_weight": torch.ones(views, 16**3),
        "canonical_xyz": torch.randn(16**3, 3, generator=generator),
        "depth_enabled": True,
    }


class ViewIdentityLiftingTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(11)
        self.probe = ViewIdentityPoseDepthProbe(
            visual_channels=12,
            hidden_dim=16,
            pair_dim=8,
            min_views=2,
        )
        self.x_t = torch.randn(1, 8, 16, 16, 16)
        self.stock = torch.randn_like(self.x_t)
        self.t = torch.tensor([500.0])
        self.evidence = fake_evidence()

    def randomize_output(self) -> None:
        with torch.no_grad():
            self.probe.output.weight.normal_(std=0.02)
            self.probe.output.bias.normal_(std=0.02)

    def test_spatial_misalignment_keeps_support_and_breaks_locations(self) -> None:
        changed = spatially_misalign_view_evidence(self.evidence)
        self.assertTrue(torch.equal(changed["view_weight"], self.evidence["view_weight"]))
        self.assertFalse(
            torch.equal(changed["sampled_visual"], self.evidence["sampled_visual"])
        )
        self.assertFalse(torch.equal(changed["geometry"], self.evidence["geometry"]))
        self.assertEqual(changed["mode"], "spatial_view_misaligned")
        self.assertEqual(len(changed["spatial_misalignment_shifts"]), 3)
        self.assertNotEqual(
            changed["spatial_misalignment_shifts"][0],
            changed["spatial_misalignment_shifts"][1],
        )

    def test_zero_init_and_disabled_are_exact_zero(self) -> None:
        enabled, _ = self.probe(self.x_t, self.stock, self.t, self.evidence)
        disabled, _ = self.probe(
            self.x_t,
            self.stock,
            self.t,
            self.evidence,
            physical_present=False,
        )
        self.assertEqual(float(enabled.abs().max()), 0.0)
        self.assertEqual(float(disabled.abs().max()), 0.0)

    def test_null_is_exact_after_training_parameters_change(self) -> None:
        self.randomize_output()
        null = make_null_view_identity_evidence(self.evidence)
        delta, _ = self.probe(self.x_t, self.stock, self.t, null)
        self.assertEqual(float(delta.abs().max()), 0.0)

    def test_fixed_support_keeps_neutral_voxels_exact_zero(self) -> None:
        self.randomize_output()
        support = torch.zeros(16**3)
        support[:31] = 1.0
        delta, stats = self.probe(
            self.x_t,
            self.stock,
            self.t,
            self.evidence,
            support_gate_override=support,
        )
        flat = delta.permute(0, 2, 3, 4, 1).reshape(-1, 8)
        self.assertGreater(float(flat[:31].abs().max()), 0.0)
        self.assertEqual(float(flat[31:].abs().max()), 0.0)
        self.assertEqual(float(stats["neutral_abs_max"]), 0.0)

    def test_no_cross_voxel_information_mixing(self) -> None:
        self.randomize_output()
        baseline, _ = self.probe(self.x_t, self.stock, self.t, self.evidence)
        changed = dict(self.evidence)
        changed_visual = self.evidence["sampled_visual"].clone()
        changed_visual[:, 17, 0] += 3.0
        changed["sampled_visual"] = changed_visual
        candidate, _ = self.probe(self.x_t, self.stock, self.t, changed)
        difference = (candidate - baseline).permute(0, 2, 3, 4, 1).reshape(-1, 8)
        active = difference.abs().amax(dim=1).gt(1.0e-7).nonzero().flatten().tolist()
        self.assertEqual(active, [17])

    def test_prepared_full_matches_direct_forward(self) -> None:
        self.randomize_output()
        direct, _ = self.probe(self.x_t, self.stock, self.t, self.evidence)
        prepared = self.probe.prepare_evidence(self.evidence)
        candidate, _ = self.probe.forward_prepared(
            self.x_t, self.stock, self.t, prepared
        )
        self.assertTrue(torch.equal(direct, candidate))

    def test_diagnostic_ablation_modes_are_finite(self) -> None:
        self.randomize_output()
        support = torch.zeros(16**3)
        support[:47] = 1.0
        for mode in VIEW_IDENTITY_ABLATION_MODES:
            delta, stats = self.probe(
                self.x_t,
                self.stock,
                self.t,
                self.evidence,
                support_gate_override=support,
                ablation_mode=mode,
            )
            self.assertTrue(bool(torch.isfinite(delta).all()))
            self.assertEqual(float(stats["neutral_abs_max"]), 0.0)

    def test_state_only_is_fixed_support_output_bias(self) -> None:
        self.randomize_output()
        support = torch.zeros(16**3)
        support[:19] = 1.0
        delta, _ = self.probe(
            self.x_t,
            self.stock,
            self.t,
            self.evidence,
            support_gate_override=support,
            ablation_mode="state_only",
        )
        flat = delta.permute(0, 2, 3, 4, 1).reshape(-1, 8)
        expected = self.probe.output.bias.detach()
        self.assertTrue(torch.allclose(flat[:19], expected[None].expand(19, -1)))
        self.assertEqual(float(flat[19:].abs().max()), 0.0)

    def test_visual_and_geometry_paths_receive_gradients(self) -> None:
        self.randomize_output()
        delta, _ = self.probe(self.x_t, self.stock, self.t, self.evidence)
        delta.square().mean().backward()
        visual_grad = sum(
            float(parameter.grad.abs().sum())
            for parameter in self.probe.visual_encoder.parameters()
            if parameter.grad is not None
        )
        geometry_grad = sum(
            float(parameter.grad.abs().sum())
            for parameter in self.probe.geometry_encoder.parameters()
            if parameter.grad is not None
        )
        pair_grad = sum(
            float(parameter.grad.abs().sum())
            for parameter in self.probe.pair_projection.parameters()
            if parameter.grad is not None
        )
        self.assertGreater(visual_grad, 0.0)
        self.assertGreater(geometry_grad, 0.0)
        self.assertGreater(pair_grad, 0.0)

    def test_rollout_batches_flow_but_preserves_per_view_evidence(self) -> None:
        evidences = [
            dict(self.evidence)
            for _ in range(1 + len(VIEW_IDENTITY_CONTROL_NAMES))
        ]
        support = self.probe.support_gate(self.evidence)
        condition = torch.zeros(1, 4096, 4)
        result, stats = rollout_view_branches(
            UnusedSampler(),
            ZeroFlow(),
            self.probe,
            self.x_t,
            condition,
            condition,
            evidences,
            self.evidence["view_weight"],
            support,
            steps=2,
            cfg_strength=1.0,
            cfg_interval=(0.0, 1.0),
            rescale_t=1.0,
            guidance_rescale=0.0,
            physical_scale=1.0,
        )
        self.assertEqual(
            result.shape,
            (1 + len(VIEW_IDENTITY_CONTROL_NAMES), 8, 16, 16, 16),
        )
        self.assertTrue(torch.equal(result, self.x_t.expand_as(result)))
        self.assertEqual(stats["direct_neutral_delta_abs_max"], 0.0)


if __name__ == "__main__":
    unittest.main()
