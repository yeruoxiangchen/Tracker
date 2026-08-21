from __future__ import annotations

import unittest

import torch

from pose_point_depth_mv.geometry import EVIDENCE_NAMES
from pose_point_depth_mv.local_target_probe import (
    EVIDENCE_ABLATIONS,
    NEGATIVE_INDEX,
    POSITIVE_INDEX,
    PPDLocalTargetProbe,
    ablate_evidence,
    make_null_evidence,
)


class PPDLocalTargetProbeTests(unittest.TestCase):
    @staticmethod
    def inputs() -> tuple[torch.Tensor, ...]:
        generator = torch.Generator().manual_seed(42)
        x_t = torch.randn((1, 8, 16, 16, 16), generator=generator)
        stock = torch.randn((1, 8, 16, 16, 16), generator=generator)
        t = torch.tensor([750.0])
        evidence = torch.zeros((1, len(EVIDENCE_NAMES), 16, 16, 16))
        evidence[:, POSITIVE_INDEX, 2, 3, 4] = 1.0
        evidence[:, NEGATIVE_INDEX, 10, 11, 12] = 1.0
        evidence[:, EVIDENCE_NAMES.index("surface_support"), 2, 3, 4] = 0.8
        evidence[:, EVIDENCE_NAMES.index("free_space_support"), 10, 11, 12] = 0.9
        axis = torch.linspace(-1.0, 1.0, 16)
        xyz = torch.stack(
            torch.meshgrid(axis, axis, axis, indexing="ij"), dim=0
        )
        for offset, name in enumerate(("x", "y", "z")):
            evidence[:, EVIDENCE_NAMES.index(name)] = xyz[offset]
        return x_t, stock, t, evidence

    def test_metadata_rejects_old_c2_and_global_attention(self) -> None:
        metadata = PPDLocalTargetProbe(rank=4).metadata()
        self.assertFalse(metadata["uses_old_c2_residual"])
        self.assertFalse(metadata["global_attention"])
        self.assertEqual(metadata["local_neighborhood"], 1)
        self.assertEqual(metadata["time_normalization"], "t_div_1000")

    def test_zero_init_is_stock_equivalent(self) -> None:
        probe = PPDLocalTargetProbe(rank=4)
        delta, stats = probe(*self.inputs())
        self.assertEqual(int(torch.count_nonzero(delta)), 0)
        self.assertEqual(float(stats["neutral_abs_max"]), 0.0)

    def test_physical_off_and_null_are_exact_after_randomization(self) -> None:
        probe = PPDLocalTargetProbe(rank=4)
        with torch.no_grad():
            probe.output.weight.normal_(mean=0.0, std=0.1)
        x_t, stock, t, evidence = self.inputs()
        disabled, _ = probe(
            x_t, stock, t, evidence, physical_present=False
        )
        null_evidence = make_null_evidence(evidence)
        for name in ("x", "y", "z"):
            index = EVIDENCE_NAMES.index(name)
            self.assertTrue(torch.equal(null_evidence[:, index], evidence[:, index]))
        null, _ = probe(x_t, stock, t, null_evidence)
        self.assertEqual(int(torch.count_nonzero(disabled)), 0)
        self.assertEqual(int(torch.count_nonzero(null)), 0)

    def test_neutral_voxels_remain_exact_zero_after_randomization(self) -> None:
        probe = PPDLocalTargetProbe(rank=4)
        with torch.no_grad():
            probe.output.weight.normal_(mean=0.0, std=0.1)
        x_t, stock, t, evidence = self.inputs()
        delta, stats = probe(x_t, stock, t, evidence)
        active = probe.active_mask(evidence).bool().expand_as(delta)
        self.assertGreater(int(torch.count_nonzero(delta[active])), 0)
        self.assertEqual(int(torch.count_nonzero(delta[~active])), 0)
        self.assertEqual(float(stats["neutral_abs_max"]), 0.0)

    def test_active_mask_override_controls_output_coverage(self) -> None:
        probe = PPDLocalTargetProbe(rank=4)
        with torch.no_grad():
            probe.output.weight.normal_(mean=0.0, std=0.1)
        x_t, stock, t, evidence = self.inputs()
        override = torch.zeros((1, 1, 16, 16, 16))
        override[:, :, 7, 8, 9] = 1.0
        delta, stats = probe(
            x_t,
            stock,
            t,
            evidence,
            active_mask_override=override,
        )
        applied = override.bool().expand_as(delta)
        self.assertEqual(int(torch.count_nonzero(delta[~applied])), 0)
        self.assertGreater(int(torch.count_nonzero(delta[applied])), 0)
        self.assertAlmostEqual(float(stats["active_ratio"]), 1.0 / 4096.0)

    def test_evidence_ablations_share_labels_xyz_and_mask(self) -> None:
        _, _, _, evidence = self.inputs()
        reference = PPDLocalTargetProbe.active_mask(evidence)
        outputs = {
            mode: ablate_evidence(
                evidence, mode, reference_active_mask=reference
            )
            for mode in EVIDENCE_ABLATIONS
        }
        self.assertTrue(torch.equal(outputs["full"], evidence))
        for mode in EVIDENCE_ABLATIONS[1:]:
            output = outputs[mode]
            self.assertTrue(
                torch.equal(
                    PPDLocalTargetProbe.active_mask(output), reference
                )
            )
            for name in ("x", "y", "z"):
                index = EVIDENCE_NAMES.index(name)
                self.assertTrue(torch.equal(output[:, index], evidence[:, index]))
        point_index = EVIDENCE_NAMES.index("prior_distance")
        depth_index = EVIDENCE_NAMES.index("surface_support")
        self.assertEqual(
            int(torch.count_nonzero(outputs["active_mask_only"][:, depth_index])),
            0,
        )
        self.assertTrue(
            torch.equal(outputs["point_only"][:, point_index], evidence[:, point_index])
        )
        self.assertTrue(
            torch.equal(
                outputs["pose_depth_only"][:, depth_index], evidence[:, depth_index]
            )
        )

    def test_zero_init_first_backward_only_starts_output(self) -> None:
        probe = PPDLocalTargetProbe(rank=4)
        delta, _ = probe(*self.inputs())
        loss = (delta - 0.5).square().mean()
        loss.backward()
        self.assertGreater(float(probe.output.weight.grad.abs().sum()), 0.0)
        self.assertEqual(
            float(probe.evidence_projection.weight.grad.abs().sum()), 0.0
        )

    def test_upstream_parameters_receive_gradients_after_output_starts(self) -> None:
        probe = PPDLocalTargetProbe(rank=4)
        with torch.no_grad():
            probe.output.weight.normal_(mean=0.0, std=0.1)
        delta, _ = probe(*self.inputs())
        delta.square().mean().backward()
        self.assertGreater(
            float(probe.evidence_projection.weight.grad.abs().sum()), 0.0
        )
        self.assertGreater(float(probe.state_projection.weight.grad.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
