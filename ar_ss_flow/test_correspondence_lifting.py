from __future__ import annotations

import sys
from pathlib import Path
import types
import unittest

import numpy as np
import torch

# Allow this unit test to run outside the full Tracker checkout.
if "trellis_point_prior_mv.common" not in sys.modules:
    package = types.ModuleType("trellis_point_prior_mv")
    common = types.ModuleType("trellis_point_prior_mv.common")
    common.apply_grid_transform = lambda points, _: np.asarray(points)
    common.coords_to_points = lambda coords, side: (
        (np.asarray(coords)[..., -3:].astype(np.float32) + 0.5) / float(side) - 0.5
    )
    sys.modules.setdefault("trellis_point_prior_mv", package)
    sys.modules["trellis_point_prior_mv.common"] = common

# Support both `python -m ar_ss_flow.<module>` and direct script execution.
TRACKER_ROOT = Path(__file__).resolve().parents[1]
if str(TRACKER_ROOT) not in sys.path:
    sys.path.insert(0, str(TRACKER_ROOT))

from ar_ss_flow.correspondence_gated_flow import (  # noqa: E402
    CorrespondenceGatedVelocityAdapter,
)
from ar_ss_flow.correspondence_lifting import (  # noqa: E402
    CORRESPONDENCE_METADATA_NAMES,
    LocalVoxelCorrespondence,
    pair_feature_dim,
    pose_variant_extrinsics,
)


class CorrespondenceLiftingTest(unittest.TestCase):
    def synthetic_evidence(self) -> dict:
        torch.manual_seed(7)
        views = 4
        side = 2
        voxels = side**3
        patch_side = 3
        channels = 8
        return {
            "views": views,
            "patch_side": patch_side,
            "channels": channels,
            "volume_side": side,
            "visual_patch_features": torch.randn(
                views, patch_side * patch_side, channels
            ),
            "sampled_visual": torch.randn(views, voxels, channels),
            "patch_grid": torch.zeros(views, voxels, 2),
            "base_weight": torch.ones(views, voxels),
            "per_view_geometry": torch.randn(views, voxels, 10),
        }

    def test_pose_variants(self) -> None:
        extrinsics = torch.eye(4).repeat(4, 1, 1)
        extrinsics[:, 0, 3] = torch.arange(4)
        cyclic = pose_variant_extrinsics(extrinsics, "pose_cyclic1")
        reverse = pose_variant_extrinsics(extrinsics, "pose_reverse")
        self.assertEqual(cyclic[:, 0, 3].tolist(), [3.0, 0.0, 1.0, 2.0])
        self.assertEqual(reverse[:, 0, 3].tolist(), [3.0, 2.0, 1.0, 0.0])

        source_only = pose_variant_extrinsics(
            extrinsics, "pose_cyclic1", heldout_index=1
        )
        self.assertTrue(torch.equal(source_only[1], extrinsics[1]))
        self.assertFalse(torch.equal(source_only[0], extrinsics[0]))

    def test_fixed_target_evidence(self) -> None:
        correct = self.synthetic_evidence()
        wrong = dict(correct)
        wrong["patch_grid"] = correct["patch_grid"].clone()
        wrong["patch_grid"][0] = 0.75
        model = LocalVoxelCorrespondence(
            visual_channels=8, embedding_dim=16, pairwise_dim=8
        )
        correct_maps = model.encode_patch_maps(correct["visual_patch_features"])
        wrong_maps = model.encode_patch_maps(wrong["visual_patch_features"])
        correct_out = model.evaluate_heldout(
            correct,
            0,
            neighborhood_radius=0,
            min_source_views=2,
            encoded_patch_maps=correct_maps,
            target_evidence=correct,
            target_encoded_patch_maps=correct_maps,
        )
        wrong_out = model.evaluate_heldout(
            wrong,
            0,
            neighborhood_radius=0,
            min_source_views=2,
            encoded_patch_maps=wrong_maps,
            target_evidence=correct,
            target_encoded_patch_maps=correct_maps,
        )
        self.assertTrue(torch.equal(correct_out.target, wrong_out.target))

    def test_heldout_output_shapes(self) -> None:
        evidence = self.synthetic_evidence()
        model = LocalVoxelCorrespondence(
            visual_channels=8, embedding_dim=16, pairwise_dim=8
        )
        output = model.evaluate_heldout(
            evidence,
            0,
            neighborhood_radius=1,
            min_source_views=2,
        )
        self.assertEqual(tuple(output.error.shape), (8,))
        self.assertEqual(tuple(output.reconstruction.shape), (8, 16))
        self.assertTrue(bool(output.valid_mask.all().item()))
        self.assertTrue(bool(torch.isfinite(output.error).all().item()))


    def test_pairwise_before_aggregation_fields(self) -> None:
        evidence = self.synthetic_evidence()
        model = LocalVoxelCorrespondence(
            visual_channels=8, embedding_dim=16, pairwise_dim=8
        )
        output = model.evaluate_heldout(
            evidence,
            1,
            neighborhood_radius=0,
            min_source_views=2,
        )
        self.assertEqual(
            tuple(output.per_view_pairwise_confidence.shape), (4, 8)
        )
        self.assertEqual(tuple(output.pairwise_peer_count.shape), (4, 8))
        self.assertEqual(tuple(output.final_source_weight.shape), (4, 8))
        self.assertEqual(float(output.final_source_weight[1].abs().max().item()), 0.0)
        self.assertEqual(float(output.pairwise_peer_count[1].abs().max().item()), 0.0)
        self.assertTrue(
            bool(torch.isfinite(output.pairwise_confidence).all().item())
        )

    def test_aggregate_volume_shapes(self) -> None:
        evidence = self.synthetic_evidence()
        model = LocalVoxelCorrespondence(
            visual_channels=8, embedding_dim=16, pairwise_dim=8
        )
        output = model.aggregate_volume(
            evidence,
            neighborhood_radius=1,
            min_source_views=2,
            include_pair_features=True,
        )
        self.assertEqual(tuple(output["visual_volume"].shape), (8, 2, 2, 2))
        self.assertEqual(tuple(output["confidence"].shape), (1, 2, 2, 2))
        self.assertEqual(tuple(output["disagreement"].shape), (1, 2, 2, 2))
        self.assertEqual(
            tuple(output["pair_features"].shape),
            (6, pair_feature_dim(8), 2, 2, 2),
        )
        self.assertEqual(tuple(output["pair_valid"].shape), (6, 1, 2, 2, 2))
        self.assertEqual(tuple(output["pair_indices"].shape), (6, 2))
        self.assertTrue(bool(torch.isfinite(output["visual_volume"]).all().item()))
        self.assertTrue(bool(torch.isfinite(output["pair_features"]).all().item()))

    def test_gated_adapter_zero_init_and_bypass(self) -> None:
        adapter = CorrespondenceGatedVelocityAdapter(
            visual_channels=8,
            hidden_dim=16,
        )
        x_t = torch.randn(1, 8, 16, 16, 16)
        stock = torch.randn_like(x_t)
        visual = torch.randn(1, 8, 16, 16, 16)
        metadata = torch.zeros(
            1, len(CORRESPONDENCE_METADATA_NAMES), 16, 16, 16
        )
        metadata[:, 0] = 1.0
        metadata[:, -3] = 1.0
        t = torch.tensor([500.0])
        delta, _ = adapter(x_t, stock, t, visual, metadata)
        disabled, _ = adapter(
            x_t, stock, t, visual, metadata, physical_present=False
        )
        self.assertEqual(float(delta.abs().max().item()), 0.0)
        self.assertEqual(float(disabled.abs().max().item()), 0.0)


if __name__ == "__main__":
    unittest.main()
