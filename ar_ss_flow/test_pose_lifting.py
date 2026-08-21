from __future__ import annotations

import unittest

import numpy as np
import torch
from torch import nn

from ar_ss_flow.local_pose_lifting_flow import LocalPoseLiftingVelocityAdapter
from ar_ss_flow.object_frame import (
    estimate_point_pca_similarity,
    make_similarity,
    rotation_xyz,
    transform_camera_extrinsics,
    transform_points,
)
from ar_ss_flow.pose_lifting import (
    LIFTING_METADATA_NAMES,
    _fit_depth_model,
    build_lifting_volume,
    build_projection_geometry,
    canonical_voxel_points,
    perturb_extrinsics,
    projection_roundtrip_audit,
)


class ZeroStockFlow(nn.Module):
    def forward(self, x_t: torch.Tensor, t: torch.Tensor, condition: torch.Tensor):
        return torch.zeros_like(x_t)


class PoseLiftingTests(unittest.TestCase):
    def test_robust_depth_scale_and_affine(self) -> None:
        predicted = np.linspace(0.5, 2.0, 64)
        target = 1.7 * predicted + 0.23
        target[0] += 4.0
        confidence = np.ones_like(predicted)
        scale = _fit_depth_model(predicted, target, confidence, affine=False)
        affine = _fit_depth_model(predicted, target, confidence, affine=True)
        self.assertTrue(scale["positive_scale"])
        self.assertAlmostEqual(affine["scale"], 1.7, places=2)
        self.assertAlmostEqual(affine["shift"], 0.23, places=2)
        self.assertLess(
            affine["median_abs_residual"], scale["median_abs_residual"]
        )

    def test_pose_variants_preserve_shape_and_change_geometry(self) -> None:
        extrinsics = torch.eye(4).repeat(3, 1, 1)
        extrinsics[:, 0, 3] = torch.tensor((-0.2, 0.0, 0.2))
        shuffled = perturb_extrinsics(
            extrinsics, mode="pose_shuffle", extrinsics_type="c2w"
        )
        perturbed = perturb_extrinsics(
            extrinsics, mode="pose_perturb", extrinsics_type="c2w"
        )
        self.assertEqual(shuffled.shape, extrinsics.shape)
        self.assertEqual(perturbed.shape, extrinsics.shape)
        self.assertFalse(torch.equal(shuffled, extrinsics))
        self.assertFalse(torch.equal(perturbed, extrinsics))

    def test_projection_geometry_is_16_cubed(self) -> None:
        intrinsic = torch.tensor(
            [[[40.0, 0.0, 31.5], [0.0, 40.0, 31.5], [0.0, 0.0, 1.0]]]
        )
        extrinsic = torch.eye(4)[None]
        extrinsic[:, 2, 3] = -2.0
        geometry = build_projection_geometry(
            intrinsics=intrinsic,
            extrinsics=extrinsic,
            grid_transform="identity",
            extrinsics_type="c2w",
            camera_forward_sign=1.0,
            image_height=64,
            image_width=64,
            patch_grid_side=4,
        )
        self.assertEqual(tuple(geometry["patch_grid"].shape), (1, 4096, 2))
        self.assertEqual(tuple(geometry["camera_depth"].shape), (1, 4096))
        self.assertTrue(bool(geometry["valid"].any().item()))
        roundtrip = projection_roundtrip_audit(
            intrinsics=intrinsic,
            extrinsics=extrinsic,
            grid_transform="identity",
            extrinsics_type="c2w",
            camera_forward_sign=1.0,
            image_height=64,
            image_width=64,
            patch_grid_side=4,
        )
        self.assertGreater(roundtrip["valid_projection_count"], 0)
        self.assertLess(roundtrip["max_error"], 1.0e-5)

    def test_oracle_sim3_preserves_projection_geometry(self) -> None:
        intrinsic = torch.tensor(
            [[[80.0, 0.0, 63.5], [0.0, 80.0, 63.5], [0.0, 0.0, 1.0]]]
        )
        c2w = torch.eye(4)[None]
        c2w[:, 2, 3] = -2.0
        transform = make_similarity(
            scale=1.3,
            rotation=rotation_xyz((12.0, -18.0, 27.0)),
            translation=torch.tensor((0.4, -0.2, 0.7)),
        )
        transformed_c2w = transform_camera_extrinsics(
            c2w, transform, extrinsics_type="c2w"
        )
        baseline = build_projection_geometry(
            intrinsics=intrinsic,
            extrinsics=c2w,
            grid_transform="identity",
            extrinsics_type="c2w",
            camera_forward_sign=1.0,
            image_height=128,
            image_width=128,
            patch_grid_side=8,
        )
        oracle = build_projection_geometry(
            intrinsics=intrinsic,
            extrinsics=transformed_c2w,
            grid_transform="identity",
            extrinsics_type="c2w",
            camera_forward_sign=1.0,
            image_height=128,
            image_width=128,
            patch_grid_side=8,
            object_to_world=transform,
        )
        missing = build_projection_geometry(
            intrinsics=intrinsic,
            extrinsics=transformed_c2w,
            grid_transform="identity",
            extrinsics_type="c2w",
            camera_forward_sign=1.0,
            image_height=128,
            image_width=128,
            patch_grid_side=8,
        )
        self.assertLess(
            float((baseline["patch_grid"] - oracle["patch_grid"]).abs().max()),
            1.0e-5,
        )
        self.assertGreater(
            float((baseline["patch_grid"] - missing["patch_grid"]).abs().mean()),
            1.0e-3,
        )
        self.assertLess(
            float(
                (
                    baseline["camera_depth"] * 1.3
                    - oracle["camera_depth"]
                ).abs().max()
            ),
            1.0e-5,
        )

    def test_point_pca_frame_is_finite(self) -> None:
        points = torch.randn((128, 3)) * torch.tensor((0.3, 0.2, 0.1))
        transform = make_similarity(
            scale=0.8,
            rotation=rotation_xyz((5.0, 20.0, -15.0)),
            translation=torch.tensor((0.2, -0.1, 0.4)),
        )
        world = transform_points(points, transform)
        estimate = estimate_point_pca_similarity(world)
        self.assertTrue(torch.isfinite(estimate).all())
        self.assertGreater(float(torch.linalg.det(estimate[:3, :3])), 0.0)

    def test_volume_flatten_sentinels_have_z_fastest_order(self) -> None:
        points = canonical_voxel_points(16, "identity")
        step = 1.0 / 16.0
        np.testing.assert_allclose(points[1] - points[0], (0.0, 0.0, step))
        np.testing.assert_allclose(points[16] - points[0], (0.0, step, 0.0))
        np.testing.assert_allclose(points[256] - points[0], (step, 0.0, 0.0))

    def test_lifting_reuses_visual_content_but_pose_changes_volume(self) -> None:
        visual = torch.arange(2 * 16 * 4, dtype=torch.float32).reshape(2, 16, 4)
        depth = torch.ones((2, 64, 64), dtype=torch.float32) * 2.0
        confidence = torch.ones_like(depth)
        masks = torch.ones_like(depth)
        intrinsic = torch.tensor(
            [
                [[40.0, 0.0, 31.5], [0.0, 40.0, 31.5], [0.0, 0.0, 1.0]],
                [[40.0, 0.0, 31.5], [0.0, 40.0, 31.5], [0.0, 0.0, 1.0]],
            ]
        )
        extrinsic = torch.eye(4).repeat(2, 1, 1)
        extrinsic[:, 2, 3] = -2.0
        extrinsic[1, 0, 3] = 0.35
        kwargs = dict(
            visual_patch_features=visual,
            predicted_depth=depth,
            depth_confidence=confidence,
            masks=masks,
            intrinsics=intrinsic,
            extrinsics=extrinsic,
            prior_coords=torch.tensor([[32, 32, 32]], dtype=torch.int64),
            prior_confidence=torch.tensor([1.0]),
            calibration={"enabled": False, "fallback": "mask_visual_hull_only"},
            grid_transform="identity",
            extrinsics_type="c2w",
            camera_forward_sign=1.0,
        )
        correct, metadata, _ = build_lifting_volume(**kwargs, pose_mode="correct")
        shuffled, shuffled_metadata, _ = build_lifting_volume(
            **kwargs, pose_mode="pose_shuffle"
        )
        self.assertEqual(tuple(correct.shape), (4, 16, 16, 16))
        self.assertEqual(
            tuple(metadata.shape), (len(LIFTING_METADATA_NAMES), 16, 16, 16)
        )
        self.assertGreater(float((correct - shuffled).abs().mean()), 0.0)
        self.assertTrue(torch.isfinite(shuffled_metadata).all())

    def test_local_adapter_zero_init_and_hard_off(self) -> None:
        adapter = LocalPoseLiftingVelocityAdapter(
            visual_channels=6, latent_channels=8, hidden_dim=16
        )
        self.assertFalse(any(isinstance(module, nn.MultiheadAttention) for module in adapter.modules()))
        x_t = torch.randn((1, 8, 16, 16, 16))
        stock = torch.randn_like(x_t)
        visual = torch.randn((1, 6, 16, 16, 16))
        metadata = torch.randn((1, len(LIFTING_METADATA_NAMES), 16, 16, 16))
        metadata[:, 0] = 1.0
        t = torch.tensor([0.5])
        delta, _ = adapter(x_t, stock, t, visual, metadata)
        self.assertEqual(float(delta.abs().max()), 0.0)
        loss = (delta - torch.ones_like(delta)).square().mean()
        loss.backward()
        self.assertGreater(float(adapter.output.weight.grad.abs().sum()), 0.0)
        with torch.no_grad():
            adapter.output.weight.normal_(std=0.01)
        disabled, _ = adapter(
            x_t, stock, t, visual, metadata, physical_present=False
        )
        self.assertEqual(float(disabled.abs().max()), 0.0)

    def test_local_adapter_normalizes_flow_time_to_unit_interval(self) -> None:
        adapter = LocalPoseLiftingVelocityAdapter(
            visual_channels=2, latent_channels=8, hidden_dim=4
        )
        with torch.no_grad():
            adapter.state_projection.weight.zero_()
            adapter.state_projection.bias.zero_()
            adapter.state_projection.weight[:, -1] = 1.0
        x_t = torch.zeros((1, 8, 16, 16, 16))
        stock = torch.zeros_like(x_t)
        visual = torch.zeros((1, 2, 16, 16, 16))
        metadata = torch.zeros((1, len(LIFTING_METADATA_NAMES), 16, 16, 16))
        metadata[:, 0] = 1.0
        _, stats = adapter(
            x_t, stock, torch.tensor([1000.0]), visual, metadata
        )
        self.assertAlmostEqual(float(stats["state_hidden_rms"]), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
