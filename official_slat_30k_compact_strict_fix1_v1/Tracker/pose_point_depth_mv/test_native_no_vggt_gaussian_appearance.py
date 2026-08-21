from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from pose_point_depth_mv.evaluate_native_no_vggt_gaussian_appearance import (
    _atomic_export_glb_mesh,
    _to_textured_glb_with_grad,
    appearance_metrics,
    mask_quality,
    normalize_render_cameras,
    render_registered_center_splat,
)


class NativeNoVGGTGaussianAppearanceTests(unittest.TestCase):
    def test_atomic_glb_export_keeps_glb_suffix_and_explicit_type(self) -> None:
        calls: list[tuple[Path, str]] = []

        class GlbStub:
            @staticmethod
            def export(path, *, file_type):
                resolved = Path(path)
                calls.append((resolved, file_type))
                resolved.write_bytes(b"glb")

        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "mesh_textured.glb"
            _atomic_export_glb_mesh(GlbStub(), destination)
            self.assertEqual(destination.read_bytes(), b"glb")
            self.assertEqual(calls[0][0].suffix, ".glb")
            self.assertEqual(calls[0][1], "glb")
            self.assertEqual(list(destination.parent.glob(".*.tmp-*.glb")), [])

    def test_textured_glb_baking_reenables_grad_inside_no_grad_inference(self) -> None:
        calls: list[bool] = []
        texture = torch.nn.Parameter(torch.zeros(1))

        class PostprocessingStub:
            @staticmethod
            def to_glb(*args, **kwargs):
                calls.append(torch.is_grad_enabled())
                loss = torch.square(texture - 1.0).sum()
                loss.backward()
                return "textured-glb"

        @torch.no_grad()
        def invoke():
            return _to_textured_glb_with_grad(
                PostprocessingStub,
                object(),
                object(),
                texture_size=1024,
            )

        self.assertEqual(invoke(), "textured-glb")
        self.assertEqual(calls, [True])
        self.assertIsNotNone(texture.grad)

    def test_camera_normalization_removes_similarity_scale(self) -> None:
        intrinsics = np.asarray(
            [[[400.0, 0.0, 250.0], [0.0, 420.0, 260.0], [0.0, 0.0, 1.0]]]
        )
        physical = np.eye(4, dtype=np.float64)[None]
        physical[0, :3, :3] *= 2.0
        physical[0, :3, 3] = [0.2, -0.4, 4.0]
        rigid = physical.copy()
        rigid[0, :3, :] /= 2.0
        normalized, poses = normalize_render_cameras(
            intrinsics, physical, rigid, width=500, height=520
        )
        np.testing.assert_allclose(normalized[0, 0], [0.8, 0.0, 0.5])
        np.testing.assert_allclose(normalized[0, 1], [0.0, 420.0 / 520.0, 0.5])
        np.testing.assert_allclose(poses, rigid)

    def test_fragmented_mask_is_flagged(self) -> None:
        mask = np.zeros((100, 100), dtype=bool)
        mask[10:30, 10:30] = True
        mask[60:80, 60:80] = True
        quality = mask_quality(mask)
        self.assertFalse(quality["passed"])
        self.assertIn("fragmented_foreground", quality["reasons"])

    def test_identical_appearance_scores_are_ideal(self) -> None:
        target = np.zeros((64, 64, 3), dtype=np.float32)
        mask = np.zeros((64, 64), dtype=bool)
        mask[12:52, 16:48] = True
        target[mask] = np.asarray([0.2, 0.5, 0.8], dtype=np.float32)
        metrics = appearance_metrics(
            target,
            target,
            mask,
            mask.astype(np.float32),
            alpha_threshold=0.05,
            use_lpips=False,
        )
        self.assertGreaterEqual(float(metrics["masked_psnr"]), 100.0)
        self.assertAlmostEqual(float(metrics["crop_ssim"]), 1.0, places=5)
        self.assertAlmostEqual(float(metrics["alpha_iou"]), 1.0, places=7)
        self.assertIsNone(metrics["crop_lpips"])

    def test_registered_center_splat_uses_exact_camera_projection(self) -> None:
        class GaussianStub:
            get_xyz = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
            get_color = torch.tensor([[0.25, 0.50, 0.75]], dtype=torch.float32)
            get_opacity = torch.tensor([[1.0]], dtype=torch.float32)

        extrinsic = torch.eye(4, dtype=torch.float32)
        extrinsic[2, 3] = 1.0
        intrinsic = torch.tensor(
            [[0.25, 0.0, 0.50], [0.0, 0.25, 0.50], [0.0, 0.0, 1.0]],
            dtype=torch.float32,
        )
        color, alpha = render_registered_center_splat(
            GaussianStub(), extrinsic, intrinsic, resolution=4, opacity_density=2.0
        )
        expected_alpha = 1.0 - np.exp(-2.0)
        self.assertAlmostEqual(float(alpha[2, 2]), expected_alpha, places=6)
        np.testing.assert_allclose(
            color[2, 2].numpy(),
            np.asarray([0.25, 0.50, 0.75]) * expected_alpha,
            rtol=1.0e-6,
            atol=1.0e-6,
        )
        self.assertEqual(int(torch.count_nonzero(alpha)), 1)


if __name__ == "__main__":
    unittest.main()
