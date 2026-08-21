#!/usr/bin/env python3
from __future__ import annotations

import math
import unittest

import numpy as np

from pose_point_depth_mv.dataset_tools.build_reconviagen_dorabench300_benchmark import (
    SELECTED_INPUT_VIEW_INDICES,
    SOURCE_TO_MODEL_O,
    object_camera_plan,
    seeded_order,
    sphere_hammersley_sequence,
)


class Dora300ProtocolTest(unittest.TestCase):
    def test_registered_input_indices_are_exact(self) -> None:
        self.assertEqual(SELECTED_INPUT_VIEW_INDICES, [0, 9, 19, 29])

    def test_registered_sample_order_is_deterministic(self) -> None:
        values = [f"object/{index:04d}.obj" for index in range(3202)]
        first = seeded_order(values, "registered_dora300", 20260821)[:300]
        second = seeded_order(values, "registered_dora300", 20260821)[:300]
        self.assertEqual(first, second)
        self.assertEqual(len(set(first)), 300)

    def test_public_trellis_hammersley_remap(self) -> None:
        yaw, pitch = sphere_hammersley_sequence(0, 40, (0.5, 0.25))
        self.assertAlmostEqual(yaw, 0.5 * math.pi)
        expected_u = 2.0 * (0.5 / 40.0)
        self.assertAlmostEqual(pitch, math.acos(1.0 - 2.0 * expected_u) - math.pi / 2.0)

    def test_per_object_camera_plan_is_deterministic_rigid_and_varying_fov(self) -> None:
        first = object_camera_plan("dora_fixture", 20260821, 1024)
        second = object_camera_plan("dora_fixture", 20260821, 1024)
        self.assertEqual(first, second)
        cameras = first["cameras"]
        self.assertEqual(len(cameras), 40)
        focals = []
        for index, camera in enumerate(cameras):
            self.assertEqual(camera["view_index"], index)
            c2w = np.asarray(camera["c2w_opencv_model_o"], dtype=np.float64)
            intrinsic = np.asarray(camera["intrinsic"], dtype=np.float64)
            self.assertTrue(np.isfinite(c2w).all())
            self.assertTrue(np.allclose(c2w[:3, :3].T @ c2w[:3, :3], np.eye(3), atol=1e-7))
            self.assertGreater(float(np.linalg.det(c2w[:3, :3])), 0.999999)
            self.assertEqual(intrinsic.shape, (3, 3))
            focals.append(float(intrinsic[0, 0]))
        self.assertGreater(max(focals) - min(focals), 1.0)

    def test_model_o_axis_rotation_is_proper(self) -> None:
        self.assertTrue(np.allclose(SOURCE_TO_MODEL_O.T @ SOURCE_TO_MODEL_O, np.eye(3)))
        self.assertAlmostEqual(float(np.linalg.det(SOURCE_TO_MODEL_O)), 1.0)


if __name__ == "__main__":
    unittest.main()
