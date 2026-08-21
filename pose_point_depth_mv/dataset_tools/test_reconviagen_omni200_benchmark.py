#!/usr/bin/env python3
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from pose_point_depth_mv.dataset_tools.build_reconviagen_omni200_benchmark import (
    SOURCE_TO_MODEL_O,
    camera_plan,
    canonical_sha256,
    seeded_order,
    selected_view_indices,
)


class Omni200ProtocolTest(unittest.TestCase):
    def test_seeded_order_is_stable_and_complete(self) -> None:
        values = ["a", "b", "c", "d"]
        first = seeded_order(values, "fixture", 42)
        second = seeded_order(values, "fixture", 42)
        self.assertEqual(first, second)
        self.assertEqual(set(first), set(values))

    def test_four_of_twenty_four_is_exact(self) -> None:
        indices = selected_view_indices("omni_fixture_001", 20260821)
        self.assertEqual(len(indices), 4)
        self.assertEqual(len(set(indices)), 4)
        self.assertTrue(all(0 <= index < 24 for index in indices))

    def test_camera_matrix_contract(self) -> None:
        intrinsic, cameras = camera_plan(image_size=512, focal_ratio=1.25, radius=2.0)
        self.assertEqual(intrinsic.shape, (3, 3))
        self.assertEqual(len(cameras), 24)
        self.assertEqual({row["elevation_deg"] for row in cameras}, {-20.0, 10.0, 40.0})
        for row in cameras:
            c2w = np.asarray(row["c2w_opencv_model_o"], dtype=np.float64)
            self.assertTrue(np.allclose(c2w[:3, :3].T @ c2w[:3, :3], np.eye(3), atol=1e-7))
            self.assertGreater(np.linalg.det(c2w[:3, :3]), 0.999999)

    def test_source_to_model_o_is_proper_rotation(self) -> None:
        self.assertTrue(np.allclose(SOURCE_TO_MODEL_O.T @ SOURCE_TO_MODEL_O, np.eye(3)))
        self.assertAlmostEqual(float(np.linalg.det(SOURCE_TO_MODEL_O)), 1.0)

    def test_canonical_hash_ignores_json_key_order(self) -> None:
        self.assertEqual(canonical_sha256({"a": 1, "b": 2}), canonical_sha256({"b": 2, "a": 1}))


if __name__ == "__main__":
    unittest.main()
