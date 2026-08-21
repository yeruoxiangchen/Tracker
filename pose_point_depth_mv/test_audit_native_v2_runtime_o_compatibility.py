from __future__ import annotations

import math
import unittest

import numpy as np

from pose_point_depth_mv.audit_native_v2_runtime_o_compatibility import (
    coordinate_iou,
    deterministic_proper_similarity,
    grid_rotation,
    native_coords_to_physical,
    physical_to_native_coords,
    rotation_angle_degrees,
    transform_camera_gauge,
)
from pose_point_depth_mv.real_object_canonicalization import (
    apply_transform,
    normalize_similarity_extrinsics,
)


def _normalize(value: np.ndarray) -> np.ndarray:
    return value / np.linalg.norm(value)


def _look_at(center: np.ndarray) -> np.ndarray:
    forward = _normalize(-center)
    up = np.asarray([0.0, 1.0, 0.0])
    right = _normalize(np.cross(forward, up))
    down = _normalize(np.cross(forward, right))
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3] = np.stack((right, down, forward), axis=1)
    c2w[:3, 3] = center
    return c2w


def _project(points: np.ndarray, K: np.ndarray, pose: np.ndarray) -> np.ndarray:
    homogeneous = np.concatenate((points, np.ones((len(points), 1))), axis=1)
    camera = (pose @ homogeneous.T).T[:, :3]
    pixels = (K @ camera.T).T
    return pixels[:, :2] / pixels[:, 2:3]


class NativeV2RuntimeOCompatibilityTest(unittest.TestCase):
    def test_camera_gauge_preserves_image_projection(self) -> None:
        centers = [
            np.asarray([2.0 * math.sin(index), 0.2, 2.0 * math.cos(index)])
            for index in np.linspace(0.0, 2.0 * math.pi, 5, endpoint=False)
        ]
        poses = np.stack([np.linalg.inv(_look_at(center)) for center in centers])
        points = np.asarray(
            [[0.0, 0.0, 0.0], [0.2, -0.1, 0.15], [-0.18, 0.08, -0.1]]
        )
        K = np.asarray([[120.0, 0.0, 64.0], [0.0, 120.0, 64.0], [0.0, 0.0, 1.0]])
        gauge = deterministic_proper_similarity(42)
        transformed_points = apply_transform(points, gauge)
        transformed_poses = transform_camera_gauge(poses, gauge)
        for pose, transformed_pose in zip(poses, transformed_poses):
            np.testing.assert_allclose(
                _project(points, K, pose),
                _project(transformed_points, K, transformed_pose),
                rtol=1.0e-10,
                atol=1.0e-10,
            )

    def test_native_coordinate_roundtrip_identity(self) -> None:
        coords = np.asarray([[0, 0, 0], [7, 11, 19], [63, 63, 63]], dtype=np.int32)
        points = native_coords_to_physical(
            coords, resolution=64, grid_transform="identity"
        )
        np.testing.assert_array_equal(
            physical_to_native_coords(points, resolution=64), coords
        )
        self.assertEqual(coordinate_iou(coords, coords), 1.0)

    def test_similarity_extrinsic_normalization_preserves_projection(self) -> None:
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] *= 2.5
        pose[:3, 3] = [0.4, -0.2, 4.0]
        normalized = normalize_similarity_extrinsics(pose[None])[0]
        points = np.asarray([[0.1, -0.2, 0.0], [-0.25, 0.12, 0.3]])
        K = np.asarray([[100.0, 0.0, 64.0], [0.0, 100.0, 64.0], [0.0, 0.0, 1.0]])
        np.testing.assert_allclose(
            _project(points, K, pose),
            _project(points, K, normalized),
            rtol=1.0e-12,
            atol=1.0e-12,
        )
        np.testing.assert_allclose(normalized[:3, :3], np.eye(3))

    def test_pixal_rotation_is_proper_and_known(self) -> None:
        rotation = grid_rotation("pixal3d_rotation")
        self.assertAlmostEqual(float(np.linalg.det(rotation)), 1.0)
        self.assertAlmostEqual(rotation_angle_degrees(rotation), 90.0)

    def test_coordinate_iou_handles_partial_overlap(self) -> None:
        left = np.asarray([[0, 0, 0], [1, 1, 1]], dtype=np.int32)
        right = np.asarray([[1, 1, 1], [2, 2, 2]], dtype=np.int32)
        self.assertAlmostEqual(coordinate_iou(left, right), 1.0 / 3.0)


if __name__ == "__main__":
    unittest.main()
