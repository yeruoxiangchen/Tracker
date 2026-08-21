from __future__ import annotations

import math
import tempfile
from pathlib import Path
import unittest

import numpy as np
from PIL import Image

from ar_ss_flow.shared_object_preprocessing import (
    prepare_shared_object_arrays,
    prepare_shared_object_views,
)
from pose_point_depth_mv.real_object_canonicalization import (
    InsufficientObjectPointsError,
    RuntimeObjectFrameConfig,
    apply_transform,
    filter_mask_supported_points,
    normalize_similarity_extrinsics,
    prepare_runtime_object_observation,
    project_object_points,
    undistort_mask_view,
    undistort_rgb_mask_views,
)
from pose_point_depth_mv.real_object_label_binding import (
    bind_scan_to_runtime_object,
)


def _normalize(value: np.ndarray) -> np.ndarray:
    return value / np.linalg.norm(value)


def _look_at(camera_center: np.ndarray, target: np.ndarray) -> np.ndarray:
    forward = _normalize(target - camera_center)
    world_up = np.asarray([0.0, 1.0, 0.0])
    right = _normalize(np.cross(forward, world_up))
    down = _normalize(np.cross(forward, right))
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3] = np.stack((right, down, forward), axis=1)
    c2w[:3, 3] = camera_center
    return c2w


def _rotation_xyz(degrees: tuple[float, float, float]) -> np.ndarray:
    angles = [math.radians(value) for value in degrees]
    cx, cy, cz = [math.cos(value) for value in angles]
    sx, sy, sz = [math.sin(value) for value in angles]
    rx = np.asarray([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float64)
    ry = np.asarray([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)
    rz = np.asarray([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float64)
    return rz @ ry @ rx


def _make_global_similarity() -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = 1.7 * _rotation_xyz((19.0, -31.0, 13.0))
    transform[:3, 3] = [0.8, -1.1, 0.45]
    return transform


def _transform_cameras(T_W2C: np.ndarray, gauge: np.ndarray) -> np.ndarray:
    scale = float(np.cbrt(np.linalg.det(gauge[:3, :3])))
    rotation = gauge[:3, :3] / scale
    translation = gauge[:3, 3]
    output = []
    for pose in T_W2C:
        c2w = np.linalg.inv(pose)
        transformed = np.eye(4, dtype=np.float64)
        transformed[:3, :3] = rotation @ c2w[:3, :3]
        transformed[:3, 3] = scale * rotation @ c2w[:3, 3] + translation
        output.append(np.linalg.inv(transformed))
    return np.stack(output)


def _project_world(points: np.ndarray, K: np.ndarray, T_W2C: np.ndarray) -> np.ndarray:
    points_h = np.concatenate((points, np.ones((len(points), 1))), axis=1)
    camera = (T_W2C @ points_h.T).T[:, :3]
    pixels_h = (K @ camera.T).T
    return pixels_h[:, :2] / pixels_h[:, 2:3]


def _synthetic_observation() -> dict[str, object]:
    rng = np.random.default_rng(20260805)
    directions = rng.normal(size=(480, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    radii = rng.uniform(0.72, 1.0, size=(len(directions), 1))
    object_points = directions * radii * np.asarray([0.34, 0.25, 0.18])
    object_points[:, 0] += 0.035 * object_points[:, 2] ** 2

    camera_centers = []
    for index in range(8):
        angle = 2.0 * math.pi * index / 8.0
        camera_centers.append(
            np.asarray([2.8 * math.sin(angle), 0.18, 2.8 * math.cos(angle)])
        )
    T_W2C = np.stack(
        [np.linalg.inv(_look_at(center, np.zeros(3))) for center in camera_centers]
    )
    K = np.repeat(
        np.asarray([[[112.0, 0.0, 64.0], [0.0, 112.0, 64.0], [0.0, 0.0, 1.0]]]),
        len(T_W2C),
        axis=0,
    )
    masks = []
    images = []
    for view, pose in enumerate(T_W2C):
        pixels = _project_world(object_points, K[view], pose)
        low = np.floor(pixels.min(axis=0) - 3.0).astype(int)
        high = np.ceil(pixels.max(axis=0) + 3.0).astype(int)
        low = np.maximum(low, 0)
        high = np.minimum(high, 127)
        mask = np.zeros((128, 128), dtype=np.uint8)
        mask[low[1] : high[1] + 1, low[0] : high[0] + 1] = 255
        image = np.zeros((128, 128, 3), dtype=np.uint8)
        image[mask > 0] = np.asarray([80 + view, 130, 210], dtype=np.uint8)
        masks.append(mask)
        images.append(image)
    return {
        "images": images,
        "masks": masks,
        "K": K,
        "T_W2C": T_W2C,
        "P_W": object_points,
    }


class RealObjectCanonicalizationTest(unittest.TestCase):
    def test_similarity_normalization_repairs_float32_rotation_noise(self) -> None:
        pose = np.eye(4, dtype=np.float64)
        near_rotation = np.eye(3, dtype=np.float64)
        near_rotation[0, 1] = 1.3e-7
        pose[:3, :3] = 1.7 * near_rotation
        pose[:3, 3] = [0.2, -0.4, 3.0]

        normalized = normalize_similarity_extrinsics(pose[None])[0]
        rotation = normalized[:3, :3]
        np.testing.assert_allclose(
            rotation.T @ rotation, np.eye(3), rtol=0.0, atol=1.0e-12
        )
        self.assertAlmostEqual(float(np.linalg.det(rotation)), 1.0, places=12)
        np.testing.assert_allclose(
            normalized[:3, 3], pose[:3, 3] / 1.7, rtol=0.0, atol=1.0e-12
        )

    def test_similarity_normalization_preserves_existing_valid_rotation_bits(self) -> None:
        rotation = _rotation_xyz((17.0, -29.0, 11.0))
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = 1.3 * rotation
        normalized = normalize_similarity_extrinsics(pose[None])[0]
        np.testing.assert_array_equal(normalized[:3, :3], rotation)

    def test_similarity_normalization_rejects_anisotropic_linear_part(self) -> None:
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = np.diag([1.0, 1.0, 1.01])
        with self.assertRaisesRegex(ValueError, "not an isotropic similarity"):
            normalize_similarity_extrinsics(pose[None])

    def test_array_and_path_shared_preprocessing_are_identical(self) -> None:
        sample = _synthetic_observation()
        images = list(sample["images"][:2])
        masks = list(sample["masks"][:2])
        arrays = prepare_shared_object_arrays(
            images, masks, resolution=64, foreground_margin=1.1, alpha_threshold=0.8
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_paths = []
            mask_paths = []
            for index, (image, mask) in enumerate(zip(images, masks)):
                image_path = root / f"image_{index}.png"
                mask_path = root / f"mask_{index}.png"
                Image.fromarray(image).save(image_path)
                Image.fromarray(mask).save(mask_path)
                image_paths.append(str(image_path))
                mask_paths.append(str(mask_path))
            paths = prepare_shared_object_views(
                image_paths,
                mask_paths,
                resolution=64,
                foreground_margin=1.1,
                alpha_threshold=0.8,
            )
        self.assertEqual(arrays.geometry_record(), paths.geometry_record())
        np.testing.assert_array_equal(arrays.masks, paths.masks)
        for left, right in zip(arrays.images, paths.images):
            np.testing.assert_array_equal(np.asarray(left), np.asarray(right))

    def test_undistortion_is_noop_for_pinhole_and_keeps_rgb_mask_aligned(self) -> None:
        image = np.zeros((64, 80, 3), dtype=np.uint8)
        mask = np.zeros((64, 80), dtype=np.uint8)
        image[20:44, 26:54, 0] = 255
        mask[20:44, 26:54] = 255
        K = np.asarray([[[70.0, 0.0, 40.0], [0.0, 70.0, 32.0], [0.0, 0.0, 1.0]]])
        pinhole_rgb, pinhole_mask, pinhole_k, records = undistort_rgb_mask_views(
            [image], [mask], K, camera_models=["PINHOLE"], distortion_coefficients=[[]]
        )
        np.testing.assert_array_equal(pinhole_rgb[0], image)
        np.testing.assert_array_equal(pinhole_mask[0], mask)
        np.testing.assert_array_equal(pinhole_k, K)
        self.assertFalse(records[0]["active"])

        radial_rgb, radial_mask, radial_k, records = undistort_rgb_mask_views(
            [image],
            [mask],
            K,
            camera_models=["SIMPLE_RADIAL"],
            distortion_coefficients=[[0.08]],
        )
        rgb_support = radial_rgb[0][..., 0] > 127
        mask_support = radial_mask[0] > 127
        self.assertLess(float(np.mean(rgb_support != mask_support)), 0.01)
        np.testing.assert_array_equal(radial_k, K)
        self.assertTrue(records[0]["active"])
        np.testing.assert_array_equal(
            undistort_mask_view(
                mask,
                K[0],
                camera_model="SIMPLE_RADIAL",
                distortion_coefficients=[0.08],
            ),
            radial_mask[0],
        )

    def test_mask_filter_rejects_unobserved_background_points(self) -> None:
        sample = _synthetic_observation()
        points = np.concatenate(
            (sample["P_W"], np.asarray([[20.0, 20.0, 20.0], [-20.0, 8.0, 3.0]])),
            axis=0,
        )
        keep, stats = filter_mask_supported_points(
            points,
            sample["K"],
            sample["T_W2C"],
            sample["masks"],
            config=RuntimeObjectFrameConfig(),
        )
        self.assertTrue(bool(np.all(keep[:-2])))
        self.assertFalse(bool(np.any(keep[-2:])))
        self.assertEqual(stats["mask_supported_point_count"], len(sample["P_W"]))

    def test_point_quality_rejections_are_typed_and_structured(self) -> None:
        sample = _synthetic_observation()
        config = RuntimeObjectFrameConfig(min_object_points=32)
        with self.assertRaises(InsufficientObjectPointsError) as raw_context:
            filter_mask_supported_points(
                sample["P_W"][:16],
                sample["K"],
                sample["T_W2C"],
                sample["masks"],
                config=config,
            )
        self.assertEqual(raw_context.exception.stage, "raw")
        self.assertEqual(raw_context.exception.available, 16)

        empty_masks = [np.zeros_like(mask) for mask in sample["masks"]]
        with self.assertRaises(InsufficientObjectPointsError) as mask_context:
            filter_mask_supported_points(
                sample["P_W"],
                sample["K"],
                sample["T_W2C"],
                empty_masks,
                config=config,
            )
        self.assertEqual(mask_context.exception.stage, "mask-supported")
        self.assertEqual(mask_context.exception.available, 0)

    def test_world_sim3_changes_gauge_but_not_object_space_or_projection(self) -> None:
        sample = _synthetic_observation()
        common = {
            "camera_models": ["PINHOLE"] * 8,
            "distortion_coefficients": [[] for _ in range(8)],
            "feature_resolution": 64,
            "frame_config": RuntimeObjectFrameConfig(scale_padding=1.0),
        }
        baseline = prepare_runtime_object_observation(
            sample["images"],
            sample["masks"],
            sample["K"],
            sample["T_W2C"],
            sample["P_W"],
            **common,
        )
        gauge = _make_global_similarity()
        transformed_points = apply_transform(sample["P_W"], gauge)
        transformed_poses = _transform_cameras(sample["T_W2C"], gauge)
        transformed = prepare_runtime_object_observation(
            sample["images"],
            sample["masks"],
            sample["K"],
            transformed_poses,
            transformed_points,
            **common,
        )

        np.testing.assert_allclose(
            transformed.frame.T_O2W,
            gauge @ baseline.frame.T_O2W,
            rtol=2.0e-6,
            atol=2.0e-7,
        )
        np.testing.assert_allclose(
            transformed.frame.P_O, baseline.frame.P_O, rtol=2.0e-6, atol=2.0e-7
        )
        sentinels = np.asarray(
            [[0.0, 0.0, 0.0], [0.15, -0.08, 0.12], [-0.22, 0.10, -0.05]]
        )
        baseline_uv, baseline_depth = project_object_points(
            sentinels, baseline.intrinsics, baseline.frame.T_O2C
        )
        transformed_uv, transformed_depth = project_object_points(
            sentinels, transformed.intrinsics, transformed.frame.T_O2C
        )
        np.testing.assert_allclose(
            transformed_uv, baseline_uv, rtol=2.0e-6, atol=2.0e-6
        )
        np.testing.assert_allclose(
            transformed_depth, 1.7 * baseline_depth, rtol=2.0e-6, atol=2.0e-6
        )
        np.testing.assert_allclose(
            normalize_similarity_extrinsics(transformed.frame.T_O2C),
            normalize_similarity_extrinsics(baseline.frame.T_O2C),
            rtol=2.0e-6,
            atol=2.0e-7,
        )
        self.assertIn("T_O2C_lifting_sha256", transformed.condition_record)

    def test_object_projection_chain_matches_world_projection(self) -> None:
        sample = _synthetic_observation()
        observation = prepare_runtime_object_observation(
            sample["images"],
            sample["masks"],
            sample["K"],
            sample["T_W2C"],
            sample["P_W"],
            feature_resolution=64,
        )
        sentinels = np.asarray([[0.0, 0.0, 0.0], [0.1, -0.07, 0.16]])
        object_uv, _ = project_object_points(
            sentinels, sample["K"], observation.frame.T_O2C
        )
        world_points = apply_transform(sentinels, observation.frame.T_O2W)
        expected = np.stack(
            [
                _project_world(world_points, sample["K"][index], sample["T_W2C"][index])
                for index in range(8)
            ]
        )
        np.testing.assert_allclose(object_uv, expected, rtol=1.0e-7, atol=1.0e-7)

    def test_gt_label_binding_cannot_change_observable_condition_identity(self) -> None:
        sample = _synthetic_observation()
        observation = prepare_runtime_object_observation(
            sample["images"],
            sample["masks"],
            sample["K"],
            sample["T_W2C"],
            sample["P_W"],
            feature_resolution=64,
        )
        first = np.eye(4, dtype=np.float64)
        second = np.eye(4, dtype=np.float64)
        second[:3, :3] = 2.0 * _rotation_xyz((0.0, 35.0, 0.0))
        second[:3, 3] = [1.0, -0.5, 0.25]
        first_binding = bind_scan_to_runtime_object(
            first, T_O2W=observation.frame.T_O2W
        )
        second_binding = bind_scan_to_runtime_object(
            second, T_O2W=observation.frame.T_O2W
        )
        self.assertFalse(np.allclose(first_binding, second_binding))
        identity_before = observation.condition_sha256
        identity_after = observation.condition_record["condition_sha256"]
        self.assertEqual(identity_before, identity_after)
        for forbidden in ("scan_obj", "T_Scan2W", "target_ss", "target_slat"):
            self.assertNotIn(forbidden, observation.condition_record)
            self.assertNotIn(forbidden, observation.condition_record["runtime_frame"])


if __name__ == "__main__":
    unittest.main()
