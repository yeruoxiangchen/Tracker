from __future__ import annotations

import ast
import inspect
import math
import tempfile
from pathlib import Path
import unittest

import numpy as np

from pose_point_depth_mv.dataset_tools.prepare_omni_real_model_inputs import (
    load_runtime_lifting_geometry,
)
from pose_point_depth_mv.dataset_tools.prepare_omni_real_pose_mask_runtime_inputs import (
    build_object_pose_mask_runtime_input,
    deterministic_subset_keys,
    validate_protocol_object_sets,
)
from pose_point_depth_mv.evaluate_pose_mask_pointcloud_ablation import (
    paired_metric_improvements,
)
from pose_point_depth_mv.evaluate_pose_mask_external_bases import (
    paired_method_comparison,
)
from pose_point_depth_mv.pose_mask_object_canonicalization import (
    POSE_MASK_INPUT_FRONTEND_VERSION,
    PoseMaskObjectFrameConfig,
    canonicalize_pose_mask_runtime_object_frame,
    prepare_pose_mask_runtime_object_observation,
)
from pose_point_depth_mv.real_object_canonicalization import (
    apply_transform,
)
from pose_point_depth_mv.rebase_pose_mask_inference_to_reference_o import (
    object_frame_rebase_transform,
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


def _pose_mask_observation() -> dict[str, object]:
    center = np.asarray([0.35, -0.12, 0.28], dtype=np.float64)
    extent = 0.62
    camera_centers = []
    for index in range(8):
        angle = 2.0 * math.pi * index / 8.0
        camera_centers.append(
            center
            + np.asarray(
                [3.1 * math.sin(angle), 0.22, 3.1 * math.cos(angle)],
                dtype=np.float64,
            )
        )
    T_W2C = np.stack(
        [np.linalg.inv(_look_at(camera, center)) for camera in camera_centers]
    )
    K = np.repeat(
        np.asarray([[[118.0, 0.0, 64.0], [0.0, 118.0, 64.0], [0.0, 0.0, 1.0]]]),
        8,
        axis=0,
    )
    masks = []
    images = []
    center_h = np.concatenate((center, [1.0]))
    for index, pose in enumerate(T_W2C):
        camera = pose @ center_h
        pixel_h = K[index] @ camera[:3]
        u, v = pixel_h[:2] / pixel_h[2]
        span = max(8, int(round(K[index, 0, 0] * extent / camera[2])))
        low_x = max(0, int(round(u - span / 2)))
        high_x = min(127, low_x + span - 1)
        low_y = max(0, int(round(v - span / 2)))
        high_y = min(127, low_y + span - 1)
        mask = np.zeros((128, 128), dtype=np.uint8)
        mask[low_y : high_y + 1, low_x : high_x + 1] = 255
        image = np.zeros((128, 128, 3), dtype=np.uint8)
        image[mask > 0] = np.asarray([70 + index, 140, 205], dtype=np.uint8)
        masks.append(mask)
        images.append(image)
    return {
        "center": center,
        "extent": extent,
        "K": K,
        "T_W2C": T_W2C,
        "masks": masks,
        "images": images,
    }


class PoseMaskPointCloudAblationTest(unittest.TestCase):
    def test_pose_mask_frame_recovers_center_and_mask_scale(self) -> None:
        sample = _pose_mask_observation()
        config = PoseMaskObjectFrameConfig(
            expected_object_extent=0.9, scale_padding=1.05
        )
        frame = canonicalize_pose_mask_runtime_object_frame(
            sample["K"], sample["T_W2C"], sample["masks"], config=config
        )
        np.testing.assert_allclose(
            frame.T_O2W[:3, 3], sample["center"], rtol=0.0, atol=0.02
        )
        expected_scale = (
            frame.stats["mask_extent_median_W"]
            * config.scale_padding
            / config.expected_object_extent
        )
        self.assertAlmostEqual(frame.stats["scale_O2W"], expected_scale, places=12)
        self.assertFalse(frame.stats["point_cloud_consumed"])
        self.assertEqual(frame.P_O.shape, (0, 3))
        self.assertEqual(frame.point_keep_mask.shape, (0,))

    def test_official_z_up_is_proper_reexpression_of_legacy_frame(self) -> None:
        sample = _pose_mask_observation()
        legacy = canonicalize_pose_mask_runtime_object_frame(
            sample["K"], sample["T_W2C"], sample["masks"]
        )
        official = canonicalize_pose_mask_runtime_object_frame(
            sample["K"],
            sample["T_W2C"],
            sample["masks"],
            axis_convention="official_z_up",
        )
        np.testing.assert_allclose(
            official.T_O2W[:3, 3], legacy.T_O2W[:3, 3], atol=1.0e-12
        )
        np.testing.assert_allclose(
            official.T_O2W[:, 2], legacy.T_O2W[:, 1], atol=1.0e-12
        )
        np.testing.assert_allclose(
            official.T_O2W[:3, 0], legacy.T_O2W[:3, 0], atol=1.0e-12
        )
        np.testing.assert_allclose(
            official.T_O2W[:3, 1], -legacy.T_O2W[:3, 2], atol=1.0e-12
        )
        self.assertEqual(official.stats["axes"]["model_up_axis"], "+Z")

    def test_fixed_all_view_frame_rebinds_only_selected_camera_chain(self) -> None:
        sample = _pose_mask_observation()
        fixed = canonicalize_pose_mask_runtime_object_frame(
            sample["K"],
            sample["T_W2C"],
            sample["masks"],
            axis_convention="official_z_up",
        )
        selected = np.asarray([0, 2, 4, 6], dtype=np.int64)
        observation = prepare_pose_mask_runtime_object_observation(
            [sample["images"][index] for index in selected],
            [sample["masks"][index] for index in selected],
            sample["K"][selected],
            sample["T_W2C"][selected],
            camera_models=["PINHOLE"] * len(selected),
            distortion_coefficients=[[] for _ in selected],
            axis_convention="official_z_up",
            runtime_frame_override=fixed,
            feature_resolution=64,
        )
        np.testing.assert_array_equal(observation.frame.T_O2W, fixed.T_O2W)
        np.testing.assert_allclose(
            observation.frame.T_O2C,
            sample["T_W2C"][selected] @ fixed.T_O2W[None],
            atol=1.0e-12,
        )
        self.assertTrue(
            observation.frame.stats["runtime_frame_rebound_to_selected_views"]
        )

    def test_frontend_has_no_point_argument_or_raw_point_subscript(self) -> None:
        signature = inspect.signature(prepare_pose_mask_runtime_object_observation)
        self.assertNotIn("P_W", signature.parameters)
        tree = ast.parse(inspect.getsource(build_object_pose_mask_runtime_input))
        point_subscripts = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Subscript):
                continue
            if not isinstance(node.value, ast.Name) or node.value.id != "source":
                continue
            if isinstance(node.slice, ast.Constant) and node.slice.value == "P_W":
                point_subscripts.append(node)
        self.assertEqual(point_subscripts, [])

    def test_observation_contract_contains_no_gt_values(self) -> None:
        sample = _pose_mask_observation()
        observation = prepare_pose_mask_runtime_object_observation(
            sample["images"],
            sample["masks"],
            sample["K"],
            sample["T_W2C"],
            camera_models=["PINHOLE"] * 8,
            distortion_coefficients=[[] for _ in range(8)],
            feature_resolution=64,
        )
        self.assertEqual(
            observation.condition_record["format"], POSE_MASK_INPUT_FRONTEND_VERSION
        )
        self.assertFalse(observation.condition_record["point_cloud_consumed"])
        for forbidden in (
            "P_W",
            "scan_obj",
            "mesh_o",
            "target_ss",
            "target_slat",
        ):
            self.assertNotIn(forbidden, observation.condition_record)

    def test_empty_point_cache_remains_compatible_with_dino_loader(self) -> None:
        sample = _pose_mask_observation()
        frame = canonicalize_pose_mask_runtime_object_frame(
            sample["K"], sample["T_W2C"], sample["masks"]
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "runtime.npz"
            np.savez_compressed(
                path,
                K_feature=np.asarray(sample["K"], dtype=np.float32),
                T_O2C=frame.T_O2C,
                T_O2C_lifting=frame.T_O2C_lifting,
                P_O=np.empty((0, 3), dtype=np.float32),
            )
            intrinsics, extrinsics, points = load_runtime_lifting_geometry(path)
        self.assertEqual(intrinsics.shape, (8, 3, 3))
        self.assertEqual(extrinsics.shape, (8, 4, 4))
        self.assertEqual(points.shape, (0, 3))

    def test_rebase_transform_matches_explicit_world_chain_and_roundtrip(self) -> None:
        angle = math.radians(27.0)
        rotation = np.asarray(
            [
                [math.cos(angle), 0.0, math.sin(angle)],
                [0.0, 1.0, 0.0],
                [-math.sin(angle), 0.0, math.cos(angle)],
            ]
        )
        alt_O2W = np.eye(4)
        alt_O2W[:3, :3] = 1.7 * rotation
        alt_O2W[:3, 3] = [0.4, -0.2, 0.8]
        ref_O2W = np.eye(4)
        ref_O2W[:3, :3] = 0.8 * np.eye(3)
        ref_O2W[:3, 3] = [-0.3, 0.1, 0.2]
        ref_W2O = np.linalg.inv(ref_O2W)
        transform = object_frame_rebase_transform(alt_O2W, ref_W2O)
        points = np.asarray([[0.0, 0.0, 0.0], [0.2, -0.1, 0.35]])
        explicit = apply_transform(apply_transform(points, alt_O2W), ref_W2O)
        direct = apply_transform(points, transform)
        np.testing.assert_allclose(direct, explicit, rtol=0.0, atol=1.0e-12)
        np.testing.assert_allclose(
            apply_transform(direct, np.linalg.inv(transform)),
            points,
            rtol=0.0,
            atol=1.0e-12,
        )

    def test_paired_sign_convention_is_pose_mask_positive(self) -> None:
        common = {
            "object_key": "box:box_002",
            "seed": 42,
            "chamfer_l1": 0.3,
            "chamfer_l2": 0.2,
            "fscore_0p01": 0.4,
            "fscore_0p02": 0.5,
            "normal_consistency": 0.6,
        }
        point = {**common, "method": "point_mask"}
        pose = {
            **common,
            "method": "pose_mask",
            "chamfer_l1": 0.2,
            "chamfer_l2": 0.1,
            "fscore_0p01": 0.5,
            "fscore_0p02": 0.7,
            "normal_consistency": 0.8,
        }
        result = paired_metric_improvements([point, pose])
        for metric in result["metrics"].values():
            self.assertGreater(metric["mean"], 0.0)
            self.assertEqual(metric["positive_rate"], 1.0)

    def test_hash_ranked_half_split_is_order_independent(self) -> None:
        keys = [f"category:object_{index:02d}" for index in range(32)]
        first, first_ranking = deterministic_subset_keys(keys, count=16, seed=20260810)
        second, second_ranking = deterministic_subset_keys(
            list(reversed(keys)), count=16, seed=20260810
        )
        self.assertEqual(first, second)
        self.assertEqual(first_ranking, second_ranking)
        self.assertEqual(len(first), 16)

    def test_hash_ranked_confirmation_half_is_exact_complement(self) -> None:
        keys = [f"category:object_{index:02d}" for index in range(32)]
        development, ranking = deterministic_subset_keys(
            keys, count=16, seed=20260810, offset=0
        )
        confirmation, confirmation_ranking = deterministic_subset_keys(
            keys, count=16, seed=20260810, offset=16
        )
        self.assertEqual(ranking, confirmation_ranking)
        self.assertTrue(set(development).isdisjoint(confirmation))
        self.assertEqual(set(development).union(confirmation), set(keys))

    def test_coarsemodel_scope_allows_raw_cache_superset(self) -> None:
        raw = {"capture:a", "capture:b"}
        reference = {"capture:a"}
        validate_protocol_object_sets(
            raw, reference, protocol_scope="coarsemodel_real_qualitative"
        )
        with self.assertRaisesRegex(RuntimeError, "Benchmark32 object sets differ"):
            validate_protocol_object_sets(
                raw, reference, protocol_scope="benchmark32_development"
            )

    def test_formal_holdout_scope_rejects_raw_cache_superset(self) -> None:
        raw = {"capture:a", "capture:b"}
        reference = {"capture:a"}
        with self.assertRaisesRegex(RuntimeError, "formal Holdout64 object sets differ"):
            validate_protocol_object_sets(
                raw,
                reference,
                protocol_scope="formal_holdout64_blind_addendum",
            )

    def test_external_base_comparison_uses_pose_positive_sign(self) -> None:
        common = {
            "object_key": "box:box_002",
            "seed": 42,
            "chamfer_l1": 0.3,
            "chamfer_l2": 0.2,
            "fscore_0p01": 0.4,
            "fscore_0p02": 0.5,
            "normal_consistency": 0.6,
        }
        pose = {
            **common,
            "method": "pose_mask",
            "chamfer_l1": 0.2,
            "chamfer_l2": 0.1,
            "fscore_0p01": 0.5,
            "fscore_0p02": 0.7,
            "normal_consistency": 0.8,
        }
        result = paired_method_comparison(
            [pose, {**common, "method": "reconviagen_original"}],
            left="pose_mask",
            right="reconviagen_original",
        )
        for metric in result["metrics"].values():
            self.assertGreater(metric["mean"], 0.0)
            self.assertEqual(metric["left_win_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
