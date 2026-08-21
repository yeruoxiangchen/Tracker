from __future__ import annotations

from dataclasses import asdict
import json
import math
import tempfile
from pathlib import Path
import unittest
from unittest import mock

import numpy as np
from PIL import Image

from ar_ss_flow.shared_object_preprocessing import canonical_json_sha256
from pose_point_depth_mv.dataset_tools.prepare_omni_real_runtime_inputs import (
    InsufficientForegroundViewsError,
    build_parser,
    build_object_runtime_input,
    evenly_spaced_frame_indices,
    foreground_valid_frame_indices,
    object_azimuth_balanced_frame_indices,
    object_spherical_farthest_frame_indices,
    pose_continuity_segments,
    pose_mask_object_azimuth_balanced_frame_indices,
    pose_mask_object_spherical_farthest_frame_indices,
    runtime_input_quality_record,
)
from pose_point_depth_mv.real_object_canonicalization import RuntimeObjectFrameConfig
import pose_point_depth_mv.dataset_tools.prepare_omni_real_runtime_inputs as runtime_inputs
from pose_point_depth_mv.real_object_canonicalization import (
    REAL_INPUT_FRONTEND_VERSION,
    array_sha256,
    normalize_similarity_extrinsics,
)


def _normalize(vector: np.ndarray) -> np.ndarray:
    return vector / np.linalg.norm(vector)


def _camera_pose(center: np.ndarray) -> np.ndarray:
    forward = _normalize(-center)
    right = _normalize(np.cross(forward, np.asarray([0.0, 1.0, 0.0])))
    down = _normalize(np.cross(forward, right))
    c2w = np.eye(4)
    c2w[:3, :3] = np.stack((right, down, forward), axis=1)
    c2w[:3, 3] = center
    return np.linalg.inv(c2w)


class PrepareOmniRealRuntimeInputsTest(unittest.TestCase):
    def test_pose_mask_selection_tries_alternative_final8_before_rejecting(self) -> None:
        names = [f"frame_{index:04d}.png" for index in range(16)]
        poses = np.stack(
            [
                _camera_pose(
                    np.asarray(
                        [2.0 * math.sin(angle), 0.0, 2.0 * math.cos(angle)]
                    )
                )
                for angle in np.linspace(0.0, 2.0 * math.pi, 16, endpoint=False)
            ]
        )
        first = list(range(0, 16, 2))
        second = list(range(1, 16, 2))
        base_selection = {
            "policy": "pose_mask_object_azimuth_balanced_valid_mask",
            "selection_candidate_count": 2,
            "selection_candidates": [
                {
                    "selected_source_view_indices": first,
                    "selected_frame_names": [names[index] for index in first],
                    "azimuth_coverage_degrees": 315.0,
                    "maximum_azimuth_gap_degrees": 45.0,
                    "azimuth_gap_std_degrees": 0.0,
                },
                {
                    "selected_source_view_indices": second,
                    "selected_frame_names": [names[index] for index in second],
                    "azimuth_coverage_degrees": 315.0,
                    "maximum_azimuth_gap_degrees": 45.0,
                    "azimuth_gap_std_degrees": 0.0,
                },
            ],
        }

        def quality(*, selected_indices, **_kwargs):
            passed = int(selected_indices[0]) == 1
            median = 0.15 if passed else 0.205
            checks = {
                "ray_residual_median": passed,
                "ray_residual_p90": True,
                "orbit_gravity_agreement": True,
                "azimuth_coverage": True,
                "maximum_azimuth_gap": True,
            }
            return {
                "formal_input_passed": passed,
                "checks": checks,
                "values": {
                    "ray_residual_median_over_mask_extent": median,
                    "ray_residual_p90_over_mask_extent": 0.25,
                    "orbit_gravity_agreement": 0.99,
                    "azimuth_coverage_degrees": 315.0,
                    "maximum_azimuth_gap_degrees": 45.0,
                },
                "thresholds": {
                    "max_ray_residual_median_over_mask_extent": 0.2,
                    "max_ray_residual_p90_over_mask_extent": 0.4,
                    "min_orbit_gravity_agreement": 0.8,
                    "min_azimuth_coverage_degrees": 240.0,
                    "max_azimuth_gap_degrees": 120.0,
                },
            }

        with (
            mock.patch.object(
                runtime_inputs,
                "_pose_mask_single_segment_frame_indices",
                return_value=(np.asarray(first), base_selection),
            ),
            mock.patch.object(
                runtime_inputs,
                "_pose_mask_selected_quality",
                side_effect=quality,
            ),
        ):
            selected, record = pose_mask_object_azimuth_balanced_frame_indices(
                names,
                Path("unused"),
                8,
                alpha_threshold=0.8,
                intrinsics=np.repeat(np.eye(3)[None], len(names), axis=0),
                T_W2C=poses,
                camera_by_name={name: {} for name in names},
                gravity_up_w=np.asarray([0.0, 1.0, 0.0]),
            )

        self.assertEqual(selected.tolist(), second)
        self.assertTrue(
            record["selected_segment_final_view_quality"]["formal_input_passed"]
        )
        search = record["segment_trials"][0]["final8_candidate_search"]
        self.assertEqual(search["candidate_count"], 2)
        self.assertEqual(search["selected_candidate_rank"], 1)

    def test_pose_mask_selection_uses_one_continuous_segment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            masks_dir = Path(directory)
            names = [f"frame_{index:04d}.png" for index in range(24)]
            poses = []
            K = np.repeat(
                np.asarray(
                    [[[42.0, 0.0, 31.5], [0.0, 42.0, 31.5], [0.0, 0.0, 1.0]]]
                ),
                len(names),
                axis=0,
            )
            for index, name in enumerate(names):
                mask = np.zeros((64, 64), dtype=np.uint8)
                mask[20:44, 20:44] = 255
                Image.fromarray(mask).save(masks_dir / name)
                if index < 12:
                    angle = 2.0 * math.pi * index / 12.0
                    offset = np.zeros(3)
                else:
                    angle = 0.5 * math.pi * (index - 12) / 11.0
                    offset = np.asarray([50.0, 0.0, 0.0])
                c2w = np.linalg.inv(
                    _camera_pose(
                        np.asarray(
                            [2.0 * math.sin(angle), 0.0, 2.0 * math.cos(angle)]
                        )
                    )
                )
                c2w[:3, 3] += offset
                poses.append(np.linalg.inv(c2w))
            poses_array = np.stack(poses)
            segments, continuity = pose_continuity_segments(names, poses_array)
            self.assertEqual(continuity["jump_count"], 1)
            self.assertEqual([row["frame_count"] for row in segments], [12, 12])

            selected, record = pose_mask_object_azimuth_balanced_frame_indices(
                names,
                masks_dir,
                8,
                alpha_threshold=0.8,
                intrinsics=K,
                T_W2C=poses_array,
                camera_by_name={
                    name: {"model": "PINHOLE", "distortion": []}
                    for name in names
                },
                gravity_up_w=np.asarray([0.0, 1.0, 0.0]),
            )
            self.assertTrue(all(int(index) < 12 for index in selected))
            self.assertEqual(record["selected_segment_index"], 0)
            self.assertEqual(record["trajectory_continuity"]["jump_count"], 1)
            self.assertTrue(
                record["selected_segment_final_view_quality"][
                    "formal_input_passed"
                ]
            )
            self.assertEqual(len(record["segment_trials"]), 2)

    def test_pose_mask_final_view_quality_uses_capture_aligned_gates(self) -> None:
        poses = np.stack(
            [
                _camera_pose(
                    np.asarray(
                        [2.0 * math.cos(angle), 0.2, 2.0 * math.sin(angle)]
                    )
                )
                for angle in np.linspace(0.0, 2.0 * math.pi, 8, endpoint=False)
            ]
        )
        report = runtime_input_quality_record(
            frame_stats={
                "mask_extent_median_W": 1.0,
                "ray_center": {
                    "ray_residual_median": 0.05,
                    "ray_residual_p90": 0.10,
                },
                "axes": {"orbit_camera_up_agreement": 0.99},
            },
            T_W2C=poses,
            view_selection={
                "azimuth_coverage_degrees": 300.0,
                "maximum_azimuth_gap_degrees": 60.0,
            },
            gravity_up_w=np.asarray([0.0, 1.0, 0.0]),
            geometry_mode="pose_mask",
        )
        self.assertTrue(report["formal_input_passed"])
        self.assertEqual(report["profile"], "pose_mask_final8_capture_aligned_v1")
        self.assertNotIn("camera_roll_median", report["checks"])
        self.assertIsNone(report["values"]["point_to_mask_extent_ratio"])

        failed = runtime_input_quality_record(
            frame_stats={
                **{
                    "mask_extent_median_W": 1.0,
                    "axes": {"orbit_camera_up_agreement": 0.99},
                },
                "ray_center": {
                    "ray_residual_median": 0.21,
                    "ray_residual_p90": 0.30,
                },
            },
            T_W2C=poses,
            view_selection={
                "azimuth_coverage_degrees": 300.0,
                "maximum_azimuth_gap_degrees": 60.0,
            },
            gravity_up_w=np.asarray([0.0, 1.0, 0.0]),
            geometry_mode="pose_mask",
        )
        self.assertFalse(failed["formal_input_passed"])
        self.assertFalse(failed["checks"]["ray_residual_median"])

    def test_generic_runtime_parser_keeps_gravity_optional(self) -> None:
        args = build_parser().parse_args(
            ["--raw_cache_report", "raw.json", "--output_dir", "output"]
        )
        self.assertIsNone(args.gravity_up_w)

    def test_frame_selection_is_lexical_and_deterministic(self) -> None:
        names = ["0005.jpg", "0001.jpg", "0004.jpg", "0002.jpg", "0003.jpg"]
        indices = evenly_spaced_frame_indices(names, 3)
        self.assertEqual(
            [names[index] for index in indices], ["0001.jpg", "0003.jpg", "0005.jpg"]
        )

    def test_empty_even_frame_is_replaced_by_nearest_valid_frame(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            masks_dir = Path(directory)
            names = [f"{index:05d}.png" for index in range(8)]
            for index, name in enumerate(names):
                mask = np.zeros((8, 8), dtype=np.uint8)
                if index != 4:
                    mask[2:6, 2:6] = 255
                Image.fromarray(mask).save(masks_dir / name)

            indices, record = foreground_valid_frame_indices(
                names, masks_dir, 3, alpha_threshold=0.8
            )

            self.assertEqual([names[index] for index in indices], [
                "00000.png",
                "00003.png",
                "00007.png",
            ])
            self.assertTrue(record["fallback_used"])
            self.assertEqual(
                record["replacements"],
                [{
                    "slot": 1,
                    "empty_frame": "00004.png",
                    "replacement_frame": "00003.png",
                    "reason": "source_mask",
                }],
            )

    def test_mask_that_disappears_after_undistortion_is_replaced(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            masks_dir = Path(directory)
            names = [f"{index:05d}.png" for index in range(5)]
            for index, name in enumerate(names):
                mask = np.zeros((64, 80), dtype=np.uint8)
                if index == 4:
                    # Raw validity passes, but the isolated off-axis pixel falls
                    # below the frozen 0.8 threshold after linear undistortion.
                    mask[5, 5] = 255
                else:
                    mask[20:44, 26:54] = 255
                Image.fromarray(mask).save(masks_dir / name)
            K = np.repeat(
                np.asarray(
                    [[[70.0, 0.0, 40.0], [0.0, 70.0, 32.0], [0.0, 0.0, 1.0]]]
                ),
                len(names),
                axis=0,
            )
            camera_by_name = {
                name: {
                    "frame_name": name,
                    "model": "SIMPLE_RADIAL",
                    "distortion": [0.08],
                }
                for name in names
            }

            indices, record = foreground_valid_frame_indices(
                names,
                masks_dir,
                3,
                alpha_threshold=0.8,
                intrinsics=K,
                camera_by_name=camera_by_name,
            )

            self.assertEqual(
                [names[index] for index in indices],
                ["00000.png", "00002.png", "00003.png"],
            )
            self.assertEqual(
                record,
                {
                    "policy": "lexical_even_with_nearest_valid_mask_fallback",
                    "validity_domain": "post_runtime_undistortion_mask",
                    "fallback_used": True,
                    "replacements": [
                        {
                            "slot": 2,
                            "empty_frame": "00004.png",
                            "replacement_frame": "00003.png",
                            "reason": "post_runtime_undistortion_mask",
                        }
                    ],
                },
            )

    def test_frame_selection_rejects_too_few_valid_masks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            masks_dir = Path(directory)
            names = [f"{index:05d}.png" for index in range(5)]
            for index, name in enumerate(names):
                mask = np.zeros((8, 8), dtype=np.uint8)
                if index < 2:
                    mask[2:6, 2:6] = 255
                Image.fromarray(mask).save(masks_dir / name)
            with self.assertRaises(InsufficientForegroundViewsError) as context:
                foreground_valid_frame_indices(
                    names, masks_dir, 3, alpha_threshold=0.8
                )
            self.assertEqual(context.exception.available, 2)
            self.assertEqual(context.exception.required, 3)

    def test_object_azimuth_selection_minimizes_maximum_gap_deterministically(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            masks_dir = Path(directory)
            names = [f"{index:05d}.png" for index in range(12)]
            poses = []
            for index, name in enumerate(names):
                mask = np.zeros((32, 32), dtype=np.uint8)
                mask[8:24, 8:24] = 255
                Image.fromarray(mask).save(masks_dir / name)
                angle = 2.0 * math.pi * index / len(names)
                poses.append(
                    _camera_pose(
                        np.asarray([2.0 * math.sin(angle), 0.1, 2.0 * math.cos(angle)])
                    )
                )
            K = np.repeat(
                np.asarray([[[30.0, 0.0, 15.5], [0.0, 30.0, 15.5], [0.0, 0.0, 1.0]]]),
                len(names),
                axis=0,
            )
            cameras = {
                name: {"model": "PINHOLE", "distortion": []} for name in names
            }
            kwargs = {
                "alpha_threshold": 0.8,
                "intrinsics": K,
                "T_W2C": np.stack(poses),
                "object_points_W": np.asarray(
                    [[-0.1, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0]]
                ),
                "camera_by_name": cameras,
                "gravity_up_w": np.asarray([0.0, 1.0, 0.0]),
            }
            first, record = object_azimuth_balanced_frame_indices(
                names, masks_dir, 4, **kwargs
            )
            second, second_record = object_azimuth_balanced_frame_indices(
                names, masks_dir, 4, **kwargs
            )
            np.testing.assert_array_equal(first, second)
            self.assertEqual(
                [names[index] for index in first],
                ["00000.png", "00003.png", "00006.png", "00009.png"],
            )
            self.assertAlmostEqual(record["maximum_azimuth_gap_degrees"], 90.0)
            self.assertAlmostEqual(record["azimuth_coverage_degrees"], 270.0)
            self.assertEqual(record, second_record)

    def test_object_spherical_farthest_matches_official_greedy_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            masks_dir = Path(directory)
            names = [f"view_{index:02d}.png" for index in range(18)]
            centers = []
            poses = []
            for index, name in enumerate(names):
                azimuth = 2.0 * math.pi * index / len(names)
                elevation = math.radians((-35.0, 0.0, 35.0)[index % 3])
                center = 2.0 * np.asarray(
                    [
                        math.cos(elevation) * math.sin(azimuth),
                        math.sin(elevation),
                        math.cos(elevation) * math.cos(azimuth),
                    ]
                )
                centers.append(center)
                poses.append(_camera_pose(center))
                mask = np.zeros((40, 40), dtype=np.uint8)
                if index != 7:
                    mask[10:30, 10:30] = 255
                Image.fromarray(mask).save(masks_dir / name)
            K = np.repeat(
                np.asarray(
                    [[[38.0, 0.0, 19.5], [0.0, 38.0, 19.5], [0.0, 0.0, 1.0]]]
                ),
                len(names),
                axis=0,
            )
            identity = "real:test-object"
            kwargs = {
                "alpha_threshold": 0.8,
                "intrinsics": K,
                "T_W2C": np.stack(poses),
                "object_points_W": np.asarray(
                    [[-0.01, 0.0, 0.0], [0.01, 0.0, 0.0], [0.0, 0.0, 0.0]]
                ),
                "camera_by_name": {
                    name: {"model": "PINHOLE", "distortion": []}
                    for name in names
                },
                "gravity_up_w": np.asarray([0.0, 1.0, 0.0]),
                "selection_identity": identity,
            }
            selected, record = object_spherical_farthest_frame_indices(
                names, masks_dir, 8, **kwargs
            )
            repeated, repeated_record = object_spherical_farthest_frame_indices(
                names, masks_dir, 8, **kwargs
            )

            np.testing.assert_array_equal(selected, repeated)
            self.assertEqual(record, repeated_record)
            self.assertNotIn(7, selected.tolist())
            valid = [index for index in range(len(names)) if index != 7]
            expected_first = min(
                valid,
                key=lambda index: canonical_json_sha256(
                    {"uid": identity, "view": names[index]}
                ),
            )
            self.assertEqual(int(selected[0]), expected_first)

            directions = np.stack(centers)
            directions /= np.linalg.norm(directions, axis=1, keepdims=True)
            for position in range(1, len(selected)):
                already = selected[:position]

                def minimum_angle(index: int) -> float:
                    cosine = directions[already] @ directions[index]
                    return float(np.arccos(np.clip(cosine, -1.0, 1.0)).min())

                chosen_score = minimum_angle(int(selected[position]))
                best_score = max(
                    minimum_angle(index)
                    for index in valid
                    if index not in already
                )
                self.assertAlmostEqual(chosen_score, best_score, places=12)
            self.assertEqual(
                record["algorithm"],
                "official_style_object_centered_spherical_farthest_point_v1",
            )
            self.assertEqual(record["selection_candidate_count"], 1)
            self.assertGreater(
                record["minimum_pairwise_angular_separation_degrees"], 25.0
            )

    def test_runtime_parser_accepts_spherical_farthest_policy(self) -> None:
        args = build_parser().parse_args(
            [
                "--raw_cache_report",
                "raw.json",
                "--output_dir",
                "output",
                "--view_selection_policy",
                "object_spherical_farthest_valid_mask",
            ]
        )
        self.assertEqual(
            args.view_selection_policy, "object_spherical_farthest_valid_mask"
        )

    def test_pose_mask_spherical_wrapper_requests_spherical_core(self) -> None:
        expected = np.asarray([0, 1, 2, 3], dtype=np.int64)
        with mock.patch.object(
            runtime_inputs,
            "pose_mask_object_azimuth_balanced_frame_indices",
            return_value=(expected, {"policy": "sentinel"}),
        ) as core:
            selected, record = pose_mask_object_spherical_farthest_frame_indices(
                [f"{index}.png" for index in range(4)],
                Path("unused"),
                4,
                alpha_threshold=0.8,
                intrinsics=np.repeat(np.eye(3)[None], 4, axis=0),
                T_W2C=np.repeat(np.eye(4)[None], 4, axis=0),
                camera_by_name={},
                gravity_up_w=np.asarray([0.0, 1.0, 0.0]),
                selection_identity="object:test",
            )
        np.testing.assert_array_equal(selected, expected)
        self.assertEqual(record["policy"], "sentinel")
        self.assertEqual(core.call_args.kwargs["_selection_algorithm"], "spherical_farthest")
        self.assertEqual(core.call_args.kwargs["selection_identity"], "object:test")

    def test_pose_mask_selection_uses_full_pool_without_points(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            masks_dir = Path(directory)
            names = [f"{index:05d}.png" for index in range(12)]
            poses = []
            for index, name in enumerate(names):
                mask = np.zeros((32, 32), dtype=np.uint8)
                mask[8:24, 8:24] = 255
                Image.fromarray(mask).save(masks_dir / name)
                angle = 2.0 * math.pi * index / len(names)
                poses.append(
                    _camera_pose(
                        np.asarray([2.0 * math.sin(angle), 0.0, 2.0 * math.cos(angle)])
                    )
                )
            K = np.repeat(
                np.asarray([[[30.0, 0.0, 15.5], [0.0, 30.0, 15.5], [0.0, 0.0, 1.0]]]),
                len(names),
                axis=0,
            )
            selected, record = pose_mask_object_azimuth_balanced_frame_indices(
                names,
                masks_dir,
                4,
                alpha_threshold=0.8,
                intrinsics=K,
                T_W2C=np.stack(poses),
                camera_by_name={
                    name: {"model": "PINHOLE", "distortion": []}
                    for name in names
                },
                gravity_up_w=np.asarray([0.0, 1.0, 0.0]),
            )
            self.assertEqual(
                [names[index] for index in selected],
                ["00000.png", "00003.png", "00006.png", "00009.png"],
            )
            self.assertEqual(record["candidate_frame_count"], 12)
            self.assertEqual(record["foreground_valid_frame_count"], 12)
            self.assertFalse(record["point_cloud_consumed"])
            self.assertEqual(
                record["object_center_source"],
                "least_squares_mask_centroid_ray_intersection",
            )

            spherical, spherical_record = (
                pose_mask_object_spherical_farthest_frame_indices(
                    names,
                    masks_dir,
                    4,
                    alpha_threshold=0.8,
                    intrinsics=K,
                    T_W2C=np.stack(poses),
                    camera_by_name={
                        name: {"model": "PINHOLE", "distortion": []}
                        for name in names
                    },
                    gravity_up_w=np.asarray([0.0, 1.0, 0.0]),
                    selection_identity="pose-mask:test",
                )
            )
            self.assertEqual(len(spherical), 4)
            self.assertEqual(
                spherical_record["policy"],
                "pose_mask_segmented_object_spherical_farthest_valid_mask",
            )
            self.assertEqual(
                spherical_record["algorithm"],
                "official_style_object_centered_spherical_farthest_point_v1",
            )
            self.assertTrue(
                spherical_record["selected_segment_final_view_quality"][
                    "formal_input_passed"
                ]
            )

    def test_builder_emits_input_only_atomic_cache(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_root = root / "source"
            images_dir = source_root / "images"
            masks_dir = source_root / "masks"
            images_dir.mkdir(parents=True)
            masks_dir.mkdir(parents=True)
            rng = np.random.default_rng(9)
            points = rng.normal(size=(240, 3))
            points /= np.linalg.norm(points, axis=1, keepdims=True)
            points *= rng.uniform(0.12, 0.30, size=(len(points), 1))
            points *= np.asarray([1.0, 0.8, 0.6])
            frame_names = [f"{index:05d}.png" for index in range(6)]
            poses = []
            K = np.repeat(
                np.asarray([[[90.0, 0.0, 48.0], [0.0, 90.0, 48.0], [0.0, 0.0, 1.0]]]),
                len(frame_names),
                axis=0,
            )
            camera_rows = []
            for index, name in enumerate(frame_names):
                angle = 2.0 * math.pi * index / len(frame_names)
                pose = _camera_pose(
                    np.asarray([2.4 * math.sin(angle), 0.1, 2.4 * math.cos(angle)])
                )
                poses.append(pose)
                points_h = np.concatenate((points, np.ones((len(points), 1))), axis=1)
                camera = (pose @ points_h.T).T[:, :3]
                pixels_h = (K[index] @ camera.T).T
                pixels = pixels_h[:, :2] / pixels_h[:, 2:3]
                low = np.maximum(np.floor(pixels.min(axis=0) - 2).astype(int), 0)
                high = np.minimum(np.ceil(pixels.max(axis=0) + 2).astype(int), 95)
                mask = np.zeros((96, 96), dtype=np.uint8)
                mask[low[1] : high[1] + 1, low[0] : high[0] + 1] = 255
                image = np.zeros((96, 96, 3), dtype=np.uint8)
                image[mask > 0] = [100, 150, 200]
                Image.fromarray(image).save(images_dir / name)
                Image.fromarray(mask).save(masks_dir / name)
                camera_rows.append(
                    {
                        "frame_name": name,
                        "model": "PINHOLE",
                        "distortion": [],
                    }
                )
            cache_path = source_root / "raw_camera_point_cache.npz"
            np.savez_compressed(
                cache_path,
                frame_name=np.asarray(frame_names),
                K=K,
                T_W2C=np.stack(poses),
                P_W=points,
                point_confidence_proxy=np.ones(len(points)),
            )
            row = {
                "category": "synthetic",
                "object_id": "object_001",
                "cache_npz": str(cache_path),
                "images_dir": str(images_dir),
                "masks_dir": str(masks_dir),
                "cameras": camera_rows,
                "scan_obj": "/forbidden/label/Scan.obj",
            }
            config = RuntimeObjectFrameConfig(min_object_points=32)
            build_config = {
                "selected_view_count": 4,
                "feature_resolution": 32,
                "frame_config": asdict(config),
            }
            output_dir = root / "output"
            report, reused = build_object_runtime_input(
                row,
                output_dir=output_dir,
                selected_view_count=4,
                feature_resolution=32,
                foreground_margin=1.1,
                alpha_threshold=0.8,
                frame_config=config,
                build_config_sha256=canonical_json_sha256(build_config),
            )
            second, second_reused = build_object_runtime_input(
                row,
                output_dir=output_dir,
                selected_view_count=4,
                feature_resolution=32,
                foreground_margin=1.1,
                alpha_threshold=0.8,
                frame_config=config,
                build_config_sha256=canonical_json_sha256(build_config),
            )

            self.assertFalse(reused)
            self.assertTrue(second_reused)
            self.assertEqual(report["condition_sha256"], second["condition_sha256"])
            self.assertFalse(report["training_ready"])
            self.assertEqual(
                report["input_frontend_format"], REAL_INPUT_FRONTEND_VERSION
            )
            self.assertTrue(report["forbidden_gt_fields_absent"])
            self.assertNotIn("scan_obj", report)
            self.assertNotIn("Scan.obj", json.dumps(report))
            self.assertTrue(Path(report["cache_npz"]).is_file())
            self.assertEqual(len(report["prepared_rgb_paths"]), 4)
            selected_mask_areas = []
            for name in report["selected_frame_names"]:
                with Image.open(masks_dir / name) as handle:
                    selected_mask_areas.append(
                        int(np.count_nonzero(np.asarray(handle.convert("L")) > 127))
                    )
            expected_reference = max(
                range(len(selected_mask_areas)),
                key=lambda index: (selected_mask_areas[index], -index),
            )
            self.assertEqual(report["reference_view_index"], expected_reference)
            with np.load(report["cache_npz"], allow_pickle=False) as cache:
                self.assertEqual(cache["K_feature"].shape, (4, 3, 3))
                self.assertEqual(cache["T_O2C"].shape, (4, 4, 4))
                self.assertEqual(cache["T_O2C_lifting"].shape, (4, 4, 4))
                np.testing.assert_allclose(
                    cache["T_O2C_lifting"],
                    normalize_similarity_extrinsics(cache["T_O2C"]),
                    rtol=1.0e-9,
                    atol=1.0e-10,
                )
                self.assertEqual(
                    report["T_O2C_lifting_sha256"],
                    array_sha256(cache["T_O2C_lifting"]),
                )
                self.assertGreater(len(cache["P_O"]), 32)
            condition = json.loads(Path(report["condition_record"]).read_text())
            self.assertEqual(
                condition["T_O2C_lifting_sha256"],
                report["T_O2C_lifting_sha256"],
            )


if __name__ == "__main__":
    unittest.main()
