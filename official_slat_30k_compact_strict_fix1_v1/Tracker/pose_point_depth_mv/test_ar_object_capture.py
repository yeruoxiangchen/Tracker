#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import cv2
import numpy as np

from pose_point_depth_mv.ar_object_capture import (
    ARPointFilterConfig,
    CAPTURE_FORMAT,
    COLLECTION_FORMAT,
    ProjectionView,
    filter_points_by_masks,
    finalize_ar_capture,
    fuse_ar_points,
    select_image_camera_rotation,
)
from trellis_point_prior_mv.build_ar_session_smoke_dataset import (
    read_phone_poses,
    unity_pose_to_colmap_w2c,
)
from pose_point_depth_mv.dataset_tools.prepare_coarsemodel_real_raw_cache import (
    build_dataset_cache,
)


class ARObjectCaptureTest(unittest.TestCase):
    @staticmethod
    def _rotmat_to_unity_quat(rotation: np.ndarray) -> np.ndarray:
        matrix = np.asarray(rotation, dtype=np.float64)
        trace = float(np.trace(matrix))
        if trace > 0.0:
            scale = np.sqrt(trace + 1.0) * 2.0
            w = 0.25 * scale
            x = (matrix[2, 1] - matrix[1, 2]) / scale
            y = (matrix[0, 2] - matrix[2, 0]) / scale
            z = (matrix[1, 0] - matrix[0, 1]) / scale
        else:
            eigenvalues, eigenvectors = np.linalg.eigh(
                np.asarray(
                    [
                        [matrix[0, 0] - matrix[1, 1] - matrix[2, 2], matrix[0, 1] + matrix[1, 0], matrix[0, 2] + matrix[2, 0], matrix[2, 1] - matrix[1, 2]],
                        [matrix[0, 1] + matrix[1, 0], matrix[1, 1] - matrix[0, 0] - matrix[2, 2], matrix[1, 2] + matrix[2, 1], matrix[0, 2] - matrix[2, 0]],
                        [matrix[0, 2] + matrix[2, 0], matrix[1, 2] + matrix[2, 1], matrix[2, 2] - matrix[0, 0] - matrix[1, 1], matrix[1, 0] - matrix[0, 1]],
                        [matrix[2, 1] - matrix[1, 2], matrix[0, 2] - matrix[2, 0], matrix[1, 0] - matrix[0, 1], matrix.trace()],
                    ]
                )
            )
            x, y, z, w = eigenvectors[:, int(np.argmax(eigenvalues))]
        quat = np.asarray([x, y, z, w], dtype=np.float64)
        return quat / np.linalg.norm(quat)

    def test_temporal_voxel_fusion_requires_distinct_frames(self) -> None:
        config = ARPointFilterConfig(
            voxel_size_m=0.01,
            min_temporal_observations=2,
            min_object_points=1,
        )
        rows = [
            ("frame_0000.jpg", np.array([0.001, 0.001, 1.001]), 1.0),
            ("frame_0000.jpg", np.array([0.002, 0.001, 1.001]), 1.0),
            ("frame_0001.jpg", np.array([0.003, 0.001, 1.001]), 0.8),
            ("frame_0000.jpg", np.array([0.5, 0.5, 1.0]), 1.0),
        ]
        points, confidence, temporal, stats = fuse_ar_points(rows, config)
        self.assertEqual(points.shape, (1, 3))
        self.assertEqual(temporal.tolist(), [2])
        self.assertAlmostEqual(float(confidence[0]), (1.0 + 1.0 + 0.8) / 3.0)
        self.assertEqual(stats["raw_sample_count"], 4)
        self.assertEqual(stats["temporally_supported_voxel_count"], 1)

    def test_multiview_masks_reject_background_points(self) -> None:
        image = np.full((64, 64, 3), (40, 80, 120), dtype=np.uint8)
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[20:44, 20:44] = 255
        K = np.array([[64.0, 0.0, 31.5], [0.0, 64.0, 31.5], [0.0, 0.0, 1.0]])
        views = [
            ProjectionView(
                frame_name=f"frame_{index:04d}.jpg",
                image_path=Path("unused.jpg"),
                mask_path=Path("unused.png"),
                K=K,
                T_W2C=np.eye(4),
                image_rgb=image,
                mask=mask,
            )
            for index in range(2)
        ]
        points = np.array(
            [[0.0, 0.0, 1.0], [0.05, -0.05, 1.0], [0.32, 0.0, 1.0]],
            dtype=np.float64,
        )
        config = ARPointFilterConfig(
            min_mask_observations=2,
            min_mask_support_ratio=1.0,
            mask_dilation_px=0,
            min_object_points=1,
        )
        filtered, confidence, colors, stats = filter_points_by_masks(
            points,
            np.ones(3),
            np.full(3, 2),
            views,
            config,
        )
        np.testing.assert_allclose(filtered, points[:2])
        self.assertEqual(confidence.shape, (2,))
        self.assertEqual(colors.tolist(), [[40, 80, 120], [40, 80, 120]])
        self.assertEqual(stats["mask_supported_point_count"], 2)
        self.assertEqual(stats["per_view"][0]["mask_hit_point_count"], 2)

    def _write_capture_inputs(self, root: Path, *, with_points: bool) -> tuple[Path, Path]:
        data_dir = root / "raw"
        mask_dir = root / "masks"
        data_dir.mkdir()
        mask_dir.mkdir()
        pose_lines = []
        for index in range(2):
            frame_name = f"frame_{index:04d}.jpg"
            image = np.full((64, 64, 3), 127, dtype=np.uint8)
            mask = np.zeros((64, 64), dtype=np.uint8)
            mask[20:44, 20:44] = 255
            self.assertTrue(cv2.imwrite(str(data_dir / frame_name), image))
            self.assertTrue(
                cv2.imwrite(str(mask_dir / f"frame_{index:04d}.png"), mask)
            )
            pose_lines.append(
                f"{frame_name},0,0,0,0,0,0,0,0,0,1,64,64,31.5,31.5,"
                "64,64,64,64,64,64,None,1.0,1000000000,1.0,0.0,"
                "camera_frame_received,Portrait,SessionTracking,None,None\n"
            )
        (data_dir / "poses.txt").write_text("".join(pose_lines), encoding="utf-8")
        if with_points:
            unity_points = [
                {"x": x, "y": y, "z": 1.0, "confidence": 1.0}
                for x, y in [(-0.04, -0.04), (-0.04, 0.04), (0.04, -0.04), (0.04, 0.04)]
            ]
            with (data_dir / "slam_points.jsonl").open("w", encoding="utf-8") as handle:
                for index in range(2):
                    handle.write(
                        json.dumps(
                            {
                                "coordinate_frame": "unity_world",
                                "frame_name": f"frame_{index:04d}.jpg",
                                "points": unity_points,
                            }
                        )
                        + "\n"
                    )
        return data_dir, mask_dir

    def test_missing_ar_points_is_a_hard_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_dir, mask_dir = self._write_capture_inputs(root, with_points=False)
            with self.assertRaisesRegex(RuntimeError, "ARPointCloudManager"):
                finalize_ar_capture(
                    session_id="missing_points",
                    data_dir=data_dir,
                    mask_dir=mask_dir,
                    frame_names=["frame_0000.jpg", "frame_0001.jpg"],
                    output_root=root / "output",
                    input_qc={"qc_pass": True},
                    config=ARPointFilterConfig(min_object_points=1),
                )
            failure = json.loads(
                (data_dir / "capture_failure_report.json").read_text(encoding="utf-8")
            )
            self.assertFalse(failure["passed"])
            self.assertIn("ARPointCloudManager", failure["failures"][0])

    def test_pose_mask_capture_bypasses_missing_ar_points(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_dir, mask_dir = self._write_capture_inputs(root, with_points=False)
            with mock.patch(
                "pose_point_depth_mv.ar_object_capture.select_image_camera_rotation",
                side_effect=AssertionError(
                    "pose-mask capture must not run full-session geometry"
                ),
            ):
                report = finalize_ar_capture(
                    session_id="pose_mask_no_points",
                    data_dir=data_dir,
                    mask_dir=mask_dir,
                    frame_names=["frame_0000.jpg", "frame_0001.jpg"],
                    output_root=root / "output",
                    input_qc={"qc_pass": True},
                    config=ARPointFilterConfig(
                        require_pose_mask_geometry=False,
                        min_synchronized_frame_ratio=1.0,
                    ),
                    require_point_cloud=False,
                )
            dataset = Path(report["dataset_dir"])
            self.assertTrue(report["passed"])
            self.assertEqual(report["geometry_mode"], "pose_mask_only")
            self.assertFalse(report["point_cloud_consumed"])
            self.assertEqual(report["selected_frame_count"], 2)
            self.assertEqual(report["mask_support"]["mask_supported_point_count"], 0)
            geometry = report["geometry_diagnostics"]
            self.assertEqual(
                geometry["cross_view_geometry_role"],
                "deferred_to_runtime_final8",
            )
            self.assertFalse(geometry["cross_view_geometry_evaluated"])
            self.assertNotIn("candidates", geometry)
            self.assertNotIn("ray_residual_median_over_mask_extent", geometry)
            self.assertFalse((dataset / "slam_points_raw.jsonl").exists())
            with np.load(dataset / "sparse/0/object_points.npz") as points:
                self.assertEqual(points["P_W"].shape, (0, 3))

    def test_finalize_writes_sparse_dataset_and_collection_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_dir, mask_dir = self._write_capture_inputs(root, with_points=True)
            output_root = root / "output"
            report = finalize_ar_capture(
                session_id="object_0001",
                data_dir=data_dir,
                mask_dir=mask_dir,
                frame_names=["frame_0000.jpg", "frame_0001.jpg"],
                output_root=output_root,
                input_qc={"qc_pass": True},
                config=ARPointFilterConfig(
                    min_object_points=4,
                    mask_dilation_px=0,
                    max_point_to_mask_extent_ratio=10.0,
                    require_pose_mask_geometry=False,
                ),
            )
            dataset = output_root / "datasets" / "object_0001"
            self.assertTrue(report["passed"])
            self.assertEqual(report["format"], CAPTURE_FORMAT)
            for relative in (
                "images/frame_0000.jpg",
                "masks/frame_0000.png",
                "poses.txt",
                "slam_points_raw.jsonl",
                "sparse/0/cameras.txt",
                "sparse/0/images.txt",
                "sparse/0/points3D.txt",
                "sparse/0/object_points.ply",
                "sparse/0/object_points.npz",
                "sparse/0/phone_pose_meta.json",
                "capture_report.json",
            ):
                self.assertTrue((dataset / relative).is_file(), relative)
            points = np.load(dataset / "sparse/0/object_points.npz")
            self.assertEqual(points["P_W"].shape, (4, 3))
            collection = json.loads(
                (output_root / "capture_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(collection["format"], COLLECTION_FORMAT)
            self.assertEqual(collection["object_count"], 1)

            raw_row, was_reused = build_dataset_cache(
                dataset,
                sparse=dataset / "sparse" / "0",
                output_dir=root / "raw_adapter",
                min_registered_pairs=2,
                resume=False,
            )
            self.assertFalse(was_reused)
            self.assertTrue(raw_row["passed"])
            self.assertEqual(raw_row["registered_pair_count"], 2)
            self.assertEqual(raw_row["sparse_point_count"], 4)

            # A retry after an interrupted collection-manifest update must heal it.
            (output_root / "capture_manifest.json").unlink()
            reused = finalize_ar_capture(
                session_id="object_0001",
                data_dir=data_dir,
                mask_dir=mask_dir,
                frame_names=["frame_0000.jpg", "frame_0001.jpg"],
                output_root=output_root,
                input_qc={"qc_pass": True},
                config=ARPointFilterConfig(
                    min_object_points=4,
                    mask_dilation_px=0,
                    max_point_to_mask_extent_ratio=10.0,
                    require_pose_mask_geometry=False,
                ),
            )
            self.assertTrue(reused["passed"])
            self.assertTrue((output_root / "capture_manifest.json").is_file())

    def test_pose_parser_distinguishes_legacy_and_synchronized_v2_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            pose_path = Path(temporary) / "poses.txt"
            legacy = (
                "legacy.jpg,0,0,0,0,0,0,0,0,0,1,64,64,31.5,31.5,"
                "64,64,64,64,64,64,None"
            )
            current = (
                "current.jpg,0,0,0,0,0,0,0,0,0,1,64,64,31.5,31.5,"
                "64,64,64,64,64,64,None,1.0,1000000000,1.0,0.01,"
                "camera_frame_received,Portrait,SessionTracking,None,None"
            )
            pose_path.write_text(legacy + "\n" + current + "\n", encoding="utf-8")
            poses = read_phone_poses(pose_path)
            self.assertEqual(poses["legacy.jpg"]["pose_binding"], "legacy_unversioned")
            self.assertFalse(poses["legacy.jpg"]["strictly_synchronized"])
            self.assertTrue(poses["current.jpg"]["strictly_synchronized"])

    def test_image_camera_rotation_is_an_explicit_w2c_rotation(self) -> None:
        pose = {
            "frame_name": "frame.jpg",
            "pos": np.asarray([0.2, 0.3, 1.4]),
            "quat": np.asarray([0.0, 0.0, 0.0, 1.0]),
            "rot_deg": None,
        }
        r0, t0 = unity_pose_to_colmap_w2c(
            pose, image_camera_rotation_degrees=0.0
        )
        r90, t90 = unity_pose_to_colmap_w2c(
            pose, image_camera_rotation_degrees=90.0
        )
        rz90 = np.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        np.testing.assert_allclose(r90, rz90 @ r0, atol=1.0e-12)
        np.testing.assert_allclose(t90, rz90 @ t0, atol=1.0e-12)

    def test_saved_cpu_image_contract_uses_direct_display_uv_rotation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_dir = root / "data"
            mask_dir = root / "masks"
            data_dir.mkdir()
            mask_dir.mkdir()
            poses = {}
            width = height = 128
            K = np.asarray([[110.0, 0.0, 63.5], [0.0, 110.0, 63.5], [0.0, 0.0, 1.0]])
            target_w = np.asarray([0.12, 0.06, -0.04])
            flip_world = np.diag([1.0, 1.0, -1.0])
            flip_camera = np.diag([1.0, -1.0, 1.0])
            # This is the same portrait ARCore display-matrix layout emitted by
            # the phone client.  It is a screen/display-camera -> native camera
            # texture UV transform, so its -90 degree rotation is applied
            # directly to the pose camera rather than inverted.
            angle_degrees = -90.0
            angle = np.deg2rad(angle_degrees)
            image_rotation = np.asarray(
                [[np.cos(angle), -np.sin(angle), 0.0], [np.sin(angle), np.cos(angle), 0.0], [0.0, 0.0, 1.0]]
            )
            for index in range(8):
                orbit = 2.0 * np.pi * index / 8.0
                center_w = np.asarray([1.6 * np.sin(orbit), 0.15, 1.6 * np.cos(orbit)])
                forward = -center_w / np.linalg.norm(center_w)
                right = np.cross(forward, np.asarray([0.0, 1.0, 0.0]))
                right /= np.linalg.norm(right)
                down = np.cross(forward, right)
                r_cv_c2w = np.stack((right, down, forward), axis=1)
                r_base_w2c = r_cv_c2w.T
                t_base = -r_base_w2c @ center_w
                r_image_w2c = image_rotation @ r_base_w2c
                t_image = image_rotation @ t_base
                target_c = r_image_w2c @ target_w + t_image
                uv = K @ target_c
                uv = uv[:2] / uv[2]
                frame_name = f"frame_{index:04d}.jpg"
                image = np.full((height, width, 3), 96, dtype=np.uint8)
                mask = np.zeros((height, width), dtype=np.uint8)
                x, y = np.rint(uv).astype(int)
                mask[max(0, y - 8) : min(height, y + 9), max(0, x - 8) : min(width, x + 9)] = 255
                self.assertTrue(cv2.imwrite(str(data_dir / frame_name), image))
                self.assertTrue(cv2.imwrite(str(mask_dir / f"frame_{index:04d}.png"), mask))
                r_unity_c2w = flip_world @ r_cv_c2w @ flip_camera
                poses[frame_name] = {
                    "frame_name": frame_name,
                    "pos": flip_world @ center_w,
                    "quat": self._rotmat_to_unity_quat(r_unity_c2w),
                    "rot_deg": None,
                    "intrinsics": {
                        "fx": K[0, 0],
                        "fy": K[1, 1],
                        "cx": K[0, 2],
                        "cy": K[1, 2],
                        "cpu_image_width": width,
                        "cpu_image_height": height,
                    },
                    "image_transform": "None",
                    "screen_orientation": "Portrait",
                    "display_matrix": (
                        "0 1 0 0 -0.8 0 0.9 0 0 0 0 0 0 0 0 0"
                    ),
                }
            selected, _views, diagnostics = select_image_camera_rotation(
                data_dir,
                mask_dir,
                list(poses),
                poses=poses,
            )
            self.assertEqual(selected, -90.0)
            self.assertEqual(
                diagnostics["selection_source"],
                "xrcpuimage_none_direct_display_uv_v5",
            )
            contract = diagnostics["saved_image_axis_contract"]
            self.assertEqual(contract["saved_image_rotation_degrees"], 0.0)
            self.assertEqual(
                contract["display_rotation_degrees_native_to_screen"], -90.0
            )
            self.assertEqual(
                contract["pose_camera_to_native_image_rotation_degrees"], -90.0
            )
            self.assertEqual(
                contract["display_matrix_application"],
                "direct_pose_camera_to_native_cpu_image_uv_rotation",
            )
            self.assertTrue(contract["axis_contract_valid"])
            minimum_ray_candidate = min(
                diagnostics["candidates"],
                key=lambda row: row["ray_residual_median_over_mask_extent"],
            )
            self.assertEqual(
                minimum_ray_candidate["image_camera_rotation_degrees"], angle_degrees
            )
            self.assertLess(
                minimum_ray_candidate["ray_residual_median_over_mask_extent"], 0.03
            )


if __name__ == "__main__":
    unittest.main()
