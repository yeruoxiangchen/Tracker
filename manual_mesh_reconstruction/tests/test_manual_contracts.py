from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from manual_mesh_reconstruction.data_adapters.common import (
    CameraFrame,
    deferred_selection_request,
    materialize_raw_cache,
)
from manual_mesh_reconstruction.mesh_coordinates import (
    DECODER_TO_SPARSE_GRID,
    MESH_FRAME_CONTRACT,
    mesh_frame_contract_fields,
    validate_runtime_o_mesh_frame_contract,
)
from manual_mesh_reconstruction.pose_mask import (
    LEGACY_Y_UP_OBJECT_FRAME_POLICY,
    OFFICIAL_COMPATIBLE_Z_UP_OBJECT_FRAME_POLICY,
    Q_OFFICIAL_MODEL_O_TO_LEGACY_RUNTIME_O,
    PoseMaskObjectFrameConfig,
    apply_pose_mask_object_frame_policy,
    bind_runtime_object_frame_to_cameras,
)
from manual_mesh_reconstruction.projection import boundary, validate_world_projection_chain
from manual_mesh_reconstruction.runtime_o import (
    build_object_runtime_input,
    evenly_spaced_frame_indices,
    fixed_frame_name_indices,
    object_spherical_farthest_frame_indices,
)
from manual_mesh_reconstruction.canonicalization import RuntimeObjectFrame
from pose_point_depth_mv.dataset_tools.prepare_omni_real_runtime_inputs import (
    evenly_spaced_frame_indices as legacy_evenly_spaced_frame_indices,
)


class ManualContractTests(unittest.TestCase):
    @staticmethod
    def _look_at_world_to_camera(center: np.ndarray) -> np.ndarray:
        center = np.asarray(center, dtype=np.float64)
        forward = -center / np.linalg.norm(center)
        world_up = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
        right = np.cross(forward, world_up)
        right /= np.linalg.norm(right)
        down = np.cross(forward, right)
        down /= np.linalg.norm(down)
        rotation = np.stack((right, down, forward), axis=0)
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = rotation
        pose[:3, 3] = -rotation @ center
        return pose

    def test_official_compatible_axis_mapping_is_exact_proper_rotation(self) -> None:
        legacy = np.eye(3, dtype=np.float64)
        official, audit = apply_pose_mask_object_frame_policy(
            legacy,
            policy=OFFICIAL_COMPATIBLE_Z_UP_OBJECT_FRAME_POLICY,
        )
        self.assertTrue(
            np.array_equal(
                official,
                Q_OFFICIAL_MODEL_O_TO_LEGACY_RUNTIME_O[:3, :3],
            )
        )
        self.assertEqual(float(np.linalg.det(official)), 1.0)
        self.assertTrue(np.array_equal(official[:, 0], legacy[:, 0]))
        self.assertTrue(np.array_equal(official[:, 1], -legacy[:, 2]))
        self.assertTrue(np.array_equal(official[:, 2], legacy[:, 1]))
        self.assertFalse(audit["center_changed"])
        self.assertFalse(audit["scale_changed"])

    def test_legacy_axis_policy_remains_bit_exact(self) -> None:
        angle = np.deg2rad(31.0)
        legacy = np.asarray(
            [
                [np.cos(angle), 0.0, np.sin(angle)],
                [0.0, 1.0, 0.0],
                [-np.sin(angle), 0.0, np.cos(angle)],
            ],
            dtype=np.float64,
        )
        result, audit = apply_pose_mask_object_frame_policy(
            legacy,
            policy=LEGACY_Y_UP_OBJECT_FRAME_POLICY,
        )
        self.assertTrue(np.array_equal(result, legacy))
        self.assertTrue(
            np.array_equal(
                np.asarray(audit["Q_model_O_to_legacy_runtime_O"]), np.eye(4)
            )
        )

    def test_all_view_o_rebind_does_not_reestimate_object_frame(self) -> None:
        scale = 2.75
        T_O2W = np.eye(4, dtype=np.float64)
        T_O2W[:3, :3] = (
            Q_OFFICIAL_MODEL_O_TO_LEGACY_RUNTIME_O[:3, :3] * scale
        )
        T_O2W[:3, 3] = [0.4, -0.2, 1.7]
        T_W2O = np.linalg.inv(T_O2W)
        source = RuntimeObjectFrame(
            T_O2W=T_O2W,
            T_W2O=T_W2O,
            T_O2C=np.empty((0, 4, 4), dtype=np.float64),
            T_C2O=np.empty((0, 4, 4), dtype=np.float64),
            P_O=np.empty((0, 3), dtype=np.float64),
            point_keep_mask=np.empty((0,), dtype=bool),
            stats={"axes": {"reference_view_index": 7}},
            contract={
                "object_frame_policy": OFFICIAL_COMPATIBLE_Z_UP_OBJECT_FRAME_POLICY
            },
        )
        poses = np.stack([np.eye(4), np.eye(4)])
        poses[1, :3, 3] = [0.1, 0.3, 2.0]
        rebound = bind_runtime_object_frame_to_cameras(
            source,
            poses,
            selected_source_view_indices=[2, 9],
        )
        self.assertTrue(np.array_equal(rebound.T_O2W, source.T_O2W))
        self.assertTrue(np.array_equal(rebound.T_W2O, source.T_W2O))
        self.assertTrue(np.array_equal(rebound.T_O2C, poses @ T_O2W))
        self.assertTrue(rebound.stats["all_view_o_frozen_before_selection"])
        self.assertEqual(rebound.stats["selected_source_view_indices"], [2, 9])

    def test_runtime_freezes_same_all_view_o_before_time_or_random_selection(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            frames = []
            K = np.asarray(
                [[52.0, 0.0, 31.5], [0.0, 52.0, 23.5], [0.0, 0.0, 1.0]],
                dtype=np.float64,
            )
            for index in range(12):
                angle = 2.0 * np.pi * float(index) / 12.0
                center = np.asarray(
                    [3.0 * np.sin(angle), 0.15 * np.sin(2.0 * angle), 3.0 * np.cos(angle)],
                    dtype=np.float64,
                )
                image_path = source / f"{index:04d}_rgb.png"
                mask_path = source / f"{index:04d}_mask.png"
                rgb = np.zeros((48, 64, 3), dtype=np.uint8)
                rgb[..., 0] = 40 + index
                rgb[14:34, 20:44, 1] = 180
                mask = np.zeros((48, 64), dtype=np.uint8)
                mask[14:34, 20:44] = 255
                Image.fromarray(rgb, mode="RGB").save(image_path)
                Image.fromarray(mask, mode="L").save(mask_path)
                frames.append(
                    CameraFrame(
                        source_index=index,
                        source_name=f"frame_{index:04d}.png",
                        image_path=image_path,
                        mask_path=mask_path,
                        K=K,
                        T_W2C=self._look_at_world_to_camera(center),
                        pose_source="synthetic_orbit_fixture",
                    )
                )

            _, row = materialize_raw_cache(
                output_dir=root / "raw",
                dataset_type="synthetic",
                source_path=source,
                category="fixture",
                object_id="all_view_o",
                input_frames=frames,
                selection_request=deferred_selection_request(
                    len(frames), 4, policy="time_uniform", random_seed=19
                ),
                source_binding={"fixture": True},
            )
            config = PoseMaskObjectFrameConfig(
                object_frame_policy=OFFICIAL_COMPATIBLE_Z_UP_OBJECT_FRAME_POLICY
            )
            common = {
                "row": row,
                "selected_view_count": 4,
                "feature_resolution": 64,
                "foreground_margin": 1.10,
                "alpha_threshold": 0.80,
                "geometry_mode": "pose_mask",
                "resume_partial": False,
                "frame_config": config,
                "gravity_up_w": np.asarray([0.0, 1.0, 0.0]),
                "selection_seed": 19,
            }
            time_report, _ = build_object_runtime_input(
                output_dir=root / "runtime_time",
                view_selection_policy="time_uniform_valid_mask",
                build_config_sha256="time-fixture-v1",
                **common,
            )
            random_report, _ = build_object_runtime_input(
                output_dir=root / "runtime_random",
                view_selection_policy="random_valid_mask",
                build_config_sha256="random-fixture-v1",
                **common,
            )
            spherical_report, _ = build_object_runtime_input(
                output_dir=root / "runtime_spherical",
                view_selection_policy="object_spherical_farthest_valid_mask",
                build_config_sha256="spherical-fixture-v1",
                **common,
            )
            training_spherical_report, _ = build_object_runtime_input(
                output_dir=root / "runtime_training_spherical",
                view_selection_policy="training_spherical_farthest_valid_mask",
                build_config_sha256="training-spherical-fixture-v1",
                **common,
            )
            fixed_names = [
                "frame_0000.png",
                "frame_0003.png",
                "frame_0007.png",
                "frame_0011.png",
            ]
            fixed_report, _ = build_object_runtime_input(
                output_dir=root / "runtime_fixed",
                view_selection_policy="fixed_frame_names_valid_mask",
                fixed_frame_names=fixed_names,
                build_config_sha256="fixed-fixture-v1",
                **common,
            )
            with np.load(time_report["cache_npz"], allow_pickle=False) as payload:
                time_o = np.asarray(payload["T_O2W"], dtype=np.float64)
                time_selected = payload["selected_source_view_index"].tolist()
            with np.load(random_report["cache_npz"], allow_pickle=False) as payload:
                random_o = np.asarray(payload["T_O2W"], dtype=np.float64)
                random_selected = payload["selected_source_view_index"].tolist()
            with np.load(spherical_report["cache_npz"], allow_pickle=False) as payload:
                spherical_o = np.asarray(payload["T_O2W"], dtype=np.float64)
            with np.load(
                training_spherical_report["cache_npz"], allow_pickle=False
            ) as payload:
                training_spherical_o = np.asarray(payload["T_O2W"], dtype=np.float64)
            with np.load(fixed_report["cache_npz"], allow_pickle=False) as payload:
                fixed_o = np.asarray(payload["T_O2W"], dtype=np.float64)
                fixed_selected = payload["selected_source_view_index"].tolist()

            self.assertTrue(np.array_equal(time_o, random_o))
            self.assertTrue(np.array_equal(time_o, spherical_o))
            self.assertTrue(np.array_equal(time_o, training_spherical_o))
            self.assertTrue(np.array_equal(time_o, fixed_o))
            self.assertNotEqual(time_selected, random_selected)
            self.assertEqual(fixed_selected, [0, 3, 7, 11])
            self.assertEqual(
                fixed_report["selected_frame_names"],
                ["view_0000.png", "view_0003.png", "view_0007.png", "view_0011.png"],
            )
            self.assertTrue(
                spherical_report["view_selection"]["o_frozen_before_selection"]
            )
            self.assertFalse(
                spherical_report["view_selection"]["selection_reestimated_o"]
            )
            self.assertEqual(
                training_spherical_report["view_selection"]["algorithm"],
                "official_training_single_seed_spherical_fps_v1",
            )
            self.assertFalse(
                training_spherical_report["view_selection"][
                    "quality_gate_used_for_selection"
                ]
            )
            self.assertTrue(
                training_spherical_report["view_selection"][
                    "o_frozen_before_selection"
                ]
            )
            for report in (
                time_report,
                random_report,
                spherical_report,
                training_spherical_report,
                fixed_report,
            ):
                self.assertEqual(report["all_input_view_count"], 12)
                self.assertEqual(report["selected_view_count"], 4)
                self.assertTrue(report["o_frozen_before_view_selection"])
                self.assertEqual(
                    report["pose_mask_object_frame_policy"],
                    OFFICIAL_COMPATIBLE_Z_UP_OBJECT_FRAME_POLICY,
                )
                self.assertEqual(
                    report["runtime_frame_stats"]["axes"]["estimation_view_count"],
                    12,
                )

            camera_by_name = {
                frame.source_name: {
                    "frame_name": frame.source_name,
                    "model": "PINHOLE",
                    "distortion": [],
                }
                for frame in frames
            }
            with self.assertRaisesRegex(ValueError, "absent from raw cache"):
                fixed_frame_name_indices(
                    [frame.source_name for frame in frames],
                    Path(row["masks_dir"]),
                    2,
                    requested_frame_names=["frame_0000.png", "missing.png"],
                    alpha_threshold=0.80,
                    intrinsics=np.stack([K] * len(frames)),
                    camera_by_name=camera_by_name,
                )

    def test_pose_mask_config_rejects_unknown_axis_policy(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported pose-mask"):
            PoseMaskObjectFrameConfig(object_frame_policy="unknown").validate()

    def test_spherical_quality_rejects_border_clipped_foreground(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            names = [f"view_{index:02d}.png" for index in range(9)]
            K = np.asarray(
                [[52.0, 0.0, 31.5], [0.0, 52.0, 23.5], [0.0, 0.0, 1.0]],
                dtype=np.float64,
            )
            poses = []
            camera_by_name = {}
            for index, name in enumerate(names):
                mask = np.zeros((48, 64), dtype=np.uint8)
                if index == 0:
                    mask[0:22, 20:44] = 255
                else:
                    mask[10:38, 18:46] = 255
                Image.fromarray(mask, mode="L").save(root / name)
                angle = 2.0 * np.pi * float(index) / float(len(names))
                center = np.asarray(
                    [3.0 * np.sin(angle), 0.0, 3.0 * np.cos(angle)],
                    dtype=np.float64,
                )
                poses.append(self._look_at_world_to_camera(center))
                camera_by_name[name] = {
                    "frame_name": name,
                    "model": "PINHOLE",
                    "distortion": [],
                }

            selected, record = object_spherical_farthest_frame_indices(
                names,
                root,
                8,
                alpha_threshold=0.80,
                intrinsics=np.stack([K] * len(names)),
                T_W2C=np.stack(poses),
                object_points_W=np.asarray([[0.0, 0.0, 0.0]]),
                camera_by_name=camera_by_name,
                gravity_up_w=np.asarray([0.0, 1.0, 0.0]),
                selection_identity="border-visibility-fixture",
            )
            self.assertNotIn(0, selected.tolist())
            self.assertEqual(record["border_rejected_frame_names"], [names[0]])
            self.assertEqual(record["border_rejected_frame_count"], 1)
            self.assertGreaterEqual(
                min(
                    record[
                        "selected_foreground_border_margin_pixels_by_frame"
                    ].values()
                ),
                1,
            )

    def test_mesh_axes_are_native_identity(self) -> None:
        self.assertTrue(np.array_equal(DECODER_TO_SPARSE_GRID, np.eye(4)))
        payload = {
            "output_frame": "runtime-O",
            **mesh_frame_contract_fields(export_policy="fixture"),
        }
        validate_runtime_o_mesh_frame_contract(payload)
        self.assertEqual(payload["mesh_frame_contract"], MESH_FRAME_CONTRACT)

    def test_physical_projection_chain(self) -> None:
        angle = np.deg2rad(17.0)
        rotation = np.asarray(
            [
                [np.cos(angle), -np.sin(angle), 0.0],
                [np.sin(angle), np.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        T_O2W = np.eye(4)
        T_O2W[:3, :3] = rotation * 1.7
        T_O2W[:3, 3] = [0.2, -0.4, 1.3]
        T_W2C = np.stack([np.eye(4), np.eye(4)])
        T_W2C[1, :3, 3] = [0.3, 0.2, 2.0]
        stored = np.matmul(T_W2C, T_O2W[None])
        result = validate_world_projection_chain(T_O2W, T_W2C, stored)
        self.assertTrue(result["passed"])
        self.assertEqual(result["max_abs"], 0.0)

    def test_boundary_width(self) -> None:
        mask = np.zeros((9, 9), dtype=np.uint8)
        mask[4, 4] = 1
        ring = boundary(mask, width=2)
        self.assertTrue(ring[2, 2])
        self.assertFalse(ring[4, 4])

    def test_runtime_view_selection_regression(self) -> None:
        frames = [f"{index:05d}.jpg" for index in range(19)]
        expected = legacy_evenly_spaced_frame_indices(frames, 8)
        actual = evenly_spaced_frame_indices(frames, 8)
        self.assertTrue(np.array_equal(actual, expected))


if __name__ == "__main__":
    unittest.main()
