from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from pose_point_depth_mv import server2
from pose_point_depth_mv import reconstruct_real_proobjaverse_official_ss_slat as official


def _poses(names: list[str], centers: np.ndarray) -> dict[str, dict]:
    return {
        name: {"pos": np.asarray(center, dtype=np.float64)}
        for name, center in zip(names, centers)
    }


class FilterFreeSelectionTest(unittest.TestCase):
    def test_time_uniform_is_unique_chronological_and_endpoint_inclusive(self) -> None:
        self.assertEqual(
            server2.uniform_time_indices(16),
            [0, 2, 4, 6, 9, 11, 13, 15],
        )
        self.assertEqual(server2.uniform_time_indices(8), list(range(8)))

    def test_trajectory_uniform_uses_cumulative_translation_not_frame_index(self) -> None:
        # Most translation happens late, so the trajectory result must differ
        # from a uniform selection over the 12 chronological frame positions.
        x = np.asarray(
            [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        )
        centers = np.stack([x, np.zeros_like(x), np.zeros_like(x)], axis=1)
        selected, audit = server2.uniform_trajectory_indices(centers)
        self.assertEqual(len(selected), 8)
        self.assertEqual(len(set(selected)), 8)
        self.assertEqual(selected, sorted(selected))
        self.assertEqual(selected[0], 0)
        self.assertEqual(selected[-1], len(centers) - 1)
        self.assertNotEqual(selected, server2.uniform_time_indices(len(centers)))
        self.assertAlmostEqual(audit["total_translation_m"], 6.0)
        self.assertFalse(audit["stationary_trajectory_fallback"])

    def test_stationary_trajectory_falls_back_only_to_time_uniformity(self) -> None:
        centers = np.zeros((20, 3), dtype=np.float64)
        selected, audit = server2.uniform_trajectory_indices(centers)
        self.assertEqual(selected, server2.uniform_time_indices(20))
        self.assertTrue(audit["stationary_trajectory_fallback"])
        self.assertEqual(audit["fallback_policy"], "time_uniform8")

    def test_plan_ignores_client_selection_and_records_no_filtering(self) -> None:
        names = [f"frame_{index:04d}.jpg" for index in range(24)]
        centers = np.stack(
            [
                np.linspace(0.0, 2.0, len(names)),
                np.zeros(len(names)),
                np.zeros(len(names)),
            ],
            axis=1,
        )
        plan = server2.build_uniform_selection_plan(
            names,
            _poses(names, centers),
            client_selected=[23, 0, 7],
        )
        self.assertTrue(plan["passed"])
        self.assertTrue(plan["client_selection_ignored"])
        self.assertFalse(plan["frame_filtering_applied"])
        self.assertFalse(plan["quality_metric_used_for_selection"])
        self.assertEqual(plan["client_selected_payload"], [23, 0, 7])
        for mode in server2.SELECTION_MODES:
            branch = plan["branches"][mode]
            self.assertEqual(len(branch["selected_indices"]), 8)
            self.assertEqual(len(branch["selected_frame_names"]), 8)

    def test_missing_pose_is_an_error_and_is_never_silently_filtered(self) -> None:
        names = [f"frame_{index:04d}.jpg" for index in range(9)]
        centers = np.zeros((8, 3), dtype=np.float64)
        with self.assertRaisesRegex(ValueError, "no frames were filtered or replaced"):
            server2.build_uniform_selection_plan(
                names,
                _poses(names[:8], centers),
            )


class ARAxisContractTest(unittest.TestCase):
    def test_transpose_swaps_intrinsics_and_dimensions_as_one_contract(self) -> None:
        pose = {
            "intrinsics": {
                "fx": 100.0,
                "fy": 120.0,
                "cx": 1.2,
                "cy": 0.8,
                "cpu_image_width": 4,
                "cpu_image_height": 3,
            },
            "image_transform": "None",
        }
        result = server2.corrected_native_image_intrinsics(pose, 4, 3)
        self.assertEqual((result["width"], result["height"]), (3, 4))
        self.assertEqual(
            (result["fx"], result["fy"], result["cx"], result["cy"]),
            (120.0, 100.0, 0.8, 1.2),
        )

    def test_projection_view_physically_transposes_rgb_mask_and_uses_base_pose(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data"
            masks = root / "masks"
            data.mkdir()
            masks.mkdir()
            frame = "frame_0000.jpg"
            image = np.zeros((3, 4, 3), dtype=np.uint8)
            image[1, 2] = np.asarray([10, 20, 30], dtype=np.uint8)
            mask = np.zeros((3, 4), dtype=np.uint8)
            mask[1, 2] = 255
            self.assertTrue(cv2.imwrite(str(data / frame), image))
            self.assertTrue(cv2.imwrite(str(masks / "frame_0000.png"), mask))
            pose = {
                "frame_name": frame,
                "pos": np.asarray([1.0, 2.0, 3.0]),
                "quat": np.asarray([0.0, 0.0, 0.0, 1.0]),
                "rot_deg": None,
                "intrinsics": {
                    "fx": 100.0,
                    "fy": 120.0,
                    "cx": 1.2,
                    "cy": 0.8,
                    "cpu_image_width": 4,
                    "cpu_image_height": 3,
                },
                "image_transform": "None",
            }
            views, records = server2.build_axis_corrected_projection_views(
                data, masks, [frame], poses={frame: pose}
            )
            self.assertEqual(len(views), 1)
            self.assertEqual(views[0].mask.shape, (4, 3))
            self.assertEqual(views[0].image_rgb.shape, (4, 3, 3))
            self.assertEqual(int(views[0].mask[2, 1]), 255)
            np.testing.assert_allclose(
                views[0].K,
                np.asarray(
                    [[120.0, 0.0, 0.8], [0.0, 100.0, 1.2], [0.0, 0.0, 1.0]]
                ),
            )
            expected_rotation, expected_translation = (
                server2.unity_pose_to_colmap_w2c(
                    pose, image_camera_rotation_degrees=0.0
                )
            )
            np.testing.assert_allclose(
                views[0].T_W2C[:3, :3], expected_rotation
            )
            np.testing.assert_allclose(
                views[0].T_W2C[:3, 3], expected_translation
            )
            self.assertEqual(records[0]["pose_camera_rotation_degrees"], 0.0)
            self.assertEqual(
                records[0]["display_matrix_role"],
                "screen_rendering_metadata_only",
            )


class Server2ConfigurationTest(unittest.TestCase):
    def test_defaults_do_not_change_legacy_server_output_root(self) -> None:
        args = server2.build_parser().parse_args(["--capture_only"])
        self.assertEqual(args.output_root, server2.DEFAULT_OUTPUT_ROOT)
        self.assertIn("outputs2", str(args.output_root))
        self.assertNotEqual(server2.DEFAULT_OUTPUT_ROOT, server2.base.DEFAULT_OUTPUT_ROOT)
        self.assertEqual(args.selection_mode, "both")
        self.assertEqual(args.geometry_mode, "pose_mask")

    def test_model_config_freezes_exact_eight_and_disables_quality_gate(self) -> None:
        args = server2.build_parser().parse_args([])
        config = server2.build_reconstruction_config(args)
        self.assertIsNotNone(config)
        assert config is not None
        self.assertEqual(config.selected_view_count, 8)
        self.assertEqual(config.view_selection_policy, "lexical_even")
        self.assertEqual(config.geometry_mode, "pose_mask")
        self.assertEqual(config.model_o_axis_convention, "official_z_up")
        self.assertTrue(config.diagnostic_bypass_pose_mask_quality)

        paths = official.ReconstructionPaths.build(Path("/tmp/server2-test"), "session")
        commands = official._commands(Path("/tmp/server2-dataset"), paths, config)
        runtime_command = dict(
            (name, command) for name, command, _expected_report in commands
        )["runtime_o"]
        axis_index = runtime_command.index("--model_o_axis_convention")
        self.assertEqual(runtime_command[axis_index + 1], "official_z_up")

    def test_legacy_deployment_is_rejected_because_it_lacks_contour_contract(self) -> None:
        args = server2.build_parser().parse_args(["--deployment", "legacy_mixed"])
        with self.assertRaisesRegex(ValueError, "requires --deployment official"):
            server2.build_reconstruction_config(args)

    def test_branch_session_ids_are_explicit_and_safe(self) -> None:
        self.assertEqual(
            server2._branch_session_id("20260819_123456_001", "trajectory_uniform8"),
            (
                "20260819_123456_001__trajectory_uniform8"
                "__araxis_xytranspose_v1"
            ),
        )
        with self.assertRaises(ValueError):
            server2._branch_session_id("../unsafe", "time_uniform8")

    def test_new_entrypoint_exists_without_replacing_old_entrypoint(self) -> None:
        self.assertTrue(Path(server2.__file__).is_file())
        self.assertTrue((server2.TRACKER_ROOT / "pose_point_depth_mv/server.py").is_file())


if __name__ == "__main__":
    unittest.main()
