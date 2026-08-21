from __future__ import annotations

import json
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
from PIL import Image
import trimesh

from manual_mesh_reconstruction.common import sha256_file
from manual_mesh_reconstruction.mesh_coordinates import mesh_frame_contract_fields
from manual_mesh_reconstruction.server import (
    AXIS_CONTRACT,
    MOBILE_MESH_COORDINATE_FRAME,
    PHONE_POSE_BINDING,
    PHONE_POSE_COORDINATE_FRAME,
    build_selection_plan,
    export_world_and_mobile_meshes,
    materialize_all_view_phone_capture,
    restore_phone_orientation_contours,
    unity_object_pose_from_t_o2a0,
)
from manual_mesh_reconstruction.alignment_refinement import (
    MAX_OPTIMIZATION_VIEWS,
    MIN_OPTIMIZATION_VIEWS,
    evenly_spaced_indices,
    internal_camera_from_unity_metadata,
    internal_similarity_to_unity,
    mask_prompt,
    object_centered_spherical_farthest_indices,
    unity_similarity_to_internal,
)
from pose_point_depth_mv.ar_mobile_overlay import read_mobile_overlay_mesh
import manual_mesh_reconstruction.server as phone_server


def _write_phone_pose_file(path: Path, frame_count: int) -> None:
    header = [
        "frame_name",
        "px",
        "py",
        "pz",
        "rx",
        "ry",
        "rz",
        "qx",
        "qy",
        "qz",
        "qw",
        "fx",
        "fy",
        "cx",
        "cy",
        "width",
        "height",
        "image_width",
        "image_height",
        "cpu_image_width",
        "cpu_image_height",
        "image_transform",
        "cpu_image_timestamp_s",
        "camera_frame_timestamp_ns",
        "pose_sample_realtime_s",
        "camera_frame_timestamp_delta_s",
        "pose_binding",
        "screen_orientation",
        "tracking_state",
        "display_matrix",
        "projection_matrix",
    ]
    rows = [",".join(header)]
    for index in range(frame_count):
        values = [
            f"frame_{index:04d}.jpg",
            f"{index * 0.1:.6f}",
            "0",
            "0",
            "0",
            "0",
            "0",
            "0",
            "0",
            "0",
            "1",
            "30",
            "31",
            "15.5",
            "11.5",
            "32",
            "24",
            "32",
            "24",
            "32",
            "24",
            "None",
            f"{index * 0.2:.6f}",
            str(index * 200_000_000),
            f"{index * 0.2:.6f}",
            "0.001",
            PHONE_POSE_BINDING,
            "Portrait",
            "Tracking",
            "",
            "",
            "unity_capture_anchor_a0",
            "Tracking",
        ]
        rows.append(",".join(values))
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


class PhonePhaseOneServerTests(unittest.TestCase):
    def test_alignment_camera_contract_matches_primary_capture(self) -> None:
        metadata = {
            "pos_x": 0.23,
            "pos_y": -0.41,
            "pos_z": 0.77,
            "quat_x": 0.10259784,
            "quat_y": -0.20519567,
            "quat_z": 0.30779351,
            "quat_w": 0.92338052,
        }
        observed = internal_camera_from_unity_metadata(metadata)
        rotation, translation = phone_server.unity_pose_to_colmap_w2c(
            {
                "pos": np.asarray(
                    [metadata["pos_x"], metadata["pos_y"], metadata["pos_z"]],
                    dtype=np.float64,
                ),
                "quat": np.asarray(
                    [
                        metadata["quat_x"],
                        metadata["quat_y"],
                        metadata["quat_z"],
                        metadata["quat_w"],
                    ],
                    dtype=np.float64,
                ),
            },
            image_camera_rotation_degrees=0.0,
        )
        expected = np.eye(4, dtype=np.float64)
        expected[:3, :3] = rotation
        expected[:3, 3] = translation
        np.testing.assert_allclose(observed, expected, atol=1.0e-9)

    def test_mobile_overlay_audit_upload_is_bounded_and_diagnostic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in (
                "00_mobile_screen_composite",
                "01_mobile_outline_texture",
                "02_frame_metadata",
                "03_raw_camera_rgb",
                "04_server_raw_sensor_reprojection",
                "05_server_display_aligned_reprojection",
                "06_original_reconstruction_input",
                "07_live_vs_reconstruction_input",
                "08_phone_vs_server_display_comparison",
            ):
                (root / name).mkdir()
            target_image = root / "reconstruction_input.jpg"
            Image.new("RGB", (64, 96), (30, 50, 70)).save(target_image)
            state = {
                "format": phone_server.MOBILE_OVERLAY_AUDIT_FORMAT,
                "status": "capturing",
                "created_at_utc": phone_server.utc_now(),
                "session_id": "unit_test_session",
                "lifecycle_generation": 7,
                "audit_id": "attempt_001",
                "runtime_o_sha256": "runtime",
                "requested_pose_sha256": "pose",
                "mobile_mesh": "/diagnostic/mesh.armesh",
                "mobile_mesh_sha256": "mesh",
                "strict_reconstruction_input_pose_matching": True,
                "target_translation_tolerance_meters": 0.025,
                "target_rotation_tolerance_degrees": 3.0,
                "pose_targets": [
                    {
                        "target_index": 0,
                        "source_frame_name": "frame_0000.jpg",
                        "source_image": str(target_image),
                        "source_image_sha256": sha256_file(target_image),
                        "position": [0.1, 0.2, 0.3],
                        "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
                        "pose_binding": phone_server.PHONE_POSE_BINDING,
                        "pose_coordinate_frame": (
                            phone_server.PHONE_POSE_COORDINATE_FRAME
                        ),
                    }
                ],
                "maximum_frame_count": 1,
                "rows": [],
                "root": str(root),
            }
            phone_server._MOBILE_OVERLAY_AUDITS["unit_test_session"] = state
            image = Image.new("RGB", (64, 96), (20, 40, 60))
            encoded = io.BytesIO()
            image.save(encoded, format="PNG")
            encoded.seek(0)
            raw_encoded = io.BytesIO()
            image.save(raw_encoded, format="JPEG")
            raw_encoded.seek(0)
            form = {
                "session_id": "unit_test_session",
                "lifecycle_generation": "7",
                "audit_id": "attempt_001",
                "overlay_contract": phone_server.MOBILE_OVERLAY_AUDIT_CONTRACT,
                "target_index": "0",
                "target_source_frame_name": "frame_0000.jpg",
                "screen_capture_encoding": "png_lossless_native",
                "capture_anchor_tracking_state": "Tracking",
                "camera_pos_x": "0.1",
                "camera_pos_y": "0.2",
                "camera_pos_z": "0.3",
                "camera_quat_x": "0",
                "camera_quat_y": "0",
                "camera_quat_z": "0",
                "camera_quat_w": "1",
                "raw_camera_pos_x": "0.1",
                "raw_camera_pos_y": "0.2",
                "raw_camera_pos_z": "0.3",
                "raw_camera_quat_x": "0",
                "raw_camera_quat_y": "0",
                "raw_camera_quat_z": "0",
                "raw_camera_quat_w": "1",
                "mesh_pos_x": "0",
                "mesh_pos_y": "0",
                "mesh_pos_z": "0",
                "mesh_quat_x": "0",
                "mesh_quat_y": "0",
                "mesh_quat_z": "0",
                "mesh_quat_w": "1",
                "mesh_uniform_scale": "1",
                "screen_capture_realtime_s": "10.1",
                "camera_pose_sample_realtime_s": "10.0",
                "camera_pose_to_screen_capture_delta_s": "0.1",
                "raw_cpu_image_timestamp_s": "20.0",
                "raw_camera_frame_timestamp_ns": "20000000000",
                "raw_pose_sample_realtime_s": "10.0",
                "raw_cpu_to_camera_frame_timestamp_delta_s": "0.0",
                "raw_cpu_image_width": "64",
                "raw_cpu_image_height": "96",
                "raw_image_transform": "None",
                "fx": "60",
                "fy": "60",
                "cx": "32",
                "cy": "48",
                "intrinsic_width": "64",
                "intrinsic_height": "96",
                "screen_width": "64",
                "screen_height": "96",
                "screen_orientation": "Portrait",
                "outline_method": "ViewDependentMeshLines",
                "outline_display_requested": "true",
                "display_matrix": "1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1",
                "projection_matrix": "1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1",
                "raw_display_matrix": "1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1",
                "raw_projection_matrix": "1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1",
                "a0_world_pos_x": "1",
                "a0_world_pos_y": "2",
                "a0_world_pos_z": "3",
                "a0_world_quat_x": "0",
                "a0_world_quat_y": "0",
                "a0_world_quat_z": "0",
                "a0_world_quat_w": "1",
                "camera_world_pos_x": "1.1",
                "camera_world_pos_y": "2.2",
                "camera_world_pos_z": "3.3",
                "camera_world_quat_x": "0",
                "camera_world_quat_y": "0",
                "camera_world_quat_z": "0",
                "camera_world_quat_w": "1",
                "mesh_world_pos_x": "1",
                "mesh_world_pos_y": "2",
                "mesh_world_pos_z": "3",
                "mesh_world_quat_x": "0",
                "mesh_world_quat_y": "0",
                "mesh_world_quat_z": "0",
                "mesh_world_quat_w": "1",
                "mesh_world_scale_x": "1",
                "mesh_world_scale_y": "1",
                "mesh_world_scale_z": "1",
                "capture_anchor_trackable_id": "unit-anchor",
                "capture_anchor_pose_valid": "true",
                "capture_anchor_tracking_stable": "true",
                "capture_anchor_ever_tracked": "true",
                "capture_anchor_uses_tracked_ar_anchor": "true",
                "capture_anchor_tracking_since_realtime_s": "5",
                "ar_session_state": "SessionTracking",
                "application_paused": "false",
                "application_focused": "true",
                "camera_frame_sequence": "123",
                "device_model": "unit-phone",
                "operating_system": "unit-os",
                "application_version": "1",
                "battery_level": "0.8",
                "battery_status": "Discharging",
                "alignment_refinement_state": "Ready",
                "diagnostic_stage": "pre_fast_alignment",
                "mobile_realtime_s": "10.1",
                "composite": (encoded, "screen.png"),
                "raw_camera": (raw_encoded, "raw.jpg"),
            }
            app = phone_server.transport_server.legacy.app
            def fake_reprojection(**kwargs):
                destination = Path(kwargs["destination"])
                with Image.open(kwargs["raw_camera_path"]) as decoded:
                    overlay = decoded.convert("RGB")
                overlay.save(destination)
                return {
                    "passed": True,
                    "overlay": str(destination.resolve()),
                    "overlay_sha256": sha256_file(destination),
                }

            with mock.patch.object(
                phone_server,
                "_render_mobile_pose_diagnostic",
                side_effect=fake_reprojection,
            ):
                with app.test_request_context(
                    "/mobile_overlay_audit/upload",
                    method="POST",
                    data=form,
                    content_type="multipart/form-data",
                ):
                    response, status = (
                        phone_server.collection_mobile_overlay_audit_upload()
                    )
            self.assertEqual(status, 200)
            self.assertTrue(response.get_json()["complete"])
            report = json.loads((root / "report.json").read_text(encoding="utf-8"))
            self.assertTrue(report["complete"])
            self.assertEqual(report["captured_frame_count"], 1)
            self.assertTrue(report["all_reconstruction_input_pose_targets_captured"])
            self.assertTrue(report["all_server_same_pose_reprojections_passed"])
            self.assertIn("excluded from reconstruction", report["scope_guard"])
            self.assertTrue((root / "手机实际最终渲染总览.jpg").is_file())
            self.assertTrue(
                (root / "服务器显示方向同位姿Mesh复投影总览.jpg").is_file()
            )
            self.assertTrue(
                (root / "手机实际与服务器复算逐帧对照总览.jpg").is_file()
            )
            self.assertTrue(
                (root / "重建原始输入与现场严格同位姿图像对照总览.jpg").is_file()
            )
            phone_server._MOBILE_OVERLAY_AUDITS.pop("unit_test_session", None)

    def test_mobile_display_matrix_rotates_without_an_extra_mirror(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "raw.png"
            destination = root / "display.png"
            raw = Image.new("RGB", (40, 30), (0, 0, 0))
            for y in range(raw.height):
                for x in range(raw.width):
                    raw.putpixel(
                        (x, y),
                        (
                            (255, 0, 0)
                            if x < 20 and y < 15
                            else (0, 255, 0)
                            if x >= 20 and y < 15
                            else (0, 0, 255)
                            if x < 20
                            else (255, 255, 0)
                        ),
                    )
            raw.save(source)
            phone_server._write_display_aligned_mobile_image(
                source,
                destination,
                display_matrix=(
                    "0 1 0 0 "
                    "-1 0 1 0 "
                    "0 0 1 0 "
                    "0 0 0 1"
                ),
                display_size=(30, 40),
            )
            with Image.open(destination) as decoded:
                aligned = decoded.convert("RGB")
            self.assertEqual(aligned.getpixel((5, 5)), (0, 0, 255))
            self.assertEqual(aligned.getpixel((24, 5)), (255, 0, 0))
            self.assertEqual(aligned.getpixel((5, 34)), (255, 255, 0))
            self.assertEqual(aligned.getpixel((24, 34)), (0, 255, 0))

    def test_alignment_similarity_unity_internal_roundtrip(self) -> None:
        source = {
            "position_x": 0.03,
            "position_y": -0.04,
            "position_z": 0.05,
            "quaternion_x": 0.0,
            "quaternion_y": 0.0871557427,
            "quaternion_z": 0.0,
            "quaternion_w": 0.9961946981,
            "uniform_scale": 1.04,
        }
        internal = unity_similarity_to_internal(source)
        restored = internal_similarity_to_unity(internal)
        self.assertAlmostEqual(restored["position_x"], source["position_x"], places=7)
        self.assertAlmostEqual(restored["position_y"], source["position_y"], places=7)
        self.assertAlmostEqual(restored["position_z"], source["position_z"], places=7)
        self.assertAlmostEqual(restored["uniform_scale"], source["uniform_scale"], places=7)
        observed = unity_similarity_to_internal(restored)
        np.testing.assert_allclose(observed, internal, atol=1.0e-7)

    def test_alignment_uniform_subset_and_mesh_prompt_are_deterministic(self) -> None:
        self.assertEqual(evenly_spaced_indices(4), [0, 1, 2, 3])
        self.assertEqual(evenly_spaced_indices(16, 8), [0, 2, 4, 6, 9, 11, 13, 15])
        mask = np.zeros((80, 100), dtype=np.uint8)
        mask[20:65, 30:76] = 1
        prompt_a = mask_prompt(mask, 3)
        prompt_b = mask_prompt(mask, 3)
        self.assertEqual(prompt_a, prompt_b)
        self.assertEqual(prompt_a["frame_index"], 3)
        self.assertGreaterEqual(prompt_a["labels"].count(1), 1)
        self.assertTrue(all(len(point) == 2 for point in prompt_a["points"]))

    def test_alignment_uses_current_mesh_centered_spherical_fps16(self) -> None:
        cameras = []
        for angle in np.linspace(0.0, 2.0 * np.pi, 32, endpoint=False):
            center = np.asarray(
                [np.sin(angle), 0.15 * np.sin(2.0 * angle), np.cos(angle)],
                dtype=np.float64,
            )
            camera = np.eye(4, dtype=np.float64)
            camera[:3, 3] = -center
            cameras.append(camera)
        selected, report = object_centered_spherical_farthest_indices(
            cameras,
            np.zeros(3, dtype=np.float64),
        )
        repeated, repeated_report = object_centered_spherical_farthest_indices(
            cameras,
            np.zeros(3, dtype=np.float64),
        )
        self.assertEqual(MIN_OPTIMIZATION_VIEWS, 16)
        self.assertEqual(MAX_OPTIMIZATION_VIEWS, 16)
        self.assertEqual(selected, repeated)
        self.assertEqual(report, repeated_report)
        self.assertEqual(len(selected), 16)
        self.assertEqual(len(set(selected)), 16)
        self.assertEqual(
            report["algorithm"],
            "mean_opposite_seed_then_greedy_spherical_fps_v1",
        )
        self.assertGreaterEqual(
            report["minimum_pairwise_angular_separation_degrees"], 20.0
        )

    def test_spherical_selection_plan_is_single_branch_and_retry_stable(self) -> None:
        names = [f"frame_{index:04d}.jpg" for index in range(12)]
        poses = {
            name: {"pos": np.asarray([index * index * 0.01, 0.0, 0.0])}
            for index, name in enumerate(names)
        }
        selected = [3, 9, 0, 6, 11, 1, 5, 8]
        view_selection = {
            "policy": "pose_mask_training_exact_spherical_farthest_valid_mask",
            "algorithm": "official_training_single_seed_spherical_fps_v1",
            "selected_source_view_indices": selected,
            "selected_source_frame_names": [names[index] for index in selected],
            "quality_gate_used_for_selection": False,
        }
        binding = {
            "o_frozen_before_view_selection": True,
            "view_selection": view_selection,
        }
        plan_a = build_selection_plan(
            names,
            poses,
            runtime_binding=binding,
            client_selected=[1, 2, 3],
        )
        plan_b = build_selection_plan(
            names,
            poses,
            runtime_binding=binding,
            client_selected=[9],
        )
        self.assertEqual(plan_a["sha256"], plan_b["sha256"])
        self.assertTrue(plan_a["client_selection_ignored"])
        self.assertTrue(plan_a["frame_filtering_applied"])
        self.assertEqual(list(plan_a["branches"]), ["training_spherical_farthest8"])
        branch = plan_a["branches"]["training_spherical_farthest8"]
        self.assertEqual(branch["selected_indices"], selected)
        self.assertEqual(branch["selected_frame_names"], [names[i] for i in selected])
        self.assertFalse(branch["audit"]["quality_gate_used_for_selection"])

    def test_all_view_materialization_transposes_rgb_mask_and_intrinsics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "runtime/data/session_a"
            masks = root / "runtime/masks/session_a"
            data.mkdir(parents=True)
            masks.mkdir(parents=True)
            count = 10
            names = []
            for index in range(count):
                name = f"frame_{index:04d}.jpg"
                names.append(name)
                rgb = np.zeros((24, 32, 3), dtype=np.uint8)
                rgb[:, :, 0] = index
                Image.fromarray(rgb, mode="RGB").save(data / name)
                mask = np.zeros((24, 32), dtype=np.uint8)
                mask[4:20, 7:27] = 255
                Image.fromarray(mask, mode="L").save(
                    masks / f"frame_{index:04d}.png"
                )
            _write_phone_pose_file(data / "poses.txt", count)
            session_root = root / "output/reconstructions/session_a"
            session_root.mkdir(parents=True)
            report = materialize_all_view_phone_capture(
                session_id="session_a",
                data_dir=data,
                mask_dir=masks,
                frame_names=names,
                session_root=session_root,
            )
            self.assertTrue(report["passed"])
            self.assertEqual(report["candidate_count"], count)
            self.assertEqual(report["image_axis_contract"], AXIS_CONTRACT)
            self.assertEqual(
                report["axis_records"][0]["pose_coordinate_frame"],
                "unity_capture_anchor_a0",
            )
            self.assertEqual(
                report["axis_records"][0]["capture_anchor_tracking_state"],
                "Tracking",
            )
            raw = json.loads(
                Path(report["raw_cache_report"]).read_text(encoding="utf-8")
            )
            row = raw["objects"][0]
            self.assertEqual(row["input_view_count"], count)
            self.assertEqual(row["selected_source_indices"], [])
            with np.load(row["cache_npz"], allow_pickle=False) as payload:
                self.assertEqual(payload["K"].shape, (count, 3, 3))
                self.assertAlmostEqual(float(payload["K"][0, 0, 0]), 31.0)
                self.assertAlmostEqual(float(payload["K"][0, 1, 1]), 30.0)
                self.assertEqual(payload["source_frame_name"].tolist(), names)
            with Image.open(Path(row["images_dir"]) / "view_0000.png") as image:
                self.assertEqual(image.size, (24, 32))
            reused = materialize_all_view_phone_capture(
                session_id="session_a",
                data_dir=data,
                mask_dir=masks,
                frame_names=names,
                session_root=session_root,
            )
            self.assertEqual(reused["raw_cache_report_sha256"], report["raw_cache_report_sha256"])

    def test_world_and_mobile_export_use_physical_t_o2w(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            mesh_o = root / "mesh_o.obj"
            trimesh.Trimesh(
                vertices=np.asarray(
                    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
                ),
                faces=np.asarray([[0, 1, 2]]),
                process=False,
            ).export(mesh_o)
            frame_report = root / "mesh_result.json"
            frame_payload = {
                "passed": True,
                "mesh": str(mesh_o.resolve()),
                "mesh_sha256": sha256_file(mesh_o),
                "output_frame": "runtime-O",
                **mesh_frame_contract_fields(
                    export_policy="decoded.to_trimesh(transform_pose=False)"
                ),
            }
            frame_report.write_text(json.dumps(frame_payload), encoding="utf-8")
            T_O2W = np.eye(4, dtype=np.float64)
            T_O2W[:3, :3] *= 2.0
            T_O2W[:3, 3] = [3.0, 4.0, 5.0]
            cache = root / "runtime.npz"
            np.savez_compressed(cache, T_O2W=T_O2W)
            runtime_manifest = root / "runtime_manifest.json"
            runtime_manifest.write_text(
                json.dumps(
                    {
                        "passed": True,
                        "objects": [
                            {
                                "category": "fixture",
                                "object_id": "object",
                                "object_key": "fixture:object",
                                "cache_npz": str(cache),
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            report = export_world_and_mobile_meshes(
                mesh_o=mesh_o,
                mesh_frame_report=frame_report,
                runtime_input_manifest=runtime_manifest,
                object_key="fixture:object",
                output_dir=root / "export",
            )
            world = trimesh.load(report["internal_world_obj"], force="mesh", process=False)
            observed = np.asarray(world.vertices)
            expected = np.asarray([[3.0, 4.0, 5.0], [5.0, 4.0, 5.0], [3.0, 6.0, 5.0]])
            self.assertEqual(
                {tuple(np.round(row, 6)) for row in observed},
                {tuple(row) for row in expected},
            )
            mobile = read_mobile_overlay_mesh(
                Path(report["unity_capture_anchor_a0_armesh"])
            )
            self.assertEqual(
                {tuple(np.round(row, 6)) for row in mobile["vertices"]},
                {(3.0, 4.0, -5.0), (5.0, 4.0, -5.0), (3.0, 6.0, -5.0)},
            )
            self.assertEqual(
                report["mobile_overlay"]["coordinate_frame"],
                MOBILE_MESH_COORDINATE_FRAME,
            )
            self.assertEqual(
                report["runtime_W_semantics"], "internal_capture_anchor_a0"
            )
            object_pose = report["object_pose_unity_a0"]
            self.assertEqual(object_pose["parent_frame"], PHONE_POSE_COORDINATE_FRAME)
            self.assertEqual(
                [
                    object_pose["position_x"],
                    object_pose["position_y"],
                    object_pose["position_z"],
                ],
                [3.0, 4.0, -5.0],
            )
            np.testing.assert_allclose(
                object_pose["quaternion_xyzw"], [0.0, 0.0, 0.0, 1.0]
            )
            self.assertAlmostEqual(
                object_pose["runtime_o_scale_meters_per_unit"], 2.0
            )

            phone_root = root / "phone_output"
            session_id = "endpoint_fixture"
            branch_dir = (
                phone_root
                / "reconstructions"
                / session_id
                / "branches"
                / "01_training_spherical_farthest8"
            )
            branch_dir.mkdir(parents=True)
            endpoint_mesh = branch_dir / "fixture.armesh"
            endpoint_mesh.write_bytes(
                Path(report["unity_capture_anchor_a0_armesh"]).read_bytes()
            )
            (branch_dir / "branch_report.json").write_text(
                json.dumps(
                    {
                        "format": phone_server.BRANCH_FORMAT,
                        "passed": True,
                        "mesh": {
                            "unity_capture_anchor_a0_armesh": str(endpoint_mesh),
                            "mobile_overlay": report["mobile_overlay"],
                        },
                    }
                ),
                encoding="utf-8",
            )
            previous_config = phone_server._CONFIG
            try:
                phone_server._CONFIG = phone_server.ServerConfig(
                    output_root=phone_root,
                    python=Path(sys.executable),
                    gpu="0",
                )
                with phone_server.transport_server.legacy.app.test_request_context(
                    f"/manual_reconstruction_mesh/{session_id}"
                ):
                    response = phone_server.manual_reconstruction_mesh(session_id)
                    response.direct_passthrough = False
                    self.assertEqual(response.status_code, 200)
                    self.assertEqual(response.get_data(), endpoint_mesh.read_bytes())
                    self.assertEqual(
                        response.headers["X-AR-Mesh-SHA256"],
                        sha256_file(endpoint_mesh),
                    )
                    self.assertEqual(
                        response.headers["X-AR-Coordinate-Frame"],
                        MOBILE_MESH_COORDINATE_FRAME,
                    )
            finally:
                phone_server._CONFIG = previous_config

    def test_object_pose_conjugates_internal_rotation_into_unity_a0(self) -> None:
        angle = np.deg2rad(90.0)
        rotation = np.asarray(
            [
                [np.cos(angle), -np.sin(angle), 0.0],
                [np.sin(angle), np.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = rotation * 0.7
        transform[:3, 3] = [0.2, -0.4, 1.3]
        object_pose = unity_object_pose_from_t_o2a0(transform)
        self.assertEqual(object_pose["parent_frame"], PHONE_POSE_COORDINATE_FRAME)
        self.assertEqual(
            [
                object_pose["position_x"],
                object_pose["position_y"],
                object_pose["position_z"],
            ],
            [0.2, -0.4, -1.3],
        )
        self.assertAlmostEqual(
            object_pose["runtime_o_scale_meters_per_unit"], 0.7
        )
        self.assertAlmostEqual(
            np.linalg.norm(object_pose["quaternion_xyzw"]), 1.0
        )

    def test_calibrated_contour_is_returned_to_uploaded_orientation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data"
            data.mkdir()
            raw = np.zeros((3, 5, 3), dtype=np.uint8)
            raw[:, :, 1] = 17
            Image.fromarray(raw, mode="RGB").save(data / "frame_0000.png")
            calibrated = Image.fromarray(raw, mode="RGB").transpose(
                Image.Transpose.TRANSPOSE
            )
            calibrated_array = np.asarray(calibrated).copy()
            calibrated_array[1, 2] = [0, 255, 255]
            calibrated_overlay = root / "calibrated.png"
            Image.fromarray(calibrated_array, mode="RGB").save(calibrated_overlay)
            calibrated_report = root / "calibrated_report.json"
            calibrated_report.write_text(
                json.dumps(
                    {
                        "passed": True,
                        "projection_formula": "Mesh_O -> T_O2W -> Mesh_W -> T_W2C -> K_raw",
                        "views": [{"overlay": str(calibrated_overlay)}],
                    }
                ),
                encoding="utf-8",
            )
            report = restore_phone_orientation_contours(
                calibrated_contour_report=calibrated_report,
                selected_source_names=["frame_0000.png"],
                data_dir=data,
                output_dir=root / "phone_contours",
            )
            with Image.open(report["views"][0]["phone_orientation_overlay"]) as image:
                self.assertEqual(image.size, (5, 3))
                self.assertEqual(tuple(np.asarray(image)[2, 1]), (0, 255, 255))

    def test_unity_phase_one_a0_capture_and_display_contract(self) -> None:
        tracker_root = Path(__file__).resolve().parents[2]
        source = (tracker_root / "ARposeTTracker.cs").read_text(encoding="utf-8")
        server_source = (tracker_root / "manual_mesh_reconstruction/server.py").read_text(
            encoding="utf-8"
        )
        begin = source[
            source.index("IEnumerator BeginRecordingWithFreshCaptureGeneration()") :
            source.index("// ========== 更新：全局取消与重置 ==========")
        ]
        toggle = source[
            source.index("void ToggleRecording()") :
            source.index("IEnumerator BeginRecordingWithFreshCaptureGeneration()")
        ]
        tracking = source[
            source.index("void UpdateCaptureReferenceAnchorTracking()") :
            source.index("void ClearCaptureReferenceAnchor()")
        ]
        cancel = source[
            source.index("void CancelReview()") : source.index("void Update()")
        ]
        self.assertIn("bool anchorCreated = CreateCaptureReferenceAnchor();", begin)
        self.assertNotIn("CreateCaptureReferenceAnchor();", toggle)
        self.assertIn("TrackableId previousA0Id", begin)
        self.assertNotIn("ObjectCenter", source)
        self.assertNotIn("objectCenter", source)
        self.assertIn("yield return new WaitForEndOfFrame();", begin)
        self.assertIn("AnchorTrackableStillRegistered(previousA0Id)", begin)
        self.assertIn("cameraFrameSequence > preTransitionCameraFrameSequence", begin)
        self.assertIn("lifecycleGeneration != arSessionLifecycleGeneration", begin)
        self.assertNotIn("arSession.Reset();", begin)
        self.assertIn("ResetARSessionAfterExplicitCancel", cancel)
        self.assertIn("arSession.Reset();", cancel)
        self.assertIn('"session_reset=false "', begin)
        self.assertLess(
            begin.index("ClearCaptureReferenceAnchor();"),
            begin.index("bool anchorCreated = CreateCaptureReferenceAnchor();"),
        )
        self.assertIn("a0State == TrackingState.Tracking", tracking)
        self.assertIn("captureReferenceAnchorTrackingStable", tracking)
        self.assertIn("camera_frame_received_anchor_a0_relative_v1", source)
        self.assertIn("Vector3 requestedPosition = cameraFramePosition", source)
        self.assertIn("Quaternion requestedRotation = Quaternion.identity", source)
        self.assertIn("TryFinalizeMeshUnderCaptureAnchor", source)
        self.assertIn('serverURL + "/prepare_runtime_o"', source)
        self.assertIn(
            "reconstructedMeshRoot.transform.SetParent(\n"
            "            captureReferenceAnchorObject.transform",
            source,
        )
        self.assertIn("localPosition = Vector3.zero", source)
        self.assertIn("localRotation = Quaternion.identity", source)
        self.assertIn('mobileAR.placement != "capture_anchor_a0_direct"', source)
        self.assertIn('"placement": MOBILE_MESH_PLACEMENT', server_source)
        self.assertIn(
            'MOBILE_MESH_COORDINATE_FRAME = PHONE_POSE_COORDINATE_FRAME',
            server_source,
        )
        self.assertIn(
            "unity_native_screen_display_aligned_raw_rgb_strict_input_pose_v3",
            source,
        )
        self.assertIn("strict_reconstruction_input_pose_matching = true", source)
        self.assertIn("composite.EncodeToPNG()", source)
        self.assertIn('form.AddField("target_index"', source)
        self.assertIn("_write_display_aligned_mobile_image", server_source)
        self.assertIn(
            "mobile diagnostic cannot complete without all reconstruction input poses",
            server_source,
        )
        self.assertNotIn("object_anchor", server_source)
        self.assertNotIn("A1", source)
        self.assertIn("ARPrimaryRecordSafeAreaDock", source)
        self.assertIn("ConfigureUnifiedReviewButtons", source)
        self.assertIn("primaryRecordButtonHeight", source)
        self.assertIn("Screen.sleepTimeout = SleepTimeout.NeverSleep", source)
        self.assertNotIn("CaptureReferenceWorldPoseFallback", source)
        self.assertNotIn("RecoveredCaptureReferenceWorldPoseFallback", source)
        self.assertIn("ToggleReconstructedOutlineMethod", source)
        self.assertIn("ServerStyleScreenSpace", source)

        mask_shader = (
            tracker_root
            / "manual_mesh_reconstruction/unity/ARMeshSilhouetteMask.shader"
        ).read_text(encoding="utf-8")
        edge_shader = (
            tracker_root
            / "manual_mesh_reconstruction/unity/ARMeshScreenSpaceOutline.shader"
        ).read_text(encoding="utf-8")
        self.assertIn('Shader "Tracker/ARMeshSilhouetteMask"', mask_shader)
        self.assertIn('Shader "Tracker/ARMeshScreenSpaceOutline"', edge_shader)
        self.assertIn("neighbor - center", edge_shader)


if __name__ == "__main__":
    unittest.main()
