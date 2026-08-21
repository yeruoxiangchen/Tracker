#!/usr/bin/env python3

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import trimesh

import pose_point_depth_mv.ar_object_reconstruction as reconstruction
import pose_point_depth_mv.infer_omni_real_native_ss_stock_slat_no_vggt as stock_infer
import pose_point_depth_mv.server as collection_server
from pose_point_depth_mv.ar_object_reconstruction import (
    ARReconstructionConfig,
    ReconstructionPaths,
    reconstruction_commands,
    run_ar_object_frame_selection,
    run_ar_object_reconstruction,
)


class ARObjectReconstructionTest(unittest.TestCase):
    def test_pose_mask_deployment_requires_final_selected_view_quality(self) -> None:
        good = {
            "objects": [
                {
                    "input_quality": {
                        "formal_input_passed": True,
                        "checks": {"ray_residual_median": True},
                    }
                }
            ]
        }
        reconstruction._validate_deployment_runtime_quality(
            good, geometry_mode="pose_mask", report_path=Path("runtime.json")
        )
        with self.assertRaisesRegex(RuntimeError, "final selected views failed"):
            reconstruction._validate_deployment_runtime_quality(
                {
                    "objects": [
                        {
                            "input_quality": {
                                "formal_input_passed": False,
                                "checks": {"ray_residual_median": False},
                                "values": {"ray": 0.3},
                            }
                        }
                    ]
                },
                geometry_mode="pose_mask",
                report_path=Path("runtime.json"),
            )

    def _config(self, root: Path) -> ARReconstructionConfig:
        files = {}
        for name in (
            "python",
            "ss.pt",
            "slat.pt",
            "ss_contract.json",
            "slat_contract.json",
            "stock.json",
        ):
            path = root / name
            path.write_text("test", encoding="utf-8")
            files[name] = path
        return ARReconstructionConfig(
            python=files["python"],
            gpu="6",
            native_ss_checkpoint=files["ss.pt"],
            native_slat_checkpoint=files["slat.pt"],
            ss_migration_contract=files["ss_contract.json"],
            slat_migration_contract=files["slat_contract.json"],
            stock_slat_freeze=files["stock.json"],
        )

    def test_commands_are_the_frozen_five_stage_no_vggt_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self._config(root)
            paths = ReconstructionPaths.build(root / "output", "session_1")
            commands = reconstruction_commands(root / "dataset", paths, config)
            self.assertEqual(
                [name for name, _command, _report in commands],
                [
                    "raw_cache",
                    "runtime_o",
                    "dino_only_input",
                    "no_vggt_ss_slat_mesh",
                    "world_mesh_bundle",
                ],
            )
            inference = commands[3][1]
            runtime = commands[1][1]
            self.assertIn("pose_point_depth_mv.infer_omni_real_native_no_vggt_mixed", inference)
            self.assertIn(str(config.native_ss_checkpoint), inference)
            self.assertIn(str(config.native_slat_checkpoint), inference)
            self.assertNotIn("vggt", " ".join(commands[2][1]).lower())
            gravity_index = runtime.index("--gravity_up_w")
            self.assertEqual(runtime[gravity_index + 1 : gravity_index + 4], ["0", "1", "0"])
            policy_index = runtime.index("--view_selection_policy")
            self.assertEqual(
                runtime[policy_index + 1], "object_spherical_farthest_valid_mask"
            )
            self.assertIn("--allow_empty_points", commands[0][1])
            geometry_index = runtime.index("--geometry_mode")
            self.assertEqual(runtime[geometry_index + 1], "pose_mask")

    def test_stock_slat_backend_uses_native_ss_without_native_slat_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base = self._config(root)
            base.native_slat_checkpoint.unlink()
            base.slat_migration_contract.unlink()
            config = ARReconstructionConfig(
                **{
                    **base.__dict__,
                    "slat_backend": "stock",
                }
            )
            config.validate()
            paths = ReconstructionPaths.build(root / "output", "session_stock")
            commands = reconstruction_commands(root / "dataset", paths, config)
            self.assertEqual(
                commands[3][0], "no_vggt_native_ss_stock_slat_mesh"
            )
            inference = commands[3][1]
            self.assertIn(
                "pose_point_depth_mv.infer_omni_real_native_ss_stock_slat_no_vggt",
                inference,
            )
            self.assertIn(str(config.native_ss_checkpoint), inference)
            self.assertIn(str(config.stock_slat_freeze), inference)
            self.assertNotIn("--native_slat_checkpoint", inference)
            self.assertNotIn("--slat_migration_contract", inference)
            cfg_index = inference.index("--native_ss_cfg_strength")
            self.assertEqual(inference[cfg_index + 1], "3.0")

    def test_stock_slat_sampling_contract_is_frozen(self) -> None:
        params = stock_infer.stock_sampling_params(
            {
                "steps": 25,
                "cfg_strength": 5.0,
                "cfg_interval": (0.5, 1.0),
                "rescale_t": 3.0,
            }
        )
        self.assertEqual(params["guidance_rescale"], 0.0)
        with self.assertRaisesRegex(RuntimeError, "sampler changed"):
            stock_infer.stock_sampling_params(
                {
                    **params,
                    "cfg_strength": 3.0,
                }
            )

    def test_colmap_world_can_disable_gravity_and_freeze_lexical_views(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = ARReconstructionConfig(
                **{
                    **self._config(root).__dict__,
                    "view_selection_policy": "lexical_even",
                    "gravity_up_w": None,
                }
            )
            paths = ReconstructionPaths.build(root / "output", "session_colmap")
            runtime = reconstruction_commands(root / "dataset", paths, config)[1][1]
            self.assertNotIn("--gravity_up_w", runtime)
            policy_index = runtime.index("--view_selection_policy")
            self.assertEqual(runtime[policy_index + 1], "lexical_even")

    def test_raw_adapter_can_freeze_exact_source_frame_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            frames = tuple(f"frame_{index:04d}.jpg" for index in range(8))
            config = ARReconstructionConfig(
                **{
                    **self._config(root).__dict__,
                    "source_frame_names": frames,
                }
            )
            paths = ReconstructionPaths.build(root / "output", "session_frames")
            raw = reconstruction_commands(root / "dataset", paths, config)[0][1]
            selected = [
                raw[index + 1]
                for index, value in enumerate(raw[:-1])
                if value == "--frame_name"
            ]
            self.assertEqual(selected, list(frames))

    def test_collection_server_has_complete_direct_launch_defaults(self) -> None:
        args = collection_server.build_parser().parse_args([])
        self.assertEqual(args.output_root, collection_server.DEFAULT_OUTPUT_ROOT)
        self.assertEqual(args.gpu, collection_server.DEFAULT_GPU)
        self.assertEqual(args.deployment, "official")
        self.assertEqual(args.expected_slat_step, 25000)
        self.assertEqual(
            args.native_ss_report,
            collection_server.DEFAULT_OFFICIAL_SS_REPORT,
        )
        self.assertEqual(
            args.cross_deployment_bridge_report,
            collection_server.DEFAULT_OFFICIAL_BRIDGE_REPORT,
        )
        self.assertEqual(args.host, "0.0.0.0")
        self.assertEqual(args.port, 5000)
        self.assertEqual(args.preview_count, 6)
        self.assertEqual(args.preview_resolution, 512)
        self.assertEqual(args.geometry_mode, "pose_mask")

    def test_collection_server_default_builds_official_ss2k_slat25k(self) -> None:
        args = collection_server.build_parser().parse_args([])
        config = collection_server.build_reconstruction_config(args)
        self.assertIsInstance(
            config,
            collection_server.OfficialReconstructionConfig,
        )
        self.assertEqual(config.native_ss_report, collection_server.DEFAULT_OFFICIAL_SS_REPORT)
        self.assertEqual(
            config.native_slat_checkpoint,
            collection_server.DEFAULT_OFFICIAL_SLAT_CHECKPOINT,
        )
        self.assertEqual(config.expected_slat_step, 25000)
        self.assertEqual(
            config.cross_deployment_bridge_report,
            collection_server.DEFAULT_OFFICIAL_BRIDGE_REPORT,
        )
        self.assertEqual(config.geometry_mode, "pose_mask")
        self.assertEqual(config.view_selection_policy, "object_spherical_farthest_valid_mask")
        self.assertEqual(config.gravity_up_w, (0.0, 1.0, 0.0))

    def test_collection_server_capture_only_skips_every_model(self) -> None:
        args = collection_server.build_parser().parse_args(["--capture_only"])
        self.assertIsNone(collection_server.build_reconstruction_config(args))

    def test_collection_server_official_runner_is_used(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = collection_server.OfficialReconstructionConfig(
                python=root / "python",
                gpu="0",
                geometry_mode="pose_mask",
                view_selection_policy="object_spherical_farthest_valid_mask",
                selected_view_count=8,
                min_object_points=100,
                min_mask_observations=2,
                min_mask_support_ratio=0.35,
                gravity_up_w=(0.0, 1.0, 0.0),
                native_ss_report=root / "ss.json",
                native_slat_checkpoint=root / "slat.pt",
                expected_slat_step=25000,
                cross_deployment_bridge_report=root / "bridge.json",
                stock_slat_freeze=root / "freeze.json",
                seed=42,
                amp_dtype="bf16",
                preview_render_frames=24,
                preview_count=6,
                preview_resolution=512,
                diagnostic_bypass_pose_mask_quality=False,
            )
            result = {
                "run_dir": str(root / "run"),
                "meshes": {"world_glb": "world.glb"},
                "previews": {"contact_sheet": "sheet.png"},
            }
            with (
                mock.patch.object(collection_server, "_RECONSTRUCTION_CONFIG", config),
                mock.patch.object(
                    collection_server,
                    "run_official_reconstruction",
                    return_value=result,
                ) as official_run,
            ):
                actual = collection_server._run_configured_reconstruction(
                    session_id="phone-session",
                    dataset_dir=root / "dataset",
                    output_root=root / "output",
                )
            self.assertIs(actual, result)
            official_run.assert_called_once_with(
                session_id="phone-session",
                dataset_dir=root / "dataset",
                output_root=root / "output",
                config=config,
            )

    def test_server_serves_only_session_scoped_mobile_mesh(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            mesh = (
                output
                / "reconstructions"
                / "session_1"
                / "final"
                / "reconstructed_object_world.armesh"
            )
            mesh.parent.mkdir(parents=True)
            mesh.write_bytes(b"mobile-mesh")
            with (
                mock.patch.object(collection_server, "_OUTPUT_ROOT", output),
                collection_server.legacy.app.test_request_context(),
            ):
                response = collection_server.reconstruction_mesh("session_1")
                self.assertEqual(response.status_code, 200)
                response.direct_passthrough = False
                self.assertEqual(response.get_data(), b"mobile-mesh")
                self.assertEqual(
                    response.headers["X-AR-Coordinate-Frame"], "unity_world"
                )
                invalid, status = collection_server.reconstruction_mesh("../outside")
                self.assertEqual(status, 400)
                self.assertEqual(invalid.get_json()["status"], "error")

    def test_phone_quality_error_is_concise_but_server_keeps_detail(self) -> None:
        detail = (
            "pose-mask runtime final selected views failed input quality: "
            "checks=['ray_residual_median'] values={'ray': 0.2025} "
            "report=/very/long/server/path/runtime_input_manifest.json"
        )
        message = collection_server._phone_error_message(RuntimeError(detail))
        self.assertIn("最终8帧质量检查未通过", message)
        self.assertIn("ray_residual_median", message)
        self.assertNotIn("/very/long/server/path", message)
        self.assertLess(len(message), 100)

    def test_server_keeps_all_uploaded_frames_for_current_reconstruction(self) -> None:
        names = [f"frame_{index:04d}.jpg" for index in range(30)]
        with (
            mock.patch.object(collection_server.legacy, "_load_current_session"),
            mock.patch.object(
                collection_server.legacy,
                "_list_session_images",
                return_value=names,
            ),
        ):
            client, candidates, frame_names, record = collection_server._selected_frames(
                {"selected": list(range(18))}
            )
        self.assertEqual(client, list(range(18)))
        self.assertEqual(candidates, list(range(30)))
        self.assertEqual(frame_names, names)
        self.assertNotIn("legacy_qc_selected_indices", record)
        self.assertEqual(record["reconstruction_candidate_count"], 30)
        self.assertEqual(
            record["authoritative_selection_stage"],
            "runtime_o_segmented_all_candidates_to_final8",
        )
        self.assertIn("legacy 18-frame QC sampler", record["removed_redundant_gates"])

    def test_subprocess_environment_prepends_matching_python_bin(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self._config(root)
            with mock.patch.dict(os.environ, {"PATH": "/usr/local/bin:/usr/bin"}):
                environment = reconstruction._subprocess_environment(config)
            self.assertEqual(
                environment["PATH"].split(os.pathsep)[0],
                str(config.python.resolve().parent),
            )
            self.assertIn("/usr/local/bin:/usr/bin", environment["PATH"])
            self.assertEqual(environment["CUDA_VISIBLE_DEVICES"], "6")

    def test_orchestrator_materializes_world_obj_glb_and_progress(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset"
            dataset.mkdir()
            output = root / "output"
            config = self._config(root)
            source_runtime = root / "runtime.obj"
            source_world = root / "world.obj"
            trimesh.creation.box().export(source_runtime)
            world_mesh = trimesh.creation.box()
            world_mesh.apply_translation([1.0, 2.0, 3.0])
            world_mesh.export(source_world)
            seen_progress = []

            def fake_stage(**kwargs):
                expected = Path(kwargs["expected_report"])
                expected.parent.mkdir(parents=True, exist_ok=True)
                payload = {"passed": True}
                if kwargs["name"] == "runtime_o":
                    payload["objects"] = [
                        {
                            "input_quality": {
                                "formal_input_passed": True,
                                "checks": {"ray_residual_median": True},
                            }
                        }
                    ]
                elif kwargs["name"] == "world_mesh_bundle":
                    payload["cases"] = [
                        {
                            "predicted_runtime_o": {"path": str(source_runtime)},
                            "predicted_sparse_world": {"path": str(source_world)},
                            }
                        ]
                elif kwargs["name"] == "mesh_previews":
                    images = []
                    for index in range(config.preview_count):
                        image = expected.parent / f"mesh_view_{index:02d}.png"
                        image.write_bytes(f"preview-{index}".encode("ascii"))
                        images.append({"path": str(image)})
                    sheet = expected.parent / "mesh_views_contact_sheet.png"
                    sheet.write_bytes(b"sheet")
                    payload.update(
                        {
                            "images": images,
                            "contact_sheet": {"path": str(sheet)},
                            "render_config": {
                                "render_frames": config.preview_render_frames,
                                "preview_count": config.preview_count,
                                "resolution": config.preview_resolution,
                            },
                        }
                    )
                expected.write_text(json.dumps(payload), encoding="utf-8")
                return payload

            with mock.patch.object(reconstruction, "_run_stage", side_effect=fake_stage) as run:
                result = run_ar_object_reconstruction(
                    session_id="session_1",
                    dataset_dir=dataset,
                    output_root=output,
                    config=config,
                    progress_callback=seen_progress.append,
                )
            self.assertEqual(run.call_count, 6)
            self.assertTrue(result["passed"])
            self.assertFalse(result["vggt_model_loaded"])
            self.assertFalse(result["vggt_model_executed"])
            self.assertTrue(Path(result["meshes"]["runtime_o_obj"]).is_file())
            self.assertTrue(Path(result["meshes"]["world_obj"]).is_file())
            self.assertTrue(Path(result["meshes"]["world_glb"]).is_file())
            self.assertTrue(Path(result["meshes"]["mobile_overlay"]).is_file())
            self.assertTrue(Path(result["meshes"]["mobile_overlay_report"]).is_file())
            self.assertEqual(len(result["previews"]["images"]), 6)
            self.assertTrue(Path(result["previews"]["contact_sheet"]).is_file())
            self.assertEqual(seen_progress[-1]["status"], "complete")
            self.assertTrue((output / "latest_reconstruction.json").is_file())

            # A repeated phone submission must reuse the completed result.
            with mock.patch.object(reconstruction, "_run_stage") as rerun:
                reused = run_ar_object_reconstruction(
                    session_id="session_1",
                    dataset_dir=dataset,
                    output_root=output,
                    config=config,
                )
            rerun.assert_not_called()
            self.assertEqual(reused["meshes"], result["meshes"])

    def test_selection_only_runs_exactly_raw_and_runtime_stages(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset"
            dataset.mkdir()
            candidate_names = [f"frame_{index:04d}.jpg" for index in range(64)]
            (dataset / "capture_report.json").write_text(
                json.dumps(
                    {
                        "passed": True,
                        "selected_frame_names": candidate_names,
                    }
                ),
                encoding="utf-8",
            )
            config = self._config(root)

            def fake_stage(**kwargs):
                if kwargs["name"] == "runtime_o":
                    return {
                        "passed": True,
                        "objects": [
                            {
                                "geometry_mode": "pose_mask",
                                "point_cloud_consumed": False,
                                "selected_view_count": 8,
                                "selected_source_view_indices": list(range(0, 64, 8)),
                                "selected_frame_names": candidate_names[::8],
                                "view_selection": {
                                    "policy": "pose_mask_segmented_object_spherical_farthest_valid_mask"
                                },
                                "input_quality": {"formal_input_passed": True},
                                "prepared_rgb_paths": [f"rgb_{i}.png" for i in range(8)],
                                "prepared_mask_paths": [f"mask_{i}.png" for i in range(8)],
                            }
                        ],
                    }
                return {"passed": True}

            with mock.patch.object(
                reconstruction, "_run_stage", side_effect=fake_stage
            ) as run:
                result = run_ar_object_frame_selection(
                    session_id="session_select",
                    dataset_dir=dataset,
                    output_root=root / "output",
                    config=config,
                )
            self.assertEqual(run.call_count, 2)
            self.assertEqual(result["candidate_frame_count"], 64)
            self.assertEqual(result["selected_view_count"], 8)
            self.assertFalse(result["point_cloud_consumed"])
            self.assertTrue(result["stopped_before_model_inference"])


if __name__ == "__main__":
    unittest.main()
