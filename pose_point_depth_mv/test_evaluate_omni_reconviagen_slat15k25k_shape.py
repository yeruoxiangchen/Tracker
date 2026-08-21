#!/usr/bin/env python3

from __future__ import annotations

import tempfile
import json
from pathlib import Path
from unittest import mock
import unittest

import numpy as np
import trimesh

from pose_point_depth_mv.evaluate_omni_reconviagen_slat15k25k_shape import (
    OFFICIAL_MANIFEST_FORMAT,
    OFFICIAL_RECORD_METHOD,
    RECON_MANIFEST_FORMAT,
    RECON_RECORD_METHOD,
    _official_artifact_identity,
    _validate_official_manifest,
    _validate_recon_manifest,
    main,
    numeric_summary,
)
from pose_point_depth_mv.dataset_tools.prepare_omni_real_label_cache import (
    MANIFEST_FORMAT as LABEL_MANIFEST_FORMAT,
)
from pose_point_depth_mv.evaluate_holdout64_current_vs_reconviagen_shape import (
    normalize_mesh_bbox,
)
from pose_point_depth_mv.omni_real_benchmark_common import sha256_file


class OmniOfficialShapeTest(unittest.TestCase):
    def test_bbox_normalization_is_independent_of_translation_and_scale(self) -> None:
        left = trimesh.creation.box(extents=(1.0, 2.0, 4.0))
        right = left.copy()
        right.apply_scale(7.0)
        right.apply_translation((10.0, -3.0, 8.0))
        left_n, _ = normalize_mesh_bbox(left)
        right_n, _ = normalize_mesh_bbox(right)
        self.assertTrue(
            np.allclose(
                np.asarray(left_n.vertices), np.asarray(right_n.vertices), atol=1.0e-7
            )
        )

    def test_recon_manifest_requires_exact_seed_and_coverage(self) -> None:
        manifest = {
            "format": RECON_MANIFEST_FORMAT,
            "method": RECON_RECORD_METHOD,
            "seeds": [42],
            "object_count": 1,
            "record_count": 1,
            "passed": True,
            "objects": [
                {
                    "object_key": "toy:one",
                    "method": RECON_RECORD_METHOD,
                    "seed": 42,
                    "explicit_runtime_pose_condition": False,
                    "passed": True,
                }
            ],
        }
        rows = _validate_recon_manifest(manifest, expected_objects=1, seed=42)
        self.assertEqual(set(rows), {"toy:one"})
        manifest["seeds"] = [43]
        with self.assertRaises(RuntimeError):
            _validate_recon_manifest(manifest, expected_objects=1, seed=42)

    def test_official_manifest_binds_step_and_no_vggt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "checkpoint.pt"
            bridge = root / "bridge.json"
            stock = root / "stock.json"
            ss_report = root / "ss_report.json"
            ss_checkpoint = root / "ss_checkpoint.pt"
            checkpoint.write_bytes(b"checkpoint")
            bridge.write_bytes(b"bridge")
            stock.write_bytes(b"stock")
            ss_report.write_bytes(b"ss-report")
            ss_checkpoint.write_bytes(b"ss-checkpoint")
            manifest = {
                "format": OFFICIAL_MANIFEST_FORMAT,
                "method": OFFICIAL_RECORD_METHOD,
                "seeds": [42],
                "object_count": 1,
                "record_count": 1,
                "output_frame": "runtime-O",
                "vggt_model_loaded": False,
                "vggt_model_executed": False,
                "passed": True,
                "native_slat_deployment": {
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": sha256_file(checkpoint),
                    "checkpoint_step": 15000,
                    "weights": "ema",
                },
                "cross_deployment_bridge": {
                    "path": str(bridge),
                    "sha256": sha256_file(bridge),
                    "passed": True,
                },
                "native_ss_deployment": {
                    "report": str(ss_report),
                    "report_sha256": sha256_file(ss_report),
                    "checkpoint": str(ss_checkpoint),
                    "checkpoint_sha256": sha256_file(ss_checkpoint),
                    "checkpoint_step": 2000,
                    "weights": "ema",
                    "cfg_strength": 5.0,
                    "steps": 25,
                    "cfg_interval": [0.5, 1.0],
                    "guidance_rescale": 0.0,
                    "rescale_t": 3.0,
                    "amp_dtype": "bf16",
                    "false_checks": [],
                },
                "stock_slat_freeze": str(stock),
                "stock_slat_freeze_sha256": sha256_file(stock),
                "objects": [
                    {
                        "object_key": "toy:one",
                        "method": OFFICIAL_RECORD_METHOD,
                        "seed": 42,
                        "native_slat_checkpoint_step": 15000,
                        "native_slat_weights": "ema",
                        "output_frame": "runtime-O",
                        "vggt_model_loaded": False,
                        "vggt_model_executed": False,
                        "target_or_metric_consumed": False,
                        "formal_claim_allowed": False,
                        "passed": True,
                    }
                ],
            }
            rows = _validate_official_manifest(
                manifest, expected_objects=1, seed=42, expected_step=15000
            )
            self.assertEqual(set(rows), {"toy:one"})
            manifest["objects"][0]["vggt_model_executed"] = True
            with self.assertRaises(RuntimeError):
                _validate_official_manifest(
                    manifest, expected_objects=1, seed=42, expected_step=15000
                )

    def test_shared_identity_excludes_only_slat_checkpoint(self) -> None:
        manifest = {
            "model_input_manifest_sha256": "inputs",
            "native_ss_deployment": {
                "report_sha256": "report",
                "checkpoint_sha256": "ss",
                "weights": "ema",
                "cfg_strength": 5.0,
                "steps": 25,
                "cfg_interval": [0.5, 1.0],
                "guidance_rescale": 0.0,
                "rescale_t": 3.0,
            },
            "stock_slat_freeze_sha256": "stock",
            "native_slat_deployment": {"checkpoint_sha256": "ignored-a"},
        }
        other = dict(manifest)
        other["native_slat_deployment"] = {"checkpoint_sha256": "ignored-b"}
        self.assertEqual(
            _official_artifact_identity(manifest), _official_artifact_identity(other)
        )

    def test_numeric_summary_is_object_bootstrap(self) -> None:
        summary = numeric_summary([1.0, -1.0, 2.0], bootstrap_samples=100, seed=7)
        self.assertEqual(summary["count"], 3)
        self.assertAlmostEqual(summary["positive_rate"], 2.0 / 3.0)
        self.assertEqual(len(summary["bootstrap_mean_95_ci"]), 2)

    def test_one_object_end_to_end_without_sim3(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)

            def write_json(path: Path, value: object) -> None:
                path.write_text(json.dumps(value), encoding="utf-8")

            runtime = root / "runtime.json"
            model_input = root / "model_input.json"
            write_json(
                runtime,
                {
                    "format": "pose_point_depth_mv.omni_real_runtime_input_manifest.v2",
                    "passed": True,
                    "build_config": {
                        "input_frontend_format": "pose_point_depth_mv.real_input_frontend.v2",
                        "frame_config": {"min_object_points": 1},
                    },
                    "objects": [
                        {
                            "object_key": "toy:one",
                            "category": "toy",
                            "object_id": "one",
                            "runtime_frame_stats": {
                                "support": {"mask_supported_point_count": 1}
                            },
                        }
                    ],
                },
            )
            write_json(
                model_input,
                {
                    "format": "pose_point_depth_mv.omni_real_dino_only_model_input_manifest.v1",
                    "passed": True,
                    "vggt_model_loaded": False,
                    "vggt_model_executed": False,
                    "runtime_input_manifest_sha256": sha256_file(runtime),
                    "objects": [
                        {
                            "object_key": "toy:one",
                            "category": "toy",
                            "object_id": "one",
                            "target_or_mesh_consumed": False,
                        }
                    ],
                },
            )
            target = root / "target.obj"
            recon_mesh = root / "recon.obj"
            slat15_mesh = root / "slat15.obj"
            slat25_mesh = root / "slat25.obj"
            base = trimesh.creation.icosphere(subdivisions=1, radius=1.0)
            base.export(target)
            base.export(recon_mesh)
            base.export(slat15_mesh)
            base.export(slat25_mesh)
            checkpoint15 = root / "step15.pt"
            checkpoint25 = root / "step25.pt"
            bridge15 = root / "bridge15.json"
            bridge25 = root / "bridge25.json"
            stock = root / "stock.json"
            ss_report = root / "ss_report.json"
            ss_checkpoint = root / "ss_checkpoint.pt"
            for path in (
                checkpoint15,
                checkpoint25,
                bridge15,
                bridge25,
                stock,
                ss_report,
                ss_checkpoint,
            ):
                path.write_bytes(path.name.encode("utf-8"))

            label_manifest = root / "labels.json"
            write_json(
                label_manifest,
                {
                    "format": LABEL_MANIFEST_FORMAT,
                    "passed": True,
                    "selected_object_count": 1,
                    "completed_object_count": 1,
                    "runtime_input_manifest": str(runtime),
                    "runtime_input_manifest_sha256": sha256_file(runtime),
                    "objects": [
                        {
                            "object_key": "toy:one",
                            "category": "toy",
                            "object_id": "one",
                            "mesh_o": str(target),
                            "mesh_o_sha256": sha256_file(target),
                        }
                    ],
                },
            )
            recon_manifest = root / "recon_manifest.json"
            write_json(
                recon_manifest,
                {
                    "format": RECON_MANIFEST_FORMAT,
                    "method": RECON_RECORD_METHOD,
                    "seeds": [42],
                    "object_count": 1,
                    "record_count": 1,
                    "passed": True,
                    "runtime_input_manifest": str(runtime),
                    "runtime_input_manifest_sha256": sha256_file(runtime),
                    "objects": [
                        {
                            "object_key": "toy:one",
                            "method": RECON_RECORD_METHOD,
                            "seed": 42,
                            "explicit_runtime_pose_condition": False,
                            "mesh": str(recon_mesh),
                            "mesh_sha256": sha256_file(recon_mesh),
                            "passed": True,
                        }
                    ],
                },
            )

            shared_ss = {
                "report": str(ss_report),
                "report_sha256": sha256_file(ss_report),
                "checkpoint": str(ss_checkpoint),
                "checkpoint_sha256": sha256_file(ss_checkpoint),
                "checkpoint_step": 2000,
                "weights": "ema",
                "cfg_strength": 5.0,
                "steps": 25,
                "cfg_interval": [0.5, 1.0],
                "guidance_rescale": 0.0,
                "rescale_t": 3.0,
                "amp_dtype": "bf16",
                "false_checks": [],
            }

            def official_manifest(
                *, step: int, checkpoint: Path, bridge: Path, mesh: Path
            ) -> dict[str, object]:
                return {
                    "format": OFFICIAL_MANIFEST_FORMAT,
                    "method": OFFICIAL_RECORD_METHOD,
                    "seeds": [42],
                    "object_count": 1,
                    "record_count": 1,
                    "output_frame": "runtime-O",
                    "vggt_model_loaded": False,
                    "vggt_model_executed": False,
                    "passed": True,
                    "runtime_input_manifest": str(runtime),
                    "runtime_input_manifest_sha256": sha256_file(runtime),
                    "model_input_manifest": str(model_input),
                    "model_input_manifest_sha256": sha256_file(model_input),
                    "native_ss_deployment": shared_ss,
                    "native_slat_deployment": {
                        "checkpoint": str(checkpoint),
                        "checkpoint_sha256": sha256_file(checkpoint),
                        "checkpoint_step": step,
                        "weights": "ema",
                    },
                    "cross_deployment_bridge": {
                        "path": str(bridge),
                        "sha256": sha256_file(bridge),
                        "passed": True,
                    },
                    "stock_slat_freeze": str(stock),
                    "stock_slat_freeze_sha256": sha256_file(stock),
                    "objects": [
                        {
                            "object_key": "toy:one",
                            "method": OFFICIAL_RECORD_METHOD,
                            "seed": 42,
                            "native_slat_checkpoint_step": step,
                            "native_slat_weights": "ema",
                            "output_frame": "runtime-O",
                            "vggt_model_loaded": False,
                            "vggt_model_executed": False,
                            "target_or_metric_consumed": False,
                            "formal_claim_allowed": False,
                            "mesh": str(mesh),
                            "mesh_sha256": sha256_file(mesh),
                            "passed": True,
                        }
                    ],
                }

            manifest15 = root / "manifest15.json"
            manifest25 = root / "manifest25.json"
            write_json(
                manifest15,
                official_manifest(
                    step=15000,
                    checkpoint=checkpoint15,
                    bridge=bridge15,
                    mesh=slat15_mesh,
                ),
            )
            write_json(
                manifest25,
                official_manifest(
                    step=25000,
                    checkpoint=checkpoint25,
                    bridge=bridge25,
                    mesh=slat25_mesh,
                ),
            )
            output = root / "evaluation"
            argv = [
                "evaluate",
                "--label_manifest",
                str(label_manifest),
                "--reconviagen_manifest",
                str(recon_manifest),
                "--slat15000_manifest",
                str(manifest15),
                "--slat25000_manifest",
                str(manifest25),
                "--output_dir",
                str(output),
                "--expected_objects",
                "1",
                "--surface_samples",
                "100",
                "--bootstrap_samples",
                "100",
                "--skip_proper_sim3",
            ]
            with mock.patch("sys.argv", argv):
                main()
            report = json.loads((output / "report.json").read_text(encoding="utf-8"))
            self.assertTrue(report["passed"])
            self.assertEqual(report["object_count"], 1)
            self.assertEqual(report["record_count"], 3)


if __name__ == "__main__":
    unittest.main()
