from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import trimesh

from pose_point_depth_mv.evaluate_holdout64_current_vs_reconviagen_shape import (
    CURRENT_MANIFEST_FORMAT,
    CURRENT_RECORD_METHOD,
    RECONVIAGEN_MANIFEST_FORMAT,
    RECONVIAGEN_RECORD_METHOD,
    RECONVIAGEN_DECODER_TO_REFERENCE,
    REPORT_FORMAT,
    main,
    normalize_mesh_bbox,
    paired_comparison,
    surface_metrics_from_samples,
)
from pose_point_depth_mv.dataset_tools.prepare_omni_real_label_cache import (
    MANIFEST_FORMAT as LABEL_MANIFEST_FORMAT,
)
from pose_point_depth_mv.mesh_benchmark_metrics import deterministic_surface_sample
from pose_point_depth_mv.omni_real_benchmark_common import sha256_file


class Holdout64CurrentVsReconViaGenShapeTest(unittest.TestCase):
    def test_bbox_normalization_removes_translation_and_uniform_scale(self) -> None:
        mesh = trimesh.creation.box(extents=[2.0, 4.0, 1.0])
        mesh.vertices = np.asarray(mesh.vertices) * 3.5 + np.asarray([8.0, -2.0, 5.0])
        normalized, record = normalize_mesh_bbox(mesh)
        vertices = np.asarray(normalized.vertices)
        np.testing.assert_allclose(vertices.min(axis=0) + vertices.max(axis=0), 0.0)
        self.assertAlmostEqual(float(np.ptp(vertices, axis=0).max()), 1.0)
        self.assertAlmostEqual(record["uniform_scale_divisor"], 14.0)

    def test_reconviagen_fixed_axis_is_proper_and_has_expected_mapping(self) -> None:
        rotation = RECONVIAGEN_DECODER_TO_REFERENCE[:3, :3]
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(3))
        self.assertAlmostEqual(float(np.linalg.det(rotation)), 1.0)
        point = np.asarray([[2.0, 3.0, 5.0, 1.0]])
        transformed = point @ RECONVIAGEN_DECODER_TO_REFERENCE.T
        np.testing.assert_allclose(transformed[0, :3], [2.0, 5.0, -3.0])

    def test_sample_metric_is_exact_for_identical_mesh(self) -> None:
        mesh, _ = normalize_mesh_bbox(trimesh.creation.icosphere(subdivisions=2))
        samples = deterministic_surface_sample(mesh, 2000, 42)
        metrics = surface_metrics_from_samples(
            samples, samples, thresholds=(0.01, 0.02), workers=1
        )
        self.assertAlmostEqual(metrics["chamfer_l1"], 0.0)
        self.assertAlmostEqual(metrics["chamfer_l2"], 0.0)
        self.assertAlmostEqual(metrics["fscore_0p01"], 1.0)
        self.assertAlmostEqual(metrics["normal_consistency"], 1.0)

    def test_paired_sign_is_positive_when_current_is_better(self) -> None:
        records = []
        for method, distance, score in (
            ("current_point_mask", 0.1, 0.8),
            ("reconviagen_original", 0.2, 0.6),
        ):
            records.append({
                "method": method,
                "object_key": "category:object",
                "pred_to_gt_mean": distance,
                "gt_to_pred_mean": distance,
                "chamfer_l1": distance,
                "chamfer_l2": distance,
                "fscore_0p01": score,
                "fscore_0p02": score,
                "normal_consistency": score,
            })
        comparison = paired_comparison(records)
        for metric in comparison["metrics"].values():
            self.assertGreater(metric["mean"], 0.0)
            self.assertEqual(metric["current_win_count"], 1)
            self.assertEqual(metric["reconviagen_win_count"], 0)

    def test_end_to_end_report_and_completed_run_reuse(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mesh_path = root / "mesh.obj"
            trimesh.creation.box(extents=[1.0, 0.7, 0.4]).export(mesh_path)
            mesh_sha256 = sha256_file(mesh_path)
            runtime_path = root / "runtime.json"
            runtime_path.write_text('{"passed":true}\n', encoding="utf-8")
            runtime_sha256 = sha256_file(runtime_path)
            label_rows = []
            current_rows = []
            recon_rows = []
            for index in range(64):
                object_id = f"object_{index:02d}"
                key = f"category:{object_id}"
                label_rows.append({
                    "category": "category",
                    "object_id": object_id,
                    "object_key": key,
                    "mesh_o": str(mesh_path),
                    "mesh_o_sha256": mesh_sha256,
                    "alignment_quality_passed": True,
                    "passed": True,
                })
                common = {
                    "object_key": key,
                    "category": "category",
                    "object_id": object_id,
                    "seed": 42,
                    "mesh": str(mesh_path),
                    "mesh_sha256": mesh_sha256,
                    "passed": True,
                }
                current_rows.append({**common, "method": CURRENT_RECORD_METHOD})
                recon_rows.append({**common, "method": RECONVIAGEN_RECORD_METHOD})

            def write_manifest(name: str, payload: dict) -> Path:
                path = root / name
                path.write_text(json.dumps(payload), encoding="utf-8")
                return path

            label_path = write_manifest("labels.json", {
                "format": LABEL_MANIFEST_FORMAT,
                "passed": True,
                "selected_object_count": 64,
                "completed_object_count": 64,
                "runtime_input_manifest": str(runtime_path),
                "runtime_input_manifest_sha256": runtime_sha256,
                "objects": label_rows,
            })
            common_manifest = {
                "object_count": 64,
                "record_count": 64,
                "seeds": [42],
                "passed": True,
                "runtime_input_manifest": str(runtime_path),
                "runtime_input_manifest_sha256": runtime_sha256,
            }
            current_path = write_manifest("current.json", {
                **common_manifest,
                "format": CURRENT_MANIFEST_FORMAT,
                "method": CURRENT_RECORD_METHOD,
                "objects": current_rows,
            })
            recon_path = write_manifest("recon.json", {
                **common_manifest,
                "format": RECONVIAGEN_MANIFEST_FORMAT,
                "method": RECONVIAGEN_RECORD_METHOD,
                "objects": recon_rows,
            })
            output_dir = root / "output"
            argv = [
                "evaluate",
                "--label_manifest", str(label_path),
                "--current_manifest", str(current_path),
                "--reconviagen_manifest", str(recon_path),
                "--output_dir", str(output_dir),
                "--surface_samples", "32",
                "--workers", "1",
                "--resume",
            ]
            with mock.patch.object(sys, "argv", argv), redirect_stdout(StringIO()):
                main()
            report = json.loads((output_dir / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["format"], REPORT_FORMAT)
            self.assertTrue(report["passed"])
            self.assertFalse(report["formal"])
            self.assertEqual(report["object_count"], 64)
            self.assertEqual(report["record_count"], 128)
            with mock.patch.object(sys, "argv", argv), redirect_stdout(StringIO()):
                main()


if __name__ == "__main__":
    unittest.main()
