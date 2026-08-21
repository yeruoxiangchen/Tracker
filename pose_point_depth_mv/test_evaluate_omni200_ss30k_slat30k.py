#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import trimesh

from manual_mesh_reconstruction.common import canonical_sha256, sha256_file
from pose_point_depth_mv.evaluate_omni200_ss30k_slat30k import (
    BENCHMARK_MANIFEST_FORMAT,
    _normalize_points,
    _paper_metrics,
    cmd_prepare,
)


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


class Omni200CurrentOnlyTest(unittest.TestCase):
    def test_point_normalization_uses_independent_max_extent(self) -> None:
        points = np.asarray([[1.0, 2.0, 3.0], [5.0, 4.0, 3.0]])
        normalized, record = _normalize_points(points)
        self.assertAlmostEqual(float(np.max(np.ptp(normalized, axis=0))), 2.0)
        self.assertEqual(record["normalization"], "independent AABB center/max-extent to [-1,1]")

    def test_identical_mesh_metric_is_near_exact(self) -> None:
        mesh = trimesh.creation.icosphere(subdivisions=2, radius=0.4)
        metrics, _ = _paper_metrics(mesh, mesh.copy(), count=20000, seed=42, radius=0.1)
        self.assertLess(metrics["chamfer_distance"], 0.015)
        self.assertGreater(metrics["fscore"], 0.99)

    def test_prepare_keeps_exact_model_o_camera(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            object_root = root / "render"
            rgba_paths = []
            mask_paths = []
            rendered_files = []
            for index in range(4):
                rgba_path = object_root / "rgba" / f"view_{index:02d}.png"
                mask_path = object_root / "mask" / f"view_{index:02d}.png"
                rgba_path.parent.mkdir(parents=True, exist_ok=True)
                mask_path.parent.mkdir(parents=True, exist_ok=True)
                rgba = np.zeros((64, 64, 4), dtype=np.uint8)
                rgba[16:48, 16:48, :3] = 180
                rgba[16:48, 16:48, 3] = 255
                mask = rgba[..., 3]
                Image.fromarray(rgba, mode="RGBA").save(rgba_path)
                Image.fromarray(mask, mode="L").save(mask_path)
                rgba_paths.append(str(rgba_path))
                mask_paths.append(str(mask_path))
                rendered_files.append(
                    {
                        "rgba": {"path": str(rgba_path), "sha256": sha256_file(rgba_path)},
                        "mask": {"path": str(mask_path), "sha256": sha256_file(mask_path)},
                    }
                )
            object_report = {
                "passed": True,
                "uid": "fixture_001",
                "source_scan_tree_sha256": "a" * 64,
                "rendered_files": rendered_files,
            }
            object_report_path = object_root / "report.json"
            write_json(object_report_path, object_report)
            angle = np.pi / 4.0
            rotation = np.asarray(
                [
                    [np.cos(angle), -np.sin(angle), 0.0],
                    [np.sin(angle), np.cos(angle), 0.0],
                    [0.0, 0.0, 1.0],
                ]
            )
            c2w = np.repeat(np.eye(4)[None], 4, axis=0)
            c2w[:, :3, :3] = rotation
            c2w[:, :3, 3] = np.asarray([0.0, -2.0, 0.2])
            row = {
                "uid": "fixture_001",
                "category": "fixture",
                "source_mesh": str(root / "unused.obj"),
                "source_scan_tree_sha256": "a" * 64,
                "object_report": str(object_report_path),
                "object_report_sha256": sha256_file(object_report_path),
                "selected_input_view_indices": [0, 1, 2, 3],
                "rgba_images": rgba_paths,
                "mask_images": mask_paths,
                "rgb_white_images": rgba_paths,
                "intrinsic": [[80.0, 0.0, 31.5], [0.0, 80.0, 31.5], [0.0, 0.0, 1.0]],
                "c2w_opencv_model_o": c2w.tolist(),
                "source_to_model_o_4x4": np.eye(4).tolist(),
            }
            manifest = {
                "format": BENCHMARK_MANIFEST_FORMAT,
                "protocol_sha256": "b" * 64,
                "objects": [row],
            }
            manifest["manifest_identity"] = canonical_sha256(manifest)
            manifest_path = root / "manifest.json"
            write_json(manifest_path, manifest)
            output = root / "runtime"
            cmd_prepare(
                argparse.Namespace(
                    benchmark_manifest=str(manifest_path),
                    output_dir=str(output),
                    expected_objects=1,
                    resume=False,
                )
            )
            runtime = json.loads((output / "runtime_input_manifest.json").read_text())
            self.assertTrue(runtime["passed"])
            cache = Path(runtime["objects"][0]["cache_npz"])
            with np.load(cache, allow_pickle=False) as payload:
                observed = np.asarray(payload["T_C2O"])
                lifting = np.asarray(payload["T_O2C_lifting"])
            self.assertTrue(np.allclose(observed, c2w))
            self.assertTrue(np.allclose(lifting, np.linalg.inv(c2w)))
            self.assertEqual(runtime["objects"][0]["geometry_mode"], "exact_frozen_model_o")


if __name__ == "__main__":
    unittest.main()
