#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from pose_point_depth_mv.ar_object_capture import CAPTURE_FORMAT
from pose_point_depth_mv.omni_real_benchmark_common import sha256_file
from pose_point_depth_mv.run_ar_offline_colmap_ab import (
    DEFAULT_OUTPUT_ROOT,
    build_parser,
    camera_diagnostics,
    colmap_commands,
    load_ar_control,
    materialize_frozen_dataset,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


class AROfflineColmapABTest(unittest.TestCase):
    def test_default_uses_new_full_image_output_without_foreground_flag(self) -> None:
        args = build_parser().parse_args([])
        self.assertFalse(args.foreground_features)
        self.assertEqual(Path(args.output_root), DEFAULT_OUTPUT_ROOT)
        self.assertIn("fullimage", str(DEFAULT_OUTPUT_ROOT))

    def test_ar_control_recovers_exact_source_frames_and_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "capture"
            replay = root / "replay"
            frames = [f"frame_{index:04d}.jpg" for index in range(8)]
            bindings = []
            for index, name in enumerate(frames):
                image = source / "images" / name
                mask = source / "masks" / f"{Path(name).stem}.png"
                image.parent.mkdir(parents=True, exist_ok=True)
                mask.parent.mkdir(parents=True, exist_ok=True)
                image.write_bytes(f"rgb-{index}".encode("ascii"))
                mask.write_bytes(f"mask-{index}".encode("ascii"))
                bindings.append({
                    "source_frame_name": name,
                    "image_sha256": sha256_file(image),
                    "mask_sha256": sha256_file(mask),
                })
            _write_json(source / "capture_report.json", {
                "format": CAPTURE_FORMAT,
                "passed": True,
                "selected_frame_names": frames,
            })
            raw_cache = replay / "raw.npz"
            raw_cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                raw_cache,
                source_frame_name=np.asarray(frames),
            )
            raw_report = replay / "shared/01_raw_cache/raw_cache_report.json"
            _write_json(raw_report, {
                "passed": True,
                "objects": [{
                    "cache_npz": str(raw_cache),
                    "source_binding": {"frames": bindings},
                }],
            })
            runtime = replay / "shared/02_runtime_o/runtime_input_manifest.json"
            _write_json(runtime, {
                "passed": True,
                "source_raw_cache_report": str(raw_report),
                "objects": [{"selected_source_view_indices": list(range(8))}],
            })
            inference = replay / "mixed/04_no_vggt_inference/inference_manifest.json"
            _write_json(inference, {"passed": True})
            sheet = replay / "mixed/final/previews/mesh_views_contact_sheet.png"
            sheet.parent.mkdir(parents=True, exist_ok=True)
            sheet.write_bytes(b"sheet")

            control = load_ar_control(source, replay, frames)
            self.assertEqual(control["fixed_frames"], frames)
            self.assertEqual(len(control["frozen_bindings"]), 8)
            with self.assertRaisesRegex(RuntimeError, "fixed-frame contract differs"):
                load_ar_control(source, replay, list(reversed(frames)))

    def test_colmap_commands_use_single_camera_and_colmap_mask_naming(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            commands = dict(colmap_commands(
                colmap_bin=Path("/opt/colmap"),
                workspace=workspace,
                use_foreground_masks=True,
            ))
            feature = commands["feature_extractor"]
            self.assertIn("--ImageReader.mask_path", feature)
            self.assertEqual(feature[feature.index("--ImageReader.single_camera") + 1], "1")
            self.assertEqual(feature[feature.index("--SiftExtraction.gpu_index") + 1], "0")
            self.assertIn("--Mapper.tri_ignore_two_view_tracks", commands["mapper"])

    def test_frozen_dataset_exposes_only_requested_frames(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            fixed = ("frame_0001.jpg", "frame_0003.jpg")
            for index in range(4):
                name = f"frame_{index:04d}.jpg"
                image = source / "images" / name
                mask = source / "masks" / f"frame_{index:04d}.png"
                image.parent.mkdir(parents=True, exist_ok=True)
                mask.parent.mkdir(parents=True, exist_ok=True)
                image.write_bytes(name.encode("ascii"))
                mask.write_bytes(f"mask-{name}".encode("ascii"))
            model = root / "model"
            model.mkdir()
            (model / "cameras.txt").write_text(
                "1 SIMPLE_PINHOLE 640 480 500 320 240\n", encoding="utf-8"
            )
            image_lines = ["# test model"]
            for index in range(4):
                image_lines.extend([
                    f"{index + 1} 1 0 0 0 0 0 0 1 frame_{index:04d}.jpg",
                    "0 0 -1",
                ])
            (model / "images.txt").write_text(
                "\n".join(image_lines) + "\n", encoding="utf-8"
            )
            (model / "points3D.txt").write_text("# no points needed here\n", encoding="utf-8")
            destination = root / "frozen"
            report = materialize_frozen_dataset(
                source_dataset=source,
                fixed_frames=fixed,
                text_model=model,
                destination=destination,
            )
            self.assertTrue(report["passed"])
            self.assertEqual(
                sorted(path.name for path in (destination / "images").iterdir()),
                sorted(fixed),
            )

    def test_camera_diagnostics_are_similarity_gauge_invariant(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            names = np.asarray(["a.jpg", "b.jpg", "c.jpg"])
            ar_k = np.repeat(np.eye(3)[None], 3, axis=0)
            ar_k[:, 0, 0] = 500.0
            ar_k[:, 1, 1] = 510.0
            colmap_k = ar_k.copy()
            colmap_k[:, 0, 0] *= 1.1
            colmap_k[:, 1, 1] *= 0.9
            ar_pose = np.repeat(np.eye(4)[None], 3, axis=0)
            colmap_pose = np.repeat(np.eye(4)[None], 3, axis=0)
            for index in range(3):
                ar_pose[index, 0, 3] = -float(index)
                colmap_pose[index, 0, 3] = -3.0 * float(index)
            ar_path = root / "ar.npz"
            colmap_path = root / "colmap.npz"
            np.savez_compressed(ar_path, source_frame_name=names, K=ar_k, T_W2C=ar_pose)
            np.savez_compressed(
                colmap_path,
                source_frame_name=names,
                K=colmap_k,
                T_W2C=colmap_pose,
            )
            report = camera_diagnostics(ar_path, colmap_path, names.tolist())
            self.assertAlmostEqual(
                report["pose"]["pairwise_relative_rotation_error_degrees"]["max"],
                0.0,
            )
            self.assertAlmostEqual(
                report["pose"]["scale_normalized_pairwise_baseline_absolute_error"]["max"],
                0.0,
            )
            self.assertAlmostEqual(
                report["intrinsics"]["colmap_over_ar_fx_minus_one"]["mean"],
                0.1,
            )


if __name__ == "__main__":
    unittest.main()
