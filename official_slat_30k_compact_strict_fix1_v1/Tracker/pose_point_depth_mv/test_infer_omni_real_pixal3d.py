#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
import tempfile
import unittest

from pose_point_depth_mv.infer_omni_real_pixal3d import (
    _pixal3d_worker_environment,
    _validate_saved_result,
    _worker_command,
)


def worker_args() -> argparse.Namespace:
    return argparse.Namespace(
        runtime_input_manifest="runtime.json",
        output_dir="output",
        model_path="model",
        naf_repo=Path("naf_repo"),
        naf_checkpoint=Path("naf.pt"),
        naf_source_manifest=Path("source.sha256"),
        device="cuda",
        low_vram=True,
        seeds="42,43",
        mesh_scale=1.0,
        resolution=1024,
        max_num_tokens=49152,
        sampling_steps=12,
    )


class OmniRealPixal3DIsolationTest(unittest.TestCase):
    def test_worker_environment_does_not_inherit_reconviagen_flash_attn(self) -> None:
        environment = _pixal3d_worker_environment(
            {
                "ATTN_BACKEND": "flash_attn",
                "SPARSE_ATTN_BACKEND": "flash_attn",
                "KEEP_ME": "yes",
            }
        )
        self.assertEqual(environment["ATTN_BACKEND"], "sdpa")
        self.assertEqual(environment["SPARSE_ATTN_BACKEND"], "sdpa")
        self.assertEqual(environment["KEEP_ME"], "yes")

    def test_worker_command_preserves_protocol_and_batches_objects(self) -> None:
        command = _worker_command(
            worker_args(),
            keys=["box:box_002", "book:book_006"],
            seed=43,
        )
        self.assertIn("--_worker", command)
        self.assertNotIn("--isolate_objects", command)
        self.assertIn("--low_vram", command)
        self.assertEqual(command[command.index("--seeds") + 1], "43")
        self.assertEqual(command[command.index("--resolution") + 1], "1024")
        self.assertEqual(command[command.index("--max_num_tokens") + 1], "49152")
        self.assertEqual(command.count("--object"), 2)
        selected = [
            command[index + 1]
            for index, value in enumerate(command)
            if value == "--object"
        ]
        self.assertEqual(selected, ["box:box_002", "book:book_006"])

    def test_saved_result_rejects_partial_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mesh = root / "mesh.glb"
            mesh.write_bytes(b"mesh")
            with self.assertRaisesRegex(RuntimeError, "partial Pixal3D output"):
                _validate_saved_result(
                    root / "result.json",
                    mesh,
                    row={
                        "category": "box",
                        "object_id": "box_002",
                        "reference_view_index": 0,
                    },
                    seed=42,
                    runtime_path=root / "runtime.json",
                    runtime_sha256="runtime",
                    rgba_path=root / "input.png",
                    rgba_sha256="rgba",
                    model_path="model",
                    snapshot={},
                    naf={},
                    inference_config_sha256="config",
                )


if __name__ == "__main__":
    unittest.main()
