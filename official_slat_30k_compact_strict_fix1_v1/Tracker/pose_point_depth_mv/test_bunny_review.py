from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from pose_point_depth_mv.bunny_review.common import (
    load_method_result,
    load_protocol,
    write_method_result,
)
from pose_point_depth_mv.bunny_review.prepare import border_connected_white, main
from pose_point_depth_mv.bunny_review.trained_adapter import run_command


class BunnyMaskTests(unittest.TestCase):
    def test_only_border_connected_white_is_background(self) -> None:
        rgb = np.full((7, 7, 3), 255, dtype=np.uint8)
        rgb[1:6, 1:6] = 20
        rgb[3, 3] = 255
        background = border_connected_white(rgb, 245)
        self.assertTrue(background[0, 0])
        self.assertFalse(background[2, 2])
        self.assertFalse(background[3, 3])


class BunnyProtocolTests(unittest.TestCase):
    def test_prepare_and_result_contract(self) -> None:
        tracker = Path(__file__).resolve().parents[1]
        bunny = tracker / "pose_point_depth_mv" / "bunny"
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "review"
            from pose_point_depth_mv.bunny_review.prepare import make_parser, prepare

            args = make_parser().parse_args(
                [
                    "--bunny_root",
                    str(bunny),
                    "--output_dir",
                    str(output),
                    "--view_indices",
                    "0,1,2,3,4",
                    "--single_view_index",
                    "0",
                ]
            )
            prepare(args)
            protocol = load_protocol(output / "protocol.json")
            self.assertEqual(protocol["view_indices"], [0, 1, 2, 3, 4])
            self.assertEqual(protocol["single_view_index"], 0)
            self.assertEqual(len(protocol["views"]), 5)
            reference = load_method_result(output / "protocol.json", "reference")
            self.assertTrue(reference["complete"])

            result = write_method_result(
                protocol_path=output / "protocol.json",
                method_id="dummy",
                display_name="Dummy",
                mesh_path=bunny / "meshes" / "model.obj",
                input_view_indices=[0, 1],
                backend={"kind": "unit_test"},
            )
            self.assertTrue(result.is_file())
            loaded = load_method_result(output / "protocol.json", "dummy")
            self.assertEqual(loaded["input_view_count"], 2)

            adapter_path = output / "test_adapter.json"
            adapter_path.write_text(
                json.dumps(
                    {
                        "format": "pose_point_depth_mv.bunny_command_adapter.v1",
                        "command": [
                            "/bin/cp",
                            str(bunny / "meshes" / "model.obj"),
                            "{output_dir}/mesh.obj",
                        ],
                        "expected_mesh": "mesh.obj",
                        "input_paths": [str(output / "protocol.json")],
                    }
                ),
                encoding="utf-8",
            )
            run_command(
                argparse.Namespace(
                    protocol=output / "protocol.json",
                    adapter=adapter_path,
                    method_id="adapter_input_binding",
                    display_name="Adapter input binding",
                    allow_reference_as_input=False,
                )
            )
            adapter_result = load_method_result(
                output / "protocol.json", "adapter_input_binding"
            )
            self.assertEqual(
                Path(
                    adapter_result["backend"]["input_bindings"]["input_00"]["path"]
                ),
                output / "protocol.json",
            )

            rgba = Path(protocol["views"][0]["rgba"]["path"])
            original = rgba.read_bytes()
            rgba.write_bytes(original + b"tamper")
            with self.assertRaises(RuntimeError):
                load_protocol(output / "protocol.json")


if __name__ == "__main__":
    unittest.main()
