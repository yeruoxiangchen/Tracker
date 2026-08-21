#!/usr/bin/env python3
"""CPU regressions for the official real-input TRELLIS Mesh frame fix."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import trimesh

from pose_point_depth_mv.infer_omni_real_official_with_vggt import (
    MANIFEST_FORMAT,
    REPORT_FORMAT,
)
from pose_point_depth_mv.omni_real_benchmark_common import sha256_file
from pose_point_depth_mv.reframe_omni_official_with_vggt_mesh import (
    reframe_manifest,
)
from pose_point_depth_mv.trellis_mesh_coordinate_contract import (
    DECODER_TO_SPARSE_GRID,
    LEGACY_MESH_FRAME_CONTRACT,
    LEGACY_V1_ERRONEOUS_AXIS_TRANSFORM,
    decoded_mesh_to_sparse_grid_frame,
    mesh_frame_contract_fields,
    validate_runtime_o_mesh_frame_contract,
)


class _DecodedProbe:
    def __init__(self) -> None:
        self.transform_pose: bool | None = None

    def to_trimesh(self, *, transform_pose: bool = False) -> str:
        self.transform_pose = transform_pose
        return "mesh"


class OmniWithVggtMeshCoordinateContractTest(unittest.TestCase):
    def test_decoder_export_requires_transform_pose_false(self) -> None:
        decoded = _DecodedProbe()
        self.assertEqual(decoded_mesh_to_sparse_grid_frame(decoded), "mesh")
        self.assertIs(decoded.transform_pose, False)

    def test_decoder_and_sparse_grid_axis_mapping_is_identity(self) -> None:
        points = np.asarray(
            [[1.0, 2.0, 3.0], [-4.0, 5.0, -6.0]], dtype=np.float64
        )
        observed = trimesh.transform_points(points, DECODER_TO_SPARSE_GRID)
        expected = points
        np.testing.assert_array_equal(observed, expected)
        np.testing.assert_allclose(
            np.linalg.det(DECODER_TO_SPARSE_GRID[:3, :3]), 1.0
        )

    def test_legacy_artifact_is_reframed_once_and_hash_bound(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            legacy_root = root / "legacy"
            mesh_path = legacy_root / "meshes" / "plant" / "plant_012" / "seed_42" / "mesh_o.obj"
            mesh_path.parent.mkdir(parents=True)
            mesh = trimesh.Trimesh(
                vertices=np.asarray(
                    [
                        [0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0],
                        [0.0, 2.0, 0.0],
                        [0.0, 0.0, 3.0],
                    ],
                    dtype=np.float64,
                ),
                faces=np.asarray(
                    [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]],
                    dtype=np.int64,
                ),
                process=False,
            )
            mesh.export(mesh_path, file_type="obj")
            row = {
                "format": REPORT_FORMAT,
                "passed": True,
                "object_key": "plant:plant_012",
                "category": "plant",
                "object_id": "plant_012",
                "seed": 42,
                "mesh": str(mesh_path),
                "mesh_sha256": sha256_file(mesh_path),
                "output_frame": "runtime-O",
                "mesh_frame_contract": LEGACY_MESH_FRAME_CONTRACT,
                "decoder_to_runtime_o_axis_transform_applied": True,
                "decoder_to_runtime_o_axis_transform": (
                    LEGACY_V1_ERRONEOUS_AXIS_TRANSFORM.tolist()
                ),
                "decoder_to_runtime_o_axis_rule": "(x,y,z)->(x,z,-y)",
                "decoder_mesh_export_policy": (
                    "decoded.to_trimesh(transform_pose=True)"
                ),
            }
            result_path = mesh_path.parent / "result.json"
            result_path.write_text(
                json.dumps(row, indent=2) + "\n", encoding="utf-8"
            )
            manifest_path = legacy_root / "inference_manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "format": MANIFEST_FORMAT,
                        "passed": True,
                        "objects": [row],
                        "output_frame": "runtime-O",
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

            output = root / "corrected"
            correction = reframe_manifest(
                source_manifest_path=manifest_path,
                output_dir=output,
                expected_source_mesh_sha256=sha256_file(mesh_path),
            )
            corrected_manifest = json.loads(
                (output / "inference_manifest.json").read_text(encoding="utf-8")
            )
            corrected_row = corrected_manifest["objects"][0]
            validate_runtime_o_mesh_frame_contract(corrected_manifest)
            validate_runtime_o_mesh_frame_contract(corrected_row)
            self.assertTrue(corrected_row["legacy_posthoc_reframe"])
            self.assertEqual(
                correction["source_bbox_extent"], [1.0, 2.0, 3.0]
            )
            self.assertEqual(
                correction["corrected_bbox_extent"], [1.0, 3.0, 2.0]
            )
            corrected_mesh = trimesh.load(
                corrected_row["mesh"], force="mesh", process=False
            )
            self.assertEqual(len(corrected_mesh.vertices), len(mesh.vertices))
            self.assertEqual(len(corrected_mesh.faces), len(mesh.faces))

            with self.assertRaisesRegex(RuntimeError, "legacy runtime-O"):
                reframe_manifest(
                    source_manifest_path=output / "inference_manifest.json",
                    output_dir=root / "double_transform",
                    expected_source_mesh_sha256=corrected_row["mesh_sha256"],
                )

    def test_missing_axis_contract_fails_closed(self) -> None:
        payload = {
            "output_frame": "runtime-O",
            **mesh_frame_contract_fields(export_policy="test"),
        }
        validate_runtime_o_mesh_frame_contract(payload)
        payload["decoder_to_runtime_o_axis_transform_applied"] = True
        with self.assertRaisesRegex(RuntimeError, "identity contract"):
            validate_runtime_o_mesh_frame_contract(payload)


if __name__ == "__main__":
    unittest.main()
