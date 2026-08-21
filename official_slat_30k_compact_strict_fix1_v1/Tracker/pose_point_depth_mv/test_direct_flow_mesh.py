from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
import trimesh

from pose_point_depth_mv.export_direct_flow_mesh_pairs import (
    aggregate_report,
    canonical_coords,
    mesh_structure_metrics,
    sha256_file,
    shared_noise_audit,
    surface_metrics,
    validate_completion_manifest,
    vertex_chamfer_metrics,
)


class DirectFlowMeshProtocolTest(unittest.TestCase):
    def test_coordinate_keyed_noise_is_order_invariant(self) -> None:
        master = torch.arange(4 * 4 * 4 * 3, dtype=torch.float32).reshape(4, 4, 4, 3)
        stock = canonical_coords(
            np.asarray([[0, 1, 2], [3, 2, 1], [1, 1, 1]], dtype=np.int32),
            resolution=4,
        )
        correct = canonical_coords(
            np.asarray([[1, 1, 1], [0, 1, 2], [2, 2, 2]], dtype=np.int32),
            resolution=4,
        )
        stock_tensor = torch.from_numpy(stock)
        correct_tensor = torch.from_numpy(correct)
        stock_xyz = stock_tensor[:, 1:].long()
        correct_xyz = correct_tensor[:, 1:].long()
        stock_feats = master[stock_xyz[:, 0], stock_xyz[:, 1], stock_xyz[:, 2]]
        correct_feats = master[
            correct_xyz[:, 0], correct_xyz[:, 1], correct_xyz[:, 2]
        ]
        audit = shared_noise_audit(
            stock_tensor, stock_feats, correct_tensor, correct_feats
        )
        self.assertEqual(audit["common_coord_count"], 2)
        self.assertEqual(audit["common_coord_noise_max_abs"], 0.0)
        self.assertTrue(audit["common_coord_noise_bit_exact"])
        changed = correct_feats.clone()
        changed[0, 0] += 1.0
        mismatch = shared_noise_audit(
            stock_tensor, stock_feats, correct_tensor, changed
        )
        self.assertGreater(mismatch["common_coord_noise_max_abs"], 0.0)

    def test_canonical_coords_unique_sorted(self) -> None:
        value = canonical_coords(
            np.asarray([[2, 1, 0], [0, 0, 0], [2, 1, 0]], dtype=np.int32),
            resolution=4,
        )
        np.testing.assert_array_equal(
            value,
            np.asarray([[0, 0, 0, 0], [0, 2, 1, 0]], dtype=np.int32),
        )

    def test_identity_surface_metrics_are_exact(self) -> None:
        cube = trimesh.creation.box(extents=(0.9, 0.8, 0.7))
        first = surface_metrics(
            cube, cube.copy(), count=2000, seed=123, thresholds=(0.01, 0.02)
        )
        second = surface_metrics(
            cube, cube.copy(), count=2000, seed=123, thresholds=(0.01, 0.02)
        )
        self.assertEqual(first, second)
        self.assertEqual(first["chamfer_l1"], 0.0)
        self.assertEqual(first["fscore_0p02"], 1.0)
        self.assertAlmostEqual(first["normal_consistency"], 1.0, places=7)
        vertex_repeat = vertex_chamfer_metrics(cube, cube.copy())
        self.assertEqual(vertex_repeat["chamfer_l1"], 0.0)

    def test_component_and_watertight_metrics(self) -> None:
        first = trimesh.creation.box(extents=(0.4, 0.4, 0.4))
        second = trimesh.creation.box(extents=(0.1, 0.1, 0.1))
        second.apply_translation((1.0, 0.0, 0.0))
        joined = trimesh.util.concatenate((first, second))
        metrics = mesh_structure_metrics(joined)
        self.assertTrue(metrics["mesh_success"])
        self.assertTrue(metrics["is_watertight"])
        self.assertEqual(metrics["component_count"], 2)
        self.assertGreater(metrics["largest_component_ratio"], 0.49)
        self.assertEqual(metrics["boundary_edge_count"], 0)
        self.assertEqual(metrics["nonmanifold_edge_count"], 0)

    def test_object_balanced_mesh_gate(self) -> None:
        records = []
        for object_uid in ("a", "b"):
            for seed in (42, 43):
                for branch in ("stock", "correct"):
                    correct = branch == "correct"
                    records.append(
                        {
                            "object_uid": object_uid,
                            "seed": seed,
                            "branch": branch,
                            "passed": True,
                            "surface": {
                                "chamfer_l1": 0.08 if not correct else 0.06,
                                "fscore_0p02": 0.40 if not correct else 0.45,
                                "normal_consistency": 0.70 if not correct else 0.72,
                            },
                            "structure": {
                                "largest_component_ratio": 0.90
                                if not correct
                                else 0.91,
                                "mesh_success": True,
                            },
                        }
                    )
        protocol = {
            "sampling": {"joint_seeds": [42, 43]},
            "statistics": {
                "bootstrap_samples": 100,
                "checks": {
                    "chamfer_object_win_rate_min": 0.55,
                    "fscore_0p02_mean_delta_min": 0.0,
                    "mesh_success_rate_delta_min": 0.0,
                    "largest_component_ratio_mean_delta_min": -0.02,
                    "minimum_nonnegative_seed_directions": 2,
                },
            },
        }
        decision = aggregate_report(
            records, protocol=protocol, expected_pairs=4, formal=True
        )
        self.assertTrue(decision["passed"])
        self.assertEqual(decision["completed_pair_count"], 4)
        self.assertAlmostEqual(
            decision["summary"]["chamfer_l1_improvement"]["mean"], 0.02
        )

    def test_completion_manifest_detects_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = root / "report.json"
            artifact = root / "artifact.txt"
            report.write_text(
                json.dumps({"formal": False, "protocol_sha256": "protocol"}),
                encoding="utf-8",
            )
            artifact.write_text("frozen\n", encoding="utf-8")
            files = [report, artifact]
            completion = {
                "complete": True,
                "formal": False,
                "protocol_sha256": "protocol",
                "science_passed": False,
                "all_records_passed": True,
                "runtime_exit_code": 0,
                "file_count": len(files),
                "files": [
                    {
                        "path": str(path.relative_to(root)),
                        "sha256": sha256_file(path),
                    }
                    for path in files
                ],
            }
            (root / "completion_manifest.json").write_text(
                json.dumps(completion), encoding="utf-8"
            )
            self.assertEqual(
                validate_completion_manifest(root, expected_formal=False)[
                    "runtime_exit_code"
                ],
                0,
            )
            artifact.write_text("mutated\n", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                validate_completion_manifest(root, expected_formal=False)


if __name__ == "__main__":
    unittest.main()
