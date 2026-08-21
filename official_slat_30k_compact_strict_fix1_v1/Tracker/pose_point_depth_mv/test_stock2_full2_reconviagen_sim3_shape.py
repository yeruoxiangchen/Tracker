from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import trimesh

from pose_point_depth_mv.evaluate_stock2_full2_reconviagen_sim3_shape import (
    CHAPTER93_FORMAT,
    CHAPTER94_FORMAT,
    audit_proper_sim3,
    build_cases,
    file_binding,
    summarize_track,
)


class Stock2Full2ReconViaGenSim3ShapeTest(unittest.TestCase):
    def test_proper_sim3_audit_accepts_uniform_scale_and_rejects_reflection(self) -> None:
        angle = 0.37
        rotation = np.asarray(
            [
                [np.cos(angle), -np.sin(angle), 0.0],
                [np.sin(angle), np.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        matrix = np.eye(4)
        matrix[:3, :3] = 2.5 * rotation
        matrix[:3, 3] = [1.0, -2.0, 3.0]
        result = audit_proper_sim3({"matrix": matrix.tolist(), "cost": 0.1})
        self.assertTrue(result["proper_sim3_validated"])
        self.assertAlmostEqual(result["isotropic_scale"], 2.5)
        self.assertAlmostEqual(result["rotation_determinant"], 1.0)
        reflected = matrix.copy()
        reflected[:3, 0] *= -1.0
        with self.assertRaises(RuntimeError):
            audit_proper_sim3({"matrix": reflected.tolist(), "cost": 0.1})

    def test_paired_summary_uses_positive_as_candidate_better(self) -> None:
        records = []
        values = {
            "reconviagen_original": (0.30, 0.40),
            "stock2": (0.20, 0.60),
            "full2": (0.19, 0.61),
        }
        for index in range(2):
            methods = {}
            for method, (distance, score) in values.items():
                metrics = {
                    "pred_to_gt_mean": distance,
                    "gt_to_pred_mean": distance,
                    "chamfer_l1": distance,
                    "chamfer_l2": distance * distance,
                    "fscore_0p01": score,
                    "fscore_0p02": score,
                    "fscore_0p05": score,
                    "normal_consistency": score,
                }
                methods[method] = {"sim3_shape_only": metrics}
            records.append({"uid": f"uid-{index}", "methods": methods})
        summary = summarize_track(
            records,
            track="sim3_shape_only",
            bootstrap_samples=20,
            seed=42,
        )
        for comparison in summary["comparisons"].values():
            self.assertGreater(comparison["metrics"]["chamfer_l1"]["mean"], 0.0)
            self.assertGreater(comparison["metrics"]["fscore_0p02"]["mean"], 0.0)

    def test_build_cases_binds_the_seed_intersection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "target.obj"
            recon = root / "recon.obj"
            stock2_t3 = root / "stock2_t3.obj"
            mesh = trimesh.creation.box(extents=[1.0, 0.7, 0.4])
            for path in (target, recon, stock2_t3):
                mesh.export(path)
            report94_root = root / "chapter94"
            pair_root = report94_root / "mesh_pairs" / "obj_0000_seed_42"
            stock2 = pair_root / "stock" / "mesh_canonical.obj"
            full2 = pair_root / "full" / "mesh_canonical.obj"
            stock2.parent.mkdir(parents=True)
            full2.parent.mkdir(parents=True)
            mesh.export(stock2)
            mesh.export(full2)

            uid = "object_seq000"
            report93 = {
                "format": CHAPTER93_FORMAT,
                "passed": True,
                "formal": False,
                "methods": [
                    "native_full",
                    "stock",
                    "pixal3d_official",
                    "genrecon_official",
                ],
                "coordinate_policy": {
                    "forbidden": ["per-method ICP", "per-method scaling", "reflection"]
                },
                "records": [
                    {
                        "case_id": "case_s42",
                        "uid": uid,
                        "target_mesh": file_binding(target),
                        "method_bindings": {
                            "stock": {"mesh": file_binding(recon)},
                            "native_full": {"mesh": file_binding(stock2_t3)},
                        },
                    }
                ],
            }
            report94 = {
                "format": CHAPTER94_FORMAT,
                "passed": True,
                "formal": False,
                "run_config": {"seeds": [42, 43, 44]},
                "records": [
                    {
                        "pair_id": "obj_0000_seed_42",
                        "uid": uid,
                        "object_uid": "object",
                        "seed": 42,
                        "same_native_ss_coordinates": True,
                        "same_initial_noise": True,
                        "target": {
                            "source_glb_sha256": "source",
                            "normalize_center": [0.0, 0.0, 0.0],
                            "normalize_scale": 1.0,
                            "canonical_margin": 0.9,
                            "frame": "canonical latent frame; no per-branch normalization or ICP",
                        },
                    }
                ],
            }
            report93_path = root / "chapter93.json"
            report94_path = report94_root / "report.json"
            report94_root.mkdir(exist_ok=True)
            report93_path.write_text(json.dumps(report93), encoding="utf-8")
            report94_path.write_text(json.dumps(report94), encoding="utf-8")
            cases, audit = build_cases(
                report93_path, report94_path, seed=42, expected_objects=1
            )
            self.assertEqual(len(cases), 1)
            self.assertEqual(audit["common_object_count"], 1)
            self.assertEqual(cases[0]["methods"]["stock2"]["path"], str(stock2.resolve()))
            self.assertEqual(cases[0]["methods"]["full2"]["path"], str(full2.resolve()))


if __name__ == "__main__":
    unittest.main()
