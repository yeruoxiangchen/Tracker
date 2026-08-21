from __future__ import annotations

import unittest

import numpy as np
import trimesh

from pose_point_depth_mv.evaluate_holdout64_current_vs_reconviagen_shape import (
    RECONVIAGEN_DECODER_TO_REFERENCE,
    normalize_mesh_bbox,
)
from pose_point_depth_mv.evaluate_omni_pose_mask_ss30k_slat30k_vs_reconviagen_shape import (
    METHODS,
    numeric_summary,
    summarize_population,
    surface_metrics_from_meshes,
)
from pose_point_depth_mv.infer_omni_real_reconviagen import (
    IMAGE_ONLY_RUNTIME_MANIFEST_FORMATS,
)


class OmniPoseMaskSS30KSLat30KShapeTest(unittest.TestCase):
    def test_frozen_pose_mask_runtime_v2_is_accepted_by_reconviagen(self) -> None:
        self.assertIn(
            "pose_point_depth_mv.omni_real_runtime_input_manifest.v2",
            IMAGE_ONLY_RUNTIME_MANIFEST_FORMATS,
        )

    def test_fixed_reconviagen_axis_is_proper(self) -> None:
        rotation = RECONVIAGEN_DECODER_TO_REFERENCE[:3, :3]
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-12)
        self.assertAlmostEqual(float(np.linalg.det(rotation)), 1.0, places=12)

    def test_every_mesh_normalization_has_unit_longest_extent(self) -> None:
        mesh = trimesh.creation.box(extents=(2.0, 4.0, 8.0))
        mesh.apply_translation((10.0, -3.0, 7.0))
        normalized, audit = normalize_mesh_bbox(mesh)
        self.assertAlmostEqual(float(np.max(np.ptp(normalized.vertices, axis=0))), 1.0)
        np.testing.assert_allclose(
            0.5 * (normalized.bounds[0] + normalized.bounds[1]),
            np.zeros(3),
            atol=1.0e-12,
        )
        self.assertEqual(audit["uniform_scale_divisor"], 8.0)

    def test_identical_normalized_mesh_metrics_are_exact(self) -> None:
        mesh = trimesh.creation.icosphere(subdivisions=2, radius=0.5)
        metrics = surface_metrics_from_meshes(
            mesh, mesh, count=1000, seed=17, thresholds=(0.01, 0.02, 0.05)
        )
        self.assertAlmostEqual(metrics["chamfer_l1"], 0.0, places=12)
        self.assertAlmostEqual(metrics["fscore_0p02"], 1.0, places=12)
        self.assertAlmostEqual(metrics["normal_consistency"], 1.0, places=12)

    def test_paired_sign_is_positive_when_candidate_is_better(self) -> None:
        records = []
        for position in range(3):
            candidate = {
                "pred_to_gt_mean": 0.1,
                "gt_to_pred_mean": 0.1,
                "chamfer_l1": 0.1,
                "chamfer_l2": 0.01,
                "fscore_0p01": 0.8,
                "fscore_0p02": 0.9,
                "fscore_0p05": 1.0,
                "normal_consistency": 0.9,
            }
            baseline = {
                "pred_to_gt_mean": 0.2,
                "gt_to_pred_mean": 0.2,
                "chamfer_l1": 0.2,
                "chamfer_l2": 0.04,
                "fscore_0p01": 0.4,
                "fscore_0p02": 0.5,
                "fscore_0p05": 0.8,
                "normal_consistency": 0.7,
            }
            records.append(
                {
                    "object_key": f"object:{position}",
                    "methods": {
                        METHODS[0]: {"metrics": candidate},
                        METHODS[1]: {"metrics": baseline},
                    },
                }
            )
        summary = summarize_population(records, bootstrap_samples=100, seed=4)
        paired = summary["ss30k_slat30k_vs_reconviagen"]["metrics"]
        self.assertGreater(paired["chamfer_l1"]["mean"], 0.0)
        self.assertGreater(paired["fscore_0p02"]["mean"], 0.0)
        self.assertEqual(paired["normal_consistency"]["positive_rate"], 1.0)

    def test_numeric_summary_rejects_nonfinite(self) -> None:
        with self.assertRaises(ValueError):
            numeric_summary([1.0, float("nan")], bootstrap_samples=10, seed=1)


if __name__ == "__main__":
    unittest.main()
