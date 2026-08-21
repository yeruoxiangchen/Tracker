from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
import trimesh

from pose_point_depth_mv.bunny_review.common import binding
from pose_point_depth_mv.render_reconviagen_stock_full_six import (
    METHOD_IDS,
    align_meshes_for_review,
    canonical_prediction_mesh,
    method_label,
    normalized_target_mesh,
)


class RenderSixTest(unittest.TestCase):
    def test_normalized_target_uses_frozen_center_scale_margin(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            source = trimesh.creation.box(extents=(2.0, 4.0, 6.0))
            source.apply_translation((10.0, 20.0, 30.0))
            source_path = tmp_path / "target.ply"
            source.export(source_path)
            target = {
                "source_glb": binding(source_path),
                "normalize_center": [10.0, 20.0, 30.0],
                "normalize_scale": 6.0,
                "canonical_margin": 0.9,
            }
            result = normalized_target_mesh(target)
            self.assertTrue(
                np.allclose(result.bounds.mean(axis=0), np.zeros(3), atol=1.0e-7)
            )
            self.assertTrue(np.isclose(np.max(result.extents), 0.9))

    def test_prediction_uses_source_then_alignment_transform(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            mesh = trimesh.creation.box(extents=(1.0, 2.0, 3.0))
            path = tmp_path / "prediction.ply"
            mesh.export(path)
            source = np.eye(4)
            source[:3, 3] = [1.0, 2.0, 3.0]
            alignment = np.eye(4)
            alignment[:3, 3] = [-0.5, 0.25, 1.0]
            method = {
                "mesh": binding(path),
                "source_to_reference": {"matrix": source.tolist()},
                "alignment": {"matrix": alignment.tolist()},
            }
            result = canonical_prediction_mesh(method, label="test")
            self.assertTrue(
                np.allclose(
                    result.bounds.mean(axis=0),
                    np.array([0.5, 2.25, 4.0]),
                    atol=1.0e-7,
                )
            )

    def test_fixed_column_order_and_labels(self) -> None:
        self.assertEqual(
            METHOD_IDS,
            (
                "gt",
                "reconviagen_original",
                "direct_stock",
                "direct_full",
            ),
        )
        self.assertEqual(method_label("gt", view_count=4), "GT / Reference")
        self.assertIn("4 views", method_label("direct_full", view_count=4))

    def test_canonical_alignment_keeps_mesh_objects_and_identity(self) -> None:
        meshes = {
            method_id: trimesh.creation.box(extents=(1.0, 1.0, 1.0))
            for method_id in METHOD_IDS
        }
        aligned, audits = align_meshes_for_review(
            meshes,
            alignment_mode="canonical_pose",
            alignment_seed=1,
            candidate_samples=10,
            alignment_samples=20,
            candidate_iterations=1,
            final_iterations=1,
        )
        self.assertIs(aligned, meshes)
        for method_id in METHOD_IDS:
            self.assertTrue(
                np.allclose(audits[method_id]["matrix"], np.eye(4))
            )
            self.assertFalse(audits[method_id]["gt_assisted"])


if __name__ == "__main__":
    unittest.main()
