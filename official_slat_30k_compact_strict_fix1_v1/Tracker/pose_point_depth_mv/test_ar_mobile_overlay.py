#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
import trimesh

from pose_point_depth_mv.ar_mobile_overlay import (
    build_mobile_overlay_mesh,
    read_mobile_overlay_mesh,
)


class ARMobileOverlayTest(unittest.TestCase):
    def test_world_mesh_is_compact_and_converted_to_unity_world(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "world.obj"
            output = root / "world.armesh"
            mesh = trimesh.creation.box(extents=[0.4, 0.6, 0.8])
            mesh.apply_translation([1.0, 2.0, 3.0])
            mesh.export(source)

            report = build_mobile_overlay_mesh(source, output, max_triangles=100)
            loaded = read_mobile_overlay_mesh(output)

            self.assertTrue(report["passed"])
            self.assertEqual(report["coordinate_frame"], "unity_world")
            self.assertEqual(report["triangle_count"], 12)
            self.assertLess(output.stat().st_size, source.stat().st_size)
            expected = np.asarray(mesh.vertices, dtype=np.float32).copy()
            expected[:, 2] *= -1.0
            np.testing.assert_allclose(
                np.sort(loaded["vertices"], axis=0),
                np.sort(expected, axis=0),
                atol=1.0e-6,
            )
            lengths = np.linalg.norm(loaded["normals"], axis=1)
            np.testing.assert_allclose(lengths, 1.0, atol=1.0e-6)
            center = loaded["vertices"].mean(axis=0)
            self.assertTrue(
                np.all(
                    np.sum(
                        loaded["normals"] * (loaded["vertices"] - center), axis=1
                    )
                    > 0.0
                )
            )
            self.assertEqual(loaded["faces"].shape, (12, 3))

    def test_vertex_clustering_obeys_mobile_triangle_budget(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "sphere.obj"
            output = root / "sphere.armesh"
            trimesh.creation.icosphere(subdivisions=4).export(source)

            report = build_mobile_overlay_mesh(source, output, max_triangles=300)
            loaded = read_mobile_overlay_mesh(output)

            self.assertGreater(report["triangle_count"], 0)
            self.assertLessEqual(report["triangle_count"], 300)
            self.assertEqual(report["reduction"]["method"], "isotropic_vertex_clustering")
            self.assertEqual(len(loaded["faces"]), report["triangle_count"])


if __name__ == "__main__":
    unittest.main()
