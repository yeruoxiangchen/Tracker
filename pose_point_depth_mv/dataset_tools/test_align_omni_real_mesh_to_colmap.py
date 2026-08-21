from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from pose_point_depth_mv.dataset_tools.align_omni_real_mesh_to_colmap import (
    AlignmentConfig,
    apply_transform,
    coarse_align_mesh,
    invert_similarity,
    transform_obj_geometry,
)


def proper_similarity() -> np.ndarray:
    rotation = np.asarray(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = 2.5 * rotation
    transform[:3, 3] = [3.0, -4.0, 1.5]
    return transform


class OmniRealMeshCoarseAlignmentTest(unittest.TestCase):
    def test_obj_transform_preserves_topology_and_rotates_normals(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.obj"
            output = root / "output.obj"
            source.write_text(
                "mtllib material.mtl\n"
                "v 1 0 0 0.1 0.2 0.3\n"
                "v 0 1 0\n"
                "v 0 0 1\n"
                "vn 1 0 0\n"
                "usemtl material\n"
                "f 1//1 2//1 3//1\n",
                encoding="utf-8",
            )

            stats = transform_obj_geometry(
                source, output, proper_similarity(), strip_materials=True
            )
            lines = output.read_text(encoding="utf-8").splitlines()

            self.assertEqual(stats["vertex_count"], 3)
            self.assertEqual(stats["normal_count"], 1)
            self.assertEqual(stats["stripped_material_line_count"], 2)
            self.assertFalse(any(line.startswith("mtllib ") for line in lines))
            self.assertFalse(any(line.startswith("usemtl ") for line in lines))
            self.assertIn("f 1//1 2//1 3//1", lines)
            vertex = np.asarray([float(value) for value in lines[0].split()[1:4]])
            expected = apply_transform(
                np.asarray([[1.0, 0.0, 0.0]]), proper_similarity()
            )[0]
            np.testing.assert_allclose(vertex, expected, atol=1.0e-10)
            normal_line = next(line for line in lines if line.startswith("vn "))
            normal = np.asarray([float(value) for value in normal_line.split()[1:4]])
            np.testing.assert_allclose(normal, [0.0, 1.0, 0.0], atol=1.0e-10)

    def test_coarse_alignment_exports_scan_in_colmap_world(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rng = np.random.default_rng(123)
            scan = rng.normal(size=(300, 3)) * np.asarray([1.7, 0.8, 0.35])
            scan[:, 0] += 0.2 * scan[:, 1] ** 2
            world_to_scan = proper_similarity()
            scan_to_world = invert_similarity(world_to_scan)
            world = apply_transform(scan, scan_to_world)
            scan_obj = root / "Scan.obj"
            with scan_obj.open("w", encoding="utf-8") as handle:
                for point in scan:
                    handle.write(f"v {point[0]} {point[1]} {point[2]}\n")

            report = coarse_align_mesh(
                scan_obj=scan_obj,
                world_points=world,
                output_dir=root / "aligned",
                config=AlignmentConfig(
                    min_object_points=50,
                    max_source_points=300,
                    max_scan_vertices=300,
                    max_median_normalized=1.0e-5,
                    max_p90_normalized=1.0e-5,
                    min_inlier_rate_3pct=0.99,
                ),
            )

            self.assertTrue(report["automatic_passed"])
            self.assertLess(report["median_normalized"], 1.0e-8)
            self.assertTrue(Path(report["aligned_mesh"]).is_file())
            self.assertTrue(Path(report["cache_npz"]).is_file())
            with np.load(report["cache_npz"], allow_pickle=False) as cache:
                product = (
                    cache["T_Scan_to_COLMAP_W"]
                    @ cache["T_COLMAP_W_to_Scan"]
                )
                np.testing.assert_allclose(product, np.eye(4), atol=1.0e-9)
                np.testing.assert_allclose(
                    cache["P_W_from_Scan"], world, atol=1.0e-7
                )


if __name__ == "__main__":
    unittest.main()
