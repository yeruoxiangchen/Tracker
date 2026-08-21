from __future__ import annotations

import unittest
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image
import trimesh

from pose_point_depth_mv.prepare_omni_real_holdout_mesh_review import (
    _apply_camera_distortion,
    _image_pixels_to_nvdiffrast_ndc,
    _per_view_comparison_sheets,
    _prepare_prediction,
    _write_turntable_video,
    _silhouette_iou,
    _world_mesh_buffers,
    score_final_records,
    select_cases,
    validate_current_gaussian_appearance_report,
    validate_world_projection_chain,
)
from pose_point_depth_mv.omni_real_benchmark_common import sha256_file
from pose_point_depth_mv.dataset_tools.build_objaverse_multiview_sparse_data import (
    vertex_colors_from_visual,
)


class HoldoutMeshReviewSelectionTest(unittest.TestCase):
    def _rows(self):
        rows = []
        for index in range(8):
            rows.append(
                {
                    "method": "final_native_no_vggt",
                    "object_key": f"item:item_{index:03d}",
                    "chamfer_l1": 0.01 + index * 0.01,
                    "fscore_0p02": 0.9 - index * 0.05,
                    "normal_consistency": 0.95 - index * 0.03,
                    "largest_component_ratio": 1.0 - index * 0.02,
                    "alignment_quality_tier": (
                        "low_confidence" if index == 0 else "reliable"
                    ),
                }
            )
        return rows

    def test_scoring_is_deterministic_and_top_excludes_low_confidence_label(self):
        rows = self._rows()
        scores = score_final_records(rows)
        self.assertGreater(
            scores["item:item_000"]["aggregate_score"],
            scores["item:item_007"]["aggregate_score"],
        )
        report = {"records": rows}
        first = select_cases(report, top_count=2, random_count=2, random_seed=42)
        second = select_cases(report, top_count=2, random_count=2, random_seed=42)
        self.assertEqual(first, second)
        self.assertEqual(first[0]["object_key"], "item:item_001")
        self.assertEqual(first[1]["object_key"], "item:item_002")
        self.assertEqual({row["selection"] for row in first[2:]}, {"fixed_random"})

    def test_additional_random_selection_excludes_prior_review(self):
        report = {"records": self._rows()}
        excluded = {"item:item_001", "item:item_004"}
        selected = select_cases(
            report,
            top_count=0,
            random_count=4,
            random_seed=20260810,
            excluded_object_keys=excluded,
        )
        self.assertEqual(len(selected), 4)
        self.assertTrue(excluded.isdisjoint(row["object_key"] for row in selected))
        self.assertEqual({row["selection"] for row in selected}, {"fixed_random"})

    def test_grouped_render_layout_writes_one_all_method_sheet_per_view(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            methods = ("gt", "final_native_no_vggt")
            images = {method: [] for method in methods}
            for method_index, method in enumerate(methods):
                for view in range(2):
                    path = root / f"{method}_{view}.png"
                    Image.new(
                        "RGBA",
                        (16, 12),
                        (50 + 100 * method_index, 40 + 60 * view, 80, 255),
                    ).save(path)
                    images[method].append(path)
            outputs = _per_view_comparison_sheets(
                images,
                root / "grouped",
                view_count=2,
                method_order=methods,
            )
            self.assertEqual([path.name for path in outputs], [
                "view_000_六路并排.png",
                "view_001_六路并排.png",
            ])
            self.assertTrue(all(path.is_file() for path in outputs))

    def test_turntable_video_combines_methods_without_retaining_extra_sheets(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            methods = ("gt", "final_native_no_vggt")
            images = {method: [] for method in methods}
            for method_index, method in enumerate(methods):
                for frame in range(4):
                    path = root / f"{method}_{frame}.png"
                    Image.new(
                        "RGBA",
                        (32, 24),
                        (60 + 90 * method_index, 30 + 40 * frame, 100, 255),
                    ).save(path)
                    images[method].append(path)
            destination = root / "turntable.mp4"
            _write_turntable_video(
                images,
                destination,
                fps=4,
                method_order=methods,
            )
            self.assertTrue(destination.is_file())
            self.assertGreater(destination.stat().st_size, 0)

    def test_video_mode_references_prediction_mesh_without_copying_obj(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.obj"
            trimesh.Trimesh(
                vertices=np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float),
                faces=np.asarray([[0, 1, 2]], dtype=np.int64),
                process=False,
            ).export(source)
            destination = root / "review_mesh_assets"
            row = _prepare_prediction(
                {"mesh": str(source), "mesh_sha256": sha256_file(source)},
                destination,
                copy_assets=False,
            )
            self.assertFalse(destination.exists())
            self.assertEqual(Path(row["display_mesh"]), source.resolve())
            self.assertNotIn("copied_mesh", row)

    def test_existing_vertex_rgb_is_not_replaced_by_gray_fallback(self):
        mesh = trimesh.Trimesh(
            vertices=np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float),
            faces=np.asarray([[0, 1, 2]], dtype=np.int64),
            vertex_colors=np.asarray(
                [[255, 0, 0, 255], [0, 255, 0, 255], [0, 0, 255, 255]],
                dtype=np.uint8,
            ),
            process=False,
        )
        np.testing.assert_array_equal(
            vertex_colors_from_visual(mesh),
            np.asarray([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8),
        )

    def test_physical_world_projection_chain_uses_o2w_then_w2c(self):
        angle = np.deg2rad(30.0)
        rotation = np.asarray(
            [
                [np.cos(angle), 0.0, np.sin(angle)],
                [0.0, 1.0, 0.0],
                [-np.sin(angle), 0.0, np.cos(angle)],
            ]
        )
        T_O2W = np.eye(4)
        T_O2W[:3, :3] = 1.7 * rotation
        T_O2W[:3, 3] = [0.2, -0.4, 1.3]
        T_W2C = np.stack([np.eye(4), np.eye(4)])
        T_W2C[0, :3, 3] = [0.1, 0.2, 2.0]
        T_W2C[1, :3, 3] = [-0.3, 0.1, 2.4]
        stored = np.matmul(T_W2C, T_O2W[None])
        audit = validate_world_projection_chain(T_O2W, T_W2C, stored)
        self.assertTrue(audit["passed"])
        self.assertLess(audit["max_abs"], 1.0e-12)
        with self.assertRaises(RuntimeError):
            validate_world_projection_chain(
                T_O2W, T_W2C, stored + 1.0e-3, tolerance=1.0e-6
            )

    def test_world_mesh_buffer_applies_scale_rotation_and_translation(self):
        import tempfile
        from pathlib import Path

        mesh = trimesh.Trimesh(
            vertices=np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float),
            faces=np.asarray([[0, 1, 2]], dtype=np.int64),
            process=False,
        )
        T_O2W = np.eye(4)
        T_O2W[:3, :3] = 2.0 * np.asarray(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        )
        T_O2W[:3, 3] = [3.0, 4.0, 5.0]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "mesh.obj"
            mesh.export(path)
            vertices_w, faces = _world_mesh_buffers(path, T_O2W)
        np.testing.assert_allclose(
            vertices_w,
            np.asarray([[3, 4, 5], [3, 6, 5], [1, 4, 5]], dtype=float),
            atol=1.0e-7,
        )
        np.testing.assert_array_equal(faces, np.asarray([[0, 1, 2]]))

    def test_silhouette_iou(self):
        first = np.asarray([[1, 1], [0, 0]], dtype=bool)
        second = np.asarray([[1, 0], [1, 0]], dtype=bool)
        self.assertAlmostEqual(_silhouette_iou(first, second), 1.0 / 3.0)

    def test_raw_simple_radial_distortion(self):
        x = np.asarray([0.5])
        y = np.asarray([0.25])
        output_x, output_y = _apply_camera_distortion(
            x, y, {"model": "SIMPLE_RADIAL", "distortion": [0.1]}
        )
        factor = 1.0 + 0.1 * (0.5**2 + 0.25**2)
        np.testing.assert_allclose(output_x, [0.5 * factor])
        np.testing.assert_allclose(output_y, [0.25 * factor])

    def test_nvdiffrast_ndc_preserves_image_down_row_order(self):
        x, y = _image_pixels_to_nvdiffrast_ndc(
            np.asarray([0.0, 99.0]), np.asarray([0.0, 49.0]), 100, 50
        )
        np.testing.assert_allclose(x, [-1.0, 1.0])
        np.testing.assert_allclose(y, [-1.0, 1.0])

    def test_current_texture_evidence_is_bound_to_selected_gaussian_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inference = root / "inference_manifest.json"
            inference.write_text("{}", encoding="utf-8")
            contact = root / "contact.png"
            contact.write_bytes(b"contact")
            artifacts = {}
            branches = {}
            for branch in ("stock", "full"):
                ply = root / f"{branch}.ply"
                textured_glb = root / f"{branch}_textured.glb"
                rgb = root / f"{branch}_rgb.png"
                alpha = root / f"{branch}_alpha.png"
                ply.write_bytes(branch.encode("utf-8"))
                textured_glb.write_bytes(f"{branch}-glb".encode("utf-8"))
                rgb.write_bytes(b"rgb")
                alpha.write_bytes(b"alpha")
                artifacts[branch] = {
                    "gaussian_ply": str(ply),
                    "gaussian_ply_sha256": sha256_file(ply),
                    "textured_glb": str(textured_glb),
                    "textured_glb_sha256": sha256_file(textured_glb),
                }
                branches[branch] = [
                    {"view_index": 0, "rgb": str(rgb), "alpha": str(alpha)}
                ]
            benchmark = {
                "method_manifests": {
                    "final_native_no_vggt": {
                        "path": str(inference),
                        "sha256": sha256_file(inference),
                    }
                }
            }
            selected = [{"object_key": "item:item_001"}]
            gaussian = {
                "format": "pose_point_depth_mv.native_no_vggt_gaussian_appearance.v1",
                "passed": True,
                "formal": False,
                "run_config": {
                    "source_inference_manifest": str(inference),
                    "source_inference_manifest_sha256": sha256_file(inference),
                    "export_glb": True,
                },
                "objects": [
                    {
                        "format": "pose_point_depth_mv.native_no_vggt_gaussian_appearance_object.v1",
                        "object_key": "item:item_001",
                        "passed": True,
                        "artifacts": artifacts,
                        "branches": branches,
                        "contact_sheet": str(contact),
                    }
                ],
            }
            validated = validate_current_gaussian_appearance_report(
                gaussian, benchmark_report=benchmark, selected=selected
            )
            self.assertEqual(set(validated), {"item:item_001"})
            gaussian["objects"][0]["object_key"] = "item:item_002"
            with self.assertRaises(RuntimeError):
                validate_current_gaussian_appearance_report(
                    gaussian, benchmark_report=benchmark, selected=selected
                )


if __name__ == "__main__":
    unittest.main()
