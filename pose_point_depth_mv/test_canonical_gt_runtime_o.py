from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
import trimesh

from pose_point_depth_mv.export_direct_flow_mesh_pairs import load_canonical_gt
from pose_point_depth_mv.omni_real_benchmark_common import sha256_file


class CanonicalGtRuntimeOTest(unittest.TestCase):
    @staticmethod
    def _native_binding(latent_path: Path, mesh_path: Path, margin: float) -> dict:
        return {
            "format": "pose_point_depth_mv.native_objaverse_canonical_normalization.v1",
            "canonical_margin": margin,
            "canonical_latent_frame": "pixal3d_sparse_structure",
            "normalization_policy": "imported_frame_center_scale_v2",
            "ss_latent": str(latent_path),
            "source_glb": str(mesh_path),
            "render_manifest_sha256": "1" * 64,
            "render_inventory_sha256": "2" * 64,
            "dataset_builder_sha256": "3" * 64,
        }

    def test_existing_canonical_normalization_is_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mesh_path = root / "mesh.obj"
            source = trimesh.creation.box(extents=(2.0, 4.0, 6.0))
            source.apply_translation((1.0, 2.0, 3.0))
            source.export(mesh_path)
            latent_path = root / "ss.npz"
            np.savez_compressed(
                latent_path,
                source_glb=np.asarray(str(mesh_path)),
                normalize_center=np.asarray([1.0, 2.0, 3.0]),
                normalize_scale=np.asarray(2.0),
                canonical_margin=np.asarray(0.5),
            )
            loaded, metadata = load_canonical_gt({"ss_latent": str(latent_path)})
            expected = (source.bounds - np.asarray([1.0, 2.0, 3.0])) / 2.0 * 0.5
            np.testing.assert_allclose(loaded.bounds, expected, atol=1.0e-8)
            self.assertEqual(metadata["canonical_margin"], 0.5)
            self.assertEqual(metadata["canonical_margin_source"], "ss_latent")

    def test_bound_native_objaverse_margin_recovers_legacy_latent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mesh_path = root / "mesh.obj"
            source = trimesh.creation.box(extents=(2.0, 4.0, 6.0))
            source.apply_translation((1.0, 2.0, 3.0))
            source.export(mesh_path)
            latent_path = root / "ss.npz"
            np.savez_compressed(
                latent_path,
                source_glb=np.asarray(str(mesh_path)),
                normalize_center=np.asarray([1.0, 2.0, 3.0]),
                normalize_scale=np.asarray(6.0),
            )
            binding = self._native_binding(latent_path, mesh_path, 0.9)
            loaded, metadata = load_canonical_gt(
                {"ss_latent": str(latent_path)},
                canonical_margin_binding=binding,
            )
            expected = (source.bounds - np.asarray([1.0, 2.0, 3.0])) / 6.0 * 0.9
            np.testing.assert_allclose(loaded.bounds, expected, atol=1.0e-8)
            self.assertEqual(metadata["canonical_margin"], 0.9)
            self.assertEqual(
                metadata["canonical_margin_source"],
                "frozen_objaverse_render_manifest",
            )
            self.assertEqual(metadata["canonical_margin_provenance"], binding)

    def test_bound_native_objaverse_margin_rejects_latent_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mesh_path = root / "mesh.obj"
            trimesh.creation.box().export(mesh_path)
            latent_path = root / "ss.npz"
            np.savez_compressed(
                latent_path,
                source_glb=np.asarray(str(mesh_path)),
                normalize_center=np.zeros(3),
                normalize_scale=np.asarray(1.0),
            )
            binding = self._native_binding(root / "different.npz", mesh_path, 0.9)
            with self.assertRaisesRegex(RuntimeError, "binding latent differs"):
                load_canonical_gt(
                    {"ss_latent": str(latent_path)},
                    canonical_margin_binding=binding,
                )

    def test_runtime_o_mesh_is_loaded_without_renormalization(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mesh_path = root / "mesh.obj"
            source = trimesh.creation.box(extents=(0.2, 0.4, 0.6))
            source.apply_translation((0.1, -0.2, 0.05))
            source.export(mesh_path)
            latent_path = root / "ss.npz"
            np.savez_compressed(
                latent_path,
                source_glb=np.asarray(str(mesh_path)),
                repair_format=np.asarray("object_level_ss_repair.v1"),
                repair_target_mode=np.asarray("decoder_projected"),
                coordinate_frame=np.asarray("runtime-O"),
            )
            loaded, metadata = load_canonical_gt(
                {
                    "ss_latent": str(latent_path),
                    "source_glb": str(mesh_path),
                    "source_glb_sha256": sha256_file(mesh_path),
                }
            )
            np.testing.assert_allclose(loaded.bounds, source.bounds, atol=1.0e-8)
            self.assertEqual(metadata["coordinate_frame"], "runtime-O")
            self.assertEqual(metadata["normalization"], "identity")

    def test_unknown_schema_is_not_silently_treated_as_runtime_o(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mesh_path = root / "mesh.obj"
            trimesh.creation.box().export(mesh_path)
            latent_path = root / "ss.npz"
            np.savez_compressed(latent_path, source_glb=np.asarray(str(mesh_path)))
            with self.assertRaisesRegex(ValueError, "not an audited runtime-O target"):
                load_canonical_gt({"ss_latent": str(latent_path)})


if __name__ == "__main__":
    unittest.main()
