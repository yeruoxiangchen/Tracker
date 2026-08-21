from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from pose_point_depth_mv.dataset_tools.prepare_coarsemodel_real_raw_cache import (
    ADAPTER_FORMAT,
    build_dataset_cache,
    resolve_sparse,
)


class CoarseModelRawAdapterTests(unittest.TestCase):
    def _make_dataset(self, root: Path, *, with_points: bool = True) -> tuple[Path, Path]:
        dataset = root / "capture"
        images = dataset / "images"
        masks = dataset / "masks"
        sparse = dataset / "sparse" / "0"
        images.mkdir(parents=True)
        masks.mkdir(parents=True)
        sparse.mkdir(parents=True)
        for index in range(2):
            name = f"frame_{index:02d}.jpg"
            Image.new("RGB", (32, 24), (40 + index, 80, 120)).save(images / name)
            mask = np.zeros((24, 32), dtype=np.uint8)
            mask[5:20, 8:25] = 255
            Image.fromarray(mask).save(masks / f"frame_{index:02d}.png")
        (sparse / "cameras.txt").write_text(
            "1 PINHOLE 32 24 30 30 16 12\n", encoding="utf-8"
        )
        (sparse / "images.txt").write_text(
            "1 1 0 0 0 0 0 2 1 frame_00.jpg\n"
            "0 0 -1\n"
            "2 1 0 0 0 1 0 2 1 frame_01.jpg\n"
            "0 0 -1\n",
            encoding="utf-8",
        )
        (sparse / "points3D.txt").write_text(
            "1 0 0 0 255 0 0 0.1 1 0 2 0\n" if with_points else "",
            encoding="utf-8",
        )
        models = dataset / "models"
        models.mkdir()
        (models / "capture_norm.obj").write_text(
            "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", encoding="utf-8"
        )
        return dataset, sparse

    def test_build_and_resume_preserve_input_only_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset, sparse = self._make_dataset(root)
            output = root / "output"
            row, reused = build_dataset_cache(
                dataset,
                sparse=sparse,
                output_dir=output,
                min_registered_pairs=2,
                resume=False,
            )
            self.assertFalse(reused)
            self.assertEqual(row["adapter_format"], ADAPTER_FORMAT)
            self.assertEqual(row["registered_pair_count"], 2)
            self.assertEqual(row["sparse_point_count"], 1)
            self.assertFalse(row["training_ready"])
            self.assertEqual(
                row["reference_mesh_role"],
                "visual_reference_only_never_consumed_by_model",
            )
            with np.load(row["cache_npz"], allow_pickle=False) as cache:
                self.assertEqual(cache["T_W2C"].shape, (2, 4, 4))
                self.assertEqual(cache["K"].shape, (2, 3, 3))
                self.assertEqual(cache["P_W"].shape, (1, 3))
                self.assertEqual(cache["frame_name"].tolist(), ["view_0000.png", "view_0001.png"])
            metadata = json.loads(
                (Path(row["object_root"]) / "raw_cache.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metadata["source_binding"]["sha256"], row["source_binding"]["sha256"])
            resumed, reused = build_dataset_cache(
                dataset,
                sparse=sparse,
                output_dir=output,
                min_registered_pairs=2,
                resume=True,
            )
            self.assertTrue(reused)
            self.assertEqual(resumed["cache_npz"], row["cache_npz"])

    def test_empty_sparse_points_are_not_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            dataset, sparse = self._make_dataset(Path(temporary), with_points=False)
            with self.assertRaises(FileNotFoundError):
                resolve_sparse(dataset, sparse)

    def test_all_image_mode_completes_only_unregistered_phone_pose(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "capture_all"
            images = dataset / "images"
            masks = dataset / "masks"
            sparse = dataset / "sparse" / "0"
            images.mkdir(parents=True)
            masks.mkdir(parents=True)
            sparse.mkdir(parents=True)
            unity_positions = [
                (0.0, 0.0, 0.0),
                (0.2, 0.0, 0.1),
                (0.0, 0.3, 0.2),
                (-0.2, 0.1, 0.4),
                (0.1, -0.2, 0.6),
                (0.3, 0.2, 0.8),
            ]
            pose_lines = []
            image_lines = []
            rotation_w2c = np.diag([1.0, -1.0, -1.0])
            translation_world = np.asarray([1.0, 2.0, 3.0])
            for index, position in enumerate(unity_positions):
                name = f"frame_{index:04d}.jpg"
                Image.new("RGB", (32, 24), (30 + index, 80, 120)).save(images / name)
                mask = np.zeros((24, 32), dtype=np.uint8)
                mask[4:20, 7:26] = 255
                Image.fromarray(mask).save(masks / f"frame_{index:04d}.png")
                pose_lines.append(
                    f"{name},{position[0]},{position[1]},{position[2]},0,0,0\n"
                )
                if index < 5:
                    phone_center = np.asarray([position[0], position[1], -position[2]])
                    colmap_center = 2.0 * phone_center + translation_world
                    tvec = -rotation_w2c @ colmap_center
                    image_lines.extend(
                        [
                            f"{index + 1} 0 1 0 0 {tvec[0]} {tvec[1]} {tvec[2]} 1 {name}\n",
                            "0 0 -1\n",
                        ]
                    )
            (dataset / "poses.txt").write_text("".join(pose_lines), encoding="utf-8")
            (sparse / "cameras.txt").write_text(
                "1 PINHOLE 32 24 30 30 16 12\n", encoding="utf-8"
            )
            (sparse / "images.txt").write_text("".join(image_lines), encoding="utf-8")
            (sparse / "points3D.txt").write_text(
                "1 1 2 3 255 0 0 0.1 1 0 2 0\n", encoding="utf-8"
            )
            row, reused = build_dataset_cache(
                dataset,
                sparse=sparse,
                output_dir=root / "output",
                min_registered_pairs=5,
                resume=False,
                include_all_images=True,
            )
            self.assertFalse(reused)
            self.assertEqual(row["registered_pair_count"], 5)
            self.assertEqual(row["input_view_count"], 6)
            self.assertEqual(row["phone_pose_augmented_count"], 1)
            self.assertTrue(row["pose_completion_audit"]["passed"])
            with np.load(row["cache_npz"], allow_pickle=False) as cache:
                self.assertEqual(cache["T_W2C"].shape, (6, 4, 4))
                expected_center = 2.0 * np.asarray([0.3, 0.2, -0.8]) + translation_world
                actual = cache["T_W2C"][5]
                actual_center = -actual[:3, :3].T @ actual[:3, 3]
                np.testing.assert_allclose(actual_center, expected_center, atol=1.0e-8)


if __name__ == "__main__":
    unittest.main()
