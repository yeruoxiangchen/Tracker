from __future__ import annotations

import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from pose_point_depth_mv.dataset_tools.audit_omni_real_video_sim3 import (
    apply_similarity,
    umeyama_proper,
)
from pose_point_depth_mv.dataset_tools.prepare_omni_real_video_cache import (
    build_object_cache,
    extract_category,
    inventory_archive,
)


def add_bytes(bundle: tarfile.TarFile, name: str, payload: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    bundle.addfile(info, io.BytesIO(payload))


class OmniRealVideoCacheTest(unittest.TestCase):
    def test_inventory_counts_video_objects_not_scan_objects(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scan_root = root / "scans"
            for object_id in ("cup_001", "cup_002"):
                scan = scan_root / "cup" / object_id / "Scan" / "Scan.obj"
                scan.parent.mkdir(parents=True)
                scan.write_bytes(b"v 0 0 0\n" * 200)
            archive = root / "cup.tar.gz"
            prefix = "release/cup/cup_001/standard"
            with tarfile.open(archive, "w:gz") as bundle:
                add_bytes(bundle, f"{prefix}/images/00000.jpg", b"image")
                add_bytes(bundle, f"{prefix}/matting/00000.jpg", b"mask")
                add_bytes(bundle, f"{prefix}/sparse/0_txt/cameras.txt", b"camera")
                add_bytes(bundle, f"{prefix}/sparse/0_txt/images.txt", b"images")
                add_bytes(bundle, f"{prefix}/sparse/0_txt/points3D.txt", b"points")
                add_bytes(bundle, f"{prefix}/poses_bounds.npy", b"npy")

            report = inventory_archive(archive, "cup", scan_root)

            self.assertTrue(report["passed"])
            self.assertEqual(report["video_object_count"], 1)
            self.assertEqual(report["objects"][0]["object_id"], "cup_001")

    def test_raw_cache_preserves_simple_radial_distortion(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "cup_001"
            standard = root / "standard"
            images = standard / "images"
            masks = standard / "matting"
            sparse = standard / "sparse" / "0_txt"
            images.mkdir(parents=True)
            masks.mkdir(parents=True)
            sparse.mkdir(parents=True)
            Image.new("RGB", (8, 6), "white").save(images / "00000.jpg")
            Image.new("L", (8, 6), 255).save(masks / "00000.jpg")
            (sparse / "cameras.txt").write_text(
                "1 SIMPLE_RADIAL 8 6 10 4 3 0.125\n", encoding="utf-8"
            )
            (sparse / "images.txt").write_text(
                "1 1 0 0 0 0 0 2 1 00000.jpg\n4 3 7\n", encoding="utf-8"
            )
            (sparse / "points3D.txt").write_text(
                "7 0 0 0 1 2 3 0.25 1 0\n", encoding="utf-8"
            )
            np.save(standard / "poses_bounds.npy", np.zeros((1, 17)))
            scan = Path(directory) / "Scan.obj"
            scan.write_bytes(b"v 0 0 0\n" * 200)

            report = build_object_cache(
                root,
                {"category": "cup", "object_id": "cup_001", "scan_obj": str(scan)},
            )

            self.assertEqual(report["camera_models"], ["SIMPLE_RADIAL"])
            self.assertEqual(report["cameras"][0]["distortion"], [0.125])
            with np.load(report["cache_npz"], allow_pickle=False) as cache:
                self.assertEqual(cache["K"].shape, (1, 3, 3))
                self.assertEqual(cache["T_W2C"].shape, (1, 4, 4))
                self.assertAlmostEqual(float(cache["T_W2C"][0, 2, 3]), 2.0)

    def test_category_extraction_is_atomic_and_reusable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "cup.tar.gz"
            image_buffer = io.BytesIO()
            Image.new("RGB", (8, 6), "white").save(image_buffer, format="JPEG")
            mask_buffer = io.BytesIO()
            Image.new("L", (8, 6), 255).save(mask_buffer, format="JPEG")
            poses_buffer = io.BytesIO()
            np.save(poses_buffer, np.zeros((1, 17)))
            prefix = "release/cup/cup_001/standard"
            with tarfile.open(archive, "w:gz") as bundle:
                add_bytes(bundle, f"{prefix}/images/00000.jpg", image_buffer.getvalue())
                add_bytes(bundle, f"{prefix}/matting/00000.jpg", mask_buffer.getvalue())
                add_bytes(
                    bundle,
                    f"{prefix}/sparse/0_txt/cameras.txt",
                    b"1 SIMPLE_RADIAL 8 6 10 4 3 0.125\n",
                )
                add_bytes(
                    bundle,
                    f"{prefix}/sparse/0_txt/images.txt",
                    b"1 1 0 0 0 0 0 2 1 00000.jpg\n4 3 7\n",
                )
                add_bytes(
                    bundle,
                    f"{prefix}/sparse/0_txt/points3D.txt",
                    b"7 0 0 0 1 2 3 0.25 1 0\n",
                )
                add_bytes(bundle, f"{prefix}/poses_bounds.npy", poses_buffer.getvalue())
            source_row = {
                "category": "cup",
                "object_id": "cup_001",
                "scan_obj": str(root / "scan.obj"),
            }
            category_row = {
                "category": "cup",
                "archive": str(archive),
                "objects": [source_row],
            }
            output = root / "output"

            first, first_reused = extract_category(category_row, output, "inventory-sha")
            second, second_reused = extract_category(category_row, output, "inventory-sha")

            self.assertFalse(first_reused)
            self.assertTrue(second_reused)
            self.assertEqual(first[0]["metadata"], second[0]["metadata"])
            self.assertTrue(Path(first[0]["cache_npz"]).is_file())
            self.assertTrue(
                (output / "raw_objects" / "cup" / "_CATEGORY_COMPLETE.json").is_file()
            )

    def test_umeyama_recovers_proper_similarity(self) -> None:
        rng = np.random.default_rng(7)
        source = rng.normal(size=(100, 3))
        rotation = np.asarray(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        target = apply_similarity(
            source, 2.5, rotation, np.asarray([3.0, -4.0, 1.5])
        )

        scale, estimate_rotation, translation = umeyama_proper(source, target)
        recovered = apply_similarity(source, scale, estimate_rotation, translation)

        self.assertAlmostEqual(scale, 2.5, places=8)
        self.assertGreater(np.linalg.det(estimate_rotation), 0.999999)
        self.assertLess(float(np.max(np.abs(recovered - target))), 1.0e-8)


if __name__ == "__main__":
    unittest.main()
