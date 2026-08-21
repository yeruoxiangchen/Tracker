from __future__ import annotations

import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

import numpy as np
from PIL import Image

from pose_point_depth_mv.prepare_proobjaverse_official_slat_dino_cache import (
    _front_sign,
    _load_views,
    _load_views_with_audit,
    _pose_diverse_indices,
)
from pose_point_depth_mv.proobjaverse_official_slat_protocol import (
    DEFAULT_SPLITS,
    _audit_slat,
    _audit_tar,
    split_counts_for_train,
)
from pose_point_depth_mv.summarize_proobjaverse_official_slat_arms import positive_gate


class ProObjaverseOfficialSLatProtocolTests(unittest.TestCase):
    def test_fixed_split_is_exactly_2000(self) -> None:
        self.assertEqual(sum(DEFAULT_SPLITS.values()), 2000)
        self.assertEqual(DEFAULT_SPLITS["train"], 1872)

    def test_train2000_keeps_all_reserved_splits_disjoint(self) -> None:
        splits = split_counts_for_train(2000)
        self.assertEqual(splits["train"], 2000)
        self.assertEqual(splits["dev"], 64)
        self.assertEqual(splits["predicted_support_bridge"], 32)
        self.assertEqual(splits["decoder_audit"], 32)
        self.assertEqual(sum(splits.values()), 2128)

    def test_official_slat_schema(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "target.npz"
            np.savez_compressed(
                path,
                coords=np.asarray(((0, 1, 2), (63, 62, 61)), dtype=np.uint8),
                feats=np.ones((2, 8), dtype=np.float32),
            )
            result = _audit_slat(path)
            self.assertEqual(result["point_count"], 2)
            self.assertEqual(result["coord_max"], 63)

    def test_official_render_tar_schema(self) -> None:
        uid = "abc"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / f"{uid}.tar"
            meta = {
                "intrinsic": [[100.0, 0.0, 32.0], [0.0, 100.0, 32.0], [0.0, 0.0, 1.0]],
                "extrinsic": [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, -2.0], [0.0, 0.0, 0.0, 1.0]],
            }
            image_buffer = io.BytesIO()
            Image.new("RGBA", (64, 64), (255, 0, 0, 255)).save(
                image_buffer, format="WEBP"
            )
            with tarfile.open(path, "w") as archive:
                for name, payload in (
                    (f"{uid}/000.json", json.dumps(meta).encode("utf-8")),
                    (f"{uid}/000.rgba.webp", image_buffer.getvalue()),
                ):
                    info = tarfile.TarInfo(name)
                    info.size = len(payload)
                    archive.addfile(info, io.BytesIO(payload))
            result = _audit_tar(path, uid)
            self.assertEqual(result["view_count"], 1)
            self.assertAlmostEqual(result["camera_radius_min"], 2.0)
            self.assertEqual(result["source_extrinsics_type"], "camera_to_world")
            loaded = _load_views(path, uid)
            self.assertTrue(np.allclose(loaded[0]["center"], (0.0, 0.0, -2.0)))
            self.assertTrue(
                np.allclose(
                    loaded[0]["extrinsic"],
                    np.asarray(
                        (
                            (1.0, 0.0, 0.0, 0.0),
                            (0.0, 1.0, 0.0, 0.0),
                            (0.0, 0.0, 1.0, 2.0),
                            (0.0, 0.0, 0.0, 1.0),
                        ),
                        dtype=np.float32,
                    ),
                )
            )

    def test_pose_diversity_and_forward_sign(self) -> None:
        centers = (
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (-1.0, 0.0, 0.0),
            (0.0, -1.0, 0.0),
        )
        views = [
            {"id": index, "center": np.asarray(center, dtype=np.float64)}
            for index, center in enumerate(centers)
        ]
        selected = _pose_diverse_indices(views, 4, "uid")
        self.assertEqual(len(selected), 4)
        self.assertEqual(len(set(selected)), 4)
        extrinsics = np.repeat(np.eye(4, dtype=np.float32)[None], 4, axis=0)
        extrinsics[:, 2, 3] = -2.0
        self.assertEqual(_front_sign(extrinsics), -1.0)

    def test_trailing_truncated_tar_recovers_only_complete_views(self) -> None:
        uid = "truncated"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / f"{uid}.tar"
            image_buffer = io.BytesIO()
            Image.new("RGBA", (64, 64), (0, 255, 0, 255)).save(
                image_buffer, format="WEBP"
            )
            image_bytes = image_buffer.getvalue()
            with tarfile.open(path, "w") as archive:
                for index in range(10):
                    meta = {
                        "image_index": index,
                        "intrinsic": [[100.0, 0.0, 32.0], [0.0, 100.0, 32.0], [0.0, 0.0, 1.0]],
                        "extrinsic": [[1.0, 0.0, 0.0, float(index + 1)], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, -2.0], [0.0, 0.0, 0.0, 1.0]],
                    }
                    for name, payload in (
                        (f"{uid}/{index:03d}.json", json.dumps(meta).encode("utf-8")),
                        (f"{uid}/{index:03d}.rgba.webp", image_bytes),
                    ):
                        info = tarfile.TarInfo(name)
                        info.size = len(payload)
                        archive.addfile(info, io.BytesIO(payload))
            with tarfile.open(path, "r") as archive:
                last = archive.getmember(f"{uid}/009.rgba.webp")
                truncate_at = last.offset_data + max(1, last.size // 2)
            with path.open("r+b") as handle:
                handle.truncate(truncate_at)
            loaded, audit = _load_views_with_audit(path, uid)
            self.assertEqual(len(loaded), 9)
            self.assertTrue(audit["archive_recovered"])
            self.assertFalse(audit["archive_complete"])
            self.assertIn("unexpected end of data", audit["archive_read_error"])

    def test_train16_fit_gate_requires_mean_median_and_ten_wins(self) -> None:
        def summary(mean: float, median: float, positive_rate: float) -> dict:
            return {
                "chamfer_l1_improvement": {
                    "mean": mean,
                    "median": median,
                    "positive_rate": positive_rate,
                }
            }

        self.assertTrue(positive_gate(summary(1.0e-4, 1.0e-5, 10.0 / 16.0)))
        self.assertFalse(positive_gate(summary(1.0e-4, 1.0e-5, 9.0 / 16.0)))
        self.assertFalse(positive_gate(summary(1.0e-4, -1.0e-5, 10.0 / 16.0)))


if __name__ == "__main__":
    unittest.main()
