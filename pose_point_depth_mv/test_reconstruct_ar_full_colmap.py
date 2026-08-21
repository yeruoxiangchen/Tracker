from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from pose_point_depth_mv.reconstruct_ar_full_colmap import (
    clean_binary_mask,
    filter_track_points,
    parse_images_observations,
    robust_colmap_to_ar_sim3,
)


class FullColmapReconstructionTests(unittest.TestCase):
    def test_clean_mask_removes_speckle_but_keeps_main_object(self):
        mask = np.zeros((40, 50), dtype=np.uint8)
        mask[10:30, 12:38] = 1
        mask[1, 1] = 1
        cleaned = clean_binary_mask(mask) > 0
        self.assertTrue(cleaned[20, 20])
        self.assertFalse(cleaned[1, 1])

    def test_track_filter_uses_only_observed_mask_locations(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "model"
            dataset = root / "dataset"
            output = root / "filtered"
            model.mkdir()
            (dataset / "masks").mkdir(parents=True)
            # Point 10 is observed inside both masks. Point 20 is outside both.
            (model / "images.txt").write_text(
                "# test\n"
                "1 1 0 0 0 0 0 0 1 a.jpg\n"
                "10 10 10 40 40 20\n"
                "2 1 0 0 0 0 0 0 1 b.jpg\n"
                "11 10 10 41 40 20\n",
                encoding="utf-8",
            )
            (model / "points3D.txt").write_text(
                "10 0 0 1 255 0 0 0.5 1 0 2 0\n"
                "20 1 0 1 0 255 0 0.5 1 1 2 1\n",
                encoding="utf-8",
            )
            (model / "cameras.txt").write_text(
                "1 PINHOLE 64 64 50 50 32 32\n", encoding="utf-8"
            )
            mask = np.zeros((64, 64), dtype=np.uint8)
            mask[5:20, 5:20] = 255
            Image.fromarray(mask).save(dataset / "masks/a.png")
            Image.fromarray(mask).save(dataset / "masks/b.png")

            report = filter_track_points(
                text_model=model,
                mask_dataset=dataset,
                output_sparse=output,
                min_observations=2,
                min_positive=2,
                min_ratio=0.6,
                max_error=4.0,
                mask_dilation=0,
                min_points=1,
            )
            self.assertEqual(report["kept_point_count"], 1)
            self.assertTrue((output / "points3D.txt").read_text().splitlines()[1].startswith("10 "))
            rows = parse_images_observations(output / "images_distorted.txt")
            self.assertEqual(rows[0]["observations"][0][2], 10)
            self.assertEqual(rows[0]["observations"][1][2], -1)

    def test_colmap_alignment_targets_raw_unity_world_positions(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "model"
            model.mkdir()
            source = np.asarray(
                [
                    [0, 0, 0],
                    [1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1],
                    [1, 1, 0],
                    [1, 0, 1],
                    [0, 1, 1],
                    [1, 1, 1],
                ],
                dtype=np.float64,
            )
            scale = 0.25
            translation = np.asarray([2.0, -1.0, 0.5])
            image_lines = []
            pose_lines = []
            for index, center in enumerate(source):
                name = f"frame_{index:04d}.jpg"
                # Identity W2C => t=-camera_center.
                image_lines.extend(
                    [
                        f"{index+1} 1 0 0 0 {-center[0]} {-center[1]} {-center[2]} 1 {name}",
                        "0 0 -1",
                    ]
                )
                target = scale * center + translation
                pose_lines.append(
                    f"{name},{target[0]},{target[1]},{target[2]},0,0,0,0,0,0,1"
                )
            (model / "images.txt").write_text(
                "\n".join(image_lines) + "\n", encoding="utf-8"
            )
            poses = root / "poses.txt"
            poses.write_text("\n".join(pose_lines) + "\n", encoding="utf-8")
            report = robust_colmap_to_ar_sim3(model, poses)
            self.assertTrue(report["passed"])
            self.assertAlmostEqual(report["scale"], scale, places=8)
            self.assertLess(report["center_error_ar"]["maximum"], 1.0e-8)


if __name__ == "__main__":
    unittest.main()
