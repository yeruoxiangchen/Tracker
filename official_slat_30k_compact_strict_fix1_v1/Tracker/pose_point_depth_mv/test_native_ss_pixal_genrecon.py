from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from pose_point_depth_mv.evaluate_native_ss_pixal_genrecon import (
    METHODS,
    _summaries,
)
from pose_point_depth_mv.infer_official_genrecon_objects import (
    CAMERA_CONVERSION,
    _masked_square_inputs,
    build_parser,
)


class NativeSotaGeometryTests(unittest.TestCase):
    def test_genrecon_cli_binds_distinct_stage_train_configs(self):
        args = build_parser().parse_args(
            [
                "--protocol",
                "protocol.json",
                "--case_id",
                "case0",
                "--output_root",
                "output",
                "--ss_checkpoint",
                "sparse_structure.pt",
                "--shape_checkpoint",
                "shape_slat.pt",
                "--ss_train_config",
                "ss_config.json",
                "--shape_train_config",
                "shape_config.json",
            ]
        )
        self.assertEqual(args.ss_train_config, Path("ss_config.json"))
        self.assertEqual(args.shape_train_config, Path("shape_config.json"))

    def test_camera_conversion_is_versioned(self):
        self.assertIn("inverse(blender_c2w)", CAMERA_CONVERSION)
        self.assertIn("identity_chunk0", CAMERA_CONVERSION)

    def test_masked_square_input_updates_intrinsic(self):
        with tempfile.TemporaryDirectory() as root_value:
            root = Path(root_value)
            rgb = np.full((80, 100, 3), 255, dtype=np.uint8)
            mask = np.zeros((80, 100), dtype=np.uint8)
            mask[20:60, 30:70] = 255
            Image.fromarray(rgb).save(root / "image.png")
            Image.fromarray(mask).save(root / "mask.png")
            k = torch.tensor([[[50.0, 0.0, 50.0], [0.0, 50.0, 40.0], [0.0, 0.0, 1.0]]])
            intr, image512, image1024, audit = _masked_square_inputs(
                [str(root / "image.png")], [str(root / "mask.png")], k
            )
            self.assertEqual(tuple(intr.shape), (1, 3, 3))
            self.assertEqual(tuple(image512.shape), (1, 3, 512, 512))
            self.assertEqual(tuple(image1024.shape), (1, 3, 1024, 1024))
            self.assertEqual(audit[0]["crop_left_top_size"], [10, 0, 80])
            self.assertAlmostEqual(float(intr[0, 0, 2]), 0.5)
            self.assertAlmostEqual(float(intr[0, 1, 2]), 0.5)
            self.assertGreater(float(image512.max()), 0.9)
            self.assertEqual(float(image512[:, :, 0, 0].max()), 0.0)

    def test_comparison_sign_is_native_better(self):
        records = []
        metric_names = (
            "chamfer_l1",
            "pred_to_gt_mean",
            "gt_to_pred_mean",
            "fscore_0p01",
            "fscore_0p02",
            "fscore_0p05",
            "normal_consistency",
            "precision_0p02",
            "recall_0p02",
            "largest_component_ratio",
        )
        for _ in range(2):
            methods = {}
            for method in METHODS:
                is_native = method == "native_full"
                methods[method] = {
                    metric: (0.1 if metric.endswith("mean") or metric == "chamfer_l1" else 0.9)
                    if is_native
                    else (0.2 if metric.endswith("mean") or metric == "chamfer_l1" else 0.8)
                    for metric in metric_names
                }
            records.append({"methods": methods})
        _, comparisons = _summaries(records, bootstrap_samples=20, seed=7)
        for values in comparisons.values():
            self.assertGreater(values["chamfer_l1"]["mean"], 0.0)
            self.assertGreater(values["fscore_0p02"]["mean"], 0.0)


if __name__ == "__main__":
    unittest.main()
