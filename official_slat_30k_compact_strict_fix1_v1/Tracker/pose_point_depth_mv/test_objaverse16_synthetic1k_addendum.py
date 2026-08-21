from __future__ import annotations

import unittest

from pose_point_depth_mv.evaluate_objaverse16_synthetic1k_addendum import (
    paired_comparison,
)


def _record(chamfer: float, fscore: float) -> dict:
    return {
        "surface": {
            "chamfer_l1": chamfer,
            "fscore_0p02": fscore,
        }
    }


class Objaverse16Synthetic1kAddendumTest(unittest.TestCase):
    def test_pairing_uses_metric_direction_and_explicit_method_names(self) -> None:
        synthetic = {
            "a": _record(0.10, 0.60),
            "b": _record(0.30, 0.20),
        }
        recon = {
            "a": _record(0.20, 0.50),
            "b": _record(0.20, 0.40),
        }
        result = paired_comparison(
            synthetic,
            recon,
            left_method="synthetic1k_no_vggt",
            right_method="reconviagen_original",
        )
        self.assertEqual(result["left"], "synthetic1k_no_vggt")
        self.assertEqual(result["right"], "reconviagen_original")
        self.assertEqual(
            result["chamfer_l1_wins"],
            {
                "synthetic1k_no_vggt": 1,
                "reconviagen_original": 1,
                "ties": 0,
            },
        )
        chamfer = result["metric_deltas"]["chamfer_l1_left_improvement"]
        fscore = result["metric_deltas"]["fscore_0p02_left_improvement"]
        self.assertAlmostEqual(chamfer["mean"], 0.0)
        self.assertAlmostEqual(fscore["mean"], -0.05)
        self.assertEqual(chamfer["positive_rate"], 0.5)

    def test_pairing_rejects_different_object_sets(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "object sets differ"):
            paired_comparison(
                {"a": _record(0.1, 0.5)},
                {"b": _record(0.1, 0.5)},
                left_method="synthetic1k_no_vggt",
                right_method="reconviagen_original",
            )


if __name__ == "__main__":
    unittest.main()
