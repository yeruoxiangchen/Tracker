from __future__ import annotations

import unittest

from pose_point_depth_mv.evaluate_omni_real_no_vggt_final import no_vggt_decision


def _summary(chamfer_mean, chamfer_median, fscore, normal):
    return {
        "chamfer_l1": {"mean": chamfer_mean, "median": chamfer_median},
        "fscore_0p02": {"mean": fscore},
        "normal_consistency": {"mean": normal},
    }


def _comparison(mean, median, win_rate):
    return {
        "metrics": {
            "chamfer_l1_left_improvement": {
                "mean": mean,
                "median": median,
                "positive_rate": win_rate,
            }
        }
    }


class NoVggtDecisionTest(unittest.TestCase):
    def test_holdout_unlock_uses_frozen_primary_non_regression(self) -> None:
        summaries = {
            "final_native_no_vggt": _summary(0.102, 0.101, 0.48, 0.73),
            "real_adapted_native_v2_full": _summary(0.100, 0.100, 0.50, 0.74),
        }
        decision = no_vggt_decision(
            summaries, _comparison(-0.002, -0.001, 0.44)
        )
        self.assertFalse(decision["superiority_passed"])
        self.assertTrue(decision["primary_non_regression_passed"])
        self.assertTrue(decision["holdout_unlock_passed"])

    def test_holdout_stays_locked_on_chamfer_regression(self) -> None:
        summaries = {
            "final_native_no_vggt": _summary(0.106, 0.104, 0.51, 0.75),
            "real_adapted_native_v2_full": _summary(0.100, 0.100, 0.50, 0.74),
        }
        decision = no_vggt_decision(
            summaries, _comparison(-0.006, -0.004, 0.55)
        )
        self.assertFalse(decision["primary_non_regression_passed"])
        self.assertFalse(decision["holdout_unlock_passed"])


if __name__ == "__main__":
    unittest.main()
