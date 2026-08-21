from __future__ import annotations

import unittest

from pose_point_depth_mv.evaluate_objaverse16_slat_oracle_swap import paired_comparison


def row(method: str, object_uid: str, *, chamfer: float, fscore: float, normal: float):
    return {
        "method": method,
        "object_uid": object_uid,
        "surface": {
            "chamfer_l1": chamfer,
            "chamfer_l2": chamfer**2,
            "pred_to_gt_mean": chamfer,
            "gt_to_pred_mean": chamfer,
            "fscore_0p02": fscore,
            "normal_consistency": normal,
        },
    }


class ObjaverseSLatOracleSwapTest(unittest.TestCase):
    def test_advantage_direction_and_gate(self) -> None:
        records = []
        for index in range(8):
            uid = str(index)
            records.append(row("left", uid, chamfer=0.1, fscore=0.7, normal=0.8))
            records.append(row("right", uid, chamfer=0.2, fscore=0.5, normal=0.6))
        result = paired_comparison(records, left="left", right="right")
        self.assertEqual(result["decision"], "exploratory_advantage_supported")
        self.assertAlmostEqual(
            result["metric_deltas"]["chamfer_l1_left_improvement"]["mean"], 0.1
        )
        self.assertAlmostEqual(
            result["metric_deltas"]["fscore_0p02_left_improvement"]["mean"], 0.2
        )

    def test_mixed_directions_do_not_claim_advantage(self) -> None:
        records = []
        for index in range(8):
            uid = str(index)
            left_chamfer = 0.1 if index < 4 else 0.3
            records.append(
                row("left", uid, chamfer=left_chamfer, fscore=0.6, normal=0.6)
            )
            records.append(row("right", uid, chamfer=0.2, fscore=0.6, normal=0.6))
        result = paired_comparison(records, left="left", right="right")
        self.assertEqual(result["decision"], "advantage_not_established")


if __name__ == "__main__":
    unittest.main()
