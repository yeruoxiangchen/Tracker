import json
import tempfile
import unittest
from pathlib import Path

from pose_point_depth_mv.summarize_direct_slat_checkpoint_curve import (
    parse_step_paths,
    summarize_flow_stats,
    training_window,
)


class DirectSlatCheckpointCurveTest(unittest.TestCase):
    def test_training_window_separates_rollout_rows(self) -> None:
        history = [
            {
                "step": step,
                "gain_vs_stock": float(step),
                "raw_delta_ratio_max": 0.01,
                "raw_delta_excess_loss": 0.0,
                "support_dropout_loss": 0.0,
                "wrong_support_stock_loss": 0.0,
                "rollout_evaluated": step % 2 == 0,
                "rollout_gain_vs_stock": 0.2,
                "rollout_loss": 0.3,
                "rollout_stock_loss": 0.4,
                "endpoint_x0_loss": 0.5,
                "support_dropout_evaluated": False,
                "wrong_support_evaluated": False,
                "delta_clip_activated": False,
                "delta_clip_scale": 1.0,
            }
            for step in range(1, 5)
        ]
        result = training_window(history, 1, 4)
        self.assertEqual(result["micro_steps"], 4)
        self.assertEqual(result["rollout_events"], 2)
        self.assertAlmostEqual(result["means"]["gain_vs_stock"], 2.5)
        self.assertAlmostEqual(result["means"]["endpoint_x0_loss"], 0.5)

    def test_flow_summary_uses_scale_reduction_for_smooth_bound(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "full_flow_stats.json"
            payload = {
                "policy_version": "post_cfg_v2",
                "slat_delta_scale": 1.0,
                "slat_delta_rms_ratio_cap": 0.1,
                "slat_delta_bound_mode": "smooth_rms_v2",
                "support_interval_policy": "cfg_active_only_v1",
                "cfg_strength": 5.0,
                "cfg_interval": [0.5, 1.0],
                "by_timestep": [
                    {
                        "support_active": 1.0,
                        "stock_guided_velocity_rms": 2.0,
                        "raw_guided_delta_rms": 0.4,
                        "effective_guided_delta_rms": 0.2,
                        "guided_delta_clip_scale": 0.5,
                        "guided_delta_clip_activated": 0.0,
                    },
                    {
                        "support_active": 0.0,
                        "stock_guided_velocity_rms": 2.0,
                        "raw_guided_delta_rms": 0.0,
                        "effective_guided_delta_rms": 0.0,
                        "guided_delta_clip_scale": 1.0,
                        "guided_delta_clip_activated": 0.0,
                    },
                ],
            }
            path.write_text(json.dumps(payload), encoding="utf-8")
            result = summarize_flow_stats([path])
        active = result["active_interval"]
        self.assertEqual(active["raw_ratio_over_cap_calls"], 1)
        self.assertEqual(active["smooth_scale_reduced_calls"], 1)
        self.assertEqual(active["official_clip_flag_calls"], 0)
        self.assertAlmostEqual(active["raw_delta_to_stock_rms_mean"], 0.2)
        self.assertAlmostEqual(
            active["effective_delta_to_stock_rms_mean"], 0.1
        )
        self.assertTrue(result["inactive_interval"]["exact_stock_pass"])

    def test_step_path_parser(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "step_000100.pt"
            path.write_bytes(b"x")
            parsed = parse_step_paths([f"100={path}"], "checkpoint")
        self.assertEqual(list(parsed), [100])


if __name__ == "__main__":
    unittest.main()
