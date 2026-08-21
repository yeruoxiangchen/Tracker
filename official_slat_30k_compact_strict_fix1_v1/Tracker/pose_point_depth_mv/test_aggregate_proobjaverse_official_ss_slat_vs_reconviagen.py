from __future__ import annotations

import unittest

from pose_point_depth_mv.aggregate_proobjaverse_official_ss_slat_vs_reconviagen import (
    _validate_current_failure,
    make_parser,
    paired_route_summary,
)


class PairedRouteSummaryTest(unittest.TestCase):
    def test_positive_means_candidate_better_for_lower_and_higher_metrics(self):
        candidate = {
            "u0": {"chamfer_l1": 1.0, "fscore_0p02": 0.8},
            "u1": {"chamfer_l1": 2.0, "fscore_0p02": 0.7},
        }
        baseline = {
            "u0": {"chamfer_l1": 2.0, "fscore_0p02": 0.6},
            "u1": {"chamfer_l1": 4.0, "fscore_0p02": 0.5},
        }
        result = paired_route_summary(
            candidate,
            baseline,
            candidate_name="candidate",
            baseline_name="baseline",
            bootstrap_samples=100,
            bootstrap_seed=7,
            comparison_kind="test",
            clean_component_isolation=True,
            caveat="test",
        )
        self.assertEqual(result["metric_deltas"]["chamfer_l1"]["mean"], 1.5)
        self.assertAlmostEqual(
            result["metric_deltas"]["fscore_0p02"]["mean"], 0.2
        )
        self.assertEqual(
            result["metric_deltas"]["chamfer_l1"]["delta_formula"],
            "baseline_minus_candidate",
        )
        self.assertEqual(
            result["metric_deltas"]["fscore_0p02"]["delta_formula"],
            "candidate_minus_baseline",
        )

    def test_object_set_mismatch_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "object sets differ"):
            paired_route_summary(
                {"u0": {"chamfer_l1": 1.0}},
                {"u1": {"chamfer_l1": 1.0}},
                candidate_name="candidate",
                baseline_name="baseline",
                bootstrap_samples=10,
                bootstrap_seed=7,
                comparison_kind="test",
                clean_component_isolation=True,
                caveat="test",
            )

    def test_metric_set_mismatch_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "metric sets differ"):
            paired_route_summary(
                {"u0": {"chamfer_l1": 1.0}},
                {"u0": {"fscore_0p02": 1.0}},
                candidate_name="candidate",
                baseline_name="baseline",
                bootstrap_samples=10,
                bootstrap_seed=7,
                comparison_kind="test",
                clean_component_isolation=True,
                caveat="test",
            )


class CurrentFailureContractTest(unittest.TestCase):
    def test_approved_active_point_failure(self):
        _validate_current_failure(
            {
                "branch": "native_trained",
                "object_uid": "uid",
                "seed": 42,
                "passed": False,
                "error": {
                    "type": "RuntimeError",
                    "stage": "native_trained_slat_mesh_decode",
                    "message": (
                        "SLat decoder input exceeds safe active-point limit: "
                        "points=90000 limit=80000"
                    ),
                },
            }
        )

    def test_wrong_failure_stage_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "unapproved failed"):
            _validate_current_failure(
                {
                    "branch": "native_trained",
                    "object_uid": "uid",
                    "seed": 42,
                    "passed": False,
                    "error": {
                        "type": "RuntimeError",
                        "stage": "cuda_oom",
                        "message": "CUDA out of memory",
                    },
                }
            )


class ParserScopeTest(unittest.TestCase):
    def test_parser_has_no_train_or_gt_support_switch(self):
        option_strings = {
            option
            for action in make_parser()._actions
            for option in action.option_strings
        }
        self.assertNotIn("--train_split", option_strings)
        self.assertNotIn("--gt_support", option_strings)
        self.assertIn("--current_reports", option_strings)
        self.assertIn("--recon_reports", option_strings)


if __name__ == "__main__":
    unittest.main()
