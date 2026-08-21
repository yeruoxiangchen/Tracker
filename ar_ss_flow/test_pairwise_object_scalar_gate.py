#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

SPEC = importlib.util.spec_from_file_location(
    "pairwise_object_scalar_gate",
    HERE / "pairwise_object_scalar_gate.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class ObjectScalarGateTest(unittest.TestCase):
    def test_parse_statistics(self) -> None:
        rows = MODULE.parse_statistics("mean,median,trimmed_mean_10,top20_mean")
        self.assertEqual([row.name for row in rows], [
            "mean", "median", "trimmed_mean_10", "top20_mean"
        ])
        self.assertAlmostEqual(rows[2].fraction, 0.10)
        self.assertAlmostEqual(rows[3].fraction, 0.20)

    def test_object_score_mean_and_median(self) -> None:
        confidence = np.asarray([0.1, 0.3, 0.9, 10.0])
        valid = np.asarray([1, 1, 1, 0], dtype=bool)
        mean_stat, median_stat = MODULE.parse_statistics("mean,median")
        self.assertAlmostEqual(MODULE.object_score(confidence, valid, mean_stat), 1.3 / 3.0)
        self.assertAlmostEqual(MODULE.object_score(confidence, valid, median_stat), 0.3)

    def test_trimmed_mean_rejects_extremes(self) -> None:
        confidence = np.concatenate([np.linspace(0.4, 0.6, 8), [-10.0, 10.0]])
        valid = np.ones(confidence.size, dtype=bool)
        statistic = MODULE.parse_statistics("trimmed_mean_10")[0]
        self.assertAlmostEqual(MODULE.object_score(confidence, valid, statistic), 0.5, places=6)

    def test_top20_mean(self) -> None:
        confidence = np.arange(10, dtype=np.float64)
        valid = np.ones(10, dtype=bool)
        statistic = MODULE.parse_statistics("top20_mean")[0]
        self.assertAlmostEqual(MODULE.object_score(confidence, valid, statistic), 8.5)

    def test_binary_auc(self) -> None:
        self.assertAlmostEqual(
            MODULE.binary_auc(np.asarray([0.8, 0.9]), np.asarray([0.1, 0.2])), 1.0
        )
        self.assertAlmostEqual(
            MODULE.binary_auc(np.asarray([0.5, 0.5]), np.asarray([0.5, 0.5])), 0.5
        )

    def test_threshold_and_gate(self) -> None:
        positive = np.linspace(0.65, 0.85, 40)
        negative = np.linspace(0.35, 0.70, 80)
        row = MODULE.calibrate_thresholds(
            positive,
            negative,
            min_correct_coverage=0.5,
            max_correct_coverage=0.95,
            tau_high_quantile=0.9,
        )
        self.assertGreater(row["tau_high"], row["tau_low"])
        gates = MODULE.scalar_gate(
            np.asarray([row["tau_low"] - 1.0, row["tau_low"], row["tau_high"], row["tau_high"] + 1.0]),
            row["tau_low"],
            row["tau_high"],
        )
        np.testing.assert_allclose(gates, np.asarray([0.0, 0.0, 1.0, 1.0]))

    def test_rank_correlation(self) -> None:
        x = np.arange(10, dtype=np.float64)
        self.assertAlmostEqual(MODULE.rank_correlation(x, x), 1.0)
        self.assertAlmostEqual(MODULE.rank_correlation(x, -x), -1.0)

    def test_compute_gate_from_calibration(self) -> None:
        calibration = {
            "statistic": MODULE.parse_statistics("mean")[0].to_dict(),
            "minimum_valid_voxels": 2,
            "tau_low": 0.2,
            "tau_high": 0.8,
        }
        score, gate = MODULE.compute_gate_from_calibration(
            np.asarray([0.4, 0.6, 9.0]),
            np.asarray([1, 1, 0], dtype=bool),
            calibration,
        )
        self.assertAlmostEqual(score, 0.5)
        self.assertAlmostEqual(gate, 0.5)


if __name__ == "__main__":
    unittest.main()
