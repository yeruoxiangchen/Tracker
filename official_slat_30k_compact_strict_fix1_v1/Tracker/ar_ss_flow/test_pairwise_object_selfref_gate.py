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
    "pairwise_object_selfref_gate", HERE / "pairwise_object_selfref_gate.py"
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class ObjectSelfRefGateTest(unittest.TestCase):
    def test_parse_statistics_and_reducers(self) -> None:
        stats = MODULE.parse_statistics("mean,median,trimmed_mean_10,top20_mean")
        self.assertEqual([row.name for row in stats], ["mean", "median", "trimmed_mean_10", "top20_mean"])
        self.assertEqual(MODULE.parse_reference_reducers("median,mean,max"), ["median", "mean", "max"])

    def test_object_score(self) -> None:
        confidence = np.asarray([0.1, 0.3, 0.9, 10.0])
        valid = np.asarray([1, 1, 1, 0], dtype=bool)
        mean_stat, median_stat = MODULE.parse_statistics("mean,median")
        self.assertAlmostEqual(MODULE.object_score(confidence, valid, mean_stat), 1.3 / 3.0)
        self.assertAlmostEqual(MODULE.object_score(confidence, valid, median_stat), 0.3)

    def test_contrastive_median(self) -> None:
        config = MODULE.SelfReferenceConfig(
            statistic=MODULE.parse_statistics("mean")[0], reference_reducer="median"
        )
        confidence = np.asarray(
            [[0.8, 0.8], [0.4, 0.4], [0.5, 0.5], [0.6, 0.6]], dtype=np.float64
        )
        result = MODULE.contrastive_object_score(
            confidence, np.ones(2, dtype=bool), config, min_valid_voxels=2
        )
        self.assertAlmostEqual(result["observed_score"], 0.8)
        self.assertAlmostEqual(result["reference_score"], 0.5)
        self.assertAlmostEqual(result["score"], 0.3)

    def test_contrastive_max_is_conservative(self) -> None:
        statistic = MODULE.parse_statistics("mean")[0]
        confidence = np.asarray([[0.8], [0.4], [0.5], [0.6]])
        median = MODULE.contrastive_object_score(
            confidence,
            np.ones(1, dtype=bool),
            MODULE.SelfReferenceConfig(statistic, "median"),
        )["score"]
        maximum = MODULE.contrastive_object_score(
            confidence,
            np.ones(1, dtype=bool),
            MODULE.SelfReferenceConfig(statistic, "max"),
        )["score"]
        self.assertGreater(median, maximum)
        self.assertAlmostEqual(maximum, 0.2)

    def test_wrong_hypothesis_can_be_negative(self) -> None:
        config = MODULE.SelfReferenceConfig(MODULE.parse_statistics("mean")[0], "median")
        confidence = np.asarray([[0.5], [0.8], [0.4], [0.6]])
        result = MODULE.contrastive_object_score(confidence, np.ones(1, dtype=bool), config)
        self.assertLess(result["score"], 0.0)

    def test_binary_auc(self) -> None:
        self.assertAlmostEqual(MODULE.binary_auc(np.asarray([0.8, 0.9]), np.asarray([0.1, 0.2])), 1.0)
        self.assertAlmostEqual(MODULE.binary_auc(np.asarray([0.5]), np.asarray([0.5])), 0.5)

    def test_threshold_and_gate(self) -> None:
        row = MODULE.calibrate_thresholds(
            np.linspace(0.1, 0.4, 40),
            np.linspace(-0.3, 0.15, 80),
            min_correct_coverage=0.5,
            max_correct_coverage=0.95,
            tau_high_quantile=0.9,
            target_correct_coverage=0.9,
        )
        self.assertGreater(row["tau_high"], row["tau_low"])
        gates = MODULE.scalar_gate(
            np.asarray([row["tau_low"] - 1.0, row["tau_low"], row["tau_high"], row["tau_high"] + 1.0]),
            row["tau_low"], row["tau_high"],
        )
        np.testing.assert_allclose(gates, np.asarray([0.0, 0.0, 1.0, 1.0]))

    def test_compute_gate_from_calibration(self) -> None:
        config = MODULE.SelfReferenceConfig(MODULE.parse_statistics("mean")[0], "median")
        calibration = {
            "config": config.to_dict(),
            "minimum_valid_voxels": 2,
            "tau_low": 0.0,
            "tau_high": 0.4,
        }
        confidence = np.asarray(
            [[0.8, 0.8], [0.4, 0.4], [0.5, 0.5], [0.6, 0.6]], dtype=np.float64
        )
        score, gate = MODULE.compute_gate_from_calibration(
            confidence, np.ones(2, dtype=bool), calibration
        )
        self.assertAlmostEqual(score, 0.3)
        self.assertAlmostEqual(gate, 0.75)


if __name__ == "__main__":
    unittest.main()
