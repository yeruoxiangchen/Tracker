#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "deployment_summary",
    HERE / "summarize_pairwise_deployment_calibration.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class DeploymentCalibrationTest(unittest.TestCase):
    def test_binary_auc(self) -> None:
        self.assertAlmostEqual(
            MODULE.binary_auc(np.array([0.8, 0.9]), np.array([0.1, 0.2])),
            1.0,
        )
        self.assertAlmostEqual(
            MODULE.binary_auc(np.array([0.5, 0.5]), np.array([0.5, 0.5])),
            0.5,
        )

    def test_threshold_has_bounded_correct_coverage(self) -> None:
        positive = np.linspace(0.4, 1.0, 100)
        negative = np.linspace(0.0, 0.6, 100)
        row = MODULE.choose_threshold(
            positive,
            negative,
            min_coverage=0.2,
            max_coverage=0.8,
        )
        self.assertGreaterEqual(row["train_correct_coverage"], 0.2)
        self.assertLessEqual(row["train_correct_coverage"], 0.8)
        self.assertGreater(row["train_youden"], 0.0)

    def test_top30_uplift(self) -> None:
        confidence = np.linspace(0.0, 1.0, 100)
        advantage = confidence.copy()
        row = MODULE.top30_uplift(confidence, advantage)
        self.assertGreater(row["top30_mean"], row["overall_mean"])
        self.assertGreater(row["absolute_uplift"], 0.0)


if __name__ == "__main__":
    unittest.main()
