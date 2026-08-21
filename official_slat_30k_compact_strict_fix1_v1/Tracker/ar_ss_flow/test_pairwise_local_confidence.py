#!/usr/bin/env python3
from __future__ import annotations

import unittest

import numpy as np
import torch

from ar_ss_flow.pairwise_local_confidence import (
    infer_volume_side,
    support_aware_local_mean,
    support_aware_local_topk,
)
from ar_ss_flow.summarize_pairwise_local_calibration import (
    binary_auc,
    choose_threshold,
    select_method,
)


class PairwiseLocalConfidenceTest(unittest.TestCase):
    def test_infer_volume_side(self) -> None:
        self.assertEqual(infer_volume_side(16**3), 16)
        with self.assertRaises(ValueError):
            infer_volume_side(100)

    def test_support_aware_local_mean(self) -> None:
        values = torch.zeros(1, 27)
        support = torch.zeros(1, 27)
        center = 13
        neighbor = 14
        values[0, center] = 0.2
        values[0, neighbor] = 0.8
        support[0, center] = 1.0
        support[0, neighbor] = 1.0
        result = support_aware_local_mean(values, support, radius=1)
        self.assertAlmostEqual(float(result[0, center]), 0.5, places=6)

    def test_support_aware_local_topk(self) -> None:
        values = torch.zeros(1, 27)
        support = torch.zeros(1, 27)
        center = 13
        values[0, center] = 0.2
        values[0, 14] = 0.8
        values[0, 12] = 0.6
        support[0, center] = 1.0
        support[0, 14] = 1.0
        support[0, 12] = 1.0
        result = support_aware_local_topk(values, support, radius=1, topk=2)
        self.assertAlmostEqual(float(result[0, center]), 0.7, places=6)

    def test_binary_auc_and_threshold(self) -> None:
        positive = np.linspace(0.5, 1.0, 100)
        negative = np.linspace(0.0, 0.5, 100)
        self.assertGreater(binary_auc(positive, negative), 0.99)
        row = choose_threshold(
            positive,
            negative,
            min_coverage=0.2,
            max_coverage=0.8,
        )
        self.assertGreaterEqual(row["train_correct_coverage"], 0.2)
        self.assertLessEqual(row["train_correct_coverage"], 0.8)

    def test_method_selection_uses_train_score(self) -> None:
        rows = {
            "raw": {"selection_score": 0.65},
            "local_mean": {"selection_score": 0.72},
            "local_topk": {"selection_score": 0.70},
        }
        self.assertEqual(select_method(rows), "local_mean")
        tied = {
            "raw": {"selection_score": 0.70},
            "local_mean": {"selection_score": 0.70},
        }
        self.assertEqual(select_method(tied), "raw")


if __name__ == "__main__":
    unittest.main()
