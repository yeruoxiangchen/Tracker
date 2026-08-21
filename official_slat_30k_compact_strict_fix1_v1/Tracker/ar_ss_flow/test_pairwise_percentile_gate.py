#!/usr/bin/env python3
from __future__ import annotations

import unittest

import numpy as np

from ar_ss_flow.pairwise_percentile_gate import (
    SelectorMeans,
    aggregate_selector_means,
    evaluate_object_selectors,
    parse_fractions,
    percentile_rank_gate,
    stable_top_fraction_mask,
)


class PercentileGateTest(unittest.TestCase):
    def test_parse_fractions(self) -> None:
        self.assertEqual(parse_fractions("0.2,0.3,0.4"), (0.2, 0.3, 0.4))
        with self.assertRaises(ValueError):
            parse_fractions("0,0.3")
        with self.assertRaises(ValueError):
            parse_fractions("0.3,0.3")

    def test_stable_top_fraction_exact_count(self) -> None:
        values = np.asarray([0.1, 0.8, 0.5, 0.9, 0.2], dtype=np.float32)
        valid = np.asarray([1, 1, 1, 1, 0], dtype=bool)
        mask = stable_top_fraction_mask(values, valid, 0.5)
        self.assertEqual(int(mask.sum()), 2)
        self.assertTrue(mask[1])
        self.assertTrue(mask[3])
        self.assertFalse(mask[4])

    def test_stable_tie_break(self) -> None:
        values = np.ones(5, dtype=np.float32)
        valid = np.ones(5, dtype=bool)
        mask = stable_top_fraction_mask(values, valid, 0.4)
        self.assertEqual(np.flatnonzero(mask).tolist(), [0, 1])

    def test_soft_gate_is_rank_based_and_monotonic_invariant(self) -> None:
        values = np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        valid = np.ones(4, dtype=bool)
        gate = percentile_rank_gate(values, valid, 0.5, soft=True)
        transformed = percentile_rank_gate(values**3 + 17.0, valid, 0.5, soft=True)
        np.testing.assert_allclose(gate, transformed)
        self.assertEqual(float(gate[0]), 0.0)
        self.assertEqual(float(gate[1]), 0.0)
        self.assertGreater(float(gate[3]), float(gate[2]))
        self.assertAlmostEqual(float(gate[3]), 1.0, places=6)

    def test_correct_selector_beats_controls(self) -> None:
        count = 200
        correct = np.linspace(0.0, 1.0, count, dtype=np.float32)
        wrong = correct[::-1].copy()
        shuffle = np.roll(correct, count // 2)
        advantage = correct * 0.04 - 0.01
        shuffle_advantage = correct * 0.05 - 0.01
        valid = np.ones(count, dtype=bool)
        row = evaluate_object_selectors(
            correct_confidence=correct,
            wrong_confidence=wrong,
            shuffle_confidence=shuffle,
            reprojection_advantage=advantage,
            shuffle_reprojection_advantage=shuffle_advantage,
            valid=valid,
            fraction=0.3,
            random_trials=64,
            seed=42,
        )
        self.assertGreater(row.correct_selected_wrong, row.random_selected_wrong)
        self.assertGreater(row.correct_selected_wrong, row.wrong_selected_wrong)
        self.assertGreater(row.correct_selected_shuffle, row.shuffle_selected_shuffle)
        self.assertEqual(row.selected_count, 60)

    def test_aggregate_selector_means_equal_volume_weight(self) -> None:
        first = SelectorMeans(
            overall_wrong=1.0,
            correct_selected_wrong=2.0,
            wrong_selected_wrong=0.0,
            shuffle_selected_wrong=0.0,
            random_selected_wrong=1.0,
            overall_shuffle=1.0,
            correct_selected_shuffle=2.0,
            wrong_selected_shuffle=0.0,
            shuffle_selected_shuffle=0.0,
            random_selected_shuffle=1.0,
            selected_count=10,
            valid_count=100,
        )
        second = SelectorMeans(
            overall_wrong=3.0,
            correct_selected_wrong=4.0,
            wrong_selected_wrong=2.0,
            shuffle_selected_wrong=2.0,
            random_selected_wrong=3.0,
            overall_shuffle=3.0,
            correct_selected_shuffle=4.0,
            wrong_selected_shuffle=2.0,
            shuffle_selected_shuffle=2.0,
            random_selected_shuffle=3.0,
            selected_count=20,
            valid_count=200,
        )
        combined = aggregate_selector_means([first, second])
        self.assertEqual(combined.overall_wrong, 2.0)
        self.assertEqual(combined.correct_selected_wrong, 3.0)
        self.assertEqual(combined.selected_count, 30)
        self.assertEqual(combined.valid_count, 300)


if __name__ == "__main__":
    unittest.main()
