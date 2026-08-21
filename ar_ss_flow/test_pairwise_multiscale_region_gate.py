#!/usr/bin/env python3
from __future__ import annotations

import unittest
from pathlib import Path
import sys

import numpy as np

try:
    from ar_ss_flow.pairwise_multiscale_region_gate import (
        binary_auc,
        bounded_region_gate,
        build_candidates,
        percentile_region_gate,
        rank_correlation,
        reduce_regions,
        region_index,
        region_label_means,
        shrink_region_scores,
        trimmed_mean,
        weighted_region_mean,
    )
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from pairwise_multiscale_region_gate import (
        binary_auc,
        bounded_region_gate,
        build_candidates,
        percentile_region_gate,
        rank_correlation,
        reduce_regions,
        region_index,
        region_label_means,
        shrink_region_scores,
        trimmed_mean,
        weighted_region_mean,
    )


class MultiScaleRegionGateTests(unittest.TestCase):
    def test_region_index_counts(self) -> None:
        ids = region_index(16, 2)
        self.assertEqual(ids.shape, (4096,))
        counts = np.bincount(ids)
        self.assertEqual(counts.size, 8)
        self.assertTrue(np.all(counts == 512))

    def test_trimmed_mean_removes_extremes(self) -> None:
        x = np.asarray([-100.0, 1.0, 2.0, 3.0, 100.0])
        self.assertAlmostEqual(trimmed_mean(x, 0.2), 2.0)

    def test_reduce_and_shrink(self) -> None:
        side = 4
        ids = region_index(side, 2)
        values = ids.astype(np.float64)
        valid = np.ones(values.shape, dtype=bool)
        region = reduce_regions(
            values, valid, volume_side=side, divisions=2,
            trim_fraction=0.0, min_region_voxels=1,
        )
        self.assertEqual(int(region.valid.sum()), 8)
        shrunk = shrink_region_scores(region, kappa=1000.0)
        original_spread = float(np.nanmax(region.scores) - np.nanmin(region.scores))
        shrunk_spread = float(np.nanmax(shrunk.scores) - np.nanmin(shrunk.scores))
        self.assertLess(shrunk_spread, original_spread)

    def test_bounded_region_gate(self) -> None:
        gate = bounded_region_gate(np.asarray([-1.0, 0.4, 2.0]), np.ones(3, dtype=bool))
        self.assertTrue(np.allclose(gate, [0.0, 0.4, 1.0]))

    def test_percentile_gate_monotonic(self) -> None:
        scores = np.asarray([3.0, 1.0, 2.0])
        gate = percentile_region_gate(scores, np.ones(3, dtype=bool))
        self.assertGreater(gate[0], gate[2])
        self.assertGreater(gate[2], gate[1])
        self.assertAlmostEqual(float(gate[0]), 1.0)

    def test_weighted_region_mean_prefers_high_label(self) -> None:
        labels = np.asarray([0.0, 1.0])
        counts = np.asarray([10, 10])
        valid = np.ones(2, dtype=bool)
        uniform = weighted_region_mean(labels, np.ones(2), counts, valid)
        gated = weighted_region_mean(labels, np.asarray([0.1, 1.0]), counts, valid)
        self.assertGreater(gated, uniform)

    def test_region_label_means(self) -> None:
        side = 4
        ids = region_index(side, 2)
        labels = ids.astype(np.float64)
        means, counts, valid = region_label_means(
            labels, np.ones(labels.shape, dtype=bool),
            volume_side=side, divisions=2, min_region_voxels=1,
        )
        self.assertTrue(np.all(valid))
        self.assertTrue(np.all(counts == 8))
        self.assertTrue(np.allclose(means, np.arange(8)))

    def test_rank_correlation_and_auc(self) -> None:
        x = np.arange(8, dtype=np.float64)
        self.assertAlmostEqual(rank_correlation(x, x, np.ones(8, dtype=bool)), 1.0)
        self.assertAlmostEqual(binary_auc(np.asarray([2.0, 3.0]), np.asarray([0.0, 1.0])), 1.0)
        self.assertAlmostEqual(binary_auc(np.asarray([1.0]), np.asarray([1.0])), 0.5)

    def test_candidate_list(self) -> None:
        candidates = build_candidates((1, 2, 4), (32, 64))
        names = [x.name for x in candidates]
        self.assertEqual(names[0], "object")
        self.assertIn("octant8_shrink_k32", names)
        self.assertIn("grid64_shrink_k64", names)
        self.assertEqual(len(names), 7)


if __name__ == "__main__":
    unittest.main()
