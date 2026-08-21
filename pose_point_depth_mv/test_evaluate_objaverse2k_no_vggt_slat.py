#!/usr/bin/env python3

import copy
import unittest

from pose_point_depth_mv.evaluate_objaverse2k_no_vggt_slat import (
    aggregate_reports,
    paired_improvement,
    select_worker_matrix,
)


def branch(mesh_sha: str, chamfer: float, fscore: float) -> dict:
    return {
        "mesh_sha256": mesh_sha,
        "surface": {
            "chamfer_l1": chamfer,
            "chamfer_l2": chamfer * chamfer,
            "fscore_0p01": fscore - 0.1,
            "fscore_0p02": fscore,
            "fscore_0p05": fscore + 0.1,
            "normal_consistency": fscore,
        },
        "structure": {
            "largest_component_ratio": fscore,
            "component_count": 2,
        },
    }


class Objaverse2KSLatEvaluationTest(unittest.TestCase):
    def test_worker_matrix_uses_requested_prefix_and_partitions_it(self) -> None:
        rows = []
        for object_index in range(20):
            for seed in (42, 43, 44):
                rows.append(
                    {
                        "object_uid": f"object_{object_index:02d}",
                        "uid": f"object_{object_index:02d}_seq000",
                        "support_seed": seed,
                    }
                )
        selected, start, end = select_worker_matrix(
            rows,
            seeds=[42, 43, 44],
            worker_index=3,
            num_workers=4,
            expected_objects=16,
        )
        self.assertEqual((start, end), (12, 16))
        self.assertEqual(len(selected), 4)
        self.assertEqual(selected[0][0], "object_12")
        self.assertEqual(selected[-1][0], "object_15")

    def test_training_matrix_uses_repeatable_hashed_object_selection(self) -> None:
        rows = [
            {
                "object_uid": f"object_{object_index:02d}",
                "uid": f"object_{object_index:02d}_seq000",
                "support_seed": 42,
            }
            for object_index in range(20)
        ]
        selected_a, start_a, end_a = select_worker_matrix(
            rows,
            seeds=[42],
            worker_index=0,
            num_workers=2,
            expected_objects=8,
            object_selection_seed=20260813,
        )
        selected_b, start_b, end_b = select_worker_matrix(
            list(reversed(rows)),
            seeds=[42],
            worker_index=0,
            num_workers=2,
            expected_objects=8,
            object_selection_seed=20260813,
        )
        self.assertEqual((start_a, end_a), (0, 4))
        self.assertEqual((start_b, end_b), (0, 4))
        self.assertEqual(
            [(row[0], row[1]) for row in selected_a],
            [(row[0], row[1]) for row in selected_b],
        )
        self.assertNotEqual([row[0] for row in selected_a], [f"object_{i:02d}" for i in range(4)])

    def _fixtures(self):
        invariant = {
            "cache_manifest_sha256": "cache",
            "lifting_cache_manifest_sha256": "lifting",
            "native_ss_report_sha256": "ss",
            "stock_slat_freeze_sha256": "freeze",
            "sampling": {"steps": 25},
            "joint_seeds": [42],
            "noise_protocol": "matched",
            "noise_seed": 7,
            "surface_samples": 20,
            "expected_objects": 1,
            "num_workers": 1,
        }
        m8_config = {
            **invariant,
            "checkpoint": "/m8.pt",
            "checkpoint_sha256": "m8",
            "checkpoint_step": 2000,
            "weights": "ema",
        }
        candidate_config = {
            **invariant,
            "checkpoint": "/candidate.pt",
            "checkpoint_sha256": "candidate",
            "checkpoint_step": 800,
            "weights": "ema",
        }
        common_identity = {
            "object_uid": "object",
            "uid": "object_seq000",
            "support_seed": 42,
            "master_noise_seed": 123,
            "metric_seed": 456,
            "cache_manifest_sha256": "cache",
        }
        key = ("object", "object_seq000", 42)
        stock = branch("same-stock", 0.30, 0.40)
        m8 = {
            "identity": {
                **common_identity,
                "model_label": "m8",
                "checkpoint_sha256": "m8",
            },
            "branches": {"stock": stock, "full": branch("m8-full", 0.20, 0.50)},
        }
        candidate = {
            "identity": {
                **common_identity,
                "model_label": "objaverse2k",
                "checkpoint_sha256": "candidate",
            },
            "branches": {
                "stock": copy.deepcopy(stock),
                "full": branch("candidate-full", 0.10, 0.60),
            },
        }
        return (
            [{"run_config": m8_config}],
            {key: m8},
            [{"run_config": candidate_config}],
            {key: candidate},
        )

    def test_positive_improvement_uses_metric_direction(self) -> None:
        better = branch("better", 0.1, 0.8)
        worse = branch("worse", 0.2, 0.6)
        improvement = paired_improvement(better, worse)
        self.assertAlmostEqual(improvement["chamfer_l1"], 0.1)
        self.assertAlmostEqual(improvement["fscore_0p02"], 0.2)

    def test_aggregate_builds_stock_m8_candidate_comparison(self) -> None:
        m8_reports, m8_records, candidate_reports, candidate_records = self._fixtures()
        report = aggregate_reports(
            m8_reports=m8_reports,
            m8_records=m8_records,
            candidate_reports=candidate_reports,
            candidate_records=candidate_records,
            bootstrap_samples=20,
        )
        self.assertTrue(report["passed"])
        self.assertEqual(
            report["branches"],
            ["stock", "objaverse2k_stock", "m8", "objaverse2k"],
        )
        self.assertTrue(report["stock_numerical_reproduction"]["passed"])
        self.assertAlmostEqual(
            report["summary"]["objaverse2k_vs_m8"]["chamfer_l1"]["mean"],
            0.1,
        )

    def test_aggregate_accepts_nonidentical_stock_within_numerical_tolerance(self) -> None:
        m8_reports, m8_records, candidate_reports, candidate_records = self._fixtures()
        stock = next(iter(candidate_records.values()))["branches"]["stock"]
        stock["mesh_sha256"] = "different"
        stock["surface"]["chamfer_l1"] += 1.0e-4
        report = aggregate_reports(
            m8_reports=m8_reports,
            m8_records=m8_records,
            candidate_reports=candidate_reports,
            candidate_records=candidate_records,
            bootstrap_samples=20,
        )
        self.assertFalse(report["stock_mesh_exact_match_across_checkpoint_runs"])
        self.assertEqual(
            report["stock_numerical_reproduction"]["mesh_exact_match_count"], 0
        )

    def test_aggregate_rejects_stock_beyond_numerical_tolerance(self) -> None:
        m8_reports, m8_records, candidate_reports, candidate_records = self._fixtures()
        stock = next(iter(candidate_records.values()))["branches"]["stock"]
        stock["surface"]["chamfer_l1"] += 2.0e-3
        with self.assertRaisesRegex(RuntimeError, "exceeds tolerance"):
            aggregate_reports(
                m8_reports=m8_reports,
                m8_records=m8_records,
                candidate_reports=candidate_reports,
                candidate_records=candidate_records,
                bootstrap_samples=20,
            )

    def test_aggregate_records_explicit_stock_tolerance_override(self) -> None:
        m8_reports, m8_records, candidate_reports, candidate_records = self._fixtures()
        stock = next(iter(candidate_records.values()))["branches"]["stock"]
        stock["surface"]["normal_consistency"] += 1.2e-2
        report = aggregate_reports(
            m8_reports=m8_reports,
            m8_records=m8_records,
            candidate_reports=candidate_reports,
            candidate_records=candidate_records,
            bootstrap_samples=20,
            stock_reproduction_tolerances={"normal_consistency": 1.5e-2},
        )
        reproduction = report["stock_numerical_reproduction"]
        self.assertEqual(
            reproduction["tolerance_overrides"], {"normal_consistency": 1.5e-2}
        )

    def test_candidate_uses_its_own_stock_baseline(self) -> None:
        m8_reports, m8_records, candidate_reports, candidate_records = self._fixtures()
        stock = next(iter(candidate_records.values()))["branches"]["stock"]
        stock["surface"]["chamfer_l1"] += 5.0e-4
        report = aggregate_reports(
            m8_reports=m8_reports,
            m8_records=m8_records,
            candidate_reports=candidate_reports,
            candidate_records=candidate_records,
            bootstrap_samples=20,
        )
        self.assertAlmostEqual(
            report["summary"]["objaverse2k_vs_stock"]["chamfer_l1"]["mean"],
            0.2005,
        )


if __name__ == "__main__":
    unittest.main()
