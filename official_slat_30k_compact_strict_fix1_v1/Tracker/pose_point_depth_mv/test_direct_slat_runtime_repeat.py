from __future__ import annotations

import unittest
from pathlib import Path
import tempfile

import torch

from pose_point_depth_mv.direct_slat_blind import repeat_floors, sha256_file
from pose_point_depth_mv.direct_slat_runtime_repeat import (
    evaluate_runtime,
    repeat_policy_from_criteria,
    sparse_payload_diff,
    summarize_comparisons,
)
from pose_point_depth_mv.aggregate_direct_slat_runtime_repeat import (
    build_runtime_comparisons,
)


def criteria() -> dict:
    metrics = {
        "latent_feature_rms": 0.02,
        "latent_feature_max_abs": 0.2,
        "chamfer_l1_abs": 0.001,
        "fscore_0p02_abs": 0.01,
        "largest_component_ratio_abs": 0.01,
        "boundary_edge_count_abs": 10.0,
        "boundary_total_length_abs": 0.1,
        "nonmanifold_edge_count_abs": 2.0,
        "component_count_abs": 3.0,
    }
    return {
        "regular_p95_max": metrics,
        "catastrophic_max": {
            name: 2.0 * value for name, value in metrics.items()
        },
        "topology_flip_rate_max": {
            "is_watertight": 0.5,
            "zero_boundary": 0.5,
            "nonmanifold_free": 0.5,
        },
    }


def comparison(value: float, *, branch: str = "stock") -> dict:
    return {
        "scope": "slat_same_process",
        "branch": branch,
        "object_uid": "object",
        "metric_abs_diff": {
            "latent_feature_rms": value,
            "latent_feature_max_abs": value,
            "chamfer_l1_abs": value,
            "fscore_0p02_abs": value,
            "largest_component_ratio_abs": value,
            "boundary_edge_count_abs": value,
            "boundary_total_length_abs": value,
            "nonmanifold_edge_count_abs": value,
            "component_count_abs": value,
        },
        "topology_changed": {
            "is_watertight": False,
            "zero_boundary": False,
            "nonmanifold_free": False,
        },
        "hard_integrity_passed": True,
        "coords_exact": True,
    }


class RuntimeRepeatTest(unittest.TestCase):
    def test_sparse_payload_diff_reports_rms_and_exact_coords(self) -> None:
        left = {
            "coords": torch.tensor([[0, 1, 2, 3]], dtype=torch.int32),
            "feats": torch.tensor([[1.0, 2.0]]),
        }
        right = {
            "coords": left["coords"].clone(),
            "feats": torch.tensor([[1.0, 4.0]]),
        }
        result = sparse_payload_diff(left, right)
        self.assertTrue(result["coords_exact"])
        self.assertFalse(result["features_exact"])
        self.assertAlmostEqual(result["latent_feature_max_abs"], 2.0)
        self.assertAlmostEqual(result["latent_feature_rms"], 2.0**0.5)

    def test_runtime_uses_worst_branch_p95_and_catastrophic_max(self) -> None:
        rows = [comparison(0.0001, branch="stock") for _ in range(20)]
        rows.extend(comparison(0.0002, branch="full") for _ in range(19))
        rows.append(comparison(0.003, branch="full"))
        result = evaluate_runtime(rows, criteria())
        self.assertFalse(result["passed"])
        self.assertFalse(result["checks"]["p95_chamfer_l1_abs"])

    def test_summary_is_separate_by_branch_and_object(self) -> None:
        rows = [comparison(0.0001, branch="stock"), comparison(0.0002, branch="full")]
        summary = summarize_comparisons(rows)
        scoped = summary["scopes"]["slat_same_process"]
        self.assertEqual(set(scoped["by_branch"]), {"stock", "full"})
        self.assertEqual(scoped["by_object"]["object"]["comparison_count"], 2)

    def test_multi_repeat_policy_allows_only_preregistered_jitter(self) -> None:
        policy = repeat_policy_from_criteria(criteria())
        rows = []
        for index in range(20):
            rows.append(
                {
                    "metric_abs_diff": {
                        "chamfer_l1_abs": 0.0001,
                        "fscore_0p02_abs": 0.001,
                        "largest_component_ratio_abs": 0.001,
                        "boundary_edge_count_abs": 1.0,
                        "boundary_total_length_abs": 0.01,
                        "nonmanifold_edge_count_abs": 0.0,
                        "component_count_abs": 1.0,
                    },
                    "topology_changed": {
                        "mesh_success": False,
                        "is_watertight": index == 0,
                        "zero_boundary": False,
                        "nonmanifold_free": False,
                    },
                }
            )
        result = repeat_floors(rows, policy=policy)
        self.assertTrue(result["passed"], result["checks"])
        policy["topology_flip_rate_max"]["is_watertight"] = 0.0
        self.assertFalse(repeat_floors(rows, policy=policy)["passed"])

    def test_synthetic_same_and_independent_process_aggregation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reports = []
            for process_index in range(3):
                process_root = root / f"process_{process_index}"
                process_root.mkdir()
                repeat_count = 5 if process_index == 0 else 1
                records = []
                for stage in ("decoder_only", "slat"):
                    for run_index in range(repeat_count):
                        relative = Path(f"latent_{stage}_{run_index}.pt")
                        payload = {
                            "coords": torch.tensor(
                                [[0, 1, 2, 3]], dtype=torch.int32
                            ),
                            "feats": torch.tensor(
                                [[float(process_index + run_index), 0.0]]
                            ),
                        }
                        torch.save(payload, process_root / relative)
                        structure_row = {
                            "mesh_success": True,
                            "vertices_finite": True,
                            "is_winding_consistent": True,
                            "is_watertight": True,
                            "boundary_edge_count": 0,
                            "boundary_total_length": 0.0,
                            "nonmanifold_edge_count": 0,
                            "component_count": 1,
                            "largest_component_ratio": 1.0,
                        }
                        records.append(
                            {
                                "stage": stage,
                                "pair_id": "pair",
                                "side": "A",
                                "branch": "stock",
                                "object_uid": "object",
                                "uid": "uid",
                                "seed": 42,
                                "run_index": run_index,
                                "latent": {
                                    "path": str(relative),
                                    "sha256": (
                                        "decoder-sha"
                                        if stage == "decoder_only"
                                        else sha256_file(process_root / relative)
                                    ),
                                },
                                "structure": structure_row,
                                "surface": {
                                    "chamfer_l1": float(process_index + run_index)
                                    * 1.0e-5,
                                    "fscore_0p02": 0.5,
                                },
                                "hard_integrity": {"passed": True},
                            }
                        )
                reports.append(
                    {
                        "process_index": process_index,
                        "slat_repeats": repeat_count,
                        "decoder_repeats": repeat_count,
                        "case_selection": {"case": "same"},
                        "protocol": {"protocol_sha256": "protocol"},
                        "runtime": {"runtime_id": "D0"},
                        "records": records,
                        "_root": process_root,
                    }
                )
            rows = build_runtime_comparisons(
                reports, expected_processes=3, same_process_repeats=5
            )
            counts = {}
            for row in rows:
                counts[row["scope"]] = counts.get(row["scope"], 0) + 1
            self.assertEqual(counts["decoder_only_same_process"], 10)
            self.assertEqual(counts["slat_same_process"], 10)
            self.assertEqual(counts["decoder_only_independent_process"], 3)
            self.assertEqual(counts["slat_independent_process"], 3)


if __name__ == "__main__":
    unittest.main()
