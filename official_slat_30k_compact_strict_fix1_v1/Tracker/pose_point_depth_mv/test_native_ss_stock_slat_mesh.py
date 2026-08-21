from __future__ import annotations

import unittest
from pathlib import Path
import json
import tempfile

from pose_point_depth_mv.evaluate_native_ss_stock_slat_mesh import (
    aggregate_transfer_records,
    load_split_metadata,
    pair_id,
    validate_ss_evidence_payload,
)
from pose_point_depth_mv.native_ss_genrecon import NATIVE_SS_GENRECON_EVAL


def ss_report(*, false_check: str = "count_ratio_lower") -> dict:
    checks = {
        "iou_gain_mean": True,
        "recall_gain_mean": True,
        "latent_mse_gain_mean": True,
        "count_ratio_lower": True,
        "count_ratio_upper": True,
        "stock_baseline_nonempty": True,
        "iou_object_win_rate": True,
        "pose_control_iou_advantage": True,
        "disabled_stock_equivalence": True,
    }
    checks[false_check] = False
    return {
        "format": NATIVE_SS_GENRECON_EVAL,
        "checks": checks,
        "protocol": {
            "checkpoint": "/tmp/step.pt",
            "checkpoint_sha256": "abc",
            "checkpoint_step": 2000,
            "weights": "ema",
            "condition_scale_policy": "learned_projection_only",
            "post_cfg_cap": False,
            "steps": 25,
            "cfg_interval": [0.5, 1.0],
            "guidance_rescale": 0.0,
            "rescale_t": 3.0,
            "amp_dtype": "bf16",
        },
        "calibrated_parameters": {
            "cfg_strength": 5.0,
            "condition_scale_policy": "learned_projection_only",
            "post_cfg_cap": False,
        },
    }


def mesh_row(
    object_uid: str,
    seed: int,
    branch: str,
    *,
    chamfer: float,
    fscore: float,
    normal: float,
    lcr: float,
) -> dict:
    return {
        "object_uid": object_uid,
        "seed": seed,
        "branch": branch,
        "passed": True,
        "surface": {
            "chamfer_l1": chamfer,
            "fscore_0p02": fscore,
            "normal_consistency": normal,
        },
        "structure": {"largest_component_ratio": lcr, "mesh_success": True},
    }


class NativeSSStockSLatMeshTest(unittest.TestCase):
    def test_count_lower_is_the_only_accepted_ss_failure(self) -> None:
        binding = validate_ss_evidence_payload(ss_report())
        self.assertEqual(binding["false_checks"], ["count_ratio_lower"])
        self.assertEqual(binding["cfg_strength"], 5.0)

        with self.assertRaisesRegex(RuntimeError, "cannot be waived"):
            validate_ss_evidence_payload(ss_report(false_check="iou_gain_mean"))

    def test_pair_id_is_stable_and_seed_specific(self) -> None:
        self.assertEqual(pair_id("uid", 42), pair_id("uid", 42))
        self.assertNotEqual(pair_id("uid", 42), pair_id("uid", 43))

    def test_transfer_averages_paired_seeds_per_object(self) -> None:
        records = []
        for object_uid in ("a", "b"):
            for seed in (42, 43):
                records.extend(
                    (
                        mesh_row(
                            object_uid,
                            seed,
                            "stock",
                            chamfer=0.20,
                            fscore=0.40,
                            normal=0.50,
                            lcr=0.90,
                        ),
                        mesh_row(
                            object_uid,
                            seed,
                            "native",
                            chamfer=0.10,
                            fscore=0.60,
                            normal=0.70,
                            lcr=0.91,
                        ),
                    )
                )
        result = aggregate_transfer_records(
            records,
            expected_pairs=4,
            seeds=[42, 43],
            bootstrap_samples=200,
            metadata_by_object={
                "a": {"source": "legacy", "view_count": 2},
                "b": {"source": "omni", "view_count": 8},
            },
            chamfer_win_rate_min=0.55,
            lcr_delta_min=-0.02,
        )
        self.assertTrue(result["passed"])
        self.assertEqual(result["valid_pair_count"], 4)
        self.assertEqual(len(result["object_rows"]), 2)
        self.assertAlmostEqual(
            result["summary"]["chamfer_l1_improvement"]["mean"], 0.1
        )
        self.assertAlmostEqual(result["summary"]["fscore_0p02_delta"]["mean"], 0.2)

    def test_missing_branch_fails_integrity(self) -> None:
        result = aggregate_transfer_records(
            [
                mesh_row(
                    "a",
                    42,
                    "stock",
                    chamfer=0.2,
                    fscore=0.4,
                    normal=0.5,
                    lcr=0.9,
                )
            ],
            expected_pairs=1,
            seeds=[42],
            bootstrap_samples=100,
            metadata_by_object={"a": {"source": "legacy", "view_count": 2}},
            chamfer_win_rate_min=0.55,
            lcr_delta_min=-0.02,
        )
        self.assertFalse(result["passed"])
        self.assertFalse(result["checks"]["no_invalid_pairs"])

    def test_split_phase_position_is_stable_for_subsets(self) -> None:
        split = {
            "phases": {
                "final": [
                    {"object_uid": "a", "source": "legacy", "view_count": 2},
                    {"object_uid": "b", "source": "gap", "view_count": 4},
                    {"object_uid": "c", "source": "omni", "view_count": 8},
                ]
            }
        }
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "split.json"
            path.write_text(json.dumps(split), encoding="utf-8")
            metadata, _ = load_split_metadata(
                str(path),
                "final",
                [{"uid": "c_seq000", "object_uid": "c"}],
            )
        self.assertEqual(metadata["c"]["phase_position"], 2)


if __name__ == "__main__":
    unittest.main()
