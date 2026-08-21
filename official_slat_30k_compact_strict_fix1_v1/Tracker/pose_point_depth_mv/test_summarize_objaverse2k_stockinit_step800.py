from __future__ import annotations

import unittest

from pose_point_depth_mv.summarize_objaverse2k_stockinit_step800 import (
    MODEL_KEYS,
    aggregate_comparison,
)


def branch(chamfer: float) -> dict:
    return {
        "surface": {
            "chamfer_l1": chamfer,
            "chamfer_l2": chamfer**2,
            "fscore_0p01": 1.0 - chamfer,
            "fscore_0p02": 1.0 - chamfer,
            "fscore_0p05": 1.0 - chamfer,
            "normal_consistency": 1.0 - chamfer,
        },
        "structure": {
            "largest_component_ratio": 1.0 - chamfer,
            "component_count": 1,
        },
    }


def record(model_key: str, object_uid: str, seed: int) -> dict:
    full_chamfer = {
        "m8_step800": 0.35,
        "stockinit_objaverse2k800": 0.20,
        "current_m8init_objaverse2k800": 0.30,
        "current_m8init_objaverse2k2000": 0.25,
    }[model_key]
    branches = {"stock": branch(0.40), "full": branch(full_chamfer)}
    if model_key == "stockinit_objaverse2k800":
        branches = {
            "stock": branch(0.40),
            "lora_only": branch(0.30),
            "pose_cyclic1": branch(0.32),
            "correct": branch(full_chamfer),
        }
    return {
        "identity": {
            "model_label": model_key,
            "checkpoint_sha256": model_key,
            "object_uid": object_uid,
            "uid": f"{object_uid}-sequence",
            "support_seed": seed,
            "master_noise_seed": seed + 10,
            "metric_seed": seed + 20,
            "cache_manifest_sha256": "cache",
        },
        "same_native_ss_coordinates": True,
        "same_initial_noise": True,
        "branches": branches,
    }


def set_stock_chamfer(value: dict, chamfer: float) -> None:
    value["branches"]["stock"] = branch(chamfer)


def run_config(model_key: str) -> dict:
    return {
        "checkpoint": f"/{model_key}.pt",
        "checkpoint_sha256": model_key,
        "checkpoint_step": 800 if "2000" not in model_key else 2000,
        "weights": "ema",
        "cache_manifest_sha256": "cache",
        "lifting_cache_manifest_sha256": "lifting",
        "native_ss_report_sha256": "ss",
        "stock_slat_freeze_sha256": "freeze",
        "sampling": {"steps": 25},
        "joint_seeds": [42, 43, 44],
        "noise_protocol": "noise.v1",
        "noise_seed": 7,
        "surface_samples": 20,
        "expected_objects": 2,
    }


class StockInitStep800SummaryTests(unittest.TestCase):
    def test_object_level_comparison_orders_all_references(self) -> None:
        report_groups = {
            key: [{"run_config": run_config(key)}] for key in MODEL_KEYS
        }
        record_groups = {
            key: {
                (object_uid, f"{object_uid}-sequence", seed): record(
                    key, object_uid, seed
                )
                for object_uid in ("object-0", "object-1")
                for seed in (42, 43, 44)
            }
            for key in MODEL_KEYS
        }
        result = aggregate_comparison(
            report_groups=report_groups,
            record_groups=record_groups,
            bootstrap_samples=20,
        )
        self.assertTrue(result["passed"])
        self.assertEqual(result["object_count"], 2)
        self.assertEqual(result["record_count"], 6)
        self.assertAlmostEqual(
            result["summary"]["stockinit800_vs_stock"]["metrics"]
            ["chamfer_l1"]["mean"],
            0.20,
        )
        self.assertAlmostEqual(
            result["summary"]["stockinit800_vs_m8_step800"]["metrics"]
            ["chamfer_l1"]["mean"],
            0.15,
        )
        self.assertAlmostEqual(
            result["summary"]["stockinit800_vs_current_m8init800"]["metrics"]
            ["chamfer_l1"]["mean"],
            0.10,
        )

    def test_model_vs_stock_uses_same_run_stock(self) -> None:
        report_groups = {
            key: [{"run_config": run_config(key)}] for key in MODEL_KEYS
        }
        record_groups = {
            key: {
                (object_uid, f"{object_uid}-sequence", seed): record(
                    key, object_uid, seed
                )
                for object_uid in ("object-0", "object-1")
                for seed in (42, 43, 44)
            }
            for key in MODEL_KEYS
        }
        for value in record_groups["m8_step800"].values():
            set_stock_chamfer(value, 0.4001)

        result = aggregate_comparison(
            report_groups=report_groups,
            record_groups=record_groups,
            bootstrap_samples=20,
        )

        self.assertAlmostEqual(
            result["summary"]["m8_step800_vs_stock"]["metrics"]
            ["chamfer_l1"]["mean"],
            0.0501,
        )
        self.assertEqual(
            result["stock_comparison_pairing"],
            "Every model-vs-Stock delta uses the Stock rollout from that model's "
            "own matched worker record.",
        )

    def test_stock_reproduction_rejects_excessive_single_record_tail(self) -> None:
        report_groups = {
            key: [{"run_config": run_config(key)}] for key in MODEL_KEYS
        }
        record_groups = {
            key: {
                (object_uid, f"{object_uid}-sequence", seed): record(
                    key, object_uid, seed
                )
                for object_uid in ("object-0", "object-1")
                for seed in (42, 43, 44)
            }
            for key in MODEL_KEYS
        }
        first = next(iter(record_groups["m8_step800"].values()))
        set_stock_chamfer(first, 0.402)

        with self.assertRaisesRegex(RuntimeError, "Stock reproduction exceeds tolerance"):
            aggregate_comparison(
                report_groups=report_groups,
                record_groups=record_groups,
                bootstrap_samples=20,
            )


if __name__ == "__main__":
    unittest.main()
