from __future__ import annotations

import unittest

from pose_point_depth_mv.evaluate_objaverse2k_no_vggt_slat_fourway import (
    BRANCHES,
    WORKER_REPORT_FORMAT,
    aggregate_reports,
    canonical_json_sha256,
    validate_worker_reports,
)


SURFACE = (
    "chamfer_l1",
    "chamfer_l2",
    "fscore_0p01",
    "fscore_0p02",
    "fscore_0p05",
    "normal_consistency",
)


def branch_row(branch: str, chamfer: float) -> dict:
    wrapper = None
    if branch == "stock":
        wrapper = {
            "mode": "stock",
            "lora": False,
            "posed_dino_residual": False,
            "projection_mode": "none",
        }
    elif branch == "lora_only":
        wrapper = {
            "mode": "lora_only",
            "positive_calls": 1,
            "negative_calls": 1,
            "posed_dino_input": False,
            "every_block_condition_output": "exact_zero_including_projection_bias",
            "projection_mode": "none",
        }
    else:
        wrapper = {
            "positive_calls": 1,
            "negative_calls": 1,
            "projection_mode": branch,
        }
    return {
        "mesh": f"/{branch}.obj",
        "mesh_sha256": branch,
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
            "component_count": int(round(chamfer * 10)) + 1,
        },
        "wrapper": wrapper,
    }


def record(object_uid: str, seed: int, *, offset: float = 0.0) -> dict:
    chamfers = {
        "stock": 0.40 + offset,
        "lora_only": 0.30 + offset,
        "pose_cyclic1": 0.35 + offset,
        "correct": 0.20 + offset,
    }
    return {
        "identity": {
            "checkpoint_sha256": "checkpoint",
            "object_uid": object_uid,
            "uid": f"{object_uid}-sequence",
            "support_seed": seed,
            "master_noise_seed": seed + 100,
            "metric_seed": seed + 200,
            "cache_manifest_sha256": "cache",
        },
        "object_position": 0,
        "same_native_ss_coordinates": True,
        "same_initial_noise": True,
        "coord_count": 10,
        "target": {},
        "branches": {
            branch: branch_row(branch, chamfers[branch]) for branch in BRANCHES
        },
    }


def report(worker: int, start: int, end: int, records: list[dict]) -> dict:
    run_config = {
        "format": WORKER_REPORT_FORMAT,
        "formal": False,
        "training_overlap": False,
        "scope": "test",
        "checkpoint": "/checkpoint.pt",
        "checkpoint_sha256": "checkpoint",
        "checkpoint_step": 2000,
        "weights": "ema",
        "cache_manifest_sha256": "cache",
        "lifting_cache_manifest_sha256": "lifting",
        "native_ss_report_sha256": "ss",
        "stock_slat_freeze_sha256": "freeze",
        "sampling": {"steps": 25},
        "branches": list(BRANCHES),
        "joint_seeds": [42, 43],
        "noise_protocol": "noise.v1",
        "noise_seed": 7,
        "surface_samples": 20,
        "worker_index": worker,
        "num_workers": 2,
        "expected_objects": 2,
        "object_start": start,
        "object_end": end,
        "selected": [{"object_uid": f"object-{worker}"}],
    }
    value = {
        "format": WORKER_REPORT_FORMAT,
        "passed": True,
        "formal": False,
        "training_overlap": False,
        "worker_index": worker,
        "num_workers": 2,
        "object_start": start,
        "object_end": end,
        "object_count": end - start,
        "record_count": len(records),
        "run_config": run_config,
        "model_summary": {},
        "records": records,
    }
    value["report_sha256"] = canonical_json_sha256(value)
    return value


class Objaverse2KFourWayTests(unittest.TestCase):
    def test_aggregate_attributes_lora_pose_and_total_effects(self) -> None:
        reports = [
            report(0, 0, 1, [record("object-0", seed) for seed in (42, 43)]),
            report(1, 1, 2, [record("object-1", seed, offset=0.01) for seed in (42, 43)]),
        ]
        records = {
            (
                row["identity"]["object_uid"],
                row["identity"]["uid"],
                row["identity"]["support_seed"],
            ): row
            for worker in reports
            for row in worker["records"]
        }
        result = aggregate_reports(
            reports=reports, records=records, bootstrap_samples=20
        )
        self.assertTrue(result["passed"])
        self.assertEqual(result["object_count"], 2)
        self.assertEqual(result["record_count"], 4)
        self.assertEqual(result["branch_rollout_count"], 16)
        self.assertAlmostEqual(
            result["comparisons"]["posed_dino_increment"]["metrics"]
            ["chamfer_l1"]["mean"],
            0.1,
        )
        self.assertAlmostEqual(
            result["comparisons"]["correct_pose_specificity"]["metrics"]
            ["chamfer_l1"]["mean"],
            0.15,
        )
        self.assertAlmostEqual(
            result["comparisons"]["generic_lora_increment"]["metrics"]
            ["chamfer_l1"]["mean"],
            0.1,
        )

    def test_validation_rejects_nonzero_lora_only_contract(self) -> None:
        bad_record = record("object-0", 42)
        bad_record["branches"]["lora_only"]["wrapper"][
            "every_block_condition_output"
        ] = "bias_remains"
        reports = [
            report(0, 0, 1, [bad_record, record("object-0", 43)]),
            report(
                1,
                1,
                2,
                [record("object-1", seed) for seed in (42, 43)],
            ),
        ]
        with self.assertRaisesRegex(RuntimeError, "LoRA-only"):
            for value in reports:
                body = dict(value)
                body.pop("report_sha256")
                value["report_sha256"] = canonical_json_sha256(body)
            # Avoid filesystem fixtures while directly exercising the same guard.
            aggregate_reports(
                reports=reports,
                records={
                    (
                        row["identity"]["object_uid"],
                        row["identity"]["uid"],
                        row["identity"]["support_seed"],
                    ): row
                    for worker in reports
                    for row in worker["records"]
                },
                bootstrap_samples=20,
            )


if __name__ == "__main__":
    unittest.main()
