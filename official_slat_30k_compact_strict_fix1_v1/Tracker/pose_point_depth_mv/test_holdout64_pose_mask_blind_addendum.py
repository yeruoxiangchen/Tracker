from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from pose_point_depth_mv.dataset_tools.prepare_omni_real_pose_mask_runtime_inputs import (
    validate_protocol_object_sets,
)
from pose_point_depth_mv.evaluate_holdout64_pose_mask_blind_addendum import (
    EXPECTED_OBJECTS,
    _validate_point_pose_sampling,
    paired_method_comparison,
    records_by_pair,
)
from pose_point_depth_mv.freeze_holdout64_pose_mask_blind_protocol import (
    FORMAT,
    _payload_sha256,
    validate_protocol_contract,
)
from pose_point_depth_mv.omni_real_benchmark_common import sha256_file


class Holdout64PoseMaskBlindAddendumTest(unittest.TestCase):
    def test_formal_scope_requires_exact_raw_reference_set(self) -> None:
        reference = {f"category:object_{index:02d}" for index in range(64)}
        validate_protocol_object_sets(
            set(reference),
            set(reference),
            protocol_scope="formal_holdout64_blind_addendum",
        )
        with self.assertRaisesRegex(RuntimeError, "formal Holdout64 object sets differ"):
            validate_protocol_object_sets(
                reference | {"extra:object"},
                reference,
                protocol_scope="formal_holdout64_blind_addendum",
            )

    def test_exact_64xseed42_record_contract(self) -> None:
        manifest = {
            "seeds": [42],
            "object_count": EXPECTED_OBJECTS,
            "record_count": EXPECTED_OBJECTS,
            "objects": [
                {
                    "object_key": f"category:object_{index:02d}",
                    "seed": 42,
                    "method": "native_no_vggt_mixed",
                    "passed": True,
                }
                for index in range(EXPECTED_OBJECTS)
            ],
        }
        rows = records_by_pair(
            manifest, expected_method="native_no_vggt_mixed", label="point_mask"
        )
        self.assertEqual(len(rows), EXPECTED_OBJECTS)
        manifest["seeds"] = [43]
        with self.assertRaisesRegex(RuntimeError, "exact 64xseed42"):
            records_by_pair(
                manifest, expected_method="native_no_vggt_mixed", label="point_mask"
            )

    def test_pose_mask_paired_sign_is_positive_when_better(self) -> None:
        common = {
            "object_key": "category:object",
            "seed": 42,
            "chamfer_l1": 0.30,
            "chamfer_l2": 0.20,
            "fscore_0p01": 0.40,
            "fscore_0p02": 0.50,
            "normal_consistency": 0.60,
        }
        pose = {
            **common,
            "method": "pose_mask",
            "chamfer_l1": 0.20,
            "chamfer_l2": 0.10,
            "fscore_0p01": 0.60,
            "fscore_0p02": 0.70,
            "normal_consistency": 0.80,
        }
        result = paired_method_comparison(
            [pose, {**common, "method": "point_mask"}],
            left="pose_mask",
            right="point_mask",
        )
        for metric in result["metrics"].values():
            self.assertGreater(metric["mean"], 0.0)
            self.assertEqual(metric["left_win_count"], 1)
            self.assertEqual(metric["right_win_count"], 0)

    def test_point_pose_sampling_requires_frozen_cfg_and_ema(self) -> None:
        point = {
            "native_ss_checkpoint_sha256": "ss",
            "native_ss_weights": "ema",
            "native_slat_checkpoint_sha256": "slat",
            "native_slat_weights": "ema",
            "stock_slat_freeze_sha256": "freeze",
            "sampling": {
                "steps": 25,
                "cfg_strength": 5.0,
                "cfg_interval": [0.5, 1.0],
                "guidance_rescale": 0.0,
                "rescale_t": 3.0,
            },
            "sampling_sha256": "sampling",
            "post_cfg_cap": False,
            "wrapper": {"condition_scale_policy": "learned_projection_only"},
        }
        pose = {
            **point,
            "condition_scale_policy": "learned_projection_only",
        }
        pair = ("category:object", 42)
        result = _validate_point_pose_sampling(
            {pair: point},
            {pair: pose},
            expected_ss_sha256="ss",
            expected_slat_sha256="slat",
            expected_stock_freeze_sha256="freeze",
        )
        self.assertTrue(result["passed"])
        pose["sampling"] = {**pose["sampling"], "cfg_strength": 3.0}
        with self.assertRaisesRegex(RuntimeError, "frozen SLat sampling changed"):
            _validate_point_pose_sampling(
                {pair: point},
                {pair: pose},
                expected_ss_sha256="ss",
                expected_slat_sha256="slat",
                expected_stock_freeze_sha256="freeze",
            )

    def test_protocol_verifier_detects_bound_file_change(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frozen = root / "frozen.json"
            code = root / "code.py"
            frozen.write_text("frozen\n", encoding="utf-8")
            code.write_text("code\n", encoding="utf-8")
            payload = {
                "format": FORMAT,
                "blindness": {
                    "result_inspected_before_freeze": False,
                    "one_shot_joint_unblinding": True,
                    "old_fiveway_report_consumption_forbidden": True,
                },
                "evaluation": {
                    "object_count": 64,
                    "seeds": [42],
                    "surface_samples": 20000,
                    "weights": "ema",
                    "point_pose_sampling_must_match": True,
                    "gt_alignment_fit_allowed": False,
                },
                "forbidden_inputs": {
                    "old_fiveway_report": (
                        "/tmp/M11M_holdout64_fiveway_no_vggt_"
                        "mixed1244_seed42_v1/report.json"
                    )
                },
                "frozen_inputs": {
                    "input": {"path": str(frozen), "sha256": sha256_file(frozen)}
                },
                "implementation": {
                    "code": {"path": str(code), "sha256": sha256_file(code)}
                },
                "passed": True,
            }
            payload["payload_sha256"] = _payload_sha256(payload)
            contract = root / "contract.json"
            contract.write_text(json.dumps(payload), encoding="utf-8")
            validate_protocol_contract(contract)
            code.write_text("changed\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "binding changed"):
                validate_protocol_contract(contract)

    def test_background_runner_never_references_old_fiveway_report(self) -> None:
        runner = (
            Path(__file__).parent
            / "background_jobs"
            / "run_holdout64_pose_mask_blind_addendum.sh"
        ).read_text(encoding="utf-8")
        self.assertNotIn("M11M_holdout64", runner)
        self.assertIn("evaluate_holdout64_pose_mask_blind_addendum", runner)


if __name__ == "__main__":
    unittest.main()
