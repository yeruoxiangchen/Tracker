#!/usr/bin/env python3

from __future__ import annotations

import unittest

import torch

from pose_point_depth_mv.native_slat_condition_only import (
    NATIVE_SLAT_BASELINE,
    NATIVE_SLAT_CONDITION_ONLY_CFG,
    NATIVE_SLAT_CONDITION_ONLY_PROJECTION,
    NATIVE_SLAT_CONDITION_ONLY_TRAINING,
)
from pose_point_depth_mv.native_slat_condition_only_no_vggt import (
    NATIVE_SLAT_CONDITION_ONLY_NO_VGGT_VERSION,
    NO_VGGT_CONDITION_ONLY_CONTRACT,
    validate_native_slat_condition_only_no_vggt_checkpoint,
)


class NativeSLatConditionOnlyNoVggtTests(unittest.TestCase):
    def _checkpoint(self) -> dict:
        upstream = {"report_sha256": "ss"}
        return {
            "format": NATIVE_SLAT_CONDITION_ONLY_NO_VGGT_VERSION,
            "step": 10,
            "args": {},
            "model_summary": {
                "pretrained": "unit-test",
                "baseline": NATIVE_SLAT_BASELINE,
                "projection": NATIVE_SLAT_CONDITION_ONLY_PROJECTION,
                "training_semantics": NATIVE_SLAT_CONDITION_ONLY_TRAINING,
                "cfg_semantics": NATIVE_SLAT_CONDITION_ONLY_CFG,
                "condition_scale_policy": "learned_projection_only",
                "post_cfg_cap": False,
                "direct_slat_residual_dependency": False,
                "flow_lora": {
                    "present": False,
                    "module_count": 0,
                    "parameter_count": 0,
                    "construction": "PEFT is not imported or installed",
                },
                "context_view_fusion": {"trainable": False},
                "stock_slat_freeze": {"freeze_sha256": "stock"},
                "upstream_native_ss": upstream,
                "input_context_contract": NO_VGGT_CONDITION_ONLY_CONTRACT,
                "vggt_model_executed": False,
            },
            "model_trainable_state": {"aggregator.x": torch.ones(1)},
            "ema_trainable_state": {"aggregator.x": torch.ones(1)},
            "ema": {"updates": 10, "target_decay": 0.9995},
        }

    def test_checkpoint_requires_dino_only_contract(self) -> None:
        checkpoint = self._checkpoint()
        validate_native_slat_condition_only_no_vggt_checkpoint(
            checkpoint,
            pretrained="unit-test",
            stock_slat_freeze={"freeze_sha256": "stock"},
            upstream_native_ss={"report_sha256": "ss"},
        )
        checkpoint["model_summary"]["vggt_model_executed"] = True
        with self.assertRaisesRegex(ValueError, "input contract|executed VGGT"):
            validate_native_slat_condition_only_no_vggt_checkpoint(
                checkpoint,
                pretrained="unit-test",
                stock_slat_freeze={"freeze_sha256": "stock"},
                upstream_native_ss={"report_sha256": "ss"},
            )

    def test_contract_is_condition_only_and_dino_only(self) -> None:
        self.assertEqual(
            NO_VGGT_CONDITION_ONLY_CONTRACT["architecture"],
            "native_slat_condition_only",
        )
        self.assertFalse(NO_VGGT_CONDITION_ONLY_CONTRACT["flow_lora"])
        self.assertFalse(NO_VGGT_CONDITION_ONLY_CONTRACT["vggt_model_executed"])


if __name__ == "__main__":
    unittest.main()
