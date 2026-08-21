#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import unittest

import torch
from torch import nn

from pose_point_depth_mv.native_slat_condition_only import (
    NATIVE_SLAT_BASELINE,
    NATIVE_SLAT_CONDITION_ONLY_CFG,
    NATIVE_SLAT_CONDITION_ONLY_PROJECTION,
    NATIVE_SLAT_CONDITION_ONLY_TRAINING,
    NATIVE_SLAT_CONDITION_ONLY_VERSION,
    _forbidden_adaptation_names,
    condition_only_parameter_group,
    validate_native_slat_condition_only_checkpoint,
)
from pose_point_depth_mv.native_ss_genrecon import (
    NATIVE_SS_GENRECON_CFG,
    NATIVE_SS_GENRECON_TRAINING,
)


class NativeSLatConditionOnlyTests(unittest.TestCase):
    def test_ss_slat_training_and_cfg_semantics_are_identical(self) -> None:
        self.assertEqual(
            NATIVE_SLAT_CONDITION_ONLY_TRAINING, NATIVE_SS_GENRECON_TRAINING
        )
        self.assertEqual(NATIVE_SLAT_CONDITION_ONLY_CFG, NATIVE_SS_GENRECON_CFG)

    def test_forbidden_adapter_audit_detects_lora(self) -> None:
        clean = nn.Sequential(nn.Linear(2, 2))
        self.assertEqual(_forbidden_adaptation_names(clean), [])
        clean[0].lora_A = nn.Parameter(torch.zeros(1))
        self.assertTrue(_forbidden_adaptation_names(clean))

    def test_optimizer_has_one_condition_group(self) -> None:
        model = nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1))
        groups = condition_only_parameter_group(model, lr=1.0e-4, weight_decay=0.01)
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["name"], "posed_dino_3d_condition")
        self.assertEqual(sum(value.numel() for value in groups[0]["params"]), 13)

    def test_checkpoint_accepts_only_condition_state(self) -> None:
        stock = {"freeze_sha256": "stock"}
        upstream = {"report_sha256": "ss"}
        summary = {
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
        }
        checkpoint = {
            "format": NATIVE_SLAT_CONDITION_ONLY_VERSION,
            "step": 10,
            "args": {},
            "model_summary": summary,
            "model_trainable_state": {"aggregator.x": torch.ones(1)},
            "ema_trainable_state": {"aggregator.x": torch.ones(1)},
            "ema": {"updates": 10, "target_decay": 0.9995},
        }
        validate_native_slat_condition_only_checkpoint(
            checkpoint,
            pretrained="unit-test",
            stock_slat_freeze=stock,
            upstream_native_ss=upstream,
        )
        checkpoint["model_trainable_state"]["flow.lora_A.weight"] = torch.ones(1)
        with self.assertRaises(ValueError):
            validate_native_slat_condition_only_checkpoint(
                checkpoint,
                pretrained="unit-test",
                stock_slat_freeze=stock,
                upstream_native_ss=upstream,
            )

    def test_trainer_exposes_no_lora_or_view_fusion_arguments(self) -> None:
        source = (
            Path(__file__).parent / "train_native_slat_condition_only.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn('add_argument("--lora_', source)
        self.assertNotIn('add_argument("--view_fusion', source)
        self.assertNotIn('add_argument("--geometry_logit', source)
        self.assertIn('add_argument("--condition_lr"', source)


if __name__ == "__main__":
    unittest.main()
