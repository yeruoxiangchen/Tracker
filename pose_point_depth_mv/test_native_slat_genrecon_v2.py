#!/usr/bin/env python3

from __future__ import annotations

import copy
import unittest

import torch

from pose_point_depth_mv.native_slat_genrecon import (
    NATIVE_SLAT_BASELINE,
    NATIVE_SLAT_GENRECON_CFG,
    NATIVE_SLAT_GENRECON_TRAINING,
)
from pose_point_depth_mv.native_slat_genrecon_v2 import (
    NATIVE_SLAT_GENRECON_V2_PROJECTION,
    NATIVE_SLAT_GENRECON_V2_VERSION,
    validate_native_slat_genrecon_v2_checkpoint,
)


def valid_checkpoint() -> tuple[dict, dict, dict]:
    stock = {"freeze_sha256": "stock-freeze"}
    upstream = {"report_sha256": "native-ss-report"}
    state = {
        "flow.base_model.model.blocks.0.self_attn.to_q.lora_A.default.weight": (
            torch.ones(1)
        )
    }
    checkpoint = {
        "format": NATIVE_SLAT_GENRECON_V2_VERSION,
        "step": 2000,
        "args": {
            "lora_rank": 8,
            "lora_alpha": 16,
            "condition_channels": 1024,
        },
        "model_summary": {
            "format": NATIVE_SLAT_GENRECON_V2_VERSION,
            "pretrained": "unit-test",
            "baseline": NATIVE_SLAT_BASELINE,
            "projection": NATIVE_SLAT_GENRECON_V2_PROJECTION,
            "training_semantics": NATIVE_SLAT_GENRECON_TRAINING,
            "cfg_semantics": NATIVE_SLAT_GENRECON_CFG,
            "condition_scale_policy": "learned_projection_only",
            "post_cfg_cap": False,
            "direct_slat_residual_dependency": False,
            "stock_slat_freeze": stock,
            "upstream_native_ss": upstream,
        },
        "model_trainable_state": copy.deepcopy(state),
        "ema_trainable_state": copy.deepcopy(state),
        "ema": {"updates": 2000, "target_decay": 0.9995},
    }
    return checkpoint, stock, upstream


class NativeSLatGenreconV2Test(unittest.TestCase):
    def validate(self, checkpoint: dict, stock: dict, upstream: dict) -> None:
        validate_native_slat_genrecon_v2_checkpoint(
            checkpoint,
            pretrained="unit-test",
            stock_slat_freeze=stock,
            upstream_native_ss=upstream,
        )

    def test_accepts_exact_v2_contract(self) -> None:
        checkpoint, stock, upstream = valid_checkpoint()
        self.validate(checkpoint, stock, upstream)

    def test_rejects_v3_summary_and_trainable_state(self) -> None:
        checkpoint, stock, upstream = valid_checkpoint()
        checkpoint["model_summary"]["context_view_fusion"] = {"version": "v3"}
        with self.assertRaises(ValueError):
            self.validate(checkpoint, stock, upstream)

        checkpoint, stock, upstream = valid_checkpoint()
        checkpoint["ema_trainable_state"]["view_fusion.transition_gate_raw"] = (
            torch.zeros(24)
        )
        with self.assertRaises(ValueError):
            self.validate(checkpoint, stock, upstream)

    def test_rejects_old_direct_slat_controls(self) -> None:
        for forbidden in (
            "condition_scale",
            "delta_rms_ratio_cap",
            "view_fusion_hidden_dim",
        ):
            with self.subTest(forbidden=forbidden):
                checkpoint, stock, upstream = valid_checkpoint()
                checkpoint["args"][forbidden] = 1
                with self.assertRaises(ValueError):
                    self.validate(checkpoint, stock, upstream)

    def test_rejects_deployment_binding_drift(self) -> None:
        checkpoint, stock, upstream = valid_checkpoint()
        wrong_stock = {"freeze_sha256": "other"}
        with self.assertRaises(RuntimeError):
            self.validate(checkpoint, wrong_stock, upstream)

        checkpoint, stock, upstream = valid_checkpoint()
        wrong_upstream = {"report_sha256": "other"}
        with self.assertRaises(RuntimeError):
            self.validate(checkpoint, stock, wrong_upstream)


if __name__ == "__main__":
    unittest.main()
