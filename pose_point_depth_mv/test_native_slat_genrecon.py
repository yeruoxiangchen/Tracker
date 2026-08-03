#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn

from pose_point_depth_mv.native_slat_genrecon import (
    NATIVE_SLAT_BASELINE,
    NATIVE_SLAT_GENRECON_CFG,
    NATIVE_SLAT_GENRECON_PROJECTION,
    NATIVE_SLAT_GENRECON_TRAINING,
    NATIVE_SLAT_GENRECON_VERSION,
    load_stock_slat_freeze,
    make_stock_slat_freeze,
    project_sparse_frustum_dino,
    validate_native_slat_genrecon_checkpoint,
)
from pose_point_depth_mv.native_ss_genrecon import (
    NATIVE_SS_GENRECON_CFG,
    NATIVE_SS_GENRECON_TRAINING,
)
from pose_point_depth_mv.direct_slat_flow import (
    SUPPORT_RUNTIME_FIELDS,
    support_generator_identity,
)
from pose_point_depth_mv.train_native_slat_genrecon import validate_decoder_audit


class NativeSLatGenreconTests(unittest.TestCase):
    def test_ss_slat_training_and_cfg_semantics_are_identical(self) -> None:
        self.assertEqual(NATIVE_SLAT_GENRECON_TRAINING, NATIVE_SS_GENRECON_TRAINING)
        self.assertEqual(NATIVE_SLAT_GENRECON_CFG, NATIVE_SS_GENRECON_CFG)

    def test_sparse_projection_uses_active_coords_and_trailing_dino(self) -> None:
        views = 2
        sample = {
            "visual_patch_features": torch.randn(views, 16 * 16, 3072),
            "intrinsics": torch.tensor(
                [[[200.0, 0.0, 112.0], [0.0, 200.0, 112.0], [0.0, 0.0, 1.0]]]
            ).repeat(views, 1, 1),
            "extrinsics": torch.eye(4)[None].repeat(views, 1, 1),
            "predicted_depth": torch.ones(views, 224, 224),
            "grid_transform": "identity",
            "extrinsics_type": "c2w",
            "camera_forward_sign": 1.0,
        }
        coords = torch.tensor(
            [[0, 15, 15, 24], [0, 16, 16, 28], [0, 15, 16, 20]],
            dtype=torch.int32,
        )
        projected, valid, stats = project_sparse_frustum_dino(
            sample, coords, device=torch.device("cpu")
        )
        self.assertEqual(tuple(projected.shape), (views, len(coords), 1024))
        self.assertEqual(tuple(valid.shape), (views, len(coords)))
        self.assertTrue(torch.isfinite(projected).all())
        self.assertGreater(float(stats["supported_fraction"]), 0.0)

    def test_stock_freeze_binds_complete_state(self) -> None:
        flow = nn.Sequential(nn.Linear(3, 4), nn.Linear(4, 2))
        flow.register_buffer("bf16_scalar", torch.tensor(1.0, dtype=torch.bfloat16))
        decoder = nn.Linear(2, 1)
        payload = make_stock_slat_freeze(
            pretrained="unit-test",
            flow=flow,
            decoder=decoder,
            sampler_params={
                "steps": 25,
                "cfg_strength": 5.0,
                "cfg_interval": (0.5, 1.0),
                "rescale_t": 3.0,
            },
            normalization={"mean": [0.0], "std": [1.0]},
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "freeze.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            loaded = load_stock_slat_freeze(path)
        self.assertEqual(loaded["baseline"], NATIVE_SLAT_BASELINE)
        self.assertEqual(loaded["freeze_sha256"], payload["freeze_sha256"])
        self.assertTrue(loaded["slat_flow"]["state_sha256"])
        self.assertTrue(loaded["mesh_decoder"]["state_sha256"])
        self.assertEqual(loaded["slat_sampler_params"]["guidance_rescale"], 0.0)

    def test_checkpoint_rejects_direct_slat_residual_fields(self) -> None:
        stock = {"freeze_sha256": "stock"}
        upstream = {"report_sha256": "ss"}
        summary = {
            "format": NATIVE_SLAT_GENRECON_VERSION,
            "pretrained": "unit-test",
            "baseline": NATIVE_SLAT_BASELINE,
            "projection": NATIVE_SLAT_GENRECON_PROJECTION,
            "training_semantics": NATIVE_SLAT_GENRECON_TRAINING,
            "cfg_semantics": NATIVE_SLAT_GENRECON_CFG,
            "condition_scale_policy": "learned_projection_only",
            "post_cfg_cap": False,
            "direct_slat_residual_dependency": False,
            "stock_slat_freeze": {"freeze_sha256": "stock"},
            "upstream_native_ss": upstream,
        }
        checkpoint = {
            "format": NATIVE_SLAT_GENRECON_VERSION,
            "step": 10,
            "args": {},
            "model_summary": summary,
            "model_trainable_state": {"x": torch.ones(1)},
            "ema_trainable_state": {"x": torch.ones(1)},
            "ema": {"updates": 10, "target_decay": 0.9995},
        }
        validate_native_slat_genrecon_checkpoint(
            checkpoint,
            pretrained="unit-test",
            stock_slat_freeze=stock,
            upstream_native_ss=upstream,
        )
        checkpoint["args"]["condition_scale"] = 1.0
        with self.assertRaises(ValueError):
            validate_native_slat_genrecon_checkpoint(
                checkpoint,
                pretrained="unit-test",
                stock_slat_freeze=stock,
                upstream_native_ss=upstream,
            )

    def test_formal_trainer_has_no_smoke_or_direct_residual_surface(self) -> None:
        source = (Path(__file__).parent / "train_native_slat_genrecon.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn('add_argument("--max_objects"', source)
        self.assertNotIn('add_argument("--condition_scale"', source)
        self.assertNotIn('add_argument("--delta_rms_ratio_cap"', source)
        self.assertIn('add_argument("--p_uncond"', source)
        self.assertIn('add_argument("--ema_decay"', source)

    def test_val_decoder_audit_allows_seed_superset_but_not_protocol_drift(self) -> None:
        config = {name: None for name in SUPPORT_RUNTIME_FIELDS}
        config.update(
            {
                "pretrained": "unit-test",
                "ss_seeds": [42],
                "mapping_version": "native_ss_genrecon_active32_every_block.v2",
                "target_source": {"kind": "unit-test-target"},
            }
        )
        audit_config = dict(config)
        audit_config["ss_seeds"] = [42, 43, 44]
        report = {
            "format": "pose_point_depth_mv.direct_slat_target_decoder_audit.v1",
            "passed": True,
            "pretrained": "unit-test",
            "summary": {"object_count": 32},
            "thresholds": {},
            "support_generator": support_generator_identity(audit_config),
            "cache_manifest": "/object-disjoint/val/cache/manifest.json",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            path.write_text(json.dumps(report), encoding="utf-8")
            binding = validate_decoder_audit(
                path, cache_config=config, pretrained="unit-test"
            )
            self.assertEqual(binding["summary"]["object_count"], 32)
            report["support_generator"]["fields"]["mapping_version"] = "drift"
            path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaises(RuntimeError):
                validate_decoder_audit(
                    path, cache_config=config, pretrained="unit-test"
                )


if __name__ == "__main__":
    unittest.main()
