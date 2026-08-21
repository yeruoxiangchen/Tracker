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
    NATIVE_SLAT_CONTEXT_FUSION,
    NATIVE_SLAT_GENRECON_PROJECTION,
    NATIVE_SLAT_GENRECON_TRAINING,
    NATIVE_SLAT_GENRECON_VERSION,
    NativeSLatCrossAttentionViewFusion,
    load_stock_slat_freeze,
    make_stock_slat_freeze,
    project_sparse_frustum_dino,
    validate_native_slat_genrecon_checkpoint,
)
from pose_point_depth_mv.train_native_slat_genrecon import (
    checkpoint_args,
    checkpoint_stock_context_views,
    make_parser,
    resolved_run_until_step,
    select_stock_context_views,
    validate_args,
    validate_resume_training_contract,
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
    def test_run_until_step_is_operational_and_not_checkpoint_identity(self) -> None:
        required = [
            "--cache_manifest", "cache.json", "--lifting_cache_manifest", "lift.json",
            "--target_decoder_audit", "audit.json", "--native_ss_report", "ss.json",
            "--stock_slat_freeze", "freeze.json", "--output_dir", "out",
            "--max_steps", "2000", "--run_until_step", "400",
        ]
        args = make_parser().parse_args(required)
        validate_args(args)
        self.assertEqual(resolved_run_until_step(args), 400)
        self.assertEqual(checkpoint_args(args)["max_steps"], 2000)
        self.assertNotIn("run_until_step", checkpoint_args(args))
        args.run_until_step = 2001
        with self.assertRaisesRegex(ValueError, "run_until_step"):
            validate_args(args)

    def test_explicit_resume_max_steps_extension(self) -> None:
        required = [
            "--cache_manifest", "cache.json", "--lifting_cache_manifest", "lift.json",
            "--target_decoder_audit", "audit.json", "--native_ss_report", "ss.json",
            "--stock_slat_freeze", "freeze.json", "--output_dir", "out",
            "--resume", "step.pt", "--max_steps", "10000", "--grad_accum", "2",
            "--allow_resume_max_steps_extension",
        ]
        args = make_parser().parse_args(required)
        validate_args(args)
        checkpoint = {
            "step": 2000,
            "args": {"max_steps": 8000, "grad_accum": 2},
            "model_summary": {
                "optimization": {
                    "world_size": 4,
                    "global_effective_batch": 8,
                }
            },
        }
        transition = validate_resume_training_contract(
            checkpoint, args, world_size=4
        )
        self.assertTrue(transition["max_steps_extended"])
        self.assertFalse(transition["topology_changed"])
        self.assertEqual(transition["current_global_effective_batch"], 8)
        saved_args = checkpoint_args(args)
        self.assertNotIn("allow_resume_max_steps_extension", saved_args)

    def test_resume_topology_change_is_rejected(self) -> None:
        required = [
            "--cache_manifest", "cache.json", "--lifting_cache_manifest", "lift.json",
            "--target_decoder_audit", "audit.json", "--native_ss_report", "ss.json",
            "--stock_slat_freeze", "freeze.json", "--output_dir", "out",
            "--resume", "step.pt", "--max_steps", "8000", "--grad_accum", "2",
        ]
        args = make_parser().parse_args(required)
        checkpoint = {
            "step": 2000,
            "args": {"max_steps": 8000, "grad_accum": 2},
            "model_summary": {
                "optimization": {
                    "world_size": 4,
                    "global_effective_batch": 8,
                }
            },
        }
        with self.assertRaisesRegex(ValueError, "world_size/grad_accum"):
            validate_resume_training_contract(checkpoint, args, world_size=8)

    def test_explicit_topology_change_preserves_global_batch(self) -> None:
        required = [
            "--cache_manifest", "cache.json", "--lifting_cache_manifest", "lift.json",
            "--target_decoder_audit", "audit.json", "--native_ss_report", "ss.json",
            "--stock_slat_freeze", "freeze.json", "--output_dir", "out",
            "--resume", "step.pt", "--max_steps", "10000", "--grad_accum", "1",
            "--allow_resume_max_steps_extension",
            "--allow_resume_topology_change",
        ]
        args = make_parser().parse_args(required)
        checkpoint = {
            "step": 4000,
            "args": {"max_steps": 8000, "grad_accum": 2},
            "model_summary": {
                "optimization": {
                    "world_size": 4,
                    "global_effective_batch": 8,
                }
            },
        }
        transition = validate_resume_training_contract(
            checkpoint, args, world_size=8
        )
        self.assertTrue(transition["max_steps_extended"])
        self.assertTrue(transition["topology_changed"])
        self.assertEqual(transition["current_global_effective_batch"], 8)
        self.assertFalse(transition["per_rank_rng_stream_exact"])

    def test_stock_context_policy_changes_only_the_context_list(self) -> None:
        contexts = [torch.full((1, 2, 3), float(index)) for index in range(4)]
        self.assertIs(select_stock_context_views(contexts, "all"), contexts)
        selected = select_stock_context_views(contexts, "first")
        self.assertEqual(len(selected), 1)
        self.assertIs(selected[0], contexts[0])
        with self.assertRaisesRegex(ValueError, "non-empty per-view list"):
            select_stock_context_views(torch.zeros(1, 2, 3), "first")

    def test_old_slat_checkpoint_defaults_to_all_view_stock_context(self) -> None:
        self.assertEqual(checkpoint_stock_context_views({"args": {}}), "all")
        self.assertEqual(
            checkpoint_stock_context_views(
                {"args": {"stock_context_views": "first"}}
            ),
            "first",
        )

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

    def test_view_fusion_starts_at_exact_stock_mean_with_live_gate_gradient(self) -> None:
        fusion = NativeSLatCrossAttentionViewFusion(
            4, 2, hidden_dim=3, geometry_logit_scale_init=1.0
        )
        cross_logits = torch.zeros(2, 2)
        geometry_logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
        valid = torch.ones(2, 2, dtype=torch.bool)
        weights, stats = fusion.effective_weights(
            0,
            cross_logits,
            geometry_logits=geometry_logits,
            valid=valid,
        )
        self.assertTrue(torch.equal(weights, torch.full_like(weights, 0.5)))
        self.assertEqual(float(stats["fusion_gate"]), 0.0)
        outputs = torch.tensor([[1.0, -2.0], [3.0, 4.0]])
        loss = (weights * outputs).sum()
        loss.backward()
        self.assertIsNotNone(fusion.transition_gate_raw.grad)
        self.assertNotEqual(float(fusion.transition_gate_raw.grad[0]), 0.0)

    def test_view_fusion_masks_invalid_views_and_all_invalid_falls_back(self) -> None:
        fusion = NativeSLatCrossAttentionViewFusion(
            4, 1, hidden_dim=2, geometry_logit_scale_init=1.0
        )
        with torch.no_grad():
            fusion.transition_gate_raw.fill_(1.0)
        cross_logits = torch.zeros(2, 2)
        geometry_logits = torch.tensor([[1.0, 4.0], [3.0, -2.0]])
        valid = torch.tensor([[True, False], [False, False]])
        weights, _ = fusion.effective_weights(
            0,
            cross_logits,
            geometry_logits=geometry_logits,
            valid=valid,
        )
        self.assertTrue(torch.equal(weights[:, 0], torch.tensor([1.0, 0.0])))
        self.assertTrue(torch.equal(weights[:, 1], torch.tensor([0.5, 0.5])))

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
            "context_view_fusion": {"version": NATIVE_SLAT_CONTEXT_FUSION},
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
