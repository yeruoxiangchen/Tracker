#!/usr/bin/env python3

from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import torch

from pose_point_depth_mv.native_slat_genrecon import (
    NATIVE_SLAT_BASELINE,
    NATIVE_SLAT_GENRECON_CFG,
    NATIVE_SLAT_GENRECON_TRAINING,
)
from pose_point_depth_mv.native_slat_genrecon_v2 import (
    NATIVE_SLAT_GENRECON_V2_PROJECTION,
)
from pose_point_depth_mv.native_slat_genrecon_with_vggt_official import (
    NATIVE_SLAT_OFFICIAL_WITH_VGGT_VERSION,
    OFFICIAL_WITH_VGGT_SLAT_CONTRACT,
    validate_native_slat_official_with_vggt_checkpoint,
)
from pose_point_depth_mv import (
    native_slat_genrecon_with_vggt_official as model_module,
)
from pose_point_depth_mv.proobjaverse_official_slat_protocol import (
    canonical_sha256,
    sha256_file,
)
from pose_point_depth_mv.proobjaverse_official_slat_with_vggt_cache import (
    WITH_VGGT_CONTEXT_VERSION,
    WITH_VGGT_LIFTING_MANIFEST_FORMAT,
    WITH_VGGT_SIDECAR_FORMAT,
    WITH_VGGT_SLAT_MANIFEST_FORMAT,
    WithVGGTNativeConditionSLatDataset,
    validate_native_slat_vggt_context_tensor,
)
from pose_point_depth_mv import (
    proobjaverse_official_slat_with_vggt_cache as cache_module,
)
from pose_point_depth_mv import (
    train_native_slat_genrecon_with_vggt_official as train_module,
)
from pose_point_depth_mv import (
    prepare_proobjaverse_official_slat_with_vggt_sidecar as builder_module,
)


class _FakeBaseDataset:
    def __init__(self, *args, **kwargs) -> None:
        del args, kwargs
        self.config = {
            "condition_arch": "native_ss_genrecon_v2",
            "native_ss_deployment": {"report_sha256": "native-ss"},
        }
        self.config_hash = "base-config"
        self.slat_normalization = {
            "mean": [0.0] * 8,
            "std": [1.0] * 8,
        }
        self.slat_normalization_hash = "normalization"
        self.rows = [
            {"uid": "u0", "object_uid": "u0", "support_seed": 42},
            {"uid": "u1", "object_uid": "u1", "support_seed": 42},
        ]
        self.slat = object()
        self.lifting = type(
            "FakeLifting", (), {"config_hash": "base-lifting-config"}
        )()

    def __getitem__(self, index: int):
        row = self.rows[index]
        return {
            **row,
            "condition": {"cond": [torch.ones(1, 3, 1024)], "neg_cond": []},
            "lifting_sample": {
                "uid": row["uid"],
                "object_uid": row["object_uid"],
                "view_ids": torch.tensor([7, 3], dtype=torch.int64),
                "visual_patch_features": torch.zeros(2, 1, 1024),
            },
        }


def _valid_checkpoint() -> tuple[dict, dict, dict]:
    stock = {"freeze_sha256": "stock-freeze"}
    upstream = {"report_sha256": "native-ss-report"}
    state = {
        "flow.base_model.model.blocks.0.self_attn.to_q.lora_A.default.weight": (
            torch.ones(1)
        )
    }
    checkpoint = {
        "format": NATIVE_SLAT_OFFICIAL_WITH_VGGT_VERSION,
        "step": 10,
        "args": {
            "lora_rank": 8,
            "lora_alpha": 16,
            "condition_channels": 1024,
        },
        "model_summary": {
            "format": NATIVE_SLAT_OFFICIAL_WITH_VGGT_VERSION,
            "protocol_version": NATIVE_SLAT_OFFICIAL_WITH_VGGT_VERSION,
            "input_context_contract": OFFICIAL_WITH_VGGT_SLAT_CONTRACT,
            "stock_floor": "V0",
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
        "data_identity": {"config_hash": "with-vggt-config"},
        "model_trainable_state": copy.deepcopy(state),
        "ema_trainable_state": copy.deepcopy(state),
        "ema": {"updates": 10, "target_decay": 0.9995},
    }
    return checkpoint, stock, upstream


class WithVGGTCacheDatasetTest(unittest.TestCase):
    def _write_pair(self, root: Path) -> tuple[Path, Path, Path]:
        base_slat = root / "base_slat.json"
        base_lifting = root / "base_lifting.json"
        base_slat.write_text("{}\n", encoding="utf-8")
        base_lifting.write_text("{}\n", encoding="utf-8")
        sidecar = root / "sidecars/u0.pt"
        sidecar.parent.mkdir()
        sidecar_contract = {
            "version": WITH_VGGT_CONTEXT_VERSION,
            "builder_version": "unit-test-builder.v1",
            "protocol_sha256": "protocol",
            "split": "train",
            "base_cache": {
                "slat_manifest_sha256": sha256_file(base_slat),
                "lifting_manifest_sha256": sha256_file(base_lifting),
                "slat_config_hash": "base-config",
                "lifting_config_hash": "base-lifting-config",
                "slat_normalization_hash": "normalization",
            },
            "selected_view_policy": (
                "read exact ordered view_ids from immutable base lifting cache; "
                "no view selection is executed"
            ),
            "selected_view_count": 2,
            "shared_geometric_preprocessing": {"version": "unit-test"},
            "shared_geometric_preprocessing_hash": "preprocessing",
            "native_condition": {
                "producer": (
                    "TrellisVGGTTo3DPipeline.vggt_feat + encode_image + get_slat_cond"
                ),
                "vggt_layers": [4, 11, 17, 23],
                "dino_sequence": "full_cls_register_patch_sequence",
                "expected_dino_prefix_tokens": 5,
                "output_layout": "[views,tokens,1024]",
                "positive_context_materialized": True,
                "negative_context_policy": "runtime_zeros_like_positive",
                "base_dino_patch_replay": {
                    "source": "same encode_image result used by native slat_vggt_cond",
                    "reference": "immutable base lifting visual_patch_features",
                    "comparison_dtype": "torch.float16",
                    "max_abs_tolerance": 0.01,
                    "mean_abs_tolerance": 0.0005,
                },
            },
            "camera_contract": {
                "vggt_camera_consumed": False,
                "vggt_depth_consumed": False,
                "posed_dino_uses_base_known_K_T": True,
                "known_K_T_replaced": False,
            },
            "native_ss_contract": "unchanged",
            "encoder_assets": {
                "pretrained": "unit-test",
                "trellis_snapshot_revision": "revision",
                "pipeline_json_sha256": "pipeline",
                "slat_vggt_cond_config_sha256": "slat-config",
                "slat_vggt_cond_weights_sha256": "slat-weights",
                "vggt_repo": "unit-test-vggt",
                "vggt_snapshot_revision": "vggt-revision",
                "vggt_config_sha256": "vggt-config",
                "vggt_weights_sha256": "vggt-weights",
                "dino_model": "dinov2_vitl14_reg",
                "dino_weights_filename": "dino.pth",
                "dino_weights_sha256": "dino-weights",
                "dino_source_tree_sha256": "dino-source",
                "vggt_source_tree_sha256": "vggt-source",
                "trellis_pipeline_source_sha256": "pipeline-source",
                "slat_condition_source_sha256": "condition-source",
                "shared_preprocessing_source_sha256": "preprocessing-source",
                "builder_source_sha256": "builder-source",
            },
        }
        sidecar_contract_hash = canonical_sha256(sidecar_contract)
        torch.save(
            {
                "format": WITH_VGGT_SIDECAR_FORMAT,
                "uid": "u0",
                "object_uid": "u0",
                "support_seed": 42,
                "sidecar_contract_hash": sidecar_contract_hash,
                "view_ids": torch.tensor([7, 3], dtype=torch.int64),
                "native_slat_vggt_cond": torch.arange(
                    2 * 6 * 1024, dtype=torch.float32
                ).reshape(2, 6, 1024),
                "negative_context_policy": "runtime_zeros_like_positive",
                "decoded_source_rgba_sha256": ["r0", "r1"],
                "processed_input_rgb_sha256": ["p0", "p1"],
                "vggt_camera_consumed": False,
                "known_K_T_replaced": False,
            },
            sidecar,
        )
        samples = [
            {
                "uid": "u0",
                "object_uid": "u0",
                "support_seed": 42,
                "base_index": 0,
                "sidecar_file": "sidecars/u0.pt",
                "sidecar_file_sha256": sha256_file(sidecar),
                "sidecar_file_size": sidecar.stat().st_size,
                "view_ids": [7, 3],
                "native_context_shape": [2, 6, 1024],
                "native_context_dtype": "torch.float32",
                "decoded_source_rgba_sha256": ["r0", "r1"],
                "processed_input_rgb_sha256": ["p0", "p1"],
                "source_render_tar_sha256": "render-hash",
            }
        ]
        pair_binding = {
            "version": WITH_VGGT_CONTEXT_VERSION,
            "base_cache": {
                "slat_manifest": "base_slat.json",
                "slat_manifest_sha256": sha256_file(base_slat),
                "lifting_manifest": "base_lifting.json",
                "lifting_manifest_sha256": sha256_file(base_lifting),
                "slat_config_hash": "base-config",
                "lifting_config_hash": "base-lifting-config",
                "slat_normalization_hash": "normalization",
            },
            "sidecar_contract": sidecar_contract,
            "sidecar_contract_hash": sidecar_contract_hash,
            "sidecar_index_sha256": canonical_sha256(samples),
            "sample_count": 1,
            "ordered_uid_sha256": canonical_sha256(["u0"]),
        }
        config = {
            "condition_arch": "native_ss_genrecon_v2",
            "native_ss_deployment": {"report_sha256": "native-ss"},
            "slat_input_context": {
                "version": WITH_VGGT_CONTEXT_VERSION,
                "stock_floor": "V0",
                "source": "native_reconviagen_vggt_plus_dinov2_slat_vggt_cond",
                "sidecar_contract_hash": sidecar_contract_hash,
                "base_no_vggt_slat_config_hash": "base-config",
                "selected_views": "exact ordered base lifting view_ids",
                "native_full_dino_sequence": True,
                "vggt_model_executed": True,
                "vggt_camera_consumed": False,
                "known_pose_dino_branch_unchanged": True,
                "negative_context_policy": "runtime_zeros_like_positive",
            },
        }
        common = {
            "pair_identity": canonical_sha256(pair_binding),
            "pair_binding": pair_binding,
            "config": config,
            "config_hash": canonical_sha256(config),
            "samples": samples,
        }
        slat_manifest = root / "with_vggt_slat_manifest.json"
        lifting_manifest = root / "with_vggt_lifting_manifest.json"
        slat_manifest.write_text(
            json.dumps({"format": WITH_VGGT_SLAT_MANIFEST_FORMAT, **common}),
            encoding="utf-8",
        )
        lifting_manifest.write_text(
            json.dumps({"format": WITH_VGGT_LIFTING_MANIFEST_FORMAT, **common}),
            encoding="utf-8",
        )
        return slat_manifest, lifting_manifest, sidecar

    def test_exact_pair_replaces_only_stock_context(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            slat, lifting, _ = self._write_pair(Path(temporary))
            with mock.patch.object(
                cache_module,
                "_BaseNativeConditionSLatDataset",
                _FakeBaseDataset,
            ):
                dataset = WithVGGTNativeConditionSLatDataset(
                    slat, lifting, verify_hashes=False
                )
                sample = dataset[0]
        self.assertEqual(dataset.config["slat_input_context"]["version"], WITH_VGGT_CONTEXT_VERSION)
        self.assertEqual(len(sample["condition"]["cond"]), 2)
        self.assertEqual(tuple(sample["condition"]["cond"][0].shape), (1, 6, 1024))
        self.assertTrue(
            all(
                int(torch.count_nonzero(value)) == 0
                for value in sample["condition"]["neg_cond"]
            )
        )
        self.assertTrue(torch.equal(sample["lifting_sample"]["view_ids"], torch.tensor([7, 3])))
        self.assertFalse(sample["with_vggt_sidecar"]["vggt_camera_consumed"])

    def test_rejects_sidecar_contract_and_view_drift(self) -> None:
        for mutation in ("contract", "views"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                slat, lifting, sidecar = self._write_pair(Path(temporary))
                payload = torch.load(sidecar, map_location="cpu", weights_only=False)
                if mutation == "contract":
                    payload["sidecar_contract_hash"] = "changed"
                else:
                    payload["view_ids"] = torch.tensor([3, 7])
                torch.save(payload, sidecar)
                # Keep hash verification off so the test reaches the semantic guard.
                with mock.patch.object(
                    cache_module,
                    "_BaseNativeConditionSLatDataset",
                    _FakeBaseDataset,
                ):
                    dataset = WithVGGTNativeConditionSLatDataset(slat, lifting)
                    with self.assertRaises((ValueError, RuntimeError)):
                        dataset[0]

    def test_rejects_pair_manifest_semantic_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            slat, lifting, _ = self._write_pair(Path(temporary))
            payload = json.loads(lifting.read_text(encoding="utf-8"))
            payload["config"]["slat_input_context"]["vggt_camera_consumed"] = True
            payload["config_hash"] = canonical_sha256(payload["config"])
            lifting.write_text(json.dumps(payload), encoding="utf-8")
            with mock.patch.object(
                cache_module,
                "_BaseNativeConditionSLatDataset",
                _FakeBaseDataset,
            ):
                with self.assertRaises(ValueError):
                    WithVGGTNativeConditionSLatDataset(slat, lifting)

    def test_rejects_unknown_v1_sidecar_contract_field(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            slat, _, _ = self._write_pair(Path(temporary))
            payload = json.loads(slat.read_text(encoding="utf-8"))
            binding = payload["pair_binding"]
            binding["sidecar_contract"]["future_semantic"] = True
            binding["sidecar_contract_hash"] = canonical_sha256(
                binding["sidecar_contract"]
            )
            payload["pair_identity"] = canonical_sha256(binding)
            with self.assertRaises(ValueError):
                cache_module._manifest_binding(
                    slat,
                    payload,
                    expected_format=WITH_VGGT_SLAT_MANIFEST_FORMAT,
                )

    def test_context_tensor_contract(self) -> None:
        valid = torch.zeros(8, 1374, 1024, dtype=torch.float16)
        self.assertIs(
            validate_native_slat_vggt_context_tensor(valid, views=8, uid="u"),
            valid,
        )
        for invalid in (
            torch.zeros(8, 1374, 1023),
            torch.zeros(7, 1374, 1024),
            torch.zeros(8, 0, 1024),
        ):
            with self.assertRaises(ValueError):
                validate_native_slat_vggt_context_tensor(invalid, views=8, uid="u")


class WithVGGTCheckpointTest(unittest.TestCase):
    def test_accepts_only_exact_own_contract(self) -> None:
        checkpoint, stock, upstream = _valid_checkpoint()
        validate_native_slat_official_with_vggt_checkpoint(
            checkpoint,
            pretrained="unit-test",
            stock_slat_freeze=stock,
            upstream_native_ss=upstream,
        )

    def test_rejects_no_vggt_or_changed_context_contract(self) -> None:
        checkpoint, stock, upstream = _valid_checkpoint()
        checkpoint["format"] = "pose_point_depth_mv.native_slat_genrecon_no_vggt.v1"
        with self.assertRaises(ValueError):
            validate_native_slat_official_with_vggt_checkpoint(
                checkpoint,
                pretrained="unit-test",
                stock_slat_freeze=stock,
                upstream_native_ss=upstream,
            )
        checkpoint, stock, upstream = _valid_checkpoint()
        checkpoint["model_summary"]["input_context_contract"] = {
            **OFFICIAL_WITH_VGGT_SLAT_CONTRACT,
            "vggt_camera_consumed": True,
        }
        with self.assertRaises(ValueError):
            validate_native_slat_official_with_vggt_checkpoint(
                checkpoint,
                pretrained="unit-test",
                stock_slat_freeze=stock,
                upstream_native_ss=upstream,
            )

    def test_step0_report_is_explicitly_v_equals_v0(self) -> None:
        exact = {
            "passed": True,
            "conditional_max_abs": 0.0,
            "unconditional_max_abs": 0.0,
        }
        with mock.patch.object(train_module, "_base_initial_stock_audit", return_value=exact):
            report = train_module._with_vggt_initial_stock_audit()
        self.assertTrue(report["step0_v_equals_v0"])
        self.assertEqual(report["reference_floor"], "V0")

        changed = {**exact, "conditional_max_abs": 1.0e-6}
        with mock.patch.object(train_module, "_base_initial_stock_audit", return_value=changed):
            with self.assertRaises(RuntimeError):
                train_module._with_vggt_initial_stock_audit()

    def test_numeric_cli_spelling_is_semantically_equal(self) -> None:
        with mock.patch.object(sys, "argv", ["trainer", "--lora_lr", "3e-5"]):
            train_module._freeze_argument("--lora_lr", "0.00003")
        with mock.patch.object(sys, "argv", ["trainer", "--lora_rank", "4"]):
            with self.assertRaises(ValueError):
                train_module._freeze_argument("--lora_rank", "8")
        with mock.patch.object(
            sys,
            "argv",
            ["trainer", "--lora_rank", "8", "--lora_rank=4"],
        ):
            with self.assertRaises(ValueError):
                train_module._freeze_argument("--lora_rank", "8")


class WithVGGTBuilderRuntimeTest(unittest.TestCase):
    def test_resume_recomputes_only_uncommitted_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            sidecar, record = builder_module._paths(output, "u0")
            sidecar.parent.mkdir(parents=True)
            sidecar.write_bytes(b"incomplete")
            self.assertFalse(record.exists())
            resumed = builder_module._validate_resumed_record(
                output=output,
                row={"uid": "u0"},
                sidecar_contract_hash="contract",
            )
        self.assertIsNone(resumed)

    def test_native_dino_replay_is_bound_to_frozen_base_patches(self) -> None:
        frozen = torch.linspace(-2.0, 2.0, 2 * 3 * 1024).reshape(
            2, 3, 1024
        ).to(torch.float16)
        full = torch.zeros(1, 2, 8, 1024, dtype=torch.float32)
        full[0, :, 5:] = frozen.float()
        report = builder_module._validate_base_dino_patch_replay(
            full, frozen, uid="u"
        )
        self.assertTrue(report["passed"])
        self.assertEqual(report["max_abs"], 0.0)

        changed = full.clone()
        changed[0, 0, 5, 0] += 0.02
        with self.assertRaises(RuntimeError):
            builder_module._validate_base_dino_patch_replay(
                changed, frozen, uid="u"
            )

    def test_training_component_loader_does_not_materialize_vggt(self) -> None:
        from trellis import pipelines

        class FullVGGT:
            pass

        class StockOnly:
            pass

        returned = (object(), object(), None, {}, {}, {})

        def fake_build(**kwargs):
            self.assertEqual(kwargs, {"marker": "x"})
            self.assertIs(pipelines.TrellisVGGTTo3DPipeline, StockOnly)
            return returned

        with (
            mock.patch.object(pipelines, "TrellisVGGTTo3DPipeline", FullVGGT),
            mock.patch.object(pipelines, "TrellisImageTo3DPipeline", StockOnly),
            mock.patch.object(
                model_module, "_build_v2_components", side_effect=fake_build
            ),
        ):
            result = model_module.build_native_slat_official_with_vggt_components(
                marker="x"
            )
            self.assertIs(pipelines.TrellisVGGTTo3DPipeline, FullVGGT)
        self.assertEqual(result[:3], returned[:3])
        self.assertIn("no per-rank VGGT", result[3]["training_component_loader"])

    def test_minimal_loader_does_not_materialize_unused_trellis_models(self) -> None:
        case = self

        class FakeVGGT(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.aggregator = torch.nn.Linear(2, 2)
                self.depth_head = torch.nn.Linear(1, 1)
                self.track_head = torch.nn.Linear(1, 1)
                self.camera_head = torch.nn.Linear(1, 1)
                self.point_head = torch.nn.Linear(1, 1)

        class FakeVGGTFactory:
            @staticmethod
            def from_pretrained(repo):
                case.assertEqual(repo, "Stable-X/vggt-object-v0-1")
                return FakeVGGT()

        class FakePipeline:
            def __init__(self) -> None:
                self.models = {}

            def _init_image_cond_model(self, name):
                case.assertEqual(name, "dinov2_vitl14_reg")
                self.models["image_cond_model"] = torch.nn.Linear(2, 2)

        slat_condition = torch.nn.Linear(3, 3)
        with (
            mock.patch(
                "trellis.models.from_pretrained", return_value=slat_condition
            ) as load_trellis,
            mock.patch(
                "trellis.pipelines.trellis_image_to_3d.TrellisVGGTTo3DPipeline",
                FakePipeline,
            ),
            mock.patch(
                "trellis.pipelines.trellis_image_to_3d.VGGT", FakeVGGTFactory
            ),
            mock.patch("torch.cuda.set_device"),
            mock.patch("torch.cuda.get_device_capability", return_value=(8, 0)),
            mock.patch("torch.cuda.empty_cache"),
        ):
            pipeline = builder_module._load_condition_pipeline(
                pretrained="Stable-X/trellis-vggt-v0-2",
                vggt_repo="Stable-X/vggt-object-v0-1",
                device=torch.device("cuda:0"),
            )
        load_trellis.assert_called_once_with(
            "Stable-X/trellis-vggt-v0-2/ckpts/slat_vggt_cond"
        )
        self.assertEqual(set(pipeline.models), {"slat_vggt_cond", "image_cond_model"})
        self.assertIs(pipeline.slat_vggt_cond, slat_condition)
        for name in ("depth_head", "track_head", "camera_head", "point_head"):
            self.assertFalse(hasattr(pipeline.VGGT_model, name))
        self.assertEqual(pipeline.VGGT_dtype, torch.bfloat16)
        self.assertTrue(pipeline.low_vram)


if __name__ == "__main__":
    unittest.main()
