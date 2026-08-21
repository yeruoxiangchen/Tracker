from __future__ import annotations

import ast
import copy
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np
import torch

from ar_ss_flow.pose_lifting import (
    LIFTING_CACHE_VERSION,
    LIFTING_METADATA_NAMES,
    schema_hash,
)
from ar_ss_flow.shared_object_preprocessing import canonical_json_sha256
from official_ss_with_vggt_perf_v1 import build_cache, model, train
from official_ss_with_vggt_perf_v1.cache import (
    CONTEXT_VERSION,
    MANIFEST_FORMAT,
    MODEL_CONTEXT_CONTRACT,
    SIDECAR_FORMAT,
    WithVGGTOfficialSSDataset,
    validate_native_ss_vggt_context_tensor,
    validate_official_ss_with_vggt_cache_contract,
    validate_official_ss_with_vggt_evaluation_cache_contract,
)
from pose_point_depth_mv.proobjaverse_official_slat_protocol import (
    canonical_sha256,
    sha256_file,
)


class Fixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.uid = "a" * 64
        self.view_ids = torch.arange(8, dtype=torch.int64)
        target = root / "target.npz"
        np.savez(
            target,
            z=np.zeros((8, 16, 16, 16), dtype=np.float16),
            target_coords=np.asarray([[0, 0, 0]], dtype=np.int64),
        )
        preprocessing = {
            "shared_geometry": {},
            "shared_geometry_hash": "fixture",
            "sample_geometry_identity_hash": "fixture",
        }
        sample = {
            "format": LIFTING_CACHE_VERSION,
            "uid": self.uid,
            "object_uid": self.uid,
            "visual_patch_features": torch.zeros((8, 4, 1024), dtype=torch.float16),
            "predicted_depth": torch.zeros((8, 2, 2), dtype=torch.float32),
            "depth_confidence": torch.zeros((8, 2, 2), dtype=torch.float32),
            "masks": torch.zeros((8, 2, 2), dtype=torch.float32),
            "intrinsics": torch.eye(3).repeat(8, 1, 1),
            "extrinsics": torch.eye(4).repeat(8, 1, 1),
            "prior_coords": torch.zeros((1, 3), dtype=torch.int64),
            "prior_confidence": torch.ones((1,), dtype=torch.float32),
            "stock_condition": torch.ones((1, 4, 1024), dtype=torch.float16),
            "view_ids": self.view_ids,
            "grid_transform": "fixture",
            "extrinsics_type": "world_to_camera",
            "camera_forward_sign": 1.0,
            "ss_latent": str(target),
            "preprocessing": preprocessing,
        }
        cache_file = root / "lifting.pt"
        torch.save(sample, cache_file)
        base_config = {
            "no_vggt": {"vggt_feature_dim": 0},
            "official_ss_targets": {"split": "train"},
        }
        base_manifest = {
            "format": LIFTING_CACHE_VERSION,
            "metadata_names": list(LIFTING_METADATA_NAMES),
            "metadata_schema_hash": schema_hash(),
            "output_dir": str(root),
            "visual_feature_dim": 1024,
            "feature_metadata": {
                "dino_feature_dim": 1024,
                "patch_count": 4,
                "patch_start_idx": 0,
            },
            "config": base_config,
            "config_hash": canonical_json_sha256(base_config),
            "samples": [
                {
                    "uid": self.uid,
                    "object_uid": self.uid,
                    "cache_file": str(cache_file),
                    "ss_latent": str(target),
                }
            ],
        }
        self.base_manifest_path = root / "base_manifest.json"
        self.base_manifest_path.write_text(json.dumps(base_manifest))
        context = torch.full((1, 4096, 1024), 2.0, dtype=torch.float16)
        contract = {
            "version": CONTEXT_VERSION,
            "protocol_sha256": "protocol",
            "split": "train",
            "selected_view_count": 8,
            "shared_geometric_preprocessing": {},
            "shared_geometric_preprocessing_hash": "geometry",
            "context_semantics": {"producer": "fixture"},
            "model_context": MODEL_CONTEXT_CONTRACT,
            "encoder_assets": {"weights": "fixture"},
        }
        contract_hash = canonical_sha256(contract)
        sidecar_payload = {
            "format": SIDECAR_FORMAT,
            "uid": self.uid,
            "object_uid": self.uid,
            "sidecar_contract_hash": contract_hash,
            "view_ids": self.view_ids,
            "native_ss_vggt_cond": context,
            "negative_context_policy": "runtime_zeros_like_positive",
            "decoded_source_rgba_sha256": ["rgba"] * 8,
            "processed_input_rgb_sha256": ["rgb"] * 8,
            "vggt_camera_consumed": False,
            "known_K_T_replaced": False,
        }
        sidecar = root / "sidecar.pt"
        torch.save(sidecar_payload, sidecar)
        rows = [
            {
                "uid": self.uid,
                "object_uid": self.uid,
                "base_index": 0,
                "sidecar_file": sidecar.name,
                "sidecar_file_sha256": sha256_file(sidecar),
                "sidecar_file_size": sidecar.stat().st_size,
                "view_ids": self.view_ids.tolist(),
                "native_context_shape": list(context.shape),
                "native_context_dtype": str(context.dtype),
                "source_render_tar_sha256": "render",
            }
        ]
        binding = {
            "version": CONTEXT_VERSION,
            "base_cache": {
                "manifest": self.base_manifest_path.name,
                "manifest_sha256": sha256_file(self.base_manifest_path),
                "config_hash": canonical_json_sha256(base_config),
            },
            "sidecar_contract": contract,
            "sidecar_contract_hash": contract_hash,
            "sidecar_index_sha256": canonical_sha256(rows),
            "sample_count": 1,
            "ordered_uid_sha256": canonical_sha256([self.uid]),
        }
        config = {"ss_input_context": MODEL_CONTEXT_CONTRACT}
        manifest = {
            "format": MANIFEST_FORMAT,
            "pair_identity": canonical_sha256(binding),
            "pair_binding": binding,
            "config": config,
            "config_hash": canonical_sha256(config),
            "samples": rows,
        }
        self.manifest_path = root / "with_vggt_ss_manifest.json"
        self.manifest_path.write_text(json.dumps(manifest))


class ContextTensorTest(unittest.TestCase):
    def test_exact_shape(self) -> None:
        value = torch.zeros((1, 4096, 1024), dtype=torch.float16)
        self.assertIs(validate_native_ss_vggt_context_tensor(value, uid="u"), value)

    def test_wrong_shape_fails(self) -> None:
        with self.assertRaises(ValueError):
            validate_native_ss_vggt_context_tensor(
                torch.zeros((8, 1369, 1024)), uid="u"
            )


class DatasetTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.fixture = Fixture(Path(self.temporary.name))

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_only_stock_context_is_replaced(self) -> None:
        dataset = WithVGGTOfficialSSDataset(self.fixture.manifest_path)
        base = dataset.base[0]
        sample = dataset[0]
        self.assertEqual(sample["stock_condition_source"], "native_ss_vggt_cond_sidecar")
        self.assertEqual(tuple(sample["stock_condition"].shape), (1, 4096, 1024))
        self.assertTrue(torch.all(sample["stock_condition"] == 2))
        self.assertFalse(torch.equal(sample["stock_condition"], base["stock_condition"]))
        for key in ("target", "visual_patch_features", "intrinsics", "extrinsics"):
            self.assertTrue(torch.equal(sample[key], base[key]), key)

    def test_pair_identity_mismatch_fails(self) -> None:
        payload = json.loads(self.fixture.manifest_path.read_text())
        payload["pair_identity"] = "0" * 64
        broken = Path(self.temporary.name) / "broken.json"
        broken.write_text(json.dumps(payload))
        with self.assertRaises(ValueError):
            WithVGGTOfficialSSDataset(broken)

    def test_strict_sidecar_subset_maps_through_bound_base_index(self) -> None:
        base = json.loads(self.fixture.base_manifest_path.read_text())
        real_row = dict(base["samples"][0])
        dummy_row = dict(real_row)
        dummy_row["uid"] = "b" * 64
        dummy_row["object_uid"] = "b" * 64
        base["samples"] = [dummy_row, real_row]
        subset_base = Path(self.temporary.name) / "subset_base_manifest.json"
        subset_base.write_text(json.dumps(base))

        manifest = json.loads(self.fixture.manifest_path.read_text())
        manifest["samples"][0]["base_index"] = 1
        binding = manifest["pair_binding"]
        binding["base_cache"]["manifest"] = subset_base.name
        binding["base_cache"]["manifest_sha256"] = sha256_file(subset_base)
        binding["sidecar_index_sha256"] = canonical_sha256(manifest["samples"])
        manifest["pair_identity"] = canonical_sha256(binding)
        subset_manifest = Path(self.temporary.name) / "subset_sidecar_manifest.json"
        subset_manifest.write_text(json.dumps(manifest))

        dataset = WithVGGTOfficialSSDataset(subset_manifest)
        self.assertEqual(len(dataset.base.rows), 2)
        self.assertEqual(len(dataset), 1)
        self.assertEqual(dataset.base_indices, [1])
        self.assertEqual(dataset.rows[0]["uid"], self.fixture.uid)
        self.assertEqual(dataset[0]["uid"], self.fixture.uid)

    def test_sidecar_subset_rejects_out_of_range_base_index(self) -> None:
        manifest = json.loads(self.fixture.manifest_path.read_text())
        manifest["samples"][0]["base_index"] = 1
        binding = manifest["pair_binding"]
        binding["sidecar_index_sha256"] = canonical_sha256(manifest["samples"])
        manifest["pair_identity"] = canonical_sha256(binding)
        broken = Path(self.temporary.name) / "out_of_range_base_index.json"
        broken.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "base index differs"):
            WithVGGTOfficialSSDataset(broken)

    @mock.patch(
        "official_ss_with_vggt_perf_v1.cache.validate_official_ss_cache_contract"
    )
    def test_train_contract_relabels_base_no_vggt(self, validator) -> None:
        validator.return_value = {
            "no_vggt": {"base": True},
            "official_ss_targets": {"split": "train", "domain_contract": {"x": 1}},
        }
        dataset = WithVGGTOfficialSSDataset(self.fixture.manifest_path)
        result = validate_official_ss_with_vggt_cache_contract(dataset)
        self.assertNotIn("no_vggt", result)
        self.assertEqual(result["base_cache_feature_contract"], {"base": True})
        self.assertEqual(result["with_vggt_ss"]["model_context"], MODEL_CONTEXT_CONTRACT)

    @mock.patch(
        "official_ss_with_vggt_perf_v1.cache.validate_official_ss_cache_contract"
    )
    def test_eval_accepts_same_semantics_but_different_pair(self, validator) -> None:
        validator.return_value = {
            "no_vggt": {"base": True},
            "official_ss_targets": {"split": "dev", "domain_contract": {"x": 1}},
        }
        dataset = WithVGGTOfficialSSDataset(self.fixture.manifest_path)
        observed = validate_official_ss_with_vggt_cache_contract(dataset)
        trained_contract = copy.deepcopy(observed["with_vggt_ss"])
        trained_contract["pair_identity"] = "train-pair"
        training = {
            "config_hash": "train-config",
            "feature_contract": {
                "official_ss_targets": {
                    "split": "train",
                    "domain_contract": {"x": 1},
                },
                "with_vggt_ss": trained_contract,
            },
        }
        result = validate_official_ss_with_vggt_evaluation_cache_contract(
            dataset, training_identity=training
        )
        self.assertEqual(
            result["evaluation_training_binding"]["mode"],
            "official_with_vggt_ss_context",
        )


class ModelAndEntrypointTest(unittest.TestCase):
    @mock.patch("official_ss_with_vggt_perf_v1.model._validate_v2_checkpoint")
    def test_checkpoint_requires_with_vggt_identity(self, base_validator) -> None:
        checkpoint = {
            "format": model.VERSION,
            "args": {},
            "model_trainable_state": {},
            "ema_trainable_state": {},
            "ema": {"updates": 2, "target_decay": 0.9995},
            "step": 2,
            "model_summary": {
                "protocol_version": model.VERSION,
                "input_context_contract": MODEL_CONTEXT_CONTRACT,
                "stock_floor": "VSS0",
                "fresh_initialization_only": True,
                "vggt_model_executed_during_training": False,
                "vggt_camera_consumed": False,
                "known_pose_dino_branch_unchanged": True,
                "official_ss_target_unchanged": True,
                "step0_straight_through_contract": (
                    model.STEP0_STRAIGHT_THROUGH_CONTRACT
                ),
                "runtime_input_policy": {
                    "ddp_device_ids": None,
                    "caller_places_model_inputs": True,
                    "complete_lifting_sample_stays_cpu_until_projection": True,
                    "vggt_context_transferred_explicitly_once": True,
                    "scientific_math_changed": False,
                },
            },
            "data_identity": {
                "feature_contract": {
                    "with_vggt_ss": {"model_context": MODEL_CONTEXT_CONTRACT}
                }
            },
        }
        model.validate_checkpoint(checkpoint, pretrained="fixture")
        base_validator.assert_called_once()
        broken = copy.deepcopy(checkpoint)
        broken["data_identity"]["feature_contract"].pop("with_vggt_ss")
        with self.assertRaises(ValueError):
            model.validate_checkpoint(broken, pretrained="fixture")

    def test_fresh_entry_forbids_parent_checkpoint(self) -> None:
        original = sys.argv
        try:
            sys.argv = ["train", "--init_checkpoint", "old.pt"]
            with self.assertRaises(ValueError):
                train._freeze_scientific_args()
        finally:
            sys.argv = original

    def test_short_smoke_is_explicit_and_bounded(self) -> None:
        original = sys.argv
        try:
            sys.argv = ["train", "--allow_short_smoke", "--max_steps", "2"]
            train._freeze_scientific_args()
            self.assertNotIn("--allow_short_smoke", sys.argv)
            self.assertEqual(train._argument("--max_steps"), "2")
        finally:
            sys.argv = original

    def test_duplicate_scientific_argument_is_rejected(self) -> None:
        original = sys.argv
        try:
            sys.argv = ["train", "--seed", "42", "--seed=43"]
            with self.assertRaisesRegex(ValueError, "duplicate scientific argument"):
                train._freeze_scientific_args()
        finally:
            sys.argv = original

    def test_builder_reuses_historical_native_ss_v2_production_path(self) -> None:
        source = Path(build_cache.__file__).read_text(encoding="utf-8")
        self.assertIn("build_native_stock_pipeline", source)
        self.assertIn("extract_stock_condition", source)
        self.assertIn("historical Native-SS v2", source)
        self.assertNotIn("pipeline.get_slat_cond(", source)

    def test_isolated_step0_reference_is_forward_exact_and_gradient_exact(self) -> None:
        value = torch.tensor(
            [0.031234567, -0.017654321], dtype=torch.float32, requires_grad=True
        )
        reference = torch.tensor(
            [-0.004321987, 0.009876543], dtype=torch.float32, requires_grad=True
        )
        output = model._exact_forward_straight_through(value, reference)
        self.assertTrue(torch.equal(output, reference.detach()))
        output.sum().backward()
        self.assertTrue(torch.equal(value.grad, torch.ones_like(value)))
        self.assertIsNone(reference.grad)

    def test_dedicated_model_installs_exact_step0_reference(self) -> None:
        from pose_point_depth_mv import native_ss_genrecon as native_v2

        original = native_v2._straight_through_reference
        try:
            model._install_exact_step0_reference()
            self.assertIs(
                native_v2._straight_through_reference,
                model._exact_forward_straight_through,
            )
        finally:
            native_v2._straight_through_reference = original

    def test_historical_extractor_observes_exact_patch_replay(self) -> None:
        cached = torch.arange(2 * 3 * 1024, dtype=torch.float16).reshape(
            2, 3, 1024
        )
        expected_context = torch.zeros(1, 4096, 1024, dtype=torch.float16)

        class Pipeline:
            def get_ss_cond(self, image_cond, aggregated, num_samples):
                self.asserted = (aggregated, num_samples)
                return {
                    "cond": expected_context,
                    "neg_cond": torch.zeros_like(expected_context),
                }

        pipeline = Pipeline()

        def extract_stock_condition(value, images):
            self.assertIs(value, pipeline)
            self.assertEqual(images, ["official-view"])
            patches = cached.unsqueeze(0)
            return value.get_ss_cond(patches, ["vggt"], 1)["cond"]

        historical = types.ModuleType("ar_ss_flow.build_pose_lifting_cache")
        historical.extract_stock_condition = extract_stock_condition
        with mock.patch.dict(
            sys.modules,
            {"ar_ss_flow.build_pose_lifting_cache": historical},
        ):
            context, replay = build_cache._extract_historical_native_ss_condition(
                pipeline,
                ["official-view"],
                cached,
                uid="fixture",
            )
        self.assertIs(context, expected_context)
        self.assertTrue(replay["passed"])
        self.assertEqual(replay["max_abs"], 0.0)
        self.assertEqual(replay["mean_abs"], 0.0)

    def test_materialize_cleanup_only_deletes_locally_bound_names(self) -> None:
        tree = ast.parse(Path(build_cache.__file__).read_text(encoding="utf-8"))
        materialize = next(
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "_materialize"
        )
        locally_bound = {
            node.id
            for node in ast.walk(materialize)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
        }
        deleted = {
            node.id
            for node in ast.walk(materialize)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Del)
        }
        self.assertTrue(deleted)
        self.assertEqual(deleted - locally_bound, set())

    @mock.patch("official_ss_with_vggt_perf_v1.train.TorchDistributedDataParallel")
    def test_ddp_policy_keeps_nested_sample_on_cpu(self, ddp) -> None:
        sentinel = object()
        ddp.return_value = sentinel
        module = object()
        result = train._ddp_preserve_cpu_sample(
            module,
            device_ids=[3],
            output_device=3,
            broadcast_buffers=False,
            find_unused_parameters=False,
        )
        self.assertIs(result, sentinel)
        ddp.assert_called_once_with(
            module,
            device_ids=None,
            broadcast_buffers=False,
            find_unused_parameters=False,
        )


if __name__ == "__main__":
    unittest.main()
