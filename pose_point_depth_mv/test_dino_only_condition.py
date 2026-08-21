from __future__ import annotations

from types import SimpleNamespace
import unittest

import torch
from torch import nn

from pose_point_depth_mv.dino_only_condition import (
    DINO_ONLY_LIFTING_VERSION,
    build_dino_only_contexts,
    deterministic_token_indices,
    dino_only_feature_metadata,
    select_dino_patch_features,
    tensor_tree_sha256,
    validate_dino_only_lifting_contract,
)
from pose_point_depth_mv.build_direct_slat_cache import (
    DIRECT_DINO_ONLY_PROVENANCE_VERSION,
    tensor_tree_sha256 as direct_slat_tensor_tree_sha256,
    validate_precomputed_slat_condition,
)
from pose_point_depth_mv.native_ss_genrecon import load_trainable_state_dict
from pose_point_depth_mv.native_ss_genrecon import (
    NATIVE_SS_GENRECON_CFG,
    NATIVE_SS_GENRECON_PROJECTION,
    NATIVE_SS_GENRECON_TRAINING,
    NATIVE_SS_GENRECON_VERSION,
)
from pose_point_depth_mv.native_ss_genrecon_no_vggt import (
    NATIVE_SS_NO_VGGT_VERSION,
    NO_VGGT_MODEL_CONTRACT,
    validate_native_ss_no_vggt_checkpoint,
)


class DinoOnlyConditionTest(unittest.TestCase):
    @staticmethod
    def _direct_precomputed_sample() -> tuple[dict, dict, dict]:
        condition = {
            "cond": [torch.randn(1, 7, 1024), torch.randn(1, 7, 1024)],
            "neg_cond": [torch.zeros(1, 7, 1024), torch.zeros(1, 7, 1024)],
        }
        condition_hash = tensor_tree_sha256(condition)
        binding_hash = "b" * 64
        sample = {
            "runtime_condition_sha256": condition_hash,
            "slat_condition_provenance": {
                "source": DIRECT_DINO_ONLY_PROVENANCE_VERSION,
                "condition_tree_sha256": condition_hash,
                "sample_input_binding_sha256": binding_hash,
                "vggt_model_loaded": False,
                "vggt_model_executed": False,
            },
            "dino_only_direct_build": {
                "version": DIRECT_DINO_ONLY_PROVENANCE_VERSION,
                "sample_input_binding_sha256": binding_hash,
                "vggt_model_loaded": False,
                "vggt_model_executed": False,
            },
            "dino_only_context_contract": {"vggt_model_executed": False},
        }
        row = {"source_input_binding_sha256": binding_hash}
        return sample, condition, row

    def test_direct_precomputed_condition_provenance_is_accepted(self) -> None:
        sample, condition, row = self._direct_precomputed_sample()
        result = validate_precomputed_slat_condition(
            sample, condition, manifest_row=row
        )
        self.assertEqual(
            result["condition_tree_sha256"], sample["runtime_condition_sha256"]
        )
        self.assertEqual(
            result["sample_input_binding_sha256"],
            row["source_input_binding_sha256"],
        )
        self.assertFalse(result["vggt_model_executed"])

    def test_direct_precomputed_condition_rejects_binding_tamper(self) -> None:
        sample, condition, row = self._direct_precomputed_sample()
        row["source_input_binding_sha256"] = "0" * 64
        with self.assertRaisesRegex(RuntimeError, "provenance differs"):
            validate_precomputed_slat_condition(sample, condition, manifest_row=row)

    def test_direct_precomputed_condition_rejects_tree_tamper(self) -> None:
        sample, condition, row = self._direct_precomputed_sample()
        condition["cond"][0][0, 0, 0] += 1
        with self.assertRaisesRegex(RuntimeError, "condition tree changed"):
            validate_precomputed_slat_condition(sample, condition, manifest_row=row)

    def test_legacy_precomputed_condition_remains_supported(self) -> None:
        condition = {
            "cond": [torch.randn(1, 7, 1024)],
            "neg_cond": [torch.zeros(1, 7, 1024)],
        }
        tree_hash = tensor_tree_sha256(condition)
        sample = {
            "runtime_condition_sha256": "condition-identity",
            "slat_condition_provenance": {
                "model_input": "/tmp/input.pt",
                "model_input_sha256": "input-hash",
                "condition_sha256": "condition-identity",
                "condition_tree_sha256": tree_hash,
            },
        }
        result = validate_precomputed_slat_condition(sample, condition)
        self.assertEqual(result["model_input"], "/tmp/input.pt")
        self.assertEqual(result["condition_tree_sha256"], tree_hash)

    def test_condition_hash_matches_direct_slat_consumer(self) -> None:
        condition = {
            "cond": [torch.randn(1, 7, 1024), torch.randn(1, 7, 1024)],
            "neg_cond": [torch.zeros(1, 7, 1024), torch.zeros(1, 7, 1024)],
        }
        self.assertEqual(
            tensor_tree_sha256(condition),
            direct_slat_tensor_tree_sha256(condition),
        )

    def test_selects_only_trailing_dino_from_full_or_dino_input(self) -> None:
        dino = torch.randn(2, 9, 1024)
        poisoned_vggt = torch.full((2, 9, 2048), float("nan"))
        full = torch.cat((poisoned_vggt, dino), dim=-1)
        torch.testing.assert_close(select_dino_patch_features(full), dino)
        torch.testing.assert_close(select_dino_patch_features(dino), dino)

    def test_contexts_are_deterministic_and_have_no_vggt_dependency(self) -> None:
        torch.manual_seed(4)
        dino = torch.randn(4, 49, 1024)
        left = torch.cat((torch.randn(4, 49, 2048), dino), dim=-1)
        right = torch.cat((torch.randn(4, 49, 2048) * 1000, dino), dim=-1)
        left_context = build_dino_only_contexts(left, ss_context_tokens=101)
        right_context = build_dino_only_contexts(right, ss_context_tokens=101)
        torch.testing.assert_close(
            left_context["stock_condition"], right_context["stock_condition"]
        )
        self.assertEqual(left_context["stock_condition"].shape, (1, 101, 1024))
        self.assertEqual(len(left_context["slat_condition"]["cond"]), 4)
        self.assertEqual(
            left_context["slat_condition"]["cond"][0].shape, (1, 49, 1024)
        )
        for negative in left_context["slat_condition"]["neg_cond"]:
            self.assertEqual(int(torch.count_nonzero(negative)), 0)

    def test_integer_midpoint_cap_is_unique_ordered_and_stable(self) -> None:
        indices = deterministic_token_indices(10952, 4096)
        self.assertEqual(len(indices), 4096)
        self.assertTrue(bool(torch.all(indices[1:] > indices[:-1]).item()))
        self.assertGreaterEqual(int(indices[0]), 0)
        self.assertLess(int(indices[-1]), 10952)
        torch.testing.assert_close(indices, deterministic_token_indices(10952, 4096))

    def test_token_indices_follow_requested_device(self) -> None:
        indices = deterministic_token_indices(19, 7, device=torch.device("cpu"))
        self.assertEqual(indices.device, torch.device("cpu"))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_context_build_keeps_index_select_on_cuda(self) -> None:
        dino = torch.randn(2, 9, 1024, device="cuda")
        contexts = build_dino_only_contexts(dino, ss_context_tokens=11)
        self.assertEqual(contexts["stock_condition"].device.type, "cuda")
        self.assertEqual(contexts["stock_condition"].shape, (1, 11, 1024))

    def test_formal_dino_only_manifest_contract(self) -> None:
        dataset = SimpleNamespace(
            visual_feature_dim=1024,
            feature_metadata=dino_only_feature_metadata(patch_count=1369),
            config={
                "no_vggt": {
                    "version": DINO_ONLY_LIFTING_VERSION,
                    "stock_condition_source": "deterministic_dino_token_context",
                    "slat_condition_source": "per_view_raw_dino_token_context",
                    "depth_policy": "zero_placeholder_not_consumed",
                }
            },
            config_hash="abc",
        )
        contract = validate_dino_only_lifting_contract(dataset)
        self.assertEqual(contract["vggt_feature_dim"], 0)
        self.assertEqual(contract["patch_side"], 37)
        dataset.visual_feature_dim = 3072
        with self.assertRaisesRegex(ValueError, "DINO-only lifting contract"):
            validate_dino_only_lifting_contract(dataset)

    def test_trainable_state_migration_is_exact(self) -> None:
        class Tiny(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.frozen = nn.Parameter(torch.zeros(2), requires_grad=False)
                self.adapter = nn.Linear(3, 2)

        source = Tiny()
        state = {
            name: torch.randn_like(parameter)
            for name, parameter in source.named_parameters()
            if parameter.requires_grad
        }
        destination = Tiny()
        load_trainable_state_dict(destination, state)
        for name, parameter in destination.named_parameters():
            if parameter.requires_grad:
                torch.testing.assert_close(parameter, state[name])
        torch.testing.assert_close(destination.frozen, torch.zeros_like(destination.frozen))

    def test_v2_checkpoint_is_parent_only_and_new_format_is_distinct(self) -> None:
        checkpoint = {
            "format": NATIVE_SS_GENRECON_VERSION,
            "step": 7,
            "args": {},
            "model_summary": {
                "pretrained": "pretrained",
                "training_semantics": NATIVE_SS_GENRECON_TRAINING,
                "cfg_semantics": NATIVE_SS_GENRECON_CFG,
                "projection": NATIVE_SS_GENRECON_PROJECTION,
                "condition_scale_policy": "learned_projection_only",
                "post_cfg_cap": False,
                "direct_slat_dependency": False,
            },
            "model_trainable_state": {},
            "ema_trainable_state": {},
            "ema": {"updates": 7, "target_decay": 0.999},
        }
        validate_native_ss_no_vggt_checkpoint(
            checkpoint, pretrained="pretrained", allow_v2_parent=True
        )
        with self.assertRaisesRegex(ValueError, "initialization-only"):
            validate_native_ss_no_vggt_checkpoint(
                checkpoint, pretrained="pretrained", allow_v2_parent=False
            )
        checkpoint["format"] = NATIVE_SS_NO_VGGT_VERSION
        checkpoint["model_summary"]["input_context_contract"] = (
            NO_VGGT_MODEL_CONTRACT
        )
        checkpoint["data_identity"] = {
            "feature_contract": {"no_vggt": {"vggt_feature_dim": 0}}
        }
        validate_native_ss_no_vggt_checkpoint(
            checkpoint, pretrained="pretrained", allow_v2_parent=False
        )


if __name__ == "__main__":
    unittest.main()
