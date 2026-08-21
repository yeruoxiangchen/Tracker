from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch

from official_ss_with_vggt_perf_v1.evaluate_ss_slat import (
    build_extended_comparisons,
)
from official_ss_with_vggt_perf_v1 import evaluate_ss_slat as endpoint_evaluator
from official_ss_with_vggt_perf_v1.model import EVAL_AGGREGATE_FORMAT
from official_ss_with_vggt_perf_v1.ss_slat_endpoint import (
    WithVGGTSSSLatEndpointDataset,
    build_trained_slat_pipeline,
    endpoint_contract,
    load_vss_deployment,
    official_target_contract,
)
from pose_point_depth_mv.native_ss_genrecon import (
    canonical_json_sha256,
    sha256_file,
)
from pose_point_depth_mv.native_slat_genrecon_with_vggt_official import (
    NATIVE_SLAT_OFFICIAL_WITH_VGGT_VERSION,
)
from pose_point_depth_mv.proobjaverse_official_ss import official_domain_contract
from pose_point_depth_mv.proobjaverse_official_slat_protocol import canonical_sha256


PROTOCOL = "a" * 64
OFFICIAL_SLAT_SHA = "b" * 64


def _lifting() -> dict[str, torch.Tensor | str | float]:
    return {
        "view_ids": torch.arange(8, dtype=torch.int64),
        "visual_patch_features": torch.arange(8 * 2 * 4, dtype=torch.float32).reshape(8, 2, 4),
        "intrinsics": torch.eye(3).repeat(8, 1, 1),
        "extrinsics": torch.eye(4).repeat(8, 1, 1),
        "stock_condition": torch.zeros((1, 4, 1024), dtype=torch.float16),
    }


class _FakeSLatDataset:
    def __init__(self, *_args, **_kwargs) -> None:
        self.rows = [
            {
                "uid": "uid-a",
                "object_uid": "object-a",
                "target_file_sha256": OFFICIAL_SLAT_SHA,
                "source_lh_slat_sha256": OFFICIAL_SLAT_SHA,
            }
        ]
        self.config = {
            "target_source": {
                "support_policy": "official_gt_slat_coordinates",
                "split": "train",
                "coordinate_resolution": 64,
                "protocol_sha256": PROTOCOL,
            }
        }
        self.config_hash = "slat-config"
        self.slat_normalization = {"mean": [0.0], "std": [1.0]}
        self.slat_normalization_hash = "normalization"
        self.lifting = object()
        self.slat = object()
        self.pair_identity = "slat-pair"
        self.sidecar_contract_hash = "slat-contract"

    def __getitem__(self, _index: int):
        return {
            "uid": "uid-a",
            "object_uid": "object-a",
            "target_coords": torch.tensor([[0, 1, 2, 3], [0, 4, 5, 6]]),
            "lifting_sample": _lifting(),
            "condition": {"cond": [torch.ones((1, 2, 1024))]},
        }


class _FakeSSDataset:
    def __init__(self, *_args, **_kwargs) -> None:
        self.rows = [
            {
                "uid": "uid-a",
                "object_uid": "object-a",
                "official_lh_slat_sha256": OFFICIAL_SLAT_SHA,
                "official_ss_roundtrip_iou": 1.0,
            }
        ]
        self.pair_identity = "ss-pair"
        self.sidecar_contract_hash = "ss-contract"
        self.sidecar_contract = {"protocol_sha256": PROTOCOL}
        self.config = {
            "official_ss_targets": {
                "domain_contract": official_domain_contract(
                    protocol_sha256=PROTOCOL,
                    encoder_pretrained="encoder",
                    decoder_pretrained="Stable-X/trellis-vggt-v0-2",
                    latent_dtype="float16",
                    minimum_roundtrip_iou=0.9,
                )
            }
        }

    def __getitem__(self, _index: int):
        value = _lifting()
        value.update(
            {
                "uid": "uid-a",
                "object_uid": "object-a",
                "target_coords": torch.tensor([[1, 2, 3], [4, 5, 6]]),
                "stock_condition": torch.full(
                    (1, 4096, 1024), 2.0, dtype=torch.float16
                ),
                "with_vggt_sidecar_path": "/fixture/ss.pt",
            }
        )
        return value


class _MissingSSDataset(_FakeSSDataset):
    def __init__(self, *_args, **_kwargs) -> None:
        super().__init__()
        self.rows = [{"uid": "different", "object_uid": "object-b"}]


class _DifferentOfficialSourceSSDataset(_FakeSSDataset):
    def __init__(self, *_args, **_kwargs) -> None:
        super().__init__()
        self.rows[0]["official_lh_slat_sha256"] = "c" * 64


class _DifferentProtocolSSDataset(_FakeSSDataset):
    def __init__(self, *_args, **_kwargs) -> None:
        super().__init__()
        other = "d" * 64
        self.sidecar_contract["protocol_sha256"] = other
        self.config["official_ss_targets"]["domain_contract"] = official_domain_contract(
            protocol_sha256=other,
            encoder_pretrained="encoder",
            decoder_pretrained="Stable-X/trellis-vggt-v0-2",
            latent_dtype="float16",
            minimum_roundtrip_iou=0.9,
        )


class _ProjectedSSDataset(_FakeSSDataset):
    def __init__(self, *_args, **_kwargs) -> None:
        super().__init__()
        self.rows[0]["official_ss_roundtrip_iou"] = 99.0 / 101.0

    def __getitem__(self, _index: int):
        value = super().__getitem__(_index)
        value["target_coords"] = torch.tensor(
            [[index, 0, 0] for index in range(99)] + [[1000, 0, 0]],
            dtype=torch.int64,
        )
        return value


class _ProjectedSLatDataset(_FakeSLatDataset):
    def __getitem__(self, _index: int):
        value = super().__getitem__(_index)
        value["target_coords"] = torch.tensor(
            [[0, index, 0, 0] for index in range(100)], dtype=torch.int64
        )
        return value


class EndpointDatasetTests(unittest.TestCase):
    def _make(
        self,
        root: Path,
        ss_factory=_FakeSSDataset,
        slat_factory=_FakeSLatDataset,
    ):
        manifest = root / "ss_manifest.json"
        manifest.write_text("{}\n", encoding="utf-8")
        with (
            mock.patch.object(
                WithVGGTSSSLatEndpointDataset,
                "slat_dataset_factory",
                slat_factory,
            ),
            mock.patch.object(
                WithVGGTSSSLatEndpointDataset,
                "ss_dataset_factory",
                ss_factory,
            ),
        ):
            return WithVGGTSSSLatEndpointDataset(
                "slat.json",
                "lifting.json",
                ss_cache_manifest=manifest,
            )

    def test_join_replaces_only_ss_stock_condition(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            dataset = self._make(Path(temporary))
            sample = dataset[0]
        self.assertEqual(
            sample["lifting_sample"]["stock_condition_source"],
            "native_ss_vggt_cond_sidecar",
        )
        self.assertTrue(torch.all(sample["lifting_sample"]["stock_condition"] == 2))
        self.assertEqual(sample["condition"]["cond"][0].shape[-1], 1024)
        self.assertFalse(dataset.endpoint_identity["gt_support_used_as_slat_input"])

    def test_pair_mismatch_fails_before_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(RuntimeError, "UID join incomplete"):
                self._make(Path(temporary), _MissingSSDataset)

    def test_same_source_decoder_projected_support_is_valid(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            dataset = self._make(
                Path(temporary),
                _ProjectedSSDataset,
                _ProjectedSLatDataset,
            )
            sample = dataset[0]
        relation = sample["with_vggt_target_join"]
        self.assertFalse(relation["supports_exact"])
        self.assertAlmostEqual(relation["roundtrip_iou"], 99.0 / 101.0)
        self.assertEqual(relation["runtime_target_source"], "slat_raw_official_lh_slat")
        self.assertEqual(len(sample["target_coords"]), 100)

    def test_different_official_source_sha_fails_before_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(RuntimeError, "official LH-SLat SHA256 differs"):
                self._make(Path(temporary), _DifferentOfficialSourceSSDataset)

    def test_different_official_protocol_fails_before_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(RuntimeError, "official protocols differ"):
                self._make(Path(temporary), _DifferentProtocolSSDataset)

    def test_train_target_contract_is_diagnostic_and_predicted_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            dataset = self._make(Path(temporary))
            with mock.patch(
                "official_ss_with_vggt_perf_v1.ss_slat_endpoint."
                "validate_dino_only_lifting_contract"
            ):
                target = official_target_contract(dataset)
        self.assertEqual(target["evaluation_role"], "training_overlap_fit_diagnosis")
        self.assertEqual(target["slat_support_input"], "predicted_only")
        self.assertFalse(target["gt_support_used_as_slat_input"])

    def test_completed_science_negative_worker_is_annotated_before_exit(self) -> None:
        args = mock.Mock(ss_cache_manifest="ss-cache.json")
        with (
            mock.patch.object(endpoint_evaluator, "activate_ss_cache_manifest"),
            mock.patch.object(
                endpoint_evaluator,
                "_BASE_RUN_WORKER",
                side_effect=SystemExit(2),
            ),
            mock.patch.object(endpoint_evaluator, "_annotate_worker") as annotate,
        ):
            with self.assertRaises(SystemExit) as raised:
                endpoint_evaluator._run_worker(args)
        self.assertEqual(raised.exception.code, 2)
        annotate.assert_called_once_with(args)


class DeploymentTests(unittest.TestCase):
    def _report(self, root: Path, *, integrity: bool = True) -> Path:
        checkpoint = root / "step_002000.pt"
        checkpoint.write_bytes(b"frozen-vss-checkpoint")
        uids = ["dev-a", "dev-b"]
        checks = {
            "correct_record_matrix_exact": integrity,
            "pose_control_record_matrix_exact": True,
            "stock_baseline_nonempty": True,
            "disabled_stock_equivalence": True,
            "iou_gain_mean": False,
        }
        report = {
            "format": EVAL_AGGREGATE_FORMAT,
            "passed": False,
            "formal": False,
            "object_count": len(uids),
            "object_uids": uids,
            "object_uid_hash": canonical_json_sha256(sorted(uids)),
            "official_ss_domain_contract": official_domain_contract(
                protocol_sha256=PROTOCOL,
                encoder_pretrained="encoder",
                decoder_pretrained="Stable-X/trellis-vggt-v0-2",
                latent_dtype="float16",
                minimum_roundtrip_iou=0.9,
            ),
            "checks": checks,
            "deployment": {
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha256_file(checkpoint),
                "checkpoint_step": 2000,
                "weights": "ema",
                "cfg_strength": 5.0,
                "steps": 25,
                "cfg_interval": [0.5, 1.0],
                "guidance_rescale": 0.0,
                "rescale_t": 3.0,
                "amp_dtype": "bf16",
            },
        }
        report["report_sha256"] = canonical_json_sha256(report)
        path = root / "report.json"
        path.write_text(json.dumps(report), encoding="utf-8")
        return path

    def test_science_negative_report_remains_valid_deployment_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            payload, binding = load_vss_deployment(
                self._report(Path(temporary), integrity=True)
            )
        self.assertFalse(payload["passed"])
        self.assertFalse(binding["science_passed"])
        self.assertEqual(binding["false_checks"], ["iou_gain_mean"])

    def test_runtime_integrity_failure_is_never_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(RuntimeError, "runtime integrity"):
                load_vss_deployment(self._report(Path(temporary), integrity=False))

    def test_branch_contract_contains_all_direct_comparisons(self) -> None:
        contract = endpoint_contract()
        self.assertEqual(set(contract["branches"]), {"A", "B", "C"})
        self.assertEqual(set(contract["comparisons"]), {"B_minus_A", "C_minus_B", "C_minus_A"})
        self.assertFalse(contract["gt_support_used_as_slat_input"])

    def test_extended_metrics_preserve_three_thresholds_and_directions(self) -> None:
        records = []
        for seed in (42, 43, 44):
            for branch, offset in (("stock", 0.0), ("native", 0.1), ("native_trained", 0.2)):
                surface = {
                    "pred_to_gt_mean": 1.0 - offset,
                    "gt_to_pred_mean": 1.1 - offset,
                    "chamfer_l1": 1.2 - offset,
                    "chamfer_l2": 1.3 - offset,
                    "normal_consistency": 0.5 + offset,
                }
                for threshold in ("0p01", "0p02", "0p05"):
                    surface[f"precision_{threshold}"] = 0.4 + offset
                    surface[f"recall_{threshold}"] = 0.3 + offset
                    surface[f"fscore_{threshold}"] = 0.35 + offset
                records.append(
                    {
                        "branch": branch,
                        "object_uid": "object-a",
                        "seed": seed,
                        "passed": True,
                        "surface": surface,
                        "structure": {"largest_component_ratio": 0.6 + offset},
                    }
                )
        result = build_extended_comparisons(
            [{"mesh_branch_records": records}],
            expected_uids={"object-a"},
            seeds=[42, 43, 44],
            bootstrap_samples=20,
        )
        summary = result[
            "C_minus_A__full_VSS_plus_V_endpoint_increment"
        ]["summary"]
        self.assertAlmostEqual(summary["chamfer_l2_improvement"]["mean"], 0.2)
        self.assertAlmostEqual(summary["precision_0p01_delta"]["mean"], 0.2)
        self.assertAlmostEqual(summary["recall_0p05_delta"]["mean"], 0.2)
        self.assertAlmostEqual(summary["fscore_0p02_delta"]["mean"], 0.2)

    def test_trained_slat_builder_accepts_base_evaluator_membership_contract(self) -> None:
        normalization = {"mean": [0.0], "std": [1.0]}
        checkpoint = {
            "format": NATIVE_SLAT_OFFICIAL_WITH_VGGT_VERSION,
            "step": 15000,
            "data_identity": {
                "native_ss": {"binding": "fixture"},
                "object_uids": ["object-a", "object-b"],
                "target_decoder_audit": {"protocol_sha256": PROTOCOL},
            },
            "model_summary": {"upstream_native_ss": {"binding": "fixture"}},
            "args": {
                "lora_rank": 8,
                "lora_alpha": 16,
                "condition_channels": 1024,
            },
            "ema_trainable_state": {},
        }
        dataset = mock.Mock()
        dataset.config = {"target_source": {"protocol_sha256": PROTOCOL}}
        dataset.slat_normalization_hash = canonical_sha256(normalization)
        sampler = mock.Mock()
        model = mock.Mock()
        decoder = mock.Mock()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "step_015000.pt"
            path.write_bytes(b"fixture checkpoint identity")
            with (
                mock.patch(
                    "official_ss_with_vggt_perf_v1.ss_slat_endpoint."
                    "validate_native_slat_official_with_vggt_checkpoint"
                ),
                mock.patch(
                    "official_ss_with_vggt_perf_v1.ss_slat_endpoint."
                    "build_native_slat_official_with_vggt_components",
                    return_value=(
                        sampler,
                        model,
                        decoder,
                        {"fixture": True},
                        {},
                        normalization,
                    ),
                ),
                mock.patch(
                    "official_ss_with_vggt_perf_v1.ss_slat_endpoint."
                    "load_slat_trainable_state_dict"
                ),
            ):
                result = build_trained_slat_pipeline(
                    checkpoint_path=path,
                    weights="ema",
                    pretrained="Stable-X/trellis-vggt-v0-2",
                    stock_freeze={},
                    dataset=dataset,
                    expected_step=15000,
                    device=torch.device("cpu"),
                    evaluation_object_uids=["object-a"],
                    allow_target_protocol_mismatch=False,
                    expected_training_membership="all_training",
                    checkpoint_payload=checkpoint,
                )
        membership = result["checkpoint_evaluation_membership"]
        self.assertEqual(membership["training_overlap_count"], 1)
        self.assertTrue(membership["all_evaluation_objects_in_checkpoint_training"])


if __name__ == "__main__":
    unittest.main()
