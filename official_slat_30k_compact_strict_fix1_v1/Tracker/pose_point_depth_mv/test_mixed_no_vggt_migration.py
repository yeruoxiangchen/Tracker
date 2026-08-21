from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import torch

from ar_ss_flow.pose_lifting import LIFTING_CACHE_VERSION, LIFTING_METADATA_NAMES, schema_hash
from ar_ss_flow.shared_object_preprocessing import shared_preprocessing_contract
from pose_point_depth_mv.dataset_tools.build_mixed_no_vggt_manifest import build_lifting
from pose_point_depth_mv.dino_only_condition import (
    DINO_ONLY_LIFTING_VERSION,
    dino_only_feature_metadata,
)
from pose_point_depth_mv.mixed_no_vggt_data import (
    DomainBalancedDistributedSampler,
    MixedPoseLiftingCacheDataset,
)
from pose_point_depth_mv.native_ss_genrecon import NATIVE_SS_GENRECON_VERSION
from pose_point_depth_mv.native_ss_genrecon_no_vggt import (
    validate_no_vggt_evaluation_cache_contract,
)
from pose_point_depth_mv.omni_real_benchmark_common import sha256_file
from pose_point_depth_mv.real_full_no_vggt_migration import (
    build_migration_contract,
    load_migration_contract,
    migration_summary,
    validate_destination_migration,
    validate_parent_payload,
)


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def dino_manifest(path: Path, prefix: str, count: int) -> Path:
    config = {
        "geometric_preprocessing": shared_preprocessing_contract(
            resolution=518, foreground_margin=1.1, alpha_threshold=0.8
        ),
        "no_vggt": {
            "version": DINO_ONLY_LIFTING_VERSION,
            "stock_condition_source": "deterministic_dino_token_context",
            "slat_condition_source": "per_view_raw_dino_token_context",
            "depth_policy": "zero_placeholder_not_consumed",
            "vggt_model_executed": False,
        }
    }
    payload = {
        "format": LIFTING_CACHE_VERSION,
        "output_dir": str(path.parent),
        "visual_feature_dim": 1024,
        "feature_metadata": dino_only_feature_metadata(patch_count=1369),
        "metadata_names": list(LIFTING_METADATA_NAMES),
        "metadata_schema_hash": schema_hash(),
        "config": config,
        "config_hash": f"{prefix}-config",
        "samples": [
            {
                "uid": f"{prefix}-{index}",
                "object_uid": f"{prefix}-object-{index}",
                "cache_file": f"unused-{index}.pt",
            }
            for index in range(count)
        ],
    }
    write_json(path, payload)
    return path


class MixedNoVggtDataTest(unittest.TestCase):
    def test_manifest_and_equal_domain_sampler(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            synthetic = dino_manifest(root / "synthetic.json", "syn", 4)
            real = dino_manifest(root / "real.json", "real", 2)
            output = root / "mixed.json"
            args = type(
                "Args",
                (),
                {
                    "synthetic_manifest": str(synthetic),
                    "real_manifest": str(real),
                    "output": str(output),
                    "expected_synthetic_objects": 4,
                    "expected_real_objects": 2,
                },
            )()
            payload = build_lifting(args)
            self.assertEqual(payload["sample_count"], 6)
            dataset = MixedPoseLiftingCacheDataset(output)
            self.assertEqual(len(dataset), 6)
            rank0 = DomainBalancedDistributedSampler(
                dataset.rows, num_replicas=2, rank=0, seed=42
            )
            rank1 = DomainBalancedDistributedSampler(
                dataset.rows, num_replicas=2, rank=1, seed=42
            )
            indices0 = list(rank0)
            indices1 = list(rank1)
            self.assertEqual(len(indices0), len(indices1))
            global_domains = []
            for left, right in zip(indices0, indices1):
                global_domains.extend(
                    (dataset.rows[left]["_mixed_domain"], dataset.rows[right]["_mixed_domain"])
                )
            self.assertEqual(global_domains.count("synthetic"), global_domains.count("real"))

            synthetic_dataset = dataset.domain_datasets["synthetic"]
            training_identity = {
                "config_hash": dataset.config_hash,
                "feature_contract": {
                    "config_hash": dataset.config_hash,
                    "mixed_domains": {
                        name: {
                            "config_hash": component.config_hash,
                            "contract": dataset.domain_contracts[name],
                        }
                        for name, component in dataset.domain_datasets.items()
                    },
                },
            }
            contract = validate_no_vggt_evaluation_cache_contract(
                synthetic_dataset, training_identity=training_identity
            )
            self.assertEqual(
                contract["evaluation_training_binding"]["domain"], "synthetic"
            )

            bad_hash = dict(training_identity)
            bad_hash["feature_contract"] = {
                **training_identity["feature_contract"],
                "mixed_domains": {
                    name: {**binding, "config_hash": f"wrong-{name}"}
                    for name, binding in training_identity["feature_contract"][
                        "mixed_domains"
                    ].items()
                },
            }
            with self.assertRaisesRegex(RuntimeError, "exactly one frozen mixed"):
                validate_no_vggt_evaluation_cache_contract(
                    synthetic_dataset, training_identity=bad_hash
                )

            bad_contract = dict(training_identity)
            bad_contract["feature_contract"] = {
                **training_identity["feature_contract"],
                "mixed_domains": {
                    name: {
                        **binding,
                        "contract": {
                            **binding["contract"],
                            "depth_policy": "tampered",
                        },
                    }
                    for name, binding in training_identity["feature_contract"][
                        "mixed_domains"
                    ].items()
                },
            }
            with self.assertRaisesRegex(RuntimeError, "differs from frozen synthetic"):
                validate_no_vggt_evaluation_cache_contract(
                    synthetic_dataset, training_identity=bad_contract
                )

    def test_resume_offset_matches_uninterrupted_suffix(self) -> None:
        rows = [
            {"uid": f"s{i}", "object_uid": f"s{i}", "_mixed_domain": "synthetic"}
            for i in range(5)
        ] + [
            {"uid": f"r{i}", "object_uid": f"r{i}", "_mixed_domain": "real"}
            for i in range(3)
        ]
        full = DomainBalancedDistributedSampler(rows, num_replicas=1, rank=0, seed=7)
        full_indices = list(full)
        resumed = DomainBalancedDistributedSampler(
            rows, num_replicas=1, rank=0, seed=7, resume_micro_step=3
        )
        self.assertEqual(list(resumed), full_indices[3:])


class MigrationContractTest(unittest.TestCase):
    def test_real_full_ema_contract_and_destination_lineage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lifting_path = root / "full_real_lifting.json"
            rows = [
                {
                    "uid": f"real-{index}",
                    "object_uid": f"real-{index}",
                    "source": "omni_real_video",
                }
                for index in range(3)
            ]
            write_json(
                lifting_path,
                {
                    "visual_feature_dim": 3072,
                    "feature_metadata": {
                        "vggt_feature_dim": 2048,
                        "dino_feature_dim": 1024,
                    },
                    "samples": rows,
                },
            )
            identity = {
                "manifest": str(lifting_path),
                "manifest_sha256": sha256_file(lifting_path),
                "object_count": 3,
            }
            checkpoint_path = root / "parent.pt"
            checkpoint = {
                "format": NATIVE_SS_GENRECON_VERSION,
                "step": 10,
                "data_identity": identity,
                "model_trainable_state": {"weight": torch.ones(2, 3)},
                "ema_trainable_state": {"weight": torch.zeros(2, 3)},
            }
            torch.save(checkpoint, checkpoint_path)
            report_path = root / "report.json"
            write_json(
                report_path,
                {
                    "format": NATIVE_SS_GENRECON_VERSION,
                    "completed": True,
                    "passed": True,
                    "step": 10,
                    "checkpoint": str(checkpoint_path),
                    "evaluation_weights": "ema",
                    "data_identity": identity,
                },
            )
            payload = build_migration_contract(
                stage="ss",
                parent_checkpoint=checkpoint_path,
                parent_report=report_path,
                min_real_objects=3,
            )
            contract_path = root / "contract.json"
            write_json(contract_path, payload)
            loaded = load_migration_contract(contract_path, stage="ss")
            validate_parent_payload(checkpoint, loaded, stage="ss")
            destination = {
                "model_summary": {
                    "migration_contract": migration_summary(loaded),
                    "initialization": {
                        "checkpoint_sha256": loaded["parent"]["checkpoint_sha256"],
                        "weights": "ema",
                        "optimizer_inherited": False,
                        "ema_reinitialized_from_selected_weights": True,
                    },
                }
            }
            validate_destination_migration(destination, loaded)


if __name__ == "__main__":
    unittest.main()
