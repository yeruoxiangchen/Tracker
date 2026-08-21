from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
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
from pose_point_depth_mv import audit_proobjaverse_official_slat_cache_v1_v2
from pose_point_depth_mv.dino_only_condition import (
    DINO_ONLY_LIFTING_VERSION,
    build_dino_only_contexts,
    dino_only_feature_metadata,
)
from pose_point_depth_mv.direct_slat_flow import DIRECT_SLAT_CACHE_VERSION
from pose_point_depth_mv.proobjaverse_official_slat_compact import (
    COMPACT_LAYOUT_VERSION,
    COMPACT_LIFTING_MANIFEST_FORMAT,
    COMPACT_OBJECT_FORMAT,
    COMPACT_SLAT_MANIFEST_FORMAT,
)


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


class OfficialSLatCacheV1V2FixtureTests(unittest.TestCase):
    def test_one_object_runtime_and_projection_are_exact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            uid = "fixture_uid"
            coords = np.asarray(((30, 30, 30), (31, 31, 31)), dtype=np.uint8)
            feats = np.arange(16, dtype=np.float32).reshape(2, 8) / 16.0
            target = root / "target.npz"
            np.savez(target, coords=coords, feats=feats)

            slat_config = {"condition_arch": "native_ss_genrecon_v2"}
            slat_hash = canonical_json_sha256(slat_config)
            normalization = {"mean": [0.0] * 8, "std": [1.0] * 8}
            normalization_hash = canonical_json_sha256(normalization)
            lifting_config = {
                "no_vggt": {
                    "version": DINO_ONLY_LIFTING_VERSION,
                    "stock_condition_source": "deterministic_dino_token_context",
                    "slat_condition_source": "per_view_raw_dino_token_context",
                    "depth_policy": "zero_placeholder_not_consumed",
                }
            }
            lifting_hash = canonical_json_sha256(lifting_config)
            visual = (
                torch.arange(2 * 4 * 1024, dtype=torch.float32)
                .reshape(2, 4, 1024)
                .remainder(127)
                .div(127)
                .to(torch.float16)
            )
            contexts = build_dino_only_contexts(visual, ss_context_tokens=8)
            intrinsics = torch.tensor(
                (((2.0, 0.0, 2.0), (0.0, 2.0, 2.0), (0.0, 0.0, 1.0)),)
            ).repeat(2, 1, 1)
            extrinsics = torch.eye(4).repeat(2, 1, 1)
            extrinsics[:, 2, 3] = 2.0
            view_ids = torch.tensor((3, 7), dtype=torch.int64)

            support_path = root / "support.pt"
            physical_path = root / "physical.pt"
            condition_path = root / "condition.pt"
            torch.save(
                {
                    "format": DIRECT_SLAT_CACHE_VERSION,
                    "uid": uid,
                    "object_uid": uid,
                    "config_hash": slat_hash,
                    "seed": 42,
                    "corrected_ss": torch.zeros((1, 8, 16, 16, 16)),
                    "occupancy_logits64": torch.zeros((1, 1, 64, 64, 64)),
                    "corrected_coords64": torch.from_numpy(coords.astype(np.int32)),
                },
                support_path,
            )
            torch.save(
                {
                    "format": DIRECT_SLAT_CACHE_VERSION,
                    "uid": uid,
                    "object_uid": uid,
                    "config_hash": slat_hash,
                    "unused_native_placeholder": True,
                },
                physical_path,
            )
            torch.save(
                {
                    "format": DIRECT_SLAT_CACHE_VERSION,
                    "uid": uid,
                    "object_uid": uid,
                    "config_hash": slat_hash,
                    "condition": contexts["slat_condition"],
                },
                condition_path,
            )
            legacy_slat = root / "slat_v1.json"
            _write_json(
                legacy_slat,
                {
                    "format": DIRECT_SLAT_CACHE_VERSION,
                    "materialized": True,
                    "output_dir": str(root),
                    "config": slat_config,
                    "config_hash": slat_hash,
                    "slat_normalization": normalization,
                    "slat_normalization_hash": normalization_hash,
                    "samples": [
                        {
                            "uid": uid,
                            "object_uid": uid,
                            "support_seed": 42,
                            "target_file": str(target),
                            "target_file_sha256": "1" * 64,
                            "support_file": str(support_path),
                            "physical_file": str(physical_path),
                            "condition_file": str(condition_path),
                        }
                    ],
                },
            )

            ss_latent = root / "ss_latent.npz"
            np.savez(
                ss_latent,
                z=np.zeros((8, 16, 16, 16), dtype=np.float32),
                target_coords=coords.astype(np.int64),
            )
            lifting_payload = root / "lifting.pt"
            torch.save(
                {
                    "format": LIFTING_CACHE_VERSION,
                    "uid": uid,
                    "object_uid": uid,
                    "visual_patch_features": visual,
                    "predicted_depth": torch.zeros((2, 4, 4)),
                    "depth_confidence": torch.zeros((2, 4, 4)),
                    "masks": torch.zeros((2, 4, 4)),
                    "intrinsics": intrinsics,
                    "extrinsics": extrinsics,
                    "prior_coords": torch.empty((0, 3)),
                    "prior_confidence": torch.empty((0,)),
                    "stock_condition": contexts["stock_condition"],
                    "view_ids": view_ids,
                    "preprocessing": {},
                    "grid_transform": "identity",
                    "extrinsics_type": "w2c",
                    "camera_forward_sign": 1.0,
                    "ss_latent": str(ss_latent),
                },
                lifting_payload,
            )
            legacy_lifting = root / "lifting_v1.json"
            lifting_metadata = dino_only_feature_metadata(patch_count=4)
            _write_json(
                legacy_lifting,
                {
                    "format": LIFTING_CACHE_VERSION,
                    "output_dir": str(root),
                    "samples": [
                        {
                            "uid": uid,
                            "object_uid": uid,
                            "cache_file": str(lifting_payload),
                            "ss_latent": str(ss_latent),
                        }
                    ],
                    "visual_feature_dim": 1024,
                    "feature_metadata": lifting_metadata,
                    "metadata_names": list(LIFTING_METADATA_NAMES),
                    "metadata_schema_hash": schema_hash(),
                    "config": lifting_config,
                    "config_hash": lifting_hash,
                },
            )

            compact_payload = root / "compact.pt"
            torch.save(
                {
                    "format": COMPACT_OBJECT_FORMAT,
                    "layout": COMPACT_LAYOUT_VERSION,
                    "uid": uid,
                    "object_uid": uid,
                    "slat_config_hash": slat_hash,
                    "lifting_config_hash": lifting_hash,
                    "visual_patch_features": visual,
                    "intrinsics": intrinsics,
                    "extrinsics": extrinsics,
                    "image_size": [4, 4],
                    "grid_transform": "identity",
                    "extrinsics_type": "w2c",
                    "camera_forward_sign": 1.0,
                    "view_ids": view_ids,
                    "preprocessing": {},
                    "context_contract": contexts["context_contract"],
                    "official_gt_support_only": True,
                },
                compact_payload,
            )
            common = {
                "uid": uid,
                "object_uid": uid,
                "compact_file": str(compact_payload),
                "compact_file_sha256": "2" * 64,
            }
            compact_slat = root / "slat_v2.json"
            _write_json(
                compact_slat,
                {
                    "format": COMPACT_SLAT_MANIFEST_FORMAT,
                    "layout": COMPACT_LAYOUT_VERSION,
                    "materialized": True,
                    "output_dir": str(root),
                    "config": slat_config,
                    "config_hash": slat_hash,
                    "slat_normalization": normalization,
                    "slat_normalization_hash": normalization_hash,
                    "samples": [
                        {
                            **common,
                            "support_seed": 42,
                            "target_file": str(target),
                            "target_file_sha256": "1" * 64,
                        }
                    ],
                    "sample_count": 1,
                    "object_count": 1,
                },
            )
            compact_lifting = root / "lifting_v2.json"
            _write_json(
                compact_lifting,
                {
                    "format": COMPACT_LIFTING_MANIFEST_FORMAT,
                    "layout": COMPACT_LAYOUT_VERSION,
                    "output_dir": str(root),
                    "samples": [common],
                    "sample_count": 1,
                    "object_count": 1,
                    "visual_feature_dim": 1024,
                    "feature_metadata": lifting_metadata,
                    "metadata_names": list(LIFTING_METADATA_NAMES),
                    "metadata_schema_hash": schema_hash(),
                    "config": lifting_config,
                    "config_hash": lifting_hash,
                },
            )

            report = root / "equivalence.json"
            argv = [
                "audit_proobjaverse_official_slat_cache_v1_v2",
                "--slat_manifest_v1",
                str(legacy_slat),
                "--lifting_manifest_v1",
                str(legacy_lifting),
                "--slat_manifest_v2",
                str(compact_slat),
                "--lifting_manifest_v2",
                str(compact_lifting),
                "--indices",
                "all",
                "--output",
                str(report),
            ]
            with mock.patch.object(sys, "argv", argv):
                audit_proobjaverse_official_slat_cache_v1_v2.main()
            result = json.loads(report.read_text(encoding="utf-8"))
            self.assertTrue(result["passed"])
            self.assertEqual(result["sample_count"], 1)
            self.assertTrue(result["checks"]["actual_projection_output_exact"])


if __name__ == "__main__":
    unittest.main()
