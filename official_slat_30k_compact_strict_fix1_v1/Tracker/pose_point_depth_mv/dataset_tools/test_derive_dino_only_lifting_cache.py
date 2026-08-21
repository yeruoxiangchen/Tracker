from __future__ import annotations

import tempfile
from pathlib import Path
from unittest import mock
import unittest

import json
import numpy as np
import torch

from ar_ss_flow.local_pose_lifting_flow import PoseLiftingCacheDataset
from ar_ss_flow.pose_lifting import (
    LIFTING_CACHE_VERSION,
    LIFTING_METADATA_NAMES,
    schema_hash,
)
from ar_ss_flow.shared_object_preprocessing import canonical_json_sha256
from pose_point_depth_mv.dataset_tools.derive_dino_only_lifting_cache import (
    derive_sample,
    main as derive_main,
)
from pose_point_depth_mv.dino_only_condition import tensor_tree_sha256


class DeriveDinoOnlyLiftingCacheTest(unittest.TestCase):
    def _source(self, root: Path) -> dict:
        dino = torch.randn(2, 25, 1024)
        return {
            "format": LIFTING_CACHE_VERSION,
            "uid": "sample-1",
            "object_uid": "object-1",
            "visual_patch_features": torch.cat(
                (torch.full((2, 25, 2048), float("nan")), dino), dim=-1
            ),
            "predicted_depth": torch.full((2, 16, 16), float("nan")),
            "depth_confidence": torch.full((2, 16, 16), float("nan")),
            "masks": torch.ones(2, 16, 16),
            "intrinsics": torch.eye(3).repeat(2, 1, 1),
            "extrinsics": torch.eye(4).repeat(2, 1, 1),
            "prior_coords": torch.zeros(4, 3, dtype=torch.int32),
            "prior_confidence": torch.ones(4),
            "stock_condition": torch.full((1, 10, 1024), float("nan")),
            "slat_condition": {
                "cond": [torch.full((1, 25, 1024), float("nan"))] * 2,
                "neg_cond": [torch.full((1, 25, 1024), float("nan"))] * 2,
            },
            "slat_condition_provenance": {"source": "poisoned"},
            "runtime_condition_sha256": "runtime-condition",
            "ss_latent": str(root / "target.npz"),
            "grid_transform": "identity",
            "extrinsics_type": "w2c",
            "camera_forward_sign": 1.0,
            "feature_image_size": [16, 16],
            "vggt_private_tensor": torch.ones(1),
        }

    def test_poisoned_vggt_and_old_conditions_are_not_consumed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "source.pt"
            source = self._source(root)
            torch.save(source, source_path)
            result = derive_sample(
                source,
                source_file=source_path,
                source_sha256="source-hash",
                output_config_hash="config-hash",
                ss_context_tokens=32,
            )
        self.assertEqual(result["visual_patch_features"].shape, (2, 25, 1024))
        self.assertTrue(bool(torch.isfinite(result["visual_patch_features"]).all()))
        self.assertEqual(result["stock_condition"].shape, (1, 32, 1024))
        self.assertEqual(int(torch.count_nonzero(result["predicted_depth"])), 0)
        self.assertEqual(
            int(torch.count_nonzero(result["depth_confidence"] == 1)), 2 * 16 * 16
        )
        self.assertNotIn("vggt_private_tensor", result)
        self.assertEqual(
            tensor_tree_sha256(result["slat_condition"]),
            result["slat_condition_provenance"]["condition_tree_sha256"],
        )
        self.assertFalse(result["dino_only_context_contract"]["vggt_model_executed"])

    def test_cli_output_is_directly_readable_by_lifting_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_dir = root / "source"
            source_dir.mkdir()
            target = root / "target.npz"
            np.savez_compressed(
                target,
                z=np.zeros((8, 16, 16, 16), dtype=np.float32),
                target_coords=np.zeros((1, 3), dtype=np.int64),
            )
            sample = self._source(root)
            sample["ss_latent"] = str(target)
            sample_path = source_dir / "sample.pt"
            torch.save(sample, sample_path)
            config = {"geometric_preprocessing": {}}
            manifest = {
                "format": LIFTING_CACHE_VERSION,
                "output_dir": str(source_dir),
                "visual_feature_dim": 3072,
                "feature_metadata": {
                    "patch_count": 25,
                    "patch_start_idx": 5,
                    "vggt_feature_dim": 2048,
                    "dino_feature_dim": 1024,
                },
                "metadata_names": list(LIFTING_METADATA_NAMES),
                "metadata_schema_hash": schema_hash(),
                "config": config,
                "config_hash": canonical_json_sha256(config),
                "samples": [
                    {
                        "uid": "sample-1",
                        "object_uid": "object-1",
                        "cache_file": "sample.pt",
                    }
                ],
            }
            source_manifest = source_dir / "lifting_manifest.json"
            source_manifest.write_text(json.dumps(manifest), encoding="utf-8")
            output = root / "derived"
            arguments = [
                "derive_dino_only_lifting_cache.py",
                "--source_manifest",
                str(source_manifest),
                "--output_dir",
                str(output),
                "--ss_context_tokens",
                "32",
            ]
            with mock.patch("sys.argv", arguments):
                derive_main()
            dataset = PoseLiftingCacheDataset(output / "lifting_manifest.json")
            loaded = dataset[0]
            derived_path = output / dataset.rows[0]["cache_file"]
            tampered = torch.load(derived_path, map_location="cpu")
            tampered["slat_condition_provenance"]["condition_tree_sha256"] = "0" * 64
            torch.save(tampered, derived_path)
            with mock.patch("sys.argv", [*arguments, "--resume"]):
                with self.assertRaisesRegex(
                    RuntimeError, "derived SLat condition hash mismatch"
                ):
                    derive_main()
        self.assertEqual(dataset.visual_feature_dim, 1024)
        self.assertEqual(loaded["visual_patch_features"].shape, (2, 25, 1024))
        self.assertEqual(loaded["stock_condition"].shape, (1, 32, 1024))


if __name__ == "__main__":
    unittest.main()
