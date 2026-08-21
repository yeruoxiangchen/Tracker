from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from pose_point_depth_mv.direct_slat_data import DirectSLatCacheDataset
from pose_point_depth_mv.direct_slat_flow import (
    DIRECT_SLAT_CACHE_VERSION,
    canonical_json_sha256,
)


class DirectSLatCacheFailClosedTests(unittest.TestCase):
    def test_target_only_manifest_cannot_train(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(
                json.dumps(
                    {
                        "format": DIRECT_SLAT_CACHE_VERSION,
                        "materialized": False,
                        "samples": [],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "targets only"):
                DirectSLatCacheDataset(path)

    def test_object_sequences_must_share_one_target(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            normalization = {"mean": [0.0] * 8, "std": [1.0] * 8}
            common = {
                "object_uid": "object",
                "support_seed": 42,
                "target_file_sha256": "a",
                "support_file": "support.pt",
                "support_file_sha256": "b",
                "physical_file": "physical.pt",
                "physical_file_sha256": "c",
                "condition_file": "condition.pt",
                "condition_file_sha256": "d",
            }
            rows = [
                {**common, "uid": "seq0", "target_file": "target0.npz"},
                {**common, "uid": "seq1", "target_file": "target1.npz"},
            ]
            manifest = {
                "format": DIRECT_SLAT_CACHE_VERSION,
                "materialized": True,
                "output_dir": str(root),
                "config": {},
                "config_hash": canonical_json_sha256({}),
                "slat_normalization": normalization,
                "slat_normalization_hash": canonical_json_sha256(normalization),
                "samples": rows,
            }
            path = root / "manifest.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "multiple target"):
                DirectSLatCacheDataset(path)


if __name__ == "__main__":
    unittest.main()
