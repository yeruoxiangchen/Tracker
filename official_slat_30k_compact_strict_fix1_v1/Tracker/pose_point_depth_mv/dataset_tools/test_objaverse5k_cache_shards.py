from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import torch

from ar_ss_flow.pose_lifting import LIFTING_CACHE_VERSION, LIFTING_METADATA_NAMES, schema_hash
from ar_ss_flow.shared_object_preprocessing import canonical_json_sha256
from pose_point_depth_mv.dino_only_condition import (
    DINO_ONLY_LIFTING_VERSION,
    dino_only_feature_metadata,
)
from pose_point_depth_mv.dataset_tools.merge_dino_only_lifting_shards import (
    SOURCE_MARKER_FORMAT,
    main as merge_main,
    sha256_file,
)
from pose_point_depth_mv.dataset_tools.process_objaverse5k_cache_shards import (
    RENDER_MARKER_FORMAT,
    discover_render_shards,
    discover_render_shards_from_roots,
)


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


class CacheShardToolsTest(unittest.TestCase):
    def test_only_completed_render_shards_are_discovered(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            complete = root / "objaverse" / "shard_000"
            active = root / "objaverse" / "shard_001"
            complete.mkdir(parents=True)
            active.mkdir(parents=True)
            manifest = complete / "manifest.json"
            write_json(manifest, {"samples": []})
            marker = complete / "_WORKER_COMPLETE.json"
            write_json(
                marker,
                {
                    "schema": RENDER_MARKER_FORMAT,
                    "source": "objaverse",
                    "shard_index": 0,
                    "render_manifest": str(manifest),
                    "render_manifest_sha256": sha256_file(manifest),
                },
            )
            write_json(active / "manifest.json", {"samples": [{"partial": True}]})
            rows = discover_render_shards(root, None)
            self.assertEqual([row["shard_index"] for row in rows], [0])

    def test_disjoint_render_roots_are_combined_by_shard_index(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            roots = [base / "limited", base / "wide"]
            for root, index in zip(roots, (0, 8)):
                shard = root / "objaverse" / f"shard_{index:03d}"
                shard.mkdir(parents=True)
                manifest = shard / "manifest.json"
                write_json(manifest, {"samples": []})
                write_json(
                    shard / "_WORKER_COMPLETE.json",
                    {
                        "schema": RENDER_MARKER_FORMAT,
                        "source": "objaverse",
                        "shard_index": index,
                        "render_manifest": str(manifest),
                        "render_manifest_sha256": sha256_file(manifest),
                    },
                )
            rows = discover_render_shards_from_roots(roots, None)
            self.assertEqual([row["shard_index"] for row in rows], [0, 8])

    def test_duplicate_shard_across_render_roots_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            roots = [base / "left", base / "right"]
            for root in roots:
                shard = root / "objaverse" / "shard_008"
                shard.mkdir(parents=True)
                manifest = shard / "manifest.json"
                write_json(manifest, {"samples": []})
                write_json(
                    shard / "_WORKER_COMPLETE.json",
                    {
                        "schema": RENDER_MARKER_FORMAT,
                        "source": "objaverse",
                        "shard_index": 8,
                        "render_manifest": str(manifest),
                        "render_manifest_sha256": sha256_file(manifest),
                    },
                )
            with self.assertRaisesRegex(RuntimeError, "duplicate completed render shard"):
                discover_render_shards_from_roots(roots, None)

    def test_merge_references_samples_without_copying(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_dir = root / "work" / "shards" / "shard_000" / "dino_only"
            sample_file = source_dir / "samples" / "sa" / "sample.pt"
            sample_file.parent.mkdir(parents=True)
            torch.save({"uid": "sample"}, sample_file)
            config = {
                "no_vggt": {
                    "version": DINO_ONLY_LIFTING_VERSION,
                    "stock_condition_source": "deterministic_dino_token_context",
                    "slat_condition_source": "per_view_raw_dino_token_context",
                    "depth_policy": "zero_placeholder_not_consumed",
                    "vggt_model_executed": False,
                },
                "geometric_preprocessing": {},
            }
            config_hash = canonical_json_sha256(config)
            manifest = source_dir / "lifting_manifest.json"
            write_json(
                manifest,
                {
                    "format": LIFTING_CACHE_VERSION,
                    "output_dir": str(source_dir),
                    "source_cache_manifest": str(root / "pointpose.json"),
                    "source_cache_manifest_sha256": "pointpose-sha",
                    "sample_count": 1,
                    "object_count": 1,
                    "failure_count": 0,
                    "feature_metadata": dino_only_feature_metadata(patch_count=4),
                    "visual_feature_dim": 1024,
                    "metadata_names": list(LIFTING_METADATA_NAMES),
                    "metadata_schema_hash": schema_hash(),
                    "config": config,
                    "config_hash": config_hash,
                    "samples": [
                        {
                            "uid": "sample",
                            "object_uid": "object",
                            "cache_file": "samples/sa/sample.pt",
                            "cache_file_sha256": sha256_file(sample_file),
                        }
                    ],
                    "passed": True,
                    "training_ready": True,
                },
            )
            write_json(
                source_dir / "_DINO_ONLY_LIFTING_COMPLETE.json",
                {
                    "format": SOURCE_MARKER_FORMAT,
                    "manifest": str(manifest),
                    "manifest_sha256": sha256_file(manifest),
                    "source_cache_manifest_sha256": "pointpose-sha",
                    "config_hash": config_hash,
                    "sample_count": 1,
                    "vggt_model_loaded": False,
                    "vggt_model_executed": False,
                    "passed": True,
                },
            )
            output = root / "merged"
            argv = [
                "merge",
                "--input_root",
                str(root / "work"),
                "--output_dir",
                str(output),
                "--expected_shards",
                "1",
            ]
            with mock.patch.object(sys, "argv", argv):
                merge_main()
            merged = json.loads((output / "lifting_manifest.json").read_text())
            self.assertEqual(merged["sample_count"], 1)
            self.assertEqual(Path(merged["samples"][0]["cache_file"]), sample_file)
            self.assertFalse((output / "samples").exists())


if __name__ == "__main__":
    unittest.main()
