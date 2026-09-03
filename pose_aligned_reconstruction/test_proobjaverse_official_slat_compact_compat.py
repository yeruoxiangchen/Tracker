from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from ar_ss_flow.pose_lifting import (
    LIFTING_CACHE_VERSION,
    LIFTING_METADATA_NAMES,
    schema_hash,
)
from ar_ss_flow.shared_object_preprocessing import canonical_json_sha256
from pose_aligned_reconstruction.dino_only_condition import (
    DINO_ONLY_LIFTING_VERSION,
    dino_only_feature_metadata,
)
from pose_aligned_reconstruction.proobjaverse_official_slat_compact import (
    COMPACT_LAYOUT_VERSION,
    COMPACT_LIFTING_MANIFEST_FORMAT,
    COMPACT_SLAT_MANIFEST_FORMAT,
    CompactNativeConditionBackend,
)
from pose_aligned_reconstruction.train_native_slat_genrecon_no_vggt import (
    validate_no_vggt_lifting_preflight,
)


def _lifting_config() -> dict:
    return {
        "no_vggt": {
            "version": DINO_ONLY_LIFTING_VERSION,
            "stock_condition_source": "deterministic_dino_token_context",
            "slat_condition_source": "per_view_raw_dino_token_context",
            "depth_policy": "zero_placeholder_not_consumed",
        }
    }


def _normalization() -> dict:
    return {"mean": [0.0] * 8, "std": [1.0] * 8}


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _legacy_lifting_manifest(root: Path) -> Path:
    config = _lifting_config()
    return _write_json(
        root / "legacy_lifting.json",
        {
            "format": LIFTING_CACHE_VERSION,
            "output_dir": str(root),
            "samples": [{"uid": "uid0", "cache_file": "unused.pt"}],
            "sample_count": 1,
            "object_count": 1,
            "visual_feature_dim": 1024,
            "feature_metadata": dino_only_feature_metadata(patch_count=4),
            "metadata_names": list(LIFTING_METADATA_NAMES),
            "metadata_schema_hash": schema_hash(),
            "config": config,
            "config_hash": canonical_json_sha256(config),
        },
    )


def _compact_manifests(root: Path) -> tuple[Path, Path]:
    slat_config = {"condition_arch": "native_ss_genrecon_v2"}
    lifting_config = _lifting_config()
    normalization = _normalization()
    common = {
        "uid": "uid0",
        "object_uid": "object0",
        "compact_file": "objects/uid0.pt",
        "compact_file_sha256": "a" * 64,
    }
    slat = {
        "format": COMPACT_SLAT_MANIFEST_FORMAT,
        "layout": COMPACT_LAYOUT_VERSION,
        "materialized": True,
        "output_dir": str(root),
        "config": slat_config,
        "config_hash": canonical_json_sha256(slat_config),
        "slat_normalization": normalization,
        "slat_normalization_hash": canonical_json_sha256(normalization),
        "samples": [
            {
                **common,
                "support_seed": 42,
                "target_file": "target.npz",
                "target_file_sha256": "b" * 64,
            }
        ],
        "sample_count": 1,
        "object_count": 1,
        "official_gt_support_only": True,
    }
    lifting = {
        "format": COMPACT_LIFTING_MANIFEST_FORMAT,
        "layout": COMPACT_LAYOUT_VERSION,
        "output_dir": str(root),
        "samples": [common],
        "sample_count": 1,
        "object_count": 1,
        "feature_metadata": dino_only_feature_metadata(patch_count=4),
        "visual_feature_dim": 1024,
        "metadata_names": list(LIFTING_METADATA_NAMES),
        "metadata_schema_hash": schema_hash(),
        "config": lifting_config,
        "config_hash": canonical_json_sha256(lifting_config),
    }
    return (
        _write_json(root / "compact_slat.json", slat),
        _write_json(root / "compact_lifting.json", lifting),
    )


class OfficialSLatCompactCompatibilityTests(unittest.TestCase):
    def test_legacy_lifting_preflight_keeps_legacy_loader(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = _legacy_lifting_manifest(Path(directory))
            contract = validate_no_vggt_lifting_preflight(None, str(manifest))
            self.assertEqual(contract["version"], DINO_ONLY_LIFTING_VERSION)
            self.assertEqual(contract["patch_side"], 2)

    def test_compact_lifting_preflight_uses_matched_lifting_view(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            slat, lifting = _compact_manifests(Path(directory))
            contract = validate_no_vggt_lifting_preflight(str(slat), str(lifting))
            backend = CompactNativeConditionBackend(slat, lifting)
            self.assertEqual(contract["version"], DINO_ONLY_LIFTING_VERSION)
            self.assertEqual(contract["patch_side"], 2)
            self.assertEqual(
                backend.slat_view.compact_pair_identity,
                backend.lifting_view.compact_pair_identity,
            )

    def test_compact_slat_lifting_pair_mismatch_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            slat, lifting = _compact_manifests(root)
            payload = json.loads(lifting.read_text(encoding="utf-8"))
            payload["samples"][0]["compact_file_sha256"] = "c" * 64
            _write_json(lifting, payload)
            with self.assertRaisesRegex(ValueError, "pair identity differs"):
                CompactNativeConditionBackend(slat, lifting)
            with self.assertRaisesRegex(ValueError, "pair identity differs"):
                validate_no_vggt_lifting_preflight(str(slat), str(lifting))

    def test_one_sided_compact_pair_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            compact_slat, _ = _compact_manifests(root)
            legacy_lifting = _legacy_lifting_manifest(root)
            with self.assertRaisesRegex(ValueError, "matched pair"):
                CompactNativeConditionBackend(compact_slat, legacy_lifting)


if __name__ == "__main__":
    unittest.main()
