#!/usr/bin/env python3
"""Compact-v2 dataset adapter for official no-VGGT Native-SS.

The adapter binds an audited official SS target to the immutable compact-v2
DINO/K/T payload.  Legacy pose-lifting manifests remain on their original
loader path; this module is selected only by its explicit derived format.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from ar_ss_flow.local_pose_lifting_flow import parse_indices
from ar_ss_flow.shared_object_preprocessing import canonical_json_sha256
from pose_point_depth_mv.dino_only_condition import build_dino_only_contexts
from pose_point_depth_mv.proobjaverse_official_slat_compact import (
    COMPACT_LIFTING_MANIFEST_FORMAT,
    COMPACT_OBJECT_FORMAT,
    COMPACT_SLAT_MANIFEST_FORMAT,
    load_compact_object,
    sha256_file,
    validate_compact_manifest_pair_payloads,
)


OFFICIAL_SS_COMPACT_MANIFEST_FORMAT = (
    "pose_point_depth_mv.proobjaverse_official_ss_compact_training_cache.v1"
)


def _load_json(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    value = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {resolved}")
    return value


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def is_official_ss_compact_manifest(path: str | Path) -> bool:
    return _load_json(path).get("format") == OFFICIAL_SS_COMPACT_MANIFEST_FORMAT


def build_official_ss_compact_manifest(
    *,
    source_slat: dict[str, Any],
    source_lifting: dict[str, Any],
    source_slat_manifest: str | Path,
    source_lifting_manifest: str | Path,
    output_dir: str | Path,
    rows: list[dict[str, Any]],
    official_ss_targets: dict[str, Any],
) -> dict[str, Any]:
    """Build a derived manifest without mutating either compact source."""

    slat_path = Path(source_slat_manifest).expanduser().resolve()
    lifting_path = Path(source_lifting_manifest).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    pair = validate_compact_manifest_pair_payloads(source_slat, source_lifting)
    if source_slat.get("format") != COMPACT_SLAT_MANIFEST_FORMAT:
        raise ValueError("source SLat manifest is not compact-v2")
    if source_lifting.get("format") != COMPACT_LIFTING_MANIFEST_FORMAT:
        raise ValueError("source lifting manifest is not compact-v2")
    if not rows:
        raise ValueError("official SS compact manifest requires at least one row")

    source_lifting_by_uid = pair["lifting_by_uid"]
    source_slat_by_uid = pair["slat_by_uid"]
    normalized_rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    object_uids: set[str] = set()
    for raw in rows:
        uid = str(raw.get("uid", ""))
        object_uid = str(raw.get("object_uid", ""))
        if not uid or not object_uid or uid in seen:
            raise ValueError("official SS compact rows contain empty/duplicate identity")
        if uid not in source_lifting_by_uid or uid not in source_slat_by_uid:
            raise ValueError(f"uid={uid} is absent from the frozen compact pair")
        source_row = source_lifting_by_uid[uid]
        expected_binding = {
            key: str(source_row.get(key, ""))
            for key in ("uid", "object_uid", "compact_file", "compact_file_sha256")
        }
        observed_binding = {
            key: str(raw.get(key, ""))
            for key in ("uid", "object_uid", "compact_file", "compact_file_sha256")
        }
        if expected_binding != observed_binding or any(
            not value for value in observed_binding.values()
        ):
            raise ValueError(
                f"uid={uid} derived compact binding differs from source lifting row"
            )
        if not str(raw.get("ss_latent", "")) or not str(
            raw.get("ss_latent_sha256", "")
        ):
            raise ValueError(f"uid={uid} derived row lacks official SS target binding")
        normalized_rows.append(dict(raw))
        seen.add(uid)
        object_uids.add(object_uid)

    config = dict(source_lifting.get("config", {}))
    config["official_ss_targets"] = dict(official_ss_targets)
    config_hash = canonical_json_sha256(config)
    result = dict(source_lifting)
    result.update(
        {
            "format": OFFICIAL_SS_COMPACT_MANIFEST_FORMAT,
            "output_dir": str(output),
            "sample_count": len(normalized_rows),
            "object_count": len(object_uids),
            "samples": normalized_rows,
            "config": config,
            "config_hash": config_hash,
            "official_gt_support_only": False,
            "source_compact_slat_manifest": str(slat_path),
            "source_compact_slat_manifest_sha256": sha256_file(slat_path),
            "source_compact_lifting_manifest": str(lifting_path),
            "source_compact_lifting_manifest_sha256": sha256_file(lifting_path),
            "source_compact_slat_config_hash": str(source_slat.get("config_hash", "")),
            "source_compact_lifting_config_hash": str(
                source_lifting.get("config_hash", "")
            ),
            "source_compact_pair_identity": str(pair["pair_identity"]),
            "storage_semantics": {
                "scientific_target": "official_ss_target.v1",
                "condition_payload": "immutable compact-v2 DINO/K/T",
                "deterministic_stock_context_rebuilt": True,
                "vggt_executed": False,
                "source_compact_manifests_mutated": False,
            },
        }
    )
    result["manifest_identity"] = canonical_json_sha256(
        {
            "format": result["format"],
            "config_hash": config_hash,
            "source_compact_pair_identity": result["source_compact_pair_identity"],
            "source_compact_slat_manifest_sha256": result[
                "source_compact_slat_manifest_sha256"
            ],
            "source_compact_lifting_manifest_sha256": result[
                "source_compact_lifting_manifest_sha256"
            ],
            "samples": [
                {
                    "uid": str(row["uid"]),
                    "object_uid": str(row["object_uid"]),
                    "compact_file": str(row["compact_file"]),
                    "compact_file_sha256": str(row["compact_file_sha256"]),
                    "ss_latent": str(row["ss_latent"]),
                    "ss_latent_sha256": str(row["ss_latent_sha256"]),
                }
                for row in normalized_rows
            ],
        }
    )
    return result


class CompactOfficialSSDataset(Dataset):
    """PoseLifting-compatible view over compact-v2 plus official SS targets."""

    def __init__(
        self,
        manifest: str | Path,
        *,
        indices: str = "all",
        verify_hashes: bool = False,
    ) -> None:
        self.manifest_path = Path(manifest).expanduser().resolve()
        payload = _load_json(self.manifest_path)
        if payload.get("format") != OFFICIAL_SS_COMPACT_MANIFEST_FORMAT:
            raise ValueError(
                f"unexpected compact official SS manifest format={payload.get('format')!r}"
            )
        self.manifest_payload = payload
        self.root = Path(payload.get("output_dir", self.manifest_path.parent)).resolve()
        self.config = dict(payload.get("config", {}))
        self.config_hash = str(payload.get("config_hash", ""))
        if canonical_json_sha256(self.config) != self.config_hash:
            raise ValueError("compact official SS config hash mismatch")

        self.source_slat_manifest_path = Path(
            str(payload.get("source_compact_slat_manifest", ""))
        ).expanduser().resolve()
        self.source_lifting_manifest_path = Path(
            str(payload.get("source_compact_lifting_manifest", ""))
        ).expanduser().resolve()
        for label, path, expected in (
            (
                "SLat",
                self.source_slat_manifest_path,
                str(payload.get("source_compact_slat_manifest_sha256", "")),
            ),
            (
                "lifting",
                self.source_lifting_manifest_path,
                str(payload.get("source_compact_lifting_manifest_sha256", "")),
            ),
        ):
            if not path.is_file() or not expected or sha256_file(path) != expected:
                raise RuntimeError(f"source compact {label} manifest changed: {path}")

        source_slat = _load_json(self.source_slat_manifest_path)
        source_lifting = _load_json(self.source_lifting_manifest_path)
        pair = validate_compact_manifest_pair_payloads(source_slat, source_lifting)
        if str(payload.get("source_compact_pair_identity", "")) != str(
            pair["pair_identity"]
        ):
            raise RuntimeError("derived official SS compact pair identity changed")
        if str(payload.get("source_compact_slat_config_hash", "")) != str(
            source_slat.get("config_hash", "")
        ):
            raise RuntimeError("derived official SS SLat config binding changed")
        if str(payload.get("source_compact_lifting_config_hash", "")) != str(
            source_lifting.get("config_hash", "")
        ):
            raise RuntimeError("derived official SS lifting config binding changed")

        all_rows = payload.get("samples")
        if not isinstance(all_rows, list) or not all_rows:
            raise ValueError("compact official SS manifest contains no samples")
        if int(payload.get("sample_count", -1)) != len(all_rows):
            raise ValueError("compact official SS sample_count differs")
        selected = parse_indices(indices, len(all_rows))
        self.rows = [dict(all_rows[index]) for index in selected]
        self._source_lifting_by_uid = pair["lifting_by_uid"]
        self._source_slat_by_uid = pair["slat_by_uid"]
        self._source_lifting_root = Path(
            source_lifting.get("output_dir", self.source_lifting_manifest_path.parent)
        ).resolve()
        self._source_slat_config_hash = str(source_slat.get("config_hash", ""))
        self._source_lifting_config_hash = str(source_lifting.get("config_hash", ""))
        self.visual_feature_dim = int(payload.get("visual_feature_dim", 0))
        self.feature_metadata = dict(payload.get("feature_metadata", {}))
        self.source_cache_manifest = str(payload.get("source_cache_manifest", ""))

        if self.visual_feature_dim != int(source_lifting.get("visual_feature_dim", 0)):
            raise RuntimeError("derived official SS visual feature dimension changed")
        if self.feature_metadata != dict(source_lifting.get("feature_metadata", {})):
            raise RuntimeError("derived official SS feature metadata changed")

        seen: set[str] = set()
        for row in self.rows:
            uid = str(row.get("uid", ""))
            if not uid or uid in seen or uid not in self._source_lifting_by_uid:
                raise RuntimeError("compact official SS row UID binding is invalid")
            seen.add(uid)
            source_row = self._source_lifting_by_uid[uid]
            for key in ("object_uid", "compact_file", "compact_file_sha256"):
                if str(row.get(key, "")) != str(source_row.get(key, "")):
                    raise RuntimeError(
                        f"uid={uid} compact official SS source binding differs: {key}"
                    )

        if verify_hashes:
            checked: set[tuple[str, str]] = set()
            for row in self.rows:
                source_row = self._source_lifting_by_uid[str(row["uid"])]
                compact_path = _resolve(
                    self._source_lifting_root, source_row["compact_file"]
                )
                target_path = _resolve(self.root, row["ss_latent"])
                for path, expected in (
                    (compact_path, str(source_row["compact_file_sha256"])),
                    (target_path, str(row["ss_latent_sha256"])),
                ):
                    identity = (str(path), expected)
                    if identity in checked:
                        continue
                    if not path.is_file() or sha256_file(path) != expected:
                        raise RuntimeError(f"compact official SS artifact changed: {path}")
                    checked.add(identity)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        uid = str(row["uid"])
        object_uid = str(row["object_uid"])
        source_row = self._source_lifting_by_uid[uid]
        compact_path = _resolve(self._source_lifting_root, source_row["compact_file"])
        compact = load_compact_object(
            compact_path,
            uid=uid,
            object_uid=object_uid,
            slat_config_hash=self._source_slat_config_hash,
            lifting_config_hash=self._source_lifting_config_hash,
        )
        context_tokens = int(compact["context_contract"]["ss_context_token_cap"])
        contexts = build_dino_only_contexts(
            compact["visual_patch_features"], ss_context_tokens=context_tokens
        )
        if contexts["context_contract"] != compact["context_contract"]:
            raise RuntimeError(f"uid={uid} deterministic context contract changed")

        target_path = _resolve(self.root, row["ss_latent"])
        if not target_path.is_file():
            raise FileNotFoundError(f"uid={uid} missing official SS target: {target_path}")
        with np.load(target_path, allow_pickle=False) as target_payload:
            target = np.asarray(target_payload["z"], dtype=np.float32)
            target_coords = np.asarray(target_payload["target_coords"], dtype=np.int64)
            target_uid = str(np.asarray(target_payload["uid"]).item())
        if target_uid != uid:
            raise RuntimeError(f"uid={uid} official SS target identity changed")
        if target.ndim == 5 and target.shape[0] == 1:
            target = target[0]
        if target.shape != (8, 16, 16, 16):
            raise ValueError(f"uid={uid} invalid official SS latent shape={target.shape}")
        if target_coords.ndim != 2 or target_coords.shape[1] != 3:
            raise ValueError(f"uid={uid} invalid official SS target coords")
        if not np.isfinite(target).all():
            raise ValueError(f"uid={uid} official SS target contains non-finite values")

        return {
            **row,
            "cache_path": str(compact_path),
            "target_path": str(target_path),
            "target": torch.from_numpy(target),
            # Match PoseLiftingCacheDataset's public batch contract.  The
            # serialized official target remains int32; callers observe int64.
            "target_coords": torch.from_numpy(target_coords),
            "visual_patch_features": compact["visual_patch_features"],
            "stock_condition": contexts["stock_condition"],
            "intrinsics": compact["intrinsics"],
            "extrinsics": compact["extrinsics"],
            "image_size": tuple(int(value) for value in compact["image_size"]),
            "grid_transform": str(compact["grid_transform"]),
            "extrinsics_type": str(compact["extrinsics_type"]),
            "camera_forward_sign": float(compact["camera_forward_sign"]),
            "view_ids": compact["view_ids"],
            "preprocessing": dict(compact["preprocessing"]),
            "official_gt_support_only": False,
            "compact_projection_only": True,
            "format": COMPACT_OBJECT_FORMAT,
        }


__all__ = [
    "OFFICIAL_SS_COMPACT_MANIFEST_FORMAT",
    "CompactOfficialSSDataset",
    "build_official_ss_compact_manifest",
    "is_official_ss_compact_manifest",
]
