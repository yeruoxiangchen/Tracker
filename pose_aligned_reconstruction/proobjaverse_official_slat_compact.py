#!/usr/bin/env python3
"""Lossless compact runtime cache for official no-VGGT Native-SLat.

The legacy official cache materializes the same fp16 DINO tensor three times:
as lifting features, positive SLat contexts, and a deterministic Stock-SS
context.  It also stores several all-zero tensors required only by a generic
pose-lifting schema.  This module defines a SLat-specific cache which stores
the DINO tensor once and reconstructs every deterministic view at load time.

The target ``lh-slat`` NPZ remains an external, hash-bound artifact.  No
feature quantization or scientific approximation is permitted by this format.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from ar_ss_flow.local_pose_lifting_flow import parse_indices
from ar_ss_flow.shared_object_preprocessing import canonical_json_sha256
from pose_aligned_reconstruction.dino_only_condition import build_dino_only_contexts
from pose_aligned_reconstruction.direct_slat_flow import validate_sparse_target_alignment


COMPACT_SLAT_MANIFEST_FORMAT = (
    "pose_point_depth_mv.proobjaverse_official_slat_compact_manifest.v2"
)
COMPACT_LIFTING_MANIFEST_FORMAT = (
    "pose_point_depth_mv.proobjaverse_official_slat_compact_lifting_manifest.v2"
)
COMPACT_OBJECT_FORMAT = (
    "pose_point_depth_mv.proobjaverse_official_slat_compact_object.v2"
)
COMPACT_LAYOUT_VERSION = "single_fp16_dino_plus_projection_geometry.v2"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def is_compact_manifest_pair(
    slat_manifest: str | Path, lifting_manifest: str | Path
) -> bool:
    slat = _load_json(slat_manifest)
    lifting = _load_json(lifting_manifest)
    formats = (slat.get("format"), lifting.get("format"))
    compact = (
        COMPACT_SLAT_MANIFEST_FORMAT,
        COMPACT_LIFTING_MANIFEST_FORMAT,
    )
    if formats == compact:
        return True
    if formats[0] in compact or formats[1] in compact:
        raise ValueError(
            "compact SLat/lifting manifests must be used as a matched pair: "
            f"formats={formats!r}"
        )
    return False


def validate_compact_manifest_pair_payloads(
    slat: dict[str, Any], lifting: dict[str, Any]
) -> dict[str, Any]:
    """Strictly bind the two compact manifests without loading sample payloads.

    The compact format deliberately keeps one physical object payload and
    exposes two logical manifests.  Joining merely on ``uid`` is insufficient:
    both manifests must bind that UID to the same object and the same immutable
    compact file.  This validator is intentionally exhaustive over the shared
    row identity and does not weaken either manifest's own config validation.
    """

    expected_formats = (
        COMPACT_SLAT_MANIFEST_FORMAT,
        COMPACT_LIFTING_MANIFEST_FORMAT,
    )
    formats = (slat.get("format"), lifting.get("format"))
    if formats != expected_formats:
        raise ValueError(
            "compact SLat/lifting manifests must be used as a matched pair: "
            f"formats={formats!r}"
        )
    if (
        slat.get("layout") != COMPACT_LAYOUT_VERSION
        or lifting.get("layout") != COMPACT_LAYOUT_VERSION
    ):
        raise ValueError("compact SLat/lifting layout differs from runtime")

    slat_rows = slat.get("samples")
    lifting_rows = lifting.get("samples")
    if not isinstance(slat_rows, list) or not slat_rows:
        raise ValueError("compact SLat manifest contains no samples")
    if not isinstance(lifting_rows, list) or not lifting_rows:
        raise ValueError("compact lifting manifest contains no samples")

    def index_rows(
        label: str, rows: list[dict[str, Any]], manifest: dict[str, Any]
    ) -> dict[str, dict[str, Any]]:
        indexed: dict[str, dict[str, Any]] = {}
        object_uids: set[str] = set()
        for raw_row in rows:
            if not isinstance(raw_row, dict):
                raise ValueError(f"compact {label} manifest row is not an object")
            uid = str(raw_row.get("uid", ""))
            object_uid = str(raw_row.get("object_uid", ""))
            if not uid or not object_uid or uid in indexed:
                raise ValueError(
                    f"compact {label} manifest has empty/duplicate UID or object UID"
                )
            indexed[uid] = raw_row
            object_uids.add(object_uid)
        declared_samples = int(manifest.get("sample_count", -1))
        declared_objects = int(manifest.get("object_count", -1))
        if declared_samples != len(rows):
            raise ValueError(
                f"compact {label} sample_count differs: "
                f"declared={declared_samples} actual={len(rows)}"
            )
        if declared_objects != len(object_uids):
            raise ValueError(
                f"compact {label} object_count differs: "
                f"declared={declared_objects} actual={len(object_uids)}"
            )
        return indexed

    slat_by_uid = index_rows("SLat", slat_rows, slat)
    lifting_by_uid = index_rows("lifting", lifting_rows, lifting)
    if set(slat_by_uid) != set(lifting_by_uid):
        missing = sorted(set(slat_by_uid) - set(lifting_by_uid))
        extra = sorted(set(lifting_by_uid) - set(slat_by_uid))
        raise ValueError(
            "compact UID join is not exact: "
            f"missing={missing[:8]} extra={extra[:8]}"
        )

    shared_fields = ("uid", "object_uid", "compact_file", "compact_file_sha256")
    bindings: list[dict[str, str]] = []
    for raw_slat_row in slat_rows:
        uid = str(raw_slat_row["uid"])
        raw_lifting_row = lifting_by_uid[uid]
        left = {key: str(raw_slat_row.get(key, "")) for key in shared_fields}
        right = {key: str(raw_lifting_row.get(key, "")) for key in shared_fields}
        if any(not value for value in left.values()) or left != right:
            raise ValueError(
                f"compact pair identity differs for uid={uid}: "
                f"slat={left!r} lifting={right!r}"
            )
        bindings.append(left)

    pair_binding = {
        "layout": COMPACT_LAYOUT_VERSION,
        "slat_config_hash": str(slat.get("config_hash", "")),
        "lifting_config_hash": str(lifting.get("config_hash", "")),
        "slat_normalization_hash": str(slat.get("slat_normalization_hash", "")),
        "sample_count": len(bindings),
        "samples": bindings,
    }
    if any(
        not pair_binding[key]
        for key in (
            "slat_config_hash",
            "lifting_config_hash",
            "slat_normalization_hash",
        )
    ):
        raise ValueError("compact pair identity has an empty config/normalization hash")
    pair_identity = canonical_json_sha256(pair_binding)
    declared = (slat.get("pair_identity"), lifting.get("pair_identity"))
    if (declared[0] is None) != (declared[1] is None):
        raise ValueError("compact pair_identity must be declared by both manifests")
    if declared[0] is not None and declared != (pair_identity, pair_identity):
        raise ValueError(
            "compact pair_identity differs from the exact manifest binding: "
            f"declared={declared!r} computed={pair_identity}"
        )
    return {
        "pair_identity": pair_identity,
        "sample_count": len(bindings),
        "slat_by_uid": slat_by_uid,
        "lifting_by_uid": lifting_by_uid,
    }


def load_compact_object(
    path: str | Path,
    *,
    uid: str,
    object_uid: str,
    slat_config_hash: str,
    lifting_config_hash: str,
) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    payload = torch.load(resolved, map_location="cpu", weights_only=False)
    expected_scalars = {
        "format": COMPACT_OBJECT_FORMAT,
        "layout": COMPACT_LAYOUT_VERSION,
        "uid": str(uid),
        "object_uid": str(object_uid),
        "slat_config_hash": str(slat_config_hash),
        "lifting_config_hash": str(lifting_config_hash),
        "official_gt_support_only": True,
    }
    actual_scalars = {key: payload.get(key) for key in expected_scalars}
    if actual_scalars != expected_scalars:
        raise ValueError(
            f"uid={uid} compact object identity differs: "
            f"expected={expected_scalars!r} actual={actual_scalars!r}"
        )
    visual = payload.get("visual_patch_features")
    intrinsics = payload.get("intrinsics")
    extrinsics = payload.get("extrinsics")
    view_ids = payload.get("view_ids")
    image_size = payload.get("image_size")
    if not torch.is_tensor(visual) or visual.ndim != 3:
        raise ValueError(f"uid={uid} compact DINO tensor must be [views,patches,1024]")
    views, patches, channels = map(int, visual.shape)
    if views <= 0 or patches <= 0 or channels != 1024 or visual.dtype != torch.float16:
        raise ValueError(
            f"uid={uid} compact DINO schema differs: shape={tuple(visual.shape)} "
            f"dtype={visual.dtype}"
        )
    if not torch.is_tensor(intrinsics) or intrinsics.shape != (views, 3, 3):
        raise ValueError(f"uid={uid} compact intrinsics schema differs")
    if not torch.is_tensor(extrinsics) or extrinsics.shape != (views, 4, 4):
        raise ValueError(f"uid={uid} compact extrinsics schema differs")
    if not torch.is_tensor(view_ids) or view_ids.shape != (views,):
        raise ValueError(f"uid={uid} compact view-id schema differs")
    if (
        not isinstance(image_size, (list, tuple))
        or len(image_size) != 2
        or any(int(value) <= 0 for value in image_size)
    ):
        raise ValueError(f"uid={uid} compact image_size is invalid")
    finite = (visual, intrinsics, extrinsics)
    if not all(bool(torch.isfinite(value.float()).all().item()) for value in finite):
        raise ValueError(f"uid={uid} compact object contains non-finite tensors")
    return payload


class CompactNativeConditionBackend(Dataset):
    """One-load backend exposing the legacy NativeCondition sample contract."""

    def __init__(
        self,
        slat_manifest: str | Path,
        lifting_manifest: str | Path,
        *,
        indices: str = "all",
        verify_hashes: bool = False,
    ) -> None:
        self.slat_manifest_path = Path(slat_manifest).expanduser().resolve()
        self.lifting_manifest_path = Path(lifting_manifest).expanduser().resolve()
        slat = _load_json(self.slat_manifest_path)
        lifting = _load_json(self.lifting_manifest_path)
        pair = validate_compact_manifest_pair_payloads(slat, lifting)
        self.pair_identity = str(pair["pair_identity"])
        all_slat_rows = slat.get("samples")
        all_lifting_rows = lifting.get("samples")
        assert isinstance(all_slat_rows, list)
        assert isinstance(all_lifting_rows, list)
        lifting_by_uid = pair["lifting_by_uid"]
        selected = parse_indices(indices, len(all_slat_rows))
        self.rows = [dict(all_slat_rows[index]) for index in selected]
        self.lifting_rows = [lifting_by_uid[str(row["uid"])] for row in self.rows]
        self.slat_root = Path(
            slat.get("output_dir", self.slat_manifest_path.parent)
        ).resolve()
        self.lifting_root = Path(
            lifting.get("output_dir", self.lifting_manifest_path.parent)
        ).resolve()
        self.config = dict(slat.get("config", {}))
        self.config_hash = str(slat.get("config_hash", ""))
        self.slat_normalization = dict(slat.get("slat_normalization", {}))
        self.slat_normalization_hash = str(slat.get("slat_normalization_hash", ""))
        self.lifting_config = dict(lifting.get("config", {}))
        self.lifting_config_hash = str(lifting.get("config_hash", ""))
        if canonical_json_sha256(self.config) != self.config_hash:
            raise ValueError("compact SLat config hash mismatch")
        if canonical_json_sha256(self.lifting_config) != self.lifting_config_hash:
            raise ValueError("compact lifting config hash mismatch")
        normalized = {
            key: [float(item) for item in value]
            for key, value in self.slat_normalization.items()
        }
        if canonical_json_sha256(normalized) != self.slat_normalization_hash:
            raise ValueError("compact SLat normalization hash mismatch")
        if sorted(normalized) != ["mean", "std"] or any(
            len(normalized[key]) != 8 for key in ("mean", "std")
        ):
            raise ValueError("compact SLat normalization schema differs")
        self.visual_feature_dim = int(lifting.get("visual_feature_dim", 0))
        self.feature_metadata = dict(lifting.get("feature_metadata", {}))
        self.source_cache_manifest = str(lifting.get("source_cache_manifest", ""))
        self.lifting_manifest_payload = lifting
        # Existing callers inspect dataset.slat and dataset.lifting metadata.
        self.slat_view = SimpleNamespace(
            rows=self.rows,
            config=self.config,
            config_hash=self.config_hash,
            slat_normalization=self.slat_normalization,
            slat_normalization_hash=self.slat_normalization_hash,
            compact_pair_identity=self.pair_identity,
        )
        self.lifting_view = SimpleNamespace(
            rows=self.lifting_rows,
            config=self.lifting_config,
            config_hash=self.lifting_config_hash,
            visual_feature_dim=self.visual_feature_dim,
            feature_metadata=self.feature_metadata,
            source_cache_manifest=self.source_cache_manifest,
            manifest_payload=self.lifting_manifest_payload,
            compact_pair_identity=self.pair_identity,
        )
        if verify_hashes:
            checked: set[tuple[str, str]] = set()
            for slat_row, lifting_row in zip(self.rows, self.lifting_rows):
                identities = (
                    (
                        _resolve(self.slat_root, slat_row["target_file"]),
                        str(slat_row["target_file_sha256"]),
                    ),
                    (
                        _resolve(self.lifting_root, lifting_row["compact_file"]),
                        str(lifting_row["compact_file_sha256"]),
                    ),
                )
                for path, expected in identities:
                    identity = (str(path), expected)
                    if identity in checked:
                        continue
                    actual = sha256_file(path)
                    if actual != expected:
                        raise RuntimeError(
                            f"compact cache artifact mutation: {path} {actual} != {expected}"
                        )
                    checked.add(identity)

    def __len__(self) -> int:
        return len(self.rows)

    def limit_objects(self, max_objects: int) -> None:
        if int(max_objects) <= 0:
            return
        allowed = set(
            sorted({str(row["object_uid"]) for row in self.rows})[: int(max_objects)]
        )
        selected = [
            index
            for index, row in enumerate(self.rows)
            if str(row["object_uid"]) in allowed
        ]
        self.rows[:] = [self.rows[index] for index in selected]
        self.lifting_rows[:] = [self.lifting_rows[index] for index in selected]

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        lifting_row = self.lifting_rows[index]
        uid = str(row["uid"])
        object_uid = str(row["object_uid"])
        if uid != str(lifting_row["uid"]):
            raise RuntimeError("compact runtime UID join changed")
        target_path = _resolve(self.slat_root, row["target_file"])
        with np.load(target_path, allow_pickle=False) as target:
            coords3 = np.asarray(target["coords"], dtype=np.int32)
            target_feats = np.asarray(target["feats"], dtype=np.float32)
        target_coords = torch.cat(
            (
                torch.zeros((len(coords3), 1), dtype=torch.int32),
                torch.from_numpy(coords3),
            ),
            dim=1,
        )
        target_feats_tensor = torch.from_numpy(target_feats)
        validate_sparse_target_alignment(
            target_coords, target_feats_tensor, require_single_batch=True
        )
        compact_path = _resolve(self.lifting_root, lifting_row["compact_file"])
        compact = load_compact_object(
            compact_path,
            uid=uid,
            object_uid=object_uid,
            slat_config_hash=self.config_hash,
            lifting_config_hash=self.lifting_config_hash,
        )
        context_tokens = int(compact["context_contract"]["ss_context_token_cap"])
        contexts = build_dino_only_contexts(
            compact["visual_patch_features"], ss_context_tokens=context_tokens
        )
        if contexts["context_contract"] != compact["context_contract"]:
            raise RuntimeError(f"uid={uid} deterministic context contract changed")
        lifting_sample = {
            "format": COMPACT_OBJECT_FORMAT,
            "uid": uid,
            "object_uid": object_uid,
            "cache_path": str(compact_path),
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
            "official_gt_support_only": True,
            "compact_projection_only": True,
        }
        return {
            **row,
            "target_coords": target_coords,
            "target_feats": target_feats_tensor,
            # These legacy fields are deterministic placeholders.  They remain
            # in RAM compatibility only and are never serialized by compact v2.
            "corrected_ss": torch.zeros((1, 8, 16, 16, 16), dtype=torch.float32),
            "occupancy_logits64": torch.zeros(
                (1, 1, 64, 64, 64), dtype=torch.float32
            ),
            "corrected_coords64": torch.from_numpy(coords3),
            "physical_tokens16": torch.empty((0,), dtype=torch.float32),
            "condition": contexts["slat_condition"],
            "lifting_sample": lifting_sample,
        }


__all__ = [
    "COMPACT_LAYOUT_VERSION",
    "COMPACT_LIFTING_MANIFEST_FORMAT",
    "COMPACT_OBJECT_FORMAT",
    "COMPACT_SLAT_MANIFEST_FORMAT",
    "CompactNativeConditionBackend",
    "is_compact_manifest_pair",
    "load_compact_object",
    "validate_compact_manifest_pair_payloads",
]
