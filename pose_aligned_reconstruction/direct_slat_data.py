#!/usr/bin/env python3
"""Dataset and integrity helpers for materialized direct-SLAT caches."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from ar_ss_flow.local_pose_lifting_flow import parse_indices
from pose_aligned_reconstruction.direct_slat_flow import (
    DIRECT_SLAT_CACHE_VERSION,
    canonical_json_sha256,
    validate_sparse_target_alignment,
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class DirectSLatCacheDataset(Dataset):
    """Per-sequence/per-support-seed native-condition SLAT training rows."""

    def __init__(
        self,
        manifest: str | Path,
        *,
        indices: str = "all",
        verify_hashes: bool = False,
    ) -> None:
        self.manifest_path = Path(manifest).resolve()
        payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if payload.get("format") != DIRECT_SLAT_CACHE_VERSION:
            raise ValueError(f"unsupported direct SLAT cache={payload.get('format')!r}")
        if payload.get("materialized") is not True:
            raise ValueError(
                "direct SLAT cache contains targets only; run full GPU materialization"
            )
        all_rows = payload.get("samples")
        if not isinstance(all_rows, list) or not all_rows:
            raise ValueError("direct SLAT cache contains no materialized samples")
        selected = parse_indices(indices, len(all_rows))
        self.rows = [all_rows[index] for index in selected]
        self.root = Path(payload.get("output_dir", self.manifest_path.parent)).resolve()
        self.config = dict(payload.get("config", {}))
        self.config_hash = str(payload.get("config_hash", ""))
        self.slat_normalization = dict(payload.get("slat_normalization", {}))
        self.slat_normalization_hash = str(payload.get("slat_normalization_hash", ""))
        if not self.config_hash or not self.slat_normalization_hash:
            raise ValueError("direct SLAT cache lacks config/normalization binding")
        if canonical_json_sha256(self.config) != self.config_hash:
            raise ValueError("direct SLAT cache config hash mismatch")
        expected_normalization_hash = canonical_json_sha256(
            {
                key: [float(item) for item in value]
                for key, value in self.slat_normalization.items()
            }
        )
        if expected_normalization_hash != self.slat_normalization_hash:
            raise ValueError("direct SLAT cache normalization hash mismatch")
        if sorted(self.slat_normalization) != ["mean", "std"]:
            raise ValueError("direct SLAT normalization must contain mean/std")
        if any(len(self.slat_normalization[key]) != 8 for key in ("mean", "std")):
            raise ValueError("direct SLAT normalization must have eight channels")
        uids = [f"{row.get('uid', '')}@{row.get('support_seed', '')}" for row in self.rows]
        if not all(row.get("uid") and row.get("object_uid") for row in self.rows):
            raise ValueError("direct SLAT subset contains empty identities")
        if len(uids) != len(set(uids)):
            raise ValueError("direct SLAT subset contains duplicate uid/seed rows")
        object_targets: dict[str, tuple[str, str]] = {}
        for row in self.rows:
            object_uid = str(row["object_uid"])
            target_identity = (
                str(row["target_file"]),
                str(row["target_file_sha256"]),
            )
            previous = object_targets.setdefault(object_uid, target_identity)
            if previous != target_identity:
                raise ValueError(
                    f"object {object_uid} maps to multiple target artifacts"
                )
        if verify_hashes:
            checked: set[tuple[str, str]] = set()
            for row in self.rows:
                for file_key, hash_key in (
                    ("target_file", "target_file_sha256"),
                    ("support_file", "support_file_sha256"),
                    ("physical_file", "physical_file_sha256"),
                    ("condition_file", "condition_file_sha256"),
                    ("source_lh_slat", "source_lh_slat_sha256"),
                    ("source_glb", "source_glb_sha256"),
                    ("ss_latent", "ss_latent_sha256"),
                ):
                    identity = (str(row[file_key]), str(row[hash_key]))
                    if identity in checked:
                        continue
                    path = self._resolve(row[file_key])
                    actual = sha256_file(path)
                    if actual != str(row[hash_key]):
                        raise RuntimeError(
                            f"cache artifact mutation: {path} {actual} != {row[hash_key]}"
                        )
                    checked.add(identity)

    def _resolve(self, value: str | Path) -> Path:
        path = Path(value)
        return path if path.is_absolute() else self.root / path

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        uid = str(row["uid"])
        object_uid = str(row["object_uid"])
        with np.load(self._resolve(row["target_file"])) as payload:
            target_coords3 = np.asarray(payload["coords"], dtype=np.int32)
            target_feats = np.asarray(payload["feats"], dtype=np.float32)
        target_coords = torch.cat(
            [
                torch.zeros((len(target_coords3), 1), dtype=torch.int32),
                torch.from_numpy(target_coords3),
            ],
            dim=1,
        )
        target_feats_tensor = torch.from_numpy(target_feats)
        validate_sparse_target_alignment(
            target_coords,
            target_feats_tensor,
            require_single_batch=True,
        )
        support = torch.load(self._resolve(row["support_file"]), map_location="cpu")
        physical = torch.load(self._resolve(row["physical_file"]), map_location="cpu")
        condition = torch.load(self._resolve(row["condition_file"]), map_location="cpu")
        for label, payload in (
            ("support", support),
            ("physical", physical),
            ("condition", condition),
        ):
            if payload.get("format") != DIRECT_SLAT_CACHE_VERSION:
                raise ValueError(f"uid={uid} invalid {label} cache format")
            if str(payload.get("uid")) != uid:
                raise ValueError(f"uid={uid} mismatched {label} identity")
            if str(payload.get("object_uid")) != object_uid:
                raise ValueError(f"uid={uid} mismatched {label} object identity")
            if str(payload.get("config_hash")) != self.config_hash:
                raise ValueError(f"uid={uid} mismatched {label} config binding")
        if int(support.get("seed", -1)) != int(row["support_seed"]):
            raise ValueError(f"uid={uid} support seed mismatch")
        corrected_ss = support["corrected_ss"].float()
        occupancy = support["occupancy_logits64"].float()
        condition_arch = str(self.config.get("condition_arch", "legacy"))
        native_every_block = condition_arch in {
            "native_every_block_v1",
            "native_ss_genrecon_v2",
        }
        if native_every_block:
            if physical.get("unused_native_placeholder") is not True:
                raise ValueError(
                    f"uid={uid} native cache physical placeholder is not explicit"
                )
            physical_tokens = torch.empty((0,), dtype=torch.float32)
        else:
            physical_tokens = physical["physical_tokens16"].float()
        if corrected_ss.shape != (1, 8, 16, 16, 16):
            raise ValueError(f"uid={uid} corrected SS shape={tuple(corrected_ss.shape)}")
        if occupancy.shape != (1, 1, 64, 64, 64):
            raise ValueError(f"uid={uid} occupancy shape={tuple(occupancy.shape)}")
        if not native_every_block and physical_tokens.shape != (1, 16**3, 1024):
            raise ValueError(f"uid={uid} physical shape={tuple(physical_tokens.shape)}")
        if not all(
            bool(torch.isfinite(value).all().item())
            for value in (
                target_feats_tensor,
                corrected_ss,
                occupancy,
                physical_tokens,
            )
        ):
            raise ValueError(f"uid={uid} contains non-finite training tensors")
        native_condition = condition.get("condition")
        if not isinstance(native_condition, dict):
            raise ValueError(f"uid={uid} native condition is not a dictionary")
        if not isinstance(native_condition.get("cond"), list):
            raise ValueError(f"uid={uid} positive SLAT condition is not per-view list")
        if not isinstance(native_condition.get("neg_cond"), list):
            raise ValueError(f"uid={uid} negative SLAT condition is not per-view list")
        positive = native_condition["cond"]
        negative = native_condition["neg_cond"]
        if not positive or len(positive) != len(negative):
            raise ValueError(f"uid={uid} invalid positive/negative view counts")
        for view_index, (pos, neg) in enumerate(zip(positive, negative)):
            if not torch.is_tensor(pos) or not torch.is_tensor(neg):
                raise ValueError(f"uid={uid} condition view {view_index} is not tensor")
            if pos.ndim != 3 or pos.shape[0] != 1 or pos.shape[-1] != 1024:
                raise ValueError(
                    f"uid={uid} condition view {view_index} shape={tuple(pos.shape)}"
                )
            if pos.shape != neg.shape:
                raise ValueError(f"uid={uid} positive/negative condition shape mismatch")
            if not bool(torch.isfinite(pos.float()).all().item()) or not bool(
                torch.isfinite(neg.float()).all().item()
            ):
                raise ValueError(f"uid={uid} condition view {view_index} is non-finite")
        return {
            **row,
            "target_coords": target_coords,
            "target_feats": target_feats_tensor,
            "corrected_ss": corrected_ss,
            "occupancy_logits64": occupancy,
            "corrected_coords64": support["corrected_coords64"].to(torch.int32),
            "physical_tokens16": physical_tokens,
            "condition": native_condition,
        }


def collate_direct_slat_one(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if len(rows) != 1:
        raise ValueError("direct SLAT training currently requires per-rank batch size 1")
    return rows[0]
