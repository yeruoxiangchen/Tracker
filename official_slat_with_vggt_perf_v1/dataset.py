"""Strict with-VGGT dataset with no redundant legacy-condition read."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from pose_point_depth_mv.direct_slat_data import DirectSLatCacheDataset
from pose_point_depth_mv.direct_slat_flow import (
    DIRECT_SLAT_CACHE_VERSION,
    validate_sparse_target_alignment,
)
from pose_point_depth_mv.proobjaverse_official_slat_with_vggt_cache import (
    WithVGGTNativeConditionSLatDataset,
)


class DirectSLatWithoutLegacyCondition(DirectSLatCacheDataset):
    """Load target/support/physical tensors while deliberately skipping condition.pt.

    The paired with-VGGT manifest still binds the immutable legacy condition
    artifact and its manifest hash.  It is not a numerical input to V, however,
    because the exact native ``slat_vggt_cond`` sidecar replaces N0 context.
    """

    def __init__(
        self,
        manifest: str | Path,
        *,
        indices: str = "all",
        verify_hashes: bool = False,
        check_tensor_finite: bool = True,
    ) -> None:
        super().__init__(
            manifest,
            indices=indices,
            verify_hashes=verify_hashes,
        )
        self.check_tensor_finite = bool(check_tensor_finite)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        uid = str(row["uid"])
        object_uid = str(row["object_uid"])
        with np.load(self._resolve(row["target_file"])) as payload:
            target_coords3 = np.asarray(payload["coords"], dtype=np.int32)
            target_feats = np.asarray(payload["feats"], dtype=np.float32)
        target_coords = torch.cat(
            (
                torch.zeros((len(target_coords3), 1), dtype=torch.int32),
                torch.from_numpy(target_coords3),
            ),
            dim=1,
        )
        target_feats_tensor = torch.from_numpy(target_feats)
        validate_sparse_target_alignment(
            target_coords,
            target_feats_tensor,
            require_single_batch=True,
        )

        support = torch.load(
            self._resolve(row["support_file"]),
            map_location="cpu",
            weights_only=False,
        )
        physical = torch.load(
            self._resolve(row["physical_file"]),
            map_location="cpu",
            weights_only=False,
        )
        for label, payload in (("support", support), ("physical", physical)):
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
        if condition_arch not in {"native_every_block_v1", "native_ss_genrecon_v2"}:
            raise ValueError(
                f"uid={uid} with-VGGT strict loader requires native condition cache"
            )
        if physical.get("unused_native_placeholder") is not True:
            raise ValueError(f"uid={uid} native physical placeholder is not explicit")
        physical_tokens = torch.empty((0,), dtype=torch.float32)
        if corrected_ss.shape != (1, 8, 16, 16, 16):
            raise ValueError(f"uid={uid} corrected SS shape={tuple(corrected_ss.shape)}")
        if occupancy.shape != (1, 1, 64, 64, 64):
            raise ValueError(f"uid={uid} occupancy shape={tuple(occupancy.shape)}")
        if self.check_tensor_finite and not all(
            bool(torch.isfinite(value).all().item())
            for value in (
                target_feats_tensor,
                corrected_ss,
                occupancy,
                physical_tokens,
            )
        ):
            raise ValueError(f"uid={uid} contains non-finite training tensors")
        return {
            **row,
            "target_coords": target_coords,
            "target_feats": target_feats_tensor,
            "corrected_ss": corrected_ss,
            "occupancy_logits64": occupancy,
            "corrected_coords64": support["corrected_coords64"].to(torch.int32),
            "physical_tokens16": physical_tokens,
            "runtime_condition_source": "with_vggt_sidecar_only",
            "legacy_condition_file_loaded": False,
        }


class StrictWithVGGTNativeConditionSLatDataset(
    WithVGGTNativeConditionSLatDataset
):
    """Pair the immutable base with sidecars and install the lean base loader."""

    def __init__(
        self,
        slat_manifest: str | Path,
        lifting_manifest: str | Path,
        *,
        indices: str = "all",
        verify_hashes: bool = False,
        check_tensor_finite: bool = True,
    ) -> None:
        super().__init__(
            slat_manifest,
            lifting_manifest,
            indices=indices,
            verify_hashes=verify_hashes,
        )
        if self.base.compact_backend is not None:
            raise RuntimeError(
                "with-VGGT strict v1 expects the frozen official v1 base cache"
            )
        original = self.base.slat
        lean = DirectSLatWithoutLegacyCondition(
            original.manifest_path,
            indices="all",
            # The parent already performed the requested one-time hash audit.
            verify_hashes=False,
            check_tensor_finite=check_tensor_finite,
        )
        if lean.rows != original.rows:
            raise RuntimeError("lean/base SLat row identity differs")
        if (
            lean.config != original.config
            or lean.config_hash != original.config_hash
            or lean.slat_normalization != original.slat_normalization
            or lean.slat_normalization_hash != original.slat_normalization_hash
        ):
            raise RuntimeError("lean/base SLat scientific contract differs")
        self.base.slat = lean
        self.base.rows = lean.rows
        self.base.config = lean.config
        self.base.config_hash = lean.config_hash
        self.base.slat_normalization = lean.slat_normalization
        self.base.slat_normalization_hash = lean.slat_normalization_hash
        self.slat = lean
        self.rows = [lean.rows[index] for index in self.base_indices]
        self.check_tensor_finite = bool(check_tensor_finite)
        self.identity["runtime_io_policy"] = {
            "version": "official_slat_with_vggt.no_legacy_condition_read.v1",
            "legacy_condition_manifest_identity_preserved": True,
            "legacy_condition_file_loaded_per_sample": False,
            "native_vggt_sidecar_loaded_per_sample": True,
            "per_sample_finite_checks": self.check_tensor_finite,
        }

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = super().__getitem__(index)
        if sample.get("legacy_condition_file_loaded") is not False:
            raise RuntimeError("strict with-VGGT loader lost its no-read guarantee")
        if sample.get("runtime_condition_source") != "with_vggt_sidecar_only":
            raise RuntimeError("strict with-VGGT condition source differs")
        return sample


__all__ = [
    "DirectSLatWithoutLegacyCondition",
    "StrictWithVGGTNativeConditionSLatDataset",
]
