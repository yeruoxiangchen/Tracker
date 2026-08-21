#!/usr/bin/env python3
"""Strict sidecar binding for official ProObjaverse with-VGGT SLat training.

The existing official no-VGGT cache is a frozen scientific asset.  This module
does not reinterpret or rewrite it.  Instead, it joins that cache with one
additional per-object tensor: ReconViaGen's native ``slat_vggt_cond`` context
computed from the exact already-frozen views and image preprocessing.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from ar_ss_flow.local_pose_lifting_flow import parse_indices
from pose_point_depth_mv.native_3d_condition import (
    NativeConditionSLatDataset as _BaseNativeConditionSLatDataset,
)
from pose_point_depth_mv.proobjaverse_official_slat_protocol import (
    canonical_sha256,
    sha256_file,
)


WITH_VGGT_CONTEXT_VERSION = (
    "pose_point_depth_mv.proobjaverse_official_native_slat_vggt_context.v1"
)
WITH_VGGT_SIDECAR_FORMAT = (
    "pose_point_depth_mv.proobjaverse_official_slat_vggt_sidecar_object.v1"
)
WITH_VGGT_SLAT_MANIFEST_FORMAT = (
    "pose_point_depth_mv.proobjaverse_official_slat_with_vggt_slat_manifest.v1"
)
WITH_VGGT_LIFTING_MANIFEST_FORMAT = (
    "pose_point_depth_mv.proobjaverse_official_slat_with_vggt_lifting_manifest.v1"
)
WITH_VGGT_CACHE_REPORT_FORMAT = (
    "pose_point_depth_mv.proobjaverse_official_slat_with_vggt_cache.v1"
)

_PAIR_BINDING_KEYS = {
    "version",
    "base_cache",
    "sidecar_contract",
    "sidecar_contract_hash",
    "sidecar_index_sha256",
    "sample_count",
    "ordered_uid_sha256",
}
_PAIR_BASE_CACHE_KEYS = {
    "slat_manifest",
    "slat_manifest_sha256",
    "lifting_manifest",
    "lifting_manifest_sha256",
    "slat_config_hash",
    "lifting_config_hash",
    "slat_normalization_hash",
}
_SIDECAR_CONTRACT_KEYS = {
    "version",
    "builder_version",
    "protocol_sha256",
    "split",
    "base_cache",
    "selected_view_policy",
    "selected_view_count",
    "shared_geometric_preprocessing",
    "shared_geometric_preprocessing_hash",
    "native_condition",
    "camera_contract",
    "native_ss_contract",
    "encoder_assets",
}
_CONTRACT_BASE_CACHE_KEYS = {
    "slat_manifest_sha256",
    "lifting_manifest_sha256",
    "slat_config_hash",
    "lifting_config_hash",
    "slat_normalization_hash",
}
_ENCODER_ASSET_KEYS = {
    "pretrained",
    "trellis_snapshot_revision",
    "pipeline_json_sha256",
    "slat_vggt_cond_config_sha256",
    "slat_vggt_cond_weights_sha256",
    "vggt_repo",
    "vggt_snapshot_revision",
    "vggt_config_sha256",
    "vggt_weights_sha256",
    "dino_model",
    "dino_weights_filename",
    "dino_weights_sha256",
    "dino_source_tree_sha256",
    "vggt_source_tree_sha256",
    "trellis_pipeline_source_sha256",
    "slat_condition_source_sha256",
    "shared_preprocessing_source_sha256",
    "builder_source_sha256",
}
_SAMPLE_ROW_KEYS = {
    "uid",
    "object_uid",
    "support_seed",
    "base_index",
    "sidecar_file",
    "sidecar_file_sha256",
    "sidecar_file_size",
    "view_ids",
    "native_context_shape",
    "native_context_dtype",
    "decoded_source_rgba_sha256",
    "processed_input_rgb_sha256",
    "source_render_tar_sha256",
}
_SIDECAR_PAYLOAD_KEYS = {
    "format",
    "uid",
    "object_uid",
    "support_seed",
    "sidecar_contract_hash",
    "view_ids",
    "native_slat_vggt_cond",
    "negative_context_policy",
    "decoded_source_rgba_sha256",
    "processed_input_rgb_sha256",
    "vggt_camera_consumed",
    "known_K_T_replaced",
}


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _resolve_reference(manifest_path: Path, value: str | Path) -> Path:
    if not str(value):
        raise ValueError(f"empty path reference in {manifest_path}")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = manifest_path.parent / path
    return path.resolve(strict=True)


def _manifest_binding(
    manifest_path: Path,
    payload: dict[str, Any],
    *,
    expected_format: str,
) -> dict[str, Any]:
    if payload.get("format") != expected_format:
        raise ValueError(
            f"unexpected with-VGGT manifest format={payload.get('format')!r}; "
            f"expected={expected_format!r}"
        )
    binding = payload.get("pair_binding")
    if not isinstance(binding, dict):
        raise ValueError(f"with-VGGT manifest lacks pair_binding: {manifest_path}")
    if payload.get("pair_identity") != canonical_sha256(binding):
        raise ValueError(f"with-VGGT pair identity mismatch: {manifest_path}")
    if set(binding) != _PAIR_BINDING_KEYS:
        raise ValueError(
            f"with-VGGT v1 pair binding schema differs: "
            f"missing={sorted(_PAIR_BINDING_KEYS - set(binding))} "
            f"unexpected={sorted(set(binding) - _PAIR_BINDING_KEYS)}"
        )
    contract = binding.get("sidecar_contract")
    if not isinstance(contract, dict) or canonical_sha256(contract) != binding.get(
        "sidecar_contract_hash"
    ):
        raise ValueError("with-VGGT sidecar contract/hash differs")
    if set(contract) != _SIDECAR_CONTRACT_KEYS:
        raise ValueError(
            "with-VGGT v1 sidecar contract schema differs: "
            f"missing={sorted(_SIDECAR_CONTRACT_KEYS - set(contract))} "
            f"unexpected={sorted(set(contract) - _SIDECAR_CONTRACT_KEYS)}"
        )
    contract_base = contract.get("base_cache")
    encoder_assets = contract.get("encoder_assets")
    if not isinstance(contract_base, dict) or set(contract_base) != (
        _CONTRACT_BASE_CACHE_KEYS
    ):
        raise ValueError("with-VGGT v1 contract base-cache schema differs")
    if not isinstance(encoder_assets, dict) or set(encoder_assets) != (
        _ENCODER_ASSET_KEYS
    ):
        raise ValueError("with-VGGT v1 encoder-asset schema differs")
    native = dict(contract.get("native_condition", {}))
    camera = dict(contract.get("camera_contract", {}))
    expected_native = {
        "producer": (
            "TrellisVGGTTo3DPipeline.vggt_feat + encode_image + get_slat_cond"
        ),
        "vggt_layers": [4, 11, 17, 23],
        "dino_sequence": "full_cls_register_patch_sequence",
        "expected_dino_prefix_tokens": 5,
        "output_layout": "[views,tokens,1024]",
        "positive_context_materialized": True,
        "negative_context_policy": "runtime_zeros_like_positive",
        "base_dino_patch_replay": {
            "source": "same encode_image result used by native slat_vggt_cond",
            "reference": "immutable base lifting visual_patch_features",
            "comparison_dtype": "torch.float16",
            "max_abs_tolerance": 0.01,
            "mean_abs_tolerance": 0.0005,
        },
    }
    expected_camera = {
        "vggt_camera_consumed": False,
        "vggt_depth_consumed": False,
        "posed_dino_uses_base_known_K_T": True,
        "known_K_T_replaced": False,
    }
    if (
        contract.get("version") != WITH_VGGT_CONTEXT_VERSION
        or native != expected_native
        or camera != expected_camera
    ):
        raise ValueError("with-VGGT v1 sidecar semantic contract differs")
    config = payload.get("config")
    if not isinstance(config, dict):
        raise ValueError(f"with-VGGT manifest lacks config: {manifest_path}")
    if payload.get("config_hash") != canonical_sha256(config):
        raise ValueError(f"with-VGGT config hash mismatch: {manifest_path}")
    return binding


def _row_identity(row: dict[str, Any]) -> tuple[str, int]:
    return str(row.get("uid", "")), int(row.get("support_seed", 42))


def validate_native_slat_vggt_context_tensor(
    context: Any,
    *,
    views: int,
    uid: str,
) -> torch.Tensor:
    if not torch.is_tensor(context) or not torch.is_floating_point(context):
        raise ValueError(f"uid={uid} with-VGGT context must be a floating tensor")
    if context.ndim != 3:
        raise ValueError(
            f"uid={uid} with-VGGT context must be [V,T,1024], got {tuple(context.shape)}"
        )
    if int(context.shape[0]) != int(views) or int(context.shape[-1]) != 1024:
        raise ValueError(
            f"uid={uid} with-VGGT context shape differs: "
            f"{tuple(context.shape)} vs views={views}, channels=1024"
        )
    if int(context.shape[1]) <= 0:
        raise ValueError(f"uid={uid} with-VGGT context has no tokens")
    if context.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError(
            f"uid={uid} unsupported with-VGGT context dtype={context.dtype}"
        )
    if not bool(torch.isfinite(context.float()).all().item()):
        raise ValueError(f"uid={uid} with-VGGT context contains non-finite values")
    return context.contiguous()


class WithVGGTNativeConditionSLatDataset(Dataset):
    """Join an immutable official cache with its native VGGT context sidecar."""

    def __init__(
        self,
        slat_manifest: str | Path,
        lifting_manifest: str | Path,
        *,
        indices: str = "all",
        verify_hashes: bool = False,
    ) -> None:
        self.slat_manifest_path = Path(slat_manifest).expanduser().resolve(strict=True)
        self.lifting_manifest_path = (
            Path(lifting_manifest).expanduser().resolve(strict=True)
        )
        slat_payload = _load_json(self.slat_manifest_path)
        lifting_payload = _load_json(self.lifting_manifest_path)
        slat_binding = _manifest_binding(
            self.slat_manifest_path,
            slat_payload,
            expected_format=WITH_VGGT_SLAT_MANIFEST_FORMAT,
        )
        lifting_binding = _manifest_binding(
            self.lifting_manifest_path,
            lifting_payload,
            expected_format=WITH_VGGT_LIFTING_MANIFEST_FORMAT,
        )
        if slat_binding != lifting_binding:
            raise ValueError("with-VGGT SLat/lifting manifests bind different pairs")
        if slat_payload.get("pair_identity") != lifting_payload.get("pair_identity"):
            raise ValueError("with-VGGT SLat/lifting pair identities differ")
        if slat_payload.get("config") != lifting_payload.get("config"):
            raise ValueError("with-VGGT SLat/lifting configs differ")

        all_slat_rows = slat_payload.get("samples")
        all_lifting_rows = lifting_payload.get("samples")
        if not isinstance(all_slat_rows, list) or not all_slat_rows:
            raise ValueError("with-VGGT SLat manifest has no samples")
        if all_slat_rows != all_lifting_rows:
            raise ValueError("with-VGGT paired manifests have different sample rows")
        bad_row_schemas = [
            index
            for index, row in enumerate(all_slat_rows)
            if not isinstance(row, dict) or set(row) != _SAMPLE_ROW_KEYS
        ]
        if bad_row_schemas:
            raise ValueError(
                f"with-VGGT v1 sample row schema differs at {bad_row_schemas[:8]}"
            )
        if int(slat_binding.get("sample_count", -1)) != len(all_slat_rows):
            raise ValueError("with-VGGT pair binding sample count differs")
        if slat_binding.get("sidecar_index_sha256") != canonical_sha256(
            all_slat_rows
        ):
            raise ValueError("with-VGGT sidecar index hash differs")
        if slat_binding.get("ordered_uid_sha256") != canonical_sha256(
            [str(row["uid"]) for row in all_slat_rows]
        ):
            raise ValueError("with-VGGT ordered UID hash differs")
        selected = parse_indices(indices, len(all_slat_rows))
        sidecar_rows = [dict(all_slat_rows[index]) for index in selected]
        identities = [_row_identity(row) for row in sidecar_rows]
        if any(not uid for uid, _ in identities) or len(set(identities)) != len(
            identities
        ):
            raise ValueError("with-VGGT subset contains empty/duplicate identities")

        base = slat_binding.get("base_cache")
        if not isinstance(base, dict) or set(base) != _PAIR_BASE_CACHE_KEYS:
            raise ValueError("with-VGGT pair binding lacks base_cache")
        contract_base = slat_binding["sidecar_contract"]["base_cache"]
        if contract_base != {
            key: base[key] for key in _CONTRACT_BASE_CACHE_KEYS
        }:
            raise RuntimeError("with-VGGT pair/sidecar base-cache bindings differ")
        base_slat_path = _resolve_reference(
            self.slat_manifest_path, base.get("slat_manifest", "")
        )
        base_lifting_path = _resolve_reference(
            self.slat_manifest_path, base.get("lifting_manifest", "")
        )
        expected_base_hashes = {
            "slat_manifest_sha256": sha256_file(base_slat_path),
            "lifting_manifest_sha256": sha256_file(base_lifting_path),
        }
        mismatch = {
            key: (base.get(key), value)
            for key, value in expected_base_hashes.items()
            if base.get(key) != value
        }
        if mismatch:
            raise RuntimeError(f"with-VGGT base cache artifact changed: {mismatch}")

        self.base = _BaseNativeConditionSLatDataset(
            base_slat_path,
            base_lifting_path,
            indices="all",
            # The historical official rows intentionally carry an empty
            # source_glb field.  Their legacy verifier tries to hash that empty
            # path as a directory.  Verify the selected, non-empty artifacts
            # explicitly below instead of weakening any real file check.
            verify_hashes=False,
        )
        if base.get("slat_config_hash") != self.base.config_hash:
            raise RuntimeError("with-VGGT/base SLat config binding differs")
        if base.get("lifting_config_hash") != self.base.lifting.config_hash:
            raise RuntimeError("with-VGGT/base lifting config binding differs")
        if base.get("slat_normalization_hash") != self.base.slat_normalization_hash:
            raise RuntimeError("with-VGGT/base normalization binding differs")

        base_by_identity: dict[tuple[str, int], int] = {}
        for index, row in enumerate(self.base.rows):
            identity = _row_identity(row)
            if identity in base_by_identity:
                raise ValueError(f"base cache duplicate identity={identity}")
            base_by_identity[identity] = index
        missing = [identity for identity in identities if identity not in base_by_identity]
        if missing:
            raise ValueError(
                f"with-VGGT/base cache UID join incomplete: {missing[:8]} count={len(missing)}"
            )
        reordered = [
            row["uid"]
            for row, identity in zip(sidecar_rows, identities)
            if int(row["base_index"]) != int(base_by_identity[identity])
        ]
        if reordered:
            raise ValueError(
                f"with-VGGT/base frozen row indices differ: {reordered[:8]}"
            )

        self.sidecar_rows = sidecar_rows
        self.base_indices = [base_by_identity[identity] for identity in identities]
        self.rows = [self.base.rows[index] for index in self.base_indices]
        self.config = copy.deepcopy(slat_payload["config"])
        self.config_hash = str(slat_payload["config_hash"])
        self.slat_normalization = copy.deepcopy(self.base.slat_normalization)
        self.slat_normalization_hash = str(self.base.slat_normalization_hash)
        self.pair_identity = str(slat_payload["pair_identity"])
        self.sidecar_contract_hash = str(slat_binding["sidecar_contract_hash"])
        expected_input_context = {
            "version": WITH_VGGT_CONTEXT_VERSION,
            "stock_floor": "V0",
            "source": "native_reconviagen_vggt_plus_dinov2_slat_vggt_cond",
            "sidecar_contract_hash": self.sidecar_contract_hash,
            "base_no_vggt_slat_config_hash": str(self.base.config_hash),
            "selected_views": "exact ordered base lifting view_ids",
            "native_full_dino_sequence": True,
            "vggt_model_executed": True,
            "vggt_camera_consumed": False,
            "known_pose_dino_branch_unchanged": True,
            "negative_context_policy": "runtime_zeros_like_positive",
        }
        if self.config.get("slat_input_context") != expected_input_context:
            raise ValueError("with-VGGT v1 manifest input-context contract differs")
        self.root = self.slat_manifest_path.parent
        self.verify_hashes = bool(verify_hashes)
        self.slat = self.base.slat
        self.lifting = self.base.lifting
        self.lifting_indices = list(self.base_indices)
        self.compact_backend = None
        self.identity = {
            "version": WITH_VGGT_CONTEXT_VERSION,
            "slat_manifest": str(self.slat_manifest_path),
            "slat_manifest_sha256": sha256_file(self.slat_manifest_path),
            "lifting_manifest": str(self.lifting_manifest_path),
            "lifting_manifest_sha256": sha256_file(self.lifting_manifest_path),
            "pair_identity": self.pair_identity,
            "sidecar_contract_hash": self.sidecar_contract_hash,
            "base_cache_identity": copy.deepcopy(base),
            "uid_count": len(self.rows),
            "uid_hash": canonical_sha256([uid for uid, _ in identities]),
            "vggt_model_executed": True,
            "vggt_camera_consumed": False,
        }
        if self.verify_hashes:
            checked: set[tuple[str, str]] = set()
            for base_index in self.base_indices:
                base_row = self.base.slat.rows[base_index]
                for file_key, hash_key in (
                    ("target_file", "target_file_sha256"),
                    ("support_file", "support_file_sha256"),
                    ("physical_file", "physical_file_sha256"),
                    ("condition_file", "condition_file_sha256"),
                    ("source_lh_slat", "source_lh_slat_sha256"),
                    ("source_glb", "source_glb_sha256"),
                    ("ss_latent", "ss_latent_sha256"),
                ):
                    raw_path = str(base_row.get(file_key, ""))
                    expected_hash = str(base_row.get(hash_key, ""))
                    if not raw_path and not expected_hash:
                        continue
                    if not raw_path or not expected_hash:
                        raise RuntimeError(
                            f"base cache incomplete file identity uid={base_row['uid']} "
                            f"field={file_key}"
                        )
                    identity = (raw_path, expected_hash)
                    if identity not in checked:
                        path = self.base.slat._resolve(raw_path)
                        if sha256_file(path) != expected_hash:
                            raise RuntimeError(f"with-VGGT base artifact changed: {path}")
                        checked.add(identity)
                lifting_index = self.base.lifting_indices[base_index]
                lifting_row = self.base.lifting.rows[lifting_index]
                raw_path = str(lifting_row["cache_file"])
                expected_hash = str(lifting_row["cache_file_sha256"])
                identity = (raw_path, expected_hash)
                if identity not in checked:
                    path = Path(raw_path)
                    if not path.is_absolute():
                        path = self.base.lifting.root / path
                    if sha256_file(path) != expected_hash:
                        raise RuntimeError(f"with-VGGT base lifting artifact changed: {path}")
                    checked.add(identity)
            for row in self.sidecar_rows:
                path = self._sidecar_path(row)
                actual = sha256_file(path)
                if actual != row.get("sidecar_file_sha256"):
                    raise RuntimeError(
                        f"with-VGGT sidecar artifact changed: {path} "
                        f"{actual} != {row.get('sidecar_file_sha256')}"
                    )

    def _sidecar_path(self, row: dict[str, Any]) -> Path:
        return _resolve_reference(self.slat_manifest_path, row["sidecar_file"])

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
        self.sidecar_rows = [self.sidecar_rows[index] for index in selected]
        self.base_indices = [self.base_indices[index] for index in selected]
        self.rows = [self.rows[index] for index in selected]
        self.lifting_indices = list(self.base_indices)
        identities = [_row_identity(row) for row in self.sidecar_rows]
        self.identity.update(
            {
                "uid_count": len(self.rows),
                "object_count": len(allowed),
                "uid_hash": canonical_sha256([uid for uid, _ in identities]),
            }
        )

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.sidecar_rows[index]
        sample = self.base[self.base_indices[index]]
        uid = str(row["uid"])
        if str(sample.get("uid")) != uid:
            raise RuntimeError("with-VGGT/base runtime UID join changed")
        path = self._sidecar_path(row)
        if path.stat().st_size != int(row["sidecar_file_size"]):
            raise RuntimeError(f"with-VGGT sidecar size changed: {path}")
        if self.verify_hashes and sha256_file(path) != row.get("sidecar_file_sha256"):
            raise RuntimeError(f"with-VGGT sidecar artifact changed: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict) or set(payload) != _SIDECAR_PAYLOAD_KEYS:
            raise ValueError(f"uid={uid} with-VGGT v1 sidecar schema differs")
        if payload.get("format") != WITH_VGGT_SIDECAR_FORMAT:
            raise ValueError(f"uid={uid} invalid with-VGGT sidecar format")
        if str(payload.get("uid")) != uid or str(payload.get("object_uid")) != str(
            row["object_uid"]
        ):
            raise ValueError(f"uid={uid} with-VGGT sidecar identity mismatch")
        if payload.get("sidecar_contract_hash") != self.sidecar_contract_hash:
            raise ValueError(f"uid={uid} with-VGGT sidecar contract mismatch")
        if int(payload.get("support_seed", -1)) != int(row["support_seed"]):
            raise ValueError(f"uid={uid} with-VGGT support seed mismatch")
        if (
            payload.get("negative_context_policy")
            != "runtime_zeros_like_positive"
            or payload.get("vggt_camera_consumed") is not False
            or payload.get("known_K_T_replaced") is not False
        ):
            raise ValueError(f"uid={uid} with-VGGT sidecar semantics differ")
        view_ids = payload.get("view_ids")
        if not torch.is_tensor(view_ids) or view_ids.ndim != 1:
            raise ValueError(f"uid={uid} sidecar view_ids must be a vector")
        runtime_view_ids = sample["lifting_sample"].get("view_ids")
        if not torch.is_tensor(runtime_view_ids) or not torch.equal(
            view_ids.to(torch.int64), runtime_view_ids.to(torch.int64)
        ):
            raise RuntimeError(f"uid={uid} with-VGGT/base selected views differ")
        context = validate_native_slat_vggt_context_tensor(
            payload.get("native_slat_vggt_cond"),
            views=len(view_ids),
            uid=uid,
        )
        visual_patch_features = sample["lifting_sample"].get(
            "visual_patch_features"
        )
        if (
            not torch.is_tensor(visual_patch_features)
            or visual_patch_features.ndim != 3
            or int(context.shape[1])
            - int(visual_patch_features.shape[1])
            != 5
        ):
            raise ValueError(
                f"uid={uid} native VGGT context does not preserve the expected "
                "five full-DINO prefix tokens"
            )
        if (
            list(context.shape) != row["native_context_shape"]
            or str(context.dtype) != row["native_context_dtype"]
            or payload["decoded_source_rgba_sha256"]
            != row["decoded_source_rgba_sha256"]
            or payload["processed_input_rgb_sha256"]
            != row["processed_input_rgb_sha256"]
        ):
            raise ValueError(f"uid={uid} with-VGGT sidecar/manifest metadata differs")
        positive = [context[view : view + 1] for view in range(len(view_ids))]
        condition = {
            "cond": positive,
            "neg_cond": [torch.zeros_like(value) for value in positive],
        }
        return {
            **sample,
            "condition": condition,
            "with_vggt_sidecar": {
                "path": str(path),
                "sidecar_contract_hash": self.sidecar_contract_hash,
                "view_ids": view_ids.clone(),
                "native_context_shape": list(context.shape),
                "negative_context_policy": "runtime_zeros_like_positive",
                "vggt_camera_consumed": False,
            },
        }


__all__ = [
    "WITH_VGGT_CONTEXT_VERSION",
    "WITH_VGGT_SIDECAR_FORMAT",
    "WITH_VGGT_SLAT_MANIFEST_FORMAT",
    "WITH_VGGT_LIFTING_MANIFEST_FORMAT",
    "WITH_VGGT_CACHE_REPORT_FORMAT",
    "WithVGGTNativeConditionSLatDataset",
    "validate_native_slat_vggt_context_tensor",
]
