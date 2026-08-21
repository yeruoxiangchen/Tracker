"""Strict base-cache + native ReconViaGen SS-context sidecar binding.

The official no-VGGT lifting/target cache remains immutable.  A paired
manifest joins each frozen row to exactly one cached ``ss_vggt_cond`` tensor.
Only ``stock_condition`` is replaced at runtime; the target, posed-DINO
features, known K/T and every other training tensor come from the base row.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from ar_ss_flow.local_pose_lifting_flow import PoseLiftingCacheDataset, parse_indices
from pose_point_depth_mv.proobjaverse_official_slat_protocol import (
    canonical_sha256,
    sha256_file,
)
from pose_point_depth_mv.proobjaverse_official_ss import (
    validate_official_ss_cache_contract,
)


CONTEXT_VERSION = (
    "official_ss_with_vggt_perf_v1.native_reconviagen_ss_context.v1"
)
SIDECAR_FORMAT = "official_ss_with_vggt_perf_v1.ss_sidecar_object.v1"
MANIFEST_FORMAT = "official_ss_with_vggt_perf_v1.paired_manifest.v1"
CACHE_REPORT_FORMAT = "official_ss_with_vggt_perf_v1.cache_report.v1"

MODEL_CONTEXT_CONTRACT = {
    "version": CONTEXT_VERSION,
    "architecture": "unchanged_native_ss_genrecon_v2",
    "stock_floor": "VSS0",
    "stock_context": "native_reconviagen_vggt_plus_dinov2_ss_vggt_cond",
    "stock_context_layout": "[1,4096,1024]",
    "spatial_condition": "posed_multiview_dino_frustum_lifting",
    "posed_dino_known_K_T": True,
    "vggt_camera_consumed": False,
    "vggt_depth_consumed": False,
    "target": "unchanged_official_ss_decoder_projected_latent",
    "negative_context_policy": "runtime_zeros_like_positive",
    "vggt_model_executed_during_cache_build": True,
    "vggt_model_executed_during_training": False,
}

_ROW_KEYS = {
    "uid",
    "object_uid",
    "base_index",
    "sidecar_file",
    "sidecar_file_sha256",
    "sidecar_file_size",
    "view_ids",
    "native_context_shape",
    "native_context_dtype",
    "source_render_tar_sha256",
}
_PAYLOAD_KEYS = {
    "format",
    "uid",
    "object_uid",
    "sidecar_contract_hash",
    "view_ids",
    "native_ss_vggt_cond",
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


def _resolve(manifest: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = manifest.parent / path
    return path.resolve(strict=True)


def validate_native_ss_vggt_context_tensor(
    value: Any, *, uid: str
) -> torch.Tensor:
    if not torch.is_tensor(value) or not torch.is_floating_point(value):
        raise ValueError(f"uid={uid} native SS context must be floating tensor")
    if tuple(value.shape) != (1, 4096, 1024):
        raise ValueError(
            f"uid={uid} native SS context shape={tuple(value.shape)}; "
            "expected=(1,4096,1024)"
        )
    if value.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError(f"uid={uid} native SS context dtype={value.dtype}")
    if not bool(torch.isfinite(value.float()).all().item()):
        raise ValueError(f"uid={uid} native SS context is non-finite")
    return value.contiguous()


class WithVGGTOfficialSSDataset(Dataset):
    """Expose the legacy dataset API while replacing only Stock SS context."""

    def __init__(self, manifest: str | Path, *, indices: str = "all") -> None:
        self.manifest_path = Path(manifest).expanduser().resolve(strict=True)
        payload = _load_json(self.manifest_path)
        if payload.get("format") != MANIFEST_FORMAT:
            raise ValueError(
                f"unexpected with-VGGT SS manifest={payload.get('format')!r}"
            )
        binding = payload.get("pair_binding")
        if not isinstance(binding, dict) or payload.get("pair_identity") != canonical_sha256(
            binding
        ):
            raise ValueError("with-VGGT SS pair identity mismatch")
        contract = binding.get("sidecar_contract")
        if not isinstance(contract, dict) or binding.get(
            "sidecar_contract_hash"
        ) != canonical_sha256(contract):
            raise ValueError("with-VGGT SS sidecar contract/hash mismatch")
        if contract.get("model_context") != MODEL_CONTEXT_CONTRACT:
            raise ValueError("with-VGGT SS model-context contract mismatch")
        base_binding = binding.get("base_cache")
        if not isinstance(base_binding, dict):
            raise ValueError("with-VGGT SS pair lacks base-cache binding")
        base_path = _resolve(self.manifest_path, base_binding.get("manifest", ""))
        if sha256_file(base_path) != base_binding.get("manifest_sha256"):
            raise ValueError("with-VGGT SS base manifest SHA256 mismatch")
        self.base = PoseLiftingCacheDataset(base_path, indices="all")
        if self.base.config_hash != base_binding.get("config_hash"):
            raise ValueError("with-VGGT SS base config identity mismatch")
        samples = payload.get("samples")
        if (
            not isinstance(samples, list)
            or not samples
            or len(samples) > len(self.base.rows)
        ):
            raise ValueError("with-VGGT SS sidecar/base sample count mismatch")
        paired_base_indices: list[int] = []
        for position, row in enumerate(samples):
            if not isinstance(row, dict) or set(row) != _ROW_KEYS:
                raise ValueError(f"with-VGGT SS row schema differs at {position}")
            base_index = int(row.get("base_index", -1))
            if not 0 <= base_index < len(self.base.rows):
                raise ValueError(
                    f"with-VGGT SS row base index differs at {position}"
                )
            paired_base_indices.append(base_index)
            base_row = self.base.rows[base_index]
            if (
                str(row.get("uid", "")) != str(base_row.get("uid", ""))
                or str(row.get("object_uid", ""))
                != str(base_row.get("object_uid", base_row.get("uid", "")))
            ):
                raise ValueError(f"with-VGGT SS row/base identity differs at {position}")
        if paired_base_indices != sorted(set(paired_base_indices)):
            raise ValueError(
                "with-VGGT SS base indices must be unique and strictly increasing"
            )
        expected_index_hash = canonical_sha256(samples)
        if binding.get("sidecar_index_sha256") != expected_index_hash:
            raise ValueError("with-VGGT SS sidecar index identity mismatch")
        if int(binding.get("sample_count", -1)) != len(samples):
            raise ValueError("with-VGGT SS bound sample count mismatch")
        selected = parse_indices(indices, len(samples))
        self.sidecar_rows = [samples[index] for index in selected]
        self.base_indices = [paired_base_indices[index] for index in selected]
        self.rows = [self.base.rows[index] for index in self.base_indices]
        self.visual_feature_dim = self.base.visual_feature_dim
        self.feature_metadata = copy.deepcopy(self.base.feature_metadata)
        self.config = copy.deepcopy(payload.get("config", {}))
        self.config_hash = str(payload.get("config_hash", ""))
        if not self.config_hash or self.config_hash != canonical_sha256(self.config):
            raise ValueError("with-VGGT SS joined config hash mismatch")
        self.source_cache_manifest = str(base_path)
        self.root = self.manifest_path.parent
        self.manifest_payload = payload
        self.pair_identity = str(payload["pair_identity"])
        self.sidecar_contract = copy.deepcopy(contract)
        self.sidecar_contract_hash = str(binding["sidecar_contract_hash"])

    def __len__(self) -> int:
        return len(self.base_indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        base_index = self.base_indices[index]
        base_sample = self.base[base_index]
        row = self.sidecar_rows[index]
        path = _resolve(self.manifest_path, row["sidecar_file"])
        if path.stat().st_size != int(row["sidecar_file_size"]):
            raise RuntimeError(f"uid={row['uid']} SS sidecar size changed")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict) or set(payload) != _PAYLOAD_KEYS:
            raise RuntimeError(f"uid={row['uid']} SS sidecar payload schema differs")
        if (
            payload.get("format") != SIDECAR_FORMAT
            or payload.get("uid") != row["uid"]
            or payload.get("object_uid") != row["object_uid"]
            or payload.get("sidecar_contract_hash") != self.sidecar_contract_hash
            or payload.get("negative_context_policy")
            != "runtime_zeros_like_positive"
            or payload.get("vggt_camera_consumed") is not False
            or payload.get("known_K_T_replaced") is not False
        ):
            raise RuntimeError(f"uid={row['uid']} SS sidecar semantics differ")
        view_ids = [int(value) for value in payload["view_ids"].tolist()]
        base_view_ids = [int(value) for value in base_sample["view_ids"].tolist()]
        if view_ids != row["view_ids"] or view_ids != base_view_ids:
            raise RuntimeError(f"uid={row['uid']} frozen view identity differs")
        context = validate_native_ss_vggt_context_tensor(
            payload["native_ss_vggt_cond"], uid=str(row["uid"])
        )
        if (
            list(context.shape) != row["native_context_shape"]
            or str(context.dtype) != row["native_context_dtype"]
        ):
            raise RuntimeError(f"uid={row['uid']} context row/payload differs")
        result = dict(base_sample)
        result["stock_condition"] = context
        result["stock_condition_source"] = "native_ss_vggt_cond_sidecar"
        result["with_vggt_sidecar_path"] = str(path)
        result["with_vggt_pair_identity"] = self.pair_identity
        return result


def validate_official_ss_with_vggt_cache_contract(
    dataset: Any, *, training_config_hash: str | None = None
) -> dict[str, Any]:
    if not isinstance(dataset, WithVGGTOfficialSSDataset):
        raise TypeError("official with-VGGT SS requires paired sidecar dataset")
    if training_config_hash is not None and str(training_config_hash) != str(
        dataset.config_hash
    ):
        raise RuntimeError("with-VGGT SS training config hash differs")
    base = validate_official_ss_cache_contract(dataset.base)
    base_no_vggt = base.pop("no_vggt", None)
    return {
        **base,
        "config_hash": dataset.config_hash,
        "base_cache_feature_contract": base_no_vggt,
        "with_vggt_ss": {
            "pair_identity": dataset.pair_identity,
            "sidecar_contract_hash": dataset.sidecar_contract_hash,
            "sidecar_contract": copy.deepcopy(dataset.sidecar_contract),
            "model_context": copy.deepcopy(MODEL_CONTEXT_CONTRACT),
        },
    }


def validate_official_ss_with_vggt_evaluation_cache_contract(
    dataset: Any, *, training_identity: dict[str, Any]
) -> dict[str, Any]:
    if not isinstance(training_identity, dict):
        raise TypeError("with-VGGT SS evaluation requires training identity")
    observed = validate_official_ss_with_vggt_cache_contract(dataset)
    trained = training_identity.get("feature_contract")
    if not isinstance(trained, dict):
        raise RuntimeError("with-VGGT SS checkpoint lacks feature contract")
    trained_targets = trained.get("official_ss_targets")
    observed_targets = observed.get("official_ss_targets")
    if not isinstance(trained_targets, dict) or not isinstance(observed_targets, dict):
        raise RuntimeError("with-VGGT SS official target binding is incomplete")
    if trained_targets.get("domain_contract") != observed_targets.get(
        "domain_contract"
    ):
        raise RuntimeError("with-VGGT SS train/eval target domains differ")
    if trained_targets.get("split") != "train" or observed_targets.get("split") == "train":
        raise RuntimeError("with-VGGT SS evaluation requires train -> held-out split")
    trained_vggt = trained.get("with_vggt_ss")
    observed_vggt = observed.get("with_vggt_ss")
    if not isinstance(trained_vggt, dict) or not isinstance(observed_vggt, dict):
        raise RuntimeError("with-VGGT SS context identity is incomplete")
    trained_contract = trained_vggt.get("sidecar_contract")
    observed_contract = observed_vggt.get("sidecar_contract")
    if not isinstance(trained_contract, dict) or not isinstance(observed_contract, dict):
        raise RuntimeError("with-VGGT SS sidecar contract is incomplete")
    comparable_keys = (
        "protocol_sha256",
        "selected_view_count",
        "shared_geometric_preprocessing",
        "shared_geometric_preprocessing_hash",
        "context_semantics",
        "model_context",
        "encoder_assets",
    )
    mismatch = {
        key: (trained_contract.get(key), observed_contract.get(key))
        for key in comparable_keys
        if trained_contract.get(key) != observed_contract.get(key)
    }
    if mismatch:
        raise RuntimeError(f"with-VGGT SS train/eval context differs={mismatch}")
    observed["evaluation_training_binding"] = {
        "mode": "official_with_vggt_ss_context",
        "training_config_hash": str(training_identity.get("config_hash", "")),
        "evaluation_config_hash": dataset.config_hash,
        "training_pair_identity": trained_vggt.get("pair_identity"),
        "evaluation_pair_identity": observed_vggt.get("pair_identity"),
        "training_split": "train",
        "evaluation_split": observed_targets.get("split"),
    }
    return observed


__all__ = [
    "CACHE_REPORT_FORMAT",
    "CONTEXT_VERSION",
    "MANIFEST_FORMAT",
    "MODEL_CONTEXT_CONTRACT",
    "SIDECAR_FORMAT",
    "WithVGGTOfficialSSDataset",
    "validate_native_ss_vggt_context_tensor",
    "validate_official_ss_with_vggt_cache_contract",
    "validate_official_ss_with_vggt_evaluation_cache_contract",
]
