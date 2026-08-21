#!/usr/bin/env python3
"""DINO-only image-condition contract shared by training and inference.

This module deliberately has no VGGT import.  It converts per-view DINO patch
tokens into the context layouts already accepted by the frozen SS and SLat
Flow models, so the model architecture and trainable parameter shapes remain
identical to Native v2 Full.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

import torch

from ar_ss_flow.shared_object_preprocessing import (
    SHARED_OBJECT_PREPROCESSING_VERSION,
    canonical_json_sha256 as preprocessing_json_sha256,
)


DINO_ONLY_CONTEXT_VERSION = "pose_point_depth_mv.dino_only_context.v1"
DINO_ONLY_LIFTING_VERSION = "pose_point_depth_mv.dino_only_lifting.v1"
DINO_FEATURE_DIM = 1024
DEFAULT_PATCH_COUNT = 1369
DEFAULT_SS_CONTEXT_TOKENS = 4096


def tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().contiguous().cpu()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
    digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def tensor_tree_sha256(value: Any) -> str:
    digest = hashlib.sha256()

    def visit(item: Any) -> None:
        if torch.is_tensor(item):
            digest.update(tensor_sha256(item).encode("ascii"))
        elif isinstance(item, dict):
            for key in sorted(item):
                digest.update(str(key).encode("utf-8"))
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)
        else:
            digest.update(repr(item).encode("utf-8"))

    visit(value)
    return digest.hexdigest()


def select_dino_patch_features(features: torch.Tensor) -> torch.Tensor:
    """Select DINO patches from Full 3072-D or already DINO-only inputs."""

    if not torch.is_tensor(features) or features.ndim != 3:
        raise ValueError("visual features must be [views,patches,channels]")
    if int(features.shape[0]) <= 0 or int(features.shape[1]) <= 0:
        raise ValueError("visual features must contain views and patches")
    channels = int(features.shape[-1])
    if channels < DINO_FEATURE_DIM:
        raise ValueError(f"DINO features require >=1024 channels, got {channels}")
    dino = features if channels == DINO_FEATURE_DIM else features[..., -DINO_FEATURE_DIM:]
    if not bool(torch.isfinite(dino.float()).all().item()):
        raise ValueError("selected DINO channels contain non-finite values")
    return dino.contiguous()


def deterministic_token_indices(
    total: int,
    maximum: int,
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Select a stable, order-preserving coverage of a flattened token stream."""

    total = int(total)
    maximum = int(maximum)
    if total <= 0 or maximum <= 0:
        raise ValueError("token counts must be positive")
    if total <= maximum:
        return torch.arange(total, dtype=torch.long, device=device)
    # Integer midpoint sampling avoids floating-point/version-dependent rounding.
    indices = (
        (2 * torch.arange(maximum, dtype=torch.int64, device=device) + 1) * total
    ) // (2 * maximum)
    if int(torch.unique_consecutive(indices).numel()) != maximum:
        raise RuntimeError("deterministic token cap produced duplicate indices")
    return indices


def build_dino_only_contexts(
    features: torch.Tensor,
    *,
    ss_context_tokens: int = DEFAULT_SS_CONTEXT_TOKENS,
) -> dict[str, Any]:
    """Build SS global context and SLat per-view contexts from DINO only."""

    dino = select_dino_patch_features(features)
    views, patches, channels = map(int, dino.shape)
    flattened = dino.reshape(views * patches, channels)
    indices = deterministic_token_indices(
        len(flattened),
        int(ss_context_tokens),
        device=flattened.device,
    )
    stock_condition = flattened.index_select(0, indices).unsqueeze(0).contiguous()
    positive = [dino[index].unsqueeze(0).contiguous() for index in range(views)]
    slat_condition = {
        "cond": positive,
        "neg_cond": [torch.zeros_like(value) for value in positive],
    }
    return {
        "visual_patch_features": dino,
        "stock_condition": stock_condition,
        "slat_condition": slat_condition,
        "context_contract": {
            "version": DINO_ONLY_CONTEXT_VERSION,
            "vggt_feature_dim": 0,
            "dino_feature_dim": DINO_FEATURE_DIM,
            "patch_count": patches,
            "view_count": views,
            "ss_context_source": "raw_dino_patch_tokens",
            "ss_context_token_cap": int(ss_context_tokens),
            "ss_context_token_count": int(stock_condition.shape[1]),
            "ss_context_selection": "flattened_order_integer_midpoint_v1",
            "slat_context_source": "raw_dino_patch_tokens_per_view",
            "negative_context": "zeros_like_positive",
            "vggt_model_executed": False,
        },
    }


def dino_only_feature_metadata(*, patch_count: int) -> dict[str, Any]:
    patch_count = int(patch_count)
    side = int(round(math.sqrt(patch_count)))
    if side * side != patch_count:
        raise ValueError(f"DINO patch count must be square, got {patch_count}")
    return {
        "aggregated_layer_count": 0,
        "selected_vggt_feature_index": None,
        "patch_start_idx": 5,
        "patch_count": patch_count,
        "patch_side": side,
        "vggt_feature_dim": 0,
        "dino_feature_dim": DINO_FEATURE_DIM,
        "context_source": "raw_dino_only",
        "vggt_model_executed": False,
        "depth_shape": [518, 518],
    }


def validate_dino_only_lifting_contract(
    dataset: Any, *, training_config_hash: str | None = None
) -> dict[str, Any]:
    metadata = dict(getattr(dataset, "feature_metadata", {}))
    config = dict(getattr(dataset, "config", {}))
    no_vggt = dict(config.get("no_vggt", {}))
    preprocessing = dict(config.get("geometric_preprocessing", {}))
    contract = {
        "version": str(no_vggt.get("version", "")),
        "visual_feature_dim": int(getattr(dataset, "visual_feature_dim", 0)),
        "vggt_feature_dim": int(metadata.get("vggt_feature_dim", -1)),
        "dino_feature_dim": int(metadata.get("dino_feature_dim", -1)),
        "patch_count": int(metadata.get("patch_count", 0)),
        "context_source": str(metadata.get("context_source", "")),
        "vggt_model_executed": metadata.get("vggt_model_executed"),
        "stock_condition_source": str(no_vggt.get("stock_condition_source", "")),
        "slat_condition_source": str(no_vggt.get("slat_condition_source", "")),
        "depth_policy": str(no_vggt.get("depth_policy", "")),
        "config_hash": str(getattr(dataset, "config_hash", "")),
        "geometric_preprocessing": preprocessing,
        "geometric_preprocessing_hash": preprocessing_json_sha256(preprocessing),
    }
    patch_side = int(round(math.sqrt(contract["patch_count"])))
    valid_preprocessing = (
        not preprocessing
        or (
            preprocessing.get("version") == SHARED_OBJECT_PREPROCESSING_VERSION
            and preprocessing.get("intrinsics_rule")
            == "K_feature=source_to_feature_affine@K_source"
        )
    )
    if (
        contract["version"] != DINO_ONLY_LIFTING_VERSION
        or contract["visual_feature_dim"] != DINO_FEATURE_DIM
        or contract["vggt_feature_dim"] != 0
        or contract["dino_feature_dim"] != DINO_FEATURE_DIM
        or patch_side * patch_side != contract["patch_count"]
        or contract["context_source"] != "raw_dino_only"
        or contract["vggt_model_executed"] is not False
        or contract["stock_condition_source"] != "deterministic_dino_token_context"
        or contract["slat_condition_source"] != "per_view_raw_dino_token_context"
        or contract["depth_policy"] != "zero_placeholder_not_consumed"
        or not contract["config_hash"]
        or not valid_preprocessing
    ):
        raise ValueError(f"DINO-only lifting contract mismatch={contract}")
    if training_config_hash is not None and contract["config_hash"] != str(
        training_config_hash
    ):
        raise RuntimeError(
            "DINO-only evaluation cache config differs from training: "
            f"{contract['config_hash']} != {training_config_hash}"
        )
    contract["patch_side"] = patch_side
    return contract


__all__ = [
    "DEFAULT_PATCH_COUNT",
    "DEFAULT_SS_CONTEXT_TOKENS",
    "DINO_FEATURE_DIM",
    "DINO_ONLY_CONTEXT_VERSION",
    "DINO_ONLY_LIFTING_VERSION",
    "build_dino_only_contexts",
    "deterministic_token_indices",
    "dino_only_feature_metadata",
    "select_dino_patch_features",
    "tensor_sha256",
    "tensor_tree_sha256",
    "validate_dino_only_lifting_contract",
]
