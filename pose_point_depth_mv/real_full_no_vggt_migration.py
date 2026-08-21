#!/usr/bin/env python3
"""Strict real-Full EMA to mixed no-VGGT checkpoint migration contracts."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import torch

from ar_ss_flow.shared_object_preprocessing import canonical_json_sha256
from pose_point_depth_mv.native_slat_genrecon_v2 import (
    NATIVE_SLAT_GENRECON_V2_VERSION,
)
from pose_point_depth_mv.native_ss_genrecon import NATIVE_SS_GENRECON_VERSION
from pose_point_depth_mv.omni_real_benchmark_common import sha256_file


MIGRATION_CONTRACT_VERSION = "pose_point_depth_mv.real_full_to_no_vggt_migration.v1"
MIGRATION_STAGES = ("ss", "slat")


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))


def _state_schema(state: Any) -> dict[str, Any]:
    if not isinstance(state, dict) or not state:
        raise ValueError("checkpoint trainable state is empty")
    rows = []
    for name, value in sorted(state.items()):
        if not torch.is_tensor(value):
            raise ValueError(f"checkpoint state {name} is not a tensor")
        rows.append(
            {
                "name": str(name),
                "shape": list(value.shape),
                "dtype": str(value.dtype),
            }
        )
    return {
        "parameter_count": len(rows),
        "schema_sha256": canonical_json_sha256(rows),
    }


def _real_lifting_manifest(checkpoint: dict[str, Any], stage: str) -> Path:
    identity = dict(checkpoint.get("data_identity", {}))
    key = "manifest" if stage == "ss" else "lifting_cache_manifest"
    path = Path(str(identity.get(key, ""))).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"parent {stage} real lifting manifest: {path}")
    expected_sha = str(identity.get(f"{key}_sha256", ""))
    if sha256_file(path) != expected_sha:
        raise RuntimeError(f"parent {stage} lifting manifest hash changed")
    return path


def _validate_real_parent_data(
    checkpoint: dict[str, Any], *, stage: str, min_real_objects: int
) -> dict[str, Any]:
    lifting_path = _real_lifting_manifest(checkpoint, stage)
    lifting = _load_json(lifting_path)
    rows = lifting.get("samples")
    if not isinstance(rows, list) or not rows:
        raise ValueError("parent real lifting manifest contains no samples")
    sources = {str(row.get("source", "")) for row in rows}
    if sources != {"omni_real_video"}:
        raise RuntimeError(f"parent checkpoint is not real-only: sources={sources}")
    object_uids = {str(row.get("object_uid", row.get("uid", ""))) for row in rows}
    if len(object_uids) < int(min_real_objects):
        raise RuntimeError(
            f"parent real objects {len(object_uids)} < {int(min_real_objects)}"
        )
    if int(lifting.get("visual_feature_dim", 0)) != 3072:
        raise RuntimeError("parent Full lifting cache must have 3072-D features")
    metadata = dict(lifting.get("feature_metadata", {}))
    if int(metadata.get("vggt_feature_dim", -1)) != 2048 or int(
        metadata.get("dino_feature_dim", -1)
    ) != 1024:
        raise RuntimeError("parent Full lifting feature split differs")
    return {
        "manifest": str(lifting_path),
        "manifest_sha256": sha256_file(lifting_path),
        "sample_count": len(rows),
        "object_count": len(object_uids),
        "source": "omni_real_video",
        "visual_feature_dim": 3072,
        "vggt_feature_dim": 2048,
        "dino_feature_dim": 1024,
    }


def build_migration_contract(
    *,
    stage: str,
    parent_checkpoint: str | Path,
    parent_report: str | Path,
    min_real_objects: int = 350,
) -> dict[str, Any]:
    stage = str(stage)
    if stage not in MIGRATION_STAGES:
        raise ValueError(f"unsupported migration stage={stage!r}")
    checkpoint_path = Path(parent_checkpoint).expanduser().resolve()
    report_path = Path(parent_report).expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    report = _load_json(report_path)
    expected_format = (
        NATIVE_SS_GENRECON_VERSION if stage == "ss" else NATIVE_SLAT_GENRECON_V2_VERSION
    )
    if checkpoint.get("format") != expected_format:
        raise ValueError(
            f"parent {stage} must be Full v2 format={expected_format!r}, "
            f"got {checkpoint.get('format')!r}"
        )
    if report.get("format") != expected_format:
        raise ValueError("parent training report format differs from checkpoint")
    if report.get("completed") is not True or report.get("passed") is not True:
        raise RuntimeError("parent real Full training report did not pass")
    if report.get("evaluation_weights") != "ema":
        raise RuntimeError("parent training report does not freeze EMA evaluation")
    if int(report.get("step", -1)) != int(checkpoint.get("step", -2)):
        raise RuntimeError("parent report/checkpoint step differs")
    report_checkpoint = Path(str(report.get("checkpoint", ""))).expanduser().resolve()
    if report_checkpoint != checkpoint_path:
        raise RuntimeError(
            "strict migration requires the checkpoint directly named by parent report"
        )
    if report.get("data_identity") != checkpoint.get("data_identity"):
        raise RuntimeError("parent report/checkpoint data identity differs")
    raw_schema = _state_schema(checkpoint.get("model_trainable_state"))
    ema_schema = _state_schema(checkpoint.get("ema_trainable_state"))
    if raw_schema != ema_schema:
        raise RuntimeError("parent raw/EMA trainable schemas differ")
    real_data = _validate_real_parent_data(
        checkpoint, stage=stage, min_real_objects=int(min_real_objects)
    )
    parent = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_format": expected_format,
        "checkpoint_step": int(checkpoint["step"]),
        "selected_weights": "ema",
        "report": str(report_path),
        "report_sha256": sha256_file(report_path),
        "report_passed": True,
        "data_identity_sha256": canonical_json_sha256(checkpoint["data_identity"]),
        "trainable_state_schema": ema_schema,
        "real_data": real_data,
    }
    policy = {
        "destination": f"mixed_no_vggt_{stage}",
        "parent_role": "initialization_only",
        "selected_weights": "ema",
        "optimizer_inherited": False,
        "scheduler_inherited": False,
        "step_inherited": False,
        "history_inherited": False,
        "rng_inherited": False,
        "ema_reinitialized_from_selected_weights": True,
        "required_vggt_feature_dim": 0,
        "required_dino_feature_dim": 1024,
        "required_vggt_model_executed": False,
    }
    binding = {"stage": stage, "parent": parent, "policy": policy}
    return {
        "format": MIGRATION_CONTRACT_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        **binding,
        "binding_sha256": canonical_json_sha256(binding),
        "passed": True,
    }


def load_migration_contract(path: str | Path, *, stage: str) -> dict[str, Any]:
    contract_path = Path(path).expanduser().resolve()
    payload = _load_json(contract_path)
    if payload.get("format") != MIGRATION_CONTRACT_VERSION:
        raise ValueError("unsupported real Full migration contract")
    if payload.get("passed") is not True or payload.get("stage") != str(stage):
        raise RuntimeError("migration contract stage/pass binding differs")
    binding = {
        "stage": payload["stage"],
        "parent": payload["parent"],
        "policy": payload["policy"],
    }
    if canonical_json_sha256(binding) != str(payload.get("binding_sha256", "")):
        raise RuntimeError("migration contract binding hash differs")
    parent = dict(payload["parent"])
    checkpoint_path = Path(str(parent["checkpoint"])).resolve()
    report_path = Path(str(parent["report"])).resolve()
    if sha256_file(checkpoint_path) != str(parent["checkpoint_sha256"]):
        raise RuntimeError("migration parent checkpoint changed")
    if sha256_file(report_path) != str(parent["report_sha256"]):
        raise RuntimeError("migration parent report changed")
    return {
        **payload,
        "contract": str(contract_path),
        "contract_sha256": sha256_file(contract_path),
    }


def migration_summary(contract: dict[str, Any]) -> dict[str, Any]:
    parent = dict(contract["parent"])
    return {
        "format": MIGRATION_CONTRACT_VERSION,
        "stage": str(contract["stage"]),
        "contract": str(contract["contract"]),
        "contract_sha256": str(contract["contract_sha256"]),
        "binding_sha256": str(contract["binding_sha256"]),
        "parent_checkpoint": str(parent["checkpoint"]),
        "parent_checkpoint_sha256": str(parent["checkpoint_sha256"]),
        "parent_checkpoint_step": int(parent["checkpoint_step"]),
        "parent_report": str(parent["report"]),
        "parent_report_sha256": str(parent["report_sha256"]),
        "selected_weights": "ema",
        "optimizer_inherited": False,
        "step_inherited": False,
        "rng_inherited": False,
    }


def validate_parent_payload(
    checkpoint: dict[str, Any], contract: dict[str, Any], *, stage: str
) -> None:
    parent = dict(contract["parent"])
    expected_format = (
        NATIVE_SS_GENRECON_VERSION if stage == "ss" else NATIVE_SLAT_GENRECON_V2_VERSION
    )
    if checkpoint.get("format") != expected_format:
        raise ValueError("initialization checkpoint is not the contracted Full parent")
    if int(checkpoint.get("step", -1)) != int(parent["checkpoint_step"]):
        raise RuntimeError("initialization checkpoint step differs from contract")
    if canonical_json_sha256(checkpoint.get("data_identity")) != str(
        parent["data_identity_sha256"]
    ):
        raise RuntimeError("initialization checkpoint data identity differs")
    if _state_schema(checkpoint.get("ema_trainable_state")) != dict(
        parent["trainable_state_schema"]
    ):
        raise RuntimeError("initialization checkpoint EMA schema differs")


def validate_destination_migration(
    checkpoint: dict[str, Any], contract: dict[str, Any]
) -> None:
    actual = dict(checkpoint.get("model_summary", {}).get("migration_contract", {}))
    expected = migration_summary(contract)
    if actual != expected:
        raise RuntimeError("destination checkpoint migration lineage differs")
    initialization = dict(checkpoint.get("model_summary", {}).get("initialization", {}))
    required = {
        "checkpoint_sha256": contract["parent"]["checkpoint_sha256"],
        "weights": "ema",
        "optimizer_inherited": False,
        "ema_reinitialized_from_selected_weights": True,
    }
    mismatch = {
        key: (initialization.get(key), value)
        for key, value in required.items()
        if initialization.get(key) != value
    }
    if mismatch:
        raise RuntimeError(f"destination initialization policy differs: {mismatch}")


__all__ = [
    "MIGRATION_CONTRACT_VERSION",
    "build_migration_contract",
    "load_migration_contract",
    "migration_summary",
    "validate_destination_migration",
    "validate_parent_payload",
]
