#!/usr/bin/env python3
"""Contracts for Native-SS training on official ProObjaverse SLat support."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ar_ss_flow.local_pose_lifting_flow import resolve_ss_latent_path
from pose_aligned_reconstruction.native_ss_genrecon import canonical_json_sha256, sha256_file
from pose_aligned_reconstruction.native_ss_genrecon_no_vggt import (
    validate_no_vggt_cache_contract,
)


OFFICIAL_SS_TARGET_FORMAT = "pose_point_depth_mv.proobjaverse_official_ss_target.v1"
OFFICIAL_SS_CACHE_FORMAT = (
    "pose_point_depth_mv.proobjaverse_official_ss_training_cache.v1"
)
OFFICIAL_SS_DOMAIN_VERSION = (
    "pose_point_depth_mv.proobjaverse_official_ss_domain_contract.v1"
)
OFFICIAL_SS_CALIBRATION = (
    "pose_point_depth_mv.proobjaverse_official_native_ss_calibration.v1"
)
OFFICIAL_SS_EVAL = "pose_point_depth_mv.proobjaverse_official_native_ss_eval.v1"
OFFICIAL_SS_EVAL_AGGREGATE = (
    "pose_point_depth_mv.proobjaverse_official_native_ss_eval_aggregate.v1"
)


def official_domain_contract(
    *,
    protocol_sha256: str,
    encoder_pretrained: str,
    decoder_pretrained: str,
    latent_dtype: str,
    minimum_roundtrip_iou: float,
) -> dict[str, Any]:
    contract = {
        "version": OFFICIAL_SS_DOMAIN_VERSION,
        "source": "Stable-X/ProObjaverse-300K official lh-slats",
        "official_slat_protocol_sha256": str(protocol_sha256),
        "coordinate_resolution": 64,
        "coordinate_frame": "official_lh_slat_64",
        "target_mode": "decoder_projected",
        "ss_encoder_pretrained": str(encoder_pretrained),
        "ss_decoder_pretrained": str(decoder_pretrained),
        "latent_dtype": str(latent_dtype),
        "minimum_roundtrip_iou": float(minimum_roundtrip_iou),
        "official_coords_are_encoder_input": True,
        "decoded_coords_are_flow_target": True,
    }
    contract["domain_contract_sha256"] = canonical_json_sha256(contract)
    return contract


def validate_official_ss_domain_contract(contract: dict[str, Any]) -> None:
    expected_hash = str(contract.get("domain_contract_sha256", ""))
    body = dict(contract)
    body.pop("domain_contract_sha256", None)
    if (
        body.get("version") != OFFICIAL_SS_DOMAIN_VERSION
        or int(body.get("coordinate_resolution", 0)) != 64
        or body.get("coordinate_frame") != "official_lh_slat_64"
        or body.get("target_mode") != "decoder_projected"
        or body.get("official_coords_are_encoder_input") is not True
        or body.get("decoded_coords_are_flow_target") is not True
        or not str(body.get("official_slat_protocol_sha256", ""))
        or not str(body.get("ss_encoder_pretrained", ""))
        or not str(body.get("ss_decoder_pretrained", ""))
        or str(body.get("latent_dtype")) not in {"float16", "float32"}
        or not 0.0 <= float(body.get("minimum_roundtrip_iou", -1.0)) <= 1.0
        or expected_hash != canonical_json_sha256(body)
    ):
        raise ValueError(f"invalid official SS domain contract={contract}")


def validate_official_ss_cache_contract(dataset: Any) -> dict[str, Any]:
    """Validate that every lifting row is bound to a real audited SS target."""

    base = validate_no_vggt_cache_contract(dataset)
    config = dict(getattr(dataset, "config", {}))
    binding = config.get("official_ss_targets")
    if not isinstance(binding, dict):
        raise RuntimeError("official SS cache lacks official_ss_targets binding")
    domain = binding.get("domain_contract")
    if not isinstance(domain, dict):
        raise RuntimeError("official SS cache lacks domain contract")
    validate_official_ss_domain_contract(domain)
    rows = list(getattr(dataset, "rows", ()))
    if int(binding.get("object_count", -1)) != len(rows) or not rows:
        raise RuntimeError("official SS target count differs from lifting rows")
    minimum = float(domain["minimum_roundtrip_iou"])
    observed_ious: list[float] = []
    seen: set[str] = set()
    root = Path(getattr(dataset, "root", "."))
    for row in rows:
        uid = str(row.get("uid", ""))
        if not uid or uid in seen:
            raise RuntimeError("official SS rows contain empty/duplicate uid")
        seen.add(uid)
        target_path = resolve_ss_latent_path(row, {}, root)
        expected_sha = str(row.get("ss_latent_sha256", ""))
        if not target_path.is_file() or not expected_sha:
            raise RuntimeError(f"uid={uid} official SS target is missing")
        if sha256_file(target_path) != expected_sha:
            raise RuntimeError(f"uid={uid} official SS target hash differs")
        with np.load(target_path, allow_pickle=False) as payload:
            required = {
                "z",
                "target_coords",
                "official_coords",
                "uid",
                "format",
                "domain_contract_sha256",
                "roundtrip_iou",
            }
            missing = sorted(required.difference(payload.files))
            if missing:
                raise RuntimeError(f"uid={uid} official SS target lacks {missing}")
            z = np.asarray(payload["z"])
            coords = np.asarray(payload["target_coords"])
            official = np.asarray(payload["official_coords"])
            target_uid = str(np.asarray(payload["uid"]).item())
            target_format = str(np.asarray(payload["format"]).item())
            domain_hash = str(
                np.asarray(payload["domain_contract_sha256"]).item()
            )
            roundtrip_iou = float(np.asarray(payload["roundtrip_iou"]).item())
        if (
            target_uid != uid
            or target_format != OFFICIAL_SS_TARGET_FORMAT
            or domain_hash != str(domain["domain_contract_sha256"])
            or z.shape not in {(8, 16, 16, 16), (1, 8, 16, 16, 16)}
            or coords.ndim != 2
            or coords.shape[1] != 3
            or official.ndim != 2
            or official.shape[1] != 3
            or len(coords) == 0
            or len(official) == 0
            or not np.isfinite(z.astype(np.float32)).all()
            or roundtrip_iou < minimum
        ):
            raise RuntimeError(f"uid={uid} invalid official SS target contract")
        observed_ious.append(roundtrip_iou)
    return {
        **base,
        "official_ss_targets": {
            **binding,
            "validated_object_count": len(rows),
            "observed_roundtrip_iou_min": min(observed_ious),
            "observed_roundtrip_iou_mean": float(np.mean(observed_ious)),
        },
    }


def validate_official_ss_evaluation_cache_contract(
    dataset: Any, *, training_identity: dict[str, Any]
) -> dict[str, Any]:
    """Bind a held-out official split to the train split's domain contract."""

    observed = validate_official_ss_cache_contract(dataset)
    feature_contract = training_identity.get("feature_contract")
    if not isinstance(feature_contract, dict):
        raise RuntimeError("checkpoint lacks Native SS feature contract")
    trained = feature_contract.get("official_ss_targets")
    if not isinstance(trained, dict):
        raise RuntimeError("checkpoint was not trained on official SS targets")
    observed_domain = observed["official_ss_targets"].get("domain_contract")
    trained_domain = trained.get("domain_contract")
    if not isinstance(trained_domain, dict) or observed_domain != trained_domain:
        raise RuntimeError("official SS train/evaluation domain contracts differ")
    train_split = str(trained.get("split", ""))
    eval_split = str(observed["official_ss_targets"].get("split", ""))
    if train_split != "train" or eval_split == "train":
        raise RuntimeError(
            f"official SS evaluation requires train -> held-out split, got "
            f"{train_split!r} -> {eval_split!r}"
        )
    observed["evaluation_training_binding"] = {
        "mode": "official_domain_contract",
        "training_config_hash": str(training_identity.get("config_hash", "")),
        "evaluation_config_hash": str(getattr(dataset, "config_hash", "")),
        "domain_contract_sha256": observed_domain["domain_contract_sha256"],
        "training_split": train_split,
        "evaluation_split": eval_split,
    }
    return observed


def load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def load_official_native_ss_deployment(
    path: str | Path,
    *,
    require_science_passed: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load a held-out aggregate as a content-bound downstream deployment.

    Existing production callers retain the strict default. Registered
    checkpoint-comparison evaluations may set ``require_science_passed=False``
    so a scientifically negative candidate is still evaluated downstream,
    without rewriting failed gates as passes.
    """

    report_path = Path(path).expanduser().resolve(strict=True)
    payload = load_json(report_path)
    body = dict(payload)
    expected_report_hash = str(body.pop("report_sha256", ""))
    if (
        payload.get("format") != OFFICIAL_SS_EVAL_AGGREGATE
        or (
            bool(require_science_passed)
            and payload.get("passed") is not True
        )
        or payload.get("formal") is not False
        or not expected_report_hash
        or canonical_json_sha256(body) != expected_report_hash
    ):
        raise ValueError("official Native SS aggregate identity/integrity differs")
    checks = payload.get("checks")
    deployment = payload.get("deployment")
    object_uids = payload.get("object_uids")
    domain = payload.get("official_ss_domain_contract")
    if (
        not isinstance(checks, dict)
        or not checks
        or any(not isinstance(value, bool) for value in checks.values())
        or (
            bool(require_science_passed)
            and any(value is not True for value in checks.values())
        )
        or not isinstance(deployment, dict)
        or not isinstance(domain, dict)
        or not isinstance(object_uids, list)
        or len(object_uids) != int(payload.get("object_count", -1))
        or len(object_uids) != len(set(str(value) for value in object_uids))
        or canonical_json_sha256(sorted(str(value) for value in object_uids))
        != str(payload.get("object_uid_hash", ""))
    ):
        raise RuntimeError("official Native SS aggregate deployment contract differs")
    validate_official_ss_domain_contract(domain)
    required = {
        "checkpoint",
        "checkpoint_sha256",
        "checkpoint_step",
        "weights",
        "cfg_strength",
        "steps",
        "cfg_interval",
        "guidance_rescale",
        "rescale_t",
        "amp_dtype",
    }
    if set(deployment) != required:
        raise ValueError(
            "official Native SS deployment fields differ: "
            f"missing={sorted(required - set(deployment))} "
            f"unexpected={sorted(set(deployment) - required)}"
        )
    checkpoint = Path(str(deployment["checkpoint"])).expanduser().resolve(strict=True)
    if sha256_file(checkpoint) != str(deployment["checkpoint_sha256"]):
        raise RuntimeError("official Native SS deployment checkpoint hash differs")
    cfg_interval = [float(value) for value in deployment["cfg_interval"]]
    if len(cfg_interval) != 2 or not 0.0 <= cfg_interval[0] <= cfg_interval[1] <= 1.0:
        raise ValueError("official Native SS deployment CFG interval is invalid")
    binding = {
        "report": str(report_path),
        "report_sha256": sha256_file(report_path),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": str(deployment["checkpoint_sha256"]),
        "checkpoint_step": int(deployment["checkpoint_step"]),
        "weights": str(deployment["weights"]),
        "cfg_strength": float(deployment["cfg_strength"]),
        "steps": int(deployment["steps"]),
        "cfg_interval": cfg_interval,
        "guidance_rescale": float(deployment["guidance_rescale"]),
        "rescale_t": float(deployment["rescale_t"]),
        "amp_dtype": str(deployment["amp_dtype"]),
        "false_checks": sorted(
            str(key) for key, value in checks.items() if value is not True
        ),
    }
    if (
        binding["checkpoint_step"] <= 0
        or binding["steps"] <= 0
        or binding["weights"] != "ema"
        or binding["cfg_strength"] <= 0.0
        or binding["guidance_rescale"] != 0.0
        or binding["rescale_t"] <= 0.0
        or binding["amp_dtype"] != "bf16"
    ):
        raise ValueError("official Native SS deployment semantics are invalid")
    return payload, binding


__all__ = [
    "OFFICIAL_SS_CACHE_FORMAT",
    "OFFICIAL_SS_CALIBRATION",
    "OFFICIAL_SS_DOMAIN_VERSION",
    "OFFICIAL_SS_EVAL",
    "OFFICIAL_SS_EVAL_AGGREGATE",
    "OFFICIAL_SS_TARGET_FORMAT",
    "load_json",
    "load_official_native_ss_deployment",
    "official_domain_contract",
    "validate_official_ss_domain_contract",
    "validate_official_ss_cache_contract",
    "validate_official_ss_evaluation_cache_contract",
]
