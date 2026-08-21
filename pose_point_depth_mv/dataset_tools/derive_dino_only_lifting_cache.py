#!/usr/bin/env python3
"""Derive an immutable no-VGGT lifting cache from an existing Full cache.

The source cache supplies geometry, masks, targets, point priors, and DINO
tokens.  Source VGGT channels, VGGT-derived contexts, and depth predictions are
never copied into the derived samples.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

from ar_ss_flow.local_pose_lifting_flow import parse_indices
from ar_ss_flow.pose_lifting import (
    LIFTING_CACHE_VERSION,
    LIFTING_METADATA_NAMES,
    schema_hash,
)
from ar_ss_flow.shared_object_preprocessing import canonical_json_sha256
from pose_point_depth_mv.dino_only_condition import (
    DEFAULT_SS_CONTEXT_TOKENS,
    DINO_ONLY_LIFTING_VERSION,
    build_dino_only_contexts,
    dino_only_feature_metadata,
    tensor_tree_sha256,
    validate_dino_only_lifting_contract,
)


MANIFEST_FORMAT = LIFTING_CACHE_VERSION
MARKER_FORMAT = "pose_point_depth_mv.dino_only_lifting_marker.v1"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def resolve_cache_file(row: dict[str, Any], root: Path) -> Path:
    path = Path(str(row["cache_file"]))
    return path if path.is_absolute() else root / path


def _finite_geometry(payload: dict[str, Any], uid: str) -> None:
    required = (
        "masks",
        "intrinsics",
        "extrinsics",
        "prior_coords",
        "prior_confidence",
    )
    missing = [name for name in required if not torch.is_tensor(payload.get(name))]
    if missing:
        raise ValueError(f"uid={uid} source sample lacks tensors={missing}")
    finite = ("masks", "intrinsics", "extrinsics", "prior_confidence")
    if not all(
        bool(torch.isfinite(payload[name].float()).all().item()) for name in finite
    ):
        raise ValueError(f"uid={uid} source geometry contains non-finite tensors")


def derive_sample(
    source: dict[str, Any],
    *,
    source_file: Path,
    source_sha256: str,
    output_config_hash: str,
    ss_context_tokens: int,
) -> dict[str, Any]:
    uid = str(source.get("uid", ""))
    if source.get("format") != LIFTING_CACHE_VERSION or not uid:
        raise ValueError(f"invalid source lifting sample: {source_file}")
    _finite_geometry(source, uid)
    contexts = build_dino_only_contexts(
        source["visual_patch_features"], ss_context_tokens=ss_context_tokens
    )
    views = int(contexts["visual_patch_features"].shape[0])
    if source["masks"].ndim != 3 or int(source["masks"].shape[0]) != views:
        raise ValueError(f"uid={uid} source mask/view layout differs")
    height, width = map(int, source["masks"].shape[-2:])
    slat_condition = contexts["slat_condition"]
    condition_identity = str(
        source.get("runtime_condition_sha256", source_sha256)
    )
    output = {
        key: value
        for key, value in source.items()
        if "vggt" not in str(key).lower()
        and key
        not in {
            "visual_patch_features",
            "predicted_depth",
            "depth_confidence",
            "stock_condition",
            "slat_condition",
            "slat_condition_provenance",
        }
    }
    output.update(
        {
            "visual_patch_features": contexts["visual_patch_features"].cpu(),
            "predicted_depth": torch.zeros(
                (views, height, width), dtype=torch.float16
            ),
            "depth_confidence": torch.ones(
                (views, height, width), dtype=torch.float16
            ),
            "stock_condition": contexts["stock_condition"].cpu(),
            "slat_condition": {
                key: [value.cpu() for value in values]
                for key, values in slat_condition.items()
            },
            "runtime_condition_sha256": condition_identity,
            "slat_condition_provenance": {
                "source": "derive_dino_only_lifting_cache.v1",
                "model_input": str(source_file),
                "model_input_sha256": source_sha256,
                "condition_sha256": condition_identity,
                "condition_tree_sha256": tensor_tree_sha256(slat_condition),
                "vggt_model_executed": False,
            },
            "depth_calibration": {
                "enabled": False,
                "reason": "DINO-only frustum projection consumes image shape only",
            },
            "dino_only_context_contract": contexts["context_contract"],
            "dino_only_derivation": {
                "version": DINO_ONLY_LIFTING_VERSION,
                "source_sample": str(source_file),
                "source_sample_sha256": source_sha256,
                "output_config_hash": output_config_hash,
                "vggt_model_executed": False,
            },
        }
    )
    preprocessing = dict(output.get("preprocessing", {}))
    if preprocessing:
        preprocessing.update(
            {
                "stock_condition": "deterministic raw DINO token context",
                "lifting_features": "DINO-only patches on shared geometry",
                "depth_policy": "zero_placeholder_not_consumed",
            }
        )
        output["preprocessing"] = preprocessing
    return output


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--indices", default="all")
    parser.add_argument(
        "--ss_context_tokens", type=int, default=DEFAULT_SS_CONTEXT_TOKENS
    )
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> None:
    args = make_parser().parse_args()
    source_path = Path(args.source_manifest).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if int(args.ss_context_tokens) <= 0:
        raise ValueError("ss_context_tokens must be positive")
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if source.get("format") != LIFTING_CACHE_VERSION:
        raise ValueError(f"unsupported source lifting format={source.get('format')!r}")
    if tuple(source.get("metadata_names", ())) != LIFTING_METADATA_NAMES:
        raise ValueError("source lifting metadata names differ")
    if source.get("metadata_schema_hash") != schema_hash():
        raise ValueError("source lifting metadata schema hash differs")
    rows = source.get("samples")
    if not isinstance(rows, list) or not rows:
        raise ValueError("source lifting manifest has no samples")
    selected = parse_indices(args.indices, len(rows))
    if not selected or len(selected) != len(set(selected)):
        raise ValueError("selected source indices must be non-empty and unique")
    source_root = Path(source.get("output_dir", source_path.parent)).resolve()
    if output_dir == source_root:
        raise ValueError("DINO-only output_dir must differ from the source cache")
    source_sha = sha256_file(source_path)
    source_config = dict(source.get("config", {}))
    no_vggt = {
        "version": DINO_ONLY_LIFTING_VERSION,
        "stock_condition_source": "deterministic_dino_token_context",
        "slat_condition_source": "per_view_raw_dino_token_context",
        "ss_context_token_cap": int(args.ss_context_tokens),
        "depth_policy": "zero_placeholder_not_consumed",
        "vggt_model_executed": False,
    }
    config = {**source_config, "no_vggt": no_vggt}
    config_hash = canonical_json_sha256(config)
    run_binding = {
        "format": MARKER_FORMAT,
        "source_manifest": str(source_path),
        "source_manifest_sha256": source_sha,
        "selected_indices": selected,
        "config": config,
        "config_hash": config_hash,
    }
    binding_path = output_dir / "run_config.json"
    if output_dir.exists() and not args.resume:
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if binding_path.is_file():
        existing = json.loads(binding_path.read_text(encoding="utf-8"))
        if existing != run_binding:
            raise RuntimeError("resume source/config binding changed")
    else:
        atomic_json(binding_path, run_binding)

    output_rows: list[dict[str, Any]] = []
    patch_counts: set[int] = set()
    for position, source_index in enumerate(selected, start=1):
        row = dict(rows[source_index])
        uid = str(row.get("uid", ""))
        if not uid:
            raise ValueError(f"source row index={source_index} has empty uid")
        source_file = resolve_cache_file(row, source_root).resolve()
        source_sample_sha = sha256_file(source_file)
        relative = Path("samples") / uid[:2] / f"{uid}.pt"
        destination = output_dir / relative
        if destination.is_file():
            payload = torch.load(destination, map_location="cpu")
            derivation = dict(payload.get("dino_only_derivation", {}))
            if (
                str(payload.get("uid", "")) != uid
                or derivation.get("source_sample_sha256") != source_sample_sha
                or derivation.get("output_config_hash") != config_hash
            ):
                raise RuntimeError(f"stale derived sample: {destination}")
        else:
            source_payload = torch.load(source_file, map_location="cpu")
            payload = derive_sample(
                source_payload,
                source_file=source_file,
                source_sha256=source_sample_sha,
                output_config_hash=config_hash,
                ss_context_tokens=int(args.ss_context_tokens),
            )
            atomic_torch_save(destination, payload)
        provenance = dict(payload.get("slat_condition_provenance", {}))
        actual_condition_hash = tensor_tree_sha256(payload.get("slat_condition"))
        if provenance.get("condition_tree_sha256") != actual_condition_hash:
            raise RuntimeError(
                f"derived SLat condition hash mismatch: {destination}"
            )
        patch_counts.add(int(payload["visual_patch_features"].shape[1]))
        output_row = {
            **row,
            "cache_file": str(relative),
            "cache_file_sha256": sha256_file(destination),
            "source_cache_file": str(source_file),
            "source_cache_file_sha256": source_sample_sha,
        }
        output_rows.append(output_row)
        print(f"[dino_only_cache] {position}/{len(selected)} uid={uid}", flush=True)
    if len(patch_counts) != 1:
        raise RuntimeError(f"derived samples have mixed patch counts={sorted(patch_counts)}")
    patch_count = next(iter(patch_counts))
    manifest = {
        **{
            key: value
            for key, value in source.items()
            if key
            not in {
                "output_dir",
                "source_cache_manifest",
                "stock_condition_source",
                "lifting_feature_source",
                "sample_count",
                "object_count",
                "failure_count",
                "feature_metadata",
                "visual_feature_dim",
                "config",
                "config_hash",
                "samples",
            }
        },
        "format": MANIFEST_FORMAT,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(output_dir),
        "source_cache_manifest": str(source_path),
        "source_cache_manifest_sha256": source_sha,
        "stock_condition_source": "deterministic DINO-only token context",
        "lifting_feature_source": "DINO-only trailing channels from immutable source",
        "sample_count": len(output_rows),
        "object_count": len(
            {str(row.get("object_uid", row["uid"])) for row in output_rows}
        ),
        "failure_count": 0,
        "feature_metadata": dino_only_feature_metadata(patch_count=patch_count),
        "visual_feature_dim": 1024,
        "metadata_names": list(LIFTING_METADATA_NAMES),
        "metadata_schema_hash": schema_hash(),
        "config": config,
        "config_hash": config_hash,
        "samples": output_rows,
        "passed": True,
        "training_ready": True,
    }
    proxy = SimpleNamespace(
        visual_feature_dim=manifest["visual_feature_dim"],
        feature_metadata=manifest["feature_metadata"],
        config=manifest["config"],
        config_hash=manifest["config_hash"],
    )
    manifest["no_vggt_contract"] = validate_dino_only_lifting_contract(proxy)
    manifest_path = output_dir / "lifting_manifest.json"
    atomic_json(manifest_path, manifest)
    marker = {
        "format": MARKER_FORMAT,
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "source_manifest_sha256": source_sha,
        "sample_count": len(output_rows),
        "config_hash": config_hash,
        "passed": True,
    }
    atomic_json(output_dir / "_DINO_ONLY_COMPLETE.json", marker)
    print(json.dumps(marker, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
