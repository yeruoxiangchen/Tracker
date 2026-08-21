#!/usr/bin/env python3
"""Build immutable meta-manifests for mixed synthetic/real no-VGGT training."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any

from ar_ss_flow.local_pose_lifting_flow import PoseLiftingCacheDataset
from ar_ss_flow.shared_object_preprocessing import canonical_json_sha256
from pose_point_depth_mv.dino_only_condition import validate_dino_only_lifting_contract
from pose_point_depth_mv.mixed_no_vggt_data import (
    DOMAIN_BALANCED_SAMPLER_VERSION,
    MIXED_LIFTING_MANIFEST_VERSION,
    MIXED_SLAT_MANIFEST_VERSION,
    MixedNativeConditionSLatDataset,
    MixedPoseLiftingCacheDataset,
)
from pose_point_depth_mv.omni_real_benchmark_common import sha256_file


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def domain_lifting_binding(name: str, manifest: str | Path) -> dict[str, Any]:
    path = Path(manifest).expanduser().resolve()
    dataset = PoseLiftingCacheDataset(path, indices="all")
    contract = validate_dino_only_lifting_contract(dataset)
    return {
        "name": str(name),
        "weight": 1,
        "manifest": str(path),
        "manifest_sha256": sha256_file(path),
        "sample_count": len(dataset.rows),
        "object_count": len(
            {str(row.get("object_uid", row["uid"])) for row in dataset.rows}
        ),
        "config_hash": dataset.config_hash,
        "no_vggt_contract": contract,
    }


def write_once(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        current = json.loads(path.read_text(encoding="utf-8"))
        stable_current = dict(current)
        stable_payload = dict(payload)
        stable_current.pop("created_at_utc", None)
        stable_payload.pop("created_at_utc", None)
        if stable_current != stable_payload:
            raise RuntimeError(f"refusing to overwrite changed manifest: {path}")
        return
    atomic_json(path, payload)


def build_lifting(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output).expanduser().resolve()
    domains = [
        domain_lifting_binding("synthetic", args.synthetic_manifest),
        domain_lifting_binding("real", args.real_manifest),
    ]
    expected = {
        "synthetic": int(args.expected_synthetic_objects),
        "real": int(args.expected_real_objects),
    }
    for domain in domains:
        minimum = expected[str(domain["name"])]
        if int(domain["object_count"]) < minimum:
            raise RuntimeError(
                f"{domain['name']} object count {domain['object_count']} < {minimum}"
            )
    config = {
        "version": MIXED_LIFTING_MANIFEST_VERSION,
        "domain_order": ["synthetic", "real"],
        "domain_weights": {"synthetic": 1, "real": 1},
        "sampler": {
            "version": DOMAIN_BALANCED_SAMPLER_VERSION,
            "policy": "equal_domain_object_cycles",
            "ratio": {"synthetic": 1, "real": 1},
        },
        "component_config_hashes": {
            str(domain["name"]): str(domain["config_hash"]) for domain in domains
        },
        "no_vggt": {
            key: domains[0]["no_vggt_contract"][key]
            for key in (
                "version",
                "visual_feature_dim",
                "vggt_feature_dim",
                "dino_feature_dim",
                "patch_count",
                "context_source",
                "vggt_model_executed",
                "stock_condition_source",
                "slat_condition_source",
                "depth_policy",
            )
        },
    }
    payload = {
        "format": MIXED_LIFTING_MANIFEST_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "domains": domains,
        "sample_count": sum(int(domain["sample_count"]) for domain in domains),
        "object_count": sum(int(domain["object_count"]) for domain in domains),
        "config": config,
        "config_hash": canonical_json_sha256(config),
        "physical_samples_copied": 0,
        "passed": True,
        "training_ready": True,
    }
    write_once(output, payload)
    dataset = MixedPoseLiftingCacheDataset(output)
    if len(dataset) != int(payload["sample_count"]):
        raise RuntimeError("written mixed lifting sample count differs")
    return payload


def build_slat(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output).expanduser().resolve()
    lifting = Path(args.lifting_manifest).expanduser().resolve()
    lifting_payload = json.loads(lifting.read_text(encoding="utf-8"))
    if lifting_payload.get("format") != MIXED_LIFTING_MANIFEST_VERSION:
        raise ValueError("--lifting_manifest is not a mixed no-VGGT lifting manifest")
    domains = []
    component_config_hashes: dict[str, str] = {}
    component_sample_counts: dict[str, int] = {}
    component_object_counts: dict[str, int] = {}
    for name, value in (
        ("synthetic", args.synthetic_manifest),
        ("real", args.real_manifest),
    ):
        path = Path(value).expanduser().resolve()
        component = json.loads(path.read_text(encoding="utf-8"))
        component_config_hash = str(component.get("config_hash", ""))
        if not component_config_hash:
            raise ValueError(f"{name} SLat manifest lacks config_hash")
        component_config_hashes[name] = component_config_hash
        component_sample_counts[name] = int(component.get("sample_count", 0))
        component_object_counts[name] = int(component.get("object_count", 0))
        if component_sample_counts[name] <= 0 or component_object_counts[name] <= 0:
            raise ValueError(f"{name} SLat manifest has invalid counts")
        domains.append(
            {
                "name": name,
                "weight": 1,
                "manifest": str(path),
                "manifest_sha256": sha256_file(path),
            }
        )
    config = {
        "version": MIXED_SLAT_MANIFEST_VERSION,
        "domains": ["synthetic", "real"],
        "ratio": {"synthetic": 1, "real": 1},
        "component_config_hashes": component_config_hashes,
    }
    payload = {
        "format": MIXED_SLAT_MANIFEST_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "lifting_manifest": str(lifting),
        "lifting_manifest_sha256": sha256_file(lifting),
        "domains": domains,
        "config_hash": canonical_json_sha256(config),
        "sample_count": sum(component_sample_counts.values()),
        "object_count": sum(component_object_counts.values()),
        "physical_samples_copied": 0,
        "passed": True,
        "training_ready": True,
    }
    write_once(output, payload)
    dataset = MixedNativeConditionSLatDataset(
        output, lifting, indices="all", verify_hashes=bool(args.verify_hashes)
    )
    if len(dataset) != int(payload["sample_count"]):
        raise RuntimeError("written mixed SLat sample count differs")
    return payload


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    lifting = subparsers.add_parser("lifting")
    lifting.add_argument("--synthetic_manifest", required=True)
    lifting.add_argument("--real_manifest", required=True)
    lifting.add_argument("--output", required=True)
    lifting.add_argument("--expected_synthetic_objects", type=int, default=868)
    lifting.add_argument("--expected_real_objects", type=int, default=350)
    lifting.set_defaults(handler=build_lifting)
    slat = subparsers.add_parser("slat")
    slat.add_argument("--synthetic_manifest", required=True)
    slat.add_argument("--real_manifest", required=True)
    slat.add_argument("--lifting_manifest", required=True)
    slat.add_argument("--output", required=True)
    slat.add_argument("--verify_hashes", action="store_true")
    slat.set_defaults(handler=build_slat)
    return parser


def main() -> None:
    args = make_parser().parse_args()
    payload = args.handler(args)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
