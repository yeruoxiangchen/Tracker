#!/usr/bin/env python3
"""Read-only contract preflight for official no-VGGT Native-SLat resume."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import torch

from ar_ss_flow.local_pose_lifting_flow import PoseLiftingCacheDataset
from pose_aligned_reconstruction.dino_only_condition import validate_dino_only_lifting_contract
from pose_aligned_reconstruction.native_3d_condition import NativeConditionSLatDataset
from pose_aligned_reconstruction.native_slat_genrecon import (
    canonical_json_sha256,
    load_stock_slat_freeze,
    sha256_file,
)
from pose_aligned_reconstruction.native_slat_genrecon_no_vggt import (
    validate_native_slat_no_vggt_checkpoint,
)
from pose_aligned_reconstruction.no_vggt_ss_evidence import load_no_vggt_ss_evidence
from pose_aligned_reconstruction.proobjaverse_official_slat_training import (
    validate_official_decoder_audit,
)
from pose_aligned_reconstruction.train_native_slat_genrecon import (
    validate_native_ss_deployment,
    validate_resume_data_identity,
    validate_resume_training_contract,
)
from pose_aligned_reconstruction.train_native_slat_genrecon_no_vggt import (
    no_vggt_upstream_binding,
)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache_manifest", required=True)
    parser.add_argument("--lifting_cache_manifest", required=True)
    parser.add_argument("--target_decoder_audit", required=True)
    parser.add_argument("--native_ss_report", required=True)
    parser.add_argument("--stock_slat_freeze", required=True)
    parser.add_argument("--resume", required=True)
    parser.add_argument("--max_steps", required=True, type=int)
    parser.add_argument("--world_size", required=True, type=int)
    parser.add_argument("--grad_accum", required=True, type=int)
    parser.add_argument("--pretrained", default="Stable-X/trellis-vggt-v0-2")
    parser.add_argument("--allow_resume_max_steps_extension", action="store_true")
    parser.add_argument("--allow_resume_topology_change", action="store_true")
    parser.add_argument("--allow_resume_data_path_relocation", action="store_true")
    return parser


def _load_checkpoint(path: str | Path) -> tuple[Path, dict[str, Any]]:
    checkpoint_path = Path(path).expanduser().resolve(strict=True)
    try:
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False, mmap=True
        )
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError("resume checkpoint payload is not a dictionary")
    return checkpoint_path, checkpoint


def run_preflight(args: argparse.Namespace) -> dict[str, Any]:
    if int(args.max_steps) <= 0 or int(args.world_size) <= 0 or int(args.grad_accum) <= 0:
        raise ValueError("max_steps/world_size/grad_accum must be positive")

    lifting = PoseLiftingCacheDataset(args.lifting_cache_manifest, indices="all")
    lifting_contract = validate_dino_only_lifting_contract(lifting)

    dataset = NativeConditionSLatDataset(
        args.cache_manifest,
        args.lifting_cache_manifest,
        indices="all",
        verify_hashes=False,
    )
    if dataset.config.get("condition_arch") != "native_ss_genrecon_v2":
        raise RuntimeError(
            "official no-VGGT Native-SLat requires native_ss_genrecon_v2 cache"
        )

    _, runtime_deployment = load_no_vggt_ss_evidence(args.native_ss_report)
    deployment_transition = validate_native_ss_deployment(
        dataset.config.get("native_ss_deployment"),
        runtime_deployment,
        allow_path_relocation=bool(args.allow_resume_data_path_relocation),
    )
    current_upstream = no_vggt_upstream_binding(runtime_deployment)

    stock_freeze = load_stock_slat_freeze(args.stock_slat_freeze)
    decoder_audit = validate_official_decoder_audit(
        args.target_decoder_audit,
        cache_config=dataset.config,
        pretrained=args.pretrained,
    )

    object_uids = sorted({str(row["object_uid"]) for row in dataset.rows})
    current_data_identity = {
        "cache_manifest": str(Path(args.cache_manifest).expanduser().resolve()),
        "cache_manifest_sha256": sha256_file(args.cache_manifest),
        "lifting_cache_manifest": str(
            Path(args.lifting_cache_manifest).expanduser().resolve()
        ),
        "lifting_cache_manifest_sha256": sha256_file(args.lifting_cache_manifest),
        "config_hash": dataset.config_hash,
        "sample_count": len(dataset),
        "object_count": len(object_uids),
        "object_uids": object_uids,
        "object_uid_hash": canonical_json_sha256(object_uids),
        "native_ss": current_upstream,
        "stock_slat_freeze_sha256": stock_freeze["freeze_sha256"],
        "target_decoder_audit": decoder_audit,
    }

    checkpoint_path, checkpoint = _load_checkpoint(args.resume)
    saved_upstream = checkpoint.get("data_identity", {}).get("native_ss")
    if not isinstance(saved_upstream, dict):
        raise ValueError("resume checkpoint lacks Native SS data identity")
    validate_native_slat_no_vggt_checkpoint(
        checkpoint,
        pretrained=args.pretrained,
        stock_slat_freeze=stock_freeze,
        upstream_native_ss=saved_upstream,
        allow_v2_parent=False,
    )
    identity_transition = validate_resume_data_identity(
        checkpoint.get("data_identity"),
        current_data_identity,
        allow_path_relocation=bool(args.allow_resume_data_path_relocation),
    )

    resume_args = argparse.Namespace(
        **{
            **dict(checkpoint["args"]),
            "resume": str(checkpoint_path),
            "max_steps": int(args.max_steps),
            "grad_accum": int(args.grad_accum),
            "allow_resume_max_steps_extension": bool(
                args.allow_resume_max_steps_extension
            ),
            "allow_resume_topology_change": bool(args.allow_resume_topology_change),
            "allow_resume_data_path_relocation": bool(
                args.allow_resume_data_path_relocation
            ),
        }
    )
    training_transition = validate_resume_training_contract(
        checkpoint, resume_args, world_size=int(args.world_size)
    )
    if (
        training_transition["saved_global_effective_batch"]
        != training_transition["current_global_effective_batch"]
    ):
        raise RuntimeError("resume global effective batch differs")

    return {
        "passed": True,
        "lifting_contract": {"passed": True, **lifting_contract},
        "dataset_contract": {
            "passed": True,
            "sample_count": len(dataset),
            "object_count": len(object_uids),
            "config_hash": dataset.config_hash,
        },
        "native_ss_deployment": {"passed": True, **deployment_transition},
        "stock_slat_freeze": {
            "passed": True,
            "freeze_sha256": stock_freeze["freeze_sha256"],
            "file_sha256": stock_freeze["file_sha256"],
        },
        "official_decoder_audit": {"passed": True, **decoder_audit},
        "resume_checkpoint": {
            "passed": True,
            "path": str(checkpoint_path),
            "sha256": sha256_file(checkpoint_path),
            "step": int(checkpoint["step"]),
        },
        "resume_data_identity": {
            "passed": True,
            "path_relocated": bool(identity_transition["applied"]),
            "relocations": identity_transition["approved_fields"],
            "all_non_path_fields_exact": identity_transition[
                "all_non_path_fields_exact"
            ],
        },
        "resume_training_contract": training_transition,
    }


def main() -> int:
    args = make_parser().parse_args()
    try:
        result = run_preflight(args)
    except Exception as error:
        print(
            json.dumps(
                {"passed": False, "error_type": type(error).__name__, "error": str(error)},
                indent=2,
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 1
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
