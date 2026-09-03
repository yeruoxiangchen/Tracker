#!/usr/bin/env python3
"""Prove runtime equivalence of official SLat cache v1 and compact v2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from pose_aligned_reconstruction.native_3d_condition import NativeConditionSLatDataset
from pose_aligned_reconstruction.native_slat_genrecon import project_sparse_frustum_dino
from pose_aligned_reconstruction.proobjaverse_official_slat_protocol import (
    atomic_json,
    canonical_sha256,
    sha256_file,
)


AUDIT_FORMAT = "pose_point_depth_mv.proobjaverse_official_slat_cache_v1_v2_audit.v1"


def _require_tensor_equal(label: str, left: torch.Tensor, right: torch.Tensor) -> None:
    if left.dtype != right.dtype or left.shape != right.shape or not torch.equal(left, right):
        difference = None
        if left.shape == right.shape and left.numel() and left.is_floating_point():
            difference = float((left.float() - right.float()).abs().amax().item())
        raise RuntimeError(
            f"{label} differs: left={left.dtype}/{tuple(left.shape)} "
            f"right={right.dtype}/{tuple(right.shape)} max_abs={difference}"
        )


def _require_tree_equal(label: str, left: Any, right: Any) -> None:
    if torch.is_tensor(left) and torch.is_tensor(right):
        _require_tensor_equal(label, left, right)
        return
    if isinstance(left, dict) and isinstance(right, dict):
        if set(left) != set(right):
            raise RuntimeError(f"{label} dictionary keys differ")
        for key in sorted(left):
            _require_tree_equal(f"{label}.{key}", left[key], right[key])
        return
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        if len(left) != len(right):
            raise RuntimeError(f"{label} sequence lengths differ")
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            _require_tree_equal(f"{label}[{index}]", left_item, right_item)
        return
    if left != right:
        raise RuntimeError(f"{label} differs: {left!r} != {right!r}")


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slat_manifest_v1", required=True)
    parser.add_argument("--lifting_manifest_v1", required=True)
    parser.add_argument("--slat_manifest_v2", required=True)
    parser.add_argument("--lifting_manifest_v2", required=True)
    parser.add_argument("--indices", default="all")
    parser.add_argument("--output", required=True)
    return parser


def main() -> None:
    args = make_parser().parse_args()
    v1 = NativeConditionSLatDataset(
        args.slat_manifest_v1, args.lifting_manifest_v1, indices=args.indices
    )
    v2 = NativeConditionSLatDataset(
        args.slat_manifest_v2, args.lifting_manifest_v2, indices=args.indices
    )
    if len(v1) != len(v2):
        raise RuntimeError(f"v1/v2 sample count differs: {len(v1)} != {len(v2)}")
    if v1.config != v2.config or v1.config_hash != v2.config_hash:
        raise RuntimeError("v1/v2 SLat config differs")
    if (
        v1.slat_normalization != v2.slat_normalization
        or v1.slat_normalization_hash != v2.slat_normalization_hash
    ):
        raise RuntimeError("v1/v2 SLat normalization differs")
    if v1.lifting.config != v2.lifting.config:
        raise RuntimeError("v1/v2 lifting config differs")
    records: list[dict[str, Any]] = []
    for index in range(len(v1)):
        left = v1[index]
        right = v2[index]
        uid = str(left["uid"])
        if uid != str(right["uid"]):
            raise RuntimeError(f"v1/v2 UID differs at index={index}")
        for key in (
            "target_coords",
            "target_feats",
            "corrected_ss",
            "occupancy_logits64",
            "corrected_coords64",
            "physical_tokens16",
        ):
            _require_tensor_equal(f"{uid}.{key}", left[key], right[key])
        _require_tree_equal(f"{uid}.condition", left["condition"], right["condition"])
        left_lifting = left["lifting_sample"]
        right_lifting = right["lifting_sample"]
        for key in (
            "visual_patch_features",
            "stock_condition",
            "intrinsics",
            "extrinsics",
            "view_ids",
        ):
            _require_tensor_equal(
                f"{uid}.lifting.{key}", left_lifting[key], right_lifting[key]
            )
        left_size = tuple(map(int, left_lifting["predicted_depth"].shape[-2:]))
        right_size = tuple(map(int, right_lifting["image_size"]))
        if left_size != right_size:
            raise RuntimeError(f"{uid} v1/v2 image size differs")
        for key in ("grid_transform", "extrinsics_type", "camera_forward_sign"):
            if left_lifting[key] != right_lifting[key]:
                raise RuntimeError(f"{uid} v1/v2 projection metadata differs: {key}")
        # Exercise the actual projection helper, including compact image_size.
        # The actual every-block projection runs after the Stock SLat 64->32
        # sparse stem.  Reproduce that coordinate domain for this metadata
        # equivalence check rather than feeding raw 64^3 target coordinates.
        coords = left["target_coords"].clone()
        coords[:, 1:] = torch.div(coords[:, 1:], 2, rounding_mode="floor")
        coords = torch.unique(coords, dim=0)[:256]
        view_indices = torch.arange(
            int(left_lifting["visual_patch_features"].shape[0]), dtype=torch.long
        )
        left_projected, left_valid, _ = project_sparse_frustum_dino(
            left_lifting,
            coords,
            device=torch.device("cpu"),
            view_indices=view_indices,
        )
        right_projected, right_valid, _ = project_sparse_frustum_dino(
            right_lifting,
            coords,
            device=torch.device("cpu"),
            view_indices=view_indices,
        )
        _require_tensor_equal(f"{uid}.projected", left_projected, right_projected)
        _require_tensor_equal(f"{uid}.projection_valid", left_valid, right_valid)
        records.append(
            {
                "index": index,
                "uid": uid,
                "target_point_count": int(left["target_coords"].shape[0]),
                "view_count": int(left_lifting["visual_patch_features"].shape[0]),
                "runtime_tensor_equality": True,
                "projection_equality": True,
            }
        )
        print(f"[official_slat_v1_v2_audit] {index + 1}/{len(v1)} {uid}", flush=True)
    report = {
        "format": AUDIT_FORMAT,
        "passed": True,
        "sample_count": len(records),
        "slat_config_hash": v1.config_hash,
        "lifting_config_hash": v1.lifting.config_hash,
        "slat_normalization_hash": v1.slat_normalization_hash,
        "v1": {
            "slat_manifest": str(Path(args.slat_manifest_v1).resolve()),
            "slat_manifest_sha256": sha256_file(args.slat_manifest_v1),
            "lifting_manifest": str(Path(args.lifting_manifest_v1).resolve()),
            "lifting_manifest_sha256": sha256_file(args.lifting_manifest_v1),
        },
        "v2": {
            "slat_manifest": str(Path(args.slat_manifest_v2).resolve()),
            "slat_manifest_sha256": sha256_file(args.slat_manifest_v2),
            "lifting_manifest": str(Path(args.lifting_manifest_v2).resolve()),
            "lifting_manifest_sha256": sha256_file(args.lifting_manifest_v2),
        },
        "checks": {
            "target_coords_feats_exact": True,
            "legacy_gt_placeholders_exact_in_memory": True,
            "positive_negative_slat_context_exact": True,
            "stock_ss_context_exact": True,
            "dino_intrinsics_extrinsics_exact": True,
            "actual_projection_output_exact": True,
        },
        "records": records,
    }
    report["report_sha256"] = canonical_sha256(report)
    output = Path(args.output).expanduser().resolve()
    atomic_json(output, report)
    print(json.dumps({key: report[key] for key in ("passed", "sample_count", "checks")}, indent=2))


if __name__ == "__main__":
    main()
