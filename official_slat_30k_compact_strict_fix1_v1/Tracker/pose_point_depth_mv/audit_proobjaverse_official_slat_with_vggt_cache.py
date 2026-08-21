#!/usr/bin/env python3
"""Read-only structural audit for an official with-VGGT SLat cache pair."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from typing import Any

import torch

from pose_point_depth_mv.proobjaverse_official_slat_with_vggt_cache import (
    WITH_VGGT_CONTEXT_VERSION,
    WithVGGTNativeConditionSLatDataset,
)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache_manifest", required=True)
    parser.add_argument("--lifting_cache_manifest", required=True)
    parser.add_argument("--indices", default="all")
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--verify_hashes", action="store_true")
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    dataset = WithVGGTNativeConditionSLatDataset(
        args.cache_manifest,
        args.lifting_cache_manifest,
        indices=args.indices,
        verify_hashes=bool(args.verify_hashes),
    )
    limit = len(dataset)
    if int(args.max_samples) > 0:
        limit = min(limit, int(args.max_samples))
    if limit <= 0:
        raise ValueError("with-VGGT audit selection is empty")
    shapes: Counter[tuple[int, ...]] = Counter()
    dtypes: Counter[str] = Counter()
    uids: list[str] = []
    for index in range(limit):
        sample = dataset[index]
        positive = sample["condition"]["cond"]
        negative = sample["condition"]["neg_cond"]
        if len(positive) != len(negative) or not positive:
            raise RuntimeError("with-VGGT condition has invalid view lists")
        context = torch.cat(positive, dim=0)
        if any(
            int(torch.count_nonzero(value).item()) != 0 for value in negative
        ):
            raise RuntimeError("with-VGGT negative condition is not exactly zero")
        if sample["with_vggt_sidecar"]["vggt_camera_consumed"] is not False:
            raise RuntimeError("with-VGGT cache unexpectedly consumes VGGT cameras")
        shapes[tuple(context.shape)] += 1
        dtypes[str(context.dtype)] += 1
        uids.append(str(sample["uid"]))
    if len(uids) != len(set(uids)):
        raise RuntimeError("with-VGGT audited subset contains duplicate UIDs")
    context_contract = dict(dataset.config.get("slat_input_context", {}))
    if (
        context_contract.get("version") != WITH_VGGT_CONTEXT_VERSION
        or context_contract.get("vggt_model_executed") is not True
        or context_contract.get("vggt_camera_consumed") is not False
        or context_contract.get("known_pose_dino_branch_unchanged") is not True
    ):
        raise RuntimeError("with-VGGT manifest context contract is invalid")
    return {
        "passed": True,
        "audited_sample_count": limit,
        "manifest_sample_count": len(dataset),
        "pair_identity": dataset.pair_identity,
        "config_hash": dataset.config_hash,
        "sidecar_contract_hash": dataset.sidecar_contract_hash,
        "native_context_shapes": {
            "x".join(map(str, shape)): count for shape, count in sorted(shapes.items())
        },
        "native_context_dtypes": dict(sorted(dtypes.items())),
        "negative_context_exact_zero": True,
        "same_base_lifting_view_ids": True,
        "known_pose_dino_branch_unchanged": True,
        "vggt_model_executed_during_sidecar_build": True,
        "vggt_camera_consumed": False,
        "base_cache_rewritten": False,
    }


def main() -> None:
    report = run(make_parser().parse_args())
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
