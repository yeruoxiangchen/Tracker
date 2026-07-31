#!/usr/bin/env python3
"""Freeze nested 2/4/8-view cases and object-stable SS/SLAT noise identities."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from pose_point_depth_mv.compare_pixal3d_singleview_smoke import (
    atomic_json,
    binding,
    canonical_sha256,
)


FORMAT = "pose_point_depth_mv.matched_view_protocol.v1"
VIEW_POSITIONS = {
    2: (0, 4),
    4: (0, 2, 4, 6),
    8: tuple(range(8)),
}


def stable_identity_seed(
    *,
    object_uid: str,
    joint_seed: int,
    stage: str,
) -> int:
    if not object_uid or not stage:
        raise ValueError("object_uid and stage must be non-empty")
    payload = f"tracker.matched-view.v1\\0{stage}\\0{object_uid}\\0{joint_seed}".encode(
        "utf-8"
    )
    # torch.Generator.manual_seed accepts signed 64-bit-compatible values.
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & (
        (1 << 63) - 1
    )


def stable_rank(selection_seed: int, object_uid: str, uid: str) -> str:
    return hashlib.sha256(
        f"tracker.matched-view.selection.v1\\0{selection_seed}\\0"
        f"{object_uid}\\0{uid}".encode("utf-8")
    ).hexdigest()


def resolve_root(manifest_path: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (manifest_path.parent / path).resolve()


def resolve_child(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def frame_binding(
    frame: dict[str, Any],
    *,
    image_root: Path,
    mask_root: Path,
) -> dict[str, Any]:
    image_path = resolve_child(image_root, frame["image"])
    mask_path = resolve_child(mask_root, frame["mask"])
    return {
        "source_view_index": int(frame["source_view_index"]),
        "intrinsic": frame["intrinsic"],
        "extrinsic": frame["extrinsic"],
        "image": binding(image_path),
        "mask": binding(mask_path),
    }


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_manifest", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--objects", type=int, default=16)
    parser.add_argument("--joint_seeds", default="42,43,44")
    parser.add_argument("--selection_seed", type=int, default=20260727)
    return parser


def parse_joint_seeds(value: str) -> list[int]:
    seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("joint_seeds must be non-empty and unique")
    return seeds


def main() -> None:
    args = make_parser().parse_args()
    if args.objects <= 0:
        raise ValueError("objects must be positive")
    source_path = args.source_manifest.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite matched-view protocol: {output_dir}")
    source = json.loads(source_path.read_text(encoding="utf-8"))
    rows = source.get("samples")
    if not isinstance(rows, list) or not rows:
        raise ValueError("source manifest contains no samples")
    image_root = resolve_root(source_path, source["image_root"])
    mask_root = resolve_root(source_path, source["mask_root"])
    eligible = [
        row
        for row in rows
        if isinstance(row.get("frames"), list) and len(row["frames"]) == 8
    ]
    ranked = sorted(
        eligible,
        key=lambda row: stable_rank(
            args.selection_seed, str(row["object_uid"]), str(row["uid"])
        ),
    )
    if len(ranked) < args.objects:
        raise RuntimeError(
            f"only {len(ranked)} exact-8-view rows are available, requested {args.objects}"
        )
    selected = ranked[: args.objects]
    joint_seeds = parse_joint_seeds(args.joint_seeds)
    cases = []
    for row in selected:
        object_uid = str(row["object_uid"])
        uid = str(row["uid"])
        bound_frames = [
            frame_binding(frame, image_root=image_root, mask_root=mask_root)
            for frame in row["frames"]
        ]
        variants = {
            str(view_count): {
                "view_count": view_count,
                "positions_in_frozen_8": list(VIEW_POSITIONS[view_count]),
                "frames": [bound_frames[index] for index in VIEW_POSITIONS[view_count]],
            }
            for view_count in (2, 4, 8)
        }
        if not (
            set(VIEW_POSITIONS[2])
            < set(VIEW_POSITIONS[4])
            < set(VIEW_POSITIONS[8])
        ):
            raise AssertionError("nested-view policy is not strictly nested")
        cases.append(
            {
                "object_uid": object_uid,
                "uid": uid,
                "selection_rank": stable_rank(
                    args.selection_seed, object_uid, uid
                ),
                "source_glb": binding(row["source_glb"]),
                "ss_latent": binding(row["ss_latent"]),
                "variants": variants,
                "noise": {
                    str(seed): {
                        "joint_seed": seed,
                        "ss_seed": stable_identity_seed(
                            object_uid=object_uid, joint_seed=seed, stage="ss"
                        ),
                        "slat_seed": stable_identity_seed(
                            object_uid=object_uid, joint_seed=seed, stage="slat"
                        ),
                    }
                    for seed in joint_seeds
                },
            }
        )
    output_dir.mkdir(parents=True)
    body = {
        "format": FORMAT,
        "formal": False,
        "source_manifest": binding(source_path),
        "selection_seed": int(args.selection_seed),
        "selection_policy": (
            "stable SHA-256 rank over exact-8-view rows; independent of model outputs"
        ),
        "view_policy": {
            "name": "nested_even_positions_from_frozen_8_v1",
            "positions": {
                str(key): list(value) for key, value in VIEW_POSITIONS.items()
            },
            "strictly_nested": True,
        },
        "noise_policy": (
            "SHA-256(stage, object_uid, joint_seed), independent of dataset index, "
            "rollout position, view count, and worker count"
        ),
        "joint_seeds": joint_seeds,
        "object_count": len(cases),
        "cases": cases,
        "guardrail": (
            "This file freezes inputs and stochastic identities only. It does not "
            "claim that 2/4/8 model outputs have been generated or evaluated."
        ),
    }
    body["protocol_sha256"] = canonical_sha256(body)
    atomic_json(output_dir / "protocol.json", body)
    print(
        json.dumps(
            {
                "passed": True,
                "object_count": len(cases),
                "variants_per_object": [2, 4, 8],
                "protocol_sha256": body["protocol_sha256"],
                "output": str(output_dir / "protocol.json"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
