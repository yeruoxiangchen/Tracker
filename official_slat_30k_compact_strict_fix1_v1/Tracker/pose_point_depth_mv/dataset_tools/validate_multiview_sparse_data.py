#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image


TRACKER_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TRACKER_ROOT))

from pixal3d_multiview.train_sparse_multiview import MultiviewSparseManifestDataset  # noqa: E402


def validate_npz(path: str, expected_shape: tuple[int, int, int, int] | None) -> tuple[tuple[int, ...], bool]:
    with np.load(path) as data:
        if "z" not in data:
            raise ValueError(f"missing key z: {path}")
        z = data["z"]
    if z.ndim != 4:
        raise ValueError(f"z should be [C,D,H,W], got {z.shape}: {path}")
    finite = bool(np.isfinite(z).all())
    if not finite:
        raise ValueError(f"non-finite z: {path}")
    if expected_shape is not None and tuple(z.shape) != expected_shape:
        raise ValueError(f"z shape mismatch: got {z.shape}, expected {expected_shape}: {path}")
    return tuple(z.shape), finite


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate pixal3d_multiview sparse training manifest.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--image_root", default=None)
    parser.add_argument("--mask_root", default=None)
    parser.add_argument("--latent_root", default=None)
    parser.add_argument("--max_samples", type=int, default=100)
    parser.add_argument("--expected_z_shape", default="8,16,16,16")
    args = parser.parse_args()

    expected_shape = None
    if args.expected_z_shape:
        expected_shape = tuple(int(x) for x in args.expected_z_shape.split(","))
        if len(expected_shape) != 4:
            raise ValueError("--expected_z_shape should be C,D,H,W")

    dataset = MultiviewSparseManifestDataset(
        args.manifest,
        image_root=args.image_root,
        mask_root=args.mask_root,
        latent_root=args.latent_root,
        max_frames=0,
        apply_mask=False,
    )
    checked = 0
    view_counts = []
    z_shapes = {}
    failures = []
    for idx in range(min(len(dataset), max(args.max_samples, 0))):
        try:
            item = dataset[idx]
            view_counts.append(len(item["images"]))
            z_shape, _ = validate_npz(item["latent_path"], expected_shape)
            z_shapes[z_shape] = z_shapes.get(z_shape, 0) + 1
            if item["intrinsics"].shape[-2:] != (3, 3):
                raise ValueError(f"bad intrinsics shape: {tuple(item['intrinsics'].shape)}")
            if item["extrinsics"].shape[-2:] != (4, 4):
                raise ValueError(f"bad extrinsics shape: {tuple(item['extrinsics'].shape)}")
            for image in item["images"]:
                if not isinstance(image, Image.Image):
                    raise ValueError("image loader did not return PIL.Image")
            checked += 1
        except Exception as exc:
            failures.append({"index": idx, "error": f"{type(exc).__name__}: {exc}"})

    summary = {
        "manifest": args.manifest,
        "dataset_samples": len(dataset),
        "checked_samples": checked,
        "failures": failures,
        "view_count_min": min(view_counts) if view_counts else 0,
        "view_count_max": max(view_counts) if view_counts else 0,
        "z_shapes": {str(k): v for k, v in z_shapes.items()},
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
