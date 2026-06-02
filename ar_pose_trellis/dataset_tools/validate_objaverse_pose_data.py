#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

TRACKER_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TRACKER_ROOT))

from ar_pose_trellis.objaverse_pose_dataset import ObjaversePoseDataset, custom_collate


def _bad_coords_np(coords: np.ndarray, resolution: int) -> np.ndarray:
    xyz = coords[:, -3:]
    return (
        ~np.isfinite(xyz).all(axis=1)
        | (xyz < 0).any(axis=1)
        | (xyz >= resolution).any(axis=1)
    )


def validate_files(data_root: Path, split: str, resolution: int, max_samples: int) -> list[dict]:
    manifest_path = data_root / f"{split}.json"
    if not manifest_path.exists():
        manifest_path = data_root / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    samples = payload["samples"] if isinstance(payload, dict) else payload
    if max_samples > 0:
        samples = samples[:max_samples]

    failures = []
    voxel_counts = []
    min_alpha_counts = []
    for idx, sample in enumerate(samples):
        npz_path = Path(sample["npz"])
        if not npz_path.is_absolute():
            npz_path = data_root / npz_path
        try:
            with np.load(npz_path) as data:
                images = data["images"]
                alpha = data["alpha"]
                intrinsics = data["intrinsics"]
                extrinsics = data["extrinsics"]
                coords = data["target_coords"]

            checks = []
            if images.ndim != 4 or images.shape[-1] != 3:
                checks.append(f"images shape={images.shape}")
            if alpha.ndim != 3:
                checks.append(f"alpha shape={alpha.shape}")
            if intrinsics.ndim != 3 or intrinsics.shape[-2:] != (3, 3) or not np.isfinite(intrinsics).all():
                checks.append(f"intrinsics invalid shape={intrinsics.shape}")
            if extrinsics.ndim != 3 or extrinsics.shape[-2:] != (4, 4) or not np.isfinite(extrinsics).all():
                checks.append(f"extrinsics invalid shape={extrinsics.shape}")
            if coords.ndim != 2 or coords.shape[1] not in (3, 4):
                checks.append(f"target_coords shape={coords.shape}")
            else:
                bad = _bad_coords_np(coords, resolution)
                if bad.any():
                    checks.append(f"target_coords bad={coords[bad][:10].tolist()}")
                voxel_counts.append(int(coords.shape[0]))
            if alpha.ndim == 3:
                alpha_counts = (alpha > 0).reshape(alpha.shape[0], -1).sum(axis=1)
                min_alpha_counts.append(int(alpha_counts.min()))
                if (alpha_counts == 0).any():
                    checks.append(f"empty alpha views={np.where(alpha_counts == 0)[0].tolist()}")

            if checks:
                failures.append({"index": idx, "uid": sample.get("uid"), "npz": str(npz_path), "errors": checks})
        except Exception as exc:
            failures.append({"index": idx, "uid": sample.get("uid"), "npz": str(npz_path), "errors": [repr(exc)]})

    print(f"[validate_files] checked={len(samples)} failures={len(failures)}")
    if voxel_counts:
        arr = np.asarray(voxel_counts)
        print(f"[validate_files] voxels min/median/max={arr.min()}/{np.median(arr):.1f}/{arr.max()}")
    if min_alpha_counts:
        arr = np.asarray(min_alpha_counts)
        print(f"[validate_files] min-alpha-pixels min/median/max={arr.min()}/{np.median(arr):.1f}/{arr.max()}")
    for row in failures[:20]:
        print(f"[validate_files][fail] {row}")
    return failures


def validate_loader(data_root: Path, split: str, num_views: int, batch_size: int, num_workers: int, batches: int, resolution: int) -> int:
    dataset = ObjaversePoseDataset(str(data_root), split=split, num_views=num_views)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
        collate_fn=custom_collate,
        drop_last=True,
    )
    failures = 0
    checked = 0
    for batch_idx, batch in enumerate(loader):
        coords = batch["target_coords"].detach().cpu().long()
        bad = (
            (coords[:, 0] < 0)
            | (coords[:, 0] >= batch_size)
            | (coords[:, 1:] < 0).any(dim=1)
            | (coords[:, 1:] >= resolution).any(dim=1)
        )
        if bad.any():
            failures += 1
            print(
                "[validate_loader][fail] "
                f"batch={batch_idx} bad_rows={coords[bad][:10].tolist()} "
                f"uids={batch.get('sample_uids')} npzs={batch.get('sample_npzs')}",
                flush=True,
            )
        checked += 1
        if batches > 0 and checked >= batches:
            break
    print(f"[validate_loader] checked_batches={checked} failures={failures} workers={num_workers}")
    return failures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--num_views", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--loader_batches", type=int, default=100)
    parser.add_argument("--skip_loader", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    file_failures = validate_files(data_root, args.split, args.resolution, args.max_samples)
    loader_failures = 0
    if not args.skip_loader:
        loader_failures = validate_loader(
            data_root,
            args.split,
            args.num_views,
            args.batch_size,
            args.num_workers,
            args.loader_batches,
            args.resolution,
        )
    if file_failures or loader_failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
