from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


def _validate_target_coords(target_coords: np.ndarray, sample_id: str, path: Path) -> np.ndarray:
    if target_coords.ndim != 2 or target_coords.shape[1] not in (3, 4):
        raise ValueError(f"target_coords should be [N,3] or [N,4], got {target_coords.shape} for {sample_id}")
    coords_xyz = target_coords[:, -3:]
    bad = (
        ~np.isfinite(coords_xyz).all(axis=1)
        | (coords_xyz < 0).any(axis=1)
        | (coords_xyz >= 64).any(axis=1)
    )
    if bad.any():
        examples = target_coords[bad][:10].tolist()
        raise ValueError(f"target_coords out of [0,64) in {path}: {examples}")
    return np.ascontiguousarray(target_coords.astype(np.int64, copy=True))


class ObjaversePoseDataset(Dataset):
    """Reader for generated AR-pose sparse-structure training samples."""

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        num_views: int = 6,
        image_size: int = 256,
        random_views: bool = True,
    ):
        self.data_root = Path(data_root)
        self.split = split
        self.num_views = num_views
        self.image_size = image_size
        self.random_views = random_views

        manifest_path = self.data_root / f"{split}.json"
        if not manifest_path.exists():
            manifest_path = self.data_root / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"No split manifest found under {self.data_root}")

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.samples: list[dict[str, Any]] = manifest["samples"] if isinstance(manifest, dict) else manifest
        if not self.samples:
            raise ValueError(f"No samples in {manifest_path}")
        print(f"[ObjaversePoseDataset] {len(self.samples)} samples from {manifest_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        for _ in range(10):
            try:
                return self._load_item(index)
            except Exception as exc:
                print(f"[ObjaversePoseDataset] failed index={index}: {exc}")
                index = random.randrange(len(self.samples))
        return self._load_item(index)

    def _resolve(self, sample: dict[str, Any]) -> Path:
        path = Path(sample["npz"])
        return path if path.is_absolute() else self.data_root / path

    def _load_item(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.samples[index]
        npz_path = self._resolve(sample)
        with np.load(npz_path) as data:
            images = data["images"].astype(np.float32) / 255.0
            alpha = data["alpha"].astype(np.float32) / 255.0
            intrinsics = data["intrinsics"].astype(np.float32)
            extrinsics = data["extrinsics"].astype(np.float32)
            target_coords = _validate_target_coords(
                data["target_coords"],
                sample.get("uid", str(index)),
                npz_path,
            )

        num_available = images.shape[0]
        if self.num_views > 0 and num_available > self.num_views:
            if self.random_views:
                view_ids = np.sort(np.random.choice(num_available, self.num_views, replace=False))
            else:
                view_ids = np.arange(self.num_views)
            images = images[view_ids]
            alpha = alpha[view_ids]
            intrinsics = intrinsics[view_ids]
            extrinsics = extrinsics[view_ids]

        images_t = torch.from_numpy(images).permute(0, 3, 1, 2).contiguous()
        alpha_t = torch.from_numpy(alpha[:, None]).contiguous()
        return {
            "ref_image": images_t,
            "alpha": alpha_t,
            "target_coords": torch.from_numpy(target_coords).contiguous().long(),
            "batch_intrinsics": torch.from_numpy(intrinsics),
            "batch_extrinsics": torch.from_numpy(extrinsics),
            "sample_uid": sample.get("uid", str(index)),
            "sample_npz": str(npz_path),
        }


def custom_collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    batched_ref, batched_alpha = [], []
    batched_intrinsics, batched_extrinsics = [], []
    batched_coords = []
    sample_uids, sample_npzs = [], []

    for batch_idx, sample in enumerate(batch):
        coords = sample["target_coords"].contiguous().long()
        coords = coords[:, 1:] if coords.shape[-1] == 4 else coords
        bad = (
            (coords < 0).any(dim=1)
            | (coords >= 64).any(dim=1)
        )
        if bad.any():
            bad_rows = coords[bad][:10].tolist()
            raise ValueError(
                "Collate target_coords out of [0,64): "
                f"bad_rows={bad_rows}, uid={sample.get('sample_uid')}, npz={sample.get('sample_npz')}"
            )
        batch_col = torch.full((coords.shape[0], 1), batch_idx, dtype=coords.dtype)
        batched_coords.append(torch.cat([batch_col, coords], dim=1))

        batched_ref.append(sample["ref_image"])
        batched_alpha.append(sample["alpha"])
        batched_intrinsics.append(sample["batch_intrinsics"])
        batched_extrinsics.append(sample["batch_extrinsics"])
        sample_uids.append(sample.get("sample_uid", str(batch_idx)))
        sample_npzs.append(sample.get("sample_npz", ""))

    return {
        "ref_image": torch.stack(batched_ref, dim=0),
        "alpha": torch.stack(batched_alpha, dim=0),
        "target_coords": torch.cat(batched_coords, dim=0),
        "batch_intrinsics": torch.stack(batched_intrinsics, dim=0),
        "batch_extrinsics": torch.stack(batched_extrinsics, dim=0),
        "sample_uids": sample_uids,
        "sample_npzs": sample_npzs,
    }
