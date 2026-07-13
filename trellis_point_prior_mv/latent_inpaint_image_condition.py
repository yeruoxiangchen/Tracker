from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from trellis_point_prior_mv.common import load_manifest, resolve_path


_FRAME_IMAGE_KEYS = ("image", "image_path", "rgb", "rgb_path", "rgba", "rgba_path", "path", "file_path")
_FRAME_MASK_KEYS = ("mask", "mask_path", "alpha", "alpha_path", "segmentation", "segmentation_path")


def _stable_seed(text: str, seed: int) -> int:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
    return (int(digest[:8], 16) + int(seed)) % (2**32)


def _select_indices(n: int, count: int, mode: str, *, seed: int = 0, uid: str = "") -> list[int]:
    if count <= 0 or n <= count:
        return list(range(n))
    mode = str(mode or "uniform").lower()
    if mode in {"first", "head"}:
        return list(range(count))
    if mode in {"random", "rand"}:
        rng = np.random.default_rng(_stable_seed(uid, seed))
        return sorted(rng.choice(n, size=count, replace=False).astype(int).tolist())
    ids = np.linspace(0, n - 1, count)
    return sorted(set(int(round(v)) for v in ids))[:count]


class SourceImageResolver:
    """Resolve source RGB frames referenced by a latent-inpaint manifest sample."""

    def __init__(self, default_source_manifest: str | Path | None = None):
        self.default_source_manifest = str(default_source_manifest) if default_source_manifest else None
        self._cache: dict[str, tuple[dict[str, Any], list[dict[str, Any]], Path]] = {}

    def _load(self, path: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]], Path]:
        p = Path(path)
        key = str(p)
        if key not in self._cache:
            payload, samples = load_manifest(p)
            self._cache[key] = (payload, samples, p)
        return self._cache[key]

    def source_sample(self, latent_sample: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], Path]:
        source_manifest = latent_sample.get("source_manifest") or self.default_source_manifest
        if not source_manifest:
            raise ValueError(f"latent sample has no source_manifest: {latent_sample.get('uid')}")
        payload, samples, manifest_path = self._load(source_manifest)
        if "source_index" in latent_sample:
            source_index = int(latent_sample["source_index"])
            if 0 <= source_index < len(samples):
                return payload, samples[source_index], manifest_path
        uid = str(latent_sample.get("uid", ""))
        for sample in samples:
            if str(sample.get("uid", sample.get("id", ""))) == uid:
                return payload, sample, manifest_path
        raise IndexError(f"cannot resolve source sample uid={uid!r} from {source_manifest}")

    def image_paths(
        self,
        latent_sample: dict[str, Any],
        *,
        max_views: int = 4,
        frame_select: str = "uniform",
        seed: int = 0,
    ) -> list[str]:
        payload, sample, manifest_path = self.source_sample(latent_sample)
        image_root = sample.get("image_root", payload.get("image_root", manifest_path.parent))
        frames = sample.get("frames") or []
        if not frames and any(key in sample for key in _FRAME_IMAGE_KEYS):
            frames = [sample]
        if not frames:
            raise ValueError(f"source sample has no frames/images: {sample.get('uid', latent_sample.get('uid'))}")
        uid = str(sample.get("uid", latent_sample.get("uid", "")))
        selected = _select_indices(len(frames), int(max_views), frame_select, seed=seed, uid=uid)
        paths: list[str] = []
        for idx in selected:
            frame = frames[idx]
            rel = None
            for key in _FRAME_IMAGE_KEYS:
                if frame.get(key):
                    rel = frame[key]
                    break
            if rel is None:
                raise ValueError(f"frame has no image path keys {list(_FRAME_IMAGE_KEYS)}: sample={uid}")
            path = resolve_path(image_root, rel)
            if not path.exists():
                raise FileNotFoundError(f"source image not found: {path}")
            paths.append(str(path))
        return paths

    def image_mask_paths(
        self,
        latent_sample: dict[str, Any],
        *,
        max_views: int = 4,
        frame_select: str = "uniform",
        seed: int = 0,
    ) -> tuple[list[str], list[str | None]]:
        image_paths, mask_paths, _ = self.image_mask_paths_with_frames(
            latent_sample,
            max_views=max_views,
            frame_select=frame_select,
            seed=seed,
        )
        return image_paths, mask_paths

    def image_mask_paths_with_frames(
        self,
        latent_sample: dict[str, Any],
        *,
        max_views: int = 4,
        frame_select: str = "uniform",
        seed: int = 0,
    ) -> tuple[list[str], list[str | None], list[dict[str, Any]]]:
        payload, sample, manifest_path = self.source_sample(latent_sample)
        image_root = sample.get("image_root", payload.get("image_root", manifest_path.parent))
        mask_root = sample.get("mask_root", payload.get("mask_root", manifest_path.parent))
        frames = sample.get("frames") or []
        if not frames and any(key in sample for key in _FRAME_IMAGE_KEYS):
            frames = [sample]
        if not frames:
            raise ValueError(f"source sample has no frames/images: {sample.get('uid', latent_sample.get('uid'))}")
        uid = str(sample.get("uid", latent_sample.get("uid", "")))
        selected = _select_indices(len(frames), int(max_views), frame_select, seed=seed, uid=uid)
        image_paths: list[str] = []
        mask_paths: list[str | None] = []
        for idx in selected:
            frame = frames[idx]
            image_rel = None
            for key in _FRAME_IMAGE_KEYS:
                if frame.get(key):
                    image_rel = frame[key]
                    break
            if image_rel is None:
                raise ValueError(f"frame has no image path keys {list(_FRAME_IMAGE_KEYS)}: sample={uid}")
            image_path = resolve_path(image_root, image_rel)
            if not image_path.exists():
                raise FileNotFoundError(f"source image not found: {image_path}")
            mask_rel = None
            for key in _FRAME_MASK_KEYS:
                if frame.get(key):
                    mask_rel = frame[key]
                    break
            if mask_rel is None:
                mask_rel = sample.get("mask")
            mask_path = resolve_path(mask_root, mask_rel) if mask_rel else None
            if mask_path is not None and not mask_path.exists():
                raise FileNotFoundError(f"source mask not found: {mask_path}")
            image_paths.append(str(image_path))
            mask_paths.append(None if mask_path is None else str(mask_path))
        return image_paths, mask_paths, [frames[idx] for idx in selected]


def apply_mask_and_crop(image_path: str | Path, mask_path: str | Path, resolution: int = 518) -> Image.Image:
    image = Image.open(image_path).convert("RGBA")
    mask = Image.open(mask_path).convert("L")
    if mask.size != image.size:
        mask = mask.resize(image.size, Image.Resampling.NEAREST)
    rgba = np.asarray(image).copy()
    alpha = np.asarray(mask)
    rgba[:, :, 3] = np.where(alpha > 127, 255, 0).astype(np.uint8)
    rgba[:, :, :3] = rgba[:, :, :3] * (rgba[:, :, 3:4] > 0)
    ys, xs = np.nonzero(rgba[:, :, 3] > 0)
    if len(xs) == 0:
        side = max(image.size)
        canvas = Image.new("RGBA", (side, side), (0, 0, 0, 0))
        canvas.paste(Image.fromarray(rgba), ((side - image.width) // 2, (side - image.height) // 2))
        return canvas.resize((resolution, resolution), Image.Resampling.BILINEAR)

    left, right = int(xs.min()), int(xs.max())
    top, bottom = int(ys.min()), int(ys.max())
    center_x = 0.5 * (left + right)
    center_y = 0.5 * (top + bottom)
    size = max(1, int(max(right - left + 1, bottom - top + 1) * 1.1))
    crop = (
        max(0, int(center_x - size // 2)),
        max(0, int(center_y - size // 2)),
        min(image.width, int(center_x + size // 2)),
        min(image.height, int(center_y + size // 2)),
    )
    cropped = Image.fromarray(rgba).crop(crop)
    side = max(cropped.size)
    canvas = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    canvas.paste(cropped, ((side - cropped.width) // 2, (side - cropped.height) // 2))
    return canvas.resize((resolution, resolution), Image.Resampling.BILINEAR)


@torch.no_grad()
def encode_image_condition(
    pipeline,
    image_paths: list[str],
    *,
    device: torch.device | str,
    dtype: torch.dtype,
    aggregation: str = "mean",
    preprocess: bool = False,
    mask_paths: list[str | None] | None = None,
    use_source_mask: bool = True,
    mask_crop_resolution: int = 518,
) -> torch.Tensor:
    if not image_paths:
        raise ValueError("image condition enabled but image_paths is empty")
    if mask_paths is not None and len(mask_paths) != len(image_paths):
        raise ValueError(f"mask_paths length {len(mask_paths)} != image_paths length {len(image_paths)}")
    images = []
    for idx, image_path in enumerate(image_paths):
        mask_path = None if mask_paths is None else mask_paths[idx]
        if use_source_mask and mask_path:
            image = apply_mask_and_crop(image_path, mask_path, resolution=int(mask_crop_resolution))
        else:
            image = Image.open(image_path)
        if preprocess and not (use_source_mask and mask_path):
            image = pipeline.preprocess_image(image)
        else:
            image = image.convert("RGB")
        images.append(image)
    cond_views = pipeline.encode_image(images).to(device=device, dtype=dtype)
    aggregation = str(aggregation or "mean").lower()
    if aggregation in {"views", "per_view", "perview"}:
        return cond_views.unsqueeze(0)
    if aggregation == "mean":
        return cond_views.mean(dim=0, keepdim=True)
    if aggregation == "first":
        return cond_views[:1]
    if aggregation == "concat":
        return cond_views.reshape(1, -1, cond_views.shape[-1])
    raise ValueError(f"unsupported image_cond_aggregation={aggregation!r}")


def fuse_point_image_condition(point_cond: torch.Tensor, image_cond: torch.Tensor | None, mode: str) -> torch.Tensor:
    mode = str(mode or "concat").lower()
    if mode in {"point_only", "point", "none"}:
        return point_cond
    if image_cond is None:
        if mode in {"image_only", "image"}:
            raise ValueError("cond_fusion=image_only requires USE_IMAGE_COND=1 / --use_image_cond")
        return point_cond
    if mode in {"image_only", "image"}:
        return image_cond
    if mode == "concat":
        if image_cond.shape[0] != point_cond.shape[0] or image_cond.shape[-1] != point_cond.shape[-1]:
            raise ValueError(
                f"image/point cond shape mismatch: image={tuple(image_cond.shape)} point={tuple(point_cond.shape)}"
            )
        return torch.cat([image_cond, point_cond], dim=1)
    raise ValueError(f"unsupported cond_fusion={mode!r}")
