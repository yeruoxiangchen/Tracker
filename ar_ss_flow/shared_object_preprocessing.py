from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any

import numpy as np
from PIL import Image


SHARED_OBJECT_PREPROCESSING_VERSION = "ar_ss_flow.shared_object_preprocessing.v1"


def shared_preprocessing_contract(
    *, resolution: int, foreground_margin: float, alpha_threshold: float
) -> dict[str, Any]:
    if int(resolution) <= 0:
        raise ValueError("shared preprocessing resolution must be positive")
    if not math.isfinite(float(foreground_margin)) or float(foreground_margin) < 1.0:
        raise ValueError("foreground_margin must be finite and at least 1")
    if not 0.0 < float(alpha_threshold) < 1.0:
        raise ValueError("alpha_threshold must be in (0,1)")
    return {
        "version": SHARED_OBJECT_PREPROCESSING_VERSION,
        "resolution": int(resolution),
        "foreground_margin": float(foreground_margin),
        "alpha_threshold": float(alpha_threshold),
        "geometry": "foreground_bbox_square_crop_with_out_of_frame_padding_then_resize",
        "background": "black_after_resized_alpha_threshold",
        "rgb_resampling": "bilinear",
        "mask_resampling": "bilinear",
        "intrinsics_rule": "K_feature=source_to_feature_affine@K_source",
        "affine_pixel_convention": "u_feature=s*u_source-s*crop_left",
    }


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass
class SharedObjectViews:
    images: list[Image.Image]
    masks: np.ndarray
    source_sizes: list[tuple[int, int]]
    source_to_feature_affines: np.ndarray
    crop_boxes: list[tuple[int, int, int, int]]
    foreground_retained_fractions: list[float]
    contract: dict[str, Any]

    def geometry_record(self) -> dict[str, Any]:
        record = {
            "contract": self.contract,
            "contract_hash": canonical_json_sha256(self.contract),
            "source_sizes_wh": [list(size) for size in self.source_sizes],
            "source_to_feature_affines": self.source_to_feature_affines.tolist(),
            "crop_boxes_xyxy": [list(box) for box in self.crop_boxes],
            "foreground_retained_fractions": self.foreground_retained_fractions,
        }
        record["geometry_hash"] = canonical_json_sha256(record)
        return record


def _prepare_view(
    image_path: str,
    mask_path: str,
    *,
    contract: dict[str, Any],
) -> tuple[Image.Image, np.ndarray, tuple[int, int], np.ndarray, tuple[int, int, int, int], float]:
    with Image.open(image_path) as handle:
        rgb = handle.convert("RGB")
    with Image.open(mask_path) as handle:
        mask = handle.convert("L")
    if rgb.size != mask.size:
        raise ValueError(
            f"mask/image size mismatch: {mask_path}={mask.size}, {image_path}={rgb.size}"
        )

    rgb_array = np.asarray(rgb, dtype=np.uint8)
    mask_array = np.asarray(mask, dtype=np.uint8)
    threshold = float(contract["alpha_threshold"]) * 255.0
    foreground = mask_array > threshold
    ys, xs = np.nonzero(foreground)
    if not len(xs):
        raise ValueError(f"foreground mask is empty: {mask_path}")

    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())
    foreground_width = x_max - x_min + 1
    foreground_height = y_max - y_min + 1
    side = max(
        1,
        int(
            math.ceil(
                max(foreground_width, foreground_height)
                * float(contract["foreground_margin"])
            )
        ),
    )
    center_x = (x_min + x_max + 1) / 2.0
    center_y = (y_min + y_max + 1) / 2.0
    crop_left = int(math.floor(center_x - side / 2.0))
    crop_top = int(math.floor(center_y - side / 2.0))
    crop_box = (crop_left, crop_top, crop_left + side, crop_top + side)

    rgba = np.concatenate((rgb_array, mask_array[..., None]), axis=-1)
    canvas = np.zeros((side, side, 4), dtype=np.uint8)
    source_left = max(0, crop_box[0])
    source_top = max(0, crop_box[1])
    source_right = min(rgb.width, crop_box[2])
    source_bottom = min(rgb.height, crop_box[3])
    if source_right <= source_left or source_bottom <= source_top:
        raise RuntimeError(f"shared crop misses source image: {crop_box} vs {rgb.size}")
    destination_left = source_left - crop_box[0]
    destination_top = source_top - crop_box[1]
    destination_right = destination_left + source_right - source_left
    destination_bottom = destination_top + source_bottom - source_top
    canvas[
        destination_top:destination_bottom, destination_left:destination_right
    ] = rgba[source_top:source_bottom, source_left:source_right]

    retained = int(foreground[source_top:source_bottom, source_left:source_right].sum())
    retained_fraction = float(retained / int(foreground.sum()))
    if retained_fraction < 1.0:
        raise RuntimeError(
            f"shared crop removed foreground pixels: retained={retained_fraction:.8f}"
        )

    resolution = int(contract["resolution"])
    resized = Image.fromarray(canvas, mode="RGBA").resize(
        (resolution, resolution), Image.Resampling.BILINEAR
    )
    resized_array = np.asarray(resized, dtype=np.uint8)
    resized_mask = resized_array[..., 3].astype(np.float32) / 255.0
    masked_rgb = resized_array[..., :3] * (
        resized_array[..., 3:4] > threshold
    ).astype(np.uint8)

    scale = float(resolution) / float(side)
    affine = np.asarray(
        (
            (scale, 0.0, -scale * float(crop_left)),
            (0.0, scale, -scale * float(crop_top)),
            (0.0, 0.0, 1.0),
        ),
        dtype=np.float32,
    )
    return (
        Image.fromarray(masked_rgb, mode="RGB"),
        resized_mask,
        rgb.size,
        affine,
        crop_box,
        retained_fraction,
    )


def prepare_shared_object_views(
    image_paths: list[str],
    mask_paths: list[str],
    *,
    resolution: int,
    foreground_margin: float = 1.10,
    alpha_threshold: float = 0.80,
) -> SharedObjectViews:
    if len(image_paths) != len(mask_paths) or not image_paths:
        raise ValueError("image/mask paths must be non-empty and aligned")
    contract = shared_preprocessing_contract(
        resolution=resolution,
        foreground_margin=foreground_margin,
        alpha_threshold=alpha_threshold,
    )
    prepared = [
        _prepare_view(image_path, mask_path, contract=contract)
        for image_path, mask_path in zip(image_paths, mask_paths)
    ]
    return SharedObjectViews(
        images=[row[0] for row in prepared],
        masks=np.stack([row[1] for row in prepared]),
        source_sizes=[row[2] for row in prepared],
        source_to_feature_affines=np.stack([row[3] for row in prepared]),
        crop_boxes=[row[4] for row in prepared],
        foreground_retained_fractions=[row[5] for row in prepared],
        contract=contract,
    )


def transform_intrinsics(
    intrinsics: np.ndarray, source_to_feature_affines: np.ndarray
) -> np.ndarray:
    source = np.asarray(intrinsics, dtype=np.float32)
    affine = np.asarray(source_to_feature_affines, dtype=np.float32)
    if source.ndim != 3 or source.shape[1:] != (3, 3):
        raise ValueError(f"intrinsics must be [V,3,3], got {source.shape}")
    if affine.shape != source.shape:
        raise ValueError(
            f"affines must match intrinsics shape: {affine.shape} != {source.shape}"
        )
    transformed = np.matmul(affine, source).astype(np.float32)
    transformed[:, 2, :] = np.asarray((0.0, 0.0, 1.0), dtype=np.float32)
    return transformed
