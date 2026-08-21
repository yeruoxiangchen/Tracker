#!/usr/bin/env python3
"""Build audited local ``lh-slats`` from existing Pixal3D multi-view renders.

This follows the released TRELLIS data path:

1. extract DINOv2 ViT-L/14 register patch tokens from canonical renders;
2. project the 64^3 sparse surface coordinates into every selected view;
3. average the sampled 1024-D tokens across views; and
4. encode them with the official deterministic SLatEncoder posterior mean.

The result has the ProObjaverse ``lh-slats`` file schema (``coords[N,3]`` and
``feats[N,8]``), but is explicitly labelled as a local limited-view rebuild.
It must pass the downstream coordinate-frame and native-decoder audits before
being admitted as direct-SLAT supervision.
"""

from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Iterable

os.environ.setdefault("SPCONV_ALGO", "native")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
os.environ.setdefault("TORCH_EXTENSIONS_DIR", "/tmp/torch_extensions")

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


TRACKER_ROOT = Path(__file__).resolve().parents[1]
RECONVIAGEN_ROOT = TRACKER_ROOT / "ReconViaGen"
VGGT_ROOT = RECONVIAGEN_ROOT / "wheels" / "vggt"
for candidate in (TRACKER_ROOT, RECONVIAGEN_ROOT, VGGT_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from pose_point_depth_mv.direct_slat_flow import (  # noqa: E402
    canonical_json_sha256,
)


LOCAL_LH_SLAT_VERSION = "pose_point_depth_mv.local_lh_slats.v2"
OFFICIAL_SLAT_ENCODER_SHA256 = (
    "21dceac6bee917ab6458ff52c9757ba89a779d03031c7bd17f9e7f0103bfd436"
)
DINOV2_VITL14_REG4_SHA256 = (
    "36e4deffbaef061a2576705b0c36f93621e2ae20bf6274694821b0b492551b51"
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_csv(value: str) -> list[str]:
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values or len(values) != len(set(values)):
        raise ValueError("CSV paths must be non-empty and unique")
    return values


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(payload, handle, indent=2)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def atomic_savez(path: Path, **payload: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        suffix=".npz", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        np.savez_compressed(temporary, **payload)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def voxel_centers_render_space(
    coords: np.ndarray,
    rotation: np.ndarray,
    *,
    resolution: int = 64,
) -> np.ndarray:
    coords = np.asarray(coords, dtype=np.int32)
    rotation = np.asarray(rotation, dtype=np.float32)
    if coords.ndim != 2 or coords.shape[1] != 3 or len(coords) == 0:
        raise ValueError(f"coords must be non-empty [N,3], got {coords.shape}")
    if int(coords.min()) < 0 or int(coords.max()) >= int(resolution):
        raise ValueError("sparse coordinates leave the encoder resolution")
    if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
        raise ValueError("canonical rotation must be finite [3,3]")
    points = (coords.astype(np.float32) + 0.5) / float(resolution) - 0.5
    return (points @ rotation.T).astype(np.float32)


def project_points_to_grid(
    points: torch.Tensor,
    intrinsics: torch.Tensor,
    c2w: torch.Tensor,
    *,
    source_height: int,
    source_width: int,
    target_height: int,
    target_width: int,
    camera_forward_sign: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project render-space points to an align_corners=False sampling grid."""

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must be [N,3]")
    if intrinsics.ndim != 3 or intrinsics.shape[-2:] != (3, 3):
        raise ValueError("intrinsics must be [V,3,3]")
    if c2w.shape != (len(intrinsics), 4, 4):
        raise ValueError("c2w must be [V,4,4] and align with intrinsics")
    if min(source_height, source_width, target_height, target_width) <= 0:
        raise ValueError("image dimensions must be positive")
    batch = int(intrinsics.shape[0])
    homogeneous = torch.cat(
        [points, torch.ones_like(points[:, :1])], dim=1
    )
    w2c = torch.linalg.inv(c2w.float())
    camera = torch.einsum("vij,nj->vni", w2c, homogeneous)[:, :, :3]
    sign = float(camera_forward_sign)
    depth = camera[:, :, 2] * sign
    x = camera[:, :, 0] * sign
    y = camera[:, :, 1] * sign
    safe_depth = depth.clamp_min(1.0e-6)
    scale_x = float(target_width) / float(source_width)
    scale_y = float(target_height) / float(source_height)
    fx = intrinsics[:, 0, 0:1] * scale_x
    fy = intrinsics[:, 1, 1:2] * scale_y
    # PIL resize maps pixel centres as (p + 0.5) * scale - 0.5.
    cx = (intrinsics[:, 0, 2:3] + 0.5) * scale_x - 0.5
    cy = (intrinsics[:, 1, 2:3] + 0.5) * scale_y - 0.5
    u = fx * (x / safe_depth) + cx
    v = fy * (y / safe_depth) + cy
    grid_x = (u + 0.5) * (2.0 / float(target_width)) - 1.0
    grid_y = (v + 0.5) * (2.0 / float(target_height)) - 1.0
    grid = torch.stack([grid_x, grid_y], dim=-1)
    valid = (
        (depth > 1.0e-5)
        & (u >= 0.0)
        & (u < float(target_width))
        & (v >= 0.0)
        & (v < float(target_height))
    )
    if grid.shape != (batch, len(points), 2):
        raise RuntimeError("unexpected projection grid shape")
    return grid, valid


def resolve_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def load_required_objects(lifting_manifests: Iterable[str]) -> set[str]:
    objects: set[str] = set()
    for value in lifting_manifests:
        path = Path(value).resolve()
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload.get("samples")
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"lifting manifest contains no samples: {path}")
        for row in rows:
            objects.add(str(row.get("object_uid", row["uid"])))
    if not objects:
        raise ValueError("required object set is empty")
    return objects


def index_render_samples(
    render_manifests: Iterable[str],
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    by_object: dict[str, list[dict[str, Any]]] = {}
    bindings: list[dict[str, Any]] = []
    for value in render_manifests:
        path = Path(value).resolve()
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("format") != "pixal3d_multiview.objaverse_sparse.v1":
            raise ValueError(f"unsupported render manifest format: {path}")
        if str(payload.get("extrinsics_type")) != "c2w":
            raise ValueError("local lh-slats builder requires c2w extrinsics")
        if int(payload.get("voxel_resolution", -1)) != 64:
            raise ValueError("local lh-slats builder requires 64^3 sparse coordinates")
        image_root = Path(payload["image_root"]).resolve()
        mask_root = Path(payload["mask_root"]).resolve()
        latent_root = Path(payload["latent_root"]).resolve()
        rows = payload.get("samples")
        if not isinstance(rows, list):
            raise ValueError(f"render manifest contains no sample list: {path}")
        for row in rows:
            enriched = dict(row)
            enriched["_manifest"] = str(path)
            enriched["_image_root"] = str(image_root)
            enriched["_mask_root"] = str(mask_root)
            enriched["_latent_root"] = str(latent_root)
            enriched["_image_size"] = int(payload["image_size"])
            enriched["_camera_forward_sign"] = float(
                payload.get("camera_forward_sign", 1.0)
            )
            object_uid = str(row.get("object_uid", row["uid"]))
            by_object.setdefault(object_uid, []).append(enriched)
        bindings.append(
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "image_root": str(image_root),
                "mask_root": str(mask_root),
                "latent_root": str(latent_root),
                "sample_count": len(rows),
            }
        )
    for rows in by_object.values():
        rows.sort(key=lambda row: str(row["uid"]))
    return by_object, bindings


def object_inputs(
    object_uid: str,
    rows: list[dict[str, Any]],
    *,
    max_views: int,
) -> dict[str, Any]:
    if not rows:
        raise ValueError(f"object={object_uid} has no render rows")
    target_coords: np.ndarray | None = None
    rotation: np.ndarray | None = None
    source_glb: str | None = None
    latent_bindings: list[dict[str, str]] = []
    frames: list[dict[str, Any]] = []
    seen_images: set[str] = set()
    for row in rows:
        latent_path = resolve_path(Path(row["_latent_root"]), row["ss_latent"])
        with np.load(latent_path) as payload:
            coords = np.asarray(payload["target_coords"], dtype=np.int32)
            current_rotation = np.asarray(
                payload["pixal3d_rotation"], dtype=np.float32
            )
            current_glb = str(Path(str(payload["source_glb"])).resolve())
        if target_coords is None:
            target_coords = coords
            rotation = current_rotation
            source_glb = current_glb
        elif not np.array_equal(target_coords, coords):
            raise RuntimeError(
                f"object={object_uid} sequences do not share repaired target coords"
            )
        elif not np.array_equal(rotation, current_rotation):
            raise RuntimeError(f"object={object_uid} canonical rotations differ")
        elif source_glb != current_glb:
            raise RuntimeError(f"object={object_uid} source GLBs differ")
        latent_bindings.append(
            {"path": str(latent_path), "sha256": sha256_file(latent_path)}
        )
        for frame_index, frame in enumerate(row.get("frames", [])):
            image_path = resolve_path(Path(row["_image_root"]), frame["image"])
            mask_path = resolve_path(Path(row["_mask_root"]), frame["mask"])
            identity = str(image_path)
            if identity in seen_images:
                continue
            seen_images.add(identity)
            frames.append(
                {
                    "uid": str(row["uid"]),
                    "frame_index": int(frame_index),
                    "image_path": str(image_path),
                    "mask_path": str(mask_path),
                    "intrinsic": frame["intrinsic"],
                    "c2w": frame["extrinsic"],
                    "source_size": int(row["_image_size"]),
                    "camera_forward_sign": float(row["_camera_forward_sign"]),
                }
            )
    frames.sort(key=lambda frame: (frame["uid"], frame["frame_index"]))
    if max_views > 0:
        frames = frames[: int(max_views)]
    if target_coords is None or rotation is None or source_glb is None:
        raise RuntimeError(f"object={object_uid} failed to resolve canonical target")
    return {
        "coords": target_coords,
        "rotation": rotation,
        "source_glb": source_glb,
        "frames": frames,
        "source_latents": latent_bindings,
    }


def load_image_pair(
    frame: dict[str, Any], *, image_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    image = Image.open(frame["image_path"]).convert("RGB").resize(
        (image_size, image_size), Image.Resampling.LANCZOS
    )
    mask = Image.open(frame["mask_path"]).convert("L").resize(
        (image_size, image_size), Image.Resampling.BILINEAR
    )
    rgb = torch.from_numpy(np.asarray(image, dtype=np.float32).copy()).permute(2, 0, 1)
    alpha = torch.from_numpy(np.asarray(mask, dtype=np.float32).copy())[None]
    rgb = (rgb / 255.0) * (alpha / 255.0)
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    return (rgb - mean) / std, alpha / 255.0


def grid_sample_with_validity(
    feature_maps: torch.Tensor,
    grid: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    """Sample ``[V,C,H,W]`` maps and force invalid projections to exact zero."""

    if feature_maps.ndim != 4 or grid.ndim != 3 or valid.ndim != 2:
        raise ValueError("invalid projected sampling tensor ranks")
    if grid.shape[:2] != valid.shape or grid.shape[-1] != 2:
        raise ValueError("grid/valid projection shapes differ")
    if feature_maps.shape[0] != grid.shape[0]:
        raise ValueError("feature-map and projection view counts differ")
    sampled = F.grid_sample(
        feature_maps.float(),
        grid[:, :, None, :],
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    ).squeeze(-1).permute(0, 2, 1)
    return sampled * valid[..., None].to(dtype=sampled.dtype)


def camera_corruption_partner_indices(frames: list[dict[str, Any]]) -> list[int]:
    """Return a deterministic half-turn camera partner within each sequence."""

    groups: dict[str, list[int]] = {}
    for index, frame in enumerate(frames):
        groups.setdefault(str(frame["uid"]), []).append(index)
    partners = [-1] * len(frames)
    for uid, indices in groups.items():
        if len(indices) < 2:
            raise ValueError(f"sequence={uid} needs at least two cameras for corruption")
        shift = max(1, len(indices) // 2)
        for offset, index in enumerate(indices):
            partners[index] = indices[(offset + shift) % len(indices)]
    if any(index < 0 for index in partners):
        raise RuntimeError("camera corruption partner construction is incomplete")
    return partners


def projection_gate_failures(
    stats: dict[str, Any],
    *,
    min_visible_view_fraction_mean: float,
    min_mask_support_view_fraction_mean: float,
    min_mask_support_ge2_ratio: float,
    min_mask_support_ge4_ratio: float,
    max_zero_mask_support_ratio: float,
    min_camera_corruption_mask_support_drop: float | None = None,
) -> list[str]:
    """Return projection-gate failures using an integer-safe zero-support limit.

    Ratio thresholds for the ordinary coverage checks remain direct comparisons.
    The zero-mask-support gate is evaluated in integer space as

        zero_count <= ceil(max_ratio * point_count)

    so a non-divisible point count is not rejected solely because the permitted
    number of voxels is fractional.  For compatibility with older test fixtures,
    ``zero_mask_support_count`` is reconstructed from the stored ratio when the
    explicit count is absent.
    """

    checks = (
        (
            "visible_view_fraction_mean",
            float(stats["visible_view_fraction_mean"]),
            float(min_visible_view_fraction_mean),
            ">=",
        ),
        (
            "mask_support_view_fraction_mean",
            float(stats["mask_support_view_fraction_mean"]),
            float(min_mask_support_view_fraction_mean),
            ">=",
        ),
        (
            "mask_support_ge2_ratio",
            float(stats["mask_support_ge2_ratio"]),
            float(min_mask_support_ge2_ratio),
            ">=",
        ),
        (
            "mask_support_ge4_ratio",
            float(stats["mask_support_ge4_ratio"]),
            float(min_mask_support_ge4_ratio),
            ">=",
        ),
    )
    if min_camera_corruption_mask_support_drop is not None:
        checks = checks + (
            (
                "camera_corruption_mask_support_drop",
                float(stats["camera_corruption_mask_support_drop"]),
                float(min_camera_corruption_mask_support_drop),
                ">=",
            ),
        )

    failures: list[str] = []
    for name, actual, threshold, operator in checks:
        passed = actual >= threshold if operator == ">=" else actual <= threshold
        if not passed:
            failures.append(
                f"{name}={actual:.6f} must be {operator} {threshold:.6f}"
            )

    point_count = int(stats["point_count"])
    zero_mask_support_ratio = float(stats["zero_mask_support_ratio"])
    if point_count <= 0:
        failures.append(f"point_count={point_count} must be > 0")
        return failures

    if "zero_mask_support_count" in stats:
        zero_mask_support_count = int(stats["zero_mask_support_count"])
    else:
        # Backward-compatible path for older unit-test fixtures or reports.
        zero_mask_support_count = int(round(zero_mask_support_ratio * point_count))
        stats["zero_mask_support_count"] = zero_mask_support_count

    if not 0 <= zero_mask_support_count <= point_count:
        failures.append(
            f"zero_mask_support_count={zero_mask_support_count} must lie in "
            f"[0, {point_count}]"
        )
        return failures

    zero_limit = Fraction(str(max_zero_mask_support_ratio))
    max_zero_mask_support_count = (
        zero_limit.numerator * point_count
        + zero_limit.denominator
        - 1
    ) // zero_limit.denominator

    stats["max_zero_mask_support_count"] = int(max_zero_mask_support_count)
    stats["zero_mask_support_gate_policy"] = (
        "ceil_ratio_times_point_count_v1"
    )

    if zero_mask_support_count > max_zero_mask_support_count:
        failures.append(
            f"zero_mask_support_count={zero_mask_support_count} "
            f"must be <= {max_zero_mask_support_count}; "
            f"zero_mask_support_ratio={zero_mask_support_ratio:.6f}; "
            f"configured_ratio={float(max_zero_mask_support_ratio):.6f}; "
            f"policy=ceil_ratio_times_point_count_v1"
        )

    return failures


@torch.no_grad()
def fuse_dinov2_features(
    *,
    model: torch.nn.Module,
    points_render: np.ndarray,
    frames: list[dict[str, Any]],
    device: torch.device,
    image_size: int,
    view_batch_size: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if not frames:
        raise ValueError("DINO fusion requires at least one view")
    points = torch.from_numpy(points_render).to(device=device, dtype=torch.float32)
    feature_sum: torch.Tensor | None = None
    valid_count = torch.zeros((len(points),), device=device, dtype=torch.float32)
    mask_count = torch.zeros_like(valid_count)
    corrupted_mask_count = torch.zeros_like(valid_count)
    patch_side = image_size // 14
    corruption_partners = camera_corruption_partner_indices(frames)
    for start in range(0, len(frames), view_batch_size):
        batch_frames = frames[start : start + view_batch_size]
        loaded = [load_image_pair(frame, image_size=image_size) for frame in batch_frames]
        images = torch.stack([item[0] for item in loaded]).to(device=device)
        masks = torch.stack([item[1] for item in loaded]).to(device=device)
        output = model(images, is_training=True)
        tokens = output["x_prenorm"]
        register_count = int(getattr(model, "num_register_tokens", 0))
        patch_tokens = tokens[:, 1 + register_count :]
        if patch_tokens.shape[1] != patch_side * patch_side:
            raise RuntimeError(
                f"DINO patch count={patch_tokens.shape[1]} != {patch_side**2}"
            )
        patch_maps = patch_tokens.permute(0, 2, 1).reshape(
            len(batch_frames), patch_tokens.shape[-1], patch_side, patch_side
        )
        intrinsics = torch.tensor(
            [frame["intrinsic"] for frame in batch_frames],
            device=device,
            dtype=torch.float32,
        )
        c2w = torch.tensor(
            [frame["c2w"] for frame in batch_frames],
            device=device,
            dtype=torch.float32,
        )
        source_sizes = {int(frame["source_size"]) for frame in batch_frames}
        forward_signs = {
            float(frame["camera_forward_sign"]) for frame in batch_frames
        }
        if len(source_sizes) != 1 or len(forward_signs) != 1:
            raise RuntimeError("mixed camera/image conventions within DINO batch")
        grid, valid = project_points_to_grid(
            points,
            intrinsics,
            c2w,
            source_height=next(iter(source_sizes)),
            source_width=next(iter(source_sizes)),
            target_height=image_size,
            target_width=image_size,
            camera_forward_sign=next(iter(forward_signs)),
        )
        sampled = grid_sample_with_validity(patch_maps, grid, valid)
        current = sampled.sum(dim=0)
        feature_sum = current if feature_sum is None else feature_sum + current
        valid_count += valid.float().sum(dim=0)
        sampled_mask = F.grid_sample(
            masks.float(),
            grid[:, :, None, :],
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        ).squeeze(1).squeeze(-1)
        mask_count += ((sampled_mask > 0.5) & valid).float().sum(dim=0)
        corrupted_frames = [
            frames[corruption_partners[index]]
            for index in range(start, start + len(batch_frames))
        ]
        corrupted_c2w = torch.tensor(
            [frame["c2w"] for frame in corrupted_frames],
            device=device,
            dtype=torch.float32,
        )
        corrupted_grid, corrupted_valid = project_points_to_grid(
            points,
            intrinsics,
            corrupted_c2w,
            source_height=next(iter(source_sizes)),
            source_width=next(iter(source_sizes)),
            target_height=image_size,
            target_width=image_size,
            camera_forward_sign=next(iter(forward_signs)),
        )
        corrupted_sampled_mask = F.grid_sample(
            masks.float(),
            corrupted_grid[:, :, None, :],
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        ).squeeze(1).squeeze(-1)
        corrupted_mask_count += (
            (corrupted_sampled_mask > 0.5) & corrupted_valid
        ).float().sum(dim=0)
        del images, masks, output, tokens, patch_tokens, patch_maps, sampled
    if feature_sum is None:
        raise RuntimeError("DINO fusion produced no features")
    # Match the released TRELLIS toolkit: average every view, including zero
    # padding for projections outside the image.
    fused = feature_sum / float(len(frames))
    view_count = float(len(frames))
    correct_mask_fraction = mask_count / view_count
    corrupted_mask_fraction = corrupted_mask_count / view_count
    zero_mask_support_count = int(
        (mask_count == 0).sum().item()
    )
    point_count = int(len(points))
    zero_mask_support_ratio = (
        float(zero_mask_support_count / point_count)
        if point_count > 0
        else 1.0
    )

    stats = {
        "view_count": len(frames),
        "point_count": point_count,
        "feature_channels": int(fused.shape[1]),
        "visible_view_fraction_mean": float(
            (valid_count / view_count).mean().item()
        ),
        "mask_support_view_fraction_mean": float(
            correct_mask_fraction.mean().item()
        ),
        "visible_ge1_ratio": float((valid_count >= 1).float().mean().item()),
        "visible_ge2_ratio": float((valid_count >= 2).float().mean().item()),
        "visible_ge4_ratio": float((valid_count >= 4).float().mean().item()),
        "mask_support_ge1_ratio": float((mask_count >= 1).float().mean().item()),
        "mask_support_ge2_ratio": float((mask_count >= 2).float().mean().item()),
        "mask_support_ge4_ratio": float((mask_count >= 4).float().mean().item()),
        "zero_mask_support_count": zero_mask_support_count,
        "zero_mask_support_ratio": zero_mask_support_ratio,
        "camera_corruption": "within-sequence deterministic half-turn c2w shift",
        "camera_corruption_mask_support_view_fraction_mean": float(
            corrupted_mask_fraction.mean().item()
        ),
        "camera_corruption_zero_mask_support_ratio": float(
            (corrupted_mask_count == 0).float().mean().item()
        ),
        "camera_corruption_mask_support_drop": float(
            (correct_mask_fraction.mean() - corrupted_mask_fraction.mean()).item()
        ),
        "feature_abs_mean": float(fused.abs().mean().item()),
        "feature_abs_max": float(fused.abs().amax().item()),
    }
    if fused.shape != (len(points), 1024) or not torch.isfinite(fused).all():
        raise RuntimeError(f"invalid fused DINO features: {tuple(fused.shape)}")
    return fused, stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render_manifests", required=True)
    parser.add_argument("--lifting_manifests", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--encoder_prefix", required=True)
    parser.add_argument("--mesh_decoder_weights", required=True)
    parser.add_argument(
        "--expected_encoder_sha256", default=OFFICIAL_SLAT_ENCODER_SHA256
    )
    parser.add_argument(
        "--expected_dinov2_sha256", default=DINOV2_VITL14_REG4_SHA256
    )
    parser.add_argument(
        "--dinov2_repo",
        default="/home/zjr/.cache/torch/hub/facebookresearch_dinov2_main",
    )
    parser.add_argument(
        "--dinov2_checkpoint",
        default=(
            "/home/zjr/.cache/torch/hub/checkpoints/"
            "dinov2_vitl14_reg4_pretrain.pth"
        ),
    )
    parser.add_argument("--dinov2_model", default="dinov2_vitl14_reg")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image_size", type=int, default=518)
    parser.add_argument("--view_batch_size", type=int, default=2)
    parser.add_argument("--min_views", type=int, default=8)
    parser.add_argument("--max_views", type=int, default=16)
    parser.add_argument("--min_visible_view_fraction_mean", type=float, default=0.25)
    parser.add_argument(
        "--min_mask_support_view_fraction_mean", type=float, default=0.10
    )
    parser.add_argument("--min_mask_support_ge2_ratio", type=float, default=0.50)
    parser.add_argument("--min_mask_support_ge4_ratio", type=float, default=0.20)
    parser.add_argument("--max_zero_mask_support_ratio", type=float, default=0.25)
    parser.add_argument(
        "--min_camera_corruption_mask_support_drop_mean", type=float, default=0.05
    )
    parser.add_argument(
        "--min_camera_corruption_mask_support_drop_median", type=float, default=0.02
    )
    parser.add_argument(
        "--min_camera_corruption_positive_object_rate", type=float, default=0.70
    )
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--max_objects", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    render_manifests = parse_csv(args.render_manifests)
    lifting_manifests = parse_csv(args.lifting_manifests)
    if int(args.image_size) != 518:
        raise ValueError("official DINOv2 ViT-L/14 feature path requires image_size=518")
    if int(args.view_batch_size) <= 0 or int(args.min_views) <= 0:
        raise ValueError("view batch/minimum sizes must be positive")
    if int(args.world_size) <= 0 or not 0 <= int(args.rank) < int(args.world_size):
        raise ValueError("rank must lie in [0, world_size)")
    projection_threshold_names = (
        "min_visible_view_fraction_mean",
        "min_mask_support_view_fraction_mean",
        "min_mask_support_ge2_ratio",
        "min_mask_support_ge4_ratio",
        "max_zero_mask_support_ratio",
        "min_camera_corruption_mask_support_drop_mean",
        "min_camera_corruption_mask_support_drop_median",
        "min_camera_corruption_positive_object_rate",
    )
    for name in projection_threshold_names:
        if not 0.0 <= float(getattr(args, name)) <= 1.0:
            raise ValueError(f"{name} must lie in [0,1]")
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("SLat encoding requires CUDA")
    encoder_prefix = Path(args.encoder_prefix).resolve()
    encoder_json = Path(f"{encoder_prefix}.json")
    encoder_weights = Path(f"{encoder_prefix}.safetensors")
    mesh_decoder_weights = Path(args.mesh_decoder_weights).resolve()
    dinov2_repo = Path(args.dinov2_repo).resolve()
    dinov2_checkpoint = Path(args.dinov2_checkpoint).resolve()
    for path in (
        encoder_json,
        encoder_weights,
        mesh_decoder_weights,
        dinov2_repo,
        dinov2_checkpoint,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    encoder_weights_sha256 = sha256_file(encoder_weights)
    dinov2_checkpoint_sha256 = sha256_file(dinov2_checkpoint)
    if encoder_weights_sha256 != str(args.expected_encoder_sha256):
        raise RuntimeError(
            "official SLatEncoder SHA mismatch: "
            f"{encoder_weights_sha256} != {args.expected_encoder_sha256}"
        )
    if dinov2_checkpoint_sha256 != str(args.expected_dinov2_sha256):
        raise RuntimeError(
            "DINOv2 ViT-L/14 register SHA mismatch: "
            f"{dinov2_checkpoint_sha256} != {args.expected_dinov2_sha256}"
        )

    required_objects = load_required_objects(lifting_manifests)
    render_index, render_bindings = index_render_samples(render_manifests)
    missing = sorted(required_objects - set(render_index))
    if missing:
        raise RuntimeError(
            f"render manifests miss {len(missing)} required objects; first={missing[:8]}"
        )
    objects = sorted(required_objects)
    if int(args.max_objects) > 0:
        objects = objects[: int(args.max_objects)]
    start = len(objects) * int(args.rank) // int(args.world_size)
    end = len(objects) * (int(args.rank) + 1) // int(args.world_size)
    assigned = objects[start:end]
    output_dir = Path(args.output_dir).resolve()
    config = {
        "format": LOCAL_LH_SLAT_VERSION,
        "source_kind": (
            "local limited-view TRELLIS-compatible rebuild; not the original "
            "ProObjaverse 150-view latent release"
        ),
        "render_manifests": render_bindings,
        "lifting_manifests": [
            {"path": str(Path(path).resolve()), "sha256": sha256_file(path)}
            for path in lifting_manifests
        ],
        "required_object_count": len(required_objects),
        "encoder_prefix": str(encoder_prefix),
        "encoder_config_sha256": sha256_file(encoder_json),
        "encoder_weights_sha256": encoder_weights_sha256,
        "mesh_decoder_weights": str(mesh_decoder_weights),
        "mesh_decoder_weights_sha256": sha256_file(mesh_decoder_weights),
        "dinov2_repo": str(dinov2_repo),
        "dinov2_hubconf_sha256": sha256_file(dinov2_repo / "hubconf.py"),
        "dinov2_checkpoint": str(dinov2_checkpoint),
        "dinov2_checkpoint_sha256": dinov2_checkpoint_sha256,
        "dinov2_model": args.dinov2_model,
        "image_size": int(args.image_size),
        "view_batch_size": int(args.view_batch_size),
        "min_views": int(args.min_views),
        "max_views": int(args.max_views),
        "min_visible_view_fraction_mean": float(
            args.min_visible_view_fraction_mean
        ),
        "min_mask_support_view_fraction_mean": float(
            args.min_mask_support_view_fraction_mean
        ),
        "min_mask_support_ge2_ratio": float(args.min_mask_support_ge2_ratio),
        "min_mask_support_ge4_ratio": float(args.min_mask_support_ge4_ratio),
        "max_zero_mask_support_ratio": float(args.max_zero_mask_support_ratio),
        "min_camera_corruption_mask_support_drop_mean": float(
            args.min_camera_corruption_mask_support_drop_mean
        ),
        "min_camera_corruption_mask_support_drop_median": float(
            args.min_camera_corruption_mask_support_drop_median
        ),
        "min_camera_corruption_positive_object_rate": float(
            args.min_camera_corruption_positive_object_rate
        ),
        "coordinate_source": "repaired SS target_coords in pixal3d latent frame",
        "feature_fusion": "official-style all-view arithmetic mean with zero padding",
        "posterior": "SLatEncoder mean (sample_posterior=False)",
    }
    config_hash = canonical_json_sha256(config)
    run_binding = {"config": config, "config_hash": config_hash}
    output_existed = output_dir.exists()
    output_dir.mkdir(parents=True, exist_ok=True)
    run_config_path = output_dir / "run_config.json"
    if run_config_path.is_file():
        existing = json.loads(run_config_path.read_text(encoding="utf-8"))
        if existing != run_binding:
            raise RuntimeError("local lh-slats resume config/source binding changed")
    elif output_existed and any(output_dir.iterdir()):
        raise RuntimeError("refusing to use an unbound local lh-slats directory")
    else:
        atomic_json(run_config_path, run_binding)
    if not args.resume and any(output_dir.glob("shard-*/*.npz")):
        raise FileExistsError("output already contains local lh-slats; pass --resume")

    torch.cuda.set_device(0 if device.index is None else int(device.index))
    dinov2 = torch.hub.load(
        str(dinov2_repo),
        args.dinov2_model,
        source="local",
        pretrained=True,
        weights=str(dinov2_checkpoint),
    ).to(device).eval()
    import trellis.models as models
    from trellis.modules import sparse as sp

    encoder = models.from_pretrained(str(encoder_prefix)).to(device).eval()
    records: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for ordinal, object_uid in enumerate(assigned, start=1):
        output_path = (
            output_dir / f"shard-{object_uid[:2]}" / f"{object_uid}.npz"
        )
        try:
            if output_path.is_file():
                with np.load(output_path) as existing:
                    coords_existing = np.asarray(existing["coords"])
                    feats_existing = np.asarray(existing["feats"])
                    fusion_stats = json.loads(str(existing["fusion_stats_json"]))
                    if (
                        coords_existing.ndim != 2
                        or coords_existing.shape[1] != 3
                        or feats_existing.shape != (len(coords_existing), 8)
                        or not np.isfinite(feats_existing).all()
                        or str(existing.get("object_uid", "")) != object_uid
                        or str(existing.get("config_hash", "")) != config_hash
                    ):
                        raise RuntimeError(f"invalid resumed latent: {output_path}")
                records.append(
                    {
                        "object_uid": object_uid,
                        "output": str(output_path),
                        "output_sha256": sha256_file(output_path),
                        "point_count": int(len(coords_existing)),
                        "fusion_stats": fusion_stats,
                        "resumed": True,
                    }
                )
                print(
                    f"[local_lh_slats] {ordinal}/{len(assigned)} {object_uid} resume",
                    flush=True,
                )
                continue
            resolved = object_inputs(
                object_uid, render_index[object_uid], max_views=int(args.max_views)
            )
            frames = resolved["frames"]
            if len(frames) < int(args.min_views):
                raise RuntimeError(
                    f"object={object_uid} has {len(frames)} views < {args.min_views}"
                )
            points_render = voxel_centers_render_space(
                resolved["coords"], resolved["rotation"], resolution=64
            )
            fused, fusion_stats = fuse_dinov2_features(
                model=dinov2,
                points_render=points_render,
                frames=frames,
                device=device,
                image_size=int(args.image_size),
                view_batch_size=int(args.view_batch_size),
            )
            projection_failures = projection_gate_failures(
                fusion_stats,
                min_visible_view_fraction_mean=args.min_visible_view_fraction_mean,
                min_mask_support_view_fraction_mean=(
                    args.min_mask_support_view_fraction_mean
                ),
                min_mask_support_ge2_ratio=args.min_mask_support_ge2_ratio,
                min_mask_support_ge4_ratio=args.min_mask_support_ge4_ratio,
                max_zero_mask_support_ratio=args.max_zero_mask_support_ratio,
            )
            if projection_failures:
                raise RuntimeError(
                    f"object={object_uid} projection hard gate failed: "
                    + "; ".join(projection_failures)
                )
            coords4 = torch.cat(
                [
                    torch.zeros((len(resolved["coords"]), 1), dtype=torch.int32),
                    torch.from_numpy(resolved["coords"].astype(np.int32)),
                ],
                dim=1,
            ).to(device=device)
            sparse_features = sp.SparseTensor(
                # Match the official TRELLIS encode_latent.py contract: the
                # sparse patch-token input remains float32.  The encoder
                # performs its own internal fp16 conversion where configured.
                feats=fused.float(), coords=coords4
            )
            latent = encoder(sparse_features, sample_posterior=False)
            output_coords = latent.coords[:, 1:].detach().cpu().numpy().astype(np.uint8)
            output_feats = latent.feats.detach().float().cpu().numpy().astype(np.float32)
            if (
                output_feats.shape != (len(output_coords), 8)
                or not np.isfinite(output_feats).all()
            ):
                raise RuntimeError("SLatEncoder returned invalid latent")
            frame_identity = [
                {
                    "image": frame["image_path"],
                    "image_sha256": sha256_file(frame["image_path"]),
                    "mask": frame["mask_path"],
                    "mask_sha256": sha256_file(frame["mask_path"]),
                }
                for frame in frames
            ]
            source_identity = {
                "object_uid": object_uid,
                "source_glb": resolved["source_glb"],
                "source_glb_sha256": sha256_file(resolved["source_glb"]),
                "source_latents": resolved["source_latents"],
                "frames": frame_identity,
            }
            source_identity_hash = canonical_json_sha256(source_identity)
            atomic_savez(
                output_path,
                coords=output_coords,
                feats=output_feats,
                object_uid=np.array(object_uid),
                config_hash=np.array(config_hash),
                source_identity_hash=np.array(source_identity_hash),
                fusion_stats_json=np.array(json.dumps(fusion_stats, sort_keys=True)),
            )
            records.append(
                {
                    "object_uid": object_uid,
                    "output": str(output_path),
                    "output_sha256": sha256_file(output_path),
                    "point_count": int(len(output_coords)),
                    **source_identity,
                    "source_identity_hash": source_identity_hash,
                    "fusion_stats": fusion_stats,
                    "resumed": False,
                }
            )
            print(
                f"[local_lh_slats] {ordinal}/{len(assigned)} {object_uid} "
                f"views={len(frames)} points={len(output_coords)}",
                flush=True,
            )
            del fused, sparse_features, latent
            torch.cuda.empty_cache()
        except Exception as error:
            failures.append({"object_uid": object_uid, "error": repr(error)})
            print(
                f"[local_lh_slats:FAIL] {ordinal}/{len(assigned)} "
                f"{object_uid}: {error}",
                flush=True,
            )

    corruption_drops = np.asarray(
        [
            float(row["fusion_stats"]["camera_corruption_mask_support_drop"])
            for row in records
        ],
        dtype=np.float64,
    )
    corruption_summary = {
        "object_count": int(len(corruption_drops)),
        "mean": (
            float(corruption_drops.mean()) if len(corruption_drops) else None
        ),
        "median": (
            float(np.median(corruption_drops)) if len(corruption_drops) else None
        ),
        "positive_object_rate": (
            float((corruption_drops > 0.0).mean())
            if len(corruption_drops)
            else None
        ),
    }
    corruption_gate_failures: list[str] = []
    if len(corruption_drops):
        corruption_checks = (
            (
                "mean",
                corruption_summary["mean"],
                float(args.min_camera_corruption_mask_support_drop_mean),
            ),
            (
                "median",
                corruption_summary["median"],
                float(args.min_camera_corruption_mask_support_drop_median),
            ),
            (
                "positive_object_rate",
                corruption_summary["positive_object_rate"],
                float(args.min_camera_corruption_positive_object_rate),
            ),
        )
        for name, actual, threshold in corruption_checks:
            if float(actual) < threshold:
                corruption_gate_failures.append(
                    f"{name}={float(actual):.6f} must be >= {threshold:.6f}"
                )
    else:
        corruption_gate_failures.append("no successful objects to audit")
    rank_passed = (
        len(records) == len(assigned)
        and not failures
        and not corruption_gate_failures
    )
    rank_report = {
        "format": LOCAL_LH_SLAT_VERSION,
        "config_hash": config_hash,
        "rank": int(args.rank),
        "world_size": int(args.world_size),
        "assigned_object_count": len(assigned),
        "success_count": len(records),
        "failure_count": len(failures),
        "projection_camera_corruption_summary": corruption_summary,
        "projection_camera_corruption_gate_failures": corruption_gate_failures,
        "passed": rank_passed,
        "records": records,
        "failures": failures,
    }
    atomic_json(output_dir / f"rank_{int(args.rank):03d}_report.json", rank_report)
    print(
        json.dumps(
            {
                "rank": args.rank,
                "assigned": len(assigned),
                "success": len(records),
                "failures": len(failures),
                "passed": rank_report["passed"],
            },
            indent=2,
        ),
        flush=True,
    )
    if not rank_report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()