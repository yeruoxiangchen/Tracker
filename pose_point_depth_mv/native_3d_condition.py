from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Sequence


TRACKER_ROOT = Path(__file__).resolve().parents[1]
RECONVIAGEN_ROOT = TRACKER_ROOT / "ReconViaGen"
VGGT_ROOT = RECONVIAGEN_ROOT / "wheels" / "vggt"
for _path in (TRACKER_ROOT, RECONVIAGEN_ROOT, VGGT_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset

from ar_ss_flow.correspondence_lifting import pose_variant_extrinsics
from ar_ss_flow.local_pose_lifting_flow import PoseLiftingCacheDataset
from pose_point_depth_mv.direct_flow import (
    flow_tokens_to_volume_xyz,
    volume_xyz_to_flow_tokens,
)
from pose_point_depth_mv.direct_slat_data import DirectSLatCacheDataset
from pose_point_depth_mv.proobjaverse_official_slat_compact import (
    CompactNativeConditionBackend,
    is_compact_manifest_pair,
)


NATIVE_3D_CONDITION_VERSION = "pose_point_depth_mv.native_3d_condition.v1"
NATIVE_SS_FLOW_VERSION_V1 = "pose_point_depth_mv.native_ss_every_block.v1"
NATIVE_SS_FLOW_VERSION = "pose_point_depth_mv.native_ss_post_cfg.v2"
NATIVE_SLAT_FLOW_VERSION = "pose_point_depth_mv.native_slat_every_block.v1"
NATIVE_SS_TRAINING_SEMANTICS = "post_cfg_bounded_v2"
NATIVE_SS_GUIDED_DELTA_POLICY = "post_cfg_v2"
NATIVE_SS_DELTA_BOUND_SMOOTH = "smooth_rms_v1"
NATIVE_SS_SUPPORT_INTERVAL_CFG_ACTIVE = "cfg_active_only_v1"
NATIVE_FEATURE_SOURCES = ("dino", "vggt", "all")
NATIVE_CONTROL_MODES = (
    "correct",
    "pose_cyclic1",
    "pose_cyclic2",
    "pose_reverse",
    "depth_corrupt",
    "visual_view_cyclic1",
)
NATIVE_GEOMETRY_NAMES = (
    "frustum_valid",
    "mask_weight",
    "confidence_weight",
    "depth_weight",
    "normalized_depth_residual",
    "combined_weight",
    "ray_x",
    "ray_y",
    "ray_z",
    "normalized_camera_depth",
)


def normalize_native_cfg_interval(
    cfg_interval: Iterable[float],
) -> tuple[float, float]:
    interval = tuple(float(value) for value in cfg_interval)
    if (
        len(interval) != 2
        or not all(math.isfinite(value) for value in interval)
        or not 0.0 <= interval[0] <= interval[1] <= 1.0
    ):
        raise ValueError("native CFG interval must satisfy finite 0 <= lo <= hi <= 1")
    return interval


def native_ss_cfg_is_active(
    t_value: float, cfg_interval: Iterable[float]
) -> bool:
    lower, upper = normalize_native_cfg_interval(cfg_interval)
    value = float(t_value)
    if not math.isfinite(value):
        raise ValueError("native SS timestep must be finite")
    return lower <= value <= upper


def native_ss_timestep_sequence(
    *, steps: int = 25, rescale_t: float = 3.0
) -> tuple[float, ...]:
    count = int(steps)
    rescale = float(rescale_t)
    if count <= 0:
        raise ValueError("native SS schedule steps must be positive")
    if not math.isfinite(rescale) or rescale <= 0.0:
        raise ValueError("native SS rescale_t must be finite and positive")
    base = torch.linspace(1.0, 0.0, count + 1, dtype=torch.float64)
    transformed = rescale * base / (1.0 + (rescale - 1.0) * base)
    return tuple(float(value) for value in transformed.tolist())


def combine_dense_cfg(
    positive: torch.Tensor,
    negative: torch.Tensor,
    *,
    cfg_strength: float,
) -> torch.Tensor:
    if positive.shape != negative.shape:
        raise ValueError(
            "positive/negative SS predictions differ in shape: "
            f"{tuple(positive.shape)} != {tuple(negative.shape)}"
        )
    strength = float(cfg_strength)
    if not math.isfinite(strength) or strength < 0.0:
        raise ValueError("native SS CFG strength must be finite and non-negative")
    return strength * positive + (1.0 - strength) * negative


def _zero_safe_sqrt(value: torch.Tensor, *, eps: float) -> torch.Tensor:
    positive = value > 0
    safe = torch.where(positive, value, torch.full_like(value, float(eps)))
    return torch.where(positive, safe.sqrt(), torch.zeros_like(value))


def bounded_native_ss_flow_delta(
    stock: torch.Tensor,
    raw_full: torch.Tensor,
    *,
    delta_scale: float = 1.0,
    delta_rms_ratio_cap: float = 0.10,
    eps: float = 1.0e-12,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Smoothly bound the final post-CFG residual per dense batch item."""

    if stock.shape != raw_full.shape or stock.ndim < 2:
        raise ValueError(
            "stock/raw Native SS predictions require equal rank>=2 shapes, got "
            f"{tuple(stock.shape)} and {tuple(raw_full.shape)}"
        )
    scale = float(delta_scale)
    cap = float(delta_rms_ratio_cap)
    if not math.isfinite(scale) or scale < 0.0:
        raise ValueError("native SS delta scale must be finite and non-negative")
    if not math.isfinite(cap) or cap <= 0.0:
        raise ValueError("native SS delta RMS ratio cap must be finite and positive")
    if not math.isfinite(float(eps)) or float(eps) <= 0.0:
        raise ValueError("native SS delta epsilon must be finite and positive")

    reduce_dims = tuple(range(1, stock.ndim))
    stock32 = stock.float()
    raw_delta = raw_full.float() - stock32
    stock_rms = _zero_safe_sqrt(
        stock32.square().mean(dim=reduce_dims), eps=float(eps)
    )
    raw_rms = _zero_safe_sqrt(
        raw_delta.square().mean(dim=reduce_dims), eps=float(eps)
    )
    allowed = cap * stock_rms.clamp_min(float(eps))
    denominator = torch.sqrt(raw_rms.square() + allowed.square()).clamp_min(
        float(eps)
    )
    smooth_scale = torch.where(
        raw_rms > float(eps), allowed / denominator, torch.ones_like(raw_rms)
    )
    view_shape = (stock.shape[0],) + (1,) * (stock.ndim - 1)
    effective_delta = raw_delta * smooth_scale.reshape(view_shape) * scale
    if scale == 0.0:
        effective = stock
        effective_delta = raw_delta * 0.0
    else:
        effective = (stock32 + effective_delta).to(dtype=raw_full.dtype)
    effective_rms = _zero_safe_sqrt(
        effective_delta.square().mean(dim=reduce_dims), eps=float(eps)
    )
    raw_ratio = raw_rms / stock_rms.clamp_min(float(eps))
    effective_ratio = effective_rms / stock_rms.clamp_min(float(eps))
    saturated = raw_rms > allowed
    stats = {
        "stock_velocity_rms": stock_rms.mean(),
        "raw_flow_delta_rms": raw_rms.mean(),
        "effective_flow_delta_rms": effective_rms.mean(),
        "raw_flow_delta_ratio": raw_ratio.mean(),
        "raw_flow_delta_ratio_max": raw_ratio.amax(),
        "effective_flow_delta_ratio": effective_ratio.mean(),
        "effective_flow_delta_ratio_max": effective_ratio.amax(),
        "delta_clip_scale": smooth_scale.amin(),
        "delta_clip_scale_mean": smooth_scale.mean(),
        "delta_clip_activated": saturated.any().to(torch.float32),
        "raw_flow_delta_abs_max": raw_delta.abs().amax(),
        "effective_flow_delta_abs_max": effective_delta.abs().amax(),
        "delta_scale": raw_delta.new_tensor(scale),
        "delta_rms_ratio_cap": raw_delta.new_tensor(cap),
        "stock_velocity_rms_per_batch": stock_rms,
        "raw_flow_delta_rms_per_batch": raw_rms,
        "effective_flow_delta_rms_per_batch": effective_rms,
        "raw_flow_delta_ratio_per_batch": raw_ratio,
        "effective_flow_delta_ratio_per_batch": effective_ratio,
        "delta_clip_scale_per_batch": smooth_scale,
        "delta_clip_activated_per_batch": saturated.to(torch.float32),
    }
    return effective, stats


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()


def native_feature_slice(
    feature_dim: int, source: str
) -> tuple[slice, int, dict[str, int | str]]:
    source = str(source)
    feature_dim = int(feature_dim)
    if source not in NATIVE_FEATURE_SOURCES:
        raise ValueError(f"unsupported native feature source={source!r}")
    if source == "dino":
        if feature_dim < 1024:
            raise ValueError(f"DINO feature source requires >=1024 channels, got {feature_dim}")
        selected = slice(feature_dim - 1024, feature_dim)
        channels = 1024
    elif source == "vggt":
        if feature_dim < 2048:
            raise ValueError(f"VGGT feature source requires >=2048 channels, got {feature_dim}")
        selected = slice(0, 2048)
        channels = 2048
    else:
        selected = slice(0, feature_dim)
        channels = feature_dim
    return selected, channels, {
        "source": source,
        "input_channels": feature_dim,
        "start": int(selected.start or 0),
        "end": int(selected.stop or feature_dim),
        "selected_channels": channels,
    }


def validate_lifting_feature_metadata(
    *,
    visual_feature_dim: int,
    feature_metadata: dict[str, Any],
    feature_source: str,
) -> dict[str, Any]:
    vggt_dim = int(feature_metadata.get("vggt_feature_dim", -1))
    dino_dim = int(feature_metadata.get("dino_feature_dim", -1))
    if vggt_dim <= 0 or dino_dim <= 0:
        raise ValueError("lifting manifest lacks explicit VGGT/DINO feature dimensions")
    if vggt_dim + dino_dim != int(visual_feature_dim):
        raise ValueError(
            "lifting feature dimension does not equal VGGT+DINO metadata: "
            f"{visual_feature_dim} != {vggt_dim}+{dino_dim}"
        )
    if dino_dim != 1024 or vggt_dim != 2048:
        raise ValueError(
            f"unsupported lifting feature split VGGT={vggt_dim} DINO={dino_dim}"
        )
    _, selected_channels, selected = native_feature_slice(
        int(visual_feature_dim), str(feature_source)
    )
    return {
        **selected,
        "layout": "VGGT_then_DINO",
        "vggt_feature_dim": vggt_dim,
        "dino_feature_dim": dino_dim,
        "selected_channels": selected_channels,
        "patch_count": int(feature_metadata.get("patch_count", -1)),
    }


def _canonical_points(
    coords: torch.Tensor,
    *,
    resolution: int,
    grid_transform: str,
) -> torch.Tensor:
    if coords.ndim != 2 or int(coords.shape[1]) not in (3, 4):
        raise ValueError(f"coords must be [N,3/4], got {tuple(coords.shape)}")
    if int(resolution) <= 0:
        raise ValueError("resolution must be positive")
    xyz = coords[:, -3:].float()
    if xyz.numel() and (
        bool((xyz < 0).any().item())
        or bool((xyz >= int(resolution)).any().item())
    ):
        raise ValueError(f"coordinates outside native resolution={resolution}")
    points = (xyz + 0.5) / float(resolution) - 0.5
    if grid_transform == "identity":
        return points
    if grid_transform == "pixal3d_rotation":
        rotation = points.new_tensor(
            ((1.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0))
        )
        return points @ rotation.transpose(0, 1)
    raise ValueError(f"unsupported grid_transform={grid_transform!r}")


def dense_native_coords(
    resolution: int,
    *,
    device: torch.device | None = None,
    with_batch: bool = True,
) -> torch.Tensor:
    axis = torch.arange(int(resolution), device=device, dtype=torch.int32)
    xyz = torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"), dim=-1)
    xyz = xyz.reshape(-1, 3)
    if not with_batch:
        return xyz
    return torch.cat(
        (torch.zeros((len(xyz), 1), device=device, dtype=torch.int32), xyz), dim=1
    )


def sparse_projection_geometry(
    *,
    coords: torch.Tensor,
    resolution: int,
    intrinsics: torch.Tensor,
    extrinsics: torch.Tensor,
    grid_transform: str,
    extrinsics_type: str,
    camera_forward_sign: float,
    image_height: int,
    image_width: int,
    patch_grid_side: int,
    object_to_world: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    if intrinsics.ndim != 3 or tuple(intrinsics.shape[1:]) != (3, 3):
        raise ValueError(f"intrinsics must be [V,3,3], got {tuple(intrinsics.shape)}")
    views = int(intrinsics.shape[0])
    if extrinsics.shape != (views, 4, 4):
        raise ValueError(f"extrinsics must be [{views},4,4], got {tuple(extrinsics.shape)}")
    if extrinsics_type not in ("c2w", "w2c"):
        raise ValueError(f"unsupported extrinsics_type={extrinsics_type!r}")
    points = _canonical_points(
        coords.to(device=intrinsics.device),
        resolution=int(resolution),
        grid_transform=str(grid_transform),
    ).float()
    if object_to_world is not None:
        transform = object_to_world.to(device=points.device, dtype=torch.float32)
        if transform.shape != (4, 4):
            raise ValueError("object_to_world must be [4,4]")
        points = points @ transform[:3, :3].transpose(0, 1) + transform[:3, 3]
    points_h = torch.cat(
        (points, torch.ones((len(points), 1), device=points.device)), dim=1
    )
    w2c = (
        torch.linalg.inv(extrinsics.float())
        if extrinsics_type == "c2w"
        else extrinsics.float()
    )
    camera = torch.einsum("vij,nj->vni", w2c, points_h)[..., :3]
    camera_depth = camera[..., 2] * float(camera_forward_sign)
    safe_depth = camera_depth.clamp_min(1.0e-6)
    intrinsics = intrinsics.float()
    u = (
        intrinsics[:, None, 0, 0] * camera[..., 0] / safe_depth
        + intrinsics[:, None, 0, 2]
    )
    v = (
        intrinsics[:, None, 1, 1] * camera[..., 1] / safe_depth
        + intrinsics[:, None, 1, 2]
    )
    valid = (
        (camera_depth > 1.0e-5)
        & (u >= 0.0)
        & (u <= float(image_width - 1))
        & (v >= 0.0)
        & (v <= float(image_height - 1))
    )
    image_grid = torch.stack(
        (
            2.0 * u / max(float(image_width - 1), 1.0) - 1.0,
            2.0 * v / max(float(image_height - 1), 1.0) - 1.0,
        ),
        dim=-1,
    )
    patch_u = (u + 0.5) / (float(image_width) / float(patch_grid_side)) - 0.5
    patch_v = (v + 0.5) / (float(image_height) / float(patch_grid_side)) - 0.5
    patch_grid = torch.stack(
        (
            2.0 * patch_u / max(float(patch_grid_side - 1), 1.0) - 1.0,
            2.0 * patch_v / max(float(patch_grid_side - 1), 1.0) - 1.0,
        ),
        dim=-1,
    )
    return {
        "points": points,
        "camera": camera,
        "camera_depth": camera_depth,
        "image_grid": image_grid,
        "patch_grid": patch_grid,
        "valid": valid,
    }


def _sample_maps(values: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    if values.ndim != 4 or grid.ndim != 3 or int(grid.shape[-1]) != 2:
        raise ValueError("map sampler expects values [V,C,H,W] and grid [V,N,2]")
    sampled = F.grid_sample(
        values,
        grid[:, :, None, :],
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    return sampled[..., 0]


def project_native_features(
    sample: dict[str, Any],
    coords: torch.Tensor,
    *,
    resolution: int,
    device: torch.device | None = None,
    mode: str = "correct",
    feature_source: str = "dino",
    object_to_world: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Project posed dense image features onto arbitrary native 3D coordinates."""

    if mode not in NATIVE_CONTROL_MODES:
        raise ValueError(f"unsupported native projection mode={mode!r}")
    visual_cpu = sample["visual_patch_features"]
    if visual_cpu.ndim != 3:
        raise ValueError("visual_patch_features must be [V,P,C]")
    target_device = device or coords.device
    visual = visual_cpu.to(device=target_device, dtype=torch.float32)
    selected, selected_channels, feature_identity = native_feature_slice(
        int(visual.shape[-1]), feature_source
    )
    visual = visual[..., selected]
    views, patch_count, channels = map(int, visual.shape)
    if channels != selected_channels:
        raise RuntimeError("native visual feature selection is inconsistent")
    patch_side = int(round(math.sqrt(patch_count)))
    if patch_side * patch_side != patch_count:
        raise ValueError(f"visual patch count is not square: {patch_count}")
    predicted_depth = sample["predicted_depth"].to(target_device).float()
    depth_confidence = sample["depth_confidence"].to(target_device).float()
    masks = sample["masks"].to(target_device).float()
    intrinsics = sample["intrinsics"].to(target_device).float()
    extrinsics = sample["extrinsics"].to(target_device).float()
    if predicted_depth.ndim != 3 or predicted_depth.shape != depth_confidence.shape:
        raise ValueError("depth/confidence must be aligned [V,H,W]")
    if masks.shape != predicted_depth.shape or int(predicted_depth.shape[0]) != views:
        raise ValueError("mask/depth/visual view count mismatch")
    if intrinsics.shape != (views, 3, 3) or extrinsics.shape != (views, 4, 4):
        raise ValueError("intrinsic/extrinsic/visual view count mismatch")

    geometry_mode = "correct" if mode in ("depth_corrupt", "visual_view_cyclic1") else mode
    variant_extrinsics = pose_variant_extrinsics(extrinsics, geometry_mode)
    height, width = map(int, predicted_depth.shape[-2:])
    geometry = sparse_projection_geometry(
        coords=coords,
        resolution=int(resolution),
        intrinsics=intrinsics,
        extrinsics=variant_extrinsics,
        grid_transform=str(sample["grid_transform"]),
        extrinsics_type=str(sample["extrinsics_type"]),
        camera_forward_sign=float(sample["camera_forward_sign"]),
        image_height=height,
        image_width=width,
        patch_grid_side=patch_side,
        object_to_world=object_to_world,
    )
    if mode == "visual_view_cyclic1" and views > 1:
        visual = torch.roll(visual, shifts=1, dims=0)
    patch_maps = visual.permute(0, 2, 1).reshape(
        views, selected_channels, patch_side, patch_side
    )
    sampled_visual = _sample_maps(patch_maps, geometry["patch_grid"].float())
    depth_maps = predicted_depth[:, None]
    if mode == "depth_corrupt":
        depth_maps = torch.flip(
            torch.roll(
                depth_maps,
                shifts=(max(1, height // 6), max(1, width // 5)),
                dims=(-2, -1),
            ),
            dims=(-1,),
        )
        view_scale = torch.linspace(
            0.8, 1.2, views, device=target_device
        ).reshape(-1, 1, 1, 1)
        depth_maps = depth_maps * view_scale * 1.15
    sampled_depth = _sample_maps(depth_maps, geometry["image_grid"].float())[:, 0]
    sampled_confidence = _sample_maps(
        depth_confidence[:, None], geometry["image_grid"].float()
    )[:, 0]
    sampled_mask = _sample_maps(
        masks[:, None], geometry["image_grid"].float()
    )[:, 0].clamp(0.0, 1.0)
    valid = geometry["valid"].float()
    mask_weight = sampled_mask * valid
    positive_confidence = sampled_confidence[sampled_confidence > 0]
    confidence_scale = (
        positive_confidence.median().clamp_min(1.0e-6)
        if positive_confidence.numel()
        else sampled_confidence.new_tensor(1.0)
    )
    confidence_weight = (sampled_confidence / confidence_scale).clamp(0.0, 1.0)
    calibration = dict(sample.get("depth_calibration", {}))
    if bool(calibration.get("enabled", False)):
        aligned_depth = (
            sampled_depth * float(calibration["scale"])
            + float(calibration["shift"])
        )
        residual = (aligned_depth - geometry["camera_depth"].float()).abs()
        tolerance = max(
            float(calibration.get("p90_abs_residual") or 0.0),
            float(calibration.get("minimum_depth_tolerance", 0.02)),
        )
        depth_weight = torch.exp(-0.5 * (residual / tolerance).square()) * valid
        normalized_residual = (residual / tolerance).clamp(0.0, 4.0) * 0.25
    else:
        depth_weight = valid
        normalized_residual = torch.zeros_like(valid)
    combined_weight = mask_weight * confidence_weight * depth_weight

    image_grid = geometry["image_grid"].float()
    u = (image_grid[..., 0] + 1.0) * max(float(width - 1), 1.0) * 0.5
    v = (image_grid[..., 1] + 1.0) * max(float(height - 1), 1.0) * 0.5
    ray_x = (u - intrinsics[:, None, 0, 2]) / intrinsics[
        :, None, 0, 0
    ].clamp_min(1.0e-6)
    ray_y = (v - intrinsics[:, None, 1, 2]) / intrinsics[
        :, None, 1, 1
    ].clamp_min(1.0e-6)
    ray_z = torch.full_like(ray_x, float(sample["camera_forward_sign"]))
    ray = F.normalize(torch.stack((ray_x, ray_y, ray_z), dim=-1), dim=-1)
    valid_depth = geometry["camera_depth"].abs() * valid
    positive_depth = valid_depth[valid_depth > 0]
    depth_scale = (
        positive_depth.median().clamp_min(1.0e-6)
        if positive_depth.numel()
        else valid_depth.new_tensor(1.0)
    )
    normalized_camera_depth = (
        (geometry["camera_depth"].abs() / depth_scale).clamp(0.0, 4.0) * 0.25
    )
    per_view_geometry = torch.stack(
        (
            valid,
            mask_weight,
            confidence_weight,
            depth_weight,
            normalized_residual,
            combined_weight,
            ray[..., 0],
            ray[..., 1],
            ray[..., 2],
            normalized_camera_depth,
        ),
        dim=-1,
    )
    projected_visual = sampled_visual.permute(0, 2, 1).contiguous()
    finite_values = (projected_visual, combined_weight, per_view_geometry)
    if not all(bool(torch.isfinite(value).all().item()) for value in finite_values):
        raise RuntimeError("native projected evidence contains non-finite values")
    return {
        "version": NATIVE_3D_CONDITION_VERSION,
        "mode": str(mode),
        "resolution": int(resolution),
        "coords": coords.to(device=target_device, dtype=torch.int32).contiguous(),
        "projected_visual": projected_visual,
        "per_view_geometry": per_view_geometry,
        "base_weight": combined_weight,
        "valid": geometry["valid"],
        "points": geometry["points"],
        "camera": geometry["camera"],
        "feature_identity": feature_identity,
        "view_count": views,
        "point_count": int(coords.shape[0]),
        "supported_point_count": int((combined_weight.sum(dim=0) > 0).sum().item()),
    }


def cached_project_native_features(
    sample: dict[str, Any],
    coords: torch.Tensor,
    *,
    resolution: int,
    device: torch.device,
    mode: str,
    feature_source: str,
) -> dict[str, Any]:
    """Reuse projection only when every coordinate and protocol field matches."""

    cache = sample.setdefault("_native_projection_cache_v1", {})
    if not isinstance(cache, dict):
        raise ValueError("native projection cache field was overwritten")
    key = (
        int(resolution),
        str(mode),
        str(feature_source),
        str(device),
    )
    cached = cache.get(key)
    coords_on_device = coords.to(device=device, dtype=torch.int32)
    if isinstance(cached, dict) and torch.is_tensor(cached.get("coords")):
        if torch.equal(cached["coords"], coords_on_device):
            return cached
    projected = project_native_features(
        sample,
        coords_on_device,
        resolution=int(resolution),
        device=device,
        mode=str(mode),
        feature_source=str(feature_source),
    )
    cache[key] = projected
    return projected


def drop_cached_native_projection(
    sample: dict[str, Any], *, mode: str | None = None
) -> None:
    cache = sample.get("_native_projection_cache_v1")
    if not isinstance(cache, dict):
        return
    if mode is None:
        cache.clear()
        return
    for key in list(cache):
        if isinstance(key, tuple) and len(key) >= 2 and key[1] == str(mode):
            cache.pop(key, None)


class NativeViewAggregator(nn.Module):
    """Permutation-invariant learned aggregation of per-view projected evidence."""

    def __init__(
        self,
        *,
        visual_channels: int,
        hidden_dim: int,
        geometry_channels: int = len(NATIVE_GEOMETRY_NAMES),
    ) -> None:
        super().__init__()
        self.visual_channels = int(visual_channels)
        self.hidden_dim = int(hidden_dim)
        self.geometry_channels = int(geometry_channels)
        self.visual_projection = nn.Linear(self.visual_channels, self.hidden_dim)
        self.geometry_projection = nn.Linear(self.geometry_channels, self.hidden_dim)
        self.view_fusion = nn.Sequential(
            nn.Linear(2 * self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
        )
        self.view_logit_residual = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, 1),
        )
        nn.init.zeros_(self.view_logit_residual[-1].weight)
        nn.init.zeros_(self.view_logit_residual[-1].bias)
        self.side_projection = nn.Sequential(
            nn.Linear(4, self.hidden_dim),
            nn.SiLU(),
        )
        self.output = nn.Sequential(
            nn.Linear(3 * self.hidden_dim, 2 * self.hidden_dim),
            nn.SiLU(),
            nn.Linear(2 * self.hidden_dim, self.hidden_dim),
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "version": NATIVE_3D_CONDITION_VERSION,
            "aggregation": "learned_residual_softmax_weighted_mean_variance.v1",
            "view_permutation_invariant": True,
            "visual_channels": self.visual_channels,
            "geometry_channels": self.geometry_channels,
            "geometry_names": list(NATIVE_GEOMETRY_NAMES),
            "hidden_dim": self.hidden_dim,
        }

    def forward(
        self,
        projected_visual: torch.Tensor,
        per_view_geometry: torch.Tensor,
        base_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if projected_visual.ndim != 3:
            raise ValueError("projected_visual must be [V,N,C]")
        views, points, channels = map(int, projected_visual.shape)
        if channels != self.visual_channels:
            raise ValueError(
                f"projected visual channels={channels} != {self.visual_channels}"
            )
        if per_view_geometry.shape != (views, points, self.geometry_channels):
            raise ValueError("per-view geometry shape does not match projected visual")
        if base_weight.shape != (views, points):
            raise ValueError("base weight shape does not match projected visual")
        visual_hidden = self.visual_projection(projected_visual.float())
        geometry_hidden = self.geometry_projection(per_view_geometry.float())
        per_view = self.view_fusion(torch.cat((visual_hidden, geometry_hidden), dim=-1))
        residual_logits = self.view_logit_residual(per_view)[..., 0]
        valid = base_weight.float() > 0
        logits = base_weight.float().clamp_min(1.0e-8).log() + residual_logits
        logits = logits.masked_fill(~valid, -1.0e4)
        logits = logits - logits.amax(dim=0, keepdim=True)
        unnormalized = torch.exp(logits) * valid.float()
        denominator = unnormalized.sum(dim=0, keepdim=True).clamp_min(1.0e-8)
        learned_weight = unnormalized / denominator
        mean = (per_view * learned_weight[..., None]).sum(dim=0)
        variance = (
            (per_view - mean[None]).square() * learned_weight[..., None]
        ).sum(dim=0)
        support_count = valid.float().sum(dim=0)
        weight_sum = base_weight.float().sum(dim=0)
        max_weight = base_weight.float().amax(dim=0)
        entropy = -(
            learned_weight.clamp_min(1.0e-8).log() * learned_weight
        ).sum(dim=0)
        side = torch.stack(
            (
                support_count / max(float(views), 1.0),
                weight_sum / max(float(views), 1.0),
                max_weight,
                entropy / max(math.log(max(views, 2)), 1.0),
            ),
            dim=-1,
        )
        output = self.output(
            torch.cat((mean, variance, self.side_projection(side)), dim=-1)
        )
        present = support_count > 0
        output = output * present[:, None].to(output.dtype)
        stats = {
            "supported_point_fraction": present.float().mean(),
            "mean_support_views": support_count.mean(),
            "mean_base_weight": base_weight.float().mean(),
            "learned_weight_entropy": entropy[present].mean()
            if bool(present.any().item())
            else entropy.new_zeros(()),
            "condition_hidden_rms": output.float().square().mean().sqrt(),
        }
        return output, stats


class EveryBlockConditionProjector(nn.Module):
    def __init__(self, *, hidden_dim: int, flow_channels: int, block_count: int) -> None:
        super().__init__()
        if int(block_count) <= 0:
            raise ValueError("every-block conditioner requires at least one block")
        self.hidden_dim = int(hidden_dim)
        self.flow_channels = int(flow_channels)
        self.projections = nn.ModuleList(
            nn.Linear(self.hidden_dim, self.flow_channels)
            for _ in range(int(block_count))
        )
        for projection in self.projections:
            nn.init.zeros_(projection.weight)
            nn.init.zeros_(projection.bias)

    def all_outputs_exact_zero(self) -> bool:
        return all(
            int(torch.count_nonzero(parameter.detach()).item()) == 0
            for parameter in self.parameters()
        )

    def forward(self, block_index: int, condition: torch.Tensor) -> torch.Tensor:
        return self.projections[int(block_index)](condition)


def _straight_through_reference(value: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if value.shape != reference.shape:
        raise ValueError("straight-through reference shape mismatch")
    return value + (reference - value).detach()


def _condition_stats(
    block_tokens: Sequence[torch.Tensor],
    aggregation_stats: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    if not block_tokens:
        raise ValueError("condition stats require at least one block token tensor")
    rms = torch.stack(
        [value.float().square().mean().sqrt() for value in block_tokens]
    )
    absolute = torch.stack([value.float().abs().amax() for value in block_tokens])
    return {
        **aggregation_stats,
        "condition_token_rms": rms.mean(),
        "condition_token_rms_max": rms.max(),
        "condition_token_abs_max": absolute.max(),
        "conditioned_block_count": rms.new_tensor(float(len(block_tokens))),
    }


class NativeEveryBlockSSFlowModel(nn.Module):
    """Frozen native SS Flow with direct 16^3 image conditioning at every block."""

    def __init__(
        self,
        flow: nn.Module,
        *,
        visual_channels: int = 1024,
        hidden_dim: int = 128,
        feature_source: str = "dino",
    ) -> None:
        super().__init__()
        self.flow = flow
        self.feature_source = str(feature_source)
        core = self.flow_core
        schema = (
            int(core.resolution),
            int(core.in_channels),
            int(core.out_channels),
            int(core.patch_size),
        )
        if schema != (16, 8, 8, 1):
            raise ValueError(f"unsupported native SS schema={schema}")
        self.aggregator = NativeViewAggregator(
            visual_channels=int(visual_channels), hidden_dim=int(hidden_dim)
        )
        self.block_condition = EveryBlockConditionProjector(
            hidden_dim=int(hidden_dim),
            flow_channels=int(core.model_channels),
            block_count=len(core.blocks),
        )

    @property
    def flow_core(self) -> nn.Module:
        base = getattr(self.flow, "base_model", None)
        core = getattr(base, "model", None)
        return core if isinstance(core, nn.Module) else self.flow

    def train(self, mode: bool = True) -> "NativeEveryBlockSSFlowModel":
        super().train(mode)
        self.flow.eval()
        self.aggregator.train(mode)
        self.block_condition.train(mode)
        return self

    def metadata(self) -> dict[str, Any]:
        return {
            "format": NATIVE_SS_FLOW_VERSION,
            "condition": self.aggregator.metadata(),
            "feature_source": self.feature_source,
            "resolution": 16,
            "injection": "independent zero-init projection before every native SS block",
            "block_count": len(self.block_condition.projections),
            "stock_fallback": "condition missing/disabled/scale=0 calls frozen stock Flow directly",
            "training_semantics": NATIVE_SS_TRAINING_SEMANTICS,
            "guided_delta_policy": NATIVE_SS_GUIDED_DELTA_POLICY,
            "delta_bound_mode": NATIVE_SS_DELTA_BOUND_SMOOTH,
            "lora": False,
        }

    def stock_prediction(self, x: torch.Tensor, t: torch.Tensor, cond: Any) -> torch.Tensor:
        return self.flow(x, t, cond)

    def _core_forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: Any,
        sample: dict[str, Any],
        *,
        condition_scale: float,
        projection_mode: str,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        from trellis.modules.spatial import patchify, unpatchify

        core = self.flow_core
        h = volume_xyz_to_flow_tokens(patchify(x, core.patch_size))
        h = core.input_layer(h) + core.pos_emb[None]
        t_emb = core.t_embedder(t)
        if core.share_mod:
            t_emb = core.adaLN_modulation(t_emb)
        h = h.type(core.dtype)
        t_emb = t_emb.type(core.dtype)
        coords = dense_native_coords(16, device=x.device, with_batch=True)
        with torch.no_grad():
            evidence = cached_project_native_features(
                sample,
                coords,
                resolution=16,
                device=x.device,
                mode=str(projection_mode),
                feature_source=self.feature_source,
            )
        condition_hidden, aggregation_stats = self.aggregator(
            evidence["projected_visual"],
            evidence["per_view_geometry"],
            evidence["base_weight"],
        )
        condition_hidden = condition_hidden[None]
        contexts = cond if isinstance(cond, list) else [cond]
        block_tokens: list[torch.Tensor] = []
        for context_value in contexts:
            context = context_value.type(core.dtype)
            for block_index, block in enumerate(core.blocks):
                tokens = self.block_condition(block_index, condition_hidden)
                tokens = tokens.to(dtype=h.dtype) * float(condition_scale)
                h = block(h + tokens, t_emb, context)
                block_tokens.append(tokens)
        h = h.type(x.dtype)
        h = F.layer_norm(h, h.shape[-1:])
        h = core.out_layer(h)
        h = flow_tokens_to_volume_xyz(h)
        return unpatchify(h, core.patch_size).contiguous(), _condition_stats(
            block_tokens, aggregation_stats
        )

    def conditioned_prediction(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: Any,
        sample: dict[str, Any] | None,
        *,
        stock_velocity: torch.Tensor | None = None,
        condition_scale: float = 1.0,
        projection_mode: str = "correct",
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        stock = self.stock_prediction(x, t, cond) if stock_velocity is None else stock_velocity
        if sample is None or float(condition_scale) == 0.0:
            zero = stock.new_zeros((), dtype=torch.float32)
            return stock, {
                "condition_token_rms": zero,
                "condition_token_rms_max": zero,
                "condition_token_abs_max": zero,
                "conditioned_block_count": zero,
                "condition_present": zero,
                "flow_delta_rms": zero,
                "flow_delta_abs_max": zero,
            }
        prediction, stats = self._core_forward(
            x,
            t,
            cond,
            sample,
            condition_scale=float(condition_scale),
            projection_mode=str(projection_mode),
        )
        zero_init_anchor = self.block_condition.all_outputs_exact_zero()
        if zero_init_anchor:
            prediction = _straight_through_reference(prediction, stock)
        delta = prediction.float() - stock.float()
        stats.update(
            {
                "condition_present": delta.new_tensor(1.0),
                "zero_init_stock_anchor": delta.new_tensor(float(zero_init_anchor)),
                "flow_delta_rms": delta.square().mean().sqrt(),
                "flow_delta_abs_max": delta.abs().amax(),
            }
        )
        return prediction, stats

    def post_cfg_conditioned_prediction(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        positive_condition: Any,
        negative_condition: Any,
        sample: dict[str, Any] | None,
        *,
        stock_positive_velocity: torch.Tensor | None = None,
        stock_negative_velocity: torch.Tensor | None = None,
        cfg_strength: float = 5.0,
        cfg_active: bool = True,
        support_active: bool = True,
        condition_scale: float = 1.0,
        projection_mode: str = "correct",
        delta_scale: float = 1.0,
        delta_rms_ratio_cap: float = 0.10,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Return the bounded deployed velocity after native CFG composition."""

        strength = float(cfg_strength)
        if not math.isfinite(strength) or strength < 0.0:
            raise ValueError("native SS post-CFG strength must be finite and non-negative")
        stock_positive = (
            self.stock_prediction(x, t, positive_condition)
            if stock_positive_velocity is None
            else stock_positive_velocity
        )
        if bool(cfg_active) and strength != 1.0:
            stock_negative = (
                self.stock_prediction(x, t, negative_condition)
                if stock_negative_velocity is None
                else stock_negative_velocity
            )
            stock_reference = combine_dense_cfg(
                stock_positive,
                stock_negative,
                cfg_strength=strength,
            )
            applied_strength = strength
        else:
            stock_negative = None
            stock_reference = stock_positive
            applied_strength = 1.0

        support_enabled = bool(
            support_active
            and sample is not None
            and float(condition_scale) != 0.0
        )
        if support_enabled:
            raw_positive, positive_stats = self.conditioned_prediction(
                x,
                t,
                positive_condition,
                sample,
                stock_velocity=stock_positive,
                condition_scale=float(condition_scale),
                projection_mode=str(projection_mode),
            )
            raw_guided = (
                combine_dense_cfg(
                    raw_positive,
                    stock_negative,
                    cfg_strength=strength,
                )
                if stock_negative is not None
                else raw_positive
            )
            zero_init_anchor = self.block_condition.all_outputs_exact_zero()
            if zero_init_anchor:
                raw_guided = _straight_through_reference(
                    raw_guided, stock_reference
                )
        else:
            raw_guided = stock_reference
            zero = stock_reference.new_zeros((), dtype=torch.float32)
            positive_stats = {
                "condition_token_rms": zero,
                "condition_token_rms_max": zero,
                "condition_token_abs_max": zero,
                "conditioned_block_count": zero,
                "condition_present": zero,
                "flow_delta_rms": zero,
                "flow_delta_abs_max": zero,
            }
            zero_init_anchor = False

        prediction, guided_stats = bounded_native_ss_flow_delta(
            stock_reference,
            raw_guided,
            delta_scale=float(delta_scale),
            delta_rms_ratio_cap=float(delta_rms_ratio_cap),
        )
        if not support_enabled or float(delta_scale) == 0.0:
            prediction = stock_reference
        stats = dict(positive_stats)
        stats.update(guided_stats)
        stats.update(
            {
                "flow_delta_rms": guided_stats["effective_flow_delta_rms"],
                "flow_delta_abs_max": guided_stats[
                    "effective_flow_delta_abs_max"
                ],
                "cfg_active": prediction.new_tensor(
                    float(bool(cfg_active)), dtype=torch.float32
                ),
                "support_active": prediction.new_tensor(
                    float(support_enabled), dtype=torch.float32
                ),
                "applied_cfg_strength": prediction.new_tensor(
                    float(applied_strength), dtype=torch.float32
                ),
                "positive_raw_flow_delta_rms": positive_stats.get(
                    "flow_delta_rms", prediction.new_zeros((), dtype=torch.float32)
                ),
                "post_cfg_zero_init_stock_anchor": prediction.new_tensor(
                    float(zero_init_anchor), dtype=torch.float32
                ),
            }
        )
        return prediction, stock_reference, stats

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: Any,
        sample: dict[str, Any],
        *,
        stock_velocity: torch.Tensor | None = None,
        negative_condition: Any | None = None,
        stock_negative_velocity: torch.Tensor | None = None,
        condition_scale: float = 1.0,
        projection_mode: str = "correct",
        cfg_strength: float = 1.0,
        cfg_active: bool = False,
        support_active: bool = True,
        delta_scale: float = 1.0,
        delta_rms_ratio_cap: float = 0.10,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        if negative_condition is not None:
            return self.post_cfg_conditioned_prediction(
                x,
                t,
                cond,
                negative_condition,
                sample,
                stock_positive_velocity=stock_velocity,
                stock_negative_velocity=stock_negative_velocity,
                cfg_strength=float(cfg_strength),
                cfg_active=bool(cfg_active),
                support_active=bool(support_active),
                condition_scale=float(condition_scale),
                projection_mode=str(projection_mode),
                delta_scale=float(delta_scale),
                delta_rms_ratio_cap=float(delta_rms_ratio_cap),
            )
        stock = self.stock_prediction(x, t, cond) if stock_velocity is None else stock_velocity
        prediction, stats = self.conditioned_prediction(
            x,
            t,
            cond,
            sample,
            stock_velocity=stock,
            condition_scale=float(condition_scale),
            projection_mode=str(projection_mode),
        )
        return prediction, stock, stats


def _slat_flow_core(flow: nn.Module) -> nn.Module:
    base = getattr(flow, "base_model", None)
    core = getattr(base, "model", None)
    return core if isinstance(core, nn.Module) else flow


class NativeEveryBlockSLatFlowModel(nn.Module):
    """Frozen native SLAT Flow with direct image lifting at active 32^3 coords."""

    def __init__(
        self,
        flow: nn.Module,
        *,
        visual_channels: int = 1024,
        hidden_dim: int = 128,
        feature_source: str = "dino",
    ) -> None:
        super().__init__()
        self.flow = flow
        self.feature_source = str(feature_source)
        core = self.flow_core
        schema = (
            int(core.resolution),
            int(core.in_channels),
            int(core.out_channels),
            int(core.patch_size),
        )
        if schema != (64, 8, 8, 2):
            raise ValueError(f"unsupported native SLAT schema={schema}")
        self.aggregator = NativeViewAggregator(
            visual_channels=int(visual_channels), hidden_dim=int(hidden_dim)
        )
        self.block_condition = EveryBlockConditionProjector(
            hidden_dim=int(hidden_dim),
            flow_channels=int(core.model_channels),
            block_count=len(core.blocks),
        )

    @property
    def flow_core(self) -> nn.Module:
        return _slat_flow_core(self.flow)

    def train(self, mode: bool = True) -> "NativeEveryBlockSLatFlowModel":
        super().train(mode)
        self.flow.eval()
        self.aggregator.train(mode)
        self.block_condition.train(mode)
        return self

    def metadata(self) -> dict[str, Any]:
        return {
            "format": NATIVE_SLAT_FLOW_VERSION,
            "condition": self.aggregator.metadata(),
            "feature_source": self.feature_source,
            "query_resolution": 32,
            "query_coordinates": "actual sparse stem coordinates after native input blocks",
            "injection": "independent zero-init projection before every native SLAT transformer block",
            "block_count": len(self.block_condition.projections),
            "uses_ss16_as_image_condition": False,
            "stock_fallback": "condition missing/disabled/scale=0 calls frozen stock Flow directly",
            "lora": False,
        }

    def stock_prediction(self, x: Any, t: torch.Tensor, cond: Any) -> Any:
        return self.flow(x, t, cond)

    def _core_forward(
        self,
        x: Any,
        t: torch.Tensor,
        cond: Any,
        sample: dict[str, Any],
        *,
        condition_scale: float,
        projection_mode: str,
    ) -> tuple[Any, dict[str, torch.Tensor]]:
        core = self.flow_core
        h = core.input_layer(x).type(core.dtype)
        t_emb = core.t_embedder(t)
        if core.share_mod:
            t_emb = core.adaLN_modulation(t_emb)
        t_emb = t_emb.type(core.dtype)
        if isinstance(cond, list):
            cond = [value.type(core.dtype) for value in cond]
        else:
            cond = cond.type(core.dtype)
        skips: list[torch.Tensor] = []
        for block in core.input_blocks:
            h = block(h, t_emb)
            skips.append(h.feats)
        if core.pe_mode == "ape":
            h = h + core.pos_embedder(h.coords[:, 1:]).type(core.dtype)
        if h.coords.ndim != 2 or int(h.coords.shape[1]) != 4:
            raise ValueError("native SLAT stem coordinates must be [N,4]")
        if h.coords.numel() and not bool((h.coords[:, 0] == 0).all().item()):
            raise ValueError("native SLAT conditioning currently requires batch size one")
        with torch.no_grad():
            evidence = cached_project_native_features(
                sample,
                h.coords,
                resolution=32,
                device=h.feats.device,
                mode=str(projection_mode),
                feature_source=self.feature_source,
            )
        if not torch.equal(evidence["coords"], h.coords.to(torch.int32)):
            raise RuntimeError("projected SLAT evidence lost native stem coordinate order")
        condition_hidden, aggregation_stats = self.aggregator(
            evidence["projected_visual"],
            evidence["per_view_geometry"],
            evidence["base_weight"],
        )
        block_tokens: list[torch.Tensor] = []
        for block_index, block in enumerate(core.blocks):
            tokens = self.block_condition(block_index, condition_hidden)
            tokens = tokens.to(dtype=h.dtype) * float(condition_scale)
            h = block(h.replace(h.feats + tokens), t_emb, cond)
            block_tokens.append(tokens)
        for block, skip in zip(core.out_blocks, reversed(skips)):
            if core.use_skip_connection:
                h = block(h.replace(torch.cat((h.feats, skip), dim=1)), t_emb)
            else:
                h = block(h, t_emb)
        h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))
        h = core.out_layer(h.type(x.dtype))
        stats = _condition_stats(block_tokens, aggregation_stats)
        stats["native_active_point_count"] = h.feats.new_tensor(
            float(condition_hidden.shape[0]), dtype=torch.float32
        )
        return h, stats

    def conditioned_prediction(
        self,
        x: Any,
        t: torch.Tensor,
        cond: Any,
        sample: dict[str, Any] | None,
        *,
        stock_velocity: Any | None = None,
        condition_scale: float = 1.0,
        projection_mode: str = "correct",
    ) -> tuple[Any, dict[str, torch.Tensor]]:
        stock = self.stock_prediction(x, t, cond) if stock_velocity is None else stock_velocity
        if sample is None or float(condition_scale) == 0.0:
            zero = stock.feats.new_zeros((), dtype=torch.float32)
            return stock, {
                "condition_token_rms": zero,
                "condition_token_rms_max": zero,
                "condition_token_abs_max": zero,
                "conditioned_block_count": zero,
                "condition_present": zero,
                "flow_delta_rms": zero,
                "flow_delta_abs_max": zero,
            }
        prediction, stats = self._core_forward(
            x,
            t,
            cond,
            sample,
            condition_scale=float(condition_scale),
            projection_mode=str(projection_mode),
        )
        if not torch.equal(prediction.coords, stock.coords):
            raise RuntimeError("native SLAT condition changed sparse coordinates")
        zero_init_anchor = self.block_condition.all_outputs_exact_zero()
        if zero_init_anchor:
            prediction = prediction.replace(
                _straight_through_reference(prediction.feats, stock.feats)
            )
        delta = prediction.feats.float() - stock.feats.float()
        stats.update(
            {
                "condition_present": delta.new_tensor(1.0),
                "zero_init_stock_anchor": delta.new_tensor(float(zero_init_anchor)),
                "flow_delta_rms": delta.square().mean().sqrt(),
                "flow_delta_abs_max": delta.abs().amax(),
            }
        )
        return prediction, stats

    def forward(
        self,
        x: Any,
        t: torch.Tensor,
        cond: Any,
        sample: dict[str, Any],
        *,
        stock_velocity: Any | None = None,
        condition_scale: float = 1.0,
        projection_mode: str = "correct",
    ) -> tuple[Any, Any, dict[str, torch.Tensor]]:
        stock = self.stock_prediction(x, t, cond) if stock_velocity is None else stock_velocity
        prediction, stats = self.conditioned_prediction(
            x,
            t,
            cond,
            sample,
            stock_velocity=stock,
            condition_scale=float(condition_scale),
            projection_mode=str(projection_mode),
        )
        return prediction, stock, stats


class NativeConditionSLatDataset(Dataset):
    """Strict UID join between materialized SLAT targets and pose-lifting rows."""

    def __init__(
        self,
        slat_manifest: str | Path,
        lifting_manifest: str | Path,
        *,
        indices: str = "all",
        verify_hashes: bool = False,
    ) -> None:
        self.compact_backend: CompactNativeConditionBackend | None = None
        if is_compact_manifest_pair(slat_manifest, lifting_manifest):
            compact = CompactNativeConditionBackend(
                slat_manifest,
                lifting_manifest,
                indices=indices,
                verify_hashes=verify_hashes,
            )
            self.compact_backend = compact
            self.slat = compact.slat_view
            self.lifting = compact.lifting_view
            self.lifting_indices = list(range(len(compact)))
            self.rows = compact.rows
            self.config = compact.config
            self.config_hash = compact.config_hash
            self.slat_normalization = compact.slat_normalization
            self.slat_normalization_hash = compact.slat_normalization_hash
            self.identity = {
                "version": NATIVE_3D_CONDITION_VERSION,
                "slat_manifest": str(Path(slat_manifest).resolve()),
                "slat_manifest_sha256": sha256_file(slat_manifest),
                "lifting_manifest": str(Path(lifting_manifest).resolve()),
                "lifting_manifest_sha256": sha256_file(lifting_manifest),
                "uid_count": len(self.rows),
                "uid_hash": canonical_json_sha256(
                    sorted(str(row["uid"]) for row in self.rows)
                ),
                "compact_cache_layout": True,
            }
            return
        self.slat = DirectSLatCacheDataset(
            slat_manifest, indices=indices, verify_hashes=verify_hashes
        )
        self.lifting = PoseLiftingCacheDataset(lifting_manifest, indices="all")
        lifting_by_uid: dict[str, int] = {}
        for index, row in enumerate(self.lifting.rows):
            uid = str(row.get("uid", ""))
            if not uid or uid in lifting_by_uid:
                raise ValueError("lifting manifest contains empty/duplicate UID")
            lifting_by_uid[uid] = index
        missing = sorted(
            {str(row["uid"]) for row in self.slat.rows} - set(lifting_by_uid)
        )
        if missing:
            raise ValueError(
                f"SLAT/lifting UID join is incomplete: missing={missing[:8]} count={len(missing)}"
            )
        self.lifting_indices = [lifting_by_uid[str(row["uid"])] for row in self.slat.rows]
        self.rows = self.slat.rows
        self.config = self.slat.config
        self.config_hash = self.slat.config_hash
        self.slat_normalization = self.slat.slat_normalization
        self.slat_normalization_hash = self.slat.slat_normalization_hash
        self.identity = {
            "version": NATIVE_3D_CONDITION_VERSION,
            "slat_manifest": str(Path(slat_manifest).resolve()),
            "slat_manifest_sha256": sha256_file(slat_manifest),
            "lifting_manifest": str(Path(lifting_manifest).resolve()),
            "lifting_manifest_sha256": sha256_file(lifting_manifest),
            "uid_count": len(self.rows),
            "uid_hash": canonical_json_sha256(sorted(str(row["uid"]) for row in self.rows)),
        }

    def __len__(self) -> int:
        if self.compact_backend is not None:
            return len(self.compact_backend)
        return len(self.slat)

    def limit_objects(self, max_objects: int) -> None:
        if int(max_objects) <= 0:
            return
        if self.compact_backend is not None:
            self.compact_backend.limit_objects(max_objects)
            self.rows = self.compact_backend.rows
            self.lifting_indices = list(range(len(self.rows)))
            allowed = {str(row["object_uid"]) for row in self.rows}
            self.identity.update(
                {
                    "uid_count": len(self.rows),
                    "object_count": len(allowed),
                    "uid_hash": canonical_json_sha256(
                        sorted(str(row["uid"]) for row in self.rows)
                    ),
                }
            )
            return
        allowed = set(
            sorted({str(row["object_uid"]) for row in self.rows})[: int(max_objects)]
        )
        selected = [
            index
            for index, row in enumerate(self.rows)
            if str(row["object_uid"]) in allowed
        ]
        self.slat.rows = [self.slat.rows[index] for index in selected]
        self.lifting_indices = [self.lifting_indices[index] for index in selected]
        self.rows = self.slat.rows
        self.identity.update(
            {
                "uid_count": len(self.rows),
                "object_count": len(allowed),
                "uid_hash": canonical_json_sha256(
                    sorted(str(row["uid"]) for row in self.rows)
                ),
            }
        )

    def __getitem__(self, index: int) -> dict[str, Any]:
        if self.compact_backend is not None:
            return self.compact_backend[index]
        slat_sample = self.slat[index]
        lifting_sample = self.lifting[self.lifting_indices[index]]
        if str(slat_sample["uid"]) != str(lifting_sample["uid"]):
            raise RuntimeError("SLAT/lifting runtime UID join changed")
        slat_object = str(slat_sample.get("object_uid", ""))
        lifting_object = str(lifting_sample.get("object_uid", slat_object))
        if slat_object and lifting_object and slat_object != lifting_object:
            raise RuntimeError("SLAT/lifting object identity mismatch")
        return {**slat_sample, "lifting_sample": lifting_sample}


def collate_native_one(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if len(rows) != 1:
        raise ValueError("native 3D condition training requires batch size one per rank")
    return rows[0]


def trainable_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def load_trainable_state_dict(
    model: nn.Module, state: dict[str, torch.Tensor]
) -> None:
    expected = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    if set(state) != expected:
        raise ValueError(
            "native trainable state keys differ: "
            f"missing={sorted(expected - set(state))[:8]} "
            f"unexpected={sorted(set(state) - expected)[:8]}"
        )
    named = dict(model.named_parameters())
    with torch.no_grad():
        for name, value in state.items():
            if named[name].shape != value.shape:
                raise ValueError(f"native checkpoint shape mismatch for {name}")
            named[name].copy_(value.to(device=named[name].device, dtype=named[name].dtype))


def native_trainable_whitelist(model: nn.Module) -> dict[str, Any]:
    names = [name for name, value in model.named_parameters() if value.requires_grad]
    unexpected = [
        name
        for name in names
        if not name.startswith("aggregator.")
        and not name.startswith("block_condition.")
    ]
    if unexpected or not names:
        raise RuntimeError(f"native adapter-only trainable whitelist failed: {unexpected}")
    return {
        "names": names,
        "parameter_count": int(
            sum(value.numel() for value in model.parameters() if value.requires_grad)
        ),
        "whitelist": ["aggregator.*", "block_condition.*"],
    }


def freeze_except_native_condition(model: nn.Module) -> dict[str, Any]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.aggregator.parameters():
        parameter.requires_grad_(True)
    for parameter in model.block_condition.parameters():
        parameter.requires_grad_(True)
    return native_trainable_whitelist(model)


class NativeStockSSFlow(nn.Module):
    def __init__(self, model: NativeEveryBlockSSFlowModel) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor, t: torch.Tensor, condition: Any) -> torch.Tensor:
        return self.model.stock_prediction(x, t, condition)


class PositiveNativeSSRolloutFlow(nn.Module):
    def __init__(
        self,
        model: NativeEveryBlockSSFlowModel,
        positive_condition: Any,
        sample: dict[str, Any],
        *,
        condition_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.model = model
        self.positive_condition = positive_condition
        self.sample = sample
        self.condition_scale = float(condition_scale)
        self.positive_calls = 0
        self.negative_calls = 0

    def _is_positive(self, condition: Any) -> bool:
        return condition is self.positive_condition or (
            torch.is_tensor(condition)
            and torch.is_tensor(self.positive_condition)
            and condition.shape == self.positive_condition.shape
            and condition.data_ptr() == self.positive_condition.data_ptr()
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor, condition: Any) -> torch.Tensor:
        if not self._is_positive(condition):
            self.negative_calls += 1
            return self.model.stock_prediction(x, t, condition)
        self.positive_calls += 1
        prediction, _ = self.model.conditioned_prediction(
            x,
            t,
            condition,
            self.sample,
            condition_scale=self.condition_scale,
        )
        return prediction


class PostCFGNativeSSRolloutFlow(nn.Module):
    """Expose one internally composed post-CFG velocity to an external CFG=1 sampler."""

    def __init__(
        self,
        model: NativeEveryBlockSSFlowModel,
        positive_condition: Any,
        negative_condition: Any,
        sample: dict[str, Any],
        *,
        cfg_strength: float = 5.0,
        cfg_interval: Iterable[float] = (0.5, 1.0),
        condition_scale: float = 1.0,
        delta_scale: float = 1.0,
        delta_rms_ratio_cap: float = 0.10,
        projection_mode: str = "correct",
    ) -> None:
        super().__init__()
        self.model = model
        self.positive_condition = positive_condition
        self.negative_condition = negative_condition
        self.sample = sample
        self.cfg_strength = float(cfg_strength)
        self.cfg_interval = normalize_native_cfg_interval(cfg_interval)
        self.condition_scale = float(condition_scale)
        self.delta_scale = float(delta_scale)
        self.delta_rms_ratio_cap = float(delta_rms_ratio_cap)
        self.projection_mode = str(projection_mode)
        self.calls = 0
        self.active_calls = 0
        self.inactive_calls = 0
        self.guided_stats: list[dict[str, float | bool]] = []

    def forward(self, x: torch.Tensor, t: torch.Tensor, _condition: Any) -> torch.Tensor:
        t_values = t.detach().float().reshape(-1) / 1000.0
        if t_values.numel() == 0 or not bool(torch.isfinite(t_values).all().item()):
            raise ValueError("post-CFG Native SS rollout received an invalid timestep")
        if float((t_values - t_values[0]).abs().amax().item()) > 1.0e-7:
            raise ValueError("post-CFG Native SS rollout requires one shared batch timestep")
        t_value = float(t_values[0].item())
        cfg_active = native_ss_cfg_is_active(t_value, self.cfg_interval)
        support_active = cfg_active
        prediction, stock_reference, stats = self.model.post_cfg_conditioned_prediction(
            x,
            t,
            self.positive_condition,
            self.negative_condition,
            self.sample,
            cfg_strength=self.cfg_strength,
            cfg_active=cfg_active,
            support_active=support_active,
            condition_scale=self.condition_scale,
            projection_mode=self.projection_mode,
            delta_scale=self.delta_scale,
            delta_rms_ratio_cap=self.delta_rms_ratio_cap,
        )
        self.calls += 1
        if cfg_active:
            self.active_calls += 1
        else:
            self.inactive_calls += 1
        self.guided_stats.append(
            {
                "t": t_value,
                "cfg_active": cfg_active,
                "support_active": bool(
                    cfg_active and self.condition_scale != 0.0
                ),
                "stock_velocity_rms": float(
                    stats["stock_velocity_rms"].detach().item()
                ),
                "raw_flow_delta_rms": float(
                    stats["raw_flow_delta_rms"].detach().item()
                ),
                "effective_flow_delta_rms": float(
                    stats["effective_flow_delta_rms"].detach().item()
                ),
                "raw_flow_delta_ratio": float(
                    stats["raw_flow_delta_ratio"].detach().item()
                ),
                "effective_flow_delta_ratio": float(
                    stats["effective_flow_delta_ratio"].detach().item()
                ),
                "delta_clip_scale": float(
                    stats["delta_clip_scale"].detach().item()
                ),
                "delta_clip_activated": bool(
                    stats["delta_clip_activated"].detach().item()
                ),
                "full_minus_stock_abs_max": float(
                    (prediction.float() - stock_reference.float()).abs().amax().item()
                ),
            }
        )
        return prediction

    def stats_summary(self) -> dict[str, Any]:
        scalar_names = (
            "stock_velocity_rms",
            "raw_flow_delta_rms",
            "effective_flow_delta_rms",
            "raw_flow_delta_ratio",
            "effective_flow_delta_ratio",
            "delta_clip_scale",
            "full_minus_stock_abs_max",
        )
        means = {
            f"mean_{name}": (
                sum(float(row[name]) for row in self.guided_stats)
                / len(self.guided_stats)
                if self.guided_stats
                else 0.0
            )
            for name in scalar_names
        }
        return {
            "calls": self.calls,
            "active_calls": self.active_calls,
            "inactive_calls": self.inactive_calls,
            "clip_activated_calls": sum(
                int(bool(row["delta_clip_activated"])) for row in self.guided_stats
            ),
            **means,
        }


class NativeStockSLatFlow(nn.Module):
    def __init__(self, model: NativeEveryBlockSLatFlowModel) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: Any, t: torch.Tensor, condition: Any) -> Any:
        return self.model.stock_prediction(x, t, condition)


class PositiveNativeSLatRolloutFlow(nn.Module):
    def __init__(
        self,
        model: NativeEveryBlockSLatFlowModel,
        positive_condition: Any,
        sample: dict[str, Any],
        *,
        condition_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.model = model
        self.positive_condition = positive_condition
        self.sample = sample
        self.condition_scale = float(condition_scale)
        self.positive_calls = 0
        self.negative_calls = 0
        self.stats: list[dict[str, float]] = []

    def _is_positive(self, condition: Any) -> bool:
        if condition is self.positive_condition:
            return True
        if isinstance(condition, list) and isinstance(self.positive_condition, list):
            if len(condition) != len(self.positive_condition):
                return False
            return all(
                left is right
                or (
                    torch.is_tensor(left)
                    and torch.is_tensor(right)
                    and left.shape == right.shape
                    and left.data_ptr() == right.data_ptr()
                )
                for left, right in zip(condition, self.positive_condition)
            )
        return False

    def forward(self, x: Any, t: torch.Tensor, condition: Any) -> Any:
        if not self._is_positive(condition):
            self.negative_calls += 1
            return self.model.stock_prediction(x, t, condition)
        self.positive_calls += 1
        prediction, stats = self.model.conditioned_prediction(
            x,
            t,
            condition,
            self.sample,
            condition_scale=self.condition_scale,
        )
        self.stats.append(
            {
                "flow_delta_rms": float(stats["flow_delta_rms"].detach().item()),
                "condition_token_rms": float(
                    stats["condition_token_rms"].detach().item()
                ),
            }
        )
        return prediction

    def stats_summary(self) -> dict[str, Any]:
        return {
            "positive_calls": self.positive_calls,
            "negative_calls": self.negative_calls,
            "mean_flow_delta_rms": (
                sum(row["flow_delta_rms"] for row in self.stats) / len(self.stats)
                if self.stats
                else 0.0
            ),
            "mean_condition_token_rms": (
                sum(row["condition_token_rms"] for row in self.stats) / len(self.stats)
                if self.stats
                else 0.0
            ),
        }


def build_native_ss_components(
    *,
    pretrained: str,
    hidden_dim: int,
    feature_source: str,
    gradient_checkpointing: bool,
    need_decoder: bool,
    device: torch.device,
    retain_pipeline: bool = False,
) -> tuple[Any, ...]:
    from pose_point_depth_mv.train_direct_flow import install_unused_model_stubs
    from trellis.pipelines import TrellisVGGTTo3DPipeline

    install_unused_model_stubs()
    pipeline = TrellisVGGTTo3DPipeline.from_pretrained(pretrained)
    pipeline._device = device
    pipeline.low_vram = False
    flow = pipeline.models["sparse_structure_flow_model"].to(device).eval()
    flow.use_checkpoint = bool(gradient_checkpointing)
    for block in flow.blocks:
        block.use_checkpoint = bool(gradient_checkpointing)
    _, visual_channels, feature_identity = native_feature_slice(3072, feature_source)
    model = NativeEveryBlockSSFlowModel(
        flow,
        visual_channels=visual_channels,
        hidden_dim=int(hidden_dim),
        feature_source=str(feature_source),
    ).to(device)
    whitelist = freeze_except_native_condition(model)
    decoder = (
        pipeline.models["sparse_structure_decoder"].to(device).eval()
        if need_decoder
        else None
    )
    if decoder is not None:
        for parameter in decoder.parameters():
            parameter.requires_grad_(False)
    summary = {
        **model.metadata(),
        "stage": "native 16^3 every-block adapter-only SS Flow",
        "pretrained": str(pretrained),
        "feature_identity": feature_identity,
        "trainable": whitelist,
        "frozen": ["native SS Flow base", "native SS decoder", "image encoders/cache"],
    }
    result = (
        pipeline.sparse_structure_sampler,
        model,
        decoder,
        summary,
        dict(pipeline.sparse_structure_sampler_params),
    )
    if retain_pipeline:
        return (*result, pipeline)
    return result


def build_native_slat_components(
    *,
    pretrained: str,
    hidden_dim: int,
    feature_source: str,
    gradient_checkpointing: bool,
    device: torch.device,
    retain_pipeline: bool = False,
) -> tuple[Any, ...]:
    from pose_point_depth_mv.train_direct_flow import install_unused_model_stubs
    from trellis.pipelines import TrellisVGGTTo3DPipeline

    install_unused_model_stubs()
    pipeline = TrellisVGGTTo3DPipeline.from_pretrained(pretrained)
    pipeline._device = device
    pipeline.low_vram = False
    flow = pipeline.models["slat_flow_model"].to(device).eval()
    flow.use_checkpoint = bool(gradient_checkpointing)
    for block in flow.blocks:
        block.use_checkpoint = bool(gradient_checkpointing)
    _, visual_channels, feature_identity = native_feature_slice(3072, feature_source)
    model = NativeEveryBlockSLatFlowModel(
        flow,
        visual_channels=visual_channels,
        hidden_dim=int(hidden_dim),
        feature_source=str(feature_source),
    ).to(device)
    whitelist = freeze_except_native_condition(model)
    summary = {
        **model.metadata(),
        "stage": "native active-32^3 every-block adapter-only SLAT Flow",
        "pretrained": str(pretrained),
        "feature_identity": feature_identity,
        "trainable": whitelist,
        "frozen": ["native SLAT Flow base", "native SLAT decoder", "native SLAT context"],
    }
    result = (
        pipeline.slat_sampler,
        model,
        dict(pipeline.slat_sampler_params),
        dict(pipeline.slat_normalization),
        summary,
    )
    if retain_pipeline:
        return (*result, pipeline)
    return result


def validate_native_checkpoint(
    checkpoint: dict[str, Any],
    *,
    expected_format: str,
    pretrained: str,
) -> None:
    if checkpoint.get("format") != expected_format:
        raise ValueError(f"unexpected native checkpoint format={checkpoint.get('format')!r}")
    if str(checkpoint.get("model_summary", {}).get("pretrained")) != str(pretrained):
        raise RuntimeError("native checkpoint pretrained binding differs")
    if not isinstance(checkpoint.get("model_trainable_state"), dict):
        raise ValueError("native checkpoint lacks trainable state")


def native_ss_deployment_from_checkpoint(
    checkpoint: dict[str, Any],
) -> dict[str, Any]:
    args = checkpoint.get("args")
    if not isinstance(args, dict):
        raise ValueError("native SS checkpoint lacks bound training arguments")
    expected = {
        "training_semantics": NATIVE_SS_TRAINING_SEMANTICS,
        "guided_delta_policy": NATIVE_SS_GUIDED_DELTA_POLICY,
        "support_interval_policy": NATIVE_SS_SUPPORT_INTERVAL_CFG_ACTIVE,
        "delta_bound_mode": NATIVE_SS_DELTA_BOUND_SMOOTH,
        "t_schedule": "native_cfg_active",
    }
    mismatch = {
        key: (args.get(key), value)
        for key, value in expected.items()
        if args.get(key) != value
    }
    if mismatch:
        raise ValueError(f"native SS v2 checkpoint protocol mismatch={mismatch}")
    deployment = {
        "steps": int(args["native_steps"]),
        "cfg_strength": float(args["train_cfg_strength"]),
        "cfg_interval": normalize_native_cfg_interval(args["train_cfg_interval"]),
        "rescale_t": float(args["rescale_t"]),
        "guidance_rescale": float(args["guidance_rescale"]),
        "condition_scale": float(args["condition_scale"]),
        "delta_scale": float(args["delta_scale"]),
        "delta_rms_ratio_cap": float(args["delta_rms_ratio_cap"]),
        **expected,
    }
    if deployment["steps"] <= 0:
        raise ValueError("native SS checkpoint has non-positive deployment steps")
    for name in ("cfg_strength", "rescale_t", "condition_scale", "delta_rms_ratio_cap"):
        if not math.isfinite(deployment[name]) or deployment[name] <= 0.0:
            raise ValueError(f"native SS checkpoint has invalid {name}")
    if not math.isfinite(deployment["delta_scale"]) or deployment["delta_scale"] < 0.0:
        raise ValueError("native SS checkpoint has invalid delta_scale")
    if deployment["guidance_rescale"] != 0.0:
        raise ValueError("post_cfg_bounded_v2 requires guidance_rescale=0")
    active = tuple(
        value
        for value in native_ss_timestep_sequence(
            steps=deployment["steps"], rescale_t=deployment["rescale_t"]
        )[:-1]
        if native_ss_cfg_is_active(value, deployment["cfg_interval"])
    )
    if not active:
        raise ValueError("native SS checkpoint deployment has no CFG-active timestep")
    deployment["active_timesteps"] = active
    deployment["external_full_cfg_strength"] = 1.0
    return deployment


def ensure_finite_trainable(model: nn.Module) -> None:
    bad = [
        name
        for name, value in model.named_parameters()
        if value.requires_grad and not bool(torch.isfinite(value).all().item())
    ]
    if bad:
        raise RuntimeError(f"non-finite native trainable parameters={bad[:8]}")


def parameter_gradient_norm(model: nn.Module) -> float:
    total = 0.0
    for parameter in model.parameters():
        if parameter.requires_grad and parameter.grad is not None:
            total += float(parameter.grad.detach().float().square().sum().item())
    return math.sqrt(total)


def block_projection_gradient_norms(model: nn.Module) -> list[float]:
    values: list[float] = []
    for projection in model.block_condition.projections:
        total = 0.0
        for parameter in projection.parameters():
            if parameter.grad is not None:
                total += float(parameter.grad.detach().float().square().sum().item())
        values.append(math.sqrt(total))
    return values
