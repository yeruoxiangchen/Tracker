from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image


TRACKER_ROOT = Path(__file__).resolve().parents[1]
PIXAL3D_ROTATION = np.asarray(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float32,
)


def resolve_path(root: Optional[str], path: str | Path) -> Path:
    path_obj = Path(path)
    if path_obj.is_absolute() or root is None:
        return path_obj
    return Path(root) / path_obj


def load_manifest(path: str | Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    samples = payload.get("samples", payload if isinstance(payload, list) else None)
    if samples is None:
        raise ValueError(f"manifest has no samples list: {path}")
    return payload if isinstance(payload, dict) else {}, samples


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_int_list(text: str) -> List[int]:
    out = []
    for part in str(text).split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    return out


def parse_indices(spec: str, size: int) -> List[int]:
    spec = str(spec or "").strip()
    if spec.lower() in {"", "all"}:
        return list(range(size))
    out: List[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    bad = [idx for idx in out if idx < 0 or idx >= size]
    if bad:
        raise IndexError(f"indices out of range for dataset size {size}: {bad}")
    return out


def load_target_latent(latent_path: str | Path) -> Tuple[np.ndarray, np.ndarray]:
    with np.load(latent_path) as data:
        z = np.asarray(data["z"], dtype=np.float32)
        target_coords = np.asarray(data["target_coords"], dtype=np.int32)
    if z.ndim == 5 and z.shape[0] == 1:
        z = z[0]
    if z.ndim != 4:
        raise ValueError(f"expected z [C,D,H,W], got {z.shape}: {latent_path}")
    if target_coords.ndim != 2 or target_coords.shape[1] not in (3, 4):
        raise ValueError(f"expected target_coords [N,3/4], got {target_coords.shape}: {latent_path}")
    return z, target_coords[:, -3:].astype(np.int32, copy=False)


def coords_to_points(coords: np.ndarray, resolution: int = 64) -> np.ndarray:
    xyz = coords[:, -3:].astype(np.float32, copy=False)
    return (xyz + 0.5) / float(resolution) - 0.5


def points_to_coords(points: np.ndarray, resolution: int = 64) -> np.ndarray:
    coords = np.floor((points + 0.5) * float(resolution)).astype(np.int32)
    return np.clip(coords, 0, resolution - 1)


def apply_grid_transform(points: np.ndarray, mode: str) -> np.ndarray:
    if mode == "identity":
        return points
    if mode == "pixal3d_rotation":
        return points @ PIXAL3D_ROTATION.T
    raise ValueError(f"unsupported grid_transform={mode!r}")


def occupancy_from_coords(coords: np.ndarray, resolution: int = 64) -> np.ndarray:
    occ = np.zeros((resolution, resolution, resolution), dtype=bool)
    if coords.size:
        xyz = coords[:, -3:].astype(np.int64, copy=False)
        valid = (xyz >= 0).all(axis=1) & (xyz < resolution).all(axis=1)
        xyz = xyz[valid]
        occ[xyz[:, 0], xyz[:, 1], xyz[:, 2]] = True
    return occ


def surface_coords_from_target(target_coords: np.ndarray, resolution: int = 64) -> np.ndarray:
    occ = occupancy_from_coords(target_coords, resolution=resolution)
    if not occ.any():
        return np.zeros((0, 3), dtype=np.int32)
    padded = np.pad(occ, 1, mode="constant", constant_values=False)
    center = padded[1:-1, 1:-1, 1:-1]
    neighbor_count = (
        padded[:-2, 1:-1, 1:-1]
        + padded[2:, 1:-1, 1:-1]
        + padded[1:-1, :-2, 1:-1]
        + padded[1:-1, 2:, 1:-1]
        + padded[1:-1, 1:-1, :-2]
        + padded[1:-1, 1:-1, 2:]
    )
    surface = center & (neighbor_count < 6)
    return np.argwhere(surface).astype(np.int32)


def load_mask(mask_path: str | Path, size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    mask = Image.open(mask_path).convert("L")
    if size is not None and mask.size != size:
        mask = mask.resize(size, Image.NEAREST)
    return np.asarray(mask).astype(np.float32) / 255.0


def load_sample_frames(
    payload: Dict[str, Any],
    sample: Dict[str, Any],
    max_frames: int,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    image_root = sample.get("image_root", payload.get("image_root"))
    mask_root = sample.get("mask_root", payload.get("mask_root"))
    top_intrinsic = sample.get("intrinsic", payload.get("intrinsic"))
    frames = sample.get("frames") or []
    if max_frames > 0:
        frames = frames[:max_frames]
    intrinsics = []
    extrinsics = []
    mask_paths = []
    for frame in frames:
        intrinsic = frame.get("intrinsic", top_intrinsic)
        if intrinsic is None:
            raise ValueError(f"missing intrinsic for frame {frame.get('image')}")
        mask_rel = frame.get("mask", sample.get("mask"))
        if mask_rel is None:
            raise ValueError(f"missing mask for frame {frame.get('image')}")
        intrinsics.append(np.asarray(intrinsic, dtype=np.float32))
        extrinsics.append(np.asarray(frame["extrinsic"], dtype=np.float32))
        mask_paths.append(str(resolve_path(mask_root, mask_rel)))
    if not intrinsics:
        raise ValueError(f"sample {sample.get('uid')} has no frames")
    return np.stack(intrinsics), np.stack(extrinsics), mask_paths


def project_points_to_masks(
    points: np.ndarray,
    masks: Sequence[np.ndarray],
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    *,
    extrinsics_type: str = "c2w",
    camera_forward_sign: float = 1.0,
    min_support: float = 1.0,
    min_support_ratio: float = 0.45,
    front_depth: bool = True,
    front_depth_epsilon: float = 0.02,
) -> Dict[str, np.ndarray | float | int]:
    if points.size == 0:
        return {
            "supported": np.zeros((0,), dtype=bool),
            "support": np.zeros((0,), dtype=np.float32),
            "visible": np.zeros((0,), dtype=np.float32),
            "ratio": np.zeros((0,), dtype=np.float32),
            "visible_mean": 0.0,
            "supported_ratio": 0.0,
        }
    pts = points.astype(np.float32, copy=False)
    n = pts.shape[0]
    support = np.zeros(n, dtype=np.float32)
    visible = np.zeros(n, dtype=np.float32)
    ones = np.ones((n, 1), dtype=np.float32)
    pts_h = np.concatenate([pts, ones], axis=1)
    for mask, K, E in zip(masks, intrinsics, extrinsics):
        w2c = np.linalg.inv(E) if extrinsics_type == "c2w" else E
        cam = (w2c @ pts_h.T).T[:, :3]
        signed_depth = cam[:, 2] * float(camera_forward_sign)
        valid_depth = signed_depth > 1e-6
        z = np.maximum(signed_depth, 1e-6)
        u = K[0, 0] * (cam[:, 0] / z) + K[0, 2]
        v = K[1, 1] * (cam[:, 1] / z) + K[1, 2]
        h, w = mask.shape[:2]
        ui = np.rint(u).astype(np.int32)
        vi = np.rint(v).astype(np.int32)
        in_image = valid_depth & (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
        ids = np.nonzero(in_image)[0]
        if len(ids):
            if front_depth:
                pix = (vi[ids] * w + ui[ids]).astype(np.int64)
                z_ids = z[ids].astype(np.float32)
                min_depth = np.full((h * w,), np.inf, dtype=np.float32)
                np.minimum.at(min_depth, pix, z_ids)
                front = z_ids <= (min_depth[pix] + float(front_depth_epsilon))
                ids = ids[front]
            visible[ids] += 1.0
            support[ids] += mask[vi[ids], ui[ids]].astype(np.float32)
    ratio = support / np.maximum(visible, 1.0)
    supported = (support >= float(min_support)) & (ratio >= float(min_support_ratio))
    return {
        "supported": supported,
        "support": support,
        "visible": visible,
        "ratio": ratio,
        "visible_mean": float(visible.mean()),
        "support_mean": float(support.mean()),
        "ratio_mean": float(ratio.mean()),
        "supported_ratio": float(supported.mean()),
    }


def make_slam_like_prior(
    target_coords: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    mask_paths: Sequence[str],
    *,
    rng: np.random.Generator,
    grid_transform: str = "pixal3d_rotation",
    extrinsics_type: str = "c2w",
    camera_forward_sign: float = 1.0,
    num_prior_views_choices: Sequence[int] = (2, 4, 8),
    point_count_choices: Sequence[int] = (50, 100, 300, 800, 1500),
    min_support: float = 1.0,
    min_support_ratio: float = 0.45,
    dropout_min: float = 0.0,
    dropout_max: float = 0.65,
    coord_jitter: int = 1,
    outlier_ratio: float = 0.03,
    front_depth: bool = True,
    front_depth_epsilon: float = 0.02,
    allow_support_fallback: bool = False,
    resolution: int = 64,
) -> Dict[str, Any]:
    n_views = len(mask_paths)
    view_count = int(rng.choice([v for v in num_prior_views_choices if v <= n_views] or [n_views]))
    view_ids = np.sort(rng.choice(n_views, size=view_count, replace=False))
    masks = [load_mask(mask_paths[i]) for i in view_ids]
    surface_coords = surface_coords_from_target(target_coords, resolution=resolution)
    surface_points = apply_grid_transform(coords_to_points(surface_coords, resolution=resolution), grid_transform)
    support = project_points_to_masks(
        surface_points,
        masks,
        intrinsics[view_ids],
        extrinsics[view_ids],
        extrinsics_type=extrinsics_type,
        camera_forward_sign=camera_forward_sign,
        min_support=min_support,
        min_support_ratio=min_support_ratio,
        front_depth=front_depth,
        front_depth_epsilon=front_depth_epsilon,
    )
    supported_coords = surface_coords[np.asarray(support["supported"], dtype=bool)]
    raw_supported_surface_count = int(supported_coords.shape[0])
    support_failed = raw_supported_surface_count == 0
    fallback_used = False
    if support_failed and allow_support_fallback:
        supported_coords = surface_coords
        fallback_used = True

    requested_count = int(rng.choice(list(point_count_choices)))
    dropout = float(rng.uniform(dropout_min, dropout_max))
    actual_count = max(1, int(round(requested_count * (1.0 - dropout))))
    actual_count = min(actual_count, supported_coords.shape[0])
    if actual_count > 0:
        chosen = supported_coords[rng.choice(supported_coords.shape[0], size=actual_count, replace=False)].copy()
        conf = np.ones((chosen.shape[0],), dtype=np.float32)
    else:
        chosen = np.zeros((0, 3), dtype=np.int32)
        conf = np.zeros((0,), dtype=np.float32)

    if coord_jitter > 0 and chosen.shape[0] > 0:
        jitter = rng.integers(-int(coord_jitter), int(coord_jitter) + 1, size=chosen.shape, dtype=np.int32)
        chosen = np.clip(chosen + jitter, 0, resolution - 1)

    outlier_count = int(round(chosen.shape[0] * float(outlier_ratio)))
    if outlier_count > 0:
        outliers = rng.integers(0, resolution, size=(outlier_count, 3), dtype=np.int32)
        chosen = np.concatenate([chosen, outliers], axis=0)
        conf = np.concatenate([conf, np.full((outlier_count,), 0.1, dtype=np.float32)], axis=0)

    if chosen.shape[0] > 1:
        order = rng.permutation(chosen.shape[0])
        chosen = chosen[order]
        conf = conf[order]
    if chosen.shape[0] > 1:
        unique, inverse = np.unique(chosen.astype(np.int32), axis=0, return_inverse=True)
        merged_conf = np.zeros((unique.shape[0],), dtype=np.float32)
        np.maximum.at(merged_conf, inverse, conf.astype(np.float32))
        chosen = unique
        conf = merged_conf
    return {
        "prior_coords": chosen.astype(np.int32),
        "prior_conf": conf.astype(np.float32),
        "surface_count": int(surface_coords.shape[0]),
        "supported_surface_count": raw_supported_surface_count,
        "sampling_pool_count": int(supported_coords.shape[0]),
        "support_failed": bool(support_failed),
        "fallback_used": bool(fallback_used),
        "requested_point_count": int(requested_count),
        "actual_point_count": int(chosen.shape[0]),
        "dropout": dropout,
        "view_ids": view_ids.astype(np.int32),
        "view_count": int(view_count),
        "support_visible_mean": float(support.get("visible_mean", 0.0)),
        "support_supported_ratio": float(support.get("supported_ratio", 0.0)),
    }


def coords_to_batched_occ(
    coords: torch.Tensor,
    batch_size: int,
    *,
    resolution: int = 64,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    occ = torch.zeros((batch_size, 1, resolution, resolution, resolution), device=device, dtype=dtype)
    if coords.numel() == 0:
        return occ
    coords = coords.to(device=device, dtype=torch.long)
    valid = (
        (coords[:, 0] >= 0)
        & (coords[:, 0] < batch_size)
        & (coords[:, 1:] >= 0).all(dim=1)
        & (coords[:, 1:] < resolution).all(dim=1)
    )
    coords = coords[valid]
    if coords.numel():
        occ[coords[:, 0], 0, coords[:, 1], coords[:, 2], coords[:, 3]] = 1.0
    return occ


def partial_latent_stats(
    coords: torch.Tensor,
    batch_size: int,
    *,
    weights: torch.Tensor | None = None,
    latent_resolution: int = 16,
    source_resolution: int = 64,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    mask = torch.zeros((batch_size, 1, latent_resolution, latent_resolution, latent_resolution), device=device)
    conf = torch.zeros_like(mask)
    if coords.numel() == 0:
        return mask, conf
    coords = coords.to(device=device, dtype=torch.long)
    if weights is None:
        weights = torch.ones((coords.shape[0],), device=device, dtype=conf.dtype)
    else:
        weights = weights.to(device=device, dtype=conf.dtype).reshape(-1)
        if weights.shape[0] != coords.shape[0]:
            raise ValueError(f"weights length {weights.shape[0]} does not match coords {coords.shape[0]}")
    scale = source_resolution // latent_resolution
    latent = torch.div(coords[:, 1:], scale, rounding_mode="floor").clamp(0, latent_resolution - 1)
    batch = coords[:, 0].clamp(0, batch_size - 1)
    mask[batch, 0, latent[:, 0], latent[:, 1], latent[:, 2]] = 1.0
    conf.index_put_(
        (batch, torch.zeros_like(batch), latent[:, 0], latent[:, 1], latent[:, 2]),
        weights,
        accumulate=True,
    )
    conf = torch.log1p(conf) / math.log(8.0)
    conf = conf.clamp(0.0, 1.0)
    return mask, conf


def sparse_overlap_metrics(pred_coords: np.ndarray, target_coords: np.ndarray) -> Dict[str, float | int]:
    pred_xyz = pred_coords[:, -3:].astype(np.int32) if pred_coords.size else np.zeros((0, 3), dtype=np.int32)
    target_xyz = target_coords[:, -3:].astype(np.int32) if target_coords.size else np.zeros((0, 3), dtype=np.int32)
    pred_set = set(map(tuple, pred_xyz.tolist()))
    target_set = set(map(tuple, target_xyz.tolist()))
    inter = len(pred_set & target_set)
    union = len(pred_set | target_set)
    return {
        "pred_unique": len(pred_set),
        "target_unique": len(target_set),
        "intersection": inter,
        "iou": float(inter / union) if union else 0.0,
        "target_recall": float(inter / len(target_set)) if target_set else 0.0,
        "pred_precision": float(inter / len(pred_set)) if pred_set else 0.0,
    }


class SparsePointPriorCond(nn.Module):
    """Map partial TRELLIS sparse-structure latent to flow condition tokens."""

    def __init__(self, latent_channels: int = 8, cond_channels: int = 1024, grid_resolution: int = 16):
        super().__init__()
        self.latent_channels = int(latent_channels)
        self.cond_channels = int(cond_channels)
        self.grid_resolution = int(grid_resolution)
        coords = torch.stack(
            torch.meshgrid(
                torch.arange(grid_resolution),
                torch.arange(grid_resolution),
                torch.arange(grid_resolution),
                indexing="ij",
            ),
            dim=-1,
        ).float()
        coords = (coords + 0.5) / float(grid_resolution) - 0.5
        self.register_buffer("grid_xyz", coords.reshape(1, -1, 3), persistent=False)
        in_dim = self.latent_channels + 1 + 1 + 3
        self.grid_tokens = nn.Parameter(torch.randn(1, grid_resolution**3, cond_channels) * 1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, cond_channels),
            nn.SiLU(),
            nn.Linear(cond_channels, cond_channels),
        )
        self.norm = nn.LayerNorm(cond_channels)

    def forward(self, partial_latent: torch.Tensor, latent_mask: torch.Tensor, confidence: torch.Tensor) -> torch.Tensor:
        if partial_latent.ndim != 5:
            raise ValueError(f"partial_latent should be [B,C,D,H,W], got {tuple(partial_latent.shape)}")
        b, c, d, h, w = partial_latent.shape
        if (d, h, w) != (self.grid_resolution, self.grid_resolution, self.grid_resolution):
            raise ValueError(f"latent grid mismatch: got {(d, h, w)}, expected {self.grid_resolution}")
        latent_tokens = partial_latent.permute(0, 2, 3, 4, 1).reshape(b, -1, c)
        mask_tokens = latent_mask.reshape(b, 1, -1).transpose(1, 2)
        conf_tokens = confidence.reshape(b, 1, -1).transpose(1, 2)
        xyz = self.grid_xyz.to(device=partial_latent.device, dtype=partial_latent.dtype).expand(b, -1, -1)
        x = torch.cat([latent_tokens, mask_tokens, conf_tokens, xyz], dim=-1)
        return self.norm(self.grid_tokens.to(device=x.device, dtype=x.dtype).expand(b, -1, -1) + self.mlp(x))
