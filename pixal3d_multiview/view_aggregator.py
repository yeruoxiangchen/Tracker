from __future__ import annotations

from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F

from .multiview_projection import resize_masks


class ViewGatedAggregator(nn.Module):
    """Per-voxel learned view gate for projected multi-view features.

    The module shares parameters across all voxels. For each voxel it only
    attends over views, so the cost is O(num_voxels * num_views), not global
    attention over every voxel-view token.
    """

    def __init__(
        self,
        feature_dim: int,
        geom_dim: int = 11,
        reduced_dim: int = 128,
        hidden_dim: int = 256,
        dropout: float = 0.0,
        residual_scale: float = 1.0,
    ):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.geom_dim = int(geom_dim)
        self.reduced_dim = int(reduced_dim)
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)

        self.feature_reduce = nn.Linear(self.feature_dim, self.reduced_dim)
        self.gate = nn.Sequential(
            nn.LayerNorm(self.reduced_dim + self.geom_dim),
            nn.Linear(self.reduced_dim + self.geom_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, 1),
        )
        self.delta_proj = nn.Linear(self.feature_dim, self.feature_dim)
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_scale), dtype=torch.float32))

        # Start as the existing weighted-mean condition. The first updates learn
        # how to add a residual without shifting the sparse-flow input abruptly.
        nn.init.zeros_(self.delta_proj.weight)
        nn.init.zeros_(self.delta_proj.bias)

    def forward(
        self,
        base_agg: torch.Tensor,
        sampled_features: torch.Tensor,
        support_weights: torch.Tensor,
        view_geom: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        if base_agg.ndim != 3 or base_agg.shape[0] != 1:
            raise ValueError(f"base_agg should be [1,N,C], got {tuple(base_agg.shape)}")
        if sampled_features.ndim != 3:
            raise ValueError(f"sampled_features should be [V,N,C], got {tuple(sampled_features.shape)}")
        if support_weights.shape != sampled_features.shape[:2]:
            raise ValueError("support_weights should be [V,N] and match sampled_features")
        if view_geom.shape[:2] != sampled_features.shape[:2]:
            raise ValueError("view_geom should be [V,N,G] and match sampled_features")
        if sampled_features.shape[-1] != self.feature_dim:
            raise ValueError(f"feature dim mismatch: got {sampled_features.shape[-1]}, expected {self.feature_dim}")
        if view_geom.shape[-1] != self.geom_dim:
            raise ValueError(f"geometry dim mismatch: got {view_geom.shape[-1]}, expected {self.geom_dim}")

        base_dtype = base_agg.dtype
        features = sampled_features.float()
        geom = view_geom.float()
        weights = support_weights.float()
        valid = weights > 0

        reduced = self.feature_reduce(features)
        gate_input = torch.cat([reduced, geom], dim=-1)
        score = self.gate(gate_input).squeeze(-1)
        score = score.masked_fill(~valid, -1.0e4)
        attn = torch.softmax(score, dim=0) * valid.float()
        attn = attn / attn.sum(dim=0, keepdim=True).clamp_min(1e-6)

        learned = (attn[..., None] * features).sum(dim=0)
        delta = self.delta_proj(learned).to(dtype=base_dtype)
        has_support = valid.any(dim=0)
        out = base_agg + self.residual_scale.to(dtype=base_dtype) * delta.unsqueeze(0)
        out = torch.where(has_support[None, :, None], out, base_agg)

        entropy = -(attn.clamp_min(1e-8) * attn.clamp_min(1e-8).log()).sum(dim=0)
        stats = {
            "enabled": True,
            "type": "gated",
            "feature_dim": self.feature_dim,
            "geom_dim": self.geom_dim,
            "reduced_dim": self.reduced_dim,
            "hidden_dim": self.hidden_dim,
            "residual_scale": float(self.residual_scale.detach().cpu().item()),
            "valid_voxel_ratio": float(has_support.float().mean().detach().cpu().item()) if has_support.numel() else 0.0,
            "support_weight_mean": float(weights.mean().detach().cpu().item()) if weights.numel() else 0.0,
            "attention_entropy_mean": float(entropy[has_support].mean().detach().cpu().item()) if has_support.any() else 0.0,
        }
        return out, stats


def sample_view_features_for_aggregation(
    feature_map: torch.Tensor,
    points_obj: torch.Tensor,
    points_2d: torch.Tensor,
    depths: torch.Tensor,
    valid_depth: torch.Tensor,
    *,
    coordinate_size: int,
    masks: Optional[torch.Tensor] = None,
    mask_threshold: float = 0.5,
    front_depth_maps: Optional[torch.Tensor] = None,
    visibility_depth_tolerance: float = 0.03,
    visibility_weight_min: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    """Return per-view sampled features and lightweight geometry tokens.

    sampled_features: [V,N,C]
    support_weights: [V,N]
    view_geom: [V,N,11]
    """
    if feature_map.ndim == 4 and feature_map.shape[1] < feature_map.shape[-1]:
        feature_map = feature_map.permute(0, 3, 1, 2)
    view_count = int(feature_map.shape[0])
    if points_2d.shape[0] != view_count or valid_depth.shape[0] != view_count:
        raise ValueError("feature, point, and valid-depth view counts do not match")

    u = points_2d[..., 0]
    v = points_2d[..., 1]
    in_image = valid_depth & (u >= 0) & (u < coordinate_size) & (v >= 0) & (v < coordinate_size)
    grid = torch.stack(
        [(u + 0.5) / float(coordinate_size) * 2.0 - 1.0, (v + 0.5) / float(coordinate_size) * 2.0 - 1.0],
        dim=-1,
    ).view(view_count, -1, 1, 2)

    sampled = F.grid_sample(feature_map, grid, mode="bilinear", align_corners=False, padding_mode="border")
    sampled = sampled.squeeze(-1).permute(0, 2, 1)

    if masks is not None:
        masks = resize_masks(masks, coordinate_size, feature_map.device)
        mask_values = F.grid_sample(masks, grid, mode="bilinear", align_corners=False, padding_mode="zeros")
        mask_values = mask_values.squeeze(1).squeeze(-1).clamp(0.0, 1.0)
        mask_hit = in_image & (mask_values > float(mask_threshold))
    else:
        mask_values = in_image.float()
        mask_hit = in_image

    visibility_weight = mask_hit.float()
    visibility_stats = {"enabled": False}
    if front_depth_maps is not None:
        if front_depth_maps.ndim == 3:
            front_depth_maps = front_depth_maps[:, None]
        if front_depth_maps.ndim != 4 or front_depth_maps.shape[1] != 1:
            raise ValueError(f"front_depth_maps should be [V,H,W] or [V,1,H,W], got {tuple(front_depth_maps.shape)}")
        if front_depth_maps.shape[-2:] != (coordinate_size, coordinate_size):
            front_depth_maps = F.interpolate(
                front_depth_maps.float(),
                size=(coordinate_size, coordinate_size),
                mode="nearest",
            )
        front_depth_maps = front_depth_maps.to(device=feature_map.device, dtype=torch.float32)
        finite_maps = torch.isfinite(front_depth_maps) & (front_depth_maps > 0)
        front_depth_for_sample = torch.where(finite_maps, front_depth_maps, torch.zeros_like(front_depth_maps))
        front_depth = F.grid_sample(
            front_depth_for_sample,
            grid,
            mode="nearest",
            align_corners=False,
            padding_mode="zeros",
        ).squeeze(1).squeeze(-1)
        front_finite = F.grid_sample(
            finite_maps.float(),
            grid,
            mode="nearest",
            align_corners=False,
            padding_mode="zeros",
        ).squeeze(1).squeeze(-1) > 0.5
        delta = (depths.to(feature_map.device, dtype=torch.float32) - front_depth).abs()
        tol = max(float(visibility_depth_tolerance), 1e-6)
        visibility_weight = (1.0 - delta / tol).clamp_min(0.0)
        visibility_weight = torch.where(front_finite, visibility_weight, torch.zeros_like(visibility_weight))
        visibility_weight = torch.where(
            visibility_weight >= float(visibility_weight_min),
            visibility_weight,
            torch.zeros_like(visibility_weight),
        )
        visibility_weight = visibility_weight * mask_hit.float()
        visibility_stats = {
            "enabled": True,
            "front_depth_coverage": float(finite_maps.float().mean().item()) if finite_maps.numel() else 0.0,
            "visibility_weight_nonzero_ratio": float((visibility_weight > 0).float().mean().item()) if visibility_weight.numel() else 0.0,
            "visibility_weight_mean": float(visibility_weight.mean().item()) if visibility_weight.numel() else 0.0,
        }

    valid_depth_values = depths[valid_depth & torch.isfinite(depths) & (depths > 0)]
    depth_scale = valid_depth_values.median().clamp_min(1e-6) if valid_depth_values.numel() else torch.tensor(1.0, device=depths.device)
    u_norm = (u + 0.5) / float(coordinate_size) * 2.0 - 1.0
    v_norm = (v + 0.5) / float(coordinate_size) * 2.0 - 1.0
    depth_norm = torch.tanh(depths.float() / depth_scale.float())
    xyz = (points_obj.to(device=feature_map.device, dtype=torch.float32) * 2.0).clamp(-1.0, 1.0)
    xyz = xyz[None].expand(view_count, -1, -1)
    view_geom = torch.cat(
        [
            visibility_weight.float()[..., None],
            mask_values.float()[..., None],
            in_image.float()[..., None],
            valid_depth.float()[..., None],
            mask_hit.float()[..., None],
            u_norm.clamp(-2.0, 2.0)[..., None],
            v_norm.clamp(-2.0, 2.0)[..., None],
            depth_norm.clamp(-1.0, 1.0)[..., None],
            xyz,
        ],
        dim=-1,
    )
    stats = {
        "num_views": view_count,
        "num_points": int(points_2d.shape[1]),
        "geom_dim": int(view_geom.shape[-1]),
        "in_image_ratio": float(in_image.float().mean().item()) if in_image.numel() else 0.0,
        "mask_hit_ratio": float(mask_hit.float().mean().item()) if mask_hit.numel() else 0.0,
        "support_weight_mean": float(visibility_weight.float().mean().item()) if visibility_weight.numel() else 0.0,
        "support_weight_nonzero_ratio": float((visibility_weight > 0).float().mean().item()) if visibility_weight.numel() else 0.0,
        "visibility": visibility_stats,
    }
    return sampled, visibility_weight.float(), view_geom, stats
