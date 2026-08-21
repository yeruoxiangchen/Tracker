from __future__ import annotations

import hashlib
import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from reconvggt_ar_adapter_a.pointpose_ss_condition import PHYSICAL_FEATURE_NAMES
from reconvggt_ar_adapter_a.stock_preserving_pointpose_bridge import (
    make_null_physical_grid,
)


SPARSE_ANCHOR_FLOW_VERSION = "reconvggt.sparse_anchor_flow.v1"
FEATURE_INDEX = {name: index for index, name in enumerate(PHYSICAL_FEATURE_NAMES)}
PRIOR_CHANNELS = tuple(FEATURE_INDEX[name] for name in (
    "prior_occupancy",
    "prior_confidence",
    "prior_log_count",
    "prior_distance",
))
PROJECTION_CHANNELS = tuple(range(
    FEATURE_INDEX["mask_support_fraction"],
    FEATURE_INDEX["x"],
))


def _validate_physical_grid(physical_grid: torch.Tensor) -> None:
    expected = (len(PHYSICAL_FEATURE_NAMES), 16, 16, 16)
    if physical_grid.ndim != 5 or tuple(physical_grid.shape[1:]) != expected:
        raise ValueError(
            f"physical_grid must be [B,{expected[0]},16,16,16], "
            f"got {tuple(physical_grid.shape)}"
        )
    if not bool(torch.isfinite(physical_grid).all().item()):
        raise ValueError("physical_grid contains non-finite values")


def make_mask_only_physical_grid(physical_grid: torch.Tensor) -> torch.Tensor:
    """Keep projection/mask evidence and canonical XYZ, but remove sparse points."""

    _validate_physical_grid(physical_grid)
    output = make_null_physical_grid(physical_grid)
    output[:, PROJECTION_CHANNELS] = physical_grid[:, PROJECTION_CHANNELS]
    return output


def make_point_only_physical_grid(physical_grid: torch.Tensor) -> torch.Tensor:
    """Keep sparse-point evidence and canonical XYZ, but remove projection evidence."""

    _validate_physical_grid(physical_grid)
    output = make_null_physical_grid(physical_grid)
    output[:, PRIOR_CHANNELS] = physical_grid[:, PRIOR_CHANNELS]
    return output


def _nonwrapping_shift_3d(values: torch.Tensor, shift: tuple[int, int, int]) -> torch.Tensor:
    if values.ndim != 5:
        raise ValueError(f"values must be [B,C,D,H,W], got {tuple(values.shape)}")
    output = torch.zeros_like(values)
    source: list[slice] = [slice(None), slice(None)]
    target: list[slice] = [slice(None), slice(None)]
    for amount, side in zip(shift, values.shape[-3:]):
        amount = int(amount)
        if abs(amount) >= int(side):
            return output
        if amount >= 0:
            source.append(slice(0, side - amount))
            target.append(slice(amount, side))
        else:
            source.append(slice(-amount, side))
            target.append(slice(0, side + amount))
    output[tuple(target)] = values[tuple(source)]
    return output


def prior_distance_from_occupancy(occupancy: torch.Tensor) -> torch.Tensor:
    """Compute the normalized 16^3 distance transform without wrapping shifts."""

    if occupancy.ndim != 5 or occupancy.shape[1] != 1:
        raise ValueError(f"occupancy must be [B,1,16,16,16], got {tuple(occupancy.shape)}")
    side = int(occupancy.shape[-1])
    if tuple(occupancy.shape[-3:]) != (side, side, side):
        raise ValueError("occupancy must have a cubic spatial shape")
    axis = torch.arange(side, device=occupancy.device, dtype=torch.float32)
    coords = torch.stack(
        torch.meshgrid(axis, axis, axis, indexing="ij"), dim=-1
    ).reshape(-1, 3)
    maximum = max(math.sqrt(3.0) * float(side - 1), 1.0)
    outputs: list[torch.Tensor] = []
    for batch_index in range(int(occupancy.shape[0])):
        occupied = torch.nonzero(
            occupancy[batch_index, 0] > 0.5, as_tuple=False
        ).to(dtype=torch.float32)
        if occupied.numel() == 0:
            distance = torch.ones(coords.shape[0], device=coords.device)
        else:
            distance = torch.cdist(coords, occupied).amin(dim=1) / maximum
        outputs.append(distance.reshape(1, side, side, side).clamp(0.0, 1.0))
    return torch.stack(outputs, dim=0).to(dtype=occupancy.dtype)


def shift_sparse_prior(
    physical_grid: torch.Tensor,
    shift: tuple[int, int, int],
) -> torch.Tensor:
    """Shift only sparse-point evidence while preserving mask, pose and XYZ."""

    _validate_physical_grid(physical_grid)
    output = physical_grid.clone()
    shifted = _nonwrapping_shift_3d(physical_grid[:, PRIOR_CHANNELS[:3]], shift)
    output[:, PRIOR_CHANNELS[:3]] = shifted
    output[:, FEATURE_INDEX["prior_distance"] : FEATURE_INDEX["prior_distance"] + 1] = (
        prior_distance_from_occupancy(
            shifted[:, 0:1]
        )
    )
    return output


def dropout_sparse_prior(
    physical_grid: torch.Tensor,
    keep_ratio: float,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if not 0.0 <= float(keep_ratio) <= 1.0:
        raise ValueError("keep_ratio must be in [0,1]")
    _validate_physical_grid(physical_grid)
    output = physical_grid.clone()
    occupancy = physical_grid[:, FEATURE_INDEX["prior_occupancy"] : FEATURE_INDEX["prior_occupancy"] + 1]
    random_values = torch.rand(
        occupancy.shape,
        device=occupancy.device,
        dtype=torch.float32,
        generator=generator,
    )
    keep = (random_values < float(keep_ratio)).to(dtype=physical_grid.dtype)
    keep = keep * (occupancy > 0.5).to(dtype=physical_grid.dtype)
    for index in PRIOR_CHANNELS[:3]:
        output[:, index : index + 1] = physical_grid[:, index : index + 1] * keep
    output[:, FEATURE_INDEX["prior_distance"] : FEATURE_INDEX["prior_distance"] + 1] = (
        prior_distance_from_occupancy(
            output[:, FEATURE_INDEX["prior_occupancy"] : FEATURE_INDEX["prior_occupancy"] + 1]
        )
    )
    return output


def target_occupancy_64(
    target_coords: torch.Tensor,
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    coords = target_coords
    if coords.ndim == 3:
        if coords.shape[0] != 1:
            raise ValueError("batched target_coords currently supports batch size 1")
        coords = coords[0]
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"target_coords must be [N,3], got {tuple(coords.shape)}")
    device = device or coords.device
    output = torch.zeros((1, 1, 64, 64, 64), device=device, dtype=torch.float32)
    xyz = coords.to(device=device, dtype=torch.long)
    valid = ((xyz >= 0) & (xyz < 64)).all(dim=1)
    xyz = xyz[valid]
    if xyz.numel():
        output[0, 0, xyz[:, 0], xyz[:, 1], xyz[:, 2]] = 1.0
    return output


def build_sparse_anchor_masks(
    physical_grid: torch.Tensor,
    target_coords: torch.Tensor,
    *,
    prior_confidence_min: float = 0.25,
    prior_mask_support_min: float = 0.0,
    anchor_radius_16: int = 1,
    outside_visible_min: float = 0.5,
    outside_ratio_min: float = 0.9,
    negative_surface_margin_64: int = 1,
) -> dict[str, torch.Tensor]:
    """Build mutually exclusive positive, reliable-negative and neutral masks."""

    _validate_physical_grid(physical_grid)
    if physical_grid.shape[0] != 1:
        raise ValueError("sparse-anchor masks currently support batch size 1")
    target64 = target_occupancy_64(target_coords, device=physical_grid.device)
    target16 = F.max_pool3d(target64, kernel_size=4, stride=4) > 0.5
    prior = physical_grid[:, FEATURE_INDEX["prior_occupancy"] : FEATURE_INDEX["prior_occupancy"] + 1] > 0.5
    confidence = physical_grid[:, FEATURE_INDEX["prior_confidence"] : FEATURE_INDEX["prior_confidence"] + 1]
    support = physical_grid[:, FEATURE_INDEX["mask_support_fraction"] : FEATURE_INDEX["mask_support_fraction"] + 1]
    anchor_seed = prior & (confidence >= float(prior_confidence_min))
    anchor_seed &= support >= float(prior_mask_support_min)
    radius = max(0, int(anchor_radius_16))
    if radius:
        anchor_region = F.max_pool3d(
            anchor_seed.float(), kernel_size=2 * radius + 1, stride=1, padding=radius
        ) > 0.5
    else:
        anchor_region = anchor_seed

    visible = physical_grid[:, FEATURE_INDEX["visible_fraction"] : FEATURE_INDEX["visible_fraction"] + 1]
    outside = physical_grid[:, FEATURE_INDEX["outside_visible_ratio"] : FEATURE_INDEX["outside_visible_ratio"] + 1]
    hull = physical_grid[:, FEATURE_INDEX["visual_hull_inside"] : FEATURE_INDEX["visual_hull_inside"] + 1]
    reliable_outside = (
        (visible >= float(outside_visible_min))
        & (outside >= float(outside_ratio_min))
        & (hull < 0.5)
        & (~anchor_region)
    )

    positive16 = anchor_region & target16
    negative16 = reliable_outside & (~target16) & (~positive16)
    neutral16 = ~(positive16 | negative16)
    if bool((positive16 & negative16).any().item()):
        raise RuntimeError("positive and negative 16^3 masks overlap")

    anchor64 = F.interpolate(anchor_region.float(), size=(64, 64, 64), mode="nearest") > 0.5
    outside64 = F.interpolate(reliable_outside.float(), size=(64, 64, 64), mode="nearest") > 0.5
    positive64 = anchor64 & (target64 > 0.5)
    margin = max(0, int(negative_surface_margin_64))
    if margin:
        target_margin64 = F.max_pool3d(
            target64, kernel_size=2 * margin + 1, stride=1, padding=margin
        ) > 0.5
    else:
        target_margin64 = target64 > 0.5
    negative64 = outside64 & (~target_margin64) & (~positive64)
    neutral64 = ~(positive64 | negative64)
    if bool((positive64 & negative64).any().item()):
        raise RuntimeError("positive and negative 64^3 masks overlap")

    return {
        "target64": target64 > 0.5,
        "target16": target16,
        "anchor_seed16": anchor_seed,
        "anchor_region16": anchor_region,
        "reliable_outside16": reliable_outside,
        "positive16": positive16,
        "negative16": negative16,
        "neutral16": neutral16,
        "positive64": positive64,
        "negative64": negative64,
        "neutral64": neutral64,
    }


def adapter_spatial_gate(
    physical_grid: torch.Tensor,
    *,
    prior_confidence_min: float,
    anchor_radius_16: int,
    outside_visible_min: float,
    outside_ratio_min: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    _validate_physical_grid(physical_grid)
    prior = physical_grid[:, FEATURE_INDEX["prior_occupancy"] : FEATURE_INDEX["prior_occupancy"] + 1]
    confidence = physical_grid[:, FEATURE_INDEX["prior_confidence"] : FEATURE_INDEX["prior_confidence"] + 1]
    anchor = ((prior > 0.5) & (confidence >= float(prior_confidence_min))).float()
    radius = max(0, int(anchor_radius_16))
    if radius:
        anchor = F.max_pool3d(anchor, kernel_size=2 * radius + 1, stride=1, padding=radius)
    visible = physical_grid[:, FEATURE_INDEX["visible_fraction"] : FEATURE_INDEX["visible_fraction"] + 1]
    outside_ratio = physical_grid[:, FEATURE_INDEX["outside_visible_ratio"] : FEATURE_INDEX["outside_visible_ratio"] + 1]
    hull = physical_grid[:, FEATURE_INDEX["visual_hull_inside"] : FEATURE_INDEX["visual_hull_inside"] + 1]
    outside = (
        (visible >= float(outside_visible_min))
        & (outside_ratio >= float(outside_ratio_min))
        & (hull < 0.5)
    ).float()
    gate = torch.clamp(anchor + outside, 0.0, 1.0)
    return gate, {"anchor_gate": anchor, "outside_gate": outside}


class SparseAnchorVelocityAdapter(nn.Module):
    """A zero-init, spatially gated velocity residual on the SS 16^3 latent."""

    def __init__(
        self,
        *,
        latent_channels: int = 8,
        feature_dim: int = len(PHYSICAL_FEATURE_NAMES),
        hidden_dim: int = 96,
        prior_confidence_min: float = 0.25,
        anchor_radius_16: int = 1,
        outside_visible_min: float = 0.5,
        outside_ratio_min: float = 0.9,
    ) -> None:
        super().__init__()
        self.latent_channels = int(latent_channels)
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.prior_confidence_min = float(prior_confidence_min)
        self.anchor_radius_16 = int(anchor_radius_16)
        self.outside_visible_min = float(outside_visible_min)
        self.outside_ratio_min = float(outside_ratio_min)
        groups = min(16, self.hidden_dim)
        while self.hidden_dim % groups:
            groups -= 1

        self.physical_encoder = nn.Sequential(
            nn.Conv3d(self.feature_dim, self.hidden_dim, 3, padding=1),
            nn.GroupNorm(groups, self.hidden_dim),
            nn.SiLU(),
            nn.Conv3d(self.hidden_dim, self.hidden_dim, 3, padding=1),
            nn.GroupNorm(groups, self.hidden_dim),
            nn.SiLU(),
        )
        self.state_encoder = nn.Sequential(
            nn.Conv3d(2 * self.latent_channels, self.hidden_dim, 3, padding=1),
            nn.GroupNorm(groups, self.hidden_dim),
            nn.SiLU(),
        )
        self.time_mlp = nn.Sequential(
            nn.Linear(16, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.fusion = nn.Sequential(
            nn.Conv3d(2 * self.hidden_dim, self.hidden_dim, 3, padding=1),
            nn.GroupNorm(groups, self.hidden_dim),
            nn.SiLU(),
            nn.Conv3d(self.hidden_dim, self.hidden_dim, 3, padding=1),
            nn.SiLU(),
        )
        self.output = nn.Conv3d(self.hidden_dim, self.latent_channels, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    @staticmethod
    def _time_features(t: torch.Tensor) -> torch.Tensor:
        if t.ndim != 1:
            raise ValueError(f"t must be [B], got {tuple(t.shape)}")
        value = t.float() / 1000.0
        frequencies = torch.arange(1, 9, device=t.device, dtype=torch.float32)
        angles = 2.0 * math.pi * value[:, None] * frequencies[None]
        return torch.cat((torch.sin(angles), torch.cos(angles)), dim=-1)

    def centered_physical(self, physical_grid: torch.Tensor) -> torch.Tensor:
        null_grid = make_null_physical_grid(physical_grid)
        paired = torch.cat((physical_grid.float(), null_grid.float()), dim=0)
        encoded = self.physical_encoder(paired)
        physical, null = encoded.chunk(2, dim=0)
        return physical - null

    def forward(
        self,
        x_t: torch.Tensor,
        stock_velocity: torch.Tensor,
        t: torch.Tensor,
        physical_grid: torch.Tensor,
        *,
        scale: float | torch.Tensor = 1.0,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        _validate_physical_grid(physical_grid)
        expected = (self.latent_channels, 16, 16, 16)
        if x_t.ndim != 5 or tuple(x_t.shape[1:]) != expected:
            raise ValueError(f"x_t must be [B,{self.latent_channels},16,16,16], got {tuple(x_t.shape)}")
        if stock_velocity.shape != x_t.shape or physical_grid.shape[0] != x_t.shape[0]:
            raise ValueError("x_t, stock_velocity and physical_grid batch/spatial shapes must match")

        gate, gate_parts = adapter_spatial_gate(
            physical_grid,
            prior_confidence_min=self.prior_confidence_min,
            anchor_radius_16=self.anchor_radius_16,
            outside_visible_min=self.outside_visible_min,
            outside_ratio_min=self.outside_ratio_min,
        )
        physical = self.centered_physical(physical_grid)
        state = self.state_encoder(torch.cat((x_t.float(), stock_velocity.float()), dim=1))
        time = self.time_mlp(self._time_features(t)).view(t.shape[0], self.hidden_dim, 1, 1, 1)
        hidden = self.fusion(torch.cat((physical, state + time), dim=1))
        raw_delta = self.output(hidden)
        delta = raw_delta * gate
        scale_tensor = torch.as_tensor(scale, device=delta.device, dtype=delta.dtype)
        if scale_tensor.ndim == 1 and scale_tensor.numel() == delta.shape[0]:
            scale_tensor = scale_tensor.view(-1, 1, 1, 1, 1)
        delta = delta * scale_tensor
        stats = {
            "delta_rms": delta.float().square().mean().sqrt(),
            "delta_abs_max": delta.float().abs().amax(),
            "physical_rms": physical.float().square().mean().sqrt(),
            "gate_ratio": gate.float().mean(),
            "anchor_gate_ratio": gate_parts["anchor_gate"].float().mean(),
            "outside_gate_ratio": gate_parts["outside_gate"].float().mean(),
        }
        return delta.to(dtype=x_t.dtype), stats

    def metadata(self) -> dict[str, Any]:
        schema = hashlib.sha256("\n".join(PHYSICAL_FEATURE_NAMES).encode("utf-8")).hexdigest()
        return {
            "version": SPARSE_ANCHOR_FLOW_VERSION,
            "type": type(self).__name__,
            "latent_resolution": 16,
            "latent_channels": self.latent_channels,
            "feature_names": list(PHYSICAL_FEATURE_NAMES),
            "feature_schema_hash": schema,
            "hidden_dim": self.hidden_dim,
            "prior_confidence_min": self.prior_confidence_min,
            "anchor_radius_16": self.anchor_radius_16,
            "outside_visible_min": self.outside_visible_min,
            "outside_ratio_min": self.outside_ratio_min,
            "zero_init_output": True,
            "stock_flow_frozen": True,
            "spatial_gate": "dilated_prior_anchor_or_reliable_outside",
        }


class SparseAnchorSSFlowModel(nn.Module):
    """Frozen stock SS Flow plus a gated 16^3 sparse-anchor velocity residual."""

    def __init__(self, stock_flow: nn.Module, adapter: SparseAnchorVelocityAdapter) -> None:
        super().__init__()
        self.stock_flow = stock_flow
        self.adapter = adapter
        for parameter in self.stock_flow.parameters():
            parameter.requires_grad = False
        self.stock_flow.eval()

    def stock_prediction(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            return self.stock_flow(x_t, t, cond)

    def adapt_from_stock(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        stock_velocity: torch.Tensor,
        physical_grid: torch.Tensor,
        *,
        physical_scale: float = 1.0,
        physical_present: bool = True,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if not physical_present or float(physical_scale) == 0.0:
            zero = stock_velocity.new_zeros((), dtype=torch.float32)
            return stock_velocity, {
                "delta_rms": zero,
                "delta_abs_max": zero,
                "physical_rms": zero,
                "gate_ratio": zero,
                "anchor_gate_ratio": zero,
                "outside_gate_ratio": zero,
            }
        evidence = physical_grid[:, : FEATURE_INDEX["x"]]
        if int(torch.count_nonzero(evidence).item()) == 0:
            zero = stock_velocity.new_zeros((), dtype=torch.float32)
            return stock_velocity, {
                "delta_rms": zero,
                "delta_abs_max": zero,
                "physical_rms": zero,
                "gate_ratio": zero,
                "anchor_gate_ratio": zero,
                "outside_gate_ratio": zero,
            }
        delta, stats = self.adapter(
            x_t,
            stock_velocity,
            t,
            physical_grid,
            scale=float(physical_scale),
        )
        return stock_velocity + delta, stats

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
        physical_grid: torch.Tensor,
        *,
        corrupted_physical_grid: torch.Tensor | None = None,
        physical_scale: float = 1.0,
        physical_present: bool = True,
    ) -> tuple[Any, ...]:
        stock = self.stock_prediction(x_t, t, cond)
        adapted, stats = self.adapt_from_stock(
            x_t,
            t,
            stock,
            physical_grid,
            physical_scale=float(physical_scale),
            physical_present=bool(physical_present),
        )
        if corrupted_physical_grid is not None:
            corrupted, corrupted_stats = self.adapt_from_stock(
                x_t,
                t,
                stock,
                corrupted_physical_grid,
                physical_scale=float(physical_scale),
                physical_present=bool(physical_present),
            )
            return adapted, corrupted, stock, stats, corrupted_stats
        return adapted, stock, stats
