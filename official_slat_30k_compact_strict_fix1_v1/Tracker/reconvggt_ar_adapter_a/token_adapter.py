from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn


@dataclass(frozen=True)
class TokenLayout:
    layer_index: int
    shape: tuple[int, ...]
    batch: int | None
    views: int | None
    tokens: int | None
    channels: int | None
    prefix_tokens: int
    spatial_tokens: int | None
    spatial_side: int | None
    square_spatial_grid: bool
    image_resolution: int | None
    pixel_per_token: float | None


class ZeroInitResidualTokenAdapter(nn.Module):
    """Zero-init residual MLP for VGGT aggregated tokens.

    The module is intentionally identity at initialization:

        y = x + zero_init(MLP(x))

    This lets us verify that inserting an adapter around VGGT tokens does not
    change ReconViaGen's sparse/SLAT conditioning before any training.
    """

    def __init__(
        self,
        *,
        token_dims: dict[int, int],
        hidden_dim: int = 512,
        layer_indices: Iterable[int] = (4, 11, 17, 23),
    ) -> None:
        super().__init__()
        selected = [int(i) for i in layer_indices]
        self.layer_indices = tuple(selected)
        self.token_dims = {int(k): int(v) for k, v in token_dims.items()}
        self.hidden_dim = int(hidden_dim)
        blocks: dict[str, nn.Module] = {}
        for layer_idx in self.layer_indices:
            if layer_idx not in self.token_dims:
                raise KeyError(f"Missing token dim for layer {layer_idx}")
            dim = self.token_dims[layer_idx]
            block = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, self.hidden_dim),
                nn.SiLU(),
                nn.Linear(self.hidden_dim, dim),
            )
            last = block[-1]
            assert isinstance(last, nn.Linear)
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)
            blocks[str(layer_idx)] = block
        self.blocks = nn.ModuleDict(blocks)

    @classmethod
    def from_tokens(
        cls,
        aggregated_tokens_list: list[torch.Tensor],
        *,
        hidden_dim: int = 512,
        layer_indices: Iterable[int] = (4, 11, 17, 23),
    ) -> "ZeroInitResidualTokenAdapter":
        token_dims: dict[int, int] = {}
        for layer_idx in layer_indices:
            x = aggregated_tokens_list[int(layer_idx)]
            if x.ndim != 4:
                raise ValueError(f"Expected layer {layer_idx} token shape [B,V,T,C], got {tuple(x.shape)}")
            token_dims[int(layer_idx)] = int(x.shape[-1])
        return cls(token_dims=token_dims, hidden_dim=hidden_dim, layer_indices=layer_indices)

    def forward(self, aggregated_tokens_list: list[torch.Tensor]) -> list[torch.Tensor]:
        out = list(aggregated_tokens_list)
        for layer_idx in self.layer_indices:
            x = out[layer_idx]
            block = self.blocks[str(layer_idx)]
            residual = block(x)
            out[layer_idx] = x + residual.to(dtype=x.dtype)
        return out

    def metadata(self) -> dict:
        return {
            "type": self.__class__.__name__,
            "layer_indices": list(self.layer_indices),
            "token_dims": dict(self.token_dims),
            "hidden_dim": self.hidden_dim,
            "zero_init_residual": True,
        }


class ProjectionAwareSpatialTokenAdapter(nn.Module):
    """Zero-init pose/projection-aware adapter for VGGT spatial tokens.

    Prefix/global/register tokens are copied unchanged.  Spatial tokens receive
    a zero-initialized feature-conditioned bias:

        y_spatial = x_spatial + zero_init(MLP(projection_features))

    With zero initialization this module is exactly identity, which keeps B
    stage initialization comparable with A stage and ReconViaGen stock behavior.
    """

    def __init__(
        self,
        *,
        token_dims: dict[int, int],
        feature_dim: int,
        hidden_dim: int = 512,
        layer_indices: Iterable[int] = (4, 11, 17, 23),
        prefix_tokens: int = 5,
        mode: str = "bias",
        gate_feature_index: int | None = None,
        gate_power: float = 1.0,
    ) -> None:
        super().__init__()
        if mode != "bias":
            raise ValueError(f"Unsupported projection adapter mode: {mode}")
        selected = [int(i) for i in layer_indices]
        self.layer_indices = tuple(selected)
        self.token_dims = {int(k): int(v) for k, v in token_dims.items()}
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.prefix_tokens = int(prefix_tokens)
        self.mode = str(mode)
        self.gate_feature_index = None if gate_feature_index is None else int(gate_feature_index)
        self.gate_power = float(gate_power)
        blocks: dict[str, nn.Module] = {}
        for layer_idx in self.layer_indices:
            if layer_idx not in self.token_dims:
                raise KeyError(f"Missing token dim for layer {layer_idx}")
            dim = self.token_dims[layer_idx]
            block = nn.Sequential(
                nn.LayerNorm(self.feature_dim),
                nn.Linear(self.feature_dim, self.hidden_dim),
                nn.SiLU(),
                nn.Linear(self.hidden_dim, dim),
            )
            last = block[-1]
            assert isinstance(last, nn.Linear)
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)
            blocks[str(layer_idx)] = block
        self.blocks = nn.ModuleDict(blocks)

    @classmethod
    def from_tokens(
        cls,
        aggregated_tokens_list: list[torch.Tensor],
        *,
        feature_dim: int,
        hidden_dim: int = 512,
        layer_indices: Iterable[int] = (4, 11, 17, 23),
        prefix_tokens: int = 5,
        mode: str = "bias",
        gate_feature_index: int | None = None,
        gate_power: float = 1.0,
    ) -> "ProjectionAwareSpatialTokenAdapter":
        token_dims: dict[int, int] = {}
        for layer_idx in layer_indices:
            x = aggregated_tokens_list[int(layer_idx)]
            if x.ndim != 4:
                raise ValueError(f"Expected layer {layer_idx} token shape [B,V,T,C], got {tuple(x.shape)}")
            token_dims[int(layer_idx)] = int(x.shape[-1])
        return cls(
            token_dims=token_dims,
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            layer_indices=layer_indices,
            prefix_tokens=prefix_tokens,
            mode=mode,
            gate_feature_index=gate_feature_index,
            gate_power=gate_power,
        )

    def apply_feature_gate(self, bias: torch.Tensor, projection_features: torch.Tensor) -> torch.Tensor:
        if self.gate_feature_index is None:
            return bias
        idx = int(self.gate_feature_index)
        if idx < 0:
            idx = int(projection_features.shape[-1]) + idx
        if idx < 0 or idx >= int(projection_features.shape[-1]):
            raise IndexError(
                f"gate_feature_index={self.gate_feature_index} resolves to {idx}, "
                f"but projection feature dim is {projection_features.shape[-1]}"
            )
        gate = projection_features[..., idx : idx + 1].float().clamp(0.0, 1.0)
        if self.gate_power != 1.0:
            gate = gate.pow(float(self.gate_power))
        return bias * gate.to(device=bias.device, dtype=bias.dtype)

    def forward(self, aggregated_tokens_list: list[torch.Tensor], projection_features: torch.Tensor) -> list[torch.Tensor]:
        if projection_features.ndim != 4:
            raise ValueError(
                f"Expected projection_features shape [B,V,S,F], got {tuple(projection_features.shape)}"
            )
        out = list(aggregated_tokens_list)
        for layer_idx in self.layer_indices:
            x = out[layer_idx]
            if x.ndim != 4:
                raise ValueError(f"Expected layer {layer_idx} token shape [B,V,T,C], got {tuple(x.shape)}")
            b, v, token_count, _ = x.shape
            spatial = token_count - self.prefix_tokens
            if spatial <= 0:
                raise ValueError(f"Layer {layer_idx} has no spatial tokens after prefix={self.prefix_tokens}")
            if tuple(projection_features.shape[:3]) != (b, v, spatial):
                raise ValueError(
                    "projection feature layout mismatch for layer "
                    f"{layer_idx}: features={tuple(projection_features.shape)}, "
                    f"tokens={tuple(x.shape)}, prefix={self.prefix_tokens}"
            )
            block = self.blocks[str(layer_idx)]
            bias = self.apply_feature_gate(block(projection_features), projection_features).to(dtype=x.dtype)
            prefix = x[:, :, : self.prefix_tokens]
            spatial_tokens = x[:, :, self.prefix_tokens :] + bias
            out[layer_idx] = torch.cat((prefix, spatial_tokens), dim=2)
        return out

    def metadata(self) -> dict:
        return {
            "type": self.__class__.__name__,
            "layer_indices": list(self.layer_indices),
            "token_dims": dict(self.token_dims),
            "feature_dim": self.feature_dim,
            "hidden_dim": self.hidden_dim,
            "prefix_tokens": self.prefix_tokens,
            "mode": self.mode,
            "gate_feature_index": self.gate_feature_index,
            "gate_power": self.gate_power,
            "zero_init_spatial_bias": True,
            "prefix_tokens_unchanged": True,
        }


def parse_layer_indices(spec: str) -> list[int]:
    layers = [int(part.strip()) for part in str(spec).split(",") if part.strip()]
    if not layers:
        raise ValueError("layer spec must contain at least one layer index")
    return layers


def infer_token_layout(
    layer_index: int,
    tokens: torch.Tensor,
    *,
    prefix_tokens: int = 5,
    image_resolution: int | None = 518,
) -> TokenLayout:
    shape = tuple(int(v) for v in tokens.shape)
    batch = views = token_count = channels = None
    spatial_tokens = spatial_side = None
    square_spatial_grid = False
    pixel_per_token = None
    if tokens.ndim == 4:
        batch, views, token_count, channels = shape
        spatial_tokens = max(0, token_count - int(prefix_tokens))
        side = int(round(spatial_tokens ** 0.5))
        square_spatial_grid = spatial_tokens > 0 and side * side == spatial_tokens
        if square_spatial_grid:
            spatial_side = side
            if image_resolution is not None and side > 0:
                pixel_per_token = float(image_resolution) / float(side)
    return TokenLayout(
        layer_index=int(layer_index),
        shape=shape,
        batch=batch,
        views=views,
        tokens=token_count,
        channels=channels,
        prefix_tokens=int(prefix_tokens),
        spatial_tokens=spatial_tokens,
        spatial_side=spatial_side,
        square_spatial_grid=bool(square_spatial_grid),
        image_resolution=image_resolution,
        pixel_per_token=pixel_per_token,
    )


def max_abs_tree_diff(a, b) -> float:
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        if a.shape != b.shape:
            raise ValueError(f"Shape mismatch: {tuple(a.shape)} vs {tuple(b.shape)}")
        return float((a.detach().float() - b.detach().float()).abs().max().item()) if a.numel() else 0.0
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            raise ValueError(f"Dict key mismatch: {sorted(a)} vs {sorted(b)}")
        return max((max_abs_tree_diff(a[k], b[k]) for k in sorted(a)), default=0.0)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            raise ValueError(f"Sequence length mismatch: {len(a)} vs {len(b)}")
        return max((max_abs_tree_diff(x, y) for x, y in zip(a, b)), default=0.0)
    raise TypeError(f"Unsupported diff types: {type(a)} vs {type(b)}")
