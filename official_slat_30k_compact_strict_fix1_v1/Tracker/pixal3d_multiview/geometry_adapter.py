from __future__ import annotations

import torch
from torch import nn


GEOMETRY_FEATURE_DIM = 17


class GeometryConsistencyAdapter(nn.Module):
    """Map explicit per-voxel geometry consistency features into z_proj residuals."""

    def __init__(
        self,
        feature_dim: int,
        geometry_dim: int = GEOMETRY_FEATURE_DIM,
        hidden_dim: int = 256,
        dropout: float = 0.0,
        residual_scale: float = 1.0,
    ):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.geometry_dim = int(geometry_dim)
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)

        self.net = nn.Sequential(
            nn.LayerNorm(self.geometry_dim),
            nn.Linear(self.geometry_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.feature_dim),
        )
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_scale), dtype=torch.float32))

        # Start as an exact no-op so enabling the adapter does not abruptly shift
        # the sparse-flow condition before training.
        final = self.net[-1]
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def forward(self, z_proj: torch.Tensor, geometry_features: torch.Tensor) -> tuple[torch.Tensor, dict]:
        if z_proj.ndim != 3 or z_proj.shape[0] != 1:
            raise ValueError(f"z_proj should be [1,N,C], got {tuple(z_proj.shape)}")
        if geometry_features.ndim != 2:
            raise ValueError(f"geometry_features should be [N,G], got {tuple(geometry_features.shape)}")
        if geometry_features.shape[0] != z_proj.shape[1]:
            raise ValueError("geometry feature count must match z_proj point count")
        if geometry_features.shape[1] != self.geometry_dim:
            raise ValueError(f"geometry dim mismatch: got {geometry_features.shape[1]}, expected {self.geometry_dim}")
        if z_proj.shape[-1] != self.feature_dim:
            raise ValueError(f"feature dim mismatch: got {z_proj.shape[-1]}, expected {self.feature_dim}")

        base_dtype = z_proj.dtype
        geom = geometry_features.to(device=z_proj.device, dtype=torch.float32)
        residual = self.net(geom).to(dtype=base_dtype)
        out = z_proj + self.residual_scale.to(dtype=base_dtype) * residual.unsqueeze(0)
        stats = {
            "enabled": True,
            "type": "mlp_residual",
            "feature_dim": self.feature_dim,
            "geometry_dim": self.geometry_dim,
            "hidden_dim": self.hidden_dim,
            "residual_scale": float(self.residual_scale.detach().cpu().item()),
            "geometry_abs_mean": float(geom.abs().mean().detach().cpu().item()) if geom.numel() else 0.0,
            "residual_abs_mean": float(residual.abs().mean().detach().cpu().item()) if residual.numel() else 0.0,
        }
        return out, stats
