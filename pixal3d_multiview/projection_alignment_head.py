from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


GEOM_MODES = ("full", "no_xyz", "uv_depth_only", "support_only")


class ProjectionAlignmentHead(nn.Module):
    """View-level 2D-3D projection alignment head.

    The head is independent from sparse flow. It predicts:
    - align_logits[v, n]: whether view v is a reliable 2D-3D correspondence for voxel n.
    - attn[v, n]: a view softmax derived from align_logits.
    - voxel_logits[n]: whether voxel n is close to the target sparse structure.

    The default geometry mode should be uv_depth_only/no_xyz for real experiments.
    full is useful only as a shortcut upper-bound diagnostic because it includes
    canonical xyz.
    """

    def __init__(
        self,
        feature_dim: int,
        geom_dim: int = 11,
        reduced_dim: int = 128,
        hidden_dim: int = 256,
        match_dim: int = 128,
        dropout: float = 0.0,
        geom_mode: str = "uv_depth_only",
    ):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.geom_dim = int(geom_dim)
        self.reduced_dim = int(reduced_dim)
        self.hidden_dim = int(hidden_dim)
        self.match_dim = int(match_dim)
        self.dropout = float(dropout)
        self.geom_mode = str(geom_mode)
        if self.geom_mode not in GEOM_MODES:
            raise ValueError(f"Unknown geom_mode={self.geom_mode!r}; valid={GEOM_MODES}")

        self.feature_reduce = nn.Linear(self.feature_dim, self.reduced_dim)
        self.view_encoder = nn.Sequential(
            nn.LayerNorm(self.reduced_dim + self.geom_dim),
            nn.Linear(self.reduced_dim + self.geom_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
        )
        self.align_head = nn.Linear(self.hidden_dim, 1)
        self.voxel_head = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, 1),
        )
        self.match_embedding_head = nn.Sequential(
            nn.LayerNorm(self.reduced_dim + self.geom_dim),
            nn.Linear(self.reduced_dim + self.geom_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.match_dim),
        )
        self.match_logit_head = nn.Sequential(
            nn.LayerNorm(self.reduced_dim + self.geom_dim),
            nn.Linear(self.reduced_dim + self.geom_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, 1),
        )

    @property
    def config(self) -> dict[str, Any]:
        return {
            "feature_dim": self.feature_dim,
            "geom_dim": self.geom_dim,
            "reduced_dim": self.reduced_dim,
            "hidden_dim": self.hidden_dim,
            "match_dim": self.match_dim,
            "dropout": self.dropout,
            "geom_mode": self.geom_mode,
        }

    def filter_geom(self, view_geom: torch.Tensor) -> torch.Tensor:
        geom = view_geom.float()
        if geom.shape[-1] != self.geom_dim:
            raise ValueError(f"geom dim mismatch: got {geom.shape[-1]}, expected {self.geom_dim}")
        if self.geom_mode == "full":
            return geom
        out = torch.zeros_like(geom)
        if self.geom_mode == "no_xyz":
            out[..., :8] = geom[..., :8]
            return out
        if self.geom_mode == "uv_depth_only":
            out[..., 5:8] = geom[..., 5:8]
            return out
        if self.geom_mode == "support_only":
            out[..., :5] = geom[..., :5]
            return out
        raise ValueError(f"Unknown geom_mode={self.geom_mode!r}")

    def forward(
        self,
        sampled_features: torch.Tensor,
        support_weights: torch.Tensor,
        view_geom: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if sampled_features.ndim != 3:
            raise ValueError(f"sampled_features should be [V,N,C], got {tuple(sampled_features.shape)}")
        if support_weights.shape != sampled_features.shape[:2]:
            raise ValueError("support_weights should be [V,N] and match sampled_features")
        if view_geom.shape[:2] != sampled_features.shape[:2]:
            raise ValueError("view_geom should be [V,N,G] and match sampled_features")
        if sampled_features.shape[-1] != self.feature_dim:
            raise ValueError(f"feature dim mismatch: got {sampled_features.shape[-1]}, expected {self.feature_dim}")

        features = sampled_features.float()
        support = support_weights.float().clamp_min(0.0)
        valid = support > 0
        geom = self.filter_geom(view_geom)

        reduced = self.feature_reduce(features)
        view_input = torch.cat([reduced, geom], dim=-1)
        encoded = self.view_encoder(view_input)
        match_embedding = F.normalize(self.match_embedding_head(view_input), dim=-1)
        match_logits = self.match_logit_head(view_input).squeeze(-1)
        align_logits = self.align_head(encoded).squeeze(-1)

        masked_logits = align_logits.masked_fill(~valid, -1.0e4)
        attn = torch.softmax(masked_logits, dim=0) * valid.float()
        attn = attn / attn.sum(dim=0, keepdim=True).clamp_min(1e-6)
        pooled = (attn[..., None] * encoded).sum(dim=0)
        voxel_logits = self.voxel_head(pooled).squeeze(-1)

        return {
            "align_logits": align_logits,
            "attn": attn,
            "voxel_logits": voxel_logits,
            "valid": valid,
            "encoded": encoded,
            "match_embedding": match_embedding,
            "match_logits": match_logits,
        }


def build_projection_alignment_head(
    *,
    feature_dim: int,
    geom_dim: int = 11,
    reduced_dim: int = 128,
    hidden_dim: int = 256,
    match_dim: int = 128,
    dropout: float = 0.0,
    geom_mode: str = "uv_depth_only",
    device: torch.device | str = "cpu",
) -> ProjectionAlignmentHead:
    return ProjectionAlignmentHead(
        feature_dim=feature_dim,
        geom_dim=geom_dim,
        reduced_dim=reduced_dim,
        hidden_dim=hidden_dim,
        match_dim=match_dim,
        dropout=dropout,
        geom_mode=geom_mode,
    ).to(device)


def load_projection_alignment_head(
    path: str | Path,
    *,
    feature_dim: int | None = None,
    device: torch.device | str = "cpu",
) -> ProjectionAlignmentHead:
    state = torch.load(str(path), map_location="cpu")
    config = dict(state.get("projection_alignment_head_config", state.get("config", {})))
    if feature_dim is not None:
        config["feature_dim"] = int(feature_dim)
    if "feature_dim" not in config:
        raise ValueError(f"projection alignment checkpoint has no feature_dim config: {path}")
    model = ProjectionAlignmentHead(**config)
    weights = state.get("projection_alignment_head", state.get("model"))
    if weights is None:
        raise ValueError(f"projection alignment checkpoint has no model state: {path}")
    model.load_state_dict(weights, strict=False)
    return model.to(device)
