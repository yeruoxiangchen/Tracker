from __future__ import annotations

from contextlib import nullcontext
import hashlib
from typing import Any

import torch
from torch import nn


PHYSICAL_FEATURE_NAMES = (
    "prior_occupancy",
    "prior_confidence",
    "prior_log_count",
    "prior_distance",
    "mask_support_fraction",
    "visible_fraction",
    "mask_hit_ratio",
    "outside_visible_ratio",
    "visual_hull_inside",
    "depth_mean",
    "depth_std",
    "x",
    "y",
    "z",
)
POINTPOSE_CONDITION_VERSION = "reconvggt.pointpose_ss_condition.v2"


def feature_schema_hash(feature_names: tuple[str, ...] = PHYSICAL_FEATURE_NAMES) -> str:
    payload = "\n".join(feature_names).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class PointPoseConditionNet(nn.Module):
    """Fuse a spatial AR evidence grid into ReconViaGen SS condition tokens.

    The physical grid keeps its 16^3 cell layout. A stride-2 encoder maps it
    to the 8^3 patch layout used by the SS DiT, then the existing 4096 global
    condition tokens query those spatial tokens. The final projection is
    zero-initialized, so scale=1 is exactly equivalent to the stock condition
    before training.
    """

    def __init__(
        self,
        *,
        feature_dim: int = len(PHYSICAL_FEATURE_NAMES),
        cond_dim: int = 1024,
        hidden_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if feature_dim <= 0 or cond_dim <= 0 or hidden_dim <= 0:
            raise ValueError("feature_dim, cond_dim and hidden_dim must be positive")
        if cond_dim % num_heads:
            raise ValueError(f"cond_dim={cond_dim} must be divisible by num_heads={num_heads}")
        self.feature_dim = int(feature_dim)
        self.cond_dim = int(cond_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        norm_groups = min(32, self.hidden_dim)
        while self.hidden_dim % norm_groups:
            norm_groups -= 1

        self.grid_encoder = nn.Sequential(
            nn.Conv3d(self.feature_dim, self.hidden_dim, 3, padding=1),
            nn.GroupNorm(norm_groups, self.hidden_dim),
            nn.SiLU(),
            nn.Conv3d(self.hidden_dim, self.hidden_dim, 3, stride=2, padding=1),
            nn.GroupNorm(norm_groups, self.hidden_dim),
            nn.SiLU(),
            nn.Conv3d(self.hidden_dim, self.cond_dim, 1),
        )
        self.query_norm = nn.LayerNorm(self.cond_dim)
        self.physical_norm = nn.LayerNorm(self.cond_dim)
        self.cross_attn = nn.MultiheadAttention(
            self.cond_dim,
            self.num_heads,
            dropout=float(dropout),
            batch_first=True,
        )
        self.output_norm = nn.LayerNorm(self.cond_dim)
        self.output_proj = nn.Linear(self.cond_dim, self.cond_dim)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def physical_tokens(self, physical_grid: torch.Tensor) -> torch.Tensor:
        if physical_grid.ndim != 5:
            raise ValueError(f"physical_grid must be [B,F,16,16,16], got {tuple(physical_grid.shape)}")
        if physical_grid.shape[1] != self.feature_dim:
            raise ValueError(
                f"physical feature mismatch: expected {self.feature_dim}, got {physical_grid.shape[1]}"
            )
        if tuple(physical_grid.shape[-3:]) != (16, 16, 16):
            raise ValueError(f"physical grid must be 16^3, got {tuple(physical_grid.shape[-3:])}")
        encoded = self.grid_encoder(physical_grid.float())
        if tuple(encoded.shape[-3:]) != (8, 8, 8):
            raise RuntimeError(f"grid encoder must produce 8^3 tokens, got {tuple(encoded.shape)}")
        return encoded.flatten(2).transpose(1, 2).contiguous()

    def forward(
        self,
        cond_base: torch.Tensor,
        physical_grid: torch.Tensor,
        *,
        scale: float | torch.Tensor = 1.0,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if cond_base.ndim != 3 or cond_base.shape[-1] != self.cond_dim:
            raise ValueError(f"cond_base must be [B,T,{self.cond_dim}], got {tuple(cond_base.shape)}")
        if physical_grid.shape[0] != cond_base.shape[0]:
            raise ValueError(
                "condition/physical batch mismatch: "
                f"cond_base={tuple(cond_base.shape)}, physical_grid={tuple(physical_grid.shape)}"
            )
        phys = self.physical_tokens(physical_grid)
        query = self.query_norm(cond_base.float())
        key_value = self.physical_norm(phys)
        attended, _ = self.cross_attn(query, key_value, key_value, need_weights=False)
        delta = self.output_proj(self.output_norm(attended))
        scale_tensor = torch.as_tensor(scale, device=delta.device, dtype=delta.dtype)
        if scale_tensor.ndim == 1 and scale_tensor.numel() == delta.shape[0]:
            scale_tensor = scale_tensor.reshape(delta.shape[0], 1, 1)
        try:
            torch.broadcast_shapes(tuple(scale_tensor.shape), tuple(delta.shape))
        except RuntimeError as exc:
            raise ValueError(
                f"scale shape {tuple(scale_tensor.shape)} cannot broadcast to delta {tuple(delta.shape)}"
            ) from exc
        fused = cond_base.float() + scale_tensor * delta
        stats = {
            "delta_rms": torch.sqrt((delta * delta).mean().clamp_min(1.0e-12)),
            "delta_abs_max": delta.abs().amax(),
            "physical_token_rms": torch.sqrt((phys * phys).mean().clamp_min(1.0e-12)),
            "attended_rms": torch.sqrt((attended * attended).mean().clamp_min(1.0e-12)),
            "physical_token_shape": tuple(int(value) for value in phys.shape),
        }
        return fused.to(dtype=cond_base.dtype), stats

    def metadata(self) -> dict[str, Any]:
        return {
            "version": POINTPOSE_CONDITION_VERSION,
            "type": type(self).__name__,
            "feature_dim": self.feature_dim,
            "feature_names": list(PHYSICAL_FEATURE_NAMES),
            "feature_schema_hash": feature_schema_hash(),
            "cond_dim": self.cond_dim,
            "hidden_dim": self.hidden_dim,
            "num_heads": self.num_heads,
            "zero_init_output": True,
            "physical_grid": [16, 16, 16],
            "physical_token_grid": [8, 8, 8],
        }


def lora_disabled(flow: nn.Module):
    disable = getattr(flow, "disable_adapter", None)
    return disable() if callable(disable) else nullcontext()


def trainable_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().clone().cpu()
        for name, parameter in module.named_parameters()
        if parameter.requires_grad
    }


def load_partial_state(
    module: nn.Module,
    state: dict[str, torch.Tensor],
    *,
    require_all_trainable: bool,
    allowed_missing_prefixes: tuple[str, ...] = (),
) -> dict[str, Any]:
    current = dict(module.named_parameters())
    unexpected = sorted(set(state) - set(current))
    shape_mismatch = sorted(
        name for name in set(state) & set(current) if tuple(state[name].shape) != tuple(current[name].shape)
    )
    if unexpected or shape_mismatch:
        raise RuntimeError(
            f"checkpoint mismatch: unexpected={unexpected}, shape_mismatch={shape_mismatch}"
        )
    trainable = {name for name, parameter in current.items() if parameter.requires_grad}
    missing_trainable = sorted(trainable - set(state))
    disallowed_missing = [
        name
        for name in missing_trainable
        if not any(name.startswith(prefix) for prefix in allowed_missing_prefixes)
    ]
    if require_all_trainable and disallowed_missing:
        raise RuntimeError(
            "checkpoint is missing trainable parameters: "
            f"missing={disallowed_missing}, allowed_prefixes={allowed_missing_prefixes}"
        )
    result = module.load_state_dict(state, strict=False)
    missing_frozen = sorted(set(result.missing_keys) - trainable)
    return {
        "missing": missing_trainable,
        "missing_trainable": missing_trainable,
        "allowed_missing_trainable": sorted(set(missing_trainable) - set(disallowed_missing)),
        "unexpected": list(result.unexpected_keys),
        "omitted_frozen_parameter_count": len(missing_frozen),
        "loaded": sorted(state),
    }
