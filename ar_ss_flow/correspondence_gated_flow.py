from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

# Support both `python -m ar_ss_flow.<module>` and direct script execution.
TRACKER_ROOT = Path(__file__).resolve().parents[1]
if str(TRACKER_ROOT) not in sys.path:
    sys.path.insert(0, str(TRACKER_ROOT))

from ar_ss_flow.correspondence_lifting import (
    CORRESPONDENCE_CONFIDENCE_INDEX,
    CORRESPONDENCE_METADATA_NAMES,
)
from ar_ss_flow.pose_lifting import SUPPORT_METADATA_INDEX


CORRESPONDENCE_GATED_FLOW_VERSION = "ar_ss_flow.correspondence_gated_velocity.v1"


class CorrespondenceGatedVelocityAdapter(nn.Module):
    """Same-voxel SS velocity residual gated by verified local correspondence."""

    def __init__(
        self,
        *,
        visual_channels: int,
        latent_channels: int = 8,
        hidden_dim: int = 96,
        metadata_channels: int = len(CORRESPONDENCE_METADATA_NAMES),
        confidence_threshold: float = 0.0,
    ) -> None:
        super().__init__()
        self.visual_channels = int(visual_channels)
        self.latent_channels = int(latent_channels)
        self.hidden_dim = int(hidden_dim)
        self.metadata_channels = int(metadata_channels)
        self.confidence_threshold = float(confidence_threshold)
        self.state_projection = nn.Conv3d(2 * latent_channels + 1, hidden_dim, 1)
        self.visual_projection = nn.Conv3d(visual_channels, hidden_dim, 1, bias=False)
        self.metadata_projection = nn.Conv3d(metadata_channels, hidden_dim, 1)
        self.fusion = nn.Sequential(
            nn.Conv3d(5 * hidden_dim, 2 * hidden_dim, 1),
            nn.SiLU(),
            nn.Conv3d(2 * hidden_dim, hidden_dim, 1),
            nn.SiLU(),
        )
        self.output = nn.Conv3d(hidden_dim, latent_channels, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def metadata(self) -> dict[str, Any]:
        return {
            "version": CORRESPONDENCE_GATED_FLOW_VERSION,
            "fusion": "same_voxel_16x16x16_pointwise_mlp",
            "global_attention": False,
            "local_neighborhood": 1,
            "visual_channels": self.visual_channels,
            "metadata_channels": self.metadata_channels,
            "metadata_names": list(CORRESPONDENCE_METADATA_NAMES),
            "hidden_dim": self.hidden_dim,
            "latent_channels": self.latent_channels,
            "zero_init_output": True,
            "gate": "weighted_support_times_correspondence_confidence",
            "confidence_threshold": self.confidence_threshold,
            "time_normalization": "t_div_1000",
        }

    def forward(
        self,
        x_t: torch.Tensor,
        stock_velocity: torch.Tensor,
        t: torch.Tensor,
        visual_volume: torch.Tensor,
        metadata: torch.Tensor,
        *,
        scale: float = 1.0,
        physical_present: bool = True,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        expected_spatial = (16, 16, 16)
        if x_t.shape != stock_velocity.shape or tuple(x_t.shape[-3:]) != expected_spatial:
            raise ValueError("x_t/stock_velocity must be aligned [B,8,16,16,16]")
        if visual_volume.shape[:2] != (x_t.shape[0], self.visual_channels):
            raise ValueError(f"invalid visual volume shape {tuple(visual_volume.shape)}")
        if metadata.shape[:2] != (x_t.shape[0], self.metadata_channels):
            raise ValueError(f"invalid metadata shape {tuple(metadata.shape)}")
        support = metadata[:, SUPPORT_METADATA_INDEX : SUPPORT_METADATA_INDEX + 1].float()
        confidence = metadata[
            :, CORRESPONDENCE_CONFIDENCE_INDEX : CORRESPONDENCE_CONFIDENCE_INDEX + 1
        ].float()
        gate = support * confidence
        if self.confidence_threshold > 0.0:
            gate = gate * confidence.ge(self.confidence_threshold).float()
        if not physical_present or float(scale) == 0.0:
            zero = torch.zeros_like(stock_velocity)
            return zero, {
                "delta_rms": zero.float().square().mean().sqrt(),
                "delta_abs_max": zero.float().abs().max(),
                "gate_ratio": gate.gt(0).float().mean(),
                "gate_mean": gate.mean(),
                "confidence_mean": confidence.mean(),
            }
        batch = int(x_t.shape[0])
        t_channel = (t.float() / 1000.0).reshape(batch, 1, 1, 1, 1).expand(
            batch, 1, *expected_spatial
        )
        state = torch.cat((x_t.float(), stock_velocity.float(), t_channel), dim=1)
        visual = visual_volume.float().permute(0, 2, 3, 4, 1)
        visual = F.layer_norm(visual, (self.visual_channels,)).permute(
            0, 4, 1, 2, 3
        ).contiguous()
        state_hidden = self.state_projection(state)
        visual_hidden = self.visual_projection(visual)
        metadata_hidden = self.metadata_projection(metadata.float())
        interaction = torch.cat(
            (
                state_hidden,
                visual_hidden,
                state_hidden * visual_hidden,
                (state_hidden - visual_hidden).abs(),
                metadata_hidden,
            ),
            dim=1,
        )
        delta = self.output(self.fusion(interaction)) * gate * float(scale)
        return delta.to(dtype=stock_velocity.dtype), {
            "delta_rms": delta.float().square().mean().sqrt(),
            "delta_abs_max": delta.float().abs().max(),
            "gate_ratio": gate.gt(0).float().mean(),
            "gate_mean": gate.mean(),
            "confidence_mean": confidence.mean(),
            "support_mean": support.mean(),
        }


class CorrespondenceGatedSSFlowModel(nn.Module):
    def __init__(
        self,
        stock_flow: nn.Module,
        adapter: CorrespondenceGatedVelocityAdapter,
    ) -> None:
        super().__init__()
        self.stock_flow = stock_flow
        self.adapter = adapter
        self.stock_flow.eval()
        for parameter in self.stock_flow.parameters():
            parameter.requires_grad = False

    @torch.no_grad()
    def stock_prediction(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        return self.stock_flow(x_t, t, condition)

    def adapt_from_stock(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        stock_velocity: torch.Tensor,
        visual_volume: torch.Tensor,
        metadata: torch.Tensor,
        *,
        scale: float = 1.0,
        physical_present: bool = True,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        delta, stats = self.adapter(
            x_t,
            stock_velocity,
            t,
            visual_volume,
            metadata,
            scale=scale,
            physical_present=physical_present,
        )
        return stock_velocity + delta, stats
