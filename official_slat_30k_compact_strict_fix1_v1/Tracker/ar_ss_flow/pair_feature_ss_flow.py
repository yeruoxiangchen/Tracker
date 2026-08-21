from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from ar_ss_flow.correspondence_lifting import (
    CORRESPONDENCE_METADATA_NAMES,
    PAIR_FEATURE_EXPORT_VERSION,
)
from ar_ss_flow.pose_lifting import SUPPORT_METADATA_INDEX


PAIR_FEATURE_SS_FLOW_VERSION = "ar_ss_flow.pair_feature_local_velocity.v1"


class LocalPairFeatureVelocityAdapter(nn.Module):
    """State-conditioned attention over view pairs at the same 16^3 voxel."""

    def __init__(
        self,
        *,
        visual_channels: int,
        pair_feature_dim: int,
        latent_channels: int = 8,
        hidden_dim: int = 96,
        metadata_channels: int = len(CORRESPONDENCE_METADATA_NAMES),
        residual_t_min: float = 0.5,
        residual_t_ramp: float = 0.1,
    ) -> None:
        super().__init__()
        self.visual_channels = int(visual_channels)
        self.pair_feature_dim = int(pair_feature_dim)
        self.latent_channels = int(latent_channels)
        self.hidden_dim = int(hidden_dim)
        self.metadata_channels = int(metadata_channels)
        self.residual_t_min = float(residual_t_min)
        self.residual_t_ramp = float(residual_t_ramp)
        if self.pair_feature_dim <= 0 or self.hidden_dim <= 0:
            raise ValueError("pair_feature_dim and hidden_dim must be positive")
        if not 0.0 <= self.residual_t_min < 1.0:
            raise ValueError("residual_t_min must lie in [0,1)")
        if self.residual_t_ramp < 0.0:
            raise ValueError("residual_t_ramp must be non-negative")

        self.state_projection = nn.Conv3d(2 * latent_channels + 1, hidden_dim, 1)
        self.visual_projection = nn.Conv3d(visual_channels, hidden_dim, 1, bias=False)
        self.metadata_projection = nn.Conv3d(metadata_channels, hidden_dim, 1)
        self.pair_norm = nn.LayerNorm(self.pair_feature_dim)
        self.pair_key = nn.Linear(self.pair_feature_dim, hidden_dim, bias=False)
        self.pair_value = nn.Linear(self.pair_feature_dim, hidden_dim, bias=False)
        self.pair_query = nn.Conv3d(hidden_dim, hidden_dim, 1, bias=False)
        self.fusion = nn.Sequential(
            nn.Conv3d(8 * hidden_dim, 2 * hidden_dim, 1),
            nn.SiLU(),
            nn.Conv3d(2 * hidden_dim, hidden_dim, 1),
            nn.SiLU(),
        )
        self.output = nn.Conv3d(hidden_dim, latent_channels, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def metadata(self) -> dict[str, Any]:
        return {
            "version": PAIR_FEATURE_SS_FLOW_VERSION,
            "pair_feature_export_version": PAIR_FEATURE_EXPORT_VERSION,
            "fusion": "same_voxel_state_query_to_view_pair_attention",
            "global_attention": False,
            "spatial_neighborhood": 1,
            "visual_channels": self.visual_channels,
            "pair_feature_dim": self.pair_feature_dim,
            "metadata_channels": self.metadata_channels,
            "metadata_names": list(CORRESPONDENCE_METADATA_NAMES),
            "hidden_dim": self.hidden_dim,
            "latent_channels": self.latent_channels,
            "zero_init_output": True,
            "support_gate": "weighted_support_times_any_valid_pair",
            "time_normalization": "t_div_1000",
            "residual_t_min": self.residual_t_min,
            "residual_t_ramp": self.residual_t_ramp,
        }

    def _time_gate(self, t: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        fraction = (t.float() / 1000.0).clamp(0.0, 1.0)
        if self.residual_t_ramp > 0.0:
            gate = ((fraction - self.residual_t_min) / self.residual_t_ramp).clamp(
                0.0, 1.0
            )
        else:
            gate = fraction.ge(self.residual_t_min).float()
        return gate.to(dtype=dtype).reshape(-1, 1, 1, 1, 1)

    def _attend_pairs(
        self,
        state_hidden: torch.Tensor,
        pair_features: torch.Tensor,
        pair_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, pair_count, feature_dim = map(int, pair_features.shape[:3])
        spatial = tuple(int(value) for value in pair_features.shape[-3:])
        if pair_count == 0:
            zero = state_hidden.new_zeros((batch, self.hidden_dim, *spatial))
            return zero, zero.new_tensor(0.0), zero.new_tensor(0.0)

        tokens = pair_features.permute(0, 1, 3, 4, 5, 2).float()
        tokens = self.pair_norm(tokens)
        keys = self.pair_key(tokens)
        values = self.pair_value(tokens)
        query = self.pair_query(state_hidden).permute(0, 2, 3, 4, 1).float()
        scores = (keys * query[:, None]).sum(dim=-1) / math.sqrt(
            float(self.hidden_dim)
        )
        valid = pair_valid[:, :, 0].bool()
        scores = scores.masked_fill(~valid, -1.0e4)
        weights = torch.softmax(scores, dim=1) * valid.float()
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1.0e-6)
        attended = (values * weights[..., None]).sum(dim=1)
        attended = attended.permute(0, 4, 1, 2, 3).contiguous()
        entropy = -(weights.clamp_min(1.0e-8).log() * weights).sum(dim=1)
        normalizer = math.log(max(pair_count, 2))
        entropy = entropy / normalizer
        active = valid.any(dim=1)
        entropy_mean = (
            entropy[active].mean() if bool(active.any().item()) else entropy.new_tensor(0.0)
        )
        return attended, entropy_mean, valid.float().mean()

    def forward(
        self,
        x_t: torch.Tensor,
        stock_velocity: torch.Tensor,
        t: torch.Tensor,
        visual_volume: torch.Tensor,
        metadata: torch.Tensor,
        pair_features: torch.Tensor,
        pair_valid: torch.Tensor,
        *,
        scale: float = 1.0,
        physical_present: bool = True,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        expected_spatial = (16, 16, 16)
        if x_t.shape != stock_velocity.shape or tuple(x_t.shape[-3:]) != expected_spatial:
            raise ValueError("x_t/stock_velocity must be [B,8,16,16,16]")
        batch = int(x_t.shape[0])
        if visual_volume.shape != (batch, self.visual_channels, *expected_spatial):
            raise ValueError(f"invalid visual volume shape {tuple(visual_volume.shape)}")
        if metadata.shape != (batch, self.metadata_channels, *expected_spatial):
            raise ValueError(f"invalid metadata shape {tuple(metadata.shape)}")
        if pair_features.ndim != 6 or pair_valid.ndim != 6:
            raise ValueError("pair tensors must be [B,P,C,16,16,16]")
        if pair_features.shape[:2] != pair_valid.shape[:2]:
            raise ValueError("pair feature/mask batch and pair count must match")
        if int(pair_features.shape[2]) != self.pair_feature_dim:
            raise ValueError(
                f"pair feature dim {pair_features.shape[2]} != {self.pair_feature_dim}"
            )
        if int(pair_valid.shape[2]) != 1:
            raise ValueError("pair_valid must have one channel")
        if tuple(pair_features.shape[-3:]) != expected_spatial:
            raise ValueError("pair feature spatial shape must be 16^3")

        pair_available = pair_valid[:, :, 0].bool().any(dim=1, keepdim=True).float()
        support = metadata[:, SUPPORT_METADATA_INDEX : SUPPORT_METADATA_INDEX + 1].float()
        time_gate = self._time_gate(t, support.dtype)
        gate = support.clamp(0.0, 1.0) * pair_available * time_gate
        zero = torch.zeros_like(stock_velocity)
        if not physical_present or float(scale) == 0.0:
            return zero, {
                "delta_rms": zero.float().square().mean().sqrt(),
                "delta_abs_max": zero.float().abs().max(),
                "gate_mean": gate.mean(),
                "gate_ratio": gate.gt(0).float().mean(),
                "time_gate_mean": time_gate.mean(),
                "pair_valid_ratio": pair_valid.float().mean(),
                "pair_attention_entropy": zero.new_tensor(0.0),
                "attended_pair_rms": zero.new_tensor(0.0),
            }

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
        attended, attention_entropy, pair_valid_ratio = self._attend_pairs(
            state_hidden, pair_features, pair_valid
        )
        interaction = torch.cat(
            (
                state_hidden,
                visual_hidden,
                attended,
                state_hidden * visual_hidden,
                (state_hidden - visual_hidden).abs(),
                state_hidden * attended,
                (state_hidden - attended).abs(),
                metadata_hidden,
            ),
            dim=1,
        )
        delta = self.output(self.fusion(interaction)) * gate * float(scale)
        return delta.to(dtype=stock_velocity.dtype), {
            "delta_rms": delta.float().square().mean().sqrt(),
            "delta_abs_max": delta.float().abs().max(),
            "gate_mean": gate.mean(),
            "gate_ratio": gate.gt(0).float().mean(),
            "time_gate_mean": time_gate.mean(),
            "pair_valid_ratio": pair_valid_ratio,
            "pair_attention_entropy": attention_entropy,
            "attended_pair_rms": attended.float().square().mean().sqrt(),
        }


class PairFeatureSSFlowModel(nn.Module):
    def __init__(self, stock_flow: nn.Module, adapter: LocalPairFeatureVelocityAdapter) -> None:
        super().__init__()
        self.stock_flow = stock_flow
        self.adapter = adapter
        self.stock_flow.eval()
        for parameter in self.stock_flow.parameters():
            parameter.requires_grad = False

    @torch.no_grad()
    def stock_prediction(
        self, x_t: torch.Tensor, t: torch.Tensor, condition: torch.Tensor
    ) -> torch.Tensor:
        return self.stock_flow(x_t, t, condition)

    def adapt_from_stock(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        stock_velocity: torch.Tensor,
        visual_volume: torch.Tensor,
        metadata: torch.Tensor,
        pair_features: torch.Tensor,
        pair_valid: torch.Tensor,
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
            pair_features,
            pair_valid,
            scale=scale,
            physical_present=physical_present,
        )
        return stock_velocity + delta, stats


class PositiveConditionRolloutFlow(nn.Module):
    """Apply the local residual only to the positive CFG condition branch."""

    def __init__(
        self,
        model: PairFeatureSSFlowModel,
        positive_condition: torch.Tensor,
        evidence: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        *,
        scale: float,
    ) -> None:
        super().__init__()
        self.model = model
        self.positive_condition = positive_condition
        self.visual_volume, self.metadata, self.pair_features, self.pair_valid = evidence
        self.scale = float(scale)
        self.positive_calls = 0
        self.negative_calls = 0

    def _is_positive(self, condition: torch.Tensor) -> bool:
        return (
            condition is self.positive_condition
            or (
                torch.is_tensor(condition)
                and condition.shape == self.positive_condition.shape
                and condition.data_ptr() == self.positive_condition.data_ptr()
            )
        )

    def forward(
        self, x_t: torch.Tensor, t: torch.Tensor, condition: torch.Tensor
    ) -> torch.Tensor:
        stock = self.model.stock_flow(x_t, t, condition)
        if not self._is_positive(condition):
            self.negative_calls += 1
            return stock
        self.positive_calls += 1
        prediction, _ = self.model.adapt_from_stock(
            x_t,
            t,
            stock,
            self.visual_volume,
            self.metadata,
            self.pair_features,
            self.pair_valid,
            scale=self.scale,
        )
        return prediction
