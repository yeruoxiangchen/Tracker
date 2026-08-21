from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Sequence

import torch
from torch import nn

from reconvggt_ar_adapter_a.pointpose_ss_condition import PHYSICAL_FEATURE_NAMES
from reconvggt_ar_adapter_a.pointpose_patch_features import (
    PROJECTED_PATCH_EVIDENCE_COUNT,
    PROJECTED_PATCH_FEATURE_NAMES,
    PROJECTED_PATCH_FEATURE_VERSION,
    make_null_projected_patch_features,
    projected_patch_feature_schema_hash,
)


STOCK_PRESERVING_BRIDGE_VERSION = "reconvggt.stock_preserving_pointpose_bridge.v1"
MULTISTAGE_PHYSICAL_BRIDGE_VERSION = "reconvggt.multistage_pointpose_bridge.v2"
CONTENT_VISUAL_PHYSICAL_BRIDGE_VERSION = (
    "reconvggt.content_visual_pointpose_bridge.v1"
)
POSE_GUIDED_PATCH_BRIDGE_VERSION = "reconvggt.pose_guided_patch_bridge.v1"
PHYSICAL_EVIDENCE_FEATURE_COUNT = PHYSICAL_FEATURE_NAMES.index("x")


def make_null_physical_grid(physical_grid: torch.Tensor) -> torch.Tensor:
    """Remove object evidence while preserving the canonical XYZ coordinate field."""

    if physical_grid.ndim != 5 or physical_grid.shape[1] != len(PHYSICAL_FEATURE_NAMES):
        raise ValueError(
            "physical_grid must be [B,14,D,H,W], "
            f"got {tuple(physical_grid.shape)}"
        )
    null_grid = torch.zeros_like(physical_grid)
    null_grid[:, PHYSICAL_EVIDENCE_FEATURE_COUNT:] = physical_grid[
        :, PHYSICAL_EVIDENCE_FEATURE_COUNT:
    ]
    return null_grid


def _feature_schema_hash() -> str:
    payload = "\n".join(PHYSICAL_FEATURE_NAMES).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class BridgePathOutput:
    cond_stock: torch.Tensor
    cond_fused: torch.Tensor
    alignment_logit: torch.Tensor
    prefix_tokens: torch.Tensor
    physical_tokens: torch.Tensor
    stats: dict[str, torch.Tensor | tuple[int, ...]]
    stage_stats: dict[str, dict[str, torch.Tensor]] | None = None
    stage_tensors: dict[str, dict[str, torch.Tensor]] | None = None


class ZeroCenteredPhysicalEncoder(nn.Module):
    """Encode a 16^3 physical field while removing the all-zero response."""

    def __init__(self, *, feature_dim: int, hidden_dim: int, cond_dim: int) -> None:
        super().__init__()
        groups = min(32, int(hidden_dim))
        while int(hidden_dim) % groups:
            groups -= 1
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.cond_dim = int(cond_dim)
        self.encoder = nn.Sequential(
            nn.Conv3d(self.feature_dim, self.hidden_dim, 3, padding=1),
            nn.GroupNorm(groups, self.hidden_dim),
            nn.SiLU(),
            nn.Conv3d(self.hidden_dim, self.hidden_dim, 3, stride=2, padding=1),
            nn.GroupNorm(groups, self.hidden_dim),
            nn.SiLU(),
            nn.Conv3d(self.hidden_dim, self.cond_dim, 1),
        )

    def forward(self, physical_grid: torch.Tensor) -> torch.Tensor:
        if physical_grid.ndim != 5:
            raise ValueError(
                f"physical_grid must be [B,F,16,16,16], got {tuple(physical_grid.shape)}"
            )
        if physical_grid.shape[1] != self.feature_dim:
            raise ValueError(
                f"physical feature mismatch: expected {self.feature_dim}, got {physical_grid.shape[1]}"
            )
        if tuple(physical_grid.shape[-3:]) != (16, 16, 16):
            raise ValueError(f"physical grid must be 16^3, got {tuple(physical_grid.shape[-3:])}")

        physical_grid = physical_grid.float()
        paired = torch.cat((physical_grid, torch.zeros_like(physical_grid)), dim=0)
        encoded = self.encoder(paired)
        encoded_grid, encoded_zero = encoded.chunk(2, dim=0)
        centered = encoded_grid - encoded_zero
        if tuple(centered.shape[-3:]) != (8, 8, 8):
            raise RuntimeError(f"physical encoder must produce 8^3, got {tuple(centered.shape)}")
        return centered.flatten(2).transpose(1, 2).contiguous()


class ZeroCenteredPhysicalGridEncoder16(nn.Module):
    """Keep the physical field at 16^3 for position-aligned bridge fusion."""

    def __init__(self, *, feature_dim: int, hidden_dim: int, cond_dim: int) -> None:
        super().__init__()
        groups = min(32, int(hidden_dim))
        while int(hidden_dim) % groups:
            groups -= 1
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.cond_dim = int(cond_dim)
        self.encoder = nn.Sequential(
            nn.Conv3d(self.feature_dim, self.hidden_dim, 3, padding=1),
            nn.GroupNorm(groups, self.hidden_dim),
            nn.SiLU(),
            nn.Conv3d(self.hidden_dim, self.hidden_dim, 3, padding=1),
            nn.GroupNorm(groups, self.hidden_dim),
            nn.SiLU(),
            nn.Conv3d(self.hidden_dim, self.cond_dim, 1),
        )

    def forward(self, physical_grid: torch.Tensor) -> torch.Tensor:
        if physical_grid.ndim != 5:
            raise ValueError(
                f"physical_grid must be [B,F,16,16,16], got {tuple(physical_grid.shape)}"
            )
        if physical_grid.shape[1] != self.feature_dim:
            raise ValueError(
                f"physical feature mismatch: expected {self.feature_dim}, got {physical_grid.shape[1]}"
            )
        if tuple(physical_grid.shape[-3:]) != (16, 16, 16):
            raise ValueError(f"physical grid must be 16^3, got {tuple(physical_grid.shape[-3:])}")
        physical_grid = physical_grid.float()
        null_grid = make_null_physical_grid(physical_grid)
        paired = torch.cat((physical_grid, null_grid), dim=0)
        encoded = self.encoder(paired)
        encoded_grid, encoded_zero = encoded.chunk(2, dim=0)
        centered = encoded_grid - encoded_zero
        if tuple(centered.shape[-3:]) != (16, 16, 16):
            raise RuntimeError(f"physical encoder must preserve 16^3, got {tuple(centered.shape)}")
        return centered.flatten(2).transpose(1, 2).contiguous()


class LocalPhysicalFusionStage(nn.Module):
    """Position-aligned bridge/physical interaction with exact zero response."""

    def __init__(self, *, cond_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.cond_dim = int(cond_dim)
        self.hidden_dim = int(hidden_dim)
        self.bridge_norm = nn.LayerNorm(self.cond_dim)
        self.physical_norm = nn.LayerNorm(self.cond_dim, elementwise_affine=False)
        self.bridge_proj = nn.Linear(self.cond_dim, self.hidden_dim)
        self.physical_proj = nn.Linear(self.cond_dim, self.hidden_dim)
        self.interaction = nn.Sequential(
            nn.LayerNorm(self.hidden_dim * 3),
            nn.Linear(self.hidden_dim * 3, self.hidden_dim),
            nn.SiLU(),
        )
        self.output_proj = nn.Linear(self.hidden_dim, self.cond_dim)
        self.alignment_proj = nn.Linear(self.hidden_dim, 1)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)
        nn.init.zeros_(self.alignment_proj.weight)
        nn.init.zeros_(self.alignment_proj.bias)

    def _interaction_features(
        self,
        bridge_low: torch.Tensor,
        physical_tokens: torch.Tensor,
    ) -> torch.Tensor:
        physical_low = self.physical_proj(self.physical_norm(physical_tokens.float()))
        interaction = torch.cat(
            (
                bridge_low * physical_low,
                torch.abs(bridge_low - physical_low),
                physical_low,
            ),
            dim=-1,
        )
        return self.interaction(interaction)

    def forward(
        self,
        bridge_tokens: torch.Tensor,
        physical_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        if bridge_tokens.shape[:2] != physical_tokens.shape[:2]:
            raise ValueError(
                "local bridge/physical token mismatch: "
                f"bridge={tuple(bridge_tokens.shape)}, physical={tuple(physical_tokens.shape)}"
            )
        bridge_low = self.bridge_proj(self.bridge_norm(bridge_tokens.float()))
        response = self._interaction_features(bridge_low, physical_tokens)
        zero_response = self._interaction_features(bridge_low, torch.zeros_like(physical_tokens))

        # Subtraction happens after every biased layer in the residual path.
        delta = self.output_proj(response) - self.output_proj(zero_response)
        token_alignment = self.alignment_proj(response) - self.alignment_proj(zero_response)
        alignment_logit = token_alignment.mean(dim=(1, 2))
        rms_epsilon = delta.new_tensor(1.0e-12)
        delta_rms = (
            torch.sqrt(delta.float().square().mean() + rms_epsilon)
            - torch.sqrt(rms_epsilon)
        ).clamp_min(0.0)
        stats = {
            "delta_rms": delta_rms,
            "delta_abs_max": delta.float().abs().amax(),
            "alignment_logit_mean": alignment_logit.float().mean(),
        }
        return delta, alignment_logit, stats


class PhysicalBridgeAdapter(nn.Module):
    """Physical cross-attention injected immediately before a frozen bridge tail."""

    def __init__(
        self,
        *,
        feature_dim: int,
        cond_dim: int,
        hidden_dim: int,
        num_heads: int,
    ) -> None:
        super().__init__()
        if cond_dim % num_heads:
            raise ValueError(f"cond_dim={cond_dim} must be divisible by num_heads={num_heads}")
        self.feature_dim = int(feature_dim)
        self.cond_dim = int(cond_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.physical_encoder = ZeroCenteredPhysicalEncoder(
            feature_dim=self.feature_dim,
            hidden_dim=self.hidden_dim,
            cond_dim=self.cond_dim,
        )
        self.query_norm = nn.LayerNorm(self.cond_dim)
        self.physical_norm = nn.LayerNorm(self.cond_dim, elementwise_affine=False)
        self.cross_attn = nn.MultiheadAttention(
            self.cond_dim,
            self.num_heads,
            dropout=0.0,
            batch_first=True,
        )
        self.output_norm = nn.LayerNorm(self.cond_dim)
        self.output_proj = nn.Linear(self.cond_dim, self.cond_dim)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

        alignment_hidden = max(32, min(self.hidden_dim, self.cond_dim))
        self.alignment_head = nn.Sequential(
            nn.LayerNorm(self.cond_dim * 2),
            nn.Linear(self.cond_dim * 2, alignment_hidden),
            nn.SiLU(),
            nn.Linear(alignment_hidden, 1),
        )
        nn.init.zeros_(self.alignment_head[-1].weight)
        nn.init.zeros_(self.alignment_head[-1].bias)

    def encode_physical(self, physical_grid: torch.Tensor) -> torch.Tensor:
        return self.physical_encoder(physical_grid)

    def alignment_logits_from_tokens(
        self,
        bridge_tokens: torch.Tensor,
        physical_tokens: torch.Tensor,
    ) -> torch.Tensor:
        if bridge_tokens.shape[0] != physical_tokens.shape[0]:
            raise ValueError(
                "bridge/physical batch mismatch: "
                f"bridge={tuple(bridge_tokens.shape)}, physical={tuple(physical_tokens.shape)}"
            )
        pooled = torch.cat(
            (bridge_tokens.float().mean(dim=1), physical_tokens.float().mean(dim=1)),
            dim=-1,
        )
        return self.alignment_head(pooled).squeeze(-1)

    def alignment_logits(
        self,
        bridge_tokens: torch.Tensor,
        physical_grid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        physical_tokens = self.encode_physical(physical_grid)
        return self.alignment_logits_from_tokens(bridge_tokens, physical_tokens), physical_tokens

    def residual_from_tokens(
        self,
        bridge_tokens: torch.Tensor,
        physical_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query = self.query_norm(bridge_tokens.float())
        key_value = self.physical_norm(physical_tokens.float())
        attended, _ = self.cross_attn(query, key_value, key_value, need_weights=False)
        delta = self.output_proj(self.output_norm(attended))
        return delta, attended

    def metadata(self) -> dict[str, Any]:
        return {
            "version": STOCK_PRESERVING_BRIDGE_VERSION,
            "feature_dim": self.feature_dim,
            "feature_names": list(PHYSICAL_FEATURE_NAMES),
            "feature_schema_hash": _feature_schema_hash(),
            "cond_dim": self.cond_dim,
            "hidden_dim": self.hidden_dim,
            "num_heads": self.num_heads,
            "zero_centered_encoder": True,
            "zero_init_output": True,
            "physical_grid": [16, 16, 16],
            "physical_token_grid": [8, 8, 8],
        }


class StockPreservingPhysicalBridge(nn.Module):
    """Frozen ReconViaGen bridge with a gated adapter before its final blocks.

    The stock bridge is never modified. `physical_present=False` calls the
    original bridge directly and does not evaluate the physical branch.
    """

    def __init__(
        self,
        bridge: nn.Module,
        *,
        bridge_last_blocks: int = 1,
        feature_dim: int = len(PHYSICAL_FEATURE_NAMES),
        cond_dim: int = 1024,
        hidden_dim: int = 256,
        num_heads: int = 8,
    ) -> None:
        super().__init__()
        blocks = list(getattr(bridge, "cond_blocks", []))
        if not blocks:
            raise ValueError("stock bridge has no cond_blocks")
        if bridge_last_blocks < 1 or bridge_last_blocks > len(blocks):
            raise ValueError(
                f"bridge_last_blocks must be in [1,{len(blocks)}], got {bridge_last_blocks}"
            )
        for parameter in bridge.parameters():
            parameter.requires_grad = False
        bridge.eval()
        self.bridge = bridge
        self.bridge_last_blocks = int(bridge_last_blocks)
        self.fusion_start = len(blocks) - self.bridge_last_blocks
        self.adapter = PhysicalBridgeAdapter(
            feature_dim=int(feature_dim),
            cond_dim=int(cond_dim),
            hidden_dim=int(hidden_dim),
            num_heads=int(num_heads),
        )

    def train(self, mode: bool = True):
        super().train(mode)
        self.bridge.eval()
        return self

    def _contexts(
        self,
        aggregated_tokens_list: Sequence[torch.Tensor],
        image_cond: torch.Tensor,
    ) -> list[torch.Tensor]:
        layer_indices = list(getattr(self.bridge, "intermediate_layer_idx", [4, 11, 17, 23]))
        blocks = list(self.bridge.cond_blocks)
        if len(layer_indices) != len(blocks):
            raise RuntimeError(
                f"bridge layer/block mismatch: layers={layer_indices}, blocks={len(blocks)}"
            )
        batch = int(aggregated_tokens_list[0].shape[0])
        contexts: list[torch.Tensor] = []
        for layer_index in layer_indices:
            token = aggregated_tokens_list[layer_index][:, :, 5:]
            context = torch.cat(
                (token.reshape(batch, -1, 2048), image_cond.reshape(batch, -1, 1024)),
                dim=-1,
            ).to(self.bridge.dtype)
            contexts.append(context)
        return contexts

    def _prefix(
        self,
        aggregated_tokens_list: Sequence[torch.Tensor],
        image_cond: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        contexts = self._contexts(aggregated_tokens_list, image_cond)
        batch = int(aggregated_tokens_list[0].shape[0])
        hidden = self.bridge.multiview_cond_tokens.repeat(batch, 1, 1)
        with torch.no_grad():
            for index in range(self.fusion_start):
                hidden = self.bridge.cond_blocks[index](hidden, contexts[index])
        return hidden.detach(), contexts

    def _tail(
        self,
        hidden: torch.Tensor,
        contexts: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        for index in range(self.fusion_start, len(self.bridge.cond_blocks)):
            hidden = self.bridge.cond_blocks[index](hidden, contexts[index])
        return hidden

    @torch.no_grad()
    def stock_condition(
        self,
        aggregated_tokens_list: Sequence[torch.Tensor],
        image_cond: torch.Tensor,
    ) -> torch.Tensor:
        return self.bridge(aggregated_tokens_list, image_cond)

    def condition_paths(
        self,
        aggregated_tokens_list: Sequence[torch.Tensor],
        image_cond: torch.Tensor,
        physical_grid: torch.Tensor,
        *,
        physical_scale: float = 1.0,
        alignment_gate_override: float | torch.Tensor | None = None,
    ) -> BridgePathOutput:
        hidden, contexts = self._prefix(aggregated_tokens_list, image_cond)
        with torch.no_grad():
            cond_stock = self._tail(hidden, contexts).detach()

        alignment_logit, physical_tokens = self.adapter.alignment_logits(hidden, physical_grid)
        learned_gate = torch.sigmoid(alignment_logit)
        if alignment_gate_override is None:
            effective_gate = learned_gate
        else:
            effective_gate = torch.as_tensor(
                alignment_gate_override,
                device=hidden.device,
                dtype=torch.float32,
            )
            if effective_gate.ndim == 0:
                effective_gate = effective_gate.expand(hidden.shape[0])
            if effective_gate.shape != learned_gate.shape:
                raise ValueError(
                    f"alignment gate shape mismatch: expected {tuple(learned_gate.shape)}, "
                    f"got {tuple(effective_gate.shape)}"
                )
        delta, attended = self.adapter.residual_from_tokens(hidden, physical_tokens)
        scale = torch.as_tensor(float(physical_scale), device=delta.device, dtype=delta.dtype)
        gated_delta = delta * effective_gate[:, None, None].to(delta.dtype) * scale
        fused_hidden = hidden + gated_delta.to(dtype=hidden.dtype)
        cond_fused = self._tail(fused_hidden, contexts)
        cond_stock_float = cond_stock.float()
        cond_fused_float = cond_fused.float()
        applied_delta = cond_fused_float - cond_stock_float
        stock_rms = torch.sqrt(cond_stock_float.square().mean().clamp_min(1.0e-12))
        rms_epsilon = applied_delta.new_tensor(1.0e-12)
        applied_rms = (
            torch.sqrt(applied_delta.square().mean() + rms_epsilon)
            - torch.sqrt(rms_epsilon)
        ).clamp_min(0.0)
        adapter_delta_rms = (
            torch.sqrt(delta.float().square().mean() + rms_epsilon)
            - torch.sqrt(rms_epsilon)
        ).clamp_min(0.0)
        stats: dict[str, torch.Tensor | tuple[int, ...]] = {
            "adapter_delta_rms": adapter_delta_rms,
            "adapter_delta_abs_max": delta.float().abs().amax(),
            "condition_delta_rms": applied_rms,
            "condition_delta_abs_max": applied_delta.abs().amax(),
            "condition_delta_to_stock_ratio": applied_rms / stock_rms,
            "stock_condition_rms": stock_rms,
            "physical_token_rms": torch.sqrt(
                physical_tokens.float().square().mean().clamp_min(1.0e-12)
            ),
            "attended_rms": torch.sqrt(attended.float().square().mean().clamp_min(1.0e-12)),
            "alignment_probability_mean": learned_gate.mean(),
            "effective_gate_mean": effective_gate.float().mean(),
            "physical_token_shape": tuple(int(value) for value in physical_tokens.shape),
        }
        return BridgePathOutput(
            cond_stock=cond_stock,
            cond_fused=cond_fused.to(dtype=cond_stock.dtype),
            alignment_logit=alignment_logit,
            prefix_tokens=hidden,
            physical_tokens=physical_tokens,
            stats=stats,
        )

    def condition(
        self,
        aggregated_tokens_list: Sequence[torch.Tensor],
        image_cond: torch.Tensor,
        physical_grid: torch.Tensor | None,
        *,
        physical_present: bool,
        physical_scale: float = 1.0,
    ) -> torch.Tensor:
        if not physical_present:
            return self.stock_condition(aggregated_tokens_list, image_cond)
        if physical_grid is None:
            raise ValueError("physical_grid is required when physical_present=True")
        return self.condition_paths(
            aggregated_tokens_list,
            image_cond,
            physical_grid,
            physical_scale=float(physical_scale),
        ).cond_fused

    def metadata(self) -> dict[str, Any]:
        return {
            "version": STOCK_PRESERVING_BRIDGE_VERSION,
            "bridge_type": type(self.bridge).__name__,
            "bridge_block_count": len(self.bridge.cond_blocks),
            "bridge_last_blocks": self.bridge_last_blocks,
            "fusion_start_block": self.fusion_start,
            "stock_bridge_frozen": True,
            "hard_stock_route": True,
            "adapter": self.adapter.metadata(),
        }


class MultiStageStockPreservingPhysicalBridge(nn.Module):
    """Fuse a 16^3 physical field after selected frozen bridge blocks."""

    paired_training = True

    def __init__(
        self,
        bridge: nn.Module,
        *,
        fusion_stages: Sequence[int] = (0, 1, 2, 3),
        feature_dim: int = len(PHYSICAL_FEATURE_NAMES),
        cond_dim: int = 1024,
        physical_hidden_dim: int = 256,
        local_hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        blocks = list(getattr(bridge, "cond_blocks", []))
        if not blocks:
            raise ValueError("stock bridge has no cond_blocks")
        stages = tuple(sorted({int(index) for index in fusion_stages}))
        if not stages or stages[0] < 0 or stages[-1] >= len(blocks):
            raise ValueError(
                f"fusion stages must be a non-empty subset of [0,{len(blocks) - 1}], got {stages}"
            )
        token_count = int(getattr(bridge, "multiview_cond_tokens").shape[1])
        if token_count != 16**3:
            raise ValueError(
                "multistage local fusion requires 4096=16^3 bridge tokens, "
                f"got {token_count}"
            )
        for parameter in bridge.parameters():
            parameter.requires_grad = False
        bridge.eval()
        self.bridge = bridge
        self.fusion_stages = stages
        self.cond_dim = int(cond_dim)
        self.physical_encoder = ZeroCenteredPhysicalGridEncoder16(
            feature_dim=int(feature_dim),
            hidden_dim=int(physical_hidden_dim),
            cond_dim=self.cond_dim,
        )
        self.stage_adapters = nn.ModuleDict(
            {
                str(index): LocalPhysicalFusionStage(
                    cond_dim=self.cond_dim,
                    hidden_dim=int(local_hidden_dim),
                )
                for index in self.fusion_stages
            }
        )

    def train(self, mode: bool = True):
        super().train(mode)
        self.bridge.eval()
        return self

    def encode_physical(self, physical_grid: torch.Tensor) -> torch.Tensor:
        tokens = self.physical_encoder(physical_grid)
        expected = int(self.bridge.multiview_cond_tokens.shape[1])
        if tokens.shape[1] != expected:
            raise RuntimeError(
                f"physical/bridge token count mismatch: physical={tokens.shape[1]}, bridge={expected}"
            )
        return tokens

    def _contexts(
        self,
        aggregated_tokens_list: Sequence[torch.Tensor],
        image_cond: torch.Tensor,
    ) -> list[torch.Tensor]:
        layer_indices = list(getattr(self.bridge, "intermediate_layer_idx", [4, 11, 17, 23]))
        blocks = list(self.bridge.cond_blocks)
        if len(layer_indices) != len(blocks):
            raise RuntimeError(
                f"bridge layer/block mismatch: layers={layer_indices}, blocks={len(blocks)}"
            )
        batch = int(aggregated_tokens_list[0].shape[0])
        contexts: list[torch.Tensor] = []
        for layer_index in layer_indices:
            token = aggregated_tokens_list[layer_index][:, :, 5:]
            context = torch.cat(
                (token.reshape(batch, -1, 2048), image_cond.reshape(batch, -1, 1024)),
                dim=-1,
            ).to(self.bridge.dtype)
            contexts.append(context)
        return contexts

    @torch.no_grad()
    def stock_condition(
        self,
        aggregated_tokens_list: Sequence[torch.Tensor],
        image_cond: torch.Tensor,
    ) -> torch.Tensor:
        return self.bridge(aggregated_tokens_list, image_cond)

    def condition_paths(
        self,
        aggregated_tokens_list: Sequence[torch.Tensor],
        image_cond: torch.Tensor,
        physical_grid: torch.Tensor,
        *,
        physical_scale: float = 1.0,
        alignment_gate_override: float | torch.Tensor | None = None,
        cond_stock: torch.Tensor | None = None,
    ) -> BridgePathOutput:
        contexts = self._contexts(aggregated_tokens_list, image_cond)
        batch = int(aggregated_tokens_list[0].shape[0])
        hidden = self.bridge.multiview_cond_tokens.repeat(batch, 1, 1).detach()
        physical_tokens = self.encode_physical(physical_grid)
        stage_logits: list[torch.Tensor] = []
        stage_delta_rms: list[torch.Tensor] = []
        stage_delta_abs_max: list[torch.Tensor] = []
        per_stage_stats: dict[str, dict[str, torch.Tensor]] = {}
        scale = torch.as_tensor(
            float(physical_scale),
            device=physical_tokens.device,
            dtype=physical_tokens.dtype,
        )

        for index, block in enumerate(self.bridge.cond_blocks):
            hidden = block(hidden, contexts[index])
            if index not in self.fusion_stages:
                continue
            delta, stage_logit, stage_stats = self.stage_adapters[str(index)](
                hidden,
                physical_tokens,
            )
            if alignment_gate_override is None:
                effective_gate = torch.sigmoid(stage_logit)
            else:
                effective_gate = torch.as_tensor(
                    alignment_gate_override,
                    device=hidden.device,
                    dtype=torch.float32,
                )
                if effective_gate.ndim == 0:
                    effective_gate = effective_gate.expand(batch)
                if effective_gate.shape != stage_logit.shape:
                    raise ValueError(
                        f"alignment gate shape mismatch: expected {tuple(stage_logit.shape)}, "
                        f"got {tuple(effective_gate.shape)}"
                    )
            effective_delta = (
                delta
                * effective_gate[:, None, None].to(delta.dtype)
                * scale
            )
            hidden_rms = torch.sqrt(hidden.float().square().mean().clamp_min(1.0e-12))
            rms_epsilon = effective_delta.new_tensor(1.0e-12)
            effective_delta_rms = (
                torch.sqrt(effective_delta.float().square().mean() + rms_epsilon)
                - torch.sqrt(rms_epsilon)
            ).clamp_min(0.0)
            hidden = hidden + effective_delta.to(dtype=hidden.dtype)
            stage_logits.append(stage_logit)
            stage_delta_rms.append(stage_stats["delta_rms"])
            stage_delta_abs_max.append(stage_stats["delta_abs_max"])
            per_stage_stats[f"stage_{index}"] = {
                "alignment_logit_mean": stage_logit.float().mean(),
                "alignment_probability_mean": torch.sigmoid(stage_logit.float()).mean(),
                "effective_gate_mean": effective_gate.float().mean(),
                "delta_rms": stage_stats["delta_rms"],
                "delta_abs_max": stage_stats["delta_abs_max"],
                "effective_delta_rms": effective_delta_rms,
                "hidden_rms": hidden_rms,
                "effective_delta_to_hidden_ratio": effective_delta_rms / hidden_rms,
            }

        if not stage_logits:
            raise RuntimeError("multistage fusion produced no alignment logits")
        alignment_logit = torch.stack(stage_logits, dim=0).mean(dim=0)
        if cond_stock is None:
            cond_stock = self.stock_condition(aggregated_tokens_list, image_cond)
        cond_stock = cond_stock.detach()
        cond_fused = hidden.to(dtype=cond_stock.dtype)
        applied_delta = cond_fused.float() - cond_stock.float()
        stock_rms = torch.sqrt(cond_stock.float().square().mean().clamp_min(1.0e-12))
        rms_epsilon = applied_delta.new_tensor(1.0e-12)
        applied_rms = (
            torch.sqrt(applied_delta.square().mean() + rms_epsilon)
            - torch.sqrt(rms_epsilon)
        ).clamp_min(0.0)
        stats: dict[str, torch.Tensor | tuple[int, ...]] = {
            "adapter_delta_rms": torch.stack(stage_delta_rms).mean(),
            "adapter_delta_abs_max": torch.stack(stage_delta_abs_max).amax(),
            "condition_delta_rms": applied_rms,
            "condition_delta_abs_max": applied_delta.abs().amax(),
            "condition_delta_to_stock_ratio": applied_rms / stock_rms,
            "stock_condition_rms": stock_rms,
            "physical_token_rms": torch.sqrt(
                physical_tokens.float().square().mean().clamp_min(1.0e-12)
            ),
            "alignment_probability_mean": torch.sigmoid(alignment_logit).mean(),
            "effective_gate_mean": torch.stack(
                [torch.sigmoid(logit).mean() for logit in stage_logits]
            ).mean(),
            "physical_token_shape": tuple(int(value) for value in physical_tokens.shape),
        }
        return BridgePathOutput(
            cond_stock=cond_stock,
            cond_fused=cond_fused,
            alignment_logit=alignment_logit,
            prefix_tokens=hidden,
            physical_tokens=physical_tokens,
            stats=stats,
            stage_stats=per_stage_stats,
        )

    def condition(
        self,
        aggregated_tokens_list: Sequence[torch.Tensor],
        image_cond: torch.Tensor,
        physical_grid: torch.Tensor | None,
        *,
        physical_present: bool,
        physical_scale: float = 1.0,
    ) -> torch.Tensor:
        if not physical_present:
            return self.stock_condition(aggregated_tokens_list, image_cond)
        if physical_grid is None:
            raise ValueError("physical_grid is required when physical_present=True")
        return self.condition_paths(
            aggregated_tokens_list,
            image_cond,
            physical_grid,
            physical_scale=float(physical_scale),
        ).cond_fused

    def metadata(self) -> dict[str, Any]:
        first_stage = self.stage_adapters[str(self.fusion_stages[0])]
        return {
            "version": MULTISTAGE_PHYSICAL_BRIDGE_VERSION,
            "bridge_type": type(self.bridge).__name__,
            "bridge_block_count": len(self.bridge.cond_blocks),
            "fusion_stages": list(self.fusion_stages),
            "fusion_position": "after_each_selected_visual_bridge_block",
            "stock_bridge_frozen": True,
            "hard_stock_route": True,
            "residual_level_zero_centering": True,
            "null_physical_semantics": "channels_0_10_zero_xyz_11_13_preserved",
            "bridge_token_grid": [16, 16, 16],
            "physical_token_grid": [16, 16, 16],
            "local_index_aligned_fusion": True,
            "bridge_spatial_semantics": "16^3 index-alignment hypothesis under evaluation",
            "feature_dim": self.physical_encoder.feature_dim,
            "feature_names": list(PHYSICAL_FEATURE_NAMES),
            "feature_schema_hash": _feature_schema_hash(),
            "cond_dim": self.cond_dim,
            "physical_hidden_dim": self.physical_encoder.hidden_dim,
            "local_hidden_dim": first_stage.hidden_dim,
        }


class ZeroCenteredPhysicalTokenEncoder8(nn.Module):
    """Encode object evidence as 8^3 tokens relative to the XYZ-preserving null."""

    def __init__(
        self,
        *,
        feature_dim: int,
        hidden_dim: int,
        token_dim: int,
    ) -> None:
        super().__init__()
        groups = min(32, int(hidden_dim))
        while int(hidden_dim) % groups:
            groups -= 1
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.token_dim = int(token_dim)
        self.encoder = nn.Sequential(
            nn.Conv3d(self.feature_dim, self.hidden_dim, 3, padding=1),
            nn.GroupNorm(groups, self.hidden_dim),
            nn.SiLU(),
            nn.Conv3d(self.hidden_dim, self.hidden_dim, 3, stride=2, padding=1),
            nn.GroupNorm(groups, self.hidden_dim),
            nn.SiLU(),
            nn.Conv3d(self.hidden_dim, self.token_dim, 1),
        )

    def forward(self, physical_grid: torch.Tensor) -> torch.Tensor:
        if physical_grid.ndim != 5:
            raise ValueError(
                "physical_grid must be [B,F,16,16,16], "
                f"got {tuple(physical_grid.shape)}"
            )
        if int(physical_grid.shape[1]) != self.feature_dim:
            raise ValueError(
                f"physical feature mismatch: expected {self.feature_dim}, "
                f"got {physical_grid.shape[1]}"
            )
        if tuple(physical_grid.shape[-3:]) != (16, 16, 16):
            raise ValueError(
                f"physical grid must be 16^3, got {tuple(physical_grid.shape[-3:])}"
            )

        physical_grid = physical_grid.float()
        null_grid = make_null_physical_grid(physical_grid)
        encoded = self.encoder(torch.cat((physical_grid, null_grid), dim=0))
        encoded_physical, encoded_null = encoded.chunk(2, dim=0)
        centered = encoded_physical - encoded_null
        if tuple(centered.shape[-3:]) != (8, 8, 8):
            raise RuntimeError(
                f"content physical encoder must produce 8^3, got {tuple(centered.shape)}"
            )
        return centered.flatten(2).transpose(1, 2).contiguous()


class ContentPhysicalVisualFusionStage(nn.Module):
    """Let visual patch context query global physical tokens before a bridge block."""

    def __init__(
        self,
        *,
        visual_dim: int,
        fusion_dim: int,
        num_heads: int,
    ) -> None:
        super().__init__()
        if int(fusion_dim) % int(num_heads):
            raise ValueError(
                f"fusion_dim={fusion_dim} must be divisible by num_heads={num_heads}"
            )
        self.visual_dim = int(visual_dim)
        self.fusion_dim = int(fusion_dim)
        self.num_heads = int(num_heads)
        self.visual_norm = nn.LayerNorm(self.visual_dim, elementwise_affine=False)
        self.query_proj = nn.Linear(self.visual_dim, self.fusion_dim)
        self.physical_norm = nn.LayerNorm(self.fusion_dim, elementwise_affine=False)
        self.cross_attn = nn.MultiheadAttention(
            self.fusion_dim,
            self.num_heads,
            dropout=0.0,
            batch_first=True,
        )
        self.output_norm = nn.LayerNorm(self.fusion_dim)
        self.output_proj = nn.Linear(self.fusion_dim, self.visual_dim)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

        gate_hidden = max(32, self.fusion_dim)
        self.gate_head = nn.Sequential(
            nn.LayerNorm(self.fusion_dim * 4),
            nn.Linear(self.fusion_dim * 4, gate_hidden),
            nn.SiLU(),
            nn.Linear(gate_hidden, 1),
        )

    def _gate_features(
        self,
        query: torch.Tensor,
        response: torch.Tensor,
    ) -> torch.Tensor:
        return torch.cat(
            (
                query,
                response,
                query * response,
                torch.abs(query - response),
            ),
            dim=-1,
        )

    def forward(
        self,
        visual_context: torch.Tensor,
        physical_tokens: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
    ]:
        if visual_context.ndim != 3 or visual_context.shape[-1] != self.visual_dim:
            raise ValueError(
                f"visual_context must be [B,L,{self.visual_dim}], "
                f"got {tuple(visual_context.shape)}"
            )
        if physical_tokens.ndim != 3 or physical_tokens.shape[-1] != self.fusion_dim:
            raise ValueError(
                f"physical_tokens must be [B,N,{self.fusion_dim}], "
                f"got {tuple(physical_tokens.shape)}"
            )
        if visual_context.shape[0] != physical_tokens.shape[0]:
            raise ValueError(
                "visual/physical batch mismatch: "
                f"visual={tuple(visual_context.shape)}, "
                f"physical={tuple(physical_tokens.shape)}"
            )

        query = self.query_proj(self.visual_norm(visual_context.float()))
        key_value = self.physical_norm(physical_tokens.float())
        paired_query = torch.cat((query, query), dim=0)
        paired_key_value = torch.cat((key_value, torch.zeros_like(key_value)), dim=0)
        paired_attended, _ = self.cross_attn(
            paired_query,
            paired_key_value,
            paired_key_value,
            need_weights=False,
        )
        attended, attended_null = paired_attended.chunk(2, dim=0)
        centered_attended = attended - attended_null

        paired_projected = self.output_proj(self.output_norm(paired_attended))
        projected, projected_null = paired_projected.chunk(2, dim=0)
        context_delta = projected - projected_null

        gate_null_response = torch.zeros_like(centered_attended)
        token_gate_logit = self.gate_head(
            self._gate_features(query, centered_attended)
        ) - self.gate_head(self._gate_features(query, gate_null_response))
        sample_alignment_logit = token_gate_logit.mean(dim=(1, 2))

        epsilon = context_delta.new_tensor(1.0e-12)
        context_delta_rms = (
            torch.sqrt(context_delta.float().square().mean() + epsilon)
            - torch.sqrt(epsilon)
        ).clamp_min(0.0)
        attended_rms = (
            torch.sqrt(centered_attended.float().square().mean() + epsilon)
            - torch.sqrt(epsilon)
        ).clamp_min(0.0)
        stats = {
            "attended_centered_rms": attended_rms,
            "context_delta_rms": context_delta_rms,
            "context_delta_abs_max": context_delta.float().abs().amax(),
            "alignment_logit_mean": sample_alignment_logit.float().mean(),
            "token_gate_logit_std": token_gate_logit.float().std(unbiased=False),
        }
        diagnostic_tensors = {
            "attended_centered": centered_attended,
            "context_delta_raw": context_delta,
        }
        return context_delta, token_gate_logit, stats, diagnostic_tensors


class ContentBasedPhysicalVisualBridge(nn.Module):
    """Pre-block content fusion without assuming bridge/voxel index alignment."""

    paired_training = True
    content_visual_fusion = True

    def __init__(
        self,
        bridge: nn.Module,
        *,
        fusion_stages: Sequence[int] = (0, 1),
        feature_dim: int = len(PHYSICAL_FEATURE_NAMES),
        physical_hidden_dim: int = 128,
        visual_dim: int = 3072,
        fusion_dim: int = 128,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        blocks = list(getattr(bridge, "cond_blocks", []))
        if not blocks:
            raise ValueError("stock bridge has no cond_blocks")
        stages = tuple(sorted({int(index) for index in fusion_stages}))
        if not stages or stages[0] < 0 or stages[-1] >= len(blocks):
            raise ValueError(
                f"fusion stages must be a non-empty subset of [0,{len(blocks) - 1}], "
                f"got {stages}"
            )
        for parameter in bridge.parameters():
            parameter.requires_grad = False
        bridge.eval()
        self.bridge = bridge
        self.fusion_stages = stages
        self.visual_dim = int(visual_dim)
        self.fusion_dim = int(fusion_dim)
        self.physical_encoder = ZeroCenteredPhysicalTokenEncoder8(
            feature_dim=int(feature_dim),
            hidden_dim=int(physical_hidden_dim),
            token_dim=self.fusion_dim,
        )
        self.stage_adapters = nn.ModuleDict(
            {
                str(index): ContentPhysicalVisualFusionStage(
                    visual_dim=self.visual_dim,
                    fusion_dim=self.fusion_dim,
                    num_heads=int(num_heads),
                )
                for index in self.fusion_stages
            }
        )

    def train(self, mode: bool = True):
        super().train(mode)
        self.bridge.eval()
        return self

    def encode_physical(self, physical_grid: torch.Tensor) -> torch.Tensor:
        return self.physical_encoder(physical_grid)

    def _contexts(
        self,
        aggregated_tokens_list: Sequence[torch.Tensor],
        image_cond: torch.Tensor,
    ) -> list[torch.Tensor]:
        layer_indices = list(
            getattr(self.bridge, "intermediate_layer_idx", [4, 11, 17, 23])
        )
        blocks = list(self.bridge.cond_blocks)
        if len(layer_indices) != len(blocks):
            raise RuntimeError(
                f"bridge layer/block mismatch: layers={layer_indices}, blocks={len(blocks)}"
            )
        batch = int(aggregated_tokens_list[0].shape[0])
        contexts: list[torch.Tensor] = []
        for layer_index in layer_indices:
            token = aggregated_tokens_list[layer_index][:, :, 5:]
            context = torch.cat(
                (
                    token.reshape(batch, -1, 2048),
                    image_cond.reshape(batch, -1, 1024),
                ),
                dim=-1,
            ).to(self.bridge.dtype)
            if int(context.shape[-1]) != self.visual_dim:
                raise RuntimeError(
                    f"visual context dim mismatch: expected {self.visual_dim}, "
                    f"got {context.shape[-1]}"
                )
            contexts.append(context)
        return contexts

    @torch.no_grad()
    def stock_condition(
        self,
        aggregated_tokens_list: Sequence[torch.Tensor],
        image_cond: torch.Tensor,
    ) -> torch.Tensor:
        return self.bridge(aggregated_tokens_list, image_cond)

    @staticmethod
    def _resolve_token_gate(
        learned_gate: torch.Tensor,
        override: float | torch.Tensor | None,
    ) -> torch.Tensor:
        if override is None:
            return learned_gate
        gate = torch.as_tensor(
            override,
            device=learned_gate.device,
            dtype=torch.float32,
        )
        if gate.ndim == 0:
            gate = gate.expand_as(learned_gate)
        elif gate.ndim == 1 and gate.shape[0] == learned_gate.shape[0]:
            gate = gate[:, None, None].expand_as(learned_gate)
        elif gate.ndim == 2 and tuple(gate.shape) == tuple(learned_gate.shape[:2]):
            gate = gate[:, :, None]
        if tuple(gate.shape) != tuple(learned_gate.shape):
            raise ValueError(
                f"token gate shape mismatch: expected {tuple(learned_gate.shape)}, "
                f"got {tuple(gate.shape)}"
            )
        return gate

    def condition_paths(
        self,
        aggregated_tokens_list: Sequence[torch.Tensor],
        image_cond: torch.Tensor,
        physical_grid: torch.Tensor,
        *,
        physical_scale: float = 1.0,
        alignment_gate_override: float | torch.Tensor | None = None,
        cond_stock: torch.Tensor | None = None,
    ) -> BridgePathOutput:
        contexts = self._contexts(aggregated_tokens_list, image_cond)
        batch = int(aggregated_tokens_list[0].shape[0])
        hidden = self.bridge.multiview_cond_tokens.repeat(batch, 1, 1).detach()
        physical_tokens = self.encode_physical(physical_grid)
        scale = torch.as_tensor(
            float(physical_scale),
            device=physical_tokens.device,
            dtype=physical_tokens.dtype,
        )
        stage_logits: list[torch.Tensor] = []
        stage_delta_rms: list[torch.Tensor] = []
        stage_delta_abs_max: list[torch.Tensor] = []
        stage_delta_ratios: list[torch.Tensor] = []
        stage_gate_means: list[torch.Tensor] = []
        per_stage_stats: dict[str, dict[str, torch.Tensor]] = {}
        per_stage_tensors: dict[str, dict[str, torch.Tensor]] = {}

        for index, block in enumerate(self.bridge.cond_blocks):
            context = contexts[index]
            if index in self.fusion_stages:
                (
                    context_delta,
                    token_gate_logit,
                    raw_stats,
                    diagnostic_tensors,
                ) = self.stage_adapters[str(index)](context, physical_tokens)
                learned_gate = torch.sigmoid(token_gate_logit.float())
                effective_gate = self._resolve_token_gate(
                    learned_gate,
                    alignment_gate_override,
                )
                effective_delta = (
                    context_delta
                    * effective_gate.to(context_delta.dtype)
                    * scale
                )
                context_rms = torch.sqrt(
                    context.float().square().mean().clamp_min(1.0e-12)
                )
                epsilon = effective_delta.new_tensor(1.0e-12)
                effective_delta_rms = (
                    torch.sqrt(effective_delta.float().square().mean() + epsilon)
                    - torch.sqrt(epsilon)
                ).clamp_min(0.0)
                delta_ratio = effective_delta_rms / context_rms
                fused_context = context + effective_delta.to(dtype=context.dtype)
                sample_logit = token_gate_logit.mean(dim=(1, 2))
                stage_logits.append(sample_logit)
                stage_delta_rms.append(raw_stats["context_delta_rms"])
                stage_delta_abs_max.append(raw_stats["context_delta_abs_max"])
                stage_delta_ratios.append(delta_ratio)
                stage_gate_means.append(effective_gate.float().mean())
                stage_name = f"stage_{index}"
                per_stage_stats[stage_name] = {
                    "alignment_logit_mean": sample_logit.float().mean(),
                    "alignment_probability_mean": learned_gate.mean(),
                    "effective_gate_mean": effective_gate.float().mean(),
                    "attended_centered_rms": raw_stats["attended_centered_rms"],
                    "delta_rms": raw_stats["context_delta_rms"],
                    "delta_abs_max": raw_stats["context_delta_abs_max"],
                    "effective_delta_rms": effective_delta_rms,
                    "hidden_rms": context_rms,
                    "context_rms": context_rms,
                    "effective_delta_to_hidden_ratio": delta_ratio,
                    "effective_delta_to_context_ratio": delta_ratio,
                    "token_gate_logit_std": raw_stats["token_gate_logit_std"],
                }
                per_stage_tensors[stage_name] = {
                    "attended_centered": diagnostic_tensors[
                        "attended_centered"
                    ].detach(),
                    "context_delta_effective": effective_delta.detach(),
                }
            else:
                fused_context = context
            hidden = block(hidden, fused_context)

        if not stage_logits:
            raise RuntimeError("content visual fusion produced no stage logits")
        alignment_logit = torch.stack(stage_logits, dim=0).mean(dim=0)
        if cond_stock is None:
            cond_stock = self.stock_condition(aggregated_tokens_list, image_cond)
        cond_stock = cond_stock.detach()
        cond_fused = hidden.to(dtype=cond_stock.dtype)
        applied_delta = cond_fused.float() - cond_stock.float()
        stock_rms = torch.sqrt(cond_stock.float().square().mean().clamp_min(1.0e-12))
        epsilon = applied_delta.new_tensor(1.0e-12)
        applied_rms = (
            torch.sqrt(applied_delta.square().mean() + epsilon)
            - torch.sqrt(epsilon)
        ).clamp_min(0.0)
        stats: dict[str, torch.Tensor | tuple[int, ...]] = {
            "adapter_delta_rms": torch.stack(stage_delta_rms).mean(),
            "adapter_delta_abs_max": torch.stack(stage_delta_abs_max).amax(),
            "condition_delta_rms": applied_rms,
            "condition_delta_abs_max": applied_delta.abs().amax(),
            "condition_delta_to_stock_ratio": applied_rms / stock_rms,
            "context_delta_to_context_ratio": torch.stack(stage_delta_ratios).mean(),
            "stock_condition_rms": stock_rms,
            "physical_token_rms": torch.sqrt(
                physical_tokens.float().square().mean().clamp_min(1.0e-12)
            ),
            "attended_rms": torch.stack(
                [
                    values["attended_centered_rms"]
                    for values in per_stage_stats.values()
                ]
            ).mean(),
            "alignment_probability_mean": torch.sigmoid(alignment_logit).mean(),
            "effective_gate_mean": torch.stack(stage_gate_means).mean(),
            "physical_token_shape": tuple(int(value) for value in physical_tokens.shape),
        }
        return BridgePathOutput(
            cond_stock=cond_stock,
            cond_fused=cond_fused,
            alignment_logit=alignment_logit,
            prefix_tokens=hidden,
            physical_tokens=physical_tokens,
            stats=stats,
            stage_stats=per_stage_stats,
            stage_tensors=per_stage_tensors,
        )

    def condition(
        self,
        aggregated_tokens_list: Sequence[torch.Tensor],
        image_cond: torch.Tensor,
        physical_grid: torch.Tensor | None,
        *,
        physical_present: bool,
        physical_scale: float = 1.0,
    ) -> torch.Tensor:
        if not physical_present:
            return self.stock_condition(aggregated_tokens_list, image_cond)
        if physical_grid is None:
            raise ValueError("physical_grid is required when physical_present=True")
        return self.condition_paths(
            aggregated_tokens_list,
            image_cond,
            physical_grid,
            physical_scale=float(physical_scale),
        ).cond_fused

    def metadata(self) -> dict[str, Any]:
        first_stage = self.stage_adapters[str(self.fusion_stages[0])]
        return {
            "version": CONTENT_VISUAL_PHYSICAL_BRIDGE_VERSION,
            "bridge_type": type(self.bridge).__name__,
            "bridge_block_count": len(self.bridge.cond_blocks),
            "fusion_stages": list(self.fusion_stages),
            "fusion_position": "before_each_selected_visual_bridge_block",
            "fusion_query": "vggt_dino_visual_context",
            "fusion_key_value": "global_8x8x8_physical_tokens",
            "stock_bridge_frozen": True,
            "hard_stock_route": True,
            "residual_level_zero_centering": True,
            "null_physical_semantics": "channels_0_10_zero_xyz_11_13_preserved",
            "physical_grid": [16, 16, 16],
            "physical_token_grid": [8, 8, 8],
            "physical_token_count": 8**3,
            "visual_context_dim": self.visual_dim,
            "content_fusion_dim": self.fusion_dim,
            "content_fusion_heads": first_stage.num_heads,
            "token_level_gate": True,
            "output_projection_zero_init": True,
            "index_aligned_fusion": False,
            "bridge_token_spatial_mapping_required": False,
            "feature_dim": self.physical_encoder.feature_dim,
            "feature_names": list(PHYSICAL_FEATURE_NAMES),
            "feature_schema_hash": _feature_schema_hash(),
            "physical_hidden_dim": self.physical_encoder.hidden_dim,
        }


class ZeroCenteredProjectedPatchEncoder(nn.Module):
    """Encode view-patch Point/Pose evidence relative to a pose-preserving null."""

    def __init__(self, *, feature_dim: int, hidden_dim: int, token_dim: int) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.token_dim = int(token_dim)
        self.input_norm = nn.LayerNorm(self.feature_dim, elementwise_affine=False)
        self.encoder = nn.Sequential(
            nn.Linear(self.feature_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.token_dim),
        )

    def forward(self, patch_features: torch.Tensor) -> torch.Tensor:
        if patch_features.ndim != 3 or int(patch_features.shape[-1]) != self.feature_dim:
            raise ValueError(
                f"patch_features must be [B,L,{self.feature_dim}], "
                f"got {tuple(patch_features.shape)}"
            )
        patch_features = patch_features.float()
        null_features = make_null_projected_patch_features(patch_features)
        paired = torch.cat((patch_features, null_features), dim=0)
        encoded = self.encoder(self.input_norm(paired))
        encoded_physical, encoded_null = encoded.chunk(2, dim=0)
        return (encoded_physical - encoded_null).contiguous()


class ProjectedPatchVisualFusionStage(nn.Module):
    """Fuse each visual patch only with Point/Pose evidence projected to that patch."""

    def __init__(self, *, visual_dim: int, fusion_dim: int) -> None:
        super().__init__()
        self.visual_dim = int(visual_dim)
        self.fusion_dim = int(fusion_dim)
        self.num_heads = 0
        self.visual_norm = nn.LayerNorm(self.visual_dim, elementwise_affine=False)
        self.query_proj = nn.Linear(self.visual_dim, self.fusion_dim)
        self.physical_norm = nn.LayerNorm(self.fusion_dim, elementwise_affine=False)
        interaction_dim = self.fusion_dim * 3
        self.interaction = nn.Sequential(
            nn.LayerNorm(interaction_dim),
            nn.Linear(interaction_dim, self.fusion_dim),
            nn.SiLU(),
        )
        self.output_norm = nn.LayerNorm(self.fusion_dim)
        self.output_proj = nn.Linear(self.fusion_dim, self.visual_dim)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)
        self.gate_head = nn.Sequential(
            nn.LayerNorm(interaction_dim),
            nn.Linear(interaction_dim, max(32, self.fusion_dim)),
            nn.SiLU(),
            nn.Linear(max(32, self.fusion_dim), 1),
        )

    @staticmethod
    def _interaction_features(
        query: torch.Tensor,
        physical: torch.Tensor,
    ) -> torch.Tensor:
        return torch.cat(
            (query * physical, torch.abs(query - physical), physical), dim=-1
        )

    def forward(
        self,
        visual_context: torch.Tensor,
        physical_tokens: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
    ]:
        if visual_context.ndim != 3 or visual_context.shape[-1] != self.visual_dim:
            raise ValueError(
                f"visual_context must be [B,L,{self.visual_dim}], "
                f"got {tuple(visual_context.shape)}"
            )
        if physical_tokens.ndim != 3 or physical_tokens.shape[-1] != self.fusion_dim:
            raise ValueError(
                f"physical_tokens must be [B,L,{self.fusion_dim}], "
                f"got {tuple(physical_tokens.shape)}"
            )
        if visual_context.shape[:2] != physical_tokens.shape[:2]:
            raise ValueError(
                "projected patch/visual index mismatch: "
                f"visual={tuple(visual_context.shape)}, physical={tuple(physical_tokens.shape)}"
            )
        query = self.query_proj(self.visual_norm(visual_context.float()))
        physical = self.physical_norm(physical_tokens.float())
        response = self.interaction(self._interaction_features(query, physical))
        null_response = self.interaction(
            self._interaction_features(query, torch.zeros_like(physical))
        )
        centered_response = response - null_response
        context_delta = self.output_proj(self.output_norm(response)) - self.output_proj(
            self.output_norm(null_response)
        )
        token_gate_logit = self.gate_head(
            self._interaction_features(query, centered_response)
        ) - self.gate_head(
            self._interaction_features(query, torch.zeros_like(centered_response))
        )
        sample_alignment_logit = token_gate_logit.mean(dim=(1, 2))
        epsilon = context_delta.new_tensor(1.0e-12)
        context_delta_rms = (
            torch.sqrt(context_delta.float().square().mean() + epsilon)
            - torch.sqrt(epsilon)
        ).clamp_min(0.0)
        response_rms = (
            torch.sqrt(centered_response.float().square().mean() + epsilon)
            - torch.sqrt(epsilon)
        ).clamp_min(0.0)
        return (
            context_delta,
            token_gate_logit,
            {
                "attended_centered_rms": response_rms,
                "context_delta_rms": context_delta_rms,
                "context_delta_abs_max": context_delta.float().abs().amax(),
                "alignment_logit_mean": sample_alignment_logit.float().mean(),
                "token_gate_logit_std": token_gate_logit.float().std(unbiased=False),
            },
            {
                "attended_centered": centered_response,
                "context_delta_raw": context_delta,
            },
        )


class PoseGuidedProjectedPatchBridge(ContentBasedPhysicalVisualBridge):
    """Pre-block view-patch fusion with explicit K/T projection correspondence."""

    projected_patch_fusion = True

    def __init__(
        self,
        bridge: nn.Module,
        *,
        fusion_stages: Sequence[int] = (0, 1),
        feature_dim: int = len(PROJECTED_PATCH_FEATURE_NAMES),
        physical_hidden_dim: int = 128,
        visual_dim: int = 3072,
        fusion_dim: int = 128,
    ) -> None:
        super().__init__(
            bridge,
            fusion_stages=fusion_stages,
            feature_dim=len(PHYSICAL_FEATURE_NAMES),
            physical_hidden_dim=physical_hidden_dim,
            visual_dim=visual_dim,
            fusion_dim=fusion_dim,
            num_heads=1,
        )
        self.physical_encoder = ZeroCenteredProjectedPatchEncoder(
            feature_dim=int(feature_dim),
            hidden_dim=int(physical_hidden_dim),
            token_dim=int(fusion_dim),
        )
        self.stage_adapters = nn.ModuleDict(
            {
                str(index): ProjectedPatchVisualFusionStage(
                    visual_dim=int(visual_dim),
                    fusion_dim=int(fusion_dim),
                )
                for index in self.fusion_stages
            }
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "version": POSE_GUIDED_PATCH_BRIDGE_VERSION,
            "patch_feature_version": PROJECTED_PATCH_FEATURE_VERSION,
            "bridge_type": type(self.bridge).__name__,
            "bridge_block_count": len(self.bridge.cond_blocks),
            "fusion_stages": list(self.fusion_stages),
            "fusion_position": "before_each_selected_visual_bridge_block",
            "fusion_query": "view_major_vggt_dino_patch",
            "fusion_key_value": "same_view_same_patch_projected_pointpose",
            "stock_bridge_frozen": True,
            "hard_stock_route": True,
            "residual_level_zero_centering": True,
            "null_physical_semantics": (
                "point_mask_depth_evidence_zero_pose_ray_uv_preserved"
            ),
            "projected_patch_evidence_feature_count": int(
                PROJECTED_PATCH_EVIDENCE_COUNT
            ),
            "visual_context_dim": self.visual_dim,
            "content_fusion_dim": self.fusion_dim,
            "token_level_gate": True,
            "output_projection_zero_init": True,
            "view_patch_index_aligned_fusion": True,
            "bridge_token_spatial_mapping_required": False,
            "feature_dim": self.physical_encoder.feature_dim,
            "feature_names": list(PROJECTED_PATCH_FEATURE_NAMES),
            "feature_schema_hash": projected_patch_feature_schema_hash(),
            "physical_hidden_dim": self.physical_encoder.hidden_dim,
        }
