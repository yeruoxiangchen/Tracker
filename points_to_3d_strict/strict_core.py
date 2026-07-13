from __future__ import annotations

import json
import math
import os
import sys
import weakref
from pathlib import Path
from typing import Any

os.environ.setdefault("SPCONV_ALGO", "native")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("TORCH_EXTENSIONS_DIR", "/tmp/torch_extensions")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("ATTN_BACKEND", "flash_attn")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

TRACKER_ROOT = Path(__file__).resolve().parents[1]
RECONVIAGEN_ROOT = TRACKER_ROOT / "ReconViaGen"
VGGT_WHEEL_ROOT = RECONVIAGEN_ROOT / "wheels" / "vggt"
for path in (TRACKER_ROOT, RECONVIAGEN_ROOT, VGGT_WHEEL_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from trellis.pipelines.trellis_image_to_3d import TrellisImageTo3DPipeline  # noqa: E402

from trellis_point_prior_mv.common import (  # noqa: E402
    apply_grid_transform,
    coords_to_batched_occ,
    coords_to_points,
    load_manifest,
    load_mask,
    parse_indices,
    project_points_to_masks,
    resolve_path,
)
from trellis_point_prior_mv.eval_latent_splice_sanity import latent_mask_from_prior, normalize_latent_mask  # noqa: E402
from trellis_point_prior_mv.latent_inpaint_image_condition import SourceImageResolver, encode_image_condition  # noqa: E402
from trellis.modules.spatial import patchify  # noqa: E402


VIEW_GATED_AGGREGATIONS = {"view_gated", "gated", "adapter", "view_adapter"}
GEOMETRY_AWARE_AGGREGATIONS = {"geometry_view_gated", "geo_view_gated", "pose_view_gated"}
PROJECTION_GRID_AWARE_AGGREGATIONS = {"projection_view_gated", "projection_aware", "projection_geo_view_gated"}
PROJECTION_TOKEN_AWARE_AGGREGATIONS = {"projection_token_view_gated", "projection_token_aware", "projection_v2"}
PROJECTION_AWARE_AGGREGATIONS = PROJECTION_GRID_AWARE_AGGREGATIONS | PROJECTION_TOKEN_AWARE_AGGREGATIONS
ADAPTER_AGGREGATIONS = VIEW_GATED_AGGREGATIONS | GEOMETRY_AWARE_AGGREGATIONS | PROJECTION_AWARE_AGGREGATIONS
VIEW_POSE_FEATURE_DIM = 32
PROJECTION_FEATURE_CHANNELS = 3
PROJECTION_TOKEN_FEATURE_CHANNELS = 6


class LoRALinear(nn.Module):
    """A small LoRA adapter around an existing Linear layer.

    The wrapped base layer is kept frozen. LoRA weights are initialized so the
    initial output is exactly the original base output.
    """

    def __init__(self, base: nn.Linear, *, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError(f"LoRALinear expects nn.Linear, got {type(base)!r}")
        self.base = base
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = float(alpha) / float(max(self.rank, 1))
        for param in self.base.parameters():
            param.requires_grad = False
        self.lora_A = nn.Parameter(torch.empty(self.rank, int(base.in_features), dtype=torch.float32, device=base.weight.device))
        self.lora_B = nn.Parameter(torch.zeros(int(base.out_features), self.rank, dtype=torch.float32, device=base.weight.device))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        lora = F.linear(F.linear(x.float(), self.lora_A.float()), self.lora_B.float()) * self.scaling
        return out + lora.to(dtype=out.dtype)


class BlockwiseGlobalModulationAdapter(nn.Module):
    """Zero-init global geometry modulation for selected sparse-flow blocks.

    The adapter consumes pooled geometry context from q_vis, m_s and optional
    projection-grid channels. Each selected block receives a tiny global bias or
    FiLM update. The final heads are zero-initialized, so injecting the adapter
    is initially an exact no-op.
    """

    def __init__(
        self,
        *,
        input_dim: int,
        channels: int,
        num_blocks: int,
        hidden_dim: int = 256,
        mode: str = "bias",
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.channels = int(channels)
        self.num_blocks = int(num_blocks)
        self.hidden_dim = int(hidden_dim)
        self.mode = str(mode or "bias").lower()
        if self.mode not in {"bias", "film"}:
            raise ValueError(f"unsupported blockwise global modulation mode={mode!r}")
        out_dim = self.channels * (2 if self.mode == "film" else 1)
        self.encoder = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.SiLU(),
        )
        self.heads = nn.ModuleList([nn.Linear(self.hidden_dim, out_dim) for _ in range(self.num_blocks)])
        for head in self.heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        self._active_context: torch.Tensor | None = None

    def set_context(self, context: torch.Tensor | None) -> None:
        self._active_context = context

    def apply_to_hidden(self, slot: int, hidden: torch.Tensor) -> torch.Tensor:
        context = self._active_context
        if context is None:
            return hidden
        if context.ndim != 2 or context.shape[1] != self.input_dim:
            raise ValueError(
                f"blockwise modulation context must be [B,{self.input_dim}], got {tuple(context.shape)}"
            )
        if context.shape[0] != hidden.shape[0]:
            raise ValueError(f"context batch {context.shape[0]} != hidden batch {hidden.shape[0]}")
        h = self.encoder(context.to(device=hidden.device, dtype=torch.float32))
        update = self.heads[int(slot)](h)
        if self.mode == "bias":
            return hidden + update[:, None, :].to(dtype=hidden.dtype)
        gamma, beta = update.chunk(2, dim=-1)
        return hidden * (1.0 + gamma[:, None, :].to(dtype=hidden.dtype)) + beta[:, None, :].to(dtype=hidden.dtype)


class BlockwiseGlobalModulatedBlock(nn.Module):
    """A wrapper that applies a shared blockwise adapter after the base block."""

    def __init__(self, block: nn.Module, adapter: BlockwiseGlobalModulationAdapter, slot: int):
        super().__init__()
        self.block = block
        self.slot = int(slot)
        object.__setattr__(self, "_adapter_ref", weakref.ref(adapter))

    def forward(self, x: torch.Tensor, mod: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        out = self.block(x, mod, context)
        adapter_ref = object.__getattribute__(self, "_adapter_ref")
        adapter = adapter_ref()
        if adapter is None:
            return out
        return adapter.apply_to_hidden(self.slot, out)


class BlockwiseTokenModulationAdapter(nn.Module):
    """Zero-init per-token geometry modulation for selected sparse-flow blocks.

    Unlike BlockwiseGlobalModulationAdapter, this consumes a token-aligned
    context [B, N, F] built with the same patchify order as G_inp. It is the
    minimal B2 variant: projection/q_vis/mask evidence stays spatially aligned
    with each sparse latent token.
    """

    def __init__(
        self,
        *,
        input_dim: int,
        channels: int,
        num_blocks: int,
        hidden_dim: int = 128,
        mode: str = "film",
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.channels = int(channels)
        self.num_blocks = int(num_blocks)
        self.hidden_dim = int(hidden_dim)
        self.mode = str(mode or "film").lower()
        if self.mode not in {"bias", "film"}:
            raise ValueError(f"unsupported blockwise token modulation mode={mode!r}")
        out_dim = self.channels * (2 if self.mode == "film" else 1)
        self.encoder = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.SiLU(),
        )
        self.heads = nn.ModuleList([nn.Linear(self.hidden_dim, out_dim) for _ in range(self.num_blocks)])
        for head in self.heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        self._active_context: torch.Tensor | None = None

    def set_context(self, context: torch.Tensor | None) -> None:
        self._active_context = context

    def apply_to_hidden(self, slot: int, hidden: torch.Tensor) -> torch.Tensor:
        context = self._active_context
        if context is None:
            return hidden
        if context.ndim != 3 or context.shape[1] != hidden.shape[1] or context.shape[2] != self.input_dim:
            raise ValueError(
                f"token modulation context must be [B,{hidden.shape[1]},{self.input_dim}], got {tuple(context.shape)}"
            )
        if context.shape[0] != hidden.shape[0]:
            raise ValueError(f"context batch {context.shape[0]} != hidden batch {hidden.shape[0]}")
        h = self.encoder(context.to(device=hidden.device, dtype=torch.float32))
        update = self.heads[int(slot)](h)
        if self.mode == "bias":
            return hidden + update.to(dtype=hidden.dtype)
        gamma, beta = update.chunk(2, dim=-1)
        return hidden * (1.0 + gamma.to(dtype=hidden.dtype)) + beta.to(dtype=hidden.dtype)


class BlockwiseTokenModulatedBlock(nn.Module):
    """A wrapper that applies a shared token-wise adapter after the base block."""

    def __init__(self, block: nn.Module, adapter: BlockwiseTokenModulationAdapter, slot: int):
        super().__init__()
        self.block = block
        self.slot = int(slot)
        object.__setattr__(self, "_adapter_ref", weakref.ref(adapter))

    def forward(self, x: torch.Tensor, mod: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        out = self.block(x, mod, context)
        adapter_ref = object.__getattribute__(self, "_adapter_ref")
        adapter = adapter_ref()
        if adapter is None:
            return out
        return adapter.apply_to_hidden(self.slot, out)


def _set_submodule(root: nn.Module, name: str, module: nn.Module) -> None:
    parts = name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], module)


def inject_flow_lora(
    flow: nn.Module,
    *,
    rank: int = 8,
    alpha: float = 16.0,
    target_blocks: int = 0,
) -> dict[str, Any]:
    """Inject LoRA into Linear layers in sparse-flow transformer blocks.

    target_blocks=0 means all blocks. A positive value means the final N blocks.
    """

    if int(rank) <= 0:
        return {"enabled": False, "rank": int(rank), "alpha": float(alpha), "target_blocks": int(target_blocks), "modules": {}, "trainable_params": 0}
    if not hasattr(flow, "blocks"):
        raise ValueError("sparse flow has no blocks; cannot inject LoRA")
    total_blocks = len(flow.blocks)
    n_blocks = int(target_blocks)
    if n_blocks <= 0:
        selected_indices = list(range(total_blocks))
        label = "all_blocks"
    else:
        selected_indices = list(range(max(0, total_blocks - n_blocks), total_blocks))
        label = f"last_{len(selected_indices)}_blocks"
    modules: dict[str, int] = {}
    for block_idx in selected_indices:
        block = flow.blocks[block_idx]
        for name, module in list(block.named_modules()):
            if not name or isinstance(module, LoRALinear) or not isinstance(module, nn.Linear):
                continue
            wrapped = LoRALinear(module, rank=int(rank), alpha=float(alpha))
            _set_submodule(block, name, wrapped)
            modules[f"blocks.{block_idx}.{name}"] = int(wrapped.lora_A.numel() + wrapped.lora_B.numel())
    trainable = int(sum(v for v in modules.values()))
    flow._points_to_3d_lora_enabled = True
    flow._points_to_3d_lora_summary = {
        "enabled": True,
        "label": label,
        "rank": int(rank),
        "alpha": float(alpha),
        "target_blocks": int(target_blocks),
        "modules": modules,
        "trainable_params": trainable,
    }
    return dict(flow._points_to_3d_lora_summary)


def set_flow_lora_trainable(
    flow: nn.Module,
    *,
    train_input_layer: bool = False,
    train_t_embedder: bool = False,
    train_out_layer: bool = False,
) -> dict[str, Any]:
    """Freeze flow except LoRA params and optional peripheral modules."""

    for param in flow.parameters():
        param.requires_grad = False
    selected: dict[str, int] = {}
    lora_params = 0
    for module_name, module in flow.named_modules():
        if isinstance(module, LoRALinear):
            module.lora_A.requires_grad = True
            module.lora_B.requires_grad = True
            count = int(module.lora_A.numel() + module.lora_B.numel())
            selected[f"{module_name}.lora"] = count
            lora_params += count
    if train_input_layer:
        selected["input_layer"] = sum(p.numel() for p in flow.input_layer.parameters())
        for p in flow.input_layer.parameters():
            p.requires_grad = True
    if train_t_embedder:
        selected["t_embedder"] = sum(p.numel() for p in flow.t_embedder.parameters())
        for p in flow.t_embedder.parameters():
            p.requires_grad = True
        if getattr(flow, "share_mod", False) and hasattr(flow, "adaLN_modulation"):
            selected["adaLN_modulation"] = sum(p.numel() for p in flow.adaLN_modulation.parameters())
            for p in flow.adaLN_modulation.parameters():
                p.requires_grad = True
    if train_out_layer:
        selected["out_layer"] = sum(p.numel() for p in flow.out_layer.parameters())
        for p in flow.out_layer.parameters():
            p.requires_grad = True
    return {
        "enabled": True,
        "label": str(getattr(flow, "_points_to_3d_lora_summary", {}).get("label", "lora")),
        "selected": selected,
        "lora_trainable_params": int(lora_params),
        "trainable_params": int(sum(p.numel() for p in flow.parameters() if p.requires_grad)),
    }


def inject_flow_blockwise_global_modulation(
    flow: nn.Module,
    *,
    input_dim: int,
    hidden_dim: int = 256,
    target_blocks: int = 4,
    mode: str = "bias",
) -> dict[str, Any]:
    """Inject zero-init block-wise global modulation wrappers into G_inp."""

    if getattr(flow, "_points_to_3d_blockwise_global_enabled", False):
        return dict(getattr(flow, "_points_to_3d_blockwise_global_summary", {}))
    if not hasattr(flow, "blocks"):
        raise ValueError("sparse flow has no blocks; cannot inject blockwise global modulation")
    total_blocks = len(flow.blocks)
    n_blocks = int(target_blocks)
    if n_blocks <= 0:
        selected_indices = list(range(total_blocks))
        label = "all_blocks"
    else:
        selected_indices = list(range(max(0, total_blocks - n_blocks), total_blocks))
        label = f"last_{len(selected_indices)}_blocks"
    adapter = BlockwiseGlobalModulationAdapter(
        input_dim=int(input_dim),
        channels=int(flow.model_channels),
        num_blocks=len(selected_indices),
        hidden_dim=int(hidden_dim),
        mode=str(mode),
    ).to(device=flow.input_layer.weight.device)
    flow.blockwise_global_modulation_adapter = adapter
    wrapped: dict[str, int] = {}
    for slot, block_idx in enumerate(selected_indices):
        flow.blocks[block_idx] = BlockwiseGlobalModulatedBlock(flow.blocks[block_idx], adapter, slot)
        wrapped[f"blocks.{block_idx}"] = int(flow.model_channels)
    summary = {
        "enabled": True,
        "label": label,
        "target_blocks": int(target_blocks),
        "selected_indices": selected_indices,
        "input_dim": int(input_dim),
        "hidden_dim": int(hidden_dim),
        "mode": str(mode),
        "wrapped": wrapped,
        "trainable_params": int(sum(p.numel() for p in adapter.parameters())),
    }
    flow._points_to_3d_blockwise_global_enabled = True
    flow._points_to_3d_blockwise_global_summary = summary
    return dict(summary)


def set_flow_blockwise_global_modulation_trainable(flow: nn.Module) -> dict[str, Any]:
    """Freeze G_inp except the block-wise global modulation adapter."""

    adapter = getattr(flow, "blockwise_global_modulation_adapter", None)
    if adapter is None:
        raise ValueError("blockwise global modulation adapter has not been injected")
    for param in flow.parameters():
        param.requires_grad = False
    for param in adapter.parameters():
        param.requires_grad = True
    summary = dict(getattr(flow, "_points_to_3d_blockwise_global_summary", {}))
    summary["trainable_params"] = int(sum(p.numel() for p in flow.parameters() if p.requires_grad))
    summary["selected"] = {"blockwise_global_modulation_adapter": summary["trainable_params"]}
    return summary


def blockwise_global_modulation_input_dim(args: Any) -> int:
    return int(getattr(args, "latent_channels", 8)) + 1 + strict_input_projection_channel_count(args)


def build_blockwise_global_modulation_context(
    *,
    q_vis: torch.Tensor,
    mask: torch.Tensor,
    extra_inputs: torch.Tensor | None = None,
    projection_channels: int = 0,
) -> torch.Tensor:
    pooled = [q_vis.float().mean(dim=(2, 3, 4)), mask.float().mean(dim=(2, 3, 4))]
    channels = int(projection_channels)
    if channels > 0:
        if extra_inputs is None:
            pooled.append(torch.zeros((q_vis.shape[0], channels), device=q_vis.device, dtype=torch.float32))
        else:
            extra = extra_inputs.to(device=q_vis.device, dtype=torch.float32)
            if extra.ndim != 5 or extra.shape[0] != q_vis.shape[0] or extra.shape[1] < channels or extra.shape[-3:] != q_vis.shape[-3:]:
                raise ValueError(
                    f"extra_inputs must be [B,{channels},D,H,W] aligned with q_vis, got {tuple(extra.shape)}"
                )
            pooled.append(extra[:, :channels].mean(dim=(2, 3, 4)))
    return torch.cat(pooled, dim=1)


def set_flow_blockwise_global_modulation_context(flow: nn.Module, context: torch.Tensor | None) -> None:
    adapter = getattr(flow, "blockwise_global_modulation_adapter", None)
    if adapter is not None:
        adapter.set_context(context)


def inject_flow_blockwise_token_modulation(
    flow: nn.Module,
    *,
    input_dim: int,
    hidden_dim: int = 128,
    target_blocks: int = 4,
    mode: str = "film",
) -> dict[str, Any]:
    """Inject zero-init token-wise geometry modulation wrappers into G_inp."""

    if getattr(flow, "_points_to_3d_blockwise_token_enabled", False):
        return dict(getattr(flow, "_points_to_3d_blockwise_token_summary", {}))
    if not hasattr(flow, "blocks"):
        raise ValueError("sparse flow has no blocks; cannot inject blockwise token modulation")
    total_blocks = len(flow.blocks)
    n_blocks = int(target_blocks)
    if n_blocks <= 0:
        selected_indices = list(range(total_blocks))
        label = "all_blocks"
    else:
        selected_indices = list(range(max(0, total_blocks - n_blocks), total_blocks))
        label = f"last_{len(selected_indices)}_blocks"
    adapter = BlockwiseTokenModulationAdapter(
        input_dim=int(input_dim),
        channels=int(flow.model_channels),
        num_blocks=len(selected_indices),
        hidden_dim=int(hidden_dim),
        mode=str(mode),
    ).to(device=flow.input_layer.weight.device)
    flow.blockwise_token_modulation_adapter = adapter
    wrapped: dict[str, int] = {}
    for slot, block_idx in enumerate(selected_indices):
        flow.blocks[block_idx] = BlockwiseTokenModulatedBlock(flow.blocks[block_idx], adapter, slot)
        wrapped[f"blocks.{block_idx}"] = int(flow.model_channels)
    summary = {
        "enabled": True,
        "label": label,
        "target_blocks": int(target_blocks),
        "selected_indices": selected_indices,
        "input_dim": int(input_dim),
        "hidden_dim": int(hidden_dim),
        "mode": str(mode),
        "wrapped": wrapped,
        "trainable_params": int(sum(p.numel() for p in adapter.parameters())),
    }
    flow._points_to_3d_blockwise_token_enabled = True
    flow._points_to_3d_blockwise_token_summary = summary
    return dict(summary)


def set_flow_blockwise_token_modulation_trainable(flow: nn.Module) -> dict[str, Any]:
    """Freeze G_inp except the block-wise token modulation adapter."""

    adapter = getattr(flow, "blockwise_token_modulation_adapter", None)
    if adapter is None:
        raise ValueError("blockwise token modulation adapter has not been injected")
    for param in flow.parameters():
        param.requires_grad = False
    for param in adapter.parameters():
        param.requires_grad = True
    summary = dict(getattr(flow, "_points_to_3d_blockwise_token_summary", {}))
    summary["trainable_params"] = int(sum(p.numel() for p in flow.parameters() if p.requires_grad))
    summary["selected"] = {"blockwise_token_modulation_adapter": summary["trainable_params"]}
    return summary


def blockwise_token_modulation_input_dim(args: Any, *, patch_size: int = 1) -> int:
    channels = int(getattr(args, "latent_channels", 8)) + 1 + strict_input_projection_channel_count(args)
    return int(channels) * (int(patch_size) ** 3)


def build_blockwise_token_modulation_context(
    *,
    q_vis: torch.Tensor,
    mask: torch.Tensor,
    extra_inputs: torch.Tensor | None = None,
    projection_channels: int = 0,
    patch_size: int = 1,
) -> torch.Tensor:
    volumes = [q_vis.float(), mask.float()]
    channels = int(projection_channels)
    if channels > 0:
        if extra_inputs is None:
            volumes.append(torch.zeros((q_vis.shape[0], channels, *q_vis.shape[-3:]), device=q_vis.device, dtype=torch.float32))
        else:
            extra = extra_inputs.to(device=q_vis.device, dtype=torch.float32)
            if extra.ndim != 5 or extra.shape[0] != q_vis.shape[0] or extra.shape[1] < channels or extra.shape[-3:] != q_vis.shape[-3:]:
                raise ValueError(
                    f"extra_inputs must be [B,{channels},D,H,W] aligned with q_vis, got {tuple(extra.shape)}"
                )
            volumes.append(extra[:, :channels])
    volume = torch.cat(volumes, dim=1)
    tokens = patchify(volume, int(patch_size))
    return tokens.view(*tokens.shape[:2], -1).permute(0, 2, 1).contiguous()


def set_flow_blockwise_token_modulation_context(flow: nn.Module, context: torch.Tensor | None) -> None:
    adapter = getattr(flow, "blockwise_token_modulation_adapter", None)
    if adapter is not None:
        adapter.set_context(context)


def replace_sparse_flow_input_layer(flow: nn.Module, *, mask_channels: int = 1) -> nn.Module:
    """Implement Points-to-3D Eq. 3 by replacing G_s input projection with C_s + C_m channels.

    The original TRELLIS sparse flow takes q_t with C_s channels. Points-to-3D
    feeds Concat[q_comb, m_s], so the first projection must accept one
    additional mask channel. All other network blocks are preserved.
    """

    if getattr(flow, "_points_to_3d_input_replaced", False):
        if int(getattr(flow, "_points_to_3d_mask_channels", mask_channels)) != int(mask_channels):
            raise ValueError(
                "sparse flow input layer was already replaced with "
                f"mask_channels={getattr(flow, '_points_to_3d_mask_channels', None)}, requested {mask_channels}"
            )
        return flow
    old_layer = flow.input_layer
    old_in_channels = int(flow.in_channels)
    patch_volume = int(flow.patch_size) ** 3
    new_in_channels = old_in_channels + int(mask_channels)
    new_layer = nn.Linear(new_in_channels * patch_volume, int(flow.model_channels), bias=old_layer.bias is not None)
    new_layer = new_layer.to(device=old_layer.weight.device, dtype=old_layer.weight.dtype)

    with torch.no_grad():
        new_layer.weight.zero_()
        old_w = old_layer.weight.view(old_layer.out_features, old_in_channels, patch_volume)
        new_w = new_layer.weight.view(old_layer.out_features, new_in_channels, patch_volume)
        new_w[:, :old_in_channels, :].copy_(old_w)
        if old_layer.bias is not None:
            new_layer.bias.copy_(old_layer.bias)

    flow.input_layer = new_layer
    flow.in_channels = new_in_channels
    flow._points_to_3d_input_replaced = True
    flow._points_to_3d_latent_channels = old_in_channels
    flow._points_to_3d_mask_channels = int(mask_channels)
    return flow


class ViewGatedConditionAdapter(nn.Module):
    """Token-wise multiview fusion initialized as exact mean aggregation.

    Input shape is [B, V, T, C]. The adapter predicts per-view weights for each
    token and returns [B, T, C]. The final gate layer is zero-initialized, so the
    initial softmax is uniform over views and the output matches simple mean.
    """

    def __init__(
        self,
        *,
        cond_dim: int = 1024,
        hidden_dim: int = 256,
        max_views: int = 8,
        use_view_embed: bool = True,
    ):
        super().__init__()
        self.cond_dim = int(cond_dim)
        self.hidden_dim = int(hidden_dim)
        self.max_views = int(max_views)
        self.use_view_embed = bool(use_view_embed)
        self.view_embed = nn.Parameter(torch.zeros(self.max_views, self.cond_dim)) if self.use_view_embed else None
        self.gate = nn.Sequential(
            nn.LayerNorm(self.cond_dim),
            nn.Linear(self.cond_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, 1),
        )
        final = self.gate[-1]
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def forward(self, cond_views: torch.Tensor) -> torch.Tensor:
        if cond_views.ndim != 4:
            raise ValueError(f"ViewGatedConditionAdapter expects [B,V,T,C], got {tuple(cond_views.shape)}")
        b, v, t, c = cond_views.shape
        if c != self.cond_dim:
            raise ValueError(f"adapter cond_dim={self.cond_dim} but input has {c}")
        if v > self.max_views:
            raise ValueError(f"adapter max_views={self.max_views} but input has {v}")
        orig_dtype = cond_views.dtype
        x = cond_views.float()
        if self.view_embed is not None:
            x = x + self.view_embed[:v].view(1, v, 1, c).float()
        logits = self.gate(x).squeeze(-1)
        weights = torch.softmax(logits, dim=1).unsqueeze(-1)
        fused = (weights * cond_views.float()).sum(dim=1)
        return fused.to(dtype=orig_dtype)


class GeometryAwareViewAdapter(nn.Module):
    """Fuse multiview image tokens with camera-pose and observed-geometry context.

    This is a deliberately small adapter-only branch. It keeps the initial
    behavior close to mean view fusion, then lets training use:

    - per-view intrinsic/extrinsic features;
    - q_vis and m_s summarized on the sparse latent grid.
    """

    def __init__(
        self,
        *,
        cond_dim: int = 1024,
        hidden_dim: int = 256,
        max_views: int = 8,
        pose_dim: int = VIEW_POSE_FEATURE_DIM,
        latent_channels: int = 8,
        projection_channels: int = 0,
        enable_projection_tokens: bool = False,
        projection_token_feature_dim: int = PROJECTION_TOKEN_FEATURE_CHANNELS,
        use_view_embed: bool = True,
    ):
        super().__init__()
        self.cond_dim = int(cond_dim)
        self.hidden_dim = int(hidden_dim)
        self.max_views = int(max_views)
        self.pose_dim = int(pose_dim)
        self.latent_channels = int(latent_channels)
        self.projection_channels = int(projection_channels)
        self.enable_projection_tokens = bool(enable_projection_tokens)
        self.projection_token_feature_dim = int(projection_token_feature_dim)
        self.use_view_embed = bool(use_view_embed)
        self.view_embed = nn.Parameter(torch.zeros(self.max_views, self.cond_dim)) if self.use_view_embed else None
        self.pose_proj = nn.Sequential(
            nn.LayerNorm(self.pose_dim),
            nn.Linear(self.pose_dim, self.cond_dim),
        )
        self.geo_encoder = nn.Sequential(
            nn.Conv3d(self.latent_channels + 1 + self.projection_channels, self.hidden_dim, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv3d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(self.hidden_dim, self.cond_dim),
        )
        self.gate = nn.Sequential(
            nn.LayerNorm(self.cond_dim),
            nn.Linear(self.cond_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, 1),
        )
        self.film = nn.Sequential(
            nn.LayerNorm(self.cond_dim),
            nn.Linear(self.cond_dim, 2 * self.cond_dim),
        )
        if self.enable_projection_tokens:
            self.projection_token_feature_proj = nn.Sequential(
                nn.LayerNorm(self.projection_token_feature_dim),
                nn.Linear(self.projection_token_feature_dim, self.cond_dim),
            )
            self.projection_token_norm = nn.LayerNorm(self.cond_dim)
            self.projection_token_scale = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
            nn.init.zeros_(self.projection_token_feature_proj[-1].weight)
            nn.init.zeros_(self.projection_token_feature_proj[-1].bias)
        else:
            self.projection_token_feature_proj = None
            self.projection_token_norm = None
            self.projection_token_scale = None
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)
        nn.init.zeros_(self.pose_proj[-1].weight)
        nn.init.zeros_(self.pose_proj[-1].bias)
        nn.init.zeros_(self.geo_encoder[-1].weight)
        nn.init.zeros_(self.geo_encoder[-1].bias)
        nn.init.zeros_(self.film[-1].weight)
        nn.init.zeros_(self.film[-1].bias)

    def forward(
        self,
        cond_views: torch.Tensor,
        *,
        q_vis: torch.Tensor,
        mask: torch.Tensor,
        view_pose: torch.Tensor | None = None,
        projection_features: torch.Tensor | None = None,
        projection_token_indices: torch.Tensor | None = None,
        projection_token_weights: torch.Tensor | None = None,
        projection_token_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if cond_views.ndim != 4:
            raise ValueError(f"GeometryAwareViewAdapter expects [B,V,T,C], got {tuple(cond_views.shape)}")
        if q_vis.ndim != 5 or mask.ndim != 5:
            raise ValueError(f"q_vis/mask must be [B,C,D,H,W] / [B,1,D,H,W], got {tuple(q_vis.shape)} {tuple(mask.shape)}")
        b, v, _t, c = cond_views.shape
        if c != self.cond_dim:
            raise ValueError(f"adapter cond_dim={self.cond_dim} but input has {c}")
        if v > self.max_views:
            raise ValueError(f"adapter max_views={self.max_views} but input has {v}")
        if q_vis.shape[0] != b or mask.shape[0] != b:
            raise ValueError(f"batch mismatch: cond_views={tuple(cond_views.shape)} q_vis={tuple(q_vis.shape)} mask={tuple(mask.shape)}")
        orig_dtype = cond_views.dtype
        x = cond_views.float()
        if self.view_embed is not None:
            x = x + self.view_embed[:v].view(1, v, 1, c).float()
        if view_pose is not None:
            pose = view_pose.to(device=x.device, dtype=torch.float32)
            if pose.ndim == 2:
                pose = pose.unsqueeze(0)
            if pose.shape[:2] != (b, v):
                raise ValueError(f"view_pose shape {tuple(pose.shape)} does not match [B,V]=[{b},{v}]")
            if pose.shape[-1] != self.pose_dim:
                raise ValueError(f"view_pose dim {pose.shape[-1]} != adapter pose_dim {self.pose_dim}")
            x = x + self.pose_proj(pose).view(b, v, 1, c)
        geo_inputs = [q_vis.float(), mask.float()]
        if self.projection_channels > 0:
            if projection_features is None:
                raise ValueError("projection-aware adapter requires projection_features")
            proj = projection_features.to(device=q_vis.device, dtype=torch.float32)
            if proj.ndim != 5:
                raise ValueError(f"projection_features must be [B,C,D,H,W], got {tuple(proj.shape)}")
            if proj.shape[0] != b or proj.shape[1] != self.projection_channels or proj.shape[-3:] != q_vis.shape[-3:]:
                raise ValueError(
                    "projection_features shape mismatch: "
                    f"got {tuple(proj.shape)} expected [B,{self.projection_channels},{q_vis.shape[-3]},{q_vis.shape[-2]},{q_vis.shape[-1]}]"
                )
            geo_inputs.append(proj)
        geo_in = torch.cat(geo_inputs, dim=1)
        geo = self.geo_encoder(geo_in).view(b, 1, 1, c)
        logits = self.gate(x + geo).squeeze(-1)
        weights = torch.softmax(logits, dim=1).unsqueeze(-1)
        fused = (weights * cond_views.float()).sum(dim=1)
        gamma_beta = self.film(geo.view(b, c)).view(b, 1, 2, c)
        gamma, beta = gamma_beta[:, :, 0], gamma_beta[:, :, 1]
        fused = fused * (1.0 + gamma) + beta
        if self.enable_projection_tokens and projection_token_indices is not None and projection_token_weights is not None:
            token_indices = projection_token_indices.to(device=x.device, dtype=torch.long)
            token_weights = projection_token_weights.to(device=x.device, dtype=torch.float32)
            if token_indices.ndim != 3 or token_weights.ndim != 3:
                raise ValueError(
                    "projection token indices/weights must be [B,N,V], "
                    f"got {tuple(token_indices.shape)} {tuple(token_weights.shape)}"
                )
            if token_indices.shape != token_weights.shape or token_indices.shape[0] != b or token_indices.shape[2] != v:
                raise ValueError(
                    "projection token indices/weights shape mismatch: "
                    f"indices={tuple(token_indices.shape)} weights={tuple(token_weights.shape)} expected [B,N,{v}]"
                )
            n_proj = int(token_indices.shape[1])
            if n_proj > 0:
                token_indices = token_indices.clamp(0, max(int(cond_views.shape[2]) - 1, 0))
                token_weights = token_weights.clamp_min(0.0)
                weight_sum = token_weights.sum(dim=2, keepdim=True)
                token_weights = token_weights / weight_sum.clamp_min(1.0e-6)
                cond_perm = cond_views.float().permute(0, 1, 3, 2)
                gather_idx = token_indices.permute(0, 2, 1).unsqueeze(2).expand(-1, -1, c, -1)
                gathered = torch.gather(cond_perm, dim=3, index=gather_idx).permute(0, 3, 1, 2)
                proj_tokens = (gathered * token_weights.unsqueeze(-1)).sum(dim=2)
                if projection_token_features is not None:
                    feat = projection_token_features.to(device=x.device, dtype=torch.float32)
                    if feat.ndim != 3 or feat.shape[0] != b or feat.shape[1] != n_proj or feat.shape[2] != self.projection_token_feature_dim:
                        raise ValueError(
                            "projection_token_features must be [B,N,F], "
                            f"got {tuple(feat.shape)} expected [B,{n_proj},{self.projection_token_feature_dim}]"
                        )
                    proj_tokens = proj_tokens + self.projection_token_feature_proj(feat)
                proj_tokens = self.projection_token_norm(proj_tokens)
                proj_tokens = proj_tokens * self.projection_token_scale.to(device=proj_tokens.device, dtype=proj_tokens.dtype)
                fused = torch.cat([fused, proj_tokens.to(dtype=fused.dtype)], dim=1)
        return fused.to(dtype=orig_dtype)


def build_condition_adapter(args: Any, *, cond_dim: int = 1024) -> ViewGatedConditionAdapter | None:
    aggregation = str(getattr(args, "image_cond_aggregation", "") or "").lower()
    if aggregation in GEOMETRY_AWARE_AGGREGATIONS or aggregation in PROJECTION_AWARE_AGGREGATIONS:
        projection_channels = 0
        enable_projection_tokens = False
        if aggregation in PROJECTION_GRID_AWARE_AGGREGATIONS:
            projection_channels = PROJECTION_FEATURE_CHANNELS
        elif aggregation in PROJECTION_TOKEN_AWARE_AGGREGATIONS:
            projection_channels = PROJECTION_TOKEN_FEATURE_CHANNELS
            enable_projection_tokens = True
        return GeometryAwareViewAdapter(
            cond_dim=int(cond_dim),
            hidden_dim=int(getattr(args, "condition_adapter_hidden_dim", 256)),
            max_views=int(getattr(args, "condition_adapter_max_views", 8)),
            pose_dim=int(getattr(args, "condition_adapter_pose_dim", VIEW_POSE_FEATURE_DIM)),
            latent_channels=int(getattr(args, "latent_channels", 8)),
            projection_channels=projection_channels,
            enable_projection_tokens=enable_projection_tokens,
            projection_token_feature_dim=PROJECTION_TOKEN_FEATURE_CHANNELS,
            use_view_embed=bool(getattr(args, "condition_adapter_use_view_embed", True)),
        )
    if aggregation not in VIEW_GATED_AGGREGATIONS:
        return None
    return ViewGatedConditionAdapter(
        cond_dim=int(cond_dim),
        hidden_dim=int(getattr(args, "condition_adapter_hidden_dim", 256)),
        max_views=int(getattr(args, "condition_adapter_max_views", 8)),
        use_view_embed=bool(getattr(args, "condition_adapter_use_view_embed", True)),
    )


def frame_pose_feature(frame: dict[str, Any], *, order: int, view_count: int) -> np.ndarray:
    feat = np.zeros((VIEW_POSE_FEATURE_DIM,), dtype=np.float32)
    offset = 0
    extrinsic = frame.get("extrinsic")
    if extrinsic is not None:
        arr = np.asarray(extrinsic, dtype=np.float32).reshape(-1)
        n = min(16, arr.size)
        feat[offset : offset + n] = arr[:n]
        feat[29] = 1.0
    offset += 16
    intrinsic = frame.get("intrinsic")
    if intrinsic is not None:
        arr = np.asarray(intrinsic, dtype=np.float32).reshape(-1)
        if arr.size >= 9:
            # Normalize the dominant pixel-scale terms so they are comparable
            # to the pose features before the adapter's LayerNorm.
            norm = max(float(abs(arr[0])), float(abs(arr[4])), 1.0)
            arr = arr / norm
        n = min(9, arr.size)
        feat[offset : offset + n] = arr[:n]
        feat[30] = 1.0
    offset += 9
    source_view_index = frame.get("source_view_index", order)
    try:
        feat[offset] = float(source_view_index) / max(float(view_count), 1.0)
    except (TypeError, ValueError):
        feat[offset] = float(order) / max(float(view_count), 1.0)
    feat[offset + 1] = float(order) / max(float(view_count - 1), 1.0)
    feat[offset + 2] = float(view_count)
    feat[31] = 1.0
    return feat


def build_projection_grid_features(
    *,
    mask_paths: list[str | None],
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    extrinsics_type: str = "c2w",
    camera_forward_sign: float = 1.0,
    grid_transform: str = "pixal3d_rotation",
    latent_resolution: int = 16,
    image_mask_crop_resolution: int = 518,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Project latent-cell centers into source masks and return [1,3,D,H,W].

    Channels are:
      0. visible view ratio;
      1. mask-support view ratio;
      2. support / visible ratio.

    This is intentionally a latent-grid visual-support feature. It is not yet
    patch-token sampling from image features.
    """

    res = int(latent_resolution)
    intr = np.asarray(intrinsics, dtype=np.float32)
    extr = np.asarray(extrinsics, dtype=np.float32)
    valid_ids: list[int] = []
    masks: list[np.ndarray] = []
    for idx, mask_path in enumerate(mask_paths):
        if mask_path is None:
            continue
        if idx >= intr.shape[0] or idx >= extr.shape[0]:
            continue
        try:
            masks.append(load_mask(mask_path))
            valid_ids.append(idx)
        except FileNotFoundError:
            continue
    if not valid_ids:
        return torch.zeros((1, PROJECTION_FEATURE_CHANNELS, res, res, res), device=device, dtype=dtype)

    coords = np.stack(
        np.meshgrid(np.arange(res), np.arange(res), np.arange(res), indexing="ij"),
        axis=-1,
    ).reshape(-1, 3).astype(np.int32)
    points = coords_to_points(coords, resolution=res)
    transform = str(grid_transform or "identity")
    if transform.lower() in {"none", "null", ""}:
        transform = "identity"
    points = apply_grid_transform(points, transform)
    support = project_points_to_masks(
        points,
        masks,
        intr[np.asarray(valid_ids, dtype=np.int64)],
        extr[np.asarray(valid_ids, dtype=np.int64)],
        extrinsics_type=str(extrinsics_type or "c2w"),
        camera_forward_sign=float(camera_forward_sign),
        min_support=1.0,
        min_support_ratio=0.0,
        front_depth=False,
    )
    visible = np.asarray(support["visible"], dtype=np.float32)
    hit = np.asarray(support["support"], dtype=np.float32)
    denom_views = float(max(len(valid_ids), 1))
    feat = np.stack(
        [
            visible / denom_views,
            hit / denom_views,
            hit / np.maximum(visible, 1.0),
        ],
        axis=0,
    ).reshape(PROJECTION_FEATURE_CHANNELS, res, res, res)
    feat = np.nan_to_num(feat, nan=0.0, posinf=1.0, neginf=0.0).clip(0.0, 1.0)
    return torch.from_numpy(feat[None].astype(np.float32)).to(device=device, dtype=dtype)


def infer_spatial_token_layout(token_count: int, *, max_special_tokens: int = 16) -> tuple[int, int] | None:
    """Infer (special-token offset, square grid size) from a TRELLIS image-token count."""

    n = int(token_count)
    if n <= 0:
        return None
    for offset in range(0, int(max_special_tokens) + 1):
        spatial = n - offset
        if spatial <= 0:
            continue
        grid = int(round(math.sqrt(float(spatial))))
        if grid * grid == spatial:
            return offset, grid
    return None


def mask_crop_uv_to_condition_uv(mask: np.ndarray, u: np.ndarray, v: np.ndarray, *, resolution: int = 518) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map source-image uv into the masked crop used by encode_image_condition."""

    h, w = mask.shape[:2]
    u = np.asarray(u, dtype=np.float32)
    v = np.asarray(v, dtype=np.float32)
    cu = np.zeros_like(u, dtype=np.float32)
    cv = np.zeros_like(v, dtype=np.float32)
    valid = np.isfinite(u) & np.isfinite(v)
    ys, xs = np.nonzero(mask > 0.5)
    if len(xs) == 0:
        side = float(max(w, h, 1))
        cu = (u + 0.5 * (side - float(w))) * (float(resolution) / side)
        cv = (v + 0.5 * (side - float(h))) * (float(resolution) / side)
        valid &= (cu >= 0.0) & (cu < float(resolution)) & (cv >= 0.0) & (cv < float(resolution))
        return cu, cv, valid

    left, right = int(xs.min()), int(xs.max())
    top, bottom = int(ys.min()), int(ys.max())
    center_x = 0.5 * (left + right)
    center_y = 0.5 * (top + bottom)
    size = max(1, int(max(right - left + 1, bottom - top + 1) * 1.1))
    crop_left = max(0, int(center_x - size // 2))
    crop_top = max(0, int(center_y - size // 2))
    crop_right = min(w, int(center_x + size // 2))
    crop_bottom = min(h, int(center_y + size // 2))
    crop_w = max(1, crop_right - crop_left)
    crop_h = max(1, crop_bottom - crop_top)
    side = float(max(crop_w, crop_h, 1))
    pad_x = float((int(side) - crop_w) // 2)
    pad_y = float((int(side) - crop_h) // 2)
    cu = (u - float(crop_left) + pad_x) * (float(resolution) / side)
    cv = (v - float(crop_top) + pad_y) * (float(resolution) / side)
    valid &= (u >= float(crop_left)) & (u < float(crop_right)) & (v >= float(crop_top)) & (v < float(crop_bottom))
    valid &= (cu >= 0.0) & (cu < float(resolution)) & (cv >= 0.0) & (cv < float(resolution))
    return cu, cv, valid


def build_projection_condition_features(
    *,
    mask_paths: list[str | None],
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    token_count: int,
    max_projection_tokens: int = 512,
    extrinsics_type: str = "c2w",
    camera_forward_sign: float = 1.0,
    grid_transform: str = "pixal3d_rotation",
    latent_resolution: int = 16,
    image_mask_crop_resolution: int = 518,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> dict[str, torch.Tensor | None]:
    """Build projection-aware v2 features.

    Returns:
      projection_features: [1,6,D,H,W] latent-grid geometry evidence;
      projection_token_indices: [1,N,V] image-token ids per selected 3D cell;
      projection_token_weights: [1,N,V] per-view support/visibility weights;
      projection_token_features: [1,N,6] selected cell features.
    """

    res = int(latent_resolution)
    view_count = len(mask_paths)
    intr = np.asarray(intrinsics, dtype=np.float32)
    extr = np.asarray(extrinsics, dtype=np.float32)
    valid_ids: list[int] = []
    masks: list[np.ndarray | None] = [None for _ in range(view_count)]
    for idx, mask_path in enumerate(mask_paths):
        if mask_path is None:
            continue
        if idx >= intr.shape[0] or idx >= extr.shape[0]:
            continue
        try:
            masks[idx] = load_mask(mask_path)
            valid_ids.append(idx)
        except FileNotFoundError:
            continue

    empty_feat = torch.zeros((1, PROJECTION_TOKEN_FEATURE_CHANNELS, res, res, res), device=device, dtype=dtype)
    if not valid_ids or view_count <= 0:
        return {
            "projection_features": empty_feat,
            "projection_token_indices": None,
            "projection_token_weights": None,
            "projection_token_features": None,
        }

    coords = np.stack(
        np.meshgrid(np.arange(res), np.arange(res), np.arange(res), indexing="ij"),
        axis=-1,
    ).reshape(-1, 3).astype(np.int32)
    points = coords_to_points(coords, resolution=res)
    transform = str(grid_transform or "identity")
    if transform.lower() in {"none", "null", ""}:
        transform = "identity"
    points = apply_grid_transform(points, transform)
    n = int(points.shape[0])
    visible = np.zeros(n, dtype=np.float32)
    support = np.zeros(n, dtype=np.float32)
    inv_depth_sum = np.zeros(n, dtype=np.float32)
    inv_depth_max = np.zeros(n, dtype=np.float32)
    token_indices = np.zeros((n, view_count), dtype=np.int64)
    token_support_weight = np.zeros((n, view_count), dtype=np.float32)
    token_visible_weight = np.zeros((n, view_count), dtype=np.float32)
    layout = infer_spatial_token_layout(int(token_count))
    ones = np.ones((n, 1), dtype=np.float32)
    pts_h = np.concatenate([points.astype(np.float32, copy=False), ones], axis=1)

    for view_idx in valid_ids:
        mask = masks[view_idx]
        if mask is None:
            continue
        K = intr[view_idx]
        E = extr[view_idx]
        w2c = np.linalg.inv(E) if str(extrinsics_type or "c2w") == "c2w" else E
        cam = (w2c @ pts_h.T).T[:, :3]
        signed_depth = cam[:, 2] * float(camera_forward_sign)
        valid_depth = signed_depth > 1.0e-6
        z = np.maximum(signed_depth, 1.0e-6)
        u = K[0, 0] * (cam[:, 0] / z) + K[0, 2]
        v = K[1, 1] * (cam[:, 1] / z) + K[1, 2]
        h, w = mask.shape[:2]
        ui = np.rint(u).astype(np.int32)
        vi = np.rint(v).astype(np.int32)
        in_image = valid_depth & (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
        ids = np.nonzero(in_image)[0]
        if not len(ids):
            continue
        hit = mask[vi[ids], ui[ids]].astype(np.float32)
        visible[ids] += 1.0
        support[ids] += hit
        inv_depth = 1.0 / (1.0 + np.abs(signed_depth[ids]).astype(np.float32))
        inv_depth_sum[ids] += inv_depth
        inv_depth_max[ids] = np.maximum(inv_depth_max[ids], inv_depth)
        token_visible_weight[ids, view_idx] = 1.0
        token_support_weight[ids, view_idx] = hit
        if layout is not None:
            offset, grid = layout
            cu, cv, crop_valid = mask_crop_uv_to_condition_uv(mask, u, v, resolution=int(image_mask_crop_resolution))
            token_valid = in_image & crop_valid
            tids = np.nonzero(token_valid)[0]
            if len(tids):
                tx = np.floor(cu[tids] * float(grid) / float(image_mask_crop_resolution)).astype(np.int64)
                ty = np.floor(cv[tids] * float(grid) / float(image_mask_crop_resolution)).astype(np.int64)
                tx = np.clip(tx, 0, grid - 1)
                ty = np.clip(ty, 0, grid - 1)
                token_indices[tids, view_idx] = int(offset) + ty * int(grid) + tx

    denom_views = float(max(len(valid_ids), 1))
    visible_ratio = visible / denom_views
    support_ratio = support / denom_views
    support_over_visible = support / np.maximum(visible, 1.0)
    inv_depth_mean = inv_depth_sum / np.maximum(visible, 1.0)
    score = support + 0.1 * visible
    max_tokens = int(max_projection_tokens)
    selected = np.zeros(n, dtype=np.float32)
    keep_ids: np.ndarray
    if max_tokens > 0 and np.any(score > 0):
        candidates = np.nonzero(score > 0)[0]
        order = np.argsort(-score[candidates], kind="stable")
        keep_ids = candidates[order[: min(max_tokens, len(order))]]
        selected[keep_ids] = 1.0
    else:
        keep_ids = np.zeros((0,), dtype=np.int64)
    feat = np.stack(
        [visible_ratio, support_ratio, support_over_visible, inv_depth_mean, inv_depth_max, selected],
        axis=0,
    ).reshape(PROJECTION_TOKEN_FEATURE_CHANNELS, res, res, res)
    feat = np.nan_to_num(feat, nan=0.0, posinf=1.0, neginf=0.0).clip(0.0, 1.0)
    projection_features = torch.from_numpy(feat[None].astype(np.float32)).to(device=device, dtype=dtype)
    if layout is None or keep_ids.size == 0:
        return {
            "projection_features": projection_features,
            "projection_token_indices": None,
            "projection_token_weights": None,
            "projection_token_features": None,
        }

    weights = token_support_weight[keep_ids].astype(np.float32)
    fallback = weights.sum(axis=1) <= 1.0e-6
    if np.any(fallback):
        weights[fallback] = token_visible_weight[keep_ids][fallback]
    token_feat = np.stack(
        [visible_ratio[keep_ids], support_ratio[keep_ids], support_over_visible[keep_ids], inv_depth_mean[keep_ids], inv_depth_max[keep_ids], selected[keep_ids]],
        axis=1,
    ).astype(np.float32)
    return {
        "projection_features": projection_features,
        "projection_token_indices": torch.from_numpy(token_indices[keep_ids][None].astype(np.int64)).to(device=device),
        "projection_token_weights": torch.from_numpy(weights[None].astype(np.float32)).to(device=device, dtype=torch.float32),
        "projection_token_features": torch.from_numpy(token_feat[None].astype(np.float32)).to(device=device, dtype=torch.float32),
    }


def strict_input_projection_channel_count(args: Any) -> int:
    """Return extra G_inp projection-grid channels requested by args."""

    if not bool(getattr(args, "strict_input_projection_grid", False)):
        return 0
    channels = int(getattr(args, "strict_input_projection_channels", PROJECTION_TOKEN_FEATURE_CHANNELS))
    if channels not in {PROJECTION_FEATURE_CHANNELS, PROJECTION_TOKEN_FEATURE_CHANNELS}:
        raise ValueError(
            f"strict_input_projection_channels must be {PROJECTION_FEATURE_CHANNELS} or "
            f"{PROJECTION_TOKEN_FEATURE_CHANNELS}, got {channels}"
        )
    return channels


def build_strict_input_projection_features(
    *,
    batch: dict[str, Any],
    args: Any,
    mask: torch.Tensor,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor | None:
    """Build projection-grid features appended directly to G_inp input.

    This is separate from projection-aware condition adapters. The returned
    tensor is [B,C,D,H,W] and is meant to be concatenated after
    Concat[q_comb, m_s]. C=3 uses the older visual-support grid. C=6 uses the
    v2 projection grid statistics without adding projection image tokens.
    """

    channels = strict_input_projection_channel_count(args)
    if channels <= 0:
        return None
    view_intrinsics_all = batch.get("view_intrinsics")
    view_extrinsics_all = batch.get("view_extrinsics")
    view_camera_forward_all = batch.get("view_camera_forward_sign")
    if view_intrinsics_all is None or view_extrinsics_all is None:
        raise ValueError("strict input projection grid requires view_intrinsics/view_extrinsics in batch")
    features: list[torch.Tensor] = []
    for row_idx, mask_paths in enumerate(batch["image_mask_paths"]):
        common_kwargs = {
            "mask_paths": list(mask_paths),
            "intrinsics": view_intrinsics_all[row_idx].detach().cpu().numpy(),
            "extrinsics": view_extrinsics_all[row_idx].detach().cpu().numpy(),
            "extrinsics_type": batch.get("view_extrinsics_type", ["c2w"])[row_idx],
            "camera_forward_sign": float(view_camera_forward_all[row_idx].item()) if view_camera_forward_all is not None else 1.0,
            "grid_transform": batch.get("projection_grid_transform", ["pixal3d_rotation"])[row_idx],
            "latent_resolution": int(mask.shape[-1]),
            "device": device,
            "dtype": dtype,
        }
        if channels == PROJECTION_FEATURE_CHANNELS:
            feat = build_projection_grid_features(**common_kwargs)
        else:
            built = build_projection_condition_features(
                **common_kwargs,
                token_count=0,
                max_projection_tokens=0,
                image_mask_crop_resolution=int(getattr(args, "image_mask_crop_resolution", 518)),
            )
            feat = built["projection_features"]
        features.append(feat)
    if not features:
        return None
    return torch.cat(features, dim=0).to(device=device, dtype=dtype)


class StrictLatentDataset(torch.utils.data.Dataset):
    """Dataset of q_gt, q_vis and m_s generated from visible point priors."""

    def __init__(
        self,
        manifest: str | Path,
        *,
        indices: str = "all",
        mask_dilate64: int = 0,
        mask_dilate16: int = 0,
        source_grid_resolution: int = 64,
        latent_grid_resolution: int = 16,
        use_image_cond: bool = True,
        image_max_views: int = 4,
        image_frame_select: str = "uniform",
        image_select_seed: int = 0,
        image_use_source_mask: bool = True,
        projection_grid_transform: str = "pixal3d_rotation",
    ):
        self.manifest_path = Path(manifest)
        self.payload, samples = load_manifest(self.manifest_path)
        selected = parse_indices(indices, len(samples))
        self.samples = [samples[i] for i in selected]
        self.latent_root = self.payload.get("latent_root") or str(self.manifest_path.parent)
        self.mask_dilate64 = int(mask_dilate64)
        self.mask_dilate16 = int(mask_dilate16)
        self.source_grid_resolution = int(source_grid_resolution or self.payload.get("source_grid_resolution", 64))
        self.latent_grid_resolution = int(latent_grid_resolution or self.payload.get("latent_grid_resolution", 16))
        self.use_image_cond = bool(use_image_cond)
        self.image_max_views = int(image_max_views)
        self.image_frame_select = str(image_frame_select)
        self.image_select_seed = int(image_select_seed)
        self.image_use_source_mask = bool(image_use_source_mask)
        self.image_resolver = SourceImageResolver(self.payload.get("source_manifest")) if self.use_image_cond else None
        self.projection_grid_transform = str(projection_grid_transform or self.payload.get("grid_transform") or "pixal3d_rotation")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        latent_path = resolve_path(self.latent_root, sample["latent_npz"])
        with np.load(latent_path) as data:
            q_gt = np.asarray(data["q_gt"], dtype=np.float32)
            q_vis = np.asarray(data["q_vis"], dtype=np.float32)
            saved_m_s = normalize_latent_mask(
                np.asarray(data["m_s"], dtype=np.float32),
                latent_resolution=self.latent_grid_resolution,
            )
            prior_coords = np.asarray(data["prior_coords"], dtype=np.int32)
            target_coords = np.asarray(data["target_coords"], dtype=np.int32)
        if self.mask_dilate64 == 0 and self.mask_dilate16 == 0:
            mask = saved_m_s
        else:
            mask = latent_mask_from_prior(
                prior_coords,
                mask_dilate64=self.mask_dilate64,
                mask_dilate16=self.mask_dilate16,
                source_resolution=self.source_grid_resolution,
                latent_resolution=self.latent_grid_resolution,
            )
        item: dict[str, Any] = {
            "uid": str(sample.get("uid", index)),
            "source_index": int(sample.get("source_index", index)),
            "q_gt": torch.from_numpy(q_gt.astype(np.float32)),
            "q_vis": torch.from_numpy(q_vis.astype(np.float32)),
            "m_s": torch.from_numpy(mask.astype(np.float32)),
            "saved_m_s": torch.from_numpy(saved_m_s.astype(np.float32)),
            "prior_coords": prior_coords.astype(np.int32),
            "target_coords": target_coords.astype(np.int32),
            "latent_path": str(latent_path),
        }
        if self.use_image_cond:
            if self.image_resolver is None:
                raise RuntimeError("image resolver was not initialized")
            source_payload, source_sample, _source_manifest_path = self.image_resolver.source_sample(sample)
            image_paths, mask_paths, frames = self.image_resolver.image_mask_paths_with_frames(
                sample,
                max_views=self.image_max_views,
                frame_select=self.image_frame_select,
                seed=self.image_select_seed,
            )
            if not self.image_use_source_mask:
                mask_paths = [None for _ in image_paths]
            item["image_paths"] = image_paths
            item["image_mask_paths"] = mask_paths
            top_intrinsic = source_sample.get("intrinsic", source_payload.get("intrinsic"))
            view_intrinsics: list[np.ndarray] = []
            view_extrinsics: list[np.ndarray] = []
            for frame in frames:
                intrinsic = frame.get("intrinsic", top_intrinsic)
                extrinsic = frame.get("extrinsic")
                if intrinsic is None:
                    intrinsic = np.eye(3, dtype=np.float32)
                if extrinsic is None:
                    extrinsic = np.eye(4, dtype=np.float32)
                view_intrinsics.append(np.asarray(intrinsic, dtype=np.float32).reshape(3, 3))
                view_extrinsics.append(np.asarray(extrinsic, dtype=np.float32).reshape(4, 4))
            item["view_intrinsics"] = torch.from_numpy(np.stack(view_intrinsics, axis=0).astype(np.float32))
            item["view_extrinsics"] = torch.from_numpy(np.stack(view_extrinsics, axis=0).astype(np.float32))
            item["view_extrinsics_type"] = str(
                source_sample.get("extrinsics_type", source_payload.get("extrinsics_type", "c2w"))
            )
            item["view_camera_forward_sign"] = float(
                source_sample.get("camera_forward_sign", source_payload.get("camera_forward_sign", 1.0))
            )
            item["projection_grid_transform"] = str(
                sample.get("grid_transform", source_sample.get("grid_transform", source_payload.get("grid_transform", self.projection_grid_transform)))
                or self.projection_grid_transform
            )
            item["view_pose_features"] = torch.from_numpy(
                np.stack(
                    [
                        frame_pose_feature(frame, order=view_order, view_count=len(frames))
                        for view_order, frame in enumerate(frames)
                    ],
                    axis=0,
                ).astype(np.float32)
            )
        else:
            item["image_paths"] = []
            item["image_mask_paths"] = []
            item["view_pose_features"] = torch.zeros((0, VIEW_POSE_FEATURE_DIM), dtype=torch.float32)
            item["view_intrinsics"] = torch.zeros((0, 3, 3), dtype=torch.float32)
            item["view_extrinsics"] = torch.zeros((0, 4, 4), dtype=torch.float32)
            item["view_extrinsics_type"] = "c2w"
            item["view_camera_forward_sign"] = 1.0
            item["projection_grid_transform"] = self.projection_grid_transform
        return item


def strict_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "uids": [x["uid"] for x in batch],
        "source_indices": [x["source_index"] for x in batch],
        "q_gt": torch.stack([x["q_gt"] for x in batch], dim=0),
        "q_vis": torch.stack([x["q_vis"] for x in batch], dim=0),
        "m_s": torch.stack([x["m_s"] for x in batch], dim=0),
        "saved_m_s": torch.stack([x["saved_m_s"] for x in batch], dim=0),
        "prior_coords": [x["prior_coords"] for x in batch],
        "target_coords": [x["target_coords"] for x in batch],
        "latent_paths": [x["latent_path"] for x in batch],
        "image_paths": [x.get("image_paths", []) for x in batch],
        "image_mask_paths": [x.get("image_mask_paths", []) for x in batch],
        "view_pose_features": torch.stack([x["view_pose_features"] for x in batch], dim=0),
        "view_intrinsics": torch.stack([x["view_intrinsics"] for x in batch], dim=0),
        "view_extrinsics": torch.stack([x["view_extrinsics"] for x in batch], dim=0),
        "view_extrinsics_type": [x.get("view_extrinsics_type", "c2w") for x in batch],
        "view_camera_forward_sign": torch.tensor([float(x.get("view_camera_forward_sign", 1.0)) for x in batch], dtype=torch.float32),
        "projection_grid_transform": [x.get("projection_grid_transform", "pixal3d_rotation") for x in batch],
    }


def encode_multiview_condition(
    pipeline,
    image_paths: list[str],
    image_mask_paths: list[str | None],
    *,
    device: torch.device | str,
    dtype: torch.dtype,
    aggregation: str = "concat",
    image_use_source_mask: bool = True,
    image_mask_crop_resolution: int = 518,
    adapter: nn.Module | None = None,
    q_vis: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
    view_pose: torch.Tensor | None = None,
    projection_features: torch.Tensor | None = None,
    projection_build_kwargs: dict[str, Any] | None = None,
    projection_token_max_cells: int = 512,
) -> torch.Tensor:
    if not image_paths:
        raise ValueError("strict Points-to-3D image condition requires at least one view")
    aggregation = str(aggregation)
    agg_lower = aggregation.lower()
    encode_aggregation = "views" if agg_lower in ADAPTER_AGGREGATIONS else aggregation
    cond = encode_image_condition(
        pipeline,
        list(image_paths),
        device=device,
        dtype=dtype,
        aggregation=encode_aggregation,
        preprocess=False,
        mask_paths=list(image_mask_paths),
        use_source_mask=bool(image_use_source_mask),
        mask_crop_resolution=int(image_mask_crop_resolution),
    )
    if agg_lower in GEOMETRY_AWARE_AGGREGATIONS or agg_lower in PROJECTION_AWARE_AGGREGATIONS:
        if adapter is None:
            raise ValueError(f"image_cond_aggregation={aggregation} requires a condition adapter")
        if q_vis is None or mask is None:
            raise ValueError(f"image_cond_aggregation={aggregation} requires q_vis and mask")
        projection_token_indices = None
        projection_token_weights = None
        projection_token_features = None
        if agg_lower in PROJECTION_GRID_AWARE_AGGREGATIONS and projection_features is None:
            if projection_build_kwargs is None:
                raise ValueError(f"image_cond_aggregation={aggregation} requires projection_build_kwargs")
            projection_features = build_projection_grid_features(**projection_build_kwargs)
        elif agg_lower in PROJECTION_TOKEN_AWARE_AGGREGATIONS:
            if projection_build_kwargs is None:
                raise ValueError(f"image_cond_aggregation={aggregation} requires projection_build_kwargs")
            built = build_projection_condition_features(
                **projection_build_kwargs,
                token_count=int(cond.shape[2]),
                max_projection_tokens=int(projection_token_max_cells),
            )
            projection_features = built["projection_features"]
            projection_token_indices = built["projection_token_indices"]
            projection_token_weights = built["projection_token_weights"]
            projection_token_features = built["projection_token_features"]
        cond = adapter(
            cond,
            q_vis=q_vis,
            mask=mask,
            view_pose=view_pose,
            projection_features=projection_features,
            projection_token_indices=projection_token_indices,
            projection_token_weights=projection_token_weights,
            projection_token_features=projection_token_features,
        )
    elif agg_lower in VIEW_GATED_AGGREGATIONS:
        if adapter is None:
            raise ValueError(f"image_cond_aggregation={aggregation} requires a condition adapter")
        cond = adapter(cond)
    return cond


def build_q_comb_t(
    *,
    q_gt: torch.Tensor,
    q_vis: torch.Tensor,
    mask: torch.Tensor,
    sampler,
    t: float,
    noise: torch.Tensor,
    extra_inputs: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the paper input q_comb(t)=m*q_vis+(1-m)*q_t and CFM target."""

    x_t, gt_v = sampler._get_model_gt(q_gt, float(t), noise)
    q_comb_t = mask * q_vis + (1.0 - mask) * x_t
    inputs = [q_comb_t, mask]
    if extra_inputs is not None:
        extra = extra_inputs.to(device=q_comb_t.device, dtype=q_comb_t.dtype)
        if extra.ndim != 5 or extra.shape[0] != q_comb_t.shape[0] or extra.shape[-3:] != q_comb_t.shape[-3:]:
            raise ValueError(f"extra_inputs must be [B,C,D,H,W] aligned with q_comb_t, got {tuple(extra.shape)}")
        inputs.append(extra)
    x_inp = torch.cat(inputs, dim=1)
    return x_inp, gt_v


@torch.no_grad()
def encode_sparse_coords_to_latent(
    encoder: nn.Module,
    coords_np: np.ndarray,
    *,
    device: torch.device | str,
    resolution: int = 64,
) -> torch.Tensor:
    coords = np.asarray(coords_np, dtype=np.int32)
    if coords.size == 0:
        coords_t = torch.zeros((0, 4), device=device, dtype=torch.long)
    else:
        xyz = coords[:, -3:].astype(np.int32, copy=False)
        batch = np.zeros((xyz.shape[0], 1), dtype=np.int32)
        coords_t = torch.from_numpy(np.concatenate([batch, xyz], axis=1)).to(device=device, dtype=torch.long)
    occ = coords_to_batched_occ(
        coords_t,
        1,
        resolution=int(resolution),
        device=device,
        dtype=next(encoder.parameters()).dtype,
    )
    return encoder(occ, sample_posterior=False).float()


@torch.no_grad()
def sample_stock_sparse_latent(
    *,
    pipeline,
    cond: torch.Tensor,
    steps: int | None = None,
    cfg_strength: float | None = None,
    rescale_t: float | None = None,
    seed: int = 0,
) -> torch.Tensor:
    """Sample native TRELLIS sparse latent with the unmodified sparse flow."""

    flow = pipeline.models["sparse_structure_flow_model"]
    if getattr(flow, "_points_to_3d_input_replaced", False):
        raise RuntimeError("stock sparse latent must be sampled before replacing the sparse flow input layer")
    gen = torch.Generator(device=cond.device)
    gen.manual_seed(int(seed))
    noise = torch.randn(
        cond.shape[0],
        int(flow.in_channels),
        int(flow.resolution),
        int(flow.resolution),
        int(flow.resolution),
        device=cond.device,
        generator=gen,
    )
    params = {**pipeline.sparse_structure_sampler_params}
    if steps is not None and int(steps) > 0:
        params["steps"] = int(steps)
    if cfg_strength is not None and float(cfg_strength) > 0:
        params["cfg_strength"] = float(cfg_strength)
    if rescale_t is not None and float(rescale_t) > 0:
        params["rescale_t"] = float(rescale_t)
    params["verbose"] = False
    return pipeline.sparse_structure_sampler.sample(
        flow,
        noise,
        cond=cond,
        neg_cond=torch.zeros_like(cond),
        **params,
    ).samples.float()


def inpaint_model_pred(
    flow,
    x_inp: torch.Tensor,
    t: float,
    cond: torch.Tensor,
    *,
    cfg_strength: float = 1.0,
    neg_cond: torch.Tensor | None = None,
) -> torch.Tensor:
    t_tensor = torch.tensor([1000.0 * float(t)] * x_inp.shape[0], device=x_inp.device, dtype=torch.float32)
    pred = flow(x_inp, t_tensor, cond)
    # TRELLIS treats cfg_strength=1 as conditional-only inference.
    if float(cfg_strength) == 1.0:
        return pred
    if neg_cond is None:
        neg_cond = torch.zeros_like(cond)
    pred_neg = flow(x_inp, t_tensor, neg_cond)
    if float(cfg_strength) == 0.0:
        return pred_neg
    return float(cfg_strength) * pred + (1.0 - float(cfg_strength)) * pred_neg


@torch.no_grad()
def sample_points_to_3d_strict(
    *,
    flow,
    sampler,
    q_vis: torch.Tensor,
    mask: torch.Tensor,
    cond: torch.Tensor,
    steps: int,
    inpaint_steps: int,
    rescale_t: float,
    cfg_strength: float,
    initial_noise: torch.Tensor | None = None,
    initial_sample: torch.Tensor | None = None,
    extra_inputs: torch.Tensor | None = None,
    start_t: float = 1.0,
) -> torch.Tensor:
    """Two-stage Points-to-3D sampling: structural inpainting then boundary refinement."""

    if steps <= 0:
        raise ValueError("steps must be positive")
    inpaint_steps = max(0, min(int(inpaint_steps), int(steps)))
    start_t = float(start_t)
    if not (0.0 < start_t <= 1.0):
        raise ValueError(f"start_t must be in (0, 1], got {start_t}")
    if initial_sample is not None:
        initial = initial_sample.to(device=q_vis.device, dtype=q_vis.dtype)
    elif initial_noise is not None:
        initial = initial_noise.to(device=q_vis.device, dtype=q_vis.dtype)
    else:
        initial = torch.randn_like(q_vis)
    sample = mask * q_vis + (1.0 - mask) * initial
    extra = None
    if extra_inputs is not None:
        extra = extra_inputs.to(device=q_vis.device, dtype=q_vis.dtype)
        if extra.ndim != 5 or extra.shape[0] != q_vis.shape[0] or extra.shape[-3:] != q_vis.shape[-3:]:
            raise ValueError(f"extra_inputs must be [B,C,D,H,W] aligned with q_vis, got {tuple(extra.shape)}")
    neg_cond = torch.zeros_like(cond)
    t_seq = np.linspace(start_t, 0, int(steps) + 1)
    t_seq = float(rescale_t) * t_seq / (1 + (float(rescale_t) - 1) * t_seq)
    ones = torch.ones_like(mask)
    for i in range(int(steps)):
        t = float(t_seq[i])
        t_prev = float(t_seq[i + 1])
        if i < inpaint_steps:
            inputs = [mask * q_vis + (1.0 - mask) * sample, mask]
            if extra is not None:
                inputs.append(extra)
            x_inp = torch.cat(inputs, dim=1)
            pred_v = inpaint_model_pred(flow, x_inp, t, cond, cfg_strength=float(cfg_strength), neg_cond=neg_cond)
            sample = sample - (t - t_prev) * pred_v
            sample = mask * q_vis + (1.0 - mask) * sample
        else:
            inputs = [sample, ones]
            if extra is not None:
                inputs.append(extra)
            x_inp = torch.cat(inputs, dim=1)
            pred_v = inpaint_model_pred(flow, x_inp, t, cond, cfg_strength=float(cfg_strength), neg_cond=neg_cond)
            sample = sample - (t - t_prev) * pred_v
    return sample.float()


def load_strict_checkpoint(flow: nn.Module, checkpoint: str | Path | None) -> None:
    if not checkpoint:
        return
    if str(checkpoint).strip().lower() in {"none", "null", "zero", "zeroshot", "base", "stock"}:
        print(f"[points_to_3d_strict] skip checkpoint load: {checkpoint}", flush=True)
        return
    state = torch.load(str(checkpoint), map_location="cpu")
    state = state.get("state_dict", state)
    flow_state = {k.replace("flow.", "", 1): v for k, v in state.items() if k.startswith("flow.")}
    if not flow_state:
        flow_state = state
    current = flow.state_dict()
    compatible = {}
    expanded = {}
    skipped = []
    unexpected_keys = []

    def candidate_load_keys(key: str) -> list[str]:
        candidates = [key]
        parts = key.split(".", 2)
        if len(parts) == 3 and parts[0] == "blocks" and parts[1].isdigit():
            candidates.append(f"blocks.{parts[1]}.block.{parts[2]}")
        out: list[str] = []
        for cand in candidates:
            out.append(cand)
            if cand.endswith(".weight"):
                out.append(f"{cand[:-len('.weight')]}.base.weight")
            if cand.endswith(".bias"):
                out.append(f"{cand[:-len('.bias')]}.base.bias")
        dedup: list[str] = []
        for cand in out:
            if cand not in dedup:
                dedup.append(cand)
        return dedup

    for key, value in flow_state.items():
        load_key = None
        for candidate in candidate_load_keys(key):
            if candidate in current:
                load_key = candidate
                break
        if load_key is None:
            unexpected_keys.append(key)
            continue
        target = current[load_key]
        if tuple(target.shape) == tuple(value.shape):
            compatible[load_key] = value
            continue
        if value.ndim == target.ndim:
            source_shape = tuple(value.shape)
            target_shape = tuple(target.shape)
            if all(t >= s for t, s in zip(target_shape, source_shape)) and any(t > s for t, s in zip(target_shape, source_shape)):
                expanded_value = torch.zeros_like(target)
                slices = tuple(slice(0, s) for s in source_shape)
                expanded_value[slices].copy_(value.to(device=target.device, dtype=target.dtype))
                compatible[load_key] = expanded_value
                expanded[load_key] = {"source_shape": source_shape, "target_shape": target_shape, "source_key": key}
                continue
        skipped.append(key)
    missing, unexpected = flow.load_state_dict(compatible, strict=False)
    unexpected = list(unexpected) + unexpected_keys
    print(
        f"[points_to_3d_strict] loaded {checkpoint}: "
        f"compatible={len(compatible) - len(expanded)} expanded_shape={len(expanded)} "
        f"skipped_shape={len(skipped)} missing={len(missing)} unexpected={len(unexpected)}",
        flush=True,
    )
    for key, meta in sorted(expanded.items()):
        print(
            f"[points_to_3d_strict] expanded flow.{key}: "
            f"{meta['source_shape']} -> {meta['target_shape']} with zero-init extra channels",
            flush=True,
        )


def load_condition_adapter_checkpoint(adapter: nn.Module | None, checkpoint: str | Path | None) -> None:
    if adapter is None or not checkpoint:
        return
    if str(checkpoint).strip().lower() in {"none", "null", "zero", "zeroshot", "base", "stock"}:
        return
    state = torch.load(str(checkpoint), map_location="cpu")
    state = state.get("state_dict", state)
    adapter_state = {}
    for key, value in state.items():
        if key.startswith("condition_adapter."):
            adapter_state[key.replace("condition_adapter.", "", 1)] = value
        elif key.startswith("adapter."):
            adapter_state[key.replace("adapter.", "", 1)] = value
    if not adapter_state:
        print(f"[points_to_3d_strict] no condition_adapter state in {checkpoint}; using initialized adapter", flush=True)
        return
    current = adapter.state_dict()
    compatible = {}
    expanded = {}
    skipped = []
    for key, value in adapter_state.items():
        if key in current and tuple(current[key].shape) == tuple(value.shape):
            compatible[key] = value
        elif key in current and value.ndim == current[key].ndim:
            target = current[key]
            source_shape = tuple(value.shape)
            target_shape = tuple(target.shape)
            if all(t >= s for t, s in zip(target_shape, source_shape)) and any(t > s for t, s in zip(target_shape, source_shape)):
                expanded_value = torch.zeros_like(target)
                slices = tuple(slice(0, s) for s in source_shape)
                expanded_value[slices].copy_(value.to(device=target.device, dtype=target.dtype))
                compatible[key] = expanded_value
                expanded[key] = {"source_shape": source_shape, "target_shape": target_shape}
            else:
                skipped.append(key)
        else:
            skipped.append(key)
    missing, unexpected = adapter.load_state_dict(compatible, strict=False)
    print(
        f"[points_to_3d_strict] loaded condition_adapter {checkpoint}: "
        f"compatible={len(compatible) - len(expanded)} expanded_shape={len(expanded)} "
        f"skipped_shape={len(skipped)} missing={len(missing)} unexpected={len(unexpected)}",
        flush=True,
    )
    for key, meta in sorted(expanded.items()):
        print(
            f"[points_to_3d_strict] expanded condition_adapter.{key}: "
            f"{meta['source_shape']} -> {meta['target_shape']} with zero-init extra channels",
            flush=True,
        )


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
