#!/usr/bin/env python3
"""Native-SLAT condition-only: posed-DINO 3D condition on frozen Stock SLat.

This module deliberately does not construct PEFT adapters, SLat attention
LoRA, a learned cross-view gate, or a replacement cross-attention path.  The
released Stock SLat Flow (including its uniform per-view cross-attention mean)
and Mesh decoder are frozen.  The only trainable path is the GenReCon-style
posed-DINO frustum projection/aggregation followed by an independent
zero-initialized projection before every native SLat transformer block.
"""

from __future__ import annotations

import gc
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from pose_point_depth_mv.native_ss_genrecon import (
    EveryBlockConditionProjection,
    GenreconViewAggregator,
    NATIVE_SS_GENRECON_CFG,
    NATIVE_SS_GENRECON_TRAINING,
)
from pose_point_depth_mv.native_slat_genrecon import (
    NATIVE_SLAT_BASELINE,
    _same_condition_identity,
    _slat_flow_core,
    _straight_through_sparse,
    canonical_json_sha256,
    load_stock_slat_freeze,
    project_sparse_frustum_dino,
    sha256_file,
    validate_runtime_stock_slat,
)


NATIVE_SLAT_CONDITION_ONLY_VERSION = (
    "pose_point_depth_mv.native_slat_condition_only.v1"
)
NATIVE_SLAT_CONDITION_ONLY_PROJECTION = (
    "active32_frustum_only_dino_genrecon_every_block_condition_only.v1"
)
NATIVE_SLAT_CONDITION_ONLY_TRAINING = NATIVE_SS_GENRECON_TRAINING
NATIVE_SLAT_CONDITION_ONLY_CFG = NATIVE_SS_GENRECON_CFG


def _forbidden_adaptation_names(model: nn.Module) -> list[str]:
    forbidden = []
    for name, module in model.named_modules():
        if hasattr(module, "lora_A") or hasattr(module, "lora_B"):
            forbidden.append(name or "<root>")
        if "view_fusion" in name or "view_gate" in name or "view_scorer" in name:
            forbidden.append(name)
    for name, _ in model.named_parameters():
        if "lora_" in name or "view_fusion" in name or "view_gate" in name:
            forbidden.append(name)
    return sorted(set(forbidden))


class NativeSLatConditionOnlyFlow(nn.Module):
    """Frozen Stock SLat plus only posed-DINO every-block 3D conditioning."""

    def __init__(self, flow: nn.Module, *, condition_channels: int = 1024) -> None:
        super().__init__()
        self.flow = flow
        self.condition_channels = int(condition_channels)
        core = self.flow_core
        schema = (
            int(core.resolution),
            int(core.in_channels),
            int(core.out_channels),
            int(core.patch_size),
        )
        if schema != (64, 8, 8, 2):
            raise ValueError(f"unsupported Stock SLat schema={schema}")
        self.aggregator = GenreconViewAggregator(self.condition_channels)
        self.block_condition = EveryBlockConditionProjection(
            self.condition_channels, int(core.model_channels), len(core.blocks)
        )

    @property
    def flow_core(self) -> nn.Module:
        return _slat_flow_core(self.flow)

    def stock_prediction(self, x: Any, t: torch.Tensor, condition: Any) -> Any:
        # ``self.flow`` is the raw frozen released model, not a PEFT wrapper.
        return self.flow(x, t, condition)

    def _stem_coords_and_count(self, x: Any, t: torch.Tensor) -> Any:
        with torch.no_grad():
            core = self.flow_core
            probe = core.input_layer(x).type(core.dtype)
            probe_t = core.t_embedder(t)
            if core.share_mod:
                probe_t = core.adaLN_modulation(probe_t)
            probe_t = probe_t.type(core.dtype)
            for block in core.input_blocks:
                probe = block(probe, probe_t)
        return probe.coords

    def _adapted_core_forward(
        self,
        x: Any,
        t: torch.Tensor,
        condition: Any,
        condition_3d: torch.Tensor,
    ) -> Any:
        core = self.flow_core
        h = core.input_layer(x).type(core.dtype)
        t_embedding = core.t_embedder(t)
        if core.share_mod:
            t_embedding = core.adaLN_modulation(t_embedding)
        t_embedding = t_embedding.type(core.dtype)
        if isinstance(condition, list):
            condition = [value.type(core.dtype) for value in condition]
        else:
            condition = condition.type(core.dtype)
        skips: list[torch.Tensor] = []
        for block in core.input_blocks:
            h = block(h, t_embedding)
            skips.append(h.feats)
        if core.pe_mode == "ape":
            h = h + core.pos_embedder(h.coords[:, 1:]).type(core.dtype)
        if int(condition_3d.shape[0]) != int(h.feats.shape[0]):
            raise ValueError("Native-SLat active condition/feature counts differ")
        for block_index, block in enumerate(core.blocks):
            residual = self.block_condition(block_index, condition_3d)
            h = block(
                h.replace(h.feats + residual.to(h.dtype)),
                t_embedding,
                condition,
            )
        for block, skip in zip(core.out_blocks, reversed(skips)):
            if core.use_skip_connection:
                h = block(h.replace(torch.cat((h.feats, skip), dim=1)), t_embedding)
            else:
                h = block(h, t_embedding)
        h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))
        return core.out_layer(h.type(x.dtype))

    def adapted_prediction(
        self,
        x: Any,
        t: torch.Tensor,
        condition: Any,
        sample: dict[str, Any] | None,
        *,
        view_indices: torch.Tensor | None = None,
        projection_mode: str = "correct",
        stock_velocity: Any | None = None,
    ) -> tuple[Any, dict[str, torch.Tensor]]:
        stock = (
            self.stock_prediction(x, t, condition)
            if stock_velocity is None
            else stock_velocity
        )
        active_coords = self._stem_coords_and_count(x, t)
        point_count = int(active_coords.shape[0])
        if sample is not None:
            projected, valid, projection_stats = project_sparse_frustum_dino(
                sample,
                active_coords,
                device=x.feats.device,
                view_indices=view_indices,
                projection_mode=projection_mode,
            )
            aggregated, aggregation_stats = self.aggregator(projected, valid)
            condition_3d = aggregated[0]
            stats = {**projection_stats, **aggregation_stats}
            condition_present = True
        else:
            condition_3d = x.feats.new_zeros(
                (point_count, self.condition_channels), dtype=torch.float32
            )
            zero = x.feats.new_zeros((), dtype=torch.float32)
            stats = {
                "view_count": zero,
                "visible_fraction": zero,
                "supported_fraction": zero,
                "active_point_count": zero.new_tensor(float(point_count)),
                "mean_visible_views": zero,
                "aggregation_entropy": zero,
                "condition_rms": zero,
            }
            condition_present = False
        prediction = self._adapted_core_forward(x, t, condition, condition_3d)
        if not torch.equal(prediction.coords, stock.coords):
            raise RuntimeError("Native-SLat condition changed sparse coordinates")
        if self.block_condition.exact_zero():
            prediction = _straight_through_sparse(prediction, stock)
        delta = prediction.feats.float() - stock.feats.float()
        stats.update(
            {
                "condition_present": delta.new_tensor(float(condition_present)),
                "flow_delta_rms": delta.square().mean().sqrt(),
                "flow_delta_abs_max": delta.abs().amax(),
            }
        )
        return prediction, stats


class NativeSLatConditionOnlyStockFlow(nn.Module):
    """The exact frozen Native-SS + released Stock-SLat baseline."""

    def __init__(self, model: NativeSLatConditionOnlyFlow) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: Any, t: torch.Tensor, condition: Any) -> Any:
        return self.model.stock_prediction(x, t, condition)


class NativeSLatConditionOnlyCFGFlow(nn.Module):
    """Apply condition-only Full to both standard CFG branches."""

    def __init__(
        self,
        model: NativeSLatConditionOnlyFlow,
        positive_condition: Any,
        sample: dict[str, Any],
        *,
        enabled: bool = True,
        projection_mode: str = "correct",
    ) -> None:
        super().__init__()
        self.model = model
        self.positive_condition = positive_condition
        self.sample = sample
        self.enabled = bool(enabled)
        self.projection_mode = str(projection_mode)
        self.positive_calls = 0
        self.negative_calls = 0
        self.delta_rms: list[float] = []

    def forward(self, x: Any, t: torch.Tensor, condition: Any) -> Any:
        if not self.enabled:
            return self.model.stock_prediction(x, t, condition)
        positive = _same_condition_identity(condition, self.positive_condition)
        self.positive_calls += int(positive)
        self.negative_calls += int(not positive)
        prediction, stats = self.model.adapted_prediction(
            x,
            t,
            condition,
            self.sample if positive else None,
            projection_mode=self.projection_mode if positive else "correct",
        )
        self.delta_rms.append(float(stats["flow_delta_rms"].detach().item()))
        return prediction

    def summary(self) -> dict[str, Any]:
        return {
            "positive_calls": self.positive_calls,
            "negative_calls": self.negative_calls,
            "mean_flow_delta_rms": (
                sum(self.delta_rms) / len(self.delta_rms) if self.delta_rms else 0.0
            ),
            "post_cfg_cap": False,
            "condition_scale_policy": "learned_projection_only",
            "trainable_slat_attention": False,
            "baseline": NATIVE_SLAT_BASELINE,
        }


def trainable_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def load_trainable_state_dict(
    model: nn.Module, state: dict[str, torch.Tensor]
) -> None:
    expected = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    if set(state) != expected:
        raise ValueError(
            "condition-only state mismatch: "
            f"missing={sorted(expected - set(state))[:8]} "
            f"unexpected={sorted(set(state) - expected)[:8]}"
        )
    named = dict(model.named_parameters())
    with torch.no_grad():
        for name, value in state.items():
            parameter = named[name]
            if parameter.shape != value.shape:
                raise ValueError(f"checkpoint shape mismatch for {name}")
            parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))


def build_native_slat_condition_only_components(
    *,
    pretrained: str,
    stock_slat_freeze: dict[str, Any],
    upstream_native_ss: dict[str, Any],
    condition_channels: int,
    gradient_checkpointing: bool,
    need_decoder: bool,
    device: torch.device,
) -> tuple[
    Any,
    NativeSLatConditionOnlyFlow,
    nn.Module | None,
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    from pose_point_depth_mv.train_direct_flow import install_unused_model_stubs
    from trellis.pipelines import TrellisVGGTTo3DPipeline

    install_unused_model_stubs()
    pipeline = TrellisVGGTTo3DPipeline.from_pretrained(pretrained)
    pipeline._device = device
    pipeline.low_vram = False
    base_flow = pipeline.models["slat_flow_model"].eval()
    decoder = pipeline.models["slat_decoder_mesh"].eval()
    validate_runtime_stock_slat(
        stock_slat_freeze,
        pretrained=pretrained,
        flow=base_flow,
        decoder=decoder,
        sampler_params=dict(pipeline.slat_sampler_params),
        normalization=dict(pipeline.slat_normalization),
    )
    for parameter in base_flow.parameters():
        parameter.requires_grad_(False)
    for parameter in decoder.parameters():
        parameter.requires_grad_(False)
    base_flow.use_checkpoint = bool(gradient_checkpointing)
    for block in base_flow.blocks:
        block.use_checkpoint = bool(gradient_checkpointing)
    base_flow = base_flow.to(device)
    if need_decoder:
        decoder = decoder.to(device)
    model = NativeSLatConditionOnlyFlow(
        base_flow, condition_channels=int(condition_channels)
    ).to(device)
    forbidden = _forbidden_adaptation_names(model)
    if forbidden:
        raise RuntimeError(f"condition-only model contains forbidden adapters={forbidden[:8]}")
    trainable_names = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    unexpected = [
        name
        for name in trainable_names
        if not name.startswith("aggregator.")
        and not name.startswith("block_condition.")
    ]
    if unexpected or not trainable_names:
        raise RuntimeError(f"condition-only trainable whitelist failed={unexpected[:8]}")
    block_count = len(model.flow_core.blocks)
    if len(model.block_condition.projections) != block_count:
        raise RuntimeError("condition projection does not cover every SLat block")
    condition_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    summary = {
        "format": NATIVE_SLAT_CONDITION_ONLY_VERSION,
        "stage": "Native-SLat condition-only posed-DINO 3D Flow",
        "pretrained": str(pretrained),
        "baseline": NATIVE_SLAT_BASELINE,
        "projection": NATIVE_SLAT_CONDITION_ONLY_PROJECTION,
        "query_coordinates": "actual active 32^3 native SLat stem coordinates",
        "condition_channels": int(condition_channels),
        "condition_injection": (
            "GenReCon posed-DINO aggregation then zero-init independent projection "
            "before every frozen Stock-SLat transformer block"
        ),
        "condition_scale_policy": "learned_projection_only",
        "training_semantics": NATIVE_SLAT_CONDITION_ONLY_TRAINING,
        "cfg_semantics": NATIVE_SLAT_CONDITION_ONLY_CFG,
        "post_cfg_cap": False,
        "block_count": block_count,
        "flow_lora": {
            "present": False,
            "module_count": 0,
            "parameter_count": 0,
            "construction": "PEFT is not imported or installed",
        },
        "context_view_fusion": {
            "policy": "frozen_released_uniform_mean",
            "trainable": False,
            "implementation": (
                "unchanged Stock-SLat block cross-attention: sum(view_output / V)"
            ),
        },
        "new_condition_parameter_count": int(condition_parameters),
        "trainable_parameter_count": int(condition_parameters),
        "stock_slat_freeze": {
            key: stock_slat_freeze[key]
            for key in ("path", "file_sha256", "freeze_sha256", "baseline")
        },
        "upstream_native_ss": dict(upstream_native_ss),
        "trainable_whitelist": ["aggregator.*", "block_condition.*"],
        "frozen": [
            "Native SS deployment bound by upstream report",
            "entire released Stock SLat Flow including self/cross attention",
            "released uniform cross-view arithmetic mean",
            "Stock SLat Mesh decoder",
            "cached image encoders/native SLat context",
        ],
        "explicitly_absent": [
            "PEFT",
            "SLat attention LoRA",
            "learned Stock/GenReCon gate",
            "learned SLat cross-view scorer",
            "Direct-SLat residual bound and rollout auxiliaries",
        ],
        "direct_slat_residual_dependency": False,
    }
    sampler = pipeline.slat_sampler
    sampler_params = dict(pipeline.slat_sampler_params)
    normalization = dict(pipeline.slat_normalization)
    if not need_decoder:
        del decoder
    del pipeline
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return (
        sampler,
        model,
        decoder if need_decoder else None,
        summary,
        sampler_params,
        normalization,
    )


def validate_native_slat_condition_only_checkpoint(
    checkpoint: dict[str, Any],
    *,
    pretrained: str,
    stock_slat_freeze: dict[str, Any],
    upstream_native_ss: dict[str, Any],
) -> None:
    if checkpoint.get("format") != NATIVE_SLAT_CONDITION_ONLY_VERSION:
        raise ValueError(f"unexpected condition-only format={checkpoint.get('format')!r}")
    summary = checkpoint.get("model_summary")
    args = checkpoint.get("args")
    if not isinstance(summary, dict) or not isinstance(args, dict):
        raise ValueError("condition-only checkpoint lacks summary/arguments")
    required = {
        "pretrained": str(pretrained),
        "baseline": NATIVE_SLAT_BASELINE,
        "projection": NATIVE_SLAT_CONDITION_ONLY_PROJECTION,
        "training_semantics": NATIVE_SLAT_CONDITION_ONLY_TRAINING,
        "cfg_semantics": NATIVE_SLAT_CONDITION_ONLY_CFG,
        "condition_scale_policy": "learned_projection_only",
        "post_cfg_cap": False,
        "direct_slat_residual_dependency": False,
    }
    mismatch = {
        key: (summary.get(key), expected)
        for key, expected in required.items()
        if summary.get(key) != expected
    }
    if mismatch:
        raise ValueError(f"condition-only protocol mismatch={mismatch}")
    if summary.get("flow_lora") != {
        "present": False,
        "module_count": 0,
        "parameter_count": 0,
        "construction": "PEFT is not imported or installed",
    }:
        raise ValueError("condition-only checkpoint is not explicitly LoRA-free")
    if summary.get("context_view_fusion", {}).get("trainable") is not False:
        raise ValueError("condition-only checkpoint has trainable context fusion")
    if summary.get("stock_slat_freeze", {}).get("freeze_sha256") != stock_slat_freeze.get(
        "freeze_sha256"
    ):
        raise RuntimeError("condition-only checkpoint Stock freeze differs")
    if summary.get("upstream_native_ss") != upstream_native_ss:
        raise RuntimeError("condition-only checkpoint upstream Native SS differs")
    forbidden_args = {
        "lora_rank",
        "lora_alpha",
        "lora_lr",
        "view_fusion_hidden_dim",
        "geometry_logit_scale_init",
        "condition_scale",
        "delta_rms_ratio_cap",
        "wrong_support_probability",
        "rollout_probability",
    }
    present = sorted(forbidden_args.intersection(args))
    if present:
        raise ValueError(f"condition-only checkpoint contains forbidden fields={present}")
    for key in ("model_trainable_state", "ema_trainable_state"):
        state = checkpoint.get(key)
        if not isinstance(state, dict) or not state:
            raise ValueError(f"condition-only checkpoint lacks {key}")
        bad = [
            name
            for name in state
            if not name.startswith("aggregator.")
            and not name.startswith("block_condition.")
        ]
        if bad:
            raise ValueError(f"forbidden trainable state in {key}={bad[:8]}")
    ema = checkpoint.get("ema")
    if (
        not isinstance(ema, dict)
        or int(ema.get("updates", -1)) != int(checkpoint.get("step", -2))
        or not 0.0 < float(ema.get("target_decay", 0.0)) < 1.0
    ):
        raise ValueError("condition-only checkpoint EMA contract mismatch")


def condition_only_parameter_group(
    model: NativeSLatConditionOnlyFlow,
    *,
    lr: float,
    weight_decay: float,
) -> list[dict[str, Any]]:
    parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not parameters:
        raise RuntimeError("condition-only optimizer has no trainable parameters")
    return [
        {
            "params": parameters,
            "lr": float(lr),
            "weight_decay": float(weight_decay),
            "name": "posed_dino_3d_condition",
        }
    ]
