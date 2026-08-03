#!/usr/bin/env python3
"""Direct-SS/GenReCon style native sparse-structure Flow components.

This module is deliberately independent of the Direct-SLAT implementation.  It
uses the stock SS Flow as the base model, PEFT LoRA on attention projections,
and GenReCon-style dense 3D image conditioning before every SS block.
"""

from __future__ import annotations

from collections import Counter
from contextlib import nullcontext
import gc
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F
from torch import nn

from ar_ss_flow.pose_lifting import build_projection_geometry
from ar_ss_flow.shared_object_preprocessing import (
    SHARED_OBJECT_PREPROCESSING_VERSION,
    canonical_json_sha256 as preprocessing_json_sha256,
)
from pose_point_depth_mv.direct_flow import (
    flow_tokens_to_volume_xyz,
    volume_xyz_to_flow_tokens,
)
from reconvggt_ar_adapter_a.pointpose_ss_condition import lora_disabled


NATIVE_SS_GENRECON_VERSION = "pose_point_depth_mv.native_ss_genrecon.v2"
NATIVE_SS_GENRECON_TRAINING = "conditional_unconditional_flow_matching.v1"
NATIVE_SS_GENRECON_PROJECTION = "frustum_only_dino_shared_geometry_every_block.v2"
NATIVE_SS_GENRECON_CFG = "standard_sampler_cfg.v1"
NATIVE_SS_GENRECON_CALIBRATION = "pose_point_depth_mv.native_ss_calibration.v2"
NATIVE_SS_GENRECON_EVAL = "pose_point_depth_mv.native_ss_genrecon_eval.v2"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def select_dino_features(features: torch.Tensor) -> torch.Tensor:
    if features.ndim != 3 or int(features.shape[-1]) < 1024:
        raise ValueError(
            "Native SS DINO features must be [views,patches,channels>=1024]"
        )
    return features[..., -1024:]


def validate_genrecon_cache_contract(
    dataset: Any, *, training_config_hash: str | None = None
) -> dict[str, Any]:
    metadata = dict(getattr(dataset, "feature_metadata", {}))
    cache_config = dict(getattr(dataset, "config", {}))
    preprocessing = dict(cache_config.get("geometric_preprocessing", {}))
    contract = {
        "visual_feature_dim": int(getattr(dataset, "visual_feature_dim", 0)),
        "dino_feature_dim": int(metadata.get("dino_feature_dim", 0)),
        "patch_count": int(metadata.get("patch_count", 0)),
        "patch_start_idx": int(metadata.get("patch_start_idx", -1)),
        "config_hash": str(getattr(dataset, "config_hash", "")),
        "dino_channel_location": "trailing",
        "geometric_preprocessing": preprocessing,
        "geometric_preprocessing_hash": preprocessing_json_sha256(preprocessing),
    }
    patch_side = int(round(math.sqrt(contract["patch_count"])))
    if (
        contract["visual_feature_dim"] < 1024
        or contract["dino_feature_dim"] != 1024
        or patch_side * patch_side != contract["patch_count"]
        or contract["patch_start_idx"] < 0
        or not contract["config_hash"]
        or preprocessing.get("version") != SHARED_OBJECT_PREPROCESSING_VERSION
        or preprocessing.get("intrinsics_rule")
        != "K_feature=source_to_feature_affine@K_source"
    ):
        raise ValueError(f"Native SS cache feature contract mismatch={contract}")
    if (
        training_config_hash is not None
        and contract["config_hash"] != str(training_config_hash)
    ):
        raise RuntimeError(
            "Native SS evaluation cache config differs from training: "
            f"{contract['config_hash']} != {training_config_hash}"
        )
    contract["patch_side"] = patch_side
    return contract


def require_disjoint_object_uids(
    evaluation_uids: Iterable[str], training_uids: Iterable[str]
) -> None:
    evaluation = {str(value) for value in evaluation_uids}
    training = {str(value) for value in training_uids}
    if not evaluation or not training or "" in evaluation or "" in training:
        raise ValueError("Native SS train/evaluation object identities are incomplete")
    overlap = sorted(evaluation.intersection(training))
    if overlap:
        raise RuntimeError(
            f"Native SS evaluation overlaps training objects: {overlap[:8]}"
        )


def _sample_patch_maps(values: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    sampled = F.grid_sample(
        values,
        grid[:, :, None, :],
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    return sampled[..., 0]


def project_frustum_dino(
    sample: dict[str, Any],
    *,
    device: torch.device,
    view_indices: torch.Tensor | None = None,
    projection_mode: str = "correct",
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Project DINO patches to all visible 16^3 voxels without depth gating."""

    if projection_mode not in ("correct", "pose_cyclic1"):
        raise ValueError(f"unsupported projection mode={projection_mode!r}")
    visual = select_dino_features(sample["visual_patch_features"]).to(
        device=device, dtype=torch.float32
    )
    intrinsics = sample["intrinsics"].to(device=device, dtype=torch.float32)
    extrinsics = sample["extrinsics"].to(device=device, dtype=torch.float32)
    if view_indices is not None:
        indices = view_indices.to(device=device, dtype=torch.long)
        visual = visual.index_select(0, indices)
        intrinsics = intrinsics.index_select(0, indices)
        extrinsics = extrinsics.index_select(0, indices)
    views, patches, channels = map(int, visual.shape)
    if views <= 0:
        raise ValueError("Native SS projection selected no views")
    patch_side = int(round(math.sqrt(patches)))
    if patch_side * patch_side != patches:
        raise ValueError(f"DINO patch count is not square: {patches}")
    if projection_mode == "pose_cyclic1" and views > 1:
        extrinsics = torch.roll(extrinsics, shifts=1, dims=0)
    image_height, image_width = map(int, sample["predicted_depth"].shape[-2:])
    geometry = build_projection_geometry(
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        grid_transform=str(sample["grid_transform"]),
        extrinsics_type=str(sample["extrinsics_type"]),
        camera_forward_sign=float(sample["camera_forward_sign"]),
        image_height=image_height,
        image_width=image_width,
        patch_grid_side=patch_side,
        volume_side=16,
    )
    patch_maps = visual.permute(0, 2, 1).reshape(
        views, channels, patch_side, patch_side
    )
    projected = _sample_patch_maps(patch_maps, geometry["patch_grid"].float())
    projected = projected.permute(0, 2, 1).contiguous()
    valid = geometry["valid"].bool()
    projected = projected * valid[..., None].to(projected.dtype)
    if projected.shape != (views, 16**3, 1024):
        raise RuntimeError(f"unexpected projected feature shape={tuple(projected.shape)}")
    if not bool(torch.isfinite(projected).all().item()):
        raise RuntimeError("frustum DINO projection produced non-finite features")
    stats = {
        "view_count": projected.new_tensor(float(views)),
        "visible_fraction": valid.float().mean(),
        "supported_fraction": valid.any(dim=0).float().mean(),
    }
    return projected, valid, stats


class GenreconViewAggregator(nn.Module):
    """GenReCon IBRNet-style learned mean/variance view aggregation."""

    def __init__(self, channels: int = 1024) -> None:
        super().__init__()
        self.channels = int(channels)
        input_dim = 3 * self.channels
        self.feature_mlp = nn.Sequential(
            nn.Linear(input_dim, self.channels),
            nn.ReLU(inplace=True),
            nn.Linear(self.channels, self.channels),
            nn.ReLU(inplace=True),
            nn.Linear(self.channels, self.channels),
        )
        self.weight_mlp = nn.Sequential(
            nn.Linear(input_dim, self.channels),
            nn.ReLU(inplace=True),
            nn.Linear(self.channels, 1),
        )
        nn.init.zeros_(self.feature_mlp[-1].weight)
        nn.init.zeros_(self.feature_mlp[-1].bias)

    def forward(
        self, features: torch.Tensor, valid: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if features.ndim != 3:
            raise ValueError("aggregator features must be [views,voxels,channels]")
        views, voxels, channels = map(int, features.shape)
        if channels != self.channels or valid.shape != (views, voxels):
            raise ValueError("aggregator feature/mask schema mismatch")
        mask = valid[..., None]
        count = mask.sum(dim=0, keepdim=True).clamp_min(1)
        masked = features * mask
        mean = masked.sum(dim=0, keepdim=True) / count
        variance = (masked.square().sum(dim=0, keepdim=True) / count - mean.square()).clamp_min(0)
        mean_expanded = mean.expand(views, -1, -1)
        variance_expanded = variance.expand(views, -1, -1)
        inputs = torch.cat((features, mean_expanded, variance_expanded), dim=-1)
        feature_delta = self.feature_mlp(inputs)
        logits = self.weight_mlp(inputs).masked_fill(~mask, -1.0e4)
        weights = torch.softmax(logits, dim=0)
        all_invalid = ~valid.any(dim=0, keepdim=True)[..., None]
        weights = torch.where(all_invalid, torch.zeros_like(weights), weights)
        aggregated = mean[0] + (feature_delta * weights).sum(dim=0)
        supported = valid.any(dim=0)
        aggregated = aggregated * supported[:, None].to(aggregated.dtype)
        entropy = -(weights.clamp_min(1.0e-8).log() * weights).sum(dim=0)[..., 0]
        stats = {
            "supported_fraction": supported.float().mean(),
            "mean_visible_views": valid.float().sum(dim=0).mean(),
            "aggregation_entropy": (
                entropy[supported].mean() if bool(supported.any().item()) else entropy.new_zeros(())
            ),
            "condition_rms": aggregated.float().square().mean().sqrt(),
        }
        return aggregated[None], stats


class EveryBlockConditionProjection(nn.Module):
    def __init__(self, condition_channels: int, flow_channels: int, blocks: int) -> None:
        super().__init__()
        self.projections = nn.ModuleList(
            nn.Linear(int(condition_channels), int(flow_channels))
            for _ in range(int(blocks))
        )
        for projection in self.projections:
            nn.init.zeros_(projection.weight)
            nn.init.zeros_(projection.bias)

    def forward(self, index: int, condition: torch.Tensor) -> torch.Tensor:
        return self.projections[int(index)](condition)

    def exact_zero(self) -> bool:
        return all(
            int(torch.count_nonzero(parameter.detach()).item()) == 0
            for parameter in self.parameters()
        )


def _straight_through_reference(
    value: torch.Tensor, reference: torch.Tensor
) -> torch.Tensor:
    if value.shape != reference.shape:
        raise ValueError("straight-through reference shape mismatch")
    return value + (reference - value).detach()


class NativeSSGenreconFlow(nn.Module):
    """Stock SS Flow plus attention LoRA and every-block projected 3D condition."""

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
        if schema != (16, 8, 8, 1):
            raise ValueError(f"unsupported stock SS Flow schema={schema}")
        self.aggregator = GenreconViewAggregator(self.condition_channels)
        self.block_condition = EveryBlockConditionProjection(
            self.condition_channels, int(core.model_channels), len(core.blocks)
        )

    @property
    def flow_core(self) -> nn.Module:
        base = getattr(self.flow, "base_model", None)
        core = getattr(base, "model", None)
        return core if isinstance(core, nn.Module) else self.flow

    def train(self, mode: bool = True) -> "NativeSSGenreconFlow":
        super().train(mode)
        self.flow.train(mode)
        return self

    def stock_prediction(
        self, x: torch.Tensor, t: torch.Tensor, condition: Any
    ) -> torch.Tensor:
        with lora_disabled(self.flow):
            return self.flow(x, t, condition)

    def _lora_outputs_exact_zero(self) -> bool:
        lora_b = [
            parameter
            for name, parameter in self.flow.named_parameters()
            if "lora_B" in name
        ]
        return bool(lora_b) and all(
            int(torch.count_nonzero(parameter.detach()).item()) == 0
            for parameter in lora_b
        )

    def _adapted_core_forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        condition: Any,
        condition_3d: torch.Tensor | None,
    ) -> torch.Tensor:
        from trellis.modules.spatial import patchify, unpatchify

        core = self.flow_core
        h = volume_xyz_to_flow_tokens(patchify(x, core.patch_size))
        h = core.input_layer(h) + core.pos_emb[None]
        t_embedding = core.t_embedder(t)
        if core.share_mod:
            t_embedding = core.adaLN_modulation(t_embedding)
        h = h.type(core.dtype)
        t_embedding = t_embedding.type(core.dtype)
        contexts = condition if isinstance(condition, list) else [condition]
        for context_value in contexts:
            context = context_value.type(core.dtype)
            for block_index, block in enumerate(core.blocks):
                if condition_3d is not None:
                    residual = self.block_condition(block_index, condition_3d)
                    h = h + residual.to(h.dtype)
                h = block(h, t_embedding, context)
        h = h.type(x.dtype)
        h = F.layer_norm(h, h.shape[-1:])
        h = core.out_layer(h)
        h = flow_tokens_to_volume_xyz(h)
        return unpatchify(h, core.patch_size).contiguous()

    def adapted_prediction(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        condition: Any,
        sample: dict[str, Any] | None,
        *,
        view_indices: torch.Tensor | None = None,
        projection_mode: str = "correct",
        stock_velocity: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        condition_3d: torch.Tensor
        stats: dict[str, torch.Tensor]
        if sample is not None:
            projected, valid, projection_stats = project_frustum_dino(
                sample,
                device=x.device,
                view_indices=view_indices,
                projection_mode=str(projection_mode),
            )
            condition_3d, aggregation_stats = self.aggregator(projected, valid)
            stats = {**projection_stats, **aggregation_stats}
            condition_present = True
        else:
            # GenReCon represents the unconditional 3D branch with zero
            # features.  The LoRA and per-block projection (including its
            # learned bias) remain active; Stock is a separate explicit path.
            condition_3d = x.new_zeros(
                (int(x.shape[0]), 16**3, self.condition_channels),
                dtype=torch.float32,
            )
            zero = x.new_zeros((), dtype=torch.float32)
            stats = {
                "view_count": zero,
                "visible_fraction": zero,
                "supported_fraction": zero,
                "mean_visible_views": zero,
                "aggregation_entropy": zero,
                "condition_rms": zero,
            }
            condition_present = False
        prediction = self._adapted_core_forward(
            x,
            t,
            condition,
            condition_3d,
        )
        stock = (
            self.stock_prediction(x, t, condition)
            if stock_velocity is None
            else stock_velocity
        )
        if self.block_condition.exact_zero() and self._lora_outputs_exact_zero():
            prediction = _straight_through_reference(prediction, stock)
        delta = prediction.float() - stock.float()
        stats.update(
            {
                "condition_present": delta.new_tensor(float(condition_present)),
                "flow_delta_rms": delta.square().mean().sqrt(),
                "flow_delta_abs_max": delta.abs().amax(),
            }
        )
        return prediction, stats


class NativeSSStockFlow(nn.Module):
    def __init__(self, model: NativeSSGenreconFlow) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor, t: torch.Tensor, condition: Any) -> torch.Tensor:
        return self.model.stock_prediction(x, t, condition)


def _same_condition_identity(value: Any, reference: Any) -> bool:
    if value is reference:
        return True
    return bool(
        torch.is_tensor(value)
        and torch.is_tensor(reference)
        and value.shape == reference.shape
        and value.data_ptr() == reference.data_ptr()
    )


class NativeSSCalibratedCFGFlow(nn.Module):
    """Use adapted conditional and unconditional branches with standard CFG."""

    def __init__(
        self,
        model: NativeSSGenreconFlow,
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

    def forward(self, x: torch.Tensor, t: torch.Tensor, condition: Any) -> torch.Tensor:
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
            # The unconditional branch uses zero 3D features, but its learned
            # projection bias must match p_uncond training and remain active.
            projection_mode=self.projection_mode if positive else "correct",
        )
        self.delta_rms.append(float(stats["flow_delta_rms"].detach().item()))
        return prediction

    def summary(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "positive_calls": self.positive_calls,
            "negative_calls": self.negative_calls,
            "mean_flow_delta_rms": (
                sum(self.delta_rms) / len(self.delta_rms) if self.delta_rms else 0.0
            ),
            "post_cfg_cap": False,
            "condition_scale_policy": "learned_projection_only",
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
            "Native SS trainable state mismatch: "
            f"missing={sorted(expected - set(state))[:8]} "
            f"unexpected={sorted(set(state) - expected)[:8]}"
        )
    named = dict(model.named_parameters())
    with torch.no_grad():
        for name, value in state.items():
            parameter = named[name]
            if parameter.shape != value.shape:
                raise ValueError(f"Native SS checkpoint shape mismatch for {name}")
            parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))


def build_native_ss_genrecon_components(
    *,
    pretrained: str,
    lora_rank: int,
    lora_alpha: int,
    condition_channels: int,
    gradient_checkpointing: bool,
    need_decoder: bool,
    device: torch.device,
) -> tuple[Any, NativeSSGenreconFlow, nn.Module | None, dict[str, Any], dict[str, Any]]:
    from pose_point_depth_mv.train_direct_flow import install_unused_model_stubs
    from trellis.pipelines import TrellisVGGTTo3DPipeline

    install_unused_model_stubs()
    pipeline = TrellisVGGTTo3DPipeline.from_pretrained(pretrained)
    pipeline._device = device
    pipeline.low_vram = False
    base_flow = pipeline.models["sparse_structure_flow_model"].to(device).eval()
    for parameter in base_flow.parameters():
        parameter.requires_grad_(False)
    base_flow.use_checkpoint = bool(gradient_checkpointing)
    for block in base_flow.blocks:
        block.use_checkpoint = bool(gradient_checkpointing)

    from peft import LoraConfig, get_peft_model

    flow = get_peft_model(
        base_flow,
        LoraConfig(
            r=int(lora_rank),
            lora_alpha=int(lora_alpha),
            lora_dropout=0.0,
            bias="none",
            target_modules=["to_q", "to_kv", "to_out", "to_qkv"],
        ),
    )
    model = NativeSSGenreconFlow(
        flow, condition_channels=int(condition_channels)
    ).to(device)
    lora_modules = sorted(
        name
        for name, module in flow.named_modules()
        if hasattr(module, "lora_A") or hasattr(module, "lora_B")
    )
    block_count = len(model.flow_core.blocks)
    covered_blocks = sorted(
        {
            int(part)
            for name in lora_modules
            for position, part in enumerate(name.split("."))
            if part.isdigit()
            and position > 0
            and name.split(".")[position - 1] == "blocks"
        }
    )
    if not lora_modules or covered_blocks != list(range(block_count)):
        raise RuntimeError(
            f"Native SS LoRA coverage failed modules={len(lora_modules)} "
            f"covered={covered_blocks}"
        )
    trainable_names = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    unexpected = [
        name
        for name in trainable_names
        if "lora_" not in name
        and not name.startswith("aggregator.")
        and not name.startswith("block_condition.")
    ]
    if unexpected or not trainable_names:
        raise RuntimeError(f"Native SS trainable whitelist failed: {unexpected}")
    decoder = (
        pipeline.models["sparse_structure_decoder"].to(device).eval()
        if need_decoder
        else None
    )
    if decoder is not None:
        for parameter in decoder.parameters():
            parameter.requires_grad_(False)
    lora_parameters = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and "lora_" in name
    )
    new_parameters = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and "lora_" not in name
    )
    summary = {
        "format": NATIVE_SS_GENRECON_VERSION,
        "stage": "Direct-SS/GenReCon native 16^3 SS Flow",
        "pretrained": str(pretrained),
        "projection": NATIVE_SS_GENRECON_PROJECTION,
        "condition_channels": int(condition_channels),
        "condition_injection": "zero-init independent projection before every SS block",
        "condition_scale_policy": "learned_projection_only",
        "block_count": block_count,
        "flow_lora": {
            "rank": int(lora_rank),
            "alpha": int(lora_alpha),
            "module_count": len(lora_modules),
            "covered_blocks": covered_blocks,
            "target_counts": dict(
                sorted(Counter(name.rsplit(".", 1)[-1] for name in lora_modules).items())
            ),
            "parameter_count": int(lora_parameters),
        },
        "new_condition_parameter_count": int(new_parameters),
        "trainable_parameter_count": int(lora_parameters + new_parameters),
        "training_semantics": NATIVE_SS_GENRECON_TRAINING,
        "cfg_semantics": NATIVE_SS_GENRECON_CFG,
        "post_cfg_cap": False,
        "direct_slat_dependency": False,
        "stock_fallback": "explicit LoRA-disabled stock SS Flow wrapper",
        "trainable_whitelist": [
            "flow.*.lora_[AB].*",
            "aggregator.*",
            "block_condition.*",
        ],
        "frozen": ["stock SS Flow base", "SS decoder", "cached image encoders"],
    }
    sampler = pipeline.sparse_structure_sampler
    sampler_params = dict(pipeline.sparse_structure_sampler_params)
    del pipeline
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return sampler, model, decoder, summary, sampler_params


def validate_native_ss_genrecon_checkpoint(
    checkpoint: dict[str, Any], *, pretrained: str
) -> None:
    if checkpoint.get("format") != NATIVE_SS_GENRECON_VERSION:
        raise ValueError(f"unexpected Native SS format={checkpoint.get('format')!r}")
    summary = checkpoint.get("model_summary")
    args = checkpoint.get("args")
    if not isinstance(summary, dict) or not isinstance(args, dict):
        raise ValueError("Native SS checkpoint lacks summary/arguments")
    if str(summary.get("pretrained")) != str(pretrained):
        raise RuntimeError("Native SS checkpoint pretrained binding differs")
    required = {
        "training_semantics": NATIVE_SS_GENRECON_TRAINING,
        "cfg_semantics": NATIVE_SS_GENRECON_CFG,
        "projection": NATIVE_SS_GENRECON_PROJECTION,
        "condition_scale_policy": "learned_projection_only",
        "post_cfg_cap": False,
        "direct_slat_dependency": False,
    }
    mismatch = {
        key: (summary.get(key), value)
        for key, value in required.items()
        if summary.get(key) != value
    }
    if mismatch:
        raise ValueError(f"Native SS GenReCon protocol mismatch={mismatch}")
    forbidden = {
        "delta_rms_ratio_cap",
        "guided_delta_policy",
        "delta_bound_mode",
        "occupancy_weight",
        "raw_delta_excess_weight",
        "condition_scale",
    }
    present = sorted(forbidden.intersection(args))
    if present:
        raise ValueError(f"Native SS checkpoint contains forbidden SLAT-era fields={present}")
    if not isinstance(checkpoint.get("model_trainable_state"), dict):
        raise ValueError("Native SS checkpoint lacks trainable state")
    if not isinstance(checkpoint.get("ema_trainable_state"), dict):
        raise ValueError("Native SS checkpoint lacks EMA trainable state")
    ema = checkpoint.get("ema")
    if (
        not isinstance(ema, dict)
        or int(ema.get("updates", -1)) != int(checkpoint.get("step", -2))
        or not 0.0 < float(ema.get("target_decay", 0.0)) < 1.0
    ):
        raise ValueError("Native SS checkpoint EMA contract mismatch")


def optimizer_parameter_groups(
    model: NativeSSGenreconFlow,
    *,
    new_lr: float,
    lora_lr: float,
    new_weight_decay: float,
) -> list[dict[str, Any]]:
    lora = []
    new = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (lora if "lora_" in name else new).append(parameter)
    if not lora or not new:
        raise RuntimeError("Native SS optimizer requires both LoRA and condition parameters")
    return [
        {"params": lora, "lr": float(lora_lr), "weight_decay": 0.0, "name": "lora"},
        {
            "params": new,
            "lr": float(new_lr),
            "weight_decay": float(new_weight_decay),
            "name": "new_condition",
        },
    ]


def select_object_indices(
    rows: Iterable[dict[str, Any]], *, start: int = 0, end: int = 0
) -> list[int]:
    first: dict[str, int] = {}
    for index, row in enumerate(rows):
        first.setdefault(str(row.get("object_uid", row["uid"])), index)
    ordered = [index for _, index in sorted(first.items())]
    stop = len(ordered) if int(end) <= 0 else int(end)
    selected = ordered[int(start) : stop]
    if not selected:
        raise ValueError(f"object slice [{start}:{end}] selected no objects")
    return selected
