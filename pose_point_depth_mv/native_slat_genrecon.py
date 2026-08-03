#!/usr/bin/env python3
"""Native-SLAT v2: Native-SS-aligned GenReCon conditioning on frozen Stock SLAT.

The model keeps the pretrained ReconViaGen/TRELLIS-vggt SLAT Flow as the
explicit Stock branch.  The trainable branch adds attention LoRA and a
GenReCon-style posed multiview DINO condition at the actual active 32^3 sparse
coordinates before every native SLAT transformer block.  Conditional and
unconditional flow matching and standard sampler CFG intentionally match
``native_ss_genrecon.py``.
"""

from __future__ import annotations

from collections import Counter
from contextlib import nullcontext
import gc
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from pose_point_depth_mv.native_3d_condition import sparse_projection_geometry
from pose_point_depth_mv.native_ss_genrecon import (
    EveryBlockConditionProjection,
    GenreconViewAggregator,
    NATIVE_SS_GENRECON_CFG,
    NATIVE_SS_GENRECON_TRAINING,
    select_dino_features,
)
from reconvggt_ar_adapter_a.pointpose_ss_condition import lora_disabled


NATIVE_SLAT_GENRECON_VERSION = "pose_point_depth_mv.native_slat_genrecon.v2"
NATIVE_SLAT_GENRECON_TRAINING = NATIVE_SS_GENRECON_TRAINING
NATIVE_SLAT_GENRECON_CFG = NATIVE_SS_GENRECON_CFG
NATIVE_SLAT_GENRECON_PROJECTION = (
    "active32_frustum_only_dino_shared_geometry_every_block.v2"
)
NATIVE_SLAT_STOCK_FREEZE_VERSION = (
    "pose_point_depth_mv.native_ss_stock_slat_freeze.v2"
)
NATIVE_SLAT_BASELINE = (
    "native_ss_step2000_ema_cfg5_plus_frozen_stock_slat.v1"
)


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


def module_state_sha256(module: nn.Module) -> str:
    """Hash the complete frozen module state without assembling one large blob."""

    digest = hashlib.sha256()
    state = module.state_dict()
    for name in sorted(state):
        tensor = state[name].detach().contiguous().cpu()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
        # Flatten first so scalar BF16/FP16 buffers can also be reinterpreted
        # byte-for-byte without NumPy dtype support.
        digest.update(
            tensor.reshape(-1).view(torch.uint8).numpy().tobytes(order="C")
        )
    return digest.hexdigest()


def module_schema(module: nn.Module) -> dict[str, Any]:
    state = module.state_dict()
    schema = [
        {
            "name": name,
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
        for name, value in sorted(state.items())
    ]
    return {
        "class": f"{type(module).__module__}.{type(module).__qualname__}",
        "tensor_count": len(schema),
        "parameter_count": int(sum(value.numel() for value in state.values())),
        "schema_sha256": canonical_json_sha256(schema),
    }


def make_stock_slat_freeze(
    *,
    pretrained: str,
    flow: nn.Module,
    decoder: nn.Module,
    sampler_params: dict[str, Any],
    normalization: dict[str, Any],
) -> dict[str, Any]:
    frozen_sampler_params = dict(sampler_params)
    frozen_sampler_params.setdefault("guidance_rescale", 0.0)
    if "cfg_interval" in frozen_sampler_params:
        frozen_sampler_params["cfg_interval"] = [
            float(value) for value in frozen_sampler_params["cfg_interval"]
        ]
    frozen_normalization = {
        key: [float(value) for value in values]
        for key, values in normalization.items()
    }
    body = {
        "format": NATIVE_SLAT_STOCK_FREEZE_VERSION,
        "pretrained": str(pretrained),
        "baseline": NATIVE_SLAT_BASELINE,
        "slat_flow": {
            **module_schema(flow),
            "state_sha256": module_state_sha256(flow),
        },
        "mesh_decoder": {
            **module_schema(decoder),
            "state_sha256": module_state_sha256(decoder),
        },
        "slat_sampler_params": frozen_sampler_params,
        "slat_normalization": frozen_normalization,
        "freeze_policy": (
            "Stock SLAT Flow and Mesh decoder are inference-only; Native-SLAT v2 "
            "trains only attention LoRA, view aggregation, and every-block projections. "
            "Sampler parameters and SLAT normalization are immutable deployment state."
        ),
    }
    return {**body, "freeze_sha256": canonical_json_sha256(body)}


def load_stock_slat_freeze(path: str | Path) -> dict[str, Any]:
    freeze_path = Path(path).resolve()
    payload = json.loads(freeze_path.read_text(encoding="utf-8"))
    if payload.get("format") != NATIVE_SLAT_STOCK_FREEZE_VERSION:
        raise ValueError(f"unexpected Stock-SLAT freeze={payload.get('format')!r}")
    body = dict(payload)
    expected = str(body.pop("freeze_sha256", ""))
    if not expected or canonical_json_sha256(body) != expected:
        raise RuntimeError("Stock-SLAT freeze manifest hash mismatch")
    payload["path"] = str(freeze_path)
    payload["file_sha256"] = sha256_file(freeze_path)
    return payload


def validate_runtime_stock_slat(
    freeze: dict[str, Any],
    *,
    pretrained: str,
    flow: nn.Module,
    decoder: nn.Module | None,
    sampler_params: dict[str, Any],
    normalization: dict[str, Any],
) -> None:
    if str(freeze.get("pretrained")) != str(pretrained):
        raise RuntimeError("Stock-SLAT pretrained identity differs from freeze")
    if freeze.get("baseline") != NATIVE_SLAT_BASELINE:
        raise RuntimeError("Stock-SLAT baseline semantic differs from freeze")
    runtime_flow = {
        **module_schema(flow),
        "state_sha256": module_state_sha256(flow),
    }
    if runtime_flow != freeze.get("slat_flow"):
        raise RuntimeError("runtime Stock SLAT Flow differs from frozen manifest")
    if decoder is not None:
        runtime_decoder = {
            **module_schema(decoder),
            "state_sha256": module_state_sha256(decoder),
        }
        if runtime_decoder != freeze.get("mesh_decoder"):
            raise RuntimeError("runtime SLAT Mesh decoder differs from frozen manifest")
    runtime_sampler = dict(sampler_params)
    runtime_sampler.setdefault("guidance_rescale", 0.0)
    if "cfg_interval" in runtime_sampler:
        runtime_sampler["cfg_interval"] = [
            float(value) for value in runtime_sampler["cfg_interval"]
        ]
    if runtime_sampler != freeze.get("slat_sampler_params"):
        raise RuntimeError("runtime Stock SLAT sampler differs from frozen manifest")
    runtime_normalization = {
        key: [float(value) for value in values]
        for key, values in normalization.items()
    }
    if runtime_normalization != freeze.get("slat_normalization"):
        raise RuntimeError("runtime Stock SLAT normalization differs from frozen manifest")


def _sample_patch_maps(values: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    sampled = F.grid_sample(
        values,
        grid[:, :, None, :],
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    return sampled[..., 0]


def project_sparse_frustum_dino(
    sample: dict[str, Any],
    coords: torch.Tensor,
    *,
    device: torch.device,
    view_indices: torch.Tensor | None = None,
    projection_mode: str = "correct",
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Project trailing DINO tokens onto the actual active 32^3 stem coords."""

    if projection_mode not in ("correct", "pose_cyclic1"):
        raise ValueError(f"unsupported Native-SLAT projection mode={projection_mode!r}")
    if coords.ndim != 2 or int(coords.shape[1]) != 4:
        raise ValueError("Native-SLAT active coords must be [N,4]")
    if coords.numel() and not bool((coords[:, 0] == 0).all().item()):
        raise ValueError("Native-SLAT v2 requires sparse batch size one per rank")
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
    if views <= 0 or channels != 1024:
        raise ValueError("Native-SLAT projection selected invalid DINO views/channels")
    patch_side = int(round(patches**0.5))
    if patch_side * patch_side != patches:
        raise ValueError(f"DINO patch count is not square: {patches}")
    if projection_mode == "pose_cyclic1" and views > 1:
        extrinsics = torch.roll(extrinsics, shifts=1, dims=0)
    image_height, image_width = map(int, sample["predicted_depth"].shape[-2:])
    geometry = sparse_projection_geometry(
        coords=coords,
        resolution=32,
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        grid_transform=str(sample["grid_transform"]),
        extrinsics_type=str(sample["extrinsics_type"]),
        camera_forward_sign=float(sample["camera_forward_sign"]),
        image_height=image_height,
        image_width=image_width,
        patch_grid_side=patch_side,
    )
    maps = visual.permute(0, 2, 1).reshape(
        views, channels, patch_side, patch_side
    )
    projected = _sample_patch_maps(maps, geometry["patch_grid"].float())
    projected = projected.permute(0, 2, 1).contiguous()
    valid = geometry["valid"].bool()
    projected = projected * valid[..., None].to(projected.dtype)
    expected = (views, int(coords.shape[0]), 1024)
    if tuple(projected.shape) != expected:
        raise RuntimeError(
            f"Native-SLAT projected shape={tuple(projected.shape)} expected={expected}"
        )
    if not bool(torch.isfinite(projected).all().item()):
        raise RuntimeError("Native-SLAT projection produced non-finite features")
    stats = {
        "view_count": projected.new_tensor(float(views)),
        "visible_fraction": valid.float().mean(),
        "supported_fraction": valid.any(dim=0).float().mean(),
        "active_point_count": projected.new_tensor(float(coords.shape[0])),
    }
    return projected, valid, stats


def _slat_flow_core(flow: nn.Module) -> nn.Module:
    base = getattr(flow, "base_model", None)
    core = getattr(base, "model", None)
    return core if isinstance(core, nn.Module) else flow


def _straight_through_sparse(value: Any, reference: Any) -> Any:
    if not torch.equal(value.coords, reference.coords):
        raise ValueError("Native-SLAT straight-through coordinates differ")
    feats = value.feats + (reference.feats - value.feats).detach()
    return value.replace(feats)


def _same_condition_identity(value: Any, reference: Any) -> bool:
    if value is reference:
        return True
    if isinstance(value, list) and isinstance(reference, list) and len(value) == len(reference):
        return all(_same_condition_identity(left, right) for left, right in zip(value, reference))
    return bool(
        torch.is_tensor(value)
        and torch.is_tensor(reference)
        and value.shape == reference.shape
        and value.data_ptr() == reference.data_ptr()
    )


class NativeSLatGenreconFlow(nn.Module):
    """Frozen Stock SLAT plus LoRA and posed 32^3 every-block conditioning."""

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
            raise ValueError(f"unsupported Stock SLAT schema={schema}")
        self.aggregator = GenreconViewAggregator(self.condition_channels)
        self.block_condition = EveryBlockConditionProjection(
            self.condition_channels, int(core.model_channels), len(core.blocks)
        )

    @property
    def flow_core(self) -> nn.Module:
        return _slat_flow_core(self.flow)

    def stock_prediction(self, x: Any, t: torch.Tensor, condition: Any) -> Any:
        with lora_disabled(self.flow):
            return self.flow(x, t, condition)

    def _lora_outputs_exact_zero(self) -> bool:
        values = [
            parameter
            for name, parameter in self.flow.named_parameters()
            if "lora_B" in name
        ]
        return bool(values) and all(
            int(torch.count_nonzero(value.detach()).item()) == 0 for value in values
        )

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
            raise ValueError("Native-SLAT active condition/feature counts differ")
        for block_index, block in enumerate(core.blocks):
            residual = self.block_condition(block_index, condition_3d)
            h = block(h.replace(h.feats + residual.to(h.dtype)), t_embedding, condition)
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
        if sample is not None:
            # Query coords are only available after the native downsampling stem.
            with torch.no_grad():
                probe = self.flow_core.input_layer(x).type(self.flow_core.dtype)
                probe_t = self.flow_core.t_embedder(t)
                if self.flow_core.share_mod:
                    probe_t = self.flow_core.adaLN_modulation(probe_t)
                probe_t = probe_t.type(self.flow_core.dtype)
                for block in self.flow_core.input_blocks:
                    probe = block(probe, probe_t)
                active_coords = probe.coords
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
            # Infer the stem point count without retaining its graph. Learned
            # projection bias and LoRA remain active, exactly as Native SS v2.
            with torch.no_grad():
                probe = self.flow_core.input_layer(x).type(self.flow_core.dtype)
                probe_t = self.flow_core.t_embedder(t)
                if self.flow_core.share_mod:
                    probe_t = self.flow_core.adaLN_modulation(probe_t)
                probe_t = probe_t.type(self.flow_core.dtype)
                for block in self.flow_core.input_blocks:
                    probe = block(probe, probe_t)
                point_count = int(probe.feats.shape[0])
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
            raise RuntimeError("Native-SLAT adaptation changed sparse coordinates")
        if self.block_condition.exact_zero() and self._lora_outputs_exact_zero():
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


class NativeSLatStockFlow(nn.Module):
    """Explicit LoRA-disabled Stock SLAT baseline."""

    def __init__(self, model: NativeSLatGenreconFlow) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: Any, t: torch.Tensor, condition: Any) -> Any:
        return self.model.stock_prediction(x, t, condition)


class NativeSLatCalibratedCFGFlow(nn.Module):
    """Adapt both positive and unconditional branches under standard CFG."""

    def __init__(
        self,
        model: NativeSLatGenreconFlow,
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
            "Native-SLAT v2 state mismatch: "
            f"missing={sorted(expected - set(state))[:8]} "
            f"unexpected={sorted(set(state) - expected)[:8]}"
        )
    named = dict(model.named_parameters())
    with torch.no_grad():
        for name, value in state.items():
            parameter = named[name]
            if parameter.shape != value.shape:
                raise ValueError(f"Native-SLAT v2 checkpoint shape mismatch for {name}")
            parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))


def build_native_slat_genrecon_components(
    *,
    pretrained: str,
    stock_slat_freeze: dict[str, Any],
    upstream_native_ss: dict[str, Any],
    lora_rank: int,
    lora_alpha: int,
    condition_channels: int,
    gradient_checkpointing: bool,
    need_decoder: bool,
    device: torch.device,
) -> tuple[Any, NativeSLatGenreconFlow, nn.Module | None, dict[str, Any], dict[str, Any], dict[str, Any]]:
    from peft import LoraConfig, get_peft_model
    from pose_point_depth_mv.train_direct_flow import install_unused_model_stubs
    from trellis.pipelines import TrellisVGGTTo3DPipeline

    install_unused_model_stubs()
    pipeline = TrellisVGGTTo3DPipeline.from_pretrained(pretrained)
    pipeline._device = device
    pipeline.low_vram = False
    # Validate the immutable pretrained tensors on CPU before moving only the
    # modules needed by this process to CUDA. Training must not retain the
    # unused mesh decoder or the rest of the pretrained pipeline.
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
    base_flow = base_flow.to(device)
    if need_decoder:
        decoder = decoder.to(device)
    for parameter in base_flow.parameters():
        parameter.requires_grad_(False)
    for parameter in decoder.parameters():
        parameter.requires_grad_(False)
    base_flow.use_checkpoint = bool(gradient_checkpointing)
    for block in base_flow.blocks:
        block.use_checkpoint = bool(gradient_checkpointing)
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
    model = NativeSLatGenreconFlow(
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
            f"Native-SLAT LoRA coverage failed modules={len(lora_modules)} "
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
        raise RuntimeError(f"Native-SLAT trainable whitelist failed: {unexpected}")
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
        "format": NATIVE_SLAT_GENRECON_VERSION,
        "stage": "Native-SLAT v2 active-32^3 GenReCon Flow",
        "pretrained": str(pretrained),
        "baseline": NATIVE_SLAT_BASELINE,
        "projection": NATIVE_SLAT_GENRECON_PROJECTION,
        "query_coordinates": "actual active 32^3 native SLAT stem coordinates",
        "condition_channels": int(condition_channels),
        "condition_injection": "zero-init independent projection before every SLAT block",
        "condition_scale_policy": "learned_projection_only",
        "training_semantics": NATIVE_SLAT_GENRECON_TRAINING,
        "cfg_semantics": NATIVE_SLAT_GENRECON_CFG,
        "post_cfg_cap": False,
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
        "stock_slat_freeze": {
            key: stock_slat_freeze[key]
            for key in ("path", "file_sha256", "freeze_sha256", "baseline")
        },
        "upstream_native_ss": dict(upstream_native_ss),
        "trainable_whitelist": [
            "flow.*.lora_[AB].*",
            "aggregator.*",
            "block_condition.*",
        ],
        "frozen": [
            "Native SS deployment bound by upstream report",
            "Stock SLAT Flow base",
            "Stock SLAT Mesh decoder",
            "cached image encoders/native SLAT context",
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


def validate_native_slat_genrecon_checkpoint(
    checkpoint: dict[str, Any],
    *,
    pretrained: str,
    stock_slat_freeze: dict[str, Any],
    upstream_native_ss: dict[str, Any],
) -> None:
    if checkpoint.get("format") != NATIVE_SLAT_GENRECON_VERSION:
        raise ValueError(f"unexpected Native-SLAT format={checkpoint.get('format')!r}")
    summary = checkpoint.get("model_summary")
    args = checkpoint.get("args")
    if not isinstance(summary, dict) or not isinstance(args, dict):
        raise ValueError("Native-SLAT checkpoint lacks summary/arguments")
    required = {
        "pretrained": str(pretrained),
        "baseline": NATIVE_SLAT_BASELINE,
        "projection": NATIVE_SLAT_GENRECON_PROJECTION,
        "training_semantics": NATIVE_SLAT_GENRECON_TRAINING,
        "cfg_semantics": NATIVE_SLAT_GENRECON_CFG,
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
        raise ValueError(f"Native-SLAT v2 protocol mismatch={mismatch}")
    if summary.get("stock_slat_freeze", {}).get("freeze_sha256") != stock_slat_freeze.get(
        "freeze_sha256"
    ):
        raise RuntimeError("Native-SLAT checkpoint Stock freeze differs")
    if summary.get("upstream_native_ss") != upstream_native_ss:
        raise RuntimeError("Native-SLAT checkpoint upstream Native SS differs")
    forbidden = {
        "condition_scale",
        "delta_rms_ratio_cap",
        "delta_norm_weight",
        "wrong_support_probability",
        "rollout_probability",
    }
    present = sorted(forbidden.intersection(args))
    if present:
        raise ValueError(f"Native-SLAT v2 contains forbidden Direct-SLAT fields={present}")
    if not isinstance(checkpoint.get("model_trainable_state"), dict):
        raise ValueError("Native-SLAT checkpoint lacks trainable state")
    if not isinstance(checkpoint.get("ema_trainable_state"), dict):
        raise ValueError("Native-SLAT checkpoint lacks EMA state")
    ema = checkpoint.get("ema")
    if (
        not isinstance(ema, dict)
        or int(ema.get("updates", -1)) != int(checkpoint.get("step", -2))
        or not 0.0 < float(ema.get("target_decay", 0.0)) < 1.0
    ):
        raise ValueError("Native-SLAT checkpoint EMA contract mismatch")


def optimizer_parameter_groups(
    model: NativeSLatGenreconFlow,
    *,
    new_lr: float,
    lora_lr: float,
    new_weight_decay: float,
) -> list[dict[str, Any]]:
    lora = []
    new = []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            (lora if "lora_" in name else new).append(parameter)
    if not lora or not new:
        raise RuntimeError("Native-SLAT optimizer requires LoRA and condition parameters")
    return [
        {"params": lora, "lr": float(lora_lr), "weight_decay": 0.0, "name": "lora"},
        {
            "params": new,
            "lr": float(new_lr),
            "weight_decay": float(new_weight_decay),
            "name": "new_condition",
        },
    ]
