#!/usr/bin/env python3
"""Native-SLAT v3: pose-aware learned per-view fusion on frozen Stock SLAT.

The model keeps the pretrained ReconViaGen/TRELLIS-vggt SLAT Flow as the
explicit Stock branch.  The trainable branch adds attention LoRA and a
GenReCon-style posed multiview DINO condition at the actual active 32^3 sparse
coordinates before every native SLAT transformer block.  In addition, every
SLAT cross-attention block replaces the released code's fixed arithmetic view
mean with a learned per-point weighting of the per-view cross-attention
results.  A zero-initialized transition gate starts exactly at the frozen
Stock mean and learns to combine paper-style cross-attention scores with posed
DINO visibility evidence.  Conditional/unconditional flow matching and
standard sampler CFG intentionally match ``native_ss_genrecon.py``.
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


NATIVE_SLAT_GENRECON_VERSION = "pose_point_depth_mv.native_slat_genrecon.v3"
NATIVE_SLAT_GENRECON_TRAINING = NATIVE_SS_GENRECON_TRAINING
NATIVE_SLAT_GENRECON_CFG = NATIVE_SS_GENRECON_CFG
NATIVE_SLAT_GENRECON_PROJECTION = (
    "active32_frustum_only_dino_shared_geometry_every_block.v3"
)
NATIVE_SLAT_CONTEXT_FUSION = (
    "per_block_cross_result_mlp_plus_pose_dino_zero_init_stock_mean.v1"
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
            "Stock SLAT Flow and Mesh decoder are inference-only; Native-SLAT v3 "
            "trains attention LoRA, view aggregation, per-block learned view fusion, "
            "and every-block projections. "
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


def _lifting_tensor_leaves(
    value: Any,
    *,
    path: str = "lifting_sample",
) -> list[tuple[str, torch.Tensor]]:
    """Return every tensor leaf without copying or inspecting tensor contents."""

    if torch.is_tensor(value):
        return [(path, value)]
    if isinstance(value, dict):
        result: list[tuple[str, torch.Tensor]] = []
        for key, child in value.items():
            result.extend(
                _lifting_tensor_leaves(child, path=f"{path}.{key}")
            )
        return result
    if isinstance(value, (list, tuple)):
        result = []
        for index, child in enumerate(value):
            result.extend(
                _lifting_tensor_leaves(child, path=f"{path}[{index}]")
            )
        return result
    return []


def validate_strict_cpu_lifting_sample(sample: dict[str, Any]) -> dict[str, int]:
    """Enforce the A72 CPU-select-before-H2D projection input contract.

    The strict performance runtime intentionally keeps the complete lifting
    payload on host memory.  Only selected DINO/K/T views are transferred by
    :func:`project_sparse_frustum_dino`.  Checking every tensor leaf also
    catches accidental DDP migration of large fields that this projection does
    not consume numerically (depth/confidence/masks/prior/stock condition).
    """

    if not isinstance(sample, dict):
        raise TypeError("Native-SLAT lifting sample must be a dictionary")
    required = (
        "visual_patch_features",
        "intrinsics",
        "extrinsics",
        "predicted_depth",
    )
    missing = [name for name in required if not torch.is_tensor(sample.get(name))]
    if missing:
        raise ValueError(
            "Native-SLAT lifting sample lacks required CPU tensors: "
            + ", ".join(missing)
        )
    leaves = _lifting_tensor_leaves(sample)
    if not leaves:
        raise ValueError("Native-SLAT lifting sample contains no tensor leaves")
    non_cpu = [
        f"{path}={tensor.device}"
        for path, tensor in leaves
        if tensor.device.type != "cpu"
    ]
    if non_cpu:
        raise RuntimeError(
            "A72 strict lifting CPU-selection contract violated before view "
            "selection: " + ", ".join(non_cpu)
        )
    return {
        "tensor_count": len(leaves),
        "tensor_bytes": int(
            sum(tensor.numel() * tensor.element_size() for _, tensor in leaves)
        ),
    }


def select_sparse_frustum_inputs_cpu(
    sample: dict[str, Any],
    view_indices: torch.Tensor | None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    tuple[int, int],
    dict[str, int],
]:
    """Select projection inputs on CPU before any large host-to-device copy."""

    inventory = validate_strict_cpu_lifting_sample(sample)
    visual_cpu = select_dino_features(sample["visual_patch_features"])
    intrinsics_cpu = sample["intrinsics"]
    extrinsics_cpu = sample["extrinsics"]
    if view_indices is not None:
        if view_indices.ndim != 1:
            raise ValueError("Native-SLAT view indices must be one-dimensional")
        # The indices may originate from the CUDA RNG stream.  This is the only
        # device transfer needed before selecting the much larger CPU tensors.
        indices_cpu = view_indices.detach().to(device="cpu", dtype=torch.long)
        visual_cpu = visual_cpu.index_select(0, indices_cpu)
        intrinsics_cpu = intrinsics_cpu.index_select(0, indices_cpu)
        extrinsics_cpu = extrinsics_cpu.index_select(0, indices_cpu)
    image_shape = tuple(map(int, sample["predicted_depth"].shape[-2:]))
    return (
        visual_cpu,
        intrinsics_cpu,
        extrinsics_cpu,
        image_shape,
        inventory,
    )


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
        raise ValueError("Native-SLAT v3 requires sparse batch size one per rank")
    (
        visual_cpu,
        intrinsics_cpu,
        extrinsics_cpu,
        (image_height, image_width),
        _,
    ) = select_sparse_frustum_inputs_cpu(sample, view_indices)
    visual = visual_cpu.to(
        device=device, dtype=torch.float32, non_blocking=True
    )
    intrinsics = intrinsics_cpu.to(
        device=device, dtype=torch.float32, non_blocking=True
    )
    extrinsics = extrinsics_cpu.to(
        device=device, dtype=torch.float32, non_blocking=True
    )
    views, patches, channels = map(int, visual.shape)
    if views <= 0 or channels != 1024:
        raise ValueError("Native-SLAT projection selected invalid DINO views/channels")
    patch_side = int(round(patches**0.5))
    if patch_side * patch_side != patches:
        raise ValueError(f"DINO patch count is not square: {patches}")
    if projection_mode == "pose_cyclic1" and views > 1:
        extrinsics = torch.roll(extrinsics, shifts=1, dims=0)
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
    # ``value + detach(reference - value)`` is mathematically identical but
    # incurs a second rounded add and can miss bit-exact Stock by ~1e-10.
    # Subtracting the tensor from its own detached view is exactly zero in the
    # forward pass while retaining a unit gradient to the adapted branch.
    feats = reference.feats.detach() + (value.feats - value.feats.detach())
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


class NativeSLatCrossAttentionViewFusion(nn.Module):
    """Per-block view weighting with an exact zero-gate Stock starting point.

    The released ReconViaGen block adds ``CrossAttn(view) / view_count`` for
    every view.  Here each block predicts a pointwise score from its own
    cross-attention result, adds the posed-DINO geometry logit, and normalizes
    across views.  The effective weights interpolate from the exact uniform
    Stock weights through a straight-through-clamped, zero-initialized gate.
    """

    def __init__(
        self,
        channels: int,
        blocks: int,
        *,
        hidden_dim: int = 64,
        geometry_logit_scale_init: float = 1.0,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.block_count = int(blocks)
        self.hidden_dim = int(hidden_dim)
        if self.channels <= 0 or self.block_count <= 0 or self.hidden_dim <= 0:
            raise ValueError("Native-SLAT view-fusion dimensions must be positive")
        if not 0.0 <= float(geometry_logit_scale_init) <= 4.0:
            raise ValueError("geometry_logit_scale_init must be in [0,4]")
        self.cross_scorers = nn.ModuleList(
            nn.Sequential(
                nn.LayerNorm(self.channels, elementwise_affine=False),
                nn.Linear(self.channels, self.hidden_dim),
                nn.SiLU(),
                nn.Linear(self.hidden_dim, 1),
            )
            for _ in range(self.block_count)
        )
        # Cross-result scores initially contribute zero.  The non-uniform
        # posed-DINO logits provide the first useful direction for opening the
        # transition gate; after it opens, these MLPs receive live gradients.
        for scorer in self.cross_scorers:
            nn.init.zeros_(scorer[-1].weight)
            nn.init.zeros_(scorer[-1].bias)
        self.transition_gate_raw = nn.Parameter(torch.zeros(self.block_count))
        self.geometry_logit_scale_raw = nn.Parameter(
            torch.full((self.block_count,), float(geometry_logit_scale_init))
        )

    @staticmethod
    def _straight_through_clamp(
        value: torch.Tensor, minimum: float, maximum: float
    ) -> torch.Tensor:
        clipped = value.clamp(float(minimum), float(maximum))
        return value + (clipped - value).detach()

    def transition_gate(self, index: int) -> torch.Tensor:
        return self._straight_through_clamp(
            self.transition_gate_raw[int(index)], 0.0, 1.0
        )

    def geometry_logit_scale(self, index: int) -> torch.Tensor:
        return self._straight_through_clamp(
            self.geometry_logit_scale_raw[int(index)], 0.0, 4.0
        )

    def score_cross_result(
        self, index: int, cross_features: torch.Tensor
    ) -> torch.Tensor:
        if cross_features.ndim != 2 or int(cross_features.shape[1]) != self.channels:
            raise ValueError(
                "Native-SLAT cross result must be [points,model_channels]"
            )
        return self.cross_scorers[int(index)](cross_features.float())[:, 0]

    def effective_weights(
        self,
        index: int,
        cross_logits: torch.Tensor,
        *,
        geometry_logits: torch.Tensor | None,
        valid: torch.Tensor | None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if cross_logits.ndim != 2:
            raise ValueError("Native-SLAT cross logits must be [views,points]")
        views, points = map(int, cross_logits.shape)
        if views <= 0 or points <= 0:
            raise ValueError("Native-SLAT view fusion received an empty axis")
        if geometry_logits is None:
            geometry_logits = torch.zeros_like(cross_logits)
        if valid is None:
            valid = torch.ones_like(cross_logits, dtype=torch.bool)
        if geometry_logits.shape != cross_logits.shape or valid.shape != cross_logits.shape:
            raise ValueError("Native-SLAT cross/geometry view schemas differ")
        valid = valid.bool()
        geometry_logits = geometry_logits.to(
            device=cross_logits.device, dtype=torch.float32
        )
        combined = cross_logits.float() + self.geometry_logit_scale(index) * geometry_logits
        masked = combined.masked_fill(~valid, -1.0e4)
        target = torch.softmax(masked, dim=0)
        uniform = torch.full_like(target, 1.0 / float(views))
        all_invalid = ~valid.any(dim=0, keepdim=True)
        target = torch.where(all_invalid, uniform, target)
        gate = self.transition_gate(index)
        effective = uniform + gate * (target - uniform)
        entropy = -(target.clamp_min(1.0e-8).log() * target).sum(dim=0)
        stats = {
            "fusion_gate": gate.float(),
            "geometry_logit_scale": self.geometry_logit_scale(index).float(),
            "target_view_weight_entropy": entropy.mean(),
            "target_view_weight_deviation": (target - uniform).abs().mean(),
            "effective_view_weight_deviation": (effective - uniform).abs().mean(),
            "fusion_valid_fraction": valid.float().mean(),
        }
        return effective, stats

    def exact_stock_mean(self) -> bool:
        return int(torch.count_nonzero(self.transition_gate_raw.detach()).item()) == 0


class NativeSLatGenreconFlow(nn.Module):
    """Frozen Stock SLAT plus LoRA, posed 3D condition, and learned view fusion."""

    def __init__(
        self,
        flow: nn.Module,
        *,
        condition_channels: int = 1024,
        view_fusion_hidden_dim: int = 64,
        geometry_logit_scale_init: float = 1.0,
    ) -> None:
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
        self.view_fusion = NativeSLatCrossAttentionViewFusion(
            int(core.model_channels),
            len(core.blocks),
            hidden_dim=int(view_fusion_hidden_dim),
            geometry_logit_scale_init=float(geometry_logit_scale_init),
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

    def _view_fused_block_body(
        self,
        block_index: int,
        block: nn.Module,
        x: Any,
        mod: torch.Tensor,
        contexts: list[torch.Tensor],
        geometry_logits: torch.Tensor,
        geometry_valid: torch.Tensor,
    ) -> tuple[Any, dict[str, torch.Tensor]]:
        if block.share_mod:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(
                6, dim=1
            )
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                block.adaLN_modulation(mod).chunk(6, dim=1)
            )
        h = x.replace(block.norm1(x.feats))
        h = h * (1 + scale_msa) + shift_msa
        h = block.self_attn(h)
        h = h * gate_msa
        x = x + h
        cross_query = x.replace(block.norm2(x.feats))

        # Fuse in one attention pass without materializing a [V,N,C] tensor.
        # A streaming log-sum-exp keeps the learned softmax numerator and
        # denominator, while a second accumulator preserves the uniform Stock
        # mean.  The final transition is therefore exactly
        # uniform + gate * (learned - uniform), but does not double the costly
        # cross-attention compute or retain every view's full feature tensor.
        views = len(contexts)
        uniform_fused = torch.zeros_like(x.feats)
        target_numerator = torch.zeros_like(x.feats)
        target_denominator = x.feats.new_zeros(
            (int(x.feats.shape[0]),), dtype=torch.float32
        )
        running_max = x.feats.new_full(
            (int(x.feats.shape[0]),), -1.0e4, dtype=torch.float32
        )
        has_valid = torch.zeros_like(running_max, dtype=torch.bool)
        cross_scores: list[torch.Tensor] = []
        geometry_scale = self.view_fusion.geometry_logit_scale(block_index)
        for view_index, context in enumerate(contexts):
            output = block.cross_attn(cross_query, context)
            if not torch.equal(output.coords, x.coords):
                raise RuntimeError("Native-SLAT per-view cross-attention changed coords")
            cross_score = self.view_fusion.score_cross_result(
                block_index, output.feats
            )
            cross_scores.append(cross_score)
            combined = (
                cross_score.float()
                + geometry_scale * geometry_logits[view_index].float()
            )
            valid = geometry_valid[view_index].bool()
            new_max = torch.where(
                valid,
                torch.where(has_valid, torch.maximum(running_max, combined), combined),
                running_max,
            )
            old_delta = torch.where(
                has_valid, running_max - new_max, torch.zeros_like(new_max)
            )
            new_delta = torch.where(
                valid, combined - new_max, torch.zeros_like(new_max)
            )
            old_factor = torch.exp(old_delta) * has_valid.to(torch.float32)
            new_factor = torch.exp(new_delta) * valid.to(torch.float32)
            target_numerator = (
                target_numerator * old_factor[:, None].to(target_numerator.dtype)
                + output.feats * new_factor[:, None].to(output.feats.dtype)
            )
            target_denominator = (
                target_denominator * old_factor + new_factor
            )
            running_max = new_max
            has_valid = has_valid | valid
            uniform_fused = uniform_fused + output.feats / float(views)
        cross_logits = torch.stack(cross_scores, dim=0)
        _, fusion_stats = self.view_fusion.effective_weights(
            block_index,
            cross_logits,
            geometry_logits=geometry_logits,
            valid=geometry_valid,
        )
        target_fused = target_numerator / target_denominator.clamp_min(1.0e-8)[
            :, None
        ].to(target_numerator.dtype)
        target_fused = torch.where(
            has_valid[:, None], target_fused, uniform_fused
        )
        gate = self.view_fusion.transition_gate(block_index).to(uniform_fused.dtype)
        fused = uniform_fused + gate * (target_fused - uniform_fused)
        x = x.replace(x.feats + fused)
        h = x.replace(block.norm3(x.feats))
        h = h * (1 + scale_mlp) + shift_mlp
        h = block.mlp(h)
        h = h * gate_mlp
        return x + h, fusion_stats

    def _view_fused_block(
        self,
        block_index: int,
        block: nn.Module,
        x: Any,
        mod: torch.Tensor,
        condition: Any,
        geometry_logits: torch.Tensor | None,
        geometry_valid: torch.Tensor | None,
    ) -> tuple[Any, dict[str, torch.Tensor] | None]:
        if not isinstance(condition, list):
            return block(x, mod, condition), None
        if not condition:
            raise ValueError("Native-SLAT view fusion received no contexts")
        views = len(condition)
        points = int(x.feats.shape[0])
        if geometry_logits is None:
            geometry_logits = x.feats.new_zeros(
                (views, points), dtype=torch.float32
            )
        if geometry_valid is None:
            geometry_valid = torch.ones(
                (views, points), device=x.feats.device, dtype=torch.bool
            )
        if geometry_logits.shape != (views, points) or geometry_valid.shape != (
            views,
            points,
        ):
            raise ValueError(
                "Native-SLAT context/posed-geometry view or point counts differ"
            )

        def run(
            sparse_x: Any,
            modulation: torch.Tensor,
            logits: torch.Tensor,
            valid: torch.Tensor,
            *contexts: torch.Tensor,
        ) -> tuple[Any, ...]:
            result, stats = self._view_fused_block_body(
                block_index,
                block,
                sparse_x,
                modulation,
                list(contexts),
                logits,
                valid,
            )
            return (
                result,
                stats["fusion_gate"],
                stats["geometry_logit_scale"],
                stats["target_view_weight_entropy"],
                stats["target_view_weight_deviation"],
                stats["effective_view_weight_deviation"],
                stats["fusion_valid_fraction"],
            )

        call_args = (x, mod, geometry_logits, geometry_valid, *condition)
        values = (
            torch.utils.checkpoint.checkpoint(
                run, *call_args, use_reentrant=False
            )
            if block.use_checkpoint and torch.is_grad_enabled()
            else run(*call_args)
        )
        result = values[0]
        names = (
            "fusion_gate",
            "geometry_logit_scale",
            "target_view_weight_entropy",
            "target_view_weight_deviation",
            "effective_view_weight_deviation",
            "fusion_valid_fraction",
        )
        return result, dict(zip(names, values[1:]))

    def _adapted_core_forward(
        self,
        x: Any,
        t: torch.Tensor,
        condition: Any,
        condition_3d: torch.Tensor,
        geometry_logits: torch.Tensor | None,
        geometry_valid: torch.Tensor | None,
    ) -> tuple[Any, dict[str, torch.Tensor]]:
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
        block_fusion_stats: list[dict[str, torch.Tensor]] = []
        for block_index, block in enumerate(core.blocks):
            residual = self.block_condition(block_index, condition_3d)
            h, fusion_stats = self._view_fused_block(
                block_index,
                block,
                h.replace(h.feats + residual.to(h.dtype)),
                t_embedding,
                condition,
                geometry_logits,
                geometry_valid,
            )
            if fusion_stats is not None:
                block_fusion_stats.append(fusion_stats)
        for block, skip in zip(core.out_blocks, reversed(skips)):
            if core.use_skip_connection:
                h = block(h.replace(torch.cat((h.feats, skip), dim=1)), t_embedding)
            else:
                h = block(h, t_embedding)
        h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))
        output = core.out_layer(h.type(x.dtype))
        zero = output.feats.new_zeros((), dtype=torch.float32)
        if block_fusion_stats:
            stacked = {
                key: torch.stack([row[key].float() for row in block_fusion_stats])
                for key in block_fusion_stats[0]
            }
            summary = {
                "fusion_gate_mean": stacked["fusion_gate"].mean(),
                "fusion_gate_max": stacked["fusion_gate"].amax(),
                "geometry_logit_scale_mean": stacked["geometry_logit_scale"].mean(),
                "target_view_weight_entropy": stacked[
                    "target_view_weight_entropy"
                ].mean(),
                "target_view_weight_deviation": stacked[
                    "target_view_weight_deviation"
                ].mean(),
                "effective_view_weight_deviation": stacked[
                    "effective_view_weight_deviation"
                ].mean(),
                "fusion_valid_fraction": stacked["fusion_valid_fraction"].mean(),
            }
        else:
            summary = {
                "fusion_gate_mean": zero,
                "fusion_gate_max": zero,
                "geometry_logit_scale_mean": zero,
                "target_view_weight_entropy": zero,
                "target_view_weight_deviation": zero,
                "effective_view_weight_deviation": zero,
                "fusion_valid_fraction": zero,
            }
        return output, summary

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
            aggregated, aggregation_stats, aggregation_details = (
                self.aggregator.aggregate_with_view_details(projected, valid)
            )
            condition_3d = aggregated[0]
            stats = {**projection_stats, **aggregation_stats}
            geometry_logits = aggregation_details["view_logits"]
            geometry_valid = aggregation_details["valid"]
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
            geometry_logits = None
            geometry_valid = None
            condition_present = False
        prediction, fusion_stats = self._adapted_core_forward(
            x,
            t,
            condition,
            condition_3d,
            geometry_logits,
            geometry_valid,
        )
        stats.update(fusion_stats)
        if not torch.equal(prediction.coords, stock.coords):
            raise RuntimeError("Native-SLAT adaptation changed sparse coordinates")
        if (
            self.block_condition.exact_zero()
            and self._lora_outputs_exact_zero()
            and self.view_fusion.exact_stock_mean()
        ):
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
            "Native-SLAT v3 state mismatch: "
            f"missing={sorted(expected - set(state))[:8]} "
            f"unexpected={sorted(set(state) - expected)[:8]}"
        )
    named = dict(model.named_parameters())
    with torch.no_grad():
        for name, value in state.items():
            parameter = named[name]
            if parameter.shape != value.shape:
                raise ValueError(f"Native-SLAT v3 checkpoint shape mismatch for {name}")
            parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))


def build_native_slat_genrecon_components(
    *,
    pretrained: str,
    stock_slat_freeze: dict[str, Any],
    upstream_native_ss: dict[str, Any],
    lora_rank: int,
    lora_alpha: int,
    condition_channels: int,
    view_fusion_hidden_dim: int,
    geometry_logit_scale_init: float,
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
        flow,
        condition_channels=int(condition_channels),
        view_fusion_hidden_dim=int(view_fusion_hidden_dim),
        geometry_logit_scale_init=float(geometry_logit_scale_init),
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
        and not name.startswith("view_fusion.")
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
    view_fusion_parameters = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and name.startswith("view_fusion.")
    )
    summary = {
        "format": NATIVE_SLAT_GENRECON_VERSION,
        "stage": "Native-SLAT v3 pose-aware learned-view-fusion Flow",
        "pretrained": str(pretrained),
        "baseline": NATIVE_SLAT_BASELINE,
        "projection": NATIVE_SLAT_GENRECON_PROJECTION,
        "query_coordinates": "actual active 32^3 native SLAT stem coordinates",
        "condition_channels": int(condition_channels),
        "condition_injection": "zero-init independent projection before every SLAT block",
        "condition_scale_policy": "learned_projection_only",
        "context_view_fusion": {
            "version": NATIVE_SLAT_CONTEXT_FUSION,
            "scope": "Full branch only; frozen Stock retains released uniform mean",
            "score_source": "per-block per-view cross-attention result MLP",
            "geometry_source": "active32 posed-DINO aggregator raw view logits and validity",
            "transition": "pointwise weighted sum interpolated from exact uniform mean",
            "implementation": "single-pass streaming log-sum-exp; no VxNxC materialization",
            "gate_init": 0.0,
            "hidden_dim": int(view_fusion_hidden_dim),
            "geometry_logit_scale_init": float(geometry_logit_scale_init),
            "invalid_view_policy": (
                "learned target masks invalid; transition starts at Stock uniform; "
                "all-invalid falls back to uniform"
            ),
        },
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
        "new_condition_parameter_count": int(
            new_parameters - view_fusion_parameters
        ),
        "view_fusion_parameter_count": int(view_fusion_parameters),
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
            "view_fusion.*",
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
        raise ValueError(f"Native-SLAT v3 protocol mismatch={mismatch}")
    fusion = summary.get("context_view_fusion")
    if not isinstance(fusion, dict) or fusion.get("version") != NATIVE_SLAT_CONTEXT_FUSION:
        raise ValueError("Native-SLAT v3 checkpoint lacks learned view-fusion binding")
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
        raise ValueError(f"Native-SLAT v3 contains forbidden Direct-SLAT fields={present}")
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
