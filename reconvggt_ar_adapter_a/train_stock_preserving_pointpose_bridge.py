#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("SPCONV_ALGO", "native")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
os.environ.setdefault("TORCH_EXTENSIONS_DIR", "/tmp/torch_extensions")

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

TRACKER_ROOT = Path(__file__).resolve().parents[1]
RECONVIAGEN_ROOT = TRACKER_ROOT / "ReconViaGen"
VGGT_ROOT = RECONVIAGEN_ROOT / "wheels" / "vggt"
for path in (TRACKER_ROOT, RECONVIAGEN_ROOT, VGGT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from trellis.pipelines import TrellisVGGTTo3DPipeline  # noqa: E402

from reconvggt_ar_adapter_a.pointpose_ss_condition import (  # noqa: E402
    load_partial_state,
    trainable_state_dict,
)
from reconvggt_ar_adapter_a.pointpose_patch_features import (  # noqa: E402
    build_projected_patch_features,
    infer_patch_grid_side,
    make_null_projected_patch_features,
)
from reconvggt_ar_adapter_a.stock_preserving_pointpose_bridge import (  # noqa: E402
    ContentBasedPhysicalVisualBridge,
    MultiStageStockPreservingPhysicalBridge,
    PoseGuidedProjectedPatchBridge,
    StockPreservingPhysicalBridge,
    make_null_physical_grid,
)
from reconvggt_ar_adapter_a.train_pointpose_ss_lora import (  # noqa: E402
    PointPoseCacheDataset,
    collate_one,
    distributed_all_true,
    distributed_mean,
    encode_frozen_features,
    finite_tree,
    gradients_finite,
    install_unused_model_stubs,
    optimizer_state_finite,
    parameters_finite,
    rgba_images,
    sample_t,
    tensors_finite,
)


class HardNegativeMiner:
    """Deterministic nearest-statistics negatives with a different object id."""

    def __init__(self, dataset: PointPoseCacheDataset) -> None:
        rows = dataset.samples
        descriptors = np.asarray(
            [
                [
                    math.log1p(float(row.get("prior_point_count", 0))),
                    math.log1p(float(row.get("target_coord_count", 0))),
                ]
                for row in rows
            ],
            dtype=np.float64,
        )
        mean = descriptors.mean(axis=0, keepdims=True)
        std = descriptors.std(axis=0, keepdims=True)
        descriptors = (descriptors - mean) / np.maximum(std, 1.0e-6)
        self.negative_indices: list[int] = []
        for index, row in enumerate(rows):
            distances = np.square(descriptors - descriptors[index]).sum(axis=1)
            order = np.argsort(distances, kind="stable")
            negative = next(
                (
                    int(candidate)
                    for candidate in order
                    if int(candidate) != index
                    and str(rows[int(candidate)].get("object_uid", ""))
                    != str(row.get("object_uid", ""))
                ),
                None,
            )
            if negative is None:
                raise RuntimeError("hard-negative mining requires at least two distinct objects")
            self.negative_indices.append(negative)

    def __getitem__(self, index: int) -> int:
        return self.negative_indices[index]


def parse_fusion_stages(text: str, block_count: int = 4) -> tuple[int, ...]:
    values = tuple(sorted({int(item.strip()) for item in str(text).split(",") if item.strip()}))
    if not values or values[0] < 0 or values[-1] >= int(block_count):
        raise ValueError(
            f"fusion_stages must be a non-empty subset of [0,{block_count - 1}], got {text!r}"
        )
    return values


@torch.no_grad()
def paired_tensor_diagnostics(
    correct: torch.Tensor,
    shuffled: torch.Tensor,
    *,
    chunk_size: int = 1_048_576,
) -> dict[str, torch.Tensor]:
    """Compute exact RMS/cosine diagnostics without materializing full FP32 copies."""

    if tuple(correct.shape) != tuple(shuffled.shape):
        raise ValueError(
            f"paired diagnostic shape mismatch: {tuple(correct.shape)} vs "
            f"{tuple(shuffled.shape)}"
        )
    left = correct.detach().reshape(-1)
    right = shuffled.detach().reshape(-1)
    sum_diff_sq = torch.zeros((), device=left.device, dtype=torch.float64)
    sum_left_sq = torch.zeros_like(sum_diff_sq)
    sum_right_sq = torch.zeros_like(sum_diff_sq)
    sum_dot = torch.zeros_like(sum_diff_sq)
    for start in range(0, int(left.numel()), int(chunk_size)):
        left_chunk = left[start : start + int(chunk_size)].float()
        right_chunk = right[start : start + int(chunk_size)].float()
        difference = left_chunk - right_chunk
        sum_diff_sq += difference.double().square().sum()
        sum_left_sq += left_chunk.double().square().sum()
        sum_right_sq += right_chunk.double().square().sum()
        sum_dot += (left_chunk.double() * right_chunk.double()).sum()
    count = max(1, int(left.numel()))
    difference_rms = torch.sqrt(sum_diff_sq / float(count)).float()
    cosine = (
        sum_dot
        / torch.sqrt(sum_left_sq.clamp_min(1.0e-24) * sum_right_sq.clamp_min(1.0e-24))
    ).float()
    return {
        "difference_rms": difference_rms,
        "cosine": cosine,
    }


class J1aBridgeModel(nn.Module):
    def __init__(self, bridge_fusion: nn.Module, flow: nn.Module) -> None:
        super().__init__()
        self.bridge_fusion = bridge_fusion
        self.flow = flow

    def train(self, mode: bool = True):
        super().train(mode)
        self.flow.eval()
        self.bridge_fusion.bridge.eval()
        return self

    def forward(
        self,
        x_t: torch.Tensor,
        t_tensor: torch.Tensor,
        aggregated_tokens: list[torch.Tensor],
        image_cond: torch.Tensor,
        physical_input: torch.Tensor,
        negative_physical_input: torch.Tensor,
        *,
        physical_scale: float,
        collect_pair_diagnostics: bool = False,
    ) -> dict[str, torch.Tensor]:
        paths = self.bridge_fusion.condition_paths(
            aggregated_tokens,
            image_cond,
            physical_input,
            physical_scale=float(physical_scale),
        )
        if bool(getattr(self.bridge_fusion, "paired_training", False)):
            shuffled_paths = self.bridge_fusion.condition_paths(
                aggregated_tokens,
                image_cond,
                negative_physical_input,
                physical_scale=float(physical_scale),
                cond_stock=paths.cond_stock,
            )
            prediction = self.flow(x_t, t_tensor, paths.cond_fused)
            shuffled_prediction = self.flow(x_t, t_tensor, shuffled_paths.cond_fused)
            with torch.no_grad():
                stock_prediction = self.flow(x_t, t_tensor, paths.cond_stock)
            tensor_stats = {
                key: value
                for key, value in paths.stats.items()
                if torch.is_tensor(value)
            }
            stage_tensor_stats: dict[str, torch.Tensor] = {}
            for source, source_paths in (
                ("correct", paths),
                ("shuffled", shuffled_paths),
            ):
                for stage_name, stage_values in (source_paths.stage_stats or {}).items():
                    for key, value in stage_values.items():
                        stage_tensor_stats[f"{source}_{stage_name}_{key}"] = value
            if collect_pair_diagnostics:
                correct_tensors = paths.stage_tensors or {}
                shuffled_tensors = shuffled_paths.stage_tensors or {}
                for stage_name, correct_values in correct_tensors.items():
                    shuffled_values = shuffled_tensors[stage_name]
                    for tensor_name in (
                        "attended_centered",
                        "context_delta_effective",
                    ):
                        pair = paired_tensor_diagnostics(
                            correct_values[tensor_name],
                            shuffled_values[tensor_name],
                        )
                        stage_tensor_stats[
                            f"pair_{stage_name}_{tensor_name}_difference_rms"
                        ] = pair["difference_rms"]
                        stage_tensor_stats[
                            f"pair_{stage_name}_{tensor_name}_cosine"
                        ] = pair["cosine"]
            return {
                "prediction": prediction,
                "shuffled_prediction": shuffled_prediction,
                "stock_prediction": stock_prediction,
                "cond_stock": paths.cond_stock,
                "cond_fused": paths.cond_fused,
                "cond_shuffled": shuffled_paths.cond_fused,
                "positive_alignment_logit": paths.alignment_logit,
                "negative_alignment_logit": shuffled_paths.alignment_logit,
                **tensor_stats,
                **stage_tensor_stats,
            }
        negative_tokens = self.bridge_fusion.adapter.encode_physical(negative_physical_input)
        negative_logit = self.bridge_fusion.adapter.alignment_logits_from_tokens(
            paths.prefix_tokens,
            negative_tokens,
        )
        prediction = self.flow(x_t, t_tensor, paths.cond_fused)
        tensor_stats = {
            key: value
            for key, value in paths.stats.items()
            if torch.is_tensor(value)
        }
        return {
            "prediction": prediction,
            "cond_stock": paths.cond_stock,
            "cond_fused": paths.cond_fused,
            "positive_alignment_logit": paths.alignment_logit,
            "negative_alignment_logit": negative_logit,
            **tensor_stats,
        }


def build_bridge_condition_inputs(
    model: J1aBridgeModel,
    sample: dict[str, Any],
    negative_sample: dict[str, Any],
    aggregated_tokens: list[torch.Tensor],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Build correct/shuffled inputs while keeping the visual sample camera fixed."""

    if not bool(getattr(model.bridge_fusion, "projected_patch_fusion", False)):
        return (
            sample["physical_grid"].unsqueeze(0).to(
                device=device, dtype=torch.float32, non_blocking=True
            ),
            negative_sample["physical_grid"].unsqueeze(0).to(
                device=device, dtype=torch.float32, non_blocking=True
            ),
            {"input_type": "physical_grid_16"},
        )
    required = (
        "prior_coords",
        "prior_conf",
        "intrinsics",
        "extrinsics",
        "grid_transform",
        "extrinsics_type",
        "camera_forward_sign",
    )
    missing = [key for key in required if key not in sample]
    if missing:
        raise KeyError(f"projected patch sample is missing metadata: {missing}")
    side = infer_patch_grid_side(aggregated_tokens)
    shared = {
        "intrinsics": sample["intrinsics"],
        "extrinsics": sample["extrinsics"],
        "mask_paths": sample["mask_paths"],
        "grid_transform": sample["grid_transform"],
        "extrinsics_type": sample["extrinsics_type"],
        "camera_forward_sign": sample["camera_forward_sign"],
        "patch_grid_side": side,
    }
    correct, correct_report = build_projected_patch_features(
        prior_coords=sample["prior_coords"],
        prior_conf=sample["prior_conf"],
        **shared,
    )
    shuffled, shuffled_report = build_projected_patch_features(
        prior_coords=negative_sample["prior_coords"],
        prior_conf=negative_sample["prior_conf"],
        **shared,
    )
    expected_length = int(aggregated_tokens[0].shape[1]) * (
        int(aggregated_tokens[0].shape[2]) - 5
    )
    if int(correct.shape[1]) != expected_length or tuple(shuffled.shape) != tuple(correct.shape):
        raise RuntimeError(
            "projected patch/visual token mismatch: "
            f"correct={tuple(correct.shape)}, shuffled={tuple(shuffled.shape)}, "
            f"expected_length={expected_length}"
        )
    return (
        correct.to(device=device, dtype=torch.float32, non_blocking=True),
        shuffled.to(device=device, dtype=torch.float32, non_blocking=True),
        {
            "input_type": "pose_guided_projected_patch",
            "negative_policy": "negative_points_with_current_visual_camera_and_mask",
            "correct": correct_report,
            "shuffled": shuffled_report,
        },
    )


def build_models(
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[Any, J1aBridgeModel, dict[str, Any]]:
    install_unused_model_stubs()
    pipeline = TrellisVGGTTo3DPipeline.from_pretrained(args.pretrained)
    pipeline._device = device
    pipeline.low_vram = False
    keep = {
        "image_cond_model",
        "sparse_structure_vggt_cond",
        "sparse_structure_flow_model",
        "sparse_structure_decoder",
    }
    for name in list(pipeline.models):
        if name not in keep:
            del pipeline.models[name]
    for name in ("slat_flow_model", "slat_vggt_cond"):
        if hasattr(pipeline, name):
            delattr(pipeline, name)
    for name in ("camera_head", "point_head", "depth_head", "track_head"):
        if hasattr(pipeline.VGGT_model, name):
            delattr(pipeline.VGGT_model, name)

    pipeline.VGGT_model.to(device).eval()
    for parameter in pipeline.VGGT_model.parameters():
        parameter.requires_grad = False
    for module in pipeline.models.values():
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad = False

    image_model = pipeline.models["image_cond_model"].to(device).eval()
    for parameter in image_model.parameters():
        parameter.requires_grad = False

    bridge = pipeline.models["sparse_structure_vggt_cond"].to(device).eval()
    flow = pipeline.models["sparse_structure_flow_model"].to(device).eval()
    flow.use_checkpoint = bool(args.gradient_checkpointing)
    for block in getattr(flow, "blocks", []):
        block.use_checkpoint = bool(args.gradient_checkpointing)
    fusion_mode = str(getattr(args, "bridge_fusion_mode", "last1_cross_attention"))
    if fusion_mode == "last1_cross_attention":
        bridge_fusion: nn.Module = StockPreservingPhysicalBridge(
            bridge,
            bridge_last_blocks=int(args.bridge_last_blocks),
            hidden_dim=int(args.physical_hidden_dim),
            num_heads=int(args.physical_heads),
        ).to(device)
        stage = "J1a"
        scope = "stock-preserving bridge-last adapter; frozen stock bridge and frozen stock SS flow"
    elif fusion_mode == "multistage_local16":
        bridge_fusion = MultiStageStockPreservingPhysicalBridge(
            bridge,
            fusion_stages=parse_fusion_stages(str(args.fusion_stages), len(bridge.cond_blocks)),
            physical_hidden_dim=int(args.physical_hidden_dim),
            local_hidden_dim=int(args.local_fusion_hidden_dim),
        ).to(device)
        stage = "J1a.1"
        scope = (
            "stock-preserving after-block multistage 16^3 local physical fusion; "
            "frozen stock bridge and frozen stock SS flow"
        )
    elif fusion_mode == "content_visual8":
        fusion_stages = parse_fusion_stages(
            str(args.fusion_stages), len(bridge.cond_blocks)
        )
        if fusion_stages != (0, 1):
            raise ValueError(
                "J1a.2-A content_visual8 is fixed to fusion_stages=0,1 for the "
                f"first controlled experiment, got {fusion_stages}"
            )
        bridge_fusion = ContentBasedPhysicalVisualBridge(
            bridge,
            fusion_stages=fusion_stages,
            physical_hidden_dim=int(args.physical_hidden_dim),
            fusion_dim=int(args.content_fusion_dim),
            num_heads=int(args.content_fusion_heads),
        ).to(device)
        stage = "J1a.2-A"
        scope = (
            "stock-preserving content-based 8^3 physical-to-visual context fusion "
            "before frozen bridge blocks 0 and 1; frozen stock SS flow"
        )
    elif fusion_mode == "pose_guided_patch":
        fusion_stages = parse_fusion_stages(
            str(args.fusion_stages), len(bridge.cond_blocks)
        )
        if fusion_stages != (0, 1):
            raise ValueError(
                "J1a.2-B1 pose_guided_patch is fixed to fusion_stages=0,1, "
                f"got {fusion_stages}"
            )
        bridge_fusion = PoseGuidedProjectedPatchBridge(
            bridge,
            fusion_stages=fusion_stages,
            physical_hidden_dim=int(args.physical_hidden_dim),
            fusion_dim=int(args.content_fusion_dim),
        ).to(device)
        stage = "J1a.2-B1"
        scope = (
            "stock-preserving pose-guided same-view same-patch Point/Pose fusion "
            "before frozen bridge blocks 0 and 1; frozen stock SS flow"
        )
    else:
        raise ValueError(f"unsupported bridge_fusion_mode={fusion_mode!r}")
    model = J1aBridgeModel(bridge_fusion, flow).to(device)
    model.train()

    audit = {
        "vggt_trainable": int(
            sum(parameter.numel() for parameter in pipeline.VGGT_model.parameters() if parameter.requires_grad)
        ),
        "image_encoder_trainable": int(
            sum(
                parameter.numel()
                for parameter in pipeline.models["image_cond_model"].parameters()
                if parameter.requires_grad
            )
        ),
        "stock_bridge_trainable": int(
            sum(parameter.numel() for parameter in bridge.parameters() if parameter.requires_grad)
        ),
        "stock_flow_trainable": int(
            sum(parameter.numel() for parameter in flow.parameters() if parameter.requires_grad)
        ),
        "adapter_trainable": int(
            sum(
                parameter.numel()
                for parameter in bridge_fusion.parameters()
                if parameter.requires_grad
            )
        ),
    }
    if (
        audit["vggt_trainable"]
        or audit["image_encoder_trainable"]
        or audit["stock_bridge_trainable"]
        or audit["stock_flow_trainable"]
        or not audit["adapter_trainable"]
    ):
        raise RuntimeError(f"J1a trainable whitelist failed: {audit}")
    summary = {
        "stage": stage,
        "scope": scope,
        "bridge": bridge_fusion.metadata(),
        "trainable_parameter_audit": audit,
        "trainable_parameter_count": int(
            sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        ),
        "total_parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "flow_lora_enabled": False,
        "occupancy_loss_enabled": False,
    }
    return pipeline, model, summary


def gradient_group_norms(model: J1aBridgeModel) -> dict[str, float]:
    if bool(getattr(model.bridge_fusion, "projected_patch_fusion", False)):
        groups = {
            "projected_patch_encoder": ("bridge_fusion.physical_encoder.",),
            "visual_query_projection": (".query_proj.",),
            "projected_patch_interaction": (".interaction.",),
            "physical_output_projection": (".output_proj.",),
            "alignment_gate": (".gate_head.",),
        }
    elif bool(getattr(model.bridge_fusion, "content_visual_fusion", False)):
        groups = {
            "physical_encoder": ("bridge_fusion.physical_encoder.",),
            "visual_query_projection": (".query_proj.",),
            "physical_cross_attention": (".cross_attn.",),
            "physical_output_projection": (".output_proj.",),
            "alignment_gate": (".gate_head.",),
        }
    elif bool(getattr(model.bridge_fusion, "paired_training", False)):
        groups = {
            "physical_encoder": ("bridge_fusion.physical_encoder.",),
            "physical_local_fusion": (
                ".bridge_norm.",
                ".physical_norm.",
                ".bridge_proj.",
                ".physical_proj.",
                ".interaction.",
            ),
            "physical_output_projection": (".output_proj.",),
            "alignment_head": (".alignment_proj.",),
        }
    else:
        groups = {
            "physical_encoder": ("bridge_fusion.adapter.physical_encoder.",),
            "physical_cross_attention": ("bridge_fusion.adapter.cross_attn.",),
            "physical_output_projection": ("bridge_fusion.adapter.output_proj.",),
            "alignment_head": ("bridge_fusion.adapter.alignment_head.",),
        }
    output: dict[str, float] = {}
    for label, patterns in groups.items():
        total = 0.0
        for name, parameter in model.named_parameters():
            matches = any(
                name.startswith(pattern) if pattern.startswith("bridge_fusion") else pattern in name
                for pattern in patterns
            )
            if matches and parameter.grad is not None:
                total += float(parameter.grad.detach().float().square().sum().item())
        output[label] = float(math.sqrt(total))
    return output


@torch.no_grad()
def architecture_audit(
    pipeline,
    model: J1aBridgeModel,
    sample: dict[str, Any],
    device: torch.device,
    negative_sample: dict[str, Any] | None = None,
    raise_on_failure: bool = True,
) -> dict[str, Any]:
    images = rgba_images(sample["image_paths"], sample["mask_paths"], pipeline)
    aggregated, image_cond = encode_frozen_features(pipeline, images)
    if negative_sample is None:
        negative_sample = sample
    physical, negative_physical, input_report = build_bridge_condition_inputs(
        model,
        sample,
        negative_sample,
        aggregated,
        device,
    )
    paired = bool(getattr(model.bridge_fusion, "paired_training", False))
    if paired:
        null_physical = (
            make_null_projected_patch_features(physical)
            if bool(getattr(model.bridge_fusion, "projected_patch_fusion", False))
            else make_null_physical_grid(physical)
        )
        null_tokens = model.bridge_fusion.encode_physical(null_physical)
    else:
        null_physical = torch.zeros_like(physical)
        null_tokens = model.bridge_fusion.adapter.encode_physical(null_physical)
    off = model.bridge_fusion.condition(
        aggregated,
        image_cond,
        None,
        physical_present=False,
    )
    paths = model.bridge_fusion.condition_paths(
        aggregated,
        image_cond,
        physical,
        physical_scale=1.0,
    )
    null_present_diff = 0.0
    null_present_exact = True
    sensitivity: dict[str, Any] = {}
    if paired:
        null_paths = model.bridge_fusion.condition_paths(
            aggregated,
            image_cond,
            null_physical,
            physical_scale=1.0,
            cond_stock=paths.cond_stock,
        )
        null_present_diff = float(
            (null_paths.cond_fused.float() - paths.cond_stock.float()).abs().max().item()
        )
        null_present_exact = bool(torch.equal(null_paths.cond_fused, paths.cond_stock))
        if bool(getattr(model.bridge_fusion, "content_visual_fusion", False)):
            shuffled_paths = model.bridge_fusion.condition_paths(
                aggregated,
                image_cond,
                negative_physical,
                physical_scale=1.0,
                cond_stock=paths.cond_stock,
            )
            physical_pair = paired_tensor_diagnostics(
                paths.physical_tokens,
                shuffled_paths.physical_tokens,
            )
            stages: dict[str, Any] = {}
            for stage_name, correct_values in (paths.stage_tensors or {}).items():
                shuffled_values = (shuffled_paths.stage_tensors or {})[stage_name]
                attended_pair = paired_tensor_diagnostics(
                    correct_values["attended_centered"],
                    shuffled_values["attended_centered"],
                )
                stages[stage_name] = {
                    "attended_correct_shuffled_rms": float(
                        attended_pair["difference_rms"].item()
                    ),
                    "attended_correct_shuffled_cosine": float(
                        attended_pair["cosine"].item()
                    ),
                    "correct_internal_attended_rms": float(
                        paths.stage_stats[stage_name]["attended_centered_rms"].item()
                    ),
                    "shuffled_internal_attended_rms": float(
                        shuffled_paths.stage_stats[stage_name][
                            "attended_centered_rms"
                        ].item()
                    ),
                }
            sensitivity = {
                "physical_token_correct_shuffled_rms": float(
                    physical_pair["difference_rms"].item()
                ),
                "physical_token_correct_shuffled_cosine": float(
                    physical_pair["cosine"].item()
                ),
                "final_correct_is_stock_at_zero_init": bool(
                    torch.equal(paths.cond_fused, paths.cond_stock)
                ),
                "final_shuffled_is_stock_at_zero_init": bool(
                    torch.equal(shuffled_paths.cond_fused, paths.cond_stock)
                ),
                "stages": stages,
            }
    hard_stock_diff = float((off.float() - paths.cond_stock.float()).abs().max().item())
    if bool(getattr(model.bridge_fusion, "content_visual_fusion", False)):
        output_projection_zero = all(
            torch.equal(stage.output_proj.weight, torch.zeros_like(stage.output_proj.weight))
            and torch.equal(stage.output_proj.bias, torch.zeros_like(stage.output_proj.bias))
            for stage in model.bridge_fusion.stage_adapters.values()
        )
    elif paired:
        output_projection_zero = all(
            torch.equal(stage.output_proj.weight, torch.zeros_like(stage.output_proj.weight))
            and torch.equal(stage.output_proj.bias, torch.zeros_like(stage.output_proj.bias))
            and torch.equal(
                stage.alignment_proj.weight, torch.zeros_like(stage.alignment_proj.weight)
            )
            and torch.equal(stage.alignment_proj.bias, torch.zeros_like(stage.alignment_proj.bias))
            for stage in model.bridge_fusion.stage_adapters.values()
        )
    else:
        output_projection_zero = bool(
            torch.equal(
                model.bridge_fusion.adapter.output_proj.weight,
                torch.zeros_like(model.bridge_fusion.adapter.output_proj.weight),
            )
            and torch.equal(
                model.bridge_fusion.adapter.output_proj.bias,
                torch.zeros_like(model.bridge_fusion.adapter.output_proj.bias),
            )
        )
    report = {
        "hard_stock_route_returns_stock_shape": tuple(off.shape) == tuple(paths.cond_stock.shape),
        "hard_stock_route_max_abs_diff": hard_stock_diff,
        "hard_stock_route_exact": bool(torch.equal(off, paths.cond_stock)),
        "null_evidence_preserves_xyz": bool(
            paired
            and not getattr(model.bridge_fusion, "projected_patch_fusion", False)
        ),
        "null_pose_ray_uv_preserved": bool(
            getattr(model.bridge_fusion, "projected_patch_fusion", False)
        ),
        "null_present_condition_max_abs_diff": null_present_diff,
        "null_present_condition_exact": null_present_exact,
        "null_centered_token_max_abs": float(null_tokens.abs().max().item()),
        "zero_init_condition_max_abs_diff": float(
            (paths.cond_fused.float() - paths.cond_stock.float()).abs().max().item()
        ),
        "output_projection_zero": output_projection_zero,
        "flow_lora_enabled": False,
        "condition_input_audit": input_report,
        "null_geometry_preserved": bool(
            getattr(model.bridge_fusion, "projected_patch_fusion", False)
        ),
        "untrained_content_sensitivity": sensitivity,
    }
    content_sensitivity_passed = True
    if bool(getattr(model.bridge_fusion, "content_visual_fusion", False)):
        content_sensitivity_passed = bool(
            sensitivity["physical_token_correct_shuffled_rms"] > 0.0
            and sensitivity["final_correct_is_stock_at_zero_init"]
            and sensitivity["final_shuffled_is_stock_at_zero_init"]
            and sensitivity["stages"]
            and all(
                row["attended_correct_shuffled_rms"] > 0.0
                for row in sensitivity["stages"].values()
            )
        )
    report["untrained_content_sensitivity_passed"] = content_sensitivity_passed
    report["passed"] = bool(
        report["hard_stock_route_returns_stock_shape"]
        and report["hard_stock_route_max_abs_diff"] == 0.0
        and report["hard_stock_route_exact"]
        and report["null_present_condition_max_abs_diff"] == 0.0
        and report["null_present_condition_exact"]
        and report["null_centered_token_max_abs"] == 0.0
        and report["zero_init_condition_max_abs_diff"] == 0.0
        and report["output_projection_zero"]
        and report["untrained_content_sensitivity_passed"]
        and not report["flow_lora_enabled"]
    )
    if not report["passed"] and raise_on_failure:
        raise RuntimeError(f"J1a architecture audit failed: {report}")
    return report


def save_checkpoint(
    path: Path,
    *,
    model: J1aBridgeModel,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    step: int,
    args: argparse.Namespace,
    model_summary: dict[str, Any],
) -> None:
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters_finite(trainable):
        raise RuntimeError(f"refusing to save non-finite J1a parameters: {path}")
    if not optimizer_state_finite(optimizer):
        raise RuntimeError(f"refusing to save non-finite J1a optimizer: {path}")
    if not finite_tree(scaler.state_dict()):
        raise RuntimeError(f"refusing to save non-finite J1a scaler: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "reconvggt.stock_preserving_pointpose_bridge.j1a.v1",
            "step": int(step),
            "args": vars(args),
            "model_summary": model_summary,
            "model_trainable_state": trainable_state_dict(model),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="J1a stock-preserving physical-aware bridge training with frozen SS flow."
    )
    parser.add_argument("--cache_manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--pretrained", default="Stable-X/trellis-vggt-v0-2")
    parser.add_argument("--indices", default="all")
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--save_every", type=int, default=25)
    parser.add_argument("--log_every", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1.0e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--amp_dtype", choices=["fp16", "bf16", "none"], default="fp16")
    parser.add_argument("--amp_init_scale", type=float, default=8192.0)
    parser.add_argument("--nonfinite_policy", choices=["error", "skip"], default="error")
    parser.add_argument("--max_nonfinite_attempts", type=int, default=0)
    parser.add_argument("--physical_hidden_dim", type=int, default=256)
    parser.add_argument("--physical_heads", type=int, default=8)
    parser.add_argument("--physical_scale", type=float, default=1.0)
    parser.add_argument(
        "--bridge_fusion_mode",
        choices=[
            "last1_cross_attention",
            "multistage_local16",
            "content_visual8",
            "pose_guided_patch",
        ],
        default="last1_cross_attention",
    )
    parser.add_argument("--bridge_last_blocks", type=int, choices=[1, 2], default=1)
    parser.add_argument("--fusion_stages", default="0,1,2,3")
    parser.add_argument("--local_fusion_hidden_dim", type=int, default=128)
    parser.add_argument("--content_fusion_dim", type=int, default=128)
    parser.add_argument("--content_fusion_heads", type=int, default=4)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--t_schedule", choices=["uniform", "logit_normal", "high_t_mix"], default="uniform")
    parser.add_argument("--alignment_weight", type=float, default=0.05)
    parser.add_argument("--alignment_warmup_steps", type=int, default=20)
    parser.add_argument("--shuffled_stock_weight", type=float, default=0.0)
    parser.add_argument("--gain_weight", type=float, default=0.0)
    parser.add_argument(
        "--gain_margin",
        type=float,
        default=0.005,
        help="Required relative correct-vs-stock flow-MSE gain (0.005 means 0.5%%).",
    )
    parser.add_argument("--delta_norm_weight", type=float, default=0.01)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--resume_weights_only", action="store_true")
    args = parser.parse_args()

    if args.max_steps <= 0 or args.save_every <= 0 or args.log_every <= 0:
        raise ValueError("max/save/log steps must be positive")
    if args.grad_accum <= 0 or args.grad_clip <= 0 or args.amp_init_scale <= 0:
        raise ValueError("grad_accum, grad_clip, and amp_init_scale must be positive")
    if (
        args.alignment_weight < 0
        or args.shuffled_stock_weight < 0
        or args.gain_weight < 0
        or args.gain_margin < 0
        or args.delta_norm_weight < 0
    ):
        raise ValueError("loss weights must be non-negative")
    if (
        args.local_fusion_hidden_dim <= 0
        or args.content_fusion_dim <= 0
        or args.content_fusion_heads <= 0
    ):
        raise ValueError("fusion dimensions and head counts must be positive")
    if args.content_fusion_dim % args.content_fusion_heads:
        raise ValueError("content_fusion_dim must be divisible by content_fusion_heads")
    parse_fusion_stages(str(args.fusion_stages))

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        dist.init_process_group(backend="nccl", timeout=datetime.timedelta(hours=2))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    seed = int(args.seed) + rank * 100003
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if rank:
        time.sleep(min(1.5 * rank, 8.0))

    dataset = PointPoseCacheDataset(args.cache_manifest, indices=args.indices)
    negative_miner = HardNegativeMiner(dataset)
    uid_to_dataset_index = {
        str(row["uid"]): index for index, row in enumerate(dataset.samples)
    }
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=int(args.seed),
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        sampler=sampler,
        num_workers=int(args.num_workers),
        collate_fn=collate_one,
        pin_memory=True,
    )
    pipeline, model, model_summary = build_models(args, device)
    model_summary["dataset_size"] = len(dataset)
    model_summary["unique_object_count"] = len(
        {str(row.get("object_uid", "")) for row in dataset.samples}
    )
    model_summary["negative_mining"] = {
        "mode": "nearest_log_prior_and_target_count",
        "same_object_forbidden": True,
        "flow_path_on_negative": bool(getattr(model.bridge_fusion, "paired_training", False)),
        "same_xt_t_noise_for_correct_and_shuffled": bool(
            getattr(model.bridge_fusion, "paired_training", False)
        ),
        "shuffled_target": (
            "stock_velocity_preservation"
            if bool(getattr(model.bridge_fusion, "paired_training", False))
            else "alignment_only"
        ),
    }
    model_summary["architecture_audit"] = architecture_audit(
        pipeline,
        model,
        dataset[0],
        device,
        negative_sample=dataset[negative_miner[0]],
    )
    if rank == 0:
        print(json.dumps(model_summary, indent=2, ensure_ascii=False), flush=True)

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        betas=(0.9, 0.95),
        eps=1.0e-8,
    )
    scaler = torch.cuda.amp.GradScaler(
        enabled=args.amp_dtype == "fp16",
        init_scale=float(args.amp_init_scale),
    )
    start_step = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        saved_args = checkpoint.get("args", {})
        expected = {
            "pretrained": str(args.pretrained),
            "physical_hidden_dim": int(args.physical_hidden_dim),
            "physical_heads": int(args.physical_heads),
            "bridge_last_blocks": int(args.bridge_last_blocks),
            "bridge_fusion_mode": str(args.bridge_fusion_mode),
            "fusion_stages": str(args.fusion_stages),
            "local_fusion_hidden_dim": int(args.local_fusion_hidden_dim),
            "content_fusion_dim": int(args.content_fusion_dim),
            "content_fusion_heads": int(args.content_fusion_heads),
        }
        legacy_defaults = {
            "bridge_fusion_mode": "last1_cross_attention",
            "fusion_stages": "0,1,2,3",
            "local_fusion_hidden_dim": 128,
            "content_fusion_dim": 128,
            "content_fusion_heads": 4,
        }
        mismatches = {
            key: {
                "checkpoint": saved_args.get(key, legacy_defaults.get(key)),
                "current": value,
            }
            for key, value in expected.items()
            if saved_args.get(key, legacy_defaults.get(key)) != value
        }
        if str(args.bridge_fusion_mode) in {
            "multistage_local16",
            "content_visual8",
            "pose_guided_patch",
        }:
            saved_bridge_version = (
                checkpoint.get("model_summary", {})
                .get("bridge", {})
                .get("version")
            )
            current_bridge_version = model.bridge_fusion.metadata()["version"]
            if saved_bridge_version != current_bridge_version:
                mismatches["bridge_version"] = {
                    "checkpoint": saved_bridge_version,
                    "current": current_bridge_version,
                }
        if mismatches:
            raise RuntimeError(f"J1a checkpoint configuration mismatch: {mismatches}")
        load_partial_state(
            model,
            checkpoint["model_trainable_state"],
            require_all_trainable=True,
        )
        if not args.resume_weights_only:
            optimizer.load_state_dict(checkpoint["optimizer"])
            scaler.load_state_dict(checkpoint["scaler"])
            start_step = int(checkpoint.get("step", 0))

    wrapped: nn.Module = model
    if world_size > 1:
        wrapped = DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            find_unused_parameters=False,
        )

    use_amp = args.amp_dtype != "none"
    amp_dtype = torch.float16 if args.amp_dtype == "fp16" else torch.bfloat16
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, Any]] = []
    global_step = start_step
    micro_step = global_step * int(args.grad_accum)
    applied_updates = 0
    nonfinite_attempts = 0
    epoch = 0
    wall_start = time.time()
    optimizer.zero_grad(set_to_none=True)

    while global_step < int(args.max_steps):
        sampler.set_epoch(epoch)
        for batch in loader:
            if global_step >= int(args.max_steps):
                break
            sample_index = uid_to_dataset_index[str(batch["uid"])]
            negative_index = negative_miner[int(sample_index)]
            negative = dataset[negative_index]
            images = rgba_images(batch["image_paths"], batch["mask_paths"], pipeline)
            aggregated, image_cond = encode_frozen_features(pipeline, images)
            physical, negative_physical, _ = build_bridge_condition_inputs(
                model,
                batch,
                negative,
                aggregated,
                device,
            )
            target = batch["target"].unsqueeze(0).to(
                device=device, dtype=torch.float32, non_blocking=True
            )
            noise = torch.randn_like(target)
            t_model = sample_t(str(args.t_schedule), device)
            x_t, gt_velocity = pipeline.sparse_structure_sampler._get_model_gt(
                target, t_model, noise
            )
            t_tensor = torch.full((1,), 1000.0 * t_model, device=device, dtype=torch.float32)
            sync_step = ((micro_step + 1) % int(args.grad_accum)) == 0
            collect_pair_diagnostics = bool(
                sync_step
                and (
                    global_step == 0
                    or (global_step + 1) % int(args.log_every) == 0
                )
            )
            sync_context = wrapped.no_sync() if world_size > 1 and not sync_step else torch.enable_grad()

            with sync_context:
                with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
                    output = wrapped(
                        x_t,
                        t_tensor,
                        aggregated,
                        image_cond,
                        physical,
                        negative_physical,
                        physical_scale=float(args.physical_scale),
                        collect_pair_diagnostics=collect_pair_diagnostics,
                    )
                    flow_loss = F.mse_loss(output["prediction"].float(), gt_velocity.float())
                    paired_training = "shuffled_prediction" in output
                    if paired_training:
                        shuffled_flow_loss = F.mse_loss(
                            output["shuffled_prediction"].float(), gt_velocity.float()
                        )
                        stock_flow_loss = F.mse_loss(
                            output["stock_prediction"].float(), gt_velocity.float()
                        )
                        stock_prediction_energy = (
                            output["stock_prediction"]
                            .float()
                            .square()
                            .mean()
                            .detach()
                            .clamp_min(1.0e-6)
                        )
                        shuffled_stock_loss_raw = F.mse_loss(
                            output["shuffled_prediction"].float(),
                            output["stock_prediction"].float(),
                        )
                        shuffled_stock_loss = (
                            shuffled_stock_loss_raw / stock_prediction_energy
                        )
                        relative_correct_gain = (
                            stock_flow_loss.detach() - flow_loss
                        ) / stock_flow_loss.detach().clamp_min(1.0e-6)
                        gain_loss = F.relu(
                            float(args.gain_margin) - relative_correct_gain
                        )
                        stock_minus_correct_flow_loss = stock_flow_loss - flow_loss.detach()
                        shuffled_minus_correct_flow_loss = (
                            shuffled_flow_loss - flow_loss.detach()
                        )
                    else:
                        shuffled_flow_loss = flow_loss.detach().new_zeros(())
                        stock_flow_loss = flow_loss.detach().new_zeros(())
                        stock_prediction_energy = flow_loss.detach().new_zeros(())
                        shuffled_stock_loss_raw = flow_loss.new_zeros(())
                        shuffled_stock_loss = flow_loss.new_zeros(())
                        relative_correct_gain = flow_loss.new_zeros(())
                        gain_loss = flow_loss.new_zeros(())
                        stock_minus_correct_flow_loss = flow_loss.detach().new_zeros(())
                        shuffled_minus_correct_flow_loss = flow_loss.detach().new_zeros(())
                    alignment_logits = torch.cat(
                        (output["positive_alignment_logit"], output["negative_alignment_logit"]),
                        dim=0,
                    ).float()
                    alignment_targets = torch.tensor([1.0, 0.0], device=device)
                    alignment_loss = F.binary_cross_entropy_with_logits(
                        alignment_logits, alignment_targets
                    )
                    warmup = max(0, int(args.alignment_warmup_steps))
                    alignment_factor = (
                        1.0
                        if warmup == 0
                        else min(1.0, float(global_step) / float(warmup))
                    )
                    delta_ratio = output.get(
                        "context_delta_to_context_ratio",
                        output["condition_delta_to_stock_ratio"],
                    ).float()
                    delta_loss = delta_ratio.square()
                    loss = (
                        flow_loss
                        + float(args.alignment_weight) * alignment_factor * alignment_loss
                        + float(args.shuffled_stock_weight) * shuffled_stock_loss
                        + float(args.gain_weight) * gain_loss
                        + float(args.delta_norm_weight) * delta_loss
                    )
                    scaled_loss = loss / float(args.grad_accum)
                scaler.scale(scaled_loss).backward()

            micro_step += 1
            if not sync_step:
                continue

            scaler.unscale_(optimizer)
            grad_norms = gradient_group_norms(model)
            diagnostics = [
                loss,
                flow_loss,
                alignment_loss,
                shuffled_flow_loss,
                stock_flow_loss,
                stock_prediction_energy,
                shuffled_stock_loss_raw,
                shuffled_stock_loss,
                relative_correct_gain,
                gain_loss,
                delta_loss,
                output["condition_delta_rms"],
                output["condition_delta_abs_max"],
                output["condition_delta_to_stock_ratio"],
                delta_ratio,
                output["alignment_probability_mean"],
            ]
            forward_finite = distributed_all_true(
                tensors_finite(diagnostics), device, world_size
            )
            gradient_finite = distributed_all_true(
                gradients_finite(trainable), device, world_size
            )
            update_finite = forward_finite and gradient_finite
            scaler_before = float(scaler.get_scale()) if scaler.is_enabled() else None
            clip_total_norm = None
            optimizer_step_applied = False
            if update_finite:
                clip_total_norm = float(
                    torch.nn.utils.clip_grad_norm_(trainable, float(args.grad_clip)).item()
                )
                scaler.step(optimizer)
                scaler.update()
                scaler_after = float(scaler.get_scale()) if scaler.is_enabled() else None
                optimizer_step_applied = not scaler.is_enabled() or scaler_after >= scaler_before
                if optimizer_step_applied:
                    post_finite = distributed_all_true(
                        parameters_finite(trainable) and optimizer_state_finite(optimizer),
                        device,
                        world_size,
                    )
                    if not post_finite:
                        raise RuntimeError("J1a optimizer produced non-finite parameters/state")
            else:
                nonfinite_attempts += 1
                if scaler.is_enabled():
                    scaler.update()
                scaler_after = float(scaler.get_scale()) if scaler.is_enabled() else None
            optimizer.zero_grad(set_to_none=True)

            if optimizer_step_applied:
                global_step += 1
                applied_updates += 1
            row = {
                "step": int(global_step),
                "micro_step": int(micro_step),
                "uid": batch["uid"],
                "negative_uid": negative["uid"],
                "flow_loss": distributed_mean(flow_loss, world_size),
                "shuffled_flow_loss": distributed_mean(shuffled_flow_loss, world_size),
                "stock_flow_loss": distributed_mean(stock_flow_loss, world_size),
                "stock_prediction_energy": distributed_mean(
                    stock_prediction_energy, world_size
                ),
                "stock_minus_correct_flow_loss": distributed_mean(
                    stock_minus_correct_flow_loss, world_size
                ),
                "shuffled_minus_correct_flow_loss": distributed_mean(
                    shuffled_minus_correct_flow_loss, world_size
                ),
                "shuffled_stock_loss_raw": distributed_mean(
                    shuffled_stock_loss_raw, world_size
                ),
                "shuffled_stock_loss_relative": distributed_mean(
                    shuffled_stock_loss, world_size
                ),
                "shuffled_stock_loss": distributed_mean(
                    shuffled_stock_loss, world_size
                ),
                "relative_correct_gain": distributed_mean(
                    relative_correct_gain, world_size
                ),
                "gain_loss": distributed_mean(gain_loss, world_size),
                "alignment_loss": distributed_mean(alignment_loss, world_size),
                "delta_norm_loss": distributed_mean(delta_loss, world_size),
                "condition_delta_rms": distributed_mean(output["condition_delta_rms"], world_size),
                "condition_delta_to_stock_ratio": distributed_mean(
                    output["condition_delta_to_stock_ratio"], world_size
                ),
                "regularization_delta_ratio": distributed_mean(delta_ratio, world_size),
                "positive_alignment_probability": distributed_mean(
                    torch.sigmoid(output["positive_alignment_logit"]).mean(), world_size
                ),
                "negative_alignment_probability": distributed_mean(
                    torch.sigmoid(output["negative_alignment_logit"]).mean(), world_size
                ),
                "alignment_factor": float(alignment_factor),
                "gradient_norms": grad_norms,
                "clip_total_norm": clip_total_norm,
                "forward_finite": bool(forward_finite),
                "gradient_finite": bool(gradient_finite),
                "update_finite": bool(update_finite),
                "optimizer_step_applied": bool(optimizer_step_applied),
                "nonfinite_attempts": int(nonfinite_attempts),
                "amp_dtype": str(args.amp_dtype),
                "scaler_before": scaler_before,
                "scaler_after": scaler_after,
                "t": float(t_model),
                "elapsed_seconds": float(time.time() - wall_start),
            }
            if paired_training:
                stage_metrics: dict[str, dict[str, float]] = {}
                for stage_index in parse_fusion_stages(str(args.fusion_stages)):
                    stage_name = f"stage_{stage_index}"
                    correct_gate_key = (
                        f"correct_{stage_name}_alignment_probability_mean"
                    )
                    shuffled_gate_key = (
                        f"shuffled_{stage_name}_alignment_probability_mean"
                    )
                    if correct_gate_key not in output:
                        continue
                    stage_metrics[stage_name] = {
                        "correct_alignment_probability": distributed_mean(
                            output[correct_gate_key], world_size
                        ),
                        "shuffled_alignment_probability": distributed_mean(
                            output[shuffled_gate_key], world_size
                        ),
                        "correct_delta_to_hidden_ratio": distributed_mean(
                            output[
                                f"correct_{stage_name}_effective_delta_to_hidden_ratio"
                            ],
                            world_size,
                        ),
                        "shuffled_delta_to_hidden_ratio": distributed_mean(
                            output[
                                f"shuffled_{stage_name}_effective_delta_to_hidden_ratio"
                            ],
                            world_size,
                        ),
                    }
                    correct_attended_key = (
                        f"correct_{stage_name}_attended_centered_rms"
                    )
                    if correct_attended_key in output:
                        stage_metrics[stage_name].update(
                            {
                                "correct_attended_rms": distributed_mean(
                                    output[correct_attended_key], world_size
                                ),
                                "shuffled_attended_rms": distributed_mean(
                                    output[
                                        f"shuffled_{stage_name}_attended_centered_rms"
                                    ],
                                    world_size,
                                ),
                            }
                        )
                    pair_prefix = f"pair_{stage_name}_"
                    pair_attended_key = (
                        pair_prefix + "attended_centered_difference_rms"
                    )
                    if pair_attended_key in output:
                        stage_metrics[stage_name].update(
                            {
                                "correct_shuffled_attended_rms": distributed_mean(
                                    output[pair_attended_key], world_size
                                ),
                                "correct_shuffled_attended_cosine": distributed_mean(
                                    output[
                                        pair_prefix + "attended_centered_cosine"
                                    ],
                                    world_size,
                                ),
                                "correct_shuffled_context_delta_rms": distributed_mean(
                                    output[
                                        pair_prefix
                                        + "context_delta_effective_difference_rms"
                                    ],
                                    world_size,
                                ),
                                "correct_shuffled_context_delta_cosine": distributed_mean(
                                    output[
                                        pair_prefix
                                        + "context_delta_effective_cosine"
                                    ],
                                    world_size,
                                ),
                            }
                        )
                row["per_stage"] = stage_metrics
            if rank == 0 and (
                not optimizer_step_applied
                or global_step == 1
                or global_step % int(args.log_every) == 0
            ):
                history.append(row)
                print(f"[j1a_train] {json.dumps(row, ensure_ascii=False)}", flush=True)

            if not optimizer_step_applied:
                message = (
                    f"J1a non-finite update attempt={nonfinite_attempts} micro_step={micro_step} "
                    f"amp={args.amp_dtype} scaler={scaler_before}->{scaler_after}"
                )
                if (
                    args.nonfinite_policy == "error"
                    or nonfinite_attempts > int(args.max_nonfinite_attempts)
                ):
                    raise RuntimeError(message)
            if optimizer_step_applied and rank == 0 and (
                global_step % int(args.save_every) == 0
                or global_step == int(args.max_steps)
            ):
                save_checkpoint(
                    output_dir / "checkpoints" / f"step_{global_step:06d}.pt",
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    step=global_step,
                    args=args,
                    model_summary=model_summary,
                )
                save_checkpoint(
                    output_dir / "checkpoints" / "last.pt",
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    step=global_step,
                    args=args,
                    model_summary=model_summary,
                )
        epoch += 1

    if rank == 0:
        report = {
            "stage": model_summary["stage"],
            "args": vars(args),
            "world_size": int(world_size),
            "per_rank_batch_size": 1,
            "effective_batch_size": int(world_size * args.grad_accum),
            "dataset_size": len(dataset),
            "unique_object_count": model_summary["unique_object_count"],
            "start_global_step": int(start_step),
            "applied_optimizer_updates": int(applied_updates),
            "completed_global_step": int(global_step),
            "nonfinite_attempts": int(nonfinite_attempts),
            "model": model_summary,
            "history": history,
        }
        (output_dir / "train_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
