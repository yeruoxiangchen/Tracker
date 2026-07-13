#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
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
from PIL import Image
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, DistributedSampler

TRACKER_ROOT = Path(__file__).resolve().parents[1]
RECONVIAGEN_ROOT = TRACKER_ROOT / "ReconViaGen"
VGGT_ROOT = RECONVIAGEN_ROOT / "wheels" / "vggt"
for path in (TRACKER_ROOT, RECONVIAGEN_ROOT, VGGT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import trellis.pipelines.trellis_image_to_3d as trellis_image_to_3d  # noqa: E402
from trellis.pipelines import TrellisVGGTTo3DPipeline  # noqa: E402

from reconvggt_ar_adapter_a.inspect_and_sanity import (  # noqa: E402
    DreamSimStub,
    normalize_image_cond,
)
from reconvggt_ar_adapter_a.pointpose_ss_condition import (  # noqa: E402
    PHYSICAL_FEATURE_NAMES,
    PointPoseConditionNet,
    feature_schema_hash,
    load_partial_state,
    trainable_state_dict,
)


class SegmentationStub(nn.Module):
    def forward(self, *args, **kwargs):
        raise RuntimeError("BiRefNet stub should not run when explicit alpha masks are supplied")


class SegmentationStubFactory:
    @staticmethod
    def from_pretrained(*args, **kwargs):
        return SegmentationStub()


def install_unused_model_stubs() -> None:
    trellis_image_to_3d.dreamsim = lambda *args, **kwargs: (DreamSimStub(), None)
    trellis_image_to_3d.AutoModelForImageSegmentation = SegmentationStubFactory


class PointPoseCacheDataset(Dataset):
    def __init__(self, manifest: str | Path, *, indices: str = "all") -> None:
        self.manifest_path = Path(manifest)
        payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.root = Path(payload.get("output_dir", self.manifest_path.parent))
        self.feature_names = tuple(payload.get("feature_names", []))
        if self.feature_names != PHYSICAL_FEATURE_NAMES:
            raise ValueError(
                f"physical feature schema mismatch: cache={self.feature_names}, code={PHYSICAL_FEATURE_NAMES}"
            )
        if payload.get("feature_schema_hash") != feature_schema_hash():
            raise ValueError("cache physical feature schema hash does not match this code version")
        samples = payload.get("samples")
        if not isinstance(samples, list) or not samples:
            raise ValueError(f"cache manifest has no samples: {manifest}")
        selected = parse_indices(indices, len(samples))
        self.samples = [samples[index] for index in selected]
        uids = [str(row.get("uid", "")) for row in self.samples]
        if not all(uids) or len(set(uids)) != len(uids):
            raise ValueError("cache subset contains empty or duplicate uid values")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.samples[index]
        physical_path = Path(row["physical_grid"])
        if not physical_path.is_absolute():
            physical_path = self.root / physical_path
        with np.load(physical_path) as data:
            physical_grid = np.asarray(data["physical_grid"], dtype=np.float32)
        latent_path = Path(row["ss_latent"])
        with np.load(latent_path) as data:
            target = np.asarray(data["z"], dtype=np.float32)
            target_coords = np.asarray(data["target_coords"], dtype=np.int64)[:, -3:]
        if target.ndim == 5 and target.shape[0] == 1:
            target = target[0]
        if physical_grid.shape != (len(PHYSICAL_FEATURE_NAMES), 16, 16, 16):
            raise ValueError(f"uid={row['uid']} invalid physical_grid shape {physical_grid.shape}")
        if target.shape != (8, 16, 16, 16):
            raise ValueError(f"uid={row['uid']} invalid target latent shape {target.shape}")
        if target_coords.ndim != 2 or target_coords.shape[1] != 3:
            raise ValueError(f"uid={row['uid']} invalid target_coords shape {target_coords.shape}")
        if not np.isfinite(physical_grid).all() or not np.isfinite(target).all():
            raise ValueError(f"uid={row['uid']} cache contains non-finite values")
        if target_coords.size and ((target_coords < 0).any() or (target_coords > 63).any()):
            raise ValueError(f"uid={row['uid']} target_coords outside [0,63]")
        return {
            "uid": str(row["uid"]),
            "object_uid": str(row.get("object_uid", "")),
            "physical_grid": torch.from_numpy(physical_grid),
            "target": torch.from_numpy(target),
            "target_coords": torch.from_numpy(target_coords),
            "image_paths": list(row["image_paths"]),
            "mask_paths": list(row["mask_paths"]),
        }


def parse_indices(spec: str, size: int) -> list[int]:
    text = str(spec).strip().lower()
    if text in {"", "all"}:
        return list(range(size))
    result: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            result.extend(range(int(start), int(end) + 1))
        else:
            result.append(int(part))
    bad = [index for index in result if index < 0 or index >= size]
    if bad:
        raise IndexError(f"indices out of range for size={size}: {bad}")
    return result


def collate_one(items: list[dict[str, Any]]) -> dict[str, Any]:
    if len(items) != 1:
        raise ValueError("PointPose SS training currently requires per-rank batch_size=1")
    return items[0]


def rgba_images(image_paths: list[str], mask_paths: list[str], pipeline) -> list[Image.Image]:
    if len(image_paths) != len(mask_paths) or not image_paths:
        raise ValueError("image/mask paths must be non-empty and aligned")
    output: list[Image.Image] = []
    for image_path, mask_path in zip(image_paths, mask_paths):
        rgb = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        if mask.size != rgb.size:
            mask = mask.resize(rgb.size, Image.Resampling.NEAREST)
        rgba = rgb.convert("RGBA")
        rgba.putalpha(mask)
        output.append(pipeline.preprocess_image(rgba).convert("RGB"))
    return output


@torch.no_grad()
def encode_frozen_features(pipeline, images: list[Image.Image]) -> tuple[list[torch.Tensor], torch.Tensor]:
    aggregated_tokens, image_tensor = pipeline.vggt_feat(images)
    raw_image_cond = pipeline.encode_image(image_tensor)
    batch = int(aggregated_tokens[0].shape[0])
    views = int(aggregated_tokens[0].shape[1])
    image_cond = normalize_image_cond(raw_image_cond, batch=batch, views=views)
    return [token.detach() for token in aggregated_tokens], image_cond[:, :, 5:].detach()


def set_bridge_trainable(bridge: nn.Module, last_blocks: int) -> dict[str, int]:
    for parameter in bridge.parameters():
        parameter.requires_grad = False
    blocks = list(getattr(bridge, "cond_blocks", []))
    count = max(0, min(int(last_blocks), len(blocks)))
    selected: dict[str, int] = {}
    for index in range(len(blocks) - count, len(blocks)):
        for parameter in blocks[index].parameters():
            parameter.requires_grad = True
        selected[f"cond_blocks.{index}"] = sum(p.numel() for p in blocks[index].parameters())
    bridge.train(count > 0)
    return selected


class PointPoseSSModel(nn.Module):
    def __init__(
        self,
        *,
        bridge: nn.Module,
        flow: nn.Module,
        physical_condition: PointPoseConditionNet,
        bridge_trainable: bool,
    ) -> None:
        super().__init__()
        self.bridge = bridge
        self.flow = flow
        self.physical_condition = physical_condition
        self.bridge_trainable = bool(bridge_trainable)

    def build_condition(
        self,
        aggregated_tokens: list[torch.Tensor],
        image_cond: torch.Tensor,
        physical_grid: torch.Tensor,
        *,
        physical_scale: float,
        drop_image: bool,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if self.bridge_trainable:
            cond_base = self.bridge(aggregated_tokens, image_cond)
        else:
            with torch.no_grad():
                cond_base = self.bridge(aggregated_tokens, image_cond)
        if drop_image:
            cond_base = torch.zeros_like(cond_base)
        return self.physical_condition(cond_base, physical_grid, scale=float(physical_scale))

    def forward(
        self,
        x_t: torch.Tensor,
        t_tensor: torch.Tensor,
        aggregated_tokens: list[torch.Tensor],
        image_cond: torch.Tensor,
        physical_grid: torch.Tensor,
        *,
        physical_scale: float = 1.0,
        drop_image: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        cond, stats = self.build_condition(
            aggregated_tokens,
            image_cond,
            physical_grid,
            physical_scale=float(physical_scale),
            drop_image=bool(drop_image),
        )
        return self.flow(x_t, t_tensor, cond), stats


def build_models(args: argparse.Namespace, device: torch.device) -> tuple[Any, PointPoseSSModel, nn.Module | None, dict[str, Any]]:
    install_unused_model_stubs()
    pipeline = TrellisVGGTTo3DPipeline.from_pretrained(args.pretrained)
    pipeline._device = device
    pipeline.low_vram = False
    keep_models = {
        "image_cond_model",
        "sparse_structure_vggt_cond",
        "sparse_structure_flow_model",
        "sparse_structure_decoder",
    }
    for name in list(pipeline.models):
        if name not in keep_models:
            del pipeline.models[name]
    for name in ("slat_flow_model", "slat_vggt_cond"):
        if hasattr(pipeline, name):
            delattr(pipeline, name)
    for name in ("camera_head", "point_head", "depth_head", "track_head"):
        if hasattr(pipeline.VGGT_model, name):
            delattr(pipeline.VGGT_model, name)
    for module in pipeline.models.values():
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad = False
    pipeline.VGGT_model.to(device).eval()
    for parameter in pipeline.VGGT_model.parameters():
        parameter.requires_grad = False
    image_model = pipeline.models["image_cond_model"].to(device).eval()
    for parameter in image_model.parameters():
        parameter.requires_grad = False

    bridge = pipeline.models["sparse_structure_vggt_cond"].to(device)
    bridge_selected = set_bridge_trainable(bridge, int(args.bridge_train_last_blocks))
    flow = pipeline.models["sparse_structure_flow_model"].to(device).eval()
    flow_block_count = len(getattr(flow, "blocks", []))
    for parameter in flow.parameters():
        parameter.requires_grad = False
    flow.use_checkpoint = bool(args.gradient_checkpointing)
    for block in getattr(flow, "blocks", []):
        block.use_checkpoint = bool(args.gradient_checkpointing)

    from peft import LoraConfig, get_peft_model

    lora_config = LoraConfig(
        r=int(args.lora_rank),
        lora_alpha=int(args.lora_alpha),
        lora_dropout=0.0,
        target_modules=["to_q", "to_kv", "to_out", "to_qkv"],
    )
    flow = get_peft_model(flow, lora_config)
    flow.train()
    physical = PointPoseConditionNet(
        feature_dim=len(PHYSICAL_FEATURE_NAMES),
        cond_dim=1024,
        hidden_dim=int(args.physical_hidden_dim),
        num_heads=int(args.physical_heads),
    ).to(device)
    model = PointPoseSSModel(
        bridge=bridge,
        flow=flow,
        physical_condition=physical,
        bridge_trainable=bool(bridge_selected),
    ).to(device)
    lora_modules = sorted(
        name for name, module in flow.named_modules() if hasattr(module, "lora_A") or hasattr(module, "lora_B")
    )
    block_count = flow_block_count
    covered_blocks = sorted(
        {
            int(part)
            for name in lora_modules
            for pos, part in enumerate(name.split("."))
            if part.isdigit() and pos > 0 and name.split(".")[pos - 1] == "blocks"
        }
    )
    if not lora_modules or block_count <= 0 or covered_blocks != list(range(block_count)):
        raise RuntimeError(
            f"LoRA coverage failure: modules={len(lora_modules)} blocks={block_count} covered={covered_blocks}"
        )
    decoder = None
    if float(args.occupancy_weight) > 0:
        decoder = pipeline.models["sparse_structure_decoder"].to(device).eval()
        for parameter in decoder.parameters():
            parameter.requires_grad = False
    summary = {
        "bridge_selected": bridge_selected,
        "lora_rank": int(args.lora_rank),
        "lora_alpha": int(args.lora_alpha),
        "physical": physical.metadata(),
        "lora_audits": {
            "matched_module_count": len(lora_modules),
            "matched_module_names": lora_modules,
            "flow_block_count": block_count,
            "covered_flow_blocks": covered_blocks,
            "target_module_counts": dict(
                sorted(Counter(name.rsplit(".", 1)[-1] for name in lora_modules).items())
            ),
            "lora_parameter_count": int(
                sum(p.numel() for name, p in flow.named_parameters() if "lora_" in name)
            ),
        },
        "trainable_params": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "total_params": int(sum(p.numel() for p in model.parameters())),
    }
    return pipeline, model, decoder, summary


def trainable_parameter_audit(model: PointPoseSSModel, pipeline, bridge_blocks: int) -> dict[str, Any]:
    names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    allowed = []
    bad = []
    for name in names:
        if name.startswith("physical_condition.") or "lora_" in name:
            allowed.append(name)
        elif int(bridge_blocks) > 0 and name.startswith("bridge.cond_blocks."):
            allowed.append(name)
        else:
            bad.append(name)
    report = {
        "trainable_names": names,
        "unexpected_trainable_names": bad,
        "vggt_trainable": int(sum(p.numel() for p in pipeline.VGGT_model.parameters() if p.requires_grad)),
        "image_encoder_trainable": int(
            sum(p.numel() for p in pipeline.models["image_cond_model"].parameters() if p.requires_grad)
        ),
        "decoder_trainable": int(
            sum(p.numel() for p in pipeline.models["sparse_structure_decoder"].parameters() if p.requires_grad)
        ),
        "bridge_trainable": int(sum(p.numel() for p in model.bridge.parameters() if p.requires_grad)),
        "original_flow_trainable": int(
            sum(p.numel() for name, p in model.flow.named_parameters() if p.requires_grad and "lora_" not in name)
        ),
        "physical_trainable": int(sum(p.numel() for p in model.physical_condition.parameters() if p.requires_grad)),
        "lora_trainable": int(sum(p.numel() for name, p in model.flow.named_parameters() if p.requires_grad and "lora_" in name)),
    }
    expected_bridge = int(bridge_blocks) > 0
    if (
        bad
        or report["vggt_trainable"]
        or report["image_encoder_trainable"]
        or report["decoder_trainable"]
        or report["original_flow_trainable"]
        or not report["physical_trainable"]
        or not report["lora_trainable"]
        or (bool(report["bridge_trainable"]) != expected_bridge)
    ):
        raise RuntimeError(f"trainable parameter whitelist failed: {report}")
    return report


def validate_checkpoint_config(checkpoint: dict[str, Any], args: argparse.Namespace, *, weights_only: bool) -> None:
    saved_args = checkpoint.get("args", {})
    expected = {
        "pretrained": str(args.pretrained),
        "lora_rank": int(args.lora_rank),
        "lora_alpha": int(args.lora_alpha),
        "physical_hidden_dim": int(args.physical_hidden_dim),
        "physical_heads": int(args.physical_heads),
    }
    if not weights_only:
        expected["amp_dtype"] = str(args.amp_dtype)
    mismatches = {
        key: {"checkpoint": saved_args.get(key), "current": value}
        for key, value in expected.items()
        if saved_args.get(key) != value
    }
    saved_features = tuple(
        checkpoint.get("model_summary", {}).get("physical", {}).get("feature_names", [])
    )
    if saved_features != PHYSICAL_FEATURE_NAMES:
        mismatches["feature_names"] = {"checkpoint": saved_features, "current": PHYSICAL_FEATURE_NAMES}
    target_schema = checkpoint.get("model_summary", {}).get("target_schema", {})
    expected_target = {"latent_shape": [8, 16, 16, 16], "resolution": 16, "channels": 8}
    for key, value in expected_target.items():
        if target_schema.get(key) != value:
            mismatches[f"target_schema.{key}"] = {
                "checkpoint": target_schema.get(key),
                "current": value,
            }
    if not weights_only and int(saved_args.get("bridge_train_last_blocks", 0)) != int(args.bridge_train_last_blocks):
        mismatches["bridge_train_last_blocks"] = {
            "checkpoint": saved_args.get("bridge_train_last_blocks"),
            "current": int(args.bridge_train_last_blocks),
        }
    if mismatches:
        raise RuntimeError(f"checkpoint configuration mismatch: {mismatches}")


@torch.no_grad()
def stock_condition_equivalence_audit(
    pipeline,
    model: PointPoseSSModel,
    batch: dict[str, Any],
    device: torch.device,
    *,
    expect_zero_init: bool,
) -> dict[str, Any]:
    images = rgba_images(batch["image_paths"], batch["mask_paths"], pipeline)
    aggregated, image_cond = encode_frozen_features(pipeline, images)
    physical = batch["physical_grid"].unsqueeze(0).to(device=device, dtype=torch.float32)
    direct = model.bridge(aggregated, image_cond)
    native = pipeline.get_ss_cond(image_cond, aggregated, num_samples=1)
    # Reuse one bridge output so this audit isolates the residual branch instead
    # of measuring low-precision differences between repeated bridge forwards.
    scale_zero, _ = model.physical_condition(direct, physical, scale=0.0)
    scale_one, _ = model.physical_condition(direct, physical, scale=1.0)
    direct_native_diff = (direct.float() - native["cond"].float()).abs()
    direct_native_atol = 1.0 / 32.0
    direct_native_rtol = 1.0e-3
    direct_native_close = bool(
        torch.allclose(
            direct.float(),
            native["cond"].float(),
            atol=direct_native_atol,
            rtol=direct_native_rtol,
        )
    )
    report = {
        "direct_vs_native_max_abs": float(direct_native_diff.max().item()),
        "direct_vs_native_mean_abs": float(direct_native_diff.mean().item()),
        "direct_vs_native_atol": direct_native_atol,
        "direct_vs_native_rtol": direct_native_rtol,
        "direct_vs_native_close": direct_native_close,
        "native_neg_abs_max": float(native["neg_cond"].abs().max().item()),
        "scale0_vs_direct_max_abs": float((scale_zero - direct).abs().max().item()),
        "scale1_vs_direct_max_abs": float((scale_one - direct).abs().max().item()),
        "expect_zero_init": bool(expect_zero_init),
    }
    if not direct_native_close or report["native_neg_abs_max"] != 0.0 or report["scale0_vs_direct_max_abs"] != 0.0:
        raise RuntimeError(f"stock condition equivalence failed: {report}")
    if expect_zero_init and report["scale1_vs_direct_max_abs"] != 0.0:
        raise RuntimeError(f"zero-init physical condition equivalence failed: {report}")
    return report


def gradient_group_norms(model: PointPoseSSModel) -> dict[str, float]:
    groups = {
        "physical_encoder": "physical_condition.grid_encoder.",
        "physical_cross_attention": "physical_condition.cross_attn.",
        "physical_output_projection": "physical_condition.output_proj.",
    }
    output: dict[str, float] = {}
    named = list(model.named_parameters())
    for label, prefix in groups.items():
        square = sum(
            float(parameter.grad.detach().float().square().sum().item())
            for name, parameter in named
            if name.startswith(prefix) and parameter.grad is not None
        )
        output[label] = square**0.5
    square = sum(
        float(parameter.grad.detach().float().square().sum().item())
        for name, parameter in named
        if "lora_" in name and parameter.grad is not None
    )
    output["flow_lora"] = square**0.5
    return output


def gradients_finite(parameters: list[nn.Parameter]) -> bool:
    return all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all().item())
        for parameter in parameters
    )


def parameters_finite(parameters: list[nn.Parameter]) -> bool:
    return all(bool(torch.isfinite(parameter).all().item()) for parameter in parameters)


def tensors_finite(values: list[torch.Tensor]) -> bool:
    return all(bool(torch.isfinite(value.detach()).all().item()) for value in values)


def finite_tree(value: Any) -> bool:
    if torch.is_tensor(value):
        return bool(torch.isfinite(value).all().item())
    if isinstance(value, dict):
        return all(finite_tree(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_tree(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def optimizer_state_finite(optimizer: torch.optim.Optimizer) -> bool:
    return finite_tree(optimizer.state)


def distributed_all_true(value: bool, device: torch.device, world_size: int) -> bool:
    flag = torch.tensor(int(bool(value)), device=device, dtype=torch.int32)
    if world_size > 1:
        dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    return bool(flag.item())


def sample_t(schedule: str, device: torch.device) -> float:
    if schedule == "uniform":
        return float(torch.rand((), device=device).item())
    if schedule == "logit_normal":
        return float(torch.sigmoid(torch.randn((), device=device)).item())
    if schedule == "high_t_mix":
        if float(torch.rand((), device=device).item()) < 0.35:
            return float(torch.empty((), device=device).uniform_(0.85, 1.0).item())
        return float(torch.rand((), device=device).item())
    raise ValueError(f"unknown t schedule: {schedule}")


def target_occupancy(coords: torch.Tensor, device: torch.device) -> torch.Tensor:
    target = torch.zeros((1, 1, 64, 64, 64), device=device, dtype=torch.float32)
    xyz = coords.to(device=device, dtype=torch.long)
    valid = ((xyz >= 0) & (xyz < 64)).all(dim=1)
    xyz = xyz[valid]
    if xyz.numel():
        target[0, 0, xyz[:, 0], xyz[:, 1], xyz[:, 2]] = 1.0
    return target


def distributed_mean(value: torch.Tensor, world_size: int) -> float:
    detached = value.detach().float()
    if world_size > 1:
        dist.all_reduce(detached, op=dist.ReduceOp.SUM)
        detached /= float(world_size)
    return float(detached.cpu().item())


def save_checkpoint(
    path: Path,
    *,
    model: PointPoseSSModel,
    optimizer: torch.optim.Optimizer,
    scaler,
    step: int,
    args: argparse.Namespace,
    summary: dict[str, Any],
) -> None:
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters_finite(trainable_parameters):
        raise RuntimeError(f"refusing to save non-finite model parameters: {path}")
    if not optimizer_state_finite(optimizer):
        raise RuntimeError(f"refusing to save non-finite optimizer state: {path}")
    scaler_state = scaler.state_dict()
    if not finite_tree(scaler_state):
        raise RuntimeError(f"refusing to save non-finite scaler state: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "reconvggt.pointpose_ss_lora.v1",
            "step": int(step),
            "model_trainable_state": trainable_state_dict(model),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler_state,
            "args": vars(args),
            "model_summary": summary,
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Joint PointPose condition + ReconViaGen SS Flow LoRA training.")
    parser.add_argument("--cache_manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--pretrained", default="Stable-X/trellis-vggt-v0-2")
    parser.add_argument("--indices", default="all")
    parser.add_argument("--resume", default="")
    parser.add_argument("--resume_weights_only", action="store_true")
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument("--save_every", type=int, default=100)
    parser.add_argument("--log_every", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=2.0e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--amp_dtype", choices=["fp16", "bf16", "none"], default="fp16")
    parser.add_argument("--amp_init_scale", type=float, default=16384.0)
    parser.add_argument("--nonfinite_policy", choices=["error", "skip"], default="error")
    parser.add_argument("--max_nonfinite_attempts", type=int, default=4)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--physical_hidden_dim", type=int, default=256)
    parser.add_argument("--physical_heads", type=int, default=8)
    parser.add_argument("--physical_scale", type=float, default=1.0)
    parser.add_argument("--bridge_train_last_blocks", type=int, default=0)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--t_schedule", choices=["uniform", "logit_normal", "high_t_mix"], default="uniform")
    parser.add_argument("--drop_all_prob", type=float, default=0.1)
    parser.add_argument("--physical_drop_prob", type=float, default=0.1)
    parser.add_argument("--occupancy_weight", type=float, default=0.0)
    parser.add_argument("--occupancy_every", type=int, default=4)
    args = parser.parse_args()

    if args.max_steps <= 0:
        raise ValueError("--max_steps must be positive")
    if args.save_every <= 0 or args.log_every <= 0 or args.grad_accum <= 0:
        raise ValueError("--save_every, --log_every, and --grad_accum must be positive")
    if args.grad_clip <= 0 or args.amp_init_scale <= 0:
        raise ValueError("--grad_clip and --amp_init_scale must be positive")
    if args.max_nonfinite_attempts < 0:
        raise ValueError("--max_nonfinite_attempts must be non-negative")

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
        time.sleep(min(rank * 1.5, 12.0))

    dataset = PointPoseCacheDataset(args.cache_manifest, indices=args.indices)
    schema_probe = dataset[0]
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=int(args.seed))
    loader = DataLoader(
        dataset,
        batch_size=1,
        sampler=sampler,
        num_workers=int(args.num_workers),
        collate_fn=collate_one,
        pin_memory=True,
    )
    pipeline, model, decoder, model_summary = build_models(args, device)
    model_summary["target_schema"] = {
        "latent_shape": list(schema_probe["target"].shape),
        "latent_dtype": str(schema_probe["target"].dtype),
        "target_coord_shape": list(schema_probe["target_coords"].shape),
        "resolution": 16,
        "channels": 8,
    }
    model_summary["trainable_parameter_audit"] = trainable_parameter_audit(
        model, pipeline, int(args.bridge_train_last_blocks)
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
        validate_checkpoint_config(checkpoint, args, weights_only=bool(args.resume_weights_only))
        load_info = load_partial_state(
            model,
            checkpoint["model_trainable_state"],
            require_all_trainable=True,
            allowed_missing_prefixes=("bridge.",) if args.resume_weights_only else (),
        )
        if "optimizer" in checkpoint and not args.resume_weights_only:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if "scaler" in checkpoint and not args.resume_weights_only:
            scaler.load_state_dict(checkpoint["scaler"])
        start_step = 0 if args.resume_weights_only else int(checkpoint.get("step", 0))
        if rank == 0:
            print(f"[pointpose_train] resumed step={start_step} loaded={len(load_info['loaded'])}", flush=True)

    stock_audit = stock_condition_equivalence_audit(
        pipeline,
        model,
        dataset[0],
        device,
        expect_zero_init=not bool(args.resume),
    )
    model_summary["stock_condition_equivalence"] = stock_audit

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
    history: list[dict[str, Any]] = []
    global_step = start_step
    micro_step = global_step * max(1, int(args.grad_accum))
    optimizer.zero_grad(set_to_none=True)
    epoch = 0
    wall_start = time.time()
    nonfinite_attempts = 0
    applied_updates = 0
    while global_step < int(args.max_steps):
        sampler.set_epoch(epoch)
        for batch in loader:
            if global_step >= int(args.max_steps):
                break
            images = rgba_images(batch["image_paths"], batch["mask_paths"], pipeline)
            aggregated, image_cond = encode_frozen_features(pipeline, images)
            physical = batch["physical_grid"].unsqueeze(0).to(device=device, dtype=torch.float32, non_blocking=True)
            target = batch["target"].unsqueeze(0).to(device=device, dtype=torch.float32, non_blocking=True)
            noise = torch.randn_like(target)
            t_model = sample_t(str(args.t_schedule), device)
            x_t, gt_v = pipeline.sparse_structure_sampler._get_model_gt(target, t_model, noise)
            t_tensor = torch.full((1,), 1000.0 * t_model, device=device, dtype=torch.float32)
            drop_all = random.random() < float(args.drop_all_prob)
            drop_physical = drop_all or random.random() < float(args.physical_drop_prob)
            physical_scale = 0.0 if drop_physical else float(args.physical_scale)

            sync_step = ((micro_step + 1) % max(1, int(args.grad_accum))) == 0
            sync_context = (
                wrapped.no_sync() if world_size > 1 and not sync_step else torch.enable_grad()
            )
            with sync_context:
                with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
                    pred_v, condition_stats = wrapped(
                        x_t,
                        t_tensor,
                        aggregated,
                        image_cond,
                        physical,
                        physical_scale=physical_scale,
                        drop_image=drop_all,
                    )
                    flow_loss = F.mse_loss(pred_v.float(), gt_v.float())
                    occupancy_loss = pred_v.new_zeros((), dtype=torch.float32)
                    if (
                        decoder is not None
                        and float(args.occupancy_weight) > 0
                        and global_step % max(1, int(args.occupancy_every)) == 0
                    ):
                        pred_x0 = pipeline.sparse_structure_sampler._pred_to_xstart(x_t, t_model, pred_v)
                        logits = decoder(pred_x0.to(dtype=next(decoder.parameters()).dtype)).float()
                        target_occ = target_occupancy(batch["target_coords"], device)
                        positive = target_occ.sum().clamp_min(1.0)
                        negative = target_occ.numel() - positive
                        pos_weight = (negative / positive).clamp(1.0, 32.0)
                        occupancy_loss = F.binary_cross_entropy_with_logits(
                            logits,
                            target_occ,
                            pos_weight=pos_weight,
                        )
                    loss = flow_loss + float(args.occupancy_weight) * occupancy_loss
                    scaled_loss = loss / float(max(1, int(args.grad_accum)))
                scaler.scale(scaled_loss).backward()

            if sync_step:
                scaler.unscale_(optimizer)
                grad_norms = gradient_group_norms(model)
                diagnostic_values = [
                    loss,
                    flow_loss,
                    occupancy_loss,
                    condition_stats["delta_rms"],
                    condition_stats["delta_abs_max"],
                    condition_stats["physical_token_rms"],
                    condition_stats["attended_rms"],
                ]
                forward_finite = distributed_all_true(
                    tensors_finite(diagnostic_values), device, world_size
                )
                gradient_finite = distributed_all_true(
                    gradients_finite(trainable), device, world_size
                )
                update_finite = forward_finite and gradient_finite
                scaler_before = float(scaler.get_scale()) if scaler.is_enabled() else None
                clip_total_norm = None
                optimizer_step_applied = False

                if update_finite:
                    clip_total_norm_tensor = torch.nn.utils.clip_grad_norm_(
                        trainable,
                        float(args.grad_clip),
                        error_if_nonfinite=True,
                    )
                    clip_total_norm = float(clip_total_norm_tensor.detach().float().cpu().item())
                    scaler.step(optimizer)
                    scaler.update()
                    scaler_after = float(scaler.get_scale()) if scaler.is_enabled() else None
                    optimizer_step_applied = not scaler.is_enabled() or scaler_after >= scaler_before
                    if optimizer_step_applied:
                        post_step_finite = distributed_all_true(
                            parameters_finite(trainable) and optimizer_state_finite(optimizer),
                            device,
                            world_size,
                        )
                        if not post_step_finite:
                            raise RuntimeError(
                                "optimizer step produced non-finite parameters or optimizer state"
                            )
                else:
                    nonfinite_attempts += 1
                    if scaler.is_enabled():
                        # Update scale from unscale_'s found_inf state without allowing an optimizer step.
                        scaler.update()
                    scaler_after = float(scaler.get_scale()) if scaler.is_enabled() else None

                optimizer.zero_grad(set_to_none=True)
                if optimizer_step_applied:
                    global_step += 1
                    applied_updates += 1

                should_log = (
                    not update_finite
                    or global_step == 1
                    or (optimizer_step_applied and global_step % int(args.log_every) == 0)
                    or global_step == int(args.max_steps)
                )
                if should_log:
                    row = {
                        "step": int(global_step),
                        "micro_step": int(micro_step + 1),
                        "flow_loss": distributed_mean(flow_loss, world_size),
                        "occupancy_loss": distributed_mean(occupancy_loss, world_size),
                        "delta_rms": distributed_mean(condition_stats["delta_rms"], world_size),
                        "delta_abs_max": distributed_mean(condition_stats["delta_abs_max"], world_size),
                        "physical_token_rms": distributed_mean(condition_stats["physical_token_rms"], world_size),
                        "attended_rms": distributed_mean(condition_stats["attended_rms"], world_size),
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
                        "drop_all": int(drop_all),
                        "drop_physical": int(drop_physical),
                        "elapsed_seconds": float(time.time() - wall_start),
                    }
                    if rank == 0:
                        history.append(row)
                        print(f"[pointpose_train] {json.dumps(row, ensure_ascii=False)}", flush=True)

                if not update_finite:
                    message = (
                        f"non-finite update attempt={nonfinite_attempts} micro_step={micro_step + 1} "
                        f"amp={args.amp_dtype} scaler={scaler_before}->{scaler_after}"
                    )
                    if args.nonfinite_policy == "error" or nonfinite_attempts > int(args.max_nonfinite_attempts):
                        raise RuntimeError(message)
                    if rank == 0:
                        print(f"[pointpose_train] skipped {message}", flush=True)

                if optimizer_step_applied and rank == 0 and (
                    global_step % int(args.save_every) == 0 or global_step == int(args.max_steps)
                ):
                    save_checkpoint(
                        output_dir / "checkpoints" / f"step_{global_step:06d}.pt",
                        model=model,
                        optimizer=optimizer,
                        scaler=scaler,
                        step=global_step,
                        args=args,
                        summary=model_summary,
                    )
                    save_checkpoint(
                        output_dir / "checkpoints" / "last.pt",
                        model=model,
                        optimizer=optimizer,
                        scaler=scaler,
                        step=global_step,
                        args=args,
                        summary=model_summary,
                    )
            micro_step += 1
        epoch += 1

    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        report = {
            "args": vars(args),
            "world_size": world_size,
            "per_rank_batch_size": 1,
            "effective_batch_size": world_size * max(1, int(args.grad_accum)),
            "applied_optimizer_updates": int(applied_updates),
            "nonfinite_attempts": int(nonfinite_attempts),
            "completed_global_step": int(global_step),
            "start_global_step": int(start_step),
            "dataset_size": len(dataset),
            "unique_object_count": len({str(row.get("object_uid", "")) for row in dataset.samples}),
            "drop_all_prob": float(args.drop_all_prob),
            "model": model_summary,
            "history": history,
        }
        (output_dir / "train_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
