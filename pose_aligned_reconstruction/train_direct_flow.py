#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import nullcontext
import datetime
import gc
import json
import math
import os
from pathlib import Path
import random
import sys
import time
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
from torch.utils.data import DataLoader, DistributedSampler, Sampler


TRACKER_ROOT = Path(__file__).resolve().parents[1]
RECONVIAGEN_ROOT = TRACKER_ROOT / "ReconViaGen"
VGGT_ROOT = RECONVIAGEN_ROOT / "wheels" / "vggt"
for path in (TRACKER_ROOT, RECONVIAGEN_ROOT, VGGT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from trellis.pipelines import TrellisVGGTTo3DPipeline  # noqa: E402

from ar_ss_flow.local_pose_lifting_flow import (  # noqa: E402
    PoseLiftingCacheDataset,
    collate_one,
)
from pose_aligned_reconstruction.direct_flow import (  # noqa: E402
    DIRECT_CORRUPTION_MODES,
    DIRECT_FLOW_VERSION,
    DirectPhysicalFlowModel,
    DirectViewTokenEncoder,
    lifting_cache_identity,
    load_frozen_correspondence_head,
    make_direct_evidence_bundle,
    null_evidence_like,
    parse_optional_csv,
    validate_n3_checkpoint,
)
from reconvggt_ar_adapter_a.pointpose_ss_condition import (  # noqa: E402
    load_partial_state,
    lora_disabled,
    trainable_state_dict,
)
from reconvggt_ar_adapter_a.train_pointpose_ss_lora import (  # noqa: E402
    distributed_all_true,
    distributed_mean,
    finite_tree,
    gradients_finite,
    install_unused_model_stubs,
    optimizer_state_finite,
    parameters_finite,
    sample_t,
    target_occupancy,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train N3-bound per-view visual/pose/depth/point tokens inside the "
            "ReconViaGen SS Flow, together with stock-preserving Flow LoRA."
        )
    )
    parser.add_argument("--cache_manifest", required=True)
    parser.add_argument("--correspondence_checkpoint", required=True)
    parser.add_argument("--n3_report", required=True)
    parser.add_argument(
        "--admission_report",
        default="",
        help="Optional full-data admission report; bind it to this cache at startup.",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--pretrained", default="Stable-X/trellis-vggt-v0-2")
    parser.add_argument("--indices", default="all")
    parser.add_argument("--resume", default="")
    parser.add_argument("--max_steps", type=int, default=300)
    parser.add_argument("--save_every", type=int, default=50)
    parser.add_argument("--log_every", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1.0e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument(
        "--amp_dtype", choices=("bf16", "fp16", "none"), default="bf16"
    )
    parser.add_argument("--amp_init_scale", type=float, default=8192.0)
    parser.add_argument(
        "--nonfinite_policy", choices=("error", "skip"), default="error"
    )
    parser.add_argument("--max_nonfinite_attempts", type=int, default=0)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--physical_hidden_dim", type=int, default=128)
    parser.add_argument("--physical_scale", type=float, default=1.0)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument(
        "--training_profile",
        choices=("performance", "weak_mechanism", "custom"),
        default="performance",
    )
    parser.add_argument(
        "--sampling_mode",
        choices=("object_balanced", "sequence"),
        default="object_balanced",
    )
    parser.add_argument(
        "--t_schedule",
        choices=("uniform", "logit_normal", "high_t_mix"),
        default="uniform",
    )
    parser.add_argument(
        "--corruption_modes",
        default=None,
    )
    parser.add_argument("--flow_weight", type=float, default=1.0)
    parser.add_argument("--occupancy_weight", type=float, default=0.001)
    parser.add_argument("--occupancy_every", type=int, default=4)
    parser.add_argument("--wrong_stock_weight", type=float, default=None)
    parser.add_argument("--correct_gain_weight", type=float, default=None)
    parser.add_argument("--correct_gain_margin", type=float, default=0.0)
    parser.add_argument("--rank_weight", type=float, default=None)
    parser.add_argument("--rank_margin", type=float, default=0.0)
    parser.add_argument("--delta_norm_weight", type=float, default=0.001)
    parser.add_argument(
        "--loss_gradient_audit_every",
        type=int,
        default=50,
        help=(
            "Run isolated Flow/occupancy gradient audits on step 1 and every N "
            "optimizer updates; 0 disables the extra backward passes."
        ),
    )
    return parser.parse_args()


def resolve_training_profile(args: argparse.Namespace) -> tuple[str, ...]:
    defaults = {
        "performance": {
            "corruption_modes": "",
            "wrong_stock_weight": 0.0,
            "correct_gain_weight": 0.0,
            "rank_weight": 0.0,
        },
        "weak_mechanism": {
            "corruption_modes": "pose_cyclic1,depth_view_cyclic1,visual_view_cyclic1",
            "wrong_stock_weight": 0.0,
            "correct_gain_weight": 0.0,
            "rank_weight": 0.01,
        },
        "custom": {
            "corruption_modes": "pose_cyclic1,depth_view_cyclic1,visual_view_cyclic1",
            "wrong_stock_weight": 0.0,
            "correct_gain_weight": 0.0,
            "rank_weight": 0.01,
        },
    }[str(args.training_profile)]
    for name, value in defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, value)
    modes = parse_optional_csv(args.corruption_modes)
    if str(args.training_profile) == "performance":
        mechanism_weights = (
            float(args.wrong_stock_weight),
            float(args.correct_gain_weight),
            float(args.rank_weight),
        )
        if modes or any(weight != 0.0 for weight in mechanism_weights):
            raise ValueError(
                "performance profile requires empty corruption modes and zero "
                "wrong-stock/correct-gain/rank weights; use custom to override"
            )
    return modes


def validate_args(args: argparse.Namespace) -> tuple[str, ...]:
    modes = resolve_training_profile(args)
    positive = (
        "max_steps",
        "save_every",
        "log_every",
        "grad_accum",
        "grad_clip",
        "amp_init_scale",
        "lora_rank",
        "lora_alpha",
        "physical_hidden_dim",
        "occupancy_every",
    )
    invalid_positive = [name for name in positive if float(getattr(args, name)) <= 0]
    if invalid_positive:
        raise ValueError(f"arguments must be positive: {invalid_positive}")
    if int(args.max_nonfinite_attempts) < 0:
        raise ValueError("max_nonfinite_attempts must be non-negative")
    if int(args.loss_gradient_audit_every) < 0:
        raise ValueError("loss_gradient_audit_every must be non-negative")
    nonnegative = (
        "flow_weight",
        "occupancy_weight",
        "wrong_stock_weight",
        "correct_gain_weight",
        "rank_weight",
        "delta_norm_weight",
    )
    invalid_weights = [name for name in nonnegative if float(getattr(args, name)) < 0]
    if invalid_weights:
        raise ValueError(f"loss weights must be non-negative: {invalid_weights}")
    bad_modes = [mode for mode in modes if mode not in DIRECT_CORRUPTION_MODES]
    if bad_modes:
        raise ValueError(f"unsupported corruption modes={bad_modes}")
    mechanism_weight = max(
        float(args.wrong_stock_weight),
        float(args.correct_gain_weight),
        float(args.rank_weight),
    )
    if mechanism_weight > 0.0 and not modes:
        raise ValueError("non-zero mechanism losses require corruption modes")
    return modes


class ObjectBalancedDistributedSampler(Sampler[int]):
    """Draw one sequence per object per epoch, then shard objects across ranks."""

    def __init__(
        self,
        rows: list[dict[str, Any]],
        *,
        num_replicas: int,
        rank: int,
        seed: int,
    ) -> None:
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.seed = int(seed)
        self.epoch = 0
        self.by_object: dict[str, list[int]] = {}
        for index, row in enumerate(rows):
            object_uid = str(row.get("object_uid", row.get("uid", "")))
            if not object_uid:
                raise ValueError("training cache contains an empty object UID")
            self.by_object.setdefault(object_uid, []).append(index)
        if not self.by_object:
            raise ValueError("object-balanced sampler received no objects")
        self.num_samples = math.ceil(len(self.by_object) / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        object_uids = sorted(self.by_object)
        order = torch.randperm(len(object_uids), generator=generator).tolist()
        selected = []
        for object_index in order:
            candidates = self.by_object[object_uids[object_index]]
            choice = int(
                torch.randint(len(candidates), (1,), generator=generator).item()
            )
            selected.append(candidates[choice])
        if len(selected) < self.total_size:
            selected.extend(selected[: self.total_size - len(selected)])
        selected = selected[self.rank : self.total_size : self.num_replicas]
        return iter(selected)

    def __len__(self) -> int:
        return self.num_samples


def _flow_core(flow: nn.Module) -> nn.Module:
    base = getattr(flow, "base_model", None)
    core = getattr(base, "model", None)
    return core if isinstance(core, nn.Module) else flow


def build_direct_components(
    *,
    pretrained: str,
    visual_channels: int,
    physical_hidden_dim: int,
    lora_rank: int,
    lora_alpha: int,
    gradient_checkpointing: bool,
    need_decoder: bool,
    device: torch.device,
    retain_pipeline: bool = False,
) -> tuple[Any, ...]:
    install_unused_model_stubs()
    pipeline = TrellisVGGTTo3DPipeline.from_pretrained(pretrained)
    pipeline._device = device
    pipeline.low_vram = False
    sampler = pipeline.sparse_structure_sampler
    sampler_params = dict(pipeline.sparse_structure_sampler_params)
    flow = pipeline.models["sparse_structure_flow_model"].to(device).eval()
    decoder = (
        pipeline.models["sparse_structure_decoder"].to(device).eval()
        if need_decoder
        else None
    )
    for parameter in flow.parameters():
        parameter.requires_grad_(False)
    if decoder is not None:
        for parameter in decoder.parameters():
            parameter.requires_grad_(False)
    flow.use_checkpoint = bool(gradient_checkpointing)
    for block in flow.blocks:
        block.use_checkpoint = bool(gradient_checkpointing)

    from peft import LoraConfig, get_peft_model

    flow_block_count = len(flow.blocks)
    flow = get_peft_model(
        flow,
        LoraConfig(
            r=int(lora_rank),
            lora_alpha=int(lora_alpha),
            lora_dropout=0.0,
            bias="none",
            target_modules=["to_q", "to_kv", "to_out", "to_qkv"],
        ),
    )
    flow.train()
    core = _flow_core(flow)
    encoder = DirectViewTokenEncoder(
        visual_channels=int(visual_channels),
        flow_channels=int(core.model_channels),
        hidden_dim=int(physical_hidden_dim),
    ).to(device)
    model = DirectPhysicalFlowModel(flow, encoder).to(device)

    lora_modules = sorted(
        name
        for name, module in flow.named_modules()
        if hasattr(module, "lora_A") or hasattr(module, "lora_B")
    )
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
    if not lora_modules or covered_blocks != list(range(flow_block_count)):
        raise RuntimeError(
            "Flow LoRA coverage failed: "
            f"modules={len(lora_modules)} covered={covered_blocks}"
        )
    trainable_names = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    unexpected = [
        name
        for name in trainable_names
        if not name.startswith("physical_encoder.") and "lora_" not in name
    ]
    if unexpected or not trainable_names:
        raise RuntimeError(f"direct Flow trainable whitelist failed: {unexpected}")
    summary = {
        "stage": "large-data direct view-identity physical SS Flow",
        "format": DIRECT_FLOW_VERSION,
        "physical_encoder": encoder.metadata(),
        "injection_location": (
            "same-voxel 16^3 tokens after input+position before flow block 0"
        ),
        "flow_lora": {
            "rank": int(lora_rank),
            "alpha": int(lora_alpha),
            "matched_module_count": len(lora_modules),
            "flow_block_count": flow_block_count,
            "covered_flow_blocks": covered_blocks,
            "target_module_counts": dict(
                sorted(Counter(name.rsplit(".", 1)[-1] for name in lora_modules).items())
            ),
            "parameter_count": int(
                sum(
                    parameter.numel()
                    for name, parameter in flow.named_parameters()
                    if "lora_" in name
                )
            ),
        },
        "stock_fallback": (
            "physical off/null disables both token injection and every Flow LoRA adapter"
        ),
        "frozen": [
            "N3 correspondence head",
            "stock SS Flow base",
            "SS decoder",
            "cached VGGT/DINO/stock condition",
        ],
        "trainable_parameter_name_count": len(trainable_names),
        "trainable_whitelist": ["physical_encoder.*", "flow.*.lora_[AB].*"],
        "trainable_parameter_count": int(
            sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        ),
        "total_parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
    }
    result = (sampler, model, decoder, summary, sampler_params)
    if retain_pipeline:
        # Full Mesh evaluation needs the frozen downstream image/SLAT/mesh
        # modules from this exact pretrained pipeline.  Keeping this opt-in
        # avoids loading a second pipeline while preserving the training and
        # teacher-evaluation memory behaviour.
        return (*result, pipeline)
    del pipeline
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def gradient_group_norms(model: DirectPhysicalFlowModel) -> dict[str, float]:
    names = list(model.named_parameters())
    groups = {
        "physical_visual": ("physical_encoder.visual_",),
        "physical_geometry": ("physical_encoder.geometry_",),
        "physical_pair": ("physical_encoder.view_fusion.", "physical_encoder.pair_"),
        "physical_metadata": ("physical_encoder.metadata_",),
        "physical_local": ("physical_encoder.local_fusion.",),
        "physical_output": ("physical_encoder.output.",),
    }
    output = {}
    for label, prefixes in groups.items():
        square = sum(
            float(parameter.grad.detach().float().square().sum().item())
            for name, parameter in names
            if parameter.grad is not None
            and any(name.startswith(prefix) for prefix in prefixes)
        )
        output[label] = square**0.5
    lora_square = sum(
        float(parameter.grad.detach().float().square().sum().item())
        for name, parameter in names
        if parameter.grad is not None and "lora_" in name
    )
    output["flow_lora"] = lora_square**0.5
    return output


def output_gradient_norm(
    loss: torch.Tensor,
    prediction: torch.Tensor,
) -> float:
    gradient = torch.autograd.grad(
        loss,
        prediction,
        retain_graph=True,
        allow_unused=True,
    )[0]
    if gradient is None:
        return 0.0
    return float(gradient.detach().float().square().sum().sqrt().item())


def t_bucket(t_value: float) -> str:
    lower = min(4, max(0, int(float(t_value) * 5.0)))
    return f"[{lower / 5.0:.1f},{(lower + 1) / 5.0:.1f})"


def exposure_summary(
    rows: list[dict[str, Any]], counts: Counter[str]
) -> dict[str, Any]:
    source_counts = Counter(
        str(row.get("object_uid", row.get("uid", ""))) for row in rows
    )
    values = sorted(int(value) for value in counts.values())
    source_values = sorted(int(value) for value in source_counts.values())

    def compact(items: list[int]) -> dict[str, float | int]:
        if not items:
            return {"count": 0, "min": 0, "median": 0.0, "max": 0}
        return {
            "count": len(items),
            "min": min(items),
            "median": float(np.median(items)),
            "max": max(items),
        }

    return {
        "unique_objects_seen": len(counts),
        "object_exposure": compact(values),
        "source_sequences_per_object": compact(source_values),
        "per_object_exposure": dict(sorted(counts.items())),
    }


def occupancy_bucket_summary(
    rows: list[dict[str, float | str]],
) -> dict[str, Any]:
    buckets: dict[str, list[dict[str, float | str]]] = defaultdict(list)
    for row in rows:
        buckets[str(row["bucket"])].append(row)

    def stats(values: list[float]) -> dict[str, float | int]:
        if not values:
            return {"count": 0, "mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
        return {
            "count": len(values),
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "min": min(values),
            "max": max(values),
        }

    return {
        bucket: {
            "raw_loss": stats([float(row["raw_loss"]) for row in items]),
            "weighted_loss": stats(
                [float(row["weighted_loss"]) for row in items]
            ),
        }
        for bucket, items in sorted(buckets.items())
    }


def tensors_finite(values: list[torch.Tensor]) -> bool:
    return all(bool(torch.isfinite(value.detach()).all().item()) for value in values)


def save_checkpoint(
    path: Path,
    *,
    model: DirectPhysicalFlowModel,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    step: int,
    micro_step: int,
    args: argparse.Namespace,
    summary: dict[str, Any],
    history: list[dict[str, Any]],
) -> None:
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters_finite(trainable):
        raise RuntimeError(f"refusing to save non-finite model: {path}")
    if not optimizer_state_finite(optimizer) or not finite_tree(scaler.state_dict()):
        raise RuntimeError(f"refusing to save non-finite optimizer/scaler: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": DIRECT_FLOW_VERSION,
            "step": int(step),
            "micro_step": int(micro_step),
            "model_trainable_state": trainable_state_dict(model),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "args": vars(args),
            "model_summary": summary,
            "history": history,
        },
        path,
    )


def validate_resume(
    checkpoint: dict[str, Any],
    args: argparse.Namespace,
    current_summary: dict[str, Any],
) -> None:
    if checkpoint.get("format") != DIRECT_FLOW_VERSION:
        raise ValueError(f"unexpected resume format={checkpoint.get('format')!r}")
    saved = checkpoint.get("args", {})
    fields = (
        "pretrained",
        "cache_manifest",
        "correspondence_checkpoint",
        "n3_report",
        "lora_rank",
        "lora_alpha",
        "physical_hidden_dim",
        "training_profile",
        "sampling_mode",
        "corruption_modes",
        "flow_weight",
        "occupancy_weight",
        "occupancy_every",
        "wrong_stock_weight",
        "correct_gain_weight",
        "rank_weight",
        "delta_norm_weight",
    )
    mismatch = {
        name: (saved.get(name), getattr(args, name))
        for name in fields
        if str(saved.get(name)) != str(getattr(args, name))
    }
    if mismatch:
        raise ValueError(f"resume protocol mismatch={mismatch}")
    saved_summary = checkpoint.get("model_summary", {})
    for name in (
        "data_identity",
        "physical_encoder",
        "correspondence",
        "n3_audit",
    ):
        if saved_summary.get(name) != current_summary.get(name):
            raise ValueError(f"resume {name} binding differs from current runtime")


@torch.no_grad()
def stock_equivalence_audit(
    *,
    model: DirectPhysicalFlowModel,
    sampler: Any,
    sample: dict[str, Any],
    correspondence_head: nn.Module,
    correspondence_runtime: dict[str, Any],
    device: torch.device,
    expect_zero_init: bool,
) -> dict[str, float | bool]:
    evidence = make_direct_evidence_bundle(
        sample,
        modes=(),
        device=device,
        correspondence_head=correspondence_head,
        correspondence_runtime=correspondence_runtime,
    )["correct"]
    target = sample["target"].unsqueeze(0).to(device=device, dtype=torch.float32)
    generator = torch.Generator(device=device).manual_seed(42020)
    noise = torch.randn(target.shape, generator=generator, device=device)
    x_t, _ = sampler._get_model_gt(target, 0.75, noise)
    t = torch.full((1,), 750.0, device=device, dtype=torch.float32)
    condition = sample["stock_condition"].to(device=device)
    stock = model.stock_prediction(x_t, t, condition)
    with lora_disabled(model.flow):
        native = model.flow(x_t, t, condition)
    disabled, _ = model.conditioned_prediction(
        x_t,
        t,
        condition,
        *evidence[:4],
        stock_velocity=stock,
        physical_present=False,
    )
    null, _ = model.conditioned_prediction(
        x_t,
        t,
        condition,
        *null_evidence_like(evidence),
        stock_velocity=stock,
    )
    enabled, _ = model.conditioned_prediction(
        x_t, t, condition, *evidence[:4], stock_velocity=stock
    )
    report = {
        "manual_stock_vs_native_max_abs": float((stock - native).abs().max().item()),
        "physical_off_max_abs": float((disabled - stock).abs().max().item()),
        "null_evidence_max_abs": float((null - stock).abs().max().item()),
        "zero_init_enabled_max_abs": float((enabled - stock).abs().max().item()),
        "expect_zero_init": bool(expect_zero_init),
    }
    required = (
        report["manual_stock_vs_native_max_abs"] == 0.0
        and report["physical_off_max_abs"] == 0.0
        and report["null_evidence_max_abs"] == 0.0
        and (
            not expect_zero_init or report["zero_init_enabled_max_abs"] == 0.0
        )
    )
    report["passed"] = required
    if not required:
        raise RuntimeError(f"stock equivalence failed: {report}")
    return report


def main() -> None:
    args = parse_args()
    corruption_modes = validate_args(args)
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        dist.init_process_group(
            backend="nccl", timeout=datetime.timedelta(hours=6)
        )
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    process_seed = int(args.seed) + rank * 100003
    random.seed(process_seed)
    np.random.seed(process_seed % (2**32 - 1))
    torch.manual_seed(process_seed)
    torch.cuda.manual_seed_all(process_seed)
    if rank:
        time.sleep(min(1.5 * rank, 12.0))

    output_dir = Path(args.output_dir)
    if rank == 0:
        if args.resume:
            output_dir.mkdir(parents=True, exist_ok=True)
        else:
            output_dir.mkdir(parents=True, exist_ok=False)
    if world_size > 1:
        dist.barrier()

    dataset = PoseLiftingCacheDataset(args.cache_manifest, indices=args.indices)
    if args.sampling_mode == "object_balanced":
        distributed_sampler: Sampler[int] = ObjectBalancedDistributedSampler(
            dataset.rows,
            num_replicas=world_size,
            rank=rank,
            seed=int(args.seed),
        )
    else:
        distributed_sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=int(args.seed),
            drop_last=False,
        )
    loader = DataLoader(
        dataset,
        batch_size=1,
        sampler=distributed_sampler,
        num_workers=int(args.num_workers),
        collate_fn=collate_one,
        pin_memory=True,
    )
    n3_audit = validate_n3_checkpoint(
        args.n3_report, args.correspondence_checkpoint
    )
    correspondence_head, correspondence_checkpoint, correspondence_runtime = (
        load_frozen_correspondence_head(
            args.correspondence_checkpoint,
            device=device,
            visual_channels=dataset.visual_feature_dim,
        )
    )
    if correspondence_runtime["checkpoint_sha256"] != n3_audit["checkpoint_sha256"]:
        raise RuntimeError("runtime correspondence checkpoint differs from N3")
    data_identity = lifting_cache_identity(
        args.cache_manifest,
        rows=dataset.rows,
    )
    admission_report = None
    if args.admission_report:
        admission_path = Path(args.admission_report).resolve()
        if not admission_path.is_file():
            raise RuntimeError(f"admission report does not exist: {admission_path}")
        admission_report = json.loads(admission_path.read_text(encoding="utf-8"))
        if admission_report.get("passed") is not True:
            raise RuntimeError(f"admission report did not pass: {admission_path}")
        admitted_identity = admission_report.get("train_identity", {})
        for key in (
            "manifest_sha256",
            "cache_schema_hash",
            "cache_config_hash",
            "uid_hash",
            "object_uid_hash",
        ):
            if admitted_identity.get(key) != data_identity.get(key):
                raise RuntimeError(
                    f"admission report identity mismatch for {key}: "
                    f"{admitted_identity.get(key)!r} != {data_identity.get(key)!r}"
                )
        admitted_n3 = admission_report.get("n3_audit", {})
        if admitted_n3.get("checkpoint_sha256") != correspondence_runtime[
            "checkpoint_sha256"
        ]:
            raise RuntimeError(
                "admission report correspondence checkpoint differs from runtime"
            )

    sampler, model, decoder, model_summary, sampler_params = build_direct_components(
        pretrained=args.pretrained,
        visual_channels=dataset.visual_feature_dim,
        physical_hidden_dim=int(args.physical_hidden_dim),
        lora_rank=int(args.lora_rank),
        lora_alpha=int(args.lora_alpha),
        gradient_checkpointing=bool(args.gradient_checkpointing),
        need_decoder=float(args.occupancy_weight) > 0.0,
        device=device,
    )
    model_summary.update(
        {
            "cache_manifest": str(Path(args.cache_manifest).resolve()),
            "cache_config_hash": dataset.config_hash,
            "data_identity": data_identity,
            "dataset_size": len(dataset),
            "unique_object_count": len(
                {
                    str(row.get("object_uid", row["uid"]))
                    for row in dataset.rows
                }
            ),
            "correspondence": correspondence_runtime,
            "n3_audit": n3_audit,
            "admission_report": (
                str(Path(args.admission_report).resolve())
                if args.admission_report
                else ""
            ),
            "sampler_params": sampler_params,
            "losses": {
                "flow": float(args.flow_weight),
                "frozen_decoder_occupancy": float(args.occupancy_weight),
                "wrong_to_stock": float(args.wrong_stock_weight),
                "correct_gain": float(args.correct_gain_weight),
                "correct_vs_wrong_rank": float(args.rank_weight),
                "delta_norm": float(args.delta_norm_weight),
            },
            "corruption_modes": list(corruption_modes),
            "training_profile": str(args.training_profile),
            "sampling": {
                "mode": str(args.sampling_mode),
                "object_balanced_per_epoch": args.sampling_mode
                == "object_balanced",
            },
        }
    )

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
    resumed_micro_step: int | None = None
    history: list[dict[str, Any]] = []
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        validate_resume(checkpoint, args, model_summary)
        load_partial_state(
            model,
            checkpoint["model_trainable_state"],
            require_all_trainable=True,
        )
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint.get("scaler", {}))
        start_step = int(checkpoint.get("step", 0))
        resumed_micro_step = int(
            checkpoint.get("micro_step", start_step * int(args.grad_accum))
        )
        history = list(checkpoint.get("history", []))
        if rank == 0:
            print(
                f"[direct_flow_train] resumed step={start_step} from={args.resume}",
                flush=True,
            )
    if start_step >= int(args.max_steps):
        raise ValueError(
            f"resume step={start_step} already reaches max_steps={args.max_steps}"
        )

    model_summary["stock_equivalence"] = stock_equivalence_audit(
        model=model,
        sampler=sampler,
        sample=dataset[0],
        correspondence_head=correspondence_head,
        correspondence_runtime=correspondence_runtime,
        device=device,
        expect_zero_init=not bool(args.resume),
    )
    if rank == 0:
        print(json.dumps(model_summary, indent=2), flush=True)

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
    global_step = start_step
    micro_step = (
        start_step * int(args.grad_accum)
        if resumed_micro_step is None
        else resumed_micro_step
    )
    applied_updates = 0
    nonfinite_attempts = 0
    epoch = 0
    wall_start = time.time()
    object_exposure: Counter[str] = Counter()
    occupancy_t_records: list[dict[str, float | str]] = []
    optimizer.zero_grad(set_to_none=True)
    model.train()

    while global_step < int(args.max_steps):
        distributed_sampler.set_epoch(epoch)
        for sample in loader:
            if global_step >= int(args.max_steps):
                break
            object_exposure[
                str(sample.get("object_uid", sample["uid"]))
            ] += 1
            wrong_mode = (
                corruption_modes[(micro_step + rank) % len(corruption_modes)]
                if corruption_modes
                else None
            )
            with torch.no_grad():
                evidence = make_direct_evidence_bundle(
                    sample,
                    modes=(() if wrong_mode is None else (wrong_mode,)),
                    device=device,
                    correspondence_head=correspondence_head,
                    correspondence_runtime=correspondence_runtime,
                )
                correct = evidence["correct"]
                wrong = evidence.get(wrong_mode) if wrong_mode is not None else None
                target = sample["target"].unsqueeze(0).to(
                    device=device, dtype=torch.float32, non_blocking=True
                )
                condition = sample["stock_condition"].to(
                    device=device, non_blocking=True
                )
                noise = torch.randn_like(target)
                t_value = sample_t(str(args.t_schedule), device)
                x_t, gt_velocity = sampler._get_model_gt(target, t_value, noise)
                t_tensor = torch.full(
                    (1,), 1000.0 * t_value, device=device, dtype=torch.float32
                )
                stock = model.stock_prediction(x_t, t_tensor, condition)

            sync_step = ((micro_step + 1) % int(args.grad_accum)) == 0
            sync_context = (
                wrapped.no_sync()
                if world_size > 1 and not sync_step
                else nullcontext()
            )
            with sync_context:
                with torch.autocast(
                    device_type="cuda", dtype=amp_dtype, enabled=use_amp
                ):
                    wrong_kwargs = {}
                    if wrong is not None:
                        wrong_kwargs = {
                            "wrong_view_visual": wrong[0],
                            "wrong_view_geometry": wrong[1],
                            "wrong_view_weight": wrong[2],
                            "wrong_metadata": wrong[3],
                        }
                    prediction, wrong_prediction, stock_velocity, stats, wrong_stats = wrapped(
                        x_t,
                        t_tensor,
                        condition,
                        *correct[:4],
                        stock_velocity=stock,
                        physical_scale=float(args.physical_scale),
                        **wrong_kwargs,
                    )
                    correct_flow = F.mse_loss(
                        prediction.float(), gt_velocity.float()
                    )
                    stock_flow = F.mse_loss(
                        stock_velocity.float(), gt_velocity.float()
                    ).detach()
                    stock_energy = (
                        stock_velocity.float().square().mean().detach().clamp_min(1.0e-6)
                    )
                    relative_gain = (
                        stock_flow - correct_flow
                    ) / stock_flow.clamp_min(1.0e-6)
                    zero = prediction.new_zeros((), dtype=torch.float32)
                    wrong_flow = zero
                    wrong_stock = zero
                    gain_loss = zero
                    relative_rank = zero
                    rank_loss = zero
                    delta_norm = F.mse_loss(
                        prediction.float(), stock_velocity.float()
                    ) / stock_energy
                    if wrong_prediction is not None:
                        if wrong_stats is None:
                            raise RuntimeError("wrong prediction is missing diagnostics")
                        wrong_flow = F.mse_loss(
                            wrong_prediction.float(), gt_velocity.float()
                        )
                        wrong_stock = F.mse_loss(
                            wrong_prediction.float(), stock_velocity.float()
                        ) / stock_energy
                        gain_loss = F.relu(
                            prediction.new_tensor(float(args.correct_gain_margin))
                            - relative_gain
                        )
                        relative_rank = (
                            wrong_flow - correct_flow
                        ) / stock_flow.clamp_min(1.0e-6)
                        rank_loss = F.relu(
                            prediction.new_tensor(float(args.rank_margin))
                            - relative_rank
                        )
                        delta_norm = 0.5 * (
                            delta_norm
                            + F.mse_loss(
                                wrong_prediction.float(), stock_velocity.float()
                            )
                            / stock_energy
                        )
                    occupancy_loss = prediction.new_zeros((), dtype=torch.float32)
                    occupancy_applied = (
                        decoder is not None
                        and float(args.occupancy_weight) > 0.0
                        and (micro_step + 1) % int(args.occupancy_every) == 0
                    )
                    if occupancy_applied:
                        pred_x0 = sampler._pred_to_xstart(
                            x_t, t_value, prediction
                        )
                        decoder_dtype = next(decoder.parameters()).dtype
                        logits = decoder(pred_x0.to(dtype=decoder_dtype)).float()
                        occupancy_target = target_occupancy(
                            sample["target_coords"], device
                        )
                        positive = occupancy_target.sum().clamp_min(1.0)
                        negative = occupancy_target.numel() - positive
                        pos_weight = (negative / positive).clamp(1.0, 32.0)
                        occupancy_loss = F.binary_cross_entropy_with_logits(
                            logits, occupancy_target, pos_weight=pos_weight
                        )
                    weighted_flow_loss = float(args.flow_weight) * correct_flow
                    weighted_occupancy_loss = (
                        float(args.occupancy_weight) * occupancy_loss
                    )
                    if occupancy_applied:
                        occupancy_t_records.append(
                            {
                                "bucket": t_bucket(float(t_value)),
                                "raw_loss": float(
                                    occupancy_loss.detach().float().item()
                                ),
                                "weighted_loss": float(
                                    weighted_occupancy_loss.detach().float().item()
                                ),
                            }
                        )
                    loss = (
                        weighted_flow_loss
                        + weighted_occupancy_loss
                        + float(args.wrong_stock_weight) * wrong_stock
                        + float(args.correct_gain_weight) * gain_loss
                        + float(args.rank_weight) * rank_loss
                        + float(args.delta_norm_weight) * delta_norm
                    )
                    next_step = global_step + 1
                    gradient_audit = (
                        sync_step
                        and int(args.loss_gradient_audit_every) > 0
                        and (
                            next_step == start_step + 1
                            or next_step % int(args.loss_gradient_audit_every) == 0
                        )
                    )
                    isolated_output_gradients = None
                    if gradient_audit:
                        flow_output_norm = output_gradient_norm(
                            weighted_flow_loss, prediction
                        )
                        occupancy_output_norm = (
                            output_gradient_norm(weighted_occupancy_loss, prediction)
                            if occupancy_applied
                            else 0.0
                        )
                        isolated_output_gradients = {
                            "weighted_flow_to_prediction": flow_output_norm,
                            "weighted_occupancy_to_prediction": occupancy_output_norm,
                            "occupancy_to_flow_ratio": occupancy_output_norm
                            / max(flow_output_norm, 1.0e-12),
                        }
                    scaled_loss = loss / float(args.grad_accum)
                scaler.scale(scaled_loss).backward()

            if sync_step:
                scaler.unscale_(optimizer)
                diagnostics = [
                    loss,
                    correct_flow,
                    stock_flow,
                    occupancy_loss,
                    relative_gain,
                    delta_norm,
                    stats["physical_token_rms"],
                    stats["flow_delta_rms"],
                ]
                if wrong_prediction is not None and wrong_stats is not None:
                    diagnostics.extend(
                        (
                            wrong_flow,
                            wrong_stock,
                            relative_rank,
                            wrong_stats["flow_delta_rms"],
                        )
                    )
                forward_finite = distributed_all_true(
                    tensors_finite(diagnostics), device, world_size
                )
                gradient_finite = distributed_all_true(
                    gradients_finite(trainable), device, world_size
                )
                update_finite = forward_finite and gradient_finite
                scaler_before = (
                    float(scaler.get_scale()) if scaler.is_enabled() else None
                )
                scaler_after = scaler_before
                clip_total_norm = None
                optimizer_step_applied = False
                gradient_norms = gradient_group_norms(model)

                if update_finite:
                    clip_tensor = torch.nn.utils.clip_grad_norm_(
                        trainable,
                        float(args.grad_clip),
                        error_if_nonfinite=True,
                    )
                    clip_total_norm = float(clip_tensor.detach().float().cpu().item())
                    scaler.step(optimizer)
                    scaler.update()
                    scaler_after = (
                        float(scaler.get_scale()) if scaler.is_enabled() else None
                    )
                    optimizer_step_applied = (
                        not scaler.is_enabled() or scaler_after >= scaler_before
                    )
                    if optimizer_step_applied:
                        finite_after = distributed_all_true(
                            parameters_finite(trainable)
                            and optimizer_state_finite(optimizer),
                            device,
                            world_size,
                        )
                        if not finite_after:
                            raise RuntimeError(
                                "direct Flow optimizer produced non-finite state"
                            )
                else:
                    nonfinite_attempts += 1
                    if scaler.is_enabled():
                        scaler.update()
                        scaler_after = float(scaler.get_scale())

                optimizer.zero_grad(set_to_none=True)
                if optimizer_step_applied:
                    global_step += 1
                    applied_updates += 1

                should_log = (
                    not optimizer_step_applied
                    or global_step == start_step + 1
                    or (
                        optimizer_step_applied
                        and global_step % int(args.log_every) == 0
                    )
                    or global_step == int(args.max_steps)
                )
                if should_log:
                    row = {
                        "step": int(global_step),
                        "micro_step": int(micro_step + 1),
                        "uid": str(sample["uid"]),
                        "object_uid": str(
                            sample.get("object_uid", sample["uid"])
                        ),
                        "views": int(correct[4]["views"]),
                        "wrong_mode": wrong_mode,
                        "training_profile": str(args.training_profile),
                        "loss": distributed_mean(loss, world_size),
                        "weighted_flow_loss": distributed_mean(
                            weighted_flow_loss, world_size
                        ),
                        "weighted_occupancy_loss": distributed_mean(
                            weighted_occupancy_loss, world_size
                        ),
                        "correct_flow_loss": distributed_mean(
                            correct_flow, world_size
                        ),
                        "wrong_flow_loss": distributed_mean(wrong_flow, world_size),
                        "stock_flow_loss": distributed_mean(stock_flow, world_size),
                        "relative_correct_gain": distributed_mean(
                            relative_gain, world_size
                        ),
                        "relative_correct_vs_wrong": distributed_mean(
                            relative_rank, world_size
                        ),
                        "occupancy_loss": distributed_mean(
                            occupancy_loss, world_size
                        ),
                        "occupancy_applied": bool(occupancy_applied),
                        "occupancy_t_bucket": (
                            t_bucket(float(t_value)) if occupancy_applied else None
                        ),
                        "isolated_output_gradient_norms": (
                            {
                                name: distributed_mean(
                                    torch.tensor(
                                        value,
                                        device=device,
                                        dtype=torch.float32,
                                    ),
                                    world_size,
                                )
                                for name, value in isolated_output_gradients.items()
                            }
                            if isolated_output_gradients is not None
                            else None
                        ),
                        "wrong_stock_loss": distributed_mean(
                            wrong_stock, world_size
                        ),
                        "delta_norm": distributed_mean(delta_norm, world_size),
                        "correct_token_rms": distributed_mean(
                            stats["physical_token_rms"], world_size
                        ),
                        "correct_delta_rms": distributed_mean(
                            stats["flow_delta_rms"], world_size
                        ),
                        "wrong_delta_rms": distributed_mean(
                            wrong_stats["flow_delta_rms"], world_size
                        )
                        if wrong_stats is not None
                        else 0.0,
                        "correct_pair_valid_ratio": distributed_mean(
                            stats["pair_valid_ratio"], world_size
                        ),
                        "correct_attention_entropy": distributed_mean(
                            stats["pair_attention_entropy"], world_size
                        ),
                        "gradient_norms": gradient_norms,
                        "clip_total_norm": clip_total_norm,
                        "forward_finite": bool(forward_finite),
                        "gradient_finite": bool(gradient_finite),
                        "update_finite": bool(update_finite),
                        "optimizer_step_applied": bool(optimizer_step_applied),
                        "nonfinite_attempts": int(nonfinite_attempts),
                        "amp_dtype": str(args.amp_dtype),
                        "scaler_before": scaler_before,
                        "scaler_after": scaler_after,
                        "t": float(t_value),
                        "elapsed_seconds": float(time.time() - wall_start),
                    }
                    if rank == 0:
                        history.append(row)
                        print(
                            f"[direct_flow_train] {json.dumps(row, ensure_ascii=False)}",
                            flush=True,
                        )

                if not optimizer_step_applied:
                    message = (
                        f"non-finite/skipped direct Flow update attempt={nonfinite_attempts} "
                        f"micro_step={micro_step + 1} scaler={scaler_before}->{scaler_after}"
                    )
                    if args.nonfinite_policy == "error" or nonfinite_attempts > int(
                        args.max_nonfinite_attempts
                    ):
                        raise RuntimeError(message)
                    if rank == 0:
                        print(f"[direct_flow_train] WARNING {message}", flush=True)

                save_now = optimizer_step_applied and (
                    global_step % int(args.save_every) == 0
                    or global_step == int(args.max_steps)
                )
                if save_now and rank == 0:
                    save_checkpoint(
                        output_dir / "checkpoints" / f"step_{global_step:06d}.pt",
                        model=model,
                        optimizer=optimizer,
                        scaler=scaler,
                        step=global_step,
                        micro_step=micro_step + 1,
                        args=args,
                        summary=model_summary,
                        history=history,
                    )
                    save_checkpoint(
                        output_dir / "checkpoints" / "last.pt",
                        model=model,
                        optimizer=optimizer,
                        scaler=scaler,
                        step=global_step,
                        micro_step=micro_step + 1,
                        args=args,
                        summary=model_summary,
                        history=history,
                    )
                if save_now and world_size > 1:
                    dist.barrier()
            micro_step += 1
        epoch += 1

    if world_size > 1:
        gathered_exposure: list[dict[str, int] | None] = [None] * world_size
        dist.all_gather_object(gathered_exposure, dict(object_exposure))
        combined_exposure: Counter[str] = Counter()
        for item in gathered_exposure:
            if item is not None:
                combined_exposure.update(item)
        gathered_occupancy: list[list[dict[str, float | str]] | None] = [
            None
        ] * world_size
        dist.all_gather_object(gathered_occupancy, occupancy_t_records)
        combined_occupancy = [
            row
            for rank_rows in gathered_occupancy
            if rank_rows is not None
            for row in rank_rows
        ]
    else:
        combined_exposure = object_exposure
        combined_occupancy = occupancy_t_records

    if rank == 0:
        report = {
            "stage": model_summary["stage"],
            "format": DIRECT_FLOW_VERSION,
            "args": vars(args),
            "model_summary": model_summary,
            "dataset_size": len(dataset),
            "start_global_step": int(start_step),
            "completed_global_step": int(global_step),
            "applied_optimizer_updates": int(applied_updates),
            "nonfinite_attempts": int(nonfinite_attempts),
            "micro_step": int(micro_step),
            "finite": parameters_finite(trainable)
            and optimizer_state_finite(optimizer),
            "object_exposure": exposure_summary(
                dataset.rows, combined_exposure
            ),
            "occupancy_by_t_bucket": occupancy_bucket_summary(
                combined_occupancy
            ),
            "history": history,
            "checkpoint": str(output_dir / "checkpoints" / "last.pt"),
            "elapsed_seconds": float(time.time() - wall_start),
        }
        (output_dir / "train_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
