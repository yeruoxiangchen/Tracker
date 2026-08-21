#!/usr/bin/env python3
"""Train adapter-only native active-32^3 every-block SLAT conditioning."""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import nullcontext
import datetime
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, Sampler

from pose_point_depth_mv.direct_slat_flow import canonical_json_sha256
from pose_point_depth_mv.native_3d_condition import (
    NATIVE_CONTROL_MODES,
    NATIVE_SLAT_FLOW_VERSION,
    NativeConditionSLatDataset,
    block_projection_gradient_norms,
    build_native_slat_components,
    collate_native_one,
    ensure_finite_trainable,
    drop_cached_native_projection,
    load_trainable_state_dict,
    sha256_file,
    trainable_state_dict,
    validate_lifting_feature_metadata,
    validate_native_checkpoint,
)
from pose_point_depth_mv.train_direct_flow import ObjectBalancedDistributedSampler
from pose_point_depth_mv.train_direct_slat_flow import normalized_target, to_device_tree
from reconvggt_ar_adapter_a.train_pointpose_ss_lora import sample_t
from trellis.modules import sparse as sp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache_manifest", required=True)
    parser.add_argument("--lifting_cache_manifest", required=True)
    parser.add_argument("--target_decoder_audit", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--pretrained", default="Stable-X/trellis-vggt-v0-2")
    parser.add_argument("--indices", default="all")
    parser.add_argument("--max_objects", type=int, default=0)
    parser.add_argument("--resume", default="")
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--save_every", type=int, default=100)
    parser.add_argument("--log_every", type=int, default=5)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--amp_dtype", choices=("bf16", "fp16", "none"), default="bf16")
    parser.add_argument("--amp_init_scale", type=float, default=8192.0)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--feature_source", choices=("dino", "vggt", "all"), default="dino")
    parser.add_argument("--condition_scale", type=float, default=1.0)
    parser.add_argument("--delta_norm_weight", type=float, default=1.0e-4)
    parser.add_argument("--max_slat_points", type=int, default=40960)
    parser.add_argument("--control_mode", choices=NATIVE_CONTROL_MODES, default="pose_cyclic1")
    parser.add_argument("--control_probability", type=float, default=0.0)
    parser.add_argument("--control_rank_weight", type=float, default=0.0)
    parser.add_argument("--control_rank_margin", type=float, default=0.0)
    parser.add_argument("--sampling_mode", choices=("object_balanced", "sequence"), default="object_balanced")
    parser.add_argument("--t_schedule", choices=("uniform", "logit_normal", "high_t_mix"), default="uniform")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--verify_cache_hashes", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if int(args.max_objects) < 0:
        raise ValueError("max_objects must be non-negative")
    positive = (
        "max_steps",
        "save_every",
        "log_every",
        "grad_accum",
        "lr",
        "grad_clip",
        "amp_init_scale",
        "hidden_dim",
        "condition_scale",
        "max_slat_points",
    )
    bad = [name for name in positive if float(getattr(args, name)) <= 0]
    if bad:
        raise ValueError(f"arguments must be positive: {bad}")
    for name in (
        "weight_decay",
        "delta_norm_weight",
        "control_probability",
        "control_rank_weight",
        "control_rank_margin",
    ):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and non-negative")
    if float(args.control_probability) > 1:
        raise ValueError("control_probability must be <= 1")
    if float(args.control_rank_weight) > 0 and float(args.control_probability) == 0:
        raise ValueError("control rank loss requires non-zero control_probability")


def checkpoint_fields() -> tuple[str, ...]:
    return (
        "cache_manifest",
        "lifting_cache_manifest",
        "target_decoder_audit",
        "pretrained",
        "indices",
        "max_objects",
        "hidden_dim",
        "feature_source",
        "condition_scale",
        "max_slat_points",
        "sampling_mode",
        "t_schedule",
        "delta_norm_weight",
        "control_mode",
        "control_probability",
        "control_rank_weight",
        "control_rank_margin",
        "grad_accum",
    )


def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    step: int,
    micro_step: int,
    epoch: int,
    samples_into_epoch: int,
    args: argparse.Namespace,
    summary: dict[str, Any],
    history: list[dict[str, Any]],
) -> None:
    ensure_finite_trainable(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": NATIVE_SLAT_FLOW_VERSION,
            "step": int(step),
            "micro_step": int(micro_step),
            "epoch": int(epoch),
            "samples_into_epoch": int(samples_into_epoch),
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
    checkpoint: dict[str, Any], args: argparse.Namespace, summary: dict[str, Any]
) -> None:
    validate_native_checkpoint(
        checkpoint, expected_format=NATIVE_SLAT_FLOW_VERSION, pretrained=args.pretrained
    )
    saved = checkpoint.get("args", {})
    mismatch = {
        name: (saved.get(name), getattr(args, name))
        for name in checkpoint_fields()
        if str(saved.get(name)) != str(getattr(args, name))
    }
    if mismatch:
        raise ValueError(f"native SLAT resume protocol mismatch={mismatch}")
    for key in ("data_identity", "feature_identity", "block_count", "target_decoder_audit"):
        if checkpoint["model_summary"].get(key) != summary.get(key):
            raise ValueError(f"native SLAT resume {key} binding differs")


def validate_decoder_audit(
    path: str | Path, *, pretrained: str
) -> dict[str, Any]:
    audit_path = Path(path).resolve()
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("format") != "pose_point_depth_mv.direct_slat_target_decoder_audit.v1":
        raise ValueError("unsupported target-decoder audit format")
    if audit.get("passed") is not True:
        raise RuntimeError("target-decoder audit did not pass")
    if str(audit.get("pretrained")) != str(pretrained):
        raise RuntimeError("target-decoder audit pretrained binding differs")
    if int(audit.get("summary", {}).get("object_count", 0)) < 32:
        raise RuntimeError("target-decoder audit must cover at least 32 objects")
    return {
        "path": str(audit_path),
        "sha256": sha256_file(audit_path),
        "format": audit["format"],
        "summary": audit.get("summary", {}),
    }


@torch.no_grad()
def stock_equivalence_audit(
    *,
    model: nn.Module,
    sampler: Any,
    sample: dict[str, Any],
    mean: torch.Tensor,
    std: torch.Tensor,
    device: torch.device,
    max_points: int,
    fresh: bool,
) -> dict[str, Any]:
    target = normalized_target(
        sample,
        mean=mean,
        std=std,
        device=device,
        max_points=max_points,
        selection_seed=17,
    )
    noise = sp.SparseTensor(feats=torch.zeros_like(target.feats), coords=target.coords)
    t_value = torch.tensor(0.5, device=device)
    x_t, _ = sampler._get_model_gt(target, t_value, noise)
    t = torch.full((1,), 500.0, device=device)
    condition = to_device_tree(sample["condition"]["cond"], device)
    stock = model.stock_prediction(x_t, t, condition)
    disabled, _ = model.conditioned_prediction(
        x_t, t, condition, None, stock_velocity=stock, condition_scale=0.0
    )
    enabled, stats = model.conditioned_prediction(
        x_t,
        t,
        condition,
        sample["lifting_sample"],
        stock_velocity=stock,
        condition_scale=1.0,
    )
    report = {
        "disabled_max_abs": float((disabled.feats - stock.feats).abs().max().item()),
        "enabled_zero_init_max_abs": float((enabled.feats - stock.feats).abs().max().item()),
        "coords_equal": bool(torch.equal(enabled.coords, stock.coords)),
        "expect_zero_init": bool(fresh),
        "conditioned_block_count": int(stats["conditioned_block_count"].item()),
        "active_point_count": int(stats["native_active_point_count"].item()),
    }
    report["passed"] = bool(
        report["disabled_max_abs"] == 0.0
        and (not fresh or report["enabled_zero_init_max_abs"] == 0.0)
        and report["coords_equal"]
        and report["conditioned_block_count"] == len(model.flow_core.blocks)
        and report["active_point_count"] > 0
    )
    if not report["passed"]:
        raise RuntimeError(f"native SLAT stock equivalence failed: {report}")
    return report


def distributed_true(value: bool, device: torch.device, world_size: int) -> bool:
    tensor = torch.tensor(int(value), device=device)
    if world_size > 1:
        dist.all_reduce(tensor, op=dist.ReduceOp.MIN)
    return bool(tensor.item())


def main() -> None:
    args = parse_args()
    validate_args(args)
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        dist.init_process_group(backend="nccl", timeout=datetime.timedelta(hours=12))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    process_seed = int(args.seed) + rank * 100003
    random.seed(process_seed)
    np.random.seed(process_seed % (2**32 - 1))
    torch.manual_seed(process_seed)
    torch.cuda.manual_seed_all(process_seed)
    output_dir = Path(args.output_dir)
    if rank == 0:
        if args.resume:
            output_dir.mkdir(parents=True, exist_ok=True)
        else:
            output_dir.mkdir(parents=True, exist_ok=False)
    if world_size > 1:
        dist.barrier()

    dataset = NativeConditionSLatDataset(
        args.cache_manifest,
        args.lifting_cache_manifest,
        indices=args.indices,
        verify_hashes=bool(args.verify_cache_hashes),
    )
    dataset.limit_objects(int(args.max_objects))
    if dataset.config.get("pretrained") != args.pretrained:
        raise RuntimeError("SLAT cache pretrained binding differs")
    feature_identity = validate_lifting_feature_metadata(
        visual_feature_dim=dataset.lifting.visual_feature_dim,
        feature_metadata=dataset.lifting.feature_metadata,
        feature_source=str(args.feature_source),
    )
    decoder_audit = validate_decoder_audit(
        args.target_decoder_audit, pretrained=args.pretrained
    )
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
        collate_fn=collate_native_one,
        pin_memory=True,
    )
    sampler, model, sampler_params, normalization, model_summary = build_native_slat_components(
        pretrained=args.pretrained,
        hidden_dim=int(args.hidden_dim),
        feature_source=str(args.feature_source),
        gradient_checkpointing=bool(args.gradient_checkpointing),
        device=device,
    )
    runtime_normalization = {
        key: [float(item) for item in value] for key, value in normalization.items()
    }
    if canonical_json_sha256(runtime_normalization) != dataset.slat_normalization_hash:
        raise RuntimeError("runtime SLAT normalization differs from cache")
    model_summary.update(
        {
            "feature_identity": feature_identity,
            "data_identity": dataset.identity,
            "target_decoder_audit": decoder_audit,
            "dataset_size": len(dataset),
            "unique_object_count": len({str(row["object_uid"]) for row in dataset.rows}),
            "slat_normalization": runtime_normalization,
            "slat_normalization_hash": dataset.slat_normalization_hash,
            "sampler_params": sampler_params,
            "training": {
                "adapter_only": True,
                "flow_matching_mse": 1.0,
                "delta_norm_weight": float(args.delta_norm_weight),
                "control_mode": str(args.control_mode),
                "control_probability": float(args.control_probability),
                "control_rank_weight": float(args.control_rank_weight),
                "control_rank_margin": float(args.control_rank_margin),
                "wrong_condition_role": "optional diagnostic/weak regularizer",
            },
        }
    )
    mean = torch.tensor(runtime_normalization["mean"], device=device)[None]
    std = torch.tensor(runtime_normalization["std"], device=device)[None]
    if bool((std <= 0).any().item()):
        raise RuntimeError("SLAT normalization std must be positive")
    trainable = [value for value in model.parameters() if value.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        betas=(0.9, 0.95),
        eps=1.0e-8,
    )
    scaler = torch.cuda.amp.GradScaler(
        enabled=args.amp_dtype == "fp16", init_scale=float(args.amp_init_scale)
    )
    start_step = 0
    micro_step = 0
    start_epoch = 0
    resume_samples = 0
    history: list[dict[str, Any]] = []
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        validate_resume(checkpoint, args, model_summary)
        load_trainable_state_dict(model, checkpoint["model_trainable_state"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint.get("scaler", {}))
        start_step = int(checkpoint["step"])
        micro_step = int(checkpoint.get("micro_step", start_step * int(args.grad_accum)))
        start_epoch = int(checkpoint.get("epoch", 0))
        resume_samples = int(checkpoint.get("samples_into_epoch", 0))
        history = list(checkpoint.get("history", []))
    if start_step >= int(args.max_steps):
        raise ValueError("resume already reached max_steps")
    model_summary["stock_equivalence"] = stock_equivalence_audit(
        model=model,
        sampler=sampler,
        sample=dataset[0],
        mean=mean,
        std=std,
        device=device,
        max_points=int(args.max_slat_points),
        fresh=not bool(args.resume),
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
    model.train()
    use_amp = args.amp_dtype != "none"
    amp_dtype = torch.float16 if args.amp_dtype == "fp16" else torch.bfloat16
    global_step = start_step
    epoch = start_epoch
    exposure: Counter[str] = Counter()
    wall_start = time.time()
    optimizer.zero_grad(set_to_none=True)
    while global_step < int(args.max_steps):
        distributed_sampler.set_epoch(epoch)
        samples_into_epoch = 0
        for loader_position, sample in enumerate(loader):
            if epoch == start_epoch and loader_position < resume_samples:
                continue
            samples_into_epoch = loader_position + 1
            if global_step >= int(args.max_steps):
                break
            exposure[str(sample["object_uid"])] += 1
            event_seed = process_seed * 1000003 + micro_step * 1013
            control_event = (
                float(args.control_probability) > 0
                and random.Random(event_seed + 41).random() < float(args.control_probability)
            )
            with torch.no_grad():
                target = normalized_target(
                    sample,
                    mean=mean,
                    std=std,
                    device=device,
                    max_points=int(args.max_slat_points),
                    selection_seed=event_seed + 11,
                )
                generator = torch.Generator(device=device).manual_seed(event_seed + 17)
                noise = sp.SparseTensor(
                    feats=torch.randn(
                        target.feats.shape,
                        generator=generator,
                        device=device,
                        dtype=target.feats.dtype,
                    ),
                    coords=target.coords,
                )
                with torch.random.fork_rng(devices=[local_rank]):
                    torch.cuda.manual_seed(event_seed + 31)
                    t_value = sample_t(str(args.t_schedule), device)
                x_t, gt_velocity = sampler._get_model_gt(target, t_value, noise)
                t = torch.full((1,), 1000.0 * t_value, device=device)
                condition = to_device_tree(sample["condition"]["cond"], device)
                stock = model.stock_prediction(x_t, t, condition)
            sync_step = (micro_step + 1) % int(args.grad_accum) == 0
            sync_context = wrapped.no_sync() if world_size > 1 and not sync_step else nullcontext()
            with sync_context:
                with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                    prediction, _, stats = wrapped(
                        x_t,
                        t,
                        condition,
                        sample["lifting_sample"],
                        stock_velocity=stock,
                        condition_scale=float(args.condition_scale),
                    )
                    flow_loss = F.mse_loss(prediction.feats.float(), gt_velocity.feats.float())
                    stock_loss = F.mse_loss(stock.feats.float(), gt_velocity.feats.float()).detach()
                    stock_energy = stock.feats.float().square().mean().detach().clamp_min(1.0e-6)
                    delta_norm = F.mse_loss(prediction.feats.float(), stock.feats.float()) / stock_energy
                    control_loss = flow_loss.new_zeros(())
                    control_gain = flow_loss.new_zeros(())
                    if control_event:
                        control_context = (
                            nullcontext()
                            if float(args.control_rank_weight) > 0
                            else torch.no_grad()
                        )
                        with control_context:
                            control, _ = model.conditioned_prediction(
                                x_t,
                                t,
                                condition,
                                sample["lifting_sample"],
                                stock_velocity=stock,
                                condition_scale=float(args.condition_scale),
                                projection_mode=str(args.control_mode),
                            )
                        control_loss = F.mse_loss(
                            control.feats.float(), gt_velocity.feats.float()
                        )
                        control_gain = control_loss - flow_loss
                        drop_cached_native_projection(
                            sample["lifting_sample"], mode=str(args.control_mode)
                        )
                    rank_loss = (
                        F.relu(flow_loss.new_tensor(float(args.control_rank_margin)) - control_gain)
                        if control_event
                        else flow_loss.new_zeros(())
                    )
                    loss = (
                        flow_loss
                        + float(args.delta_norm_weight) * delta_norm
                        + float(args.control_rank_weight) * rank_loss
                    )
                    scaled_loss = loss / float(args.grad_accum)
                scaler.scale(scaled_loss).backward()
            micro_step += 1
            if not sync_step:
                continue
            scaler.unscale_(optimizer)
            grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable, float(args.grad_clip)).item())
            block_grad_norms = block_projection_gradient_norms(model)
            finite = math.isfinite(grad_norm) and all(math.isfinite(value) for value in block_grad_norms)
            finite = finite and all(
                parameter.grad is None or bool(torch.isfinite(parameter.grad).all().item())
                for parameter in trainable
            )
            if not distributed_true(finite, device, world_size):
                raise FloatingPointError("native SLAT gradient became non-finite")
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            row = {
                "step": global_step,
                "micro_step": micro_step,
                "uid": str(sample["uid"]),
                "object_uid": str(sample["object_uid"]),
                "support_seed": int(sample["support_seed"]),
                "t": float(t_value),
                "loss": float(loss.detach().item()),
                "flow_loss": float(flow_loss.detach().item()),
                "stock_loss": float(stock_loss.item()),
                "gain_vs_stock": float((stock_loss - flow_loss.detach()).item()),
                "delta_norm": float(delta_norm.detach().item()),
                "control_evaluated": bool(control_event),
                "control_loss": float(control_loss.detach().item()),
                "correct_over_control_advantage": float(control_gain.detach().item()),
                "grad_norm": grad_norm,
                "block_projection_grad_norm_min": min(block_grad_norms),
                "block_projection_grad_norm_max": max(block_grad_norms),
                "condition_token_rms": float(stats["condition_token_rms"].detach().item()),
                "flow_delta_rms": float(stats["flow_delta_rms"].detach().item()),
                "native_active_point_count": int(stats["native_active_point_count"].detach().item()),
                "elapsed_seconds": time.time() - wall_start,
            }
            history.append(row)
            if rank == 0 and (global_step == 1 or global_step % int(args.log_every) == 0):
                print(f"[native_slat_train] {json.dumps(row)}", flush=True)
            if rank == 0 and global_step % int(args.save_every) == 0:
                save_checkpoint(
                    output_dir / "checkpoints" / f"step_{global_step:06d}.pt",
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    step=global_step,
                    micro_step=micro_step,
                    epoch=epoch,
                    samples_into_epoch=samples_into_epoch,
                    args=args,
                    summary=model_summary,
                    history=history,
                )
        epoch += 1
        resume_samples = 0
    if rank == 0:
        checkpoint_path = output_dir / "checkpoints" / f"step_{global_step:06d}.pt"
        if not checkpoint_path.is_file():
            save_checkpoint(
                checkpoint_path,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                step=global_step,
                micro_step=micro_step,
                epoch=epoch,
                samples_into_epoch=0,
                args=args,
                summary=model_summary,
                history=history,
            )
        report = {
            "format": NATIVE_SLAT_FLOW_VERSION,
            "completed": True,
            "step": global_step,
            "micro_step": micro_step,
            "model_summary": model_summary,
            "object_exposure": dict(sorted(exposure.items())),
            "history": history,
            "final_checkpoint": str(checkpoint_path),
            "finite": all(math.isfinite(float(row["loss"])) for row in history),
        }
        (output_dir / "report.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
