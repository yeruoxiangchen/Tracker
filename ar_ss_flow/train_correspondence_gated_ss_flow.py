#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
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
import torch.nn.functional as F

TRACKER_ROOT = Path(__file__).resolve().parents[1]
RECONVIAGEN_ROOT = TRACKER_ROOT / "ReconViaGen"
VGGT_ROOT = RECONVIAGEN_ROOT / "wheels" / "vggt"
for path in (TRACKER_ROOT, RECONVIAGEN_ROOT, VGGT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from trellis.pipelines import TrellisVGGTTo3DPipeline  # noqa: E402

from ar_ss_flow.correspondence_gated_flow import (  # noqa: E402
    CORRESPONDENCE_GATED_FLOW_VERSION,
    CorrespondenceGatedSSFlowModel,
    CorrespondenceGatedVelocityAdapter,
)
from ar_ss_flow.correspondence_lifting import (  # noqa: E402
    CORRESPONDENCE_NEGATIVE_MODES,
    correspondence_volume_from_sample,
    load_correspondence_checkpoint,
    parse_csv,
)
from ar_ss_flow.local_pose_lifting_flow import PoseLiftingCacheDataset  # noqa: E402
from reconvggt_ar_adapter_a.pointpose_ss_condition import trainable_state_dict  # noqa: E402
from reconvggt_ar_adapter_a.train_pointpose_ss_lora import (  # noqa: E402
    install_unused_model_stubs,
    sample_t,
)


FLOW_NEGATIVE_MODES = (*CORRESPONDENCE_NEGATIVE_MODES, "depth_corrupt")


def find_cross_sample(
    dataset: PoseLiftingCacheDataset,
    source_index: int,
    source: dict[str, Any],
) -> dict[str, Any]:
    views = int(source["visual_patch_features"].shape[0])
    shape = tuple(source["visual_patch_features"].shape[1:])
    for offset in range(1, len(dataset)):
        candidate = dataset[(source_index + offset) % len(dataset)]
        if str(candidate.get("object_uid")) == str(source.get("object_uid")):
            continue
        if int(candidate["visual_patch_features"].shape[0]) != views:
            continue
        if tuple(candidate["visual_patch_features"].shape[1:]) != shape:
            continue
        return candidate
    raise RuntimeError("no matching cross-sample negative")


def build_model(
    args: argparse.Namespace,
    device: torch.device,
    visual_channels: int,
) -> tuple[Any, CorrespondenceGatedSSFlowModel, dict[str, Any]]:
    install_unused_model_stubs()
    pipeline = TrellisVGGTTo3DPipeline.from_pretrained(args.pretrained)
    sampler = pipeline.sparse_structure_sampler
    flow = pipeline.models["sparse_structure_flow_model"].to(device).eval()
    for parameter in flow.parameters():
        parameter.requires_grad = False
    if int(flow.resolution) != 16 or int(flow.in_channels) != 8 or int(flow.out_channels) != 8:
        raise RuntimeError("unexpected sparse structure flow schema")
    adapter = CorrespondenceGatedVelocityAdapter(
        visual_channels=int(visual_channels),
        latent_channels=8,
        hidden_dim=int(args.adapter_hidden_dim),
        confidence_threshold=float(args.confidence_threshold),
    ).to(device)
    model = CorrespondenceGatedSSFlowModel(flow, adapter).to(device)
    trainable_names = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    if not trainable_names or any(not name.startswith("adapter.") for name in trainable_names):
        raise RuntimeError(f"trainable whitelist failed: {trainable_names}")
    summary = {
        "stage": "C2 correspondence-gated same-voxel SS residual",
        "adapter": adapter.metadata(),
        "trainable_parameter_names": trainable_names,
        "trainable_parameter_count": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "stock_flow_trainable_parameters": sum(
            parameter.numel() for parameter in flow.parameters() if parameter.requires_grad
        ),
        "flow_lora_enabled": False,
        "correspondence_model_frozen": True,
    }
    del pipeline
    gc.collect()
    torch.cuda.empty_cache()
    return sampler, model, summary


def save_checkpoint(
    path: Path,
    *,
    model: CorrespondenceGatedSSFlowModel,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    step: int,
    args: argparse.Namespace,
    summary: dict[str, Any],
    history: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": CORRESPONDENCE_GATED_FLOW_VERSION,
            "step": int(step),
            "model_trainable_state": trainable_state_dict(model),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "args": vars(args),
            "model_summary": summary,
            "history": history,
        },
        path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a frozen-stock same-voxel SS residual using a frozen, "
            "pre-audited correspondence checkpoint."
        )
    )
    parser.add_argument("--cache_manifest", required=True)
    parser.add_argument("--correspondence_checkpoint", required=True)
    parser.add_argument("--correspondence_audit_report", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--pretrained", default="Stable-X/trellis-vggt-v0-2")
    parser.add_argument("--indices", default="0-15")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_steps", type=int, default=50)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--log_every", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--amp_dtype", choices=("bf16", "fp16", "none"), default="bf16")
    parser.add_argument("--adapter_hidden_dim", type=int, default=96)
    parser.add_argument("--physical_scale", type=float, default=1.0)
    parser.add_argument("--confidence_threshold", type=float, default=0.55)
    parser.add_argument("--neighborhood_radius", type=int, default=1)
    parser.add_argument("--min_source_views", type=int, default=2)
    parser.add_argument("--negative_modes", default="pose_cyclic1,pose_cyclic2,pose_reverse,cross_sample")
    parser.add_argument("--t_schedule", choices=("uniform", "logit_normal", "high_t_mix"), default="uniform")
    parser.add_argument("--flow_weight", type=float, default=1.0)
    parser.add_argument("--wrong_stock_weight", type=float, default=0.10)
    parser.add_argument("--correct_gain_weight", type=float, default=0.10)
    parser.add_argument("--correct_gain_margin", type=float, default=0.005)
    parser.add_argument("--correct_wrong_rank_weight", type=float, default=0.05)
    parser.add_argument("--correct_wrong_margin", type=float, default=0.002)
    parser.add_argument("--delta_norm_weight", type=float, default=0.01)
    return parser.parse_args()


@torch.no_grad()
def stock_equivalence_audit(
    model: CorrespondenceGatedSSFlowModel,
    sample: dict[str, Any],
    correspondence_model,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, float]:
    volume, metadata, _ = correspondence_volume_from_sample(
        sample,
        device=device,
        model=correspondence_model,
        mode="correct",
        neighborhood_radius=int(args.neighborhood_radius),
        min_source_views=int(args.min_source_views),
        confidence_floor=float(args.confidence_threshold),
    )
    condition = sample["stock_condition"].to(device=device)
    generator = torch.Generator(device=device).manual_seed(42014)
    x_t = torch.randn((1, 8, 16, 16, 16), generator=generator, device=device)
    t = torch.tensor([500.0], device=device)
    stock = model.stock_prediction(x_t, t, condition)
    disabled, _ = model.adapt_from_stock(
        x_t, t, stock, volume, metadata, physical_present=False
    )
    null, _ = model.adapt_from_stock(
        x_t,
        t,
        stock,
        torch.zeros_like(volume),
        torch.zeros_like(metadata),
    )
    enabled, _ = model.adapt_from_stock(x_t, t, stock, volume, metadata)
    report = {
        "physical_off_max_abs_diff": float((disabled - stock).abs().max().item()),
        "null_max_abs_diff": float((null - stock).abs().max().item()),
        "zero_init_enabled_max_abs_diff": float((enabled - stock).abs().max().item()),
    }
    if any(value != 0.0 for value in report.values()):
        raise RuntimeError(f"stock/zero-init equivalence failed: {report}")
    return report


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    modes = parse_csv(args.negative_modes)
    invalid = [mode for mode in modes if mode not in FLOW_NEGATIVE_MODES]
    if invalid:
        raise ValueError(f"invalid flow negative modes={invalid}")
    random.seed(args.seed)
    np.random.seed(args.seed % (2**32 - 1))
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    dataset = PoseLiftingCacheDataset(args.cache_manifest, indices=args.indices)
    audit_report = json.loads(Path(args.correspondence_audit_report).read_text(encoding="utf-8"))
    if not bool(audit_report.get("passed", False)):
        raise RuntimeError("correspondence input audit did not pass; C2 gate remains closed")
    audited_checkpoint = Path(audit_report.get("protocol", {}).get("checkpoint", "")).resolve()
    requested_checkpoint = Path(args.correspondence_checkpoint).resolve()
    if audited_checkpoint != requested_checkpoint:
        raise RuntimeError(
            f"C1 audit/checkpoint mismatch: audit={audited_checkpoint}, requested={requested_checkpoint}"
        )
    correspondence_model, correspondence_checkpoint = load_correspondence_checkpoint(
        args.correspondence_checkpoint,
        device=device,
        visual_channels=dataset.visual_feature_dim,
    )
    correspondence_model.eval()
    for parameter in correspondence_model.parameters():
        parameter.requires_grad = False
    flow_sampler, model, summary = build_model(
        args, device, dataset.visual_feature_dim
    )
    summary["correspondence_checkpoint"] = str(
        Path(args.correspondence_checkpoint).resolve()
    )
    summary["correspondence_audit_report"] = str(
        Path(args.correspondence_audit_report).resolve()
    )
    summary["correspondence_checkpoint_step"] = int(
        correspondence_checkpoint.get("step", 0)
    )
    summary["stock_equivalence"] = stock_equivalence_audit(
        model, dataset[0], correspondence_model, device, args
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
        enabled=args.amp_dtype == "fp16", init_scale=8192.0
    )
    use_amp = args.amp_dtype != "none"
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    history: list[dict[str, Any]] = []
    start_time = time.time()
    model.train()

    for step in range(1, int(args.max_steps) + 1):
        sample_index = (step - 1) % len(dataset)
        sample = dataset[sample_index]
        mode = modes[(step - 1) % len(modes)]
        visual_override = None
        wrong_mode = mode
        if mode == "cross_sample":
            cross = find_cross_sample(dataset, sample_index, sample)
            visual_override = cross["visual_patch_features"]
            wrong_mode = "correct"
        with torch.no_grad():
            correct_volume, correct_metadata, correct_volume_stats = (
                correspondence_volume_from_sample(
                    sample,
                    device=device,
                    model=correspondence_model,
                    mode="correct",
                    neighborhood_radius=int(args.neighborhood_radius),
                    min_source_views=int(args.min_source_views),
                    confidence_floor=float(args.confidence_threshold),
                )
            )
            wrong_volume, wrong_metadata, wrong_volume_stats = (
                correspondence_volume_from_sample(
                    sample,
                    device=device,
                    model=correspondence_model,
                    mode=wrong_mode,
                    visual_patch_features_override=visual_override,
                    neighborhood_radius=int(args.neighborhood_radius),
                    min_source_views=int(args.min_source_views),
                    confidence_floor=float(args.confidence_threshold),
                )
            )
            target = sample["target"].unsqueeze(0).to(device=device, dtype=torch.float32)
            condition = sample["stock_condition"].to(device=device)
            noise = torch.randn_like(target)
            t_model = sample_t(str(args.t_schedule), device)
            x_t, gt_velocity = flow_sampler._get_model_gt(target, t_model, noise)
            t_tensor = torch.full(
                (1,), 1000.0 * t_model, device=device, dtype=torch.float32
            )
            stock = model.stock_prediction(x_t, t_tensor, condition)

        with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
            correct_prediction, correct_stats = model.adapt_from_stock(
                x_t,
                t_tensor,
                stock,
                correct_volume,
                correct_metadata,
                scale=float(args.physical_scale),
            )
            wrong_prediction, wrong_stats = model.adapt_from_stock(
                x_t,
                t_tensor,
                stock,
                wrong_volume,
                wrong_metadata,
                scale=float(args.physical_scale),
            )
            correct_flow = F.mse_loss(correct_prediction.float(), gt_velocity.float())
            stock_flow = F.mse_loss(stock.float(), gt_velocity.float()).detach()
            wrong_flow = F.mse_loss(wrong_prediction.float(), gt_velocity.float())
            stock_energy = stock.float().square().mean().detach().clamp_min(1.0e-6)
            wrong_stock = F.mse_loss(wrong_prediction.float(), stock.float()) / stock_energy
            relative_gain = (stock_flow - correct_flow) / stock_flow.clamp_min(1.0e-6)
            correct_gain_loss = F.relu(
                correct_prediction.new_tensor(float(args.correct_gain_margin))
                - relative_gain
            )
            relative_correct_wrong = (
                wrong_flow - correct_flow
            ) / stock_flow.clamp_min(1.0e-6)
            rank_loss = F.relu(
                correct_prediction.new_tensor(float(args.correct_wrong_margin))
                - relative_correct_wrong
            )
            delta_norm = 0.5 * (
                F.mse_loss(correct_prediction.float(), stock.float())
                + F.mse_loss(wrong_prediction.float(), stock.float())
            ) / stock_energy
            loss = (
                float(args.flow_weight) * correct_flow
                + float(args.wrong_stock_weight) * wrong_stock
                + float(args.correct_gain_weight) * correct_gain_loss
                + float(args.correct_wrong_rank_weight) * rank_loss
                + float(args.delta_norm_weight) * delta_norm
            )
        if not bool(torch.isfinite(loss).item()):
            raise RuntimeError(f"non-finite flow loss at step={step}")
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, float(args.grad_clip))
        if not bool(torch.isfinite(grad_norm).item()):
            raise RuntimeError(f"non-finite gradient at step={step}")
        scaler.step(optimizer)
        scaler.update()
        row = {
            "step": step,
            "uid": str(sample["uid"]),
            "object_uid": str(sample.get("object_uid", sample["uid"])),
            "negative_mode": mode,
            "loss": float(loss.detach().float().item()),
            "correct_flow": float(correct_flow.detach().float().item()),
            "stock_flow": float(stock_flow.detach().float().item()),
            "wrong_flow": float(wrong_flow.detach().float().item()),
            "relative_correct_gain": float(relative_gain.detach().float().item()),
            "relative_correct_vs_wrong": float(
                relative_correct_wrong.detach().float().item()
            ),
            "wrong_stock_loss": float(wrong_stock.detach().float().item()),
            "delta_norm": float(delta_norm.detach().float().item()),
            "correct_delta_rms": float(correct_stats["delta_rms"].detach().float().item()),
            "wrong_delta_rms": float(wrong_stats["delta_rms"].detach().float().item()),
            "correct_gate_mean": float(correct_stats["gate_mean"].detach().float().item()),
            "wrong_gate_mean": float(wrong_stats["gate_mean"].detach().float().item()),
            "correct_corr_confidence": float(
                correct_volume_stats["mean_correspondence_confidence"]
            ),
            "wrong_corr_confidence": float(
                wrong_volume_stats["mean_correspondence_confidence"]
            ),
            "grad_norm": float(grad_norm.detach().float().item()),
            "t": float(t_model),
            "elapsed_seconds": float(time.time() - start_time),
        }
        if step == 1 or step % int(args.log_every) == 0 or step == int(args.max_steps):
            history.append(row)
            print(f"[corr_gated_flow_train] {json.dumps(row)}", flush=True)
        if step % int(args.save_every) == 0 or step == int(args.max_steps):
            save_checkpoint(
                output_dir / "checkpoints" / f"step_{step:06d}.pt",
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                step=step,
                args=args,
                summary=summary,
                history=history,
            )
            save_checkpoint(
                output_dir / "checkpoints" / "last.pt",
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                step=step,
                args=args,
                summary=summary,
                history=history,
            )

    report = {
        "stage": summary["stage"],
        "args": vars(args),
        "model_summary": summary,
        "dataset_size": len(dataset),
        "unique_object_count": len(
            {str(row.get("object_uid", row["uid"])) for row in dataset.rows}
        ),
        "completed_steps": int(args.max_steps),
        "history": history,
        "finite": all(
            bool(torch.isfinite(parameter.detach().float()).all().item())
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "checkpoint": str(output_dir / "checkpoints" / "last.pt"),
    }
    (output_dir / "train_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
