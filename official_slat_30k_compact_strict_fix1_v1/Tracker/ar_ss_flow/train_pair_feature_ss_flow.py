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

from ar_ss_flow.correspondence_lifting import (  # noqa: E402
    CORRESPONDENCE_NEGATIVE_MODES,
    correspondence_pair_volume_from_sample,
    deterministic_view_subset,
    load_correspondence_checkpoint,
    pair_feature_dim,
    parse_csv,
)
from ar_ss_flow.local_pose_lifting_flow import PoseLiftingCacheDataset  # noqa: E402
from ar_ss_flow.pair_feature_ss_flow import (  # noqa: E402
    PAIR_FEATURE_SS_FLOW_VERSION,
    LocalPairFeatureVelocityAdapter,
    PairFeatureSSFlowModel,
)
from reconvggt_ar_adapter_a.pointpose_ss_condition import trainable_state_dict  # noqa: E402
from reconvggt_ar_adapter_a.train_pointpose_ss_lora import (  # noqa: E402
    finite_tree,
    gradients_finite,
    install_unused_model_stubs,
    optimizer_state_finite,
    parameters_finite,
)


PAIR_NEGATIVE_MODES = (*CORRESPONDENCE_NEGATIVE_MODES, "depth_corrupt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a frozen-stock local SS residual with explicit same-voxel "
            "view-pair features and actual correct/corrupt lifting branches."
        )
    )
    parser.add_argument("--cache_manifest", required=True)
    parser.add_argument("--correspondence_checkpoint", required=True)
    parser.add_argument("--correspondence_summary_report", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--pretrained", default="Stable-X/trellis-vggt-v0-2")
    parser.add_argument("--indices", default="0-15")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_steps", type=int, default=50)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--log_every", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5.0e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--amp_dtype", choices=("bf16", "fp16", "none"), default="bf16")
    parser.add_argument("--amp_init_scale", type=float, default=8192.0)
    parser.add_argument("--adapter_hidden_dim", type=int, default=96)
    parser.add_argument("--physical_scale", type=float, default=0.5)
    parser.add_argument("--residual_t_min", type=float, default=0.5)
    parser.add_argument("--residual_t_ramp", type=float, default=0.1)
    parser.add_argument("--train_t_min", type=float, default=0.5)
    parser.add_argument("--train_t_max", type=float, default=0.99)
    parser.add_argument("--confidence_floor", type=float, default=0.0)
    parser.add_argument("--neighborhood_radius", type=int, default=0)
    parser.add_argument("--min_source_views", type=int, default=2)
    parser.add_argument(
        "--negative_modes",
        default="pose_cyclic1,pose_cyclic2,pose_reverse,cross_sample",
    )
    parser.add_argument("--flow_weight", type=float, default=1.0)
    parser.add_argument("--wrong_stock_weight", type=float, default=0.25)
    parser.add_argument("--correct_gain_weight", type=float, default=0.10)
    parser.add_argument("--correct_gain_margin", type=float, default=0.002)
    parser.add_argument("--correct_wrong_rank_weight", type=float, default=0.10)
    parser.add_argument("--correct_wrong_margin", type=float, default=0.002)
    parser.add_argument("--delta_norm_weight", type=float, default=0.01)
    return parser.parse_args()


def find_cross_sample(
    dataset: PoseLiftingCacheDataset,
    source_index: int,
    source: dict[str, Any],
) -> dict[str, Any]:
    view_count = int(source["visual_patch_features"].shape[0])
    patch_shape = tuple(source["visual_patch_features"].shape[1:])
    candidates: list[tuple[int, int, dict[str, Any]]] = []
    for offset in range(1, len(dataset)):
        candidate = dataset[(source_index + offset) % len(dataset)]
        if str(candidate.get("object_uid")) == str(source.get("object_uid")):
            continue
        if tuple(candidate["visual_patch_features"].shape[1:]) != patch_shape:
            continue
        candidate_views = int(candidate["visual_patch_features"].shape[0])
        priority = 0 if candidate_views == view_count else (1 if candidate_views > view_count else 2)
        candidates.append((priority, offset, candidate))
    if not candidates:
        raise RuntimeError("no different-object cross-sample with matching patch features")
    _, _, candidate = min(candidates, key=lambda item: (item[0], item[1]))
    candidate_views = int(candidate["visual_patch_features"].shape[0])
    if candidate_views == view_count:
        indices = list(range(view_count))
        policy = "exact_view_count"
    elif candidate_views > view_count:
        indices = deterministic_view_subset(candidate_views, view_count)
        policy = "deterministic_subsample"
    else:
        indices = [index % candidate_views for index in range(view_count)]
        policy = "deterministic_cycle_pad"
    result = dict(candidate)
    result["visual_patch_features"] = candidate["visual_patch_features"].index_select(
        0, torch.as_tensor(indices, dtype=torch.long)
    )
    result["cross_sample_view_policy"] = policy
    result["cross_sample_source_view_count"] = view_count
    result["cross_sample_candidate_view_count"] = candidate_views
    return result


def validate_correspondence_summary(
    report_path: str | Path,
    checkpoint_path: str | Path,
) -> dict[str, Any]:
    report_path = Path(report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not bool(report.get("passed", False)):
        raise RuntimeError("C1 multi-seed correspondence summary did not pass")
    requested = Path(checkpoint_path).resolve()
    matched = None
    for row in report.get("seed_reports", []):
        run = Path(str(row.get("run", "")))
        if not run.is_absolute():
            run = TRACKER_ROOT / run
        candidate = (run / "checkpoints" / "last.pt").resolve()
        if candidate == requested:
            matched = row
            break
    if matched is None:
        raise RuntimeError(
            f"checkpoint {requested} is not covered by C1 summary {report_path}"
        )
    if not bool(matched.get("train_finite", False)) or int(
        matched.get("passed_mode_count", 0)
    ) < 2:
        raise RuntimeError(f"matched C1 seed row is not eligible: {matched}")
    return {
        "report": str(report_path.resolve()),
        "checkpoint": str(requested),
        "seed": int(matched["seed"]),
        "summary_passed": True,
        "passed_mode_count": int(matched["passed_mode_count"]),
    }


def build_model(
    args: argparse.Namespace,
    device: torch.device,
    visual_channels: int,
    pair_dim: int,
) -> tuple[Any, PairFeatureSSFlowModel, dict[str, Any]]:
    install_unused_model_stubs()
    pipeline = TrellisVGGTTo3DPipeline.from_pretrained(args.pretrained)
    sampler = pipeline.sparse_structure_sampler
    flow = pipeline.models["sparse_structure_flow_model"].to(device).eval()
    for parameter in flow.parameters():
        parameter.requires_grad = False
    if int(flow.resolution) != 16 or int(flow.in_channels) != 8 or int(flow.out_channels) != 8:
        raise RuntimeError("unexpected sparse structure Flow schema")
    adapter = LocalPairFeatureVelocityAdapter(
        visual_channels=int(visual_channels),
        pair_feature_dim=int(pair_dim),
        latent_channels=8,
        hidden_dim=int(args.adapter_hidden_dim),
        residual_t_min=float(args.residual_t_min),
        residual_t_ramp=float(args.residual_t_ramp),
    ).to(device)
    model = PairFeatureSSFlowModel(flow, adapter).to(device)
    trainable_names = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    unexpected = [name for name in trainable_names if not name.startswith("adapter.")]
    if unexpected or not trainable_names:
        raise RuntimeError(f"trainable whitelist failed: {unexpected}")
    summary = {
        "stage": "C3 state-conditioned same-voxel pair-feature SS residual",
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
        "slat_enabled": False,
    }
    del pipeline
    gc.collect()
    torch.cuda.empty_cache()
    return sampler, model, summary


def gradient_group_norms(model: PairFeatureSSFlowModel) -> dict[str, float]:
    groups = {
        "state": "adapter.state_projection.",
        "visual": "adapter.visual_projection.",
        "metadata": "adapter.metadata_projection.",
        "pair_attention": "adapter.pair_",
        "fusion": "adapter.fusion.",
        "output": "adapter.output.",
    }
    named = list(model.named_parameters())
    return {
        label: sum(
            float(parameter.grad.detach().float().square().sum().item())
            for name, parameter in named
            if name.startswith(prefix) and parameter.grad is not None
        )
        ** 0.5
        for label, prefix in groups.items()
    }


def save_checkpoint(
    path: Path,
    *,
    model: PairFeatureSSFlowModel,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    step: int,
    args: argparse.Namespace,
    summary: dict[str, Any],
    history: list[dict[str, Any]],
) -> None:
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters_finite(trainable):
        raise RuntimeError(f"refusing to save non-finite parameters: {path}")
    if not optimizer_state_finite(optimizer) or not finite_tree(scaler.state_dict()):
        raise RuntimeError(f"refusing to save non-finite optimizer/scaler state: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": PAIR_FEATURE_SS_FLOW_VERSION,
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


@torch.no_grad()
def stock_equivalence_audit(
    model: PairFeatureSSFlowModel,
    sampler: Any,
    sample: dict[str, Any],
    correspondence_model,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, float]:
    visual, metadata, pairs, pair_valid, _ = correspondence_pair_volume_from_sample(
        sample,
        device=device,
        model=correspondence_model,
        mode="correct",
        neighborhood_radius=int(args.neighborhood_radius),
        min_source_views=int(args.min_source_views),
        confidence_floor=float(args.confidence_floor),
    )
    condition = sample["stock_condition"].to(device=device)
    generator = torch.Generator(device=device).manual_seed(42015)
    target = sample["target"].unsqueeze(0).to(device=device, dtype=torch.float32)
    endpoint = torch.randn(target.shape, generator=generator, device=device)
    x_t, _ = sampler._get_model_gt(target, 0.75, endpoint)
    t = torch.tensor([750.0], device=device)
    stock = model.stock_prediction(x_t, t, condition)
    disabled, _ = model.adapt_from_stock(
        x_t,
        t,
        stock,
        visual,
        metadata,
        pairs,
        pair_valid,
        physical_present=False,
    )
    null, _ = model.adapt_from_stock(
        x_t,
        t,
        stock,
        torch.zeros_like(visual),
        torch.zeros_like(metadata),
        torch.zeros_like(pairs),
        torch.zeros_like(pair_valid),
    )
    enabled, _ = model.adapt_from_stock(
        x_t, t, stock, visual, metadata, pairs, pair_valid
    )
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
    if min(args.max_steps, args.save_every, args.log_every) <= 0:
        raise ValueError("step arguments must be positive")
    if not 0.0 < args.train_t_min < args.train_t_max < 1.0:
        raise ValueError("train_t_min/train_t_max must satisfy 0 < min < max < 1")
    modes = parse_csv(args.negative_modes)
    invalid = [mode for mode in modes if mode not in PAIR_NEGATIVE_MODES]
    if invalid:
        raise ValueError(f"invalid negative modes={invalid}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    random.seed(args.seed)
    np.random.seed(args.seed % (2**32 - 1))
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("C3 training requires CUDA")
    torch.cuda.set_device(0 if device.index is None else int(device.index))

    dataset = PoseLiftingCacheDataset(args.cache_manifest, indices=args.indices)
    audit = validate_correspondence_summary(
        args.correspondence_summary_report, args.correspondence_checkpoint
    )
    correspondence_model, correspondence_checkpoint = load_correspondence_checkpoint(
        args.correspondence_checkpoint,
        device=device,
        visual_channels=dataset.visual_feature_dim,
    )
    correspondence_model.eval()
    for parameter in correspondence_model.parameters():
        parameter.requires_grad = False
    pair_dim = pair_feature_dim(correspondence_model.pairwise_dim)
    sampler, model, summary = build_model(
        args, device, dataset.visual_feature_dim, pair_dim
    )
    summary.update(
        {
            "correspondence_checkpoint": str(
                Path(args.correspondence_checkpoint).resolve()
            ),
            "correspondence_checkpoint_step": int(
                correspondence_checkpoint.get("step", 0)
            ),
            "correspondence_summary_audit": audit,
            "stock_equivalence": stock_equivalence_audit(
                model, sampler, dataset[0], correspondence_model, device, args
            ),
        }
    )
    print(json.dumps(summary, indent=2), flush=True)

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
        cross_sample_policy = None
        if mode == "cross_sample":
            cross = find_cross_sample(dataset, sample_index, sample)
            visual_override = cross["visual_patch_features"]
            wrong_mode = "correct"
            cross_sample_policy = cross["cross_sample_view_policy"]
        with torch.no_grad():
            correct = correspondence_pair_volume_from_sample(
                sample,
                device=device,
                model=correspondence_model,
                mode="correct",
                neighborhood_radius=int(args.neighborhood_radius),
                min_source_views=int(args.min_source_views),
                confidence_floor=float(args.confidence_floor),
            )
            wrong = correspondence_pair_volume_from_sample(
                sample,
                device=device,
                model=correspondence_model,
                mode=wrong_mode,
                visual_patch_features_override=visual_override,
                neighborhood_radius=int(args.neighborhood_radius),
                min_source_views=int(args.min_source_views),
                confidence_floor=float(args.confidence_floor),
            )
            target = sample["target"].unsqueeze(0).to(
                device=device, dtype=torch.float32
            )
            condition = sample["stock_condition"].to(device=device)
            noise = torch.randn_like(target)
            t_value = random.uniform(float(args.train_t_min), float(args.train_t_max))
            x_t, gt_velocity = sampler._get_model_gt(target, t_value, noise)
            t_tensor = torch.full(
                (1,), 1000.0 * t_value, device=device, dtype=torch.float32
            )
            stock = model.stock_prediction(x_t, t_tensor, condition)

        with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
            correct_prediction, correct_stats = model.adapt_from_stock(
                x_t,
                t_tensor,
                stock,
                correct[0],
                correct[1],
                correct[2],
                correct[3],
                scale=float(args.physical_scale),
            )
            wrong_prediction, wrong_stats = model.adapt_from_stock(
                x_t,
                t_tensor,
                stock,
                wrong[0],
                wrong[1],
                wrong[2],
                wrong[3],
                scale=float(args.physical_scale),
            )
            correct_flow = F.mse_loss(correct_prediction.float(), gt_velocity.float())
            wrong_flow = F.mse_loss(wrong_prediction.float(), gt_velocity.float())
            stock_flow = F.mse_loss(stock.float(), gt_velocity.float()).detach()
            stock_energy = stock.float().square().mean().detach().clamp_min(1.0e-6)
            wrong_stock = F.mse_loss(wrong_prediction.float(), stock.float()) / stock_energy
            relative_gain = (stock_flow - correct_flow) / stock_flow.clamp_min(1.0e-6)
            gain_loss = F.relu(
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
                + float(args.correct_gain_weight) * gain_loss
                + float(args.correct_wrong_rank_weight) * rank_loss
                + float(args.delta_norm_weight) * delta_norm
            )

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        forward_finite = bool(torch.isfinite(loss.detach()).item())
        gradient_finite = gradients_finite(trainable)
        if not forward_finite or not gradient_finite:
            raise RuntimeError(
                f"non-finite C3 update step={step} forward={forward_finite} "
                f"gradient={gradient_finite}"
            )
        gradient_norms = gradient_group_norms(model)
        clip_total = torch.nn.utils.clip_grad_norm_(trainable, float(args.grad_clip))
        if not bool(torch.isfinite(clip_total).item()):
            raise RuntimeError(f"non-finite clipped gradient at step={step}")
        if scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        if not parameters_finite(trainable) or not optimizer_state_finite(optimizer):
            raise RuntimeError(f"non-finite C3 optimizer state at step={step}")

        row = {
            "step": step,
            "uid": str(sample["uid"]),
            "object_uid": str(sample.get("object_uid", sample["uid"])),
            "negative_mode": mode,
            "cross_sample_view_policy": cross_sample_policy,
            "loss": float(loss.detach().float().item()),
            "correct_flow": float(correct_flow.detach().float().item()),
            "wrong_flow": float(wrong_flow.detach().float().item()),
            "stock_flow": float(stock_flow.detach().float().item()),
            "relative_correct_gain": float(relative_gain.detach().float().item()),
            "relative_correct_vs_wrong": float(
                relative_correct_wrong.detach().float().item()
            ),
            "wrong_stock_loss": float(wrong_stock.detach().float().item()),
            "delta_norm": float(delta_norm.detach().float().item()),
            "correct_delta_rms": float(correct_stats["delta_rms"].float().item()),
            "wrong_delta_rms": float(wrong_stats["delta_rms"].float().item()),
            "correct_pair_valid_ratio": float(
                correct_stats["pair_valid_ratio"].float().item()
            ),
            "wrong_pair_valid_ratio": float(
                wrong_stats["pair_valid_ratio"].float().item()
            ),
            "correct_attention_entropy": float(
                correct_stats["pair_attention_entropy"].float().item()
            ),
            "wrong_attention_entropy": float(
                wrong_stats["pair_attention_entropy"].float().item()
            ),
            "correct_pair_count": int(correct[4]["pair_count"]),
            "wrong_pair_count": int(wrong[4]["pair_count"]),
            "gradient_norms": gradient_norms,
            "clip_total_norm": float(clip_total.detach().float().item()),
            "t": float(t_value),
            "elapsed_seconds": float(time.time() - start_time),
        }
        if step == 1 or step % int(args.log_every) == 0 or step == int(args.max_steps):
            history.append(row)
            print(f"[pair_feature_train] {json.dumps(row)}", flush=True)
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
        "format": PAIR_FEATURE_SS_FLOW_VERSION,
        "args": vars(args),
        "model_summary": summary,
        "dataset_size": len(dataset),
        "unique_object_count": len(
            {str(row.get("object_uid", row["uid"])) for row in dataset.rows}
        ),
        "completed_steps": int(args.max_steps),
        "finite": parameters_finite(trainable) and optimizer_state_finite(optimizer),
        "history": history,
        "checkpoint": str(output_dir / "checkpoints" / "last.pt"),
    }
    (output_dir / "train_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
