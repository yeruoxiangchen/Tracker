#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import time
from typing import Any

os.environ.setdefault("SPCONV_ALGO", "native")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
os.environ.setdefault("TORCH_EXTENSIONS_DIR", "/tmp/torch_extensions")

import numpy as np
import torch
import torch.nn.functional as F

from ar_ss_flow.correspondence_lifting import subset_sample_views
from ar_ss_flow.local_pose_lifting_flow import PoseLiftingCacheDataset
from pose_point_depth_mv.correspondence_head import (
    CORRESPONDENCE_CHECKPOINT_VERSION,
    DEFAULT_TRAIN_CONTROLS,
    VOXEL_CONTROL_RANKING_VERSION,
    VOXEL_RELIABILITY_COMPONENTS,
    VOXEL_RELIABILITY_WEIGHTING_VERSION,
    ViewCorrespondenceHead,
    correct_voxel_reliability_weight,
    correspondence_protocol_hash,
    parse_control_names,
    trainable_state_dict,
    voxel_control_ranking_loss,
)
from pose_point_depth_mv.view_identity_lifting import (
    SPATIAL_TOLERANCE_DEFINITION,
    SPATIAL_TOLERANCE_MODES,
    SPATIAL_TOLERANCE_VERSION,
    VIEW_IDENTITY_CONTROL_NAMES,
    apply_symmetric_spatial_tolerance,
    build_view_identity_evidence,
)
from reconvggt_ar_adapter_a.train_pointpose_ss_lora import (
    finite_tree,
    gradients_finite,
    optimizer_state_finite,
    parameters_finite,
    tensors_finite,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an SS-independent image-pose-depth correspondence head."
    )
    parser.add_argument("--cache_manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--indices", default="0-15")
    parser.add_argument("--max_steps", type=int, default=400)
    parser.add_argument("--save_every", type=int, default=50)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=2.0e-4)
    parser.add_argument("--weight_decay", type=float, default=1.0e-4)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--amp_dtype", choices=("fp16", "bf16", "none"), default="bf16")
    parser.add_argument("--amp_init_scale", type=float, default=8192.0)
    parser.add_argument("--nonfinite_policy", choices=("error", "skip"), default="error")
    parser.add_argument("--max_nonfinite_attempts", type=int, default=0)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--pair_hidden_dim", type=int, default=96)
    parser.add_argument("--min_views", type=int, default=2)
    parser.add_argument("--max_train_views", type=int, default=0)
    parser.add_argument(
        "--spatial_tolerance",
        choices=SPATIAL_TOLERANCE_MODES,
        default="exact",
        help=(
            "Symmetric local evidence aggregation used by correct and every "
            "training control. gaussian3 is the C0.3 neighborhood protocol."
        ),
    )
    parser.add_argument(
        "--train_controls", default=",".join(DEFAULT_TRAIN_CONTROLS)
    )
    parser.add_argument("--sample_bce_weight", type=float, default=1.0)
    parser.add_argument("--sample_rank_weight", type=float, default=1.0)
    parser.add_argument("--voxel_bce_weight", type=float, default=0.25)
    parser.add_argument("--hard_negative_weight", type=float, default=0.5)
    parser.add_argument("--rank_margin", type=float, default=0.25)
    parser.add_argument("--voxel_rank_weight", type=float, default=0.0)
    parser.add_argument("--voxel_rank_margin", type=float, default=0.25)
    parser.add_argument("--voxel_rank_temperature", type=float, default=0.10)
    parser.add_argument("--voxel_rank_hard_weight", type=float, default=0.5)
    parser.add_argument(
        "--voxel_reliability_weighting",
        choices=("uniform", "correct_geometry"),
        default="uniform",
    )
    parser.add_argument("--voxel_reliability_floor", type=float, default=0.10)
    parser.add_argument("--voxel_reliability_power", type=float, default=1.0)
    return parser.parse_args()


def choose_view_subset(
    sample: dict[str, Any],
    *,
    rng: random.Random,
    min_views: int,
    max_views: int,
) -> dict[str, Any]:
    view_count = int(sample["visual_patch_features"].shape[0])
    if view_count < int(min_views):
        raise RuntimeError(
            f"uid={sample.get('uid')} has {view_count} views < min_views={min_views}"
        )
    upper = view_count if int(max_views) <= 0 else min(view_count, int(max_views))
    requested = rng.randint(int(min_views), upper)
    if requested == view_count:
        return sample
    indices = sorted(rng.sample(range(view_count), requested))
    return subset_sample_views(sample, indices)


def gradient_group_norms(head: ViewCorrespondenceHead) -> dict[str, float]:
    groups = {
        "visual": ("visual_encoder.",),
        "geometry": ("geometry_encoder.",),
        "joint": ("joint_encoder.",),
        "pair": ("pair_encoder.",),
        "score": ("pair_score.",),
    }
    named = list(head.named_parameters())
    return {
        label: sum(
            float(parameter.grad.detach().float().square().sum().item())
            for name, parameter in named
            if any(name.startswith(prefix) for prefix in prefixes)
            and parameter.grad is not None
        )
        ** 0.5
        for label, prefixes in groups.items()
    }


def save_checkpoint(
    path: Path,
    *,
    head: ViewCorrespondenceHead,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    step: int,
    args: argparse.Namespace,
    model_summary: dict[str, Any],
) -> None:
    trainable = [parameter for parameter in head.parameters() if parameter.requires_grad]
    if not parameters_finite(trainable):
        raise RuntimeError(f"refusing to save non-finite correspondence head: {path}")
    if not optimizer_state_finite(optimizer):
        raise RuntimeError(f"refusing to save non-finite optimizer: {path}")
    if not finite_tree(scaler.state_dict()):
        raise RuntimeError(f"refusing to save non-finite scaler: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": CORRESPONDENCE_CHECKPOINT_VERSION,
            "step": int(step),
            "model_trainable_state": trainable_state_dict(head),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "args": vars(args),
            "model_summary": model_summary,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise RuntimeError("C0 correspondence training currently requires one GPU")
    if not torch.cuda.is_available():
        raise RuntimeError("C0 correspondence training requires CUDA")
    if min(args.max_steps, args.save_every, args.log_every, args.grad_accum) <= 0:
        raise ValueError("step/log/accumulation values must be positive")
    if min(
        args.sample_bce_weight,
        args.sample_rank_weight,
        args.voxel_bce_weight,
        args.hard_negative_weight,
        args.rank_margin,
        args.voxel_rank_weight,
        args.voxel_rank_margin,
        args.voxel_rank_hard_weight,
    ) < 0.0:
        raise ValueError("loss weights and margins must be non-negative")
    if float(args.voxel_rank_temperature) <= 0.0:
        raise ValueError("voxel rank temperature must be positive")
    if not 0.0 <= float(args.voxel_reliability_floor) < 1.0:
        raise ValueError("voxel reliability floor must be in [0,1)")
    if float(args.voxel_reliability_power) <= 0.0:
        raise ValueError("voxel reliability power must be positive")
    train_controls = parse_control_names(args.train_controls)
    unknown = sorted(set(train_controls) - set(VIEW_IDENTITY_CONTROL_NAMES))
    if unknown:
        raise ValueError(f"unknown train controls: {unknown}")

    rng = random.Random(int(args.seed))
    np.random.seed(int(args.seed) % (2**32 - 1))
    torch.manual_seed(int(args.seed))
    torch.cuda.manual_seed_all(int(args.seed))
    device = torch.device("cuda")
    dataset = PoseLiftingCacheDataset(args.cache_manifest, indices=args.indices)
    head = ViewCorrespondenceHead(
        visual_channels=dataset.visual_feature_dim,
        hidden_dim=int(args.hidden_dim),
        pair_hidden_dim=int(args.pair_hidden_dim),
        min_views=int(args.min_views),
    ).to(device)
    trainable = [parameter for parameter in head.parameters() if parameter.requires_grad]
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
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    object_uids = sorted(
        {str(row.get("object_uid", row["uid"])) for row in dataset.rows}
    )
    protocol = {
        "train_controls": list(train_controls),
        "fixed_correct_view_weight": True,
        "sample_bce_weight": float(args.sample_bce_weight),
        "sample_rank_weight": float(args.sample_rank_weight),
        "voxel_bce_weight": float(args.voxel_bce_weight),
        "hard_negative_weight": float(args.hard_negative_weight),
        "rank_margin": float(args.rank_margin),
        "voxel_control_ranking_version": VOXEL_CONTROL_RANKING_VERSION,
        "voxel_rank_weight": float(args.voxel_rank_weight),
        "voxel_rank_margin": float(args.voxel_rank_margin),
        "voxel_rank_temperature": float(args.voxel_rank_temperature),
        "voxel_rank_hard_weight": float(args.voxel_rank_hard_weight),
        "voxel_rank_weighting": str(args.voxel_reliability_weighting),
        "voxel_reliability_weighting_version": (
            VOXEL_RELIABILITY_WEIGHTING_VERSION
        ),
        "voxel_reliability_components": list(VOXEL_RELIABILITY_COMPONENTS),
        "voxel_reliability_source": "detached_correct_evidence_only",
        "voxel_reliability_shared_across_controls": True,
        "voxel_reliability_floor": float(args.voxel_reliability_floor),
        "voxel_reliability_power": float(args.voxel_reliability_power),
        "min_views": int(args.min_views),
        "max_train_views": int(args.max_train_views),
        "training_spatial_tolerance": str(args.spatial_tolerance),
        "spatial_tolerance_version": (
            None
            if args.spatial_tolerance == "exact"
            else SPATIAL_TOLERANCE_VERSION
        ),
        "spatial_tolerance_definition": (
            None
            if args.spatial_tolerance == "exact"
            else SPATIAL_TOLERANCE_DEFINITION
        ),
        "spatial_tolerance_symmetric_across_branches": True,
        "spatial_tolerance_fixed_correct_support": True,
    }
    model_summary = {
        "stage": (
            "C0.3 neighborhood-aware voxel correspondence head"
            if args.spatial_tolerance != "exact"
            else (
                "C0.2b reliability-weighted voxel correspondence head"
                if args.voxel_reliability_weighting != "uniform"
                else (
                    "C0.2a voxel-ranked explicit view correspondence head"
                    if float(args.voxel_rank_weight) > 0.0
                    else "C0 explicit view correspondence head"
                )
            )
        ),
        "head": head.metadata(),
        "protocol": protocol,
        "protocol_hash": correspondence_protocol_hash(protocol),
        "cache_config_hash": dataset.config_hash,
        "cache_manifest": str(dataset.manifest_path.resolve()),
        "dataset_size": len(dataset),
        "unique_object_count": len(object_uids),
        "train_object_uids": object_uids,
        "trainable_parameter_count": int(sum(p.numel() for p in trainable)),
        "trainable_parameter_names": [
            name for name, parameter in head.named_parameters() if parameter.requires_grad
        ],
        "uses_target": False,
        "uses_stock_condition": False,
        "uses_ss_flow": False,
    }
    print(json.dumps(model_summary, indent=2), flush=True)

    use_amp = args.amp_dtype != "none"
    amp_dtype = torch.float16 if args.amp_dtype == "fp16" else torch.bfloat16
    global_step = 0
    micro_step = 0
    applied_updates = 0
    nonfinite_attempts = 0
    history: list[dict[str, Any]] = []
    start_time = time.time()
    optimizer.zero_grad(set_to_none=True)

    while global_step < int(args.max_steps):
        sample = dataset[rng.randrange(len(dataset))]
        sample = choose_view_subset(
            sample,
            rng=rng,
            min_views=int(args.min_views),
            max_views=int(args.max_train_views),
        )
        correct = build_view_identity_evidence(sample, device=device, mode="correct")
        controls = {
            mode: build_view_identity_evidence(sample, device=device, mode=mode)
            for mode in train_controls
        }
        exact_fixed_weight = correct["view_weight"].float()
        if args.spatial_tolerance != "exact":
            correct, fixed_weight = apply_symmetric_spatial_tolerance(
                correct,
                fixed_correct_weight=exact_fixed_weight,
                mode=str(args.spatial_tolerance),
            )
            controls = {
                mode: apply_symmetric_spatial_tolerance(
                    evidence,
                    fixed_correct_weight=exact_fixed_weight,
                    mode=str(args.spatial_tolerance),
                )[0]
                for mode, evidence in controls.items()
            }
        else:
            fixed_weight = exact_fixed_weight
        with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
            correct_result = head(correct, view_weight_override=fixed_weight)
            control_results = {
                mode: head(evidence, view_weight_override=fixed_weight)
                for mode, evidence in controls.items()
            }
            ordered_control_results = tuple(
                control_results[mode] for mode in train_controls
            )
            correct_score = correct_result["sample_score"].float()
            control_scores = torch.stack(
                tuple(result["sample_score"].float() for result in ordered_control_results)
            )
            sample_bce = F.binary_cross_entropy_with_logits(
                correct_score, torch.ones_like(correct_score)
            ) + F.binary_cross_entropy_with_logits(
                control_scores, torch.zeros_like(control_scores)
            )
            margins = correct_score - control_scores
            sample_rank = F.relu(float(args.rank_margin) - margins).mean()
            hard_rank = F.relu(float(args.rank_margin) - margins.min())
            active = correct_result["active_mask"]
            correct_voxel = correct_result["voxel_score"][active].float()
            voxel_bce = F.binary_cross_entropy_with_logits(
                correct_voxel, torch.ones_like(correct_voxel)
            )
            voxel_bce = voxel_bce + torch.stack(
                tuple(
                    F.binary_cross_entropy_with_logits(
                        result["voxel_score"][active].float(),
                        torch.zeros_like(correct_voxel),
                    )
                    for result in ordered_control_results
                )
            ).mean()
            voxel_control_scores = torch.stack(
                tuple(result["voxel_score"].float() for result in ordered_control_results)
            )
            voxel_reliability = None
            if args.voxel_reliability_weighting == "correct_geometry":
                voxel_reliability = correct_voxel_reliability_weight(
                    correct,
                    active,
                    min_views=int(args.min_views),
                    floor=float(args.voxel_reliability_floor),
                    power=float(args.voxel_reliability_power),
                )
            voxel_ranking = voxel_control_ranking_loss(
                correct_result["voxel_score"],
                voxel_control_scores,
                active,
                margin=float(args.voxel_rank_margin),
                temperature=float(args.voxel_rank_temperature),
                hard_weight=float(args.voxel_rank_hard_weight),
                voxel_weight=(
                    None
                    if voxel_reliability is None
                    else voxel_reliability["weight"]
                ),
            )
            loss = (
                float(args.sample_bce_weight) * sample_bce
                + float(args.sample_rank_weight) * sample_rank
                + float(args.voxel_bce_weight) * voxel_bce
                + float(args.hard_negative_weight) * hard_rank
                + float(args.voxel_rank_weight) * voxel_ranking["loss"]
            )
            scaled_loss = loss / float(args.grad_accum)
        scaler.scale(scaled_loss).backward()
        sync_step = ((micro_step + 1) % int(args.grad_accum)) == 0
        if sync_step:
            scaler.unscale_(optimizer)
            diagnostics = [
                loss,
                sample_bce,
                sample_rank,
                hard_rank,
                voxel_bce,
                voxel_ranking["loss"],
                voxel_ranking["all_control_loss"],
                voxel_ranking["smooth_hard_loss"],
                voxel_ranking["hard_margin_mean"],
                voxel_ranking["hard_positive_ratio"],
                correct_score,
                correct_result["support_ratio"],
                *control_scores,
                *margins,
                *voxel_ranking["per_control_losses"],
                *voxel_ranking["control_margin_means"],
            ]
            if voxel_reliability is not None:
                diagnostics.extend(
                    (
                        voxel_reliability["active_weight_mean"],
                        voxel_reliability["active_weight_min"],
                        voxel_reliability["active_weight_max"],
                        voxel_reliability["effective_fraction"],
                    )
                )
            forward_finite = tensors_finite(diagnostics)
            gradient_finite = gradients_finite(trainable)
            update_finite = forward_finite and gradient_finite
            scaler_before = float(scaler.get_scale()) if scaler.is_enabled() else None
            optimizer_step_applied = False
            clip_total_norm = None
            gradient_norms = gradient_group_norms(head)
            if update_finite:
                clip = torch.nn.utils.clip_grad_norm_(trainable, float(args.grad_clip))
                clip_total_norm = float(clip.detach().float().item())
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer_step_applied = True
                global_step += 1
                applied_updates += 1
                if not parameters_finite(trainable) or not optimizer_state_finite(optimizer):
                    raise RuntimeError("C0 optimizer produced non-finite state")
            else:
                nonfinite_attempts += 1
                if scaler.is_enabled():
                    scaler.update()
            scaler_after = float(scaler.get_scale()) if scaler.is_enabled() else None
            optimizer.zero_grad(set_to_none=True)
            row = {
                "step": global_step,
                "micro_step": micro_step + 1,
                "uid": str(sample["uid"]),
                "views": int(correct["views"]),
                "spatial_tolerance": str(args.spatial_tolerance),
                "exact_view_support_ratio": float(
                    exact_fixed_weight.gt(1.0e-6).float().mean().item()
                ),
                "effective_view_support_ratio": float(
                    fixed_weight.gt(1.0e-6).float().mean().item()
                ),
                "loss": float(loss.detach().item()),
                "sample_bce": float(sample_bce.detach().item()),
                "sample_rank": float(sample_rank.detach().item()),
                "hard_rank": float(hard_rank.detach().item()),
                "voxel_bce": float(voxel_bce.detach().item()),
                "voxel_rank": float(voxel_ranking["loss"].detach().item()),
                "voxel_all_control_rank": float(
                    voxel_ranking["all_control_loss"].detach().item()
                ),
                "voxel_smooth_hard_rank": float(
                    voxel_ranking["smooth_hard_loss"].detach().item()
                ),
                "voxel_hard_margin_mean": float(
                    voxel_ranking["hard_margin_mean"].detach().item()
                ),
                "voxel_hard_positive_ratio": float(
                    voxel_ranking["hard_positive_ratio"].detach().item()
                ),
                "correct_score": float(correct_score.detach().item()),
                "control_scores": {
                    mode: float(result["sample_score"].detach().float().item())
                    for mode, result in control_results.items()
                },
                "margins": {
                    mode: float((correct_score - result["sample_score"].float()).detach().item())
                    for mode, result in control_results.items()
                },
                "voxel_control_losses": {
                    mode: float(voxel_ranking["per_control_losses"][index].detach().item())
                    for index, mode in enumerate(train_controls)
                },
                "voxel_control_margin_means": {
                    mode: float(
                        voxel_ranking["control_margin_means"][index].detach().item()
                    )
                    for index, mode in enumerate(train_controls)
                },
                "support_ratio": float(correct_result["support_ratio"].detach().item()),
                "voxel_reliability": (
                    None
                    if voxel_reliability is None
                    else {
                        key: float(voxel_reliability[key].detach().item())
                        for key in (
                            "active_weight_mean",
                            "active_weight_min",
                            "active_weight_max",
                            "effective_fraction",
                        )
                    }
                ),
                "gradient_norms": gradient_norms,
                "clip_total_norm": clip_total_norm,
                "forward_finite": bool(forward_finite),
                "gradient_finite": bool(gradient_finite),
                "update_finite": bool(update_finite),
                "optimizer_step_applied": bool(optimizer_step_applied),
                "nonfinite_attempts": nonfinite_attempts,
                "scaler_before": scaler_before,
                "scaler_after": scaler_after,
                "elapsed_seconds": time.time() - start_time,
            }
            if global_step <= 1 or global_step % int(args.log_every) == 0 or not update_finite:
                history.append(row)
                print(f"[correspondence_train] {json.dumps(row)}", flush=True)
            if not update_finite:
                message = (
                    "non-finite C0 update "
                    f"attempt={nonfinite_attempts} micro_step={micro_step + 1}"
                )
                if (
                    args.nonfinite_policy == "error"
                    or nonfinite_attempts > int(args.max_nonfinite_attempts)
                ):
                    raise RuntimeError(message)
            if optimizer_step_applied and (
                global_step % int(args.save_every) == 0
                or global_step == int(args.max_steps)
            ):
                save_checkpoint(
                    output_dir / "checkpoints" / f"step_{global_step:06d}.pt",
                    head=head,
                    optimizer=optimizer,
                    scaler=scaler,
                    step=global_step,
                    args=args,
                    model_summary=model_summary,
                )
                save_checkpoint(
                    output_dir / "checkpoints" / "last.pt",
                    head=head,
                    optimizer=optimizer,
                    scaler=scaler,
                    step=global_step,
                    args=args,
                    model_summary=model_summary,
                )
        micro_step += 1

    report = {
        "stage": model_summary["stage"],
        "args": vars(args),
        "model_summary": model_summary,
        "start_global_step": 0,
        "completed_global_step": global_step,
        "applied_optimizer_updates": applied_updates,
        "nonfinite_attempts": nonfinite_attempts,
        "elapsed_seconds": time.time() - start_time,
        "history": history,
    }
    (output_dir / "train_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "completed_global_step": global_step,
                "applied_optimizer_updates": applied_updates,
                "nonfinite_attempts": nonfinite_attempts,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
