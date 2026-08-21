#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
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

from ar_ss_flow.correspondence_lifting import (
    CORRESPONDENCE_NEGATIVE_MODES,
    LocalVoxelCorrespondence,
    evidence_from_sample,
    parse_csv,
    protocol_hash,
    save_correspondence_checkpoint,
)
from ar_ss_flow.local_pose_lifting_flow import PoseLiftingCacheDataset


def masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if value.shape != mask.shape:
        raise ValueError(f"masked_mean shape mismatch: {value.shape} vs {mask.shape}")
    if not bool(mask.any().item()):
        return value.new_tensor(float("nan"))
    return value[mask].mean()


def select_voxels(mask: torch.Tensor, maximum: int, generator: torch.Generator) -> torch.Tensor:
    ids = torch.nonzero(mask, as_tuple=False)[:, 0]
    if maximum > 0 and ids.numel() > maximum:
        order = torch.randperm(ids.numel(), generator=generator, device=ids.device)
        ids = ids[order[:maximum]]
    selected = torch.zeros_like(mask)
    selected[ids] = True
    return selected


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
    raise RuntimeError("could not find a cross-sample negative with matching view/patch shape")


def build_training_schedule(
    *,
    dataset_size: int,
    negative_modes: tuple[str, ...],
    cross_sample_fraction: float,
    rng: random.Random,
) -> list[tuple[int, str]]:
    """Build one balanced epoch of object/negative pairs.

    Every object sees every pose hard negative once.  Cross-sample is optional
    and capped by ``cross_sample_fraction`` so the easy semantic negative cannot
    dominate training.
    """

    pose_modes = [mode for mode in negative_modes if mode != "cross_sample"]
    if not pose_modes:
        raise ValueError("at least one pose hard negative is required")
    schedule = [
        (sample_index, mode)
        for sample_index in range(dataset_size)
        for mode in pose_modes
    ]
    if "cross_sample" in negative_modes and cross_sample_fraction > 0.0:
        fraction = min(max(float(cross_sample_fraction), 0.0), 0.5)
        cross_count = int(round(len(schedule) * fraction / max(1.0 - fraction, 1.0e-6)))
        schedule.extend(
            (rng.randrange(dataset_size), "cross_sample")
            for _ in range(cross_count)
        )
    rng.shuffle(schedule)
    return schedule


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the input-only leave-one-view-out correspondence model. "
            "No ReconViaGen Flow weights are loaded or trained."
        )
    )
    parser.add_argument("--cache_manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--indices", default="0-15")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument("--save_every", type=int, default=50)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2.0e-4)
    parser.add_argument("--weight_decay", type=float, default=1.0e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--amp_dtype", choices=("bf16", "fp16", "none"), default="bf16")
    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--geometry_hidden_dim", type=int, default=48)
    parser.add_argument("--pairwise_dim", type=int, default=64)
    parser.add_argument("--score_hidden_dim", type=int, default=192)
    parser.add_argument("--negative_modes", default=",".join(CORRESPONDENCE_NEGATIVE_MODES))
    parser.add_argument(
        "--cross_sample_fraction",
        type=float,
        default=0.10,
        help="Fraction of easy cross-object negatives; pose negatives remain exhaustive per object.",
    )
    parser.add_argument("--neighborhood_radius", type=int, default=1)
    parser.add_argument("--min_source_views", type=int, default=2)
    parser.add_argument("--max_voxels_per_step", type=int, default=2048)
    parser.add_argument("--min_common_voxels", type=int, default=64)
    parser.add_argument("--rank_margin", type=float, default=0.05)
    parser.add_argument("--pairwise_margin", type=float, default=0.02)
    parser.add_argument("--score_margin", type=float, default=0.10)
    parser.add_argument("--shortcut_margin", type=float, default=0.02)
    parser.add_argument("--reprojection_weight", type=float, default=1.0)
    parser.add_argument("--rank_weight", type=float, default=1.0)
    parser.add_argument("--pairwise_rank_weight", type=float, default=1.0)
    parser.add_argument("--pairwise_bce_weight", type=float, default=0.10)
    parser.add_argument("--score_rank_weight", type=float, default=0.0)
    parser.add_argument("--score_bce_weight", type=float, default=0.0)
    parser.add_argument("--geometry_bce_weight", type=float, default=0.0)
    parser.add_argument("--anti_shortcut_weight", type=float, default=0.0)
    parser.add_argument("--variance_weight", type=float, default=0.01)
    parser.add_argument("--variance_floor", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    if min(args.max_steps, args.save_every, args.log_every) <= 0:
        raise ValueError("steps/save/log must be positive")
    negative_modes = parse_csv(args.negative_modes)
    invalid = [mode for mode in negative_modes if mode not in CORRESPONDENCE_NEGATIVE_MODES]
    if invalid:
        raise ValueError(f"invalid negative modes={invalid}")
    if not 0.0 <= float(args.cross_sample_fraction) <= 0.5:
        raise ValueError("cross_sample_fraction must be in [0, 0.5]")

    seed = int(args.seed)
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device(args.device)
    dataset = PoseLiftingCacheDataset(args.cache_manifest, indices=args.indices)
    first = dataset[0]
    visual_channels = int(first["visual_patch_features"].shape[-1])
    model = LocalVoxelCorrespondence(
        visual_channels=visual_channels,
        embedding_dim=int(args.embedding_dim),
        pairwise_dim=int(args.pairwise_dim),
        geometry_hidden_dim=int(args.geometry_hidden_dim),
        score_hidden_dim=int(args.score_hidden_dim),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
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
    mode_counts = {mode: 0 for mode in negative_modes}
    skipped = 0
    start_time = time.time()
    model.train()

    protocol = {
        "stage": "C1 pairwise-before-aggregation held-out training v3",
        "cache_manifest": str(Path(args.cache_manifest).resolve()),
        "indices": args.indices,
        "negative_modes": list(negative_modes),
        "cross_sample_fraction": float(args.cross_sample_fraction),
        "schedule": "all_object_pose_pairs_plus_capped_cross_sample",
        "fixed_heldout_target": True,
        "source_only_pose_corruption": True,
        "pairwise_before_aggregation": True,
        "primary_objective": "reprojection_rank_plus_pairwise_rank",
        "neighborhood_radius": int(args.neighborhood_radius),
        "min_source_views": int(args.min_source_views),
        "visual_channels": visual_channels,
        "model": model.metadata(),
    }
    protocol["protocol_hash"] = protocol_hash(protocol)
    (output_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8"
    )

    update_step = 0
    attempt = 0
    max_attempts = max(int(args.max_steps) * 20, int(args.max_steps) + 10)
    schedule_rng = random.Random(seed + 0xC011E5)
    schedule = build_training_schedule(
        dataset_size=len(dataset),
        negative_modes=negative_modes,
        cross_sample_fraction=float(args.cross_sample_fraction),
        rng=schedule_rng,
    )
    schedule_cursor = 0
    schedule_epoch = 0
    while update_step < int(args.max_steps):
        attempt += 1
        if attempt > max_attempts:
            raise RuntimeError(
                f"could not obtain {args.max_steps} valid updates after {max_attempts} attempts"
            )
        if schedule_cursor >= len(schedule):
            schedule_epoch += 1
            schedule = build_training_schedule(
                dataset_size=len(dataset),
                negative_modes=negative_modes,
                cross_sample_fraction=float(args.cross_sample_fraction),
                rng=schedule_rng,
            )
            schedule_cursor = 0
        sample_index, mode = schedule[schedule_cursor]
        schedule_cursor += 1
        sample = dataset[sample_index]
        views = int(sample["visual_patch_features"].shape[0])
        if views < int(args.min_source_views) + 1:
            skipped += 1
            continue
        heldout = schedule_rng.randrange(views)
        visual_override = None
        wrong_mode = mode
        if mode == "cross_sample":
            cross = find_cross_sample(dataset, sample_index, sample)
            visual_override = cross["visual_patch_features"]
            wrong_mode = "correct"

        with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
            correct_evidence = evidence_from_sample(
                sample, device=device, mode="correct", heldout_index=heldout
            )
            correct_maps = model.encode_patch_maps(
                correct_evidence["visual_patch_features"]
            )
            wrong_evidence = evidence_from_sample(
                sample,
                device=device,
                mode=wrong_mode,
                visual_patch_features_override=visual_override,
                heldout_index=heldout,
            )
            wrong_maps = model.encode_patch_maps(wrong_evidence["visual_patch_features"])
            correct = model.evaluate_heldout(
                correct_evidence,
                heldout,
                neighborhood_radius=int(args.neighborhood_radius),
                min_source_views=int(args.min_source_views),
                encoded_patch_maps=correct_maps,
                target_evidence=correct_evidence,
                target_encoded_patch_maps=correct_maps,
                detach_target=True,
            )
            wrong = model.evaluate_heldout(
                wrong_evidence,
                heldout,
                neighborhood_radius=int(args.neighborhood_radius),
                min_source_views=int(args.min_source_views),
                encoded_patch_maps=wrong_maps,
                target_evidence=correct_evidence,
                target_encoded_patch_maps=correct_maps,
                detach_target=True,
            )
            common = correct.valid_mask & wrong.valid_mask
            generator = torch.Generator(device=device).manual_seed(
                seed * 1000003 + attempt * 9176 + heldout
            )
            common = select_voxels(common, int(args.max_voxels_per_step), generator)
            common_count = int(common.sum().item())
            if common_count < int(args.min_common_voxels):
                skipped += 1
                continue

            correct_error = masked_mean(correct.error, common)
            wrong_error = masked_mean(wrong.error, common)
            normalized_voxel_advantage = (
                (wrong.error - correct.error)
                / (wrong.error + correct.error).clamp_min(1.0e-6)
            )
            rank = masked_mean(
                F.relu(float(args.rank_margin) - normalized_voxel_advantage), common
            )
            correct_pairwise = correct.pairwise_confidence[common]
            wrong_pairwise = wrong.pairwise_confidence[common]
            pairwise_advantage = correct_pairwise - wrong_pairwise
            pairwise_rank = F.relu(
                float(args.pairwise_margin) - pairwise_advantage
            ).mean()
            pairwise_bce = 0.5 * (
                F.binary_cross_entropy_with_logits(
                    correct.pairwise_logit[common],
                    torch.ones_like(correct.pairwise_logit[common]),
                )
                + F.binary_cross_entropy_with_logits(
                    wrong.pairwise_logit[common],
                    torch.zeros_like(wrong.pairwise_logit[common]),
                )
            )
            correct_logits = correct.confidence_logit[common]
            wrong_logits = wrong.confidence_logit[common]
            score_bce = 0.5 * (
                F.binary_cross_entropy_with_logits(
                    correct_logits, torch.ones_like(correct_logits)
                )
                + F.binary_cross_entropy_with_logits(
                    wrong_logits, torch.zeros_like(wrong_logits)
                )
            )
            geometry_correct = correct.geometry_logit[common]
            geometry_wrong = wrong.geometry_logit[common]
            geometry_bce = 0.5 * (
                F.binary_cross_entropy_with_logits(
                    geometry_correct, torch.ones_like(geometry_correct)
                )
                + F.binary_cross_entropy_with_logits(
                    geometry_wrong, torch.zeros_like(geometry_wrong)
                )
            )
            visual_advantage = correct_logits - wrong_logits
            geometry_advantage = geometry_correct - geometry_wrong
            score_margin_loss = F.relu(
                float(args.score_margin) - visual_advantage
            ).mean()
            anti_shortcut = F.relu(
                float(args.shortcut_margin)
                - (visual_advantage - geometry_advantage.detach())
            ).mean()
            embeddings = torch.cat(
                (correct.reconstruction[common], correct.target[common]), dim=0
            )
            feature_std = embeddings.float().std(dim=0, unbiased=False)
            variance_loss = F.relu(
                embeddings.new_tensor(float(args.variance_floor)) - feature_std
            ).mean()
            loss = (
                float(args.reprojection_weight) * correct_error
                + float(args.rank_weight) * rank
                + float(args.pairwise_rank_weight) * pairwise_rank
                + float(args.pairwise_bce_weight) * pairwise_bce
                + float(args.score_rank_weight) * score_margin_loss
                + float(args.score_bce_weight) * score_bce
                + float(args.geometry_bce_weight) * geometry_bce
                + float(args.anti_shortcut_weight) * anti_shortcut
                + float(args.variance_weight) * variance_loss
            )

        update_step += 1
        step = update_step
        if not bool(torch.isfinite(loss).item()):
            raise RuntimeError(f"non-finite correspondence loss at step={step}")
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(args.grad_clip)
        )
        if not bool(torch.isfinite(grad_norm).item()):
            raise RuntimeError(f"non-finite correspondence gradient at step={step}")
        scaler.step(optimizer)
        scaler.update()
        mode_counts[mode] += 1

        normalized_advantage = (
            (wrong_error - correct_error)
            / (wrong_error + correct_error).clamp_min(1.0e-6)
        )
        row = {
            "step": step,
            "attempt": attempt,
            "uid": str(sample["uid"]),
            "object_uid": str(sample.get("object_uid", sample["uid"])),
            "heldout": heldout,
            "negative_mode": mode,
            "common_voxels": common_count,
            "loss": float(loss.detach().float().item()),
            "correct_error": float(correct_error.detach().float().item()),
            "wrong_error": float(wrong_error.detach().float().item()),
            "normalized_advantage": float(normalized_advantage.detach().float().item()),
            "pairwise_confidence_advantage": float(pairwise_advantage.detach().float().mean().item()),
            "pairwise_rank_loss": float(pairwise_rank.detach().float().item()),
            "pairwise_bce_loss": float(pairwise_bce.detach().float().item()),
            "visual_score_advantage": float(visual_advantage.detach().float().mean().item()),
            "geometry_score_advantage": float(
                geometry_advantage.detach().float().mean().item()
            ),
            "anti_shortcut_loss": float(anti_shortcut.detach().float().item()),
            "variance_loss": float(variance_loss.detach().float().item()),
            "grad_norm": float(grad_norm.detach().float().item()),
            "elapsed_seconds": float(time.time() - start_time),
        }
        if step == 1 or step % int(args.log_every) == 0 or step == int(args.max_steps):
            history.append(row)
            print(f"[heldout_corr_train] {json.dumps(row)}", flush=True)
        if step % int(args.save_every) == 0 or step == int(args.max_steps):
            save_correspondence_checkpoint(
                output_dir / "checkpoints" / f"step_{step:06d}.pt",
                model=model,
                optimizer=optimizer,
                step=step,
                args=vars(args),
                history=history,
            )
            save_correspondence_checkpoint(
                output_dir / "checkpoints" / "last.pt",
                model=model,
                optimizer=optimizer,
                step=step,
                args=vars(args),
                history=history,
            )

    report = {
        "stage": "C1 pairwise-before-aggregation held-out training v3",
        "args": vars(args),
        "protocol": protocol,
        "dataset_size": len(dataset),
        "unique_object_count": len(
            {str(row.get("object_uid", row["uid"])) for row in dataset.rows}
        ),
        "completed_steps": update_step,
        "attempts": attempt,
        "skipped_steps": skipped,
        "mode_counts": mode_counts,
        "history": history,
        "checkpoint": str(output_dir / "checkpoints" / "last.pt"),
        "finite": all(
            bool(torch.isfinite(parameter.detach().float()).all().item())
            for parameter in model.parameters()
        ),
    }
    (output_dir / "train_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
