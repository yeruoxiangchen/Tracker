from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm.auto import tqdm

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("ATTN_BACKEND", "flash_attn")

TRACKER_ROOT = Path(__file__).resolve().parents[1]
PIXAL3D_ROOT = TRACKER_ROOT / "Pixal3D"
sys.path.insert(0, str(TRACKER_ROOT))
sys.path.insert(0, str(PIXAL3D_ROOT))

from pixal3d_multiview.pose_consistency_head import build_pose_consistency_head, load_pose_consistency_head  # noqa: E402
from pixal3d_multiview.sparse_condition import SparseMultiviewConditionBuilder  # noqa: E402
from pixal3d_multiview.train_sparse_multiview import (  # noqa: E402
    IMAGE_COND_CONFIG,
    MultiviewSparseManifestDataset,
    apply_pose_mode,
    build_image_cond_model,
    collate_single,
    make_multiview_condition,
    parse_sample_indices,
)


POSE_CONSISTENCY_MODES = (
    "correct",
    "cyclic_shift1",
    "cyclic_shift2",
    "reverse",
    "noise",
    "large_noise",
    "identity",
    "cross_sample",
)


def parse_modes(spec: str, *, allow_correct: bool = True) -> list[str]:
    modes = [part.strip().lower() for part in str(spec).split(",") if part.strip()]
    bad = [mode for mode in modes if mode not in POSE_CONSISTENCY_MODES]
    if bad:
        raise ValueError(f"Unknown pose consistency modes: {bad}; valid={POSE_CONSISTENCY_MODES}")
    if not allow_correct and "correct" in modes:
        raise ValueError("negative modes should not include correct")
    return modes


def parse_weights(spec: str, count: int) -> Optional[list[float]]:
    if not spec:
        return None
    weights = [float(part.strip()) for part in spec.split(",") if part.strip()]
    if len(weights) != count:
        raise ValueError(f"negative_weights count {len(weights)} != negative_modes count {count}")
    total = sum(weights)
    if total <= 0:
        raise ValueError("negative_weights should sum to a positive number")
    return [w / total for w in weights]


def make_cross_sample_batch(anchor: dict, other: dict) -> dict:
    out = dict(anchor)
    view_count = int(anchor["extrinsics"].shape[0])
    other_extrinsics = other["extrinsics"]
    if int(other_extrinsics.shape[0]) < view_count:
        repeats = int(np.ceil(view_count / max(int(other_extrinsics.shape[0]), 1)))
        other_extrinsics = other_extrinsics.repeat((repeats, 1, 1))
    out["extrinsics"] = other_extrinsics[:view_count].clone()
    out["pose_mode"] = "cross_sample"
    out["cross_sample_uid"] = other.get("uid")
    out["pose_permutation"] = None
    return out


def deterministic_other_index(index: int, total: int, seed: int) -> int:
    if total <= 1:
        return index
    other = (int(index) * 1103515245 + int(seed) * 12345 + 17) % total
    if other == index:
        other = (other + 1) % total
    return int(other)


def make_negative_batch(
    batch: dict,
    *,
    mode: str,
    seed: int,
    dataset: Dataset,
    anchor_index: Optional[int] = None,
) -> dict:
    mode = mode.lower()
    if mode == "cross_sample":
        if anchor_index is None:
            other_index = random.randrange(len(dataset))
        else:
            other_index = deterministic_other_index(anchor_index, len(dataset), seed)
        other = dataset[other_index]
        return make_cross_sample_batch(batch, other)
    return apply_pose_mode(batch, mode, seed)


def setup_condition_builder(args: argparse.Namespace, image_cond_model, head, device: torch.device) -> SparseMultiviewConditionBuilder:
    builder = SparseMultiviewConditionBuilder(device=device, low_vram=False)
    builder.image_cond_model_ss = image_cond_model
    builder.pose_consistency_head = head
    return builder


def score_batch(
    condition_builder: SparseMultiviewConditionBuilder,
    image_cond_model,
    batch: dict,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    make_multiview_condition(condition_builder, image_cond_model, batch, args, device)
    tensors = condition_builder.last_pose_consistency_tensors
    if tensors is None:
        raise RuntimeError("pose consistency tensors were not produced; check condition_builder.pose_consistency_head")
    stats = condition_builder.last_multiview_stats.get("ss_condition", {}).get("pose_consistency", {})
    return tensors["sample_score"], tensors["keep_ratio"], stats


def save_checkpoint(
    output_dir: Path,
    *,
    step: int,
    epoch: int,
    head: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    name: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": int(step),
        "epoch": int(epoch),
        "pose_consistency_head": head.state_dict(),
        "pose_consistency_head_config": head.config,
        "optimizer": optimizer.state_dict(),
        "args": vars(args),
    }
    torch.save(payload, output_dir / name)
    torch.save(payload, output_dir / "last.pt")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in keys})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a pose-sensitive view consistency gate before view aggregation.")
    parser.add_argument("--train_manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--resume", default="")
    parser.add_argument("--image_cond_model", default=IMAGE_COND_CONFIG["model_name"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_frames", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--sample_indices", default="")
    parser.add_argument("--max_epochs", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--negative_modes", default="cyclic_shift1,cyclic_shift2,cross_sample,reverse,noise,large_noise")
    parser.add_argument("--negative_weights", default="")
    parser.add_argument("--num_negatives", type=int, default=2)
    parser.add_argument("--ranking_margin", type=float, default=0.08)
    parser.add_argument("--correct_keep_target", type=float, default=0.65)
    parser.add_argument("--correct_keep_weight", type=float, default=0.05)
    parser.add_argument("--wrong_min_keep", type=float, default=0.02)
    parser.add_argument("--wrong_min_keep_weight", type=float, default=0.0)
    parser.add_argument("--head_reduced_dim", type=int, default=128)
    parser.add_argument("--head_hidden_dim", type=int, default=256)
    parser.add_argument("--head_dropout", type=float, default=0.0)
    parser.add_argument("--head_min_gate", type=float, default=0.05)
    parser.add_argument("--head_initial_logit", type=float, default=2.0)
    parser.add_argument(
        "--head_score_mode",
        choices=["single", "pairwise"],
        default="single",
        help="single scores each view independently; pairwise scores view pairs and reduces them to per-view logits.",
    )
    parser.add_argument(
        "--head_pair_weight_threshold",
        type=float,
        default=0.05,
        help="Visible-surface pair threshold for pairwise mode, applied to sqrt(support_i * support_j).",
    )
    parser.add_argument(
        "--head_pair_weight_mode",
        choices=["support", "front_depth", "front_depth_binary"],
        default="support",
        help=(
            "Pair weighting source for pairwise mode. support keeps old support_weights behavior; "
            "front_depth uses view_geom[...,0] front-depth visibility; front_depth_binary binarizes that channel."
        ),
    )
    parser.add_argument("--ss_grid_resolution", type=int, default=16)
    parser.add_argument("--camera_forward_sign", type=float, default=1.0)
    parser.add_argument("--mask_threshold", type=float, default=0.5)
    parser.add_argument("--no_apply_mask", action="store_true")
    parser.add_argument("--no_auto_volume", action="store_true")
    parser.add_argument("--vh_min_visible_views", type=int, default=1)
    parser.add_argument("--vh_min_support_views", type=int, default=2)
    parser.add_argument("--vh_min_support_ratio", type=float, default=0.6)
    parser.add_argument("--vh_volume_resolution", type=int, default=48)
    parser.add_argument("--vh_volume_initial_extent_ratio", type=float, default=0.6)
    parser.add_argument("--vh_volume_padding", type=float, default=1.25)
    parser.add_argument("--vh_volume_min_extent", type=float, default=0.05)
    parser.add_argument("--vh_volume_refine_steps", type=int, default=2)
    parser.add_argument("--no_visibility_depth", action="store_true")
    parser.add_argument("--vh_visibility_resolution", type=int, default=48)
    parser.add_argument("--vh_visibility_dilation", type=int, default=3)
    parser.add_argument("--visibility_depth_tolerance", type=float, default=0.0)
    parser.add_argument("--visibility_depth_tolerance_ratio", type=float, default=0.15)
    parser.add_argument("--visibility_weight_min", type=float, default=0.05)
    parser.add_argument("--empty_policy", choices=["zero", "visible", "border", "soft"], default="zero")
    parser.add_argument("--fallback_weight", type=float, default=1.0)
    parser.add_argument("--support_confidence_power", type=float, default=1.0)
    parser.add_argument("--global_fusion", choices=["concat", "mean", "first"], default="concat")
    parser.add_argument("--geometry_feature_mode", choices=["none", "add", "replace"], default="none")
    parser.add_argument("--geometry_feature_scale", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size != 1:
        raise ValueError("pose consistency training currently supports --batch_size 1 only")
    if args.num_negatives < 1:
        raise ValueError("--num_negatives should be >= 1")
    negative_modes = parse_modes(args.negative_modes, allow_correct=False)
    negative_weights = parse_weights(args.negative_weights, len(negative_modes))
    args.negative_modes_parsed = negative_modes
    args.negative_weights_parsed = negative_weights

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "args.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    base_dataset = MultiviewSparseManifestDataset(
        args.train_manifest,
        max_frames=args.max_frames,
        apply_mask=not args.no_apply_mask,
    )
    selected_indices = parse_sample_indices(args.sample_indices, len(base_dataset))
    dataset = Subset(base_dataset, selected_indices) if selected_indices else base_dataset
    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=args.num_workers, collate_fn=collate_single)

    image_cond_model = build_image_cond_model(device, args.ss_grid_resolution, args.image_cond_model)
    feature_dim = int(getattr(image_cond_model, "embed_dim", image_cond_model.model.config.hidden_size))
    if args.resume:
        head = load_pose_consistency_head(args.resume, feature_dim=feature_dim, device=device)
        state = torch.load(args.resume, map_location="cpu")
        global_step = int(state.get("step", 0))
        start_epoch = int(state.get("epoch", 0))
    else:
        head = build_pose_consistency_head(
            feature_dim=feature_dim,
            geom_dim=11,
            reduced_dim=args.head_reduced_dim,
            hidden_dim=args.head_hidden_dim,
            dropout=args.head_dropout,
            min_gate=args.head_min_gate,
            initial_logit=args.head_initial_logit,
            score_mode=args.head_score_mode,
            pair_weight_mode=args.head_pair_weight_mode,
            pair_weight_threshold=args.head_pair_weight_threshold,
            device=device,
        )
        global_step = 0
        start_epoch = 0
    head.train()
    condition_builder = setup_condition_builder(args, image_cond_model, head, device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))
    if args.resume and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])

    print(
        f"[train_pose_consistency_head] samples={len(dataset)} base_samples={len(base_dataset)} "
        f"output={output_dir} score_mode={getattr(head, 'score_mode', 'single')} "
        f"negatives={negative_modes} weights={negative_weights}"
    )
    planned_steps = min(args.max_steps, global_step + max(0, args.max_epochs - start_epoch) * len(loader))
    pbar = tqdm(total=planned_steps, initial=min(global_step, planned_steps), desc="Pose consistency", unit="step", dynamic_ncols=True)
    rows: list[dict] = []
    start_time = time.time()
    for epoch in range(start_epoch, args.max_epochs):
        for local_batch in loader:
            if global_step >= args.max_steps:
                break
            batch = local_batch
            if isinstance(dataset, Subset):
                anchor_index = None
            else:
                anchor_index = None

            optimizer.zero_grad(set_to_none=True)
            correct_score, correct_keep, correct_stats = score_batch(condition_builder, image_cond_model, batch, args, device)
            rank_terms = []
            wrong_scores = []
            wrong_keeps = []
            selected_modes = random.choices(negative_modes, weights=negative_weights, k=int(args.num_negatives))
            for neg_idx, mode in enumerate(selected_modes):
                seed = int(args.seed + global_step * 104729 + neg_idx * 9176)
                # Cross-sample uses a deterministic random object from the base dataset.
                if mode == "cross_sample":
                    other_index = random.randrange(len(base_dataset))
                    for _ in range(8):
                        if base_dataset[other_index]["uid"] != batch["uid"]:
                            break
                        other_index = random.randrange(len(base_dataset))
                    wrong_batch = make_cross_sample_batch(batch, base_dataset[other_index])
                else:
                    wrong_batch = make_negative_batch(batch, mode=mode, seed=seed, dataset=base_dataset, anchor_index=anchor_index)
                wrong_score, wrong_keep, _ = score_batch(condition_builder, image_cond_model, wrong_batch, args, device)
                rank_terms.append(F.relu(float(args.ranking_margin) - correct_score + wrong_score))
                wrong_scores.append(wrong_score)
                wrong_keeps.append(wrong_keep)

            ranking_loss = torch.stack(rank_terms).mean()
            correct_keep_loss = F.relu(torch.as_tensor(args.correct_keep_target, device=device) - correct_keep).pow(2)
            wrong_keep_loss = torch.stack(
                [F.relu(torch.as_tensor(args.wrong_min_keep, device=device) - keep).pow(2) for keep in wrong_keeps]
            ).mean()
            loss = (
                ranking_loss
                + float(args.correct_keep_weight) * correct_keep_loss
                + float(args.wrong_min_keep_weight) * wrong_keep_loss
            )
            loss.backward()
            optimizer.step()

            global_step += 1
            row = {
                "step": global_step,
                "epoch": epoch,
                "loss": float(loss.detach().cpu().item()),
                "ranking_loss": float(ranking_loss.detach().cpu().item()),
                "correct_score": float(correct_score.detach().cpu().item()),
                "correct_keep": float(correct_keep.detach().cpu().item()),
                "wrong_score_mean": float(torch.stack(wrong_scores).mean().detach().cpu().item()),
                "wrong_keep_mean": float(torch.stack(wrong_keeps).mean().detach().cpu().item()),
                "wrong_modes": ",".join(selected_modes),
                "uid": batch["uid"],
                "correct_gate_mean": correct_stats.get("gate_mean"),
                "score_mode": correct_stats.get("score_mode"),
                "pair_weight_mode": correct_stats.get("pair_weight_mode"),
                "pair_valid_ratio": correct_stats.get("pair_valid_ratio"),
                "pair_count_mean": correct_stats.get("pair_count_mean"),
                "pair_weight_threshold": correct_stats.get("pair_weight_threshold"),
                "pair_weight_mean": correct_stats.get("pair_weight_mean"),
                "view_weight_mean": correct_stats.get("view_weight_mean"),
                "view_weight_nonzero_ratio": correct_stats.get("view_weight_nonzero_ratio"),
                "pair_supported_voxel_ratio": correct_stats.get("pair_supported_voxel_ratio"),
                "pair_sample_score": correct_stats.get("pair_sample_score"),
                "pair_keep_ratio": correct_stats.get("pair_keep_ratio"),
                "pair_logit_mean": correct_stats.get("pair_logit_mean"),
                "view_prior_abs_mean": correct_stats.get("view_prior_abs_mean"),
                "elapsed_sec": float(time.time() - start_time),
            }
            rows.append(row)
            if global_step % args.log_every == 0 or global_step == 1:
                pbar.set_postfix(
                    {
                        "loss": f"{row['loss']:.4g}",
                        "pos": f"{row['correct_score']:.3f}",
                        "neg": f"{row['wrong_score_mean']:.3f}",
                        "keep": f"{row['correct_keep']:.3f}",
                        "modes": row["wrong_modes"],
                    },
                    refresh=False,
                )
            pbar.update(1)
            if global_step % args.save_every == 0:
                save_checkpoint(output_dir, step=global_step, epoch=epoch, head=head, optimizer=optimizer, args=args, name=f"step_{global_step}.pt")

        if global_step >= args.max_steps:
            break

    pbar.close()
    save_checkpoint(output_dir, step=global_step, epoch=args.max_epochs, head=head, optimizer=optimizer, args=args, name="final.pt")
    write_csv(output_dir / "train_metrics.csv", rows)
    print(f"[train_pose_consistency_head] done step={global_step} final={output_dir / 'final.pt'}")


if __name__ == "__main__":
    main()
