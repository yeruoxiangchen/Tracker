from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from tqdm.auto import tqdm

TRACKER_ROOT = Path(__file__).resolve().parents[1]
PIXAL3D_ROOT = TRACKER_ROOT / "Pixal3D"
sys.path.insert(0, str(TRACKER_ROOT))
sys.path.insert(0, str(PIXAL3D_ROOT))

from pixal3d_multiview.diagnose_pose_condition import (  # noqa: E402
    _feature_agreement,
    _voxel_ray_diversity,
)
from pixal3d_multiview.eval_condition_view_consistency import (  # noqa: E402
    extract_patch_features,
    load_target_mask,
    parse_indices,
)
from pixal3d_multiview.multiview_projection import (  # noqa: E402
    compute_visual_hull_score_points,
    estimate_object_volume_from_visual_hull,
    pixal3d_grid_points,
    project_points_multi_view,
    resize_masks,
    scale_intrinsics_to_square,
    visual_hull_front_depth_maps,
)
from pixal3d_multiview.pose_consistency_head import load_pose_consistency_head  # noqa: E402
from pixal3d_multiview.train_sparse_multiview import (  # noqa: E402
    IMAGE_COND_CONFIG,
    MultiviewSparseManifestDataset,
    build_image_cond_model,
)
from pixal3d_multiview.view_aggregator import sample_view_features_for_aggregation  # noqa: E402


FEATURE_NAMES = [
    "support_count",
    "support_fraction",
    "support_weight_sum",
    "support_weight_mean",
    "agreement_all_mean_cos",
    "agreement_loo_cos",
    "agreement_pair_cos",
    "agreement_mean_norm",
    "voxel_ray_angle_deg",
    "voxel_camera_baseline",
    "visual_hull_score",
    "visual_hull_support_fraction",
    "visual_hull_visible_fraction",
    "head_logit_mean",
    "head_gate_mean",
    "head_voxel_score",
    "old_attention_entropy",
    "old_attention_max",
]


@dataclass
class TrainRow:
    step: int
    epoch: int
    loss: float
    pos_loss: float
    neg_loss: float
    lr: float


class TargetAwareDiagnosticHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.0):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)
        self.net = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, 1),
        )

    @property
    def config(self) -> dict:
        return {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "dropout": self.dropout,
            "feature_names": FEATURE_NAMES,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _safe_div(num: torch.Tensor, den: torch.Tensor) -> torch.Tensor:
    return num.float() / den.float().clamp_min(1e-6)


def _old_attention_stats(weights: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    valid = weights > 0
    attn = weights.float() / weights.sum(dim=0, keepdim=True).clamp_min(1e-6)
    entropy = -(attn.clamp_min(1e-8) * attn.clamp_min(1e-8).log()).sum(dim=0)
    entropy = torch.where(valid.any(dim=0), entropy, torch.zeros_like(entropy))
    max_attn = attn.max(dim=0).values
    return entropy, max_attn


def _per_voxel_feature_agreement(sampled: torch.Tensor, weights: torch.Tensor) -> dict[str, torch.Tensor]:
    features = F.normalize(sampled.float(), dim=-1, eps=1e-6)
    weights = weights.float().clamp_min(0.0)
    valid = weights > 0
    support_count = valid.sum(dim=0).float()
    weight_sum = weights.sum(dim=0).clamp_min(1e-6)
    weighted_sum = (features * weights[..., None]).sum(dim=0)
    mean_raw = weighted_sum / weight_sum[:, None]
    mean_norm = torch.linalg.norm(mean_raw, dim=-1)
    mean_feature = F.normalize(mean_raw, dim=-1, eps=1e-6)
    all_mean_cos = (features * mean_feature[None]).sum(dim=-1).clamp(-1.0, 1.0)
    all_mean_cos = (all_mean_cos * weights).sum(dim=0) / weight_sum

    other_weight = weight_sum[None] - weights
    other_sum = weighted_sum[None] - features * weights[..., None]
    loo_mean = F.normalize(other_sum / other_weight[..., None].clamp_min(1e-6), dim=-1, eps=1e-6)
    loo_cos_view = (features * loo_mean).sum(dim=-1).clamp(-1.0, 1.0)
    loo_valid = valid & (other_weight > 1e-6)
    loo_weight = torch.where(loo_valid, weights, torch.zeros_like(weights))
    loo_cos = (loo_cos_view * loo_weight).sum(dim=0) / loo_weight.sum(dim=0).clamp_min(1e-6)
    loo_cos = torch.where(loo_weight.sum(dim=0) > 0, loo_cos, torch.zeros_like(loo_cos))

    pair_cos = torch.zeros_like(weight_sum)
    if features.shape[0] > 1:
        gram = torch.einsum("vnc,wnc->vwn", features, features)
        pair_weights = weights[:, None, :] * weights[None, :, :]
        eye = torch.eye(features.shape[0], device=features.device, dtype=torch.bool)[:, :, None]
        pair_weights = torch.where(eye, torch.zeros_like(pair_weights), pair_weights)
        pair_denom = pair_weights.sum(dim=(0, 1))
        pair_cos = (gram * pair_weights).sum(dim=(0, 1)) / pair_denom.clamp_min(1e-6)
        pair_cos = torch.where(pair_denom > 0, pair_cos, torch.zeros_like(pair_cos))

    return {
        "support_count": support_count,
        "support_weight_sum": weights.sum(dim=0),
        "agreement_all_mean_cos": all_mean_cos,
        "agreement_loo_cos": loo_cos,
        "agreement_pair_cos": pair_cos,
        "agreement_mean_norm": mean_norm,
    }


def _per_voxel_ray_stats(
    points_obj: torch.Tensor,
    object_to_world: torch.Tensor,
    extrinsics: torch.Tensor,
    extrinsics_are_c2w: bool,
    support_weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    c2w = extrinsics.float() if extrinsics_are_c2w else torch.linalg.inv(extrinsics.float())
    centers = c2w[:, :3, 3].float()
    n = points_obj.shape[0]
    if centers.shape[0] <= 1:
        return torch.zeros((n,), device=points_obj.device), torch.zeros((n,), device=points_obj.device)
    points_h = torch.cat([points_obj.float(), torch.ones((n, 1), device=points_obj.device)], dim=1)
    points_world = (points_h @ object_to_world.to(points_obj.device, torch.float32).T)[:, :3]
    dirs = F.normalize(points_world[None] - centers[:, None], dim=-1, eps=1e-6)
    cos = torch.einsum("vnc,wnc->vwn", dirs, dirs).clamp(-1.0, 1.0)
    angles = torch.rad2deg(torch.acos(cos))
    weights = support_weights.float().clamp_min(0.0)
    pair_weights = weights[:, None, :] * weights[None, :, :]
    eye = torch.eye(centers.shape[0], device=points_obj.device, dtype=torch.bool)[:, :, None]
    pair_weights = torch.where(eye, torch.zeros_like(pair_weights), pair_weights)
    pair_denom = pair_weights.sum(dim=(0, 1))
    angle = (angles * pair_weights).sum(dim=(0, 1)) / pair_denom.clamp_min(1e-6)
    baseline = torch.cdist(centers, centers)
    baseline_v = (baseline[:, :, None] * pair_weights).sum(dim=(0, 1)) / pair_denom.clamp_min(1e-6)
    angle = torch.where(pair_denom > 0, angle, torch.zeros_like(angle))
    baseline_v = torch.where(pair_denom > 0, baseline_v, torch.zeros_like(baseline_v))
    return angle, baseline_v


def _target_mask_for_resolution(latent_path: str, resolution: int, device: torch.device) -> torch.Tensor:
    mask, _, _ = load_target_mask(latent_path, resolution, device)
    return mask.bool()


@torch.no_grad()
def extract_sample_features(
    dataset: MultiviewSparseManifestDataset,
    image_cond_model,
    pose_head,
    sample_index: int,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    batch = dataset[sample_index]
    patch_features, _ = extract_patch_features(image_cond_model, batch["images"], device)
    target_mask = _target_mask_for_resolution(batch["latent_path"], args.ss_grid_resolution, device)
    masks = batch["masks"].to(device=device, dtype=torch.float32)
    intrinsics = batch["intrinsics"].to(device=device, dtype=torch.float32)
    extrinsics = batch["extrinsics"].to(device=device, dtype=torch.float32)
    extrinsics_are_c2w = str(batch["extrinsics_type"]).lower() == "c2w"

    volume = estimate_object_volume_from_visual_hull(
        masks,
        intrinsics,
        extrinsics,
        extrinsics_are_c2w=extrinsics_are_c2w,
        camera_forward_sign=args.camera_forward_sign,
        mask_threshold=args.mask_threshold,
        resolution=args.vh_volume_resolution,
        min_visible_views=args.vh_min_visible_views,
        min_support_views=args.vh_min_support_views,
        min_support_ratio=args.vh_min_support_ratio,
        initial_extent_ratio=args.vh_volume_initial_extent_ratio,
        padding=args.vh_volume_padding,
        min_extent=args.vh_volume_min_extent,
        refine_steps=args.vh_volume_refine_steps,
    )
    object_to_world = volume.object_to_world
    volume_extent = float(volume.extent_world)
    image_size = int(image_cond_model.image_size)
    intrinsics_sq = scale_intrinsics_to_square(intrinsics, batch["source_sizes"], image_size, device)
    masks_sq = resize_masks(masks, image_size, device)
    points_obj, _ = pixal3d_grid_points(args.ss_grid_resolution, device=device, dtype=torch.float32)
    points_2d, depths, valid_depth = project_points_multi_view(
        points_obj,
        intrinsics_sq,
        extrinsics,
        extrinsics_are_c2w=extrinsics_are_c2w,
        camera_forward_sign=args.camera_forward_sign,
        object_to_world=object_to_world,
    )
    if args.visibility_depth_tolerance > 0:
        depth_tolerance = float(args.visibility_depth_tolerance)
    else:
        depth_tolerance = max(volume_extent * float(args.visibility_depth_tolerance_ratio), 1e-4)

    front_depth_maps, _ = visual_hull_front_depth_maps(
        masks_sq,
        intrinsics_sq,
        extrinsics,
        extrinsics_are_c2w=extrinsics_are_c2w,
        camera_forward_sign=args.camera_forward_sign,
        object_to_world=object_to_world,
        resolution=args.vh_visibility_resolution,
        coordinate_size=image_size,
        mask_threshold=args.mask_threshold,
        min_visible_views=args.vh_min_visible_views,
        min_support_views=args.vh_min_support_views,
        min_support_ratio=args.vh_min_support_ratio,
        dilation_radius=args.vh_visibility_dilation,
    )
    sampled, support_weights, view_geom, _ = sample_view_features_for_aggregation(
        patch_features,
        points_obj,
        points_2d,
        depths,
        valid_depth,
        coordinate_size=image_size,
        masks=masks_sq,
        mask_threshold=args.mask_threshold,
        front_depth_maps=front_depth_maps,
        visibility_depth_tolerance=depth_tolerance,
        visibility_weight_min=args.visibility_weight_min,
    )

    per_voxel = _per_voxel_feature_agreement(sampled, support_weights)
    old_entropy, old_max = _old_attention_stats(support_weights)
    ray_angle, baseline = _per_voxel_ray_stats(points_obj, object_to_world, extrinsics, extrinsics_are_c2w, support_weights)

    points_h = torch.cat([points_obj, torch.ones((points_obj.shape[0], 1), device=device)], dim=1)
    points_world = (points_h @ object_to_world.T)[:, :3]
    vh_score, vh_support, vh_visible = compute_visual_hull_score_points(
        points_world,
        masks,
        intrinsics,
        extrinsics,
        extrinsics_are_c2w=extrinsics_are_c2w,
        camera_forward_sign=args.camera_forward_sign,
        mask_threshold=args.mask_threshold,
        min_visible_views=args.vh_min_visible_views,
    )

    head_logit_mean = torch.zeros_like(vh_score)
    head_gate_mean = torch.zeros_like(vh_score)
    head_voxel_score = torch.zeros_like(vh_score)
    if pose_head is not None:
        _, _, head_tensors = pose_head(sampled.detach(), support_weights.detach(), view_geom.detach())
        logits = head_tensors["logits"].float()
        gate = head_tensors["gate"].float()
        weight_sum = support_weights.sum(dim=0).clamp_min(1e-6)
        head_logit_mean = (logits * support_weights).sum(dim=0) / weight_sum
        head_gate_mean = (gate * support_weights).sum(dim=0) / weight_sum
        head_logit_mean = torch.where(support_weights.sum(dim=0) > 0, head_logit_mean, torch.zeros_like(head_logit_mean))
        head_gate_mean = torch.where(support_weights.sum(dim=0) > 0, head_gate_mean, torch.zeros_like(head_gate_mean))
        head_voxel_score = head_tensors["voxel_score"].float()

    support_count = per_voxel["support_count"]
    support_weight_sum = per_voxel["support_weight_sum"]
    feature_map = {
        "support_count": support_count,
        "support_fraction": support_count / float(max(int(support_weights.shape[0]), 1)),
        "support_weight_sum": support_weight_sum,
        "support_weight_mean": support_weight_sum / float(max(int(support_weights.shape[0]), 1)),
        "agreement_all_mean_cos": per_voxel["agreement_all_mean_cos"],
        "agreement_loo_cos": per_voxel["agreement_loo_cos"],
        "agreement_pair_cos": per_voxel["agreement_pair_cos"],
        "agreement_mean_norm": per_voxel["agreement_mean_norm"],
        "voxel_ray_angle_deg": ray_angle,
        "voxel_camera_baseline": baseline,
        "visual_hull_score": vh_score.float(),
        "visual_hull_support_fraction": vh_support.float() / float(max(int(support_weights.shape[0]), 1)),
        "visual_hull_visible_fraction": vh_visible.float() / float(max(int(support_weights.shape[0]), 1)),
        "head_logit_mean": head_logit_mean,
        "head_gate_mean": head_gate_mean,
        "head_voxel_score": head_voxel_score,
        "old_attention_entropy": old_entropy,
        "old_attention_max": old_max,
    }
    features = torch.stack([feature_map[name].float() for name in FEATURE_NAMES], dim=1)
    features = torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    supported = support_weights.sum(dim=0) > 0
    labels = target_mask.float()
    meta = {
        "uid": batch["uid"],
        "target_count": int(target_mask.sum().item()),
        "supported_count": int(supported.sum().item()),
        "positive_supported_count": int((supported & target_mask).sum().item()),
        "volume_fallback": bool(volume.fallback),
        "volume_extent_world": float(volume.extent_world),
    }
    return features[supported], labels[supported], meta


def balanced_sample(
    features: torch.Tensor,
    labels: torch.Tensor,
    max_pos: int,
    neg_per_pos: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    pos_idx = torch.where(labels > 0.5)[0]
    neg_idx = torch.where(labels <= 0.5)[0]
    if pos_idx.numel() == 0 or neg_idx.numel() == 0:
        return features[:0], labels[:0]
    if pos_idx.numel() > int(max_pos):
        perm = torch.randperm(pos_idx.numel(), generator=generator, device=pos_idx.device)[: int(max_pos)]
        pos_idx = pos_idx[perm]
    neg_count = min(int(neg_idx.numel()), int(pos_idx.numel()) * int(neg_per_pos))
    if neg_idx.numel() > neg_count:
        perm = torch.randperm(neg_idx.numel(), generator=generator, device=neg_idx.device)[:neg_count]
        neg_idx = neg_idx[perm]
    idx = torch.cat([pos_idx, neg_idx], dim=0)
    perm = torch.randperm(idx.numel(), generator=generator, device=idx.device)
    idx = idx[perm]
    return features[idx], labels[idx]


def auc_score(labels: torch.Tensor, scores: torch.Tensor) -> Optional[float]:
    labels = labels.detach().cpu().float()
    scores = scores.detach().cpu().float()
    pos = labels > 0.5
    neg = ~pos
    n_pos = int(pos.sum().item())
    n_neg = int(neg.sum().item())
    if n_pos == 0 or n_neg == 0:
        return None
    order = torch.argsort(scores)
    ranks = torch.empty_like(scores)
    ranks[order] = torch.arange(1, scores.numel() + 1, dtype=torch.float32)
    pos_rank_sum = ranks[pos].sum()
    auc = (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / float(n_pos * n_neg)
    return float(auc.item())


def average_precision(labels: torch.Tensor, scores: torch.Tensor) -> Optional[float]:
    labels = labels.detach().cpu().float()
    scores = scores.detach().cpu().float()
    pos_count = int((labels > 0.5).sum().item())
    if pos_count == 0:
        return None
    order = torch.argsort(scores, descending=True)
    sorted_labels = labels[order]
    tp = torch.cumsum(sorted_labels, dim=0)
    rank = torch.arange(1, sorted_labels.numel() + 1, dtype=torch.float32)
    precision = tp / rank
    ap = (precision * sorted_labels).sum() / float(pos_count)
    return float(ap.item())


@torch.no_grad()
def evaluate_model(model: nn.Module, xs: list[torch.Tensor], ys: list[torch.Tensor], device: torch.device) -> dict:
    if not xs:
        return {"count": 0}
    x = torch.cat(xs, dim=0).to(device)
    y = torch.cat(ys, dim=0).to(device)
    logits = model(x)
    loss = F.binary_cross_entropy_with_logits(logits, y)
    scores = torch.sigmoid(logits)
    pred = scores >= 0.5
    pos = y > 0.5
    neg = ~pos
    tp = (pred & pos).sum().float()
    fp = (pred & neg).sum().float()
    fn = ((~pred) & pos).sum().float()
    precision = tp / (tp + fp).clamp_min(1.0)
    recall = tp / (tp + fn).clamp_min(1.0)
    return {
        "count": int(y.numel()),
        "pos": int(pos.sum().item()),
        "neg": int(neg.sum().item()),
        "loss": float(loss.cpu().item()),
        "auc": auc_score(y, scores),
        "ap": average_precision(y, scores),
        "precision@0.5": float(precision.cpu().item()),
        "recall@0.5": float(recall.cpu().item()),
        "target_score_mean": float(scores[pos].mean().cpu().item()) if pos.any() else None,
        "non_target_score_mean": float(scores[neg].mean().cpu().item()) if neg.any() else None,
        "score_gap": float((scores[pos].mean() - scores[neg].mean()).cpu().item()) if pos.any() and neg.any() else None,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys = sorted({k for row in rows for k in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in keys})


def write_report(path: Path, result: dict) -> None:
    lines = [
        "# Target-aware Diagnostic Head",
        "",
        f"时间：`{result['timestamp_utc']}`",
        f"train manifest：`{result['train_manifest']}`",
        f"val manifest：`{result['val_manifest']}`",
        f"head checkpoint：`{result.get('pose_head_checkpoint') or ''}`",
        "",
        "## 指标",
        "",
        "| split | loss | AUC | AP | P@0.5 | R@0.5 | target score | non-target score | gap |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split in ("train", "val"):
        row = result[f"{split}_metrics"]
        lines.append(
            "| {split} | {loss:.6f} | {auc} | {ap} | {precision:.4f} | {recall:.4f} | {target} | {non_target} | {gap} |".format(
                split=split,
                loss=float(row.get("loss", 0.0)),
                auc="" if row.get("auc") is None else f"{row['auc']:.4f}",
                ap="" if row.get("ap") is None else f"{row['ap']:.4f}",
                precision=float(row.get("precision@0.5", 0.0)),
                recall=float(row.get("recall@0.5", 0.0)),
                target="" if row.get("target_score_mean") is None else f"{row['target_score_mean']:.4f}",
                non_target="" if row.get("non_target_score_mean") is None else f"{row['non_target_score_mean']:.4f}",
                gap="" if row.get("score_gap") is None else f"{row['score_gap']:.4f}",
            )
        )
    lines.extend(
        [
            "",
            "## 如何阅读",
            "",
            "- 这个 head 不接 sparse flow，只判断当前 condition statistics 是否能区分 target / non-target voxel。",
            "- 如果 val AUC/AP 明显高于随机，并且 target score > non-target score，说明这些统计量可用于后续 geometry-aware gate loss。",
            "- 如果 val AUC 接近 0.5 或 score gap 很小/为负，说明当前 condition statistics 本身还不足以指导 target-aware gating。",
            "",
            "## Feature Names",
            "",
            "```text",
            "\n".join(result["feature_names"]),
            "```",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a small target-aware diagnostic voxel classifier without sparse-flow training.")
    parser.add_argument("--train_manifest", required=True)
    parser.add_argument("--val_manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--image_cond_model", default=IMAGE_COND_CONFIG["model_name"])
    parser.add_argument("--pose_head_checkpoint", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_frames", type=int, default=8)
    parser.add_argument("--train_indices", default="0-127")
    parser.add_argument("--val_indices", default="0-127")
    parser.add_argument("--ss_grid_resolution", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--max_pos_per_sample", type=int, default=512)
    parser.add_argument("--neg_per_pos", type=int, default=3)
    parser.add_argument("--camera_forward_sign", type=float, default=1.0)
    parser.add_argument("--mask_threshold", type=float, default=0.5)
    parser.add_argument("--vh_min_visible_views", type=int, default=1)
    parser.add_argument("--vh_min_support_views", type=int, default=2)
    parser.add_argument("--vh_min_support_ratio", type=float, default=0.6)
    parser.add_argument("--vh_volume_resolution", type=int, default=48)
    parser.add_argument("--vh_volume_initial_extent_ratio", type=float, default=0.6)
    parser.add_argument("--vh_volume_padding", type=float, default=1.25)
    parser.add_argument("--vh_volume_min_extent", type=float, default=0.05)
    parser.add_argument("--vh_volume_refine_steps", type=int, default=2)
    parser.add_argument("--vh_visibility_resolution", type=int, default=48)
    parser.add_argument("--vh_visibility_dilation", type=int, default=3)
    parser.add_argument("--visibility_depth_tolerance", type=float, default=0.0)
    parser.add_argument("--visibility_depth_tolerance_ratio", type=float, default=0.15)
    parser.add_argument("--visibility_weight_min", type=float, default=0.05)
    return parser.parse_args()


def load_feature_dataset(
    dataset: MultiviewSparseManifestDataset,
    indices: list[int],
    image_cond_model,
    pose_head,
    args: argparse.Namespace,
    device: torch.device,
    desc: str,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[dict]]:
    generator = torch.Generator(device=device)
    generator.manual_seed(int(args.seed))
    xs: list[torch.Tensor] = []
    ys: list[torch.Tensor] = []
    metas: list[dict] = []
    for idx in tqdm(indices, desc=desc, unit="sample", dynamic_ncols=True):
        feats, labels, meta = extract_sample_features(dataset, image_cond_model, pose_head, idx, args, device)
        feats, labels = balanced_sample(feats, labels, args.max_pos_per_sample, args.neg_per_pos, generator)
        if feats.numel() == 0:
            meta["used"] = False
            metas.append(meta)
            continue
        xs.append(feats.detach().cpu())
        ys.append(labels.detach().cpu())
        meta.update({"used": True, "train_voxels": int(labels.numel()), "train_pos": int((labels > 0.5).sum().item())})
        metas.append(meta)
    return xs, ys, metas


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "args.json").write_text(json.dumps(vars(args), indent=2, ensure_ascii=False), encoding="utf-8")

    train_dataset = MultiviewSparseManifestDataset(args.train_manifest, max_frames=args.max_frames, apply_mask=True)
    val_dataset = MultiviewSparseManifestDataset(args.val_manifest, max_frames=args.max_frames, apply_mask=True)
    train_indices = parse_indices(args.train_indices, len(train_dataset))
    val_indices = parse_indices(args.val_indices, len(val_dataset))
    image_cond_model = build_image_cond_model(device, args.ss_grid_resolution, args.image_cond_model)
    feature_dim = int(getattr(image_cond_model, "embed_dim", image_cond_model.model.config.hidden_size))
    pose_head = None
    if args.pose_head_checkpoint:
        pose_head = load_pose_consistency_head(args.pose_head_checkpoint, feature_dim=feature_dim, device=device).eval()

    train_xs, train_ys, train_meta = load_feature_dataset(train_dataset, train_indices, image_cond_model, pose_head, args, device, "Extract train")
    val_xs, val_ys, val_meta = load_feature_dataset(val_dataset, val_indices, image_cond_model, pose_head, args, device, "Extract val")
    if not train_xs:
        raise RuntimeError("No usable training voxels were extracted.")

    train_x = torch.cat(train_xs, dim=0)
    train_y = torch.cat(train_ys, dim=0)
    mean = train_x.mean(dim=0)
    std = train_x.std(dim=0, unbiased=False).clamp_min(1e-6)
    train_x = (train_x - mean) / std
    val_xs_norm = [(x - mean) / std for x in val_xs]

    model = TargetAwareDiagnosticHead(len(FEATURE_NAMES), hidden_dim=args.hidden_dim, dropout=args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    rows = []
    train_x_dev = train_x.to(device)
    train_y_dev = train_y.to(device)
    for epoch in tqdm(range(args.epochs), desc="Train diagnostic head", dynamic_ncols=True):
        model.train()
        perm = torch.randperm(train_x_dev.shape[0], device=device)
        total_loss = 0.0
        total_pos_loss = 0.0
        total_neg_loss = 0.0
        chunks = 0
        for start in range(0, perm.numel(), 8192):
            ids = perm[start : start + 8192]
            x = train_x_dev[ids]
            y = train_y_dev[ids]
            logits = model(x)
            loss = F.binary_cross_entropy_with_logits(logits, y)
            pos_loss = F.binary_cross_entropy_with_logits(logits[y > 0.5], y[y > 0.5]) if (y > 0.5).any() else loss * 0
            neg_loss = F.binary_cross_entropy_with_logits(logits[y <= 0.5], y[y <= 0.5]) if (y <= 0.5).any() else loss * 0
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu().item())
            total_pos_loss += float(pos_loss.detach().cpu().item())
            total_neg_loss += float(neg_loss.detach().cpu().item())
            chunks += 1
        rows.append(
            asdict(
                TrainRow(
                    step=epoch + 1,
                    epoch=epoch,
                    loss=total_loss / max(chunks, 1),
                    pos_loss=total_pos_loss / max(chunks, 1),
                    neg_loss=total_neg_loss / max(chunks, 1),
                    lr=float(optimizer.param_groups[0]["lr"]),
                )
            )
        )

    model.eval()
    train_metrics = evaluate_model(model, [train_x], [train_y], device)
    val_metrics = evaluate_model(model, val_xs_norm, val_ys, device)
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "train_manifest": args.train_manifest,
        "val_manifest": args.val_manifest,
        "train_indices": train_indices,
        "val_indices": val_indices,
        "pose_head_checkpoint": args.pose_head_checkpoint,
        "feature_names": FEATURE_NAMES,
        "feature_mean": mean.tolist(),
        "feature_std": std.tolist(),
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "train_meta": train_meta,
        "val_meta": val_meta,
    }
    torch.save(
        {
            "model": model.state_dict(),
            "config": model.config,
            "feature_mean": mean,
            "feature_std": std,
            "args": vars(args),
            "metrics": {"train": train_metrics, "val": val_metrics},
        },
        output_dir / "target_aware_diagnostic_head.pt",
    )
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(output_dir / "train_log.csv", rows)
    write_csv(output_dir / "train_meta.csv", train_meta)
    write_csv(output_dir / "val_meta.csv", val_meta)
    write_report(output_dir / "report.md", payload)
    print(json.dumps({"output_dir": str(output_dir), "train": train_metrics, "val": val_metrics}, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
