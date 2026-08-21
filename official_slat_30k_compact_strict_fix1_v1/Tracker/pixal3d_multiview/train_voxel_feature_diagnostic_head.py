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

from pixal3d_multiview.eval_condition_view_consistency import (  # noqa: E402
    extract_patch_features,
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
from pixal3d_multiview.train_target_aware_diagnostic_head import (  # noqa: E402
    FEATURE_NAMES as STAT_FEATURE_NAMES,
    _old_attention_stats,
    _per_voxel_feature_agreement,
    _per_voxel_ray_stats,
    _target_mask_for_resolution,
    auc_score,
    average_precision,
)
from pixal3d_multiview.view_aggregator import sample_view_features_for_aggregation  # noqa: E402


@dataclass
class TrainRow:
    epoch: int
    step: int
    loss: float
    pos_loss: float
    neg_loss: float
    lr: float


class VoxelFeatureDiagnosticHead(nn.Module):
    """Target/non-target diagnostic over raw per-view back-projected features.

    This is intentionally a diagnostic classifier, not a sparse-flow module. It
    answers whether per-voxel multi-view DINO features plus projection geometry
    carry enough information to separate target sparse coords from supported
    non-target coords.
    """

    def __init__(
        self,
        feature_dim: int,
        geom_dim: int,
        stat_dim: int,
        *,
        reduced_dim: int = 128,
        hidden_dim: int = 256,
        dropout: float = 0.0,
        use_stats: bool = True,
        feature_ablation: str = "full",
    ):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.geom_dim = int(geom_dim)
        self.stat_dim = int(stat_dim)
        self.reduced_dim = int(reduced_dim)
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)
        self.use_stats = bool(use_stats)
        self.feature_ablation = str(feature_ablation)

        self.feature_reduce = nn.Linear(self.feature_dim, self.reduced_dim)
        self.view_encoder = nn.Sequential(
            nn.LayerNorm(self.reduced_dim + self.geom_dim),
            nn.Linear(self.reduced_dim + self.geom_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
        )
        self.view_gate = nn.Linear(self.hidden_dim, 1)
        classifier_in = self.hidden_dim + (self.stat_dim if self.use_stats else 0)
        self.classifier = nn.Sequential(
            nn.LayerNorm(classifier_in),
            nn.Linear(classifier_in, self.hidden_dim),
            nn.SiLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, 1),
        )

    @property
    def config(self) -> dict:
        return {
            "feature_dim": self.feature_dim,
            "geom_dim": self.geom_dim,
            "stat_dim": self.stat_dim,
            "reduced_dim": self.reduced_dim,
            "hidden_dim": self.hidden_dim,
            "dropout": self.dropout,
            "use_stats": self.use_stats,
            "feature_ablation": self.feature_ablation,
            "stat_feature_names": STAT_FEATURE_NAMES,
        }

    def _apply_feature_ablation(
        self,
        sampled: torch.Tensor,
        support: torch.Tensor,
        geom: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mode = self.feature_ablation
        valid = support.float() > 0
        sampled_out = sampled.float()
        geom_out = geom.float()
        support_prior = support.float()

        if mode == "full":
            return sampled_out, support_prior, geom_out

        if mode == "feature_only":
            return sampled_out, valid.float(), torch.zeros_like(geom_out)

        if mode == "geometry_only":
            sampled_out = torch.zeros_like(sampled_out)
            geom_filtered = torch.zeros_like(geom_out)
            geom_filtered[..., 5:] = geom_out[..., 5:]
            return sampled_out, valid.float(), geom_filtered

        if mode == "uv_depth_only":
            sampled_out = torch.zeros_like(sampled_out)
            geom_filtered = torch.zeros_like(geom_out)
            geom_filtered[..., 5:8] = geom_out[..., 5:8]
            return sampled_out, valid.float(), geom_filtered

        if mode == "xyz_only":
            sampled_out = torch.zeros_like(sampled_out)
            geom_filtered = torch.zeros_like(geom_out)
            geom_filtered[..., 8:11] = geom_out[..., 8:11]
            return sampled_out, valid.float(), geom_filtered

        if mode == "uv_depth_xyz":
            sampled_out = torch.zeros_like(sampled_out)
            geom_filtered = torch.zeros_like(geom_out)
            geom_filtered[..., 5:11] = geom_out[..., 5:11]
            return sampled_out, valid.float(), geom_filtered

        if mode == "support_only":
            sampled_out = torch.zeros_like(sampled_out)
            geom_filtered = torch.zeros_like(geom_out)
            geom_filtered[..., :5] = geom_out[..., :5]
            return sampled_out, support_prior, geom_filtered

        if mode == "feature_geometry":
            geom_filtered = torch.zeros_like(geom_out)
            geom_filtered[..., 5:] = geom_out[..., 5:]
            return sampled_out, valid.float(), geom_filtered

        if mode == "feature_uv_depth":
            geom_filtered = torch.zeros_like(geom_out)
            geom_filtered[..., 5:8] = geom_out[..., 5:8]
            return sampled_out, valid.float(), geom_filtered

        if mode == "feature_xyz":
            geom_filtered = torch.zeros_like(geom_out)
            geom_filtered[..., 8:11] = geom_out[..., 8:11]
            return sampled_out, valid.float(), geom_filtered

        if mode == "feature_support":
            geom_filtered = torch.zeros_like(geom_out)
            geom_filtered[..., :5] = geom_out[..., :5]
            return sampled_out, support_prior, geom_filtered

        if mode == "geometry_support":
            return torch.zeros_like(sampled_out), support_prior, geom_out

        raise ValueError(f"Unknown feature_ablation mode: {mode}")

    def forward(self, sampled: torch.Tensor, support: torch.Tensor, geom: torch.Tensor, stats: torch.Tensor) -> torch.Tensor:
        if sampled.ndim != 3:
            raise ValueError(f"sampled should be [B,V,C], got {tuple(sampled.shape)}")
        if support.shape != sampled.shape[:2]:
            raise ValueError(f"support should be [B,V], got {tuple(support.shape)} for sampled {tuple(sampled.shape)}")
        if geom.shape[:2] != sampled.shape[:2]:
            raise ValueError(f"geom should be [B,V,G], got {tuple(geom.shape)} for sampled {tuple(sampled.shape)}")

        valid = support.float() > 0
        sampled_for_model, support_for_prior, geom_for_model = self._apply_feature_ablation(sampled, support, geom)
        reduced = self.feature_reduce(sampled_for_model)
        token = torch.cat([reduced, geom_for_model], dim=-1)
        encoded = self.view_encoder(token)
        logits = self.view_gate(encoded).squeeze(-1)
        logits = logits + torch.log(support_for_prior.clamp_min(1e-6))
        logits = logits.masked_fill(~valid, -1.0e4)
        attn = torch.softmax(logits, dim=1) * valid.float()
        attn = attn / attn.sum(dim=1, keepdim=True).clamp_min(1e-6)
        pooled = (attn[..., None] * encoded).sum(dim=1)
        if self.use_stats:
            pooled = torch.cat([pooled, stats.float()], dim=-1)
        return self.classifier(pooled).squeeze(-1)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in keys})


def _normalize_stats(stats: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (stats - mean[None]) / std[None].clamp_min(1e-6)


def balanced_indices(labels: torch.Tensor, max_pos: int, neg_per_pos: int, generator: torch.Generator) -> torch.Tensor:
    pos_idx = torch.where(labels > 0.5)[0]
    neg_idx = torch.where(labels <= 0.5)[0]
    if pos_idx.numel() == 0 or neg_idx.numel() == 0:
        return labels.new_zeros((0,), dtype=torch.long)
    if pos_idx.numel() > int(max_pos):
        perm = torch.randperm(pos_idx.numel(), generator=generator, device=pos_idx.device)[: int(max_pos)]
        pos_idx = pos_idx[perm]
    neg_count = min(int(neg_idx.numel()), int(pos_idx.numel()) * int(neg_per_pos))
    if neg_idx.numel() > neg_count:
        perm = torch.randperm(neg_idx.numel(), generator=generator, device=neg_idx.device)[:neg_count]
        neg_idx = neg_idx[perm]
    idx = torch.cat([pos_idx, neg_idx], dim=0)
    perm = torch.randperm(idx.numel(), generator=generator, device=idx.device)
    return idx[perm]


def _make_stat_features(
    sampled: torch.Tensor,
    support_weights: torch.Tensor,
    view_geom: torch.Tensor,
    points_obj: torch.Tensor,
    object_to_world: torch.Tensor,
    extrinsics: torch.Tensor,
    extrinsics_are_c2w: bool,
    masks: torch.Tensor,
    intrinsics: torch.Tensor,
    args: argparse.Namespace,
    pose_head,
) -> torch.Tensor:
    per_voxel = _per_voxel_feature_agreement(sampled, support_weights)
    old_entropy, old_max = _old_attention_stats(support_weights)
    ray_angle, baseline = _per_voxel_ray_stats(points_obj, object_to_world, extrinsics, extrinsics_are_c2w, support_weights)

    points_h = torch.cat([points_obj, torch.ones((points_obj.shape[0], 1), device=points_obj.device)], dim=1)
    points_world = (points_h @ object_to_world.to(points_obj.device, torch.float32).T)[:, :3]
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
    stats = torch.stack([feature_map[name].float() for name in STAT_FEATURE_NAMES], dim=1)
    return torch.nan_to_num(stats, nan=0.0, posinf=0.0, neginf=0.0)


@torch.no_grad()
def extract_sample_voxels(
    dataset: MultiviewSparseManifestDataset,
    image_cond_model,
    pose_head,
    sample_index: int,
    args: argparse.Namespace,
    device: torch.device,
    generator: torch.Generator,
) -> tuple[dict[str, torch.Tensor], dict]:
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
        object_to_world=volume.object_to_world,
    )
    if args.visibility_depth_tolerance > 0:
        depth_tolerance = float(args.visibility_depth_tolerance)
    else:
        depth_tolerance = max(float(volume.extent_world) * float(args.visibility_depth_tolerance_ratio), 1e-4)
    front_depth_maps, _ = visual_hull_front_depth_maps(
        masks_sq,
        intrinsics_sq,
        extrinsics,
        extrinsics_are_c2w=extrinsics_are_c2w,
        camera_forward_sign=args.camera_forward_sign,
        object_to_world=volume.object_to_world,
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

    stat_features = _make_stat_features(
        sampled,
        support_weights,
        view_geom,
        points_obj,
        volume.object_to_world,
        extrinsics,
        extrinsics_are_c2w,
        masks,
        intrinsics,
        args,
        pose_head,
    )

    supported = support_weights.sum(dim=0) > 0
    labels = target_mask.float()
    stat_supported = stat_features[supported]
    labels_supported = labels[supported]
    if stat_supported.numel() == 0:
        meta = {
            "uid": batch["uid"],
            "target_count": int(target_mask.sum().item()),
            "supported_count": int(supported.sum().item()),
            "positive_supported_count": int((supported & target_mask).sum().item()),
            "used": False,
        }
        return {
            "sampled": sampled.permute(1, 0, 2)[:0],
            "support": support_weights.T[:0],
            "geom": view_geom.permute(1, 0, 2)[:0],
            "stats": stat_supported,
            "labels": labels_supported,
        }, meta

    selected_rows_t = balanced_indices(labels_supported, args.max_pos_per_sample, args.neg_per_pos, generator)
    if selected_rows_t.numel() == 0:
        meta = {
            "uid": batch["uid"],
            "target_count": int(target_mask.sum().item()),
            "supported_count": int(supported.sum().item()),
            "positive_supported_count": int((supported & target_mask).sum().item()),
            "used": False,
        }
        return {
            "sampled": sampled.permute(1, 0, 2)[:0],
            "support": support_weights.T[:0],
            "geom": view_geom.permute(1, 0, 2)[:0],
            "stats": stat_supported[:0],
            "labels": labels_supported[:0],
        }, meta

    supported_indices = torch.where(supported)[0]
    selected_grid_idx = supported_indices[selected_rows_t]

    data = {
        "sampled": sampled[:, selected_grid_idx].permute(1, 0, 2).detach().cpu().to(torch.float16),
        "support": support_weights[:, selected_grid_idx].T.detach().cpu().to(torch.float16),
        "geom": view_geom[:, selected_grid_idx].permute(1, 0, 2).detach().cpu().to(torch.float16),
        "stats": stat_features[selected_grid_idx].detach().cpu().float(),
        "labels": labels[selected_grid_idx].detach().cpu().float(),
    }
    meta = {
        "uid": batch["uid"],
        "target_count": int(target_mask.sum().item()),
        "supported_count": int(supported.sum().item()),
        "positive_supported_count": int((supported & target_mask).sum().item()),
        "volume_fallback": bool(volume.fallback),
        "volume_extent_world": float(volume.extent_world),
        "used": True,
        "train_voxels": int(data["labels"].numel()),
        "train_pos": int((data["labels"] > 0.5).sum().item()),
    }
    return data, meta


def _concat_dataset(items: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    if not items:
        return {}
    return {key: torch.cat([item[key] for item in items], dim=0) for key in ("sampled", "support", "geom", "stats", "labels")}


def _normalize_dataset_stats(data: dict[str, torch.Tensor], mean: torch.Tensor, std: torch.Tensor) -> dict[str, torch.Tensor]:
    if not data:
        return data
    out = dict(data)
    out["stats"] = _normalize_stats(out["stats"].float(), mean.float(), std.float())
    return out


def extract_dataset(
    dataset: MultiviewSparseManifestDataset,
    indices: list[int],
    image_cond_model,
    pose_head,
    args: argparse.Namespace,
    device: torch.device,
    desc: str,
) -> tuple[dict[str, torch.Tensor], list[dict]]:
    generator = torch.Generator(device=device)
    generator.manual_seed(int(args.seed))
    items: list[dict[str, torch.Tensor]] = []
    metas: list[dict] = []
    for idx in tqdm(indices, desc=desc, unit="sample", dynamic_ncols=True):
        item, meta = extract_sample_voxels(dataset, image_cond_model, pose_head, idx, args, device, generator)
        metas.append(meta)
        if meta.get("used"):
            items.append(item)
    return _concat_dataset(items), metas


def data_loader_indices(count: int, batch_size: int, *, device: torch.device, shuffle: bool) -> list[torch.Tensor]:
    indices = torch.arange(count, device=device)
    if shuffle:
        indices = indices[torch.randperm(count, device=device)]
    return [indices[start : start + batch_size] for start in range(0, count, batch_size)]


def batch_to_device(data: dict[str, torch.Tensor], ids: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        data["sampled"][ids.cpu()].to(device=device, dtype=torch.float32),
        data["support"][ids.cpu()].to(device=device, dtype=torch.float32),
        data["geom"][ids.cpu()].to(device=device, dtype=torch.float32),
        data["stats"][ids.cpu()].to(device=device, dtype=torch.float32),
        data["labels"][ids.cpu()].to(device=device, dtype=torch.float32),
    )


@torch.no_grad()
def evaluate_model(model: nn.Module, data: dict[str, torch.Tensor], device: torch.device, batch_size: int) -> dict:
    if not data:
        return {"count": 0}
    model.eval()
    logits_all = []
    labels_all = []
    for ids in data_loader_indices(int(data["labels"].shape[0]), batch_size, device=device, shuffle=False):
        sampled, support, geom, stats, labels = batch_to_device(data, ids, device)
        logits_all.append(model(sampled, support, geom, stats).detach().cpu())
        labels_all.append(labels.detach().cpu())
    logits = torch.cat(logits_all, dim=0)
    labels = torch.cat(labels_all, dim=0)
    loss = F.binary_cross_entropy_with_logits(logits, labels)
    scores = torch.sigmoid(logits)
    pred = scores >= 0.5
    pos = labels > 0.5
    neg = ~pos
    tp = (pred & pos).sum().float()
    fp = (pred & neg).sum().float()
    fn = ((~pred) & pos).sum().float()
    precision = tp / (tp + fp).clamp_min(1.0)
    recall = tp / (tp + fn).clamp_min(1.0)
    return {
        "count": int(labels.numel()),
        "pos": int(pos.sum().item()),
        "neg": int(neg.sum().item()),
        "loss": float(loss.item()),
        "auc": auc_score(labels, scores),
        "ap": average_precision(labels, scores),
        "precision@0.5": float(precision.item()),
        "recall@0.5": float(recall.item()),
        "target_score_mean": float(scores[pos].mean().item()) if pos.any() else None,
        "non_target_score_mean": float(scores[neg].mean().item()) if neg.any() else None,
        "score_gap": float((scores[pos].mean() - scores[neg].mean()).item()) if pos.any() and neg.any() else None,
    }


def write_report(path: Path, result: dict) -> None:
    lines = [
        "# Voxel Feature Target-aware Diagnostic Head",
        "",
        f"时间：`{result['timestamp_utc']}`",
        f"train manifest：`{result['train_manifest']}`",
        f"val manifest：`{result['val_manifest']}`",
        f"pose head checkpoint：`{result.get('pose_head_checkpoint') or ''}`",
        f"feature ablation：`{result['model_config'].get('feature_ablation', 'full')}`",
        f"use stats：`{result['model_config'].get('use_stats', True)}`",
        "",
        "## 这个实验在测什么",
        "",
        "这个实验直接读取每个 voxel 的多视角 back-projected DINO features、view geometry、support weights 和少量统计量，训练一个轻量分类器区分 target sparse coords 与 supported non-target coords。",
        "",
        "如果这个 head 的 val AUC 明显高于统计量 MLP，说明原始投影特征里确实包含 target-aware 信息，只是之前统计量丢掉了；如果仍然接近随机，说明当前 2D-3D projection/visibility/volume 对齐本身不足。",
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
            "## 判断标准",
            "",
            "- `val AUC > 0.65` 且 score gap 为正：原始 voxel feature 有可用 target-aware 信号，可以继续做 feature-level gate loss。",
            "- `val AUC` 仍接近 `0.5`：当前 projection/visibility/volume 对齐或训练 target 本身不足，不建议继续接 sparse flow。",
            "",
            "## 输入组成",
            "",
            "```text",
            "sampled_features: [voxel, view, DINO_dim]",
            "support_weights:  [voxel, view]",
            "view_geom:        [voxel, view, 11]",
            "stats:            18-D condition statistics",
            "```",
            "",
            "## Feature Ablation 定义",
            "",
            "```text",
            "full              = DINO + all view_geom + support prior",
            "feature_only      = DINO only + uniform valid-view pooling",
            "geometry_only     = u/v/depth/xyz projection geometry only",
            "support_only      = mask/visibility/support channels only",
            "uv_depth_only     = u/v/depth projection only",
            "xyz_only          = canonical xyz only",
            "uv_depth_xyz      = u/v/depth + canonical xyz only",
            "feature_geometry  = DINO + projection geometry",
            "feature_support   = DINO + support channels",
            "geometry_support  = all view_geom + support prior, without DINO",
            "```",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose whether raw back-projected voxel features are target-aware.")
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
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--reduced_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--no_stats", action="store_true")
    parser.add_argument(
        "--feature_ablation",
        choices=[
            "full",
            "feature_only",
            "geometry_only",
            "uv_depth_only",
            "xyz_only",
            "uv_depth_xyz",
            "support_only",
            "feature_geometry",
            "feature_uv_depth",
            "feature_xyz",
            "feature_support",
            "geometry_support",
        ],
        default="full",
        help=(
            "Input isolation for the voxel feature diagnostic head. "
            "full=DINO+all view geometry+support prior; feature_only=DINO only with uniform valid-view pooling; "
            "geometry_only=projection xyz/u/v/depth only; support_only=mask/visibility/support only; "
            "uv_depth_only=u/v/depth projection only; xyz_only=canonical xyz only; "
            "uv_depth_xyz=u/v/depth+canonical xyz only; "
            "feature_geometry=DINO+projection geometry; feature_uv_depth=DINO+u/v/depth; "
            "feature_xyz=DINO+canonical xyz; feature_support=DINO+support channels; "
            "geometry_support=all view geometry/support without DINO."
        ),
    )
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

    train_data, train_meta = extract_dataset(train_dataset, train_indices, image_cond_model, pose_head, args, device, "Extract train")
    val_data, val_meta = extract_dataset(val_dataset, val_indices, image_cond_model, pose_head, args, device, "Extract val")
    if not train_data:
        raise RuntimeError("No usable training voxels were extracted.")

    stat_mean = train_data["stats"].float().mean(dim=0)
    stat_std = train_data["stats"].float().std(dim=0, unbiased=False).clamp_min(1e-6)
    train_data = _normalize_dataset_stats(train_data, stat_mean, stat_std)
    val_data = _normalize_dataset_stats(val_data, stat_mean, stat_std)

    model = VoxelFeatureDiagnosticHead(
        feature_dim=feature_dim,
        geom_dim=int(train_data["geom"].shape[-1]),
        stat_dim=int(train_data["stats"].shape[-1]),
        reduced_dim=args.reduced_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        use_stats=not args.no_stats,
        feature_ablation=args.feature_ablation,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    rows: list[dict] = []
    for epoch in tqdm(range(args.epochs), desc="Train voxel feature head", dynamic_ncols=True):
        model.train()
        total_loss = 0.0
        total_pos_loss = 0.0
        total_neg_loss = 0.0
        batches = 0
        for ids in data_loader_indices(int(train_data["labels"].shape[0]), args.batch_size, device=device, shuffle=True):
            sampled, support, geom, stats, labels = batch_to_device(train_data, ids, device)
            logits = model(sampled, support, geom, stats)
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            pos_loss = F.binary_cross_entropy_with_logits(logits[labels > 0.5], labels[labels > 0.5]) if (labels > 0.5).any() else loss * 0
            neg_loss = F.binary_cross_entropy_with_logits(logits[labels <= 0.5], labels[labels <= 0.5]) if (labels <= 0.5).any() else loss * 0
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu().item())
            total_pos_loss += float(pos_loss.detach().cpu().item())
            total_neg_loss += float(neg_loss.detach().cpu().item())
            batches += 1
        rows.append(
            asdict(
                TrainRow(
                    epoch=epoch,
                    step=epoch + 1,
                    loss=total_loss / max(batches, 1),
                    pos_loss=total_pos_loss / max(batches, 1),
                    neg_loss=total_neg_loss / max(batches, 1),
                    lr=float(optimizer.param_groups[0]["lr"]),
                )
            )
        )

    train_metrics = evaluate_model(model, train_data, device, args.batch_size)
    val_metrics = evaluate_model(model, val_data, device, args.batch_size)
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "train_manifest": args.train_manifest,
        "val_manifest": args.val_manifest,
        "train_indices": train_indices,
        "val_indices": val_indices,
        "pose_head_checkpoint": args.pose_head_checkpoint,
        "model_config": model.config,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "train_meta": train_meta,
        "val_meta": val_meta,
    }

    torch.save(
        {
            "model": model.state_dict(),
            "config": model.config,
            "stat_mean": stat_mean,
            "stat_std": stat_std,
            "args": vars(args),
            "metrics": {"train": train_metrics, "val": val_metrics},
        },
        output_dir / "voxel_feature_diagnostic_head.pt",
    )
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(output_dir / "train_log.csv", rows)
    write_csv(output_dir / "train_meta.csv", train_meta)
    write_csv(output_dir / "val_meta.csv", val_meta)
    write_report(output_dir / "report.md", payload)
    print(json.dumps({"output_dir": str(output_dir), "train": train_metrics, "val": val_metrics}, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
