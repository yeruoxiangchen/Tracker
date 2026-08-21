from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

TRACKER_ROOT = Path(__file__).resolve().parents[1]
PIXAL3D_ROOT = TRACKER_ROOT / "Pixal3D"
sys.path.insert(0, str(TRACKER_ROOT))
sys.path.insert(0, str(PIXAL3D_ROOT))

from pixal3d_multiview.eval_condition_view_consistency import (  # noqa: E402
    extract_patch_features,
    load_target_mask,
    parse_indices,
)
from pixal3d_multiview.multiview_projection import (  # noqa: E402
    estimate_object_volume_from_visual_hull,
    pixal3d_grid_points,
    project_points_multi_view,
    resize_masks,
    scale_intrinsics_to_square,
    visual_hull_front_depth_maps,
)
from pixal3d_multiview.projection_alignment_head import (  # noqa: E402
    GEOM_MODES,
    build_projection_alignment_head,
)
from pixal3d_multiview.train_pose_consistency_head import parse_modes, parse_weights  # noqa: E402
from pixal3d_multiview.train_target_aware_diagnostic_head import (  # noqa: E402
    auc_score,
    average_precision,
)
from pixal3d_multiview.train_sparse_multiview import (  # noqa: E402
    IMAGE_COND_CONFIG,
    MultiviewSparseManifestDataset,
    apply_pose_mode,
    build_image_cond_model,
)
from pixal3d_multiview.view_aggregator import sample_view_features_for_aggregation  # noqa: E402


NEGATIVE_MODES = (
    "reverse",
    "cyclic_shift1",
    "cyclic_shift2",
    "cross_sample",
    "identity",
    "noise",
    "large_noise",
)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in keys})


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


def make_negative_batch(dataset: MultiviewSparseManifestDataset, batch: dict, mode: str, seed: int) -> dict:
    mode = str(mode).lower()
    if mode == "cross_sample":
        other_idx = (seed * 1103515245 + 12345) % len(dataset)
        for _ in range(16):
            other = dataset[int(other_idx)]
            if other.get("uid") != batch.get("uid"):
                return make_cross_sample_batch(batch, other)
            other_idx = (other_idx + 1) % len(dataset)
        return make_cross_sample_batch(batch, dataset[int(other_idx)])
    return apply_pose_mode(batch, mode, seed)


def soft_target_from_mask(mask: torch.Tensor, resolution: int, neighbor1: float, neighbor2: float) -> torch.Tensor:
    hard = mask.float().view(1, 1, resolution, resolution, resolution)
    dil1 = F.max_pool3d(hard, kernel_size=3, stride=1, padding=1)
    dil2 = F.max_pool3d(hard, kernel_size=5, stride=1, padding=2)
    soft = torch.zeros_like(hard)
    soft = torch.where(dil2 > 0, torch.full_like(soft, float(neighbor2)), soft)
    soft = torch.where(dil1 > 0, torch.full_like(soft, float(neighbor1)), soft)
    soft = torch.where(hard > 0, torch.ones_like(soft), soft)
    return soft.flatten().clamp(0.0, 1.0)


def load_target_soft(latent_path: str, resolution: int, device: torch.device, args: argparse.Namespace) -> tuple[torch.Tensor, torch.Tensor]:
    target_mask, _, _ = load_target_mask(latent_path, resolution, device)
    target_soft = soft_target_from_mask(
        target_mask,
        resolution,
        neighbor1=float(args.target_neighbor1),
        neighbor2=float(args.target_neighbor2),
    )
    return target_mask.bool(), target_soft.float()


@torch.no_grad()
def extract_alignment_pack(
    batch: dict,
    image_cond_model,
    args: argparse.Namespace,
    device: torch.device,
    *,
    patch_features: Optional[torch.Tensor] = None,
) -> dict:
    if patch_features is None:
        patch_features, feature_meta = extract_patch_features(image_cond_model, batch["images"], device)
    else:
        feature_meta = {
            "num_views": int(patch_features.shape[0]),
            "feature_dim": int(patch_features.shape[-1]),
        }

    target_mask, target_soft = load_target_soft(batch["latent_path"], args.ss_grid_resolution, device, args)
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
    sampled, support, view_geom, sample_stats = sample_view_features_for_aggregation(
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
    return {
        "sampled": sampled.detach(),
        "support": support.detach(),
        "geom": view_geom.detach(),
        "target_mask": target_mask.detach(),
        "target_soft": target_soft.detach(),
        "patch_features": patch_features.detach(),
        "meta": {
            "uid": batch.get("uid"),
            "feature_meta": feature_meta,
            "sample_stats": sample_stats,
            "volume_fallback": bool(volume.fallback),
            "volume_extent_world": float(volume.extent_world),
            "target_count": int(target_mask.sum().item()),
            "target_soft_count": int((target_soft > 0).sum().item()),
            "supported_count": int((support.sum(dim=0) > 0).sum().item()),
        },
    }


def balanced_voxel_indices(
    target_soft: torch.Tensor,
    support: torch.Tensor,
    max_pos: int,
    neg_per_pos: int,
    generator: torch.Generator,
) -> torch.Tensor:
    supported = support.sum(dim=0) > 0
    pos = torch.where(target_soft > 0)[0]
    neg = torch.where((target_soft <= 0) & supported)[0]
    if pos.numel() == 0 or neg.numel() == 0:
        return torch.empty((0,), device=target_soft.device, dtype=torch.long)
    if pos.numel() > int(max_pos):
        pos = pos[torch.randperm(pos.numel(), generator=generator, device=pos.device)[: int(max_pos)]]
    neg_count = min(int(neg.numel()), int(pos.numel()) * int(neg_per_pos))
    if neg.numel() > neg_count:
        neg = neg[torch.randperm(neg.numel(), generator=generator, device=neg.device)[:neg_count]]
    idx = torch.cat([pos, neg], dim=0)
    return idx[torch.randperm(idx.numel(), generator=generator, device=idx.device)]


def weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return (values.float() * weights.float()).sum() / weights.float().sum().clamp_min(1e-6)


def view_feature_consistency_score(
    out: dict[str, torch.Tensor],
    support: torch.Tensor,
    target_soft: torch.Tensor,
    args: argparse.Namespace,
) -> torch.Tensor:
    """Leave-one-out view-feature agreement on fixed target voxels.

    This score targets the hard cases from the report: reverse/cyclic pose keeps
    plausible coverage, so support alone cannot distinguish it. For each target
    voxel, compare each supported view feature against the mean feature of the
    other supported views. Wrong image-pose correspondences should reduce this
    cross-view agreement.
    """
    encoded = F.normalize(out["match_embedding"].float(), dim=-1)
    support = support.float().clamp_min(0.0)
    valid = support > 0
    view_count = valid.float().sum(dim=0)
    enough_views = view_count >= int(args.consistency_min_views)

    target_weight = target_soft.float()
    target_weight = torch.where(
        target_weight > float(args.consistency_target_soft_threshold),
        target_weight,
        torch.zeros_like(target_weight),
    )
    if target_weight.sum() <= 0:
        target_weight = target_soft.float().clamp_min(0.0)

    summed = (encoded * valid[..., None].float()).sum(dim=0)
    loo_count = (view_count - 1.0).clamp_min(1.0)
    loo_mean = (summed[None] - encoded) / loo_count[None, :, None]
    loo_mean = F.normalize(loo_mean, dim=-1)
    similarity = (encoded * loo_mean).sum(dim=-1)
    similarity = torch.where(valid & enough_views[None], similarity, torch.zeros_like(similarity))

    per_voxel = similarity.sum(dim=0) / view_count.clamp_min(1.0)
    missing = torch.full_like(per_voxel, float(args.consistency_missing_score))
    per_voxel = torch.where(enough_views, per_voxel, missing)
    return weighted_mean(per_voxel, target_weight)


def match_visible_view_mask(support: torch.Tensor, geom: torch.Tensor, args: argparse.Namespace) -> torch.Tensor:
    """Visibility-filtered view mask for surface-level match supervision.

    The old consistency score only required nonzero support. That is too broad
    for voxel-level contrastive learning because internal/back-side voxels can
    project inside masks while sampling front-surface image features. This mask
    keeps only view-voxel pairs that look like visible surface observations.
    """
    valid = support.float() >= float(args.match_min_support_weight)
    if not bool(args.match_visible_surface_only):
        return valid
    geom = geom.float()
    valid = valid & (geom[..., 0] >= float(args.match_visibility_threshold))
    valid = valid & (geom[..., 1] >= float(args.match_mask_value_threshold))
    valid = valid & (geom[..., 4] >= float(args.match_mask_hit_threshold))
    if bool(args.match_require_valid_depth):
        valid = valid & (geom[..., 3] >= 0.5)
    return valid


def visible_surface_match_consistency_score(
    out: dict[str, torch.Tensor],
    support: torch.Tensor,
    geom: torch.Tensor,
    target_soft: torch.Tensor,
    args: argparse.Namespace,
) -> torch.Tensor:
    """Leave-one-out match-embedding agreement on visible surface voxels only."""
    encoded = F.normalize(out["match_embedding"].float(), dim=-1)
    valid = match_visible_view_mask(support, geom, args)
    view_count = valid.float().sum(dim=0)
    enough_views = view_count >= int(args.match_min_views)

    target_weight = target_soft.float()
    target_weight = torch.where(
        target_weight >= float(args.match_target_soft_threshold),
        target_weight,
        torch.zeros_like(target_weight),
    )
    if target_weight.sum() <= 0:
        target_weight = target_soft.float().clamp_min(0.0)

    summed = (encoded * valid[..., None].float()).sum(dim=0)
    loo_count = (view_count - 1.0).clamp_min(1.0)
    loo_mean = (summed[None] - encoded) / loo_count[None, :, None]
    loo_mean = F.normalize(loo_mean, dim=-1)
    similarity = (encoded * loo_mean).sum(dim=-1)
    similarity = torch.where(valid & enough_views[None], similarity, torch.zeros_like(similarity))

    per_voxel = similarity.sum(dim=0) / view_count.clamp_min(1.0)
    missing = torch.full_like(per_voxel, float(args.match_missing_score))
    per_voxel = torch.where(enough_views, per_voxel, missing)
    return weighted_mean(per_voxel, target_weight)


def visible_surface_match_logit_score(
    out: dict[str, torch.Tensor],
    support: torch.Tensor,
    geom: torch.Tensor,
    target_soft: torch.Tensor,
    args: argparse.Namespace,
) -> torch.Tensor:
    """Average learned match logits on target visible-surface voxels.

    Unlike visible_surface_match_consistency_score, this score uses the direct
    per-view match logit head. It is the quantity intended for the separate
    gate-logit-prior diagnostic: gate_logits_new = gate_logits_old + alpha * match_logits.
    """
    match_logits = out["match_logits"].float()
    valid = match_visible_view_mask(support, geom, args)
    view_count = valid.float().sum(dim=0)
    enough_views = view_count >= int(args.match_min_views)

    target_weight = target_soft.float()
    target_weight = torch.where(
        target_weight >= float(args.match_target_soft_threshold),
        target_weight,
        torch.zeros_like(target_weight),
    )
    if target_weight.sum() <= 0:
        target_weight = target_soft.float().clamp_min(0.0)

    per_voxel = (match_logits * valid.float()).sum(dim=0) / view_count.clamp_min(1.0)
    missing = torch.full_like(per_voxel, float(args.match_missing_score))
    per_voxel = torch.where(enough_views, per_voxel, missing)
    return weighted_mean(per_voxel, target_weight)


def pairwise_match_contrastive_loss(
    correct_out: dict[str, torch.Tensor],
    correct_support: torch.Tensor,
    correct_geom: torch.Tensor,
    wrong_outs: list[dict[str, torch.Tensor]],
    wrong_supports: list[torch.Tensor],
    wrong_geoms: list[torch.Tensor],
    target_soft: torch.Tensor,
    args: argparse.Namespace,
    generator: torch.Generator,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Supervised contrastive view correspondence loss on high-confidence voxels.

    Positives are different correct-pose views of the same target voxel.
    Negatives are wrong-pose views of the same target voxel. This directly
    targets reverse/cyclic/cross-sample image-pose mismatch.
    """
    device = target_soft.device
    correct_emb = F.normalize(correct_out["match_embedding"].float(), dim=-1)
    support_valid = correct_support.float() > 0
    support_view_count = support_valid.float().sum(dim=0)
    target_candidate = target_soft.float() >= float(args.match_target_soft_threshold)
    candidate_mask = target_candidate & (support_view_count >= int(args.match_min_views))

    correct_valid = match_visible_view_mask(correct_support, correct_geom, args)
    view_count = correct_valid.float().sum(dim=0)
    voxel_mask = target_candidate & (view_count >= int(args.match_min_views))
    voxel_indices = torch.where(voxel_mask)[0]
    if voxel_indices.numel() == 0 or not wrong_outs:
        return correct_emb.new_zeros(()), {
            "match_candidate_voxels": int(candidate_mask.sum().item()),
            "match_visible_voxels": int(voxel_mask.sum().item()),
            "match_visible_views_mean": None,
            "match_voxels": 0,
            "match_terms": 0,
            "match_positive_pairs": 0,
            "match_negative_pairs": 0,
            "match_pos_sim": None,
            "match_neg_sim": None,
        }
    max_voxels = int(args.match_max_voxels)
    if voxel_indices.numel() > max_voxels:
        perm = torch.randperm(voxel_indices.numel(), generator=generator, device=device)[:max_voxels]
        voxel_indices = voxel_indices[perm]

    losses = []
    pos_sims = []
    neg_sims = []
    positive_pair_count = 0
    negative_pair_count = 0
    temperature = float(args.match_temperature)
    wrong_valid_masks = [
        match_visible_view_mask(wrong_support, wrong_geom, args)
        for wrong_support, wrong_geom in zip(wrong_supports, wrong_geoms)
    ]
    for voxel in voxel_indices.tolist():
        c_mask = correct_valid[:, voxel]
        c_emb = correct_emb[c_mask, voxel]
        if c_emb.shape[0] < int(args.match_min_views):
            continue
        neg_chunks = []
        for wrong_out, w_mask_all in zip(wrong_outs, wrong_valid_masks):
            w_mask = w_mask_all[:, voxel]
            if w_mask.any():
                neg_chunks.append(F.normalize(wrong_out["match_embedding"].float()[w_mask, voxel], dim=-1))
        if not neg_chunks:
            continue
        neg_emb = torch.cat(neg_chunks, dim=0)
        if neg_emb.numel() == 0:
            continue
        positive_pair_count += int(c_emb.shape[0] * max(c_emb.shape[0] - 1, 0))
        negative_pair_count += int(c_emb.shape[0] * neg_emb.shape[0])

        pos_logits = (c_emb @ c_emb.T) / temperature
        diag = torch.eye(pos_logits.shape[0], device=device, dtype=torch.bool)
        pos_logits = pos_logits.masked_fill(diag, -1.0e4)
        neg_logits = (c_emb @ neg_emb.T) / temperature
        pos_logsumexp = torch.logsumexp(pos_logits, dim=1)
        all_logsumexp = torch.logsumexp(torch.cat([pos_logits, neg_logits], dim=1), dim=1)
        losses.append(-(pos_logsumexp - all_logsumexp).mean())

        with torch.no_grad():
            pos_pair_sims = (c_emb @ c_emb.T).masked_select(~diag)
            neg_pair_sims = c_emb @ neg_emb.T
            if pos_pair_sims.numel():
                pos_sims.append(pos_pair_sims.mean())
            if neg_pair_sims.numel():
                neg_sims.append(neg_pair_sims.mean())

    if not losses:
        return correct_emb.new_zeros(()), {
            "match_candidate_voxels": int(candidate_mask.sum().item()),
            "match_visible_voxels": int(voxel_mask.sum().item()),
            "match_visible_views_mean": float(view_count[voxel_mask].float().mean().detach().cpu().item()) if voxel_mask.any() else None,
            "match_voxels": int(voxel_indices.numel()),
            "match_terms": 0,
            "match_positive_pairs": 0,
            "match_negative_pairs": 0,
            "match_pos_sim": None,
            "match_neg_sim": None,
        }
    loss = torch.stack(losses).mean()
    return loss, {
        "match_candidate_voxels": int(candidate_mask.sum().item()),
        "match_visible_voxels": int(voxel_mask.sum().item()),
        "match_visible_views_mean": float(view_count[voxel_mask].float().mean().detach().cpu().item()) if voxel_mask.any() else None,
        "match_voxels": int(voxel_indices.numel()),
        "match_terms": len(losses),
        "match_positive_pairs": int(positive_pair_count),
        "match_negative_pairs": int(negative_pair_count),
        "match_pos_sim": float(torch.stack(pos_sims).mean().detach().cpu().item()) if pos_sims else None,
        "match_neg_sim": float(torch.stack(neg_sims).mean().detach().cpu().item()) if neg_sims else None,
    }


def pose_score_components(
    out: dict[str, torch.Tensor],
    support: torch.Tensor,
    geom: torch.Tensor,
    target_soft: torch.Tensor,
    args: argparse.Namespace,
) -> dict[str, torch.Tensor]:
    """Scores a condition on a fixed target voxel set.

    Old support-weighted scores can be inflated by wrong poses that keep only a
    small set of high-scoring supported voxels. The fixed-target scores below
    always normalize over target_soft[n] > threshold and explicitly penalize
    missing target support.
    """
    align_logits = out["align_logits"].float()
    support = support.float().clamp_min(0.0)
    target_weight = target_soft.float()
    target_weight = torch.where(
        target_weight > float(args.rank_target_soft_threshold),
        target_weight,
        torch.zeros_like(target_weight),
    )
    if target_weight.sum() <= 0:
        target_weight = target_soft.float().clamp_min(0.0)

    support_sum = support.sum(dim=0)
    support_any = support_sum > 0
    support_norm = support / support_sum[None].clamp_min(1e-6)
    per_voxel_align = (align_logits * support_norm).sum(dim=0)
    missing_value = torch.full_like(per_voxel_align, float(args.rank_missing_logit))
    per_voxel_align = torch.where(support_any, per_voxel_align, missing_value)

    old_support_weights = target_weight[None] * support
    old_attention_weights = target_weight[None] * out["attn"].float()
    fixed_align = weighted_mean(per_voxel_align, target_weight)
    coverage = weighted_mean(support_any.float(), target_weight)
    coverage_penalized = fixed_align - float(args.rank_coverage_weight) * (1.0 - coverage)
    voxel_score = weighted_mean(torch.sigmoid(out["voxel_logits"].float()), target_weight)
    combined = coverage_penalized + float(args.rank_voxel_weight) * voxel_score
    view_consistency = view_feature_consistency_score(out, support, target_soft, args)
    visible_match_consistency = visible_surface_match_consistency_score(out, support, geom, target_soft, args)
    visible_match_logit = visible_surface_match_logit_score(out, support, geom, target_soft, args)
    combined_consistency = combined + float(args.rank_consistency_score_weight) * view_consistency
    combined_visible_match = combined + float(args.rank_match_score_weight) * visible_match_consistency
    combined_match_logit = combined + float(args.rank_match_logit_score_weight) * visible_match_logit
    return {
        "support": weighted_mean(align_logits, old_support_weights),
        "attention": weighted_mean(align_logits, old_attention_weights),
        "fixed_align": fixed_align,
        "coverage": coverage,
        "coverage_penalized": coverage_penalized,
        "voxel": voxel_score,
        "combined": combined,
        "view_consistency": view_consistency,
        "combined_consistency": combined_consistency,
        "visible_match_consistency": visible_match_consistency,
        "combined_visible_match": combined_visible_match,
        "visible_match_logit": visible_match_logit,
        "combined_match_logit": combined_match_logit,
    }


def alignment_score(
    out: dict[str, torch.Tensor],
    support: torch.Tensor,
    geom: torch.Tensor,
    target_soft: torch.Tensor,
    args: argparse.Namespace,
    mode: str,
) -> torch.Tensor:
    scores = pose_score_components(out, support, geom, target_soft, args)
    if mode not in scores:
        raise ValueError(f"Unknown alignment score mode: {mode}; valid={sorted(scores)}")
    return scores[mode]


def compute_supervised_losses(
    out: dict[str, torch.Tensor],
    pack: dict,
    selected: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, float]]:
    target_soft = pack["target_soft"].float()
    support = pack["support"].float()
    align_logits = out["align_logits"]
    match_logits = out["match_logits"]
    voxel_logits = out["voxel_logits"]
    attn = out["attn"]

    labels = target_soft[selected]
    loss_voxel = F.binary_cross_entropy_with_logits(voxel_logits[selected], labels)

    view_label = target_soft[None, selected] * support[:, selected].clamp(0.0, 1.0)
    valid = support[:, selected] > 0
    view_loss_raw = F.binary_cross_entropy_with_logits(
        align_logits[:, selected],
        view_label,
        reduction="none",
    )
    view_weight = valid.float()
    loss_view = (view_loss_raw * view_weight).sum() / view_weight.sum().clamp_min(1.0)

    label_sum = view_label.sum(dim=0)
    attn_mask = (labels > 0) & (label_sum > 1e-6)
    if attn_mask.any():
        target_attn = view_label[:, attn_mask] / label_sum[attn_mask][None].clamp_min(1e-6)
        pred_attn = attn[:, selected][:, attn_mask].clamp_min(1e-8)
        loss_attn = -(target_attn * pred_attn.log()).sum(dim=0).mean()
    else:
        loss_attn = loss_voxel.new_zeros(())

    match_valid = match_visible_view_mask(support, pack["geom"].float(), args)[:, selected]
    if match_valid.any():
        match_targets = labels[None].expand_as(match_logits[:, selected]).clamp(0.0, 1.0)
        match_loss_raw = F.binary_cross_entropy_with_logits(
            match_logits[:, selected],
            match_targets,
            reduction="none",
        )
        match_weight = match_valid.float()
        loss_match_logit = (match_loss_raw * match_weight).sum() / match_weight.sum().clamp_min(1.0)
    else:
        loss_match_logit = loss_voxel.new_zeros(())

    match_view_label = labels[None] * support[:, selected].clamp(0.0, 1.0) * match_valid.float()
    match_label_sum = match_view_label.sum(dim=0)
    match_attn_mask = (
        (labels >= float(args.match_target_soft_threshold))
        & (match_label_sum > 1e-6)
        & (match_valid.float().sum(dim=0) >= int(args.match_min_views))
    )
    if match_attn_mask.any():
        target_match_attn = match_view_label[:, match_attn_mask] / match_label_sum[match_attn_mask][None].clamp_min(1e-6)
        pred_match_logits = match_logits[:, selected][:, match_attn_mask] / max(float(args.match_attention_temperature), 1e-6)
        pred_match_logits = pred_match_logits.masked_fill(~match_valid[:, match_attn_mask], -1.0e4)
        pred_match_attn = torch.softmax(pred_match_logits, dim=0).clamp_min(1e-8)
        loss_match_attn = -(target_match_attn * pred_match_attn.log()).sum(dim=0).mean()
    else:
        loss_match_attn = loss_voxel.new_zeros(())

    valid_logits = align_logits[support > 0]
    loss_reg = valid_logits.pow(2).mean() if valid_logits.numel() else loss_voxel.new_zeros(())
    stats = {
        "loss_voxel": float(loss_voxel.detach().cpu().item()),
        "loss_view": float(loss_view.detach().cpu().item()),
        "loss_attn": float(loss_attn.detach().cpu().item()),
        "loss_match_logit": float(loss_match_logit.detach().cpu().item()),
        "loss_match_attn": float(loss_match_attn.detach().cpu().item()),
        "loss_reg": float(loss_reg.detach().cpu().item()),
        "match_logit_visible_pairs": int(match_valid.sum().item()),
        "match_logit_positive_pairs": int((match_valid & (labels[None] >= float(args.match_target_soft_threshold))).sum().item()),
        "match_attn_voxels": int(match_attn_mask.sum().item()),
        "selected_voxels": int(selected.numel()),
        "selected_positive": int((labels > 0).sum().item()),
    }
    return (
        float(args.voxel_loss_weight) * loss_voxel
        + float(args.view_loss_weight) * loss_view
        + float(args.attn_loss_weight) * loss_attn
        + float(args.match_logit_loss_weight) * loss_match_logit
        + float(args.match_attention_loss_weight) * loss_match_attn
        + float(args.reg_loss_weight) * loss_reg,
        stats,
    )


def choose_negative_modes(modes: list[str], weights: Optional[list[float]], num: int) -> list[str]:
    if num >= len(modes):
        return list(modes)
    return random.choices(modes, weights=weights, k=num)


def train_one_step(
    model,
    dataset: MultiviewSparseManifestDataset,
    batch: dict,
    image_cond_model,
    args: argparse.Namespace,
    device: torch.device,
    negative_modes: list[str],
    negative_weights: Optional[list[float]],
    generator: torch.Generator,
    step: int,
) -> tuple[torch.Tensor, dict]:
    correct_pack = extract_alignment_pack(batch, image_cond_model, args, device)
    selected = balanced_voxel_indices(
        correct_pack["target_soft"],
        correct_pack["support"],
        args.max_pos_per_sample,
        args.neg_per_pos,
        generator,
    )
    if selected.numel() == 0:
        return torch.zeros((), device=device, requires_grad=True), {"used": False, "uid": batch.get("uid")}

    correct_out = model(correct_pack["sampled"], correct_pack["support"], correct_pack["geom"])
    supervised_loss, stats = compute_supervised_losses(correct_out, correct_pack, selected, args)
    correct_score = alignment_score(
        correct_out,
        correct_pack["support"],
        correct_pack["geom"],
        correct_pack["target_soft"],
        args,
        args.rank_score_type,
    )
    correct_score_parts = pose_score_components(
        correct_out,
        correct_pack["support"],
        correct_pack["geom"],
        correct_pack["target_soft"],
        args,
    )
    correct_consistency = correct_score_parts["view_consistency"]

    rank_terms = []
    consistency_rank_terms = []
    wrong_rows = []
    wrong_outs = []
    wrong_supports = []
    wrong_geoms = []
    selected_modes = choose_negative_modes(negative_modes, negative_weights, int(args.num_negatives))
    for i, mode in enumerate(selected_modes):
        wrong_seed = int(args.seed + step * 104729 + i * 9176 + 37)
        wrong_batch = make_negative_batch(dataset, batch, mode, wrong_seed)
        wrong_pack = extract_alignment_pack(
            wrong_batch,
            image_cond_model,
            args,
            device,
            patch_features=correct_pack["patch_features"],
        )
        wrong_out = model(wrong_pack["sampled"], wrong_pack["support"], wrong_pack["geom"])
        wrong_outs.append(wrong_out)
        wrong_supports.append(wrong_pack["support"])
        wrong_geoms.append(wrong_pack["geom"])
        wrong_score = alignment_score(
            wrong_out,
            wrong_pack["support"],
            wrong_pack["geom"],
            correct_pack["target_soft"],
            args,
            args.rank_score_type,
        )
        wrong_score_parts = pose_score_components(
            wrong_out,
            wrong_pack["support"],
            wrong_pack["geom"],
            correct_pack["target_soft"],
            args,
        )
        rank_terms.append(F.relu(float(args.rank_margin) - correct_score + wrong_score))
        wrong_consistency = wrong_score_parts["view_consistency"]
        consistency_rank_terms.append(
            F.relu(float(args.consistency_margin) - correct_consistency + wrong_consistency)
        )
        wrong_rows.append(
            {
                "mode": mode,
                "score": float(wrong_score.detach().cpu().item()),
                "view_consistency": float(wrong_consistency.detach().cpu().item()),
                "visible_match_logit": float(wrong_score_parts["visible_match_logit"].detach().cpu().item()),
                "coverage": float(wrong_score_parts["coverage"].detach().cpu().item()),
                "support_nonzero": int((wrong_pack["support"].sum(dim=0) > 0).sum().item()),
            }
        )

    loss_rank = torch.stack(rank_terms).mean() if rank_terms else supervised_loss.new_zeros(())
    loss_consistency_rank = (
        torch.stack(consistency_rank_terms).mean() if consistency_rank_terms else supervised_loss.new_zeros(())
    )
    loss_consistency_pos = 1.0 - correct_consistency
    loss_match_contrastive, match_stats = pairwise_match_contrastive_loss(
        correct_out,
        correct_pack["support"],
        correct_pack["geom"],
        wrong_outs,
        wrong_supports,
        wrong_geoms,
        correct_pack["target_soft"],
        args,
        generator,
    )
    total = (
        supervised_loss
        + float(args.rank_loss_weight) * loss_rank
        + float(args.consistency_rank_loss_weight) * loss_consistency_rank
        + float(args.consistency_positive_loss_weight) * loss_consistency_pos
        + float(args.match_contrastive_loss_weight) * loss_match_contrastive
    )
    stats.update(
        {
            "used": True,
            "uid": batch.get("uid"),
            "loss_supervised": float(supervised_loss.detach().cpu().item()),
            "loss_rank": float(loss_rank.detach().cpu().item()),
            "loss_consistency_rank": float(loss_consistency_rank.detach().cpu().item()),
            "loss_consistency_pos": float(loss_consistency_pos.detach().cpu().item()),
            "loss_match_contrastive": float(loss_match_contrastive.detach().cpu().item()),
            "correct_score": float(correct_score.detach().cpu().item()),
            "correct_coverage": float(correct_score_parts["coverage"].detach().cpu().item()),
            "correct_view_consistency": float(correct_consistency.detach().cpu().item()),
            "correct_visible_match_logit": float(correct_score_parts["visible_match_logit"].detach().cpu().item()),
            "rank_score_type": args.rank_score_type,
            "wrong_score_mean": float(np.mean([row["score"] for row in wrong_rows])) if wrong_rows else None,
            "wrong_view_consistency_mean": float(np.mean([row["view_consistency"] for row in wrong_rows])) if wrong_rows else None,
            "wrong_visible_match_logit_mean": float(np.mean([row["visible_match_logit"] for row in wrong_rows])) if wrong_rows else None,
            "wrong_coverage_mean": float(np.mean([row["coverage"] for row in wrong_rows])) if wrong_rows else None,
            "wrong_modes": ",".join([row["mode"] for row in wrong_rows]),
            "target_count": correct_pack["meta"]["target_count"],
            "target_soft_count": correct_pack["meta"]["target_soft_count"],
            "supported_count": correct_pack["meta"]["supported_count"],
            **match_stats,
        }
    )
    return total, stats


@torch.no_grad()
def target_metrics_for_split(
    model,
    dataset: MultiviewSparseManifestDataset,
    indices: list[int],
    image_cond_model,
    args: argparse.Namespace,
    device: torch.device,
    split: str,
) -> tuple[dict, list[dict], list[dict]]:
    model.eval()
    all_scores = []
    all_labels = []
    rows = []
    attention_rows = []
    for idx in tqdm(indices, desc=f"Eval target {split}", unit="sample", dynamic_ncols=True):
        batch = dataset[idx]
        pack = extract_alignment_pack(batch, image_cond_model, args, device)
        out = model(pack["sampled"], pack["support"], pack["geom"])
        support_any = pack["support"].sum(dim=0) > 0
        target = pack["target_mask"] & support_any
        non_target = (pack["target_soft"] <= 0) & support_any
        eval_mask = target | non_target
        scores = torch.sigmoid(out["voxel_logits"])
        labels = target.float()
        if eval_mask.any():
            all_scores.append(scores[eval_mask].detach().cpu())
            all_labels.append(labels[eval_mask].detach().cpu())
        target_score = scores[target]
        non_score = scores[non_target]
        rows.append(
            {
                "split": split,
                "index": idx,
                "uid": batch["uid"],
                "target_count": int(target.sum().item()),
                "non_target_count": int(non_target.sum().item()),
                "target_score_mean": float(target_score.mean().item()) if target_score.numel() else None,
                "non_target_score_mean": float(non_score.mean().item()) if non_score.numel() else None,
                "score_gap": float((target_score.mean() - non_score.mean()).item()) if target_score.numel() and non_score.numel() else None,
            }
        )

        old_attn = pack["support"] / pack["support"].sum(dim=0, keepdim=True).clamp_min(1e-6)
        attn = out["attn"]
        entropy = -(attn.clamp_min(1e-8) * attn.clamp_min(1e-8).log()).sum(dim=0)
        max_attn = attn.max(dim=0).values
        attn_l1 = (attn - old_attn).abs().sum(dim=0)
        for name, mask in (("target", target), ("non_target", non_target)):
            attention_rows.append(
                {
                    "split": split,
                    "group": name,
                    "count": int(mask.sum().item()),
                    "attention_entropy_mean": float(entropy[mask].mean().item()) if mask.any() else None,
                    "max_attention_mean": float(max_attn[mask].mean().item()) if mask.any() else None,
                    "old_vs_learned_l1_mean": float(attn_l1[mask].mean().item()) if mask.any() else None,
                }
            )

    if not all_scores:
        return {"count": 0}, rows, attention_rows
    scores = torch.cat(all_scores)
    labels = torch.cat(all_labels)
    pos = labels > 0.5
    neg = ~pos
    metrics = {
        "split": split,
        "count": int(labels.numel()),
        "target_count": int(pos.sum().item()),
        "non_target_count": int(neg.sum().item()),
        "auc": auc_score(labels, scores),
        "ap": average_precision(labels, scores),
        "target_score_mean": float(scores[pos].mean().item()) if pos.any() else None,
        "non_target_score_mean": float(scores[neg].mean().item()) if neg.any() else None,
        "score_gap": float((scores[pos].mean() - scores[neg].mean()).item()) if pos.any() and neg.any() else None,
    }
    return metrics, rows, attention_rows


@torch.no_grad()
def pose_alignment_for_split(
    model,
    dataset: MultiviewSparseManifestDataset,
    indices: list[int],
    image_cond_model,
    args: argparse.Namespace,
    device: torch.device,
    split: str,
    modes: list[str],
) -> tuple[list[dict], list[dict]]:
    model.eval()
    rows = []
    score_types = [part.strip() for part in str(args.eval_score_types).split(",") if part.strip()]
    for idx in tqdm(indices, desc=f"Eval pose {split}", unit="sample", dynamic_ncols=True):
        batch = dataset[idx]
        correct_pack = extract_alignment_pack(batch, image_cond_model, args, device)
        correct_out = model(correct_pack["sampled"], correct_pack["support"], correct_pack["geom"])
        correct_scores = pose_score_components(
            correct_out,
            correct_pack["support"],
            correct_pack["geom"],
            correct_pack["target_soft"],
            args,
        )
        for mode_i, mode in enumerate(modes):
            seed = int(args.seed + idx * 104729 + mode_i * 9176 + 11)
            wrong_batch = make_negative_batch(dataset, batch, mode, seed)
            wrong_pack = extract_alignment_pack(
                wrong_batch,
                image_cond_model,
                args,
                device,
                patch_features=correct_pack["patch_features"],
            )
            wrong_out = model(wrong_pack["sampled"], wrong_pack["support"], wrong_pack["geom"])
            wrong_scores = pose_score_components(
                wrong_out,
                wrong_pack["support"],
                wrong_pack["geom"],
                correct_pack["target_soft"],
                args,
            )
            for score_type in score_types:
                if score_type not in correct_scores:
                    raise ValueError(f"Unknown eval score type: {score_type}; valid={sorted(correct_scores)}")
                correct_score = correct_scores[score_type]
                wrong_score = wrong_scores[score_type]
                rows.append(
                    {
                        "split": split,
                        "index": idx,
                        "uid": batch["uid"],
                        "wrong_mode": mode,
                        "score_type": score_type,
                        "correct_score": float(correct_score.cpu().item()),
                        "wrong_score": float(wrong_score.cpu().item()),
                        "delta": float((correct_score - wrong_score).cpu().item()),
                        "correct_win": int((correct_score > wrong_score).item()),
                        "correct_coverage": float(correct_scores["coverage"].cpu().item()),
                        "wrong_coverage": float(wrong_scores["coverage"].cpu().item()),
                    }
                )

    summary = []
    by_mode: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_mode[row["wrong_mode"]].append(row)
    for mode, group in by_mode.items():
        by_score_type: dict[str, list[dict]] = defaultdict(list)
        for row in group:
            by_score_type[row["score_type"]].append(row)
        for score_type, score_group in by_score_type.items():
            deltas = [float(row["delta"]) for row in score_group]
            wins = [int(row["correct_win"]) for row in score_group]
            correct_coverages = [float(row["correct_coverage"]) for row in score_group]
            wrong_coverages = [float(row["wrong_coverage"]) for row in score_group]
            summary.append(
                {
                    "split": split,
                    "wrong_mode": mode,
                    "score_type": score_type,
                    "count": len(score_group),
                    "delta_mean": float(np.mean(deltas)) if deltas else None,
                    "delta_median": float(np.median(deltas)) if deltas else None,
                    "correct_wins": int(sum(wins)),
                    "correct_win_rate": float(np.mean(wins)) if wins else None,
                    "correct_coverage_mean": float(np.mean(correct_coverages)) if correct_coverages else None,
                    "wrong_coverage_mean": float(np.mean(wrong_coverages)) if wrong_coverages else None,
                }
            )
    return rows, summary


def summarize_train_log(rows: list[dict]) -> dict[str, dict[str, Optional[float]]]:
    keys = [
        "loss",
        "loss_match_logit",
        "loss_match_attn",
        "loss_match_contrastive",
        "match_logit_visible_pairs",
        "match_logit_positive_pairs",
        "match_attn_voxels",
        "match_candidate_voxels",
        "match_visible_voxels",
        "match_visible_views_mean",
        "match_terms",
        "match_positive_pairs",
        "match_negative_pairs",
        "match_pos_sim",
        "match_neg_sim",
        "correct_view_consistency",
        "wrong_view_consistency_mean",
        "correct_visible_match_logit",
        "wrong_visible_match_logit_mean",
    ]

    def summarize_window(window_rows: list[dict]) -> dict[str, Optional[float]]:
        out: dict[str, Optional[float]] = {}
        for key in keys:
            vals = [row.get(key) for row in window_rows if row.get(key) is not None]
            out[f"{key}_mean"] = float(np.mean(vals)) if vals else None
        return out

    used_rows = [row for row in rows if row.get("used", True)]
    if not used_rows:
        return {"all": {}, "first_128": {}, "last_128": {}}
    return {
        "all": summarize_window(used_rows),
        "first_128": summarize_window(used_rows[:128]),
        "last_128": summarize_window(used_rows[-128:]),
    }


def write_report(path: Path, result: dict) -> None:
    lines = [
        "# Projection Alignment Head Report",
        "",
        f"时间：`{result['timestamp_utc']}`",
        f"geom_mode：`{result['model_config']['geom_mode']}`",
        f"train manifest：`{result['train_manifest']}`",
        f"val manifest：`{result['val_manifest']}`",
        "",
        "## 这个实验在测什么",
        "",
        "阶段 1 独立训练 `ProjectionAlignmentHead`，不接 sparse flow。它同时预测 view-level `align_logit[v,n]` 与 voxel-level `voxel_logit[n]`，用于判断当前 projected DINO feature + u/v/depth 是否具备显式 2D-3D 对齐监督价值。",
        "",
        "## Target / Non-target",
        "",
        "| split | AUC | AP | target score | non-target score | gap |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in result["target_metrics"]:
        lines.append(
            "| {split} | {auc} | {ap} | {target} | {non_target} | {gap} |".format(
                split=row["split"],
                auc="" if row.get("auc") is None else f"{row['auc']:.4f}",
                ap="" if row.get("ap") is None else f"{row['ap']:.4f}",
                target="" if row.get("target_score_mean") is None else f"{row['target_score_mean']:.4f}",
                non_target="" if row.get("non_target_score_mean") is None else f"{row['non_target_score_mean']:.4f}",
                gap="" if row.get("score_gap") is None else f"{row['score_gap']:.4f}",
            )
        )
    lines.extend(
        [
            "",
            "## Correct vs Wrong Alignment",
            "",
            "| wrong pose | score type | delta mean | correct wins | win rate |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in result["pose_alignment_summary"]:
        lines.append(
            "| {wrong_mode} | {score_type} | {delta} | {wins}/{count} | {rate} |".format(
                wrong_mode=row["wrong_mode"],
                score_type=row["score_type"],
                delta="" if row.get("delta_mean") is None else f"{row['delta_mean']:.4f}",
                wins=row.get("correct_wins", 0),
                count=row.get("count", 0),
                rate="" if row.get("correct_win_rate") is None else f"{row['correct_win_rate']:.1%}",
            )
        )
    lines.extend(
        [
            "",
            "## Attention Sanity",
            "",
            "| split | group | count | entropy | max attention | old-vs-learned L1 |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in result["attention_summary"]:
        lines.append(
            "| {split} | {group} | {count} | {entropy} | {max_attn} | {l1} |".format(
                split=row["split"],
                group=row["group"],
                count=row["count"],
                entropy="" if row.get("attention_entropy_mean") is None else f"{row['attention_entropy_mean']:.4f}",
                max_attn="" if row.get("max_attention_mean") is None else f"{row['max_attention_mean']:.4f}",
                l1="" if row.get("old_vs_learned_l1_mean") is None else f"{row['old_vs_learned_l1_mean']:.4f}",
            )
        )
    train_summary = result.get("train_log_summary") or {}
    if train_summary:
        lines.extend(
            [
                "",
                "## Match / Visible Surface 训练统计",
                "",
                "| window | candidate voxels | visible voxels | visible views | pos pairs | neg pairs | pos sim | neg sim | match loss |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for window_name in ("first_128", "last_128", "all"):
            row = train_summary.get(window_name, {})
            if not row:
                continue
            lines.append(
                "| {window} | {cand} | {vis} | {views} | {pos_pairs} | {neg_pairs} | {pos_sim} | {neg_sim} | {loss} |".format(
                    window=window_name,
                    cand="" if row.get("match_candidate_voxels_mean") is None else f"{row['match_candidate_voxels_mean']:.1f}",
                    vis="" if row.get("match_visible_voxels_mean") is None else f"{row['match_visible_voxels_mean']:.1f}",
                    views="" if row.get("match_visible_views_mean_mean") is None else f"{row['match_visible_views_mean_mean']:.2f}",
                    pos_pairs="" if row.get("match_positive_pairs_mean") is None else f"{row['match_positive_pairs_mean']:.1f}",
                    neg_pairs="" if row.get("match_negative_pairs_mean") is None else f"{row['match_negative_pairs_mean']:.1f}",
                    pos_sim="" if row.get("match_pos_sim_mean") is None else f"{row['match_pos_sim_mean']:.4f}",
                    neg_sim="" if row.get("match_neg_sim_mean") is None else f"{row['match_neg_sim_mean']:.4f}",
                    loss="" if row.get("loss_match_contrastive_mean") is None else f"{row['loss_match_contrastive_mean']:.4f}",
                )
            )
        lines.extend(
            [
                "",
                "### Direct match-logit 监督",
                "",
                "| window | match-logit loss | visible pairs | positive pairs | correct match logit | wrong match logit |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for window_name in ("first_128", "last_128", "all"):
            row = train_summary.get(window_name, {})
            if not row:
                continue
            lines.append(
                "| {window} | {loss} | {pairs} | {pos_pairs} | {correct} | {wrong} |".format(
                    window=window_name,
                    loss="" if row.get("loss_match_logit_mean") is None else f"{row['loss_match_logit_mean']:.4f}",
                    pairs="" if row.get("match_logit_visible_pairs_mean") is None else f"{row['match_logit_visible_pairs_mean']:.1f}",
                    pos_pairs="" if row.get("match_logit_positive_pairs_mean") is None else f"{row['match_logit_positive_pairs_mean']:.1f}",
                    correct="" if row.get("correct_visible_match_logit_mean") is None else f"{row['correct_visible_match_logit_mean']:.4f}",
                    wrong="" if row.get("wrong_visible_match_logit_mean") is None else f"{row['wrong_visible_match_logit_mean']:.4f}",
                )
            )
        lines.extend(
            [
                "",
                "### Voxel-wise view ranking 监督",
                "",
                "| window | match-attn loss | supervised voxels |",
                "|---|---:|---:|",
            ]
        )
        for window_name in ("first_128", "last_128", "all"):
            row = train_summary.get(window_name, {})
            if not row:
                continue
            lines.append(
                "| {window} | {loss} | {voxels} |".format(
                    window=window_name,
                    loss="" if row.get("loss_match_attn_mean") is None else f"{row['loss_match_attn_mean']:.4f}",
                    voxels="" if row.get("match_attn_voxels_mean") is None else f"{row['match_attn_voxels_mean']:.1f}",
                )
            )
    lines.extend(
        [
            "",
            "## 判断标准",
            "",
            "- `val AUC > 0.60` 且 `gap > 0.05`：voxel-level target signal 有效。",
            "- `reverse / cyclic_shift1 / cyclic_shift2` 的 correct win rate >= 70%：view-level image-pose correspondence 有效。",
            "- 如果只能分开 `identity/noise/large_noise`，但分不开 `reverse/cyclic_shift`，说明仍没有解决真实难点。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_checkpoint(output_dir: Path, model, optimizer, args, step: int, name: str) -> None:
    payload = {
        "step": int(step),
        "projection_alignment_head": model.state_dict(),
        "projection_alignment_head_config": model.config,
        "optimizer": optimizer.state_dict(),
        "args": vars(args),
    }
    torch.save(payload, output_dir / name)
    torch.save(payload, output_dir / "last.pt")


def load_checkpoint(path: str, model, optimizer=None) -> int:
    state = torch.load(path, map_location="cpu")
    weights = state.get("projection_alignment_head", state.get("model"))
    if weights is None:
        raise ValueError(f"projection alignment checkpoint has no model weights: {path}")
    missing, unexpected = model.load_state_dict(weights, strict=False)
    print(
        f"[projection_alignment] checkpoint load strict=False missing={len(missing)} unexpected={len(unexpected)}"
    )
    if missing:
        print(f"[projection_alignment] missing keys sample={missing[:8]}")
    if unexpected:
        print(f"[projection_alignment] unexpected keys sample={unexpected[:8]}")
    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    return int(state.get("step", 0))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train standalone view-level 2D-3D projection alignment head.")
    parser.add_argument("--train_manifest", required=True)
    parser.add_argument("--val_manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--image_cond_model", default=IMAGE_COND_CONFIG["model_name"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_frames", type=int, default=8)
    parser.add_argument("--train_indices", default="0-255")
    parser.add_argument("--val_indices", default="0-127")
    parser.add_argument("--ss_grid_resolution", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_epochs", type=int, default=2)
    parser.add_argument("--max_steps", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--reduced_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--match_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--geom_mode", choices=GEOM_MODES, default="uv_depth_only")
    parser.add_argument("--max_pos_per_sample", type=int, default=768)
    parser.add_argument("--neg_per_pos", type=int, default=3)
    parser.add_argument("--target_neighbor1", type=float, default=0.5)
    parser.add_argument("--target_neighbor2", type=float, default=0.25)
    parser.add_argument("--voxel_loss_weight", type=float, default=1.0)
    parser.add_argument("--view_loss_weight", type=float, default=0.25)
    parser.add_argument("--attn_loss_weight", type=float, default=0.5)
    parser.add_argument("--rank_loss_weight", type=float, default=0.25)
    parser.add_argument("--reg_loss_weight", type=float, default=0.01)
    parser.add_argument("--rank_margin", type=float, default=0.2)
    parser.add_argument(
        "--rank_score_type",
        choices=[
            "support",
            "attention",
            "fixed_align",
            "coverage_penalized",
            "voxel",
            "combined",
            "view_consistency",
            "combined_consistency",
            "visible_match_consistency",
            "combined_visible_match",
            "visible_match_logit",
            "combined_match_logit",
        ],
        default="combined",
    )
    parser.add_argument("--rank_target_soft_threshold", type=float, default=0.0)
    parser.add_argument("--rank_missing_logit", type=float, default=-1.0)
    parser.add_argument("--rank_coverage_weight", type=float, default=0.5)
    parser.add_argument("--rank_voxel_weight", type=float, default=0.5)
    parser.add_argument("--rank_consistency_score_weight", type=float, default=0.25)
    parser.add_argument("--rank_match_score_weight", type=float, default=0.5)
    parser.add_argument("--rank_match_logit_score_weight", type=float, default=0.5)
    parser.add_argument("--consistency_rank_loss_weight", type=float, default=0.0)
    parser.add_argument("--consistency_positive_loss_weight", type=float, default=0.0)
    parser.add_argument("--consistency_margin", type=float, default=0.05)
    parser.add_argument("--consistency_min_views", type=int, default=2)
    parser.add_argument("--consistency_target_soft_threshold", type=float, default=0.0)
    parser.add_argument("--consistency_missing_score", type=float, default=-1.0)
    parser.add_argument("--match_contrastive_loss_weight", type=float, default=0.0)
    parser.add_argument("--match_logit_loss_weight", type=float, default=0.0)
    parser.add_argument("--match_attention_loss_weight", type=float, default=0.0)
    parser.add_argument("--match_attention_temperature", type=float, default=1.0)
    parser.add_argument("--match_temperature", type=float, default=0.07)
    parser.add_argument("--match_target_soft_threshold", type=float, default=0.999)
    parser.add_argument("--match_min_views", type=int, default=3)
    parser.add_argument("--match_max_voxels", type=int, default=256)
    parser.add_argument(
        "--match_visible_surface_only",
        type=int,
        default=0,
        help="If 1, pairwise match positives/negatives require high visibility/mask/depth support.",
    )
    parser.add_argument("--match_visibility_threshold", type=float, default=0.5)
    parser.add_argument("--match_mask_value_threshold", type=float, default=0.5)
    parser.add_argument("--match_mask_hit_threshold", type=float, default=0.5)
    parser.add_argument("--match_min_support_weight", type=float, default=1e-6)
    parser.add_argument("--match_require_valid_depth", type=int, default=1)
    parser.add_argument("--match_missing_score", type=float, default=-1.0)
    parser.add_argument(
        "--negative_modes",
        default="reverse,cyclic_shift1,cyclic_shift2,cross_sample,identity,noise,large_noise",
    )
    parser.add_argument("--negative_weights", default="0.25,0.22,0.22,0.16,0.08,0.035,0.035")
    parser.add_argument("--num_negatives", type=int, default=2)
    parser.add_argument(
        "--eval_pose_modes",
        default="reverse,cyclic_shift1,cyclic_shift2,cross_sample,identity,noise,large_noise",
    )
    parser.add_argument(
        "--eval_score_types",
        default="support,attention,fixed_align,coverage_penalized,voxel,combined,view_consistency,combined_consistency,visible_match_consistency,combined_visible_match,visible_match_logit,combined_match_logit",
    )
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--save_every", type=int, default=250)
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

    negative_modes = parse_modes(args.negative_modes, allow_correct=False)
    negative_weights = parse_weights(args.negative_weights, len(negative_modes))
    eval_pose_modes = parse_modes(args.eval_pose_modes, allow_correct=False)

    train_dataset = MultiviewSparseManifestDataset(args.train_manifest, max_frames=args.max_frames, apply_mask=True)
    val_dataset = MultiviewSparseManifestDataset(args.val_manifest, max_frames=args.max_frames, apply_mask=True)
    train_indices = parse_indices(args.train_indices, len(train_dataset))
    val_indices = parse_indices(args.val_indices, len(val_dataset))

    image_cond_model = build_image_cond_model(device, args.ss_grid_resolution, args.image_cond_model)
    feature_dim = int(getattr(image_cond_model, "embed_dim", image_cond_model.model.config.hidden_size))
    model = build_projection_alignment_head(
        feature_dim=feature_dim,
        geom_dim=11,
        reduced_dim=args.reduced_dim,
        hidden_dim=args.hidden_dim,
        match_dim=args.match_dim,
        dropout=args.dropout,
        geom_mode=args.geom_mode,
        device=device,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loaded_step = 0
    if args.checkpoint:
        loaded_step = load_checkpoint(args.checkpoint, model)
        print(f"[projection_alignment] loaded checkpoint={args.checkpoint} step={loaded_step}")
    if args.eval_only and not args.checkpoint:
        raise ValueError("--eval_only requires --checkpoint")
    generator = torch.Generator(device=device)
    generator.manual_seed(int(args.seed))

    print(
        f"[projection_alignment] train={len(train_indices)} val={len(val_indices)} "
        f"geom_mode={args.geom_mode} output={output_dir}"
    )
    rows = []
    step = int(loaded_step if args.eval_only else 0)
    if not args.eval_only:
        pbar = tqdm(total=int(args.max_steps), desc="Projection alignment", unit="step", dynamic_ncols=True)
        for epoch in range(int(args.max_epochs)):
            order = list(train_indices)
            random.shuffle(order)
            for idx in order:
                if step >= int(args.max_steps):
                    break
                batch = train_dataset[idx]
                model.train()
                optimizer.zero_grad(set_to_none=True)
                loss, stats = train_one_step(
                    model,
                    train_dataset,
                    batch,
                    image_cond_model,
                    args,
                    device,
                    negative_modes,
                    negative_weights,
                    generator,
                    step,
                )
                if stats.get("used"):
                    loss.backward()
                    optimizer.step()
                step += 1
                row = {"step": step, "epoch": epoch, "sample_index": idx, "loss": float(loss.detach().cpu().item()), **stats}
                rows.append(row)
                if step % int(args.log_every) == 0 or step == 1:
                    pbar.set_postfix(
                        {
                            "loss": f"{row['loss']:.4g}",
                            "vox": f"{row.get('loss_voxel', 0):.3g}",
                            "rank": f"{row.get('loss_rank', 0):.3g}",
                            "crank": f"{row.get('loss_consistency_rank', 0):.3g}",
                            "mlogit": f"{row.get('loss_match_logit', 0):.3g}",
                            "mattn": f"{row.get('loss_match_attn', 0):.3g}",
                            "match": f"{row.get('loss_match_contrastive', 0):.3g}",
                            "mvis": row.get("match_visible_voxels", 0),
                            "pos": f"{row.get('correct_score', 0):.3g}",
                            "neg": f"{row.get('wrong_score_mean', 0):.3g}",
                        },
                        refresh=False,
                    )
                pbar.update(1)
                if step % int(args.save_every) == 0:
                    save_checkpoint(output_dir, model, optimizer, args, step, f"step_{step}.pt")
            if step >= int(args.max_steps):
                break
        pbar.close()

        save_checkpoint(output_dir, model, optimizer, args, step, "final.pt")
        write_csv(output_dir / "train_log.csv", rows)
    else:
        print("[projection_alignment] eval_only=1, skip training")

    train_target_metrics, train_target_rows, train_attention_rows = target_metrics_for_split(
        model, train_dataset, train_indices, image_cond_model, args, device, "train"
    )
    val_target_metrics, val_target_rows, val_attention_rows = target_metrics_for_split(
        model, val_dataset, val_indices, image_cond_model, args, device, "val"
    )
    pose_rows, pose_summary = pose_alignment_for_split(
        model,
        val_dataset,
        val_indices,
        image_cond_model,
        args,
        device,
        "val",
        eval_pose_modes,
    )

    attention_summary = []
    for split_rows in (train_attention_rows, val_attention_rows):
        grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for row in split_rows:
            grouped[(row["split"], row["group"])].append(row)
        for (split, group), group_rows in grouped.items():
            count = sum(int(row["count"]) for row in group_rows)
            attention_summary.append(
                {
                    "split": split,
                    "group": group,
                    "count": count,
                    "attention_entropy_mean": float(np.mean([row["attention_entropy_mean"] for row in group_rows if row["attention_entropy_mean"] is not None]))
                    if any(row["attention_entropy_mean"] is not None for row in group_rows)
                    else None,
                    "max_attention_mean": float(np.mean([row["max_attention_mean"] for row in group_rows if row["max_attention_mean"] is not None]))
                    if any(row["max_attention_mean"] is not None for row in group_rows)
                    else None,
                    "old_vs_learned_l1_mean": float(np.mean([row["old_vs_learned_l1_mean"] for row in group_rows if row["old_vs_learned_l1_mean"] is not None]))
                    if any(row["old_vs_learned_l1_mean"] is not None for row in group_rows)
                    else None,
                }
            )

    result = {
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "train_manifest": args.train_manifest,
        "val_manifest": args.val_manifest,
        "train_indices": train_indices,
        "val_indices": val_indices,
        "model_config": model.config,
        "target_metrics": [train_target_metrics, val_target_metrics],
        "target_rows": train_target_rows + val_target_rows,
        "pose_alignment_summary": pose_summary,
        "attention_summary": attention_summary,
        "train_log_summary": summarize_train_log(rows),
        "args": vars(args),
    }
    (output_dir / "summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(output_dir / "target_voxel_metrics.csv", train_target_rows + val_target_rows)
    write_csv(output_dir / "attention_sanity.csv", train_attention_rows + val_attention_rows)
    write_csv(output_dir / "attention_summary.csv", attention_summary)
    write_csv(output_dir / "pose_alignment_rows.csv", pose_rows)
    write_csv(output_dir / "pose_alignment_summary.csv", pose_summary)
    write_report(output_dir / "report.md", result)
    print(json.dumps({"output_dir": str(output_dir), "target": result["target_metrics"], "pose": pose_summary}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
