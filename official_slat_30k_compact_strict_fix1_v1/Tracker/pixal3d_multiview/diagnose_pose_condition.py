from __future__ import annotations

import argparse
import csv
import json
import sys
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
from pixal3d_multiview.pose_consistency_head import load_pose_consistency_head  # noqa: E402
from pixal3d_multiview.train_pose_consistency_head import (  # noqa: E402
    deterministic_other_index,
    make_cross_sample_batch,
    parse_modes,
)
from pixal3d_multiview.train_sparse_multiview import (  # noqa: E402
    IMAGE_COND_CONFIG,
    MultiviewSparseManifestDataset,
    apply_pose_mode,
    build_image_cond_model,
    build_view_aggregator,
)
from pixal3d_multiview.view_aggregator import sample_view_features_for_aggregation  # noqa: E402


def _as_float(value) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return None
        return float(value.detach().float().mean().cpu().item())
    return float(value)


def _tensor_stats(values: torch.Tensor, prefix: str) -> dict:
    values = values.detach().float()
    values = values[torch.isfinite(values)]
    if values.numel() == 0:
        return {
            f"{prefix}_count": 0,
            f"{prefix}_mean": None,
            f"{prefix}_std": None,
            f"{prefix}_min": None,
            f"{prefix}_p05": None,
            f"{prefix}_median": None,
            f"{prefix}_p95": None,
            f"{prefix}_max": None,
        }
    qs = torch.quantile(values, torch.tensor([0.05, 0.5, 0.95], device=values.device))
    return {
        f"{prefix}_count": int(values.numel()),
        f"{prefix}_mean": float(values.mean().cpu().item()),
        f"{prefix}_std": float(values.std(unbiased=False).cpu().item()) if values.numel() > 1 else 0.0,
        f"{prefix}_min": float(values.min().cpu().item()),
        f"{prefix}_p05": float(qs[0].cpu().item()),
        f"{prefix}_median": float(qs[1].cpu().item()),
        f"{prefix}_p95": float(qs[2].cpu().item()),
        f"{prefix}_max": float(values.max().cpu().item()),
    }


def _masked_weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> Optional[float]:
    values = values.float()
    weights = weights.float().clamp_min(0.0)
    valid = torch.isfinite(values) & (weights > 0)
    if not valid.any():
        return None
    return float((values[valid] * weights[valid]).sum().detach().cpu().item() / weights[valid].sum().clamp_min(1e-6).detach().cpu().item())


def _softmax_attention(logits: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    masked = logits.float().masked_fill(~valid, -1.0e4)
    attn = torch.softmax(masked, dim=0) * valid.float()
    return attn / attn.sum(dim=0, keepdim=True).clamp_min(1e-6)


def _split_voxel_stats(values: torch.Tensor, target_mask: Optional[torch.Tensor], prefix: str, valid_mask: Optional[torch.Tensor] = None) -> dict:
    if target_mask is None or target_mask.numel() != values.numel():
        return {}
    values = values.detach().float()
    target = target_mask.to(device=values.device, dtype=torch.bool)
    non_target = ~target
    finite = torch.isfinite(values)
    if valid_mask is not None:
        finite = finite & valid_mask.to(device=values.device, dtype=torch.bool)
    out = {}
    for name, mask in (("target", target), ("non_target", non_target)):
        selected = values[finite & mask]
        out[f"{name}_{prefix}_mean"] = float(selected.mean().cpu().item()) if selected.numel() else None
    a = out.get(f"target_{prefix}_mean")
    b = out.get(f"non_target_{prefix}_mean")
    out[f"{prefix}_target_minus_non_target"] = float(a - b) if a is not None and b is not None else None
    return out


def _split_view_weighted_stats(values: torch.Tensor, weights: torch.Tensor, target_mask: Optional[torch.Tensor], prefix: str) -> dict:
    if target_mask is None or target_mask.numel() != values.shape[-1]:
        return {}
    target = target_mask.to(device=values.device, dtype=torch.bool)
    non_target = ~target
    out = {}
    for name, mask in (("target", target), ("non_target", non_target)):
        if mask.any():
            out[f"{name}_{prefix}_mean"] = _masked_weighted_mean(values[:, mask], weights[:, mask])
        else:
            out[f"{name}_{prefix}_mean"] = None
    a = out.get(f"target_{prefix}_mean")
    b = out.get(f"non_target_{prefix}_mean")
    out[f"{prefix}_target_minus_non_target"] = float(a - b) if a is not None and b is not None else None
    return out


def _attention_stats(attn: torch.Tensor, valid: torch.Tensor, prefix: str, target_mask: Optional[torch.Tensor] = None) -> dict:
    has_support = valid.any(dim=0)
    entropy = -(attn.clamp_min(1e-8) * attn.clamp_min(1e-8).log()).sum(dim=0)
    max_attn = attn.max(dim=0).values
    out = {
        f"{prefix}_entropy_mean": _as_float(entropy[has_support]) if has_support.any() else None,
        f"{prefix}_max_mean": _as_float(max_attn[has_support]) if has_support.any() else None,
    }
    out.update(_split_voxel_stats(entropy, target_mask, f"{prefix}_entropy", has_support))
    out.update(_split_voxel_stats(max_attn, target_mask, f"{prefix}_max", has_support))
    return out


def _attention_compare(
    old_attn: torch.Tensor,
    new_attn: torch.Tensor,
    valid: torch.Tensor,
    prefix: str,
    target_mask: Optional[torch.Tensor] = None,
) -> dict:
    has_support = valid.any(dim=0)
    l1 = (old_attn - new_attn).abs().sum(dim=0)
    old_top = old_attn.argmax(dim=0)
    new_top = new_attn.argmax(dim=0)
    top_agree = (old_top == new_top).float()
    out = {
        f"{prefix}_attn_l1_mean": _as_float(l1[has_support]) if has_support.any() else None,
        f"{prefix}_top_view_agree": _as_float(top_agree[has_support]) if has_support.any() else None,
    }
    out.update(_split_voxel_stats(l1, target_mask, f"{prefix}_attn_l1", has_support))
    out.update(_split_voxel_stats(top_agree, target_mask, f"{prefix}_top_view_agree", has_support))
    return out


def _feature_agreement(sampled_features: torch.Tensor, weights: torch.Tensor, target_mask: Optional[torch.Tensor]) -> dict:
    features = F.normalize(sampled_features.float(), dim=-1, eps=1e-6)
    weights = weights.float().clamp_min(0.0)
    valid = weights > 0
    support_count = valid.sum(dim=0)
    weight_sum = weights.sum(dim=0).clamp_min(1e-6)

    weighted_sum = (features * weights[..., None]).sum(dim=0)
    mean_feature = F.normalize(weighted_sum / weight_sum[:, None], dim=-1, eps=1e-6)
    all_mean_cos = (features * mean_feature[None]).sum(dim=-1).clamp(-1.0, 1.0)
    all_mean_cos = torch.where(valid, all_mean_cos, torch.full_like(all_mean_cos, float("nan")))

    other_weight = weight_sum[None] - weights
    other_sum = weighted_sum[None] - features * weights[..., None]
    loo_mean = F.normalize(other_sum / other_weight[..., None].clamp_min(1e-6), dim=-1, eps=1e-6)
    loo_cos = (features * loo_mean).sum(dim=-1).clamp(-1.0, 1.0)
    loo_valid = valid & (other_weight > 1e-6)
    loo_cos = torch.where(loo_valid, loo_cos, torch.full_like(loo_cos, float("nan")))

    pair_cos = None
    pair_valid = None
    if features.shape[0] > 1:
        gram = torch.einsum("vnc,wnc->vwn", features, features)
        pair_weights = weights[:, None, :] * weights[None, :, :]
        eye = torch.eye(features.shape[0], device=features.device, dtype=torch.bool)[:, :, None]
        pair_weights = torch.where(eye, torch.zeros_like(pair_weights), pair_weights)
        pair_denom = pair_weights.sum(dim=(0, 1))
        pair_valid = pair_denom > 0
        pair_cos = (gram * pair_weights).sum(dim=(0, 1)) / pair_denom.clamp_min(1e-6)

    mean_norm = torch.linalg.norm(weighted_sum / weight_sum[:, None], dim=-1)
    multi = support_count >= 2
    out = {
        "support_count_mean": _as_float(support_count.float()),
        "support_nonzero_ratio": _as_float((support_count > 0).float()),
        "support_multi_ratio": _as_float(multi.float()),
        "support_weight_sum_mean": _as_float(weights.sum(dim=0)),
        "agreement_all_mean_cos": _masked_weighted_mean(all_mean_cos, weights),
        "agreement_loo_cos": _masked_weighted_mean(loo_cos, torch.where(loo_valid, weights, torch.zeros_like(weights))),
        "agreement_mean_norm": _as_float(mean_norm[multi]) if multi.any() else None,
        "agreement_pair_cos": _as_float(pair_cos[pair_valid]) if pair_cos is not None and pair_valid is not None and pair_valid.any() else None,
    }
    if target_mask is not None and target_mask.numel() == support_count.numel():
        target = target_mask.bool()
        out.update(
            {
                "target_count": int(target.sum().detach().cpu().item()),
                "target_support_nonzero_ratio": _as_float(((support_count > 0) & target).float().sum() / target.float().sum().clamp_min(1.0)),
                "target_support_multi_ratio": _as_float((multi & target).float().sum() / target.float().sum().clamp_min(1.0)),
                "target_agreement_all_mean_cos": _masked_weighted_mean(all_mean_cos[:, target], weights[:, target]) if target.any() else None,
                "target_agreement_loo_cos": _masked_weighted_mean(
                    loo_cos[:, target],
                    torch.where(loo_valid[:, target], weights[:, target], torch.zeros_like(weights[:, target])),
                )
                if target.any()
                else None,
                "target_agreement_mean_norm": _as_float(mean_norm[multi & target]) if (multi & target).any() else None,
                "target_agreement_pair_cos": _as_float(pair_cos[pair_valid & target])
                if pair_cos is not None and pair_valid is not None and (pair_valid & target).any()
                else None,
            }
        )
    return out


def _ray_diversity(extrinsics: torch.Tensor, extrinsics_are_c2w: bool, center_world: torch.Tensor) -> dict:
    c2w = extrinsics.float() if extrinsics_are_c2w else torch.linalg.inv(extrinsics.float())
    centers = c2w[:, :3, 3]
    if centers.shape[0] == 0:
        return {}
    center_world = center_world.to(device=centers.device, dtype=torch.float32)
    to_object = F.normalize(center_world[None] - centers, dim=-1, eps=1e-6)
    distances = torch.linalg.norm(center_world[None] - centers, dim=-1)
    out = {
        "camera_distance_mean": _as_float(distances),
        "camera_distance_std": float(distances.std(unbiased=False).detach().cpu().item()) if distances.numel() > 1 else 0.0,
        "camera_center_std_mean": _as_float(centers.std(dim=0, unbiased=False)),
    }
    if centers.shape[0] > 1:
        pair_center = torch.pdist(centers)
        cos = torch.matmul(to_object, to_object.T).clamp(-1.0, 1.0)
        tri = torch.triu_indices(cos.shape[0], cos.shape[1], offset=1, device=cos.device)
        cos_vals = cos[tri[0], tri[1]]
        angles = torch.rad2deg(torch.acos(cos_vals.clamp(-1.0, 1.0)))
        out.update(
            {
                "camera_baseline_mean": _as_float(pair_center),
                "camera_baseline_max": float(pair_center.max().detach().cpu().item()) if pair_center.numel() else 0.0,
                "ray_pair_cos_mean": _as_float(cos_vals),
                "ray_pair_angle_mean_deg": _as_float(angles),
                "ray_pair_angle_max_deg": float(angles.max().detach().cpu().item()) if angles.numel() else 0.0,
            }
        )
    return out


def _voxel_ray_diversity(
    points_obj: torch.Tensor,
    object_to_world: torch.Tensor,
    extrinsics: torch.Tensor,
    extrinsics_are_c2w: bool,
    support_weights: torch.Tensor,
    target_mask: Optional[torch.Tensor],
) -> dict:
    c2w = extrinsics.float() if extrinsics_are_c2w else torch.linalg.inv(extrinsics.float())
    centers = c2w[:, :3, 3].float()
    if centers.shape[0] <= 1 or points_obj.numel() == 0:
        return {}
    object_to_world = object_to_world.to(device=points_obj.device, dtype=torch.float32)
    points_h = torch.cat([points_obj.float(), torch.ones((points_obj.shape[0], 1), device=points_obj.device)], dim=1)
    points_world = (points_h @ object_to_world.T)[:, :3]
    dirs = F.normalize(points_world[None] - centers[:, None], dim=-1, eps=1e-6)
    cos = torch.einsum("vnc,wnc->vwn", dirs, dirs).clamp(-1.0, 1.0)
    angles = torch.rad2deg(torch.acos(cos))
    pair_weights = support_weights.float().clamp_min(0.0)[:, None, :] * support_weights.float().clamp_min(0.0)[None, :, :]
    eye = torch.eye(centers.shape[0], device=points_obj.device, dtype=torch.bool)[:, :, None]
    pair_weights = torch.where(eye, torch.zeros_like(pair_weights), pair_weights)
    pair_denom = pair_weights.sum(dim=(0, 1))
    pair_valid = pair_denom > 1e-8
    voxel_angle = (angles * pair_weights).sum(dim=(0, 1)) / pair_denom.clamp_min(1e-6)
    voxel_baseline = torch.cdist(centers, centers)
    voxel_baseline = (voxel_baseline[:, :, None] * pair_weights).sum(dim=(0, 1)) / pair_denom.clamp_min(1e-6)
    out = {
        "voxel_ray_pair_valid_ratio": _as_float(pair_valid.float()),
        "voxel_ray_pair_angle_mean_deg": _as_float(voxel_angle[pair_valid]) if pair_valid.any() else None,
        "voxel_ray_pair_angle_median_deg": float(voxel_angle[pair_valid].median().cpu().item()) if pair_valid.any() else None,
        "voxel_ray_pair_angle_p95_deg": float(torch.quantile(voxel_angle[pair_valid], 0.95).cpu().item()) if pair_valid.any() else None,
        "voxel_camera_baseline_mean": _as_float(voxel_baseline[pair_valid]) if pair_valid.any() else None,
    }
    out.update(_split_voxel_stats(voxel_angle, target_mask, "voxel_ray_pair_angle_deg", pair_valid))
    out.update(_split_voxel_stats(voxel_baseline, target_mask, "voxel_camera_baseline", pair_valid))
    return out


def _load_view_aggregator_from_checkpoint(args: argparse.Namespace, image_cond_model, device: torch.device):
    if args.view_aggregator == "none":
        return None, {"enabled": False}
    view_aggregator = build_view_aggregator(args, image_cond_model, device)
    if view_aggregator is None:
        return None, {"enabled": False}
    info = {"enabled": True, "loaded": False, "checkpoint": args.checkpoint}
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location="cpu")
        if "view_aggregator" not in state:
            raise ValueError(f"checkpoint has no view_aggregator weights: {args.checkpoint}")
        missing, unexpected = view_aggregator.load_state_dict(state["view_aggregator"], strict=False)
        info.update({"loaded": True, "missing": len(missing), "unexpected": len(unexpected)})
    view_aggregator.eval()
    return view_aggregator, info


def _make_pose_batch(dataset, batch: dict, sample_index: int, pose_mode: str, seed: int) -> dict:
    if pose_mode == "correct":
        return batch
    if pose_mode == "cross_sample":
        other = dataset[deterministic_other_index(sample_index, len(dataset), seed)]
        return make_cross_sample_batch(batch, other)
    return apply_pose_mode(batch, pose_mode, seed)


def _summarize(rows: list[dict], metrics: list[str]) -> list[dict]:
    modes = sorted({row["pose_mode"] for row in rows})
    out = []
    for mode in modes:
        selected = [row for row in rows if row["pose_mode"] == mode]
        row_out = {"pose_mode": mode, "count": len(selected)}
        for metric in metrics:
            values = [row.get(metric) for row in selected]
            values = [float(v) for v in values if v is not None and np.isfinite(float(v))]
            if values:
                arr = np.asarray(values, dtype=np.float64)
                row_out[f"{metric}_mean"] = float(arr.mean())
                row_out[f"{metric}_median"] = float(np.median(arr))
            else:
                row_out[f"{metric}_mean"] = None
                row_out[f"{metric}_median"] = None
        out.append(row_out)
    return out


def _pairwise(rows: list[dict], metrics: list[str], reference: str = "correct") -> list[dict]:
    by_mode = {
        mode: {int(row["sample_index"]): row for row in rows if row["pose_mode"] == mode}
        for mode in sorted({row["pose_mode"] for row in rows})
    }
    ref = by_mode.get(reference, {})
    out = []
    for mode, selected in by_mode.items():
        if mode == reference:
            continue
        common = sorted(set(ref) & set(selected))
        for metric in metrics:
            deltas = []
            for idx in common:
                a = ref[idx].get(metric)
                b = selected[idx].get(metric)
                if a is None or b is None:
                    continue
                if not np.isfinite(float(a)) or not np.isfinite(float(b)):
                    continue
                deltas.append(float(a) - float(b))
            if not deltas:
                continue
            arr = np.asarray(deltas, dtype=np.float64)
            out.append(
                {
                    "reference_pose": reference,
                    "wrong_pose": mode,
                    "metric": metric,
                    "count": int(arr.size),
                    "mean_delta": float(arr.mean()),
                    "median_delta": float(np.median(arr)),
                    "reference_wins": int((arr > 0).sum()),
                    "reference_win_rate": float((arr > 0).mean()),
                }
            )
    return out


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys = sorted({key for row in rows for key in row.keys()})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in keys})


def _fmt(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _write_report(path: Path, result: dict) -> None:
    summary = result["summary"]
    pairwise = result["pairwise"]
    lines = [
        "# Pose Condition Diagnostics",
        "",
        f"时间：`{result['timestamp_utc']}`",
        f"manifest：`{result['manifest']}`",
        f"indices：`{result['indices']}`",
        f"pose_modes：`{result['pose_modes']}`",
        f"head：`{result.get('head_checkpoint') or ''}`",
        f"view aggregator checkpoint：`{result.get('checkpoint') or ''}`",
        "",
        "## 1. Support / Agreement 汇总",
        "",
        "| pose | count | support nz | support multi | all-mean cos | LOO cos | pair cos | global ray | voxel ray | baseline |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            "| {pose} | {count} | {support} | {multi} | {all_mean} | {loo} | {pair} | {ray} | {voxel_ray} | {baseline} |".format(
                pose=row["pose_mode"],
                count=row["count"],
                support=_fmt(row.get("support_nonzero_ratio_mean")),
                multi=_fmt(row.get("support_multi_ratio_mean")),
                all_mean=_fmt(row.get("agreement_all_mean_cos_mean")),
                loo=_fmt(row.get("agreement_loo_cos_mean")),
                pair=_fmt(row.get("agreement_pair_cos_mean")),
                ray=_fmt(row.get("ray_pair_angle_mean_deg_mean")),
                voxel_ray=_fmt(row.get("voxel_ray_pair_angle_mean_deg_mean")),
                baseline=_fmt(row.get("camera_baseline_mean_mean")),
            )
        )

    lines.extend(
        [
            "",
            "## 2. Logits / Attention 汇总",
            "",
            "| pose | head logit | head gate | old entropy | learned entropy | prior entropy | old->learned L1 | old->prior L1 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary:
        lines.append(
            "| {pose} | {head_logit} | {head_gate} | {old_entropy} | {learned_entropy} | {prior_entropy} | {l1a} | {l1b} |".format(
                pose=row["pose_mode"],
                head_logit=_fmt(row.get("head_logit_mean_mean")),
                head_gate=_fmt(row.get("head_gate_mean_mean")),
                old_entropy=_fmt(row.get("old_attention_entropy_mean_mean")),
                learned_entropy=_fmt(row.get("learned_attention_entropy_mean_mean")),
                prior_entropy=_fmt(row.get("prior_attention_entropy_mean_mean")),
                l1a=_fmt(row.get("old_vs_learned_attn_l1_mean_mean")),
                l1b=_fmt(row.get("old_vs_prior_attn_l1_mean_mean")),
            )
        )

    lines.extend(
        [
            "",
            "## 3. Target vs Non-target",
            "",
            "| pose | target head logit | non-target head logit | logit gap | target prior entropy | non-target prior entropy | target voxel ray | non-target voxel ray |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary:
        lines.append(
            "| {pose} | {th} | {nh} | {gap} | {te} | {ne} | {tr} | {nr} |".format(
                pose=row["pose_mode"],
                th=_fmt(row.get("target_head_logit_mean_mean")),
                nh=_fmt(row.get("non_target_head_logit_mean_mean")),
                gap=_fmt(row.get("head_logit_target_minus_non_target_mean")),
                te=_fmt(row.get("target_prior_attention_entropy_mean_mean")),
                ne=_fmt(row.get("non_target_prior_attention_entropy_mean_mean")),
                tr=_fmt(row.get("target_voxel_ray_pair_angle_deg_mean_mean")),
                nr=_fmt(row.get("non_target_voxel_ray_pair_angle_deg_mean_mean")),
            )
        )

    lines.extend(
        [
            "",
            "## 4. Correct-vs-Wrong 关键差值",
            "",
            "| wrong pose | metric | mean delta | median delta | correct wins |",
            "|---|---|---:|---:|---:|",
        ]
    )
    keep_metrics = {
        "support_nonzero_ratio",
        "agreement_all_mean_cos",
        "agreement_loo_cos",
        "head_logit_mean",
        "head_gate_mean",
        "head_logit_target_minus_non_target",
        "prior_attention_entropy_target_minus_non_target",
        "voxel_ray_pair_angle_deg_target_minus_non_target",
        "ray_pair_angle_mean_deg",
        "voxel_ray_pair_angle_mean_deg",
        "old_vs_prior_attn_l1_mean",
    }
    for row in pairwise:
        if row["metric"] not in keep_metrics:
            continue
        lines.append(
            f"| {row['wrong_pose']} | {row['metric']} | {row['mean_delta']:.6f} | "
            f"{row['median_delta']:.6f} | {row['reference_wins']}/{row['count']} |"
        )

    lines.extend(
        [
            "",
            "## 5. 如何阅读",
            "",
            "- `old attention` 是原始 support weight 归一化后的 view 权重。",
            "- `learned attention` 是只用 `ViewGatedAggregator` 自己的 gate logits 后的 view 权重。",
            "- `prior attention` 是 `gate_logits + alpha * pose_consistency_logits` 后的 view 权重。",
            "- `all-mean cos` 会包含当前 view 自己，容易被自一致性抬高。",
            "- `LOO cos` 是 leave-one-out agreement：当前 view 只和其它支持 view 的均值比较，更能暴露伪一致。",
            "- `voxel ray` 是每个 voxel 实际有 support 的视角之间的 ray angle，不同于全局相机轨迹跨度。",
            "- `target vs non-target` 用 sparse target coords 切分，检查 head/attention 是否真的更偏向目标结构。",
            "- 如果 identity 的 support/agreement/head logit 很高，但 ray angle/baseline 很低，说明高分主要来自重复或集中投影的自一致，而不是正确多视角几何。",
            "- 如果 prior attention 相对 old attention 的 L1 很大，但 sparse sampling 没变好，说明 head 改变了 view 选择，但改变方向没有和 sparse geometry 对齐。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose pose condition self-consistency, attention, logits, and view diversity without training.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--indices", default="0-63")
    parser.add_argument("--pose_modes", default="correct,cyclic_shift1,cyclic_shift2,reverse,noise,large_noise,cross_sample,identity")
    parser.add_argument("--head_checkpoint", default="")
    parser.add_argument("--checkpoint", default="", help="Sparse-training checkpoint containing view_aggregator weights.")
    parser.add_argument("--image_cond_model", default=IMAGE_COND_CONFIG["model_name"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_frames", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1234)
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
    parser.add_argument("--pose_consistency_alpha", type=float, default=0.5)
    parser.add_argument("--view_aggregator", choices=["none", "gated"], default="gated")
    parser.add_argument("--view_aggregator_geom_dim", type=int, default=11)
    parser.add_argument("--view_aggregator_reduced_dim", type=int, default=128)
    parser.add_argument("--view_aggregator_hidden_dim", type=int, default=256)
    parser.add_argument("--view_aggregator_dropout", type=float, default=0.0)
    parser.add_argument("--view_aggregator_residual_scale", type=float, default=1.0)
    parser.add_argument(
        "--view_aggregator_geom_mode",
        choices=["full", "no_xyz", "uv_depth_only", "support_only"],
        default="full",
    )
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = MultiviewSparseManifestDataset(args.manifest, max_frames=args.max_frames, apply_mask=not args.no_apply_mask)
    indices = parse_indices(args.indices, len(dataset))
    pose_modes = parse_modes(args.pose_modes, allow_correct=True)
    image_cond_model = build_image_cond_model(device, args.ss_grid_resolution, args.image_cond_model)
    feature_dim = int(getattr(image_cond_model, "embed_dim", image_cond_model.model.config.hidden_size))
    head = None
    if args.head_checkpoint:
        head = load_pose_consistency_head(args.head_checkpoint, feature_dim=feature_dim, device=device).eval()
    view_aggregator, view_aggregator_info = _load_view_aggregator_from_checkpoint(args, image_cond_model, device)

    rows: list[dict] = []
    for sample_index in tqdm(indices, desc="Pose condition diagnostics", unit="sample", dynamic_ncols=True):
        batch = dataset[sample_index]
        patch_features, feature_info = extract_patch_features(image_cond_model, batch["images"], device)
        target_mask, target_count, target_source_resolution = load_target_mask(batch["latent_path"], args.ss_grid_resolution, device)
        points_obj, _ = pixal3d_grid_points(args.ss_grid_resolution, device=device, dtype=torch.float32)

        for pose_mode in pose_modes:
            cond_batch = _make_pose_batch(dataset, batch, sample_index, pose_mode, args.seed + sample_index * 104729)
            masks = cond_batch["masks"].to(device=device, dtype=torch.float32)
            intrinsics = cond_batch["intrinsics"].to(device=device, dtype=torch.float32)
            extrinsics = cond_batch["extrinsics"].to(device=device, dtype=torch.float32)
            extrinsics_are_c2w = str(cond_batch["extrinsics_type"]).lower() == "c2w"

            object_to_world = None
            volume_extent = None
            volume_stats = {}
            if not args.no_auto_volume:
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
                volume_stats = {
                    "volume_fallback": bool(volume.fallback),
                    "volume_extent_world": float(volume.extent_world),
                    "volume_occupied_ratio": float(volume.occupied_ratio),
                    "volume_support_mean": float(volume.support_mean),
                    "volume_visible_mean": float(volume.visible_mean),
                }

            image_size = int(image_cond_model.image_size)
            intrinsics_sq = scale_intrinsics_to_square(intrinsics, cond_batch["source_sizes"], image_size, device)
            masks_sq = resize_masks(masks, image_size, device)
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
            elif volume_extent is not None:
                depth_tolerance = max(volume_extent * float(args.visibility_depth_tolerance_ratio), 1e-4)
            else:
                depth_tolerance = 0.03

            front_depth_maps = None
            front_depth_stats = {"enabled": False}
            if (not args.no_visibility_depth) and object_to_world is not None:
                front_depth_maps, front_depth_stats = visual_hull_front_depth_maps(
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

            sampled, support_weights, view_geom, view_token_stats = sample_view_features_for_aggregation(
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

            valid = support_weights > 0
            old_attn = support_weights.float() / support_weights.sum(dim=0, keepdim=True).clamp_min(1e-6)
            agreement = _feature_agreement(sampled, support_weights, target_mask)
            row = {
                "sample_index": int(sample_index),
                "uid": batch["uid"],
                "pose_mode": pose_mode,
                "pose_permutation": cond_batch.get("pose_permutation"),
                "cross_sample_uid": cond_batch.get("cross_sample_uid"),
                "target_count": int(target_count),
                "target_source_resolution": int(target_source_resolution),
                "depth_tolerance": float(depth_tolerance),
                "view_token_in_image_ratio": view_token_stats.get("in_image_ratio"),
                "view_token_mask_hit_ratio": view_token_stats.get("mask_hit_ratio"),
                "view_token_support_weight_mean": view_token_stats.get("support_weight_mean"),
                "view_token_support_weight_nonzero_ratio": view_token_stats.get("support_weight_nonzero_ratio"),
                "front_depth_finite_ratio": front_depth_stats.get("finite_ratio_after_dilation"),
                **volume_stats,
                **agreement,
                **_attention_stats(old_attn, valid, "old_attention", target_mask),
            }

            if object_to_world is not None:
                row.update(_ray_diversity(extrinsics, extrinsics_are_c2w, object_to_world[:3, 3]))
                row.update(
                    _voxel_ray_diversity(
                        points_obj,
                        object_to_world,
                        extrinsics,
                        extrinsics_are_c2w,
                        support_weights,
                        target_mask,
                    )
                )

            head_logits = None
            if head is not None:
                _, head_stats, head_tensors = head(sampled.detach(), support_weights.detach(), view_geom.detach())
                head_logits = head_tensors["logits"].detach().float()
                gate = head_tensors["gate"].detach().float()
                row.update(
                    {
                        "head_sample_score": float(head_tensors["sample_score"].detach().cpu().item()),
                        "head_keep_ratio": float(head_tensors["keep_ratio"].detach().cpu().item()),
                        "head_gate_mean": head_stats.get("gate_mean"),
                        "head_gate_min": head_stats.get("gate_min"),
                        "head_gate_max": head_stats.get("gate_max"),
                        "head_valid_ratio": head_stats.get("valid_ratio"),
                    }
                )
                row.update(_tensor_stats(head_logits[valid], "head_logit"))
                row.update(_tensor_stats(gate[valid], "head_gate_valid"))
                row.update(_split_view_weighted_stats(head_logits, support_weights, target_mask, "head_logit"))
                row.update(_split_view_weighted_stats(gate, support_weights, target_mask, "head_gate"))
                row.update(
                    _split_voxel_stats(
                        head_tensors["voxel_score"].detach().float(),
                        target_mask,
                        "head_voxel_score",
                        valid.any(dim=0),
                    )
                )

            if view_aggregator is not None:
                features = sampled.float()
                geom = view_aggregator._filter_geom(view_geom)
                reduced = view_aggregator.feature_reduce(features)
                gate_input = torch.cat([reduced, geom], dim=-1)
                learned_logits = view_aggregator.gate(gate_input).squeeze(-1).detach().float()
                learned_attn = _softmax_attention(learned_logits, valid)
                row.update(_tensor_stats(learned_logits[valid], "learned_logit"))
                row.update(_attention_stats(learned_attn, valid, "learned_attention", target_mask))
                row.update(_attention_compare(old_attn, learned_attn, valid, "old_vs_learned", target_mask))
                if head_logits is not None:
                    prior_logits = learned_logits + float(args.pose_consistency_alpha) * head_logits
                    prior_attn = _softmax_attention(prior_logits, valid)
                    row.update(_tensor_stats(prior_logits[valid], "prior_logit"))
                    row.update(_attention_stats(prior_attn, valid, "prior_attention", target_mask))
                    row.update(_attention_compare(old_attn, prior_attn, valid, "old_vs_prior", target_mask))
                    row.update(_attention_compare(learned_attn, prior_attn, valid, "learned_vs_prior", target_mask))

            rows.append(row)

    metrics = [
        "support_nonzero_ratio",
        "support_multi_ratio",
        "support_count_mean",
        "support_weight_sum_mean",
        "agreement_all_mean_cos",
        "agreement_loo_cos",
        "agreement_pair_cos",
        "agreement_mean_norm",
        "target_support_nonzero_ratio",
        "target_support_multi_ratio",
        "target_agreement_all_mean_cos",
        "target_agreement_loo_cos",
        "target_agreement_pair_cos",
        "target_agreement_mean_norm",
        "head_sample_score",
        "head_keep_ratio",
        "head_logit_mean",
        "head_gate_mean",
        "target_head_logit_mean",
        "non_target_head_logit_mean",
        "head_logit_target_minus_non_target",
        "target_head_gate_mean",
        "non_target_head_gate_mean",
        "head_gate_target_minus_non_target",
        "target_head_voxel_score_mean",
        "non_target_head_voxel_score_mean",
        "head_voxel_score_target_minus_non_target",
        "old_attention_entropy_mean",
        "learned_attention_entropy_mean",
        "prior_attention_entropy_mean",
        "target_old_attention_entropy_mean",
        "non_target_old_attention_entropy_mean",
        "old_attention_entropy_target_minus_non_target",
        "target_learned_attention_entropy_mean",
        "non_target_learned_attention_entropy_mean",
        "learned_attention_entropy_target_minus_non_target",
        "target_prior_attention_entropy_mean",
        "non_target_prior_attention_entropy_mean",
        "prior_attention_entropy_target_minus_non_target",
        "old_vs_learned_attn_l1_mean",
        "old_vs_prior_attn_l1_mean",
        "learned_vs_prior_attn_l1_mean",
        "target_old_vs_prior_attn_l1_mean",
        "non_target_old_vs_prior_attn_l1_mean",
        "old_vs_prior_attn_l1_target_minus_non_target",
        "old_vs_learned_top_view_agree",
        "old_vs_prior_top_view_agree",
        "target_old_vs_prior_top_view_agree_mean",
        "non_target_old_vs_prior_top_view_agree_mean",
        "old_vs_prior_top_view_agree_target_minus_non_target",
        "camera_distance_mean",
        "camera_distance_std",
        "camera_baseline_mean",
        "camera_baseline_max",
        "ray_pair_cos_mean",
        "ray_pair_angle_mean_deg",
        "ray_pair_angle_max_deg",
        "voxel_ray_pair_valid_ratio",
        "voxel_ray_pair_angle_mean_deg",
        "voxel_ray_pair_angle_median_deg",
        "voxel_ray_pair_angle_p95_deg",
        "target_voxel_ray_pair_angle_deg_mean",
        "non_target_voxel_ray_pair_angle_deg_mean",
        "voxel_ray_pair_angle_deg_target_minus_non_target",
        "voxel_camera_baseline_mean",
        "target_voxel_camera_baseline_mean",
        "non_target_voxel_camera_baseline_mean",
        "voxel_camera_baseline_target_minus_non_target",
        "volume_extent_world",
        "volume_occupied_ratio",
        "view_token_support_weight_nonzero_ratio",
    ]
    summary = _summarize(rows, metrics)
    pairwise = _pairwise(rows, metrics, reference="correct")
    result = {
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "manifest": args.manifest,
        "indices": indices,
        "pose_modes": pose_modes,
        "head_checkpoint": args.head_checkpoint,
        "checkpoint": args.checkpoint,
        "pose_consistency_alpha": args.pose_consistency_alpha,
        "view_aggregator": view_aggregator_info,
        "feature_info": feature_info,
        "summary": summary,
        "pairwise": pairwise,
        "rows": rows,
    }
    (output_dir / "diagnostics_summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_csv(output_dir / "diagnostics_rows.csv", rows)
    _write_csv(output_dir / "diagnostics_summary_by_pose.csv", summary)
    _write_csv(output_dir / "diagnostics_pairwise_vs_correct.csv", pairwise)
    _write_report(output_dir / "diagnostics_report.md", result)
    print(json.dumps({"output_dir": str(output_dir), "summary": summary}, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
