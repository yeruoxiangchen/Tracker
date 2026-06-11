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
from PIL import Image
from tqdm.auto import tqdm

TRACKER_ROOT = Path(__file__).resolve().parents[1]
PIXAL3D_ROOT = TRACKER_ROOT / "Pixal3D"
sys.path.insert(0, str(TRACKER_ROOT))
sys.path.insert(0, str(PIXAL3D_ROOT))

from pixal3d_multiview.multiview_projection import (  # noqa: E402
    estimate_object_volume_from_visual_hull,
    pixal3d_grid_points,
    project_points_multi_view,
    resize_masks,
    scale_intrinsics_to_square,
    visual_hull_front_depth_maps,
)
from pixal3d_multiview.train_sparse_multiview import (  # noqa: E402
    MultiviewSparseManifestDataset,
    POSE_MODES,
    apply_pose_mode,
    build_image_cond_model,
)


def parse_indices(spec: str, total: int) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            out.extend(range(int(start_s), int(end_s) + 1))
        else:
            out.append(int(part))
    if not out:
        out = [0]
    bad = [idx for idx in out if idx < 0 or idx >= total]
    if bad:
        raise IndexError(f"indices out of range for dataset size={total}: {bad}")
    return out


def parse_pose_modes(spec: str) -> list[str]:
    modes = [part.strip() for part in spec.split(",") if part.strip()]
    valid = set(POSE_MODES)
    bad = [mode for mode in modes if mode not in valid]
    if bad:
        raise ValueError(f"Unknown pose modes: {bad}")
    return modes or ["correct"]


def images_to_tensor(image_cond_model, images: list[Image.Image], device: torch.device) -> torch.Tensor:
    tensors = []
    for image in images:
        image = image.resize((image_cond_model.image_size, image_cond_model.image_size), Image.LANCZOS)
        arr = np.asarray(image.convert("RGB")).astype(np.float32) / 255.0
        tensors.append(torch.from_numpy(arr).permute(2, 0, 1).float())
    return torch.stack(tensors, dim=0).to(device)


@torch.no_grad()
def extract_patch_features(image_cond_model, images: list[Image.Image], device: torch.device) -> tuple[torch.Tensor, dict]:
    images_tensor = images_to_tensor(image_cond_model, images, device)
    dino_input = image_cond_model.transform(images_tensor)
    z = image_cond_model.extract_features(dino_input)
    num_reg = int(getattr(image_cond_model.model.config, "num_register_tokens", 4))
    z_patchtokens = z[:, 1 + num_reg :]
    z_patchtokens_spatial = z_patchtokens.reshape(
        images_tensor.shape[0],
        image_cond_model.patch_number,
        image_cond_model.patch_number,
        -1,
    )
    return z_patchtokens_spatial, {
        "num_views": int(images_tensor.shape[0]),
        "image_size": int(image_cond_model.image_size),
        "patch_number": int(image_cond_model.patch_number),
        "feature_dim": int(z_patchtokens_spatial.shape[-1]),
    }


def sample_per_view_features(
    feature_map: torch.Tensor,
    points_2d: torch.Tensor,
    valid_depth: torch.Tensor,
    *,
    coordinate_size: int,
    masks: Optional[torch.Tensor],
    mask_threshold: float,
    depths: Optional[torch.Tensor],
    front_depth_maps: Optional[torch.Tensor],
    visibility_depth_tolerance: float,
    visibility_weight_min: float,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    if feature_map.ndim == 4 and feature_map.shape[1] < feature_map.shape[-1]:
        feature_map = feature_map.permute(0, 3, 1, 2)
    view_count = int(feature_map.shape[0])
    if points_2d.shape[0] != view_count or valid_depth.shape[0] != view_count:
        raise ValueError("feature, point, and valid-depth view counts do not match")

    u = points_2d[..., 0]
    v = points_2d[..., 1]
    in_image = valid_depth & (u >= 0) & (u < coordinate_size) & (v >= 0) & (v < coordinate_size)
    grid = torch.stack(
        [(u + 0.5) / float(coordinate_size) * 2.0 - 1.0, (v + 0.5) / float(coordinate_size) * 2.0 - 1.0],
        dim=-1,
    ).view(view_count, -1, 1, 2)

    sampled = F.grid_sample(feature_map, grid, mode="bilinear", align_corners=False, padding_mode="border")
    sampled = sampled.squeeze(-1).permute(0, 2, 1)

    if masks is not None:
        masks = resize_masks(masks, coordinate_size, feature_map.device)
        mask_values = F.grid_sample(masks, grid, mode="bilinear", align_corners=False, padding_mode="zeros")
        mask_values = mask_values.squeeze(1).squeeze(-1).clamp(0.0, 1.0)
        mask_hit = in_image & (mask_values > float(mask_threshold))
    else:
        mask_values = in_image.float()
        mask_hit = in_image

    weights = mask_hit.float()
    visibility_stats = {"enabled": False}
    if front_depth_maps is not None:
        if depths is None:
            raise ValueError("depths is required when front_depth_maps is provided")
        if front_depth_maps.ndim == 3:
            front_depth_maps = front_depth_maps[:, None]
        if front_depth_maps.ndim != 4 or front_depth_maps.shape[1] != 1:
            raise ValueError(f"front_depth_maps should be [V,H,W] or [V,1,H,W], got {tuple(front_depth_maps.shape)}")
        if front_depth_maps.shape[-2:] != (coordinate_size, coordinate_size):
            front_depth_maps = F.interpolate(
                front_depth_maps.float(),
                size=(coordinate_size, coordinate_size),
                mode="nearest",
            )
        front_depth_maps = front_depth_maps.to(device=feature_map.device, dtype=torch.float32)
        finite_maps = torch.isfinite(front_depth_maps) & (front_depth_maps > 0)
        front_depth_for_sample = torch.where(finite_maps, front_depth_maps, torch.zeros_like(front_depth_maps))
        front_depth = F.grid_sample(
            front_depth_for_sample,
            grid,
            mode="nearest",
            align_corners=False,
            padding_mode="zeros",
        ).squeeze(1).squeeze(-1)
        front_finite = F.grid_sample(
            finite_maps.float(),
            grid,
            mode="nearest",
            align_corners=False,
            padding_mode="zeros",
        ).squeeze(1).squeeze(-1) > 0.5
        delta = (depths.to(feature_map.device, dtype=torch.float32) - front_depth).abs()
        tol = max(float(visibility_depth_tolerance), 1e-6)
        visibility_weight = (1.0 - delta / tol).clamp_min(0.0)
        visibility_weight = torch.where(front_finite, visibility_weight, torch.zeros_like(visibility_weight))
        visibility_weight = torch.where(
            visibility_weight >= float(visibility_weight_min),
            visibility_weight,
            torch.zeros_like(visibility_weight),
        )
        weights = weights * visibility_weight
        visibility_stats = {
            "enabled": True,
            "depth_tolerance": tol,
            "weight_min": float(visibility_weight_min),
            "front_depth_coverage": float(finite_maps.float().mean().item()) if finite_maps.numel() else 0.0,
            "visibility_weight_mean": float(visibility_weight.mean().item()) if visibility_weight.numel() else 0.0,
            "visibility_weight_nonzero_ratio": float((visibility_weight > 0).float().mean().item()) if visibility_weight.numel() else 0.0,
        }

    stats = {
        "view_count": view_count,
        "num_points": int(points_2d.shape[1]),
        "in_image_ratio": float(in_image.float().mean().item()) if in_image.numel() else 0.0,
        "mask_hit_ratio": float(mask_hit.float().mean().item()) if mask_hit.numel() else 0.0,
        "mask_value_mean": float(mask_values.mean().item()) if mask_values.numel() else 0.0,
        "weight_mean": float(weights.mean().item()) if weights.numel() else 0.0,
        "weight_nonzero_ratio": float((weights > 0).float().mean().item()) if weights.numel() else 0.0,
        "visibility": visibility_stats,
    }
    return sampled, weights, stats


def load_target_mask(latent_path: str, resolution: int, device: torch.device) -> tuple[torch.Tensor, int, int]:
    target_mask = torch.zeros((resolution**3,), device=device, dtype=torch.bool)
    with np.load(latent_path) as data:
        if "target_coords" not in data:
            return target_mask, 0, resolution
        coords = data["target_coords"].astype(np.int64)
    if coords.size == 0:
        return target_mask, 0, resolution
    xyz = coords[:, -3:]
    valid = np.all(xyz >= 0, axis=1)
    xyz = xyz[valid]
    if xyz.size == 0:
        return target_mask, 0, resolution
    source_resolution = max(int(xyz.max()) + 1, int(resolution))
    if source_resolution != int(resolution):
        xyz = np.floor((xyz.astype(np.float32) + 0.5) * float(resolution) / float(source_resolution)).astype(np.int64)
    xyz = np.clip(xyz, 0, int(resolution) - 1)
    flat = xyz[:, 0] * resolution * resolution + xyz[:, 1] * resolution + xyz[:, 2]
    target_mask[torch.from_numpy(flat).to(device=device, dtype=torch.long)] = True
    return target_mask, int(target_mask.sum().item()), source_resolution


def mean_or_none(values: torch.Tensor) -> Optional[float]:
    if values.numel() == 0:
        return None
    return float(values.float().mean().item())


def compute_consistency_metrics(
    sampled: torch.Tensor,
    weights: torch.Tensor,
    *,
    min_consistency_views: int,
    target_mask: Optional[torch.Tensor] = None,
) -> dict:
    features = F.normalize(sampled.float(), dim=-1, eps=1e-6)
    support_count = (weights > 0).sum(dim=0)
    weight_sum = weights.sum(dim=0)
    valid_multi = support_count >= int(min_consistency_views)

    normalized_weights = weights / weight_sum[None].clamp_min(1e-6)
    mean_feature = (features * normalized_weights[..., None]).sum(dim=0)
    mean_norm = torch.linalg.norm(mean_feature, dim=-1)
    pair_cos = None
    pair_valid = None
    if sampled.shape[0] >= 2:
        gram = torch.einsum("vnc,wnc->vwn", features, features)
        pair_weights = weights[:, None, :] * weights[None, :, :]
        eye = torch.eye(sampled.shape[0], device=sampled.device, dtype=torch.bool)[:, :, None]
        pair_weights = torch.where(eye, torch.zeros_like(pair_weights), pair_weights)
        pair_denom = pair_weights.sum(dim=(0, 1))
        pair_valid = pair_denom > 0
        pair_cos = (gram * pair_weights).sum(dim=(0, 1)) / pair_denom.clamp_min(1e-6)

    valid_features = mean_feature[valid_multi]
    result = {
        "support_count_mean": mean_or_none(support_count.float()),
        "support_count_nonzero_ratio": mean_or_none((support_count > 0).float()),
        "multi_support_voxel_ratio": mean_or_none(valid_multi.float()),
        "weight_sum_mean": mean_or_none(weight_sum),
        "mean_norm_mean": mean_or_none(mean_norm[valid_multi]),
        "mean_norm_median": float(mean_norm[valid_multi].median().item()) if valid_multi.any() else None,
        "feature_variance_proxy_mean": mean_or_none(1.0 - mean_norm[valid_multi]),
        "agg_feature_abs_mean": mean_or_none(valid_features.abs()) if valid_features.numel() else None,
        "agg_feature_channel_std_mean": mean_or_none(valid_features.std(dim=0, unbiased=False)) if valid_features.shape[0] > 1 else None,
    }
    if pair_cos is not None and pair_valid is not None:
        result["pair_cos_mean"] = mean_or_none(pair_cos[pair_valid])
        result["pair_cos_median"] = float(pair_cos[pair_valid].median().item()) if pair_valid.any() else None
        result["pair_valid_voxel_ratio"] = mean_or_none(pair_valid.float())

    if target_mask is not None:
        target_multi = valid_multi & target_mask
        result["target_count"] = int(target_mask.sum().item())
        result["target_supported_ratio"] = (
            float(((support_count > 0) & target_mask).float().sum().item() / target_mask.float().sum().clamp_min(1.0).item())
            if target_mask.numel()
            else 0.0
        )
        result["target_multi_support_ratio"] = (
            float(target_multi.float().sum().item() / target_mask.float().sum().clamp_min(1.0).item())
            if target_mask.numel()
            else 0.0
        )
        result["target_mean_norm_mean"] = mean_or_none(mean_norm[target_multi])
        if pair_cos is not None and pair_valid is not None:
            result["target_pair_cos_mean"] = mean_or_none(pair_cos[pair_valid & target_mask])
    return result


def numeric_summary(rows: list[dict], metric_keys: list[str]) -> list[dict]:
    by_pose: dict[str, list[dict]] = {}
    for row in rows:
        by_pose.setdefault(row["pose_mode"], []).append(row)
    summary_rows = []
    for pose_mode, selected in sorted(by_pose.items()):
        out = {"pose_mode": pose_mode, "count": len(selected)}
        for key in metric_keys:
            values = [float(row[key]) for row in selected if row.get(key) is not None]
            if values:
                arr = np.asarray(values, dtype=np.float64)
                out[f"{key}_mean"] = float(arr.mean())
                out[f"{key}_median"] = float(np.median(arr))
            else:
                out[f"{key}_mean"] = None
                out[f"{key}_median"] = None
        summary_rows.append(out)
    return summary_rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys = sorted({key for row in rows for key in row.keys()})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in keys})


def format_number(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def write_summary_markdown(path: Path, result: dict) -> None:
    lines = [
        "# Per-view Feature Consistency Pose Test",
        "",
        f"时间：`{result['timestamp_utc']}`",
        f"消融名称：`{result.get('ablation_name', '')}`",
        f"manifest：`{result['manifest']}`",
        f"indices：`{result['indices']}`",
        f"pose_modes：`{result['pose_modes']}`",
        f"grid_resolution：`{result['ss_grid_resolution']}`",
        f"min_consistency_views：`{result['min_consistency_views']}`",
        "",
        "## 汇总",
        "",
        "| pose | count | multi support | pair cos | mean norm | target multi | target pair cos | support count |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["summary"]:
        lines.append(
            "| {pose_mode} | {count} | {multi} | {pair} | {mean_norm} | {target_multi} | {target_pair} | {support} |".format(
                pose_mode=row["pose_mode"],
                count=row["count"],
                multi=format_number(row.get("multi_support_voxel_ratio_mean")),
                pair=format_number(row.get("pair_cos_mean_mean")),
                mean_norm=format_number(row.get("mean_norm_mean_mean")),
                target_multi=format_number(row.get("target_multi_support_ratio_mean")),
                target_pair=format_number(row.get("target_pair_cos_mean_mean")),
                support=format_number(row.get("support_count_mean_mean")),
            )
        )

    correct = next((row for row in result["summary"] if row["pose_mode"] == "correct"), None)
    if correct:
        lines.extend(["", "## Correct 差值", "", "| pose | pair cos gap | mean norm gap | target pair gap |", "|---|---:|---:|---:|"])
        for row in result["summary"]:
            if row["pose_mode"] == "correct":
                continue
            lines.append(
                "| {pose} | {pair} | {mean_norm} | {target_pair} |".format(
                    pose=row["pose_mode"],
                    pair=format_number(
                        (correct.get("pair_cos_mean_mean") or 0.0) - (row.get("pair_cos_mean_mean") or 0.0)
                        if correct.get("pair_cos_mean_mean") is not None and row.get("pair_cos_mean_mean") is not None
                        else None
                    ),
                    mean_norm=format_number(
                        (correct.get("mean_norm_mean_mean") or 0.0) - (row.get("mean_norm_mean_mean") or 0.0)
                        if correct.get("mean_norm_mean_mean") is not None and row.get("mean_norm_mean_mean") is not None
                        else None
                    ),
                    target_pair=format_number(
                        (correct.get("target_pair_cos_mean_mean") or 0.0) - (row.get("target_pair_cos_mean_mean") or 0.0)
                        if correct.get("target_pair_cos_mean_mean") is not None and row.get("target_pair_cos_mean_mean") is not None
                        else None
                    ),
                )
            )

    lines.extend(
        [
            "",
            "## 如何阅读",
            "",
            "- 这个测试不训练 sparse flow，只换 pose，检查同一个 3D grid point 从多视角反投影采到的 DINO patch feature 是否互相一致。",
            "- `pair cos` 是支持该点的视角之间的加权两两余弦相似度；越高表示多视角采到的语义/纹理特征越一致。",
            "- `mean norm` 是归一化 per-view feature 加权平均后的向量长度；越接近 1，表示不同视角 feature 越一致。",
            "- `multi support` 表示至少被 `min_consistency_views` 个视角支持的 grid 点比例。",
            "- `target_*` 只在 sparse target coords 上统计；如果 correct 在这里也不能明显高于 shuffle/noise，说明当前特征一致性仍不足以提供强 pose 约束。",
            "- 如果 correct 明显高于 shuffle/noise，下一步应把该 consistency/support score 接入 sparse condition，而不是只依赖 pooled projection feature。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate whether projected per-view DINO features distinguish correct and corrupted poses.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--indices", default="0,1,5,10,20,30,50,80,100")
    parser.add_argument("--pose_modes", default="correct,shuffle,reverse,noise,large_noise,identity")
    parser.add_argument("--image_cond_model", default="/home/zjr/Tracker/models/dinov3-vitl16-pretrain-lvd1689m")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_frames", type=int, default=8)
    parser.add_argument("--ss_grid_resolution", type=int, default=16)
    parser.add_argument("--min_consistency_views", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1234)
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
    parser.add_argument("--ablation_name", default="")
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = MultiviewSparseManifestDataset(
        args.manifest,
        max_frames=args.max_frames,
        apply_mask=not args.no_apply_mask,
    )
    indices = parse_indices(args.indices, len(dataset))
    pose_modes = parse_pose_modes(args.pose_modes)
    image_cond_model = build_image_cond_model(device, args.ss_grid_resolution, args.image_cond_model)

    rows: list[dict] = []
    feature_info = None
    for sample_index in tqdm(indices, desc="View consistency", unit="sample", dynamic_ncols=True):
        batch = dataset[sample_index]
        patch_features, feature_info = extract_patch_features(image_cond_model, batch["images"], device)
        target_mask, target_count, target_source_resolution = load_target_mask(batch["latent_path"], args.ss_grid_resolution, device)

        for pose_mode in pose_modes:
            cond_batch = apply_pose_mode(batch, pose_mode, args.seed + sample_index * 104729)
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
                }

            image_size = int(image_cond_model.image_size)
            intrinsics_sq = scale_intrinsics_to_square(intrinsics, cond_batch["source_sizes"], image_size, device)
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

            sampled, weights, sample_stats = sample_per_view_features(
                patch_features,
                points_2d,
                valid_depth,
                coordinate_size=image_size,
                masks=masks_sq,
                mask_threshold=args.mask_threshold,
                depths=depths,
                front_depth_maps=front_depth_maps,
                visibility_depth_tolerance=depth_tolerance,
                visibility_weight_min=args.visibility_weight_min,
            )
            metrics = compute_consistency_metrics(
                sampled,
                weights,
                min_consistency_views=args.min_consistency_views,
                target_mask=target_mask,
            )
            row = {
                "sample_index": int(sample_index),
                "uid": batch["uid"],
                "pose_mode": pose_mode,
                "pose_permutation": cond_batch.get("pose_permutation"),
                "target_count": int(target_count),
                "target_source_resolution": int(target_source_resolution),
                "depth_tolerance": float(depth_tolerance),
                "in_image_ratio": sample_stats["in_image_ratio"],
                "mask_hit_ratio": sample_stats["mask_hit_ratio"],
                "mask_value_mean": sample_stats["mask_value_mean"],
                "weight_mean": sample_stats["weight_mean"],
                "weight_nonzero_ratio": sample_stats["weight_nonzero_ratio"],
                "front_depth_finite_ratio": front_depth_stats.get("finite_ratio_after_dilation"),
                **volume_stats,
                **metrics,
            }
            visibility = sample_stats.get("visibility", {})
            for key, value in visibility.items():
                if isinstance(value, (int, float, bool)):
                    row[f"visibility_{key}"] = value
            rows.append(row)

    metric_keys = [
        "support_count_mean",
        "support_count_nonzero_ratio",
        "multi_support_voxel_ratio",
        "weight_sum_mean",
        "mean_norm_mean",
        "feature_variance_proxy_mean",
        "pair_cos_mean",
        "pair_valid_voxel_ratio",
        "target_supported_ratio",
        "target_multi_support_ratio",
        "target_mean_norm_mean",
        "target_pair_cos_mean",
        "in_image_ratio",
        "mask_hit_ratio",
        "weight_nonzero_ratio",
        "visibility_weight_nonzero_ratio",
        "front_depth_finite_ratio",
        "volume_extent_world",
        "volume_occupied_ratio",
    ]
    summary = numeric_summary(rows, metric_keys)
    result = {
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "ablation_name": args.ablation_name,
        "manifest": args.manifest,
        "indices": indices,
        "pose_modes": pose_modes,
        "max_frames": args.max_frames,
        "ss_grid_resolution": args.ss_grid_resolution,
        "min_consistency_views": args.min_consistency_views,
        "feature_info": feature_info,
        "summary": summary,
        "rows": rows,
    }
    (output_dir / "summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(output_dir / "metrics.csv", rows)
    write_csv(output_dir / "summary.csv", summary)
    write_summary_markdown(output_dir / "summary.md", result)
    print(json.dumps({"output_dir": str(output_dir), "summary": summary}, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
