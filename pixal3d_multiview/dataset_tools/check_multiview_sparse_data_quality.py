#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from PIL import Image

TRACKER_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TRACKER_ROOT))

from pixal3d_multiview.dataset_tools.build_objaverse_multiview_sparse_data import (  # noqa: E402
    projection_support_stats,
)
from pixal3d_multiview.multiview_projection import estimate_object_volume_from_visual_hull  # noqa: E402


def load_json(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve(root: Optional[str], path: Optional[str]) -> Optional[str]:
    if path is None:
        return None
    if os.path.isabs(path) or not root:
        return path
    return os.path.join(root, path)


def scalar_stats(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None, "p05": None, "p95": None}
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "p05": float(np.quantile(arr, 0.05)),
        "p95": float(np.quantile(arr, 0.95)),
    }


def parse_indices(spec: str, total: int, max_samples: int, seed: int, random_sample: bool) -> list[int]:
    if spec:
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
        bad = [idx for idx in out if idx < 0 or idx >= total]
        if bad:
            raise IndexError(f"indices out of range for total={total}: {bad}")
        return out
    indices = list(range(total))
    if random_sample:
        rng = random.Random(seed)
        rng.shuffle(indices)
    return indices[: min(max_samples, total)]


def load_latent_stats(latent_path: str) -> tuple[dict, np.ndarray]:
    with np.load(latent_path) as data:
        z = data["z"] if "z" in data else None
        coords = data["target_coords"].astype(np.int32) if "target_coords" in data else np.zeros((0, 3), dtype=np.int32)
    stats = {
        "latent_path": latent_path,
        "z_shape": list(z.shape) if z is not None else None,
        "z_finite": bool(np.isfinite(z).all()) if z is not None else False,
        "target_coords": int(coords.shape[0]),
    }
    if coords.size:
        stats["target_bbox_min"] = coords.min(axis=0).tolist()
        stats["target_bbox_max"] = coords.max(axis=0).tolist()
    return stats, coords


def mask_and_image_stats(
    sample: dict,
    image_root: Optional[str],
    mask_root: Optional[str],
    max_frames: int,
    mask_threshold: float,
) -> tuple[dict, list[np.ndarray]]:
    fg_pixels = []
    area_ratios = []
    bbox_margins = []
    bbox_area_ratios = []
    outside_nonzero_ratios = []
    alphas = []

    frames = sample.get("frames", [])
    if max_frames > 0:
        frames = frames[:max_frames]
    for frame in frames:
        mask_path = resolve(mask_root, frame.get("mask", sample.get("mask")))
        if mask_path is None:
            raise ValueError(f"sample {sample.get('uid')} has no mask")
        mask = np.asarray(Image.open(mask_path).convert("L"))
        alpha = mask.astype(np.uint8)
        alphas.append(alpha)
        fg = alpha > int(mask_threshold * 255.0)
        h, w = fg.shape
        fg_count = int(fg.sum())
        fg_pixels.append(fg_count)
        area_ratios.append(float(fg.mean()))
        if fg_count > 0:
            ys, xs = np.where(fg)
            bbox_margins.append(int(min(xs.min(), ys.min(), w - 1 - xs.max(), h - 1 - ys.max())))
            bbox_area_ratios.append(float(((xs.max() - xs.min() + 1) * (ys.max() - ys.min() + 1)) / float(h * w)))
        else:
            bbox_margins.append(0)
            bbox_area_ratios.append(0.0)

        image_path = resolve(image_root, frame.get("image"))
        if image_path is not None and os.path.exists(image_path):
            image = np.asarray(Image.open(image_path).convert("RGB"))
            outside = image[~fg]
            outside_nonzero_ratios.append(float((outside > 2).any(axis=-1).mean()) if outside.size else 0.0)

    stats = {
        "num_frames": len(frames),
        "fg_pixels_total": int(sum(fg_pixels)),
        "fg_pixels_per_view_min": int(min(fg_pixels)) if fg_pixels else 0,
        "fg_pixels_per_view_median": float(np.median(fg_pixels)) if fg_pixels else 0.0,
        "fg_pixels_per_view_max": int(max(fg_pixels)) if fg_pixels else 0,
        "fg_area_ratio_mean": float(np.mean(area_ratios)) if area_ratios else 0.0,
        "fg_area_ratio_min": float(np.min(area_ratios)) if area_ratios else 0.0,
        "bbox_margin_px_min": int(min(bbox_margins)) if bbox_margins else 0,
        "bbox_margin_px_median": float(np.median(bbox_margins)) if bbox_margins else 0.0,
        "bbox_area_ratio_mean": float(np.mean(bbox_area_ratios)) if bbox_area_ratios else 0.0,
        "bbox_area_ratio_max": float(np.max(bbox_area_ratios)) if bbox_area_ratios else 0.0,
        "outside_mask_nonzero_ratio_mean": float(np.mean(outside_nonzero_ratios)) if outside_nonzero_ratios else None,
    }
    return stats, alphas


def recompute_projection_stats(sample: dict, coords: np.ndarray, alphas: list[np.ndarray], max_frames: int, mask_threshold: float) -> dict:
    frames = sample.get("frames", [])
    if max_frames > 0:
        frames = frames[:max_frames]
    c2w = np.asarray([frame["extrinsic"] for frame in frames], dtype=np.float32)
    intrinsic = np.asarray(frames[0]["intrinsic"], dtype=np.float32)
    return projection_support_stats(coords, c2w, intrinsic, alphas, 64, mask_threshold)


def visual_hull_stats(sample: dict, alphas: list[np.ndarray], max_frames: int, args: argparse.Namespace) -> dict:
    frames = sample.get("frames", [])
    if max_frames > 0:
        frames = frames[:max_frames]
    masks = torch.from_numpy(np.stack([(alpha > int(args.mask_threshold * 255.0)).astype(np.float32) for alpha in alphas], axis=0))
    intrinsics = torch.tensor([frame["intrinsic"] for frame in frames], dtype=torch.float32)
    extrinsics = torch.tensor([frame["extrinsic"] for frame in frames], dtype=torch.float32)
    estimate = estimate_object_volume_from_visual_hull(
        masks,
        intrinsics,
        extrinsics,
        extrinsics_are_c2w=True,
        camera_forward_sign=args.camera_forward_sign,
        mask_threshold=0.5,
        resolution=args.vh_resolution,
        min_visible_views=args.vh_min_visible_views,
        min_support_views=args.vh_min_support_views,
        min_support_ratio=args.vh_min_support_ratio,
        initial_extent_ratio=args.vh_volume_initial_extent_ratio,
        padding=args.vh_volume_padding,
        min_extent=args.vh_volume_min_extent,
        refine_steps=args.vh_volume_refine_steps,
    )
    return estimate.to_dict()


def summarize_rows(rows: list[dict]) -> dict:
    metrics: dict[str, list[float]] = {
        "target_coords": [],
        "fg_area_ratio_mean": [],
        "fg_pixels_per_view_min": [],
        "bbox_margin_px_min": [],
        "manifest_projection_support_ratio_mean": [],
        "manifest_projection_zero_support_ratio": [],
        "train_projection_support_ratio_mean": [],
        "train_projection_zero_support_ratio": [],
        "outside_mask_nonzero_ratio_mean": [],
        "visual_hull_occupied_ratio": [],
        "visual_hull_extent_world": [],
    }
    issue_counts = {
        "low_texture": 0,
        "flat_gray_blob": 0,
        "low_projection_support_flag": 0,
        "empty_or_tiny_target": 0,
        "bad_latent": 0,
        "tiny_mask": 0,
        "border_touch": 0,
        "train_low_projection_support": 0,
        "outside_mask_not_black": 0,
        "visual_hull_empty": 0,
    }

    for row in rows:
        latent = row["latent_stats"]
        mask = row["mask_stats"]
        flags = row.get("quality_flags", {})
        manifest_proj = row.get("manifest_projection_stats", {})
        train_proj = row.get("train_projection_stats", {})
        vh = row.get("visual_hull_stats", {})

        metrics["target_coords"].append(float(latent["target_coords"]))
        metrics["fg_area_ratio_mean"].append(float(mask["fg_area_ratio_mean"]))
        metrics["fg_pixels_per_view_min"].append(float(mask["fg_pixels_per_view_min"]))
        metrics["bbox_margin_px_min"].append(float(mask["bbox_margin_px_min"]))
        if mask.get("outside_mask_nonzero_ratio_mean") is not None:
            metrics["outside_mask_nonzero_ratio_mean"].append(float(mask["outside_mask_nonzero_ratio_mean"]))
        for key, src_name in (
            ("manifest_projection_support_ratio_mean", "support_ratio_mean"),
            ("manifest_projection_zero_support_ratio", "zero_support_ratio"),
        ):
            if src_name in manifest_proj:
                metrics[key].append(float(manifest_proj[src_name]))
        for key, src_name in (
            ("train_projection_support_ratio_mean", "support_ratio_mean"),
            ("train_projection_zero_support_ratio", "zero_support_ratio"),
        ):
            if src_name in train_proj:
                metrics[key].append(float(train_proj[src_name]))
        if "occupied_ratio" in vh:
            metrics["visual_hull_occupied_ratio"].append(float(vh["occupied_ratio"]))
        if "extent_world" in vh:
            metrics["visual_hull_extent_world"].append(float(vh["extent_world"]))

        issue_counts["low_texture"] += int(bool(flags.get("low_texture", False)))
        issue_counts["flat_gray_blob"] += int(bool(flags.get("flat_gray_blob", False)))
        issue_counts["low_projection_support_flag"] += int(bool(flags.get("low_projection_support", False)))
        issue_counts["empty_or_tiny_target"] += int(latent["target_coords"] < 256)
        issue_counts["bad_latent"] += int(not latent["z_finite"])
        issue_counts["tiny_mask"] += int(mask["fg_pixels_per_view_min"] < 512 or mask["fg_area_ratio_mean"] < 0.004)
        issue_counts["border_touch"] += int(mask["bbox_margin_px_min"] <= 2)
        issue_counts["train_low_projection_support"] += int(
            train_proj.get("support_ratio_mean", 1.0) < 0.5 or train_proj.get("zero_support_ratio", 0.0) > 0.25
        )
        outside = mask.get("outside_mask_nonzero_ratio_mean")
        issue_counts["outside_mask_not_black"] += int(outside is not None and outside > 0.001)
        issue_counts["visual_hull_empty"] += int(vh.get("occupied_voxels", 1) <= 0)

    return {
        "metrics": {key: scalar_stats(values) for key, values in metrics.items()},
        "issue_counts": issue_counts,
    }


def write_markdown(path: Path, payload: dict) -> None:
    summary = payload["summary"]
    metrics = summary["metrics"]
    issues = summary["issue_counts"]
    lines = [
        "# 多视图稀疏数据质量检查",
        "",
        f"- manifest: `{payload['manifest']}`",
        f"- checked_samples: `{payload['checked_samples']}` / total `{payload['total_samples']}`",
        f"- max_frames: `{payload['max_frames']}`",
        "",
        "## 核心统计",
        "",
        "| 指标 | mean | median | min | max |",
        "|---|---:|---:|---:|---:|",
    ]
    for key in (
        "target_coords",
        "fg_area_ratio_mean",
        "fg_pixels_per_view_min",
        "bbox_margin_px_min",
        "manifest_projection_support_ratio_mean",
        "manifest_projection_zero_support_ratio",
        "train_projection_support_ratio_mean",
        "train_projection_zero_support_ratio",
        "outside_mask_nonzero_ratio_mean",
        "visual_hull_occupied_ratio",
        "visual_hull_extent_world",
    ):
        stat = metrics[key]
        if stat["count"] == 0:
            continue
        lines.append(
            f"| `{key}` | {stat['mean']:.6g} | {stat['median']:.6g} | {stat['min']:.6g} | {stat['max']:.6g} |"
        )
    lines += [
        "",
        "## 异常计数",
        "",
    ]
    for key, value in issues.items():
        lines.append(f"- `{key}`: {value}")
    lines += [
        "",
        "## 说明",
        "",
        "- `manifest_projection_*` 是数据构建时记录的投影支持度，通常使用完整视图。",
        "- `train_projection_*` 是本脚本按 `max_frames` 重算的投影支持度，更接近当前训练输入。",
        "- `outside_mask_nonzero_ratio_mean` 越接近 0，说明 masked RGB 与训练/推理一致性越好。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check pixal3d_multiview sparse dataset quality.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--indices", default="")
    parser.add_argument("--max_samples", type=int, default=64)
    parser.add_argument("--random_sample", action="store_true")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max_frames", type=int, default=8)
    parser.add_argument("--mask_threshold", type=float, default=0.5)
    parser.add_argument("--compute_visual_hull", action="store_true")
    parser.add_argument("--vh_resolution", type=int, default=32)
    parser.add_argument("--vh_min_visible_views", type=int, default=1)
    parser.add_argument("--vh_min_support_views", type=int, default=2)
    parser.add_argument("--vh_min_support_ratio", type=float, default=0.6)
    parser.add_argument("--vh_volume_initial_extent_ratio", type=float, default=0.6)
    parser.add_argument("--vh_volume_padding", type=float, default=1.25)
    parser.add_argument("--vh_volume_min_extent", type=float, default=0.05)
    parser.add_argument("--vh_volume_refine_steps", type=int, default=2)
    parser.add_argument("--camera_forward_sign", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = load_json(args.manifest)
    samples = manifest.get("samples", manifest if isinstance(manifest, list) else None)
    if samples is None:
        raise ValueError("manifest should contain a samples list")
    image_root = manifest.get("image_root")
    mask_root = manifest.get("mask_root")
    latent_root = manifest.get("latent_root")

    indices = parse_indices(args.indices, len(samples), args.max_samples, args.seed, args.random_sample)
    rows = []
    failures = []
    for idx in indices:
        sample = samples[idx]
        try:
            latent_path = resolve(latent_root, sample.get("ss_latent", sample.get("ss_latent_path", sample.get("latent"))))
            if latent_path is None:
                raise ValueError("missing sparse latent path")
            latent_stats, target_coords = load_latent_stats(latent_path)
            mask_stats, alphas = mask_and_image_stats(sample, image_root, mask_root, args.max_frames, args.mask_threshold)
            train_projection = recompute_projection_stats(sample, target_coords, alphas, args.max_frames, args.mask_threshold)
            row: dict[str, Any] = {
                "index": idx,
                "uid": sample.get("uid", str(idx)),
                "latent_stats": latent_stats,
                "mask_stats": mask_stats,
                "manifest_projection_stats": sample.get("projection_stats", {}),
                "train_projection_stats": train_projection,
                "quality_flags": sample.get("quality_flags", {}),
                "render_stats": sample.get("render_stats", {}),
                "camera_trajectory_stats": sample.get("camera_trajectory_stats", {}),
            }
            if args.compute_visual_hull:
                row["visual_hull_stats"] = visual_hull_stats(sample, alphas, args.max_frames, args)
            rows.append(row)
        except Exception as exc:
            failures.append({"index": idx, "uid": sample.get("uid", str(idx)), "error": repr(exc)})

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "manifest": args.manifest,
        "total_samples": len(samples),
        "checked_samples": len(rows),
        "failed_samples": len(failures),
        "indices": indices,
        "max_frames": args.max_frames,
        "quality_counts": manifest.get("quality_counts", {}),
        "failure_counts": manifest.get("failure_counts", {}),
        "summary": summarize_rows(rows),
        "rows": rows,
        "failures": failures,
    }
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path = output.with_suffix(".md")
    write_markdown(md_path, payload)
    print(json.dumps({"output": str(output), "markdown": str(md_path), "summary": payload["summary"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
