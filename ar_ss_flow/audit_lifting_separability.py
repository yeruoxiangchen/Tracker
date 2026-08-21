#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, median
import sys
from typing import Any

import torch
import torch.nn.functional as F

TRACKER_ROOT = Path(__file__).resolve().parents[1]
if str(TRACKER_ROOT) not in sys.path:
    sys.path.insert(0, str(TRACKER_ROOT))

from ar_ss_flow.local_pose_lifting_flow import PoseLiftingCacheDataset, volume_from_sample


CORRUPTION_MODES = ("pose_perturb", "pose_shuffle", "depth_corrupt")


def target_region_masks(target_coords: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    coords = target_coords.long()[..., -3:]
    valid = ((coords >= 0) & (coords < 64)).all(dim=1)
    coords16 = torch.div(coords[valid], 4, rounding_mode="floor").clamp(0, 15)
    occupied = torch.zeros((1, 1, 16, 16, 16), device=coords.device)
    if coords16.numel():
        occupied[0, 0, coords16[:, 0], coords16[:, 1], coords16[:, 2]] = 1.0
    near = F.max_pool3d(occupied, kernel_size=3, stride=1, padding=1).bool()
    return occupied.bool()[0, 0], (~near)[0, 0]


def region_support(metadata: torch.Tensor, mask: torch.Tensor) -> float:
    support = (
        metadata[0, 0].float()
        if metadata.ndim == 5
        else metadata[0].float()
    )
    if not bool(mask.any().item()):
        return 0.0
    return float(support[mask].mean().item())


def comparison(
    correct_volume: torch.Tensor,
    correct_metadata: torch.Tensor,
    corrupt_volume: torch.Tensor,
    corrupt_metadata: torch.Tensor,
    correct_stats: dict[str, Any],
    corrupt_stats: dict[str, Any],
    target_mask: torch.Tensor,
    far_mask: torch.Tensor,
) -> dict[str, float]:
    correct_flat = correct_volume.float().flatten()
    corrupt_flat = corrupt_volume.float().flatten()
    energy = correct_flat.square().mean().clamp_min(1.0e-8)
    normalized_mse = (correct_flat - corrupt_flat).square().mean() / energy
    cosine = F.cosine_similarity(correct_flat[None], corrupt_flat[None]).clamp(-1.0, 1.0).item()
    correct_support = correct_metadata[:, 0] > 0
    corrupt_support = corrupt_metadata[:, 0] > 0
    union = (correct_support | corrupt_support).sum().clamp_min(1)
    support_iou = (correct_support & corrupt_support).sum().float() / union.float()
    metadata_mse = (
        (correct_metadata.float() - corrupt_metadata.float()).square().mean()
    )
    correct_target_support = region_support(correct_metadata, target_mask)
    corrupt_target_support = region_support(corrupt_metadata, target_mask)
    correct_far_support = region_support(correct_metadata, far_mask)
    corrupt_far_support = region_support(corrupt_metadata, far_mask)
    correct_variance = correct_stats.get("cross_view_weighted_variance")
    corrupt_variance = corrupt_stats.get("cross_view_weighted_variance")
    correct_consistency = correct_stats.get("cross_view_cosine_consistency")
    corrupt_consistency = corrupt_stats.get("cross_view_cosine_consistency")
    return {
        "visual_normalized_mse": float(normalized_mse.item()),
        "visual_cosine": float(cosine),
        "metadata_mse": float(metadata_mse.item()),
        "support_iou": float(support_iou.item()),
        "correct_cross_view_weighted_variance": correct_variance,
        "corrupt_cross_view_weighted_variance": corrupt_variance,
        "correct_cross_view_cosine_consistency": correct_consistency,
        "corrupt_cross_view_cosine_consistency": corrupt_consistency,
        "cross_view_variance_direction": (
            None
            if correct_variance is None or corrupt_variance is None
            else float(corrupt_variance - correct_variance)
        ),
        "cross_view_cosine_direction": (
            None
            if correct_consistency is None or corrupt_consistency is None
            else float(correct_consistency - corrupt_consistency)
        ),
        "correct_target_support": correct_target_support,
        "corrupt_target_support": corrupt_target_support,
        "target_support_direction": correct_target_support - corrupt_target_support,
        "correct_far_support": correct_far_support,
        "corrupt_far_support": corrupt_far_support,
        "far_support_direction": corrupt_far_support - correct_far_support,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="P2 audit: rebuild paired correct/corrupted volumes from identical patches."
    )
    parser.add_argument("--cache_manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--indices", default="all")
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--min_visual_normalized_mse", type=float, default=1.0e-4)
    parser.add_argument("--min_sample_pass_ratio", type=float, default=0.80)
    parser.add_argument("--min_direction_pass_ratio", type=float, default=0.60)
    parser.add_argument("--fail_on_error", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = PoseLiftingCacheDataset(args.cache_manifest, indices=args.indices)
    count = len(dataset) if args.max_samples <= 0 else min(len(dataset), args.max_samples)
    device = torch.device(args.device)
    rows: list[dict[str, Any]] = []
    for index in range(count):
        sample = dataset[index]
        correct_volume, correct_metadata, correct_stats = volume_from_sample(
            sample, device=device, mode="correct", compute_cross_view_metrics=True
        )
        target_mask, far_mask = target_region_masks(
            sample["target_coords"].to(device=device)
        )
        mode_rows: dict[str, Any] = {}
        for mode in CORRUPTION_MODES:
            corrupt_volume, corrupt_metadata, corrupt_stats = volume_from_sample(
                sample, device=device, mode=mode, compute_cross_view_metrics=True
            )
            metrics = comparison(
                correct_volume,
                correct_metadata,
                corrupt_volume,
                corrupt_metadata,
                correct_stats,
                corrupt_stats,
                target_mask,
                far_mask,
            )
            metrics["corrupt_supported_voxel_ratio"] = corrupt_stats[
                "supported_voxel_ratio"
            ]
            metrics["non_identity_passed"] = metrics["visual_normalized_mse"] >= float(
                args.min_visual_normalized_mse
            )
            cross_view_directions = (
                metrics["cross_view_variance_direction"],
                metrics["cross_view_cosine_direction"],
            )
            metrics["cross_view_direction_passed"] = any(
                value is not None and value > 0.0 for value in cross_view_directions
            )
            metrics["gt_local_support_direction_passed"] = (
                metrics["target_support_direction"] > 0.0
                and metrics["far_support_direction"] >= 0.0
            )
            metrics["passed"] = bool(
                metrics["non_identity_passed"]
                and metrics["cross_view_direction_passed"]
            )
            mode_rows[mode] = metrics
        null_volume = torch.zeros_like(correct_volume)
        null_metadata = torch.zeros_like(correct_metadata)
        row = {
            "uid": sample["uid"],
            "object_uid": sample["object_uid"],
            "same_visual_feature_tensor_reused": True,
            "correct": correct_stats,
            "corruptions": mode_rows,
            "null_volume_abs_max": float(null_volume.abs().max().item()),
            "null_metadata_abs_max": float(null_metadata.abs().max().item()),
            "passed": all(values["passed"] for values in mode_rows.values()),
        }
        rows.append(row)
        print(
            f"[lifting_p2] {index + 1}/{count} uid={sample['uid']} "
            + " ".join(
                f"{mode}={mode_rows[mode]['visual_normalized_mse']:.6g}"
                for mode in CORRUPTION_MODES
            ),
            flush=True,
        )
    summaries: dict[str, Any] = {}
    for mode in CORRUPTION_MODES:
        values = [row["corruptions"][mode] for row in rows]
        variance_directions = [
            item["cross_view_variance_direction"]
            for item in values
            if item["cross_view_variance_direction"] is not None
        ]
        cosine_directions = [
            item["cross_view_cosine_direction"]
            for item in values
            if item["cross_view_cosine_direction"] is not None
        ]
        summaries[mode] = {
            "visual_normalized_mse_mean": mean(
                item["visual_normalized_mse"] for item in values
            ),
            "visual_normalized_mse_median": median(
                item["visual_normalized_mse"] for item in values
            ),
            "visual_cosine_mean": mean(item["visual_cosine"] for item in values),
            "support_iou_mean": mean(item["support_iou"] for item in values),
            "sample_pass_ratio": mean(float(item["passed"]) for item in values),
            "non_identity_pass_ratio": mean(
                float(item["non_identity_passed"]) for item in values
            ),
            "cross_view_direction_pass_ratio": mean(
                float(item["cross_view_direction_passed"]) for item in values
            ),
            "gt_local_support_direction_pass_ratio": mean(
                float(item["gt_local_support_direction_passed"]) for item in values
            ),
            "cross_view_variance_direction_mean": (
                mean(variance_directions) if variance_directions else None
            ),
            "cross_view_cosine_direction_mean": (
                mean(cosine_directions) if cosine_directions else None
            ),
            "target_support_direction_mean": mean(
                item["target_support_direction"] for item in values
            ),
            "far_support_direction_mean": mean(
                item["far_support_direction"] for item in values
            ),
        }
    checks = {
        "paired_same_visual_features": all(
            row["same_visual_feature_tensor_reused"] for row in rows
        ),
        "null_is_exact_zero": all(
            row["null_volume_abs_max"] == 0.0
            and row["null_metadata_abs_max"] == 0.0
            for row in rows
        ),
        "all_corruptions_separable": all(
            summaries[mode]["non_identity_pass_ratio"] >= float(args.min_sample_pass_ratio)
            for mode in CORRUPTION_MODES
        ),
        "correct_geometry_has_directional_advantage": all(
            summaries[mode]["cross_view_direction_pass_ratio"]
            >= float(args.min_direction_pass_ratio)
            for mode in CORRUPTION_MODES
        ),
    }
    report = {
        "passed": all(checks.values()),
        "checks": checks,
        "thresholds": {
            "min_visual_normalized_mse": float(args.min_visual_normalized_mse),
            "min_sample_pass_ratio": float(args.min_sample_pass_ratio),
            "min_direction_pass_ratio": float(args.min_direction_pass_ratio),
        },
        "sample_count": len(rows),
        "summary": summaries,
        "samples": rows,
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    markdown = [
        "# P2 Lifting Separability Audit",
        "",
        f"- passed: `{report['passed']}`",
        f"- samples: `{len(rows)}`",
        "- paired policy: every corruption reuses the exact same cached VGGT/DINO patches and only rebuilds geometry/depth sampling.",
        "",
        "| corruption | normalized MSE mean | cosine mean | nonidentity pass | cross-view direction pass | GT support direction pass |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for mode in CORRUPTION_MODES:
        item = summaries[mode]
        markdown.append(
            f"| {mode} | {item['visual_normalized_mse_mean']:.6g} | "
            f"{item['visual_cosine_mean']:.6g} | "
            f"{item['non_identity_pass_ratio']:.6g} | "
            f"{item['cross_view_direction_pass_ratio']:.6g} | "
            f"{item['gt_local_support_direction_pass_ratio']:.6g} |"
        )
    (output_dir / "report.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "summary": summaries}, indent=2))
    if args.fail_on_error and not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
