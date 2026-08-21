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

from ar_ss_flow.local_pose_lifting_flow import (  # noqa: E402
    PoseLiftingCacheDataset,
    volume_from_sample,
)
from ar_ss_flow.object_frame import (  # noqa: E402
    OBJECT_FRAME_VERSION,
    deterministic_similarity,
    estimate_point_pca_similarity,
    rotation_error_degrees,
    scale_depth_calibration,
    similarity_scale,
    transform_camera_extrinsics,
    transform_points,
)
from ar_ss_flow.pose_lifting import LIFTING_VOLUME_VERSION  # noqa: E402
from trellis_point_prior_mv.common import (  # noqa: E402
    apply_grid_transform,
    coords_to_points,
)


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "mean": mean(values) if values else 0.0,
        "median": median(values) if values else 0.0,
        "min": min(values) if values else 0.0,
        "max": max(values) if values else 0.0,
    }


def volume_comparison(
    baseline_volume: torch.Tensor,
    baseline_metadata: torch.Tensor,
    candidate_volume: torch.Tensor,
    candidate_metadata: torch.Tensor,
) -> dict[str, float]:
    baseline_flat = baseline_volume.float().flatten()
    candidate_flat = candidate_volume.float().flatten()
    energy = baseline_flat.square().mean().clamp_min(1.0e-8)
    normalized_mse = (
        (candidate_flat - baseline_flat).square().mean() / energy
    )
    cosine = F.cosine_similarity(
        baseline_flat[None], candidate_flat[None]
    ).clamp(-1.0, 1.0)
    baseline_support = baseline_metadata[:, 0:1] > 0
    candidate_support = candidate_metadata[:, 0:1] > 0
    union = (baseline_support | candidate_support).sum().clamp_min(1)
    support_iou = (
        (baseline_support & candidate_support).sum().float() / union.float()
    )
    return {
        "visual_normalized_mse": float(normalized_mse.item()),
        "visual_cosine": float(cosine.item()),
        "visual_max_abs_diff": float(
            (candidate_volume.float() - baseline_volume.float()).abs().max().item()
        ),
        "metadata_max_abs_diff": float(
            (candidate_metadata.float() - baseline_metadata.float()).abs().max().item()
        ),
        "support_iou": float(support_iou.item()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="P0.5: audit object-frame necessity with paired synthetic Sim(3)."
    )
    parser.add_argument("--cache_manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--indices", default="all")
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expected_canonical_extent", type=float, default=0.9)
    parser.add_argument("--oracle_max_normalized_mse", type=float, default=1.0e-6)
    parser.add_argument("--min_no_recovery_mse", type=float, default=1.0e-3)
    parser.add_argument("--min_no_recovery_detection_ratio", type=float, default=0.80)
    parser.add_argument("--min_estimator_improvement_ratio", type=float, default=0.60)
    parser.add_argument("--max_estimator_center_error_over_scale", type=float, default=0.20)
    parser.add_argument("--max_estimator_scale_log_error", type=float, default=0.25)
    parser.add_argument("--fail_on_error", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    dataset = PoseLiftingCacheDataset(args.cache_manifest, indices=args.indices)
    count = len(dataset) if int(args.max_samples) <= 0 else min(
        len(dataset), int(args.max_samples)
    )
    rows: list[dict[str, Any]] = []
    for index in range(count):
        sample = dataset[index]
        baseline_volume, baseline_metadata, _ = volume_from_sample(
            sample, device=device, mode="correct"
        )
        true_transform = deterministic_similarity(index, int(args.seed)).to(device)
        transformed_extrinsics = transform_camera_extrinsics(
            sample["extrinsics"].to(device),
            true_transform,
            extrinsics_type=str(sample["extrinsics_type"]),
        )
        true_scale = similarity_scale(true_transform)
        transformed_calibration = scale_depth_calibration(
            sample["depth_calibration"], true_scale
        )
        canonical_prior = torch.from_numpy(
            apply_grid_transform(
                coords_to_points(sample["prior_coords"].cpu().numpy(), 64),
                str(sample["grid_transform"]),
            )
        ).to(device=device, dtype=torch.float32)
        world_prior = transform_points(canonical_prior, true_transform)
        estimated_transform = estimate_point_pca_similarity(
            world_prior,
            expected_canonical_extent=float(args.expected_canonical_extent),
        ).to(device)

        oracle_volume, oracle_metadata, _ = volume_from_sample(
            sample,
            device=device,
            mode="correct",
            extrinsics_override=transformed_extrinsics,
            calibration_override=transformed_calibration,
            object_to_world=true_transform,
        )
        no_recovery_volume, no_recovery_metadata, _ = volume_from_sample(
            sample,
            device=device,
            mode="correct",
            extrinsics_override=transformed_extrinsics,
            calibration_override=transformed_calibration,
        )
        estimated_volume, estimated_metadata, _ = volume_from_sample(
            sample,
            device=device,
            mode="correct",
            extrinsics_override=transformed_extrinsics,
            calibration_override=transformed_calibration,
            object_to_world=estimated_transform,
        )
        oracle = volume_comparison(
            baseline_volume,
            baseline_metadata,
            oracle_volume,
            oracle_metadata,
        )
        no_recovery = volume_comparison(
            baseline_volume,
            baseline_metadata,
            no_recovery_volume,
            no_recovery_metadata,
        )
        estimated = volume_comparison(
            baseline_volume,
            baseline_metadata,
            estimated_volume,
            estimated_metadata,
        )
        center_error = float(
            torch.linalg.vector_norm(
                estimated_transform[:3, 3] - true_transform[:3, 3]
            ).item()
        )
        scale_log_error = abs(
            float(
                torch.log(
                    torch.tensor(
                        similarity_scale(estimated_transform) / true_scale
                    )
                ).item()
            )
        )
        row = {
            "uid": str(sample["uid"]),
            "object_uid": str(sample["object_uid"]),
            "view_count": int(len(sample["view_ids"])),
            "prior_point_count": int(len(sample["prior_coords"])),
            "true_scale": true_scale,
            "oracle": oracle,
            "no_recovery": no_recovery,
            "point_pca_estimate": estimated,
            "point_pca_transform_error": {
                "center_error": center_error,
                "center_error_over_true_scale": center_error / max(true_scale, 1.0e-8),
                "scale_log_error": scale_log_error,
                "rotation_error_degrees": rotation_error_degrees(
                    estimated_transform, true_transform
                ),
            },
            "estimator_improves_over_no_recovery": (
                estimated["visual_normalized_mse"]
                < no_recovery["visual_normalized_mse"]
            ),
        }
        rows.append(row)
        print(
            f"[object_frame] {index + 1}/{count} uid={sample['uid']} "
            f"oracle={oracle['visual_normalized_mse']:.6g} "
            f"none={no_recovery['visual_normalized_mse']:.6g} "
            f"pca={estimated['visual_normalized_mse']:.6g}",
            flush=True,
        )

    oracle_mse = [row["oracle"]["visual_normalized_mse"] for row in rows]
    no_recovery_mse = [
        row["no_recovery"]["visual_normalized_mse"] for row in rows
    ]
    estimated_mse = [
        row["point_pca_estimate"]["visual_normalized_mse"] for row in rows
    ]
    center_errors = [
        row["point_pca_transform_error"]["center_error_over_true_scale"]
        for row in rows
    ]
    scale_errors = [
        row["point_pca_transform_error"]["scale_log_error"] for row in rows
    ]
    rotation_errors = [
        row["point_pca_transform_error"]["rotation_error_degrees"] for row in rows
    ]
    no_recovery_detection_ratio = mean(
        float(value >= float(args.min_no_recovery_mse))
        for value in no_recovery_mse
    )
    estimator_improvement_ratio = mean(
        float(row["estimator_improves_over_no_recovery"]) for row in rows
    )
    mechanism_checks = {
        "oracle_sim3_recovers_baseline": max(oracle_mse)
        <= float(args.oracle_max_normalized_mse),
        "missing_object_frame_is_detectably_wrong": no_recovery_detection_ratio
        >= float(args.min_no_recovery_detection_ratio),
    }
    estimator_checks = {
        "point_pca_improves_over_no_recovery": estimator_improvement_ratio
        >= float(args.min_estimator_improvement_ratio),
        "point_pca_center_accuracy": median(center_errors)
        <= float(args.max_estimator_center_error_over_scale),
        "point_pca_scale_accuracy": median(scale_errors)
        <= float(args.max_estimator_scale_log_error),
    }
    report = {
        "passed": all(mechanism_checks.values()) and all(estimator_checks.values()),
        "mechanism_passed": all(mechanism_checks.values()),
        "estimated_frame_passed": all(estimator_checks.values()),
        "mechanism_checks": mechanism_checks,
        "estimator_checks": estimator_checks,
        "format": OBJECT_FRAME_VERSION,
        "lifting_volume_version": LIFTING_VOLUME_VERSION,
        "cache_manifest": str(Path(args.cache_manifest).resolve()),
        "args": vars(args),
        "sample_count": len(rows),
        "oracle_normalized_mse": summarize(oracle_mse),
        "no_recovery_normalized_mse": summarize(no_recovery_mse),
        "point_pca_normalized_mse": summarize(estimated_mse),
        "no_recovery_detection_ratio": no_recovery_detection_ratio,
        "estimator_improvement_ratio": estimator_improvement_ratio,
        "point_pca_center_error_over_scale": summarize(center_errors),
        "point_pca_scale_log_error": summarize(scale_errors),
        "point_pca_rotation_error_degrees": summarize(rotation_errors),
        "interpretation": {
            "mechanism_passed": "object-to-world is a required explicit variable",
            "estimated_frame_passed": "point-only PCA is adequate for the registered pilot",
            "rotation_warning": "PCA axes have sign/permutation and semantic orientation ambiguity",
        },
        "samples": rows,
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    lines = [
        "# P0.5 Synthetic Sim(3) Object-frame Audit",
        "",
        f"- passed: `{report['passed']}`",
        f"- mechanism passed: `{report['mechanism_passed']}`",
        f"- estimated frame passed: `{report['estimated_frame_passed']}`",
        f"- samples: `{len(rows)}`",
        f"- oracle normalized MSE mean/max: `{mean(oracle_mse):.6g} / {max(oracle_mse):.6g}`",
        f"- no-recovery normalized MSE mean: `{mean(no_recovery_mse):.6g}`",
        f"- point-PCA normalized MSE mean: `{mean(estimated_mse):.6g}`",
        f"- estimator improvement ratio: `{estimator_improvement_ratio:.6g}`",
        f"- PCA rotation error median: `{median(rotation_errors):.6g}` degrees",
        "",
        "A mechanism PASS does not authorize training unless estimated_frame_passed is also true.",
    ]
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "mechanism_checks": mechanism_checks,
                "estimator_checks": estimator_checks,
            },
            indent=2,
        )
    )
    if args.fail_on_error and not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
