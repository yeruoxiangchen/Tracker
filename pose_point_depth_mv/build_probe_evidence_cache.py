#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean
from typing import Any

import torch

from ar_ss_flow.local_pose_lifting_flow import (
    PoseLiftingCacheDataset,
    parse_indices,
)
from pose_point_depth_mv.geometry import (
    EVIDENCE_NAMES,
    build_evidence,
    prepare_frozen_crossfit_calibration,
)
from pose_point_depth_mv.local_target_probe import (
    NEGATIVE_INDEX,
    PPD_PROBE_CACHE_VERSION,
    POSITIVE_INDEX,
    PROBE_CORRUPTIONS,
)


def validate_evidence(features: torch.Tensor, *, uid: str, branch: str) -> int:
    expected = (len(EVIDENCE_NAMES), 16, 16, 16)
    if tuple(features.shape) != expected:
        raise ValueError(
            f"uid={uid} branch={branch} evidence shape={tuple(features.shape)}"
        )
    if not bool(torch.isfinite(features).all().item()):
        raise ValueError(f"uid={uid} branch={branch} evidence is non-finite")
    positive = features[POSITIVE_INDEX] > 0.5
    negative = features[NEGATIVE_INDEX] > 0.5
    if bool((positive & negative).any().item()):
        raise ValueError(f"uid={uid} branch={branch} positive/negative overlap")
    return int((positive | negative).sum().item())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Precompute strict cross-fit evidence for the PPD-3A probe."
    )
    parser.add_argument("--cache_manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--indices", default="all")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--calibration_fit_fraction", type=float, default=0.5)
    parser.add_argument("--calibration_split_seed", type=int, default=20260715)
    parser.add_argument("--minimum_points_per_split", type=int, default=4)
    parser.add_argument("--min_depth_matches", type=int, default=8)
    parser.add_argument("--min_heldout_matches", type=int, default=8)
    parser.add_argument("--calibration_mask_threshold", type=float, default=0.5)
    parser.add_argument("--calibration_zbuffer_cell_size", type=int, default=14)
    parser.add_argument("--allow_affine_calibration", action="store_true")
    parser.add_argument("--maximum_heldout_median_residual", type=float, default=0.25)
    parser.add_argument("--maximum_heldout_p90_residual", type=float, default=0.60)
    parser.add_argument("--quality_reference_residual", type=float, default=0.25)
    parser.add_argument("--min_eligible_ratio", type=float, default=0.70)
    parser.add_argument("--allow_failures", action="store_true")
    return parser.parse_args()


def different_object_sample(
    dataset: PoseLiftingCacheDataset, source_index: int
) -> dict[str, Any]:
    source = dataset[source_index]
    object_uid = str(source.get("object_uid", source["uid"]))
    for offset in range(1, len(dataset)):
        candidate = dataset[(source_index + offset) % len(dataset)]
        if str(candidate.get("object_uid", candidate["uid"])) != object_uid:
            return candidate
    raise RuntimeError("cross-object evidence requires two distinct objects")


@torch.no_grad()
def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    sample_dir = output_dir / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    source_manifest = Path(args.cache_manifest).resolve()
    dataset = PoseLiftingCacheDataset(source_manifest, indices="all")
    source_indices = parse_indices(args.indices, len(dataset))
    config = {
        "format": PPD_PROBE_CACHE_VERSION,
        "source_cache_manifest": str(source_manifest),
        "source_indices": source_indices,
        "evidence_names": list(EVIDENCE_NAMES),
        "corruption_names": list(PROBE_CORRUPTIONS),
        "calibration_fit_fraction": float(args.calibration_fit_fraction),
        "calibration_split_seed": int(args.calibration_split_seed),
        "minimum_points_per_split": int(args.minimum_points_per_split),
        "min_depth_matches": int(args.min_depth_matches),
        "min_heldout_matches": int(args.min_heldout_matches),
        "calibration_mask_threshold": float(args.calibration_mask_threshold),
        "calibration_zbuffer_cell_size": int(
            args.calibration_zbuffer_cell_size
        ),
        "force_scale_only": not bool(args.allow_affine_calibration),
        "maximum_heldout_median_residual": float(
            args.maximum_heldout_median_residual
        ),
        "maximum_heldout_p90_residual": float(
            args.maximum_heldout_p90_residual
        ),
        "quality_reference_residual": float(args.quality_reference_residual),
    }
    config_hash = hashlib.sha256(
        json.dumps(config, sort_keys=True).encode("utf-8")
    ).hexdigest()
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    common = {
        "device": device,
        "volume_side": 16,
        "minimum_depth_tolerance": 0.02,
        "maximum_depth_tolerance": 0.15,
        "surface_threshold": 0.30,
        "free_threshold": 0.30,
        "minimum_surface_views": 2,
        "minimum_free_views": 2,
        "prior_radius_voxels": 1.5,
        "gate_floor": 0.25,
        "min_depth_matches": int(args.min_depth_matches),
        "calibration_mask_threshold": float(args.calibration_mask_threshold),
        "calibration_zbuffer_cell_size": int(
            args.calibration_zbuffer_cell_size
        ),
        "force_scale_only": not bool(args.allow_affine_calibration),
        "recalibrate_each_hypothesis": False,
    }
    for position, source_index in enumerate(source_indices):
        sample = dataset[source_index]
        uid = str(sample["uid"])
        object_uid = str(sample.get("object_uid", uid))
        try:
            cross = different_object_sample(dataset, source_index)
            protocol = prepare_frozen_crossfit_calibration(
                sample,
                device=device,
                fit_fraction=float(args.calibration_fit_fraction),
                split_seed=int(args.calibration_split_seed),
                minimum_points_per_split=int(args.minimum_points_per_split),
                min_depth_matches=int(args.min_depth_matches),
                min_heldout_matches=int(args.min_heldout_matches),
                mask_threshold=float(args.calibration_mask_threshold),
                zbuffer_cell_size=int(args.calibration_zbuffer_cell_size),
                force_scale_only=not bool(args.allow_affine_calibration),
                maximum_heldout_median_residual=float(
                    args.maximum_heldout_median_residual
                ),
                maximum_heldout_p90_residual=float(
                    args.maximum_heldout_p90_residual
                ),
                quality_reference_residual=float(args.quality_reference_residual),
            )
            evidence_protocol = {
                "calibration_override": protocol["calibration"],
                "input_prior_coords": protocol["fit_coords"],
                "input_prior_confidence": protocol["fit_confidence"],
                "evaluation_prior_coords": protocol["heldout_coords"],
                "evaluation_prior_confidence": protocol["heldout_confidence"],
            }
            correct = build_evidence(
                sample, **common, **evidence_protocol
            )
            correct_active_count = validate_evidence(
                correct.features, uid=uid, branch="correct"
            )
            corrupt_features: dict[str, torch.Tensor] = {}
            corrupt_stats: dict[str, Any] = {}
            corrupt_active_counts: dict[str, int] = {}
            for name, overrides in PROBE_CORRUPTIONS.items():
                result = build_evidence(
                    sample,
                    cross_object_sample=cross,
                    **common,
                    **evidence_protocol,
                    **overrides,
                )
                corrupt_active_counts[name] = validate_evidence(
                    result.features, uid=uid, branch=name
                )
                corrupt_features[name] = result.features.cpu().to(torch.float16)
                corrupt_stats[name] = result.stats
            eligible = bool(correct.stats["depth_calibration_enabled"]) and (
                correct_active_count > 0
            )
            relative_path = Path("samples") / f"{source_index:05d}_{uid}.pt"
            torch.save(
                {
                    "format": PPD_PROBE_CACHE_VERSION,
                    "config_hash": config_hash,
                    "uid": uid,
                    "object_uid": object_uid,
                    "source_index": int(source_index),
                    "eligible": eligible,
                    "correct_features": correct.features.cpu().to(torch.float16),
                    "corrupt_features": corrupt_features,
                    "calibration": protocol["calibration"],
                    "stats": {
                        "correct": correct.stats,
                        "corruptions": corrupt_stats,
                        "correct_active_count": correct_active_count,
                        "corrupt_active_counts": corrupt_active_counts,
                        "cross_object_uid": str(
                            cross.get("object_uid", cross["uid"])
                        ),
                    },
                },
                output_dir / relative_path,
            )
            rows.append(
                {
                    "source_index": int(source_index),
                    "uid": uid,
                    "object_uid": object_uid,
                    "eligible": eligible,
                    "evidence_file": str(relative_path),
                }
            )
            print(
                f"[ppd_probe_cache] {position + 1}/{len(source_indices)} "
                f"uid={uid} eligible={eligible}",
                flush=True,
            )
        except Exception as error:  # noqa: BLE001
            failures.append(
                {
                    "source_index": int(source_index),
                    "uid": uid,
                    "object_uid": object_uid,
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            print(f"[ppd_probe_cache] FAILED uid={uid}: {error}", flush=True)
            if not args.allow_failures:
                raise
    eligible_count = sum(bool(row["eligible"]) for row in rows)
    eligible_ratio = eligible_count / max(len(source_indices), 1)
    manifest = {
        "format": PPD_PROBE_CACHE_VERSION,
        "output_dir": str(output_dir.resolve()),
        "source_cache_manifest": str(source_manifest),
        "config": config,
        "config_hash": config_hash,
        "evidence_names": list(EVIDENCE_NAMES),
        "corruption_names": list(PROBE_CORRUPTIONS),
        "sample_count": len(rows),
        "eligible_count": eligible_count,
        "eligible_ratio": eligible_ratio,
        "samples": rows,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    checks = {
        "no_failures": not failures,
        "all_requested_samples_written": len(rows) == len(source_indices),
        "eligible_ratio": eligible_ratio >= float(args.min_eligible_ratio),
    }
    report = {
        "stage": "PPD-3A strict evidence cache",
        "passed": all(checks.values()),
        "args": vars(args),
        "checks": checks,
        "sample_count": len(rows),
        "eligible_count": eligible_count,
        "eligible_ratio": eligible_ratio,
        "failure_count": len(failures),
        "failures": failures,
        "train_eligible_count_0_15": sum(
            row["eligible"] for row in rows if 0 <= row["source_index"] <= 15
        ),
        "fresh_eligible_count_16_63": sum(
            row["eligible"] for row in rows if 16 <= row["source_index"] <= 63
        ),
        "mean_source_index": mean(row["source_index"] for row in rows)
        if rows
        else None,
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
