#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
from statistics import mean, median
import sys
from typing import Any

os.environ.setdefault("SPCONV_ALGO", "native")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
os.environ.setdefault("TORCH_EXTENSIONS_DIR", "/tmp/torch_extensions")

import numpy as np
import torch

TRACKER_ROOT = Path(__file__).resolve().parents[1]
for path in (TRACKER_ROOT, TRACKER_ROOT / "ReconViaGen", TRACKER_ROOT / "ReconViaGen" / "wheels" / "vggt"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ar_ss_flow.local_pose_lifting_flow import PoseLiftingCacheDataset  # noqa: E402
from pose_point_depth_mv import PACKAGE_VERSION  # noqa: E402
from pose_point_depth_mv.geometry import (  # noqa: E402
    build_evidence,
    prepare_frozen_crossfit_calibration,
)


CORRUPTIONS = {
    "pose_cyclic1": {"pose_mode": "pose_cyclic1"},
    "pose_cyclic2": {"pose_mode": "pose_cyclic2"},
    "pose_reverse": {"pose_mode": "pose_reverse"},
    "depth_view_cyclic1": {"depth_mode": "depth_view_cyclic1"},
    "depth_spatial": {"depth_mode": "depth_spatial"},
    "point_reflect": {"point_mode": "point_reflect"},
    "point_axis_cycle": {"point_mode": "point_axis_cycle"},
    "point_cross_object": {"point_mode": "point_cross_object"},
}


def summarize(values: list[float]) -> dict[str, float | int]:
    finite = [float(value) for value in values if np.isfinite(value)]
    if not finite:
        return {"count": 0, "mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
    return {
        "count": len(finite),
        "mean": mean(finite),
        "median": median(finite),
        "min": min(finite),
        "max": max(finite),
    }


def positive_rate(values: list[float]) -> float:
    return float(mean(value > 0.0 for value in values)) if values else 0.0


def different_object_sample(dataset, index: int, count: int) -> dict[str, Any]:
    source = dataset[index]
    source_uid = str(source.get("object_uid", source["uid"]))
    for offset in range(1, count):
        candidate = dataset[(index + offset) % count]
        if str(candidate.get("object_uid", candidate["uid"])) != source_uid:
            return candidate
    raise RuntimeError("point_cross_object requires at least two distinct objects")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit local multi-view pose-point-depth geometry before any Flow integration."
    )
    parser.add_argument("--cache_manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--indices", default="16-63")
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--volume_side", type=int, default=16)
    parser.add_argument("--minimum_depth_tolerance", type=float, default=0.02)
    parser.add_argument("--maximum_depth_tolerance", type=float, default=0.15)
    parser.add_argument("--surface_threshold", type=float, default=0.30)
    parser.add_argument("--free_threshold", type=float, default=0.30)
    parser.add_argument("--minimum_surface_views", type=int, default=2)
    parser.add_argument("--minimum_free_views", type=int, default=2)
    parser.add_argument("--prior_radius_voxels", type=float, default=1.5)
    parser.add_argument("--gate_floor", type=float, default=0.25)
    parser.add_argument("--min_depth_matches", type=int, default=8)
    parser.add_argument("--min_heldout_matches", type=int, default=8)
    parser.add_argument("--affine_improvement_ratio", type=float, default=0.90)
    parser.add_argument("--calibration_fit_fraction", type=float, default=0.5)
    parser.add_argument("--calibration_split_seed", type=int, default=20260715)
    parser.add_argument("--minimum_points_per_split", type=int, default=4)
    parser.add_argument("--calibration_mask_threshold", type=float, default=0.5)
    parser.add_argument("--calibration_zbuffer_cell_size", type=int, default=14)
    parser.add_argument("--allow_affine_calibration", action="store_true")
    parser.add_argument("--maximum_heldout_median_residual", type=float, default=0.25)
    parser.add_argument("--maximum_heldout_p90_residual", type=float, default=0.60)
    parser.add_argument("--quality_reference_residual", type=float, default=0.25)
    parser.add_argument("--min_correct_calibration_rate", type=float, default=0.70)
    parser.add_argument("--min_object_win_rate", type=float, default=0.60)
    parser.add_argument("--max_inactive_gate_abs", type=float, default=0.0)
    parser.add_argument("--save_evidence", action="store_true")
    parser.add_argument("--allow_failures", action="store_true")
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    evidence_dir = output_dir / "evidence"
    if args.save_evidence:
        evidence_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    dataset = PoseLiftingCacheDataset(args.cache_manifest, indices=args.indices)
    count = len(dataset) if args.max_samples <= 0 else min(len(dataset), args.max_samples)
    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    common = {
        "device": device,
        "volume_side": int(args.volume_side),
        "minimum_depth_tolerance": float(args.minimum_depth_tolerance),
        "maximum_depth_tolerance": float(args.maximum_depth_tolerance),
        "surface_threshold": float(args.surface_threshold),
        "free_threshold": float(args.free_threshold),
        "minimum_surface_views": int(args.minimum_surface_views),
        "minimum_free_views": int(args.minimum_free_views),
        "prior_radius_voxels": float(args.prior_radius_voxels),
        "gate_floor": float(args.gate_floor),
        "min_depth_matches": int(args.min_depth_matches),
        "affine_improvement_ratio": float(args.affine_improvement_ratio),
        "calibration_mask_threshold": float(args.calibration_mask_threshold),
        "calibration_zbuffer_cell_size": int(
            args.calibration_zbuffer_cell_size
        ),
        "force_scale_only": not bool(args.allow_affine_calibration),
        "recalibrate_each_hypothesis": False,
    }

    for index in range(count):
        sample = dataset[index]
        uid = str(sample["uid"])
        object_uid = str(sample.get("object_uid", uid))
        cross = different_object_sample(dataset, index, len(dataset))
        try:
            protocol = prepare_frozen_crossfit_calibration(
                sample,
                device=device,
                fit_fraction=float(args.calibration_fit_fraction),
                split_seed=int(args.calibration_split_seed),
                minimum_points_per_split=int(args.minimum_points_per_split),
                min_depth_matches=int(args.min_depth_matches),
                min_heldout_matches=int(args.min_heldout_matches),
                affine_improvement_ratio=float(args.affine_improvement_ratio),
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
            correct = build_evidence(sample, **common, **evidence_protocol)
            mode_rows: dict[str, Any] = {"correct": correct.stats}
            gate_differences: dict[str, float] = {}
            score_differences: dict[str, float] = {}
            for name, overrides in CORRUPTIONS.items():
                result = build_evidence(
                    sample,
                    cross_object_sample=cross,
                    **common,
                    **evidence_protocol,
                    **overrides,
                )
                mode_rows[name] = result.stats
                gate_differences[name] = float(
                    (correct.local_gate - result.local_gate).abs().mean().item()
                )
                score_differences[name] = float(
                    correct.stats["object_consistency_score"]
                    - result.stats["object_consistency_score"]
                )
            row = {
                "index": index,
                "uid": uid,
                "object_uid": object_uid,
                "correct_calibration_enabled": bool(
                    correct.stats["depth_calibration_enabled"]
                ),
                "correct_calibration_match_count": int(
                    correct.stats["depth_calibration_match_count"]
                ),
                "calibration": protocol["calibration"],
                "calibration_fit_indices": protocol["fit_indices"].cpu().tolist(),
                "calibration_heldout_indices": protocol[
                    "heldout_indices"
                ].cpu().tolist(),
                "cross_object_uid": str(cross.get("object_uid", cross["uid"])),
                "modes": mode_rows,
                "correct_minus_corruption_score": score_differences,
                "correct_vs_corruption_gate_l1": gate_differences,
            }
            records.append(row)
            if args.save_evidence:
                torch.save(
                    {
                        "format": PACKAGE_VERSION,
                        "uid": uid,
                        "object_uid": object_uid,
                        "features": correct.features.cpu().to(torch.float16),
                        "local_gate": correct.local_gate.cpu().to(torch.float16),
                        "stats": correct.stats,
                        "calibration": correct.calibration,
                    },
                    evidence_dir / f"{uid}.pt",
                )
            print(
                f"[ppd_geometry] {index + 1}/{count} uid={uid} "
                f"score={correct.stats['object_consistency_score']:+.6f}",
                flush=True,
            )
        except Exception as error:  # noqa: BLE001
            failures.append(
                {
                    "index": index,
                    "uid": uid,
                    "object_uid": object_uid,
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            print(f"[ppd_geometry] FAILED uid={uid}: {error}", flush=True)
            if not args.allow_failures:
                raise

    if not records:
        raise RuntimeError("geometry audit produced no valid records")
    calibration_rate = sum(
        row["correct_calibration_enabled"] for row in records
    ) / max(count, 1)
    heldout_medians = [
        float(row["calibration"]["heldout"]["median_abs_residual"])
        for row in records
        if row["calibration"]["heldout"].get("median_abs_residual") is not None
    ]
    heldout_p90s = [
        float(row["calibration"]["heldout"]["p90_abs_residual"])
        for row in records
        if row["calibration"]["heldout"].get("p90_abs_residual") is not None
    ]
    comparisons: dict[str, Any] = {}
    eligible_rows = [
        row for row in records if row["correct_calibration_enabled"]
    ]
    for name in CORRUPTIONS:
        all_values = [
            row["correct_minus_corruption_score"][name] for row in records
        ]
        values = [
            row["correct_minus_corruption_score"][name]
            for row in eligible_rows
        ]
        gate_l1 = [
            row["correct_vs_corruption_gate_l1"][name]
            for row in eligible_rows
        ]
        comparisons[name] = {
            "eligible_object_count": len(eligible_rows),
            "score_difference": summarize(values),
            "object_win_rate": positive_rate(values),
            "gate_l1": summarize(gate_l1),
            "all_object_score_difference": summarize(all_values),
            "all_object_win_rate": positive_rate(all_values),
        }

    overlap_total = sum(
        int(mode["label_overlap"])
        for row in records
        for mode in row["modes"].values()
    )
    max_mask_zero_gate = max(
        float(mode["mask_zero_gate_mean"])
        for row in records
        for mode in row["modes"].values()
    )
    max_neutral_gate = max(
        float(mode["neutral_gate_mean"])
        for row in records
        for mode in row["modes"].values()
    )
    max_negative_gate = max(
        float(mode["negative_gate_mean"])
        for row in records
        for mode in row["modes"].values()
    )
    checks = {
        "no_sample_failures": not failures,
        "correct_depth_calibration_coverage": calibration_rate
        >= float(args.min_correct_calibration_rate),
        "positive_negative_labels_mutually_exclusive": overlap_total == 0,
        "mask_zero_gate_is_zero": max_mask_zero_gate
        <= float(args.max_inactive_gate_abs),
        "neutral_gate_is_zero": max_neutral_gate
        <= float(args.max_inactive_gate_abs),
        "negative_gate_is_zero": max_negative_gate
        <= float(args.max_inactive_gate_abs),
        "correct_beats_all_corruptions": all(
            float(item["score_difference"]["mean"]) > 0.0
            and float(item["object_win_rate"]) >= float(args.min_object_win_rate)
            for item in comparisons.values()
        ),
    }
    passed = all(checks.values())
    report = {
        "stage": "standalone multi-view pose-point-depth geometry causal audit",
        "format": PACKAGE_VERSION,
        "passed": passed,
        "args": vars(args),
        "object_count": len(records),
        "failure_count": len(failures),
        "correct_calibration_rate": calibration_rate,
        "heldout_calibration_median_residual": summarize(heldout_medians),
        "heldout_calibration_p90_residual": summarize(heldout_p90s),
        "label_overlap_total": overlap_total,
        "max_mask_zero_gate_mean": max_mask_zero_gate,
        "max_neutral_gate_mean": max_neutral_gate,
        "max_negative_gate_mean": max_negative_gate,
        "protocol": {
            "calibration": "correct-only frozen cross-fit",
            "point_protocol": "fit points as input; disjoint held-out points for scoring",
            "evaluation_points": "held-out correct-object points",
            "cross_object_must_differ": True,
            "coordinate_frame": (
                "sample object_to_world when present; otherwise cache K/T must "
                "already share the canonical SS frame"
            ),
        },
        "comparisons": comparisons,
        "checks": checks,
        "failures": failures,
        "records": records,
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    lines = [
        "# Pose-Point-Depth Geometry Audit",
        "",
        f"- passed: `{passed}`",
        f"- objects: `{len(records)}`",
        f"- failures: `{len(failures)}`",
        f"- correct depth calibration rate: `{calibration_rate:.4f}`",
        f"- held-out median residual mean: `{summarize(heldout_medians)['mean']:.6g}`",
        f"- held-out P90 residual mean: `{summarize(heldout_p90s)['mean']:.6g}`",
        f"- label overlap total: `{overlap_total}`",
        f"- max mask-zero / neutral / negative gate: "
        f"`{max_mask_zero_gate:.3g} / {max_neutral_gate:.3g} / {max_negative_gate:.3g}`",
        "",
        "| corruption | eligible N | correct-minus-corrupt mean | median | eligible win | all-object win | gate L1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, item in comparisons.items():
        lines.append(
            f"| {name} | {item['eligible_object_count']} | "
            f"{item['score_difference']['mean']:+.6g} | "
            f"{item['score_difference']['median']:+.6g} | "
            f"{item['object_win_rate']:.4f} | "
            f"{item['all_object_win_rate']:.4f} | "
            f"{item['gate_l1']['mean']:.6g} |"
        )
    lines.extend(["", "## Checks", ""])
    lines.extend(f"- {key}: `{value}`" for key, value in checks.items())
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("=" * 100)
    print(report["stage"])
    print("passed:", passed)
    print("objects:", len(records), "failures:", len(failures))
    print("correct_calibration_rate:", calibration_rate)
    for name, item in comparisons.items():
        print(
            name,
            "gap=", f"{item['score_difference']['mean']:+.6f}",
            "win=", f"{item['object_win_rate']:.4f}",
            "gate_l1=", f"{item['gate_l1']['mean']:.6f}",
        )
    print("checks:", checks)
    print("report:", output_dir / "report.json")
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
