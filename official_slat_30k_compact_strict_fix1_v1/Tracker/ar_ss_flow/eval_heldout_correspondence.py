#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import mean, median
import sys
from typing import Any

import torch

TRACKER_ROOT = Path(__file__).resolve().parents[1]
if str(TRACKER_ROOT) not in sys.path:
    sys.path.insert(0, str(TRACKER_ROOT))

from ar_ss_flow.correspondence_lifting import (
    CORRESPONDENCE_NEGATIVE_MODES,
    deterministic_view_subset,
    evidence_from_sample,
    load_correspondence_checkpoint,
    parse_csv,
    protocol_hash,
    subset_sample_views,
)
from ar_ss_flow.local_pose_lifting_flow import PoseLiftingCacheDataset


def parse_int_csv(text: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in str(text).split(",") if item.strip())
    if not values or any(value < 3 for value in values):
        raise ValueError(
            "view_counts are TOTAL views and must be >=3: at least 2 sources + 1 held-out"
        )
    return values


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": mean(values),
        "median": median(values),
        "min": min(values),
        "max": max(values),
    }


def find_cross_sample(
    dataset: PoseLiftingCacheDataset,
    source_index: int,
    source: dict[str, Any],
    view_indices: list[int],
) -> dict[str, Any] | None:
    full_views = int(source["visual_patch_features"].shape[0])
    shape = tuple(source["visual_patch_features"].shape[1:])
    for offset in range(1, len(dataset)):
        candidate = dataset[(source_index + offset) % len(dataset)]
        if str(candidate.get("object_uid")) == str(source.get("object_uid")):
            continue
        if int(candidate["visual_patch_features"].shape[0]) != full_views:
            continue
        if tuple(candidate["visual_patch_features"].shape[1:]) != shape:
            continue
        return subset_sample_views(candidate, view_indices)
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Object-balanced pairwise-before-aggregation fixed-target audit. "
            "view_counts are total views: 3 means 2 source + 1 held-out."
        )
    )
    parser.add_argument("--cache_manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--indices", default="all")
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--negative_modes", default=",".join(CORRESPONDENCE_NEGATIVE_MODES)
    )
    parser.add_argument(
        "--required_modes", default="pose_cyclic1,pose_cyclic2,pose_reverse"
    )
    parser.add_argument(
        "--view_counts",
        default="3,4",
        help="TOTAL views. 3=2 source+held-out; 4=3 source+held-out.",
    )
    parser.add_argument("--neighborhood_radius", type=int, default=1)
    parser.add_argument("--min_common_voxels", type=int, default=64)
    parser.add_argument("--min_objects", type=int, default=8)
    parser.add_argument("--min_object_win_rate", type=float, default=0.65)
    parser.add_argument("--min_pairwise_win_rate", type=float, default=0.65)
    parser.add_argument(
        "--min_pairwise_confidence_advantage", type=float, default=0.0
    )
    parser.add_argument(
        "--min_visual_score_advantage",
        type=float,
        default=0.0,
        help=(
            "Deprecated compatibility option. The legacy visual score head is "
            "diagnostic-only in pairwise v3 and does not affect PASS/FAIL."
        ),
    )
    parser.add_argument("--view_trend_tolerance", type=float, default=0.02)
    parser.add_argument("--fail_on_error", action="store_true")
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    dataset = PoseLiftingCacheDataset(args.cache_manifest, indices=args.indices)
    count = len(dataset) if args.max_samples <= 0 else min(len(dataset), args.max_samples)
    model, checkpoint = load_correspondence_checkpoint(
        args.checkpoint,
        device=device,
        visual_channels=dataset.visual_feature_dim,
    )
    model.eval()

    negative_modes = parse_csv(args.negative_modes)
    required_modes = parse_csv(args.required_modes)
    invalid = [mode for mode in negative_modes if mode not in CORRESPONDENCE_NEGATIVE_MODES]
    if invalid:
        raise ValueError(f"invalid negative modes={invalid}")
    missing_required = [mode for mode in required_modes if mode not in negative_modes]
    if missing_required:
        raise ValueError(f"required modes absent from negative_modes={missing_required}")
    requested_total_view_counts = parse_int_csv(args.view_counts)

    records: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    available_total_view_counts: set[int] = set()
    available_source_view_counts: set[int] = set()

    for sample_index in range(count):
        full_sample = dataset[sample_index]
        full_views = int(full_sample["visual_patch_features"].shape[0])
        for requested_total in requested_total_view_counts:
            if requested_total > full_views:
                continue
            view_indices = deterministic_view_subset(full_views, requested_total)
            sample = subset_sample_views(full_sample, view_indices)
            total_views = int(sample["visual_patch_features"].shape[0])
            source_views = total_views - 1
            if source_views < 2:
                continue
            available_total_view_counts.add(total_views)
            available_source_view_counts.add(source_views)

            correct_evidence = evidence_from_sample(sample, device=device, mode="correct")
            correct_maps = model.encode_patch_maps(
                correct_evidence["visual_patch_features"]
            )
            cross_sample = find_cross_sample(
                dataset, sample_index, full_sample, view_indices
            )

            for mode in negative_modes:
                visual_override = None
                wrong_mode = mode
                if mode == "cross_sample":
                    if cross_sample is None:
                        skipped.append(
                            {
                                "uid": str(full_sample["uid"]),
                                "total_view_count": total_views,
                                "source_view_count": source_views,
                                "mode": mode,
                                "reason": "no_matching_cross_sample",
                            }
                        )
                        continue
                    visual_override = cross_sample["visual_patch_features"]
                    wrong_mode = "correct"

                wrong_maps = model.encode_patch_maps(
                    visual_override.to(device=device, dtype=torch.float16)
                    if visual_override is not None
                    else correct_evidence["visual_patch_features"]
                )
                heldout_rows: list[dict[str, float | int]] = []

                for heldout in range(total_views):
                    wrong_evidence = evidence_from_sample(
                        sample,
                        device=device,
                        mode=wrong_mode,
                        visual_patch_features_override=visual_override,
                        heldout_index=heldout,
                    )
                    correct = model.evaluate_heldout(
                        correct_evidence,
                        heldout,
                        neighborhood_radius=int(args.neighborhood_radius),
                        min_source_views=source_views,
                        encoded_patch_maps=correct_maps,
                        target_evidence=correct_evidence,
                        target_encoded_patch_maps=correct_maps,
                    )
                    wrong = model.evaluate_heldout(
                        wrong_evidence,
                        heldout,
                        neighborhood_radius=int(args.neighborhood_radius),
                        min_source_views=source_views,
                        encoded_patch_maps=wrong_maps,
                        target_evidence=correct_evidence,
                        target_encoded_patch_maps=correct_maps,
                    )
                    common = correct.valid_mask & wrong.valid_mask
                    common_count = int(common.sum().item())
                    if common_count < int(args.min_common_voxels):
                        continue

                    correct_error = float(correct.error[common].mean().item())
                    wrong_error = float(wrong.error[common].mean().item())
                    advantage = (wrong_error - correct_error) / max(
                        wrong_error + correct_error, 1.0e-6
                    )
                    pairwise_confidence_advantage = float(
                        (
                            correct.pairwise_confidence[common]
                            - wrong.pairwise_confidence[common]
                        ).mean().item()
                    )
                    pairwise_logit_advantage = float(
                        (
                            correct.pairwise_logit[common]
                            - wrong.pairwise_logit[common]
                        ).mean().item()
                    )
                    visual_score_advantage = float(
                        (
                            correct.confidence_logit[common]
                            - wrong.confidence_logit[common]
                        ).mean().item()
                    )
                    geometry_score_advantage = float(
                        (
                            correct.geometry_logit[common]
                            - wrong.geometry_logit[common]
                        ).mean().item()
                    )
                    heldout_rows.append(
                        {
                            "heldout": heldout,
                            "common_voxels": common_count,
                            "correct_error": correct_error,
                            "wrong_error": wrong_error,
                            "advantage": advantage,
                            "pairwise_confidence_advantage": pairwise_confidence_advantage,
                            "pairwise_logit_advantage": pairwise_logit_advantage,
                            "visual_score_advantage": visual_score_advantage,
                            "geometry_score_advantage": geometry_score_advantage,
                        }
                    )

                if not heldout_rows:
                    skipped.append(
                        {
                            "uid": str(full_sample["uid"]),
                            "total_view_count": total_views,
                            "source_view_count": source_views,
                            "mode": mode,
                            "reason": "no_heldout_common_support",
                        }
                    )
                    continue

                row = {
                    "sample_index": sample_index,
                    "uid": str(full_sample["uid"]),
                    "object_uid": str(
                        full_sample.get("object_uid", full_sample["uid"])
                    ),
                    "total_view_count": total_views,
                    "source_view_count": source_views,
                    "view_indices": view_indices,
                    "mode": mode,
                    "heldout_count": len(heldout_rows),
                    "common_voxels_mean": mean(
                        [int(item["common_voxels"]) for item in heldout_rows]
                    ),
                    "correct_error": mean(
                        [float(item["correct_error"]) for item in heldout_rows]
                    ),
                    "wrong_error": mean(
                        [float(item["wrong_error"]) for item in heldout_rows]
                    ),
                    "advantage": mean(
                        [float(item["advantage"]) for item in heldout_rows]
                    ),
                    "pairwise_confidence_advantage": mean(
                        [
                            float(item["pairwise_confidence_advantage"])
                            for item in heldout_rows
                        ]
                    ),
                    "pairwise_logit_advantage": mean(
                        [
                            float(item["pairwise_logit_advantage"])
                            for item in heldout_rows
                        ]
                    ),
                    "visual_score_advantage": mean(
                        [
                            float(item["visual_score_advantage"])
                            for item in heldout_rows
                        ]
                    ),
                    "geometry_score_advantage": mean(
                        [
                            float(item["geometry_score_advantage"])
                            for item in heldout_rows
                        ]
                    ),
                    "heldout": heldout_rows,
                }
                records.append(row)
                print(f"[heldout_corr_eval_v3] {json.dumps(row)}", flush=True)

    object_buckets: dict[tuple[int, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        object_buckets[
            (row["source_view_count"], row["mode"], row["object_uid"])
        ].append(row)

    object_rows: list[dict[str, Any]] = []
    for (source_view_count, mode, object_uid), rows in sorted(object_buckets.items()):
        object_rows.append(
            {
                "source_view_count": source_view_count,
                "mode": mode,
                "object_uid": object_uid,
                "sample_count": len(rows),
                "advantage": mean([float(row["advantage"]) for row in rows]),
                "pairwise_confidence_advantage": mean(
                    [float(row["pairwise_confidence_advantage"]) for row in rows]
                ),
                "pairwise_logit_advantage": mean(
                    [float(row["pairwise_logit_advantage"]) for row in rows]
                ),
                "visual_score_advantage": mean(
                    [float(row["visual_score_advantage"]) for row in rows]
                ),
                "geometry_score_advantage": mean(
                    [float(row["geometry_score_advantage"]) for row in rows]
                ),
            }
        )

    summary: dict[str, Any] = {}
    for source_view_count in sorted(available_source_view_counts):
        by_mode: dict[str, Any] = {}
        for mode in negative_modes:
            rows = [
                row
                for row in object_rows
                if row["source_view_count"] == source_view_count
                and row["mode"] == mode
            ]
            advantages = [float(row["advantage"]) for row in rows]
            pairwise_scores = [
                float(row["pairwise_confidence_advantage"]) for row in rows
            ]
            pairwise_logits = [
                float(row["pairwise_logit_advantage"]) for row in rows
            ]
            visual_scores = [
                float(row["visual_score_advantage"]) for row in rows
            ]
            geometry_scores = [
                float(row["geometry_score_advantage"]) for row in rows
            ]
            by_mode[mode] = {
                "object_count": len(rows),
                "advantage": summarize(advantages),
                "object_win_rate": (
                    sum(value > 0.0 for value in advantages) / len(advantages)
                    if advantages
                    else 0.0
                ),
                "pairwise_confidence_advantage": summarize(pairwise_scores),
                "pairwise_confidence_win_rate": (
                    sum(value > 0.0 for value in pairwise_scores)
                    / len(pairwise_scores)
                    if pairwise_scores
                    else 0.0
                ),
                "pairwise_logit_advantage": summarize(pairwise_logits),
                "visual_score_advantage": summarize(visual_scores),
                "visual_score_win_rate": (
                    sum(value > 0.0 for value in visual_scores) / len(visual_scores)
                    if visual_scores
                    else 0.0
                ),
                "geometry_score_advantage": summarize(geometry_scores),
                "geometry_score_win_rate": (
                    sum(value > 0.0 for value in geometry_scores)
                    / len(geometry_scores)
                    if geometry_scores
                    else 0.0
                ),
            }
        summary[str(source_view_count)] = by_mode

    if not available_source_view_counts:
        raise RuntimeError("no requested source-view count was available in the cache")
    highest_source_view_count = max(available_source_view_counts)
    highest_total_view_count = highest_source_view_count + 1

    required_checks: dict[str, Any] = {}
    for mode in required_modes:
        row = summary[str(highest_source_view_count)][mode]
        criteria = {
            "enough_objects": int(row["object_count"]) >= int(args.min_objects),
            "reprojection_object_win_rate": (
                float(row["object_win_rate"]) >= float(args.min_object_win_rate)
            ),
            "reprojection_mean_positive": float(row["advantage"]["mean"]) > 0.0,
            "reprojection_median_positive": float(row["advantage"]["median"]) > 0.0,
            "pairwise_object_win_rate": (
                float(row["pairwise_confidence_win_rate"])
                >= float(args.min_pairwise_win_rate)
            ),
            "pairwise_mean_positive": (
                float(row["pairwise_confidence_advantage"]["mean"])
                > float(args.min_pairwise_confidence_advantage)
            ),
            "pairwise_median_positive": (
                float(row["pairwise_confidence_advantage"]["median"])
                > float(args.min_pairwise_confidence_advantage)
            ),
        }
        passed = all(criteria.values())
        required_checks[mode] = {
            "passed": passed,
            "criteria": criteria,
            "metrics": row,
            "diagnostic_only": {
                "visual_score_advantage": row["visual_score_advantage"],
                "visual_score_win_rate": row["visual_score_win_rate"],
                "geometry_score_advantage": row["geometry_score_advantage"],
                "geometry_score_win_rate": row["geometry_score_win_rate"],
            },
        }

    view_trend_checks: dict[str, Any] = {}
    sorted_source_counts = sorted(available_source_view_counts)
    if len(sorted_source_counts) >= 2:
        low = sorted_source_counts[0]
        high = sorted_source_counts[-1]
        for mode in required_modes:
            low_map = {
                str(row["object_uid"]): float(row["advantage"])
                for row in object_rows
                if row["source_view_count"] == low and row["mode"] == mode
            }
            high_map = {
                str(row["object_uid"]): float(row["advantage"])
                for row in object_rows
                if row["source_view_count"] == high and row["mode"] == mode
            }
            paired_ids = sorted(set(low_map) & set(high_map))
            paired_low = [low_map[uid] for uid in paired_ids]
            paired_high = [high_map[uid] for uid in paired_ids]
            paired_delta = [high_map[uid] - low_map[uid] for uid in paired_ids]
            delta_summary = summarize(paired_delta)
            trend_pass = (
                len(paired_ids) >= int(args.min_objects)
                and float(delta_summary["mean"])
                + float(args.view_trend_tolerance)
                >= 0.0
            )
            view_trend_checks[mode] = {
                "low_source_view_count": low,
                "high_source_view_count": high,
                "paired_object_count": len(paired_ids),
                "low_advantage": summarize(paired_low),
                "high_advantage": summarize(paired_high),
                "paired_high_minus_low": delta_summary,
                "passed": trend_pass,
            }

    trend_passed = all(row["passed"] for row in view_trend_checks.values())
    passed = all(row["passed"] for row in required_checks.values()) and trend_passed

    protocol = {
        "cache_manifest": str(Path(args.cache_manifest).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_step": int(checkpoint.get("step", 0)),
        "indices": args.indices,
        "negative_modes": list(negative_modes),
        "required_modes": list(required_modes),
        "requested_view_counts": list(requested_total_view_counts),
        "available_view_counts": sorted(available_total_view_counts),
        "requested_total_view_counts": list(requested_total_view_counts),
        "available_total_view_counts": sorted(available_total_view_counts),
        "available_source_view_counts": sorted(available_source_view_counts),
        "fixed_heldout_target": True,
        "source_only_pose_corruption": True,
        "pairwise_before_aggregation": True,
        "legacy_score_heads_diagnostic_only": True,
        "neighborhood_radius": int(args.neighborhood_radius),
        "min_source_views": min(available_source_view_counts),
    }
    protocol["protocol_hash"] = protocol_hash(protocol)

    report = {
        "stage": "C1 pairwise-before-aggregation held-out evaluation v3.1",
        "passed": passed,
        "args": vars(args),
        "protocol": protocol,
        "highest_view_count": highest_total_view_count,
        "highest_source_view_count": highest_source_view_count,
        "record_count": len(records),
        "object_record_count": len(object_rows),
        "skipped": skipped,
        "summary": summary,
        "scientific_gate_fields": [
            "object_count",
            "reprojection_object_win_rate",
            "reprojection_mean_and_median",
            "pairwise_confidence_object_win_rate",
            "pairwise_confidence_mean_and_median",
            "paired_view_trend",
        ],
        "legacy_score_heads_diagnostic_only": True,
        "required_checks": required_checks,
        "view_trend_checks": view_trend_checks,
        "records": records,
        "object_rows": object_rows,
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)
    if args.fail_on_error and not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
