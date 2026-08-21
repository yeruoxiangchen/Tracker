#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import mean, median
import sys
from types import MethodType
from typing import Any

import torch
import torch.nn.functional as F

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


ABLATIONS = (
    "full",
    "visual_zero",
    "visual_shuffle",
    "geometry_pair_off",
    "uniform_pairwise",
)


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


def ablate_source_visual(
    visual: torch.Tensor,
    *,
    heldout: int,
    ablation: str,
) -> torch.Tensor:
    """Ablate only source views; preserve the held-out target feature exactly."""
    if ablation not in ("visual_zero", "visual_shuffle"):
        return visual
    result = visual.clone()
    source_indices = [index for index in range(int(visual.shape[0])) if index != heldout]
    if ablation == "visual_zero":
        result[source_indices] = 0
    else:
        source_index = torch.as_tensor(source_indices, dtype=torch.long, device=visual.device)
        source_visual = visual.index_select(0, source_index)
        result[source_indices] = torch.roll(source_visual, shifts=1, dims=0)
    return result


def install_uniform_pairwise(model: torch.nn.Module) -> None:
    """Replace learned pairwise gating with physical-weight-only mean aggregation."""

    def uniform_pairwise(
        self: torch.nn.Module,
        per_view_embedding: torch.Tensor,
        geometry: torch.Tensor,
        physical_weight: torch.Tensor,
        *,
        min_weight: float,
    ) -> dict[str, torch.Tensor]:
        del geometry
        if per_view_embedding.ndim != 3:
            raise ValueError("per_view_embedding must be [V,N,D]")
        if physical_weight.shape != per_view_embedding.shape[:2]:
            raise ValueError("physical_weight must be [V,N]")

        views = int(per_view_embedding.shape[0])
        active = physical_weight.gt(float(min_weight))
        pair_mask = active[:, None, :] & active[None, :, :]
        eye = torch.eye(views, device=pair_mask.device, dtype=torch.bool)[:, :, None]
        pair_mask = pair_mask & ~eye
        peer_count = pair_mask.sum(dim=1)
        has_peer = peer_count.gt(0)

        per_view_confidence = has_peer.float()
        per_view_logit = torch.zeros_like(per_view_confidence)
        final_weight = physical_weight * per_view_confidence
        final_sum = final_weight.sum(dim=0)
        physical_sum = physical_weight.sum(dim=0)
        pairwise_confidence = (
            per_view_confidence * physical_weight
        ).sum(dim=0) / physical_sum.clamp_min(1.0e-6)
        pairwise_logit = torch.zeros_like(pairwise_confidence)

        value = F.normalize(
            self.value_projection(per_view_embedding), dim=-1, eps=1.0e-6
        )
        consensus = (
            value * final_weight[..., None]
        ).sum(dim=0) / final_sum.clamp_min(1.0e-6)[:, None]
        consensus = F.normalize(consensus, dim=-1, eps=1.0e-6)
        centered = value - consensus[None]
        disagreement = (
            centered.square().mean(dim=-1) * final_weight
        ).sum(dim=0) / final_sum.clamp_min(1.0e-6)
        effective_views = final_sum.square() / final_weight.square().sum(dim=0).clamp_min(
            1.0e-6
        )

        pairwise_logits = per_view_embedding.new_zeros(
            (views, views, int(per_view_embedding.shape[1]))
        )
        return {
            "pairwise_logits": pairwise_logits,
            "pairwise_probability": torch.sigmoid(pairwise_logits),
            "pairwise_mask": pair_mask,
            "per_view_confidence": per_view_confidence,
            "per_view_logit": per_view_logit,
            "peer_count": peer_count,
            "pairwise_confidence": pairwise_confidence,
            "pairwise_logit": pairwise_logit,
            "final_weight": final_weight,
            "consensus": consensus,
            "disagreement": disagreement,
            "effective_views": effective_views,
        }

    model._pairwise_consensus = MethodType(uniform_pairwise, model)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Ablation audit for pairwise-before-aggregation correspondence. "
            "The held-out target remains unmodified in every ablation."
        )
    )
    parser.add_argument("--cache_manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--indices", default="all")
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--ablation", choices=ABLATIONS, required=True)
    parser.add_argument(
        "--negative_modes",
        default="pose_cyclic1,pose_cyclic2,pose_reverse",
    )
    parser.add_argument(
        "--view_counts",
        default="3,4",
        help="TOTAL views. 3=2 source+held-out; 4=3 source+held-out.",
    )
    parser.add_argument("--neighborhood_radius", type=int, default=1)
    parser.add_argument("--min_common_voxels", type=int, default=64)
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

    if args.ablation == "geometry_pair_off":
        with torch.no_grad():
            model.geometry_pair_scale.zero_()
    elif args.ablation == "uniform_pairwise":
        install_uniform_pairwise(model)

    negative_modes = parse_csv(args.negative_modes)
    invalid = [mode for mode in negative_modes if mode not in CORRESPONDENCE_NEGATIVE_MODES]
    if invalid:
        raise ValueError(f"invalid negative modes={invalid}")
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

            target_evidence = evidence_from_sample(
                sample, device=device, mode="correct"
            )
            target_maps = model.encode_patch_maps(
                target_evidence["visual_patch_features"]
            )
            cross_sample = find_cross_sample(
                dataset, sample_index, full_sample, view_indices
            )

            for mode in negative_modes:
                wrong_base_visual: torch.Tensor | None = None
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
                    wrong_base_visual = cross_sample["visual_patch_features"]
                    wrong_mode = "correct"

                heldout_rows: list[dict[str, float | int]] = []
                for heldout in range(total_views):
                    correct_source_visual = ablate_source_visual(
                        sample["visual_patch_features"],
                        heldout=heldout,
                        ablation=args.ablation,
                    )
                    wrong_visual_source = (
                        wrong_base_visual
                        if wrong_base_visual is not None
                        else sample["visual_patch_features"]
                    )
                    wrong_source_visual = ablate_source_visual(
                        wrong_visual_source,
                        heldout=heldout,
                        ablation=args.ablation,
                    )

                    correct_evidence = evidence_from_sample(
                        sample,
                        device=device,
                        mode="correct",
                        visual_patch_features_override=correct_source_visual,
                        heldout_index=heldout,
                    )
                    wrong_evidence = evidence_from_sample(
                        sample,
                        device=device,
                        mode=wrong_mode,
                        visual_patch_features_override=wrong_source_visual,
                        heldout_index=heldout,
                    )
                    correct_maps = model.encode_patch_maps(
                        correct_evidence["visual_patch_features"]
                    )
                    wrong_maps = model.encode_patch_maps(
                        wrong_evidence["visual_patch_features"]
                    )

                    correct = model.evaluate_heldout(
                        correct_evidence,
                        heldout,
                        neighborhood_radius=int(args.neighborhood_radius),
                        min_source_views=source_views,
                        encoded_patch_maps=correct_maps,
                        target_evidence=target_evidence,
                        target_encoded_patch_maps=target_maps,
                    )
                    wrong = model.evaluate_heldout(
                        wrong_evidence,
                        heldout,
                        neighborhood_radius=int(args.neighborhood_radius),
                        min_source_views=source_views,
                        encoded_patch_maps=wrong_maps,
                        target_evidence=target_evidence,
                        target_encoded_patch_maps=target_maps,
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
                    heldout_rows.append(
                        {
                            "heldout": heldout,
                            "common_voxels": common_count,
                            "correct_error": correct_error,
                            "wrong_error": wrong_error,
                            "advantage": advantage,
                            "pairwise_confidence_advantage": pairwise_confidence_advantage,
                            "pairwise_logit_advantage": pairwise_logit_advantage,
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
                    "heldout": heldout_rows,
                }
                records.append(row)
                print(
                    f"[pairwise_ablation:{args.ablation}] {json.dumps(row)}",
                    flush=True,
                )

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
                    sum(value > 0.0 for value in pairwise_scores) / len(pairwise_scores)
                    if pairwise_scores
                    else 0.0
                ),
                "pairwise_logit_advantage": summarize(pairwise_logits),
            }
        summary[str(source_view_count)] = by_mode

    if not available_source_view_counts:
        raise RuntimeError("no requested source-view count was available in the cache")

    protocol = {
        "cache_manifest": str(Path(args.cache_manifest).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_step": int(checkpoint.get("step", 0)),
        "indices": args.indices,
        "ablation": args.ablation,
        "negative_modes": list(negative_modes),
        "requested_total_view_counts": list(requested_total_view_counts),
        "available_total_view_counts": sorted(available_total_view_counts),
        "available_source_view_counts": sorted(available_source_view_counts),
        "fixed_heldout_target": True,
        "source_only_ablation": True,
        "pairwise_before_aggregation": args.ablation != "uniform_pairwise",
        "geometry_pair_off": args.ablation == "geometry_pair_off",
        "uniform_pairwise": args.ablation == "uniform_pairwise",
    }
    protocol["protocol_hash"] = protocol_hash(protocol)

    report = {
        "stage": "C1 pairwise visual/geometry ablation audit",
        "ablation": args.ablation,
        "args": vars(args),
        "protocol": protocol,
        "highest_source_view_count": max(available_source_view_counts),
        "record_count": len(records),
        "object_record_count": len(object_rows),
        "skipped": skipped,
        "summary": summary,
        "records": records,
        "object_rows": object_rows,
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
