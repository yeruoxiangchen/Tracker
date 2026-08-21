#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

TRACKER_ROOT = Path(__file__).resolve().parents[1]
if str(TRACKER_ROOT) not in sys.path:
    sys.path.insert(0, str(TRACKER_ROOT))

from ar_ss_flow.correspondence_lifting import (
    POSE_NEGATIVE_MODES,
    deterministic_view_subset,
    evidence_from_sample,
    load_correspondence_checkpoint,
    parse_csv,
    pose_variant_extrinsics,
    protocol_hash,
    subset_sample_views,
)
from ar_ss_flow.local_pose_lifting_flow import PoseLiftingCacheDataset


HYPOTHESES = ("correct", "pose_cyclic1", "pose_cyclic2", "pose_reverse", "visual_shuffle")
VARIANTS = ("correct", "pose_cyclic1", "pose_cyclic2", "pose_reverse")


def roll_visual_all_views(visual: torch.Tensor) -> torch.Tensor:
    if int(visual.shape[0]) < 2:
        return visual.clone()
    return torch.roll(visual, shifts=1, dims=0)


def source_support(evidence: dict[str, Any], min_source_views: int) -> torch.Tensor:
    return evidence["base_weight"].gt(1.0e-6).sum(dim=0).ge(int(min_source_views))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build complete self-referenced object confidence records. For each observed "
            "hypothesis (correct, three pose corruptions, visual shuffle), recompute the "
            "observed confidence and three pose-perturbed reference confidences."
        )
    )
    parser.add_argument("--cache_manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--indices", default="all")
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--source_view_count", type=int, default=4)
    parser.add_argument(
        "--perturbation_modes",
        default="pose_cyclic1,pose_cyclic2,pose_reverse",
    )
    parser.add_argument("--min_source_views", type=int, default=2)
    parser.add_argument("--min_common_voxels", type=int, default=64)
    return parser.parse_args()


def hypothesis_sample(
    sample: dict[str, Any], hypothesis: str
) -> tuple[dict[str, Any], torch.Tensor]:
    result = dict(sample)
    visual = sample["visual_patch_features"]
    if hypothesis == "correct":
        return result, visual
    if hypothesis == "visual_shuffle":
        return result, roll_visual_all_views(visual)
    if hypothesis in POSE_NEGATIVE_MODES:
        result["extrinsics"] = pose_variant_extrinsics(sample["extrinsics"], hypothesis)
        return result, visual
    raise ValueError(f"unsupported hypothesis={hypothesis}")


@torch.no_grad()
def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    if int(args.source_view_count) < 3:
        raise ValueError("source_view_count must be >=3")

    perturbations = parse_csv(args.perturbation_modes)
    invalid = [mode for mode in perturbations if mode not in POSE_NEGATIVE_MODES]
    if invalid:
        raise ValueError(f"invalid perturbation modes={invalid}")
    expected = tuple(POSE_NEGATIVE_MODES)
    if tuple(perturbations) != expected:
        raise ValueError(
            f"self-reference protocol requires perturbations={expected}, got={perturbations}"
        )
    variants = ("correct", *perturbations)

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed) % (2**32 - 1))
    device = torch.device(args.device)
    dataset = PoseLiftingCacheDataset(args.cache_manifest, indices=args.indices)
    count = len(dataset) if args.max_samples <= 0 else min(len(dataset), args.max_samples)
    model, checkpoint = load_correspondence_checkpoint(
        args.checkpoint,
        device=device,
        visual_channels=dataset.visual_feature_dim,
    )
    model.eval()
    model.geometry_pair_scale.zero_()

    object_indices: list[int] = []
    confidence_records: list[np.ndarray] = []
    support_records: list[np.ndarray] = []
    support_counts: list[np.ndarray] = []
    object_uids: dict[str, str] = {}
    skipped: list[dict[str, Any]] = []
    volume_side: int | None = None

    for sample_index in range(count):
        full_sample = dataset[sample_index]
        full_views = int(full_sample["visual_patch_features"].shape[0])
        object_uid = str(full_sample.get("object_uid", full_sample["uid"]))
        object_uids[str(sample_index)] = object_uid
        if full_views < int(args.source_view_count):
            skipped.append(
                {
                    "sample_index": sample_index,
                    "uid": str(full_sample["uid"]),
                    "object_uid": object_uid,
                    "reason": "insufficient_views",
                    "available_views": full_views,
                    "required_views": int(args.source_view_count),
                }
            )
            continue

        view_indices = deterministic_view_subset(full_views, int(args.source_view_count))
        sample = subset_sample_views(full_sample, view_indices)
        per_hypothesis_confidence: list[np.ndarray] = []
        per_hypothesis_support: list[np.ndarray] = []
        per_hypothesis_count: list[int] = []
        failed = False

        for hypothesis in HYPOTHESES:
            base_sample, visual = hypothesis_sample(sample, hypothesis)
            variant_confidence: list[np.ndarray] = []
            variant_support: list[torch.Tensor] = []
            current_side: int | None = None

            for variant in variants:
                evidence = evidence_from_sample(
                    base_sample,
                    device=device,
                    mode=variant,
                    visual_patch_features_override=visual,
                )
                aggregate = model.aggregate_volume(
                    evidence,
                    min_source_views=int(args.min_source_views),
                )
                current_side = int(evidence["volume_side"])
                variant_confidence.append(
                    aggregate["confidence"].reshape(-1).float().cpu().numpy().astype(np.float32)
                )
                variant_support.append(source_support(evidence, int(args.min_source_views)))

            if current_side is None:
                raise RuntimeError("volume side was not produced")
            volume_side = current_side if volume_side is None else volume_side
            if int(volume_side) != int(current_side):
                raise ValueError("mixed volume_side values are not supported")

            common_support = torch.stack(variant_support, dim=0).all(dim=0)
            common_count = int(common_support.sum().item())
            if common_count < int(args.min_common_voxels):
                skipped.append(
                    {
                        "sample_index": sample_index,
                        "uid": str(full_sample["uid"]),
                        "object_uid": object_uid,
                        "hypothesis": hypothesis,
                        "reason": "insufficient_common_selfref_support",
                        "common_voxels": common_count,
                    }
                )
                failed = True
                break

            per_hypothesis_confidence.append(np.stack(variant_confidence).astype(np.float32))
            per_hypothesis_support.append(common_support.cpu().numpy().astype(np.uint8))
            per_hypothesis_count.append(common_count)

        if failed:
            continue
        object_indices.append(sample_index)
        confidence_records.append(np.stack(per_hypothesis_confidence).astype(np.float32))
        support_records.append(np.stack(per_hypothesis_support).astype(np.uint8))
        support_counts.append(np.asarray(per_hypothesis_count, dtype=np.int32))
        print(
            f"object={sample_index} hypotheses={len(HYPOTHESES)} variants={len(variants)} "
            f"min_support={min(per_hypothesis_count)}",
            flush=True,
        )

    if not confidence_records:
        raise RuntimeError("no complete self-reference records were produced")
    if volume_side is None:
        raise RuntimeError("volume_side could not be inferred")

    arrays = {
        "selfref_object_index": np.asarray(object_indices, dtype=np.int32),
        "selfref_confidence": np.stack(confidence_records).astype(np.float32),
        "selfref_common_support": np.stack(support_records).astype(np.uint8),
        "selfref_valid_voxel_count": np.stack(support_counts).astype(np.int32),
    }
    np.savez_compressed(output_dir / "selfref_samples.npz", **arrays)

    protocol = {
        "stage": "C1.6 self-referenced visual-only pairwise object audit",
        "cache_manifest": str(Path(args.cache_manifest).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_step": int(checkpoint.get("step", 0)),
        "indices": args.indices,
        "source_view_count": int(args.source_view_count),
        "hypotheses": list(HYPOTHESES),
        "variants": list(variants),
        "observed_variant_index": 0,
        "reference_variant_indices": list(range(1, len(variants))),
        "volume_side": int(volume_side),
        "visual_only_pairwise": True,
        "geometry_pair_scale_forced_zero": True,
        "complete_reperturbation_for_each_hypothesis": True,
        "visual_shuffle_reperturbed_with_pose_variants": True,
        "common_support_within_each_hypothesis": True,
    }
    protocol["protocol_hash"] = protocol_hash(protocol)
    report = {
        "stage": protocol["stage"],
        "args": vars(args),
        "protocol": protocol,
        "dataset_size": len(dataset),
        "evaluated_sample_count": count,
        "complete_object_count": int(arrays["selfref_object_index"].shape[0]),
        "object_uids": object_uids,
        "skipped": skipped,
        "selfref_samples": str((output_dir / "selfref_samples.npz").resolve()),
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("=" * 100)
    print(report["stage"])
    print("indices:", args.indices)
    print("checkpoint_step:", protocol["checkpoint_step"])
    print("complete_object_count:", report["complete_object_count"])
    print("confidence_shape:", arrays["selfref_confidence"].shape)
    print("support_shape:", arrays["selfref_common_support"].shape)
    print("skipped_count:", len(skipped))
    print("report:", output_dir / "report.json")
    print("samples:", output_dir / "selfref_samples.npz")


if __name__ == "__main__":
    main()
