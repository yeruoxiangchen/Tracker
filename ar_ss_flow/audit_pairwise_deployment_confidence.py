#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import mean, median
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
    protocol_hash,
    subset_sample_views,
)
from ar_ss_flow.local_pose_lifting_flow import PoseLiftingCacheDataset


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": mean(values),
        "median": median(values),
        "min": min(values),
        "max": max(values),
    }


def roll_visual_all_views(visual: torch.Tensor) -> torch.Tensor:
    if int(visual.shape[0]) < 2:
        return visual.clone()
    return torch.roll(visual, shifts=1, dims=0)


def roll_visual_sources(visual: torch.Tensor, heldout: int) -> torch.Tensor:
    result = visual.clone()
    source_ids = [index for index in range(int(visual.shape[0])) if index != heldout]
    if len(source_ids) < 2:
        return result
    index = torch.as_tensor(source_ids, dtype=torch.long, device=visual.device)
    source = visual.index_select(0, index)
    result.index_copy_(0, index, torch.roll(source, shifts=1, dims=0))
    return result


def select_indices(mask: torch.Tensor, max_count: int) -> torch.Tensor:
    indices = torch.nonzero(mask, as_tuple=False).flatten()
    if max_count <= 0 or int(indices.numel()) <= max_count:
        return indices
    positions = torch.linspace(
        0,
        int(indices.numel()) - 1,
        steps=max_count,
        device=indices.device,
    ).round().long()
    return indices.index_select(0, positions)


def object_summary(rows: list[dict[str, Any]], modes: tuple[str, ...]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for mode in modes:
        sample_rows = [row for row in rows if row["mode"] == mode]
        buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in sample_rows:
            buckets[str(row["object_uid"])].append(row)
        subset: list[dict[str, Any]] = []
        for object_uid, bucket in sorted(buckets.items()):
            subset.append(
                {
                    "object_uid": object_uid,
                    "correct_minus_wrong_confidence": mean(
                        [float(row["correct_minus_wrong_confidence"]) for row in bucket]
                    ),
                    "correct_minus_shuffle_confidence": mean(
                        [float(row["correct_minus_shuffle_confidence"]) for row in bucket]
                    ),
                    "reprojection_advantage": mean(
                        [float(row.get("reprojection_advantage", 0.0)) for row in bucket]
                    ),
                }
            )
        correct_wrong = [float(row["correct_minus_wrong_confidence"]) for row in subset]
        correct_shuffle = [float(row["correct_minus_shuffle_confidence"]) for row in subset]
        reproj = [float(row.get("reprojection_advantage", 0.0)) for row in subset]
        result[mode] = {
            "object_count": len(subset),
            "correct_minus_wrong_confidence": summarize(correct_wrong),
            "correct_greater_wrong_object_rate": (
                sum(value > 0.0 for value in correct_wrong) / len(correct_wrong)
                if correct_wrong
                else 0.0
            ),
            "correct_minus_shuffle_confidence": summarize(correct_shuffle),
            "correct_greater_shuffle_object_rate": (
                sum(value > 0.0 for value in correct_shuffle) / len(correct_shuffle)
                if correct_shuffle
                else 0.0
            ),
            "reprojection_advantage": summarize(reproj),
            "reprojection_object_win_rate": (
                sum(value > 0.0 for value in reproj) / len(reproj)
                if reproj
                else 0.0
            ),
        }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "C1.5 audit for deployment-form visual-only pairwise confidence. "
            "The deployment branch uses every selected view as a source. A separate "
            "fixed-target probe measures voxel confidence versus held-out reprojection."
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
        "--negative_modes",
        default="pose_cyclic1,pose_cyclic2,pose_reverse",
    )
    parser.add_argument("--neighborhood_radius", type=int, default=1)
    parser.add_argument("--min_source_views", type=int, default=2)
    parser.add_argument("--min_common_voxels", type=int, default=64)
    parser.add_argument("--max_voxels_per_heldout", type=int, default=1024)
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    if int(args.source_view_count) < 3:
        raise ValueError("source_view_count must be >=3 for deployment calibration")

    device = torch.device(args.device)
    dataset = PoseLiftingCacheDataset(args.cache_manifest, indices=args.indices)
    count = len(dataset) if args.max_samples <= 0 else min(len(dataset), args.max_samples)
    model, checkpoint = load_correspondence_checkpoint(
        args.checkpoint,
        device=device,
        visual_channels=dataset.visual_feature_dim,
    )
    model.eval()
    # C1 ablation showed that the learned geometry-pair term contributes almost
    # nothing.  Deployment calibration therefore measures the visual-only head.
    model.geometry_pair_scale.zero_()

    modes = parse_csv(args.negative_modes)
    invalid = [mode for mode in modes if mode not in POSE_NEGATIVE_MODES]
    if invalid:
        raise ValueError(f"invalid pose modes={invalid}")

    deployment_rows: list[dict[str, Any]] = []
    probe_rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    voxel_mode: list[np.ndarray] = []
    voxel_object: list[np.ndarray] = []
    voxel_correct_conf: list[np.ndarray] = []
    voxel_wrong_conf: list[np.ndarray] = []
    voxel_shuffle_conf: list[np.ndarray] = []
    voxel_reproj_adv: list[np.ndarray] = []
    voxel_shuffle_adv: list[np.ndarray] = []

    mode_to_index = {mode: index for index, mode in enumerate(modes)}
    eligible_deployment = 0
    eligible_probe = 0

    for sample_index in range(count):
        full_sample = dataset[sample_index]
        full_views = int(full_sample["visual_patch_features"].shape[0])
        object_uid = str(full_sample.get("object_uid", full_sample["uid"]))

        # ------------------------------------------------------------------
        # Deployment form: all selected views are sources, no held-out target.
        # ------------------------------------------------------------------
        if full_views >= int(args.source_view_count):
            view_indices = deterministic_view_subset(
                full_views, int(args.source_view_count)
            )
            sample = subset_sample_views(full_sample, view_indices)
            correct_evidence = evidence_from_sample(
                sample, device=device, mode="correct"
            )
            shuffled_evidence = evidence_from_sample(
                sample,
                device=device,
                mode="correct",
                visual_patch_features_override=roll_visual_all_views(
                    sample["visual_patch_features"]
                ),
            )
            correct_aggregate = model.aggregate_volume(
                correct_evidence,
                min_source_views=int(args.min_source_views),
            )
            shuffled_aggregate = model.aggregate_volume(
                shuffled_evidence,
                min_source_views=int(args.min_source_views),
            )
            correct_support = (
                correct_evidence["base_weight"].gt(1.0e-6).sum(dim=0)
                >= int(args.min_source_views)
            )
            shuffle_support = (
                shuffled_evidence["base_weight"].gt(1.0e-6).sum(dim=0)
                >= int(args.min_source_views)
            )
            correct_conf = correct_aggregate["confidence"].reshape(-1)
            shuffle_conf = shuffled_aggregate["confidence"].reshape(-1)
            eligible_deployment += 1

            for mode in modes:
                wrong_evidence = evidence_from_sample(
                    sample, device=device, mode=mode
                )
                wrong_aggregate = model.aggregate_volume(
                    wrong_evidence,
                    min_source_views=int(args.min_source_views),
                )
                wrong_support = (
                    wrong_evidence["base_weight"].gt(1.0e-6).sum(dim=0)
                    >= int(args.min_source_views)
                )
                wrong_conf = wrong_aggregate["confidence"].reshape(-1)
                common = correct_support & wrong_support & shuffle_support
                common_count = int(common.sum().item())
                if common_count < int(args.min_common_voxels):
                    skipped.append(
                        {
                            "uid": str(full_sample["uid"]),
                            "object_uid": object_uid,
                            "stage": "deployment",
                            "mode": mode,
                            "reason": "insufficient_common_support",
                            "common_voxels": common_count,
                        }
                    )
                    continue
                c = float(correct_conf[common].mean().item())
                w = float(wrong_conf[common].mean().item())
                s = float(shuffle_conf[common].mean().item())
                deployment_rows.append(
                    {
                        "sample_index": sample_index,
                        "uid": str(full_sample["uid"]),
                        "object_uid": object_uid,
                        "mode": mode,
                        "source_view_count": int(args.source_view_count),
                        "view_indices": view_indices,
                        "common_voxels": common_count,
                        "correct_confidence": c,
                        "wrong_confidence": w,
                        "shuffle_confidence": s,
                        "correct_minus_wrong_confidence": c - w,
                        "correct_minus_shuffle_confidence": c - s,
                    }
                )
        else:
            skipped.append(
                {
                    "uid": str(full_sample["uid"]),
                    "object_uid": object_uid,
                    "stage": "deployment",
                    "reason": "insufficient_views",
                    "available_views": full_views,
                    "required_views": int(args.source_view_count),
                }
            )

        # ------------------------------------------------------------------
        # Fixed-target probe: 4 sources + 1 held-out by default.
        # ------------------------------------------------------------------
        probe_total_views = int(args.source_view_count) + 1
        if full_views < probe_total_views:
            skipped.append(
                {
                    "uid": str(full_sample["uid"]),
                    "object_uid": object_uid,
                    "stage": "heldout_probe",
                    "reason": "insufficient_views",
                    "available_views": full_views,
                    "required_views": probe_total_views,
                }
            )
            continue

        probe_indices = deterministic_view_subset(full_views, probe_total_views)
        probe_sample = subset_sample_views(full_sample, probe_indices)
        target_evidence = evidence_from_sample(
            probe_sample, device=device, mode="correct"
        )
        target_maps = model.encode_patch_maps(
            target_evidence["visual_patch_features"]
        )
        eligible_probe += 1

        per_object_mode_rows: dict[str, list[dict[str, float]]] = defaultdict(list)
        for heldout in range(probe_total_views):
            correct_evidence = evidence_from_sample(
                probe_sample,
                device=device,
                mode="correct",
                heldout_index=heldout,
            )
            shuffled_visual = roll_visual_sources(
                probe_sample["visual_patch_features"], heldout
            )
            shuffled_evidence = evidence_from_sample(
                probe_sample,
                device=device,
                mode="correct",
                visual_patch_features_override=shuffled_visual,
                heldout_index=heldout,
            )
            correct_maps = model.encode_patch_maps(
                correct_evidence["visual_patch_features"]
            )
            shuffled_maps = model.encode_patch_maps(
                shuffled_evidence["visual_patch_features"]
            )
            correct = model.evaluate_heldout(
                correct_evidence,
                heldout,
                neighborhood_radius=int(args.neighborhood_radius),
                min_source_views=int(args.min_source_views),
                encoded_patch_maps=correct_maps,
                target_evidence=target_evidence,
                target_encoded_patch_maps=target_maps,
            )
            shuffled = model.evaluate_heldout(
                shuffled_evidence,
                heldout,
                neighborhood_radius=int(args.neighborhood_radius),
                min_source_views=int(args.min_source_views),
                encoded_patch_maps=shuffled_maps,
                target_evidence=target_evidence,
                target_encoded_patch_maps=target_maps,
            )

            for mode in modes:
                wrong_evidence = evidence_from_sample(
                    probe_sample,
                    device=device,
                    mode=mode,
                    heldout_index=heldout,
                )
                wrong_maps = model.encode_patch_maps(
                    wrong_evidence["visual_patch_features"]
                )
                wrong = model.evaluate_heldout(
                    wrong_evidence,
                    heldout,
                    neighborhood_radius=int(args.neighborhood_radius),
                    min_source_views=int(args.min_source_views),
                    encoded_patch_maps=wrong_maps,
                    target_evidence=target_evidence,
                    target_encoded_patch_maps=target_maps,
                )
                common = correct.valid_mask & wrong.valid_mask & shuffled.valid_mask
                common_count = int(common.sum().item())
                if common_count < int(args.min_common_voxels):
                    continue

                c_error = correct.error[common]
                w_error = wrong.error[common]
                s_error = shuffled.error[common]
                reproj_adv = (w_error - c_error) / (w_error + c_error).clamp_min(1.0e-6)
                shuffle_adv = (s_error - c_error) / (s_error + c_error).clamp_min(1.0e-6)
                c_conf = correct.pairwise_confidence[common]
                w_conf = wrong.pairwise_confidence[common]
                s_conf = shuffled.pairwise_confidence[common]

                per_object_mode_rows[mode].append(
                    {
                        "correct_minus_wrong_confidence": float(
                            (c_conf - w_conf).mean().item()
                        ),
                        "correct_minus_shuffle_confidence": float(
                            (c_conf - s_conf).mean().item()
                        ),
                        "reprojection_advantage": float(reproj_adv.mean().item()),
                        "shuffle_reprojection_advantage": float(
                            shuffle_adv.mean().item()
                        ),
                        "common_voxels": float(common_count),
                    }
                )

                chosen = select_indices(
                    common,
                    int(args.max_voxels_per_heldout),
                )
                if int(chosen.numel()) == 0:
                    continue
                voxel_mode.append(
                    np.full(
                        int(chosen.numel()),
                        mode_to_index[mode],
                        dtype=np.int8,
                    )
                )
                voxel_object.append(
                    np.full(int(chosen.numel()), sample_index, dtype=np.int32)
                )
                voxel_correct_conf.append(
                    correct.pairwise_confidence.index_select(0, chosen)
                    .float()
                    .cpu()
                    .numpy()
                )
                voxel_wrong_conf.append(
                    wrong.pairwise_confidence.index_select(0, chosen)
                    .float()
                    .cpu()
                    .numpy()
                )
                voxel_shuffle_conf.append(
                    shuffled.pairwise_confidence.index_select(0, chosen)
                    .float()
                    .cpu()
                    .numpy()
                )
                chosen_c_error = correct.error.index_select(0, chosen)
                chosen_w_error = wrong.error.index_select(0, chosen)
                chosen_s_error = shuffled.error.index_select(0, chosen)
                voxel_reproj_adv.append(
                    (
                        (chosen_w_error - chosen_c_error)
                        / (chosen_w_error + chosen_c_error).clamp_min(1.0e-6)
                    )
                    .float()
                    .cpu()
                    .numpy()
                )
                voxel_shuffle_adv.append(
                    (
                        (chosen_s_error - chosen_c_error)
                        / (chosen_s_error + chosen_c_error).clamp_min(1.0e-6)
                    )
                    .float()
                    .cpu()
                    .numpy()
                )

        for mode in modes:
            rows = per_object_mode_rows.get(mode, [])
            if not rows:
                skipped.append(
                    {
                        "uid": str(full_sample["uid"]),
                        "object_uid": object_uid,
                        "stage": "heldout_probe",
                        "mode": mode,
                        "reason": "no_common_heldout_support",
                    }
                )
                continue
            probe_rows.append(
                {
                    "sample_index": sample_index,
                    "uid": str(full_sample["uid"]),
                    "object_uid": object_uid,
                    "mode": mode,
                    "source_view_count": int(args.source_view_count),
                    "probe_view_indices": probe_indices,
                    "heldout_count": len(rows),
                    "common_voxels_mean": mean(
                        [float(row["common_voxels"]) for row in rows]
                    ),
                    "correct_minus_wrong_confidence": mean(
                        [
                            float(row["correct_minus_wrong_confidence"])
                            for row in rows
                        ]
                    ),
                    "correct_minus_shuffle_confidence": mean(
                        [
                            float(row["correct_minus_shuffle_confidence"])
                            for row in rows
                        ]
                    ),
                    "reprojection_advantage": mean(
                        [float(row["reprojection_advantage"]) for row in rows]
                    ),
                    "shuffle_reprojection_advantage": mean(
                        [
                            float(row["shuffle_reprojection_advantage"])
                            for row in rows
                        ]
                    ),
                }
            )

    if not voxel_correct_conf:
        raise RuntimeError("no valid voxel calibration samples were produced")

    arrays = {
        "mode_index": np.concatenate(voxel_mode),
        "object_index": np.concatenate(voxel_object),
        "correct_confidence": np.concatenate(voxel_correct_conf).astype(np.float32),
        "wrong_confidence": np.concatenate(voxel_wrong_conf).astype(np.float32),
        "shuffle_confidence": np.concatenate(voxel_shuffle_conf).astype(np.float32),
        "reprojection_advantage": np.concatenate(voxel_reproj_adv).astype(np.float32),
        "shuffle_reprojection_advantage": np.concatenate(voxel_shuffle_adv).astype(
            np.float32
        ),
    }
    np.savez_compressed(output_dir / "voxel_samples.npz", **arrays)

    protocol = {
        "stage": "C1.5 visual-only deployment confidence audit",
        "cache_manifest": str(Path(args.cache_manifest).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_step": int(checkpoint.get("step", 0)),
        "indices": args.indices,
        "source_view_count": int(args.source_view_count),
        "probe_total_view_count": int(args.source_view_count) + 1,
        "negative_modes": list(modes),
        "all_selected_views_are_sources_in_deployment_branch": True,
        "heldout_probe_uses_fixed_target": True,
        "visual_only_pairwise": True,
        "geometry_pair_scale_forced_zero": True,
        "source_visual_shuffle_keeps_geometry_fixed": True,
        "max_voxels_per_heldout": int(args.max_voxels_per_heldout),
    }
    protocol["protocol_hash"] = protocol_hash(protocol)

    report = {
        "stage": "C1.5 visual-only deployment confidence audit",
        "args": vars(args),
        "protocol": protocol,
        "dataset_size": len(dataset),
        "evaluated_sample_count": count,
        "eligible_deployment_sample_count": eligible_deployment,
        "eligible_probe_sample_count": eligible_probe,
        "voxel_sample_count": int(arrays["correct_confidence"].shape[0]),
        "mode_to_index": mode_to_index,
        "deployment_summary": object_summary(deployment_rows, modes),
        "heldout_probe_summary": object_summary(probe_rows, modes),
        "deployment_rows": deployment_rows,
        "heldout_probe_rows": probe_rows,
        "skipped": skipped,
        "voxel_samples": str((output_dir / "voxel_samples.npz").resolve()),
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    print("=" * 100)
    print(report["stage"])
    print("checkpoint_step:", protocol["checkpoint_step"])
    print("source_view_count:", args.source_view_count)
    print("voxel_sample_count:", report["voxel_sample_count"])
    for mode in modes:
        deployment = report["deployment_summary"][mode]
        probe = report["heldout_probe_summary"][mode]
        print(
            f"{mode}: "
            f"deployment_obj_win={deployment['correct_greater_wrong_object_rate']:.4f} "
            f"deployment_gap={deployment['correct_minus_wrong_confidence']['mean']:+.6f} "
            f"probe_obj_win={probe['correct_greater_wrong_object_rate']:.4f} "
            f"probe_gap={probe['correct_minus_wrong_confidence']['mean']:+.6f} "
            f"reproj_win={probe['reprojection_object_win_rate']:.4f}"
        )
    print("report:", output_dir / "report.json")
    print("voxel samples:", output_dir / "voxel_samples.npz")


if __name__ == "__main__":
    main()
