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
    protocol_hash,
    subset_sample_views,
)
from ar_ss_flow.local_pose_lifting_flow import PoseLiftingCacheDataset


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


def source_support(evidence: dict[str, Any], min_source_views: int) -> torch.Tensor:
    return evidence["base_weight"].gt(1.0e-6).sum(dim=0).ge(int(min_source_views))


def normalized_advantage(wrong: torch.Tensor, correct: torch.Tensor) -> torch.Tensor:
    return (wrong - correct) / (wrong + correct).clamp_min(1.0e-6)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create full-volume records for raw/local visual-only pairwise confidence "
            "calibration. Every branch in a comparison uses one common binary source-"
            "support map, preventing local smoothing from introducing a geometry shortcut."
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
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    if int(args.source_view_count) < 3:
        raise ValueError("source_view_count must be >=3")

    device = torch.device(args.device)
    dataset = PoseLiftingCacheDataset(args.cache_manifest, indices=args.indices)
    count = len(dataset) if args.max_samples <= 0 else min(len(dataset), args.max_samples)
    model, checkpoint = load_correspondence_checkpoint(
        args.checkpoint,
        device=device,
        visual_channels=dataset.visual_feature_dim,
    )
    model.eval()
    # Previous ablation showed geometry-pair similarity contributed almost nothing.
    # Force it off so the stored signal is the deployable visual-only confidence.
    model.geometry_pair_scale.zero_()

    modes = parse_csv(args.negative_modes)
    invalid = [mode for mode in modes if mode not in POSE_NEGATIVE_MODES]
    if invalid:
        raise ValueError(f"invalid pose modes={invalid}")
    mode_to_index = {mode: index for index, mode in enumerate(modes)}

    dep_mode: list[int] = []
    dep_object: list[int] = []
    dep_correct: list[np.ndarray] = []
    dep_wrong: list[np.ndarray] = []
    dep_shuffle: list[np.ndarray] = []
    dep_support: list[np.ndarray] = []

    probe_mode: list[int] = []
    probe_object: list[int] = []
    probe_heldout: list[int] = []
    probe_correct: list[np.ndarray] = []
    probe_wrong: list[np.ndarray] = []
    probe_shuffle: list[np.ndarray] = []
    probe_support: list[np.ndarray] = []
    probe_valid: list[np.ndarray] = []
    probe_reprojection: list[np.ndarray] = []
    probe_shuffle_reprojection: list[np.ndarray] = []

    object_uids: dict[str, str] = {}
    skipped: list[dict[str, Any]] = []
    volume_side: int | None = None

    for sample_index in range(count):
        full_sample = dataset[sample_index]
        full_views = int(full_sample["visual_patch_features"].shape[0])
        object_uid = str(full_sample.get("object_uid", full_sample["uid"]))
        object_uids[str(sample_index)] = object_uid

        # --------------------------------------------------------------
        # Deployment form: all selected views are sources.
        # --------------------------------------------------------------
        if full_views >= int(args.source_view_count):
            view_indices = deterministic_view_subset(full_views, int(args.source_view_count))
            sample = subset_sample_views(full_sample, view_indices)
            correct_evidence = evidence_from_sample(sample, device=device, mode="correct")
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
            current_side = int(correct_evidence["volume_side"])
            volume_side = current_side if volume_side is None else volume_side
            if volume_side != current_side:
                raise ValueError("mixed volume_side values are not supported")
            c_support = source_support(correct_evidence, int(args.min_source_views))
            s_support = source_support(shuffled_evidence, int(args.min_source_views))
            c_conf = correct_aggregate["confidence"].reshape(-1)
            s_conf = shuffled_aggregate["confidence"].reshape(-1)

            for mode in modes:
                wrong_evidence = evidence_from_sample(sample, device=device, mode=mode)
                wrong_aggregate = model.aggregate_volume(
                    wrong_evidence,
                    min_source_views=int(args.min_source_views),
                )
                w_support = source_support(wrong_evidence, int(args.min_source_views))
                common_support = c_support & w_support & s_support
                common_count = int(common_support.sum().item())
                if common_count < int(args.min_common_voxels):
                    skipped.append(
                        {
                            "sample_index": sample_index,
                            "uid": str(full_sample["uid"]),
                            "object_uid": object_uid,
                            "stage": "deployment",
                            "mode": mode,
                            "reason": "insufficient_common_source_support",
                            "common_voxels": common_count,
                        }
                    )
                    continue
                dep_mode.append(mode_to_index[mode])
                dep_object.append(sample_index)
                dep_correct.append(c_conf.float().cpu().numpy().astype(np.float32))
                dep_wrong.append(
                    wrong_aggregate["confidence"]
                    .reshape(-1)
                    .float()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )
                dep_shuffle.append(s_conf.float().cpu().numpy().astype(np.float32))
                dep_support.append(common_support.cpu().numpy().astype(np.uint8))
        else:
            skipped.append(
                {
                    "sample_index": sample_index,
                    "uid": str(full_sample["uid"]),
                    "object_uid": object_uid,
                    "stage": "deployment",
                    "reason": "insufficient_views",
                    "available_views": full_views,
                    "required_views": int(args.source_view_count),
                }
            )

        # --------------------------------------------------------------
        # Probe form: same number of sources plus one fixed held-out view.
        # The target is used only for reprojection labels, never confidence.
        # --------------------------------------------------------------
        probe_total_views = int(args.source_view_count) + 1
        if full_views < probe_total_views:
            skipped.append(
                {
                    "sample_index": sample_index,
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
        target_evidence = evidence_from_sample(probe_sample, device=device, mode="correct")
        target_maps = model.encode_patch_maps(target_evidence["visual_patch_features"])

        for heldout in range(probe_total_views):
            correct_evidence = evidence_from_sample(
                probe_sample,
                device=device,
                mode="correct",
                heldout_index=heldout,
            )
            shuffled_evidence = evidence_from_sample(
                probe_sample,
                device=device,
                mode="correct",
                visual_patch_features_override=roll_visual_sources(
                    probe_sample["visual_patch_features"], heldout
                ),
                heldout_index=heldout,
            )
            correct_maps = model.encode_patch_maps(correct_evidence["visual_patch_features"])
            shuffled_maps = model.encode_patch_maps(shuffled_evidence["visual_patch_features"])
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
            c_support = source_support(correct_evidence, int(args.min_source_views))
            s_support = source_support(shuffled_evidence, int(args.min_source_views))

            for mode in modes:
                wrong_evidence = evidence_from_sample(
                    probe_sample,
                    device=device,
                    mode=mode,
                    heldout_index=heldout,
                )
                wrong_maps = model.encode_patch_maps(wrong_evidence["visual_patch_features"])
                wrong = model.evaluate_heldout(
                    wrong_evidence,
                    heldout,
                    neighborhood_radius=int(args.neighborhood_radius),
                    min_source_views=int(args.min_source_views),
                    encoded_patch_maps=wrong_maps,
                    target_evidence=target_evidence,
                    target_encoded_patch_maps=target_maps,
                )
                w_support = source_support(wrong_evidence, int(args.min_source_views))
                common_source_support = c_support & w_support & s_support
                common_valid = correct.valid_mask & wrong.valid_mask & shuffled.valid_mask
                valid = common_source_support & common_valid
                valid_count = int(valid.sum().item())
                if valid_count < int(args.min_common_voxels):
                    skipped.append(
                        {
                            "sample_index": sample_index,
                            "uid": str(full_sample["uid"]),
                            "object_uid": object_uid,
                            "stage": "heldout_probe",
                            "mode": mode,
                            "heldout": heldout,
                            "reason": "insufficient_common_heldout_support",
                            "common_voxels": valid_count,
                        }
                    )
                    continue

                probe_mode.append(mode_to_index[mode])
                probe_object.append(sample_index)
                probe_heldout.append(heldout)
                probe_correct.append(
                    correct.pairwise_confidence.float().cpu().numpy().astype(np.float32)
                )
                probe_wrong.append(
                    wrong.pairwise_confidence.float().cpu().numpy().astype(np.float32)
                )
                probe_shuffle.append(
                    shuffled.pairwise_confidence.float().cpu().numpy().astype(np.float32)
                )
                probe_support.append(
                    common_source_support.cpu().numpy().astype(np.uint8)
                )
                probe_valid.append(valid.cpu().numpy().astype(np.uint8))
                probe_reprojection.append(
                    normalized_advantage(wrong.error, correct.error)
                    .float()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )
                probe_shuffle_reprojection.append(
                    normalized_advantage(shuffled.error, correct.error)
                    .float()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )

    if not dep_correct:
        raise RuntimeError("no deployment records were produced")
    if not probe_correct:
        raise RuntimeError("no heldout probe records were produced")
    if volume_side is None:
        raise RuntimeError("volume_side could not be inferred")

    arrays = {
        "deployment_mode_index": np.asarray(dep_mode, dtype=np.int8),
        "deployment_object_index": np.asarray(dep_object, dtype=np.int32),
        "deployment_correct_confidence": np.stack(dep_correct).astype(np.float32),
        "deployment_wrong_confidence": np.stack(dep_wrong).astype(np.float32),
        "deployment_shuffle_confidence": np.stack(dep_shuffle).astype(np.float32),
        "deployment_common_support": np.stack(dep_support).astype(np.uint8),
        "probe_mode_index": np.asarray(probe_mode, dtype=np.int8),
        "probe_object_index": np.asarray(probe_object, dtype=np.int32),
        "probe_heldout_index": np.asarray(probe_heldout, dtype=np.int8),
        "probe_correct_confidence": np.stack(probe_correct).astype(np.float32),
        "probe_wrong_confidence": np.stack(probe_wrong).astype(np.float32),
        "probe_shuffle_confidence": np.stack(probe_shuffle).astype(np.float32),
        "probe_common_source_support": np.stack(probe_support).astype(np.uint8),
        "probe_valid_mask": np.stack(probe_valid).astype(np.uint8),
        "probe_reprojection_advantage": np.stack(probe_reprojection).astype(np.float32),
        "probe_shuffle_reprojection_advantage": np.stack(
            probe_shuffle_reprojection
        ).astype(np.float32),
    }
    np.savez_compressed(output_dir / "volume_samples.npz", **arrays)

    protocol = {
        "stage": "C1.5-v2 local visual-only deployment confidence audit",
        "cache_manifest": str(Path(args.cache_manifest).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_step": int(checkpoint.get("step", 0)),
        "indices": args.indices,
        "source_view_count": int(args.source_view_count),
        "probe_total_view_count": int(args.source_view_count) + 1,
        "negative_modes": list(modes),
        "volume_side": int(volume_side),
        "all_selected_views_are_sources_in_deployment_branch": True,
        "heldout_probe_uses_fixed_target": True,
        "confidence_does_not_read_heldout_target": True,
        "visual_only_pairwise": True,
        "geometry_pair_scale_forced_zero": True,
        "source_visual_shuffle_keeps_geometry_fixed": True,
        "local_smoothing_support_policy": "common_binary_source_support_shared_by_all_branches",
        "full_volume_records_saved": True,
    }
    protocol["protocol_hash"] = protocol_hash(protocol)

    report = {
        "stage": protocol["stage"],
        "args": vars(args),
        "protocol": protocol,
        "dataset_size": len(dataset),
        "evaluated_sample_count": count,
        "object_uids": object_uids,
        "mode_to_index": mode_to_index,
        "deployment_record_count": int(arrays["deployment_mode_index"].shape[0]),
        "probe_record_count": int(arrays["probe_mode_index"].shape[0]),
        "voxel_count_per_record": int(volume_side**3),
        "skipped": skipped,
        "volume_samples": str((output_dir / "volume_samples.npz").resolve()),
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    print("=" * 100)
    print(report["stage"])
    print("indices:", args.indices)
    print("checkpoint_step:", protocol["checkpoint_step"])
    print("volume_side:", volume_side)
    print("deployment_record_count:", report["deployment_record_count"])
    print("probe_record_count:", report["probe_record_count"])
    print("skipped_count:", len(skipped))
    print("report:", output_dir / "report.json")
    print("volume samples:", output_dir / "volume_samples.npz")


if __name__ == "__main__":
    main()
