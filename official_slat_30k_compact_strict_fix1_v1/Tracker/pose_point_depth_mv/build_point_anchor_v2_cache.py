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
from pose_point_depth_mv.point_anchor_v2 import (
    ACTIVE_INDEX,
    POINT_ANCHOR_CACHE_VERSION,
    POINT_CONTROL_NAMES,
    POINT_EVIDENCE_NAMES,
    build_point_evidence,
    deterministic_seed,
    deterministic_subsample_points,
    make_constant_evidence,
    make_drop_evidence,
    match_cross_object_points,
    row_hash,
    sha256_file,
    transform_points,
    validate_points,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build matched-control point-only local-anchor v2 evidence."
    )
    parser.add_argument("--source_cache_manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--indices", default="all")
    parser.add_argument("--radius_voxels", type=float, default=1.0)
    parser.add_argument("--max_points", type=int, default=512)
    parser.add_argument("--matching_seed", type=int, default=20260715)
    parser.add_argument("--max_active_ratio", type=float, default=0.35)
    parser.add_argument("--allow_cross_replacement", action="store_true")
    parser.add_argument("--allow_failures", action="store_true")
    return parser.parse_args()


def uid_hash(values: list[str]) -> str:
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def choose_cross_candidate(
    records: list[dict[str, Any]],
    source_index: int,
    *,
    matching_seed: int,
) -> dict[str, Any]:
    source = records[source_index]
    if source_index < 16:
        split_name = "train16"
        split_indices = range(0, min(16, len(records)))
    elif source_index < 64:
        split_name = "fresh48"
        split_indices = range(16, min(64, len(records)))
    else:
        raise ValueError("point-anchor v2 expects source indices in [0,63]")
    candidates = [
        records[index]
        for index in split_indices
        if index != source_index
        and records[index]["object_uid"] != source["object_uid"]
    ]
    if not candidates:
        raise RuntimeError(
            f"cross-object control requires distinct objects within {split_name}"
        )
    selected = min(
        candidates,
        key=lambda row: (
            abs(int(row["point_count"]) - int(source["point_count"])),
            deterministic_seed(
                source["uid"], row["uid"], matching_seed, "candidate_order"
            ),
        ),
    )
    return {**selected, "matching_split": split_name}


def validate_control(
    correct: torch.Tensor,
    control: torch.Tensor,
    *,
    uid: str,
    name: str,
) -> dict[str, float]:
    expected = (len(POINT_EVIDENCE_NAMES), 16, 16, 16)
    if tuple(control.shape) != expected:
        raise ValueError(f"uid={uid} control={name} shape={tuple(control.shape)}")
    if not bool(torch.isfinite(control).all().item()):
        raise ValueError(f"uid={uid} control={name} is non-finite")
    if not torch.equal(control[ACTIVE_INDEX], correct[ACTIVE_INDEX]):
        raise ValueError(f"uid={uid} control={name} changed correct active mask")
    point_difference = float((control[:3] - correct[:3]).abs().mean().item())
    if point_difference <= 0.0:
        raise ValueError(f"uid={uid} control={name} did not alter point content")
    return {"point_content_l1_from_correct": point_difference}


@torch.no_grad()
def main() -> None:
    args = parse_args()
    if args.radius_voxels <= 0.0:
        raise ValueError("radius_voxels must be positive")
    if args.max_points <= 0:
        raise ValueError("max_points must be positive")
    if not 0.0 < args.max_active_ratio < 1.0:
        raise ValueError("max_active_ratio must be in (0,1)")
    source_manifest = Path(args.source_cache_manifest).resolve()
    source_payload = json.loads(source_manifest.read_text(encoding="utf-8"))
    dataset = PoseLiftingCacheDataset(source_manifest, indices="all")
    source_indices = parse_indices(args.indices, len(dataset))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    sample_dir = output_dir / "samples"
    sample_dir.mkdir(parents=True)

    point_records: list[dict[str, Any]] = []
    for source_index in range(len(dataset)):
        sample = dataset[source_index]
        uid = str(sample["uid"])
        object_uid = str(sample.get("object_uid", uid))
        xyz, confidence = validate_points(
            sample["prior_coords"], sample["prior_confidence"], uid=uid
        )
        point_records.append(
            {
                "source_index": source_index,
                "uid": uid,
                "object_uid": object_uid,
                "xyz": xyz.cpu(),
                "confidence": confidence.cpu(),
                "point_count": int(len(xyz)),
            }
        )

    config = {
        "format": POINT_ANCHOR_CACHE_VERSION,
        "source_cache_manifest": str(source_manifest),
        "source_manifest_sha256": sha256_file(source_manifest),
        "source_manifest_format": source_payload.get("format"),
        "source_indices": source_indices,
        "evidence_names": list(POINT_EVIDENCE_NAMES),
        "control_names": list(POINT_CONTROL_NAMES),
        "radius_voxels": float(args.radius_voxels),
        "max_points": int(args.max_points),
        "matching_seed": int(args.matching_seed),
        "cross_confidence_policy": "exact_correct_confidence_multiset",
        "cross_object_policy": "same_split_nearest_count_no_replacement",
        "correct_point_policy": "deterministic_common_count_subsample",
        "active_mask_policy": "fixed_correct_point_neighborhood",
        "uses_pose_depth": False,
        "heldout_point_quality_in_input": False,
    }
    config_hash = hashlib.sha256(
        json.dumps(config, sort_keys=True).encode("utf-8")
    ).hexdigest()
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for position, source_index in enumerate(source_indices):
        record = point_records[source_index]
        uid = record["uid"]
        try:
            cross = choose_cross_candidate(
                point_records,
                source_index,
                matching_seed=int(args.matching_seed),
            )
            common_count = min(
                int(record["point_count"]),
                int(cross["point_count"]),
                int(args.max_points),
            )
            correct_xyz, correct_confidence = deterministic_subsample_points(
                record["xyz"],
                record["confidence"],
                common_count,
                uid=uid,
                seed=int(args.matching_seed),
            )
            matched_xyz, matched_confidence, matching = match_cross_object_points(
                correct_xyz,
                correct_confidence,
                cross["xyz"],
                cross["confidence"],
                uid=uid,
                candidate_uid=cross["uid"],
                seed=int(args.matching_seed),
            )
            if matching["used_coordinate_replacement"] and not args.allow_cross_replacement:
                raise RuntimeError(
                    "no cross-object candidate has enough points; coordinate replacement "
                    "is disabled"
                )
            correct = build_point_evidence(
                correct_xyz,
                correct_confidence,
                radius_voxels=float(args.radius_voxels),
            )
            correct_mask = correct[ACTIVE_INDEX]
            active_count = int(correct_mask.sum().item())
            active_ratio = float(correct_mask.mean().item())
            if active_count <= 0 or active_ratio > float(args.max_active_ratio):
                raise RuntimeError(
                    f"invalid point active region count={active_count} ratio={active_ratio}"
                )
            controls = {
                "point_reflect": build_point_evidence(
                    transform_points(correct_xyz, "point_reflect"),
                    correct_confidence,
                    reference_active_mask=correct_mask,
                ),
                "point_axis_cycle": build_point_evidence(
                    transform_points(correct_xyz, "point_axis_cycle"),
                    correct_confidence,
                    reference_active_mask=correct_mask,
                ),
                "point_spatial_roll": build_point_evidence(
                    transform_points(correct_xyz, "point_spatial_roll"),
                    correct_confidence,
                    reference_active_mask=correct_mask,
                ),
                "point_cross_object_matched": build_point_evidence(
                    matched_xyz,
                    matched_confidence,
                    reference_active_mask=correct_mask,
                ),
                "point_drop": make_drop_evidence(correct),
                "constant_prior": make_constant_evidence(correct),
            }
            if tuple(controls) != POINT_CONTROL_NAMES:
                raise RuntimeError("point control ordering changed")
            control_stats = {
                name: validate_control(correct, evidence, uid=uid, name=name)
                for name, evidence in controls.items()
            }
            relative_path = Path("samples") / f"{source_index:05d}_{uid}.pt"
            torch.save(
                {
                    "format": POINT_ANCHOR_CACHE_VERSION,
                    "config_hash": config_hash,
                    "uid": uid,
                    "object_uid": record["object_uid"],
                    "source_index": source_index,
                    "correct_evidence": correct.to(torch.float16),
                    "controls": {
                        name: evidence.to(torch.float16)
                        for name, evidence in controls.items()
                    },
                    "stats": {
                        "correct_point_count": int(len(correct_xyz)),
                        "original_correct_point_count": int(record["point_count"]),
                        "correct_points_subsampled": bool(
                            len(correct_xyz) != int(record["point_count"])
                        ),
                        "correct_unique_voxel_count": int(correct[0].sum().item()),
                        "correct_active_count": active_count,
                        "correct_active_ratio": active_ratio,
                        "cross_object_uid": cross["uid"],
                        "cross_object_object_uid": cross["object_uid"],
                        "cross_object_source_index": int(cross["source_index"]),
                        "cross_object_matching_split": cross["matching_split"],
                        "matching": matching,
                        "controls": control_stats,
                    },
                },
                output_dir / relative_path,
            )
            rows.append(
                {
                    "source_index": source_index,
                    "uid": uid,
                    "object_uid": record["object_uid"],
                    "source_row_hash": row_hash(dataset.rows[source_index]),
                    "evidence_file": str(relative_path),
                    "point_count": int(len(correct_xyz)),
                    "original_point_count": int(record["point_count"]),
                    "cross_matched_count": int(matching["matched_count"]),
                    "cross_confidence_multiset_max_abs_diff": float(
                        matching["confidence_multiset_max_abs_diff"]
                    ),
                    "cross_used_coordinate_replacement": bool(
                        matching["used_coordinate_replacement"]
                    ),
                    "active_count": active_count,
                    "active_ratio": active_ratio,
                    "cross_object_uid": cross["uid"],
                    "cross_source_index": int(cross["source_index"]),
                    "matching_split": cross["matching_split"],
                }
            )
            print(
                f"[point_anchor_cache] {position + 1}/{len(source_indices)} "
                f"uid={uid} points={len(correct_xyz)} active={active_count}",
                flush=True,
            )
        except Exception as error:  # noqa: BLE001
            failures.append(
                {
                    "source_index": source_index,
                    "uid": uid,
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            print(f"[point_anchor_cache] FAILED uid={uid}: {error}", flush=True)
            if not args.allow_failures:
                raise

    train_objects = {
        row["object_uid"] for row in rows if 0 <= row["source_index"] <= 15
    }
    fresh_objects = {
        row["object_uid"] for row in rows if 16 <= row["source_index"] <= 63
    }
    uid_values = [row["uid"] for row in rows]
    checks = {
        "no_failures": not failures,
        "all_requested_samples_written": len(rows) == len(source_indices),
        "uids_unique": len(uid_values) == len(set(uid_values)),
        "object_disjoint_train_fresh": not bool(train_objects & fresh_objects),
        "all_active_regions_nonempty": all(row["active_count"] > 0 for row in rows),
        "all_active_regions_local": all(
            row["active_ratio"] <= float(args.max_active_ratio) for row in rows
        ),
        "all_cross_point_counts_matched": all(
            row["point_count"] == row["cross_matched_count"] for row in rows
        ),
        "all_cross_confidence_multisets_matched": all(
            row["cross_confidence_multiset_max_abs_diff"] == 0.0 for row in rows
        ),
        "no_cross_coordinate_replacement": all(
            not row["cross_used_coordinate_replacement"] for row in rows
        ),
        "cross_controls_stay_within_split": all(
            (
                row["source_index"] < 16
                and row["cross_source_index"] < 16
                and row["matching_split"] == "train16"
            )
            or (
                16 <= row["source_index"] < 64
                and 16 <= row["cross_source_index"] < 64
                and row["matching_split"] == "fresh48"
            )
            for row in rows
        ),
    }
    manifest = {
        "format": POINT_ANCHOR_CACHE_VERSION,
        "output_dir": str(output_dir.resolve()),
        "source_cache_manifest": str(source_manifest),
        "source_manifest_sha256": config["source_manifest_sha256"],
        "config": config,
        "config_hash": config_hash,
        "evidence_names": list(POINT_EVIDENCE_NAMES),
        "control_names": list(POINT_CONTROL_NAMES),
        "sample_count": len(rows),
        "uid_hash": uid_hash(uid_values),
        "train_object_uids": sorted(train_objects),
        "fresh_object_uids": sorted(fresh_objects),
        "samples": rows,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    report = {
        "stage": "Point-only local-anchor v2 cache audit",
        "passed": all(checks.values()),
        "args": vars(args),
        "checks": checks,
        "config_hash": config_hash,
        "sample_count": len(rows),
        "failure_count": len(failures),
        "failures": failures,
        "unique_object_count": len({row["object_uid"] for row in rows}),
        "train_object_count": len(train_objects),
        "fresh_object_count": len(fresh_objects),
        "active_ratio_mean": mean(row["active_ratio"] for row in rows) if rows else 0.0,
        "active_ratio_max": max((row["active_ratio"] for row in rows), default=0.0),
        "point_count_mean": mean(row["point_count"] for row in rows) if rows else 0.0,
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
