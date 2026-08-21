#!/usr/bin/env python3
"""Build a deterministic pose+mask runtime-O subset without reading point clouds."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any

import numpy as np

from ar_ss_flow.shared_object_preprocessing import canonical_json_sha256
from pose_point_depth_mv.dataset_tools.prepare_omni_real_runtime_inputs import (
    MANIFEST_FORMAT,
    _camera_rows_by_name,
    _load_rgb_mask,
)
from pose_point_depth_mv.dataset_tools.prepare_omni_real_video_cache import (
    RAW_CACHE_FORMAT,
    sha256_file,
    utc_now,
    write_json,
    write_npz,
)
from pose_point_depth_mv.omni_real_benchmark_common import index_objects, object_key
from pose_point_depth_mv.evaluate_omni_real_native_adaptation import (
    _formal_holdout_binding,
)
from pose_point_depth_mv.pose_mask_object_canonicalization import (
    POSE_MASK_INPUT_FRONTEND_VERSION,
    PoseMaskObjectFrameConfig,
    prepare_pose_mask_runtime_object_observation,
)
from pose_point_depth_mv.real_object_canonicalization import array_sha256


OBJECT_FORMAT = "pose_point_depth_mv.omni_real_pose_mask_runtime_input_object.v1"
MARKER_FORMAT = "pose_point_depth_mv.omni_real_pose_mask_runtime_input_marker.v1"
MANIFEST_VARIANT = "pose_point_depth_mv.omni_real_pose_mask_runtime_ablation.v1"
SPLIT_FORMAT = "pose_point_depth_mv.hash_ranked_benchmark_subset.v1"
PROTOCOL_SCOPES = (
    "benchmark32_development",
    "coarsemodel_real_qualitative",
    "formal_holdout64_blind_addendum",
)
FORMAL_HOLDOUT_OBJECT_COUNT = 64


def validate_protocol_object_sets(
    raw_keys: set[str], reference_keys: set[str], *, protocol_scope: str
) -> None:
    if protocol_scope not in PROTOCOL_SCOPES:
        raise ValueError(f"unsupported protocol_scope={protocol_scope}")
    missing = reference_keys.difference(raw_keys)
    if missing:
        raise RuntimeError(f"reference runtime objects missing from raw cache: {sorted(missing)}")
    if protocol_scope == "benchmark32_development" and raw_keys != reference_keys:
        raise RuntimeError("raw and reference Benchmark32 object sets differ")
    if (
        protocol_scope == "formal_holdout64_blind_addendum"
        and raw_keys != reference_keys
    ):
        raise RuntimeError("raw and reference formal Holdout64 object sets differ")


def deterministic_subset_keys(
    object_keys: list[str], *, count: int, seed: int, offset: int = 0
) -> tuple[list[str], list[dict[str, Any]]]:
    """Select a stable subset independent of source-manifest row order."""

    keys = [str(value) for value in object_keys]
    if len(keys) != len(set(keys)):
        raise ValueError("object_keys must be unique")
    if int(count) <= 0:
        raise ValueError("subset_count must be positive")
    if int(offset) < 0 or int(offset) + int(count) > len(keys):
        raise ValueError(
            f"subset rank slice [{int(offset)},{int(offset) + int(count)}) "
            f"must fit in [0,{len(keys)})"
        )
    ranking = []
    for key in keys:
        digest = hashlib.sha256(f"{int(seed)}|{key}".encode("utf-8")).hexdigest()
        ranking.append({"object_key": key, "rank_sha256": digest})
    ranking.sort(key=lambda row: (row["rank_sha256"], row["object_key"]))
    selected = [
        row["object_key"]
        for row in ranking[int(offset) : int(offset) + int(count)]
    ]
    return selected, ranking


def _load_reusable(
    destination: Path,
    *,
    source_cache_sha256: str,
    reference_report_sha256: str,
    build_config_sha256: str,
) -> dict[str, Any] | None:
    marker_path = destination / "_POSE_MASK_RUNTIME_INPUT_COMPLETE.json"
    report_path = destination / "report.json"
    if not marker_path.is_file() or not report_path.is_file():
        return None
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    expected = {
        "format": MARKER_FORMAT,
        "source_raw_cache_sha256": source_cache_sha256,
        "reference_runtime_report_sha256": reference_report_sha256,
        "build_config_sha256": build_config_sha256,
    }
    if any(marker.get(key) != value for key, value in expected.items()):
        raise RuntimeError(f"stale pose+mask runtime output: {destination}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    cache = Path(str(report.get("cache_npz", "")))
    condition = Path(str(report.get("condition_record", "")))
    if (
        report.get("format") != OBJECT_FORMAT
        or report.get("passed") is not True
        or report.get("point_cloud_consumed") is not False
        or not cache.is_file()
        or sha256_file(cache) != report.get("cache_npz_sha256")
        or not condition.is_file()
    ):
        raise RuntimeError(f"invalid reusable pose+mask runtime output: {destination}")
    return report


def build_object_pose_mask_runtime_input(
    raw_row: dict[str, Any],
    reference_row: dict[str, Any],
    *,
    output_dir: Path,
    frame_config: PoseMaskObjectFrameConfig,
    feature_resolution: int,
    foreground_margin: float,
    alpha_threshold: float,
    build_config_sha256: str,
    resume_partial: bool,
    scope_guard: str = (
        "Benchmark32 development ablation input only. O uses calibrated poses "
        "and masks; P_W, Scan, alignment, labels, and target latents are not read."
    ),
) -> tuple[dict[str, Any], bool]:
    key = object_key(raw_row)
    if object_key(reference_row) != key:
        raise RuntimeError(f"raw/reference object mismatch: {key}")
    source_cache = Path(raw_row["cache_npz"]).resolve()
    source_hash = sha256_file(source_cache)
    if (
        Path(str(reference_row["source_raw_cache"])).resolve() != source_cache
        or str(reference_row["source_raw_cache_sha256"]) != source_hash
    ):
        raise RuntimeError(f"reference runtime source cache differs: {key}")
    reference_report_path = Path(reference_row["cache_npz"]).resolve().parent / "report.json"
    reference_report_hash = sha256_file(reference_report_path)
    destination = output_dir / "objects" / raw_row["category"] / raw_row["object_id"]
    reusable = _load_reusable(
        destination,
        source_cache_sha256=source_hash,
        reference_report_sha256=reference_report_hash,
        build_config_sha256=build_config_sha256,
    )
    if reusable is not None:
        return reusable, True
    if destination.exists():
        raise RuntimeError(f"partial pose+mask runtime output exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / f".{raw_row['object_id']}.pose-mask-building"
    if staging.exists():
        if not resume_partial:
            raise RuntimeError(f"partial pose+mask staging exists: {staging}")
        shutil.rmtree(staging)
    staging.mkdir()

    try:
        selected_indices = np.asarray(
            reference_row["selected_source_view_indices"], dtype=np.int64
        )
        reference_frame_names = [str(value) for value in reference_row["selected_frame_names"]]
        reference_view_index = int(reference_row["reference_view_index"])
        with np.load(source_cache, allow_pickle=False) as source:
            # Deliberately load only names, intrinsics, and calibrated camera poses.
            frame_names_all = [str(value) for value in source["frame_name"].tolist()]
            if np.any(selected_indices < 0) or np.any(selected_indices >= len(frame_names_all)):
                raise RuntimeError(f"reference view indices are out of range: {key}")
            frame_names = [frame_names_all[int(index)] for index in selected_indices]
            if frame_names != reference_frame_names:
                raise RuntimeError(f"reference frame names changed in raw cache: {key}")
            K = np.asarray(source["K"][selected_indices], dtype=np.float64)
            T_W2C = np.asarray(source["T_W2C"][selected_indices], dtype=np.float64)

        camera_by_name = _camera_rows_by_name(raw_row)
        missing = [name for name in frame_names if name not in camera_by_name]
        if missing:
            raise RuntimeError(f"missing camera metadata for {key}: {missing}")
        cameras = [camera_by_name[name] for name in frame_names]
        images, masks = _load_rgb_mask(
            Path(raw_row["images_dir"]), Path(raw_row["masks_dir"]), frame_names
        )
        observation = prepare_pose_mask_runtime_object_observation(
            images,
            masks,
            K,
            T_W2C,
            camera_models=[str(camera["model"]) for camera in cameras],
            distortion_coefficients=[camera.get("distortion", []) for camera in cameras],
            frame_config=frame_config,
            reference_view_index=reference_view_index,
            feature_resolution=int(feature_resolution),
            foreground_margin=float(foreground_margin),
            alpha_threshold=float(alpha_threshold),
        )

        reference_condition = json.loads(
            Path(reference_row["condition_record"]).read_text(encoding="utf-8")
        )
        if reference_condition.get("condition_sha256") != reference_row.get(
            "condition_sha256"
        ):
            raise RuntimeError(f"reference condition identity changed: {key}")
        visible_fields = (
            "prepared_rgb_sha256",
            "prepared_mask_sha256",
            "K_feature_sha256",
        )
        visible_equivalence = {
            field: observation.condition_record[field] == reference_condition[field]
            for field in visible_fields
        }
        if not all(visible_equivalence.values()):
            raise RuntimeError(f"pose+mask changed externally visible image inputs: {key}")

        cache_name = "runtime_input_cache.npz"
        T_O2C = np.asarray(observation.frame.T_O2C, dtype=np.float64)
        T_O2C_lifting = np.asarray(
            observation.frame.T_O2C_lifting, dtype=np.float64
        )
        write_npz(
            staging / cache_name,
            selected_source_view_index=selected_indices,
            frame_name=np.asarray(frame_names),
            K_feature=np.asarray(observation.intrinsics, dtype=np.float32),
            T_O2C=T_O2C,
            T_O2C_lifting=T_O2C_lifting,
            T_C2O=np.asarray(observation.frame.T_C2O, dtype=np.float64),
            T_O2W=np.asarray(observation.frame.T_O2W, dtype=np.float64),
            T_W2O=np.asarray(observation.frame.T_W2O, dtype=np.float64),
            P_O=np.empty((0, 3), dtype=np.float32),
            object_point_source_index=np.empty((0,), dtype=np.int64),
            source_to_feature_affine=np.asarray(
                observation.prepared_views.source_to_feature_affines, dtype=np.float32
            ),
        )
        condition_name = "condition_record.json"
        write_json(staging / condition_name, observation.condition_record)
        report = {
            "format": OBJECT_FORMAT,
            "created_at_utc": utc_now(),
            "category": str(raw_row["category"]),
            "object_id": str(raw_row["object_id"]),
            "object_key": key,
            "source_raw_cache": str(source_cache),
            "source_raw_cache_sha256": source_hash,
            "reference_runtime_report": str(reference_report_path),
            "reference_runtime_report_sha256": reference_report_hash,
            "build_config_sha256": build_config_sha256,
            "input_frontend_format": POSE_MASK_INPUT_FRONTEND_VERSION,
            "selected_view_count": len(selected_indices),
            "selected_source_view_indices": selected_indices.tolist(),
            "selected_frame_names": frame_names,
            "view_selection": reference_row.get("view_selection"),
            "reference_view_index": reference_view_index,
            "cache_npz": str((destination / cache_name).resolve()),
            "cache_npz_sha256": sha256_file(staging / cache_name),
            "condition_record": str((destination / condition_name).resolve()),
            "condition_sha256": observation.condition_sha256,
            "T_O2C_sha256": array_sha256(T_O2C),
            "T_O2C_lifting_sha256": array_sha256(T_O2C_lifting),
            "lifting_extrinsics_policy": reference_row["lifting_extrinsics_policy"],
            # Reuse the exact image artifacts from A so image bytes cannot drift.
            "prepared_rgb_paths": list(reference_row["prepared_rgb_paths"]),
            "prepared_mask_paths": list(reference_row["prepared_mask_paths"]),
            "external_visible_equivalence": visible_equivalence,
            "runtime_frame_stats": observation.frame.stats,
            "point_cloud_consumed": False,
            "point_cloud_fields_read": [],
            "gt_consumed": False,
            "old_mesh_consumed": False,
            "metric_or_ranking_consumed": False,
            "forbidden_gt_fields_absent": True,
            "training_ready": False,
            "scope_guard": str(scope_guard),
            "passed": True,
        }
        write_json(staging / "report.json", report)
        write_json(
            staging / "_POSE_MASK_RUNTIME_INPUT_COMPLETE.json",
            {
                "format": MARKER_FORMAT,
                "completed_at_utc": utc_now(),
                "object_key": key,
                "source_raw_cache_sha256": source_hash,
                "reference_runtime_report_sha256": reference_report_hash,
                "build_config_sha256": build_config_sha256,
                "condition_sha256": observation.condition_sha256,
                "point_cloud_consumed": False,
                "gt_consumed": False,
                "old_mesh_consumed": False,
                "metric_or_ranking_consumed": False,
            },
        )
        staging.replace(destination)
        return report, False
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw_cache_report", required=True)
    parser.add_argument("--reference_runtime_manifest", required=True)
    parser.add_argument("--frozen_split_manifest", default="")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--protocol_scope",
        choices=PROTOCOL_SCOPES,
        default="benchmark32_development",
    )
    parser.add_argument("--subset_count", type=int, default=16)
    parser.add_argument("--subset_offset", type=int, default=0)
    parser.add_argument("--subset_seed", type=int, default=20260810)
    parser.add_argument("--selected_view_count", type=int, default=8)
    parser.add_argument("--feature_resolution", type=int, default=518)
    parser.add_argument("--foreground_margin", type=float, default=1.10)
    parser.add_argument("--alpha_threshold", type=float, default=0.80)
    parser.add_argument("--mask_threshold", type=float, default=0.5)
    parser.add_argument("--expected_object_extent", type=float, default=0.90)
    parser.add_argument("--scale_padding", type=float, default=1.05)
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> None:
    args = make_parser().parse_args()
    raw_path = Path(args.raw_cache_report).expanduser().resolve()
    reference_path = Path(args.reference_runtime_manifest).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    if raw.get("format") != RAW_CACHE_FORMAT or raw.get("passed") is not True:
        raise RuntimeError(f"raw cache report did not pass: {raw_path}")
    if reference.get("format") != MANIFEST_FORMAT or reference.get("passed") is not True:
        raise RuntimeError(f"reference runtime manifest did not pass: {reference_path}")
    if (
        Path(str(reference["raw_cache_report"])).resolve() != raw_path
        or str(reference["raw_cache_report_sha256"]) != sha256_file(raw_path)
    ):
        raise RuntimeError("reference runtime is not bound to the requested raw cache")

    raw_by_key = index_objects(raw.get("objects", []), label="raw source objects")
    reference_by_key = index_objects(
        reference.get("objects", []), label="reference runtime objects"
    )
    validate_protocol_object_sets(
        set(raw_by_key),
        set(reference_by_key),
        protocol_scope=str(args.protocol_scope),
    )
    formal_binding = None
    if args.protocol_scope == "formal_holdout64_blind_addendum":
        if (
            int(args.subset_count) != FORMAL_HOLDOUT_OBJECT_COUNT
            or int(args.subset_offset) != 0
            or not args.frozen_split_manifest
        ):
            raise ValueError(
                "formal Holdout64 requires subset_count=64, subset_offset=0, "
                "and --frozen_split_manifest"
            )
        if len(reference_by_key) != FORMAL_HOLDOUT_OBJECT_COUNT:
            raise RuntimeError("formal reference runtime must contain exactly 64 objects")
        selected_keys = [object_key(row) for row in reference.get("objects", [])]
        if len(selected_keys) != len(set(selected_keys)):
            raise RuntimeError("formal reference runtime order contains duplicate objects")
        full_ranking = [
            {"object_key": key, "reference_runtime_position": position}
            for position, key in enumerate(selected_keys)
        ]
        formal_binding = _formal_holdout_binding(
            split_path=Path(args.frozen_split_manifest),
            label_keys=set(selected_keys),
            runtime_path=reference_path,
            runtime=reference,
            expected_objects=FORMAL_HOLDOUT_OBJECT_COUNT,
        )
    else:
        if args.frozen_split_manifest:
            raise ValueError("frozen split is only valid for formal Holdout64")
        selected_keys, full_ranking = deterministic_subset_keys(
            list(reference_by_key),
            count=int(args.subset_count),
            seed=int(args.subset_seed),
            offset=int(args.subset_offset),
        )
    if any(
        int(reference_by_key[key]["selected_view_count"])
        != int(args.selected_view_count)
        for key in selected_keys
    ):
        raise RuntimeError("reference runtime selected_view_count differs from request")

    frame_config = PoseMaskObjectFrameConfig(
        mask_threshold=float(args.mask_threshold),
        expected_object_extent=float(args.expected_object_extent),
        scale_padding=float(args.scale_padding),
    )
    frame_config.validate()
    if args.protocol_scope == "benchmark32_development":
        object_scope_guard = (
            "Benchmark32 development ablation input only. O uses calibrated poses "
            "and masks; P_W, Scan, alignment, labels, and target latents are not read."
        )
        manifest_scope_guard = (
            "Benchmark32 subset development ablation. This compatibility-format "
            "runtime manifest contains pose+mask O caches and no point-cloud input."
        )
    elif args.protocol_scope == "coarsemodel_real_qualitative":
        object_scope_guard = (
            "CoarseModel local real-capture qualitative input. O uses only calibrated "
            "poses, K, and masks; P_W, old meshes, labels, metrics, and target latents "
            "are not read."
        )
        manifest_scope_guard = (
            "CoarseModel local real-capture qualitative Pose+Mask runtime. No point "
            "cloud or reference mesh is consumed; the point-based runtime supplies "
            "only the frozen object/view list and externally visible image artifacts."
        )
    else:
        object_scope_guard = (
            "Formal Holdout64 blind Pose+Mask addendum input. O uses only frozen "
            "calibrated poses, K, masks, and the reference view list; P_W, labels, "
            "old meshes, metrics, rankings, and target latents are not read."
        )
        manifest_scope_guard = (
            "Formal Holdout64 blind addendum. Exact M11C object and view order; no "
            "point cloud, GT, old mesh, metric, ranking, or target is consumed."
        )
    split = {
        "format": SPLIT_FORMAT,
        "protocol_scope": str(args.protocol_scope),
        "source_object_count": len(reference_by_key),
        "subset_count": len(selected_keys),
        "subset_offset": int(args.subset_offset),
        "subset_seed": int(args.subset_seed),
        "rank_rule": (
            "exact reference-runtime object order"
            if formal_binding is not None
            else "sha256(f'{subset_seed}|{object_key}') ascending, then "
            "rank[subset_offset:subset_offset+subset_count]"
        ),
        "selected_object_keys": selected_keys,
        "full_ranking": full_ranking,
        "reference_runtime_manifest": str(reference_path),
        "reference_runtime_manifest_sha256": sha256_file(reference_path),
        "formal_holdout_binding": formal_binding,
    }
    split["split_sha256"] = canonical_json_sha256(split)
    build_config = {
        "protocol_scope": str(args.protocol_scope),
        "input_frontend_format": POSE_MASK_INPUT_FRONTEND_VERSION,
        "selected_view_count": int(args.selected_view_count),
        "view_selection": "exactly_reuse_reference_runtime_selected_views",
        "reference_view": "exactly_reuse_reference_runtime_reference_view",
        "feature_resolution": int(args.feature_resolution),
        "foreground_margin": float(args.foreground_margin),
        "alpha_threshold": float(args.alpha_threshold),
        "frame_config": asdict(frame_config),
        "point_cloud_consumed": False,
        "gt_consumed": False,
        "old_mesh_consumed": False,
        "metric_or_ranking_consumed": False,
        "subset_sha256": split["split_sha256"],
    }
    build_hash = canonical_json_sha256(build_config)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "ablation_split.json", split)

    reports = []
    reused = []
    failures = []
    for position, key in enumerate(selected_keys, start=1):
        print(f"[pose_mask_runtime] {position}/{len(selected_keys)} object={key}", flush=True)
        try:
            report, was_reused = build_object_pose_mask_runtime_input(
                raw_by_key[key],
                reference_by_key[key],
                output_dir=output_dir,
                frame_config=frame_config,
                feature_resolution=int(args.feature_resolution),
                foreground_margin=float(args.foreground_margin),
                alpha_threshold=float(args.alpha_threshold),
                build_config_sha256=build_hash,
                resume_partial=bool(args.resume),
                scope_guard=object_scope_guard,
            )
            reports.append(report)
            if was_reused:
                reused.append(key)
            print(
                f"[pose_mask_runtime] object={key} "
                f"scale={report['runtime_frame_stats']['scale_O2W']:.6g} "
                f"reused={was_reused}",
                flush=True,
            )
        except Exception as error:
            failures.append({"object_key": key, "error": repr(error)})
            print(f"[pose_mask_runtime] FAILED object={key}: {error!r}", flush=True)
            raise

    manifest = {
        "format": MANIFEST_FORMAT,
        "manifest_variant": MANIFEST_VARIANT,
        "created_at_utc": utc_now(),
        "raw_cache_report": str(raw_path),
        "raw_cache_report_sha256": sha256_file(raw_path),
        "source_raw_cache_report": str(raw_path),
        "source_raw_cache_report_sha256": sha256_file(raw_path),
        "reference_runtime_manifest": str(reference_path),
        "reference_runtime_manifest_sha256": sha256_file(reference_path),
        "ablation_split": str((output_dir / "ablation_split.json").resolve()),
        "ablation_split_sha256": sha256_file(output_dir / "ablation_split.json"),
        "build_config": build_config,
        "build_config_sha256": build_hash,
        "source_selected_object_count": len(selected_keys),
        "selected_object_count": len(reports),
        "completed_object_count": len(reports),
        "reused_objects": reused,
        "objects": reports,
        "failures": failures,
        "point_cloud_consumed": False,
        "gt_consumed": False,
        "old_mesh_consumed": False,
        "metric_or_ranking_consumed": False,
        "training_ready": False,
        "formal": formal_binding is not None,
        "formal_holdout_binding": formal_binding,
        "protocol_scope": str(args.protocol_scope),
        "scope_guard": manifest_scope_guard,
        "passed": len(reports) == len(selected_keys) and not failures,
    }
    manifest_path = output_dir / "runtime_input_manifest.json"
    write_json(manifest_path, manifest)
    print(
        json.dumps(
            {
                "passed": manifest["passed"],
                "selected_object_count": len(selected_keys),
                "point_cloud_consumed": False,
                "manifest": str(manifest_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
