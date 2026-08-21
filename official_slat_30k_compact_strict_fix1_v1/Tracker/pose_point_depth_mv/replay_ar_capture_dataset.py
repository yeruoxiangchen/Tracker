#!/usr/bin/env python3
"""Rebuild an immutable AR capture dataset after camera-axis contract changes."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import shutil
from typing import Any

import numpy as np

from pose_point_depth_mv.ar_object_capture import (
    ARPointFilterConfig,
    CAPTURE_FORMAT,
    filter_points_by_masks,
    fuse_ar_points,
    iter_ar_point_rows,
    mask_frame_diagnostics,
    select_image_camera_rotation,
    utc_now,
    write_json,
    write_phone_sparse_model,
    write_point_ply,
    write_points3d,
)
from pose_point_depth_mv.omni_real_benchmark_common import sha256_file
from trellis_point_prior_mv.build_ar_session_smoke_dataset import read_phone_poses


REPLAY_FORMAT = "pose_point_depth_mv.ar_capture_axis_replay.v1"


def _quality_failures(
    diagnostics: dict[str, Any],
    *,
    point_count: int,
    synchronized_fraction: float,
    config: ARPointFilterConfig,
) -> list[str]:
    failures = []
    if not diagnostics.get("available"):
        return [f"pose+mask geometry unavailable: {diagnostics.get('error')}"]
    checks = (
        (
            point_count >= config.min_object_points,
            f"mask-supported point count {point_count} < {config.min_object_points}",
        ),
        (
            diagnostics["point_to_mask_extent_ratio"]
            <= config.max_point_to_mask_extent_ratio,
            "point/mask extent ratio "
            f"{diagnostics['point_to_mask_extent_ratio']:.3f} > "
            f"{config.max_point_to_mask_extent_ratio:.3f}",
        ),
        (
            diagnostics["ray_residual_median_over_mask_extent"]
            <= config.max_ray_residual_median_over_mask_extent,
            "mask-ray median/extent "
            f"{diagnostics['ray_residual_median_over_mask_extent']:.3f} > "
            f"{config.max_ray_residual_median_over_mask_extent:.3f}",
        ),
        (
            diagnostics["ray_residual_p90_over_mask_extent"]
            <= config.max_ray_residual_p90_over_mask_extent,
            "mask-ray p90/extent "
            f"{diagnostics['ray_residual_p90_over_mask_extent']:.3f} > "
            f"{config.max_ray_residual_p90_over_mask_extent:.3f}",
        ),
        (
            diagnostics["camera_roll_median_degrees"]
            <= config.max_camera_roll_median_degrees,
            "camera roll median "
            f"{diagnostics['camera_roll_median_degrees']:.2f} > "
            f"{config.max_camera_roll_median_degrees:.2f} deg",
        ),
        (
            diagnostics["orbit_gravity_agreement"]
            >= config.min_orbit_gravity_agreement,
            "orbit/gravity agreement "
            f"{diagnostics['orbit_gravity_agreement']:.3f} < "
            f"{config.min_orbit_gravity_agreement:.3f}",
        ),
        (
            synchronized_fraction >= config.min_synchronized_frame_ratio,
            "synchronized frame fraction "
            f"{synchronized_fraction:.3f} < {config.min_synchronized_frame_ratio:.3f}",
        ),
    )
    failures.extend(message for passed, message in checks if not passed)
    return failures


def replay_capture_dataset(
    source_dataset: Path,
    output_dataset: Path,
    *,
    allow_quality_failure: bool,
) -> dict[str, Any]:
    source = Path(source_dataset).expanduser().resolve()
    destination = Path(output_dataset).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"replay output already exists: {destination}")
    source_report_path = source / "capture_report.json"
    source_report = json.loads(source_report_path.read_text(encoding="utf-8"))
    if source_report.get("format") != CAPTURE_FORMAT:
        raise RuntimeError(f"source is not an AR capture v2 dataset: {source}")
    frame_names = [str(value) for value in source_report["selected_frame_names"]]
    config_fields = ARPointFilterConfig.__dataclass_fields__
    config = ARPointFilterConfig(
        **{
            key: value
            for key, value in source_report.get("config", {}).items()
            if key in config_fields
        }
    )
    staging = destination.parent / f".{destination.name}.building"
    if staging.exists():
        raise FileExistsError(f"partial replay staging exists: {staging}")
    staging.parent.mkdir(parents=True, exist_ok=True)
    staging.mkdir()
    try:
        for directory in ("images", "rgb", "masks"):
            source_dir = source / directory
            if source_dir.is_dir():
                shutil.copytree(source_dir, staging / directory)
        for filename in ("poses.txt", "slam_points_raw.jsonl", "frame_metadata.jsonl"):
            source_path = source / filename
            if source_path.is_file():
                shutil.copy2(source_path, staging / filename)
        poses = read_phone_poses(staging / "poses.txt")
        angle, views, axis_diagnostics = select_image_camera_rotation(
            staging / "images", staging / "masks", frame_names, poses=poses
        )
        sparse, sparse_metadata = write_phone_sparse_model(
            staging,
            frame_names,
            poses,
            image_camera_rotation_degrees=angle,
            camera_axis_diagnostics=axis_diagnostics,
        )
        raw_points = staging / "slam_points_raw.jsonl"
        fused, source_confidence, temporal, fusion_stats = fuse_ar_points(
            iter_ar_point_rows(raw_points), config
        )
        filtered, confidence, colors, support_stats = filter_points_by_masks(
            fused, source_confidence, temporal, views, config
        )
        diagnostics = mask_frame_diagnostics(
            filtered, views, config, pose_mask_geometry=axis_diagnostics
        )
        synchronized_fraction = float(
            sparse_metadata["pose_binding"]["strictly_synchronized_fraction"]
        )
        failures = _quality_failures(
            diagnostics,
            point_count=len(filtered),
            synchronized_fraction=synchronized_fraction,
            config=config,
        )
        if failures and not allow_quality_failure:
            raise RuntimeError("; ".join(failures))
        write_points3d(sparse / "points3D.txt", filtered, colors, confidence)
        write_point_ply(sparse / "object_points.ply", filtered, colors)
        np.savez_compressed(
            sparse / "object_points.npz",
            P_W=filtered.astype(np.float32),
            rgb=colors,
            confidence=confidence.astype(np.float32),
        )
        report = {
            "format": CAPTURE_FORMAT,
            "replay_format": REPLAY_FORMAT,
            "created_at_utc": utc_now(),
            "session_id": destination.name,
            "passed": True,
            "formal_input_passed": not failures,
            "diagnostic_override_used": bool(failures and allow_quality_failure),
            "quality_failures": failures,
            "dataset_dir": str(destination),
            "selected_frame_count": len(frame_names),
            "selected_frame_names": frame_names,
            "source_dataset": str(source),
            "source_capture_report": str(source_report_path),
            "source_capture_report_sha256": sha256_file(source_report_path),
            "config": asdict(config),
            "fusion": fusion_stats,
            "mask_support": support_stats,
            "geometry_diagnostics": diagnostics,
            "sparse_metadata": sparse_metadata,
            "scope_guard": (
                "Deterministic replay of frozen RGB/mask/AR pose/intrinsics/point cloud; "
                "no generated mesh, ReconViaGen output, or GT geometry consumed."
            ),
        }
        write_json(staging / "capture_report.json", report)
        staging.replace(destination)
        return report
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_dataset", required=True)
    parser.add_argument("--output_dataset", required=True)
    parser.add_argument("--allow_quality_failure", action="store_true")
    args = parser.parse_args()
    report = replay_capture_dataset(
        Path(args.source_dataset),
        Path(args.output_dataset),
        allow_quality_failure=bool(args.allow_quality_failure),
    )
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "formal_input_passed": report["formal_input_passed"],
                "diagnostic_override_used": report["diagnostic_override_used"],
                "quality_failures": report["quality_failures"],
                "dataset_dir": report["dataset_dir"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
