#!/usr/bin/env python3
"""Phone AR reconstruction with two deliberately filter-free 8-frame policies.

This entry point keeps :mod:`pose_point_depth_mv.server` unchanged and reuses
its upload/SAM2/reconstruction endpoints.  The only frame decisions are:

* ``trajectory_uniform8``: eight chronological frames at uniform cumulative
  camera-translation distances;
* ``time_uniform8``: eight chronological frames at uniform frame positions.

No mask score, view angle, azimuth/elevation coverage, camera roll, pose jump,
mask-ray residual, client selection, or point-cloud statistic may replace or
remove a selected frame.  Missing RGB/mask/pose data can still make the fixed
input impossible to materialize; that is an input contract error, not a frame
selection fallback.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import threading
from typing import Any, Mapping, Sequence

import cv2
import numpy as np


TRACKER_ROOT = Path(__file__).resolve().parents[1]
if str(TRACKER_ROOT) not in sys.path:
    sys.path.insert(0, str(TRACKER_ROOT))

DEFAULT_OUTPUT_ROOT = (
    TRACKER_ROOT
    / "pose_point_depth_mv"
    / "outputs2"
    / "手机AR无筛选_位移轨迹与时间均匀8帧"
)
DEFAULT_GPU = os.environ.get("AR_OBJECT_GPU", os.environ.get("GPU", "4"))
SELECTED_VIEW_COUNT = 8
SELECTION_MODES = ("trajectory_uniform8", "time_uniform8")
FORMAT = "pose_point_depth_mv.server2_filter_free_dual_uniform8.v2"
BRANCH_FORMAT = "pose_point_depth_mv.server2_filter_free_branch.v2"
CAPTURE_FORMAT = "pose_point_depth_mv.server2_ar_axis_corrected_capture.v1"
AR_IMAGE_AXIS_CONTRACT = "xrcpuimage_none_xy_transpose_base_unity_cv.v1"
AR_IMAGE_AXIS_SESSION_SUFFIX = "araxis_xytranspose_v1"
_SAFE_SESSION_ID = re.compile(r"^[A-Za-z0-9_.-]+$")

# Configure the direct-launch process before importing CUDA/model modules.
if __name__ == "__main__":
    environment_bin = str(Path(sys.executable).resolve().parent)
    inherited_path = os.environ.get("PATH", "")
    os.environ["PATH"] = os.pathsep.join(
        part for part in (environment_bin, inherited_path) if part
    )
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", DEFAULT_GPU)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    # server2 records diagnostics but never turns them into an input-quality gate.
    os.environ["RECON_ENFORCE_INPUT_QC"] = "0"
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    os.environ.setdefault("ATTN_BACKEND", "flash_attn")
    os.environ.setdefault("SPCONV_ALGO", "native")

from flask import jsonify, request

from pose_point_depth_mv import ar_object_capture as capture_tools
from pose_point_depth_mv import ar_object_reconstruction as reconstruction_tools
from pose_point_depth_mv import server as base
from pose_point_depth_mv.ar_object_capture import (
    ARPointFilterConfig,
    read_phone_poses,
    utc_now,
    write_json,
)
from pose_point_depth_mv.reconstruct_real_proobjaverse_official_ss_slat import (
    RuntimeConfig as OfficialReconstructionConfig,
)
from trellis_point_prior_mv.build_ar_session_smoke_dataset import (
    intrinsics_for_pose,
    rotmat_to_qvec,
    unity_pose_to_colmap_w2c,
)


_ACTIVE_SELECTION_MODES: tuple[str, ...] = SELECTION_MODES
_BACKGROUND_BRANCH_THREADS: dict[str, threading.Thread] = {}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def corrected_native_image_intrinsics(
    pose: Mapping[str, Any],
    raw_width: int,
    raw_height: int,
) -> dict[str, Any]:
    """Return K after physically transposing a native CPU image.

    The source ARFoundation intrinsics belong to the unmodified
    ``XRCpuImage.Transformation.None`` pixel array.  A physical x/y transpose
    maps ``(u, v) -> (v, u)`` and therefore swaps focal lengths, principal
    points, and image dimensions together.
    """

    raw_width = int(raw_width)
    raw_height = int(raw_height)
    if raw_width <= 0 or raw_height <= 0:
        raise ValueError("raw image dimensions must be positive")
    fx, fy, cx, cy, source = intrinsics_for_pose(
        dict(pose), raw_width, raw_height
    )
    values = np.asarray([fx, fy, cx, cy], dtype=np.float64)
    if not np.isfinite(values).all() or fx <= 0.0 or fy <= 0.0:
        raise ValueError("source AR intrinsics are invalid")
    return {
        "width": raw_height,
        "height": raw_width,
        "fx": float(fy),
        "fy": float(fx),
        "cx": float(cy),
        "cy": float(cx),
        "source": str(source),
        "source_width": raw_width,
        "source_height": raw_height,
        "source_fx": float(fx),
        "source_fy": float(fy),
        "source_cx": float(cx),
        "source_cy": float(cy),
    }


def _read_axis_corrected_frame(
    image_path: Path,
    mask_path: Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Load one native frame and materialize the audited x/y transpose."""

    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if image_bgr is None:
        raise FileNotFoundError(f"cannot read uploaded RGB frame: {image_path}")
    if mask is None:
        raise FileNotFoundError(f"cannot read uploaded mask: {mask_path}")
    raw_height, raw_width = image_bgr.shape[:2]
    source_mask_shape = [int(value) for value in mask.shape]
    mask_resized = tuple(mask.shape) != (raw_height, raw_width)
    if mask_resized:
        mask = cv2.resize(
            mask,
            (raw_width, raw_height),
            interpolation=cv2.INTER_NEAREST,
        )
    corrected_bgr = np.ascontiguousarray(np.transpose(image_bgr, (1, 0, 2)))
    corrected_mask = np.ascontiguousarray(mask.T)
    return corrected_bgr, corrected_mask, {
        "source_image_size_wh": [int(raw_width), int(raw_height)],
        "source_mask_shape_hw": source_mask_shape,
        "mask_resized_to_source_image": bool(mask_resized),
        "materialized_image_size_wh": [int(raw_height), int(raw_width)],
        "pixel_transform": "transpose_xy: (u_materialized,v_materialized)=(v_raw,u_raw)",
    }


def build_axis_corrected_projection_views(
    data_dir: Path,
    mask_dir: Path,
    frame_names: Sequence[str],
    *,
    poses: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[list[capture_tools.ProjectionView], list[dict[str, Any]]]:
    """Build server2 views without using ``displayMatrix`` as an extrinsic.

    This helper is shared by capture materialization and regression diagnostics.
    It does not write files and never selects or replaces a frame.
    """

    resolved_poses = dict(poses or read_phone_poses(data_dir / "poses.txt"))
    views: list[capture_tools.ProjectionView] = []
    records: list[dict[str, Any]] = []
    for frame_name in frame_names:
        if frame_name not in resolved_poses:
            raise FileNotFoundError(f"phone pose missing for selected frame: {frame_name}")
        pose = resolved_poses[frame_name]
        image_transform = str(pose.get("image_transform") or "unknown")
        if image_transform.lower() != "none":
            raise RuntimeError(
                "server2 AR-axis correction requires raw "
                "XRCpuImage.Transformation.None; "
                f"frame={frame_name} transform={image_transform!r}"
            )
        image_path = data_dir / frame_name
        mask_path = mask_dir / f"{Path(frame_name).stem}.png"
        corrected_bgr, corrected_mask, transform_record = (
            _read_axis_corrected_frame(image_path, mask_path)
        )
        raw_width, raw_height = transform_record["source_image_size_wh"]
        intrinsics = corrected_native_image_intrinsics(
            pose, raw_width, raw_height
        )
        K = np.asarray(
            [
                [intrinsics["fx"], 0.0, intrinsics["cx"]],
                [0.0, intrinsics["fy"], intrinsics["cy"]],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        rotation, translation = unity_pose_to_colmap_w2c(
            dict(pose), image_camera_rotation_degrees=0.0
        )
        T_W2C = np.eye(4, dtype=np.float64)
        T_W2C[:3, :3] = rotation
        T_W2C[:3, 3] = translation
        views.append(
            capture_tools.ProjectionView(
                frame_name=str(frame_name),
                image_path=image_path,
                mask_path=mask_path,
                K=K,
                T_W2C=T_W2C,
                image_rgb=corrected_bgr[:, :, ::-1],
                mask=corrected_mask,
            )
        )
        records.append(
            {
                "frame_name": str(frame_name),
                **transform_record,
                "intrinsics": intrinsics,
                "pose_camera_rotation_degrees": 0.0,
                "display_matrix_role": "screen_rendering_metadata_only",
                "screen_orientation": str(
                    pose.get("screen_orientation") or "unknown"
                ),
                "tracking_state": str(pose.get("tracking_state") or "unknown"),
                "strictly_synchronized": bool(pose.get("strictly_synchronized")),
            }
        )
    return views, records


def _write_axis_corrected_sparse_model(
    dataset_dir: Path,
    views: Sequence[capture_tools.ProjectionView],
    poses: Mapping[str, Mapping[str, Any]],
    frame_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    sparse = dataset_dir / "sparse" / "0"
    sparse.mkdir(parents=True, exist_ok=True)
    camera_lines: list[str] = []
    image_lines: list[str] = []
    intrinsics_sources: dict[str, int] = {}
    for image_id, (view, record) in enumerate(
        zip(views, frame_records), start=1
    ):
        intrinsics = dict(record["intrinsics"])
        width = int(intrinsics["width"])
        height = int(intrinsics["height"])
        if view.mask.shape != (height, width):
            raise RuntimeError(
                f"materialized mask/K dimensions differ: {view.frame_name}"
            )
        source = str(intrinsics["source"])
        intrinsics_sources[source] = intrinsics_sources.get(source, 0) + 1
        camera_lines.append(
            f"{image_id} PINHOLE {width} {height} "
            f"{intrinsics['fx']:.10f} {intrinsics['fy']:.10f} "
            f"{intrinsics['cx']:.10f} {intrinsics['cy']:.10f}\n"
        )
        quaternion = rotmat_to_qvec(view.T_W2C[:3, :3])
        translation = view.T_W2C[:3, 3]
        image_lines.append(
            f"{image_id} {quaternion[0]:.12f} {quaternion[1]:.12f} "
            f"{quaternion[2]:.12f} {quaternion[3]:.12f} "
            f"{translation[0]:.12f} {translation[1]:.12f} "
            f"{translation[2]:.12f} {image_id} {view.frame_name}\n"
            "0.0 0.0 -1\n"
        )
    (sparse / "cameras.txt").write_text(
        "# Camera list with one line of data per camera:\n"
        "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n"
        f"# Number of cameras: {len(camera_lines)}\n"
        + "".join(camera_lines),
        encoding="utf-8",
    )
    (sparse / "images.txt").write_text(
        "# Image list with two lines of data per image:\n"
        "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n"
        "#   POINTS2D[] as (X, Y, POINT3D_ID)\n"
        f"# Number of images: {len(image_lines)}\n"
        + "".join(image_lines),
        encoding="utf-8",
    )
    empty_points = np.empty((0, 3), dtype=np.float64)
    empty_colors = np.empty((0, 3), dtype=np.uint8)
    empty_confidence = np.empty((0,), dtype=np.float64)
    capture_tools.write_points3d(
        sparse / "points3D.txt", empty_points, empty_colors, empty_confidence
    )
    capture_tools.write_point_ply(
        sparse / "object_points.ply", empty_points, empty_colors
    )
    np.savez_compressed(
        sparse / "object_points.npz",
        P_W=empty_points.astype(np.float32),
        rgb=empty_colors,
        confidence=empty_confidence.astype(np.float32),
    )
    selected_names = [view.frame_name for view in views]
    metadata = {
        "format": "pose_point_depth_mv.server2_phone_pose_meta.v1",
        "created_at_utc": utc_now(),
        "pose_source": "Unity ARFoundation camera transform",
        "point_source": "none_pose_mask_only",
        "num_images": len(views),
        "intrinsics_sources": intrinsics_sources,
        "image_camera_rotation_degrees": 0.0,
        "image_axis_contract": {
            "format": AR_IMAGE_AXIS_CONTRACT,
            "source_storage": "XRCpuImage.Transformation.None",
            "materialized_pixel_transform": "transpose_xy",
            "intrinsics_transform": "(fx,fy,cx,cy,W,H)->(fy,fx,cy,cx,H,W)",
            "extrinsic_transform": "base Unity-to-CV only; no displayMatrix Rz",
            "display_matrix_role": "screen_rendering_metadata_only",
        },
        "pose_binding": {
            "strictly_synchronized_count": int(
                sum(bool(poses[name].get("strictly_synchronized")) for name in selected_names)
            ),
            "selected_frame_count": int(len(selected_names)),
            "strictly_synchronized_fraction": float(
                sum(bool(poses[name].get("strictly_synchronized")) for name in selected_names)
                / max(len(selected_names), 1)
            ),
            "binding_by_frame": {
                name: str(poses[name].get("pose_binding") or "legacy_unversioned")
                for name in selected_names
            },
            "timestamp_delta_seconds_by_frame": {
                name: poses[name].get("camera_frame_timestamp_delta_s")
                for name in selected_names
            },
        },
        "coordinate_conversion": {
            "world": "diag(1,1,-1) * unity_world",
            "camera": "Unity pose camera converted to CV x-right y-down z-forward",
            "cpu_image_camera_from_pose_camera": (
                "physical RGB/mask transpose absorbs the audited native-image "
                "x/y swap; sparse T_W2C uses Rz(0deg)"
            ),
        },
        "per_frame": [dict(record) for record in frame_records],
    }
    write_json(sparse / "phone_pose_meta.json", metadata)
    return metadata


def finalize_axis_corrected_ar_capture(
    *,
    session_id: str,
    data_dir: Path,
    mask_dir: Path,
    frame_names: Sequence[str],
    output_root: Path,
    input_qc: Mapping[str, Any],
    config: ARPointFilterConfig,
) -> dict[str, Any]:
    """Create an immutable pose-mask dataset under the server2 axis contract."""

    config.validate()
    if len(frame_names) < 2:
        raise ValueError("at least two selected frames are required")
    destination = output_root / "datasets" / session_id
    report_path = destination / "capture_report.json"
    if report_path.is_file():
        report = _load_json(report_path)
        if (
            report.get("passed") is True
            and report.get("format") == CAPTURE_FORMAT
            and (report.get("image_axis_contract") or {}).get("format")
            == AR_IMAGE_AXIS_CONTRACT
        ):
            capture_tools.update_collection_manifest(output_root, report)
            return report
        raise RuntimeError(
            f"existing immutable capture has a different AR-axis contract: {report_path}"
        )
    if destination.exists():
        raise RuntimeError(f"partial capture dataset already exists: {destination}")
    staging = destination.parent / f".{session_id}.building"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        poses = read_phone_poses(data_dir / "poses.txt")
        views, frame_records = build_axis_corrected_projection_views(
            data_dir,
            mask_dir,
            frame_names,
            poses=poses,
        )
        for name in ("images", "rgb", "masks"):
            (staging / name).mkdir(parents=True, exist_ok=True)
        for view, record in zip(views, frame_records):
            image_output = staging / "images" / view.frame_name
            mask_output = staging / "masks" / f"{Path(view.frame_name).stem}.png"
            image_bgr = np.ascontiguousarray(view.image_rgb[:, :, ::-1])
            image_options: list[int] = []
            if image_output.suffix.lower() in {".jpg", ".jpeg"}:
                image_options = [cv2.IMWRITE_JPEG_QUALITY, 95]
            if not cv2.imwrite(str(image_output), image_bgr, image_options):
                raise RuntimeError(f"failed to write materialized RGB: {image_output}")
            if not cv2.imwrite(str(mask_output), view.mask):
                raise RuntimeError(f"failed to write materialized mask: {mask_output}")
            shutil.copy2(image_output, staging / "rgb" / view.frame_name)
            record["source_image_sha256"] = _sha256_file(
                data_dir / view.frame_name
            )
            record["source_mask_sha256"] = _sha256_file(
                mask_dir / f"{Path(view.frame_name).stem}.png"
            )
            record["materialized_image_sha256"] = _sha256_file(image_output)
            record["materialized_mask_sha256"] = _sha256_file(mask_output)

        shutil.copy2(data_dir / "poses.txt", staging / "poses_raw_unity.txt")
        # Keep the established filename for provenance tools.  The authoritative
        # model cameras are sparse/0/{cameras,images}.txt and are explicitly
        # bound to the server2 axis contract below.
        shutil.copy2(data_dir / "poses.txt", staging / "poses.txt")
        metadata_path = data_dir / "frame_metadata.jsonl"
        if metadata_path.is_file():
            shutil.copy2(metadata_path, staging / "frame_metadata_raw.jsonl")
        points_path = data_dir / "slam_points.jsonl"
        point_cloud_available = points_path.is_file() and points_path.stat().st_size > 0
        if point_cloud_available:
            shutil.copy2(points_path, staging / "slam_points_diagnostic_only.jsonl")

        sparse_metadata = _write_axis_corrected_sparse_model(
            staging, views, poses, frame_records
        )
        try:
            geometry = capture_tools._pose_mask_geometry(views)
        except Exception as error:
            geometry = {"available": False, "error": repr(error)}
        geometry.update(
            {
                "point_cloud_consumed": False,
                "point_to_mask_extent_ratio": None,
                "cross_view_geometry_role": "diagnostic_only_no_frame_rejection",
            }
        )
        image_axis_contract = {
            "format": AR_IMAGE_AXIS_CONTRACT,
            "passed": True,
            "selected_frame_count": len(frame_names),
            "source_image_transform": "XRCpuImage.Transformation.None",
            "materialized_pixel_transform": "transpose_xy",
            "materialized_intrinsics_transform": (
                "(fx,fy,cx,cy,W,H)->(fy,fx,cy,cx,H,W)"
            ),
            "pose_transform": "base Unity-to-CV Rz(0deg)",
            "display_matrix_used_for_extrinsics": False,
            "unity_world_to_internal_world": "diag(1,1,-1)",
            "frame_records": frame_records,
        }
        report = {
            "format": CAPTURE_FORMAT,
            "created_at_utc": utc_now(),
            "session_id": session_id,
            "passed": True,
            "dataset_dir": str(destination.resolve()),
            "selected_frame_count": len(frame_names),
            "selected_frame_names": list(frame_names),
            "geometry_mode": "pose_mask_only",
            "point_cloud_consumed": False,
            "source": {
                "camera_pose": "ARFoundation camera transform",
                "intrinsics": "ARCameraManager.TryGetIntrinsics",
                "point_cloud": "optional diagnostic only; not consumed",
                "mask": "user-supervised SAM2 video masks",
            },
            "config": asdict(config),
            "fusion": {
                "point_cloud_consumed": False,
                "point_cloud_available_for_diagnostics": bool(point_cloud_available),
                "raw_sample_count": 0,
                "temporally_supported_voxel_count": 0,
            },
            "mask_support": {
                "point_cloud_consumed": False,
                "mask_supported_point_count": 0,
                "policy": "bypassed_for_pose_mask_runtime",
            },
            "geometry_diagnostics": geometry,
            "input_qc": dict(input_qc),
            "sparse_metadata": sparse_metadata,
            "image_axis_contract": image_axis_contract,
            "outputs": {
                "cameras": str((destination / "sparse/0/cameras.txt").resolve()),
                "images": str((destination / "sparse/0/images.txt").resolve()),
                "points3D": str((destination / "sparse/0/points3D.txt").resolve()),
                "point_ply": str((destination / "sparse/0/object_points.ply").resolve()),
                "point_npz": str((destination / "sparse/0/object_points.npz").resolve()),
                "raw_unity_poses": str((destination / "poses_raw_unity.txt").resolve()),
                "diagnostic_ar_points": (
                    str((destination / "slam_points_diagnostic_only.jsonl").resolve())
                    if point_cloud_available
                    else None
                ),
            },
            "scope_guard": (
                "server2 pose-mask capture with an immutable x/y-transposed native "
                "CPU-image contract. No displayMatrix-derived rotation enters T_W2C; "
                "no point cloud, quality score, or generated geometry selects frames."
            ),
        }
        write_json(staging / "capture_report.json", report)
        staging.replace(destination)
        capture_tools.update_collection_manifest(output_root, report)
        return report
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def uniform_time_indices(frame_count: int, count: int = SELECTED_VIEW_COUNT) -> list[int]:
    """Return unique, endpoint-inclusive indices uniformly along frame order."""

    frame_count = int(frame_count)
    count = int(count)
    if count <= 0:
        raise ValueError("uniform frame count must be positive")
    if frame_count < count:
        raise ValueError(f"uploaded frame count {frame_count} < required {count}")
    indices = np.rint(np.linspace(0, frame_count - 1, count)).astype(np.int64)
    result = [int(value) for value in indices]
    if len(result) != count or len(set(result)) != count:
        raise RuntimeError("time-uniform selection did not produce unique indices")
    if result[0] != 0 or result[-1] != frame_count - 1:
        raise RuntimeError("time-uniform selection lost an endpoint")
    return result


def uniform_trajectory_indices(
    camera_centers: np.ndarray,
    count: int = SELECTED_VIEW_COUNT,
) -> tuple[list[int], dict[str, Any]]:
    """Select unique frames nearest uniform cumulative translation targets.

    Chronological order and both endpoints are fixed.  A constrained nearest
    choice leaves enough later frames for all remaining targets.  A stationary
    trajectory has no distance parameterization, so it deterministically falls
    back to frame-time uniformity without inspecting image or mask content.
    """

    centers = np.asarray(camera_centers, dtype=np.float64)
    count = int(count)
    if centers.ndim != 2 or centers.shape[1] != 3:
        raise ValueError(f"camera centers must have shape [N,3], got {centers.shape}")
    if not np.isfinite(centers).all():
        raise ValueError("camera centers contain NaN or infinity")
    if len(centers) < count:
        raise ValueError(f"uploaded pose count {len(centers)} < required {count}")

    steps = np.linalg.norm(np.diff(centers, axis=0), axis=1)
    cumulative = np.concatenate([np.zeros(1, dtype=np.float64), np.cumsum(steps)])
    total = float(cumulative[-1])
    targets = np.linspace(0.0, total, count, dtype=np.float64)
    fallback = total <= 1.0e-9
    if fallback:
        selected = uniform_time_indices(len(centers), count)
    else:
        selected = [0]
        for target_position in range(1, count - 1):
            lower = selected[-1] + 1
            upper = len(centers) - (count - target_position)
            candidate_indices = np.arange(lower, upper + 1, dtype=np.int64)
            nearest = int(
                candidate_indices[
                    np.argmin(np.abs(cumulative[candidate_indices] - targets[target_position]))
                ]
            )
            selected.append(nearest)
        selected.append(len(centers) - 1)

    if len(selected) != count or len(set(selected)) != count:
        raise RuntimeError("trajectory-uniform selection did not produce unique indices")
    if selected != sorted(selected):
        raise RuntimeError("trajectory-uniform selection changed chronological order")
    achieved = [float(cumulative[index]) for index in selected]
    audit = {
        "policy": "chronological_cumulative_camera_translation_uniform8_v1",
        "camera_center_source": "uploaded_ARFoundation_camera_position_W",
        "candidate_count": int(len(centers)),
        "translation_step_m": [float(value) for value in steps],
        "cumulative_translation_m": [float(value) for value in cumulative],
        "total_translation_m": total,
        "target_translation_m": [float(value) for value in targets],
        "selected_translation_m": achieved,
        "absolute_target_error_m": [
            float(abs(value - target)) for value, target in zip(achieved, targets)
        ],
        "stationary_trajectory_fallback": bool(fallback),
        "fallback_policy": "time_uniform8" if fallback else None,
    }
    return selected, audit


def build_uniform_selection_plan(
    frame_names: Sequence[str],
    poses: Mapping[str, Mapping[str, Any]],
    *,
    client_selected: Any = None,
    count: int = SELECTED_VIEW_COUNT,
) -> dict[str, Any]:
    """Build both fixed selections without dropping or substituting candidates."""

    names = [str(value) for value in frame_names]
    if len(names) != len(set(names)):
        raise ValueError("uploaded frame names are not unique")
    if len(names) < int(count):
        raise ValueError(f"uploaded frame count {len(names)} < required {int(count)}")
    missing = [name for name in names if name not in poses]
    if missing:
        raise ValueError(
            "uploaded RGB frames lack camera poses; no frames were filtered or replaced: "
            + ", ".join(missing[:12])
        )
    centers = []
    for name in names:
        center = np.asarray(poses[name].get("pos"), dtype=np.float64)
        if center.shape != (3,) or not np.isfinite(center).all():
            raise ValueError(f"invalid uploaded camera position for frame {name}")
        centers.append(center)
    centers_array = np.stack(centers, axis=0)
    time_indices = uniform_time_indices(len(names), int(count))
    trajectory_indices, trajectory_audit = uniform_trajectory_indices(
        centers_array, int(count)
    )
    client_record = client_selected
    if isinstance(client_selected, tuple):
        client_record = list(client_selected)
    plan = {
        "format": FORMAT,
        "created_at_utc": utc_now(),
        "passed": True,
        "candidate_count": len(names),
        "candidate_frame_names": names,
        "camera_centers_W_m": centers_array.tolist(),
        "required_view_count": int(count),
        "client_selected_payload": client_record,
        "client_selection_ignored": True,
        "selection_inputs": ["chronological_frame_order", "camera_position_W"],
        "image_or_mask_content_used_for_selection": False,
        "quality_metric_used_for_selection": False,
        "frame_filtering_applied": False,
        "disabled_selection_or_rejection_signals": [
            "client selected indices",
            "foreground area or mask score",
            "nearest-valid-mask substitution",
            "mask-ray residual",
            "azimuth/elevation coverage",
            "spherical farthest sampling",
            "camera roll",
            "pose continuity segmentation or reset rejection",
            "tracking state",
            "point-cloud support",
        ],
        "branches": {
            "trajectory_uniform8": {
                "policy": trajectory_audit["policy"],
                "selected_indices": trajectory_indices,
                "selected_frame_names": [names[index] for index in trajectory_indices],
                "audit": trajectory_audit,
            },
            "time_uniform8": {
                "policy": "chronological_frame_index_uniform8_including_endpoints_v1",
                "selected_indices": time_indices,
                "selected_frame_names": [names[index] for index in time_indices],
                "audit": {
                    "candidate_count": len(names),
                    "target_frame_positions": np.linspace(
                        0.0, len(names) - 1, int(count)
                    ).tolist(),
                    "selected_frame_positions": time_indices,
                },
            },
        },
    }
    return plan


def _current_selection_plan(payload: dict[str, Any]) -> dict[str, Any]:
    base.legacy._load_current_session()
    frame_names = base.legacy._list_session_images()
    poses = read_phone_poses(Path(base.legacy.current_data_dir) / "poses.txt")
    return build_uniform_selection_plan(
        frame_names,
        poses,
        client_selected=payload.get("selected"),
    )


def _branch_session_id(source_session_id: str, mode: str) -> str:
    source = str(source_session_id)
    if not _SAFE_SESSION_ID.fullmatch(source):
        raise ValueError(f"unsafe source session id: {source!r}")
    if mode not in SELECTION_MODES:
        raise ValueError(f"unsupported selection mode: {mode}")
    # The suffix prevents an old server2 dataset that used the incomplete
    # displayMatrix/Rz contract from being silently reused after this fix.
    result = f"{source}__{mode}__{AR_IMAGE_AXIS_SESSION_SUFFIX}"
    if not _SAFE_SESSION_ID.fullmatch(result):
        raise ValueError(f"unsafe branch session id: {result!r}")
    return result


def _branch_input_record(plan: dict[str, Any], mode: str) -> dict[str, Any]:
    branch = plan["branches"][mode]
    return {
        "profile": "server2_filter_free_fixed8_v1",
        "qc_pass": True,
        "fail_reasons": [],
        "warnings": [],
        "selection_mode": mode,
        "selection_policy": branch["policy"],
        "candidate_count": int(plan["candidate_count"]),
        "candidate_frame_names": list(plan["candidate_frame_names"]),
        "selected_indices": list(branch["selected_indices"]),
        "selected_frame_names": list(branch["selected_frame_names"]),
        "client_selection_ignored": True,
        "frame_filtering_applied": False,
        "runtime_o_candidate_contract": "exactly_the_fixed_8_frames",
        "runtime_o_secondary_selection": False,
        "quality_diagnostics_are_non_blocking": True,
        "image_axis_contract": AR_IMAGE_AXIS_CONTRACT,
        "disabled_selection_or_rejection_signals": list(
            plan["disabled_selection_or_rejection_signals"]
        ),
        "selection_audit": branch["audit"],
    }


def collection_input_qc():
    """Report only whether the fixed 8-frame contract can be constructed."""

    try:
        payload = request.get_json(silent=True) or {}
        plan = _current_selection_plan(payload)
        primary = plan["branches"]["trajectory_uniform8"]
        return jsonify(
            {
                "status": "ok",
                "message": (
                    "输入可用：不做质量筛选；将分别按相机累计位移和时间轴均匀取8帧"
                ),
                "client_selected_indices": payload.get("selected"),
                "selected_indices": primary["selected_indices"],
                "selection_plan": plan,
                "point_cloud_required": False,
                "geometry_mode": "pose_mask",
                "input_qc": {
                    "profile": "server2_mechanical_readiness_only_v1",
                    "qc_pass": True,
                    "frame_filtering_applied": False,
                },
            }
        ), 200
    except Exception as error:
        base.legacy.logging.exception("server2 fixed-frame readiness check failed")
        return jsonify(
            {
                "status": "warning",
                "message": f"无法构造固定8帧输入：{error}",
                "frame_filtering_applied": False,
            }
        ), 200


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _render_input_contours(
    reconstruction: dict[str, Any],
    *,
    mode: str,
) -> dict[str, Any]:
    """Project the exact runtime-O Mesh through physical T_O2W/T_W2C cameras."""

    config = base._RECONSTRUCTION_CONFIG
    if config is None:
        raise RuntimeError("contours require an active reconstruction deployment")
    runtime_manifest = Path(reconstruction["stage_reports"]["runtime_o"])
    inference_manifest = Path(
        reconstruction["stage_reports"]["official_ss_trained_slat"]
    )
    inference = _load_json(inference_manifest)
    rows = list(inference.get("objects") or [])
    if len(rows) != 1 or rows[0].get("passed") is not True:
        raise RuntimeError("server2 contour export requires one passed inference object")
    row = rows[0]
    mesh_o = Path(str(row["mesh"])).expanduser().resolve(strict=True)
    result_path = Path(str(row.get("result") or mesh_o.with_name("result.json")))
    result_path = result_path.expanduser().resolve(strict=True)
    run_dir = Path(reconstruction["run_dir"])
    output_dir = run_dir / "final" / "输入8帧Mesh轮廓回投影"
    report_path = output_dir / "report.json"
    if report_path.is_file():
        report = reconstruction_tools._load_passed_report(report_path)
    else:
        if output_dir.exists():
            raise RuntimeError(
                f"partial contour output exists; preserve and inspect: {output_dir}"
            )
        command = [
            str(config.python),
            "-u",
            "-m",
            "pose_point_depth_mv.render_runtime_o_mesh_camera_contours",
            "--runtime_input_manifest",
            str(runtime_manifest),
            "--mesh_o",
            str(mesh_o),
            "--mesh_frame_report",
            str(result_path),
            "--output_dir",
            str(output_dir),
            "--contour_width",
            "3",
            "--method_label",
            f"server2 {mode}",
            "--overview_name",
            f"{mode}_输入8帧Mesh轮廓回投影总览.png",
        ]
        report = reconstruction_tools._run_stage(
            name=f"input_camera_contours_{mode}",
            command=command,
            expected_report=report_path,
            log_path=run_dir / "reconstruction.log",
            environment=reconstruction_tools._subprocess_environment(config),
        )
    return {
        "passed": True,
        "report": str(report_path.resolve()),
        "overview": str(Path(report["overview"]).resolve()),
        "overlay_directory": str(
            (output_dir / "相机位姿Mesh轮廓叠加").resolve()
        ),
        "projection_formula": report["projection_formula"],
        "selected_frame_names": report["selected_frame_names"],
    }


def _run_branch(
    *,
    source_session_id: str,
    mode: str,
    plan: dict[str, Any],
    data_dir: Path,
    mask_dir: Path,
) -> dict[str, Any]:
    if base._OUTPUT_ROOT is None:
        raise RuntimeError("server2 output root was not configured")
    branch = plan["branches"][mode]
    branch_session_id = _branch_session_id(source_session_id, mode)
    base._set_latest_progress(
        {
            "status": "running",
            "stage": f"{mode}:capture",
            "source_session_id": source_session_id,
            "branch_session_id": branch_session_id,
        }
    )
    capture = finalize_axis_corrected_ar_capture(
        session_id=branch_session_id,
        data_dir=data_dir,
        mask_dir=mask_dir,
        frame_names=branch["selected_frame_names"],
        output_root=base._OUTPUT_ROOT,
        input_qc=_branch_input_record(plan, mode),
        config=base._POINT_CONFIG,
    )
    reconstruction = None
    contours = None
    if base._RECONSTRUCTION_CONFIG is not None:
        base._set_latest_progress(
            {
                "status": "running",
                "stage": f"{mode}:reconstruction",
                "source_session_id": source_session_id,
                "branch_session_id": branch_session_id,
            }
        )
        reconstruction = base._run_configured_reconstruction(
            session_id=branch_session_id,
            dataset_dir=Path(capture["dataset_dir"]),
            output_root=base._OUTPUT_ROOT,
        )
        base._set_latest_progress(
            {
                "status": "running",
                "stage": f"{mode}:input_camera_contours",
                "source_session_id": source_session_id,
                "branch_session_id": branch_session_id,
            }
        )
        contours = _render_input_contours(reconstruction, mode=mode)
    result = {
        "format": BRANCH_FORMAT,
        "created_at_utc": utc_now(),
        "passed": True,
        "source_session_id": source_session_id,
        "branch_session_id": branch_session_id,
        "selection_mode": mode,
        "selection": branch,
        "capture": capture,
        "reconstruction": reconstruction,
        "input_camera_contours": contours,
        "frame_filtering_applied": False,
        "quality_diagnostics_are_non_blocking": True,
        "image_axis_contract": AR_IMAGE_AXIS_CONTRACT,
        "model_o_axis_convention": "official_z_up",
    }
    branch_report = (
        base._OUTPUT_ROOT
        / "reconstructions"
        / branch_session_id
        / "server2_branch_report.json"
    )
    write_json(branch_report, result)
    result["report"] = str(branch_report.resolve())
    return result


def _summary_path(source_session_id: str) -> Path:
    if base._OUTPUT_ROOT is None:
        raise RuntimeError("server2 output root was not configured")
    return (
        base._OUTPUT_ROOT
        / "sessions"
        / source_session_id
        / "server2_dual_uniform8_report.json"
    )


def _write_session_summary(
    *,
    source_session_id: str,
    plan: dict[str, Any],
    primary_mode: str,
    branches: dict[str, Any],
    branch_status: dict[str, str],
    background_error: str | None = None,
) -> Path:
    completed = all(value == "complete" for value in branch_status.values())
    failed = any(value == "failed" for value in branch_status.values())
    summary = {
        "format": FORMAT,
        "created_at_utc": utc_now(),
        "passed": bool(completed and not failed),
        "status": "failed" if failed else ("complete" if completed else "running"),
        "source_session_id": source_session_id,
        "output_root": str(base._OUTPUT_ROOT.resolve()),
        "selection_plan": plan,
        "executed_modes": list(_ACTIVE_SELECTION_MODES),
        "primary_mode": primary_mode,
        "primary_branch_session_id": _branch_session_id(
            source_session_id, primary_mode
        ),
        "branch_status": branch_status,
        "branches": branches,
        "background_error": background_error,
        "frame_filtering_applied": False,
        "image_axis_contract": AR_IMAGE_AXIS_CONTRACT,
        "model_o_axis_convention": "official_z_up",
        "scope_guard": (
            "Both branches use exact mechanically selected 8-frame inputs. "
            "RGB/mask/K/T_W2C use the server2 native-image x/y-transpose "
            "axis contract. "
            "Quality diagnostics are recorded but cannot filter, substitute, "
            "re-rank, or reject a selected view. The trajectory branch is returned "
            "to the phone first; the time branch completes in the server background."
        ),
    }
    path = _summary_path(source_session_id)
    write_json(path, summary)
    return path


def _run_background_branch(
    *,
    source_session_id: str,
    mode: str,
    plan: dict[str, Any],
    data_dir: Path,
    mask_dir: Path,
    primary_mode: str,
    primary: dict[str, Any],
) -> None:
    branches: dict[str, Any] = {primary_mode: primary}
    status = {primary_mode: "complete", mode: "running"}
    try:
        _write_session_summary(
            source_session_id=source_session_id,
            plan=plan,
            primary_mode=primary_mode,
            branches=branches,
            branch_status=status,
        )
        with base._FINALIZE_LOCK:
            branches[mode] = _run_branch(
                source_session_id=source_session_id,
                mode=mode,
                plan=plan,
                data_dir=data_dir,
                mask_dir=mask_dir,
            )
        status[mode] = "complete"
        summary_path = _write_session_summary(
            source_session_id=source_session_id,
            plan=plan,
            primary_mode=primary_mode,
            branches=branches,
            branch_status=status,
        )
        base._set_latest_progress(
            {
                "status": "complete",
                "stage": "complete",
                "source_session_id": source_session_id,
                "summary_report": str(summary_path.resolve()),
                "branches": {
                    branch_mode: {
                        "branch_session_id": row["branch_session_id"],
                        "report": row["report"],
                        "contours": row["input_camera_contours"],
                    }
                    for branch_mode, row in branches.items()
                },
            }
        )
    except Exception as error:
        status[mode] = "failed"
        _write_session_summary(
            source_session_id=source_session_id,
            plan=plan,
            primary_mode=primary_mode,
            branches=branches,
            branch_status=status,
            background_error=repr(error),
        )
        base.legacy.logging.exception(
            "server2 background branch failed: source=%s mode=%s",
            source_session_id,
            mode,
        )
        base._set_latest_progress(
            {
                "status": "failed",
                "stage": f"{mode}:background",
                "source_session_id": source_session_id,
                "error": repr(error),
            }
        )
    finally:
        _BACKGROUND_BRANCH_THREADS.pop(source_session_id, None)


def collection_generate():
    try:
        payload = request.get_json(silent=True) or {}
        plan = _current_selection_plan(payload)
        if base._OUTPUT_ROOT is None:
            raise RuntimeError("server2 output root was not configured")
        source_session_id = str(base.legacy.current_session_id)
        data_dir = Path(base.legacy.current_data_dir).resolve(strict=True)
        mask_dir = Path(base.legacy.current_mask_dir).resolve(strict=True)
        primary_mode = (
            "trajectory_uniform8"
            if "trajectory_uniform8" in _ACTIVE_SELECTION_MODES
            else _ACTIVE_SELECTION_MODES[0]
        )
        with base._FINALIZE_LOCK:
            primary = _run_branch(
                source_session_id=source_session_id,
                mode=primary_mode,
                plan=plan,
                data_dir=data_dir,
                mask_dir=mask_dir,
            )
        branches: dict[str, Any] = {primary_mode: primary}
        branch_status = {primary_mode: "complete"}
        secondary_modes = [
            mode for mode in _ACTIVE_SELECTION_MODES if mode != primary_mode
        ]
        for mode in secondary_modes:
            branch_status[mode] = "pending_background"
        summary_path = _write_session_summary(
            source_session_id=source_session_id,
            plan=plan,
            primary_mode=primary_mode,
            branches=branches,
            branch_status=branch_status,
        )
        background_thread: threading.Thread | None = None
        if secondary_modes:
            if source_session_id in _BACKGROUND_BRANCH_THREADS:
                raise RuntimeError(
                    f"background branch is already running for {source_session_id}"
                )
            secondary_mode = secondary_modes[0]
            background_thread = threading.Thread(
                target=_run_background_branch,
                kwargs={
                    "source_session_id": source_session_id,
                    "mode": secondary_mode,
                    "plan": plan,
                    "data_dir": data_dir,
                    "mask_dir": mask_dir,
                    "primary_mode": primary_mode,
                    "primary": primary,
                },
                name=f"server2-{source_session_id}-{secondary_mode}",
                daemon=True,
            )
            _BACKGROUND_BRANCH_THREADS[source_session_id] = background_thread
        reconstruction = primary["reconstruction"]
        mobile_ar = None
        if reconstruction is not None:
            mobile_ar = {
                "format": "yxc_unity_ar_mesh.v1",
                "mesh_url": f"/reconstruction_mesh/{primary['branch_session_id']}",
                "coordinate_frame": "unity_world",
                "placement": "world_identity",
                "display_only": True,
            }
        base._set_latest_progress(
            {
                "status": (
                    "primary_complete_secondary_running"
                    if secondary_modes
                    else "complete"
                ),
                "stage": (
                    f"{secondary_modes[0]}:background"
                    if secondary_modes
                    else "complete"
                ),
                "source_session_id": source_session_id,
                "summary_report": str(summary_path.resolve()),
                "primary_branch_session_id": primary["branch_session_id"],
                "primary_mesh": (
                    None
                    if reconstruction is None
                    else reconstruction["meshes"]["world_glb"]
                ),
            }
        )
        if background_thread is not None:
            background_thread.start()
        primary_label = {
            "trajectory_uniform8": "相机位移轨迹均匀8帧",
            "time_uniform8": "时间轴均匀8帧",
        }[primary_mode]
        primary_message = (
            f"{primary_label}主分支已完成并返回Mesh；"
            if reconstruction is not None
            else f"{primary_label}主分支的固定8帧采集集已保存（capture_only）；"
        )
        return jsonify(
            {
                "status": "success",
                "message": (
                    primary_message
                    + (
                        "时间均匀8帧分支正在服务端后台运行；"
                        if secondary_modes
                        else ""
                    )
                    + f"汇总报告：{summary_path.resolve()}"
                ),
                "session_id": source_session_id,
                "primary_branch_session_id": primary["branch_session_id"],
                "dataset_dir": primary["capture"]["dataset_dir"],
                "capture_report": str(
                    Path(primary["capture"]["dataset_dir"]) / "capture_report.json"
                ),
                "client_selected_indices": payload.get("selected"),
                "selected_indices": primary["selection"]["selected_indices"],
                # Keep the legacy response key for the existing phone client.  It
                # contains the mechanically selected indices, not a filtered set.
                "filtered_selected_indices": primary["selection"]["selected_indices"],
                "frame_filtering_applied": False,
                "selection_plan": plan,
                "capture": primary["capture"],
                "reconstruction": reconstruction,
                "reconstruction_branches": branches,
                "branch_status": branch_status,
                "summary_report": str(summary_path.resolve()),
                "deployment_profile": "server2_filter_free_dual_uniform8",
                "mobile_ar": mobile_ar,
            }
        ), 200
    except Exception as error:
        base.legacy.logging.exception("server2 dual-uniform reconstruction failed")
        base._set_latest_progress(
            {
                "status": "failed",
                "stage": "server2",
                "source_session_id": base.legacy.current_session_id,
                "error": repr(error),
            }
        )
        return jsonify(
            {
                "status": "error",
                "message": base._phone_error_message(error),
                "error_detail": str(error),
                "session_id": base.legacy.current_session_id,
                "session_data_dir": base.legacy.current_data_dir,
            }
        ), 500


def build_parser() -> argparse.ArgumentParser:
    parser = base.build_parser()
    parser.description = (
        "Capture phone AR data and reconstruct exact trajectory-uniform/time-uniform "
        "8-frame branches without input-quality frame filtering"
    )
    parser.set_defaults(
        output_root=DEFAULT_OUTPUT_ROOT,
        gpu=DEFAULT_GPU,
        geometry_mode="pose_mask",
        deployment="official",
    )
    parser.add_argument(
        "--selection_mode",
        choices=("both", *SELECTION_MODES),
        default="both",
        help="both is the direct-launch default; single modes are diagnostic conveniences",
    )
    return parser


def build_reconstruction_config(
    args: argparse.Namespace,
) -> base.ARReconstructionConfig | OfficialReconstructionConfig | None:
    config = base.build_reconstruction_config(args)
    if config is None:
        return None
    if not isinstance(config, OfficialReconstructionConfig):
        raise ValueError(
            "server2 requires --deployment official so the runtime-O Mesh frame "
            "contract can be audited before camera-contour projection"
        )
    return replace(
        config,
        selected_view_count=SELECTED_VIEW_COUNT,
        geometry_mode="pose_mask",
        view_selection_policy="lexical_even",
        gravity_up_w=(0.0, 1.0, 0.0),
        model_o_axis_convention="official_z_up",
        diagnostic_bypass_pose_mask_quality=True,
    )


def configure_server2(
    *,
    output_root: Path,
    point_config: ARPointFilterConfig,
    reconstruction_config: (
        base.ARReconstructionConfig | OfficialReconstructionConfig | None
    ),
    selection_mode: str,
) -> None:
    global _ACTIVE_SELECTION_MODES
    _ACTIVE_SELECTION_MODES = (
        SELECTION_MODES if selection_mode == "both" else (str(selection_mode),)
    )
    base.configure_server(
        output_root,
        point_config,
        reconstruction_config,
        require_point_cloud=False,
    )
    base._RECONSTRUCTION_DEPLOYMENT = "server2_filter_free_dual_uniform8"
    # Replace only the two policy endpoints. Upload, SAM2, status, and Mesh serving
    # remain the proven phone-compatible implementation from server.py.
    base.legacy.app.view_functions["input_qc"] = collection_input_qc
    base.legacy.app.view_functions["generate"] = collection_generate


def main() -> None:
    args = build_parser().parse_args()
    if args.geometry_mode != "pose_mask":
        raise ValueError("server2 is intentionally pose_mask-only; point filtering is disabled")
    point_config = ARPointFilterConfig(
        voxel_size_m=args.voxel_size_m,
        min_temporal_observations=args.min_temporal_observations,
        min_mask_observations=args.min_mask_observations,
        min_mask_support_ratio=args.min_mask_support_ratio,
        mask_dilation_px=args.mask_dilation_px,
        min_object_points=args.min_object_points,
        max_object_points=args.max_object_points,
        max_point_to_mask_extent_ratio=args.max_point_to_mask_extent_ratio,
        max_ray_residual_median_over_mask_extent=(
            args.max_ray_residual_median_over_mask_extent
        ),
        max_ray_residual_p90_over_mask_extent=(
            args.max_ray_residual_p90_over_mask_extent
        ),
        min_orbit_gravity_agreement=args.min_orbit_gravity_agreement,
        min_synchronized_frame_ratio=args.min_synchronized_frame_ratio,
    )
    reconstruction_config = build_reconstruction_config(args)
    configure_server2(
        output_root=args.output_root,
        point_config=point_config,
        reconstruction_config=reconstruction_config,
        selection_mode=str(args.selection_mode),
    )
    base.legacy.clean_environment()
    print(">>> [server2] 手机 AR 无输入质量筛选重建服务启动", flush=True)
    print(f">>> 输出根目录: {base._OUTPUT_ROOT}", flush=True)
    print(f">>> 执行分支: {list(_ACTIVE_SELECTION_MODES)}", flush=True)
    print(">>> 选帧仅使用: 时间顺序、相机中心累计位移", flush=True)
    print(">>> AR像素轴: 原生RGB/mask转置x/y；K同步交换；T_W2C不使用displayMatrix旋转", flush=True)
    print(f">>> AR像素轴合同: {AR_IMAGE_AXIS_CONTRACT}", flush=True)
    print(
        ">>> Runtime-O: 固定8帧 lexical_even；official_z_up；"
        "质量诊断不作为拒绝门",
        flush=True,
    )
    print(">>> 每个 Mesh 均回投影到其固定输入8帧并输出青色轮廓", flush=True)
    if reconstruction_config is None:
        print(">>> capture_only：仅固化两个8帧数据集，不运行模型", flush=True)
    else:
        print(f">>> 重建 GPU: physical cuda:{reconstruction_config.gpu}", flush=True)
        print(f">>> 部署配置: {asdict(reconstruction_config)}", flush=True)
    base.legacy.app.run(
        host=args.host,
        port=args.port,
        debug=False,
        use_reloader=False,
    )


if __name__ == "__main__":
    main()
