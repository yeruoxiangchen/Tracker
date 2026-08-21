#!/usr/bin/env python3
"""First-stage phone AR reconstruction built on the manual Mesh pipeline.

The legacy :mod:`pose_point_depth_mv.server` remains untouched and is used only
as the proven HTTP upload and interactive-SAM2 transport.  Reconstruction is
owned by this package:

* every uploaded RGB/mask/pose candidate is axis-corrected and retained;
* one official-compatible runtime-O is estimated from that complete domain;
* the exact single-seed spherical farthest-point eight-frame subset used by
  official training is selected only after O has been frozen;
* no-VGGT SS30K + SLat30K produces a native runtime-O Mesh;
* the Mesh is exported in runtime-O, internal world, GLB, and Unity ``.armesh``
  forms and reprojected onto the exact eight input frames.

There is exactly one reconstruction branch.  Time-uniform, trajectory-uniform,
random, client-side, and quality-ranked alternatives are not executed.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import io
import math
import os
from pathlib import Path
import re
import shutil
import sys
import threading
from typing import Any, Mapping, Sequence


TRACKER_ROOT = Path(__file__).resolve().parents[1]
if str(TRACKER_ROOT) not in sys.path:
    sys.path.insert(0, str(TRACKER_ROOT))

DEFAULT_OUTPUT_ROOT = (
    TRACKER_ROOT
    / "pose_point_depth_mv"
    / "outputs2"
    / "手机AR第一阶段_全视图统一O_训练一致球面最远8帧"
)
DEFAULT_GPU = os.environ.get("AR_OBJECT_GPU", os.environ.get("GPU", "4"))
SELECTED_VIEW_COUNT = 8
SELECTION_MODES = ("training_spherical_farthest8",)
PRIMARY_MODE = "training_spherical_farthest8"
FORMAT = "manual_mesh_reconstruction.phone_server_phase1.v3"
BRANCH_FORMAT = "manual_mesh_reconstruction.phone_server_phase1_branch.v4"
CAPTURE_FORMAT = "manual_mesh_reconstruction.phone_all_view_capture.v3"
WORLD_EXPORT_FORMAT = "manual_mesh_reconstruction.runtime_o_a0_mesh_export.v5"
PHONE_CONTOUR_FORMAT = "manual_mesh_reconstruction.phone_original_contours.v2"
AXIS_CONTRACT = "xrcpuimage_none_xy_transpose_anchor_a0_unity_cv.v1"
PHONE_POSE_BINDING = "camera_frame_received_anchor_a0_relative_v1"
PHONE_POSE_COORDINATE_FRAME = "unity_capture_anchor_a0"
MOBILE_MESH_COORDINATE_FRAME = PHONE_POSE_COORDINATE_FRAME
MOBILE_MESH_PLACEMENT = "capture_anchor_a0_direct"
RUNTIME_O_PREPARE_FORMAT = "manual_mesh_reconstruction.runtime_o_prepare.v2"
MOBILE_OVERLAY_AUDIT_FORMAT = (
    "manual_mesh_reconstruction.mobile_pose_diagnostic_recording.v3"
)
MOBILE_OVERLAY_AUDIT_CONTRACT = (
    "unity_native_screen_display_aligned_raw_rgb_strict_input_pose_v3"
)
MOBILE_OVERLAY_AUDIT_MAX_FRAMES = 8
MOBILE_OVERLAY_TARGET_TRANSLATION_TOLERANCE_METERS = 0.025
MOBILE_OVERLAY_TARGET_ROTATION_TOLERANCE_DEGREES = 3.0
MOBILE_OVERLAY_MAX_IMAGE_POSE_TIME_DELTA_SECONDS = 0.10
_SAFE_SESSION_ID = re.compile(r"^[A-Za-z0-9_.-]+$")

# A direct ``python manual_mesh_reconstruction/server.py`` launch must bind the
# requested physical GPU before importing Torch/TRELLIS-facing modules.
if __name__ == "__main__":
    environment_bin = str(Path(sys.executable).resolve().parent)
    inherited_path = os.environ.get("PATH", "")
    os.environ["PATH"] = os.pathsep.join(
        value for value in (environment_bin, inherited_path) if value
    )
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", DEFAULT_GPU)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    os.environ.setdefault("ATTN_BACKEND", "flash_attn")
    os.environ.setdefault("SPCONV_ALGO", "native")
    os.environ["RECON_ENFORCE_INPUT_QC"] = "0"

import numpy as np
from PIL import Image, ImageDraw
import trimesh
from flask import jsonify, request, send_file

from manual_mesh_reconstruction.canonicalization import (
    array_sha256,
    validate_proper_similarity,
)
from manual_mesh_reconstruction.common import (
    atomic_json,
    canonical_sha256,
    load_json,
    sha256_file,
)
from manual_mesh_reconstruction.data_adapters.common import (
    CameraFrame,
    deferred_selection_request,
    materialize_raw_cache,
)
from manual_mesh_reconstruction.defaults import (
    ABC_R_BRIDGE,
    PRETRAINED,
    SLAT30K_CHECKPOINT,
    SLAT_STEP,
    SS30K_REPORT,
    STOCK_SLAT_FREEZE,
)
from manual_mesh_reconstruction.mesh_coordinates import (
    validate_runtime_o_mesh_frame_contract,
)
from manual_mesh_reconstruction.pipeline import (
    _environment,
    _one_inference_record,
    _one_runtime_object,
    _passed,
    _run_stage,
)
from manual_mesh_reconstruction.pose_mask import (
    OFFICIAL_COMPATIBLE_Z_UP_OBJECT_FRAME_POLICY,
)
from manual_mesh_reconstruction.projection import (
    boundary,
    load_meshes,
    make_headless_raster_context,
    rasterize_world_silhouette,
)
from manual_mesh_reconstruction.alignment_refinement import (
    FORMAT as ALIGNMENT_REFINEMENT_FORMAT,
    MAX_OPTIMIZATION_VIEWS,
    MIN_OPTIMIZATION_VIEWS,
    POSE_BINDING as ALIGNMENT_POSE_BINDING,
    POSE_COORDINATE_FRAME as ALIGNMENT_POSE_COORDINATE_FRAME,
    RECOMMENDED_CAPTURE_FRAMES,
    corrected_intrinsic,
    internal_camera_from_unity_metadata,
    mobile_mesh_internal_buffers,
    run_refinement,
    unity_similarity_to_internal,
)
from manual_mesh_reconstruction.sam2_worker_client import (
    DEFAULT_SAM2_PYTHON,
    Sam2TinyVideoWorkerClient,
)
from pose_point_depth_mv import server as transport_server
from pose_point_depth_mv.ar_mobile_overlay import build_mobile_overlay_mesh
from pose_point_depth_mv.ar_object_capture import ARPointFilterConfig
from trellis_point_prior_mv.build_ar_session_smoke_dataset import (
    intrinsics_for_pose,
    read_phone_poses,
    unity_pose_to_colmap_w2c,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _rotation_matrix_to_quaternion_xyzw(rotation: np.ndarray) -> list[float]:
    """Return a normalized Unity-compatible xyzw quaternion."""

    matrix = np.asarray(rotation, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("rotation must be a finite 3x3 matrix")
    if not np.allclose(matrix.T @ matrix, np.eye(3), atol=1.0e-6):
        raise ValueError("rotation is not orthonormal")
    if not math.isclose(float(np.linalg.det(matrix)), 1.0, abs_tol=1.0e-6):
        raise ValueError("rotation is not proper")

    trace = float(np.trace(matrix))
    if trace > 0.0:
        root = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * root
        x = (matrix[2, 1] - matrix[1, 2]) / root
        y = (matrix[0, 2] - matrix[2, 0]) / root
        z = (matrix[1, 0] - matrix[0, 1]) / root
    else:
        diagonal = np.diag(matrix)
        axis = int(np.argmax(diagonal))
        if axis == 0:
            root = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            w = (matrix[2, 1] - matrix[1, 2]) / root
            x = 0.25 * root
            y = (matrix[0, 1] + matrix[1, 0]) / root
            z = (matrix[0, 2] + matrix[2, 0]) / root
        elif axis == 1:
            root = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            w = (matrix[0, 2] - matrix[2, 0]) / root
            x = (matrix[0, 1] + matrix[1, 0]) / root
            y = 0.25 * root
            z = (matrix[1, 2] + matrix[2, 1]) / root
        else:
            root = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            w = (matrix[1, 0] - matrix[0, 1]) / root
            x = (matrix[0, 2] + matrix[2, 0]) / root
            y = (matrix[1, 2] + matrix[2, 1]) / root
            z = 0.25 * root
    quaternion = np.asarray([x, y, z, w], dtype=np.float64)
    norm = float(np.linalg.norm(quaternion))
    if not math.isfinite(norm) or norm <= 1.0e-12:
        raise ValueError("rotation produced an invalid quaternion")
    quaternion /= norm
    if quaternion[3] < 0.0:
        quaternion *= -1.0
    return quaternion.tolist()


def unity_object_pose_from_t_o2a0(T_O2A0: np.ndarray) -> dict[str, Any]:
    """Extract informational object pose in the Unity A0 coordinate frame."""

    transform = validate_proper_similarity(T_O2A0, name="T_O2A0")
    linear = np.asarray(transform[:3, :3], dtype=np.float64)
    scale = float(np.cbrt(np.linalg.det(linear)))
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"T_O2A0 has invalid scale: {scale}")
    rotation_internal = linear / scale
    reflection = np.diag([1.0, 1.0, -1.0])
    rotation_unity = reflection @ rotation_internal @ reflection
    translation_unity = reflection @ np.asarray(transform[:3, 3], dtype=np.float64)
    return {
        "format": "manual_mesh_reconstruction.object_pose_unity_a0.v1",
        "parent_frame": PHONE_POSE_COORDINATE_FRAME,
        "position_x": float(translation_unity[0]),
        "position_y": float(translation_unity[1]),
        "position_z": float(translation_unity[2]),
        "quaternion_xyzw": _rotation_matrix_to_quaternion_xyzw(rotation_unity),
        "runtime_o_scale_meters_per_unit": scale,
        "rotation_contract": "R_unity=diag(1,1,-1)@R_internal@diag(1,1,-1)",
    }


@dataclass(frozen=True)
class ServerConfig:
    output_root: Path
    python: Path
    gpu: str
    seed: int = 42
    selected_view_count: int = SELECTED_VIEW_COUNT
    contour_width: int = 3
    o2w_refinement_iterations: int = 40
    o2w_refinement_max_views: int = 24
    o2w_refinement_render_long_side: int = 160
    o2w_refinement_batch_size: int = 4
    o2w_refinement_face_limit: int = 30_000
    selection_modes: tuple[str, ...] = SELECTION_MODES
    capture_only: bool = False
    sam2_python: Path = DEFAULT_SAM2_PYTHON
    sam2_worker_port: int = 5091

    def validate(self) -> None:
        if not self.python.is_file():
            raise FileNotFoundError(self.python)
        if int(self.selected_view_count) != SELECTED_VIEW_COUNT:
            raise ValueError("the phase-one phone contract requires exactly eight views")
        if int(self.contour_width) < 1:
            raise ValueError("contour width must be positive")
        if int(self.o2w_refinement_iterations) <= 0:
            raise ValueError("O2W refinement iterations must be positive")
        if (
            int(self.o2w_refinement_max_views) != 0
            and int(self.o2w_refinement_max_views) < 8
        ):
            raise ValueError("O2W refinement max views must be zero or at least eight")
        if int(self.o2w_refinement_render_long_side) < 96:
            raise ValueError("O2W refinement render long side must be at least 96")
        if int(self.o2w_refinement_batch_size) <= 0:
            raise ValueError("O2W refinement batch size must be positive")
        if int(self.o2w_refinement_face_limit) < 1_000:
            raise ValueError("O2W refinement face limit must be at least 1000")
        if tuple(self.selection_modes) != SELECTION_MODES:
            raise ValueError(
                "phase one exposes only official-training spherical FPS8; "
                f"got selection_modes={self.selection_modes}"
            )
        if not self.capture_only and not self.sam2_python.is_file():
            raise FileNotFoundError(
                f"SAM2 Tiny environment Python is missing: {self.sam2_python}"
            )
        if not 1024 <= int(self.sam2_worker_port) <= 65535:
            raise ValueError("SAM2 worker port must be in [1024,65535]")


_CONFIG: ServerConfig | None = None
_PIPELINE_LOCK = threading.Lock()
_ALIGNMENT_LOCK = threading.Lock()
_ALIGNMENT_STATE_LOCK = threading.Lock()
_ALIGNMENT_ATTEMPTS: dict[str, dict[str, Any]] = {}
_MOBILE_OVERLAY_AUDIT_LOCK = threading.Lock()
_MOBILE_OVERLAY_AUDITS: dict[str, dict[str, Any]] = {}
_SAM2_CLIENT: Sam2TinyVideoWorkerClient | None = None


def build_selection_plan(
    frame_names: Sequence[str],
    poses: Mapping[str, Mapping[str, Any]],
    *,
    runtime_binding: Mapping[str, Any],
    client_selected: Any = None,
    count: int = SELECTED_VIEW_COUNT,
) -> dict[str, Any]:
    """Bind the one post-O, official-training spherical FPS8 selection."""

    names = [str(value) for value in frame_names]
    if len(names) != len(set(names)):
        raise ValueError("uploaded frame names are not unique")
    if any(Path(name).name != name for name in names):
        raise ValueError("uploaded frame names must be plain basenames")
    if len(names) < int(count):
        raise ValueError(f"uploaded frame count {len(names)} < required {int(count)}")
    missing = [name for name in names if name not in poses]
    if missing:
        raise ValueError(
            "uploaded RGB frames lack camera poses; no frame was filtered or replaced: "
            + ", ".join(missing[:12])
        )
    centers: list[np.ndarray] = []
    for name in names:
        center = np.asarray(poses[name].get("pos"), dtype=np.float64)
        if center.shape != (3,) or not np.isfinite(center).all():
            raise ValueError(f"invalid uploaded camera position: {name}")
        centers.append(center)
    centers_array = np.stack(centers, axis=0)
    view_selection = dict(runtime_binding.get("view_selection") or {})
    expected_policy = "pose_mask_training_exact_spherical_farthest_valid_mask"
    if view_selection.get("policy") != expected_policy:
        raise RuntimeError(
            "runtime did not use the sole phase-one spherical FPS policy: "
            f"{view_selection.get('policy')!r}"
        )
    selected_indices = [
        int(value)
        for value in view_selection.get("selected_source_view_indices") or []
    ]
    selected_names = [
        str(value)
        for value in view_selection.get("selected_source_frame_names") or []
    ]
    if len(selected_indices) != int(count) or len(set(selected_indices)) != int(count):
        raise RuntimeError("spherical FPS did not produce eight unique source indices")
    if any(index < 0 or index >= len(names) for index in selected_indices):
        raise RuntimeError("spherical FPS produced an out-of-range source index")
    expected_names = [names[index] for index in selected_indices]
    if selected_names != expected_names:
        raise RuntimeError(
            "runtime spherical FPS source-name binding differs from capture order"
        )
    if view_selection.get("quality_gate_used_for_selection") is not False:
        raise RuntimeError("spherical FPS unexpectedly used quality-gate reranking")
    if runtime_binding.get("o_frozen_before_view_selection") is not True:
        raise RuntimeError("model-O was not frozen before spherical FPS")
    plan = {
        "format": FORMAT,
        "created_at_utc": utc_now(),
        "passed": True,
        "candidate_count": len(names),
        "candidate_frame_names": names,
        "camera_centers_Unity_W_m": centers_array.tolist(),
        "required_view_count": int(count),
        "client_selected_payload": client_selected,
        "client_selection_ignored": True,
        "selection_occurs_after_all_view_o": True,
        "image_content_used_for_selection": False,
        "mask_content_used_only_for_foreground_validity": True,
        "quality_metric_used_for_selection": False,
        "quality_gate_used_for_selection": False,
        "foreground_validity_filter_applied": True,
        "frame_filtering_applied": True,
        "disabled_selection_or_rejection_signals": [
            "client selected indices",
            "nearest-valid-mask substitution",
            "mask-ray residual",
            "quality-ranked alternative FPS seeds",
            "time-uniform sampling",
            "cumulative-trajectory-uniform sampling",
            "random sampling",
            "camera roll",
            "pose jump or relocalization rejection",
            "tracking state",
            "point-cloud support",
        ],
        "branches": {
            PRIMARY_MODE: {
                "policy": expected_policy,
                "algorithm": "official_training_single_seed_spherical_fps_v1",
                "selected_indices": selected_indices,
                "selected_frame_names": selected_names,
                "audit": view_selection,
            },
        },
    }
    # The client-selected payload is deliberately ignored and the wall-clock
    # timestamp is provenance only.  Neither may make an otherwise identical
    # capture unreusable when Unity retries ``/generate``.
    identity = {
        key: value
        for key, value in plan.items()
        if key not in {"created_at_utc", "client_selected_payload", "sha256"}
    }
    plan["sha256"] = canonical_sha256(identity)
    return plan


def _corrected_intrinsics(
    pose: Mapping[str, Any], raw_width: int, raw_height: int
) -> tuple[np.ndarray, dict[str, Any]]:
    fx, fy, cx, cy, source = intrinsics_for_pose(
        dict(pose), int(raw_width), int(raw_height)
    )
    values = np.asarray([fx, fy, cx, cy], dtype=np.float64)
    if not np.isfinite(values).all() or fx <= 0.0 or fy <= 0.0:
        raise ValueError("source AR intrinsics are invalid")
    K = np.asarray(
        [[fy, 0.0, cy], [0.0, fx, cx], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return K, {
        "source": str(source),
        "source_size_wh": [int(raw_width), int(raw_height)],
        "materialized_size_wh": [int(raw_height), int(raw_width)],
        "source_fx_fy_cx_cy": [float(fx), float(fy), float(cx), float(cy)],
        "materialized_fx_fy_cx_cy": [float(fy), float(fx), float(cy), float(cx)],
    }


def _source_assets_binding(
    data_dir: Path, mask_dir: Path, frame_names: Sequence[str]
) -> dict[str, Any]:
    rows = []
    for name in frame_names:
        image = data_dir / name
        mask = mask_dir / f"{Path(name).stem}.png"
        if not image.is_file():
            raise FileNotFoundError(f"missing uploaded RGB: {image}")
        if not mask.is_file():
            raise FileNotFoundError(
                f"missing final SAM2 mask; no replacement is allowed: {mask}"
            )
        rows.append(
            {
                "frame_name": name,
                "image": str(image.resolve()),
                "image_sha256": sha256_file(image),
                "mask": str(mask.resolve()),
                "mask_sha256": sha256_file(mask),
            }
        )
    poses_path = data_dir / "poses.txt"
    if not poses_path.is_file():
        raise FileNotFoundError(poses_path)
    binding = {
        "poses": str(poses_path.resolve()),
        "poses_sha256": sha256_file(poses_path),
        "frames": rows,
    }
    binding["sha256"] = canonical_sha256(binding)
    return binding


def materialize_all_view_phone_capture(
    *,
    session_id: str,
    data_dir: Path,
    mask_dir: Path,
    frame_names: Sequence[str],
    session_root: Path,
) -> dict[str, Any]:
    """Materialize the audited AR pixel/camera contract for every candidate."""

    names = [str(value) for value in frame_names]
    source_binding = _source_assets_binding(data_dir, mask_dir, names)
    report_path = session_root / "00_all_view_capture_report.json"
    if report_path.is_file():
        report = load_json(report_path)
        raw_path = Path(str(report.get("raw_cache_report", "")))
        if (
            report.get("format") != CAPTURE_FORMAT
            or report.get("passed") is not True
            or report.get("session_id") != session_id
            or report.get("source_assets_sha256") != source_binding["sha256"]
            or report.get("candidate_frame_names") != names
            or not raw_path.is_file()
            or report.get("raw_cache_report_sha256") != sha256_file(raw_path)
        ):
            raise RuntimeError(f"stale all-view capture report: {report_path}")
        return report

    axis_dir = session_root / "00_all_view_axis_corrected"
    raw_dir = session_root / "01_all_view_raw_cache"
    if axis_dir.exists() or raw_dir.exists():
        raise RuntimeError(
            "partial all-view materialization exists; preserve and inspect: "
            f"{session_root}"
        )
    axis_dir.mkdir(parents=True, exist_ok=False)
    axis_images = axis_dir / "images"
    axis_masks = axis_dir / "masks"
    axis_images.mkdir()
    axis_masks.mkdir()
    poses = read_phone_poses(data_dir / "poses.txt")
    missing_pose = [name for name in names if name not in poses]
    if missing_pose:
        raise FileNotFoundError(f"phone poses are missing: {missing_pose[:12]}")
    incompatible_pose = [
        name
        for name in names
        if str(poses[name].get("pose_binding") or "") != PHONE_POSE_BINDING
    ]
    if incompatible_pose:
        examples = [
            {
                "frame": name,
                "pose_binding": str(poses[name].get("pose_binding") or "unknown"),
            }
            for name in incompatible_pose[:8]
        ]
        raise RuntimeError(
            "phase-one A0 contract requires every camera pose to be relative "
            f"to the capture-start ARAnchor; expected={PHONE_POSE_BINDING!r} "
            f"incompatible={examples}"
        )
    incompatible_coordinate_frame = [
        name
        for name in names
        if str(poses[name].get("pose_coordinate_frame") or "")
        != PHONE_POSE_COORDINATE_FRAME
    ]
    if incompatible_coordinate_frame:
        examples = [
            {
                "frame": name,
                "pose_coordinate_frame": str(
                    poses[name].get("pose_coordinate_frame") or "unknown"
                ),
            }
            for name in incompatible_coordinate_frame[:8]
        ]
        raise RuntimeError(
            "phase-one A0 contract requires an explicit A0-relative coordinate "
            f"frame; expected={PHONE_POSE_COORDINATE_FRAME!r} "
            f"incompatible={examples}"
        )
    nontracking_a0 = [
        name
        for name in names
        if str(poses[name].get("capture_anchor_tracking_state") or "")
        != "Tracking"
    ]
    if nontracking_a0:
        examples = [
            {
                "frame": name,
                "capture_anchor_tracking_state": str(
                    poses[name].get("capture_anchor_tracking_state") or "unknown"
                ),
            }
            for name in nontracking_a0[:8]
        ]
        raise RuntimeError(
            "phase-one A0 contract forbids camera samples captured while A0 is "
            f"not Tracking; incompatible={examples}"
        )

    frames: list[CameraFrame] = []
    axis_records: list[dict[str, Any]] = []
    try:
        for index, name in enumerate(names):
            pose = poses[name]
            image_transform = str(pose.get("image_transform") or "unknown")
            if image_transform.lower() != "none":
                raise RuntimeError(
                    "phase-one AR axis contract requires "
                    "XRCpuImage.Transformation.None; "
                    f"frame={name} image_transform={image_transform!r}"
                )
            image_source = data_dir / name
            mask_source = mask_dir / f"{Path(name).stem}.png"
            with Image.open(image_source) as handle:
                source_image = handle.convert("RGB")
                raw_width, raw_height = source_image.size
                corrected_image = source_image.transpose(Image.Transpose.TRANSPOSE)
            with Image.open(mask_source) as handle:
                source_mask = handle.convert("L")
                mask_resized = source_mask.size != (raw_width, raw_height)
                if mask_resized:
                    source_mask = source_mask.resize(
                        (raw_width, raw_height), Image.Resampling.NEAREST
                    )
                corrected_mask = source_mask.transpose(Image.Transpose.TRANSPOSE)
            corrected_name = f"candidate_{index:04d}.png"
            image_path = axis_images / corrected_name
            mask_path = axis_masks / corrected_name
            corrected_image.save(image_path, format="PNG")
            corrected_mask.save(mask_path, format="PNG")
            K, intrinsic_record = _corrected_intrinsics(
                pose, raw_width, raw_height
            )
            rotation, translation = unity_pose_to_colmap_w2c(
                dict(pose), image_camera_rotation_degrees=0.0
            )
            T_W2C = np.eye(4, dtype=np.float64)
            T_W2C[:3, :3] = rotation
            T_W2C[:3, 3] = translation
            frames.append(
                CameraFrame(
                    source_index=index,
                    source_name=name,
                    image_path=image_path,
                    mask_path=mask_path,
                    K=K,
                    T_W2C=T_W2C,
                    camera_model="PINHOLE",
                    distortion=(),
                    pose_source=(
                        "synchronized_ARFoundation_A0_relative_pose_with_native_xy_transpose"
                    ),
                )
            )
            axis_records.append(
                {
                    "candidate_index": index,
                    "source_frame_name": name,
                    "source_image": str(image_source.resolve()),
                    "source_mask": str(mask_source.resolve()),
                    "axis_corrected_image": str(image_path.resolve()),
                    "axis_corrected_mask": str(mask_path.resolve()),
                    "axis_corrected_image_sha256": sha256_file(image_path),
                    "axis_corrected_mask_sha256": sha256_file(mask_path),
                    "mask_resized_to_source_image": bool(mask_resized),
                    "image_transform": image_transform,
                    "pixel_transform": "transpose_xy:(u,v)->(v,u)",
                    "intrinsics": intrinsic_record,
                    "pose_binding": str(pose.get("pose_binding") or "unknown"),
                    "pose_coordinate_frame": str(
                        pose.get("pose_coordinate_frame") or "unknown"
                    ),
                    "capture_anchor_tracking_state": str(
                        pose.get("capture_anchor_tracking_state") or "unknown"
                    ),
                    "strictly_synchronized": bool(pose.get("strictly_synchronized")),
                    "tracking_state": str(pose.get("tracking_state") or "unknown"),
                    "display_matrix_used_for_extrinsics": False,
                }
            )

        selection = deferred_selection_request(
            len(frames),
            SELECTED_VIEW_COUNT,
            policy="training_spherical_farthest",
            random_seed=0,
        )
        selection.update(
            {
                "requested_policies": list(SELECTION_MODES),
                "selection_implemented_by_server": True,
                "selection_occurs_after_all_view_o": True,
                "single_seed_spherical_fps_at_runtime_o": True,
                "quality_ranked_alternative_subsets": False,
            }
        )
        raw_path, row = materialize_raw_cache(
            output_dir=raw_dir,
            dataset_type="phone_live_phase1",
            source_path=data_dir,
            category="phone_capture",
            object_id=session_id,
            input_frames=frames,
            selection_request=selection,
            source_binding={
                "format": CAPTURE_FORMAT,
                "session_id": session_id,
                "source_assets": source_binding,
                "axis_contract": AXIS_CONTRACT,
                "pose_binding": PHONE_POSE_BINDING,
                "pose_coordinate_frame": PHONE_POSE_COORDINATE_FRAME,
                "axis_records": axis_records,
                "gravity_up_internal_W": [0.0, 1.0, 0.0],
            },
            extra_report={
                "geometry_mode": "pose_mask",
                "gravity_up_W": [0.0, 1.0, 0.0],
                "phone_pose_consumed": True,
                "point_cloud_consumed": False,
                "colmap_executed": False,
                "image_axis_contract": AXIS_CONTRACT,
                "pose_coordinate_frame": PHONE_POSE_COORDINATE_FRAME,
            },
        )
        report = {
            "format": CAPTURE_FORMAT,
            "created_at_utc": utc_now(),
            "passed": True,
            "session_id": session_id,
            "candidate_count": len(names),
            "candidate_frame_names": names,
            "all_candidates_retained": True,
            "frame_filtering_applied": False,
            "point_cloud_consumed": False,
            "source_assets": source_binding,
            "source_assets_sha256": source_binding["sha256"],
            "image_axis_contract": AXIS_CONTRACT,
            "pose_binding": PHONE_POSE_BINDING,
            "pose_coordinate_frame": PHONE_POSE_COORDINATE_FRAME,
            "axis_corrected_directory": str(axis_dir.resolve()),
            "axis_records": axis_records,
            "raw_cache_report": str(raw_path.resolve()),
            "raw_cache_report_sha256": sha256_file(raw_path),
            "object_key": row["object_key"],
            "scope_guard": (
                "All uploaded RGB/mask/pose candidates are retained. The physical "
                "x/y transpose and synchronized K transform are applied before "
                "runtime-O. Every pose is expressed relative to the real A0 "
                "ARAnchor created at capture start; displayMatrix and point "
                "clouds are not consumed."
            ),
        }
        atomic_json(report_path, report)
        return report
    except Exception:
        # Preserve a successfully written raw cache, but remove an axis-only
        # staging directory that cannot be bound by a completion report.
        if not raw_dir.exists() and axis_dir.exists():
            shutil.rmtree(axis_dir)
        raise


def _atomic_export_mesh(mesh: trimesh.Trimesh, destination: Path, file_type: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.stem}.tmp-{os.getpid()}{destination.suffix}"
    )
    mesh.export(temporary, file_type=file_type)
    os.replace(temporary, destination)


def _atomic_export_glb(mesh: trimesh.Trimesh, destination: Path) -> None:
    payload = trimesh.Scene(mesh).export(file_type="glb")
    if not isinstance(payload, (bytes, bytearray)):
        raise RuntimeError("trimesh GLB export returned non-binary data")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    temporary.write_bytes(bytes(payload))
    os.replace(temporary, destination)


def export_world_and_mobile_meshes(
    *,
    mesh_o: Path,
    mesh_frame_report: Path,
    runtime_input_manifest: Path,
    object_key: str,
    output_dir: Path,
    T_O2W_override: np.ndarray | None = None,
    T_O2W_refinement_report: Path | None = None,
) -> dict[str, Any]:
    """Export runtime-O directly into the immutable capture-A0 frame.

    ``T_O2W_override`` is a post-Mesh derivative.  It never mutates the frozen
    runtime cache and must be bound to a passed input-mask refinement report.
    """

    report_path = output_dir / "report.json"
    runtime_row = _one_runtime_object(runtime_input_manifest, object_key)
    with np.load(runtime_row["cache_npz"], allow_pickle=False) as payload:
        initial_T_O2W = validate_proper_similarity(
            np.asarray(payload["T_O2W"], dtype=np.float64), name="initial_T_O2W"
        )
    T_O2W = (
        initial_T_O2W
        if T_O2W_override is None
        else validate_proper_similarity(
            np.asarray(T_O2W_override, dtype=np.float64), name="selected_T_O2W"
        )
    )
    refinement_binding = None
    if T_O2W_refinement_report is not None:
        refinement_path = T_O2W_refinement_report.expanduser().resolve(strict=True)
        refinement = load_json(refinement_path)
        if refinement.get("passed") is not True:
            raise RuntimeError("O2W refinement report did not pass")
        if refinement.get("selected_T_O2W_sha256") != array_sha256(T_O2W):
            raise RuntimeError("O2W refinement report/override transform differs")
        if refinement.get("runtime_input_manifest") != str(
            runtime_input_manifest.resolve()
        ):
            raise RuntimeError("O2W refinement/runtime manifest binding differs")
        if refinement.get("mesh_o") != str(mesh_o.resolve()):
            raise RuntimeError("O2W refinement/source Mesh binding differs")
        refinement_binding = {
            "report": str(refinement_path),
            "report_sha256": sha256_file(refinement_path),
            "accepted": bool(refinement.get("accepted")),
            "decision": refinement.get("decision"),
        }
    elif T_O2W_override is not None:
        raise ValueError("T_O2W_override requires T_O2W_refinement_report")
    if report_path.is_file():
        report = load_json(report_path)
        required = [
            report.get("runtime_o_obj"),
            report.get("internal_world_obj"),
            report.get("internal_world_glb"),
            report.get("unity_capture_anchor_a0_armesh"),
        ]
        if (
            report.get("format") == WORLD_EXPORT_FORMAT
            and report.get("passed") is True
            and report.get("source_mesh_o_sha256") == sha256_file(mesh_o)
            and report.get("T_O2A0_sha256") == array_sha256(T_O2W)
            and all(isinstance(value, str) and Path(value).is_file() for value in required)
        ):
            return report
        raise RuntimeError(f"stale world Mesh export: {output_dir}")
    if output_dir.exists():
        raise RuntimeError(f"partial world Mesh export exists: {output_dir}")
    output_dir.mkdir(parents=True)
    frame_report = load_json(mesh_frame_report)
    if frame_report.get("passed") is not True:
        raise RuntimeError("native runtime-O Mesh frame report did not pass")
    if Path(str(frame_report.get("mesh", ""))).resolve() != mesh_o.resolve():
        raise RuntimeError("Mesh frame report path differs from source Mesh")
    if frame_report.get("mesh_sha256") != sha256_file(mesh_o):
        raise RuntimeError("Mesh frame report hash differs from source Mesh")
    validate_runtime_o_mesh_frame_contract(frame_report)
    transformed = []
    for source in load_meshes(mesh_o):
        world = source.copy()
        world.apply_transform(T_O2W)
        transformed.append(world)
    mesh_world = trimesh.util.concatenate(transformed)
    if not len(mesh_world.vertices) or not len(mesh_world.faces):
        raise RuntimeError("world Mesh is empty")
    runtime_copy = output_dir / "reconstructed_object_runtime_o.obj"
    world_obj = output_dir / "reconstructed_object_internal_world.obj"
    world_glb = output_dir / "reconstructed_object_internal_world.glb"
    mobile_a0 = output_dir / "reconstructed_object_unity_capture_anchor_a0.armesh"
    shutil.copy2(mesh_o, runtime_copy)
    _atomic_export_mesh(mesh_world, world_obj, "obj")
    _atomic_export_glb(mesh_world, world_glb)
    mobile_a0_report = build_mobile_overlay_mesh(
        world_obj,
        mobile_a0,
        source_coordinate_frame="internal_capture_anchor_a0",
        output_coordinate_frame=PHONE_POSE_COORDINATE_FRAME,
    )
    object_pose_unity_a0 = unity_object_pose_from_t_o2a0(T_O2W)
    report = {
        "format": WORLD_EXPORT_FORMAT,
        "created_at_utc": utc_now(),
        "passed": True,
        "object_key": object_key,
        "source_mesh_o": str(mesh_o.resolve()),
        "source_mesh_o_sha256": sha256_file(mesh_o),
        "mesh_frame_report": str(mesh_frame_report.resolve()),
        "mesh_frame_report_sha256": sha256_file(mesh_frame_report),
        "runtime_input_manifest": str(runtime_input_manifest.resolve()),
        "runtime_input_manifest_sha256": sha256_file(runtime_input_manifest),
        "initial_T_O2A0": initial_T_O2W.tolist(),
        "initial_T_O2A0_sha256": array_sha256(initial_T_O2W),
        "T_O2A0": T_O2W.tolist(),
        "T_O2A0_sha256": array_sha256(T_O2W),
        "T_O2A0_source": (
            "input_mask_o2w_refinement_selected"
            if T_O2W_override is not None
            else "immutable_runtime_cache"
        ),
        "T_O2A0_refinement": refinement_binding,
        "runtime_W_semantics": "internal_capture_anchor_a0",
        "runtime_o_obj": str(runtime_copy.resolve()),
        "runtime_o_obj_sha256": sha256_file(runtime_copy),
        "internal_world_obj": str(world_obj.resolve()),
        "internal_world_obj_sha256": sha256_file(world_obj),
        "internal_world_glb": str(world_glb.resolve()),
        "internal_world_glb_sha256": sha256_file(world_glb),
        "unity_capture_anchor_a0_armesh": str(mobile_a0.resolve()),
        "unity_capture_anchor_a0_armesh_sha256": sha256_file(mobile_a0),
        "unity_capture_anchor_a0_armesh_report": str(
            mobile_a0.with_suffix(mobile_a0.suffix + ".json").resolve()
        ),
        "mobile_overlay": mobile_a0_report,
        "object_pose_unity_a0": object_pose_unity_a0,
        "projection_world_contract": "Mesh_A0=T_O2A0@Mesh_O",
        "unity_capture_anchor_contract": (
            "Mesh_A0=T_O2A0@Mesh_O; reflect internal z into Unity z, then "
            "attach the returned A0-local Mesh below the capture A0 with "
            "identity local transform"
        ),
    }
    atomic_json(report_path, report)
    return report


def _make_contact_sheet(
    images: Sequence[Image.Image], labels: Sequence[str], destination: Path
) -> None:
    if len(images) != len(labels) or not images:
        raise ValueError("contact-sheet image/label count differs")
    columns = 2
    cell_width, cell_height, header = 480, 270, 28
    rows = int(math.ceil(len(images) / columns))
    sheet = Image.new(
        "RGB", (columns * cell_width, rows * (cell_height + header)), (20, 20, 20)
    )
    draw = ImageDraw.Draw(sheet)
    for index, (source, label) in enumerate(zip(images, labels)):
        image = source.copy()
        image.thumbnail((cell_width, cell_height), Image.Resampling.LANCZOS)
        x = (index % columns) * cell_width
        y = (index // columns) * (cell_height + header)
        sheet.paste(image, (x + (cell_width - image.width) // 2, y + header))
        draw.text((x + 8, y + 7), f"view {index}: {label}", fill=(238, 238, 238))
    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination)


def _make_mobile_contact_sheet(
    images: Sequence[Image.Image], labels: Sequence[str], destination: Path
) -> None:
    """Write a legible portrait index while preserving full-resolution files."""

    if len(images) != len(labels) or not images:
        raise ValueError("mobile contact-sheet image/label count differs")
    columns = 2
    cell_width, cell_height, header = 720, 1600, 44
    rows = int(math.ceil(len(images) / columns))
    sheet = Image.new(
        "RGB", (columns * cell_width, rows * (cell_height + header)), (20, 20, 20)
    )
    draw = ImageDraw.Draw(sheet)
    for index, (source, label) in enumerate(zip(images, labels)):
        image = source.copy()
        image.thumbnail(
            (cell_width - 12, cell_height - 12), Image.Resampling.LANCZOS
        )
        x = (index % columns) * cell_width
        y = (index // columns) * (cell_height + header)
        sheet.paste(image, (x + (cell_width - image.width) // 2, y + header))
        draw.text((x + 8, y + 10), label, fill=(238, 238, 238))
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp.jpg")
    sheet.save(temporary, quality=95, subsampling=0)
    os.replace(temporary, destination)


def restore_phone_orientation_contours(
    *,
    calibrated_contour_report: Path,
    selected_source_names: Sequence[str],
    data_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Transpose calibrated overlays back onto the uploaded phone orientation."""

    source_hash = sha256_file(calibrated_contour_report)
    report_path = output_dir / "report.json"
    if report_path.is_file():
        report = load_json(report_path)
        if (
            report.get("format") == PHONE_CONTOUR_FORMAT
            and report.get("passed") is True
            and report.get("calibrated_contour_report_sha256") == source_hash
        ):
            return report
        raise RuntimeError(f"stale phone contour export: {output_dir}")
    if output_dir.exists():
        raise RuntimeError(f"partial phone contour export exists: {output_dir}")
    output_dir.mkdir(parents=True)
    calibrated = load_json(calibrated_contour_report)
    views = list(calibrated.get("views") or [])
    names = [str(value) for value in selected_source_names]
    if calibrated.get("passed") is not True or len(views) != len(names):
        raise RuntimeError("calibrated contour report/selected frame count differs")
    records = []
    thumbnails: list[Image.Image] = []
    labels: list[str] = []
    for index, (view, name) in enumerate(zip(views, names)):
        source = (data_dir / name).resolve(strict=True)
        overlay_calibrated = Path(view["overlay"]).resolve(strict=True)
        with Image.open(overlay_calibrated) as handle:
            overlay = handle.convert("RGB").transpose(Image.Transpose.TRANSPOSE)
        with Image.open(source) as handle:
            source_size = handle.size
        if overlay.size != source_size:
            raise RuntimeError(
                f"inverse-transposed contour size differs for frame={name}: "
                f"overlay={overlay.size} source={source_size}"
            )
        original_copy = output_dir / "原始手机输入8帧" / f"view_{index:02d}_{name}"
        overlay_path = (
            output_dir
            / "原始手机朝向Mesh轮廓叠加"
            / f"view_{index:02d}_{Path(name).stem}_contour.png"
        )
        original_copy.parent.mkdir(parents=True, exist_ok=True)
        overlay_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, original_copy)
        overlay.save(overlay_path)
        thumbnails.append(overlay.copy())
        labels.append(name)
        records.append(
            {
                "view_index": index,
                "source_frame_name": name,
                "uploaded_rgb": str(source),
                "uploaded_rgb_sha256": sha256_file(source),
                "copied_uploaded_rgb": str(original_copy.resolve()),
                "phone_orientation_overlay": str(overlay_path.resolve()),
                "phone_orientation_overlay_sha256": sha256_file(overlay_path),
                "calibrated_overlay": str(overlay_calibrated),
                "inverse_pixel_transform": "transpose_xy:(u,v)->(v,u)",
            }
        )
    overview = output_dir / "SS30K_SLat30K_原始手机输入8帧轮廓总览.png"
    _make_contact_sheet(thumbnails, labels, overview)
    report = {
        "format": PHONE_CONTOUR_FORMAT,
        "created_at_utc": utc_now(),
        "passed": True,
        "calibrated_contour_report": str(calibrated_contour_report.resolve()),
        "calibrated_contour_report_sha256": source_hash,
        "selected_source_frame_names": names,
        "axis_contract": AXIS_CONTRACT,
        "projection_formula": calibrated.get("projection_formula"),
        "post_projection_display_transform": "inverse transpose_xy only",
        "views": records,
        "overview": str(overview.resolve()),
        "overview_sha256": sha256_file(overview),
        "scope_guard": (
            "The cyan contour is first projected in the calibrated transposed "
            "camera frame, then both RGB and contour are transposed back together. "
            "No pose, scale, or Mesh fitting is performed."
        ),
    }
    atomic_json(report_path, report)
    return report


def _runtime_o_binding(runtime_manifest: Path, object_key: str) -> dict[str, Any]:
    row = _one_runtime_object(runtime_manifest, object_key)
    with np.load(row["cache_npz"], allow_pickle=False) as payload:
        T_O2W = np.asarray(payload["T_O2W"], dtype=np.float64)
    return {
        "runtime_input_manifest": str(runtime_manifest.resolve()),
        "runtime_input_manifest_sha256": sha256_file(runtime_manifest),
        "raw_cache_sha256": row["source_raw_cache_sha256"],
        "all_input_view_count": int(row["all_input_view_count"]),
        "o_frozen_before_view_selection": bool(row["o_frozen_before_view_selection"]),
        "T_O2W": T_O2W,
        "T_O2W_sha256": array_sha256(T_O2W),
        "selected_frame_names": list(row["selected_frame_names"]),
        "selected_source_frame_names": list(
            row["view_selection"].get("selected_source_frame_names") or []
        ),
        "selected_source_view_indices": [
            int(value)
            for value in row["view_selection"].get(
                "selected_source_view_indices", []
            )
        ],
        "view_selection": dict(row["view_selection"]),
    }


def _branch_dir(session_root: Path, mode: str) -> Path:
    order = SELECTION_MODES.index(mode) + 1
    return session_root / "branches" / f"{order:02d}_{mode}"


def _ensure_current_branch_runtime(
    *,
    config: ServerConfig,
    session_root: Path,
    raw_cache_report: Path,
    object_key: str,
    mode: str,
) -> dict[str, Any]:
    """Freeze all-view O and exact training FPS8 without loading SS/SLat."""

    branch_dir = _branch_dir(session_root, mode)
    branch_dir.mkdir(parents=True, exist_ok=True)
    log_path = branch_dir / "pipeline.log"
    environment = _environment(config.gpu)
    python = str(config.python.resolve())
    runtime_dir = branch_dir / "02_runtime_o_all_views_then_spherical_fps8"
    runtime_manifest = runtime_dir / "runtime_input_manifest.json"
    if not _passed(runtime_manifest):
        _run_stage(
            f"{mode}: 02 all-view O then training spherical FPS8",
            [
                python,
                "-u",
                "-m",
                "manual_mesh_reconstruction.runtime_o",
                "--raw_cache_report",
                str(raw_cache_report),
                "--output_dir",
                str(runtime_dir),
                "--geometry_mode",
                "pose_mask",
                "--view_selection_policy",
                "training_spherical_farthest_valid_mask",
                "--selected_view_count",
                str(config.selected_view_count),
                "--min_completed_objects",
                "1",
                "--pose_mask_object_frame_policy",
                OFFICIAL_COMPATIBLE_Z_UP_OBJECT_FRAME_POLICY,
                "--gravity_up_w",
                "0",
                "1",
                "0",
                "--resume",
                "--object",
                object_key,
            ],
            environment=environment,
            log_path=log_path,
            dry_run=False,
        )
    binding = _runtime_o_binding(runtime_manifest, object_key)
    if not binding["o_frozen_before_view_selection"]:
        raise RuntimeError("runtime-O was not frozen before spherical FPS")
    if binding["all_input_view_count"] < config.selected_view_count:
        raise RuntimeError("runtime-O did not consume the complete candidate domain")
    view_selection = dict(binding["view_selection"])
    if (
        view_selection.get("policy")
        != "pose_mask_training_exact_spherical_farthest_valid_mask"
        or view_selection.get("algorithm")
        != "official_training_single_seed_spherical_fps_v1"
        or view_selection.get("quality_gate_used_for_selection") is not False
    ):
        raise RuntimeError(
            "runtime selection is not the exact training spherical FPS contract"
        )
    selected_names = [str(value) for value in binding["selected_source_frame_names"]]
    selected_indices = [int(value) for value in binding["selected_source_view_indices"]]
    if len(selected_names) != config.selected_view_count:
        raise RuntimeError("runtime spherical FPS did not produce exactly eight views")
    return {
        "branch_dir": branch_dir,
        "log_path": log_path,
        "environment": environment,
        "python": python,
        "runtime_manifest": runtime_manifest,
        "binding": binding,
        "selection": {
            "policy": view_selection["policy"],
            "algorithm": view_selection["algorithm"],
            "selected_indices": selected_indices,
            "selected_frame_names": selected_names,
            "audit": view_selection,
        },
    }


def _run_current_branch(
    *,
    config: ServerConfig,
    session_id: str,
    session_root: Path,
    raw_cache_report: Path,
    object_key: str,
    mode: str,
    data_dir: Path,
) -> dict[str, Any]:
    branch_dir = _branch_dir(session_root, mode)
    branch_report_path = branch_dir / "branch_report.json"
    if branch_report_path.is_file():
        existing = load_json(branch_report_path)
        if (
            existing.get("format") == BRANCH_FORMAT
            and existing.get("passed") is True
            and existing.get("mode") == mode
            and existing.get("raw_cache_report_sha256")
            == sha256_file(raw_cache_report)
            and (existing.get("selection") or {}).get("algorithm")
            == "official_training_single_seed_spherical_fps_v1"
        ):
            return existing
        raise RuntimeError(f"stale branch report: {branch_report_path}")
    prepared = _ensure_current_branch_runtime(
        config=config,
        session_root=session_root,
        raw_cache_report=raw_cache_report,
        object_key=object_key,
        mode=mode,
    )
    branch_dir = prepared["branch_dir"]
    log_path = prepared["log_path"]
    environment = prepared["environment"]
    python = prepared["python"]
    runtime_manifest = prepared["runtime_manifest"]
    binding = prepared["binding"]
    selection = prepared["selection"]
    selected_names = list(selection["selected_frame_names"])

    if config.capture_only:
        report = {
            "format": BRANCH_FORMAT,
            "created_at_utc": utc_now(),
            "passed": True,
            "capture_only": True,
            "session_id": session_id,
            "mode": mode,
            "selection": dict(selection),
            "selected_source_frame_names": selected_names,
            "raw_cache_report": str(raw_cache_report.resolve()),
            "raw_cache_report_sha256": sha256_file(raw_cache_report),
            "runtime_o": {
                key: value
                for key, value in binding.items()
                if key != "T_O2W"
            },
            "runtime_input_manifest": str(runtime_manifest.resolve()),
            "mesh": None,
            "contours": None,
        }
        atomic_json(branch_report_path, report)
        return report

    model_dir = branch_dir / "03_dino_only_model_input"
    model_manifest = model_dir / "model_input_manifest.json"
    if not _passed(model_manifest):
        _run_stage(
            f"{mode}: 03 DINO-only model input",
            [
                python,
                "-u",
                "-m",
                "manual_mesh_reconstruction.model_inputs",
                "--runtime_input_manifest",
                str(runtime_manifest),
                "--output_dir",
                str(model_dir),
                "--pretrained",
                PRETRAINED,
                "--device",
                "cuda",
                "--resume",
                "--object",
                object_key,
            ],
            environment=environment,
            log_path=log_path,
            dry_run=False,
        )

    current_dir = branch_dir / "04_current_ss30k_slat30k"
    current_manifest = current_dir / "inference_manifest.json"
    if not _passed(current_manifest):
        _run_stage(
            f"{mode}: 04 current SS30K+SLat30K",
            [
                python,
                "-u",
                "-m",
                "manual_mesh_reconstruction.current_model",
                "--model_input_manifest",
                str(model_manifest),
                "--native_ss_report",
                str(SS30K_REPORT.path),
                "--native_slat_checkpoint",
                str(SLAT30K_CHECKPOINT.path),
                "--expected_slat_step",
                str(SLAT_STEP),
                "--cross_deployment_bridge_report",
                str(ABC_R_BRIDGE.path),
                "--stock_slat_freeze",
                str(STOCK_SLAT_FREEZE.path),
                "--output_dir",
                str(current_dir),
                "--pretrained",
                PRETRAINED,
                "--seeds",
                str(config.seed),
                "--weights",
                "ema",
                "--device",
                "cuda",
                "--amp_dtype",
                "bf16",
                "--object",
                object_key,
            ],
            environment=environment,
            log_path=log_path,
            dry_run=False,
        )
    inference = _one_inference_record(current_manifest, object_key, config.seed)
    mesh_o = Path(inference["mesh"]).resolve(strict=True)
    mesh_frame_report = Path(inference["result"]).resolve(strict=True)

    o2w_refinement_dir = branch_dir / "04b_input_o2w_refinement"
    o2w_refinement_report = o2w_refinement_dir / "report.json"
    if not _passed(o2w_refinement_report):
        _run_stage(
            f"{mode}: 04b all-input mask O2W refinement",
            [
                python,
                "-u",
                "-m",
                "manual_mesh_reconstruction.optimize_o2w",
                "--runtime_input_manifest",
                str(runtime_manifest),
                "--mesh_o",
                str(mesh_o),
                "--mesh_frame_report",
                str(mesh_frame_report),
                "--output_dir",
                str(o2w_refinement_dir),
                "--object",
                object_key,
                "--max_optimization_views",
                str(config.o2w_refinement_max_views),
                "--iterations",
                str(config.o2w_refinement_iterations),
                "--render_long_side",
                str(config.o2w_refinement_render_long_side),
                "--batch_size",
                str(config.o2w_refinement_batch_size),
                "--optimization_face_limit",
                str(config.o2w_refinement_face_limit),
                "--contour_width",
                str(config.contour_width),
                "--resume",
            ],
            environment=environment,
            log_path=log_path,
            dry_run=False,
        )
    o2w_refinement = load_json(o2w_refinement_report)
    if o2w_refinement.get("passed") is not True:
        raise RuntimeError("all-input O2W refinement report did not pass")
    selected_o2w_npz = Path(
        o2w_refinement["selected_T_O2W_npz"]
    ).resolve(strict=True)
    if o2w_refinement.get("selected_T_O2W_npz_sha256") != sha256_file(
        selected_o2w_npz
    ):
        raise RuntimeError("selected O2W NPZ binding changed")
    with np.load(selected_o2w_npz, allow_pickle=False) as payload:
        selected_T_O2W = validate_proper_similarity(
            np.asarray(payload["T_O2W"], dtype=np.float64),
            name="selected_T_O2W",
        )
    if o2w_refinement.get("selected_T_O2W_sha256") != array_sha256(
        selected_T_O2W
    ):
        raise RuntimeError("selected O2W matrix/report binding changed")

    mesh_exports = export_world_and_mobile_meshes(
        mesh_o=mesh_o,
        mesh_frame_report=mesh_frame_report,
        runtime_input_manifest=runtime_manifest,
        object_key=object_key,
        output_dir=branch_dir / "05_mesh_exports",
        T_O2W_override=selected_T_O2W,
        T_O2W_refinement_report=o2w_refinement_report,
    )

    contour_dir = branch_dir / "06_calibrated_input_contours"
    contour_report = contour_dir / "report.json"
    if not _passed(contour_report):
        _run_stage(
            f"{mode}: 06 calibrated selected-input contours",
            [
                python,
                "-u",
                "-m",
                "manual_mesh_reconstruction.contours",
                "--runtime_input_manifest",
                str(runtime_manifest),
                "--mesh_o",
                str(mesh_o),
                "--mesh_frame_report",
                str(mesh_frame_report),
                "--T_O2W_npz",
                str(selected_o2w_npz),
                "--output_dir",
                str(contour_dir),
                "--object",
                object_key,
                "--contour_width",
                str(config.contour_width),
                "--method_label",
                f"phase1 {mode} SS30K+SLat30K",
                "--overview_name",
                f"{mode}_标定输入8帧轮廓总览.png",
                "--resume",
            ],
            environment=environment,
            log_path=log_path,
            dry_run=False,
        )
    phone_contours = restore_phone_orientation_contours(
        calibrated_contour_report=contour_report,
        selected_source_names=selected_names,
        data_dir=data_dir,
        output_dir=branch_dir / "07_original_phone_input_contours",
    )

    preview_dir = branch_dir / "08_runtime_o_mesh_previews"
    preview_report = preview_dir / "preview_report.json"
    if not _passed(preview_report):
        _run_stage(
            f"{mode}: 08 runtime-O Mesh previews",
            [
                python,
                "-u",
                "-m",
                "manual_mesh_reconstruction.render_mesh",
                "--mesh",
                str(mesh_o),
                "--output_dir",
                str(preview_dir),
                "--device",
                "cuda",
                "--method_label",
                f"phase1 {mode} SS30K+SLat30K",
                "--resume",
            ],
            environment=environment,
            log_path=log_path,
            dry_run=False,
        )

    report = {
        "format": BRANCH_FORMAT,
        "created_at_utc": utc_now(),
        "passed": True,
        "capture_only": False,
        "session_id": session_id,
        "mode": mode,
        "selection": dict(selection),
        "selected_source_frame_names": selected_names,
        "raw_cache_report": str(raw_cache_report.resolve()),
        "raw_cache_report_sha256": sha256_file(raw_cache_report),
        "runtime_o": {
            key: value for key, value in binding.items() if key != "T_O2W"
        },
        "runtime_input_manifest": str(runtime_manifest.resolve()),
        "model_input_manifest": str(model_manifest.resolve()),
        "inference_manifest": str(current_manifest.resolve()),
        "input_o2w_refinement": {
            "report": str(o2w_refinement_report.resolve()),
            "report_sha256": sha256_file(o2w_refinement_report),
            "accepted": bool(o2w_refinement["accepted"]),
            "selected_T_O2W_npz": str(selected_o2w_npz),
            "selected_T_O2W_sha256": o2w_refinement[
                "selected_T_O2W_sha256"
            ],
            "optimized_contour_overview": o2w_refinement["contours"][
                "selected_optimized_overview"
            ],
        },
        "mesh": mesh_exports,
        "calibrated_input_contours": {
            "report": str(contour_report.resolve()),
            "overview": load_json(contour_report)["overview"],
        },
        "original_phone_input_contours": {
            "report": str((branch_dir / "07_original_phone_input_contours/report.json").resolve()),
            "overview": phone_contours["overview"],
        },
        "mesh_previews": {
            "report": str(preview_report.resolve()),
            "contact_sheet": load_json(preview_report)["contact_sheet"],
        },
        "vggt_model_loaded": False,
        "vggt_model_executed": False,
        "projection_formula": "Mesh_O -> T_O2W -> Mesh_W -> T_W2C -> K_raw",
    }
    atomic_json(branch_report_path, report)
    return report


def verify_shared_all_view_o(branches: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Fail closed unless every completed branch has the exact same all-view O."""

    if not branches:
        raise ValueError("no branches were supplied for O verification")
    bindings = []
    matrices = []
    for mode, branch in branches.items():
        runtime = Path(str(branch["runtime_input_manifest"])).resolve(strict=True)
        object_key = _one_runtime_object(runtime, None)["object_key"]
        binding = _runtime_o_binding(runtime, object_key)
        bindings.append(
            {
                "mode": mode,
                "T_O2W_sha256": binding["T_O2W_sha256"],
                "raw_cache_sha256": binding["raw_cache_sha256"],
                "all_input_view_count": binding["all_input_view_count"],
                "o_frozen_before_view_selection": binding[
                    "o_frozen_before_view_selection"
                ],
            }
        )
        matrices.append(binding["T_O2W"])
    reference = matrices[0]
    exact = all(np.array_equal(reference, value) for value in matrices[1:])
    same_raw = len({row["raw_cache_sha256"] for row in bindings}) == 1
    all_frozen = all(row["o_frozen_before_view_selection"] for row in bindings)
    same_count = len({row["all_input_view_count"] for row in bindings}) == 1
    if not (exact and same_raw and all_frozen and same_count):
        raise RuntimeError(
            "branch O contract differs: "
            f"exact_T_O2W={exact} same_raw={same_raw} "
            f"all_frozen={all_frozen} same_count={same_count}"
        )
    return {
        "passed": True,
        "exact_T_O2W_array_equal": True,
        "same_all_view_raw_cache": True,
        "all_branches_freeze_o_before_selection": True,
        "all_input_view_count": bindings[0]["all_input_view_count"],
        "T_O2W_sha256": bindings[0]["T_O2W_sha256"],
        "branches": bindings,
    }


def _session_root(session_id: str) -> Path:
    if _CONFIG is None:
        raise RuntimeError("manual phone server is not configured")
    return _CONFIG.output_root / "reconstructions" / session_id


def _summary_path(session_id: str) -> Path:
    return _session_root(session_id) / "phase1_session_report.json"


def _write_summary(
    *,
    session_id: str,
    selection_plan: Mapping[str, Any],
    capture: Mapping[str, Any],
    branches: Mapping[str, Any],
    branch_status: Mapping[str, str],
    shared_o: Mapping[str, Any] | None,
    background_error: str | None = None,
) -> Path:
    completed = all(value == "complete" for value in branch_status.values())
    failed = any(value == "failed" for value in branch_status.values())
    primary_mode = (
        PRIMARY_MODE
        if PRIMARY_MODE in branch_status
        else next(iter(branch_status), PRIMARY_MODE)
    )
    summary = {
        "format": FORMAT,
        "created_at_utc": utc_now(),
        "passed": bool(completed and not failed and shared_o is not None),
        "status": "failed" if failed else ("complete" if completed else "running"),
        "session_id": session_id,
        "output_root": str(_CONFIG.output_root.resolve()) if _CONFIG else None,
        "selection_plan": dict(selection_plan),
        "capture": dict(capture),
        "branch_status": dict(branch_status),
        "branches": dict(branches),
        "shared_all_view_o": None if shared_o is None else dict(shared_o),
        "background_error": background_error,
        "primary_mode": primary_mode,
        "frame_filtering_applied": True,
        "point_cloud_consumed": False,
        "axis_contract": AXIS_CONTRACT,
        "scope_guard": (
            "One capture pass supplies every candidate view to one deterministic "
            "all-view model-O contract. One exact official-training single-seed "
            "spherical FPS8 subset is chosen after O. No time/trajectory/random "
            "alternative and no quality-ranked subset search is executed."
        ),
    }
    path = _summary_path(session_id)
    atomic_json(path, summary)
    return path


def _current_capture_context(payload: Mapping[str, Any]) -> tuple[
    str, Path, Path, list[str], dict[str, Any]
]:
    transport_server.legacy._load_current_session()
    session_id = str(transport_server.legacy.current_session_id)
    if not _SAFE_SESSION_ID.fullmatch(session_id):
        raise ValueError(f"unsafe or missing session id: {session_id!r}")
    requested_session_id = str(payload.get("session_id") or "")
    if requested_session_id and requested_session_id != session_id:
        raise RuntimeError(
            "request session differs from the active capture session: "
            f"requested={requested_session_id!r} active={session_id!r}"
        )
    data_dir = Path(transport_server.legacy.current_data_dir).resolve(strict=True)
    mask_dir = Path(transport_server.legacy.current_mask_dir).resolve(strict=True)
    frame_names = [str(value) for value in transport_server.legacy._list_session_images()]
    poses = read_phone_poses(data_dir / "poses.txt")
    if len(frame_names) < SELECTED_VIEW_COUNT:
        raise ValueError(
            f"uploaded frame count {len(frame_names)} < required {SELECTED_VIEW_COUNT}"
        )
    missing = [name for name in frame_names if name not in poses]
    if missing:
        raise ValueError(
            "uploaded RGB frames lack camera poses: " + ", ".join(missing[:12])
        )
    wrong_binding = [
        name
        for name in frame_names
        if str(poses[name].get("pose_binding") or "") != PHONE_POSE_BINDING
    ]
    if wrong_binding:
        observed = sorted(
            {
                str(poses[name].get("pose_binding") or "unknown")
                for name in wrong_binding
            }
        )
        raise RuntimeError(
            "the installed phone client is not using the capture-start A0 "
            f"coordinate contract; expected={PHONE_POSE_BINDING!r} "
            f"observed={observed}"
        )
    _source_assets_binding(data_dir, mask_dir, frame_names)
    return session_id, data_dir, mask_dir, frame_names, poses


def collection_input_qc():
    """Check capture completeness; the sole FPS8 selection runs after all-view O."""

    try:
        payload = request.get_json(silent=True) or {}
        session_id, _data, _masks, names, _poses = _current_capture_context(payload)
        return jsonify(
            {
                "status": "ok",
                "message": (
                    "输入可用：全部候选先构建统一O，再执行唯一的训练一致球面最远8帧"
                ),
                "session_id": session_id,
                "client_selected_indices": payload.get("selected"),
                "selected_indices": [],
                "selection_deferred_until_all_view_o": True,
                "selection_policy": (
                    "official_training_single_seed_spherical_fps_v1"
                ),
                "point_cloud_required": False,
                "geometry_mode": "pose_mask",
                "input_qc": {
                    "profile": "phase1_spherical_fps_readiness_only_v2",
                    "qc_pass": True,
                    "frame_filtering_applied": True,
                    "candidate_count": len(names),
                    "all_candidate_masks_and_poses_present": True,
                },
            }
        ), 200
    except Exception as error:
        transport_server.legacy.logging.exception(
            "manual phase-one readiness check failed"
        )
        return jsonify(
            {
                "status": "warning",
                "message": f"无法构造第一阶段输入：{error}",
                "frame_filtering_applied": False,
            }
        ), 200


def _runtime_o_prepare_path(session_id: str) -> Path:
    return _session_root(session_id) / "runtime_o_prepare.json"


def collection_prepare_runtime_o():
    """Freeze all-view Runtime-O before model inference and view selection."""

    try:
        if _CONFIG is None:
            raise RuntimeError("manual phase-one server is not configured")
        payload = request.get_json(silent=True) or {}
        lifecycle_generation = int(payload.get("lifecycle_generation", -1))
        if lifecycle_generation < 0:
            raise ValueError("prepare request lacks lifecycle_generation")
        session_id, data_dir, mask_dir, frame_names, poses = (
            _current_capture_context(payload)
        )
        session_root = _session_root(session_id)
        session_root.mkdir(parents=True, exist_ok=True)
        with _PIPELINE_LOCK:
            transport_server._set_latest_progress(
                {
                    "status": "running",
                    "stage": "freeze_all_view_runtime_o",
                    "session_id": session_id,
                    "candidate_count": len(frame_names),
                }
            )
            capture = materialize_all_view_phone_capture(
                session_id=session_id,
                data_dir=data_dir,
                mask_dir=mask_dir,
                frame_names=frame_names,
                session_root=session_root,
            )
            raw_cache_report = Path(capture["raw_cache_report"]).resolve(strict=True)
            object_key = str(capture["object_key"])
            prepared = _ensure_current_branch_runtime(
                config=_CONFIG,
                session_root=session_root,
                raw_cache_report=raw_cache_report,
                object_key=object_key,
                mode=PRIMARY_MODE,
            )
        runtime_binding = prepared["binding"]
        selection = build_selection_plan(
            frame_names,
            poses,
            runtime_binding=runtime_binding,
            client_selected=payload.get("selected"),
            count=_CONFIG.selected_view_count,
        )
        plan_path = session_root / "selection_plan.json"
        if plan_path.is_file():
            existing_plan = load_json(plan_path)
            if existing_plan.get("sha256") != selection.get("sha256"):
                raise RuntimeError("prepared selection plan changed within one session")
        else:
            atomic_json(plan_path, selection)
        requested_pose = unity_object_pose_from_t_o2a0(runtime_binding["T_O2W"])
        requested_pose_sha256 = canonical_sha256(requested_pose)
        prepared_report = {
            "format": RUNTIME_O_PREPARE_FORMAT,
            "created_at_utc": utc_now(),
            "passed": True,
            "session_id": session_id,
            "lifecycle_generation": lifecycle_generation,
            "candidate_count": len(frame_names),
            "capture_report": str(
                (session_root / "00_all_view_capture_report.json").resolve()
            ),
            "capture_report_sha256": sha256_file(
                session_root / "00_all_view_capture_report.json"
            ),
            "runtime_input_manifest": str(
                Path(prepared["runtime_manifest"]).resolve()
            ),
            "runtime_input_manifest_sha256": runtime_binding[
                "runtime_input_manifest_sha256"
            ],
            "runtime_o_sha256": runtime_binding["T_O2W_sha256"],
            "raw_cache_sha256": runtime_binding["raw_cache_sha256"],
            "o_frozen_before_view_selection": True,
            "selection_plan": selection,
            "object_pose_unity_a0": requested_pose,
            "requested_pose_sha256": requested_pose_sha256,
            "model_inference_started": False,
        }
        prepare_path = _runtime_o_prepare_path(session_id)
        if prepare_path.is_file():
            existing = load_json(prepare_path)
            identity_keys = (
                "format",
                "session_id",
                "lifecycle_generation",
                "runtime_o_sha256",
                "requested_pose_sha256",
                "capture_report_sha256",
            )
            if any(existing.get(key) != prepared_report.get(key) for key in identity_keys):
                raise RuntimeError("stale Runtime-O prepare binding exists")
            prepared_report = existing
        else:
            atomic_json(prepare_path, prepared_report)
        transport_server._set_latest_progress(
            {
                "status": "ready_for_model",
                "stage": "runtime_o_frozen",
                "session_id": session_id,
                "runtime_o_sha256": runtime_binding["T_O2W_sha256"],
            }
        )
        return jsonify(
            {
                "status": "success",
                "message": "全部视图 Runtime-O 已冻结，可直接启动模型",
                "session_id": session_id,
                "lifecycle_generation": lifecycle_generation,
                "runtime_o_sha256": prepared_report["runtime_o_sha256"],
                "requested_pose_sha256": prepared_report[
                    "requested_pose_sha256"
                ],
                "object_pose_unity_a0": prepared_report[
                    "object_pose_unity_a0"
                ],
                "selected_indices": selection["branches"][PRIMARY_MODE][
                    "selected_indices"
                ],
                "model_inference_started": False,
            }
        ), 200
    except Exception as error:
        transport_server.legacy.logging.exception(
            "manual phase-one Runtime-O preparation failed"
        )
        return jsonify(
            {
                "status": "error",
                "message": f"无法冻结 Runtime-O：{error}",
                "error_detail": str(error),
                "session_id": transport_server.legacy.current_session_id,
            }
        ), 500


def collection_generate():
    try:
        if _CONFIG is None:
            raise RuntimeError("manual phase-one server is not configured")
        payload = request.get_json(silent=True) or {}
        session_id, data_dir, mask_dir, frame_names, poses = (
            _current_capture_context(payload)
        )
        session_root = _session_root(session_id)
        session_root.mkdir(parents=True, exist_ok=True)
        prepare_path = _runtime_o_prepare_path(session_id)
        if not prepare_path.is_file():
            raise RuntimeError(
                "Runtime-O prepare report is missing; call /prepare_runtime_o first"
            )
        prepared_runtime = load_json(prepare_path)
        if (
            prepared_runtime.get("format") != RUNTIME_O_PREPARE_FORMAT
            or prepared_runtime.get("passed") is not True
            or prepared_runtime.get("session_id") != session_id
        ):
            raise RuntimeError("Runtime-O prepare report is invalid")
        if int(payload.get("lifecycle_generation", -1)) != int(
            prepared_runtime["lifecycle_generation"]
        ):
            raise RuntimeError("generate lifecycle differs from Runtime-O prepare")
        if str(payload.get("runtime_o_sha256") or "") != str(
            prepared_runtime["runtime_o_sha256"]
        ):
            raise RuntimeError("generate Runtime-O hash differs from prepare")
        if str(payload.get("requested_pose_sha256") or "") != str(
            prepared_runtime["requested_pose_sha256"]
        ):
            raise RuntimeError("generate Runtime-O pose hash differs from prepare")
        plan_path = session_root / "selection_plan.json"
        with _PIPELINE_LOCK:
            transport_server._set_latest_progress(
                {
                    "status": "running",
                    "stage": "all_view_axis_and_raw_cache",
                    "session_id": session_id,
                    "candidate_count": len(frame_names),
                }
            )
            capture = materialize_all_view_phone_capture(
                session_id=session_id,
                data_dir=data_dir,
                mask_dir=mask_dir,
                frame_names=frame_names,
                session_root=session_root,
            )
            raw_cache_report = Path(capture["raw_cache_report"]).resolve(strict=True)
            object_key = str(capture["object_key"])
            primary_mode = PRIMARY_MODE
            transport_server._set_latest_progress(
                {
                    "status": "running",
                    "stage": f"{primary_mode}:reconstruction",
                    "session_id": session_id,
                }
            )
            primary = _run_current_branch(
                config=_CONFIG,
                session_id=session_id,
                session_root=session_root,
                raw_cache_report=raw_cache_report,
                object_key=object_key,
                mode=primary_mode,
                data_dir=data_dir,
            )

        branches: dict[str, Any] = {primary_mode: primary}
        status = {primary_mode: "complete"}
        runtime_binding = _runtime_o_binding(
            Path(primary["runtime_input_manifest"]), object_key
        )
        plan = build_selection_plan(
            frame_names,
            poses,
            runtime_binding=runtime_binding,
            client_selected=payload.get("selected"),
            count=_CONFIG.selected_view_count,
        )
        if plan_path.is_file():
            existing_plan = load_json(plan_path)
            if existing_plan.get("sha256") != plan.get("sha256"):
                raise RuntimeError(f"selection plan changed for session={session_id}")
        else:
            atomic_json(plan_path, plan)
        shared_o = verify_shared_all_view_o(branches)
        summary_path = _write_summary(
            session_id=session_id,
            selection_plan=plan,
            capture=capture,
            branches=branches,
            branch_status=status,
            shared_o=shared_o,
        )

        mobile_ar = None
        if not _CONFIG.capture_only:
            overlay = dict((primary.get("mesh") or {}).get("mobile_overlay") or {})
            mesh_report = dict(primary.get("mesh") or {})
            mesh_path = Path(mesh_report["unity_capture_anchor_a0_armesh"])
            mobile_ar = {
                "format": "yxc_unity_ar_mesh.v1",
                "mesh_url": f"/manual_reconstruction_mesh/{session_id}",
                "mesh_sha256": sha256_file(mesh_path),
                "byte_count": int(overlay["byte_count"]),
                "vertex_count": int(overlay["vertex_count"]),
                "triangle_count": int(overlay["triangle_count"]),
                "coordinate_frame": MOBILE_MESH_COORDINATE_FRAME,
                "placement": MOBILE_MESH_PLACEMENT,
                "session_id": session_id,
                "lifecycle_generation": int(
                    prepared_runtime["lifecycle_generation"]
                ),
                "runtime_o_sha256": prepared_runtime["runtime_o_sha256"],
                "requested_pose_sha256": prepared_runtime[
                    "requested_pose_sha256"
                ],
                "input_o2w_refinement": dict(
                    primary.get("input_o2w_refinement") or {}
                ),
                "capture_reference_contract": (
                    "A0 owns uploaded camera poses and display placement; the "
                    "returned Mesh is already A0-local. Attach below capture "
                    "A0 with identity local transform."
                ),
                "display_only": True,
            }
        transport_server._set_latest_progress(
            {
                "status": "complete",
                "stage": "complete",
                "session_id": session_id,
                "summary_report": str(summary_path.resolve()),
                "primary_mode": primary_mode,
                "primary_contours": primary.get("original_phone_input_contours"),
            }
        )
        # The heavy reconstruction subprocess has exited at this point.  Warm
        # the separate Hydra/SAM2 environment now, so the optional return-to-
        # scene calibration button does not pay model-import latency later.
        if _SAM2_CLIENT is not None and not _CONFIG.capture_only:
            _SAM2_CLIENT.prewarm_async()
        primary_indices = plan["branches"][primary_mode]["selected_indices"]
        return jsonify(
            {
                "status": "success",
                "message": "第一阶段训练一致球面最远8帧重建完成",
                "session_id": session_id,
                "dataset_dir": capture["axis_corrected_directory"],
                "capture_report": str(
                    (session_root / "00_all_view_capture_report.json").resolve()
                ),
                "client_selected_indices": payload.get("selected"),
                "selected_indices": primary_indices,
                "filtered_selected_indices": primary_indices,
                "frame_filtering_applied": True,
                "selection_plan": plan,
                "capture": capture,
                "reconstruction": primary,
                "reconstruction_branches": branches,
                "branch_status": status,
                "summary_report": str(summary_path.resolve()),
                "deployment_profile": "manual_phase1_ss30k_slat30k",
                "mobile_ar": mobile_ar,
            }
        ), 200
    except Exception as error:
        transport_server.legacy.logging.exception(
            "manual phase-one phone reconstruction failed"
        )
        transport_server._set_latest_progress(
            {
                "status": "failed",
                "stage": "manual_phase1",
                "session_id": transport_server.legacy.current_session_id,
                "error": repr(error),
            }
        )
        detail = str(error).strip()
        if len(detail) > 300:
            detail = detail[:297].rstrip() + "..."
        return jsonify(
            {
                "status": "error",
                "message": "第一阶段重建未完成：" + (detail or type(error).__name__),
                "error_detail": str(error),
                "session_id": transport_server.legacy.current_session_id,
                "session_data_dir": transport_server.legacy.current_data_dir,
            }
        ), 500


def manual_reconstruction_mesh(session_id: str):
    """Serve the primary branch's immutable Unity-A0-local mobile Mesh."""

    if _CONFIG is None:
        return jsonify({"status": "error", "message": "server is not configured"}), 503
    if not _SAFE_SESSION_ID.fullmatch(str(session_id)):
        return jsonify({"status": "error", "message": "invalid session id"}), 400
    primary_mode = PRIMARY_MODE
    branch_report = (
        _branch_dir(_session_root(session_id), primary_mode) / "branch_report.json"
    )
    if not branch_report.is_file():
        return jsonify(
            {"status": "error", "message": f"Mesh is unavailable for {session_id}"}
        ), 404
    branch = load_json(branch_report)
    mesh_value = (branch.get("mesh") or {}).get("unity_capture_anchor_a0_armesh")
    overlay = dict((branch.get("mesh") or {}).get("mobile_overlay") or {})
    mesh = Path(str(mesh_value)) if mesh_value else Path()
    if branch.get("passed") is not True or not mesh.is_file():
        return jsonify(
            {"status": "error", "message": f"Mesh is unavailable for {session_id}"}
        ), 404
    response = send_file(
        mesh,
        mimetype="application/octet-stream",
        as_attachment=False,
        download_name=f"{session_id}.armesh",
        conditional=True,
    )
    response.headers["X-AR-Mesh-Format"] = "yxc_unity_ar_mesh.v1"
    response.headers["X-AR-Coordinate-Frame"] = MOBILE_MESH_COORDINATE_FRAME
    response.headers["X-AR-Mesh-SHA256"] = sha256_file(mesh)
    response.headers["X-AR-Mesh-Vertices"] = str(overlay.get("vertex_count", ""))
    response.headers["X-AR-Mesh-Triangles"] = str(overlay.get("triangle_count", ""))
    response.headers["Cache-Control"] = "private, max-age=3600, immutable"
    return response


def _alignment_attempt_root(session_id: str, refinement_id: str) -> Path:
    return (
        _session_root(session_id)
        / "08_fast_a0_silhouette_refinement"
        / refinement_id
    )


def _completed_mobile_mesh(session_id: str) -> tuple[Path, dict[str, Any]]:
    branch_report = (
        _branch_dir(_session_root(session_id), PRIMARY_MODE) / "branch_report.json"
    )
    if not branch_report.is_file():
        raise FileNotFoundError(
            f"completed primary reconstruction is missing: {branch_report}"
        )
    branch = load_json(branch_report)
    mesh_value = (branch.get("mesh") or {}).get("unity_capture_anchor_a0_armesh")
    mesh = Path(str(mesh_value)).resolve() if mesh_value else Path()
    if branch.get("passed") is not True or not mesh.is_file():
        raise RuntimeError("completed A0-local mobile Mesh is unavailable")
    return mesh, branch


def _reconstruction_input_pose_targets(
    branch: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Load the exact eight Unity/A0 camera poses consumed by reconstruction.

    These targets intentionally come from the immutable original ``poses.txt``
    rows, not from a later inversion of the internal CV matrices.  The phone
    can therefore revisit the exact A0-relative acquisition poses without an
    additional axis conversion or numerical round trip.
    """

    selected_names = [
        str(value) for value in branch.get("selected_source_frame_names", [])
    ]
    if len(selected_names) != SELECTED_VIEW_COUNT or len(set(selected_names)) != len(
        selected_names
    ):
        raise RuntimeError(
            "completed reconstruction lacks eight unique selected source frames"
        )
    raw_report_path = Path(str(branch.get("raw_cache_report") or "")).resolve(
        strict=True
    )
    raw_report = load_json(raw_report_path)
    objects = list(raw_report.get("objects") or [])
    if raw_report.get("passed") is not True or len(objects) != 1:
        raise RuntimeError("completed reconstruction raw-cache binding is invalid")
    source_root = Path(str(objects[0].get("source_dataset") or "")).resolve(
        strict=True
    )
    poses_path = (source_root / "poses.txt").resolve(strict=True)
    poses = read_phone_poses(poses_path)
    targets: list[dict[str, Any]] = []
    for target_index, frame_name in enumerate(selected_names):
        pose = poses.get(frame_name)
        if pose is None:
            raise RuntimeError(
                f"selected reconstruction pose is missing from poses.txt: {frame_name}"
            )
        position = np.asarray(pose.get("pos"), dtype=np.float64)
        quaternion = np.asarray(pose.get("quat"), dtype=np.float64)
        quaternion_norm = float(np.linalg.norm(quaternion))
        if (
            position.shape != (3,)
            or quaternion.shape != (4,)
            or not np.isfinite(position).all()
            or not np.isfinite(quaternion).all()
            or quaternion_norm <= 1.0e-8
        ):
            raise RuntimeError(
                f"selected reconstruction pose is incomplete: {frame_name}"
            )
        if str(pose.get("pose_binding") or "") != PHONE_POSE_BINDING:
            raise RuntimeError(
                f"selected reconstruction pose binding differs: {frame_name}"
            )
        if str(pose.get("pose_coordinate_frame") or "") != PHONE_POSE_COORDINATE_FRAME:
            raise RuntimeError(
                f"selected reconstruction pose coordinate frame differs: {frame_name}"
            )
        if str(pose.get("capture_anchor_tracking_state") or "") != "Tracking":
            raise RuntimeError(
                f"selected reconstruction pose was not captured under Tracking A0: {frame_name}"
            )
        source_image = (source_root / frame_name).resolve(strict=True)
        quaternion = quaternion / quaternion_norm
        targets.append(
            {
                "target_index": target_index,
                "source_frame_name": frame_name,
                "source_image": str(source_image),
                "source_image_sha256": sha256_file(source_image),
                "position": position.tolist(),
                "quaternion_xyzw": quaternion.tolist(),
                "pose_binding": PHONE_POSE_BINDING,
                "pose_coordinate_frame": PHONE_POSE_COORDINATE_FRAME,
            }
        )
    return targets


def _public_mobile_pose_target(target: Mapping[str, Any]) -> dict[str, Any]:
    position = list(target["position"])
    quaternion = list(target["quaternion_xyzw"])
    return {
        "target_index": int(target["target_index"]),
        "source_frame_name": str(target["source_frame_name"]),
        "source_image_sha256": str(target["source_image_sha256"]),
        "position_x": float(position[0]),
        "position_y": float(position[1]),
        "position_z": float(position[2]),
        "quaternion_x": float(quaternion[0]),
        "quaternion_y": float(quaternion[1]),
        "quaternion_z": float(quaternion[2]),
        "quaternion_w": float(quaternion[3]),
    }


def _alignment_identity(payload: Mapping[str, Any]) -> tuple[str, int]:
    session_id = str(payload.get("session_id") or "")
    lifecycle_generation = int(payload.get("lifecycle_generation", -1))
    if not _SAFE_SESSION_ID.fullmatch(session_id):
        raise ValueError(f"invalid refinement session id: {session_id!r}")
    if lifecycle_generation < 0:
        raise ValueError("refinement request lacks lifecycle_generation")
    transport_server.legacy._load_current_session()
    if str(transport_server.legacy.current_session_id or "") != session_id:
        raise RuntimeError(
            "refinement session is no longer the active phone capture session"
        )
    prepare = load_json(_runtime_o_prepare_path(session_id))
    if int(prepare.get("lifecycle_generation", -2)) != lifecycle_generation:
        raise RuntimeError("refinement lifecycle differs from reconstructed Mesh")
    runtime_o_sha256 = str(payload.get("runtime_o_sha256") or "")
    requested_pose_sha256 = str(payload.get("requested_pose_sha256") or "")
    if runtime_o_sha256 != str(prepare.get("runtime_o_sha256") or ""):
        raise RuntimeError("refinement Runtime-O hash differs from reconstructed Mesh")
    if requested_pose_sha256 != str(prepare.get("requested_pose_sha256") or ""):
        raise RuntimeError("refinement Runtime-O pose hash differs")
    return session_id, lifecycle_generation


def collection_alignment_refine_start():
    """Open one short A0-relative calibration clip without touching geometry."""

    try:
        if _CONFIG is None or _SAM2_CLIENT is None:
            raise RuntimeError("manual phone server is not configured")
        payload = request.get_json(silent=True) or {}
        session_id, lifecycle_generation = _alignment_identity(payload)
        mobile_mesh, branch = _completed_mobile_mesh(session_id)
        current_transform = payload.get("current_mesh_transform_unity") or {}
        parent = _session_root(session_id) / "08_fast_a0_silhouette_refinement"
        parent.mkdir(parents=True, exist_ok=True)
        existing = [
            int(path.name.split("_")[-1])
            for path in parent.glob("attempt_*")
            if path.is_dir() and path.name.split("_")[-1].isdigit()
        ]
        refinement_id = f"attempt_{max(existing, default=0) + 1:03d}"
        root = _alignment_attempt_root(session_id, refinement_id)
        root.mkdir(parents=True, exist_ok=False)
        (root / "00_phone_rgb_a0").mkdir()
        state = {
            "format": ALIGNMENT_REFINEMENT_FORMAT,
            "status": "capturing",
            "created_at_utc": utc_now(),
            "session_id": session_id,
            "lifecycle_generation": lifecycle_generation,
            "refinement_id": refinement_id,
            "runtime_o_sha256": str(payload["runtime_o_sha256"]),
            "requested_pose_sha256": str(payload["requested_pose_sha256"]),
            "current_mesh_transform_unity": dict(current_transform),
            "mobile_mesh": str(mobile_mesh),
            "mobile_mesh_sha256": sha256_file(mobile_mesh),
            "candidate_frame_limit": None,
            "recommended_candidate_frames": RECOMMENDED_CAPTURE_FRAMES,
            "optimization_view_count": MAX_OPTIMIZATION_VIEWS,
            "view_selection": "current_mesh_centered_spherical_farthest",
            "branch_report_sha256": sha256_file(
                _branch_dir(_session_root(session_id), PRIMARY_MODE)
                / "branch_report.json"
            ),
            "rows": [],
            "root": str(root),
        }
        state["start_binding_sha256"] = canonical_sha256(
            {key: value for key, value in state.items() if key not in {"rows", "root"}}
        )
        atomic_json(root / "start_report.json", state)
        with _ALIGNMENT_STATE_LOCK:
            _ALIGNMENT_ATTEMPTS[session_id] = state
        _SAM2_CLIENT.prewarm_async()
        return jsonify(
            {
                "status": "capturing",
                "message": (
                    "快速校准已开始：围绕物体中心缓慢环绕；至少采集16个候选帧，"
                    "建议32帧或更多，再按按钮执行球面FPS 16帧优化"
                ),
                "session_id": session_id,
                "lifecycle_generation": lifecycle_generation,
                "refinement_id": refinement_id,
                "minimum_frames": MIN_OPTIMIZATION_VIEWS,
                "recommended_frames": RECOMMENDED_CAPTURE_FRAMES,
                "optimization_views": MAX_OPTIMIZATION_VIEWS,
                "capture_frame_limit": None,
                "view_selection": "current_mesh_centered_spherical_farthest",
                "sam2_predictor": "sam2.1_hiera_tiny_video_predictor_persistent",
                "geometry_regenerated": False,
            }
        ), 200
    except Exception as error:
        transport_server.legacy.logging.exception("alignment refinement start failed")
        return jsonify(
            {"status": "error", "message": f"无法开始快速校准：{error}"}
        ), 500


def _required_form_float(name: str) -> float:
    value = float(request.form.get(name, ""))
    if not math.isfinite(value):
        raise ValueError(f"non-finite form field: {name}")
    return value


def collection_alignment_refine_upload():
    """Store one exact camera-frame/A0-pose calibration pair."""

    try:
        session_id = str(request.form.get("session_id") or "")
        lifecycle_generation = int(request.form.get("lifecycle_generation", "-1"))
        refinement_id = str(request.form.get("refinement_id") or "")
        with _ALIGNMENT_STATE_LOCK:
            state = _ALIGNMENT_ATTEMPTS.get(session_id)
            if state is None:
                raise RuntimeError("no active refinement capture for this session")
            if state.get("status") != "capturing":
                raise RuntimeError("refinement capture is not accepting frames")
            if int(state["lifecycle_generation"]) != lifecycle_generation:
                raise RuntimeError("refinement upload lifecycle differs")
            if str(state["refinement_id"]) != refinement_id:
                raise RuntimeError("refinement upload attempt differs")
            if request.form.get("pose_binding") != ALIGNMENT_POSE_BINDING:
                raise RuntimeError("refinement camera pose is not A0-frame-bound")
            if request.form.get("pose_coordinate_frame") != ALIGNMENT_POSE_COORDINATE_FRAME:
                raise RuntimeError("refinement camera coordinate frame is not A0")
            if request.form.get("capture_anchor_tracking_state") != "Tracking":
                raise RuntimeError("A0 must be Tracking at calibration-frame capture")
            image_file = request.files.get("image")
            if image_file is None:
                raise ValueError("refinement upload lacks image")
            image_bytes = image_file.read()
            if not image_bytes or len(image_bytes) > 20 * 1024 * 1024:
                raise ValueError("refinement image byte count is invalid")
            with Image.open(io.BytesIO(image_bytes)) as decoded:
                rgb = decoded.convert("RGB")
                width, height = rgb.size
                index = len(state["rows"])
                image_path = (
                    Path(state["root"])
                    / "00_phone_rgb_a0"
                    / f"frame_{index:04d}.jpg"
                )
                temporary = image_path.with_name(f".{image_path.name}.tmp.jpg")
                rgb.save(temporary, quality=90)
                os.replace(temporary, image_path)
            row = {
                "frame_index": index,
                "frame_name": image_path.name,
                "image": str(image_path.resolve()),
                "image_sha256": sha256_file(image_path),
                "image_width": int(width),
                "image_height": int(height),
                "pos_x": _required_form_float("pos_x"),
                "pos_y": _required_form_float("pos_y"),
                "pos_z": _required_form_float("pos_z"),
                "quat_x": _required_form_float("quat_x"),
                "quat_y": _required_form_float("quat_y"),
                "quat_z": _required_form_float("quat_z"),
                "quat_w": _required_form_float("quat_w"),
                "fx": _required_form_float("fx"),
                "fy": _required_form_float("fy"),
                "cx": _required_form_float("cx"),
                "cy": _required_form_float("cy"),
                "intrinsic_width": int(request.form.get("intrinsic_width", width)),
                "intrinsic_height": int(request.form.get("intrinsic_height", height)),
                "cpu_image_timestamp_s": _required_form_float(
                    "cpu_image_timestamp_s"
                ),
                "camera_frame_timestamp_ns": int(
                    request.form.get("camera_frame_timestamp_ns", "-1")
                ),
                "pose_sample_realtime_s": _required_form_float(
                    "pose_sample_realtime_s"
                ),
                "pose_binding": ALIGNMENT_POSE_BINDING,
                "pose_coordinate_frame": ALIGNMENT_POSE_COORDINATE_FRAME,
                "capture_anchor_tracking_state": "Tracking",
                "screen_orientation": str(
                    request.form.get("screen_orientation") or "unknown"
                ),
                "image_transform": str(
                    request.form.get("image_transform") or "None"
                ),
            }
            state["rows"].append(row)
            atomic_json(Path(state["root"]) / "capture_state.json", state)
            count = len(state["rows"])
        return jsonify(
            {
                "status": "success",
                "session_id": session_id,
                "refinement_id": refinement_id,
                "captured_frames": count,
            }
        ), 200
    except Exception as error:
        transport_server.legacy.logging.exception("alignment refinement upload failed")
        return jsonify(
            {"status": "error", "message": f"快速校准帧上传失败：{error}"}
        ), 500


def collection_alignment_refine_optimize():
    """Run SAM2 Tiny observation and bounded Mesh-pose optimization."""

    session_id_for_failure = ""
    try:
        if _SAM2_CLIENT is None:
            raise RuntimeError("SAM2 Tiny worker client is unavailable")
        payload = request.get_json(silent=True) or {}
        session_id, lifecycle_generation = _alignment_identity(payload)
        session_id_for_failure = session_id
        refinement_id = str(payload.get("refinement_id") or "")
        with _ALIGNMENT_STATE_LOCK:
            state = _ALIGNMENT_ATTEMPTS.get(session_id)
            if state is None:
                raise RuntimeError("no active refinement attempt")
            if state["refinement_id"] != refinement_id:
                raise RuntimeError("refinement optimize attempt differs")
            if int(state["lifecycle_generation"]) != lifecycle_generation:
                raise RuntimeError("refinement optimize lifecycle differs")
            if state["status"] != "capturing":
                raise RuntimeError(f"refinement attempt is {state['status']!r}")
            if len(state["rows"]) < MIN_OPTIMIZATION_VIEWS:
                raise ValueError(
                    f"快速校准至少需要{MIN_OPTIMIZATION_VIEWS}帧，当前只有"
                    f"{len(state['rows'])}帧"
                )
            state["status"] = "optimizing"
            rows = [dict(row) for row in state["rows"]]
            atomic_json(Path(state["root"]) / "capture_state.json", state)

        mobile_mesh, _branch = _completed_mobile_mesh(session_id)
        with _ALIGNMENT_LOCK:
            report = run_refinement(
                mobile_mesh=mobile_mesh,
                frame_rows=rows,
                output_dir=Path(state["root"]) / "04_alignment_optimization",
                sam2_client=_SAM2_CLIENT,
                current_unity_transform=state.get("current_mesh_transform_unity"),
                iterations=50,
            )
        with _ALIGNMENT_STATE_LOCK:
            state["status"] = "complete"
            state["accepted"] = bool(report["accepted"])
            state["report"] = str(
                Path(state["root"]) / "04_alignment_optimization" / "report.json"
            )
            atomic_json(Path(state["root"]) / "capture_state.json", state)
        accepted = bool(report["accepted"])
        return jsonify(
            {
                "status": "success",
                "message": (
                    "快速轮廓校准已通过并更新 Mesh 位姿"
                    if accepted
                    else "快速轮廓校准未通过安全门，已保留原 Mesh 位姿"
                ),
                "session_id": session_id,
                "lifecycle_generation": lifecycle_generation,
                "refinement_id": refinement_id,
                "accepted": accepted,
                "geometry_regenerated": False,
                "selected_mesh_transform_unity": report[
                    "selected_transform_unity"
                ],
                "initial_iou_mean": report["optimization"]["initial_iou_mean"],
                "optimized_iou_mean": report["optimization"][
                    "optimized_iou_mean"
                ],
                "iou_gain_mean": report["optimization"]["iou_gain_mean"],
                "checks": report["optimization"]["checks"],
                "report": state["report"],
            }
        ), 200
    except Exception as error:
        if session_id_for_failure:
            with _ALIGNMENT_STATE_LOCK:
                failed_state = _ALIGNMENT_ATTEMPTS.get(session_id_for_failure)
                if failed_state is not None:
                    failed_state["status"] = "failed"
                    failed_state["error"] = repr(error)
                    atomic_json(
                        Path(failed_state["root"]) / "capture_state.json",
                        failed_state,
                    )
        transport_server.legacy.logging.exception("alignment refinement optimize failed")
        return jsonify(
            {
                "status": "error",
                "message": f"快速校准失败，原 Mesh 位姿保持不变：{error}",
            }
        ), 500


def _mobile_overlay_audit_root(session_id: str, audit_id: str) -> Path:
    return _session_root(session_id) / "09_mobile_render_overlay_audit" / audit_id


def _bounded_mobile_form_string(name: str, *, maximum: int = 4096) -> str:
    value = str(request.form.get(name) or "")
    if len(value) > int(maximum):
        raise ValueError(f"mobile diagnostic field is oversized: {name}")
    return value


def _mobile_pose_delta(
    first_position: Sequence[float],
    first_quaternion: Sequence[float],
    second_position: Sequence[float],
    second_quaternion: Sequence[float],
) -> dict[str, float]:
    first_p = np.asarray(first_position, dtype=np.float64)
    second_p = np.asarray(second_position, dtype=np.float64)
    first_q = np.asarray(first_quaternion, dtype=np.float64)
    second_q = np.asarray(second_quaternion, dtype=np.float64)
    if first_p.shape != (3,) or second_p.shape != (3,):
        raise ValueError("mobile diagnostic positions must be [3]")
    first_norm = float(np.linalg.norm(first_q))
    second_norm = float(np.linalg.norm(second_q))
    if first_q.shape != (4,) or second_q.shape != (4,) or min(first_norm, second_norm) <= 1.0e-8:
        raise ValueError("mobile diagnostic quaternions must be nonzero xyzw")
    cosine = abs(float((first_q / first_norm) @ (second_q / second_norm)))
    angle = math.degrees(2.0 * math.acos(float(np.clip(cosine, -1.0, 1.0))))
    return {
        "translation_meters": float(np.linalg.norm(first_p - second_p)),
        "rotation_degrees": angle,
    }


def _render_mobile_pose_diagnostic(
    *,
    mobile_mesh: Path,
    raw_camera_path: Path,
    camera_metadata: Mapping[str, Any],
    mesh_transform_unity: Mapping[str, Any],
    destination: Path,
) -> dict[str, Any]:
    """Reproject the returned phone Mesh in the native XRCpuImage domain.

    This deliberately reuses the same XRCpuImage ``None`` + explicit x/y
    transpose contract as reconstruction and fast alignment.  The result is
    transposed back into the raw sensor orientation before it is saved.  A
    separate display-matrix warp below converts it to the phone screenshot
    orientation; keeping both artifacts prevents a display transform from
    being mistaken for a camera-extrinsic transform.
    """

    with Image.open(raw_camera_path) as decoded:
        raw = decoded.convert("RGB")
    raw_size = raw.size
    corrected = raw.transpose(Image.Transpose.TRANSPOSE)
    intrinsic = corrected_intrinsic(camera_metadata, raw_size)
    T_A0_2C = internal_camera_from_unity_metadata(camera_metadata)
    vertices, faces = mobile_mesh_internal_buffers(mobile_mesh)
    transform = unity_similarity_to_internal(mesh_transform_unity)
    transformed = (
        vertices @ transform[:3, :3].T + transform[:3, 3][None]
    ).astype(np.float32)
    raster_context = make_headless_raster_context()
    try:
        projected = rasterize_world_silhouette(
            transformed,
            faces,
            T_A0_2C,
            intrinsic,
            {"model": "PINHOLE", "distortion": []},
            (corrected.height, corrected.width),
            raster_context,
        )
    finally:
        del raster_context
    if int(projected.sum()) <= 0:
        raise RuntimeError("server same-pose reprojection produced an empty silhouette")
    contour = boundary(projected, width=3)
    corrected_array = np.asarray(corrected, dtype=np.uint8).copy()
    corrected_array[contour] = np.asarray([0, 255, 255], dtype=np.uint8)
    overlay = Image.fromarray(corrected_array, mode="RGB").transpose(
        Image.Transpose.TRANSPOSE
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp.png")
    overlay.save(temporary)
    os.replace(temporary, destination)
    return {
        "passed": True,
        "projection_contract": (
            "returned_A0_local_armesh -> uploaded_Unity_local_similarity -> "
            "uploaded_A0_relative_camera -> corrected_K -> raw_phone_orientation"
        ),
        "pixel_axis_contract": AXIS_CONTRACT,
        "silhouette_pixels": int(projected.sum()),
        "contour_pixels": int(contour.sum()),
        "raw_camera_size": [int(raw_size[0]), int(raw_size[1])],
        "corrected_camera_size": [int(corrected.width), int(corrected.height)],
        "K_corrected": intrinsic.tolist(),
        "K_corrected_sha256": array_sha256(intrinsic),
        "T_A0_2C_internal": T_A0_2C.tolist(),
        "T_A0_2C_internal_sha256": array_sha256(T_A0_2C),
        "mesh_similarity_internal": transform.tolist(),
        "mesh_similarity_internal_sha256": array_sha256(transform),
        "overlay": str(destination.resolve()),
        "overlay_sha256": sha256_file(destination),
    }


def _parse_mobile_display_matrix(value: str) -> np.ndarray:
    fields = str(value).split()
    if len(fields) != 16:
        raise ValueError("mobile display matrix must contain 16 finite values")
    matrix = np.asarray([float(field) for field in fields], dtype=np.float64).reshape(
        4, 4
    )
    if not np.isfinite(matrix).all():
        raise ValueError("mobile display matrix is not finite")
    affine = matrix[:2, :3]
    corners = np.asarray(
        [
            affine @ [0.0, 0.0, 1.0],
            affine @ [1.0, 0.0, 1.0],
            affine @ [0.0, 1.0, 1.0],
            affine @ [1.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )
    if (
        abs(float(np.linalg.det(affine[:, :2]))) <= 1.0e-9
        or float(corners.min()) < -0.25
        or float(corners.max()) > 1.25
    ):
        raise ValueError("mobile display matrix has an invalid UV mapping")
    return matrix


def _write_display_aligned_mobile_image(
    source: Path,
    destination: Path,
    *,
    display_matrix: str,
    display_size: tuple[int, int],
) -> dict[str, Any]:
    """Warp a raw XRCpuImage into the exact portrait Unity display UV domain.

    ARFoundation's display matrix maps normalized display UV to normalized
    camera-image UV.  Pillow's affine transform also consumes an output-to-input
    mapping, so the first two matrix rows can be used directly.  Both images
    are stored with a top-left origin here; no additional mirror or vertical
    flip is applied.  This is the convention verified by the uploaded
    end-of-frame screenshot and avoids rotating an extrinsic to fix pixels.
    """

    matrix = _parse_mobile_display_matrix(display_matrix)
    width, height = (int(display_size[0]), int(display_size[1]))
    if width <= 0 or height <= 0 or width * height > 20_000_000:
        raise ValueError(f"invalid mobile display size: {(width, height)}")
    with Image.open(source) as decoded:
        image = decoded.convert("RGB")
    source_width, source_height = image.size
    # Pillow evaluates the inverse affine at output pixel centres.  Convert
    # between pixel-centre and normalized UV coordinates explicitly; using
    # ``size - 1`` here creates a one-pixel black strip after 90-degree turns.
    coefficients = (
        float(matrix[0, 0]) * source_width / width,
        float(matrix[0, 1]) * source_width / height,
        (
            float(matrix[0, 0]) * 0.5 / width
            + float(matrix[0, 1]) * 0.5 / height
            + float(matrix[0, 2])
        )
        * source_width,
        float(matrix[1, 0]) * source_height / width,
        float(matrix[1, 1]) * source_height / height,
        (
            float(matrix[1, 0]) * 0.5 / width
            + float(matrix[1, 1]) * 0.5 / height
            + float(matrix[1, 2])
        )
        * source_height,
    )
    aligned = image.transform(
        (width, height),
        Image.Transform.AFFINE,
        coefficients,
        resample=Image.Resampling.BICUBIC,
        fillcolor=(0, 0, 0),
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp.png")
    aligned.save(temporary, compress_level=2)
    os.replace(temporary, destination)
    return {
        "passed": True,
        "contract": "ARFoundation displayUV -> rawCameraUV; top-left image origin",
        "display_matrix": matrix.tolist(),
        "display_matrix_sha256": array_sha256(matrix),
        "raw_size": [source_width, source_height],
        "display_size": [width, height],
        "pillow_output_to_input_affine": list(coefficients),
        "overlay": str(destination.resolve()),
        "overlay_sha256": sha256_file(destination),
    }


def _write_failed_mobile_pose_diagnostic(
    raw_camera_path: Path, destination: Path, error: Exception
) -> dict[str, Any]:
    with Image.open(raw_camera_path) as decoded:
        image = decoded.convert("RGB")
    draw = ImageDraw.Draw(image)
    message = f"SERVER REPROJECTION FAILED: {type(error).__name__}: {error}"
    draw.rectangle((0, 0, image.width, min(72, image.height)), fill=(110, 0, 0))
    draw.text((10, 10), message[:180], fill=(255, 255, 255))
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp.png")
    image.save(temporary)
    os.replace(temporary, destination)
    return {
        "passed": False,
        "error_type": type(error).__name__,
        "error": str(error),
        "overlay": str(destination.resolve()),
        "overlay_sha256": sha256_file(destination),
    }


def _write_mobile_pose_pair(
    phone_screen: Path, server_overlay: Path, destination: Path
) -> None:
    panel_width, panel_height, header = 900, 1600, 48
    canvas = Image.new(
        "RGB", (panel_width * 2, panel_height + header), (18, 18, 18)
    )
    draw = ImageDraw.Draw(canvas)
    for index, (path, label) in enumerate(
        (
            (phone_screen, "Unity end-of-frame screen"),
            (server_overlay, "Server same-pose Mesh reprojection"),
        )
    ):
        with Image.open(path) as decoded:
            image = decoded.convert("RGB")
        image.thumbnail(
            (panel_width - 12, panel_height - 12), Image.Resampling.LANCZOS
        )
        x = index * panel_width + (panel_width - image.width) // 2
        y = header + (panel_height - image.height) // 2
        canvas.paste(image, (x, y))
        draw.text((index * panel_width + 10, 9), label, fill=(240, 240, 240))
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp.jpg")
    canvas.save(temporary, quality=96, subsampling=0)
    os.replace(temporary, destination)


def _write_raw_target_pair(
    target_image: Path, live_image: Path, destination: Path
) -> None:
    panel_width, panel_height, header = 720, 540, 42
    canvas = Image.new(
        "RGB", (panel_width * 2, panel_height + header), (18, 18, 18)
    )
    draw = ImageDraw.Draw(canvas)
    for index, (path, label) in enumerate(
        (
            (target_image, "Original reconstruction input"),
            (live_image, "Live strict-pose revisit"),
        )
    ):
        with Image.open(path) as decoded:
            image = decoded.convert("RGB")
        image.thumbnail(
            (panel_width - 12, panel_height - 12), Image.Resampling.LANCZOS
        )
        x = index * panel_width + (panel_width - image.width) // 2
        y = header + (panel_height - image.height) // 2
        canvas.paste(image, (x, y))
        draw.text((index * panel_width + 10, 12), label, fill=(240, 240, 240))
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp.jpg")
    canvas.save(temporary, quality=96, subsampling=0)
    os.replace(temporary, destination)


def _mobile_overlay_audit_report(state: Mapping[str, Any]) -> dict[str, Any]:
    rows = [dict(row) for row in state.get("rows", [])]
    targets = [dict(row) for row in state.get("pose_targets", [])]
    captured_target_indices = sorted(
        int(row["matched_reconstruction_target"]["target_index"])
        for row in rows
        if row.get("matched_reconstruction_target")
    )
    server_reprojection_failures = [
        {
            "frame_index": int(row["frame_index"]),
            "error": row.get("server_same_pose_reprojection"),
        }
        for row in rows
        if (row.get("server_same_pose_reprojection") or {}).get("passed") is not True
    ]
    return {
        "format": MOBILE_OVERLAY_AUDIT_FORMAT,
        "created_at_utc": state["created_at_utc"],
        "updated_at_utc": utc_now(),
        "passed": True,
        "complete": state.get("status") == "complete",
        "status": state.get("status"),
        "session_id": state["session_id"],
        "lifecycle_generation": int(state["lifecycle_generation"]),
        "audit_id": state["audit_id"],
        "runtime_o_sha256": state["runtime_o_sha256"],
        "requested_pose_sha256": state["requested_pose_sha256"],
        "mobile_mesh": state["mobile_mesh"],
        "mobile_mesh_sha256": state["mobile_mesh_sha256"],
        "overlay_contract": MOBILE_OVERLAY_AUDIT_CONTRACT,
        "diagnostic_stage": state.get("diagnostic_stage"),
        "alignment_refinement_state": state.get("alignment_refinement_state"),
        "last_alignment_refinement_accepted": state.get(
            "last_alignment_refinement_accepted"
        ),
        "last_alignment_refinement_report": state.get(
            "last_alignment_refinement_report"
        ),
        "starting_mesh_transform_unity": state.get(
            "starting_mesh_transform_unity"
        ),
        "reconstruction_binding": state.get("reconstruction_binding"),
        "strict_reconstruction_input_pose_matching": True,
        "target_translation_tolerance_meters": float(
            state["target_translation_tolerance_meters"]
        ),
        "target_rotation_tolerance_degrees": float(
            state["target_rotation_tolerance_degrees"]
        ),
        "reconstruction_input_pose_targets": [
            _public_mobile_pose_target(target) for target in targets
        ],
        "captured_target_indices": captured_target_indices,
        "all_reconstruction_input_pose_targets_captured": (
            captured_target_indices == list(range(len(targets)))
        ),
        "maximum_frame_count": int(state["maximum_frame_count"]),
        "captured_frame_count": len(rows),
        "all_server_same_pose_reprojections_passed": (
            len(rows) > 0 and not server_reprojection_failures
        ),
        "server_same_pose_reprojection_failures": server_reprojection_failures,
        "rows": rows,
        "contact_sheet": state.get("contact_sheet"),
        "server_reprojection_contact_sheet": state.get(
            "server_reprojection_contact_sheet"
        ),
        "phone_vs_server_contact_sheet": state.get(
            "phone_vs_server_contact_sheet"
        ),
        "live_vs_reconstruction_input_contact_sheet": state.get(
            "live_vs_reconstruction_input_contact_sheet"
        ),
        "scope_guard": (
            "Explicit, bounded phone-pose diagnostic recording. It stores the "
            "native-resolution Unity end-of-frame composite and raw XRCpuImage. "
            "Every live frame is accepted only when both its raw-camera and "
            "screen-camera A0-relative poses match one unique reconstruction "
            "input pose within the recorded strict tolerances. The raw server "
            "reprojection is additionally warped with the uploaded ARFoundation "
            "display matrix before comparison, without changing extrinsics. "
            "Every artifact is excluded from "
            "reconstruction, frame selection, SAM2, fast alignment "
            "optimization, and every scientific metric."
        ),
    }


def _write_mobile_overlay_audit_report(state: Mapping[str, Any]) -> Path:
    root = Path(str(state["root"]))
    report_path = root / "report.json"
    atomic_json(report_path, _mobile_overlay_audit_report(state))
    return report_path


def collection_mobile_overlay_audit_start():
    """Open one bounded, diagnostic-only phone-render readback attempt."""

    try:
        if _CONFIG is None:
            raise RuntimeError("manual phone server is not configured")
        payload = request.get_json(silent=True) or {}
        session_id, lifecycle_generation = _alignment_identity(payload)
        mobile_mesh, branch = _completed_mobile_mesh(session_id)
        starting_mesh_transform = dict(
            payload.get("current_mesh_transform_unity") or {}
        )
        # Parse once at attempt creation so an invalid phone transform cannot
        # create a diagnostic directory that later looks authoritative.
        unity_similarity_to_internal(starting_mesh_transform)
        diagnostic_stage = str(
            payload.get("diagnostic_stage") or "manual_pose_diagnostic"
        )
        if len(diagnostic_stage) > 96:
            raise ValueError("mobile diagnostic stage label is oversized")
        alignment_state = str(
            payload.get("alignment_refinement_state") or "unknown"
        )
        if len(alignment_state) > 64:
            raise ValueError("mobile diagnostic alignment state is oversized")
        last_alignment_report = str(
            payload.get("last_alignment_refinement_report") or ""
        )
        if len(last_alignment_report) > 4096:
            raise ValueError("mobile diagnostic alignment report is oversized")
        maximum_frames = max(
            1,
            min(
                MOBILE_OVERLAY_AUDIT_MAX_FRAMES,
                int(payload.get("maximum_frames", MOBILE_OVERLAY_AUDIT_MAX_FRAMES)),
            ),
        )
        pose_targets = _reconstruction_input_pose_targets(branch)
        if payload.get("strict_reconstruction_input_pose_matching") is not True:
            raise RuntimeError(
                "phone diagnostic must explicitly request strict reconstruction-input pose matching"
            )
        if maximum_frames != len(pose_targets):
            raise RuntimeError(
                "mobile diagnostic must cover every reconstruction input pose exactly once"
            )
        translation_tolerance = float(
            payload.get(
                "target_translation_tolerance_meters",
                MOBILE_OVERLAY_TARGET_TRANSLATION_TOLERANCE_METERS,
            )
        )
        rotation_tolerance = float(
            payload.get(
                "target_rotation_tolerance_degrees",
                MOBILE_OVERLAY_TARGET_ROTATION_TOLERANCE_DEGREES,
            )
        )
        if (
            not math.isfinite(translation_tolerance)
            or not 0.005 <= translation_tolerance <= 0.10
        ):
            raise ValueError("target translation tolerance must be in [0.005, 0.10] m")
        if (
            not math.isfinite(rotation_tolerance)
            or not 0.5 <= rotation_tolerance <= 15.0
        ):
            raise ValueError("target rotation tolerance must be in [0.5, 15] degrees")
        parent = _session_root(session_id) / "09_mobile_render_overlay_audit"
        parent.mkdir(parents=True, exist_ok=True)
        existing = [
            int(path.name.split("_")[-1])
            for path in parent.glob("attempt_*")
            if path.is_dir() and path.name.split("_")[-1].isdigit()
        ]
        audit_id = f"attempt_{max(existing, default=0) + 1:03d}"
        root = _mobile_overlay_audit_root(session_id, audit_id)
        root.mkdir(parents=True, exist_ok=False)
        for name in (
            "00_mobile_screen_composite",
            "01_mobile_outline_texture",
            "02_frame_metadata",
            "03_raw_camera_rgb",
            "04_server_raw_sensor_reprojection",
            "05_server_display_aligned_reprojection",
            "06_original_reconstruction_input",
            "07_live_vs_reconstruction_input",
            "08_phone_vs_server_display_comparison",
        ):
            (root / name).mkdir()
        branch_report = (
            _branch_dir(_session_root(session_id), PRIMARY_MODE)
            / "branch_report.json"
        ).resolve(strict=True)
        mesh_binding = dict(branch.get("mesh") or {})
        reconstruction_binding = {
            "branch_report": str(branch_report),
            "branch_report_sha256": sha256_file(branch_report),
            "T_O2A0": mesh_binding.get("T_O2A0"),
            "T_O2A0_sha256": mesh_binding.get("T_O2A0_sha256"),
            "object_pose_unity_a0": mesh_binding.get("object_pose_unity_a0"),
            "projection_world_contract": mesh_binding.get(
                "projection_world_contract"
            ),
            "unity_capture_anchor_contract": mesh_binding.get(
                "unity_capture_anchor_contract"
            ),
        }
        state = {
            "format": MOBILE_OVERLAY_AUDIT_FORMAT,
            "status": "capturing",
            "created_at_utc": utc_now(),
            "session_id": session_id,
            "lifecycle_generation": lifecycle_generation,
            "audit_id": audit_id,
            "runtime_o_sha256": str(payload["runtime_o_sha256"]),
            "requested_pose_sha256": str(payload["requested_pose_sha256"]),
            "mobile_mesh": str(mobile_mesh),
            "mobile_mesh_sha256": sha256_file(mobile_mesh),
            "diagnostic_stage": diagnostic_stage,
            "alignment_refinement_state": alignment_state,
            "last_alignment_refinement_accepted": bool(
                payload.get("last_alignment_refinement_accepted", False)
            ),
            "last_alignment_refinement_report": last_alignment_report,
            "starting_mesh_transform_unity": starting_mesh_transform,
            "reconstruction_binding": reconstruction_binding,
            "strict_reconstruction_input_pose_matching": True,
            "target_translation_tolerance_meters": translation_tolerance,
            "target_rotation_tolerance_degrees": rotation_tolerance,
            "pose_targets": pose_targets,
            "maximum_frame_count": len(pose_targets),
            "rows": [],
            "root": str(root),
        }
        atomic_json(root / "attempt_start_state.json", state)
        with _MOBILE_OVERLAY_AUDIT_LOCK:
            _MOBILE_OVERLAY_AUDITS[session_id] = state
            report_path = _write_mobile_overlay_audit_report(state)
        return jsonify(
            {
                "status": "capturing",
                "message": (
                    f"位姿诊断录制已开始（{diagnostic_stage}）；"
                    "请按提示逐一回到重建所用的八个相机位姿"
                ),
                "session_id": session_id,
                "lifecycle_generation": lifecycle_generation,
                "audit_id": audit_id,
                "maximum_frames": len(pose_targets),
                "strict_reconstruction_input_pose_matching": True,
                "target_translation_tolerance_meters": translation_tolerance,
                "target_rotation_tolerance_degrees": rotation_tolerance,
                "pose_targets": [
                    _public_mobile_pose_target(target) for target in pose_targets
                ],
                "report": str(report_path),
                "diagnostic_only": True,
            }
        ), 200
    except Exception as error:
        transport_server.legacy.logging.exception(
            "mobile render overlay audit start failed"
        )
        return jsonify(
            {"status": "error", "message": f"无法开始手机渲染回传：{error}"}
        ), 500


def _mobile_overlay_form_vector(prefix: str, components: Sequence[str]) -> list[float]:
    return [_required_form_float(f"{prefix}_{component}") for component in components]


def collection_mobile_overlay_audit_upload():
    """Persist one complete phone-pose diagnostic frame and reproject it."""

    try:
        session_id = str(request.form.get("session_id") or "")
        lifecycle_generation = int(request.form.get("lifecycle_generation", "-1"))
        audit_id = str(request.form.get("audit_id") or "")
        target_index = int(request.form.get("target_index", "-1"))
        target_source_frame_name = str(
            request.form.get("target_source_frame_name") or ""
        )
        if request.form.get("overlay_contract") != MOBILE_OVERLAY_AUDIT_CONTRACT:
            raise RuntimeError("phone overlay audit contract differs")
        if request.form.get("capture_anchor_tracking_state") != "Tracking":
            raise RuntimeError("A0 must be Tracking at phone-render readback")
        composite_file = request.files.get("composite")
        if composite_file is None:
            raise ValueError("mobile overlay upload lacks screen composite")
        if request.form.get("screen_capture_encoding") != "png_lossless_native":
            raise RuntimeError(
                "mobile overlay screen must use native-resolution lossless PNG"
            )
        composite_bytes = composite_file.read()
        if not composite_bytes or len(composite_bytes) > 24 * 1024 * 1024:
            raise ValueError("mobile screen-composite byte count is invalid")
        outline_file = request.files.get("outline")
        outline_bytes = outline_file.read() if outline_file is not None else b""
        if len(outline_bytes) > 8 * 1024 * 1024:
            raise ValueError("mobile outline-texture byte count is invalid")
        raw_camera_file = request.files.get("raw_camera")
        if raw_camera_file is None:
            raise ValueError("mobile pose diagnostic lacks raw XRCpuImage")
        raw_camera_bytes = raw_camera_file.read()
        if not raw_camera_bytes or len(raw_camera_bytes) > 20 * 1024 * 1024:
            raise ValueError("mobile raw-camera byte count is invalid")

        with _MOBILE_OVERLAY_AUDIT_LOCK:
            state = _MOBILE_OVERLAY_AUDITS.get(session_id)
            if state is None:
                raise RuntimeError("no active mobile overlay audit for this session")
            if state.get("status") != "capturing":
                raise RuntimeError("mobile overlay audit is not accepting frames")
            if int(state["lifecycle_generation"]) != lifecycle_generation:
                raise RuntimeError("mobile overlay lifecycle differs")
            if str(state["audit_id"]) != audit_id:
                raise RuntimeError("mobile overlay audit attempt differs")
            index = len(state["rows"])
            maximum = int(state["maximum_frame_count"])
            if index >= maximum:
                raise RuntimeError("mobile overlay audit reached its frame limit")
            pose_targets = list(state.get("pose_targets") or [])
            if not 0 <= target_index < len(pose_targets):
                raise RuntimeError("mobile diagnostic target index is invalid")
            target = dict(pose_targets[target_index])
            if target_source_frame_name != str(target["source_frame_name"]):
                raise RuntimeError("mobile diagnostic target frame binding differs")
            captured_targets = {
                int(row["matched_reconstruction_target"]["target_index"])
                for row in state["rows"]
                if row.get("matched_reconstruction_target")
            }
            if target_index in captured_targets:
                raise RuntimeError("reconstruction input pose target was already captured")
            root = Path(str(state["root"]))

            composite_path = (
                root / "00_mobile_screen_composite" / f"frame_{index:02d}.png"
            )
            with Image.open(io.BytesIO(composite_bytes)) as decoded:
                decoded.load()
                if decoded.format != "PNG":
                    raise RuntimeError("mobile screen composite is not a PNG")
                composite_width, composite_height = decoded.size
            reported_screen_size = (
                int(request.form.get("screen_width", "0")),
                int(request.form.get("screen_height", "0")),
            )
            if (composite_width, composite_height) != reported_screen_size:
                raise RuntimeError(
                    "mobile screen composite was resized before upload: "
                    f"image={(composite_width, composite_height)} "
                    f"native={reported_screen_size}"
                )
            temporary = composite_path.with_name(f".{composite_path.name}.tmp.png")
            temporary.write_bytes(composite_bytes)
            os.replace(temporary, composite_path)

            outline_path: Path | None = None
            outline_width = 0
            outline_height = 0
            if outline_bytes:
                outline_path = (
                    root / "01_mobile_outline_texture" / f"frame_{index:02d}.png"
                )
                with Image.open(io.BytesIO(outline_bytes)) as decoded:
                    outline = decoded.convert("RGBA")
                    outline_width, outline_height = outline.size
                    temporary = outline_path.with_name(
                        f".{outline_path.name}.tmp.png"
                    )
                    outline.save(temporary)
                    os.replace(temporary, outline_path)

            raw_camera_path = (
                root / "03_raw_camera_rgb" / f"frame_{index:02d}.jpg"
            )
            with Image.open(io.BytesIO(raw_camera_bytes)) as decoded:
                decoded.load()
                raw_camera_width, raw_camera_height = decoded.size
            temporary = raw_camera_path.with_name(
                f".{raw_camera_path.name}.tmp.jpg"
            )
            temporary.write_bytes(raw_camera_bytes)
            os.replace(temporary, raw_camera_path)

            screen_camera_position = _mobile_overlay_form_vector(
                "camera_pos", ("x", "y", "z")
            )
            screen_camera_quaternion = _mobile_overlay_form_vector(
                "camera_quat", ("x", "y", "z", "w")
            )
            raw_camera_position = _mobile_overlay_form_vector(
                "raw_camera_pos", ("x", "y", "z")
            )
            raw_camera_quaternion = _mobile_overlay_form_vector(
                "raw_camera_quat", ("x", "y", "z", "w")
            )
            mesh_position = _mobile_overlay_form_vector(
                "mesh_pos", ("x", "y", "z")
            )
            mesh_quaternion = _mobile_overlay_form_vector(
                "mesh_quat", ("x", "y", "z", "w")
            )
            mesh_scale = _required_form_float("mesh_uniform_scale")
            if min(
                np.linalg.norm(screen_camera_quaternion),
                np.linalg.norm(raw_camera_quaternion),
            ) <= 1.0e-6:
                raise ValueError("mobile diagnostic camera quaternion has zero norm")
            target_position = list(target["position"])
            target_quaternion = list(target["quaternion_xyzw"])
            raw_target_pose_delta = _mobile_pose_delta(
                target_position,
                target_quaternion,
                raw_camera_position,
                raw_camera_quaternion,
            )
            screen_target_pose_delta = _mobile_pose_delta(
                target_position,
                target_quaternion,
                screen_camera_position,
                screen_camera_quaternion,
            )
            translation_tolerance = float(
                state["target_translation_tolerance_meters"]
            )
            rotation_tolerance = float(state["target_rotation_tolerance_degrees"])
            for domain, delta in (
                ("raw camera", raw_target_pose_delta),
                ("screen camera", screen_target_pose_delta),
            ):
                if (
                    float(delta["translation_meters"]) > translation_tolerance
                    or float(delta["rotation_degrees"]) > rotation_tolerance
                ):
                    raise RuntimeError(
                        f"{domain} does not strictly match reconstruction target "
                        f"{target_index}: translation={delta['translation_meters']:.6f} "
                        f"> {translation_tolerance:.6f} m or "
                        f"rotation={delta['rotation_degrees']:.6f} "
                        f"> {rotation_tolerance:.6f} deg"
                    )
            if np.linalg.norm(mesh_quaternion) <= 1.0e-6 or mesh_scale <= 0.0:
                raise ValueError("mobile audit Mesh transform is invalid")
            mesh_transform = {
                "position_x": mesh_position[0],
                "position_y": mesh_position[1],
                "position_z": mesh_position[2],
                "quaternion_x": mesh_quaternion[0],
                "quaternion_y": mesh_quaternion[1],
                "quaternion_z": mesh_quaternion[2],
                "quaternion_w": mesh_quaternion[3],
                "uniform_scale": mesh_scale,
            }
            unity_similarity_to_internal(mesh_transform)
            raw_camera_metadata = {
                "pos_x": raw_camera_position[0],
                "pos_y": raw_camera_position[1],
                "pos_z": raw_camera_position[2],
                "quat_x": raw_camera_quaternion[0],
                "quat_y": raw_camera_quaternion[1],
                "quat_z": raw_camera_quaternion[2],
                "quat_w": raw_camera_quaternion[3],
                "fx": _required_form_float("fx"),
                "fy": _required_form_float("fy"),
                "cx": _required_form_float("cx"),
                "cy": _required_form_float("cy"),
                "intrinsic_width": int(request.form.get("intrinsic_width", "0")),
                "intrinsic_height": int(request.form.get("intrinsic_height", "0")),
            }
            if min(
                raw_camera_metadata["fx"],
                raw_camera_metadata["fy"],
                raw_camera_metadata["intrinsic_width"],
                raw_camera_metadata["intrinsic_height"],
            ) <= 0:
                raise ValueError("mobile diagnostic camera intrinsics are invalid")
            raw_image_pose_time_delta = _required_form_float(
                "raw_cpu_to_camera_frame_timestamp_delta_s"
            )
            screen_image_pose_time_delta = _required_form_float(
                "camera_pose_to_screen_capture_delta_s"
            )
            for domain, delta in (
                ("raw camera", raw_image_pose_time_delta),
                ("screen capture", screen_image_pose_time_delta),
            ):
                if not 0.0 <= delta <= MOBILE_OVERLAY_MAX_IMAGE_POSE_TIME_DELTA_SECONDS:
                    raise RuntimeError(
                        f"{domain} image/pose timestamp delta is not strict: "
                        f"{delta:.6f} s > "
                        f"{MOBILE_OVERLAY_MAX_IMAGE_POSE_TIME_DELTA_SECONDS:.6f} s"
                    )
            display_matrix = _bounded_mobile_form_string("display_matrix")
            projection_matrix = _bounded_mobile_form_string("projection_matrix")
            raw_display_matrix = _bounded_mobile_form_string(
                "raw_display_matrix"
            )
            raw_projection_matrix = _bounded_mobile_form_string(
                "raw_projection_matrix"
            )
            a0_world_position = _mobile_overlay_form_vector(
                "a0_world_pos", ("x", "y", "z")
            )
            a0_world_quaternion = _mobile_overlay_form_vector(
                "a0_world_quat", ("x", "y", "z", "w")
            )
            camera_world_position = _mobile_overlay_form_vector(
                "camera_world_pos", ("x", "y", "z")
            )
            camera_world_quaternion = _mobile_overlay_form_vector(
                "camera_world_quat", ("x", "y", "z", "w")
            )
            mesh_world_position = _mobile_overlay_form_vector(
                "mesh_world_pos", ("x", "y", "z")
            )
            mesh_world_quaternion = _mobile_overlay_form_vector(
                "mesh_world_quat", ("x", "y", "z", "w")
            )
            mesh_world_scale = _mobile_overlay_form_vector(
                "mesh_world_scale", ("x", "y", "z")
            )
            for name, quaternion in (
                ("A0 world", a0_world_quaternion),
                ("camera world", camera_world_quaternion),
                ("Mesh world", mesh_world_quaternion),
            ):
                if np.linalg.norm(quaternion) <= 1.0e-6:
                    raise ValueError(f"mobile diagnostic {name} quaternion is zero")

            server_raw_overlay_path = (
                root
                / "04_server_raw_sensor_reprojection"
                / f"frame_{index:02d}.png"
            )
            try:
                server_reprojection = _render_mobile_pose_diagnostic(
                    mobile_mesh=Path(str(state["mobile_mesh"])),
                    raw_camera_path=raw_camera_path,
                    camera_metadata=raw_camera_metadata,
                    mesh_transform_unity=mesh_transform,
                    destination=server_raw_overlay_path,
                )
            except Exception as projection_error:
                transport_server.legacy.logging.exception(
                    "server same-pose mobile Mesh reprojection failed"
                )
                server_reprojection = _write_failed_mobile_pose_diagnostic(
                    raw_camera_path, server_raw_overlay_path, projection_error
                )
            server_display_overlay_path = (
                root
                / "05_server_display_aligned_reprojection"
                / f"frame_{index:02d}.png"
            )
            display_alignment = _write_display_aligned_mobile_image(
                server_raw_overlay_path,
                server_display_overlay_path,
                display_matrix=raw_display_matrix,
                display_size=(composite_width, composite_height),
            )
            target_copy_path = (
                root
                / "06_original_reconstruction_input"
                / f"target_{target_index:02d}_{Path(target_source_frame_name).name}"
            )
            shutil.copy2(Path(str(target["source_image"])), target_copy_path)
            target_comparison_path = (
                root
                / "07_live_vs_reconstruction_input"
                / f"frame_{index:02d}_target_{target_index:02d}.jpg"
            )
            _write_raw_target_pair(
                target_copy_path, raw_camera_path, target_comparison_path
            )
            comparison_path = (
                root
                / "08_phone_vs_server_display_comparison"
                / f"frame_{index:02d}.jpg"
            )
            _write_mobile_pose_pair(
                composite_path, server_display_overlay_path, comparison_path
            )
            screen_vs_raw_pose_delta = _mobile_pose_delta(
                screen_camera_position,
                screen_camera_quaternion,
                raw_camera_position,
                raw_camera_quaternion,
            )
            row = {
                "frame_index": index,
                "matched_reconstruction_target": {
                    **_public_mobile_pose_target(target),
                    "raw_camera_pose_delta": raw_target_pose_delta,
                    "screen_camera_pose_delta": screen_target_pose_delta,
                    "translation_tolerance_meters": translation_tolerance,
                    "rotation_tolerance_degrees": rotation_tolerance,
                    "strict_match_passed": True,
                    "original_input_copy": str(target_copy_path.resolve()),
                    "original_input_copy_sha256": sha256_file(target_copy_path),
                    "live_vs_original_comparison": str(
                        target_comparison_path.resolve()
                    ),
                    "live_vs_original_comparison_sha256": sha256_file(
                        target_comparison_path
                    ),
                },
                "captured_at_utc": utc_now(),
                "screen_composite": str(composite_path.resolve()),
                "screen_composite_sha256": sha256_file(composite_path),
                "screen_composite_size": [composite_width, composite_height],
                "outline_texture": (
                    str(outline_path.resolve()) if outline_path is not None else None
                ),
                "outline_texture_sha256": (
                    sha256_file(outline_path) if outline_path is not None else None
                ),
                "outline_texture_size": [outline_width, outline_height],
                "raw_camera_rgb": str(raw_camera_path.resolve()),
                "raw_camera_rgb_sha256": sha256_file(raw_camera_path),
                "raw_camera_rgb_size": [raw_camera_width, raw_camera_height],
                "server_same_pose_reprojection": server_reprojection,
                "server_display_aligned_reprojection": display_alignment,
                "phone_vs_server_comparison": str(comparison_path.resolve()),
                "phone_vs_server_comparison_sha256": sha256_file(
                    comparison_path
                ),
                "screen_camera_pose_a0": {
                    "position": screen_camera_position,
                    "quaternion_xyzw": screen_camera_quaternion,
                },
                "raw_camera_pose_a0": {
                    "position": raw_camera_position,
                    "quaternion_xyzw": raw_camera_quaternion,
                },
                "screen_vs_raw_camera_pose_delta": screen_vs_raw_pose_delta,
                "mesh_local_transform_a0": {
                    "position": mesh_position,
                    "quaternion_xyzw": mesh_quaternion,
                    "uniform_scale": mesh_scale,
                },
                "screen_capture_realtime_s": _required_form_float(
                    "screen_capture_realtime_s"
                ),
                "camera_pose_sample_realtime_s": _required_form_float(
                    "camera_pose_sample_realtime_s"
                ),
                "camera_pose_to_screen_capture_delta_s": (
                    screen_image_pose_time_delta
                ),
                "raw_cpu_image_timestamp_s": _required_form_float(
                    "raw_cpu_image_timestamp_s"
                ),
                "raw_camera_frame_timestamp_ns": int(
                    request.form.get("raw_camera_frame_timestamp_ns", "-1")
                ),
                "raw_pose_sample_realtime_s": _required_form_float(
                    "raw_pose_sample_realtime_s"
                ),
                "raw_cpu_to_camera_frame_timestamp_delta_s": (
                    raw_image_pose_time_delta
                ),
                "maximum_image_pose_time_delta_seconds": (
                    MOBILE_OVERLAY_MAX_IMAGE_POSE_TIME_DELTA_SECONDS
                ),
                "raw_image_transform": _bounded_mobile_form_string(
                    "raw_image_transform", maximum=64
                ),
                "raw_cpu_image_size": [
                    int(request.form.get("raw_cpu_image_width", "0")),
                    int(request.form.get("raw_cpu_image_height", "0")),
                ],
                "intrinsics": {
                    key: raw_camera_metadata[key]
                    for key in (
                        "fx",
                        "fy",
                        "cx",
                        "cy",
                        "intrinsic_width",
                        "intrinsic_height",
                    )
                },
                "screen_size": [
                    reported_screen_size[0],
                    reported_screen_size[1],
                ],
                "screen_capture_encoding": "png_lossless_native",
                "screen_orientation": str(
                    request.form.get("screen_orientation") or "unknown"
                ),
                "outline_method": str(
                    request.form.get("outline_method") or "unknown"
                ),
                "outline_display_requested": (
                    request.form.get("outline_display_requested") == "true"
                ),
                "display_matrix": display_matrix,
                "projection_matrix": projection_matrix,
                "raw_display_matrix": raw_display_matrix,
                "raw_projection_matrix": raw_projection_matrix,
                "capture_anchor_tracking_state": "Tracking",
                "capture_anchor": {
                    "trackable_id": _bounded_mobile_form_string(
                        "capture_anchor_trackable_id", maximum=128
                    ),
                    "world_position": a0_world_position,
                    "world_quaternion_xyzw": a0_world_quaternion,
                    "pose_valid": request.form.get(
                        "capture_anchor_pose_valid"
                    )
                    == "true",
                    "tracking_stable": request.form.get(
                        "capture_anchor_tracking_stable"
                    )
                    == "true",
                    "ever_tracked": request.form.get(
                        "capture_anchor_ever_tracked"
                    )
                    == "true",
                    "uses_tracked_ar_anchor": request.form.get(
                        "capture_anchor_uses_tracked_ar_anchor"
                    )
                    == "true",
                    "tracking_since_realtime_s": _required_form_float(
                        "capture_anchor_tracking_since_realtime_s"
                    ),
                },
                "camera_world_pose": {
                    "position": camera_world_position,
                    "quaternion_xyzw": camera_world_quaternion,
                },
                "mesh_world_transform": {
                    "position": mesh_world_position,
                    "quaternion_xyzw": mesh_world_quaternion,
                    "lossy_scale": mesh_world_scale,
                },
                "phone_runtime_state": {
                    "ar_session_state": _bounded_mobile_form_string(
                        "ar_session_state", maximum=128
                    ),
                    "application_paused": request.form.get(
                        "application_paused"
                    )
                    == "true",
                    "application_focused": request.form.get(
                        "application_focused"
                    )
                    == "true",
                    "camera_frame_sequence": int(
                        request.form.get("camera_frame_sequence", "-1")
                    ),
                    "device_model": _bounded_mobile_form_string(
                        "device_model", maximum=256
                    ),
                    "operating_system": _bounded_mobile_form_string(
                        "operating_system", maximum=512
                    ),
                    "application_version": _bounded_mobile_form_string(
                        "application_version", maximum=128
                    ),
                    "battery_level": _required_form_float("battery_level"),
                    "battery_status": _bounded_mobile_form_string(
                        "battery_status", maximum=64
                    ),
                    "alignment_refinement_state": _bounded_mobile_form_string(
                        "alignment_refinement_state", maximum=64
                    ),
                    "diagnostic_stage": _bounded_mobile_form_string(
                        "diagnostic_stage", maximum=96
                    ),
                    "mobile_realtime_s": _required_form_float(
                        "mobile_realtime_s"
                    ),
                },
            }
            metadata_path = root / "02_frame_metadata" / f"frame_{index:02d}.json"
            atomic_json(metadata_path, row)
            row["metadata"] = str(metadata_path.resolve())
            row["metadata_sha256"] = sha256_file(metadata_path)
            state["rows"].append(row)
            if len(state["rows"]) >= maximum:
                captured_target_indices = sorted(
                    int(item["matched_reconstruction_target"]["target_index"])
                    for item in state["rows"]
                )
                if captured_target_indices != list(range(maximum)):
                    raise RuntimeError(
                        "mobile diagnostic cannot complete without all reconstruction input poses"
                    )
                screen_images: list[Image.Image] = []
                server_images: list[Image.Image] = []
                comparison_images: list[Image.Image] = []
                target_comparison_images: list[Image.Image] = []
                labels: list[str] = []
                for item in state["rows"]:
                    with Image.open(item["screen_composite"]) as image:
                        screen_images.append(image.convert("RGB"))
                    with Image.open(
                        item["server_display_aligned_reprojection"]["overlay"]
                    ) as image:
                        server_images.append(image.convert("RGB"))
                    with Image.open(item["phone_vs_server_comparison"]) as image:
                        comparison_images.append(image.convert("RGB"))
                    with Image.open(
                        item["matched_reconstruction_target"][
                            "live_vs_original_comparison"
                        ]
                    ) as image:
                        target_comparison_images.append(image.convert("RGB"))
                    match = item["matched_reconstruction_target"]
                    labels.append(
                        f"target {match['target_index']}: "
                        f"{match['source_frame_name']} | "
                        f"d={match['raw_camera_pose_delta']['translation_meters'] * 100.0:.1f}cm "
                        f"a={match['raw_camera_pose_delta']['rotation_degrees']:.1f}deg"
                    )
                contact_sheet = root / "手机实际最终渲染总览.jpg"
                _make_mobile_contact_sheet(screen_images, labels, contact_sheet)
                server_sheet = root / "服务器显示方向同位姿Mesh复投影总览.jpg"
                _make_mobile_contact_sheet(server_images, labels, server_sheet)
                comparison_sheet = root / "手机实际与服务器复算逐帧对照总览.jpg"
                _make_mobile_contact_sheet(
                    comparison_images, labels, comparison_sheet
                )
                target_sheet = root / "重建原始输入与现场严格同位姿图像对照总览.jpg"
                _make_contact_sheet(
                    target_comparison_images, labels, target_sheet
                )
                for image in (
                    screen_images
                    + server_images
                    + comparison_images
                    + target_comparison_images
                ):
                    image.close()
                state["contact_sheet"] = str(contact_sheet.resolve())
                state["server_reprojection_contact_sheet"] = str(
                    server_sheet.resolve()
                )
                state["phone_vs_server_contact_sheet"] = str(
                    comparison_sheet.resolve()
                )
                state["live_vs_reconstruction_input_contact_sheet"] = str(
                    target_sheet.resolve()
                )
                state["status"] = "complete"
            report_path = _write_mobile_overlay_audit_report(state)
            complete = state["status"] == "complete"
            captured_count = len(state["rows"])

        return jsonify(
            {
                "status": "success",
                "session_id": session_id,
                "audit_id": audit_id,
                "captured_frames": captured_count,
                "maximum_frames": maximum,
                "complete": complete,
                "report": str(report_path),
                "diagnostic_only": True,
                "matched_target_index": target_index,
                "matched_target_source_frame_name": target_source_frame_name,
            }
        ), 200
    except Exception as error:
        transport_server.legacy.logging.exception(
            "mobile render overlay audit upload failed"
        )
        return jsonify(
            {"status": "error", "message": f"手机渲染回传失败：{error}"}
        ), 500


def configure_server(config: ServerConfig) -> None:
    global _CONFIG, _SAM2_CLIENT
    config.validate()
    config.output_root.mkdir(parents=True, exist_ok=True)
    _CONFIG = config
    _SAM2_CLIENT = Sam2TinyVideoWorkerClient(
        python=config.sam2_python,
        port=int(config.sam2_worker_port),
        log_path=config.output_root / "logs" / "sam2_tiny_video_worker.log",
    )
    transport_server.configure_server(
        config.output_root,
        ARPointFilterConfig(),
        reconstruction_config=None,
        require_point_cloud=False,
    )
    transport_server._RECONSTRUCTION_DEPLOYMENT = "manual_phase1_ss30k_slat30k"
    app = transport_server.legacy.app
    app.view_functions["input_qc"] = collection_input_qc
    app.view_functions["generate"] = collection_generate
    if "prepare_runtime_o" in app.view_functions:
        app.view_functions["prepare_runtime_o"] = collection_prepare_runtime_o
    else:
        app.add_url_rule(
            "/prepare_runtime_o",
            endpoint="prepare_runtime_o",
            view_func=collection_prepare_runtime_o,
            methods=["POST"],
        )
    if "manual_reconstruction_mesh" in app.view_functions:
        app.view_functions["manual_reconstruction_mesh"] = manual_reconstruction_mesh
    else:
        app.add_url_rule(
            "/manual_reconstruction_mesh/<session_id>",
            endpoint="manual_reconstruction_mesh",
            view_func=manual_reconstruction_mesh,
            methods=["GET"],
        )
    alignment_routes = (
        (
            "alignment_refine_start",
            "/alignment_refine/start",
            collection_alignment_refine_start,
        ),
        (
            "alignment_refine_upload",
            "/alignment_refine/upload",
            collection_alignment_refine_upload,
        ),
        (
            "alignment_refine_optimize",
            "/alignment_refine/optimize",
            collection_alignment_refine_optimize,
        ),
        (
            "mobile_overlay_audit_start",
            "/mobile_overlay_audit/start",
            collection_mobile_overlay_audit_start,
        ),
        (
            "mobile_overlay_audit_upload",
            "/mobile_overlay_audit/upload",
            collection_mobile_overlay_audit_upload,
        ),
    )
    for endpoint, rule, view in alignment_routes:
        if endpoint in app.view_functions:
            app.view_functions[endpoint] = view
        else:
            app.add_url_rule(rule, endpoint=endpoint, view_func=view, methods=["POST"])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--gpu", default=DEFAULT_GPU)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--contour_width", type=int, default=3)
    parser.add_argument("--o2w_refinement_iterations", type=int, default=40)
    parser.add_argument("--o2w_refinement_max_views", type=int, default=24)
    parser.add_argument("--o2w_refinement_render_long_side", type=int, default=160)
    parser.add_argument("--o2w_refinement_batch_size", type=int, default=4)
    parser.add_argument("--o2w_refinement_face_limit", type=int, default=30_000)
    parser.add_argument(
        "--sam2_python",
        type=Path,
        default=DEFAULT_SAM2_PYTHON,
        help="Python executable containing Hydra + SAM2.1 Tiny",
    )
    parser.add_argument("--sam2_worker_port", type=int, default=5091)
    parser.add_argument(
        "--capture_only",
        action="store_true",
        help="build all-view O and the spherical FPS8 selection without SS/SLat",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = ServerConfig(
        output_root=args.output_root.expanduser().resolve(),
        python=args.python.expanduser().resolve(),
        gpu=str(args.gpu),
        seed=int(args.seed),
        contour_width=int(args.contour_width),
        o2w_refinement_iterations=int(args.o2w_refinement_iterations),
        o2w_refinement_max_views=int(args.o2w_refinement_max_views),
        o2w_refinement_render_long_side=int(
            args.o2w_refinement_render_long_side
        ),
        o2w_refinement_batch_size=int(args.o2w_refinement_batch_size),
        o2w_refinement_face_limit=int(args.o2w_refinement_face_limit),
        selection_modes=SELECTION_MODES,
        capture_only=bool(args.capture_only),
        sam2_python=args.sam2_python.expanduser().resolve(),
        sam2_worker_port=int(args.sam2_worker_port),
    )
    configure_server(config)
    transport_server.legacy.clean_environment()
    print(">>> [manual phase1] 手机 AR 重建服务启动", flush=True)
    print(f">>> 输出根目录: {config.output_root}", flush=True)
    print(f">>> GPU: physical cuda:{config.gpu}", flush=True)
    print(f">>> 唯一分支: {PRIMARY_MODE}", flush=True)
    print(">>> 所有候选先构建唯一 official-compatible O，再执行训练一致球面FPS8", flush=True)
    print(">>> 不使用点云、质量排序、客户端/时间/轨迹/随机选帧", flush=True)
    print(f">>> AR 像素轴合同: {AXIS_CONTRACT}", flush=True)
    print(
        ">>> Anchor 合同: 首帧相机位置建重力轴 A0，全部相机位姿相对 A0；"
        "Mask 后冻结 O；返回 Mesh 已是 A0-local，单位变换直接挂 A0",
        flush=True,
    )
    print(">>> 模型: no-VGGT SS30K + SLat30K", flush=True)
    print(
        ">>> Mesh 后处理: 用原采集全部有效 mask/pose 优化 O2W；球面 FPS "
        f"最多 {config.o2w_refinement_max_views} 帧参与梯度优化，全部输入帧复投影验收",
        flush=True,
    )
    print(
        f">>> 快速回场校准: SAM2.1 Tiny video predictor 常驻端口 "
        f"127.0.0.1:{config.sam2_worker_port}；只优化 A0-local Mesh 小相似变换",
        flush=True,
    )
    print(">>> 输出: runtime-O/world/Unity Mesh + 标定帧与原手机朝向轮廓", flush=True)
    if config.capture_only:
        print(">>> capture_only 已启用：不执行 DINO/SS/SLat/Mesh", flush=True)
    transport_server.legacy.app.run(
        host=args.host,
        port=int(args.port),
        debug=False,
        use_reloader=False,
    )


if __name__ == "__main__":
    main()
