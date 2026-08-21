#!/usr/bin/env python3
"""Finalize ARFoundation captures into auditable object-point datasets."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
from typing import Any, Iterable, Sequence

import cv2
import numpy as np
from trellis_point_prior_mv.build_ar_session_smoke_dataset import (
    image_size,
    intrinsics_for_pose,
    read_phone_poses,
    rotmat_to_qvec,
    unity_pose_to_colmap_w2c,
    unity_world_point_to_colmap,
)


CAPTURE_FORMAT = "pose_point_depth_mv.ar_object_capture.v2"
COLLECTION_FORMAT = "pose_point_depth_mv.ar_object_capture_collection.v1"
IMAGE_CAMERA_ROTATION_CANDIDATES_DEGREES = (0.0, 90.0, -90.0, 180.0)
AR_GRAVITY_UP_COLMAP_W = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


@dataclass(frozen=True)
class ARPointFilterConfig:
    voxel_size_m: float = 0.005
    min_temporal_observations: int = 2
    min_mask_observations: int = 2
    min_mask_support_ratio: float = 0.35
    mask_dilation_px: int = 5
    min_object_points: int = 100
    max_object_points: int = 50000
    point_extent_quantile: float = 0.02
    max_point_to_mask_extent_ratio: float = 2.0
    max_ray_residual_median_over_mask_extent: float = 0.20
    max_ray_residual_p90_over_mask_extent: float = 0.40
    max_camera_roll_median_degrees: float = 12.5
    min_orbit_gravity_agreement: float = 0.80
    min_synchronized_frame_ratio: float = 0.90
    require_pose_mask_geometry: bool = True

    def validate(self) -> None:
        if self.voxel_size_m <= 0:
            raise ValueError("voxel_size_m must be positive")
        if self.min_temporal_observations < 1:
            raise ValueError("min_temporal_observations must be >= 1")
        if self.min_mask_observations < 1:
            raise ValueError("min_mask_observations must be >= 1")
        if not 0.0 <= self.min_mask_support_ratio <= 1.0:
            raise ValueError("min_mask_support_ratio must be in [0,1]")
        if self.mask_dilation_px < 0:
            raise ValueError("mask_dilation_px must be nonnegative")
        if self.min_object_points < 1:
            raise ValueError("min_object_points must be positive")
        if self.max_object_points < self.min_object_points:
            raise ValueError("max_object_points must be >= min_object_points")
        if not 0.0 <= self.point_extent_quantile < 0.5:
            raise ValueError("point_extent_quantile must be in [0,0.5)")
        if self.max_point_to_mask_extent_ratio <= 1.0:
            raise ValueError("max_point_to_mask_extent_ratio must be > 1")
        if self.max_ray_residual_median_over_mask_extent <= 0.0:
            raise ValueError("max ray residual median must be positive")
        if self.max_ray_residual_p90_over_mask_extent <= 0.0:
            raise ValueError("max ray residual p90 must be positive")
        if not 0.0 < self.max_camera_roll_median_degrees < 180.0:
            raise ValueError("max camera roll median must be in (0,180)")
        if not 0.0 <= self.min_orbit_gravity_agreement <= 1.0:
            raise ValueError("min_orbit_gravity_agreement must be in [0,1]")
        if not 0.0 <= self.min_synchronized_frame_ratio <= 1.0:
            raise ValueError("min_synchronized_frame_ratio must be in [0,1]")


@dataclass(frozen=True)
class ProjectionView:
    frame_name: str
    image_path: Path
    mask_path: Path
    K: np.ndarray
    T_W2C: np.ndarray
    image_rgb: np.ndarray
    mask: np.ndarray


def _finite_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def iter_ar_point_rows(path: Path) -> Iterable[tuple[str, np.ndarray, float]]:
    """Yield frame-associated Unity-world points from the upload JSONL."""

    if not path.is_file():
        return
    for line_number, raw in enumerate(
        path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1
    ):
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid AR point JSONL line {line_number}: {path}") from exc
        if row.get("coordinate_frame", "unity_world") != "unity_world":
            raise ValueError(
                f"unsupported AR point frame on line {line_number}: "
                f"{row.get('coordinate_frame')!r}"
            )
        frame_name = str(row.get("frame_name") or f"line_{line_number:06d}")
        points = row.get("points")
        if not isinstance(points, list):
            continue
        for item in points:
            if isinstance(item, dict):
                values = (item.get("x"), item.get("y"), item.get("z"))
                confidence = item.get("confidence", item.get("conf", 1.0))
            elif isinstance(item, (list, tuple)) and len(item) >= 3:
                values = item[:3]
                confidence = item[3] if len(item) > 3 else 1.0
            else:
                continue
            xyz = [_finite_float(value) for value in values]
            conf = _finite_float(confidence)
            if any(value is None for value in xyz):
                continue
            point = unity_world_point_to_colmap(np.asarray(xyz, dtype=np.float64))
            yield frame_name, point, float(1.0 if conf is None else max(conf, 0.0))


def fuse_ar_points(
    rows: Iterable[tuple[str, np.ndarray, float]], config: ARPointFilterConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Voxel-fuse repeated snapshots and retain temporal observation counts."""

    config.validate()
    accumulators: dict[tuple[int, int, int], dict[str, Any]] = {}
    raw_count = 0
    source_frames: set[str] = set()
    for frame_name, point, confidence in rows:
        point = np.asarray(point, dtype=np.float64)
        if point.shape != (3,) or not np.isfinite(point).all():
            continue
        raw_count += 1
        source_frames.add(frame_name)
        key = tuple(np.floor(point / config.voxel_size_m).astype(np.int64).tolist())
        cell = accumulators.setdefault(
            key,
            {
                "weighted_sum": np.zeros(3, dtype=np.float64),
                "weight": 0.0,
                "confidence_sum": 0.0,
                "sample_count": 0,
                "frames": set(),
            },
        )
        weight = max(float(confidence), 1.0e-3)
        cell["weighted_sum"] += point * weight
        cell["weight"] += weight
        cell["confidence_sum"] += float(confidence)
        cell["sample_count"] += 1
        cell["frames"].add(frame_name)

    fused = []
    confidence = []
    temporal = []
    for key in sorted(accumulators):
        cell = accumulators[key]
        frame_count = len(cell["frames"])
        if frame_count < config.min_temporal_observations:
            continue
        fused.append(cell["weighted_sum"] / max(cell["weight"], 1.0e-12))
        confidence.append(cell["confidence_sum"] / max(cell["sample_count"], 1))
        temporal.append(frame_count)

    points = (
        np.asarray(fused, dtype=np.float64).reshape(-1, 3)
        if fused
        else np.zeros((0, 3), dtype=np.float64)
    )
    confidence_array = np.asarray(confidence, dtype=np.float64)
    temporal_array = np.asarray(temporal, dtype=np.int64)
    return points, confidence_array, temporal_array, {
        "raw_sample_count": int(raw_count),
        "raw_source_frame_count": int(len(source_frames)),
        "voxel_count": int(len(accumulators)),
        "temporally_supported_voxel_count": int(len(points)),
    }


def _load_projection_view(
    frame_name: str,
    image_path: Path,
    mask_path: Path,
    pose: dict[str, Any],
    *,
    image_camera_rotation_degrees: float = 90.0,
) -> ProjectionView:
    width, height = image_size(image_path)
    fx, fy, cx, cy, _source = intrinsics_for_pose(pose, width, height)
    K = np.asarray([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
    rotation, translation = unity_pose_to_colmap_w2c(
        pose,
        image_camera_rotation_degrees=image_camera_rotation_degrees,
    )
    T_W2C = np.eye(4, dtype=np.float64)
    T_W2C[:3, :3] = rotation
    T_W2C[:3, 3] = translation
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if image_bgr is None:
        raise FileNotFoundError(f"cannot read capture image: {image_path}")
    if mask is None:
        raise FileNotFoundError(f"cannot read capture mask: {mask_path}")
    if mask.shape != image_bgr.shape[:2]:
        mask = cv2.resize(
            mask, (image_bgr.shape[1], image_bgr.shape[0]), interpolation=cv2.INTER_NEAREST
        )
    return ProjectionView(
        frame_name=frame_name,
        image_path=image_path,
        mask_path=mask_path,
        K=K,
        T_W2C=T_W2C,
        image_rgb=image_bgr[:, :, ::-1],
        mask=mask,
    )


def build_projection_views(
    data_dir: Path,
    mask_dir: Path,
    frame_names: Sequence[str],
    *,
    poses: dict[str, dict] | None = None,
    image_camera_rotation_degrees: float = 90.0,
) -> list[ProjectionView]:
    poses = poses or read_phone_poses(data_dir / "poses.txt")
    views = []
    for frame_name in frame_names:
        if frame_name not in poses:
            raise FileNotFoundError(f"phone pose missing for selected frame: {frame_name}")
        image_path = data_dir / frame_name
        mask_path = mask_dir / f"{Path(frame_name).stem}.png"
        views.append(
            _load_projection_view(
                frame_name,
                image_path,
                mask_path,
                poses[frame_name],
                image_camera_rotation_degrees=image_camera_rotation_degrees,
            )
        )
    return views


def _pose_mask_geometry(views: Sequence[ProjectionView]) -> dict[str, Any]:
    from pose_point_depth_mv.pose_mask_object_canonicalization import (
        canonicalize_pose_mask_runtime_object_frame,
    )

    frame = canonicalize_pose_mask_runtime_object_frame(
        np.stack([view.K for view in views]),
        np.stack([view.T_W2C for view in views]),
        [view.mask for view in views],
        gravity_up_W=AR_GRAVITY_UP_COLMAP_W,
        reference_view_index=0,
    )
    camera_centers_w = np.linalg.inv(
        np.stack([view.T_W2C for view in views])
    )[:, :3, 3]
    camera_centers_o = (
        frame.T_W2O[:3, :3] @ camera_centers_w.T
    ).T + frame.T_W2O[:3, 3]
    azimuth = np.degrees(
        np.arctan2(camera_centers_o[:, 0], camera_centers_o[:, 2])
    ) % 360.0
    horizontal = np.linalg.norm(camera_centers_o[:, [0, 2]], axis=1)
    elevation = np.degrees(
        np.arctan2(camera_centers_o[:, 1], np.maximum(horizontal, 1.0e-12))
    )
    sorted_azimuth = np.sort(azimuth)
    gaps = np.diff(np.concatenate((sorted_azimuth, sorted_azimuth[:1] + 360.0)))
    roll_by_name = {}
    for view in views:
        rotation_w2c = np.asarray(view.T_W2C[:3, :3], dtype=np.float64)
        forward_w = rotation_w2c.T @ np.asarray([0.0, 0.0, 1.0])
        gravity_in_image_plane = AR_GRAVITY_UP_COLMAP_W - (
            float(np.dot(AR_GRAVITY_UP_COLMAP_W, forward_w)) * forward_w
        )
        norm = float(np.linalg.norm(gravity_in_image_plane))
        if norm <= 1.0e-10:
            roll = 90.0
        else:
            image_up_w = rotation_w2c.T @ np.asarray([0.0, -1.0, 0.0])
            cosine = float(
                np.clip(np.dot(image_up_w, gravity_in_image_plane / norm), -1.0, 1.0)
            )
            roll = float(np.degrees(np.arccos(cosine)))
        roll_by_name[view.frame_name] = roll
    roll_values = np.asarray(list(roll_by_name.values()), dtype=np.float64)
    stats = frame.stats
    ray_center = stats["ray_center"]
    mask_extent = max(float(stats["mask_extent_median_W"]), 1.0e-12)
    ray_residual_over_mask_extent_by_name = {
        views[int(view_index)].frame_name: float(residual) / mask_extent
        for view_index, residual in zip(
            ray_center["used_view_indices"],
            ray_center["ray_residuals"],
        )
    }
    return {
        "available": True,
        "mask_ray_center_W": frame.T_O2W[:3, 3].tolist(),
        "mask_extent_median_W": float(stats["mask_extent_median_W"]),
        "ray_residual_median_over_mask_extent": float(
            stats["ray_residual_median_over_mask_extent"]
        ),
        "ray_residual_p90_over_mask_extent": float(
            stats["ray_residual_p90_over_mask_extent"]
        ),
        "ray_residual_max_over_mask_extent": float(
            stats["ray_residual_max_over_mask_extent"]
        ),
        "ray_residual_over_mask_extent_by_name": (
            ray_residual_over_mask_extent_by_name
        ),
        "orbit_gravity_agreement": float(
            stats["axes"]["orbit_camera_up_agreement"]
        ),
        "azimuth_span_deg": float(360.0 - np.max(gaps)),
        "azimuth_by_name": {
            view.frame_name: float(value) for view, value in zip(views, azimuth)
        },
        "elevation_range_deg": float(np.max(elevation) - np.min(elevation)),
        "camera_roll_median_degrees": float(np.median(roll_values)),
        "camera_roll_p90_degrees": float(np.quantile(roll_values, 0.90)),
        "camera_roll_max_degrees": float(np.max(roll_values)),
        "camera_roll_degrees_by_name": roll_by_name,
    }


def _parse_unity_matrix(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if isinstance(value, str):
        fields = value.split()
    else:
        fields = list(value)
    if len(fields) != 16:
        return None
    try:
        matrix = np.asarray([float(field) for field in fields], dtype=np.float64).reshape(4, 4)
    except (TypeError, ValueError):
        return None
    return matrix if np.isfinite(matrix).all() else None


def _saved_image_axis_contract(
    poses: dict[str, dict], frame_names: Sequence[str]
) -> dict[str, Any]:
    """Describe pixel storage and pose-to-native-image axes independently.

    ARFoundation supplies the display matrix as the UV transform used while the
    camera texture is rendered.  The client saves
    ``XRCpuImage.Transformation.None`` directly, so that transform has not been
    baked into the JPEG.  Its in-plane UV rotation therefore maps the Unity
    display-camera axes into the saved native CPU-image axes directly.  Treating
    it as a native-to-screen Euclidean transform and inverting it swaps +90 and
    -90 degrees on portrait ARCore captures.
    """

    rows = [poses[name] for name in frame_names if name in poses]
    transforms = [str(row.get("image_transform") or "unknown") for row in rows]
    orientations = [str(row.get("screen_orientation") or "unknown") for row in rows]
    matrices = [
        matrix
        for matrix in (_parse_unity_matrix(row.get("display_matrix")) for row in rows)
        if matrix is not None
    ]
    matrix_hashes = [
        hashlib.sha256(np.ascontiguousarray(matrix).view(np.uint8)).hexdigest()
        for matrix in matrices
    ]
    display_rotations = []
    for matrix in matrices:
        linear = matrix[:2, :2]
        left, _singular, right = np.linalg.svd(linear)
        rotation = left @ right
        if float(np.linalg.det(rotation)) < 0.0:
            left[:, -1] *= -1.0
            rotation = left @ right
        display_rotations.append(
            float(np.degrees(np.arctan2(rotation[1, 0], rotation[0, 0])))
        )

    def normalized_degrees(value: float) -> float:
        result = (float(value) + 180.0) % 360.0 - 180.0
        return 180.0 if np.isclose(result, -180.0) else result

    display_rotation = display_rotations[0] if display_rotations else None
    display_rotation_max_delta = (
        None
        if not display_rotations
        else float(
            max(
                abs(normalized_degrees(value - display_rotations[0]))
                for value in display_rotations
            )
        )
    )
    complete_native_capture = bool(
        rows
        and len(rows) == len(frame_names)
        and len(matrices) == len(rows)
        and all(value.lower() == "none" for value in transforms)
    )
    failures = []
    if complete_native_capture and len(set(orientations)) != 1:
        failures.append("screen orientation changed within the selected capture")
    if (
        complete_native_capture
        and display_rotation_max_delta is not None
        and display_rotation_max_delta > 1.0
    ):
        failures.append("display-matrix rotation changed within the selected capture")

    pose_to_native_rotation = None
    if complete_native_capture and display_rotation is not None and not failures:
        cardinal = min(
            IMAGE_CAMERA_ROTATION_CANDIDATES_DEGREES,
            key=lambda value: abs(
                normalized_degrees(display_rotation - float(value))
            ),
        )
        cardinal_error = abs(
            normalized_degrees(display_rotation - float(cardinal))
        )
        if cardinal_error > 5.0:
            failures.append(
                "display-matrix UV rotation "
                f"{display_rotation:.3f} deg is not near a cardinal axis"
            )
        else:
            pose_to_native_rotation = float(cardinal)

    axis_contract_valid = bool(
        complete_native_capture
        and pose_to_native_rotation is not None
        and not failures
    )
    return {
        "available": bool(matrices),
        "complete_native_capture": complete_native_capture,
        "axis_contract_valid": axis_contract_valid,
        "axis_contract_failures": failures,
        "selected_frame_count": int(len(frame_names)),
        "parsed_display_matrix_count": int(len(matrices)),
        "display_matrix_unique_sha256": sorted(set(matrix_hashes)),
        "display_matrix_max_abs_delta": (
            None
            if not matrices
            else float(max(np.max(np.abs(matrix - matrices[0])) for matrix in matrices))
        ),
        "display_rotation_degrees_native_to_screen": display_rotation,
        "display_rotation_degrees_for_screen_only": display_rotation,
        "display_matrix_uv_rotation_degrees": display_rotation,
        "display_rotation_max_delta_degrees": display_rotation_max_delta,
        "image_transforms": sorted(set(transforms)),
        "screen_orientations": sorted(set(orientations)),
        "saved_image_rotation_degrees": 0.0 if complete_native_capture else None,
        "pose_camera_to_native_image_rotation_degrees": pose_to_native_rotation,
        "contract_source": (
            "xrcpuimage_none_direct_display_uv_v5"
            if axis_contract_valid
            else "incomplete_capture_metadata"
        ),
        "display_matrix_application": (
            "direct_pose_camera_to_native_cpu_image_uv_rotation"
            if axis_contract_valid
            else None
        ),
        "display_matrix_applied_to_saved_jpeg": False,
    }


def resolve_image_camera_rotation_from_metadata(
    poses: dict[str, dict], frame_names: Sequence[str]
) -> tuple[float, dict[str, Any]]:
    """Resolve pose-camera to stored-image axes without cross-view geometry."""

    saved_image_contract = _saved_image_axis_contract(poses, frame_names)
    contracted_angle = saved_image_contract.get(
        "pose_camera_to_native_image_rotation_degrees"
    )
    if saved_image_contract.get("complete_native_capture") and contracted_angle is None:
        raise RuntimeError(
            "saved CPU-image/display axis metadata is inconsistent: "
            + "; ".join(saved_image_contract.get("axis_contract_failures") or [])
        )
    if contracted_angle is None:
        # Pre-v2 clients did not record a display matrix. Their established export
        # contract used Rz(+90deg); runtime final-view QC validates the result.
        contracted_angle = 90.0
        selection_source = "legacy_explicit_Rz(+90deg)_camera_axis_contract"
        selection_rule = "legacy capture metadata contract; explicit Rz(+90deg)"
    else:
        selection_source = saved_image_contract["contract_source"]
        selection_rule = (
            "direct saved XRCpuImage display-matrix UV rotation"
        )
    angle = float(contracted_angle)
    return angle, {
        "available": True,
        "resolution_mode": "capture_metadata_only",
        "cross_view_geometry_evaluated": False,
        "selection_rule": selection_rule,
        "selection_source": selection_source,
        "saved_image_axis_contract": saved_image_contract,
        "selected_image_camera_rotation_degrees": angle,
    }


def select_image_camera_rotation(
    data_dir: Path,
    mask_dir: Path,
    frame_names: Sequence[str],
    *,
    poses: dict[str, dict] | None = None,
    candidates_degrees: Sequence[float] = IMAGE_CAMERA_ROTATION_CANDIDATES_DEGREES,
) -> tuple[float, list[ProjectionView], dict[str, Any]]:
    """Resolve saved-image axes from capture metadata and validate with geometry."""

    poses = poses or read_phone_poses(data_dir / "poses.txt")
    contracted_angle, metadata_diagnostics = (
        resolve_image_camera_rotation_from_metadata(poses, frame_names)
    )
    candidates = []
    views_by_angle: dict[float, list[ProjectionView]] = {}
    for raw_angle in candidates_degrees:
        angle = float(raw_angle)
        try:
            views = build_projection_views(
                data_dir,
                mask_dir,
                frame_names,
                poses=poses,
                image_camera_rotation_degrees=angle,
            )
            geometry = _pose_mask_geometry(views)
            views_by_angle[angle] = views
            candidates.append(
                {
                    "image_camera_rotation_degrees": angle,
                    **geometry,
                }
            )
        except Exception as exc:
            candidates.append(
                {
                    "image_camera_rotation_degrees": angle,
                    "available": False,
                    "error": repr(exc),
                }
            )
    valid = [row for row in candidates if row.get("available")]
    if not valid:
        views = build_projection_views(
            data_dir,
            mask_dir,
            frame_names,
            poses=poses,
            image_camera_rotation_degrees=contracted_angle,
        )
        return contracted_angle, views, {
            **metadata_diagnostics,
            "available": False,
            "cross_view_geometry_evaluated": True,
            "geometry_validation": "all pose-mask candidates failed",
            "candidates": candidates,
        }
    selected = next(
        (
            row
            for row in valid
            if np.isclose(
                float(row["image_camera_rotation_degrees"]), float(contracted_angle)
            )
        ),
        None,
    )
    if selected is None:
        raise RuntimeError(
            f"metadata-contracted camera-axis angle {contracted_angle} is unavailable"
        )
    angle = float(selected["image_camera_rotation_degrees"])
    return angle, views_by_angle[angle], {
        **selected,
        **metadata_diagnostics,
        "available": True,
        "cross_view_geometry_evaluated": True,
        "selection_rule": (
            "direct saved XRCpuImage display-matrix UV rotation; "
            "mask-ray residual and gravity roll are QC only"
        ),
        "candidates": candidates,
    }


def filter_points_by_masks(
    points: np.ndarray,
    source_confidence: np.ndarray,
    temporal_observations: np.ndarray,
    views: Sequence[ProjectionView],
    config: ARPointFilterConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Keep visual-hull-supported points and assign image-derived colors."""

    config.validate()
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1:] != (3,):
        raise ValueError(f"points must be [N,3], got {points.shape}")
    if len(source_confidence) != len(points) or len(temporal_observations) != len(points):
        raise ValueError("point metadata length mismatch")
    if not views:
        raise ValueError("at least one projection view is required")

    observed = np.zeros(len(points), dtype=np.int32)
    mask_hits = np.zeros(len(points), dtype=np.int32)
    color_sum = np.zeros((len(points), 3), dtype=np.float64)
    color_count = np.zeros(len(points), dtype=np.int32)
    per_view = []
    for view in views:
        mask = view.mask
        if config.mask_dilation_px > 0:
            radius = int(config.mask_dilation_px)
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
            )
            mask = cv2.dilate(mask, kernel)
        camera = (view.T_W2C[:3, :3] @ points.T).T + view.T_W2C[:3, 3]
        z = camera[:, 2]
        projected = (view.K @ camera.T).T
        uv = np.zeros((len(points), 2), dtype=np.float64)
        positive = z > 1.0e-6
        uv[positive] = projected[positive, :2] / z[positive, None]
        height, width = mask.shape
        inside = (
            positive
            & (uv[:, 0] >= 0.0)
            & (uv[:, 0] < width)
            & (uv[:, 1] >= 0.0)
            & (uv[:, 1] < height)
        )
        ids = np.flatnonzero(inside)
        observed[ids] += 1
        view_hit_count = 0
        if len(ids):
            x = np.clip(np.rint(uv[ids, 0]).astype(np.int64), 0, width - 1)
            y = np.clip(np.rint(uv[ids, 1]).astype(np.int64), 0, height - 1)
            hit = mask[y, x] > 127
            hit_ids = ids[hit]
            view_hit_count = int(np.count_nonzero(hit))
            mask_hits[hit_ids] += 1
            color_sum[hit_ids] += view.image_rgb[y[hit], x[hit]].astype(np.float64)
            color_count[hit_ids] += 1
        per_view.append(
            {
                "frame_name": view.frame_name,
                "projected_point_count": int(len(ids)),
                "mask_hit_point_count": view_hit_count,
            }
        )

    support_ratio = mask_hits.astype(np.float64) / np.maximum(observed, 1)
    keep = (
        (mask_hits >= config.min_mask_observations)
        & (support_ratio >= config.min_mask_support_ratio)
    )
    keep_ids = np.flatnonzero(keep)
    if len(keep_ids) > config.max_object_points:
        temporal_norm = temporal_observations.astype(np.float64) / max(
            int(temporal_observations.max()), 1
        )
        ranking_score = (
            0.45 * support_ratio
            + 0.35 * temporal_norm
            + 0.20 * np.clip(source_confidence, 0.0, 1.0)
        )
        keep_ids = keep_ids[
            np.argsort(-ranking_score[keep_ids], kind="stable")[: config.max_object_points]
        ]
        keep_ids.sort()

    filtered_points = points[keep_ids]
    filtered_confidence = np.clip(
        0.5 * support_ratio[keep_ids]
        + 0.3
        * (
            temporal_observations[keep_ids]
            / max(int(temporal_observations.max()) if len(temporal_observations) else 1, 1)
        )
        + 0.2 * np.clip(source_confidence[keep_ids], 0.0, 1.0),
        0.0,
        1.0,
    )
    colors = np.full((len(keep_ids), 3), 128, dtype=np.uint8)
    valid_color = color_count[keep_ids] > 0
    if np.any(valid_color):
        colors[valid_color] = np.clip(
            color_sum[keep_ids[valid_color]]
            / color_count[keep_ids[valid_color], None],
            0,
            255,
        ).astype(np.uint8)
    return filtered_points, filtered_confidence, colors, {
        "projection_view_count": int(len(views)),
        "mask_supported_point_count": int(len(filtered_points)),
        "mask_supported_fraction": float(len(filtered_points) / max(len(points), 1)),
        "mask_hit_count_median": float(np.median(mask_hits[keep_ids])) if len(keep_ids) else 0.0,
        "mask_support_ratio_median": (
            float(np.median(support_ratio[keep_ids])) if len(keep_ids) else 0.0
        ),
        "temporal_observation_median": (
            float(np.median(temporal_observations[keep_ids])) if len(keep_ids) else 0.0
        ),
        "per_view": per_view,
    }


def robust_point_extent(points: np.ndarray, quantile: float) -> float:
    if len(points) == 0:
        return 0.0
    center = np.median(points, axis=0)
    local = points - center
    low = np.quantile(local, quantile, axis=0)
    high = np.quantile(local, 1.0 - quantile, axis=0)
    return float(2.0 * np.max(np.maximum(np.abs(low), np.abs(high))))


def mask_frame_diagnostics(
    points: np.ndarray,
    views: Sequence[ProjectionView],
    config: ARPointFilterConfig,
    *,
    pose_mask_geometry: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compare point extent with the independently mask-derived object frame."""

    try:
        geometry = pose_mask_geometry or _pose_mask_geometry(views)
        mask_extent = float(geometry["mask_extent_median_W"])
        point_extent = robust_point_extent(points, config.point_extent_quantile)
        ratio = point_extent / max(mask_extent, 1.0e-12)
        return {
            **geometry,
            "point_extent_W": point_extent,
            "point_to_mask_extent_ratio": ratio,
        }
    except Exception as exc:
        return {"available": False, "error": repr(exc)}


def write_points3d(
    path: Path, points: np.ndarray, colors: np.ndarray, confidence: np.ndarray
) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# 3D point list with one line of data per point:\n")
        handle.write(
            "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, "
            "TRACK[] as (IMAGE_ID, POINT2D_IDX)\n"
        )
        handle.write(f"# Number of points: {len(points)}, mean track length: 0\n")
        for index, (point, color, score) in enumerate(
            zip(points, colors, confidence), start=1
        ):
            error = max(0.0, 1.0 - float(score))
            handle.write(
                f"{index} {point[0]:.9f} {point[1]:.9f} {point[2]:.9f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])} {error:.6f}\n"
            )


def write_point_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write(f"element vertex {len(points)}\n")
        handle.write("property float x\nproperty float y\nproperty float z\n")
        handle.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        handle.write("end_header\n")
        for point, color in zip(points, colors):
            handle.write(
                f"{point[0]:.9f} {point[1]:.9f} {point[2]:.9f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )


def write_phone_sparse_model(
    dataset_dir: Path,
    frame_names: Sequence[str],
    poses: dict[str, dict],
    *,
    image_camera_rotation_degrees: float,
    camera_axis_diagnostics: dict[str, Any],
    point_source: str = "ar_foundation_arpointcloudmanager",
) -> tuple[Path, dict[str, Any]]:
    sparse = dataset_dir / "sparse" / "0"
    sparse.mkdir(parents=True, exist_ok=True)
    camera_lines = []
    image_lines = []
    intrinsics_sources: dict[str, int] = {}
    for image_id, frame_name in enumerate(frame_names, start=1):
        image_path = dataset_dir / "images" / frame_name
        width, height = image_size(image_path)
        pose = poses[frame_name]
        rotation, translation = unity_pose_to_colmap_w2c(
            pose,
            image_camera_rotation_degrees=image_camera_rotation_degrees,
        )
        quaternion = rotmat_to_qvec(rotation)
        fx, fy, cx, cy, source = intrinsics_for_pose(pose, width, height)
        intrinsics_sources[source] = intrinsics_sources.get(source, 0) + 1
        camera_lines.append(
            f"{image_id} PINHOLE {width} {height} {fx:.10f} {fy:.10f} "
            f"{cx:.10f} {cy:.10f}\n"
        )
        image_lines.append(
            f"{image_id} {quaternion[0]:.12f} {quaternion[1]:.12f} "
            f"{quaternion[2]:.12f} {quaternion[3]:.12f} "
            f"{translation[0]:.12f} {translation[1]:.12f} "
            f"{translation[2]:.12f} {image_id} {frame_name}\n"
            # Keep a nonempty legal POINTS2D line because the repository's
            # strict COLMAP parser removes blank lines before pairing records.
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
    metadata = {
        "pose_source": "unity_ar_pose",
        "point_source": str(point_source),
        "num_images": len(frame_names),
        "intrinsics_sources": intrinsics_sources,
        "image_camera_rotation_degrees": float(image_camera_rotation_degrees),
        "camera_axis_diagnostics": camera_axis_diagnostics,
        "pose_binding": {
            "strictly_synchronized_count": int(
                sum(bool(poses[name].get("strictly_synchronized")) for name in frame_names)
            ),
            "selected_frame_count": int(len(frame_names)),
            "strictly_synchronized_fraction": float(
                sum(bool(poses[name].get("strictly_synchronized")) for name in frame_names)
                / max(len(frame_names), 1)
            ),
            "binding_by_frame": {
                name: str(poses[name].get("pose_binding") or "legacy_unversioned")
                for name in frame_names
            },
            "timestamp_delta_seconds_by_frame": {
                name: poses[name].get("camera_frame_timestamp_delta_s")
                for name in frame_names
            },
        },
        "coordinate_conversion": {
            "world": "diag(1,1,-1) * unity_world",
            "camera": "Unity camera converted to COLMAP x-right y-down z-forward",
            "cpu_image_camera_from_pose_camera": (
                f"Rz({float(image_camera_rotation_degrees):g}deg); "
                "resolved from saved CPU-image metadata; legacy captures use the "
                "explicit Rz(+90deg) contract"
            ),
        },
    }
    write_json(sparse / "phone_pose_meta.json", metadata)
    return sparse, metadata


def _copy_capture_inputs(
    data_dir: Path,
    mask_dir: Path,
    frame_names: Sequence[str],
    destination: Path,
    *,
    include_point_cloud: bool = True,
) -> None:
    for name in ("images", "rgb", "masks"):
        (destination / name).mkdir(parents=True, exist_ok=True)
    for frame_name in frame_names:
        source_image = data_dir / frame_name
        source_mask = mask_dir / f"{Path(frame_name).stem}.png"
        if not source_image.is_file():
            raise FileNotFoundError(f"selected image missing: {source_image}")
        if not source_mask.is_file():
            raise FileNotFoundError(f"selected mask missing: {source_mask}")
        shutil.copy2(source_image, destination / "images" / frame_name)
        shutil.copy2(source_image, destination / "rgb" / frame_name)
        shutil.copy2(source_mask, destination / "masks" / source_mask.name)
    shutil.copy2(data_dir / "poses.txt", destination / "poses.txt")
    points_path = data_dir / "slam_points.jsonl"
    if include_point_cloud:
        if not points_path.is_file():
            raise FileNotFoundError(f"AR point upload is missing: {points_path}")
        shutil.copy2(points_path, destination / "slam_points_raw.jsonl")
    elif points_path.is_file() and points_path.stat().st_size > 0:
        # Keep an uploaded cloud for diagnostics, but the pose+mask contract must
        # never consume it to construct O or to gate reconstruction.
        shutil.copy2(points_path, destination / "slam_points_diagnostic_only.jsonl")
    metadata = data_dir / "frame_metadata.jsonl"
    if metadata.is_file():
        shutil.copy2(metadata, destination / "frame_metadata.jsonl")


def finalize_ar_capture(
    *,
    session_id: str,
    data_dir: Path,
    mask_dir: Path,
    frame_names: Sequence[str],
    output_root: Path,
    input_qc: dict[str, Any],
    config: ARPointFilterConfig,
    require_point_cloud: bool = True,
) -> dict[str, Any]:
    """Create one immutable CoarseModel/runtime-compatible capture dataset."""

    config.validate()
    if len(frame_names) < 2:
        raise ValueError("at least two selected frames are required")
    points_path = data_dir / "slam_points.jsonl"
    point_cloud_available = points_path.is_file() and points_path.stat().st_size > 0
    if require_point_cloud and not point_cloud_available:
        message = (
            "ARPointCloudManager produced no uploaded points. Confirm that the Unity "
            "AR Point Cloud Manager is enabled and assigned to pointCloudManager."
        )
        write_json(
            data_dir / "capture_failure_report.json",
            {
                "format": CAPTURE_FORMAT,
                "created_at_utc": utc_now(),
                "session_id": session_id,
                "passed": False,
                "failures": [message],
                "config": asdict(config),
                "input_qc": input_qc,
            },
        )
        raise RuntimeError(message)
    destination = output_root / "datasets" / session_id
    complete_report = destination / "capture_report.json"
    if complete_report.is_file():
        report = json.loads(complete_report.read_text(encoding="utf-8"))
        expected_geometry = (
            "ar_point_mask" if require_point_cloud else "pose_mask_only"
        )
        if (
            report.get("passed") is True
            and report.get("format") == CAPTURE_FORMAT
            and report.get("geometry_mode", "ar_point_mask") == expected_geometry
        ):
            update_collection_manifest(output_root, report)
            return report
        raise RuntimeError(
            f"existing immutable capture has a different geometry contract: "
            f"{complete_report}"
        )
    if destination.exists():
        raise RuntimeError(f"partial capture dataset already exists: {destination}")
    staging = destination.parent / f".{session_id}.building"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        _copy_capture_inputs(
            data_dir,
            mask_dir,
            frame_names,
            staging,
            include_point_cloud=bool(require_point_cloud),
        )
        poses = read_phone_poses(staging / "poses.txt")
        if require_point_cloud:
            image_camera_rotation_degrees, views, camera_axis_diagnostics = (
                select_image_camera_rotation(
                    data_dir,
                    mask_dir,
                    frame_names,
                    poses=poses,
                )
            )
        else:
            image_camera_rotation_degrees, camera_axis_diagnostics = (
                resolve_image_camera_rotation_from_metadata(poses, frame_names)
            )
            views = []
        sparse, sparse_metadata = write_phone_sparse_model(
            staging,
            frame_names,
            poses,
            image_camera_rotation_degrees=image_camera_rotation_degrees,
            camera_axis_diagnostics=camera_axis_diagnostics,
            point_source=(
                "ar_foundation_arpointcloudmanager"
                if require_point_cloud
                else "none_pose_mask_only"
            ),
        )
        if require_point_cloud:
            fused, source_confidence, temporal, fusion_stats = fuse_ar_points(
                iter_ar_point_rows(points_path), config
            )
            filtered, confidence, colors, support_stats = filter_points_by_masks(
                fused, source_confidence, temporal, views, config
            )
            diagnostics = mask_frame_diagnostics(
                filtered,
                views,
                config,
                pose_mask_geometry=(
                    camera_axis_diagnostics
                    if camera_axis_diagnostics.get("available")
                    else None
                ),
            )
        else:
            filtered = np.empty((0, 3), dtype=np.float64)
            confidence = np.empty((0,), dtype=np.float64)
            colors = np.empty((0, 3), dtype=np.uint8)
            fusion_stats = {
                "point_cloud_consumed": False,
                "point_cloud_available_for_diagnostics": bool(point_cloud_available),
                "raw_sample_count": 0,
                "temporally_supported_voxel_count": 0,
            }
            support_stats = {
                "point_cloud_consumed": False,
                "mask_supported_point_count": 0,
                "policy": "bypassed_for_pose_mask_runtime",
            }
            diagnostics = {
                **camera_axis_diagnostics,
                "point_cloud_consumed": False,
                "point_to_mask_extent_ratio": None,
                "cross_view_geometry_role": "deferred_to_runtime_final8",
            }
        failures = []
        if (
            require_point_cloud
            and config.require_pose_mask_geometry
            and not diagnostics.get("available")
        ):
            failures.append(
                "pose+mask camera geometry unavailable: "
                f"{diagnostics.get('error', 'all camera-axis candidates failed')}"
            )
        if require_point_cloud and len(filtered) < config.min_object_points:
            failures.append(
                f"mask-supported AR point count {len(filtered)} < {config.min_object_points}"
            )
        if (
            require_point_cloud
            and diagnostics.get("available")
            and diagnostics["point_to_mask_extent_ratio"]
            > config.max_point_to_mask_extent_ratio
        ):
            failures.append(
                "AR point extent / mask extent "
                f"{diagnostics['point_to_mask_extent_ratio']:.3f} > "
                f"{config.max_point_to_mask_extent_ratio:.3f}"
            )
        if require_point_cloud and diagnostics.get("available") and (
            diagnostics["ray_residual_median_over_mask_extent"]
            > config.max_ray_residual_median_over_mask_extent
        ):
            failures.append(
                "mask-ray median residual / mask extent "
                f"{diagnostics['ray_residual_median_over_mask_extent']:.3f} > "
                f"{config.max_ray_residual_median_over_mask_extent:.3f}"
            )
        if require_point_cloud and diagnostics.get("available") and (
            diagnostics["ray_residual_p90_over_mask_extent"]
            > config.max_ray_residual_p90_over_mask_extent
        ):
            failures.append(
                "mask-ray p90 residual / mask extent "
                f"{diagnostics['ray_residual_p90_over_mask_extent']:.3f} > "
                f"{config.max_ray_residual_p90_over_mask_extent:.3f}"
            )
        if require_point_cloud and diagnostics.get("available") and (
            diagnostics["camera_roll_median_degrees"]
            > config.max_camera_roll_median_degrees
        ):
            failures.append(
                "camera roll median "
                f"{diagnostics['camera_roll_median_degrees']:.2f} deg > "
                f"{config.max_camera_roll_median_degrees:.2f} deg"
            )
        if require_point_cloud and diagnostics.get("available") and (
            diagnostics["orbit_gravity_agreement"]
            < config.min_orbit_gravity_agreement
        ):
            failures.append(
                "camera-orbit / Unity-gravity agreement "
                f"{diagnostics['orbit_gravity_agreement']:.3f} < "
                f"{config.min_orbit_gravity_agreement:.3f}"
            )
        synchronized_fraction = float(
            sparse_metadata["pose_binding"]["strictly_synchronized_fraction"]
        )
        if (
            require_point_cloud
            and synchronized_fraction < config.min_synchronized_frame_ratio
        ):
            failures.append(
                "strictly synchronized camera-frame fraction "
                f"{synchronized_fraction:.3f} < "
                f"{config.min_synchronized_frame_ratio:.3f}"
            )
        if failures:
            failure_report = {
                "format": CAPTURE_FORMAT,
                "created_at_utc": utc_now(),
                "session_id": session_id,
                "passed": False,
                "failures": failures,
                "config": asdict(config),
                "fusion": fusion_stats,
                "mask_support": support_stats,
                "geometry_diagnostics": diagnostics,
                "input_qc": input_qc,
            }
            write_json(data_dir / "capture_failure_report.json", failure_report)
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
            "created_at_utc": utc_now(),
            "session_id": session_id,
            "passed": True,
            "dataset_dir": str(destination.resolve()),
            "selected_frame_count": len(frame_names),
            "selected_frame_names": list(frame_names),
            "geometry_mode": (
                "ar_point_mask" if require_point_cloud else "pose_mask_only"
            ),
            "point_cloud_consumed": bool(require_point_cloud),
            "source": {
                "camera_pose": "ARFoundation camera transform",
                "intrinsics": "ARCameraManager.TryGetIntrinsics",
                "point_cloud": (
                    "ARPointCloudManager sparse visual-SLAM feature points"
                    if require_point_cloud
                    else "optional diagnostic only; not consumed"
                ),
                "mask": "user-supervised SAM2 video masks",
            },
            "config": asdict(config),
            "fusion": fusion_stats,
            "mask_support": support_stats,
            "geometry_diagnostics": diagnostics,
            "input_qc": input_qc,
            "sparse_metadata": sparse_metadata,
            "outputs": {
                "cameras": str((destination / "sparse/0/cameras.txt").resolve()),
                "images": str((destination / "sparse/0/images.txt").resolve()),
                "points3D": str((destination / "sparse/0/points3D.txt").resolve()),
                "point_ply": str((destination / "sparse/0/object_points.ply").resolve()),
                "point_npz": str((destination / "sparse/0/object_points.npz").resolve()),
                "raw_ar_points": (
                    str((destination / "slam_points_raw.jsonl").resolve())
                    if require_point_cloud
                    else None
                ),
                "diagnostic_ar_points": (
                    str((destination / "slam_points_diagnostic_only.jsonl").resolve())
                    if (not require_point_cloud and point_cloud_available)
                    else None
                ),
            },
            "scope_guard": (
                "Observable real capture only. Pose+mask mode consumes RGB, mask, "
                "AR pose/intrinsics and no point cloud; strict legacy mode additionally "
                "consumes the ARFoundation point cloud. No generated mesh or GT geometry "
                "is consumed."
            ),
        }
        write_json(staging / "capture_report.json", report)
        staging.replace(destination)
        update_collection_manifest(output_root, report)
        return report
    except Exception as exc:
        if staging.exists():
            shutil.rmtree(staging)
        failure_path = data_dir / "capture_failure_report.json"
        if not complete_report.is_file() and not failure_path.is_file():
            write_json(
                failure_path,
                {
                    "format": CAPTURE_FORMAT,
                    "created_at_utc": utc_now(),
                    "session_id": session_id,
                    "passed": False,
                    "failures": [repr(exc)],
                    "config": asdict(config),
                    "input_qc": input_qc,
                },
            )
        raise


def update_collection_manifest(output_root: Path, report: dict[str, Any]) -> None:
    path = output_root / "capture_manifest.json"
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("format") != COLLECTION_FORMAT:
            raise RuntimeError(f"unexpected collection manifest format: {path}")
    else:
        payload = {"format": COLLECTION_FORMAT, "objects": []}
    by_id = {str(row["session_id"]): row for row in payload.get("objects", [])}
    by_id[str(report["session_id"])] = {
        "session_id": report["session_id"],
        "dataset_dir": report["dataset_dir"],
        "capture_report": str(
            (Path(report["dataset_dir"]) / "capture_report.json").resolve()
        ),
        "selected_frame_count": report["selected_frame_count"],
        "point_count": report["mask_support"]["mask_supported_point_count"],
        "passed": True,
    }
    payload.update(
        {
            "updated_at_utc": utc_now(),
            "object_count": len(by_id),
            "objects": [by_id[key] for key in sorted(by_id)],
            "passed": bool(by_id),
        }
    )
    write_json(path, payload)
