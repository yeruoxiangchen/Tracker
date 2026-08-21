"""AR-capture input QC, frame selection, and lightweight mesh/mask sanity checks.

The routines here intentionally avoid importing Trellis. They are used by both
the Flask server and ReconViaGen worker before heavy model execution.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


IMAGE_EXTS = (".jpg", ".jpeg", ".png")


def _env_float(name: str, default: float) -> float:
    text = os.environ.get(name, "").strip()
    if not text:
        return default
    try:
        value = float(text)
    except ValueError:
        return default
    return value if math.isfinite(value) else default


def _env_int(name: str, default: int) -> int:
    text = os.environ.get(name, "").strip()
    if not text:
        return default
    try:
        return int(text)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    text = os.environ.get(name, "").strip().lower()
    if not text:
        return default
    return text not in {"0", "false", "no", "off"}


@dataclass(frozen=True)
class QCThresholds:
    max_frames: int = 18
    min_frames: int = 8
    min_pose_frames: int = 6
    min_azimuth_span: float = 160.0
    good_azimuth_span: float = 220.0
    min_elevation_range: float = 12.0
    min_mask_coverage: float = 0.08
    max_mask_coverage: float = 0.78
    max_border_touch_ratio: float = 0.18
    max_ray_residual_median_over_mask_extent: float = 0.20
    max_ray_residual_p90_over_mask_extent: float = 0.40
    max_camera_roll_median_degrees: float = 12.5
    min_orbit_gravity_agreement: float = 0.80
    min_synchronized_frame_ratio: float = 0.90

    @classmethod
    def from_env(cls) -> "QCThresholds":
        return cls(
            max_frames=max(1, _env_int("RECON_MAX_FRAMES", cls.max_frames)),
            min_frames=max(1, _env_int("RECON_MIN_FRAMES", cls.min_frames)),
            min_pose_frames=max(1, _env_int("RECON_MIN_POSE_FRAMES", cls.min_pose_frames)),
            min_azimuth_span=_env_float("RECON_QC_MIN_AZIMUTH_SPAN", cls.min_azimuth_span),
            good_azimuth_span=_env_float("RECON_QC_GOOD_AZIMUTH_SPAN", cls.good_azimuth_span),
            min_elevation_range=_env_float("RECON_QC_MIN_ELEVATION_RANGE", cls.min_elevation_range),
            min_mask_coverage=_env_float("RECON_QC_MIN_MASK_COVERAGE", cls.min_mask_coverage),
            max_mask_coverage=_env_float("RECON_QC_MAX_MASK_COVERAGE", cls.max_mask_coverage),
            max_border_touch_ratio=_env_float("RECON_QC_MAX_BORDER_TOUCH_RATIO", cls.max_border_touch_ratio),
            max_ray_residual_median_over_mask_extent=_env_float(
                "RECON_QC_MAX_RAY_RESIDUAL_MEDIAN_OVER_MASK_EXTENT",
                cls.max_ray_residual_median_over_mask_extent,
            ),
            max_ray_residual_p90_over_mask_extent=_env_float(
                "RECON_QC_MAX_RAY_RESIDUAL_P90_OVER_MASK_EXTENT",
                cls.max_ray_residual_p90_over_mask_extent,
            ),
            max_camera_roll_median_degrees=_env_float(
                "RECON_QC_MAX_CAMERA_ROLL_MEDIAN_DEGREES",
                cls.max_camera_roll_median_degrees,
            ),
            min_orbit_gravity_agreement=_env_float(
                "RECON_QC_MIN_ORBIT_GRAVITY_AGREEMENT",
                cls.min_orbit_gravity_agreement,
            ),
            min_synchronized_frame_ratio=_env_float(
                "RECON_QC_MIN_SYNCHRONIZED_FRAME_RATIO",
                cls.min_synchronized_frame_ratio,
            ),
        )


def enforce_input_qc_from_env() -> bool:
    return _env_bool("RECON_ENFORCE_INPUT_QC", True)


def pose_rerank_enabled_from_env() -> bool:
    return _env_bool("RECON_POSE_RERANK", True)


def pose_rerank_weight_from_env() -> float:
    return max(0.0, _env_float("RECON_POSE_RERANK_WEIGHT", 0.45))


def parse_pose_file(data_dir: str) -> Dict[str, Dict[str, Any]]:
    pose_path = os.path.join(data_dir, "poses.txt")
    poses: Dict[str, Dict[str, Any]] = {}
    if not os.path.exists(pose_path):
        return poses

    with open(pose_path, "r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 7 or parts[0] == "frame_name":
                continue
            try:
                pos = np.array([float(x) for x in parts[1:4]], dtype=np.float64)
                rot = np.array([float(x) for x in parts[4:7]], dtype=np.float64)
            except ValueError:
                continue
            pose: Dict[str, Any] = {
                "frame_name": parts[0],
                "pos": pos,
                "rot": rot,
                "rot_deg": rot,
                "pose_binding": "legacy_unversioned",
                "camera_frame_timestamp_delta_s": None,
                "strictly_synchronized": False,
            }
            if len(parts) >= 11:
                try:
                    pose["quat"] = np.array([float(x) for x in parts[7:11]], dtype=np.float64)
                except ValueError:
                    pose["quat"] = None
            if len(parts) >= 15:
                try:
                    pose["intrinsics"] = {
                        "fx": float(parts[11]),
                        "fy": float(parts[12]),
                        "cx": float(parts[13]),
                        "cy": float(parts[14]),
                        "width": int(float(parts[15])) if len(parts) > 15 and parts[15] else None,
                        "height": int(float(parts[16])) if len(parts) > 16 and parts[16] else None,
                        "image_width": int(float(parts[17])) if len(parts) > 17 and parts[17] else None,
                        "image_height": int(float(parts[18])) if len(parts) > 18 and parts[18] else None,
                    }
                except ValueError:
                    pose["intrinsics"] = None
            pose["image_transform"] = parts[21] if len(parts) > 21 else "None"
            if len(parts) > 25 and parts[25]:
                try:
                    delta = float(parts[25])
                    pose["camera_frame_timestamp_delta_s"] = (
                        delta if math.isfinite(delta) else None
                    )
                except ValueError:
                    pass
            if len(parts) > 26 and parts[26]:
                pose["pose_binding"] = parts[26]
            delta = pose["camera_frame_timestamp_delta_s"]
            pose["strictly_synchronized"] = bool(
                pose["pose_binding"] == "camera_frame_received"
                and delta is not None
                and 0.0 <= delta <= 0.05
            )
            poses[parts[0]] = pose
    return poses


def circular_span_degrees(angles: Sequence[float]) -> float:
    if len(angles) < 2:
        return 0.0
    values = sorted(float(a) % 360.0 for a in angles)
    gaps = [values[i + 1] - values[i] for i in range(len(values) - 1)]
    gaps.append(values[0] + 360.0 - values[-1])
    return float(360.0 - max(gaps))


def pose_coverage_stats(frame_names: Sequence[str], poses: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    pose_items = [(name, poses[name]["pos"]) for name in frame_names if name in poses]
    if not pose_items:
        return {
            "pose_count": 0,
            "azimuth_span_deg": 0.0,
            "azimuth_centered_span_deg": 0.0,
            "elevation_range_deg": 0.0,
            "distance_mean": None,
            "distance_min": None,
            "distance_max": None,
            "z_min": None,
            "z_max": None,
        }

    positions = np.stack([np.asarray(p, dtype=np.float64) for _, p in pose_items], axis=0)
    distances = np.linalg.norm(positions, axis=1)
    azimuth = np.degrees(np.arctan2(positions[:, 0], positions[:, 2])) % 360.0
    centered = positions - positions.mean(axis=0, keepdims=True)
    azimuth_centered = np.degrees(np.arctan2(centered[:, 0], centered[:, 2])) % 360.0
    horiz = np.linalg.norm(positions[:, [0, 2]], axis=1)
    elevation = np.degrees(np.arctan2(positions[:, 1], np.maximum(horiz, 1e-9)))

    return {
        "pose_count": int(len(pose_items)),
        "azimuth_span_deg": circular_span_degrees(azimuth.tolist()),
        "azimuth_centered_span_deg": circular_span_degrees(azimuth_centered.tolist()),
        "azimuth_sorted_deg": [float(x) for x in sorted(azimuth.tolist())],
        "elevation_mean_deg": float(np.mean(elevation)),
        "elevation_min_deg": float(np.min(elevation)),
        "elevation_max_deg": float(np.max(elevation)),
        "elevation_range_deg": float(np.max(elevation) - np.min(elevation)),
        "distance_mean": float(np.mean(distances)),
        "distance_min": float(np.min(distances)),
        "distance_max": float(np.max(distances)),
        "z_min": float(np.min(positions[:, 2])),
        "z_max": float(np.max(positions[:, 2])),
        "position_min": [float(x) for x in positions.min(axis=0).tolist()],
        "position_max": [float(x) for x in positions.max(axis=0).tolist()],
    }


def _unity_quat_to_rotmat(quat_xyzw: Sequence[float]) -> np.ndarray:
    q = np.asarray(quat_xyzw, dtype=np.float64)
    norm = np.linalg.norm(q)
    if norm < 1e-12:
        raise ValueError("zero quaternion")
    x, y, z, w = q / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _pose_rotmat(pose: Dict[str, Any]) -> Optional[np.ndarray]:
    quat = pose.get("quat")
    if quat is not None:
        try:
            return _unity_quat_to_rotmat(quat)
        except Exception:
            pass
    rot = pose.get("rot")
    if rot is None:
        return None
    return _unity_euler_to_rotmat(rot)


def _full_mask_path(data_dir: str, preview_dir: str, frame_name: str) -> Optional[str]:
    stem = os.path.splitext(frame_name)[0]
    candidates = [
        os.path.join(data_dir, "masks", f"{stem}.png"),
        os.path.join(os.path.dirname(data_dir), "masks", os.path.basename(data_dir), f"{stem}.png"),
        os.path.join(os.path.dirname(os.path.dirname(data_dir)), "masks", os.path.basename(data_dir), f"{stem}.png"),
    ]
    if preview_dir:
        candidates.append(os.path.join(os.path.dirname(os.path.dirname(preview_dir)), "masks", os.path.basename(preview_dir), f"{stem}.png"))
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def _mask_center(mask_path: str) -> Optional[Tuple[float, float, int, int]]:
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    fg = mask > 0
    if not np.any(fg):
        return None
    ys, xs = np.nonzero(fg)
    h, w = fg.shape[:2]
    return float(xs.mean()), float(ys.mean()), int(w), int(h)


def _intrinsics_for_pose(pose: Dict[str, Any], width: int, height: int) -> Tuple[float, float, float, float]:
    intr = pose.get("intrinsics") or {}
    fx = float(intr.get("fx") or max(width, height))
    fy = float(intr.get("fy") or max(width, height))
    cx = float(intr.get("cx") if intr.get("cx") is not None else (width - 1) * 0.5)
    cy = float(intr.get("cy") if intr.get("cy") is not None else (height - 1) * 0.5)
    src_w = intr.get("image_width") or intr.get("width")
    src_h = intr.get("image_height") or intr.get("height")
    if src_w and src_h and int(src_w) > 0 and int(src_h) > 0 and (int(src_w) != width or int(src_h) != height):
        sx = float(width) / float(src_w)
        sy = float(height) / float(src_h)
        fx *= sx
        fy *= sy
        cx *= sx
        cy *= sy
    transform = str(pose.get("image_transform") or "None").lower()
    if "mirrorx" in transform:
        cx = (width - 1) - cx
    if "mirrory" in transform:
        cy = (height - 1) - cy
    return fx, fy, cx, cy


def object_centered_coverage_stats(
    frame_names: Sequence[str],
    poses: Dict[str, Dict[str, Any]],
    data_dir: str,
    preview_dir: str,
) -> Dict[str, Any]:
    mask_paths = [_full_mask_path(data_dir, preview_dir, name) for name in frame_names]
    if frame_names and all(path is not None for path in mask_paths):
        mask_parents = {os.path.dirname(str(path)) for path in mask_paths if path}
        if len(mask_parents) == 1:
            try:
                from pathlib import Path

                from pose_point_depth_mv.ar_object_capture import (
                    select_image_camera_rotation,
                )
                from trellis_point_prior_mv.build_ar_session_smoke_dataset import (
                    read_phone_poses,
                )

                parsed = read_phone_poses(Path(data_dir) / "poses.txt")
                angle, _views, diagnostics = select_image_camera_rotation(
                    Path(data_dir),
                    Path(next(iter(mask_parents))),
                    frame_names,
                    poses=parsed,
                )
                if diagnostics.get("available"):
                    synchronized = [
                        bool(parsed[name].get("strictly_synchronized"))
                        for name in frame_names
                        if name in parsed
                    ]
                    return {
                        "enabled": True,
                        "reason": "shared_saved_cpu_image_camera_axis_contract_v4",
                        "ray_count": int(len(frame_names)),
                        "used_names": list(frame_names),
                        "object_center": diagnostics["mask_ray_center_W"],
                        "azimuth_span_deg": diagnostics["azimuth_span_deg"],
                        "azimuth_by_name": diagnostics["azimuth_by_name"],
                        "elevation_range_deg": diagnostics["elevation_range_deg"],
                        "image_camera_rotation_degrees": float(angle),
                        "ray_residual_median_over_mask_extent": diagnostics[
                            "ray_residual_median_over_mask_extent"
                        ],
                        "ray_residual_p90_over_mask_extent": diagnostics[
                            "ray_residual_p90_over_mask_extent"
                        ],
                        "ray_residual_max_over_mask_extent": diagnostics[
                            "ray_residual_max_over_mask_extent"
                        ],
                        "ray_residual_over_mask_extent_by_name": diagnostics[
                            "ray_residual_over_mask_extent_by_name"
                        ],
                        "orbit_gravity_agreement": diagnostics[
                            "orbit_gravity_agreement"
                        ],
                        "camera_roll_median_degrees": diagnostics[
                            "camera_roll_median_degrees"
                        ],
                        "camera_roll_p90_degrees": diagnostics[
                            "camera_roll_p90_degrees"
                        ],
                        "camera_roll_max_degrees": diagnostics[
                            "camera_roll_max_degrees"
                        ],
                        "camera_roll_degrees_by_name": diagnostics[
                            "camera_roll_degrees_by_name"
                        ],
                        "saved_image_axis_contract": diagnostics[
                            "saved_image_axis_contract"
                        ],
                        "strictly_synchronized_fraction": float(
                            sum(synchronized) / max(len(synchronized), 1)
                        ),
                        "camera_axis_candidates": diagnostics["candidates"],
                    }
            except Exception as exc:
                shared_contract_error = repr(exc)
            else:
                shared_contract_error = "shared camera-axis diagnostics unavailable"
        else:
            shared_contract_error = "selected masks do not share one directory"
    else:
        shared_contract_error = "selected full-resolution masks are missing"

    centers: List[np.ndarray] = []
    rays: List[np.ndarray] = []
    used_names: List[str] = []
    for name in frame_names:
        pose = poses.get(name)
        if pose is None:
            continue
        mask_path = _full_mask_path(data_dir, preview_dir, name)
        if not mask_path:
            continue
        center = _mask_center(mask_path)
        if center is None:
            continue
        u, v, width, height = center
        rot = _pose_rotmat(pose)
        if rot is None:
            continue
        fx, fy, cx, cy = _intrinsics_for_pose(pose, width, height)
        if fx <= 0 or fy <= 0:
            continue
        ray_cam = np.array([(u - cx) / fx, -(v - cy) / fy, 1.0], dtype=np.float64)
        ray_cam /= max(float(np.linalg.norm(ray_cam)), 1e-12)
        ray_world = rot @ ray_cam
        ray_world /= max(float(np.linalg.norm(ray_world)), 1e-12)
        centers.append(np.asarray(pose["pos"], dtype=np.float64))
        rays.append(ray_world)
        used_names.append(name)

    if len(rays) < 3:
        return {
            "enabled": False,
            "reason": "too_few_mask_pose_rays",
            "ray_count": len(rays),
            "shared_contract_error": shared_contract_error,
        }

    A = np.zeros((3, 3), dtype=np.float64)
    b = np.zeros(3, dtype=np.float64)
    for cam_center, ray in zip(centers, rays):
        proj = np.eye(3, dtype=np.float64) - np.outer(ray, ray)
        A += proj
        b += proj @ cam_center
    try:
        object_center = np.linalg.lstsq(A, b, rcond=None)[0]
    except np.linalg.LinAlgError:
        return {"enabled": False, "reason": "object_center_lstsq_failed", "ray_count": len(rays)}

    positions = np.stack(centers, axis=0)
    rel = positions - object_center[None]
    horiz = np.linalg.norm(rel[:, [0, 2]], axis=1)
    azimuth = np.degrees(np.arctan2(rel[:, 0], rel[:, 2])) % 360.0
    elevation = np.degrees(np.arctan2(rel[:, 1], np.maximum(horiz, 1e-9)))
    distances = np.linalg.norm(rel, axis=1)
    residuals = [float(np.linalg.norm(np.cross(object_center - c, d))) for c, d in zip(centers, rays)]
    return {
        "enabled": True,
        "reason": "legacy_mask_center_ray_object_center_fallback",
        "shared_contract_error": shared_contract_error,
        "ray_count": int(len(rays)),
        "used_names": used_names,
        "object_center": [float(x) for x in object_center.tolist()],
        "azimuth_span_deg": circular_span_degrees(azimuth.tolist()),
        "azimuth_by_name": {name: float(value) for name, value in zip(used_names, azimuth.tolist())},
        "azimuth_sorted_deg": [float(x) for x in sorted(azimuth.tolist())],
        "elevation_range_deg": float(np.max(elevation) - np.min(elevation)),
        "distance_mean": float(np.mean(distances)),
        "distance_min": float(np.min(distances)),
        "distance_max": float(np.max(distances)),
        "ray_residual_mean": float(np.mean(residuals)),
        "ray_residual_p90": float(np.percentile(residuals, 90)),
    }


def mask_stats_from_preview(preview_path: str) -> Dict[str, Any]:
    preview = cv2.imread(preview_path, cv2.IMREAD_UNCHANGED)
    stats: Dict[str, Any] = {
        "coverage": 0.0,
        "bbox": None,
        "bbox_coverage": 0.0,
        "bbox_aspect": 0.0,
        "border_touch_ratio": 1.0,
        "empty_mask": True,
    }
    if preview is None:
        stats["missing_preview"] = True
        return stats
    if preview.ndim == 3 and preview.shape[2] == 4:
        mask = preview[:, :, 3] > 128
    elif preview.ndim == 3:
        mask = cv2.cvtColor(preview, cv2.COLOR_BGR2GRAY) > 0
    else:
        mask = preview > 0
    if not np.any(mask):
        return stats

    h, w = mask.shape[:2]
    ys, xs = np.nonzero(mask)
    left, right = int(xs.min()), int(xs.max())
    top, bottom = int(ys.min()), int(ys.max())
    bbox_w = max(1, right - left + 1)
    bbox_h = max(1, bottom - top + 1)
    border_margin = max(2, int(round(min(w, h) * 0.03)))
    touches = int(left <= border_margin) + int(top <= border_margin) + int(right >= w - 1 - border_margin) + int(bottom >= h - 1 - border_margin)
    stats.update(
        {
            "coverage": float(mask.mean()),
            "bbox": [left, top, right, bottom],
            "bbox_coverage": float(mask.sum() / float(bbox_w * bbox_h)),
            "bbox_aspect": float(bbox_w / float(bbox_h)),
            "border_touch_ratio": float(touches / 4.0),
            "empty_mask": False,
        }
    )
    return stats


def frame_quality_score(frame_index: int, image_names: Sequence[str], data_dir: str, preview_dir: str) -> Tuple[float, Dict[str, Any]]:
    if frame_index < 0 or frame_index >= len(image_names):
        return 0.0, {"invalid_index": True}

    preview_path = os.path.join(preview_dir, f"{frame_index}.png")
    mask_stats = mask_stats_from_preview(preview_path)
    coverage = float(mask_stats.get("coverage") or 0.0)
    bbox_aspect = float(mask_stats.get("bbox_aspect") or 0.0)
    border_touch = float(mask_stats.get("border_touch_ratio") or 0.0)

    coverage_score = max(0.0, 1.0 - abs(coverage - 0.45) / 0.45)
    aspect_score = 0.5
    if bbox_aspect > 0:
        aspect_score = float(np.clip(1.0 - abs(math.log(max(bbox_aspect, 1e-6))) / math.log(3.0), 0.0, 1.0))
    border_score = float(np.clip(1.0 - border_touch, 0.0, 1.0))

    sharpness_score = 0.5
    image_path = os.path.join(data_dir, image_names[frame_index])
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    sharpness = None
    if img is not None:
        sharpness = float(cv2.Laplacian(img, cv2.CV_64F).var())
        sharpness_score = min(sharpness / 500.0, 1.0)

    score = 0.45 * coverage_score + 0.25 * sharpness_score + 0.20 * border_score + 0.10 * aspect_score
    details = {
        "quality_score": float(score),
        "sharpness": sharpness,
        "coverage_score": float(coverage_score),
        "sharpness_score": float(sharpness_score),
        "border_score": float(border_score),
        "aspect_score": float(aspect_score),
        **mask_stats,
    }
    return float(score), details


def _azimuth_for_index(idx: int, image_names: Sequence[str], poses: Dict[str, Dict[str, Any]]) -> Optional[float]:
    if idx < 0 or idx >= len(image_names):
        return None
    pose = poses.get(image_names[idx])
    if pose is None:
        return None
    pos = np.asarray(pose["pos"], dtype=np.float64)
    return float(np.degrees(np.arctan2(pos[0], pos[2])) % 360.0)


def _azimuth_map_from_object_coverage(
    indices: Sequence[int],
    image_names: Sequence[str],
    poses: Dict[str, Dict[str, Any]],
    data_dir: str,
    preview_dir: str,
) -> Tuple[Dict[int, float], Dict[str, Any]]:
    names = [image_names[i] for i in indices if 0 <= i < len(image_names)]
    object_coverage = object_centered_coverage_stats(names, poses, data_dir, preview_dir)
    name_to_azimuth = object_coverage.get("azimuth_by_name") if object_coverage.get("enabled") else None
    if not isinstance(name_to_azimuth, dict):
        return {}, object_coverage
    out: Dict[int, float] = {}
    for idx in indices:
        if 0 <= idx < len(image_names) and image_names[idx] in name_to_azimuth:
            out[int(idx)] = float(name_to_azimuth[image_names[idx]]) % 360.0
    return out, object_coverage


def select_frames_for_reconstruction(
    selected_indices: Sequence[int],
    image_names: Sequence[str],
    data_dir: str,
    preview_dir: str,
    thresholds: Optional[QCThresholds] = None,
) -> Tuple[List[int], Dict[str, Any]]:
    thresholds = thresholds or QCThresholds.from_env()
    valid = sorted({int(i) for i in selected_indices if 0 <= int(i) < len(image_names)})
    poses = parse_pose_file(data_dir)
    quality_by_idx: Dict[int, Dict[str, Any]] = {}
    score_by_idx: Dict[int, float] = {}
    for idx in valid:
        score, details = frame_quality_score(idx, image_names, data_dir, preview_dir)
        score_by_idx[idx] = score
        quality_by_idx[idx] = details

    usable = [
        idx
        for idx in valid
        if not quality_by_idx[idx].get("empty_mask")
        and thresholds.min_mask_coverage <= float(quality_by_idx[idx].get("coverage") or 0.0) <= thresholds.max_mask_coverage
        and float(quality_by_idx[idx].get("border_touch_ratio") or 0.0) <= thresholds.max_border_touch_ratio
    ]
    if len(usable) < max(thresholds.min_frames, min(len(valid), thresholds.min_frames)):
        usable = valid

    chosen: List[int]
    if len(usable) <= thresholds.max_frames:
        chosen = list(usable)
        selection_reason = "under_limit" if len(valid) <= thresholds.max_frames else "quality_filtered_under_limit"
    else:
        object_azimuth_by_idx, selection_object_coverage = _azimuth_map_from_object_coverage(
            usable,
            image_names,
            poses,
            data_dir,
            preview_dir,
        )
        posed = [
            idx
            for idx in usable
            if idx in object_azimuth_by_idx or _azimuth_for_index(idx, image_names, poses) is not None
        ]
        if len(posed) < max(6, thresholds.max_frames // 2):
            chosen = sorted(sorted(usable, key=lambda i: score_by_idx.get(i, 0.0), reverse=True)[: thresholds.max_frames])
            selection_reason = "quality_only_no_pose"
        else:
            bin_count = min(thresholds.max_frames, max(6, int(round(thresholds.good_azimuth_span / 20.0))))
            bin_size = 360.0 / float(bin_count)
            bins: Dict[int, Tuple[int, float]] = {}
            for idx in posed:
                azimuth = object_azimuth_by_idx.get(idx)
                if azimuth is None:
                    azimuth = _azimuth_for_index(idx, image_names, poses)
                if azimuth is None:
                    continue
                bin_id = int(azimuth // bin_size)
                q = score_by_idx.get(idx, 0.0)
                current = bins.get(bin_id)
                if current is None or q > current[1]:
                    bins[bin_id] = (idx, q)
            chosen = [item[0] for item in bins.values()]
            if len(chosen) < thresholds.max_frames:
                remaining = [idx for idx in usable if idx not in chosen]
                remaining = sorted(remaining, key=lambda i: score_by_idx.get(i, 0.0), reverse=True)
                chosen.extend(remaining[: thresholds.max_frames - len(chosen)])
            elif len(chosen) > thresholds.max_frames:
                chosen = sorted(chosen, key=lambda i: score_by_idx.get(i, 0.0), reverse=True)[: thresholds.max_frames]
            chosen = sorted(chosen)
            selection_reason = (
                "object_centered_pose_azimuth_balanced_quality"
                if object_azimuth_by_idx
                else "pose_azimuth_balanced_quality"
            )

    chosen_names = [image_names[i] for i in chosen if 0 <= i < len(image_names)]
    coverage = pose_coverage_stats(chosen_names, poses)
    object_coverage = object_centered_coverage_stats(chosen_names, poses, data_dir, preview_dir)
    if object_coverage.get("enabled"):
        coverage["object_centered"] = object_coverage
        coverage["azimuth_effective_span_deg"] = object_coverage["azimuth_span_deg"]
        coverage["elevation_effective_range_deg"] = object_coverage["elevation_range_deg"]
        coverage["effective_coverage_source"] = object_coverage["reason"]
    else:
        coverage["object_centered"] = object_coverage
        coverage["azimuth_effective_span_deg"] = coverage["azimuth_span_deg"]
        coverage["elevation_effective_range_deg"] = coverage["elevation_range_deg"]
        coverage["effective_coverage_source"] = "world_origin_pose"
    warnings: List[str] = []
    fail_reasons: List[str] = []
    if len(chosen) < thresholds.min_frames:
        fail_reasons.append(f"selected frame count {len(chosen)} < {thresholds.min_frames}")
    if coverage["pose_count"] < thresholds.min_pose_frames:
        warnings.append(f"pose count {coverage['pose_count']} < {thresholds.min_pose_frames}; pose coverage cannot be trusted")
    else:
        effective_azimuth_span = float(coverage.get("azimuth_effective_span_deg") or 0.0)
        effective_elevation_range = float(coverage.get("elevation_effective_range_deg") or 0.0)
        if effective_azimuth_span < thresholds.min_azimuth_span:
            fail_reasons.append(
                f"azimuth span {effective_azimuth_span:.1f} < {thresholds.min_azimuth_span:.1f} deg "
                f"({coverage.get('effective_coverage_source')})"
            )
        elif effective_azimuth_span < thresholds.good_azimuth_span:
            warnings.append(
                f"azimuth span {effective_azimuth_span:.1f} is usable but below target "
                f"{thresholds.good_azimuth_span:.1f} deg ({coverage.get('effective_coverage_source')})"
            )
        if effective_elevation_range < thresholds.min_elevation_range:
            warnings.append(
                f"elevation range {effective_elevation_range:.1f} < {thresholds.min_elevation_range:.1f} deg"
            )
        if object_coverage.get("reason") == "shared_saved_cpu_image_camera_axis_contract_v4":
            median_residual = float(
                object_coverage["ray_residual_median_over_mask_extent"]
            )
            p90_residual = float(
                object_coverage["ray_residual_p90_over_mask_extent"]
            )
            gravity_agreement = float(object_coverage["orbit_gravity_agreement"])
            camera_roll_median = float(
                object_coverage["camera_roll_median_degrees"]
            )
            synchronized_fraction = float(
                object_coverage["strictly_synchronized_fraction"]
            )
            if median_residual > thresholds.max_ray_residual_median_over_mask_extent:
                fail_reasons.append(
                    "mask-ray median residual / mask extent "
                    f"{median_residual:.3f} > "
                    f"{thresholds.max_ray_residual_median_over_mask_extent:.3f}"
                )
            if p90_residual > thresholds.max_ray_residual_p90_over_mask_extent:
                fail_reasons.append(
                    "mask-ray p90 residual / mask extent "
                    f"{p90_residual:.3f} > "
                    f"{thresholds.max_ray_residual_p90_over_mask_extent:.3f}"
                )
            if gravity_agreement < thresholds.min_orbit_gravity_agreement:
                fail_reasons.append(
                    "camera-orbit / Unity-gravity agreement "
                    f"{gravity_agreement:.3f} < "
                    f"{thresholds.min_orbit_gravity_agreement:.3f}"
                )
            if camera_roll_median > thresholds.max_camera_roll_median_degrees:
                fail_reasons.append(
                    "camera roll median "
                    f"{camera_roll_median:.2f} deg > "
                    f"{thresholds.max_camera_roll_median_degrees:.2f} deg"
                )
            if synchronized_fraction < thresholds.min_synchronized_frame_ratio:
                fail_reasons.append(
                    "strictly synchronized camera-frame fraction "
                    f"{synchronized_fraction:.3f} < "
                    f"{thresholds.min_synchronized_frame_ratio:.3f}"
                )
        elif object_coverage.get("enabled"):
            fail_reasons.append(
                "camera/image pose contract is legacy or unavailable; recollect with AR client v2"
            )

    report = {
        "enabled": True,
        "reason": selection_reason,
        "input_count": int(len(valid)),
        "output_count": int(len(chosen)),
        "max_frames": int(thresholds.max_frames),
        "selected_indices": [int(i) for i in chosen],
        "selected_names": chosen_names,
        "coverage": coverage,
        "thresholds": thresholds.__dict__,
        "warnings": warnings,
        "fail_reasons": fail_reasons,
        "qc_pass": len(fail_reasons) == 0,
        "frame_quality": {
            str(int(idx)): {
                "frame_name": image_names[idx],
                **quality_by_idx.get(idx, {}),
            }
            for idx in valid
        },
    }
    return [int(i) for i in chosen], report


def _mask_from_rgba_image(image: Any) -> np.ndarray:
    rgba = np.asarray(image.convert("RGBA"))
    return rgba[:, :, 3] > 128


def _unity_euler_to_rotmat(rot_deg: Sequence[float]) -> np.ndarray:
    rx, ry, rz = np.deg2rad(np.asarray(rot_deg, dtype=np.float64))
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    rx_mat = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float64)
    ry_mat = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)
    rz_mat = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float64)
    return ry_mat @ rx_mat @ rz_mat


def normalized_w2c_matrices(
    poses: Dict[str, Dict[str, Any]],
    frame_names: Sequence[str],
    resolution: int,
    fov_degrees: float = 60.0,
) -> Tuple[List[np.ndarray], np.ndarray]:
    pose_rows = [poses[name] for name in frame_names if name in poses]
    if not pose_rows:
        return [], np.eye(3, dtype=np.float64)
    positions = np.stack([np.asarray(p["pos"], dtype=np.float64) for p in pose_rows], axis=0)
    centered = positions - positions.mean(axis=0, keepdims=True)
    avg_radius = float(np.mean(np.linalg.norm(centered, axis=1)))
    scale = 1.5 / max(avg_radius, 1e-6)

    focal = (resolution / 2.0) / math.tan(math.radians(fov_degrees / 2.0))
    k = np.array([[focal, 0, resolution / 2.0], [0, focal, resolution / 2.0], [0, 0, 1]], dtype=np.float64)

    matrices: List[np.ndarray] = []
    for pose, centered_pos in zip(pose_rows, centered):
        pos = centered_pos * scale
        pos_rhs = np.array([pos[0], pos[1], -pos[2]], dtype=np.float64)
        rot_matrix = _unity_euler_to_rotmat(pose.get("rot", [0.0, 0.0, 0.0]))
        rot_matrix[:, 2] = -rot_matrix[:, 2]
        rot_matrix[2, :] = -rot_matrix[2, :]
        c2w = np.eye(4, dtype=np.float64)
        c2w[:3, :3] = rot_matrix
        c2w[:3, 3] = pos_rhs
        matrices.append(np.linalg.inv(c2w))
    return matrices, k


def _project_points_mask(points: np.ndarray, w2c: np.ndarray, k: np.ndarray, resolution: int) -> np.ndarray:
    if points.size == 0:
        return np.zeros((resolution, resolution), dtype=np.uint8)
    homo = np.concatenate([points, np.ones((len(points), 1), dtype=np.float64)], axis=1)
    cam = (w2c @ homo.T).T[:, :3]
    z = cam[:, 2]
    valid = z > 0.05
    if not np.any(valid):
        return np.zeros((resolution, resolution), dtype=np.uint8)
    cam = cam[valid]
    proj = (k @ cam.T).T
    uv = proj[:, :2] / np.maximum(proj[:, 2:3], 1e-8)
    xy = np.rint(uv).astype(np.int32)
    keep = (xy[:, 0] >= 0) & (xy[:, 0] < resolution) & (xy[:, 1] >= 0) & (xy[:, 1] < resolution)
    xy = xy[keep]
    mask = np.zeros((resolution, resolution), dtype=np.uint8)
    if len(xy) == 0:
        return mask
    mask[xy[:, 1], xy[:, 0]] = 255
    kernel = np.ones((9, 9), dtype=np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((17, 17), dtype=np.uint8), iterations=1)
    return mask


def pose_mask_sanity_for_mesh(
    mesh: Any,
    selected_images: Sequence[Any],
    selected_names: Sequence[str],
    data_dir: str,
    max_points: int = 30000,
) -> Dict[str, Any]:
    poses = parse_pose_file(data_dir)
    if not poses:
        return {"enabled": False, "reason": "no_pose_file", "score": None}
    frame_names = [name for name in selected_names if name in poses]
    if len(frame_names) < 3:
        return {"enabled": False, "reason": "too_few_pose_frames", "score": None}
    resolution = int(selected_images[0].width) if selected_images else 518
    w2c_list, k = normalized_w2c_matrices(poses, frame_names, resolution=resolution)
    if len(w2c_list) != len(frame_names):
        return {"enabled": False, "reason": "pose_matrix_build_failed", "score": None}

    try:
        verts = mesh.vertices.detach().float().cpu().numpy().astype(np.float64)
    except Exception as exc:
        return {"enabled": False, "reason": f"mesh_vertices_unavailable: {exc}", "score": None}
    if len(verts) == 0:
        return {"enabled": False, "reason": "empty_mesh", "score": None}
    if len(verts) > max_points:
        rng = np.random.default_rng(12345)
        verts = verts[rng.choice(len(verts), size=max_points, replace=False)]

    center = 0.5 * (verts.min(axis=0) + verts.max(axis=0))
    extent = float(np.max(verts.max(axis=0) - verts.min(axis=0)))
    points = (verts - center[None]) / max(extent, 1e-6)

    image_by_name = {name: image for name, image in zip(selected_names, selected_images)}
    records = []
    for name, w2c in zip(frame_names, w2c_list):
        target = _mask_from_rgba_image(image_by_name[name])
        pred = _project_points_mask(points, w2c, k, resolution=resolution) > 0
        inter = float(np.logical_and(pred, target).sum())
        union = float(np.logical_or(pred, target).sum())
        pred_sum = float(pred.sum())
        target_sum = float(target.sum())
        iou = inter / union if union > 0 else 0.0
        recall = inter / target_sum if target_sum > 0 else 0.0
        precision = inter / pred_sum if pred_sum > 0 else 0.0
        records.append(
            {
                "frame_name": name,
                "iou": float(iou),
                "recall": float(recall),
                "precision": float(precision),
                "pred_coverage": float(pred.mean()),
                "target_coverage": float(target.mean()),
            }
        )

    if not records:
        return {"enabled": False, "reason": "no_records", "score": None}
    ious = np.array([r["iou"] for r in records], dtype=np.float64)
    recalls = np.array([r["recall"] for r in records], dtype=np.float64)
    precisions = np.array([r["precision"] for r in records], dtype=np.float64)
    score = 0.55 * float(np.mean(ious)) + 0.25 * float(np.mean(recalls)) + 0.20 * float(np.mean(precisions))
    return {
        "enabled": True,
        "reason": "vertex_projection_mask_overlap",
        "score": float(score),
        "iou_mean": float(np.mean(ious)),
        "iou_median": float(np.median(ious)),
        "recall_mean": float(np.mean(recalls)),
        "precision_mean": float(np.mean(precisions)),
        "records": records,
        "note": "Approximate sanity check: projects normalized mesh vertices into normalized AR-pose cameras; not a full visible-surface render.",
    }
