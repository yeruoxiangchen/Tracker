#!/usr/bin/env python3

"""Pose-guided ReconViaGen wrapper.

This module intentionally lives outside the existing server/CoarseModel flow.
It can run ReconViaGen per seed, score each mesh against masks using phone-AR
or COLMAP poses, choose the best candidate, and optionally apply a conservative
visual-hull face filter.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import trimesh


ROOT = Path("/home/zjr/Tracker")
RECON_ROOT = ROOT / "ReconViaGen"
DEFAULT_OUTPUT_ROOT = ROOT / "CoarseModel" / "pose_guided_recon" / "outputs"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


@dataclass
class CameraFrame:
    name: str
    image_path: Path
    mask_path: Path
    width: int
    height: int
    K: np.ndarray
    R_w2c: np.ndarray
    t_w2c: np.ndarray
    center_w: np.ndarray


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def reset_dir(path: Path) -> Path:
    if path.exists() or path.is_symlink():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return path


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def link_or_copy(src: Path, dst: Path, mode: str = "symlink") -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    if mode == "copy":
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
    else:
        os.symlink(src.resolve(), dst, target_is_directory=src.is_dir())


def image_files(path: Path) -> List[Path]:
    if not path.exists():
        return []
    return sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def find_mask(mask_dir: Path, stem: str) -> Optional[Path]:
    for suffix in [".png", ".jpg", ".jpeg"]:
        path = mask_dir / f"{stem}{suffix}"
        if path.exists():
            return path
    return None


def safe_name(text: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in text)[:96]


def prepare_workspace_dataset(
    input_type: str,
    dataset_dir: Optional[Path],
    phone_data_dir: Optional[Path],
    phone_mask_dir: Optional[Path],
    output_root: Path,
    case_name: str,
    link_mode: str,
) -> Path:
    workspace = reset_dir(output_root / "workspace" / "raw_dataset")
    for name in ["rgb", "images", "masks", "models"]:
        ensure_dir(workspace / name)

    if input_type == "phone_ar_session":
        if phone_data_dir is None:
            raise ValueError("--phone_data_dir is required for phone_ar_session")
        session = phone_data_dir.name
        if phone_mask_dir is None:
            phone_mask_dir = ROOT / "ReconViaGen" / "ar_tracker" / "masks" / session
        src_image_dir = phone_data_dir
        src_mask_dir = phone_mask_dir
        src_pose = phone_data_dir / "poses.txt"
    else:
        if dataset_dir is None:
            raise ValueError("--dataset_dir is required for coarse_dataset/colmap_dataset")
        src_image_dir = dataset_dir / "rgb"
        if not src_image_dir.exists():
            src_image_dir = dataset_dir / "images"
        src_mask_dir = dataset_dir / "masks"
        src_pose = dataset_dir / "poses.txt"
        sparse = dataset_dir / "sparse"
        if sparse.exists():
            link_or_copy(sparse, workspace / "sparse", mode=link_mode)

    if not src_image_dir.exists() or not src_mask_dir.exists():
        raise FileNotFoundError(f"Missing images or masks: {src_image_dir}, {src_mask_dir}")

    selected_frames: List[str] = []
    frame_reports: List[Dict[str, Any]] = []
    for src_img in image_files(src_image_dir):
        mask = find_mask(src_mask_dir, src_img.stem)
        if mask is None:
            continue
        selected_frames.append(src_img.name)
        link_or_copy(src_img, workspace / "rgb" / src_img.name, mode=link_mode)
        link_or_copy(src_img, workspace / "images" / src_img.name, mode=link_mode)
        link_or_copy(mask, workspace / "masks" / f"{src_img.stem}.png", mode=link_mode)
        frame_reports.append(
            {
                "name": src_img.name,
                "source_image": str(src_img),
                "source_mask": str(mask),
            }
        )
    if not selected_frames:
        raise RuntimeError(f"No image/mask pairs found in {src_image_dir} and {src_mask_dir}")
    if src_pose.exists():
        link_or_copy(src_pose, workspace / "poses.txt", mode=link_mode)

    meta = {
        "case_name": case_name,
        "input_type": input_type,
        "source_dataset_dir": str(dataset_dir) if dataset_dir else None,
        "phone_data_dir": str(phone_data_dir) if phone_data_dir else None,
        "phone_mask_dir": str(phone_mask_dir) if phone_mask_dir else None,
        "dataset_dir": str(workspace),
        "selected_frames": selected_frames,
        "selected_indices": [frame_index_from_name(n, i) for i, n in enumerate(selected_frames)],
        "frames": frame_reports,
    }
    write_json(workspace / "reconviagen_meta.json", meta)
    return workspace


def create_selected_workspace(
    raw_workspace: Path,
    output_root: Path,
    selected_frame_names: Sequence[str],
    link_mode: str,
) -> Path:
    selected = reset_dir(output_root / "workspace" / "selected_dataset")
    for name in ["rgb", "images", "masks", "models"]:
        ensure_dir(selected / name)

    selected_set = set(selected_frame_names)
    for src_img in image_files(raw_workspace / "rgb"):
        if src_img.name not in selected_set:
            continue
        mask = find_mask(raw_workspace / "masks", src_img.stem)
        if mask is None:
            continue
        link_or_copy(src_img, selected / "rgb" / src_img.name, mode=link_mode)
        link_or_copy(src_img, selected / "images" / src_img.name, mode=link_mode)
        link_or_copy(mask, selected / "masks" / f"{src_img.stem}.png", mode=link_mode)

    for aux in ["poses.txt", "reconviagen_meta.json"]:
        src = raw_workspace / aux
        if src.exists():
            link_or_copy(src, selected / aux, mode=link_mode)
    sparse = raw_workspace / "sparse"
    if sparse.exists():
        link_or_copy(sparse, selected / "sparse", mode=link_mode)

    meta = read_json(raw_workspace / "reconviagen_meta.json") if (raw_workspace / "reconviagen_meta.json").exists() else {}
    meta["raw_dataset_dir"] = str(raw_workspace)
    meta["dataset_dir"] = str(selected)
    meta["selected_frames"] = list(selected_frame_names)
    meta["selected_indices"] = [frame_index_from_name(n, i) for i, n in enumerate(selected_frame_names)]
    write_json(selected / "reconviagen_meta.json", meta)
    return selected


def frame_index_from_name(name: str, fallback: int) -> int:
    stem = Path(name).stem
    digits = "".join(c for c in stem if c.isdigit())
    return int(digits) if digits else int(fallback)


def qvec_to_rotmat(qvec: Sequence[float]) -> np.ndarray:
    qw, qx, qy, qz = [float(v) for v in qvec]
    return np.array(
        [
            [1 - 2 * qy * qy - 2 * qz * qz, 2 * qx * qy - 2 * qz * qw, 2 * qx * qz + 2 * qy * qw],
            [2 * qx * qy + 2 * qz * qw, 1 - 2 * qx * qx - 2 * qz * qz, 2 * qy * qz - 2 * qx * qw],
            [2 * qx * qz - 2 * qy * qw, 2 * qy * qz + 2 * qx * qw, 1 - 2 * qx * qx - 2 * qy * qy],
        ],
        dtype=np.float64,
    )


def rotmat_to_qvec(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=np.float64)
    K = np.array(
        [
            [R[0, 0] - R[1, 1] - R[2, 2], 0.0, 0.0, 0.0],
            [R[1, 0] + R[0, 1], R[1, 1] - R[0, 0] - R[2, 2], 0.0, 0.0],
            [R[2, 0] + R[0, 2], R[2, 1] + R[1, 2], R[2, 2] - R[0, 0] - R[1, 1], 0.0],
            [R[1, 2] - R[2, 1], R[2, 0] - R[0, 2], R[0, 1] - R[1, 0], R[0, 0] + R[1, 1] + R[2, 2]],
        ],
        dtype=np.float64,
    )
    K /= 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec


def parse_colmap_cameras(path: Path) -> Dict[int, Dict[str, Any]]:
    cameras: Dict[int, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            cam_id = int(parts[0])
            model = parts[1]
            width, height = int(parts[2]), int(parts[3])
            params = [float(x) for x in parts[4:]]
            if model in {"SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"}:
                fx = fy = params[0]
                cx, cy = params[1], params[2]
            elif model == "PINHOLE":
                fx, fy, cx, cy = params[:4]
            else:
                raise ValueError(f"Unsupported COLMAP camera model: {model}")
            cameras[cam_id] = {"width": width, "height": height, "K": np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)}
    return cameras


def parse_colmap_images(path: Path) -> Dict[str, Dict[str, Any]]:
    images: Dict[str, Dict[str, Any]] = {}
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line or line.startswith("#"):
            i += 1
            continue
        parts = line.split()
        if len(parts) >= 10:
            qvec = [float(v) for v in parts[1:5]]
            tvec = np.array([float(v) for v in parts[5:8]], dtype=np.float64)
            cam_id = int(parts[8])
            name = " ".join(parts[9:])
            R_w2c = qvec_to_rotmat(qvec)
            images[name] = {"R_w2c": R_w2c, "t_w2c": tvec, "camera_id": cam_id}
            i += 2
        else:
            i += 1
    return images


def unity_quat_to_rotmat(quat_xyzw: Sequence[float]) -> np.ndarray:
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


def unity_euler_to_rotmat(rot_deg: Sequence[float]) -> np.ndarray:
    rx, ry, rz = np.deg2rad(np.asarray(rot_deg, dtype=np.float64))
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    rx_mat = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float64)
    ry_mat = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)
    rz_mat = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float64)
    return ry_mat @ rx_mat @ rz_mat


def nearest_rotation_matrix(rot: np.ndarray) -> np.ndarray:
    u, _s, vt = np.linalg.svd(rot)
    fixed = u @ vt
    if np.linalg.det(fixed) < 0:
        u[:, -1] *= -1
        fixed = u @ vt
    return fixed


def phone_pose_to_w2c(pose: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    if pose.get("quat") is not None:
        r_unity_c2w = unity_quat_to_rotmat(pose["quat"])
    else:
        r_unity_c2w = unity_euler_to_rotmat(pose["rot_deg"])
    unity_to_cv_world = np.diag([1.0, 1.0, -1.0])
    unity_cam_to_cv_cam = np.diag([1.0, -1.0, 1.0])
    center_w = unity_to_cv_world @ np.asarray(pose["pos"], dtype=np.float64)
    r_cv_c2w = nearest_rotation_matrix(unity_to_cv_world @ r_unity_c2w @ unity_cam_to_cv_cam)
    R_w2c = r_cv_c2w.T
    t_w2c = -R_w2c @ center_w
    return R_w2c, t_w2c


def read_phone_poses(path: Path) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = [p.strip() for p in line.strip().split(",")]
            if len(parts) < 7 or parts[0] == "frame_name":
                continue
            try:
                pos = [float(v) for v in parts[1:4]]
                rot = [float(v) for v in parts[4:7]]
            except ValueError:
                continue
            pose: Dict[str, Any] = {"pos": np.asarray(pos, dtype=np.float64), "rot_deg": np.asarray(rot, dtype=np.float64), "quat": None, "intrinsics": None, "image_transform": "None"}
            if len(parts) >= 11:
                try:
                    pose["quat"] = np.asarray([float(v) for v in parts[7:11]], dtype=np.float64)
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
            if len(parts) > 21:
                pose["image_transform"] = parts[21] or "None"
            out[parts[0]] = pose
    return out


def image_size(path: Path) -> Tuple[int, int]:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(path)
    h, w = image.shape[:2]
    return int(w), int(h)


def intrinsics_for_phone_pose(pose: Dict[str, Any], width: int, height: int) -> np.ndarray:
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
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


def load_camera_frames(dataset_dir: Path, pose_source: str) -> List[CameraFrame]:
    image_dir = dataset_dir / "rgb"
    if not image_dir.exists():
        image_dir = dataset_dir / "images"
    mask_dir = dataset_dir / "masks"
    frames: List[CameraFrame] = []

    if pose_source == "auto":
        pose_source = "colmap" if (dataset_dir / "sparse" / "0" / "images.txt").exists() else "phone_ar"

    if pose_source == "colmap":
        sparse = dataset_dir / "sparse" / "0"
        cameras = parse_colmap_cameras(sparse / "cameras.txt")
        images = parse_colmap_images(sparse / "images.txt")
        for image_path in image_files(image_dir):
            record = images.get(image_path.name)
            mask = find_mask(mask_dir, image_path.stem)
            if record is None or mask is None:
                continue
            camera = cameras[record["camera_id"]]
            K = camera["K"].copy()
            width, height = image_size(image_path)
            if width != camera["width"] or height != camera["height"]:
                K[0, :] *= width / float(camera["width"])
                K[1, :] *= height / float(camera["height"])
            R_w2c = record["R_w2c"]
            t_w2c = record["t_w2c"]
            frames.append(CameraFrame(image_path.name, image_path, mask, width, height, K, R_w2c, t_w2c, -R_w2c.T @ t_w2c))
    elif pose_source == "phone_ar":
        poses = read_phone_poses(dataset_dir / "poses.txt")
        for image_path in image_files(image_dir):
            pose = poses.get(image_path.name)
            mask = find_mask(mask_dir, image_path.stem)
            if pose is None or mask is None:
                continue
            width, height = image_size(image_path)
            K = intrinsics_for_phone_pose(pose, width, height)
            R_w2c, t_w2c = phone_pose_to_w2c(pose)
            frames.append(CameraFrame(image_path.name, image_path, mask, width, height, K, R_w2c, t_w2c, -R_w2c.T @ t_w2c))
    else:
        raise ValueError(f"Unsupported pose_source={pose_source}")

    if not frames:
        raise RuntimeError(f"No camera frames loaded from {dataset_dir} with pose_source={pose_source}")
    return frames


def mask_binary(path: Path, width: Optional[int] = None, height: Optional[int] = None) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(path)
    if width is not None and height is not None and (mask.shape[1] != width or mask.shape[0] != height):
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    return mask > 0


def mask_center(mask: np.ndarray) -> Optional[Tuple[float, float]]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return float(xs.mean()), float(ys.mean())


def estimate_object_center(frames: Sequence[CameraFrame]) -> Dict[str, Any]:
    A = np.zeros((3, 3), dtype=np.float64)
    b = np.zeros(3, dtype=np.float64)
    used = []
    rays = []
    centers = []
    for frame in frames:
        mask = mask_binary(frame.mask_path, frame.width, frame.height)
        center_px = mask_center(mask)
        if center_px is None:
            continue
        u, v = center_px
        ray_cam = np.array([(u - frame.K[0, 2]) / frame.K[0, 0], (v - frame.K[1, 2]) / frame.K[1, 1], 1.0], dtype=np.float64)
        ray_cam /= max(float(np.linalg.norm(ray_cam)), 1e-12)
        ray_w = frame.R_w2c.T @ ray_cam
        ray_w /= max(float(np.linalg.norm(ray_w)), 1e-12)
        proj = np.eye(3, dtype=np.float64) - np.outer(ray_w, ray_w)
        A += proj
        b += proj @ frame.center_w
        used.append(frame.name)
        rays.append(ray_w)
        centers.append(frame.center_w)
    if len(used) < 3:
        raise RuntimeError("Too few mask/pose rays to estimate object center")
    center = np.linalg.lstsq(A, b, rcond=None)[0]
    residuals = [float(np.linalg.norm(np.cross(center - c, r))) for c, r in zip(centers, rays)]
    distances = [float(np.linalg.norm(c - center)) for c in centers]
    return {
        "center": center,
        "used_names": used,
        "ray_count": len(used),
        "ray_residual_mean": float(np.mean(residuals)),
        "ray_residual_p90": float(np.percentile(residuals, 90)),
        "camera_distance_median": float(np.median(distances)),
    }


def estimate_object_radius(frames: Sequence[CameraFrame], center: np.ndarray) -> float:
    radii = []
    for frame in frames:
        mask = mask_binary(frame.mask_path, frame.width, frame.height)
        area = float(mask.mean())
        if area <= 0:
            continue
        dist = float(np.linalg.norm(frame.center_w - center))
        focal = float(math.sqrt(frame.K[0, 0] * frame.K[1, 1]))
        pixel_radius = math.sqrt(float(mask.sum()) / math.pi)
        radii.append(dist * pixel_radius / max(focal, 1e-6))
    return float(np.median(radii)) if radii else 0.25


def circular_distance_deg(a: float, b: float) -> float:
    return abs((float(a) - float(b) + 180.0) % 360.0 - 180.0)


def image_sharpness(path: Path) -> float:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return 0.0
    small = cv2.resize(image, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA) if max(image.shape[:2]) > 720 else image
    return float(cv2.Laplacian(small, cv2.CV_64F).var())


def frame_pose_features(frame: CameraFrame, object_center: np.ndarray) -> Dict[str, Any]:
    vec = frame.center_w - object_center
    horiz = math.sqrt(float(vec[0] * vec[0] + vec[2] * vec[2]))
    azimuth = (math.degrees(math.atan2(float(vec[0]), float(vec[2]))) + 360.0) % 360.0
    elevation = math.degrees(math.atan2(float(vec[1]), max(horiz, 1e-8)))
    target = mask_binary(frame.mask_path, frame.width, frame.height)
    area = float(target.mean())
    return {
        "name": frame.name,
        "azimuth_deg": float(azimuth),
        "elevation_deg": float(elevation),
        "camera_distance": float(np.linalg.norm(vec)),
        "mask_area": area,
        "sharpness": image_sharpness(frame.image_path),
    }


def select_coverage_frames(
    frames: Sequence[CameraFrame],
    object_center: np.ndarray,
    max_frames: int,
    min_frames: int,
    min_mask_area: float,
    max_mask_area: float,
) -> Dict[str, Any]:
    features = [frame_pose_features(frame, object_center) for frame in frames]
    if max_frames <= 0 or max_frames >= len(frames):
        selected = sorted(features, key=lambda x: frame_index_from_name(x["name"], 0))
        return {
            "mode": "all",
            "requested_max_frames": int(max_frames),
            "selected_frame_names": [f["name"] for f in selected],
            "selected_count": len(selected),
            "all_frames": features,
        }

    valid = [f for f in features if min_mask_area <= float(f["mask_area"]) <= max_mask_area]
    if len(valid) < max(1, int(min_frames)):
        valid = sorted(features, key=lambda x: abs(float(x["mask_area"]) - 0.12))[: max(int(min_frames), min(len(features), max_frames))]
    sharp_values = np.asarray([f["sharpness"] for f in valid], dtype=np.float64)
    sharp_lo = float(np.percentile(sharp_values, 5)) if len(sharp_values) else 0.0
    sharp_hi = float(np.percentile(sharp_values, 95)) if len(sharp_values) else 1.0

    def quality(feat: Dict[str, Any]) -> float:
        area = float(feat["mask_area"])
        area_score = max(0.0, 1.0 - abs(area - 0.12) / 0.12)
        sharp_score = (float(feat["sharpness"]) - sharp_lo) / max(sharp_hi - sharp_lo, 1e-6)
        sharp_score = float(np.clip(sharp_score, 0.0, 1.0))
        return 0.70 * area_score + 0.30 * sharp_score

    selected: List[Dict[str, Any]] = []
    candidates = list(valid)
    while candidates and len(selected) < max_frames:
        best_idx = 0
        best_score = -1e9
        for idx, feat in enumerate(candidates):
            q = quality(feat)
            if not selected:
                novelty = 1.0
                elev_novelty = 1.0
            else:
                min_az = min(circular_distance_deg(feat["azimuth_deg"], s["azimuth_deg"]) for s in selected)
                min_el = min(abs(float(feat["elevation_deg"]) - float(s["elevation_deg"])) for s in selected)
                novelty = min(min_az / 90.0, 1.0)
                elev_novelty = min(min_el / 30.0, 1.0)
            score = 0.58 * novelty + 0.17 * elev_novelty + 0.25 * q
            if score > best_score:
                best_score = score
                best_idx = idx
        selected.append(candidates.pop(best_idx))

    selected = sorted(selected, key=lambda x: frame_index_from_name(x["name"], 0))
    azimuths = [float(f["azimuth_deg"]) for f in selected]
    if len(azimuths) > 1:
        az_sorted = sorted(azimuths)
        gaps = [az_sorted[(i + 1) % len(az_sorted)] - az_sorted[i] for i in range(len(az_sorted) - 1)]
        gaps.append(az_sorted[0] + 360.0 - az_sorted[-1])
        coverage_deg = 360.0 - max(gaps)
    else:
        coverage_deg = 0.0
    return {
        "mode": "coverage_greedy",
        "requested_max_frames": int(max_frames),
        "requested_min_frames": int(min_frames),
        "selected_count": len(selected),
        "selected_frame_names": [f["name"] for f in selected],
        "azimuth_coverage_deg": float(coverage_deg),
        "mean_mask_area": float(np.mean([f["mask_area"] for f in selected])) if selected else 0.0,
        "mean_sharpness": float(np.mean([f["sharpness"] for f in selected])) if selected else 0.0,
        "selected_frames": selected,
        "valid_frame_count": len(valid),
        "all_frames": features,
    }


def load_mesh_points(mesh_path: Path, max_points: int, seed: int) -> Tuple[trimesh.Trimesh, np.ndarray]:
    mesh_obj = trimesh.load(mesh_path, force="scene", process=False)
    if isinstance(mesh_obj, trimesh.Scene):
        parts = [g for g in mesh_obj.dump(concatenate=False) if isinstance(g, trimesh.Trimesh) and len(g.vertices) > 0]
        if not parts:
            raise RuntimeError(f"Mesh has no geometry: {mesh_path}")
        mesh = trimesh.util.concatenate(parts)
    else:
        mesh = mesh_obj
    if len(mesh.faces) > 0:
        rng_state = np.random.get_state()
        np.random.seed(seed)
        try:
            points, _ = trimesh.sample.sample_surface(mesh, int(max_points))
        finally:
            np.random.set_state(rng_state)
    else:
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        if len(vertices) > max_points:
            rng = np.random.default_rng(seed)
            vertices = vertices[rng.choice(len(vertices), size=max_points, replace=False)]
        points = vertices
    return mesh, np.asarray(points, dtype=np.float64)


def yaw_matrix(deg: float) -> np.ndarray:
    a = math.radians(float(deg))
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def project_points(points_w: np.ndarray, frame: CameraFrame) -> Tuple[np.ndarray, np.ndarray]:
    cam = (frame.R_w2c @ points_w.T).T + frame.t_w2c[None]
    z = cam[:, 2]
    valid = z > 1e-4
    if not np.any(valid):
        return np.zeros((0, 2), dtype=np.int32), valid
    cam_valid = cam[valid]
    proj = (frame.K @ cam_valid.T).T
    uv = proj[:, :2] / np.maximum(proj[:, 2:3], 1e-8)
    xy = np.rint(uv).astype(np.int32)
    keep = (xy[:, 0] >= 0) & (xy[:, 0] < frame.width) & (xy[:, 1] >= 0) & (xy[:, 1] < frame.height)
    return xy[keep], valid


def rasterize_points_mask(xy: np.ndarray, width: int, height: int, dilation: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    if len(xy) == 0:
        return mask.astype(bool)
    mask[xy[:, 1], xy[:, 0]] = 255
    k = max(1, int(dilation))
    kernel = np.ones((k, k), dtype=np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((max(3, k * 2 + 1), max(3, k * 2 + 1)), dtype=np.uint8), iterations=1)
    return mask > 0


def binary_metrics(pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    inter = float(np.logical_and(pred, target).sum())
    union = float(np.logical_or(pred, target).sum())
    pred_sum = float(pred.sum())
    target_sum = float(target.sum())
    return {
        "iou": inter / union if union > 0 else 0.0,
        "recall": inter / target_sum if target_sum > 0 else 0.0,
        "precision": inter / pred_sum if pred_sum > 0 else 0.0,
        "pred_area": pred_sum / float(pred.size),
        "target_area": target_sum / float(target.size),
    }


def transform_points(points: np.ndarray, mesh_center: np.ndarray, object_center: np.ndarray, scale: float, yaw_deg: float) -> np.ndarray:
    R = yaw_matrix(yaw_deg)
    return object_center[None] + float(scale) * ((R @ (points - mesh_center).T).T)


def score_transform(
    points: np.ndarray,
    mesh_center: np.ndarray,
    frames: Sequence[CameraFrame],
    object_center: np.ndarray,
    scale: float,
    yaw_deg: float,
    point_dilation: int,
) -> Dict[str, Any]:
    points_w = transform_points(points, mesh_center, object_center, scale, yaw_deg)
    rows = []
    for frame in frames:
        xy, _valid = project_points(points_w, frame)
        pred = rasterize_points_mask(xy, frame.width, frame.height, dilation=point_dilation)
        target = mask_binary(frame.mask_path, frame.width, frame.height)
        row = {"frame_name": frame.name, **binary_metrics(pred, target), "projected_points": int(len(xy))}
        row["score"] = 0.45 * row["iou"] + 0.25 * row["recall"] + 0.20 * row["precision"] - 0.10 * abs(row["pred_area"] - row["target_area"])
        rows.append(row)
    keys = ["score", "iou", "recall", "precision", "pred_area", "target_area"]
    summary = {f"{key}_mean": float(np.mean([r[key] for r in rows])) for key in keys}
    summary.update({f"{key}_median": float(np.median([r[key] for r in rows])) for key in keys})
    return {"summary": summary, "rows": rows, "scale": float(scale), "yaw_deg": float(yaw_deg)}


def grid_score_mesh(
    mesh_path: Path,
    frames: Sequence[CameraFrame],
    object_report: Dict[str, Any],
    max_points: int,
    yaw_steps: int,
    scale_factors: Sequence[float],
    point_dilation: int,
    seed: int,
) -> Dict[str, Any]:
    mesh, points = load_mesh_points(mesh_path, max_points=max_points, seed=seed)
    mesh_vertices = np.asarray(mesh.vertices, dtype=np.float64)
    mesh_center = 0.5 * (mesh_vertices.min(axis=0) + mesh_vertices.max(axis=0))
    mesh_radius = float(np.median(np.linalg.norm(points - mesh_center[None], axis=1)))
    object_center = np.asarray(object_report["center"], dtype=np.float64)
    object_radius = estimate_object_radius(frames, object_center)
    base_scale = object_radius / max(mesh_radius, 1e-8)
    best = None
    trials = []
    for yaw in np.linspace(0.0, 360.0, max(1, int(yaw_steps)), endpoint=False):
        for factor in scale_factors:
            scale = base_scale * float(factor)
            score = score_transform(points, mesh_center, frames, object_center, scale, float(yaw), point_dilation)
            trial = {
                "yaw_deg": float(yaw),
                "scale": float(scale),
                "scale_factor": float(factor),
                "score_mean": score["summary"]["score_mean"],
                "iou_mean": score["summary"]["iou_mean"],
                "recall_mean": score["summary"]["recall_mean"],
                "precision_mean": score["summary"]["precision_mean"],
                "pred_area_mean": score["summary"]["pred_area_mean"],
                "target_area_mean": score["summary"]["target_area_mean"],
            }
            trials.append(trial)
            if best is None or trial["score_mean"] > best["summary"]["score_mean"]:
                best = score
    assert best is not None
    return {
        "mesh_path": str(mesh_path),
        "mesh_basic": {
            "vertices": int(len(mesh.vertices)),
            "faces": int(len(mesh.faces)),
            "bbox_extent": [float(v) for v in mesh.bounding_box.extents.tolist()],
        },
        "mesh_center": [float(v) for v in mesh_center.tolist()],
        "mesh_radius_median": mesh_radius,
        "object_radius_est": object_radius,
        "base_scale": base_scale,
        "best": best,
        "top_trials": sorted(trials, key=lambda x: x["score_mean"], reverse=True)[:20],
    }


def latest_child_dir(path: Path, not_before: float = 0.0) -> Optional[Path]:
    if not path.exists():
        return None
    children = [p for p in path.iterdir() if p.is_dir() and p.stat().st_mtime >= not_before]
    if not children:
        children = [p for p in path.iterdir() if p.is_dir()]
    return max(children, key=lambda p: p.stat().st_mtime) if children else None


def run_recon_seed(dataset_dir: Path, output_root: Path, python_bin: str, seed: int, resolution: int, mesh_simplify: float) -> Dict[str, Any]:
    recon_root = ensure_dir(output_root / "reconviagen_seed_runs" / f"seed_{seed}")
    script = RECON_ROOT / "rebuild_mesh_from_coarse_dataset.py"
    cmd = [
        python_bin,
        "-u",
        str(script),
        "--dataset_dir",
        str(dataset_dir),
        "--source",
        "dataset_masks",
        "--output_root",
        str(recon_root),
        "--resolution",
        str(resolution),
        "--seeds",
        str(seed),
        "--mesh_simplify",
        str(mesh_simplify),
    ]
    env = os.environ.copy()
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    env.setdefault("ATTN_BACKEND", "flash_attn")
    env.setdefault("SPCONV_ALGO", "native")
    env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    env.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("RECON_POSE_RERANK", "0")
    start = time.time()
    (output_root / f"recon_seed_{seed}_command.txt").write_text(" ".join(cmd) + "\n", encoding="utf-8")
    print(f"[pose_guided] ReconViaGen seed={seed}", flush=True)
    proc = subprocess.run(cmd, cwd=str(RECON_ROOT), env=env, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ReconViaGen seed={seed} failed: returncode={proc.returncode}")
    child = latest_child_dir(recon_root, not_before=start - 1.0)
    if child is None:
        raise RuntimeError(f"No ReconViaGen output under {recon_root}")
    mesh = child / "reconstructed_object.glb"
    if not mesh.exists():
        raise FileNotFoundError(mesh)
    report = child / "rebuild_report.json"
    return {"seed": int(seed), "mesh_path": str(mesh), "output_dir": str(child), "rebuild_report": str(report) if report.exists() else None}


def parse_seed_list(text: str) -> List[int]:
    seeds = []
    for item in text.split(","):
        item = item.strip()
        if item:
            seeds.append(int(item))
    return seeds or [0]


def parse_float_list(text: str) -> List[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def parse_mesh_list(text: Optional[str]) -> List[Path]:
    if not text:
        return []
    return [Path(x.strip()) for x in text.split(",") if x.strip()]


def visual_hull_filter_mesh(
    mesh_path: Path,
    score_report: Dict[str, Any],
    frames: Sequence[CameraFrame],
    output_path: Path,
    inside_ratio_threshold: float,
    min_visible_views: int,
    mask_dilation: int,
) -> Dict[str, Any]:
    mesh_obj = trimesh.load(mesh_path, force="scene", process=False)
    if isinstance(mesh_obj, trimesh.Scene):
        parts = [g for g in mesh_obj.dump(concatenate=False) if isinstance(g, trimesh.Trimesh) and len(g.vertices) > 0]
        mesh = trimesh.util.concatenate(parts)
    else:
        mesh = mesh_obj
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    mesh_center = np.asarray(score_report["mesh_center"], dtype=np.float64)
    object_center = np.asarray(score_report.get("object_center") or score_report["object_report"]["center"], dtype=np.float64)
    scale = float(score_report["best"]["scale"])
    yaw = float(score_report["best"]["yaw_deg"])
    vertices_w = transform_points(vertices, mesh_center, object_center, scale, yaw)
    visible = np.zeros(len(vertices), dtype=np.int32)
    inside = np.zeros(len(vertices), dtype=np.int32)
    for frame in frames:
        cam = (frame.R_w2c @ vertices_w.T).T + frame.t_w2c[None]
        z = cam[:, 2]
        valid = z > 1e-4
        proj = (frame.K @ cam[valid].T).T if np.any(valid) else np.zeros((0, 3))
        uv = proj[:, :2] / np.maximum(proj[:, 2:3], 1e-8) if len(proj) else np.zeros((0, 2))
        xy = np.rint(uv).astype(np.int32) if len(uv) else np.zeros((0, 2), dtype=np.int32)
        valid_ids = np.nonzero(valid)[0]
        keep = (xy[:, 0] >= 0) & (xy[:, 0] < frame.width) & (xy[:, 1] >= 0) & (xy[:, 1] < frame.height)
        ids = valid_ids[keep]
        xy = xy[keep]
        visible[ids] += 1
        mask = mask_binary(frame.mask_path, frame.width, frame.height).astype(np.uint8) * 255
        if mask_dilation > 0:
            mask = cv2.dilate(mask, np.ones((mask_dilation, mask_dilation), dtype=np.uint8), iterations=1)
        ok = mask[xy[:, 1], xy[:, 0]] > 0
        inside[ids[ok]] += 1
    ratio = inside / np.maximum(visible, 1)
    vertex_keep = (visible < int(min_visible_views)) | (ratio >= float(inside_ratio_threshold))
    faces = np.asarray(mesh.faces, dtype=np.int64)
    face_keep = vertex_keep[faces].mean(axis=1) >= 0.67
    filtered = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces[face_keep], visual=mesh.visual.copy() if mesh.visual is not None else None, process=False)
    components = filtered.split(only_watertight=False)
    if components:
        filtered = max(components, key=lambda m: len(m.faces))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    filtered.export(output_path)
    return {
        "input_mesh": str(mesh_path),
        "output_mesh": str(output_path),
        "vertices_before": int(len(mesh.vertices)),
        "faces_before": int(len(mesh.faces)),
        "faces_after": int(len(filtered.faces)),
        "kept_face_ratio": float(len(filtered.faces) / max(len(mesh.faces), 1)),
        "inside_ratio_threshold": float(inside_ratio_threshold),
        "min_visible_views": int(min_visible_views),
    }


def export_selected_mesh(mesh_path: Path, output_path: Path) -> str:
    mesh = trimesh.load(mesh_path, force="scene", process=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(output_path)
    return str(output_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_type", choices=["coarse_dataset", "phone_ar_session", "colmap_dataset"], default="coarse_dataset")
    parser.add_argument("--dataset_dir", default=None)
    parser.add_argument("--phone_data_dir", default=None)
    parser.add_argument("--phone_mask_dir", default=None)
    parser.add_argument("--pose_source", choices=["auto", "phone_ar", "colmap"], default="auto")
    parser.add_argument("--case_name", default=None)
    parser.add_argument("--output_root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--python_bin", default=sys.executable)
    parser.add_argument("--seeds", default="0,1")
    parser.add_argument("--existing_meshes", default=None, help="Comma-separated mesh paths; skips ReconViaGen generation for these meshes.")
    parser.add_argument("--run_recon", type=int, default=1)
    parser.add_argument("--resolution", type=int, default=518)
    parser.add_argument("--mesh_simplify", type=float, default=0.75)
    parser.add_argument("--link_mode", choices=["symlink", "copy"], default="symlink")
    parser.add_argument("--max_input_frames", type=int, default=18)
    parser.add_argument("--min_input_frames", type=int, default=8)
    parser.add_argument("--min_mask_area", type=float, default=0.005)
    parser.add_argument("--max_mask_area", type=float, default=0.65)
    parser.add_argument("--score_all_frames", type=int, default=1)
    parser.add_argument("--max_score_points", type=int, default=12000)
    parser.add_argument("--yaw_steps", type=int, default=24)
    parser.add_argument("--scale_factors", default="0.75,0.9,1.0,1.1,1.25")
    parser.add_argument("--point_dilation", type=int, default=9)
    parser.add_argument("--apply_visual_hull_filter", type=int, default=0)
    parser.add_argument("--vh_inside_ratio_threshold", type=float, default=0.45)
    parser.add_argument("--vh_min_visible_views", type=int, default=3)
    parser.add_argument("--vh_mask_dilation", type=int, default=7)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir).resolve() if args.dataset_dir else None
    phone_data_dir = Path(args.phone_data_dir).resolve() if args.phone_data_dir else None
    phone_mask_dir = Path(args.phone_mask_dir).resolve() if args.phone_mask_dir else None
    if args.case_name:
        case_name = safe_name(args.case_name)
    elif dataset_dir is not None:
        case_name = safe_name(dataset_dir.name)
    elif phone_data_dir is not None:
        case_name = safe_name(phone_data_dir.name)
    else:
        case_name = "pose_guided_recon"
    output_root = ensure_dir(Path(args.output_root).resolve() / case_name)
    raw_workspace = prepare_workspace_dataset(
        input_type=args.input_type,
        dataset_dir=dataset_dir,
        phone_data_dir=phone_data_dir,
        phone_mask_dir=phone_mask_dir,
        output_root=output_root,
        case_name=case_name,
        link_mode=args.link_mode,
    )
    all_frames = load_camera_frames(raw_workspace, args.pose_source)
    full_object_report = estimate_object_center(all_frames)
    selection_report = select_coverage_frames(
        all_frames,
        np.asarray(full_object_report["center"], dtype=np.float64),
        max_frames=args.max_input_frames,
        min_frames=args.min_input_frames,
        min_mask_area=args.min_mask_area,
        max_mask_area=args.max_mask_area,
    )
    write_json(output_root / "frame_selection_report.json", selection_report)
    dataset_workspace = create_selected_workspace(
        raw_workspace=raw_workspace,
        output_root=output_root,
        selected_frame_names=selection_report["selected_frame_names"],
        link_mode=args.link_mode,
    )
    selected_frames = load_camera_frames(dataset_workspace, args.pose_source)
    frames = all_frames if int(args.score_all_frames) else selected_frames
    object_report = full_object_report if int(args.score_all_frames) else estimate_object_center(selected_frames)
    object_report_json = dict(object_report)
    object_report_json["center"] = [float(v) for v in np.asarray(object_report["center"]).tolist()]
    write_json(output_root / "object_center_report.json", object_report_json)

    candidates: List[Dict[str, Any]] = []
    if args.run_recon:
        for seed in parse_seed_list(args.seeds):
            candidates.append(run_recon_seed(dataset_workspace, output_root, args.python_bin, seed, args.resolution, args.mesh_simplify))
    for idx, mesh_path in enumerate(parse_mesh_list(args.existing_meshes)):
        candidates.append({"seed": f"existing_{idx}", "mesh_path": str(mesh_path.resolve()), "output_dir": str(mesh_path.resolve().parent), "rebuild_report": None})
    if not candidates:
        raise RuntimeError("No candidates. Use --run_recon 1 or --existing_meshes.")

    scored = []
    for i, candidate in enumerate(candidates):
        mesh_path = Path(str(candidate["mesh_path"]))
        print(f"[pose_guided] scoring candidate {i + 1}/{len(candidates)}: {mesh_path}", flush=True)
        score = grid_score_mesh(
            mesh_path=mesh_path,
            frames=frames,
            object_report=object_report,
            max_points=args.max_score_points,
            yaw_steps=args.yaw_steps,
            scale_factors=parse_float_list(args.scale_factors),
            point_dilation=args.point_dilation,
            seed=1234 + i,
        )
        score["candidate"] = candidate
        score["object_report"] = object_report_json
        score["object_center"] = object_report_json["center"]
        scored.append(score)
        write_json(output_root / "candidate_scores" / f"candidate_{i:02d}.json", score)
    best = max(scored, key=lambda item: item["best"]["summary"]["score_mean"])
    selected_dir = ensure_dir(output_root / "selected")
    selected_mesh = export_selected_mesh(Path(best["mesh_path"]), selected_dir / "selected_reconviagen_mesh.glb")
    vh_report = None
    if args.apply_visual_hull_filter:
        vh_report = visual_hull_filter_mesh(
            mesh_path=Path(best["mesh_path"]),
            score_report=best,
            frames=frames,
            output_path=selected_dir / "selected_visual_hull_filtered.obj",
            inside_ratio_threshold=args.vh_inside_ratio_threshold,
            min_visible_views=args.vh_min_visible_views,
            mask_dilation=args.vh_mask_dilation,
        )
    summary = {
        "case_name": case_name,
        "dataset_workspace": str(dataset_workspace),
        "raw_dataset_workspace": str(raw_workspace),
        "pose_source": args.pose_source,
        "input_type": args.input_type,
        "recon_input_frame_count": len(selected_frames),
        "score_frame_count": len(frames),
        "frame_selection_report": str(output_root / "frame_selection_report.json"),
        "object_center": object_report_json,
        "candidate_count": len(scored),
        "best_mesh": best["mesh_path"],
        "selected_mesh": selected_mesh,
        "visual_hull_filter": vh_report,
        "best_score": {
            "score_mean": best["best"]["summary"]["score_mean"],
            "iou_mean": best["best"]["summary"]["iou_mean"],
            "recall_mean": best["best"]["summary"]["recall_mean"],
            "precision_mean": best["best"]["summary"]["precision_mean"],
            "scale": best["best"]["scale"],
            "yaw_deg": best["best"]["yaw_deg"],
            "candidate": best["candidate"],
        },
        "ranked_candidates": [
            {
                "mesh_path": item["mesh_path"],
                "candidate": item["candidate"],
                "score_mean": item["best"]["summary"]["score_mean"],
                "iou_mean": item["best"]["summary"]["iou_mean"],
                "recall_mean": item["best"]["summary"]["recall_mean"],
                "precision_mean": item["best"]["summary"]["precision_mean"],
                "scale": item["best"]["scale"],
                "yaw_deg": item["best"]["yaw_deg"],
            }
            for item in sorted(scored, key=lambda x: x["best"]["summary"]["score_mean"], reverse=True)
        ],
    }
    write_json(output_root / "pose_guided_summary.json", summary)
    print(json.dumps({"summary": str(output_root / "pose_guided_summary.json"), "selected_mesh": selected_mesh}, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
