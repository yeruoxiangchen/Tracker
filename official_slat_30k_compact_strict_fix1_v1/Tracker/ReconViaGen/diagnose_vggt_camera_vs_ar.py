#!/usr/bin/env python3
"""Compare VGGT camera-head predictions against recorded AR poses.

This is a diagnostic script for ReconViaGen AR tracker sessions. It uses the
same 518x518 segmented previews that ReconViaGen passes into VGGT/Trellis by
default, decodes VGGT cameras with camera_head, and compares them to the phone
AR poses after converting the phone poses to OpenCV/COLMAP camera coordinates.
"""

import argparse
import csv
import itertools
import json
import math
import os
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg_cache")
os.environ.setdefault("TORCH_HOME", os.path.expanduser("~/.cache/torch"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
for _cache_dir in [
    os.environ["NUMBA_CACHE_DIR"],
    os.environ["MPLCONFIGDIR"],
    os.environ["XDG_CACHE_HOME"],
    os.environ["TORCH_HOME"],
]:
    os.makedirs(_cache_dir, exist_ok=True)

import numpy as np
from PIL import Image
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VGGT_WHEEL_DIR = os.path.join(BASE_DIR, "wheels", "vggt")
if VGGT_WHEEL_DIR not in sys.path:
    sys.path.insert(0, VGGT_WHEEL_DIR)

from vggt.models.vggt import VGGT  # noqa: E402
from vggt.utils.pose_enc import pose_encoding_to_extri_intri  # noqa: E402


DEFAULT_SESSIONS = [
    "20260520_021333_413",
    "20260514_070952_645",
]
DEFAULT_VGGT_REPO = "Stable-X/vggt-object-v0-1"
DEFAULT_RESOLUTION = 518


def hf_snapshot_path(repo_id: str) -> str:
    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    model_dir = os.path.join(hf_home, "hub", f"models--{repo_id.replace('/', '--')}")
    refs_main = os.path.join(model_dir, "refs", "main")
    if os.path.exists(refs_main):
        with open(refs_main, "r") as f:
            revision = f.read().strip()
        snapshot = os.path.join(model_dir, "snapshots", revision)
        if os.path.isdir(snapshot):
            return snapshot

    snapshots_dir = os.path.join(model_dir, "snapshots")
    if os.path.isdir(snapshots_dir):
        snapshots = [
            os.path.join(snapshots_dir, name)
            for name in sorted(os.listdir(snapshots_dir))
            if os.path.isdir(os.path.join(snapshots_dir, name))
        ]
        if snapshots:
            return snapshots[-1]
    raise FileNotFoundError(f"Local Hugging Face snapshot not found for {repo_id}: {model_dir}")


def parse_float(value: str) -> Optional[float]:
    try:
        if value is None or str(value).strip() in {"", "None", "nan"}:
            return None
        return float(value)
    except ValueError:
        return None


def parse_int(value: str) -> Optional[int]:
    try:
        if value is None or str(value).strip() in {"", "None", "nan"}:
            return None
        return int(float(value))
    except ValueError:
        return None


def read_phone_poses(pose_path: str) -> Dict[str, dict]:
    poses = {}
    with open(pose_path, "r") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(",")]
            if parts[0] == "frame_name":
                continue
            if len(parts) < 7:
                print(f"[Pose] skip short row {line_no}: {line}")
                continue

            frame_name = parts[0]
            pos = [parse_float(v) for v in parts[1:4]]
            rot_deg = [parse_float(v) for v in parts[4:7]]
            if any(v is None for v in pos):
                print(f"[Pose] skip row without position: {frame_name}")
                continue
            pose = {
                "frame_name": frame_name,
                "pos": np.array(pos, dtype=np.float64),
                "rot_deg": None if any(v is None for v in rot_deg) else np.array(rot_deg, dtype=np.float64),
                "quat": None,
                "intrinsics": None,
                "image_transform": "unknown",
            }

            if len(parts) >= 11:
                quat = [parse_float(v) for v in parts[7:11]]
                if not any(v is None for v in quat):
                    pose["quat"] = np.array(quat, dtype=np.float64)

            if len(parts) >= 15:
                fx, fy, cx, cy = [parse_float(v) for v in parts[11:15]]
                if None not in (fx, fy, cx, cy):
                    pose["intrinsics"] = {
                        "fx": fx,
                        "fy": fy,
                        "cx": cx,
                        "cy": cy,
                        "width": parse_int(parts[15]) if len(parts) > 15 else None,
                        "height": parse_int(parts[16]) if len(parts) > 16 else None,
                        "image_width": parse_int(parts[17]) if len(parts) > 17 else None,
                        "image_height": parse_int(parts[18]) if len(parts) > 18 else None,
                        "cpu_image_width": parse_int(parts[19]) if len(parts) > 19 else None,
                        "cpu_image_height": parse_int(parts[20]) if len(parts) > 20 else None,
                    }
            if len(parts) > 21:
                pose["image_transform"] = parts[21] or "None"
            elif len(parts) >= 19:
                pose["image_transform"] = "MirrorY"
            poses[frame_name] = pose

    if not poses:
        raise ValueError(f"No valid phone poses parsed from {pose_path}")
    return poses


def unity_quat_to_rotmat(quat_xyzw: np.ndarray) -> np.ndarray:
    q = np.asarray(quat_xyzw, dtype=np.float64)
    norm = np.linalg.norm(q)
    if norm < 1e-12:
        raise ValueError("Unity quaternion has zero norm")
    x, y, z, w = q / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def unity_euler_to_rotmat(rot_deg: np.ndarray) -> np.ndarray:
    rx, ry, rz = np.deg2rad(rot_deg)
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
        u[:, -1] *= -1.0
        fixed = u @ vt
    return fixed


def phone_pose_to_opencv_w2c(pose: dict) -> np.ndarray:
    if pose.get("quat") is not None:
        r_unity_c2w = unity_quat_to_rotmat(pose["quat"])
    elif pose.get("rot_deg") is not None:
        r_unity_c2w = unity_euler_to_rotmat(pose["rot_deg"])
    else:
        raise ValueError(f"No rotation found for {pose.get('frame_name', '<unknown>')}")

    unity_to_cv_world = np.diag([1.0, 1.0, -1.0])
    unity_cam_to_cv_cam = np.diag([1.0, -1.0, 1.0])
    cam_center_w = unity_to_cv_world @ pose["pos"]
    r_cv_c2w = unity_to_cv_world @ r_unity_c2w @ unity_cam_to_cv_cam
    r_cv_c2w = nearest_rotation_matrix(r_cv_c2w)
    r_w2c = r_cv_c2w.T
    t_w2c = -r_w2c @ cam_center_w

    # Match the saved XRCpuImage pixel frame used by CoarseModel/server.py.
    image_cam_from_pose_cam = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    r_w2c = image_cam_from_pose_cam @ r_w2c
    t_w2c = image_cam_from_pose_cam @ t_w2c

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = nearest_rotation_matrix(r_w2c)
    T[:3, 3] = t_w2c
    return T


def intrinsics_for_pose(pose: dict, image_width: int, image_height: int) -> Tuple[np.ndarray, str]:
    """Return AR intrinsics in the saved raw image pixel frame.

    This mirrors CoarseModel/connect/server.py::_intrinsics_for_pose so that the
    diagnostic uses the same raw-image camera model as the pose-scale pipeline.
    """
    intr = pose.get("intrinsics")
    source = "fallback_image_size"
    if intr is not None:
        fx, fy, cx, cy = intr["fx"], intr["fy"], intr["cx"], intr["cy"]
        src_w = intr.get("cpu_image_width") or intr.get("width") or intr.get("image_width") or image_width
        src_h = intr.get("cpu_image_height") or intr.get("height") or intr.get("image_height") or image_height
        if src_w and src_h and src_w > 0 and src_h > 0:
            scale_x = image_width / float(src_w)
            scale_y = image_height / float(src_h)
            fx *= scale_x
            fy *= scale_y
            cx *= scale_x
            cy *= scale_y

        image_transform = str(pose.get("image_transform") or "None").lower()
        if "mirrorx" in image_transform:
            cx = (image_width - 1) - cx
        if "mirrory" in image_transform:
            cy = (image_height - 1) - cy

        if fx > 0 and fy > 0 and 0 <= cx <= image_width and 0 <= cy <= image_height:
            source = f"ar_foundation_{pose.get('image_transform') or 'None'}"
            return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64), source

    focal = float(max(image_width, image_height))
    return (
        np.array(
            [
                [focal, 0.0, (image_width - 1) * 0.5],
                [0.0, focal, (image_height - 1) * 0.5],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        ),
        source,
    )


def crop_meta_from_mask(image_path: str, mask_path: str, resolution: int) -> dict:
    """Recompute the exact preview crop/pad/resize metadata from raw image+mask."""
    image = Image.open(image_path).convert("RGBA")
    mask = Image.open(mask_path).convert("L")
    if mask.size != image.size:
        mask = mask.resize(image.size, Image.Resampling.NEAREST)

    alpha = np.array(mask)
    ys, xs = np.nonzero(alpha > 127)
    if len(xs) == 0:
        return {
            "empty_mask": True,
            "original_size": [int(image.width), int(image.height)],
            "crop_box": [0, 0, int(image.width), int(image.height)],
            "pad": [0, 0],
            "side": int(max(image.width, image.height)),
            "scale": float(resolution / float(max(image.width, image.height))),
        }

    left, right = xs.min(), xs.max()
    top, bottom = ys.min(), ys.max()
    center_x = 0.5 * (left + right)
    center_y = 0.5 * (top + bottom)
    size = int(max(right - left, bottom - top) * 1.1)
    size = max(size, 1)
    crop_box = (
        max(0, int(center_x - size // 2)),
        max(0, int(center_y - size // 2)),
        min(image.width, int(center_x + size // 2)),
        min(image.height, int(center_y + size // 2)),
    )
    crop_w = int(crop_box[2] - crop_box[0])
    crop_h = int(crop_box[3] - crop_box[1])
    side = max(crop_w, crop_h)
    pad_x = (side - crop_w) // 2
    pad_y = (side - crop_h) // 2
    return {
        "empty_mask": False,
        "original_size": [int(image.width), int(image.height)],
        "crop_box": [int(v) for v in crop_box],
        "pad": [int(pad_x), int(pad_y)],
        "side": int(side),
        "scale": float(resolution / float(side)),
    }


def adjust_K_for_preview_crop(K_raw: np.ndarray, crop_meta: dict, resolution: int) -> np.ndarray:
    """Map raw-image K into the 518x518 cropped preview coordinate frame."""
    K = np.array(K_raw, dtype=np.float64, copy=True)
    left, top, _right, _bottom = crop_meta["crop_box"]
    pad_x, pad_y = crop_meta["pad"]
    scale = float(crop_meta["scale"])
    K[0, 2] = (K[0, 2] - float(left) + float(pad_x)) * scale
    K[1, 2] = (K[1, 2] - float(top) + float(pad_y)) * scale
    K[0, 0] *= scale
    K[1, 1] *= scale
    return K


def adjust_K_for_square_resize(K_raw: np.ndarray, image_width: int, image_height: int, resolution: int) -> np.ndarray:
    """Map raw-image K into VGGT's square-resized raw input coordinate frame."""
    K = np.array(K_raw, dtype=np.float64, copy=True)
    K[0, 0] *= resolution / float(image_width)
    K[1, 1] *= resolution / float(image_height)
    K[0, 2] *= resolution / float(image_width)
    K[1, 2] *= resolution / float(image_height)
    return K


def fov_degrees_from_K(K: np.ndarray, height: int = DEFAULT_RESOLUTION, width: int = DEFAULT_RESOLUTION) -> Tuple[float, float]:
    fov_h = 2.0 * math.atan((height / 2.0) / float(K[1, 1]))
    fov_w = 2.0 * math.atan((width / 2.0) / float(K[0, 0]))
    return float(math.degrees(fov_h)), float(math.degrees(fov_w))


def list_frame_names(data_dir: str) -> List[str]:
    return sorted(
        name
        for name in os.listdir(data_dir)
        if name.lower().endswith((".jpg", ".jpeg", ".png"))
    )


def load_input_image(data_dir: str, preview_dir: str, frame_name: str, frame_index: int, source: str) -> Image.Image:
    if source == "preview":
        preview_path = os.path.join(preview_dir, f"{frame_index}.png")
        if not os.path.exists(preview_path):
            raise FileNotFoundError(f"Missing preview image: {preview_path}")
        rgba = np.array(Image.open(preview_path).convert("RGBA"))
        rgb = rgba[:, :, :3].copy()
        rgb[rgba[:, :, 3] <= 16] = 0
        return Image.fromarray(rgb, mode="RGB")

    image_path = os.path.join(data_dir, frame_name)
    return Image.open(image_path).convert("RGB")


def image_quality_score(data_dir: str, preview_dir: str, frame_name: str, frame_index: int) -> float:
    coverage_score = 0.5
    preview_path = os.path.join(preview_dir, f"{frame_index}.png")
    if os.path.exists(preview_path):
        arr = np.array(Image.open(preview_path).convert("RGBA"))
        mask = arr[:, :, 3] > 128
        coverage = float(mask.mean()) if mask.size else 0.0
        coverage_score = max(0.0, 1.0 - abs(coverage - 0.45) / 0.45)

    image_path = os.path.join(data_dir, frame_name)
    gray = np.array(Image.open(image_path).convert("L"), dtype=np.float32)
    gy, gx = np.gradient(gray)
    sharpness = float(np.var(gx) + np.var(gy))
    sharpness_score = min(sharpness / 800.0, 1.0)
    return 0.65 * coverage_score + 0.35 * sharpness_score


def parse_frame_indices(text: Optional[str]) -> Optional[List[int]]:
    if not text:
        return None
    result = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        result.append(int(item))
    return result


def select_frame_indices(
    frame_names: Sequence[str],
    poses: Dict[str, dict],
    data_dir: str,
    preview_dir: str,
    max_frames: int,
    explicit_indices: Optional[List[int]],
) -> Tuple[List[int], dict]:
    valid = [idx for idx, name in enumerate(frame_names) if name in poses]
    if explicit_indices is not None:
        explicit = sorted({idx for idx in explicit_indices if idx in valid})
        return explicit, {"enabled": True, "reason": "explicit_indices", "selected_indices": explicit}

    if max_frames <= 0 or len(valid) <= max_frames:
        return valid, {"enabled": False, "reason": "all_valid_frames", "input_count": len(valid)}

    pose_candidates = [(idx, poses[frame_names[idx]]["pos"]) for idx in valid]
    positions = np.stack([item[1] for item in pose_candidates], axis=0)
    centroid = positions.mean(axis=0)
    angle_by_idx = {}
    for idx, pos in pose_candidates:
        angle_by_idx[idx] = (np.degrees(np.arctan2(pos[2] - centroid[2], pos[0] - centroid[0])) + 360.0) % 360.0

    bin_size = 360.0 / float(max_frames)
    bins = {}
    for idx in valid:
        bin_id = int(angle_by_idx[idx] // bin_size)
        quality = image_quality_score(data_dir, preview_dir, frame_names[idx], idx)
        current = bins.get(bin_id)
        if current is None or quality > current[1]:
            bins[bin_id] = (idx, quality)

    chosen = [item[0] for item in bins.values()]
    if len(chosen) < max_frames:
        remaining = [idx for idx in valid if idx not in chosen]
        remaining = sorted(
            remaining,
            key=lambda idx: image_quality_score(data_dir, preview_dir, frame_names[idx], idx),
            reverse=True,
        )
        chosen.extend(remaining[: max_frames - len(chosen)])
    elif len(chosen) > max_frames:
        chosen = sorted(
            chosen,
            key=lambda idx: image_quality_score(data_dir, preview_dir, frame_names[idx], idx),
            reverse=True,
        )[:max_frames]

    chosen = sorted(chosen)
    return chosen, {
        "enabled": True,
        "reason": "angle_balanced_quality",
        "input_count": len(valid),
        "output_count": len(chosen),
        "max_frames": max_frames,
        "selected_indices": chosen,
    }


def load_vggt_model(device: torch.device, model_path: Optional[str]) -> VGGT:
    model_path = model_path or hf_snapshot_path(DEFAULT_VGGT_REPO)
    print(f"[VGGT] loading camera diagnostic model from {model_path}")
    model = VGGT.from_pretrained(model_path).to(device)
    # Keep only aggregator + camera_head for this diagnostic.
    if hasattr(model, "depth_head"):
        del model.depth_head
    if hasattr(model, "point_head"):
        del model.point_head
    if hasattr(model, "track_head"):
        del model.track_head
    model.eval()
    return model


def predict_vggt_cameras(
    model: VGGT,
    images: List[Image.Image],
    device: torch.device,
    resolution: int,
) -> Tuple[np.ndarray, np.ndarray]:
    tensors = []
    for image in images:
        resized = image.resize((resolution, resolution), Image.Resampling.LANCZOS)
        arr = np.array(resized).astype(np.float32) / 255.0
        tensors.append(torch.from_numpy(arr).permute(2, 0, 1))
    image_tensor = torch.stack(tensors, dim=0).to(device)

    use_cuda = device.type == "cuda"
    autocast_dtype = torch.bfloat16
    if use_cuda and torch.cuda.get_device_capability(device)[0] < 8:
        autocast_dtype = torch.float16

    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=use_cuda, dtype=autocast_dtype):
            aggregated_tokens_list, _patch_start_idx = model.aggregator(image_tensor[None])
        with torch.cuda.amp.autocast(enabled=False):
            pose_enc = model.camera_head(aggregated_tokens_list)[-1]
            extrinsics, intrinsics = pose_encoding_to_extri_intri(
                pose_enc.float(),
                image_tensor.shape[-2:],
            )

    T = np.tile(np.eye(4, dtype=np.float64), (len(images), 1, 1))
    T[:, :3, :] = extrinsics[0].detach().cpu().numpy().astype(np.float64)
    K = intrinsics[0].detach().cpu().numpy().astype(np.float64)
    return T, K


def invert_T(T: np.ndarray) -> np.ndarray:
    inv = np.eye(4, dtype=np.float64)
    R = T[:3, :3]
    t = T[:3, 3]
    inv[:3, :3] = R.T
    inv[:3, 3] = -R.T @ t
    return inv


def camera_centers(T_w2c: np.ndarray) -> np.ndarray:
    R = T_w2c[:, :3, :3]
    t = T_w2c[:, :3, 3]
    return -np.einsum("nij,nj->ni", np.transpose(R, (0, 2, 1)), t)


def rotation_angle_deg(R: np.ndarray) -> float:
    value = (np.trace(R) - 1.0) / 2.0
    value = float(np.clip(value, -1.0, 1.0))
    return float(np.degrees(np.arccos(value)))


def vector_angle_deg(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-9 or nb < 1e-9:
        return None
    value = float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))
    return float(np.degrees(np.arccos(value)))


def umeyama_sim3(src: np.ndarray, dst: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    """Return s, R, t so that dst ~= s * R @ src + t."""
    if len(src) < 3:
        raise ValueError("At least 3 camera centers are required for Sim(3) alignment")
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    X = src - mu_src
    Y = dst - mu_dst
    cov = (Y.T @ X) / float(len(src))
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        S[-1, -1] = -1.0
    R = U @ S @ Vt
    var_src = float(np.sum(X * X) / float(len(src)))
    scale = float(np.sum(D * np.diag(S)) / max(var_src, 1e-12))
    t = mu_dst - scale * (R @ mu_src)
    return scale, R, t


def estimate_camera_frame_correction(source_c2w: np.ndarray, target_c2w: np.ndarray) -> np.ndarray:
    """Estimate Q so target_c2w ~= source_c2w @ Q.

    Sim(3) center alignment fixes the world-frame rotation. A remaining
    constant right-multiplied rotation usually means the two systems use
    different camera-axis conventions.
    """
    if len(source_c2w) != len(target_c2w):
        raise ValueError("source_c2w and target_c2w must have the same length")
    M = np.zeros((3, 3), dtype=np.float64)
    for source_R, target_R in zip(source_c2w, target_c2w):
        M += source_R.T @ target_R
    U, _S, Vt = np.linalg.svd(M)
    Q = U @ Vt
    if np.linalg.det(Q) < 0:
        U[:, -1] *= -1.0
        Q = U @ Vt
    return nearest_rotation_matrix(Q)


def c2w_rotation_errors(source_c2w: np.ndarray, target_c2w: np.ndarray) -> List[float]:
    return [
        rotation_angle_deg(target_R.T @ source_R)
        for source_R, target_R in zip(source_c2w, target_c2w)
    ]


def signed_permutation_rotations() -> List[np.ndarray]:
    rotations = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product([-1.0, 1.0], repeat=3):
            Q = np.zeros((3, 3), dtype=np.float64)
            for col, row in enumerate(perm):
                Q[row, col] = signs[col]
            if np.linalg.det(Q) > 0.0:
                rotations.append(Q)
    return rotations


def best_axis_permutation_correction(
    source_c2w: np.ndarray,
    target_c2w: np.ndarray,
) -> Tuple[np.ndarray, List[float]]:
    best_Q = np.eye(3, dtype=np.float64)
    best_errors = c2w_rotation_errors(source_c2w, target_c2w)
    best_median = float(np.median(best_errors)) if best_errors else float("inf")
    for Q in signed_permutation_rotations():
        errors = c2w_rotation_errors(np.einsum("nij,jk->nik", source_c2w, Q), target_c2w)
        median = float(np.median(errors)) if errors else float("inf")
        if median < best_median:
            best_Q = Q
            best_errors = errors
            best_median = median
    return best_Q, best_errors


def pairwise_rotation_errors_from_c2w(
    ar_c2w: np.ndarray,
    vggt_c2w: np.ndarray,
    consecutive_only: bool,
) -> List[dict]:
    n = len(ar_c2w)
    iterator = [(i, i + 1) for i in range(n - 1)] if consecutive_only else [
        (i, j) for i in range(n) for j in range(i + 1, n)
    ]
    rows = []
    for i, j in iterator:
        rel_ar = ar_c2w[i].T @ ar_c2w[j]
        rel_vggt = vggt_c2w[i].T @ vggt_c2w[j]
        rows.append(
            {
                "i": i,
                "j": j,
                "relative_rotation_error_after_camera_axis_correction_deg": rotation_angle_deg(
                    rel_ar.T @ rel_vggt
                ),
            }
        )
    return rows


def summarize_values(values: Sequence[float]) -> dict:
    vals = np.array([v for v in values if v is not None and np.isfinite(v)], dtype=np.float64)
    if len(vals) == 0:
        return {"count": 0, "mean": None, "median": None, "p90": None, "max": None}
    return {
        "count": int(len(vals)),
        "mean": float(vals.mean()),
        "median": float(np.median(vals)),
        "p90": float(np.percentile(vals, 90)),
        "max": float(vals.max()),
    }


def pairwise_errors(T_ar: np.ndarray, T_vggt: np.ndarray, scale: float, consecutive_only: bool) -> List[dict]:
    n = len(T_ar)
    pairs = []
    iterator = [(i, i + 1) for i in range(n - 1)] if consecutive_only else [
        (i, j) for i in range(n) for j in range(i + 1, n)
    ]
    for i, j in iterator:
        rel_ar = T_ar[j] @ invert_T(T_ar[i])
        rel_v = T_vggt[j] @ invert_T(T_vggt[i])
        rot_err = rotation_angle_deg(rel_v[:3, :3] @ rel_ar[:3, :3].T)
        tdir_err = vector_angle_deg(rel_ar[:3, 3], rel_v[:3, 3])
        ar_baseline = float(np.linalg.norm(rel_ar[:3, 3]))
        vggt_baseline = float(np.linalg.norm(rel_v[:3, 3]))
        pairs.append(
            {
                "i": i,
                "j": j,
                "relative_rotation_error_deg": rot_err,
                "relative_translation_dir_error_deg": tdir_err,
                "ar_relative_baseline": ar_baseline,
                "vggt_relative_baseline": vggt_baseline,
                "sim3_scaled_baseline_ratio": (scale * vggt_baseline / ar_baseline) if ar_baseline > 1e-9 else None,
            }
        )
    return pairs


def write_csv(path: str, rows: List[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def diagnose_session(args, model: VGGT, device: torch.device, session_arg: str) -> dict:
    data_root = os.path.abspath(args.data_root)
    preview_root = os.path.abspath(args.preview_root)
    mask_root = os.path.abspath(args.mask_root)
    if os.path.isdir(session_arg):
        data_dir = os.path.abspath(session_arg)
        session_id = os.path.basename(data_dir.rstrip(os.sep))
    else:
        session_id = session_arg
        data_dir = os.path.join(data_root, session_id)
    preview_dir = os.path.join(preview_root, session_id)
    mask_dir = os.path.join(mask_root, session_id)
    pose_path = os.path.join(data_dir, "poses.txt")
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Session data directory not found: {data_dir}")
    if args.source == "preview" and not os.path.isdir(preview_dir):
        raise FileNotFoundError(f"Preview directory not found: {preview_dir}")
    if args.source == "preview" and not os.path.isdir(mask_dir):
        raise FileNotFoundError(f"Mask directory not found: {mask_dir}")
    if not os.path.exists(pose_path):
        raise FileNotFoundError(f"Pose file not found: {pose_path}")

    frame_names = list_frame_names(data_dir)
    poses = read_phone_poses(pose_path)
    selected_indices, selection_report = select_frame_indices(
        frame_names,
        poses,
        data_dir,
        preview_dir,
        args.max_frames,
        parse_frame_indices(args.frame_indices),
    )
    selected_names = [frame_names[idx] for idx in selected_indices]
    if len(selected_names) < 3:
        raise RuntimeError(f"Need at least 3 selected frames for Sim(3) diagnostics, got {len(selected_names)}")

    print(f"[{session_id}] selected {len(selected_names)} frames: {selected_indices}")
    images = [
        load_input_image(data_dir, preview_dir, name, idx, args.source)
        for idx, name in zip(selected_indices, selected_names)
    ]
    T_ar = np.stack([phone_pose_to_opencv_w2c(poses[name]) for name in selected_names], axis=0)
    T_vggt, K_vggt = predict_vggt_cameras(model, images, device, args.resolution)
    K_ar_input = []
    ar_intrinsic_sources = []
    crop_metas = []
    for frame_idx, name in zip(selected_indices, selected_names):
        image_path = os.path.join(data_dir, name)
        raw_image = Image.open(image_path)
        K_raw, intrinsic_source = intrinsics_for_pose(poses[name], raw_image.width, raw_image.height)
        if args.source == "preview":
            mask_path = os.path.join(mask_dir, f"{os.path.splitext(name)[0]}.png")
            if not os.path.exists(mask_path):
                raise FileNotFoundError(f"Mask for preview crop metadata not found: {mask_path}")
            crop_meta = crop_meta_from_mask(image_path, mask_path, args.resolution)
            K_input = adjust_K_for_preview_crop(K_raw, crop_meta, args.resolution)
        else:
            crop_meta = {
                "empty_mask": None,
                "original_size": [int(raw_image.width), int(raw_image.height)],
                "crop_box": [0, 0, int(raw_image.width), int(raw_image.height)],
                "pad": [0, 0],
                "side": None,
                "scale": None,
            }
            K_input = adjust_K_for_square_resize(K_raw, raw_image.width, raw_image.height, args.resolution)
        K_ar_input.append(K_input)
        ar_intrinsic_sources.append(intrinsic_source)
        crop_metas.append(crop_meta)
    K_ar_input = np.stack(K_ar_input, axis=0)

    C_ar = camera_centers(T_ar)
    C_vggt = camera_centers(T_vggt)
    scale, R_align, t_align = umeyama_sim3(C_vggt, C_ar)
    C_vggt_aligned = scale * (R_align @ C_vggt.T).T + t_align
    center_errors = np.linalg.norm(C_vggt_aligned - C_ar, axis=1)
    ar_radius = float(np.median(np.linalg.norm(C_ar - C_ar.mean(axis=0), axis=1)))
    ar_radius = max(ar_radius, 1e-9)

    R_ar_c2w_all = np.transpose(T_ar[:, :3, :3], (0, 2, 1))
    R_vggt_c2w_all = np.transpose(T_vggt[:, :3, :3], (0, 2, 1))
    R_vggt_c2w_aligned_all = np.einsum("ij,njk->nik", R_align, R_vggt_c2w_all)
    Q_cam = estimate_camera_frame_correction(R_vggt_c2w_aligned_all, R_ar_c2w_all)
    R_vggt_c2w_camcorr_all = np.einsum("nij,jk->nik", R_vggt_c2w_aligned_all, Q_cam)
    Q_axis, rot_errors_axis = best_axis_permutation_correction(R_vggt_c2w_aligned_all, R_ar_c2w_all)
    R_vggt_c2w_axiscorr_all = np.einsum("nij,jk->nik", R_vggt_c2w_aligned_all, Q_axis)

    rot_errors = c2w_rotation_errors(R_vggt_c2w_aligned_all, R_ar_c2w_all)
    rot_errors_camcorr = c2w_rotation_errors(R_vggt_c2w_camcorr_all, R_ar_c2w_all)
    per_frame_rows = []
    for local_idx, (frame_idx, name) in enumerate(zip(selected_indices, selected_names)):
        rot_err = rot_errors[local_idx]
        rot_err_camcorr = rot_errors_camcorr[local_idx]
        rot_err_axiscorr = rot_errors_axis[local_idx]
        ar_fov_h, ar_fov_w = fov_degrees_from_K(K_ar_input[local_idx], args.resolution, args.resolution)
        vggt_fov_h, vggt_fov_w = fov_degrees_from_K(K_vggt[local_idx], args.resolution, args.resolution)
        crop_meta = crop_metas[local_idx]
        per_frame_rows.append(
            {
                "local_index": local_idx,
                "frame_index": frame_idx,
                "frame_name": name,
                "center_error_after_sim3": float(center_errors[local_idx]),
                "center_error_normalized": float(center_errors[local_idx] / ar_radius),
                "rotation_error_after_sim3_deg": float(rot_err),
                "rotation_error_after_camera_axis_correction_deg": float(rot_err_camcorr),
                "rotation_error_after_best_axis_permutation_deg": float(rot_err_axiscorr),
                "ar_center_x": float(C_ar[local_idx, 0]),
                "ar_center_y": float(C_ar[local_idx, 1]),
                "ar_center_z": float(C_ar[local_idx, 2]),
                "vggt_center_x": float(C_vggt[local_idx, 0]),
                "vggt_center_y": float(C_vggt[local_idx, 1]),
                "vggt_center_z": float(C_vggt[local_idx, 2]),
                "vggt_aligned_center_x": float(C_vggt_aligned[local_idx, 0]),
                "vggt_aligned_center_y": float(C_vggt_aligned[local_idx, 1]),
                "vggt_aligned_center_z": float(C_vggt_aligned[local_idx, 2]),
                "ar_intrinsic_source": ar_intrinsic_sources[local_idx],
                "ar_fx_input": float(K_ar_input[local_idx, 0, 0]),
                "ar_fy_input": float(K_ar_input[local_idx, 1, 1]),
                "ar_cx_input": float(K_ar_input[local_idx, 0, 2]),
                "ar_cy_input": float(K_ar_input[local_idx, 1, 2]),
                "vggt_fx_518": float(K_vggt[local_idx, 0, 0]),
                "vggt_fy_518": float(K_vggt[local_idx, 1, 1]),
                "vggt_cx_518": float(K_vggt[local_idx, 0, 2]),
                "vggt_cy_518": float(K_vggt[local_idx, 1, 2]),
                "vggt_fx_over_ar_fx": float(K_vggt[local_idx, 0, 0] / K_ar_input[local_idx, 0, 0]),
                "vggt_fy_over_ar_fy": float(K_vggt[local_idx, 1, 1] / K_ar_input[local_idx, 1, 1]),
                "ar_fov_h_deg": ar_fov_h,
                "ar_fov_w_deg": ar_fov_w,
                "vggt_fov_h_deg": vggt_fov_h,
                "vggt_fov_w_deg": vggt_fov_w,
                "crop_left": crop_meta["crop_box"][0],
                "crop_top": crop_meta["crop_box"][1],
                "crop_right": crop_meta["crop_box"][2],
                "crop_bottom": crop_meta["crop_box"][3],
                "crop_pad_x": crop_meta["pad"][0],
                "crop_pad_y": crop_meta["pad"][1],
                "crop_scale": crop_meta["scale"],
            }
        )

    consecutive = pairwise_errors(T_ar, T_vggt, scale, consecutive_only=True)
    all_pairs = pairwise_errors(T_ar, T_vggt, scale, consecutive_only=False)
    consecutive_camcorr = pairwise_rotation_errors_from_c2w(
        R_ar_c2w_all, R_vggt_c2w_camcorr_all, consecutive_only=True
    )
    all_pairs_camcorr = pairwise_rotation_errors_from_c2w(
        R_ar_c2w_all, R_vggt_c2w_camcorr_all, consecutive_only=False
    )
    output_dir = os.path.join(args.output_root, f"{session_id}_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(output_dir, exist_ok=True)
    write_csv(os.path.join(output_dir, "per_frame.csv"), per_frame_rows)
    write_csv(os.path.join(output_dir, "consecutive_pairs.csv"), consecutive)
    write_csv(os.path.join(output_dir, "consecutive_pairs_camera_axis_corrected.csv"), consecutive_camcorr)
    if args.write_all_pairs:
        write_csv(os.path.join(output_dir, "all_pairs.csv"), all_pairs)
        write_csv(os.path.join(output_dir, "all_pairs_camera_axis_corrected.csv"), all_pairs_camcorr)

    summary = {
        "session_id": session_id,
        "data_dir": data_dir,
        "preview_dir": preview_dir if args.source == "preview" else None,
        "mask_dir": mask_dir if args.source == "preview" else None,
        "source": args.source,
        "resolution": args.resolution,
        "selected_indices": selected_indices,
        "selected_names": selected_names,
        "selection": selection_report,
        "sim3_vggt_to_ar": {
            "scale": scale,
            "rotation": R_align.tolist(),
            "translation": t_align.tolist(),
        },
        "camera_frame_correction_vggt_to_ar": {
            "rotation": Q_cam.tolist(),
            "best_signed_axis_permutation": Q_axis.tolist(),
            "description": "Apply as R_ar_c2w ~= R_align @ R_vggt_c2w @ rotation.",
        },
        "metrics": {
            "frame_count": len(selected_names),
            "ar_scene_radius_median": ar_radius,
            "center_error_after_sim3": summarize_values(center_errors.tolist()),
            "center_error_after_sim3_normalized": summarize_values((center_errors / ar_radius).tolist()),
            "rotation_error_after_sim3_deg": summarize_values(rot_errors),
            "rotation_error_after_camera_axis_correction_deg": summarize_values(rot_errors_camcorr),
            "rotation_error_after_best_axis_permutation_deg": summarize_values(rot_errors_axis),
            "consecutive_relative_rotation_error_deg": summarize_values(
                [p["relative_rotation_error_deg"] for p in consecutive]
            ),
            "consecutive_relative_rotation_error_after_camera_axis_correction_deg": summarize_values(
                [p["relative_rotation_error_after_camera_axis_correction_deg"] for p in consecutive_camcorr]
            ),
            "consecutive_relative_translation_dir_error_deg": summarize_values(
                [p["relative_translation_dir_error_deg"] for p in consecutive]
            ),
            "all_pairs_relative_rotation_error_deg": summarize_values(
                [p["relative_rotation_error_deg"] for p in all_pairs]
            ),
            "all_pairs_relative_rotation_error_after_camera_axis_correction_deg": summarize_values(
                [p["relative_rotation_error_after_camera_axis_correction_deg"] for p in all_pairs_camcorr]
            ),
            "all_pairs_relative_translation_dir_error_deg": summarize_values(
                [p["relative_translation_dir_error_deg"] for p in all_pairs]
            ),
            "all_pairs_sim3_scaled_baseline_ratio": summarize_values(
                [p["sim3_scaled_baseline_ratio"] for p in all_pairs]
            ),
            "vggt_fx_518": summarize_values(K_vggt[:, 0, 0].tolist()),
            "vggt_fy_518": summarize_values(K_vggt[:, 1, 1].tolist()),
            "ar_fx_input": summarize_values(K_ar_input[:, 0, 0].tolist()),
            "ar_fy_input": summarize_values(K_ar_input[:, 1, 1].tolist()),
            "ar_cx_input": summarize_values(K_ar_input[:, 0, 2].tolist()),
            "ar_cy_input": summarize_values(K_ar_input[:, 1, 2].tolist()),
            "vggt_fx_over_ar_fx": summarize_values((K_vggt[:, 0, 0] / K_ar_input[:, 0, 0]).tolist()),
            "vggt_fy_over_ar_fy": summarize_values((K_vggt[:, 1, 1] / K_ar_input[:, 1, 1]).tolist()),
            "vggt_fov_h_deg": summarize_values([fov_degrees_from_K(k, args.resolution, args.resolution)[0] for k in K_vggt]),
            "vggt_fov_w_deg": summarize_values([fov_degrees_from_K(k, args.resolution, args.resolution)[1] for k in K_vggt]),
            "ar_fov_h_deg": summarize_values([fov_degrees_from_K(k, args.resolution, args.resolution)[0] for k in K_ar_input]),
            "ar_fov_w_deg": summarize_values([fov_degrees_from_K(k, args.resolution, args.resolution)[1] for k in K_ar_input]),
        },
        "notes": [
            "AR poses are converted with the same Unity-to-OpenCV image-frame convention used by CoarseModel/connect/server.py.",
            "For source=preview, AR intrinsics are transformed by the same mask crop, square padding, and 518x518 resize used to write previews.",
            "For source=raw, AR intrinsics are transformed by the same direct square resize used before feeding raw frames to VGGT.",
            "A fixed right-multiplied camera-frame correction is estimated after Sim(3) center alignment to diagnose VGGT-vs-AR camera-axis convention mismatch.",
            "camera_head error diagnoses VGGT geometry tokens; mesh quality can still be limited by Trellis sparse structure and generative priors.",
        ],
        "output_dir": output_dir,
    }
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[{session_id}] output: {output_dir}")
    print(
        f"[{session_id}] Sim3 center median="
        f"{summary['metrics']['center_error_after_sim3']['median']:.6f}, "
        f"norm median={summary['metrics']['center_error_after_sim3_normalized']['median']:.4f}, "
        f"rot median={summary['metrics']['rotation_error_after_sim3_deg']['median']:.2f} deg, "
        f"cam-axis rot median="
        f"{summary['metrics']['rotation_error_after_camera_axis_correction_deg']['median']:.2f} deg"
    )
    print(
        f"[{session_id}] pair rot median="
        f"{summary['metrics']['all_pairs_relative_rotation_error_deg']['median']:.2f} deg, "
        f"cam-axis pair rot median="
        f"{summary['metrics']['all_pairs_relative_rotation_error_after_camera_axis_correction_deg']['median']:.2f} deg, "
        f"pair trans-dir median="
        f"{summary['metrics']['all_pairs_relative_translation_dir_error_deg']['median']:.2f} deg"
    )
    print(
        f"[{session_id}] focal ratio median="
        f"fx {summary['metrics']['vggt_fx_over_ar_fx']['median']:.3f}, "
        f"fy {summary['metrics']['vggt_fy_over_ar_fy']['median']:.3f}"
    )
    return summary


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Decode VGGT camera_head cameras and compare them against AR tracker poses."
    )
    parser.add_argument("sessions", nargs="*", default=DEFAULT_SESSIONS)
    parser.add_argument("--data-root", default=os.path.join(BASE_DIR, "ar_tracker", "data"))
    parser.add_argument("--preview-root", default=os.path.join(BASE_DIR, "ar_tracker", "previews"))
    parser.add_argument("--mask-root", default=os.path.join(BASE_DIR, "ar_tracker", "masks"))
    parser.add_argument("--output-root", default=os.path.join(BASE_DIR, "ar_tracker", "vggt_camera_diagnostics"))
    parser.add_argument("--source", choices=["preview", "raw"], default="preview")
    parser.add_argument("--resolution", type=int, default=518)
    parser.add_argument("--max-frames", type=int, default=18, help="0 means use all valid frames.")
    parser.add_argument("--frame-indices", default=None, help="Comma-separated original frame indices, e.g. 0,3,8.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--vggt-model-path", default=None)
    parser.add_argument("--write-all-pairs", action="store_true")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    args.output_root = os.path.abspath(args.output_root)
    os.makedirs(args.output_root, exist_ok=True)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Use --device cpu only for a very slow smoke test.")
    device = torch.device(args.device)
    model = load_vggt_model(device, args.vggt_model_path)

    reports = []
    for session in args.sessions:
        reports.append(diagnose_session(args, model, device, session))

    combined_path = os.path.join(args.output_root, f"combined_{time.strftime('%Y%m%d_%H%M%S')}.json")
    with open(combined_path, "w") as f:
        json.dump(reports, f, indent=2)
    print(f"[Done] combined report: {combined_path}")


if __name__ == "__main__":
    main()
