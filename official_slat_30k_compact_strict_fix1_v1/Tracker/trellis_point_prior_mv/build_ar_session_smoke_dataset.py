#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

TRACKER_ROOT = Path(__file__).resolve().parents[1]
if str(TRACKER_ROOT) not in sys.path:
    sys.path.insert(0, str(TRACKER_ROOT))

from trellis_point_prior_mv.common import write_json  # noqa: E402


def safe_name(text: str) -> str:
    text = str(text).strip()
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("._") or "ar_session"


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def parse_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = float(text)
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def parse_int(value: Any) -> int | None:
    parsed = parse_float(value)
    if parsed is None:
        return None
    return int(round(parsed))


def read_current_session(root: Path) -> dict[str, Any] | None:
    candidates = [
        root / "trellis_point_prior_mv" / "ar_tracker" / "flags" / "current_session.json",
        root / "ReconViaGen" / "ar_tracker" / "flags" / "current_session.json",
    ]
    for path in candidates:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    return None


def resolve_session_paths(args: argparse.Namespace) -> tuple[str, Path, Path]:
    current = read_current_session(TRACKER_ROOT)
    if args.session_data_dir:
        data_dir = Path(args.session_data_dir)
        session_id = args.session_id or data_dir.name
    else:
        session_id = args.session_id or (current or {}).get("session_id")
        if session_id:
            data_dir = TRACKER_ROOT / "trellis_point_prior_mv" / "ar_tracker" / "data" / session_id
            if not data_dir.exists():
                data_dir = TRACKER_ROOT / "ReconViaGen" / "ar_tracker" / "data" / session_id
        else:
            raise ValueError("set --session_id or --session_data_dir; no current session flag was found")

    if not session_id:
        raise ValueError("set --session_id or --session_data_dir; no current session flag was found")

    if args.session_mask_dir:
        mask_dir = Path(args.session_mask_dir)
    elif current and current.get("session_id") == session_id and current.get("mask_dir"):
        mask_dir = Path(current["mask_dir"])
    elif data_dir.parent.name == "data":
        mask_dir = data_dir.parent.parent / "masks" / str(session_id)
    else:
        mask_dir = TRACKER_ROOT / "ReconViaGen" / "ar_tracker" / "masks" / str(session_id)

    return safe_name(session_id or data_dir.name), data_dir, mask_dir


def read_phone_poses(pose_path: Path) -> dict[str, dict]:
    poses: dict[str, dict] = {}
    if not pose_path.exists():
        raise FileNotFoundError(f"missing phone pose file: {pose_path}")
    for line_no, raw_line in enumerate(pose_path.read_text(encoding="utf-8", errors="ignore").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split(",")]
        if parts[0] == "frame_name":
            continue
        if len(parts) < 7:
            print(f"[ar_session_smoke][WARN] skip short pose line {line_no}: {line}", flush=True)
            continue
        frame_name = parts[0]
        pos = [parse_float(v) for v in parts[1:4]]
        rot = [parse_float(v) for v in parts[4:7]]
        if any(v is None for v in pos):
            print(f"[ar_session_smoke][WARN] skip pose without position: {frame_name}", flush=True)
            continue
        pose = {
            "frame_name": frame_name,
            "pos": np.asarray(pos, dtype=np.float64),
            "rot_deg": None if any(v is None for v in rot) else np.asarray(rot, dtype=np.float64),
            "quat": None,
            "intrinsics": None,
            "image_transform": "unknown",
            "cpu_image_timestamp_s": None,
            "camera_frame_timestamp_ns": None,
            "pose_sample_realtime_s": None,
            "camera_frame_timestamp_delta_s": None,
            "pose_binding": "legacy_unversioned",
            "screen_orientation": "unknown",
            "tracking_state": "unknown",
            "display_matrix": None,
            "projection_matrix": None,
            "strictly_synchronized": False,
        }
        if len(parts) >= 11:
            quat = [parse_float(v) for v in parts[7:11]]
            if not any(v is None for v in quat):
                pose["quat"] = np.asarray(quat, dtype=np.float64)
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
        if len(parts) > 22:
            pose["cpu_image_timestamp_s"] = parse_float(parts[22])
        if len(parts) > 23:
            pose["camera_frame_timestamp_ns"] = parse_int(parts[23])
        if len(parts) > 24:
            pose["pose_sample_realtime_s"] = parse_float(parts[24])
        if len(parts) > 25:
            pose["camera_frame_timestamp_delta_s"] = parse_float(parts[25])
        if len(parts) > 26 and parts[26]:
            pose["pose_binding"] = parts[26]
        if len(parts) > 27 and parts[27]:
            pose["screen_orientation"] = parts[27]
        if len(parts) > 28 and parts[28]:
            pose["tracking_state"] = parts[28]
        if len(parts) > 29 and parts[29]:
            pose["display_matrix"] = parts[29]
        if len(parts) > 30 and parts[30]:
            pose["projection_matrix"] = parts[30]
        delta = pose["camera_frame_timestamp_delta_s"]
        pose["strictly_synchronized"] = bool(
            pose["pose_binding"] == "camera_frame_received"
            and delta is not None
            and 0.0 <= delta <= 0.05
        )
        poses[frame_name] = pose
    if not poses:
        raise ValueError(f"no valid phone poses parsed from {pose_path}")
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
        u[:, -1] *= -1
        fixed = u @ vt
    return fixed


def unity_world_point_to_colmap(point: np.ndarray) -> np.ndarray:
    return np.asarray([point[0], point[1], -point[2]], dtype=np.float64)


def image_camera_rotation_matrix(degrees: float) -> np.ndarray:
    angle = math.radians(float(degrees))
    cosine, sine = math.cos(angle), math.sin(angle)
    return np.asarray(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def unity_pose_to_colmap_w2c(
    pose: dict, *, image_camera_rotation_degrees: float = 90.0
) -> tuple[np.ndarray, np.ndarray]:
    if pose.get("quat") is not None:
        r_unity_c2w = unity_quat_to_rotmat(pose["quat"])
    elif pose.get("rot_deg") is not None:
        r_unity_c2w = unity_euler_to_rotmat(pose["rot_deg"])
    else:
        raise ValueError(f"no rotation found for {pose.get('frame_name', '<unknown>')}")

    unity_to_cv_world = np.diag([1.0, 1.0, -1.0])
    unity_cam_to_cv_cam = np.diag([1.0, -1.0, 1.0])
    cam_center_w = unity_to_cv_world @ pose["pos"]
    r_cv_c2w = unity_to_cv_world @ r_unity_c2w @ unity_cam_to_cv_cam
    r_cv_c2w = nearest_rotation_matrix(r_cv_c2w)
    r_w2c = r_cv_c2w.T
    t_w2c = -r_w2c @ cam_center_w

    image_cam_from_pose_cam = image_camera_rotation_matrix(
        image_camera_rotation_degrees
    )
    return image_cam_from_pose_cam @ r_w2c, image_cam_from_pose_cam @ t_w2c


def rotmat_to_qvec(rot: np.ndarray) -> np.ndarray:
    r = np.asarray(rot, dtype=np.float64)
    trace = np.trace(r)
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (r[2, 1] - r[1, 2]) / s
        qy = (r[0, 2] - r[2, 0]) / s
        qz = (r[1, 0] - r[0, 1]) / s
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2.0
        qw = (r[2, 1] - r[1, 2]) / s
        qx = 0.25 * s
        qy = (r[0, 1] + r[1, 0]) / s
        qz = (r[0, 2] + r[2, 0]) / s
    elif r[1, 1] > r[2, 2]:
        s = math.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2.0
        qw = (r[0, 2] - r[2, 0]) / s
        qx = (r[0, 1] + r[1, 0]) / s
        qy = 0.25 * s
        qz = (r[1, 2] + r[2, 1]) / s
    else:
        s = math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2.0
        qw = (r[1, 0] - r[0, 1]) / s
        qx = (r[0, 2] + r[2, 0]) / s
        qy = (r[1, 2] + r[2, 1]) / s
        qz = 0.25 * s
    qvec = np.asarray([qw, qx, qy, qz], dtype=np.float64)
    qvec /= np.linalg.norm(qvec) + 1e-12
    if qvec[0] < 0:
        qvec *= -1.0
    return qvec


def image_size(image_path: Path) -> tuple[int, int]:
    with Image.open(image_path) as image:
        return image.size


def intrinsics_for_pose(pose: dict, image_width: int, image_height: int) -> tuple[float, float, float, float, str]:
    intr = pose.get("intrinsics")
    source = "fallback_image_size"
    if intr is not None:
        fx, fy, cx, cy = intr["fx"], intr["fy"], intr["cx"], intr["cy"]
        src_w = intr.get("cpu_image_width") or intr.get("width") or intr.get("image_width") or image_width
        src_h = intr.get("cpu_image_height") or intr.get("height") or intr.get("image_height") or image_height
        if src_w and src_h and src_w > 0 and src_h > 0:
            fx *= image_width / float(src_w)
            fy *= image_height / float(src_h)
            cx *= image_width / float(src_w)
            cy *= image_height / float(src_h)
        image_transform = str(pose.get("image_transform") or "None").lower()
        if "mirrorx" in image_transform:
            cx = (image_width - 1) - cx
        if "mirrory" in image_transform:
            cy = (image_height - 1) - cy
        if fx > 0 and fy > 0 and 0 <= cx <= image_width and 0 <= cy <= image_height:
            return fx, fy, cx, cy, f"ar_foundation_{pose.get('image_transform') or 'None'}"
    focal = float(max(image_width, image_height))
    return focal, focal, (image_width - 1) * 0.5, (image_height - 1) * 0.5, source


def list_session_images(data_dir: Path) -> list[Path]:
    return sorted(p for p in data_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"} and p.name.startswith("frame_"))


def parse_selected_indices(text: str | None) -> set[int] | None:
    if text is None:
        return None
    text = str(text).strip()
    if not text or text.lower() == "all":
        return None
    out: set[int] = set()
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            out.update(range(int(start), int(end) + 1))
        else:
            out.add(int(part))
    return out


def copy_session_files(
    data_dir: Path,
    mask_dir: Path,
    dataset_dir: Path,
    poses: dict,
    *,
    require_masks: bool,
    selected_indices: set[int] | None,
) -> list[str]:
    images_dir = dataset_dir / "images"
    rgb_dir = dataset_dir / "rgb"
    masks_out = dataset_dir / "masks"
    for path in (images_dir, rgb_dir, masks_out):
        path.mkdir(parents=True, exist_ok=True)
    kept: list[str] = []
    missing_masks: list[str] = []
    for frame_index, image_path in enumerate(list_session_images(data_dir)):
        if selected_indices is not None and frame_index not in selected_indices:
            continue
        if image_path.name not in poses:
            continue
        mask_path = mask_dir / f"{image_path.stem}.png"
        if not mask_path.exists():
            missing_masks.append(image_path.name)
            if require_masks:
                continue
        shutil.copy2(image_path, images_dir / image_path.name)
        shutil.copy2(image_path, rgb_dir / image_path.name)
        if mask_path.exists():
            shutil.copy2(mask_path, masks_out / f"{image_path.stem}.png")
        kept.append(image_path.name)
    if require_masks and missing_masks:
        preview = ", ".join(missing_masks[:8])
        raise FileNotFoundError(f"missing final masks for session frames: {preview}")
    if not kept:
        raise ValueError(f"no frames with image/pose/mask were copied from {data_dir}")
    shutil.copy2(data_dir / "poses.txt", dataset_dir / "poses.txt")
    frame_metadata = data_dir / "frame_metadata.jsonl"
    if frame_metadata.is_file():
        shutil.copy2(frame_metadata, dataset_dir / "frame_metadata.jsonl")
    return kept


def write_phone_sparse(dataset_dir: Path, frame_names: list[str], poses: dict, sparse_subdir: str) -> dict[str, Any]:
    sparse_dir = dataset_dir / sparse_subdir
    sparse_dir.mkdir(parents=True, exist_ok=True)
    image_dir = dataset_dir / "images"
    camera_lines: list[str] = []
    image_lines: list[str] = []
    intrinsics_sources: dict[str, int] = {}
    for image_id, frame_name in enumerate(frame_names, start=1):
        image_path = image_dir / frame_name
        width, height = image_size(image_path)
        pose = poses[frame_name]
        r_w2c, t_w2c = unity_pose_to_colmap_w2c(pose)
        qvec = rotmat_to_qvec(r_w2c)
        fx, fy, cx, cy, source = intrinsics_for_pose(pose, width, height)
        intrinsics_sources[source] = intrinsics_sources.get(source, 0) + 1
        camera_lines.append(f"{image_id} PINHOLE {width} {height} {fx:.10f} {fy:.10f} {cx:.10f} {cy:.10f}\n")
        image_lines.append(
            f"{image_id} {qvec[0]:.12f} {qvec[1]:.12f} {qvec[2]:.12f} {qvec[3]:.12f} "
            f"{t_w2c[0]:.12f} {t_w2c[1]:.12f} {t_w2c[2]:.12f} {image_id} {frame_name}\n\n"
        )
    (sparse_dir / "cameras.txt").write_text(
        "# Camera list with one line of data per camera:\n"
        "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n"
        f"# Number of cameras: {len(camera_lines)}\n"
        + "".join(camera_lines),
        encoding="utf-8",
    )
    (sparse_dir / "images.txt").write_text(
        "# Image list with two lines of data per image:\n"
        "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n"
        "#   POINTS2D[] as (X, Y, POINT3D_ID)\n"
        f"# Number of images: {len(image_lines)}\n"
        + "".join(image_lines),
        encoding="utf-8",
    )
    (sparse_dir / "points3D.txt").write_text(
        "# 3D point list with one line of data per point:\n"
        "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n"
        "# Number of points: 0, mean track length: 0\n",
        encoding="utf-8",
    )
    meta = {
        "pose_source": "unity_ar_pose",
        "num_images": len(frame_names),
        "intrinsics_sources": intrinsics_sources,
        "coordinate_conversion": {
            "world": "diag(1, 1, -1) * unity_world",
            "camera": "Unity camera converted to COLMAP x-right y-down z-forward",
            "cpu_image_camera_from_pose_camera": "Rz(+90deg): [[0,-1,0],[1,0,0],[0,0,1]]",
        },
    }
    write_json(sparse_dir / "phone_pose_meta.json", meta)
    return meta


def iter_slam_point_rows(path: Path):
    if not path.exists():
        return
    for line_no, raw_line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            print(f"[ar_session_smoke][WARN] skip invalid JSONL line {line_no}: {path}", flush=True)
            continue
        points = row.get("points", []) if isinstance(row, dict) else []
        if not isinstance(points, list):
            continue
        for point in points:
            if isinstance(point, dict):
                values = [point.get("x"), point.get("y"), point.get("z")]
                conf = point.get("confidence", point.get("conf", 1.0))
            elif isinstance(point, (list, tuple)) and len(point) >= 3:
                values = list(point[:3])
                conf = point[3] if len(point) > 3 else 1.0
            else:
                continue
            xyz = [parse_float(v) for v in values]
            confidence = parse_float(conf)
            if any(v is None for v in xyz):
                continue
            yield np.asarray(xyz, dtype=np.float64), float(confidence if confidence is not None else 1.0)


def load_slam_points(path: Path, coordinate_frame: str, max_points: int, voxel_size: float) -> tuple[np.ndarray, np.ndarray, dict]:
    raw_points = []
    raw_conf = []
    for point, confidence in iter_slam_point_rows(path) or []:
        if coordinate_frame == "unity_world":
            point = unity_world_point_to_colmap(point)
        elif coordinate_frame == "colmap_world":
            point = np.asarray(point, dtype=np.float64)
        else:
            raise ValueError(f"unsupported slam point coordinate frame: {coordinate_frame}")
        if np.isfinite(point).all() and math.isfinite(confidence):
            raw_points.append(point.astype(np.float32))
            raw_conf.append(float(confidence))
    if not raw_points:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.float32), {
            "raw_point_count": 0,
            "merged_point_count": 0,
        }
    points = np.stack(raw_points, axis=0).astype(np.float32)
    conf = np.asarray(raw_conf, dtype=np.float32)
    if voxel_size > 0:
        keys = np.floor(points / float(voxel_size)).astype(np.int64)
        _, inverse = np.unique(keys, axis=0, return_inverse=True)
        merged = np.zeros((int(inverse.max()) + 1, 3), dtype=np.float64)
        conf_sum = np.zeros((merged.shape[0],), dtype=np.float64)
        count = np.zeros((merged.shape[0],), dtype=np.float64)
        np.add.at(merged, inverse, points.astype(np.float64))
        np.add.at(conf_sum, inverse, conf.astype(np.float64))
        np.add.at(count, inverse, 1.0)
        points = (merged / np.maximum(count[:, None], 1.0)).astype(np.float32)
        conf = (conf_sum / np.maximum(count, 1.0)).astype(np.float32)
    if max_points > 0 and points.shape[0] > max_points:
        order = np.argsort(-conf)[: int(max_points)]
        order = np.sort(order)
        points = points[order]
        conf = conf[order]
    return points, conf, {
        "raw_point_count": int(len(raw_points)),
        "merged_point_count": int(points.shape[0]),
        "coordinate_frame": coordinate_frame,
        "merge_voxel_size": float(voxel_size),
    }


def copy_pose_sparse(input_sparse: Path, output_sparse: Path) -> None:
    output_sparse.mkdir(parents=True, exist_ok=True)
    for name in ("cameras.txt", "images.txt", "phone_pose_meta.json"):
        src = input_sparse / name
        if src.exists():
            shutil.copy2(src, output_sparse / name)


def write_points3d(path: Path, points: np.ndarray, confidence: np.ndarray) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        f.write(f"# Number of points: {points.shape[0]}, mean track length: 0\n")
        for idx, point in enumerate(points, start=1):
            conf = float(confidence[idx - 1]) if idx - 1 < confidence.shape[0] else 1.0
            err = max(0.0, 1.0 - conf)
            f.write(f"{idx} {point[0]:.9f} {point[1]:.9f} {point[2]:.9f} 128 128 128 {err:.6f}\n")


def write_direct_ar_sparse(dataset_dir: Path, points_path: Path, args: argparse.Namespace) -> dict[str, Any]:
    points, conf, stats = load_slam_points(
        points_path,
        coordinate_frame=args.slam_points_coordinate_frame,
        max_points=int(args.max_ar_points),
        voxel_size=float(args.ar_point_merge_voxel_size),
    )
    out_sparse = dataset_dir / args.direct_sparse_subdir
    copy_pose_sparse(dataset_dir / args.pose_sparse_subdir, out_sparse)
    write_points3d(out_sparse / "points3D.txt", points, conf)
    meta = {
        "source": "ar_foundation_point_cloud_upload",
        "input_jsonl": str(points_path),
        "output_sparse": str(out_sparse),
        "point_count": int(points.shape[0]),
        **stats,
    }
    write_json(out_sparse / "ar_direct_points_meta.json", meta)
    return meta


def build(args: argparse.Namespace) -> None:
    session_id, data_dir, mask_dir = resolve_session_paths(args)
    output_dir = Path(args.output_dir)
    dataset_name = safe_name(args.dataset_name or f"ar_session_{session_id}")
    dataset_dir = output_dir / "dataset" / dataset_name
    if dataset_dir.exists() and args.overwrite:
        shutil.rmtree(dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    poses = read_phone_poses(data_dir / "poses.txt")
    selected_indices = parse_selected_indices(args.selected_indices)
    frame_names = copy_session_files(
        data_dir,
        mask_dir,
        dataset_dir,
        poses,
        require_masks=not args.allow_missing_masks,
        selected_indices=selected_indices,
    )
    pose_meta = write_phone_sparse(dataset_dir, frame_names, poses, args.pose_sparse_subdir)

    slam_points_path = Path(args.slam_points_jsonl) if args.slam_points_jsonl else data_dir / "slam_points.jsonl"
    direct_meta = None
    if slam_points_path.exists():
        direct_meta = write_direct_ar_sparse(dataset_dir, slam_points_path, args)
    elif args.require_slam_points:
        raise FileNotFoundError(f"missing AR slam points jsonl: {slam_points_path}")

    report = {
        "schema": "ar_session_smoke_dataset_v1",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "session_id": session_id,
        "session_data_dir": str(data_dir),
        "session_mask_dir": str(mask_dir),
        "dataset_name": dataset_name,
        "dataset_dir": str(dataset_dir),
        "frame_count": int(len(frame_names)),
        "selected_indices": sorted(selected_indices) if selected_indices is not None else None,
        "frames": frame_names,
        "pose_sparse_subdir": args.pose_sparse_subdir,
        "direct_sparse_subdir": args.direct_sparse_subdir if direct_meta is not None else "",
        "pose_meta": pose_meta,
        "direct_ar_points_meta": direct_meta,
        "slam_points_jsonl": str(slam_points_path) if slam_points_path.exists() else "",
        "args": vars(args),
    }
    write_json(output_dir / "prepare_report.json", report)
    write_json(dataset_dir / "ar_session_smoke_meta.json", report)
    write_csv(output_dir / "prepare_frames.csv", [{"frame_name": name} for name in frame_names])
    print(
        f"[ar_session_smoke] dataset={dataset_dir} frames={len(frame_names)} "
        f"ar_points={0 if direct_meta is None else direct_meta.get('point_count', 0)}",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a CoarseModel phone AR session for TRELLIS point-prior smoke tests.")
    parser.add_argument("--session_id", default=None)
    parser.add_argument("--session_data_dir", default=None)
    parser.add_argument("--session_mask_dir", default=None)
    parser.add_argument("--slam_points_jsonl", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--dataset_name", default=None)
    parser.add_argument("--selected_indices", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow_missing_masks", action="store_true")
    parser.add_argument("--require_slam_points", action="store_true")
    parser.add_argument("--pose_sparse_subdir", default="sparse/0")
    parser.add_argument("--direct_sparse_subdir", default="sparse_ar_direct/0")
    parser.add_argument("--slam_points_coordinate_frame", choices=["unity_world", "colmap_world"], default="unity_world")
    parser.add_argument("--max_ar_points", type=int, default=50000)
    parser.add_argument("--ar_point_merge_voxel_size", type=float, default=0.002)
    return parser.parse_args()


def main() -> None:
    try:
        build(parse_args())
    except (ValueError, FileNotFoundError) as exc:
        raise SystemExit(f"[ar_session_smoke][ERROR] {exc}") from None


if __name__ == "__main__":
    main()
