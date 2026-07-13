import os
import shutil
import time
import json
import subprocess
import sys
import re
import logging
import math
from flask import Flask, request, jsonify, send_from_directory
import cv2
import numpy as np
from PIL import Image, ImageDraw

try:
    from sam2_mask import write_prompted_image_mask, write_prompted_video_masks
except ImportError:
    from .sam2_mask import write_prompted_image_mask, write_prompted_video_masks

app = Flask(__name__)
logging.getLogger("werkzeug").setLevel(logging.WARNING)

# 统一目录与信号文件配置
TRACKER_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RECONVIAGEN_DIR = os.path.join(TRACKER_ROOT, "ReconViaGen")
COARSEMODEL_DIR = os.path.join(TRACKER_ROOT, "CoarseModel")
COARSE_CORE_DIR = os.path.join(COARSEMODEL_DIR, "core")

BASE_DIR = os.path.join(RECONVIAGEN_DIR, "ar_tracker")
DATA_ROOT = os.path.join(BASE_DIR, "data")
PREVIEW_ROOT = os.path.join(BASE_DIR, "previews")
MASK_ROOT = os.path.join(BASE_DIR, "masks")
REVIEW_ROOT = os.path.join(BASE_DIR, "review_previews")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
FLAG_DIR = os.path.join(BASE_DIR, "flags")
DATA_DIR = DATA_ROOT
PREVIEW_DIR = PREVIEW_ROOT

FLAG_START_PREPROCESS = os.path.join(FLAG_DIR, "start_preprocess.flag")
FLAG_PREPROCESS_DONE = os.path.join(FLAG_DIR, "preprocess_done.json")
FLAG_START_GENERATE = os.path.join(FLAG_DIR, "start_generate.json")
FLAG_GENERATE_DONE = os.path.join(FLAG_DIR, "generate_done.flag")
FLAG_POSTPROCESS_DONE = os.path.join(FLAG_DIR, "postprocess_done.flag")
FLAG_CURRENT_SESSION = os.path.join(FLAG_DIR, "current_session.json")

frame_counter = 0
current_session_id = None
current_data_dir = DATA_DIR
current_preview_dir = PREVIEW_DIR
current_mask_dir = MASK_ROOT
current_review_dir = REVIEW_ROOT
current_seg_points = []
current_seed_frames = set()

for path in [COARSEMODEL_DIR, COARSE_CORE_DIR]:
    if path not in sys.path:
        sys.path.insert(0, path)
if RECONVIAGEN_DIR not in sys.path:
    sys.path.insert(0, RECONVIAGEN_DIR)

import ar_pose_quality as arq

def clean_environment():
    # data / previews / output 都按 session 保留；只清理运行态 flag。
    shutil.rmtree(FLAG_DIR, ignore_errors=True)
    for d in [DATA_ROOT, PREVIEW_ROOT, MASK_ROOT, REVIEW_ROOT, OUTPUT_DIR, FLAG_DIR]:
        os.makedirs(d, exist_ok=True)


def _make_session_id():
    return f"{time.strftime('%Y%m%d_%H%M%S')}_{int((time.time() % 1) * 1000):03d}"


def _set_current_session(session_id, reset_points=False):
    global current_session_id, current_data_dir, current_preview_dir, current_mask_dir, current_review_dir
    global current_seg_points, current_seed_frames, frame_counter
    current_session_id = _safe_dataset_name(session_id)
    current_data_dir = os.path.join(DATA_ROOT, current_session_id)
    current_preview_dir = os.path.join(PREVIEW_ROOT, current_session_id)
    current_mask_dir = os.path.join(MASK_ROOT, current_session_id)
    current_review_dir = os.path.join(REVIEW_ROOT, current_session_id)
    for d in [current_data_dir, current_preview_dir, current_mask_dir, current_review_dir, OUTPUT_DIR, FLAG_DIR]:
        os.makedirs(d, exist_ok=True)
    if reset_points:
        current_seg_points = []
        current_seed_frames = set()
    else:
        _load_seg_points()
        _load_seed_frames()

    with open(FLAG_CURRENT_SESSION, "w") as f:
        json.dump(
            {
                "session_id": current_session_id,
                "data_dir": current_data_dir,
                "preview_dir": current_preview_dir,
                "mask_dir": current_mask_dir,
                "review_dir": current_review_dir,
            },
            f,
            indent=4,
        )
    existing_ids = []
    for name in os.listdir(current_data_dir):
        match = re.match(r"frame_(\d+)\.(jpg|jpeg|png)$", name, re.IGNORECASE)
        if match:
            existing_ids.append(int(match.group(1)))
    frame_counter = max(existing_ids) + 1 if existing_ids else 0
    return current_session_id


def _load_current_session():
    global current_mask_dir, current_review_dir
    if os.path.exists(FLAG_CURRENT_SESSION):
        with open(FLAG_CURRENT_SESSION, "r") as f:
            data = json.load(f)
        _set_current_session(data["session_id"])
        current_mask_dir = data.get("mask_dir") or os.path.join(MASK_ROOT, data["session_id"])
        current_review_dir = data.get("review_dir") or os.path.join(REVIEW_ROOT, data["session_id"])
        os.makedirs(current_mask_dir, exist_ok=True)
        os.makedirs(current_review_dir, exist_ok=True)
        _load_seg_points()
        _load_seed_frames()
    elif current_session_id is None:
        _set_current_session(_make_session_id(), reset_points=True)


def _list_session_images():
    _load_current_session()
    return sorted(
        f for f in os.listdir(current_data_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )


def _safe_dataset_name(name):
    name = re.sub(r"[^0-9a-zA-Z_]+", "_", name)
    return name.strip("_") or time.strftime("%Y%m%d_%H%M%S")


def _latest_recon_output_dir():
    if not os.path.exists(OUTPUT_DIR):
        raise FileNotFoundError(f"ReconViaGen output dir not found: {OUTPUT_DIR}")
    candidates = [
        os.path.join(OUTPUT_DIR, d)
        for d in os.listdir(OUTPUT_DIR)
        if os.path.isdir(os.path.join(OUTPUT_DIR, d))
    ]
    candidates = [
        d for d in candidates
        if any(os.path.exists(os.path.join(d, f"reconstructed_object{ext}")) for ext in [".obj", ".glb", ".ply"])
    ]
    if not candidates:
        raise FileNotFoundError(f"No reconstructed_object mesh found under {OUTPUT_DIR}")
    return max(candidates, key=os.path.getmtime)


def _find_recon_mesh(output_dir):
    for ext in [".obj", ".glb", ".ply"]:
        mesh_path = os.path.join(output_dir, f"reconstructed_object{ext}")
        if os.path.exists(mesh_path):
            return mesh_path
    raise FileNotFoundError(f"No reconstructed_object mesh found in {output_dir}")


def _export_mesh_as_obj(mesh_path, obj_path):
    import trimesh

    loaded = trimesh.load(mesh_path, force="scene")
    if isinstance(loaded, trimesh.Scene):
        geometries = [g for g in loaded.geometry.values() if len(g.vertices) > 0]
        if not geometries:
            raise ValueError(f"No geometry in mesh: {mesh_path}")
        mesh = trimesh.util.concatenate(geometries)
    else:
        mesh = loaded

    if mesh.vertices is None or len(mesh.vertices) == 0:
        raise ValueError(f"Mesh has no vertices: {mesh_path}")

    os.makedirs(os.path.dirname(obj_path), exist_ok=True)
    mesh.export(obj_path)


def _write_mask_from_preview(preview_path, mask_path):
    preview = cv2.imread(preview_path, cv2.IMREAD_UNCHANGED)
    if preview is None:
        raise FileNotFoundError(f"Preview image not found: {preview_path}")
    if preview.ndim == 3 and preview.shape[2] == 4:
        mask = preview[:, :, 3]
    elif preview.ndim == 3:
        gray = cv2.cvtColor(preview, cv2.COLOR_BGR2GRAY)
        mask = np.where(gray > 0, 255, 0).astype(np.uint8)
    else:
        mask = np.where(preview > 0, 255, 0).astype(np.uint8)
    cv2.imwrite(mask_path, mask)


def _apply_mask_and_crop(image_path, mask_path, preview_path, resolution=518):
    image = Image.open(image_path).convert("RGBA")
    mask = Image.open(mask_path).convert("L")
    if mask.size != image.size:
        mask = mask.resize(image.size, Image.Resampling.NEAREST)

    rgba = np.array(image)
    alpha = np.array(mask)
    rgba[:, :, 3] = np.where(alpha > 127, 255, 0).astype(np.uint8)
    output = Image.fromarray(rgba, mode="RGBA")

    ys, xs = np.nonzero(rgba[:, :, 3] > 204)
    if len(xs) == 0:
        output.save(preview_path, "PNG")
        return

    left, right = xs.min(), xs.max()
    top, bottom = ys.min(), ys.max()
    center_x = 0.5 * (left + right)
    center_y = 0.5 * (top + bottom)
    size = int(max(right - left, bottom - top) * 1.1)
    size = max(size, 1)

    crop_box = (
        max(0, int(center_x - size // 2)),
        max(0, int(center_y - size // 2)),
        min(output.width, int(center_x + size // 2)),
        min(output.height, int(center_y + size // 2)),
    )
    output = output.crop(crop_box)
    width, height = output.size
    side = max(width, height)
    padded = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    padded.paste(output, ((side - width) // 2, (side - height) // 2))
    padded = padded.resize((resolution, resolution), Image.Resampling.BILINEAR)

    arr = np.array(padded)
    fg = arr[:, :, 3] > 204
    arr[:, :, :3] = arr[:, :, :3] * fg[:, :, None]
    Image.fromarray(arr, mode="RGBA").save(preview_path, "PNG")


def _point_to_pixel(point, image_size):
    width, height = image_size
    x = float(point["x"])
    y = float(point["y"])
    if point.get("normalized", True):
        x *= width
        y *= height
    x = int(np.clip(round(x), 0, width - 1))
    y = int(np.clip(round(y), 0, height - 1))
    return x, y


def _write_fullsize_mask_preview(image_path, mask_path, preview_path, points=None):
    image = Image.open(image_path).convert("RGB")
    last_exc = None
    mask = None
    for _ in range(5):
        try:
            mask = Image.open(mask_path).convert("L")
            break
        except Exception as exc:
            last_exc = exc
            time.sleep(0.08)
    if mask is None:
        raise RuntimeError(f"Mask file is not readable after write: {mask_path}; {last_exc}")
    if mask.size != image.size:
        mask = mask.resize(image.size, Image.Resampling.NEAREST)

    image_arr = np.array(image).astype(np.float32)
    mask_arr = np.array(mask) > 127
    dimmed = image_arr * 0.35
    tinted = image_arr * 0.70 + np.array([0.0, 210.0, 80.0], dtype=np.float32) * 0.30
    output = np.where(mask_arr[:, :, None], tinted, dimmed)
    output = np.clip(output, 0, 255).astype(np.uint8)

    os.makedirs(os.path.dirname(preview_path), exist_ok=True)
    preview = Image.fromarray(output, mode="RGB")
    if points:
        draw = ImageDraw.Draw(preview)
        radius = max(5, int(min(preview.size) * 0.012))
        for point in points:
            x, y = _point_to_pixel(point, preview.size)
            color = (0, 255, 0) if int(point.get("label", 1)) == 1 else (255, 0, 0)
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color, outline=(255, 255, 255), width=2)
            draw.line((x - radius * 2, y, x + radius * 2, y), fill=(255, 255, 255), width=1)
            draw.line((x, y - radius * 2, x, y + radius * 2), fill=(255, 255, 255), width=1)
    preview.save(preview_path, "PNG")


def _seg_points_path():
    return os.path.join(current_mask_dir, "segmentation_points.json")


def _seed_frames_path():
    return os.path.join(current_mask_dir, "seed_frames.json")


def _load_seg_points():
    global current_seg_points
    path = _seg_points_path()
    if not os.path.exists(path):
        current_seg_points = []
        return
    try:
        with open(path, "r") as f:
            data = json.load(f)
        current_seg_points = data.get("points", [])
    except Exception:
        current_seg_points = []


def _load_seed_frames():
    global current_seed_frames
    path = _seed_frames_path()
    if not os.path.exists(path):
        current_seed_frames = set()
        return
    try:
        with open(path, "r") as f:
            data = json.load(f)
        current_seed_frames = set(int(i) for i in data.get("seed_frames", []))
    except Exception:
        current_seed_frames = set()


def _save_seg_points():
    os.makedirs(current_mask_dir, exist_ok=True)
    with open(_seg_points_path(), "w") as f:
        json.dump({"points": current_seg_points}, f, indent=4)


def _save_seed_frames():
    os.makedirs(current_mask_dir, exist_ok=True)
    with open(_seed_frames_path(), "w") as f:
        json.dump({"seed_frames": sorted(current_seed_frames)}, f, indent=4)


def _run_interactive_segmentation():
    _load_current_session()
    image_files = _list_session_images()
    if not image_files:
        raise RuntimeError("No captured frames to segment.")
    if not current_seg_points:
        raise RuntimeError("No user foreground/background points. Tap the object in at least one original frame first.")
    if not current_seed_frames:
        raise RuntimeError("No approved seed frames. Confirm at least one good frame before running video segmentation.")

    image_paths = [os.path.join(current_data_dir, name) for name in image_files]
    mask_paths = [
        os.path.join(current_mask_dir, f"{os.path.splitext(name)[0]}.png")
        for name in image_files
    ]
    write_prompted_video_masks(
        image_paths,
        mask_paths,
        current_seg_points,
        seed_frame_indices=sorted(current_seed_frames),
    )

    for idx, (image_path, mask_path) in enumerate(zip(image_paths, mask_paths)):
        _apply_mask_and_crop(image_path, mask_path, os.path.join(current_preview_dir, f"{idx}.png"))
        frame_points = [p for p in current_seg_points if int(p.get("frame_index", -1)) == idx]
        _write_fullsize_mask_preview(
            image_path,
            mask_path,
            os.path.join(current_review_dir, f"{idx}.png"),
            points=frame_points,
        )

    _save_seg_points()
    return len(image_files)


def _run_single_frame_segmentation_preview(frame_index):
    _load_current_session()
    image_files = _list_session_images()
    if frame_index < 0 or frame_index >= len(image_files):
        raise RuntimeError("frame index out of range")

    frame_points = [p for p in current_seg_points if int(p.get("frame_index", -1)) == int(frame_index)]
    if not frame_points:
        raise RuntimeError("No prompt point on current frame.")

    frame_name = image_files[frame_index]
    stem = os.path.splitext(frame_name)[0]
    image_path = os.path.join(current_data_dir, frame_name)
    mask_path = os.path.join(current_mask_dir, f"{stem}_prompt.png")
    review_path = os.path.join(current_review_dir, f"{frame_index}_prompt.png")
    write_prompted_image_mask(image_path, mask_path, current_seg_points, frame_index=frame_index)
    _write_fullsize_mask_preview(image_path, mask_path, review_path, points=frame_points)
    return review_path


def _prepare_coarsemodel_dataset(selected_indices, recon_output_dir=None):
    _load_current_session()
    if recon_output_dir:
        output_root = os.path.realpath(OUTPUT_DIR)
        recon_output_dir = os.path.realpath(recon_output_dir)
        if not (recon_output_dir == output_root or recon_output_dir.startswith(output_root + os.sep)):
            raise ValueError(f"ReconViaGen output dir is outside output root: {recon_output_dir}")
        if not os.path.isdir(recon_output_dir):
            raise FileNotFoundError(f"ReconViaGen output dir not found: {recon_output_dir}")
    else:
        recon_output_dir = _latest_recon_output_dir()
    recon_mesh_path = _find_recon_mesh(recon_output_dir)
    dataset_name = _safe_dataset_name(f"reconviagen_{os.path.basename(recon_output_dir)}")

    dataset_dir = os.path.join(COARSEMODEL_DIR, "datasets", dataset_name)
    model_dir = os.path.join(dataset_dir, "models")
    rgb_dir = os.path.join(dataset_dir, "rgb")
    images_dir = os.path.join(dataset_dir, "images")
    mask_dir = os.path.join(dataset_dir, "masks")
    source_dir = os.path.join(dataset_dir, "reconviagen_output")
    for d in [model_dir, rgb_dir, images_dir, mask_dir, source_dir]:
        os.makedirs(d, exist_ok=True)

    obj_path = os.path.join(model_dir, f"{dataset_name}.obj")
    _export_mesh_as_obj(recon_mesh_path, obj_path)

    for filename in os.listdir(recon_output_dir):
        src = os.path.join(recon_output_dir, filename)
        dst = os.path.join(source_dir, filename)
        if os.path.isfile(src):
            shutil.copy2(src, dst)
        elif os.path.isdir(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)

    image_files = _list_session_images()
    kept = []
    for idx in selected_indices:
        if idx < 0 or idx >= len(image_files):
            continue
        frame_name = image_files[idx]
        stem = os.path.splitext(frame_name)[0]
        src_image = os.path.join(current_data_dir, frame_name)
        preview_path = os.path.join(current_preview_dir, f"{idx}.png")
        mask_path = os.path.join(mask_dir, f"{stem}.png")
        session_mask_path = os.path.join(current_mask_dir, f"{stem}.png")
        shutil.copy2(src_image, os.path.join(rgb_dir, frame_name))
        shutil.copy2(src_image, os.path.join(images_dir, frame_name))
        if not os.path.exists(session_mask_path):
            raise FileNotFoundError(
                f"Interactive mask missing for {frame_name}: {session_mask_path}. Run segmentation before generate."
            )
        shutil.copy2(session_mask_path, mask_path)
        kept.append(frame_name)

    pose_path = os.path.join(current_data_dir, "poses.txt")
    if os.path.exists(pose_path):
        shutil.copy2(pose_path, os.path.join(dataset_dir, "poses.txt"))

    meta = {
        "dataset_name": dataset_name,
        "session_id": current_session_id,
        "session_data_dir": current_data_dir,
        "session_preview_dir": current_preview_dir,
        "source_output_dir": recon_output_dir,
        "source_mesh": recon_mesh_path,
        "coarse_obj": obj_path,
        "selected_indices": selected_indices,
        "selected_frames": kept,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(dataset_dir, "reconviagen_meta.json"), "w") as f:
        json.dump(meta, f, indent=4)

    return dataset_name, dataset_dir


def _generate_coarse_templates_and_repre(dataset_name):
    from gen_template_auto import generate_templates_for_dataset
    from gen_repre_auto import generate_repre_for_dataset

    print(f">>> [调度] [CoarseModel 1/2] 开始生成模板: {dataset_name}", flush=True)
    template_config = generate_templates_for_dataset(dataset_name)
    print(f">>> [调度] [CoarseModel 1/2] 模板生成完成: {template_config}", flush=True)
    print(f">>> [调度] [CoarseModel 2/2] 开始提取模板特征: {dataset_name}", flush=True)
    repre_config = generate_repre_for_dataset(dataset_name)
    print(f">>> [调度] [CoarseModel 2/2] 特征提取完成: {repre_config}", flush=True)
    return str(template_config), str(repre_config)


def _colmap_sparse_ready(dataset_dir):
    sparse_dir = os.path.join(dataset_dir, "sparse", "0")
    required = ["cameras.txt", "images.txt", "points3D.txt"]
    return all(os.path.exists(os.path.join(sparse_dir, name)) for name in required)


def _parse_float(value):
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    try:
        parsed = float(text)
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def _parse_int(value):
    parsed = _parse_float(value)
    if parsed is None:
        return None
    return int(round(parsed))


def _read_phone_poses(pose_path):
    poses = {}
    if not os.path.exists(pose_path):
        raise FileNotFoundError(f"Phone pose file not found: {pose_path}")

    with open(pose_path, "r") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(",")]
            if parts[0] == "frame_name":
                continue
            if len(parts) < 7:
                print(f">>> [调度] [PhonePose] poses.txt 第 {line_no} 行字段不足，跳过: {line}", flush=True)
                continue

            frame_name = parts[0]
            pos = [_parse_float(v) for v in parts[1:4]]
            rot_deg = [_parse_float(v) for v in parts[4:7]]
            if any(v is None for v in pos):
                print(f">>> [调度] [PhonePose] {frame_name} 缺少位置，跳过", flush=True)
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
                quat = [_parse_float(v) for v in parts[7:11]]
                if not any(v is None for v in quat):
                    pose["quat"] = np.array(quat, dtype=np.float64)  # Unity order: x, y, z, w

            if len(parts) >= 15:
                fx, fy, cx, cy = [_parse_float(v) for v in parts[11:15]]
                if None not in (fx, fy, cx, cy):
                    intrinsics = {
                        "fx": fx,
                        "fy": fy,
                        "cx": cx,
                        "cy": cy,
                        "width": _parse_int(parts[15]) if len(parts) > 15 else None,
                        "height": _parse_int(parts[16]) if len(parts) > 16 else None,
                        "image_width": _parse_int(parts[17]) if len(parts) > 17 else None,
                        "image_height": _parse_int(parts[18]) if len(parts) > 18 else None,
                        "cpu_image_width": _parse_int(parts[19]) if len(parts) > 19 else None,
                        "cpu_image_height": _parse_int(parts[20]) if len(parts) > 20 else None,
                    }
                    pose["intrinsics"] = intrinsics

            if len(parts) > 21:
                pose["image_transform"] = parts[21] or "None"
            elif len(parts) >= 19:
                # Compatibility for pose logs written by the previous phone client.
                pose["image_transform"] = "MirrorY"

            poses[frame_name] = pose

    if not poses:
        raise ValueError(f"No valid phone poses parsed from: {pose_path}")
    return poses


def _unity_quat_to_rotmat(quat_xyzw):
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


def _unity_euler_to_rotmat(rot_deg):
    # Fallback for old pose logs. New captures use quaternion to avoid Euler-order ambiguity.
    rx, ry, rz = np.deg2rad(rot_deg)
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    rx_mat = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float64)
    ry_mat = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)
    rz_mat = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float64)
    return ry_mat @ rx_mat @ rz_mat


def _nearest_rotation_matrix(rot):
    u, _s, vt = np.linalg.svd(rot)
    fixed = u @ vt
    if np.linalg.det(fixed) < 0:
        u[:, -1] *= -1
        fixed = u @ vt
    return fixed


def _unity_pose_to_colmap_w2c(pose):
    if pose.get("quat") is not None:
        r_unity_c2w = _unity_quat_to_rotmat(pose["quat"])
    elif pose.get("rot_deg") is not None:
        r_unity_c2w = _unity_euler_to_rotmat(pose["rot_deg"])
    else:
        raise ValueError(f"No rotation found for {pose.get('frame_name', '<unknown>')}")

    # Unity uses a left-handed world/camera convention. CoarseModel reads COLMAP-style
    # right-handed world coordinates with camera axes x-right, y-down, z-forward.
    unity_to_cv_world = np.diag([1.0, 1.0, -1.0])
    unity_cam_to_cv_cam = np.diag([1.0, -1.0, 1.0])

    cam_center_w = unity_to_cv_world @ pose["pos"]
    r_cv_c2w = unity_to_cv_world @ r_unity_c2w @ unity_cam_to_cv_cam
    r_cv_c2w = _nearest_rotation_matrix(r_cv_c2w)
    r_w2c = r_cv_c2w.T
    t_w2c = -r_w2c @ cam_center_w

    # XRCpuImage.Transformation.None gives the CPU image in sensor/image memory
    # orientation, while ARCamera.transform is in the Unity display camera frame.
    # PnP is solved in the saved image pixel frame, so the exported sparse camera
    # frame must be rotated into that same image frame.
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
    return r_w2c, t_w2c


def _rotmat_to_qvec(rot):
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

    qvec = np.array([qw, qx, qy, qz], dtype=np.float64)
    qvec /= np.linalg.norm(qvec) + 1e-12
    if qvec[0] < 0:
        qvec *= -1.0
    return qvec


def _image_size(image_path):
    with Image.open(image_path) as img:
        return img.size


def _intrinsics_for_pose(pose, image_width, image_height):
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
            return fx, fy, cx, cy, source

    focal = float(max(image_width, image_height))
    return focal, focal, (image_width - 1) * 0.5, (image_height - 1) * 0.5, source


def _write_phone_pose_sparse_model(dataset_dir):
    pose_path = os.path.join(dataset_dir, "poses.txt")
    image_dir = os.path.join(dataset_dir, "images")
    sparse_model_path = os.path.join(dataset_dir, "sparse", "0")
    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"Dataset images directory not found: {image_dir}")

    image_files = sorted(f for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    if not image_files:
        raise FileNotFoundError(f"No images found for phone-pose sparse model: {image_dir}")

    poses = _read_phone_poses(pose_path)
    missing = [name for name in image_files if name not in poses]
    if missing:
        preview = ", ".join(missing[:8])
        suffix = "" if len(missing) <= 8 else f", ... ({len(missing)} missing)"
        raise FileNotFoundError(f"Phone poses missing for selected frames: {preview}{suffix}")

    os.makedirs(sparse_model_path, exist_ok=True)
    cameras_txt = os.path.join(sparse_model_path, "cameras.txt")
    images_txt = os.path.join(sparse_model_path, "images.txt")
    points3d_txt = os.path.join(sparse_model_path, "points3D.txt")

    camera_lines = []
    image_lines = []
    intrinsics_sources = {"fallback_image_size": 0}

    for image_id, frame_name in enumerate(image_files, start=1):
        image_path = os.path.join(image_dir, frame_name)
        width, height = _image_size(image_path)
        pose = poses[frame_name]
        r_w2c, t_w2c = _unity_pose_to_colmap_w2c(pose)
        qvec = _rotmat_to_qvec(r_w2c)
        fx, fy, cx, cy, source = _intrinsics_for_pose(pose, width, height)
        intrinsics_sources[source] = intrinsics_sources.get(source, 0) + 1

        camera_id = image_id
        camera_lines.append(
            f"{camera_id} PINHOLE {width} {height} {fx:.10f} {fy:.10f} {cx:.10f} {cy:.10f}\n"
        )
        image_lines.append(
            f"{image_id} {qvec[0]:.12f} {qvec[1]:.12f} {qvec[2]:.12f} {qvec[3]:.12f} "
            f"{t_w2c[0]:.12f} {t_w2c[1]:.12f} {t_w2c[2]:.12f} {camera_id} {frame_name}\n\n"
        )

    with open(cameras_txt, "w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"# Number of cameras: {len(camera_lines)}\n")
        f.writelines(camera_lines)

    with open(images_txt, "w") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of images: {len(image_lines)}\n")
        f.writelines(image_lines)

    with open(points3d_txt, "w") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        f.write("# Number of points: 0, mean track length: 0\n")

    meta = {
        "pose_source": "unity_ar_pose",
        "pose_file": pose_path,
        "num_images": len(image_files),
        "intrinsics_sources": intrinsics_sources,
        "coordinate_conversion": {
            "world": "diag(1, 1, -1) * unity_world",
            "camera": "unity camera converted to COLMAP x-right y-down z-forward",
            "cpu_image_camera_from_pose_camera": "Rz(+90deg): [[0,-1,0],[1,0,0],[0,0,1]]",
            "image_transform": "new captures use XRCpuImage.Transformation.None; old MirrorY logs only adjust principal point",
        },
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(sparse_model_path, "phone_pose_meta.json"), "w") as f:
        json.dump(meta, f, indent=4)

    print(
        f">>> [调度] [PhonePose] 已写入 sparse/0: {len(image_files)} 帧, "
        f"内参来源 {intrinsics_sources}",
        flush=True,
    )
    return sparse_model_path


def _run_colmap_for_dataset(dataset_dir):
    if _colmap_sparse_ready(dataset_dir):
        print(f">>> [调度] COLMAP sparse 已存在，跳过重建: {dataset_dir}", flush=True)
        return os.path.join(dataset_dir, "sparse", "0")

    database_path = os.path.join(dataset_dir, "database.db")
    image_path = os.path.join(dataset_dir, "images")
    sparse_path = os.path.join(dataset_dir, "sparse")
    sparse_model_path = os.path.join(sparse_path, "0")

    if not os.path.isdir(image_path) or not os.listdir(image_path):
        raise FileNotFoundError(f"No images found for COLMAP: {image_path}")

    if os.path.exists(database_path):
        os.remove(database_path)
    shutil.rmtree(sparse_path, ignore_errors=True)
    os.makedirs(sparse_path, exist_ok=True)

    colmap_steps = [
        (
            "feature_extractor",
            [
                "colmap", "feature_extractor",
                "--database_path", database_path,
                "--image_path", image_path,
                "--ImageReader.single_camera", "1",
                "--SiftExtraction.use_gpu", "0",
            ],
        ),
        (
            "exhaustive_matcher",
            [
                "colmap", "exhaustive_matcher",
                "--database_path", database_path,
                "--SiftMatching.use_gpu", "0",
            ],
        ),
        (
            "mapper",
            [
                "colmap", "mapper",
                "--database_path", database_path,
                "--image_path", image_path,
                "--output_path", sparse_path,
            ],
        ),
        (
            "model_converter_txt",
            [
                "colmap", "model_converter",
                "--input_path", sparse_model_path,
                "--output_path", sparse_model_path,
                "--output_type", "TXT",
            ],
        ),
        (
            "model_converter_ply",
            [
                "colmap", "model_converter",
                "--input_path", sparse_model_path,
                "--output_path", os.path.join(sparse_model_path, "model.ply"),
                "--output_type", "PLY",
            ],
        ),
    ]

    for step_name, cmd in colmap_steps:
        print(f">>> [调度] [COLMAP] {step_name}...", flush=True)
        subprocess.run(cmd, cwd=dataset_dir, check=True)

    if not _colmap_sparse_ready(dataset_dir):
        raise FileNotFoundError(f"COLMAP finished but sparse TXT files are missing: {sparse_model_path}")
    return sparse_model_path


def _run_pose_scale_optimization(dataset_name):
    script_path = os.path.join(COARSEMODEL_DIR, "estimation_4stage_defo_fin.py")
    infer_config = os.path.join(COARSEMODEL_DIR, "configs", "infer", dataset_name, f"{dataset_name}.json")
    if not os.path.exists(infer_config):
        raise FileNotFoundError(f"Infer config not found: {infer_config}")

    print(f">>> [调度] [CoarseModel 3/3] 开始位姿/尺度优化: {dataset_name}", flush=True)
    subprocess.run(
        [sys.executable, "-u", script_path, "--dataset_name", dataset_name],
        cwd=COARSEMODEL_DIR,
        check=True,
    )

    output_dir = os.path.join(COARSEMODEL_DIR, "results", "refine_model", dataset_name)
    refined_model = os.path.join(output_dir, "refine", "refined_model.obj")
    if not os.path.exists(refined_model):
        raise FileNotFoundError(f"Pose optimization finished but refined model is missing: {refined_model}")

    print(f">>> [调度] [CoarseModel 3/3] 位姿/尺度优化完成: {refined_model}", flush=True)
    return output_dir, refined_model


def _invalidate_optimization_cache(dataset_name):
    cache_path = os.path.join(
        COARSEMODEL_DIR,
        "configs",
        "infer",
        dataset_name,
        dataset_name,
        "cached_optimization_data.pkl",
    )
    if os.path.exists(cache_path):
        os.remove(cache_path)
        print(f">>> [调度] 已清理旧优化缓存，强制使用当前手机位姿: {cache_path}", flush=True)


def _notify_postprocess_done():
    os.makedirs(FLAG_DIR, exist_ok=True)
    with open(FLAG_POSTPROCESS_DONE, "w") as f:
        f.write("done")

@app.route('/start_record', methods=['POST'])
def start_record():
    global frame_counter
    print("\n>>> [调度] 收到开始录制请求，正在创建新采集 session...")
    clean_environment()
    session_id = _set_current_session(_make_session_id(), reset_points=True)
    frame_counter = 0
    return jsonify({
        "status": "ready",
        "session_id": session_id,
        "data_dir": current_data_dir,
        "preview_dir": current_preview_dir,
    }), 200

@app.route('/upload', methods=['POST'])
def upload():
    global frame_counter
    try:
        _load_current_session()
        pos_x, pos_y, pos_z = request.form.get('pos_x'), request.form.get('pos_y'), request.form.get('pos_z')
        rot_x, rot_y, rot_z = request.form.get('rot_x'), request.form.get('rot_y'), request.form.get('rot_z')
        quat_x = request.form.get('quat_x', '')
        quat_y = request.form.get('quat_y', '')
        quat_z = request.form.get('quat_z', '')
        quat_w = request.form.get('quat_w', '')
        fx = request.form.get('fx', '')
        fy = request.form.get('fy', '')
        cx = request.form.get('cx', '')
        cy = request.form.get('cy', '')
        intrinsic_width = request.form.get('intrinsic_width', '')
        intrinsic_height = request.form.get('intrinsic_height', '')
        upload_image_width = request.form.get('image_width', '')
        upload_image_height = request.form.get('image_height', '')
        cpu_image_width = request.form.get('cpu_image_width', '')
        cpu_image_height = request.form.get('cpu_image_height', '')
        image_transform = request.form.get('image_transform', 'unknown')

        img_bytes = request.files['image'].read()
        img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode uploaded image")
        image_height, image_width = img.shape[:2]

        frame_name = f"frame_{frame_counter:04d}.jpg"
        cv2.imwrite(os.path.join(current_data_dir, frame_name), img)
        
        with open(os.path.join(current_data_dir, "poses.txt"), "a") as f:
            f.write(
                f"{frame_name},{pos_x},{pos_y},{pos_z},{rot_x},{rot_y},{rot_z},"
                f"{quat_x},{quat_y},{quat_z},{quat_w},"
                f"{fx},{fy},{cx},{cy},{intrinsic_width},{intrinsic_height},"
                f"{upload_image_width or image_width},{upload_image_height or image_height},"
                f"{cpu_image_width},{cpu_image_height},{image_transform}\n"
            )

        frame_counter += 1
        return jsonify({"status": "success"}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/preprocess', methods=['POST'])
def preprocess():
    _load_current_session()
    image_files = _list_session_images()
    print(f"\n>>> [调度] 停止录制，进入用户监督原图分割阶段，共 {len(image_files)} 张")
    return jsonify({
        "status": "needs_segmentation",
        "total_images": len(image_files),
        "session_id": current_session_id,
    }), 200


@app.route('/get_frame/<int:img_id>', methods=['GET'])
def get_frame(img_id):
    _load_current_session()
    image_files = _list_session_images()
    if img_id < 0 or img_id >= len(image_files):
        return jsonify({"status": "error", "message": "frame index out of range"}), 404
    return send_from_directory(current_data_dir, image_files[img_id])


@app.route('/add_seg_point', methods=['POST'])
def add_seg_point():
    _load_current_session()
    data = request.get_json(force=True) or {}
    frame_index = int(data.get("frame_index", -1))
    x = float(data.get("x", -1.0))
    y = float(data.get("y", -1.0))
    label = int(data.get("label", 1))
    normalized = bool(data.get("normalized", True))

    image_files = _list_session_images()
    if frame_index < 0 or frame_index >= len(image_files):
        return jsonify({"status": "error", "message": "frame index out of range"}), 400
    if label not in (0, 1):
        return jsonify({"status": "error", "message": "label must be 0 or 1"}), 400

    point = {
        "frame_index": frame_index,
        "x": x,
        "y": y,
        "label": label,
        "normalized": normalized,
    }
    current_seg_points.append(point)
    if frame_index in current_seed_frames:
        current_seed_frames.discard(frame_index)
        _save_seed_frames()
    _save_seg_points()

    image_path = os.path.join(current_data_dir, image_files[frame_index])
    with Image.open(image_path) as image:
        pixel_x, pixel_y = _point_to_pixel(point, image.size)
    print(
        f">>> [分割点] frame={frame_index}, label={label}, norm=({x:.4f},{y:.4f}), pixel=({pixel_x},{pixel_y})",
        flush=True,
    )
    return jsonify({
        "status": "ok",
        "num_points": len(current_seg_points),
        "seed_frames": sorted(current_seed_frames),
        "pixel_x": pixel_x,
        "pixel_y": pixel_y,
    }), 200


@app.route('/approve_seed_frame/<int:img_id>', methods=['POST'])
def approve_seed_frame(img_id):
    _load_current_session()
    image_files = _list_session_images()
    if img_id < 0 or img_id >= len(image_files):
        return jsonify({"status": "error", "message": "frame index out of range"}), 400

    frame_points = [p for p in current_seg_points if int(p.get("frame_index", -1)) == int(img_id)]
    if not frame_points:
        return jsonify({"status": "error", "message": "No prompt points on this frame"}), 400

    stem = os.path.splitext(image_files[img_id])[0]
    prompt_mask = os.path.join(current_mask_dir, f"{stem}_prompt.png")
    if not os.path.exists(prompt_mask):
        _run_single_frame_segmentation_preview(img_id)

    current_seed_frames.add(int(img_id))
    _save_seed_frames()
    print(f">>> [种子帧] 已确认 frame={img_id}, seed_frames={sorted(current_seed_frames)}", flush=True)
    return jsonify({
        "status": "ok",
        "seed_frames": sorted(current_seed_frames),
    }), 200


@app.route('/unapprove_seed_frame/<int:img_id>', methods=['POST'])
def unapprove_seed_frame(img_id):
    _load_current_session()
    current_seed_frames.discard(int(img_id))
    _save_seed_frames()
    return jsonify({
        "status": "ok",
        "seed_frames": sorted(current_seed_frames),
    }), 200


@app.route('/segment_frame/<int:img_id>', methods=['POST'])
def segment_frame(img_id):
    try:
        _run_single_frame_segmentation_preview(img_id)
        return jsonify({
            "status": "segmented",
            "preview_url": f"/get_prompt_preview/{img_id}",
        }), 200
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(exc)}), 500


@app.route('/clear_seg_points', methods=['POST'])
def clear_seg_points():
    _load_current_session()
    current_seg_points.clear()
    current_seed_frames.clear()
    _save_seg_points()
    _save_seed_frames()
    return jsonify({"status": "ok", "num_points": 0}), 200


@app.route('/run_segmentation', methods=['POST'])
def run_segmentation():
    try:
        total = _run_interactive_segmentation()
        print(f">>> [调度] 用户监督 SAM2 video 分割完成，生成 {total} 张裁剪 preview")
        return jsonify({
            "status": "segmented",
            "total_images": total,
            "num_points": len(current_seg_points),
            "seed_frames": sorted(current_seed_frames),
        }), 200
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(exc)}), 500


@app.route('/input_qc', methods=['POST'])
def input_qc():
    try:
        _load_current_session()
        payload = request.get_json(silent=True) or {}
        image_files = _list_session_images()
        selected_indices = payload.get("selected")
        if selected_indices is None:
            selected_indices = list(range(len(image_files)))
        selected_indices = [int(i) for i in selected_indices]
        filtered_indices, report = arq.select_frames_for_reconstruction(
            selected_indices,
            image_files,
            current_data_dir,
            current_preview_dir,
        )
        return jsonify({
            "status": "ok" if report.get("qc_pass") else "warning",
            "message": "input qc passed" if report.get("qc_pass") else (
                "输入帧覆盖不足: " + "; ".join(report.get("fail_reasons") or ["unknown reason"])
            ),
            "selected_indices": filtered_indices,
            "input_qc": report,
        }), 200
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(exc)}), 500

@app.route('/get_preview/<int:img_id>', methods=['GET'])
def get_preview(img_id):
    _load_current_session()
    review_path = os.path.join(current_review_dir, f"{img_id}.png")
    if os.path.exists(review_path):
        return send_from_directory(current_review_dir, f"{img_id}.png")
    return send_from_directory(current_preview_dir, f"{img_id}.png")


@app.route('/get_prompt_preview/<int:img_id>', methods=['GET'])
def get_prompt_preview(img_id):
    _load_current_session()
    prompt_name = f"{img_id}_prompt.png"
    if not os.path.exists(os.path.join(current_review_dir, prompt_name)):
        return jsonify({"status": "error", "message": "prompt preview not found"}), 404
    return send_from_directory(current_review_dir, prompt_name)

# ================= 新增：取消并重置接口 =================
@app.route('/cancel_review', methods=['POST'])
def cancel_review():
    global frame_counter
    print("\n>>> [调度] 收到手机端【重置/退出】指令！强制中断并重置管线...")
    
    # 给 run_local 发送一个空列表，让它立刻中断当前的生成并退回开头
    with open(FLAG_START_GENERATE, 'w') as f:
        json.dump({"selected": []}, f)
        
    # ⚠️ 极其关键：暂停 0.5 秒，给后台的 run_local.py 一点点时间去读取上面的中断指令
    time.sleep(0.5) 
    
    clean_environment()
    _set_current_session(_make_session_id(), reset_points=True)
    frame_counter = 0
    return jsonify({"status": "cancelled"}), 200
# ==============================================================

@app.route('/generate', methods=['POST'])
def generate():
    _load_current_session()
    selected_indices = [int(i) for i in request.json.get('selected', [])]
    print(f"\n>>> [调度] 收到手机发来的保留序号 {selected_indices}，通知后台筛选并生成...", flush=True)
    image_files = _list_session_images()
    filtered_indices, input_qc_report = arq.select_frames_for_reconstruction(
        selected_indices,
        image_files,
        current_data_dir,
        current_preview_dir,
    )
    print(
        ">>> [调度] 输入帧 QC: "
        f"pass={input_qc_report.get('qc_pass')} "
        f"selected={filtered_indices} "
        f"coverage={input_qc_report.get('coverage', {})}",
        flush=True,
    )
    if arq.enforce_input_qc_from_env() and not input_qc_report.get("qc_pass", True):
        message = "输入帧覆盖不足，建议继续绕物体采集后重试: " + "; ".join(
            input_qc_report.get("fail_reasons") or ["unknown reason"]
        )
        return jsonify({
            "status": "error",
            "message": message,
            "client_selected_indices": selected_indices,
            "filtered_selected_indices": filtered_indices,
            "frame_filter": input_qc_report,
            "input_qc": input_qc_report,
        }), 400
    
    with open(FLAG_START_GENERATE, 'w') as f:
        json.dump(
            {
                "selected": filtered_indices,
                "client_selected": selected_indices,
                "input_qc": input_qc_report,
            },
            f,
            indent=4,
        )
        
    for _ in range(600):
        if os.path.exists(FLAG_GENERATE_DONE):
            generate_done = {}
            try:
                with open(FLAG_GENERATE_DONE, "r") as f:
                    generate_done = json.load(f)
            except (json.JSONDecodeError, OSError):
                generate_done = {}
            os.remove(FLAG_GENERATE_DONE)
            filtered_selected_indices = generate_done.get("selected_indices", selected_indices)
            frame_filter = generate_done.get("frame_filter", {"enabled": False, "reason": "not_reported"})
            if generate_done.get("status") == "error":
                _notify_postprocess_done()
                return jsonify({
                    "status": "error",
                    "message": generate_done.get("message", "ReconViaGen generation failed."),
                    "client_selected_indices": selected_indices,
                    "filtered_selected_indices": filtered_selected_indices,
                    "frame_filter": frame_filter,
                }), 500
            print(">>> [调度] 后台生成成功，开始整理 CoarseModel 数据并生成模板/特征...", flush=True)
            try:
                recon_output_dir = generate_done.get("output_dir")
                dataset_name, dataset_dir = _prepare_coarsemodel_dataset(filtered_selected_indices, recon_output_dir)
                print(f">>> [调度] CoarseModel 数据集已准备: {dataset_dir}", flush=True)
                colmap_dir = _write_phone_pose_sparse_model(dataset_dir)
                template_config, repre_config = _generate_coarse_templates_and_repre(dataset_name)
                _invalidate_optimization_cache(dataset_name)
                refine_output_dir, refined_model = _run_pose_scale_optimization(dataset_name)
            except Exception as e:
                import traceback
                traceback.print_exc()
                _notify_postprocess_done()
                return jsonify({
                    "status": "error",
                    "message": f"Mesh generated, but CoarseModel postprocess failed: {e}",
                }), 500

            _notify_postprocess_done()
            print(f">>> [调度] CoarseModel 后处理全部完成: {dataset_name}", flush=True)
            return jsonify({
                "status": "success",
                "message": "Mesh, phone-pose sparse model, templates, object representation, and refined pose/scale model generated.",
                "dataset_name": dataset_name,
                "dataset_dir": dataset_dir,
                "session_id": current_session_id,
                "session_data_dir": current_data_dir,
                "client_selected_indices": selected_indices,
                "filtered_selected_indices": filtered_selected_indices,
                "frame_filter": frame_filter,
                "input_qc": input_qc_report,
                "recon_output_dir": recon_output_dir,
                "pose_source": "unity_ar_pose",
                "colmap_dir": colmap_dir,
                "template_config": template_config,
                "repre_config": repre_config,
                "refine_output_dir": refine_output_dir,
                "refined_model": refined_model,
            }), 200
        time.sleep(1)

    return jsonify({"status": "error", "message": "后台生成超时"}), 500

if __name__ == '__main__':
    clean_environment()
    print(f">>> [调度] 正在由 Server 自动唤醒后台 3D 重建进程...")
    script_path = os.path.join(RECONVIAGEN_DIR, "run_local.py")
    try:
        subprocess.Popen(
            [
                "conda", "run", "-n", "reconviagen", 
                "--no-capture-output", "python", "-u", script_path
            ], 
            cwd=RECONVIAGEN_DIR
        )
    except Exception as e:
        print(f"❌ 自动拉起 run_local.py 失败: {e}")
    
    print(f"服务器启动，作为调度器监听 5000 端口...")
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
