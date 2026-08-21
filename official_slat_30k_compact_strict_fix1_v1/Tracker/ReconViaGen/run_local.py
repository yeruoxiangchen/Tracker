import sys
import os
import time
import json
import math

# ================= 强行注入 CUDA 12.1 环境变量 =================
os.environ['CUDA_HOME'] = '/home/zjr/cuda-12.1'
# 将 nvcc 所在的 bin 目录强行塞到 PATH 的最前面，确保系统第一眼就能找到它
os.environ['PATH'] = f"/home/zjr/cuda-12.1/bin:{os.environ.get('PATH', '')}"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "3")
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# 强制注入自定义依赖路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "wheels/vggt"))

os.environ['SPCONV_ALGO'] = 'native'
import torch
import numpy as np
import imageio
from PIL import Image
import argparse
import cv2
from scipy.spatial.transform import Rotation as R  # 用于处理欧拉角转矩阵

from trellis.pipelines import TrellisVGGTTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils
import ar_pose_quality as arq

DEFAULT_CANDIDATE_SEEDS = [0, 1]
MAX_CANDIDATE_COUNT = 3
DEFAULT_MESH_SIMPLIFY = 0.75

# 1. 统一目录与信号文件配置
TRACKER_DIR = os.path.join(BASE_DIR, "ar_tracker")
DATA_ROOT = os.path.join(TRACKER_DIR, "data")
PREVIEW_ROOT = os.path.join(TRACKER_DIR, "previews")
OUTPUT_DIR = os.path.join(TRACKER_DIR, "output")
FLAG_DIR = os.path.join(TRACKER_DIR, "flags")

for d in [DATA_ROOT, PREVIEW_ROOT, OUTPUT_DIR, FLAG_DIR]:
    os.makedirs(d, exist_ok=True)

FLAG_START_PREPROCESS = os.path.join(FLAG_DIR, "start_preprocess.flag")
FLAG_PREPROCESS_DONE = os.path.join(FLAG_DIR, "preprocess_done.json")
FLAG_START_GENERATE = os.path.join(FLAG_DIR, "start_generate.json")
FLAG_GENERATE_DONE = os.path.join(FLAG_DIR, "generate_done.flag")
FLAG_POSTPROCESS_DONE = os.path.join(FLAG_DIR, "postprocess_done.flag")
FLAG_CURRENT_SESSION = os.path.join(FLAG_DIR, "current_session.json")

def get_current_session_dirs():
    if os.path.exists(FLAG_CURRENT_SESSION):
        with open(FLAG_CURRENT_SESSION, "r") as f:
            data = json.load(f)
        data_dir = data.get("data_dir") or os.path.join(DATA_ROOT, data["session_id"])
        preview_dir = data.get("preview_dir") or os.path.join(PREVIEW_ROOT, data["session_id"])
    else:
        data_dir = DATA_ROOT
        preview_dir = PREVIEW_ROOT
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(preview_dir, exist_ok=True)
    return data_dir, preview_dir


def wait_for_server_postprocess_ack():
    print("[Worker] 等待 server.py 确认已处理生成结果/错误...")
    while not os.path.exists(FLAG_POSTPROCESS_DONE):
        time.sleep(1)
    os.remove(FLAG_POSTPROCESS_DONE)

def init_pipeline():
    print("Initializing Pipeline into VRAM...")
    pipeline = TrellisVGGTTo3DPipeline.from_pretrained("Stable-X/trellis-vggt-v0-2")
    pipeline._device = torch.device('cuda')
    pipeline.low_vram = True
    pipeline.birefnet_model.cuda()
    return pipeline


def make_unique_output_dir(root_dir):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(root_dir, timestamp)
    suffix = 1
    while os.path.exists(output_dir):
        output_dir = os.path.join(root_dir, f"{timestamp}_{suffix:02d}")
        suffix += 1
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def get_candidate_seeds():
    raw_seeds = os.environ.get("RECON_CANDIDATE_SEEDS", "").strip()
    seeds = []
    if raw_seeds:
        for item in raw_seeds.split(","):
            item = item.strip()
            if not item:
                continue
            try:
                seeds.append(int(item))
            except ValueError:
                print(f"[Worker] 忽略非法 RECON_CANDIDATE_SEEDS 项: {item}")
    else:
        raw_count = os.environ.get("RECON_NUM_CANDIDATES", "").strip()
        try:
            count = int(raw_count) if raw_count else len(DEFAULT_CANDIDATE_SEEDS)
        except ValueError:
            count = len(DEFAULT_CANDIDATE_SEEDS)
        count = max(1, min(count, MAX_CANDIDATE_COUNT))
        seeds = list(range(count))

    unique = []
    for seed in seeds:
        if seed not in unique:
            unique.append(seed)
        if len(unique) >= MAX_CANDIDATE_COUNT:
            break
    return unique or DEFAULT_CANDIDATE_SEEDS[:1]


def get_mesh_simplify_ratio():
    raw_value = os.environ.get("RECON_MESH_SIMPLIFY", "").strip()
    if not raw_value:
        return DEFAULT_MESH_SIMPLIFY
    try:
        value = float(raw_value)
    except ValueError:
        print(f"[Worker] 忽略非法 RECON_MESH_SIMPLIFY={raw_value}，使用 {DEFAULT_MESH_SIMPLIFY}")
        return DEFAULT_MESH_SIMPLIFY
    return float(np.clip(value, 0.0, 0.95))


def rgba_mask_stats(image):
    rgba = np.array(image.convert("RGBA"))
    mask = rgba[:, :, 3] > 128
    coverage = float(np.mean(mask)) if mask.size else 0.0
    stats = {
        "width": int(image.width),
        "height": int(image.height),
        "coverage": coverage,
        "bbox": None,
        "bbox_coverage": 0.0,
        "bbox_aspect": 0.0,
    }
    if not np.any(mask):
        return stats

    ys, xs = np.nonzero(mask)
    left, right = int(xs.min()), int(xs.max())
    top, bottom = int(ys.min()), int(ys.max())
    bbox_w = max(1, right - left + 1)
    bbox_h = max(1, bottom - top + 1)
    stats.update(
        {
            "bbox": [left, top, right, bottom],
            "bbox_coverage": float(mask.sum() / float(bbox_w * bbox_h)),
            "bbox_aspect": float(bbox_w / float(bbox_h)),
        }
    )
    return stats


def save_generation_inputs(output_dir, selected_images, selected_names, selected_indices, client_selected_indices, frame_filter, candidate_seeds):
    inputs_dir = os.path.join(output_dir, "inputs")
    masks_dir = os.path.join(inputs_dir, "masks")
    os.makedirs(inputs_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)

    input_stats = []
    for order, (image, name, frame_idx) in enumerate(zip(selected_images, selected_names, selected_indices)):
        safe_name = os.path.splitext(os.path.basename(name))[0]
        image_path = os.path.join(inputs_dir, f"{order:03d}_frame_{frame_idx:04d}_{safe_name}.png")
        mask_path = os.path.join(masks_dir, f"{order:03d}_frame_{frame_idx:04d}_{safe_name}_mask.png")
        rgba = image.convert("RGBA")
        rgba.save(image_path, "PNG")
        alpha = Image.fromarray(np.array(rgba)[:, :, 3], mode="L")
        alpha.save(mask_path, "PNG")
        stats = rgba_mask_stats(rgba)
        stats.update(
            {
                "order": int(order),
                "frame_index": int(frame_idx),
                "frame_name": name,
                "image_path": image_path,
                "mask_path": mask_path,
            }
        )
        input_stats.append(stats)

    if selected_images:
        thumb_size = 160
        cols = min(6, len(selected_images))
        rows = int(math.ceil(len(selected_images) / float(cols)))
        sheet = Image.new("RGBA", (cols * thumb_size, rows * thumb_size), (0, 0, 0, 255))
        for order, image in enumerate(selected_images):
            thumb = image.convert("RGBA").resize((thumb_size, thumb_size), Image.Resampling.LANCZOS)
            x = (order % cols) * thumb_size
            y = (order // cols) * thumb_size
            sheet.paste(thumb, (x, y))
        sheet.save(os.path.join(inputs_dir, "contact_sheet.png"), "PNG")

    manifest = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "client_selected_indices": [int(i) for i in client_selected_indices],
        "selected_indices": [int(i) for i in selected_indices],
        "selected_names": selected_names,
        "frame_filter": frame_filter,
        "candidate_seeds": [int(seed) for seed in candidate_seeds],
        "input_stats": input_stats,
        "note": "These PNGs are the exact segmented previews passed to ReconViaGen.",
    }
    with open(os.path.join(inputs_dir, "selection_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=4)
    return manifest


def score_mesh_candidate(mesh):
    metrics = {
        "score": 0.0,
        "vertex_count": 0,
        "face_count": 0,
        "bbox_extent": [0.0, 0.0, 0.0],
        "extent_balance": 0.0,
        "face_score": 0.0,
        "silhouette_score": 0.0,
        "silhouette_coverage_mean": 0.0,
        "silhouette_coverage_std": 0.0,
        "silhouette_largest_component_mean": 0.0,
    }

    try:
        verts = mesh.vertices.detach().float().cpu().numpy()
        faces = mesh.faces.detach().cpu().numpy()
    except Exception as exc:
        metrics["error"] = f"mesh_tensor_read_failed: {exc}"
        return metrics

    metrics["vertex_count"] = int(len(verts))
    metrics["face_count"] = int(len(faces))
    if len(verts) < 32 or len(faces) < 32:
        metrics["error"] = "too_few_vertices_or_faces"
        return metrics

    extent = np.ptp(verts, axis=0)
    extent = np.maximum(extent, 1e-6)
    sorted_extent = np.sort(extent)[::-1]
    extent_balance = float(sorted_extent[-1] / sorted_extent[0])
    metrics["bbox_extent"] = [float(x) for x in extent.tolist()]
    metrics["extent_balance"] = extent_balance
    extent_score = float(np.clip(extent_balance / 0.22, 0.0, 1.0))
    face_score = float(np.clip(math.log1p(len(faces)) / math.log1p(18000), 0.0, 1.0))
    metrics["face_score"] = face_score

    silhouette_score = 0.5
    try:
        video_geo = render_utils.render_video(
            mesh,
            resolution=192,
            ssaa=1,
            num_frames=16,
        )["normal"]
        coverages = []
        largest_components = []
        for frame in video_geo:
            arr = np.asarray(frame)
            if arr.ndim < 3:
                continue
            rgb = arr[:, :, :3].astype(np.float32)
            threshold = 2.0 if rgb.max() > 2.0 else 1e-3
            mask = np.linalg.norm(rgb, axis=-1) > threshold
            coverage = float(np.mean(mask))
            coverages.append(coverage)
            if np.any(mask):
                num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
                if num_labels > 1:
                    largest = float(stats[1:, cv2.CC_STAT_AREA].max())
                    largest_components.append(largest / float(mask.sum()))
                else:
                    largest_components.append(0.0)
            else:
                largest_components.append(0.0)
        if coverages:
            cov_mean = float(np.mean(coverages))
            cov_std = float(np.std(coverages))
            largest_mean = float(np.mean(largest_components)) if largest_components else 0.0
            area_ok = 1.0 if 0.015 <= cov_mean <= 0.85 else 0.2
            stability = float(np.clip(1.0 - cov_std / (cov_mean + 1e-6), 0.0, 1.0))
            silhouette_score = area_ok * (0.6 * stability + 0.4 * largest_mean)
            metrics["silhouette_coverage_mean"] = cov_mean
            metrics["silhouette_coverage_std"] = cov_std
            metrics["silhouette_largest_component_mean"] = largest_mean
        del video_geo
    except Exception as exc:
        metrics["silhouette_error"] = str(exc)

    metrics["silhouette_score"] = float(silhouette_score)
    metrics["score"] = float(0.45 * extent_score + 0.25 * face_score + 0.30 * silhouette_score)
    return metrics


def run_limited_candidate_generation(
    pipeline,
    selected_images,
    selected_names,
    data_dir,
    candidate_seeds,
    pose_rerank_enabled=True,
    pose_rerank_weight=0.45,
):
    best = None
    reports = []
    for order, seed in enumerate(candidate_seeds):
        print(f"[Worker] 候选生成 {order + 1}/{len(candidate_seeds)}，seed={seed}")
        outputs, coords, ss_noise = pipeline.run(
            image=selected_images,
            seed=seed,
            formats=["gaussian", "mesh"],
            preprocess_image=False,
            mode="multidiffusion",
        )
        gs, mesh = outputs["gaussian"][0], outputs["mesh"][0]
        metrics = score_mesh_candidate(mesh)
        pose_sanity = {"enabled": False, "reason": "disabled", "score": None}
        if pose_rerank_enabled:
            pose_sanity = arq.pose_mask_sanity_for_mesh(
                mesh,
                selected_images=selected_images,
                selected_names=selected_names,
                data_dir=data_dir,
            )
        metrics["pose_mask_sanity"] = pose_sanity
        pose_score = pose_sanity.get("score")
        if pose_sanity.get("enabled") and pose_score is not None:
            metrics["selection_score"] = float(metrics["score"] + pose_rerank_weight * float(pose_score))
        else:
            metrics["selection_score"] = float(metrics["score"])
        metrics.update({"seed": int(seed), "candidate_index": int(order)})
        reports.append({k: v for k, v in metrics.items() if k not in {"outputs", "coords", "ss_noise"}})
        print(
            f"[Worker] 候选 seed={seed} base_score={metrics['score']:.4f} "
            f"selection_score={metrics['selection_score']:.4f} "
            f"pose_sanity={pose_sanity.get('score')}"
        )

        candidate = {
            "seed": seed,
            "outputs": outputs,
            "coords": coords,
            "ss_noise": ss_noise,
            "metrics": metrics,
        }
        if best is None or metrics["selection_score"] > best["metrics"]["selection_score"]:
            if best is not None:
                del best["outputs"], best["coords"], best["ss_noise"]
                torch.cuda.empty_cache()
            best = candidate
        else:
            del outputs, coords, ss_noise, candidate
            torch.cuda.empty_cache()

    if best is None:
        raise RuntimeError("No ReconViaGen candidate was generated.")
    return best, reports

def load_presegmented_previews(data_dir, preview_dir):
    print(f"\n[Worker] 读取 server.py 已生成的用户监督分割 preview...")
    valid_exts = ('.png', '.jpg', '.jpeg')
    image_files = sorted([f for f in os.listdir(data_dir) if f.lower().endswith(valid_exts)])
    
    if not image_files:
        print("[Worker] 警告: 没有找到图片！")
        return []

    processed_images_with_names = []
    for idx, img_name in enumerate(image_files):
        preview_path = os.path.join(preview_dir, f"{idx}.png")
        if not os.path.exists(preview_path):
            raise FileNotFoundError(
                f"Missing segmented preview {preview_path}. Run interactive segmentation before generate."
            )
        processed_img = Image.open(preview_path).convert("RGBA")
        processed_images_with_names.append((processed_img, img_name))
        print(f"\r  读取 preview -> {idx}.png", end="", flush=True)
        
    print("", flush=True)
    return processed_images_with_names


def read_pose_dict(data_dir):
    pose_file = os.path.join(data_dir, "poses.txt")
    pose_dict = {}
    if os.path.exists(pose_file):
        with open(pose_file, "r") as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 7:
                    try:
                        pose_dict[parts[0]] = {
                            'pos': [float(x) for x in parts[1:4]],
                            'rot': [float(x) for x in parts[4:7]]
                        }
                    except ValueError:
                        continue
    return pose_dict


def frame_quality_score(frame_index, images_with_names, data_dir, preview_dir):
    if frame_index < 0 or frame_index >= len(images_with_names):
        return 0.0

    coverage_score = 0.5
    preview_path = os.path.join(preview_dir, f"{frame_index}.png")
    preview = cv2.imread(preview_path, cv2.IMREAD_UNCHANGED)
    if preview is not None:
        if preview.ndim == 3 and preview.shape[2] == 4:
            mask = preview[:, :, 3] > 128
        elif preview.ndim == 3:
            mask = cv2.cvtColor(preview, cv2.COLOR_BGR2GRAY) > 0
        else:
            mask = preview > 0
        coverage = float(np.mean(mask))
        coverage_score = max(0.0, 1.0 - abs(coverage - 0.45) / 0.45)

    sharpness_score = 0.5
    image_name = images_with_names[frame_index][1]
    img = cv2.imread(os.path.join(data_dir, image_name), cv2.IMREAD_GRAYSCALE)
    if img is not None:
        sharpness = float(cv2.Laplacian(img, cv2.CV_64F).var())
        sharpness_score = min(sharpness / 500.0, 1.0)

    return 0.65 * coverage_score + 0.35 * sharpness_score


def filter_selected_frame_indices(selected_indices, images_with_names, data_dir, preview_dir, max_frames=18):
    valid = sorted({int(i) for i in selected_indices if 0 <= int(i) < len(images_with_names)})
    if len(valid) <= max_frames:
        return valid, {"enabled": False, "reason": "under_limit", "max_frames": max_frames}

    pose_dict = read_pose_dict(data_dir)
    pose_candidates = []
    for idx in valid:
        frame_name = images_with_names[idx][1]
        if frame_name in pose_dict:
            pose_candidates.append((idx, np.array(pose_dict[frame_name]["pos"], dtype=np.float64)))

    score = lambda idx: frame_quality_score(idx, images_with_names, data_dir, preview_dir)
    if len(pose_candidates) < max(6, max_frames // 2):
        ranked = sorted(valid, key=score, reverse=True)[:max_frames]
        return sorted(ranked), {
            "enabled": True,
            "reason": "quality_only_no_pose",
            "max_frames": max_frames,
            "input_count": len(valid),
            "output_count": len(ranked),
        }

    positions = np.array([item[1] for item in pose_candidates], dtype=np.float64)
    centroid = positions.mean(axis=0)
    angle_by_idx = {}
    for idx, pos in pose_candidates:
        angle_by_idx[idx] = (np.degrees(np.arctan2(pos[2] - centroid[2], pos[0] - centroid[0])) + 360.0) % 360.0

    bin_size = 360.0 / float(max_frames)
    bins = {}
    for idx in valid:
        if idx not in angle_by_idx:
            continue
        bin_id = int(angle_by_idx[idx] // bin_size)
        quality = score(idx)
        current = bins.get(bin_id)
        if current is None or quality > current[1]:
            bins[bin_id] = (idx, quality)

    chosen = [item[0] for item in bins.values()]
    if len(chosen) < max_frames:
        remaining = [idx for idx in valid if idx not in chosen]
        remaining = sorted(remaining, key=score, reverse=True)
        chosen.extend(remaining[:max_frames - len(chosen)])
    elif len(chosen) > max_frames:
        chosen = sorted(chosen, key=score, reverse=True)[:max_frames]

    chosen = sorted(chosen)
    return chosen, {
        "enabled": True,
        "reason": "angle_balanced_quality",
        "max_frames": max_frames,
        "input_count": len(valid),
        "output_count": len(chosen),
        "selected_indices": chosen,
    }

def get_normalized_camera_matrices(pose_dict, selected_names, fov_degrees=60.0, resolution=518):
    """
    读取、居中、缩放并转换所有相机的位姿
    """
    poses = [pose_dict.get(name, {'pos':[0,0,0], 'rot':[0,0,0]}) for name in selected_names]
    
    # 1. 提取所有位置，计算质心 (假设你的相机是环绕物体拍摄的，质心即为物体中心)
    positions = np.array([p['pos'] for p in poses])
    centroid = np.mean(positions, axis=0)
    centered_positions = positions - centroid
    
    # 2. 计算缩放因子 (将平均相机距离缩放到模型期望的 1.5 半径)
    avg_radius = np.mean(np.linalg.norm(centered_positions, axis=1))
    scale_factor = 1.5 / (avg_radius + 1e-6) # 避免除 0
    
    extrinsics_list = []
    intrinsics_list = []
    
    # 提前算好内参 (所有图片共用)
    focal_length = (resolution / 2.0) / math.tan(math.radians(fov_degrees / 2.0))
    K = np.array([
        [focal_length, 0, resolution / 2.0],
        [0, focal_length, resolution / 2.0],
        [0, 0, 1]
    ], dtype=np.float32)
    
    for i, p in enumerate(poses):
        # 应用居中和缩放
        pos = centered_positions[i] * scale_factor
        
        # 3. 坐标系转换: Unity (LHS, Z-forward, Y-up) -> 常见右手系 (如 OpenGL: Z-backward, Y-up)
        # 反转 Z 轴即可完成左手到右手的平移转换
        pos_rhs = np.array([pos[0], pos[1], -pos[2]])
        
        # 处理旋转矩阵的转换
        r = R.from_euler('zxy', [p['rot'][2], p['rot'][0], p['rot'][1]], degrees=True)
        rot_matrix = r.as_matrix()
        
        # 反转 Z 轴对应的旋转矩阵行列符号
        rot_matrix[:, 2] = -rot_matrix[:, 2]
        rot_matrix[2, :] = -rot_matrix[2, :]
        
        # 构造 Camera-to-World (C2W)
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, :3] = rot_matrix
        c2w[:3, 3] = pos_rhs
        
        # Trellis 需要 World-to-Camera (W2C)
        w2c = np.linalg.inv(c2w)
        
        extrinsics_list.append(w2c)
        intrinsics_list.append(K)
        
    return extrinsics_list, intrinsics_list

def main():
    pipeline = init_pipeline()
    
    while True:
        print("\n" + "="*50)
        print("[Worker] 模型已就绪，正在后台监听来自 server.py 的指令...")
        
        while not os.path.exists(FLAG_START_GENERATE):
            time.sleep(1)
            
        with open(FLAG_START_GENERATE, 'r') as f:
            generate_payload = json.load(f)
            selected_indices = generate_payload["selected"]
            client_selected_indices_from_payload = generate_payload.get("client_selected", selected_indices)
            server_input_qc = generate_payload.get("input_qc")
        os.remove(FLAG_START_GENERATE)

        data_dir, preview_dir = get_current_session_dirs()
        images_with_names = load_presegmented_previews(data_dir, preview_dir)
        
        if not selected_indices:
            print("[Worker] 手机端取消了生成或未选择图像，重置状态。")
            continue

        client_selected_indices = client_selected_indices_from_payload
        image_names = [name for _img, name in images_with_names]
        selected_indices, frame_filter = arq.select_frames_for_reconstruction(
            selected_indices,
            image_names,
            data_dir,
            preview_dir,
        )
        if server_input_qc is not None:
            frame_filter["server_input_qc"] = server_input_qc
        print(f"[Worker] 收到手机端指令: {client_selected_indices}")
        print(f"[Worker] 筛选后用于生成的图像序号: {selected_indices}")

        if arq.enforce_input_qc_from_env() and not frame_filter.get("qc_pass", True):
            message = "Input frame QC failed: " + "; ".join(frame_filter.get("fail_reasons") or ["unknown reason"])
            print(f"[Worker] {message}")
            with open(FLAG_GENERATE_DONE, 'w') as f:
                json.dump(
                    {
                        "status": "error",
                        "message": message,
                        "client_selected_indices": client_selected_indices,
                        "selected_indices": selected_indices,
                        "frame_filter": frame_filter,
                    },
                    f,
                    indent=4,
                )
            wait_for_server_postprocess_ack()
            continue
        
        # 提取选择的图像和文件名
        selected_images = [images_with_names[i][0] for i in selected_indices if i < len(images_with_names)]
        selected_names = [images_with_names[i][1] for i in selected_indices if i < len(images_with_names)]
        if not selected_images:
            print("[Worker] 筛选后没有有效图像，重置状态。")
            with open(FLAG_GENERATE_DONE, 'w') as f:
                json.dump(
                    {
                        "status": "error",
                        "message": "No valid images after frame filtering.",
                        "client_selected_indices": client_selected_indices,
                        "selected_indices": selected_indices,
                        "frame_filter": frame_filter,
                    },
                    f,
                    indent=4,
                )
            wait_for_server_postprocess_ack()
            continue
        
        # ================= 核心新增：读取并对齐位姿数据 =================
        pose_dict = read_pose_dict(data_dir)
        
        extrinsics_list = []
        intrinsics_list = []
        
        if pose_dict:
            # 批量获取归一化后的内外参
            extrinsics_list, intrinsics_list = get_normalized_camera_matrices(pose_dict, selected_names)
            
        # 转换为 PyTorch Tensors，形状为 (B, 4, 4) 和 (B, 3, 3)
        device = pipeline.device
        extrinsics_tensor = torch.tensor(np.array(extrinsics_list), dtype=torch.float32).to(device) if extrinsics_list else None
        intrinsics_tensor = torch.tensor(np.array(intrinsics_list), dtype=torch.float32).to(device) if intrinsics_list else None
        # ================================================================

        current_output_dir = None
        candidate_report = []
        best_candidate_metrics = None
        generation_error = None
        try:
            current_output_dir = make_unique_output_dir(OUTPUT_DIR)
            base_filename = os.path.join(current_output_dir, "reconstructed_object")
            candidate_seeds = get_candidate_seeds()
            mesh_simplify = get_mesh_simplify_ratio()
            print(f"[Worker] 本次结果将保存在新建目录: {current_output_dir}")
            print(f"[Worker] 本次最多生成 {len(candidate_seeds)} 个候选: {candidate_seeds}")
            print(f"[Worker] Mesh 导出 simplify={mesh_simplify:.2f}")
            input_manifest = save_generation_inputs(
                current_output_dir,
                selected_images,
                selected_names,
                selected_indices,
                client_selected_indices,
                frame_filter,
                candidate_seeds,
            )

            print("[Worker] 第一阶段：执行少量候选 3D 骨架预测并自动选择...")
            best_candidate, candidate_report = run_limited_candidate_generation(
                pipeline,
                selected_images,
                selected_names,
                data_dir,
                candidate_seeds,
                pose_rerank_enabled=arq.pose_rerank_enabled_from_env(),
                pose_rerank_weight=arq.pose_rerank_weight_from_env(),
            )
            outputs = best_candidate["outputs"]
            coords = best_candidate["coords"]
            ss_noise = best_candidate["ss_noise"]
            best_candidate_metrics = best_candidate["metrics"]
            with open(os.path.join(current_output_dir, "candidate_report.json"), "w") as f:
                json.dump(
                    {
                        "selected_seed": int(best_candidate["seed"]),
                        "selected_candidate": best_candidate_metrics,
                        "candidates": candidate_report,
                        "mesh_simplify": mesh_simplify,
                        "input_manifest": input_manifest,
                        "pose_rerank_enabled": arq.pose_rerank_enabled_from_env(),
                        "pose_rerank_weight": arq.pose_rerank_weight_from_env(),
                    },
                    f,
                    indent=4,
                )
            print(
                f"[Worker] 选择 seed={best_candidate['seed']} "
                f"base_score={best_candidate_metrics['score']:.4f} "
                f"selection_score={best_candidate_metrics.get('selection_score', best_candidate_metrics['score']):.4f}"
            )
            
            # torch.cuda.empty_cache() # 释放部分显存备战 Refine
            
            # print("[Worker] 第二阶段：基于相机位姿进行几何与材质 Refine 优化 (耗时较长)...")
            # # 第二阶段：带入相机位姿执行 run_refine
            # # 注意：因为我们传了 coords，给 input_points 传个占位的 Dummy 即可
            # B = len(selected_images)
            # dummy_input_points = torch.zeros((B, 3), dtype=torch.long, device=device)
            
            # outputs = pipeline.run_refine(
            #     image=selected_images,
            #     ss_learning_rate=1e-3,          # 几何优化学习率
            #     ss_start_t=0.6,
            #     apperance_learning_rate=1e-3,   # 外观材质优化学习率
            #     apperance_start_t=0.6,
            #     extrinsics=extrinsics_tensor,   # 传入外参
            #     intrinsics=intrinsics_tensor,   # 传入内参
            #     ss_noise=ss_noise,              # 传入第一步的初始噪声
            #     input_points=dummy_input_points,
            #     ss_refine_type='deltav',        # 可选 'noise' 或 'deltav'
            #     coords=coords,                  # 传入初始几何体坐标
            #     formats=["gaussian", "mesh"],
            #     mode="multidiffusion"
            # )
            
            gs, mesh = outputs['gaussian'][0], outputs['mesh'][0]
            torch.cuda.empty_cache() 

            print("[Worker] 正在导出并保存 3D 高斯点云 (Gaussian Splatting .ply)...")
            gs.save_ply(f"{base_filename}.ply")

            print(f"[Worker] 正在导出并保存 3D 网格模型 (Mesh .glb)，simplify={mesh_simplify:.2f}...")
            glb = postprocessing_utils.to_glb(gs, mesh, simplify=mesh_simplify, texture_size=1024, verbose=False)
            glb.export(f"{base_filename}.glb")
            del glb

            print("[Worker] 正在渲染 360 度预览视频 (.mp4)，请稍候...")
            video_color = render_utils.render_video(gs, num_frames=120)['color']
            video_geo = render_utils.render_video(mesh, num_frames=120)['normal']
            video = [np.concatenate([video_color[i], video_geo[i]], axis=1) for i in range(len(video_color))]
            imageio.mimsave(f"{base_filename}.mp4", video, fps=15)
            del video_color, video_geo, video
            del outputs, coords, ss_noise, best_candidate
            torch.cuda.empty_cache()
            
            print(f"[Worker] ✅ 所有任务生成完毕！Mesh、点云和视频均已保存在 {current_output_dir}")
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            generation_error = str(e)
            print(f"[Worker] ❌ 生成失败: {e}")
            
        with open(FLAG_GENERATE_DONE, 'w') as f:
            json.dump(
                {
                    "status": "error" if generation_error else "done",
                    "message": generation_error,
                    "client_selected_indices": client_selected_indices,
                    "selected_indices": selected_indices,
                    "selected_names": selected_names,
                    "frame_filter": frame_filter,
                    "output_dir": current_output_dir,
                    "candidate_report": candidate_report,
                    "selected_candidate": best_candidate_metrics,
                },
                f,
                indent=4,
            )
        print("[Worker] 已通知 server.py 进入 CoarseModel 模板/特征后处理，等待后处理完成...")
        wait_for_server_postprocess_ack()

if __name__ == "__main__":
    main()
