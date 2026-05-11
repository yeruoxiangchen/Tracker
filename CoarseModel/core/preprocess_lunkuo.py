# core/preprocess.py
import os
import cv2
import numpy as np
import logging
import pickle

from .config import AppConfig

from core.corresp_lunkuo import extract_correspondences
from core.util import (
    quat_to_rotmat, read_colmap_cameras_txt, 
    read_colmap_images_txt, colmap_camera_to_K
)

import torch
from utils import  repre_util

logger = logging.getLogger(__name__)

def save_optimization_cache(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f)


def load_optimization_cache(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["optimization_data"], data["corresps_data"]


def visualize_template_candidates(
    active_mask_files, 
    mask_dir, 
    per_frame_candidates, 
    template_base_dir, 
    output_root
):
    """
    可视化每一帧的候选模板
    """
    save_dir = os.path.join(output_root, "candtemp")
    os.makedirs(save_dir, exist_ok=True)

    for i, (m_file, candidates) in enumerate(zip(active_mask_files, per_frame_candidates)):
        # 1. 加载 Query Mask
        mask_path = os.path.join(mask_dir, m_file)
        query_mask = cv2.imread(mask_path)
        if query_mask is None: continue

        # 2. 加载所有候选模板图片并存储
        candidate_imgs = []
        subset_cands = candidates[:5] # 限制前5个防止图片过长
        
        for cand_id in subset_cands:
            # 根据你的实际格式修改文件名，如 f"{cand_id}.png"
            tpl_filename = f"template_{cand_id:04d}.png" 
            tpl_path = os.path.join(template_base_dir, "mask", tpl_filename)
            
            img = cv2.imread(tpl_path)
            if img is not None:
                # 在模板图上画上 ID 方便辨认
                cv2.putText(img, f"ID:{cand_id}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                candidate_imgs.append(img)
            else:
                # 找不到图则跳过或放入空阵列
                candidate_imgs.append(None)
        # 3. 计算对齐所需的最大高度和总宽度
        all_imgs = [query_mask] + [img for img in candidate_imgs if img is not None]
        max_h = max(img.shape[0] for img in all_imgs)
        max_w = max(img.shape[1] for img in all_imgs)
        
        # 4. 创建黑色背景画布 (高度为 max_h, 宽度为 max_w * 图片数量)
        num_imgs = len(all_imgs)
        canvas = np.zeros((max_h, max_w * num_imgs, 3), dtype=np.uint8)

        # 5. 将图片逐个填入画布（不缩放，仅 Padding）
        for idx, img in enumerate(all_imgs):
            h, w = img.shape[:2]
            # 计算起始坐标 (这里靠左上对齐，如果想居中可以修改 y_offset 和 x_offset)
            y_offset = 0 
            x_offset = idx * max_w
            
            canvas[y_offset:y_offset+h, x_offset:x_offset+w] = img
            
            # 加上分隔线
            if idx > 0:
                cv2.line(canvas, (x_offset, 0), (x_offset, max_h), (50, 50, 50), 2)

        # 标注 Query 帧
        cv2.putText(canvas, f"Frame {i} (Query)", (10, max_h - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 6. 保存结果
        save_path = os.path.join(save_dir, f"frame_{i:04d}_padded.jpg")
        cv2.imwrite(save_path, canvas)
        # print(f"Saved visualization to: {save_path}")
        
def visualize_best_sequence(
    active_mask_files, 
    mask_dir, 
    best_path, 
    template_base_dir, 
    output_root
):
    """
    可视化连续性筛选后的最优模板序列
    """
    save_dir = os.path.join(output_root, "candtemp", "best_path")
    os.makedirs(save_dir, exist_ok=True)

    # 模板渲染图通常在 template_base_dir/visual 或直接在 template_base_dir
    # 这里假设渲染图路径与 mask 类似，但在不同的子目录，或者直接根据 ID 查找
    render_dir = os.path.join(template_base_dir, "visual") 
    if not os.path.exists(render_dir):
        render_dir = os.path.join(template_base_dir, "mask") # 回退到 mask 目录

    for i, (m_file, t_id) in enumerate(zip(active_mask_files, best_path)):
        # 1. 加载 Query
        query_img = cv2.imread(os.path.join(mask_dir, m_file))
        
        # 2. 加载选中的最优模板
        # 注意：这里文件名格式需与你磁盘实际存储一致
        tpl_filename = f"template_{t_id:04d}.png" 
        tpl_path = os.path.join(render_dir, tpl_filename)
        
        tpl_img = cv2.imread(tpl_path)
        if tpl_img is None:
            # 尝试另一种命名格式
            tpl_img = cv2.imread(os.path.join(render_dir, f"{t_id:05d}.png"))

        # 3. 拼接处理 (使用之前讨论的 Padding 逻辑以保持原样)
        imgs_to_combine = [query_img]
        if tpl_img is not None:
            imgs_to_combine.append(tpl_img)
        else:
            # 占位图
            imgs_to_combine.append(np.zeros_like(query_img))

        max_h = max(img.shape[0] for img in imgs_to_combine)
        max_w = max(img.shape[1] for img in imgs_to_combine)
        
        canvas = np.zeros((max_h, max_w * 2, 3), dtype=np.uint8)
        
        for idx, img in enumerate(imgs_to_combine):
            h, w = img.shape[:2]
            canvas[:h, idx*max_w : idx*max_w+w] = img
            
        # 绘制文字标注
        cv2.putText(canvas, f"Frame {i} Query", (10, 30), 1, 2, (0, 255, 0), 2)
        cv2.putText(canvas, f"Best Template: {t_id}", (max_w + 10, 30), 1, 2, (0, 0, 255), 2)
        
        # 4. 保存
        save_path = os.path.join(save_dir, f"seq_{i:02d}_final.jpg")
        cv2.imwrite(save_path, canvas)

    print(f"Final sequence visualization saved to: {save_dir}")
    
def get_aligned_roi(mask, target_size=256):
    """
    提取 Mask 中的目标区域，保持比例缩放并放置在固定大小的画布中心。
    """
    # 找到非零像素（物体区域）
    coords = cv2.findNonZero(mask)
    if coords is None:
        return np.zeros((target_size, target_size), dtype=np.uint8)
    
    # 获取最小外接正矩形 (Bounding Box)
    x, y, w, h = cv2.boundingRect(coords)
    roi = mask[y:y+h, x:x+w]
    
    # 计算缩放比例，留出小量边距 (10px) 避免贴边影响轮廓提取
    scale = (target_size - 10) / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    
    # 缩放 ROI
    roi_resized = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    
    # 创建黑色画布并居中放置
    canvas = np.zeros((target_size, target_size), dtype=np.uint8)
    start_x = (target_size - new_w) // 2
    start_y = (target_size - new_h) // 2
    canvas[start_y:start_y+new_h, start_x:start_x+new_w] = roi_resized
    
    return canvas

import numpy as np

def resample_contour(cnt, n=128):
    """
    Args:
        cnt: OpenCV contour, shape (N, 1, 2) or (N, 2)
        n: number of resampled points

    Returns:
        resampled contour of shape (n, 2)
    """
    cnt = np.asarray(cnt, dtype=np.float32)
    if cnt.ndim == 3:
        cnt = cnt[:, 0, :]  # (N, 2)

    # 闭合轮廓
    if not np.allclose(cnt[0], cnt[-1]):
        cnt = np.vstack([cnt, cnt[0]])

    # 计算弧长
    seg_lens = np.linalg.norm(cnt[1:] - cnt[:-1], axis=1)
    arc_len = np.concatenate([[0], np.cumsum(seg_lens)])
    total_len = arc_len[-1]

    if total_len < 1e-6:
        return np.repeat(cnt[:1], n, axis=0)

    # 均匀采样的弧长位置
    sample_lens = np.linspace(0, total_len, n + 1)[:-1]

    resampled = []
    seg_idx = 0
    for s in sample_lens:
        while arc_len[seg_idx + 1] < s:
            seg_idx += 1
        t = (s - arc_len[seg_idx]) / (arc_len[seg_idx + 1] - arc_len[seg_idx] + 1e-8)
        p = (1 - t) * cnt[seg_idx] + t * cnt[seg_idx + 1]
        resampled.append(p)

    return np.array(resampled, dtype=np.float32)

def contour_signed_curvature(cnt, n=128):
    cnt = resample_contour(cnt, n)  # (n, 2)
    curv = np.zeros(n, dtype=np.int8)

    for i in range(n):
        p_prev = cnt[i - 1]
        p = cnt[i]
        p_next = cnt[(i + 1) % n]

        v1 = p - p_prev
        v2 = p_next - p

        # z 分量（2D 叉乘）
        cross_z = v1[0] * v2[1] - v1[1] * v2[0]

        if cross_z > 1e-3:
            curv[i] = 1
        elif cross_z < -1e-3:
            curv[i] = -1
        else:
            curv[i] = 0

    return curv

def contour_pose_descriptor(cnt):
    cnt = np.asarray(cnt)
    
    # 主轴方向（PCA）
    pts = cnt.reshape(-1, 2).astype(np.float32)
    mean = pts.mean(axis=0)
    pts_c = pts - mean
    cov = np.cov(pts_c.T)
    eigvals, eigvecs = np.linalg.eig(cov)
    main_axis = eigvecs[:, np.argmax(eigvals)]
    angle = np.arctan2(main_axis[1], main_axis[0])  # radians

    # bounding box
    x, y, w, h = cv2.boundingRect(cnt)
    aspect = w / (h + 1e-6)

    # centroid offset (normalized)
    M = cv2.moments(cnt)
    if M["m00"] > 1e-6:
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        cx_n = (cx - x) / w
        cy_n = (cy - y) / h
    else:
        cx_n, cy_n = 0.5, 0.5

    return {
        "angle": angle,
        "aspect": aspect,
        "centroid": np.array([cx_n, cy_n])
    }
def find_candidate_templates(input_mask, 
                             precomputed_templates, 
                             template_ids, 
                             num_candidates=10, 
                             curvature_weight=0.5):
    """
    级联筛选策略：
    1. IoU 粗筛位姿 (Pose Filter)
    2. Shape Match 精选轮廓 (Contour Refinement)
    """
    pre_candidates = []
    target_res = 256
    
    # 1. 预处理输入
    aligned_input = get_aligned_roi(input_mask, target_size=target_res)
    cnt_in, _ = cv2.findContours(aligned_input, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnt_in: return []
    cnt_in = max(cnt_in, key=cv2.contourArea)

    # 预计算输入轮廓曲率
    curv_in = contour_signed_curvature(cnt_in)
    
    # 第一阶段：IoU 粗筛
    for t_id, t_aligned in enumerate(precomputed_templates):
        # 矩阵交并比
        intersection = np.logical_and(aligned_input, t_aligned).sum()
        union = np.logical_or(aligned_input, t_aligned).sum()
        iou = intersection / (union + 1e-6)
        
        pre_candidates.append({"id": t_id, "iou": iou})
    
    # 取前 30 个位姿最接近的
    pre_candidates.sort(key=lambda x: x["iou"], reverse=True)
    top_pose_candidates = pre_candidates[:30]

    # 3. ShapeMatchScore 精选
    final_candidates = []
    for cand in top_pose_candidates:
        t_id = cand["id"]
        t_aligned = precomputed_templates[t_id] # 直接获取预处理好的图
        
        cnt_tpl, _ = cv2.findContours(t_aligned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnt_tpl: continue
        cnt_tpl = max(cnt_tpl, key=cv2.contourArea)
        
        shape_score = cv2.matchShapes(
            cnt_in, cnt_tpl, cv2.CONTOURS_MATCH_I1, 0.0
        )
        curv_tpl = contour_signed_curvature(cnt_tpl)

        valid = (curv_in != 0) & (curv_tpl != 0)
        if valid.sum() > 0:
            curvature_mismatch = np.mean(curv_in[valid] != curv_tpl[valid])
        else:
            curvature_mismatch = 0.5  # 保守惩罚
        final_score = shape_score + curvature_weight * curvature_mismatch
        final_candidates.append((template_ids[t_id], final_score))

    # 最终排序：按轮廓相似度升序（越小越像）
    final_candidates.sort(key=lambda x: x[1])
    
    return [c[0] for c in final_candidates[:num_candidates]]
    
def build_optimization_data(
    rgb_dir: str,
    mask_dir: str,
    colmap_dir: str,
    opts,
):
    """
    Returns:
        all_optimization_data: list[dict]
        all_corresps_data: list[CorrespondenceData]
    """
    cameras_colmap = read_colmap_cameras_txt(
        os.path.join(colmap_dir, "cameras.txt")
    )
    
    images_txt = os.path.join(colmap_dir, "images.txt")
    images_colmap = read_colmap_images_txt(images_txt)

    rgb_files = sorted(f for f in os.listdir(rgb_dir) if f.endswith((".jpg", ".png")))
    mask_files = sorted(f for f in os.listdir(mask_dir) if f.endswith(".png"))

    object_lids = opts.object_lids 
    
    object_lid = object_lids[0]
    templates_path = os.path.join(
                "templates",
                opts.version,
                opts.object_dataset,
                str(object_lid),
            )
    template_base_dir = os.path.join(
        AppConfig.OUTPUT_ROOT,
        templates_path,
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    repre_dir = os.path.join(
            AppConfig.OUTPUT_ROOT, "object_repre", 
            opts.object_dataset, opts.version, str(object_lid))
    repre = repre_util.load_object_repre(
        repre_dir=repre_dir, tensor_device=device)
            
    # 仅取前 5 帧
    active_rgb_files = rgb_files[:5]
    active_mask_files = mask_files[:5]
    
    # --- 离线预处理模板逻辑 ---
    target_res = 256
    cache_dir = os.path.join(AppConfig.REFINE_ROOT, opts.object_dataset)
    os.makedirs(cache_dir, exist_ok=True)
    
    # 缓存文件名包含版本或 object_lid 以防混淆
    cache_path = os.path.join(cache_dir, f"aligned_templates_{object_lid}.npy")
    # 确定模板存放目录
    mask_template_dir = os.path.join(template_base_dir, "mask")
    # 获取目录下所有 png 模板并排序（确保索引固定）
    all_mask_files = sorted([f for f in os.listdir(mask_template_dir) if f.endswith(".png")])
    if os.path.exists(cache_path):
        print(f"Loading pre-aligned templates from {cache_path}")
        # 加载后的形状为 (798, 256, 256)
        precomputed_templates = np.load(cache_path)
    else:
        print("Pre-processing templates (this may take a while for the first time)...")
        temp_list = []
        for f_name in all_mask_files:
            t_path = os.path.join(mask_template_dir, f_name)
            t_mask = cv2.imread(t_path, cv2.IMREAD_GRAYSCALE)
            if t_mask is not None:
                aligned = get_aligned_roi(t_mask, target_size=target_res)
            else:
                aligned = np.zeros((target_res, target_res), dtype=np.uint8)
            temp_list.append(aligned)
        
        precomputed_templates = np.array(temp_list)
        np.save(cache_path, precomputed_templates)
        print(f"Pre-aligned templates saved to {cache_path}")
    
    # 建立 ID 映射（如果文件名是 template_0001.png，我们可以提取数字作为 ID）
    # 如果文件名不规律，则直接使用 all_mask_files 的索引
    import re
    template_ids = []
    for f in all_mask_files:
        nums = re.findall(r'\d+', f)
        template_ids.append(int(nums[0]) if nums else f)
        
    # 第一步：为每一帧寻找轮廓候选集
    per_frame_candidates = []
    for m_file in active_mask_files:
        mask = cv2.imread(os.path.join(mask_dir, m_file), cv2.IMREAD_GRAYSCALE)
        cands = find_candidate_templates(mask, precomputed_templates, template_ids, num_candidates=10)
        per_frame_candidates.append(cands)
        
    visualize_template_candidates(
        active_mask_files, 
        mask_dir, 
        per_frame_candidates, 
        template_base_dir, 
        os.path.join(AppConfig.REFINE_ROOT, opts.object_dataset)
    )
    # 第二步：利用连续性筛选最优路径
    # 策略：计算模板位姿之间的距离，寻找平滑变换路径
    best_template_sequence = []
    current_best_path = [per_frame_candidates[0][0]] # 简化版：从第一帧最相似开始
    
    for i in range(1, 5):
        prev_tpl_id = current_best_path[-1]
        candidates = per_frame_candidates[i]
        
        # 计算当前候选模板与前一帧模板的位姿差异 (Rotation distance)
        # R_prev = repre.template_cameras_cam_from_model[prev_tpl_id].R
        cam_prev = repre.template_cameras_cam_from_model[prev_tpl_id]
        R_prev = cam_prev.T_world_from_eye[:3, :3]
        t_prev = cam_prev.T_world_from_eye[:3, 3]
        best_next_tpl = candidates[0]
        min_rot_diff = float('inf')
        
        for cand_id in candidates:
    
            cam_curr = repre.template_cameras_cam_from_model[cand_id]
            R_curr = cam_curr.T_world_from_eye[:3, :3]
            t_curr = cam_curr.T_world_from_eye[:3, 3]
            # 旋转矩阵点积计算角度差
            rot_diff = np.arccos(np.clip((np.trace(R_prev.T @ R_curr) - 1) / 2.0, -1, 1))
            if rot_diff < min_rot_diff:
                min_rot_diff = rot_diff
                best_next_tpl = cand_id
        current_best_path.append(best_next_tpl)
    
    visualize_best_sequence(
        active_mask_files=active_mask_files,
        mask_dir=mask_dir,
        best_path=current_best_path,
        template_base_dir=template_base_dir,
        output_root=os.path.join(AppConfig.REFINE_ROOT, opts.object_dataset)
    )
    
    # 第三步：基于选定的模板序列建立对应关系
    all_corresps_data = []
    all_optimization_data = [] # 初始化优化数据列表
    
    # 获取 COLMAP 数据
    cameras_colmap = read_colmap_cameras_txt(os.path.join(colmap_dir, "cameras.txt"))
    images_colmap = read_colmap_images_txt(images_txt)
    
    # 遍历所有文件（不再仅限于 active_rgb_files）
    for i, rgb_f in enumerate(rgb_files):
        frame_name = rgb_f
        if frame_name not in images_colmap:
            logger.warning(f"{frame_name} not in COLMAP, skip")
            continue

        # 基础数据准备
        image_path = os.path.join(rgb_dir, rgb_f)
        # 假设 mask 文件名和 rgb 一一对应
        mask_f = mask_files[i] if i < len(mask_files) else None
        mask_path = os.path.join(mask_dir, mask_f) if mask_f else None
        
        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) if mask_path else None
        if mask is None:
            mask = np.zeros(image.shape[:2], dtype=np.uint8)

        # 获取相机内参和外参 (T_w2c)
        img_entry = images_colmap[frame_name]
        cam = cameras_colmap[img_entry["cam_id"]]
        K = colmap_camera_to_K(cam)
        
        R_w2c = img_entry["R"]
        t_w2c = img_entry["t"]
        T_w2c = np.eye(4, dtype=np.float32)
        T_w2c[:3, :3] = R_w2c
        T_w2c[:3, 3] = t_w2c

        # --- 核心逻辑：前 5 帧分支 ---
        corresp = None
        q_2d, X_3d, X_ids = None, None, None
        
        if i < 5:
            # 获取当前帧匹配到的最优模板 ID
            current_tpl_id = current_best_path[i]
            
            corresp = extract_correspondences(
                image_raw=image,
                mask_raw=mask,
                camK=K,
                frame_name=frame_name,
                opts=opts,
                template_id=392,#current_tpl_id, # 使用序列中对应的 ID
                repre=repre,
            )
            
            if corresp is not None:
                q_2d = corresp.query_2d_pts.astype(np.float32)
                X_3d = corresp.model_3d_pts.astype(np.float32)
                # X_ids = corresp.model_feat_ids
                logger.info(f"Frame {i}: Extracted correspondences using Template {current_tpl_id}")
        else:
            # 5 帧之后，保持 corresp 和相应点为 None
            pass

        # 填充结果
        all_corresps_data.append(corresp)
        all_optimization_data.append(
            dict(
                frame_name=frame_name,
                K=K.astype(np.float32),
                T_w2c=T_w2c,
                q_2d=q_2d,
                X_3d=X_3d,
                X_ids=X_ids,
                image_path=image_path,
            )
        )

    return all_optimization_data, all_corresps_data
