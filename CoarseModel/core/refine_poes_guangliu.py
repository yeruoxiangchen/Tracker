import os
import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares
from script.vis_util import (
    visualize_projection,
)
import logging

logger = logging.getLogger(__name__)

def visualize_anchor_tracking_pair(
    prev_img,
    curr_img,
    prev_pts,
    curr_pts,
    frame_idx,
    anchor_id,
    output_dir,
    max_vis=100
):
    """
    prev_img, curr_img: BGR
    prev_pts, curr_pts: Nx2
    """
    h = max(prev_img.shape[0], curr_img.shape[0])
    w = prev_img.shape[1] + curr_img.shape[1]

    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[:prev_img.shape[0], :prev_img.shape[1]] = prev_img
    canvas[:curr_img.shape[0], prev_img.shape[1]:] = curr_img

    n = min(len(prev_pts), max_vis)

    for i in range(n):
        p0 = prev_pts[i].astype(int)
        p1 = curr_pts[i].astype(int)
        p1_shift = p1.copy()
        p1_shift[0] += prev_img.shape[1]

        color = (
            int(50 + (i * 37) % 205),
            int(50 + (i * 17) % 205),
            int(50 + (i * 97) % 205),
        )

        cv2.circle(canvas, tuple(p0), 3, color, -1)
        cv2.circle(canvas, tuple(p1_shift), 3, color, -1)
        cv2.line(canvas, tuple(p0), tuple(p1_shift), color, 1)

        cv2.putText(
            canvas, str(i),
            tuple(p0 + np.array([3, -3])),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4, color, 1
        )
        cv2.putText(
            canvas, str(i),
            tuple(p1_shift + np.array([3, -3])),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4, color, 1
        )

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(
        output_dir, f"frame_{frame_idx:03d}_anchor_{anchor_id:02d}.jpg"
    )
    cv2.imwrite(out_path, canvas)


class AnchorGroup:
    def __init__(self, X_3d, q2d, feats, frame_id):
        self.X_3d = X_3d          # (N,3)
        self.q2d = q2d            # (N,2) 上一帧2D
        self.feats = feats        # (N,C)
        self.frame_id = frame_id

def build_anchor_group(
    frame,
    T_M2W,
    reproj_thresh,
    s_fixed
):
    X_3d = frame["X_3d"]
    q_2d = frame["q_2d"]
    K = frame["K"]
    T_w2c = frame["T_w2c"]

    err = compute_reproj_err(
        T_M2W, X_3d, q_2d, K, T_w2c, s_fixed
    )
    mask = err < reproj_thresh

    if mask.sum() < 10:
        return None

    feat_map = frame["feature_map_chw_proj"]
    meta = frame["feat_map_meta"]

    feats = sample_features_at_q2d_with_meta(
        feat_map,
        q_2d[mask],
        meta
    )
    feats /= np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8

    return AnchorGroup(
        X_3d=X_3d[mask],
        q2d=q_2d[mask],
        feats=feats,
        frame_id=frame["frame_id"]
    )

def track_anchor_group(
    anchor: AnchorGroup,
    prev_img_gray,
    curr_img_gray,
    curr_frame,
    T_M2W,
    s_fixed
):
    # 1. 几何预测
    pred_q2d = project_model_to_image(
        anchor.X_3d,
        T_M2W,
        curr_frame["K"],
        curr_frame["T_w2c"],
        s_fixed
    )

    # 2. 光流
    tracked_q2d, track_mask = optical_flow_tracking(
        prev_img_gray,
        curr_img_gray,
        anchor.q2d,
        pred_q2d=pred_q2d,
        max_dist=20.0
    )

    # 3. mask 过滤
    track_mask = filter_tracked_points_by_mask(
        tracked_q2d,
        track_mask,
        curr_frame["image_path"]
    )

    # 4. 更新 anchor 内部状态
    anchor.q2d = tracked_q2d
    anchor.X_3d = anchor.X_3d[track_mask]
    anchor.feats = anchor.feats[track_mask]

    return track_mask.sum()

def collect_ba_observations(anchor_pool, curr_frame):
    X_all = []
    q_all = []
    idx_offset = 0
    obs = []

    for anchor in anchor_pool:
        n = len(anchor.X_3d)
        if n < 10:
            continue

        X_all.append(anchor.X_3d)
        q_all.append(anchor.q2d)

        obs.append({
            "anchor_indices": np.arange(idx_offset, idx_offset + n),
            "q_2d": anchor.q2d,
            "K": curr_frame["K"],
            "T_w2c": curr_frame["T_w2c"],
            "image_path": curr_frame["image_path"]
        })

        idx_offset += n

    if len(X_all) == 0:
        return None, None, None

    return np.vstack(X_all), obs, idx_offset


def filter_tracked_points_by_mask(
    tracked_q2d,
    track_mask,
    image_path,
    mask_threshold=128
):
    """
    如果 tracked_q2d 落在 mask 的背景区域，则将 track_mask 置 False
    mask_threshold: mask > threshold 视为前景
    """
    if tracked_q2d.shape[0] == 0:
        return track_mask

    # 构造 mask 路径
    mask_path = image_path.replace("/rgb/", "/masks/")
    mask_path = os.path.splitext(mask_path)[0] + ".png"

    if not os.path.exists(mask_path):
        logger.warning(f"Mask not found: {mask_path}, skip mask filtering.")
        return track_mask

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        logger.warning(f"Failed to read mask: {mask_path}")
        return track_mask

    H, W = mask.shape

    for i in range(len(tracked_q2d)):
        if not track_mask[i]:
            continue

        x, y = tracked_q2d[i]
        ix, iy = int(round(x)), int(round(y))

        # 出界 → 判为无效
        if ix < 0 or ix >= W or iy < 0 or iy >= H:
            track_mask[i] = False
            continue

        # 背景 → 删除
        if mask[iy, ix] < mask_threshold:
            track_mask[i] = False

    return track_mask

def optical_flow_tracking(
    prev_img_gray,
    curr_img_gray,
    prev_q2d,
    pred_q2d=None,
    max_dist=15.0,
    lk_win_size=21,
    lk_max_level=3
):
    """
    使用 LK 光流跟踪 2D 点
    Args:
        prev_img_gray: (H,W) uint8
        curr_img_gray: (H,W) uint8
        prev_q2d: (N,2) float32，上一帧 2D
        pred_q2d: (N,2) float32，当前帧几何预测（可选，用于 gating）
        max_dist: float，几何 gating 阈值（像素）
    Returns:
        curr_q2d: (N,2)
        track_mask: (N,) bool
    """

    if len(prev_q2d) == 0:
        return np.zeros((0, 2), np.float32), np.zeros(0, bool)

    prev_pts = prev_q2d.reshape(-1, 1, 2).astype(np.float32)

    lk_params = dict(
        winSize=(lk_win_size, lk_win_size),
        maxLevel=lk_max_level,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )

    curr_pts, status, err = cv2.calcOpticalFlowPyrLK(
        prev_img_gray,
        curr_img_gray,
        prev_pts,
        None,
        **lk_params
    )

    curr_pts = curr_pts.reshape(-1, 2)
    status = status.reshape(-1).astype(bool)

    track_mask = status.copy()

    # --- 几何 gating（非常重要）---
    if pred_q2d is not None:
        d = np.linalg.norm(curr_pts - pred_q2d, axis=1)
        track_mask &= (d < max_dist)

    # 输出时对失败点填 0（防止 NaN）
    curr_pts[~track_mask] = 0.0

    return curr_pts, track_mask


def refine_global_pose_ba(
    all_optimization_data, 
    s_fixed, 
    T_M2W_init, 
    model_vertices, # 用于最终可视化投影
    model_faces,
    max_frames=100, 
    reproj_thresh=5.0,
    search_radius=20,
    dist_thresh=0.5,
    output_dir="/home/zjr/Tracker/CoarseModel/results/refine_model/broccoli/"
):
    """
    使用多视图锚定跟踪（Anchored Tracking）和 BA 精修位姿
    """
    os.makedirs(output_dir, exist_ok=True)
    # --- 1. 初始化 Anchor (第一帧) ---
    first_frame = all_optimization_data[0]
    first_image_path = first_frame["image_path"]
    prev_img_gray = cv2.imread(first_image_path, cv2.IMREAD_GRAYSCALE)
    
    anchor_pool = []

    first_frame = all_optimization_data[0]
    first_frame["frame_id"] = 0
    prev_img_gray = cv2.imread(first_frame["image_path"], cv2.IMREAD_GRAYSCALE)

    anchor0 = build_anchor_group(
        first_frame,
        T_M2W_init,
        reproj_thresh,
        s_fixed
    )

    anchor_pool.append(anchor0)
    T_M2W_curr = T_M2W_init.copy()
    comp_dir = os.path.join(
        output_dir, "comp"
    )
    os.makedirs(comp_dir, exist_ok=True)
    # --- 2. 锚定跟踪 (Frame 0 -> Frame i) ---
    for i in range(1, min(len(all_optimization_data), max_frames)):
        frame = all_optimization_data[i]
        frame["frame_id"] = i
        curr_img_gray = cv2.imread(frame["image_path"], cv2.IMREAD_GRAYSCALE)

        total_tracked = 0
        prev_img_color = cv2.imread(
            all_optimization_data[i - 1]["image_path"]
        )
        curr_img_color = cv2.imread(
            frame["image_path"]
        )
        # ---- 跟踪所有 AnchorGroup ----
        anchor_id = 0
        for anchor in anchor_pool:
            if len(anchor.X_3d) < 20:
                anchor_id += 1
                continue

            pred_q2d = project_model_to_image(
                anchor.X_3d,
                T_M2W_curr,
                frame["K"],
                frame["T_w2c"],
                s_fixed
            )

            tracked_q2d, mask = optical_flow_tracking(
                prev_img_gray,
                curr_img_gray,
                anchor.q2d,
                pred_q2d,
                max_dist=20.0
            )

            mask = filter_tracked_points_by_mask(
                tracked_q2d,
                mask,
                frame["image_path"]
            )
            # ---------- ⭐ 可视化 ----------
            if mask.sum() > 0:
                visualize_anchor_tracking_pair(
                    prev_img=prev_img_color,
                    curr_img=curr_img_color,
                    prev_pts=anchor.q2d[mask],
                    curr_pts=tracked_q2d[mask],
                    frame_idx=i,
                    anchor_id=anchor_id,
                    output_dir=comp_dir,
                    max_vis=80
                )
            anchor.q2d = tracked_q2d[mask]
            anchor.X_3d = anchor.X_3d[mask]
            anchor.feats = anchor.feats[mask]

            total_tracked += mask.sum()

        # logger.info(f"Frame {i}: total tracked = {total_tracked}")

        # ---- 关键帧触发（核心）----
        if total_tracked < 30 or i % 10 == 0:
            new_anchor = build_anchor_group(
                frame,
                T_M2W_curr,
                reproj_thresh,
                s_fixed
            )
            if new_anchor is not None:
                anchor_pool.append(new_anchor)
                logger.info(
                    f"Add keyframe @ {i}, points={len(new_anchor.X_3d)}"
                )

        # ---- 收集 BA 观测 ----
        X_all = []
        ba_obs = []
        offset = 0

        for anchor in anchor_pool:
            n = len(anchor.X_3d)
            if n < 20:
                continue

            X_all.append(anchor.X_3d)
            ba_obs.append({
                "anchor_indices": np.arange(offset, offset + n),
                "q_2d": anchor.q2d,
                "K": frame["K"],
                "T_w2c": frame["T_w2c"],
                "image_path": frame["image_path"]
            })
            offset += n

        if len(X_all) == 0:
            continue

        X_all = np.vstack(X_all)

        # ---- BA ----
        # --- 3. 执行全局 BA 优化 ---
        def ba_objective(params, observations, X_3d_anchors, s):
            rvec, tvec = params[:3], params[3:6]
            R = Rotation.from_rotvec(rvec).as_matrix()
            
            residuals = []
            for obs in observations:
                # 提取当前帧能看到的 3D 点
                X_m = X_3d_anchors[obs['anchor_indices']]
                q_obs = obs['q_2d']
                
                # 模型 -> 世界 -> 相机
                X_w = s * ((R @ X_m.T).T + tvec)
                X_w_h = np.column_stack([X_w, np.ones(len(X_w))])
                X_c = (obs['T_w2c'] @ X_w_h.T).T[:, :3]
                
                # 投影 (z 缓冲)
                z = X_c[:, 2:]
                z[z < 1e-4] = 1e-4
                uv = (X_c[:, :2] @ obs['K'][:2, :2].T) / z + obs['K'][:2, 2]
                
                residuals.append((uv - q_obs).flatten())
                
            return np.concatenate(residuals)
        def ba_obj(params):
            return ba_objective(params, ba_obs, X_all, s_fixed)

        x0 = np.concatenate([
            Rotation.from_matrix(T_M2W_curr[:3, :3]).as_rotvec(),
            T_M2W_curr[:3, 3]
        ])

        res = least_squares(
            ba_obj,
            x0,
            loss="huber",
            ftol=1e-3
        )

        T_M2W_curr[:3, :3] = Rotation.from_rotvec(res.x[:3]).as_matrix()
        T_M2W_curr[:3, 3] = res.x[3:6]

        prev_img_gray = curr_img_gray

    T_M2W_refined = T_M2W_curr.copy()
    T_M2W_refined[:3, :3] *= s_fixed # 缩放旋转部分用于渲染
    T_M2W_refined[:3, 3] *= s_fixed
    
    return T_M2W_refined

# --- 辅助函数改进 ---

def compute_reproj_err(T, X, q, K, Tw2c, s):
    R, t = T[:3, :3], T[:3, 3]
    X_w = s * ((R @ X.T).T + t)
    X_w_h = np.column_stack([X_w, np.ones(len(X_w))])
    X_c = (Tw2c @ X_w_h.T).T[:, :3]
    z = X_c[:, 2:]
    uv = (X_c[:, :2] @ K[:2, :2].T) / z + K[:2, 2]
    return np.linalg.norm(uv - q, axis=1)

def local_feature_tracking(anchor_feat_vectors, curr_feat_map, pred_q2d, feat_map_meta, search_radius=20, dist_thresh=0.5):
    """
    匹配逻辑：以预测点为中心，在 curr_feat_map 寻找与 anchor_feat_vectors 最接近的 2D 位置
    """
    device = curr_feat_map.device
    C, H, W = curr_feat_map.shape
    num_pts = len(pred_q2d)
    
    tracked_q2d = np.zeros_like(pred_q2d)
    mask = np.zeros(num_pts, dtype=bool)
    target_feats = torch.from_numpy(anchor_feat_vectors).to(device)
    
    orig_camera = feat_map_meta["orig_camera"]
    crop_camera = feat_map_meta["crop_camera"]
    feat_stride = feat_map_meta["feat_stride"]
    use_crop = feat_map_meta["use_crop"]

    # 原图 → feature map
    Z = np.ones(num_pts, dtype=np.float32)
    pred_q2d_crop = warp_points_perspective(
        src_camera=orig_camera,
        dst_camera=crop_camera,
        src_points=pred_q2d,
        src_depths=Z,
    )
    # ---------- 2. crop 图 → feature map ----------
    pred_q2d_feat = pred_q2d_crop / feat_stride
    # 限制搜索范围，加速计算
    for i in range(num_pts):
        fx, fy = pred_q2d_feat[i]
        ix, iy = int(round(fx)), int(round(fy))

        if not (0 <= ix < W and 0 <= iy < H):
            continue

        x_min = max(0, ix - search_radius)
        x_max = min(W, ix + search_radius + 1)
        y_min = max(0, iy - search_radius)
        y_max = min(H, iy + search_radius + 1)

        patch = curr_feat_map[:, y_min:y_max, x_min:x_max]
        patch_flat = patch.reshape(C, -1).permute(1, 0)
        patch_flat = patch_flat / (
            torch.norm(patch_flat, dim=1, keepdim=True) + 1e-8
        )
        dists = torch.norm(
            patch_flat - target_feats[i].unsqueeze(0),
            dim=1
        )
        
        best = torch.argmin(dists)
        if dists[best] < dist_thresh:
            w = x_max - x_min
            ry, rx = divmod(best.item(), w)

            bx = x_min + rx
            by = y_min + ry

             # ---------- 4. feature → crop ----------
            found_crop = np.array([
                bx * feat_stride[0],
                by * feat_stride[1],
            ], dtype=np.float32)
            Z = np.ones(1, dtype=np.float32)
            found_orig = warp_points_perspective(
                src_camera=crop_camera,
                dst_camera=orig_camera,
                src_points=found_crop[None],
                src_depths=Z,
            )[0]
            tracked_q2d[i] = found_orig
            mask[i] = True
            
    return tracked_q2d, mask

def get_camera_matrix(camera_model):
    """从相机模型中获取 3x3 内参矩阵 K。"""
    fx, fy = camera_model.f
    cx, cy = camera_model.c
    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=np.float32)
    return K

def warp_points_perspective(src_camera, dst_camera, src_points, src_depths):
    """
    将 2D 点从源相机坐标系 (src_camera) 映射到目标相机坐标系 (dst_camera)，
    采用完整透视几何（使用真实深度 Z）。
    """
    assert src_points.shape[0] == src_depths.shape[0], "点与深度数量不匹配"

    # --- Step 1. 从源像素坐标反投影到相机坐标系 ---
    K_src = get_camera_matrix(src_camera)           # (3,3)
    K_src_inv = np.linalg.inv(K_src)
    src_pts_h = np.hstack([src_points, np.ones((len(src_points), 1))])  # (N,3)
    # 单位深度下的归一化坐标
    rays_src = (K_src_inv @ src_pts_h.T).T          # (N,3)
    # 乘以真实深度，得到3D坐标（相机坐标系下）
    pts_src_eye = rays_src * src_depths[:, None]

    # --- Step 2. 从源相机坐标 -> 世界坐标 ---
    T_w_from_src = src_camera.T_world_from_eye      # (4,4)
    pts_src_eye_h = np.hstack([pts_src_eye, np.ones((len(src_points), 1))])
    pts_world_h = (T_w_from_src @ pts_src_eye_h.T).T

    # --- Step 3. 从世界坐标 -> 目标相机坐标 ---
    T_dst_from_w = np.linalg.inv(dst_camera.T_world_from_eye)
    pts_dst_eye_h = (T_dst_from_w @ pts_world_h.T).T
    pts_dst_eye = pts_dst_eye_h[:, :3]

    # --- Step 4. 目标相机投影到像素平面 ---
    K_dst = get_camera_matrix(dst_camera)
    pts_dst_img_h = (K_dst @ pts_dst_eye.T).T
    pts_dst_img = pts_dst_img_h[:, :2] / pts_dst_img_h[:, 2:3]

    return pts_dst_img.astype(np.float32)

def project_model_to_image(X_m, T_M2W, K, Tw2c, s):
    R, t = T_M2W[:3, :3], T_M2W[:3, 3]
    X_w = s * ((R @ X_m.T).T + t)
    X_w_h = np.column_stack([X_w, np.ones(len(X_w))])
    X_c = (Tw2c @ X_w_h.T).T[:, :3]
    # 简单的投影，不处理 z<0，因为只用于 search ROI
    z = X_c[:, 2:]
    z[np.abs(z) < 1e-4] = 1e-4
    uv = (X_c[:, :2] @ K[:2, :2].T) / z + K[:2, 2]
    return uv

def sample_features_at_q2d(feat_map, q2d, K):
    """准确采样第一帧 2D 对应的特征向量"""
    C, H, W = feat_map.shape
    q2d_torch = torch.from_numpy(q2d).to(feat_map.device).float()
    
    # 归一化到 grid_sample 要求的 [-1, 1]
    grid_x = 2.0 * q2d_torch[:, 0] / (W - 1) - 1.0
    grid_y = 2.0 * q2d_torch[:, 1] / (H - 1) - 1.0
    grid = torch.stack([grid_x, grid_y], dim=-1).view(1, 1, -1, 2)
    
    sampled = torch.nn.functional.grid_sample(
        feat_map.unsqueeze(0), grid, mode='bilinear', align_corners=True
    )
    return sampled.view(C, -1).permute(1, 0).cpu().numpy()

def sample_features_at_q2d_with_meta(
    feat_map, q2d_orig, feat_map_meta
):
    """
    q2d_orig: 原图坐标
    """
    C, Hf, Wf = feat_map.shape
    device = feat_map.device

    orig_camera = feat_map_meta["orig_camera"]
    crop_camera = feat_map_meta["crop_camera"]
    feat_stride = feat_map_meta["feat_stride"]
    use_crop = feat_map_meta["use_crop"]

    N = len(q2d_orig)

    # ---------- 原图 → crop ----------
    if use_crop:
        Z = np.ones(N, dtype=np.float32)
        q_crop = warp_points_perspective(
            src_camera=orig_camera,
            dst_camera=crop_camera,
            src_points=q2d_orig,
            src_depths=Z,
        )
    else:
        q_crop = q2d_orig.copy()

    # ---------- crop → feature ----------
    q_feat = q_crop / feat_stride
    q_feat = torch.from_numpy(q_feat).float().to(device)

    grid_x = 2.0 * q_feat[:, 0] / (Wf - 1) - 1.0
    grid_y = 2.0 * q_feat[:, 1] / (Hf - 1) - 1.0
    grid = torch.stack([grid_x, grid_y], dim=-1).view(1, 1, -1, 2)

    sampled = torch.nn.functional.grid_sample(
        feat_map.unsqueeze(0),
        grid,
        mode="bilinear",
        align_corners=True
    )

    return sampled.view(C, -1).permute(1, 0).cpu().numpy()
