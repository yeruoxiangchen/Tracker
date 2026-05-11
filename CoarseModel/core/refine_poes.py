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

def visualize_anchor_match_side_by_side(
    img0_path,
    img1_path,
    q2d_0,
    q2d_1,
    anchor_indices,
    output_path,
    max_vis=50
):
    """
    左：第一帧；右：当前帧
    同一个 anchor 用相同编号标注
    """
    img0 = cv2.imread(img0_path)
    img1 = cv2.imread(img1_path)

    if img0 is None or img1 is None:
        return

    H = max(img0.shape[0], img1.shape[0])
    W = img0.shape[1] + img1.shape[1]

    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    canvas[:img0.shape[0], :img0.shape[1]] = img0
    canvas[:img1.shape[0], img0.shape[1]:] = img1

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45
    thickness = 1

    # 最多画 max_vis 个，避免太乱
    for vis_i, idx in enumerate(anchor_indices[:max_vis]):
        p0 = q2d_0[vis_i]
        p1 = q2d_1[vis_i]

        color = tuple(np.random.randint(50, 255, size=3).tolist())

        x0, y0 = int(p0[0]), int(p0[1])
        x1, y1 = int(p1[0]) + img0.shape[1], int(p1[1])

        cv2.circle(canvas, (x0, y0), 3, color, -1)
        cv2.circle(canvas, (x1, y1), 3, color, -1)

        cv2.putText(canvas, str(idx), (x0 + 4, y0 - 4),
                    font, font_scale, color, thickness)
        cv2.putText(canvas, str(idx), (x1 + 4, y1 - 4),
                    font, font_scale, color, thickness)

    cv2.imwrite(output_path, canvas)


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


def refine_global_pose_ba(
    all_optimization_data, 
    s_fixed, 
    T_M2W_init, 
    model_vertices, # 用于最终可视化投影
    model_faces,
    max_frames=10, 
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
    

    X_3d_all = first_frame['X_3d']
    q_2d_all = first_frame['q_2d']
    K_0 = first_frame['K']
    T_w2c_0 = first_frame['T_w2c']
    
    # 筛选高质量初始点作为全局 Anchor
    err0 = compute_reproj_err(T_M2W_init, X_3d_all, q_2d_all, K_0, T_w2c_0, s_fixed)
    anchor_mask = err0 < reproj_thresh
    
    first_q2d_anchor = q_2d_all[anchor_mask]
    
    # 全局固定的 3D 点集和对应的特征向量
    X_3d_anchors = X_3d_all[anchor_mask]
    first_feat_map = first_frame.get('feature_map_chw_proj')
    # anchor_feats = sample_features_at_q2d(first_feat_map, q_2d_all[anchor_mask], K_0)
    first_meta = first_frame.get("feat_map_meta")

    anchor_feats = sample_features_at_q2d_with_meta(
        first_feat_map,
        q_2d_all[anchor_mask],
        first_meta
    )
    anchor_feats = anchor_feats / (
        np.linalg.norm(anchor_feats, axis=1, keepdims=True) + 1e-8
    )

    # ba_observations 显式存储每帧的观测信息
    ba_observations = []
    
    # 第一帧观测 (所有 anchor 都在)
    ba_observations.append({
        'anchor_indices': np.arange(len(X_3d_anchors)),
        'q_2d': q_2d_all[anchor_mask],
        'K': K_0,
        'T_w2c': T_w2c_0,
        'image_path': first_frame['image_path'],
    })

    # --- 2. 锚定跟踪 (Frame 0 -> Frame i) ---
    for i in range(0, min(len(all_optimization_data), max_frames)):
        curr_frame = all_optimization_data[i]
        curr_feat_map = curr_frame.get('feature_map_chw_proj')
        
        # A. 使用 3D 投影确定搜索窗口 (ROI)
        # 注意：这里只用 T_M2W_init 引导搜索
        pred_q2d = project_model_to_image(X_3d_anchors, T_M2W_init, curr_frame['K'], curr_frame['T_w2c'], s_fixed)
        
        # B. 局部特征匹配 (始终与第一帧特征匹配)
        tracked_q2d, track_mask = local_feature_tracking(
            anchor_feats, 
            curr_feat_map, 
            pred_q2d,
            curr_frame["feat_map_meta"], 
            search_radius=search_radius,
            dist_thresh=dist_thresh
        )
        track_mask = filter_tracked_points_by_mask(
            tracked_q2d,
            track_mask,
            curr_frame["image_path"]
        )
        # C. 可视化跟踪结果
        vis_img = cv2.imread(curr_frame['image_path'])
        for p_pred, p_track, m in zip(pred_q2d, tracked_q2d, track_mask):
            if m:
                # 预测点红色，跟踪点绿色，连线显示偏移
                cv2.circle(vis_img, tuple(p_pred.astype(int)), 3, (0, 0, 255), -1)
                cv2.circle(vis_img, tuple(p_track.astype(int)), 2, (0, 255, 0), -1)
                cv2.line(vis_img, tuple(p_pred.astype(int)), tuple(p_track.astype(int)), (255, 255, 0), 1)
        
        cv2.putText(vis_img, f"Frame {i} Tracking", (20, 40), 1, 1.5, (255,255,255), 2)
        track_dir = output_dir + "/track"
        os.makedirs(track_dir, exist_ok=True)
        cv2.imwrite(os.path.join(track_dir, f"track_frame_{i:03d}.jpg"), vis_img)
        idxs = np.where(track_mask)[0]
        if len(idxs) < 8:
            logger.warning(f"Frame {i}: Too few tracked points ({len(idxs)}), stopping sequence.")
            break
        comp_dir = output_dir + "/comp"
        os.makedirs(comp_dir, exist_ok=True)
        visualize_anchor_match_side_by_side(
            img0_path=first_image_path,
            img1_path=curr_frame["image_path"],
            q2d_0=first_q2d_anchor[idxs],
            q2d_1=tracked_q2d[idxs],
            anchor_indices=idxs,
            output_path=os.path.join(
                comp_dir, f"compare_frame_{i:03d}.jpg"
            ),
            max_vis=50
        )
        ba_observations.append({
            'anchor_indices': idxs,
            'q_2d': tracked_q2d[idxs],
            'K': curr_frame['K'],
            'T_w2c': curr_frame['T_w2c'],
            'image_path': curr_frame['image_path'],
        })
        logger.info(f"Frame {i}: Tracked {len(idxs)} / {len(X_3d_anchors)} anchor points.")

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

    init_rvec = Rotation.from_matrix(T_M2W_init[:3, :3]).as_rotvec()
    init_tvec = T_M2W_init[:3, 3]
    x0 = np.concatenate([init_rvec, init_tvec])
    
    res = least_squares(
        ba_objective, x0, 
        args=(ba_observations, X_3d_anchors, s_fixed),
        method='trf', 
        loss='huber',
        ftol=1e-3, 
        verbose=2
    )

    T_M2W_refined = np.eye(4)
    T_M2W_refined[:3, :3] = Rotation.from_rotvec(res.x[:3]).as_matrix()
    T_M2W_refined[:3, 3] = res.x[3:6]
    
    for obs in ba_observations:
        # 注意渲染时需要应用尺度。因为渲染函数通常接受标准 T_M2C，
        # 我们传入：T_M2C_eff = T_W2C @ [sR | t]
        T_M2W_scaled = T_M2W_refined.copy()
        T_M2W_scaled[:3, :3] *= s_fixed # 缩放旋转部分用于渲染
        T_M2W_scaled[:3, 3] *= s_fixed
        # 注意：如果平移 t 已经是世界坐标，则不乘 s。如果是 s*t 则需乘。
        # 按照公式 X_w = s*RX_m + t, 渲染器投影点时应为 K(R_w2c(s*R_m2w*X_m + t_m2w) + t_w2c)
        
        visualize_projection(
            image_path=obs['image_path'],
            output_dir=output_dir,
            K=obs['K'],
            T_W2C=obs['T_w2c'],
            T_M2W=T_M2W_scaled, # 这里的 scaled 仅供 render_mesh 兼容旧接口使用
            model_vertices=model_vertices,
            model_faces=model_faces,
            root_name="ba"
        )
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
