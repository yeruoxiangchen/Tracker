# core/global_optimize.py
import os
import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from datetime import datetime

from core.util import ( 
    axis_angle_to_rotmat, project_point, 
    read_colmap_points3D, read_colmap_images_txt
)
from collections import defaultdict

from contextlib import redirect_stdout
from scipy.spatial import cKDTree
from script.vis_util import visualize_contour_matches
from script.vis_util import (
    visualize_projection,
    visualize_per_frame_alignment
    )
import time
from collections import defaultdict

import pyrender
import trimesh
import numpy as np
import cv2

import torch
import kornia
from kornia.feature import LoFTR

# 全局初始化 LoFTR (建议放在文件头部)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loftr_matcher = LoFTR(pretrained='outdoor').to(device).eval() # 对于一般物体，outdoor 权重通常鲁棒性更好

def project_points_batch(P_W, K, T_w2c):
    # (保持不变)
    R = T_w2c[:3, :3]
    t = T_w2c[:3, 3]

    P_C = (R @ P_W.T).T + t
    z = P_C[:, 2]

    uvw = (K @ P_C.T).T
    uv = uvw[:, :2] / uvw[:, 2:3]

    return uv, z


def get_object_3d_points(
    points3d_path,
    all_optimization_data,
    mask_dir,
    images_colmap,
    min_observations=3,
):
    # (保持不变)
    print("\n--- Collecting object 3D points (FAST VERSION) ---")

    points3d = read_colmap_points3D(points3d_path)
    if not points3d:
        return np.array([])

    id_to_data = {}
    for entry in all_optimization_data:
        frame_name = entry['frame_name']
        if frame_name in images_colmap:
            image_id = images_colmap[frame_name]['image_id']
            id_to_data[image_id] = {
                'K': entry['K'],
                'T_w2c': entry['T_w2c'],
                'mask_path': os.path.join(
                    mask_dir,
                    frame_name.replace(".jpg", ".png").replace(".JPG", ".png")
                )
            }

    image_to_points = defaultdict(list)
    points_xyz = {}

    for pid, pdata in points3d.items():
        points_xyz[pid] = pdata['xyz']
        track = pdata['track_list']
        for i in range(0, len(track), 2):
            image_id = track[i]
            if image_id in id_to_data:
                image_to_points[image_id].append(pid)

    mask_cache = {}

    def load_mask(path):
        if path not in mask_cache:
            if not os.path.exists(path):
                mask_cache[path] = None
            else:
                mask_cache[path] = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        return mask_cache[path]

    point_in_mask_count = defaultdict(int)

    for image_id, pids in image_to_points.items():
        data = id_to_data[image_id]
        mask = load_mask(data['mask_path'])
        if mask is None:
            continue

        K = data['K']
        T_w2c = data['T_w2c']
        h, w = mask.shape

        P_W = np.stack([points_xyz[pid] for pid in pids], axis=0)
        uv, z = project_points_batch(P_W, K, T_w2c)

        valid = (
            (z > 0) &
            (uv[:, 0] >= 0) & (uv[:, 0] < w) &
            (uv[:, 1] >= 0) & (uv[:, 1] < h)
        )

        if not np.any(valid):
            continue

        uv = uv[valid].astype(np.int32)
        valid_pids = np.array(pids)[valid]

        mask_values = mask[uv[:, 1], uv[:, 0]] > 0

        for pid in valid_pids[mask_values]:
            point_in_mask_count[pid] += 1

    object_points_W = [
        points_xyz[pid]
        for pid, cnt in point_in_mask_count.items()
        if cnt >= min_observations
    ]

    print(f"Selected {len(object_points_W)} object 3D points.")
    return np.array(object_points_W, dtype=np.float64)


def get_median_pairwise_distance(points, num_pairs=20000):
    # (保持不变)
    n = points.shape[0]
    if n < 2:
        return 1e-6
    
    idx1 = np.random.randint(0, n, num_pairs)
    idx2 = np.random.randint(0, n, num_pairs)
    
    valid = idx1 != idx2
    p1 = points[idx1[valid]]
    p2 = points[idx2[valid]]
    
    if len(p1) == 0:
        return 1e-6
        
    dists = np.linalg.norm(p1 - p2, axis=1)
    return np.median(dists)


def calculate_initial_scales_from_median_dist(model_vertices, X_W_obj):
    # (保持不变)
    if model_vertices.size == 0 or X_W_obj.size == 0:
        raise ValueError("Model vertices or object 3D points are empty.")

    print("\n--- Calculating Initial Scale using Pairwise Distance Median ---")
    dist_M = get_median_pairwise_distance(model_vertices)
    dist_W = get_median_pairwise_distance(X_W_obj)
    
    scale_init = dist_W / (dist_M + 1e-6)
    scale_init = max(scale_init, 1e-6)
    
    s_xyz_init = np.array([scale_init, scale_init, scale_init])
    
    print(f"Model Median Distance: {dist_M:.6f}")
    print(f"World Median Distance: {dist_W:.6f}")
    print(f"Initial Uniform 3-Axis Scales (sx, sy, sz): {s_xyz_init}")
    
    return s_xyz_init


def _compute_reprojection_residuals_core(scales, rvec, t, all_data):
    """提取公共的 2D 重投影残差计算核心"""
    from scipy.spatial.transform import Rotation
    R = Rotation.from_rotvec(rvec).as_matrix()
    residuals = []

    Z_PENALTY_WEIGHT = 500.0
    EPS = 1e-4
    num_2d_points = 0
    
    for d in all_data:
        X = d["X_3d"]
        q = d["q_2d"]
        K = d["K"]
        T_w2c = d["T_w2c"]
        
        num_2d_points += len(q)
        
        X_scaled = X * scales[None, :]
        Xw = (R @ X_scaled.T).T + t
        Xw_h = np.hstack([Xw, np.ones((len(Xw), 1))])
        Xc = (T_w2c @ Xw_h.T).T[:, :3]

        Z = Xc[:, 2]
        res = np.zeros_like(q)
        mask_front = Z > EPS
        mask_behind = ~mask_front

        if np.any(mask_front):
            z_inv = 1.0 / Z[mask_front]
            u_proj = (K[0, 0] * Xc[mask_front, 0] * z_inv) + K[0, 2]
            v_proj = (K[1, 1] * Xc[mask_front, 1] * z_inv) + K[1, 2]
            res[mask_front, 0] = u_proj - q[mask_front, 0]
            res[mask_front, 1] = v_proj - q[mask_front, 1]

        if np.any(mask_behind):
            penalty = Z_PENALTY_WEIGHT * (EPS - Z[mask_behind])
            res[mask_behind, 0] = penalty
            res[mask_behind, 1] = penalty

        residuals.append(res.ravel())

    return np.concatenate(residuals), num_2d_points


def error_optimize_joint(params, all_data, s_init, lambda_3d):
    """Phase 1: 联合优化残差函数"""
    scales = params[0:3]
    rvec = params[3:6]
    t = params[6:9]
    
    res_2d, num_2d_points = _compute_reprojection_residuals_core(scales, rvec, t, all_data)
    
    # 3D 惩罚
    if lambda_3d > 0:
        scale_penalty_weight = lambda_3d * np.sqrt(max(num_2d_points, 1))
        res_3d_penalty = scale_penalty_weight * (scales - s_init)
        return np.concatenate([res_2d, res_3d_penalty])
    
    return res_2d


def error_optimize_scale_only(scales, fixed_rvec, fixed_t, all_data, s_init, lambda_3d_phase2):
    """Phase 2: 仅优化 Scale 残差函数"""
    res_2d, num_2d_points = _compute_reprojection_residuals_core(scales, fixed_rvec, fixed_t, all_data)
    
    if lambda_3d_phase2 > 0:
        scale_penalty_weight = lambda_3d_phase2 * np.sqrt(max(num_2d_points, 1))
        res_3d_penalty = scale_penalty_weight * (scales - s_init)
        return np.concatenate([res_2d, res_3d_penalty])
        
    return res_2d

def error_optimize_uniform_joint(params, all_data, s_init_scalar, lambda_3d):
    """联合优化：单轴 Scale (params[0]) + RT (params[1:7])"""
    s_uni = params[0]
    rvec = params[1:4]
    t = params[4:7]
    
    # 将单轴扩展为三轴传给核心函数
    scales_3d = np.array([s_uni, s_uni, s_uni])
    res_2d, num_2d_points = _compute_reprojection_residuals_core(scales_3d, rvec, t, all_data)
    
    if lambda_3d > 0:
        # 对单轴 scale 进行 3D 惩罚
        scale_penalty_weight = lambda_3d * np.sqrt(max(num_2d_points, 1))
        res_3d_penalty = scale_penalty_weight * (s_uni - s_init_scalar)
        return np.concatenate([res_2d, [res_3d_penalty]])
    return res_2d

def optimize_global_pose(
    all_optimization_data,
    all_corresps_data,
    model_vertices,
    model_faces,
    output_dir,
    colmap_dir,
    mask_dir,
    init_frame_id=0,
    lambda_3d_strong=500.0, # 用于 Phase 1，极强约束保持等比例
    lambda_3d_weak=10.0      # 用于后续微调
):
    """两阶段优化：先联合优化，再固定 RT 优化 Scale"""

    # --- 1. 获取初始位姿与 Scale ---
    best_pose = all_corresps_data[init_frame_id].best_pose
    T_m2c = np.eye(4)
    T_m2c[:3, :3] = best_pose["R_m2c"]
    T_m2c[:3, 3] = best_pose["t_m2c"]

    T_w2c = all_optimization_data[init_frame_id]["T_w2c"]
    T_m2w_init = np.linalg.inv(T_w2c) @ T_m2c

    R_init = T_m2w_init[:3, :3]
    t_init = T_m2w_init[:3, 3]

    print(f"\n--- Initializing Scale via Camera Baseline (Frame {init_frame_id} & {init_frame_id+2}) ---")
    idx1 = init_frame_id
    idx2 = init_frame_id + 2
    def get_cam_center(opt_data, corr_data, f_id):
        # World 空间相机中心
        T_w2c = opt_data[f_id]["T_w2c"]
        C_w = -T_w2c[:3, :3].T @ T_w2c[:3, 3]
        # Model 空间相机中心 (未缩放)
        best_pose = corr_data[f_id].best_pose
        R_m2c, t_m2c = best_pose["R_m2c"], best_pose["t_m2c"]
        C_m = -R_m2c.T @ t_m2c.ravel()
        return C_w, C_m
    try:
        Cw1, Cm1 = get_cam_center(all_optimization_data, all_corresps_data, idx1)
        Cw2, Cm2 = get_cam_center(all_optimization_data, all_corresps_data, idx2)
        dist_w = np.linalg.norm(Cw1 - Cw2)
        dist_m = np.linalg.norm(Cm1 - Cm2)
        s_init_scalar = dist_w / (dist_m + 1e-6)
    except (IndexError, KeyError):
        print("Warning: Baseline frames not found, fallback to scale=1.0")
        s_init_scalar = 1.0
    s_init = np.array([s_init_scalar, s_init_scalar, s_init_scalar])
    print(f"Calculated s_init: {s_init_scalar:.6f} (Baseline W: {dist_w:.4f}, M: {dist_m:.4f})")
    
    best_pose = all_corresps_data[init_frame_id].best_pose
    T_m2c_scaled = np.eye(4)
    T_m2c_scaled[:3, :3] = best_pose["R_m2c"]
    T_m2c_scaled[:3, 3] = best_pose["t_m2c"].ravel() * s_init_scalar # 缩放平移

    T_w2c_init = all_optimization_data[init_frame_id]["T_w2c"]
    T_m2w_init = np.linalg.inv(T_w2c_init) @ T_m2c_scaled

    R_init = T_m2w_init[:3, :3]
    t_init = T_m2w_init[:3, 3]
    rvec_init = Rotation.from_matrix(R_init).as_rotvec()
    
    # ==========================================================
    # STAGE 1: 锁定 Scale, 只优化 R 和 T
    # ==========================================================
    print("\n=== Stage 1: Fix Scale, Optimize RT Only ===")
    
    def error_rt_only(rt_params, fixed_s, all_data):
        r_val = rt_params[0:3]
        t_val = rt_params[3:6]
        # 调用核心残差函数，但传入固定的 scales
        res_2d, _ = _compute_reprojection_residuals_core(fixed_s, r_val, t_val, all_data)
        return res_2d

    date_str = datetime.now().strftime("%Y-%m-%d")
    log_filename = f"4stage_least_squares_log_{date_str}.txt"
    log_filepath = os.path.join(output_dir, log_filename)
    from contextlib import redirect_stdout
    from scipy.optimize import least_squares
    with open(log_filepath, 'w') as f:
        with redirect_stdout(f):
            rt_init = np.concatenate([rvec_init, t_init])
            res_stage1 = least_squares(
                fun=error_rt_only,
                x0=rt_init,
                args=(s_init, all_optimization_data),
                method='trf',
                verbose=2,
                loss='soft_l1'
            )
            
            rvec_after_s1 = res_stage1.x[0:3]
            t_after_s1 = res_stage1.x[3:6]
            print(f"Stage 1 Finished. RT adjusted while keeping Scale fixed at {s_init}")
            # ===== Stage1 Visualization =====
            from script.vis_util import visualize_projection

            R_s1 = Rotation.from_rotvec(rvec_after_s1).as_matrix()

            T_stage1 = np.eye(4, dtype=np.float32)
            T_stage1[:3, :3] = R_s1 @ np.diag(s_init)
            T_stage1[:3, 3] = t_after_s1

            # ==========================================================
            # STAGE 3: 联合优化 (三轴 Scale + RT)
            # ==========================================================
            print("\n=== Stage 3: Joint Three-axis Scale (3-axis) + RT ===")
            # 用 Stage 2 的结果初始化，三轴初值设为一样的
            # 2. 将旋转矩阵 R_s1 转换为旋转向量 (3维)，否则维度对不上
            r_vec_init = Rotation.from_matrix(R_s1).as_rotvec()
            # 3. 确保 t 也是平的一维向量 (3维)
            t_vec_init = t_after_s1.flatten()
            # 4. 现在拼接，总长度应该是 3 (scale) + 3 (rotation) + 3 (translation) = 9
            params_s3_init = np.concatenate([s_init, r_vec_init, t_vec_init])
            res_s3 = least_squares(
                fun=error_optimize_joint, 
                x0=params_s3_init, 
                args=(all_optimization_data, s_init, lambda_3d_strong), 
                method='trf',
                bounds=(np.array([1e-4]*3 + [-np.inf]*6), np.array([np.inf]*9)),
                verbose=2, loss='soft_l1'
            )
            s_s3, r_s3, t_s3 = res_s3.x[0:3], res_s3.x[3:6], res_s3.x[6:9]
            R_s3 = Rotation.from_rotvec(r_s3).as_matrix()
            
            T_stage3 = np.eye(4, dtype=np.float32)
            T_stage3[:3, :3] = R_s3 @ np.diag(s_s3)
            T_stage3[:3, 3] = t_s3
            
    final_s = s_s3
    final_t = t_s3
    final_R = R_s3
    print(f"Final Optimized Scales: {s_s3}")
    T_final = np.eye(4, dtype=np.float32)
    T_final[:3, :3] = final_R @ np.diag(final_s)
    T_final[:3, 3] = final_t

    return T_final, res_s3, final_s


def preprocess_2d_contours(all_optimization_data, num_samples=20):
    """
    遍历数据，读取 mask，提取轮廓点位置和法向（梯度）
    """
    print(f"--- Preprocessing 2D Contours (samples={num_samples}) ---")
    for data in all_optimization_data:
        # 1. 路径转换 logic: .../rgb/0000.jpg -> .../masks/0000.png
        image_path = data['image_path']
        mask_path = image_path.replace("/rgb/", "/masks/").replace(".jpg", ".png")
        
        if not os.path.exists(mask_path):
            print(f"Warning: Mask not found {mask_path}")
            data['edge_2d'] = None
            continue
            
        # 2. 读取 mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # 3. 提取轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if len(contours) == 0:
            data['edge_2d'] = None
            continue
            
        # 合并最长轮廓（假设只有一个主要物体）
        c = max(contours, key=cv2.contourArea)
        pts = c.squeeze() # (N, 2)
        
        # 4. 均匀采样
        if len(pts) > num_samples:
            indices = np.linspace(0, len(pts)-1, num_samples, dtype=int)
            sampled_pts = pts[indices]
        else:
            sampled_pts = pts
            
        # 5. 计算梯度作为 2D 法向
        # 使用 Distance Transform 的梯度更平滑，指向物体外部
        dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        # 注意：dist map 内部值大，边缘小。梯度指向内部。我们需要指向外部，所以取反。
        dy, dx = np.gradient(dist_transform) 
        
        edge_grads = []
        valid_pts = []
        
        for pt in sampled_pts:
            gx = -dx[pt[1], pt[0]] # 取反指向外
            gy = -dy[pt[1], pt[0]]
            norm = np.sqrt(gx**2 + gy**2)
            if norm > 1e-6:
                edge_grads.append([gx/norm, gy/norm])
                valid_pts.append(pt)
        
        data['edge_2d'] = np.array(valid_pts, dtype=np.float32)
        data['edge_grad'] = np.array(edge_grads, dtype=np.float32)

def compute_vertex_normals(vertices, faces):
    """
    使用纯 Numpy 快速计算顶点法线 (加权平均面法线)
    """
    # 1. 计算面法线
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    # 叉乘得到面法线 (未归一化，模长即面积的两倍，正好作为权重)
    face_normals = np.cross(v1 - v0, v2 - v0)
    
    # 2. 累加到顶点
    vertex_normals = np.zeros_like(vertices)
    # 使用 add.at 进行原位累加
    np.add.at(vertex_normals, faces[:, 0], face_normals)
    np.add.at(vertex_normals, faces[:, 1], face_normals)
    np.add.at(vertex_normals, faces[:, 2], face_normals)
    
    # 3. 归一化
    norms = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
    vertex_normals = vertex_normals / np.maximum(norms, 1e-8)
    return vertex_normals

def visualize_loftr_matches(img_real, img_render, pts_real, pts_render, save_path):
    """
    将 LoFTR 匹配结果画出来
    """
    h, w = img_real.shape[:2]
    # 左右拼接
    canvas = np.zeros((h, w * 2, 3), dtype=np.uint8)
    canvas[:, :w] = img_real
    canvas[:, w:] = img_render
    
    # 画线
    for p1, p2 in zip(pts_real, pts_render):
        pt1 = (int(p1[0]), int(p1[1]))
        pt2 = (int(p2[0] + w), int(p2[1])) # 偏移到右图
        cv2.line(canvas, pt1, pt2, (0, 255, 0), 1)
        cv2.circle(canvas, pt1, 2, (0, 0, 255), -1)
        cv2.circle(canvas, pt2, 2, (0, 0, 255), -1)
        
    cv2.imwrite(save_path, canvas)
def render_current_mesh(V_current, model_faces, data, T_M2W, mesh_template):
    """
    使用 pyrender 进行离屏渲染
    V_current: (N, 3) 当前变形后的顶点
    model_faces: (M, 3) 面片索引
    data: 包含 'K' (内参) 和 'T_w2c' (相机外参) 的字典
    T_M2W: 模型到世界的变换
    """
    # 1. 准备内参和图像尺寸
    K = data['K']
    T_w2c = data['T_w2c']
    img_example = cv2.imread(data['image_path'])
    h, w = img_example.shape[:2]
    
    # 2. 创建 trimesh 对象 (可以添加顶点颜色以增强 LoFTR 匹配)
    # 如果你的模型有贴图，在这里加载 vertex_colors 或 visual
    # 更新顶点位置，保持 UV 坐标不变
    mesh = mesh_template.copy()
    mesh.vertices = V_current
    
    # 给模型一个默认材质，或者使用原始模型的颜色
    # 提示：LoFTR 在有纹理的情况下表现最好。如果没纹理，建议渲染 Normal Map 或加随机色块
    scene_mesh = pyrender.Mesh.from_trimesh(mesh)
    
    # 3. 构建 pyrender 场景
    scene = pyrender.Scene(ambient_light=[0.5, 0.5, 0.5], bg_color=[0, 0, 0])
    scene.add(scene_mesh)
    
    # 4. 设置相机
    # pyrender 使用 OpenGL 坐标系，需要转换：将 OpenCV 相机坐标系转为 OpenGL
    # OpenCV: x-right, y-down, z-forward | OpenGL: x-right, y-up, z-backward
    T_opencv_to_opengl = np.array([
        [1,  0,  0, 0],
        [0, -1,  0, 0],
        [0,  0, -1, 0],
        [0,  0,  0, 1]
    ])
    
    # 计算相机在世界空间的位置
    T_total = T_w2c @ T_M2W
    cam_pose_world = np.linalg.inv(T_total) @ T_opencv_to_opengl
    
    camera = pyrender.IntrinsicsCamera(
        fx=K[0, 0], fy=K[1, 1], 
        cx=K[0, 2], cy=K[1, 2]
    )
    scene.add(camera, pose=cam_pose_world)
    
    # 5. 添加光照 (为了让 LoFTR 能看到几何特征)
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0)
    scene.add(light, pose=cam_pose_world)

    # 6. 执行渲染
    try:
        r = pyrender.OffscreenRenderer(viewport_width=w, viewport_height=h)
        color, _ = r.render(scene)
        r.delete() # 释放资源，防止显存泄漏
        color_bgr = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
        return color_bgr # 返回 RGB 数组
    except Exception as e:
        print(f"Render failed: {e}")
        return None

def extract_loftr_matches(img_real, img_render):
    """
    使用 LoFTR 提取两张图像的匹配点
    """
    # 转换为灰度图并转为 Tensor (1, 1, H, W)，归一化到 [0, 1]
    img1_tensor = kornia.image_to_tensor(cv2.cvtColor(img_real, cv2.COLOR_BGR2GRAY), False).float() / 255.0
    img2_tensor = kornia.image_to_tensor(cv2.cvtColor(img_render, cv2.COLOR_BGR2GRAY), False).float() / 255.0
    
    img1_tensor = img1_tensor.to(device)
    img2_tensor = img2_tensor.to(device)
    
    input_dict = {"image0": img1_tensor, "image1": img2_tensor}
    
    with torch.no_grad():
        correspondences = loftr_matcher(input_dict)
        
    # 获取匹配的 2D 坐标
    mkpts0 = correspondences['keypoints0'].cpu().numpy() # real image pts
    mkpts1 = correspondences['keypoints1'].cpu().numpy() # render image pts
    confidence = correspondences['confidence'].cpu().numpy()
    
    # 过滤低置信度匹配
    valid = confidence > 0.8
    return mkpts0[valid], mkpts1[valid]

def update_correspondences(V_current, model_faces, T_M2W, all_optimization_data, outer_iter, output_dir, mesh_template):
    """
    在当前模型 V_current 下，为每一帧的 2D 轮廓点找到 Mesh 上对应的 3D 顶点索引。
    结果存储在 data['contour_v_ids'] 中。
    """
    # --- 1. 实时计算顶点法线 ---
    # 因为 V_current 是变形后的，法线变了
    N_3d = compute_vertex_normals(V_current, model_faces)
    # V_current 是世界坐标系下的 (或者是初始对齐后的)
    
    for data in all_optimization_data:
        if data.get('edge_2d') is None or data.get('edge_grad') is None:
            continue
            
        K_cam = data['K']
        T_w2c = data['T_w2c']
        edge_2d = data['edge_2d']       # (M, 2)
        edge_grad = data['edge_grad']   # (M, 2) 图像边缘的法向 (指向外)
        
        # --- 2. 顶点投影 (3D -> 2D) ---
        T_total = T_w2c @ T_M2W
        # 坐标变换
        V_h = np.hstack([V_current, np.ones((len(V_current), 1))])
        V_cam = (T_total @ V_h.T).T[:, :3]
        z = V_cam[:, 2:3]
        # 法线变换 (只旋转，不平移)
        # Normal_cam = R * Normal_world
        R_total = T_total[:3, :3]
        N_cam = (R_total @ N_3d.T).T
        # 近似 2D 法线：直接取相机空间法线的 x,y 分量
        # 这一步假设透视投影在局部是近似正交的，对于轮廓点通常成立
        N_2d_proj = N_cam[:, :2]
        # 归一化 2D 法线
        norm_n2d = np.linalg.norm(N_2d_proj, axis=1, keepdims=True)
        N_2d_proj = N_2d_proj / np.maximum(norm_n2d, 1e-6)
        # --- 3. 可见性剔除 ---
        # 只有法线朝向相机 (z > 0) 且位置在相机前方 (z > 0) 的点才考虑
        # N_cam[:, 2] 是法线在相机 Z 轴投影，如果 > 0 表示背向相机(取决于坐标系定义，通常 OpenGL 是 < 0 朝向相机)
        # 这里为了稳健，我们主要依赖位置 z > 0.1
        valid_mask = (z > 0.1).squeeze()
        valid_indices = np.where(valid_mask)[0]
        
        if len(valid_indices) == 0:
            data['contour_v_ids'] = None
            continue
        
        # 提取有效点的 2D 坐标和 2D 法线
        proj_all = (K_cam @ V_cam.T).T[:, :2] / np.maximum(z, 1e-6)
        proj_valid = proj_all[valid_indices]
        normals_valid = N_2d_proj[valid_indices]
        
        # --- 4. 混合匹配策略 (Spatial KNN + Normal Dot) ---
        tree = cKDTree(proj_valid)
        
        # A. 搜索 K 个最近邻 (候选池)
        k_neighbors = 5
        # dists: (M, k), neighbor_idx_local: (M, k) 是 valid_indices 中的下标
        dists, neighbor_idx_local = tree.query(edge_2d, k=k_neighbors)
        
        # 处理 K=1 的情况 (如果 tree 中点太少)
        if k_neighbors == 1 or len(proj_valid) < k_neighbors:
             # 回退到简单匹配
             mesh_v_ids = valid_indices[neighbor_idx_local]
             data['contour_v_ids'] = mesh_v_ids
             continue

        # B. 获取候选点的法线
        # candidates_normals shape: (M, k, 2)
        candidates_normals = normals_valid[neighbor_idx_local] 
        
        # C. 计算法线相似度 (Cosine Similarity)
        # edge_grad shape: (M, 2) -> (M, 1, 2)
        target_normals = edge_grad[:, None, :]
        
        # dot product: sum(a * b, axis=2) -> (M, k)
        # 这里的 edge_grad 是图像梯度（指向外），candidates_normals 是顶点法线（指向外）
        # 我们期望它们方向一致，所以找 max dot product
        cos_sim = np.sum(candidates_normals * target_normals, axis=2)
        
        # D. 综合打分 (Score = Similarity)
        # 如果你希望距离也占权重，可以设计 score = w1 * (1/dist) + w2 * cos_sim
        # 但通常在该半径内，法线一致性更重要，直接取法线最对齐的
        # 过滤掉法线完全反向的点 (cos_sim < 0) - 可选
        # cos_sim[cos_sim < 0] = -1.0 
        
        best_neighbor_idx = np.argmax(cos_sim, axis=1) # (M,) 获取每行最大的索引 0..k-1
        
        # E. 提取最终索引
        # advanced indexing: [row_indices, col_indices]
        row_indices = np.arange(len(edge_2d))
        final_local_indices = neighbor_idx_local[row_indices, best_neighbor_idx]
        
        # 映射回 Mesh 全局索引
        mesh_v_ids = valid_indices[final_local_indices]
        data['contour_v_ids'] = mesh_v_ids
        data['last_v_proj'] = proj_all[mesh_v_ids]
        
        # ==========================================
        # 模块 B: LoFTR 特征匹配逻辑 (新增)
        # ==========================================
        # 1. 渲染当前 Mesh
        img_render = render_current_mesh(V_current, model_faces, data, T_M2W, mesh_template)
        img_real = cv2.imread(data['image_path'])
        
        if img_render is not None and img_real is not None:
            # 2. 获取 2D-2D 匹配点
            pts_real, pts_render = extract_loftr_matches(img_real, img_render)
            
            if len(pts_render) > 0:
                # 3. 反投影: 为渲染图上的特征点找到最近的 3D 顶点
                # 查询距离渲染特征点最近的 2D 投影顶点
                dists_feat, feat_idx_local = tree.query(pts_render, k=1)
                
                # 过滤掉距离过远的点（可能是遮挡边缘产生的误匹配）
                valid_feat_mask = dists_feat < 5.0 # 像素误差容忍度
                
                feat_mesh_v_ids = valid_indices[feat_idx_local[valid_feat_mask]]
                feat_targets_2d = pts_real[valid_feat_mask]
                
                data['feature_v_ids'] = feat_mesh_v_ids
                data['feature_targets_2d'] = feat_targets_2d
            else:
                data['feature_v_ids'] = None
            # 3. 保存可视化 (每个 Outer Loop 保存一次)
            frame_name = os.path.basename(data['image_path']).split('.')[0]
            feature_dir = os.path.join(output_dir, "loftr")
            os.makedirs(feature_dir,exist_ok=True)
            vis_path = os.path.join(feature_dir, f"loftr_iter{outer_iter}_{frame_name}.jpg")
            
            visualize_loftr_matches(img_real, img_render, pts_real, pts_render, vis_path)
        
    
def refine_model_with_deformation_graph(
    model_vertices,
    model_faces,
    T_M2W,
    all_optimization_data,
    output_dir,
    num_nodes=20,
    knn_node=4,
    knn_vertex=4,
    lambda_reg=1e4
):
    obj_path = "/home/zjr/Tracker/CoarseModel/datasets/snoopy/models/snoopy.obj"
    # trimesh 会自动根据 .obj 里的 mtllib 找到 .mtl 和 .png
    mesh_template = trimesh.load(obj_path, process=False)
    # --- 初始化统计字典 ---
    stats = defaultdict(float)
    total_start = time.time()

    # --- 0. 预处理 2D 轮廓 ---
    t_start = time.time()
    # 只需执行一次
    preprocess_2d_contours(all_optimization_data, num_samples=20)
    stats["0_Preprocess_2D"] = time.time() - t_start
    
    V0 = model_vertices.copy()
    
    # --- 1. 逻辑焊接预处理 ---
    t_start = time.time()
    # 找到物理位置重合的顶点。round(6) 用于处理微小的浮点数精度问题
    _, inverse_mapping = np.unique(np.round(V0, 6), axis=0, return_inverse=True)
    stats["1_Vertex_Welding"] = time.time() - t_start
    
    # --- 2. 采样与图构建 ---
    t_start = time.time()
    node_indices = farthest_point_sampling(V0, num_nodes)
    node_pos = V0[node_indices]   # (K,3)
    
    # --- 2. 构建 graph ---
    graph_edges = build_knn_graph(node_pos, knn_node)
    vertex_nodes, vertex_weights = compute_vertex_node_weights(
        V0, node_pos, k=knn_vertex
    )
    graph_edges = np.array(graph_edges)
    stats["2_Graph_Building"] = time.time() - t_start
    
    # --- [修改：计算统一权重] ---
    # 1. 提取不重复的顶点位置
    # --- 3. 权重计算与广播 ---
    t_start = time.time()
    unique_v_indices = np.unique(inverse_mapping, return_index=True)[1]
    unique_V0 = V0[unique_v_indices]
    
    # 2. 只对唯一的位置计算 KNN 权重
    unique_vertex_nodes, unique_vertex_weights = compute_vertex_node_weights(
        unique_V0, node_pos, k=knn_vertex
    )
    
    # 3. 将权重广播回所有原始顶点索引
    # 这样，即使 OBJ 拆分了多个顶点，只要位置相同，形变位移就绝对一致
    vertex_nodes = unique_vertex_nodes[inverse_mapping]
    vertex_weights = unique_vertex_weights[inverse_mapping]

    stats["3_Weight_Broadcasting"] = time.time() - t_start
    # --- 4. 迭代优化 (ICP-style Loop) ---
    # 定义外层循环次数，通常 2-3 次即可收敛
    outer_iterations = 3
    
    K = node_pos.shape[0]
    # 初始参数 x0 (平移为0, 旋转为0即单位阵)
    current_x = np.zeros(6 * K, dtype=np.float64)
    
    for outer_iter in range(outer_iterations):
        iter_start = time.time()
        print(f"=== Outer Iteration {outer_iter + 1}/{outer_iterations} ===")
        
        # A. 根据当前的 deformation 参数，计算当前的 Mesh 形态
        t_sub = time.time()
        R_list, t_list = unpack_params(current_x)
        V_current = apply_deformation_graph(
            V0, node_pos, R_list, t_list, vertex_nodes, vertex_weights
        )
        
        stats["4A_Apply_Deformation"] += time.time() - t_sub
        
        # B. 更新 3D-2D 对应关系 (核心瓶颈)
        t_sub = time.time()
        update_correspondences(V_current, model_faces, T_M2W, all_optimization_data, outer_iter, output_dir, mesh_template)
        visualize_per_frame_alignment(
                V_current, 
                model_faces, 
                all_optimization_data, 
                outer_iter, 
                output_dir, 
                T_M2W
            )
        batched_data = prepare_batched_data_for_solver(all_optimization_data,T_M2W)
        stats["4B_Correspondence_Search"] += time.time() - t_sub
        
        # D. 最小二乘求解 (Levenberg-Marquardt / TRF)
        t_sub = time.time()
        # --- [新增可视化] ---
        visualize_contour_matches(V_current, T_M2W, all_optimization_data, outer_iter, output_dir)
        
        # C. 最小二乘优化
        #    注意：将 current_x 作为本次优化的初值
        result = least_squares(
            deformation_graph_residual,
            current_x,
            verbose=1,
            method="trf",
            x_scale="jac",
            # 可以在后期迭代减少 max_nfev 以加快速度
            max_nfev=20 if outer_iter > 0 else 50, 
            args=(
                V0, node_pos, vertex_nodes, vertex_weights,
                graph_edges, batched_data, lambda_reg
            )
        )
        current_x = result.x
        stats["4D_Least_Squares_Solve"] += time.time() - t_sub
        
        print(f"Iteration {outer_iter+1} done in {time.time() - iter_start:.4f}s")
        
        # --- 保存残差逻辑 ---
        final_x = result.x
        # 获取分段残差字典
        res_parts = deformation_graph_residual(
            final_x, V0, node_pos, vertex_nodes, vertex_weights, 
            graph_edges, batched_data, lambda_reg, return_dict=True
        )
        # 准备输出目录
        save_dir = "/home/zjr/Tracker/CoarseModel/results/refine_model/snoopy"
        os.makedirs(save_dir, exist_ok=True)
        res_file = os.path.join(save_dir, f"residuals_iter_{outer_iter}.txt")
        with open(res_file, "w") as f:
            f.write(f"=== Optimization Residuals (Outer Iteration {outer_iter + 1}) ===\n")
            f.write(f"{'Type':<15} | {'Sum of Squares':<18} | {'Mean Abs':<12} | {'Count':<8}\n")
            f.write("-" * 60 + "\n")
            
            total_sum_squares = 0.0
            
            for name, val in res_parts.items():
                # 计算该项残差的平方和 (Total Cost Contribution)
                sum_squares = np.sum(np.square(val))
                mean_abs = np.mean(np.abs(val))
                count = len(val)
                
                total_sum_squares += sum_squares
                
                f.write(f"{name:<15} | {sum_squares:>18.6f} | {mean_abs:>12.6f} | {count:>8}\n")
            
            f.write("-" * 60 + "\n")
            # result.cost 对应的是 0.5 * sum(res**2)，所以这里乘以 2 以对应总平方和
            f.write(f"Calculated Total SS: {total_sum_squares:.6f}\n")
            f.write(f"Solver Reported Cost: {result.cost:.6f} (0.5 * SS)\n")
            f.write(f"Optimality: {result.optimality:.6e}\n")
            
        print(f"Iteration {outer_iter+1} residuals saved to {res_file}")
    # --- 5. 应用最终形变 ---
    # --- 5. 应用最终形变 ---
    t_start = time.time()
    R_list, t_list = unpack_params(current_x)
    V_refined = apply_deformation_graph(
        V0, node_pos, R_list, t_list,
        vertex_nodes, vertex_weights
    )
    stats["5_Final_Apply"] = time.time() - t_start
    
    total_duration = time.time() - total_start

    # --- 打印性能报告 ---
    print("\n" + "="*50)
    print(f"{'STAGE':<30} | {'TIME (s)':<10} | {'%':<5}")
    print("-" * 50)
    for stage, duration in sorted(stats.items()):
        percentage = (duration / total_duration) * 100
        print(f"{stage:<30} | {duration:>10.4f} | {percentage:>5.1f}%")
    print("-" * 50)
    print(f"{'TOTAL DURATION':<30} | {total_duration:>10.4f} | 100%")
    print("="*50 + "\n")
    
    # --- [新增：最终结果可视化] ---
    print(f"--- Generating Final Visualizations to {output_dir} ---")
    # 这里的 T_M2W 是传入函数的初始对齐矩阵
    # 如果优化过程中没有修改 T_M2W，则直接使用
    for data in all_optimization_data:
        # 为了区分中间过程，我们将最终结果存在 'final_results' 目录下
        img_name = os.path.basename(data['image_path'])
        visualize_projection(
            image_path=data["image_path"],
            output_dir=output_dir,
            K=data["K"],
            T_W2C=data["T_w2c"],
            T_M2W=T_M2W, # 最终位姿
            model_vertices=V_refined, # 使用优化后的顶点
            model_faces=model_faces,
            image_name=f"refined_{img_name}",
            root_name="final_results"
        )
    return V_refined

# 在外层迭代中，完成 update_contour_correspondences 后提取数据
def prepare_batched_data_for_solver(all_optimization_data, T_M2W):
    batched_data = {
        'contour': {'v_ids': [], 'T_total': [], 'K': [], 'targets': [], 'grads': []},
        'feature': {'v_ids': [], 'T_total': [], 'K': [], 'targets': []}
    }

    for data in all_optimization_data:
        T_total = data["T_w2c"] @ T_M2W
        K = data["K"]
        
        # 1. 收集轮廓数据
        c_v_ids = data.get('contour_v_ids')
        if c_v_ids is not None and len(c_v_ids) > 0:
            N_c = len(c_v_ids)
            batched_data['contour']['v_ids'].append(c_v_ids)
            batched_data['contour']['targets'].append(data['edge_2d'])
            batched_data['contour']['grads'].append(data['edge_grad'])
            batched_data['contour']['T_total'].append(np.tile(T_total, (N_c, 1, 1))) 
            batched_data['contour']['K'].append(np.tile(K, (N_c, 1, 1)))
            
        # 2. 收集内部特征数据 (LoFTR)
        f_v_ids = data.get('feature_v_ids')
        if f_v_ids is not None and len(f_v_ids) > 0:
            N_f = len(f_v_ids)
            batched_data['feature']['v_ids'].append(f_v_ids)
            batched_data['feature']['targets'].append(data['feature_targets_2d'])
            batched_data['feature']['T_total'].append(np.tile(T_total, (N_f, 1, 1)))
            batched_data['feature']['K'].append(np.tile(K, (N_f, 1, 1)))

    # 合并数组
    for key in ['contour', 'feature']:
        if batched_data[key]['v_ids']:
            batched_data[key] = {k: np.concatenate(v, axis=0) for k, v in batched_data[key].items()}
        else:
            batched_data[key] = None

    return batched_data
    
def deformation_graph_residual(
    x,
    V0,
    node_pos,
    vertex_nodes,
    vertex_weights,
    graph_edges,
    batched_data,
    lambda_reg,
    return_dict=False
):
    R_list, t_list = unpack_params(x)
    residuals = []
    K_nodes = node_pos.shape[0]
    residuals_map = {} # 用于存储分段残差
    
    # 为了批量计算变形，可以先只提取出本轮所有需要的顶点
    # 但由于 Vertex Deformation 依赖 KNN 权重，按帧处理逻辑比较清晰
    
    # 1. 批量变形所有涉及的 3D 顶点 (一次性完成!)
    c_data = batched_data['contour']
    V_sub = V0[c_data['v_ids']]
    vw_sub = vertex_weights[c_data['v_ids']]
    vn_sub = vertex_nodes[c_data['v_ids']]
    V_def = apply_deformation_vectorized(V_sub, node_pos, R_list, t_list, vn_sub, vw_sub)
    
    # 2. 批量 3D->2D 投影
    V_def_h = np.hstack([V_def, np.ones((len(V_def), 1))]) # (N, 4)
    # 批量矩阵乘法 (N, 3, 4) @ (N, 4, 1) -> (N, 3, 1)
    V_c = np.matmul(c_data['T_total'][:, :3, :], V_def_h[..., None]).squeeze(-1)
    z = np.maximum(V_c[:, 2:3], 1e-6)
    proj_contour = np.matmul(c_data['K'][:, :2, :3], V_c[..., None]).squeeze(-1) / z
    
    # 3. 计算残差
    diff = proj_contour - c_data['targets']
    res_contour = np.sum(diff * c_data['grads'], axis=1) # 点到平面距离
    residuals_map['contour'] = res_contour
    
    # ✅ 补上遗漏的这一行
    contour_weight = 1.0
    residuals.append(contour_weight * res_contour)
    # 2. ARAP 正则项 (Smoothness)
    # ... (与原代码保持一致) ...
    idx_k = graph_edges[:, 0]
    idx_l = graph_edges[:, 1]
    g_k, g_l = node_pos[idx_k], node_pos[idx_l]
    vec_kl = (g_l - g_k)[..., None]
    pred = (R_list[idx_k] @ vec_kl).squeeze(-1) + g_k + t_list[idx_k]
    target = g_l + t_list[idx_l]
    reg_res = np.sqrt(lambda_reg) * (pred - target)
    residuals.append(reg_res.ravel())
    residuals_map['smoothness'] = reg_res.ravel()
    
    # ==========================================
    # 2. 内部特征残差 (Point-to-Point Reprojection) - 新增！
    # ==========================================
    if batched_data.get('feature') is not None:
        f_data = batched_data['feature']
        V_sub = V0[f_data['v_ids']]
        vw_sub = vertex_weights[f_data['v_ids']]
        vn_sub = vertex_nodes[f_data['v_ids']]
        
        V_def = apply_deformation_vectorized(V_sub, node_pos, R_list, t_list, vn_sub, vw_sub)
        V_def_h = np.hstack([V_def, np.ones((len(V_def), 1))])
        
        V_c = np.matmul(f_data['T_total'][:, :3, :], V_def_h[..., None]).squeeze(-1)
        z = np.maximum(V_c[:, 2:3], 1e-6)
        proj_feat = np.matmul(f_data['K'][:, :2, :3], V_c[..., None]).squeeze(-1) / z
        
        # 特征点优化：直接优化 2D 坐标的欧氏距离 (x, y 方向都要对齐)
        diff_f = proj_feat - f_data['targets']
        
        # 这里权重可以调高一些，因为特征点通常比轮廓更可靠，拉扯内部几何
        feature_weight = 1.0 
        
        # 注意: least_squares 需要 1D array，所以我们要用 .ravel() 展平 x 和 y 残差
        residuals.append((feature_weight * diff_f).ravel())
        residuals_map['feature'] = (feature_weight * diff_f).ravel()
    # # 3. Prior term
    # # ... (与原代码保持一致) ...
    # prior_weight = 1.0
    # for i in range(K_nodes):
    #     residuals.append(prior_weight * t_list[i])
    #     rot_res = Rotation.from_matrix(R_list[i]).as_rotvec()
    #     residuals.append(prior_weight * rot_res)
        
    # # Anchor term
    # anchor_weight = 1e6
    # residuals.append(anchor_weight * (R_list[0] - np.eye(3)).ravel())
    # residuals.append(anchor_weight * t_list[0])
    
    if return_dict:
        return residuals_map
    # 确保所有部分都是 float64 数组，并拼接
    final_res = []
    # 按照固定顺序拼接，确保雅可比矩阵一致性
    for key in ['contour', 'smoothness', 'feature']:
        if key in residuals_map and residuals_map[key] is not None:
            # 强制转为 1D float64
            final_res.append(np.array(residuals_map[key], dtype=np.float64).ravel())
    
    return np.concatenate(final_res)

    # return np.concatenate(residuals)

def apply_deformation_vectorized(V_subset, node_pos, R_list, t_list, v_nodes, v_weights):
    """
    V_subset: (M, 3) 待变形的顶点子集
    node_pos: (K, 3) 节点位置
    R_list: (K, 3, 3) 所有节点的旋转
    t_list: (K, 3) 所有节点的位移
    v_nodes: (M, knn) 每个顶点关联的节点索引
    v_weights: (M, knn) 每个顶点关联的节点权重
    """
    M = V_subset.shape[0]
    knn = v_nodes.shape[1]
    
    # 提取关联节点的参数 (M, knn, 3, 3) 和 (M, knn, 3)
    R_selected = R_list[v_nodes] 
    t_selected = t_list[v_nodes]
    g_selected = node_pos[v_nodes]
    
    # 计算 (v - g): (M, 1, 3) -> (M, knn, 3, 1)
    v_minus_g = (V_subset[:, None, :] - g_selected)[..., None]
    
    # 核心公式: R * (v - g) + g + t
    # 使用 matmul: (M, knn, 3, 3) @ (M, knn, 3, 1) -> (M, knn, 3, 1)
    deformed_parts = np.matmul(R_selected, v_minus_g).squeeze(-1) + g_selected + t_selected
    
    # 加权求和: (M, knn, 3) * (M, knn, 1) -> (M, 3)
    V_deformed = np.sum(deformed_parts * v_weights[..., None], axis=1)
    
    return V_deformed

def apply_deformation_graph(
    V0, node_pos,
    R_list, t_list,
    vertex_nodes, vertex_weights
):
    V_refined = apply_deformation_vectorized(
        V0, 
        node_pos, 
        R_list, 
        t_list, 
        vertex_nodes, 
        vertex_weights
    )
    
    return V_refined

def unpack_params(x):
    R_list, t_list = [], []
    for i in range(0, len(x), 6):
        rvec = x[i:i+3]
        t = x[i+3:i+6]
        R_list.append(axis_angle_to_rotmat(rvec))
        t_list.append(t)
    R_list = np.stack(R_list, axis=0)   # (K, 3, 3)
    t_list = np.stack(t_list, axis=0)   # (K, 3)
    return R_list, t_list

def to_homo(v):
    return np.append(v, 1.0)

def project_point_camera(K, X):
    x = X[0] / X[2]
    y = X[1] / X[2]
    u = K[0,0] * x + K[0,2]
    v = K[1,1] * y + K[1,2]
    return np.array([u, v])

def farthest_point_sampling(points, num_samples):
    """
    points: (N,3)
    return: (num_samples,) indices
    """
    N = points.shape[0]
    indices = np.zeros(num_samples, dtype=np.int64)

    # 随机选一个起点
    indices[0] = np.random.randint(N)
    dist = np.full(N, np.inf)

    for i in range(1, num_samples):
        p = points[indices[i - 1]]
        dist = np.minimum(dist, np.linalg.norm(points - p, axis=1))
        indices[i] = np.argmax(dist)

    return indices

def build_knn_graph(node_pos, k=4):
    """
    node_pos: (K,3)
    return: list of (i,j)
    """
    tree = cKDTree(node_pos)
    _, nn = tree.query(node_pos, k=k+1)

    edges = set()
    for i in range(node_pos.shape[0]):
        for j in nn[i][1:]:
            edges.add((i, j))
            edges.add((j, i))  # 无向图

    return list(edges)

def compute_vertex_node_weights(vertices, node_pos, k=4):
    """
    vertices: (N,3)
    node_pos: (K,3)

    return:
        vertex_nodes: (N,k) int
        vertex_weights: (N,k) float
    """
    tree = cKDTree(node_pos)
    dists, idxs = tree.query(vertices, k=k)

    d_max = dists[:, -1][:, None] + 1e-6 
    
    # ED 论文公式: weight = (1 - dist / d_max)^2
    # 这样保证在影响半径边缘权重平滑降为 0
    weights = (1.0 - dists / d_max) ** 2
    weights[dists > d_max] = 0 # 理论上不会发生，因为我们是用 knn 查的，但为了保险
    
    # 3. 归一化 (Partition of Unity) !!! 这一步如果不做，就会出现裂痕
    w_sum = np.sum(weights, axis=1, keepdims=True)
    weights = weights / (w_sum + 1e-8)
    
    return idxs, weights

