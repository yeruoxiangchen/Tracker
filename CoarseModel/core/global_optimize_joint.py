# core/global_optimize.py
import os
import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from datetime import datetime

from sklearn.decomposition import PCA

from core.util import ( 
                       axis_angle_to_rotmat, project_point, 
                       read_colmap_points3D, read_colmap_images_txt
)
from collections import defaultdict

from contextlib import redirect_stdout

def project_points_batch(P_W, K, T_w2c):
    """
    批量投影 3D 点到像素坐标

    Args:
        P_W: (N, 3) world points
        K: (3, 3)
        T_w2c: (4, 4)

    Returns:
        uv: (N, 2) 像素坐标
        z: (N,) 相机坐标系深度
    """
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
    """
    高性能版本：从 COLMAP 点云中筛选落在至少 min_observations 个 Mask 内的 3D 点
    """

    print("\n--- Collecting object 3D points (FAST VERSION) ---")

    points3d = read_colmap_points3D(points3d_path)
    if not points3d:
        return np.array([])

    # -------------------------------------------------
    # 1. image_id -> 相机数据
    # -------------------------------------------------
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

    # -------------------------------------------------
    # 2. image_id -> 该图像观测到的 3D 点
    # -------------------------------------------------
    image_to_points = defaultdict(list)
    points_xyz = {}

    for pid, pdata in points3d.items():
        points_xyz[pid] = pdata['xyz']
        track = pdata['track_list']
        for i in range(0, len(track), 2):
            image_id = track[i]
            if image_id in id_to_data:
                image_to_points[image_id].append(pid)

    # -------------------------------------------------
    # 3. Mask 缓存
    # -------------------------------------------------
    mask_cache = {}

    def load_mask(path):
        if path not in mask_cache:
            if not os.path.exists(path):
                mask_cache[path] = None
            else:
                mask_cache[path] = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        return mask_cache[path]

    # -------------------------------------------------
    # 4. 统计每个 3D 点在 Mask 内的观测次数
    # -------------------------------------------------
    point_in_mask_count = defaultdict(int)

    for image_id, pids in image_to_points.items():

        data = id_to_data[image_id]
        mask = load_mask(data['mask_path'])
        if mask is None:
            continue

        K = data['K']
        T_w2c = data['T_w2c']
        h, w = mask.shape

        # 取出该 image 的所有 3D 点
        P_W = np.stack([points_xyz[pid] for pid in pids], axis=0)

        # 批量投影
        uv, z = project_points_batch(P_W, K, T_w2c)

        # 深度 & 图像范围裁剪
        valid = (
            (z > 0) &
            (uv[:, 0] >= 0) & (uv[:, 0] < w) &
            (uv[:, 1] >= 0) & (uv[:, 1] < h)
        )

        if not np.any(valid):
            continue

        uv = uv[valid].astype(np.int32)
        valid_pids = np.array(pids)[valid]

        # Mask 查询（向量化）
        mask_values = mask[uv[:, 1], uv[:, 0]] > 0

        for pid in valid_pids[mask_values]:
            point_in_mask_count[pid] += 1

    # -------------------------------------------------
    # 5. 筛选满足最小观测次数的 3D 点
    # -------------------------------------------------
    object_points_W = [
        points_xyz[pid]
        for pid, cnt in point_in_mask_count.items()
        if cnt >= min_observations
    ]

    print(f"Selected {len(object_points_W)} object 3D points.")

    return np.array(object_points_W, dtype=np.float64)

def get_obb_extents(points):
    """
    使用 PCA 计算点云的有向包围盒（OBB）的三个轴向长度。
    """
    if points.shape[0] < 3:
        return np.array([0, 0, 0])
    
    # 1. 初始化 PCA
    pca = PCA(n_components=3)
    # 2. 对点云进行坐标变换，转换到主成分坐标系（特征向量空间）
    points_rotated = pca.fit_transform(points)
    
    # 3. 在这个旋转后的局部坐标系下计算 AABB
    # 这等价于原始空间中的 OBB
    min_p = np.min(points_rotated, axis=0)
    max_p = np.max(points_rotated, axis=0)
    
    # 返回三个主轴方向的长度，并排序（防止轴序不一致）
    extents = max_p - min_p
    return np.sort(extents)

def calculate_initial_TMW_from_obb(model_vertices, X_W_obj):
    """
    基于有向包围盒 (OBB) 计算初始尺度 s_init。
    """
    if model_vertices.size == 0 or X_W_obj.size == 0:
        raise ValueError("Model vertices or object 3D points are empty.")

    print("\n--- Calculating Initial T_M2W using OBB Alignment ---")
    
    # 获取模型在 OBB 意义下的三个轴长
    L_M_obb = get_obb_extents(model_vertices)
    
    # 获取观测点云在 OBB 意义下的三个轴长
    L_W_obb = get_obb_extents(X_W_obj)
    
    # 计算三个维度上的缩放比例
    epsilon = 1e-6
    scale_ratios = L_W_obb / (L_M_obb + epsilon)
    
    s_xyz_init = np.maximum(scale_ratios, 1e-6)
    # 剔除极小值的影响（如果有维度被压扁了）
    print(f"Model Principal Extents: {L_M_obb}")
    print(f"World Principal Extents: {L_W_obb}")
    print(f"Initial 3-Axis Scales (sx, sy, sz): {s_xyz_init}")
    
    return s_xyz_init # 返回 np.array([sx, sy, sz])

def _compute_reprojection_residuals(scales, rvec, t, all_data):
    """
    核心的重投影残差计算逻辑，供交替优化调用。
    """
    from scipy.spatial.transform import Rotation
    R = Rotation.from_rotvec(rvec).as_matrix()
    residuals = []

    Z_PENALTY_WEIGHT = 500.0
    EPS = 1e-4
    
    for d in all_data:
        X = d["X_3d"]
        q = d["q_2d"]
        K = d["K"]
        T_w2c = d["T_w2c"]
        
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
            # 连续梯度惩罚
            penalty = Z_PENALTY_WEIGHT * (EPS - Z[mask_behind])
            res[mask_behind, 0] = penalty
            res[mask_behind, 1] = penalty

        residuals.append(res.ravel())

    return np.concatenate(residuals)

def error_optimize_rt(rt_params, fixed_scales, all_data):
    """固定 scales，仅优化 rvec 和 t (6 DOF)"""
    rvec = rt_params[0:3]
    t = rt_params[3:6]
    return _compute_reprojection_residuals(fixed_scales, rvec, t, all_data)

def error_optimize_s(s_params, fixed_rt, all_data):
    """固定 rvec 和 t，仅优化 scales (3 DOF)"""
    rvec = fixed_rt[0:3]
    t = fixed_rt[3:6]
    return _compute_reprojection_residuals(s_params, rvec, t, all_data)


def optimize_global_pose(
    all_optimization_data,
    all_corresps_data,
    model_vertices,
    model_faces,
    output_dir,
    colmap_dir,
    mask_dir,
    init_frame_id=0,
    num_iterations=3 # 新增参数：交替迭代的次数
):
    """
    Returns:
        T_M2W_final (4x4)
        result (scipy OptimizeResult)
    """

    # --- 初始化 ---
    best_pose = all_corresps_data[init_frame_id].best_pose
    R_m2c = best_pose["R_m2c"]
    t_m2c = best_pose["t_m2c"]

    T_m2c = np.eye(4)
    T_m2c[:3, :3] = R_m2c
    T_m2c[:3, 3] = t_m2c

    T_w2c = all_optimization_data[init_frame_id]["T_w2c"]
    T_c2w = np.linalg.inv(T_w2c)
    T_m2w_init = T_c2w @ T_m2c

    R_init = T_m2w_init[:3, :3]
    t_init = T_m2w_init[:3, 3]

    # --- scale 初始化 ---
    colmap_points3D_path = os.path.join(colmap_dir, 'points3D.txt')
    images_txt = os.path.join(colmap_dir, 'images.txt')
    images_colmap = read_colmap_images_txt(images_txt) # 假设返回字典
    X_W_obj = get_object_3d_points(
        points3d_path=os.path.join(colmap_dir, "points3D.txt"),
        all_optimization_data=all_optimization_data,
        mask_dir=mask_dir,
        images_colmap=images_colmap,
        min_observations=3,
    )
    import open3d as o3d
    pcd_filter = o3d.geometry.PointCloud()
    pcd_filter.points = o3d.utility.Vector3dVector(X_W_obj)
    # 这里的参数要严一点，剔除掉离群的点
    pcd_filter, _ = pcd_filter.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.0)
    X_W_obj_clean = np.asarray(pcd_filter.points)
    s_init = calculate_initial_TMW_from_obb(model_vertices, X_W_obj_clean)

    rvec_init = Rotation.from_matrix(R_init).as_rotvec()
    current_scales = s_init.copy()
    current_rt = np.concatenate([rvec_init, t_init])
    
    # 初始 RMSE
    res0 = _compute_reprojection_residuals(current_scales, current_rt[0:3], current_rt[3:6], all_optimization_data)
    print("Initial reprojection RMSE:", np.sqrt(np.mean(res0**2)))
    
    date_str = datetime.now().strftime("%Y-%m-%d")
    log_filename = f"least_squares_log_{date_str}.txt"
    log_filepath = os.path.join(output_dir, log_filename)
    
    # --- 准备可视化数据 ---
    # 选取一帧作为可视化参考（通常选初始帧）
    vis_data = all_optimization_data[init_frame_id]
    
    # --- 交替迭代优化 (Alternating Optimization) ---
    last_result = None
    
    with open(log_filepath, 'w') as f:
        with redirect_stdout(f):
            print("Initial reprojection RMSE:", np.sqrt(np.mean(res0**2)))
            
            for iteration in range(num_iterations):
                print(f"\n=== Iteration {iteration + 1}/{num_iterations} ===")
                
                # Step 1: 优化 RT (固定 Scales)
                print(f"--- Step 1: Optimizing R & t (Scales fixed at {current_scales}) ---")
                res_rt = least_squares(
                    fun=error_optimize_rt, 
                    x0=current_rt, 
                    args=(current_scales, all_optimization_data), 
                    method='trf', 
                    verbose=2, 
                    loss='soft_l1'
                )
                current_rt = res_rt.x  # 更新 rt
                # Step 1 后的可视化与矩阵输出
                curr_R1 = Rotation.from_rotvec(current_rt[0:3]).as_matrix()
                curr_t1 = current_rt[3:6]
                T_M2W_step1 = np.eye(4, dtype=np.float32)
                T_M2W_step1[:3, :3] = curr_R1 @ np.diag(current_scales)
                T_M2W_step1[:3, 3] = curr_t1

                print(f"\n[Iter {iteration+1} Step 1 (RT) Finished]")
                print("T_M2W after RT optimization:\n", T_M2W_step1)
                
                from script.vis_util import visualize_projection
                visualize_projection(
                    image_path=vis_data["image_path"],
                    output_dir=output_dir,
                    K=vis_data["K"],
                    T_W2C=vis_data["T_w2c"],
                    T_M2W=T_M2W_step1,
                    model_vertices=model_vertices,
                    model_faces=model_faces,
                    root_name="iter",
                    image_name=f"iter_{iteration+1}_step1_rt.jpg" 
                )
                
                # Step 2: 优化 Scale (固定 RT)
                print(f"--- Step 2: Optimizing Scales (R & t fixed) ---")
                lower_bounds_s = np.array([1e-4, 1e-4, 1e-4])
                upper_bounds_s = np.array([np.inf, np.inf, np.inf])
                
                res_s = least_squares(
                    fun=error_optimize_s, 
                    x0=current_scales, 
                    args=(current_rt, all_optimization_data), 
                    method='trf', 
                    bounds=(lower_bounds_s, upper_bounds_s), # 强制正值限制
                    verbose=2, 
                    loss='soft_l1'
                )
                current_scales = res_s.x  # 更新 scales
                last_result = res_s # 记录最后一次优化的结果对象
                
                # 计算本轮结束后的 RMSE
                current_res = _compute_reprojection_residuals(current_scales, current_rt[0:3], current_rt[3:6], all_optimization_data)
                print(f"RMSE after Iteration {iteration + 1}:", np.sqrt(np.mean(current_res**2)))
                
                # --- 构造当前 T_M2W 用于可视化 ---
                curr_R2 = Rotation.from_rotvec(current_rt[0:3]).as_matrix()
                curr_t2 = current_rt[3:6]
                T_M2W_step2 = np.eye(4, dtype=np.float32)
                T_M2W_step2[:3, :3] = curr_R2 @ np.diag(current_scales)
                T_M2W_step2[:3, 3] = curr_t2
                print(f"\n[Iter {iteration+1} Step 2 (Scale) Finished]")
                print("T_M2W after Scale optimization:\n", T_M2W_step2)
                # --- 输出可视化结果 (每一轮迭代存一张) ---
                from script.vis_util import visualize_projection # 假设路径在此
                visualize_projection(
                    image_path=vis_data["image_path"],
                    output_dir=output_dir,
                    K=vis_data["K"],
                    T_W2C=vis_data["T_w2c"],
                    T_M2W=T_M2W_step2,
                    model_vertices=model_vertices,
                    model_faces=model_faces,
                    root_name="iter",
                    image_name=f"iter_{iteration+1}_step2_scale.jpg" # 区分文件名
                )
                a = 1 

    # --- 整理输出 ---
    final_scales = current_scales
    final_R = Rotation.from_rotvec(current_rt[0:3]).as_matrix()
    final_t = current_rt[3:6]
    
    T_final = np.eye(4, dtype=np.float32)
    # T = R @ S + t
    T_final[:3, :3] = final_R @ np.diag(final_scales)
    T_final[:3, 3] = final_t

    # 为了兼容你外部可能调用 result.x 获取全量 9 个参数的代码，伪造/拼接一下最终的 x
    if last_result is not None:
        last_result.x = np.concatenate([final_scales, current_rt])

    return T_final, last_result, final_scales
