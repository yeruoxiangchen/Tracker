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

    colmap_points3D_path = os.path.join(colmap_dir, 'points3D.txt')
    images_txt = os.path.join(colmap_dir, 'images.txt')
    images_colmap = read_colmap_images_txt(images_txt)
    
    X_W_obj = get_object_3d_points(
        points3d_path=colmap_points3D_path,
        all_optimization_data=all_optimization_data,
        mask_dir=mask_dir,
        images_colmap=images_colmap,
        min_observations=3,
    )
    
    import open3d as o3d
    pcd_filter = o3d.geometry.PointCloud()
    pcd_filter.points = o3d.utility.Vector3dVector(X_W_obj)
    pcd_filter, _ = pcd_filter.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.0)
    X_W_obj_clean = np.asarray(pcd_filter.points)
    
    s_init = calculate_initial_scales_from_median_dist(model_vertices, X_W_obj_clean)
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
    log_filename = f"3stage_least_squares_log_{date_str}.txt"
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

            visualize_projection(
                image_path=all_optimization_data[init_frame_id]["image_path"],
                output_dir=output_dir,
                K=all_optimization_data[init_frame_id]["K"],
                T_W2C=all_optimization_data[init_frame_id]["T_w2c"],
                T_M2W=T_stage1,
                model_vertices=model_vertices,
                model_faces=model_faces,
                root_name="3stage",
                image_name="stage1_rt.jpg"
            )
            # ==========================================================
            # STAGE 2: 联合优化 (Scale + RT) - 带有较强 3D 约束
            # 这一步是为了让 Scale 在位姿正确的前提下开始介入
            # ==========================================================
            print("\n=== Stage 2: Joint Optimization (Scale + RT) ===")
            params_s2 = np.concatenate([s_init, rvec_after_s1, t_after_s1])
            
            res_stage2 = least_squares(
                fun=error_optimize_joint, 
                x0=params_s2, 
                args=(all_optimization_data, s_init, lambda_3d_strong), 
                method='trf',
                bounds=(np.array([1e-4]*3 + [-np.inf]*6), np.array([np.inf]*9)),
                verbose=2, 
                loss='soft_l1'
            )
            
            s_after_s2 = res_stage2.x[0:3]
            r_after_s2 = res_stage2.x[3:6]
            t_after_s2 = res_stage2.x[6:9]
            print(f"Stage 2 Finished. Scale at {s_after_s2}")
            # ===== Stage2 Visualization =====
            R_s2 = Rotation.from_rotvec(r_after_s2).as_matrix()

            T_stage2 = np.eye(4, dtype=np.float32)
            T_stage2[:3, :3] = R_s2 @ np.diag(s_after_s2)
            T_stage2[:3, 3] = t_after_s2

            visualize_projection(
                image_path=all_optimization_data[init_frame_id]["image_path"],
                output_dir=output_dir,
                K=all_optimization_data[init_frame_id]["K"],
                T_W2C=all_optimization_data[init_frame_id]["T_w2c"],
                T_M2W=T_stage2,
                model_vertices=model_vertices,
                model_faces=model_faces,
                root_name="3stage",
                image_name="stage2_joint.jpg"
            )
            # ==========================================================
            # STAGE 3: 固定 RT, 最终微调 Scale (去掉 3D 约束)
            # ==========================================================
            print("\n=== Stage 3: Fix RT, Final Scale Refinement ===")
            res_stage3 = least_squares(
                fun=error_optimize_scale_only, 
                x0=s_after_s2, 
                args=(r_after_s2, t_after_s2, all_optimization_data, s_init, 0.0), # lambda=0
                method='trf', 
                bounds=(1e-4, np.inf),
                verbose=2, 
                loss='soft_l1'
            )
    
    final_s = res_stage3.x
    final_r = r_after_s2
    final_t = t_after_s2
    
    print(f"Final Optimized Scales: {final_s}")
    
    # --- 构造最终矩阵并可视化 ---
    final_R = Rotation.from_rotvec(final_r).as_matrix()
    T_final = np.eye(4, dtype=np.float32)
    T_final[:3, :3] = final_R @ np.diag(final_s)
    T_final[:3, 3] = final_t
            
    # --- 输出可视化结果 ---
    from script.vis_util import visualize_projection
    visualize_projection(
        image_path=all_optimization_data[init_frame_id]["image_path"],
        output_dir=output_dir,
        K=all_optimization_data[init_frame_id]["K"],
        T_W2C=all_optimization_data[init_frame_id]["T_w2c"],
        T_M2W=T_final,
        model_vertices=model_vertices,
        model_faces=model_faces,
        root_name="3stage",
        image_name="3stage_optimization_final.jpg" 
    )

    return T_final, res_stage3, final_s