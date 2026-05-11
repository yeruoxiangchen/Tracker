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

def get_object_3d_points(points3d_path, all_optimization_data, mask_dir, images_colmap, min_observations=3):
    """
    从 COLMAP 点云中筛选出落在至少 min_observations 个 Mask 内部的 3D 点。
    
    Args:
        points3d_path (str): COLMAP points3D 文件路径。
        all_optimization_data (list): 包含 'frame_name', 'K', 'T_w2c' 的列表。
        mask_dir (str): Mask 文件所在目录。
        images_colmap (dict): {image_name: {'id': image_id, ...}}
        min_observations (int): 最小需要的 Mask 内观测次数。
        
    Returns:
        np.ndarray: (N, 3) 形状的物体 3D 点云 (世界坐标系 W)。
    """
    points3d = read_colmap_points3D(points3d_path)
    if not points3d:
        return np.array([])
        
    # 建立查找表：image_id -> 优化数据 (K, T_w2c, mask_path)
    id_to_data = {}
    for entry in all_optimization_data:
        frame_name = entry['frame_name']
        if frame_name in images_colmap:
            image_id = images_colmap[frame_name]['image_id']
            id_to_data[image_id] = {
                'K': entry['K'],
                'T_w2c': entry['T_w2c'],
                'mask_path': os.path.join(mask_dir, frame_name.replace(".jpg", ".png").replace(".JPG", ".png"))
            }

    object_points_W = []
    
    for point_id, point_data in points3d.items():
        P_W = point_data['xyz']
        track_list = point_data['track_list']
        
        in_mask_count = 0
        
        # track_list 是 [image_id, point2D_idx, ...]
        for i in range(0, len(track_list), 2):
            image_id = track_list[i]
            # point2D_idx = track_list[i+1] # 这里的 2D 索引不是像素坐标，忽略
            
            if image_id in id_to_data:
                data = id_to_data[image_id]
                
                # 投影 3D 点到当前图像
                u, v = project_point(P_W, data['K'], data['T_w2c'])
                
                mask_path = data['mask_path']
                if not os.path.exists(mask_path):
                    continue

                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    continue

                h, w = mask.shape
                
                # 检查投影点是否在图像范围内
                if 0 <= u < w and 0 <= v < h:
                    # 检查投影点是否在 Mask 内部 (假设 Mask 是二值的，非零即物体)
                    pixel_value = mask[int(v), int(u)] 
                    if pixel_value > 0:
                        in_mask_count += 1
        
        if in_mask_count >= min_observations:
            object_points_W.append(P_W)
            
    return np.array(object_points_W, dtype=np.float64)

def calculate_initial_TMW_from_bbox(model_vertices, X_W_obj):
    """
    基于轴对齐的闭包（Bounding Box）计算初始尺度 s_init 和 T_M2W。
    
    Args:
        model_vertices (np.ndarray): 模型坐标系下的顶点 (N, 3)。
        X_W_obj (np.ndarray): 世界坐标系下属于物体的 3D 点云 (N', 3)。
        
    Returns:
        tuple: (s_init, R_init, t_init)
    """
    if model_vertices.size == 0 or X_W_obj.size == 0:
        raise ValueError("Model vertices or object 3D points are empty.")

    print("\n--- Calculating Initial T_M2W using BBox Alignment ---")
    
    # --- 1. 计算闭包 ---
    
    # Model BBox (在模型坐标系 M)
    min_M = np.min(model_vertices, axis=0)
    max_M = np.max(model_vertices, axis=0)
    L_M = max_M - min_M
    center_M = (min_M + max_M) / 2.0
    
    # World BBox (在世界坐标系 W)
    min_W = np.min(X_W_obj, axis=0)
    max_W = np.max(X_W_obj, axis=0)
    L_W = max_W - min_W
    center_W = (min_W + max_W) / 2.0
    
    # --- 2. 计算初始尺度 s_init ---
    
    # 计算每个轴的尺度比率 (为避免除以零，添加 epsilon)
    epsilon = 1e-6
    scale_ratios = L_W / (L_M + epsilon)
    
    # 使用平均尺度作为初始尺度 (更鲁棒)
    s_init = np.mean(scale_ratios)
    
    print(f"Model BBox Extents (L_M): {L_M}")
    print(f"World BBox Extents (L_W): {L_W}")
    print(f"Calculated Scale Ratios (L_W/L_M): {scale_ratios}")
    print(f"Initial Scale s_init (Mean): {s_init:.4f}")
    
    return s_init



def global_reprojection_error(params, all_data):
    from scipy.spatial.transform import Rotation

    s = params[0]
    rvec = params[1:4]
    t = params[4:7]

    R = Rotation.from_rotvec(rvec).as_matrix()
    residuals = []

    for d in all_data:
        X = d["X_3d"]
        q = d["q_2d"]
        K = d["K"]
        T_w2c = d["T_w2c"]

        Xw = s * ((R @ X.T).T + t)
        Xw_h = np.hstack([Xw, np.ones((len(Xw), 1))])
        Xc = (T_w2c @ Xw_h.T).T[:, :3]

        Z = Xc[:, 2]
        uv = np.zeros_like(q)

        mask = Z > 1e-6
        uv[mask] = (K @ Xc[mask].T).T[:, :2] / Z[mask, None]
        uv[~mask] = 1e4

        res = (uv - q).reshape(-1)
        residuals.append(res)

    return np.concatenate(residuals)


def optimize_global_pose(
    all_optimization_data,
    all_corresps_data,
    model_vertices,
    output_dir,
    colmap_dir,
    mask_dir,
    init_frame_id=0,
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
    
    s_init = calculate_initial_TMW_from_bbox(model_vertices, X_W_obj)

    rvec_init = Rotation.from_matrix(R_init).as_rotvec()
    initial_params = np.array([s_init, *rvec_init, *t_init], dtype=np.float64)
    
    def check_parameter_sensitivity(fun, params, eps=1e-6):
        base = fun(params, all_optimization_data)
        base_norm = np.linalg.norm(base)
        print("base_norm:", base_norm)
        for i in range(len(params)):
            p = params.copy()
            p[i] += eps
            r = fun(p, all_optimization_data)
            if np.isnan(r).any() or np.isinf(r).any():
                print(f"param {i}: yields NaN/Inf")
                continue
            delta = np.linalg.norm(r) - base_norm
            print(f"param {i}: norm change (eps={eps}): {delta:.6e}")
        return base_norm
    _ = check_parameter_sensitivity(global_reprojection_error, initial_params, eps=1e-6)
    
    res0 = global_reprojection_error(initial_params, all_optimization_data)
    print("Initial reprojection RMSE:", np.sqrt(np.mean(res0**2)))
    date_str = datetime.now().strftime("%Y-%m-%d")
    log_filename = f"least_squares_log_{date_str}.txt"
    log_filepath = os.path.join(output_dir, log_filename)
    
    # --- 优化 ---
    from contextlib import redirect_stdout
    from scipy.optimize import least_squares
    with open(log_filepath, 'w') as f:
        with redirect_stdout(f):
            # 捕获 initial RMSE 打印
            print("Initial reprojection RMSE:", np.sqrt(np.mean(res0**2)))
            # 调用 least_squares，其 verbose=2 的输出将被捕获
            result = least_squares(
                fun=global_reprojection_error, 
                x0=initial_params, 
                args=(all_optimization_data,), 
                method='trf', 
                verbose=2  # 这个输出会被重定向到文件
                # loss='soft_l1'
            )

    s = result.x[0]
    R = axis_angle_to_rotmat(result.x[1:4])
    t = result.x[4:7]
    
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = s * R
    T[:3, 3] = s * t

    return T, result, s
