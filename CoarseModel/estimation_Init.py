#!/usr/bin/env python3

"""Infers pose from objects."""


import sys
REPO_PATH = "/home/zjr/Tracker/CoarseModel"
BOP_TOOLKIT_PATH = "/home/zjr/Tracker/CoarseModel/external/bop_toolkit"

sys.path.insert(0, REPO_PATH)
sys.path.insert(0, BOP_TOOLKIT_PATH)  # 添加这一行

import os
import cv2
import numpy as np
from utils import logging
from tqdm import tqdm
import pickle

from core.config import AppConfig, InferOpts
from core.corresp import extract_correspondences, CorrespondenceData

logger: logging.Logger = logging.get_logger()


def read_colmap_images_txt(path):
    """Parse COLMAP images.txt (text export). Return dict image_name -> dict with qw,qx,qy,qz, tx,ty,tz."""
    images = {}
    with open(path, 'r') as f:
        for line in f:
            if line.startswith('#') or line.strip() == '':
                continue
            parts = line.strip().split()
            if len(parts) < 9:
                continue
            # image line: IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
            img_id = int(parts[0])
            qw, qx, qy, qz = map(float, parts[1:5])
            tx, ty, tz = map(float, parts[5:8])
            cam_id = int(parts[8])
            name = parts[9]
            images[name] = {
                'image_id': img_id,
                'qvec': np.array([qw, qx, qy, qz], dtype=np.float64),
                'tvec': np.array([tx, ty, tz], dtype=np.float64),
                'cam_id': cam_id,
                'name': name
            }
            # skip the following POINTS2D[] line (one line after each image)
            _ = next(f, None)
    return images

def read_colmap_cameras_txt(path):
    """
    Parse COLMAP cameras.txt
    Returns dict: camera_id -> dict with model, width, height, params
    """
    cameras = {}
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or line == "":
                continue

            parts = line.split()
            cam_id = int(parts[0])
            model = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            params = np.array(list(map(float, parts[4:])), dtype=np.float32)

            cameras[cam_id] = {
                "model": model,
                "width": width,
                "height": height,
                "params": params,
            }
    return cameras

def colmap_camera_to_K(cam):
    """
    Build intrinsic matrix K from COLMAP camera dict
    """
    model = cam["model"]
    params = cam["params"]
    if model == "SIMPLE_RADIAL":
        f, cx, cy, _k = params
        fx = fy = f
    elif model == "PINHOLE":
        fx, fy, cx, cy = params
    elif model == "SIMPLE_PINHOLE":
        f, cx, cy = params
        fx = fy = f
    else:
        raise ValueError(f"Unsupported camera model: {model}")
    K = np.array([
        [fx, 0,  cx],
        [0,  fy, cy],
        [0,  0,  1]
    ], dtype=np.float32)
    return K

def read_colmap_points3D(points3d_path):
    """
    读取 COLMAP 的 points3D 文件。
    
    Args:
        points3d_path (str): points3D.txt 或 points3D.bin 文件的路径。
        
    Returns:
        dict: {point_id: {'xyz': np.ndarray, 'track_list': np.ndarray}}
              其中 track_list 是 [image_id, point2D_idx, image_id, point2D_idx, ...]
    """
    points3d = {}
    
    if points3d_path.endswith(".txt"):
        print(f"Reading COLMAP points3D from: {points3d_path} (TXT format)")
        try:
            with open(points3d_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('#'):
                        continue
                    
                    # 格式: ID, X, Y, Z, R, G, B, ERROR, TRACK...
                    parts = line.split()
                    if len(parts) < 8:
                        continue
                        
                    point_id = int(parts[0])
                    xyz = np.array([float(parts[1]), float(parts[2]), float(parts[3])], dtype=np.float64)
                    
                    # 忽略颜色和误差，解析 track 列表
                    # track 从第 8 个元素开始
                    track_data = np.array([int(x) for x in parts[8:]], dtype=np.int64)
                    
                    points3d[point_id] = {
                        'xyz': xyz,
                        # track_list 格式: [image_id, point2D_idx, image_id, point2D_idx, ...]
                        'track_list': track_data
                    }
            return points3d
            
        except FileNotFoundError:
            print(f"Error: points3D file not found at {points3d_path}")
            return {}
        except Exception as e:
            print(f"Error reading points3D.txt: {e}")
            return {}

    elif points3d_path.endswith(".bin"):
        # TODO: 对于 .bin 文件，通常需要使用 pycolmap 或自定义的二进制读取器。
        # 这里仅作占位符，假定您有外部工具或自己实现。
        print(f"Attempting to read COLMAP points3D from: {points3d_path} (BIN format) - Requires external library (e.g., pycolmap).")
        # 暂时返回空，或调用您的 pycolmap 实用程序
        return {} 
        
    else:
        print("Unsupported points3D file format.")
        return {}
    
def project_points(K, X_c):
        # 辅助函数：将相机坐标系下的 3D 点投影到 2D 图像平面
    # X_c: (N, 3) 
    # 归一化坐标
    x_norm = X_c[:, 0] / X_c[:, 2]
    y_norm = X_c[:, 1] / X_c[:, 2]
    
    # 像素坐标
    u = K[0, 0] * x_norm + K[0, 2]
    v = K[1, 1] * y_norm + K[1, 2]
    
    return np.stack([u, v], axis=1) # (N, 2)
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


def quat_to_rotmat(q):
    # q = [qw, qx, qy, qz]
    qw, qx, qy, qz = q
    R = np.array([
        [1-2*(qy**2+qz**2),     2*(qx*qy- qz*qw),     2*(qx*qz+ qy*qw)],
        [2*(qx*qy+ qz*qw),   1-2*(qx**2+qz**2),     2*(qy*qz- qx*qw)],
        [2*(qx*qz- qy*qw),     2*(qy*qz+ qx*qw),   1-2*(qx**2+qy**2)]
    ], dtype=np.float64)
    return R




def skew(v):
    # 辅助函数：计算反对称矩阵，用于李代数到李群的指数映射
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])

def axis_angle_to_rotmat(axis_angle):
    # 辅助函数：将轴角（3维向量）转换为旋转矩阵
    angle = np.linalg.norm(axis_angle)
    if angle < 1e-6:
        return np.eye(3)
    axis = axis_angle / angle
    K = skew(axis)
    # 罗德里格斯公式
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    return R

def project_point(P_W, K, T_W2C):
    """
    将世界坐标系下的点 P_W 投影到相机图像平面。
    
    Args:
        P_W (np.ndarray): 世界坐标下的3D点 (3,)
        K (np.ndarray): 相机内参矩阵 (3, 3)
        T_W2C (np.ndarray): 世界到相机姿态矩阵 (4, 4)
        
    Returns:
        np.ndarray: 图像平面上的2D像素坐标 (2,) (u, v)
    """
    P_W_homo = np.append(P_W, 1.0) # 转换为齐次坐标 (4,)
    
    # 相机坐标系下的点 P_C = T_W2C @ P_W_homo
    P_C_homo = T_W2C @ P_W_homo
    P_C = P_C_homo[:3] / P_C_homo[3] # 转换为非齐次坐标 (3,)
    
    # 投影到图像平面 (u, v, w) = K @ P_C
    P_img_homo = K @ P_C
    
    # 归一化 (u, v)
    u = P_img_homo[0] / P_img_homo[2]
    v = P_img_homo[1] / P_img_homo[2]
    
    return np.array([u, v])


def global_reprojection_error(params, all_optimization_data):
    """
    params: [s, rx, ry, rz, tx, ty, tz]
    all_optimization_data: list of dicts with keys ['X_3d', 'q_2d', 'K', 'T_w2c']
    """
    import numpy as np
    from scipy.spatial.transform import Rotation

    s = params[0]
    r_vec = params[1:4]
    t = params[4:7]

    R_delta = Rotation.from_rotvec(r_vec).as_matrix()  # 3x3

    residuals = []

    for data in all_optimization_data:
        X_m = data['X_3d']  # N x 3
        q_2d = data['q_2d']  # N x 2
        K = data['K']         # 3 x 3
        T_w2c = data['T_w2c'] # 4 x 4

        # 模型到世界坐标（含尺度）
        X_W = (R_delta @ X_m.T).T + t  # N x 3
        X_W *= s

        # 转换到相机坐标
        X_W_h = np.hstack([X_W, np.ones((X_W.shape[0], 1), dtype=np.float32)])  # N x 4
        X_C_h = (T_w2c @ X_W_h.T).T
        X_C = X_C_h[:, :3]
        
        # # 投影到像素坐标
        uv = (K @ X_C.T).T
        uv = uv[:, :2] / uv[:, 2:3]

        eps = 1e-6
        Z = X_C[:, 2].copy()
        mask_in_front = Z > eps
        mask_behind = ~mask_in_front # 在后面或深度为 0 的点

        # 4. 投影计算
        uv = np.zeros((X_C.shape[0], 2), dtype=np.float32)
        
        if np.any(mask_in_front):
            X_C_valid = X_C[mask_in_front]
            # K @ X_C_valid.T -> 3 x N_valid
            uv_valid_h = (K @ X_C_valid.T).T 
            uv[mask_in_front] = uv_valid_h[:, :2] / uv_valid_h[:, 2:3]

        # 5. 残差计算 (关键修改区域)
        res = np.zeros((X_C.shape[0], 2), dtype=np.float32)
        
        # A. 对于在前面的点：计算实际重投影误差
        res[mask_in_front] = uv[mask_in_front] - q_2d[mask_in_front]
        
        # B. 对于在后面的点：施加固定的大惩罚 (例如 1000 像素误差)
        if np.any(mask_behind):
             # 为每个点（x, y）设置 [OUTLIER_PENALTY, OUTLIER_PENALTY]
            res[mask_behind] = np.array([10000.0, 10000.0])
            
        res = res.reshape(-1)
        residuals.append(res)

    return np.concatenate(residuals)

# --- 主函数 ---

def main() -> None:
    
    # 路径配置
    base_dir = "/home/zjr/Tracker/CoarseModel/datasets/wogua"
    opts_path = "/home/zjr/Tracker/CoarseModel/configs/wogua.json"
    cameras_txt = os.path.join(colmap_dir, "cameras.txt")
    rgb_dir = os.path.join(base_dir, "rgb")
    mask_dir = os.path.join(base_dir, "masks")
    colmap_dir = os.path.join(base_dir, "sparse/0")
    model_path = AppConfig.MODEL_PATH
    data_cache_path = os.path.join(os.path.dirname(opts_path), "cached_optimization_data.pkl")
    output_dir = os.path.join(
        AppConfig.OUTPUT_BASE_DIR, "refine_model", "wogua"
    )
    output_vis_dir = output_dir
    save_dir = output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    opts = InferOpts(
        version="v1",
        object_dataset="wogua",
        object_lids=[1],
        crop=True,
        crop_size=(420, 420),
        debug=True,
        save_vis=True,
    )
    
    # --- 阶段 1: 数据预处理 (Infer / 提取对应点) ---
    cameras_colmap = read_colmap_cameras_txt(cameras_txt)
    
    all_corresps_data = [] 
    all_optimization_data = [] 
    try:
        all_optimization_data, all_corresps_data = load_optimization_data(data_cache_path)
        logger.info("Loaded cached optimization data.")
    except (FileNotFoundError, EOFError, pickle.UnpicklingError, IsADirectoryError):
        logger.info("Cache file not found or corrupted. Starting data preprocessing...")
             
        images_txt = os.path.join(colmap_dir, 'images.txt')
        images_colmap = read_colmap_images_txt(images_txt) # 假设返回字典

        rgb_files = sorted([f for f in os.listdir(rgb_dir) if f.endswith((".jpg", ".png"))])
        mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith(".png")])

        for i, (rgb_file, mask_file) in enumerate(zip(rgb_files, mask_files)):
            frame_name = rgb_file
            image_path = os.path.join(rgb_dir, rgb_file)
            mask_path = os.path.join(mask_dir, mask_file)
            
            if frame_name not in images_colmap:
                logger.warn(f"Frame {frame_name} not found in COLMAP data. Skipping.")
                continue
            
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转为 RGB
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                 mask = np.zeros(image.shape[:2], dtype=np.uint8)

            # camK = np.array([[1087.8726353191628, 0, 360], [0, 1087.8726353191628, 640], [0, 0, 1]], dtype=np.float32)
            img_entry = images_colmap[frame_name]
            cam_id = img_entry["cam_id"]
            cam = cameras_colmap[cam_id]
            camK = colmap_camera_to_K(cam)
            
            corresp_data = extract_correspondences(
                image_raw=image,
                mask_raw=mask,
                cam_k=camK,
                frame_name=frame_name,
                opts=opts,
            )
            if corresp_data is None:
                continue
            
            all_corresps_data.append(corresp_data)
            
            R_colmap = quat_to_rotmat(img_entry['qvec'])  # W -> C Rotation
            t_colmap = img_entry['tvec'].astype(np.float32) # W -> C Translation
            
            # T_w2c = [R_colmap | t_colmap]
            T_w2c = np.eye(4, dtype=np.float32)
            T_w2c[:3, :3] = R_colmap
            T_w2c[:3, 3] = t_colmap 
            
            os.makedirs(save_dir, exist_ok=True)
            R_m2c = corresp_data.best_pose["R_m2c"]
            t_m2c = corresp_data.best_pose["t_m2c"]
                
            current_data_entry = {
                'frame_name': frame_name,
                'K': camK.astype(np.float32), 
                'T_w2c': T_w2c, 
                'q_2d': corresp_data.query_2d_pts.astype(np.float32), 
                'X_3d': corresp_data.model_3d_pts.astype(np.float32), 
                'X_ids': corresp_data.model_feat_ids,
                'image_path': image_path 
            }
            
            all_optimization_data.append(current_data_entry)

        if all_optimization_data:
            data_to_save = {'optimization_data': all_optimization_data, 'corresps_data': all_corresps_data}
            save_optimization_data(data_to_save, data_cache_path)

    if not all_optimization_data:
        logger.error("No valid optimization data collected or loaded. Aborting.")
        return
    
    model_vertices, model_faces = load_ply_vertices_faces(model_path)
    if model_vertices.size == 0:
        logger.error("Model vertices could not be loaded. Aborting.")
        return
            
    # # 1. 初始化 T_M2W
    first_data = all_optimization_data[0]

    R_m2c = all_corresps_data[10].best_pose["R_m2c"]
    t_m2c = all_corresps_data[10].best_pose["t_m2c"]
    T_m2c = np.eye(4)
    T_m2c[:3, :3] = R_m2c
    T_m2c[:3, 3] = t_m2c
    
    T_w2c = all_optimization_data[10]['T_w2c']
    T_c2w = np.linalg.inv(T_w2c)
    T_m2w_init = T_c2w @ T_m2c
    R_init = T_m2w_init[:3, :3]
    t_init = T_m2w_init[:3, 3]

    colmap_points3D_path = os.path.join(colmap_dir, 'points3D.txt')
    images_txt = os.path.join(colmap_dir, 'images.txt')
    images_colmap = read_colmap_images_txt(images_txt) # 假设返回字典
    X_W_obj = get_object_3d_points(
        colmap_points3D_path, 
        all_optimization_data, 
        mask_dir,
        images_colmap # 您在预处理阶段读取的 COLMAP 图像信息
    )
    s_init = calculate_initial_TMW_from_bbox(
        model_vertices, 
        X_W_obj
    )

    from scipy.spatial.transform import Rotation
    r_vec_init = Rotation.from_matrix(R_init).as_rotvec()
    initial_params = np.array([s_init, r_vec_init[0], r_vec_init[1], r_vec_init[2], 
                            t_init[0], t_init[1], t_init[2]], dtype=np.float64)
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
    log_filename = "least_squares_log.txt"
    log_filepath = os.path.join(output_dir, log_filename)

    
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
    
    # 3. 提取优化结果
    final_params = result.x
    s_final = final_params[0]
    R_M2W_final = axis_angle_to_rotmat(final_params[1:4])
    t_M2W_final = final_params[4:7]
    print(f"Optimized Scale (s): {s_final}")
    print(f"Optimized R_M2W:\n{R_M2W_final}")
    print(f"Optimized t_M2W:\n{t_M2W_final}")
    
    # 4. 构造最终的 T_M2W 矩阵
    T_M2W_final = np.eye(4, dtype=np.float32)
    T_M2W_final[:3, :3] = s_final * R_M2W_final 
    T_M2W_final[:3, 3] = s_final * t_M2W_final
    
    logger.info("Starting visualization of optimized pose...")
    
    # --- 阶段 3: 最终可视化 (可选，使用最后一批次的 T_M2W) ---
    if T_M2W_final is not None:
        logger.info("Starting final visualization using last optimized pose...")
        for data in tqdm(all_optimization_data, desc="Final Visualizing"):
            T_W2C = data['T_w2c']
            K = data['K']
            image_path = data['image_path']
            visualize_projection(
                image_path=image_path,
                output_dir=output_vis_dir,
                K=K,
                T_W2C=T_W2C,
                T_M2W=T_M2W_final,
                model_vertices=model_vertices
            )
        for i, data in enumerate(all_optimization_data):
            T_W2C = data['T_w2c']
            K = data['K']
            image_path = data['image_path']
            # 获取局部最佳姿态
            best_pose = all_corresps_data[i].best_pose 
            visualize_projection_multi_pose( # <<< 调用新的多姿态渲染函数
                image_path=image_path,
                output_dir=output_vis_dir,
                K=K,
                T_W2C=T_W2C,
                T_M2W_final=T_M2W_final,
                best_pose=best_pose,
                model_vertices=model_vertices,
                model_faces=model_faces # <<< 传入面片索引
            )
            
        
    logger.info("All processes complete.")

def load_ply_vertices_faces(path):
    import open3d as o3d # 用于读取PLY文件
    mesh = o3d.io.read_triangle_mesh(path)
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    return vertices, faces

def load_model_vertices(model_path: str) -> np.ndarray:
    """加载 PLY 模型文件并返回顶点坐标 (N, 3)。"""
    try:
        import open3d as o3d # 用于读取PLY文件
        mesh = o3d.io.read_triangle_mesh(model_path)
        if not mesh.has_vertices():
            raise ValueError("PLY file does not contain vertices.")
        vertices = np.asarray(mesh.vertices, dtype=np.float32)
        # 简化：仅取 1000 个点进行投影，以提高速度
        if vertices.shape[0] > 1000:
            indices = np.random.choice(vertices.shape[0], 1000, replace=False)
            vertices = vertices[indices]
        return vertices
    except Exception as e:
        logger.error(f"Error loading model {model_path}: {e}")
        return np.array([])

def visualize_projection(
    image_path: str,
    output_dir: str,
    K: np.ndarray,
    T_W2C: np.ndarray,
    T_M2W: np.ndarray,
    model_vertices: np.ndarray
):
    """
    将模型顶点投影到图像上并保存结果。
    
    T_M2W 是 [sR | t] 形式，其中 R 包含尺度 s。
    """
    
    image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image_bgr is None:
        logger.error(f"Could not load image: {image_path}")
        return

    # 2. 变换模型顶点: M -> W -> C
    
    # M -> W 变换 (处理尺度)
    # T_M2W 已经是 [sR | t] 形式
    X_3d_M_h = np.concatenate([model_vertices, np.ones((model_vertices.shape[0], 1))], axis=1) # (N, 4)
    X_3d_W_h = (T_M2W @ X_3d_M_h.T).T # (N, 4)
    
    # W -> C 变换
    X_3d_C_h = (T_W2C @ X_3d_W_h.T).T
    X_3d_C = X_3d_C_h[:, :3] # (N, 3)
    
    # 3. 投影到 2D
    q_2d_proj = project_points(K, X_3d_C)
    
    # 4. 绘制投影点
    output_image = image_bgr.copy()
    
    # 确保投影点在图像边界内
    h, w = output_image.shape[:2]
    valid_mask = (q_2d_proj[:, 0] >= 0) & (q_2d_proj[:, 0] < w) & \
                 (q_2d_proj[:, 1] >= 0) & (q_2d_proj[:, 1] < h) & \
                 (X_3d_C[:, 2] > 0) # 确保点在相机前面 (深度 > 0)
    
    valid_points = q_2d_proj[valid_mask].astype(int)
    
    # 绘制：绿色点
    for pt in valid_points:
        # 使用低不透明度（即在原图上画点，看起来像半透明）
        # 这里用一个小圆圈代替半透明效果，因为 OpenCV 绘制基本图形不支持 alpha 混合
        # 或者直接画点
        cv2.circle(output_image, tuple(pt), 2, (0, 255, 0), -1) # BGR: 绿色
        
    # 5. 保存结果
    base_name = os.path.basename(image_path)
    output_dir = output_dir + "/fres"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"projected_{base_name}")
    cv2.imwrite(output_path, output_image)
    
def visualize_projection_mesh(
    image_path: str,
    output_dir: str,
    K: np.ndarray,
    T_W2C: np.ndarray,
    T_M2W: np.ndarray,
    model_vertices: np.ndarray,
    model_faces: np.ndarray,   # <-- 新增：渲染用 mesh faces
    color=(0, 255, 0)
):
    """
    将模型线框渲染到图像上。
    T_M2W 是 [sR | t]。
    """

    # 1. 载入图像
    image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image_bgr is None:
        print(f"Could not load image: {image_path}")
        return

    # 2. 顶点 M → W → C
    X_3d_M_h = np.concatenate([model_vertices, np.ones((model_vertices.shape[0], 1))], axis=1)
    X_3d_W_h = (T_M2W @ X_3d_M_h.T).T
    X_3d_C_h = (T_W2C @ X_3d_W_h.T).T
    X_3d_C = X_3d_C_h[:, :3]

    # 3. 投影
    q_2d_proj = project_points(K, X_3d_C)

    # 图像尺寸检查
    h, w = image_bgr.shape[:2]

    # 创建输出图像
    output_image = image_bgr.copy()

    # 4. 渲染 mesh（线框）
    for f in model_faces:
        v0, v1, v2 = f
        pts = q_2d_proj[f]  # (3,2)

        # 深度检查：三个点都在相机前方
        if np.any(X_3d_C[f, 2] <= 0):
            continue

        # 在图像范围检查
        if np.any(pts[:, 0] < 0) or np.any(pts[:, 0] >= w) or \
           np.any(pts[:, 1] < 0) or np.any(pts[:, 1] >= h):
            continue

        p0 = tuple(pts[0].astype(int))
        p1 = tuple(pts[1].astype(int))
        p2 = tuple(pts[2].astype(int))

        cv2.line(output_image, p0, p1, color, 1)
        cv2.line(output_image, p1, p2, color, 1)
        cv2.line(output_image, p2, p0, color, 1)

    # 5. 保存
    output_dir = os.path.join(output_dir, "fres")
    os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.basename(image_path)
    output_path = os.path.join(output_dir, f"mesh_{base_name}")

    cv2.imwrite(output_path, output_image)

def visualize_projection_multi_pose(
    image_path: str,
    output_dir: str,
    K: np.ndarray,
    T_W2C: np.ndarray,
    T_M2W_final: np.ndarray, # 优化后的全局 M -> W 姿态
    best_pose: dict, # corresp_data.best_pose
    model_vertices: np.ndarray,
    model_faces: np.ndarray
):
    """
    将两种姿态下的模型（线框）渲染到同一图像上。
    
    姿态 1 (优化后): T_M2C_final = T_W2C @ T_M2W_final
    姿态 2 (局部PnP): T_M2C_best_pose
    """
    
    # 1. 图像加载
    image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image_bgr is None:
        logger.error(f"Could not load image: {image_path}")
        return

    # 2. 姿态 1: 优化后的全局姿态 (Final Optimized Pose)
    T_M2C_final = T_W2C @ T_M2W_final
    
    # 3. 姿态 2: 局部最佳姿态 (Best Pose from PnP/Infer)
    R_m2c_local = best_pose["R_m2c"]
    t_m2c_local = best_pose["t_m2c"]
    T_M2C_local = np.eye(4, dtype=np.float32)
    T_M2C_local[:3, :3] = R_m2c_local
    T_M2C_local[:3, 3] = t_m2c_local

    # 4. 渲染
    output_image = image_bgr.copy()
    
    # 渲染姿态 2 (局部PnP) - 例如使用红色
    output_image = visualize_wireframe(
        output_image, 
        K, 
        T_M2C_local, 
        model_vertices, 
        model_faces, 
        color=(0, 0, 255), # BGR: 红色 (局部姿态)
        thickness=1
    )

    # 渲染姿态 1 (优化后) - 例如使用绿色
    output_image = visualize_wireframe(
        output_image, 
        K, 
        T_M2C_final, 
        model_vertices, 
        model_faces, 
        color=(255, 255, 0), # BGR: 绿色 (优化后全局姿态)
        thickness=1
    )
        
    # 5. 保存结果
    base_name = os.path.basename(image_path)
    output_dir = output_dir + "/vis" # 避免重复添加
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"projected_wireframe_{base_name}")
    cv2.imwrite(output_path, output_image)
    logger.info(f"Saved wireframe visualization to {output_path}")

def visualize_wireframe(
    image_bgr: np.ndarray,
    K: np.ndarray,
    T_M2C: np.ndarray,
    model_vertices: np.ndarray,
    model_faces: np.ndarray,
    color: tuple, # BGR 格式
    thickness: int = 1,
    alpha: float = 0.2
):
    """
    将模型渲染为线框（Wireframe），不带深度缓冲。
    
    T_M2C 是 (4, 4) 矩阵。
    model_vertices 是 (N, 3) 顶点。
    model_faces 是 (M, 3) 索引（三角形）。
    """
    
    if model_vertices.size == 0 or model_faces.size == 0:
        return image_bgr
        
    # 1. 变换模型顶点: M -> C
    X_3d_M_h = np.concatenate([model_vertices, np.ones((model_vertices.shape[0], 1))], axis=1) # (N, 4)
    X_3d_C_h = (T_M2C @ X_3d_M_h.T).T
    X_3d_C = X_3d_C_h[:, :3] # (N, 3)
    
    # 2. 深度检查和投影
    # 仅投影位于相机前方的点 (Z > 0)
    valid_mask = X_3d_C[:, 2] > 1e-6 # 检查深度
    
    # 投影所有点
    q_2d_proj = project_points(K, X_3d_C) # (N, 2)
    
    # 创建一个用于绘制线框的临时透明层
    overlay = image_bgr.copy()

    # 3. 绘制投影后的线框 (绘制到 overlay 层)
    for face in model_faces:
        # 确保面片的三个顶点都在相机前面
        try:
            if not (valid_mask[face[0]] and valid_mask[face[1]] and valid_mask[face[2]]):
                continue
        except IndexError as e:
            # 捕获索引越界错误，通常表示 model_faces 中的索引大于 model_vertices 的大小
            # 建议打印 face[0], face[1], face[2] 和 valid_mask.shape 来进一步诊断
            logger.error(f"IndexError in visualize_wireframe: {e}. Check model_faces vs model_vertices size.")
            # 为了防止程序中断，跳过此面片
            continue 
            
        # 提取投影点
        pts = q_2d_proj[face].astype(int)
        
        # 绘制三角形的三条边
        cv2.line(overlay, tuple(pts[0]), tuple(pts[1]), color, thickness)
        cv2.line(overlay, tuple(pts[1]), tuple(pts[2]), color, thickness)
        cv2.line(overlay, tuple(pts[2]), tuple(pts[0]), color, thickness)

    # 4. Alpha 混合 (将绘制的 overlay 层与原图 image_bgr 混合)
    # 使用 cv2.addWeighted 实现 I_out = alpha * I_overlay + (1 - alpha) * I_original
    output_image = cv2.addWeighted(overlay, alpha, image_bgr, 1 - alpha, 0)

    return output_image


# --- 缓存函数 ---
def save_optimization_data(data_to_save, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'wb') as f:
        pickle.dump(data_to_save, f)
    logger.info(f"Data saved successfully to {file_path}")

def load_optimization_data(file_path):
    with open(file_path, 'rb') as f:
        data_loaded = pickle.load(f)
    logger.info(f"Data loaded successfully from {file_path}")
    return data_loaded['optimization_data'], data_loaded['corresps_data']

# --- 可视化 2D-3D 对应点 ---
def visualize_2d_3d_correspondences(image_np_hwc, q_2d, X_3d, R_m2c, t_m2c, camK, output_path, frame_name):
    """
    image_np_hwc: 原始图像 numpy array, HWC, float32 [0,1]
    q_2d: 对应的2D点坐标, numpy array (N,2)
    X_3d: 对应的3D点坐标, numpy array (N,3)
    R_m2c: 模型到相机旋转矩阵, shape (3,3)
    t_m2c: 模型到相机平移向量, shape (3,1)
    camK: 相机内参, shape (3,3)
    output_path: 保存目录
    frame_name: 图像名称
    """
    os.makedirs(output_path, exist_ok=True)
    img_vis = (image_np_hwc * 255).astype(np.uint8).copy()
    
    # 投影 3D 点到图像平面
    # X_c = R_m2c @ X_3d.T + t_m2c
    X_c = (R_m2c @ X_3d.T) + t_m2c  # shape (3,N)
    # 投影到像素平面
    x = (camK[0,0] * X_c[0] / X_c[2]) + camK[0,2]
    y = (camK[1,1] * X_c[1] / X_c[2]) + camK[1,2]
    pts_3d_proj = np.stack([x, y], axis=1)  # shape (N,2)
    
    # 绘制 2D 点
    for pt in q_2d:
        px, py = int(pt[0]), int(pt[1])
        cv2.circle(img_vis, (px, py), radius=3, color=(0,0,255), thickness=-1)  # 红色
    
    # 绘制投影后的 3D 点
    for pt in pts_3d_proj:
        px, py = int(pt[0]), int(pt[1])
        cv2.circle(img_vis, (px, py), radius=3, color=(0,255,0), thickness=-1)  # 绿色
    
    save_file = os.path.join(output_path, f"{frame_name}_2d_3d.png")
    cv2.imwrite(save_file, img_vis)
    print(f"[INFO] 2D-3D correspondence visualization saved: {save_file}")

if __name__ == "__main__":
    main()