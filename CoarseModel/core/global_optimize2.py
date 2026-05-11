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


def global_reprojection_error(params, all_data):
    from scipy.spatial.transform import Rotation

    # 参数解包：3个尺度, 3个旋转(轴角), 3个平移
    scales = params[0:3]  # [sx, sy, sz]
    rvec = params[3:6]
    t = params[6:9]

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
        # Xw = s * ((R @ X.T).T + t)
        Xw_h = np.hstack([Xw, np.ones((len(Xw), 1))])
        Xc = (T_w2c @ Xw_h.T).T[:, :3]

        Z = Xc[:, 2]
        # uv = np.zeros_like(q)
        # 4. 投影计算
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
    import open3d as o3d
    pcd_filter = o3d.geometry.PointCloud()
    pcd_filter.points = o3d.utility.Vector3dVector(X_W_obj)
    # 这里的参数要严一点，剔除掉离群的点
    pcd_filter, _ = pcd_filter.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.0)
    X_W_obj_clean = np.asarray(pcd_filter.points)
    s_init = calculate_initial_TMW_from_obb(model_vertices, X_W_obj_clean)

    rvec_init = Rotation.from_matrix(R_init).as_rotvec()
    initial_params = np.array([*s_init, *rvec_init, *t_init], dtype=np.float64)
    
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
                verbose=2,  # 这个输出会被重定向到文件
                loss='soft_l1'
            )

    final_scales = result.x[0:3]
    final_R = Rotation.from_rotvec(result.x[3:6]).as_matrix()
    final_t = result.x[6:9]
    
    T_final = np.eye(4, dtype=np.float32)
    # T = R @ S + t
    T_final[:3, :3] = final_R @ np.diag(final_scales)
    T_final[:3, 3] = final_t

    return T_final, result, final_scales
