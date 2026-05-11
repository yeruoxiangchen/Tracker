# core/global_optimize.py
import os
import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from scipy.spatial import cKDTree

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
    # colmap_points3D_path = os.path.join(colmap_dir, 'points3D.txt')
    # images_txt = os.path.join(colmap_dir, 'images.txt')
    # images_colmap = read_colmap_images_txt(images_txt) # 假设返回字典
    # X_W_obj = get_object_3d_points(
    #     points3d_path=os.path.join(colmap_dir, "points3D.txt"),
    #     all_optimization_data=all_optimization_data,
    #     mask_dir=mask_dir,
    #     images_colmap=images_colmap,
    #     min_observations=3,
    # )
    
    # s_init = calculate_initial_TMW_from_bbox(model_vertices, X_W_obj)
    s_init = 3.3057

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

    return T, result


def refine_model_with_deformation_graph(
    model_vertices,
    model_faces,
    T_M2W,
    all_optimization_data,
    num_nodes=20,
    knn_node=4,
    knn_vertex=4,
    lambda_reg=1e4
):
    """
    model_vertices: (N,3)
    T_M2W: (4,4)
    """

    V0 = model_vertices.copy()
    N = V0.shape[0]

    # --- 1. 采样 graph nodes ---
    node_indices = farthest_point_sampling(V0, num_nodes)
    node_pos = V0[node_indices]   # (K,3)
    
    # --- 2. 构建 graph ---
    graph_edges = build_knn_graph(node_pos, knn_node)
    vertex_nodes, vertex_weights = compute_vertex_node_weights(
        V0, node_pos, k=knn_vertex
    )

    # --- 3. 初始参数 ---
    K = node_pos.shape[0]
    x0 = np.zeros(6 * K, dtype=np.float64)
    
    fixed_obs_indices = []
    max_obs_per_frame = 50 # 增加到 50 个点以保证约束强度
    for data in all_optimization_data:
        valid_mask = np.where(data["X_ids"] != -1)[0]
        if len(valid_mask) > max_obs_per_frame:
            sampled = np.random.choice(valid_mask, max_obs_per_frame, replace=False)
        else:
            sampled = valid_mask
        fixed_obs_indices.append(sampled)
    
    # --- 将 graph_edges 转为 numpy 数组加速索引 ---
    graph_edges = np.array(graph_edges)
    
    # --- 4. 最小二乘 ---
    # from contextlib import redirect_stdout
    # log_path = os.path.join("/home/zjr/Tracker/CoarseModel/results/refine_model/wogua", "deformation_graph_optimization_log.txt")

    # with open(log_path, "w") as f:
    #     with redirect_stdout(f):
    result = least_squares(
        deformation_graph_residual,
        x0,
        verbose=2,          # 关键
        method="trf",
        x_scale="jac",
        args=(
            V0, node_pos, vertex_nodes, vertex_weights,
            graph_edges, T_M2W, all_optimization_data, lambda_reg, fixed_obs_indices
        )
    )

    # --- 5. 应用最终形变 ---
    R_list, t_list = unpack_params(result.x)
    V_refined = apply_deformation_graph(
        V0, node_pos, R_list, t_list,
        vertex_nodes, vertex_weights
    )

    return V_refined

def deformation_graph_residual(
    x,
    V0,
    node_pos,
    vertex_nodes,
    vertex_weights,
    graph_edges,
    T_M2W,
    all_optimization_data,
    lambda_reg,
    fixed_obs_indices
):
    R_list, t_list = unpack_params(x)
    residuals = []
    K_nodes = node_pos.shape[0]

    # ---------- Reprojection term ----------
    for i, data in enumerate(all_optimization_data):
        obs_idx = fixed_obs_indices[i]
        if len(obs_idx) == 0: continue
    
        v_ids = data["X_ids"][obs_idx]
        q2d = data["q_2d"][obs_idx]
        K = data["K"]
        T_w2c = data["T_w2c"]

        V_sub = V0[v_ids]
        vw_sub = vertex_weights[v_ids]
        vn_sub = vertex_nodes[v_ids]

        # 矢量化变形
        V_def = apply_deformation_vectorized(V_sub, node_pos, R_list, t_list, vn_sub, vw_sub)

        # 投影到图像: V_def -> World -> Camera -> Pixel
        # T_total = T_w2c @ T_M2W
        T_total = T_w2c @ T_M2W
        V_def_h = np.hstack([V_def, np.ones((len(V_def), 1))])
        V_c = (T_total @ V_def_h.T).T[:, :3]
        
        # 避免除以 0
        z = V_c[:, 2:3]
        z = np.maximum(z, 1e-6)
        proj = (K @ V_c.T).T[:, :2] / z
        
        reproj_weight = 0.1   # 非常重要
        residuals.append(reproj_weight * (proj - q2d).ravel())

    # 2. ARAP 正则项 (Smoothness)
    # 矢量化计算边约束
    idx_k = graph_edges[:, 0]
    idx_l = graph_edges[:, 1]
    
    # pred = R_k * (g_l - g_k) + g_k + t_k
    g_k, g_l = node_pos[idx_k], node_pos[idx_l]
    vec_kl = (g_l - g_k)[..., None]
    pred = (R_list[idx_k] @ vec_kl).squeeze(-1) + g_k + t_list[idx_k]
    target = g_l + t_list[idx_l]
    
    reg_res = np.sqrt(lambda_reg) * (pred - target)
    residuals.append(reg_res.ravel())
    
    # ---------- Small deformation prior ----------
    prior_weight = 1.0

    for i in range(K_nodes):
        # penalize large translations
        residuals.append(prior_weight * t_list[i])

        # penalize rotation away from identity
        rot_res = Rotation.from_matrix(R_list[i]).as_rotvec()
        residuals.append(prior_weight * rot_res)
        
    # 这里有点疑问，强制固定某个点
    anchor_weight = 1e6
    R0 = R_list[0]
    t0 = t_list[0]
    # R0 ≈ I
    residuals.append(anchor_weight * (R0 - np.eye(3)).ravel())
    # t0 ≈ 0
    residuals.append(anchor_weight * t0)
    
    return np.concatenate(residuals)

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

    dists = np.maximum(dists, 1e-8)
    weights = 1.0 / dists
    weights /= weights.sum(axis=1, keepdims=True)

    return idxs, weights

