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

from script.vis_util import visualize_contour_matches
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
    # 窝瓜 s_init = 3.3057 菜花s_init =  1.2808 背包s_init = 2.0575

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

    s_final = result.x[0]
    R_final = axis_angle_to_rotmat(result.x[1:4])
    t_final = result.x[4:7]
    
    # 返回纯位姿矩阵 (不带尺度缩放)
    T_M2W_pure = np.eye(4, dtype=np.float32)
    T_M2W_pure[:3, :3] = R_final
    T_M2W_pure[:3, 3] = t_final

    return T_M2W_pure, s_final, result

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

def update_contour_correspondences(V_current, model_faces, T_M2W, all_optimization_data):
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
        
def refine_model_with_deformation_graph(
    model_vertices,
    model_faces,
    T_M2W,
    all_optimization_data,
    output_dir,
    num_nodes=20,
    knn_node=4,
    knn_vertex=4,
    lambda_reg=1
):
    # --- 0. 预处理 2D 轮廓 ---
    # 只需执行一次
    preprocess_2d_contours(all_optimization_data, num_samples=20)

    V0 = model_vertices.copy()
    
    # --- [新增：逻辑焊接预处理] ---
    # 找到物理位置重合的顶点。round(6) 用于处理微小的浮点数精度问题
    _, inverse_mapping = np.unique(np.round(V0, 6), axis=0, return_inverse=True)
    
    # --- 1. 采样 graph nodes ---
    node_indices = farthest_point_sampling(V0, num_nodes)
    node_pos = V0[node_indices]   # (K,3)
    
    # --- 2. 构建 graph ---
    graph_edges = build_knn_graph(node_pos, knn_node)
    vertex_nodes, vertex_weights = compute_vertex_node_weights(
        V0, node_pos, k=knn_vertex
    )
    graph_edges = np.array(graph_edges)
    
    # --- [修改：计算统一权重] ---
    # 1. 提取不重复的顶点位置
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

    # --- 4. 迭代优化 (ICP-style Loop) ---
    # 定义外层循环次数，通常 2-3 次即可收敛
    outer_iterations = 3
    
    K = node_pos.shape[0]
    # 初始参数 x0 (平移为0, 旋转为0即单位阵)
    current_x = np.zeros(6 * K, dtype=np.float64)
    
    for outer_iter in range(outer_iterations):
        print(f"=== Outer Iteration {outer_iter + 1}/{outer_iterations} ===")
        
        # A. 根据当前的 deformation 参数，计算当前的 Mesh 形态
        R_list, t_list = unpack_params(current_x)
        V_current = apply_deformation_graph(
            V0, node_pos, R_list, t_list, vertex_nodes, vertex_weights
        )
        
        # B. 更新 3D-2D 轮廓对应关系
        #    这步计算量相对较大，但在外层循环做可以接受
        update_contour_correspondences(V_current, model_faces, T_M2W, all_optimization_data)
        
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
                graph_edges, T_M2W, all_optimization_data, 
                lambda_reg
            )
        )
        current_x = result.x

    # --- 5. 应用最终形变 ---
    R_list, t_list = unpack_params(current_x)
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
):
    R_list, t_list = unpack_params(x)
    residuals = []
    K_nodes = node_pos.shape[0]

    # 为了批量计算变形，可以先只提取出本轮所有需要的顶点
    # 但由于 Vertex Deformation 依赖 KNN 权重，按帧处理逻辑比较清晰
    
    for i, data in enumerate(all_optimization_data):
        # 从外层循环预计算的结果中获取轮廓点索引
        v_ids_contour = data.get('contour_v_ids')
        if v_ids_contour is None:
            v_ids_contour = np.array([], dtype=int)
        
        # 2. 变形这些特定的 3D 顶点
        V_sub = V0[v_ids_contour]
        vw_sub = vertex_weights[v_ids_contour]
        vn_sub = vertex_nodes[v_ids_contour]
        
        V_def = apply_deformation_vectorized(V_sub, node_pos, R_list, t_list, vn_sub, vw_sub)
        
        # 3. 投影到 2D
        T_total = data["T_w2c"] @ T_M2W
        V_def_h = np.hstack([V_def, np.ones((len(V_def), 1))])
        V_c = (T_total @ V_def_h.T).T[:, :3]
        z = np.maximum(V_c[:, 2:3], 1e-6)
        proj_contour = (data["K"] @ V_c.T).T[:, :2] / z
        
        # 4. 计算与 2D 目标边缘的残差
        target_2d = data['edge_2d'] # 预处理提取的 20 个点
        diff = proj_contour - target_2d
        
        # 核心：使用 Point-to-Plane (点到切线) 残差
        # 这种方式允许 3D 点在轮廓线上滑动，只惩罚离开轮廓线的位移
        if data.get('edge_grad') is not None:
            grads = data['edge_grad'] # 预处理提取的 2D 法向
            # 每个点只产生 1 个标量残差：投影偏差在法向上的投影
            res_contour = np.sum(diff * grads, axis=1)
        else:
            # 退而求其次，使用点到点欧式距离 (产生 2 个残差: dx, dy)
            res_contour = diff.ravel()
            
        contour_weight = 1.0 # 纯轮廓驱动，权重可以设大
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

    d_max = dists[:, -1][:, None] + 1e-6 
    
    # ED 论文公式: weight = (1 - dist / d_max)^2
    # 这样保证在影响半径边缘权重平滑降为 0
    weights = (1.0 - dists / d_max) ** 2
    weights[dists > d_max] = 0 # 理论上不会发生，因为我们是用 knn 查的，但为了保险
    
    # 3. 归一化 (Partition of Unity) !!! 这一步如果不做，就会出现裂痕
    w_sum = np.sum(weights, axis=1, keepdims=True)
    weights = weights / (w_sum + 1e-8)
    
    return idxs, weights

