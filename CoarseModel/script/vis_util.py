import os
import cv2
import trimesh
import traceback
import numpy as np
from utils.structs import PinholePlaneCameraModel

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

def visualize_pose_overlay(
    image_raw: np.ndarray,
    best_pose: dict,
    model_path: str,
    orig_camera: PinholePlaneCameraModel,
    crop_camera: PinholePlaneCameraModel,
    frame_name: str,
    output_dir: str,
    alpha: float = 0.4
):
    """
    将模型线框投影并渲染到原始图像上，包含坐标系转换逻辑。
    """
    # 1. 坐标系转换：从裁剪相机 (Crop) 变换到原始相机 (Orig)
    # T_m2c_crop: 模型到裁剪相机
    R_m2c_crop = best_pose["R_m2c"]
    t_m2c_crop = best_pose["t_m2c"].reshape(3, 1)

    # 获取相机到世界的变换矩阵
    T_c2w_crop = crop_camera.T_world_from_eye
    T_c2w_orig = orig_camera.T_world_from_eye
    T_w2c_orig = np.linalg.inv(T_c2w_orig)

    # 复合变换：T_m2c_orig = T_w2c_orig * T_c2w_crop * T_m2c_crop
    R_w2c_orig = T_w2c_orig[:3, :3]
    t_w2c_orig = T_w2c_orig[:3, 3:4]
    R_c2w_crop = T_c2w_crop[:3, :3]
    t_c2w_crop = T_c2w_crop[:3, 3:4]

    R_m2c_orig = R_w2c_orig @ R_c2w_crop @ R_m2c_crop
    t_m2c_orig = R_w2c_orig @ (R_c2w_crop @ t_m2c_crop + t_c2w_crop) + t_w2c_orig

    # 2. 加载模型并投影到原图像素坐标
    mesh = trimesh.load(model_path, force='mesh')
    verts = np.asarray(mesh.vertices, dtype=np.float32)
    
    # 投影公式: p = K * (R*X + t)
    Xc_orig = (R_m2c_orig @ verts.T).T + t_m2c_orig.reshape(3)
    
    fx, fy = orig_camera.f
    cx, cy = orig_camera.c
    u = (fx * Xc_orig[:, 0] / Xc_orig[:, 2]) + cx
    v = (fy * Xc_orig[:, 1] / Xc_orig[:, 2]) + cy
    pts_2d = np.stack([u, v], axis=1).astype(int)

    # 3. 渲染线框
    vis_img = image_raw.astype(np.uint8).copy()
    overlay = vis_img.copy()
    
    # 绘制 Mesh 面片边缘
    for face in mesh.faces:
        p1, p2, p3 = pts_2d[face]
        cv2.polylines(overlay, [np.array([p1, p2, p3])], True, (150, 25, 120), 1, cv2.LINE_AA)

    # 半透明叠加
    cv2.addWeighted(overlay, alpha, vis_img, 1 - alpha, 0, vis_img)

    # 4. 保存结果
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{frame_name}_overlay.png")
    cv2.imwrite(out_path, cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))

def visualize_pose_overlay2(
    image_raw: np.ndarray,
    best_pose: dict,
    mesh: trimesh.Trimesh,      # 直接传入 mesh 对象
    orig_camera: PinholePlaneCameraModel,
    crop_camera: PinholePlaneCameraModel,
    frame_name: str,
    output_dir: str,
    alpha: float = 0.4
):
    """
    将模型线框投影并渲染到原始图像上。
    """
    import cv2
    import numpy as np

    # 1. 坐标系转换逻辑
    # 获取 4x4 变换矩阵
    # R_m2c_crop = best_pose["R_m2c"]
    # t_m2c_crop = best_pose["t_m2c"].reshape(3, 1)
    # T_m2c_crop = np.eye(4)
    # T_m2c_crop[:3, :3] = R_m2c_crop
    # T_m2c_crop[:3, 3:4] = t_m2c_crop

    # # 获取相机到世界的变换矩阵
    # T_c2w_crop = crop_camera.T_world_from_eye
    # T_c2w_orig = orig_camera.T_world_from_eye
    # T_w2c_orig = np.linalg.inv(T_c2w_orig)

    # # 计算模型到原始相机的完整变换: T_m2c_orig = T_w2c_orig @ T_c2w_crop @ T_m2c_crop
    # T_m2c_orig = T_w2c_orig @ T_c2w_crop @ T_m2c_crop
    # R_m2c_orig = T_m2c_orig[:3, :3]
    # t_m2c_orig = T_m2c_orig[:3, 3]
    R_m2c_orig = best_pose["R_m2c"]
    t_m2c_orig = best_pose["t_m2c"].reshape(3)

    # 2. 使用传入的 mesh 进行投影
    # 获取顶点并转为齐次坐标进行一次性变换
    verts = np.asarray(mesh.vertices, dtype=np.float32)
    # 投影公式: p = K * (R*X + t)
    Xc_orig = (R_m2c_orig @ verts.T).T + t_m2c_orig
    
    fx, fy = orig_camera.f
    cx, cy = orig_camera.c
    
    # 防止除以 0 导致崩溃
    z = np.maximum(Xc_orig[:, 2], 1e-6)
    u = (fx * Xc_orig[:, 0] / z) + cx
    v = (fy * Xc_orig[:, 1] / z) + cy
    pts_2d = np.stack([u, v], axis=1).astype(int)

    # 3. 渲染线框
    # 确保图像格式正确
    vis_img = image_raw.copy()
    if vis_img.dtype != np.uint8:
        vis_img = (vis_img * 255).astype(np.uint8)
    
    # 转换为 BGR 用于 OpenCV 绘制
    if vis_img.shape[-1] == 3:
        vis_img = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)

    overlay = vis_img.copy()
    
    # 绘制 Mesh 面片边缘 (优化建议：如果面片过多，可以只画外轮廓或采样绘制)
    # 这里保持原逻辑：遍历 faces
    for face in mesh.faces:
        p1, p2, p3 = pts_2d[face]
        cv2.polylines(overlay, [np.array([p1, p2, p3])], True, (150, 25, 120), 1, cv2.LINE_AA)

    # 半透明叠加
    cv2.addWeighted(overlay, alpha, vis_img, 1 - alpha, 0, vis_img)

    # 4. 保存结果
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{frame_name}_overlay.png")
    cv2.imwrite(out_path, vis_img)
    
def render_mesh_mask(
    image_shape,
    K: np.ndarray,
    T_M2C: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
):
    """
    将 mesh 渲染成二值 mask（silhouette）
    """
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    # 顶点投影
    V_h = np.concatenate([vertices, np.ones((vertices.shape[0], 1))], axis=1)
    V_c = (T_M2C @ V_h.T).T[:, :3]

    valid = V_c[:, 2] > 1e-6
    V_c = V_c[valid]
    idx_map = np.full(vertices.shape[0], -1)
    idx_map[np.where(valid)[0]] = np.arange(len(V_c))

    uv = project_points(K, V_c).astype(np.int32)

    for f in faces:
        if not valid[f].all():
            continue
        pts = uv[idx_map[f]]
        cv2.fillConvexPoly(mask, pts, 255)

    return mask

def draw_mask_contour(
    image,
    mask,
    color=(0, 255, 0),
    thickness=2,
):
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(image, contours, -1, color, thickness)
    return image


def visualize_projection(
    image_path: str,
    output_dir: str,
    K: np.ndarray,
    T_W2C: np.ndarray,
    T_M2W: np.ndarray,
    model_vertices: np.ndarray,
    model_faces: np.ndarray,
    image_name:str,
    root_name="fres",
):
    """
    将模型顶点投影到图像上并保存结果。
    
    T_M2W 是 [sR | t] 形式，其中 R 包含尺度 s。
    """
    
    image = cv2.imread(image_path)
    T_M2C = T_W2C @ T_M2W

    mask = render_mesh_mask(
        image.shape,
        K,
        T_M2C,
        model_vertices,
        model_faces,
    )

    output = draw_mask_contour(
        image,
        mask,
        color=(155, 38, 57),  # 绿色轮廓
        thickness=2,
    )

    out_dir = os.path.join(output_dir, root_name)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, image_name)
    cv2.imwrite(out_path, output)
    
def visualize_projection2(
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
    
    # 1. 图像加载
    image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
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
    # logger.info(f"Saved visualization to {output_path}")
    
    
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
    image = cv2.imread(image_path)

    # 全局优化姿态
    T_M2C_final = T_W2C @ T_M2W_final

    # 局部 PnP 姿态
    T_M2C_local = np.eye(4, dtype=np.float32)
    T_M2C_local[:3, :3] = best_pose["R_m2c"]
    T_M2C_local[:3, 3] = best_pose["t_m2c"]

    # 渲染 mask
    mask_final = render_mesh_mask(
        image.shape, K, T_M2C_final, model_vertices, model_faces
    )
    mask_local = render_mesh_mask(
        image.shape, K, T_M2C_local, model_vertices, model_faces
    )

    output = image.copy()
    output = draw_mask_contour(output, mask_local, color=(0, 0, 255), thickness=2)
    output = draw_mask_contour(output, mask_final, color=(0, 255, 0), thickness=2)

    out_dir = os.path.join(output_dir, "vis")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"contour_cmp_{os.path.basename(image_path)}")
    cv2.imwrite(out_path, output)

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
            print("--- 捕获到索引越界错误 ---")
            # 打印完整的报错堆栈信息，就像程序崩溃时显示的那样
            traceback.print_exc()
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


def visualize_contour_matches(V_current, T_M2W, all_optimization_data, outer_iter, output_dir):
    """
    可视化 2D 轮廓 (红色) 与 3D 投影轮廓 (蓝色) 及其法线
    """
    save_dir = os.path.join(output_dir,"contour")
    os.makedirs(save_dir, exist_ok=True)

    for i, data in enumerate(all_optimization_data):
        # 获取基础信息
        img_path = data['image_path']
        img = cv2.imread(img_path)
        if img is None: continue
        
        # 获取 2D 轮廓数据
        edge_2d = data.get('edge_2d')       # (M, 2)
        edge_grad = data.get('edge_grad')   # (M, 2)
        p3d_proj = data.get('last_v_proj')   # (M,)
        
        if edge_2d is None or p3d_proj is None: continue

        # 画布
        canvas = img.copy()
        arrow_scale = 15  # 箭头长度

        for j in range(len(edge_2d)):
            # 1. 绘制 2D 轮廓点及其法线 (红色)
            p2d = edge_2d[j].astype(int)
            g2d = edge_grad[j]
            cv2.circle(canvas, tuple(p2d), 3, (0, 0, 255), -1) # 红色点
            p2d_end = (p2d + g2d * arrow_scale).astype(int)
            cv2.arrowedLine(canvas, tuple(p2d), tuple(p2d_end), (0, 0, 255), 1, tipLength=0.3)

            # 2. 绘制 3D 投影点 (蓝色)
            p3d = p3d_proj[j].astype(int)
            cv2.circle(canvas, tuple(p3d), 2, (255, 0, 0), -1) # 蓝色点
            
            # 绘制匹配连线（浅绿色），查看对应关系是否正确
            cv2.line(canvas, tuple(p2d), tuple(p3d), (0, 255, 0), 1)

        # 保存图片，命名包含迭代次数和帧序号
        save_path = os.path.join(save_dir, f"iter_{outer_iter}_frame_{i:04d}.jpg")
        cv2.imwrite(save_path, canvas)

    print(f"Visualization saved to {save_dir}")
    
def visualize_per_frame_alignment(V_current, model_faces, all_optimization_data, outer_iter, output_dir, T_M2W):
    """
    可视化当前迭代中，所有帧的投影对齐情况。
    """
    import cv2
    import os

    # 为当前迭代创建文件夹，例如 output/iter_00, output/iter_01
    iter_dir = os.path.join(output_dir, f"iter_{outer_iter:02d}")
    os.makedirs(iter_dir, exist_ok=True)

    for i, data in enumerate(all_optimization_data):
        # 1. 获取基础数据
        image = cv2.imread(data["image_path"])
        if image is None: continue
        
        # 2. 计算当前帧的总变换矩阵 (Model -> Camera)
        # 注意：这里 T_M2W 通常是初始对齐，V_current 是在该空间下的形变结果
        T_total = data["T_w2c"] @ T_M2W
        
        # 3. 渲染 Mask (调用你现有的 render_mesh_mask)
        mask = render_mesh_mask(
            image.shape,
            data["K"],
            T_total,
            V_current,
            model_faces,
        )

        # 4. 绘制轮廓 (调用你现有的 draw_mask_contour)
        # 使用红色 (0, 0, 255) 表示当前形变状态
        result_img = draw_mask_contour(
            image,
            mask,
            color=(0, 0, 255), 
            thickness=2
        )

        # 5. 可选：在图片上标注帧信息和迭代次数
        cv2.putText(result_img, f"Iter: {outer_iter} Frame: {i}", (30, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 6. 保存图片
        base_name = os.path.basename(data["image_path"])
        save_path = os.path.join(iter_dir, f"frame_{i:03d}_{base_name}")
        cv2.imwrite(save_path, result_img)