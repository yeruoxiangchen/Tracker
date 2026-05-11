import numpy as np
import trimesh
from typing import Any, List, Optional, Dict

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

def read_colmap_cameras_txt(path):
    cameras = {}
    with open(path, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            elems = line.strip().split()
            if len(elems) < 5:
                continue
            cam_id = int(elems[0])
            model = elems[1]
            w, h = int(elems[2]), int(elems[3])
            params = np.array(list(map(float, elems[4:])))
            cameras[cam_id] = dict(
                model=model,
                width=w,
                height=h,
                params=params
            )
    return cameras

def read_colmap_images_txt(path):
    """
    Returns:
        images: dict[image_name] = {
            'image_id': int,
            'cam_id': int,
            'R': (3,3),
            't': (3,)
        }
    """
    images = {}

    with open(path, 'r') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # 跳过空行 / 注释
        if line == "" or line.startswith("#"):
            i += 1
            continue

        elems = line.split()
        if len(elems) < 10:
            raise ValueError(f"Invalid image header line: {line}")

        image_id = int(elems[0])
        qw, qx, qy, qz = map(float, elems[1:5])
        tx, ty, tz = map(float, elems[5:8])
        cam_id = int(elems[8])
        image_name = elems[9]

        # quaternion → rotation
        R = quat_to_rotmat([qw, qx, qy, qz])
        t = np.array([tx, ty, tz], dtype=np.float32)

        images[image_name] = {
            "image_id": image_id,
            "cam_id": cam_id,
            "R": R,
            "t": t,
        }

        # ⚠️ 关键：跳过下一行（POINTS2D）
        i += 2

    return images

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

def load_ply_vertices_faces(path):
    import open3d as o3d # 用于读取PLY文件
    mesh = o3d.io.read_triangle_mesh(path)
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    return vertices, faces

def load_obj_with_visual(path):
    """
    加载 OBJ 模型，保留顶点、面片和视觉信息（纹理/UV）
    """
    # process=False 保证不重新合并顶点，维持顶点索引顺序不变
    mesh = trimesh.load(path, process=False)
    
    # 如果是 Scene 对象（OBJ 有时会被读成场景），取其几何体
    if isinstance(mesh, trimesh.Scene):
        if len(mesh.geometry) == 0:
            return None, None, None
        # 拿到第一个几何体
        mesh = next(iter(mesh.geometry.values()))
        
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    visual = mesh.visual
    
    return vertices, faces, visual

def get_instances_from_mask( 
    obj_id: int,
    mask: np.ndarray,
) -> List[Dict[str, Any]]:
    # 提取mask的边界框 (x1, y1, x2, y2)
    ys, xs = mask.nonzero()
    if len(xs) == 0 or len(ys) == 0:
        return []  # 空mask直接返回

    x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
    box_amodal = np.array([x1, y1, x2, y2], dtype=np.float32)

    instance_infos = [{
        "input_box_amodal": box_amodal,
        "input_mask_modal": mask.astype(np.uint8),
    }]
    return instance_infos