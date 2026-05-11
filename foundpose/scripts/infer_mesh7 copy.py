#!/usr/bin/env python3

"""Infers pose from objects."""

import datetime

import sys
REPO_PATH = "/home/zjr/Tracker/foundpose"
BOP_TOOLKIT_PATH = "/home/zjr/Tracker/foundpose/external/bop_toolkit"

sys.path.insert(0, REPO_PATH)
sys.path.insert(0, BOP_TOOLKIT_PATH)  # 添加这一行

import os
import gc
import time

from typing import List, NamedTuple, Optional, Tuple

import cv2

import numpy as np

import torch

from utils.misc import array_to_tensor

from utils import (
    corresp_util,
    config_util,
    eval_util,
    feature_util,
    infer_pose_util,
    knn_util,
    misc as misc_util,
    pnp_util,
    projector_util,
    repre_util,
    data_util,
    json_util, 
    logging,
    misc,
)

from bop_toolkit_lib import inout, dataset_params
import bop_toolkit_lib.config as bop_config
import bop_toolkit_lib.misc as bop_misc

from utils.structs import AlignedBox2f, PinholePlaneCameraModel
from utils.misc import  warp_image

from tqdm import tqdm
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
import pickle

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

def read_colmap_points3D_txt(path):
    """Parse points3D.txt. Return dict point3d_id -> (xyz, rgb, error, track)"""
    pts = {}
    with open(path, 'r') as f:
        for line in f:
            if line.startswith('#') or line.strip() == '':
                continue
            parts = line.strip().split()
            # POINT3D_ID X Y Z R G B ERROR TRACK...
            pid = int(parts[0])
            x, y, z = map(float, parts[1:4])
            r, g, b = map(int, parts[4:7])
            err = float(parts[7])
            track = list(map(int, parts[8:]))  # sequence of IMAGE_ID, POINT2D_IDX
            pts[pid] = {'xyz': np.array([x, y, z], dtype=np.float64), 'rgb': (r,g,b), 'err': err, 'track': track}
    return pts

def quat_to_rotmat(q):
    # q = [qw, qx, qy, qz]
    qw, qx, qy, qz = q
    R = np.array([
        [1-2*(qy**2+qz**2),     2*(qx*qy- qz*qw),     2*(qx*qz+ qy*qw)],
        [2*(qx*qy+ qz*qw),   1-2*(qx**2+qz**2),     2*(qy*qz- qx*qw)],
        [2*(qx*qz- qy*qw),     2*(qy*qz+ qx*qw),   1-2*(qx**2+qy**2)]
    ], dtype=np.float64)
    return R

def umeyama_align(X_src, X_tgt, with_scale=True):
    """
    Solve for s, R, t that minimizes || X_tgt - (s R X_src + t) ||.
    X_src, X_tgt: (N,3)
    Return s, R, t
    """
    assert X_src.shape == X_tgt.shape and X_src.shape[0] >= 3
    mu_src = X_src.mean(axis=0)
    mu_tgt = X_tgt.mean(axis=0)
    Xs0 = X_src - mu_src
    Xt0 = X_tgt - mu_tgt
    cov = Xt0.T @ Xs0 / X_src.shape[0]
    from scipy.linalg import svd
    U, D, Vt = svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2,2] = -1
    R = U @ S @ Vt
    if with_scale:
        var_src = (Xs0**2).sum() / X_src.shape[0]
        s = np.trace(np.diag(D) @ S) / var_src
    else:
        s = 1.0
    t = mu_tgt - s * R @ mu_src
    return s, R, t

class InferOpts(NamedTuple):
    """Options that can be specified via the command line."""

    version: str
    repre_version: str
    object_dataset: str
    object_lids: Optional[List[int]] = None
    max_sym_disc_step: float = 0.01

    # Cropping options.
    crop: bool = True
    crop_rel_pad: float = 0.2
    crop_size: Tuple[int, int] = (420, 420)

    # Object instance options.
    use_detections: bool = True
    num_preds_factor: float = 1.0
    min_visibility: float = 0.1

    # Feature extraction options.
    extractor_name: str = "dinov2_vitl14"
    grid_cell_size: float = 1.0
    max_num_queries: int = 1000000

    # Feature matching options.
    match_template_type: str = "tfidf"
    match_top_n_templates: int = 5
    match_feat_matching_type: str = "cyclic_buddies"
    match_top_k_buddies: int = 300

    # PnP options.
    pnp_type: str = "opencv"
    pnp_ransac_iter: int = 1000
    pnp_required_ransac_conf: float = 0.99
    pnp_inlier_thresh: float = 10.0
    pnp_refine_lm: bool = True

    final_pose_type: str = "best_coarse"

    # Other options.
    save_estimates: bool = True
    vis_results: bool = True
    vis_corresp_top_n: int = 100
    vis_feat_map: bool = True
    vis_for_paper: bool = True
    debug: bool = True


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

# def global_reprojection_error(params, all_data_for_optimization):
#     """
#     计算所有帧的总重投影误差（残差向量）。
    
#     优化参数 params: [s, rx, ry, rz, tx, ty, tz]
#         s: 尺度因子 (1 维)
#         r: 模型到世界坐标系的旋转 (轴角表示, 3 维)
#         t: 模型到世界坐标系的平移 (3 维)
        
#     返回: 形状为 (N_total_points * 2,) 的残差向量 (u - u_obs, v - v_obs, ...)
#     """
    
#     s = params[0]
#     # print("params:", params)
#     r_vec = params[1:4]
#     t_vec = params[4:7]
    
#     # M -> W 变换矩阵
#     R_delta_m2w = axis_angle_to_rotmat(r_vec)
#     T_delta_m2w = np.eye(4, dtype=np.float32)
#     T_delta_m2w[:3, :3] = s * R_delta_m2w # 尺度只应用于旋转部分
#     T_delta_m2w[:3, 3] = s * t_vec
    
#     all_residuals = []
    
#     for data in all_data_for_optimization:
#         K = data['K']
#         T_W2C = data['T_w2c'] # COLMAP 相机位姿 (W -> C)
#         X_3d_C1 = data['X_3d'] # 3D 模型点 
#         q_2d_obs = data['q_2d'] # 2D 观测点 (像素)
        
#         # 1. 3D 点从 M 坐标系变换到 W 坐标系
#         # X_C1 (N, 3) -> X_W (N, 3)
#         X_3d_C1_h = np.concatenate([X_3d_C1, np.ones((X_3d_C1.shape[0], 1))], axis=1) # 齐次坐标 (N, 4)
#         X_3d_W_h = (T_delta_m2w @ X_3d_C1_h.T).T # (4, 4) @ (4, N) -> (4, N).T -> (N, 4)
#         X_3d_W = X_3d_W_h[:, :3] # (N, 3)
        
#         # 2. 3D 点从 W 坐标系变换到 C 坐标系
#         # X_W (N, 3) -> X_C (N, 3)
#         X_3d_W_h = X_3d_W_h # 已经有齐次坐标 (N, 4)
#         X_3d_C_h = (T_W2C @ X_3d_W_h.T).T
#         X_3d_C = X_3d_C_h[:, :3] # (N, 3)
        
#         # 3. 投影到 2D 图像平面
#         q_2d_pred = project_points(K, X_3d_C)
    
#         # print("mean depth:", np.mean(X_3d_C[:, 2]))
#         # print("min depth:", np.min(X_3d_C[:, 2]))
#         # print("max depth:", np.max(X_3d_C[:, 2]))
#         # print("sample projected points:", q_2d_pred[:5])
        
#         # 4. 计算残差
#         residuals = (q_2d_pred - q_2d_obs).flatten() # (N*2,)
#         # print(f"Residual mean={np.mean(np.abs(residuals)):.3e}, max={np.max(np.abs(residuals)):.3e}")
#         all_residuals.append(residuals)
        
#     return np.concatenate(all_residuals)
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

        # 仅前方点
        valid = X_C[:, 2] > 1e-6
        X_C = X_C[valid]
        q_2d_valid = q_2d[valid]

        # 投影到像素坐标
        uv = (K @ X_C.T).T
        uv = uv[:, :2] / uv[:, 2:3]

        # 残差
        res = (uv - q_2d_valid).reshape(-1)
        residuals.append(res)

    return np.concatenate(residuals)

import os
import cv2
import numpy as np

def visualize_2d_2d_correspondences(
    q_crop,
    q_orig,
    image_np_hwc,
    orig_image_np_hwc,
    base_frame,
    output_dir="/home/zjr/Tracker/foundpose/results/refine_model/wogua/2d-2d_test",
):
    """
    可视化裁剪图与原图上的 2D-2D 映射关系。

    Args:
        all_q_crop: list[np.ndarray]，每个元素形状 (N_i, 2)，裁剪图上的点坐标
        all_q_2d: list[np.ndarray]，每个元素形状 (N_i, 2)，映射回原图的点坐标
        image_np_hwc: np.ndarray, 当前裁剪图（float 0..1, RGB）
        orig_image_np_hwc: np.ndarray, 原始图像（float 0..1, RGB）
        image_path: str, 当前帧图像路径（用于生成文件名）
        output_dir: str, 输出保存目录
    """

    os.makedirs(output_dir, exist_ok=True)

    def draw_points_on(img, pts, color=(0, 0, 255), radius=2):
        """在图像上绘制点（越界点自动跳过）"""
        if pts is None or len(pts) == 0:
            return img
        pts_int = np.round(pts).astype(int)
        H, W = img.shape[:2]
        for (x, y) in pts_int:
            if 0 <= x < W and 0 <= y < H:
                cv2.circle(img, (int(x), int(y)), radius, color, -1)
        return img


    # 处理图像格式
    try:
        crop_vis = (image_np_hwc * 255.0).astype(np.uint8).copy()
    except Exception:
        crop_vis = (orig_image_np_hwc * 255.0).astype(np.uint8).copy()

    orig_vis = (orig_image_np_hwc * 255.0).astype(np.uint8).copy()

    # 转 BGR
    crop_vis_bgr = cv2.cvtColor(crop_vis, cv2.COLOR_RGB2BGR) if crop_vis.ndim == 3 else crop_vis
    orig_vis_bgr = cv2.cvtColor(orig_vis, cv2.COLOR_RGB2BGR) if orig_vis.ndim == 3 else orig_vis

    # 绘制红点（crop）和绿点（映射回原图）
    crop_vis_bgr = draw_points_on(crop_vis_bgr, q_crop, color=(0, 0, 255), radius=2)
    orig_vis_bgr = draw_points_on(orig_vis_bgr, q_orig, color=(0, 255, 0), radius=2)

    # 文件名
    # crop_fname = os.path.join(output_dir, f"{base_frame}_crop_pts.png")
    # orig_fname = os.path.join(output_dir, f"{base_frame}_orig_pts.png")
    combined_fname = os.path.join(output_dir, f"{base_frame}_combined.png")

    # # 保存单独图像
    # cv2.imwrite(crop_fname, crop_vis_bgr)
    # cv2.imwrite(orig_fname, orig_vis_bgr)

    # 生成并保存拼接图
    try:
        h_o, w_o = orig_vis_bgr.shape[:2]
        h_c, w_c = crop_vis_bgr.shape[:2]
        if h_c != h_o:
            scale = h_o / h_c
            new_w = int(w_c * scale)
            crop_rs = cv2.resize(crop_vis_bgr, (new_w, h_o), interpolation=cv2.INTER_AREA)
        else:
            crop_rs = crop_vis_bgr

        combined = np.concatenate([crop_rs, orig_vis_bgr], axis=1)
        cv2.imwrite(combined_fname, combined)
    except Exception as e:
        print(f"[WARN] Failed to create combined image for tpl{i}: {e}")

    print(f"[VIS] Saved combined:{combined_fname}")


# 定义一个用于存储对应关系的类/字典结构 3d-2d对应类
class CorrespondenceData(object):
    def __init__(self, frame_name, query_2d_pts, model_3d_pts, model_feat_ids, best_pose = None, camera_c2w = None):
        # 帧名
        self.frame_name = frame_name
        # 2D 图像点 (N, 2) numpy array
        self.query_2d_pts = query_2d_pts
        # 3D 模型点 (N, 3) numpy array
        self.model_3d_pts = model_3d_pts
        # 匹配到的 3D 模型特征点的 ID (N,) numpy array
        self.model_feat_ids = model_feat_ids
        self.best_pose = best_pose
        self.cam = camera_c2w

import numpy as np
import trimesh
import cv2

def read_colmap_images(images_txt_path):
    """
    解析 COLMAP images.txt，返回 dict: frame_name -> Tw2c (4x4)
    """
    Tw2c_dict = {}
    with open(images_txt_path, "r") as f:
        lines = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    i = 0
    while i < len(lines):
        parts = lines[i].split()
        if len(parts) < 10:
            i += 2  # 跳过 POINTS2D 行
            continue

        # 第一行: IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
        qw, qx, qy, qz = map(float, parts[1:5])
        tx, ty, tz = map(float, parts[5:8])
        image_name = parts[9]

        # 四元数 -> 旋转矩阵
        q = np.array([qw, qx, qy, qz], dtype=np.float32)
        R = quat2mat(q)  # 自定义函数
        t = np.array([tx, ty, tz], dtype=np.float32).reshape(3,1)

        # world -> camera
        Tw2c = np.eye(4, dtype=np.float32)
        Tw2c[:3,:3] = R
        Tw2c[:3,3:4] = t
        Tw2c_dict[image_name] = Tw2c

        i += 2  # 跳过下一行 POINTS2D
    return Tw2c_dict

import numpy as np

def read_colmap_camera_K(cameras_txt_path):
    """
    读取 COLMAP 的 cameras.txt，返回 {camera_id: K}
    """
    K_dict = {}
    with open(cameras_txt_path, "r") as f:
        lines = f.readlines()
    for line in lines:
        if line.startswith("#") or line.strip() == "":
            continue
        parts = line.strip().split()
        cam_id = int(parts[0])
        model = parts[1]
        width, height = map(int, parts[2:4])
        params = list(map(float, parts[4:]))

        if model == "PINHOLE":
            fx, fy, cx, cy = params
        elif model == "SIMPLE_PINHOLE":
            fx = fy = params[0]
            cx, cy = params[1:]
        elif model == "SIMPLE_RADIAL":
            fx = fy = params[0]
            cx, cy = params[1:3]
        elif model == "RADIAL":
            fx = fy = params[0]
            cx, cy = params[1:3]
        else:
            raise ValueError(f"Unsupported COLMAP camera model: {model}")

        K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0,  0,  1]
        ], dtype=np.float32)
        K_dict[cam_id] = K
    return K_dict


def quat2mat(q):
    """
    四元数转旋转矩阵
    q = [qw, qx, qy, qz]
    """
    qw, qx, qy, qz = q
    R = np.array([
        [1-2*qy**2-2*qz**2, 2*qx*qy-2*qz*qw, 2*qx*qz+2*qy*qw],
        [2*qx*qy+2*qz*qw, 1-2*qx**2-2*qz**2, 2*qy*qz-2*qx*qw],
        [2*qx*qz-2*qy*qw, 2*qy*qz+2*qx*qw, 1-2*qx**2-2*qy**2]
    ], dtype=np.float32)
    return R

def infer(opt_file_path, image_raw, mask_raw, camK, frame_name) -> CorrespondenceData:
    """
    使用 COLMAP 位姿直接生成部分顶点的 2D-3D 对应
    """
    orig_h, orig_w = image_raw.shape[:2]
    camK = np.array(camK).squeeze().astype(np.float32)
    fx, fy, cx, cy = camK[0,0], camK[1,1], camK[0,2], camK[1,2]

    # 读取 mesh
    mesh_path = "/home/zjr/Tracker/foundpose/datasets/china/mesh/mesh.obj"
    mesh = trimesh.load(mesh_path, force='mesh')
    verts = np.asarray(mesh.vertices, dtype=np.float32)
    num_verts = verts.shape[0]

    # 随机采样 80 个顶点
    sample_size = min(1000, num_verts)
    sampled_indices = np.random.choice(num_verts, size=sample_size, replace=False)
    verts_sampled = verts[sampled_indices]

    # 解析 COLMAP images.txt
    images_txt_path = "/home/zjr/Tracker/foundpose/datasets/china/sparse/0/images.txt"
    Tw2c_dict = read_colmap_images(images_txt_path)
    Tw2c = Tw2c_dict[frame_name]  # world -> camera

    # 投影 mesh 顶点到相机坐标系
    verts_h = np.hstack([verts_sampled, np.ones((sample_size,1), dtype=np.float32)])
    Xc = (Tw2c @ verts_h.T).T[:, :3]

    # 投影到图像平面
    u = (fx * Xc[:,0] / Xc[:,2]) + cx
    v = (fy * Xc[:,1] / Xc[:,2]) + cy
    q_2d = np.stack([u,v], axis=1)

    # 用 mask 过滤可见点
    if mask_raw is not None:
        u_int = np.clip(np.round(u).astype(int), 0, orig_w-1)
        v_int = np.clip(np.round(v).astype(int), 0, orig_h-1)
        mask_vals = mask_raw[v_int, u_int] > 0
        q_2d = q_2d[mask_vals]
        X_3d = verts_sampled[mask_vals]
        X_ids = sampled_indices[mask_vals]
    else:
        X_3d = verts_sampled
        X_ids = sampled_indices
    
    best_pose = {
        "R_m2c": Tw2c[:3, :3],       # 旋转矩阵
        "t_m2c": Tw2c[:3, 3].reshape(3)  # 平移向量
    }

    # 返回对应数据
    return CorrespondenceData(
        frame_name=frame_name,
        query_2d_pts=q_2d,
        model_3d_pts=X_3d,
        model_feat_ids=X_ids,
        best_pose=best_pose,
        camera_c2w=np.linalg.inv(Tw2c)
    )

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

# --- 主函数 ---

def main() -> None:
    import numpy as np
    # 路径配置
    base_dir = "/home/zjr/Tracker/foundpose/datasets/china"
    opts_path = "/home/zjr/Tracker/foundpose/configs/infer/china.json"
    rgb_dir = os.path.join(base_dir, "rgb")
    mask_dir = os.path.join(base_dir, "masks")
    colmap_dir = os.path.join(base_dir, "sparse/0")
    model_path = os.path.join(base_dir, "mesh/mesh.obj")
    output_vis_dir = "/home/zjr/Tracker/foundpose/results/refine_model/china"
    data_cache_path = os.path.join(os.path.dirname(opts_path), "cached_optimization_data_infer7.pkl")
    
    # --- 阶段 1: 数据预处理 (Infer / 提取对应点) ---
    all_corresps_data = [] 
    all_optimization_data = [] 
    data_loaded = False
    
    try:
        all_optimization_data, all_corresps_data = load_optimization_data(data_cache_path)
        data_loaded = True
    except (FileNotFoundError, EOFError, pickle.UnpicklingError, IsADirectoryError):
        logger.info("Cache file not found or corrupted. Starting data preprocessing...")
        
        # 实际 COLMAP 数据和 RGB/Mask 路径检查 (占位符环境中可能不存在)
        if not os.path.exists(rgb_dir):
             logger.error(f"Directory not found: {rgb_dir}. Aborting preprocessing.")
             return
             
        images_txt = os.path.join(colmap_dir, 'images.txt')
        images_colmap = read_colmap_images_txt(images_txt) # 假设返回字典

        rgb_files = sorted([f for f in os.listdir(rgb_dir) if f.endswith((".jpg", ".png"))])
        mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith(".png")])

        if len(rgb_files) != len(mask_files):
             logger.error("RGB and Mask file counts do not match. Aborting.")
             return

        for i, (rgb_file, mask_file) in enumerate(tqdm(
            zip(rgb_files, mask_files), 
            total=len(rgb_files), 
            desc="Preprocessing frames"
        )):
            frame_name = rgb_file
            image_path = os.path.join(rgb_dir, rgb_file)
            mask_path = os.path.join(mask_dir, mask_file)
            
            if frame_name not in images_colmap:
                logger.warn(f"Frame {frame_name} not found in COLMAP data. Skipping.")
                continue
            # # 使用简单的索引来模拟 COLMAP 检查，以适应占位符环境
            # if i >= 10: # 仅处理前10帧
            #     continue
            
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转为 RGB
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                 mask = np.zeros(image.shape[:2], dtype=np.uint8)

            # h, w = image.shape[:2]
            # scale = 1.5
            # fx = fy = scale * max(h, w)
            # cx = w / 2
            # cy = h / 2
            # camK = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
            cameras_txt_path = "/home/zjr/Tracker/foundpose/datasets/china/sparse/0/cameras.txt"
            K_dict = read_colmap_camera_K(cameras_txt_path)
            camK = K_dict[1]
            corresp_data = infer(opts_path, image, mask, camK, frame_name)
            
            if corresp_data is not None:
                all_corresps_data.append(corresp_data)

                img_entry = images_colmap[frame_name]
                R_colmap = quat_to_rotmat(img_entry['qvec'])  # W -> C Rotation
                t_colmap = img_entry['tvec'].astype(np.float32) # W -> C Translation
                
                # T_w2c = [R_colmap | t_colmap]
                T_w2c = np.eye(4, dtype=np.float32)
                T_w2c[:3, :3] = R_colmap
                T_w2c[:3, 3] = t_colmap 
                
                # save_dir = "/home/zjr/Tracker/foundpose/results/refine_model/wogua"
                # os.makedirs(save_dir, exist_ok=True)
                # pose_save_path = os.path.join(save_dir, "pose_new.txt")
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

    model_vertices = load_model_vertices(model_path)
    if model_vertices.size == 0:
        logger.error("Model vertices could not be loaded. Aborting.")
        return
    # output_corresp_dir = "/home/zjr/Tracker/foundpose/results/refine_model/china/2d-3d"
    # os.makedirs(output_corresp_dir, exist_ok=True)
    # for data in all_optimization_data:
    #     image_path = data['image_path']
    #     image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    #     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
    #     K = data['K']
    #     T_w2c = data['T_w2c']  # world -> camera
    #     X_3d = data['X_3d']    # 3D 点 (N,3)
        
    #     # 投影 3D 点到图像
    #     R = T_w2c[:3, :3]
    #     t = T_w2c[:3, 3].reshape(3,1)
    #     Xc = (R @ X_3d.T + t).T  # (N,3)
    #     # 投影到像素平面
    #     u = (K[0,0] * Xc[:,0] / Xc[:,2]) + K[0,2]
    #     v = (K[1,1] * Xc[:,1] / Xc[:,2]) + K[1,2]
    #     pts_2d = np.stack([u,v], axis=1).astype(int)
        
    #     # 可视化
    #     vis_img = image.copy()
    #     h, w = vis_img.shape[:2]
    #     for pt in pts_2d:
    #         x, y = pt
    #         if 0 <= x < w and 0 <= y < h:
    #             cv2.circle(vis_img, (x,y), radius=1, color=(255,0,0), thickness=-1)
        
    #     # 保存可视化结果
    #     frame_name = os.path.splitext(os.path.basename(image_path))[0]
    #     out_path = os.path.join(output_corresp_dir, f"{frame_name}_2d3d.png")
    #     cv2.imwrite(out_path, cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))
    
    # # 1. 初始化 T_M2W
    first_data = all_optimization_data[0]

    R_m2c = all_corresps_data[0].best_pose["R_m2c"]
    t_m2c = all_corresps_data[0].best_pose["t_m2c"]
    T_m2c = np.eye(4)
    T_m2c[:3, :3] = R_m2c
    T_m2c[:3, 3] = t_m2c
    t_m2c = T_m2c[:3, 3].reshape(3,1)
    # X_m = all_optimization_data[0]['X_3d']  # 模型坐标下的3D点 (N, 3)
    # X_c = (R_m2c @ X_m.T).T + t_m2c  # 结果是 (N, 3)
    
    # 确保 model_vertices 在优化前加载RGB
    from scipy.spatial.transform import Rotation
    T_w2c_first = np.eye(4)  # world -> camera
   
    # 固定小扰动参数
    angle_perturb_deg = np.array([55.0, -10.0, 25])  # xyz 旋转角度，单位为度
    angle_perturb = np.deg2rad(angle_perturb_deg)
    rot_perturb = Rotation.from_euler('xyz', angle_perturb).as_matrix()
    trans_perturb = np.array([50, -38, 10.0])  # 固定平移扰动
    scale_perturb = 0.005  # 固定缩放
    # 构造扰动后的 Tw2m
    Tw2m_init = np.eye(4, dtype=np.float32)
    Tw2m_init[:3, :3] = rot_perturb @ T_w2c_first[:3, :3]  # 旋转扰动乘在原旋转上
    Tw2m_init[:3, 3] = scale_perturb * (T_w2c_first[:3, 3] + trans_perturb)  # 平移+缩放
    # 4. 将模型顶点变换到模型空间
    model_vertices_m = (Tw2m_init[:3, :3] @ model_vertices.T).T + Tw2m_init[:3, 3]
    # model_vertices_m = model_vertices_c
    
    for data in all_optimization_data:
        R_w2m_init = Tw2m_init[:3, :3]
        t_w2m_init = Tw2m_init[:3, 3]
        X_3d = data['X_3d']  # shape (N, 3)
        # 应用与模型相同的初始扰动变换
        X_3d_m_perturbed = (R_w2m_init @ X_3d.T).T + t_w2m_init
        data['X_3d'] = X_3d_m_perturbed.astype(np.float32)
        X_3d = data['X_3d']
        # 可视化3d
        frame_name = data['frame_name']
        image_path = data['image_path']
        K = data['K']
        T_w2c = data['T_w2c']
        # === 加载图像 ===
        img = cv2.imread(image_path)
        if img is None:
            print(f"[WARN] 无法读取图像: {image_path}")
            continue
        h, w = img.shape[:2]
        X_W_h = np.concatenate([X_3d, np.ones((X_3d.shape[0], 1))], axis=1)
        X_C_h = (T_w2c @ X_W_h.T).T
        X_C = X_C_h[:, :3]
        # 只保留前方点
        valid = X_C[:, 2] > 1e-6
        X_C = X_C[valid]
        # 投影到像素坐标
        uv = (K @ X_C.T).T
        uv = uv[:, :2] / uv[:, 2:3]
        img_vis = img.copy()
        for pt in uv:
            u, v = int(round(pt[0])), int(round(pt[1]))
            if 0 <= u < w and 0 <= v < h:
                cv2.circle(img_vis, (u, v), 2, (0, 255, 0), -1)  # 绿色点
        # === 保存 ===
        output_dir = "/home/zjr/Tracker/foundpose/results/refine_model/china/model_trans"
        save_path = os.path.join(output_dir, f"{frame_name}_proj.jpg")
        cv2.imwrite(save_path, img_vis)
    
    T_delta_m2w = np.eye(4, dtype=np.float32)
    # T_delta_m2w [:3, :3] = s_init * R_init 
    # T_delta_m2w [:3, 3] = s_init * t_init
    R_init = T_delta_m2w[:3, :3]
    t_init = T_delta_m2w[:3, 3]
    s_init = 1.0
    
    T_gt = np.linalg.inv(Tw2m_init)
    print("gt:\n",T_gt)
    print("init:\n",T_delta_m2w)
    
    from scipy.spatial.transform import Rotation
    # 转换 R_init 到轴角表示 (用于优化库)
    r_vec_init = Rotation.from_matrix(R_init).as_rotvec()

    # 优化参数初始化: [s, rx, ry, rz, tx, ty, tz]
    initial_params = np.array([s_init, r_vec_init[0], r_vec_init[1], r_vec_init[2], 
                            t_init[0], t_init[1], t_init[2]], dtype=np.float64)

    # 2. 调用非线性最小二乘优化
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
    from scipy.optimize import least_squares
    result = least_squares(
        fun=global_reprojection_error, 
        x0=initial_params, 
        args=(all_optimization_data,), 
        method='lm', # Levenberg-Marquardt 算法
        verbose=2
        # loss='soft_l1' # 可选：使用鲁棒损失函数以处理异常值
    )
    
    # 3. 提取优化结果
    final_params = result.x
    # final_params = res_full.x
    s_final = final_params[0]
    R_delta_m2w_final = axis_angle_to_rotmat(final_params[1:4])
    t_delta_m2w_final = final_params[4:7]
    # s_final = s_init
    # R_delta_m2w_final = R_init
    # t_delta_m2w_final = t_init
    print(f"Optimized Scale (s): {s_final}")
    print(f"Optimized R_M2W:\n{R_delta_m2w_final}")
    print(f"Optimized t_M2W:\n{t_delta_m2w_final}")
    
    # # 4. 构造最终的W 矩阵
    T_delta_m2w_final = np.eye(4, dtype=np.float32)
    T_delta_m2w_final[:3, :3] = s_final * R_delta_m2w_final 
    T_delta_m2w_final[:3, 3] = s_final * t_delta_m2w_final
    
    # T_delta_m2w_final = T_gt
    
    logger.info("Starting visualization of optimized pose...")
    
    # --- 阶段 3: 最终可视化 (可选，使用最后一批次的 T_M2W) ---
    if T_delta_m2w_final is not None:
        logger.info("Starting final visualization using last optimized pose...")
        for data in tqdm(all_optimization_data, desc="Final Visualizing"):
            T_W2C = data['T_w2c']
            K = data['K']
            image_path = data['image_path']
            # X_m = data['X_3d']  # 模型坐标下的3D点 (N, 3)
            # X_c = (R_m2c @ X_m.T).T + t_m2c  # 结果是 (N, 3)
            # 使用最后一批优化的 T_M2W_final 进行全序列可视化
            visualize_projection(
                image_path=image_path,
                output_dir=output_vis_dir,
                K=K,
                T_W2C=T_W2C,
                T_M2W=T_delta_m2w_final,
                model_vertices=model_vertices_m
            )
        
    logger.info("All processes complete.")

def visualize_projection2(data_entry, output_vis_dir):
    """
    可视化：将 3D 模型点投影到原图像上，并显示对应的 2D 匹配点。

    Args:
        data_entry: dict, 包含以下键：
            - 'image_path': 原图路径
            - 'K': 相机内参 (3x3)
            - 'T_w2c': 世界到相机的位姿 (4x4)
            - 'q_2d': 检测得到的 2D 特征点 (N, 2)
            - 'X_3d': 模型的 3D 点 (N, 3)
        output_vis_dir: str, 可视化输出路径
    """

    os.makedirs(output_vis_dir, exist_ok=True)
    
    # === 1️⃣ 读取图像 ===
    image_path = data_entry["image_path"]
    frame_name = os.path.splitext(os.path.basename(image_path))[0]
    image = cv2.imread(image_path)
    if image is None:
        print(f"[WARN] 图像 {image_path} 读取失败，跳过可视化。")
        return

    h, w = image.shape[:2]

    # === 2️⃣ 准备数据 ===
    # camera_c2w = data_entry["K"]
    K = data_entry["K"]
    T_w2c = data_entry["T_w2c"]
    q_2d = data_entry["q_2d"]
    X_3d = data_entry["X_3d"]

    # === 3️⃣ 将 3D 点从世界坐标投影到像素坐标 ===
    # 世界坐标 -> 相机坐标
    X_homo = np.hstack([X_3d, np.ones((X_3d.shape[0], 1))])  # (N,4)
    X_cam = (T_w2c @ X_homo.T).T[:, :3]                      # (N,3)
    # R_m2c_orig = T_w2c["R_m2c"]
    # t_m2c_orig = T_w2c["t_m2c"].reshape(3,1)
    # X_cam = (R_m2c_orig @ X_3d.T) + t_m2c_orig  # (3, N)
    # X_cam = X_cam.T  # (N, 3)

    # 筛掉相机后方的点 (Z <= 0)
    valid_mask = X_cam[:, 2] > 0
    X_cam = X_cam[valid_mask]
    q_2d = q_2d[valid_mask]

    if X_cam.shape[0] == 0:
        print(f"[WARN] {frame_name}: 所有点都在相机后方，跳过可视化。")
        return

    # 相机坐标 -> 像素坐标
    proj = (K @ X_cam.T).T  # (N,3)
    proj[:, 0] /= proj[:, 2]
    proj[:, 1] /= proj[:, 2]
    proj_2d = proj[:, :2]
    # fx, fy = camera_c2w.f
    # cx, cy = camera_c2w.c
    # u = (fx * X_cam[:, 0] / X_cam[:, 2]) + cx
    # v = (fy * X_cam[:, 1] / X_cam[:, 2]) + cy
    # proj_2d = np.stack([u, v], axis=1).astype(int)

    # # === 4️⃣ 过滤不在图像范围内的点 ===
    # inside_mask = (
    #     (proj_2d[:, 0] >= 0)
    #     & (proj_2d[:, 0] < w)
    #     & (proj_2d[:, 1] >= 0)
    #     & (proj_2d[:, 1] < h)
    # )
    # proj_2d = proj_2d[inside_mask]
    # q_2d = q_2d[inside_mask]

    # === 5️⃣ 绘制匹配点与连线 ===
    vis_img = image.copy()

    for (u_pred, v_pred), (u_proj, v_proj) in zip(q_2d, proj_2d):
        # 绘制 2D 观测点 (蓝色)
        cv2.circle(vis_img, (int(u_pred), int(v_pred)), 4, (255, 0, 0), -1)
        # 绘制 3D 投影点 (绿色)
        cv2.circle(vis_img, (int(u_proj), int(v_proj)), 4, (0, 255, 0), -1)
        # 绘制连线 (黄色)
        cv2.line(
            vis_img,
            (int(u_pred), int(v_pred)),
            (int(u_proj), int(v_proj)),
            (0, 255, 255),
            1,
            lineType=cv2.LINE_AA
        )

    # === 6️⃣ 输出 ===
    output_vis_dir = output_vis_dir + "/2d_3d_valid"
    out_path = os.path.join(output_vis_dir, f"{frame_name}_proj_vis.jpg")
    cv2.imwrite(out_path, vis_img)
    print(f"[INFO] 可视化结果已保存到: {out_path}")

    return vis_img


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
    
    # 1. 图像加载
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
    # logger.info(f"Saved visualization to {output_path}")

if __name__ == "__main__":
    main()