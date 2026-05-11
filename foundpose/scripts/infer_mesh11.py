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
    feature_util,
    infer_pose_util,
    knn_util,
    misc as misc_util,
    pnp_util,
    projector_util,
    repre_util2 as repre_util,
    data_util,
    json_util, 
    logging,
    misc,
    template_util,
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

def read_cameras_binary(path):
    """
    Reads Colmap cameras.bin file and returns a dict mapping CAMERA_ID to camera info.
    Info includes: camera_id, model, width, height, params.
    """
    cameras = {}
    with open(path, "rb") as fid:
        # 读取相机数量
        import struct
        num_cameras, = struct.unpack('<Q', fid.read(8))
        for _ in range(num_cameras):
            camera_id, model_id, width, height = struct.unpack('<IiiQ', fid.read(20))
            
            # 简化处理：假设SIMPLE_RADIAL (模型ID=1) 且参数为 f, cx, cy, k
            # 不同的模型ID有不同的参数数量和顺序，这里需要根据实际情况调整！
            # 实际Colmap模型ID和参数数量：
            # SIMPLE_RADIAL (1): 4 params (f, cx, cy, k)
            
            # 读取模型名称长度
            model_name_length, = struct.unpack('<I', fid.read(4))
            # 读取模型名称
            model_name = fid.read(model_name_length).decode('utf-8')
            
            # 假设我们只关心 SIMPLE_RADIAL (4个参数) 或 SIMPLE_PINHOLE (3个参数)
            if model_name == 'SIMPLE_RADIAL':
                num_params = 4
            elif model_name == 'SIMPLE_PINHOLE':
                num_params = 3
            else:
                # 警告或跳过不支持的模型
                print(f"Warning: Camera model {model_name} not fully supported by simple reader.")
                num_params = 0 # 需要根据实际模型参数数量确定

            params = []
            for _ in range(num_params):
                # 读取双精度浮点数
                param, = struct.unpack('<d', fid.read(8)) 
                params.append(param)

            cameras[camera_id] = {
                'id': camera_id,
                'model': model_name,
                'width': width,
                'height': height,
                'params': np.array(params, dtype=np.float64)
            }
    return cameras

def quat_to_rotmat(q):
    # q = [qw, qx, qy, qz]
    qw, qx, qy, qz = q
    R = np.array([
        [1-2*(qy**2+qz**2),     2*(qx*qy- qz*qw),     2*(qx*qz+ qy*qw)],
        [2*(qx*qy+ qz*qw),   1-2*(qx**2+qz**2),     2*(qy*qz- qx*qw)],
        [2*(qx*qz- qy*qw),     2*(qy*qz+ qx*qw),   1-2*(qx**2+qy**2)]
    ], dtype=np.float64)
    return R


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

        # # 仅前方点
        # valid = X_C[:, 2] > 1e-6
        # X_C = X_C[valid]
        # q_2d_valid = q_2d[valid]

        # # 投影到像素坐标
        uv = (K @ X_C.T).T
        uv = uv[:, :2] / uv[:, 2:3]

        # # 残差
        # res = (uv - q_2d_valid).reshape(-1)
        # eps = 1e-6
        # Z = X_C[:, 2].copy()
        # mask = Z > eps
        # res = np.zeros((X_C.shape[0], 2), dtype=np.float32)
        # # 只对前方点计算残差
        # res[mask] = uv[mask] - q_2d[mask]
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

import omniglue
from omniglue import superpoint_extract, dino_extract
import tensorflow as tf
import torch
import numpy as np

MODEL_EXPORT_PATH = "/home/zjr/Tracker/omniglue"
OG_EXPORT_PATH = MODEL_EXPORT_PATH + "/models/og_export"
SP_EXPORT_PATH = MODEL_EXPORT_PATH + "/models/sp_v6"
DINO_EXPORT_PATH = MODEL_EXPORT_PATH + "/models/dinov2_vitb14_pretrain.pth"
DINO_FEATURE_DIM = 768 # (或你的 DINOv2 模型的维度)

# def unproject_pts_to_3d(pts_2d_np: np.ndarray, 
#                         depth_hw: torch.Tensor, 
#                         camera: PinholePlaneCameraModel, 
#                         T_model_from_camera: torch.Tensor,
#                         device: str) -> torch.Tensor:
#     """
#     将模板图像上的 2D 点 (u,v) 及其深度反投影到 3D 模型坐标。
#     """
#     if pts_2d_np.shape[0] == 0:
#         return torch.empty((0, 3), device=device, dtype=torch.float32)
#     depth_hw_cpu = depth_hw.cpu()
#     # 确保 pts_2d 是整数以便索引
#     pts_2d_int = np.round(pts_2d_np).astype(int)
    
#     # 过滤掉超出边界的点
#     h, w = depth_hw.shape
#     valid_mask = (pts_2d_int[:, 0] >= 0) & (pts_2d_int[:, 0] < w) & \
#                  (pts_2d_int[:, 1] >= 0) & (pts_2d_int[:, 1] < h)
    
#     pts_2d_valid = pts_2d_int[valid_mask]
    
#     if pts_2d_valid.shape[0] == 0:
#         return torch.empty((0, 3), device=device, dtype=torch.float32)

#     # 1. 获取深度
#     depth_values = depth_hw_cpu[pts_2d_valid[:, 1], pts_2d_valid[:, 0]] # (N,)
    
#     # 2. 过滤掉深度无效的点 (深度图通常用 0 表示无效)
#     valid_depth_mask = depth_values > 0
#     pts_2d_torch_temp = torch.as_tensor(pts_2d_valid, device='cpu')
#     pts_2d_valid_torch = pts_2d_torch_temp[valid_depth_mask]
#     depth_values = depth_values[valid_depth_mask]
#     if pts_2d_valid_torch.shape[0] == 0:
#         return torch.empty((0, 3), device=device, dtype=torch.float32)
        
#     pts_2d_torch = pts_2d_valid_torch.to(device, dtype=torch.float32)

#     # 3. 反投影到相机坐标系 (X_c, Y_c, Z_c)
#     # Z_c = d
#     Z_c = depth_values.to(device)
#     # X_c = (u - cx) * Z_c / fx
#     X_c = (pts_2d_torch[:, 0] - camera.c[0]) * Z_c / camera.f[0]
#     # Y_c = (v - cy) * Z_c / fy
#     Y_c = (pts_2d_torch[:, 1] - camera.c[1]) * Z_c / camera.f[1]
    
#     # 组合为 (N, 3) 齐次坐标 (N, 4)
#     pts_cam_space = torch.stack([X_c, Y_c, Z_c], dim=1) # (N, 3)
#     pts_cam_homogeneous = torch.cat(
#         [pts_cam_space, torch.ones((pts_cam_space.shape[0], 1), device=device)], dim=1
#     ) # (N, 4)

#     # 4. 转换到模型坐标系
#     # P_model = T_model_from_camera @ P_cam
#     pts_model_homogeneous = (T_model_from_camera @ pts_cam_homogeneous.T).T # (N, 4)
    
#     # 返回 (X, Y, Z)
#     return pts_model_homogeneous[:, :3]
def unproject_pts_to_3d(
        pts_2d_np: np.ndarray,
        depth_hw: torch.Tensor,
        camera: PinholePlaneCameraModel,
        T_model_from_camera: torch.Tensor,
        device: str):

    if pts_2d_np.shape[0] == 0:
        return torch.empty((0, 3), device=device), np.array([], dtype=int)

    depth_hw_cpu = depth_hw.cpu()
    pts_2d_int = np.round(pts_2d_np).astype(int)

    # 1. 越界过滤
    h, w = depth_hw.shape
    valid_mask_xy = (
        (pts_2d_int[:, 0] >= 0) & (pts_2d_int[:, 0] < w) &
        (pts_2d_int[:, 1] >= 0) & (pts_2d_int[:, 1] < h)
    )
    pts_2d_valid = pts_2d_int[valid_mask_xy]
    valid_indices_xy = np.where(valid_mask_xy)[0]

    if pts_2d_valid.shape[0] == 0:
        return torch.empty((0, 3), device=device), np.array([], dtype=int)

    # 2. 深度过滤
    depth_values = depth_hw_cpu[pts_2d_valid[:, 1], pts_2d_valid[:, 0]]
    valid_mask_depth = depth_values > 0

    pts_2d_valid = pts_2d_valid[valid_mask_depth]
    depth_values = depth_values[valid_mask_depth]
    valid_indices = valid_indices_xy[valid_mask_depth]

    if pts_2d_valid.shape[0] == 0:
        return torch.empty((0, 3), device=device), np.array([], dtype=int)

    # 3. 转到 GPU，做反投影
    pts_2d_torch = torch.as_tensor(pts_2d_valid, device=device, dtype=torch.float32)
    Z_c = depth_values.to(device)
    X_c = (pts_2d_torch[:, 0] - camera.c[0]) * Z_c / camera.f[0]
    Y_c = (pts_2d_torch[:, 1] - camera.c[1]) * Z_c / camera.f[1]

    pts_cam_space = torch.stack([X_c, Y_c, Z_c], dim=1)
    pts_cam_homo = torch.cat([pts_cam_space, torch.ones((pts_cam_space.shape[0], 1), device=device)], dim=1)

    pts_model_homo = (T_model_from_camera @ pts_cam_homo.T).T

    return pts_model_homo[:, :3], valid_indices

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


def infer(opt_file_path, image_raw, mask_raw, camK, frame_name) -> None:

    opts = config_util.load_opts_from_json(
        path=opt_file_path, 
        opts_types={"infer_opts": InferOpts}
    )["infer_opts"]
    
    extractor = feature_util.make_feature_extractor(opts.extractor_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    extractor.to(device)
    
    # +++ (新增：加载 OmniGlue 实例) +++
    og = omniglue.OmniGlue(
        og_export=OG_EXPORT_PATH,
        sp_export=SP_EXPORT_PATH,
        dino_export=DINO_EXPORT_PATH,
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # orig_w, orig_h = image_raw.size
    orig_h, orig_w = image_raw.shape[:2]
    
    camK = np.array(camK).squeeze().astype(np.float32)
    fx, fy, cx, cy = camK[0, 0], camK[1, 1], camK[0, 2], camK[1, 2]
    scene_cameras = {
        0: PinholePlaneCameraModel(
            width=orig_w,
            height=orig_h,
            f=(fx, fy),
            c=(cx, cy),
            T_world_from_eye=np.eye(4, dtype=np.float32)
        )
    }

    object_lids = opts.object_lids # 从配置里拿到要处理的 物体 ID 列表
    

    for object_lid in object_lids:

        version = opts.version
        if version == "":
            version = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        signature = misc.slugify(opts.object_dataset) + "_{}".format(version)
        output_dir = os.path.join(
            bop_config.output_path, "inference", signature, str(object_lid)
        )
        os.makedirs(output_dir, exist_ok=True)

        # Save parameters to a file.
        config_path = os.path.join(output_dir, "config.json")
        json_util.save_json(config_path, opts)

        # Create a pose evaluator.
        # pose_evaluator = eval_util.EvaluatorPose([object_lid])

        base_repre_dir = os.path.join(bop_config.output_path, "object_repre")
        repre_dir = repre_util.get_object_repre_dir_path(
            base_repre_dir, opts.version, opts.object_dataset, object_lid
        )
        repre = repre_util.load_object_repre(
            repre_dir=repre_dir,
            tensor_device=device,
        )

        # +++ (新增：为 OmniGlue 粗匹配构建全局 KNN 索引) +++
        # (假设 repre.feat_vectors 是 OmniGlue DINO 描述符)
        # logger.info("Building global KNN index for OmniGlue coarse matching...")
        # knn_coarse_matching = knn_util.KNN(k=5, metric="l2") # k=5 (可调)
        # knn_coarse_matching.fit(repre.feat_vectors)
        # logger.info("...Done.")
        # Build a kNN index from object feature vectors.
        visual_words_knn_index = None
        if opts.match_template_type == "tfidf":
            visual_words_knn_index = knn_util.KNN(
                k=repre.template_desc_opts.tfidf_knn_k,
                metric=repre.template_desc_opts.tfidf_knn_metric
            )
            visual_words_knn_index.fit(repre.feat_cluster_centroids)
        # Build per-template KNN index with features from that template.
        template_knn_indices = [] # 每个模板单独有一个 KNN 索引，避免全局混淆
        if opts.match_feat_matching_type == "cyclic_buddies":
            for template_id in range(len(repre.template_cameras_cam_from_model)):
                tpl_feat_mask = repre.feat_to_template_ids == template_id # 找出属于该模板的特征点。
                tpl_feat_ids = torch.nonzero(tpl_feat_mask).flatten() # 提取这些特征点的索引
                template_feats = repre.feat_vectors[tpl_feat_ids] # 拿到该模板的特征向量集合
                # Build knn index for object features.
                template_knn_index = knn_util.KNN(k=1, metric="l2")
                template_knn_index.fit(template_feats.cpu())
                template_knn_indices.append(template_knn_index)
        
        # Perform inference on each selected image.
        sample = data_util.prepare_sample3(image_raw, scene_cameras)
        # Camera parameters.
        orig_camera_c2w = sample.camera
        orig_image_size = (
            orig_camera_c2w.width,
            orig_camera_c2w.height,
        )

        # Get info about object instances for which we want to estimate pose. 是一个列表，每个元素对应 当前图像中一个对象实例
        instances = infer_pose_util.get_instances_from_mask(
            obj_id=object_lid,
            mask=mask_raw,
        )
        if len(instances) == 0:
            logger.info("No object instance, skipping.")
            continue

        # Estimate pose for each object instance.
        for inst_j, instance in enumerate(instances):
            # Get the input image. 获取图像和实例掩码 获取该实例的 modal mask（可见部分）和 amodal bounding box
            orig_image_np_hwc = sample.image.astype(np.float32)/255.0

            # Get the modal mask and amodal bounding box of the instance.
            orig_mask_modal = instance["input_mask_modal"]
            orig_box_amodal = AlignedBox2f(
                left=instance["input_box_amodal"][0],
                top=instance["input_box_amodal"][1],
                right=instance["input_box_amodal"][2],
                bottom=instance["input_box_amodal"][3],
            )

            # Optional cropping. 把图像裁剪成只包含目标物体的区域
            if not opts.crop:
                camera_c2w = orig_camera_c2w
                image_np_hwc = orig_image_np_hwc
                mask_modal = orig_mask_modal
                box_amodal = orig_box_amodal
            else:
                # Get box for cropping.
                crop_box = misc_util.calc_crop_box(
                    box=orig_box_amodal,
                    make_square=True,
                )

                # Construct a virtual camera focused on the crop. 裁剪图像后，原来的相机参数（orig_camera_c2w）不再对应裁剪后的图像， 所以需要构建一个 虚拟相机，让裁剪后的图像在这个相机坐标系下看起来和原始图像一致。
                crop_camera_model_c2w = misc_util.construct_crop_camera(
                    box=crop_box,
                    camera_model_c2w=orig_camera_c2w,
                    viewport_size=opts.crop_size,
                    viewport_rel_pad=opts.crop_rel_pad,
                )

                # Map images to the virtual camera.
                interpolation = (
                    cv2.INTER_AREA
                    if crop_box.width >= crop_camera_model_c2w.width
                    else cv2.INTER_LINEAR
                )
                image_np_hwc = warp_image(
                    src_camera=orig_camera_c2w,
                    dst_camera=crop_camera_model_c2w,
                    src_image=orig_image_np_hwc,
                    interpolation=interpolation,
                )
                mask_modal = warp_image(
                    src_camera=orig_camera_c2w,
                    dst_camera=crop_camera_model_c2w,
                    src_image=orig_mask_modal,
                    interpolation=cv2.INTER_NEAREST,
                )

                # Recalculate the object bounding box (it changed if we constructed the virtual camera).
                ys, xs = mask_modal.nonzero()
                box = np.array(misc_util.calc_2d_box(xs, ys))
                box_amodal = AlignedBox2f(
                    left=box[0],
                    top=box[1],
                    right=box[2],
                    bottom=box[3],
                )

                # The virtual camera is becoming the main camera.
                camera_c2w = crop_camera_model_c2w

            # Extract feature map from the crop. 将图像转为张量并提取特征
            if opts.crop:
                grid_size = opts.crop_size
            else:
                grid_size = orig_image_size
            grid_points = feature_util.generate_grid_points(
                grid_size=grid_size,
                cell_size=opts.grid_cell_size,
            )
            grid_points = grid_points.to(device)
            
            image_tensor_chw = array_to_tensor(image_np_hwc).to(torch.float32).permute(2,0,1).to(device)
            image_tensor_bchw = image_tensor_chw.unsqueeze(0)
            logger.info("Extracting query features...")
            # Extract feature map from the crop. 将图像转为张量并提取特征
            image_tensor_chw = array_to_tensor(image_np_hwc).to(torch.float32).permute(2,0,1).to(device)
            image_tensor_bchw = image_tensor_chw.unsqueeze(0)
            extractor_output = extractor(image_tensor_bchw)
            feature_map_chw = extractor_output["feature_maps"][0] # 取第一个层的特征图，形状 (C_feat, H_feat, W_feat)。

            # Keep only points inside the object mask. 保留物体内部的网格点
            mask_modal_tensor = array_to_tensor(mask_modal).to(device)
            query_points = feature_util.filter_points_by_mask(
                grid_points, mask_modal_tensor
            )
            # Subsample query points if we have too many. 随机下采样过多的查询点
            if query_points.shape[0] > opts.max_num_queries:
                perm = torch.randperm(query_points.shape[0])
                query_points = query_points[perm[: opts.max_num_queries]]
                msg = (
                    "Randomly sumbsampled queries "
                    f"({perm.shape[0]} -> {query_points.shape[0]}))"
                )
            # 1. 提取 SP 和 DINO
            query_features = feature_util.sample_feature_map_at_points(
                feature_map_chw=feature_map_chw,
                points=query_points,
                image_size=(image_np_hwc.shape[1], image_np_hwc.shape[0]),
            ).contiguous()

            # Potentially project features to a PCA space.
            if (
                query_features.shape[1] != repre.feat_vectors.shape[1]
                and len(repre.feat_raw_projectors) != 0
            ):
                query_features_proj = projector_util.project_features(
                    feat_vectors=query_features,
                    projectors=repre.feat_raw_projectors,
                ).contiguous()

                _c, _h, _w = feature_map_chw.shape
                feature_map_chw_proj = (
                    projector_util.project_features(
                        feat_vectors=feature_map_chw.permute(1, 2, 0).view(-1, _c),
                        projectors=repre.feat_raw_projectors,
                    )
                    .view(_h, _w, -1)
                    .permute(2, 0, 1)
                )
            else:
                query_features_proj = query_features
                feature_map_chw_proj = feature_map_chw
            
            template_ids, template_scores = template_util.template_matching(
                query_features=query_features_proj,
                object_repre=repre,
                top_n_templates=1,
                matching_type=opts.match_template_type,
                visual_words_knn_index=visual_words_knn_index,
            )
            best_template_id = template_ids.squeeze(0).cpu().numpy()
            logger.info(f"Best template voted: {best_template_id}")
            
            # +++ (新增：OmniGlue 精匹配) +++
            logger.info("Performing Fine Matching (OmniGlue)...")

            # 1. 加载模板图像
            template_image_chw = repre.templates[best_template_id]
            template_image_hwc_np = template_image_chw.permute(1, 2, 0).cpu().numpy() # (H, W, C) uint8
            
            query_img_np_u8 = (image_np_hwc * 255).astype(np.uint8)
            sp_features_query = og.sp_extract(query_img_np_u8)
            dino_features_query_dense = og.dino_extract(query_img_np_u8)
            # 2. 运行 OmniGlue.FindMatches
            # (FindMatches 内部会自己提取 SP 和 DINO)
            match_kp_query, match_kp_template, match_conf = og.FindMatches(
                query_img_np_u8, template_image_hwc_np
            )
            
            if match_kp_query.shape[0] < 6: # PnP 需要至少 N 个点
                logger.info(f"OmniGlue found only {match_kp_query.shape[0]} matches. Skipping.")
                continue
                
            logger.info(f"OmniGlue found {match_kp_query.shape[0]} matches.")
            
            # +++ (新增：使用 mask_modal 过滤 2D 查询点) +++
            # 1. 将 match_kp_query 转换为整数像素坐标
            pts_2d_int = np.round(match_kp_query).astype(int)
            # 2. 越界/前景过滤
            # mask_modal 是 (H, W) numpy array, 1 表示前景，0 表示背景
            h, w = mask_modal.shape
            # 检查是否在图像边界内 并且 在 mask_modal 内部 (值为 1)
            # 注意：pts_2d_int[:, 0] 是 x 坐标 (对应宽度 w)，pts_2d_int[:, 1] 是 y 坐标 (对应高度 h)
            valid_mask_in_mask = (
                (pts_2d_int[:, 0] >= 0) & (pts_2d_int[:, 0] < w) &
                (pts_2d_int[:, 1] >= 0) & (pts_2d_int[:, 1] < h)
            )
            # 只有在边界内的点才能检查 mask 值
            pts_2d_valid_check = pts_2d_int[valid_mask_in_mask]
            valid_mask_in_mask[valid_mask_in_mask] = mask_modal[pts_2d_valid_check[:, 1], pts_2d_valid_check[:, 0]] > 0
            # 3. 过滤匹配点
            match_kp_query = match_kp_query[valid_mask_in_mask]
            match_kp_template = match_kp_template[valid_mask_in_mask]
            match_conf = match_conf[valid_mask_in_mask]
            logger.info(f"After mask filtering, {match_kp_query.shape[0]} matches remain.")
            
            # +++ (新增：2D-2D -> 3D-2D 转换) +++
            logger.info("Converting 2D-2D matches to 3D-2D...")
            # 1. 加载模板的深度图和相机
            # (需要加载 metadata.json，就像 generate_repre 中一样)
            metadata_path = os.path.join(
                bop_config.output_path, "templates", opts.version, 
                opts.object_dataset, str(object_lid), "metadata.json"
            )
            metadata = json_util.load_json(metadata_path)
            
            depth_path = metadata[best_template_id]["depth_map_path"]
            depth_image = inout.load_depth(depth_path)
            depth_image_hw = array_to_tensor(depth_image).to(torch.float32).to(device)
            
            template_cam = repre.template_cameras_cam_from_model[best_template_id]
            T_model_from_camera = torch.linalg.inv(template_cam.extrinsics) # 获取 T_m_c

            # 2. 反向投影
            model_3d_pts_torch, valid_indices = unproject_pts_to_3d(
                match_kp_template,
                depth_image_hw,
                template_cam,
                T_model_from_camera,
                device
            )
            
            num_matches = match_kp_query.shape[0]
            # 1. 创建一个 (N, N) 的单位矩阵
            # 因为 match_kp_query 和 match_kp_template 已经是 (N, 2) 的匹配对
            # 我们传入一个单位矩阵，告诉 visualize_matches 将 kp0[i] 与 kp1[i] 连接
            match_identity_matrix = np.eye(num_matches) 
            logger.info(f"Visualizing {num_matches} matches...")
            # 2. 调用你的可视化函数
            viz = visualize_matches( 
                image0=query_img_np_u8,       # 查询图像
                image1=template_image_hwc_np, # 模板图像
                kp0=match_kp_query,           # 查询点 (N, 2)
                kp1=match_kp_template,        # 模板点 (N, 2)
                match_matrix=match_identity_matrix, # 匹配矩阵
                show_keypoints=True,
                highlight_unmatched=False, # 我们只传入了匹配点，所以设为 False
                title=f"OmniGlue Matches (Tpl {best_template_id}): {num_matches}",
                line_width=1,
                circle_radius=3,
                circle_thickness=1
            )
            # 3. 保存图像
            # (确保 infer 函数能接收到 frame_name 变量，以便生成唯一文件名)
            # output_filename = f"/home/zjr/Tracker/foundpose/results/refine_model/plane/2d-2d-omni/match_viz_{frame_name}_tpl{best_template_id}.png"
            frame_clean = os.path.splitext(frame_name)[0]
            save_dir = "/home/zjr/Tracker/foundpose/results/refine_model/plane/2d-2d-omni"
            os.makedirs(save_dir, exist_ok=True)
            output_filename = os.path.join(
                save_dir,
                f"match_viz_{frame_clean}.png"
            )
            import matplotlib.pyplot as plt
            plt.figure(figsize=(20, 10), dpi=150)
            plt.axis("off")
            plt.imshow(viz)
            plt.savefig(output_filename, bbox_inches='tight', pad_inches=0)
            plt.close() # 关闭图像，防止在循环中消耗过多内存
            logger.info(f"Saved match visualization to {output_filename}")
            
            # 3. 过滤掉 unproject 失败的点
            # (假设 unproject_pts_to_3d 返回的点数可能少于输入)
            # (这是一个简化的假设，一个鲁棒的实现会返回索引)
            # (为简单起见，我们假设所有点都成功了)
            query_2d_pts_torch = torch.tensor(match_kp_query, device=device, dtype=torch.float32)

            # if model_3d_pts_torch.shape[0] != query_2d_pts_torch.shape[0]:
            #      logger.warning("Mismatch in 3D/2D points after unprojection. Skipping.")
            #      # (一个鲁棒的实现会在这里使用 unproject 返回的索引来过滤 query_2d_pts_torch)
            #      continue
                 
            if model_3d_pts_torch.shape[0] < 6:
                logger.info(f"Only {model_3d_pts_torch.shape[0]} valid 3D points. Skipping.")
                continue
            
            query_2d_pts_torch = torch.tensor(match_kp_query[valid_indices],
                                  device=device, dtype=torch.float32)
            match_conf_torch = torch.tensor(match_conf[valid_indices],
                                            device=device, dtype=torch.float32)

            # 确保匹配数量一致
            assert model_3d_pts_torch.shape[0] == query_2d_pts_torch.shape[0]
            # +++ (新增：格式化为 corresp) +++
            corresp_final = {
                "template_id": best_template_id,
                "coord_2d": query_2d_pts_torch,
                "coord_3d": model_3d_pts_torch,
                "coord_conf": torch.tensor(match_conf, device=device, dtype=torch.float32),
                # (nn_vertex_ids 在这个流程中意义不大，但 PnP 可能需要)
                "nn_vertex_ids": torch.arange(model_3d_pts_torch.shape[0], device=device) 
            }
            corresp = [corresp_final] # PnP 循环期望一个列表
            
                
            # Estimate coarse poses from corespondences.
            coarse_poses = []
            for corresp_id, corresp_curr in enumerate(corresp): # 遍历之前生成的每组 2D-3D 对应关系 corresp
                corresp_curr["coord_3d"] = corresp_curr["coord_3d"]
                # We need at least 3 correspondences for P3P.
                num_corresp = len(corresp_curr["coord_2d"])
                if num_corresp < 6:
                    logger.info(f"Only {num_corresp} correspondences, skipping.")
                    continue
                (
                    coarse_pose_success,
                    R_m2c_coarse,
                    t_m2c_coarse,
                    inliers_coarse,
                    quality_coarse,
                ) = pnp_util.estimate_pose( # 调用 estimate_pose 进行 PnP 求解
                    corresp=corresp_curr,
                    camera_c2w=camera_c2w,
                    pnp_type=opts.pnp_type,
                    pnp_ransac_iter=opts.pnp_ransac_iter,
                    pnp_inlier_thresh=opts.pnp_inlier_thresh,
                    pnp_required_ransac_conf=opts.pnp_required_ransac_conf,
                    pnp_refine_lm=opts.pnp_refine_lm,
                )

                if coarse_pose_success:
                    coarse_poses.append(
                        {
                            "type": "coarse",
                            "R_m2c": R_m2c_coarse,
                            "t_m2c": t_m2c_coarse,
                            "corresp_id": corresp_id,
                            "quality": quality_coarse,
                            "inliers": inliers_coarse,
                        }
                    )
            # Find the best coarse pose.
            best_coarse_quality = None
            best_coarse_pose_id = 0
            for coarse_pose_id, pose in enumerate(coarse_poses):
                if (
                    best_coarse_quality is None
                    or pose["quality"] > best_coarse_quality
                ):
                    best_coarse_pose_id = coarse_pose_id
                    best_coarse_quality = pose["quality"]
            best_corresp_id = coarse_poses[best_coarse_pose_id]["corresp_id"]
            corresp_final = corresp[best_corresp_id]
          
            # 2D 观测点 (N, 2)
            q_2d = corresp_final["coord_2d"].cpu().numpy()
            # 3D 模型点坐标 (N, 3)
            X_3d = corresp_final["coord_3d"].cpu().numpy()
            # 3D 模型特征点 ID (N,)
            X_ids = corresp_final["nn_vertex_ids"].cpu().numpy()
            corresp_scores = corresp_final["coord_conf"]
            
            best_pose = coarse_poses[best_coarse_pose_id]
            # # 测试pose
            # # # ------------------ 测试 ------------------
            # 读取模型
            R_m2c = best_pose["R_m2c"]  # (3,3)
            t_m2c = best_pose["t_m2c"].reshape(3)  # (3,)
            model_tpath = "/home/zjr/Tracker/foundpose/datasets/plane/models/plane_mm.obj"
            import trimesh
            model_mesh = trimesh.load(model_tpath, force='mesh')
            verts = np.asarray(model_mesh.vertices, dtype=np.float32)
            # 投影 3D 点到 2D
            Xc = (R_m2c @ verts.T).T + t_m2c  # (N,3)
            Xc_h = np.hstack([Xc, np.ones((Xc.shape[0],1))])
            verts_world = (crop_camera_model_c2w.T_world_from_eye @ Xc_h.T).T[:, :3]  # (N,3)
            Xc_orig = (np.linalg.inv(orig_camera_c2w.T_world_from_eye) @ np.hstack([verts_world, np.ones((verts_world.shape[0],1))]).T).T[:, :3]
            fx, fy = orig_camera_c2w.f
            cx, cy = orig_camera_c2w.c
            u = (fx * Xc_orig[:,0] / Xc_orig[:,2]) + cx
            v = (fy * Xc_orig[:,1] / Xc_orig[:,2]) + cy
            pts_2d = np.stack([u,v], axis=1).astype(int)
            # 复制一份原图
            # vis_img = image_np_hwc.astype(np.uint8).copy()
            vis_img = image_raw.astype(np.uint8).copy()
            overlay = vis_img.copy()  # 复制一层用于绘制
            alpha = 0.4  # 透明度 0~1
            for f in model_mesh.faces:
                p1, p2, p3 = pts_2d[f]
                cv2.line(overlay, tuple(p1), tuple(p2), (150,25,120), 1)
                cv2.line(overlay, tuple(p2), tuple(p3), (150,25,120), 1)
                cv2.line(overlay, tuple(p3), tuple(p1), (150,25,120), 1)
            # 将 overlay 叠加回原图
            cv2.addWeighted(overlay, alpha, vis_img, 1-alpha, 0, vis_img)
            out_dir_vis = os.path.join(output_dir, "infer_rgb")
            out_dir_vis = "/home/zjr/Tracker/foundpose/results/refine_model/plane/pose_es"
            os.makedirs(out_dir_vis, exist_ok=True)
            # 保存时用 frame_name 来保持一致的文件名
            out_path = os.path.join(out_dir_vis, f"{frame_name}.png")
            cv2.imwrite(out_path, vis_img) 
            
            R_m2c_crop = best_pose["R_m2c"]
            t_m2c_crop = best_pose["t_m2c"].reshape(3, 1)
            # ========== 一步式：把裁剪相机坐标系的位姿变换为原相机坐标系下 ==========
            R_c2w_crop = crop_camera_model_c2w.T_world_from_eye[:3, :3]
            t_c2w_crop = crop_camera_model_c2w.T_world_from_eye[:3, 3:4]
            R_c2w_orig = orig_camera_c2w.T_world_from_eye[:3, :3]
            t_c2w_orig = orig_camera_c2w.T_world_from_eye[:3, 3:4]
            # 世界 -> 原相机
            R_w2c_orig = R_c2w_orig.T
            t_w2c_orig = -R_c2w_orig.T @ t_c2w_orig
            # 合成模型 -> 原相机的变换
            R_m2c_orig = R_w2c_orig @ R_c2w_crop @ R_m2c_crop
            t_m2c_orig = R_w2c_orig @ (R_c2w_crop @ t_m2c_crop + t_c2w_crop) + t_w2c_orig
            best_pose["R_m2c"] = R_m2c_orig
            best_pose["t_m2c"] = t_m2c_orig.reshape(3)
            
            
            
            # --- 若启用 crop，相机不同，需要把裁剪图坐标还原到原图坐标 ---
            if opts.crop:
                Z_3d = X_3d[:, 2]  # 每个点的深度（与 q_2d 一一对应）

                q_2d_in_orig = warp_points_perspective(
                    src_camera=camera_c2w,          # 裁剪后的虚拟相机
                    dst_camera=orig_camera_c2w,     # 原始相机
                    src_points=q_2d,                # 裁剪图上的 2D 坐标
                    src_depths=Z_3d                 # 对应深度
                )
                q_crop = q_2d
                q_2d = q_2d_in_orig  # 替换为映射到原图坐标的点
                
            final_q_2d = q_2d
            final_X_3d = X_3d
            final_X_ids = X_ids
            all_scores = corresp_scores
            
            #重复3d-2d对应剪枝
            unique_ids = np.unique(final_X_ids)
            best_indices = []
            for uid in unique_ids:
                # 找到所有对应该 3D 点的索引
                idxs = np.where(final_X_ids == uid)[0]
                if len(idxs) == 1:
                    # 只有一个对应，直接保留
                    best_indices.append(idxs[0])
                else:
                    # 多个对应，选择置信度最高的
                    best_idx = idxs[np.argmax(all_scores[idxs].cpu().numpy())]
                    best_indices.append(best_idx)
            best_indices = np.array(best_indices)
            final_q_2d = final_q_2d[best_indices]
            final_X_3d = final_X_3d[best_indices]
            final_X_ids = final_X_ids[best_indices]
            all_scores = all_scores[best_indices]
            
            
            # ========== (新增) 基于 best_pose 的 3D→2D 投影误差筛选 ==========
            logger.info("Filtering correspondences using reprojection error...")
            # 提取相机内参
            fx, fy, cx, cy = camK[0,0], camK[1,1], camK[0,2], camK[1,2]
            # f = np.array(camera.f)         # [fx, fy]
            # c = np.array(camera.c)         # [cx, cy]
            # width, height = camera.width, camera.height
            # 提取姿态
            R = best_pose["R_m2c"]     # 3x3
            t = best_pose["t_m2c"].reshape(3, 1)  # 3x1
            # 将 3D 点变换到相机坐标
            pts3d_cam = (R @ final_X_3d.T + t).T   # Nx3
            # 投影回 2D
            proj2d = np.zeros((pts3d_cam.shape[0], 2))
            proj2d[:, 0] = (pts3d_cam[:, 0] / pts3d_cam[:, 2]) * fx + cx
            proj2d[:, 1] = (pts3d_cam[:, 1] / pts3d_cam[:, 2]) * fy + cy
            # 计算欧氏距离误差
            errors = np.linalg.norm(proj2d - final_q_2d, axis=1)
            # 选择一个阈值（你可以调整）
            REPROJ_THRESH = 5.0   # 像素阈值，可调
            valid_mask = errors < REPROJ_THRESH
            num_removed = np.sum(~valid_mask)
            logger.info(f"Reprojection filtering: removed {num_removed} correspondences")
            # 更新所有对应关系
            final_q_2d   = final_q_2d[valid_mask]
            final_X_3d   = final_X_3d[valid_mask]
            final_X_ids  = final_X_ids[valid_mask]
            all_scores   = all_scores[valid_mask]

            q_2d = final_q_2d  # <--- (修正 1: 使用过滤后的原图坐标)
            m_3d = final_X_3d  # 3D点 (已经过重投影过滤)
            # 2. 获取对应模板相机
            template_id = int(corresp_final["template_id"]) if "template_id" in corresp_final else 0
            camera = repre.template_cameras_cam_from_model[template_id]
            # 3. 提取相机参数
            f = np.array(camera.f)      # [fx, fy]
            c = np.array(camera.c)      # [cx, cy]
            width, height = camera.width, camera.height
            T_world_from_eye = np.array(camera.T_world_from_eye)  # 4x4
            # 4. 将3D点从模型坐标系转换到相机坐标系
            ones = np.ones((m_3d.shape[0], 1))
            pts_h = np.hstack([m_3d, ones])   # Nx4
            pts_cam = (np.linalg.inv(T_world_from_eye) @ pts_h.T).T[:, :3]  # Nx3
            # 5. 投影到模板图像 (针孔模型)
            proj_2d = np.zeros((pts_cam.shape[0], 2))
            proj_2d[:, 0] = (pts_cam[:, 0] / pts_cam[:, 2]) * f[0] + c[0]
            proj_2d[:, 1] = (pts_cam[:, 1] / pts_cam[:, 2]) * f[1] + c[1]
            template_dir = "/home/zjr/Tracker/foundpose/results/templates/v1/plane/1"
            template_path = os.path.join(template_dir, f"rgb/template_{template_id:04d}.png")
            if os.path.exists(template_path):
                template_img = cv2.imread(template_path)
                template_img = cv2.cvtColor(template_img, cv2.COLOR_BGR2RGB)
            else:
                print(f"Warning: template image {template_path} not found. Using blank image instead.")
                template_img = np.zeros((height, width, 3), dtype=np.uint8)
            proj_points = proj_2d.astype(int)
            q_points = q_2d.astype(int)

            # 7. 创建保存目录
            save_dir = "/home/zjr/Tracker/foundpose/results/refine_model/plane/2d-3d-template"
            frame_dir = os.path.join(save_dir, frame_clean)
            os.makedirs(frame_dir, exist_ok=True)

            # 8. 分批绘制，每批大约 num_points_per_batch 个点
            num_points = q_points.shape[0]
            num_batches = 5
            num_points_per_batch = (num_points + num_batches - 1) // num_batches

            for batch_idx in range(num_batches):
                start_idx = batch_idx * num_points_per_batch
                end_idx = min((batch_idx + 1) * num_points_per_batch, num_points)
                if start_idx >= end_idx:
                    continue  # 防止最后一批为空
                batch_q = q_points[start_idx:end_idx]
                batch_proj = proj_points[start_idx:end_idx]

                # 拷贝图像，避免覆盖
                # <--- (修正 2: 使用 'orig_image_np_hwc' (原始图像))
                image_with_2d = (orig_image_np_hwc * 255).astype(np.uint8).copy() 
                template_img_copy = template_img.copy()

                # 绘制点和编号
                for idx, (pt2d, pt3d) in enumerate(zip(batch_q, batch_proj), start=start_idx):
                    # 绘制点
                    cv2.circle(image_with_2d, tuple(pt2d), 1, (0, 255, 0), -1)      # 原图2D点，绿色
                    cv2.circle(template_img_copy, tuple(pt3d), 1, (255, 0, 0), -1)    # 模板投影点，红色

                    # 绘制编号
                    cv2.putText(image_with_2d, str(idx), (pt2d[0]+1, pt2d[1]-1),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 255, 0), 1, cv2.LINE_AA)
                    cv2.putText(template_img_copy, str(idx), (pt3d[0]+3, pt3d[1]-3),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.2, (255, 0, 0), 1, cv2.LINE_AA)
                # --- (新增) 修复：在 hstack 之前统一高度 ---
                h1, w1 = image_with_2d.shape[:2]
                h2, w2 = template_img_copy.shape[:2]
                
                # 找到最大高度
                max_height = max(h1, h2)

                # 1. 创建一个空白的（黑色的）背景，高度为 max_height，宽度为原图宽度
                # 假设通道数都为 3
                img1_padded = np.zeros((max_height, w1, 3), dtype=np.uint8)
                # 将原图复制到这个背景的左上角
                img1_padded[0:h1, :, :] = image_with_2d

                # 2. 为模板图创建同样的背景
                img2_padded = np.zeros((max_height, w2, 3), dtype=np.uint8)
                # 将模板图复制到这个背景的左上角
                img2_padded[0:h2, :, :] = template_img_copy
                # 拼接
                combined_img = np.hstack([img1_padded, img2_padded])

                # 保存每批图像到 frame_name 目录
                save_path = os.path.join(frame_dir, f"{best_corresp_id:04d}_batch{batch_idx+1}.png")
                cv2.imwrite(save_path, cv2.cvtColor(combined_img, cv2.COLOR_RGB2BGR))
            logger.info(f"Saved comparison batch to {save_path}")
            
            # 创建并返回此帧的对应数据
            return CorrespondenceData(
                frame_name=frame_name,
                query_2d_pts=final_q_2d,
                model_3d_pts=final_X_3d,
                model_feat_ids=final_X_ids,
                best_pose = best_pose,
                camera_c2w= orig_camera_c2w
            )

# --- 主函数 ---

def main() -> None:
    
    # 路径配置
    base_dir = "/home/zjr/Tracker/foundpose/datasets/plane"
    opts_path = "/home/zjr/Tracker/foundpose/configs/infer/plane.json"
    rgb_dir = os.path.join(base_dir, "rgb")
    mask_dir = os.path.join(base_dir, "masks")
    colmap_dir = os.path.join(base_dir, "sparse/0")
    model_path = os.path.join(base_dir, "models/plane_mm.obj")
    output_vis_dir = "/home/zjr/Tracker/foundpose/results/refine_model/plane"
    data_cache_path = os.path.join(os.path.dirname(opts_path), "cached_optimization_data_infer11_plane.pkl")
    
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

        # for i, (rgb_file, mask_file) in enumerate(tqdm(
        #     zip(rgb_files, mask_files), 
        #     total=len(rgb_files), 
        #     desc="Preprocessing frames"
        # )):
        for i, (rgb_file, mask_file) in enumerate(zip(rgb_files, mask_files)):
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

            h, w = image.shape[:2]
            scale = 1.5
            fx = fy = scale * max(h, w)
            cx = w / 2
            cy = h / 2
            camK = np.array([[1089.0690450152401, 0, 360], [0, 1089.0690450152401, 640], [0, 0, 1]], dtype=np.float32)
            # camK = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
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
                
                save_dir = "/home/zjr/Tracker/foundpose/results/refine_model/plane"
                os.makedirs(save_dir, exist_ok=True)
                pose_save_path = os.path.join(save_dir, "pose_new.txt")
                R_m2c = corresp_data.best_pose["R_m2c"]
                t_m2c = corresp_data.best_pose["t_m2c"]
                # # 写入 best_pose 和 T_w2c
                # with open(pose_save_path, "a") as f:
                #     f.write(f"\n=== Frame: {frame_name} ===\n")
                #     f.write("R_pose:\n")
                #     np.savetxt(f, R_m2c, fmt="%.6f")
                #     f.write("R_col:\n")
                #     np.savetxt(f, R_colmap, fmt="%.6f")
                #     f.write("t_pose:\n")
                #     np.savetxt(f, t_m2c, fmt="%.6f")
                #     f.write("t_col:\n")
                #     np.savetxt(f, t_colmap, fmt="%.6f")
                #     f.write("\n")
                    
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

    # 确保 model_vertices 在优化前加载
    model_vertices = load_model_vertices(model_path)
    if model_vertices.size == 0:
        logger.error("Model vertices could not be loaded. Aborting.")
        return
            
    # # 1. 初始化 T_M2W
    first_data = all_optimization_data[0]

    R_m2c = all_corresps_data[31].best_pose["R_m2c"]
    t_m2c = all_corresps_data[31].best_pose["t_m2c"]
    T_m2c = np.eye(4)
    T_m2c[:3, :3] = R_m2c
    T_m2c[:3, 3] = t_m2c
    
    T_w2c = all_optimization_data[31]['T_w2c']
    # print(T_m2c)
    T_c2w = np.linalg.inv(T_w2c)
    T_m2w_init = T_c2w @ T_m2c
    R_init = T_m2w_init[:3, :3]
    t_init = T_m2w_init[:3, 3]
    s_init = 1.0 # 可能得改
            
    # s_init = np.linalg.norm(R_init) / np.sqrt(3)  # 近似尺度
    # R_init /= s_init
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
    R_M2W_final = axis_angle_to_rotmat(final_params[1:4])
    t_M2W_final = final_params[4:7]
    # s_final = s_init
    # R_M2W_final = R_init
    # t_M2W_final = t_init
    print(f"Optimized Scale (s): {s_final}")
    print(f"Optimized R_M2W:\n{R_M2W_final}")
    print(f"Optimized t_M2W:\n{t_M2W_final}")
    
    # 4. 构造最终的 T_M2W 矩阵
    T_M2W_final = np.eye(4, dtype=np.float32)
    T_M2W_final[:3, :3] = s_final * R_M2W_final 
    T_M2W_final[:3, 3] = s_final * t_M2W_final
    
    logger.info("Starting visualization of optimized pose...")
    model_faces = load_model_faces(model_path)
    # --- 阶段 3: 最终可视化 (可选，使用最后一批次的 T_M2W) ---
    if T_M2W_final is not None:
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
                T_M2W=T_M2W_final,
                model_vertices=model_vertices
            )
    # if T_M2W_final is not None:
    #     logger.info("Starting final wireframe visualization...")
        
    #     for i, data in enumerate(all_optimization_data):
    #         T_W2C = data['T_w2c']
    #         K = data['K']
    #         image_path = data['image_path']
    #         # 获取局部最佳姿态
    #         best_pose = all_corresps_data[i].best_pose 
            
    #         visualize_projection_multi_pose( # <<< 调用新的多姿态渲染函数
    #             image_path=image_path,
    #             output_dir=output_vis_dir,
    #             K=K,
    #             T_W2C=T_W2C,
    #             T_M2W_final=T_M2W_final,
    #             best_pose=best_pose,
    #             model_vertices=model_vertices,
    #             model_faces=model_faces # <<< 传入面片索引
    #         )
        
    logger.info("All processes complete.")


def load_model_faces(model_path: str) -> np.ndarray:
    """
    使用 trimesh 库加载模型面片/三角形索引 (Faces/Indices)。
    """
    if not os.path.exists(model_path):
        logger.warning(f"Model file not found at {model_path}. Returning placeholder faces.")
        # 返回占位符以确保代码能够运行
        return np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
        
    try:
        # 加载整个网格
        import trimesh
        mesh = trimesh.load_mesh(model_path, process=False)
        # 确保加载的是一个网格，并且有面片数据
        if isinstance(mesh, trimesh.Trimesh):
            faces = np.array(mesh.faces, dtype=np.int32)
            logger.info(f"Successfully loaded {faces.shape[0]} faces using trimesh.")
            return faces
        else:
            logger.error(f"File {model_path} loaded but is not a Trimesh object or has no faces.")
            return np.array([], dtype=np.int32).reshape(0, 3)

    except Exception as e:
        logger.error(f"Error loading model faces with trimesh: {e}")
        return np.array([], dtype=np.int32).reshape(0, 3)
    
def load_model_vertices(model_path: str) -> np.ndarray:
    """加载模型文件并返回顶点坐标 (N, 3)。支持多种格式（如 .obj, .ply）。"""
    try:
        # 导入 trimesh (假设前面已经导入)
        import trimesh
        
        # 使用 trimesh 加载网格，它能自动处理文件类型
        mesh = trimesh.load_mesh(model_path, process=False)
        
        # 确保加载的是一个 Trimesh 对象（有顶点）
        if not isinstance(mesh, trimesh.Trimesh) or not mesh.vertices.size:
            raise ValueError(f"File {model_path} loaded but is not a Trimesh object or has no vertices.")
            
        vertices = np.array(mesh.vertices, dtype=np.float32)
        logger.info(f"Successfully loaded {vertices.shape[0]} vertices using trimesh.")
        
        # ... (如果你需要随机采样点的代码)
        
        return vertices
    
    except Exception as e:
        logger.error(f"Error loading model {model_path} with trimesh: {e}")
        return np.array([], dtype=np.float32).reshape(0, 3) # 返回 Nx3 形状

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
    
def warp_points_perspective(src_camera, dst_camera, src_points, src_depths):
    """
    将 2D 点从源相机坐标系 (src_camera) 映射到目标相机坐标系 (dst_camera)，
    采用完整透视几何（使用真实深度 Z）。

    Args:
        src_camera: 源相机模型（例如裁剪相机）
        dst_camera: 目标相机模型（例如原图相机）
        src_points: 2D 点数组，形状 (N, 2)，像素坐标
        src_depths: 每个点的深度（相机坐标系下 Z 值），形状 (N,)

    Returns:
        dst_points: 转换后的 2D 点坐标，形状 (N, 2)
    """
    assert src_points.shape[0] == src_depths.shape[0], "点与深度数量不匹配"

    # --- Step 1. 从源像素坐标反投影到相机坐标系 ---
    K_src = get_camera_matrix(src_camera)           # (3,3)
    K_src_inv = np.linalg.inv(K_src)
    src_pts_h = np.hstack([src_points, np.ones((len(src_points), 1))])  # (N,3)
    # 单位深度下的归一化坐标
    rays_src = (K_src_inv @ src_pts_h.T).T          # (N,3)
    # 乘以真实深度，得到3D坐标（相机坐标系下）
    pts_src_eye = rays_src * src_depths[:, None]

    # --- Step 2. 从源相机坐标 -> 世界坐标 ---
    T_w_from_src = src_camera.T_world_from_eye      # (4,4)
    pts_src_eye_h = np.hstack([pts_src_eye, np.ones((len(src_points), 1))])
    pts_world_h = (T_w_from_src @ pts_src_eye_h.T).T

    # --- Step 3. 从世界坐标 -> 目标相机坐标 ---
    T_dst_from_w = np.linalg.inv(dst_camera.T_world_from_eye)
    pts_dst_eye_h = (T_dst_from_w @ pts_world_h.T).T
    pts_dst_eye = pts_dst_eye_h[:, :3]

    # --- Step 4. 目标相机投影到像素平面 ---
    K_dst = get_camera_matrix(dst_camera)
    pts_dst_img_h = (K_dst @ pts_dst_eye.T).T
    pts_dst_img = pts_dst_img_h[:, :2] / pts_dst_img_h[:, 2:3]

    return pts_dst_img.astype(np.float32)

def get_camera_matrix(camera_model):
    """从相机模型中获取 3x3 内参矩阵 K。"""
    fx, fy = camera_model.f
    cx, cy = camera_model.c
    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=np.float32)
    return K


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

def visualize_matches(
    image0: np.ndarray,
    image1: np.ndarray,
    kp0: np.ndarray,
    kp1: np.ndarray,
    match_matrix: np.ndarray,
    match_labels: Optional[np.ndarray] = None,
    show_keypoints: bool = False,
    highlight_unmatched: bool = False,
    title: Optional[str] = None,
    line_width: int = 1,
    circle_radius: int = 4,
    circle_thickness: int = 2,
    rng: Optional['np.random.Generator'] = None,
):
  """Generates visualization of keypoints and matches for two images.

  Stacks image0 and image1 horizontally. In case the two images have different
  heights, scales image1 (and its keypoints) to match image0's height. Note
  that keypoints must be in (x, y) format, NOT (row, col). If match_matrix
  includes unmatched dustbins, the dustbins will be removed before visualizing
  matches.

  Args:
    image0: (H, W, 3) array containing image0 contents.
    image1: (H, W, 3) array containing image1 contents.
    kp0: (N, 2) array where each row represents (x, y) coordinates of keypoints
      in image0.
    kp1: (M, 2) array, where each row represents (x, y) coordinates of keypoints
      in image1.
    match_matrix: (N, M) binary array, where values are non-zero for keypoint
      indices making up a match.
    match_labels: (N, M) binary array, where values are non-zero for keypoint
      indices making up a ground-truth match. When None, matches from
      'match_matrix' are colored randomly. Otherwise, matches from
      'match_matrix' are colored according to accuracy (compared to labels).
    show_keypoints: if True, all image0 and image1 keypoints (including
      unmatched ones) are visualized.
    highlight_unmatched: if True, highlights unmatched keypoints in blue.
    title: if not None, adds title text to top left of visualization.
    line_width: width of correspondence line, in pixels.
    circle_radius: radius of keypoint circles, if visualized.
    circle_thickness: thickness of keypoint circles, if visualized.
    rng: np random number generator to generate the line colors.

  Returns:
    Numpy array of image0 and image1 side-by-side, with lines between matches
    according to match_matrix. If show_keypoints is True, keypoints from both
    images are also visualized.
  """
  # initialize RNG
  if rng is None:
    rng = np.random.default_rng()

  # Make copy of input param that may be modified in this function.
  kp1 = np.copy(kp1)

  # Detect unmatched dustbins.
  has_unmatched_dustbins = (match_matrix.shape[0] == kp0.shape[0] + 1) and (
      match_matrix.shape[1] == kp1.shape[0] + 1
  )

  # If necessary, resize image1 so that the pair can be stacked horizontally.
  height0 = image0.shape[0]
  height1 = image1.shape[0]
  if height0 != height1:
    scale_factor = height0 / height1
    if scale_factor <= 1.0:
      interp_method = cv2.INTER_AREA
    else:
      interp_method = cv2.INTER_LINEAR
    new_dim1 = (int(image1.shape[1] * scale_factor), height0)
    image1 = cv2.resize(image1, new_dim1, interpolation=interp_method)
    kp1 *= scale_factor

  # Create side-by-side image and add lines for all matches.
  viz = cv2.hconcat([image0, image1])
  w0 = image0.shape[1]
  matches = np.argwhere(
      match_matrix[:-1, :-1] if has_unmatched_dustbins else match_matrix
  )
  for match in matches:
    pt0 = (int(kp0[match[0], 0]), int(kp0[match[0], 1]))
    pt1 = (int(kp1[match[1], 0] + w0), int(kp1[match[1], 1]))
    if match_labels is None:
      color = tuple(rng.integers(0, 255, size=3).tolist())
    else:
      if match_labels[match[0], match[1]]:
        color = (0, 255, 0)
      else:
        color = (255, 0, 0)
    cv2.line(viz, pt0, pt1, color, line_width)

  # Optionally, add circles to output image to represent each keypoint.
  if show_keypoints:
    for i in range(np.shape(kp0)[0]):
      kp = kp0[i, :]
      if highlight_unmatched and has_unmatched_dustbins and match_matrix[i, -1]:
        cv2.circle(
            viz,
            tuple(kp.astype(np.int32).tolist()),
            circle_radius,
            (255, 0, 0),
            circle_thickness,
        )
      else:
        cv2.circle(
            viz,
            tuple(kp.astype(np.int32).tolist()),
            circle_radius,
            (0, 0, 255),
            circle_thickness,
        )
    for j in range(np.shape(kp1)[0]):
      kp = kp1[j, :]
      kp[0] += w0
      if highlight_unmatched and has_unmatched_dustbins and match_matrix[-1, j]:
        cv2.circle(
            viz,
            tuple(kp.astype(np.int32).tolist()),
            circle_radius,
            (255, 0, 0),
            circle_thickness,
        )
      else:
        cv2.circle(
            viz,
            tuple(kp.astype(np.int32).tolist()),
            circle_radius,
            (0, 0, 255),
            circle_thickness,
        )
  if title is not None:
    viz = cv2.putText(
        viz,
        title,
        (5, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )
  return viz


# --- 新增的 Wireframe 渲染函数 ---
def visualize_wireframe(
    image_bgr: np.ndarray,
    K: np.ndarray,
    T_M2C: np.ndarray,
    model_vertices: np.ndarray,
    model_faces: np.ndarray,
    color: tuple, # BGR 格式
    thickness: int = 1,
    alpha: float = 0.3
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

# --- 核心可视化函数更新 ---
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
        color=(0, 255, 0), # BGR: 绿色 (优化后全局姿态)
        thickness=1
    )
        
    # 5. 保存结果
    base_name = os.path.basename(image_path)
    output_dir = output_dir + "/vis" # 避免重复添加
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"projected_wireframe_{base_name}")
    cv2.imwrite(output_path, output_image)
    logger.info(f"Saved wireframe visualization to {output_path}")

if __name__ == "__main__":
    main()