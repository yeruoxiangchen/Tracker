#!/usr/bin/env python3

"""Infers pose from objects."""

import datetime

import sys
REPO_PATH = "/home/zjr/Tracker/CoarseModel"
BOP_TOOLKIT_PATH = "/home/zjr/Tracker/CoarseModel/external/bop_toolkit"

sys.path.insert(0, REPO_PATH)
sys.path.insert(0, BOP_TOOLKIT_PATH)  # 添加这一行

import os
import cv2
import numpy as np
import torch

from script import vis_util

from utils import (
    corresp_util, feature_util, pnp_util, repre_util, json_util,
    knn_util, misc as misc_util, projector_util, data_util, logging, misc,
)
from utils.structs import AlignedBox2f, PinholePlaneCameraModel
from utils.misc import  warp_image, array_to_tensor
from .config import AppConfig, InferOpts
from typing import List, Optional, Any

from core.util import get_instances_from_mask

logger: logging.Logger = logging.get_logger()

import time

class CorrespondenceData(object):
    def __init__(self, frame_name, query_2d_pts, model_3d_pts, model_feat_ids, best_pose = None, camera_c2w = None):
        self.frame_name = frame_name
        self.query_2d_pts = query_2d_pts
        self.model_3d_pts = model_3d_pts
        self.model_feat_ids = model_feat_ids
        self.best_pose = best_pose
        self.cam = camera_c2w
        
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

def warp_points_perspective(src_camera, dst_camera, src_points, src_depths):
    """
    将 2D 点从源相机坐标系 (src_camera) 映射到目标相机坐标系 (dst_camera)，
    采用完整透视几何（使用真实深度 Z）。
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

def visualize_correspondences(
    data: CorrespondenceData,
    template_id: int,
    image_raw: np.ndarray,
    repre,  # Object representation 包含模板相机信息
    template_base_dir: str,
    output_dir: str,
    num_batches: int = 5
):
    """
    将提取的2D-3D对应关系投影到模板图像上，并与原图对比保存。
    """
    frame_id = os.path.splitext(data.frame_name)[0]
    frame_dir = os.path.join(output_dir, frame_id)
    os.makedirs(frame_dir, exist_ok=True)

    # 1. 准备原始图像 (确保是 uint8 格式)
    if image_raw.max() <= 1.0:
        vis_base_img = (image_raw * 255).astype(np.uint8)
    else:
        vis_base_img = image_raw.astype(np.uint8).copy()

    # 2. 获取匹配时使用的模板信息
    # 注意：CorrespondenceData 中需要包含匹配到的 template_id
    # 如果你的 data 对象里没存，需要从推理过程中传出来
    # template_id = getattr(data, 'template_id', 0) 
    tpl_camera = repre.template_cameras_cam_from_model[template_id]
    
    # 3. 加载模板图像
    tpl_path = os.path.join(template_base_dir, f"rgb/template_{template_id:04d}.png")
    if os.path.exists(tpl_path):
        tpl_img = cv2.imread(tpl_path)
        tpl_img = cv2.cvtColor(tpl_img, cv2.COLOR_BGR2RGB)
    else:
        logger.warning(f"Template image not found: {tpl_path}")
        tpl_img = np.zeros((tpl_camera.height, tpl_camera.width, 3), dtype=np.uint8)

    # 4. 计算 3D 点在模板上的投影
    # T_world_from_eye 在模板相机里实际上是 T_cam_from_model 的逆
    T_c2m = np.linalg.inv(tpl_camera.T_world_from_eye)
    pts_h = np.hstack([data.model_3d_pts, np.ones((data.model_3d_pts.shape[0], 1))])
    pts_cam = (T_c2m @ pts_h.T).T[:, :3]
    
    f = np.array(tpl_camera .f)         # [fx, fy]
    c = np.array(tpl_camera .c)         # [cx, cy]
    proj_2d = np.zeros((pts_cam.shape[0], 2))
    proj_2d[:, 0] = (pts_cam[:, 0] / pts_cam[:, 2]) * f[0] + c[0]
    proj_2d[:, 1] = (pts_cam[:, 1] / pts_cam[:, 2]) * f[1] + c[1]
    
    proj_points = proj_2d.astype(int)
    query_points = data.query_2d_pts.astype(int)

    # 5. 分批次绘制并保存
    num_points = query_points.shape[0]
    batch_size = (num_points + num_batches - 1) // num_batches

    for b in range(num_batches):
        start = b * batch_size
        end = min((b + 1) * batch_size, num_points)
        if start >= end: break

        img_left = vis_base_img.copy()
        img_right = tpl_img.copy()

        for i in range(start, end):
            p2d = tuple(query_points[i])
            p3d = tuple(proj_points[i])
            color_2d = (0, 255, 0)  # 绿色
            color_3d = (255, 0, 0)  # 蓝色 (RGB)

            # 绘制圆点
            cv2.circle(img_left, p2d, 2, color_2d, -1)
            cv2.circle(img_right, p3d, 2, color_3d, -1)

            # 绘制索引编号
            cv2.putText(img_left, str(i), (p2d[0]+2, p2d[1]-2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, color_2d, 1)
            cv2.putText(img_right, str(i), (p3d[0]+2, p3d[1]-2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, color_3d, 1)
        # === 统一高度，便于 hstack ===
        h_left = img_left.shape[0]
        h_right = img_right.shape[0]

        if h_left != h_right:
            scale = h_left / h_right
            new_w = int(img_right.shape[1] * scale)
            img_right = cv2.resize(img_right, (new_w, h_left))
        # 水平拼接并保存
        combined = np.hstack([img_left, img_right])
        save_path = os.path.join(frame_dir, f"batch_{b+1:02d}.png")
        cv2.imwrite(save_path, cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))

    logger.info(f"Visualized {num_points} points in {num_batches} batches to {frame_dir}")

def visualize_correspondences2(
    data: CorrespondenceData,
    template_id: int,
    image_raw: np.ndarray,
    repre,
    template_base_dir: str,
    output_dir: str,
    mode: str = "grid",   # "all_raw" 或 "grid"
    grid_size: int = 4,
    file_name: str = "all_raw",
):
    frame_id = os.path.splitext(data.frame_name)[0]
    frame_dir = os.path.join(output_dir, frame_id)
    os.makedirs(frame_dir, exist_ok=True)

    # 1. 准备图像
    vis_base_img = (image_raw * 255).astype(np.uint8) if image_raw.max() <= 1.0 else image_raw.astype(np.uint8).copy()
    
    # 2. 加载模板
    tpl_camera = repre.template_cameras_cam_from_model[template_id]
    tpl_path = os.path.join(template_base_dir, f"rgb/template_{template_id:04d}.png")
    if os.path.exists(tpl_path):
        tpl_img = cv2.imread(tpl_path)
        tpl_img = cv2.cvtColor(tpl_img, cv2.COLOR_BGR2RGB)
    else:
        tpl_img = np.zeros((tpl_camera.height, tpl_camera.width, 3), dtype=np.uint8)

    # 3. 计算 3D 点在模板上的投影
    T_c2m = np.linalg.inv(tpl_camera.T_world_from_eye)
    pts_h = np.hstack([data.model_3d_pts, np.ones((data.model_3d_pts.shape[0], 1))])
    pts_cam = (T_c2m @ pts_h.T).T[:, :3]
    
    f, c = np.array(tpl_camera.f), np.array(tpl_camera.c)
    proj_points = np.zeros((pts_cam.shape[0], 2))
    proj_points[:, 0] = (pts_cam[:, 0] / pts_cam[:, 2]) * f[0] + c[0]
    proj_points[:, 1] = (pts_cam[:, 1] / pts_cam[:, 2]) * f[1] + c[1]
    
    query_points = data.query_2d_pts.astype(int)
    proj_points = proj_points.astype(int)

    # 4. 绘图逻辑
    def save_combined(indices, filename):
        if len(indices) == 0: return
        img_l, img_r = vis_base_img.copy(), tpl_img.copy()
        for idx in indices:
            p2d, p3d = tuple(query_points[idx]), tuple(proj_points[idx])
            # 绘制点和索引
            cv2.circle(img_l, p2d, 1, (0, 255, 0), -1)
            cv2.circle(img_r, p3d, 1, (255, 0, 0), -1)
            cv2.putText(img_l, str(idx), (p2d[0]+3, p2d[1]-3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            cv2.putText(img_r, str(idx), (p3d[0]+3, p3d[1]-3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
        
        # 拼接
        if img_l.shape[0] != img_r.shape[0]:
            img_r = cv2.resize(img_r, (int(img_r.shape[1] * (img_l.shape[0]/img_r.shape[0])), img_l.shape[0]))
        combined = np.hstack([img_l, img_r])
        cv2.imwrite(os.path.join(frame_dir, filename), cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))

    # 5. 执行不同的可视化策略
    if mode == "all_raw":
        # 输出全量点，命名为 00_all_correspondences.png
        save_combined(range(len(query_points)), f"00_{file_name}.png")
    
    else:
        # 计算网格归属
        min_p, max_p = np.min(query_points, axis=0), np.max(query_points, axis=0)
        dx, dy = max(max_p[0]-min_p[0], 1), max(max_p[1]-min_p[1], 1)
        
        cell_x = np.clip(((query_points[:, 0] - min_p[0]) / dx * grid_size).astype(int), 0, grid_size-1)
        cell_y = np.clip(((query_points[:, 1] - min_p[1]) / dy * grid_size).astype(int), 0, grid_size-1)
        cell_indices = cell_y * grid_size + cell_x

        # 遍历每个网格分别保存
        unique_cells = np.unique(cell_indices)
        for cell_id in unique_cells:
            target_indices = np.where(cell_indices == cell_id)[0]
            save_combined(target_indices, f"grid_cell_{cell_id:02d}.png")
    
def build_crop_view(
    orig_image: np.ndarray,
    orig_mask: np.ndarray,
    orig_camera: PinholePlaneCameraModel,
    box_amodal: AlignedBox2f,
    crop_size: tuple[int, int],
    crop_rel_pad: float,
):
    """
    根据 amodal box 构建裁剪视图和虚拟相机
    返回：
        image_crop, mask_crop, crop_camera, crop_box, box_amodal_crop
    """
    # 1. crop box
    crop_box = misc_util.calc_crop_box(
        box=box_amodal,
        make_square=True,
    )

    # 2. virtual camera
    crop_camera = misc_util.construct_crop_camera(
        box=crop_box,
        camera_model_c2w=orig_camera,
        viewport_size=crop_size,
        viewport_rel_pad=crop_rel_pad,
    )

    # 3. warp image & mask
    interpolation = (
        cv2.INTER_AREA
        if crop_box.width >= crop_camera.width
        else cv2.INTER_LINEAR
    )

    image_crop = warp_image(
        src_camera=orig_camera,
        dst_camera=crop_camera,
        src_image=orig_image,
        interpolation=interpolation,
    )

    mask_crop = warp_image(
        src_camera=orig_camera,
        dst_camera=crop_camera,
        src_image=orig_mask,
        interpolation=cv2.INTER_NEAREST,
    )

    # 4. recompute bounding box in crop
    ys, xs = mask_crop.nonzero()
    box = np.array(misc_util.calc_2d_box(xs, ys))
    box_amodal_crop = AlignedBox2f(
        left=box[0], top=box[1], right=box[2], bottom=box[3]
    )

    return image_crop, mask_crop, crop_camera, crop_box, box_amodal_crop

def recover_pose_and_points_to_orig(
    best_pose: dict,
    q_2d: np.ndarray,
    X_3d: np.ndarray,
    crop_camera: PinholePlaneCameraModel,
    orig_camera: PinholePlaneCameraModel,
    use_crop: bool,
):
    """
    1. 将模型位姿从裁剪相机坐标系恢复到原始相机坐标系
    2. 若启用 crop，将裁剪图上的 2D 点恢复到原图坐标
    返回：
        best_pose_orig, q_2d_orig, q_2d_crop(optional)
    """
    # ---------- 1. pose: crop -> orig ----------
    R_m2c_crop = best_pose["R_m2c"]
    t_m2c_crop = best_pose["t_m2c"].reshape(3, 1)

    R_c2w_crop = crop_camera.T_world_from_eye[:3, :3]
    t_c2w_crop = crop_camera.T_world_from_eye[:3, 3:4]

    R_c2w_orig = orig_camera.T_world_from_eye[:3, :3]
    t_c2w_orig = orig_camera.T_world_from_eye[:3, 3:4]
    
    # world -> original camera
    R_w2c_orig = R_c2w_orig.T
    t_w2c_orig = -R_w2c_orig @ t_c2w_orig
    # model -> original camera
    R_m2c_orig = R_w2c_orig @ R_c2w_crop @ R_m2c_crop
    t_m2c_orig = (
        R_w2c_orig @ (R_c2w_crop @ t_m2c_crop + t_c2w_crop)
        + t_w2c_orig
    )
    best_pose_orig = best_pose.copy()
    best_pose_orig["R_m2c"] = R_m2c_orig
    best_pose_orig["t_m2c"] = t_m2c_orig.reshape(3)

    # ---------- 2. points: crop -> orig ----------
    q_2d_crop = None
    q_2d_orig = q_2d
    
    Z_3d = X_3d[:, 2]  # depth for each 2D point
    q_2d_orig = warp_points_perspective(
        src_camera=crop_camera,
        dst_camera=orig_camera,
        src_points=q_2d,
        src_depths=Z_3d,
    )
    q_2d_crop = q_2d

    return best_pose_orig, q_2d_orig, q_2d_crop

def extract_correspondences(
    image_raw: np.ndarray, 
    mask_raw: np.ndarray, 
    camK: np.ndarray, 
    frame_name: str,
    opts: InferOpts,
    # --- 新增传入的预加载资源 ---
    extractor: torch.nn.Module,
    repre: Any,
    visual_words_knn_index: Optional[Any],
    template_knn_indices: List[Any],
    device: str
) -> Optional[CorrespondenceData]:
    
    t_func_start = time.perf_counter()
    stats = {}
    
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

    object_lids = opts.object_lids 
    for object_lid in object_lids:
        version = opts.version
        if version == "":
            version = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        signature = misc.slugify(opts.object_dataset) + "_{}".format(version)
        output_dir = os.path.join(
            AppConfig.OUTPUT_ROOT, "inference", signature, str(object_lid)
        )
        os.makedirs(output_dir, exist_ok=True)
        # Save parameters to a file.
        config_path = os.path.join(output_dir, "config.json")
        json_util.save_json(config_path, opts)

        # 4. 图像准备与神经网络推理 (GPU Inference)
        # 4. 特征提取 (Inference)
        t_inf = time.perf_counter()
        # Perform inference on each selected image.
        sample = data_util.prepare_sample3(image_raw, scene_cameras)
        # Camera parameters.
        orig_camera_c2w = sample.camera
        orig_image_size = (
            orig_camera_c2w.width,
            orig_camera_c2w.height,
        )
        # Get info about object instances for which we want to estimate pose. 是一个列表，每个元素对应 当前图像中一个对象实例
        instances = get_instances_from_mask(
            obj_id=object_lid,
            mask=mask_raw,
        )
        if len(instances) == 0:
            logger.info("No object instance, skipping.")
            continue
        # Generate grid points at which to sample the feature vectors. 在图像或裁剪区域上生成 规则网格点。用以采样
        if opts.crop:
            grid_size = opts.crop_size
        else:
            grid_size = orig_image_size
        grid_points = feature_util.generate_grid_points(
            grid_size=grid_size,
            cell_size=opts.grid_cell_size,
        )
        grid_points = grid_points.to(device)

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
                image_np_hwc, mask_modal, crop_camera_model_c2w, _, box_amodal = build_crop_view(
                    orig_image=orig_image_np_hwc,
                    orig_mask=orig_mask_modal,
                    orig_camera=orig_camera_c2w,
                    box_amodal=orig_box_amodal,
                    crop_size=opts.crop_size,
                    crop_rel_pad=opts.crop_rel_pad,
                )
                # The virtual camera is becoming the main camera.
                camera_c2w = crop_camera_model_c2w

            # Extract feature map from the crop. 将图像转为张量并提取特征
            image_tensor_chw = array_to_tensor(image_np_hwc).to(torch.float32).permute(2,0,1).to(device)
            image_tensor_bchw = image_tensor_chw.unsqueeze(0)
            extractor_output = extractor(image_tensor_bchw)
            feature_map_chw = extractor_output["feature_maps"][0] # 取第一个层的特征图，形状 (C_feat, H_feat, W_feat)。

            stats['Net Inference'] = time.perf_counter() - t_inf
            # 5. 特征采样与匹配 (核心瓶颈)
            t_match = time.perf_counter()
            
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
                # logging.log_heading(logger, msg, style=logging.RED_BOLD)

            # Extract features at the selected points, of shape (num_points, feat_dims). 从特征图中提取查询点的特征向量
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

            # Establish 2D-3D correspondences. 对每个 2D 查询点在物体所有模板中寻找最相似的特征点，记录对应的 2D 点坐标、3D 模型点坐标 以及匹配模板信息
            corresp = []
            if len(query_points) != 0:
                corresp = corresp_util.establish_correspondences(
                    query_points=query_points,
                    query_features=query_features_proj,
                    object_repre=repre,
                    template_matching_type=opts.match_template_type,
                    template_knn_indices=template_knn_indices,
                    feat_matching_type=opts.match_feat_matching_type,
                    top_n_templates=opts.match_top_n_templates,
                    top_k_buddies=opts.match_top_k_buddies,
                    visual_words_knn_index=visual_words_knn_index,
                    debug=opts.debug,
                )
            stats['Feature Matching'] = time.perf_counter() - t_match
            # 6. PnP 求解 (RANSAC 迭代)
            t_pnp = time.perf_counter()
            
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
            
            # [新增] 输出最终选定的模板 ID
            final_template_id = corresp_final["template_id"]
            logger.info(f"Final selected template ID: {final_template_id} (from coarse_pose_id: {best_coarse_pose_id})")
            
            # 2D 观测点 (N, 2)
            q_2d = corresp_final["coord_2d"].cpu().numpy()
            # 3D 模型点坐标 (N, 3)
            X_3d = corresp_final["coord_3d"].cpu().numpy()
            # 3D 模型特征点 ID (N,)
            X_ids = corresp_final["nn_vertex_ids"].cpu().numpy()
            corresp_scores = corresp_final["coord_conf"]
            
            best_pose = coarse_poses[best_coarse_pose_id]
            # ... 执行 estimate_pose ...
            stats['PnP Solver'] = time.perf_counter() - t_pnp
            # 7. 可视化与 IO
            t_vis = time.perf_counter()
            # # # ------------------ 输出位姿 ------------------
            dataset_model_dir = os.path.join(AppConfig.DATASETS_PATH, opts.object_dataset, "models")
            model_path = os.path.join(dataset_model_dir, f"{opts.object_dataset}_norm.obj")
            pose_path = os.path.join(AppConfig.REFINE_ROOT, opts.object_dataset, "pose_es")
            vis_util.visualize_pose_overlay(
                image_raw=image_raw,
                best_pose=best_pose,
                model_path=model_path,
                orig_camera=orig_camera_c2w,
                crop_camera=crop_camera_model_c2w,
                frame_name=frame_name,
                output_dir=pose_path
            )
            
            # 恢复裁剪位姿和2d点
            best_pose, q_2d, q_crop = recover_pose_and_points_to_orig(
                best_pose=best_pose,
                q_2d=q_2d,
                X_3d=X_3d,
                crop_camera=camera_c2w,
                orig_camera=orig_camera_c2w,
                use_crop=opts.crop,
            )
                
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
            q_crop = q_crop[best_indices]
            
            # --- [1. 新增] 在筛选前，可视化所有原始对应点 ---
            raw_correspondence_data = CorrespondenceData(
                frame_name=frame_name,
                query_2d_pts=final_q_2d,
                model_3d_pts=final_X_3d,
                model_feat_ids=final_X_ids,
                best_pose=best_pose,
                camera_c2w=orig_camera_c2w
            )
            template_id = int(corresp_final["template_id"].item()) if "template_id" in corresp_final else 0
            templates_path = os.path.join(
                "templates",
                opts.version,
                opts.object_dataset,
                str(object_lid),
            )
            template_base_dir = os.path.join(
                AppConfig.OUTPUT_ROOT,
                templates_path,
            )
            vis_debug_path = os.path.join(AppConfig.REFINE_ROOT, opts.object_dataset, "vis_debug")
            # 输出到特殊的 "all_raw" 目录或以 00 开头的文件
            visualize_correspondences2(
                data=raw_correspondence_data,
                template_id=template_id,
                image_raw=image_raw,
                repre=repre,
                template_base_dir=template_base_dir,
                output_dir=vis_debug_path,
                mode="all_raw"  # 特殊模式：一次性画完所有点
            )
            
            # --- [新增] 基于重投影误差剔除离群点 ---
            if len(final_q_2d) > 0:
                # 1. 获取位姿和内参
                R_m2c = best_pose["R_m2c"]
                t_m2c = best_pose["t_m2c"]
                # 构造内参矩阵 K
                fx, fy = orig_camera_c2w.f
                cx, cy = orig_camera_c2w.c
                K = np.array([
                    [fx, 0,  cx],
                    [0,  fy, cy],
                    [0,  0,  1]
                ], dtype=np.float32)

                # 2. 将 3D 点变换到相机坐标系: Pc = R*X + t
                pts_c = (R_m2c @ final_X_3d.T).T + t_m2c.reshape(1, 3)

                # 3. 投影到 2D 平面
                z = pts_c[:, 2]
                # 避免除以 0 或投影在相机背面的点
                valid_z = z > 1e-6
                
                uv_proj = (K @ pts_c.T).T
                uv_proj = uv_proj[:, :2] / uv_proj[:, 2:3]

                # 4. 计算重投影误差 (欧氏距离)
                reproj_errors = np.linalg.norm(uv_proj - final_q_2d, axis=1)

                # ==========================================================
                # 5. [修改] 基于网格的自适应阈值与均匀化策略
                # ==========================================================
                # --- 参数配置 (可提取至 opts) ---
                grid_size = 4                 # 将 2D 区域划分为 8x8 的网格
                min_thresh = 5.0              # 稠密区域的严格阈值 (px)
                max_thresh = 20.0             # 稀疏/形变区域的宽松阈值 (px)
                expected_density = 10         # 当网格内候选点达到 10 个时，使用最严格的 min_thresh
                max_pts_per_cell = 3          # 每个网格最多保留的点数 (防止分布过度集中)
                
                inlier_mask = np.zeros(len(final_q_2d), dtype=bool)

                if valid_z.any():
                    # 确定当前 2D 点的边界范围
                    min_x, min_y = np.min(final_q_2d, axis=0)
                    max_x, max_y = np.max(final_q_2d, axis=0)
                    
                    # 防止由于点在同一直线上导致的除零错误
                    dx = max(max_x - min_x, 1e-5)
                    dy = max(max_y - min_y, 1e-5)
                    
                    # 计算每个 2D 点所属的网格坐标 (0 到 grid_size-1)
                    cell_x = np.clip(np.floor((final_q_2d[:, 0] - min_x) / dx * grid_size).astype(int), 0, grid_size - 1)
                    cell_y = np.clip(np.floor((final_q_2d[:, 1] - min_y) / dy * grid_size).astype(int), 0, grid_size - 1)
                    
                    # 将二维网格坐标展平为一维索引
                    cell_indices = cell_y * grid_size + cell_x

                    # 统计每个网格内的点数
                    unique_cells, cell_counts = np.unique(cell_indices, return_counts=True)
                    count_dict = dict(zip(unique_cells, cell_counts))

                    # 遍历每一个有点的网格进行局部筛选
                    for cell_id in unique_cells:
                        # 获取落入当前网格的所有点的全局索引
                        pt_idxs = np.where(cell_indices == cell_id)[0]
                        
                        # --- 核心策略 1：计算自适应阈值 ---
                        count = count_dict[cell_id]
                        # 计算稠密度比例 (0.0 到 1.0)
                        ratio = min(count / expected_density, 1.0)
                        # 线性插值：点越少，ratio越接近0，阈值越接近 max_thresh
                        cell_thresh = max_thresh - ratio * (max_thresh - min_thresh)
                        
                        # 初步筛选：Z轴在相机前方 且 误差小于动态阈值
                        valid_pts_in_cell = pt_idxs[valid_z[pt_idxs] & (reproj_errors[pt_idxs] < cell_thresh)]
                        
                        # --- 核心策略 2：空间均匀化 (Non-Maximum Suppression) ---
                        if len(valid_pts_in_cell) > 0:
                            # 提取这些点的误差并按从小到大排序
                            errors_in_cell = reproj_errors[valid_pts_in_cell]
                            sorted_sub_idxs = np.argsort(errors_in_cell)
                            
                            # 截取误差最小的 Top-K 个点
                            top_k_idxs = valid_pts_in_cell[sorted_sub_idxs[:max_pts_per_cell]]
                            inlier_mask[top_k_idxs] = True

                # 6. 应用掩码更新所有相关数据
                num_before = len(final_q_2d)
                num_after = np.sum(inlier_mask)
                logger.info(f"Adaptive filtering: {num_before} -> {num_after} points (Grid: {grid_size}x{grid_size}, Thresh: {min_thresh}-{max_thresh}px)")
                
                final_q_2d = final_q_2d[inlier_mask]
                final_X_3d = final_X_3d[inlier_mask]
                final_X_ids = final_X_ids[inlier_mask]
                all_scores = all_scores[inlier_mask]
            
            filtered_data = CorrespondenceData(
                frame_name=frame_name,
                query_2d_pts=final_q_2d,
                model_3d_pts=final_X_3d,
                model_feat_ids=final_X_ids,
                best_pose=best_pose,
                camera_c2w=orig_camera_c2w
            )
            # template_id = int(corresp_final["template_id"].item()) if "template_id" in corresp_final else 0
            # templates_path = os.path.join(
            #     "templates",
            #     opts.version,
            #     opts.object_dataset,
            #     str(object_lid),
            # )
            # template_base_dir = os.path.join(
            #     AppConfig.OUTPUT_ROOT,
            #     templates_path,
            # )
            # vis_debug_path = os.path.join(AppConfig.REFINE_ROOT, opts.object_dataset, "vis_debug")
            visualize_correspondences2(
                data=filtered_data,
                template_id=template_id,
                image_raw=image_raw,
                repre=repre,
                template_base_dir=template_base_dir,
                output_dir=vis_debug_path,
                mode="all_raw",     # 模式：按网格输出多张图
                grid_size=grid_size,
                file_name="flittered", 
            )
            visualize_correspondences2(
                data=filtered_data,
                template_id=template_id,
                image_raw=image_raw,
                repre=repre,
                template_base_dir=template_base_dir,
                output_dir=vis_debug_path,
                mode="grid",     # 模式：按网格输出多张图
                grid_size=grid_size 
            )
            # 创建并返回此帧的对应数据
            stats['Visualization & Post'] = time.perf_counter() - t_vis
    # --- 计算并打印总时长 ---
    total_duration = time.perf_counter() - t_func_start
    
    # 打印精美的耗时统计表
    logger.info(f"\n{'='*30}\n Performance Summary ({frame_name})\n{'-'*30}")
    for stage, duration in stats.items():
        logger.info(f"{stage:<20}: {duration:>7.4f}s ({duration/total_duration:2.1%})")
    logger.info(f"{'-'*30}\n{'TOTAL TIME':<20}: {total_duration:>7.4f}s\n{'='*30}")
    return filtered_data