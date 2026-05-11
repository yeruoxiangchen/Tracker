#!/usr/bin/env python3

"""Infers pose from objects."""

import datetime

import sys
REPO_PATH = "/home/zjr/Tracker/CoarseModel"
BOP_TOOLKIT_PATH = "/home/zjr/Tracker/CoarseModel/external/bop_toolkit"

sys.path.insert(0, REPO_PATH)
sys.path.insert(0, BOP_TOOLKIT_PATH)  # 添加这一行

import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'
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

import torch.nn.functional as F
import pyrender
import trimesh
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
            
            # --- 3. [新增] 轮廓匹配整合 ---
            t_contour = time.perf_counter()
            # 提取当前图像轮廓 (2D)
            cur_contours, _ = cv2.findContours((mask_modal * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            has_cur_contour = len(cur_contours) > 0
                        
            # 遍历每一个候选模板，为其增加轮廓约束
            for cor_idx in range(len(corresp)):
                t_id = int(corresp[cor_idx]["template_id"].item())
                
                # --- Feature correspondences confidence filtering ---
                max_feat = 50
                if "coord_conf" in corresp[cor_idx]:
                    conf = corresp[cor_idx]["coord_conf"]
                    if conf.shape[0] > max_feat:
                        topk = torch.topk(conf, k=max_feat)
                        idx = topk.indices
                        corresp[cor_idx]["coord_2d"] = corresp[cor_idx]["coord_2d"][idx]
                        corresp[cor_idx]["coord_3d"] = corresp[cor_idx]["coord_3d"][idx]
                        corresp[cor_idx]["coord_conf"] = corresp[cor_idx]["coord_conf"][idx]
                        if "nn_vertex_ids" in corresp[cor_idx]:
                            corresp[cor_idx]["nn_vertex_ids"] = corresp[cor_idx]["nn_vertex_ids"][idx]
                            
                # 加载该模板的 3D 轮廓点
                tpl_mask_path = os.path.join(template_base_dir, f"mask/template_{t_id:04d}.png")
                tpl_depth_path = os.path.join(template_base_dir, f"depth/template_{t_id:04d}.png")
                
                if os.path.exists(tpl_mask_path) and os.path.exists(tpl_depth_path) and has_cur_contour:
                    tpl_mask = cv2.imread(tpl_mask_path, cv2.IMREAD_GRAYSCALE)
                    tpl_depth = cv2.imread(tpl_depth_path, cv2.IMREAD_ANYDEPTH)
                    tpl_camera = repre.template_cameras_cam_from_model[t_id]
                    
                    # 提取模板 3D 轮廓 (逻辑同你之前的代码)
                    tpl_cont, _ = cv2.findContours(tpl_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                    if len(tpl_cont) > 0:
                        tpl_pts_2d = tpl_cont[0].reshape(-1, 2)
                        # 采样并反投影
                        z_c = tpl_depth[np.clip(tpl_pts_2d[:, 1], 0, tpl_depth.shape[0]-1), 
                                         np.clip(tpl_pts_2d[:, 0], 0, tpl_depth.shape[1]-1)].astype(np.float32)
                        valid_z = z_c > 0.1
                        
                        # 计算模型空间的 3D 点
                        cx = tpl_camera.c[0].item() if torch.is_tensor(tpl_camera.c[0]) else tpl_camera.c[0]
                        cy = tpl_camera.c[1].item() if torch.is_tensor(tpl_camera.c[1]) else tpl_camera.c[1]
                        fx = tpl_camera.f[0].item() if torch.is_tensor(tpl_camera.f[0]) else tpl_camera.f[0]
                        fy = tpl_camera.f[1].item() if torch.is_tensor(tpl_camera.f[1]) else tpl_camera.f[1]

                        # 使用转换后的变量进行计算
                        x_c = (tpl_pts_2d[valid_z, 0] - cx) * z_c[valid_z] / fx
                        y_c = (tpl_pts_2d[valid_z, 1] - cy) * z_c[valid_z] / fy
                        # y_c = (tpl_pts_2d[valid_z, 1] - tpl_camera.c[1]) * z_c[valid_z] / tpl_camera.f[1]
                        pts_3d_cam_h = np.column_stack([x_c, y_c, z_c[valid_z], np.ones_like(x_c)])
                        pts_3d_model = (tpl_camera.T_world_from_eye @ pts_3d_cam_h.T).T[:, :3]
                        
                        # 对齐采样 (各取 50 个点)
                        target_n = 50
                        idx_cur = np.linspace(0, len(cur_contours[0]) - 1, target_n).astype(int)
                        idx_tpl = np.linspace(0, len(pts_3d_model) - 1, target_n).astype(int)
                        
                        cont_2d = torch.as_tensor(cur_contours[0].reshape(-1, 2)[idx_cur], device=device, dtype=torch.float32)
                        cont_3d = torch.as_tensor(pts_3d_model[idx_tpl], device=device, dtype=torch.float32)
                        
                        # 合并到当前 corresp
                        corresp[cor_idx]["coord_2d"] = torch.cat([corresp[cor_idx]["coord_2d"], cont_2d], dim=0)
                        corresp[cor_idx]["coord_3d"] = torch.cat([corresp[cor_idx]["coord_3d"], cont_3d], dim=0)
                        # 为轮廓点补充 dummy 信息
                        if "coord_conf" in corresp[cor_idx]:
                            cont_conf = torch.full((target_n,), 0.5, device=device)
                            corresp[cor_idx]["coord_conf"] = torch.cat([corresp[cor_idx]["coord_conf"], cont_conf], dim=0)
                        if "nn_vertex_ids" in corresp[cor_idx]:
                            dummy_ids = torch.full((target_n,), -1, dtype=torch.long, device=device)
                            corresp[cor_idx]["nn_vertex_ids"] = torch.cat([corresp[cor_idx]["nn_vertex_ids"], dummy_ids], dim=0)
            
            stats['Contour Matching'] = time.perf_counter() - t_contour
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
                idxs = np.where(final_X_ids == uid)[0]
                best_idx = idxs[np.argmax(all_scores[idxs].cpu().numpy())]
                best_indices.append(best_idx)
            # contour points
            contour_indices = np.where(final_X_ids < 0)[0]
            best_indices.extend(contour_indices)
        
            best_indices = np.array(best_indices)
            final_q_2d = final_q_2d[best_indices]
            final_X_3d = final_X_3d[best_indices]
            final_X_ids = final_X_ids[best_indices]
            all_scores = all_scores[best_indices]
            q_crop = q_crop[best_indices]
            
            correspondence_data = CorrespondenceData(
                frame_name=frame_name,
                query_2d_pts=final_q_2d,
                model_3d_pts=final_X_3d,
                model_feat_ids=final_X_ids,
                best_pose = best_pose,
                camera_c2w= orig_camera_c2w
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
            visualize_correspondences(
                data=correspondence_data,
                template_id=template_id,
                image_raw=image_raw,
                repre=repre,
                template_base_dir=template_base_dir,
                output_dir=vis_debug_path
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
    return correspondence_data

def render_mesh_rgb_depth(
    mesh: trimesh.Trimesh, 
    K: np.ndarray, 
    T_m2c: np.ndarray, 
    image_size: tuple,
    image_raw: np.ndarray = None,   # ✅ 新增
    draw_overlay: bool = False      # ✅ 是否画投影
):
    width, height = image_size

    scene = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=[0.5, 0.5, 0.5])

    # --- OpenCV → OpenGL ---
    T_cv2gl = np.array([
        [1,  0,  0,  0],
        [0, -1,  0,  0],
        [0,  0, -1,  0],
        [0,  0,  0,  1]
    ], dtype=np.float32)

    T_m2c_gl = T_cv2gl @ T_m2c

    # mesh
    renderer_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)
    scene.add(renderer_mesh, pose=T_m2c_gl)

    # camera
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    camera = pyrender.IntrinsicsCamera(
        fx=fx, fy=fy, cx=cx, cy=cy,
        znear=0.01,
        zfar=10000.0
    )
    scene.add(camera, pose=np.eye(4))

    # light
    light = pyrender.DirectionalLight(color=[1,1,1], intensity=5.0)
    scene.add(light, pose=np.eye(4))

    # render
    r = pyrender.OffscreenRenderer(width, height)
    color, depth = r.render(scene)
    r.delete()

    mask = (depth > 0).astype(np.uint8) * 255

    # ================== ✅ 2D投影 overlay ==================
    overlay_img = None

    if draw_overlay and image_raw is not None:
        overlay_img = image_raw.copy()

        # --- 1. 取 mesh 顶点 ---
        verts = np.asarray(mesh.vertices)

        # --- 2. 变换到相机坐标系 ---
        verts_h = np.hstack([verts, np.ones((verts.shape[0], 1))])
        verts_cam = (T_m2c @ verts_h.T).T[:, :3]

        z = verts_cam[:, 2]
        valid = z > 1e-6

        verts_cam = verts_cam[valid]
        z = z[valid]

        # --- 3. 投影到像素 ---
        u = fx * verts_cam[:, 0] / z + cx
        v = fy * verts_cam[:, 1] / z + cy

        pts_2d = np.stack([u, v], axis=1)

        # --- 4. 画点 ---
        for p in pts_2d.astype(int):
            if 0 <= p[0] < width and 0 <= p[1] < height:
                cv2.circle(overlay_img, tuple(p), 1, (0, 255, 0), -1)

        # --- 5. 画 mask 轮廓（更清晰） ---
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(overlay_img, contours, -1, (0, 0, 255), 2)

    return color, depth, mask, overlay_img

def visualize_input_vs_render(
    image_raw: np.ndarray,      # 原始输入图 (H, W, 3)
    render_color: np.ndarray,   # 在线渲染图 (H, W, 3)
    final_q_2d: np.ndarray,     # 输入图上的最终2D点 (N, 2)
    final_X_3d: np.ndarray,     # 模型坐标系下的3D点 (N, 3)
    camK: np.ndarray,           # 相机内参 3x3
    T_m2c: np.ndarray,          # 渲染时使用的位姿 (Model to Camera)
    frame_name: str,
    output_dir: str
):
    """
    可视化输入图特征点与渲染图特征点的对应关系
    """
    import cv2
    import os

    # 1. 准备图像格式
    img_left = image_raw.copy()
    if img_left.max() <= 1.0: img_left = (img_left * 255).astype(np.uint8)
    
    img_right = render_color.copy()
    if img_right.max() <= 1.0: img_right = (img_right * 255).astype(np.uint8)
    
    # 确保是 RGB
    if img_left.shape[-1] == 3: img_left = cv2.cvtColor(img_left, cv2.COLOR_RGB2BGR)
    if img_right.shape[-1] == 3: img_right = cv2.cvtColor(img_right, cv2.COLOR_RGB2BGR)

    # --- 核心修复：对齐高度 ---
    h1, w1 = img_left.shape[:2]
    h2, w2 = img_right.shape[:2]
    
    max_h = max(h1, h2)
    
    # 如果高度不一致，给较短的图补黑边
    if h1 < max_h:
        img_left = cv2.copyMakeBorder(img_left, 0, max_h - h1, 0, 0, cv2.BORDER_CONSTANT, value=(0,0,0))
    if h2 < max_h:
        img_right = cv2.copyMakeBorder(img_right, 0, max_h - h2, 0, 0, cv2.BORDER_CONSTANT, value=(0,0,0))
    # -----------------------
    
    # 2. 将 3D 点投影到渲染图的 2D 坐标上
    # P_c = T_m2c * P_m
    pts_3d_h = np.column_stack([final_X_3d, np.ones(len(final_X_3d))])
    pts_cam = (T_m2c @ pts_3d_h.T).T[:, :3]
    
    # Project to 2D: x = K * P_c
    fx, fy, cx, cy = camK[0, 0], camK[1, 1], camK[0, 2], camK[1, 2]
    render_q_2d = np.zeros((len(pts_cam), 2))
    render_q_2d[:, 0] = pts_cam[:, 0] * fx / pts_cam[:, 2] + cx
    render_q_2d[:, 1] = pts_cam[:, 1] * fy / pts_cam[:, 2] + cy

    # 3. 拼接图像
    h, w = img_left.shape[:2]
    combined = np.hstack([img_left, img_right])

    # 4. 绘制对应关系
    for i in range(len(final_q_2d)):
        # 左图点坐标
        pt_in = tuple(final_q_2d[i].astype(int))
        # 右图点坐标 (需要加上左图的宽度偏移)
        pt_re = (int(render_q_2d[i, 0] + w), int(render_q_2d[i, 1]))
        
        # 随机颜色
        color = [int(c) for c in np.random.randint(50, 255, 3)]
        
        # 画点和线
        cv2.circle(combined, pt_in, 3, color, -1)
        cv2.circle(combined, pt_re, 3, color, -1)
        cv2.line(combined, pt_in, pt_re, color, 1, cv2.LINE_AA)
        
        # 标序号 (可选)
        cv2.putText(combined, str(i), (pt_in[0], pt_in[1]-2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

    # 5. 保存结果
    save_path = os.path.join(output_dir, f"{os.path.splitext(frame_name)[0]}_corresp_render.jpg")
    os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(save_path, combined)
    # logger.info(f"Correspondence visualization saved to {save_path}")
    
def extract_correspondences2(
    image_raw: np.ndarray, 
    mask_raw: np.ndarray, 
    camK: np.ndarray, 
    frame_name: str,
    opts: Any, # InferOpts
    extractor: torch.nn.Module,
    mesh: trimesh.Trimesh,      # 新增
    T_m2c: np.ndarray,          # 新增
    device: str
) -> Optional[Any]: # CorrespondenceData
    
    t_func_start = time.perf_counter()
    stats = {}
    
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
    # 此处假设我们只处理第一个 object_lid (如你原代码)
    object_lid = object_lids[0]
    
    # 建立目录结构和存参等原有流程保留
    version = opts.version if opts.version != "" else datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = os.path.join(AppConfig.OUTPUT_ROOT, "refine_model", opts.object_dataset)
    os.makedirs(output_dir, exist_ok=True)
    json_util.save_json(os.path.join(output_dir, "config.json"), opts)

    # 4. 在线渲染及特征提取 (Inference)
    t_inf = time.perf_counter()
    sample = data_util.prepare_sample3(image_raw, scene_cameras)
    orig_camera_c2w = sample.camera
    orig_image_size = (orig_camera_c2w.width, orig_camera_c2w.height)

    instances = get_instances_from_mask(obj_id=object_lid, mask=mask_raw)
    if len(instances) == 0:
        logger.info("No object instance, skipping.")
        return None

    grid_size = opts.crop_size if opts.crop else orig_image_size
    grid_points = feature_util.generate_grid_points(grid_size=grid_size, cell_size=opts.grid_cell_size).to(device)

    # 我们仅针对第一个 instance 操作
    instance = instances[0]
    orig_image_np_hwc = sample.image.astype(np.float32)/255.0
    orig_mask_modal = instance["input_mask_modal"]
    orig_box_amodal = AlignedBox2f(
        left=instance["input_box_amodal"][0],
        top=instance["input_box_amodal"][1],
        right=instance["input_box_amodal"][2],
        bottom=instance["input_box_amodal"][3],
    )

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
        camera_c2w = crop_camera_model_c2w

    # ========================== 特征提取阶段 ==========================
    # a. 输入图像提取特征
    image_tensor_chw = array_to_tensor(image_np_hwc).to(torch.float32).permute(2,0,1).to(device)
    extractor_output = extractor(image_tensor_chw.unsqueeze(0))
    feature_map_chw = extractor_output["feature_maps"][0] 

    # 筛选有效的 Query 点和提取特征
    mask_modal_tensor = array_to_tensor(mask_modal).to(device)
    query_points = feature_util.filter_points_by_mask(grid_points, mask_modal_tensor)
    if query_points.shape[0] > opts.max_num_queries:
        perm = torch.randperm(query_points.shape[0])
        query_points = query_points[perm[: opts.max_num_queries]]

    query_features = feature_util.sample_feature_map_at_points(
        feature_map_chw=feature_map_chw,
        points=query_points,
        image_size=(image_np_hwc.shape[1], image_np_hwc.shape[0]),
    ).contiguous()

    # b. 渲染出参考图，并提取特征
    t_render = time.perf_counter()
    # [优化 FIX]: 直接使用当前相机 (全图或Crop) 的内参进行渲染，拒绝 Warp 造成的深度畸变！
    cur_cam_K = np.array([
        [camera_c2w.f[0], 0, camera_c2w.c[0]],
        [0, camera_c2w.f[1], camera_c2w.c[1]],
        [0, 0, 1]
    ], dtype=np.float32)
    cur_w, cur_h = camera_c2w.width, camera_c2w.height

    # 由于外参 T_m2c 描述的是光心到物体的相对位姿，在裁剪中通常光心不变，所以 T_m2c 直接复用即可
    r_color, r_depth, r_mask, overlay = render_mesh_rgb_depth(
        mesh=mesh, K=cur_cam_K, T_m2c=T_m2c, image_size=(cur_w, cur_h),
        image_raw=(image_np_hwc * 255).astype(np.uint8), # 用当前的底图
        draw_overlay=True
    )
    
    render_color = r_color.astype(np.float32) / 255.0
    render_mask = r_mask
    render_depth = r_depth

    render_tensor = array_to_tensor(render_color).permute(2,0,1).to(device)
    render_output = extractor(render_tensor.unsqueeze(0))
    render_feat_map = render_output["feature_maps"][0]
    
    render_mask_tensor = array_to_tensor(render_mask).to(device)
    render_points = feature_util.generate_grid_points(grid_size=grid_size, cell_size=opts.grid_cell_size).to(device)
    render_points = feature_util.filter_points_by_mask(render_points, render_mask_tensor)

    if render_points.shape[0] > opts.max_num_queries * 2:
        perm_r = torch.randperm(render_points.shape[0])
        render_points = render_points[perm_r[: opts.max_num_queries * 2]]

    render_features = feature_util.sample_feature_map_at_points(
        feature_map_chw=render_feat_map,
        points=render_points,
        image_size=(render_color.shape[1], render_color.shape[0]),
    ).contiguous()
    
    # # 【调试保存】将裁剪后的结果存入指定目录
    # save_debug_path = "/home/zjr/Tracker/CoarseModel/results/refine_model/snoopy"
    # os.makedirs(save_debug_path, exist_ok=True)
    # cv2.imwrite(f"{save_debug_path}/{frame_name}_crop_input.png", (image_np_hwc * 255).astype(np.uint8)[:,:,::-1])
    # cv2.imwrite(f"{save_debug_path}/{frame_name}_crop_render_mask.png", (r_mask))
    # cv2.imwrite(f"{save_debug_path}/{frame_name}_crop_render_color.png", (r_color[:,:,::-1]))
    # cv2.imwrite(f"{save_debug_path}/{frame_name}_overlay.png", overlay[:,:,::-1])
    
    stats['Net Inference & Render'] = time.perf_counter() - t_inf

    # ========================== 特征匹配阶段 ==========================
    t_match = time.perf_counter()
    corresp = []
    
    # 获取当前图像(Crop/Orig)对应的真实尺寸和内参
    c_fx, c_fy = camera_c2w.f
    c_cx, c_cy = camera_c2w.c
    h_r, w_r = render_depth.shape
    
    if len(query_points) > 0 and len(render_points) > 0:
        # 1. 余弦相似度计算与 Mutual Nearest Neighbor 匹配
        q_feat_norm = F.normalize(query_features, p=2, dim=1)
        r_feat_norm = F.normalize(render_features, p=2, dim=1)
        
        sim = torch.mm(q_feat_norm, r_feat_norm.t()) # (N_q, N_r)
        
        val_q, nn_q = sim.max(dim=1)
        val_r, nn_r = sim.max(dim=0)
        
        # 找到双向奔赴的最优匹配点
        valid_matches = (nn_r[nn_q] == torch.arange(len(q_feat_norm), device=device))
        # 置信度阈值过滤，如果 opts 有指定特征匹配阈值可以用那个，这里暂定 0.5 作为保险阈值
        valid_matches = valid_matches & (val_q > 0.5) 
        
        matched_q_idx = torch.nonzero(valid_matches).squeeze(-1)
        matched_r_idx = nn_q[matched_q_idx]
        
        matched_q_pts = query_points[matched_q_idx]         # 在实拍图上的2D点
        matched_r_pts = render_points[matched_r_idx]        # 在渲染图上的2D点
        matched_conf = val_q[matched_q_idx]                 # 置信度分数

        # 2. 从渲染图的 2D 提取 3D 模型坐标
        u = matched_r_pts[:, 0].cpu().numpy()
        v = matched_r_pts[:, 1].cpu().numpy()

        # --- FIX 2: 使用当前 render_depth 的真实宽高进行边界截断 ---
        u_idx = np.clip(np.round(u).astype(int), 0, w_r - 1)
        v_idx = np.clip(np.round(v).astype(int), 0, h_r - 1)
        z_c = render_depth[v_idx, u_idx].astype(np.float32)
        
        # 只保留有深度的有效点
        valid_z = z_c > 0.1
        
        # [核心 FIX]: 严格使用当前相机空间的内参 c_cx, c_cy, c_fx, c_fy 进行反投影！
        x_c = (u[valid_z] - c_cx) * z_c[valid_z] / c_fx
        y_c = (v[valid_z] - c_cy) * z_c[valid_z] / c_fy
        pts_3d_cam_h = np.column_stack([x_c, y_c, z_c[valid_z], np.ones_like(x_c)])
        
        # 将 Camera 3D 转回 Model 3D: P_m = inv(T_m2c) * P_c
        T_c2m = np.linalg.inv(T_m2c)
        pts_3d_model = (T_c2m @ pts_3d_cam_h.T).T[:, :3]
        print(f"Sample 3D model point: {pts_3d_model[0]}")
        # 3. 构造字典给 PnP 使用 (只封装一个 corresp_curr 因为只有这一组伪模板渲染)
        corresp_curr = {
            "coord_2d": matched_q_pts[torch.from_numpy(valid_z).to(device)], 
            "coord_3d": torch.as_tensor(pts_3d_model, dtype=torch.float32, device=device),
            "coord_conf": matched_conf[torch.from_numpy(valid_z).to(device)],
            "nn_vertex_ids": torch.arange(pts_3d_model.shape[0], dtype=torch.long, device=device), # 用自增 ID 代替特征点 ID
            "template_id": torch.tensor(0) # 虚拟一个 0 给后续使用
        }
        corresp.append(corresp_curr)

    stats['Feature Matching & Unproject'] = time.perf_counter() - t_match

    # ========================== 轮廓匹配整合阶段 ==========================
    t_contour = time.perf_counter()
    
    # 提取当前图像轮廓 (2D)
    cur_contours, _ = cv2.findContours((mask_modal * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    has_cur_contour = len(cur_contours) > 0
                
    if len(corresp) > 0 and has_cur_contour:
        cor_idx = 0
        # --- Feature correspondences confidence filtering ---
        max_feat = 50
        conf = corresp[cor_idx]["coord_conf"]
        if conf.shape[0] > max_feat:
            topk = torch.topk(conf, k=max_feat)
            idx = topk.indices
            corresp[cor_idx]["coord_2d"] = corresp[cor_idx]["coord_2d"][idx]
            corresp[cor_idx]["coord_3d"] = corresp[cor_idx]["coord_3d"][idx]
            corresp[cor_idx]["coord_conf"] = corresp[cor_idx]["coord_conf"][idx]
            if "nn_vertex_ids" in corresp[cor_idx]:
                corresp[cor_idx]["nn_vertex_ids"] = corresp[cor_idx]["nn_vertex_ids"][idx]
                
        # 直接利用前面提取出来的 render_mask 和 render_depth
        tpl_cont, _ = cv2.findContours(render_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        if len(tpl_cont) > 0:
            tpl_pts_2d = tpl_cont[0].reshape(-1, 2)
            z_c = render_depth[
                np.clip(tpl_pts_2d[:, 1], 0, render_depth.shape[0]-1), 
                np.clip(tpl_pts_2d[:, 0], 0, render_depth.shape[1]-1)
            ].astype(np.float32)
            valid_z = z_c > 0.1
            
            # --- FIX 4: 轮廓点同样需要使用当前相机内参 (c_fx, c_fy, c_cx, c_cy) 反投影 ---
            x_c = (tpl_pts_2d[valid_z, 0] - c_cx) * z_c[valid_z] / c_fx
            y_c = (tpl_pts_2d[valid_z, 1] - c_cy) * z_c[valid_z] / c_fy
            pts_3d_cam_h = np.column_stack([x_c, y_c, z_c[valid_z], np.ones_like(x_c)])
            
            # 同样将其从 相机系转到 模型系
            T_c2m = np.linalg.inv(T_m2c)
            pts_3d_model = (T_c2m @ pts_3d_cam_h.T).T[:, :3]
            
            target_n = 50
            if len(pts_3d_model) > 0 and len(cur_contours[0]) > 0:
                idx_cur = np.linspace(0, len(cur_contours[0]) - 1, target_n).astype(int)
                idx_tpl = np.linspace(0, len(pts_3d_model) - 1, target_n).astype(int)
                
                cont_2d = torch.as_tensor(cur_contours[0].reshape(-1, 2)[idx_cur], device=device, dtype=torch.float32)
                cont_3d = torch.as_tensor(pts_3d_model[idx_tpl], device=device, dtype=torch.float32)
                
                corresp[cor_idx]["coord_2d"] = torch.cat([corresp[cor_idx]["coord_2d"], cont_2d], dim=0)
                corresp[cor_idx]["coord_3d"] = torch.cat([corresp[cor_idx]["coord_3d"], cont_3d], dim=0)
                
                cont_conf = torch.full((target_n,), 0.5, device=device)
                corresp[cor_idx]["coord_conf"] = torch.cat([corresp[cor_idx]["coord_conf"], cont_conf], dim=0)
                
                dummy_ids = torch.full((target_n,), -1, dtype=torch.long, device=device)
                corresp[cor_idx]["nn_vertex_ids"] = torch.cat([corresp[cor_idx]["nn_vertex_ids"], dummy_ids], dim=0)

    stats['Contour Matching'] = time.perf_counter() - t_contour

    # ========================== PnP 求解 ==========================
    t_pnp = time.perf_counter()
    coarse_poses = []
    
    for corresp_id, corresp_curr in enumerate(corresp):
        num_corresp = len(corresp_curr["coord_2d"])
        if num_corresp < 6:
            logger.info(f"Only {num_corresp} correspondences, skipping.")
            continue
            
        # (
        #     coarse_pose_success,
        #     R_m2c_coarse,
        #     t_m2c_coarse,
        #     inliers_coarse,
        #     quality_coarse,
        # ) = pnp_util.estimate_pose(
        #     corresp=corresp_curr,
        #     camera_c2w=camera_c2w,
        #     pnp_type=opts.pnp_type,
        #     pnp_ransac_iter=opts.pnp_ransac_iter,
        #     pnp_inlier_thresh=opts.pnp_inlier_thresh,
        #     pnp_required_ransac_conf=opts.pnp_required_ransac_conf,
        #     pnp_refine_lm=opts.pnp_refine_lm,
        # )


        # if coarse_pose_success:
        #     coarse_poses.append(
        #         {
        #             "type": "coarse",
        #             "R_m2c": R_m2c_coarse,
        #             "t_m2c": t_m2c_coarse,
        #             "corresp_id": corresp_id,
        #             "quality": quality_coarse,
        #             "inliers": inliers_coarse,
        #         }
        #     )
        
        if hasattr(corresp_curr["coord_3d"], "cpu"):
            object_points = corresp_curr["coord_3d"].cpu().numpy().astype(np.float32)
            image_points = corresp_curr["coord_2d"].cpu().numpy().astype(np.float32)
        else:
            object_points = np.array(corresp_curr["coord_3d"], dtype=np.float32)
            image_points = np.array(corresp_curr["coord_2d"], dtype=np.float32)
            
        # 获取相机内参 K
        K = misc.get_intrinsic_matrix(camera_c2w)
        
        # 2. 构造 PnP 初始猜测值 (核心修复点)
        # ⚠️ 请确保此处的 T_m2c 是你用于渲染或上一帧追踪到的初始位姿矩阵
        rvec_init, _ = cv2.Rodrigues(T_m2c[:3, :3])
        tvec_init = T_m2c[:3, 3].astype(np.float32).reshape(3, 1)

        pose_est_success = False
        inliers = None

        try:
            # 直接调用带有 useExtrinsicGuess 的 PnP
            pose_est_success, rvec_est, t_est, inliers = cv2.solvePnPRansac(
                objectPoints=object_points,
                imagePoints=image_points,
                cameraMatrix=K,
                distCoeffs=None,
                rvec=rvec_init.copy(),   # 传入初始旋转
                tvec=tvec_init.copy(),   # 传入初始平移
                useExtrinsicGuess=True,  # 开启初始值约束，防止跳到相机背面！
                iterationsCount=opts.pnp_ransac_iter,
                reprojectionError=opts.pnp_inlier_thresh,
                confidence=opts.pnp_required_ransac_conf,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
        except Exception as e:
            logger.warning(f"PnP Failed or Crashed: {e}")
            pose_est_success = False

        if pose_est_success and inliers is not None and len(inliers) > 0:
            # 3. 可选的 LM 优化精修 (对齐原函数的 pnp_refine_lm 逻辑)
            if opts.pnp_refine_lm:
                try:
                    rvec_est, t_est = cv2.solvePnPRefineLM(
                        objectPoints=object_points[inliers],
                        imagePoints=image_points[inliers],
                        cameraMatrix=K,
                        distCoeffs=None,
                        rvec=rvec_est,
                        tvec=t_est,
                    )
                except Exception:
                    pass # 如果 LM 失败，则保留 RANSAC 的结果
                    
            # 转换为常规的旋转矩阵和平移向量
            R_m2c_coarse = cv2.Rodrigues(rvec_est)[0]
            t_m2c_coarse = t_est.reshape(3)
            quality_coarse = float(len(inliers))
            
            # 4. 终极防御机制：检查镜像陷阱 (Z < 0)
            if t_m2c_coarse[2] < 0:
                logger.warning(f"Detected PnP mirror solution (Z={t_m2c_coarse[2]:.2f} < 0)! Reverting to initial pose.")
                # 如果即使给了猜测值还是算出了负数深度，说明点云对应关系极差，
                # 直接回退到初始位姿，或者你也可以将 pose_est_success = False 舍弃这一帧
                R_m2c_coarse = T_m2c[:3, :3]
                t_m2c_coarse = T_m2c[:3, 3]
                # pose_est_success = False # 取消注释此行可以直接抛弃该结果
            # 封装进结果列表
            coarse_poses.append(
                {
                    "type": "coarse",
                    "R_m2c": R_m2c_coarse,
                    "t_m2c": t_m2c_coarse,
                    "corresp_id": corresp_id,
                    "quality": quality_coarse,
                    "inliers": inliers,
                }
            )
            

    if not coarse_poses:
        return None
    best_coarse_quality = None
    best_coarse_pose_id = 0
    for coarse_pose_id, pose in enumerate(coarse_poses):
        if best_coarse_quality is None or pose["quality"] > best_coarse_quality:
            best_coarse_pose_id = coarse_pose_id
            best_coarse_quality = pose["quality"]
            
    best_corresp_id = coarse_poses[best_coarse_pose_id]["corresp_id"]
    corresp_final = corresp[best_corresp_id]
    
    q_2d = corresp_final["coord_2d"].cpu().numpy()
    X_3d = corresp_final["coord_3d"].cpu().numpy()
    X_ids = corresp_final["nn_vertex_ids"].cpu().numpy()
    corresp_scores = corresp_final["coord_conf"]
    best_pose = coarse_poses[best_coarse_pose_id]

    stats['PnP Solver'] = time.perf_counter() - t_pnp

    # ========================== 可视化与 IO ==========================
        #####################################################
    if len(coarse_poses) > 0:
        # 准备保存路径
        crop_vis_path = os.path.join(output_dir, "vis_crop_pose")
        def visualize_pose_on_crop(
            image_crop: np.ndarray,      # 裁剪后的图像 (H, W, 3)
            best_pose: dict,             # PnP 解出的位姿 (包含 R_m2c, t_m2c)
            mesh: trimesh.Trimesh,       # 传入 mesh 对象，避免重复加载
            crop_camera: PinholePlaneCameraModel,
            frame_name: str,
            output_dir: str,
            alpha: float = 0.5
        ):
            """
            将位姿直接投影到裁剪后的图像上进行验证。
            """
            # 1. 直接获取 PnP 在裁剪系下的 R, t
            R_m2c = best_pose["R_m2c"]
            t_m2c = best_pose["t_m2c"].reshape(3)

            # 2. 变换顶点到裁剪相机空间
            verts = np.asarray(mesh.vertices, dtype=np.float32)
            # P_c = R * P_m + t
            Xc_crop = (R_m2c @ verts.T).T + t_m2c

            # 3. 使用裁剪相机的内参投影
            # 注意：build_crop_view 修正后的 crop_camera 已经包含了裁剪后的 fx, fy, cx, cy
            fx, fy = crop_camera.f
            cx, cy = crop_camera.c
            
            u = (fx * Xc_crop[:, 0] / Xc_crop[:, 2]) + cx
            v = (fy * Xc_crop[:, 1] / Xc_crop[:, 2]) + cy
            pts_2d = np.stack([u, v], axis=1).astype(int)

            # 4. 绘图
            # 如果 image_crop 是 0-1 范围的 float32，转为 0-255 uint8
            if image_crop.dtype == np.float32:
                vis_img = (image_crop * 255).astype(np.uint8).copy()
            else:
                vis_img = image_crop.copy()
                
            overlay = vis_img.copy()
            
            # 绘制 Mesh 线框
            # 这里的 mesh.faces 是面片索引
            for face in mesh.faces:
                # 简单起见，只画三角形的三条边
                pts = pts_2d[face]
                cv2.polylines(overlay, [pts], True, (0, 255, 0), 1, cv2.LINE_AA)

            # 叠加
            cv2.addWeighted(overlay, alpha, vis_img, 1 - alpha, 0, vis_img)

            # 保存
            os.makedirs(output_dir, exist_ok=True)
            out_path = os.path.join(output_dir, f"{frame_name}_crop_overlay.png")
            cv2.imwrite(out_path, cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))
        # # 调用上面的函数
        # test_pose = {
        #     "R_m2c": T_m2c[:3, :3],
        #     "t_m2c": T_m2c[:3, 3].reshape(3, 1),
        #     "quality": 1.0,
        #     "inliers": np.array([]) 
        # }
        visualize_pose_on_crop(
            image_crop=image_np_hwc,      # 裁剪后的输入图
            best_pose=best_pose,          # 刚解出的最佳位姿
            mesh=mesh,                    # 外部传入的 mesh
            crop_camera=camera_c2w,       # 对应的裁剪相机模型
            frame_name=frame_name,
            output_dir=crop_vis_path
        )
    
    ################################
    t_vis = time.perf_counter()
    dataset_model_dir = os.path.join(AppConfig.DATASETS_PATH, opts.object_dataset, "models")
    model_path = os.path.join(dataset_model_dir, f"{opts.object_dataset}_norm.obj")
    pose_path = os.path.join(AppConfig.REFINE_ROOT, opts.object_dataset, "pose_es")
    
    # 恢复裁剪位姿和2d点
    best_pose, q_2d, q_crop = recover_pose_and_points_to_orig(
        best_pose=best_pose,
        q_2d=q_2d,
        X_3d=X_3d,
        crop_camera=camera_c2w,
        orig_camera=orig_camera_c2w,
        use_crop=opts.crop,
    )
    
    vis_util.visualize_pose_overlay2(
        image_raw=image_raw,
        best_pose=best_pose,
        mesh=mesh,
        orig_camera=orig_camera_c2w,
        crop_camera=crop_camera_model_c2w if opts.crop else orig_camera_c2w,
        frame_name=frame_name,
        output_dir=pose_path
    )
    # # 使用深度做透视反投影再投影
    # Z_3d = X_3d[:, 2].astype(np.float32)
    # q_2d_orig = warp_points_perspective(
    #     src_camera=camera_c2w,
    #     dst_camera=orig_camera_c2w,
    #     src_points=q_2d,
    #     src_depths=Z_3d,
    # )
    # q_2d=q_2d_orig
    # org_cam_K = np.array([
    #     [orig_camera_c2w.f[0], 0, orig_camera_c2w.c[0]],
    #     [0, orig_camera_c2w.f[1], orig_camera_c2w.c[1]],
    #     [0, 0, 1]
    # ], dtype=np.float32)
    # org_w, org_h = orig_camera_c2w.width, orig_camera_c2w.height
    
    # R = best_pose["R_m2c"]              # (3,3)
    # t = best_pose["t_m2c"].reshape(3)   # (3,
    # T_M2C = np.eye(4)
    # T_M2C[:3, :3] = R
    # T_M2C[:3, 3] = t
    # verts = np.asarray(mesh.vertices, dtype=np.float32)
    # Xc = (best_pose["R_m2c"] @ verts.T).T + best_pose["t_m2c"].reshape(3)
    # print("min z:", Xc[:,2].min(), "max z:", Xc[:,2].max())
    # org_color, org_depth, org_mask, org_overlay = render_mesh_rgb_depth(
    #     mesh=mesh, K=org_cam_K, T_m2c=T_M2C, image_size=(org_w, org_h),
    #     image_raw=(image_raw).astype(np.uint8), # 用当前的底图
    #     draw_overlay=True
    # )
    # # 【调试保存】将裁剪后的结果存入指定目录
    # save_debug_path = "/home/zjr/Tracker/CoarseModel/results/refine_model/snoopy"
    # os.makedirs(save_debug_path, exist_ok=True)
    # cv2.imwrite(f"{save_debug_path}/{frame_name}_crop_input.png", (image_np_hwc * 255).astype(np.uint8)[:,:,::-1])
    # cv2.imwrite(f"{save_debug_path}/{frame_name}_crop_render_mask.png", (org_mask))
    # cv2.imwrite(f"{save_debug_path}/{frame_name}_crop_render_color.png", (org_color[:,:,::-1]))
    # cv2.imwrite(f"{save_debug_path}/{frame_name}_debug_overlay.png", org_overlay[:,:,::-1])
    final_q_2d = q_2d
    final_X_3d = X_3d
    final_X_ids = X_ids
    all_scores = corresp_scores
    # 重复 3d-2d 对应剪枝逻辑
    unique_ids = np.unique(X_ids)
    best_indices = []
    for uid in unique_ids:
        if uid < 0: # 略过轮廓假特征点的 uid (-1)
            continue
        idxs = np.where(X_ids == uid)[0]
        best_idx = idxs[np.argmax(corresp_scores[idxs].detach().cpu().numpy())]
        best_indices.append(best_idx)
        
    contour_indices = np.where(X_ids < 0)[0]
    best_indices.extend(contour_indices)

    best_indices = np.array(best_indices)
    final_q_2d = q_2d[best_indices]
    final_X_3d = X_3d[best_indices]
    final_X_ids = X_ids[best_indices]
    
    correspondence_data = CorrespondenceData(
        frame_name=frame_name,
        query_2d_pts=final_q_2d,
        model_3d_pts=final_X_3d,
        model_feat_ids=final_X_ids,
        best_pose=best_pose,
        camera_c2w=orig_camera_c2w
    )

    # Note: 之前这里有可视化 correspondences 依赖模板基目录，
    # 现在直接用在线生成的，不再适用旧的可视化逻辑，如果需要你可以直接使用 cv2 在 q_2d 画点。
    # 这里保持接口通过，防止你外部调用崩溃。
    vis_render_dir = os.path.join(output_dir, "vis_render_corresp")
    
    org_cam_K = np.array([
        [orig_camera_c2w.f[0], 0, orig_camera_c2w.c[0]],
        [0, orig_camera_c2w.f[1], orig_camera_c2w.c[1]],
        [0, 0, 1]
    ], dtype=np.float32)
    
    R = best_pose["R_m2c"]              # (3,3)
    t = best_pose["t_m2c"].reshape(3)   # (3,
    T_fin = np.eye(4)
    T_fin[:3, :3] = R
    T_fin[:3, 3] = t
    visualize_input_vs_render(
        image_raw=image_raw,          # 原始输入
        render_color=r_color,    # 你代码中生成的 render_color
        final_q_2d=final_q_2d,        # 经过剪枝和恢复后的 2D 点
        final_X_3d=final_X_3d,        # 经过剪枝后的 3D 点
        camK=cur_cam_K,                    # 你的相机内参
        T_m2c=T_fin,                  # 你渲染时传入的位姿6
        frame_name=frame_name,
        output_dir=vis_render_dir
    )
    stats['Visualization & Post'] = time.perf_counter() - t_vis

    total_duration = time.perf_counter() - t_func_start
    logger.info(f"\n{'='*30}\n Performance Summary ({frame_name})\n{'-'*30}")
    for stage, duration in stats.items():
        logger.info(f"{stage:<20}: {duration:>7.4f}s ({duration/total_duration:2.1%})")
    logger.info(f"{'-'*30}\n{'TOTAL TIME':<20}: {total_duration:>7.4f}s\n{'='*30}")
    
    return correspondence_data