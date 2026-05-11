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
    corresp_util, feature_util_lunkuo as feature_util, pnp_util, repre_util, json_util,
    knn_util, misc as misc_util, projector_util, data_util, logging, misc,
)
from utils.structs import AlignedBox2f, PinholePlaneCameraModel
from utils.misc import  warp_image, array_to_tensor
from .config import AppConfig, InferOpts
from typing import List, Optional, Dict

from core.util import get_instances_from_mask

logger: logging.Logger = logging.get_logger()


class CorrespondenceData(object):
    def __init__(self, frame_name, query_2d_pts, model_3d_pts, model_feat_ids = None, best_pose = None, camera_c2w = None):
        self.frame_name = frame_name
        self.query_2d_pts = query_2d_pts
        self.model_3d_pts = model_3d_pts
        # self.model_feat_ids = model_feat_ids
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

from typing import Any, Dict, List, Optional, Tuple
def cyclic_buddies_matching(
    query_points: torch.Tensor,
    query_features: torch.Tensor,
    query_knn_index: knn_util.KNN,
    object_features: torch.Tensor,
    object_knn_index: knn_util.KNN,
    top_k: int,
    debug: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Find best buddies via cyclic distance (https://arxiv.org/pdf/2204.03635.pdf)."""

    # Find nearest neighbours in both directions.
    query2obj_nn_ids = object_knn_index.search(query_features)[1].flatten()
    obj2query_nn_ids = query_knn_index.search(object_features)[1].flatten()

    # 2D locations of the query points.
    u1 = query_points

    # 2D locations of the cyclic points.
    query2obj_nn_ids = query2obj_nn_ids.to(obj2query_nn_ids.device)     # device 问题
    cycle_ids = obj2query_nn_ids[query2obj_nn_ids]
    u2 = query_points[cycle_ids]

    # L2 distances between the query and cyclic points.
    cycle_dists = torch.linalg.norm(u1 - u2, axis=1)

    # Keep only top k best buddies.
    top_k = min(top_k, query_points.shape[0])
    _, query_bb_ids = torch.topk(-cycle_dists, k=top_k, sorted=True)

    # Best buddy scores.
    bb_dists = cycle_dists[query_bb_ids]
    bb_scores = torch.as_tensor(1.0 - (bb_dists / bb_dists.max()))

    # Returns IDs of the best buddies.
    query_bb_ids = query_bb_ids.to(query2obj_nn_ids.device)     # device 问题
    object_bb_ids = query2obj_nn_ids[query_bb_ids]


    return query_bb_ids, object_bb_ids, bb_dists, bb_scores
def extract_correspondences(
    image_raw: np.ndarray, 
    mask_raw: np.ndarray, 
    camK: np.ndarray, 
    frame_name: str,
    opts: InferOpts,
    repre,              # [新增参数] 外部初始化好的 Object Repre
    template_id: int    # [新增参数] 指定的模板 ID
) -> Optional[CorrespondenceData]:
    
    extractor = feature_util.make_feature_extractor(opts.extractor_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    extractor.to(device)
    
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
    object_lid = opts.object_lids[0]
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

    templates_path = os.path.join("templates", opts.version, opts.object_dataset, str(object_lid))
    template_base_dir = os.path.join(AppConfig.OUTPUT_ROOT, templates_path)
    # Build per-template KNN index with features from that template.
    tpl_feat_mask = torch.as_tensor(repre.feat_to_template_ids == template_id).to(device)
    tpl_feat_ids = torch.nonzero(tpl_feat_mask).flatten()
    if len(tpl_feat_ids) == 0:
        logger.warning(f"Template {template_id} has no features!")
        return None

    template_feats = repre.feat_vectors[tpl_feat_ids] # 拿到该模板的特征向量集合
    # Build knn index for object features.
    template_knn_index = knn_util.KNN(k=1, metric="l2")
    template_knn_index.fit(template_feats)

    # B. [新增] 提取模板轮廓 2D-3D 对应关系
    # 1. 安全地获取 3D 顶点 (处理设备兼容性)
    # 确保所有参与索引的 tensor 都在同一设备上
    tpl_mask_path = os.path.join(template_base_dir, f"mask/template_{template_id:04d}.png")
    tpl_depth_path = os.path.join(template_base_dir, f"depth/template_{template_id:04d}.png")
    
    tpl_mask = cv2.imread(tpl_mask_path, cv2.IMREAD_GRAYSCALE)
    # 读取深度图 (假设是 uint16 毫米单位，根据实际数据调整缩放系数)
    tpl_depth = cv2.imread(tpl_depth_path, cv2.IMREAD_ANYDEPTH) 
    
    if tpl_mask is None or tpl_depth is None:
        logger.error(f"Cannot load mask or depth for template {template_id}")
        return None
    # 1. 提取模板轮廓点 (2D)
    tpl_contours, _ = cv2.findContours(tpl_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(tpl_contours) == 0:
        return None
    tpl_contour_pts = tpl_contours[0].reshape(-1, 2).astype(np.float32) # [N, 2] (u, v)

    # 2. 反投影获取 3D 坐标
    # 获取相机内参
    tpl_camera = repre.template_cameras_cam_from_model[template_id]
    fx, fy = tpl_camera.f
    cx, cy = tpl_camera.c
    if torch.is_tensor(cx): cx = cx.item()
    if torch.is_tensor(cy): cy = cy.item()
        
    # 提取轮廓处的深度值
    u = tpl_contour_pts[:, 0].astype(int)
    v = tpl_contour_pts[:, 1].astype(int)
    # 防止索引越界
    u = np.clip(u, 0, tpl_depth.shape[1] - 1)
    v = np.clip(v, 0, tpl_depth.shape[0] - 1)
    
    z_c = tpl_depth[v, u].astype(np.float32)
    
    # 过滤掉深度为 0 的无效点
    valid_depth_mask = z_c > 0.1 
    tpl_contour_pts = tpl_contour_pts[valid_depth_mask]
    z_c = z_c[valid_depth_mask]
    u_v = tpl_contour_pts
    
    # 计算相机坐标系下的 3D 点 (Camera Space)
    x_c = (u_v[:, 0] - cx) * z_c / fx
    y_c = (u_v[:, 1] - cy) * z_c / fy
    pts_3d_cam = np.stack([x_c, y_c, z_c], axis=1) # [N, 3]

    # 3. [关键步骤] 将相机坐标转回模型坐标 (Model Space)
    # 因为 pnp_util.estimate_pose 期望输入的是模型空间坐标
    # tpl_camera.T_world_from_eye 实际上是 T_c2m (Model to Camera)
    T_c2m = tpl_camera.T_world_from_eye 
    # T_m2c = np.linalg.inv(T_c2m) # 得到 Camera to Model 变换
    
    pts_3d_cam_h = np.hstack([pts_3d_cam, np.ones((len(pts_3d_cam), 1))])
    tpl_contour_3d = (T_c2m @ pts_3d_cam_h.T).T[:, :3] # 转回模型空间

    # 将结果转为 Tensor 以便后续合并
    tpl_contour_3d = torch.as_tensor(tpl_contour_3d, device=device, dtype=torch.float32)
    tpl_contour_pts_2d = torch.as_tensor(tpl_contour_pts, device=device, dtype=torch.float32)
    
    
    # Perform inference on each selected image.
    sample = data_util.prepare_sample3(image_raw, scene_cameras)
    # Camera parameters.
    orig_camera_c2w = sample.camera
    orig_image_size = (
        orig_camera_c2w.width,
        orig_camera_c2w.height,
    )
    # Get info about object instances for which we want to estimate pose. 是一个列表，每个元素对应 当前图像中一个对象实例
    # 获取实例 (这里假设 mask_raw 只包含我们关心的那个物体，或者逻辑外部处理)
    # 如果 opts.object_lids 是列表，这里取第一个或者根据逻辑调整，通常这里取 repre 对应的 id
    
    instances = get_instances_from_mask(
        obj_id=object_lid,
        mask=mask_raw,
    )
    if len(instances) == 0:
        logger.info("No object instance, skipping.")
        return None
    
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

        # Keep only points inside the object mask. 保留物体内部的网格点
        mask_modal_tensor = array_to_tensor(mask_modal).to(device)
        query_points = feature_util.sample_mask_contour_points(
            mask_modal_tensor,
            max_points=opts.max_num_queries,
            interior_ratio=0.3,
        ).to(device)

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
        else:
            query_features_proj = query_features

        import torch.nn.functional as F
        query_features_proj = F.normalize(query_features_proj, dim=1)
        # Establish 2D-3D correspondences. 对每个 2D 查询点在物体所有模板中寻找最相似的特征点，记录对应的 2D 点坐标、3D 模型点坐标 以及匹配模板信息
        corresp = []
        if len(query_points) != 0:
            # 6.1 为 Query Features 建立 KNN
            query_knn_index = knn_util.KNN(k=1, metric="l2")
            query_knn_index.fit(query_features_proj)

            # 6.2 执行循环一致性匹配 (Cyclic Buddies)
            # 假设 cyclic_buddies_matching 在当前上下文可用 (通常在 corresp_util 中)
            # 注意：如果 cyclic_buddies_matching 没导入，需要从 corresp_util 导入
            (
                match_query_ids,
                match_obj_ids,  # 这是相对于 template_feats 的索引
                match_dists,
                match_scores,
            ) = cyclic_buddies_matching(
                query_points=query_points,
                query_features=query_features_proj,
                query_knn_index=query_knn_index,
                object_features=template_feats,    # 只传入当前模板的特征
                object_knn_index=template_knn_index, # 只传入当前模板的KNN
                top_k=opts.match_top_k_buddies,
                debug=opts.debug,
            )
            
            # --- [新增] 特征点下采样至 50 个 ---
            target_feat_num = 25
            if len(match_query_ids) > target_feat_num:
                feat_perm = torch.randperm(len(match_query_ids))[:target_feat_num]
                match_query_ids = match_query_ids[feat_perm]
                match_obj_ids = match_obj_ids[feat_perm]
                match_scores = match_scores[feat_perm]
                
            # 6.3 映射回全局 Object Feature ID 和 3D 坐标
            # match_obj_ids 是局部索引，需要映射回 repre 的全局索引
            match_obj_feat_ids = tpl_feat_ids[match_obj_ids]
            
            feat_2d = query_points[match_query_ids]
            feat_3d = repre.vertices[match_obj_feat_ids] # 获取 3D 坐标
            feat_conf = match_scores
            # B. [核心新增] 轮廓点匹配 (基于 Mask)
            # 从当前图像的 mask_modal 提取轮廓
            cur_contours, _ = cv2.findContours((mask_modal * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            target_cont_num = 25
            if len(cur_contours) > 0 and len(tpl_contour_pts) > 0:
                cur_contour_pts = cur_contours[0].reshape(-1, 2).astype(np.float32)
                
                # 固定采样 target_cont_num (50) 个点
                # 使用 linspace 确保从头到尾均匀覆盖轮廓
                idx_cur = np.linspace(0, len(cur_contour_pts) - 1, target_cont_num).astype(int)
                idx_tpl = np.linspace(0, len(tpl_contour_pts) - 1, target_cont_num).astype(int)
                
                cont_2d = torch.as_tensor(cur_contour_pts[idx_cur]).to(device)
                cont_3d = torch.as_tensor(tpl_contour_3d[idx_tpl]).to(device)
                cont_conf = torch.ones(target_cont_num).to(device) * 0.5 
            else:
                cont_2d = torch.empty((0, 2), device=device)
                cont_3d = torch.empty((0, 3), device=device)
                cont_conf = torch.empty(0, device=device)

            # C. 合并特征点和轮廓点
            all_2d = torch.cat([feat_2d], dim=0)
            all_3d = torch.cat([feat_3d], dim=0)
            all_conf = torch.cat([feat_conf], dim=0)
            
            # dummy_cont_ids = torch.full((len(cont_2d),), -1, dtype=torch.long, device=device)
            # all_X_ids = torch.cat([match_obj_feat_ids, dummy_cont_ids], dim=0)
            # 构造结果字典
            corresp_final = {
                "template_id": torch.tensor(template_id), # 记录ID用于可视化
                "coord_2d": all_2d,
                "coord_3d": all_3d,
                "coord_conf": all_conf,
                # "nn_vertex_ids": all_X_ids,
            }
            # 封装进列表，保持原有结构以便后续兼容
            corresp = [corresp_final]
                
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
            # 既然只有一个模板，其实就是取第一个，或者按 quality 排序
        best_coarse_pose = max(coarse_poses, key=lambda x: x["quality"])
        
        # 提取数据用于输出
        corresp_used = corresp[best_coarse_pose["corresp_id"]]
        q_2d = corresp_used["coord_2d"].cpu().numpy()
        X_3d = corresp_used["coord_3d"].cpu().numpy()
        # X_ids = corresp_used["nn_vertex_ids"].cpu().numpy()
        corresp_scores = corresp_final["coord_conf"]
            
        # best_pose = coarse_poses[best_coarse_pose_id]
        
        # # # # ------------------ 输出位姿 ------------------
        # dataset_model_dir = os.path.join(AppConfig.DATASETS_PATH, opts.object_dataset, "models")
        # model_path = os.path.join(dataset_model_dir, f"{opts.object_dataset}_norm.obj")
        # pose_path = os.path.join(AppConfig.REFINE_ROOT, opts.object_dataset, "pose_es")
        # vis_util.visualize_pose_overlay(
        #     image_raw=image_raw,
        #     best_pose=best_pose,
        #     model_path=model_path,
        #     orig_camera=orig_camera_c2w,
        #     crop_camera=crop_camera_model_c2w,
        #     frame_name=frame_name,
        #     output_dir=pose_path
        # )
            
        # 恢复裁剪位姿和2d点
        best_pose, q_2d, q_crop = recover_pose_and_points_to_orig(
            best_pose=best_coarse_pose,
            q_2d=q_2d,
            X_3d=X_3d,
            crop_camera=camera_c2w,
            orig_camera=orig_camera_c2w,
            use_crop=opts.crop,
        )
            
        final_q_2d = q_2d
        final_X_3d = X_3d
        # final_X_ids = X_ids
        all_scores = corresp_scores
            
        # #重复3d-2d对应剪枝
        # unique_ids = np.unique(final_X_ids)
        # best_indices = []
        # for uid in unique_ids:
        #     # 找到所有对应该 3D 点的索引
        #     idxs = np.where(final_X_ids == uid)[0]
        #     if len(idxs) == 1:
        #         # 只有一个对应，直接保留
        #         best_indices.append(idxs[0])
        #     else:
        #         # 多个对应，选择置信度最高的
        #         best_idx = idxs[np.argmax(all_scores[idxs].cpu().numpy())]
        #         best_indices.append(best_idx)
    
        # best_indices = np.array(best_indices)
        # final_q_2d = final_q_2d[best_indices]
        # final_X_3d = final_X_3d[best_indices]
        # final_X_ids = final_X_ids[best_indices]
        # all_scores = all_scores[best_indices]
        # q_crop = q_crop[best_indices]
            
        correspondence_data = CorrespondenceData(
            frame_name=frame_name,
            query_2d_pts=final_q_2d,
            model_3d_pts=final_X_3d,
            # model_feat_ids=final_X_ids,
            best_pose = best_pose,
            camera_c2w= orig_camera_c2w
        )
        vis_debug_path = os.path.join(AppConfig.REFINE_ROOT, opts.object_dataset, "vis_debug")
        visualize_correspondences(
            data=correspondence_data,
            template_id=template_id,
            image_raw=image_raw,
            repre=repre,
            template_base_dir=template_base_dir,
            output_dir=vis_debug_path,
            num_batches=1
        )
        # 创建并返回此帧的对应数据
        return correspondence_data