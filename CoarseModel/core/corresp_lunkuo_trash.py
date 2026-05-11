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
from typing import List, Optional, Dict

from core.util import get_instances_from_mask

logger: logging.Logger = logging.get_logger()


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
    repre,
    template_base_dir: str,
    output_dir: str,
    num_batches: int = 5
):
    frame_id = os.path.splitext(data.frame_name)[0] if data.frame_name else "frame"
    frame_dir = os.path.join(output_dir, frame_id)
    os.makedirs(frame_dir, exist_ok=True)

    # ---------- 1. 准备左图 (原图) ----------
    if image_raw.max() <= 1.0:
        img_left_base = (image_raw * 255).astype(np.uint8)
    else:
        img_left_base = image_raw.astype(np.uint8).copy()

    # ---------- 2. 准备右图 (模板图) 并预先 Resize ----------
    tpl_camera = repre.template_cameras_cam_from_model[template_id]
    tpl_path = os.path.join(template_base_dir, f"rgb/template_{template_id:04d}.png")

    if os.path.exists(tpl_path):
        img_right_base = cv2.imread(tpl_path)
        img_right_base = cv2.cvtColor(img_right_base, cv2.COLOR_BGR2RGB)
    else:
        img_right_base = np.zeros(
            (tpl_camera.height, tpl_camera.width, 3), dtype=np.uint8
        )

    # ★ 修复：先计算缩放比例并调整右图，确保坐标和图像对齐
    scale_factor = 1.0
    if img_left_base.shape[0] != img_right_base.shape[0]:
        scale_factor = img_left_base.shape[0] / img_right_base.shape[0]
        new_w = int(img_right_base.shape[1] * scale_factor)
        # 先 Resize 图片
        img_right_base = cv2.resize(img_right_base, (new_w, img_left_base.shape[0]))

    # ---------- 3. 3D → 模板相机投影 ----------
    pts_model = data.model_3d_pts
    pts_h = np.hstack([pts_model, np.ones((pts_model.shape[0], 1))])

    # 投影到模板相机坐标系
    T_cam_from_model = np.linalg.inv(tpl_camera.T_world_from_eye)
    pts_cam = (T_cam_from_model @ pts_h.T).T[:, :3]

    # 过滤深度 < 0 的点
    valid = pts_cam[:, 2] > 1e-6
    pts_cam = pts_cam[valid]
    pts_2d = data.query_2d_pts[valid]

    # 投影到像素平面
    f = np.array(tpl_camera.f)
    c = np.array(tpl_camera.c)
    
    proj_2d = np.zeros((pts_cam.shape[0], 2))
    proj_2d[:, 0] = (pts_cam[:, 0] / pts_cam[:, 2]) * f[0] + c[0]
    proj_2d[:, 1] = (pts_cam[:, 1] / pts_cam[:, 2]) * f[1] + c[1]

    # ★ 修复：将投影点坐标也乘以缩放比例
    if scale_factor != 1.0:
        proj_2d = proj_2d * scale_factor

    proj_2d = proj_2d.astype(int)
    query_2d = pts_2d.astype(int)

    # ---------- 4. 分批绘制 ----------
    num_points = query_2d.shape[0]
    batch_size = max(1, (num_points + num_batches - 1) // num_batches)

    for b in range(num_batches):
        s = b * batch_size
        e = min((b + 1) * batch_size, num_points)
        if s >= e: break

        img_left = img_left_base.copy()
        img_right = img_right_base.copy() # 这里已经是 Resize 过的了

        for i in range(s, e):
            u, v = query_2d[i]
            x, y = proj_2d[i]

            # 绘制左图
            if 0 <= u < img_left.shape[1] and 0 <= v < img_left.shape[0]:
                cv2.circle(img_left, (u, v), 3, (0, 255, 0), -1)
                # 可选：画个十字更准
                # cv2.drawMarker(img_left, (u, v), (0, 255, 0), cv2.MARKER_CROSS, 5)

            # 绘制右图
            if 0 <= x < img_right.shape[1] and 0 <= y < img_right.shape[0]:
                cv2.circle(img_right, (x, y), 3, (255, 0, 0), -1)
                # cv2.drawMarker(img_right, (x, y), (255, 0, 0), cv2.MARKER_CROSS, 5)

        # 拼接 (高度已经对齐)
        combined = np.hstack([img_left, img_right])
        save_path = os.path.join(frame_dir, f"batch_{b+1:02d}.png")
        cv2.imwrite(save_path, cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))

    logger.info(f"Visualized {num_points} correspondences → {frame_dir}")

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


def map_points_crop_to_orig(points_2d, crop_cam, orig_cam):
    """
    将裁剪图(crop_cam)中的2D点映射回原图(orig_cam)。
    原理：利用内参矩阵 K 进行反投影。
    假设 crop_cam 和 orig_cam 的旋转/平移一致(即光心重合)，只是视口(Intrinsics)不同。
    """
    if points_2d.shape[0] == 0:
        return points_2d

    # 1. 获取内参矩阵
    K_crop = get_camera_matrix(crop_cam)
    K_orig = get_camera_matrix(orig_cam)

    # 2. 齐次化坐标 (u, v) -> (u, v, 1)
    pts_h = np.hstack([points_2d, np.ones((len(points_2d), 1))])

    # 3. 反投影归一化平面: K_crop_inv @ pixel_crop
    #    然后投影回原图: K_orig @ normalized_point
    #    合并为: H = K_orig @ K_crop_inv
    H = K_orig @ np.linalg.inv(K_crop)
    
    pts_orig_h = (H @ pts_h.T).T
    
    # 4. 去齐次化
    pts_orig = pts_orig_h[:, :2] / pts_orig_h[:, 2:3]
    
    return pts_orig.astype(np.float32)

def extract_mixed_correspondences(
    image_raw: np.ndarray,
    mask_raw: np.ndarray,
    camK: np.ndarray, 
    template_id: int,
    repre,
    template_base_dir: str,
    opts,
):
    
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
            for template_idx in range(len(repre.template_cameras_cam_from_model)):
                tpl_feat_mask = repre.feat_to_template_ids == template_idx # 找出属于该模板的特征点。
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

            # ★ 5. 只针对指定 template_id 建立 KNN（关键改动）
            # ------------------------------------------------------------
            tpl_feat_mask = repre.feat_to_template_ids == template_id
            tpl_feat_ids = torch.nonzero(tpl_feat_mask).flatten().to(repre.feat_to_vertex_ids.device)
            # 检查该模板是否有特征点
            if len(tpl_feat_ids) == 0:
                raise ValueError(f"Template ID {template_id} has no associated features.")

            tpl_feats = repre.feat_vectors[tpl_feat_ids]
            tpl_vertex_ids = repre.feat_to_vertex_ids[tpl_feat_ids]

            template_knn = knn_util.KNN(k=1, metric="l2")
            template_knn.fit(tpl_feats.cpu())
            
            # ------------------------------------------------------------
            # ★ 6. 特征点 2D–3D 对应（无模板竞争）
            # ------------------------------------------------------------
            nn_dists, nn_ids = template_knn.search(query_features_proj.cpu())
            nn_ids = nn_ids.squeeze(1)

            feat_2d = query_points.cpu().numpy()
            feat_vertex_ids = tpl_vertex_ids[nn_ids]
            feat_3d = repre.vertices[feat_vertex_ids].cpu().numpy()
            
            # ------------------------------------------------------------
            # ★ 7. 轮廓点 2D–3D 对应
            # ------------------------------------------------------------
            tpl_mask = cv2.imread(
                os.path.join(template_base_dir, f"mask/template_{template_id:04d}.png"),
                cv2.IMREAD_GRAYSCALE,
            )
            tpl_depth = cv2.imread(
                os.path.join(template_base_dir, f"depth/template_{template_id:04d}.png"),
                cv2.IMREAD_UNCHANGED,
            ).astype(np.float32)

            tpl_cam = repre.template_cameras_cam_from_model[template_id]

            contours_in, _ = cv2.findContours(
                mask_modal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )
            contours_tpl, _ = cv2.findContours(
                tpl_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )
            contour_2d = []
            contour_3d = []

            if contours_in and contours_tpl:
                cnt_in = max(contours_in, key=cv2.contourArea)
                cnt_tpl = max(contours_tpl, key=cv2.contourArea)

                num_pts = min(40, len(cnt_in), len(cnt_tpl))
                idx = np.linspace(0, num_pts - 1, num_pts).astype(int)

                for i in idx:
                    u, v = cnt_in[i, 0]
                    tu, tv = cnt_tpl[i, 0]
                    z = tpl_depth[int(tv), int(tu)]
                    if z <= 0:
                        continue

                    X = (tu - cx) * z / fx
                    Y = (tv - cy) * z / fy
                    T_c2m = np.linalg.inv(tpl_cam.T_world_from_eye)
                    P_cam = np.array([X, Y, z, 1.0])
                    P_obj = (T_c2m @ P_cam)[:3]

                    contour_2d.append([u, v])
                    contour_3d.append(P_obj)
                    
            # 修正后的逻辑：
            if opts.crop:
                # 使用相机内参变换，自动处理 offset 和 scale
                feat_2d = map_points_crop_to_orig(feat_2d, camera_c2w, orig_camera_c2w)
                
                # 如果有轮廓点，也需要变换
                if len(contour_2d) > 0:
                    contour_2d = np.asarray(contour_2d, dtype=np.float32)
                    contour_2d = map_points_crop_to_orig(contour_2d, camera_c2w, orig_camera_c2w)
            else:
                # 非 crop 模式，坐标无需变换 (如果 contour_2d 是列表需转 numpy)
                if len(contour_2d) > 0:
                    contour_2d = np.asarray(contour_2d, dtype=np.float32)
            # all_2d = np.vstack([feat_2d, contour_2d])
            # all_3d = np.vstack([feat_3d, contour_3d])
            # 合并点
            if len(contour_2d) > 0:
                all_2d = np.vstack([feat_2d, contour_2d])
                all_3d = np.vstack([feat_3d, contour_3d]) # contour_3d 已经是正确的世界坐标/模型坐标
            else:
                all_2d = feat_2d
                all_3d = feat_3d

            # 直接调用可视化函数
            
            correspondence_data = CorrespondenceData(
                frame_name="",
                query_2d_pts=all_2d,
                model_3d_pts=all_3d,
                model_feat_ids=None,
                best_pose=None,
                camera_c2w=orig_camera_c2w,
            )
            vis_dir = os.path.join(AppConfig.REFINE_ROOT, opts.object_dataset)
            visualize_correspondences(
                data=correspondence_data,
                template_id=template_id,
                image_raw=image_raw,
                repre=repre,
                template_base_dir=template_base_dir,
                output_dir=vis_dir,
                num_batches=1  # 每批显示的点数，可按需调整
            )
            return correspondence_data