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
        eps = 1e-6
        Z = X_C[:, 2].copy()
        mask = Z > eps
        res = np.zeros((X_C.shape[0], 2), dtype=np.float32)
        # 只对前方点计算残差
        res[mask] = uv[mask] - q_2d[mask]
        # 对 Z<=eps 的点给一个大残差
        res[~mask] = 100000.0  # 任选一个大的惩罚值
        res = res.reshape(-1)
        residuals.append(res)

    return np.concatenate(residuals)

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
        pose_evaluator = eval_util.EvaluatorPose([object_lid])

        base_repre_dir = os.path.join(bop_config.output_path, "object_repre")
        repre_dir = repre_util.get_object_repre_dir_path(
            base_repre_dir, opts.version, opts.object_dataset, object_lid
        )
        repre = repre_util.load_object_repre(
            repre_dir=repre_dir,
            tensor_device=device,
        )

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
                        
            ## =============================================
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
            q_crop = q_crop[best_indices]
            ##=================================================
            # 1. 提取原图2D点和3D点
            q_2d = q_crop   # 原图2D点
            m_3d = final_X_3d   # 3D点
            # 2. 获取对应模板相机
            template_id = int(corresp_final["template_id"].item()) if "template_id" in corresp_final else 0
            camera = repre.template_cameras_cam_from_model[template_id]
            # 3. 提取相机参数
            f = np.array(camera.f)         # [fx, fy]
            c = np.array(camera.c)         # [cx, cy]
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
            template_dir = "/home/zjr/Tracker/foundpose/results/templates/v1/wogua/1"
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
            save_dir = "/home/zjr/Tracker/foundpose/results/refine_model/wogua/2d-3d-template"
            frame_dir = os.path.join(save_dir, frame_name)
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
                image_with_2d = (image_np_hwc * 255).astype(np.uint8).copy()
                template_img_copy = template_img.copy()

                # 绘制点和编号
                for idx, (pt2d, pt3d) in enumerate(zip(batch_q, batch_proj), start=start_idx):
                    # 绘制点
                    cv2.circle(image_with_2d, tuple(pt2d), 1, (0, 255, 0), -1)       # 原图2D点，绿色
                    cv2.circle(template_img_copy, tuple(pt3d), 1, (255, 0, 0), -1)    # 模板投影点，红色

                    # 绘制编号
                    cv2.putText(image_with_2d, str(idx), (pt2d[0]+1, pt2d[1]-1),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 255, 0), 1, cv2.LINE_AA)
                    cv2.putText(template_img_copy, str(idx), (pt3d[0]+3, pt3d[1]-3),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.2, (255, 0, 0), 1, cv2.LINE_AA)

                # 拼接
                combined_img = np.hstack([image_with_2d, template_img_copy])

                # 保存每批图像到 frame_name 目录
                save_path = os.path.join(frame_dir, f"{best_corresp_id:04d}_batch{batch_idx+1}.png")
                cv2.imwrite(save_path, cv2.cvtColor(combined_img, cv2.COLOR_RGB2BGR))
                # print(f"Saved comparison batch {batch_idx+1} to {save_path}")
                
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
    base_dir = "/home/zjr/Tracker/foundpose/datasets/wogua"
    opts_path = "/home/zjr/Tracker/foundpose/configs/infer/wogua.json"
    rgb_dir = os.path.join(base_dir, "rgb")
    mask_dir = os.path.join(base_dir, "masks")
    colmap_dir = os.path.join(base_dir, "sparse/0")
    model_path = os.path.join(base_dir, "models/wogua_mm.ply")
    output_vis_dir = "/home/zjr/Tracker/foundpose/results/refine_model/wogua"
    data_cache_path = os.path.join(os.path.dirname(opts_path), "cached_optimization_data_infer8 copy.pkl")
    
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
            camK = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

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
                
                save_dir = "/home/zjr/Tracker/foundpose/results/refine_model/wogua"
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
    
    # model_vertices = load_model_vertices(model_path)
    model_vertices, model_faces = load_ply_vertices_faces(model_path)
    if model_vertices.size == 0:
        logger.error("Model vertices could not be loaded. Aborting.")
        return
            
    # # 1. 初始化 T_M2W
    first_data = all_optimization_data[0]

    R_m2c = all_corresps_data[10].best_pose["R_m2c"]
    t_m2c = all_corresps_data[10].best_pose["t_m2c"]
    T_m2c = np.eye(4)
    T_m2c[:3, :3] = R_m2c
    T_m2c[:3, 3] = t_m2c
    
    T_w2c = all_optimization_data[10]['T_w2c']
    # print(T_m2c)
    T_c2w = np.linalg.inv(T_w2c)
    T_m2w_init = T_c2w @ T_m2c
    R_init = T_m2w_init[:3, :3]
    t_init = T_m2w_init[:3, 3]
    s_init = 1.0 # 可能得改
    
    # for data in tqdm(all_optimization_data, desc="Final Visualizing"):
    #     T_W2C = data['T_w2c']
    #     K = data['K']
    #     image_path = data['image_path']
    #     # X_m = data['X_3d']  # 模型坐标下的3D点 (N, 3)
    #     # X_c = (R_m2c @ X_m.T).T + t_m2c  # 结果是 (N, 3)
    #     # 使用最后一批优化的 T_M2W_final 进行全序列可视化
    #     visualize_projection(
    #         image_path=image_path,
    #         output_dir="/home/zjr/Tracker/foundpose/results/refine_model/wogua/init_model",
    #         K=K,
    #         T_W2C=T_W2C,
    #         T_M2W=T_m2w_init,
    #         model_vertices=model_vertices
    #     )
        
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
            visualize_projection_mesh(
                image_path=image_path,
                output_dir=output_vis_dir,
                K=K,
                T_W2C=T_W2C,
                T_M2W=T_M2W_final,
                model_vertices=model_vertices,
                model_faces=model_faces
            )
        
    logger.info("All processes complete.")

def load_ply_vertices_faces(path):
    import open3d as o3d # 用于读取PLY文件
    mesh = o3d.io.read_triangle_mesh(path)
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    return vertices, faces

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
    
def visualize_projection_mesh(
    image_path: str,
    output_dir: str,
    K: np.ndarray,
    T_W2C: np.ndarray,
    T_M2W: np.ndarray,
    model_vertices: np.ndarray,
    model_faces: np.ndarray,   # <-- 新增：渲染用 mesh faces
    color=(0, 255, 0)
):
    """
    将模型线框渲染到图像上。
    T_M2W 是 [sR | t]。
    """

    # 1. 载入图像
    image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image_bgr is None:
        print(f"Could not load image: {image_path}")
        return

    # 2. 顶点 M → W → C
    X_3d_M_h = np.concatenate([model_vertices, np.ones((model_vertices.shape[0], 1))], axis=1)
    X_3d_W_h = (T_M2W @ X_3d_M_h.T).T
    X_3d_C_h = (T_W2C @ X_3d_W_h.T).T
    X_3d_C = X_3d_C_h[:, :3]

    # 3. 投影
    q_2d_proj = project_points(K, X_3d_C)

    # 图像尺寸检查
    h, w = image_bgr.shape[:2]

    # 创建输出图像
    output_image = image_bgr.copy()

    # 4. 渲染 mesh（线框）
    for f in model_faces:
        v0, v1, v2 = f
        pts = q_2d_proj[f]  # (3,2)

        # 深度检查：三个点都在相机前方
        if np.any(X_3d_C[f, 2] <= 0):
            continue

        # 在图像范围检查
        if np.any(pts[:, 0] < 0) or np.any(pts[:, 0] >= w) or \
           np.any(pts[:, 1] < 0) or np.any(pts[:, 1] >= h):
            continue

        p0 = tuple(pts[0].astype(int))
        p1 = tuple(pts[1].astype(int))
        p2 = tuple(pts[2].astype(int))

        cv2.line(output_image, p0, p1, color, 1)
        cv2.line(output_image, p1, p2, color, 1)
        cv2.line(output_image, p2, p0, color, 1)

    # 5. 保存
    output_dir = os.path.join(output_dir, "fres")
    os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.basename(image_path)
    output_path = os.path.join(output_dir, f"mesh_{base_name}")

    cv2.imwrite(output_path, output_image)

    
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

if __name__ == "__main__":
    main()