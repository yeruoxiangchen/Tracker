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

# --- 核心误差函数 ---
def global_reprojection_error(params, all_data_for_optimization):
    """
    计算所有帧的总重投影误差（残差向量）。
    
    优化参数 params: [s, rx, ry, rz, tx, ty, tz]
        s: 尺度因子 (1 维)
        r: 模型到世界坐标系的旋转 (轴角表示, 3 维)
        t: 模型到世界坐标系的平移 (3 维)
        
    返回: 形状为 (N_total_points * 2,) 的残差向量 (u - u_obs, v - v_obs, ...)
    """
    
    s = params[0]
    r_vec = params[1:4]
    t_vec = params[4:7]
    
    # M -> W 变换矩阵
    R_M2W = axis_angle_to_rotmat(r_vec)
    T_M2W = np.eye(4, dtype=np.float32)
    T_M2W[:3, :3] = s * R_M2W # 尺度只应用于旋转部分
    T_M2W[:3, 3] = t_vec
    
    all_residuals = []
    
    for data in all_data_for_optimization:
        K = data['K']
        T_W2C = data['T_w2c'] # COLMAP 相机位姿 (W -> C)
        X_3d_M = data['X_3d'] # 3D 模型点 (M 坐标系)
        q_2d_obs = data['q_2d'] # 2D 观测点 (像素)
        
        # 1. 3D 点从 M 坐标系变换到 W 坐标系
        # X_M (N, 3) -> X_W (N, 3)
        X_3d_M_h = np.concatenate([X_3d_M, np.ones((X_3d_M.shape[0], 1))], axis=1) # 齐次坐标 (N, 4)
        X_3d_W_h = (T_M2W @ X_3d_M_h.T).T # (4, 4) @ (4, N) -> (4, N).T -> (N, 4)
        X_3d_W = X_3d_W_h[:, :3] # (N, 3)
        
        # 2. 3D 点从 W 坐标系变换到 C 坐标系
        # X_W (N, 3) -> X_C (N, 3)
        X_3d_W_h = X_3d_W_h # 已经有齐次坐标 (N, 4)
        X_3d_C_h = (T_W2C @ X_3d_W_h.T).T
        X_3d_C = X_3d_C_h[:, :3] # (N, 3)
        
        # 3. 投影到 2D 图像平面
        q_2d_pred = project_points(K, X_3d_C)
        
        # 4. 计算残差
        residuals = (q_2d_pred - q_2d_obs).flatten() # (N*2,)
        all_residuals.append(residuals)
        
    return np.concatenate(all_residuals)

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
                corresp_curr["coord_3d"] = corresp_curr["coord_3d"] / 1000.0
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
            
            # 测试pose
            # ------------------ 测试 ------------------
            # if best_coarse_quality is not None:
            #     best_pose = coarse_poses[best_coarse_pose_id]
            #     R_m2c = best_pose["R_m2c"]  # (3,3)
            #     t_m2c = best_pose["t_m2c"].reshape(3)  # (3,)

            #     # 读取模型
            #     model_tpath = "/home/zjr/Tracker/foundpose/datasets/wogua/models/wogua_mm.ply"
            #     import trimesh
            #     model_mesh = trimesh.load(model_tpath, force='mesh')
            #     verts = np.asarray(model_mesh.vertices, dtype=np.float32)

            #     # 投影 3D 点到 2D
            #     Xc = (R_m2c @ verts.T).T + t_m2c  # (N,3)
                
            #     Xc_h = np.hstack([Xc, np.ones((Xc.shape[0],1))])
            #     verts_world = (crop_camera_model_c2w.T_world_from_eye @ Xc_h.T).T[:, :3]  # (N,3)
            #     Xc_orig = (np.linalg.inv(orig_camera_c2w.T_world_from_eye) @ np.hstack([verts_world, np.ones((verts_world.shape[0],1))]).T).T[:, :3]

            #     fx, fy = orig_camera_c2w.f
            #     cx, cy = orig_camera_c2w.c

            #     u = (fx * Xc_orig[:,0] / Xc_orig[:,2]) + cx
            #     v = (fy * Xc_orig[:,1] / Xc_orig[:,2]) + cy
            #     pts_2d = np.stack([u,v], axis=1).astype(int)
            #     # 复制一份原图
            #     # vis_img = image_np_hwc.astype(np.uint8).copy()
            #     vis_img = image_raw.astype(np.uint8).copy()

            #     overlay = vis_img.copy()  # 复制一层用于绘制
            #     alpha = 0.4  # 透明度 0~1

            #     for f in model_mesh.faces:
            #         p1, p2, p3 = pts_2d[f]
            #         cv2.line(overlay, tuple(p1), tuple(p2), (150,25,120), 1)
            #         cv2.line(overlay, tuple(p2), tuple(p3), (150,25,120), 1)
            #         cv2.line(overlay, tuple(p3), tuple(p1), (150,25,120), 1)

            #     # 将 overlay 叠加回原图
            #     cv2.addWeighted(overlay, alpha, vis_img, 1-alpha, 0, vis_img)

            #     out_dir_vis = os.path.join(output_dir, "infer_rgb")
            #     out_dir_vis = "/home/zjr/Tracker/foundpose/results/refine_model/wogua/pose_es"
            #     os.makedirs(out_dir_vis, exist_ok=True)

            #     # 保存时用 frame_name 来保持一致的文件名
            #     out_path = os.path.join(out_dir_vis, f"{frame_name}.png")
            #     cv2.imwrite(out_path, vis_img) 
          
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
            # # ========== 投影 3D 点到原图坐标 ==========
            # X_cam_orig = (R_m2c_orig @ X_3d.T) + t_m2c_orig  # (3, N)
            # X_cam_orig = X_cam_orig.T  # (N, 3)
            # # Step 2. 过滤掉深度太小或负的点
            # valid_mask = X_cam_orig[:, 2] > 1e-6
            # X_cam_orig = X_cam_orig[valid_mask]
            # q_2d_valid = q_2d[valid_mask]
            # # Step 3. 使用原相机内参进行投影
            # fx, fy = orig_camera_c2w.f
            # cx, cy = orig_camera_c2w.c
            # u = (fx * X_cam_orig[:, 0] / X_cam_orig[:, 2]) + cx
            # v = (fy * X_cam_orig[:, 1] / X_cam_orig[:, 2]) + cy
            # proj_2d = np.stack([u, v], axis=1).astype(int)
            # # ========== 可视化 ==========
            # vis_img = image_raw.copy()
            # for p in proj_2d:
            #     cv2.circle(vis_img, tuple(p), 3, (0, 255, 0), -1)  # 投影点 (绿色)
            # for q in q_2d_valid.astype(int):
            #     cv2.circle(vis_img, tuple(q), 3, (0, 0, 255), -1)  # 原始匹配点 (红色)
            # # 叠加可视化
            # out_dir_vis = "/home/zjr/Tracker/foundpose/results/refine_model/wogua/vis_proj"
            # os.makedirs(out_dir_vis, exist_ok=True)
            # out_path = os.path.join(out_dir_vis, f"{frame_name}.png")
            # cv2.imwrite(out_path, vis_img)
            
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
                
                # ## 2d-2d验证
                # visualize_2d_2d_correspondences(
                #     q_crop=q_crop,
                #     q_orig=q_2d,
                #     image_np_hwc=image_np_hwc,
                #     orig_image_np_hwc=orig_image_np_hwc,
                #     base_frame=frame_name,
                # )

            final_q_2d = q_2d
            final_X_3d = X_3d
            final_X_ids = X_ids
            all_scores = corresp_scores
            unique_ids, first_occurrence_indices, id_map = np.unique(
                        final_X_ids, 
                        return_index=True, 
                        return_inverse=True
                    )
            best_match_indices = first_occurrence_indices.copy()
            # for i in range(len(final_X_ids)):
            #             # 找到当前匹配对应的唯一 ID 在 unique_ids 数组中的索引 (j)
            #             unique_id_idx = id_map[i]
                        
            #             # 检查当前匹配的分数是否优于目前记录的该 ID 的最佳分数
            #             current_best_score_index = best_match_indices[unique_id_idx]
                        
            #             if all_scores[i] > all_scores[current_best_score_index]:
            #                 # 如果当前分数更高，则更新最佳匹配的索引
            #                 best_match_indices[unique_id_idx] = i
            # final_q_2d_dedup = final_q_2d[best_match_indices]
            # final_X_3d_dedup = final_X_3d[best_match_indices]
            # final_X_ids_dedup = final_X_ids[best_match_indices]
            
            # print(f"原始对应关系数量: {len(final_X_ids)}")
            # print(f"去重后的对应关系数量: {len(final_X_ids_dedup)} (基于最佳匹配)")

            # # 替换最终数据
            # final_q_2d = final_q_2d_dedup
            # final_X_3d = final_X_3d_dedup
            # final_X_ids = final_X_ids_dedup               
            # best_pose = coarse_poses[best_coarse_pose_id]
            # 创建并返回此帧的对应数据
            return CorrespondenceData(
                frame_name=frame_name,
                query_2d_pts=final_q_2d,
                model_3d_pts=final_X_3d,
                model_feat_ids=final_X_ids,
                best_pose = best_pose,
                camera_c2w= orig_camera_c2w
            )

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


def main() -> None:
    
    opts_path = "/home/zjr/Tracker/foundpose/configs/infer/wogua.json"
    rgb_dir = "/home/zjr/Tracker/foundpose/datasets/wogua/rgb"
    mask_dir = "/home/zjr/Tracker/foundpose/datasets/wogua/masks"
    colmap_dir = "/home/zjr/Tracker/foundpose/datasets/wogua/sparse/0" # COLMAP 稀疏重建目录
    model_path = "/home/zjr/Tracker/foundpose/datasets/wogua/models/wogua.ply" # 模型文件路径
    output_vis_dir = "/home/zjr/Tracker/foundpose/results/refine_model/wogua" # 结果输出路径
    
    model_vertices = load_model_vertices(model_path)
    if model_vertices.size == 0:
        logger.error("Model vertices could not be loaded. Aborting.")
        return
    
    # --- 预加载 COLMAP 位姿 ---
    images_txt = os.path.join(colmap_dir, 'images.txt')
    images_colmap = read_colmap_images_txt(images_txt)
    
    # colmap的points3D.txt是三维物体的世界坐标，image.txt是每帧图像的相机位姿，也就是把世界到模型的位姿w2c
    
    # 存储所有帧的对应关系和几何信息
    all_corresps_data = [] # 存储 CorrespondenceData 对象
    all_optimization_data = [] # 存储用于优化的字典数据 (q_2d, X_3d, K, T_w2c)

    # 按照文件名顺序读取所有帧
    rgb_files = sorted([f for f in os.listdir(rgb_dir) if f.endswith(".jpg")])
    mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith(".png")])

    assert len(rgb_files) == len(mask_files), "RGB 和 Mask 数量不一致！"
    from tqdm import tqdm
    for rgb_file, mask_file in tqdm(
        zip(rgb_files, mask_files), 
        total=len(rgb_files), 
        desc="Processing frames"
    ):
        frame_name = rgb_file  # "0000" 这种
        image_path = os.path.join(rgb_dir, rgb_file)
        mask_path = os.path.join(mask_dir, mask_file) # 检查 COLMAP 位姿是否存在
        
        if frame_name not in images_colmap:
            logger.warn(f"Frame {frame_name} not found in COLMAP data. Skipping.")
            continue

        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转为 RGB
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # 相机内参 (假设不随帧变化)
        h, w = image.shape[:2]
        scale = 1.5
        fx = fy = scale * max(h, w)
        cx = w / 2
        cy = h / 2
        camK = np.array([[fx, 0, cx],
                         [0, fy, cy],
                         [0, 0, 1]], dtype=np.float32)

        # 调用 infer
        corresp_data = infer(
                opt_file_path=opts_path,
                image_raw=image,
                mask_raw=mask,
                camK=camK,
                frame_name=frame_name,
            )
        
        if corresp_data is not None:
            all_corresps_data.append(corresp_data)

            # 准备此帧的优化输入数据
            img_entry = images_colmap[frame_name]
            R_colmap = quat_to_rotmat(img_entry['qvec'])  # W -> C Rotation
            t_colmap = img_entry['tvec'].astype(np.float32) # W -> C Translation
            
            # T_w2c = [R_colmap | t_colmap]
            T_w2c = np.eye(4, dtype=np.float32)
            T_w2c[:3, :3] = R_colmap
            T_w2c[:3, 3] = t_colmap 
            
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
            
        logger.info(f"Collected correspondences from {len(all_optimization_data)} frames.")
    
    # # 1. 初始化 T_M2W
    # --- 分批设置 ---
    BATCH_SIZE = 5  # 每批处理的帧数
    num_frames = len(all_optimization_data)
    num_batches = int(np.ceil(num_frames / BATCH_SIZE))
    
    T_M2W_final = np.eye(4, dtype=np.float32)
    # 初始化 T_M2W
    T_M2W_current = None
    s_current = 0
    
    for batch_idx in range(num_batches):
        logger.info(f"--- Starting optimization for Batch {batch_idx + 1}/{num_batches} ---")
        
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min((batch_idx + 1) * BATCH_SIZE, num_frames)
        current_batch_data = all_optimization_data[start_idx:end_idx]
        
        if not current_batch_data:
            continue
            
        # 1. 确定初始参数 (initial_params)
        if T_M2W_current is None:
            # --- 第一批：使用您的原始初始化逻辑 ---
            first_data = current_batch_data[0]
            T_w2c_0 = first_data['T_w2c']
            T_c2w_0 = np.linalg.inv(T_w2c_0)
            # 简化：假设 T_m2c 接近 identity（模型中心在相机原点）
            R_m2c = all_corresps_data[0].best_pose["R_m2c"]
            t_m2c = all_corresps_data[0].best_pose["t_m2c"]
            T_m2c = np.eye(4)
            T_m2c[:3, :3] = R_m2c
            T_m2c[:3, 3] = t_m2c
            T_m2w_init = T_c2w_0 @ T_m2c
            R_init = T_m2w_init[:3, :3]
            t_init = T_m2w_init[:3, 3]
            s_init = np.linalg.norm(R_init) / np.sqrt(3)  # 近似尺度
            R_init /= s_init
            
            # --- 结束初始化 ---
            from scipy.spatial.transform import Rotation
            r_vec_init = Rotation.from_matrix(R_init).as_rotvec()
            initial_params = np.array([s_init, r_vec_init[0], r_vec_init[1], r_vec_init[2], 
                                    t_init[0], t_init[1], t_init[2]], dtype=np.float64)
        else:
            # --- 后续批次：使用前一批次的优化结果作为初始化 ---
            # s_prev = T_M2W_current[0, 0] / np.linalg.norm(T_M2W_current[:3, 0]) # 重新计算尺度
            s_prev = s_current
            R_prev = T_M2W_current[:3, :3] / s_prev
            t_prev = T_M2W_current[:3, 3]
            r_vec_prev = Rotation.from_matrix(R_prev).as_rotvec()
            initial_params = np.array([s_prev, r_vec_prev[0], r_vec_prev[1], r_vec_prev[2], 
                                    t_prev[0], t_prev[1], t_prev[2]], dtype=np.float64)
            logger.info(f"Initialized with previous batch result (s={s_prev:.4f}).")

        # 2. 调用非线性最小二乘优化
        from scipy.optimize import least_squares
        result = least_squares(
            fun=global_reprojection_error, 
            x0=initial_params, 
            args=(current_batch_data,), 
            method='lm',
            # loss='soft_l1'
        )
        
        # 3. 提取优化结果
        final_params = result.x
        s_final = final_params[0]
        R_M2W_final = axis_angle_to_rotmat(final_params[1:4])
        t_M2W_final = final_params[4:7]
        
        print(f"Optimized Scale (s): {s_final}")
        print(f"Optimized R_M2W:\n{R_M2W_final}")
        print(f"Optimized t_M2W:\n{t_M2W_final}")
        
        logger.info(f"Batch {batch_idx + 1} Optimized Scale (s): {s_final:.4f}")
        
        # 4. 构造最终的 T_M2W 矩阵并更新
        
        T_M2W_final[:3, :3] = s_final * R_M2W_final 
        T_M2W_final[:3, 3] = t_M2W_final
        T_M2W_current = T_M2W_final # 存储用于下一个batch的初始化
        s_current = s_final
        # 5. 可视化 (可选: 可以在每个 batch 结束后对该 batch 的帧进行可视化)
        for data in current_batch_data:
            visualize_projection(
                image_path=data['image_path'],
                output_dir=output_vis_dir, # 假设 output_vis_dir 仍然可用
                K=data['K'],
                T_W2C=data['T_w2c'],
                T_M2W=T_M2W_final, # 使用该 batch 优化的 T_M2W
                model_vertices=model_vertices # 假设 model_vertices 仍然可用
            )
    
    # 对每帧数据进行可视化
    for data in tqdm(all_optimization_data, desc="Visualizing frames"):
        # 提取帧特定的数据
        T_W2C = data['T_w2c']
        K = data['K']
        image_path = data['image_path']
        X_3d = data['X_3d']
        
        # 调用可视化函数
        visualize_projection(
            image_path=image_path,
            output_dir=output_vis_dir,
            K=K,
            T_W2C=T_W2C,
            T_M2W=T_M2W_final, # 使用全局优化的最终位姿
            model_vertices=model_vertices
        )
        
    logger.info("Visualization complete.")

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