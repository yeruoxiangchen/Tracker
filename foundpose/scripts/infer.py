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

            # ------------------ 在这里接入渲染 ------------------
            if best_coarse_quality is not None:
                best_pose = coarse_poses[best_coarse_pose_id]
                R_m2c = best_pose["R_m2c"]  # (3,3)
                t_m2c = best_pose["t_m2c"].reshape(3)  # (3,)

                # 读取模型
                model_tpath = "/home/zjr/Tracker/foundpose/datasets/wogua/models/wogua_mm.ply"
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
                os.makedirs(out_dir_vis, exist_ok=True)

                # 保存时用 frame_name 来保持一致的文件名
                out_path = os.path.join(out_dir_vis, f"{frame_name}.png")
                cv2.imwrite(out_path, vis_img)   


def main() -> None:
    
    opts_path = "/home/zjr/Tracker/foundpose/configs/infer/wogua.json"
    rgb_dir = "/home/zjr/Tracker/foundpose/datasets/wogua/rgb"
    mask_dir = "/home/zjr/Tracker/foundpose/datasets/wogua/masks"

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
        frame_name = os.path.splitext(rgb_file)[0]  # "0000" 这种
        image_path = os.path.join(rgb_dir, rgb_file)
        mask_path = os.path.join(mask_dir, mask_file)

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
        infer(
            opt_file_path=opts_path,
            image_raw=image,
            mask_raw=mask,
            camK=camK,
            frame_name=frame_name,
        )


if __name__ == "__main__":
    main()