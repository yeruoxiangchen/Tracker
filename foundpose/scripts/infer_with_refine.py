#!/usr/bin/env python3

"""Infers pose from objects."""

import datetime

import os
import gc
import time

from typing import List, NamedTuple, Optional, Tuple

import cv2

import numpy as np

import torch

from utils.misc import array_to_tensor, tensor_to_array, tensors_to_arrays

from bop_toolkit_lib import inout, dataset_params
import bop_toolkit_lib.config as bop_config
import bop_toolkit_lib.misc as bop_misc

from utils import (
    corresp_util,
    config_util,
    eval_errors,
    eval_util,
    feature_util,
    infer_pose_util,
    knn_util,
    misc as misc_util,
    pnp_util,
    projector_util,
    repre_util,
    vis_util,
    data_util,
    renderer_builder,
    json_util, 
    logging,
    misc,
    structs,
)

from utils.structs import AlignedBox2f, PinholePlaneCameraModel
from utils.misc import warp_depth_image, warp_image, warp_box

from collections import defaultdict

from gotrack.scripts.run_gotrack import run_inference


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
    debug: bool = False


def load_detections(boxes_filt, image_size, object_id):
    detections = defaultdict(list)

    for i, box in enumerate(boxes_filt):
        cx, cy, w, h = box.tolist()
        img_w, img_h = image_size
        x = (cx - w / 2) * img_w
        y = (cy - h / 2) * img_h
        abs_w = w * img_w
        abs_h = h * img_h

        detections[object_id].append(
            {
                "bbox": [round(x, 1), round(y, 1), round(abs_w, 1), round(abs_h, 1)]
            }
        )

    return detections


def pre_infer(opt_file_path, object_lid):
    opts = config_util.load_opts_from_json(
        path=opt_file_path, 
        opts_types={"infer_opts": InferOpts}
    )["infer_opts"]

    # Prepare feature extractor.
    extractor = feature_util.make_feature_extractor(opts.extractor_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    extractor.to(device)

    # Load the object representation.
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
    template_knn_indices = []

    if opts.match_feat_matching_type == "cyclic_buddies":
        logger.info("Building per-template KNN indices...")
        for template_id in range(len(repre.template_cameras_cam_from_model)):
            # logger.info(f"Building KNN index for template {template_id}...")
            tpl_feat_mask = repre.feat_to_template_ids == template_id
            tpl_feat_ids = torch.nonzero(tpl_feat_mask).flatten()

            template_feats = repre.feat_vectors[tpl_feat_ids]

            # Build knn index for object features.
            template_knn_index = knn_util.KNN(k=1, metric="l2")
            template_knn_index.fit(template_feats.cpu())
            template_knn_indices.append(template_knn_index)
        logger.info("Per-template KNN indices built.")

    return opts, extractor, repre, template_knn_indices, visual_words_knn_index


def infer(
    opts, extractor, foundpose_detection, camK, object_lid, repre, template_knn_indices, visual_words_knn_index, image_raw,
    trainer, model, test_dataset, test_dataloader
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    orig_w, orig_h = image_raw.size
    orig_size = (orig_w, orig_h)

    detections = foundpose_detection

    # camera
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

    # Perform inference on image.
    num_target_insts = 1   # 实例数
    sample = data_util.prepare_sample3(image_raw, scene_cameras)

    # Camera parameters.
    orig_camera_c2w = sample.camera
    orig_image_size = (
        orig_camera_c2w.width,
        orig_camera_c2w.height,
    )

    # Get info about object instances for which we want to estimate pose.
    instances = infer_pose_util.get_instances_for_pose_estimation2(
        obj_id=object_lid,
        use_detections=opts.use_detections,
        detections=detections,
        image_size = orig_image_size,
        orig_size = orig_size
    )

    # Generate grid points at which to sample the feature vectors.
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
    all_rts = []
    coarse_pose_list = []
    for inst_j, instance in enumerate(instances):
        times = {}

        # Get the input image.
        orig_image_np_hwc = sample.image.astype(np.float32)/255.0

        # Get the modal mask and amodal bounding box of the instance.
        orig_box_amodal = AlignedBox2f(
            left=instance["input_box_amodal"][0],
            top=instance["input_box_amodal"][1],
            right=instance["input_box_amodal"][2],
            bottom=instance["input_box_amodal"][3],
        )

        # Optional cropping.
        if not opts.crop:
            camera_c2w = orig_camera_c2w            
            image_np_hwc = orig_image_np_hwc        
            box_amodal = orig_box_amodal
        else:
            # Get box for cropping.
            crop_box = misc_util.calc_crop_box(
                box=orig_box_amodal,
                make_square=True,
            )

            # Construct a virtual camera focused on the crop.
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

            # Recalculate the object bounding box (it changed if we constructed the virtual camera).
            box_amodal = warp_box(
                src_camera=orig_camera_c2w,
                dst_camera=crop_camera_model_c2w,
                box=orig_box_amodal
            )

            # The virtual camera is becoming the main camera.
            camera_c2w = crop_camera_model_c2w

        # Extract feature map from the crop.
        image_tensor_chw = array_to_tensor(image_np_hwc).to(torch.float32).permute(2,0,1).to(device)
        image_tensor_bchw = image_tensor_chw.unsqueeze(0)
        extractor_output = extractor(image_tensor_bchw)
        feature_map_chw = extractor_output["feature_maps"][0]

        # Keep only points inside the object bbox
        query_points, _ = feature_util.filter_points_by_box(
            points=grid_points,
            box=(box_amodal.left, box_amodal.top, box_amodal.right, box_amodal.bottom),
        )

        # Subsample query points if we have too many.
        if query_points.shape[0] > opts.max_num_queries:
            perm = torch.randperm(query_points.shape[0])
            query_points = query_points[perm[: opts.max_num_queries]]
            msg = (
                "Randomly sumbsampled queries "
                f"({perm.shape[0]} -> {query_points.shape[0]}))"
            )
            logging.log_heading(logger, msg, style=logging.RED_BOLD)

        # Extract features at the selected points, of shape (num_points, feat_dims).
        query_features = feature_util.sample_feature_map_at_points(
            feature_map_chw=feature_map_chw,    # [C, H, W]
            points=query_points,                # [N, 2]
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

        # Establish 2D-3D correspondences.
        corresp = []
        if len(query_points) != 0:
            corresp = corresp_util.establish_correspondences(
                query_points=query_points,
                query_features=query_features_proj,
                object_repre=repre,                                 # 当前物体的 3D 模型及其模板特征库
                template_matching_type=opts.match_template_type,    # 模板匹配类型
                template_knn_indices=template_knn_indices,          # 预先筛选出的候选模板
                feat_matching_type=opts.match_feat_matching_type,   # 匹配方式
                top_n_templates=opts.match_top_n_templates,         # 与图像匹配的前 N 个模板
                top_k_buddies=opts.match_top_k_buddies,             # 每个模板中匹配 top-k 的点
                visual_words_knn_index=visual_words_knn_index,      # 向量量化（视觉词）加速结构
                debug=opts.debug,
            )

        # Estimate coarse poses from corespondences.
        coarse_poses = []
        for corresp_id, corresp_curr in enumerate(corresp):

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
            ) = pnp_util.estimate_pose(
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

        # Select the final pose estimate.
        final_poses = []
        
        if opts.final_pose_type in [
            "best_coarse",
        ]:
        
            # If no successful coarse pose, continue.
            if len(coarse_poses) == 0:
                continue
            
            # Select the refined pose corresponding to the best coarse pose as the final pose.
            final_pose = None

            if opts.final_pose_type in [
                "best_coarse",
            ]:
                final_pose = coarse_poses[best_coarse_pose_id]

            if final_pose is not None:
                final_poses.append(final_pose)

        else:
            raise ValueError(f"Unknown final pose type {opts.final_pose_type}")

        # Iterate over the final poses to collect visuals.
        for hypothesis_id, final_pose in enumerate(final_poses):

            # Increment hypothesis id by one for each found pose hypothesis.
            pose_m2w = None

            # Express the estimated pose as an m2w transformation.
            pose_est_m2c = structs.ObjectPose(
                R=final_pose["R_m2c"], t=final_pose["t_m2c"]
            )
            trans_c2w = camera_c2w.T_world_from_eye

            trans_m2w = trans_c2w.dot(misc.get_rigid_matrix(pose_est_m2c))
            # 物体在世界坐标系下的位姿
            pose_m2w = structs.ObjectPose(                  
                R=trans_m2w[:3, :3], t=trans_m2w[:3, 3:]
            )  
            coarse_pose_list.append(pose_m2w)

            # return 客户端需要的信息
            rt_matrix = misc.get_rigid_matrix(pose_m2w)[:3, :4]   
            if not isinstance(rt_matrix, np.ndarray):
                rt_matrix = np.array(rt_matrix)

            # all_rts.append(rt_matrix) # 将每个物体的 Rt 矩阵添加到列表中  

    outputs = run_inference(
        camK=camK,
        image_raw=image_raw,
        coarse_poses=coarse_pose_list,
        trainer=trainer,
        model=model,
        test_dataset=test_dataset,
        test_dataloader=test_dataloader
    )

    for pose_m2w in outputs["objects"].poses_world_from_model:
        rt_matrix2 = pose_m2w.cpu().numpy()[:3, :4]   # [3x4] 相机外参
        all_rts.append(rt_matrix2)

    return {"Rts": all_rts}
