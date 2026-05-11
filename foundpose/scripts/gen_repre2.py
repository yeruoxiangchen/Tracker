#!/usr/bin/env python3

"""Generates a feature-based object representation."""

import os
import logging

import sys
REPO_PATH = "/home/zjr/Tracker/foundpose"
BOP_TOOLKIT_PATH = "/home/zjr/Tracker/foundpose/external/bop_toolkit"

sys.path.insert(0, REPO_PATH)
sys.path.insert(0, BOP_TOOLKIT_PATH)  # 添加这一行

from typing import Any, Dict, List, NamedTuple, Optional

import torch

from utils.misc import array_to_tensor

from bop_toolkit_lib import inout, dataset_params

import bop_toolkit_lib.config as bop_config

from utils import (
    feature_util2 as feature_util,
    projector_util,
    repre_util2 as repre_util,
    config_util,
    json_util,
    logging,
    misc,
)

from utils.structs import PinholePlaneCameraModel

import numpy as np

import omniglue
from omniglue import superpoint_extract, dino_extract
import tensorflow as tf
import torch
import numpy as np

# --- (定义 OmniGlue 模型路径) ---
MODEL_EXPORT_PATH = "/home/zjr/Tracker/omniglue"
OG_EXPORT_PATH = MODEL_EXPORT_PATH + "/models/og_export"
SP_EXPORT_PATH = MODEL_EXPORT_PATH + "/models/sp_v6"
DINO_EXPORT_PATH = MODEL_EXPORT_PATH + "/models/dinov2_vitb14_pretrain.pth"
DINO_FEATURE_DIM = 768 # (或你的 DINOv2 模型的维度)

# --- (新的辅助函数：2D + 深度 -> 3D) ---
def unproject_pts_to_3d(pts_2d_np: np.ndarray, 
                        depth_hw: torch.Tensor, 
                        camera: PinholePlaneCameraModel, 
                        T_model_from_camera: torch.Tensor,
                        device: str) -> torch.Tensor:
    """
    将模板图像上的 2D 点 (u,v) 及其深度反投影到 3D 模型坐标。
    """
    if pts_2d_np.shape[0] == 0:
        return torch.empty((0, 3), device=device, dtype=torch.float32)
    depth_hw_cpu = depth_hw.cpu()
    # 确保 pts_2d 是整数以便索引
    pts_2d_int = np.round(pts_2d_np).astype(int)
    
    # 过滤掉超出边界的点
    h, w = depth_hw.shape
    valid_mask = (pts_2d_int[:, 0] >= 0) & (pts_2d_int[:, 0] < w) & \
                 (pts_2d_int[:, 1] >= 0) & (pts_2d_int[:, 1] < h)
    
    pts_2d_valid = pts_2d_int[valid_mask]
    
    if pts_2d_valid.shape[0] == 0:
        return torch.empty((0, 3), device=device, dtype=torch.float32)

    # 1. 获取深度
    depth_values = depth_hw_cpu[pts_2d_valid[:, 1], pts_2d_valid[:, 0]] # (N,)
    
    # 2. 过滤掉深度无效的点 (深度图通常用 0 表示无效)
    valid_depth_mask = depth_values > 0
    pts_2d_torch_temp = torch.as_tensor(pts_2d_valid, device='cpu')
    pts_2d_valid_torch = pts_2d_torch_temp[valid_depth_mask]
    depth_values = depth_values[valid_depth_mask]
    if pts_2d_valid_torch.shape[0] == 0:
        return torch.empty((0, 3), device=device, dtype=torch.float32)
        
    pts_2d_torch = pts_2d_valid_torch.to(device, dtype=torch.float32)

    # 3. 反投影到相机坐标系 (X_c, Y_c, Z_c)
    # Z_c = d
    Z_c = depth_values.to(device)
    # X_c = (u - cx) * Z_c / fx
    X_c = (pts_2d_torch[:, 0] - camera.c[0]) * Z_c / camera.f[0]
    # Y_c = (v - cy) * Z_c / fy
    Y_c = (pts_2d_torch[:, 1] - camera.c[1]) * Z_c / camera.f[1]
    
    # 组合为 (N, 3) 齐次坐标 (N, 4)
    pts_cam_space = torch.stack([X_c, Y_c, Z_c], dim=1) # (N, 3)
    pts_cam_homogeneous = torch.cat(
        [pts_cam_space, torch.ones((pts_cam_space.shape[0], 1), device=device)], dim=1
    ) # (N, 4)

    # 4. 转换到模型坐标系
    # P_model = T_model_from_camera @ P_cam
    pts_model_homogeneous = (T_model_from_camera @ pts_cam_homogeneous.T).T # (N, 4)
    
    # 返回 (X, Y, Z)
    return pts_model_homogeneous[:, :3]

class GenRepreOpts(NamedTuple):
    """Options that can be specified via the command line."""

    version: str
    templates_version: str
    object_dataset: str
    object_lids: Optional[List[int]] = None
    
    # --- (旧的提取选项被移除) ---
    # # Feature extraction options.
    # extractor_name: str = "dinov2_vits14_reg"
    # grid_cell_size: float = 14.0

    # Feature PCA options.
    apply_pca: bool = True
    pca_components: int = 256
    pca_whiten: bool = False
    pca_max_samples_for_fitting: int = 100000

    # --- (聚类和模板描述符选项被移除) ---
    # # Feature clustering options.
    # cluster_features: bool = True
    # cluster_num: int = 2048

    # # Template descriptor options.
    # template_desc_opts: Optional[repre_util.TemplateDescOpts] = None

    # Other options.
    overwrite: bool = True
    debug: bool = True


def generate_raw_repre(
    opts: GenRepreOpts,
    object_dataset: str,
    object_lid: int,
    # 传递 SP 和 DINO 提取器
    sp_extract: superpoint_extract.SuperPointExtract,
    dino_extract_module: dino_extract.DINOExtract,
    output_dir: str,
    device: str = "cuda",
    debug: bool = False,
) -> repre_util.FeatureBasedObjectRepre:

    logger = logging.get_logger(level=logging.INFO if opts.debug else logging.WARNING)

    # Prepare a timer.
    timer = misc.Timer(enabled=debug)

    datasets_path = bop_config.datasets_path

    # Load the template metadata.
    # metadata_path = "/Users/evinpinar/Documents/opensource_foundpose/output/templates/v1/lmo/1/metadata.json"
    metadata_path = os.path.join(
        bop_config.output_path,
        "templates",
        opts.templates_version,
        opts.object_dataset,
        str(object_lid),
        "metadata.json"
    )
    metadata = json_util.load_json(metadata_path)

    # --- (为 OmniGlue 风格的新数据准备列表) ---
    all_feat_vectors_list = []      # 存储 DINO 描述符
    all_sp_keypoints_list = []      # 存储 SP 关键点 (u,v)
    all_vertices_in_model_list = [] # 存储 3D 坐标 (X,Y,Z)
    all_feat_to_template_ids_list = []
    
    templates_list = []
    template_cameras_cam_from_model_list = []
    
    # # Prepare structures for storing data.
    # feat_vectors_list = []
    # feat_to_vertex_ids_list = []
    # vertices_in_model_list = []
    # feat_to_template_ids_list = []
    # templates_list = []
    # template_cameras_cam_from_model_list = []

    # Use template images specified in the metadata.
    template_id = 0
    num_templates = len(metadata)
    for data_id, data_sample in enumerate(metadata):

        logger.info(
            f"Processing dataset {data_id}/{num_templates}, "
        )

        timer.start()

        camera_sample = data_sample["cameras"]
        camera_world_from_cam = PinholePlaneCameraModel(
                width=camera_sample["ImageSizeX"],
                height=camera_sample["ImageSizeY"],
                f=(camera_sample["fx"],camera_sample["fy"]),
                c=(camera_sample["cx"],camera_sample["cy"]),
                T_world_from_eye=np.array(camera_sample["T_WorldFromCamera"])
            )

        # RGB/monochrome and depth images (in mm).
        image_path = data_sample["rgb_image_path"]
        depth_path = data_sample["depth_map_path"]
        mask_path = data_sample["binary_mask_path"]

        image_arr_np = inout.load_im(image_path) # H,W,C
        depth_image_arr = inout.load_depth(depth_path)
        mask_image_arr = inout.load_im(mask_path)

        image_chw = array_to_tensor(image_arr_np).to(torch.float32).permute(2,0,1).to(device) / 255.0
        depth_image_hw = array_to_tensor(depth_image_arr).to(torch.float32).to(device)
        object_mask_modal = array_to_tensor(mask_image_arr).to(torch.float32).to(device)

        # Get the object annotation.
        assert data_sample["dataset"] == object_dataset
        assert data_sample["lid"] == object_lid
        assert data_sample["template_id"] == data_id

        object_pose = data_sample["pose"]

        # Transformations.
        object_pose_rigid_matrix = np.eye(4)
        object_pose_rigid_matrix[:3, :3] = object_pose["R"]
        object_pose_rigid_matrix[:3, 3:] = object_pose["t"]
        T_world_from_model = (
            array_to_tensor(object_pose_rigid_matrix)
            .to(torch.float32)
            .to(device)
        )
        T_model_from_world = torch.linalg.inv(T_world_from_model)
        T_world_from_camera = (
            array_to_tensor(camera_world_from_cam.T_world_from_eye)
            .to(torch.float32)
            .to(device)
        )
        T_model_from_camera = torch.matmul(T_model_from_world, T_world_from_camera)

        timer.elapsed("Time for getting template data")
        timer.start()

        # # Extract features from the current template.
        # (
        #     feat_vectors,
        #     feat_to_vertex_ids,
        #     vertices_in_model,
        # ) = feature_util.get_visual_features_registered_in_3d(
        #     image_chw=image_chw,
        #     depth_image_hw=depth_image_hw,
        #     object_mask=object_mask_modal,
        #     camera=camera_world_from_cam,
        #     T_model_from_camera=T_model_from_camera,
        #     extractor=extractor,
        #     grid_cell_size=opts.grid_cell_size,
        #     debug=False,
        # )
        # --- (新的 OmniGlue 风格特征提取) ---
        # 1. 运行 SuperPoint
        # 注意: sp_extract 需要 np.ndarray (H, W, C)
        
        sp_features = sp_extract(image_arr_np) 
        sp_keypoints_np = sp_features[0] # (N, 2) [x, y] 坐标
        sp_descriptors = sp_features[1] # (N, D_sp)
        sp_scores = sp_features[2]      # (N,)

        # 2. 过滤掉掩码外的点
        # filter_points_by_mask 需要 [x,y] 格式
        sp_keypoints_torch = torch.tensor(sp_keypoints_np, dtype=torch.float32, device=device)
        valid_indices = feature_util.filter_points_by_mask_indice(
            sp_keypoints_torch, object_mask_modal, return_indices=True
        )
        if valid_indices.shape[0] == 0:
            logger.info("No valid SP keypoints found in mask.")
            template_id += 1
            continue

        sp_keypoints_np_masked = sp_keypoints_np[valid_indices.cpu().numpy()]
        sp_keypoints_torch_masked = sp_keypoints_torch[valid_indices]
        
        # 3. 运行 DINO
        # dino_extract 需要 (H, W, C) np.ndarray
        dino_features_dense = dino_extract_module(image_arr_np)
        
        # 4. 在 SP 关键点位置采样 DINO 描述符
        # get_dino_descriptors 需要 (N, 2) torch.Tensor [x, y]
        dino_descriptors = dino_extract.get_dino_descriptors(
            dino_features_dense,
            tf.convert_to_tensor(sp_keypoints_np_masked, dtype=tf.float32),
            tf.convert_to_tensor(image_arr_np.shape[0], dtype=tf.int32), # height
            tf.convert_to_tensor(image_arr_np.shape[1], dtype=tf.int32), # width
            DINO_FEATURE_DIM,
        )
        dino_descriptors_torch = torch.tensor(dino_descriptors.numpy(), device=device) # (N_valid, D_dino)

        # 5. 将 2D 点反向投影到 3D
        # (我们使用 sp_keypoints_np_masked, 因为 unproject 需要 numpy)
        vertices_in_model = unproject_pts_to_3d(
            sp_keypoints_np_masked,
            depth_image_hw,
            camera_world_from_cam, # 传递原始相机模型
            T_model_from_camera,
            device
        )
        
        # 6. 再次过滤，因为 unproject 可能会因为无效深度而移除一些点
        if vertices_in_model.shape[0] == 0:
             logger.info("No valid 3D points after unprojection.")
             template_id += 1
             continue
        timer.elapsed("Time for feature extraction")
        timer.start()

        # Store data.
        # feat_vectors_list.append(feat_vectors)
        # feat_to_vertex_ids_list.append(feat_to_vertex_ids)
        # vertices_in_model_list.append(vertices_in_model)
        # feat_to_template_ids = template_id * torch.ones(
        #     feat_vectors.shape[0], dtype=torch.int32, device=device
        # )
        # feat_to_template_ids_list.append(feat_to_template_ids)
        
        # --- (存储新数据) ---
        all_feat_vectors_list.append(dino_descriptors_torch)
        all_vertices_in_model_list.append(vertices_in_model)
        
        # (复用 feat_to_vertex_ids 存储 2D keypoints [x,y])
        all_sp_keypoints_list.append(sp_keypoints_torch_masked) 
        
        feat_to_template_ids = template_id * torch.ones(
            dino_descriptors_torch.shape[0], dtype=torch.int32, device=device
        )
        all_feat_to_template_ids_list.append(feat_to_template_ids)

        # Save the template as uint8 to save space.
        image_chw_uint8 = (image_chw * 255).to(torch.uint8)
        templates_list.append(image_chw_uint8)

        # Store camera model of the current template.
        camera_model = camera_world_from_cam.copy()
        camera_model.extrinsics = torch.linalg.inv(T_model_from_camera)
        template_cameras_cam_from_model_list.append(camera_model)

        # Increment the template ID.
        template_id += 1

        timer.elapsed("Time for storing data")

    logger.info("Processing done.")

    # # Build the object representation from the collected data.
    # return repre_util.FeatureBasedObjectRepre(
    #     vertices=torch.cat(vertices_in_model_list),
    #     feat_vectors=torch.cat(feat_vectors_list),
    #     feat_opts=repre_util.FeatureOpts(extractor_name=opts.extractor_name),
    #     feat_to_vertex_ids=torch.cat(feat_to_vertex_ids_list),
    #     feat_to_template_ids=torch.cat(feat_to_template_ids_list),
    #     templates=torch.stack(templates_list),
    #     template_cameras_cam_from_model=template_cameras_cam_from_model_list,
    # )
    # --- (构建新的 repre 对象) ---
    # 我们将 2D SP keypoints 存储在 feat_to_vertex_ids 中 (权宜之计)
    return repre_util.FeatureBasedObjectRepre(
        vertices=torch.cat(all_vertices_in_model_list),
        feat_vectors=torch.cat(all_feat_vectors_list),
        feat_opts=repre_util.FeatureOpts(extractor_name="OmniGlue_SP_DINO"),
        feat_to_vertex_ids=torch.cat(all_sp_keypoints_list), # 复用字段
        feat_to_template_ids=torch.cat(all_feat_to_template_ids_list),
        templates=torch.stack(templates_list),
        template_cameras_cam_from_model=template_cameras_cam_from_model_list,
        template_desc_opts=repre_util.TemplateDescOpts(),
    )


def generate_repre(
    opts: GenRepreOpts,
    dataset: str,
    lid: int,
    device: str = "cuda",
    # extractor: Optional[torch.nn.Module] = None,
    # 传递 SP 和 DINO 提取器
    sp_extract: Optional[superpoint_extract.SuperPointExtract] = None,
    dino_extract_module: Optional[dino_extract.DINOExtract] = None,
) -> None:

    logger = logging.get_logger(level=logging.INFO if opts.debug else logging.WARNING)

    datasets_path = bop_config.datasets_path

    # Prepare a timer.
    timer = misc.Timer(enabled=opts.debug)
    timer.start()

    # Prepare the output folder.
    base_repre_dir = os.path.join(bop_config.output_path, "object_repre")
    output_dir = repre_util.get_object_repre_dir_path(
        base_repre_dir, opts.version, dataset, lid
    )
    if os.path.exists(output_dir) and not opts.overwrite:
        raise ValueError(f"Output directory already exists: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    # Save parameters to a JSON file.
    json_util.save_json(os.path.join(output_dir, "config.json"), opts)

    # # Prepare a feature extractor.
    # if extractor is None:
    #     extractor = feature_util.make_feature_extractor(opts.extractor_name)
    # extractor.to(device)
    
    # --- (准备 OmniGlue 提取器) ---
    if sp_extract is None or dino_extract_module is None:
        # (假设 OmniGlue 及其子模块的加载逻辑)
        sp_extract = superpoint_extract.SuperPointExtract(SP_EXPORT_PATH)
        dino_extract_module = dino_extract.DINOExtract(DINO_EXPORT_PATH, feature_layer=1)

    timer.elapsed("Time for preparation")
    timer.start()

    # Build raw object representation.
    repre = generate_raw_repre(
        opts=opts,
        object_dataset=dataset,
        object_lid=lid,
        sp_extract=sp_extract,           # 传递新提取器
        dino_extract_module=dino_extract_module, # 传递新提取器
        output_dir=output_dir,
        device=device,
    )

    feat_vectors = repre.feat_vectors
    assert feat_vectors is not None

    timer.elapsed("Time for generating raw representation")

    # Optionally transform the feature vectors to a PCA space.
    if opts.apply_pca:
        timer.start()

        # Prepare a PCA projector.
        logger.info("Preparing PCA...")
        pca_projector = projector_util.PCAProjector(
            n_components=opts.pca_components, whiten=opts.pca_whiten
        )
        pca_projector.fit(feat_vectors, max_samples=opts.pca_max_samples_for_fitting)
        repre.feat_raw_projectors.append(pca_projector)

        # Transform the selected feature vectors to the PCA space.
        feat_vectors = pca_projector.transform(feat_vectors)

        timer.elapsed("Time for PCA")

    # # Cluster features into visual words.
    # if opts.cluster_features:
    #     timer.start()

    #     logger.info(f"Clustering features into {opts.cluster_num} visual words...")
    #     centroids, cluster_ids, centroid_distances = cluster_util.kmeans(
    #         samples=feat_vectors,
    #         num_centroids=opts.cluster_num,
    #         verbose=True,
    #     )

    #     # Store the clustering results in the object repre.
    #     repre.feat_cluster_centroids = centroids
    #     repre.feat_to_cluster_ids = cluster_ids

    #     # Get cluster sizes.
    #     unique_ids, unique_counts = torch.unique(cluster_ids, return_counts=True)

    #     timer.elapsed("Time for feature clustering")
    #     logging.log_heading(
    #         logger,
    #         f"{feat_vectors.shape[0]} feature vectors were clustered into {len(centroids)} clusters "
    #         f"with {unique_counts.min()} to {unique_counts.max()} elements.",
    #     )

    # # Generate template descriptors.
    # if opts.template_desc_opts is not None:
    #     timer.start()

    #     repre.template_desc_opts = opts.template_desc_opts

    #     # Calculate tf-idf descriptors.
    #     if opts.template_desc_opts.desc_type == "tfidf":

    #         assert feat_vectors is not None
    #         assert repre.feat_cluster_centroids is not None
    #         assert repre.feat_to_cluster_ids is not None
    #         assert repre.feat_to_template_ids is not None
    #         assert repre.templates is not None

    #         repre.template_descs, repre.feat_cluster_idfs = (
    #             template_util.calc_tfidf_descriptors(
    #                 feat_vectors=feat_vectors,
    #                 feat_words=repre.feat_cluster_centroids,
    #                 feat_to_word_ids=repre.feat_to_cluster_ids,
    #                 feat_to_template_ids=repre.feat_to_template_ids,
    #                 num_templates=len(repre.templates),
    #                 tfidf_knn_k=opts.template_desc_opts.tfidf_knn_k,
    #                 tfidf_soft_assign=opts.template_desc_opts.tfidf_soft_assign,
    #                 tfidf_soft_sigma_squared=opts.template_desc_opts.tfidf_soft_sigma_squared,
    #             )
    #         )

    #     else:
    #         raise ValueError(
    #             f"Unknown template descriptor type: {opts.template_desc_opts.desc_type}"
    #         )

    #     timer.elapsed("Time for generating template descriptors")

    # timer.start()

    # Create a PCA projector for visualization purposes (or reuse an existing one).
    if len(repre.feat_raw_projectors) and isinstance(
        repre.feat_raw_projectors[0], projector_util.PCAProjector
    ):
        repre.feat_vis_projectors = [repre.feat_raw_projectors[0]]
    else:
        # Prepare a PCA projector.
        num_pca_dims_vis = 3
        pca_projector_vis = projector_util.PCAProjector(
            n_components=num_pca_dims_vis, whiten=False
        )
        pca_projector_vis.fit(
            feat_vectors, max_samples=opts.pca_max_samples_for_fitting
        )
        repre.feat_vis_projectors = [pca_projector_vis]

    repre.feat_vectors = feat_vectors

    timer.elapsed("Time for finding PCA for visualizations")
    timer.start()

    # Save the generated object representation.
    repre_dir = repre_util.get_object_repre_dir_path(
        base_repre_dir, opts.version, dataset, lid
    )
    repre_util.save_object_repre(repre, repre_dir)

    timer.elapsed("Time for saving the object representation")


def generate_repre_from_list(opts: GenRepreOpts) -> None:

    # Get IDs of objects to process.
    object_lids = opts.object_lids
    if object_lids is None:
        datasets_path = bop_config.datasets_path
        bop_model_props = dataset_params.get_model_params(datasets_path=datasets_path, dataset_name=opts.object_dataset)
        object_lids = bop_model_props["obj_ids"]

    # # Prepare a feature extractor.
    # extractor = feature_util.make_feature_extractor(opts.extractor_name)
    # --- (准备 OmniGlue 提取器) ---
    sp_extract = superpoint_extract.SuperPointExtract(SP_EXPORT_PATH)
    dino_extract_module = dino_extract.DINOExtract(DINO_EXPORT_PATH, feature_layer=1)

    # Prepare a device.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device: ", device)

    # Process each image separately.
    for object_lid in object_lids:
        # generate_repre(opts, opts.object_dataset, object_lid, device, extractor)
        generate_repre(
            opts, 
            opts.object_dataset, 
            object_lid, 
            device, 
            sp_extract=sp_extract, # 传递实例
            dino_extract_module=dino_extract_module # 传递实例
        )


def main() -> None:
    generate_repre_from_list(
        config_util.load_opts_from_json_or_command_line(GenRepreOpts)[0]
    )


if __name__ == "__main__":
    main()
