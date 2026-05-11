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
            
            # 2d->3d
            # colmap_dir = os.path.join(bop_config.output_path, 'colmap')  # <-- adjust if necessary
            colmap_dir = "/home/zjr/Tracker/foundpose/datasets/wogua/sparse/0" # <-- adjust if necessary
            images_txt = os.path.join(colmap_dir, 'images.txt')
            points3d_txt = os.path.join(colmap_dir, 'points3D.txt')
            images_colmap = read_colmap_images_txt(images_txt)
            points3d_colmap = read_colmap_points3D_txt(points3d_txt)
            img_entry = images_colmap[frame_name]
            qvec = img_entry['qvec']
            tvec = img_entry['tvec']
            R_colmap = quat_to_rotmat(qvec)            # world -> camera rotation
            t_colmap = tvec.astype(np.float64)         # world -> camera translation (so X_cam = R * X_world + t)
            C_colmap = -R_colmap.T @ t_colmap          # camera center in world coords
            
            # Build arrays of points3D
            # project points3D into the *current* camera (camera_c2w) to find nearest pixels.
            pts_ids = []
            pts_xyz = []
            for pid, rec in points3d_colmap.items():
                pts_ids.append(pid)
                pts_xyz.append(rec['xyz'])
            pts_xyz = np.vstack(pts_xyz)  # (M,3)
            
            # Get current camera intrinsics for projection (use camera_c2w if available, else orig_camera_c2w)
            cam_for_proj = camera_c2w # if 'camera_c2w' in locals() else orig_camera_c2w
            fx_p, fy_p = cam_for_proj.f
            cx_p, cy_p = cam_for_proj.c
            K_proj = np.array([[fx_p, 0, cx_p],[0, fy_p, cy_p],[0,0,1]], dtype=np.float64)

            # Need current camera R,t (world -> cam). If you have colmap pose for this frame, you can use it.
            # Try to use colmap R_colmap,t_colmap (world->cam). If colmap not matching the current camera coordinate
            # system, results will be inconsistent.
            use_colmap_pose = True
            if use_colmap_pose:
                R_w2c = R_colmap
                t_w2c = t_colmap
            
            # Project each COLMAP 3D point into the current camera image 投影稀疏点云到图像平面
            # X_cam = R_w2c @ X_world + t_w2c
            X_cam = (R_w2c @ pts_xyz.T).T + t_w2c.reshape(1,3)
            valid_mask = X_cam[:,2] > 1e-6
            proj_uv = np.zeros((pts_xyz.shape[0], 2), dtype=np.float64)
            proj_uv[valid_mask,0] = (K_proj[0,0] * X_cam[valid_mask,0] / X_cam[valid_mask,2]) + K_proj[0,2]
            proj_uv[valid_mask,1] = (K_proj[1,1] * X_cam[valid_mask,1] / X_cam[valid_mask,2]) + K_proj[1,2]
            
            # invalid ones left as zeros; we can mark them far away
            proj_uv[~valid_mask] = np.array([-1e6, -1e6], dtype=np.float64)
            from scipy.spatial import cKDTree
            # Build KDTree on projected 2D positions 建立 KDTree，找到每个 2D 查询点最近的投影点
            kdt = cKDTree(proj_uv)
            # Convert query_points (torch.Tensor) -> numpy 2D pixel coords
            # query_points assumed to be (N,2) with (x,y) pixel coordinates relative to current camera image (same origin)
            query_pts_np = query_points.cpu().numpy().astype(np.float64) 

            # For each query 2D point, find nearest projected COLMAP point within threshold
            max_px_dist = 6.0  # px threshold to accept a match (tuneable)
            matched_idxs = []
            matched_pts3d = []
            matched_query_idxs = []

            # Query KDTree # query_points[qi] ↔ COLMAP 3D 点 (X_world)
            dists, idxs = kdt.query(query_pts_np, k=1, n_jobs=-1)
            for qi, (d, idx) in enumerate(zip(dists, idxs)):
                if d <= max_px_dist and valid_mask[idx]:
                    matched_query_idxs.append(qi)
                    matched_idxs.append(idx)
                    matched_pts3d.append(pts_xyz[idx])

            matched_query_idxs = np.array(matched_query_idxs, dtype=int)
            if len(matched_pts3d) == 0:
                logger.warn("No COLMAP 3D points matched to query_points (2D->3D). Cannot estimate scale.")
                matched_pts3d = None
            else:
                matched_pts3d = np.vstack(matched_pts3d)  # (M_match, 3)
                
            matched_query_idxs = np.array(matched_query_idxs, dtype=int)
            if len(matched_pts3d) == 0:
                logger.warn("No COLMAP 3D points matched to query_points (2D->3D). Cannot estimate scale.")
                matched_pts3d = None
            else:
                matched_pts3d = np.vstack(matched_pts3d)  # (M_match, 3)
                
            # Now we have for some subset of query points: their matched 3D world points (X).模型点 X′ 和 COLMAP 世界点 X 对齐
            # We also need the corresponding model template 3D points X' for the *same* query indices.
            # To do this, examine `corresp` returned above: each template entry has "coord_2d" and "coord_3d".
            # We'll try to match query indices: corresp item "coord_2d_ids" are indices into query_points.
            model_src_pts = []   # X' (model coordinates)
            world_tgt_pts = []   # X  (colmap world coordinates)
            for tpl in corresp:
                if len(tpl.get("coord_2d_ids", [])) == 0:
                    continue
                tpl_q_ids = tpl["coord_2d_ids"].cpu().numpy().astype(int)  # indices into query_points
                tpl_model_pts = tpl["coord_3d"].cpu().numpy().astype(np.float64)  # model points X' for those indices
                # For each query id that also has a matched COLMAP 3D point, add pair
                for local_idx, qid in enumerate(tpl_q_ids):
                    # find qid in matched_query_idxs
                    found = np.where(matched_query_idxs == qid)[0]
                    if found.size > 0:
                        wi = found[0]
                        world_tgt_pts.append(matched_pts3d[wi])
                        model_src_pts.append(tpl_model_pts[local_idx])
            
            Xp = np.vstack(model_src_pts)   # (N,3) model points (X')
            Xw = np.vstack(world_tgt_pts)   # (N,3) world points (X)
            # # Run Umeyama alignment: Xw ≈ s * R * Xp + t
            # s_est, R_est, t_est = umeyama_align(Xp, Xw, with_scale=True)
            # logger.info(f"Estimated model->world scale s={s_est:.6f}, R_det={np.linalg.det(R_est):.6f}, #pairs={Xp.shape[0]}")
            # # Save results
            # align_out = {
            #     'scale': float(s_est),
            #     'R': R_est.astype(np.float32).tolist(),
            #     't': t_est.astype(np.float32).tolist(),
            #     'num_pairs': int(Xp.shape[0]),
            # }
            # # Write to disk for inspection
            # np.savez(os.path.join(output_dir, f"{frame_name}_model_to_world_align.npz"),
            #         scale=s_est, R=R_est, t=t_est, Xp=Xp, Xw=Xw)
            # # Optionally: if you want to scale your mesh later, you can multiply mesh vertices by s_est,
            # # and apply R_est, t_est to move mesh into COLMAP world coordinates (or into current world).
            return Xp, Xw



def main() -> None:
    
    opts_path = "/home/zjr/Tracker/foundpose/configs/infer/plane.json"
    rgb_dir = "/home/zjr/Tracker/foundpose/datasets/plane/rgb"
    mask_dir = "/home/zjr/Tracker/foundpose/datasets/plane/masks"
    
    # colmap的points3D.txt是三维物体的世界坐标，image.txt是每帧图像的相机位姿，也就是把世界到模型的位姿w2c

    # 按照文件名顺序读取所有帧
    rgb_files = sorted([f for f in os.listdir(rgb_dir) if f.endswith(".jpg")])
    mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith(".png")])
    
    all_Xp, all_Xw = [], []

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
        Xp, Xw = infer(
                opt_file_path=opts_path,
                image_raw=image,
                mask_raw=mask,
                camK=camK,
                frame_name=frame_name,
            )
        Xp_all = np.vstack(all_Xp)
        Xw_all = np.vstack(all_Xw)
        logger.info(f"Collected {Xp_all.shape[0]} 3D-3D pairs from all frames")
        
        # ---- RANSAC + Umeyama ----
        s_best, R_best, t_best, inliers_best = None, None, None, []
        n_iter, threshold = 2000, 0.02  # 0.02m 容差，可调
        rng = np.random.default_rng()
        
        for _ in range(n_iter):
            ids = rng.choice(Xp_all.shape[0], size=4, replace=False)
            try:
                s_est, R_est, t_est = umeyama_align(Xp_all[ids], Xw_all[ids], with_scale=True)
            except Exception:
                continue
            # 应用变换
            Xp_trans = s_est * (R_est @ Xp_all.T).T + t_est
            errors = np.linalg.norm(Xp_trans - Xw_all, axis=1)
            inliers = np.where(errors < threshold)[0]
            if len(inliers) > len(inliers_best):
                s_best, R_best, t_best = s_est, R_est, t_est
                inliers_best = inliers
        
        if s_best is None:
            logger.error("RANSAC 失败，可能匹配点太少")
            return

        logger.info(f"RANSAC 找到 {len(inliers_best)} 个内点 / {Xp_all.shape[0]} 点对")
        logger.info(f"Estimated scale = {s_best:.6f}")

        # 最终 refine
        s_final, R_final, t_final = umeyama_align(Xp_all[inliers_best], Xw_all[inliers_best], with_scale=True)

        np.savez("global_model_to_world_align.npz",
                scale=s_final, R=R_final, t=t_final,
                Xp=Xp_all[inliers_best], Xw=Xw_all[inliers_best])
        logger.info("保存全局尺度估计到 global_model_to_world_align.npz")

       


if __name__ == "__main__":
    main()