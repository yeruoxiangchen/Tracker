#!/usr/bin/env python3

"""Infers pose from objects."""


import sys
REPO_PATH = "/home/zjr/Tracker/CoarseModel"

sys.path.insert(0, REPO_PATH)

import os
import cv2
import numpy as np
from utils import logging, config_util
from tqdm import tqdm
import pickle

from core.config import AppConfig, InferOpts
# from core.preprocess import(
#     build_optimization_data, 
#     save_optimization_cache, 
#     load_optimization_cache
# )
# from core.preprocess_lunkuo import(
#     build_optimization_data, 
#     save_optimization_cache, 
#     load_optimization_cache
# )
from core.preprocess_lunkuo_quanjuchouzhen import(
    build_optimization_data, 
    save_optimization_cache, 
    load_optimization_cache
)

# from core.global_optimize import optimize_global_pose
# from core.global_optimize2 import optimize_global_pose
# from core.global_optimize_deformation import optimize_global_pose, refine_model_with_deformation_graph
# from core.global_optimize_deformation2contour import optimize_global_pose, refine_model_with_deformation_graph
# from core.global_optimize_deformationcontour_xiaoliehen import optimize_global_pose, refine_model_with_deformation_graph
# from core.global_optimize_deformationcontour_xiaoliehen_new_init2 import optimize_global_pose, refine_model_with_deformation_graph
from core.global_optimize2 import optimize_global_pose
from core.util import load_obj_with_visual
from script.vis_util import (
    visualize_projection,
    visualize_projection2,
    visualize_projection_multi_pose,
)
from utils.logging import get_logger

logger: logging.Logger = logging.get_logger()

import argparse

# --- 主函数 ---

def main() -> None:
    
    # 路径配置
    # base_dir = "/home/zjr/Tracker/CoarseModel/datasets/wogua"
    # opts_path = "/home/zjr/Tracker/CoarseModel/configs/infer/wogua.json"
    parser = argparse.ArgumentParser(description="Global Pose Optimization")
    parser.add_argument(
        "--dataset_name", 
        type=str, 
        default="test", 
        help="Name of the dataset folder under AppConfig.DATASETS_PATH"
    )
    args = parser.parse_args()
    dataset_name = args.dataset_name
    opts_path = os.path.join(AppConfig.PROJECT_ROOT, "configs/infer", dataset_name, f"{dataset_name}.json")
    opts = config_util.load_opts_from_json(
        path=opts_path, 
        opts_types={"infer_opts": InferOpts}
    )["infer_opts"]
    
    base_dir = AppConfig.DATASETS_PATH / dataset_name
    rgb_dir = os.path.join(base_dir, "rgb")
    mask_dir = os.path.join(base_dir, "masks")
    colmap_dir = os.path.join(base_dir, "sparse/0")
    data_cache_path = os.path.join(os.path.dirname(opts_path), dataset_name, "cached_optimization_data.pkl")
    output_dir = os.path.join(
        AppConfig.OUTPUT_ROOT, "refine_model", dataset_name
    )
    os.makedirs(output_dir, exist_ok=True)

     # -------------------------------------------------------------------------
    # Load object model
    # -------------------------------------------------------------------------
    logger.info("Loading object model...")
    dataset_model_dir = os.path.join(AppConfig.DATASETS_PATH, opts.object_dataset, "models")
    model_path = os.path.join(dataset_model_dir, f"{opts.object_dataset}_norm.obj")
    model_vertices, model_faces, original_visual = load_obj_with_visual(model_path)
    if model_vertices.size == 0:
        logger.error("Model vertices are empty. Abort.")
        return
    
    # -------------------------------------------------------------------------
    # Stage 1: Per-frame correspondences (with cache)
    # -------------------------------------------------------------------------
    logger.info("Preparing per-frame optimization data...")
    try:
        all_optimization_data, all_corresps_data = load_optimization_cache(data_cache_path)
        logger.info(f"Loaded cached data: {len(all_optimization_data)} frames.")
    except (FileNotFoundError, EOFError, pickle.UnpicklingError):
        logger.info("Cache not found or invalid. Rebuilding...")
        all_optimization_data, all_corresps_data = build_optimization_data(
            rgb_dir=rgb_dir,
            mask_dir=mask_dir,
            colmap_dir=colmap_dir,
            opts=opts,)
        if len(all_optimization_data) == 0:
            logger.error("No valid frames collected. Abort.")
            return
        save_optimization_cache(
            {
                "optimization_data": all_optimization_data,
                "corresps_data": all_corresps_data,
            },
            data_cache_path,
        )
        logger.info(f"Saved optimization cache: {len(all_optimization_data)} frames.")

    # -------------------------------------------------------------------------
    # Stage 2: Global pose + scale optimization (INIT INCLUDED)
    # -------------------------------------------------------------------------
    logger.info("Starting global pose & scale optimization...")
    valid_opt_data = []
    valid_corresps = []

    for opt_data, corresp in zip(all_optimization_data, all_corresps_data):
        if corresp is None:
            continue
        if opt_data["q_2d"] is None or opt_data["X_3d"] is None:
            continue
        valid_opt_data.append(opt_data)
        valid_corresps.append(corresp)

    if len(valid_opt_data) < 2:
        logger.error("Not enough keyframes with correspondences for optimization.")
        return

    T_M2W_final, result = optimize_global_pose(
        all_optimization_data=valid_opt_data,
        all_corresps_data=valid_corresps,
        model_vertices=model_vertices,
        output_dir=output_dir,
        colmap_dir=colmap_dir,
        mask_dir=mask_dir,
        init_frame_id=0,   # 注意：现在是 keyframe list 内的 index
    )

    logger.info("Optimization finished.")
    logger.info(f"Final cost: {result.cost:.6f}")
    # logger.info(f"new scale: {result.x[0]:.6f}")
    logger.info(f"Optimized T_M2W:\n{T_M2W_final}")
    
    # # # -------------------------------------------------------------------------
    # # # Stage 3: Visualization
    # # # -------------------------------------------------------------------------
    logger.info("Visualizing final results...")
    for data in tqdm(all_optimization_data, desc="Final visualization"):
        visualize_projection(
            image_path=data["image_path"],
            output_dir=output_dir,
            K=data["K"],
            T_W2C=data["T_w2c"],
            T_M2W=T_M2W_final,
            model_vertices=model_vertices,
            model_faces=model_faces,
        )

    # # 将模型转换到世界坐标系 (Mb)
    # V_world_h = np.hstack([model_vertices, np.ones((len(model_vertices), 1))])
    # V_world = (T_M2W_final @ V_world_h.T).T[:, :3]
    
    # refined_vertices = refine_model_with_deformation_graph(
    #     model_vertices=V_world,
    #     model_faces=model_faces,
    #     T_M2W=np.eye(4),
    #     all_optimization_data=all_optimization_data[:5],
    #     output_dir=output_dir,
    # )
    # # model_vertices = refined_vertices
    # logger.info("Visualizing final results...")
    # import trimesh
    # # refined_vertices = trimesh.Trimesh(vertices=refined_vertices, faces=model_faces)
    # logger.info("Exporting refined model with texture...")
    # # 导出文件
    # refined_mesh = trimesh.Trimesh(
    #     vertices=refined_vertices, 
    #     faces=model_faces,
    #     visual=original_visual.copy(), # 复制 UV 坐标和材质
    #     process=False                  # 必须为 False，防止顶点重排
    # )

    # # 建议导出为 .obj，因为 .ply 对复杂纹理贴图的支持并不标准
    # output_subdir = os.path.join(output_dir, "refine")
    # os.makedirs(output_subdir, exist_ok=True)
    # output_model_path = os.path.join(output_subdir, "refined_model.obj")
    # refined_mesh.export(output_model_path)
    # logger.info(f"Refined model saved to: {output_model_path}")
    # for data in tqdm(all_optimization_data, desc="Final visualization"):
    #     visualize_projection(
    #         image_path=data["image_path"],
    #         output_dir=output_subdir,
    #         K=data["K"],
    #         T_W2C=data["T_w2c"],
    #         T_M2W=np.eye(4),
    #         model_vertices=refined_vertices,
    #         model_faces=model_faces,
    #     )

    # logger.info("========== All Done ==========")


if __name__ == "__main__":
    main()