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
from core.preprocess_sim import(
    build_optimization_data, 
    save_optimization_cache, 
    load_optimization_cache
)
# from core.global_optimize import optimize_global_pose
from core.global_optimize_4stage_defor import optimize_global_pose
from core.util import load_ply_vertices_faces
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
    # base_dir = "/home/zjr/Tracker/CoarseModel/datasets/heimei"
    # opts_path = "/home/zjr/Tracker/CoarseModel/configs/infer/heimei/heimei.json"
    # rgb_dir = os.path.join(base_dir, "rgb")
    # mask_dir = os.path.join(base_dir, "masks")
    # colmap_dir = os.path.join(base_dir, "sparse/0")
    # data_cache_path = os.path.join(os.path.dirname(opts_path), "cached_optimization_data.pkl")
    # output_dir = os.path.join(
    #     AppConfig.OUTPUT_ROOT, "refine_model", "heimei"
    # )
    # os.makedirs(output_dir, exist_ok=True)

    opts = config_util.load_opts_from_json(
        path=opts_path, 
        opts_types={"infer_opts": InferOpts}
    )["infer_opts"]
    
     # -------------------------------------------------------------------------
    # Load object model
    # -------------------------------------------------------------------------
    logger.info("Loading object model...")
    dataset_model_dir = os.path.join(AppConfig.DATASETS_PATH, opts.object_dataset, "models")
    model_path = os.path.join(dataset_model_dir, f"{opts.object_dataset}_norm.obj")
    model_vertices, model_faces = load_ply_vertices_faces(
        model_path
    )
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

    T_M2W_final, result, final_scale = optimize_global_pose(
        all_optimization_data=all_optimization_data,
        all_corresps_data=all_corresps_data,
        model_vertices=model_vertices,
        model_faces=model_faces,
        output_dir=output_dir,
        colmap_dir=colmap_dir,
        mask_dir = mask_dir,
        init_frame_id=1,
    )

    logger.info("Optimization finished.")
    logger.info(f"Final cost: {result.cost:.6f}")
    logger.info(f"final_scale: {final_scale}")
    logger.info(f"Optimized T_M2W:\n{T_M2W_final}")
    
    # -------------------------------------------------------------------------
    # Stage 3: Visualization
    # -------------------------------------------------------------------------
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
            image_name=f"projected_{os.path.basename(data['image_path'])}"
        )

    # # Optional: compare local PnP vs global pose
    # for i, data in enumerate(tqdm(all_optimization_data, desc="compare visualization")):
    #     visualize_projection_multi_pose(
    #         image_path=data["image_path"],
    #         output_dir=output_dir,
    #         K=data["K"],
    #         T_W2C=data["T_w2c"],
    #         T_M2W_final=T_M2W_final,
    #         best_pose=all_corresps_data[i].best_pose,
    #         model_vertices=model_vertices,
    #         model_faces=model_faces,
    #     )

    logger.info("========== All Done ==========")


if __name__ == "__main__":
    main()