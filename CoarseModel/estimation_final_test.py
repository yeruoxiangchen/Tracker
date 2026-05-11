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
from core.preprocess_sim_fin_test import(
    build_optimization_data, 
    build_optimization_data2, 
    save_optimization_cache, 
    load_optimization_cache
)
# from core.global_optimize import optimize_global_pose
from core.global_optimize_2stage_defor_newinit_test import optimize_global_pose, refine_model_with_deformation_graph
from core.util import load_ply_vertices_faces
from script.vis_util import (
    visualize_projection,
    visualize_projection2,
    visualize_projection_multi_pose,
)
from utils.logging import get_logger

from core.util import load_obj_with_visual

logger: logging.Logger = logging.get_logger()

import argparse
import time  # 确保导入了 time 模块
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
    model_vertices, model_faces, original_visual = load_obj_with_visual(model_path)
    if model_vertices.size == 0:
        logger.error("Model vertices are empty. Abort.")
        return
    
    start_total = time.time() # 记录程序开始总时间
    # -------------------------------------------------------------------------
    # Stage 1: Per-frame correspondences (with cache)
    # -------------------------------------------------------------------------
    logger.info("Preparing per-frame optimization data...")
    t1_start = time.time()
    logger.info("Cache not found or invalid. Rebuilding...")
    all_optimization_data, all_corresps_data = build_optimization_data(
        rgb_dir=rgb_dir,
        mask_dir=mask_dir,
        colmap_dir=colmap_dir,
        opts=opts,
        start_idx=0,
        end_idx=50,
        stride=5
        )
    
    if len(all_optimization_data) == 0:
        logger.error("No valid frames collected. Abort.")
        return
    logger.info(f"Saved optimization cache: {len(all_optimization_data)} frames.")

    t1_end = time.time()
    stage_1_time = t1_end - t1_start
    # -------------------------------------------------------------------------
    # Stage 2: Global pose + scale optimization (INIT INCLUDED)
    # -------------------------------------------------------------------------
    logger.info("Starting global pose & scale optimization...")
    t2_start = time.time()
    T_M2W_final, result, final_scale = optimize_global_pose(
        all_optimization_data=all_optimization_data,
        all_corresps_data=all_corresps_data,
        model_vertices=model_vertices,
        model_faces=model_faces,
        output_dir=output_dir,
        colmap_dir=colmap_dir,
        mask_dir = mask_dir,
        init_frame_id=0,
    )

    logger.info("Optimization finished.")
    logger.info(f"Final cost: {result.cost:.6f}")
    logger.info(f"final_scale: {final_scale}")
    logger.info(f"Optimized T_M2W:\n{T_M2W_final}")
    
    t2_end = time.time()
    stage_2_time = t2_end - t2_start
    logger.info(f"Stage 2 (Global Pose) finished in {stage_2_time:.2f}s")
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
            image_name=f"projected_{os.path.basename(data['image_path'])}",
            root_name="optvis1"
        )

    logger.info("Starting model deformation refinement...")
    t3_start = time.time()
    # 将模型转换到世界坐标系 (Mb)
    V_world_h = np.hstack([model_vertices, np.ones((len(model_vertices), 1))])
    V_world = (T_M2W_final @ V_world_h.T).T[:, :3]
    
    refined_vertices = refine_model_with_deformation_graph(
        model_vertices=V_world,
        model_faces=model_faces,
        output_dir=output_dir,
        T_M2W=np.eye(4),
        all_optimization_data=all_optimization_data,
    )
    t3_end = time.time()
    stage_3_time = t3_end - t3_start
    logger.info(f"Stage 3 (Deformation) finished in {stage_3_time:.2f}s")
    t4_start = time.time()
    # model_vertices = refined_vertices
    logger.info("Visualizing final results...")
    import trimesh
    # refined_vertices = trimesh.Trimesh(vertices=refined_vertices, faces=model_faces)
    logger.info("Exporting refined model with texture...")
    # 导出文件
    refined_mesh = trimesh.Trimesh(
        vertices=refined_vertices, 
        faces=model_faces,
        visual=original_visual.copy(), # 复制 UV 坐标和材质
        process=False                  # 必须为 False，防止顶点重排
    )

    # 建议导出为 .obj，因为 .ply 对复杂纹理贴图的支持并不标准
    output_subdir = os.path.join(output_dir, "refine1")
    os.makedirs(output_subdir, exist_ok=True)
    output_model_path = os.path.join(output_subdir, "refined_model.obj")
    refined_mesh.export(output_model_path)
    logger.info(f"Refined model saved to: {output_model_path}")
    for data in tqdm(all_optimization_data, desc="Final visualization"):
        visualize_projection(
            image_path=data["image_path"],
            output_dir=output_dir,
            K=data["K"],
            T_W2C=data["T_w2c"],
            T_M2W=np.eye(4),
            model_vertices=refined_vertices,
            model_faces=model_faces,
            image_name=f"projected_{os.path.basename(data['image_path'])}",
            root_name="meshvis1"
        )

    t4_end = time.time()
    stage_4_time = t4_end - t4_start
    total_time = t4_end - start_total
    
    # -----------------------二阶堂—------------------------------------------------###############################
    start_total = time.time() # 记录程序开始总时间
    # -------------------------------------------------------------------------
    # Stage 1: Per-frame correspondences (with cache)
    # -------------------------------------------------------------------------
    logger.info("Preparing per-frame optimization data...")
    t1_start = time.time()
    logger.info("Cache not found or invalid. Rebuilding...")
    all_optimization_data2, all_corresps_data2 = build_optimization_data2(
        rgb_dir=rgb_dir,
        mask_dir=mask_dir,
        colmap_dir=colmap_dir,
        opts=opts,
        mesh=refined_mesh,
        start_idx=60,
        end_idx=150,
        stride=5
        )
    
    if len(all_optimization_data2) == 0:
        logger.error("No valid frames collected. Abort.")
        return
        
    logger.info("Phase 2 data alignment complete.")
    
    t1_end = time.time()
    stage_1_time = t1_end - t1_start
    # -------------------------------------------------------------------------
    # Stage 2: Global pose + scale optimization (INIT INCLUDED)
    # -------------------------------------------------------------------------
    logger.info("Starting global pose & scale optimization...")
    t2_start = time.time()
    T_M2W_phase2, result2, final_scale2 = optimize_global_pose(
        all_optimization_data=all_optimization_data2,
        all_corresps_data=all_corresps_data2,
        model_vertices=refined_vertices,
        model_faces=model_faces,
        output_dir=output_dir,
        colmap_dir=colmap_dir,
        mask_dir = mask_dir,
        init_frame_id=0,
    )

    logger.info("Optimization finished.")
    logger.info(f"Final cost: {result2.cost:.6f}")
    logger.info(f"final_scale: {final_scale2}")
    logger.info(f"Optimized T_M2W:\n{T_M2W_phase2}")
    
    t2_end = time.time()
    stage_2_time = t2_end - t2_start
    logger.info(f"Stage 2 (Global Pose) finished in {stage_2_time:.2f}s")
    # -------------------------------------------------------------------------
    # Stage 3: Visualization
    # -------------------------------------------------------------------------
    logger.info("Visualizing final results...")
    for data in tqdm(all_optimization_data2, desc="Final visualization"):
        visualize_projection(
            image_path=data["image_path"],
            output_dir=output_dir,
            K=data["K"],
            T_W2C=data["T_w2c"],
            T_M2W=T_M2W_phase2,
            model_vertices=refined_vertices,
            model_faces=model_faces,
            image_name=f"projected_{os.path.basename(data['image_path'])}",
            root_name="optvis2"
        )

    logger.info("Starting model deformation refinement...")
    t3_start = time.time()
    # 将模型转换到世界坐标系 (Mb)
    V_world_h = np.hstack([refined_vertices, np.ones((len(refined_vertices), 1))])
    V_world = (T_M2W_phase2 @ V_world_h.T).T[:, :3]
    
    refined_vertices2 = refine_model_with_deformation_graph(
        model_vertices=V_world,
        model_faces=model_faces,
        output_dir=output_dir,
        T_M2W=np.eye(4),
        all_optimization_data=all_optimization_data2,
    )
    t3_end = time.time()
    stage_3_time = t3_end - t3_start
    logger.info(f"Stage 3 (Deformation) finished in {stage_3_time:.2f}s")
    t4_start = time.time()
    # model_vertices = refined_vertices
    logger.info("Visualizing final results...")
    # refined_vertices = trimesh.Trimesh(vertices=refined_vertices, faces=model_faces)
    logger.info("Exporting refined model with texture...")
    # 导出文件
    refined_mesh2 = trimesh.Trimesh(
        vertices=refined_vertices2, 
        faces=model_faces,
        visual=original_visual.copy(), # 复制 UV 坐标和材质
        process=False                  # 必须为 False，防止顶点重排
    )

    # 建议导出为 .obj，因为 .ply 对复杂纹理贴图的支持并不标准
    output_subdir = os.path.join(output_dir, "refine2")
    os.makedirs(output_subdir, exist_ok=True)
    output_model_path = os.path.join(output_subdir, "refined_model.obj")
    refined_mesh2.export(output_model_path)
    logger.info(f"Refined model saved to: {output_model_path}")
    for data in tqdm(all_optimization_data2, desc="Final visualization"):
        visualize_projection(
            image_path=data["image_path"],
            output_dir=output_dir,
            K=data["K"],
            T_W2C=data["T_w2c"],
            T_M2W=np.eye(4),
            model_vertices=refined_vertices2,
            model_faces=model_faces,
            image_name=f"projected_{os.path.basename(data['image_path'])}",
            root_name="meshvis2"
        )

    t4_end = time.time()
    stage_4_time = t4_end - t4_start
    total_time = t4_end - start_total

    # --- 打印最终统计表 ---
    print("\n" + "="*50)
    print(f"{'Optimization Phase':<30} | {'Time (s)':<10}")
    print("-" * 50)
    print(f"{'Stage 1: Data Preparation':<30} | {stage_1_time:>10.2f}s")
    print(f"{'Stage 2: Global Pose & Scale':<30} | {stage_2_time:>10.2f}s")
    print(f"{'Stage 3: Deformation Graph':<30} | {stage_3_time:>10.2f}s")
    print(f"{'Stage 4: Model Export':<30} | {stage_4_time:>10.2f}s")
    print("-" * 50)
    print(f"{'Total Execution Time':<30} | {total_time:>10.2f}s")
    print("="*50 + "\n")

    logger.info("========== All Done ==========")


if __name__ == "__main__":
    main()