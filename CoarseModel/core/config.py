# core.config.py
import os
from typing import NamedTuple, List, Tuple, Optional
from pathlib import Path

class AppConfig:
    """全局路径配置"""
    PROJECT_ROOT = Path("/home/zjr/Tracker/CoarseModel")
    
    OBJECT_LIDS = [1]

    # 模型路径
    # MODEL_PATH = "/home/zjr/Tracker/CoarseModel/datasets/wogua/models/wogua_norm.obj"# PROJECT_ROOT / "datasets/wogua/models/wogua_mm.obj"
    DATASETS_PATH = PROJECT_ROOT / "datasets"
    # OUTPUT_BASE_DIR = "/home/zjr/Tracker/foundpose/results"
    # 输出目录
    OUTPUT_ROOT = PROJECT_ROOT / "results"
    TEMPLATE_ROOT = OUTPUT_ROOT / "templates"
    REPRE_ROOT = OUTPUT_ROOT / "object_repre"
    REFINE_ROOT = OUTPUT_ROOT / "refine_model"

    # 设备
    DEVICE = "cuda"
    # 计算资源
    NUM_WORKERS = 10
    USE_GPU = True

class InferOpts(NamedTuple):
    """推理参数配置"""
    # --- 基本信息 ---
    version: str
    repre_version: str
    object_dataset: str
    object_lids: Optional[List[int]] = None
    max_sym_disc_step: float = 0.01

    # --- 图像预处理 ---
    crop: bool = True
    crop_size: Tuple[int, int] = (420, 420)
    crop_rel_pad: float = 0.2

    # --- 特征提取 ---
    extractor_name: str = "dinov2_vitl14"
    grid_cell_size: float = 1.0
    max_num_queries: int = 1000000

    # --- 特征匹配 ---
    match_template_type: str = "tfidf"
    match_top_n_templates: int = 5
    match_feat_matching_type: str = "cyclic_buddies"
    match_top_k_buddies: int = 300

    # --- PnP ---
    pnp_type: str = "opencv"
    pnp_ransac_iter: int = 1000
    pnp_required_ransac_conf: float = 0.99
    pnp_inlier_thresh: float = 10.0
    pnp_refine_lm: bool = True

    final_pose_type: str = "best_coarse"

    # --- Object instance ---
    use_detections: bool = True
    num_preds_factor: float = 1.0
    min_visibility: float = 0.1

    # --- 可视化 / 调试 ---
    save_estimates: bool = True
    save_vis: bool = True
    vis_results: bool = True
    vis_corresp_top_n: int = 100
    vis_feat_map: bool = True
    vis_for_paper: bool = True
    debug: bool = True