# core/preprocess.py
import os
import cv2
import numpy as np
import logging
import pickle

from core.corresp_BA import extract_correspondences
from core.util import (
    quat_to_rotmat, read_colmap_cameras_txt, 
    read_colmap_images_txt, colmap_camera_to_K
)

logger = logging.getLogger(__name__)

def save_optimization_cache(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f)


def load_optimization_cache(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["optimization_data"], data["corresps_data"]

def is_keyframe(i):
    if i < 6:
        return True
    return False

def build_optimization_data(
    rgb_dir: str,
    mask_dir: str,
    colmap_dir: str,
    opts,
):
    """
    Returns:
        all_optimization_data: list[dict]
        all_corresps_data: list[CorrespondenceData]
    """
    cameras_colmap = read_colmap_cameras_txt(
        os.path.join(colmap_dir, "cameras.txt")
    )
    
    images_txt = os.path.join(colmap_dir, "images.txt")
    images_colmap = read_colmap_images_txt(images_txt)

    rgb_files = sorted(f for f in os.listdir(rgb_dir) if f.endswith((".jpg", ".png")))
    mask_files = sorted(f for f in os.listdir(mask_dir) if f.endswith(".png"))

    all_optimization_data = []
    all_corresps_data = []

    # 在循环开始前初始化计数器
    frame_count = 0 

    for rgb_file, mask_file in zip(rgb_files, mask_files):
        frame_name = rgb_file
        if frame_name not in images_colmap:
            logger.warning(f"{frame_name} not in COLMAP, skip")
            continue

        image_path = os.path.join(rgb_dir, rgb_file)
        mask_path = os.path.join(mask_dir, mask_file)

        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            mask = np.zeros(image.shape[:2], dtype=np.uint8)

        img_entry = images_colmap[frame_name]
        cam = cameras_colmap[img_entry["cam_id"]]
        K = colmap_camera_to_K(cam)

        corresp, feature_map_chw_proj, feat_meat = extract_correspondences(
                image_raw=image,
                mask_raw=mask,
                camK=K,
                frame_name=frame_name,
                opts=opts,
            )
        # --- 修改部分：前五帧逻辑控制 ---
        is_kf = is_keyframe(frame_count)
        
        # 计数器累加
        frame_count += 1
        # ------------------------------
        # COLMAP T_w2c 始终计算，保持不变
        R_w2c = img_entry["R"]
        t_w2c = img_entry["t"]
        T_w2c = np.eye(4, dtype=np.float32)
        T_w2c[:3, :3] = R_w2c
        T_w2c[:3, 3] = t_w2c

        q_2d = corresp.query_2d_pts.astype(np.float32)
        X_3d = corresp.model_3d_pts.astype(np.float32)
        X_ids = corresp.model_feat_ids
        if is_kf:
            all_corresps_data.append(corresp)
        else:
            Nonecorresp = None
            all_corresps_data.append(Nonecorresp)
        all_optimization_data.append(
            dict(
                frame_name=frame_name,
                is_keyframe=is_kf,
                K=K.astype(np.float32),
                T_w2c=T_w2c,
                q_2d=q_2d,
                X_3d=X_3d,
                X_ids=X_ids,
                image_path=image_path,
                feat_map_meta=feat_meat, # 存入特征图供后续 BA 跟踪使用
                feature_map_chw_proj=feature_map_chw_proj,
            )
        )
    return all_optimization_data, all_corresps_data
# The `c` variable is being used as a placeholder for each candidate template ID along with
# its corresponding similarity score in the `find_candidate_templates` function. It is part of
# a list of tuples where each tuple contains the template ID and its similarity score with
# respect to the input image contour.
