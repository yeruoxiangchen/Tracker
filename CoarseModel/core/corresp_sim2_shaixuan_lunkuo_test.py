#!/usr/bin/env python3

"""Infers pose from objects."""

import datetime

import sys
REPO_PATH = "/home/zjr/Tracker/CoarseModel"
BOP_TOOLKIT_PATH = "/home/zjr/Tracker/CoarseModel/external/bop_toolkit"

sys.path.insert(0, REPO_PATH)
sys.path.insert(0, BOP_TOOLKIT_PATH)  # 添加这一行

import os
import cv2
import numpy as np
import torch

from script import vis_util

from utils import (
    corresp_util, feature_util, pnp_util, repre_util, json_util,
    knn_util, misc as misc_util, projector_util, data_util, logging, misc,
)
from utils.structs import AlignedBox2f, PinholePlaneCameraModel
from utils.misc import  warp_image, array_to_tensor
from .config import AppConfig, InferOpts
from typing import List, Optional, Any

from core.util import get_instances_from_mask

logger: logging.Logger = logging.get_logger()

import time


def _to_numpy_transform(mat):
    if torch.is_tensor(mat):
        return mat.detach().cpu().numpy()
    return np.asarray(mat)


def _normalize_np_score(score, mask=None):
    score = np.asarray(score, dtype=np.float32)
    out = np.zeros_like(score, dtype=np.float32)
    if mask is None:
        valid = np.isfinite(score)
    else:
        valid = np.asarray(mask).astype(bool) & np.isfinite(score)
    if not np.any(valid):
        return out
    vals = score[valid]
    lo = float(np.percentile(vals, 5))
    hi = float(np.percentile(vals, 95))
    if hi <= lo + 1e-8:
        hi = float(vals.max())
        lo = float(vals.min())
    if hi <= lo + 1e-8:
        out[valid] = 1.0
        return out
    out[valid] = np.clip((score[valid] - lo) / (hi - lo), 0.0, 1.0)
    return out


def _normalize_torch_score(score):
    score = score.float()
    score = torch.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    if score.numel() == 0:
        return score
    lo = torch.quantile(score, 0.05)
    hi = torch.quantile(score, 0.95)
    if float(hi - lo) <= 1e-8:
        max_v = torch.max(score)
        min_v = torch.min(score)
        if float(max_v - min_v) <= 1e-8:
            return torch.ones_like(score)
        lo, hi = min_v, max_v
    return torch.clamp((score - lo) / (hi - lo), 0.0, 1.0)


def _distance_to_edge_score(mask, sigma_px):
    mask_u8 = (np.asarray(mask) > 0).astype(np.uint8) * 255
    if mask_u8.max() == 0:
        return np.zeros(mask_u8.shape, dtype=np.float32)
    kernel = np.ones((3, 3), np.uint8)
    edge = cv2.morphologyEx(mask_u8, cv2.MORPH_GRADIENT, kernel)
    if edge.max() == 0:
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        edge = np.zeros_like(mask_u8)
        cv2.drawContours(edge, contours, -1, 255, 1)
    non_edge = np.where(edge > 0, 0, 255).astype(np.uint8)
    dist = cv2.distanceTransform(non_edge, cv2.DIST_L2, 5)
    sigma_px = max(float(sigma_px), 1e-6)
    return np.exp(-dist / sigma_px).astype(np.float32)


def _sample_np_map(score_map, points_xy, device):
    if points_xy.numel() == 0:
        return torch.empty((0,), dtype=torch.float32, device=device)
    pts = points_xy.detach().cpu().numpy()
    h, w = score_map.shape[:2]
    x = np.clip(np.rint(pts[:, 0]).astype(np.int32), 0, w - 1)
    y = np.clip(np.rint(pts[:, 1]).astype(np.int32), 0, h - 1)
    return torch.as_tensor(score_map[y, x], dtype=torch.float32, device=device)


def _feature_gradient_score_map(feature_map_chw):
    feat = feature_map_chw.float()
    c, h, w = feat.shape
    grad = torch.zeros((h, w), dtype=torch.float32, device=feat.device)
    if w > 1:
        grad[:, :-1] += torch.linalg.norm(feat[:, :, 1:] - feat[:, :, :-1], dim=0)
    if h > 1:
        grad[:-1, :] += torch.linalg.norm(feat[:, 1:, :] - feat[:, :-1, :], dim=0)
    flat = grad.flatten()
    if flat.numel() == 0:
        return grad
    hi = torch.quantile(flat, 0.95)
    if float(hi) <= 1e-8:
        hi = torch.max(flat)
    if float(hi) <= 1e-8:
        return torch.zeros_like(grad)
    return torch.clamp(grad / hi, 0.0, 1.0)


def _sample_feature_score_map(score_hw, points_xy, image_size):
    return feature_util.sample_feature_map_at_points(
        feature_map_chw=score_hw.unsqueeze(0),
        points=points_xy,
        image_size=image_size,
    ).flatten()


def _template_geometry_maps(template_id, template_base_dir, edge_sigma):
    tpl_id = int(template_id.item()) if torch.is_tensor(template_id) else int(template_id)
    mask_path = os.path.join(template_base_dir, f"mask/template_{tpl_id:04d}.png")
    depth_path = os.path.join(template_base_dir, f"depth/template_{tpl_id:04d}.png")

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    mask_bin = mask > 0
    silhouette = _distance_to_edge_score(mask_bin, edge_sigma)

    depth_edge = np.zeros(mask.shape, dtype=np.float32)
    normal_edge = np.zeros(mask.shape, dtype=np.float32)
    depth = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
    if depth is not None:
        depth_f = depth.astype(np.float32)
        valid = mask_bin & np.isfinite(depth_f) & (depth_f > 0)
        if np.any(valid):
            med = float(np.median(depth_f[valid]))
            scale = max(float(np.percentile(depth_f[valid], 95) - np.percentile(depth_f[valid], 5)), 1e-6)
            depth_norm = np.zeros_like(depth_f, dtype=np.float32)
            depth_norm[valid] = (depth_f[valid] - med) / scale
            depth_norm[~valid] = 0.0
            gx = cv2.Sobel(depth_norm, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(depth_norm, cv2.CV_32F, 0, 1, ksize=3)
            depth_edge = _normalize_np_score(np.sqrt(gx * gx + gy * gy), valid)

            nx = -gx
            ny = -gy
            nz = np.ones_like(depth_norm, dtype=np.float32)
            norm = np.sqrt(nx * nx + ny * ny + nz * nz) + 1e-8
            nx, ny, nz = nx / norm, ny / norm, nz / norm
            normal_grad = np.zeros_like(depth_norm, dtype=np.float32)
            for comp in (nx, ny, nz):
                cgx = cv2.Sobel(comp, cv2.CV_32F, 1, 0, ksize=3)
                cgy = cv2.Sobel(comp, cv2.CV_32F, 0, 1, ksize=3)
                normal_grad += cgx * cgx + cgy * cgy
            normal_edge = _normalize_np_score(np.sqrt(normal_grad), valid)

    return {
        "silhouette": silhouette.astype(np.float32),
        "depth_edge": depth_edge.astype(np.float32),
        "normal_edge": normal_edge.astype(np.float32),
        "geom_edge": np.maximum.reduce([silhouette, depth_edge, normal_edge]).astype(np.float32),
    }


def _project_model_points_to_template(points_3d, tpl_camera):
    pts_np = points_3d.detach().cpu().numpy() if torch.is_tensor(points_3d) else np.asarray(points_3d)
    if pts_np.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    T_world_from_eye = _to_numpy_transform(tpl_camera.T_world_from_eye)
    T_eye_from_world = np.linalg.inv(T_world_from_eye)
    pts_h = np.hstack([pts_np, np.ones((pts_np.shape[0], 1), dtype=pts_np.dtype)])
    pts_cam = (T_eye_from_world @ pts_h.T).T[:, :3]
    f = np.asarray(tpl_camera.f, dtype=np.float64)
    c = np.asarray(tpl_camera.c, dtype=np.float64)
    z = np.maximum(pts_cam[:, 2], 1e-8)
    uv = np.zeros((pts_cam.shape[0], 2), dtype=np.float32)
    uv[:, 0] = (pts_cam[:, 0] / z) * f[0] + c[0]
    uv[:, 1] = (pts_cam[:, 1] / z) * f[1] + c[1]
    return uv


def _subset_correspondence(corresp, indices):
    if torch.is_tensor(indices):
        idx_cpu = indices.detach().cpu().long()
    else:
        idx_cpu = torch.as_tensor(indices, dtype=torch.long)
    n = int(corresp["coord_2d"].shape[0])
    out = {}
    for key, value in corresp.items():
        if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == n:
            out[key] = value[idx_cpu.to(value.device)]
        elif isinstance(value, np.ndarray) and value.ndim > 0 and value.shape[0] == n:
            out[key] = value[idx_cpu.numpy()]
        else:
            out[key] = value
    return out


def _coverage_select_indices(points_3d, scores, max_points, min_points, voxel_bins, points_per_voxel):
    n = int(points_3d.shape[0])
    if n == 0:
        return torch.empty((0,), dtype=torch.long, device=scores.device)
    max_points = min(int(max_points), n)
    min_points = min(int(min_points), n)
    pts_np = points_3d.detach().cpu().numpy()
    scores_np = scores.detach().cpu().numpy()
    order = np.argsort(-scores_np)

    mins = pts_np.min(axis=0)
    maxs = pts_np.max(axis=0)
    extent = np.maximum(maxs - mins, 1e-6)
    bins = max(int(voxel_bins), 1)
    vox = np.floor((pts_np - mins[None, :]) / extent[None, :] * bins).astype(np.int32)
    vox = np.clip(vox, 0, bins - 1)

    selected = []
    voxel_counts = {}
    for idx in order:
        key = tuple(int(v) for v in vox[idx])
        count = voxel_counts.get(key, 0)
        if count >= int(points_per_voxel):
            continue
        selected.append(int(idx))
        voxel_counts[key] = count + 1
        if len(selected) >= max_points:
            break

    if len(selected) < min_points:
        selected_set = set(selected)
        for idx in order:
            idx = int(idx)
            if idx not in selected_set:
                selected.append(idx)
                selected_set.add(idx)
            if len(selected) >= min_points:
                break

    return torch.as_tensor(selected, dtype=torch.long, device=scores.device)


def _geometry_filter_correspondence(
    corresp,
    query_dino_edge_scores,
    query_contour_scores,
    repre,
    template_base_dir,
    template_geom_cache,
    opts,
):
    n = int(corresp["coord_2d"].shape[0])
    device = corresp["coord_2d"].device
    min_corr = int(getattr(opts, "geom_filter_min_corr", 24))
    if n < max(6, min_corr):
        return corresp

    q_ids = corresp.get("coord_2d_ids")
    if q_ids is not None and torch.is_tensor(q_ids):
        q_ids = q_ids.long().to(query_dino_edge_scores.device)
        q_edge = query_dino_edge_scores[q_ids].to(device)
        q_contour = query_contour_scores[q_ids].to(device)
    else:
        q_edge = torch.zeros((n,), dtype=torch.float32, device=device)
        q_contour = torch.zeros((n,), dtype=torch.float32, device=device)

    template_id = corresp["template_id"]
    tpl_key = int(template_id.item()) if torch.is_tensor(template_id) else int(template_id)
    if tpl_key not in template_geom_cache:
        template_geom_cache[tpl_key] = _template_geometry_maps(
            tpl_key,
            template_base_dir,
            getattr(opts, "geom_filter_template_edge_sigma", 4.0),
        )

    if template_geom_cache[tpl_key] is None:
        t_silhouette = torch.zeros((n,), dtype=torch.float32, device=device)
        t_geom = torch.zeros((n,), dtype=torch.float32, device=device)
    else:
        tpl_camera = repre.template_cameras_cam_from_model[tpl_key]
        tpl_uv = _project_model_points_to_template(corresp["coord_3d"], tpl_camera)
        tpl_uv_t = torch.as_tensor(tpl_uv, dtype=torch.float32, device=device)
        maps = template_geom_cache[tpl_key]
        t_silhouette = _sample_np_map(maps["silhouette"], tpl_uv_t, device)
        t_depth = _sample_np_map(maps["depth_edge"], tpl_uv_t, device)
        t_normal = _sample_np_map(maps["normal_edge"], tpl_uv_t, device)
        t_geom = torch.maximum(torch.maximum(t_silhouette, t_depth), t_normal)

    dino_conf = corresp.get("coord_conf", torch.ones((n,), dtype=torch.float32, device=device)).float().to(device)
    dino_conf = _normalize_torch_score(dino_conf)
    q_edge = _normalize_torch_score(q_edge)
    q_contour = torch.clamp(q_contour.float(), 0.0, 1.0)
    t_geom = torch.clamp(t_geom.float(), 0.0, 1.0)
    t_silhouette = torch.clamp(t_silhouette.float(), 0.0, 1.0)

    boundary_compat = q_contour * torch.maximum(t_silhouette, t_geom)
    interior_compat = (1.0 - q_contour) * (1.0 - t_silhouette)
    edge_compat = q_edge * t_geom
    geom_compat = torch.clamp(0.45 * boundary_compat + 0.35 * interior_compat + 0.20 * edge_compat, 0.0, 1.0)

    geom_score = (
        0.45 * dino_conf
        + 0.20 * q_edge
        + 0.20 * t_geom
        + 0.15 * geom_compat
    )
    geom_score = torch.nan_to_num(geom_score, nan=0.0, posinf=0.0, neginf=0.0)

    quantile = float(getattr(opts, "geom_filter_min_score_quantile", 0.25))
    if 0.0 < quantile < 1.0 and geom_score.numel() > min_corr:
        score_thr = torch.quantile(geom_score, quantile)
        candidate_idx = torch.nonzero(geom_score >= score_thr).flatten()
    else:
        candidate_idx = torch.arange(n, dtype=torch.long, device=device)
    if candidate_idx.numel() < min_corr:
        candidate_idx = torch.topk(geom_score, k=min(min_corr, n), largest=True).indices

    corresp_scored = _subset_correspondence(corresp, candidate_idx)
    scored_values = geom_score[candidate_idx]
    selected_local = _coverage_select_indices(
        corresp_scored["coord_3d"],
        scored_values,
        max_points=getattr(opts, "geom_filter_max_corr", 160),
        min_points=min_corr,
        voxel_bins=getattr(opts, "geom_filter_voxel_bins", 6),
        points_per_voxel=getattr(opts, "geom_filter_points_per_voxel", 3),
    )
    filtered = _subset_correspondence(corresp_scored, selected_local)
    filtered["coord_conf"] = scored_values[selected_local].to(device)
    filtered["geom_score"] = filtered["coord_conf"]
    filtered["geom_filter_stats"] = {
        "before": n,
        "after_score": int(candidate_idx.numel()),
        "after_coverage": int(filtered["coord_2d"].shape[0]),
    }
    return filtered


def _pnp_inlier_correspondence(corresp, inliers):
    if inliers is None:
        return corresp
    idx = np.asarray(inliers).reshape(-1)
    if idx.size < 6:
        return corresp
    return _subset_correspondence(corresp, idx)


def _camera_intrinsic_np(camera):
    fx, fy = camera.f
    cx, cy = camera.c
    return np.array(
        [
            [float(fx), 0.0, float(cx)],
            [0.0, float(fy), float(cy)],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _project_points_m2c(points_3d, R_m2c, t_m2c, camera):
    pts = points_3d.detach().cpu().numpy() if torch.is_tensor(points_3d) else np.asarray(points_3d)
    if pts.size == 0:
        return np.zeros((0, 2), dtype=np.float64), np.zeros((0,), dtype=np.float64)
    R = np.asarray(R_m2c, dtype=np.float64)
    t = np.asarray(t_m2c, dtype=np.float64).reshape(3)
    Xc = (R @ pts.T).T + t[None, :]
    z = Xc[:, 2]
    K = _camera_intrinsic_np(camera)
    uvw = (K @ Xc.T).T
    uv = uvw[:, :2] / np.maximum(uvw[:, 2:3], 1e-8)
    return uv, z


def _reprojection_error_stats(corresp, R_m2c, t_m2c, camera):
    uv, z = _project_points_m2c(corresp["coord_3d"], R_m2c, t_m2c, camera)
    q = corresp["coord_2d"].detach().cpu().numpy() if torch.is_tensor(corresp["coord_2d"]) else np.asarray(corresp["coord_2d"])
    valid = z > 1e-6
    if not np.any(valid):
        return {"median": float("inf"), "mean": float("inf"), "max": float("inf"), "valid": 0}
    err = np.linalg.norm(uv[valid] - q[valid], axis=1)
    if err.size == 0:
        return {"median": float("inf"), "mean": float("inf"), "max": float("inf"), "valid": 0}
    return {
        "median": float(np.median(err)),
        "mean": float(np.mean(err)),
        "max": float(np.max(err)),
        "valid": int(err.size),
    }


def _coverage_score(points_3d, model_vertices):
    pts = points_3d.detach().cpu().numpy() if torch.is_tensor(points_3d) else np.asarray(points_3d)
    verts = model_vertices.detach().cpu().numpy() if torch.is_tensor(model_vertices) else np.asarray(model_vertices)
    if pts.shape[0] < 4 or verts.size == 0:
        return 0.0
    model_extent = np.maximum(verts.max(axis=0) - verts.min(axis=0), 1e-6)
    pts_extent = np.maximum(pts.max(axis=0) - pts.min(axis=0), 0.0)
    axis_cover = np.clip(pts_extent / model_extent, 0.0, 1.0)

    bins = 4
    mins = verts.min(axis=0)
    vox = np.floor((pts - mins[None, :]) / model_extent[None, :] * bins).astype(np.int32)
    vox = np.clip(vox, 0, bins - 1)
    occupied = len({tuple(v) for v in vox})
    occupied_score = occupied / float(min(pts.shape[0], bins ** 3))
    return float(np.clip(0.55 * axis_cover.mean() + 0.45 * occupied_score, 0.0, 1.0))


def _bbox_iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    if denom <= 1e-8:
        return 0.0
    return float(inter / denom)


def _pose_mask_score(R_m2c, t_m2c, camera, mask_modal, model_vertices):
    verts = model_vertices.detach().cpu().numpy() if torch.is_tensor(model_vertices) else np.asarray(model_vertices)
    if verts.size == 0:
        return 0.0
    if verts.shape[0] > 3000:
        sample_ids = np.linspace(0, verts.shape[0] - 1, 3000).astype(np.int64)
        verts = verts[sample_ids]

    mask = (np.asarray(mask_modal) > 0).astype(np.uint8)
    if mask.max() == 0:
        return 0.0
    h, w = mask.shape[:2]
    uv, z = _project_points_m2c(verts, R_m2c, t_m2c, camera)
    visible = (
        (z > 1e-6)
        & (uv[:, 0] >= 0) & (uv[:, 0] < w)
        & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    )
    if not np.any(visible):
        return 0.0

    uv_vis = uv[visible]
    xi = np.clip(np.rint(uv_vis[:, 0]).astype(np.int32), 0, w - 1)
    yi = np.clip(np.rint(uv_vis[:, 1]).astype(np.int32), 0, h - 1)
    inside_ratio = float(np.mean(mask[yi, xi] > 0))

    proj_bbox = np.array([uv_vis[:, 0].min(), uv_vis[:, 1].min(), uv_vis[:, 0].max(), uv_vis[:, 1].max()], dtype=np.float64)
    ys, xs = np.nonzero(mask)
    mask_bbox = np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float64)
    bbox_iou = _bbox_iou_xyxy(proj_bbox, mask_bbox)
    return float(np.clip(0.65 * inside_ratio + 0.35 * bbox_iou, 0.0, 1.0))


def _pose_selection_score(num_inliers, reproj_stats, coverage_score, mask_score, pnp_inlier_thresh):
    inlier_score = 1.0 - np.exp(-float(num_inliers) / 40.0)
    reproj_median = reproj_stats.get("median", float("inf"))
    if not np.isfinite(reproj_median):
        reproj_score = 0.0
    else:
        reproj_score = float(np.exp(-reproj_median / max(float(pnp_inlier_thresh), 1e-6)))
    return float(
        0.30 * inlier_score
        + 0.30 * reproj_score
        + 0.20 * float(np.clip(coverage_score, 0.0, 1.0))
        + 0.20 * float(np.clip(mask_score, 0.0, 1.0))
    )


class CorrespondenceData(object):
    def __init__(self, frame_name, query_2d_pts, model_3d_pts, model_feat_ids, best_pose = None, camera_c2w = None):
        self.frame_name = frame_name
        self.query_2d_pts = query_2d_pts
        self.model_3d_pts = model_3d_pts
        self.model_feat_ids = model_feat_ids
        self.best_pose = best_pose
        self.cam = camera_c2w
        
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

def warp_points_perspective(src_camera, dst_camera, src_points, src_depths):
    """
    将 2D 点从源相机坐标系 (src_camera) 映射到目标相机坐标系 (dst_camera)，
    采用完整透视几何（使用真实深度 Z）。
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

def visualize_correspondences(
    data: CorrespondenceData,
    template_id: int,
    image_raw: np.ndarray,
    repre,  # Object representation 包含模板相机信息
    template_base_dir: str,
    output_dir: str,
    num_batches: int = 5
):
    """
    将提取的2D-3D对应关系投影到模板图像上，并与原图对比保存。
    """
    frame_id = os.path.splitext(data.frame_name)[0]
    frame_dir = os.path.join(output_dir, frame_id)
    os.makedirs(frame_dir, exist_ok=True)

    # 1. 准备原始图像 (确保是 uint8 格式)
    if image_raw.max() <= 1.0:
        vis_base_img = (image_raw * 255).astype(np.uint8)
    else:
        vis_base_img = image_raw.astype(np.uint8).copy()

    # 2. 获取匹配时使用的模板信息
    # 注意：CorrespondenceData 中需要包含匹配到的 template_id
    # 如果你的 data 对象里没存，需要从推理过程中传出来
    # template_id = getattr(data, 'template_id', 0) 
    tpl_camera = repre.template_cameras_cam_from_model[template_id]
    
    # 3. 加载模板图像
    tpl_path = os.path.join(template_base_dir, f"rgb/template_{template_id:04d}.png")
    if os.path.exists(tpl_path):
        tpl_img = cv2.imread(tpl_path)
        tpl_img = cv2.cvtColor(tpl_img, cv2.COLOR_BGR2RGB)
    else:
        logger.warning(f"Template image not found: {tpl_path}")
        tpl_img = np.zeros((tpl_camera.height, tpl_camera.width, 3), dtype=np.uint8)

    # 4. 计算 3D 点在模板上的投影
    # T_world_from_eye 在模板相机里实际上是 T_cam_from_model 的逆
    T_c2m = np.linalg.inv(tpl_camera.T_world_from_eye)
    pts_h = np.hstack([data.model_3d_pts, np.ones((data.model_3d_pts.shape[0], 1))])
    pts_cam = (T_c2m @ pts_h.T).T[:, :3]
    
    f = np.array(tpl_camera .f)         # [fx, fy]
    c = np.array(tpl_camera .c)         # [cx, cy]
    proj_2d = np.zeros((pts_cam.shape[0], 2))
    proj_2d[:, 0] = (pts_cam[:, 0] / pts_cam[:, 2]) * f[0] + c[0]
    proj_2d[:, 1] = (pts_cam[:, 1] / pts_cam[:, 2]) * f[1] + c[1]
    
    proj_points = proj_2d.astype(int)
    query_points = data.query_2d_pts.astype(int)

    # 5. 分批次绘制并保存
    num_points = query_points.shape[0]
    batch_size = (num_points + num_batches - 1) // num_batches

    for b in range(num_batches):
        start = b * batch_size
        end = min((b + 1) * batch_size, num_points)
        if start >= end: break

        img_left = vis_base_img.copy()
        img_right = tpl_img.copy()

        for i in range(start, end):
            p2d = tuple(query_points[i])
            p3d = tuple(proj_points[i])
            color_2d = (0, 255, 0)  # 绿色
            color_3d = (255, 0, 0)  # 蓝色 (RGB)

            # 绘制圆点
            cv2.circle(img_left, p2d, 2, color_2d, -1)
            cv2.circle(img_right, p3d, 2, color_3d, -1)

            # 绘制索引编号
            cv2.putText(img_left, str(i), (p2d[0]+2, p2d[1]-2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, color_2d, 1)
            cv2.putText(img_right, str(i), (p3d[0]+2, p3d[1]-2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, color_3d, 1)
        # === 统一高度，便于 hstack ===
        h_left = img_left.shape[0]
        h_right = img_right.shape[0]

        if h_left != h_right:
            scale = h_left / h_right
            new_w = int(img_right.shape[1] * scale)
            img_right = cv2.resize(img_right, (new_w, h_left))
        # 水平拼接并保存
        combined = np.hstack([img_left, img_right])
        save_path = os.path.join(frame_dir, f"batch_{b+1:02d}.png")
        cv2.imwrite(save_path, cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))

    logger.info(f"Visualized {num_points} points in {num_batches} batches to {frame_dir}")

def build_crop_view(
    orig_image: np.ndarray,
    orig_mask: np.ndarray,
    orig_camera: PinholePlaneCameraModel,
    box_amodal: AlignedBox2f,
    crop_size: tuple[int, int],
    crop_rel_pad: float,
):
    """
    根据 amodal box 构建裁剪视图和虚拟相机
    返回：
        image_crop, mask_crop, crop_camera, crop_box, box_amodal_crop
    """
    # 1. crop box
    crop_box = misc_util.calc_crop_box(
        box=box_amodal,
        make_square=True,
    )

    # 2. virtual camera
    crop_camera = misc_util.construct_crop_camera(
        box=crop_box,
        camera_model_c2w=orig_camera,
        viewport_size=crop_size,
        viewport_rel_pad=crop_rel_pad,
    )

    # 3. warp image & mask
    interpolation = (
        cv2.INTER_AREA
        if crop_box.width >= crop_camera.width
        else cv2.INTER_LINEAR
    )

    image_crop = warp_image(
        src_camera=orig_camera,
        dst_camera=crop_camera,
        src_image=orig_image,
        interpolation=interpolation,
    )

    mask_crop = warp_image(
        src_camera=orig_camera,
        dst_camera=crop_camera,
        src_image=orig_mask,
        interpolation=cv2.INTER_NEAREST,
    )

    # 4. recompute bounding box in crop
    ys, xs = mask_crop.nonzero()
    box = np.array(misc_util.calc_2d_box(xs, ys))
    box_amodal_crop = AlignedBox2f(
        left=box[0], top=box[1], right=box[2], bottom=box[3]
    )

    return image_crop, mask_crop, crop_camera, crop_box, box_amodal_crop

def recover_pose_and_points_to_orig(
    best_pose: dict,
    q_2d: np.ndarray,
    X_3d: np.ndarray,
    crop_camera: PinholePlaneCameraModel,
    orig_camera: PinholePlaneCameraModel,
    use_crop: bool,
):
    """
    1. 将模型位姿从裁剪相机坐标系恢复到原始相机坐标系
    2. 若启用 crop，将裁剪图上的 2D 点恢复到原图坐标
    返回：
        best_pose_orig, q_2d_orig, q_2d_crop(optional)
    """
    # ---------- 1. pose: crop -> orig ----------
    R_m2c_crop = best_pose["R_m2c"]
    t_m2c_crop = best_pose["t_m2c"].reshape(3, 1)

    R_c2w_crop = crop_camera.T_world_from_eye[:3, :3]
    t_c2w_crop = crop_camera.T_world_from_eye[:3, 3:4]

    R_c2w_orig = orig_camera.T_world_from_eye[:3, :3]
    t_c2w_orig = orig_camera.T_world_from_eye[:3, 3:4]
    
    # world -> original camera
    R_w2c_orig = R_c2w_orig.T
    t_w2c_orig = -R_w2c_orig @ t_c2w_orig
    # model -> original camera
    R_m2c_orig = R_w2c_orig @ R_c2w_crop @ R_m2c_crop
    t_m2c_orig = (
        R_w2c_orig @ (R_c2w_crop @ t_m2c_crop + t_c2w_crop)
        + t_w2c_orig
    )
    best_pose_orig = best_pose.copy()
    best_pose_orig["R_m2c"] = R_m2c_orig
    best_pose_orig["t_m2c"] = t_m2c_orig.reshape(3)

    # ---------- 2. points: crop -> orig ----------
    q_2d_crop = None
    q_2d_orig = q_2d
    
    X_crop_cam = (R_m2c_crop @ X_3d.T).T + t_m2c_crop.reshape(3)
    Z_3d = np.maximum(X_crop_cam[:, 2], 1e-6)
    q_2d_orig = warp_points_perspective(
        src_camera=crop_camera,
        dst_camera=orig_camera,
        src_points=q_2d,
        src_depths=Z_3d,
    )
    q_2d_crop = q_2d

    return best_pose_orig, q_2d_orig, q_2d_crop

def extract_correspondences(
    image_raw: np.ndarray, 
    mask_raw: np.ndarray, 
    camK: np.ndarray, 
    frame_name: str,
    opts: InferOpts,
    # --- 新增传入的预加载资源 ---
    extractor: torch.nn.Module,
    repre: Any,
    visual_words_knn_index: Optional[Any],
    template_knn_indices: List[Any],
    device: str
) -> Optional[CorrespondenceData]:
    
    t_func_start = time.perf_counter()
    stats = {}
    
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

    object_lids = opts.object_lids 
    for object_lid in object_lids:
        version = opts.version
        if version == "":
            version = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        signature = misc.slugify(opts.object_dataset) + "_{}".format(version)
        output_dir = os.path.join(
            AppConfig.OUTPUT_ROOT, "inference", signature, str(object_lid)
        )
        os.makedirs(output_dir, exist_ok=True)
        # Save parameters to a file.
        config_path = os.path.join(output_dir, "config.json")
        json_util.save_json(config_path, opts)

        templates_path = os.path.join(
            "templates",
            opts.version,
            opts.object_dataset,
            str(object_lid),
        )
        template_base_dir = os.path.join(
            AppConfig.OUTPUT_ROOT,
            templates_path,
        )
        # 4. 图像准备与神经网络推理 (GPU Inference)
        # 4. 特征提取 (Inference)
        t_inf = time.perf_counter()
        # Perform inference on each selected image.
        sample = data_util.prepare_sample3(image_raw, scene_cameras)
        # Camera parameters.
        orig_camera_c2w = sample.camera
        orig_image_size = (
            orig_camera_c2w.width,
            orig_camera_c2w.height,
        )
        # Get info about object instances for which we want to estimate pose. 是一个列表，每个元素对应 当前图像中一个对象实例
        instances = get_instances_from_mask(
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
                image_np_hwc, mask_modal, crop_camera_model_c2w, _, box_amodal = build_crop_view(
                    orig_image=orig_image_np_hwc,
                    orig_mask=orig_mask_modal,
                    orig_camera=orig_camera_c2w,
                    box_amodal=orig_box_amodal,
                    crop_size=opts.crop_size,
                    crop_rel_pad=opts.crop_rel_pad,
                )
                # The virtual camera is becoming the main camera.
                camera_c2w = crop_camera_model_c2w

            # Extract feature map from the crop. 将图像转为张量并提取特征
            image_tensor_chw = array_to_tensor(image_np_hwc).to(torch.float32).permute(2,0,1).to(device)
            image_tensor_bchw = image_tensor_chw.unsqueeze(0)
            extractor_output = extractor(image_tensor_bchw)
            feature_map_chw = extractor_output["feature_maps"][0] # 取第一个层的特征图，形状 (C_feat, H_feat, W_feat)。

            stats['Net Inference'] = time.perf_counter() - t_inf
            # 5. 特征采样与匹配 (核心瓶颈)
            t_match = time.perf_counter()
            
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

            query_dino_edge_scores = torch.zeros((query_points.shape[0],), dtype=torch.float32, device=device)
            query_contour_scores = torch.zeros((query_points.shape[0],), dtype=torch.float32, device=device)
            if getattr(opts, "use_geometry_corresp_filter", True) and query_points.shape[0] > 0:
                dino_edge_map = _feature_gradient_score_map(feature_map_chw_proj)
                query_dino_edge_scores = _sample_feature_score_map(
                    score_hw=dino_edge_map,
                    points_xy=query_points,
                    image_size=(image_np_hwc.shape[1], image_np_hwc.shape[0]),
                )
                contour_score_map = _distance_to_edge_score(
                    mask_modal,
                    getattr(opts, "geom_filter_contour_sigma", 12.0),
                )
                query_contour_scores = _sample_np_map(contour_score_map, query_points, device)

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
            stats['Feature Matching'] = time.perf_counter() - t_match
            
            # --- 3. [新增] 轮廓匹配整合 ---
            t_contour = time.perf_counter()
            # 提取当前图像轮廓 (2D)
            cur_contours, _ = cv2.findContours((mask_modal * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            has_cur_contour = len(cur_contours) > 0
                        
            # # 遍历每一个候选模板，为其增加轮廓约束
            # for cor_idx in range(len(corresp)):
            #     t_id = int(corresp[cor_idx]["template_id"].item())
                
            #     # --- Feature correspondences confidence filtering ---
            #     max_feat = 50
            #     if "coord_conf" in corresp[cor_idx]:
            #         conf = corresp[cor_idx]["coord_conf"]
            #         if conf.shape[0] > max_feat:
            #             topk = torch.topk(conf, k=max_feat)
            #             idx = topk.indices
            #             corresp[cor_idx]["coord_2d"] = corresp[cor_idx]["coord_2d"][idx]
            #             corresp[cor_idx]["coord_3d"] = corresp[cor_idx]["coord_3d"][idx]
            #             corresp[cor_idx]["coord_conf"] = corresp[cor_idx]["coord_conf"][idx]
            #             if "nn_vertex_ids" in corresp[cor_idx]:
            #                 corresp[cor_idx]["nn_vertex_ids"] = corresp[cor_idx]["nn_vertex_ids"][idx]
                            
            #     # 加载该模板的 3D 轮廓点
            #     tpl_mask_path = os.path.join(template_base_dir, f"mask/template_{t_id:04d}.png")
            #     tpl_depth_path = os.path.join(template_base_dir, f"depth/template_{t_id:04d}.png")
                
            #     if os.path.exists(tpl_mask_path) and os.path.exists(tpl_depth_path) and has_cur_contour:
            #         tpl_mask = cv2.imread(tpl_mask_path, cv2.IMREAD_GRAYSCALE)
            #         tpl_depth = cv2.imread(tpl_depth_path, cv2.IMREAD_ANYDEPTH)
            #         tpl_camera = repre.template_cameras_cam_from_model[t_id]
                    
            #         # 提取模板 3D 轮廓 (逻辑同你之前的代码)
            #         tpl_cont, _ = cv2.findContours(tpl_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            #         if len(tpl_cont) > 0:
            #             tpl_pts_2d = tpl_cont[0].reshape(-1, 2)
            #             # 采样并反投影
            #             z_c = tpl_depth[np.clip(tpl_pts_2d[:, 1], 0, tpl_depth.shape[0]-1), 
            #                              np.clip(tpl_pts_2d[:, 0], 0, tpl_depth.shape[1]-1)].astype(np.float32)
            #             valid_z = z_c > 0.1
                        
            #             # 计算模型空间的 3D 点
            #             cx = tpl_camera.c[0].item() if torch.is_tensor(tpl_camera.c[0]) else tpl_camera.c[0]
            #             cy = tpl_camera.c[1].item() if torch.is_tensor(tpl_camera.c[1]) else tpl_camera.c[1]
            #             fx = tpl_camera.f[0].item() if torch.is_tensor(tpl_camera.f[0]) else tpl_camera.f[0]
            #             fy = tpl_camera.f[1].item() if torch.is_tensor(tpl_camera.f[1]) else tpl_camera.f[1]

            #             # 使用转换后的变量进行计算
            #             x_c = (tpl_pts_2d[valid_z, 0] - cx) * z_c[valid_z] / fx
            #             y_c = (tpl_pts_2d[valid_z, 1] - cy) * z_c[valid_z] / fy
            #             # y_c = (tpl_pts_2d[valid_z, 1] - tpl_camera.c[1]) * z_c[valid_z] / tpl_camera.f[1]
            #             pts_3d_cam_h = np.column_stack([x_c, y_c, z_c[valid_z], np.ones_like(x_c)])
            #             pts_3d_model = (tpl_camera.T_world_from_eye @ pts_3d_cam_h.T).T[:, :3]
                        
            #             # 对齐采样 (各取 50 个点)
            #             target_n = 50
            #             idx_cur = np.linspace(0, len(cur_contours[0]) - 1, target_n).astype(int)
            #             idx_tpl = np.linspace(0, len(pts_3d_model) - 1, target_n).astype(int)
                        
            #             cont_2d = torch.as_tensor(cur_contours[0].reshape(-1, 2)[idx_cur], device=device, dtype=torch.float32)
            #             cont_3d = torch.as_tensor(pts_3d_model[idx_tpl], device=device, dtype=torch.float32)
                        
            #             # 合并到当前 corresp
            #             corresp[cor_idx]["coord_2d"] = torch.cat([corresp[cor_idx]["coord_2d"], cont_2d], dim=0)
            #             corresp[cor_idx]["coord_3d"] = torch.cat([corresp[cor_idx]["coord_3d"], cont_3d], dim=0)
            #             # 为轮廓点补充 dummy 信息
            #             if "coord_conf" in corresp[cor_idx]:
            #                 cont_conf = torch.full((target_n,), 0.5, device=device)
            #                 corresp[cor_idx]["coord_conf"] = torch.cat([corresp[cor_idx]["coord_conf"], cont_conf], dim=0)
            #             if "nn_vertex_ids" in corresp[cor_idx]:
            #                 dummy_ids = torch.full((target_n,), -1, dtype=torch.long, device=device)
            #                 corresp[cor_idx]["nn_vertex_ids"] = torch.cat([corresp[cor_idx]["nn_vertex_ids"], dummy_ids], dim=0)
            
            stats['Contour Matching'] = time.perf_counter() - t_contour
            # 6. PnP 求解 (RANSAC 迭代)
            t_pnp = time.perf_counter()
            
            # Estimate coarse poses from corespondences.
            coarse_poses = []
            template_geom_cache = {}
            model_vertices_for_scoring = repre.vertices
            for corresp_id, corresp_curr in enumerate(corresp): # 遍历之前生成的每组 2D-3D 对应关系 corresp
                if getattr(opts, "use_geometry_corresp_filter", True):
                    corresp_for_pose = _geometry_filter_correspondence(
                        corresp=corresp_curr,
                        query_dino_edge_scores=query_dino_edge_scores,
                        query_contour_scores=query_contour_scores,
                        repre=repre,
                        template_base_dir=template_base_dir,
                        template_geom_cache=template_geom_cache,
                        opts=opts,
                    )
                else:
                    corresp_for_pose = corresp_curr

                corresp_for_pose["coord_3d"] = corresp_for_pose["coord_3d"]
                # We need at least 3 correspondences for P3P.
                num_corresp = len(corresp_for_pose["coord_2d"])
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
                    corresp=corresp_for_pose,
                    camera_c2w=camera_c2w,
                    pnp_type=opts.pnp_type,
                    pnp_ransac_iter=opts.pnp_ransac_iter,
                    pnp_inlier_thresh=opts.pnp_inlier_thresh,
                    pnp_required_ransac_conf=opts.pnp_required_ransac_conf,
                    pnp_refine_lm=opts.pnp_refine_lm,
                )

                if coarse_pose_success:
                    final_corresp_for_pose = corresp_for_pose
                    if getattr(opts, "pnp_use_inlier_corresp", True):
                        final_corresp_for_pose = _pnp_inlier_correspondence(corresp_for_pose, inliers_coarse)
                        if len(final_corresp_for_pose["coord_2d"]) >= 6:
                            (
                                refined_success,
                                R_refined,
                                t_refined,
                                inliers_refined,
                                quality_refined,
                            ) = pnp_util.estimate_pose(
                                corresp=final_corresp_for_pose,
                                camera_c2w=camera_c2w,
                                pnp_type=opts.pnp_type,
                                pnp_ransac_iter=max(100, opts.pnp_ransac_iter // 4),
                                pnp_inlier_thresh=opts.pnp_inlier_thresh,
                                pnp_required_ransac_conf=opts.pnp_required_ransac_conf,
                                pnp_refine_lm=opts.pnp_refine_lm,
                            )
                            if refined_success:
                                R_m2c_coarse = R_refined
                                t_m2c_coarse = t_refined
                                inliers_coarse = inliers_refined
                                quality_coarse = quality_refined
                                final_corresp_for_pose = _pnp_inlier_correspondence(final_corresp_for_pose, inliers_refined)

                    reproj_stats = _reprojection_error_stats(
                        final_corresp_for_pose,
                        R_m2c_coarse,
                        t_m2c_coarse,
                        camera_c2w,
                    )
                    coverage = _coverage_score(final_corresp_for_pose["coord_3d"], model_vertices_for_scoring)
                    mask_score = _pose_mask_score(
                        R_m2c=R_m2c_coarse,
                        t_m2c=t_m2c_coarse,
                        camera=camera_c2w,
                        mask_modal=mask_modal,
                        model_vertices=model_vertices_for_scoring,
                    )
                    selection_score = _pose_selection_score(
                        num_inliers=len(final_corresp_for_pose["coord_2d"]),
                        reproj_stats=reproj_stats,
                        coverage_score=coverage,
                        mask_score=mask_score,
                        pnp_inlier_thresh=opts.pnp_inlier_thresh,
                    )

                    coarse_poses.append(
                        {
                            "type": "coarse",
                            "R_m2c": R_m2c_coarse,
                            "t_m2c": t_m2c_coarse,
                            "corresp_id": corresp_id,
                            "quality": quality_coarse,
                            "inliers": inliers_coarse,
                            "num_corr_before_pnp": int(num_corresp),
                            "num_corr_after_pnp": int(len(final_corresp_for_pose["coord_2d"])),
                            "corresp": final_corresp_for_pose,
                            "reproj_median": reproj_stats["median"],
                            "reproj_mean": reproj_stats["mean"],
                            "coverage_score": coverage,
                            "mask_score": mask_score,
                            "selection_score": selection_score,
                        }
                    )
            if not coarse_poses:
                logger.info("No valid coarse poses after geometry filtering/PnP.")
                continue
            # Find the best coarse pose.
            best_coarse_quality = None
            best_coarse_pose_id = 0
            for coarse_pose_id, pose in enumerate(coarse_poses):
                if (
                    best_coarse_quality is None
                    or pose["selection_score"] > best_coarse_quality
                ):
                    best_coarse_pose_id = coarse_pose_id
                    best_coarse_quality = pose["selection_score"]
            best_corresp_id = coarse_poses[best_coarse_pose_id]["corresp_id"]
            corresp_final = coarse_poses[best_coarse_pose_id]["corresp"]
            
            # [新增] 输出最终选定的模板 ID
            final_template_id = corresp_final["template_id"]
            logger.info(
                f"Final selected template ID: {final_template_id} "
                f"(from coarse_pose_id: {best_coarse_pose_id}, "
                f"corr {coarse_poses[best_coarse_pose_id]['num_corr_before_pnp']} -> "
                f"{coarse_poses[best_coarse_pose_id]['num_corr_after_pnp']}, "
                f"score={coarse_poses[best_coarse_pose_id]['selection_score']:.3f}, "
                f"repr={coarse_poses[best_coarse_pose_id]['reproj_median']:.2f}px, "
                f"cov={coarse_poses[best_coarse_pose_id]['coverage_score']:.2f}, "
                f"mask={coarse_poses[best_coarse_pose_id]['mask_score']:.2f})"
            )
            
            # 2D 观测点 (N, 2)
            q_2d = corresp_final["coord_2d"].cpu().numpy()
            # 3D 模型点坐标 (N, 3)
            X_3d = corresp_final["coord_3d"].cpu().numpy()
            # 3D 模型特征点 ID (N,)
            X_ids = corresp_final["nn_vertex_ids"].cpu().numpy()
            corresp_scores = corresp_final["coord_conf"]
            
            best_pose = coarse_poses[best_coarse_pose_id]
            # ... 执行 estimate_pose ...
            stats['PnP Solver'] = time.perf_counter() - t_pnp
            # 7. 可视化与 IO
            t_vis = time.perf_counter()
            # # # ------------------ 输出位姿 ------------------
            dataset_model_dir = os.path.join(AppConfig.DATASETS_PATH, opts.object_dataset, "models")
            model_path = os.path.join(dataset_model_dir, f"{opts.object_dataset}_norm.obj")
            pose_path = os.path.join(AppConfig.REFINE_ROOT, opts.object_dataset, "pose_es")
            vis_util.visualize_pose_overlay(
                image_raw=image_raw,
                best_pose=best_pose,
                model_path=model_path,
                orig_camera=orig_camera_c2w,
                crop_camera=crop_camera_model_c2w,
                frame_name=frame_name,
                output_dir=pose_path
            )
            
            # 恢复裁剪位姿和2d点
            best_pose, q_2d, q_crop = recover_pose_and_points_to_orig(
                best_pose=best_pose,
                q_2d=q_2d,
                X_3d=X_3d,
                crop_camera=camera_c2w,
                orig_camera=orig_camera_c2w,
                use_crop=opts.crop,
            )
                
            final_q_2d = q_2d
            final_X_3d = X_3d
            final_X_ids = X_ids
            all_scores = corresp_scores
            
            #重复3d-2d对应剪枝
            unique_ids = np.unique(final_X_ids)
            best_indices = []
            for uid in unique_ids:
                idxs = np.where(final_X_ids == uid)[0]
                best_idx = idxs[np.argmax(all_scores[idxs].cpu().numpy())]
                best_indices.append(best_idx)
            # contour points
            contour_indices = np.where(final_X_ids < 0)[0]
            best_indices.extend(contour_indices)
        
            best_indices = np.array(best_indices)
            final_q_2d = final_q_2d[best_indices]
            final_X_3d = final_X_3d[best_indices]
            final_X_ids = final_X_ids[best_indices]
            all_scores = all_scores[best_indices]
            q_crop = q_crop[best_indices]
            
            correspondence_data = CorrespondenceData(
                frame_name=frame_name,
                query_2d_pts=final_q_2d,
                model_3d_pts=final_X_3d,
                model_feat_ids=final_X_ids,
                best_pose = best_pose,
                camera_c2w= orig_camera_c2w
            )
            template_id = int(corresp_final["template_id"].item()) if "template_id" in corresp_final else 0
            templates_path = os.path.join(
                "templates",
                opts.version,
                opts.object_dataset,
                str(object_lid),
            )
            template_base_dir = os.path.join(
                AppConfig.OUTPUT_ROOT,
                templates_path,
            )
            vis_debug_path = os.path.join(AppConfig.REFINE_ROOT, opts.object_dataset, "vis_debug")
            visualize_correspondences(
                data=correspondence_data,
                template_id=template_id,
                image_raw=image_raw,
                repre=repre,
                template_base_dir=template_base_dir,
                output_dir=vis_debug_path
            )
            # 创建并返回此帧的对应数据
            stats['Visualization & Post'] = time.perf_counter() - t_vis
    # --- 计算并打印总时长 ---
    total_duration = time.perf_counter() - t_func_start
    
    # 打印精美的耗时统计表
    logger.info(f"\n{'='*30}\n Performance Summary ({frame_name})\n{'-'*30}")
    for stage, duration in stats.items():
        logger.info(f"{stage:<20}: {duration:>7.4f}s ({duration/total_duration:2.1%})")
    logger.info(f"{'-'*30}\n{'TOTAL TIME':<20}: {total_duration:>7.4f}s\n{'='*30}")
    return correspondence_data
