# core/global_optimize.py
import os
import cv2
import json
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from datetime import datetime

from core.util import ( 
    axis_angle_to_rotmat, project_point, 
    read_colmap_points3D, read_colmap_images_txt
)
from collections import defaultdict

from contextlib import redirect_stdout
from scipy.spatial import cKDTree
from script.vis_util import visualize_contour_matches
from script.vis_util import (
    visualize_projection,
    visualize_per_frame_alignment
    )
import time
from collections import defaultdict

def project_points_batch(P_W, K, T_w2c):
    # (保持不变)
    R = T_w2c[:3, :3]
    t = T_w2c[:3, 3]

    P_C = (R @ P_W.T).T + t
    z = P_C[:, 2]

    uvw = (K @ P_C.T).T
    uv = uvw[:, :2] / uvw[:, 2:3]

    return uv, z


def get_object_3d_points(
    points3d_path,
    all_optimization_data,
    mask_dir,
    images_colmap,
    min_observations=3,
):
    # (保持不变)
    print("\n--- Collecting object 3D points (FAST VERSION) ---")

    points3d = read_colmap_points3D(points3d_path)
    if not points3d:
        return np.array([])

    id_to_data = {}
    for entry in all_optimization_data:
        frame_name = entry['frame_name']
        if frame_name in images_colmap:
            image_id = images_colmap[frame_name]['image_id']
            id_to_data[image_id] = {
                'K': entry['K'],
                'T_w2c': entry['T_w2c'],
                'mask_path': os.path.join(
                    mask_dir,
                    frame_name.replace(".jpg", ".png").replace(".JPG", ".png")
                )
            }

    image_to_points = defaultdict(list)
    points_xyz = {}

    for pid, pdata in points3d.items():
        points_xyz[pid] = pdata['xyz']
        track = pdata['track_list']
        for i in range(0, len(track), 2):
            image_id = track[i]
            if image_id in id_to_data:
                image_to_points[image_id].append(pid)

    mask_cache = {}

    def load_mask(path):
        if path not in mask_cache:
            if not os.path.exists(path):
                mask_cache[path] = None
            else:
                mask_cache[path] = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        return mask_cache[path]

    point_in_mask_count = defaultdict(int)

    for image_id, pids in image_to_points.items():
        data = id_to_data[image_id]
        mask = load_mask(data['mask_path'])
        if mask is None:
            continue

        K = data['K']
        T_w2c = data['T_w2c']
        h, w = mask.shape

        P_W = np.stack([points_xyz[pid] for pid in pids], axis=0)
        uv, z = project_points_batch(P_W, K, T_w2c)

        valid = (
            (z > 0) &
            (uv[:, 0] >= 0) & (uv[:, 0] < w) &
            (uv[:, 1] >= 0) & (uv[:, 1] < h)
        )

        if not np.any(valid):
            continue

        uv = uv[valid].astype(np.int32)
        valid_pids = np.array(pids)[valid]

        mask_values = mask[uv[:, 1], uv[:, 0]] > 0

        for pid in valid_pids[mask_values]:
            point_in_mask_count[pid] += 1

    object_points_W = [
        points_xyz[pid]
        for pid, cnt in point_in_mask_count.items()
        if cnt >= min_observations
    ]

    print(f"Selected {len(object_points_W)} object 3D points.")
    return np.array(object_points_W, dtype=np.float64)


def get_median_pairwise_distance(points, num_pairs=20000):
    # (保持不变)
    n = points.shape[0]
    if n < 2:
        return 1e-6
    
    idx1 = np.random.randint(0, n, num_pairs)
    idx2 = np.random.randint(0, n, num_pairs)
    
    valid = idx1 != idx2
    p1 = points[idx1[valid]]
    p2 = points[idx2[valid]]
    
    if len(p1) == 0:
        return 1e-6
        
    dists = np.linalg.norm(p1 - p2, axis=1)
    return np.median(dists)


def calculate_initial_scales_from_median_dist(model_vertices, X_W_obj):
    # (保持不变)
    if model_vertices.size == 0 or X_W_obj.size == 0:
        raise ValueError("Model vertices or object 3D points are empty.")

    print("\n--- Calculating Initial Scale using Pairwise Distance Median ---")
    dist_M = get_median_pairwise_distance(model_vertices)
    dist_W = get_median_pairwise_distance(X_W_obj)
    
    scale_init = dist_W / (dist_M + 1e-6)
    scale_init = max(scale_init, 1e-6)
    
    s_xyz_init = np.array([scale_init, scale_init, scale_init])
    
    print(f"Model Median Distance: {dist_M:.6f}")
    print(f"World Median Distance: {dist_W:.6f}")
    print(f"Initial Uniform 3-Axis Scales (sx, sy, sz): {s_xyz_init}")
    
    return s_xyz_init


def _compute_reprojection_residuals_core(scales, rvec, t, all_data):
    """提取公共的 2D 重投影残差计算核心"""
    from scipy.spatial.transform import Rotation
    R = Rotation.from_rotvec(rvec).as_matrix()
    residuals = []

    Z_PENALTY_WEIGHT = 500.0
    EPS = 1e-4
    num_2d_points = 0
    
    for d in all_data:
        X = d["X_3d"]
        q = d["q_2d"]
        K = d["K"]
        T_w2c = d["T_w2c"]
        
        num_2d_points += len(q)
        
        X_scaled = X * scales[None, :]
        Xw = (R @ X_scaled.T).T + t
        Xw_h = np.hstack([Xw, np.ones((len(Xw), 1))])
        Xc = (T_w2c @ Xw_h.T).T[:, :3]

        Z = Xc[:, 2]
        res = np.zeros_like(q)
        mask_front = Z > EPS
        mask_behind = ~mask_front

        if np.any(mask_front):
            z_inv = 1.0 / Z[mask_front]
            u_proj = (K[0, 0] * Xc[mask_front, 0] * z_inv) + K[0, 2]
            v_proj = (K[1, 1] * Xc[mask_front, 1] * z_inv) + K[1, 2]
            res[mask_front, 0] = u_proj - q[mask_front, 0]
            res[mask_front, 1] = v_proj - q[mask_front, 1]

        if np.any(mask_behind):
            penalty = Z_PENALTY_WEIGHT * (EPS - Z[mask_behind])
            res[mask_behind, 0] = penalty
            res[mask_behind, 1] = penalty

        residuals.append(res.ravel())

    return np.concatenate(residuals), num_2d_points


def error_optimize_joint(params, all_data, s_init, lambda_3d):
    """Phase 1: 联合优化残差函数"""
    scales = params[0:3]
    rvec = params[3:6]
    t = params[6:9]
    
    res_2d, num_2d_points = _compute_reprojection_residuals_core(scales, rvec, t, all_data)
    
    # 3D 惩罚
    if lambda_3d > 0:
        scale_penalty_weight = lambda_3d * np.sqrt(max(num_2d_points, 1))
        res_3d_penalty = scale_penalty_weight * (scales - s_init)
        return np.concatenate([res_2d, res_3d_penalty])
    
    return res_2d


def error_optimize_scale_only(scales, fixed_rvec, fixed_t, all_data, s_init, lambda_3d_phase2):
    """Phase 2: 仅优化 Scale 残差函数"""
    res_2d, num_2d_points = _compute_reprojection_residuals_core(scales, fixed_rvec, fixed_t, all_data)
    
    if lambda_3d_phase2 > 0:
        scale_penalty_weight = lambda_3d_phase2 * np.sqrt(max(num_2d_points, 1))
        res_3d_penalty = scale_penalty_weight * (scales - s_init)
        return np.concatenate([res_2d, res_3d_penalty])
        
    return res_2d

def error_optimize_uniform_joint(params, all_data, s_init_scalar, lambda_3d):
    """联合优化：单轴 Scale (params[0]) + RT (params[1:7])"""
    s_uni = params[0]
    rvec = params[1:4]
    t = params[4:7]
    
    # 将单轴扩展为三轴传给核心函数
    scales_3d = np.array([s_uni, s_uni, s_uni])
    res_2d, num_2d_points = _compute_reprojection_residuals_core(scales_3d, rvec, t, all_data)
    
    if lambda_3d > 0:
        # 对单轴 scale 进行 3D 惩罚
        scale_penalty_weight = lambda_3d * np.sqrt(max(num_2d_points, 1))
        res_3d_penalty = scale_penalty_weight * (s_uni - s_init_scalar)
        return np.concatenate([res_2d, [res_3d_penalty]])
    return res_2d


def _best_pose_to_T_m2c(corr_data, scale=1.0, scaled_rotation=False):
    best_pose = corr_data.best_pose
    T_m2c = np.eye(4, dtype=np.float64)
    R_m2c = np.asarray(best_pose["R_m2c"], dtype=np.float64)
    t_m2c = np.asarray(best_pose["t_m2c"], dtype=np.float64).reshape(3)
    T_m2c[:3, :3] = R_m2c * scale if scaled_rotation else R_m2c
    T_m2c[:3, 3] = t_m2c * scale
    return T_m2c


def _camera_center_from_T_w2c(T_w2c):
    T_w2c = np.asarray(T_w2c, dtype=np.float64)
    return -T_w2c[:3, :3].T @ T_w2c[:3, 3]


def _model_camera_center_from_corr(corr_data):
    best_pose = corr_data.best_pose
    R_m2c = np.asarray(best_pose["R_m2c"], dtype=np.float64)
    t_m2c = np.asarray(best_pose["t_m2c"], dtype=np.float64).reshape(3)
    return -R_m2c.T @ t_m2c


def _robust_threshold(values, floor_value, cap_value):
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return cap_value
    med = float(np.median(finite))
    mad = float(np.median(np.abs(finite - med)))
    sigma = 1.4826 * mad
    threshold = med + 3.0 * sigma
    threshold = max(threshold, floor_value)
    return min(threshold, cap_value)


def _compute_robust_initial_scale(all_optimization_data, all_corresps_data):
    """Estimate uniform scale from all usable camera-baseline pairs."""
    centers_w = [_camera_center_from_T_w2c(d["T_w2c"]) for d in all_optimization_data]
    centers_m = [_model_camera_center_from_corr(c) for c in all_corresps_data]

    ratios = []
    pair_info = []
    for i in range(len(centers_w)):
        for j in range(i + 1, len(centers_w)):
            dist_w = float(np.linalg.norm(centers_w[i] - centers_w[j]))
            dist_m = float(np.linalg.norm(centers_m[i] - centers_m[j]))
            if dist_w < 0.01 or dist_m < 1e-5:
                continue
            ratio = dist_w / (dist_m + 1e-9)
            if np.isfinite(ratio) and ratio > 0:
                ratios.append(ratio)
                pair_info.append((i, j, dist_w, dist_m, ratio))

    if not ratios:
        print("Warning: no usable camera-baseline pairs, fallback to scale=1.0")
        return 1.0, {
            "pair_count": 0,
            "inlier_pair_count": 0,
            "median": 1.0,
            "min": None,
            "max": None,
        }

    ratios = np.asarray(ratios, dtype=np.float64)
    median = float(np.median(ratios))
    mad = float(np.median(np.abs(ratios - median)))
    sigma = max(1.4826 * mad, 1e-9)
    inlier_mask = np.abs(ratios - median) <= max(3.0 * sigma, 0.15 * median)
    inlier_ratios = ratios[inlier_mask]
    if inlier_ratios.size == 0:
        inlier_ratios = ratios

    scale = float(np.median(inlier_ratios))
    print("\n--- Initializing Scale via Robust Multi-frame Camera Baselines ---")
    print(
        f"Calculated s_init: {scale:.6f} "
        f"(pairs: {len(ratios)}, inliers: {len(inlier_ratios)}, "
        f"ratio min/median/max: {ratios.min():.6f}/{median:.6f}/{ratios.max():.6f})"
    )
    return scale, {
        "pair_count": int(len(ratios)),
        "inlier_pair_count": int(len(inlier_ratios)),
        "median": median,
        "min": float(ratios.min()),
        "max": float(ratios.max()),
        "scale": scale,
        "pairs": [
            {
                "i": int(i),
                "j": int(j),
                "dist_w": float(dist_w),
                "dist_m": float(dist_m),
                "ratio": float(ratio),
                "inlier": bool(inlier_mask[k]),
            }
            for k, (i, j, dist_w, dist_m, ratio) in enumerate(pair_info)
        ],
    }


def _compose_frame_m2w(opt_data, corr_data, scale, scaled_rotation=False):
    T_m2c = _best_pose_to_T_m2c(corr_data, scale=scale, scaled_rotation=scaled_rotation)
    return np.linalg.inv(np.asarray(opt_data["T_w2c"], dtype=np.float64)) @ T_m2c


def _reference_pose_from_indices(frame_Ts, indices):
    indices = list(indices)
    if not indices:
        indices = list(range(len(frame_Ts)))
    translations = np.stack([frame_Ts[i][:3, 3] for i in indices], axis=0)
    rotations = np.stack([frame_Ts[i][:3, :3] for i in indices], axis=0)
    t_ref = np.median(translations, axis=0)
    R_ref = Rotation.from_matrix(rotations).mean().as_matrix()
    return R_ref, t_ref


def _projection_error_for_T(data, T_M2W):
    X = data["X_3d"]
    q = data["q_2d"]
    if len(X) == 0:
        return float("inf"), float("inf")
    X_h = np.hstack([X, np.ones((len(X), 1), dtype=X.dtype)])
    X_c = (data["T_w2c"] @ T_M2W @ X_h.T).T[:, :3]
    valid = X_c[:, 2] > 1e-6
    if not np.any(valid):
        return float("inf"), float("inf")
    X_c = X_c[valid]
    q = q[valid]
    uvw = (data["K"] @ X_c.T).T
    uv = uvw[:, :2] / uvw[:, 2:3]
    err = np.linalg.norm(uv - q, axis=1)
    if err.size == 0:
        return float("inf"), float("inf")
    return float(np.median(err)), float(np.mean(err))


def _make_project_T(R_m2w, t_m2w, scale):
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R_m2w @ np.diag([scale, scale, scale])
    T[:3, 3] = t_m2w
    return T


def _measure_frame_consistency(all_optimization_data, all_corresps_data, frame_Ts, ref_indices, scale):
    R_ref, t_ref = _reference_pose_from_indices(frame_Ts, ref_indices)
    ref_rot = Rotation.from_matrix(R_ref)
    T_ref_project = _make_project_T(R_ref, t_ref, scale)
    records = []

    for idx, (opt_data, corr_data, T_i) in enumerate(zip(all_optimization_data, all_corresps_data, frame_Ts)):
        rot_err = (ref_rot.inv() * Rotation.from_matrix(T_i[:3, :3])).magnitude() * 180.0 / np.pi
        trans_err = float(np.linalg.norm(T_i[:3, 3] - t_ref))
        ref_reproj_median, ref_reproj_mean = _projection_error_for_T(opt_data, T_ref_project)
        self_project_T = _compose_frame_m2w(opt_data, corr_data, scale, scaled_rotation=True)
        self_reproj_median, self_reproj_mean = _projection_error_for_T(opt_data, self_project_T)
        records.append(
            {
                "index": idx,
                "frame_name": opt_data["frame_name"],
                "num_corr": int(len(opt_data["q_2d"])),
                "rot_err_deg": float(rot_err),
                "trans_err": trans_err,
                "ref_reproj_median_px": ref_reproj_median,
                "ref_reproj_mean_px": ref_reproj_mean,
                "self_reproj_median_px": self_reproj_median,
                "self_reproj_mean_px": self_reproj_mean,
            }
        )
    return records, R_ref, t_ref


def _select_consistent_frames(records, min_keep):
    rot_values = [r["rot_err_deg"] for r in records]
    trans_values = [r["trans_err"] for r in records]
    reproj_values = [r["ref_reproj_median_px"] for r in records]
    rot_thr = _robust_threshold(rot_values, floor_value=18.0, cap_value=45.0)
    trans_thr = _robust_threshold(trans_values, floor_value=0.04, cap_value=0.18)
    reproj_thr = _robust_threshold(reproj_values, floor_value=30.0, cap_value=90.0)
    min_corr = 25

    kept = []
    for r in records:
        reasons = []
        if r["num_corr"] < min_corr:
            reasons.append("few_corr")
        if r["rot_err_deg"] > rot_thr:
            reasons.append("rot")
        if r["trans_err"] > trans_thr:
            reasons.append("trans")
        if r["ref_reproj_median_px"] > reproj_thr:
            reasons.append("reproj")
        r["drop_reasons"] = reasons
        r["keep"] = len(reasons) == 0
        if r["keep"]:
            kept.append(r["index"])

    if len(kept) < min_keep:
        def score(r):
            corr_penalty = max(0.0, (min_corr - r["num_corr"]) / float(min_corr))
            return (
                r["rot_err_deg"] / max(rot_thr, 1e-6)
                + r["trans_err"] / max(trans_thr, 1e-6)
                + r["ref_reproj_median_px"] / max(reproj_thr, 1e-6)
                + corr_penalty
            )

        ranked = sorted(records, key=score)
        keep_set = set(r["index"] for r in ranked[:min_keep])
        kept = sorted(keep_set)
        for r in records:
            if r["index"] in keep_set:
                r["keep"] = True
                r["drop_reasons"] = []
            else:
                r["keep"] = False
                if not r["drop_reasons"]:
                    r["drop_reasons"] = ["ranked_out"]

    thresholds = {
        "rot_err_deg": float(rot_thr),
        "trans_err": float(trans_thr),
        "ref_reproj_median_px": float(reproj_thr),
        "min_corr": int(min_corr),
    }
    return kept, thresholds


def _write_pose_consistency_outputs(
    output_dir,
    records,
    thresholds,
    scale,
    scale_stats,
    all_optimization_data,
    all_corresps_data,
    model_vertices,
    model_faces,
    R_ref,
    t_ref,
):
    diag_dir = os.path.join(output_dir, "pose_consistency")
    os.makedirs(diag_dir, exist_ok=True)

    payload = {
        "scale": float(scale),
        "scale_stats": scale_stats,
        "thresholds": thresholds,
        "kept_frames": [r["frame_name"] for r in records if r.get("keep")],
        "dropped_frames": [
            {"frame_name": r["frame_name"], "reasons": r.get("drop_reasons", [])}
            for r in records if not r.get("keep")
        ],
        "records": records,
    }
    with open(os.path.join(diag_dir, "pose_consistency.json"), "w") as f:
        json.dump(payload, f, indent=2)

    csv_path = os.path.join(diag_dir, "pose_consistency.csv")
    with open(csv_path, "w") as f:
        f.write(
            "index,frame_name,keep,num_corr,rot_err_deg,trans_err,"
            "ref_reproj_median_px,self_reproj_median_px,drop_reasons\n"
        )
        for r in records:
            f.write(
                f"{r['index']},{r['frame_name']},{int(bool(r.get('keep')))},"
                f"{r['num_corr']},{r['rot_err_deg']:.6f},{r['trans_err']:.6f},"
                f"{r['ref_reproj_median_px']:.6f},{r['self_reproj_median_px']:.6f},"
                f"{'|'.join(r.get('drop_reasons', []))}\n"
            )

    row_h = 28
    width = 1250
    height = max(240, 120 + row_h * len(records))
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    cv2.putText(canvas, f"Pose consistency diagnostics  scale={scale:.6f}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2)
    cv2.putText(
        canvas,
        f"thresholds: rot<={thresholds['rot_err_deg']:.1f}deg  trans<={thresholds['trans_err']:.3f}  ref_reproj<={thresholds['ref_reproj_median_px']:.1f}px",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (40, 40, 40),
        1,
    )
    header = "idx keep frame              corr   rot_deg   trans_m   ref_px   self_px   reasons"
    cv2.putText(canvas, header, (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1)
    for row, r in enumerate(records):
        y = 135 + row * row_h
        color = (20, 130, 20) if r.get("keep") else (20, 20, 220)
        text = (
            f"{r['index']:03d}  {int(bool(r.get('keep')))}    {r['frame_name']:<16} "
            f"{r['num_corr']:4d}  {r['rot_err_deg']:8.2f}  {r['trans_err']:8.4f} "
            f"{r['ref_reproj_median_px']:7.2f}  {r['self_reproj_median_px']:7.2f} "
            f"{'|'.join(r.get('drop_reasons', []))}"
        )
        cv2.putText(canvas, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1)
    cv2.imwrite(os.path.join(diag_dir, "pose_consistency_summary.png"), canvas)

    T_ref_project = _make_project_T(R_ref, t_ref, scale)
    for opt_data, corr_data in zip(all_optimization_data, all_corresps_data):
        image_name = f"ref_{os.path.basename(opt_data['image_path'])}"
        visualize_projection(
            image_path=opt_data["image_path"],
            output_dir=output_dir,
            K=opt_data["K"],
            T_W2C=opt_data["T_w2c"],
            T_M2W=T_ref_project,
            model_vertices=model_vertices,
            model_faces=model_faces,
            image_name=image_name,
            root_name="pose_consistency_ref",
        )
        T_self_project = _compose_frame_m2w(opt_data, corr_data, scale, scaled_rotation=True)
        visualize_projection(
            image_path=opt_data["image_path"],
            output_dir=output_dir,
            K=opt_data["K"],
            T_W2C=opt_data["T_w2c"],
            T_M2W=T_self_project,
            model_vertices=model_vertices,
            model_faces=model_faces,
            image_name=f"self_{os.path.basename(opt_data['image_path'])}",
            root_name="pose_consistency_self",
        )


def _diagnose_and_filter_pose_frames(
    all_optimization_data,
    all_corresps_data,
    model_vertices,
    model_faces,
    output_dir,
    scale,
    scale_stats,
):
    n = len(all_optimization_data)
    if n <= 2:
        print("Warning: not enough frames for pose-consistency filtering; keeping all frames.")
        return list(range(n))

    frame_Ts = [
        _compose_frame_m2w(opt_data, corr_data, scale, scaled_rotation=False)
        for opt_data, corr_data in zip(all_optimization_data, all_corresps_data)
    ]

    records, _R_ref, _t_ref = _measure_frame_consistency(
        all_optimization_data,
        all_corresps_data,
        frame_Ts,
        ref_indices=range(n),
        scale=scale,
    )
    min_keep = min(n, max(3, min(8, n // 2)))
    provisional_kept, _thresholds = _select_consistent_frames(records, min_keep=min_keep)

    records, R_ref, t_ref = _measure_frame_consistency(
        all_optimization_data,
        all_corresps_data,
        frame_Ts,
        ref_indices=provisional_kept,
        scale=scale,
    )
    kept, thresholds = _select_consistent_frames(records, min_keep=min_keep)
    _write_pose_consistency_outputs(
        output_dir=output_dir,
        records=records,
        thresholds=thresholds,
        scale=scale,
        scale_stats=scale_stats,
        all_optimization_data=all_optimization_data,
        all_corresps_data=all_corresps_data,
        model_vertices=model_vertices,
        model_faces=model_faces,
        R_ref=R_ref,
        t_ref=t_ref,
    )

    dropped = [r for r in records if not r.get("keep")]
    if dropped:
        print("\n--- Pose Consistency Filtering ---")
        print(f"Kept {len(kept)}/{n} frames. Diagnostics written to: {os.path.join(output_dir, 'pose_consistency')}")
        for r in dropped:
            print(
                f"Drop {r['frame_name']}: reasons={r.get('drop_reasons', [])}, "
                f"rot={r['rot_err_deg']:.2f}deg, trans={r['trans_err']:.4f}, "
                f"ref_reproj={r['ref_reproj_median_px']:.2f}px"
            )
    else:
        print(f"Pose consistency check kept all {n} frames. Diagnostics written to: {os.path.join(output_dir, 'pose_consistency')}")

    return kept

def optimize_global_pose(
    all_optimization_data,
    all_corresps_data,
    model_vertices,
    model_faces,
    output_dir,
    colmap_dir,
    mask_dir,
    init_frame_id=0,
    lambda_3d_strong=500.0, # 用于 Phase 1，极强约束保持等比例
    lambda_3d_weak=10.0      # 用于后续微调
):
    """两阶段优化：先联合优化，再固定 RT 优化 Scale"""

    if len(all_optimization_data) != len(all_corresps_data):
        raise ValueError("Optimization data and correspondence data length mismatch.")
    if len(all_optimization_data) == 0:
        raise ValueError("No valid frames for global pose optimization.")

    # --- 1. Robust scale init + pose-consistency diagnostics/filtering ---
    s_init_scalar, scale_stats = _compute_robust_initial_scale(all_optimization_data, all_corresps_data)
    kept_indices = _diagnose_and_filter_pose_frames(
        all_optimization_data=all_optimization_data,
        all_corresps_data=all_corresps_data,
        model_vertices=model_vertices,
        model_faces=model_faces,
        output_dir=output_dir,
        scale=s_init_scalar,
        scale_stats=scale_stats,
    )
    if len(kept_indices) < len(all_optimization_data):
        all_optimization_data[:] = [all_optimization_data[i] for i in kept_indices]
        all_corresps_data[:] = [all_corresps_data[i] for i in kept_indices]
        init_frame_id = 0
        s_init_scalar, scale_stats = _compute_robust_initial_scale(all_optimization_data, all_corresps_data)
    else:
        init_frame_id = min(init_frame_id, len(all_optimization_data) - 1)

    s_init = np.array([s_init_scalar, s_init_scalar, s_init_scalar])
    
    best_pose = all_corresps_data[init_frame_id].best_pose
    T_m2c_scaled = np.eye(4)
    T_m2c_scaled[:3, :3] = best_pose["R_m2c"]
    T_m2c_scaled[:3, 3] = best_pose["t_m2c"].ravel() * s_init_scalar # 缩放平移

    T_w2c_init = all_optimization_data[init_frame_id]["T_w2c"]
    T_m2w_init = np.linalg.inv(T_w2c_init) @ T_m2c_scaled

    R_init = T_m2w_init[:3, :3]
    t_init = T_m2w_init[:3, 3]
    rvec_init = Rotation.from_matrix(R_init).as_rotvec()
    
    # ==========================================================
    # STAGE 1: 锁定 Scale, 只优化 R 和 T
    # ==========================================================
    print("\n=== Stage 1: Fix Scale, Optimize RT Only ===")
    
    def error_rt_only(rt_params, fixed_s, all_data):
        r_val = rt_params[0:3]
        t_val = rt_params[3:6]
        # 调用核心残差函数，但传入固定的 scales
        res_2d, _ = _compute_reprojection_residuals_core(fixed_s, r_val, t_val, all_data)
        return res_2d

    date_str = datetime.now().strftime("%Y-%m-%d")
    log_filename = f"4stage_least_squares_log_{date_str}.txt"
    log_filepath = os.path.join(output_dir, log_filename)
    from contextlib import redirect_stdout
    from scipy.optimize import least_squares
    with open(log_filepath, 'w') as f:
        with redirect_stdout(f):
            rt_init = np.concatenate([rvec_init, t_init])
            res_stage1 = least_squares(
                fun=error_rt_only,
                x0=rt_init,
                args=(s_init, all_optimization_data),
                method='trf',
                verbose=2,
                loss='soft_l1'
            )
            
            rvec_after_s1 = res_stage1.x[0:3]
            t_after_s1 = res_stage1.x[3:6]
            print(f"Stage 1 Finished. RT adjusted while keeping Scale fixed at {s_init}")
            # ===== Stage1 Visualization =====
            from script.vis_util import visualize_projection

            R_s1 = Rotation.from_rotvec(rvec_after_s1).as_matrix()

            T_stage1 = np.eye(4, dtype=np.float32)
            T_stage1[:3, :3] = R_s1 @ np.diag(s_init)
            T_stage1[:3, 3] = t_after_s1

            # ==========================================================
            # STAGE 3: 联合优化 (三轴 Scale + RT)
            # ==========================================================
            print("\n=== Stage 3: Joint Three-axis Scale (3-axis) + RT ===")
            # 用 Stage 2 的结果初始化，三轴初值设为一样的
            # 2. 将旋转矩阵 R_s1 转换为旋转向量 (3维)，否则维度对不上
            r_vec_init = Rotation.from_matrix(R_s1).as_rotvec()
            # 3. 确保 t 也是平的一维向量 (3维)
            t_vec_init = t_after_s1.flatten()
            # 4. 现在拼接，总长度应该是 3 (scale) + 3 (rotation) + 3 (translation) = 9
            params_s3_init = np.concatenate([s_init, r_vec_init, t_vec_init])
            res_s3 = least_squares(
                fun=error_optimize_joint, 
                x0=params_s3_init, 
                args=(all_optimization_data, s_init, lambda_3d_strong), 
                method='trf',
                bounds=(np.array([1e-4]*3 + [-np.inf]*6), np.array([np.inf]*9)),
                verbose=2, loss='soft_l1'
            )
            s_s3, r_s3, t_s3 = res_s3.x[0:3], res_s3.x[3:6], res_s3.x[6:9]
            R_s3 = Rotation.from_rotvec(r_s3).as_matrix()
            
            T_stage3 = np.eye(4, dtype=np.float32)
            T_stage3[:3, :3] = R_s3 @ np.diag(s_s3)
            T_stage3[:3, 3] = t_s3
            
    final_s = s_s3
    final_t = t_s3
    final_R = R_s3
    print(f"Final Optimized Scales: {s_s3}")
    T_final = np.eye(4, dtype=np.float32)
    T_final[:3, :3] = final_R @ np.diag(final_s)
    T_final[:3, 3] = final_t

    return T_final, res_s3, final_s

def get_mask_info(mask):
    """
    预计算 Mask 的距离变换和梯度，用于快速寻找边缘和法向
    """
    # 距离变换：值代表像素到边缘的距离
    dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    # 梯度：指向物体内部（数值变大的方向）
    dy, dx = np.gradient(dist_transform)
    # 归一化梯度作为 Mask 边缘的法线 (指向物体外部)
    grad_norm = np.sqrt(dx**2 + dy**2) + 1e-8
    mask_normals_x = -dx / grad_norm
    mask_normals_y = -dy / grad_norm
    
    return dist_transform, mask_normals_x, mask_normals_y

def preprocess_2d_contours(all_optimization_data, num_samples=20):
    """
    遍历数据，读取 mask，提取轮廓点位置和法向（梯度）
    """
    print(f"--- Preprocessing 2D Contours (samples={num_samples}) ---")
    for data in all_optimization_data:
        # 1. 路径转换 logic: .../rgb/0000.jpg -> .../masks/0000.png
        image_path = data['image_path']
        mask_path = image_path.replace("/rgb/", "/masks/").replace(".jpg", ".png")
        
        if not os.path.exists(mask_path):
            print(f"Warning: Mask not found {mask_path}")
            data['edge_2d'] = None
            continue
            
        # 2. 读取 mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # 3. 提取轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if len(contours) == 0:
            data['edge_2d'] = None
            continue
            
        # 合并最长轮廓（假设只有一个主要物体）
        c = max(contours, key=cv2.contourArea)
        pts = c.squeeze() # (N, 2)
        
        # 4. 均匀采样
        if len(pts) > num_samples:
            indices = np.linspace(0, len(pts)-1, num_samples, dtype=int)
            sampled_pts = pts[indices]
        else:
            sampled_pts = pts
            
        # 5. 计算梯度作为 2D 法向
        # 使用 Distance Transform 的梯度更平滑，指向物体外部
        dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        # 注意：dist map 内部值大，边缘小。梯度指向内部。我们需要指向外部，所以取反。
        dy, dx = np.gradient(dist_transform) 
        
        edge_grads = []
        valid_pts = []
        
        for pt in sampled_pts:
            gx = -dx[pt[1], pt[0]] # 取反指向外
            gy = -dy[pt[1], pt[0]]
            norm = np.sqrt(gx**2 + gy**2)
            if norm > 1e-6:
                edge_grads.append([gx/norm, gy/norm])
                valid_pts.append(pt)
        
        data['edge_2d'] = np.array(valid_pts, dtype=np.float32)
        data['edge_grad'] = np.array(edge_grads, dtype=np.float32)

def compute_vertex_normals(vertices, faces):
    """
    使用纯 Numpy 快速计算顶点法线 (加权平均面法线)
    """
    # 1. 计算面法线
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    # 叉乘得到面法线 (未归一化，模长即面积的两倍，正好作为权重)
    face_normals = np.cross(v1 - v0, v2 - v0)
    
    # 2. 累加到顶点
    vertex_normals = np.zeros_like(vertices)
    # 使用 add.at 进行原位累加
    np.add.at(vertex_normals, faces[:, 0], face_normals)
    np.add.at(vertex_normals, faces[:, 1], face_normals)
    np.add.at(vertex_normals, faces[:, 2], face_normals)
    
    # 3. 归一化
    norms = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
    vertex_normals = vertex_normals / np.maximum(norms, 1e-8)
    return vertex_normals

def find_silhouette_vertices(V_cam, faces):
    """
    基于几何拓扑寻找轮廓顶点：连接了“前向面”和“后向面”的边即为轮廓边
    """
    v0, v1, v2 = V_cam[faces[:, 0]], V_cam[faces[:, 1]], V_cam[faces[:, 2]]
    normals = np.cross(v1 - v0, v2 - v0)
    
    # 视线向量（从面指向相机，相机在原点）
    face_centers = (v0 + v1 + v2) / 3.0
    view_dirs = -face_centers 
    is_front_facing = np.sum(normals * view_dirs, axis=1) > 0

    # 提取所有边并排序
    edges = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    edges.sort(axis=1)
    edge_front_facing = np.repeat(is_front_facing, 3)

    # 统计每条边连接的前向面数量
    unique_edges, inverse_indices = np.unique(edges, axis=0, return_inverse=True)
    front_face_counts = np.bincount(inverse_indices, weights=edge_front_facing.astype(int))
    total_face_counts = np.bincount(inverse_indices)

    # 轮廓边定义：连接一个前向面和一个后向面，或者是边界上的前向面边
    is_silhouette = (total_face_counts == 2) & (front_face_counts == 1)
    is_boundary = (total_face_counts == 1) & (front_face_counts == 1)
    
    silhouette_v_ids = np.unique(unique_edges[is_silhouette | is_boundary].flatten())
    return silhouette_v_ids

def update_contour_correspondences(V_current, model_faces, T_M2W, all_optimization_data, output_dir, outer_iter):
    """
    在当前模型 V_current 下，为每一帧的 2D 轮廓点找到 Mesh 上对应的 3D 顶点索引。
    结果存储在 data['contour_v_ids'] 中。
    """
    # --- 1. 实时计算顶点法线 ---
    # 因为 V_current 是变形后的，法线变了
    N_3d = compute_vertex_normals(V_current, model_faces)
    # V_current 是世界坐标系下的 (或者是初始对齐后的)
    
    for data in all_optimization_data:
       # 获取预处理好的 2D 轮廓点集和法向
        target_pts = data.get('edge_2d')
        target_grads = data.get('edge_grad')
        
        # 如果当前帧没有提取到轮廓，则跳过
        if target_pts is None or len(target_pts) == 0:
            continue
            
        h, w = cv2.imread(data['image_path'], cv2.IMREAD_GRAYSCALE).shape[:2]
        
        K = data['K']
        T_w2c = data['T_w2c']
        T_total = T_w2c @ T_M2W
        
        # 1. & 2. 渲染当前 Mask 并建立边缘树 (用于过滤遮挡)
        from script.vis_util import render_mesh_mask
        current_rendered_mask = render_mesh_mask((h, w), K, T_total, V_current, model_faces)
        contours_rendered, _ = cv2.findContours(current_rendered_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if len(contours_rendered) == 0: continue
        rendered_edge_pts = max(contours_rendered, key=cv2.contourArea).squeeze()
        rendered_edge_tree = cKDTree(rendered_edge_pts)

        # 3. 提取 3D 几何轮廓点并过滤
        V_h = np.hstack([V_current, np.ones((len(V_current), 1))])
        V_cam = (T_total @ V_h.T).T[:, :3]
        potential_v_ids = find_silhouette_vertices(V_cam, model_faces)
        
        v_ids_cam_valid = potential_v_ids[V_cam[potential_v_ids, 2] > 0.1]
        pts_cam = V_cam[v_ids_cam_valid]
        proj_h = (K @ pts_cam.T).T
        proj_2d = proj_h[:, :2] / proj_h[:, 2:3]
        
        dists, _ = rendered_edge_tree.query(proj_2d, k=1)
        is_outer = dists < 2.0 
        
        final_v_ids = v_ids_cam_valid[is_outer]
        final_proj_2d = proj_2d[is_outer]
        
        # 获取这批保留下来的 3D 轮廓点的 2D 投影法向
        N_cam = (T_total[:3, :3] @ N_3d[final_v_ids].T).T
        n_2d = N_cam[:, :2]
        n_2d /= (np.linalg.norm(n_2d, axis=1, keepdims=True) + 1e-8)

        # --- [核心修改：全新的 2D-3D 匹配逻辑] ---
        matched_v_ids, matched_targets, matched_grads = [], [], []
        
        # 设置搜索超参数
        line_dist_thresh = 3.0    # 距离射线多远认为是“交点” (像素)
        search_radius = 15.0      # 在交点 p0 附近采样的半径 p` (像素)
        max_dist_to_p0 = 60.0     # 限制 p0 不能跑到画面外太远
        normal_dot_thresh = 0.5   # 最终采样的 p` 与 3D 投影法向的夹角阈值 (点积 > 0.5 即 < 60度)

        for i in range(len(final_proj_2d)):
            pt = final_proj_2d[i]
            n = n_2d[i]
            
            # --- 步骤 1: 寻找法线射线上的最近交点 p0 ---
            # 射线方程: P = pt + t * n
            # 点 T 到射线所在直线的垂直距离 d = |(T - pt) x n|
            vecs = target_pts - pt
            dist_to_line = np.abs(vecs[:, 0] * n[1] - vecs[:, 1] * n[0])
            dist_to_pt = np.linalg.norm(vecs, axis=1) # 实际空间距离 (即 |t|)
            
            # 找到在法线所在直线附近 (误差小于阈值)，且没有离得太远的点
            valid_line_mask = (dist_to_line < line_dist_thresh) & (dist_to_pt < max_dist_to_p0)
            if not np.any(valid_line_mask):
                continue # 没有找到交点
                
            # 从候选点中找出离 pt 最近的那个，作为射线交点 p0
            valid_indices = np.where(valid_line_mask)[0]
            # np.argmin 返回局部索引，需转换回全局索引
            p0_idx = valid_indices[np.argmin(dist_to_pt[valid_indices])]
            p0 = target_pts[p0_idx]
            
            # --- 步骤 2: 在 p0 附近采样轮廓点 p` ---
            # 重新计算所有目标点到 p0 的距离
            dist_to_p0 = np.linalg.norm(target_pts - p0, axis=1)
            nearby_mask = dist_to_p0 < search_radius
            nearby_indices = np.where(nearby_mask)[0]
            
            # --- 步骤 3: 在 p` 中寻找法向最一致的点 ---
            nearby_grads = target_grads[nearby_indices]
            # 点积：计算两个法向的相似度
            dots = np.sum(nearby_grads * n, axis=1)
            
            # 找到最一致的点 (点积最大)
            best_local_idx = np.argmax(dots)
            best_dot = dots[best_local_idx]
            
            # 校验法向一致性
            if best_dot > normal_dot_thresh:
                best_idx = nearby_indices[best_local_idx]
                matched_v_ids.append(final_v_ids[i])
                matched_targets.append(target_pts[best_idx])
                matched_grads.append(target_grads[best_idx])

        # 记录匹配结果
        data['contour_v_ids'] = np.array(matched_v_ids)
        data['matched_edge_2d'] = np.array(matched_targets)
        data['matched_edge_grad'] = np.array(matched_grads)
        
        # # 调用可视化函数 (记得传 final_v_ids)
        # save_dir = output_dir + "/newc"
        # visualize_contour_matching(
        #     data, final_proj_2d, final_v_ids, save_dir, outer_iter
        # )
        
    
def refine_model_with_deformation_graph(
    model_vertices,
    model_faces,
    T_M2W,
    all_optimization_data,
    output_dir,
    num_nodes=20,
    knn_node=4,
    knn_vertex=4,
    lambda_reg=1e5
):
    # --- 初始化统计字典 ---
    stats = defaultdict(float)
    total_start = time.time()

    # --- 0. 预处理 2D 轮廓 ---
    t_start = time.time()
    # 只需执行一次
    preprocess_2d_contours(all_optimization_data, num_samples=200)
    stats["0_Preprocess_2D"] = time.time() - t_start
    
    V0 = model_vertices.copy()
    
    # --- 1. 逻辑焊接预处理 ---
    t_start = time.time()
    # 找到物理位置重合的顶点。round(6) 用于处理微小的浮点数精度问题
    _, inverse_mapping = np.unique(np.round(V0, 6), axis=0, return_inverse=True)
    stats["1_Vertex_Welding"] = time.time() - t_start
    
    # --- 2. 采样与图构建 ---
    t_start = time.time()
    node_indices = farthest_point_sampling(V0, num_nodes)
    node_pos = V0[node_indices]   # (K,3)
    
    # --- 2. 构建 graph ---
    graph_edges = build_knn_graph(node_pos, knn_node)
    vertex_nodes, vertex_weights = compute_vertex_node_weights(
        V0, node_pos, k=knn_vertex
    )
    graph_edges = np.array(graph_edges)
    stats["2_Graph_Building"] = time.time() - t_start
    
    # --- [修改：计算统一权重] ---
    # 1. 提取不重复的顶点位置
    # --- 3. 权重计算与广播 ---
    t_start = time.time()
    unique_v_indices = np.unique(inverse_mapping, return_index=True)[1]
    unique_V0 = V0[unique_v_indices]
    
    # 2. 只对唯一的位置计算 KNN 权重
    unique_vertex_nodes, unique_vertex_weights = compute_vertex_node_weights(
        unique_V0, node_pos, k=knn_vertex
    )
    
    # 3. 将权重广播回所有原始顶点索引
    # 这样，即使 OBJ 拆分了多个顶点，只要位置相同，形变位移就绝对一致
    vertex_nodes = unique_vertex_nodes[inverse_mapping]
    vertex_weights = unique_vertex_weights[inverse_mapping]

    stats["3_Weight_Broadcasting"] = time.time() - t_start
    # --- 4. 迭代优化 (ICP-style Loop) ---
    # 定义外层循环次数，通常 2-3 次即可收敛
    outer_iterations = 3
    
    K = node_pos.shape[0]
    # 初始参数 x0 (平移为0, 旋转为0即单位阵)
    current_x = np.zeros(6 * K, dtype=np.float64)
    
    for outer_iter in range(outer_iterations):
        iter_start = time.time()
        print(f"=== Outer Iteration {outer_iter + 1}/{outer_iterations} ===")
        
        # A. 根据当前的 deformation 参数，计算当前的 Mesh 形态
        t_sub = time.time()
        R_list, t_list = unpack_params(current_x)
        V_current = apply_deformation_graph(
            V0, node_pos, R_list, t_list, vertex_nodes, vertex_weights
        )
        
        stats["4A_Apply_Deformation"] += time.time() - t_sub
        
        # B. 更新 3D-2D 对应关系 (核心瓶颈)
        t_sub = time.time()
        update_contour_correspondences(V_current, model_faces, T_M2W, all_optimization_data, output_dir, outer_iter)
        # visualize_per_frame_alignment(
        #         V_current, 
        #         model_faces, 
        #         all_optimization_data, 
        #         outer_iter, 
        #         output_dir, 
        #         T_M2W
        #     )
        batched_data = prepare_batched_data_for_solver(all_optimization_data,T_M2W)
        stats["4B_Correspondence_Search"] += time.time() - t_sub
        
        # D. 最小二乘求解 (Levenberg-Marquardt / TRF)
        t_sub = time.time()
        # --- [新增可视化] ---
        # visualize_contour_matches(V_current, T_M2W, all_optimization_data, outer_iter, output_dir)
        
        # C. 最小二乘优化
        #    注意：将 current_x 作为本次优化的初值
        result = least_squares(
            deformation_graph_residual,
            current_x,
            verbose=1,
            method="trf",
            x_scale="jac",
            # 可以在后期迭代减少 max_nfev 以加快速度
            max_nfev=20 if outer_iter > 0 else 50, 
            args=(
                V0, node_pos, vertex_nodes, vertex_weights,
                graph_edges, batched_data, lambda_reg
            )
        )
        current_x = result.x
        stats["4D_Least_Squares_Solve"] += time.time() - t_sub
        
        print(f"Iteration {outer_iter+1} done in {time.time() - iter_start:.4f}s")
    # --- 5. 应用最终形变 ---
    # --- 5. 应用最终形变 ---
    t_start = time.time()
    R_list, t_list = unpack_params(current_x)
    V_refined = apply_deformation_graph(
        V0, node_pos, R_list, t_list,
        vertex_nodes, vertex_weights
    )
    stats["5_Final_Apply"] = time.time() - t_start
    
    total_duration = time.time() - total_start

    # --- 打印性能报告 ---
    print("\n" + "="*50)
    print(f"{'STAGE':<30} | {'TIME (s)':<10} | {'%':<5}")
    print("-" * 50)
    for stage, duration in sorted(stats.items()):
        percentage = (duration / total_duration) * 100
        print(f"{stage:<30} | {duration:>10.4f} | {percentage:>5.1f}%")
    print("-" * 50)
    print(f"{'TOTAL DURATION':<30} | {total_duration:>10.4f} | 100%")
    print("="*50 + "\n")
    
    # --- [新增：最终结果可视化] ---
    print(f"--- Generating Final Visualizations to {output_dir} ---")
    # 这里的 T_M2W 是传入函数的初始对齐矩阵
    # 如果优化过程中没有修改 T_M2W，则直接使用
    # for data in all_optimization_data:
    #     # 为了区分中间过程，我们将最终结果存在 'final_results' 目录下
    #     img_name = os.path.basename(data['image_path'])
    #     visualize_projection(
    #         image_path=data["image_path"],
    #         output_dir=output_dir,
    #         K=data["K"],
    #         T_W2C=data["T_w2c"],
    #         T_M2W=T_M2W, # 最终位姿
    #         model_vertices=V_refined, # 使用优化后的顶点
    #         model_faces=model_faces,
    #         image_name=f"refined_{img_name}",
    #         root_name="final_results"
    #     )
    return V_refined

# 在外层迭代中，完成 update_contour_correspondences 后提取数据
def prepare_batched_data_for_solver(all_optimization_data, T_M2W):
    batch = {'v_ids': [], 'T_total': [], 'K': [], 'targets': [], 'grads': []}
    for data in all_optimization_data:
        v_ids = data.get('contour_v_ids')
        if v_ids is None or len(v_ids) == 0: continue
        
        num = len(v_ids)
        batch['v_ids'].append(v_ids)
        batch['targets'].append(data['matched_edge_2d'])
        batch['grads'].append(data['matched_edge_grad'])
        batch['T_total'].append(np.tile(data['T_w2c'] @ T_M2W, (num, 1, 1)))
        batch['K'].append(np.tile(data['K'], (num, 1, 1)))
        
    if not batch['v_ids']: return None
    return {k: np.concatenate(v, axis=0) for k, v in batch.items()}

def visualize_contour_matching(data, proj_2d, v_ids, output_dir, iter_idx):
    """
    可视化当前帧的轮廓匹配情况
    """
    img = cv2.imread(data['image_path'])
    if img is None: return
    
    # 画出所有 3D 轮廓顶点的投影 (蓝色)
    for p in proj_2d:
        cv2.circle(img, tuple(p.astype(int)), 2, (255, 0, 0), -1)
        
    # 画出匹配关系 (红色连线)
    if data.get('contour_v_ids') is not None:
        matched_v_ids = data['contour_v_ids'].tolist()
        v_ids_list = v_ids.tolist()
        
        for i, v_id in enumerate(matched_v_ids):
            # 找到对应的初始投影点索引
            p_idx = v_ids_list.index(v_id)
            p_m = proj_2d[p_idx].astype(int)
            p_t = data['matched_edge_2d'][i].astype(int)
            
            # 画连线
            cv2.line(img, tuple(p_m), tuple(p_t), (0, 0, 255), 1)
            # 画匹配成功的图像目标点 (绿色)
            cv2.circle(img, tuple(p_t), 2, (0, 255, 0), -1)
            
    save_path = os.path.join(output_dir, f"match_iter_{iter_idx}_{os.path.basename(data['image_path'])}")
    os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(save_path, img)
def deformation_graph_residual(
    x,
    V0,
    node_pos,
    vertex_nodes,
    vertex_weights,
    graph_edges,
    batched_data,
    lambda_reg,
):
    R_list, t_list = unpack_params(x)
    residuals = []
    K_nodes = node_pos.shape[0]

    # 为了批量计算变形，可以先只提取出本轮所有需要的顶点
    # 但由于 Vertex Deformation 依赖 KNN 权重，按帧处理逻辑比较清晰
    
    # 1. 批量变形所有涉及的 3D 顶点 (一次性完成!)
    v_ids = batched_data['v_ids']
    V_sub = V0[v_ids]
    vw_sub = vertex_weights[v_ids]
    vn_sub = vertex_nodes[v_ids]
    V_def = apply_deformation_vectorized(V_sub, node_pos, R_list, t_list, vn_sub, vw_sub)
    
    # 2. 批量 3D->2D 投影
    V_def_h = np.hstack([V_def, np.ones((len(V_def), 1))]) # (N, 4)
    # 批量矩阵乘法 (N, 3, 4) @ (N, 4, 1) -> (N, 3, 1)
    V_c = np.matmul(batched_data['T_total'][:, :3, :], V_def_h[..., None]).squeeze(-1)
    z = np.maximum(V_c[:, 2:3], 1e-6)
    proj_contour = np.matmul(batched_data['K'][:, :2, :3], V_c[..., None]).squeeze(-1) / z
    
    # 3. 计算残差
    diff = proj_contour - batched_data['targets']
    res_contour = np.sum(diff * batched_data['grads'], axis=1) # 点到平面距离

    # ✅ 补上遗漏的这一行
    contour_weight = 1.0
    residuals.append(contour_weight * res_contour)
    # 2. ARAP 正则项 (Smoothness)
    # ... (与原代码保持一致) ...
    idx_k = graph_edges[:, 0]
    idx_l = graph_edges[:, 1]
    g_k, g_l = node_pos[idx_k], node_pos[idx_l]
    vec_kl = (g_l - g_k)[..., None]
    pred = (R_list[idx_k] @ vec_kl).squeeze(-1) + g_k + t_list[idx_k]
    target = g_l + t_list[idx_l]
    reg_res = np.sqrt(lambda_reg) * (pred - target)
    residuals.append(reg_res.ravel())
    
    # # 3. Prior term
    # # ... (与原代码保持一致) ...
    # prior_weight = 1.0
    # for i in range(K_nodes):
    #     residuals.append(prior_weight * t_list[i])
    #     rot_res = Rotation.from_matrix(R_list[i]).as_rotvec()
    #     residuals.append(prior_weight * rot_res)
        
    # # Anchor term
    # anchor_weight = 1e6
    # residuals.append(anchor_weight * (R_list[0] - np.eye(3)).ravel())
    # residuals.append(anchor_weight * t_list[0])
    
    return np.concatenate(residuals)

def apply_deformation_vectorized(V_subset, node_pos, R_list, t_list, v_nodes, v_weights):
    """
    V_subset: (M, 3) 待变形的顶点子集
    node_pos: (K, 3) 节点位置
    R_list: (K, 3, 3) 所有节点的旋转
    t_list: (K, 3) 所有节点的位移
    v_nodes: (M, knn) 每个顶点关联的节点索引
    v_weights: (M, knn) 每个顶点关联的节点权重
    """
    M = V_subset.shape[0]
    knn = v_nodes.shape[1]
    
    # 提取关联节点的参数 (M, knn, 3, 3) 和 (M, knn, 3)
    R_selected = R_list[v_nodes] 
    t_selected = t_list[v_nodes]
    g_selected = node_pos[v_nodes]
    
    # 计算 (v - g): (M, 1, 3) -> (M, knn, 3, 1)
    v_minus_g = (V_subset[:, None, :] - g_selected)[..., None]
    
    # 核心公式: R * (v - g) + g + t
    # 使用 matmul: (M, knn, 3, 3) @ (M, knn, 3, 1) -> (M, knn, 3, 1)
    deformed_parts = np.matmul(R_selected, v_minus_g).squeeze(-1) + g_selected + t_selected
    
    # 加权求和: (M, knn, 3) * (M, knn, 1) -> (M, 3)
    V_deformed = np.sum(deformed_parts * v_weights[..., None], axis=1)
    
    return V_deformed

def apply_deformation_graph(
    V0, node_pos,
    R_list, t_list,
    vertex_nodes, vertex_weights
):
    V_refined = apply_deformation_vectorized(
        V0, 
        node_pos, 
        R_list, 
        t_list, 
        vertex_nodes, 
        vertex_weights
    )
    
    return V_refined

def unpack_params(x):
    R_list, t_list = [], []
    for i in range(0, len(x), 6):
        rvec = x[i:i+3]
        t = x[i+3:i+6]
        R_list.append(axis_angle_to_rotmat(rvec))
        t_list.append(t)
    R_list = np.stack(R_list, axis=0)   # (K, 3, 3)
    t_list = np.stack(t_list, axis=0)   # (K, 3)
    return R_list, t_list

def to_homo(v):
    return np.append(v, 1.0)

def project_point_camera(K, X):
    x = X[0] / X[2]
    y = X[1] / X[2]
    u = K[0,0] * x + K[0,2]
    v = K[1,1] * y + K[1,2]
    return np.array([u, v])

def farthest_point_sampling(points, num_samples):
    """
    points: (N,3)
    return: (num_samples,) indices
    """
    N = points.shape[0]
    indices = np.zeros(num_samples, dtype=np.int64)

    # 随机选一个起点
    indices[0] = np.random.randint(N)
    dist = np.full(N, np.inf)

    for i in range(1, num_samples):
        p = points[indices[i - 1]]
        dist = np.minimum(dist, np.linalg.norm(points - p, axis=1))
        indices[i] = np.argmax(dist)

    return indices

def build_knn_graph(node_pos, k=4):
    """
    node_pos: (K,3)
    return: list of (i,j)
    """
    tree = cKDTree(node_pos)
    _, nn = tree.query(node_pos, k=k+1)

    edges = set()
    for i in range(node_pos.shape[0]):
        for j in nn[i][1:]:
            edges.add((i, j))
            edges.add((j, i))  # 无向图

    return list(edges)

def compute_vertex_node_weights(vertices, node_pos, k=4):
    """
    vertices: (N,3)
    node_pos: (K,3)

    return:
        vertex_nodes: (N,k) int
        vertex_weights: (N,k) float
    """
    tree = cKDTree(node_pos)
    dists, idxs = tree.query(vertices, k=k)

    d_max = dists[:, -1][:, None] + 1e-6 
    
    # ED 论文公式: weight = (1 - dist / d_max)^2
    # 这样保证在影响半径边缘权重平滑降为 0
    weights = (1.0 - dists / d_max) ** 2
    weights[dists > d_max] = 0 # 理论上不会发生，因为我们是用 knn 查的，但为了保险
    
    # 3. 归一化 (Partition of Unity) !!! 这一步如果不做，就会出现裂痕
    w_sum = np.sum(weights, axis=1, keepdims=True)
    weights = weights / (w_sum + 1e-8)
    
    return idxs, weights
