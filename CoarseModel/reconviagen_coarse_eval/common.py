#!/usr/bin/env python3

"""Shared utilities for the ReconViaGen + CoarseModel synthetic evaluation."""

from __future__ import annotations

import json
import math
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import trimesh
from PIL import Image
from scipy.spatial import cKDTree


ROOT = Path("/home/zjr/Tracker")
COARSE_ROOT = ROOT / "CoarseModel"
RECON_ROOT = ROOT / "ReconViaGen"
EVAL_ROOT = COARSE_ROOT / "reconviagen_coarse_eval"
DEFAULT_OUTPUT_ROOT = EVAL_ROOT / "outputs"


def ensure_dir(path: os.PathLike | str) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path: os.PathLike | str) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: os.PathLike | str, payload: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return path


def load_manifest(manifest_path: os.PathLike | str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    manifest_path = Path(manifest_path)
    data = read_json(manifest_path)
    samples = data.get("samples", data if isinstance(data, list) else None)
    if not isinstance(samples, list):
        raise ValueError(f"Unsupported manifest format: {manifest_path}")

    meta_path = manifest_path.parent / "manifest.json"
    meta = read_json(meta_path) if meta_path.exists() else {}
    meta.setdefault("manifest_path", str(manifest_path))
    return samples, meta


def sample_case_name(sample: Dict[str, Any], sample_index: int, prefix: str = "pixalv9") -> str:
    uid = str(sample.get("uid") or sample.get("object_uid") or f"idx{sample_index:04d}")
    safe_uid = "".join(c if c.isalnum() or c in "-_" else "_" for c in uid)
    return f"{prefix}_idx{sample_index:04d}_{safe_uid[:64]}"


def resolve_sample_path(root: os.PathLike | str, rel_or_abs: str) -> Path:
    path = Path(rel_or_abs)
    if path.is_absolute():
        return path
    return Path(root) / path


def image_to_rgb_and_mask(image_path: Path, mask_path: Optional[Path]) -> Tuple[np.ndarray, np.ndarray]:
    image = Image.open(image_path).convert("RGBA")
    rgba = np.array(image)
    rgb = rgba[:, :, :3].copy()
    alpha = rgba[:, :, 3]

    mask = None
    if mask_path is not None and mask_path.exists():
        mask_img = Image.open(mask_path).convert("RGBA")
        mask_rgba = np.array(mask_img)
        mask_alpha = mask_rgba[:, :, 3]
        if mask_rgba.shape[2] == 4 and mask_alpha.max() > mask_alpha.min():
            mask = mask_rgba[:, :, 3]
        else:
            mask = cv2.cvtColor(mask_rgba[:, :, :3], cv2.COLOR_RGB2GRAY)
    if mask is None or mask.max() == 0:
        if alpha.max() > alpha.min():
            mask = alpha
        else:
            mask = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    mask = (mask > 2).astype(np.uint8) * 255
    rgb[mask == 0] = 0
    return rgb, mask


def rotmat_to_qvec(R: np.ndarray) -> np.ndarray:
    """COLMAP quaternion order: qw, qx, qy, qz."""
    R = np.asarray(R, dtype=np.float64)
    K = np.array(
        [
            [R[0, 0] - R[1, 1] - R[2, 2], 0.0, 0.0, 0.0],
            [R[1, 0] + R[0, 1], R[1, 1] - R[0, 0] - R[2, 2], 0.0, 0.0],
            [R[2, 0] + R[0, 2], R[2, 1] + R[1, 2], R[2, 2] - R[0, 0] - R[1, 1], 0.0],
            [R[1, 2] - R[2, 1], R[2, 0] - R[0, 2], R[0, 1] - R[1, 0], R[0, 0] + R[1, 1] + R[2, 2]],
        ],
        dtype=np.float64,
    )
    K /= 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec


def c2w_to_w2c(c2w: np.ndarray) -> np.ndarray:
    c2w = np.asarray(c2w, dtype=np.float64)
    return np.linalg.inv(c2w)


def prepare_coarse_dataset_from_pixal_sample(
    manifest_path: os.PathLike | str,
    sample_index: int,
    output_root: os.PathLike | str = DEFAULT_OUTPUT_ROOT,
    case_name: Optional[str] = None,
    max_frames: int = 8,
    case_prefix: str = "pixalv9",
) -> Dict[str, Any]:
    samples, meta = load_manifest(manifest_path)
    if sample_index < 0 or sample_index >= len(samples):
        raise IndexError(f"sample_index {sample_index} out of range 0..{len(samples) - 1}")
    sample = samples[sample_index]

    output_root = Path(output_root)
    workspace = ensure_dir(output_root / "workspace")
    datasets_root = ensure_dir(workspace / "datasets")
    runs_root = ensure_dir(output_root / "runs")

    case_name = case_name or sample_case_name(sample, sample_index, prefix=case_prefix)
    dataset_dir = ensure_dir(datasets_root / case_name)
    rgb_dir = ensure_dir(dataset_dir / "rgb")
    mask_dir = ensure_dir(dataset_dir / "masks")
    sparse_dir = ensure_dir(dataset_dir / "sparse" / "0")
    models_dir = ensure_dir(dataset_dir / "models")
    run_dir = ensure_dir(runs_root / case_name)

    image_root = meta.get("image_root") or str(Path(manifest_path).parent / "images")
    mask_root = meta.get("mask_root") or str(Path(manifest_path).parent / "masks")
    extrinsics_type = meta.get("extrinsics_type", "c2w")
    frames = list(sample.get("frames", []))[: int(max_frames)]
    if not frames:
        raise ValueError(f"Sample has no frames: {sample.get('uid')}")

    cameras_lines = ["# Camera list with one line of data per camera:", "# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]"]
    images_lines = [
        "# Image list with two lines of data per image:",
        "# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, IMAGE_NAME",
        "# POINTS2D[] as (X, Y, POINT3D_ID)",
    ]
    frame_reports = []

    for i, frame in enumerate(frames):
        image_path = resolve_sample_path(image_root, frame["image"])
        mask_path = resolve_sample_path(mask_root, frame.get("mask", frame["image"]))
        rgb, mask = image_to_rgb_and_mask(image_path, mask_path)
        h, w = rgb.shape[:2]
        name = f"frame_{i:04d}.png"
        cv2.imwrite(str(rgb_dir / name), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(mask_dir / name), mask)

        K = np.asarray(frame["intrinsic"], dtype=np.float64)
        T = np.asarray(frame["extrinsic"], dtype=np.float64)
        if extrinsics_type == "c2w":
            T_w2c = c2w_to_w2c(T)
            T_c2w = T
        elif extrinsics_type == "w2c":
            T_w2c = T
            T_c2w = np.linalg.inv(T)
        else:
            raise ValueError(f"Unsupported extrinsics_type={extrinsics_type}")

        cam_id = i + 1
        image_id = i + 1
        fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
        cameras_lines.append(f"{cam_id} PINHOLE {w} {h} {fx:.9f} {fy:.9f} {cx:.9f} {cy:.9f}")

        qvec = rotmat_to_qvec(T_w2c[:3, :3])
        tvec = T_w2c[:3, 3]
        q_text = " ".join(f"{v:.12g}" for v in qvec)
        t_text = " ".join(f"{v:.12g}" for v in tvec)
        images_lines.append(f"{image_id} {q_text} {t_text} {cam_id} {name}")
        images_lines.append("")

        frame_reports.append(
            {
                "frame_index": i,
                "image_name": name,
                "source_image": str(image_path),
                "source_mask": str(mask_path),
                "intrinsic": K.tolist(),
                "T_w2c": T_w2c.tolist(),
                "T_c2w": T_c2w.tolist(),
                "mask_pixels": int((mask > 0).sum()),
                "mask_area_ratio": float((mask > 0).mean()),
            }
        )

    (sparse_dir / "cameras.txt").write_text("\n".join(cameras_lines) + "\n", encoding="utf-8")
    (sparse_dir / "images.txt").write_text("\n".join(images_lines) + "\n", encoding="utf-8")
    (sparse_dir / "points3D.txt").write_text("# Empty synthetic sparse points. CoarseModel 4stage path does not require this file.\n", encoding="utf-8")

    source_glb = sample.get("source_glb")
    latent_root = meta.get("latent_root") or str(Path(manifest_path).parent / "ss_latents")
    ss_latent = sample.get("ss_latent")
    ss_latent_path = str(resolve_sample_path(latent_root, ss_latent)) if ss_latent else None
    report = {
        "case_name": case_name,
        "sample_index": int(sample_index),
        "uid": sample.get("uid"),
        "object_uid": sample.get("object_uid"),
        "source_glb": source_glb,
        "ss_latent": ss_latent,
        "ss_latent_path": ss_latent_path,
        "manifest_path": str(manifest_path),
        "dataset_dir": str(dataset_dir),
        "workspace": str(workspace),
        "models_dir": str(models_dir),
        "run_dir": str(run_dir),
        "extrinsics_type": extrinsics_type,
        "frames": frame_reports,
    }
    write_json(dataset_dir / "reconviagen_meta.json", report)
    write_json(run_dir / "prepared_sample.json", report)
    return report


def as_mesh(obj: trimesh.Trimesh | trimesh.Scene) -> trimesh.Trimesh:
    if isinstance(obj, trimesh.Scene):
        meshes = [m for m in obj.dump(concatenate=False) if isinstance(m, trimesh.Trimesh) and len(m.vertices) > 0]
        if not meshes:
            raise ValueError("Scene has no mesh geometry")
        return trimesh.util.concatenate(meshes)
    if isinstance(obj, trimesh.Trimesh):
        return obj
    raise TypeError(f"Unsupported trimesh object: {type(obj)}")


def load_mesh(path: os.PathLike | str) -> trimesh.Trimesh:
    try:
        obj = trimesh.load(path, force="mesh", process=False)
        if isinstance(obj, trimesh.Trimesh) and len(obj.vertices) > 0:
            return obj
    except Exception:
        pass
    return as_mesh(trimesh.load(path, force="scene", process=False))


def export_mesh_as_obj(mesh_path: os.PathLike | str, obj_path: os.PathLike | str) -> Path:
    mesh = load_mesh(mesh_path)
    obj_path = Path(obj_path)
    obj_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(obj_path)
    return obj_path


def mesh_basic_stats(mesh: trimesh.Trimesh) -> Dict[str, Any]:
    extents = np.asarray(mesh.bounding_box.extents, dtype=np.float64)
    components = None
    if len(mesh.faces) < 200000:
        try:
            components = int(len(mesh.split(only_watertight=False))) if len(mesh.faces) else 0
        except Exception:
            components = None
    return {
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "is_watertight": bool(mesh.is_watertight),
        "components": components,
        "bbox_extent": [float(v) for v in extents.tolist()],
        "bbox_diag": float(np.linalg.norm(extents)),
        "surface_area": float(mesh.area) if np.isfinite(mesh.area) else None,
        "volume": float(mesh.volume) if mesh.is_watertight and np.isfinite(mesh.volume) else None,
    }


def sample_mesh_points(mesh: trimesh.Trimesh, count: int, seed: int = 0) -> np.ndarray:
    rng_state = np.random.get_state()
    np.random.seed(seed)
    try:
        if len(mesh.faces) > 0:
            points, _ = trimesh.sample.sample_surface(mesh, count)
        else:
            verts = np.asarray(mesh.vertices, dtype=np.float64)
            idx = np.random.choice(len(verts), size=min(count, len(verts)), replace=len(verts) < count)
            points = verts[idx]
    finally:
        np.random.set_state(rng_state)
    return np.asarray(points, dtype=np.float64)


def normalize_points(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    center = 0.5 * (points.min(axis=0) + points.max(axis=0))
    extent = np.max(points.max(axis=0) - points.min(axis=0))
    return (points - center[None]) / max(float(extent), 1e-8)


def umeyama_similarity(src: np.ndarray, dst: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_c = src - mu_src
    dst_c = dst - mu_dst
    cov = (dst_c.T @ src_c) / max(len(src), 1)
    U, S, Vt = np.linalg.svd(cov)
    D = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        D[-1, -1] = -1
    R = U @ D @ Vt
    var_src = np.mean(np.sum(src_c * src_c, axis=1))
    scale = float(np.trace(np.diag(S) @ D) / max(var_src, 1e-12))
    t = mu_dst - scale * (R @ mu_src)
    return scale, R, t


def icp_similarity_align(src: np.ndarray, dst: np.ndarray, iterations: int = 12) -> np.ndarray:
    aligned = np.asarray(src, dtype=np.float64).copy()
    tree = cKDTree(dst)
    for _ in range(max(0, int(iterations))):
        _, ids = tree.query(aligned, k=1, workers=-1)
        scale, R, t = umeyama_similarity(aligned, dst[ids])
        aligned = (scale * (R @ aligned.T)).T + t[None]
    return aligned


def chamfer_metrics(pred_points: np.ndarray, gt_points: np.ndarray, thresholds: Sequence[float]) -> Dict[str, float]:
    tree_gt = cKDTree(gt_points)
    tree_pred = cKDTree(pred_points)
    d_pred, _ = tree_gt.query(pred_points, k=1, workers=-1)
    d_gt, _ = tree_pred.query(gt_points, k=1, workers=-1)
    out = {
        "pred_to_gt_mean": float(np.mean(d_pred)),
        "gt_to_pred_mean": float(np.mean(d_gt)),
        "chamfer_l1": float(0.5 * (np.mean(d_pred) + np.mean(d_gt))),
        "chamfer_l2": float(0.5 * (np.mean(d_pred ** 2) + np.mean(d_gt ** 2))),
        "pred_to_gt_p95": float(np.percentile(d_pred, 95)),
        "gt_to_pred_p95": float(np.percentile(d_gt, 95)),
    }
    for thr in thresholds:
        precision = float(np.mean(d_pred < thr))
        recall = float(np.mean(d_gt < thr))
        fscore = 0.0 if precision + recall <= 1e-12 else 2.0 * precision * recall / (precision + recall)
        key = str(thr).replace(".", "p")
        out[f"precision_{key}"] = precision
        out[f"recall_{key}"] = recall
        out[f"fscore_{key}"] = float(fscore)
    return out


def evaluate_mesh_quality(
    pred_mesh_path: os.PathLike | str,
    gt_mesh_path: os.PathLike | str,
    output_json: os.PathLike | str,
    samples: int = 20000,
    seed: int = 0,
    icp_iters: int = 12,
) -> Dict[str, Any]:
    pred_mesh = load_mesh(pred_mesh_path)
    gt_mesh = pred_mesh if str(pred_mesh_path) == str(gt_mesh_path) else load_mesh(gt_mesh_path)
    pred_points = normalize_points(sample_mesh_points(pred_mesh, samples, seed=seed))
    gt_points = normalize_points(sample_mesh_points(gt_mesh, samples, seed=seed + 17))
    pred_aligned = icp_similarity_align(pred_points, gt_points, iterations=icp_iters)
    metrics = {
        "pred_mesh": str(pred_mesh_path),
        "gt_mesh": str(gt_mesh_path),
        "normalization": "bbox-center and max-extent to unit cube, then pred-to-gt Sim3 ICP",
        "num_sample_points": int(samples),
        "pred_basic": mesh_basic_stats(pred_mesh),
        "gt_basic": mesh_basic_stats(gt_mesh),
        "surface_metrics": chamfer_metrics(pred_aligned, gt_points, thresholds=[0.01, 0.02, 0.05, 0.10]),
    }
    write_json(output_json, metrics)
    return metrics


def load_sparse_target_points(npz_path: os.PathLike | str, voxel_resolution: int = 64) -> np.ndarray:
    data = np.load(npz_path)
    if "target_coords" not in data:
        raise KeyError(f"{npz_path} has no target_coords")
    coords = np.asarray(data["target_coords"], dtype=np.float64)
    return (coords + 0.5) / float(voxel_resolution) - 0.5


def evaluate_mesh_quality_against_points(
    pred_mesh_path: os.PathLike | str,
    gt_points: np.ndarray,
    output_json: os.PathLike | str,
    samples: int = 20000,
    seed: int = 0,
    icp_iters: int = 12,
) -> Dict[str, Any]:
    pred_mesh = load_mesh(pred_mesh_path)
    pred_points = normalize_points(sample_mesh_points(pred_mesh, samples, seed=seed))
    gt_points_norm = normalize_points(np.asarray(gt_points, dtype=np.float64))
    if len(gt_points_norm) > samples:
        rng = np.random.default_rng(seed + 19)
        gt_points_norm = gt_points_norm[rng.choice(len(gt_points_norm), size=samples, replace=False)]
    pred_aligned = icp_similarity_align(pred_points, gt_points_norm, iterations=icp_iters)
    metrics = {
        "pred_mesh": str(pred_mesh_path),
        "gt_reference": "pixal3d_sparse_target_coords",
        "normalization": "bbox-center and max-extent to unit cube, then pred-to-target Sim3 ICP",
        "num_pred_sample_points": int(len(pred_points)),
        "num_gt_points": int(len(gt_points_norm)),
        "pred_basic": mesh_basic_stats(pred_mesh),
        "surface_to_sparse_metrics": chamfer_metrics(pred_aligned, gt_points_norm, thresholds=[0.01, 0.02, 0.05, 0.10]),
    }
    write_json(output_json, metrics)
    return metrics


def mask_metrics(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    pred_b = pred > 0
    gt_b = gt > 0
    inter = float(np.logical_and(pred_b, gt_b).sum())
    union = float(np.logical_or(pred_b, gt_b).sum())
    pred_sum = float(pred_b.sum())
    gt_sum = float(gt_b.sum())
    return {
        "iou": inter / union if union > 0 else 0.0,
        "precision": inter / pred_sum if pred_sum > 0 else 0.0,
        "recall": inter / gt_sum if gt_sum > 0 else 0.0,
        "pred_area": pred_sum,
        "gt_area": gt_sum,
    }


def contour_chamfer_px(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    def edge(mask: np.ndarray) -> np.ndarray:
        contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return np.zeros((0, 2), dtype=np.float32)
        return np.concatenate([c.reshape(-1, 2) for c in contours], axis=0).astype(np.float32)

    pred_pts = edge(pred)
    gt_pts = edge(gt)
    if len(pred_pts) == 0 or len(gt_pts) == 0:
        return {"contour_chamfer_px": float("inf"), "pred_to_gt_contour_px": float("inf"), "gt_to_pred_contour_px": float("inf")}
    pred_to_gt, _ = cKDTree(gt_pts).query(pred_pts, k=1, workers=-1)
    gt_to_pred, _ = cKDTree(pred_pts).query(gt_pts, k=1, workers=-1)
    return {
        "contour_chamfer_px": float(0.5 * (np.mean(pred_to_gt) + np.mean(gt_to_pred))),
        "pred_to_gt_contour_px": float(np.mean(pred_to_gt)),
        "gt_to_pred_contour_px": float(np.mean(gt_to_pred)),
    }


def add_coarse_import_paths() -> None:
    for path in [str(COARSE_ROOT), str(COARSE_ROOT / "core"), str(COARSE_ROOT / "external" / "dinov2")]:
        if path not in sys.path:
            sys.path.insert(0, path)


def patch_coarse_app_config(workspace: os.PathLike | str) -> None:
    add_coarse_import_paths()
    workspace = Path(workspace)
    datasets = ensure_dir(workspace / "datasets")
    results = ensure_dir(workspace / "results")
    configs = ensure_dir(workspace / "configs")

    import importlib

    modules = []
    for name in ["core.config", "config"]:
        try:
            modules.append(importlib.import_module(name))
        except Exception:
            pass

    for module in modules:
        app_config = getattr(module, "AppConfig", None)
        if app_config is None:
            continue
        app_config.PROJECT_ROOT = workspace
        app_config.DATASETS_PATH = datasets
        app_config.OUTPUT_ROOT = results
        app_config.TEMPLATE_ROOT = results / "templates"
        app_config.REPRE_ROOT = results / "object_repre"
        app_config.REFINE_ROOT = results / "refine_model"


def summarize_rows(rows: List[Dict[str, float]], prefix: str = "") -> Dict[str, float]:
    if not rows:
        return {}
    keys = [k for k, v in rows[0].items() if isinstance(v, (int, float, np.floating)) and math.isfinite(float(v))]
    out = {}
    for key in keys:
        vals = [float(row[key]) for row in rows if key in row and math.isfinite(float(row[key]))]
        if not vals:
            continue
        name = f"{prefix}{key}" if prefix else key
        out[f"{name}_mean"] = float(np.mean(vals))
        out[f"{name}_median"] = float(np.median(vals))
        out[f"{name}_min"] = float(np.min(vals))
        out[f"{name}_max"] = float(np.max(vals))
    return out


def copy_text_report(src: os.PathLike | str, dst: os.PathLike | str) -> Optional[str]:
    src = Path(src)
    if not src.exists():
        return None
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return str(dst)


def link_or_copy(src: os.PathLike | str, dst: os.PathLike | str, mode: str = "symlink") -> Path:
    src = Path(src)
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return dst
    if mode == "symlink":
        os.symlink(src, dst, target_is_directory=src.is_dir())
    elif mode == "copy":
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
    else:
        raise ValueError(f"Unsupported link_or_copy mode: {mode}")
    return dst


def copy_tree_contents(src: os.PathLike | str, dst: os.PathLike | str) -> Path:
    src = Path(src)
    dst = ensure_dir(dst)
    for child in src.iterdir():
        target = dst / child.name
        if target.exists() or target.is_symlink():
            continue
        if child.is_dir():
            shutil.copytree(child, target)
        else:
            shutil.copy2(child, target)
    return dst


def prepare_existing_coarse_dataset(
    dataset_dir: os.PathLike | str,
    output_root: os.PathLike | str = DEFAULT_OUTPUT_ROOT,
    case_name: Optional[str] = None,
    max_frames: Optional[int] = None,
    link_mode: str = "symlink",
) -> Dict[str, Any]:
    """Register an existing CoarseModel dataset under the eval workspace.

    The dataset folder itself is not modified. Large immutable inputs are
    symlinked by default; `models/` is copied so generated/reinstalled meshes do
    not write through to the original dataset.
    """
    source_dir = Path(dataset_dir).resolve()
    if not source_dir.exists():
        raise FileNotFoundError(source_dir)
    case_name = case_name or source_dir.name

    output_root = Path(output_root)
    workspace = ensure_dir(output_root / "workspace")
    datasets_root = ensure_dir(workspace / "datasets")
    runs_root = ensure_dir(output_root / "runs")
    dst_dir = ensure_dir(datasets_root / case_name)
    run_dir = ensure_dir(runs_root / case_name)

    for name in ["sparse", "poses.txt", "reconviagen_output"]:
        src = source_dir / name
        if src.exists():
            link_or_copy(src, dst_dir / name, mode=link_mode)

    src_rgb = source_dir / "rgb"
    if not src_rgb.exists():
        src_rgb = source_dir / "images"
    src_images = source_dir / "images"
    src_masks = source_dir / "masks"
    selected_rgb_files: List[Path] = []
    if src_rgb.exists():
        selected_rgb_files = sorted(p for p in src_rgb.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        if max_frames is not None:
            selected_rgb_files = selected_rgb_files[: int(max_frames)]

    if max_frames is None:
        for name in ["rgb", "images", "masks"]:
            src = source_dir / name
            if src.exists():
                link_or_copy(src, dst_dir / name, mode=link_mode)
    else:
        for name in ["rgb", "images", "masks"]:
            target = dst_dir / name
            if target.is_symlink():
                target.unlink()
            ensure_dir(target)
        for image_path in selected_rgb_files:
            for out_name, src_base in [("rgb", src_rgb), ("images", src_images)]:
                if src_base.exists():
                    src_image = src_base / image_path.name
                    if src_image.exists():
                        link_or_copy(src_image, dst_dir / out_name / image_path.name, mode=link_mode)
            if src_masks.exists():
                mask_path = src_masks / f"{image_path.stem}.png"
                if mask_path.exists():
                    link_or_copy(mask_path, dst_dir / "masks" / mask_path.name, mode=link_mode)

    src_models = source_dir / "models"
    if src_models.exists():
        copy_tree_contents(src_models, dst_dir / "models")
    else:
        ensure_dir(dst_dir / "models")

    rgb_dir = dst_dir / "rgb"
    if not rgb_dir.exists() and (dst_dir / "images").exists():
        link_or_copy(dst_dir / "images", rgb_dir, mode="symlink")

    frame_reports = []
    rgb_source = rgb_dir if rgb_dir.exists() else (dst_dir / "images")
    mask_source = dst_dir / "masks"
    if rgb_source.exists() and mask_source.exists():
        rgb_files = sorted(p for p in rgb_source.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        for i, image_path in enumerate(rgb_files):
            mask_path = mask_source / f"{image_path.stem}.png"
            mask_area_ratio = None
            mask_pixels = None
            if mask_path.exists():
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    mask_bool = mask > 0
                    mask_area_ratio = float(mask_bool.mean())
                    mask_pixels = int(mask_bool.sum())
            frame_reports.append(
                {
                    "frame_index": i,
                    "image_name": image_path.name,
                    "image_path": str(image_path),
                    "mask_path": str(mask_path) if mask_path.exists() else None,
                    "mask_pixels": mask_pixels,
                    "mask_area_ratio": mask_area_ratio,
                }
            )

    default_recon_mesh = source_dir / "reconviagen_output" / "reconstructed_object.glb"
    source_model_mesh = source_dir / "models" / f"{source_dir.name}.obj"
    source_model_norm_mesh = source_dir / "models" / f"{source_dir.name}_norm.obj"
    if not source_model_mesh.exists() and (source_dir / "models").exists():
        obj_candidates = sorted((source_dir / "models").glob("*.obj"))
        source_model_mesh = obj_candidates[0] if obj_candidates else source_model_mesh
    report = {
        "case_name": case_name,
        "source_dataset_dir": str(source_dir),
        "dataset_dir": str(dst_dir),
        "workspace": str(workspace),
        "models_dir": str(dst_dir / "models"),
        "run_dir": str(run_dir),
        "default_recon_mesh": str(default_recon_mesh) if default_recon_mesh.exists() else None,
        "source_model_mesh": str(source_model_mesh) if source_model_mesh.exists() else None,
        "source_model_norm_mesh": str(source_model_norm_mesh) if source_model_norm_mesh.exists() else None,
        "input_type": "existing_coarse_dataset",
        "link_mode": link_mode,
        "frames": frame_reports,
    }
    meta_path = dst_dir / "reconviagen_meta.json"
    if meta_path.is_symlink():
        meta_path.unlink()
    write_json(meta_path, report)
    write_json(run_dir / "prepared_sample.json", report)
    return report


def evaluate_mesh_basic(
    pred_mesh_path: os.PathLike | str,
    output_json: os.PathLike | str,
) -> Dict[str, Any]:
    mesh = load_mesh(pred_mesh_path)
    metrics = {
        "pred_mesh": str(pred_mesh_path),
        "gt_reference": None,
        "note": "No GT mesh or sparse target coords were available; this report only contains intrinsic mesh statistics.",
        "pred_basic": mesh_basic_stats(mesh),
    }
    write_json(output_json, metrics)
    return metrics
