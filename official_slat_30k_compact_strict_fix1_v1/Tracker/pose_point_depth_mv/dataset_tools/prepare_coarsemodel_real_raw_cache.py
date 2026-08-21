#!/usr/bin/env python3
"""Adapt local CoarseModel captures to the audited real runtime-input frontend.

This adapter consumes RGB, foreground masks, a COLMAP/phone-pose sparse model,
and its point cloud.  Existing CoarseModel/ReconViaGen meshes are copied only as
visual references in metadata; they are never consumed by model inference.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import shutil
from typing import Any, Sequence

import numpy as np
from PIL import Image

from ar_ss_flow.shared_object_preprocessing import canonical_json_sha256
from pose_point_depth_mv.dataset_tools.prepare_omni_real_video_cache import (
    IMAGE_SUFFIXES,
    OBJECT_CACHE_FORMAT,
    RAW_CACHE_FORMAT,
    camera_intrinsics,
    parse_cameras,
    parse_points,
    parse_registered_images,
    qvec_to_rotation,
    sha256_file,
    utc_now,
    write_json,
    write_npz,
)


ADAPTER_FORMAT = "pose_point_depth_mv.coarsemodel_real_raw_adapter.v1"
DEFAULT_DATASETS = (
    "CoarseModel/datasets/GOOD_MESH_TEST",
    "CoarseModel/datasets/reconviagen_20260514_071732",
    "CoarseModel/datasets/reconviagen_20260513_022427",
    "CoarseModel/datasets/heimei",
)
AUTO_SPARSE_CANDIDATES = (
    "sparse_native_no_vggt_eval_20260810_v1/0",
    "sparse_slam_eval_four_v2/0",
    "sparse_colmap_arproxy_rgb64_v1/0",
    "sparse_slam_relaxed_countcheck/0",
    "sparse_slam_strictmask_countcheck/0",
    "sparse_slam/0",
    "sparse_arproxy_streaming_from_existing_v1/0",
    "sparse/0",
)


def safe_object_id(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    if not result:
        raise ValueError(f"invalid dataset name: {value!r}")
    return result


def parse_overrides(values: Sequence[str]) -> dict[str, Path]:
    output: dict[str, Path] = {}
    for raw in values:
        if "=" not in raw:
            raise ValueError(f"sparse override must be DATASET_NAME=PATH: {raw!r}")
        name, path = raw.split("=", 1)
        name = safe_object_id(name)
        if name in output:
            raise ValueError(f"duplicate sparse override: {name}")
        output[name] = Path(path).expanduser().resolve()
    return output


def find_image_dir(dataset: Path) -> Path:
    for name in ("images", "rgb", "color"):
        candidate = dataset / name
        if candidate.is_dir() and any(
            path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
            for path in candidate.iterdir()
        ):
            return candidate.resolve()
    raise FileNotFoundError(f"no RGB directory with images: {dataset}")


def find_mask(mask_dir: Path, frame_name: str) -> Path | None:
    stem = Path(frame_name).stem
    for suffix in (".png", ".jpg", ".jpeg"):
        candidate = mask_dir / f"{stem}{suffix}"
        if candidate.is_file():
            return candidate.resolve()
    return None


def point_count(sparse: Path) -> int:
    try:
        return int(len(parse_points(sparse / "points3D.txt")["xyz"]))
    except (FileNotFoundError, RuntimeError, ValueError):
        return 0


def resolve_sparse(
    dataset: Path, override: Path | None, *, allow_empty_points: bool = False
) -> Path:
    candidates = [override] if override is not None else [
        dataset / relative for relative in AUTO_SPARSE_CANDIDATES
    ]
    for sparse in candidates:
        if sparse is None:
            continue
        sparse = Path(sparse).resolve()
        required = tuple(sparse / name for name in ("cameras.txt", "images.txt", "points3D.txt"))
        if all(path.is_file() for path in required) and (
            allow_empty_points or point_count(sparse) > 0
        ):
            return sparse
    description = str(override) if override is not None else ", ".join(AUTO_SPARSE_CANDIDATES)
    raise FileNotFoundError(
        f"no usable sparse camera model for {dataset.name}; searched {description}; "
        f"allow_empty_points={bool(allow_empty_points)}"
    )


def parse_points_allow_empty(path: Path, *, allow_empty: bool) -> dict[str, np.ndarray]:
    try:
        return parse_points(path)
    except RuntimeError as error:
        if not allow_empty or "empty points3D model" not in str(error):
            raise
    return {
        "point_id": np.empty((0,), dtype=np.int64),
        "xyz": np.empty((0, 3), dtype=np.float64),
        "rgb": np.empty((0, 3), dtype=np.uint8),
        "reprojection_error": np.empty((0,), dtype=np.float64),
        "track_length": np.empty((0,), dtype=np.int32),
        "confidence_proxy": np.empty((0,), dtype=np.float64),
    }


def reference_mesh(dataset: Path) -> Path | None:
    normalized = sorted((dataset / "models").glob("*_norm.obj"))
    candidates = normalized + sorted((dataset / "models").glob("*.obj"))
    return candidates[0].resolve() if candidates else None


def legacy_reconstruction(dataset: Path) -> Path | None:
    for suffix in ("glb", "obj", "ply"):
        candidate = dataset / "reconviagen_output" / f"reconstructed_object.{suffix}"
        if candidate.is_file():
            return candidate.resolve()
    return None


def legacy_reconstruction_video(dataset: Path) -> Path | None:
    candidate = dataset / "reconviagen_output" / "reconstructed_object.mp4"
    return candidate.resolve() if candidate.is_file() else None


def nearest_rotation(matrix: np.ndarray) -> np.ndarray:
    u, _singular, vt = np.linalg.svd(np.asarray(matrix, dtype=np.float64))
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    return rotation


def unity_euler_to_rotation(rot_deg: Sequence[float]) -> np.ndarray:
    """Match CoarseModel's legacy Unity Euler convention exactly."""

    rx, ry, rz = np.deg2rad(np.asarray(rot_deg, dtype=np.float64))
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    rx_matrix = np.asarray([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry_matrix = np.asarray([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rz_matrix = np.asarray([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return ry_matrix @ rx_matrix @ rz_matrix


def unity_quaternion_to_rotation(quaternion_xyzw: Sequence[float]) -> np.ndarray:
    x, y, z, w = np.asarray(quaternion_xyzw, dtype=np.float64)
    norm = float(np.linalg.norm([x, y, z, w]))
    if norm <= 1.0e-12:
        raise ValueError("Unity quaternion has zero norm")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def read_phone_poses(path: Path) -> dict[str, dict[str, np.ndarray | None]]:
    poses: dict[str, dict[str, np.ndarray | None]] = {}
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        fields = [value.strip() for value in raw.split(",")]
        if not fields or fields[0] == "frame_name" or len(fields) < 7:
            continue
        try:
            position = np.asarray([float(value) for value in fields[1:4]], dtype=np.float64)
            euler = np.asarray([float(value) for value in fields[4:7]], dtype=np.float64)
        except ValueError:
            continue
        quaternion = None
        if len(fields) >= 11 and all(fields[index] for index in range(7, 11)):
            try:
                quaternion = np.asarray(
                    [float(value) for value in fields[7:11]], dtype=np.float64
                )
            except ValueError:
                quaternion = None
        poses[Path(fields[0]).name] = {
            "position": position,
            "euler": euler,
            "quaternion": quaternion,
        }
    if not poses:
        raise RuntimeError(f"no usable phone poses: {path}")
    return poses


def phone_pose_c2w(pose: dict[str, np.ndarray | None]) -> tuple[np.ndarray, np.ndarray]:
    quaternion = pose.get("quaternion")
    if quaternion is not None:
        unity_c2w = unity_quaternion_to_rotation(quaternion)
    else:
        euler = pose.get("euler")
        if euler is None:
            raise RuntimeError("phone pose has neither quaternion nor Euler rotation")
        unity_c2w = unity_euler_to_rotation(euler)
    unity_to_cv_world = np.diag([1.0, 1.0, -1.0])
    unity_camera_to_cv_camera = np.diag([1.0, -1.0, 1.0])
    center = unity_to_cv_world @ np.asarray(pose["position"], dtype=np.float64)
    rotation = nearest_rotation(
        unity_to_cv_world @ unity_c2w @ unity_camera_to_cv_camera
    )
    return rotation, center


def proper_umeyama(source: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("Umeyama inputs must be matching [N,3] arrays")
    if len(source) < 3:
        raise ValueError("proper Umeyama requires at least three correspondences")
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    covariance = target_centered.T @ source_centered / float(len(source))
    u, singular, vt = np.linalg.svd(covariance)
    sign = np.ones(3, dtype=np.float64)
    if np.linalg.det(u @ vt) < 0:
        sign[-1] = -1.0
    rotation = u @ np.diag(sign) @ vt
    variance = float(np.mean(np.sum(source_centered * source_centered, axis=1)))
    if variance <= 1.0e-12:
        raise ValueError("phone camera centers are degenerate")
    scale = float(np.sum(singular * sign) / variance)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("phone-to-COLMAP similarity has an invalid scale")
    translation = target_mean - scale * (rotation @ source_mean)
    return scale, rotation, translation


def rotation_error_degrees(first: np.ndarray, second: np.ndarray) -> float:
    relative = np.asarray(first).T @ np.asarray(second)
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def registered_c2w(row: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    rotation_w2c = qvec_to_rotation(row["qvec"])
    center = -rotation_w2c.T @ np.asarray(row["tvec"], dtype=np.float64)
    return rotation_w2c.T, center


def row_rotation_w2c(row: dict[str, Any]) -> np.ndarray:
    if "rotation_w2c" in row:
        return np.asarray(row["rotation_w2c"], dtype=np.float64)
    return qvec_to_rotation(row["qvec"])


def fit_local_phone_to_colmap(
    target_c2w: np.ndarray,
    target_center: np.ndarray,
    anchors: list[dict[str, Any]],
    *,
    neighbors: int = 5,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    count = min(int(neighbors), len(anchors))
    if count < 3:
        raise RuntimeError("at least three registered phone/COLMAP anchors are required")
    distances = np.asarray(
        [np.linalg.norm(anchor["phone_center"] - target_center) for anchor in anchors]
    )
    selected = [anchors[int(index)] for index in np.argsort(distances)[:count]]
    source = np.asarray([row["phone_center"] for row in selected])
    target = np.asarray([row["colmap_center"] for row in selected])
    scale, world_rotation, translation = proper_umeyama(source, target)
    camera_basis_candidates = [
        (world_rotation @ row["phone_c2w"]).T @ row["colmap_c2w"]
        for row in selected
    ]
    camera_basis = nearest_rotation(np.sum(camera_basis_candidates, axis=0))
    colmap_center = scale * (world_rotation @ target_center) + translation
    colmap_c2w = nearest_rotation(world_rotation @ target_c2w @ camera_basis)
    return colmap_c2w, colmap_center, {
        "neighbor_frame_names": [str(row["name"]) for row in selected],
        "local_similarity_scale": scale,
    }


def phone_pose_completion(
    dataset: Path,
    registered: list[dict[str, Any]],
    frame_names: Sequence[str],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    pose_path = dataset / "poses.txt"
    if not pose_path.is_file():
        raise FileNotFoundError(f"all-image mode requires phone poses: {pose_path}")
    phone = read_phone_poses(pose_path)
    registered_by_name = {str(row["name"]): row for row in registered}
    missing_phone = sorted(set(frame_names) - set(phone))
    if missing_phone:
        raise RuntimeError(f"phone poses missing for selected images: {missing_phone}")
    anchors: list[dict[str, Any]] = []
    for name in sorted(set(registered_by_name) & set(phone)):
        colmap_c2w, colmap_center = registered_c2w(registered_by_name[name])
        phone_c2w, phone_center = phone_pose_c2w(phone[name])
        anchors.append(
            {
                "name": name,
                "phone_c2w": phone_c2w,
                "phone_center": phone_center,
                "colmap_c2w": colmap_c2w,
                "colmap_center": colmap_center,
                "camera_id": int(registered_by_name[name]["camera_id"]),
            }
        )
    if len(anchors) < 4:
        raise RuntimeError(
            f"all-image mode requires >=4 registered phone/COLMAP anchors; got {len(anchors)}"
        )

    # Leave-one-out measures the same local estimator on known COLMAP cameras.
    center_errors = []
    rotation_errors = []
    for held_out in anchors:
        remaining = [row for row in anchors if row["name"] != held_out["name"]]
        predicted_rotation, predicted_center, _ = fit_local_phone_to_colmap(
            held_out["phone_c2w"], held_out["phone_center"], remaining
        )
        center_errors.append(float(np.linalg.norm(predicted_center - held_out["colmap_center"])))
        rotation_errors.append(
            rotation_error_degrees(predicted_rotation, held_out["colmap_c2w"])
        )
    colmap_centers = np.asarray([row["colmap_center"] for row in anchors])
    camera_diameter = float(
        np.max(np.linalg.norm(colmap_centers[:, None] - colmap_centers[None, :], axis=2))
    )
    normalized_center_errors = np.asarray(center_errors) / max(camera_diameter, 1.0e-12)
    checks = {
        "loo_center_median_over_camera_diameter_le_0p05": bool(
            np.median(normalized_center_errors) <= 0.05
        ),
        "loo_center_max_over_camera_diameter_le_0p10": bool(
            np.max(normalized_center_errors) <= 0.10
        ),
        "loo_rotation_median_deg_le_5": bool(np.median(rotation_errors) <= 5.0),
        "loo_rotation_max_deg_le_15": bool(np.max(rotation_errors) <= 15.0),
    }
    audit = {
        "method": "local_5nn_phone_to_colmap_proper_sim3_plus_camera_basis.v1",
        "registered_anchor_count": len(anchors),
        "camera_center_diameter": camera_diameter,
        "loo_center_error": {
            "median": float(np.median(center_errors)),
            "maximum": float(np.max(center_errors)),
            "median_over_camera_diameter": float(np.median(normalized_center_errors)),
            "maximum_over_camera_diameter": float(np.max(normalized_center_errors)),
        },
        "loo_rotation_error_degrees": {
            "median": float(np.median(rotation_errors)),
            "maximum": float(np.max(rotation_errors)),
        },
        "checks": checks,
        "passed": all(checks.values()),
    }
    if not audit["passed"]:
        raise RuntimeError(f"phone-to-COLMAP pose-completion audit failed: {audit}")

    completed: dict[str, dict[str, Any]] = {}
    next_image_id = max(int(row["image_id"]) for row in registered) + 1
    for name in frame_names:
        if name in registered_by_name:
            row = dict(registered_by_name[name])
            row["pose_source"] = "colmap_registered"
            completed[name] = row
            continue
        phone_c2w, phone_center = phone_pose_c2w(phone[name])
        colmap_c2w, colmap_center, fit = fit_local_phone_to_colmap(
            phone_c2w, phone_center, anchors
        )
        nearest_anchor = min(
            anchors, key=lambda row: float(np.linalg.norm(row["phone_center"] - phone_center))
        )
        rotation_w2c = colmap_c2w.T
        completed[name] = {
            "image_id": next_image_id,
            "camera_id": int(nearest_anchor["camera_id"]),
            "name": name,
            "rotation_w2c": rotation_w2c,
            "tvec": -rotation_w2c @ colmap_center,
            "pose_source": "phone_pose_local_colmap_completion",
            "pose_completion": fit,
        }
        next_image_id += 1
    audit["completed_frame_count"] = sum(
        row["pose_source"] == "phone_pose_local_colmap_completion"
        for row in completed.values()
    )
    audit["pose_file"] = {"path": str(pose_path.resolve()), "sha256": sha256_file(pose_path)}
    return completed, audit


def _source_binding(
    dataset: Path,
    sparse: Path,
    pairs: list[tuple[dict[str, Any], Path, Path]],
    pose_completion_audit: dict[str, Any] | None,
) -> dict[str, Any]:
    sparse_files = {
        name: {
            "path": str((sparse / name).resolve()),
            "sha256": sha256_file(sparse / name),
        }
        for name in ("cameras.txt", "images.txt", "points3D.txt")
    }
    frames = []
    for row, image, mask in pairs:
        frame = {
            "source_frame_name": str(row["name"]),
            "image": str(image),
            "image_sha256": sha256_file(image),
            "mask": str(mask),
            "mask_sha256": sha256_file(mask),
        }
        if pose_completion_audit is not None:
            frame["pose_source"] = str(row.get("pose_source", "colmap_registered"))
            frame["T_W2C"] = (
                np.vstack(
                    [
                        np.column_stack(
                            [
                                row_rotation_w2c(row),
                                np.asarray(row["tvec"], dtype=np.float64),
                            ]
                        ),
                        np.asarray([0.0, 0.0, 0.0, 1.0]),
                    ]
                ).tolist()
            )
        frames.append(frame)
    payload = {
        "dataset": str(dataset.resolve()),
        "sparse": str(sparse),
        "sparse_files": sparse_files,
        "frames": frames,
    }
    if pose_completion_audit is not None:
        payload["pose_completion"] = pose_completion_audit
    payload["sha256"] = canonical_json_sha256(payload)
    return payload


def _load_reusable(destination: Path, source_sha256: str) -> dict[str, Any] | None:
    metadata = destination / "raw_cache.json"
    if not metadata.is_file():
        return None
    row = json.loads(metadata.read_text(encoding="utf-8"))
    if row.get("adapter_format") != ADAPTER_FORMAT:
        raise RuntimeError(f"stale CoarseModel raw cache format: {metadata}")
    if row.get("source_binding", {}).get("sha256") != source_sha256:
        raise RuntimeError(f"CoarseModel source binding changed: {metadata}")
    required = [Path(row["cache_npz"]), Path(row["images_dir"]), Path(row["masks_dir"])]
    if not all(path.exists() for path in required):
        raise RuntimeError(f"reusable CoarseModel cache is incomplete: {metadata}")
    return row


def build_dataset_cache(
    dataset: Path,
    *,
    sparse: Path,
    output_dir: Path,
    min_registered_pairs: int,
    resume: bool,
    include_all_images: bool = False,
    selected_frame_names: Sequence[str] | None = None,
    allow_empty_points: bool = False,
) -> tuple[dict[str, Any], bool]:
    object_id = safe_object_id(dataset.name)
    category = "coarsemodel_real"
    image_dir = find_image_dir(dataset)
    mask_dir = (dataset / "masks").resolve()
    if not mask_dir.is_dir():
        raise FileNotFoundError(f"missing foreground mask directory: {mask_dir}")
    cameras = parse_cameras(sparse / "cameras.txt")
    registered = parse_registered_images(sparse / "images.txt")
    points = parse_points_allow_empty(
        sparse / "points3D.txt", allow_empty=bool(allow_empty_points)
    )
    registered_pairs: list[tuple[dict[str, Any], Path, Path]] = []
    for row in registered:
        image = (image_dir / str(row["name"])).resolve()
        mask = find_mask(mask_dir, str(row["name"]))
        if image.is_file() and mask is not None:
            registered_pairs.append((row, image, mask))
    available_registered_pair_count = len(registered_pairs)
    if selected_frame_names:
        requested_names = [str(value) for value in selected_frame_names]
        if len(requested_names) != len(set(requested_names)):
            raise ValueError(f"{dataset.name}: selected frame names must be unique")
        pair_by_name = {str(row[0]["name"]): row for row in registered_pairs}
        missing = [name for name in requested_names if name not in pair_by_name]
        if missing:
            raise RuntimeError(
                f"{dataset.name}: selected registered RGB/mask frames are missing: {missing}"
            )
        registered_pairs = [pair_by_name[name] for name in requested_names]
    if len(registered_pairs) < int(min_registered_pairs):
        raise RuntimeError(
            f"{dataset.name}: only {len(registered_pairs)} registered RGB/mask pairs; "
            f"need {int(min_registered_pairs)}"
        )
    pose_completion_audit = None
    if include_all_images:
        image_paths = sorted(
            path.resolve()
            for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        all_pairs = []
        for image in image_paths:
            mask = find_mask(mask_dir, image.name)
            if mask is None:
                raise FileNotFoundError(f"missing mask for all-image input: {image.name}")
            all_pairs.append((image, mask))
        completed, pose_completion_audit = phone_pose_completion(
            dataset, registered, [image.name for image, _mask in all_pairs]
        )
        pairs = [(completed[image.name], image, mask) for image, mask in all_pairs]
    else:
        pairs = registered_pairs
        for row, _image, _mask in pairs:
            row["pose_source"] = "colmap_registered"
    source_binding = _source_binding(dataset, sparse, pairs, pose_completion_audit)
    destination = output_dir / "objects" / category / object_id
    reusable = _load_reusable(destination, str(source_binding["sha256"]))
    if reusable is not None:
        if not resume:
            raise RuntimeError(f"output already exists; pass --resume: {destination}")
        return reusable, True
    if destination.exists():
        raise RuntimeError(f"partial CoarseModel raw cache exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / f".{object_id}.building"
    if staging.exists():
        if not resume:
            raise RuntimeError(f"partial CoarseModel staging exists: {staging}")
        shutil.rmtree(staging)
    images_out = staging / "images"
    masks_out = staging / "masks"
    images_out.mkdir(parents=True)
    masks_out.mkdir(parents=True)

    frame_names = []
    source_frame_names = []
    matrices = []
    intrinsics = []
    image_ids = []
    camera_ids = []
    camera_rows = []
    for index, (row, image_path, mask_path) in enumerate(pairs):
        normalized_name = f"view_{index:04d}.png"
        with Image.open(image_path) as handle:
            image = handle.convert("RGB")
            width, height = image.size
            image.save(images_out / normalized_name, format="PNG")
        with Image.open(mask_path) as handle:
            mask = handle.convert("L")
            if mask.size != (width, height):
                mask = mask.resize((width, height), resample=Image.Resampling.NEAREST)
            mask.save(masks_out / normalized_name, format="PNG")
        camera = cameras[int(row["camera_id"])]
        K, distortion = camera_intrinsics(camera)
        if int(camera["width"]) != width or int(camera["height"]) != height:
            K[0, :] *= float(width) / float(camera["width"])
            K[1, :] *= float(height) / float(camera["height"])
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = row_rotation_w2c(row)
        transform[:3, 3] = row["tvec"]
        frame_names.append(normalized_name)
        source_frame_names.append(str(row["name"]))
        matrices.append(transform)
        intrinsics.append(K)
        image_ids.append(int(row["image_id"]))
        camera_ids.append(int(row["camera_id"]))
        camera_rows.append(
            {
                "frame_name": normalized_name,
                "source_frame_name": str(row["name"]),
                "camera_id": int(row["camera_id"]),
                "model": str(camera["model"]),
                "width": int(width),
                "height": int(height),
                "params": [float(value) for value in camera["params"]],
                "distortion": distortion,
                "pose_source": str(row["pose_source"]),
                "pose_completion": row.get("pose_completion"),
            }
        )

    cache_name = "raw_camera_point_cache.npz"
    write_npz(
        staging / cache_name,
        frame_name=np.asarray(frame_names),
        source_frame_name=np.asarray(source_frame_names),
        image_id=np.asarray(image_ids, dtype=np.int64),
        camera_id=np.asarray(camera_ids, dtype=np.int64),
        K=np.asarray(intrinsics, dtype=np.float64),
        T_W2C=np.asarray(matrices, dtype=np.float64),
        P_W=points["xyz"],
        point_id=points["point_id"],
        point_rgb=points["rgb"],
        point_reprojection_error=points["reprojection_error"],
        point_track_length=points["track_length"],
        point_confidence_proxy=points["confidence_proxy"],
    )
    final_images = destination / "images"
    final_masks = destination / "masks"
    final_cache = destination / cache_name
    ref = reference_mesh(dataset)
    legacy = legacy_reconstruction(dataset)
    legacy_video = legacy_reconstruction_video(dataset)
    row = {
        "format": OBJECT_CACHE_FORMAT,
        "adapter_format": ADAPTER_FORMAT,
        "created_at_utc": utc_now(),
        "category": category,
        "object_id": object_id,
        "object_key": f"{category}:{object_id}",
        "object_root": str(destination.resolve()),
        "source_dataset": str(dataset.resolve()),
        "images_dir": str(final_images.resolve()),
        "masks_dir": str(final_masks.resolve()),
        "authoritative_colmap_dir": str(sparse),
        "cache_npz": str(final_cache.resolve()),
        "registered_pair_count": len(registered_pairs),
        "available_registered_pair_count": available_registered_pair_count,
        "selected_source_frame_names": (
            None
            if selected_frame_names is None
            else [str(value) for value in selected_frame_names]
        ),
        "input_view_count": len(frame_names),
        "phone_pose_augmented_count": sum(
            row[0]["pose_source"] == "phone_pose_local_colmap_completion" for row in pairs
        ),
        "pose_completion_audit": pose_completion_audit,
        "sparse_point_count": int(len(points["xyz"])),
        "point_cloud_consumed": bool(len(points["xyz"])),
        "geometry_availability": (
            "camera_pose_intrinsics_mask_only"
            if not len(points["xyz"])
            else "camera_pose_intrinsics_mask_and_sparse_points"
        ),
        "camera_models": sorted({str(camera["model"]) for camera in camera_rows}),
        "cameras": camera_rows,
        "source_binding": source_binding,
        "reference_mesh": None if ref is None else str(ref),
        "legacy_reconviagen_mesh": None if legacy is None else str(legacy),
        "legacy_reconviagen_video": (
            None if legacy_video is None else str(legacy_video)
        ),
        "reference_mesh_role": "visual_reference_only_never_consumed_by_model",
        "coordinate_policy": (
            "T_W2C stays in the selected sparse world frame. Point-mask mode derives "
            "runtime-O from masks and P_W; pose-mask mode requires P_W to be empty and "
            "derives runtime-O from calibrated poses, intrinsics, and masks only."
        ),
        "training_ready": False,
        "passed": True,
    }
    write_json(staging / "raw_cache.json", row)
    staging.replace(destination)
    return row, False


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", action="append", default=[])
    parser.add_argument("--sparse_override", action="append", default=[])
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--min_registered_pairs", type=int, default=8)
    parser.add_argument("--include_all_images", action="store_true")
    parser.add_argument(
        "--frame_name",
        action="append",
        help="optional exact registered source frame; repeat to freeze order",
    )
    parser.add_argument("--allow_missing", action="store_true")
    parser.add_argument(
        "--allow_empty_points",
        action="store_true",
        help="accept a camera-only COLMAP text model for the pose+mask runtime",
    )
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> None:
    args = make_parser().parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    datasets = [Path(value).expanduser().resolve() for value in (args.dataset or DEFAULT_DATASETS)]
    overrides = parse_overrides(args.sparse_override)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    rejections = []
    reused = []
    for index, dataset in enumerate(datasets, start=1):
        name = safe_object_id(dataset.name)
        print(f"[coarsemodel_raw] {index}/{len(datasets)} dataset={name}", flush=True)
        try:
            if not dataset.is_dir():
                raise FileNotFoundError(f"dataset directory is missing: {dataset}")
            sparse = resolve_sparse(
                dataset,
                overrides.get(name),
                allow_empty_points=bool(args.allow_empty_points),
            )
            row, was_reused = build_dataset_cache(
                dataset,
                sparse=sparse,
                output_dir=output_dir,
                min_registered_pairs=int(args.min_registered_pairs),
                resume=bool(args.resume),
                include_all_images=bool(args.include_all_images),
                selected_frame_names=args.frame_name,
                allow_empty_points=bool(args.allow_empty_points),
            )
            rows.append(row)
            if was_reused:
                reused.append(row["object_key"])
            print(
                f"[coarsemodel_raw] accepted={row['object_key']} "
                f"views={row.get('input_view_count', row['registered_pair_count'])} "
                f"registered={row['registered_pair_count']} "
                f"completed={row.get('phone_pose_augmented_count', 0)} "
                f"points={row['sparse_point_count']} "
                f"sparse={row['authoritative_colmap_dir']} reused={was_reused}",
                flush=True,
            )
        except (FileNotFoundError, RuntimeError, ValueError) as error:
            rejection = {
                "dataset": str(dataset),
                "dataset_name": name,
                "reason": repr(error),
            }
            rejections.append(rejection)
            print(f"[coarsemodel_raw] rejected={name}: {error}", flush=True)
            if not args.allow_missing:
                raise
    report = {
        "format": RAW_CACHE_FORMAT,
        "adapter_format": ADAPTER_FORMAT,
        "created_at_utc": utc_now(),
        "output_dir": str(output_dir),
        "requested_dataset_count": len(datasets),
        "category_count": 1 if rows else 0,
        "object_count": len(rows),
        "reused_objects": reused,
        "objects": rows,
        "rejections": rejections,
        "authoritative_sparse_policy": list(AUTO_SPARSE_CANDIDATES),
        "alignment_passed": False,
        "training_ready": False,
        "scope_guard": (
            "Input-only adapter for local CoarseModel captures. Existing reference "
            "and ReconViaGen meshes are metadata-only and are not model inputs or GT."
        ),
        "passed": bool(rows),
    }
    report_path = output_dir / "raw_cache_report.json"
    write_json(report_path, report)
    print(json.dumps({
        "passed": report["passed"],
        "requested": len(datasets),
        "accepted": len(rows),
        "rejected": len(rejections),
        "report": str(report_path),
    }, indent=2, ensure_ascii=False))
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
