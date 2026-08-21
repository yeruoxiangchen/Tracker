#!/usr/bin/env python3
"""Refine a frozen runtime ``T_O2W`` from its original RGB/mask/camera views.

The Mesh geometry remains in the native TRELLIS runtime-O frame.  This module
only estimates one bounded physical similarity from Mesh-O to the capture
world/A0 frame.  It has two entry points:

* :func:`run_o2w_refinement`, used immediately after Mesh inference by the
  phone server;
* the command line interface, which can resolve and refine an already finished
  reconstruction without running DINO, SS, SLat or the decoder again.

Observed masks come from the immutable reconstruction input cache.  Mesh
projections are predictions, never mask prompts or target replacements.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw
import trimesh

from manual_mesh_reconstruction.alignment_refinement import (
    object_centered_spherical_farthest_indices,
)
from manual_mesh_reconstruction.canonicalization import (
    array_sha256,
    similarity_scale,
    validate_proper_similarity,
)
from manual_mesh_reconstruction.common import (
    atomic_json,
    atomic_npz,
    canonical_sha256,
    load_json,
    sha256_file,
)
from manual_mesh_reconstruction.mesh_coordinates import (
    validate_runtime_o_mesh_frame_contract,
)
from manual_mesh_reconstruction.projection import (
    boundary,
    load_meshes,
    make_headless_raster_context,
    rasterize_world_silhouette,
    world_mesh_buffers,
)


FORMAT = "manual_mesh_reconstruction.input_mask_o2w_refinement.v1"
TRANSFORM_FORMAT = "manual_mesh_reconstruction.selected_o2w.v1"
DEFAULT_MAX_OPTIMIZATION_VIEWS = 24
DEFAULT_MIN_OPTIMIZATION_VIEWS = 8
DEFAULT_ITERATIONS = 40
DEFAULT_RENDER_LONG_SIDE = 160
DEFAULT_BATCH_SIZE = 4
DEFAULT_OPTIMIZATION_FACE_LIMIT = 30_000
DEFAULT_MAX_ROTATION_DEGREES = 25.0
DEFAULT_MIN_SCALE_RATIO = 0.75
DEFAULT_MAX_SCALE_RATIO = 1.25
DEFAULT_MAX_TRANSLATION_SCALE_RATIO = 0.75
MIN_MASK_RATIO = 0.001
MAX_MASK_RATIO = 0.80
OBSERVED_CONTOUR_RGB = np.asarray([255, 0, 255], dtype=np.uint8)
INITIAL_CONTOUR_RGB = np.asarray([255, 165, 0], dtype=np.uint8)
SELECTED_CONTOUR_RGB = np.asarray([0, 255, 255], dtype=np.uint8)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _frame_asset(directory: Path, frame_name: str) -> Path:
    exact = directory / frame_name
    if exact.is_file():
        return exact.resolve()
    matches = sorted(
        value
        for value in directory.glob(f"{Path(frame_name).stem}.*")
        if value.is_file()
    )
    if len(matches) != 1:
        raise FileNotFoundError(
            f"no unique asset for frame={frame_name} directory={directory}"
        )
    return matches[0].resolve()


def _one_object(rows: Sequence[Mapping[str, Any]], object_key: str) -> dict[str, Any]:
    selected = [dict(row) for row in rows if str(row.get("object_key")) == object_key]
    if len(selected) != 1:
        raise RuntimeError(
            f"expected one object={object_key}, found={len(selected)}"
        )
    return selected[0]


def _camera_metadata(raw_row: Mapping[str, Any], frame_names: Sequence[str]) -> list[dict[str, Any]]:
    indexed = {str(row["frame_name"]): dict(row) for row in raw_row["cameras"]}
    cameras: list[dict[str, Any]] = []
    for name in frame_names:
        if name not in indexed:
            raise RuntimeError(f"raw camera metadata is missing frame={name}")
        source = indexed[name]
        model = str(source["model"]).upper()
        if model not in {"PINHOLE", "SIMPLE_PINHOLE", "SIMPLE_RADIAL"}:
            raise RuntimeError(f"unsupported camera model for O2W optimization: {model}")
        distortion = [float(value) for value in source.get("distortion", [])]
        if model == "SIMPLE_RADIAL" and len(distortion) != 1:
            raise RuntimeError("SIMPLE_RADIAL requires exactly one coefficient")
        if model != "SIMPLE_RADIAL" and any(abs(value) > 1.0e-15 for value in distortion):
            raise RuntimeError(f"{model} unexpectedly carries distortion={distortion}")
        cameras.append(
            {
                "model": model,
                "distortion": distortion,
                "width": int(source["width"]),
                "height": int(source["height"]),
            }
        )
    return cameras


def load_all_view_contract(
    runtime_input_manifest: Path,
    *,
    object_key: str = "",
) -> dict[str, Any]:
    """Load the complete pre-selection RGB/mask/K/T_W2C domain."""

    runtime_path = runtime_input_manifest.expanduser().resolve(strict=True)
    runtime = load_json(runtime_path)
    if runtime.get("passed") is not True:
        raise RuntimeError(f"runtime manifest did not pass: {runtime_path}")
    rows = [dict(row) for row in runtime.get("objects", [])]
    if object_key:
        row = _one_object(rows, object_key)
    elif len(rows) == 1:
        row = rows[0]
        object_key = str(row["object_key"])
    else:
        raise RuntimeError("--object is required for a multi-object runtime manifest")

    runtime_cache = Path(row["cache_npz"]).expanduser().resolve(strict=True)
    raw_cache = Path(row["source_raw_cache"]).expanduser().resolve(strict=True)
    if row.get("source_raw_cache_sha256") not in {None, sha256_file(raw_cache)}:
        raise RuntimeError("source raw-cache binding changed")
    with np.load(runtime_cache, allow_pickle=False) as payload:
        initial_T_O2W = validate_proper_similarity(
            np.asarray(payload["T_O2W"], dtype=np.float64), name="initial_T_O2W"
        )
        stored_T_W2O = np.asarray(payload["T_W2O"], dtype=np.float64)
    inverse_error = float(
        np.max(np.abs(initial_T_O2W @ stored_T_W2O - np.eye(4)))
    )
    if inverse_error > 1.0e-7:
        raise RuntimeError(f"runtime T_O2W/T_W2O roundtrip failed: {inverse_error}")

    raw_report_value = (
        runtime.get("source_raw_cache_report")
        or runtime.get("raw_cache_report")
        or raw_cache.parents[3] / "raw_cache_report.json"
    )
    raw_report_path = Path(raw_report_value).expanduser().resolve(strict=True)
    expected_raw_report_hash = (
        runtime.get("source_raw_cache_report_sha256")
        or runtime.get("raw_cache_report_sha256")
    )
    if expected_raw_report_hash and sha256_file(raw_report_path) != str(
        expected_raw_report_hash
    ):
        raise RuntimeError("raw-cache report binding changed")
    raw_report = load_json(raw_report_path)
    raw_row = _one_object(raw_report.get("objects", []), object_key)
    if Path(raw_row["cache_npz"]).resolve(strict=True) != raw_cache:
        raise RuntimeError("runtime object and raw report bind different raw caches")

    with np.load(raw_cache, allow_pickle=False) as payload:
        frame_names = np.asarray(payload["frame_name"]).astype(str).tolist()
        intrinsics = np.asarray(payload["K"], dtype=np.float64)
        cameras_w2c = np.asarray(payload["T_W2C"], dtype=np.float64)
        source_frame_names = (
            np.asarray(payload["source_frame_name"]).astype(str).tolist()
            if "source_frame_name" in payload
            else list(frame_names)
        )
        source_indices = (
            np.asarray(payload["source_frame_index"], dtype=np.int64).tolist()
            if "source_frame_index" in payload
            else list(range(len(frame_names)))
        )
    view_count = len(frame_names)
    if intrinsics.shape != (view_count, 3, 3):
        raise RuntimeError("all-view intrinsic count/shape differs")
    if cameras_w2c.shape != (view_count, 4, 4):
        raise RuntimeError("all-view pose count/shape differs")
    if int(row.get("all_input_view_count", view_count)) != view_count:
        raise RuntimeError("runtime all_input_view_count differs from raw cache")

    image_dir = Path(raw_row["images_dir"]).expanduser().resolve(strict=True)
    mask_dir = Path(raw_row["masks_dir"]).expanduser().resolve(strict=True)
    images = [_frame_asset(image_dir, name) for name in frame_names]
    masks = [_frame_asset(mask_dir, name) for name in frame_names]
    camera_metadata = _camera_metadata(raw_row, frame_names)
    mask_arrays: list[np.ndarray] = []
    view_rows: list[dict[str, Any]] = []
    valid_indices: list[int] = []
    for index, (name, image_path, mask_path, camera) in enumerate(
        zip(frame_names, images, masks, camera_metadata)
    ):
        with Image.open(image_path) as image:
            image_size = image.size
        with Image.open(mask_path) as mask_image:
            mask = np.asarray(mask_image.convert("L"), dtype=np.uint8) > 127
            mask_size = mask_image.size
        expected_size = (camera["width"], camera["height"])
        if image_size != expected_size or mask_size != expected_size:
            raise RuntimeError(
                f"RGB/mask/camera size mismatch frame={name}: "
                f"rgb={image_size} mask={mask_size} camera={expected_size}"
            )
        ratio = float(mask.mean())
        valid = bool(
            int(mask.sum()) >= 64
            and MIN_MASK_RATIO <= ratio <= MAX_MASK_RATIO
            and np.isfinite(intrinsics[index]).all()
            and np.isfinite(cameras_w2c[index]).all()
        )
        if valid:
            valid_indices.append(index)
        mask_arrays.append(mask)
        view_rows.append(
            {
                "view_index": index,
                "frame_name": name,
                "source_frame_name": source_frame_names[index],
                "source_frame_index": int(source_indices[index]),
                "image": str(image_path),
                "mask": str(mask_path),
                "mask_foreground_pixels": int(mask.sum()),
                "mask_foreground_ratio": ratio,
                "valid_for_optimization": valid,
                "camera_model": camera["model"],
                "camera_distortion": camera["distortion"],
            }
        )
    if len(valid_indices) < DEFAULT_MIN_OPTIMIZATION_VIEWS:
        raise RuntimeError(
            "too few valid all-view masks for O2W refinement: "
            f"{len(valid_indices)} < {DEFAULT_MIN_OPTIMIZATION_VIEWS}"
        )
    return {
        "runtime_input_manifest": runtime_path,
        "runtime_object": row,
        "object_key": object_key,
        "runtime_cache": runtime_cache,
        "raw_cache": raw_cache,
        "raw_cache_report": raw_report_path,
        "raw_row": raw_row,
        "frame_names": frame_names,
        "source_frame_names": source_frame_names,
        "source_indices": source_indices,
        "images": images,
        "masks": masks,
        "mask_arrays": mask_arrays,
        "intrinsics": intrinsics,
        "T_W2C": cameras_w2c,
        "cameras": camera_metadata,
        "initial_T_O2W": initial_T_O2W,
        "valid_indices": valid_indices,
        "view_rows": view_rows,
        "bindings": {
            "runtime_input_manifest_sha256": sha256_file(runtime_path),
            "runtime_cache_sha256": sha256_file(runtime_cache),
            "raw_cache_sha256": sha256_file(raw_cache),
            "raw_cache_report_sha256": sha256_file(raw_report_path),
            "initial_T_O2W_sha256": array_sha256(initial_T_O2W),
            "all_T_W2C_sha256": array_sha256(cameras_w2c),
            "all_K_sha256": array_sha256(intrinsics),
            "frame_names_sha256": canonical_sha256(frame_names),
        },
    }


def mesh_o_buffers(mesh_path: Path) -> tuple[np.ndarray, np.ndarray]:
    return world_mesh_buffers(mesh_path, np.eye(4, dtype=np.float64))


def build_optimization_proxy(
    mesh_path: Path,
    *,
    face_limit: int = DEFAULT_OPTIMIZATION_FACE_LIMIT,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Build a bounded triangle proxy used only for differentiable rendering.

    The decoded Mesh remains the sole geometry used for full-resolution gates,
    exported assets, and phone display.  Quadric decimation only reduces the
    cost of repeated silhouette gradients; a candidate transform must still
    improve projections rendered from the exact decoded Mesh before deployment.
    """

    exact_vertices, exact_faces = mesh_o_buffers(mesh_path)
    limit = int(face_limit)
    if limit < 1_000:
        raise ValueError("optimization_face_limit must be at least 1000")
    simplified = len(exact_faces) > limit
    if simplified:
        try:
            import open3d as o3d
        except ImportError as error:
            raise RuntimeError(
                "Open3D is required to simplify this Mesh for O2W optimization"
            ) from error
        source = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(exact_vertices.astype(np.float64)),
            o3d.utility.Vector3iVector(exact_faces.astype(np.int32)),
        )
        proxy = source.simplify_quadric_decimation(
            target_number_of_triangles=limit,
            boundary_weight=10.0,
        )
        proxy.remove_degenerate_triangles()
        proxy.remove_duplicated_triangles()
        proxy.remove_unreferenced_vertices()
        vertices = np.asarray(proxy.vertices, dtype=np.float32).copy()
        faces = np.asarray(proxy.triangles, dtype=np.int32).copy()
    else:
        vertices = np.asarray(exact_vertices, dtype=np.float32)
        faces = np.asarray(exact_faces, dtype=np.int32)
    if (
        vertices.ndim != 2
        or vertices.shape[1] != 3
        or faces.ndim != 2
        or faces.shape[1] != 3
        or len(vertices) < 3
        or len(faces) < 1
        or not np.isfinite(vertices).all()
        or int(faces.min()) < 0
        or int(faces.max()) >= len(vertices)
    ):
        raise RuntimeError("O2W optimization proxy Mesh is invalid")
    return vertices, faces, {
        "policy": "open3d_quadric_decimation_boundary_weight10"
        if simplified
        else "exact_mesh_below_face_limit",
        "used_only_for_differentiable_optimization": True,
        "exact_mesh_used_for_full_resolution_acceptance_export_and_phone": True,
        "face_limit": limit,
        "exact_vertex_count": int(len(exact_vertices)),
        "exact_face_count": int(len(exact_faces)),
        "proxy_vertex_count": int(len(vertices)),
        "proxy_face_count": int(len(faces)),
        "proxy_vertices_sha256": array_sha256(vertices),
        "proxy_faces_sha256": array_sha256(faces),
    }


def compose_o2w_candidate(
    initial_T_O2W: np.ndarray,
    rotation_delta: np.ndarray,
    translation_delta: Sequence[float],
    scale_ratio: float,
) -> np.ndarray:
    """Compose around the O origin without scaling the world translation."""

    initial = validate_proper_similarity(initial_T_O2W, name="initial_T_O2W")
    delta_rotation = np.asarray(rotation_delta, dtype=np.float64)
    if delta_rotation.shape != (3, 3) or not np.allclose(
        delta_rotation.T @ delta_rotation, np.eye(3), atol=1.0e-6
    ) or float(np.linalg.det(delta_rotation)) <= 0.0:
        raise ValueError("rotation_delta must be a proper 3x3 rotation")
    translation = np.asarray(translation_delta, dtype=np.float64)
    if translation.shape != (3,) or not np.isfinite(translation).all():
        raise ValueError("translation_delta must be finite [3]")
    ratio = float(scale_ratio)
    if not math.isfinite(ratio) or ratio <= 0.0:
        raise ValueError("scale_ratio must be positive")
    output = np.eye(4, dtype=np.float64)
    output[:3, :3] = ratio * delta_rotation @ initial[:3, :3]
    output[:3, 3] = initial[:3, 3] + translation
    return validate_proper_similarity(output, name="candidate_T_O2W")


def _axis_angle_matrix(vector):
    import torch

    x, y, z = vector.unbind()
    zero = torch.zeros((), device=vector.device, dtype=vector.dtype)
    skew = torch.stack([zero, -z, y, z, zero, -x, -y, x, zero]).reshape(3, 3)
    return torch.matrix_exp(skew)


def _bounded_parameters(
    raw,
    *,
    maximum_angle_radians: float,
    maximum_translation_meters: float,
    minimum_scale_ratio: float,
    maximum_scale_ratio: float,
):
    import torch

    rotation_raw = raw[:3]
    rotation_norm = torch.linalg.vector_norm(rotation_raw)
    rotation_vector = (
        float(maximum_angle_radians)
        * rotation_raw
        / (1.0 + rotation_norm)
    )
    rotation = _axis_angle_matrix(rotation_vector)
    translation_raw = raw[3:6]
    translation_norm = torch.linalg.vector_norm(translation_raw)
    translation = (
        float(maximum_translation_meters)
        * translation_raw
        / (1.0 + translation_norm)
    )
    signed = torch.tanh(raw[6])
    minimum_log = math.log(float(minimum_scale_ratio))
    maximum_log = math.log(float(maximum_scale_ratio))
    log_ratio = torch.where(
        signed >= 0.0,
        maximum_log * signed,
        (-minimum_log) * signed,
    )
    return rotation, translation, torch.exp(log_ratio), rotation_vector


def _render_soft_batch(
    vertices_world,
    faces,
    cameras,
    intrinsics,
    radial_k1,
    *,
    height: int,
    width: int,
    context,
):
    import nvdiffrast.torch as dr
    import torch

    batch = int(cameras.shape[0])
    vertices_h = torch.cat(
        [
            vertices_world,
            torch.ones(
                (len(vertices_world), 1),
                dtype=vertices_world.dtype,
                device=vertices_world.device,
            ),
        ],
        dim=1,
    )
    camera_vertices_h = torch.matmul(
        vertices_h.unsqueeze(0), cameras.transpose(1, 2)
    )
    camera_vertices = camera_vertices_h[..., :3]
    depth = camera_vertices[..., 2].clamp(min=1.0e-4)
    x = camera_vertices[..., 0] / depth
    y = camera_vertices[..., 1] / depth
    radial = 1.0 + radial_k1[:, None] * (x * x + y * y)
    x = x * radial
    y = y * radial
    u = (
        intrinsics[:, None, 0, 0] * x
        + intrinsics[:, None, 0, 1] * y
        + intrinsics[:, None, 0, 2]
    )
    v = intrinsics[:, None, 1, 1] * y + intrinsics[:, None, 1, 2]
    x_ndc = 2.0 * u / max(int(width) - 1, 1) - 1.0
    y_ndc = 2.0 * v / max(int(height) - 1, 1) - 1.0
    near, far = 0.005, 20.0
    z_ndc = 2.0 * (depth - near) / (far - near) - 1.0
    clip = torch.stack(
        [x_ndc * depth, y_ndc * depth, z_ndc * depth, depth], dim=-1
    )
    rast, _ = dr.rasterize(
        context, clip, faces, resolution=(int(height), int(width))
    )
    attributes = torch.ones(
        (batch, len(vertices_world), 1),
        dtype=vertices_world.dtype,
        device=vertices_world.device,
    )
    coverage, _ = dr.interpolate(attributes, rast, faces)
    return dr.antialias(coverage, rast, clip, faces)[..., 0].clamp(0.0, 1.0)


def _summary(values: Sequence[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "count": int(len(array)),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "positive_rate": float(np.mean(array > 0.0)),
    }


def optimize_o2w_similarity(
    *,
    vertices_o: np.ndarray,
    faces: np.ndarray,
    initial_T_O2W: np.ndarray,
    cameras_w2c: Sequence[np.ndarray],
    intrinsics: Sequence[np.ndarray],
    camera_models: Sequence[Mapping[str, Any]],
    masks: Sequence[np.ndarray],
    training_view_indices: Sequence[int],
    validation_view_indices: Sequence[int],
    iterations: int = DEFAULT_ITERATIONS,
    render_long_side: int = DEFAULT_RENDER_LONG_SIDE,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_rotation_degrees: float = DEFAULT_MAX_ROTATION_DEGREES,
    min_scale_ratio: float = DEFAULT_MIN_SCALE_RATIO,
    max_scale_ratio: float = DEFAULT_MAX_SCALE_RATIO,
    max_translation_scale_ratio: float = DEFAULT_MAX_TRANSLATION_SCALE_RATIO,
) -> dict[str, Any]:
    """Optimize a bounded 7-DoF O-to-world similarity with held-out views."""

    import torch
    import nvdiffrast.torch as dr

    view_count = len(cameras_w2c)
    if view_count < DEFAULT_MIN_OPTIMIZATION_VIEWS or not (
        view_count
        == len(intrinsics)
        == len(camera_models)
        == len(masks)
    ):
        raise ValueError("O2W optimizer requires at least eight matched views")
    train = [int(value) for value in training_view_indices]
    heldout = [int(value) for value in validation_view_indices]
    expected = set(range(view_count))
    if (
        not train
        or not heldout
        or set(train) & set(heldout)
        or set(train) | set(heldout) != expected
    ):
        raise ValueError("train/heldout indices must be disjoint and cover all views")
    initial = validate_proper_similarity(initial_T_O2W, name="initial_T_O2W")
    initial_scale = similarity_scale(initial)
    maximum_translation = max(
        0.01, float(initial_scale) * float(max_translation_scale_ratio)
    )
    device = torch.device("cuda")
    base_vertices = torch.as_tensor(
        vertices_o, dtype=torch.float32, device=device
    )
    faces_tensor = torch.as_tensor(faces, dtype=torch.int32, device=device)
    initial_rotation = torch.as_tensor(
        initial[:3, :3] / initial_scale, dtype=torch.float32, device=device
    )
    initial_translation = torch.as_tensor(
        initial[:3, 3], dtype=torch.float32, device=device
    )

    prepared: list[dict[str, Any]] = []
    for camera, intrinsic, metadata, mask in zip(
        cameras_w2c, intrinsics, camera_models, masks
    ):
        binary = (np.asarray(mask) > 0).astype(np.float32)
        source_height, source_width = binary.shape
        resize_scale = min(
            1.0, float(render_long_side) / max(source_height, source_width)
        )
        height = max(48, int(round(source_height * resize_scale)))
        width = max(48, int(round(source_width * resize_scale)))
        target = cv2.resize(
            binary, (width, height), interpolation=cv2.INTER_NEAREST
        )
        outside_distance = cv2.distanceTransform(
            (target <= 0.5).astype(np.uint8), cv2.DIST_L2, 5
        ).astype(np.float32)
        outside_distance /= max(float(max(height, width)), 1.0)
        scaled_k = np.asarray(intrinsic, dtype=np.float64).copy()
        scaled_k[0, :] *= width / source_width
        scaled_k[1, :] *= height / source_height
        distortion = [float(value) for value in metadata.get("distortion", [])]
        radial_k1 = distortion[0] if str(metadata["model"]) == "SIMPLE_RADIAL" else 0.0
        prepared.append(
            {
                "camera": torch.as_tensor(camera, dtype=torch.float32, device=device),
                "intrinsic": torch.as_tensor(scaled_k, dtype=torch.float32, device=device),
                "target": torch.as_tensor(target, dtype=torch.float32, device=device),
                "outside_distance": torch.as_tensor(
                    outside_distance, dtype=torch.float32, device=device
                ),
                "radial_k1": float(radial_k1),
                "height": height,
                "width": width,
            }
        )
    context = dr.RasterizeCudaContext()
    raw = torch.zeros(7, dtype=torch.float32, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([raw], lr=0.06)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(int(iterations), 1), eta_min=0.006
    )
    maximum_angle = math.radians(float(max_rotation_degrees))

    def transform_from_raw(parameters):
        rotation_delta, translation_delta, scale_ratio, rotation_vector = (
            _bounded_parameters(
                parameters,
                maximum_angle_radians=maximum_angle,
                maximum_translation_meters=maximum_translation,
                minimum_scale_ratio=min_scale_ratio,
                maximum_scale_ratio=max_scale_ratio,
            )
        )
        rotation = rotation_delta @ initial_rotation
        linear = float(initial_scale) * scale_ratio * rotation
        translation = initial_translation + translation_delta
        vertices_world = base_vertices @ linear.T + translation
        return vertices_world, rotation_delta, translation_delta, scale_ratio, rotation_vector

    def prediction_vectors(indices: Sequence[int], parameters):
        vertices_world, *_rest = transform_from_raw(parameters)
        predictions: dict[int, Any] = {}
        grouped: dict[tuple[int, int], list[int]] = defaultdict(list)
        for index in indices:
            grouped[(prepared[index]["height"], prepared[index]["width"])].append(index)
        for (height, width), group in grouped.items():
            for offset in range(0, len(group), int(batch_size)):
                chunk = group[offset : offset + int(batch_size)]
                cameras = torch.stack([prepared[index]["camera"] for index in chunk])
                Ks = torch.stack([prepared[index]["intrinsic"] for index in chunk])
                k1 = torch.as_tensor(
                    [prepared[index]["radial_k1"] for index in chunk],
                    dtype=torch.float32,
                    device=device,
                )
                rendered = _render_soft_batch(
                    vertices_world,
                    faces_tensor,
                    cameras,
                    Ks,
                    k1,
                    height=height,
                    width=width,
                    context=context,
                )
                for local_index, view_index in enumerate(chunk):
                    predictions[view_index] = rendered[local_index]
        return [predictions[index] for index in indices]

    def objective(indices: Sequence[int], parameters):
        predictions = prediction_vectors(indices, parameters)
        losses = []
        soft_ious = []
        for index, prediction in zip(indices, predictions):
            target = prepared[index]["target"]
            intersection = (prediction * target).sum()
            union = prediction.sum() + target.sum() - intersection
            soft_iou = (intersection + 1.0) / (union + 1.0)
            outside = (
                prediction * prepared[index]["outside_distance"]
            ).sum() / prediction.sum().clamp(min=1.0)
            pixel_l1 = torch.abs(prediction - target).mean()
            losses.append((1.0 - soft_iou) + 0.15 * outside + 0.08 * pixel_l1)
            soft_ious.append(soft_iou)
        return torch.stack(losses), torch.stack(soft_ious)

    def hard_ious(indices: Sequence[int], parameters) -> list[float]:
        predictions = prediction_vectors(indices, parameters)
        result = []
        for index, prediction in zip(indices, predictions):
            target = prepared[index]["target"] > 0.5
            predicted = prediction > 0.5
            intersection = torch.logical_and(predicted, target).sum().float()
            union = torch.logical_or(predicted, target).sum().float()
            result.append(float((intersection / union.clamp(min=1)).detach().cpu()))
        return result

    with torch.no_grad():
        initial_train = hard_ious(train, raw)
        initial_heldout = hard_ious(heldout, raw)
    best_raw = raw.detach().clone()
    best_score = float(np.mean(initial_heldout) + 0.25 * np.mean(initial_train))
    history = [
        {
            "iteration": 0,
            "training_iou_mean": float(np.mean(initial_train)),
            "heldout_iou_mean": float(np.mean(initial_heldout)),
            "selection_score": best_score,
        }
    ]
    final_training_loss = None
    for iteration in range(1, int(iterations) + 1):
        optimizer.zero_grad(set_to_none=True)
        losses, _soft = objective(train, raw)
        _world, _rotation, translation, ratio, rotation_vector = transform_from_raw(raw)
        prior = (
            0.003 * (rotation_vector / maximum_angle).square().mean()
            + 0.003 * (translation / maximum_translation).square().mean()
            + 0.002 * torch.log(ratio).square()
        )
        loss = losses.mean() + prior
        loss.backward()
        torch.nn.utils.clip_grad_norm_([raw], 2.0)
        optimizer.step()
        scheduler.step()
        final_training_loss = float(loss.detach().cpu())
        if iteration % 5 == 0 or iteration == int(iterations):
            with torch.no_grad():
                train_iou = hard_ious(train, raw)
                heldout_iou = hard_ious(heldout, raw)
            score = float(np.mean(heldout_iou) + 0.25 * np.mean(train_iou))
            history.append(
                {
                    "iteration": iteration,
                    "training_iou_mean": float(np.mean(train_iou)),
                    "heldout_iou_mean": float(np.mean(heldout_iou)),
                    "selection_score": score,
                    "training_loss": final_training_loss,
                }
            )
            if score > best_score:
                best_score = score
                best_raw = raw.detach().clone()

    with torch.no_grad():
        candidate_train = hard_ious(train, best_raw)
        candidate_heldout = hard_ious(heldout, best_raw)
        (
            _vertices,
            candidate_rotation_delta,
            candidate_translation_delta,
            candidate_scale_ratio,
            candidate_rotation_vector,
        ) = transform_from_raw(best_raw)
    rotation_numpy = candidate_rotation_delta.detach().cpu().numpy().astype(np.float64)
    translation_numpy = (
        candidate_translation_delta.detach().cpu().numpy().astype(np.float64)
    )
    ratio_float = float(candidate_scale_ratio.detach().cpu())
    candidate_transform = compose_o2w_candidate(
        initial, rotation_numpy, translation_numpy, ratio_float
    )
    initial_all = initial_train + initial_heldout
    candidate_all = candidate_train + candidate_heldout
    gains = [after - before for before, after in zip(initial_all, candidate_all)]
    train_gain = float(np.mean(np.asarray(candidate_train) - np.asarray(initial_train)))
    heldout_gain = float(
        np.mean(np.asarray(candidate_heldout) - np.asarray(initial_heldout))
    )
    mean_gain = float(np.mean(gains))
    positive_rate = float(np.mean(np.asarray(gains) > 0.0))
    checks = {
        "training_iou_gain_positive": train_gain > 0.0,
        "heldout_iou_non_degrading": heldout_gain >= -0.002,
        "selected_view_iou_gain_ge_0p001": mean_gain >= 0.001,
        "view_positive_rate_ge_0p45": positive_rate >= 0.45,
    }
    accepted = all(checks.values())
    selected_transform = candidate_transform if accepted else initial
    torch.cuda.empty_cache()
    return {
        "passed": True,
        "accepted_by_optimization_subset": accepted,
        "iterations": int(iterations),
        "render_long_side": int(render_long_side),
        "batch_size": int(batch_size),
        "final_training_loss": final_training_loss,
        "training_view_indices": train,
        "heldout_view_indices": heldout,
        "initial_training_iou": initial_train,
        "candidate_training_iou": candidate_train,
        "initial_heldout_iou": initial_heldout,
        "candidate_heldout_iou": candidate_heldout,
        "training_iou_gain_mean": train_gain,
        "heldout_iou_gain_mean": heldout_gain,
        "selected_view_iou_gain_mean": mean_gain,
        "selected_view_positive_rate": positive_rate,
        "checks": checks,
        "history": history,
        "bounds": {
            "max_rotation_degrees": float(max_rotation_degrees),
            "max_translation_meters": maximum_translation,
            "min_scale_ratio": float(min_scale_ratio),
            "max_scale_ratio": float(max_scale_ratio),
        },
        "candidate_delta": {
            "rotation_axis_angle_degrees": np.degrees(
                candidate_rotation_vector.detach().cpu().numpy()
            ).tolist(),
            "rotation_angle_degrees": float(
                np.degrees(
                    np.linalg.norm(candidate_rotation_vector.detach().cpu().numpy())
                )
            ),
            "translation_world": translation_numpy.tolist(),
            "translation_norm_meters": float(np.linalg.norm(translation_numpy)),
            "scale_ratio": ratio_float,
        },
        "initial_T_O2W": initial.tolist(),
        "candidate_T_O2W": candidate_transform.tolist(),
        "selected_T_O2W_from_subset_gate": selected_transform.tolist(),
        "rejection_policy": "preserve_initial_T_O2W_if_any_heldout_gate_fails",
    }


def binary_iou(first: np.ndarray, second: np.ndarray) -> float:
    a = np.asarray(first, dtype=bool)
    b = np.asarray(second, dtype=bool)
    union = int(np.logical_or(a, b).sum())
    if union == 0:
        return 1.0
    return float(np.logical_and(a, b).sum() / union)


def _atomic_export_mesh(mesh: trimesh.Trimesh, destination: Path, file_type: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.stem}.tmp-{os.getpid()}{destination.suffix}"
    )
    mesh.export(temporary, file_type=file_type)
    os.replace(temporary, destination)


def _atomic_export_glb(mesh: trimesh.Trimesh, destination: Path) -> None:
    payload = trimesh.Scene(mesh).export(file_type="glb")
    if not isinstance(payload, (bytes, bytearray)):
        raise RuntimeError("trimesh GLB export returned non-binary data")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    temporary.write_bytes(bytes(payload))
    os.replace(temporary, destination)


def _export_selected_world_mesh(
    mesh_o: Path, T_O2W: np.ndarray, output_dir: Path
) -> dict[str, Any]:
    transformed = []
    for source in load_meshes(mesh_o):
        mesh = source.copy()
        mesh.apply_transform(T_O2W)
        transformed.append(mesh)
    world = trimesh.util.concatenate(transformed)
    if not len(world.vertices) or not len(world.faces):
        raise RuntimeError("selected O2W world Mesh is empty")
    obj = output_dir / "mesh_selected_o2w_world.obj"
    glb = output_dir / "mesh_selected_o2w_world.glb"
    _atomic_export_mesh(world, obj, "obj")
    _atomic_export_glb(world, glb)
    return {
        "obj": str(obj.resolve()),
        "obj_sha256": sha256_file(obj),
        "glb": str(glb.resolve()),
        "glb_sha256": sha256_file(glb),
        "vertex_count": int(len(world.vertices)),
        "face_count": int(len(world.faces)),
    }


def _save_overlay(
    image_path: Path,
    observed_mask: np.ndarray,
    mesh_mask: np.ndarray,
    destination: Path,
    *,
    mesh_color: np.ndarray,
    contour_width: int,
) -> Image.Image:
    with Image.open(image_path) as source:
        array = np.asarray(source.convert("RGB"), dtype=np.uint8).copy()
    observed = boundary(observed_mask, width=max(1, int(contour_width) - 1))
    predicted = boundary(mesh_mask, width=int(contour_width))
    array[observed] = OBSERVED_CONTOUR_RGB
    array[predicted] = mesh_color
    output = Image.fromarray(array, mode="RGB")
    destination.parent.mkdir(parents=True, exist_ok=True)
    output.save(destination)
    return output


def _contact_sheet_pages(
    images: Sequence[Path],
    labels: Sequence[str],
    output_dir: Path,
    *,
    prefix: str,
) -> list[str]:
    if len(images) != len(labels):
        raise ValueError("contact-sheet image/label count differs")
    output_dir.mkdir(parents=True, exist_ok=True)
    pages: list[str] = []
    per_page = 8
    columns, rows = 2, 4
    cell_width, cell_height, header = 480, 360, 30
    for page_index, offset in enumerate(range(0, len(images), per_page), start=1):
        page_images = images[offset : offset + per_page]
        page_labels = labels[offset : offset + per_page]
        sheet = Image.new(
            "RGB",
            (columns * cell_width, rows * (cell_height + header)),
            (20, 20, 20),
        )
        draw = ImageDraw.Draw(sheet)
        for local, (path, label) in enumerate(zip(page_images, page_labels)):
            with Image.open(path) as source:
                thumb = source.convert("RGB")
                thumb.thumbnail((cell_width, cell_height), Image.Resampling.LANCZOS)
            x = (local % columns) * cell_width
            y = (local // columns) * (cell_height + header)
            sheet.paste(
                thumb,
                (x + (cell_width - thumb.width) // 2, y + header),
            )
            draw.text((x + 8, y + 8), label, fill=(238, 238, 238))
        destination = output_dir / f"{prefix}_第{page_index:02d}页.png"
        sheet.save(destination)
        pages.append(str(destination.resolve()))
    return pages


def _render_all_view_audit(
    *,
    mesh_o: Path,
    contract: Mapping[str, Any],
    initial_T_O2W: np.ndarray,
    candidate_T_O2W: np.ndarray,
    output_dir: Path,
    contour_width: int,
) -> dict[str, Any]:
    initial_vertices, faces = world_mesh_buffers(mesh_o, initial_T_O2W)
    candidate_vertices, candidate_faces = world_mesh_buffers(mesh_o, candidate_T_O2W)
    if not np.array_equal(faces, candidate_faces):
        raise RuntimeError("initial/candidate Mesh topology differs")
    context = make_headless_raster_context()
    records: list[dict[str, Any]] = []
    initial_masks: dict[int, np.ndarray] = {}
    candidate_masks: dict[int, np.ndarray] = {}
    failures: list[dict[str, Any]] = []
    for index in contract["valid_indices"]:
        camera = contract["cameras"][index]
        image_path = contract["images"][index]
        with Image.open(image_path) as image:
            width, height = image.size
        try:
            initial_projection = rasterize_world_silhouette(
                initial_vertices,
                faces,
                contract["T_W2C"][index],
                contract["intrinsics"][index],
                camera,
                (height, width),
                context,
            )
            candidate_projection = rasterize_world_silhouette(
                candidate_vertices,
                faces,
                contract["T_W2C"][index],
                contract["intrinsics"][index],
                camera,
                (height, width),
                context,
            )
            if int(initial_projection.sum()) <= 0 or int(candidate_projection.sum()) <= 0:
                raise RuntimeError("empty projected silhouette")
        except Exception as error:
            failures.append(
                {
                    "view_index": int(index),
                    "frame_name": contract["frame_names"][index],
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
            continue
        observed = contract["mask_arrays"][index]
        initial_iou = binary_iou(initial_projection, observed)
        candidate_iou = binary_iou(candidate_projection, observed)
        initial_masks[index] = initial_projection
        candidate_masks[index] = candidate_projection
        records.append(
            {
                "view_index": int(index),
                "frame_name": contract["frame_names"][index],
                "source_frame_name": contract["source_frame_names"][index],
                "image": str(image_path),
                "mask": str(contract["masks"][index]),
                "initial_iou": initial_iou,
                "candidate_iou": candidate_iou,
                "candidate_iou_gain": candidate_iou - initial_iou,
                "initial_silhouette_pixels": int(initial_projection.sum()),
                "candidate_silhouette_pixels": int(candidate_projection.sum()),
            }
        )
    del context
    minimum_success = max(
        DEFAULT_MIN_OPTIMIZATION_VIEWS,
        int(math.ceil(0.75 * len(contract["valid_indices"]))),
    )
    if len(records) < minimum_success:
        raise RuntimeError(
            "too few full-resolution projection audits succeeded: "
            f"{len(records)} < {minimum_success}"
        )
    gains = [float(row["candidate_iou_gain"]) for row in records]
    full_checks = {
        "full_view_iou_gain_nonnegative": float(np.mean(gains)) >= 0.0,
        "full_view_positive_rate_ge_0p45": float(np.mean(np.asarray(gains) > 0))
        >= 0.45,
        "projection_success_rate_ge_0p75": len(records)
        >= int(math.ceil(0.75 * len(contract["valid_indices"]))),
    }
    return {
        "records": records,
        "failures": failures,
        "initial_masks": initial_masks,
        "candidate_masks": candidate_masks,
        "initial_iou": _summary([float(row["initial_iou"]) for row in records]),
        "candidate_iou": _summary([float(row["candidate_iou"]) for row in records]),
        "candidate_iou_gain": _summary(gains),
        "checks": full_checks,
        "passed_for_selection": all(full_checks.values()),
    }


def _write_all_view_overlays(
    *,
    contract: Mapping[str, Any],
    audit: dict[str, Any],
    selected_is_candidate: bool,
    output_dir: Path,
    contour_width: int,
) -> dict[str, Any]:
    initial_dir = output_dir / "01_初始O2W输入轮廓"
    selected_dir = output_dir / "02_优化后O2W输入轮廓"
    candidate_dir = output_dir / "03_未通过门的候选O2W轮廓"
    initial_paths: list[Path] = []
    selected_paths: list[Path] = []
    candidate_paths: list[Path] = []
    labels: list[str] = []
    row_by_index = {int(row["view_index"]): row for row in audit["records"]}
    for index in sorted(row_by_index):
        row = row_by_index[index]
        frame_name = str(row["frame_name"])
        safe_name = f"view_{index:04d}_{Path(frame_name).stem}.png"
        initial_path = initial_dir / safe_name
        _save_overlay(
            contract["images"][index],
            contract["mask_arrays"][index],
            audit["initial_masks"][index],
            initial_path,
            mesh_color=INITIAL_CONTOUR_RGB,
            contour_width=contour_width,
        )
        selected_mask = (
            audit["candidate_masks"][index]
            if selected_is_candidate
            else audit["initial_masks"][index]
        )
        selected_path = selected_dir / safe_name
        _save_overlay(
            contract["images"][index],
            contract["mask_arrays"][index],
            selected_mask,
            selected_path,
            mesh_color=SELECTED_CONTOUR_RGB,
            contour_width=contour_width,
        )
        row["initial_overlay"] = str(initial_path.resolve())
        row["initial_overlay_sha256"] = sha256_file(initial_path)
        row["selected_overlay"] = str(selected_path.resolve())
        row["selected_overlay_sha256"] = sha256_file(selected_path)
        if not selected_is_candidate:
            candidate_path = candidate_dir / safe_name
            _save_overlay(
                contract["images"][index],
                contract["mask_arrays"][index],
                audit["candidate_masks"][index],
                candidate_path,
                mesh_color=SELECTED_CONTOUR_RGB,
                contour_width=contour_width,
            )
            row["rejected_candidate_overlay"] = str(candidate_path.resolve())
            candidate_paths.append(candidate_path)
        initial_paths.append(initial_path)
        selected_paths.append(selected_path)
        labels.append(
            f"{index}: {frame_name}  IoU {row['initial_iou']:.3f}->{(row['candidate_iou'] if selected_is_candidate else row['initial_iou']):.3f}"
        )
    initial_pages = _contact_sheet_pages(
        initial_paths,
        labels,
        output_dir / "04_初始O2W总览",
        prefix="初始O2W轮廓",
    )
    selected_pages = _contact_sheet_pages(
        selected_paths,
        labels,
        output_dir / "05_优化后O2W总览",
        prefix="优化后O2W轮廓",
    )
    candidate_pages: list[str] = []
    if candidate_paths:
        candidate_pages = _contact_sheet_pages(
            candidate_paths,
            labels,
            output_dir / "06_被拒绝候选O2W总览",
            prefix="被拒绝候选O2W轮廓",
        )
    return {
        "legend": {
            "observed_mask_boundary_rgb": OBSERVED_CONTOUR_RGB.astype(int).tolist(),
            "initial_mesh_boundary_rgb": INITIAL_CONTOUR_RGB.astype(int).tolist(),
            "selected_mesh_boundary_rgb": SELECTED_CONTOUR_RGB.astype(int).tolist(),
        },
        "initial_pages": initial_pages,
        "selected_optimized_pages": selected_pages,
        "rejected_candidate_pages": candidate_pages,
        "initial_overview": initial_pages[0] if initial_pages else None,
        "selected_optimized_overview": selected_pages[0] if selected_pages else None,
    }


def _validate_mesh_binding(mesh_o: Path, mesh_frame_report: Path) -> dict[str, Any]:
    report = load_json(mesh_frame_report)
    if report.get("passed") is not True:
        raise RuntimeError("runtime-O Mesh frame report did not pass")
    if Path(str(report.get("mesh", ""))).expanduser().resolve(strict=True) != mesh_o:
        raise RuntimeError("Mesh frame report path differs from Mesh-O")
    if report.get("mesh_sha256") != sha256_file(mesh_o):
        raise RuntimeError("Mesh frame report hash differs from Mesh-O")
    validate_runtime_o_mesh_frame_contract(report)
    return report


def run_o2w_refinement(
    *,
    runtime_input_manifest: Path,
    mesh_o: Path,
    mesh_frame_report: Path,
    output_dir: Path,
    object_key: str = "",
    max_optimization_views: int = DEFAULT_MAX_OPTIMIZATION_VIEWS,
    iterations: int = DEFAULT_ITERATIONS,
    render_long_side: int = DEFAULT_RENDER_LONG_SIDE,
    batch_size: int = DEFAULT_BATCH_SIZE,
    optimization_face_limit: int = DEFAULT_OPTIMIZATION_FACE_LIMIT,
    contour_width: int = 3,
    resume: bool = False,
) -> dict[str, Any]:
    runtime_path = runtime_input_manifest.expanduser().resolve(strict=True)
    mesh_path = mesh_o.expanduser().resolve(strict=True)
    mesh_report_path = mesh_frame_report.expanduser().resolve(strict=True)
    output = output_dir.expanduser().resolve()
    configuration = {
        "max_optimization_views": int(max_optimization_views),
        "iterations": int(iterations),
        "render_long_side": int(render_long_side),
        "batch_size": int(batch_size),
        "optimization_face_limit": int(optimization_face_limit),
        "contour_width": int(contour_width),
    }
    if int(max_optimization_views) != 0 and int(max_optimization_views) < 8:
        raise ValueError("max_optimization_views must be zero or at least eight")
    if int(iterations) <= 0 or int(render_long_side) < 96 or int(batch_size) <= 0:
        raise ValueError("invalid O2W optimization iteration/render/batch setting")
    if int(optimization_face_limit) < 1_000:
        raise ValueError("optimization_face_limit must be at least 1000")
    if int(contour_width) < 1:
        raise ValueError("contour_width must be positive")
    expected_binding = {
        "runtime_input_manifest_sha256": sha256_file(runtime_path),
        "mesh_o_sha256": sha256_file(mesh_path),
        "mesh_frame_report_sha256": sha256_file(mesh_report_path),
        "configuration_sha256": canonical_sha256(configuration),
    }
    report_path = output / "report.json"
    if report_path.is_file():
        existing = load_json(report_path)
        selected_npz = Path(str(existing.get("selected_T_O2W_npz", "")))
        if (
            resume
            and existing.get("format") == FORMAT
            and existing.get("passed") is True
            and existing.get("bindings") == expected_binding
            and selected_npz.is_file()
            and existing.get("selected_T_O2W_npz_sha256") == sha256_file(selected_npz)
        ):
            existing["reused"] = True
            return existing
        raise RuntimeError(f"stale O2W refinement output: {output}")
    if output.exists() and not resume:
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    _validate_mesh_binding(mesh_path, mesh_report_path)
    contract = load_all_view_contract(runtime_path, object_key=object_key)
    object_key = str(contract["object_key"])
    exact_vertices_o, _exact_faces = mesh_o_buffers(mesh_path)
    vertices_o, faces, optimization_proxy = build_optimization_proxy(
        mesh_path,
        face_limit=int(optimization_face_limit),
    )
    proxy_path = output / "optimization_proxy_mesh_o.obj"
    _atomic_export_mesh(
        trimesh.Trimesh(vertices=vertices_o, faces=faces, process=False),
        proxy_path,
        "obj",
    )
    optimization_proxy.update(
        {
            "mesh": str(proxy_path.resolve()),
            "mesh_sha256": sha256_file(proxy_path),
        }
    )
    initial = contract["initial_T_O2W"]
    initial_vertices = (
        exact_vertices_o @ initial[:3, :3].T + initial[:3, 3][None]
    )
    object_center = 0.5 * (
        np.min(initial_vertices, axis=0) + np.max(initial_vertices, axis=0)
    )
    valid_indices = list(contract["valid_indices"])
    valid_cameras = [contract["T_W2C"][index] for index in valid_indices]
    limit = (
        len(valid_indices)
        if int(max_optimization_views) == 0
        else min(len(valid_indices), int(max_optimization_views))
    )
    fps_local, selection = object_centered_spherical_farthest_indices(
        valid_cameras, object_center, limit
    )
    optimization_source_indices = [valid_indices[index] for index in fps_local]
    if len(optimization_source_indices) < DEFAULT_MIN_OPTIMIZATION_VIEWS:
        raise RuntimeError("spherical FPS returned fewer than eight optimization views")
    training_local = list(range(0, len(optimization_source_indices), 2))
    heldout_local = list(range(1, len(optimization_source_indices), 2))
    if not heldout_local:
        raise RuntimeError("O2W refinement has no held-out views")
    selection.update(
        {
            "valid_all_view_count": len(valid_indices),
            "optimization_view_limit": int(max_optimization_views),
            "optimization_source_indices_fps_order": optimization_source_indices,
            "optimization_frame_names_fps_order": [
                contract["frame_names"][index]
                for index in optimization_source_indices
            ],
            "training_source_indices_fps_alternating": [
                optimization_source_indices[index] for index in training_local
            ],
            "heldout_source_indices_fps_alternating": [
                optimization_source_indices[index] for index in heldout_local
            ],
            "all_views_used_for_full_resolution_acceptance_and_rendering": True,
        }
    )
    optimization = optimize_o2w_similarity(
        vertices_o=vertices_o,
        faces=faces,
        initial_T_O2W=initial,
        cameras_w2c=[
            contract["T_W2C"][index] for index in optimization_source_indices
        ],
        intrinsics=[
            contract["intrinsics"][index] for index in optimization_source_indices
        ],
        camera_models=[
            contract["cameras"][index] for index in optimization_source_indices
        ],
        masks=[
            contract["mask_arrays"][index] for index in optimization_source_indices
        ],
        training_view_indices=training_local,
        validation_view_indices=heldout_local,
        iterations=iterations,
        render_long_side=render_long_side,
        batch_size=batch_size,
    )
    candidate = validate_proper_similarity(
        np.asarray(optimization["candidate_T_O2W"], dtype=np.float64),
        name="candidate_T_O2W",
    )
    audit = _render_all_view_audit(
        mesh_o=mesh_path,
        contract=contract,
        initial_T_O2W=initial,
        candidate_T_O2W=candidate,
        output_dir=output,
        contour_width=contour_width,
    )
    accepted = bool(
        optimization["accepted_by_optimization_subset"]
        and audit["passed_for_selection"]
    )
    selected = candidate if accepted else initial
    selected_npz = output / "selected_T_O2W.npz"
    atomic_npz(
        selected_npz,
        T_O2W=np.asarray(selected, dtype=np.float64),
        T_W2O=np.linalg.inv(selected).astype(np.float64),
        initial_T_O2W=np.asarray(initial, dtype=np.float64),
        candidate_T_O2W=np.asarray(candidate, dtype=np.float64),
        accepted=np.asarray(accepted, dtype=np.bool_),
    )
    transform_report_path = output / "selected_T_O2W.json"
    transform_report = {
        "format": TRANSFORM_FORMAT,
        "created_at_utc": utc_now(),
        "passed": True,
        "accepted": accepted,
        "object_key": object_key,
        "T_O2W": selected.tolist(),
        "T_W2O": np.linalg.inv(selected).tolist(),
        "T_O2W_sha256": array_sha256(selected),
        "initial_T_O2W_sha256": array_sha256(initial),
        "candidate_T_O2W_sha256": array_sha256(candidate),
        "npz": str(selected_npz.resolve()),
        "npz_sha256": sha256_file(selected_npz),
        "source_report": str(report_path.resolve()),
        "selection_policy": (
            "candidate_if_optimization_subset_and_all_input_full_resolution_gates_pass;"
            "otherwise_initial"
        ),
    }
    atomic_json(transform_report_path, transform_report)
    overlays = _write_all_view_overlays(
        contract=contract,
        audit=audit,
        selected_is_candidate=accepted,
        output_dir=output,
        contour_width=contour_width,
    )
    world_mesh = _export_selected_world_mesh(mesh_path, selected, output / "07_优化后世界Mesh")
    # Large in-memory masks are not JSON data and are no longer needed after
    # overlays have been materialised.
    audit.pop("initial_masks", None)
    audit.pop("candidate_masks", None)
    report = {
        "format": FORMAT,
        "created_at_utc": utc_now(),
        "passed": True,
        "accepted": accepted,
        "reused": False,
        "object_key": object_key,
        "runtime_input_manifest": str(runtime_path),
        "mesh_o": str(mesh_path),
        "mesh_frame_report": str(mesh_report_path),
        "bindings": expected_binding,
        "input_contract_bindings": contract["bindings"],
        "configuration": configuration,
        "candidate_view_count": len(contract["frame_names"]),
        "valid_input_view_count": len(valid_indices),
        "view_selection": selection,
        "optimization_proxy": optimization_proxy,
        "initial_T_O2W": initial.tolist(),
        "initial_T_O2W_sha256": array_sha256(initial),
        "candidate_T_O2W": candidate.tolist(),
        "candidate_T_O2W_sha256": array_sha256(candidate),
        "selected_T_O2W": selected.tolist(),
        "selected_T_O2W_sha256": array_sha256(selected),
        "selected_T_O2W_npz": str(selected_npz.resolve()),
        "selected_T_O2W_npz_sha256": sha256_file(selected_npz),
        "selected_T_O2W_json": str(transform_report_path.resolve()),
        "selected_T_O2W_json_sha256": sha256_file(transform_report_path),
        "optimization": optimization,
        "full_input_projection_audit": audit,
        "contours": overlays,
        "selected_world_mesh": world_mesh,
        "view_inputs": contract["view_rows"],
        "decision": {
            "accepted": accepted,
            "optimization_subset_gate": bool(
                optimization["accepted_by_optimization_subset"]
            ),
            "all_input_full_resolution_gate": bool(audit["passed_for_selection"]),
            "selected": "candidate_T_O2W" if accepted else "initial_T_O2W",
            "geometry_changed": False,
        },
        "projection_formula": (
            "Mesh_O -> selected_T_O2W -> capture_A0/world -> raw_T_W2C -> "
            "K_raw + raw_distortion"
        ),
        "scope_guard": (
            "Input-image pose refinement only. Mesh-O vertices/faces, model inputs, "
            "SS/SLat outputs and the original runtime cache remain immutable. "
            "The candidate is deployed only when spatially interleaved held-out "
            "views and the full-resolution all-input audit do not regress."
        ),
    }
    atomic_json(report_path, report)
    return report


@dataclass(frozen=True)
class ResolvedInputs:
    runtime_input_manifest: Path
    mesh_o: Path
    mesh_frame_report: Path
    output_dir: Path
    object_key: str
    branch_dir: Path | None


def _resolve_existing_branch(path: Path, branch_name: str = "") -> Path:
    value = path.expanduser().resolve(strict=True)
    if value.is_file():
        if value.name != "branch_report.json":
            raise ValueError("--reconstruction_dir file must be branch_report.json")
        return value.parent
    if (value / "branch_report.json").is_file() or any(
        value.glob("02_runtime_o*/runtime_input_manifest.json")
    ):
        return value
    branches_root = value / "branches"
    if not branches_root.is_dir():
        raise FileNotFoundError(f"no reconstruction branches under: {value}")
    if branch_name:
        branch = branches_root / branch_name
        if not branch.is_dir():
            raise FileNotFoundError(branch)
        return branch
    preferred = branches_root / "01_training_spherical_farthest8"
    if preferred.is_dir():
        return preferred
    branches = sorted(path for path in branches_root.iterdir() if path.is_dir())
    if len(branches) != 1:
        raise RuntimeError(
            "reconstruction has multiple branches; supply --branch_name"
        )
    return branches[0]


def resolve_existing_reconstruction(
    reconstruction_dir: Path,
    *,
    branch_name: str = "",
    object_key: str = "",
    seed: int = 42,
    output_dir: Path | None = None,
) -> ResolvedInputs:
    branch = _resolve_existing_branch(reconstruction_dir, branch_name)
    branch_report_path = branch / "branch_report.json"
    branch_report = load_json(branch_report_path) if branch_report_path.is_file() else {}
    runtime_value = branch_report.get("runtime_input_manifest")
    if runtime_value:
        runtime = Path(runtime_value).expanduser().resolve(strict=True)
    else:
        values = sorted(branch.glob("02_runtime_o*/runtime_input_manifest.json"))
        if len(values) != 1:
            raise RuntimeError("could not resolve one runtime input manifest")
        runtime = values[0].resolve(strict=True)
    runtime_payload = load_json(runtime)
    runtime_rows = list(runtime_payload.get("objects", []))
    if object_key:
        runtime_row = _one_object(runtime_rows, object_key)
    elif len(runtime_rows) == 1:
        runtime_row = dict(runtime_rows[0])
        object_key = str(runtime_row["object_key"])
    else:
        raise RuntimeError("existing reconstruction needs --object")

    inference_value = branch_report.get("inference_manifest")
    if inference_value:
        inference_path = Path(inference_value).expanduser().resolve(strict=True)
    else:
        values = sorted(branch.glob("04_current*/inference_manifest.json"))
        if len(values) != 1:
            raise RuntimeError("could not resolve one current-model inference manifest")
        inference_path = values[0].resolve(strict=True)
    inference = load_json(inference_path)
    records = [
        dict(row)
        for row in inference.get("objects", [])
        if row.get("passed") is True
        and str(row.get("object_key")) == object_key
        and int(row.get("seed", seed)) == int(seed)
    ]
    if len(records) != 1:
        raise RuntimeError(
            f"could not resolve one Mesh record object={object_key} seed={seed}"
        )
    record = records[0]
    mesh = Path(record["mesh"]).expanduser().resolve(strict=True)
    mesh_report = Path(record["result"]).expanduser().resolve(strict=True)
    destination = (
        output_dir.expanduser().resolve()
        if output_dir is not None
        else branch / "04b_input_o2w_refinement"
    )
    return ResolvedInputs(
        runtime_input_manifest=runtime,
        mesh_o=mesh,
        mesh_frame_report=mesh_report,
        output_dir=destination,
        object_key=object_key,
        branch_dir=branch,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--reconstruction_dir",
        type=Path,
        help="existing session root, branch directory, or branch_report.json",
    )
    source.add_argument("--runtime_input_manifest", type=Path)
    parser.add_argument("--mesh_o", type=Path)
    parser.add_argument("--mesh_frame_report", type=Path)
    parser.add_argument("--output_dir", type=Path)
    parser.add_argument("--branch_name", default="")
    parser.add_argument("--object", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", default=os.environ.get("GPU", "0"))
    parser.add_argument(
        "--max_optimization_views", type=int, default=DEFAULT_MAX_OPTIMIZATION_VIEWS
    )
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    parser.add_argument(
        "--render_long_side", type=int, default=DEFAULT_RENDER_LONG_SIDE
    )
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--optimization_face_limit",
        type=int,
        default=DEFAULT_OPTIMIZATION_FACE_LIMIT,
    )
    parser.add_argument("--contour_width", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    # A direct module launch is commonly issued from ``base`` while the Python
    # executable itself belongs to the reconviagen environment.  nvdiffrast's
    # first-use JIT still resolves ``ninja`` through PATH, so bind the running
    # environment explicitly instead of relying on the caller's activated
    # shell.
    environment_bin = str(Path(sys.executable).resolve().parent)
    inherited_path = os.environ.get("PATH", "")
    os.environ["PATH"] = os.pathsep.join(
        value for value in (environment_bin, inherited_path) if value
    )
    if not os.environ.get("CUDA_HOME"):
        for candidate in (Path("/home/zjr/cuda-12.1"), Path("/usr/local/cuda")):
            if (candidate / "bin" / "nvcc").is_file():
                os.environ["CUDA_HOME"] = str(candidate)
                break
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu))
    os.environ.setdefault("TORCH_EXTENSIONS_DIR", "/tmp/torch_extensions")
    if args.reconstruction_dir is not None:
        resolved = resolve_existing_reconstruction(
            args.reconstruction_dir,
            branch_name=str(args.branch_name),
            object_key=str(args.object),
            seed=int(args.seed),
            output_dir=args.output_dir,
        )
    else:
        if args.mesh_o is None or args.mesh_frame_report is None or args.output_dir is None:
            raise ValueError(
                "explicit mode requires --mesh_o, --mesh_frame_report and --output_dir"
            )
        resolved = ResolvedInputs(
            runtime_input_manifest=args.runtime_input_manifest,
            mesh_o=args.mesh_o,
            mesh_frame_report=args.mesh_frame_report,
            output_dir=args.output_dir,
            object_key=str(args.object),
            branch_dir=None,
        )
    report = run_o2w_refinement(
        runtime_input_manifest=resolved.runtime_input_manifest,
        mesh_o=resolved.mesh_o,
        mesh_frame_report=resolved.mesh_frame_report,
        output_dir=resolved.output_dir,
        object_key=resolved.object_key,
        max_optimization_views=int(args.max_optimization_views),
        iterations=int(args.iterations),
        render_long_side=int(args.render_long_side),
        batch_size=int(args.batch_size),
        optimization_face_limit=int(args.optimization_face_limit),
        contour_width=int(args.contour_width),
        resume=bool(args.resume),
    )
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "accepted": report["accepted"],
                "object_key": report["object_key"],
                "initial_iou": report["full_input_projection_audit"]["initial_iou"],
                "selected": report["decision"]["selected"],
                "selected_T_O2W_npz": report["selected_T_O2W_npz"],
                "optimized_contour_overview": report["contours"][
                    "selected_optimized_overview"
                ],
                "report": str((resolved.output_dir / "report.json").resolve()),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
