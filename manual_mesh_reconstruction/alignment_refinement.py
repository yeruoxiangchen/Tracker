"""Fast post-return Mesh-to-mask alignment in the existing A0 frame.

This module never regenerates geometry.  It estimates one tightly bounded
similarity transform for the already downloaded A0-local mobile Mesh from a
short A0-relative RGB/pose clip.  SAM2 masks are observations; Mesh projections
are used only to construct sparse prompts.
"""

from __future__ import annotations

from datetime import datetime, timezone
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
from PIL import Image

from manual_mesh_reconstruction.canonicalization import array_sha256
from manual_mesh_reconstruction.common import atomic_json, sha256_file
from manual_mesh_reconstruction.projection import (
    make_headless_raster_context,
    rasterize_world_silhouette,
)
from pose_point_depth_mv.ar_mobile_overlay import read_mobile_overlay_mesh
from trellis_point_prior_mv.build_ar_session_smoke_dataset import (
    unity_pose_to_colmap_w2c,
)


FORMAT = "manual_mesh_reconstruction.fast_a0_silhouette_refinement.v4"
POSE_BINDING = "camera_frame_received_anchor_a0_relative_refinement_v1"
POSE_COORDINATE_FRAME = "unity_capture_anchor_a0"
MAX_OPTIMIZATION_VIEWS = 16
MIN_OPTIMIZATION_VIEWS = 16
RECOMMENDED_CAPTURE_FRAMES = 32
MAX_ROTATION_DEGREES = 12.0
MAX_TRANSLATION_METERS = 0.12
MIN_SCALE = 0.88
MAX_SCALE = 1.12
MIN_MASK_RATIO = 0.002
MAX_MASK_RATIO = 0.70


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def quaternion_xyzw_to_rotation(quaternion: Sequence[float]) -> np.ndarray:
    value = np.asarray(quaternion, dtype=np.float64)
    if value.shape != (4,) or not np.isfinite(value).all():
        raise ValueError("quaternion must contain four finite xyzw values")
    norm = float(np.linalg.norm(value))
    if norm <= 1.0e-12:
        raise ValueError("quaternion has zero norm")
    x, y, z, w = value / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def rotation_to_quaternion_xyzw(rotation: np.ndarray) -> list[float]:
    matrix = np.asarray(rotation, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError("rotation must be 3x3")
    trace = float(np.trace(matrix))
    if trace > 0.0:
        root = math.sqrt(trace + 1.0) * 2.0
        quaternion = [
            (matrix[2, 1] - matrix[1, 2]) / root,
            (matrix[0, 2] - matrix[2, 0]) / root,
            (matrix[1, 0] - matrix[0, 1]) / root,
            0.25 * root,
        ]
    else:
        diagonal = np.diag(matrix)
        axis = int(np.argmax(diagonal))
        if axis == 0:
            root = math.sqrt(1 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2
            quaternion = [
                0.25 * root,
                (matrix[0, 1] + matrix[1, 0]) / root,
                (matrix[0, 2] + matrix[2, 0]) / root,
                (matrix[2, 1] - matrix[1, 2]) / root,
            ]
        elif axis == 1:
            root = math.sqrt(1 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2
            quaternion = [
                (matrix[0, 1] + matrix[1, 0]) / root,
                0.25 * root,
                (matrix[1, 2] + matrix[2, 1]) / root,
                (matrix[0, 2] - matrix[2, 0]) / root,
            ]
        else:
            root = math.sqrt(1 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2
            quaternion = [
                (matrix[0, 2] + matrix[2, 0]) / root,
                (matrix[1, 2] + matrix[2, 1]) / root,
                0.25 * root,
                (matrix[1, 0] - matrix[0, 1]) / root,
            ]
    value = np.asarray(quaternion, dtype=np.float64)
    value /= np.linalg.norm(value)
    if value[3] < 0:
        value *= -1
    return value.tolist()


def unity_similarity_to_internal(payload: Mapping[str, Any] | None) -> np.ndarray:
    """Convert a Unity A0-local Transform into the internal reflected axes."""

    if not payload:
        return np.eye(4, dtype=np.float64)
    position = np.asarray(
        [payload.get("position_x", 0), payload.get("position_y", 0), payload.get("position_z", 0)],
        dtype=np.float64,
    )
    quaternion = [
        payload.get("quaternion_x", 0),
        payload.get("quaternion_y", 0),
        payload.get("quaternion_z", 0),
        payload.get("quaternion_w", 1),
    ]
    scale = float(payload.get("uniform_scale", 1.0))
    if not np.isfinite(position).all() or not math.isfinite(scale) or scale <= 0:
        raise ValueError("current Unity Mesh transform is invalid")
    reflection = np.diag([1.0, 1.0, -1.0])
    rotation_unity = quaternion_xyzw_to_rotation(quaternion)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = scale * reflection @ rotation_unity @ reflection
    transform[:3, 3] = reflection @ position
    return transform


def internal_similarity_to_unity(transform: np.ndarray) -> dict[str, Any]:
    value = np.asarray(transform, dtype=np.float64)
    if value.shape != (4, 4) or not np.isfinite(value).all():
        raise ValueError("internal similarity must be a finite 4x4 matrix")
    scale = float(np.cbrt(np.linalg.det(value[:3, :3])))
    if scale <= 0:
        raise ValueError("internal similarity has invalid scale")
    rotation_internal = value[:3, :3] / scale
    reflection = np.diag([1.0, 1.0, -1.0])
    rotation_unity = reflection @ rotation_internal @ reflection
    translation_unity = reflection @ value[:3, 3]
    quaternion = rotation_to_quaternion_xyzw(rotation_unity)
    return {
        "position_x": float(translation_unity[0]),
        "position_y": float(translation_unity[1]),
        "position_z": float(translation_unity[2]),
        "quaternion_x": float(quaternion[0]),
        "quaternion_y": float(quaternion[1]),
        "quaternion_z": float(quaternion[2]),
        "quaternion_w": float(quaternion[3]),
        "uniform_scale": scale,
    }


def evenly_spaced_indices(count: int, limit: int = MAX_OPTIMIZATION_VIEWS) -> list[int]:
    if count <= 0:
        return []
    target = min(int(count), int(limit))
    if target == count:
        return list(range(count))
    values = np.linspace(0, count - 1, target)
    indices = [int(round(value)) for value in values]
    if len(set(indices)) != target:
        raise RuntimeError("uniform calibration subset produced duplicate indices")
    return indices


def object_centered_spherical_farthest_indices(
    cameras: Sequence[np.ndarray],
    object_center_internal: np.ndarray,
    limit: int = MAX_OPTIMIZATION_VIEWS,
) -> tuple[list[int], dict[str, Any]]:
    """Select camera views by spherical FPS around the current Mesh center.

    The candidate list may be arbitrarily long. Selection is O(N * K): one
    deterministic seed is chosen opposite the mean observed direction, then
    every next view maximises its minimum angular distance to the selected
    directions. The returned indices retain FPS order; callers may sort the
    selected set temporally before sending it to a video mask propagator.
    """

    target = min(len(cameras), int(limit))
    if target <= 0:
        return [], {
            "policy": "current_mesh_centered_spherical_farthest",
            "algorithm": "mean_opposite_seed_then_greedy_spherical_fps_v1",
            "candidate_view_count": len(cameras),
            "selected_view_count": 0,
            "selected_source_indices_fps_order": [],
        }
    center = np.asarray(object_center_internal, dtype=np.float64)
    if center.shape != (3,) or not np.isfinite(center).all():
        raise ValueError("calibration object center must be a finite [3] vector")

    camera_centers = []
    for index, camera in enumerate(cameras):
        value = np.asarray(camera, dtype=np.float64)
        if value.shape != (4, 4) or not np.isfinite(value).all():
            raise ValueError(
                f"calibration camera {index} must be a finite 4x4 T_W2C"
            )
        camera_centers.append(np.linalg.inv(value)[:3, 3])
    camera_centers_array = np.asarray(camera_centers, dtype=np.float64)
    offsets = camera_centers_array - center[None]
    radii = np.linalg.norm(offsets, axis=1)
    invalid = np.flatnonzero(~np.isfinite(radii) | (radii <= 1.0e-8))
    if len(invalid):
        raise ValueError(
            "calibration camera center coincides with the current Mesh center: "
            f"indices={invalid.astype(int).tolist()}"
        )
    directions = offsets / radii[:, None]

    mean_direction = directions.mean(axis=0)
    mean_norm = float(np.linalg.norm(mean_direction))
    if mean_norm > 1.0e-8:
        mean_direction /= mean_norm
        seed = min(
            range(len(directions)),
            key=lambda index: (float(directions[index] @ mean_direction), index),
        )
        seed_policy = "direction_farthest_from_normalized_candidate_mean"
    else:
        seed = 0
        seed_policy = "earliest_candidate_when_direction_mean_is_degenerate"

    selected = [int(seed)]
    # Maximum cosine to the selected set corresponds to the minimum angular
    # distance. Minimising it is spherical farthest sampling without repeated
    # arccos calls for every candidate.
    maximum_cosine = directions @ directions[seed]
    maximum_cosine[seed] = np.inf
    while len(selected) < target:
        next_index = min(
            (index for index in range(len(directions)) if index not in selected),
            key=lambda index: (float(maximum_cosine[index]), index),
        )
        selected.append(int(next_index))
        maximum_cosine = np.maximum(
            maximum_cosine, directions @ directions[next_index]
        )
        maximum_cosine[selected] = np.inf

    selected_directions = directions[selected]
    cosine = np.clip(selected_directions @ selected_directions.T, -1.0, 1.0)
    angular = np.degrees(np.arccos(cosine))
    np.fill_diagonal(angular, np.inf)
    nearest = np.min(angular, axis=1)

    horizontal = offsets[selected][:, [0, 2]]
    horizontal_norm = np.linalg.norm(horizontal, axis=1)
    azimuth = (
        np.degrees(np.arctan2(horizontal[:, 0], horizontal[:, 1])) % 360.0
    )
    azimuth = azimuth[horizontal_norm > 1.0e-8]
    if len(azimuth) >= 2:
        ordered = np.sort(azimuth)
        gaps = np.diff(np.concatenate([ordered, ordered[:1] + 360.0]))
        maximum_gap = float(np.max(gaps))
        azimuth_coverage = 360.0 - maximum_gap
    else:
        maximum_gap = 360.0
        azimuth_coverage = 0.0

    return selected, {
        "policy": "current_mesh_centered_spherical_farthest",
        "algorithm": "mean_opposite_seed_then_greedy_spherical_fps_v1",
        "seed_policy": seed_policy,
        "seed_source_index": int(seed),
        "candidate_view_count": len(cameras),
        "selected_view_count": len(selected),
        "selected_source_indices_fps_order": selected,
        "object_center_source": "current_transformed_mobile_mesh_aabb_center",
        "object_center_internal": center.tolist(),
        "selected_camera_centers_internal_fps_order": camera_centers_array[
            selected
        ].tolist(),
        "selected_camera_directions_internal_fps_order": selected_directions.tolist(),
        "selected_camera_radii_meters_fps_order": radii[selected].tolist(),
        "minimum_pairwise_angular_separation_degrees": float(np.min(nearest)),
        "mean_nearest_angular_separation_degrees": float(np.mean(nearest)),
        "azimuth_coverage_degrees_about_internal_y": azimuth_coverage,
        "maximum_azimuth_gap_degrees_about_internal_y": maximum_gap,
    }


def materialize_corrected_calibration_frame(
    image_path: Path, output_path: Path
) -> tuple[int, int]:
    with Image.open(image_path) as source:
        corrected = source.convert("RGB").transpose(Image.Transpose.TRANSPOSE)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    corrected.save(output_path, quality=95)
    return corrected.size


def corrected_intrinsic(metadata: Mapping[str, Any], raw_size: tuple[int, int]) -> np.ndarray:
    raw_width, raw_height = (int(raw_size[0]), int(raw_size[1]))
    source_width = int(metadata.get("intrinsic_width") or raw_width)
    source_height = int(metadata.get("intrinsic_height") or raw_height)
    fx = float(metadata["fx"]) * raw_width / source_width
    fy = float(metadata["fy"]) * raw_height / source_height
    cx = float(metadata["cx"]) * raw_width / source_width
    cy = float(metadata["cy"]) * raw_height / source_height
    return np.asarray(
        [[fy, 0.0, cy], [0.0, fx, cx], [0.0, 0.0, 1.0]], dtype=np.float64
    )


def internal_camera_from_unity_metadata(metadata: Mapping[str, Any]) -> np.ndarray:
    pose = {
        "pos": np.asarray(
            [metadata["pos_x"], metadata["pos_y"], metadata["pos_z"]],
            dtype=np.float64,
        ),
        "quat": np.asarray(
            [
                metadata["quat_x"],
                metadata["quat_y"],
                metadata["quat_z"],
                metadata["quat_w"],
            ],
            dtype=np.float64,
        ),
    }
    # The phone uploads XRCpuImage with Transformation.None.  The server then
    # applies one explicit x/y image transpose and swaps K in
    # ``corrected_intrinsic``.  That pixel-domain operation must not be counted
    # a second time as a camera optical-axis rotation.  The normal
    # reconstruction path uses the same image contract and therefore the same
    # zero-degree camera conversion.
    rotation, translation = unity_pose_to_colmap_w2c(
        pose, image_camera_rotation_degrees=0.0
    )
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rotation
    result[:3, 3] = translation
    return result


def mobile_mesh_internal_buffers(path: Path) -> tuple[np.ndarray, np.ndarray]:
    payload = read_mobile_overlay_mesh(path)
    vertices_unity = np.asarray(payload["vertices"], dtype=np.float32)
    faces = np.asarray(payload["faces"], dtype=np.int32)
    reflection = np.asarray([1.0, 1.0, -1.0], dtype=np.float32)
    vertices_internal = vertices_unity * reflection[None]
    if len(vertices_internal) < 3 or len(faces) < 1:
        raise RuntimeError("mobile A0 Mesh is empty")
    return vertices_internal, faces


def mask_prompt(mask: np.ndarray, frame_index: int) -> dict[str, Any]:
    binary = np.asarray(mask, dtype=np.uint8) > 0
    ys, xs = np.nonzero(binary)
    if len(xs) < 32:
        raise ValueError("Mesh projection is too small to prompt SAM2")
    height, width = binary.shape
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    # The projected Mesh is only a locator.  A deliberately loose box leaves
    # room for SAM2 to observe and correct the very translation/scale error we
    # are trying to estimate; a tight Mesh box would make the result circular.
    pad_x = max(8, int(round((x1 - x0 + 1) * 0.25)))
    pad_y = max(8, int(round((y1 - y0 + 1) * 0.25)))
    box = [
        float(max(0, x0 - pad_x)),
        float(max(0, y0 - pad_y)),
        float(min(width - 1, x1 + pad_x)),
        float(min(height - 1, y1 + pad_y)),
    ]
    distance = cv2.distanceTransform(binary.astype(np.uint8), cv2.DIST_L2, 5)
    points: list[list[float]] = []
    working = distance.copy()
    suppression = max(4, int(round(min(height, width) * 0.04)))
    for _ in range(2):
        _min, maximum, _min_location, location = cv2.minMaxLoc(working)
        if maximum <= 0:
            break
        x, y = location
        points.append([float(x), float(y)])
        cv2.circle(working, (x, y), suppression, 0.0, -1)
    if not points:
        points.append([float(np.median(xs)), float(np.median(ys))])
    labels = [1] * len(points)
    # Negative points just outside the projected support discourage SAM2 from
    # swallowing the surrounding table/box while leaving the true boundary to
    # the image model.
    outside_x = max(8, int(round(width * 0.04)))
    outside_y = max(8, int(round(height * 0.04)))
    candidates = [
        (int(box[0]) - outside_x, (y0 + y1) // 2),
        (int(box[2]) + outside_x, (y0 + y1) // 2),
        ((x0 + x1) // 2, int(box[1]) - outside_y),
        ((x0 + x1) // 2, int(box[3]) + outside_y),
    ]
    for x, y in candidates:
        x = int(np.clip(x, 0, width - 1))
        y = int(np.clip(y, 0, height - 1))
        if not binary[y, x]:
            points.append([float(x), float(y)])
            labels.append(0)
    return {
        "frame_index": int(frame_index),
        "points": points,
        "labels": labels,
        "box": box,
        "source": "current_mesh_projection_sparse_box_points",
    }


def binary_iou(first: np.ndarray, second: np.ndarray) -> float:
    a = np.asarray(first, dtype=bool)
    b = np.asarray(second, dtype=bool)
    union = int(np.logical_or(a, b).sum())
    if union == 0:
        return 1.0
    return float(np.logical_and(a, b).sum() / union)


def _axis_angle_matrix(vector):
    import torch

    x, y, z = vector.unbind()
    zero = torch.zeros((), device=vector.device, dtype=vector.dtype)
    skew = torch.stack(
        [zero, -z, y, z, zero, -x, -y, x, zero]
    ).reshape(3, 3)
    return torch.matrix_exp(skew)


def _render_soft_silhouette(vertices, faces, camera, intrinsic, height, width, context):
    import nvdiffrast.torch as dr
    import torch

    vertices_h = torch.cat(
        [vertices, torch.ones((len(vertices), 1), device=vertices.device)], dim=1
    )
    camera_vertices = (vertices_h @ camera.T)[:, :3]
    positive = camera_vertices[:, 2] > 1.0e-4
    visible_faces = faces[positive[faces].all(dim=1)]
    if len(visible_faces) == 0:
        raise RuntimeError("alignment Mesh has no front-camera triangles")
    depth = camera_vertices[:, 2].clamp(min=1.0e-4)
    u = intrinsic[0, 0] * camera_vertices[:, 0] / depth + intrinsic[0, 2]
    v = intrinsic[1, 1] * camera_vertices[:, 1] / depth + intrinsic[1, 2]
    x_ndc = 2.0 * u / max(width - 1, 1) - 1.0
    y_ndc = 2.0 * v / max(height - 1, 1) - 1.0
    near = torch.tensor(0.01, device=vertices.device)
    far = torch.tensor(20.0, device=vertices.device)
    z_ndc = 2.0 * (depth - near) / (far - near) - 1.0
    clip = torch.stack([x_ndc * depth, y_ndc * depth, z_ndc * depth, depth], dim=1)
    rast, _ = dr.rasterize(
        context, clip[None], visible_faces, resolution=(height, width)
    )
    attributes = torch.ones((1, len(vertices), 1), device=vertices.device)
    coverage, _ = dr.interpolate(attributes, rast, visible_faces)
    return dr.antialias(
        coverage, rast, clip[None], visible_faces
    )[0, ..., 0].clamp(0, 1)


def optimize_similarity(
    *,
    vertices: np.ndarray,
    faces: np.ndarray,
    cameras: Sequence[np.ndarray],
    intrinsics: Sequence[np.ndarray],
    masks: Sequence[np.ndarray],
    current_transform: np.ndarray,
    training_view_indices: Sequence[int] | None = None,
    validation_view_indices: Sequence[int] | None = None,
    iterations: int = 50,
    render_long_side: int = 256,
) -> dict[str, Any]:
    import torch
    import nvdiffrast.torch as dr

    if len(cameras) < MIN_OPTIMIZATION_VIEWS or not (
        len(cameras) == len(intrinsics) == len(masks)
    ):
        raise ValueError(
            f"alignment requires at least {MIN_OPTIMIZATION_VIEWS} matched "
            "camera/mask views"
        )
    device = torch.device("cuda")
    base_vertices = torch.as_tensor(vertices, dtype=torch.float32, device=device)
    faces_tensor = torch.as_tensor(faces, dtype=torch.int32, device=device)
    current = torch.as_tensor(current_transform, dtype=torch.float32, device=device)
    raw = torch.zeros(7, dtype=torch.float32, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([raw], lr=0.035)
    context = dr.RasterizeCudaContext()

    prepared = []
    for camera_value, intrinsic_value, mask_value in zip(cameras, intrinsics, masks):
        source_height, source_width = mask_value.shape
        scale = min(1.0, float(render_long_side) / max(source_height, source_width))
        height = max(32, int(round(source_height * scale)))
        width = max(32, int(round(source_width * scale)))
        target = cv2.resize(
            (np.asarray(mask_value) > 0).astype(np.float32),
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        )
        intrinsic_scaled = np.asarray(intrinsic_value, dtype=np.float64).copy()
        intrinsic_scaled[0, :] *= width / source_width
        intrinsic_scaled[1, :] *= height / source_height
        prepared.append(
            (
                torch.as_tensor(camera_value, dtype=torch.float32, device=device),
                torch.as_tensor(intrinsic_scaled, dtype=torch.float32, device=device),
                torch.as_tensor(target, dtype=torch.float32, device=device),
                height,
                width,
            )
        )
    if training_view_indices is None and validation_view_indices is None:
        train_indices = [index for index in range(len(prepared)) if index % 2 == 0]
        validation_indices = [index for index in range(len(prepared)) if index % 2 == 1]
    elif training_view_indices is None or validation_view_indices is None:
        raise ValueError("training and validation calibration indices must be paired")
    else:
        train_indices = [int(index) for index in training_view_indices]
        validation_indices = [int(index) for index in validation_view_indices]
    expected_indices = set(range(len(prepared)))
    if (
        not train_indices
        or not validation_indices
        or set(train_indices) & set(validation_indices)
        or set(train_indices) | set(validation_indices) != expected_indices
        or len(set(train_indices)) != len(train_indices)
        or len(set(validation_indices)) != len(validation_indices)
    ):
        raise ValueError(
            "calibration train/validation indices must be nonempty, disjoint, "
            "unique and cover every selected view"
        )

    maximum_angle = math.radians(MAX_ROTATION_DEGREES)
    minimum_log_scale = math.log(MIN_SCALE)
    maximum_log_scale = math.log(MAX_SCALE)

    def perturbation():
        vector = raw[:3]
        norm = torch.linalg.vector_norm(vector)
        regular_ratio = torch.tanh(norm) / torch.clamp(norm, min=1.0e-6)
        taylor_ratio = 1.0 - norm * norm / 3.0
        ratio = torch.where(norm < 1.0e-4, taylor_ratio, regular_ratio)
        bounded = maximum_angle * ratio * vector
        rotation = _axis_angle_matrix(bounded)
        translation = MAX_TRANSLATION_METERS * torch.tanh(raw[3:6])
        signed = torch.tanh(raw[6])
        log_scale = torch.where(
            signed >= 0,
            maximum_log_scale * signed,
            (-minimum_log_scale) * signed,
        )
        scale = torch.exp(log_scale)
        delta = torch.eye(4, device=device)
        delta[:3, :3] = scale * rotation
        delta[:3, 3] = translation
        return delta

    def transformed_vertices(transform):
        return base_vertices @ transform[:3, :3].T + transform[:3, 3]

    def losses(indices, transform):
        transformed = transformed_vertices(transform)
        values = []
        for index in indices:
            camera, intrinsic, target, height, width = prepared[index]
            prediction = _render_soft_silhouette(
                transformed, faces_tensor, camera, intrinsic, height, width, context
            )
            intersection = (prediction * target).sum()
            union = prediction.sum() + target.sum() - intersection
            soft_iou = (intersection + 1.0) / (union + 1.0)
            values.append(1.0 - soft_iou)
        return torch.stack(values)

    current_loss = None
    for _iteration in range(int(iterations)):
        optimizer.zero_grad(set_to_none=True)
        transform = perturbation() @ current
        data_loss = losses(train_indices, transform).mean()
        prior = 0.002 * (raw[:6] ** 2).mean() + 0.001 * raw[6] ** 2
        loss = data_loss + prior
        loss.backward()
        torch.nn.utils.clip_grad_norm_([raw], 2.0)
        optimizer.step()
        current_loss = float(loss.detach().cpu())

    with torch.no_grad():
        initial = current
        optimized = perturbation() @ current

        def hard_ious(transform):
            transformed = transformed_vertices(transform)
            result = []
            for camera, intrinsic, target, height, width in prepared:
                prediction = _render_soft_silhouette(
                    transformed, faces_tensor, camera, intrinsic, height, width, context
                )
                prediction = prediction > 0.5
                target_binary = target > 0.5
                intersection = torch.logical_and(prediction, target_binary).sum().float()
                union = torch.logical_or(prediction, target_binary).sum().float()
                result.append(float((intersection / union.clamp(min=1)).cpu()))
            return result

        initial_ious = hard_ious(initial)
        optimized_ious = hard_ious(optimized)
        optimized_numpy = optimized.detach().cpu().numpy().astype(np.float64)

    del context
    torch.cuda.empty_cache()
    gains = [after - before for before, after in zip(initial_ious, optimized_ious)]
    train_gain = float(np.mean([gains[index] for index in train_indices]))
    validation_gain = float(np.mean([gains[index] for index in validation_indices]))
    mean_gain = float(np.mean(gains))
    positive_rate = float(np.mean(np.asarray(gains) > 0.0))
    initial_mean = float(np.mean(initial_ious))
    optimized_mean = float(np.mean(optimized_ious))
    checks = {
        "initial_alignment_not_catastrophic": initial_mean >= 0.10,
        "overall_iou_gain_ge_0p01": mean_gain >= 0.01,
        "training_iou_gain_positive": train_gain > 0.0,
        "heldout_iou_non_degrading": validation_gain >= -0.005,
        "view_positive_rate_ge_half": positive_rate >= 0.5,
    }
    accepted = all(checks.values())
    return {
        "passed": True,
        "accepted": accepted,
        "iterations": int(iterations),
        "final_training_loss": current_loss,
        "train_indices": train_indices,
        "validation_indices": validation_indices,
        "initial_iou": initial_ious,
        "optimized_iou": optimized_ious,
        "iou_gain": gains,
        "initial_iou_mean": initial_mean,
        "optimized_iou_mean": optimized_mean,
        "iou_gain_mean": mean_gain,
        "train_iou_gain_mean": train_gain,
        "heldout_iou_gain_mean": validation_gain,
        "positive_view_rate": positive_rate,
        "checks": checks,
        "selected_transform_internal": (
            optimized_numpy if accepted else np.asarray(current_transform)
        ).tolist(),
        "optimized_transform_internal": optimized_numpy.tolist(),
        "selected_transform_unity": internal_similarity_to_unity(
            optimized_numpy if accepted else np.asarray(current_transform)
        ),
        "rejection_policy": "preserve_current_mesh_transform_if_any_gate_fails",
    }


def run_refinement(
    *,
    mobile_mesh: Path,
    frame_rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
    sam2_client,
    current_unity_transform: Mapping[str, Any] | None,
    iterations: int = 50,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if len(frame_rows) < MIN_OPTIMIZATION_VIEWS:
        raise ValueError(
            f"alignment requires at least {MIN_OPTIMIZATION_VIEWS} candidate frames"
        )

    vertices, faces = mobile_mesh_internal_buffers(mobile_mesh)
    current_transform = unity_similarity_to_internal(current_unity_transform)
    current_vertices = (
        vertices @ current_transform[:3, :3].T + current_transform[:3, 3]
    )
    object_center_internal = 0.5 * (
        np.min(current_vertices, axis=0) + np.max(current_vertices, axis=0)
    )
    all_cameras = [
        internal_camera_from_unity_metadata(row) for row in frame_rows
    ]
    fps_order, view_selection = object_centered_spherical_farthest_indices(
        all_cameras,
        object_center_internal,
        MAX_OPTIMIZATION_VIEWS,
    )
    if len(fps_order) != MAX_OPTIMIZATION_VIEWS:
        raise RuntimeError(
            f"spherical calibration selection returned {len(fps_order)} views; "
            f"expected {MAX_OPTIMIZATION_VIEWS}"
        )

    # SAM2 is a video predictor, so materialise the selected FPS set in source
    # time order. The FPS rank is preserved separately and alternated between
    # train/held-out splits so both halves remain spatially diverse.
    selected_indices = sorted(fps_order)
    selected_rows = [dict(frame_rows[index]) for index in selected_indices]
    source_to_view = {
        source_index: view_index
        for view_index, source_index in enumerate(selected_indices)
    }
    training_source_indices = fps_order[::2]
    validation_source_indices = fps_order[1::2]
    training_view_indices = [
        source_to_view[source_index] for source_index in training_source_indices
    ]
    validation_view_indices = [
        source_to_view[source_index] for source_index in validation_source_indices
    ]
    view_selection.update(
        {
            "selected_source_indices_temporal_order": selected_indices,
            "selected_frame_names_fps_order": [
                str(
                    frame_rows[index].get("frame_name")
                    or Path(str(frame_rows[index]["image"])).name
                )
                for index in fps_order
            ],
            "selected_frame_names_temporal_order": [
                str(row.get("frame_name") or Path(str(row["image"])).name)
                for row in selected_rows
            ],
            "training_source_indices_fps_alternating": training_source_indices,
            "validation_source_indices_fps_alternating": validation_source_indices,
            "training_selected_view_indices": training_view_indices,
            "validation_selected_view_indices": validation_view_indices,
            "video_propagation_order": "selected_source_temporal_order",
        }
    )
    corrected_dir = output_dir / "01_corrected_rgb"
    projection_dir = output_dir / "02_mesh_prompt_projection"
    mask_dir = output_dir / "03_sam2_tiny_observed_masks"
    corrected_paths: list[Path] = []
    cameras: list[np.ndarray] = []
    intrinsics: list[np.ndarray] = []
    for selected_ordinal, (source_index, row) in enumerate(
        zip(selected_indices, selected_rows)
    ):
        source = Path(str(row["image"])).resolve(strict=True)
        with Image.open(source) as image:
            raw_size = image.size
        destination = corrected_dir / f"view_{selected_ordinal:02d}.jpg"
        materialize_corrected_calibration_frame(source, destination)
        corrected_paths.append(destination)
        cameras.append(all_cameras[source_index])
        intrinsics.append(corrected_intrinsic(row, raw_size))
    raster_context = make_headless_raster_context()
    projected_masks: list[np.ndarray] = []
    prompt_candidates = []
    projection_dir.mkdir(parents=True, exist_ok=True)
    for index, (image_path, camera, intrinsic) in enumerate(
        zip(corrected_paths, cameras, intrinsics)
    ):
        with Image.open(image_path) as image:
            width, height = image.size
        projected = rasterize_world_silhouette(
            current_vertices,
            faces,
            camera,
            intrinsic,
            {"model": "PINHOLE", "distortion": []},
            (height, width),
            raster_context,
        )
        projected_masks.append(projected)
        path = projection_dir / f"view_{index:02d}.png"
        Image.fromarray(projected * 255, mode="L").save(path)
        area = int(projected.sum())
        touches_border = bool(
            projected[0].any()
            or projected[-1].any()
            or projected[:, 0].any()
            or projected[:, -1].any()
        )
        prompt_candidates.append((touches_border, -area, index))
    del raster_context

    # Direct Mesh prompts are restricted to the spatially interleaved training
    # half. Held-out FPS views receive masks only through RGB video propagation.
    training_view_set = set(training_view_indices)
    train_prompt_candidates = [
        row for row in prompt_candidates if row[2] in training_view_set
    ]
    prompt_count = min(3, len(train_prompt_candidates))
    prompt_indices = [
        row[2] for row in sorted(train_prompt_candidates)[:prompt_count]
    ]
    prompts = [mask_prompt(projected_masks[index], index) for index in prompt_indices]
    mask_paths = [mask_dir / f"view_{index:02d}.png" for index in range(len(corrected_paths))]
    sam2_report = sam2_client.segment(
        image_paths=corrected_paths,
        mask_paths=mask_paths,
        prompts=prompts,
    )
    observed_masks = []
    mask_rows = []
    for index, path in enumerate(mask_paths):
        observed = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if observed is None:
            raise RuntimeError(f"SAM2 mask is unreadable: {path}")
        binary = observed > 0
        ratio = float(binary.mean())
        valid = MIN_MASK_RATIO <= ratio <= MAX_MASK_RATIO
        mask_rows.append(
            {
                "view_index": index,
                "source_index": selected_indices[index],
                "split": "train" if index in training_view_set else "heldout",
                "mask": str(path),
                "mask_sha256": sha256_file(path),
                "foreground_ratio": ratio,
                "valid": valid,
                "prompt_projection_iou": binary_iou(binary, projected_masks[index]),
            }
        )
        if not valid:
            raise RuntimeError(
                f"SAM2 mask foreground ratio is unsafe view={index}: {ratio:.6f}"
            )
        observed_masks.append(binary)

    optimization = optimize_similarity(
        vertices=vertices,
        faces=faces,
        cameras=cameras,
        intrinsics=intrinsics,
        masks=observed_masks,
        current_transform=current_transform,
        training_view_indices=training_view_indices,
        validation_view_indices=validation_view_indices,
        iterations=iterations,
    )
    report = {
        "format": FORMAT,
        "created_at_utc": utc_now(),
        "passed": True,
        "accepted": bool(optimization["accepted"]),
        "mobile_mesh": str(mobile_mesh.resolve()),
        "mobile_mesh_sha256": sha256_file(mobile_mesh),
        "captured_frame_count": len(frame_rows),
        "selected_source_indices": selected_indices,
        "selected_view_count": len(selected_rows),
        "view_selection": view_selection,
        "pose_binding": POSE_BINDING,
        "pose_coordinate_frame": POSE_COORDINATE_FRAME,
        "pixel_axis_contract": "xrcpuimage_none_then_xy_transpose_with_K_swap_v1",
        "unity_to_internal_camera_contract": (
            "same_as_primary_capture_zero_optical_axis_rotation_v1"
        ),
        "image_camera_rotation_degrees": 0.0,
        "prompt_policy": "current_mesh_projection_to_sparse_box_points_only",
        "prompt_frame_indices": prompt_indices,
        "prompts": prompts,
        "sam2": sam2_report,
        "masks": mask_rows,
        "current_transform_internal": current_transform.tolist(),
        "current_transform_internal_sha256": array_sha256(current_transform),
        "optimization": optimization,
        "selected_transform_unity": optimization["selected_transform_unity"],
        "scope_guard": (
            "Display-pose refinement only. SS/SLat geometry, A0 and the "
            "stored reconstruction Mesh are immutable. A failed gate preserves "
            "the pre-refinement Mesh transform exactly."
        ),
    }
    atomic_json(output_dir / "report.json", report)
    return report
