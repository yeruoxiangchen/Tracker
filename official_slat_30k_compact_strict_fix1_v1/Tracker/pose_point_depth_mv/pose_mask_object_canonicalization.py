#!/usr/bin/env python3
"""Runtime object frame derived from calibrated poses and foreground masks only."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Sequence

import numpy as np

from ar_ss_flow.shared_object_preprocessing import (
    canonical_json_sha256,
    prepare_shared_object_arrays,
    transform_intrinsics,
)
from pose_point_depth_mv.real_object_canonicalization import (
    RuntimeObjectFrame,
    RuntimeObjectObservation,
    _estimate_axes,
    _mask_centroid_ray_center,
    _mask_extent_at_center,
    _validate_camera_inputs,
    array_sha256,
    invert_similarity,
    undistort_rgb_mask_views,
)


POSE_MASK_OBJECT_FRAME_VERSION = "pose_point_depth_mv.pose_mask_object_frame.v1"
POSE_MASK_INPUT_FRONTEND_VERSION = "pose_point_depth_mv.pose_mask_input_frontend.v1"


@dataclass(frozen=True)
class PoseMaskObjectFrameConfig:
    """Frozen rules for estimating O without reading a point cloud."""

    mask_threshold: float = 0.5
    expected_object_extent: float = 0.90
    scale_padding: float = 1.05
    minimum_scale: float = 1.0e-6

    def validate(self) -> None:
        if not 0.0 < float(self.mask_threshold) < 1.0:
            raise ValueError("mask_threshold must be in (0,1)")
        if float(self.expected_object_extent) <= 0.0:
            raise ValueError("expected_object_extent must be positive")
        if float(self.scale_padding) < 1.0:
            raise ValueError("scale_padding must be at least one")
        if float(self.minimum_scale) <= 0.0:
            raise ValueError("minimum_scale must be positive")


def canonicalize_pose_mask_runtime_object_frame(
    intrinsics: np.ndarray,
    T_W2C: np.ndarray,
    masks: Sequence[np.ndarray],
    *,
    config: PoseMaskObjectFrameConfig | None = None,
    gravity_up_W: np.ndarray | None = None,
    reference_view_index: int = 0,
) -> RuntimeObjectFrame:
    """Estimate O from mask-centroid rays, camera poses, and mask extent only."""

    rules = config or PoseMaskObjectFrameConfig()
    rules.validate()
    view_count = len(masks)
    k_all, poses = _validate_camera_inputs(intrinsics, T_W2C, view_count)
    if not 0 <= int(reference_view_index) < view_count:
        raise ValueError("reference_view_index is outside the selected views")

    center, ray_stats = _mask_centroid_ray_center(
        masks, k_all, poses, float(rules.mask_threshold)
    )
    rotation, axis_stats = _estimate_axes(
        center,
        poses,
        gravity_up_W=gravity_up_W,
        reference_view_index=int(reference_view_index),
    )
    mask_extent, mask_spans = _mask_extent_at_center(
        center, masks, k_all, poses, float(rules.mask_threshold)
    )
    scale = (
        float(mask_extent)
        * float(rules.scale_padding)
        / float(rules.expected_object_extent)
    )
    if not math.isfinite(scale) or scale < float(rules.minimum_scale):
        raise RuntimeError(f"estimated pose+mask object scale is invalid: {scale}")

    T_O2W = np.eye(4, dtype=np.float64)
    T_O2W[:3, :3] = rotation * scale
    T_O2W[:3, 3] = center
    T_W2O = invert_similarity(T_O2W)
    T_O2C = np.matmul(poses, T_O2W[None])
    T_C2O = np.matmul(T_W2O[None], np.linalg.inv(poses))
    empty_points = np.empty((0, 3), dtype=np.float64)
    empty_keep = np.empty((0,), dtype=bool)

    ray_median = float(ray_stats["ray_residual_median"])
    ray_p90 = float(ray_stats["ray_residual_p90"])
    ray_max = float(ray_stats["ray_residual_max"])
    stats = {
        "point_cloud_consumed": False,
        "center_source": "mask_centroid_rays",
        "scale_source": "median_mask_extent",
        "ray_center": ray_stats,
        "axes": axis_stats,
        "center_W": center.tolist(),
        "mask_extent_median_W": float(mask_extent),
        "mask_extent_per_view_W": [float(value) for value in mask_spans],
        "scale_O2W": float(scale),
        "ray_residual_median_over_mask_extent": ray_median / float(mask_extent),
        "ray_residual_p90_over_mask_extent": ray_p90 / float(mask_extent),
        "ray_residual_max_over_mask_extent": ray_max / float(mask_extent),
        "P_O_shape": [0, 3],
    }
    contract = {
        "format": POSE_MASK_OBJECT_FRAME_VERSION,
        "observable_inputs": ["mask", "K", "T_W2C", "optional_gravity"],
        "forbidden_inputs": [
            "P_W",
            "point_confidence",
            "Scan.obj",
            "T_Scan2W",
            "target_ss",
            "target_slat",
            "gt_mesh_bounds",
        ],
        "point_cloud_consumed": False,
        "camera_convention": "COLMAP pinhole +z forward; T_W2C",
        "axis_rule": (
            "up=gravity_or_camera_up_with_orbit_normal; "
            "front=reference_camera_from_mask_ray_center"
        ),
        "center_rule": "least_squares_intersection_of_mask_centroid_rays",
        "scale_rule": "median_mask_extent*padding/expected_extent",
        "config": asdict(rules),
    }
    return RuntimeObjectFrame(
        T_O2W=T_O2W,
        T_W2O=T_W2O,
        T_O2C=T_O2C,
        T_C2O=T_C2O,
        P_O=empty_points,
        point_keep_mask=empty_keep,
        stats=stats,
        contract=contract,
    )


def prepare_pose_mask_runtime_object_observation(
    images: Sequence[np.ndarray],
    masks: Sequence[np.ndarray],
    intrinsics: np.ndarray,
    T_W2C: np.ndarray,
    *,
    camera_models: Sequence[str] | None = None,
    distortion_coefficients: Sequence[Sequence[float]] | None = None,
    frame_config: PoseMaskObjectFrameConfig | None = None,
    gravity_up_W: np.ndarray | None = None,
    reference_view_index: int | None = 0,
    feature_resolution: int = 518,
    foreground_margin: float = 1.10,
    alpha_threshold: float = 0.80,
) -> RuntimeObjectObservation:
    """Prepare model-visible arrays while guaranteeing no point-cloud read."""

    undistorted_images, undistorted_masks, undistorted_k, distortion_records = (
        undistort_rgb_mask_views(
            images,
            masks,
            intrinsics,
            camera_models=camera_models,
            distortion_coefficients=distortion_coefficients,
        )
    )
    if reference_view_index is None:
        reference_view_index = max(
            range(len(undistorted_masks)),
            key=lambda index: (
                int(np.count_nonzero(np.asarray(undistorted_masks[index]) > 127)),
                -index,
            ),
        )
    frame = canonicalize_pose_mask_runtime_object_frame(
        undistorted_k,
        T_W2C,
        undistorted_masks,
        config=frame_config,
        gravity_up_W=gravity_up_W,
        reference_view_index=int(reference_view_index),
    )
    prepared = prepare_shared_object_arrays(
        undistorted_images,
        undistorted_masks,
        resolution=int(feature_resolution),
        foreground_margin=float(foreground_margin),
        alpha_threshold=float(alpha_threshold),
    )
    feature_k = transform_intrinsics(
        undistorted_k.astype(np.float32), prepared.source_to_feature_affines
    )
    prepared_rgb = np.stack(
        [np.asarray(image, dtype=np.uint8) for image in prepared.images]
    )
    condition: dict[str, Any] = {
        "format": POSE_MASK_INPUT_FRONTEND_VERSION,
        "runtime_frame": frame.record(),
        "shared_image_geometry": prepared.geometry_record(),
        "undistortion": distortion_records,
        "prepared_rgb_sha256": array_sha256(prepared_rgb),
        "prepared_mask_sha256": array_sha256(prepared.masks),
        "K_feature_sha256": array_sha256(feature_k),
        "T_O2C_sha256": array_sha256(frame.T_O2C),
        "T_O2C_lifting_sha256": array_sha256(frame.T_O2C_lifting),
        "P_O_sha256": array_sha256(frame.P_O),
        "point_cloud_consumed": False,
        "condition_scope": "pose+mask observable inputs only; no P_W/Scan/GT/target",
    }
    condition["condition_sha256"] = canonical_json_sha256(condition)
    return RuntimeObjectObservation(
        prepared_views=prepared,
        intrinsics=feature_k,
        frame=frame,
        undistortion=distortion_records,
        condition_record=condition,
    )


__all__ = [
    "POSE_MASK_INPUT_FRONTEND_VERSION",
    "POSE_MASK_OBJECT_FRAME_VERSION",
    "PoseMaskObjectFrameConfig",
    "canonicalize_pose_mask_runtime_object_frame",
    "prepare_pose_mask_runtime_object_observation",
]
