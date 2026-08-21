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
from manual_mesh_reconstruction.canonicalization import (
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


LEGACY_Y_UP_OBJECT_FRAME_POLICY = "legacy_y_up_v1"
OFFICIAL_COMPATIBLE_Z_UP_OBJECT_FRAME_POLICY = "official_compatible_z_up_v1"
POSE_MASK_OBJECT_FRAME_POLICIES = (
    LEGACY_Y_UP_OBJECT_FRAME_POLICY,
    OFFICIAL_COMPATIBLE_Z_UP_OBJECT_FRAME_POLICY,
)

LEGACY_POSE_MASK_OBJECT_FRAME_VERSION = (
    "pose_point_depth_mv.pose_mask_object_frame.v1"
)
OFFICIAL_COMPATIBLE_POSE_MASK_OBJECT_FRAME_VERSION = (
    "manual_mesh_reconstruction.pose_mask_object_frame.official_z_up.v1"
)
LEGACY_POSE_MASK_INPUT_FRONTEND_VERSION = (
    "pose_point_depth_mv.pose_mask_input_frontend.v1"
)
OFFICIAL_COMPATIBLE_POSE_MASK_INPUT_FRONTEND_VERSION = (
    "manual_mesh_reconstruction.pose_mask_input_frontend.official_z_up.v1"
)

# Backwards-compatible aliases for callers that intentionally retain the old
# direct pose-mask API.  The unified manual pipeline explicitly requests the
# official-compatible policy instead of changing this low-level default.
POSE_MASK_OBJECT_FRAME_VERSION = LEGACY_POSE_MASK_OBJECT_FRAME_VERSION
POSE_MASK_INPUT_FRONTEND_VERSION = LEGACY_POSE_MASK_INPUT_FRONTEND_VERSION

# Homogeneous proper rotation from the new model-O into the legacy runtime-O.
# It is the exact axis contract validated by the Snoopy paired reconstruction:
# model +X -> legacy +X, model +Y -> legacy -Z, model +Z -> legacy +Y.
Q_OFFICIAL_MODEL_O_TO_LEGACY_RUNTIME_O = np.asarray(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


@dataclass(frozen=True)
class PoseMaskObjectFrameConfig:
    """Frozen rules for estimating O without reading a point cloud."""

    mask_threshold: float = 0.5
    expected_object_extent: float = 0.90
    scale_padding: float = 1.05
    minimum_scale: float = 1.0e-6
    object_frame_policy: str = LEGACY_Y_UP_OBJECT_FRAME_POLICY

    def validate(self) -> None:
        if not 0.0 < float(self.mask_threshold) < 1.0:
            raise ValueError("mask_threshold must be in (0,1)")
        if float(self.expected_object_extent) <= 0.0:
            raise ValueError("expected_object_extent must be positive")
        if float(self.scale_padding) < 1.0:
            raise ValueError("scale_padding must be at least one")
        if float(self.minimum_scale) <= 0.0:
            raise ValueError("minimum_scale must be positive")
        if str(self.object_frame_policy) not in POSE_MASK_OBJECT_FRAME_POLICIES:
            raise ValueError(
                "unsupported pose-mask object-frame policy: "
                f"{self.object_frame_policy!r}"
            )


def pose_mask_object_frame_version(policy: str) -> str:
    value = str(policy)
    if value == LEGACY_Y_UP_OBJECT_FRAME_POLICY:
        return LEGACY_POSE_MASK_OBJECT_FRAME_VERSION
    if value == OFFICIAL_COMPATIBLE_Z_UP_OBJECT_FRAME_POLICY:
        return OFFICIAL_COMPATIBLE_POSE_MASK_OBJECT_FRAME_VERSION
    raise ValueError(f"unsupported pose-mask object-frame policy: {value!r}")


def pose_mask_input_frontend_version(policy: str) -> str:
    value = str(policy)
    if value == LEGACY_Y_UP_OBJECT_FRAME_POLICY:
        return LEGACY_POSE_MASK_INPUT_FRONTEND_VERSION
    if value == OFFICIAL_COMPATIBLE_Z_UP_OBJECT_FRAME_POLICY:
        return OFFICIAL_COMPATIBLE_POSE_MASK_INPUT_FRONTEND_VERSION
    raise ValueError(f"unsupported pose-mask object-frame policy: {value!r}")


def apply_pose_mask_object_frame_policy(
    legacy_rotation_O2W: np.ndarray,
    *,
    policy: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return the versioned model-O basis without changing center or scale."""

    legacy = np.asarray(legacy_rotation_O2W, dtype=np.float64)
    if legacy.shape != (3, 3) or not np.isfinite(legacy).all():
        raise ValueError("legacy pose-mask rotation must be finite [3,3]")
    if not np.allclose(legacy.T @ legacy, np.eye(3), atol=1.0e-8):
        raise ValueError("legacy pose-mask rotation is not orthonormal")
    if not np.isclose(np.linalg.det(legacy), 1.0, atol=1.0e-8):
        raise ValueError("legacy pose-mask rotation is not proper")

    value = str(policy)
    if value == LEGACY_Y_UP_OBJECT_FRAME_POLICY:
        rotation = legacy.copy()
        mapping = {
            "model_plus_x": "legacy_runtime_plus_x",
            "model_plus_y": "legacy_runtime_plus_y_estimated_up",
            "model_plus_z": "legacy_runtime_plus_z_reference_front",
        }
        q = np.eye(4, dtype=np.float64)
    elif value == OFFICIAL_COMPATIBLE_Z_UP_OBJECT_FRAME_POLICY:
        q = Q_OFFICIAL_MODEL_O_TO_LEGACY_RUNTIME_O.copy()
        rotation = legacy @ q[:3, :3]
        mapping = {
            "model_plus_x": "legacy_runtime_plus_x",
            "model_plus_y": "legacy_runtime_minus_z_reference_front",
            "model_plus_z": "legacy_runtime_plus_y_estimated_up",
        }
    else:
        raise ValueError(f"unsupported pose-mask object-frame policy: {value!r}")

    determinant = float(np.linalg.det(rotation))
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-8):
        raise RuntimeError("pose-mask model-O rotation is not orthonormal")
    if not np.isclose(determinant, 1.0, atol=1.0e-8):
        raise RuntimeError("pose-mask model-O rotation is not proper")
    return rotation, {
        "object_frame_policy": value,
        "axis_mapping": mapping,
        "Q_model_O_to_legacy_runtime_O": q.tolist(),
        "legacy_rotation_O2W": legacy.tolist(),
        "model_rotation_O2W": rotation.tolist(),
        "determinant": determinant,
        "center_changed": False,
        "scale_changed": False,
    }


def bind_runtime_object_frame_to_cameras(
    source: RuntimeObjectFrame,
    T_W2C: np.ndarray,
    *,
    selected_source_view_indices: Sequence[int] | None = None,
) -> RuntimeObjectFrame:
    """Bind an all-view O frame to selected cameras without estimating O again."""

    poses = np.asarray(T_W2C, dtype=np.float64)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4) or not np.isfinite(poses).all():
        raise ValueError("selected T_W2C must be finite [V,4,4]")
    T_O2W = np.asarray(source.T_O2W, dtype=np.float64).copy()
    T_W2O = np.asarray(source.T_W2O, dtype=np.float64).copy()
    T_O2C = np.matmul(poses, T_O2W[None])
    T_C2O = np.matmul(T_W2O[None], np.linalg.inv(poses))
    stats = dict(source.stats)
    stats["all_view_o_frozen_before_selection"] = True
    stats["selected_camera_count"] = int(len(poses))
    if selected_source_view_indices is not None:
        stats["selected_source_view_indices"] = [
            int(value) for value in selected_source_view_indices
        ]
    contract = dict(source.contract)
    contract["selection_binding_rule"] = (
        "O estimated once from every eligible input view; selected cameras are "
        "rebound through T_O2C=T_W2C@T_O2W without re-estimating O"
    )
    return RuntimeObjectFrame(
        T_O2W=T_O2W,
        T_W2O=T_W2O,
        T_O2C=T_O2C,
        T_C2O=T_C2O,
        P_O=np.asarray(source.P_O).copy(),
        point_keep_mask=np.asarray(source.point_keep_mask).copy(),
        stats=stats,
        contract=contract,
    )


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
    legacy_rotation, legacy_axis_stats = _estimate_axes(
        center,
        poses,
        gravity_up_W=gravity_up_W,
        reference_view_index=int(reference_view_index),
    )
    rotation, axis_policy_stats = apply_pose_mask_object_frame_policy(
        legacy_rotation,
        policy=str(rules.object_frame_policy),
    )
    axis_stats = {
        **legacy_axis_stats,
        **axis_policy_stats,
        "estimation_view_count": int(view_count),
        "estimation_used_view_indices": [
            int(value) for value in ray_stats["used_view_indices"]
        ],
        "model_up_axis": (
            "+Z"
            if str(rules.object_frame_policy)
            == OFFICIAL_COMPATIBLE_Z_UP_OBJECT_FRAME_POLICY
            else "+Y"
        ),
    }
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
        "format": pose_mask_object_frame_version(str(rules.object_frame_policy)),
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
            "legacy_y_up_v1: +Y=estimated_up,+Z=reference_front; "
            "official_compatible_z_up_v1: +X=legacy+X,+Y=legacy-Z,+Z=legacy+Y"
        ),
        "object_frame_policy": str(rules.object_frame_policy),
        "o_estimation_domain": "all_eligible_input_views_before_model_view_selection",
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
    precomputed_object_frame: RuntimeObjectFrame | None = None,
    selected_source_view_indices: Sequence[int] | None = None,
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
    if precomputed_object_frame is None:
        frame = canonicalize_pose_mask_runtime_object_frame(
            undistorted_k,
            T_W2C,
            undistorted_masks,
            config=frame_config,
            gravity_up_W=gravity_up_W,
            reference_view_index=int(reference_view_index),
        )
    else:
        frame = bind_runtime_object_frame_to_cameras(
            precomputed_object_frame,
            T_W2C,
            selected_source_view_indices=selected_source_view_indices,
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
    effective_policy = str(
        frame.contract.get(
            "object_frame_policy",
            (frame_config or PoseMaskObjectFrameConfig()).object_frame_policy,
        )
    )
    condition: dict[str, Any] = {
        "format": pose_mask_input_frontend_version(effective_policy),
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
    "LEGACY_POSE_MASK_INPUT_FRONTEND_VERSION",
    "LEGACY_POSE_MASK_OBJECT_FRAME_VERSION",
    "LEGACY_Y_UP_OBJECT_FRAME_POLICY",
    "OFFICIAL_COMPATIBLE_POSE_MASK_INPUT_FRONTEND_VERSION",
    "OFFICIAL_COMPATIBLE_POSE_MASK_OBJECT_FRAME_VERSION",
    "OFFICIAL_COMPATIBLE_Z_UP_OBJECT_FRAME_POLICY",
    "POSE_MASK_OBJECT_FRAME_POLICIES",
    "POSE_MASK_INPUT_FRONTEND_VERSION",
    "POSE_MASK_OBJECT_FRAME_VERSION",
    "PoseMaskObjectFrameConfig",
    "Q_OFFICIAL_MODEL_O_TO_LEGACY_RUNTIME_O",
    "apply_pose_mask_object_frame_policy",
    "bind_runtime_object_frame_to_cameras",
    "canonicalize_pose_mask_runtime_object_frame",
    "pose_mask_input_frontend_version",
    "pose_mask_object_frame_version",
    "prepare_pose_mask_runtime_object_observation",
]
