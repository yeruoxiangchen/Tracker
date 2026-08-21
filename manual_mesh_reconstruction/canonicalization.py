from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Any, Sequence

import numpy as np

from ar_ss_flow.shared_object_preprocessing import (
    SharedObjectViews,
    canonical_json_sha256,
    prepare_shared_object_arrays,
    transform_intrinsics,
)


RUNTIME_OBJECT_FRAME_VERSION = "pose_point_depth_mv.runtime_object_frame.v1"
REAL_INPUT_FRONTEND_VERSION = "pose_point_depth_mv.real_input_frontend.v2"
DISTORTION_FRONTEND_VERSION = "pose_point_depth_mv.distortion_frontend.v1"


class InsufficientObjectPointsError(RuntimeError):
    """The observation cannot support a stable deployable object frame."""

    def __init__(self, *, stage: str, available: int, required: int) -> None:
        self.stage = str(stage)
        self.available = int(available)
        self.required = int(required)
        super().__init__(
            f"insufficient {self.stage} object points: "
            f"{self.available} < {self.required}"
        )


@dataclass(frozen=True)
class RuntimeObjectFrameConfig:
    """Frozen, deployable rules for deriving object space from observations."""

    mask_threshold: float = 0.5
    min_mask_observations: int = 2
    min_mask_support_ratio: float = 0.60
    min_object_points: int = 32
    point_trim_quantile: float = 0.98
    extent_quantile: float = 0.02
    point_center_weight: float = 0.75
    expected_object_extent: float = 0.90
    scale_padding: float = 1.05
    minimum_scale: float = 1.0e-6

    def validate(self) -> None:
        if not 0.0 < float(self.mask_threshold) < 1.0:
            raise ValueError("mask_threshold must be in (0,1)")
        if int(self.min_mask_observations) <= 0:
            raise ValueError("min_mask_observations must be positive")
        if not 0.0 < float(self.min_mask_support_ratio) <= 1.0:
            raise ValueError("min_mask_support_ratio must be in (0,1]")
        if int(self.min_object_points) < 4:
            raise ValueError("min_object_points must be at least four")
        if not 0.5 < float(self.point_trim_quantile) <= 1.0:
            raise ValueError("point_trim_quantile must be in (0.5,1]")
        if not 0.0 <= float(self.extent_quantile) < 0.5:
            raise ValueError("extent_quantile must be in [0,0.5)")
        if not 0.0 <= float(self.point_center_weight) <= 1.0:
            raise ValueError("point_center_weight must be in [0,1]")
        if float(self.expected_object_extent) <= 0.0:
            raise ValueError("expected_object_extent must be positive")
        if float(self.scale_padding) < 1.0:
            raise ValueError("scale_padding must be at least one")
        if float(self.minimum_scale) <= 0.0:
            raise ValueError("minimum_scale must be positive")


@dataclass
class RuntimeObjectFrame:
    T_O2W: np.ndarray
    T_W2O: np.ndarray
    T_O2C: np.ndarray
    T_C2O: np.ndarray
    P_O: np.ndarray
    point_keep_mask: np.ndarray
    stats: dict[str, Any]
    contract: dict[str, Any]

    @property
    def T_O2C_lifting(self) -> np.ndarray:
        """Projectively normalized O-to-camera matrices consumed by Native v2."""

        return normalize_similarity_extrinsics(self.T_O2C)

    def record(self) -> dict[str, Any]:
        payload = {
            "format": RUNTIME_OBJECT_FRAME_VERSION,
            "contract": self.contract,
            "contract_sha256": canonical_json_sha256(self.contract),
            "T_O2W": self.T_O2W.tolist(),
            "T_W2O": self.T_W2O.tolist(),
            "T_O2C_sha256": array_sha256(self.T_O2C),
            "T_O2C_lifting_sha256": array_sha256(self.T_O2C_lifting),
            "T_C2O_sha256": array_sha256(self.T_C2O),
            "P_O_sha256": array_sha256(self.P_O),
            "point_keep_mask_sha256": array_sha256(self.point_keep_mask),
            "stats": self.stats,
        }
        payload["frame_sha256"] = canonical_json_sha256(payload)
        return payload


@dataclass
class RuntimeObjectObservation:
    prepared_views: SharedObjectViews
    intrinsics: np.ndarray
    frame: RuntimeObjectFrame
    undistortion: list[dict[str, Any]]
    condition_record: dict[str, Any]

    @property
    def condition_sha256(self) -> str:
        return str(self.condition_record["condition_sha256"])


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(array.view(np.uint8).tobytes(order="C"))
    return digest.hexdigest()


def apply_transform(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64)
    matrix = np.asarray(transform, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("points must be [N,3]")
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("transform must be a finite [4,4] matrix")
    return values @ matrix[:3, :3].T + matrix[:3, 3]


def similarity_scale(transform: np.ndarray) -> float:
    matrix = np.asarray(transform, dtype=np.float64)
    linear = matrix[:3, :3]
    return float(np.cbrt(np.linalg.det(linear)))


def validate_proper_similarity(transform: np.ndarray, *, name: str) -> np.ndarray:
    matrix = np.asarray(transform, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError(f"{name} must be a finite [4,4] matrix")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1.0e-9):
        raise ValueError(f"{name} has an invalid homogeneous row")
    scale = similarity_scale(matrix)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"{name} must have positive scale")
    rotation = matrix[:3, :3] / scale
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-6):
        raise ValueError(f"{name} is not an isotropic similarity")
    if float(np.linalg.det(rotation)) <= 0.0:
        raise ValueError(f"{name} must contain a proper rotation")
    return matrix


def normalize_similarity_extrinsics(T_O2C: np.ndarray) -> np.ndarray:
    """Remove projectively irrelevant object scale from O-to-camera poses.

    Runtime-O is a similarity frame, so ``T_O2C`` has an isotropic scale in
    its first three rows. Perspective projection is unchanged when all three
    rows are divided by that scale. Native v2 lifting must consume this rigid,
    gauge-invariant derivative rather than the physical similarity matrix.
    """

    poses = np.asarray(T_O2C, dtype=np.float64)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError("T_O2C must be [V,4,4]")
    normalized = np.empty_like(poses)
    for index, pose in enumerate(poses):
        matrix = validate_proper_similarity(pose, name=f"T_O2C[{index}]")
        scale = similarity_scale(matrix)
        normalized[index] = matrix
        normalized[index, :3, :] /= scale
        rotation = normalized[index, :3, :3]
        # Preserve frozen v2 cache bits; repair only tolerated float32 drift.
        if not np.allclose(
            rotation.T @ rotation, np.eye(3), rtol=1.0e-7, atol=1.0e-7
        ):
            left, _, right = np.linalg.svd(rotation)
            projected = left @ right
            if float(np.linalg.det(projected)) <= 0.0:
                left[:, -1] *= -1.0
                projected = left @ right
            normalized[index, :3, :3] = projected
            if not np.allclose(
                projected.T @ projected, np.eye(3), rtol=0.0, atol=1.0e-12
            ) or not np.isclose(
                np.linalg.det(projected), 1.0, rtol=0.0, atol=1.0e-12
            ):
                raise RuntimeError(
                    f"T_O2C[{index}] could not be projected to a proper rotation"
                )
    return np.ascontiguousarray(normalized)


def invert_similarity(transform: np.ndarray) -> np.ndarray:
    matrix = validate_proper_similarity(transform, name="similarity")
    return np.linalg.inv(matrix)


def _validate_camera_inputs(
    intrinsics: np.ndarray,
    T_W2C: np.ndarray,
    view_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    k_all = np.asarray(intrinsics, dtype=np.float64)
    poses = np.asarray(T_W2C, dtype=np.float64)
    if k_all.shape != (view_count, 3, 3):
        raise ValueError(f"intrinsics must be [{view_count},3,3], got {k_all.shape}")
    if poses.shape != (view_count, 4, 4):
        raise ValueError(f"T_W2C must be [{view_count},4,4], got {poses.shape}")
    if not np.isfinite(k_all).all() or not np.isfinite(poses).all():
        raise ValueError("camera matrices contain non-finite values")
    if np.any(k_all[:, 0, 0] <= 0.0) or np.any(k_all[:, 1, 1] <= 0.0):
        raise ValueError("camera focal lengths must be positive")
    for index, pose in enumerate(poses):
        rotation = pose[:3, :3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-5):
            raise ValueError(f"T_W2C[{index}] rotation is not orthonormal")
        if float(np.linalg.det(rotation)) <= 0.0:
            raise ValueError(f"T_W2C[{index}] rotation is not proper")
        if not np.allclose(pose[3], [0.0, 0.0, 0.0, 1.0], atol=1.0e-8):
            raise ValueError(f"T_W2C[{index}] has an invalid homogeneous row")
    return k_all, poses


def _as_uint8_rgb(value: np.ndarray, *, index: int) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"image[{index}] must be [H,W,3]")
    if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
        raise ValueError(f"image[{index}] must be finite numeric data")
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(array)


def _as_uint8_mask(
    value: np.ndarray, *, shape: tuple[int, int], index: int
) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 2 or array.shape != shape:
        raise ValueError(f"mask[{index}] must have shape {shape}, got {array.shape}")
    if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
        raise ValueError(f"mask[{index}] must be finite numeric data")
    if array.dtype != np.uint8:
        scale = 255.0 if (float(array.max()) if array.size else 0.0) <= 1.0 else 1.0
        array = np.clip(array * scale, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(array)


def _opencv_distortion(model: str, coefficients: Sequence[float]) -> np.ndarray:
    name = str(model).upper()
    values = [float(value) for value in coefficients]
    if name in {"PINHOLE", "SIMPLE_PINHOLE"}:
        if values and any(abs(value) > 1.0e-15 for value in values):
            raise ValueError(f"{name} must not carry nonzero distortion coefficients")
        return np.zeros(5, dtype=np.float64)
    if name == "SIMPLE_RADIAL" and len(values) == 1:
        return np.asarray([values[0], 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    if name == "RADIAL" and len(values) == 2:
        return np.asarray([values[0], values[1], 0.0, 0.0, 0.0], dtype=np.float64)
    if name == "OPENCV" and len(values) in {4, 5}:
        return np.asarray(values, dtype=np.float64)
    raise ValueError(f"unsupported camera distortion contract: {name} {values}")


def _undistortion_map(
    intrinsic: np.ndarray,
    distortion: np.ndarray,
    *,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray]:
    import cv2

    return cv2.initUndistortRectifyMap(
        np.asarray(intrinsic, dtype=np.float64),
        np.asarray(distortion, dtype=np.float64),
        None,
        np.asarray(intrinsic, dtype=np.float64),
        (int(width), int(height)),
        cv2.CV_32FC1,
    )


def undistort_mask_view(
    mask: np.ndarray,
    intrinsic: np.ndarray,
    *,
    camera_model: str,
    distortion_coefficients: Sequence[float],
) -> np.ndarray:
    """Apply the exact runtime mask undistortion used by the RGB/mask frontend."""

    value = np.asarray(mask)
    if value.ndim != 2:
        raise ValueError(f"mask must be [H,W], got {value.shape}")
    prepared = _as_uint8_mask(value, shape=value.shape, index=0)
    k = np.asarray(intrinsic, dtype=np.float64)
    if k.shape != (3, 3) or not np.isfinite(k).all():
        raise ValueError("intrinsic must be a finite [3,3] matrix")
    distortion = _opencv_distortion(camera_model, distortion_coefficients)
    if not np.any(np.abs(distortion) > 1.0e-15):
        return np.ascontiguousarray(prepared)

    import cv2

    height, width = prepared.shape
    map_x, map_y = _undistortion_map(
        k, distortion, width=width, height=height
    )
    return np.ascontiguousarray(
        cv2.remap(
            prepared,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
    )


def undistort_rgb_mask_views(
    images: Sequence[np.ndarray],
    masks: Sequence[np.ndarray],
    intrinsics: np.ndarray,
    *,
    camera_models: Sequence[str] | None = None,
    distortion_coefficients: Sequence[Sequence[float]] | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray, list[dict[str, Any]]]:
    """Undistort aligned RGB/mask pairs with one frozen map per view.

    The derivative keeps the source image dimensions and the supplied pinhole
    K.  RGB and soft masks use the same geometric map; only interpolation mode
    differs.  Pinhole inputs are a bit-exact no-op.
    """

    if len(images) != len(masks) or not images:
        raise ValueError("images/masks must be non-empty and aligned")
    view_count = len(images)
    k_all = np.asarray(intrinsics, dtype=np.float64)
    if k_all.shape != (view_count, 3, 3) or not np.isfinite(k_all).all():
        raise ValueError("intrinsics must be finite [V,3,3]")
    models = list(camera_models or ["PINHOLE"] * view_count)
    distortions = list(distortion_coefficients or [[] for _ in range(view_count)])
    if len(models) != view_count or len(distortions) != view_count:
        raise ValueError("camera model/distortion view count mismatch")

    output_images: list[np.ndarray] = []
    output_masks: list[np.ndarray] = []
    records: list[dict[str, Any]] = []
    for index, (raw_image, raw_mask, model, coefficients) in enumerate(
        zip(images, masks, models, distortions)
    ):
        image = _as_uint8_rgb(raw_image, index=index)
        mask = _as_uint8_mask(raw_mask, shape=image.shape[:2], index=index)
        distortion = _opencv_distortion(model, coefficients)
        active = bool(np.any(np.abs(distortion) > 1.0e-15))
        if active:
            import cv2

            height, width = image.shape[:2]
            map_x, map_y = _undistortion_map(
                k_all[index], distortion, width=width, height=height
            )
            image = cv2.remap(
                image,
                map_x,
                map_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            mask = cv2.remap(
                mask,
                map_x,
                map_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
        output_images.append(np.ascontiguousarray(image))
        output_masks.append(np.ascontiguousarray(mask))
        records.append(
            {
                "format": DISTORTION_FRONTEND_VERSION,
                "view_index": index,
                "source_model": str(model).upper(),
                "source_distortion": [float(value) for value in coefficients],
                "active": active,
                "output_size_wh": [int(image.shape[1]), int(image.shape[0])],
                "output_K": k_all[index].tolist(),
            }
        )
    return output_images, output_masks, k_all.copy(), records


def _normalize(vector: np.ndarray, *, name: str) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(value))
    if not math.isfinite(norm) or norm <= 1.0e-10:
        raise ValueError(f"cannot normalize degenerate {name}")
    return value / norm


def _sample_bilinear(image: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    values = np.asarray(image, dtype=np.float64)
    height, width = values.shape
    x0 = np.floor(u).astype(np.int64)
    y0 = np.floor(v).astype(np.int64)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    wx = u - x0
    wy = v - y0
    return (
        values[y0, x0] * (1.0 - wx) * (1.0 - wy)
        + values[y0, x1] * wx * (1.0 - wy)
        + values[y1, x0] * (1.0 - wx) * wy
        + values[y1, x1] * wx * wy
    )


def filter_mask_supported_points(
    P_W: np.ndarray,
    intrinsics: np.ndarray,
    T_W2C: np.ndarray,
    masks: Sequence[np.ndarray],
    *,
    config: RuntimeObjectFrameConfig,
    point_view_visibility: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    points = np.asarray(P_W, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("P_W must be finite [N,3]")
    if len(points) < int(config.min_object_points):
        raise InsufficientObjectPointsError(
            stage="raw",
            available=len(points),
            required=int(config.min_object_points),
        )
    view_count = len(masks)
    k_all, poses = _validate_camera_inputs(intrinsics, T_W2C, view_count)
    visibility = (
        None
        if point_view_visibility is None
        else np.asarray(point_view_visibility, dtype=bool)
    )
    if visibility is not None and visibility.shape != (view_count, len(points)):
        raise ValueError("point_view_visibility must be [V,N]")

    observed = np.zeros(len(points), dtype=np.int32)
    supported = np.zeros(len(points), dtype=np.int32)
    homogeneous = np.concatenate((points, np.ones((len(points), 1))), axis=1)
    per_view: list[dict[str, int]] = []
    for view_index, raw_mask in enumerate(masks):
        mask = np.asarray(raw_mask, dtype=np.float64)
        if mask.ndim != 2 or not np.isfinite(mask).all():
            raise ValueError(f"mask[{view_index}] must be a finite [H,W] array")
        if float(mask.max()) > 1.0:
            mask = mask / 255.0
        camera = (poses[view_index] @ homogeneous.T).T[:, :3]
        depth = camera[:, 2]
        safe_depth = np.where(depth > 1.0e-10, depth, 1.0)
        pixels = (k_all[view_index] @ camera.T).T
        u = pixels[:, 0] / safe_depth
        v = pixels[:, 1] / safe_depth
        height, width = mask.shape
        valid = (
            (depth > 1.0e-8)
            & (u >= 0.0)
            & (u <= width - 1.0)
            & (v >= 0.0)
            & (v <= height - 1.0)
        )
        if visibility is not None:
            valid &= visibility[view_index]
        ids = np.nonzero(valid)[0]
        observed[ids] += 1
        values = _sample_bilinear(mask, u[ids], v[ids]) if len(ids) else np.empty(0)
        positive = ids[values >= float(config.mask_threshold)]
        supported[positive] += 1
        per_view.append(
            {
                "observed_point_count": int(len(ids)),
                "mask_supported_point_count": int(len(positive)),
            }
        )

    ratio = supported.astype(np.float64) / np.maximum(observed, 1)
    keep = (observed >= int(config.min_mask_observations)) & (
        ratio >= float(config.min_mask_support_ratio)
    )
    if int(keep.sum()) < int(config.min_object_points):
        raise InsufficientObjectPointsError(
            stage="mask-supported",
            available=int(keep.sum()),
            required=int(config.min_object_points),
        )
    selected_ratio = ratio[keep]
    return keep, {
        "all_point_count": int(len(points)),
        "mask_supported_point_count": int(keep.sum()),
        "mask_supported_fraction": float(keep.mean()),
        "support_ratio_median": float(np.median(selected_ratio)),
        "observation_count_median": float(np.median(observed[keep])),
        "per_view": per_view,
    }


def _geometric_median(
    points: np.ndarray, weights: np.ndarray | None = None
) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64)
    if weights is None:
        weight = np.ones(len(values), dtype=np.float64)
    else:
        weight = np.asarray(weights, dtype=np.float64)
        if (
            weight.shape != (len(values),)
            or not np.isfinite(weight).all()
            or np.any(weight < 0.0)
        ):
            raise ValueError("point weights must be finite nonnegative [N]")
        if float(weight.sum()) <= 0.0:
            raise ValueError("point weights must contain positive mass")
    estimate = np.average(values, axis=0, weights=weight)
    for _ in range(64):
        distance = np.linalg.norm(values - estimate, axis=1)
        near = distance <= 1.0e-12
        if np.any(near):
            candidate = np.average(values[near], axis=0, weights=weight[near])
        else:
            inverse = weight / np.maximum(distance, 1.0e-12)
            candidate = np.sum(values * inverse[:, None], axis=0) / float(inverse.sum())
        if float(np.linalg.norm(candidate - estimate)) <= 1.0e-10 * max(
            1.0, float(np.linalg.norm(estimate))
        ):
            estimate = candidate
            break
        estimate = candidate
    return estimate


def _mask_centroid_ray_center(
    masks: Sequence[np.ndarray],
    intrinsics: np.ndarray,
    T_W2C: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    projectors = []
    right_hand_sides = []
    used_views = []
    for index, raw_mask in enumerate(masks):
        mask = np.asarray(raw_mask, dtype=np.float64)
        if float(mask.max()) > 1.0:
            mask = mask / 255.0
        foreground = mask >= float(threshold)
        ys, xs = np.nonzero(foreground)
        if len(xs) < 4:
            continue
        weights = mask[ys, xs]
        u = float(np.average(xs, weights=weights))
        v = float(np.average(ys, weights=weights))
        c2w = np.linalg.inv(T_W2C[index])
        direction_c = np.linalg.inv(intrinsics[index]) @ np.asarray([u, v, 1.0])
        direction_w = _normalize(c2w[:3, :3] @ direction_c, name="mask centroid ray")
        camera_center = c2w[:3, 3]
        projector = np.eye(3) - np.outer(direction_w, direction_w)
        projectors.append(projector)
        right_hand_sides.append(projector @ camera_center)
        used_views.append(index)
    if len(projectors) < 2:
        raise RuntimeError("at least two non-empty masks are required for ray center")
    system = np.sum(projectors, axis=0)
    condition = float(np.linalg.cond(system))
    if not math.isfinite(condition) or condition > 1.0e10:
        raise RuntimeError(f"mask centroid rays are degenerate: condition={condition}")
    center = np.linalg.solve(system, np.sum(right_hand_sides, axis=0))
    residuals = []
    for projector, rhs in zip(projectors, right_hand_sides):
        residuals.append(float(np.linalg.norm(projector @ center - rhs)))
    return center, {
        "used_view_indices": used_views,
        "linear_system_condition": condition,
        "ray_residuals": residuals,
        "ray_residual_median": float(np.median(residuals)),
        "ray_residual_p90": float(np.percentile(residuals, 90)),
        "ray_residual_max": float(np.max(residuals)),
    }


def _estimate_axes(
    center: np.ndarray,
    T_W2C: np.ndarray,
    *,
    gravity_up_W: np.ndarray | None,
    reference_view_index: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    c2w = np.linalg.inv(T_W2C)
    camera_centers = c2w[:, :3, 3]
    camera_ups = -c2w[:, :3, 1]
    mean_camera_up = np.mean(camera_ups, axis=0)
    if float(np.linalg.norm(mean_camera_up)) <= 1.0e-8:
        mean_camera_up = camera_ups[reference_view_index]
    mean_camera_up = _normalize(mean_camera_up, name="camera up")

    centered_cameras = camera_centers - np.mean(camera_centers, axis=0)
    covariance = centered_cameras.T @ centered_cameras / max(len(camera_centers) - 1, 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    orbit_normal = _normalize(eigenvectors[:, 0], name="orbit normal")
    anchor_up = (
        _normalize(np.asarray(gravity_up_W, dtype=np.float64), name="gravity up")
        if gravity_up_W is not None
        else mean_camera_up
    )
    if float(np.dot(orbit_normal, anchor_up)) < 0.0:
        orbit_normal = -orbit_normal
    agreement = float(np.dot(orbit_normal, anchor_up))
    if agreement >= 0.5:
        up = _normalize(orbit_normal + anchor_up, name="blended up")
        up_source = (
            "gravity_plus_orbit" if gravity_up_W is not None else "camera_up_plus_orbit"
        )
    else:
        up = anchor_up
        up_source = "gravity" if gravity_up_W is not None else "camera_up"

    front = camera_centers[reference_view_index] - center
    front = front - up * float(np.dot(front, up))
    if float(np.linalg.norm(front)) <= 1.0e-8:
        front = -c2w[reference_view_index, :3, 2]
        front = front - up * float(np.dot(front, up))
    z_axis = _normalize(front, name="reference-view front")
    x_axis = _normalize(np.cross(up, z_axis), name="object right")
    z_axis = _normalize(np.cross(x_axis, up), name="object front")
    rotation = np.stack((x_axis, up, z_axis), axis=1)
    if float(np.linalg.det(rotation)) <= 0.0:
        raise RuntimeError("estimated object frame is not right handed")
    return rotation, {
        "up_source": up_source,
        "orbit_camera_up_agreement": agreement,
        "orbit_covariance_eigenvalues": eigenvalues.tolist(),
        "reference_view_index": int(reference_view_index),
    }


def _mask_extent_at_center(
    center: np.ndarray,
    masks: Sequence[np.ndarray],
    intrinsics: np.ndarray,
    T_W2C: np.ndarray,
    threshold: float,
) -> tuple[float, list[float]]:
    spans = []
    center_h = np.concatenate((center, [1.0]))
    for index, raw_mask in enumerate(masks):
        mask = np.asarray(raw_mask, dtype=np.float64)
        if float(mask.max()) > 1.0:
            mask = mask / 255.0
        ys, xs = np.nonzero(mask >= float(threshold))
        if len(xs) < 4:
            continue
        depth = float((T_W2C[index] @ center_h)[2])
        if depth <= 1.0e-8:
            continue
        width = float(xs.max() - xs.min() + 1)
        height = float(ys.max() - ys.min() + 1)
        span = max(
            depth * width / float(intrinsics[index, 0, 0]),
            depth * height / float(intrinsics[index, 1, 1]),
        )
        if math.isfinite(span) and span > 0.0:
            spans.append(span)
    if not spans:
        raise RuntimeError("cannot estimate mask-derived object extent")
    return float(np.median(spans)), spans


def canonicalize_runtime_object_frame(
    P_W: np.ndarray,
    intrinsics: np.ndarray,
    T_W2C: np.ndarray,
    masks: Sequence[np.ndarray],
    *,
    config: RuntimeObjectFrameConfig | None = None,
    point_confidence: np.ndarray | None = None,
    point_view_visibility: np.ndarray | None = None,
    gravity_up_W: np.ndarray | None = None,
    reference_view_index: int | None = 0,
) -> RuntimeObjectFrame:
    """Estimate deployable object space O without reading GT geometry."""

    rules = config or RuntimeObjectFrameConfig()
    rules.validate()
    points = np.asarray(P_W, dtype=np.float64)
    view_count = len(masks)
    k_all, poses = _validate_camera_inputs(intrinsics, T_W2C, view_count)
    if not 0 <= int(reference_view_index) < view_count:
        raise ValueError("reference_view_index is outside the selected views")
    keep, support_stats = filter_mask_supported_points(
        points,
        k_all,
        poses,
        masks,
        config=rules,
        point_view_visibility=point_view_visibility,
    )
    object_points = points[keep]
    weights = None
    if point_confidence is not None:
        confidence = np.asarray(point_confidence, dtype=np.float64)
        if confidence.shape != (len(points),) or not np.isfinite(confidence).all():
            raise ValueError("point_confidence must be finite [N]")
        weights = np.maximum(confidence[keep], 1.0e-6)

    preliminary_center = _geometric_median(object_points, weights)
    radial = np.linalg.norm(object_points - preliminary_center, axis=1)
    cutoff = float(np.quantile(radial, float(rules.point_trim_quantile)))
    core_mask = radial <= max(cutoff, float(rules.minimum_scale))
    core_points = object_points[core_mask]
    core_weights = None if weights is None else weights[core_mask]
    point_center = _geometric_median(core_points, core_weights)
    ray_center, ray_stats = _mask_centroid_ray_center(
        masks, k_all, poses, float(rules.mask_threshold)
    )
    point_weight = float(rules.point_center_weight)
    center = point_weight * point_center + (1.0 - point_weight) * ray_center

    rotation, axis_stats = _estimate_axes(
        center,
        poses,
        gravity_up_W=gravity_up_W,
        reference_view_index=int(reference_view_index),
    )
    local_unscaled = (core_points - center) @ rotation
    quantile = float(rules.extent_quantile)
    low = np.quantile(local_unscaled, quantile, axis=0)
    high = np.quantile(local_unscaled, 1.0 - quantile, axis=0)
    point_extent = float(2.0 * np.max(np.maximum(np.abs(low), np.abs(high))))
    mask_extent, mask_spans = _mask_extent_at_center(
        center, masks, k_all, poses, float(rules.mask_threshold)
    )
    observed_extent = max(point_extent, mask_extent)
    scale = (
        observed_extent
        * float(rules.scale_padding)
        / float(rules.expected_object_extent)
    )
    if not math.isfinite(scale) or scale < float(rules.minimum_scale):
        raise RuntimeError(f"estimated object scale is invalid: {scale}")

    T_O2W = np.eye(4, dtype=np.float64)
    T_O2W[:3, :3] = rotation * scale
    T_O2W[:3, 3] = center
    T_W2O = invert_similarity(T_O2W)
    T_O2C = np.matmul(poses, T_O2W[None])
    T_C2O = np.matmul(T_W2O[None], np.linalg.inv(poses))
    P_O = apply_transform(object_points, T_W2O)
    normalized_low = np.quantile(P_O, quantile, axis=0)
    normalized_high = np.quantile(P_O, 1.0 - quantile, axis=0)
    contract = {
        "format": RUNTIME_OBJECT_FRAME_VERSION,
        "observable_inputs": [
            "mask",
            "K",
            "T_W2C",
            "P_W",
            "point_confidence",
            "optional_gravity",
        ],
        "forbidden_inputs": [
            "Scan.obj",
            "T_Scan2W",
            "target_ss",
            "target_slat",
            "gt_mesh_bounds",
        ],
        "camera_convention": "COLMAP pinhole +z forward; T_W2C",
        "axis_rule": "up=gravity_or_camera_up_with_orbit_normal; front=reference_camera_from_center",
        "center_rule": "weighted_geometric_median_points_plus_mask_centroid_ray_intersection",
        "scale_rule": "max(robust_point_extent,median_mask_extent)*padding/expected_extent",
        "config": asdict(rules),
    }
    stats = {
        "support": support_stats,
        "ray_center": ray_stats,
        "axes": axis_stats,
        "core_point_count": int(len(core_points)),
        "point_trim_radius": cutoff,
        "point_center_W": point_center.tolist(),
        "ray_center_W": ray_center.tolist(),
        "center_disagreement_over_scale": float(
            np.linalg.norm(point_center - ray_center) / scale
        ),
        "point_extent_W": point_extent,
        "mask_extent_median_W": mask_extent,
        "mask_extent_per_view_W": mask_spans,
        "selected_extent_W": observed_extent,
        "scale_O2W": scale,
        "normalized_point_bounds_low": normalized_low.tolist(),
        "normalized_point_bounds_high": normalized_high.tolist(),
    }
    return RuntimeObjectFrame(
        T_O2W=T_O2W,
        T_W2O=T_W2O,
        T_O2C=T_O2C,
        T_C2O=T_C2O,
        P_O=P_O,
        point_keep_mask=keep,
        stats=stats,
        contract=contract,
    )


def prepare_runtime_object_observation(
    images: Sequence[np.ndarray],
    masks: Sequence[np.ndarray],
    intrinsics: np.ndarray,
    T_W2C: np.ndarray,
    P_W: np.ndarray,
    *,
    camera_models: Sequence[str] | None = None,
    distortion_coefficients: Sequence[Sequence[float]] | None = None,
    frame_config: RuntimeObjectFrameConfig | None = None,
    point_confidence: np.ndarray | None = None,
    point_view_visibility: np.ndarray | None = None,
    gravity_up_W: np.ndarray | None = None,
    reference_view_index: int = 0,
    feature_resolution: int = 518,
    foreground_margin: float = 1.10,
    alpha_threshold: float = 0.80,
) -> RuntimeObjectObservation:
    """Shared training/inference frontend for real multiview object inputs."""

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
    frame = canonicalize_runtime_object_frame(
        P_W,
        undistorted_k,
        T_W2C,
        undistorted_masks,
        config=frame_config,
        point_confidence=point_confidence,
        point_view_visibility=point_view_visibility,
        gravity_up_W=gravity_up_W,
        reference_view_index=reference_view_index,
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
    geometry = prepared.geometry_record()
    prepared_rgb = np.stack(
        [np.asarray(image, dtype=np.uint8) for image in prepared.images]
    )
    condition = {
        "format": REAL_INPUT_FRONTEND_VERSION,
        "runtime_frame": frame.record(),
        "shared_image_geometry": geometry,
        "undistortion": distortion_records,
        "prepared_rgb_sha256": array_sha256(prepared_rgb),
        "prepared_mask_sha256": array_sha256(prepared.masks),
        "K_feature_sha256": array_sha256(feature_k),
        "T_O2C_sha256": array_sha256(frame.T_O2C),
        "T_O2C_lifting_sha256": array_sha256(frame.T_O2C_lifting),
        "lifting_extrinsics_policy": (
            "divide each physical T_O2C first-three-row block by "
            "cbrt(det(T_O2C[:3,:3]))"
        ),
        "P_O_sha256": array_sha256(frame.P_O),
        "condition_scope": "observable inputs only; no Scan/GT/target fields",
    }
    condition["condition_sha256"] = canonical_json_sha256(condition)
    return RuntimeObjectObservation(
        prepared_views=prepared,
        intrinsics=feature_k,
        frame=frame,
        undistortion=distortion_records,
        condition_record=condition,
    )


def project_object_points(
    P_O: np.ndarray, intrinsics: np.ndarray, T_O2C: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Project O-space sentinels through the exact lifting camera chain."""

    points = np.asarray(P_O, dtype=np.float64)
    view_count = len(T_O2C)
    k_all = np.asarray(intrinsics, dtype=np.float64)
    poses = np.asarray(T_O2C, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("P_O must be [N,3]")
    if k_all.shape != (view_count, 3, 3) or poses.shape != (view_count, 4, 4):
        raise ValueError("projection camera shapes are inconsistent")
    homogeneous = np.concatenate((points, np.ones((len(points), 1))), axis=1)
    camera = np.einsum("vij,nj->vni", poses, homogeneous)[..., :3]
    depth = camera[..., 2]
    pixels_h = np.einsum("vij,vnj->vni", k_all, camera)
    safe = np.where(np.abs(depth) > 1.0e-12, depth, 1.0)
    pixels = pixels_h[..., :2] / safe[..., None]
    return pixels, depth
