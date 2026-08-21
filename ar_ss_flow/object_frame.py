from __future__ import annotations

import math

import numpy as np
import torch


OBJECT_FRAME_VERSION = "ar_ss_flow.object_frame.v1"


def rotation_xyz(degrees_xyz: tuple[float, float, float]) -> torch.Tensor:
    ax, ay, az = (math.radians(float(value)) for value in degrees_xyz)
    cx, sx = math.cos(ax), math.sin(ax)
    cy, sy = math.cos(ay), math.sin(ay)
    cz, sz = math.cos(az), math.sin(az)
    rx = torch.tensor(((1.0, 0.0, 0.0), (0.0, cx, -sx), (0.0, sx, cx)))
    ry = torch.tensor(((cy, 0.0, sy), (0.0, 1.0, 0.0), (-sy, 0.0, cy)))
    rz = torch.tensor(((cz, -sz, 0.0), (sz, cz, 0.0), (0.0, 0.0, 1.0)))
    return (rz @ ry @ rx).float()


def make_similarity(
    *, scale: float, rotation: torch.Tensor, translation: torch.Tensor
) -> torch.Tensor:
    if float(scale) <= 0.0:
        raise ValueError("similarity scale must be positive")
    if rotation.shape != (3, 3) or translation.shape != (3,):
        raise ValueError("rotation/translation must be [3,3]/[3]")
    transform = torch.eye(4, dtype=torch.float32)
    transform[:3, :3] = rotation.float() * float(scale)
    transform[:3, 3] = translation.float()
    return transform


def similarity_scale(transform: torch.Tensor) -> float:
    linear = transform[:3, :3].float()
    return float(torch.linalg.det(linear).abs().pow(1.0 / 3.0).item())


def similarity_rotation(transform: torch.Tensor) -> torch.Tensor:
    return transform[:3, :3].float() / similarity_scale(transform)


def transform_points(points: torch.Tensor, transform: torch.Tensor) -> torch.Tensor:
    return points.float() @ transform[:3, :3].float().transpose(0, 1) + transform[
        :3, 3
    ].float()


def transform_camera_extrinsics(
    extrinsics: torch.Tensor,
    object_to_world: torch.Tensor,
    *,
    extrinsics_type: str,
) -> torch.Tensor:
    """Apply one world-frame Sim(3) while keeping camera axes orthonormal."""
    c2w = (
        extrinsics.float()
        if extrinsics_type == "c2w"
        else torch.linalg.inv(extrinsics.float())
    )
    scale = similarity_scale(object_to_world)
    rotation = similarity_rotation(object_to_world)
    translation = object_to_world[:3, 3].float()
    output = c2w.clone()
    output[:, :3, :3] = torch.einsum("ij,vjk->vik", rotation, c2w[:, :3, :3])
    output[:, :3, 3] = (
        scale * torch.einsum("ij,vj->vi", rotation, c2w[:, :3, 3])
        + translation
    )
    return output if extrinsics_type == "c2w" else torch.linalg.inv(output)


def scale_depth_calibration(calibration: dict, scale: float) -> dict:
    output = dict(calibration)
    if not bool(output.get("enabled", False)):
        return output
    for key in ("scale", "shift", "median_abs_residual", "p90_abs_residual"):
        value = output.get(key)
        if value is not None:
            output[key] = float(value) * float(scale)
    output["minimum_depth_tolerance"] = 0.02 * float(scale)
    return output


def estimate_point_pca_similarity(
    world_points: torch.Tensor,
    *,
    expected_canonical_extent: float = 0.9,
    quantile: float = 0.05,
) -> torch.Tensor:
    """Estimate a deterministic object frame from points; axis semantics remain ambiguous."""
    points = world_points.float()
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 4:
        raise ValueError("at least four world points [N,3] are required")
    low = torch.quantile(points, float(quantile), dim=0)
    high = torch.quantile(points, 1.0 - float(quantile), dim=0)
    center = 0.5 * (low + high)
    centered = points - center
    covariance = centered.transpose(0, 1) @ centered / max(len(points) - 1, 1)
    _, eigenvectors = torch.linalg.eigh(covariance)
    rotation = eigenvectors.flip(dims=(1,))
    if torch.linalg.det(rotation) < 0:
        rotation[:, -1] *= -1.0
    local = centered @ rotation
    local_low = torch.quantile(local, float(quantile), dim=0)
    local_high = torch.quantile(local, 1.0 - float(quantile), dim=0)
    extent = float((local_high - local_low).max().clamp_min(1.0e-6).item())
    scale = extent / float(expected_canonical_extent)
    return make_similarity(scale=scale, rotation=rotation, translation=center)


def rotation_error_degrees(estimate: torch.Tensor, reference: torch.Tensor) -> float:
    relative = similarity_rotation(estimate).transpose(0, 1) @ similarity_rotation(
        reference
    )
    cosine = ((torch.trace(relative) - 1.0) * 0.5).clamp(-1.0, 1.0)
    return float(torch.rad2deg(torch.acos(cosine)).item())


def deterministic_similarity(sample_index: int, seed: int) -> torch.Tensor:
    rng = np.random.default_rng(int(seed) * 1000003 + int(sample_index) * 9176)
    angles = tuple(float(value) for value in rng.uniform(-35.0, 35.0, size=3))
    scale = float(rng.uniform(0.65, 1.45))
    translation = torch.from_numpy(rng.uniform(-0.8, 0.8, size=3).astype(np.float32))
    return make_similarity(
        scale=scale,
        rotation=rotation_xyz(angles),
        translation=translation,
    )
