from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any, Literal

import numpy as np
import torch
import torch.nn.functional as F

from ar_ss_flow.pose_lifting import (
    build_projection_geometry,
    calibrate_vggt_depth,
    evaluate_vggt_depth_calibration,
)


EVIDENCE_NAMES = (
    "surface_support",
    "free_space_support",
    "occluded_support",
    "valid_view_fraction",
    "mask_view_fraction",
    "depth_confidence",
    "signed_depth_mean",
    "signed_depth_std",
    "prior_occupancy",
    "prior_confidence",
    "prior_distance",
    "positive_label",
    "negative_label",
    "neutral_label",
    "x",
    "y",
    "z",
)

POSE_MODES = (
    "correct",
    "pose_cyclic1",
    "pose_cyclic2",
    "pose_reverse",
)
DEPTH_MODES = (
    "correct",
    "depth_view_cyclic1",
    "depth_view_cyclic2",
    "depth_spatial",
)
POINT_MODES = (
    "correct",
    "point_reflect",
    "point_axis_cycle",
    "point_cross_object",
)


@dataclass
class EvidenceResult:
    features: torch.Tensor
    local_gate: torch.Tensor
    stats: dict[str, Any]
    calibration: dict[str, Any]


def deterministic_point_split(
    prior_coords: torch.Tensor,
    *,
    uid: str,
    fit_fraction: float,
    split_seed: int,
    minimum_points_per_split: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split sparse points reproducibly without depending on manifest order."""

    count = int(prior_coords.shape[0])
    minimum = int(minimum_points_per_split)
    if prior_coords.ndim != 2 or prior_coords.shape[1] not in (3, 4):
        raise ValueError("prior_coords must be [N,3/4]")
    if not 0.0 < float(fit_fraction) < 1.0:
        raise ValueError("fit_fraction must lie in (0,1)")
    if minimum <= 0 or count < 2 * minimum:
        raise ValueError(
            f"need at least {2 * minimum} points for cross-fit calibration, got {count}"
        )
    fit_count = int(round(count * float(fit_fraction)))
    fit_count = min(max(fit_count, minimum), count - minimum)
    digest = hashlib.sha256(f"{uid}:{int(split_seed)}".encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], byteorder="little", signed=False)
    generator = np.random.default_rng(seed)
    permutation = generator.permutation(count)
    fit = torch.as_tensor(
        permutation[:fit_count], device=prior_coords.device, dtype=torch.long
    )
    heldout = torch.as_tensor(
        permutation[fit_count:], device=prior_coords.device, dtype=torch.long
    )
    return fit, heldout


def _sample_maps(values: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    """Sample [V,C,H,W] maps at [V,N,2] normalized coordinates."""
    if values.ndim != 4 or grid.ndim != 3 or grid.shape[-1] != 2:
        raise ValueError(
            f"expected values [V,C,H,W] and grid [V,N,2], got "
            f"{tuple(values.shape)}/{tuple(grid.shape)}"
        )
    sampled = F.grid_sample(
        values,
        grid[:, None],
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    return sampled[:, :, 0, :]


def _variant_extrinsics(extrinsics: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "correct":
        return extrinsics
    views = int(extrinsics.shape[0])
    if views < 2:
        return extrinsics
    if mode == "pose_cyclic1":
        return torch.roll(extrinsics, shifts=1, dims=0)
    if mode == "pose_cyclic2":
        return torch.roll(extrinsics, shifts=2 if views > 2 else 1, dims=0)
    if mode == "pose_reverse":
        return torch.flip(extrinsics, dims=(0,))
    raise ValueError(f"unsupported pose mode={mode!r}")


def _variant_depth(
    depth: torch.Tensor,
    confidence: torch.Tensor,
    mode: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if mode == "correct":
        return depth, confidence
    views, height, width = depth.shape
    if mode == "depth_view_cyclic1":
        return torch.roll(depth, 1, 0), torch.roll(confidence, 1, 0)
    if mode == "depth_view_cyclic2":
        shift = 2 if views > 2 else 1
        return torch.roll(depth, shift, 0), torch.roll(confidence, shift, 0)
    if mode == "depth_spatial":
        shifts = (max(1, height // 7), max(1, width // 5))
        return (
            torch.flip(torch.roll(depth, shifts=shifts, dims=(-2, -1)), dims=(-1,)),
            torch.flip(
                torch.roll(confidence, shifts=shifts, dims=(-2, -1)), dims=(-1,)
            ),
        )
    raise ValueError(f"unsupported depth mode={mode!r}")


def _variant_points(
    prior_coords: torch.Tensor,
    prior_confidence: torch.Tensor,
    mode: str,
    *,
    cross_object_coords: torch.Tensor | None = None,
    cross_object_confidence: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if mode == "correct":
        return prior_coords, prior_confidence
    if mode == "point_cross_object":
        if cross_object_coords is None or cross_object_confidence is None:
            raise ValueError("point_cross_object requires another object's points")
        return cross_object_coords, cross_object_confidence
    output = prior_coords.clone()
    xyz = output[:, -3:].clone()
    if mode == "point_reflect":
        xyz[:, 0] = 63 - xyz[:, 0]
        xyz[:, 2] = 63 - xyz[:, 2]
    elif mode == "point_axis_cycle":
        xyz = xyz[:, (1, 2, 0)]
    else:
        raise ValueError(f"unsupported point mode={mode!r}")
    output[:, -3:] = xyz
    return output, prior_confidence


def _prior_volume(
    prior_coords: torch.Tensor,
    prior_confidence: torch.Tensor,
    *,
    side: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    occupancy = torch.zeros(side**3, device=device, dtype=torch.float32)
    confidence = torch.zeros_like(occupancy)
    xyz64 = prior_coords[:, -3:].to(device=device, dtype=torch.long)
    conf = prior_confidence.to(device=device, dtype=torch.float32)
    valid = ((xyz64 >= 0) & (xyz64 < 64)).all(dim=1) & torch.isfinite(conf)
    xyz64 = xyz64[valid]
    conf = conf[valid].clamp(0.0, 1.0)
    if xyz64.numel() == 0:
        distance = torch.ones(side**3, device=device, dtype=torch.float32)
        return (
            occupancy.reshape(side, side, side),
            confidence.reshape(side, side, side),
            distance.reshape(side, side, side),
        )
    xyz = torch.div(xyz64 * side, 64, rounding_mode="floor").clamp(0, side - 1)
    flat = xyz[:, 0] * side * side + xyz[:, 1] * side + xyz[:, 2]
    occupancy[flat] = 1.0
    confidence.scatter_reduce_(0, flat, conf, reduce="amax", include_self=True)
    axis = torch.arange(side, device=device, dtype=torch.float32)
    centers = torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"), -1).reshape(-1, 3)
    distance = torch.cdist(centers, xyz.float()).amin(1)
    distance = distance / max(math.sqrt(3.0) * float(side - 1), 1.0)
    return (
        occupancy.reshape(side, side, side),
        confidence.reshape(side, side, side),
        distance.clamp(0.0, 1.0).reshape(side, side, side),
    )


def _confidence_normalize(confidence: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    values = confidence[(confidence > 0) & valid]
    scale = values.median().clamp_min(1.0e-6) if values.numel() else confidence.new_tensor(1.0)
    return (confidence / scale).clamp(0.0, 1.0)


def _recalibrate(
    *,
    predicted_depth: torch.Tensor,
    depth_confidence: torch.Tensor,
    prior_coords: torch.Tensor,
    prior_confidence: torch.Tensor,
    masks: torch.Tensor,
    intrinsics: torch.Tensor,
    extrinsics: torch.Tensor,
    grid_transform: str,
    extrinsics_type: str,
    camera_forward_sign: float,
    min_matches: int,
    affine_improvement_ratio: float,
    mask_threshold: float,
    zbuffer_cell_size: int,
    object_to_world: torch.Tensor | None,
    force_scale_only: bool,
) -> dict[str, Any]:
    return calibrate_vggt_depth(
        predicted_depth=predicted_depth.detach().float().cpu().numpy(),
        depth_confidence=depth_confidence.detach().float().cpu().numpy(),
        prior_coords=prior_coords.detach().cpu().numpy(),
        prior_confidence=prior_confidence.detach().float().cpu().numpy(),
        masks=masks.detach().float().cpu().numpy(),
        intrinsics=intrinsics.detach().float().cpu().numpy(),
        extrinsics=extrinsics.detach().float().cpu().numpy(),
        grid_transform=str(grid_transform),
        extrinsics_type=str(extrinsics_type),
        camera_forward_sign=float(camera_forward_sign),
        min_matches=int(min_matches),
        affine_improvement_ratio=float(affine_improvement_ratio),
        mask_threshold=float(mask_threshold),
        zbuffer_cell_size=int(zbuffer_cell_size),
        object_to_world=(
            None
            if object_to_world is None
            else object_to_world.detach().float().cpu().numpy()
        ),
        force_scale_only=bool(force_scale_only),
    )


def prepare_frozen_crossfit_calibration(
    sample: dict[str, Any],
    *,
    device: torch.device,
    fit_fraction: float = 0.5,
    split_seed: int = 20260715,
    minimum_points_per_split: int = 4,
    min_depth_matches: int = 8,
    min_heldout_matches: int = 8,
    affine_improvement_ratio: float = 0.90,
    mask_threshold: float = 0.5,
    zbuffer_cell_size: int = 14,
    force_scale_only: bool = True,
    maximum_heldout_median_residual: float = 0.25,
    maximum_heldout_p90_residual: float = 0.60,
    quality_reference_residual: float = 0.25,
) -> dict[str, Any]:
    """Fit correct-pose depth calibration and score it on disjoint points."""

    prior_coords = sample["prior_coords"].to(device=device)
    prior_confidence = sample["prior_confidence"].to(
        device=device, dtype=torch.float32
    )
    fit_indices, heldout_indices = deterministic_point_split(
        prior_coords,
        uid=str(sample["uid"]),
        fit_fraction=float(fit_fraction),
        split_seed=int(split_seed),
        minimum_points_per_split=int(minimum_points_per_split),
    )
    fit_coords = prior_coords[fit_indices]
    fit_confidence = prior_confidence[fit_indices]
    heldout_coords = prior_coords[heldout_indices]
    heldout_confidence = prior_confidence[heldout_indices]
    depth = sample["predicted_depth"].to(device=device, dtype=torch.float32)
    depth_confidence = sample["depth_confidence"].to(
        device=device, dtype=torch.float32
    )
    masks = sample["masks"].to(device=device, dtype=torch.float32)
    intrinsics = sample["intrinsics"].to(device=device, dtype=torch.float32)
    extrinsics = sample["extrinsics"].to(device=device, dtype=torch.float32)
    object_to_world = sample.get("object_to_world")
    if object_to_world is not None:
        object_to_world = torch.as_tensor(
            object_to_world, device=device, dtype=torch.float32
        )
    calibration = _recalibrate(
        predicted_depth=depth,
        depth_confidence=depth_confidence,
        prior_coords=fit_coords,
        prior_confidence=fit_confidence,
        masks=masks,
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        grid_transform=str(sample["grid_transform"]),
        extrinsics_type=str(sample["extrinsics_type"]),
        camera_forward_sign=float(sample["camera_forward_sign"]),
        min_matches=int(min_depth_matches),
        affine_improvement_ratio=float(affine_improvement_ratio),
        mask_threshold=float(mask_threshold),
        zbuffer_cell_size=int(zbuffer_cell_size),
        object_to_world=object_to_world,
        force_scale_only=bool(force_scale_only),
    )
    fit_enabled = bool(calibration.get("enabled", False))
    heldout = evaluate_vggt_depth_calibration(
        calibration=calibration,
        predicted_depth=depth.detach().cpu().numpy(),
        depth_confidence=depth_confidence.detach().cpu().numpy(),
        prior_coords=heldout_coords.detach().cpu().numpy(),
        prior_confidence=heldout_confidence.detach().cpu().numpy(),
        masks=masks.detach().cpu().numpy(),
        intrinsics=intrinsics.detach().cpu().numpy(),
        extrinsics=extrinsics.detach().cpu().numpy(),
        grid_transform=str(sample["grid_transform"]),
        extrinsics_type=str(sample["extrinsics_type"]),
        camera_forward_sign=float(sample["camera_forward_sign"]),
        mask_threshold=float(mask_threshold),
        zbuffer_cell_size=int(zbuffer_cell_size),
        object_to_world=(
            None
            if object_to_world is None
            else object_to_world.detach().cpu().numpy()
        ),
        tolerance=float(maximum_heldout_median_residual),
    )
    median_residual = heldout.get("median_abs_residual")
    p90_residual = heldout.get("p90_abs_residual")
    heldout_finite = (
        median_residual is not None
        and p90_residual is not None
        and math.isfinite(float(median_residual))
        and math.isfinite(float(p90_residual))
    )
    quality_passed = bool(
        fit_enabled
        and int(heldout.get("match_count", 0)) >= int(min_heldout_matches)
        and heldout_finite
        and float(median_residual) <= float(maximum_heldout_median_residual)
        and float(p90_residual) <= float(maximum_heldout_p90_residual)
    )
    reference = max(float(quality_reference_residual), 1.0e-6)
    quality_weight = (
        math.exp(-0.5 * (float(median_residual) / reference) ** 2)
        if heldout_finite
        else 0.0
    )
    calibration = dict(calibration)
    calibration.update(
        {
            "fit_enabled": fit_enabled,
            "enabled": quality_passed,
            "quality_passed": quality_passed,
            "quality_weight": float(quality_weight),
            "fit_point_count": int(len(fit_indices)),
            "heldout_point_count": int(len(heldout_indices)),
            "heldout": heldout,
            "protocol": "correct_only_frozen_crossfit",
            "fit_fraction": float(fit_fraction),
            "split_seed": int(split_seed),
            "zbuffer_cell_size": int(zbuffer_cell_size),
            "mask_threshold": float(mask_threshold),
            "force_scale_only": bool(force_scale_only),
            "maximum_heldout_median_residual": float(
                maximum_heldout_median_residual
            ),
            "maximum_heldout_p90_residual": float(maximum_heldout_p90_residual),
        }
    )
    if not quality_passed:
        calibration["fallback"] = "heldout_calibration_quality_rejected"
    return {
        "calibration": calibration,
        "fit_indices": fit_indices,
        "heldout_indices": heldout_indices,
        "fit_coords": fit_coords,
        "fit_confidence": fit_confidence,
        "heldout_coords": heldout_coords,
        "heldout_confidence": heldout_confidence,
    }


def build_evidence(
    sample: dict[str, Any],
    *,
    device: torch.device,
    pose_mode: Literal[
        "correct", "pose_cyclic1", "pose_cyclic2", "pose_reverse"
    ] = "correct",
    depth_mode: Literal[
        "correct", "depth_view_cyclic1", "depth_view_cyclic2", "depth_spatial"
    ] = "correct",
    point_mode: Literal[
        "correct", "point_reflect", "point_axis_cycle", "point_cross_object"
    ] = "correct",
    cross_object_sample: dict[str, Any] | None = None,
    calibration_override: dict[str, Any] | None = None,
    input_prior_coords: torch.Tensor | None = None,
    input_prior_confidence: torch.Tensor | None = None,
    evaluation_prior_coords: torch.Tensor | None = None,
    evaluation_prior_confidence: torch.Tensor | None = None,
    volume_side: int = 16,
    minimum_depth_tolerance: float = 0.02,
    maximum_depth_tolerance: float = 0.15,
    surface_band_multiplier: float = 1.0,
    free_space_margin_multiplier: float = 1.5,
    sigmoid_temperature: float = 0.20,
    surface_threshold: float = 0.30,
    free_threshold: float = 0.30,
    minimum_surface_views: int = 2,
    minimum_free_views: int = 2,
    prior_radius_voxels: float = 1.5,
    gate_floor: float = 0.25,
    min_depth_matches: int = 8,
    affine_improvement_ratio: float = 0.90,
    calibration_mask_threshold: float = 0.5,
    calibration_zbuffer_cell_size: int = 0,
    force_scale_only: bool = False,
    recalibrate_each_hypothesis: bool = True,
) -> EvidenceResult:
    """Construct local surface/free-space/unknown evidence on the 16^3 SS grid.

    The sign convention is physically explicit.  For a projected voxel with
    camera depth ``z_voxel`` and aligned predicted first-surface depth
    ``z_surface``:

    * ``z_surface - z_voxel > 0``: the voxel is in observed free space;
    * near zero: the voxel is on the visible surface;
    * negative: the voxel lies behind the first surface and is unknown/occluded.
    """
    if pose_mode not in POSE_MODES:
        raise ValueError(f"unsupported pose_mode={pose_mode!r}")
    if depth_mode not in DEPTH_MODES:
        raise ValueError(f"unsupported depth_mode={depth_mode!r}")
    if point_mode not in POINT_MODES:
        raise ValueError(f"unsupported point_mode={point_mode!r}")
    if not 0.0 <= gate_floor <= 1.0:
        raise ValueError("gate_floor must be in [0,1]")
    if sigmoid_temperature <= 0.0:
        raise ValueError("sigmoid_temperature must be positive")
    if not 0.0 < minimum_depth_tolerance <= maximum_depth_tolerance:
        raise ValueError("depth tolerance bounds are invalid")

    depth = sample["predicted_depth"].to(device=device, dtype=torch.float32)
    confidence = sample["depth_confidence"].to(device=device, dtype=torch.float32)
    masks = sample["masks"].to(device=device, dtype=torch.float32)
    intrinsics = sample["intrinsics"].to(device=device, dtype=torch.float32)
    extrinsics = sample["extrinsics"].to(device=device, dtype=torch.float32)
    prior_coords = sample["prior_coords"].to(device=device)
    prior_confidence = sample["prior_confidence"].to(device=device, dtype=torch.float32)
    if input_prior_coords is not None:
        if input_prior_confidence is None:
            raise ValueError(
                "input_prior_confidence is required with input_prior_coords"
            )
        prior_coords = input_prior_coords.to(device=device)
        prior_confidence = input_prior_confidence.to(
            device=device, dtype=torch.float32
        )
    object_to_world = sample.get("object_to_world")
    if object_to_world is not None:
        object_to_world = torch.as_tensor(
            object_to_world, device=device, dtype=torch.float32
        )

    depth, confidence = _variant_depth(depth, confidence, depth_mode)
    extrinsics = _variant_extrinsics(extrinsics, pose_mode)
    other_coords = None
    other_confidence = None
    if point_mode == "point_cross_object":
        if cross_object_sample is None:
            raise ValueError("cross_object_sample is required")
        other_coords = cross_object_sample["prior_coords"].to(device=device)
        other_confidence = cross_object_sample["prior_confidence"].to(
            device=device, dtype=torch.float32
        )
    prior_coords, prior_confidence = _variant_points(
        prior_coords,
        prior_confidence,
        point_mode,
        cross_object_coords=other_coords,
        cross_object_confidence=other_confidence,
    )

    if calibration_override is not None:
        calibration = dict(calibration_override)
    elif recalibrate_each_hypothesis or any(
        mode != "correct" for mode in (pose_mode, depth_mode, point_mode)
    ):
        calibration = _recalibrate(
            predicted_depth=depth,
            depth_confidence=confidence,
            prior_coords=prior_coords,
            prior_confidence=prior_confidence,
            masks=masks,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            grid_transform=str(sample["grid_transform"]),
            extrinsics_type=str(sample["extrinsics_type"]),
            camera_forward_sign=float(sample["camera_forward_sign"]),
            min_matches=min_depth_matches,
            affine_improvement_ratio=affine_improvement_ratio,
            mask_threshold=calibration_mask_threshold,
            zbuffer_cell_size=calibration_zbuffer_cell_size,
            object_to_world=object_to_world,
            force_scale_only=force_scale_only,
        )
    else:
        calibration = dict(sample["depth_calibration"])

    height, width = map(int, depth.shape[-2:])
    geometry = build_projection_geometry(
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        grid_transform=str(sample["grid_transform"]),
        extrinsics_type=str(sample["extrinsics_type"]),
        camera_forward_sign=float(sample["camera_forward_sign"]),
        image_height=height,
        image_width=width,
        patch_grid_side=1,
        volume_side=int(volume_side),
        object_to_world=object_to_world,
    )
    sampled_depth = _sample_maps(depth[:, None], geometry["image_grid"].float())[:, 0]
    sampled_confidence = _sample_maps(
        confidence[:, None], geometry["image_grid"].float()
    )[:, 0]
    sampled_mask = _sample_maps(masks[:, None], geometry["image_grid"].float())[:, 0]
    finite = torch.isfinite(sampled_depth) & torch.isfinite(sampled_confidence)
    valid = geometry["valid"] & finite & (sampled_depth > 1.0e-6)
    mask_weight = sampled_mask.clamp(0.0, 1.0) * valid.float()
    confidence_weight = _confidence_normalize(sampled_confidence, valid) * valid.float()
    observation_weight = mask_weight * confidence_weight

    depth_enabled = bool(calibration.get("enabled", False))
    if depth_enabled:
        scale = float(calibration["scale"])
        shift = float(calibration["shift"])
        surface_depth = scale * sampled_depth + shift
        tolerance = min(
            float(maximum_depth_tolerance),
            max(
                float(minimum_depth_tolerance),
                float(calibration.get("p90_abs_residual") or 0.0),
            ),
        )
        depth_quality_weight = float(calibration.get("quality_weight", 1.0))
        observation_weight = observation_weight * depth_quality_weight
    else:
        # A disabled calibration cannot produce metric ray labels.  Keep prior
        # evidence but zero all depth-derived evidence.
        surface_depth = sampled_depth
        tolerance = float(minimum_depth_tolerance)
        depth_quality_weight = 0.0
        observation_weight = torch.zeros_like(observation_weight)

    signed = surface_depth - geometry["camera_depth"].float()
    normalized = signed / max(tolerance, 1.0e-6)
    surface_per_view = torch.exp(
        -0.5 * (normalized / max(float(surface_band_multiplier), 1.0e-6)).square()
    ) * observation_weight
    temperature = float(sigmoid_temperature)
    free_per_view = torch.sigmoid(
        (normalized - float(free_space_margin_multiplier)) / temperature
    ) * observation_weight
    occluded_per_view = torch.sigmoid(
        (-normalized - float(free_space_margin_multiplier)) / temperature
    ) * observation_weight

    views = max(int(depth.shape[0]), 1)
    surface = surface_per_view.sum(0) / float(views)
    free = free_per_view.sum(0) / float(views)
    occluded = occluded_per_view.sum(0) / float(views)
    valid_fraction = valid.float().sum(0) / float(views)
    mask_fraction = mask_weight.sum(0) / float(views)
    mean_confidence = (
        confidence_weight.sum(0) / valid.float().sum(0).clamp_min(1.0)
    )
    signed_weight = observation_weight.sum(0).clamp_min(1.0e-6)
    signed_mean = (normalized * observation_weight).sum(0) / signed_weight
    signed_var = (
        (normalized - signed_mean[None]).square() * observation_weight
    ).sum(0) / signed_weight
    signed_std = signed_var.clamp_min(0.0).sqrt()
    signed_mean = signed_mean.clamp(-4.0, 4.0) * 0.25
    signed_std = signed_std.clamp(0.0, 4.0) * 0.25
    observed_evidence = observation_weight.sum(0) > 1.0e-8

    spatial_shape = (int(volume_side), int(volume_side), int(volume_side))
    surface = surface.reshape(spatial_shape)
    free = free.reshape(spatial_shape)
    occluded = occluded.reshape(spatial_shape)
    valid_fraction = valid_fraction.reshape(spatial_shape)
    mask_fraction = mask_fraction.reshape(spatial_shape)
    mean_confidence = mean_confidence.reshape(spatial_shape)
    signed_mean = signed_mean.reshape(spatial_shape)
    signed_std = signed_std.reshape(spatial_shape)
    observed_evidence = observed_evidence.reshape(spatial_shape)
    surface_views = (surface_per_view >= 0.35).sum(0).reshape(spatial_shape)
    free_views = (free_per_view >= 0.35).sum(0).reshape(spatial_shape)

    occupancy, prior_conf, prior_distance = _prior_volume(
        prior_coords, prior_confidence, side=int(volume_side), device=device
    )
    radius_norm = float(prior_radius_voxels) / max(
        math.sqrt(3.0) * float(volume_side - 1), 1.0
    )
    prior_near = torch.exp(-prior_distance / max(radius_norm, 1.0e-6))
    prior_anchor = torch.maximum(occupancy, prior_near * prior_conf)

    positive = (
        (surface >= float(surface_threshold))
        & (surface_views >= int(minimum_surface_views))
    ) | (
        (prior_distance <= radius_norm)
        & (surface >= 0.15)
        & (surface_views >= 1)
    )
    negative = (
        ~positive
        & (free >= float(free_threshold))
        & (free_views >= int(minimum_free_views))
        & (prior_distance > radius_norm)
    )
    neutral = ~(positive | negative)

    support = torch.maximum(surface, prior_anchor)
    trust = torch.sigmoid((support - float(surface_threshold)) / temperature)
    conflict = torch.sigmoid((free - float(free_threshold)) / temperature)
    gate = float(gate_floor) + (1.0 - float(gate_floor)) * trust
    gate = gate * (1.0 - conflict)
    no_evidence = (~observed_evidence) & (prior_anchor <= 0.0)
    gate = torch.where(no_evidence, torch.zeros_like(gate), gate).clamp(0.0, 1.0)
    # This gate selects trusted surface residuals only. Free-space and unknown
    # voxels must preserve the stock prediction in the hand-gated audit.
    gate = gate * positive.float()

    axis = (torch.arange(volume_side, device=device, dtype=torch.float32) + 0.5)
    axis = axis / float(volume_side) * 2.0 - 1.0
    xyz = torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"), 0)
    features = torch.stack(
        (
            surface,
            free,
            occluded,
            valid_fraction,
            mask_fraction,
            mean_confidence,
            signed_mean,
            signed_std,
            occupancy,
            prior_conf,
            prior_distance,
            positive.float(),
            negative.float(),
            neutral.float(),
            xyz[0],
            xyz[1],
            xyz[2],
        ),
        dim=0,
    ).float()
    if tuple(features.shape) != (len(EVIDENCE_NAMES), volume_side, volume_side, volume_side):
        raise RuntimeError(f"unexpected evidence shape={tuple(features.shape)}")
    if not bool(torch.isfinite(features).all().item()) or not bool(
        torch.isfinite(gate).all().item()
    ):
        raise RuntimeError("non-finite pose-point-depth evidence")

    score_coords = (
        prior_coords
        if evaluation_prior_coords is None
        else evaluation_prior_coords.to(device=device)
    )
    score_confidence = (
        prior_confidence
        if evaluation_prior_confidence is None
        else evaluation_prior_confidence.to(device=device, dtype=torch.float32)
    )
    _, _, score_distance = _prior_volume(
        score_coords,
        score_confidence,
        side=int(volume_side),
        device=device,
    )
    prior_mask = score_distance <= radius_norm
    prior_gate_support = (
        gate[prior_mask].mean() if bool(prior_mask.any().item()) else gate.mean()
    )
    prior_free_conflict = (
        free[prior_mask].mean() if bool(prior_mask.any().item()) else free.mean()
    )
    mask_zero = mask_fraction <= 1.0e-8
    stats = {
        "pose_mode": pose_mode,
        "depth_mode": depth_mode,
        "point_mode": point_mode,
        "depth_calibration_enabled": depth_enabled,
        "depth_calibration_quality_passed": bool(
            calibration.get("quality_passed", depth_enabled)
        ),
        "depth_calibration_match_count": int(calibration.get("match_count", 0)),
        "depth_calibration_heldout_match_count": int(
            (calibration.get("heldout") or {}).get("match_count", 0)
        ),
        "depth_calibration_heldout_median_residual": (
            (calibration.get("heldout") or {}).get("median_abs_residual")
        ),
        "depth_calibration_heldout_p90_residual": (
            (calibration.get("heldout") or {}).get("p90_abs_residual")
        ),
        "depth_quality_weight": float(depth_quality_weight),
        "depth_tolerance": float(tolerance),
        "mean_surface_support": float(surface.mean().item()),
        "mean_free_space_support": float(free.mean().item()),
        "mean_occluded_support": float(occluded.mean().item()),
        "positive_ratio": float(positive.float().mean().item()),
        "negative_ratio": float(negative.float().mean().item()),
        "neutral_ratio": float(neutral.float().mean().item()),
        "label_overlap": int((positive & negative).sum().item()),
        "mean_local_gate": float(gate.mean().item()),
        "mask_zero_ratio": float(mask_zero.float().mean().item()),
        "mask_zero_gate_mean": float(gate[mask_zero].mean().item())
        if bool(mask_zero.any().item())
        else 0.0,
        "neutral_gate_mean": float(gate[neutral].mean().item())
        if bool(neutral.any().item())
        else 0.0,
        "negative_gate_mean": float(gate[negative].mean().item())
        if bool(negative.any().item())
        else 0.0,
        "evaluation_point_count": int(len(score_coords)),
        "prior_surface_support": float(surface[prior_mask].mean().item())
        if bool(prior_mask.any().item())
        else 0.0,
        "prior_gate_support": float(prior_gate_support.item()),
        "prior_free_space_conflict": float(prior_free_conflict.item()),
        "surface_to_free_gap": float((surface - free).mean().item()),
        "object_consistency_score": float(
            (prior_gate_support - prior_free_conflict).item()
        ),
    }
    return EvidenceResult(
        features=features,
        local_gate=gate[None],
        stats=stats,
        calibration=calibration,
    )


def mean_match_gate(
    candidate: torch.Tensor,
    reference: torch.Tensor,
    *,
    eps: float = 1.0e-8,
    iterations: int = 24,
) -> torch.Tensor:
    """Match a candidate gate's mean to the reference while staying in [0,1]."""
    if candidate.shape != reference.shape:
        raise ValueError("gate shapes differ")
    candidate = candidate.float().clamp(0.0, 1.0)
    reference_mean = float(reference.float().mean().item())
    if reference_mean <= eps:
        return torch.zeros_like(candidate)
    if reference_mean >= 1.0 - eps:
        return torch.ones_like(candidate)
    if float(candidate.mean().item()) <= eps:
        return torch.full_like(candidate, reference_mean)
    low, high = 0.0, max(2.0, reference_mean / max(float(candidate.mean()), eps) * 2.0)
    while float((candidate * high).clamp(0.0, 1.0).mean().item()) < reference_mean:
        high *= 2.0
        if high > 1.0e6:
            break
    for _ in range(int(iterations)):
        middle = 0.5 * (low + high)
        value = float((candidate * middle).clamp(0.0, 1.0).mean().item())
        if value < reference_mean:
            low = middle
        else:
            high = middle
    return (candidate * (0.5 * (low + high))).clamp(0.0, 1.0)


def match_applied_delta_rms(
    candidate_delta: torch.Tensor,
    reference_delta: torch.Tensor,
    *,
    eps: float = 1.0e-12,
    maximum_scale: float = 10.0,
) -> tuple[torch.Tensor, float]:
    """Match actual residual RMS, not merely the mean of its spatial gate."""

    if candidate_delta.shape != reference_delta.shape:
        raise ValueError("candidate/reference residual shapes differ")
    if maximum_scale <= 0.0:
        raise ValueError("maximum_scale must be positive")
    candidate_rms = float(candidate_delta.float().square().mean().sqrt().item())
    reference_rms = float(reference_delta.float().square().mean().sqrt().item())
    if reference_rms <= eps:
        return torch.zeros_like(candidate_delta), 0.0
    if candidate_rms <= eps:
        raise ValueError("cannot energy-match a zero candidate residual")
    scale = reference_rms / candidate_rms
    if not math.isfinite(scale) or scale > float(maximum_scale):
        raise ValueError(
            f"applied-delta RMS scale={scale} exceeds maximum={maximum_scale}"
        )
    return candidate_delta * scale, float(scale)


def match_gated_delta_rms(
    raw_delta: torch.Tensor,
    candidate_gate: torch.Tensor,
    reference_delta: torch.Tensor,
    *,
    eps: float = 1.0e-12,
    iterations: int = 40,
    maximum_scale: float = 1000.0,
    relative_tolerance: float = 1.0e-6,
) -> tuple[torch.Tensor, torch.Tensor, float, bool, float]:
    """RMS-match by scaling a gate while keeping every multiplier in [0, 1].

    A large scalar is not itself unsafe when the original gate is very small.
    The physically meaningful constraint is that the effective gate never
    amplifies ``raw_delta``. If the candidate support cannot reach the target
    RMS even after saturation, the returned ``attainable`` flag is false.
    """

    if raw_delta.shape != reference_delta.shape:
        raise ValueError("raw/reference residual shapes differ")
    if candidate_gate.ndim != raw_delta.ndim:
        raise ValueError("candidate gate rank must match raw residual rank")
    if candidate_gate.shape[0] != raw_delta.shape[0]:
        raise ValueError("candidate gate batch differs from raw residual")
    if candidate_gate.shape[1] not in (1, raw_delta.shape[1]):
        raise ValueError("candidate gate channel dimension is not broadcastable")
    if candidate_gate.shape[2:] != raw_delta.shape[2:]:
        raise ValueError("candidate gate spatial shape differs from raw residual")
    if maximum_scale <= 0.0 or iterations <= 0:
        raise ValueError("maximum_scale and iterations must be positive")

    raw = raw_delta.float()
    gate = candidate_gate.float().clamp(0.0, 1.0)
    target_rms = float(reference_delta.float().square().mean().sqrt().item())
    if target_rms <= eps:
        zero_gate = torch.zeros_like(gate)
        return torch.zeros_like(raw), zero_gate, 0.0, True, 0.0

    def apply(scale: float) -> tuple[torch.Tensor, torch.Tensor, float]:
        effective_gate = (gate * float(scale)).clamp(0.0, 1.0)
        delta = raw * effective_gate
        rms = float(delta.square().mean().sqrt().item())
        return delta, effective_gate, rms

    maximum_delta, maximum_gate, maximum_rms = apply(float(maximum_scale))
    if maximum_rms < target_rms * (1.0 - float(relative_tolerance)):
        relative_error = abs(maximum_rms - target_rms) / max(target_rms, eps)
        return (
            maximum_delta,
            maximum_gate,
            float(maximum_scale),
            False,
            float(relative_error),
        )

    low = 0.0
    high = 1.0
    _, _, high_rms = apply(high)
    while high_rms < target_rms and high < float(maximum_scale):
        low = high
        high = min(2.0 * high, float(maximum_scale))
        _, _, high_rms = apply(high)
    for _ in range(int(iterations)):
        middle = 0.5 * (low + high)
        _, _, middle_rms = apply(middle)
        if middle_rms < target_rms:
            low = middle
        else:
            high = middle
    scale = 0.5 * (low + high)
    matched_delta, effective_gate, matched_rms = apply(scale)
    relative_error = abs(matched_rms - target_rms) / max(target_rms, eps)
    attainable = relative_error <= float(relative_tolerance)
    return (
        matched_delta,
        effective_gate,
        float(scale),
        bool(attainable),
        float(relative_error),
    )
