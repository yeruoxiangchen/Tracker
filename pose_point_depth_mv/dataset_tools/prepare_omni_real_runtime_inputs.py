#!/usr/bin/env python3
"""Build input-only runtime-O caches from Omni real-video raw observations.

This F0B derivative is shared by future training-cache and inference adapters.
It deliberately excludes Scan.obj, Scan alignment, and target latents, and it
must remain ``training_ready=false`` until the later label/compatibility gates.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import heapq
import itertools
import json
import math
from pathlib import Path
import re
import shutil
import statistics
from typing import Any, Sequence

import numpy as np
from PIL import Image

from ar_ss_flow.shared_object_preprocessing import (
    EmptyForegroundMaskError,
    canonical_json_sha256,
)
from pose_point_depth_mv.dataset_tools.prepare_omni_real_video_cache import (
    RAW_CACHE_FORMAT,
    sha256_file,
    utc_now,
    write_json,
    write_npz,
)
from pose_point_depth_mv.real_object_canonicalization import (
    InsufficientObjectPointsError,
    REAL_INPUT_FRONTEND_VERSION,
    RuntimeObjectFrame,
    RuntimeObjectFrameConfig,
    _mask_centroid_ray_center,
    array_sha256,
    prepare_runtime_object_observation,
    undistort_mask_view,
)
from pose_point_depth_mv.pose_mask_object_canonicalization import (
    POSE_MASK_AXIS_CONVENTIONS,
    POSE_MASK_INPUT_FRONTEND_VERSION,
    PoseMaskObjectFrameConfig,
    canonicalize_pose_mask_runtime_object_frame,
    prepare_pose_mask_runtime_object_observation,
)


OBJECT_FORMAT = "pose_point_depth_mv.omni_real_runtime_input_object.v3"
MANIFEST_FORMAT = "pose_point_depth_mv.omni_real_runtime_input_manifest.v3"
MARKER_FORMAT = "pose_point_depth_mv.omni_real_runtime_input_marker.v3"


class InsufficientForegroundViewsError(RuntimeError):
    def __init__(self, *, available: int, required: int) -> None:
        self.stage = "foreground-views"
        self.available = int(available)
        self.required = int(required)
        super().__init__(
            f"insufficient foreground-valid views: {self.available} < {self.required}"
        )


def _frame_ordinal(name: str) -> int | None:
    match = re.search(r"(\d+)$", Path(str(name)).stem)
    return None if match is None else int(match.group(1))


def pose_continuity_segments(
    frame_names: Sequence[str],
    T_W2C: np.ndarray,
    *,
    jump_floor_m: float = 0.75,
    jump_typical_multiplier: float = 12.0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Split a chronological capture when adjacent AR poses change world frame."""

    names = [str(value) for value in frame_names]
    poses = np.asarray(T_W2C, dtype=np.float64)
    if poses.shape != (len(names), 4, 4) or not np.isfinite(poses).all():
        raise ValueError("pose continuity expects finite T_W2C for every frame")
    if float(jump_floor_m) <= 0.0 or float(jump_typical_multiplier) <= 0.0:
        raise ValueError("pose continuity thresholds must be positive")
    if not names:
        return [], {
            "policy": "adjacent_source_frame_camera_center_discontinuity_v1",
            "transition_count": 0,
            "checked_transition_count": 0,
            "jump_count": 0,
            "jumps": [],
            "segments": [],
        }

    centers = np.linalg.inv(poses)[:, :3, 3]
    ordinals = [_frame_ordinal(name) for name in names]
    transitions = []
    checked_steps = []
    for index in range(len(names) - 1):
        adjacent = bool(
            ordinals[index] is not None
            and ordinals[index + 1] is not None
            and ordinals[index + 1] == ordinals[index] + 1
        )
        distance = float(np.linalg.norm(centers[index + 1] - centers[index]))
        transitions.append(
            {
                "from_index": int(index),
                "to_index": int(index + 1),
                "from_name": names[index],
                "to_name": names[index + 1],
                "translation_delta_m": distance,
                "temporally_adjacent_source_frames": adjacent,
            }
        )
        if adjacent:
            checked_steps.append(distance)
    lower_half = sorted(checked_steps)[: max(1, (len(checked_steps) + 1) // 2)]
    robust_typical = (
        float(statistics.median(lower_half)) if lower_half else 0.0
    )
    threshold = max(
        float(jump_floor_m), float(jump_typical_multiplier) * robust_typical
    )
    jumps = [
        row
        for row in transitions
        if row["temporally_adjacent_source_frames"]
        and row["translation_delta_m"] > threshold
    ]
    boundaries = {int(row["to_index"]) for row in jumps}
    starts = [0, *sorted(boundaries)]
    ends = [*sorted(boundaries), len(names)]
    segments = []
    for segment_index, (start, end) in enumerate(zip(starts, ends)):
        segments.append(
            {
                "segment_index": int(segment_index),
                "start_index": int(start),
                "end_index_exclusive": int(end),
                "frame_count": int(end - start),
                "source_indices": list(range(int(start), int(end))),
                "first_frame_name": names[start],
                "last_frame_name": names[end - 1],
            }
        )
    diagnostics = {
        "policy": "adjacent_source_frame_camera_center_discontinuity_v1",
        "jump_floor_m": float(jump_floor_m),
        "jump_typical_multiplier": float(jump_typical_multiplier),
        "robust_typical_step_m": robust_typical,
        "translation_jump_threshold_m": float(threshold),
        "transition_count": int(len(transitions)),
        "checked_transition_count": int(len(checked_steps)),
        "skipped_nonconsecutive_transition_count": int(
            len(transitions) - len(checked_steps)
        ),
        "jump_count": int(len(jumps)),
        "jumps": jumps,
        "segments": segments,
    }
    return segments, diagnostics


def evenly_spaced_frame_indices(frame_names: Sequence[str], count: int) -> np.ndarray:
    names = [str(value) for value in frame_names]
    requested = int(count)
    if requested <= 0:
        raise ValueError("selected_view_count must be positive")
    if len(names) < requested:
        raise ValueError(f"need {requested} registered frames, got {len(names)}")
    ordered = sorted(range(len(names)), key=lambda index: (names[index], index))
    positions = np.rint(np.linspace(0, len(ordered) - 1, requested)).astype(np.int64)
    selected = np.asarray(
        [ordered[int(position)] for position in positions], dtype=np.int64
    )
    if len(np.unique(selected)) != requested:
        raise RuntimeError("even frame selection produced duplicate indices")
    return selected


def foreground_valid_frame_indices(
    frame_names: Sequence[str],
    masks_dir: Path,
    count: int,
    *,
    alpha_threshold: float,
    intrinsics: np.ndarray | None = None,
    camera_by_name: dict[str, dict[str, Any]] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Keep the even selection, replacing only slots with an unusable mask.

    When camera calibration is supplied, validity is evaluated after the exact
    runtime undistortion step.  This prevents a tiny source mask from passing
    selection and disappearing under the frontend's linear remap.
    """

    names = [str(value) for value in frame_names]
    initial = evenly_spaced_frame_indices(names, count)
    threshold = float(alpha_threshold) * 255.0
    if not 0.0 < float(alpha_threshold) < 1.0:
        raise ValueError("alpha_threshold must be in (0,1)")
    if (intrinsics is None) != (camera_by_name is None):
        raise ValueError("intrinsics and camera_by_name must be supplied together")
    k_all = None if intrinsics is None else np.asarray(intrinsics, dtype=np.float64)
    if k_all is not None and (
        k_all.shape != (len(names), 3, 3) or not np.isfinite(k_all).all()
    ):
        raise ValueError(f"intrinsics must be finite [{len(names)},3,3]")
    validity_domain = (
        "source_mask" if k_all is None else "post_runtime_undistortion_mask"
    )

    validity: dict[int, bool] = {}

    def is_valid(index: int) -> bool:
        index = int(index)
        if index not in validity:
            path = Path(masks_dir) / names[index]
            if not path.is_file():
                raise FileNotFoundError(f"missing foreground mask: {path}")
            with Image.open(path) as handle:
                mask = np.asarray(handle.convert("L"), dtype=np.uint8)
            if k_all is not None:
                camera = camera_by_name.get(names[index])
                if camera is None:
                    raise RuntimeError(
                        f"missing camera metadata for frame={names[index]}"
                    )
                mask = undistort_mask_view(
                    mask,
                    k_all[index],
                    camera_model=str(camera["model"]),
                    distortion_coefficients=camera.get("distortion", []),
                )
            validity[index] = bool(np.any(mask > threshold))
        return validity[index]

    invalid_slots = [slot for slot, index in enumerate(initial) if not is_valid(index)]
    if not invalid_slots:
        return initial, {
            "policy": "lexical_even_with_nearest_valid_mask_fallback",
            "validity_domain": validity_domain,
            "fallback_used": False,
            "replacements": [],
        }

    ordered = sorted(range(len(names)), key=lambda index: (names[index], index))
    lexical_position = {index: position for position, index in enumerate(ordered)}
    selected = initial.copy()
    used = {int(index) for slot, index in enumerate(initial) if slot not in invalid_slots}
    replacements = []
    for slot in invalid_slots:
        original = int(initial[slot])
        candidates = sorted(
            (index for index in ordered if index not in used),
            key=lambda index: (
                abs(lexical_position[index] - lexical_position[original]),
                lexical_position[index],
                index,
            ),
        )
        replacement = next((index for index in candidates if is_valid(index)), None)
        if replacement is None:
            available = sum(1 for index in ordered if is_valid(index))
            raise InsufficientForegroundViewsError(
                available=available, required=int(count)
            )
        selected[slot] = replacement
        used.add(replacement)
        replacements.append(
            {
                "slot": int(slot),
                "empty_frame": names[original],
                "replacement_frame": names[replacement],
                "reason": validity_domain,
            }
        )
    if len(np.unique(selected)) != int(count) or not all(
        is_valid(index) for index in selected
    ):
        raise RuntimeError("valid-mask frame fallback produced an invalid selection")
    return selected, {
        "policy": "lexical_even_with_nearest_valid_mask_fallback",
        "validity_domain": validity_domain,
        "fallback_used": True,
        "replacements": replacements,
    }


def _circular_gap_stats(angles_degrees: Sequence[float]) -> tuple[float, float, float]:
    values = np.sort(np.asarray(angles_degrees, dtype=np.float64) % 360.0)
    if len(values) < 2:
        return 360.0, 0.0, 0.0
    gaps = np.diff(np.concatenate((values, values[:1] + 360.0)))
    maximum = float(np.max(gaps))
    return maximum, float(360.0 - maximum), float(np.std(gaps))


def object_azimuth_balanced_frame_indices(
    frame_names: Sequence[str],
    masks_dir: Path,
    count: int,
    *,
    alpha_threshold: float,
    intrinsics: np.ndarray,
    T_W2C: np.ndarray,
    object_points_W: np.ndarray,
    camera_by_name: dict[str, dict[str, Any]],
    gravity_up_w: np.ndarray | None,
    candidate_subset_limit: int = 1,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Select foreground-valid views by object-centered circular coverage."""

    names = [str(value) for value in frame_names]
    requested = int(count)
    candidate_limit = int(candidate_subset_limit)
    k_all = np.asarray(intrinsics, dtype=np.float64)
    poses = np.asarray(T_W2C, dtype=np.float64)
    points = np.asarray(object_points_W, dtype=np.float64)
    if k_all.shape != (len(names), 3, 3) or poses.shape != (len(names), 4, 4):
        raise ValueError("azimuth selection expects aligned K/T_W2C for every frame")
    if points.ndim != 2 or points.shape[1:] != (3,) or not len(points):
        raise ValueError("azimuth selection requires nonempty object_points_W [N,3]")
    if not 0.0 < float(alpha_threshold) < 1.0:
        raise ValueError("alpha_threshold must be in (0,1)")
    if candidate_limit < 1:
        raise ValueError("candidate_subset_limit must be positive")

    threshold = float(alpha_threshold) * 255.0
    valid = []
    foreground_area: dict[int, int] = {}
    for index, name in enumerate(names):
        path = Path(masks_dir) / name
        if not path.is_file():
            raise FileNotFoundError(f"missing foreground mask: {path}")
        with Image.open(path) as handle:
            mask = np.asarray(handle.convert("L"), dtype=np.uint8)
        camera = camera_by_name.get(name)
        if camera is None:
            raise RuntimeError(f"missing camera metadata for frame={name}")
        mask = undistort_mask_view(
            mask,
            k_all[index],
            camera_model=str(camera["model"]),
            distortion_coefficients=camera.get("distortion", []),
        )
        area = int(np.count_nonzero(mask > threshold))
        foreground_area[index] = area
        if area > 0:
            valid.append(index)
    if len(valid) < requested:
        raise InsufficientForegroundViewsError(available=len(valid), required=requested)

    center_w = np.median(points[np.isfinite(points).all(axis=1)], axis=0)
    gravity = np.asarray(
        [0.0, 1.0, 0.0] if gravity_up_w is None else gravity_up_w,
        dtype=np.float64,
    )
    gravity /= max(float(np.linalg.norm(gravity)), 1.0e-12)
    candidate_axis = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    basis_x = candidate_axis - float(np.dot(candidate_axis, gravity)) * gravity
    if float(np.linalg.norm(basis_x)) <= 1.0e-8:
        candidate_axis = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
        basis_x = candidate_axis - float(np.dot(candidate_axis, gravity)) * gravity
    basis_x /= float(np.linalg.norm(basis_x))
    basis_z = np.cross(gravity, basis_x)
    basis_z /= float(np.linalg.norm(basis_z))
    centers = np.linalg.inv(poses)[:, :3, 3]
    directions = centers - center_w[None]
    angles = (
        np.degrees(
            np.arctan2(directions @ basis_x, directions @ basis_z)
        )
        % 360.0
    )
    rolls = []
    for pose in poses:
        rotation = pose[:3, :3]
        forward = rotation.T @ np.asarray([0.0, 0.0, 1.0])
        projected = gravity - float(np.dot(gravity, forward)) * forward
        projected /= max(float(np.linalg.norm(projected)), 1.0e-12)
        image_up = rotation.T @ np.asarray([0.0, -1.0, 0.0])
        rolls.append(
            float(
                np.degrees(
                    np.arccos(np.clip(np.dot(image_up, projected), -1.0, 1.0))
                )
            )
        )

    def subset_score(indices: Sequence[int]) -> tuple[Any, ...]:
        maximum_gap, _coverage, gap_std = _circular_gap_stats(angles[list(indices)])
        lexical = tuple(sorted((names[index], int(index)) for index in indices))
        # Foreground area is a tertiary tie-break only; geometry owns selection.
        area = -sum(math.log1p(foreground_area[index]) for index in indices)
        return round(maximum_gap, 10), round(gap_std, 10), area, lexical

    if len(valid) <= 20:
        ranked_candidates = heapq.nsmallest(
            candidate_limit,
            itertools.combinations(valid, requested),
            key=subset_score,
        )
        search = "exhaustive_minimum_max_circular_gap"
    else:
        candidates = []
        for start in valid:
            chosen = [start]
            while len(chosen) < requested:
                remaining = [index for index in valid if index not in chosen]
                next_index = max(
                    remaining,
                    key=lambda index: (
                        min(
                            abs((angles[index] - angles[other] + 180.0) % 360.0 - 180.0)
                            for other in chosen
                        ),
                        foreground_area[index],
                        tuple(-ord(char) for char in names[index]),
                    ),
                )
                chosen.append(next_index)
            candidates.append(tuple(chosen))
        unique_candidates: dict[tuple[int, ...], tuple[int, ...]] = {}
        for candidate in candidates:
            key = tuple(sorted(int(index) for index in candidate))
            unique_candidates.setdefault(key, candidate)
        ranked_candidates = sorted(
            unique_candidates.values(), key=subset_score
        )[:candidate_limit]
        search = "deterministic_multistart_farthest_azimuth"
    if not ranked_candidates:
        raise RuntimeError("azimuth selection produced no candidate subsets")
    selected_tuple = ranked_candidates[0]

    def candidate_record(indices: Sequence[int]) -> dict[str, Any]:
        ordered_indices = sorted(indices, key=lambda index: (names[index], index))
        maximum_gap, coverage, gap_std = _circular_gap_stats(
            angles[ordered_indices]
        )
        return {
            "selected_source_view_indices": [int(index) for index in ordered_indices],
            "selected_frame_names": [names[index] for index in ordered_indices],
            "selected_camera_roll_degrees_by_frame": {
                names[index]: float(rolls[index]) for index in ordered_indices
            },
            "selected_azimuth_degrees_by_frame": {
                names[index]: float(angles[index]) for index in ordered_indices
            },
            "azimuth_coverage_degrees": coverage,
            "maximum_azimuth_gap_degrees": maximum_gap,
            "azimuth_gap_std_degrees": gap_std,
        }

    selection_candidates = [
        candidate_record(candidate) for candidate in ranked_candidates
    ]
    selected = np.asarray(
        sorted(selected_tuple, key=lambda index: (names[index], index)), dtype=np.int64
    )
    max_gap, coverage, gap_std = _circular_gap_stats(angles[selected])
    return selected, {
        "policy": "object_azimuth_balanced_valid_mask",
        "validity_domain": "post_runtime_undistortion_mask",
        "object_center_source": "coordinatewise_median_sparse_object_points_W",
        "object_center_W": center_w.tolist(),
        "gravity_up_W": gravity.tolist(),
        "search": search,
        "selection_candidate_limit": candidate_limit,
        "selection_candidate_count": len(selection_candidates),
        "selection_candidates": selection_candidates,
        "foreground_valid_frame_count": int(len(valid)),
        "camera_roll_role": "recorded_quality_metric_not_a_selection_filter",
        "selected_camera_roll_degrees_by_frame": {
            names[index]: float(rolls[index]) for index in selected
        },
        "selected_azimuth_degrees_by_frame": {
            names[index]: float(angles[index]) for index in selected
        },
        "azimuth_coverage_degrees": coverage,
        "maximum_azimuth_gap_degrees": max_gap,
        "azimuth_gap_std_degrees": gap_std,
        "fallback_used": False,
        "replacements": [],
    }


def _inverse_lexical_hash(value: str) -> str:
    """Return the same deterministic tie-break ordering as the official cache."""

    return "".join(chr(255 - ord(char)) for char in value)


def object_spherical_farthest_frame_indices(
    frame_names: Sequence[str],
    masks_dir: Path,
    count: int,
    *,
    alpha_threshold: float,
    intrinsics: np.ndarray,
    T_W2C: np.ndarray,
    object_points_W: np.ndarray,
    camera_by_name: dict[str, dict[str, Any]],
    gravity_up_w: np.ndarray | None,
    selection_identity: str = "real_capture",
    candidate_subset_limit: int = 1,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Select valid views with object-centred spherical farthest-point sampling.

    Candidate zero exactly follows the official ProObjaverse policy: choose a
    canonical-hash seed, then greedily maximise the minimum angular distance to
    the selected camera directions.  Pose-mask runtime-O may request additional
    deterministic seed starts; those candidates are evaluated by the existing
    final-view quality gates rather than silently weakening them.
    """

    names = [str(value) for value in frame_names]
    requested = int(count)
    candidate_limit = int(candidate_subset_limit)
    k_all = np.asarray(intrinsics, dtype=np.float64)
    poses = np.asarray(T_W2C, dtype=np.float64)
    points = np.asarray(object_points_W, dtype=np.float64)
    if k_all.shape != (len(names), 3, 3) or poses.shape != (len(names), 4, 4):
        raise ValueError("spherical selection expects aligned K/T_W2C for every frame")
    if points.ndim != 2 or points.shape[1:] != (3,):
        raise ValueError("spherical selection requires nonempty object_points_W [N,3]")
    finite_points = points[np.isfinite(points).all(axis=1)]
    if not len(finite_points):
        raise ValueError("spherical selection requires finite object_points_W")
    if requested <= 0:
        raise ValueError("selected_view_count must be positive")
    if not 0.0 < float(alpha_threshold) < 1.0:
        raise ValueError("alpha_threshold must be in (0,1)")
    if candidate_limit < 1:
        raise ValueError("candidate_subset_limit must be positive")

    threshold = float(alpha_threshold) * 255.0
    foreground_area: dict[int, int] = {}
    foreground_valid = []
    for index, name in enumerate(names):
        path = Path(masks_dir) / name
        if not path.is_file():
            raise FileNotFoundError(f"missing foreground mask: {path}")
        with Image.open(path) as handle:
            mask = np.asarray(handle.convert("L"), dtype=np.uint8)
        camera = camera_by_name.get(name)
        if camera is None:
            raise RuntimeError(f"missing camera metadata for frame={name}")
        mask = undistort_mask_view(
            mask,
            k_all[index],
            camera_model=str(camera["model"]),
            distortion_coefficients=camera.get("distortion", []),
        )
        area = int(np.count_nonzero(mask > threshold))
        foreground_area[index] = area
        if area > 0:
            foreground_valid.append(index)

    center_w = np.median(finite_points, axis=0)
    centers = np.linalg.inv(poses)[:, :3, 3]
    offsets = centers - center_w[None]
    radii = np.linalg.norm(offsets, axis=1)
    valid = [index for index in foreground_valid if radii[index] > 1.0e-8]
    if len(valid) < requested:
        raise InsufficientForegroundViewsError(available=len(valid), required=requested)
    directions = offsets / np.maximum(radii[:, None], 1.0e-12)

    gravity = np.asarray(
        [0.0, 1.0, 0.0] if gravity_up_w is None else gravity_up_w,
        dtype=np.float64,
    )
    gravity /= max(float(np.linalg.norm(gravity)), 1.0e-12)
    candidate_axis = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    basis_x = candidate_axis - float(np.dot(candidate_axis, gravity)) * gravity
    if float(np.linalg.norm(basis_x)) <= 1.0e-8:
        candidate_axis = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
        basis_x = candidate_axis - float(np.dot(candidate_axis, gravity)) * gravity
    basis_x /= float(np.linalg.norm(basis_x))
    basis_z = np.cross(gravity, basis_x)
    basis_z /= float(np.linalg.norm(basis_z))
    azimuths = np.degrees(
        np.arctan2(offsets @ basis_x, offsets @ basis_z)
    ) % 360.0
    rolls = []
    for pose in poses:
        rotation = pose[:3, :3]
        forward = rotation.T @ np.asarray([0.0, 0.0, 1.0])
        projected = gravity - float(np.dot(gravity, forward)) * forward
        projected /= max(float(np.linalg.norm(projected)), 1.0e-12)
        image_up = rotation.T @ np.asarray([0.0, -1.0, 0.0])
        rolls.append(
            float(
                np.degrees(
                    np.arccos(np.clip(np.dot(image_up, projected), -1.0, 1.0))
                )
            )
        )

    seed_hashes = {
        index: canonical_json_sha256(
            {"uid": str(selection_identity), "view": names[index]}
        )
        for index in valid
    }

    def farthest_from(start: int) -> tuple[int, ...]:
        selected = [int(start)]
        while len(selected) < requested:
            candidates = [index for index in valid if index not in selected]

            def score(index: int) -> tuple[float, str]:
                cosine = directions[selected] @ directions[index]
                minimum_angle = float(
                    np.arccos(np.clip(cosine, -1.0, 1.0)).min()
                )
                return minimum_angle, _inverse_lexical_hash(seed_hashes[index])

            selected.append(max(candidates, key=score))
        return tuple(selected)

    starts = sorted(valid, key=lambda index: seed_hashes[index])
    candidates = []
    seen_subsets: set[tuple[int, ...]] = set()
    for start in starts:
        candidate = farthest_from(start)
        subset_key = tuple(sorted(candidate))
        if subset_key in seen_subsets:
            continue
        seen_subsets.add(subset_key)
        candidates.append(candidate)
        if len(candidates) >= candidate_limit:
            break
    if not candidates:
        raise RuntimeError("spherical farthest selection produced no candidates")

    def candidate_record(indices: Sequence[int], rank: int) -> dict[str, Any]:
        selected_directions = directions[list(indices)]
        cosine = np.clip(
            selected_directions @ selected_directions.T, -1.0, 1.0
        )
        angles = np.degrees(np.arccos(cosine))
        np.fill_diagonal(angles, np.inf)
        nearest = np.min(angles, axis=1)
        maximum_gap, coverage, gap_std = _circular_gap_stats(
            azimuths[list(indices)]
        )
        return {
            "candidate_rank": int(rank),
            "official_seed_policy": (
                "canonical_sha256_minimum" if rank == 0 else "next_canonical_seed"
            ),
            "selected_source_view_indices": [int(index) for index in indices],
            "selected_frame_names": [names[index] for index in indices],
            "selected_camera_direction_W_by_frame": {
                names[index]: directions[index].tolist() for index in indices
            },
            "selected_camera_roll_degrees_by_frame": {
                names[index]: float(rolls[index]) for index in indices
            },
            "selected_azimuth_degrees_by_frame": {
                names[index]: float(azimuths[index]) for index in indices
            },
            "minimum_pairwise_angular_separation_degrees": float(np.min(nearest)),
            "mean_nearest_angular_separation_degrees": float(np.mean(nearest)),
            "azimuth_coverage_degrees": coverage,
            "maximum_azimuth_gap_degrees": maximum_gap,
            "azimuth_gap_std_degrees": gap_std,
        }

    selection_candidates = [
        candidate_record(candidate, rank) for rank, candidate in enumerate(candidates)
    ]
    selected = np.asarray(candidates[0], dtype=np.int64)
    selected_record = selection_candidates[0]
    return selected, {
        "policy": "object_spherical_farthest_valid_mask",
        "algorithm": "official_style_object_centered_spherical_farthest_point_v1",
        "validity_domain": "post_runtime_undistortion_mask",
        "object_center_source": "coordinatewise_median_sparse_object_points_W",
        "object_center_W": center_w.tolist(),
        "gravity_up_W": gravity.tolist(),
        "selection_identity": str(selection_identity),
        "search": "canonical_hash_seed_then_greedy_maximum_minimum_angular_distance",
        "selection_candidate_limit": candidate_limit,
        "selection_candidate_count": len(selection_candidates),
        "selection_candidates": selection_candidates,
        "foreground_valid_frame_count": int(len(foreground_valid)),
        "nondegenerate_direction_frame_count": int(len(valid)),
        "camera_roll_role": "recorded_quality_metric_not_a_selection_filter",
        **{
            key: value
            for key, value in selected_record.items()
            if key not in {"candidate_rank", "official_seed_policy"}
        },
        "fallback_used": False,
        "replacements": [],
    }


def _pose_mask_single_segment_frame_indices(
    frame_names: Sequence[str],
    masks_dir: Path,
    count: int,
    *,
    alpha_threshold: float,
    intrinsics: np.ndarray,
    T_W2C: np.ndarray,
    camera_by_name: dict[str, dict[str, Any]],
    gravity_up_w: np.ndarray | None,
    selection_algorithm: str = "azimuth",
    selection_identity: str = "real_capture",
) -> tuple[np.ndarray, dict[str, Any]]:
    """Select views from one continuous AR coordinate segment."""

    names = [str(value) for value in frame_names]
    k_all = np.asarray(intrinsics, dtype=np.float64)
    poses = np.asarray(T_W2C, dtype=np.float64)
    threshold = float(alpha_threshold) * 255.0
    valid_indices: list[int] = []
    valid_masks: list[np.ndarray] = []
    for index, name in enumerate(names):
        path = Path(masks_dir) / name
        if not path.is_file():
            raise FileNotFoundError(f"missing foreground mask: {path}")
        with Image.open(path) as handle:
            mask = np.asarray(handle.convert("L"), dtype=np.uint8)
        camera = camera_by_name.get(name)
        if camera is None:
            raise RuntimeError(f"missing camera metadata for frame={name}")
        mask = undistort_mask_view(
            mask,
            k_all[index],
            camera_model=str(camera["model"]),
            distortion_coefficients=camera.get("distortion", []),
        )
        if int(np.count_nonzero(mask > threshold)):
            valid_indices.append(index)
            valid_masks.append(mask)
    if len(valid_indices) < int(count):
        raise InsufficientForegroundViewsError(
            available=len(valid_indices), required=int(count)
        )
    valid = np.asarray(valid_indices, dtype=np.int64)
    center_w, ray_stats = _mask_centroid_ray_center(
        valid_masks,
        k_all[valid],
        poses[valid],
        float(alpha_threshold),
    )
    selector = (
        object_spherical_farthest_frame_indices
        if selection_algorithm == "spherical_farthest"
        else object_azimuth_balanced_frame_indices
    )
    selected, record = selector(
        names,
        masks_dir,
        int(count),
        alpha_threshold=float(alpha_threshold),
        intrinsics=k_all,
        T_W2C=poses,
        object_points_W=np.asarray(center_w, dtype=np.float64).reshape(1, 3),
        camera_by_name=camera_by_name,
        gravity_up_w=gravity_up_w,
        candidate_subset_limit=32,
        **(
            {"selection_identity": str(selection_identity)}
            if selection_algorithm == "spherical_farthest"
            else {}
        ),
    )
    record.update(
        {
            "policy": (
                "pose_mask_object_spherical_farthest_valid_mask"
                if selection_algorithm == "spherical_farthest"
                else "pose_mask_object_azimuth_balanced_valid_mask"
            ),
            "object_center_source": "least_squares_mask_centroid_ray_intersection",
            "object_center_W": np.asarray(center_w, dtype=np.float64).tolist(),
            "mask_ray_center_stats": ray_stats,
            "point_cloud_consumed": False,
            "candidate_frame_count": len(names),
        }
    )
    return selected, record


def _pose_mask_selected_quality(
    *,
    selected_indices: np.ndarray,
    frame_names: Sequence[str],
    masks_dir: Path,
    intrinsics: np.ndarray,
    T_W2C: np.ndarray,
    camera_by_name: dict[str, dict[str, Any]],
    gravity_up_w: np.ndarray | None,
    frame_config: PoseMaskObjectFrameConfig | None,
    view_selection: dict[str, Any],
) -> dict[str, Any]:
    masks = []
    k_selected = np.asarray(intrinsics, dtype=np.float64)[selected_indices]
    poses_selected = np.asarray(T_W2C, dtype=np.float64)[selected_indices]
    for index in selected_indices:
        name = str(frame_names[int(index)])
        camera = camera_by_name.get(name)
        if camera is None:
            raise RuntimeError(f"missing camera metadata for frame={name}")
        path = Path(masks_dir) / name
        if not path.is_file():
            raise FileNotFoundError(f"missing foreground mask: {path}")
        with Image.open(path) as handle:
            mask = np.asarray(handle.convert("L"), dtype=np.uint8)
        masks.append(
            undistort_mask_view(
                mask,
                np.asarray(intrinsics, dtype=np.float64)[int(index)],
                camera_model=str(camera["model"]),
                distortion_coefficients=camera.get("distortion", []),
            )
        )
    reference = max(
        range(len(masks)),
        key=lambda index: (int(np.count_nonzero(masks[index] > 127)), -index),
    )
    frame = canonicalize_pose_mask_runtime_object_frame(
        k_selected,
        poses_selected,
        masks,
        config=frame_config,
        gravity_up_W=gravity_up_w,
        reference_view_index=int(reference),
    )
    return runtime_input_quality_record(
        frame_stats=frame.stats,
        T_W2C=poses_selected,
        view_selection=view_selection,
        gravity_up_w=gravity_up_w,
        geometry_mode="pose_mask",
    )


def _pose_mask_quality_candidate_score(
    quality: dict[str, Any],
    selection: dict[str, Any],
    candidate_rank: int,
) -> tuple[Any, ...]:
    """Prefer formal candidates, then maximize margin to every final-8 gate."""

    values = quality.get("values") or {}
    thresholds = quality.get("thresholds") or {}
    checks = quality.get("checks") or {}

    def upper_ratio(value_key: str, threshold_key: str) -> float:
        value = values.get(value_key)
        threshold = thresholds.get(threshold_key)
        if value is None or threshold is None or float(threshold) <= 0.0:
            return float("inf")
        return float(value) / float(threshold)

    def lower_ratio(value_key: str, threshold_key: str) -> float:
        value = values.get(value_key)
        threshold = thresholds.get(threshold_key)
        if value is None or threshold is None or float(value) <= 0.0:
            return float("inf")
        return float(threshold) / float(value)

    risks = (
        upper_ratio(
            "ray_residual_median_over_mask_extent",
            "max_ray_residual_median_over_mask_extent",
        ),
        upper_ratio(
            "ray_residual_p90_over_mask_extent",
            "max_ray_residual_p90_over_mask_extent",
        ),
        lower_ratio("orbit_gravity_agreement", "min_orbit_gravity_agreement"),
        lower_ratio("azimuth_coverage_degrees", "min_azimuth_coverage_degrees"),
        upper_ratio("maximum_azimuth_gap_degrees", "max_azimuth_gap_degrees"),
    )
    return (
        0 if quality.get("formal_input_passed") else 1,
        sum(not bool(value) for value in checks.values()),
        max(risks),
        risks[0],
        risks[1],
        -float(selection.get("azimuth_coverage_degrees") or 0.0),
        float(selection.get("maximum_azimuth_gap_degrees") or 360.0),
        int(candidate_rank),
    )


def pose_mask_object_azimuth_balanced_frame_indices(
    frame_names: Sequence[str],
    masks_dir: Path,
    count: int,
    *,
    alpha_threshold: float,
    intrinsics: np.ndarray,
    T_W2C: np.ndarray,
    camera_by_name: dict[str, dict[str, Any]],
    gravity_up_w: np.ndarray | None,
    frame_config: PoseMaskObjectFrameConfig | None = None,
    _selection_algorithm: str = "azimuth",
    selection_identity: str = "real_capture",
) -> tuple[np.ndarray, dict[str, Any]]:
    """Select the best final views from one internally consistent AR segment."""

    names = [str(value) for value in frame_names]
    requested = int(count)
    k_all = np.asarray(intrinsics, dtype=np.float64)
    poses = np.asarray(T_W2C, dtype=np.float64)
    if k_all.shape != (len(names), 3, 3) or poses.shape != (len(names), 4, 4):
        raise ValueError("segmented pose-mask selection expects aligned K/T_W2C")
    segments, continuity = pose_continuity_segments(names, poses)
    trials = []
    candidates = []
    for segment in segments:
        source_indices = np.asarray(segment["source_indices"], dtype=np.int64)
        trial = {
            "segment_index": int(segment["segment_index"]),
            "start_index": int(segment["start_index"]),
            "end_index_exclusive": int(segment["end_index_exclusive"]),
            "frame_count": int(segment["frame_count"]),
            "first_frame_name": segment["first_frame_name"],
            "last_frame_name": segment["last_frame_name"],
        }
        if len(source_indices) < requested:
            trial.update(
                {
                    "status": "insufficient_segment_frames",
                    "required_view_count": requested,
                    "formal_input_passed": False,
                }
            )
            trials.append(trial)
            continue
        segment_names = [names[int(index)] for index in source_indices]
        try:
            _initial_selected_local, selection = _pose_mask_single_segment_frame_indices(
                segment_names,
                masks_dir,
                requested,
                alpha_threshold=float(alpha_threshold),
                intrinsics=k_all[source_indices],
                T_W2C=poses[source_indices],
                camera_by_name=camera_by_name,
                gravity_up_w=gravity_up_w,
                selection_algorithm=_selection_algorithm,
                selection_identity=(
                    f"{selection_identity}:segment{int(segment['segment_index'])}"
                ),
            )
            selection_candidates = list(selection.pop("selection_candidates", []))
            if not selection_candidates:
                raise RuntimeError("pose-mask selector returned no final-8 candidates")
            candidate_choices = []
            candidate_trials = []
            for candidate_rank, candidate_record in enumerate(selection_candidates):
                selected_local = np.asarray(
                    candidate_record["selected_source_view_indices"], dtype=np.int64
                )
                if (
                    selected_local.shape != (requested,)
                    or len(np.unique(selected_local)) != requested
                    or np.any(selected_local < 0)
                    or np.any(selected_local >= len(source_indices))
                ):
                    raise RuntimeError("pose-mask final-8 candidate indices are invalid")
                selected_global = source_indices[selected_local]
                candidate_selection = {
                    **selection,
                    **candidate_record,
                    "selected_candidate_rank": int(candidate_rank),
                }
                quality = _pose_mask_selected_quality(
                    selected_indices=selected_global,
                    frame_names=names,
                    masks_dir=masks_dir,
                    intrinsics=k_all,
                    T_W2C=poses,
                    camera_by_name=camera_by_name,
                    gravity_up_w=gravity_up_w,
                    frame_config=frame_config,
                    view_selection=candidate_selection,
                )
                candidate_score = _pose_mask_quality_candidate_score(
                    quality, candidate_selection, candidate_rank
                )
                candidate_trials.append(
                    {
                        "candidate_rank": int(candidate_rank),
                        "selected_source_view_indices": selected_global.tolist(),
                        "selected_frame_names": [
                            names[int(index)] for index in selected_global
                        ],
                        "azimuth_coverage_degrees": candidate_selection[
                            "azimuth_coverage_degrees"
                        ],
                        "maximum_azimuth_gap_degrees": candidate_selection[
                            "maximum_azimuth_gap_degrees"
                        ],
                        "minimum_pairwise_angular_separation_degrees": (
                            candidate_selection.get(
                                "minimum_pairwise_angular_separation_degrees"
                            )
                        ),
                        "mean_nearest_angular_separation_degrees": (
                            candidate_selection.get(
                                "mean_nearest_angular_separation_degrees"
                            )
                        ),
                        "official_seed_policy": candidate_selection.get(
                            "official_seed_policy"
                        ),
                        "input_quality": quality,
                        "formal_input_passed": bool(
                            quality.get("formal_input_passed")
                        ),
                    }
                )
                candidate_choices.append(
                    (
                        candidate_score,
                        selected_global,
                        candidate_selection,
                        quality,
                    )
                )
            candidate_score, selected_global, selection, quality = min(
                candidate_choices, key=lambda row: row[0]
            )
            trial.update(
                {
                    "status": "evaluated_final_views",
                    "selected_source_view_indices": selected_global.tolist(),
                    "selected_frame_names": [
                        names[int(index)] for index in selected_global
                    ],
                    "view_selection": selection,
                    "final_selected_view_quality": quality,
                    "final8_candidate_search": {
                        "policy": (
                            "evaluate up to 32 deterministic spherical-farthest "
                            "subsets; prefer formal pass then largest normalized "
                            "gate margin"
                            if _selection_algorithm == "spherical_farthest"
                            else "evaluate up to 32 azimuth-balanced subsets; "
                            "prefer formal pass then largest normalized gate margin"
                        ),
                        "candidate_count": int(len(candidate_trials)),
                        "selected_candidate_rank": int(
                            selection["selected_candidate_rank"]
                        ),
                        "trials": candidate_trials,
                    },
                    "formal_input_passed": bool(
                        quality.get("formal_input_passed")
                    ),
                }
            )
            score = (
                *candidate_score[:-1],
                -int(segment["frame_count"]),
                int(segment["segment_index"]),
                candidate_score[-1],
            )
            candidates.append((score, selected_global, selection, quality, trial))
        except Exception as exc:
            trial.update(
                {
                    "status": "segment_selection_failed",
                    "formal_input_passed": False,
                    "error": repr(exc),
                }
            )
        trials.append(trial)
    if not candidates:
        raise RuntimeError(
            "no pose-continuous segment can produce the requested views: "
            + json.dumps(trials, ensure_ascii=False)
        )
    _score, selected, selection, quality, chosen_trial = min(
        candidates, key=lambda row: row[0]
    )
    record = {
        **selection,
        "policy": (
            "pose_mask_segmented_object_spherical_farthest_valid_mask"
            if _selection_algorithm == "spherical_farthest"
            else "pose_mask_segmented_object_azimuth_balanced_valid_mask"
        ),
        "candidate_frame_count": int(len(names)),
        "selected_segment_candidate_frame_count": int(
            chosen_trial["frame_count"]
        ),
        "selected_segment_index": int(chosen_trial["segment_index"]),
        "trajectory_continuity": continuity,
        "segment_trials": trials,
        "selection_decision": (
            "prefer formal final-view pass, then maximize gate margin; each "
            "candidate is generated by spherical maximum-minimum angular spacing"
            if _selection_algorithm == "spherical_farthest"
            else "prefer formal final-view pass, then maximize azimuth coverage "
            "and minimize gap/ray residual"
        ),
        "selected_segment_final_view_quality": quality,
    }
    return np.asarray(selected, dtype=np.int64), record


def pose_mask_object_spherical_farthest_frame_indices(
    frame_names: Sequence[str],
    masks_dir: Path,
    count: int,
    *,
    alpha_threshold: float,
    intrinsics: np.ndarray,
    T_W2C: np.ndarray,
    camera_by_name: dict[str, dict[str, Any]],
    gravity_up_w: np.ndarray | None,
    frame_config: PoseMaskObjectFrameConfig | None = None,
    selection_identity: str = "real_capture",
) -> tuple[np.ndarray, dict[str, Any]]:
    """Pose-mask wrapper for the official-style spherical selector."""

    return pose_mask_object_azimuth_balanced_frame_indices(
        frame_names,
        masks_dir,
        count,
        alpha_threshold=alpha_threshold,
        intrinsics=intrinsics,
        T_W2C=T_W2C,
        camera_by_name=camera_by_name,
        gravity_up_w=gravity_up_w,
        frame_config=frame_config,
        _selection_algorithm="spherical_farthest",
        selection_identity=selection_identity,
    )


def pose_mask_training_spherical_farthest_frame_indices(
    frame_names: Sequence[str],
    masks_dir: Path,
    count: int,
    *,
    alpha_threshold: float,
    intrinsics: np.ndarray,
    T_W2C: np.ndarray,
    camera_by_name: dict[str, dict[str, Any]],
    gravity_up_w: np.ndarray | None,
    selection_identity: str = "real_capture",
    object_center_w: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Run the exact single-seed 3-D view FPS used by official training caches.

    Unlike :func:`pose_mask_object_spherical_farthest_frame_indices`, this
    policy does not enumerate alternative seeds and does not rank subsets by
    real-input quality gates.  It is therefore the controlled training-policy
    arm, rather than a quality-constrained selector that happens to use FPS.
    """

    names = [str(value) for value in frame_names]
    k_all = np.asarray(intrinsics, dtype=np.float64)
    poses = np.asarray(T_W2C, dtype=np.float64)
    center = None if object_center_w is None else np.asarray(object_center_w, dtype=np.float64)
    ray_stats = None
    if center is None:
        threshold = float(alpha_threshold) * 255.0
        valid_indices: list[int] = []
        valid_masks: list[np.ndarray] = []
        for index, name in enumerate(names):
            path = Path(masks_dir) / name
            if not path.is_file():
                raise FileNotFoundError(f"missing foreground mask: {path}")
            with Image.open(path) as handle:
                mask = np.asarray(handle.convert("L"), dtype=np.uint8)
            camera = camera_by_name.get(name)
            if camera is None:
                raise RuntimeError(f"missing camera metadata for frame={name}")
            mask = undistort_mask_view(
                mask,
                k_all[index],
                camera_model=str(camera["model"]),
                distortion_coefficients=camera.get("distortion", []),
            )
            if int(np.count_nonzero(mask > threshold)):
                valid_indices.append(index)
                valid_masks.append(mask)
        if len(valid_indices) < int(count):
            raise InsufficientForegroundViewsError(
                available=len(valid_indices), required=int(count)
            )
        valid = np.asarray(valid_indices, dtype=np.int64)
        center, ray_stats = _mask_centroid_ray_center(
            valid_masks,
            k_all[valid],
            poses[valid],
            float(alpha_threshold),
        )
        center_source = "least_squares_mask_centroid_ray_intersection"
    else:
        if center.shape != (3,) or not np.isfinite(center).all():
            raise ValueError("object_center_w must be a finite [3] vector")
        center_source = "fixed_all_foreground_valid_runtime_frame_center"

    selected, record = object_spherical_farthest_frame_indices(
        names,
        masks_dir,
        int(count),
        alpha_threshold=float(alpha_threshold),
        intrinsics=k_all,
        T_W2C=poses,
        object_points_W=center.reshape(1, 3),
        camera_by_name=camera_by_name,
        gravity_up_w=gravity_up_w,
        selection_identity=str(selection_identity),
        candidate_subset_limit=1,
    )
    record.update(
        {
            "policy": "pose_mask_training_exact_spherical_farthest_valid_mask",
            "algorithm": "official_training_single_seed_spherical_fps_v1",
            "object_center_source": center_source,
            "object_center_W": center.tolist(),
            "mask_ray_center_stats": ray_stats,
            "point_cloud_consumed": False,
            "candidate_frame_count": len(names),
            "quality_gate_used_for_selection": False,
        }
    )
    return selected, record


def runtime_input_quality_record(
    *,
    frame_stats: dict[str, Any],
    T_W2C: np.ndarray,
    view_selection: dict[str, Any],
    gravity_up_w: np.ndarray | None,
    geometry_mode: str = "point_mask",
) -> dict[str, Any]:
    """Record the strict real-input gates without blocking diagnostic replay."""

    if geometry_mode not in {"point_mask", "pose_mask"}:
        raise ValueError(f"unsupported geometry_mode={geometry_mode!r}")

    poses = np.asarray(T_W2C, dtype=np.float64)
    gravity = np.asarray(
        [0.0, 1.0, 0.0] if gravity_up_w is None else gravity_up_w,
        dtype=np.float64,
    )
    gravity /= max(float(np.linalg.norm(gravity)), 1.0e-12)
    rolls = []
    for pose in poses:
        rotation = pose[:3, :3]
        forward = rotation.T @ np.asarray([0.0, 0.0, 1.0])
        projected = gravity - float(np.dot(gravity, forward)) * forward
        projected /= max(float(np.linalg.norm(projected)), 1.0e-12)
        image_up = rotation.T @ np.asarray([0.0, -1.0, 0.0])
        rolls.append(float(np.degrees(np.arccos(np.clip(np.dot(image_up, projected), -1.0, 1.0)))))
    ray = frame_stats["ray_center"]
    mask_extent = float(frame_stats["mask_extent_median_W"])
    point_extent = (
        float(frame_stats["point_extent_W"])
        if geometry_mode == "point_mask"
        else None
    )
    orbit_gravity_agreement = float(
        frame_stats.get("axes", {}).get("orbit_camera_up_agreement", 0.0)
    )
    values = {
        "ray_residual_median_over_mask_extent": float(ray["ray_residual_median"]) / max(mask_extent, 1.0e-12),
        "ray_residual_p90_over_mask_extent": float(ray["ray_residual_p90"]) / max(mask_extent, 1.0e-12),
        "camera_roll_median_degrees": float(np.median(rolls)),
        "camera_roll_max_degrees": float(np.max(rolls)),
        "point_to_mask_extent_ratio": (
            None
            if point_extent is None
            else point_extent / max(mask_extent, 1.0e-12)
        ),
        "orbit_gravity_agreement": orbit_gravity_agreement,
        "azimuth_coverage_degrees": view_selection.get("azimuth_coverage_degrees"),
        "maximum_azimuth_gap_degrees": view_selection.get("maximum_azimuth_gap_degrees"),
    }
    if geometry_mode == "pose_mask":
        thresholds = {
            "max_ray_residual_median_over_mask_extent": 0.20,
            "max_ray_residual_p90_over_mask_extent": 0.40,
            "min_orbit_gravity_agreement": 0.80,
            "min_azimuth_coverage_degrees": 240.0,
            "max_azimuth_gap_degrees": 120.0,
        }
        checks = {
            "ray_residual_median": (
                values["ray_residual_median_over_mask_extent"]
                <= thresholds["max_ray_residual_median_over_mask_extent"]
            ),
            "ray_residual_p90": (
                values["ray_residual_p90_over_mask_extent"]
                <= thresholds["max_ray_residual_p90_over_mask_extent"]
            ),
            "orbit_gravity_agreement": (
                values["orbit_gravity_agreement"]
                >= thresholds["min_orbit_gravity_agreement"]
            ),
            "azimuth_coverage": (
                values["azimuth_coverage_degrees"] is not None
                and values["azimuth_coverage_degrees"]
                >= thresholds["min_azimuth_coverage_degrees"]
            ),
            "maximum_azimuth_gap": (
                values["maximum_azimuth_gap_degrees"] is not None
                and values["maximum_azimuth_gap_degrees"]
                <= thresholds["max_azimuth_gap_degrees"]
            ),
        }
        return {
            "profile": "pose_mask_final8_capture_aligned_v1",
            "geometry_mode": geometry_mode,
            "values": values,
            "thresholds": thresholds,
            "checks": checks,
            "diagnostic_only": {
                "camera_roll_median_degrees": values[
                    "camera_roll_median_degrees"
                ],
                "camera_roll_max_degrees": values["camera_roll_max_degrees"],
                "reason": (
                    "K and T_W2C preserve a stable physical camera roll; roll is "
                    "recorded but is not a pose-mask rejection criterion"
                ),
            },
            "formal_input_passed": bool(all(checks.values())),
            "diagnostic_replay_allowed": True,
        }
    thresholds = {
        "max_ray_residual_median_over_mask_extent": 0.102,
        "max_camera_roll_median_degrees": 12.5,
        "max_point_to_mask_extent_ratio": 1.423,
        "min_azimuth_coverage_degrees": 240.0,
        "max_azimuth_gap_degrees": 120.0,
    }
    checks = {
        "ray_residual_median": values["ray_residual_median_over_mask_extent"] <= thresholds["max_ray_residual_median_over_mask_extent"],
        "camera_roll_median": values["camera_roll_median_degrees"] <= thresholds["max_camera_roll_median_degrees"],
        "point_to_mask_extent": values["point_to_mask_extent_ratio"] <= thresholds["max_point_to_mask_extent_ratio"],
        "azimuth_coverage": values["azimuth_coverage_degrees"] is not None and values["azimuth_coverage_degrees"] >= thresholds["min_azimuth_coverage_degrees"],
        "maximum_azimuth_gap": values["maximum_azimuth_gap_degrees"] is not None and values["maximum_azimuth_gap_degrees"] <= thresholds["max_azimuth_gap_degrees"],
    }
    return {
        "profile": "m11c_holdout64_p95_anchored_ar_input_v1",
        "geometry_mode": geometry_mode,
        "values": values,
        "thresholds": thresholds,
        "checks": checks,
        "formal_input_passed": bool(all(checks.values())),
        "diagnostic_replay_allowed": True,
    }


def _object_key(row: dict[str, Any]) -> str:
    return f"{row['category']}:{row['object_id']}"


def select_rows(
    rows: list[dict[str, Any]], selectors: Sequence[str] | None
) -> list[dict[str, Any]]:
    if not selectors:
        return rows
    by_key = {_object_key(row): row for row in rows}
    by_object: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_object.setdefault(str(row["object_id"]), []).append(row)
    selected = []
    for value in selectors:
        if value in by_key:
            row = by_key[value]
        else:
            matches = by_object.get(value, [])
            if len(matches) != 1:
                raise ValueError(
                    f"object selector {value!r} matched {len(matches)} rows; "
                    "use category:object_id"
                )
            row = matches[0]
        if row not in selected:
            selected.append(row)
    return selected


def _load_rgb_mask(
    images_dir: Path, masks_dir: Path, frame_names: Sequence[str]
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    images = []
    masks = []
    for name in frame_names:
        image_path = images_dir / str(name)
        mask_path = masks_dir / str(name)
        if not image_path.is_file() or not mask_path.is_file():
            raise FileNotFoundError(
                f"missing aligned RGB/mask pair: {image_path} / {mask_path}"
            )
        with Image.open(image_path) as handle:
            images.append(np.asarray(handle.convert("RGB"), dtype=np.uint8))
        with Image.open(mask_path) as handle:
            masks.append(np.asarray(handle.convert("L"), dtype=np.uint8))
    return images, masks


def _camera_rows_by_name(row: dict[str, Any]) -> dict[str, dict[str, Any]]:
    output = {}
    for camera in row.get("cameras", []):
        name = str(camera["frame_name"])
        if name in output:
            raise RuntimeError(f"duplicate camera metadata for frame={name}")
        output[name] = dict(camera)
    return output


def _pose_mask_all_valid_runtime_frame(
    *,
    frame_names: Sequence[str],
    masks_dir: Path,
    intrinsics: np.ndarray,
    T_W2C: np.ndarray,
    camera_by_name: dict[str, dict[str, Any]],
    alpha_threshold: float,
    frame_config: PoseMaskObjectFrameConfig,
    gravity_up_w: np.ndarray | None,
    axis_convention: str,
) -> tuple[RuntimeObjectFrame, dict[str, Any]]:
    """Construct one pose-mask O from the full foreground-valid candidate pool."""

    names = [str(value) for value in frame_names]
    k_all = np.asarray(intrinsics, dtype=np.float64)
    poses = np.asarray(T_W2C, dtype=np.float64)
    if k_all.shape != (len(names), 3, 3) or poses.shape != (len(names), 4, 4):
        raise ValueError("all-valid object-frame construction expects aligned K/T_W2C")
    threshold = float(alpha_threshold) * 255.0
    valid_indices: list[int] = []
    valid_masks: list[np.ndarray] = []
    foreground_areas: list[int] = []
    for index, name in enumerate(names):
        camera = camera_by_name.get(name)
        if camera is None:
            raise RuntimeError(f"missing camera metadata for frame={name}")
        path = Path(masks_dir) / name
        if not path.is_file():
            raise FileNotFoundError(f"missing foreground mask: {path}")
        with Image.open(path) as handle:
            mask = np.asarray(handle.convert("L"), dtype=np.uint8)
        mask = undistort_mask_view(
            mask,
            k_all[index],
            camera_model=str(camera["model"]),
            distortion_coefficients=camera.get("distortion", []),
        )
        area = int(np.count_nonzero(mask > threshold))
        if area > 0:
            valid_indices.append(index)
            valid_masks.append(mask)
            foreground_areas.append(area)
    if len(valid_indices) < 2:
        raise InsufficientForegroundViewsError(available=len(valid_indices), required=2)
    valid = np.asarray(valid_indices, dtype=np.int64)
    reference_local = max(
        range(len(valid)), key=lambda index: (foreground_areas[index], -valid_indices[index])
    )
    frame = canonicalize_pose_mask_runtime_object_frame(
        k_all[valid],
        poses[valid],
        valid_masks,
        config=frame_config,
        gravity_up_W=gravity_up_w,
        reference_view_index=int(reference_local),
        axis_convention=str(axis_convention),
    )
    reference_source = int(valid[reference_local])
    construction = {
        "scope": "all_foreground_valid_candidates",
        "candidate_frame_count": int(len(names)),
        "foreground_valid_frame_count": int(len(valid)),
        "foreground_valid_source_view_indices": valid.tolist(),
        "foreground_valid_frame_names": [names[int(index)] for index in valid],
        "reference_valid_view_index": int(reference_local),
        "reference_source_view_index": reference_source,
        "reference_frame_name": names[reference_source],
        "reference_rule": "largest_undistorted_foreground_area_then_earliest_source_index",
        "axis_convention": str(axis_convention),
    }
    frame.stats["object_frame_construction"] = construction
    frame.contract["object_frame_view_scope"] = "all_foreground_valid_candidates"
    return frame, construction


def _load_reusable(
    destination: Path, *, source_cache_sha256: str, build_config_sha256: str
) -> dict[str, Any] | None:
    marker_path = destination / "_RUNTIME_INPUT_COMPLETE.json"
    report_path = destination / "report.json"
    if not marker_path.is_file() or not report_path.is_file():
        return None
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if (
        marker.get("format") != MARKER_FORMAT
        or marker.get("source_cache_sha256") != source_cache_sha256
        or marker.get("build_config_sha256") != build_config_sha256
    ):
        raise RuntimeError(f"stale runtime-input output: {destination}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("format") != OBJECT_FORMAT or report.get("passed") is not True:
        raise RuntimeError(f"invalid reusable runtime-input report: {report_path}")
    required = [report.get("cache_npz"), report.get("condition_record")]
    required.extend(report.get("prepared_rgb_paths", []))
    required.extend(report.get("prepared_mask_paths", []))
    if any(not isinstance(path, str) or not Path(path).is_file() for path in required):
        raise RuntimeError(f"runtime-input marker has missing artifacts: {destination}")
    return report


def build_object_runtime_input(
    row: dict[str, Any],
    *,
    output_dir: Path,
    selected_view_count: int,
    feature_resolution: int,
    foreground_margin: float,
    alpha_threshold: float,
    view_selection_policy: str = "lexical_even_valid_mask_fallback",
    geometry_mode: str = "point_mask",
    object_frame_view_scope: str = "selected",
    model_o_axis_convention: str = "legacy_y_up",
    resume_partial: bool = False,
    frame_config: RuntimeObjectFrameConfig | PoseMaskObjectFrameConfig,
    build_config_sha256: str,
    gravity_up_w: np.ndarray | None = None,
) -> tuple[dict[str, Any], bool]:
    category = str(row["category"])
    object_id = str(row["object_id"])
    source_cache = Path(row["cache_npz"]).resolve()
    source_hash = sha256_file(source_cache)
    destination = output_dir / "objects" / category / object_id
    reusable = _load_reusable(
        destination,
        source_cache_sha256=source_hash,
        build_config_sha256=build_config_sha256,
    )
    if reusable is not None:
        return reusable, True
    if destination.exists():
        raise RuntimeError(f"partial runtime-input output exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / f".{object_id}.runtime-input-building"
    if staging.exists():
        if not resume_partial:
            raise RuntimeError(f"partial runtime-input staging exists: {staging}")
        shutil.rmtree(staging)
    staging.mkdir()

    camera_by_name = _camera_rows_by_name(row)
    fixed_pose_mask_frame: RuntimeObjectFrame | None = None
    object_frame_construction: dict[str, Any] | None = None
    effective_gravity_up_w = gravity_up_w
    try:
        with np.load(source_cache, allow_pickle=False) as source:
            frame_names_all = [str(value) for value in source["frame_name"].tolist()]
            K_all = np.asarray(source["K"], dtype=np.float64)
            T_W2C_all = np.asarray(source["T_W2C"], dtype=np.float64)
            P_W = np.asarray(source["P_W"], dtype=np.float64)
            if geometry_mode == "pose_mask" and object_frame_view_scope == "all_foreground_valid":
                if not isinstance(frame_config, PoseMaskObjectFrameConfig):
                    raise TypeError("pose-mask all-view O requires PoseMaskObjectFrameConfig")
                fixed_pose_mask_frame, object_frame_construction = (
                    _pose_mask_all_valid_runtime_frame(
                        frame_names=frame_names_all,
                        masks_dir=Path(row["masks_dir"]),
                        intrinsics=K_all,
                        T_W2C=T_W2C_all,
                        camera_by_name=camera_by_name,
                        alpha_threshold=float(alpha_threshold),
                        frame_config=frame_config,
                        gravity_up_w=gravity_up_w,
                        axis_convention=str(model_o_axis_convention),
                    )
                )
                if effective_gravity_up_w is None:
                    linear = np.asarray(fixed_pose_mask_frame.T_O2W[:3, :3], dtype=np.float64)
                    scale = float(np.linalg.norm(linear[:, 0]))
                    up_axis = 2 if model_o_axis_convention == "official_z_up" else 1
                    effective_gravity_up_w = linear[:, up_axis] / max(scale, 1.0e-12)
            if view_selection_policy == "lexical_even":
                selected_indices = evenly_spaced_frame_indices(
                    frame_names_all, int(selected_view_count)
                )
                selection_record = {
                    "policy": "lexical_frame_order_evenly_spaced_including_endpoints",
                    "fallback_used": False,
                    "replacements": [],
                }
            elif view_selection_policy == "lexical_even_valid_mask_fallback":
                selected_indices, selection_record = foreground_valid_frame_indices(
                    frame_names_all,
                    Path(row["masks_dir"]),
                    int(selected_view_count),
                    alpha_threshold=float(alpha_threshold),
                    intrinsics=K_all,
                    camera_by_name=camera_by_name,
                )
            elif view_selection_policy == "training_spherical_farthest_valid_mask":
                if geometry_mode == "pose_mask":
                    selected_indices, selection_record = (
                        pose_mask_training_spherical_farthest_frame_indices(
                            frame_names_all,
                            Path(row["masks_dir"]),
                            int(selected_view_count),
                            alpha_threshold=float(alpha_threshold),
                            intrinsics=K_all,
                            T_W2C=T_W2C_all,
                            camera_by_name=camera_by_name,
                            gravity_up_w=effective_gravity_up_w,
                            selection_identity=_object_key(row),
                            object_center_w=(
                                None
                                if fixed_pose_mask_frame is None
                                else np.asarray(
                                    fixed_pose_mask_frame.T_O2W[:3, 3], dtype=np.float64
                                )
                            ),
                        )
                    )
                else:
                    selected_indices, selection_record = (
                        object_spherical_farthest_frame_indices(
                            frame_names_all,
                            Path(row["masks_dir"]),
                            int(selected_view_count),
                            alpha_threshold=float(alpha_threshold),
                            intrinsics=K_all,
                            T_W2C=T_W2C_all,
                            object_points_W=P_W,
                            camera_by_name=camera_by_name,
                            gravity_up_w=effective_gravity_up_w,
                            selection_identity=_object_key(row),
                            candidate_subset_limit=1,
                        )
                    )
                    selection_record["policy"] = (
                        "training_exact_object_spherical_farthest_valid_mask"
                    )
                    selection_record["quality_gate_used_for_selection"] = False
            elif view_selection_policy in {
                "object_azimuth_balanced_valid_mask",
                "object_spherical_farthest_valid_mask",
            }:
                spherical = (
                    view_selection_policy == "object_spherical_farthest_valid_mask"
                )
                if geometry_mode == "pose_mask":
                    selector = (
                        pose_mask_object_spherical_farthest_frame_indices
                        if spherical
                        else pose_mask_object_azimuth_balanced_frame_indices
                    )
                else:
                    selector = (
                        object_spherical_farthest_frame_indices
                        if spherical
                        else object_azimuth_balanced_frame_indices
                    )
                selector_kwargs: dict[str, Any] = {
                    "alpha_threshold": float(alpha_threshold),
                    "intrinsics": K_all,
                    "T_W2C": T_W2C_all,
                    "camera_by_name": camera_by_name,
                    "gravity_up_w": effective_gravity_up_w,
                }
                if geometry_mode == "point_mask":
                    selector_kwargs["object_points_W"] = P_W
                else:
                    selector_kwargs["frame_config"] = frame_config
                if spherical:
                    selector_kwargs["selection_identity"] = _object_key(row)
                selected_indices, selection_record = selector(
                        frame_names_all,
                        Path(row["masks_dir"]),
                        int(selected_view_count),
                        **selector_kwargs,
                )
            else:
                raise ValueError(
                    f"unsupported view_selection_policy={view_selection_policy!r}"
                )
            frame_names = [frame_names_all[int(index)] for index in selected_indices]
            K = np.asarray(K_all[selected_indices], dtype=np.float64)
            T_W2C = np.asarray(T_W2C_all[selected_indices], dtype=np.float64)
            confidence = (
                np.asarray(source["point_confidence_proxy"], dtype=np.float64)
                if geometry_mode == "point_mask"
                and "point_confidence_proxy" in source.files
                else None
            )
    except InsufficientForegroundViewsError:
        shutil.rmtree(staging)
        raise

    missing_camera_rows = [name for name in frame_names if name not in camera_by_name]
    if missing_camera_rows:
        raise RuntimeError(f"missing camera metadata for frames={missing_camera_rows}")
    cameras = [camera_by_name[name] for name in frame_names]
    images, masks = _load_rgb_mask(
        Path(row["images_dir"]), Path(row["masks_dir"]), frame_names
    )
    try:
        common_observation_kwargs = {
            "camera_models": [str(camera["model"]) for camera in cameras],
            "distortion_coefficients": [camera.get("distortion", []) for camera in cameras],
            "frame_config": frame_config,
            "gravity_up_W": effective_gravity_up_w,
            "reference_view_index": None,
            "feature_resolution": int(feature_resolution),
            "foreground_margin": float(foreground_margin),
            "alpha_threshold": float(alpha_threshold),
        }
        if geometry_mode == "pose_mask":
            observation = prepare_pose_mask_runtime_object_observation(
                images,
                masks,
                K,
                T_W2C,
                axis_convention=str(model_o_axis_convention),
                runtime_frame_override=fixed_pose_mask_frame,
                **common_observation_kwargs,
            )
        else:
            observation = prepare_runtime_object_observation(
                images,
                masks,
                K,
                T_W2C,
                P_W,
                point_confidence=confidence,
                **common_observation_kwargs,
            )
    except (InsufficientObjectPointsError, EmptyForegroundMaskError):
        shutil.rmtree(staging)
        raise

    view_dir = staging / "views"
    view_dir.mkdir()
    rgb_names = []
    mask_names = []
    for view_index, (image, mask) in enumerate(
        zip(observation.prepared_views.images, observation.prepared_views.masks)
    ):
        rgb_name = f"view_{view_index:02d}_rgb.png"
        mask_name = f"view_{view_index:02d}_mask.png"
        image.save(view_dir / rgb_name)
        Image.fromarray(np.clip(np.rint(mask * 255.0), 0, 255).astype(np.uint8)).save(
            view_dir / mask_name
        )
        rgb_names.append(rgb_name)
        mask_names.append(mask_name)

    cache_name = "runtime_input_cache.npz"
    T_O2C = np.asarray(observation.frame.T_O2C, dtype=np.float64)
    T_O2C_lifting = np.asarray(observation.frame.T_O2C_lifting, dtype=np.float64)
    write_npz(
        staging / cache_name,
        selected_source_view_index=selected_indices,
        frame_name=np.asarray(frame_names),
        K_feature=np.asarray(observation.intrinsics, dtype=np.float32),
        T_O2C=T_O2C,
        T_O2C_lifting=T_O2C_lifting,
        T_C2O=np.asarray(observation.frame.T_C2O, dtype=np.float64),
        T_O2W=np.asarray(observation.frame.T_O2W, dtype=np.float64),
        T_W2O=np.asarray(observation.frame.T_W2O, dtype=np.float64),
        P_O=np.asarray(observation.frame.P_O, dtype=np.float32),
        object_point_source_index=np.flatnonzero(
            observation.frame.point_keep_mask
        ).astype(np.int64),
        source_to_feature_affine=np.asarray(
            observation.prepared_views.source_to_feature_affines, dtype=np.float32
        ),
    )
    condition_name = "condition_record.json"
    write_json(staging / condition_name, observation.condition_record)
    input_quality = (
        runtime_input_quality_record(
            frame_stats=observation.frame.stats,
            T_W2C=T_W2C,
            view_selection=selection_record,
            gravity_up_w=effective_gravity_up_w,
            geometry_mode=geometry_mode,
        )
        if selection_record.get("policy")
        in {
            "object_azimuth_balanced_valid_mask",
            "pose_mask_object_azimuth_balanced_valid_mask",
            "pose_mask_segmented_object_azimuth_balanced_valid_mask",
            "object_spherical_farthest_valid_mask",
            "pose_mask_object_spherical_farthest_valid_mask",
            "pose_mask_segmented_object_spherical_farthest_valid_mask",
            "pose_mask_training_exact_spherical_farthest_valid_mask",
            "training_exact_object_spherical_farthest_valid_mask",
        }
        else None
    )

    final_view_dir = destination / "views"
    report = {
        "format": OBJECT_FORMAT,
        "created_at_utc": utc_now(),
        "category": category,
        "object_id": object_id,
        "object_key": _object_key(row),
        "source_raw_cache": str(source_cache),
        "source_raw_cache_sha256": source_hash,
        "build_config_sha256": build_config_sha256,
        "input_frontend_format": (
            POSE_MASK_INPUT_FRONTEND_VERSION
            if geometry_mode == "pose_mask"
            else REAL_INPUT_FRONTEND_VERSION
        ),
        "geometry_mode": str(geometry_mode),
        "point_cloud_consumed": geometry_mode == "point_mask",
        "object_frame_view_scope": str(object_frame_view_scope),
        "model_o_axis_convention": str(model_o_axis_convention),
        "object_frame_construction": object_frame_construction,
        "selected_view_count": int(selected_view_count),
        "selected_source_view_indices": selected_indices.tolist(),
        "selected_frame_names": frame_names,
        "view_selection": selection_record,
        "reference_view_index": int(
            observation.frame.stats["axes"]["reference_view_index"]
        ),
        "cache_npz": str((destination / cache_name).resolve()),
        "condition_record": str((destination / condition_name).resolve()),
        "condition_sha256": observation.condition_sha256,
        "T_O2C_sha256": array_sha256(T_O2C),
        "T_O2C_lifting_sha256": array_sha256(T_O2C_lifting),
        "T_O2W_sha256": array_sha256(
            np.asarray(observation.frame.T_O2W, dtype=np.float64)
        ),
        "lifting_extrinsics_policy": (
            "physical T_O2C retained for audit/depth; projectively normalized "
            "T_O2C_lifting is the only matrix exported to Native v2 lifting"
        ),
        "prepared_rgb_paths": [
            str((final_view_dir / name).resolve()) for name in rgb_names
        ],
        "prepared_mask_paths": [
            str((final_view_dir / name).resolve()) for name in mask_names
        ],
        "runtime_frame_stats": observation.frame.stats,
        "input_quality": input_quality,
        "formal_input_passed": (
            None if input_quality is None else input_quality["formal_input_passed"]
        ),
        "forbidden_gt_fields_absent": True,
        "training_ready": False,
        "scope_guard": (
            "Input-only runtime-O derivative. Pose-mask mode derives O solely from "
            "calibrated camera poses, intrinsics, and masks; point-mask mode also "
            "consumes P_W. No Scan alignment, GT Mesh, or target latent is consumed."
        ),
        "passed": True,
    }
    write_json(staging / "report.json", report)
    write_json(
        staging / "_RUNTIME_INPUT_COMPLETE.json",
        {
            "format": MARKER_FORMAT,
            "completed_at_utc": utc_now(),
            "object_key": report["object_key"],
            "source_cache_sha256": source_hash,
            "build_config_sha256": build_config_sha256,
            "condition_sha256": observation.condition_sha256,
            "T_O2C_lifting_sha256": array_sha256(T_O2C_lifting),
        },
    )
    staging.replace(destination)
    return report, False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--raw_cache_report", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--object", action="append")
    parser.add_argument("--allow_failures", action="store_true")
    parser.add_argument("--record_quality_rejections", action="store_true")
    parser.add_argument("--min_completed_objects", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--selected_view_count", type=int, default=8)
    parser.add_argument(
        "--geometry_mode",
        choices=("point_mask", "pose_mask"),
        default="point_mask",
    )
    parser.add_argument(
        "--view_selection_policy",
        choices=(
            "lexical_even",
            "lexical_even_valid_mask_fallback",
            "training_spherical_farthest_valid_mask",
            "object_azimuth_balanced_valid_mask",
            "object_spherical_farthest_valid_mask",
        ),
        default="lexical_even_valid_mask_fallback",
    )
    parser.add_argument(
        "--object_frame_view_scope",
        choices=("selected", "all_foreground_valid"),
        default="selected",
        help=(
            "views used to estimate pose-mask O; all_foreground_valid freezes one "
            "O before the final view-selection policy is applied"
        ),
    )
    parser.add_argument(
        "--model_o_axis_convention",
        choices=POSE_MASK_AXIS_CONVENTIONS,
        default="legacy_y_up",
    )
    parser.add_argument("--feature_resolution", type=int, default=518)
    parser.add_argument("--foreground_margin", type=float, default=1.10)
    parser.add_argument("--alpha_threshold", type=float, default=0.80)
    parser.add_argument("--mask_threshold", type=float, default=0.5)
    parser.add_argument("--min_mask_observations", type=int, default=2)
    parser.add_argument("--min_mask_support_ratio", type=float, default=0.60)
    parser.add_argument("--min_object_points", type=int, default=100)
    parser.add_argument("--point_trim_quantile", type=float, default=0.98)
    parser.add_argument("--extent_quantile", type=float, default=0.02)
    parser.add_argument("--point_center_weight", type=float, default=0.75)
    parser.add_argument("--expected_object_extent", type=float, default=0.90)
    parser.add_argument("--scale_padding", type=float, default=1.05)
    parser.add_argument(
        "--gravity_up_w",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help="optional observable gravity-up vector in sparse world coordinates",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.geometry_mode != "pose_mask" and (
        args.object_frame_view_scope != "selected"
        or args.model_o_axis_convention != "legacy_y_up"
    ):
        raise ValueError(
            "object-frame scope/axis overrides currently apply only to pose_mask"
        )
    if args.geometry_mode == "pose_mask":
        frame_config = PoseMaskObjectFrameConfig(
            mask_threshold=args.mask_threshold,
            expected_object_extent=args.expected_object_extent,
            scale_padding=args.scale_padding,
        )
    else:
        frame_config = RuntimeObjectFrameConfig(
            mask_threshold=args.mask_threshold,
            min_mask_observations=args.min_mask_observations,
            min_mask_support_ratio=args.min_mask_support_ratio,
            min_object_points=args.min_object_points,
            point_trim_quantile=args.point_trim_quantile,
            extent_quantile=args.extent_quantile,
            point_center_weight=args.point_center_weight,
            expected_object_extent=args.expected_object_extent,
            scale_padding=args.scale_padding,
        )
    frame_config.validate()
    view_selection_contract = {
        "lexical_even": "lexical_frame_order_evenly_spaced_including_endpoints",
        "lexical_even_valid_mask_fallback": (
            "lexical_even_with_nearest_valid_mask_fallback"
        ),
        "training_spherical_farthest_valid_mask": (
            "official_training_single_seed_object_centered_spherical_fps_v1"
        ),
        "object_azimuth_balanced_valid_mask": (
            "object_centered_minimum_max_circular_azimuth_gap_with_valid_masks"
        ),
        "object_spherical_farthest_valid_mask": (
            "official_style_object_centered_spherical_farthest_point_with_valid_masks_v1"
        ),
    }[args.view_selection_policy]
    build_config = {
        "input_frontend_format": (
            POSE_MASK_INPUT_FRONTEND_VERSION
            if args.geometry_mode == "pose_mask"
            else REAL_INPUT_FRONTEND_VERSION
        ),
        "geometry_mode": str(args.geometry_mode),
        "point_cloud_consumed": args.geometry_mode == "point_mask",
        "selected_view_count": int(args.selected_view_count),
        "view_selection": view_selection_contract,
        "object_frame_view_scope": str(args.object_frame_view_scope),
        "model_o_axis_convention": str(args.model_o_axis_convention),
        "reference_view": (
            "largest_undistorted_foreground_mask_then_earliest_all_valid_view"
            if args.object_frame_view_scope == "all_foreground_valid"
            else "largest_undistorted_foreground_mask_then_earliest_selected_view"
        ),
        "feature_resolution": int(args.feature_resolution),
        "foreground_margin": float(args.foreground_margin),
        "alpha_threshold": float(args.alpha_threshold),
        "frame_config": asdict(frame_config),
        "gravity_up_W": (
            None
            if args.gravity_up_w is None
            else [float(value) for value in args.gravity_up_w]
        ),
    }
    build_hash = canonical_json_sha256(build_config)
    raw_report_path = Path(args.raw_cache_report).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    raw = json.loads(raw_report_path.read_text(encoding="utf-8"))
    if raw.get("format") != RAW_CACHE_FORMAT or raw.get("passed") is not True:
        raise RuntimeError(f"raw cache report is not eligible: {raw_report_path}")
    rows = select_rows(list(raw["objects"]), args.object)
    if not rows:
        raise RuntimeError("runtime-input build selected no objects")
    output_dir.mkdir(parents=True, exist_ok=True)

    reports = []
    failures = []
    quality_rejections = []
    reused = []
    for index, row in enumerate(rows, start=1):
        key = _object_key(row)
        print(f"[real_runtime_input] {index}/{len(rows)} object={key}", flush=True)
        try:
            report, was_reused = build_object_runtime_input(
                row,
                output_dir=output_dir,
                selected_view_count=int(args.selected_view_count),
                feature_resolution=int(args.feature_resolution),
                foreground_margin=float(args.foreground_margin),
                alpha_threshold=float(args.alpha_threshold),
                view_selection_policy=str(args.view_selection_policy),
                geometry_mode=str(args.geometry_mode),
                object_frame_view_scope=str(args.object_frame_view_scope),
                model_o_axis_convention=str(args.model_o_axis_convention),
                resume_partial=bool(args.resume),
                frame_config=frame_config,
                build_config_sha256=build_hash,
                gravity_up_w=(
                    None
                    if args.gravity_up_w is None
                    else np.asarray(args.gravity_up_w, dtype=np.float64)
                ),
            )
            reports.append(report)
            if was_reused:
                reused.append(key)
            print(
                f"[real_runtime_input] object={key} "
                f"geometry={report['geometry_mode']} "
                f"points={report['runtime_frame_stats'].get('support', {}).get('mask_supported_point_count', 0)} "
                f"condition={report['condition_sha256'][:12]} reused={was_reused}",
                flush=True,
            )
        except (
            InsufficientObjectPointsError,
            InsufficientForegroundViewsError,
            EmptyForegroundMaskError,
        ) as error:
            if not args.record_quality_rejections:
                failures.append({"object_key": key, "error": repr(error)})
                print(f"[real_runtime_input] FAILED object={key}: {error!r}", flush=True)
                raise
            if isinstance(error, EmptyForegroundMaskError):
                rejection = {
                    "reason": "empty_foreground_after_undistortion",
                    "stage": "shared-preprocessing",
                    "source_name": error.source_name,
                }
                detail = f"source={error.source_name}"
            else:
                rejection = {
                    "reason": (
                        "insufficient_object_points"
                        if isinstance(error, InsufficientObjectPointsError)
                        else "insufficient_foreground_views"
                    ),
                    "stage": error.stage,
                    "available": error.available,
                    "required": error.required,
                }
                detail = f"stage={error.stage} available={error.available}<{error.required}"
            quality_rejections.append(
                {
                    "object_key": key,
                    "category": str(row["category"]),
                    "object_id": str(row["object_id"]),
                    **rejection,
                }
            )
            print(
                f"[real_runtime_input] REJECTED object={key} {detail}",
                flush=True,
            )
        except Exception as error:
            failures.append({"object_key": key, "error": repr(error)})
            print(f"[real_runtime_input] FAILED object={key}: {error!r}", flush=True)
            if not args.allow_failures:
                raise

    effective_raw_report_path = raw_report_path
    effective_raw_report_sha256 = sha256_file(raw_report_path)
    eligible_raw_path = None
    if args.record_quality_rejections:
        completed_keys = {str(report["object_key"]) for report in reports}
        eligible_rows = [row for row in rows if _object_key(row) in completed_keys]
        if len(eligible_rows) != len(reports):
            raise RuntimeError("eligible raw/runtime object identity mismatch")
        eligible_raw_path = output_dir / "eligible_raw_cache_report.json"
        eligibility_policy = {
            "runtime_frame_min_object_points": int(frame_config.min_object_points),
            "quality_rejection_types": [
                "InsufficientObjectPointsError",
                "InsufficientForegroundViewsError",
                "EmptyForegroundMaskError",
            ],
            "minimum_completed_object_count": int(args.min_completed_objects),
        }
        eligible_raw = {
            **raw,
            "created_at_utc": utc_now(),
            "source_raw_cache_report": str(raw_report_path),
            "source_raw_cache_report_sha256": sha256_file(raw_report_path),
            "source_object_count": len(rows),
            "category_count": len({str(row["category"]) for row in eligible_rows}),
            "object_count": len(eligible_rows),
            "objects": eligible_rows,
            "runtime_input_eligibility_policy": eligibility_policy,
            "quality_rejections": quality_rejections,
            "alignment_passed": False,
            "training_ready": False,
            "scope_guard": (
                "Raw-cache subset whose object identities exactly match successful "
                "input-only runtime-O construction. No GT alignment or target is included."
            ),
            "passed": len(eligible_rows) >= int(args.min_completed_objects),
        }
        if eligible_raw_path.is_file():
            existing = json.loads(eligible_raw_path.read_text(encoding="utf-8"))
            comparable_existing = dict(existing)
            comparable_expected = dict(eligible_raw)
            comparable_existing.pop("created_at_utc", None)
            comparable_expected.pop("created_at_utc", None)
            if comparable_existing != comparable_expected:
                raise RuntimeError(
                    f"existing eligible raw-cache report differs: {eligible_raw_path}"
                )
            eligible_raw = existing
        else:
            write_json(eligible_raw_path, eligible_raw)
        effective_raw_report_path = eligible_raw_path.resolve()
        effective_raw_report_sha256 = sha256_file(eligible_raw_path)

    manifest = {
        "format": MANIFEST_FORMAT,
        "created_at_utc": utc_now(),
        "raw_cache_report": str(effective_raw_report_path),
        "raw_cache_report_sha256": effective_raw_report_sha256,
        "source_raw_cache_report": str(raw_report_path),
        "source_raw_cache_report_sha256": sha256_file(raw_report_path),
        "build_config": build_config,
        "build_config_sha256": build_hash,
        "source_selected_object_count": len(rows),
        "selected_object_count": len(reports),
        "completed_object_count": len(reports),
        "reused_objects": reused,
        "objects": reports,
        "quality_rejections": quality_rejections,
        "eligible_raw_cache_report": (
            None if eligible_raw_path is None else str(eligible_raw_path.resolve())
        ),
        "failures": failures,
        "training_ready": False,
        "scope_guard": (
            "F0B input-only runtime-O caches. F0C-F0E compatibility and label gates "
            "remain required before any real v2 Full training manifest."
        ),
    }
    manifest["passed"] = (
        len(reports) >= int(args.min_completed_objects) and not failures
    )
    manifest_path = output_dir / "runtime_input_manifest.json"
    write_json(manifest_path, manifest)
    print(
        json.dumps(
            {
                "passed": manifest["passed"],
                "source_selected_object_count": len(rows),
                "completed_object_count": len(reports),
                "quality_rejection_count": len(quality_rejections),
                "failure_count": len(failures),
                "eligible_raw_cache_report": manifest["eligible_raw_cache_report"],
                "training_ready": False,
                "manifest": str(manifest_path),
            },
            indent=2,
        )
    )
    if not manifest["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
