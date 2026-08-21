"""Training-label registration for real objects.

This module is intentionally separate from ``real_object_canonicalization``.
It may read a GT-assisted ``T_Scan2W`` but must never be called by inference or
used when hashing model conditions.
"""

from __future__ import annotations

import numpy as np

from pose_point_depth_mv.real_object_canonicalization import (
    apply_transform,
    invert_similarity,
    validate_proper_similarity,
)


REAL_OBJECT_LABEL_BINDING_VERSION = "pose_point_depth_mv.real_object_label_binding.v1"


def bind_scan_to_runtime_object(
    T_Scan2W: np.ndarray, *, T_O2W: np.ndarray
) -> np.ndarray:
    """Return label-only ``T_Scan2O = inverse(T_O2W) @ T_Scan2W``."""

    scan_to_world = validate_proper_similarity(T_Scan2W, name="T_Scan2W")
    object_to_world = validate_proper_similarity(T_O2W, name="T_O2W")
    return invert_similarity(object_to_world) @ scan_to_world


def transform_scan_vertices_to_runtime_object(
    vertices_scan: np.ndarray,
    *,
    T_Scan2W: np.ndarray,
    T_O2W: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    transform = bind_scan_to_runtime_object(T_Scan2W, T_O2W=T_O2W)
    return apply_transform(vertices_scan, transform), transform
