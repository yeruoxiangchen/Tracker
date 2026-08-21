#!/usr/bin/env python3
"""TRELLIS decoded-Mesh to runtime sparse-grid coordinate-frame contract.

The FlexiCubes decoder constructs vertices directly from the same ordered
``(x, y, z)`` grid coordinates carried by the sparse latent.  A Mesh projected
with runtime-O cameras must therefore use ``transform_pose=False``.  TRELLIS'
optional ``transform_pose=True`` rotation is an external/source-asset
presentation transform; applying it here rotates an already aligned runtime-O
Mesh a second time.
"""

from __future__ import annotations

from typing import Any

import numpy as np


MESH_FRAME_CONTRACT = (
    "pose_point_depth_mv.trellis_decoder_native_sparse_grid_frame.v2"
)
LEGACY_MESH_FRAME_CONTRACT = (
    "pose_point_depth_mv.trellis_decoder_to_sparse_grid_frame.v1"
)

# Decoder vertices and sparse-grid coordinates share their axes exactly.
DECODER_TO_SPARSE_GRID = np.eye(4, dtype=np.float64)

# Historical v1 real-input artifacts incorrectly treated TRELLIS'
# ``transform_pose=True`` presentation rotation as a decoder-to-runtime-O
# transform.  Retain the exact matrix so those artifacts can be repaired
# losslessly without rerunning SS/SLat.
LEGACY_V1_ERRONEOUS_AXIS_TRANSFORM = np.asarray(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)
LEGACY_V1_TO_RUNTIME_O = np.linalg.inv(LEGACY_V1_ERRONEOUS_AXIS_TRANSFORM)


def mesh_frame_contract_fields(*, export_policy: str) -> dict[str, Any]:
    """Return immutable fields proving native decoder/runtime-O axis identity."""

    return {
        "mesh_frame_contract": MESH_FRAME_CONTRACT,
        "decoder_to_runtime_o_axis_transform_applied": False,
        "decoder_to_runtime_o_axis_transform": DECODER_TO_SPARSE_GRID.tolist(),
        "decoder_to_runtime_o_axis_rule": "identity:(x,y,z)->(x,y,z)",
        "decoder_mesh_export_policy": str(export_policy),
    }


def validate_runtime_o_mesh_frame_contract(payload: dict[str, Any]) -> None:
    """Fail closed unless an artifact proves native decoder/runtime-O axes."""

    if payload.get("output_frame") != "runtime-O":
        raise RuntimeError("Mesh output frame is not runtime-O")
    if payload.get("mesh_frame_contract") != MESH_FRAME_CONTRACT:
        raise RuntimeError("runtime-O Mesh frame contract differs")
    if payload.get("decoder_to_runtime_o_axis_transform_applied") is not False:
        raise RuntimeError("decoder/runtime-O identity contract differs")
    observed = np.asarray(
        payload.get("decoder_to_runtime_o_axis_transform"), dtype=np.float64
    )
    if observed.shape != (4, 4) or not np.array_equal(
        observed, DECODER_TO_SPARSE_GRID
    ):
        raise RuntimeError("decoder-to-runtime-O identity matrix differs")
    if (
        payload.get("decoder_to_runtime_o_axis_rule")
        != "identity:(x,y,z)->(x,y,z)"
    ):
        raise RuntimeError("decoder-to-runtime-O axis rule differs")
    if not str(payload.get("decoder_mesh_export_policy", "")):
        raise RuntimeError("decoder Mesh export policy is missing")


def validate_erroneous_v1_mesh_frame_contract(payload: dict[str, Any]) -> None:
    """Strictly recognize only the historical, repairable v1 artifact."""

    if payload.get("output_frame") != "runtime-O":
        raise RuntimeError("legacy Mesh output frame is not runtime-O")
    if payload.get("mesh_frame_contract") != LEGACY_MESH_FRAME_CONTRACT:
        raise RuntimeError("legacy runtime-O Mesh frame contract differs")
    if payload.get("decoder_to_runtime_o_axis_transform_applied") is not True:
        raise RuntimeError("legacy v1 axis transform flag differs")
    observed = np.asarray(
        payload.get("decoder_to_runtime_o_axis_transform"), dtype=np.float64
    )
    if observed.shape != (4, 4) or not np.array_equal(
        observed, LEGACY_V1_ERRONEOUS_AXIS_TRANSFORM
    ):
        raise RuntimeError("legacy v1 axis transform matrix differs")
    if payload.get("decoder_to_runtime_o_axis_rule") != "(x,y,z)->(x,z,-y)":
        raise RuntimeError("legacy v1 axis rule differs")


def decoded_mesh_to_sparse_grid_frame(decoded: Any) -> Any:
    """Export a TRELLIS decoded Mesh in its native sparse-grid/runtime-O frame."""

    return decoded.to_trimesh(transform_pose=False)


def repair_erroneous_v1_runtime_o_mesh(mesh: Any) -> Any:
    """Undo the erroneous v1 presentation rotation on a stored runtime Mesh."""

    corrected = mesh.copy()
    corrected.apply_transform(LEGACY_V1_TO_RUNTIME_O)
    return corrected


def reframe_legacy_decoder_mesh(mesh: Any) -> Any:
    """Compatibility alias for repairing the historical v1 real-input Mesh."""

    return repair_erroneous_v1_runtime_o_mesh(mesh)
