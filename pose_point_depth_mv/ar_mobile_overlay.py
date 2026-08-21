#!/usr/bin/env python3
"""Build compact, dependency-free Unity AR overlay meshes."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import struct
from typing import Any

import numpy as np
import trimesh


MOBILE_OVERLAY_FORMAT = "yxc_unity_ar_mesh.v1"
MOBILE_OVERLAY_MAGIC = b"YXCARM01"
MOBILE_OVERLAY_VERSION = 1
MOBILE_OVERLAY_MAX_TRIANGLES = 60_000
_HEADER = struct.Struct("<8sIIII")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_world_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    loaded = trimesh.load(path, force="scene", process=False)
    scene = loaded if isinstance(loaded, trimesh.Scene) else trimesh.Scene(loaded)
    geometries = []
    for node_name in scene.graph.nodes_geometry:
        transform, geometry_name = scene.graph[node_name]
        geometry = scene.geometry[geometry_name]
        if not isinstance(geometry, trimesh.Trimesh) or not len(geometry.faces):
            continue
        copy = geometry.copy()
        copy.apply_transform(np.asarray(transform, dtype=np.float64))
        geometries.append(copy)
    if not geometries:
        raise RuntimeError(f"world mesh contains no triangle geometry: {path}")
    mesh = trimesh.util.concatenate(geometries)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    valid = (
        np.isfinite(vertices).all(axis=1)
        if len(vertices)
        else np.empty((0,), dtype=bool)
    )
    if not valid.all():
        remap = np.full(len(vertices), -1, dtype=np.int64)
        remap[valid] = np.arange(int(valid.sum()), dtype=np.int64)
        faces = remap[faces]
        faces = faces[(faces >= 0).all(axis=1)]
        vertices = vertices[valid]
    faces = faces[
        (faces[:, 0] != faces[:, 1])
        & (faces[:, 1] != faces[:, 2])
        & (faces[:, 0] != faces[:, 2])
    ]
    if not len(vertices) or not len(faces):
        raise RuntimeError(f"world mesh is empty after validation: {path}")
    return vertices, faces


def _compact_vertices(
    vertices: np.ndarray, faces: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    used, inverse = np.unique(faces.reshape(-1), return_inverse=True)
    return vertices[used], inverse.reshape(-1, 3)


def _cluster_at_resolution(
    vertices: np.ndarray, faces: np.ndarray, divisions: int
) -> tuple[np.ndarray, np.ndarray]:
    minimum = vertices.min(axis=0)
    extent = vertices.max(axis=0) - minimum
    longest = max(float(extent.max()), 1.0e-12)
    voxel = longest / max(int(divisions), 1)
    keys = np.floor((vertices - minimum) / voxel).astype(np.int64)
    _unique, inverse = np.unique(keys, axis=0, return_inverse=True)
    count = np.bincount(inverse).astype(np.float64)
    clustered = np.stack(
        [np.bincount(inverse, weights=vertices[:, axis]) for axis in range(3)],
        axis=1,
    ) / count[:, None]
    mapped = inverse[faces]
    mapped = mapped[
        (mapped[:, 0] != mapped[:, 1])
        & (mapped[:, 1] != mapped[:, 2])
        & (mapped[:, 0] != mapped[:, 2])
    ]
    if not len(mapped):
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 3), dtype=np.int64)
    canonical = np.sort(mapped, axis=1)
    _rows, unique_indices = np.unique(canonical, axis=0, return_index=True)
    mapped = mapped[np.sort(unique_indices)]
    return _compact_vertices(clustered, mapped)


def _vertex_cluster_to_budget(
    vertices: np.ndarray, faces: np.ndarray, max_triangles: int
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if len(faces) <= max_triangles:
        compact_vertices, compact_faces = _compact_vertices(vertices, faces)
        return compact_vertices, compact_faces, {
            "method": "identity_compaction",
            "grid_divisions": None,
        }

    low = 2
    high = 1024
    best: tuple[np.ndarray, np.ndarray, int] | None = None
    while low <= high:
        divisions = (low + high) // 2
        candidate_vertices, candidate_faces = _cluster_at_resolution(
            vertices, faces, divisions
        )
        if 0 < len(candidate_faces) <= max_triangles:
            best = candidate_vertices, candidate_faces, divisions
            low = divisions + 1
        elif len(candidate_faces) > max_triangles:
            high = divisions - 1
        else:
            low = divisions + 1
    if best is None:
        for divisions in range(2, 33):
            candidate_vertices, candidate_faces = _cluster_at_resolution(
                vertices, faces, divisions
            )
            if 0 < len(candidate_faces) <= max_triangles:
                best = candidate_vertices, candidate_faces, divisions
        if best is None:
            raise RuntimeError("vertex clustering could not produce a nonempty mobile mesh")
    result_vertices, result_faces, divisions = best
    return result_vertices, result_faces, {
        "method": "isotropic_vertex_clustering",
        "grid_divisions": int(divisions),
    }


def _vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    normals = np.zeros_like(vertices, dtype=np.float64)
    triangles = vertices[faces]
    face_normals = np.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    )
    for corner in range(3):
        np.add.at(normals, faces[:, corner], face_normals)
    lengths = np.linalg.norm(normals, axis=1)
    valid = lengths > 1.0e-12
    normals[valid] /= lengths[valid, None]
    normals[~valid] = np.asarray([0.0, 1.0, 0.0])
    return normals


def build_mobile_overlay_mesh(
    world_obj: Path,
    output_path: Path,
    *,
    max_triangles: int = MOBILE_OVERLAY_MAX_TRIANGLES,
    source_coordinate_frame: str = "internal_world",
    output_coordinate_frame: str = "unity_world",
) -> dict[str, Any]:
    """Reflect an internal right-handed Mesh into a compact Unity binary Mesh.

    The numeric conversion is the same for an AR capture-anchor frame as for a
    conventional Unity world frame.  Explicit frame labels prevent a caller
    from silently interpreting A0-relative vertices as session-world vertices.
    """

    world_obj = world_obj.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    if max_triangles < 1:
        raise ValueError("max_triangles must be positive")
    if not str(source_coordinate_frame).strip():
        raise ValueError("source_coordinate_frame must be nonempty")
    if not str(output_coordinate_frame).strip():
        raise ValueError("output_coordinate_frame must be nonempty")
    source_vertices, source_faces = _load_world_mesh(world_obj)
    vertices, faces, reduction = _vertex_cluster_to_budget(
        source_vertices, source_faces, int(max_triangles)
    )

    # The capture sparse model stores W_colmap = diag(1,1,-1) * W_unity.
    # Reflect vertices back and reverse winding so front faces/normals stay valid.
    unity_vertices = np.asarray(vertices, dtype=np.float64).copy()
    unity_vertices[:, 2] *= -1.0
    unity_faces = np.asarray(faces[:, [0, 2, 1]], dtype=np.int64)
    unity_normals = _vertex_normals(unity_vertices, unity_faces)

    vertices_f32 = np.ascontiguousarray(unity_vertices, dtype="<f4")
    normals_f32 = np.ascontiguousarray(unity_normals, dtype="<f4")
    indices_u32 = np.ascontiguousarray(unity_faces.reshape(-1), dtype="<u4")
    flags = 1  # vertex normals are present
    payload = b"".join(
        (
            _HEADER.pack(
                MOBILE_OVERLAY_MAGIC,
                MOBILE_OVERLAY_VERSION,
                len(vertices_f32),
                len(indices_u32),
                flags,
            ),
            vertices_f32.tobytes(),
            normals_f32.tobytes(),
            indices_u32.tobytes(),
        )
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    temporary.write_bytes(payload)
    temporary.replace(output_path)

    report = {
        "format": MOBILE_OVERLAY_FORMAT,
        "created_at_utc": _utc_now(),
        "passed": True,
        "source_world_obj": str(world_obj),
        "output": str(output_path),
        "source_coordinate_frame": str(source_coordinate_frame),
        "coordinate_frame": str(output_coordinate_frame),
        "source_to_output": (
            f"(x,y,z)_{source_coordinate_frame} -> "
            f"(x,y,-z)_{output_coordinate_frame}"
        ),
        "source_vertex_count": int(len(source_vertices)),
        "source_triangle_count": int(len(source_faces)),
        "vertex_count": int(len(vertices_f32)),
        "triangle_count": int(len(unity_faces)),
        "max_triangles": int(max_triangles),
        "byte_count": int(len(payload)),
        "bounds_min": unity_vertices.min(axis=0).tolist(),
        "bounds_max": unity_vertices.max(axis=0).tolist(),
        "reduction": reduction,
        "display_only": True,
        "formal_mesh_unchanged": True,
    }
    report_path = output_path.with_suffix(output_path.suffix + ".json")
    temporary_report = report_path.with_name(f".{report_path.name}.tmp")
    temporary_report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary_report.replace(report_path)
    return report


def read_mobile_overlay_mesh(path: Path) -> dict[str, Any]:
    """Read the mobile format for tests and offline diagnostics."""

    payload = path.read_bytes()
    if len(payload) < _HEADER.size:
        raise ValueError("mobile overlay payload is truncated")
    magic, version, vertex_count, index_count, flags = _HEADER.unpack_from(payload)
    if magic != MOBILE_OVERLAY_MAGIC or version != MOBILE_OVERLAY_VERSION:
        raise ValueError("unsupported mobile overlay mesh header")
    expected = _HEADER.size + vertex_count * 3 * 4 * 2 + index_count * 4
    if len(payload) != expected:
        raise ValueError(f"mobile overlay byte count mismatch: {len(payload)} != {expected}")
    offset = _HEADER.size
    vertices = np.frombuffer(
        payload, dtype="<f4", count=vertex_count * 3, offset=offset
    ).reshape(vertex_count, 3)
    offset += vertex_count * 3 * 4
    normals = np.frombuffer(
        payload, dtype="<f4", count=vertex_count * 3, offset=offset
    ).reshape(vertex_count, 3)
    offset += vertex_count * 3 * 4
    indices = np.frombuffer(
        payload, dtype="<u4", count=index_count, offset=offset
    ).reshape(-1, 3)
    return {
        "version": int(version),
        "flags": int(flags),
        "vertices": vertices,
        "normals": normals,
        "faces": indices,
    }
