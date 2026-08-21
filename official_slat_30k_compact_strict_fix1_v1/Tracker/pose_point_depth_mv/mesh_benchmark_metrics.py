#!/usr/bin/env python3
"""Lightweight deterministic Mesh metrics for reconstruction benchmarks."""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable

import numpy as np
from scipy.spatial import cKDTree
import trimesh


def deterministic_surface_sample(
    mesh: trimesh.Trimesh,
    count: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if not len(vertices) or not len(faces):
        raise ValueError("cannot sample an empty mesh")
    triangles = vertices[faces]
    cross = np.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    )
    double_area = np.linalg.norm(cross, axis=1)
    valid = np.isfinite(double_area) & (double_area > 1.0e-15)
    if not np.any(valid):
        raise ValueError("mesh contains no finite non-degenerate triangles")
    valid_ids = np.flatnonzero(valid)
    probability = double_area[valid] / double_area[valid].sum()
    rng = np.random.default_rng(int(seed))
    face_ids = rng.choice(valid_ids, size=int(count), replace=True, p=probability)
    u = rng.random(int(count))
    v = rng.random(int(count))
    sqrt_u = np.sqrt(u)
    weights = np.stack(
        (1.0 - sqrt_u, sqrt_u * (1.0 - v), sqrt_u * v), axis=1
    )
    selected = triangles[face_ids]
    points = np.sum(selected * weights[:, :, None], axis=1)
    normals = cross[face_ids]
    normals /= np.maximum(
        np.linalg.norm(normals, axis=1, keepdims=True), 1.0e-15
    )
    return points, normals


def surface_metrics(
    predicted: trimesh.Trimesh,
    target: trimesh.Trimesh,
    *,
    count: int,
    seed: int,
    thresholds: Iterable[float],
) -> dict[str, float]:
    pred_points, pred_normals = deterministic_surface_sample(predicted, count, seed)
    # Shared random variates make an identical triangulated Mesh exactly equal.
    target_points, target_normals = deterministic_surface_sample(target, count, seed)
    target_tree = cKDTree(target_points)
    pred_tree = cKDTree(pred_points)
    pred_distance, pred_index = target_tree.query(pred_points, k=1, workers=-1)
    target_distance, target_index = pred_tree.query(target_points, k=1, workers=-1)
    output = {
        "pred_to_gt_mean": float(np.mean(pred_distance)),
        "gt_to_pred_mean": float(np.mean(target_distance)),
        "chamfer_l1": float(
            0.5 * (np.mean(pred_distance) + np.mean(target_distance))
        ),
        "chamfer_l2": float(
            0.5 * (np.mean(pred_distance**2) + np.mean(target_distance**2))
        ),
        "normal_consistency": float(
            0.5
            * (
                np.mean(
                    np.abs(
                        np.sum(
                            pred_normals * target_normals[pred_index], axis=1
                        )
                    )
                )
                + np.mean(
                    np.abs(
                        np.sum(
                            target_normals * pred_normals[target_index], axis=1
                        )
                    )
                )
            )
        ),
    }
    for threshold in thresholds:
        key = str(float(threshold)).replace(".", "p")
        precision = float(np.mean(pred_distance < float(threshold)))
        recall = float(np.mean(target_distance < float(threshold)))
        output[f"precision_{key}"] = precision
        output[f"recall_{key}"] = recall
        output[f"fscore_{key}"] = (
            0.0
            if precision + recall <= 1.0e-12
            else float(2.0 * precision * recall / (precision + recall))
        )
    return output


def mesh_structure_metrics(mesh: trimesh.Trimesh) -> dict[str, Any]:
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    finite = bool(np.isfinite(vertices).all())
    result: dict[str, Any] = {
        "mesh_success": bool(len(vertices) > 0 and len(faces) > 0 and finite),
        "vertex_count": int(len(vertices)),
        "face_count": int(len(faces)),
        "vertices_finite": finite,
        "is_watertight": bool(mesh.is_watertight) if len(faces) else False,
        "is_winding_consistent": bool(mesh.is_winding_consistent)
        if len(faces)
        else False,
    }
    if len(vertices):
        extent = np.ptp(vertices, axis=0)
        result["bbox_extent"] = [float(value) for value in extent]
        result["bbox_diag"] = float(np.linalg.norm(extent))
    if not len(faces):
        result.update(
            {
                "component_count": 0,
                "largest_component_ratio": 0.0,
                "small_component_vertex_ratio_lt100": 0.0,
                "boundary_edge_count": 0,
                "boundary_total_length": 0.0,
                "nonmanifold_edge_count": 0,
            }
        )
        return result

    parent = np.arange(len(vertices), dtype=np.int64)
    size = np.ones(len(vertices), dtype=np.int64)

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return int(index)

    def union(left: int, right: int) -> None:
        left_root, right_root = find(int(left)), find(int(right))
        if left_root == right_root:
            return
        if size[left_root] < size[right_root]:
            left_root, right_root = right_root, left_root
        parent[right_root] = left_root
        size[left_root] += size[right_root]

    for face in faces:
        union(face[0], face[1])
        union(face[1], face[2])
        union(face[2], face[0])
    referenced = np.unique(faces.reshape(-1))
    component_sizes = Counter(find(int(index)) for index in referenced)
    sizes = sorted(component_sizes.values(), reverse=True)
    edges = np.sort(
        np.concatenate(
            (faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]), axis=0
        ),
        axis=1,
    )
    unique_edges, edge_counts = np.unique(edges, axis=0, return_counts=True)
    boundary_edges = unique_edges[edge_counts == 1]
    boundary_total_length = (
        float(
            np.linalg.norm(
                vertices[boundary_edges[:, 0]] - vertices[boundary_edges[:, 1]],
                axis=1,
            ).sum()
        )
        if len(boundary_edges)
        else 0.0
    )
    result.update(
        {
            "component_count": len(sizes),
            "largest_component_ratio": float(sizes[0] / max(sum(sizes), 1)),
            "small_component_vertex_ratio_lt100": float(
                sum(value for value in sizes if value < 100) / max(sum(sizes), 1)
            ),
            "boundary_edge_count": int(np.sum(edge_counts == 1)),
            "boundary_total_length": boundary_total_length,
            "nonmanifold_edge_count": int(np.sum(edge_counts > 2)),
        }
    )
    return result
