#!/usr/bin/env python3
"""Symmetric, CPU-only cleanup diagnostic for exploratory Stage-B Mesh pairs.

This program never re-runs SS/SLAT and never changes the source meshes.  It
applies the same connected-component policy to both blinded sides, recomputes
surface/topology metrics in the frozen canonical frame, and reports paired
stock/correct deltas.  Every result is exploratory and non-formal.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any, Iterable

import numpy as np
from scipy import sparse
from scipy.sparse import csgraph
from scipy.spatial import cKDTree
import trimesh


REPORT_FORMAT = "pose_point_depth_mv.mesh_cleanup_diagnostic.v1"
POLICIES = ("raw", "largest_component", "drop_components_lt_min_vertices")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_output", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--min_component_vertices", type=int, default=100)
    parser.add_argument("--surface_samples", type=int, default=20000)
    parser.add_argument("--bootstrap_samples", type=int, default=5000)
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="CPU workers used by each scipy KD-tree query (no CUDA is used).",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--export_cleaned_meshes",
        action="store_true",
        help="Export blinded canonical OBJ files for the two cleanup policies.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(
        path,
        json.dumps(value, indent=2, sort_keys=False, allow_nan=False) + "\n",
    )


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def numeric_tree_is_finite(value: Any) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return bool(np.isfinite(float(value)))
    if isinstance(value, dict):
        return all(numeric_tree_is_finite(child) for child in value.values())
    if isinstance(value, (list, tuple)):
        return all(numeric_tree_is_finite(child) for child in value)
    return False


def validate_listed_completion(root: Path, completion: dict[str, Any]) -> None:
    rows = completion.get("files", [])
    if len(rows) != int(completion.get("file_count", -1)):
        raise RuntimeError("source completion file count differs")
    seen: set[str] = set()
    for row in rows:
        relative = Path(str(row["path"]))
        key = relative.as_posix()
        if relative.is_absolute() or ".." in relative.parts or key in seen:
            raise RuntimeError(f"unsafe/duplicate source completion path: {relative}")
        seen.add(key)
        artifact = root / relative
        if not artifact.is_file() or sha256_file(artifact) != str(row["sha256"]):
            raise RuntimeError(f"source completion artifact hash mismatch: {artifact}")


def load_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, force="scene", process=False)
    if isinstance(loaded, trimesh.Scene):
        pieces = [
            item
            for item in loaded.dump(concatenate=False)
            if isinstance(item, trimesh.Trimesh) and len(item.vertices) and len(item.faces)
        ]
        if not pieces:
            raise ValueError(f"mesh contains no triangles: {path}")
        mesh = trimesh.util.concatenate(pieces)
    elif isinstance(loaded, trimesh.Trimesh):
        mesh = loaded
    else:
        raise TypeError(f"unsupported mesh type={type(loaded)}: {path}")
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    if not len(vertices) or not len(faces) or not np.isfinite(vertices).all():
        raise ValueError(f"empty or non-finite mesh: {path}")
    return mesh


def load_canonical_target(target_row: dict[str, Any]) -> trimesh.Trimesh:
    source_path = Path(target_row["source_glb"]).resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    expected = str(target_row.get("source_glb_sha256", ""))
    if expected and sha256_file(source_path) != expected:
        raise RuntimeError(f"target GLB hash mismatch: {source_path}")
    loaded = trimesh.load(source_path, force="scene", process=False)
    if isinstance(loaded, trimesh.Scene):
        pieces = [
            item
            for item in loaded.dump(concatenate=False)
            if isinstance(item, trimesh.Trimesh) and len(item.vertices) and len(item.faces)
        ]
        if not pieces:
            raise ValueError(f"target GLB contains no triangles: {source_path}")
        mesh = trimesh.util.concatenate(pieces)
    elif isinstance(loaded, trimesh.Trimesh):
        mesh = loaded.copy()
    else:
        raise TypeError(f"unsupported target mesh type={type(loaded)}: {source_path}")
    center = np.asarray(target_row["normalize_center"], dtype=np.float64)
    scale = float(target_row["normalize_scale"])
    margin = float(target_row["canonical_margin"])
    if center.shape != (3,) or not np.isfinite(center).all() or scale <= 0.0:
        raise ValueError(f"invalid canonical target transform: {target_row}")
    mesh.vertices = (np.asarray(mesh.vertices, dtype=np.float64) - center[None]) / scale * margin
    return mesh


def component_partition(mesh: trimesh.Trimesh) -> list[dict[str, Any]]:
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if not len(vertices) or not len(faces):
        return []
    if np.min(faces) < 0 or np.max(faces) >= len(vertices):
        raise ValueError("face index outside vertex array")

    edges = np.concatenate(
        (faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]), axis=0
    )
    graph = sparse.coo_matrix(
        (np.ones(len(edges), dtype=np.int32), (edges[:, 0], edges[:, 1])),
        shape=(len(vertices), len(vertices)),
    ).tocsr()
    _, labels = csgraph.connected_components(graph, directed=False, return_labels=True)
    face_labels = labels[faces[:, 0]]
    output = []
    for label in np.unique(face_labels):
        face_array = np.flatnonzero(face_labels == label).astype(np.int64, copy=False)
        referenced = np.unique(faces[face_array].reshape(-1))
        output.append(
            {
                "face_indices": face_array,
                "vertex_count": int(len(referenced)),
                "face_count": int(len(face_array)),
                "minimum_vertex_index": int(referenced.min()),
            }
        )
    output.sort(
        key=lambda row: (
            -int(row["vertex_count"]),
            int(row["minimum_vertex_index"]),
        )
    )
    return output


def select_faces(
    mesh: trimesh.Trimesh,
    *,
    policy: str,
    min_component_vertices: int,
    components: list[dict[str, Any]] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    faces = np.asarray(mesh.faces, dtype=np.int64)
    components = component_partition(mesh) if components is None else components
    if not components:
        raise ValueError("cannot clean a mesh without connected components")
    fallback = False
    if policy == "raw":
        kept = components
    elif policy == "largest_component":
        kept = components[:1]
    elif policy == "drop_components_lt_min_vertices":
        kept = [
            row
            for row in components
            if int(row["vertex_count"]) >= int(min_component_vertices)
        ]
        if not kept:
            # A valid diagnostic must still yield a mesh.  The fallback is
            # deterministic and is applied identically to stock and correct.
            kept = components[:1]
            fallback = True
    else:
        raise ValueError(f"unknown cleanup policy={policy}")
    face_indices = np.sort(
        np.concatenate([np.asarray(row["face_indices"], dtype=np.int64) for row in kept])
    )
    if not len(face_indices):
        raise ValueError(f"cleanup policy produced no faces: {policy}")
    signature = hashlib.sha256(face_indices.tobytes(order="C")).hexdigest()
    metadata = {
        "policy": policy,
        "threshold_vertices": int(min_component_vertices),
        "source_component_count": int(len(components)),
        "retained_component_count": int(len(kept)),
        "removed_component_count": int(len(components) - len(kept)),
        "source_referenced_vertex_count": int(
            sum(int(row["vertex_count"]) for row in components)
        ),
        "retained_referenced_vertex_count_before_compaction": int(
            sum(int(row["vertex_count"]) for row in kept)
        ),
        "retained_component_vertex_counts": [
            int(row["vertex_count"]) for row in kept
        ],
        "fallback_to_largest_component": bool(fallback),
        "face_selection_sha256": signature,
        "selected_face_count": int(len(face_indices)),
        "source_face_count": int(len(faces)),
    }
    return face_indices, metadata


def compact_submesh(mesh: trimesh.Trimesh, face_indices: np.ndarray) -> trimesh.Trimesh:
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces, dtype=np.int64)[face_indices]
    used = np.unique(faces.reshape(-1))
    remap = np.full(len(vertices), -1, dtype=np.int64)
    remap[used] = np.arange(len(used), dtype=np.int64)
    kwargs: dict[str, Any] = {}
    try:
        colors = np.asarray(mesh.visual.vertex_colors)
        if len(colors) == len(vertices):
            kwargs["vertex_colors"] = colors[used]
    except (AttributeError, ValueError):
        pass
    output = trimesh.Trimesh(
        vertices=np.asarray(vertices[used]).copy(),
        faces=remap[faces],
        process=False,
        **kwargs,
    )
    if not len(output.vertices) or not len(output.faces):
        raise ValueError("compacted cleanup mesh is empty")
    return output


def deterministic_surface_sample(
    mesh: trimesh.Trimesh,
    count: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    triangles = vertices[faces]
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    double_area = np.linalg.norm(cross, axis=1)
    valid = np.isfinite(double_area) & (double_area > 1.0e-15)
    if not np.any(valid):
        raise ValueError("mesh contains no finite, non-degenerate triangles")
    valid_ids = np.flatnonzero(valid)
    probability = double_area[valid] / double_area[valid].sum()
    rng = np.random.default_rng(int(seed))
    face_ids = rng.choice(valid_ids, size=int(count), replace=True, p=probability)
    u = rng.random(int(count))
    v = rng.random(int(count))
    sqrt_u = np.sqrt(u)
    weights = np.stack((1.0 - sqrt_u, sqrt_u * (1.0 - v), sqrt_u * v), axis=1)
    selected = triangles[face_ids]
    points = np.sum(selected * weights[:, :, None], axis=1)
    normals = cross[face_ids]
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1.0e-15)
    return points, normals


def prepare_target_sample(
    target: trimesh.Trimesh,
    *,
    count: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, cKDTree]:
    points, normals = deterministic_surface_sample(target, count, seed)
    return points, normals, cKDTree(points)


def surface_metrics(
    predicted: trimesh.Trimesh,
    target_sample: tuple[np.ndarray, np.ndarray, cKDTree],
    *,
    count: int,
    seed: int,
    thresholds: Iterable[float],
    workers: int,
) -> dict[str, float]:
    pred_points, pred_normals = deterministic_surface_sample(predicted, count, seed)
    target_points, target_normals, target_tree = target_sample
    pred_tree = cKDTree(pred_points)
    pred_distance, pred_index = target_tree.query(
        pred_points, k=1, workers=int(workers)
    )
    target_distance, target_index = pred_tree.query(
        target_points, k=1, workers=int(workers)
    )
    output = {
        "pred_to_gt_mean": float(np.mean(pred_distance)),
        "gt_to_pred_mean": float(np.mean(target_distance)),
        "chamfer_l1": float(0.5 * (np.mean(pred_distance) + np.mean(target_distance))),
        "chamfer_l2": float(
            0.5 * (np.mean(pred_distance**2) + np.mean(target_distance**2))
        ),
        "normal_consistency": float(
            0.5
            * (
                np.mean(np.abs(np.sum(pred_normals * target_normals[pred_index], axis=1)))
                + np.mean(
                    np.abs(np.sum(target_normals * pred_normals[target_index], axis=1))
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


def mesh_structure_metrics(
    mesh: trimesh.Trimesh,
    *,
    component_vertex_counts: Iterable[int] | None = None,
) -> dict[str, Any]:
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    finite = bool(np.isfinite(vertices).all())
    result: dict[str, Any] = {
        "mesh_success": bool(len(vertices) > 0 and len(faces) > 0 and finite),
        "vertex_count": int(len(vertices)),
        "face_count": int(len(faces)),
        "vertices_finite": finite,
        "is_watertight": bool(mesh.is_watertight) if len(faces) else False,
        "is_winding_consistent": bool(mesh.is_winding_consistent) if len(faces) else False,
    }
    if len(vertices):
        extent = np.ptp(vertices, axis=0)
        result["bbox_extent"] = [float(value) for value in extent]
        result["bbox_diag"] = float(np.linalg.norm(extent))
    sizes = (
        [int(value) for value in component_vertex_counts]
        if component_vertex_counts is not None
        else [
            int(row["vertex_count"])
            for row in (component_partition(mesh) if len(faces) else [])
        ]
    )
    if not sizes:
        result.update(
            {
                "component_count": 0,
                "largest_component_ratio": 0.0,
                "small_component_vertex_ratio_lt100": 0.0,
                "boundary_edge_count": 0,
                "nonmanifold_edge_count": 0,
            }
        )
        return result
    edges = np.sort(
        np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]), axis=0),
        axis=1,
    )
    _, edge_counts = np.unique(edges, axis=0, return_counts=True)
    result.update(
        {
            "component_count": int(len(sizes)),
            "largest_component_ratio": float(max(sizes) / max(sum(sizes), 1)),
            "small_component_vertex_ratio_lt100": float(
                sum(value for value in sizes if value < 100) / max(sum(sizes), 1)
            ),
            "boundary_edge_count": int(np.sum(edge_counts == 1)),
            "nonmanifold_edge_count": int(np.sum(edge_counts > 2)),
        }
    )
    return result


def summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return {"count": 0, "mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
    return {
        "count": int(len(array)),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def bootstrap_mean_ci(values: list[float], *, samples: int, seed: int) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return [0.0, 0.0]
    rng = np.random.default_rng(int(seed))
    means = np.empty(int(samples), dtype=np.float64)
    # Chunking avoids a large bootstrap index matrix.
    chunk = 512
    for start in range(0, int(samples), chunk):
        stop = min(start + chunk, int(samples))
        indices = rng.integers(0, len(array), size=(stop - start, len(array)))
        means[start:stop] = np.mean(array[indices], axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def summarize_paired(
    records: list[dict[str, Any]],
    *,
    bootstrap_samples: int,
    primary_fscore: float,
) -> dict[str, Any]:
    by_pair: dict[tuple[str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    duplicates: Counter[tuple[str, int, str]] = Counter()
    for row in records:
        key = (str(row["object_uid"]), int(row["seed"]))
        branch = str(row["branch"])
        duplicates[(key[0], key[1], branch)] += 1
        by_pair[key][branch] = row
    fscore_key = f"fscore_{str(float(primary_fscore)).replace('.', 'p')}"
    pair_deltas = []
    invalid_pairs = []
    for key, branches in sorted(by_pair.items()):
        counts = {
            branch: duplicates[(key[0], key[1], branch)]
            for branch in ("stock", "correct")
        }
        if counts != {"stock": 1, "correct": 1} or set(branches) != {"stock", "correct"}:
            invalid_pairs.append({"object_uid": key[0], "seed": key[1], "counts": counts})
            continue
        stock, correct = branches["stock"], branches["correct"]
        pair_deltas.append(
            {
                "object_uid": key[0],
                "seed": key[1],
                "chamfer_l1_improvement": float(stock["surface"]["chamfer_l1"])
                - float(correct["surface"]["chamfer_l1"]),
                "fscore_0p02_delta": float(correct["surface"][fscore_key])
                - float(stock["surface"][fscore_key]),
                "normal_consistency_delta": float(
                    correct["surface"]["normal_consistency"]
                )
                - float(stock["surface"]["normal_consistency"]),
                "largest_component_ratio_delta": float(
                    correct["structure"]["largest_component_ratio"]
                )
                - float(stock["structure"]["largest_component_ratio"]),
                "component_count_improvement": float(stock["structure"]["component_count"])
                - float(correct["structure"]["component_count"]),
                "small_component_vertex_ratio_improvement": float(
                    stock["structure"]["small_component_vertex_ratio_lt100"]
                )
                - float(correct["structure"]["small_component_vertex_ratio_lt100"]),
                "boundary_edge_count_improvement": float(
                    stock["structure"]["boundary_edge_count"]
                )
                - float(correct["structure"]["boundary_edge_count"]),
                "nonmanifold_edge_count_improvement": float(
                    stock["structure"]["nonmanifold_edge_count"]
                )
                - float(correct["structure"]["nonmanifold_edge_count"]),
                "watertight_delta": float(correct["structure"]["is_watertight"])
                - float(stock["structure"]["is_watertight"]),
                "mesh_success_delta": float(correct["structure"]["mesh_success"])
                - float(stock["structure"]["mesh_success"]),
            }
        )
    metrics = tuple(key for key in pair_deltas[0] if key not in {"object_uid", "seed"}) if pair_deltas else ()
    by_object: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in pair_deltas:
        by_object[str(row["object_uid"])].append(row)
    object_rows = []
    for object_uid, rows in sorted(by_object.items()):
        object_rows.append(
            {
                "object_uid": object_uid,
                **{
                    metric: float(np.mean([float(row[metric]) for row in rows]))
                    for metric in metrics
                },
            }
        )
    summary = {}
    for index, metric in enumerate(metrics):
        values = [float(row[metric]) for row in object_rows]
        summary[metric] = {
            **summarize(values),
            "object_win_rate": float(np.mean(np.asarray(values) > 0.0)) if values else 0.0,
            "bootstrap_mean_95_ci": bootstrap_mean_ci(
                values, samples=bootstrap_samples, seed=88000 + index
            ),
        }
    seed_summary = {}
    for seed in sorted({int(row["seed"]) for row in pair_deltas}):
        rows = [row for row in pair_deltas if int(row["seed"]) == seed]
        seed_summary[str(seed)] = {
            metric: summarize([float(row[metric]) for row in rows]) for metric in metrics
        }
    return {
        "formal": False,
        "claim_limit": "exploratory symmetric CPU post-processing diagnostic only",
        "record_count": int(len(records)),
        "completed_pair_count": int(len(by_pair)),
        "valid_pair_count": int(len(pair_deltas)),
        "invalid_pairs": invalid_pairs,
        "formal_weighting": "paired seeds averaged per object, then object bootstrap",
        "summary": summary,
        "seed_summary": seed_summary,
        "object_rows": object_rows,
        "pair_deltas": pair_deltas,
    }


def raw_reconciliation(
    raw_records: list[dict[str, Any]],
    source_records: list[dict[str, Any]],
    *,
    evaluated: bool,
) -> dict[str, Any]:
    """Check that reopening/exporting the raw OBJ preserves the metric contract."""
    if not evaluated:
        return {
            "evaluated": False,
            "passed": False,
            "reason": "surface_samples differs from the frozen source evaluation",
        }
    source_by_key = {
        (str(row["pair_id"]), str(row["side"])): row for row in source_records
    }
    raw_by_key = {
        (str(row["pair_id"]), str(row["side"])): row for row in raw_records
    }
    source_keys = [(str(row["pair_id"]), str(row["side"])) for row in source_records]
    raw_keys = [(str(row["pair_id"]), str(row["side"])) for row in raw_records]
    duplicate_source_keys = len(source_keys) - len(set(source_keys))
    duplicate_raw_keys = len(raw_keys) - len(set(raw_keys))
    key_sets_match = set(source_by_key) == set(raw_by_key)
    maximum_surface = 0.0
    maximum_structure = 0.0
    boolean_mismatches = 0
    missing = []
    missing_metrics = []
    for row in raw_records:
        key = (str(row["pair_id"]), str(row["side"]))
        source = source_by_key.get(key)
        if source is None:
            missing.append(list(key))
            continue
        for group in ("surface", "structure"):
            if set(row[group]) != set(source[group]):
                missing_metrics.append(
                    f"{key}:{group}:keys raw={sorted(row[group])} "
                    f"source={sorted(source[group])}"
                )
        for metric, value in row["surface"].items():
            if metric in source["surface"]:
                maximum_surface = max(
                    maximum_surface,
                    abs(float(value) - float(source["surface"][metric])),
                )
            else:
                missing_metrics.append(f"{key}:surface.{metric}")
        for metric, value in row["structure"].items():
            if metric not in source["structure"]:
                missing_metrics.append(f"{key}:structure.{metric}")
                continue
            source_value = source["structure"][metric]
            if isinstance(value, bool):
                boolean_mismatches += int(bool(value) != bool(source_value))
            elif isinstance(value, (int, float)):
                maximum_structure = max(
                    maximum_structure, abs(float(value) - float(source_value))
                )
            elif isinstance(value, list) and isinstance(source_value, list):
                if len(value) == len(source_value) and value:
                    maximum_structure = max(
                        maximum_structure,
                        float(
                            np.max(
                                np.abs(
                                    np.asarray(value, dtype=np.float64)
                                    - np.asarray(source_value, dtype=np.float64)
                                )
                            )
                        ),
                    )
                elif len(value) != len(source_value):
                    boolean_mismatches += 1
    surface_tolerance = 5.0e-6
    structure_tolerance = 5.0e-6
    return {
        "evaluated": True,
        "passed": bool(
            not missing
            and key_sets_match
            and duplicate_source_keys == 0
            and duplicate_raw_keys == 0
            and not missing_metrics
            and boolean_mismatches == 0
            and maximum_surface <= surface_tolerance
            and maximum_structure <= structure_tolerance
        ),
        "record_count": len(raw_records),
        "missing_source_records": missing,
        "key_sets_match": bool(key_sets_match),
        "duplicate_source_key_count": int(duplicate_source_keys),
        "duplicate_raw_key_count": int(duplicate_raw_keys),
        "missing_metric_count": len(missing_metrics),
        "first_missing_metrics": missing_metrics[:20],
        "surface_max_abs": float(maximum_surface),
        "surface_abs_tolerance": surface_tolerance,
        "structure_numeric_max_abs": float(maximum_structure),
        "structure_abs_tolerance": structure_tolerance,
        "structure_boolean_mismatch_count": int(boolean_mismatches),
    }


def summarize_object_weighted_rows(
    rows: list[dict[str, Any]],
    *,
    metrics: Iterable[str],
    bootstrap_samples: int,
    seed_base: int,
) -> dict[str, Any]:
    by_object: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_object[str(row["object_uid"])].append(row)
    object_rows = []
    for object_uid, object_values in sorted(by_object.items()):
        object_rows.append(
            {
                "object_uid": object_uid,
                **{
                    metric: float(
                        np.mean([float(value[metric]) for value in object_values])
                    )
                    for metric in metrics
                },
            }
        )
    summary = {}
    for index, metric in enumerate(metrics):
        values = [float(row[metric]) for row in object_rows]
        summary[metric] = {
            **summarize(values),
            "object_win_rate": float(np.mean(np.asarray(values) > 0.0)) if values else 0.0,
            "bootstrap_mean_95_ci": bootstrap_mean_ci(
                values,
                samples=int(bootstrap_samples),
                seed=int(seed_base) + index,
            ),
        }
    return {"summary": summary, "object_rows": object_rows, "rows": rows}


def cleanup_effects(
    records_by_policy: dict[str, list[dict[str, Any]]],
    policy_reports: dict[str, dict[str, Any]],
    *,
    bootstrap_samples: int,
    primary_fscore: float,
) -> dict[str, Any]:
    raw_records = {
        (str(row["object_uid"]), int(row["seed"]), str(row["branch"])): row
        for row in records_by_policy["raw"]
    }
    fscore_key = f"fscore_{str(float(primary_fscore)).replace('.', 'p')}"
    effect_metrics = (
        "chamfer_l1_improvement",
        "fscore_0p02_delta",
        "normal_consistency_delta",
        "largest_component_ratio_delta",
        "component_count_reduction",
        "small_component_vertex_ratio_reduction",
        "boundary_edge_count_reduction",
    )
    paired_metrics = tuple(
        key
        for key in policy_reports["raw"].get("pair_deltas", [{}])[0]
        if key not in {"object_uid", "seed"}
    ) if policy_reports["raw"].get("pair_deltas") else ()
    raw_pair = {
        (str(row["object_uid"]), int(row["seed"])): row
        for row in policy_reports["raw"]["pair_deltas"]
    }
    output = {}
    for policy_index, policy in enumerate(POLICIES[1:]):
        branch_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in records_by_policy[policy]:
            key = (str(row["object_uid"]), int(row["seed"]), str(row["branch"]))
            raw = raw_records[key]
            branch_rows[str(row["branch"])].append(
                {
                    "object_uid": key[0],
                    "seed": key[1],
                    "branch": key[2],
                    # All signs below are oriented so positive means cleanup helped.
                    "chamfer_l1_improvement": float(raw["surface"]["chamfer_l1"])
                    - float(row["surface"]["chamfer_l1"]),
                    "fscore_0p02_delta": float(row["surface"][fscore_key])
                    - float(raw["surface"][fscore_key]),
                    "normal_consistency_delta": float(
                        row["surface"]["normal_consistency"]
                    )
                    - float(raw["surface"]["normal_consistency"]),
                    "largest_component_ratio_delta": float(
                        row["structure"]["largest_component_ratio"]
                    )
                    - float(raw["structure"]["largest_component_ratio"]),
                    "component_count_reduction": float(
                        raw["structure"]["component_count"]
                    )
                    - float(row["structure"]["component_count"]),
                    "small_component_vertex_ratio_reduction": float(
                        raw["structure"]["small_component_vertex_ratio_lt100"]
                    )
                    - float(row["structure"]["small_component_vertex_ratio_lt100"]),
                    "boundary_edge_count_reduction": float(
                        raw["structure"]["boundary_edge_count"]
                    )
                    - float(row["structure"]["boundary_edge_count"]),
                }
            )
        paired_change_rows = []
        for row in policy_reports[policy]["pair_deltas"]:
            key = (str(row["object_uid"]), int(row["seed"]))
            raw = raw_pair[key]
            paired_change_rows.append(
                {
                    "object_uid": key[0],
                    "seed": key[1],
                    **{
                        metric: float(row[metric]) - float(raw[metric])
                        for metric in paired_metrics
                    },
                }
            )
        output[policy] = {
            "sign_convention": (
                "branch effects are positive when cleanup improves that branch; paired "
                "changes are cleanup-policy correct-vs-stock delta minus raw delta"
            ),
            "within_branch": {
                branch: summarize_object_weighted_rows(
                    rows,
                    metrics=effect_metrics,
                    bootstrap_samples=int(bootstrap_samples),
                    seed_base=91000 + policy_index * 100 + branch_index * 10,
                )
                for branch_index, (branch, rows) in enumerate(sorted(branch_rows.items()))
            },
            "paired_improvement_change_vs_raw": summarize_object_weighted_rows(
                paired_change_rows,
                metrics=paired_metrics,
                bootstrap_samples=int(bootstrap_samples),
                seed_base=93000 + policy_index * 100,
            ),
        }
    return output


def safe_export_obj(mesh: trimesh.Trimesh, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.obj")
    mesh.export(temporary, file_type="obj")
    os.replace(temporary, path)


def atomic_link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
    if temporary.exists():
        temporary.unlink()
    try:
        os.link(source, temporary)
    except OSError:
        shutil.copy2(source, temporary)
    os.replace(temporary, destination)


def audit_cleaned_blind_tree(root: Path, *, expected_files: int) -> dict[str, Any]:
    if not root.exists():
        return {
            "evaluated": False,
            "passed": expected_files == 0,
            "expected_file_count": int(expected_files),
            "file_count": 0,
        }
    files = sorted(path for path in root.rglob("*") if path.is_file())
    bad_paths = []
    bad_types = []
    for path in files:
        relative = path.relative_to(root).as_posix()
        lowered = relative.lower()
        if "stock" in lowered or "correct" in lowered or "branch" in lowered:
            bad_paths.append(relative)
        if path.suffix.lower() != ".obj":
            bad_types.append(relative)
    return {
        "evaluated": True,
        "passed": bool(
            len(files) == int(expected_files) and not bad_paths and not bad_types
        ),
        "expected_file_count": int(expected_files),
        "file_count": len(files),
        "branch_leaking_paths": bad_paths,
        "non_obj_files": bad_types,
    }


def validate_source(
    source_output: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, int]]:
    report_path = source_output / "report.json"
    protocol_path = source_output / "source_frozen_protocol.json"
    completion_path = source_output / "completion_manifest.json"
    if not report_path.is_file() or not protocol_path.is_file() or not completion_path.is_file():
        raise FileNotFoundError(
            "source report.json/source_frozen_protocol.json/completion_manifest.json is missing"
        )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    if not (
        completion.get("complete") is True
        and completion.get("exploratory") is True
        and completion.get("formal") is False
        and completion.get("all_records_passed") is True
        and int(completion.get("runtime_exit_code", -1)) == 0
    ):
        raise RuntimeError("source completion manifest is not a successful exploratory run")
    validate_listed_completion(source_output, completion)
    if report.get("exploratory") is not True or report.get("formal") is not False:
        raise RuntimeError("source output is not the expected exploratory/non-formal run")
    if report.get("all_records_passed") is not True:
        raise RuntimeError("source report does not declare all_records_passed=true")
    records = list(report.get("records", []))
    if not records or int(report.get("record_count", -1)) != len(records):
        raise RuntimeError("source report record count is missing or inconsistent")
    if not all(bool(row.get("passed")) for row in records):
        raise RuntimeError("source report contains failed Mesh records")
    if str(report.get("protocol_sha256")) != str(protocol.get("protocol_sha256")):
        raise RuntimeError("source report/protocol binding mismatch")
    if str(completion.get("protocol_sha256")) != str(protocol.get("protocol_sha256")):
        raise RuntimeError("source completion/protocol binding mismatch")
    protocol_payload = dict(protocol)
    claimed_protocol_sha256 = str(protocol_payload.pop("protocol_sha256", ""))
    if not claimed_protocol_sha256 or canonical_sha256(protocol_payload) != claimed_protocol_sha256:
        raise RuntimeError("source frozen protocol canonical SHA-256 is invalid")
    rollout_by_uid = {
        str(row["uid"]): int(row["rollout_position"])
        for row in protocol["selection"]["rows"]
    }
    if any(str(row["uid"]) not in rollout_by_uid for row in records):
        raise RuntimeError("source record UID is absent from frozen protocol")
    expected_pairs = len(protocol["selection"]["rows"]) * len(
        protocol["sampling"]["joint_seeds"]
    )
    if len(records) != 2 * expected_pairs:
        raise RuntimeError(
            f"expected {2 * expected_pairs} source records, found {len(records)}"
        )
    if expected_pairs != 96 or len(protocol["selection"]["rows"]) != 32:
        raise RuntimeError(
            "this diagnostic expects the frozen 32-object x 3-seed Stage-B protocol"
        )
    expected_seeds = {42, 43, 44}
    if {int(value) for value in protocol["sampling"]["joint_seeds"]} != expected_seeds:
        raise RuntimeError("frozen Stage-B joint seeds are not {42,43,44}")

    by_pair: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    target_by_uid: dict[str, str] = {}
    for row in records:
        pair_key = (str(row["object_uid"]), int(row["seed"]))
        by_pair[pair_key].append(row)
        target_identity = json.dumps(row["target"], sort_keys=True, separators=(",", ":"))
        previous = target_by_uid.setdefault(str(row["uid"]), target_identity)
        if previous != target_identity:
            raise RuntimeError(f"target identity differs within uid={row['uid']}")
    if len(by_pair) != expected_pairs:
        raise RuntimeError(f"expected {expected_pairs} unique object/seed pairs, got {len(by_pair)}")
    if len({key[0] for key in by_pair}) != 32:
        raise RuntimeError("source records do not contain exactly 32 unique objects")
    if len({str(rows[0]["pair_id"]) for rows in by_pair.values()}) != expected_pairs:
        raise RuntimeError("blind pair_id values are not unique across object/seed pairs")
    for pair_key, rows in by_pair.items():
        if len(rows) != 2:
            raise RuntimeError(f"pair {pair_key} has {len(rows)} records, expected 2")
        if {str(row["side"]) for row in rows} != {"A", "B"}:
            raise RuntimeError(f"pair {pair_key} does not contain exactly sides A/B")
        if {str(row["branch"]) for row in rows} != {"stock", "correct"}:
            raise RuntimeError(f"pair {pair_key} does not contain stock/correct")
        if len({str(row["pair_id"]) for row in rows}) != 1:
            raise RuntimeError(f"pair {pair_key} has inconsistent blind pair_id")
    return report, protocol, rollout_by_uid


def render_summary(report: dict[str, Any]) -> str:
    lines = [
        "Stage-B symmetric CPU cleanup diagnostic",
        "========================================",
        "",
        "FORMAL: false (exploratory diagnostic; does not replace B3)",
        f"diagnostic valid: {report.get('diagnostic_valid', False)}",
        f"source: {report['source']['output_dir']}",
        f"records: {report['source']['record_count']}",
        f"surface samples: {report['parameters']['surface_samples']}",
        f"minimum component vertices: {report['parameters']['min_component_vertices']}",
        f"raw/source reconciliation passed: "
        f"{report['raw_vs_source_reconciliation'].get('passed', False)}",
        "",
    ]
    for policy in POLICIES:
        block = report["policies"][policy]
        lines.extend([f"[{policy}]", f"valid pairs: {block['valid_pair_count']}"])
        for metric in (
            "chamfer_l1_improvement",
            "fscore_0p02_delta",
            "normal_consistency_delta",
            "largest_component_ratio_delta",
            "component_count_improvement",
            "small_component_vertex_ratio_improvement",
            "boundary_edge_count_improvement",
        ):
            row = block["summary"].get(metric, {})
            lines.append(
                f"{metric}: mean={row.get('mean', 0.0):+.8f} "
                f"median={row.get('median', 0.0):+.8f} "
                f"win={row.get('object_win_rate', 0.0):.6f} "
                f"CI={row.get('bootstrap_mean_95_ci', [0.0, 0.0])}"
            )
        lines.append("")
    lines.extend(
        [
            "Interpretation guardrail:",
            "Positive cleanup-policy deltas are diagnostic evidence only. The cleanup",
            "threshold was selected after the exploratory Mesh result and is not a frozen",
            "formal Stage-B gate.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if int(args.min_component_vertices) <= 0:
        raise ValueError("min_component_vertices must be positive")
    if int(args.surface_samples) <= 0:
        raise ValueError("surface_samples must be positive")
    if int(args.bootstrap_samples) <= 0:
        raise ValueError("bootstrap_samples must be positive")
    if int(args.workers) <= 0:
        raise ValueError("workers must be positive")

    source_output = Path(args.source_output).resolve()
    output_dir = Path(args.output_dir).resolve()
    if (
        output_dir == source_output
        or source_output in output_dir.parents
        or output_dir in source_output.parents
    ):
        raise ValueError("output_dir and source_output must be separate sibling trees")
    source_report, protocol, rollout_by_uid = validate_source(source_output)
    source_report_path = source_output / "report.json"
    protocol_path = source_output / "source_frozen_protocol.json"
    identity = {
        "format": REPORT_FORMAT,
        "source_output": str(source_output),
        "source_report_sha256": sha256_file(source_report_path),
        "source_protocol_sha256": sha256_file(protocol_path),
        "min_component_vertices": int(args.min_component_vertices),
        "surface_samples": int(args.surface_samples),
        "bootstrap_samples": int(args.bootstrap_samples),
        "workers": int(args.workers),
        "export_cleaned_meshes": bool(args.export_cleaned_meshes),
        "policies": list(POLICIES),
        "implementation_sha256": sha256_file(Path(__file__).resolve()),
    }
    identity_sha256 = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    identity_path = output_dir / "run_identity.json"
    if output_dir.exists() and not args.resume:
        raise FileExistsError(output_dir)
    if identity_path.exists():
        existing = json.loads(identity_path.read_text(encoding="utf-8"))
        if existing != identity:
            raise RuntimeError("resume arguments differ from existing run identity")
    output_dir.mkdir(parents=True, exist_ok=True)
    if not identity_path.exists():
        atomic_write_json(identity_path, identity)
    completion_path = output_dir / "completion_manifest.json"
    if completion_path.exists():
        previous_completion = json.loads(completion_path.read_text(encoding="utf-8"))
        atomic_write_json(
            completion_path,
            {
                "format": REPORT_FORMAT,
                "complete": False,
                "formal": False,
                "exploratory": True,
                "reason": "resume validation/reaggregation in progress",
                "previous_report_sha256": previous_completion.get("report_sha256"),
            },
        )

    thresholds = [float(value) for value in protocol["mesh"]["fscore_thresholds"]]
    primary_fscore = float(protocol["mesh"]["primary_fscore_threshold"])
    records_by_policy: dict[str, list[dict[str, Any]]] = {policy: [] for policy in POLICIES}
    source_records = list(source_report["records"])
    current_uid: str | None = None
    current_target: trimesh.Trimesh | None = None
    target_samples: dict[int, tuple[np.ndarray, np.ndarray, cKDTree]] = {}

    for record_index, source_row in enumerate(source_records):
        uid = str(source_row["uid"])
        object_uid = str(source_row["object_uid"])
        seed = int(source_row["seed"])
        pair_id = str(source_row["pair_id"])
        side = str(source_row["side"])
        branch = str(source_row["branch"])
        if side not in {"A", "B"} or branch not in {"stock", "correct"}:
            raise RuntimeError(f"invalid source side/branch: {source_row}")
        source_obj = Path(source_row["canonical_obj"]).resolve()
        try:
            source_obj.relative_to(source_output)
        except ValueError as error:
            raise RuntimeError(f"source OBJ escapes source output: {source_obj}") from error
        if not source_obj.is_file():
            raise FileNotFoundError(source_obj)
        if sha256_file(source_obj) != str(source_row["canonical_obj_sha256"]):
            raise RuntimeError(f"source OBJ hash mismatch: {source_obj}")

        if current_uid != uid:
            current_uid = uid
            current_target = load_canonical_target(source_row["target"])
            target_samples = {}
        assert current_target is not None
        rollout_position = int(rollout_by_uid[uid])
        metric_seed = seed * 1009 + rollout_position * 9173
        if metric_seed not in target_samples:
            target_samples[metric_seed] = prepare_target_sample(
                current_target,
                count=int(args.surface_samples),
                seed=metric_seed,
            )

        source_mesh = load_mesh(source_obj)
        source_components = component_partition(source_mesh)
        computed_by_signature: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
        exported_by_signature: dict[str, Path] = {}
        for policy in POLICIES:
            record_path = output_dir / "record_metrics" / policy / pair_id / f"{side}.json"
            if args.resume and record_path.is_file():
                cached = json.loads(record_path.read_text(encoding="utf-8"))
                cache_valid = bool(
                    cached.get("source_obj_sha256")
                    == str(source_row["canonical_obj_sha256"])
                    and cached.get("policy") == policy
                    and cached.get("run_identity_sha256") == identity_sha256
                    and cached.get("pair_id") == pair_id
                    and cached.get("side") == side
                    and cached.get("uid") == uid
                    and cached.get("object_uid") == object_uid
                    and int(cached.get("views", -1)) == int(source_row["views"])
                    and int(cached.get("seed", -1)) == seed
                    and cached.get("branch") == branch
                    and int(cached.get("metric_seed", -1)) == metric_seed
                    and cached.get("passed") is True
                    and int(cached.get("surface_samples", -1)) == int(args.surface_samples)
                    and int(cached.get("min_component_vertices", -1))
                    == int(args.min_component_vertices)
                    and isinstance(cached.get("surface"), dict)
                    and isinstance(cached.get("structure"), dict)
                    and {
                        "chamfer_l1",
                        "normal_consistency",
                        f"fscore_{str(float(primary_fscore)).replace('.', 'p')}",
                    }.issubset(cached.get("surface", {}))
                    and {
                        "mesh_success",
                        "component_count",
                        "largest_component_ratio",
                        "small_component_vertex_ratio_lt100",
                        "boundary_edge_count",
                        "nonmanifold_edge_count",
                        "is_watertight",
                    }.issubset(cached.get("structure", {}))
                    and numeric_tree_is_finite(cached.get("surface"))
                    and numeric_tree_is_finite(cached.get("structure"))
                )
                if cache_valid:
                    cleaned_path_value = cached.get("cleaned_obj")
                    if args.export_cleaned_meshes and policy != "raw":
                        if not cleaned_path_value:
                            raise RuntimeError(f"cached cleaned OBJ path is missing: {record_path}")
                        cleaned_path = Path(str(cleaned_path_value)).resolve()
                        expected_cleaned_path = (
                            output_dir
                            / "cleaned_blind"
                            / policy
                            / pair_id
                            / side
                            / "mesh_canonical.obj"
                        ).resolve()
                        if (
                            cleaned_path != expected_cleaned_path
                            or not cleaned_path.is_file()
                            or sha256_file(cleaned_path)
                            != str(cached.get("cleaned_obj_sha256"))
                        ):
                            raise RuntimeError(f"cached cleaned OBJ is invalid: {cleaned_path}")
                    records_by_policy[policy].append(cached)
                    continue
                raise RuntimeError(f"cached record identity/metrics are invalid: {record_path}")

            face_indices, cleanup = select_faces(
                source_mesh,
                policy=policy,
                min_component_vertices=int(args.min_component_vertices),
                components=source_components,
            )
            signature = str(cleanup["face_selection_sha256"])
            cleaned = (
                source_mesh.copy()
                if policy == "raw"
                else compact_submesh(source_mesh, face_indices)
            )
            if policy == "raw":
                # Keep raw as an actual OBJ round-trip control, including any
                # unreferenced vertices.  Do not alias it to a compacted policy.
                signature = f"raw:{signature}"
            if signature in computed_by_signature:
                surface, structure = copy.deepcopy(computed_by_signature[signature])
            else:
                surface = surface_metrics(
                    cleaned,
                    target_samples[metric_seed],
                    count=int(args.surface_samples),
                    seed=metric_seed,
                    thresholds=thresholds,
                    workers=int(args.workers),
                )
                structure = mesh_structure_metrics(
                    cleaned,
                    component_vertex_counts=cleanup["retained_component_vertex_counts"],
                )
                computed_by_signature[signature] = (copy.deepcopy(surface), copy.deepcopy(structure))

            exported_obj: str | None = None
            exported_sha: str | None = None
            if args.export_cleaned_meshes and policy != "raw":
                destination = (
                    output_dir
                    / "cleaned_blind"
                    / policy
                    / pair_id
                    / side
                    / "mesh_canonical.obj"
                )
                if signature in exported_by_signature and exported_by_signature[signature].is_file():
                    atomic_link_or_copy(exported_by_signature[signature], destination)
                else:
                    safe_export_obj(cleaned, destination)
                    exported_by_signature[signature] = destination
                exported_obj = str(destination)
                exported_sha = sha256_file(destination)

            row = {
                "pair_id": pair_id,
                "side": side,
                "uid": uid,
                "object_uid": object_uid,
                "views": int(source_row["views"]),
                "seed": seed,
                "branch": branch,
                "policy": policy,
                "formal": False,
                "exploratory": True,
                "passed": True,
                "source_obj": str(source_obj),
                "source_obj_sha256": str(source_row["canonical_obj_sha256"]),
                "surface_samples": int(args.surface_samples),
                "min_component_vertices": int(args.min_component_vertices),
                "metric_seed": int(metric_seed),
                "run_identity_sha256": identity_sha256,
                "cleanup": cleanup,
                "structure": structure,
                "surface": surface,
                "cleaned_obj": exported_obj,
                "cleaned_obj_sha256": exported_sha,
            }
            atomic_write_json(record_path, row)
            records_by_policy[policy].append(row)
        print(
            f"[cleanup_cpu] {record_index + 1}/{len(source_records)} "
            f"{uid} seed={seed} side={side}",
            flush=True,
        )

    policy_reports = {
        policy: summarize_paired(
            records_by_policy[policy],
            bootstrap_samples=int(args.bootstrap_samples),
            primary_fscore=primary_fscore,
        )
        for policy in POLICIES
    }
    for policy in POLICIES:
        cleanup_rows = [row["cleanup"] for row in records_by_policy[policy]]
        policy_reports[policy]["cleanup_counts"] = {
            "record_count": len(cleanup_rows),
            "fallback_to_largest_component_count": sum(
                bool(row["fallback_to_largest_component"]) for row in cleanup_rows
            ),
            "removed_component_count_sum": sum(
                int(row["removed_component_count"]) for row in cleanup_rows
            ),
            "retained_component_count_mean": float(
                np.mean([int(row["retained_component_count"]) for row in cleanup_rows])
            )
            if cleanup_rows
            else 0.0,
        }
    reconciliation = raw_reconciliation(
        records_by_policy["raw"],
        source_records,
        evaluated=int(args.surface_samples) == int(protocol["mesh"]["surface_samples"]),
    )
    effects = cleanup_effects(
        records_by_policy,
        policy_reports,
        bootstrap_samples=int(args.bootstrap_samples),
        primary_fscore=primary_fscore,
    )
    blind_audit = audit_cleaned_blind_tree(
        output_dir / "cleaned_blind",
        expected_files=(2 * len(source_records) if args.export_cleaned_meshes else 0),
    )
    validity_checks = {
        "source_record_count_192": len(source_records) == 192,
        "raw_source_reconciliation_passed": reconciliation.get("passed") is True,
        "cleaned_blinding_audit_passed": blind_audit.get("passed") is True,
    }
    for policy in POLICIES:
        block = policy_reports[policy]
        validity_checks[f"{policy}_record_count_192"] = (
            int(block["record_count"]) == 192
        )
        validity_checks[f"{policy}_completed_pair_count_96"] = (
            int(block["completed_pair_count"]) == 96
        )
        validity_checks[f"{policy}_valid_pair_count_96"] = (
            int(block["valid_pair_count"]) == 96
        )
        validity_checks[f"{policy}_no_invalid_pairs"] = not block["invalid_pairs"]
    diagnostic_valid = all(validity_checks.values())
    report = {
        "format": REPORT_FORMAT,
        "stage": "B4_symmetric_cpu_cleanup_diagnostic",
        "formal": False,
        "exploratory": True,
        "claim_limit": (
            "Post-hoc symmetric CPU cleanup diagnostic; it cannot satisfy or replace "
            "the preregistered formal Stage-B gate."
        ),
        "diagnostic_valid": bool(diagnostic_valid),
        "validity_checks": validity_checks,
        "source": {
            "output_dir": str(source_output),
            "report": str(source_report_path),
            "report_sha256": identity["source_report_sha256"],
            "protocol": str(protocol_path),
            "protocol_file_sha256": identity["source_protocol_sha256"],
            "protocol_sha256": protocol["protocol_sha256"],
            "record_count": len(source_records),
        },
        "parameters": {
            "policies": list(POLICIES),
            "min_component_vertices": int(args.min_component_vertices),
            "surface_samples": int(args.surface_samples),
            "bootstrap_samples": int(args.bootstrap_samples),
            "workers": int(args.workers),
            "fscore_thresholds": thresholds,
            "primary_fscore_threshold": primary_fscore,
            "symmetry_rule": (
                "Each policy, threshold, canonical frame, surface sample count, metric "
                "seed formula, and metric implementation is identical for stock/correct."
            ),
            "empty_cleanup_fallback": (
                "If threshold cleanup removes every component, retain the deterministic "
                "largest component on either branch."
            ),
            "export_cleaned_meshes": bool(args.export_cleaned_meshes),
        },
        "raw_vs_source_reconciliation": reconciliation,
        "cleaned_blinding_audit": blind_audit,
        "policies": policy_reports,
        "cleanup_effects_vs_raw": effects,
        "records": records_by_policy,
    }
    atomic_write_json(output_dir / "report.json", report)
    atomic_write_text(output_dir / "summary.txt", render_summary(report))
    runtime_exit_code = 0 if diagnostic_valid else 2
    artifact_rows = []
    for artifact in sorted(path for path in output_dir.rglob("*") if path.is_file()):
        if artifact == completion_path or ".tmp." in artifact.name:
            continue
        artifact_rows.append(
            {
                "path": artifact.relative_to(output_dir).as_posix(),
                "sha256": sha256_file(artifact),
            }
        )
    atomic_write_json(
        completion_path,
        {
            "format": REPORT_FORMAT,
            "complete": True,
            "formal": False,
            "exploratory": True,
            "diagnostic_valid": bool(diagnostic_valid),
            "validity_checks": validity_checks,
            "runtime_exit_code": int(runtime_exit_code),
            "source_report_sha256": identity["source_report_sha256"],
            "report_sha256": sha256_file(output_dir / "report.json"),
            "summary_sha256": sha256_file(output_dir / "summary.txt"),
            "record_count_by_policy": {
                policy: len(records_by_policy[policy]) for policy in POLICIES
            },
            "file_count": len(artifact_rows),
            "files": artifact_rows,
        },
    )
    print(render_summary(report), flush=True)
    raise SystemExit(runtime_exit_code)


if __name__ == "__main__":
    main()
