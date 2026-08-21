#!/usr/bin/env python3
"""Prepare and score the frozen Omni200 benchmark with SS30K+SLat30K only.

This module deliberately has no ReconViaGen inference branch.  The rendered
camera matrices already live in the official-compatible model-O frame, so the
runtime cache binds that frame directly instead of estimating O from masks.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
from typing import Any

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
import trimesh

from ar_ss_flow.shared_object_preprocessing import (
    prepare_shared_object_arrays,
    transform_intrinsics,
)
from manual_mesh_reconstruction.canonicalization import array_sha256
from manual_mesh_reconstruction.common import (
    atomic_json,
    atomic_npz,
    canonical_sha256,
    load_json,
    sha256_file,
)
from manual_mesh_reconstruction.model_geometry import (
    normalize_similarity_extrinsics,
)
from manual_mesh_reconstruction.runtime_o import (
    MANIFEST_FORMAT as RUNTIME_MANIFEST_FORMAT,
    OBJECT_FORMAT as RUNTIME_OBJECT_FORMAT,
)
from pose_point_depth_mv.dataset_tools.build_objaverse_multiview_sparse_data import (
    load_meshes,
)
from pose_point_depth_mv.dataset_tools.build_reconviagen_omni200_benchmark import (
    FINAL_FORMAT as BENCHMARK_MANIFEST_FORMAT,
)
from pose_point_depth_mv.mesh_benchmark_metrics import deterministic_surface_sample


PREPARE_FORMAT = "reconviagen.omniobject3d_omni200_exact_model_o_runtime.v1"
METRIC_OBJECT_FORMAT = "reconviagen.omniobject3d_omni200_ss30k_slat30k_metric.v1"
METRIC_WORKER_FORMAT = (
    "reconviagen.omniobject3d_omni200_ss30k_slat30k_metric_worker.v1"
)
METRIC_AGGREGATE_FORMAT = (
    "reconviagen.omniobject3d_omni200_ss30k_slat30k_metric_aggregate.v1"
)
CURRENT_INFERENCE_FORMAT = (
    "pose_point_depth_mv.real_proobjaverse_official_ss_slat_inference_manifest.v1"
)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _atomic_png(path: Path, image: Image.Image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp-{os.getpid()}.png")
    image.save(temporary, format="PNG")
    os.replace(temporary, path)


def _benchmark(path: Path, *, expected_objects: int | None = None) -> dict[str, Any]:
    payload = load_json(path)
    if payload.get("format") != BENCHMARK_MANIFEST_FORMAT:
        raise RuntimeError(f"Omni200 benchmark manifest format differs: {path}")
    rows = list(payload.get("objects") or [])
    saved_identity = str(payload.get("manifest_identity", ""))
    identity_payload = dict(payload)
    identity_payload.pop("manifest_identity", None)
    if not saved_identity or canonical_sha256(identity_payload) != saved_identity:
        raise RuntimeError(f"Omni200 benchmark manifest identity differs: {path}")
    if expected_objects is not None and len(rows) != int(expected_objects):
        raise RuntimeError(
            f"Omni200 object count differs: {len(rows)} != {expected_objects}"
        )
    if len({str(row["uid"]) for row in rows}) != len(rows):
        raise RuntimeError("Omni200 benchmark contains duplicate UIDs")
    return payload


def _prepare_one(
    row: dict[str, Any],
    *,
    benchmark_sha256: str,
    build_config_sha256: str,
    output_dir: Path,
    resume: bool,
) -> tuple[dict[str, Any], bool]:
    category = str(row["category"])
    object_id = str(row["uid"])
    destination = output_dir / "objects" / category / object_id
    report_path = destination / "report.json"
    marker_path = destination / "_EXACT_MODEL_O_RUNTIME_COMPLETE.json"
    if report_path.is_file() and marker_path.is_file():
        report = load_json(report_path)
        marker = load_json(marker_path)
        if (
            report.get("format") != RUNTIME_OBJECT_FORMAT
            or report.get("passed") is not True
            or marker.get("format") != PREPARE_FORMAT
            or marker.get("benchmark_manifest_sha256") != benchmark_sha256
            or marker.get("build_config_sha256") != build_config_sha256
        ):
            raise RuntimeError(f"stale exact-model-O runtime object: {destination}")
        cache = Path(str(report["cache_npz"]))
        if not cache.is_file() or sha256_file(cache) != report["cache_npz_sha256"]:
            raise RuntimeError(f"runtime cache binding differs: {cache}")
        return report, True
    if destination.exists():
        if not resume:
            raise RuntimeError(f"partial exact-model-O runtime object: {destination}")
        shutil.rmtree(destination)

    rgba_paths = [Path(value).resolve(strict=True) for value in row["rgba_images"]]
    mask_paths = [Path(value).resolve(strict=True) for value in row["mask_images"]]
    object_report_path = Path(row["object_report"]).resolve(strict=True)
    if sha256_file(object_report_path) != str(row["object_report_sha256"]):
        raise RuntimeError(f"{object_id} frozen render report SHA256 differs")
    object_report = load_json(object_report_path)
    if (
        object_report.get("passed") is not True
        or object_report.get("uid") != object_id
        or object_report.get("source_scan_tree_sha256")
        != row["source_scan_tree_sha256"]
    ):
        raise RuntimeError(f"{object_id} frozen render report identity differs")
    frozen_rgba = [Path(item["rgba"]["path"]).resolve() for item in object_report["rendered_files"]]
    frozen_masks = [Path(item["mask"]["path"]).resolve() for item in object_report["rendered_files"]]
    if rgba_paths != frozen_rgba or mask_paths != frozen_masks:
        raise RuntimeError(f"{object_id} frozen render paths differ")
    for item in object_report["rendered_files"]:
        for label in ("rgba", "mask"):
            bound = Path(item[label]["path"]).resolve(strict=True)
            if sha256_file(bound) != str(item[label]["sha256"]):
                raise RuntimeError(f"{object_id} {label} SHA256 differs: {bound}")
    c2w = np.asarray(row["c2w_opencv_model_o"], dtype=np.float64)
    intrinsic = np.asarray(row["intrinsic"], dtype=np.float64)
    if len(rgba_paths) != 4 or len(mask_paths) != 4:
        raise RuntimeError(f"{object_id} does not bind exactly four input views")
    if c2w.shape != (4, 4, 4) or intrinsic.shape not in {(3, 3), (4, 3, 3)}:
        raise RuntimeError(f"{object_id} camera contract differs")
    if not np.isfinite(c2w).all() or not np.isfinite(intrinsic).all():
        raise RuntimeError(f"{object_id} camera contains non-finite values")
    rotations = c2w[:, :3, :3]
    if not np.allclose(
        np.matmul(np.swapaxes(rotations, 1, 2), rotations),
        np.eye(3)[None],
        atol=1.0e-6,
    ) or not np.all(np.linalg.det(rotations) > 0.999999):
        raise RuntimeError(f"{object_id} c2w is not a proper rigid camera")

    images: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    for rgba_path, mask_path in zip(rgba_paths, mask_paths):
        with Image.open(rgba_path) as handle:
            images.append(np.asarray(handle.convert("RGB"), dtype=np.uint8))
        with Image.open(mask_path) as handle:
            masks.append(np.asarray(handle.convert("L"), dtype=np.uint8))
    prepared = prepare_shared_object_arrays(
        images,
        masks,
        resolution=518,
        foreground_margin=1.10,
        alpha_threshold=0.80,
    )
    k_source = (
        np.repeat(intrinsic[None], 4, axis=0)
        if intrinsic.shape == (3, 3)
        else intrinsic
    )
    k_feature = transform_intrinsics(
        k_source.astype(np.float32), prepared.source_to_feature_affines
    )
    T_O2C = np.linalg.inv(c2w)
    T_O2C_lifting = normalize_similarity_extrinsics(T_O2C).astype(np.float64)
    if not np.allclose(T_O2C, T_O2C_lifting, rtol=1.0e-9, atol=1.0e-10):
        raise RuntimeError(f"{object_id} rigid camera changed during lifting normalization")

    staging = destination.parent / f".{object_id}.exact-model-o-building"
    if staging.exists():
        if not resume:
            raise RuntimeError(f"partial staging exists: {staging}")
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    view_dir = staging / "views"
    view_dir.mkdir()
    rgb_names: list[str] = []
    mask_names: list[str] = []
    for index, (image, mask) in enumerate(
        zip(prepared.images, prepared.masks)
    ):
        rgb_name = f"view_{index:02d}_rgb.png"
        mask_name = f"view_{index:02d}_mask.png"
        _atomic_png(view_dir / rgb_name, image)
        _atomic_png(
            view_dir / mask_name,
            Image.fromarray(
                np.clip(np.rint(mask * 255.0), 0, 255).astype(np.uint8), mode="L"
            ),
        )
        rgb_names.append(rgb_name)
        mask_names.append(mask_name)

    cache_path = staging / "runtime_input_cache.npz"
    atomic_npz(
        cache_path,
        selected_source_view_index=np.arange(4, dtype=np.int64),
        frame_name=np.asarray(
            [Path(value).stem for value in row["rgba_images"]], dtype="U128"
        ),
        K_feature=k_feature.astype(np.float32),
        T_O2C=T_O2C.astype(np.float64),
        T_O2C_lifting=T_O2C_lifting,
        T_C2O=c2w.astype(np.float64),
        T_O2W=np.eye(4, dtype=np.float64),
        T_W2O=np.eye(4, dtype=np.float64),
        P_O=np.empty((0, 3), dtype=np.float32),
        object_point_source_index=np.empty((0,), dtype=np.int64),
        source_to_feature_affine=prepared.source_to_feature_affines.astype(np.float32),
    )
    prepared_rgb = np.stack(
        [np.asarray(image, dtype=np.uint8) for image in prepared.images]
    )
    condition = {
        "format": PREPARE_FORMAT,
        "benchmark_manifest_sha256": benchmark_sha256,
        "uid": object_id,
        "selected_input_view_indices": list(row["selected_input_view_indices"]),
        "shared_image_geometry": prepared.geometry_record(),
        "prepared_rgb_sha256": array_sha256(prepared_rgb),
        "prepared_mask_sha256": array_sha256(prepared.masks),
        "K_feature_sha256": array_sha256(k_feature),
        "T_O2C_sha256": array_sha256(T_O2C),
        "T_O2C_lifting_sha256": array_sha256(T_O2C_lifting),
        "object_frame": "exact frozen model-O from synthetic renderer",
        "model_o_up_axis": "+Z",
        "point_cloud_consumed": False,
        "target_or_gt_mesh_consumed": False,
    }
    condition["condition_sha256"] = canonical_sha256(condition)
    atomic_json(staging / "condition_record.json", condition)

    final_view_dir = destination / "views"
    final_cache = destination / cache_path.name
    report = {
        "format": RUNTIME_OBJECT_FORMAT,
        "created_at_utc": utc_now(),
        "category": category,
        "object_id": object_id,
        "object_key": f"{category}:{object_id}",
        "benchmark_manifest_sha256": benchmark_sha256,
        "build_config_sha256": build_config_sha256,
        "geometry_mode": "exact_frozen_model_o",
        "point_cloud_consumed": False,
        "selected_view_count": 4,
        "all_input_view_count": 4,
        "o_frozen_before_view_selection": True,
        "selected_source_view_indices": list(range(4)),
        "selected_frame_names": [Path(value).stem for value in row["rgba_images"]],
        "reference_view_index": 0,
        "cache_npz": str(final_cache),
        "cache_npz_sha256": sha256_file(cache_path),
        "condition_record": str(destination / "condition_record.json"),
        "condition_sha256": condition["condition_sha256"],
        "T_O2C_sha256": array_sha256(T_O2C),
        "T_O2C_lifting_sha256": array_sha256(T_O2C_lifting),
        "prepared_rgb_paths": [str(final_view_dir / name) for name in rgb_names],
        "prepared_mask_paths": [str(final_view_dir / name) for name in mask_names],
        "runtime_frame_stats": {
            "axes": {"policy": "exact_renderer_model_o", "up_axis": "+Z"},
            "scale_O2W": 1.0,
            "center_W": [0.0, 0.0, 0.0],
            "P_O_shape": [0, 3],
        },
        "formal_input_passed": True,
        "forbidden_gt_fields_absent": True,
        "training_ready": False,
        "scope_guard": (
            "Four frozen synthetic observations and exact model-O cameras only; "
            "GT Mesh is not read by the model-input path."
        ),
        "passed": True,
    }
    atomic_json(staging / "report.json", report)
    atomic_json(
        staging / "_EXACT_MODEL_O_RUNTIME_COMPLETE.json",
        {
            "format": PREPARE_FORMAT,
            "benchmark_manifest_sha256": benchmark_sha256,
            "build_config_sha256": build_config_sha256,
            "condition_sha256": condition["condition_sha256"],
            "runtime_cache_sha256": report["cache_npz_sha256"],
            "passed": True,
        },
    )
    staging.replace(destination)
    return report, False


def cmd_prepare(args: argparse.Namespace) -> None:
    manifest_path = Path(args.benchmark_manifest).expanduser().resolve(strict=True)
    benchmark = _benchmark(manifest_path, expected_objects=int(args.expected_objects))
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    benchmark_sha256 = sha256_file(manifest_path)
    config = {
        "format": PREPARE_FORMAT,
        "benchmark_manifest": str(manifest_path),
        "benchmark_manifest_sha256": benchmark_sha256,
        "feature_resolution": 518,
        "foreground_margin": 1.10,
        "alpha_threshold": 0.80,
        "object_frame": "exact frozen renderer model-O",
        "model_o_up_axis": "+Z",
        "selected_view_count": 4,
    }
    build_config_sha256 = canonical_sha256(config)
    reports = []
    reused = 0
    for index, row in enumerate(benchmark["objects"], 1):
        report, was_reused = _prepare_one(
            row,
            benchmark_sha256=benchmark_sha256,
            build_config_sha256=build_config_sha256,
            output_dir=output_dir,
            resume=bool(args.resume),
        )
        reports.append(report)
        reused += int(was_reused)
        print(
            f"[omni200:runtime] {index}/{len(benchmark['objects'])} "
            f"uid={row['uid']} reused={was_reused}",
            flush=True,
        )
    manifest = {
        "format": RUNTIME_MANIFEST_FORMAT,
        "prepare_format": PREPARE_FORMAT,
        "passed": len(reports) == int(args.expected_objects),
        "benchmark_manifest": str(manifest_path),
        "benchmark_manifest_sha256": benchmark_sha256,
        "protocol_sha256": benchmark["protocol_sha256"],
        "build_config": config,
        "build_config_sha256": build_config_sha256,
        "selected_object_count": len(reports),
        "completed_object_count": len(reports),
        "reused_object_count": reused,
        "objects": reports,
        "failures": [],
        "training_ready": False,
        "scope_guard": "Exact renderer model-O input preparation; no GT metric read.",
    }
    atomic_json(output_dir / "runtime_input_manifest.json", manifest)
    if not manifest["passed"]:
        raise SystemExit(2)
    print(json.dumps({"passed": True, "objects": len(reports)}, indent=2))


def _one_mesh(value: str | Path) -> trimesh.Trimesh:
    loaded = trimesh.load(str(value), force="scene", process=False)
    if isinstance(loaded, trimesh.Scene):
        meshes = [
            item
            for item in loaded.dump(concatenate=False)
            if isinstance(item, trimesh.Trimesh) and len(item.faces)
        ]
        if not meshes:
            raise RuntimeError(f"mesh scene is empty: {value}")
        return trimesh.util.concatenate(meshes)
    if not isinstance(loaded, trimesh.Trimesh) or not len(loaded.faces):
        raise RuntimeError(f"mesh is empty: {value}")
    return loaded


def _gt_mesh(row: dict[str, Any]) -> trimesh.Trimesh:
    meshes = load_meshes(str(row["source_mesh"]))
    mesh = trimesh.util.concatenate([item.copy() for item in meshes])
    transform = np.asarray(row["source_to_model_o_4x4"], dtype=np.float64)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise RuntimeError(f"invalid GT source-to-model-O transform: {row['uid']}")
    mesh.apply_transform(transform)
    return mesh


def _normalize_points(points: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    value = np.asarray(points, dtype=np.float64)
    low = value.min(axis=0)
    high = value.max(axis=0)
    center = 0.5 * (low + high)
    max_extent = float(np.max(high - low))
    if not math.isfinite(max_extent) or max_extent <= 1.0e-12:
        raise RuntimeError("surface sample has invalid extent")
    normalized = (value - center[None]) * (2.0 / max_extent)
    return normalized, {
        "sample_bounds_low": low.tolist(),
        "sample_bounds_high": high.tolist(),
        "sample_bounds_center": center.tolist(),
        "sample_max_extent": max_extent,
        "normalization": "independent AABB center/max-extent to [-1,1]",
    }


def _paper_metrics(
    predicted: trimesh.Trimesh,
    target: trimesh.Trimesh,
    *,
    count: int,
    seed: int,
    radius: float,
) -> tuple[dict[str, float], dict[str, Any]]:
    predicted_points, _ = deterministic_surface_sample(predicted, count, seed)
    target_points, _ = deterministic_surface_sample(target, count, seed + 1)
    predicted_points, predicted_norm = _normalize_points(predicted_points)
    target_points, target_norm = _normalize_points(target_points)
    target_tree = cKDTree(target_points)
    predicted_tree = cKDTree(predicted_points)
    pred_to_gt = target_tree.query(predicted_points, k=1, workers=-1)[0]
    gt_to_pred = predicted_tree.query(target_points, k=1, workers=-1)[0]
    precision = float(np.mean(pred_to_gt < float(radius)))
    recall = float(np.mean(gt_to_pred < float(radius)))
    fscore = (
        0.0
        if precision + recall <= 1.0e-12
        else float(2.0 * precision * recall / (precision + recall))
    )
    symmetric_mean = float(0.5 * (pred_to_gt.mean() + gt_to_pred.mean()))
    return {
        "chamfer_distance": symmetric_mean,
        "chamfer_distance_symmetric_sum": float(2.0 * symmetric_mean),
        "pred_to_gt_mean": float(pred_to_gt.mean()),
        "gt_to_pred_mean": float(gt_to_pred.mean()),
        "fscore": fscore,
        "precision": precision,
        "recall": recall,
    }, {"predicted": predicted_norm, "target": target_norm}


def cmd_metric_worker(args: argparse.Namespace) -> None:
    benchmark_path = Path(args.benchmark_manifest).expanduser().resolve(strict=True)
    inference_path = Path(args.inference_manifest).expanduser().resolve(strict=True)
    benchmark = _benchmark(benchmark_path)
    inference = load_json(inference_path)
    if (
        inference.get("format") != CURRENT_INFERENCE_FORMAT
        or inference.get("passed") is not True
    ):
        raise RuntimeError(f"current SS30K+SLat30K inference did not pass: {inference_path}")
    benchmark_by_uid = {str(row["uid"]): row for row in benchmark["objects"]}
    records = list(inference.get("objects") or [])
    if not records:
        raise RuntimeError("current inference manifest has no Mesh records")
    if any(int(row.get("seed", -1)) != int(args.seed) for row in records):
        raise RuntimeError("current inference seed differs from metric seed")
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for index, record in enumerate(records, 1):
        uid = str(record["object_id"])
        if uid not in benchmark_by_uid:
            raise RuntimeError(f"inference UID is absent from benchmark: {uid}")
        source = benchmark_by_uid[uid]
        mesh_path = Path(record["mesh"]).resolve(strict=True)
        destination = output_dir / "objects" / uid / "metric.json"
        if destination.is_file():
            cached = load_json(destination)
            if (
                cached.get("format") == METRIC_OBJECT_FORMAT
                and cached.get("passed") is True
                and cached.get("predicted_mesh_sha256") == sha256_file(mesh_path)
                and cached.get("benchmark_manifest_sha256") == sha256_file(benchmark_path)
                and int(cached.get("surface_points", -1)) == int(args.surface_points)
                and float(cached.get("fscore_radius", -1.0)) == float(args.fscore_radius)
            ):
                results.append(cached)
                print(f"[omni200:metric] {index}/{len(records)} reused uid={uid}", flush=True)
                continue
            raise RuntimeError(f"stale metric result: {destination}")
        predicted = _one_mesh(mesh_path)
        target = _gt_mesh(source)
        uid_seed = int.from_bytes(
            hashlib.sha256(uid.encode("utf-8")).digest()[:8], "big"
        )
        metric_seed = (int(args.seed) * 1_000_003 + uid_seed) % (2**63 - 1)
        metrics, normalization = _paper_metrics(
            predicted,
            target,
            count=int(args.surface_points),
            seed=metric_seed,
            radius=float(args.fscore_radius),
        )
        result = {
            "format": METRIC_OBJECT_FORMAT,
            "passed": True,
            "uid": uid,
            "category": source["category"],
            "seed": int(args.seed),
            "benchmark_manifest": str(benchmark_path),
            "benchmark_manifest_sha256": sha256_file(benchmark_path),
            "inference_manifest": str(inference_path),
            "inference_manifest_sha256": sha256_file(inference_path),
            "predicted_mesh": str(mesh_path),
            "predicted_mesh_sha256": sha256_file(mesh_path),
            "gt_mesh": str(Path(source["source_mesh"]).resolve(strict=True)),
            "gt_scan_tree_sha256": source["source_scan_tree_sha256"],
            "surface_points": int(args.surface_points),
            "fscore_radius": float(args.fscore_radius),
            "metric_seed": metric_seed,
            "normalization": normalization,
            "metrics": metrics,
            "alignment": "none; only independent metric point-cloud normalization",
            "reconviagen_model_loaded_or_run": False,
        }
        atomic_json(destination, result)
        results.append(result)
        print(
            f"[omni200:metric] {index}/{len(records)} uid={uid} "
            f"CD={metrics['chamfer_distance']:.8f} F={metrics['fscore']:.8f}",
            flush=True,
        )
    worker_report = {
        "format": METRIC_WORKER_FORMAT,
        "passed": len(results) == len(records),
        "worker_index": int(args.worker_index),
        "benchmark_manifest": str(benchmark_path),
        "benchmark_manifest_sha256": sha256_file(benchmark_path),
        "inference_manifest": str(inference_path),
        "inference_manifest_sha256": sha256_file(inference_path),
        "object_count": len(results),
        "objects": results,
        "reconviagen_model_loaded_or_run": False,
    }
    atomic_json(output_dir / "metrics_report.json", worker_report)
    if not worker_report["passed"]:
        raise SystemExit(2)


def _summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(len(array)),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def cmd_aggregate(args: argparse.Namespace) -> None:
    benchmark_path = Path(args.benchmark_manifest).expanduser().resolve(strict=True)
    benchmark = _benchmark(benchmark_path, expected_objects=int(args.expected_objects))
    workers_root = Path(args.workers_root).expanduser().resolve(strict=True)
    reports = sorted(workers_root.glob("worker_*/03_metrics/metrics_report.json"))
    if len(reports) != int(args.expected_workers):
        raise RuntimeError(
            f"metric worker report count differs: {len(reports)} != {args.expected_workers}"
        )
    rows = []
    for path in reports:
        report = load_json(path)
        if report.get("format") != METRIC_WORKER_FORMAT or report.get("passed") is not True:
            raise RuntimeError(f"metric worker did not pass: {path}")
        if report.get("benchmark_manifest_sha256") != sha256_file(benchmark_path):
            raise RuntimeError(f"metric worker benchmark identity differs: {path}")
        rows.extend(report["objects"])
    uids = [str(row["uid"]) for row in rows]
    expected_uids = {str(row["uid"]) for row in benchmark["objects"]}
    if len(rows) != int(args.expected_objects) or set(uids) != expected_uids or len(set(uids)) != len(uids):
        raise RuntimeError("metric workers do not exactly cover Omni200")
    by_category: dict[str, dict[str, Any]] = {}
    for category in sorted({str(row["category"]) for row in rows}):
        selected = [row for row in rows if row["category"] == category]
        by_category[category] = {
            "object_count": len(selected),
            "chamfer_distance": _summary(
                [float(row["metrics"]["chamfer_distance"]) for row in selected]
            ),
            "fscore": _summary([float(row["metrics"]["fscore"]) for row in selected]),
        }
    aggregate = {
        "format": METRIC_AGGREGATE_FORMAT,
        "passed": True,
        "method": "no-VGGT SS30K step30000 + no-VGGT SLat30K step30000",
        "benchmark_manifest": str(benchmark_path),
        "benchmark_manifest_sha256": sha256_file(benchmark_path),
        "protocol_sha256": benchmark["protocol_sha256"],
        "object_count": len(rows),
        "category_count": len(by_category),
        "surface_points_per_mesh": int(args.surface_points),
        "fscore_radius": float(args.fscore_radius),
        "chamfer_distance": _summary(
            [float(row["metrics"]["chamfer_distance"]) for row in rows]
        ),
        "fscore": _summary([float(row["metrics"]["fscore"]) for row in rows]),
        "by_category": by_category,
        "objects": rows,
        "worker_reports": [
            {"path": str(path), "sha256": sha256_file(path)} for path in reports
        ],
        "reconviagen_model_loaded_or_run": False,
        "scope_guard": (
            "SS30K+SLat30K only. ReconViaGen original is neither loaded nor run. "
            "Object metrics use 100k surface samples and F-score radius 0.1."
        ),
    }
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(output_dir / "report.json", aggregate)
    summary = [
        "OmniObject3D 200-object SS30K+SLat30K evaluation",
        "=================================================",
        f"objects: {len(rows)} categories: {len(by_category)}",
        f"Chamfer Distance: mean={aggregate['chamfer_distance']['mean']:.8f} median={aggregate['chamfer_distance']['median']:.8f}",
        f"F-score@0.1: mean={aggregate['fscore']['mean']:.8f} median={aggregate['fscore']['median']:.8f}",
        "ReconViaGen original loaded/run: no",
        f"report: {output_dir / 'report.json'}",
    ]
    (output_dir / "summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print("\n".join(summary))


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--benchmark_manifest", required=True)
    prepare.add_argument("--output_dir", required=True)
    prepare.add_argument("--expected_objects", type=int, default=200)
    prepare.add_argument("--resume", action="store_true")
    prepare.set_defaults(func=cmd_prepare)

    metric = sub.add_parser("metric-worker")
    metric.add_argument("--benchmark_manifest", required=True)
    metric.add_argument("--inference_manifest", required=True)
    metric.add_argument("--output_dir", required=True)
    metric.add_argument("--worker_index", type=int, required=True)
    metric.add_argument("--surface_points", type=int, default=100000)
    metric.add_argument("--fscore_radius", type=float, default=0.1)
    metric.add_argument("--seed", type=int, default=42)
    metric.set_defaults(func=cmd_metric_worker)

    aggregate = sub.add_parser("aggregate")
    aggregate.add_argument("--benchmark_manifest", required=True)
    aggregate.add_argument("--workers_root", required=True)
    aggregate.add_argument("--output_dir", required=True)
    aggregate.add_argument("--expected_workers", type=int, required=True)
    aggregate.add_argument("--expected_objects", type=int, default=200)
    aggregate.add_argument("--surface_points", type=int, default=100000)
    aggregate.add_argument("--fscore_radius", type=float, default=0.1)
    aggregate.set_defaults(func=cmd_aggregate)
    return parser


def main() -> None:
    args = make_parser().parse_args()
    if hasattr(args, "surface_points") and int(args.surface_points) <= 0:
        raise ValueError("surface_points must be positive")
    if hasattr(args, "fscore_radius") and float(args.fscore_radius) <= 0.0:
        raise ValueError("fscore_radius must be positive")
    args.func(args)


if __name__ == "__main__":
    main()
