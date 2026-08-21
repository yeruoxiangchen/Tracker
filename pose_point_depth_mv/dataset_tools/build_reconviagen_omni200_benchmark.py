#!/usr/bin/env python3
"""Freeze and render a ReconViaGen-style OmniObject3D 200/20 benchmark.

This is deliberately independent from the historical mixed-training renderer.
It binds 20 categories x 10 real-scanned objects, registers 24 Z-up cameras,
selects four cameras per object with a deterministic hash lottery, and renders
only those four inputs.  No DINO, SS target, training cache, COLMAP, or
Pose+Mask runtime-O is constructed here.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from pose_point_depth_mv.dataset_tools.build_objaverse_multiview_sparse_data import (
    load_meshes,
    normalize_vertices,
    pixal3d_render_space,
    render_mesh_blender,
)


FORMAT = "reconviagen.omniobject3d_omni200_20cat_render4.v1"
OBJECT_FORMAT = "reconviagen.omniobject3d_omni200_object_render4.v1"
FINAL_FORMAT = "reconviagen.omniobject3d_omni200_render_manifest.v1"
SOURCE_TO_MODEL_O = np.asarray(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float64,
)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def seeded_order(values: list[str], namespace: str, seed: int) -> list[str]:
    return sorted(
        values,
        key=lambda value: hashlib.sha256(
            f"{seed}:{namespace}:{value}".encode("utf-8")
        ).hexdigest(),
    )


def selected_view_indices(uid: str, seed: int, count: int = 4) -> list[int]:
    ordered = seeded_order(
        [str(index) for index in range(24)],
        f"input_views:{uid}",
        seed,
    )
    return sorted(int(value) for value in ordered[:count])


def look_at_z_up(eye: np.ndarray) -> np.ndarray:
    target = np.zeros(3, dtype=np.float64)
    forward = target - eye
    forward /= np.linalg.norm(forward)
    up = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    right = np.cross(up, forward)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    down /= np.linalg.norm(down)
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, 0] = right
    c2w[:3, 1] = down
    c2w[:3, 2] = forward
    c2w[:3, 3] = eye
    return c2w


def camera_plan(
    *,
    image_size: int,
    focal_ratio: float,
    radius: float,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    focal = float(image_size) * float(focal_ratio)
    intrinsic = np.asarray(
        [
            [focal, 0.0, (image_size - 1) * 0.5],
            [0.0, focal, (image_size - 1) * 0.5],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    rows: list[dict[str, Any]] = []
    view_index = 0
    # Registered interpretation of "24 views at different elevations":
    # three elevation rings and eight uniformly spaced azimuths per ring.
    for elevation_deg in (-20.0, 10.0, 40.0):
        elevation = math.radians(elevation_deg)
        for azimuth_index in range(8):
            azimuth_deg = float(45 * azimuth_index)
            azimuth = math.radians(azimuth_deg)
            eye = np.asarray(
                [
                    radius * math.cos(elevation) * math.sin(azimuth),
                    -radius * math.cos(elevation) * math.cos(azimuth),
                    radius * math.sin(elevation),
                ],
                dtype=np.float64,
            )
            c2w = look_at_z_up(eye)
            rows.append(
                {
                    "view_index": view_index,
                    "azimuth_deg": azimuth_deg,
                    "elevation_deg": elevation_deg,
                    "radius": float(radius),
                    "c2w_opencv_model_o": c2w.tolist(),
                }
            )
            view_index += 1
    assert len(rows) == 24
    return intrinsic, rows


def scan_tree_inventory(scan_dir: Path) -> tuple[list[dict[str, Any]], str, int]:
    files = sorted(path for path in scan_dir.rglob("*") if path.is_file())
    if not files:
        raise RuntimeError(f"empty Scan directory: {scan_dir}")
    rows: list[dict[str, Any]] = []
    total = 0
    for path in files:
        size = path.stat().st_size
        total += size
        rows.append(
            {
                "relative_path": path.relative_to(scan_dir).as_posix(),
                "bytes": int(size),
                "sha256": sha256_file(path),
            }
        )
    return rows, canonical_sha256(rows), total


def git_identity() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    def run(*args: str) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=root,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return result.stdout.strip()
    dirty = run("status", "--short", "--untracked-files=all")
    return {
        "root": str(root),
        "branch": run("branch", "--show-current"),
        "head": run("rev-parse", "HEAD"),
        "dirty": bool(dirty),
        "dirty_status_sha256": hashlib.sha256(dirty.encode("utf-8")).hexdigest(),
    }


def eligible_objects(categories_root: Path) -> dict[str, list[Path]]:
    result: dict[str, list[Path]] = {}
    for category_dir in sorted(path for path in categories_root.iterdir() if path.is_dir()):
        objects: list[Path] = []
        for object_dir in sorted(path for path in category_dir.iterdir() if path.is_dir()):
            scan = object_dir / "Scan"
            obj = scan / "Scan.obj"
            mtl = scan / "Scan.mtl"
            texture = scan / "Scan.jpg"
            if all(path.is_file() and path.stat().st_size > 0 for path in (obj, mtl, texture)):
                objects.append(object_dir)
        if len(objects) >= 10:
            result[category_dir.name] = objects
    return result


def protocol_without_hash(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if key != "protocol_sha256"}


def validate_protocol(payload: dict[str, Any]) -> None:
    if payload.get("format") != FORMAT:
        raise RuntimeError(f"protocol format differs: {payload.get('format')}")
    expected = canonical_sha256(protocol_without_hash(payload))
    if payload.get("protocol_sha256") != expected:
        raise RuntimeError(
            f"protocol hash differs: expected={expected} actual={payload.get('protocol_sha256')}"
        )
    rows = payload.get("objects")
    if not isinstance(rows, list) or len(rows) != int(payload["object_count"]):
        raise RuntimeError("protocol object matrix differs")
    categories = {str(row["category"]) for row in rows}
    if len(categories) != int(payload["category_count"]):
        raise RuntimeError("protocol category coverage differs")
    counts = {category: 0 for category in categories}
    for row in rows:
        counts[str(row["category"])] += 1
        selected = row.get("selected_input_view_indices")
        if not isinstance(selected, list) or len(selected) != 4 or len(set(selected)) != 4:
            raise RuntimeError(f"selected view matrix differs: {row.get('uid')}")
    if set(counts.values()) != {int(payload["objects_per_category"])}:
        raise RuntimeError(f"per-category object counts differ: {counts}")


def freeze(args: argparse.Namespace) -> None:
    output_root = Path(args.output_root).expanduser().resolve()
    protocol_path = output_root / "protocol.json"
    if protocol_path.exists():
        payload = json.loads(protocol_path.read_text(encoding="utf-8"))
        validate_protocol(payload)
        expected = {
            "selection_seed": int(args.seed),
            "category_count": int(args.category_count),
            "objects_per_category": int(args.objects_per_category),
            "image_size": int(args.image_size),
        }
        for key, value in expected.items():
            if payload.get(key) != value:
                raise RuntimeError(f"existing protocol {key} differs: {payload.get(key)} != {value}")
        print(json.dumps({"passed": True, "reused": True, "protocol": str(protocol_path), "protocol_sha256": payload["protocol_sha256"]}))
        return

    categories_root = Path(args.categories_root).expanduser().resolve()
    if not categories_root.is_dir():
        raise FileNotFoundError(categories_root)
    eligible = eligible_objects(categories_root)
    if len(eligible) < int(args.category_count):
        raise RuntimeError(f"only {len(eligible)} categories have >=10 complete scans")
    chosen_categories = seeded_order(list(eligible), "categories", int(args.seed))[
        : int(args.category_count)
    ]
    intrinsic, cameras = camera_plan(
        image_size=int(args.image_size),
        focal_ratio=float(args.focal_ratio),
        radius=float(args.camera_radius),
    )
    objects: list[dict[str, Any]] = []
    for category in sorted(chosen_categories):
        choices = seeded_order(
            [path.name for path in eligible[category]],
            f"objects:{category}",
            int(args.seed),
        )[: int(args.objects_per_category)]
        for object_name in choices:
            object_dir = categories_root / category / object_name
            scan_dir = object_dir / "Scan"
            inventory, tree_sha256, total_bytes = scan_tree_inventory(scan_dir)
            uid = f"omni_{category}_{object_name.rsplit('_', 1)[-1]}"
            extract_report = categories_root / category / "_EXTRACT_REPORT.json"
            row = {
                "uid": uid,
                "category": category,
                "object_name": object_name,
                "source_object_dir": str(object_dir),
                "source_scan_dir": str(scan_dir),
                "source_mesh": str(scan_dir / "Scan.obj"),
                "source_scan_tree_bytes": int(total_bytes),
                "source_scan_tree_sha256": tree_sha256,
                "source_scan_files": inventory,
                "category_extract_report": str(extract_report),
                "category_extract_report_sha256": sha256_file(extract_report),
                "selected_input_view_indices": selected_view_indices(uid, int(args.seed)),
            }
            objects.append(row)
            print(
                f"[freeze] {len(objects)}/{int(args.category_count) * int(args.objects_per_category)} "
                f"category={category} uid={uid} bytes={total_bytes}",
                flush=True,
            )

    code_path = Path(__file__).resolve()
    renderer_path = code_path.with_name("blender_pbr_render_multiview.py")
    blender = Path(args.blender_path).expanduser().resolve()
    payload: dict[str, Any] = {
        "format": FORMAT,
        "created_at_utc": utc_now(),
        "scope": "ReconViaGen-style independent reproduction; original paper UID list/seed unavailable",
        "categories_root": str(categories_root),
        "selection_policy": "seeded_sha256_without_replacement_v1",
        "selection_seed": int(args.seed),
        "eligible_category_count": len(eligible),
        "category_count": int(args.category_count),
        "objects_per_category": int(args.objects_per_category),
        "object_count": len(objects),
        "rendered_views_per_object": 4,
        "registered_camera_count": 24,
        "view_selection": "per-object seeded SHA256 random 4-of-24 without replacement",
        "camera_grid": "3 elevations (-20,10,40 deg) x 8 azimuths (45 deg interval)",
        "camera_coordinate_contract": "OpenCV c2w in model-O; +Z common up",
        "source_to_model_o_rotation": SOURCE_TO_MODEL_O.tolist(),
        "normalization": {
            "center": "source axis-aligned bounds midpoint",
            "scale": "source maximum axis extent",
            "canonical_margin": float(args.canonical_margin),
            "source_to_model_o": "[x,y,z] -> [x,-z,y] after centering/scaling",
        },
        "image_size": int(args.image_size),
        "intrinsic": intrinsic.tolist(),
        "focal_ratio": float(args.focal_ratio),
        "camera_radius": float(args.camera_radius),
        "all_24_cameras": cameras,
        "renderer": {
            "engine": str(args.blender_engine),
            "samples": int(args.blender_samples),
            "cycles_device": str(args.blender_cycles_device),
            "material_mode": "source",
            "transparent_background": True,
            "world_strength": float(args.world_strength),
            "light_energy": float(args.light_energy),
            "blender_path": str(blender),
            "blender_sha256": sha256_file(blender),
        },
        "metric_contract": {
            "metrics": ["Chamfer Distance", "F-score"],
            "surface_points_per_mesh": 100000,
            "metric_normalization": "all object points normalized to [-1,1]^3",
            "fscore_radius": 0.1,
            "dataset_aggregation": "arithmetic object mean",
        },
        "code_identity": {
            "builder": str(code_path),
            "builder_sha256": sha256_file(code_path),
            "blender_renderer": str(renderer_path),
            "blender_renderer_sha256": sha256_file(renderer_path),
            "git": git_identity(),
        },
        "objects": objects,
    }
    payload["protocol_sha256"] = canonical_sha256(payload)
    validate_protocol(payload)
    atomic_json(protocol_path, payload)
    print(
        json.dumps(
            {
                "passed": True,
                "reused": False,
                "protocol": str(protocol_path),
                "protocol_sha256": payload["protocol_sha256"],
                "categories": payload["category_count"],
                "objects": payload["object_count"],
            },
            indent=2,
        )
    )


def load_protocol(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    validate_protocol(payload)
    return payload


def source_tree_matches(row: dict[str, Any]) -> None:
    scan_dir = Path(row["source_scan_dir"])
    inventory, tree_sha256, total_bytes = scan_tree_inventory(scan_dir)
    if tree_sha256 != row["source_scan_tree_sha256"]:
        raise RuntimeError(f"source scan tree changed: {row['uid']}")
    if total_bytes != int(row["source_scan_tree_bytes"]):
        raise RuntimeError(f"source scan bytes changed: {row['uid']}")
    if inventory != row["source_scan_files"]:
        raise RuntimeError(f"source scan inventory changed: {row['uid']}")


def report_is_complete(path: Path, protocol_sha256: str, uid: str) -> bool:
    if not path.is_file():
        return False
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("format") != OBJECT_FORMAT:
            return False
        if report.get("protocol_sha256") != protocol_sha256 or report.get("uid") != uid:
            return False
        files = report.get("rendered_files")
        if not isinstance(files, list) or len(files) != 4:
            return False
        for row in files:
            for key in ("rgba", "mask", "rgb_white"):
                item = row[key]
                target = Path(item["path"])
                if not target.is_file() or target.stat().st_size != int(item["bytes"]):
                    return False
                if sha256_file(target) != item["sha256"]:
                    return False
        return True
    except Exception:
        return False


def save_png(path: Path, image: np.ndarray, mode: str) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp.{os.getpid()}.png")
    Image.fromarray(image, mode=mode).save(temporary, format="PNG")
    temporary.replace(path)
    return {
        "path": str(path),
        "bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


def render_one(
    protocol: dict[str, Any],
    row: dict[str, Any],
    output_root: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    started = time.monotonic()
    source_tree_matches(row)
    meshes = load_meshes(row["source_mesh"])
    vertices = np.concatenate(
        [np.asarray(mesh.vertices, dtype=np.float32) for mesh in meshes],
        axis=0,
    )
    normalized, center, scale = normalize_vertices(
        vertices,
        float(protocol["normalization"]["canonical_margin"]),
    )
    expected_model_o = pixal3d_render_space(normalized)
    selected = [int(index) for index in row["selected_input_view_indices"]]
    cameras_by_index = {
        int(camera["view_index"]): camera
        for camera in protocol["all_24_cameras"]
    }
    c2w = np.asarray(
        [cameras_by_index[index]["c2w_opencv_model_o"] for index in selected],
        dtype=np.float32,
    )
    intrinsic = np.asarray(protocol["intrinsic"], dtype=np.float32)
    # render_mesh_blender consumes these attributes from an argparse-like object.
    render_args = argparse.Namespace(
        blender_path=args.blender_path,
        xvfb_run_path=None,
        canonical_margin=float(protocol["normalization"]["canonical_margin"]),
        blender_bounds_tolerance=float(args.bounds_tolerance),
        blender_engine=protocol["renderer"]["engine"],
        blender_samples=int(protocol["renderer"]["samples"]),
        blender_cycles_device=protocol["renderer"]["cycles_device"],
        blender_world_strength=float(protocol["renderer"]["world_strength"]),
        blender_light_energy=float(protocol["renderer"]["light_energy"]),
        blender_quiet=True,
    )
    frames, bounds_audit = render_mesh_blender(
        row["source_mesh"],
        center,
        float(scale),
        c2w,
        intrinsic,
        int(protocol["image_size"]),
        expected_model_o,
        render_args,
    )
    if len(frames) != 4:
        raise RuntimeError(f"renderer returned {len(frames)} frames for {row['uid']}")
    object_root = output_root / "objects" / row["uid"]
    rendered_files: list[dict[str, Any]] = []
    for local_index, ((rgb_black, alpha), view_index) in enumerate(zip(frames, selected)):
        mask = alpha > 0
        fg_pixels = int(mask.sum())
        if fg_pixels < int(args.min_foreground_pixels):
            raise RuntimeError(
                f"rendered foreground too small: uid={row['uid']} view={view_index} pixels={fg_pixels}"
            )
        rgba = np.concatenate([rgb_black, alpha[..., None]], axis=-1)
        opacity = alpha[..., None].astype(np.float32) / 255.0
        white = np.clip(
            rgb_black.astype(np.float32) + 255.0 * (1.0 - opacity),
            0.0,
            255.0,
        ).astype(np.uint8)
        stem = f"view_{view_index:02d}"
        rendered_files.append(
            {
                "local_render_index": local_index,
                "view_index": view_index,
                "camera": cameras_by_index[view_index],
                "foreground_pixels": fg_pixels,
                "foreground_fraction": float(mask.mean()),
                "rgba": save_png(object_root / "rgba" / f"{stem}.png", rgba, "RGBA"),
                "mask": save_png(object_root / "masks" / f"{stem}.png", alpha, "L"),
                "rgb_white": save_png(object_root / "rgb_white" / f"{stem}.png", white, "RGB"),
            }
        )
    scale_factor = float(protocol["normalization"]["canonical_margin"]) / float(scale)
    source_to_model_o = np.eye(4, dtype=np.float64)
    source_to_model_o[:3, :3] = scale_factor * SOURCE_TO_MODEL_O
    source_to_model_o[:3, 3] = -scale_factor * (SOURCE_TO_MODEL_O @ center.astype(np.float64))
    return {
        "format": OBJECT_FORMAT,
        "created_at_utc": utc_now(),
        "protocol_sha256": protocol["protocol_sha256"],
        "uid": row["uid"],
        "category": row["category"],
        "source_mesh": row["source_mesh"],
        "source_scan_tree_sha256": row["source_scan_tree_sha256"],
        "normalization": {
            "source_bounds_center": center.astype(float).tolist(),
            "source_max_extent_scale": float(scale),
            "canonical_margin": float(protocol["normalization"]["canonical_margin"]),
            "source_to_model_o_4x4": source_to_model_o.tolist(),
            "model_o_up_axis": "+Z",
        },
        "intrinsic": protocol["intrinsic"],
        "selected_input_view_indices": selected,
        "rendered_files": rendered_files,
        "bounds_audit": bounds_audit,
        "elapsed_seconds": float(time.monotonic() - started),
        "passed": True,
    }


def worker(args: argparse.Namespace) -> None:
    protocol_path = Path(args.protocol).expanduser().resolve()
    protocol = load_protocol(protocol_path)
    output_root = protocol_path.parent
    rows = [
        row
        for index, row in enumerate(protocol["objects"])
        if index % int(args.num_workers) == int(args.worker_index)
    ]
    failures: list[dict[str, Any]] = []
    completed = 0
    reused = 0
    for local_index, row in enumerate(rows, 1):
        object_root = output_root / "objects" / row["uid"]
        report_path = object_root / "report.json"
        if report_is_complete(report_path, protocol["protocol_sha256"], row["uid"]):
            reused += 1
            print(
                f"[omni200] worker={args.worker_index} {local_index}/{len(rows)} "
                f"reused uid={row['uid']}",
                flush=True,
            )
            continue
        try:
            report = render_one(protocol, row, output_root, args)
            atomic_json(report_path, report)
            completed += 1
            print(
                f"[omni200] worker={args.worker_index} {local_index}/{len(rows)} "
                f"complete uid={row['uid']} elapsed={report['elapsed_seconds']:.1f}s",
                flush=True,
            )
        except Exception as exc:
            failure = {
                "uid": row["uid"],
                "category": row["category"],
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            failures.append(failure)
            print(
                f"[omni200] worker={args.worker_index} {local_index}/{len(rows)} "
                f"FAILED uid={row['uid']} error={failure['error_type']}: {failure['error']}",
                flush=True,
            )
    worker_report = {
        "format": "reconviagen.omniobject3d_omni200_worker.v1",
        "created_at_utc": utc_now(),
        "protocol_sha256": protocol["protocol_sha256"],
        "worker_index": int(args.worker_index),
        "num_workers": int(args.num_workers),
        "assigned_objects": len(rows),
        "completed_objects": completed,
        "reused_objects": reused,
        "failure_count": len(failures),
        "failures": failures,
        "passed": not failures and completed + reused == len(rows),
    }
    atomic_json(output_root / "logs" / f"worker_{int(args.worker_index):02d}_report.json", worker_report)
    if not worker_report["passed"]:
        raise SystemExit(2)


def finalize(args: argparse.Namespace) -> None:
    protocol_path = Path(args.protocol).expanduser().resolve()
    protocol = load_protocol(protocol_path)
    output_root = protocol_path.parent
    rows: list[dict[str, Any]] = []
    for source in protocol["objects"]:
        report_path = output_root / "objects" / source["uid"] / "report.json"
        if not report_is_complete(report_path, protocol["protocol_sha256"], source["uid"]):
            raise RuntimeError(f"object render incomplete: {source['uid']}")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "uid": source["uid"],
                "category": source["category"],
                "source_mesh": source["source_mesh"],
                "source_scan_tree_sha256": source["source_scan_tree_sha256"],
                "object_report": str(report_path),
                "object_report_sha256": sha256_file(report_path),
                "selected_input_view_indices": source["selected_input_view_indices"],
                "rgba_images": [item["rgba"]["path"] for item in report["rendered_files"]],
                "mask_images": [item["mask"]["path"] for item in report["rendered_files"]],
                "rgb_white_images": [item["rgb_white"]["path"] for item in report["rendered_files"]],
                "intrinsic": report["intrinsic"],
                "c2w_opencv_model_o": [item["camera"]["c2w_opencv_model_o"] for item in report["rendered_files"]],
                "source_to_model_o_4x4": report["normalization"]["source_to_model_o_4x4"],
            }
        )
    manifest: dict[str, Any] = {
        "format": FINAL_FORMAT,
        "created_at_utc": utc_now(),
        "protocol": str(protocol_path),
        "protocol_sha256": protocol["protocol_sha256"],
        "category_count": 20,
        "object_count": 200,
        "views_per_object": 4,
        "image_count": 800,
        "metric_contract": protocol["metric_contract"],
        "objects": rows,
    }
    manifest["manifest_identity"] = canonical_sha256(manifest)
    atomic_json(output_root / "manifest.json", manifest)
    report = {
        "format": "reconviagen.omniobject3d_omni200_build_report.v1",
        "completed_at_utc": utc_now(),
        "passed": True,
        "protocol": str(protocol_path),
        "protocol_sha256": protocol["protocol_sha256"],
        "manifest": str(output_root / "manifest.json"),
        "manifest_sha256": sha256_file(output_root / "manifest.json"),
        "category_count": len({row["category"] for row in rows}),
        "object_count": len(rows),
        "rgba_image_count": sum(len(row["rgba_images"]) for row in rows),
        "mask_image_count": sum(len(row["mask_images"]) for row in rows),
        "rgb_white_image_count": sum(len(row["rgb_white_images"]) for row in rows),
    }
    atomic_json(output_root / "report.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False))


def status(args: argparse.Namespace) -> None:
    protocol_path = Path(args.protocol).expanduser().resolve()
    if not protocol_path.is_file():
        print(json.dumps({"stage": "freeze", "protocol": str(protocol_path), "exists": False}))
        return
    protocol = load_protocol(protocol_path)
    output_root = protocol_path.parent
    complete = 0
    per_category: dict[str, int] = {}
    for row in protocol["objects"]:
        report_path = output_root / "objects" / row["uid"] / "report.json"
        if report_is_complete(report_path, protocol["protocol_sha256"], row["uid"]):
            complete += 1
            per_category[row["category"]] = per_category.get(row["category"], 0) + 1
    worker_reports = sorted((output_root / "logs").glob("worker_*_report.json")) if (output_root / "logs").is_dir() else []
    final = output_root / "report.json"
    print(
        json.dumps(
            {
                "stage": "complete" if final.is_file() else "render",
                "objects_complete": complete,
                "objects_total": int(protocol["object_count"]),
                "images_complete": complete * 4,
                "images_total": int(protocol["object_count"]) * 4,
                "percent": round(100.0 * complete / max(int(protocol["object_count"]), 1), 2),
                "categories_with_any_complete": len(per_category),
                "worker_reports": len(worker_reports),
                "final_report": str(final) if final.is_file() else None,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("freeze")
    p.add_argument("--categories_root", required=True)
    p.add_argument("--output_root", required=True)
    p.add_argument("--blender_path", required=True)
    p.add_argument("--seed", type=int, default=20260821)
    p.add_argument("--category_count", type=int, default=20)
    p.add_argument("--objects_per_category", type=int, default=10)
    p.add_argument("--image_size", type=int, default=512)
    p.add_argument("--focal_ratio", type=float, default=1.25)
    p.add_argument("--camera_radius", type=float, default=2.0)
    p.add_argument("--canonical_margin", type=float, default=0.9)
    p.add_argument("--blender_engine", default="CYCLES")
    p.add_argument("--blender_samples", type=int, default=16)
    p.add_argument("--blender_cycles_device", default="CUDA")
    p.add_argument("--world_strength", type=float, default=0.45)
    p.add_argument("--light_energy", type=float, default=500.0)
    p.set_defaults(func=freeze)

    p = sub.add_parser("worker")
    p.add_argument("--protocol", required=True)
    p.add_argument("--worker_index", type=int, required=True)
    p.add_argument("--num_workers", type=int, required=True)
    p.add_argument("--blender_path", required=True)
    p.add_argument("--bounds_tolerance", type=float, default=1.0e-3)
    p.add_argument("--min_foreground_pixels", type=int, default=512)
    p.set_defaults(func=worker)

    p = sub.add_parser("finalize")
    p.add_argument("--protocol", required=True)
    p.set_defaults(func=finalize)

    p = sub.add_parser("status")
    p.add_argument("--protocol", required=True)
    p.set_defaults(func=status)
    return parser


def main() -> None:
    args = make_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
