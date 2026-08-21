#!/usr/bin/env python3
"""Freeze and render a registered ReconViaGen-style Dora-Bench-300 test.

The public ReconViaGen paper fixes the Dora input cameras to TRELLIS' 40-view
trajectory and consumes views 0, 9, 19 and 29.  The paper does not publish its
300 object IDs or random seed, so this builder freezes an explicit replacement
sample rather than claiming bit-exact reproduction of the paper table.

Only the four consumed images are materialized.  All forty TRELLIS cameras are
nevertheless frozen in the protocol, making the four input indices auditable.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import shutil
import subprocess
import time
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
from PIL import Image

from pose_point_depth_mv.dataset_tools.build_objaverse_multiview_sparse_data import (
    load_meshes,
    normalize_vertices,
    pixal3d_render_space,
    render_mesh_blender,
)
from pose_point_depth_mv.dataset_tools.build_reconviagen_omni200_benchmark import (
    look_at_z_up,
)


FORMAT = "reconviagen.dorabench_dora300_trellis40_input4.v1"
OBJECT_FORMAT = "reconviagen.dorabench_dora300_object_render4.v1"
FINAL_FORMAT = "reconviagen.dorabench_dora300_render_manifest.v1"
WORKER_FORMAT = "reconviagen.dorabench_dora300_render_worker.v1"
BUILD_REPORT_FORMAT = "reconviagen.dorabench_dora300_build_report.v1"
SELECTED_INPUT_VIEW_INDICES = [0, 9, 19, 29]
EXPECTED_ARCHIVE_SHA256 = (
    "bfbdadb1a99ddb6067d3b781c6f8e6bb01455bc24f3effb0240fe21e8f607ba2"
)
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


def radical_inverse(base: int, n: int) -> float:
    value = 0.0
    inv_base = 1.0 / float(base)
    inv = inv_base
    while n > 0:
        digit = n % base
        value += digit * inv
        n //= base
        inv *= inv_base
    return value


def sphere_hammersley_sequence(
    index: int,
    count: int,
    offset: tuple[float, float],
) -> tuple[float, float]:
    """Exact public TRELLIS dataset-toolkit remapping."""

    u = float(index) / float(count) + float(offset[0]) / float(count)
    v = radical_inverse(2, int(index)) + float(offset[1])
    u = 2.0 * u if u < 0.25 else (2.0 / 3.0) * u + 1.0 / 3.0
    pitch = float(np.arccos(1.0 - 2.0 * u) - np.pi / 2.0)
    yaw = float(v * 2.0 * np.pi)
    return yaw, pitch


def object_camera_plan(uid: str, selection_seed: int, image_size: int) -> dict[str, Any]:
    seed_bytes = hashlib.sha256(
        f"{selection_seed}:trellis40:{uid}".encode("utf-8")
    ).digest()[:4]
    camera_seed = int.from_bytes(seed_bytes, "big")
    rng = np.random.RandomState(camera_seed)
    offset = (float(rng.rand()), float(rng.rand()))
    fov_min, fov_max = 10.0, 70.0
    radius_min = math.sqrt(3.0) / 2.0 / math.sin(fov_max / 360.0 * math.pi)
    radius_max = math.sqrt(3.0) / 2.0 / math.sin(fov_min / 360.0 * math.pi)
    k_min = 1.0 / radius_max**2
    k_max = 1.0 / radius_min**2
    radii = 1.0 / np.sqrt(rng.uniform(k_min, k_max, (40,)))
    fovs = 2.0 * np.arcsin(math.sqrt(3.0) / 2.0 / radii)
    cameras: list[dict[str, Any]] = []
    for index in range(40):
        yaw, pitch = sphere_hammersley_sequence(index, 40, offset)
        radius = float(radii[index])
        fov = float(fovs[index])
        eye = np.asarray(
            [
                radius * math.cos(yaw) * math.cos(pitch),
                radius * math.sin(yaw) * math.cos(pitch),
                radius * math.sin(pitch),
            ],
            dtype=np.float64,
        )
        c2w = look_at_z_up(eye)
        focal = 0.5 * float(image_size) / math.tan(0.5 * fov)
        intrinsic = np.asarray(
            [
                [focal, 0.0, (image_size - 1) * 0.5],
                [0.0, focal, (image_size - 1) * 0.5],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        cameras.append(
            {
                "view_index": index,
                "yaw_radians": yaw,
                "pitch_radians": pitch,
                "radius": radius,
                "fov_radians": fov,
                "fov_degrees": math.degrees(fov),
                "intrinsic": intrinsic.tolist(),
                "c2w_opencv_model_o": c2w.tolist(),
            }
        )
    return {
        "rng": "numpy.random.RandomState(MT19937)",
        "camera_seed": camera_seed,
        "hammersley_offset": list(offset),
        "registered_camera_count": 40,
        "cameras": cameras,
    }


def git_identity() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]

    def run(*args: str) -> str:
        completed = subprocess.run(
            ["git", *args],
            cwd=root,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return completed.stdout.strip()

    dirty = run("status", "--short", "--untracked-files=all")
    return {
        "root": str(root),
        "branch": run("branch", "--show-current"),
        "head": run("rev-parse", "HEAD"),
        "dirty": bool(dirty),
        "dirty_status_sha256": hashlib.sha256(dirty.encode("utf-8")).hexdigest(),
    }


def protocol_without_hash(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if key != "protocol_sha256"}


def validate_protocol(payload: dict[str, Any]) -> None:
    if payload.get("format") != FORMAT:
        raise RuntimeError(f"Dora300 protocol format differs: {payload.get('format')}")
    expected = canonical_sha256(protocol_without_hash(payload))
    if payload.get("protocol_sha256") != expected:
        raise RuntimeError("Dora300 protocol hash differs")
    rows = list(payload.get("objects") or [])
    if int(payload.get("object_count", -1)) != 300 or len(rows) != 300:
        raise RuntimeError("Dora300 protocol does not contain exactly 300 objects")
    uids = [str(row["uid"]) for row in rows]
    sources = [str(row["dora_source_record"]) for row in rows]
    if len(set(uids)) != 300 or len(set(sources)) != 300:
        raise RuntimeError("Dora300 protocol contains duplicate object identity")
    for row in rows:
        if row.get("selected_input_view_indices") != SELECTED_INPUT_VIEW_INDICES:
            raise RuntimeError(f"Dora input indices differ: {row['uid']}")
        plan = row.get("trellis_camera_plan") or {}
        cameras = list(plan.get("cameras") or [])
        if int(plan.get("registered_camera_count", -1)) != 40 or len(cameras) != 40:
            raise RuntimeError(f"Dora TRELLIS camera matrix differs: {row['uid']}")
        if [int(camera["view_index"]) for camera in cameras] != list(range(40)):
            raise RuntimeError(f"Dora TRELLIS camera indices differ: {row['uid']}")


def _metadata(root: Path) -> tuple[list[str], dict[str, str], list[dict[str, Any]]]:
    level_by_record: dict[str, str] = {}
    bindings: list[dict[str, Any]] = []
    for level in range(1, 5):
        path = root / f"Level{level}.json"
        values = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(values, list):
            raise RuntimeError(f"Dora level metadata is not a list: {path}")
        for value in values:
            record = str(value)
            if record in level_by_record:
                raise RuntimeError(f"Dora level identity overlaps: {record}")
            level_by_record[record] = f"Level{level}"
        bindings.append(
            {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        )
    all_path = root / "Level_all.json"
    all_rows = [str(value) for value in json.loads(all_path.read_text(encoding="utf-8"))]
    if len(all_rows) != 3202 or len(set(all_rows)) != 3202:
        raise RuntimeError("Dora Level_all identity/count differs")
    if set(all_rows) != set(level_by_record):
        raise RuntimeError("Dora Level1-4 union differs from Level_all")
    bindings.append(
        {
            "path": str(all_path),
            "bytes": all_path.stat().st_size,
            "sha256": sha256_file(all_path),
        }
    )
    return all_rows, level_by_record, bindings


def _extract_selected(
    archive: Path,
    output_root: Path,
    selected: list[str],
    level_by_record: dict[str, str],
    selection_seed: int,
    image_size: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with zipfile.ZipFile(archive) as bundle:
        infos = [info for info in bundle.infolist() if info.filename.lower().endswith(".obj")]
        by_basename: dict[str, zipfile.ZipInfo] = {}
        for info in infos:
            basename = PurePosixPath(info.filename).name
            if basename in by_basename:
                raise RuntimeError(f"duplicate Dora archive basename: {basename}")
            by_basename[basename] = info
        if len(by_basename) != 3202:
            raise RuntimeError(f"Dora archive OBJ count differs: {len(by_basename)}")
        for index, source_record in enumerate(selected, 1):
            basename = PurePosixPath(source_record).name
            info = by_basename.get(basename)
            if info is None:
                raise RuntimeError(f"Dora metadata object is absent from ZIP: {source_record}")
            uid = f"dora_{hashlib.sha256(source_record.encode('utf-8')).hexdigest()[:20]}"
            object_dir = output_root / "source_meshes" / uid
            destination = object_dir / "mesh.obj"
            if not destination.is_file() or destination.stat().st_size != int(info.file_size):
                object_dir.mkdir(parents=True, exist_ok=True)
                temporary = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
                with bundle.open(info) as source, temporary.open("wb") as target:
                    shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
                if temporary.stat().st_size != int(info.file_size):
                    temporary.unlink(missing_ok=True)
                    raise RuntimeError(f"Dora extracted size differs: {source_record}")
                temporary.replace(destination)
            mesh_sha = sha256_file(destination)
            inventory = [
                {"relative_path": "mesh.obj", "bytes": destination.stat().st_size, "sha256": mesh_sha}
            ]
            tree_sha = canonical_sha256(inventory)
            source_kind = source_record.split("/", 1)[0]
            row = {
                "uid": uid,
                "category": level_by_record[source_record],
                "complexity_level": level_by_record[source_record],
                "source_dataset": source_kind,
                "dora_source_record": source_record,
                "dora_archive_member": info.filename,
                "dora_archive_crc32": f"{int(info.CRC):08x}",
                "source_object_dir": str(object_dir),
                "source_scan_dir": str(object_dir),
                "source_mesh": str(destination),
                "source_scan_tree_bytes": int(destination.stat().st_size),
                "source_scan_tree_sha256": tree_sha,
                "source_scan_files": inventory,
                "selected_input_view_indices": list(SELECTED_INPUT_VIEW_INDICES),
                "trellis_camera_plan": object_camera_plan(uid, selection_seed, image_size),
            }
            rows.append(row)
            print(
                f"[dora300:freeze] {index}/300 level={row['category']} "
                f"source={source_kind} uid={uid} bytes={destination.stat().st_size}",
                flush=True,
            )
    return rows


def freeze(args: argparse.Namespace) -> None:
    source_root = Path(args.source_root).expanduser().resolve(strict=True)
    output_root = Path(args.output_root).expanduser().resolve()
    protocol_path = output_root / "protocol.json"
    if protocol_path.is_file():
        payload = json.loads(protocol_path.read_text(encoding="utf-8"))
        validate_protocol(payload)
        if (
            int(payload["selection_seed"]) != int(args.seed)
            or int(payload["image_size"]) != int(args.image_size)
            or int(payload["object_count"]) != int(args.object_count)
        ):
            raise RuntimeError("existing Dora300 frozen configuration differs")
        print(
            json.dumps(
                {
                    "passed": True,
                    "reused": True,
                    "protocol": str(protocol_path),
                    "protocol_sha256": payload["protocol_sha256"],
                }
            )
        )
        return
    if int(args.object_count) != 300:
        raise ValueError("registered Dora-Bench run requires object_count=300")
    archive = source_root / "dora-bench-256.zip"
    if not archive.is_file():
        raise FileNotFoundError(archive)
    archive_sha = sha256_file(archive)
    if archive_sha != EXPECTED_ARCHIVE_SHA256:
        raise RuntimeError(
            f"Dora archive SHA256 differs: {archive_sha} != {EXPECTED_ARCHIVE_SHA256}"
        )
    all_rows, level_by_record, metadata_bindings = _metadata(source_root)
    selected = seeded_order(all_rows, "registered_dora300", int(args.seed))[:300]
    output_root.mkdir(parents=True, exist_ok=True)
    objects = _extract_selected(
        archive,
        output_root,
        selected,
        level_by_record,
        int(args.seed),
        int(args.image_size),
    )
    code_path = Path(__file__).resolve()
    renderer_path = code_path.with_name("blender_pbr_render_multiview.py")
    blender = Path(args.blender_path).expanduser().resolve(strict=True)
    payload: dict[str, Any] = {
        "format": FORMAT,
        "created_at_utc": utc_now(),
        "scope": (
            "registered ReconViaGen-style Dora-Bench reproduction; the paper's "
            "original 300 UID list and selection seed are not public"
        ),
        "source_root": str(source_root),
        "source_archive": {
            "path": str(archive),
            "bytes": archive.stat().st_size,
            "sha256": archive_sha,
        },
        "source_metadata": metadata_bindings,
        "source_population_count": len(all_rows),
        "selection_policy": "seeded_sha256_without_replacement_v1",
        "selection_seed": int(args.seed),
        "object_count": len(objects),
        "complexity_level_count": 4,
        "registered_camera_count": 40,
        "rendered_views_per_object": 4,
        "selected_input_view_indices": list(SELECTED_INPUT_VIEW_INDICES),
        "trajectory": (
            "TRELLIS public dataset_toolkits/render_cond.py: per-object random-offset "
            "sphere Hammersley trajectory and inverse-square-uniform camera radius/FOV"
        ),
        "trajectory_randomness": (
            "public TRELLIS trajectory distribution with per-object deterministic "
            "MT19937 seed derived from the registered selection seed and UID"
        ),
        "physical_render_scope": (
            "all 40 cameras frozen; only consumed input indices 0,9,19,29 rendered "
            "because evaluation requests CD/F-score rather than novel-view metrics"
        ),
        "camera_coordinate_contract": "OpenCV c2w in model-O; +Z common up",
        "source_to_model_o_rotation": SOURCE_TO_MODEL_O.tolist(),
        "normalization": {
            "center": "source axis-aligned bounds midpoint",
            "scale": "source maximum axis extent",
            "canonical_margin": float(args.canonical_margin),
            "source_to_model_o": "[x,y,z] -> [x,-z,y] after centering/scaling",
        },
        "image_size": int(args.image_size),
        "renderer": {
            "engine": str(args.blender_engine),
            "samples": int(args.blender_samples),
            "cycles_device": str(args.blender_cycles_device),
            "material_mode": "source (Dora-Bench-256 archive contains OBJ geometry only)",
            "transparent_background": True,
            "world_strength": float(args.world_strength),
            "light_energy": float(args.light_energy),
            "blender_path": str(blender),
            "blender_sha256": sha256_file(blender),
        },
        "metric_contract": {
            "metrics": ["Chamfer Distance", "F-score"],
            "surface_points_per_mesh": 100000,
            "metric_normalization": "each mesh independently AABB-normalized to [-1,1]",
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
                "objects": 300,
                "protocol": str(protocol_path),
                "protocol_sha256": payload["protocol_sha256"],
            },
            indent=2,
        )
    )


def load_protocol(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    validate_protocol(payload)
    return payload


def source_tree_matches(row: dict[str, Any]) -> None:
    mesh = Path(row["source_mesh"])
    inventory = [
        {"relative_path": "mesh.obj", "bytes": mesh.stat().st_size, "sha256": sha256_file(mesh)}
    ]
    if canonical_sha256(inventory) != row["source_scan_tree_sha256"]:
        raise RuntimeError(f"Dora source mesh changed: {row['uid']}")


def report_is_complete(path: Path, protocol_sha256: str, uid: str) -> bool:
    if not path.is_file():
        return False
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
        if (
            report.get("format") != OBJECT_FORMAT
            or report.get("protocol_sha256") != protocol_sha256
            or report.get("uid") != uid
            or report.get("passed") is not True
        ):
            return False
        files = list(report.get("rendered_files") or [])
        if len(files) != 4:
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
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


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
        [np.asarray(mesh.vertices, dtype=np.float32) for mesh in meshes], axis=0
    )
    normalized, center, scale = normalize_vertices(
        vertices, float(protocol["normalization"]["canonical_margin"])
    )
    expected_model_o = pixal3d_render_space(normalized)
    selected = list(SELECTED_INPUT_VIEW_INDICES)
    cameras = {int(item["view_index"]): item for item in row["trellis_camera_plan"]["cameras"]}
    selected_cameras = [cameras[index] for index in selected]
    c2w = np.asarray(
        [item["c2w_opencv_model_o"] for item in selected_cameras], dtype=np.float32
    )
    intrinsics = np.asarray([item["intrinsic"] for item in selected_cameras], dtype=np.float32)
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
        intrinsics,
        int(protocol["image_size"]),
        expected_model_o,
        render_args,
    )
    if len(frames) != 4:
        raise RuntimeError(f"Dora renderer returned {len(frames)} frames: {row['uid']}")
    object_root = output_root / "objects" / row["uid"]
    rendered_files = []
    for local_index, ((rgb_black, alpha), view_index, camera) in enumerate(
        zip(frames, selected, selected_cameras)
    ):
        foreground = alpha > 0
        foreground_pixels = int(foreground.sum())
        if foreground_pixels < int(args.min_foreground_pixels):
            raise RuntimeError(
                f"Dora foreground too small: uid={row['uid']} view={view_index} "
                f"pixels={foreground_pixels}"
            )
        rgba = np.concatenate([rgb_black, alpha[..., None]], axis=-1)
        opacity = alpha[..., None].astype(np.float32) / 255.0
        white = np.clip(
            rgb_black.astype(np.float32) + 255.0 * (1.0 - opacity), 0.0, 255.0
        ).astype(np.uint8)
        stem = f"view_{view_index:02d}"
        rendered_files.append(
            {
                "local_render_index": local_index,
                "view_index": view_index,
                "camera": camera,
                "foreground_pixels": foreground_pixels,
                "foreground_fraction": float(foreground.mean()),
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
        "complexity_level": row["complexity_level"],
        "source_dataset": row["source_dataset"],
        "dora_source_record": row["dora_source_record"],
        "source_mesh": row["source_mesh"],
        "source_scan_tree_sha256": row["source_scan_tree_sha256"],
        "normalization": {
            "source_bounds_center": center.astype(float).tolist(),
            "source_max_extent_scale": float(scale),
            "canonical_margin": float(protocol["normalization"]["canonical_margin"]),
            "source_to_model_o_4x4": source_to_model_o.tolist(),
            "model_o_up_axis": "+Z",
        },
        "selected_input_view_indices": selected,
        "intrinsic": intrinsics.tolist(),
        "rendered_files": rendered_files,
        "bounds_audit": bounds_audit,
        "elapsed_seconds": float(time.monotonic() - started),
        "passed": True,
    }


def worker(args: argparse.Namespace) -> None:
    protocol_path = Path(args.protocol).expanduser().resolve(strict=True)
    protocol = load_protocol(protocol_path)
    output_root = protocol_path.parent
    rows = [
        row
        for index, row in enumerate(protocol["objects"])
        if index % int(args.num_workers) == int(args.worker_index)
    ]
    failures: list[dict[str, Any]] = []
    completed = reused = 0
    for local_index, row in enumerate(rows, 1):
        report_path = output_root / "objects" / row["uid"] / "report.json"
        if report_is_complete(report_path, protocol["protocol_sha256"], row["uid"]):
            reused += 1
            print(
                f"[dora300] worker={args.worker_index} {local_index}/{len(rows)} "
                f"reused uid={row['uid']}",
                flush=True,
            )
            continue
        try:
            report = render_one(protocol, row, output_root, args)
            atomic_json(report_path, report)
            completed += 1
            print(
                f"[dora300] worker={args.worker_index} {local_index}/{len(rows)} "
                f"complete uid={row['uid']} elapsed={report['elapsed_seconds']:.1f}s",
                flush=True,
            )
        except Exception as exc:
            failure = {
                "uid": row["uid"],
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            failures.append(failure)
            print(
                f"[dora300] worker={args.worker_index} {local_index}/{len(rows)} "
                f"FAILED uid={row['uid']} error={failure['error_type']}: {failure['error']}",
                flush=True,
            )
    report = {
        "format": WORKER_FORMAT,
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
    atomic_json(output_root / "logs" / f"worker_{int(args.worker_index):02d}_report.json", report)
    if not report["passed"]:
        raise SystemExit(2)


def finalize(args: argparse.Namespace) -> None:
    protocol_path = Path(args.protocol).expanduser().resolve(strict=True)
    protocol = load_protocol(protocol_path)
    output_root = protocol_path.parent
    rows = []
    for source in protocol["objects"]:
        report_path = output_root / "objects" / source["uid"] / "report.json"
        if not report_is_complete(report_path, protocol["protocol_sha256"], source["uid"]):
            raise RuntimeError(f"Dora object render incomplete: {source['uid']}")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "uid": source["uid"],
                "category": source["category"],
                "complexity_level": source["complexity_level"],
                "source_dataset": source["source_dataset"],
                "dora_source_record": source["dora_source_record"],
                "source_mesh": source["source_mesh"],
                "source_scan_tree_sha256": source["source_scan_tree_sha256"],
                "object_report": str(report_path),
                "object_report_sha256": sha256_file(report_path),
                "selected_input_view_indices": list(SELECTED_INPUT_VIEW_INDICES),
                "rgba_images": [item["rgba"]["path"] for item in report["rendered_files"]],
                "mask_images": [item["mask"]["path"] for item in report["rendered_files"]],
                "rgb_white_images": [item["rgb_white"]["path"] for item in report["rendered_files"]],
                "intrinsic": report["intrinsic"],
                "c2w_opencv_model_o": [
                    item["camera"]["c2w_opencv_model_o"] for item in report["rendered_files"]
                ],
                "source_to_model_o_4x4": report["normalization"]["source_to_model_o_4x4"],
            }
        )
    manifest: dict[str, Any] = {
        "format": FINAL_FORMAT,
        "created_at_utc": utc_now(),
        "protocol": str(protocol_path),
        "protocol_sha256": protocol["protocol_sha256"],
        "object_count": 300,
        "category_count": 4,
        "views_per_object": 4,
        "registered_camera_count": 40,
        "selected_input_view_indices": list(SELECTED_INPUT_VIEW_INDICES),
        "image_count": 1200,
        "metric_contract": protocol["metric_contract"],
        "objects": rows,
    }
    manifest["manifest_identity"] = canonical_sha256(manifest)
    atomic_json(output_root / "manifest.json", manifest)
    report = {
        "format": BUILD_REPORT_FORMAT,
        "completed_at_utc": utc_now(),
        "passed": True,
        "protocol": str(protocol_path),
        "protocol_sha256": protocol["protocol_sha256"],
        "manifest": str(output_root / "manifest.json"),
        "manifest_sha256": sha256_file(output_root / "manifest.json"),
        "object_count": 300,
        "level_count": 4,
        "rgba_image_count": 1200,
        "mask_image_count": 1200,
        "rgb_white_image_count": 1200,
    }
    atomic_json(output_root / "report.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False))


def status(args: argparse.Namespace) -> None:
    protocol_path = Path(args.protocol).expanduser().resolve()
    if not protocol_path.is_file():
        print(json.dumps({"stage": "freeze_or_extract", "protocol": str(protocol_path)}))
        return
    protocol = load_protocol(protocol_path)
    output_root = protocol_path.parent
    complete = sum(
        report_is_complete(
            output_root / "objects" / row["uid"] / "report.json",
            protocol["protocol_sha256"],
            row["uid"],
        )
        for row in protocol["objects"]
    )
    worker_reports = list((output_root / "logs").glob("worker_*_report.json"))
    final = output_root / "report.json"
    print(
        json.dumps(
            {
                "stage": "complete" if final.is_file() else "render",
                "objects_complete": complete,
                "objects_total": 300,
                "images_complete": complete * 4,
                "images_total": 1200,
                "percent": round(100.0 * complete / 300.0, 2),
                "worker_reports": len(worker_reports),
                "final_report": str(final) if final.is_file() else None,
            },
            indent=2,
        )
    )


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("freeze")
    p.add_argument("--source_root", required=True)
    p.add_argument("--output_root", required=True)
    p.add_argument("--blender_path", required=True)
    p.add_argument("--seed", type=int, default=20260821)
    p.add_argument("--object_count", type=int, default=300)
    p.add_argument("--image_size", type=int, default=1024)
    p.add_argument("--canonical_margin", type=float, default=1.0)
    p.add_argument("--blender_engine", default="CYCLES")
    p.add_argument("--blender_samples", type=int, default=128)
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
    p.add_argument("--min_foreground_pixels", type=int, default=2048)
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
