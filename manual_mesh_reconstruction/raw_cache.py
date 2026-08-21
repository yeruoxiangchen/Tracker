#!/usr/bin/env python3
"""Inventory and prepare raw OmniObject3D real-video reconstruction inputs.

This tool intentionally stops before producing a training manifest.  The official
COLMAP model and Scan.obj use unrelated coordinate frames until the separate
sampled Sim(3) diagnostic has passed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import tarfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
INVENTORY_FORMAT = "pose_point_depth_mv.omni_real_video_inventory.v1"
RAW_CACHE_FORMAT = "pose_point_depth_mv.omni_real_video_raw_cache.v1"
CATEGORY_MARKER_FORMAT = "pose_point_depth_mv.omni_real_video_category_extract.v1"
OBJECT_CACHE_FORMAT = "pose_point_depth_mv.omni_real_video_object_raw_cache.v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def load_categories(path: Path) -> list[str]:
    categories = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not categories or len(categories) != len(set(categories)):
        raise ValueError(f"empty or duplicate category list: {path}")
    for category in categories:
        if re.fullmatch(r"[a-z0-9_]+", category) is None:
            raise ValueError(f"invalid category: {category!r}")
    return categories


def safe_parts(member: tarfile.TarInfo) -> tuple[str, ...]:
    pure = PurePosixPath(member.name)
    if pure.is_absolute() or ".." in pure.parts:
        raise RuntimeError(f"unsafe archive path: {member.name!r}")
    if member.issym() or member.islnk() or member.isdev():
        raise RuntimeError(f"unsafe archive member type: {member.name!r}")
    return pure.parts


def object_member(
    parts: tuple[str, ...], category: str
) -> tuple[str, tuple[str, ...]] | None:
    pattern = re.compile(re.escape(category) + r"_[0-9]+$")
    for index, part in enumerate(parts):
        if pattern.fullmatch(part):
            return part, parts[index + 1 :]
    return None


def classify_relative(relative: tuple[str, ...]) -> tuple[str, str] | None:
    if len(relative) == 3 and relative[:2] == ("standard", "images"):
        suffix = Path(relative[-1]).suffix.lower()
        return ("image", relative[-1]) if suffix in IMAGE_SUFFIXES else None
    if len(relative) == 3 and relative[:2] == ("standard", "matting"):
        suffix = Path(relative[-1]).suffix.lower()
        return ("mask", relative[-1]) if suffix in IMAGE_SUFFIXES else None
    if relative in {
        ("standard", "sparse", "0_txt", "cameras.txt"),
        ("standard", "sparse", "0_txt", "images.txt"),
        ("standard", "sparse", "0_txt", "points3D.txt"),
    }:
        return ("colmap", relative[-1])
    if relative == ("standard", "poses_bounds.npy"):
        return ("poses_bounds", relative[-1])
    return None


def inventory_archive(archive: Path, category: str, scan_root: Path) -> dict[str, Any]:
    objects: dict[str, dict[str, Any]] = {}
    member_count = 0
    with tarfile.open(archive, "r:gz") as bundle:
        for member in bundle:
            parts = safe_parts(member)
            if not member.isfile():
                continue
            member_count += 1
            resolved = object_member(parts, category)
            if resolved is None:
                continue
            object_id, relative = resolved
            kind = classify_relative(relative)
            if kind is None:
                continue
            row = objects.setdefault(
                object_id,
                {
                    "object_id": object_id,
                    "image_names": set(),
                    "mask_names": set(),
                    "colmap_files": set(),
                    "poses_bounds": False,
                },
            )
            key, value = kind
            if key == "image":
                row["image_names"].add(value)
            elif key == "mask":
                row["mask_names"].add(value)
            elif key == "colmap":
                row["colmap_files"].add(value)
            elif key == "poses_bounds":
                row["poses_bounds"] = True

    output_rows = []
    required_colmap = {"cameras.txt", "images.txt", "points3D.txt"}
    for object_id, raw in sorted(objects.items()):
        scan_obj = scan_root / category / object_id / "Scan" / "Scan.obj"
        image_names = set(raw["image_names"])
        mask_names = set(raw["mask_names"])
        paired = image_names & mask_names
        checks = {
            "images_present": len(image_names) > 0,
            "image_mask_pair_ratio": len(paired) / max(len(image_names), 1) >= 0.95,
            "colmap_text_complete": set(raw["colmap_files"]) == required_colmap,
            "poses_bounds_present": bool(raw["poses_bounds"]),
            "scan_obj_present": scan_obj.is_file() and scan_obj.stat().st_size >= 1024,
        }
        output_rows.append(
            {
                "category": category,
                "object_id": object_id,
                "archive": str(archive),
                "scan_obj": str(scan_obj),
                "image_count": len(image_names),
                "mask_count": len(mask_names),
                "paired_count": len(paired),
                "colmap_files": sorted(raw["colmap_files"]),
                "checks": checks,
                "passed": all(checks.values()),
            }
        )
    return {
        "category": category,
        "archive": str(archive),
        "archive_bytes": archive.stat().st_size,
        "archive_member_count": member_count,
        "video_object_count": len(output_rows),
        "passed_object_count": sum(row["passed"] for row in output_rows),
        "objects": output_rows,
        "passed": bool(output_rows) and all(row["passed"] for row in output_rows),
    }


def command_inventory(args: argparse.Namespace) -> None:
    archive_root = Path(args.archive_root).expanduser().resolve()
    category_list = Path(args.category_list).expanduser().resolve()
    scan_root = Path(args.scan_root).expanduser().resolve()
    p7_report = Path(args.p7_report).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    categories = load_categories(category_list)
    p7 = json.loads(p7_report.read_text(encoding="utf-8"))
    if p7.get("passed") is not True:
        raise RuntimeError(f"P7 report did not pass: {p7_report}")
    p7_rows = {row["category"]: row for row in p7["categories"]}
    if set(categories) != set(p7_rows):
        raise RuntimeError("category list differs from the P7 frozen report")

    rows = []
    for index, category in enumerate(categories, start=1):
        archive = archive_root / f"{category}.tar.gz"
        frozen = p7_rows[category]
        if not archive.is_file() or archive.stat().st_size != frozen["archive_bytes"]:
            raise RuntimeError(f"archive size differs from P7: {archive}")
        print(f"[omni_inventory] {index}/{len(categories)} {category}", flush=True)
        rows.append(inventory_archive(archive, category, scan_root))

    objects = [obj for row in rows for obj in row["objects"]]
    payload = {
        "format": INVENTORY_FORMAT,
        "created_at_utc": utc_now(),
        "archive_root": str(archive_root),
        "scan_root": str(scan_root),
        "category_list": str(category_list),
        "category_list_sha256": sha256_file(category_list),
        "p7_report": str(p7_report),
        "p7_report_sha256": sha256_file(p7_report),
        "category_count": len(rows),
        "video_object_count": len(objects),
        "passed_object_count": sum(obj["passed"] for obj in objects),
        "categories": rows,
        "objects": objects,
        "scope_guard": (
            "Archive inventory and shared-field contract only. This is not a "
            "COLMAP-to-Scan alignment pass and is not a training manifest."
        ),
    }
    payload["passed"] = bool(objects) and all(row["passed"] for row in rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = output_dir / "inventory.json"
    write_json(report, payload)
    (output_dir / "video_objects.txt").write_text(
        "".join(f"{row['category']}:{row['object_id']}\n" for row in objects),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "passed": payload["passed"],
                "category_count": payload["category_count"],
                "video_object_count": payload["video_object_count"],
                "passed_object_count": payload["passed_object_count"],
                "report": str(report),
            },
            indent=2,
        )
    )
    if not payload["passed"]:
        raise SystemExit(2)


def data_lines(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def parse_cameras(path: Path) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    for line in data_lines(path):
        fields = line.split()
        if len(fields) < 5:
            raise RuntimeError(f"invalid camera row in {path}: {line[:200]}")
        rows[int(fields[0])] = {
            "model": fields[1],
            "width": int(fields[2]),
            "height": int(fields[3]),
            "params": [float(value) for value in fields[4:]],
        }
    return rows


def parse_registered_images(path: Path) -> list[dict[str, Any]]:
    lines = data_lines(path)
    if len(lines) % 2:
        raise RuntimeError(f"odd COLMAP images data-line count: {path}")
    rows = []
    for pose_line in lines[0::2]:
        fields = pose_line.split()
        if len(fields) < 10:
            raise RuntimeError(f"invalid image row in {path}: {pose_line[:200]}")
        rows.append(
            {
                "image_id": int(fields[0]),
                "qvec": np.asarray([float(v) for v in fields[1:5]], dtype=np.float64),
                "tvec": np.asarray([float(v) for v in fields[5:8]], dtype=np.float64),
                "camera_id": int(fields[8]),
                "name": Path(" ".join(fields[9:])).name,
            }
        )
    return rows


def qvec_to_rotation(qvec: np.ndarray) -> np.ndarray:
    q = np.asarray(qvec, dtype=np.float64)
    q = q / max(float(np.linalg.norm(q)), 1.0e-12)
    w, x, y, z = q
    return np.asarray(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
            [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
            [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
        ],
        dtype=np.float64,
    )


def camera_intrinsics(camera: dict[str, Any]) -> tuple[np.ndarray, list[float]]:
    model = camera["model"]
    params = camera["params"]
    if model == "SIMPLE_PINHOLE" and len(params) == 3:
        f, cx, cy = params
        fx, fy, distortion = f, f, []
    elif model == "PINHOLE" and len(params) == 4:
        fx, fy, cx, cy = params
        distortion = []
    elif model == "SIMPLE_RADIAL" and len(params) == 4:
        f, cx, cy, k1 = params
        fx, fy, distortion = f, f, [k1]
    elif model == "RADIAL" and len(params) == 5:
        f, cx, cy, k1, k2 = params
        fx, fy, distortion = f, f, [k1, k2]
    elif model == "OPENCV" and len(params) == 8:
        fx, fy, cx, cy, k1, k2, p1, p2 = params
        distortion = [k1, k2, p1, p2]
    else:
        raise RuntimeError(f"unsupported COLMAP camera model/params: {model} {params}")
    matrix = np.asarray(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64
    )
    return matrix, [float(value) for value in distortion]


def parse_points(path: Path) -> dict[str, np.ndarray]:
    point_ids = []
    xyz = []
    rgb = []
    errors = []
    track_lengths = []
    for line in data_lines(path):
        fields = line.split()
        if len(fields) < 8 or (len(fields) - 8) % 2:
            raise RuntimeError(f"invalid points3D row in {path}: {line[:200]}")
        point_ids.append(int(fields[0]))
        xyz.append([float(value) for value in fields[1:4]])
        rgb.append([int(value) for value in fields[4:7]])
        errors.append(float(fields[7]))
        track_lengths.append((len(fields) - 8) // 2)
    if not point_ids:
        raise RuntimeError(f"empty points3D model: {path}")
    error_array = np.asarray(errors, dtype=np.float64)
    track_array = np.asarray(track_lengths, dtype=np.int32)
    error_scale = max(float(np.median(error_array)), 1.0e-6)
    confidence = (1.0 - np.exp(-track_array.astype(np.float64) / 4.0)) * np.exp(
        -error_array / error_scale
    )
    return {
        "point_id": np.asarray(point_ids, dtype=np.int64),
        "xyz": np.asarray(xyz, dtype=np.float64),
        "rgb": np.asarray(rgb, dtype=np.uint8),
        "reprojection_error": error_array,
        "track_length": track_array,
        "confidence_proxy": confidence.astype(np.float64),
    }


def build_object_cache(object_root: Path, source_row: dict[str, Any]) -> dict[str, Any]:
    standard = object_root / "standard"
    sparse = standard / "sparse" / "0_txt"
    images_dir = standard / "images"
    masks_dir = standard / "matting"
    cameras = parse_cameras(sparse / "cameras.txt")
    registered = parse_registered_images(sparse / "images.txt")
    points = parse_points(sparse / "points3D.txt")
    image_names = {
        path.name
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    }
    mask_names = {
        path.name
        for path in masks_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    }

    matrices = []
    intrinsics = []
    image_ids = []
    camera_ids = []
    frame_names = []
    camera_rows = []
    for row in registered:
        if row["name"] not in image_names or row["name"] not in mask_names:
            continue
        camera = cameras[row["camera_id"]]
        k_matrix, distortion = camera_intrinsics(camera)
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = qvec_to_rotation(row["qvec"])
        transform[:3, 3] = row["tvec"]
        matrices.append(transform)
        intrinsics.append(k_matrix)
        image_ids.append(row["image_id"])
        camera_ids.append(row["camera_id"])
        frame_names.append(row["name"])
        camera_rows.append(
            {
                "frame_name": row["name"],
                "camera_id": row["camera_id"],
                "model": camera["model"],
                "width": camera["width"],
                "height": camera["height"],
                "params": camera["params"],
                "distortion": distortion,
            }
        )
    if not frame_names:
        raise RuntimeError(f"no registered RGB/mask pairs: {object_root}")

    npz_path = object_root / "raw_camera_point_cache.npz"
    write_npz(
        npz_path,
        frame_name=np.asarray(frame_names),
        image_id=np.asarray(image_ids, dtype=np.int64),
        camera_id=np.asarray(camera_ids, dtype=np.int64),
        K=np.asarray(intrinsics, dtype=np.float64),
        T_W2C=np.asarray(matrices, dtype=np.float64),
        P_W=points["xyz"],
        point_id=points["point_id"],
        point_rgb=points["rgb"],
        point_reprojection_error=points["reprojection_error"],
        point_track_length=points["track_length"],
        point_confidence_proxy=points["confidence_proxy"],
    )
    payload = {
        "format": OBJECT_CACHE_FORMAT,
        "created_at_utc": utc_now(),
        "category": source_row["category"],
        "object_id": source_row["object_id"],
        "object_root": str(object_root),
        "images_dir": str(images_dir),
        "masks_dir": str(masks_dir),
        "authoritative_colmap_dir": str(sparse),
        "poses_bounds": str(standard / "poses_bounds.npy"),
        "scan_obj": source_row["scan_obj"],
        "cache_npz": str(npz_path),
        "registered_pair_count": len(frame_names),
        "sparse_point_count": len(points["xyz"]),
        "camera_models": sorted({row["model"] for row in camera_rows}),
        "cameras": camera_rows,
        "confidence_proxy_definition": (
            "(1-exp(-track_length/4))*exp(-reprojection_error/median_error); "
            "diagnostic proxy, not calibrated probability"
        ),
        "distortion_policy": (
            "Raw images remain distorted. K and the original COLMAP model/params "
            "are retained; downstream projection must model distortion or create "
            "an explicitly audited undistorted derivative."
        ),
        "coordinate_policy": (
            "P_W and T_W2C remain in raw COLMAP world coordinates. Scan.obj is a "
            "separate frame until the sampled proper-Sim(3) diagnostic passes."
        ),
        "training_ready": False,
    }
    metadata = object_root / "raw_cache.json"
    payload["metadata"] = str(metadata)
    write_json(metadata, payload)
    return payload


def extract_category(
    category_row: dict[str, Any], output_root: Path, inventory_sha256: str
) -> tuple[list[dict[str, Any]], bool]:
    category = category_row["category"]
    archive = Path(category_row["archive"])
    category_root = output_root / "raw_objects" / category
    marker = category_root / "_CATEGORY_COMPLETE.json"
    expected = {row["object_id"]: row for row in category_row["objects"]}
    if marker.is_file():
        old = json.loads(marker.read_text(encoding="utf-8"))
        if (
            old.get("format") != CATEGORY_MARKER_FORMAT
            or old.get("inventory_sha256") != inventory_sha256
            or old.get("object_ids") != sorted(expected)
            or old.get("passed") is not True
        ):
            raise RuntimeError(f"stale category marker: {marker}")
        cached = [
            json.loads((category_root / object_id / "raw_cache.json").read_text(encoding="utf-8"))
            for object_id in sorted(expected)
        ]
        return cached, True
    if category_root.exists():
        raise RuntimeError(f"partial category output exists: {category_root}")

    staging = category_root.with_name(f".{category}.extracting")
    if staging.exists():
        raise RuntimeError(f"partial category staging exists: {staging}")
    staging.mkdir(parents=True)
    extracted_files = 0
    try:
        with tarfile.open(archive, "r:gz") as bundle:
            for member in bundle:
                parts = safe_parts(member)
                if not member.isfile():
                    continue
                resolved = object_member(parts, category)
                if resolved is None:
                    continue
                object_id, relative = resolved
                if object_id not in expected or classify_relative(relative) is None:
                    continue
                target = staging / object_id / Path(*relative)
                target.parent.mkdir(parents=True, exist_ok=True)
                source = bundle.extractfile(member)
                if source is None:
                    raise RuntimeError(f"cannot extract member: {member.name}")
                with source, target.open("wb") as output:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
                extracted_files += 1
        category_root.parent.mkdir(parents=True, exist_ok=True)
        staging.replace(category_root)
        cached = []
        for object_id, row in sorted(expected.items()):
            cached.append(build_object_cache(category_root / object_id, row))
        write_json(
            marker,
            {
                "format": CATEGORY_MARKER_FORMAT,
                "completed_at_utc": utc_now(),
                "category": category,
                "archive": str(archive),
                "inventory_sha256": inventory_sha256,
                "object_ids": sorted(expected),
                "object_count": len(expected),
                "extracted_file_count": extracted_files,
                "passed": True,
            },
        )
        return cached, False
    except Exception:
        # Preserve staging/output so an interrupted category is never mistaken for complete.
        raise


def command_extract_cache(args: argparse.Namespace) -> None:
    inventory_path = Path(args.inventory).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    if inventory.get("format") != INVENTORY_FORMAT or inventory.get("passed") is not True:
        raise RuntimeError(f"inventory is not eligible: {inventory_path}")
    inventory_sha256 = sha256_file(inventory_path)
    rows = []
    reused_categories = []
    for index, category_row in enumerate(inventory["categories"], start=1):
        print(
            f"[omni_raw_cache] {index}/{len(inventory['categories'])} "
            f"category={category_row['category']}",
            flush=True,
        )
        cached, reused = extract_category(category_row, output_dir, inventory_sha256)
        rows.extend(cached)
        if reused:
            reused_categories.append(category_row["category"])
    payload = {
        "format": RAW_CACHE_FORMAT,
        "created_at_utc": utc_now(),
        "inventory": str(inventory_path),
        "inventory_sha256": inventory_sha256,
        "output_dir": str(output_dir),
        "category_count": len(inventory["categories"]),
        "object_count": len(rows),
        "reused_categories": reused_categories,
        "objects": rows,
        "authoritative_colmap": "standard/sparse/0_txt",
        "excluded_inputs": ["standard/sparse/0_txt_rescaled", "standard/sparse/0_txt_rescaled_x1000"],
        "alignment_passed": False,
        "training_ready": False,
        "scope_guard": (
            "Raw camera/point cache only. No COLMAP-to-Scan alignment has been "
            "accepted and no training manifest is emitted."
        ),
        "passed": len(rows) == inventory["video_object_count"],
    }
    report = output_dir / "raw_cache_report.json"
    write_json(report, payload)
    print(
        json.dumps(
            {
                "passed": payload["passed"],
                "category_count": payload["category_count"],
                "object_count": payload["object_count"],
                "training_ready": payload["training_ready"],
                "report": str(report),
            },
            indent=2,
        )
    )
    if not payload["passed"]:
        raise SystemExit(2)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    inventory = subparsers.add_parser("inventory")
    inventory.add_argument("--archive_root", required=True)
    inventory.add_argument("--category_list", required=True)
    inventory.add_argument("--scan_root", required=True)
    inventory.add_argument("--p7_report", required=True)
    inventory.add_argument("--output_dir", required=True)
    inventory.set_defaults(handler=command_inventory)

    extract = subparsers.add_parser("extract-cache")
    extract.add_argument("--inventory", required=True)
    extract.add_argument("--output_dir", required=True)
    extract.set_defaults(handler=command_extract_cache)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
