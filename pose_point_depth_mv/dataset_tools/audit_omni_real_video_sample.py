#!/usr/bin/env python3
"""Audit a stratified sample of OmniObject3D processed real-video objects."""

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
from PIL import Image, ImageDraw


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
REPORT_FORMAT = "pose_point_depth_mv.omni_real_video_sample_audit.v1"
EXTRACT_FORMAT = "pose_point_depth_mv.omni_real_video_sample_extract.v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive_root", required=True)
    parser.add_argument("--scan_root", required=True)
    parser.add_argument("--extract_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--sample",
        action="append",
        required=True,
        help="Frozen category:object_id pair; repeat for each sampled object.",
    )
    parser.add_argument("--min_frames", type=int, default=32)
    parser.add_argument("--min_registration_ratio", type=float, default=0.70)
    parser.add_argument("--min_point_count", type=int, default=100)
    parser.add_argument("--preview_frames", type=int, default=12)
    return parser.parse_args()


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


def parse_sample(value: str) -> tuple[str, str]:
    if ":" not in value:
        raise ValueError(f"sample must be category:object_id, got {value!r}")
    category, object_id = value.split(":", 1)
    if not re.fullmatch(r"[a-z0-9_]+", category):
        raise ValueError(f"invalid sample category={category!r}")
    if not re.fullmatch(re.escape(category) + r"_[0-9]+", object_id):
        raise ValueError(
            f"sample object ID {object_id!r} does not belong to {category!r}"
        )
    return category, object_id


def safe_relative_member(member: tarfile.TarInfo, object_id: str) -> Path | None:
    pure = PurePosixPath(member.name)
    if pure.is_absolute() or ".." in pure.parts:
        raise RuntimeError(f"unsafe archive path: {member.name!r}")
    if member.issym() or member.islnk() or member.isdev():
        raise RuntimeError(f"unsafe archive member type: {member.name!r}")
    try:
        object_index = pure.parts.index(object_id)
    except ValueError:
        return None
    suffix = pure.parts[object_index + 1 :]
    return Path(*suffix) if suffix else Path()


def extract_sample(
    archive: Path,
    category: str,
    object_id: str,
    extract_root: Path,
) -> tuple[Path, dict[str, Any]]:
    category_root = extract_root / category
    object_root = category_root / object_id
    marker = object_root / "_SAMPLE_EXTRACT_COMPLETE.json"
    if marker.is_file():
        payload = json.loads(marker.read_text(encoding="utf-8"))
        if (
            payload.get("format") != EXTRACT_FORMAT
            or payload.get("archive_sha256") != sha256_file(archive)
            or payload.get("object_id") != object_id
            or payload.get("passed") is not True
        ):
            raise RuntimeError(f"stale sample extraction marker: {marker}")
        return object_root, payload
    if object_root.exists():
        raise RuntimeError(
            f"partial sample extraction exists; preserve and inspect: {object_root}"
        )

    category_root.mkdir(parents=True, exist_ok=True)
    staging = category_root / f".{object_id}.extracting"
    if staging.exists():
        raise RuntimeError(f"partial staging extraction exists: {staging}")
    staging.mkdir(parents=False)

    file_count = 0
    total_bytes = 0
    try:
        with tarfile.open(archive, "r:gz") as bundle:
            for member in bundle:
                relative = safe_relative_member(member, object_id)
                if relative is None or relative == Path() or member.isdir():
                    continue
                if not member.isfile():
                    raise RuntimeError(f"unsupported member type: {member.name!r}")
                target = staging / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                source = bundle.extractfile(member)
                if source is None:
                    raise RuntimeError(f"cannot read archive member: {member.name!r}")
                with source, target.open("wb") as output:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
                file_count += 1
                total_bytes += target.stat().st_size
        if file_count == 0:
            raise RuntimeError(f"object {object_id!r} is absent from {archive}")
        staging.replace(object_root)
    except Exception:
        # Preserve the staging directory as a failure scene.
        raise

    payload = {
        "format": EXTRACT_FORMAT,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "category": category,
        "object_id": object_id,
        "archive": str(archive),
        "archive_sha256": sha256_file(archive),
        "extract_root": str(object_root),
        "file_count_before_marker": file_count,
        "bytes_before_marker": total_bytes,
        "passed": True,
    }
    write_json(marker, payload)
    return object_root, payload


def data_lines(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def parse_cameras(path: Path) -> dict[int, dict[str, Any]]:
    cameras: dict[int, dict[str, Any]] = {}
    for line in data_lines(path):
        fields = line.split()
        if len(fields) < 5:
            raise RuntimeError(f"invalid camera row in {path}: {line[:200]}")
        camera_id = int(fields[0])
        values = [float(value) for value in fields[4:]]
        if not all(math.isfinite(value) for value in values):
            raise RuntimeError(f"non-finite camera parameters in {path}")
        cameras[camera_id] = {
            "model": fields[1],
            "width": int(fields[2]),
            "height": int(fields[3]),
            "params": values,
        }
    return cameras


def parse_images(path: Path) -> list[dict[str, Any]]:
    lines = data_lines(path)
    if len(lines) % 2:
        raise RuntimeError(f"COLMAP images file has an odd data-line count: {path}")
    rows = []
    for line in lines[0::2]:
        fields = line.split()
        if len(fields) < 10:
            raise RuntimeError(f"invalid image pose row in {path}: {line[:200]}")
        pose = [float(value) for value in fields[1:8]]
        if not all(math.isfinite(value) for value in pose):
            raise RuntimeError(f"non-finite image pose in {path}")
        quaternion_norm = float(np.linalg.norm(np.asarray(pose[:4], dtype=np.float64)))
        rows.append(
            {
                "image_id": int(fields[0]),
                "qvec": pose[:4],
                "tvec": pose[4:7],
                "camera_id": int(fields[8]),
                "name": " ".join(fields[9:]),
                "quaternion_norm": quaternion_norm,
            }
        )
    return rows


def count_points(path: Path) -> int:
    return len(data_lines(path))


def image_paths(path: Path) -> list[Path]:
    return sorted(
        item for item in path.iterdir() if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES
    )


def evenly_spaced(items: list[Path], count: int) -> list[Path]:
    if len(items) <= count:
        return items
    indices = np.linspace(0, len(items) - 1, count, dtype=np.int64)
    return [items[int(index)] for index in indices]


def mask_area(path: Path) -> float:
    values = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
    return float((values >= 127).mean())


def make_contact_sheet(
    images: list[Path],
    mask_by_name: dict[str, Path],
    output: Path,
    count: int,
) -> None:
    selected = evenly_spaced(images, count)
    tile_width, tile_height, caption_height = 320, 180, 22
    columns = 4
    rows = int(math.ceil(len(selected) / columns))
    canvas = Image.new(
        "RGB", (columns * tile_width, rows * (tile_height + caption_height)), "white"
    )
    draw = ImageDraw.Draw(canvas)
    for index, image_path in enumerate(selected):
        rgb = Image.open(image_path).convert("RGB").resize((tile_width, tile_height))
        mask_path = mask_by_name[image_path.name]
        mask = Image.open(mask_path).convert("L").resize(
            (tile_width, tile_height), Image.Resampling.NEAREST
        )
        rgb_values = np.asarray(rgb, dtype=np.uint8).copy()
        foreground = np.asarray(mask, dtype=np.uint8) >= 127
        rgb_values[~foreground] = (rgb_values[~foreground].astype(np.float32) * 0.25).astype(
            np.uint8
        )
        overlay = Image.fromarray(rgb_values, mode="RGB")
        x = (index % columns) * tile_width
        y = (index // columns) * (tile_height + caption_height)
        canvas.paste(overlay, (x, y))
        draw.text((x + 4, y + tile_height + 3), image_path.name, fill="black")
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=92)


def audit_object(
    category: str,
    object_id: str,
    object_root: Path,
    scan_root: Path,
    preview_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    standard = object_root / "standard"
    images_dir = standard / "images"
    masks_dir = standard / "matting"
    sparse_txt = standard / "sparse" / "0_txt"
    required = [
        images_dir,
        masks_dir,
        sparse_txt / "cameras.txt",
        sparse_txt / "images.txt",
        sparse_txt / "points3D.txt",
        standard / "sparse" / "0" / "cameras.bin",
        standard / "sparse" / "0" / "images.bin",
        standard / "sparse" / "0" / "points3D.bin",
        standard / "poses_bounds.npy",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        return {
            "category": category,
            "object_id": object_id,
            "automatic_passed": False,
            "missing_required_paths": missing,
        }

    images = image_paths(images_dir)
    masks = image_paths(masks_dir)
    image_names = {path.name for path in images}
    mask_by_name = {path.name: path for path in masks}
    paired_names = image_names & set(mask_by_name)
    cameras = parse_cameras(sparse_txt / "cameras.txt")
    registered = parse_images(sparse_txt / "images.txt")
    registered_basenames = {Path(row["name"]).name for row in registered}
    unknown_registered_names = sorted(registered_basenames - image_names)
    unknown_camera_ids = sorted(
        {row["camera_id"] for row in registered if row["camera_id"] not in cameras}
    )
    quaternion_errors = [abs(row["quaternion_norm"] - 1.0) for row in registered]
    point_count = count_points(sparse_txt / "points3D.txt")
    sampled_masks = evenly_spaced([mask_by_name[name] for name in sorted(paired_names)], 20)
    mask_areas = [mask_area(path) for path in sampled_masks]
    poses_bounds = np.load(standard / "poses_bounds.npy")
    scan_obj = scan_root / category / object_id / "Scan" / "Scan.obj"
    preview = preview_dir / f"{object_id}_rgb_mask_contact.jpg"
    make_contact_sheet(images, mask_by_name, preview, args.preview_frames)

    registration_ratio = len(registered_basenames & image_names) / max(len(images), 1)
    pair_ratio = len(paired_names) / max(len(images), 1)
    checks = {
        "minimum_frames": len(images) >= args.min_frames,
        "image_mask_pair_ratio": pair_ratio >= 0.95,
        "registration_ratio": registration_ratio >= args.min_registration_ratio,
        "registered_names_resolve": not unknown_registered_names,
        "camera_ids_resolve": not unknown_camera_ids,
        "quaternions_unit": max(quaternion_errors, default=math.inf) <= 1.0e-4,
        "minimum_sparse_points": point_count >= args.min_point_count,
        "poses_bounds_shape": poses_bounds.ndim == 2
        and poses_bounds.shape[0] == len(registered)
        and poses_bounds.shape[1] == 17,
        "poses_bounds_finite": bool(np.isfinite(poses_bounds).all()),
        "mask_area_nonempty": bool(mask_areas) and min(mask_areas) >= 0.001,
        "mask_area_not_full": bool(mask_areas) and max(mask_areas) <= 0.95,
        "scan_obj_present": scan_obj.is_file() and scan_obj.stat().st_size >= 1024,
    }
    return {
        "category": category,
        "object_id": object_id,
        "object_root": str(object_root),
        "scan_obj": str(scan_obj),
        "image_count": len(images),
        "mask_count": len(masks),
        "paired_count": len(paired_names),
        "pair_ratio": pair_ratio,
        "camera_count": len(cameras),
        "camera_models": sorted({row["model"] for row in cameras.values()}),
        "registered_image_count": len(registered),
        "registration_ratio": registration_ratio,
        "sparse_point_count": point_count,
        "unknown_registered_names": unknown_registered_names,
        "unknown_camera_ids": unknown_camera_ids,
        "max_quaternion_norm_error": max(quaternion_errors, default=None),
        "poses_bounds_shape": list(poses_bounds.shape),
        "sampled_mask_area_min": min(mask_areas, default=None),
        "sampled_mask_area_median": float(np.median(mask_areas)) if mask_areas else None,
        "sampled_mask_area_max": max(mask_areas, default=None),
        "preview": str(preview),
        "checks": checks,
        "automatic_passed": all(checks.values()),
        "notes": (
            "Use standard/sparse/0_txt as the authoritative raw COLMAP model. "
            "The rescaled variants remain auxiliary until scan-frame alignment is audited."
        ),
    }


def main() -> None:
    args = parse_args()
    if args.min_frames <= 0 or not 0 < args.min_registration_ratio <= 1:
        raise ValueError("invalid audit thresholds")
    archive_root = Path(args.archive_root).expanduser().resolve()
    scan_root = Path(args.scan_root).expanduser().resolve()
    extract_root = Path(args.extract_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    samples = [parse_sample(value) for value in args.sample]
    if len(samples) != len(set(samples)):
        raise ValueError("duplicate sample requested")

    output_dir.mkdir(parents=True, exist_ok=True)
    preview_dir = output_dir / "previews"
    rows = []
    extractions = []
    for index, (category, object_id) in enumerate(samples, start=1):
        archive = archive_root / f"{category}.tar.gz"
        if not archive.is_file():
            raise FileNotFoundError(archive)
        print(
            f"[omni_real_sample] {index}/{len(samples)} "
            f"category={category} object={object_id}",
            flush=True,
        )
        object_root, extraction = extract_sample(
            archive, category, object_id, extract_root
        )
        extractions.append(extraction)
        row = audit_object(
            category,
            object_id,
            object_root,
            scan_root,
            preview_dir,
            args,
        )
        rows.append(row)
        print(
            f"[omni_real_sample] object={object_id} "
            f"automatic_passed={row['automatic_passed']}",
            flush=True,
        )

    payload = {
        "format": REPORT_FORMAT,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": (
            "fixed stratified sample; acceptance applies to the shared official "
            "videos_processed contract, not to every future object"
        ),
        "archive_root": str(archive_root),
        "scan_root": str(scan_root),
        "extract_root": str(extract_root),
        "sample_count": len(rows),
        "automatic_pass_count": sum(row["automatic_passed"] for row in rows),
        "automatic_passed": all(row["automatic_passed"] for row in rows),
        "manual_review_required": True,
        "manual_passed": False,
        "samples": rows,
        "extractions": extractions,
    }
    report = output_dir / "sample_audit.json"
    write_json(report, payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if not payload["automatic_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
