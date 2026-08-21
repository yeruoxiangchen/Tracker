#!/usr/bin/env python3
"""Rebuild one phone AR capture with full-sequence SAM2 masks and COLMAP.

COLMAP sees complete RGB frames so background texture can stabilize SfM. SAM2
masks are propagated from existing user-supervised key frames and are used only
after SfM: a 3D point survives when its actual feature-track observations land
inside the object mask in multiple views. RGB and masks are then undistorted
together before the current no-VGGT reconstruction model is called.

All stages are resumable and use a new output tree. Existing capture and
diagnostic products are never modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
from typing import Any, Iterable, Sequence

import numpy as np
from PIL import Image, ImageDraw

from pose_point_depth_mv.ar_object_capture import utc_now, write_json
from pose_point_depth_mv.ar_object_reconstruction import (
    ARReconstructionConfig,
    ReconstructionPaths,
    run_ar_object_reconstruction,
)
from pose_point_depth_mv.dataset_tools.prepare_coarsemodel_real_raw_cache import (
    nearest_rotation,
    proper_umeyama,
    read_phone_poses,
    rotation_error_degrees,
    unity_quaternion_to_rotation,
)
from pose_point_depth_mv.dataset_tools.prepare_omni_real_video_cache import (
    camera_intrinsics,
    parse_cameras,
    parse_registered_images,
    qvec_to_rotation,
    sha256_file,
)
from pose_point_depth_mv.run_ar_offline_colmap_ab import run_colmap


REPO_ROOT = Path(__file__).resolve().parents[1]
FORMAT = "pose_point_depth_mv.ar_full_colmap_reconstruction.v1"
DEFAULT_RAW = REPO_ROOT / "pose_point_depth_mv/outputs/可视AR/runtime/data/20260810_133235_817"
DEFAULT_SEEDS = REPO_ROOT / "pose_point_depth_mv/outputs/可视AR/datasets/20260810_133235_817"
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "pose_point_depth_mv/outputs/可视AR/全96帧COLMAP重建/"
    "20260810_133235_817_full96_colmap_v1"
)
DEFAULT_COLMAP = Path("/home/zjr/anaconda3/envs/foundpose/bin/colmap")
DEFAULT_PYTHON = Path("/home/zjr/anaconda3/envs/reconviagen/bin/python")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def image_paths(directory: Path) -> list[Path]:
    return sorted(
        path.resolve()
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def mask_for_name(directory: Path, frame_name: str) -> Path:
    stem = Path(frame_name).stem
    for suffix in (".png", ".jpg", ".jpeg"):
        path = directory / f"{stem}{suffix}"
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError(f"mask is missing for {frame_name}: {directory}")


def copy_bound(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        if sha256_file(source) != sha256_file(destination):
            raise RuntimeError(f"existing immutable copy differs: {destination}")
        return
    shutil.copy2(source, destination)


def binary_mask(path: Path) -> np.ndarray:
    with Image.open(path) as handle:
        return np.asarray(handle.convert("L")) >= 128


def clean_binary_mask(mask: np.ndarray) -> np.ndarray:
    """Drop tiny disconnected mask speckles while retaining real thin parts."""

    import cv2

    value = np.asarray(mask, dtype=bool).astype(np.uint8) * 255
    value = cv2.morphologyEx(
        value, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8), iterations=1
    )
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        (value > 0).astype(np.uint8), connectivity=8
    )
    if count <= 1:
        return value
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest = int(np.argmax(areas)) + 1
    largest_area = int(areas[largest - 1])
    keep = [largest]
    keep.extend(
        index
        for index in range(1, count)
        if index != largest
        and int(stats[index, cv2.CC_STAT_AREA]) >= max(64, largest_area // 25)
    )
    return np.isin(labels, keep).astype(np.uint8) * 255


def _sam2_propagate(
    frames: Sequence[Path], seed_masks: dict[int, np.ndarray]
) -> dict[int, np.ndarray]:
    import cv2
    import tempfile
    import torch

    from CoarseModel.connect.sam2_mask import _load_sam2_video_predictor

    predictor = _load_sam2_video_predictor()
    with tempfile.TemporaryDirectory(prefix="ar817_sam2_full96_") as temporary:
        video_dir = Path(temporary) / "frames"
        video_dir.mkdir()
        shapes: list[tuple[int, int]] = []
        for index, source in enumerate(frames):
            image = cv2.imread(str(source), cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(source)
            shapes.append(image.shape[:2])
            if not cv2.imwrite(str(video_dir / f"{index:05d}.jpg"), image):
                raise RuntimeError(f"failed to stage SAM2 frame: {source}")

        state = predictor.init_state(
            video_path=str(video_dir),
            offload_video_to_cpu=True,
            offload_state_to_cpu=True,
            async_loading_frames=False,
        )
        for index in sorted(seed_masks):
            predictor.add_new_mask(
                inference_state=state,
                frame_idx=int(index),
                obj_id=1,
                mask=np.asarray(seed_masks[index], dtype=bool),
            )

        outputs: dict[int, np.ndarray] = {}
        start = min(seed_masks)
        autocast = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if torch.cuda.is_available()
            else torch.no_grad()
        )
        with torch.inference_mode(), autocast:
            for reverse in (False, True):
                for frame_index, object_ids, logits in predictor.propagate_in_video(
                    state, start_frame_idx=start, reverse=reverse
                ):
                    ids = [int(value) for value in object_ids]
                    object_index = ids.index(1) if 1 in ids else 0
                    mask = (logits[object_index] > 0.0).detach().cpu().numpy().squeeze()
                    if mask.shape != shapes[int(frame_index)]:
                        mask = cv2.resize(
                            mask.astype(np.uint8),
                            (shapes[int(frame_index)][1], shapes[int(frame_index)][0]),
                            interpolation=cv2.INTER_NEAREST,
                        ) > 0
                    outputs[int(frame_index)] = clean_binary_mask(mask)

        # User-supervised key-frame masks remain authoritative.
        for index, mask in seed_masks.items():
            outputs[index] = np.asarray(mask, dtype=bool).astype(np.uint8) * 255
        missing = sorted(set(range(len(frames))) - set(outputs))
        if missing:
            raise RuntimeError(f"SAM2 did not return every frame: {missing}")
        return outputs


def mask_qc(mask: np.ndarray) -> dict[str, Any]:
    mask = np.asarray(mask, dtype=bool)
    height, width = mask.shape
    area = int(mask.sum())
    ys, xs = np.nonzero(mask)
    bbox = (
        [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
        if area
        else None
    )
    centroid = (
        [float(xs.mean() / width), float(ys.mean() / height)] if area else None
    )
    border = np.zeros_like(mask)
    border[:2] = border[-2:] = True
    border[:, :2] = border[:, -2:] = True
    return {
        "foreground_pixels": area,
        "foreground_ratio": float(area / max(height * width, 1)),
        "border_pixels": int(np.logical_and(mask, border).sum()),
        "bbox_xyxy": bbox,
        "centroid_normalized": centroid,
    }


def write_mask_contact_sheet(
    frames: Sequence[Path],
    masks: Sequence[Path],
    destination: Path,
    *,
    max_tiles: int = 16,
    tile_size: tuple[int, int] = (320, 240),
    columns: int = 4,
) -> None:
    indices = np.linspace(
        0, len(frames) - 1, min(max_tiles, len(frames)), dtype=int
    )
    tile_width, tile_height = tile_size
    tiles = []
    for index in indices:
        with Image.open(frames[int(index)]) as handle:
            image = handle.convert("RGB").resize(tile_size)
        with Image.open(masks[int(index)]) as handle:
            mask = handle.convert("L").resize(tile_size, Image.Resampling.NEAREST)
        rgb = np.asarray(image).copy()
        foreground = np.asarray(mask) >= 128
        rgb[foreground] = np.clip(
            0.55 * rgb[foreground] + 0.45 * np.asarray([40, 255, 80]), 0, 255
        )
        tile = Image.fromarray(rgb.astype(np.uint8))
        draw = ImageDraw.Draw(tile)
        draw.rectangle((0, 0, 125, 20), fill=(0, 0, 0))
        draw.text((4, 3), frames[int(index)].name, fill=(255, 255, 255))
        tiles.append(tile)
    sheet = Image.new(
        "RGB",
        (
            columns * tile_width,
            math.ceil(len(tiles) / columns) * tile_height,
        ),
        (24, 24, 24),
    )
    for index, tile in enumerate(tiles):
        sheet.paste(
            tile,
            (
                (index % columns) * tile_width,
                (index // columns) * tile_height,
            ),
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination)


def prepare_masks(
    *, raw_session: Path, seed_dataset: Path, output_dir: Path, expected_frames: int
) -> dict[str, Any]:
    report_path = output_dir / "01_masks/mask_report.json"
    if report_path.is_file():
        report = read_json(report_path)
        if report.get("passed") is True:
            print(f"[ar_full_colmap] reuse masks: {report_path}", flush=True)
            return report
    frames = image_paths(raw_session)
    if len(frames) != expected_frames:
        raise RuntimeError(f"expected {expected_frames} raw frames, got {len(frames)}")
    frame_index = {path.name: index for index, path in enumerate(frames)}
    seeds: dict[int, np.ndarray] = {}
    seed_bindings = []
    for image in image_paths(seed_dataset / "images"):
        if image.name not in frame_index:
            raise RuntimeError(f"seed frame is absent from raw session: {image.name}")
        path = mask_for_name(seed_dataset / "masks", image.name)
        mask = binary_mask(path)
        with Image.open(frames[frame_index[image.name]]) as handle:
            expected_size = (handle.height, handle.width)
        if mask.shape != expected_size:
            raise RuntimeError(f"seed mask/image dimensions differ for {image.name}")
        seeds[frame_index[image.name]] = mask
        seed_bindings.append(
            {
                "frame_name": image.name,
                "frame_index": frame_index[image.name],
                "mask": str(path),
                "mask_sha256": sha256_file(path),
            }
        )
    if len(seeds) < 3:
        raise RuntimeError(f"need at least 3 supervised seed masks, got {len(seeds)}")
    propagated = _sam2_propagate(frames, seeds)

    dataset = output_dir / "01_masks/dataset_all96"
    images_out = dataset / "images"
    masks_out = dataset / "masks"
    rows = []
    for index, source in enumerate(frames):
        copy_bound(source, images_out / source.name)
        destination = masks_out / f"{source.stem}.png"
        destination.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(propagated[index]).save(destination)
        quality = mask_qc(propagated[index] > 0)
        quality.update(
            {
                "frame_name": source.name,
                "seed": index in seeds,
                "image_sha256": sha256_file(source),
                "mask_sha256": sha256_file(destination),
            }
        )
        rows.append(quality)
    for name in ("poses.txt", "frame_metadata.jsonl"):
        source = raw_session / name
        if not source.is_file():
            raise FileNotFoundError(f"raw capture metadata is missing: {source}")
        copy_bound(source, dataset / name)
    ratios = np.asarray([row["foreground_ratio"] for row in rows], dtype=np.float64)
    empty = [row["frame_name"] for row in rows if row["foreground_pixels"] == 0]
    implausible = [
        row["frame_name"]
        for row in rows
        if not 0.003 <= float(row["foreground_ratio"]) <= 0.85
    ]
    sheet = output_dir / "01_masks/mask_contact_sheet_16.png"
    sheet_all = output_dir / "01_masks/mask_contact_sheet_all96.png"
    mask_paths = [masks_out / f"{path.stem}.png" for path in frames]
    write_mask_contact_sheet(
        frames, mask_paths, sheet
    )
    write_mask_contact_sheet(
        frames,
        mask_paths,
        sheet_all,
        max_tiles=len(frames),
        tile_size=(160, 120),
        columns=8,
    )
    report = {
        "format": FORMAT,
        "stage": "full_sequence_sam2_masks",
        "created_at_utc": utc_now(),
        "raw_session": str(raw_session.resolve()),
        "seed_dataset": str(seed_dataset.resolve()),
        "dataset": str(dataset.resolve()),
        "frame_count": len(frames),
        "seed_count": len(seeds),
        "seed_bindings": seed_bindings,
        "foreground_ratio": {
            "minimum": float(ratios.min()),
            "median": float(np.median(ratios)),
            "maximum": float(ratios.max()),
        },
        "empty_masks": empty,
        "implausible_area_masks": implausible,
        "frames": rows,
        "contact_sheet": str(sheet.resolve()),
        "all_frame_contact_sheet": str(sheet_all.resolve()),
        "passed": not empty and not implausible and len(frames) == expected_frames,
    }
    write_json(report_path, report)
    if not report["passed"]:
        raise RuntimeError(f"full-sequence mask QC failed: {report_path}")
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "passed",
                    "frame_count",
                    "seed_count",
                    "foreground_ratio",
                    "contact_sheet",
                )
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return report


def parse_images_observations(path: Path) -> list[dict[str, Any]]:
    data = [
        line.strip()
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if len(data) % 2:
        raise RuntimeError(f"invalid COLMAP images.txt pair count: {path}")
    rows = []
    for pose, points in zip(data[0::2], data[1::2]):
        pose_fields = pose.split()
        point_fields = points.split()
        if len(pose_fields) < 10 or len(point_fields) % 3:
            raise RuntimeError(f"invalid COLMAP image record: {pose[:120]}")
        observations = [
            [float(point_fields[i]), float(point_fields[i + 1]), int(point_fields[i + 2])]
            for i in range(0, len(point_fields), 3)
        ]
        rows.append(
            {
                "image_id": int(pose_fields[0]),
                "camera_id": int(pose_fields[8]),
                "name": Path(" ".join(pose_fields[9:])).name,
                "pose_line": pose,
                "observations": observations,
            }
        )
    return rows


def point_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 8 or (len(fields) - 8) % 2:
            raise RuntimeError(f"invalid COLMAP point row: {line[:120]}")
        rows.append(
            {
                "point_id": int(fields[0]),
                "xyz": np.asarray([float(value) for value in fields[1:4]]),
                "error": float(fields[7]),
                "track": [
                    (int(fields[i]), int(fields[i + 1]))
                    for i in range(8, len(fields), 2)
                ],
                "line": line.strip(),
            }
        )
    return rows


def dilated_masks(
    dataset: Path, names: Iterable[str], radius: int
) -> dict[str, np.ndarray]:
    import cv2

    kernel = np.ones((2 * radius + 1, 2 * radius + 1), dtype=np.uint8)
    output = {}
    for name in names:
        mask = binary_mask(mask_for_name(dataset / "masks", name)).astype(np.uint8)
        if radius:
            mask = cv2.dilate(mask, kernel, iterations=1)
        output[name] = mask > 0
    return output


def filter_track_points(
    *,
    text_model: Path,
    mask_dataset: Path,
    output_sparse: Path,
    min_observations: int,
    min_positive: int,
    min_ratio: float,
    max_error: float,
    mask_dilation: int,
    min_points: int,
) -> dict[str, Any]:
    images = parse_images_observations(text_model / "images.txt")
    points = point_rows(text_model / "points3D.txt")
    by_id = {row["image_id"]: row for row in images}
    masks = dilated_masks(mask_dataset, (row["name"] for row in images), mask_dilation)
    kept = []
    diagnostics = []
    for point in points:
        observed = positive = mismatched = 0
        for image_id, point2d_index in point["track"]:
            image = by_id.get(image_id)
            if image is None or not (0 <= point2d_index < len(image["observations"])):
                continue
            x, y, linked_id = image["observations"][point2d_index]
            if linked_id != point["point_id"]:
                mismatched += 1
                continue
            mask = masks[image["name"]]
            px, py = int(round(x)), int(round(y))
            if not (0 <= px < mask.shape[1] and 0 <= py < mask.shape[0]):
                continue
            observed += 1
            positive += int(mask[py, px])
        ratio = float(positive / observed) if observed else 0.0
        passed = (
            observed >= min_observations
            and positive >= min_positive
            and ratio >= min_ratio
            and point["error"] <= max_error
            and mismatched == 0
        )
        diagnostics.append((observed, positive, ratio, point["error"], passed))
        if passed:
            kept.append(point)
    if len(kept) < min_points:
        raise RuntimeError(
            f"track-aware object cloud has only {len(kept)} points; need {min_points}. "
            "Inspect masks/registration instead of relaxing automatically."
        )
    keep_ids = {row["point_id"] for row in kept}
    output_sparse.mkdir(parents=True, exist_ok=True)
    shutil.copy2(text_model / "cameras.txt", output_sparse / "cameras_distorted.txt")
    with (output_sparse / "points3D.txt").open("w", encoding="utf-8") as handle:
        handle.write("# Track-aware object points filtered by foreground observations\n")
        for row in kept:
            handle.write(row["line"] + "\n")
    with (output_sparse / "images_distorted.txt").open("w", encoding="utf-8") as handle:
        handle.write("# Image list; removed object-external point IDs are replaced by -1\n")
        for row in images:
            handle.write(row["pose_line"] + "\n")
            fields = []
            for x, y, point_id in row["observations"]:
                fields.extend(
                    [
                        f"{x:.12g}",
                        f"{y:.12g}",
                        str(point_id if point_id in keep_ids else -1),
                    ]
                )
            handle.write(" ".join(fields) + "\n")
    values = np.asarray(diagnostics, dtype=np.float64)
    return {
        "input_point_count": len(points),
        "kept_point_count": len(kept),
        "kept_fraction": float(len(kept) / max(len(points), 1)),
        "thresholds": {
            "min_track_observations": min_observations,
            "min_positive_mask_observations": min_positive,
            "min_positive_mask_ratio": min_ratio,
            "max_reprojection_error_px": max_error,
            "mask_dilation_px": mask_dilation,
            "min_output_points": min_points,
        },
        "all_points_observed_track_count_median": float(np.median(values[:, 0])),
        "all_points_mask_ratio_median": float(np.median(values[:, 2])),
        "kept_point_ids_sha256": hashlib.sha256(
            "\n".join(str(value) for value in sorted(keep_ids)).encode()
        ).hexdigest(),
        "passed": True,
    }


def robust_colmap_to_ar_sim3(text_model: Path, poses_path: Path) -> dict[str, Any]:
    registered = parse_registered_images(text_model / "images.txt")
    phone = read_phone_poses(poses_path)
    by_name = {str(row["name"]): row for row in registered}
    names = sorted(set(phone) & set(by_name))
    if len(names) < 8:
        raise RuntimeError(f"need >=8 common phone/COLMAP poses, got {len(names)}")
    source, target = [], []
    colmap_c2w, unity_c2w = [], []
    for name in names:
        row = by_name[name]
        rotation_w2c = qvec_to_rotation(row["qvec"])
        source.append(-rotation_w2c.T @ np.asarray(row["tvec"], dtype=np.float64))
        # poses.txt is authored by Unity and its position is already the AR
        # world position.  Reflecting Z here a second time creates a false
        # handedness error (verified against the earlier 18-frame model).
        target.append(np.asarray(phone[name]["position"], dtype=np.float64))
        colmap_c2w.append(rotation_w2c.T)
        quaternion = phone[name].get("quaternion")
        if quaternion is None:
            raise RuntimeError(f"Unity quaternion is missing for {name}")
        unity_c2w.append(unity_quaternion_to_rotation(quaternion))
    source_array = np.asarray(source)
    target_array = np.asarray(target)
    active = np.ones(len(names), dtype=bool)
    for _ in range(5):
        scale, rotation, translation = proper_umeyama(
            source_array[active], target_array[active]
        )
        predicted = scale * (source_array @ rotation.T) + translation
        errors = np.linalg.norm(predicted - target_array, axis=1)
        active_errors = errors[active]
        median = float(np.median(active_errors))
        mad = float(np.median(np.abs(active_errors - median)))
        threshold = max(median + 3.5 * 1.4826 * mad, median * 2.0, 0.01)
        updated = errors <= threshold
        if int(updated.sum()) < 8 or np.array_equal(updated, active):
            break
        active = updated
    scale, rotation, translation = proper_umeyama(
        source_array[active], target_array[active]
    )
    predicted = scale * (source_array @ rotation.T) + translation
    errors = np.linalg.norm(predicted - target_array, axis=1)
    diameter = float(
        np.max(np.linalg.norm(target_array[:, None] - target_array[None, :], axis=2))
    )
    active_errors = errors[active]
    matrix = np.eye(4)
    matrix[:3, :3] = scale * rotation
    matrix[:3, 3] = translation
    gravity = rotation.T @ np.asarray([0.0, 1.0, 0.0])
    gravity /= max(float(np.linalg.norm(gravity)), 1.0e-12)
    # COLMAP camera coordinates and Unity camera coordinates differ by one
    # constant axis convention.  Estimate it explicitly, then audit whether
    # the residual is genuinely constant rather than hiding per-frame drift.
    camera_basis = nearest_rotation(
        sum(
            (rotation @ colmap_rotation).T @ unity_rotation
            for keep, colmap_rotation, unity_rotation in zip(
                active, colmap_c2w, unity_c2w
            )
            if keep
        )
    )
    all_rotation_errors = np.asarray(
        [
            rotation_error_degrees(
                rotation @ colmap_rotation @ camera_basis, unity_rotation
            )
            for colmap_rotation, unity_rotation in zip(colmap_c2w, unity_c2w)
        ],
        dtype=np.float64,
    )
    rotation_errors = all_rotation_errors[active]
    checks = {
        "inlier_count_ge_8": int(active.sum()) >= 8,
        "inlier_fraction_ge_0p75": float(active.mean()) >= 0.75,
        "median_center_error_over_trajectory_le_0p05": float(
            np.median(active_errors) / max(diameter, 1.0e-12)
        )
        <= 0.05,
        "p90_center_error_over_trajectory_le_0p10": float(
            np.quantile(active_errors, 0.9) / max(diameter, 1.0e-12)
        )
        <= 0.10,
        "proper_positive_similarity": bool(
            scale > 0 and np.linalg.det(rotation) > 0.999
        ),
        "camera_axis_residual_median_deg_le_5": float(np.median(rotation_errors))
        <= 5.0,
        "camera_axis_residual_p90_deg_le_10": float(np.quantile(rotation_errors, 0.9))
        <= 10.0,
    }
    return {
        "method": "robust_proper_umeyama_colmap_centers_to_unity_ar_world.v2",
        "common_pose_count": len(names),
        "inlier_count": int(active.sum()),
        "inlier_frames": [name for name, keep in zip(names, active) if keep],
        "rejected_frames": [name for name, keep in zip(names, active) if not keep],
        "scale": float(scale),
        "rotation": rotation.tolist(),
        "translation": translation.tolist(),
        "T_colmap_to_ar": matrix.tolist(),
        "colmap_camera_to_unity_camera_basis": camera_basis.tolist(),
        "gravity_up_colmap": gravity.tolist(),
        "trajectory_diameter_ar": diameter,
        "center_error_ar": {
            "median": float(np.median(active_errors)),
            "p90": float(np.quantile(active_errors, 0.9)),
            "maximum": float(np.max(active_errors)),
        },
        "camera_axis_residual_degrees": {
            "median": float(np.median(rotation_errors)),
            "p90": float(np.quantile(rotation_errors, 0.9)),
            "maximum": float(np.max(rotation_errors)),
        },
        "checks": checks,
        "passed": all(checks.values()),
    }


def undistort_dataset(
    *, source_dataset: Path, source_sparse: Path, destination: Path
) -> dict[str, Any]:
    import cv2

    cameras = parse_cameras(source_sparse / "cameras_distorted.txt")
    image_rows = parse_images_observations(source_sparse / "images_distorted.txt")
    maps = {}
    for camera_id, camera in cameras.items():
        K, distortion = camera_intrinsics(camera)
        width, height = int(camera["width"]), int(camera["height"])
        vector = np.zeros(5)
        vector[: len(distortion)] = distortion
        new_K, _ = cv2.getOptimalNewCameraMatrix(
            K, vector, (width, height), 0.0, (width, height)
        )
        map_x, map_y = cv2.initUndistortRectifyMap(
            K, vector, None, new_K, (width, height), cv2.CV_32FC1
        )
        maps[int(camera_id)] = (map_x, map_y, new_K, (width, height), K, vector)

    images_out = destination / "images"
    masks_out = destination / "masks"
    sparse_out = destination / "sparse/0"
    images_out.mkdir(parents=True, exist_ok=True)
    masks_out.mkdir(parents=True, exist_ok=True)
    sparse_out.mkdir(parents=True, exist_ok=True)
    output_rows = []
    for row in image_rows:
        camera_id = int(row["camera_id"])
        map_x, map_y, new_K, _size, old_K, distortion = maps[camera_id]
        image = cv2.imread(
            str(source_dataset / "images" / row["name"]), cv2.IMREAD_COLOR
        )
        mask = cv2.imread(
            str(mask_for_name(source_dataset / "masks", row["name"])),
            cv2.IMREAD_GRAYSCALE,
        )
        if image is None or mask is None:
            raise FileNotFoundError(f"registered RGB/mask is missing: {row['name']}")
        image_u = cv2.remap(
            image, map_x, map_y, cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT
        )
        mask_u = clean_binary_mask(
            cv2.remap(
                mask,
                map_x,
                map_y,
                cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
            )
            >= 128
        )
        cv2.imwrite(str(images_out / row["name"]), image_u)
        cv2.imwrite(str(masks_out / f"{Path(row['name']).stem}.png"), mask_u)
        xy = np.asarray(
            [[item[0], item[1]] for item in row["observations"]], dtype=np.float64
        )
        xy_u = (
            cv2.undistortPoints(xy[:, None, :], old_K, distortion, P=new_K)[:, 0, :]
            if len(xy)
            else xy
        )
        observations = [
            [float(new[0]), float(new[1]), int(old[2])]
            for old, new in zip(row["observations"], xy_u)
        ]
        output_rows.append({**row, "observations": observations})

    with (sparse_out / "cameras.txt").open("w", encoding="utf-8") as handle:
        handle.write("# Undistorted PINHOLE cameras\n")
        for camera_id in sorted(cameras):
            _, _, K, (width, height), _, _ = maps[camera_id]
            handle.write(
                f"{camera_id} PINHOLE {width} {height} "
                f"{K[0,0]:.12g} {K[1,1]:.12g} {K[0,2]:.12g} {K[1,2]:.12g}\n"
            )
    with (sparse_out / "images.txt").open("w", encoding="utf-8") as handle:
        handle.write("# Undistorted observations; poses are unchanged\n")
        for row in output_rows:
            handle.write(row["pose_line"] + "\n")
            fields = []
            for x, y, point_id in row["observations"]:
                fields.extend([f"{x:.12g}", f"{y:.12g}", str(point_id)])
            handle.write(" ".join(fields) + "\n")
    shutil.copy2(source_sparse / "points3D.txt", sparse_out / "points3D.txt")
    for name in ("poses.txt", "frame_metadata.jsonl"):
        source = source_dataset / name
        if source.is_file():
            shutil.copy2(source, destination / name)
    return {
        "registered_output_frames": len(output_rows),
        "camera_count": len(cameras),
        "camera_models": ["PINHOLE"],
        "rgb_mask_same_undistortion_map": True,
        "sparse": str(sparse_out.resolve()),
        "passed": True,
    }


def prepare_colmap(
    *,
    output_dir: Path,
    colmap_bin: Path,
    gpu: str,
    expected_frames: int,
    min_registration_ratio: float,
    min_track_observations: int,
    min_positive_observations: int,
    min_positive_ratio: float,
    max_reprojection_error: float,
    mask_dilation: int,
    min_object_points: int,
    resume: bool,
) -> dict[str, Any]:
    report_path = output_dir / "02_colmap/colmap_object_report.json"
    if report_path.is_file():
        report = read_json(report_path)
        if report.get("passed") is True:
            print(f"[ar_full_colmap] reuse COLMAP object dataset: {report_path}")
            return report
    mask_report = read_json(output_dir / "01_masks/mask_report.json")
    if mask_report.get("passed") is not True:
        raise RuntimeError("mask stage has not passed")
    dataset = Path(mask_report["dataset"])
    frames = image_paths(dataset / "images")
    if len(frames) != expected_frames:
        raise RuntimeError(f"mask dataset has {len(frames)} frames; expected {expected_frames}")
    preferred = [
        "frame_0005.jpg",
        "frame_0013.jpg",
        "frame_0025.jpg",
        "frame_0032.jpg",
        "frame_0042.jpg",
        "frame_0058.jpg",
        "frame_0075.jpg",
        "frame_0091.jpg",
    ]
    text_model, selection = run_colmap(
        source_dataset=dataset,
        workspace=output_dir / "02_colmap/workspace",
        all_frames=[path.name for path in frames],
        fixed_frames=preferred,
        colmap_bin=colmap_bin,
        gpu=gpu,
        use_foreground_masks=False,
        resume=resume,
    )
    registered_count = int(selection["final_registered_count"])
    registration_ratio = float(registered_count / expected_frames)
    if registration_ratio < min_registration_ratio:
        raise RuntimeError(
            f"COLMAP registered only {registered_count}/{expected_frames} frames "
            f"({registration_ratio:.3f}); required {min_registration_ratio:.3f}"
        )
    filtered_sparse = output_dir / "02_colmap/filtered_distorted_sparse"
    track_audit = filter_track_points(
        text_model=text_model,
        mask_dataset=dataset,
        output_sparse=filtered_sparse,
        min_observations=min_track_observations,
        min_positive=min_positive_observations,
        min_ratio=min_positive_ratio,
        max_error=max_reprojection_error,
        mask_dilation=mask_dilation,
        min_points=min_object_points,
    )
    sim3 = robust_colmap_to_ar_sim3(text_model, dataset / "poses.txt")
    object_dataset = output_dir / "02_colmap/dataset_registered_undistorted_object"
    undistortion = undistort_dataset(
        source_dataset=dataset,
        source_sparse=filtered_sparse,
        destination=object_dataset,
    )
    report = {
        "format": FORMAT,
        "stage": "full_rgb_colmap_track_mask_filter_and_undistort",
        "created_at_utc": utc_now(),
        "input_dataset": str(dataset.resolve()),
        "object_dataset": str(object_dataset.resolve()),
        "colmap_workspace": str((output_dir / "02_colmap/workspace").resolve()),
        "colmap_model": str(text_model.resolve()),
        "expected_frame_count": expected_frames,
        "registered_frame_count": registered_count,
        "registration_ratio": registration_ratio,
        "registration_minimum": min_registration_ratio,
        "feature_domain": "full_rgb",
        "object_isolation": "actual_colmap_track_observations_sampled_in_sam2_masks",
        "selection": selection,
        "track_filter": track_audit,
        "undistortion": undistortion,
        "colmap_to_ar": sim3,
        "passed": bool(
            track_audit["passed"] and undistortion["passed"] and sim3["passed"]
        ),
    }
    write_json(report_path, report)
    if not report["passed"]:
        raise RuntimeError(f"COLMAP/AR alignment audit failed: {report_path}")
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "registered": f"{registered_count}/{expected_frames}",
                "object_points": track_audit["kept_point_count"],
                "sim3": {
                    "scale": sim3["scale"],
                    "center_error_ar": sim3["center_error_ar"],
                },
                "dataset": report["object_dataset"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return report


def transform_mesh(source: Path, destination: Path, matrix: np.ndarray) -> None:
    import trimesh

    loaded = trimesh.load(source, force="scene", process=False)
    scene = loaded if isinstance(loaded, trimesh.Scene) else trimesh.Scene(loaded)
    if not scene.geometry:
        raise RuntimeError(f"empty mesh: {source}")
    scene.apply_transform(matrix)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.suffix.lower() == ".glb":
        destination.write_bytes(bytes(scene.export(file_type="glb")))
    else:
        scene.export(destination)


def projection_overlays(
    *, dataset: Path, mesh_path: Path, frame_names: Sequence[str], output_dir: Path
) -> list[str]:
    import trimesh

    cameras = parse_cameras(dataset / "sparse/0/cameras.txt")
    rows = {
        str(row["name"]): row
        for row in parse_registered_images(dataset / "sparse/0/images.txt")
    }
    loaded = trimesh.load(mesh_path, force="scene", process=False)
    mesh = loaded.dump(concatenate=True)
    points = np.asarray(
        mesh.sample(min(60000, max(10000, len(mesh.vertices) * 3))), dtype=np.float64
    )
    outputs = []
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in frame_names:
        row = rows[name]
        K, _ = camera_intrinsics(cameras[int(row["camera_id"])])
        rotation = qvec_to_rotation(row["qvec"])
        camera_points = points @ rotation.T + np.asarray(row["tvec"])
        camera_points = camera_points[camera_points[:, 2] > 1.0e-6]
        uvw = camera_points @ K.T
        uv = uvw[:, :2] / uvw[:, 2:3]
        with Image.open(dataset / "images" / name) as handle:
            image = np.asarray(handle.convert("RGB")).copy()
        x = np.rint(uv[:, 0]).astype(np.int64)
        y = np.rint(uv[:, 1]).astype(np.int64)
        inside = (x >= 0) & (x < image.shape[1]) & (y >= 0) & (y < image.shape[0])
        x, y = x[inside], y[inside]
        image[y, x] = np.asarray([255, 32, 32], dtype=np.uint8)
        image[np.clip(y + 1, 0, image.shape[0] - 1), x] = np.asarray(
            [255, 200, 32], dtype=np.uint8
        )
        destination = output_dir / f"{Path(name).stem}_mesh_projection.png"
        Image.fromarray(image).save(destination)
        outputs.append(str(destination.resolve()))
    return outputs


def reconstruct(
    *, output_dir: Path, gpu: str, python: Path, session_id: str
) -> dict[str, Any]:
    report_path = output_dir / "03_reconstruction/final_report.json"
    if report_path.is_file():
        report = read_json(report_path)
        if report.get("passed") is True:
            print(f"[ar_full_colmap] reuse reconstruction: {report_path}")
            return report
    colmap = read_json(output_dir / "02_colmap/colmap_object_report.json")
    if colmap.get("passed") is not True:
        raise RuntimeError("COLMAP stage has not passed")
    gravity = tuple(
        float(value) for value in colmap["colmap_to_ar"]["gravity_up_colmap"]
    )
    reconstruction_root = output_dir / "03_reconstruction"
    config = ARReconstructionConfig(
        python=python.resolve(),
        gpu=str(gpu),
        selected_view_count=8,
        min_object_points=100,
        min_mask_observations=2,
        min_mask_support_ratio=0.35,
        view_selection_policy="object_spherical_farthest_valid_mask",
        gravity_up_w=gravity,
        seed=42,
    )
    result = run_ar_object_reconstruction(
        session_id=session_id,
        dataset_dir=Path(colmap["object_dataset"]),
        output_root=reconstruction_root,
        config=config,
    )
    matrix = np.asarray(
        colmap["colmap_to_ar"]["T_colmap_to_ar"], dtype=np.float64
    )
    final_dir = reconstruction_root / "final_ar_world"
    ar_obj = final_dir / "reconstructed_object_ar_world.obj"
    ar_glb = final_dir / "reconstructed_object_ar_world.glb"
    transform_mesh(Path(result["meshes"]["world_obj"]), ar_obj, matrix)
    transform_mesh(Path(result["meshes"]["world_obj"]), ar_glb, matrix)

    paths = ReconstructionPaths.build(reconstruction_root.resolve(), session_id)
    runtime_manifest = read_json(paths.runtime_dir / "runtime_input_manifest.json")
    raw_manifest = read_json(paths.raw_dir / "raw_cache_report.json")
    runtime_object = runtime_manifest["objects"][0]
    raw_object = raw_manifest["objects"][0]
    source_names = [
        row["source_frame_name"] for row in raw_object["source_binding"]["frames"]
    ]
    selected_names = [
        source_names[int(index)]
        for index in runtime_object["selected_source_view_indices"]
    ]
    overlays = projection_overlays(
        dataset=Path(colmap["object_dataset"]),
        mesh_path=Path(result["meshes"]["world_obj"]),
        frame_names=selected_names,
        output_dir=final_dir / "selected8_projection_audit",
    )
    report = {
        "format": FORMAT,
        "stage": "mixed_no_vggt_reconstruction_and_ar_world_export",
        "created_at_utc": utc_now(),
        "source_colmap_report": str(
            (output_dir / "02_colmap/colmap_object_report.json").resolve()
        ),
        "reconstruction_report": str(
            (paths.run_dir / "reconstruction_report.json").resolve()
        ),
        "selected_inference_frames": selected_names,
        "mesh_colmap_world_obj": result["meshes"]["world_obj"],
        "mesh_ar_world_obj": str(ar_obj.resolve()),
        "mesh_ar_world_glb": str(ar_glb.resolve()),
        "projection_overlays": overlays,
        "coordinate_transform": colmap["colmap_to_ar"],
        "scope_guard": (
            "COLMAP uses all 96 RGB frames for camera/BA; the deployed model consumes "
            "the audited best eight registered, jointly undistorted RGB/mask views."
        ),
        "passed": True,
    }
    write_json(report_path, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    sub = root.add_subparsers(dest="command", required=True)
    masks = sub.add_parser("masks")
    masks.add_argument("--raw_session", type=Path, default=DEFAULT_RAW)
    masks.add_argument("--seed_dataset", type=Path, default=DEFAULT_SEEDS)
    masks.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    masks.add_argument("--expected_frames", type=int, default=96)

    colmap = sub.add_parser("colmap")
    colmap.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    colmap.add_argument("--colmap_bin", type=Path, default=DEFAULT_COLMAP)
    colmap.add_argument("--gpu", default="4")
    colmap.add_argument("--expected_frames", type=int, default=96)
    colmap.add_argument("--min_registration_ratio", type=float, default=0.85)
    colmap.add_argument("--min_track_observations", type=int, default=2)
    colmap.add_argument("--min_positive_observations", type=int, default=2)
    colmap.add_argument("--min_positive_ratio", type=float, default=0.60)
    colmap.add_argument("--max_reprojection_error", type=float, default=4.0)
    colmap.add_argument("--mask_dilation", type=int, default=3)
    colmap.add_argument("--min_object_points", type=int, default=256)
    colmap.add_argument("--resume", action="store_true")

    mesh = sub.add_parser("reconstruct")
    mesh.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    mesh.add_argument("--gpu", default="4")
    mesh.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    mesh.add_argument(
        "--session_id", default="20260810_133235_817_full96_colmap_v1"
    )
    return root


def main() -> None:
    args = parser().parse_args()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.command == "masks":
        prepare_masks(
            raw_session=args.raw_session.expanduser().resolve(),
            seed_dataset=args.seed_dataset.expanduser().resolve(),
            output_dir=args.output_dir,
            expected_frames=args.expected_frames,
        )
    elif args.command == "colmap":
        prepare_colmap(
            output_dir=args.output_dir,
            colmap_bin=args.colmap_bin.expanduser().resolve(),
            gpu=str(args.gpu),
            expected_frames=args.expected_frames,
            min_registration_ratio=args.min_registration_ratio,
            min_track_observations=args.min_track_observations,
            min_positive_observations=args.min_positive_observations,
            min_positive_ratio=args.min_positive_ratio,
            max_reprojection_error=args.max_reprojection_error,
            mask_dilation=args.mask_dilation,
            min_object_points=args.min_object_points,
            resume=bool(args.resume),
        )
    else:
        reconstruct(
            output_dir=args.output_dir,
            gpu=str(args.gpu),
            python=args.python.expanduser().resolve(),
            session_id=str(args.session_id),
        )


if __name__ == "__main__":
    main()
