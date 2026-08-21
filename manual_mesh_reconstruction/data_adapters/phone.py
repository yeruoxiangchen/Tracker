#!/usr/bin/env python3
"""Adapter for frozen ARFoundation/phone captures.

Accepted inputs include a completed reconstruction directory, a finalized
capture dataset, ``runtime/data/<session>``, ``runtime/masks/<session>``, or the
``runtime`` root together with ``--session-id``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image

from manual_mesh_reconstruction.common import load_json, sha256_file
from manual_mesh_reconstruction.data_adapters.common import (
    CameraFrame,
    deferred_selection_request,
    indexed_media,
    materialize_raw_cache,
    natural_key,
    safe_id,
)


def _parse_float(value: Any) -> float | None:
    try:
        parsed = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _parse_int(value: Any) -> int | None:
    parsed = _parse_float(value)
    return None if parsed is None else int(round(parsed))


def read_phone_poses(path: Path) -> dict[str, dict[str, Any]]:
    poses: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        raise FileNotFoundError(f"phone poses are missing: {path}")
    for line_number, raw in enumerate(
        path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1
    ):
        fields = [value.strip() for value in raw.split(",")]
        if not fields or fields[0] == "frame_name" or len(fields) < 7:
            continue
        position = [_parse_float(value) for value in fields[1:4]]
        euler = [_parse_float(value) for value in fields[4:7]]
        if any(value is None for value in position):
            continue
        row: dict[str, Any] = {
            "frame_name": Path(fields[0]).name,
            "position": np.asarray(position, dtype=np.float64),
            "euler": (
                None
                if any(value is None for value in euler)
                else np.asarray(euler, dtype=np.float64)
            ),
            "quaternion": None,
            "intrinsics": None,
            "image_transform": "unknown",
            "screen_orientation": "unknown",
            "display_matrix": None,
            "pose_binding": "legacy_unversioned",
            "timestamp_delta_seconds": None,
        }
        if len(fields) >= 11:
            quaternion = [_parse_float(value) for value in fields[7:11]]
            if not any(value is None for value in quaternion):
                row["quaternion"] = np.asarray(quaternion, dtype=np.float64)
        if len(fields) >= 15:
            values = [_parse_float(value) for value in fields[11:15]]
            if not any(value is None for value in values):
                row["intrinsics"] = {
                    "fx": values[0],
                    "fy": values[1],
                    "cx": values[2],
                    "cy": values[3],
                    "width": _parse_int(fields[15]) if len(fields) > 15 else None,
                    "height": _parse_int(fields[16]) if len(fields) > 16 else None,
                    "image_width": _parse_int(fields[17]) if len(fields) > 17 else None,
                    "image_height": _parse_int(fields[18]) if len(fields) > 18 else None,
                    "cpu_image_width": _parse_int(fields[19]) if len(fields) > 19 else None,
                    "cpu_image_height": _parse_int(fields[20]) if len(fields) > 20 else None,
                }
        if len(fields) > 21:
            row["image_transform"] = fields[21] or "None"
        elif len(fields) >= 19:
            row["image_transform"] = "MirrorY"
        if len(fields) > 25:
            row["timestamp_delta_seconds"] = _parse_float(fields[25])
        if len(fields) > 26 and fields[26]:
            row["pose_binding"] = fields[26]
        if len(fields) > 27 and fields[27]:
            row["screen_orientation"] = fields[27]
        if len(fields) > 29 and fields[29]:
            row["display_matrix"] = fields[29]
        poses[row["frame_name"]] = row
    if not poses:
        raise RuntimeError(f"no usable phone pose rows: {path}")
    return poses


def _nearest_rotation(matrix: np.ndarray) -> np.ndarray:
    left, _singular, right = np.linalg.svd(np.asarray(matrix, dtype=np.float64))
    rotation = left @ right
    if float(np.linalg.det(rotation)) < 0.0:
        left[:, -1] *= -1.0
        rotation = left @ right
    return rotation


def _unity_quaternion_to_rotation(quaternion_xyzw: Sequence[float]) -> np.ndarray:
    x, y, z, w = np.asarray(quaternion_xyzw, dtype=np.float64)
    norm = float(np.linalg.norm([x, y, z, w]))
    if norm <= 1.0e-12:
        raise ValueError("Unity quaternion has zero norm")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _unity_euler_to_rotation(rot_deg: Sequence[float]) -> np.ndarray:
    rx, ry, rz = np.deg2rad(np.asarray(rot_deg, dtype=np.float64))
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    rx_matrix = np.asarray([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry_matrix = np.asarray([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rz_matrix = np.asarray([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return ry_matrix @ rx_matrix @ rz_matrix


def _parse_display_matrix(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    fields = str(value).split()
    if len(fields) != 16:
        return None
    try:
        matrix = np.asarray([float(item) for item in fields], dtype=np.float64).reshape(4, 4)
    except ValueError:
        return None
    return matrix if np.isfinite(matrix).all() else None


def resolve_image_camera_rotation(
    poses: dict[str, dict[str, Any]], frame_names: Sequence[str]
) -> tuple[float, dict[str, Any]]:
    """Resolve pose-camera to saved CPU-image axes using frozen capture metadata."""

    rows = [poses[name] for name in frame_names]
    matrices = [
        matrix
        for matrix in (_parse_display_matrix(row.get("display_matrix")) for row in rows)
        if matrix is not None
    ]
    transforms = [str(row.get("image_transform") or "unknown") for row in rows]
    orientations = [str(row.get("screen_orientation") or "unknown") for row in rows]
    rotations: list[float] = []
    for matrix in matrices:
        left, _singular, right = np.linalg.svd(matrix[:2, :2])
        rotation = left @ right
        if float(np.linalg.det(rotation)) < 0.0:
            left[:, -1] *= -1.0
            rotation = left @ right
        rotations.append(float(np.degrees(np.arctan2(rotation[1, 0], rotation[0, 0]))))
    complete = bool(
        rows
        and len(matrices) == len(rows)
        and all(value.lower() == "none" for value in transforms)
    )
    if complete:
        if len(set(orientations)) != 1:
            raise RuntimeError("phone screen orientation changed within selected capture")
        def circular_error(value: float, reference: float) -> float:
            return abs((value - reference + 180.0) % 360.0 - 180.0)
        if max(circular_error(value, rotations[0]) for value in rotations) > 1.0:
            raise RuntimeError("phone display-matrix rotation changed within capture")
        cardinal = min((0.0, 90.0, -90.0, 180.0), key=lambda x: circular_error(rotations[0], x))
        if circular_error(rotations[0], cardinal) > 5.0:
            raise RuntimeError("phone display-matrix rotation is not near a cardinal axis")
        angle = float(cardinal)
        source = "xrcpuimage_none_direct_display_uv_v5"
    else:
        angle = 90.0
        source = "legacy_explicit_Rz(+90deg)_camera_axis_contract"
    return angle, {
        "passed": True,
        "source": source,
        "pose_camera_to_saved_image_rotation_degrees": angle,
        "selected_frame_count": len(frame_names),
        "display_matrix_count": len(matrices),
        "image_transforms": sorted(set(transforms)),
        "screen_orientations": sorted(set(orientations)),
    }


def phone_pose_to_w2c(pose: dict[str, Any], image_rotation_degrees: float) -> np.ndarray:
    if pose.get("quaternion") is not None:
        unity_c2w = _unity_quaternion_to_rotation(pose["quaternion"])
    elif pose.get("euler") is not None:
        unity_c2w = _unity_euler_to_rotation(pose["euler"])
    else:
        raise RuntimeError(f"phone pose has no rotation: {pose['frame_name']}")
    unity_to_cv_world = np.diag([1.0, 1.0, -1.0])
    unity_camera_to_cv_camera = np.diag([1.0, -1.0, 1.0])
    center = unity_to_cv_world @ np.asarray(pose["position"], dtype=np.float64)
    c2w = _nearest_rotation(unity_to_cv_world @ unity_c2w @ unity_camera_to_cv_camera)
    w2c = c2w.T
    translation = -w2c @ center
    angle = math.radians(float(image_rotation_degrees))
    image_from_pose = np.asarray(
        [
            [math.cos(angle), -math.sin(angle), 0.0],
            [math.sin(angle), math.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    output = np.eye(4, dtype=np.float64)
    output[:3, :3] = image_from_pose @ w2c
    output[:3, 3] = image_from_pose @ translation
    return output


def intrinsics_for_pose(
    pose: dict[str, Any], image_width: int, image_height: int
) -> tuple[np.ndarray, str]:
    intrinsics = pose.get("intrinsics")
    if intrinsics is not None:
        fx, fy = float(intrinsics["fx"]), float(intrinsics["fy"])
        cx, cy = float(intrinsics["cx"]), float(intrinsics["cy"])
        source_width = (
            intrinsics.get("cpu_image_width")
            or intrinsics.get("width")
            or intrinsics.get("image_width")
            or image_width
        )
        source_height = (
            intrinsics.get("cpu_image_height")
            or intrinsics.get("height")
            or intrinsics.get("image_height")
            or image_height
        )
        fx *= image_width / float(source_width)
        fy *= image_height / float(source_height)
        cx *= image_width / float(source_width)
        cy *= image_height / float(source_height)
        transform = str(pose.get("image_transform") or "None").lower()
        if "mirrorx" in transform:
            cx = (image_width - 1) - cx
        if "mirrory" in transform:
            cy = (image_height - 1) - cy
        if fx > 0 and fy > 0 and 0 <= cx <= image_width and 0 <= cy <= image_height:
            return np.asarray(
                [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
                dtype=np.float64,
            ), f"ar_foundation_{pose.get('image_transform') or 'None'}"
    focal = float(max(image_width, image_height))
    return np.asarray(
        [
            [focal, 0.0, (image_width - 1) * 0.5],
            [0.0, focal, (image_height - 1) * 0.5],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    ), "fallback_image_size"


def _raw_report_candidates(path: Path) -> list[Path]:
    candidates = []
    if path.is_file() and path.name == "raw_cache_report.json":
        candidates.append(path)
    if path.is_dir():
        candidates.extend(
            [
                path / "raw_cache_report.json",
                path / "01_raw_cache/raw_cache_report.json",
            ]
        )
        reconstruction = path / "reconstruction_report.json"
        if reconstruction.is_file():
            report = load_json(reconstruction)
            raw = report.get("stage_reports", {}).get("raw_cache")
            if raw:
                candidates.append(Path(str(raw)))
    return [candidate.expanduser().resolve() for candidate in candidates]


def _frames_from_raw_report(path: Path) -> tuple[list[CameraFrame], dict[str, Any]]:
    payload = load_json(path)
    if payload.get("passed") is not True or len(payload.get("objects", [])) != 1:
        raise RuntimeError(f"phone source raw report must contain one passed object: {path}")
    row = payload["objects"][0]
    cache = Path(row["cache_npz"]).resolve(strict=True)
    cameras = {str(value["frame_name"]): value for value in row.get("cameras", [])}
    with np.load(cache, allow_pickle=False) as arrays:
        names = [str(value) for value in arrays["frame_name"].tolist()]
        source_names = (
            [str(value) for value in arrays["source_frame_name"].tolist()]
            if "source_frame_name" in arrays.files
            else names
        )
        K = np.asarray(arrays["K"], dtype=np.float64)
        poses = np.asarray(arrays["T_W2C"], dtype=np.float64)
    frames = []
    for index, (name, source_name) in enumerate(zip(names, source_names)):
        camera = cameras.get(name, {})
        frames.append(
            CameraFrame(
                source_index=index,
                source_name=source_name,
                image_path=Path(row["images_dir"]) / name,
                mask_path=Path(row["masks_dir"]) / name,
                K=K[index],
                T_W2C=poses[index],
                camera_model=str(camera.get("model", "PINHOLE")),
                distortion=tuple(float(value) for value in camera.get("distortion", [])),
                pose_source=str(camera.get("pose_source", "frozen_raw_cache")),
            )
        )
    return frames, {
        "source_kind": "existing_raw_cache_report",
        "raw_cache_report": str(path),
        "raw_cache_report_sha256": sha256_file(path),
        "source_cache": str(cache),
        "source_cache_sha256": sha256_file(cache),
        "source_object_key": row.get("object_key"),
    }


def _resolve_runtime_paths(
    input_path: Path, session_id: str | None
) -> tuple[str, Path, Path, Path]:
    path = input_path.resolve(strict=True)
    if (path / "poses.txt").is_file():
        data_dir = path
        session = session_id or path.name
        if path.parent.name == "data":
            mask_dir = path.parent.parent / "masks" / session
        elif (path / "masks").is_dir():
            mask_dir = path / "masks"
        else:
            mask_dir = path.parent / "masks" / session
        return session, data_dir, mask_dir.resolve(strict=True), data_dir / "poses.txt"
    if path.name == "runtime" or ((path / "data").is_dir() and (path / "masks").is_dir()):
        if not session_id:
            raise ValueError("passing a runtime root requires --session-id")
        data_dir = (path / "data" / session_id).resolve(strict=True)
        return (
            session_id,
            data_dir,
            (path / "masks" / session_id).resolve(strict=True),
            data_dir / "poses.txt",
        )
    if path.parent.name == "masks":
        session = session_id or path.name
        data_dir = (path.parent.parent / "data" / session).resolve(strict=True)
        return session, data_dir, path, data_dir / "poses.txt"
    # Finalized capture dataset.
    for image_name in ("images", "rgb"):
        if (path / image_name).is_dir() and (path / "masks").is_dir() and (path / "poses.txt").is_file():
            return session_id or path.name, path / image_name, path / "masks", path / "poses.txt"
    raise RuntimeError(f"cannot resolve phone capture input: {input_path}")


def _frames_from_runtime(
    input_path: Path, session_id: str | None
) -> tuple[str, list[CameraFrame], dict[str, Any]]:
    session, data_dir, mask_dir, poses_path = _resolve_runtime_paths(input_path, session_id)
    poses = read_phone_poses(poses_path)
    images = indexed_media(data_dir)
    masks = indexed_media(mask_dir)
    matched = sorted(set(images) & set(masks), key=natural_key)
    names = [poses[name]["frame_name"] for name in sorted(poses, key=natural_key) if Path(name).stem in matched]
    # Above maps through stems because masks normally use PNG while RGB uses JPG.
    frames: list[CameraFrame] = []
    if len(names) < 1:
        raise RuntimeError("phone capture has no RGB/mask/pose intersection")
    angle, axis = resolve_image_camera_rotation(poses, names)
    intrinsics_sources: dict[str, int] = {}
    for source_index, name in enumerate(names):
        stem = Path(name).stem
        image_path = images[stem]
        mask_path = masks[stem]
        with Image.open(image_path) as image:
            width, height = image.size
        K, intrinsics_source = intrinsics_for_pose(poses[name], width, height)
        intrinsics_sources[intrinsics_source] = intrinsics_sources.get(intrinsics_source, 0) + 1
        frames.append(
            CameraFrame(
                source_index=source_index,
                source_name=name,
                image_path=image_path,
                mask_path=mask_path,
                K=K,
                T_W2C=phone_pose_to_w2c(poses[name], angle),
                camera_model="PINHOLE",
                distortion=(),
                pose_source="synchronized_ar_foundation_camera_pose",
            )
        )
    metadata = data_dir / "frame_metadata.jsonl"
    return session, frames, {
        "source_kind": "ar_foundation_runtime_capture",
        "session_id": session,
        "data_dir": str(data_dir.resolve()),
        "mask_dir": str(mask_dir.resolve()),
        "poses": str(poses_path.resolve()),
        "poses_sha256": sha256_file(poses_path),
        "frame_metadata": str(metadata.resolve()) if metadata.is_file() else None,
        "frame_metadata_sha256": sha256_file(metadata) if metadata.is_file() else None,
        "image_camera_axis_contract": axis,
        "intrinsics_sources": intrinsics_sources,
        "gravity_up_W": [0.0, 1.0, 0.0],
    }


def adapt(
    *,
    input_path: Path,
    output_dir: Path,
    selected_view_count: int,
    selection_policy: str,
    random_seed: int,
    session_id: str | None,
) -> dict[str, Any]:
    input_path = input_path.expanduser().resolve(strict=True)
    frames: list[CameraFrame] | None = None
    source_binding: dict[str, Any] | None = None
    object_id = session_id or input_path.name
    for candidate in _raw_report_candidates(input_path):
        if candidate.is_file():
            frames, source_binding = _frames_from_raw_report(candidate)
            if not session_id:
                object_id = safe_id(input_path.name, label="phone session")
            break
    if frames is None or source_binding is None:
        object_id, frames, source_binding = _frames_from_runtime(input_path, session_id)
    selection = deferred_selection_request(
        len(frames),
        int(selected_view_count),
        policy=str(selection_policy),
        random_seed=int(random_seed),
    )
    selection["source_frame_names"] = [frame.source_name for frame in frames]
    raw_path, row = materialize_raw_cache(
        output_dir=output_dir,
        dataset_type="phone",
        source_path=input_path,
        category="phone_capture",
        object_id=object_id,
        input_frames=frames,
        selection_request=selection,
        source_binding=source_binding,
        extra_report={
            "gravity_up_W": [0.0, 1.0, 0.0],
            "phone_pose_consumed": True,
            "colmap_executed": False,
        },
    )
    return {
        "raw_cache_report": str(raw_path.resolve()),
        "raw_cache_report_sha256": sha256_file(raw_path),
        "runtime_input_manifest": None,
        "runtime_input_manifest_sha256": None,
        "object_key": row["object_key"],
        "geometry_mode": "pose_mask",
        "gravity_up_W": [0.0, 1.0, 0.0],
        "selection": row["view_selection"],
        "source_binding": source_binding,
    }
