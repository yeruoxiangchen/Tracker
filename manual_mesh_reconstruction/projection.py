"""Physical runtime-O/world/original-camera Mesh projection helpers.

This module deliberately owns the projection path used by manual tests.  The
decoded Mesh stays in the TRELLIS native sparse-grid/runtime-O axes; the only
physical transform applied before camera projection is ``T_O2W``.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
from PIL import Image
import trimesh

from manual_mesh_reconstruction.canonicalization import (
    array_sha256,
    validate_proper_similarity,
)
from manual_mesh_reconstruction.common import load_json, sha256_file


PROJECTION_CHAIN_TOLERANCE = 1.0e-6


def load_meshes(path: Path) -> list[trimesh.Trimesh]:
    loaded = trimesh.load(path, force="scene", process=False)
    if isinstance(loaded, trimesh.Trimesh):
        meshes = [loaded]
    elif isinstance(loaded, trimesh.Scene):
        meshes = [
            mesh
            for mesh in loaded.dump(concatenate=False)
            if isinstance(mesh, trimesh.Trimesh)
            and len(mesh.vertices) > 0
            and len(mesh.faces) > 0
        ]
    else:
        raise TypeError(f"unsupported Mesh type={type(loaded)}: {path}")
    if not meshes:
        raise RuntimeError(f"Mesh contains no triangles: {path}")
    return meshes


def validate_world_projection_chain(
    T_O2W: np.ndarray,
    T_W2C: np.ndarray,
    stored_T_O2C: np.ndarray,
    *,
    tolerance: float = PROJECTION_CHAIN_TOLERANCE,
) -> dict[str, Any]:
    object_to_world = validate_proper_similarity(T_O2W, name="T_O2W")
    world_to_camera = np.asarray(T_W2C, dtype=np.float64)
    stored = np.asarray(stored_T_O2C, dtype=np.float64)
    if world_to_camera.ndim != 3 or world_to_camera.shape[1:] != (4, 4):
        raise ValueError("T_W2C must be [V,4,4]")
    if stored.shape != world_to_camera.shape:
        raise ValueError("stored T_O2C must match T_W2C")
    composed = np.matmul(world_to_camera, object_to_world[None])
    max_abs = float(np.max(np.abs(composed - stored)))
    if not math.isfinite(max_abs) or max_abs > float(tolerance):
        raise RuntimeError(
            "physical world projection chain mismatch: "
            f"max_abs={max_abs} tolerance={tolerance}"
        )
    return {
        "formula": "T_O2C_physical = T_W2C @ T_O2W",
        "max_abs": max_abs,
        "tolerance": float(tolerance),
        "passed": True,
    }


def _frame_asset(directory: Path, frame_name: str) -> Path:
    exact = directory / frame_name
    if exact.is_file():
        return exact.resolve()
    matches = sorted(
        path
        for path in directory.glob(f"{Path(frame_name).stem}.*")
        if path.is_file()
    )
    if len(matches) != 1:
        raise FileNotFoundError(
            f"no unique raw asset for frame={frame_name} directory={directory}"
        )
    return matches[0].resolve()


def load_projection_contract(runtime_report_path: Path, object_key: str) -> dict[str, Any]:
    runtime_report_path = runtime_report_path.expanduser().resolve(strict=True)
    runtime_report = load_json(runtime_report_path)
    if runtime_report.get("passed") is not True:
        raise RuntimeError(f"runtime object report did not pass: {runtime_report_path}")
    if str(runtime_report.get("object_key")) != str(object_key):
        raise RuntimeError("runtime report object identity differs")
    runtime_cache_path = Path(runtime_report["cache_npz"]).resolve(strict=True)
    raw_cache_path = Path(runtime_report["source_raw_cache"]).resolve(strict=True)
    with np.load(runtime_cache_path, allow_pickle=False) as runtime:
        selected = np.asarray(runtime["selected_source_view_index"], dtype=np.int64)
        frame_names = np.asarray(runtime["frame_name"]).astype(str)
        feature_intrinsics = np.asarray(runtime["K_feature"], dtype=np.float64)
        stored_T_O2C = np.asarray(runtime["T_O2C"], dtype=np.float64)
        T_O2W = np.asarray(runtime["T_O2W"], dtype=np.float64)
        T_W2O = np.asarray(runtime["T_W2O"], dtype=np.float64)
    with np.load(raw_cache_path, allow_pickle=False) as raw:
        raw_frame_names = np.asarray(raw["frame_name"]).astype(str)
        T_W2C = np.asarray(raw["T_W2C"][selected], dtype=np.float64)
        raw_intrinsics = np.asarray(raw["K"][selected], dtype=np.float64)
    if not np.array_equal(raw_frame_names[selected], frame_names):
        raise RuntimeError("selected raw camera rows do not match runtime frame names")

    raw_report_path = raw_cache_path.parents[3] / "raw_cache_report.json"
    raw_report = load_json(raw_report_path)
    raw_rows = [
        row
        for row in raw_report["objects"]
        if f'{row["category"]}:{row["object_id"]}' == str(object_key)
    ]
    if len(raw_rows) != 1:
        raise RuntimeError(f"raw-cache report has no unique object row: {object_key}")
    raw_row = raw_rows[0]
    image_dir = Path(raw_row["images_dir"]).resolve(strict=True)
    mask_dir = Path(raw_row["masks_dir"]).resolve(strict=True)
    images = [_frame_asset(image_dir, name) for name in frame_names]
    masks = [_frame_asset(mask_dir, name) for name in frame_names]
    camera_by_frame = {str(row["frame_name"]): row for row in raw_row["cameras"]}
    cameras = []
    for name in frame_names:
        if name not in camera_by_frame:
            raise RuntimeError(f"raw camera metadata is missing frame={name}")
        camera = camera_by_frame[name]
        model = str(camera["model"])
        if model not in {"SIMPLE_RADIAL", "SIMPLE_PINHOLE", "PINHOLE"}:
            raise RuntimeError(f"unsupported raw projection camera model={model}")
        cameras.append(
            {
                "model": model,
                "distortion": [float(value) for value in camera.get("distortion", [])],
                "width": int(camera["width"]),
                "height": int(camera["height"]),
            }
        )
    for image, camera in zip(images, cameras):
        with Image.open(image) as payload:
            if payload.size != (camera["width"], camera["height"]):
                raise RuntimeError(f"raw image/camera size differs: {image}")

    prepared_images = [Path(value).resolve(strict=True) for value in runtime_report["prepared_rgb_paths"]]
    prepared_masks = [Path(value).resolve(strict=True) for value in runtime_report["prepared_mask_paths"]]
    view_count = len(images)
    if not (
        view_count
        == len(masks)
        == len(raw_intrinsics)
        == len(feature_intrinsics)
        == len(T_W2C)
        == len(stored_T_O2C)
        == len(frame_names)
        == len(cameras)
        == len(prepared_images)
        == len(prepared_masks)
    ):
        raise RuntimeError("runtime image/camera/projection view counts differ")
    inverse_roundtrip = float(np.max(np.abs(T_O2W @ T_W2O - np.eye(4))))
    if inverse_roundtrip > 1.0e-7:
        raise RuntimeError(f"T_O2W/T_W2O roundtrip failed: {inverse_roundtrip}")
    chain = validate_world_projection_chain(T_O2W, T_W2C, stored_T_O2C)
    return {
        "runtime_report": runtime_report_path,
        "runtime_cache": runtime_cache_path,
        "raw_cache": raw_cache_path,
        "images": images,
        "masks": masks,
        "prepared_images": prepared_images,
        "prepared_masks": prepared_masks,
        "frame_names": frame_names.tolist(),
        "intrinsics": raw_intrinsics,
        "feature_intrinsics": feature_intrinsics,
        "cameras": cameras,
        "T_W2C": T_W2C,
        "T_O2W": T_O2W,
        "inverse_roundtrip_max_abs": inverse_roundtrip,
        "chain_audit": chain,
        "bindings": {
            "runtime_report_sha256": sha256_file(runtime_report_path),
            "runtime_cache_sha256": sha256_file(runtime_cache_path),
            "raw_cache_sha256": sha256_file(raw_cache_path),
            "raw_cache_report": str(raw_report_path),
            "raw_cache_report_sha256": sha256_file(raw_report_path),
            "T_O2W_sha256": array_sha256(T_O2W),
            "T_W2C_sha256": array_sha256(T_W2C),
            "K_raw_sha256": array_sha256(raw_intrinsics),
            "K_feature_sha256": array_sha256(feature_intrinsics),
            "raw_camera_models": [camera["model"] for camera in cameras],
            "raw_distortion": [camera["distortion"] for camera in cameras],
        },
    }


def world_mesh_buffers(mesh_path: Path, T_O2W: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    transform = validate_proper_similarity(T_O2W, name="T_O2W")
    vertices_all: list[np.ndarray] = []
    faces_all: list[np.ndarray] = []
    offset = 0
    for mesh in load_meshes(mesh_path):
        vertices_o = np.asarray(mesh.vertices, dtype=np.float64)
        vertices_w = (
            vertices_o @ transform[:3, :3].T + transform[:3, 3][None]
        ).astype(np.float32)
        faces = np.asarray(mesh.faces, dtype=np.int32) + offset
        vertices_all.append(vertices_w)
        faces_all.append(faces)
        offset += len(vertices_w)
    return np.concatenate(vertices_all), np.concatenate(faces_all)


def _apply_camera_distortion(x, y, camera: dict[str, Any]):
    model = str(camera["model"])
    distortion = [float(value) for value in camera.get("distortion", [])]
    if model == "SIMPLE_RADIAL":
        if len(distortion) != 1:
            raise RuntimeError("SIMPLE_RADIAL requires exactly one distortion value")
        factor = 1.0 + distortion[0] * (x * x + y * y)
        return x * factor, y * factor
    if model in {"SIMPLE_PINHOLE", "PINHOLE"}:
        return x, y
    raise RuntimeError(f"unsupported raw projection camera model={model}")


def rasterize_world_silhouette(
    vertices_w: np.ndarray,
    faces: np.ndarray,
    T_W2C: np.ndarray,
    intrinsic: np.ndarray,
    camera: dict[str, Any],
    image_shape: tuple[int, int],
    rastctx,
) -> np.ndarray:
    import torch
    import nvdiffrast.torch as dr

    height, width = (int(image_shape[0]), int(image_shape[1]))
    device = torch.device("cuda")
    verts = torch.as_tensor(vertices_w, dtype=torch.float32, device=device)
    faces_t = torch.as_tensor(faces, dtype=torch.int32, device=device)
    w2c = torch.as_tensor(T_W2C, dtype=torch.float32, device=device)
    K = torch.as_tensor(intrinsic, dtype=torch.float32, device=device)
    verts_h = torch.cat(
        [verts, torch.ones((len(verts), 1), device=device, dtype=torch.float32)],
        dim=1,
    )
    cam = (verts_h @ w2c.T)[:, :3]
    z_raw = cam[:, 2]
    positive = z_raw > 1.0e-5
    if int(positive.sum().item()) < 3:
        raise RuntimeError("world Mesh is behind the selected camera")
    visible_faces = faces_t[positive[faces_t].all(dim=1)]
    if len(visible_faces) == 0:
        raise RuntimeError("world Mesh has no front-camera triangles")
    z = z_raw.clamp(min=1.0e-5)
    x, y = cam[:, 0] / z, cam[:, 1] / z
    x, y = _apply_camera_distortion(x, y, camera)
    u = K[0, 0] * x + K[0, 1] * y + K[0, 2]
    v = K[1, 1] * y + K[1, 2]
    x_ndc = 2.0 * u / max(width - 1, 1) - 1.0
    y_ndc = 2.0 * v / max(height - 1, 1) - 1.0
    positive_depth = z_raw[positive]
    near = torch.clamp(positive_depth.min() * 0.5, min=1.0e-5)
    far = torch.maximum(positive_depth.max() * 1.5, near + 1.0e-3)
    z_ndc = 2.0 * (z - near) / (far - near) - 1.0
    pos_clip = torch.stack([x_ndc * z, y_ndc * z, z_ndc * z, z], dim=1)
    rast, _ = dr.rasterize(
        rastctx, pos_clip[None], visible_faces, resolution=(height, width)
    )
    return (rast[0, ..., 3].detach().cpu().numpy() > 0).astype(np.uint8)


def boundary(mask: np.ndarray, width: int = 2) -> np.ndarray:
    if int(width) < 1:
        raise ValueError("boundary width must be positive")
    value = np.asarray(mask, dtype=bool)
    for _ in range(int(width)):
        padded = np.pad(value, 1, mode="constant")
        dilated = np.logical_or.reduce(
            [
                padded[dy : dy + value.shape[0], dx : dx + value.shape[1]]
                for dy in range(3)
                for dx in range(3)
            ]
        )
        value = dilated
    return value ^ np.asarray(mask, dtype=bool)


def make_headless_raster_context():
    environment_bin = str(Path(sys.executable).resolve().parent)
    path_entries = os.environ.get("PATH", "").split(os.pathsep)
    if environment_bin not in path_entries:
        os.environ["PATH"] = environment_bin + os.pathsep + os.environ.get("PATH", "")
    os.environ.setdefault("TORCH_EXTENSIONS_DIR", "/tmp/torch_extensions")
    try:
        import nvdiffrast.torch as dr
        import torch
    except Exception as error:
        raise RuntimeError("camera projection requires torch+nvdiffrast") from error
    if not torch.cuda.is_available():
        raise RuntimeError("camera projection requires CUDA")
    return dr.RasterizeCudaContext()
