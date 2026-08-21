#!/usr/bin/env python3
"""Build a selected-case appearance, geometry and world-projection Mesh review bundle.

The bundle is display-only. It never changes an inference Mesh, GT alignment,
metric, checkpoint, or the frozen holdout. Cases may be selected post hoc by a
documented aggregate score and/or sampled with a fixed seed.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
from types import SimpleNamespace
from typing import Any

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import trimesh

from pose_point_depth_mv.dataset_tools.align_omni_real_mesh_to_colmap import (
    transform_obj_geometry,
)
from pose_point_depth_mv.dataset_tools.build_objaverse_multiview_sparse_data import (
    build_render_buffers,
    load_meshes,
    make_camera_ring,
    make_nvdiffrast_context,
    render_mesh_rgb,
)
from pose_point_depth_mv.omni_real_benchmark_common import (
    atomic_json,
    load_json,
    sha256_file,
)
from pose_point_depth_mv.real_object_canonicalization import (
    array_sha256,
    validate_proper_similarity,
)


TRACKER_ROOT = Path(__file__).resolve().parents[1]
FORMAT = "pose_point_depth_mv.omni_real_holdout_mesh_review.v5"
EXPECTED_REPORT_FORMAT = "pose_point_depth_mv.omni_real_no_vggt_final_benchmark.v1"
GAUSSIAN_APPEARANCE_REPORT_FORMAT = (
    "pose_point_depth_mv.native_no_vggt_gaussian_appearance.v1"
)
GAUSSIAN_APPEARANCE_OBJECT_FORMAT = (
    "pose_point_depth_mv.native_no_vggt_gaussian_appearance_object.v1"
)
METHOD_ORDER = (
    "gt",
    "final_native_no_vggt",
    "real_adapted_native_v2_full",
    "synthetic_parent_native_v2_full",
    "reconviagen_original",
    "pixal3d_official",
)
CURRENT_METHOD = "final_native_no_vggt"
METHOD_LABELS = {
    "gt": "GT Scan",
    "final_native_no_vggt": "Current No-VGGT",
    "real_adapted_native_v2_full": "Real-adapted Full",
    "synthetic_parent_native_v2_full": "Synthetic Full",
    "reconviagen_original": "ReconViaGen",
    "pixal3d_official": "Pixal3D",
}
METHOD_DIRS = {
    "gt": "01_GT真实扫描",
    "final_native_no_vggt": "02_当前NoVGGT",
    "real_adapted_native_v2_full": "03_真实域Full",
    "synthetic_parent_native_v2_full": "04_合成域Full",
    "reconviagen_original": "05_ReconViaGen",
    "pixal3d_official": "06_Pixal3D",
}
RANK_WEIGHTS = {
    "chamfer_l1": 0.40,
    "fscore_0p02": 0.30,
    "normal_consistency": 0.20,
    "largest_component_ratio": 0.10,
}
LOWER_IS_BETTER = {"chamfer_l1"}
PROJECTION_CHAIN_TOLERANCE = 1.0e-6
PROJECTED_FILL_ALPHA = 0.26
INPUT_MASK_COLOR = np.asarray([255, 214, 10], dtype=np.float32)
PROJECTION_COLORS = {
    "gt": np.asarray([80, 230, 120], dtype=np.float32),
    "final_native_no_vggt": np.asarray([40, 220, 255], dtype=np.float32),
    "real_adapted_native_v2_full": np.asarray([255, 105, 180], dtype=np.float32),
    "synthetic_parent_native_v2_full": np.asarray([190, 120, 255], dtype=np.float32),
    "reconviagen_original": np.asarray([255, 145, 45], dtype=np.float32),
    "pixal3d_official": np.asarray([235, 235, 235], dtype=np.float32),
}


def _safe_name(value: str) -> str:
    return value.replace(":", "_").replace("/", "_")


def _copy_bound(source: Path, destination: Path) -> dict[str, Any]:
    source = source.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if sha256_file(destination) != sha256_file(source):
            raise RuntimeError(f"existing copied asset differs: {destination}")
    else:
        shutil.copy2(source, destination)
    return {"path": str(destination), "sha256": sha256_file(destination)}


def _mesh_scene(path: Path) -> trimesh.Scene:
    loaded = trimesh.load(path, force="scene", process=False)
    if isinstance(loaded, trimesh.Trimesh):
        scene = trimesh.Scene(loaded)
    elif isinstance(loaded, trimesh.Scene):
        scene = loaded
    else:
        raise TypeError(f"unsupported Mesh type={type(loaded)}: {path}")
    pieces = [g for g in scene.geometry.values() if len(g.vertices) and len(g.faces)]
    if not pieces:
        raise RuntimeError(f"Mesh contains no triangles: {path}")
    return scene


def _appearance_kind(path: Path) -> str:
    scene = _mesh_scene(path)
    kinds = {str(getattr(mesh.visual, "kind", None)) for mesh in scene.geometry.values()}
    if "texture" in kinds:
        return "uv_texture"
    if "vertex" in kinds:
        return "vertex_color"
    return "geometry_only"


def _export_display_glb(source: Path, destination: Path) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.is_file():
        payload = _mesh_scene(source).export(file_type="glb")
        destination.write_bytes(payload)
    return {"path": str(destination), "sha256": sha256_file(destination)}


def _copy_obj_dependencies(source_obj: Path, destination_dir: Path) -> list[dict[str, Any]]:
    dependencies: list[dict[str, Any]] = []
    for source in sorted(source_obj.parent.iterdir()):
        if source.is_file() and source.suffix.lower() in {
            ".mtl", ".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"
        }:
            dependencies.append(_copy_bound(source, destination_dir / source.name))
    return dependencies


def _font(size: int) -> ImageFont.ImageFont:
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    return ImageFont.truetype(str(path), size=size) if path.is_file() else ImageFont.load_default()


def _opaque_rgb(path: Path, *, background=(145, 145, 145)) -> Image.Image:
    image = Image.open(path).convert("RGBA")
    base = Image.new("RGBA", image.size, (*background, 255))
    base.alpha_composite(image)
    return base.convert("RGB")


def _comparison_sheet(
    images: dict[str, list[Path]],
    destination: Path,
    *,
    view_count: int,
    method_order: tuple[str, ...] = METHOD_ORDER,
) -> None:
    first = _opaque_rgb(next(iter(images.values()))[0])
    cell_w, cell_h = first.size
    header = 34
    canvas = Image.new(
        "RGB", (cell_w * len(method_order), (cell_h + header) * view_count), (28, 28, 28)
    )
    draw = ImageDraw.Draw(canvas)
    font = _font(17)
    for column, method in enumerate(method_order):
        rows = images[method]
        if len(rows) != view_count:
            raise RuntimeError(f"render view count differs for {method}")
        for view, path in enumerate(rows):
            top = view * (cell_h + header)
            canvas.paste(_opaque_rgb(path), (column * cell_w, top + header))
            draw.text(
                (column * cell_w + 8, top + 8),
                METHOD_LABELS[method],
                fill=(242, 242, 242),
                font=font,
            )
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination)


def _per_view_comparison_sheets(
    images: dict[str, list[Path]],
    destination_dir: Path,
    *,
    view_count: int,
    method_order: tuple[str, ...] = METHOD_ORDER,
) -> list[Path]:
    """Write one horizontal all-method comparison for every camera view."""

    destination_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for view in range(int(view_count)):
        sliced = {method: [images[method][view]] for method in method_order}
        destination = destination_dir / f"view_{view:03d}_六路并排.png"
        _comparison_sheet(
            sliced,
            destination,
            view_count=1,
            method_order=method_order,
        )
        outputs.append(destination)
    return outputs


def _horizontal_comparison_frame(
    images: dict[str, list[Path]],
    *,
    frame_index: int,
    method_order: tuple[str, ...] = METHOD_ORDER,
) -> np.ndarray:
    first = _opaque_rgb(images[method_order[0]][frame_index])
    cell_w, cell_h = first.size
    header = 40
    canvas = Image.new(
        "RGB", (cell_w * len(method_order), cell_h + header), (28, 28, 28)
    )
    draw = ImageDraw.Draw(canvas)
    font = _font(16)
    for column, method in enumerate(method_order):
        frame = _opaque_rgb(images[method][frame_index])
        if frame.size != (cell_w, cell_h):
            frame = frame.resize((cell_w, cell_h), Image.Resampling.LANCZOS)
        left = column * cell_w
        canvas.paste(frame, (left, header))
        draw.text(
            (left + 8, 10),
            f"{METHOD_LABELS[method]} | frame {frame_index:03d}",
            fill=(242, 242, 242),
            font=font,
        )
    return np.asarray(canvas)


def _write_turntable_video(
    images: dict[str, list[Path]],
    destination: Path,
    *,
    fps: int,
    method_order: tuple[str, ...] = METHOD_ORDER,
) -> None:
    counts = {method: len(images[method]) for method in method_order}
    if len(set(counts.values())) != 1 or not counts or next(iter(counts.values())) <= 1:
        raise RuntimeError(f"turntable frame counts differ or are too small: {counts}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.stem}.tmp-{os.getpid()}{destination.suffix}"
    )
    writer = None
    try:
        writer = imageio.get_writer(
            temporary,
            fps=int(fps),
            codec="libx264",
            quality=8,
            macro_block_size=None,
        )
        for frame_index in range(next(iter(counts.values()))):
            writer.append_data(
                _horizontal_comparison_frame(
                    images,
                    frame_index=frame_index,
                    method_order=method_order,
                )
            )
        writer.close()
        writer = None
        if not temporary.is_file() or temporary.stat().st_size <= 0:
            raise RuntimeError(f"turntable encoder produced no video: {temporary}")
        os.replace(temporary, destination)
    finally:
        if writer is not None:
            writer.close()
        if temporary.exists():
            temporary.unlink()


def _projection_comparison_sheet(
    images: dict[str, list[Path]],
    per_view_iou: dict[str, list[float]],
    destination: Path,
    *,
    view_indices: list[int] | None = None,
) -> None:
    first = _opaque_rgb(next(iter(images.values()))[0])
    source_w, source_h = first.size
    scale = min(1.0, 512.0 / max(source_w, 1))
    cell_w = max(1, int(round(source_w * scale)))
    cell_h = max(1, int(round(source_h * scale)))
    view_count = len(next(iter(images.values())))
    header = 42
    canvas = Image.new(
        "RGB",
        (cell_w * len(METHOD_ORDER), (cell_h + header) * view_count),
        (28, 28, 28),
    )
    draw = ImageDraw.Draw(canvas)
    font = _font(16)
    resolved_view_indices = (
        list(range(view_count)) if view_indices is None else list(view_indices)
    )
    if len(resolved_view_indices) != view_count:
        raise ValueError("projection view_indices differ from image view count")
    for column, method in enumerate(METHOD_ORDER):
        rows = images[method]
        ious = per_view_iou[method]
        if len(rows) != view_count or len(ious) != view_count:
            raise RuntimeError(f"projection view count differs for {method}")
        for view, (path, iou) in enumerate(zip(rows, ious)):
            top = view * (cell_h + header)
            image = _opaque_rgb(path)
            if image.size != (cell_w, cell_h):
                image = image.resize((cell_w, cell_h), Image.Resampling.LANCZOS)
            canvas.paste(image, (column * cell_w, top + header))
            draw.text(
                (column * cell_w + 8, top + 9),
                f"{METHOD_LABELS[method]} | view {resolved_view_indices[view]:02d} | IoU={iou:.3f}",
                fill=(242, 242, 242),
                font=font,
            )
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination)


def _per_view_projection_sheets(
    images: dict[str, list[Path]],
    per_view_iou: dict[str, list[float]],
    destination_dir: Path,
) -> list[Path]:
    """Write one horizontal all-method world-projection sheet per input view."""

    view_count = len(next(iter(images.values())))
    destination_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for view in range(view_count):
        sliced_images = {method: [images[method][view]] for method in METHOD_ORDER}
        sliced_ious = {method: [per_view_iou[method][view]] for method in METHOD_ORDER}
        destination = destination_dir / f"view_{view:03d}_六路并排.png"
        _projection_comparison_sheet(
            sliced_images,
            sliced_ious,
            destination,
            view_indices=[view],
        )
        outputs.append(destination)
    return outputs


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


def _load_projection_contract(label: dict[str, Any]) -> dict[str, Any]:
    runtime_report_path = Path(label["runtime_input_report"]).resolve()
    runtime_report = load_json(runtime_report_path)
    runtime_cache_path = Path(runtime_report["cache_npz"]).resolve()
    raw_cache_path = Path(runtime_report["source_raw_cache"]).resolve()
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
    object_key = str(label["object_key"])
    raw_rows = [
        row
        for row in raw_report["objects"]
        if f'{row["category"]}:{row["object_id"]}' == object_key
    ]
    if len(raw_rows) != 1:
        raise RuntimeError(f"raw-cache report has no unique object row: {object_key}")
    raw_row = raw_rows[0]
    image_dir = Path(raw_row["images_dir"]).resolve()
    mask_dir = Path(raw_row["masks_dir"]).resolve()

    def frame_asset(directory: Path, frame_name: str) -> Path:
        exact = directory / frame_name
        if exact.is_file():
            return exact.resolve()
        matches = sorted(path for path in directory.glob(f"{Path(frame_name).stem}.*") if path.is_file())
        if len(matches) != 1:
            raise FileNotFoundError(
                f"no unique raw asset for frame={frame_name} directory={directory}"
            )
        return matches[0].resolve()

    images = [frame_asset(image_dir, name) for name in frame_names]
    masks = [frame_asset(mask_dir, name) for name in frame_names]
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

    prepared_images = [
        Path(value).resolve() for value in runtime_report["prepared_rgb_paths"]
    ]
    prepared_masks = [
        Path(value).resolve() for value in runtime_report["prepared_mask_paths"]
    ]
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
    if not all(
        path.is_file()
        for path in [*images, *masks, *prepared_images, *prepared_masks]
    ):
        raise FileNotFoundError("a raw or prepared RGB/mask file is missing")
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


def _world_mesh_buffers(
    mesh_path: Path, T_O2W: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    transform = validate_proper_similarity(T_O2W, name="T_O2W")
    vertices_all: list[np.ndarray] = []
    faces_all: list[np.ndarray] = []
    offset = 0
    for mesh in load_meshes(str(mesh_path)):
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


def _image_pixels_to_nvdiffrast_ndc(u, v, width: int, height: int):
    # nvdiffrast's returned row 0 maps to clip-space y=-1.
    return (
        2.0 * u / max(int(width) - 1, 1) - 1.0,
        2.0 * v / max(int(height) - 1, 1) - 1.0,
    )


def _rasterize_world_silhouette(
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
    face_front = positive[faces_t].all(dim=1)
    visible_faces = faces_t[face_front]
    if len(visible_faces) == 0:
        raise RuntimeError("world Mesh has no front-camera triangles")
    z = z_raw.clamp(min=1.0e-5)
    x = cam[:, 0] / z
    y = cam[:, 1] / z
    x, y = _apply_camera_distortion(x, y, camera)
    u = K[0, 0] * x + K[0, 1] * y + K[0, 2]
    v = K[1, 1] * y + K[1, 2]
    x_ndc, y_ndc = _image_pixels_to_nvdiffrast_ndc(u, v, width, height)
    positive_depth = z_raw[positive]
    near = torch.clamp(positive_depth.min() * 0.5, min=1.0e-5)
    far = torch.maximum(positive_depth.max() * 1.5, near + 1.0e-3)
    z_ndc = 2.0 * (z - near) / (far - near) - 1.0
    pos_clip = torch.stack([x_ndc * z, y_ndc * z, z_ndc * z, z], dim=1)
    rast, _ = dr.rasterize(
        rastctx, pos_clip[None], visible_faces, resolution=(height, width)
    )
    return (rast[0, ..., 3].detach().cpu().numpy() > 0).astype(np.uint8)


def _dilate(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    value = np.asarray(mask, dtype=bool)
    for _ in range(int(iterations)):
        padded = np.pad(value, 1, mode="constant")
        value = np.logical_or.reduce(
            [
                padded[dy : dy + value.shape[0], dx : dx + value.shape[1]]
                for dy in range(3)
                for dx in range(3)
            ]
        )
    return value


def _boundary(mask: np.ndarray, width: int = 2) -> np.ndarray:
    value = np.asarray(mask, dtype=bool)
    return _dilate(value, width) ^ value


def _silhouette_iou(first: np.ndarray, second: np.ndarray) -> float:
    left, right = np.asarray(first, dtype=bool), np.asarray(second, dtype=bool)
    union = np.logical_or(left, right).sum()
    return float(np.logical_and(left, right).sum() / union) if union else 1.0


def _overlay_projection(
    image_path: Path,
    input_mask_path: Path,
    projected_mask: np.ndarray,
    color: np.ndarray,
    destination: Path,
) -> float:
    image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.float32)
    input_mask = np.asarray(Image.open(input_mask_path).convert("L")) > 127
    projected = np.asarray(projected_mask, dtype=bool)
    if image.shape[:2] != input_mask.shape or input_mask.shape != projected.shape:
        raise RuntimeError("RGB, input mask and projected mask shapes differ")
    output = image.copy()
    output[projected] = (
        (1.0 - PROJECTED_FILL_ALPHA) * output[projected]
        + PROJECTED_FILL_ALPHA * color[None]
    )
    output[_boundary(input_mask)] = INPUT_MASK_COLOR
    output[_boundary(projected)] = color
    destination.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(output, 0, 255).astype(np.uint8), mode="RGB").save(
        destination
    )
    return _silhouette_iou(projected, input_mask)


def _render_world_projection_overlays(
    *,
    method: str,
    mesh: Path,
    contract: dict[str, Any],
    output_dir: Path,
    rastctx,
    resume: bool,
) -> tuple[list[Path], list[float]]:
    images: list[Path] = contract["images"]
    masks: list[Path] = contract["masks"]
    expected = [output_dir / f"view_{index:02d}_projection.png" for index in range(len(images))]
    stats_path = output_dir / "projection_stats.json"
    if all(path.is_file() for path in expected) and stats_path.is_file():
        stats = load_json(stats_path)
        return expected, [float(value) for value in stats["per_view_silhouette_iou"]]
    if output_dir.exists() and not resume:
        raise RuntimeError(f"partial world projection exists; pass --resume: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    vertices_w, faces = _world_mesh_buffers(mesh, contract["T_O2W"])
    ious: list[float] = []
    foreground_pixels: list[int] = []
    for index, (image_path, mask_path, T_W2C, K, camera) in enumerate(
        zip(
            images,
            masks,
            contract["T_W2C"],
            contract["intrinsics"],
            contract["cameras"],
        )
    ):
        with Image.open(image_path) as image:
            width, height = image.size
        projected = _rasterize_world_silhouette(
            vertices_w, faces, T_W2C, K, camera, (height, width), rastctx
        )
        foreground_pixels.append(int(projected.sum()))
        ious.append(
            _overlay_projection(
                image_path,
                mask_path,
                projected,
                PROJECTION_COLORS[method],
                expected[index],
            )
        )
    stats = {
        "method": method,
        "mesh_o": str(mesh),
        "projection_formula": (
            "Mesh_O -> T_O2W -> Mesh_W -> T_W2C -> K_raw + raw camera distortion"
        ),
        "used_physical_T_O2W": True,
        "used_raw_T_W2C": True,
        "used_T_O2C_lifting": False,
        "input_image_semantics": "original registered 1920x1080 RGB frame",
        "camera_models": [camera["model"] for camera in contract["cameras"]],
        "camera_distortion": [
            camera["distortion"] for camera in contract["cameras"]
        ],
        "input_mask_outline": "yellow",
        "projected_mesh_fill_and_outline_rgb": PROJECTION_COLORS[method].astype(int).tolist(),
        "per_view_silhouette_iou": ious,
        "mean_silhouette_iou": float(np.mean(ious)),
        "projected_foreground_pixels": foreground_pixels,
        "passed": all(value > 0 for value in foreground_pixels),
    }
    atomic_json(stats_path, stats)
    if not stats["passed"]:
        raise RuntimeError(f"an empty Mesh projection was produced for {method}")
    return expected, ious


def _rank_percentiles(
    rows: list[dict[str, Any]], field: str, *, lower_is_better: bool
) -> dict[str, float]:
    ordered = sorted(
        rows,
        key=lambda row: (
            float(row[field]) if lower_is_better else -float(row[field]),
            str(row["object_key"]),
        ),
    )
    count = len(ordered)
    if count == 1:
        return {str(ordered[0]["object_key"]): 1.0}
    output: dict[str, float] = {}
    position = 0
    while position < count:
        value = float(ordered[position][field])
        end = position + 1
        while end < count and float(ordered[end][field]) == value:
            end += 1
        average_rank = 0.5 * (position + end - 1)
        percentile = 1.0 - average_rank / (count - 1)
        for index in range(position, end):
            output[str(ordered[index]["object_key"])] = float(percentile)
        position = end
    return output


def score_final_records(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    if not rows:
        raise RuntimeError("final method contains no per-object metric records")
    component_scores = {
        field: _rank_percentiles(
            rows, field, lower_is_better=field in LOWER_IS_BETTER
        )
        for field in RANK_WEIGHTS
    }
    output = {}
    by_key = {str(row["object_key"]): row for row in rows}
    for key, row in by_key.items():
        components = {field: component_scores[field][key] for field in RANK_WEIGHTS}
        output[key] = {
            "aggregate_score": float(
                sum(RANK_WEIGHTS[field] * components[field] for field in RANK_WEIGHTS)
            ),
            "rank_percentiles": components,
            "metrics": {field: float(row[field]) for field in RANK_WEIGHTS},
            "alignment_quality_tier": str(
                row.get("alignment_quality_tier", "reliable")
            ),
        }
    return output


def select_cases(
    report: dict[str, Any],
    *,
    top_count: int,
    random_count: int,
    random_seed: int,
    excluded_object_keys: set[str] | None = None,
) -> list[dict[str, Any]]:
    records = list(report.get("records", []))
    final_rows = [row for row in records if row.get("method") == "final_native_no_vggt"]
    scores = score_final_records(final_rows)
    excluded = set(excluded_object_keys or ())
    unknown_exclusions = sorted(excluded - set(scores))
    if unknown_exclusions:
        raise RuntimeError(
            f"excluded objects are absent from the benchmark: {unknown_exclusions}"
        )
    reliable = [
        key for key, value in scores.items()
        if value["alignment_quality_tier"] == "reliable" and key not in excluded
    ]
    best = sorted(reliable, key=lambda key: (-scores[key]["aggregate_score"], key))[
        :top_count
    ]
    if len(best) != top_count:
        raise RuntimeError("not enough reliable Holdout objects for top selection")
    remaining = sorted(set(scores) - set(best) - excluded)
    if len(remaining) < random_count:
        raise RuntimeError("not enough remaining Holdout objects for random selection")
    random_keys = random.Random(int(random_seed)).sample(remaining, random_count)
    return [
        {
            "object_key": key,
            "selection": "posthoc_best" if key in best else "fixed_random",
            "aggregate": scores[key],
        }
        for key in [*best, *random_keys]
    ]


def load_inference_records(
    report: dict[str, Any], report_path: Path
) -> dict[tuple[str, str], dict[str, Any]]:
    del report_path  # Retained in the signature for future provenance diagnostics.
    if report.get("records"):
        return {
            (str(row["object_key"]), str(row["method"])): row
            for row in report["records"]
        }
    bindings = report.get("method_manifests")
    if not isinstance(bindings, dict):
        raise RuntimeError("report has neither metric records nor method manifests")
    output: dict[tuple[str, str], dict[str, Any]] = {}
    for method in METHOD_ORDER[1:]:
        binding = bindings.get(method)
        if not isinstance(binding, dict):
            raise RuntimeError(f"missing method manifest binding: {method}")
        manifest_path = Path(binding["path"]).expanduser().resolve()
        if sha256_file(manifest_path) != str(binding["sha256"]):
            raise RuntimeError(f"method manifest binding changed: {method}")
        manifest = load_json(manifest_path)
        if manifest.get("passed") is not True:
            raise RuntimeError(f"method inference manifest did not pass: {method}")
        rows = list(manifest.get("objects", []))
        if len(rows) != int(report["object_count"]):
            raise RuntimeError(f"method object count differs: {method}")
        for row in rows:
            key = (str(row["object_key"]), method)
            if key in output:
                raise RuntimeError(f"duplicate method/object inference row: {key}")
            output[key] = row
    return output


def validate_current_gaussian_appearance_report(
    gaussian_report: dict[str, Any],
    *,
    benchmark_report: dict[str, Any],
    selected: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Bind a GS appearance run to the exact current-method inference and cases."""

    if (
        gaussian_report.get("format") != GAUSSIAN_APPEARANCE_REPORT_FORMAT
        or gaussian_report.get("passed") is not True
        or gaussian_report.get("formal") is not False
    ):
        raise RuntimeError("current-model Gaussian appearance report did not pass")
    binding = benchmark_report.get("method_manifests", {}).get(CURRENT_METHOD)
    if not isinstance(binding, dict):
        raise RuntimeError("benchmark report does not bind the current inference manifest")
    run_config = gaussian_report.get("run_config", {})
    if run_config.get("export_glb") is not True:
        raise RuntimeError(
            "current-model Gaussian appearance did not export GS-baked textured GLBs"
        )
    source_path = Path(str(run_config.get("source_inference_manifest", ""))).resolve()
    bound_path = Path(str(binding.get("path", ""))).resolve()
    if source_path != bound_path:
        raise RuntimeError("Gaussian appearance used a different current inference manifest")
    if (
        str(run_config.get("source_inference_manifest_sha256", ""))
        != str(binding.get("sha256", ""))
        or not bound_path.is_file()
        or sha256_file(bound_path) != str(binding.get("sha256", ""))
    ):
        raise RuntimeError("current inference manifest binding changed")
    expected_keys = [str(row["object_key"]) for row in selected]
    objects = list(gaussian_report.get("objects", []))
    actual_keys = [str(row.get("object_key")) for row in objects]
    if actual_keys != expected_keys:
        raise RuntimeError(
            "Gaussian appearance object order differs from qualitative selection: "
            f"expected={expected_keys} actual={actual_keys}"
        )
    output: dict[str, dict[str, Any]] = {}
    for row in objects:
        if row.get("format") != GAUSSIAN_APPEARANCE_OBJECT_FORMAT or row.get("passed") is not True:
            raise RuntimeError("a current-model Gaussian appearance object did not pass")
        for branch in ("stock", "full"):
            artifact = row.get("artifacts", {}).get(branch, {})
            ply = Path(str(artifact.get("gaussian_ply", ""))).resolve()
            if not ply.is_file() or sha256_file(ply) != str(
                artifact.get("gaussian_ply_sha256", "")
            ):
                raise RuntimeError(f"Gaussian PLY binding changed: {branch} {ply}")
            textured_glb = Path(str(artifact.get("textured_glb", ""))).resolve()
            if not textured_glb.is_file() or sha256_file(textured_glb) != str(
                artifact.get("textured_glb_sha256", "")
            ):
                raise RuntimeError(
                    f"GS-baked textured GLB binding changed: {branch} {textured_glb}"
                )
            records = row.get("branches", {}).get(branch, [])
            if not records or not all(
                Path(str(record.get("rgb", ""))).is_file()
                and Path(str(record.get("alpha", ""))).is_file()
                for record in records
            ):
                raise RuntimeError(f"Gaussian registered renders are incomplete: {branch}")
        contact = Path(str(row.get("contact_sheet", ""))).resolve()
        if not contact.is_file():
            raise RuntimeError(f"Gaussian contact sheet is missing: {contact}")
        output[str(row["object_key"])] = row
    return output


def _ensure_current_gaussian_appearance(
    *,
    benchmark_report: dict[str, Any],
    selected: list[dict[str, Any]],
    selection_path: Path,
    output_dir: Path,
    gaussian_cache_dir: Path | None = None,
) -> tuple[Path, dict[str, dict[str, Any]]]:
    """Replay the selected current outputs and render frozen-GS appearance."""

    binding = benchmark_report.get("method_manifests", {}).get(CURRENT_METHOD)
    if not isinstance(binding, dict):
        raise RuntimeError("current inference manifest binding is unavailable")
    gaussian_dir = (
        gaussian_cache_dir.resolve()
        if gaussian_cache_dir is not None
        else output_dir / "当前NoVGGT_Gaussian注册视图"
    )
    gaussian_report_path = gaussian_dir / "report.json"
    if not gaussian_report_path.is_file():
        command = [
            sys.executable,
            "-u",
            "-m",
            "pose_point_depth_mv.evaluate_native_no_vggt_gaussian_appearance",
            "--inference_manifest",
            str(Path(binding["path"]).resolve()),
            "--output_dir",
            str(gaussian_dir),
            "--selection_manifest",
            str(selection_path),
            "--views",
            "0,1,2,3,4,5,6,7",
            "--resolution",
            "518",
            "--amp_dtype",
            "bf16",
            "--export_glb",
            "--texture_size",
            "1024",
        ]
        if gaussian_dir.exists():
            command.append("--resume")
        # Keep the expensive replay's complete output outside ``gaussian_dir``:
        # the evaluator must create that immutable directory itself on a fresh
        # run.  Stream every line to the terminal as before while also keeping
        # a durable sibling log for failures that happen before report.json is
        # committed.
        log_path = gaussian_dir.with_name(f"{gaussian_dir.name}.log")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as log:
            log.write("\n=== Gaussian appearance replay ===\n")
            log.write(json.dumps(command, ensure_ascii=False) + "\n")
            log.flush()
            process = subprocess.Popen(
                command,
                cwd=str(TRACKER_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                log.write(line)
                log.flush()
            return_code = process.wait()
        if return_code != 0:
            lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
            tail = "\n".join(lines[-60:])
            raise RuntimeError(
                f"Gaussian appearance replay failed with exit status {return_code}. "
                f"Full log: {log_path}\nLast log lines:\n{tail}"
            )
    gaussian_report = load_json(gaussian_report_path)
    by_key = validate_current_gaussian_appearance_report(
        gaussian_report,
        benchmark_report=benchmark_report,
        selected=selected,
    )
    return gaussian_report_path, by_key


def _copy_current_gaussian_assets(
    row: dict[str, Any], *, case_dir: Path, current_mesh_dir: Path
) -> dict[str, Any]:
    destination = case_dir / "当前NoVGGT_GS外观"
    branch_bindings: dict[str, Any] = {}
    for branch in ("stock", "full"):
        branch_dir = destination / branch
        artifact = row["artifacts"][branch]
        ply = _copy_bound(
            Path(artifact["gaussian_ply"]),
            current_mesh_dir / "Gaussian外观" / f"{branch}.ply",
        )
        textured_glb = _copy_bound(
            Path(artifact["textured_glb"]),
            current_mesh_dir / "Gaussian外观" / f"{branch}_GS烘焙纹理.glb",
        )
        renders = []
        for record in row["branches"][branch]:
            view = int(record["view_index"])
            rgb = _copy_bound(
                Path(record["rgb"]), branch_dir / f"view_{view:02d}_rgb.png"
            )
            alpha = _copy_bound(
                Path(record["alpha"]), branch_dir / f"view_{view:02d}_alpha.png"
            )
            renders.append({"view_index": view, "rgb": rgb, "alpha": alpha})
        branch_bindings[branch] = {
            "gaussian_ply": ply,
            "gs_baked_textured_glb": textured_glb,
            "registered_views": renders,
        }
    sheet = _copy_bound(
        Path(row["contact_sheet"]),
        case_dir / "当前NoVGGT_输入_StockGS_FullGS_注册8视图对比.png",
    )
    return {
        "representation": "frozen_trellis_gaussian_decoder",
        "renderer": "registered_center_splat_v1",
        "input_role": "reference_only_not_reprojected_as_texture",
        "branches": branch_bindings,
        "sheet": sheet,
    }


def _reference_current_gaussian_assets(row: dict[str, Any]) -> dict[str, Any]:
    """Bind frozen GS artifacts in place without copying them into the review."""

    branches: dict[str, Any] = {}
    for branch in ("stock", "full"):
        artifact = row["artifacts"][branch]
        ply = Path(artifact["gaussian_ply"]).resolve()
        textured_glb = Path(artifact["textured_glb"]).resolve()
        branches[branch] = {
            "gaussian_ply": {"path": str(ply), "sha256": sha256_file(ply)},
            "gs_baked_textured_glb": {
                "path": str(textured_glb),
                "sha256": sha256_file(textured_glb),
            },
            "registered_views": [
                {
                    "view_index": int(record["view_index"]),
                    "rgb": {
                        "path": str(Path(record["rgb"]).resolve()),
                        "sha256": sha256_file(Path(record["rgb"])),
                    },
                    "alpha": {
                        "path": str(Path(record["alpha"]).resolve()),
                        "sha256": sha256_file(Path(record["alpha"])),
                    },
                }
                for record in row["branches"][branch]
            ],
        }
    sheet = Path(row["contact_sheet"]).resolve()
    return {
        "representation": "frozen_trellis_gaussian_decoder",
        "renderer": "registered_center_splat_v1",
        "input_role": "reference_only_not_reprojected_as_texture",
        "asset_policy": "source bindings only; nothing copied into review bundle",
        "branches": branches,
        "sheet": {"path": str(sheet), "sha256": sha256_file(sheet)},
    }


def _mesh_bounds(path: Path) -> tuple[list[float], float]:
    scene = _mesh_scene(path)
    vertices = np.concatenate(
        [np.asarray(mesh.vertices, dtype=np.float64) for mesh in scene.geometry.values()],
        axis=0,
    )
    minimum, maximum = vertices.min(axis=0), vertices.max(axis=0)
    center = 0.5 * (minimum + maximum)
    scale = float(np.max(maximum - minimum))
    if not math.isfinite(scale) or scale <= 1.0e-8:
        raise RuntimeError(f"invalid GT display bounds: {path}")
    return center.tolist(), scale


def _intrinsic(image_size: int, fov_degrees: float) -> list[list[float]]:
    focal = 0.5 * image_size / math.tan(0.5 * math.radians(fov_degrees))
    center = 0.5 * (image_size - 1)
    return [[focal, 0.0, center], [0.0, focal, center], [0.0, 0.0, 1.0]]


def _render_blender(
    *,
    blender: Path,
    renderer: Path,
    mesh: Path,
    output_dir: Path,
    center: list[float],
    scale: float,
    cameras: np.ndarray,
    image_size: int,
    material_mode: str,
    resume: bool,
) -> list[Path]:
    expected = [output_dir / f"view_{index:03d}.png" for index in range(len(cameras))]
    if all(path.is_file() for path in expected):
        return expected
    if output_dir.exists() and not resume:
        raise RuntimeError(f"partial render exists; pass --resume: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    request = {
        "glb_path": str(mesh),
        "output_dir": str(output_dir),
        "center": center,
        "scale": scale,
        "margin": 0.78,
        "image_size": image_size,
        "samples": 32,
        "world_strength": 0.65,
        "light_energy": 650.0,
        "cycles_device": "CPU",
        "engine": "BLENDER_EEVEE",
        "material_mode": material_mode,
        "intrinsic": _intrinsic(image_size, 42.0),
        "c2w": cameras.tolist(),
    }
    request_path = output_dir / "render_request.json"
    atomic_json(request_path, request)
    log_path = output_dir / "blender.log"
    with log_path.open("w", encoding="utf-8") as log:
        subprocess.run(
            [
                str(blender), "-b", "--python", str(renderer), "--", str(request_path)
            ],
            cwd=str(renderer.parents[2]),
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    if not all(path.is_file() for path in expected):
        raise RuntimeError(f"Blender did not create every requested view: {output_dir}")
    return expected


def _render_nvdiffrast(
    *,
    mesh: Path,
    output_dir: Path,
    center: list[float],
    scale: float,
    cameras: np.ndarray,
    image_size: int,
    material_mode: str,
    resume: bool,
    rastctx,
) -> list[Path]:
    expected = [output_dir / f"view_{index:03d}.png" for index in range(len(cameras))]
    if all(path.is_file() for path in expected):
        return expected
    if output_dir.exists() and not resume:
        raise RuntimeError(f"partial render exists; pass --resume: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    meshes = load_meshes(str(mesh))
    vertices, faces, colors, normals = build_render_buffers(
        meshes,
        np.asarray(center, dtype=np.float32),
        float(scale),
        0.78,
    )
    if material_mode == "clay":
        colors = np.repeat(
            np.asarray([[158, 171, 189]], dtype=np.uint8), len(vertices), axis=0
        )
    elif material_mode != "source":
        raise ValueError(f"unsupported material_mode={material_mode!r}")
    render_args = SimpleNamespace(
        renderer="nvdiffrast",
        shading_mode="normal",
        shading_ambient=0.46,
        shading_diffuse=0.68,
    )
    intrinsic = np.asarray(_intrinsic(image_size, 42.0), dtype=np.float32)
    for index, c2w in enumerate(cameras):
        image, alpha = render_mesh_rgb(
            vertices,
            faces,
            colors,
            normals,
            np.asarray(c2w, dtype=np.float32),
            intrinsic,
            int(image_size),
            rastctx,
            render_args,
        )
        rgba = np.concatenate([image, alpha[..., None]], axis=-1)
        Image.fromarray(rgba, mode="RGBA").save(expected[index])
    return expected


def _make_headless_raster_context():
    """Make nvdiffrast work when Python is invoked by absolute Conda path."""

    environment_bin = str(Path(sys.executable).resolve().parent)
    path_entries = os.environ.get("PATH", "").split(os.pathsep)
    if environment_bin not in path_entries:
        os.environ["PATH"] = environment_bin + os.pathsep + os.environ.get("PATH", "")
    os.environ.setdefault("TORCH_EXTENSIONS_DIR", "/tmp/torch_extensions")
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.6")
    return make_nvdiffrast_context()


def _render_views(*, backend: str, rastctx=None, **kwargs) -> list[Path]:
    if backend == "nvdiffrast":
        blender_only = {"blender", "renderer"}
        return _render_nvdiffrast(
            rastctx=rastctx,
            **{key: value for key, value in kwargs.items() if key not in blender_only},
        )
    if backend == "blender":
        return _render_blender(**kwargs)
    raise ValueError(f"unsupported renderer backend={backend!r}")


def _copy_inputs(label: dict[str, Any], destination: Path) -> list[dict[str, Any]]:
    report_path = Path(str(label.get("runtime_input_report", "")))
    if not report_path.is_file():
        return []
    report = load_json(report_path)
    bindings = []
    for index, value in enumerate(report.get("prepared_rgb_paths", [])):
        source = Path(value)
        bindings.append(_copy_bound(source, destination / f"view_{index:02d}{source.suffix}"))
    return bindings


def _prepare_gt(
    label: dict[str, Any], mesh_dir: Path, *, copy_assets: bool = True
) -> dict[str, Any]:
    metric = Path(label["mesh_o"]).resolve()
    metric_binding = (
        _copy_bound(metric, mesh_dir / "metric_mesh_o.obj")
        if copy_assets
        else {"path": str(metric), "sha256": sha256_file(metric)}
    )
    source_scan = Path(label["scan_obj"]).resolve()
    with np.load(label["label_cache"], allow_pickle=False) as payload:
        transform = np.asarray(payload["T_Scan2O"], dtype=np.float64)
    textured_dir = mesh_dir / "带纹理GT"
    textured_obj = textured_dir / "Scan_in_runtime_O_textured.obj"
    if not textured_obj.is_file():
        transform_obj_geometry(
            source_scan, textured_obj, transform, strip_materials=False
        )
    dependencies = _copy_obj_dependencies(source_scan, textured_dir)
    return {
        "source_mesh": str(source_scan),
        "source_mesh_sha256": sha256_file(source_scan),
        "metric_mesh": metric_binding,
        **(
            {
                "copied_mesh": {
                    "path": str(textured_obj),
                    "sha256": sha256_file(textured_obj),
                },
                "dependencies": dependencies,
            }
            if copy_assets
            else {"asset_policy": "source bindings only; temporary textured GT removed"}
        ),
        "display_mesh": str(textured_obj if copy_assets else metric),
        "_appearance_mesh": str(textured_obj),
        "appearance_kind": _appearance_kind(textured_obj),
        "transform_changed_for_display": False,
        "display_export": "same frozen T_Scan2O as metric GT; materials retained",
    }


def _prepare_prediction(
    record: dict[str, Any], mesh_dir: Path, *, copy_assets: bool = True
) -> dict[str, Any]:
    source = Path(record["mesh"]).resolve()
    actual_sha256 = sha256_file(source)
    if actual_sha256 != str(record["mesh_sha256"]):
        raise RuntimeError(f"prediction Mesh binding changed: {source}")
    if not copy_assets:
        return {
            "source_mesh": str(source),
            "source_mesh_sha256": actual_sha256,
            "display_mesh": str(source),
            "appearance_kind": _appearance_kind(source),
            "asset_policy": "source binding only; Mesh not copied into review bundle",
            "transform_changed_for_display": False,
            "display_export": "source Mesh used in place; no conversion or copy",
        }
    copied = _copy_bound(source, mesh_dir / f"原始输出{source.suffix.lower()}")
    dependencies = _copy_obj_dependencies(source, mesh_dir) if source.suffix.lower() == ".obj" else []
    appearance = _appearance_kind(source)
    if source.suffix.lower() in {".glb", ".gltf"}:
        display = Path(copied["path"])
        display_binding = copied
    else:
        display = mesh_dir / "display_vertex_color.glb"
        display_binding = _export_display_glb(source, display)
    return {
        "source_mesh": str(source),
        "source_mesh_sha256": str(record["mesh_sha256"]),
        "copied_mesh": copied,
        "dependencies": dependencies,
        "display_mesh": str(display),
        "display_mesh_binding": display_binding,
        "appearance_kind": appearance,
        "transform_changed_for_display": False,
        "display_export": "format conversion only; no centering, scaling, ICP, or pose fit",
    }


def _html_index(
    output_dir: Path, cases: list[dict[str, Any]], *, source_scope: str
) -> None:
    sections = []
    for case in cases:
        relative = os.path.relpath(case["directory"], output_dir).replace(os.sep, "/")
        grouped_root = Path(case["directory"]) / "渲染" / "六路并排对比"
        grouped_sections = []
        for label, directory in (
            ("Native appearance — same view, six methods", "自身纹理或顶点颜色"),
            ("Shared clay — same view, six methods", "统一材质几何"),
            ("World projection — same input view, six methods", "物理To2w世界投影"),
        ):
            paths = sorted((grouped_root / directory).glob("view_*_六路并排.png"))
            if paths:
                images = "".join(
                    f'<img src="{html.escape(os.path.relpath(path, output_dir).replace(os.sep, "/"))}">'
                    for path in paths
                )
                grouped_sections.append(f"<h3>{html.escape(label)}</h3>{images}")
        appearance_video = (
            Path(case["directory"])
            / "渲染"
            / "六路同步环绕_自身纹理或顶点颜色.mp4"
        )
        clay_video = (
            Path(case["directory"])
            / "渲染"
            / "六路同步环绕_统一材质几何.mp4"
        )
        if appearance_video.is_file() and clay_video.is_file():
            render_review = (
                "<h3>Six-method synchronized native-appearance turntable</h3>"
                f'<video controls loop muted playsinline src="{relative}/渲染/'
                '六路同步环绕_自身纹理或顶点颜色.mp4"></video>'
                "<h3>Six-method synchronized shared-clay turntable</h3>"
                f'<video controls loop muted playsinline src="{relative}/渲染/'
                '六路同步环绕_统一材质几何.mp4"></video>'
            )
        else:
            render_review = (
                "<h3>Six-method native appearance</h3>"
                "<p>The Current No-VGGT column is the Full SLat decoded by the "
                "frozen Gaussian decoder and baked onto the matching frozen Mesh "
                "decoder output. The legacy OBJ vertex colors are not used.</p>"
                f'<img src="{relative}/自身纹理或顶点颜色_6视图对比.png">'
                "<h3>Supplementary registered GS audit: input vs Stock vs Full</h3>"
                f'<img src="{relative}/当前NoVGGT_输入_StockGS_FullGS_注册8视图对比.png">'
                f'<h3>Shared clay geometry</h3><img src="{relative}/统一材质几何_6视图对比.png">'
            )
        sections.append(
            f"<section><h2>{html.escape(case['object_key'])} — "
            f"{html.escape(case['selection'])}</h2>"
            f'<p><a href="{relative}/对象报告.json">object report</a></p>'
            f"{render_review}"
            f'<h3>Physical T_O2W + T_W2C projection on original RGB</h3>'
            f'<p>Yellow: original-image mask. Method color: projected Mesh silhouette.</p>'
            f'<img src="{relative}/物理To2w_Tw2c投影到原始8视图对比.png">'
            f"{''.join(grouped_sections)}"
            "</section>"
        )
    page = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Omni Holdout64 Mesh review</title><style>
body{{background:#151515;color:#eee;font-family:sans-serif;margin:24px}}
a{{color:#7cc7ff}}img,video{{max-width:100%;border:1px solid #555}}
section{{background:#222;margin:24px 0;padding:16px}}</style></head><body>
<h1>Omni real Mesh qualitative review — {html.escape(source_scope)}</h1>
<p>Post-hoc qualitative review only. All methods share the GT-derived display
center, scale and camera ring for geometry. In the main appearance review, the
current No-VGGT column is the Full SLat decoded by the frozen Gaussian decoder
and GS-baked onto the matching frozen Mesh decoder output; its legacy OBJ vertex
colors are not used. Registered Input/Stock/Full GS artifacts are supplementary.
Other methods retain their native appearance. Appearance and neutral-clay renders
are separate.
The projection sheet uses the physical chain Mesh_O → T_O2W → Mesh_W → T_W2C
→ raw K plus the frozen COLMAP camera distortion, never T_O2C_lifting.</p>
{''.join(sections)}</body></html>"""
    (output_dir / "index.html").write_text(page, encoding="utf-8")


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final_report", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--blender", default="/tmp/blender-3.0.1-linux-x64/blender")
    parser.add_argument(
        "--blender_renderer",
        default="pose_point_depth_mv/dataset_tools/blender_pbr_render_multiview.py",
    )
    parser.add_argument("--top_count", type=int, default=2)
    parser.add_argument("--random_count", type=int, default=2)
    parser.add_argument("--random_seed", type=int, default=20260809)
    parser.add_argument("--view_count", type=int, default=6)
    parser.add_argument("--image_size", type=int, default=384)
    parser.add_argument(
        "--renderer_backend",
        choices=("nvdiffrast", "blender"),
        default="nvdiffrast",
        help=(
            "nvdiffrast is the headless default and renders vertex colors plus "
            "UV textures sampled to Mesh vertices; Blender preserves exact PBR "
            "materials but requires a working display/Xvfb on this host."
        ),
    )
    parser.add_argument(
        "--skip_world_projection",
        action="store_true",
        help="Skip the physical T_O2W/T_W2C projection overlay (diagnostic only).",
    )
    parser.add_argument(
        "--development_best_plus_random",
        action="store_true",
        help=(
            "Accept the frozen formal=false development Benchmark32 report and "
            "select metric-ranked best cases plus fixed-random cases from the "
            "remaining complete object population. This is post-hoc development "
            "inspection only."
        ),
    )
    parser.add_argument(
        "--exclude_selection_manifest",
        help=(
            "Exclude every object in a prior selection manifest bound to the same "
            "benchmark report. Intended for a disjoint additional random review."
        ),
    )
    parser.add_argument(
        "--group_renders_by_view",
        action="store_true",
        help=(
            "Keep single-method renders only as resumable intermediates and add "
            "one horizontal six-method comparison image per camera view under "
            "渲染/六路并排对比."
        ),
    )
    parser.add_argument(
        "--turntable_video_only",
        action="store_true",
        help=(
            "For camera-ring rendering, output only two synchronized six-method "
            "MP4 turntables (native appearance and shared clay). Render frames, "
            "input copies and Mesh copies are omitted from the completed bundle; "
            "world-projection overlay PNGs remain unchanged."
        ),
    )
    parser.add_argument("--turntable_fps", type=int, default=18)
    parser.add_argument(
        "--gaussian_cache_dir",
        help=(
            "Optional external cache for frozen Gaussian replay artifacts, keeping "
            "registered GS frames and decoded assets outside the review bundle."
        ),
    )
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> None:
    args = make_parser().parse_args()
    report_path = Path(args.final_report).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    blender = Path(args.blender).expanduser().resolve()
    renderer = Path(args.blender_renderer).expanduser().resolve()
    report = load_json(report_path)
    if report.get("format") != EXPECTED_REPORT_FORMAT or report.get("passed") is not True:
        raise RuntimeError("a passed final benchmark report is required")
    development_best_plus_random = bool(args.development_best_plus_random)
    if development_best_plus_random:
        if (
            report.get("formal") is not False
            or report.get("protocol_scope") != "development_benchmark32"
            or int(report.get("object_count", -1)) != 32
            or not report.get("records")
        ):
            raise RuntimeError(
                "development best+random mode requires a formal=false Benchmark32 "
                "report with per-object metric records"
            )
    elif (
        report.get("formal") is not True
        or int(report.get("object_count", -1)) != 64
        or not report.get("records")
    ):
        raise RuntimeError(
            "a passed formal Holdout64 report with per-object records is required"
        )
    if args.renderer_backend == "blender" and (
        not blender.is_file() or not renderer.is_file()
    ):
        raise FileNotFoundError(f"Blender or renderer is missing: {blender}, {renderer}")
    if (
        int(args.top_count) < 0
        or min(args.random_count, args.view_count, args.image_size) <= 0
    ):
        raise ValueError("selection counts, view_count and image_size are invalid")
    if int(args.turntable_fps) <= 0:
        raise ValueError("turntable_fps must be positive")
    if args.turntable_video_only and args.group_renders_by_view:
        raise ValueError(
            "turntable_video_only and group_renders_by_view are mutually exclusive"
        )

    labels = load_json(report["label_manifest"])
    label_by_key = {str(row["object_key"]): row for row in labels["objects"]}
    if len(label_by_key) != int(report["object_count"]):
        raise RuntimeError("label population differs from benchmark report")
    exclusion_binding = None
    excluded_object_keys: set[str] = set()
    if args.exclude_selection_manifest:
        exclusion_path = Path(args.exclude_selection_manifest).expanduser().resolve()
        exclusion = load_json(exclusion_path)
        if (
            str(exclusion.get("final_report_sha256", "")) != sha256_file(report_path)
            or str(exclusion.get("source_protocol_scope", ""))
            != str(report.get("protocol_scope", ""))
        ):
            raise RuntimeError(
                "excluded selection manifest is not bound to this benchmark report"
            )
        excluded_rows = list(exclusion.get("selected", []))
        excluded_object_keys = {
            str(row["object_key"])
            for row in excluded_rows
            if isinstance(row, dict) and row.get("object_key")
        }
        if len(excluded_object_keys) != len(excluded_rows) or not excluded_object_keys:
            raise RuntimeError("excluded selection manifest has invalid/duplicate objects")
        exclusion_binding = {
            "path": str(exclusion_path),
            "sha256": sha256_file(exclusion_path),
            "object_keys": sorted(excluded_object_keys),
        }
    selected = select_cases(
        report,
        top_count=int(args.top_count),
        random_count=int(args.random_count),
        random_seed=int(args.random_seed),
        excluded_object_keys=excluded_object_keys,
    )
    records = load_inference_records(report, report_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    selection_manifest = {
        "format": FORMAT,
        "purpose": (
            "development Benchmark32 posthoc best-plus-fixed-random qualitative "
            "inspection; forbidden for claims or model selection"
            if development_best_plus_random
            else "posthoc qualitative inspection; forbidden for model selection or formal claims"
        ),
        "source_protocol_scope": str(report.get("protocol_scope")),
        "source_formal": bool(report.get("formal")),
        "final_report": str(report_path),
        "final_report_sha256": sha256_file(report_path),
        "render_layout": (
            "two synchronized six-method turntable MP4 files; no completed render "
            "frames, input copies, or copied Mesh assets"
            if args.turntable_video_only
            else (
                "same-view six-method comparisons; single-method renders retained only "
                "under _单路中间产物 for resume/audit"
                if args.group_renders_by_view
                else "legacy per-method renders plus all-view comparison sheets"
            )
        ),
        "selection_policy": (
            {
                "mode": "development_benchmark32_posthoc_best_plus_fixed_random",
                "top_count": int(args.top_count),
                "top_eligible_population": "all alignment-quality reliable Benchmark32 objects",
                "rank_weights": RANK_WEIGHTS,
                "random_count": int(args.random_count),
                "random_seed": int(args.random_seed),
                "random_population": (
                    "all remaining Benchmark32 objects after top selection and "
                    "the bound prior-selection exclusions"
                ),
                "exclusion_binding": exclusion_binding,
                "comparison_population": "all 32 objects; 31/32 ReconViaGen wins is not a filter",
            }
            if development_best_plus_random
            else {
                "mode": "posthoc_best_plus_fixed_random",
                "top_count": int(args.top_count),
                "top_eligible_population": "alignment-quality reliable objects",
                "rank_weights": RANK_WEIGHTS,
                "random_count": int(args.random_count),
                "random_seed": int(args.random_seed),
                "random_population": "all remaining frozen Holdout64 objects",
            }
        ),
        "selected": selected,
    }
    selection_path = output_dir / "选择清单.json"
    if selection_path.is_file():
        if load_json(selection_path) != selection_manifest:
            raise RuntimeError(f"existing review selection differs: {selection_path}")
    else:
        atomic_json(selection_path, selection_manifest)

    gaussian_report_path, gaussian_by_key = _ensure_current_gaussian_appearance(
        benchmark_report=report,
        selected=selected,
        selection_path=selection_path,
        output_dir=output_dir,
        gaussian_cache_dir=(
            Path(args.gaussian_cache_dir).expanduser()
            if args.gaussian_cache_dir
            else None
        ),
    )

    cameras = make_camera_ring(int(args.view_count), radius=2.7, elev_deg=18.0)
    rastctx = (
        _make_headless_raster_context()
        if args.renderer_backend == "nvdiffrast" or not args.skip_world_projection
        else None
    )
    case_reports = []
    for index, selection in enumerate(selected, start=1):
        key = selection["object_key"]
        label = label_by_key[key]
        prefix = "最佳" if selection["selection"] == "posthoc_best" else "随机"
        case_dir = output_dir / f"{index:02d}_{prefix}_{_safe_name(key)}"
        temporary_asset_root = case_dir / "._临时视频渲染资产"
        mesh_root = (
            temporary_asset_root if args.turntable_video_only else case_dir / "Mesh原始资产"
        )
        inputs = (
            []
            if args.turntable_video_only
            else _copy_inputs(label, case_dir / "输入8视图")
        )
        methods: dict[str, dict[str, Any]] = {
            "gt": _prepare_gt(
                label,
                mesh_root / METHOD_DIRS["gt"],
                copy_assets=not bool(args.turntable_video_only),
            )
        }
        for method in METHOD_ORDER[1:]:
            methods[method] = _prepare_prediction(
                records[(key, method)],
                mesh_root / METHOD_DIRS[method],
                copy_assets=not bool(args.turntable_video_only),
            )
        current_gaussian = (
            _reference_current_gaussian_assets(gaussian_by_key[key])
            if args.turntable_video_only
            else _copy_current_gaussian_assets(
                gaussian_by_key[key],
                case_dir=case_dir,
                current_mesh_dir=mesh_root / METHOD_DIRS[CURRENT_METHOD],
            )
        )
        center, scale = _mesh_bounds(Path(label["mesh_o"]))
        individual_render_root = case_dir / "渲染"
        if args.group_renders_by_view:
            individual_render_root = individual_render_root / "_单路中间产物"
        appearance_images: dict[str, list[Path]] = {}
        clay_images: dict[str, list[Path]] = {}
        appearance_sheet = case_dir / "自身纹理或顶点颜色_6视图对比.png"
        clay_sheet = case_dir / "统一材质几何_6视图对比.png"
        appearance_video = case_dir / "渲染" / "六路同步环绕_自身纹理或顶点颜色.mp4"
        clay_video = case_dir / "渲染" / "六路同步环绕_统一材质几何.mp4"
        videos_reusable = bool(
            args.turntable_video_only
            and args.resume
            and appearance_video.is_file()
            and clay_video.is_file()
        )
        if not videos_reusable:
            if args.turntable_video_only:
                individual_render_root = case_dir / "._临时环绕帧"
            for method in METHOD_ORDER:
                display_mesh = Path(methods[method]["display_mesh"])
                appearance_mesh = (
                    Path(
                        current_gaussian["branches"]["full"]
                        ["gs_baked_textured_glb"]["path"]
                    )
                    if method == CURRENT_METHOD
                    else Path(methods[method].get("_appearance_mesh", display_mesh))
                )
                appearance_images[method] = _render_views(
                    backend=str(args.renderer_backend),
                    rastctx=rastctx,
                    blender=blender,
                    renderer=renderer,
                    mesh=appearance_mesh,
                    output_dir=(
                        individual_render_root
                        / "自身纹理或顶点颜色"
                        / METHOD_DIRS[method]
                    ),
                    center=center,
                    scale=scale,
                    cameras=cameras,
                    image_size=int(args.image_size),
                    material_mode="source",
                    resume=bool(args.resume),
                )
                clay_images[method] = _render_views(
                    backend=str(args.renderer_backend),
                    rastctx=rastctx,
                    blender=blender,
                    renderer=renderer,
                    mesh=display_mesh,
                    output_dir=(
                        individual_render_root
                        / "统一材质几何"
                        / METHOD_DIRS[method]
                    ),
                    center=center,
                    scale=scale,
                    cameras=cameras,
                    image_size=int(args.image_size),
                    material_mode="clay",
                    resume=bool(args.resume),
                )
            if args.turntable_video_only:
                _write_turntable_video(
                    appearance_images,
                    appearance_video,
                    fps=int(args.turntable_fps),
                )
                _write_turntable_video(
                    clay_images,
                    clay_video,
                    fps=int(args.turntable_fps),
                )
                shutil.rmtree(individual_render_root)
            else:
                _comparison_sheet(
                    appearance_images,
                    appearance_sheet,
                    view_count=int(args.view_count),
                    method_order=METHOD_ORDER,
                )
                _comparison_sheet(
                    clay_images, clay_sheet, view_count=int(args.view_count)
                )
        grouped_appearance_views: list[Path] = []
        grouped_clay_views: list[Path] = []
        if args.group_renders_by_view:
            grouped_root = case_dir / "渲染" / "六路并排对比"
            grouped_appearance_views = _per_view_comparison_sheets(
                appearance_images,
                grouped_root / "自身纹理或顶点颜色",
                view_count=int(args.view_count),
            )
            grouped_clay_views = _per_view_comparison_sheets(
                clay_images,
                grouped_root / "统一材质几何",
                view_count=int(args.view_count),
            )
        projection_contract = None
        projection_images: dict[str, list[Path]] = {}
        projection_ious: dict[str, list[float]] = {}
        grouped_projection_views: list[Path] = []
        projection_sheet = case_dir / "物理To2w_Tw2c投影到原始8视图对比.png"
        if not args.skip_world_projection:
            projection_contract = _load_projection_contract(label)
            for method in METHOD_ORDER:
                projection_images[method], projection_ious[method] = (
                    _render_world_projection_overlays(
                        method=method,
                        mesh=Path(methods[method]["display_mesh"]),
                        contract=projection_contract,
                        output_dir=(
                            case_dir
                            / "原始图像世界坐标投影叠加"
                            / METHOD_DIRS[method]
                        ),
                        rastctx=rastctx,
                        resume=bool(args.resume),
                    )
                )
            _projection_comparison_sheet(
                projection_images, projection_ious, projection_sheet
            )
            if args.group_renders_by_view:
                grouped_projection_views = _per_view_projection_sheets(
                    projection_images,
                    projection_ious,
                    case_dir / "渲染" / "六路并排对比" / "物理To2w世界投影",
                )
        if args.turntable_video_only:
            if temporary_asset_root.exists():
                shutil.rmtree(temporary_asset_root)
            stale_frame_root = case_dir / "._临时环绕帧"
            if stale_frame_root.exists() and appearance_video.is_file() and clay_video.is_file():
                shutil.rmtree(stale_frame_root)
        public_methods = {
            method: {
                name: value
                for name, value in methods[method].items()
                if not name.startswith("_")
            }
            for method in METHOD_ORDER
        }
        render_outputs = (
            {
                "turntable_video_only": True,
                "render_frames_retained": False,
                "copied_mesh_assets": False,
                "appearance_turntable": {
                    "path": str(appearance_video),
                    "sha256": sha256_file(appearance_video),
                },
                "clay_turntable": {
                    "path": str(clay_video),
                    "sha256": sha256_file(clay_video),
                },
                "frame_count": int(args.view_count),
                "fps": int(args.turntable_fps),
            }
            if args.turntable_video_only
            else {
                "current_gaussian_appearance_sheet": current_gaussian["sheet"],
                "all_methods_appearance_sheet": {
                    "path": str(appearance_sheet),
                    "sha256": sha256_file(appearance_sheet),
                },
                "clay_sheet": {
                    "path": str(clay_sheet),
                    "sha256": sha256_file(clay_sheet),
                },
                "grouped_render_layout": {
                    "enabled": bool(args.group_renders_by_view),
                    "single_method_images_are_intermediate_only": bool(
                        args.group_renders_by_view
                    ),
                    "appearance_per_view": [
                        {"path": str(path), "sha256": sha256_file(path)}
                        for path in grouped_appearance_views
                    ],
                    "clay_per_view": [
                        {"path": str(path), "sha256": sha256_file(path)}
                        for path in grouped_clay_views
                    ],
                    "world_projection_per_view": [
                        {"path": str(path), "sha256": sha256_file(path)}
                        for path in grouped_projection_views
                    ],
                },
            }
        )
        case_report = {
            "format": FORMAT,
            "object_key": key,
            "selection": selection["selection"],
            "aggregate": selection["aggregate"],
            "input_images": inputs,
            "shared_display": {
                "owner": "metric GT runtime-O bounds",
                "center": center,
                "scale": scale,
                "camera_count": int(args.view_count),
                "renderer_backend": str(args.renderer_backend),
                "per_method_alignment": False,
                "per_method_autoframe": False,
            },
            "methods": public_methods,
            "current_gaussian_appearance": {
                "source_report": str(gaussian_report_path),
                "source_report_sha256": sha256_file(gaussian_report_path),
                **current_gaussian,
            },
            "world_projection": (
                {
                    "formula": (
                        "Mesh_O -> T_O2W -> Mesh_W -> T_W2C -> "
                        "K_raw + frozen COLMAP camera distortion"
                    ),
                    "image_semantics": "original registered 1920x1080 RGB frames",
                    "physical_T_O2W_and_raw_T_W2C": True,
                    "T_O2C_lifting_used": False,
                    "inverse_roundtrip_max_abs": projection_contract[
                        "inverse_roundtrip_max_abs"
                    ],
                    "chain_audit": projection_contract["chain_audit"],
                    "bindings": projection_contract["bindings"],
                    "frame_names": projection_contract["frame_names"],
                    "method_mean_silhouette_iou": {
                        method: float(np.mean(projection_ious[method]))
                        for method in METHOD_ORDER
                    },
                    "sheet": {
                        "path": str(projection_sheet),
                        "sha256": sha256_file(projection_sheet),
                    },
                }
                if projection_contract is not None
                else {"skipped": True}
            ),
            "outputs": {
                **render_outputs,
                **(
                    {
                        "world_projection_sheet": {
                            "path": str(projection_sheet),
                            "sha256": sha256_file(projection_sheet),
                        }
                    }
                    if projection_sheet.is_file()
                    else {}
                ),
            },
            "scope_guard": (
                "Display-only asset bundle. The main Current No-VGGT appearance "
                "column uses Full SLat decoded by the frozen Gaussian decoder and "
                "GS-baked onto the matching frozen Mesh decoder output; its legacy "
                "OBJ vertex colors are excluded and input RGB is reference-only. "
                "Other methods keep their native appearance; "
                "use the shared-clay turntable/sheet for geometry comparison. The world "
                "projection is a post-hoc alignment diagnostic and does not alter "
                "any Mesh, metric, T_O2W, camera, checkpoint, or formal decision."
            ),
            "passed": True,
        }
        atomic_json(case_dir / "对象报告.json", case_report)
        case_reports.append({
            "object_key": key,
            "selection": selection["selection"],
            "directory": str(case_dir),
            "report": str(case_dir / "对象报告.json"),
        })
        print(f"[holdout_mesh_review] {index}/{len(selected)} {key}", flush=True)

    final = {
        **selection_manifest,
        "cases": case_reports,
        "appearance_semantics": {
            "final_native_no_vggt": (
                "Full SLat decoded by the frozen TRELLIS Gaussian decoder and "
                "GS-baked onto the matching frozen TRELLIS Mesh decoder output for "
                "the main six-method camera-ring render; legacy OBJ vertex RGB is "
                "explicitly excluded from current-model texture evidence"
            ),
            "stock_reference": (
                "matched Stock SLat decoded with the same frozen Gaussian decoder, "
                "same Native-SS coordinates and same within-run initial noise"
            ),
            "input_rgb": (
                "reference only; never projected onto the predicted Mesh as texture"
            ),
            "gt": (
                "UV texture from Omni Scan.mtl + Scan.jpg; the default "
                "nvdiffrast backend samples it at Mesh vertices"
            ),
            "pixal3d_official": (
                "embedded GLB texture/material; the default nvdiffrast backend "
                "samples its base-color texture at Mesh vertices"
            ),
            "other_native_and_reconviagen": (
                "legacy/native source appearance only; decoded per-vertex RGB when "
                "present and kept separate from current-model GS evidence"
            ),
            "geometry_fallback": "source appearance may be material-only when no texture/color exists",
        },
        "world_projection_semantics": {
            "formula": (
                "Mesh_O -> physical T_O2W -> Mesh_W -> raw T_W2C -> "
                "K_raw + frozen COLMAP camera distortion"
            ),
            "image_semantics": "original registered RGB frame, not the 518x518 crop",
            "yellow_outline": "observed original-image mask",
            "method_color": "projected Mesh silhouette fill and outline",
            "T_O2C_lifting_used": False,
            "role": "post-hoc qualitative alignment diagnostic only",
        },
        "passed": True,
    }
    atomic_json(output_dir / "report.json", final)
    _html_index(
        output_dir,
        case_reports,
        source_scope=str(report.get("protocol_scope")),
    )
    print(json.dumps({"passed": True, "output_dir": str(output_dir), "cases": case_reports}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
