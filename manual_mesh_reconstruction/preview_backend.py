"""Self-contained display-only normal renderer used by manual Mesh reviews."""

from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import trimesh


TRACKER_ROOT = Path(__file__).resolve().parents[1]
RECONVIAGEN_ROOT = TRACKER_ROOT / "ReconViaGen"
VGGT_ROOT = RECONVIAGEN_ROOT / "wheels/vggt"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def binding(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).resolve(strict=True)
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "bytes": int(resolved.stat().st_size),
    }


def validate_binding(value: dict[str, Any], *, label: str) -> Path:
    if not isinstance(value, dict):
        raise TypeError(f"{label} binding is not a dictionary")
    path = Path(str(value.get("path", ""))).resolve(strict=True)
    if sha256_file(path) != str(value.get("sha256", "")):
        raise RuntimeError(f"{label} changed after freeze: {path}")
    if "bytes" in value and int(value["bytes"]) != int(path.stat().st_size):
        raise RuntimeError(f"{label} size changed after freeze: {path}")
    return path


def load_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, force="scene", process=False)
    if isinstance(loaded, trimesh.Scene):
        pieces = [
            item
            for item in loaded.dump(concatenate=False)
            if isinstance(item, trimesh.Trimesh)
            and len(item.vertices)
            and len(item.faces)
        ]
        if not pieces:
            raise ValueError(f"Mesh scene has no triangles: {path}")
        mesh = trimesh.util.concatenate(pieces)
    elif isinstance(loaded, trimesh.Trimesh):
        mesh = loaded.copy()
    else:
        raise TypeError(f"unsupported Mesh object={type(loaded)} path={path}")
    mesh.remove_unreferenced_vertices()
    if not len(mesh.vertices) or not len(mesh.faces):
        raise ValueError(f"empty Mesh: {path}")
    if not np.isfinite(np.asarray(mesh.vertices)).all():
        raise ValueError(f"non-finite Mesh vertices: {path}")
    return mesh


def display_normalize(
    mesh: trimesh.Trimesh, margin: float
) -> tuple[trimesh.Trimesh, dict[str, Any]]:
    bounds = np.asarray(mesh.bounds, dtype=np.float64)
    center = bounds.mean(axis=0)
    extent = bounds[1] - bounds[0]
    scale = float(np.max(extent))
    if not np.isfinite(scale) or scale <= 1.0e-10:
        raise ValueError("Mesh has invalid display scale")
    factor = float(margin) / scale
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] *= factor
    matrix[:3, 3] = -factor * center
    normalized = mesh.copy()
    normalized.apply_transform(matrix)
    return normalized, {
        "matrix": matrix.tolist(),
        "native_bounds": bounds.tolist(),
        "native_center": center.tolist(),
        "native_max_extent": scale,
        "display_margin": float(margin),
        "owner": "self",
        "applied_output_bounds": np.asarray(normalized.bounds).tolist(),
        "policy": "independent bbox centering/isotropic scale for display only",
    }


def _uint8_rgb(value: Any) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != np.uint8:
        maximum = float(np.nanmax(array)) if array.size else 0.0
        array = np.clip(array * (255.0 if maximum <= 1.5 else 1.0), 0, 255)
        array = array.astype(np.uint8)
    if array.ndim != 3 or array.shape[2] < 3:
        raise ValueError(f"renderer returned invalid frame shape={array.shape}")
    return array[:, :, :3]


def save_image_atomic(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp-{os.getpid()}.png")
    image.save(temporary)
    os.replace(temporary, path)


def _labeled_frame(frame: np.ndarray, label: str, header: int = 34) -> Image.Image:
    image = Image.fromarray(frame)
    output = Image.new("RGB", (image.width, image.height + header), (28, 28, 28))
    output.paste(image, (0, header))
    ImageDraw.Draw(output).text(
        (8, 10), label, fill=(245, 245, 245), font=ImageFont.load_default()
    )
    return output


def contact_sheet(
    frames: list[np.ndarray], label: str, destination: Path, *, count: int
) -> None:
    chosen = np.linspace(0, len(frames) - 1, min(count, len(frames)), dtype=int)
    labeled = [
        _labeled_frame(frames[index], f"{label}  frame {index}") for index in chosen
    ]
    columns = min(3, len(labeled))
    rows = int(math.ceil(len(labeled) / columns))
    width = max(image.width for image in labeled)
    height = max(image.height for image in labeled)
    sheet = Image.new("RGB", (columns * width, rows * height), (18, 18, 18))
    for index, image in enumerate(labeled):
        sheet.paste(image, ((index % columns) * width, (index // columns) * height))
    save_image_atomic(sheet, destination)


def render_method(
    mesh: trimesh.Trimesh,
    *,
    device: torch.device,
    frames: int,
    resolution: int,
    display_margin: float,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    for path in (RECONVIAGEN_ROOT, VGGT_ROOT):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from trellis.representations import MeshExtractResult
    from trellis.utils import render_utils

    normalized, transform = display_normalize(mesh, display_margin)
    vertices = torch.as_tensor(
        np.asarray(normalized.vertices), dtype=torch.float32, device=device
    )
    faces = torch.as_tensor(
        np.asarray(normalized.faces), dtype=torch.int64, device=device
    )
    attrs = torch.full(
        (vertices.shape[0], 3), 0.72, dtype=torch.float32, device=device
    )
    render_mesh = MeshExtractResult(vertices, faces, attrs)
    output = render_utils.render_video(
        render_mesh,
        resolution=int(resolution),
        ssaa=1,
        bg_color=(0, 0, 0),
        num_frames=int(frames),
        r=2.0,
        fov=40,
        pitch=0.25,
        return_types=["normal", "nocs", "depth", "mask"],
        verbose=False,
    )
    arrays = [_uint8_rgb(frame) for frame in output["normal"]]
    del output, render_mesh, vertices, faces, attrs
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return arrays, transform

