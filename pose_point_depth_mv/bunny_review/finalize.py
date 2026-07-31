#!/usr/bin/env python3
"""Render all Bunny Meshes with one camera path and build a side-by-side review."""

from __future__ import annotations

import argparse
import gc
import html
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import trimesh

from .common import (
    REPORT_FORMAT,
    atomic_json,
    atomic_text,
    binding,
    canonical_sha256,
    code_bindings,
    load_method_result,
    load_protocol,
    method_dir,
    sha256_file,
)


TRACKER_ROOT = Path(__file__).resolve().parents[2]
RECONVIAGEN_ROOT = TRACKER_ROOT / "ReconViaGen"
VGGT_ROOT = RECONVIAGEN_ROOT / "wheels" / "vggt"


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


def display_transform_from_mesh(
    mesh: trimesh.Trimesh,
    margin: float,
    *,
    owner: str = "self",
) -> dict[str, Any]:
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
    return {
        "matrix": matrix.tolist(),
        "native_bounds": bounds.tolist(),
        "native_center": center.tolist(),
        "native_max_extent": scale,
        "display_margin": float(margin),
        "owner": str(owner),
        "policy": (
            "bbox centering/isotropic scale derived once from the owner Mesh; "
            "the matrix may be shared by multiple methods"
        ),
    }


def apply_display_transform(
    mesh: trimesh.Trimesh,
    transform: dict[str, Any],
) -> tuple[trimesh.Trimesh, dict[str, Any]]:
    matrix = np.asarray(transform.get("matrix"), dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("display transform must contain a finite 4x4 matrix")
    if not np.allclose(matrix[3], np.array([0.0, 0.0, 0.0, 1.0])):
        raise ValueError("display transform must be affine")
    result = mesh.copy()
    input_bounds = np.asarray(result.bounds, dtype=np.float64)
    result.apply_transform(matrix)
    audit = dict(transform)
    audit["applied_input_bounds"] = input_bounds.tolist()
    audit["applied_output_bounds"] = np.asarray(result.bounds, dtype=np.float64).tolist()
    return result, audit


def display_normalize(mesh: trimesh.Trimesh, margin: float) -> tuple[trimesh.Trimesh, dict]:
    transform = display_transform_from_mesh(mesh, margin, owner="self")
    result, audit = apply_display_transform(mesh, transform)
    audit["policy"] = "independent bbox centering/isotropic scale for display only"
    return result, audit


def mesh_stats(mesh: trimesh.Trimesh) -> dict[str, Any]:
    return {
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        # ``split`` materializes one Trimesh per connected component and can
        # transiently consume tens of GiB for official remeshed GLBs.  Trimesh
        # already exposes the same graph count without copying component
        # geometry.
        "components": int(mesh.body_count),
        "watertight": bool(mesh.is_watertight),
        "winding_consistent": bool(mesh.is_winding_consistent),
        "euler_number": int(mesh.euler_number),
        "bounds": np.asarray(mesh.bounds, dtype=np.float64).tolist(),
        "extent": np.asarray(mesh.extents, dtype=np.float64).tolist(),
    }


def uint8_rgb(value: Any) -> np.ndarray:
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


def save_video_atomic(frames: list[np.ndarray], path: Path, fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp-{os.getpid()}{path.suffix}")
    imageio.mimsave(temporary, frames, fps=int(fps))
    os.replace(temporary, path)


def labeled_frame(frame: np.ndarray, label: str, *, header: int = 34) -> Image.Image:
    image = Image.fromarray(frame)
    output = Image.new("RGB", (image.width, image.height + header), (28, 28, 28))
    output.paste(image, (0, header))
    draw = ImageDraw.Draw(output)
    draw.text((8, 10), label, fill=(245, 245, 245), font=ImageFont.load_default())
    return output


def contact_sheet(
    frames: list[np.ndarray],
    label: str,
    destination: Path,
    *,
    count: int,
) -> None:
    chosen = np.linspace(0, len(frames) - 1, min(count, len(frames)), dtype=int)
    labeled = [labeled_frame(frames[index], f"{label}  frame {index}") for index in chosen]
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
    shared_display_transform: dict[str, Any] | None = None,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    for path in (RECONVIAGEN_ROOT, VGGT_ROOT):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from trellis.representations import MeshExtractResult
    from trellis.utils import render_utils

    if shared_display_transform is None:
        normalized, transform = display_normalize(mesh, display_margin)
    else:
        normalized, transform = apply_display_transform(
            mesh,
            shared_display_transform,
        )
        transform["policy"] = (
            "shared owner-derived centering/isotropic scale for display only; "
            "no per-method bbox normalization"
        )
    vertices = torch.as_tensor(
        np.asarray(normalized.vertices),
        dtype=torch.float32,
        device=device,
    )
    faces = torch.as_tensor(
        np.asarray(normalized.faces),
        dtype=torch.int64,
        device=device,
    )
    attrs = torch.full(
        (vertices.shape[0], 3),
        0.72,
        dtype=torch.float32,
        device=device,
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
        # Vendored render_utils currently materializes these keys unconditionally.
        return_types=["normal", "nocs", "depth", "mask"],
        verbose=False,
    )
    arrays = [uint8_rgb(frame) for frame in output["normal"]]
    del output, render_mesh, vertices, faces, attrs
    torch.cuda.empty_cache()
    return arrays, transform


def comparison_frames(
    method_frames: list[tuple[str, list[np.ndarray]]],
) -> list[np.ndarray]:
    count = min(len(frames) for _, frames in method_frames)
    output: list[np.ndarray] = []
    for frame_index in range(count):
        images = [
            labeled_frame(frames[frame_index], label)
            for label, frames in method_frames
        ]
        width = sum(image.width for image in images)
        height = max(image.height for image in images)
        row = Image.new("RGB", (width, height), (18, 18, 18))
        x = 0
        for image in images:
            row.paste(image, (x, 0))
            x += image.width
        output.append(np.asarray(row))
    return output


def comparison_contact_sheet(
    frames: list[np.ndarray],
    destination: Path,
    *,
    count: int,
) -> None:
    chosen = np.linspace(0, len(frames) - 1, min(count, len(frames)), dtype=int)
    images = [Image.fromarray(frames[index]) for index in chosen]
    width = max(image.width for image in images)
    height = max(image.height for image in images)
    sheet = Image.new("RGB", (width, height * len(images)), (18, 18, 18))
    for position, image in enumerate(images):
        sheet.paste(image, (0, position * height))
    save_image_atomic(sheet, destination)


def html_report(
    *,
    protocol_path: Path,
    review_dir: Path,
    rows: list[dict[str, Any]],
    comparison_sheet: Path,
    comparison_video: Path,
) -> str:
    root = protocol_path.parent

    def relative(path: str | Path) -> str:
        return os.path.relpath(Path(path), review_dir).replace(os.sep, "/")

    method_items = []
    for row in rows:
        result = row["result"]
        method_items.append(
            "<section>"
            f"<h2>{html.escape(result['display_name'])}</h2>"
            f"<img src=\"{html.escape(relative(row['preview']['contact_sheet']['path']))}\">"
            f"<p><a href=\"{html.escape(relative(result['mesh']['path']))}\">Primary Mesh</a>"
            f" · <a href=\"{html.escape(relative(method_dir(protocol_path, result['method_id']) / 'result.json'))}\">result.json</a></p>"
            f"<pre>{html.escape(json.dumps(row['mesh_stats'], indent=2))}</pre>"
            "</section>"
        )
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>Bunny reconstruction review</title>
<style>
body {{ background:#151515; color:#eee; font-family:sans-serif; margin:24px; }}
a {{ color:#7cc7ff; }} img {{ max-width:100%; border:1px solid #555; }}
section {{ margin:30px 0; padding:16px; background:#222; }}
pre {{ white-space:pre-wrap; }}
</style>
</head>
<body>
<h1>Bunny Racer 人工审查</h1>
<p>所有法线图使用同一相机路径；每个 Mesh 仅为显示而独立居中并做各向同性缩放。</p>
<p><a href="{html.escape(relative(protocol_path))}">protocol.json</a>
 · <a href="{html.escape(relative(root / 'inputs' / 'input_contact_sheet.png'))}">输入视图</a>
 · <a href="{html.escape(relative(comparison_video))}">并排视频</a></p>
<img src="{html.escape(relative(comparison_sheet))}">
{''.join(method_items)}
</body>
</html>
"""


def finalize(args: argparse.Namespace) -> None:
    protocol_path = args.protocol.resolve()
    protocol = load_protocol(protocol_path)
    method_ids = [item.strip() for item in args.methods.split(",") if item.strip()]
    if not method_ids or len(method_ids) != len(set(method_ids)):
        raise ValueError("--methods must be a non-empty unique CSV")
    results = [load_method_result(protocol_path, method_id) for method_id in method_ids]
    review_dir = protocol_path.parent / "comparison" / args.review_id
    report_path = review_dir / "report.json"
    run_config = {
        "protocol_sha256": protocol["protocol_sha256"],
        "method_ids": method_ids,
        "render_frames": int(args.render_frames),
        "render_resolution": int(args.render_resolution),
        "fps": int(args.fps),
        "display_margin": float(args.display_margin),
    }
    run_config["config_sha256"] = canonical_sha256(run_config)
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("run_config") != run_config:
            raise RuntimeError(
                "existing review uses different method/render configuration; "
                "choose a new --review_id"
            )
        print(json.dumps({"status": "reused", "report": str(report_path)}, indent=2))
        return
    if review_dir.exists() and any(review_dir.iterdir()):
        raise RuntimeError(f"partial review exists; preserve and inspect: {review_dir}")
    review_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("the shared nvdiffrast normal renderer requires --device cuda")

    rows: list[dict[str, Any]] = []
    all_frames: list[tuple[str, list[np.ndarray]]] = []
    for position, result in enumerate(results, start=1):
        method_id = result["method_id"]
        print(
            f"[bunny_finalize] {position}/{len(results)} {method_id}",
            flush=True,
        )
        source = Path(result["mesh"]["path"])
        mesh = load_mesh(source)
        stats = mesh_stats(mesh)
        frames, display = render_method(
            mesh,
            device=device,
            frames=int(args.render_frames),
            resolution=int(args.render_resolution),
            display_margin=float(args.display_margin),
        )
        render_dir = review_dir / "methods" / method_id
        video_path = render_dir / "normal_turntable.mp4"
        sheet_path = render_dir / "normal_contact_sheet.png"
        save_video_atomic(frames, video_path, int(args.fps))
        contact_sheet(
            frames,
            result["display_name"],
            sheet_path,
            count=int(args.contact_frames),
        )
        row = {
            "method_id": method_id,
            "result": result,
            "mesh_stats": stats,
            "display_transform": display,
            "preview": {
                "turntable": binding(video_path),
                "contact_sheet": binding(sheet_path),
            },
        }
        rows.append(row)
        all_frames.append((result["display_name"], frames))
        del mesh, frames
        gc.collect()
        torch.cuda.empty_cache()

    combined = comparison_frames(all_frames)
    comparison_video = review_dir / "normal_side_by_side.mp4"
    comparison_sheet = review_dir / "normal_contact_sheet.png"
    save_video_atomic(combined, comparison_video, int(args.fps))
    comparison_contact_sheet(
        combined,
        comparison_sheet,
        count=int(args.contact_frames),
    )
    index_path = review_dir / "index.html"
    atomic_text(
        index_path,
        html_report(
            protocol_path=protocol_path,
            review_dir=review_dir,
            rows=rows,
            comparison_sheet=comparison_sheet,
            comparison_video=comparison_video,
        ),
    )
    report = {
        "format": REPORT_FORMAT,
        "passed": True,
        "formal": False,
        "purpose": "human visual review, not automatic checkpoint selection",
        "protocol": binding(protocol_path),
        "run_config": run_config,
        "comparison": {
            "normal_turntable": binding(comparison_video),
            "normal_contact_sheet": binding(comparison_sheet),
            "html": binding(index_path),
        },
        "methods": rows,
        "render_policy": {
            "camera_path": "shared yaw 0..2pi, fixed pitch=0.25 rad, r=2, fov=40deg",
            "normalization": (
                "each method independently bbox-centered and isotropically scaled "
                "for display; native Mesh files are never modified"
            ),
            "primary_signal": "manual shape/completeness/topology inspection",
        },
        "guardrails": protocol["guardrails"],
        "code_bindings": code_bindings(
            {
                "finalize": Path(__file__).resolve(),
                "common": Path(__file__).resolve().with_name("common.py"),
                "render_utils": RECONVIAGEN_ROOT / "trellis" / "utils" / "render_utils.py",
            }
        ),
    }
    atomic_json(report_path, report)
    print(
        json.dumps(
            {
                "status": "complete",
                "report": str(report_path),
                "html": str(index_path),
                "contact_sheet": str(comparison_sheet),
                "turntable": str(comparison_video),
                "methods": method_ids,
            },
            indent=2,
        )
    )


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument(
        "--methods",
        default="reference,pixal3d,reconviagen_stock,trained_full",
    )
    parser.add_argument("--review_id", default="default")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--render_frames", type=int, default=72)
    parser.add_argument("--render_resolution", type=int, default=320)
    parser.add_argument("--contact_frames", type=int, default=6)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--display_margin", type=float, default=0.9)
    return parser


def main() -> None:
    finalize(make_parser().parse_args())


if __name__ == "__main__":
    main()
