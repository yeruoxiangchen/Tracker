#!/usr/bin/env python3
"""Freeze Bunny inputs and create deterministic RGBA/mask review assets."""

from __future__ import annotations

import argparse
from collections import deque
import json
from pathlib import Path
import shutil
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .common import (
    PROTOCOL_FORMAT,
    atomic_copy,
    atomic_json,
    binding,
    canonical_sha256,
    code_bindings,
    load_protocol,
    parse_int_csv,
    write_method_result,
)


TRACKER_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BUNNY_ROOT = TRACKER_ROOT / "pose_point_depth_mv" / "bunny"


def border_connected_white(rgb: np.ndarray, threshold: int) -> np.ndarray:
    """Return the near-white component connected to any image border."""

    candidate = np.all(rgb >= int(threshold), axis=2)
    height, width = candidate.shape
    background = np.zeros_like(candidate, dtype=bool)
    queue: deque[tuple[int, int]] = deque()
    for x in range(width):
        if candidate[0, x]:
            queue.append((0, x))
        if candidate[height - 1, x]:
            queue.append((height - 1, x))
    for y in range(height):
        if candidate[y, 0]:
            queue.append((y, 0))
        if candidate[y, width - 1]:
            queue.append((y, width - 1))
    while queue:
        y, x = queue.popleft()
        if background[y, x] or not candidate[y, x]:
            continue
        background[y, x] = True
        if y:
            queue.append((y - 1, x))
        if y + 1 < height:
            queue.append((y + 1, x))
        if x:
            queue.append((y, x - 1))
        if x + 1 < width:
            queue.append((y, x + 1))
    return background


def make_rgba(source: Path, *, white_threshold: int) -> tuple[Image.Image, Image.Image]:
    rgb_image = Image.open(source).convert("RGB")
    rgb = np.asarray(rgb_image)
    background = border_connected_white(rgb, white_threshold)
    alpha = np.where(background, 0, 255).astype(np.uint8)
    rgba = np.concatenate((rgb, alpha[:, :, None]), axis=2)
    return Image.fromarray(rgba, mode="RGBA"), Image.fromarray(alpha, mode="L")


def save_image_atomic(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.png")
    image.save(temporary)
    temporary.replace(path)


def input_contact_sheet(views: list[dict[str, Any]], destination: Path) -> None:
    images = [Image.open(view["rgba"]["path"]).convert("RGBA") for view in views]
    thumb_width, thumb_height = 320, 230
    label_height = 32
    gap = 8
    sheet = Image.new(
        "RGB",
        (
            gap + len(images) * (thumb_width + gap),
            gap + label_height + thumb_height + gap,
        ),
        (35, 35, 35),
    )
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for position, (image, view) in enumerate(zip(images, views)):
        canvas = Image.new("RGB", image.size, (255, 255, 255))
        canvas.paste(image.convert("RGB"), mask=image.getchannel("A"))
        canvas.thumbnail((thumb_width, thumb_height), Image.Resampling.LANCZOS)
        x = gap + position * (thumb_width + gap)
        y = gap + label_height
        paste_x = x + (thumb_width - canvas.width) // 2
        paste_y = y + (thumb_height - canvas.height) // 2
        sheet.paste(canvas, (paste_x, paste_y))
        draw.text(
            (x + 6, gap + 8),
            f"view {view['view_index']}  {view['role']}",
            fill=(240, 240, 240),
            font=font,
        )
    save_image_atomic(sheet, destination)


def copy_reference_asset(bunny_root: Path, output_dir: Path) -> Path:
    source_obj = bunny_root / "meshes" / "model.obj"
    source_mtl = bunny_root / "meshes" / "model.mtl"
    source_texture = bunny_root / "materials" / "textures" / "texture.png"
    destination = output_dir / "methods" / "reference" / "mesh"
    atomic_copy(source_obj, destination / "model.obj")
    atomic_copy(source_mtl, destination / "model.mtl")
    atomic_copy(source_texture, destination / "texture.png")
    return destination / "model.obj"


def prepare(args: argparse.Namespace) -> None:
    bunny_root = args.bunny_root.resolve()
    output_dir = args.output_dir.resolve()
    protocol_path = output_dir / "protocol.json"
    indices = parse_int_csv(args.view_indices)
    if args.single_view_index not in indices:
        raise ValueError("--single_view_index must be included in --view_indices")
    source_paths = [bunny_root / "thumbnails" / f"{index}.jpg" for index in indices]
    for path in source_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    if protocol_path.is_file():
        protocol = load_protocol(protocol_path)
        expected = [int(view["view_index"]) for view in protocol["views"]]
        if expected != indices or int(protocol["single_view_index"]) != int(
            args.single_view_index
        ):
            raise RuntimeError(
                "existing immutable protocol uses different view selection; "
                "choose a new output directory"
            )
        print(
            json.dumps(
                {
                    "status": "reused",
                    "protocol": str(protocol_path),
                    "protocol_sha256": protocol["protocol_sha256"],
                    "views": expected,
                },
                indent=2,
            )
        )
        return
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(
            f"partial non-empty output exists without protocol; preserve and inspect: "
            f"{output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    view_rows: list[dict[str, Any]] = []
    for index, source in zip(indices, source_paths):
        rgba, mask = make_rgba(source, white_threshold=args.white_threshold)
        rgba_path = output_dir / "inputs" / f"view_{index:02d}.png"
        mask_path = output_dir / "inputs" / f"view_{index:02d}_mask.png"
        save_image_atomic(rgba, rgba_path)
        save_image_atomic(mask, mask_path)
        foreground = int(np.count_nonzero(np.asarray(mask)))
        view_rows.append(
            {
                "view_index": int(index),
                "role": (
                    "Pixal3D single view + multiview"
                    if int(index) == int(args.single_view_index)
                    else "multiview only"
                ),
                "source": binding(source),
                "rgba": binding(rgba_path),
                "mask": binding(mask_path),
                "width": int(rgba.width),
                "height": int(rgba.height),
                "foreground_pixels": foreground,
                "foreground_fraction": float(foreground / (rgba.width * rgba.height)),
                "camera": None,
            }
        )
    sheet_path = output_dir / "inputs" / "input_contact_sheet.png"
    input_contact_sheet(view_rows, sheet_path)
    reference_copy = copy_reference_asset(bunny_root, output_dir)
    reference_source = bunny_root / "meshes" / "model.obj"
    body = {
        "format": PROTOCOL_FORMAT,
        "formal": False,
        "purpose": (
            "manual side-by-side Bunny Racer reconstruction review; quantitative "
            "ranking is not the primary decision signal"
        ),
        "bunny_root": str(bunny_root),
        "asset_name": "BUNNY_RACER",
        "single_view_index": int(args.single_view_index),
        "view_indices": indices,
        "views": view_rows,
        "input_contact_sheet": binding(sheet_path),
        "background_policy": {
            "name": "border_connected_near_white_v1",
            "white_threshold": int(args.white_threshold),
            "internal_white_regions_remain_foreground": True,
            "rembg_network_required": False,
        },
        "camera_policy": {
            "calibrated": False,
            "statement": (
                "Google Scanned Objects thumbnails provide no camera intrinsics/"
                "extrinsics here; no synthetic camera labels are attached."
            ),
        },
        "reference": {
            "mesh": binding(reference_source),
            "review_copy": binding(reference_copy),
            "texture": binding(bunny_root / "materials" / "textures" / "texture.png"),
            "use": "visual reference only; never a model input unless explicitly declared",
        },
        "method_contract": {
            "required_format": "pose_point_depth_mv.bunny_method_result.v1",
            "default_methods": [
                "reference",
                "pixal3d",
                "reconviagen_stock",
                "trained_full",
            ],
            "replaceability": (
                "trained inference may change freely; it must only emit a bound "
                "Mesh and result.json matching this protocol hash"
            ),
        },
        "code_bindings": code_bindings(
            {
                "prepare": Path(__file__).resolve(),
                "common": Path(__file__).resolve().with_name("common.py"),
            }
        ),
        "guardrails": [
            "Pixal3D consumes one frozen view; ReconViaGen/trained methods may consume all.",
            "The five thumbnails have unknown camera calibration.",
            "Reference Mesh is for human comparison and optional metrics, not hidden input.",
            "A Direct-SS/Direct-SLAT Full run must declare its sparse-point/TM2W source.",
        ],
    }
    body["protocol_sha256"] = canonical_sha256(body)
    atomic_json(protocol_path, body)
    write_method_result(
        protocol_path=protocol_path,
        method_id="reference",
        display_name="Reference scan",
        mesh_path=reference_copy,
        input_view_indices=[],
        backend={
            "kind": "frozen_reference",
            "source_mesh": binding(reference_source),
            "copied_material": binding(
                output_dir / "methods" / "reference" / "mesh" / "model.mtl"
            ),
            "copied_texture": binding(
                output_dir / "methods" / "reference" / "mesh" / "texture.png"
            ),
        },
        notes=["Reference scan is not an inferred result."],
    )
    print(
        json.dumps(
            {
                "status": "prepared",
                "protocol": str(protocol_path),
                "protocol_sha256": body["protocol_sha256"],
                "views": indices,
                "single_view_index": int(args.single_view_index),
                "input_contact_sheet": str(sheet_path),
            },
            indent=2,
        )
    )


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bunny_root", type=Path, default=DEFAULT_BUNNY_ROOT)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--view_indices", default="0,1,2,3,4")
    parser.add_argument("--single_view_index", type=int, default=0)
    parser.add_argument("--white_threshold", type=int, default=245)
    return parser


def main() -> None:
    args = make_parser().parse_args()
    if not 1 <= int(args.white_threshold) <= 254:
        raise ValueError("--white_threshold must be in [1,254]")
    prepare(args)


if __name__ == "__main__":
    main()
