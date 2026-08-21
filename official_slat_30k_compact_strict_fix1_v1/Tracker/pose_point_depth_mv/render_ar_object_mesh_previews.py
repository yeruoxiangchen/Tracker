#!/usr/bin/env python3
"""Render display-only normal previews for one reconstructed AR world Mesh."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch

from pose_point_depth_mv.bunny_review.common import binding, validate_binding
from pose_point_depth_mv.bunny_review.finalize import (
    contact_sheet,
    load_mesh,
    render_method,
    save_image_atomic,
)
from pose_point_depth_mv.dataset_tools.prepare_omni_real_video_cache import (
    utc_now,
    write_json,
)


FORMAT = "pose_point_depth_mv.ar_object_mesh_previews.v1"


def _reusable(
    report: dict[str, Any],
    *,
    source_mesh: dict[str, Any],
    render_config: dict[str, Any],
) -> bool:
    if (
        report.get("format") != FORMAT
        or report.get("passed") is not True
        or report.get("source_mesh") != source_mesh
        or report.get("render_config") != render_config
    ):
        return False
    images = report.get("images")
    if not isinstance(images, list) or len(images) != int(
        render_config["preview_count"]
    ):
        return False
    try:
        for index, value in enumerate(images):
            validate_binding(value, label=f"AR Mesh preview[{index}]")
        validate_binding(report["contact_sheet"], label="AR Mesh preview contact sheet")
    except (FileNotFoundError, KeyError, RuntimeError, TypeError, ValueError):
        return False
    return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--render_frames", type=int, default=24)
    parser.add_argument("--preview_count", type=int, default=6)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--display_margin", type=float, default=1.0)
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if int(args.preview_count) < 1:
        raise ValueError("preview_count must be positive")
    if int(args.render_frames) < int(args.preview_count):
        raise ValueError("render_frames must be at least preview_count")
    if int(args.resolution) < 64:
        raise ValueError("resolution must be at least 64")
    if float(args.display_margin) <= 0.0:
        raise ValueError("display_margin must be positive")

    mesh_path = Path(args.mesh).expanduser().resolve()
    if not mesh_path.is_file():
        raise FileNotFoundError(mesh_path)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "preview_report.json"
    source_mesh = binding(mesh_path)
    render_config = {
        "render_frames": int(args.render_frames),
        "preview_count": int(args.preview_count),
        "resolution": int(args.resolution),
        "display_margin": float(args.display_margin),
        "render_pass": "normal",
        "camera": {"radius": 2.0, "fov_degrees": 40.0, "pitch_radians": 0.25},
        "display_normalization": "independent bbox centering and isotropic scale",
    }
    if args.resume and report_path.is_file():
        existing = json.loads(report_path.read_text(encoding="utf-8"))
        if _reusable(
            existing,
            source_mesh=source_mesh,
            render_config=render_config,
        ):
            print(
                json.dumps(
                    {
                        "passed": True,
                        "reused": True,
                        "preview_count": len(existing["images"]),
                        "report": str(report_path),
                    },
                    indent=2,
                )
            )
            return

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA Mesh preview rendering requested but CUDA is unavailable")
    mesh = load_mesh(mesh_path)
    frames, display_transform = render_method(
        mesh,
        device=device,
        frames=int(args.render_frames),
        resolution=int(args.resolution),
        display_margin=float(args.display_margin),
    )
    selected_indices = np.floor(
        np.linspace(0, len(frames), int(args.preview_count), endpoint=False)
    ).astype(np.int64)
    selected_frames = [frames[int(index)] for index in selected_indices]
    image_bindings = []
    for position, (frame_index, frame) in enumerate(
        zip(selected_indices.tolist(), selected_frames)
    ):
        image_path = output_dir / f"mesh_view_{position:02d}.png"
        save_image_atomic(Image.fromarray(frame), image_path)
        value = binding(image_path)
        value["turntable_frame_index"] = int(frame_index)
        image_bindings.append(value)
    sheet_path = output_dir / "mesh_views_contact_sheet.png"
    contact_sheet(
        selected_frames,
        "No-VGGT world Mesh",
        sheet_path,
        count=len(selected_frames),
    )
    report = {
        "format": FORMAT,
        "created_at_utc": utc_now(),
        "source_mesh": source_mesh,
        "images": image_bindings,
        "contact_sheet": binding(sheet_path),
        "render_config": render_config,
        "display_transform": display_transform,
        "display_only": True,
        "world_scale_or_pose_changed": False,
        "passed": True,
    }
    write_json(report_path, report)
    print(
        json.dumps(
            {
                "passed": True,
                "reused": False,
                "preview_count": len(image_bindings),
                "report": str(report_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
