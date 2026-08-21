#!/usr/bin/env python3
"""Render synchronized normal turntables for one AR Mesh and ReconViaGen Mesh.

The source meshes are never modified.  Because the current prediction is in AR
world coordinates while the original ReconViaGen prediction is in its canonical
object frame, each mesh is independently bbox-centered and isotropically scaled
for this display-only shape comparison.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from pose_point_depth_mv.bunny_review.common import (
    atomic_json,
    binding,
    validate_binding,
)
from pose_point_depth_mv.bunny_review.finalize import (
    comparison_contact_sheet,
    comparison_frames,
    load_mesh,
    render_method,
    save_video_atomic,
)
from pose_point_depth_mv.dataset_tools.prepare_omni_real_video_cache import utc_now


FORMAT = "pose_point_depth_mv.ar_reconviagen_turntable_comparison.v1"


def _validate_positive(name: str, value: int | float) -> None:
    if float(value) <= 0.0:
        raise ValueError(f"{name} must be positive")


def _reusable(
    report: dict[str, Any],
    *,
    current_mesh: dict[str, Any],
    reconviagen_mesh: dict[str, Any],
    render_config: dict[str, Any],
) -> bool:
    if (
        report.get("format") != FORMAT
        or report.get("passed") is not True
        or report.get("current_mesh") != current_mesh
        or report.get("reconviagen_mesh") != reconviagen_mesh
        or report.get("render_config") != render_config
    ):
        return False
    try:
        for key in (
            "current_turntable",
            "reconviagen_turntable",
            "side_by_side_turntable",
            "side_by_side_contact_sheet",
        ):
            validate_binding(report[key], label=key)
    except (FileNotFoundError, KeyError, RuntimeError, TypeError, ValueError):
        return False
    return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current_mesh", required=True)
    parser.add_argument("--reconviagen_mesh", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--render_frames", type=int, default=72)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--fps", type=int, default=18)
    parser.add_argument("--display_margin", type=float, default=1.0)
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    _validate_positive("render_frames", args.render_frames)
    _validate_positive("resolution", args.resolution)
    _validate_positive("fps", args.fps)
    _validate_positive("display_margin", args.display_margin)
    if int(args.resolution) < 64:
        raise ValueError("resolution must be at least 64")

    current_path = Path(args.current_mesh).expanduser().resolve()
    reconviagen_path = Path(args.reconviagen_mesh).expanduser().resolve()
    current_binding = binding(current_path)
    reconviagen_binding = binding(reconviagen_path)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "report.json"
    render_config = {
        "render_frames": int(args.render_frames),
        "resolution": int(args.resolution),
        "fps": int(args.fps),
        "display_margin": float(args.display_margin),
        "render_pass": "normal",
        "camera": {"radius": 2.0, "fov_degrees": 40.0, "pitch_radians": 0.25},
        "camera_path": "synchronized fixed-pitch yaw turntable",
        "display_normalization": (
            "independent bbox centering and isotropic scaling for shape display only"
        ),
    }

    if args.resume and report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if _reusable(
            report,
            current_mesh=current_binding,
            reconviagen_mesh=reconviagen_binding,
            render_config=render_config,
        ):
            print(
                json.dumps(
                    {
                        "passed": True,
                        "reused": True,
                        "report": str(report_path),
                        "side_by_side": report["side_by_side_turntable"]["path"],
                    },
                    indent=2,
                    ensure_ascii=False,
                )
            )
            return

    output_paths = {
        "current_turntable": output_dir / "当前NoVGGT_normal环绕.mp4",
        "reconviagen_turntable": output_dir / "ReconViaGen原版_normal环绕.mp4",
        "side_by_side_turntable": output_dir
        / "当前NoVGGT_vs_ReconViaGen原版_normal并排.mp4",
        "side_by_side_contact_sheet": output_dir / "当前与ReconViaGen_六视图对比抽帧.png",
    }
    partial = [str(path) for path in output_paths.values() if path.exists()]
    if partial:
        raise RuntimeError(
            "partial or stale comparison output exists; preserve and inspect it, or use "
            f"a new immutable output directory: {partial}"
        )

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA rendering requested but CUDA is unavailable")

    current = load_mesh(current_path)
    reconviagen = load_mesh(reconviagen_path)
    current_frames, current_display = render_method(
        current,
        device=device,
        frames=int(args.render_frames),
        resolution=int(args.resolution),
        display_margin=float(args.display_margin),
    )
    reconviagen_frames, reconviagen_display = render_method(
        reconviagen,
        device=device,
        frames=int(args.render_frames),
        resolution=int(args.resolution),
        display_margin=float(args.display_margin),
    )
    save_video_atomic(current_frames, output_paths["current_turntable"], int(args.fps))
    save_video_atomic(
        reconviagen_frames,
        output_paths["reconviagen_turntable"],
        int(args.fps),
    )
    combined = comparison_frames(
        [
            ("Current No-VGGT", current_frames),
            ("ReconViaGen original", reconviagen_frames),
        ]
    )
    save_video_atomic(combined, output_paths["side_by_side_turntable"], int(args.fps))
    comparison_contact_sheet(
        combined,
        output_paths["side_by_side_contact_sheet"],
        count=6,
    )

    report = {
        "format": FORMAT,
        "created_at_utc": utc_now(),
        "current_mesh": current_binding,
        "reconviagen_mesh": reconviagen_binding,
        "current_turntable": binding(output_paths["current_turntable"]),
        "reconviagen_turntable": binding(output_paths["reconviagen_turntable"]),
        "side_by_side_turntable": binding(output_paths["side_by_side_turntable"]),
        "side_by_side_contact_sheet": binding(
            output_paths["side_by_side_contact_sheet"]
        ),
        "render_config": render_config,
        "display_transforms": {
            "current_no_vggt": current_display,
            "reconviagen_original": reconviagen_display,
        },
        "source_meshes_modified": False,
        "display_only": True,
        "world_pose_or_scale_comparison": False,
        "shape_comparison": True,
        "passed": True,
    }
    atomic_json(report_path, report)
    print(
        json.dumps(
            {
                "passed": True,
                "reused": False,
                "report": str(report_path),
                "current_video": str(output_paths["current_turntable"]),
                "reconviagen_video": str(output_paths["reconviagen_turntable"]),
                "side_by_side": str(output_paths["side_by_side_turntable"]),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
