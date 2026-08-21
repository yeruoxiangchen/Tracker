#!/usr/bin/env python3
"""Render independent normal turntables for a packaged CoarseModel Pose+Mask bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from pose_point_depth_mv.bunny_review.common import binding, validate_binding
from pose_point_depth_mv.bunny_review.finalize import (
    contact_sheet,
    load_mesh,
    render_method,
    save_video_atomic,
)
from pose_point_depth_mv.dataset_tools.prepare_omni_real_video_cache import (
    utc_now,
    write_json,
)
from pose_point_depth_mv.package_coarsemodel_real_no_vggt_results import (
    FORMAT as BUNDLE_FORMAT,
)


FORMAT = "pose_point_depth_mv.coarsemodel_pose_mask_turntable.v1"


def reusable_turntable(
    report: dict[str, Any],
    *,
    source_mesh: dict[str, str],
    render_config: dict[str, Any],
) -> bool:
    if (
        report.get("format") != FORMAT
        or report.get("passed") is not True
        or report.get("source_mesh") != source_mesh
        or report.get("render_config") != render_config
    ):
        return False
    for name in ("video", "contact_sheet"):
        value = report.get(name, {})
        path = Path(str(value.get("path", "")))
        if not path.is_file() or binding(path) != value:
            return False
    return True


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle_report", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--render_frames", type=int, default=48)
    parser.add_argument("--render_resolution", type=int, default=512)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--display_margin", type=float, default=1.0)
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> None:
    args = make_parser().parse_args()
    if min(args.render_frames, args.render_resolution, args.fps) <= 0:
        raise ValueError("frames, resolution, and fps must be positive")
    if float(args.display_margin) <= 0.0:
        raise ValueError("display_margin must be positive")
    report_path = Path(args.bundle_report).expanduser().resolve()
    bundle = json.loads(report_path.read_text(encoding="utf-8"))
    if bundle.get("format") != BUNDLE_FORMAT or bundle.get("passed") is not True:
        raise RuntimeError(f"CoarseModel bundle did not pass: {report_path}")
    cases = list(bundle.get("cases", []))
    if not cases or int(bundle.get("object_count", -1)) != len(cases):
        raise RuntimeError("CoarseModel bundle object count differs")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA turntable rendering requested but CUDA is unavailable")
    render_config = {
        "render_frames": int(args.render_frames),
        "render_resolution": int(args.render_resolution),
        "fps": int(args.fps),
        "display_margin": float(args.display_margin),
        "render_pass": "normal",
        "camera": {"radius": 2.0, "fov_degrees": 40.0, "pitch_radians": 0.25},
        "display_normalization": "independent bbox centering and isotropic scale",
    }
    completed: list[dict[str, Any]] = []
    for position, case in enumerate(cases, start=1):
        key = str(case["object_key"])
        source_mesh = dict(case["predicted_runtime_o"])
        mesh_path = validate_binding(source_mesh, label=f"packaged Pose+Mask Mesh {key}")
        case_dir = mesh_path.parent
        object_report_path = case_dir / "对象报告.json"
        object_report = json.loads(object_report_path.read_text(encoding="utf-8"))
        existing = object_report.get("turntable")
        if args.resume and isinstance(existing, dict) and reusable_turntable(
            existing,
            source_mesh=source_mesh,
            render_config=render_config,
        ):
            turntable = existing
            print(f"[pose_mask_turntable] {position}/{len(cases)} {key} reused", flush=True)
        else:
            video_path = case_dir / "PoseMask_NoPoint_normal环绕.mp4"
            sheet_path = case_dir / "PoseMask_NoPoint_normal环绕抽帧.png"
            if video_path.exists() or sheet_path.exists() or existing is not None:
                raise RuntimeError(f"partial or stale turntable output exists: {case_dir}")
            mesh = load_mesh(mesh_path)
            frames, display_transform = render_method(
                mesh,
                device=device,
                frames=int(args.render_frames),
                resolution=int(args.render_resolution),
                display_margin=float(args.display_margin),
            )
            save_video_atomic(frames, video_path, int(args.fps))
            contact_sheet(
                frames,
                "Pose+Mask No-Point No-VGGT",
                sheet_path,
                count=6,
            )
            turntable = {
                "format": FORMAT,
                "created_at_utc": utc_now(),
                "object_key": key,
                "source_mesh": source_mesh,
                "video": binding(video_path),
                "contact_sheet": binding(sheet_path),
                "render_config": render_config,
                "display_transform": display_transform,
                "display_only": True,
                "world_scale_or_pose_changed": False,
                "passed": True,
            }
            object_report["turntable"] = turntable
            write_json(object_report_path, object_report)
            del mesh, frames
            if device.type == "cuda":
                torch.cuda.empty_cache()
            print(f"[pose_mask_turntable] {position}/{len(cases)} {key}", flush=True)
        updated_case = dict(case)
        updated_case["turntable"] = turntable
        completed.append(updated_case)

    bundle["cases"] = completed
    bundle["turntable_review"] = {
        "format": FORMAT,
        "created_at_utc": utc_now(),
        "object_count": len(completed),
        "render_config": render_config,
        "independent_single_method_videos": True,
        "comparison_or_metric_claim": False,
        "passed": len(completed) == len(cases),
    }
    write_json(report_path, bundle)
    print(
        json.dumps(
            {
                "passed": True,
                "objects": len(completed),
                "bundle_report": str(report_path),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
