#!/usr/bin/env python3
"""Render current/Pixal Mesh videos beside a captured ReconViaGen output video.

The legacy ReconViaGen MP4 is copied verbatim as requested.  Current and Pixal3D
meshes are rendered independently as textureless normal turntables.  The three
videos are also temporally resampled into a display-only side-by-side MP4.  An
optional fixed video-level crop makes their foreground occupancy comparable;
this display autoframe is not metric scale or camera alignment.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
from typing import Any, Sequence

import cv2
import imageio.v2 as imageio
import numpy as np
from PIL import Image
import torch

from pose_point_depth_mv.bunny_review.common import binding, sha256_file
from pose_point_depth_mv.bunny_review.finalize import (
    comparison_contact_sheet,
    comparison_frames,
    load_mesh,
    render_method,
    save_video_atomic,
)
from pose_point_depth_mv.dataset_tools.prepare_coarsemodel_real_raw_cache import (
    ADAPTER_FORMAT,
)
from pose_point_depth_mv.dataset_tools.prepare_omni_real_video_cache import (
    RAW_CACHE_FORMAT,
    utc_now,
    write_json,
)
from pose_point_depth_mv.infer_omni_real_native_no_vggt_mixed import (
    MANIFEST_FORMAT as CURRENT_MANIFEST_FORMAT,
)
from pose_point_depth_mv.infer_omni_real_pixal3d import (
    MANIFEST_FORMAT as PIXAL_MANIFEST_FORMAT,
)


FORMAT = "pose_point_depth_mv.coarsemodel_current_recon_pixal_video_review.v2"
DEFAULT_CURRENT_DISPLAY_NAME = "Current No-VGGT"


def object_key(row: dict[str, Any]) -> str:
    return str(row.get("object_key") or f"{row['category']}:{row['object_id']}")


def load_records(paths: Sequence[str], expected_format: str, label: str) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for value in paths:
        path = Path(value).expanduser().resolve()
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("format") != expected_format or manifest.get("passed") is not True:
            raise RuntimeError(f"invalid {label} manifest: {path}")
        for row in manifest.get("objects", []):
            key = object_key(row)
            if key in records:
                raise RuntimeError(f"duplicate {label} object={key}")
            record = dict(row)
            record["_manifest"] = str(path)
            records[key] = record
    if not records:
        raise RuntimeError(f"no {label} records")
    return records


def uint8_rgb(frame: Any) -> np.ndarray:
    array = np.asarray(frame)
    if array.dtype != np.uint8:
        maximum = float(np.nanmax(array)) if array.size else 0.0
        array = np.clip(array * (255.0 if maximum <= 1.5 else 1.0), 0, 255).astype(np.uint8)
    if array.ndim != 3 or array.shape[2] < 3:
        raise ValueError(f"invalid video frame shape={array.shape}")
    return array[:, :, :3]


def fit_square(frame: Any, resolution: int) -> np.ndarray:
    image = Image.fromarray(uint8_rgb(frame))
    scale = min(float(resolution) / image.width, float(resolution) / image.height)
    size = (
        max(1, int(round(image.width * scale))),
        max(1, int(round(image.height * scale))),
    )
    resized = image.resize(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (resolution, resolution), (0, 0, 0))
    canvas.paste(resized, ((resolution - size[0]) // 2, (resolution - size[1]) // 2))
    return np.asarray(canvas)


def resample_video(path: Path, *, count: int, resolution: int) -> list[np.ndarray]:
    reader = imageio.get_reader(path)
    try:
        frames = [uint8_rgb(frame) for frame in reader]
    finally:
        reader.close()
    if not frames:
        raise RuntimeError(f"video contains no frames: {path}")
    indices = np.rint(np.linspace(0, len(frames) - 1, int(count))).astype(np.int64)
    return [fit_square(frames[int(index)], int(resolution)) for index in indices]


def largest_chroma_component_bbox(
    frame: Any,
    *,
    chroma_threshold: int,
    roi_x_fraction: tuple[float, float] = (0.0, 1.0),
) -> tuple[int, int, int, int] | None:
    """Return the largest colorful component as an exclusive pixel bbox."""
    array = uint8_rgb(frame)
    height, width = array.shape[:2]
    roi_start = int(math.floor(float(roi_x_fraction[0]) * width))
    roi_stop = int(math.ceil(float(roi_x_fraction[1]) * width))
    roi_start = min(max(roi_start, 0), width)
    roi_stop = min(max(roi_stop, roi_start), width)
    colors = array.astype(np.int16)
    chroma = colors.max(axis=2) - colors.min(axis=2)
    brightness = colors.max(axis=2)
    mask = (chroma >= int(chroma_threshold)) & (brightness >= 24)
    mask[:, :roi_start] = False
    mask[:, roi_stop:] = False
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8),
        connectivity=8,
    )
    if count <= 1:
        return None
    minimum_area = max(16, int(round(height * width * 0.0001)))
    candidates = [
        index
        for index in range(1, count)
        if int(stats[index, cv2.CC_STAT_AREA]) >= minimum_area
    ]
    if not candidates:
        return None
    component = max(candidates, key=lambda index: int(stats[index, cv2.CC_STAT_AREA]))
    x = int(stats[component, cv2.CC_STAT_LEFT])
    y = int(stats[component, cv2.CC_STAT_TOP])
    component_width = int(stats[component, cv2.CC_STAT_WIDTH])
    component_height = int(stats[component, cv2.CC_STAT_HEIGHT])
    return x, y, x + component_width, y + component_height


def _crop_background_color(
    frame: np.ndarray,
    *,
    left: int,
    top: int,
    side: int,
) -> tuple[int, int, int]:
    height, width = frame.shape[:2]
    x0, x1 = max(0, left), min(width, left + side)
    y0, y1 = max(0, top), min(height, top + side)
    if x1 <= x0 or y1 <= y0:
        return 0, 0, 0
    region = frame[y0:y1, x0:x1]
    band = max(1, min(region.shape[:2]) // 32)
    border = np.concatenate(
        [
            region[:band].reshape(-1, 3),
            region[-band:].reshape(-1, 3),
            region[:, :band].reshape(-1, 3),
            region[:, -band:].reshape(-1, 3),
        ],
        axis=0,
    )
    color = np.rint(np.median(border, axis=0)).astype(np.uint8)
    return tuple(int(value) for value in color)


def _fixed_square_crop(
    frame: Any,
    *,
    left: int,
    top: int,
    side: int,
    resolution: int,
) -> np.ndarray:
    array = uint8_rgb(frame)
    height, width = array.shape[:2]
    background = _crop_background_color(array, left=left, top=top, side=side)
    canvas = Image.new("RGB", (side, side), background)
    source_left, source_top = max(0, left), max(0, top)
    source_right, source_bottom = min(width, left + side), min(height, top + side)
    if source_right > source_left and source_bottom > source_top:
        patch = Image.fromarray(array).crop(
            (source_left, source_top, source_right, source_bottom)
        )
        canvas.paste(patch, (source_left - left, source_top - top))
    if side != int(resolution):
        canvas = canvas.resize((int(resolution), int(resolution)), Image.Resampling.LANCZOS)
    return np.asarray(canvas)


def autoframe_video(
    frames: Sequence[np.ndarray],
    *,
    target_fill: float,
    chroma_threshold: int,
    roi_x_fraction: tuple[float, float] = (0.0, 1.0),
) -> tuple[list[np.ndarray], dict[str, Any]]:
    """Apply one fixed crop to every frame so foreground scale does not jitter."""
    if not frames:
        raise ValueError("autoframe requires at least one frame")
    if not 0.1 <= float(target_fill) <= 0.95:
        raise ValueError("target_fill must be in [0.1, 0.95]")
    if not 1 <= int(chroma_threshold) <= 255:
        raise ValueError("chroma_threshold must be in [1, 255]")
    if not 0.0 <= roi_x_fraction[0] < roi_x_fraction[1] <= 1.0:
        raise ValueError("roi_x_fraction must be a valid interval in [0, 1]")
    arrays = [uint8_rgb(frame) for frame in frames]
    shape = arrays[0].shape
    if any(frame.shape != shape for frame in arrays):
        raise ValueError("autoframe frames must have identical shapes")
    bboxes = [
        largest_chroma_component_bbox(
            frame,
            chroma_threshold=int(chroma_threshold),
            roi_x_fraction=roi_x_fraction,
        )
        for frame in arrays
    ]
    valid = [bbox for bbox in bboxes if bbox is not None]
    if not valid:
        raise RuntimeError(
            "display autoframe found no colorful foreground component; "
            "lower --comparison_foreground_chroma or disable autoframe"
        )
    x0 = min(bbox[0] for bbox in valid)
    y0 = min(bbox[1] for bbox in valid)
    x1 = max(bbox[2] for bbox in valid)
    y1 = max(bbox[3] for bbox in valid)
    foreground_side = max(x1 - x0, y1 - y0)
    crop_side = max(2, int(math.ceil(foreground_side / float(target_fill))))
    center_x = 0.5 * (x0 + x1)
    center_y = 0.5 * (y0 + y1)
    left = int(math.floor(center_x - 0.5 * crop_side))
    top = int(math.floor(center_y - 0.5 * crop_side))
    resolution = int(shape[0])
    adjusted = [
        _fixed_square_crop(
            frame,
            left=left,
            top=top,
            side=crop_side,
            resolution=resolution,
        )
        for frame in arrays
    ]
    return adjusted, {
        "policy": "fixed_video_level_chroma_foreground_square_crop_and_isotropic_resize",
        "source_frame_shape": [int(value) for value in shape],
        "foreground_bbox_xyxy_exclusive": [x0, y0, x1, y1],
        "crop_square_left_top_side": [left, top, crop_side],
        "target_foreground_fill": float(target_fill),
        "foreground_chroma_threshold": int(chroma_threshold),
        "roi_x_fraction": [float(value) for value in roi_x_fraction],
        "detected_frame_count": len(valid),
        "total_frame_count": len(arrays),
        "fixed_crop_across_frames": True,
        "isotropic_resize": True,
    }


def copy_exact(source: Path, destination: Path) -> dict[str, Any]:
    source = source.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if sha256_file(destination) != sha256_file(source):
            raise RuntimeError(f"existing copied ReconViaGen video differs: {destination}")
    else:
        shutil.copy2(source, destination)
    return binding(destination)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current_manifest", action="append", required=True)
    parser.add_argument("--pixal_manifest", action="append", required=True)
    parser.add_argument("--raw_cache_report", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--render_frames", type=int, default=48)
    parser.add_argument("--render_resolution", type=int, default=512)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--display_margin", type=float, default=1.6)
    parser.add_argument("--comparison_autoframe", action="store_true")
    parser.add_argument("--comparison_target_fill", type=float, default=0.72)
    parser.add_argument("--comparison_foreground_chroma", type=int, default=18)
    parser.add_argument(
        "--current_display_name",
        default=DEFAULT_CURRENT_DISPLAY_NAME,
        help="Label for the first method in display-only comparison videos",
    )
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> None:
    args = make_parser().parse_args()
    if min(args.render_frames, args.render_resolution, args.fps) <= 0:
        raise ValueError("render_frames, render_resolution and fps must be positive")
    render_config = {
        "render_frames": int(args.render_frames),
        "render_resolution": int(args.render_resolution),
        "fps": int(args.fps),
        "display_margin": float(args.display_margin),
        "comparison_autoframe": bool(args.comparison_autoframe),
        "comparison_target_fill": float(args.comparison_target_fill),
        "comparison_foreground_chroma": int(args.comparison_foreground_chroma),
    }
    current_display_name = str(args.current_display_name).strip()
    if not current_display_name:
        raise ValueError("current_display_name must not be empty")
    # Preserve old report identity when the default label is used.
    if current_display_name != DEFAULT_CURRENT_DISPLAY_NAME:
        render_config["current_display_name"] = current_display_name
    raw_path = Path(args.raw_cache_report).expanduser().resolve()
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    if (
        raw.get("format") != RAW_CACHE_FORMAT
        or raw.get("adapter_format") != ADAPTER_FORMAT
        or raw.get("passed") is not True
    ):
        raise RuntimeError(f"invalid CoarseModel raw report: {raw_path}")
    raw_by_key = {object_key(row): row for row in raw.get("objects", [])}
    current = load_records(args.current_manifest, CURRENT_MANIFEST_FORMAT, "current")
    pixal = load_records(args.pixal_manifest, PIXAL_MANIFEST_FORMAT, "Pixal3D")
    if set(current) != set(pixal):
        raise RuntimeError(
            f"current/Pixal object mismatch: {sorted(current)} != {sorted(pixal)}"
        )
    missing_raw = sorted(set(current) - set(raw_by_key))
    if missing_raw:
        raise RuntimeError(f"objects missing from raw report: {missing_raw}")
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    cases = []
    for position, key in enumerate(sorted(current), start=1):
        raw_row = raw_by_key[key]
        legacy_value = raw_row.get("legacy_reconviagen_video")
        if not legacy_value or not Path(legacy_value).is_file():
            raise FileNotFoundError(f"missing captured ReconViaGen MP4 for {key}")
        case_dir = output_dir / str(raw_row["object_id"])
        report_path = case_dir / "report.json"
        expected_sources = {
            "current_mesh": binding(Path(current[key]["mesh"])),
            "pixal_mesh": binding(Path(pixal[key]["mesh"])),
            "reconviagen_original_video": binding(Path(legacy_value)),
        }
        if args.resume and report_path.is_file():
            report = json.loads(report_path.read_text(encoding="utf-8"))
            if report.get("format") != FORMAT or report.get("passed") is not True:
                raise RuntimeError(f"invalid reusable video report: {report_path}")
            if report.get("source_assets") != expected_sources:
                raise RuntimeError(f"source assets changed for reusable object={key}")
            if report.get("render_config") != render_config:
                raise RuntimeError(f"render config changed for reusable object={key}")
            for value in report["videos"].values():
                if not Path(value["path"]).is_file() or sha256_file(value["path"]) != value["sha256"]:
                    raise RuntimeError(f"reusable video changed: {value}")
            cases.append(report)
            print(f"[coarsemodel_threeway_video] {position}/{len(current)} {key} reused", flush=True)
            continue
        if case_dir.exists():
            raise RuntimeError(f"partial video output exists: {case_dir}")
        case_dir.mkdir(parents=True)
        current_mesh = load_mesh(Path(current[key]["mesh"]))
        pixal_mesh = load_mesh(Path(pixal[key]["mesh"]))
        current_frames, current_display = render_method(
            current_mesh,
            device=device,
            frames=int(args.render_frames),
            resolution=int(args.render_resolution),
            display_margin=float(args.display_margin),
        )
        pixal_frames, pixal_display = render_method(
            pixal_mesh,
            device=device,
            frames=int(args.render_frames),
            resolution=int(args.render_resolution),
            display_margin=float(args.display_margin),
        )
        legacy_source = Path(legacy_value)
        legacy_frames = resample_video(
            legacy_source,
            count=int(args.render_frames),
            resolution=int(args.render_resolution),
        )
        display_frames = {
            "current": current_frames,
            "reconviagen": legacy_frames,
            "pixal3d": pixal_frames,
        }
        autoframe_reports: dict[str, Any] = {}
        if args.comparison_autoframe:
            display_frames["current"], autoframe_reports["current"] = autoframe_video(
                current_frames,
                target_fill=float(args.comparison_target_fill),
                chroma_threshold=int(args.comparison_foreground_chroma),
            )
            display_frames["reconviagen"], autoframe_reports["reconviagen"] = autoframe_video(
                legacy_frames,
                target_fill=float(args.comparison_target_fill),
                chroma_threshold=int(args.comparison_foreground_chroma),
                roi_x_fraction=(0.5, 1.0),
            )
            display_frames["pixal3d"], autoframe_reports["pixal3d"] = autoframe_video(
                pixal_frames,
                target_fill=float(args.comparison_target_fill),
                chroma_threshold=int(args.comparison_foreground_chroma),
            )
        if args.comparison_autoframe:
            comparison_labels = [
                f"{current_display_name} ({raw_row.get('input_view_count', raw_row['registered_pair_count'])} views; display-normalized)",
                "ReconViaGen normal mesh (display-normalized)",
                "Pixal3D official final (1 view; display-normalized)",
            ]
            combined_name = "04_三路并排_显示自动取景仅形状比较.mp4"
            sheet_name = "05_三路并排_显示自动取景抽帧.png"
            reconviagen_display = (
                "02 retains the original captured MP4 byte-for-byte; the combined review "
                "temporally resamples it and auto-frames the right-half normal-mesh panel"
            )
        else:
            comparison_labels = [
                f"{current_display_name} ({raw_row.get('input_view_count', raw_row['registered_pair_count'])} views)",
                "ReconViaGen captured output (original camera)",
                "Pixal3D official final (1 view)",
            ]
            combined_name = "04_三路并排_非同步相机仅人工比较.mp4"
            sheet_name = "05_三路并排抽帧.png"
            reconviagen_display = (
                "original captured output video, resized and temporally resampled only "
                "in the combined MP4"
            )
        current_path = case_dir / "01_当前NoVGGT_全视图_normal.mp4"
        recon_path = case_dir / "02_ReconViaGen原始输出.mp4"
        pixal_path = case_dir / "03_Pixal3D单视图_normal.mp4"
        combined_path = case_dir / combined_name
        sheet_path = case_dir / sheet_name
        save_video_atomic(current_frames, current_path, int(args.fps))
        recon_binding = copy_exact(legacy_source, recon_path)
        save_video_atomic(pixal_frames, pixal_path, int(args.fps))
        combined = comparison_frames(
            [
                (comparison_labels[0], display_frames["current"]),
                (comparison_labels[1], display_frames["reconviagen"]),
                (comparison_labels[2], display_frames["pixal3d"]),
            ]
        )
        save_video_atomic(combined, combined_path, int(args.fps))
        comparison_contact_sheet(combined, sheet_path, count=6)
        # Single-method sheets are intentionally omitted; the requested deliverable is video.
        report = {
            "format": FORMAT,
            "created_at_utc": utc_now(),
            "object_key": key,
            "current_input_view_count": int(
                raw_row.get("input_view_count", raw_row["registered_pair_count"])
            ),
            "render_config": render_config,
            "videos": {
                "current_no_vggt": binding(current_path),
                "reconviagen_original": recon_binding,
                "pixal3d_official": binding(pixal_path),
                "threeway_display": binding(combined_path),
                "threeway_contact_sheet": binding(sheet_path),
            },
            "source_assets": {
                **expected_sources,
            },
            "display": {
                "current": current_display,
                "pixal3d": pixal_display,
                "reconviagen": reconviagen_display,
                "comparison_autoframe": autoframe_reports,
            },
            "comparison_scope": (
                "qualitative shape review only: fixed per-video display crops make foreground "
                "occupancy comparable; they do not align world scale, pose, or camera paths"
            ),
            "mesh_files_copied_to_review_directory": False,
            "passed": True,
        }
        write_json(report_path, report)
        cases.append(report)
        print(f"[coarsemodel_threeway_video] {position}/{len(current)} {key}", flush=True)
        del current_mesh, pixal_mesh, current_frames, pixal_frames, legacy_frames, combined
        if device.type == "cuda":
            torch.cuda.empty_cache()
    final = {
        "format": FORMAT,
        "created_at_utc": utc_now(),
        "raw_cache_report": str(raw_path),
        "raw_cache_report_sha256": sha256_file(raw_path),
        "object_count": len(cases),
        "cases": cases,
        "scope_guard": "two-capture qualitative three-way video review; no GT and no metric claim",
        "passed": len(cases) == len(current) and bool(cases),
    }
    write_json(output_dir / "report.json", final)
    print(json.dumps({
        "passed": final["passed"],
        "objects": len(cases),
        "output_dir": str(output_dir),
    }, indent=2, ensure_ascii=False))
    if not final["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
