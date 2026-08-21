#!/usr/bin/env python3
"""Project one runtime-O Mesh onto the exact original selected camera frames."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil

import numpy as np
from PIL import Image, ImageDraw

from manual_mesh_reconstruction.common import (
    atomic_json,
    load_json,
    sha256_file,
)
from manual_mesh_reconstruction.projection import (
    boundary,
    load_projection_contract,
    make_headless_raster_context,
    rasterize_world_silhouette,
    world_mesh_buffers,
)
from manual_mesh_reconstruction.canonicalization import array_sha256
from manual_mesh_reconstruction.canonicalization import validate_proper_similarity
from manual_mesh_reconstruction.mesh_coordinates import (
    validate_runtime_o_mesh_frame_contract,
)


FORMAT = "pose_point_depth_mv.runtime_o_mesh_original_camera_contours.v1"
CONTOUR_RGB = np.asarray([0, 255, 255], dtype=np.uint8)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    shutil.copy2(source, temporary)
    os.replace(temporary, destination)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime_input_manifest", required=True)
    parser.add_argument("--mesh_o", required=True)
    parser.add_argument("--mesh_frame_report", required=True)
    parser.add_argument(
        "--T_O2W_npz",
        default="",
        help=(
            "optional selected_T_O2W.npz produced by optimize_o2w; the original "
            "runtime cache remains immutable"
        ),
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--object", default="")
    parser.add_argument("--contour_width", type=int, default=3)
    parser.add_argument("--method_label", default="no-VGGT SS30K+SLat30K")
    parser.add_argument("--overview_name", default="SS30K_SLat30K_原始8帧相机位姿轮廓总览.png")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="regenerate an interrupted contour directory or reuse its passed report",
    )
    return parser


def main() -> None:
    args = make_parser().parse_args()
    runtime_path = Path(args.runtime_input_manifest).expanduser().resolve(strict=True)
    runtime = load_json(runtime_path)
    if runtime.get("passed") is not True:
        raise RuntimeError(f"runtime input manifest did not pass: {runtime_path}")
    rows = list(runtime.get("objects", []))
    if args.object:
        rows = [row for row in rows if str(row.get("object_key")) == str(args.object)]
    if len(rows) != 1:
        raise RuntimeError(f"contour export requires exactly one runtime object, got={len(rows)}")
    row = rows[0]
    runtime_report = Path(row["cache_npz"]).expanduser().resolve(strict=True).parent / "report.json"
    if not runtime_report.is_file():
        raise FileNotFoundError(runtime_report)
    contract = load_projection_contract(runtime_report, row["object_key"])
    initial_T_O2W = np.asarray(contract["T_O2W"], dtype=np.float64)
    transform_source = "original_runtime_cache_T_O2W"
    transform_binding = None
    if args.T_O2W_npz:
        transform_path = Path(args.T_O2W_npz).expanduser().resolve(strict=True)
        with np.load(transform_path, allow_pickle=False) as payload:
            selected_T_O2W = validate_proper_similarity(
                np.asarray(payload["T_O2W"], dtype=np.float64),
                name="selected_T_O2W",
            )
        contract["T_O2W"] = selected_T_O2W
        transform_source = "input_mask_refinement_selected_T_O2W_npz"
        transform_binding = {
            "path": str(transform_path),
            "sha256": sha256_file(transform_path),
            "T_O2W_sha256": array_sha256(selected_T_O2W),
            "initial_T_O2W_sha256": array_sha256(initial_T_O2W),
        }
        contract["chain_audit"] = {
            "formula": "T_O2C_selected = T_W2C_raw @ T_O2W_selected",
            "passed": True,
            "stored_runtime_T_O2C_compared": False,
            "reason": (
                "the stored T_O2C is bound to the immutable initial runtime O; "
                "the selected post-Mesh transform is composed directly at render time"
            ),
        }
    expected_frames = [str(value) for value in row["selected_frame_names"]]
    if contract["frame_names"] != expected_frames:
        raise RuntimeError("projection cameras do not match frozen selected frame names")
    mesh_path = Path(args.mesh_o).expanduser().resolve(strict=True)
    mesh_frame_report_path = (
        Path(args.mesh_frame_report).expanduser().resolve(strict=True)
    )
    mesh_frame_report = load_json(mesh_frame_report_path)
    if mesh_frame_report.get("passed") is not True:
        raise RuntimeError("runtime-O Mesh frame report did not pass")
    if Path(mesh_frame_report.get("mesh", "")).expanduser().resolve(strict=True) != mesh_path:
        raise RuntimeError("runtime-O Mesh frame report path differs")
    if mesh_frame_report.get("mesh_sha256") != sha256_file(mesh_path):
        raise RuntimeError("runtime-O Mesh frame report hash differs")
    validate_runtime_o_mesh_frame_contract(mesh_frame_report)
    output = Path(args.output_dir).expanduser().resolve()
    selected_transform_sha256 = array_sha256(
        np.asarray(contract["T_O2W"], dtype=np.float64)
    )
    if output.exists():
        report_path = output / "report.json"
        if report_path.is_file():
            existing = load_json(report_path)
            if (
                args.resume
                and existing.get("passed") is True
                and existing.get("mesh_o_sha256") == sha256_file(mesh_path)
                and existing.get("runtime_input_manifest_sha256")
                == sha256_file(runtime_path)
                and existing.get("selected_T_O2W_sha256")
                == selected_transform_sha256
            ):
                print(
                    json.dumps(
                        {
                            "passed": True,
                            "reused": True,
                            "views": len(existing.get("views", [])),
                            "overview": existing.get("overview"),
                            "report": str(report_path),
                        },
                        indent=2,
                    )
                )
                return
        if not args.resume:
            raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    vertices_w, faces = world_mesh_buffers(mesh_path, contract["T_O2W"])
    rastctx = make_headless_raster_context()
    records = []
    overlay_images: list[Image.Image] = []
    for index, (frame_name, image_path, T_W2C, K, camera) in enumerate(
        zip(
            contract["frame_names"],
            contract["images"],
            contract["T_W2C"],
            contract["intrinsics"],
            contract["cameras"],
        )
    ):
        original = Image.open(image_path).convert("RGB")
        width, height = original.size
        projected = rasterize_world_silhouette(
            vertices_w, faces, T_W2C, K, camera, (height, width), rastctx
        )
        if int(projected.sum()) <= 0:
            raise RuntimeError(f"empty projected silhouette for frame={frame_name}")
        contour = boundary(projected, width=int(args.contour_width))
        array = np.asarray(original, dtype=np.uint8).copy()
        array[contour] = CONTOUR_RGB
        overlay = Image.fromarray(array, mode="RGB")
        original_copy = output / "原始选帧" / f"view_{index:02d}_{frame_name}"
        overlay_path = output / "相机位姿Mesh轮廓叠加" / f"view_{index:02d}_{Path(frame_name).stem}_contour.png"
        _atomic_copy(Path(image_path), original_copy)
        overlay_path.parent.mkdir(parents=True, exist_ok=True)
        overlay.save(overlay_path)
        thumb = overlay.copy()
        thumb.thumbnail((480, 270), Image.Resampling.LANCZOS)
        overlay_images.append(thumb)
        records.append(
            {
                "view_index": index,
                "frame_name": frame_name,
                "source_image": str(image_path),
                "source_image_sha256": sha256_file(image_path),
                "copied_original": str(original_copy),
                "copied_original_sha256": sha256_file(original_copy),
                "overlay": str(overlay_path),
                "overlay_sha256": sha256_file(overlay_path),
                "silhouette_pixels": int(projected.sum()),
                "contour_pixels": int(contour.sum()),
                "camera_model": camera["model"],
                "camera_distortion": camera["distortion"],
                "T_W2C_sha256": array_sha256(np.asarray(T_W2C, dtype=np.float64)),
                "K_raw_sha256": array_sha256(np.asarray(K, dtype=np.float64)),
            }
        )
    cell_w, cell_h, header = 480, 270, 28
    sheet = Image.new("RGB", (cell_w * 2, (cell_h + header) * 4), (20, 20, 20))
    draw = ImageDraw.Draw(sheet)
    for index, image in enumerate(overlay_images):
        x = (index % 2) * cell_w
        y = (index // 2) * (cell_h + header)
        sheet.paste(image, (x + (cell_w - image.width) // 2, y + header))
        draw.text((x + 8, y + 7), f"view {index}: {contract['frame_names'][index]}", fill=(238, 238, 238))
    overview_name = Path(str(args.overview_name)).name
    if not overview_name.lower().endswith(".png"):
        raise ValueError("overview_name must be a PNG filename")
    sheet_path = output / overview_name
    sheet.save(sheet_path)
    report = {
        "format": FORMAT,
        "created_at_utc": utc_now(),
        "passed": True,
        "object_key": row["object_key"],
        "mesh_o": str(mesh_path),
        "mesh_o_sha256": sha256_file(mesh_path),
        "mesh_frame_report": str(mesh_frame_report_path),
        "mesh_frame_report_sha256": sha256_file(mesh_frame_report_path),
        "mesh_frame_contract_verified": True,
        "mesh_frame_contract": mesh_frame_report["mesh_frame_contract"],
        "decoder_to_runtime_o_axis_transform": mesh_frame_report[
            "decoder_to_runtime_o_axis_transform"
        ],
        "runtime_input_manifest": str(runtime_path),
        "runtime_input_manifest_sha256": sha256_file(runtime_path),
        "runtime_object_report": str(runtime_report),
        "runtime_object_report_sha256": sha256_file(runtime_report),
        "selected_frame_names": expected_frames,
        "projection_formula": "Mesh_O -> T_O2W -> Mesh_W -> T_W2C -> K_raw + raw distortion",
        "used_physical_T_O2W": True,
        "T_O2W_source": transform_source,
        "initial_T_O2W_sha256": array_sha256(initial_T_O2W),
        "selected_T_O2W_sha256": selected_transform_sha256,
        "selected_T_O2W_binding": transform_binding,
        "used_raw_T_W2C": True,
        "used_T_O2C_lifting": False,
        "contour_rgb": CONTOUR_RGB.astype(int).tolist(),
        "contour_width": int(args.contour_width),
        "projection_contract": contract["bindings"],
        "projection_chain_audit": contract["chain_audit"],
        "views": records,
        "overview": str(sheet_path),
        "overview_sha256": sha256_file(sheet_path),
        "scope_guard": (
            f"Cyan is the {args.method_label} Mesh silhouette boundary projected with "
            "the physical runtime-O/world/original-camera chain. It is a qualitative "
            "overlay, not a new held-out metric."
        ),
    }
    atomic_json(output / "report.json", report)
    print(json.dumps({"passed": True, "views": len(records), "overview": str(sheet_path), "report": str(output / 'report.json')}, indent=2))


if __name__ == "__main__":
    main()
