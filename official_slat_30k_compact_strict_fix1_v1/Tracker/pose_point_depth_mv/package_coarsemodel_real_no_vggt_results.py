#!/usr/bin/env python3
"""Package CoarseModel-capture No-VGGT inference in runtime-O and sparse-world frames."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
import trimesh

from pose_point_depth_mv.dataset_tools.prepare_omni_real_runtime_inputs import (
    MANIFEST_FORMAT as RUNTIME_MANIFEST_FORMAT,
)
from pose_point_depth_mv.dataset_tools.prepare_omni_real_video_cache import (
    RAW_CACHE_FORMAT,
    sha256_file,
    utc_now,
    write_json,
)
from pose_point_depth_mv.infer_omni_real_native_no_vggt_mixed import (
    MANIFEST_FORMAT as MIXED_INFERENCE_MANIFEST_FORMAT,
)
from pose_point_depth_mv.infer_omni_real_native_no_vggt_synthetic import (
    MANIFEST_FORMAT as SYNTHETIC_INFERENCE_MANIFEST_FORMAT,
)
from pose_point_depth_mv.infer_omni_real_native_ss_stock_slat_no_vggt import (
    MANIFEST_FORMAT as STOCK_SLAT_INFERENCE_MANIFEST_FORMAT,
)


FORMAT = "pose_point_depth_mv.coarsemodel_real_no_vggt_result_bundle.v1"
INFERENCE_MANIFEST_FORMATS = {
    MIXED_INFERENCE_MANIFEST_FORMAT,
    SYNTHETIC_INFERENCE_MANIFEST_FORMAT,
    STOCK_SLAT_INFERENCE_MANIFEST_FORMAT,
}


def object_key(row: dict[str, Any]) -> str:
    return str(row.get("object_key") or f"{row['category']}:{row['object_id']}")


def safe_name(value: str) -> str:
    return "".join(character if character.isalnum() or character in "-_" else "_" for character in value)


def copy_bound(source: Path, destination: Path) -> dict[str, str]:
    source = source.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if sha256_file(destination) != sha256_file(source):
            raise RuntimeError(f"existing copied asset differs: {destination}")
    else:
        shutil.copy2(source, destination)
    return {"path": str(destination.resolve()), "sha256": sha256_file(destination)}


def export_world_mesh(source: Path, destination: Path, transform: np.ndarray) -> dict[str, str]:
    if destination.exists():
        return {"path": str(destination.resolve()), "sha256": sha256_file(destination)}
    loaded = trimesh.load(source, force="scene", process=False)
    scene = loaded if isinstance(loaded, trimesh.Scene) else trimesh.Scene(loaded)
    for geometry in scene.geometry.values():
        geometry.apply_transform(transform)
    destination.parent.mkdir(parents=True, exist_ok=True)
    exported = scene.export(file_type="obj")
    if isinstance(exported, dict):
        # No-VGGT currently emits one OBJ geometry. Refuse ambiguous multi-file exports.
        obj_payloads = [value for name, value in exported.items() if str(name).endswith(".obj")]
        if len(obj_payloads) != 1:
            raise RuntimeError(f"ambiguous world OBJ export for {source}: {list(exported)}")
        exported = obj_payloads[0]
    if isinstance(exported, str):
        destination.write_text(exported, encoding="utf-8")
    else:
        destination.write_bytes(bytes(exported))
    return {"path": str(destination.resolve()), "sha256": sha256_file(destination)}


def input_sheet(images: list[Path], masks: list[Path], destination: Path) -> dict[str, str]:
    if destination.exists():
        return {"path": str(destination.resolve()), "sha256": sha256_file(destination)}
    panels = []
    for index, (image_path, mask_path) in enumerate(zip(images, masks)):
        with Image.open(image_path) as handle:
            image = handle.convert("RGB").resize((320, 320), Image.Resampling.LANCZOS)
        with Image.open(mask_path) as handle:
            mask = handle.convert("L").resize((320, 320), Image.Resampling.NEAREST)
        background = Image.new("RGB", image.size, (0, 0, 0))
        foreground = Image.composite(image, background, mask)
        canvas = Image.new("RGB", (320, 346), (24, 24, 24))
        canvas.paste(foreground, (0, 26))
        ImageDraw.Draw(canvas).text((8, 6), f"view {index}", fill=(255, 255, 255))
        panels.append(canvas)
    columns = 4
    rows = (len(panels) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * 320, rows * 346), (12, 12, 12))
    for index, panel in enumerate(panels):
        sheet.paste(panel, ((index % columns) * 320, (index // columns) * 346))
    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination)
    return {"path": str(destination.resolve()), "sha256": sha256_file(destination)}


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inference_manifest", required=True)
    parser.add_argument("--runtime_input_manifest", required=True)
    parser.add_argument("--raw_cache_report", required=True)
    parser.add_argument("--output_dir", required=True)
    return parser


def main() -> None:
    args = make_parser().parse_args()
    inference_path = Path(args.inference_manifest).expanduser().resolve()
    runtime_path = Path(args.runtime_input_manifest).expanduser().resolve()
    raw_path = Path(args.raw_cache_report).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    inference = json.loads(inference_path.read_text(encoding="utf-8"))
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    if (
        inference.get("format") not in INFERENCE_MANIFEST_FORMATS
        or inference.get("passed") is not True
    ):
        raise RuntimeError(f"No-VGGT inference manifest did not pass: {inference_path}")
    if runtime.get("format") != RUNTIME_MANIFEST_FORMAT or runtime.get("passed") is not True:
        raise RuntimeError(f"runtime input manifest did not pass: {runtime_path}")
    if raw.get("format") != RAW_CACHE_FORMAT or raw.get("passed") is not True:
        raise RuntimeError(f"raw cache report did not pass: {raw_path}")
    runtime_by_key = {object_key(row): row for row in runtime["objects"]}
    raw_by_key = {object_key(row): row for row in raw["objects"]}
    records = list(inference.get("objects", []))
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = []
    for record in records:
        key = object_key(record)
        runtime_row = runtime_by_key[key]
        raw_row = raw_by_key[key]
        case_dir = output_dir / safe_name(key)
        with np.load(runtime_row["cache_npz"], allow_pickle=False) as cache:
            transform = np.asarray(cache["T_O2W"], dtype=np.float64)
        if transform.shape != (4, 4) or not np.isfinite(transform).all():
            raise RuntimeError(f"invalid T_O2W for {key}")
        predicted_o = copy_bound(Path(record["mesh"]), case_dir / "预测Mesh_runtime_O.obj")
        predicted_w = export_world_mesh(
            Path(record["mesh"]), case_dir / "预测Mesh_sparse_world.obj", transform
        )
        sheet = input_sheet(
            [Path(path) for path in runtime_row["prepared_rgb_paths"]],
            [Path(path) for path in runtime_row["prepared_mask_paths"]],
            case_dir / "输入8视图前景.png",
        )
        references = {}
        for label, field in (
            ("coarsemodel_reference", "reference_mesh"),
            ("legacy_reconviagen", "legacy_reconviagen_mesh"),
        ):
            value = raw_row.get(field)
            if value and Path(value).is_file():
                source = Path(value)
                references[label] = copy_bound(
                    source, case_dir / "仅供人工参考_不作为GT" / f"{label}{source.suffix.lower()}"
                )
        report = {
            "format": FORMAT,
            "created_at_utc": utc_now(),
            "object_key": key,
            "input_sheet": sheet,
            "predicted_runtime_o": predicted_o,
            "predicted_sparse_world": predicted_w,
            "T_O2W": transform.tolist(),
            "references": references,
            "reference_policy": "visual_only_not_metric_gt",
            "inference_method": inference.get("method"),
            "slat_backend": inference.get("slat_backend", "native_v2"),
            "passed": True,
        }
        write_json(case_dir / "对象报告.json", report)
        cases.append(report)
        print(f"[coarsemodel_package] {len(cases)}/{len(records)} {key}", flush=True)
    final = {
        "format": FORMAT,
        "created_at_utc": utc_now(),
        "inference_manifest": str(inference_path),
        "inference_manifest_sha256": sha256_file(inference_path),
        "runtime_input_manifest": str(runtime_path),
        "runtime_input_manifest_sha256": sha256_file(runtime_path),
        "raw_cache_report": str(raw_path),
        "raw_cache_report_sha256": sha256_file(raw_path),
        "object_count": len(cases),
        "inference_method": inference.get("method"),
        "slat_backend": inference.get("slat_backend", "native_v2"),
        "cases": cases,
        "scope_guard": (
            "Qualitative real-capture deployment bundle. CoarseModel and legacy "
            "ReconViaGen meshes are visual references, not registered GT or metrics."
        ),
        "passed": len(cases) == len(records) and bool(cases),
    }
    write_json(output_dir / "report.json", final)
    print(json.dumps({
        "passed": final["passed"],
        "objects": len(cases),
        "output_dir": str(output_dir),
        "report": str(output_dir / "report.json"),
    }, indent=2, ensure_ascii=False))
    if not final["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
