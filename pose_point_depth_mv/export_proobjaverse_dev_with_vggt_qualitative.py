#!/usr/bin/env python3
"""Freeze and export two held-out Dev qualitative comparisons.

The random draw is performed only once by ``prepare`` and is content bound in
``selection.json``.  Eligibility means that the already completed seed-42
strict ReconViaGen and VSS+V-SLat endpoint records both decoded a valid Mesh;
this prevents a qualitative job from silently replacing a sampled decoder
failure with a hand-picked object.

The expensive VSS+V-SLat inference is deliberately delegated to the audited
``official_ss_with_vggt_perf_v1.evaluate_ss_slat`` worker.  This module owns
only input freezing, the strict ReconViaGen Mesh export, and friendly artifact
packaging.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
import torch

from pose_point_depth_mv.evaluate_proobjaverse_official_reconviagen import (
    _selected_images,
)
from pose_point_depth_mv.infer_objaverse16_reconviagen import _build_pipeline
from pose_point_depth_mv.mesh_benchmark_metrics import mesh_structure_metrics
from pose_point_depth_mv.prepare_proobjaverse_official_slat_dino_cache import (
    _load_views_with_audit,
)
from pose_point_depth_mv.proobjaverse_official_slat_protocol import (
    atomic_json,
    canonical_sha256,
    load_json,
    sha256_file,
)


SELECTION_FORMAT = "pose_point_depth_mv.proobjaverse_dev_with_vggt_qualitative_selection.v1"
RECON_EXPORT_FORMAT = "pose_point_depth_mv.proobjaverse_dev_reconviagen_mesh_export.v1"
CASE_FORMAT = "pose_point_depth_mv.proobjaverse_dev_with_vggt_qualitative_case.v1"

STRICT_SAMPLING = {
    "sparse_structure": {
        "steps": 30,
        "cfg_strength": 7.5,
        "cfg_interval": [0.6, 1.0],
        "guidance_rescale": 0.7,
        "rescale_t": 5.0,
    },
    "slat": {
        "steps": 12,
        "cfg_strength": 7.5,
        "cfg_interval": [0.6, 1.0],
        "guidance_rescale": 0.5,
        "rescale_t": 3.0,
    },
    "multiimage_algo": "multidiffusion",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_hashed_json(path: Path, payload: dict[str, Any]) -> None:
    value = copy.deepcopy(payload)
    value.pop("manifest_sha256", None)
    value["manifest_sha256"] = canonical_sha256(value)
    atomic_json(path, value)


def _load_hashed_json(path: Path, *, expected_format: str) -> dict[str, Any]:
    payload = load_json(path)
    body = dict(payload)
    saved = str(body.pop("manifest_sha256", ""))
    if (
        payload.get("format") != expected_format
        or payload.get("passed") is not True
        or not saved
        or canonical_sha256(body) != saved
    ):
        raise RuntimeError(f"hashed qualitative artifact differs: {path}")
    return payload


def _verify_report(path: Path) -> dict[str, Any]:
    payload = load_json(path)
    body = dict(payload)
    saved = str(body.pop("report_sha256", ""))
    if saved and canonical_sha256(body) != saved:
        raise RuntimeError(f"report internal SHA256 differs: {path}")
    return payload


def _endpoint_valid_seed42(paths: list[Path]) -> tuple[set[str], dict[str, Any]]:
    valid: set[str] = set()
    rows: dict[str, Any] = {}
    for path in paths:
        report = _verify_report(path)
        for row in report.get("mesh_branch_records", []):
            if row.get("branch") != "native_trained" or int(row.get("seed", -1)) != 42:
                continue
            uid = str(row.get("object_uid", ""))
            if not uid or uid in rows:
                raise RuntimeError(f"duplicate/empty V endpoint seed42 UID: {uid!r}")
            rows[uid] = {
                "passed": row.get("passed") is True,
                "surface": row.get("surface"),
                "structure": row.get("structure"),
                "error": row.get("error"),
                "source_report": str(path),
                "source_report_sha256": sha256_file(path),
            }
            if row.get("passed") is True:
                valid.add(uid)
    return valid, rows


def _recon_valid_seed42(paths: list[Path]) -> tuple[set[str], dict[str, Any]]:
    valid: set[str] = set()
    rows: dict[str, Any] = {}
    for path in paths:
        report = _verify_report(path)
        for row in report.get("records", []):
            if int(row.get("seed", -1)) != 42:
                continue
            uid = str(row.get("object_uid", ""))
            if not uid or uid in rows:
                raise RuntimeError(f"duplicate/empty ReconViaGen seed42 UID: {uid!r}")
            rows[uid] = {
                "passed": row.get("passed") is True,
                "surface": row.get("surface"),
                "structure": row.get("structure"),
                "error": row.get("error"),
                "source_report": str(path),
                "source_report_sha256": sha256_file(path),
            }
            if row.get("passed") is True:
                valid.add(uid)
    return valid, rows


def _save_input_views(
    *, render_tar: Path, uid: str, view_ids: list[int], destination: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    views, archive_audit = _load_views_with_audit(render_tar, uid)
    by_id = {int(row["id"]): row for row in views}
    if len(by_id) != len(views):
        raise RuntimeError(f"uid={uid} render archive has duplicate view IDs")
    missing = [view_id for view_id in view_ids if view_id not in by_id]
    if missing:
        raise RuntimeError(f"uid={uid} selected views are missing: {missing}")
    destination.mkdir(parents=True, exist_ok=False)
    records: list[dict[str, Any]] = []
    thumbnails: list[tuple[Image.Image, str]] = []
    for position, view_id in enumerate(view_ids):
        rgba = np.ascontiguousarray(by_id[view_id]["rgba"], dtype=np.uint8)
        image = Image.fromarray(rgba, mode="RGBA")
        rgba_path = destination / f"view_{position:02d}_id_{view_id:03d}_rgba.png"
        image.save(rgba_path)
        white = Image.new("RGB", image.size, "white")
        white.paste(image.convert("RGB"), mask=image.getchannel("A"))
        rgb_path = destination / f"view_{position:02d}_id_{view_id:03d}_rgb.png"
        white.save(rgb_path)
        thumb = white.copy()
        thumb.thumbnail((320, 320), Image.Resampling.LANCZOS)
        thumbnails.append((thumb, f"selected[{position}]  view_id={view_id}"))
        records.append(
            {
                "position": position,
                "view_id": view_id,
                "rgba_shape": list(rgba.shape),
                "rgba_array_sha256": hashlib.sha256(rgba.tobytes()).hexdigest(),
                "rgba_png": str(rgba_path),
                "rgba_png_sha256": sha256_file(rgba_path),
                "rgb_png": str(rgb_path),
                "rgb_png_sha256": sha256_file(rgb_path),
            }
        )
    cell_w, cell_h, header = 320, 320, 28
    sheet = Image.new("RGB", (cell_w * 4, (cell_h + header) * 2), (24, 24, 24))
    draw = ImageDraw.Draw(sheet)
    for position, (thumb, label) in enumerate(thumbnails):
        x = (position % 4) * cell_w
        y = (position // 4) * (cell_h + header)
        sheet.paste(thumb, (x + (cell_w - thumb.width) // 2, y + header))
        draw.text((x + 6, y + 7), label, fill=(240, 240, 240))
    sheet_path = destination / "冻结输入8视图总览.png"
    sheet.save(sheet_path)
    return records, {
        "path": str(sheet_path),
        "sha256": sha256_file(sheet_path),
        "render_archive_audit": archive_audit,
    }


def prepare(args: argparse.Namespace) -> None:
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    dev_path = Path(args.dev_split).expanduser().resolve(strict=True)
    cache_path = Path(args.cache_report).expanduser().resolve(strict=True)
    endpoint_paths = [Path(value).expanduser().resolve(strict=True) for value in args.endpoint_reports]
    recon_paths = [Path(value).expanduser().resolve(strict=True) for value in args.recon_reports]
    dev = load_json(dev_path)
    cache = _verify_report(cache_path)
    if (
        dev.get("name") != "dev"
        or int(dev.get("count", -1)) != 64
        or len(dev.get("rows", [])) != 64
        or cache.get("split") != "dev"
        or int(cache.get("object_count", -1)) != 64
        or dev.get("protocol_sha256") != cache.get("protocol_sha256")
    ):
        raise RuntimeError("frozen official Dev64 split/cache identity differs")
    dev_rows = list(dev["rows"])
    cache_rows = list(cache["records"])
    if [str(row["uid"]) for row in dev_rows] != [str(row["uid"]) for row in cache_rows]:
        raise RuntimeError("Dev64 split/cache UID ordering differs")
    endpoint_valid, endpoint_rows = _endpoint_valid_seed42(endpoint_paths)
    recon_valid, recon_rows = _recon_valid_seed42(recon_paths)
    eligible = [
        index
        for index, row in enumerate(dev_rows)
        if index >= int(args.object_start)
        and str(row["uid"]) in endpoint_valid
        and str(row["uid"]) in recon_valid
    ]
    if len(eligible) < int(args.count):
        raise RuntimeError("too few two-endpoint-valid held-out Dev objects")
    chosen = random.Random(int(args.random_seed)).sample(eligible, int(args.count))
    output.mkdir(parents=True)
    selections = []
    for position, index in enumerate(chosen, start=1):
        split_row = dev_rows[index]
        cache_row = cache_rows[index]
        uid = str(split_row["uid"])
        render_tar = Path(split_row["render_tar"]).expanduser().resolve(strict=True)
        if render_tar.stat().st_size != int(split_row["render_size"]):
            raise RuntimeError(f"render size differs: {render_tar}")
        view_ids = [int(value) for value in cache_row["selected_view_ids"]]
        case_dir = output / f"case_{position:02d}_dev_index_{index:02d}_{uid[:12]}"
        view_records, sheet = _save_input_views(
            render_tar=render_tar,
            uid=uid,
            view_ids=view_ids,
            destination=case_dir / "原始冻结输入8视图",
        )
        selections.append(
            {
                "selection_position": position,
                "dev_index": index,
                "uid": uid,
                "object_uid": uid,
                "case_dir": str(case_dir),
                "render_tar": str(render_tar),
                "render_tar_sha256": sha256_file(render_tar),
                "selected_view_ids": view_ids,
                "selected_views": view_records,
                "input_sheet": sheet,
                "existing_seed42_endpoint_record": endpoint_rows[uid],
                "existing_seed42_reconviagen_record": recon_rows[uid],
            }
        )
    excluded = [
        {
            "dev_index": index,
            "uid": str(row["uid"]),
            "vss_vslat_seed42_valid": str(row["uid"]) in endpoint_valid,
            "reconviagen_seed42_valid": str(row["uid"]) in recon_valid,
        }
        for index, row in enumerate(dev_rows)
        if index >= int(args.object_start) and index not in eligible
    ]
    payload = {
        "format": SELECTION_FORMAT,
        "created_at_utc": utc_now(),
        "passed": True,
        "random_seed": int(args.random_seed),
        "inference_seed": 42,
        "requested_count": int(args.count),
        "sampling_frame": (
            "official held-out Dev[16:64), restricted before random draw to objects "
            "whose already-completed seed42 strict ReconViaGen and VSS2k+V-SLat15k "
            "records both decoded valid Meshes"
        ),
        "eligible_count": len(eligible),
        "eligible_dev_indices": eligible,
        "excluded_ineligible": excluded,
        "selection_order": chosen,
        "dev_split": str(dev_path),
        "dev_split_sha256": sha256_file(dev_path),
        "cache_report": str(cache_path),
        "cache_report_sha256": sha256_file(cache_path),
        "endpoint_reports": [
            {"path": str(path), "sha256": sha256_file(path)} for path in endpoint_paths
        ],
        "reconviagen_reports": [
            {"path": str(path), "sha256": sha256_file(path)} for path in recon_paths
        ],
        "selections": selections,
        "scope_guard": (
            "Qualitative fixed random sample; it is not a statistical estimate. "
            "Chapter 158 C-A is positive, but C-B does not establish an incremental "
            "V-SLat15k gain; do not attribute the full endpoint gain to V-SLat alone."
        ),
    }
    _write_hashed_json(output / "selection.json", payload)
    (output / "README.md").write_text(
        "# ProObjaverse Dev 固定随机两对象定性对照\n\n"
        f"- 随机种子：{args.random_seed}\n"
        "- 采样域：held-out Dev[16:64)，先排除任一端 seed42 无有效 Mesh 的对象。\n"
        f"- 合格对象：{len(eligible)}；抽样索引：{chosen}\n"
        "- 对照：strict ReconViaGen 与 VSS step2000 + V-SLat step15000。\n"
        "- 每例保存 frozen selected_view_ids 对应的全部 8 帧原始 RGBA/RGB。\n\n"
        "注意：158 章的 C-A 端到端收益为正，但 C-B 未证明 V-SLat15k 相对 V0 的独立收益；"
        "这里不得把完整端点收益全部写成 pose-DINO SLat 的收益。\n",
        encoding="utf-8",
    )
    print(json.dumps({"passed": True, "eligible": len(eligible), "chosen": chosen, "selection": str(output / 'selection.json')}, indent=2))


def _selection(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    path = Path(args.selection).expanduser().resolve(strict=True)
    manifest = _load_hashed_json(path, expected_format=SELECTION_FORMAT)
    position = int(args.selection_position)
    matches = [row for row in manifest["selections"] if int(row["selection_position"]) == position]
    if len(matches) != 1:
        raise RuntimeError(f"selection position is not unique: {position}")
    return manifest, matches[0]


@torch.no_grad()
def reconviagen(args: argparse.Namespace) -> None:
    manifest, row = _selection(args)
    output = Path(row["case_dir"]) / "ReconViaGen_original_seed42"
    mesh_path = output / "mesh.obj"
    report_path = output / "result.json"
    if output.exists():
        if not args.resume or not mesh_path.is_file() or not report_path.is_file():
            raise FileExistsError(output)
        report = _load_hashed_json(report_path, expected_format=RECON_EXPORT_FORMAT)
        if report.get("mesh_sha256") != sha256_file(mesh_path):
            raise RuntimeError("reusable strict ReconViaGen Mesh hash differs")
        print(json.dumps({"reused": True, "mesh": str(mesh_path)}, indent=2))
        return
    render = Path(row["render_tar"]).resolve(strict=True)
    if sha256_file(render) != row["render_tar_sha256"]:
        raise RuntimeError("selected render archive changed")
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("strict ReconViaGen export requires CUDA")
    pipeline = _build_pipeline(args.pretrained, device, bool(args.low_vram))
    outputs = coords = ss_noise = decoded = mesh = None
    try:
        source = {
            "uid": row["uid"],
            "render_tar": render,
            "selected_view_ids": row["selected_view_ids"],
        }
        images, input_identity, archive_audit = _selected_images(source, pipeline)
        if images is None or len(images) != 8:
            raise RuntimeError("strict ReconViaGen did not receive eight frozen views")
        outputs, coords, ss_noise = pipeline.run(
            image=images,
            seed=42,
            formats=["mesh"],
            preprocess_image=False,
            sparse_structure_sampler_params=STRICT_SAMPLING["sparse_structure"],
            slat_sampler_params=STRICT_SAMPLING["slat"],
            mode=STRICT_SAMPLING["multiimage_algo"],
        )
        decoded = outputs["mesh"][0]
        mesh = decoded.to_trimesh(transform_pose=False)
        structure = mesh_structure_metrics(mesh)
        if not structure["mesh_success"]:
            raise RuntimeError("strict ReconViaGen qualitative Mesh is invalid")
        output.mkdir(parents=True, exist_ok=False)
        temporary = mesh_path.with_name(f".{mesh_path.name}.tmp-{os.getpid()}")
        mesh.export(temporary, file_type="obj")
        os.replace(temporary, mesh_path)
        report = {
            "format": RECON_EXPORT_FORMAT,
            "created_at_utc": utc_now(),
            "passed": True,
            "method": "strict_reconviagen_vggt_stock_ss_stock_slat_stock_decoder",
            "uid": row["uid"],
            "dev_index": int(row["dev_index"]),
            "seed": 42,
            "selection_manifest": str(Path(args.selection).resolve()),
            "selection_manifest_sha256": sha256_file(args.selection),
            "render_tar": str(render),
            "render_tar_sha256": row["render_tar_sha256"],
            "selected_view_ids": row["selected_view_ids"],
            "selected_input_sha256": input_identity,
            "render_archive_audit": archive_audit,
            "pretrained": str(args.pretrained),
            "sampling": STRICT_SAMPLING,
            "sampling_sha256": canonical_sha256(STRICT_SAMPLING),
            "coord_count": int(coords.shape[0]),
            "ss_noise_shape": list(ss_noise.shape),
            "structure": structure,
            "mesh": str(mesh_path),
            "mesh_sha256": sha256_file(mesh_path),
            "output_frame": "latent decoder canonical; transform_pose=False",
            "explicit_camera_pose_consumed": False,
            "target_or_metric_consumed_during_inference": False,
        }
        _write_hashed_json(report_path, report)
        print(json.dumps({"passed": True, "mesh": str(mesh_path), "coord_count": int(coords.shape[0])}, indent=2))
    finally:
        del pipeline, outputs, coords, ss_noise, decoded, mesh
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def finalize(args: argparse.Namespace) -> None:
    manifest, selected = _selection(args)
    case_dir = Path(selected["case_dir"])
    endpoint = Path(args.endpoint_worker).expanduser().resolve(strict=True)
    endpoint_report_path = endpoint / "report.json"
    endpoint_report = _verify_report(endpoint_report_path)
    rows = [
        row
        for row in endpoint_report.get("mesh_branch_records", [])
        if row.get("branch") == "native_trained"
        and int(row.get("seed", -1)) == 42
        and str(row.get("object_uid", "")) == str(selected["uid"])
    ]
    if len(rows) != 1 or rows[0].get("passed") is not True:
        raise RuntimeError("endpoint worker lacks one valid native_trained seed42 row")
    source_mesh = Path(rows[0].get("mesh", "")).resolve(strict=True)
    if rows[0].get("mesh_sha256") != sha256_file(source_mesh):
        raise RuntimeError("endpoint source Mesh hash differs")
    destination = case_dir / "VSS2k_VSLat15k_seed42" / "mesh.obj"
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=False)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    shutil.copy2(source_mesh, temporary)
    os.replace(temporary, destination)
    recon_report_path = case_dir / "ReconViaGen_original_seed42" / "result.json"
    recon_report = _load_hashed_json(recon_report_path, expected_format=RECON_EXPORT_FORMAT)
    report = {
        "format": CASE_FORMAT,
        "created_at_utc": utc_now(),
        "passed": True,
        "selection_manifest": str(Path(args.selection).resolve()),
        "selection_manifest_sha256": sha256_file(args.selection),
        "selection_position": int(selected["selection_position"]),
        "dev_index": int(selected["dev_index"]),
        "uid": selected["uid"],
        "seed": 42,
        "selected_view_ids": selected["selected_view_ids"],
        "input_sheet": selected["input_sheet"],
        "reconviagen": {
            "mesh": recon_report["mesh"],
            "mesh_sha256": recon_report["mesh_sha256"],
            "structure": recon_report["structure"],
        },
        "vss2k_vslat15k": {
            "mesh": str(destination),
            "mesh_sha256": sha256_file(destination),
            "structure": rows[0]["structure"],
            "surface": rows[0].get("surface"),
            "source_worker_report": str(endpoint_report_path),
            "source_worker_report_sha256": sha256_file(endpoint_report_path),
        },
        "comparison_note": (
            "Meshes come from different complete pipelines. Chapter 158 establishes a "
            "positive C-A full endpoint result, but does not establish a positive C-B "
            "increment from V-SLat15k alone."
        ),
    }
    _write_hashed_json(case_dir / "case_report.json", report)
    print(json.dumps({"passed": True, "case": str(case_dir), "ours_mesh": str(destination)}, indent=2))


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare")
    p.add_argument("--dev_split", required=True)
    p.add_argument("--cache_report", required=True)
    p.add_argument("--endpoint_reports", action="append", required=True)
    p.add_argument("--recon_reports", action="append", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--object_start", type=int, default=16)
    p.add_argument("--count", type=int, default=2)
    p.add_argument("--random_seed", type=int, default=20260818)
    r = commands.add_parser("reconviagen")
    r.add_argument("--selection", required=True)
    r.add_argument("--selection_position", type=int, required=True)
    r.add_argument("--pretrained", default="Stable-X/trellis-vggt-v0-2")
    r.add_argument("--device", default="cuda")
    r.add_argument("--low_vram", action="store_true")
    r.add_argument("--resume", action="store_true")
    f = commands.add_parser("finalize")
    f.add_argument("--selection", required=True)
    f.add_argument("--selection_position", type=int, required=True)
    f.add_argument("--endpoint_worker", required=True)
    return parser


def main() -> None:
    args = make_parser().parse_args()
    {"prepare": prepare, "reconviagen": reconviagen, "finalize": finalize}[args.command](args)


if __name__ == "__main__":
    main()
