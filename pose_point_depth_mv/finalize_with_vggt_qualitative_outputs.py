#!/usr/bin/env python3
"""Validate and package the fixed Dev/Omni with-VGGT qualitative outputs."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
from typing import Any

from pose_point_depth_mv.export_proobjaverse_dev_with_vggt_qualitative import (
    CASE_FORMAT,
    SELECTION_FORMAT,
)
from pose_point_depth_mv.infer_omni_real_official_with_vggt import (
    MANIFEST_FORMAT as OURS_MANIFEST_FORMAT,
)
from pose_point_depth_mv.infer_omni_real_reconviagen import (
    MANIFEST_FORMAT as RECON_MANIFEST_FORMAT,
)
from pose_point_depth_mv.omni_real_benchmark_common import (
    atomic_json,
    load_json,
    sha256_file,
)
from pose_point_depth_mv.proobjaverse_official_slat_protocol import canonical_sha256
from pose_point_depth_mv.render_runtime_o_mesh_camera_contours import (
    FORMAT as CONTOUR_FORMAT,
)
from pose_point_depth_mv.trellis_mesh_coordinate_contract import (
    validate_runtime_o_mesh_frame_contract,
)


DEV_SUMMARY_FORMAT = "pose_point_depth_mv.proobjaverse_dev_with_vggt_qualitative_summary.v1"
OMNI_SUMMARY_FORMAT = "pose_point_depth_mv.omni_with_vggt_qualitative_summary.v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _hashed(path: Path, *, expected_format: str, hash_key: str) -> dict[str, Any]:
    payload = load_json(path)
    if payload.get("format") != expected_format or payload.get("passed") is not True:
        raise RuntimeError(f"qualitative artifact did not pass: {path}")
    if hash_key:
        body = dict(payload)
        saved = str(body.pop(hash_key, ""))
        if not saved or canonical_sha256(body) != saved:
            raise RuntimeError(f"qualitative artifact hash differs: {path}")
    return payload


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    shutil.copy2(source, temporary)
    os.replace(temporary, destination)


def finalize_dev(args: argparse.Namespace) -> None:
    root = Path(args.output_dir).expanduser().resolve(strict=True)
    selection_path = root / "selection.json"
    selection = _hashed(
        selection_path, expected_format=SELECTION_FORMAT, hash_key="manifest_sha256"
    )
    if len(selection.get("selections", [])) != 2:
        raise RuntimeError("Dev qualitative selection is not exactly two objects")
    cases = []
    for row in selection["selections"]:
        case_dir = Path(row["case_dir"]).resolve(strict=True)
        report_path = case_dir / "case_report.json"
        report = _hashed(
            report_path, expected_format=CASE_FORMAT, hash_key="manifest_sha256"
        )
        for branch in ("reconviagen", "vss2k_vslat15k"):
            mesh = Path(report[branch]["mesh"]).resolve(strict=True)
            if report[branch]["mesh_sha256"] != sha256_file(mesh):
                raise RuntimeError(f"Dev qualitative Mesh hash differs: {mesh}")
        cases.append(
            {
                "selection_position": row["selection_position"],
                "dev_index": row["dev_index"],
                "uid": row["uid"],
                "case_dir": str(case_dir),
                "case_report": str(report_path),
                "case_report_sha256": sha256_file(report_path),
                "input_overview": row["input_sheet"]["path"],
                "reconviagen_mesh": report["reconviagen"]["mesh"],
                "vss2k_vslat15k_mesh": report["vss2k_vslat15k"]["mesh"],
            }
        )
    summary = {
        "format": DEV_SUMMARY_FORMAT,
        "created_at_utc": utc_now(),
        "passed": True,
        "selection": str(selection_path),
        "selection_sha256": sha256_file(selection_path),
        "random_seed": selection["random_seed"],
        "inference_seed": selection["inference_seed"],
        "object_count": len(cases),
        "cases": cases,
        "scope_guard": selection["scope_guard"],
    }
    atomic_json(root / "summary.json", summary)
    print(
        json.dumps(
            {"passed": True, "objects": len(cases), "summary": str(root / "summary.json")},
            indent=2,
            ensure_ascii=False,
        )
    )


def _single_result(manifest: dict[str, Any], *, label: str) -> dict[str, Any]:
    rows = manifest.get("objects", [])
    if len(rows) != 1:
        raise RuntimeError(f"{label} manifest is not one object/seed")
    row = rows[0]
    mesh = Path(row["mesh"]).resolve(strict=True)
    if row.get("mesh_sha256") != sha256_file(mesh):
        raise RuntimeError(f"{label} Mesh hash differs: {mesh}")
    return row


def finalize_omni(args: argparse.Namespace) -> None:
    root = Path(args.output_dir).expanduser().resolve(strict=True)
    runtime_path = Path(args.runtime_input_manifest).expanduser().resolve(strict=True)
    ours_path = Path(args.ours_manifest).expanduser().resolve(strict=True)
    recon_path = Path(args.reconviagen_manifest).expanduser().resolve(strict=True)
    contour_path = Path(args.contour_report).expanduser().resolve(strict=True)
    runtime = load_json(runtime_path)
    ours = _hashed(ours_path, expected_format=OURS_MANIFEST_FORMAT, hash_key="")
    recon = _hashed(recon_path, expected_format=RECON_MANIFEST_FORMAT, hash_key="")
    contour = _hashed(contour_path, expected_format=CONTOUR_FORMAT, hash_key="")
    if len(runtime.get("objects", [])) != 1:
        raise RuntimeError("frozen Omni runtime manifest is not exactly one object")
    frozen = runtime["objects"][0]
    expected_names = [
        "00000.jpg",
        "00084.jpg",
        "00171.jpg",
        "00255.jpg",
        "00342.jpg",
        "00426.jpg",
        "00515.jpg",
        "00599.jpg",
    ]
    expected_indices = [0, 9, 18, 27, 36, 45, 54, 63]
    if (
        frozen.get("selected_frame_names") != expected_names
        or frozen.get("selected_source_view_indices") != expected_indices
        or frozen.get("view_selection", {}).get("fallback_used") is not False
    ):
        raise RuntimeError("Omni replay no longer has the frozen exact eight-view selection")
    ours_row = _single_result(ours, label="VSS2k+V-SLat15k")
    validate_runtime_o_mesh_frame_contract(ours)
    validate_runtime_o_mesh_frame_contract(ours_row)
    recon_row = _single_result(recon, label="ReconViaGen")
    if (
        ours_row.get("object_key") != frozen.get("object_key")
        or recon_row.get("object_key") != frozen.get("object_key")
        or int(ours_row.get("seed", -1)) != 42
        or int(recon_row.get("seed", -1)) != 42
        or contour.get("mesh_o_sha256") != ours_row.get("mesh_sha256")
        or contour.get("mesh_frame_contract_verified") is not True
        or contour.get("mesh_frame_report") != ours_row.get("result")
        or contour.get("mesh_frame_report_sha256")
        != sha256_file(Path(ours_row["result"]).resolve(strict=True))
        or contour.get("selected_frame_names") != expected_names
    ):
        raise RuntimeError("Omni qualitative object/seed/contour binding differs")
    friendly_recon = root / "ReconViaGen_original_seed42" / "mesh_reference_o.obj"
    friendly_ours = root / "VSS2k_VSLat15k_seed42" / "mesh_o.obj"
    for source, destination in (
        (Path(recon_row["mesh"]), friendly_recon),
        (Path(ours_row["mesh"]), friendly_ours),
    ):
        if destination.is_file():
            if sha256_file(destination) != sha256_file(source):
                raise RuntimeError(f"friendly Mesh differs: {destination}")
        else:
            _atomic_copy(source, destination)
    summary = {
        "format": OMNI_SUMMARY_FORMAT,
        "created_at_utc": utc_now(),
        "passed": True,
        "object_key": frozen["object_key"],
        "inference_seed": 42,
        "runtime_input_manifest": str(runtime_path),
        "runtime_input_manifest_sha256": sha256_file(runtime_path),
        "selected_frame_names": expected_names,
        "selected_source_view_indices": expected_indices,
        "selection_policy": frozen["view_selection"],
        "reconviagen": {
            "mesh": str(friendly_recon),
            "mesh_sha256": sha256_file(friendly_recon),
            "native_output_frame": recon_row["output_frame"],
            "source_manifest": str(recon_path),
            "source_manifest_sha256": sha256_file(recon_path),
        },
        "vss2k_vslat15k": {
            "mesh": str(friendly_ours),
            "mesh_sha256": sha256_file(friendly_ours),
            "native_output_frame": ours_row["output_frame"],
            "mesh_frame_contract": ours_row["mesh_frame_contract"],
            "decoder_to_runtime_o_axis_transform": ours_row[
                "decoder_to_runtime_o_axis_transform"
            ],
            "decoder_mesh_export_policy": ours_row["decoder_mesh_export_policy"],
            "source_manifest": str(ours_path),
            "source_manifest_sha256": sha256_file(ours_path),
            "camera_contour_report": str(contour_path),
            "camera_contour_report_sha256": sha256_file(contour_path),
            "camera_contour_overview": contour["overview"],
        },
        "coordinate_note": (
            "The official endpoint decoder Mesh first applies the fixed TRELLIS "
            "(x,y,z)->(x,z,-y) axis rotation, then is runtime-O and is projected "
            "onto raw frames. Original ReconViaGen emits its own reference-view "
            "canonical frame."
        ),
        "scope_guard": (
            "Qualitative replay on one previously frozen real Omni sample. Chapter 158 "
            "supports the complete C-A endpoint; it does not isolate all gain to V-SLat."
        ),
    }
    atomic_json(root / "summary.json", summary)
    (root / "README.md").write_text(
        "# Omni 冻结 8 视图：ReconViaGen 与 VSS2k+V-SLat15k\n\n"
        "- 输入严格复用原回放目录已冻结的 8 帧，不重新筛帧。\n"
        "- `ReconViaGen_original_seed42`：原版 ReconViaGen 的原生参考视图坐标 Mesh。\n"
        "- `VSS2k_VSLat15k_seed42`：先执行 TRELLIS 固定轴变换 "
        "`(x,y,z)->(x,z,-y)` 后的真正 runtime-O Mesh。\n"
        "- `VSS2k_VSLat15k_相机轮廓`：通过物理 T_O2W、原始 T_W2C、K 与畸变投影到原图的青色轮廓。\n"
        "- 不使用 projective-normalized T_O2C_lifting 做原图轮廓。\n\n"
        "注意：这是单个真实样本的定性回放；158 章支持完整 C-A 端点收益，不能将全部收益单独归因给 V-SLat。\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "passed": True,
                "summary": str(root / "summary.json"),
                "ours_mesh": str(friendly_ours),
                "reconviagen_mesh": str(friendly_recon),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    dev = commands.add_parser("dev")
    dev.add_argument("--output_dir", required=True)
    omni = commands.add_parser("omni")
    omni.add_argument("--output_dir", required=True)
    omni.add_argument("--runtime_input_manifest", required=True)
    omni.add_argument("--ours_manifest", required=True)
    omni.add_argument("--reconviagen_manifest", required=True)
    omni.add_argument("--contour_report", required=True)
    return parser


def main() -> None:
    args = make_parser().parse_args()
    {"dev": finalize_dev, "omni": finalize_omni}[args.command](args)


if __name__ == "__main__":
    main()
