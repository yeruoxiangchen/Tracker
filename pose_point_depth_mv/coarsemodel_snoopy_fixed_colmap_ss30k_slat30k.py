#!/usr/bin/env python3
"""Audit, materialize, and finalize the fixed eight-view Snoopy test.

The experiment exposes only the eight RGB frames requested by the user.  It
uses their source masks and the existing all-frame COLMAP model only after a
strict registration/reprojection audit.  The current endpoint and strict
ReconViaGen endpoint are both evaluated from the same frozen derived input.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from typing import Any

import cv2
import numpy as np

from pose_point_depth_mv.coarsemodel_fixed_colmap_ss30k_slat30k import (
    ABC_R_EVIDENCE,
    SLAT_CHECKPOINT,
    SS_CHECKPOINT,
    SS_REPORT,
    STOCK_FREEZE,
    _asset,
    _colmap_audit,
    _copy_verified,
    _only_object,
    _source_mask,
)
from pose_point_depth_mv.omni_real_benchmark_common import (
    atomic_json,
    load_json,
    sha256_file,
)
from pose_point_depth_mv.trellis_mesh_coordinate_contract import (
    MESH_FRAME_CONTRACT,
    validate_runtime_o_mesh_frame_contract,
)


TRACKER_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DATASET = TRACKER_ROOT / "CoarseModel/datasets/snoopy"
DEPENDENCY_SOURCE = (
    TRACKER_ROOT / "pose_point_depth_mv/coarsemodel_fixed_colmap_ss30k_slat30k.py"
)
DEFAULT_OUTPUT = (
    TRACKER_ROOT
    / "pose_point_depth_mv/outputs2/"
    "CoarseModel_snoopy_指定8帧_COLMAPPose_"
    "ReconViaGen_vs_SS30K_SLat30K_runtimeO轮廓_20260819_v1"
)
FRAMES = (
    "00001.jpg",
    "00021.jpg",
    "00051.jpg",
    "00089.jpg",
    "00101.jpg",
    "00125.jpg",
    "00131.jpg",
    "00150.jpg",
)

PLAN_FORMAT = "pose_point_depth_mv.coarsemodel_snoopy_fixed_colmap_plan.v1"
INPUT_FORMAT = "pose_point_depth_mv.coarsemodel_snoopy_fixed_colmap_input.v1"
REPORT_FORMAT = "pose_point_depth_mv.coarsemodel_snoopy_fixed_colmap_ss30k_slat30k.v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _input_root(output: Path) -> Path:
    return output / "objects/snoopy/00_fixed_input"


def _derived_dataset(output: Path) -> Path:
    return _input_root(output) / "snoopy_fixed8"


def _mask_qc(rgb_path: Path, mask_path: Path) -> dict[str, Any]:
    rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if rgb is None or mask is None:
        raise RuntimeError(f"failed to read RGB/mask: {rgb_path} {mask_path}")
    if rgb.shape[:2] != mask.shape:
        raise RuntimeError(f"RGB/mask shape differs: {rgb_path} {mask_path}")
    binary = np.asarray(mask > 127, dtype=np.uint8)
    foreground = int(binary.sum())
    components, _, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    component_areas = sorted(
        [int(value) for value in stats[1:, cv2.CC_STAT_AREA]], reverse=True
    )
    ratio = foreground / max(1, int(binary.size))
    checks = {
        "nonempty": foreground > 0,
        "foreground_ratio_in_0p005_0p40": 0.005 <= ratio <= 0.40,
        "component_count_le_4": components - 1 <= 4,
        "largest_component_fraction_ge_0p70": bool(
            foreground and component_areas and component_areas[0] / foreground >= 0.70
        ),
    }
    return {
        "passed": all(checks.values()),
        "height": int(mask.shape[0]),
        "width": int(mask.shape[1]),
        "foreground_pixels": foreground,
        "foreground_ratio": ratio,
        "component_count": components - 1,
        "component_areas": component_areas,
        "checks": checks,
    }


def _requested_plan(output: Path) -> dict[str, Any]:
    dataset = SOURCE_DATASET.resolve(strict=True)
    audit = _colmap_audit(dataset, FRAMES)
    if audit.get("passed") is not True:
        raise RuntimeError(f"source COLMAP failed frozen gates: {audit['checks']}")
    bindings = []
    for frame in FRAMES:
        color = (dataset / "color" / frame).resolve(strict=True)
        image_copy = (dataset / "images" / frame).resolve(strict=True)
        rgb_copy = (dataset / "rgb" / frame).resolve(strict=True)
        if not (sha256_file(color) == sha256_file(image_copy) == sha256_file(rgb_copy)):
            raise RuntimeError(f"color/images/rgb source copies differ: {frame}")
        mask = _source_mask(dataset, frame)
        qc = _mask_qc(color, mask)
        if qc["passed"] is not True:
            raise RuntimeError(f"source mask failed frozen QC: {frame}: {qc['checks']}")
        bindings.append(
            {
                "frame_name": frame,
                "requested_color": _asset(color),
                "images_copy_same_sha256": True,
                "rgb_copy_same_sha256": True,
                "source_mask": _asset(mask),
                "source_mask_qc": qc,
            }
        )
    return {
        "format": PLAN_FORMAT,
        "created_at_utc": utc_now(),
        "passed": True,
        "object_count": 1,
        "object": {
            "name": "snoopy",
            "source_dataset": str(dataset),
            "requested_frames_in_order": list(FRAMES),
            "selected_view_count": len(FRAMES),
            "mask_source": "frozen_source_masks",
            "source_bindings": bindings,
            "colmap_model": {
                filename: _asset(dataset / "sparse/0" / filename)
                for filename in ("cameras.txt", "images.txt", "points3D.txt")
            },
            "colmap_quality_audit": audit,
            "colmap_recomputed": False,
            "derived_dataset": str(_derived_dataset(output).resolve()),
        },
        "deployments": {
            "native_ss_report": _asset(SS_REPORT),
            "native_ss_checkpoint": _asset(SS_CHECKPOINT),
            "native_ss_checkpoint_step": 30000,
            "native_slat_checkpoint": _asset(SLAT_CHECKPOINT),
            "native_slat_checkpoint_step": 30000,
            "abc_r_evidence": _asset(ABC_R_EVIDENCE),
            "stock_slat_freeze": _asset(STOCK_FREEZE),
        },
        "geometry_mode": "pose_mask",
        "view_selection": "exact_user_order_all_views",
        "current_endpoint": (
            "DINO-only -> official Native-SS30K -> Native-SLat30K -> Stock decoder"
        ),
        "reconviagen_endpoint": (
            "strict original VGGT -> Stock SS -> Stock SLat -> Stock decoder"
        ),
        "mesh_frame_contract": MESH_FRAME_CONTRACT,
        "implementation": _asset(Path(__file__)),
        "shared_dependency": _asset(DEPENDENCY_SOURCE),
        "scope_guard": (
            "Only the eight explicitly requested Snoopy RGB frames and their exact source "
            "masks are exposed to either endpoint. Intrinsics and T_W2C come from the "
            "existing all-frame sparse/0 COLMAP model after strict audit. No target Mesh, "
            "old reconstruction Mesh, ICP, or camera fitting is consumed."
        ),
    }


def prepare(output: Path) -> None:
    output = output.expanduser().resolve()
    requested = _requested_plan(output)
    plan_path = output / "experiment_plan.json"
    if plan_path.is_file():
        existing = load_json(plan_path)
        for key, value in requested.items():
            if key != "created_at_utc" and existing.get(key) != value:
                raise RuntimeError(f"existing experiment plan differs: field={key}")
        reused = True
    else:
        if output.exists() and any(output.iterdir()):
            raise RuntimeError(f"unbound nonempty output: {output}")
        output.mkdir(parents=True, exist_ok=True)
        atomic_json(plan_path, requested)
        reused = False
    print(json.dumps({"passed": True, "reused": reused, "plan": str(plan_path)}, indent=2))


def materialize_input(output: Path) -> None:
    output = output.expanduser().resolve(strict=True)
    plan = load_json(output / "experiment_plan.json")
    if plan.get("format") != PLAN_FORMAT or plan.get("passed") is not True:
        raise RuntimeError("experiment plan is not eligible")
    dataset = _derived_dataset(output)
    report_path = _input_root(output) / "fixed_input_report.json"
    if report_path.is_file():
        report = load_json(report_path)
        if report.get("passed") is not True:
            raise RuntimeError("existing fixed input report did not pass")
        print(json.dumps({"passed": True, "reused": True, "report": str(report_path)}, indent=2))
        return
    if dataset.exists() and any(dataset.iterdir()):
        raise RuntimeError(f"unbound partial derived dataset: {dataset}")
    (dataset / "images").mkdir(parents=True, exist_ok=True)
    (dataset / "masks").mkdir(parents=True, exist_ok=True)
    (dataset / "sparse/0").mkdir(parents=True, exist_ok=True)
    bindings = []
    for frame in FRAMES:
        source_rgb = SOURCE_DATASET / "color" / frame
        source_mask = _source_mask(SOURCE_DATASET, frame)
        target_rgb = dataset / "images" / frame
        target_mask = dataset / "masks" / f"{Path(frame).stem}.png"
        _copy_verified(source_rgb, target_rgb)
        _copy_verified(source_mask, target_mask)
        bindings.append(
            {
                "frame_name": frame,
                "source_color": _asset(source_rgb),
                "derived_image": _asset(target_rgb),
                "source_mask": _asset(source_mask),
                "derived_mask": _asset(target_mask),
                "mask_qc": _mask_qc(target_rgb, target_mask),
            }
        )
    for filename in ("cameras.txt", "images.txt", "points3D.txt"):
        _copy_verified(
            SOURCE_DATASET / "sparse/0" / filename,
            dataset / "sparse/0" / filename,
        )
    if sorted(path.name for path in (dataset / "images").iterdir()) != sorted(FRAMES):
        raise RuntimeError("derived image scope differs")
    expected_masks = sorted(f"{Path(frame).stem}.png" for frame in FRAMES)
    if sorted(path.name for path in (dataset / "masks").iterdir()) != expected_masks:
        raise RuntimeError("derived mask scope differs")
    payload = {
        "format": INPUT_FORMAT,
        "created_at_utc": utc_now(),
        "passed": True,
        "object": "snoopy",
        "source_dataset": str(SOURCE_DATASET.resolve(strict=True)),
        "derived_dataset": str(dataset.resolve(strict=True)),
        "frames_in_user_order": list(FRAMES),
        "view_count": len(FRAMES),
        "mask_source": "frozen_source_masks",
        "bindings": bindings,
        "colmap_model": plan["object"]["colmap_model"],
        "colmap_quality_audit": plan["object"]["colmap_quality_audit"],
        "only_requested_images_exposed": True,
        "source_dataset_modified": False,
    }
    atomic_json(report_path, payload)
    print(json.dumps({"passed": True, "reused": False, "report": str(report_path)}, indent=2))


def finalize(output: Path) -> None:
    output = output.expanduser().resolve(strict=True)
    plan_path = output / "experiment_plan.json"
    plan = load_json(plan_path)
    if plan.get("format") != PLAN_FORMAT or plan.get("passed") is not True:
        raise RuntimeError("experiment plan is not eligible")
    root = output / "objects/snoopy"
    paths = {
        "input": root / "00_fixed_input/fixed_input_report.json",
        "raw": root / "01_raw_cache/raw_cache_report.json",
        "runtime": root / "02_runtime_o/runtime_input_manifest.json",
        "model": root / "03_dino_only_input/model_input_manifest.json",
        "current": root / "04_current_ss30k_slat30k/inference_manifest.json",
        "recon": root / "05_reconviagen/inference_manifest.json",
        "contour": root / "06_current_camera_contours/report.json",
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(f"incomplete experiment artifact: {path}")
    payloads = {key: load_json(path) for key, path in paths.items()}
    if not all(payload.get("passed") is True for payload in payloads.values()):
        raise RuntimeError("one or more stages did not pass")
    raw_row = _only_object(payloads["raw"], paths["raw"])
    runtime_row = _only_object(payloads["runtime"], paths["runtime"])
    current_row = _only_object(payloads["current"], paths["current"])
    recon_row = _only_object(payloads["recon"], paths["recon"])
    if raw_row.get("selected_source_frame_names") != list(FRAMES):
        raise RuntimeError("raw exact-frame order differs")
    indices = [int(value) for value in runtime_row["selected_source_view_indices"]]
    if indices != list(range(len(FRAMES))):
        raise RuntimeError("runtime did not retain every requested view in order")
    if runtime_row.get("geometry_mode") != "pose_mask":
        raise RuntimeError("runtime geometry differs")
    validate_runtime_o_mesh_frame_contract(current_row)
    current_mesh = Path(current_row["mesh"]).resolve(strict=True)
    recon_mesh = Path(recon_row["mesh"]).resolve(strict=True)
    deployments = plan["deployments"]
    if current_row.get("native_ss_checkpoint_sha256") != deployments["native_ss_checkpoint"]["sha256"]:
        raise RuntimeError("current Native-SS identity differs")
    if current_row.get("native_slat_checkpoint_sha256") != deployments["native_slat_checkpoint"]["sha256"]:
        raise RuntimeError("current Native-SLat identity differs")
    if int(current_row.get("native_slat_checkpoint_step", -1)) != 30000:
        raise RuntimeError("current Native-SLat step differs")
    if current_row.get("vggt_model_executed") is not False:
        raise RuntimeError("current endpoint unexpectedly executed VGGT")
    if recon_row.get("method") != "reconviagen_original":
        raise RuntimeError("strict ReconViaGen identity differs")
    contour = payloads["contour"]
    if contour.get("mesh_o_sha256") != sha256_file(current_mesh):
        raise RuntimeError("contour/current Mesh binding differs")
    if contour.get("runtime_input_manifest_sha256") != sha256_file(paths["runtime"]):
        raise RuntimeError("contour/runtime binding differs")
    if contour.get("mesh_frame_contract") != MESH_FRAME_CONTRACT:
        raise RuntimeError("contour Mesh frame differs")
    if contour.get("projection_chain_audit", {}).get("passed") is not True:
        raise RuntimeError("contour projection chain audit failed")

    presentation = root / "07_结果汇总"
    current_copy = presentation / "当前_SS30K_SLat30K_runtime-O.obj"
    recon_copy = presentation / "ReconViaGen原版.obj"
    _copy_verified(current_mesh, current_copy)
    _copy_verified(recon_mesh, recon_copy)
    report = {
        "format": REPORT_FORMAT,
        "created_at_utc": utc_now(),
        "passed": True,
        "object_count": 1,
        "object": "snoopy",
        "view_count": len(FRAMES),
        "requested_frames_in_order": list(FRAMES),
        "colmap_recomputed": False,
        "colmap_quality_audit": payloads["input"]["colmap_quality_audit"],
        "fixed_input_report": _asset(paths["input"]),
        "runtime_input_manifest": _asset(paths["runtime"]),
        "current_inference_manifest": _asset(paths["current"]),
        "reconviagen_inference_manifest": _asset(paths["recon"]),
        "current_mesh": _asset(current_copy),
        "reconviagen_mesh": _asset(recon_copy),
        "contour_report": _asset(paths["contour"]),
        "contour_overview": _asset(Path(contour["overview"])),
        "experiment_plan": _asset(plan_path),
        "current_endpoint": plan["current_endpoint"],
        "reconviagen_endpoint": plan["reconviagen_endpoint"],
        "all_current_meshes_native_runtime_o": True,
        "all_contours_bound_to_exact_current_mesh_and_colmap_runtime": True,
        "source_colmap_recomputed": False,
        "source_dataset_modified": False,
        "interpretation": (
            "This is a one-object qualitative real-image test with eight fixed user-selected "
            "frames. Cyan contours project the exact native runtime-O current Mesh through "
            "T_O2W, the source all-frame COLMAP T_W2C, calibrated intrinsics and distortion."
        ),
    }
    atomic_json(output / "report.json", report)
    print(json.dumps({"passed": True, "report": str(output / 'report.json')}, ensure_ascii=False, indent=2))


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "materialize-input", "finalize"):
        child = sub.add_parser(name)
        child.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main() -> None:
    args = make_parser().parse_args()
    if args.command == "prepare":
        prepare(args.output)
    elif args.command == "materialize-input":
        materialize_input(args.output)
    else:
        finalize(args.output)


if __name__ == "__main__":
    main()
