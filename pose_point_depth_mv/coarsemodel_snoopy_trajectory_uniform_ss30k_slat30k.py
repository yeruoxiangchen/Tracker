#!/usr/bin/env python3
"""Freeze and finalize Snoopy trajectory-uniform 4/16-view reconstruction."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any

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
    _qvec_to_rotation,
    _source_mask,
)
from pose_point_depth_mv.coarsemodel_snoopy_fixed_colmap_ss30k_slat30k import (
    _mask_qc,
)
from pose_point_depth_mv.dataset_tools.prepare_omni_real_runtime_inputs import (
    pose_continuity_segments,
)
from pose_point_depth_mv.omni_real_benchmark_common import (
    atomic_json,
    load_json,
    sha256_file,
)
from pose_point_depth_mv.server2 import uniform_trajectory_indices
from pose_point_depth_mv.trellis_mesh_coordinate_contract import (
    validate_runtime_o_mesh_frame_contract,
)


TRACKER_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DATASET = TRACKER_ROOT / "CoarseModel/datasets/snoopy"
DEFAULT_OUTPUT = (
    TRACKER_ROOT
    / "pose_point_depth_mv/outputs2/"
    "CoarseModel_snoopy_轨迹均匀4帧16帧_"
    "ReconViaGen_vs_SS30K_SLat30K_20260819_v1"
)
EXPECTED_SELECTIONS = {
    "trajectory_uniform4": ["00001.jpg", "00064.jpg", "00106.jpg", "00153.jpg"],
    "trajectory_uniform16": [
        "00001.jpg",
        "00018.jpg",
        "00032.jpg",
        "00045.jpg",
        "00054.jpg",
        "00064.jpg",
        "00075.jpg",
        "00087.jpg",
        "00092.jpg",
        "00098.jpg",
        "00106.jpg",
        "00115.jpg",
        "00124.jpg",
        "00134.jpg",
        "00143.jpg",
        "00153.jpg",
    ],
}

PLAN_FORMAT = "pose_point_depth_mv.coarsemodel_snoopy_trajectory_uniform_plan.v1"
INPUT_FORMAT = "pose_point_depth_mv.coarsemodel_snoopy_trajectory_uniform_input.v1"
REPORT_FORMAT = "pose_point_depth_mv.coarsemodel_snoopy_trajectory_uniform_ss30k_slat30k.v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_colmap_trajectory() -> tuple[list[str], np.ndarray, np.ndarray]:
    lines = [
        line.strip()
        for line in (SOURCE_DATASET / "sparse/0/images.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    if len(lines) % 2:
        raise RuntimeError("invalid COLMAP two-line image records")
    records = []
    for offset in range(0, len(lines), 2):
        header = lines[offset].split()
        name = str(header[9])
        rotation = _qvec_to_rotation(list(map(float, header[1:5])))
        translation = np.asarray(list(map(float, header[5:8])), dtype=np.float64)
        T_W2C = np.eye(4, dtype=np.float64)
        T_W2C[:3, :3] = rotation
        T_W2C[:3, 3] = translation
        records.append((name, -rotation.T @ translation, T_W2C))
    records.sort(key=lambda row: (int(Path(row[0]).stem), row[0]))
    names = [row[0] for row in records]
    centers = np.stack([row[1] for row in records], axis=0)
    poses = np.stack([row[2] for row in records], axis=0)
    color_names = sorted(
        path.name for path in (SOURCE_DATASET / "color").iterdir() if path.is_file()
    )
    if names != sorted(color_names, key=lambda name: (int(Path(name).stem), name)):
        raise RuntimeError("COLMAP registered trajectory does not exactly cover source RGB")
    return names, centers, poses


def _selection_contract() -> dict[str, Any]:
    names, centers, poses = _load_colmap_trajectory()
    segments, continuity = pose_continuity_segments(names, poses)
    if continuity["jump_count"] != 0 or len(segments) != 1:
        raise RuntimeError(f"COLMAP trajectory is discontinuous: {continuity}")
    steps = np.linalg.norm(np.diff(centers, axis=0), axis=1)
    diameter = max(
        float(np.linalg.norm(centers[i] - centers[j]))
        for i in range(len(centers))
        for j in range(i + 1, len(centers))
    )
    endpoints = float(np.linalg.norm(centers[0] - centers[-1]))
    branches: dict[str, Any] = {}
    for count in (4, 16):
        indices, audit = uniform_trajectory_indices(centers, count)
        selected = [names[index] for index in indices]
        key = f"trajectory_uniform{count}"
        if selected != EXPECTED_SELECTIONS[key]:
            raise RuntimeError(f"frozen trajectory selection differs: {key}: {selected}")
        audit = dict(audit)
        audit.update(
            {
                "policy": "chronological_cumulative_colmap_camera_translation_uniform_v1",
                "camera_center_source": "inverse_of_source_sparse_0_COLMAP_T_W2C",
                "required_view_count": count,
            }
        )
        branches[key] = {
            "view_count": count,
            "selected_indices": indices,
            "selected_frame_names": selected,
            "audit": audit,
        }
    if not set(branches["trajectory_uniform4"]["selected_indices"]).issubset(
        set(branches["trajectory_uniform16"]["selected_indices"])
    ):
        raise RuntimeError("trajectory-uniform 4-view selection is not nested in 16-view")
    return {
        "format": "pose_point_depth_mv.colmap_camera_trajectory_uniform_selection.v1",
        "passed": True,
        "candidate_count": len(names),
        "candidate_frame_names": names,
        "total_translation": float(steps.sum()),
        "translation_step_median": float(np.median(steps)),
        "translation_step_p95": float(np.percentile(steps, 95)),
        "translation_step_max": float(np.max(steps)),
        "trajectory_diameter": diameter,
        "endpoint_distance": endpoints,
        "endpoint_distance_over_diameter": endpoints / diameter,
        "trajectory_continuity": continuity,
        "branches": branches,
        "selection_inputs": ["chronological_registered_frame_order", "COLMAP_camera_center_W"],
        "image_or_mask_content_used_for_selection": False,
        "quality_metric_used_for_selection": False,
        "four_view_is_exact_subset_of_sixteen_view": True,
    }


def _derived_dataset(output: Path, branch: str) -> Path:
    count = int(branch.removeprefix("trajectory_uniform"))
    return output / "branches" / branch / "00_fixed_input" / f"snoopy_{branch}_{count}views"


def _requested_plan(output: Path) -> dict[str, Any]:
    selection = _selection_contract()
    branches = {}
    for branch, record in selection["branches"].items():
        frames = list(record["selected_frame_names"])
        audit = _colmap_audit(SOURCE_DATASET, frames)
        if audit["passed"] is not True:
            raise RuntimeError(f"selected COLMAP audit failed: {branch}: {audit['checks']}")
        bindings = []
        for frame in frames:
            color = (SOURCE_DATASET / "color" / frame).resolve(strict=True)
            mask = _source_mask(SOURCE_DATASET, frame)
            qc = _mask_qc(color, mask)
            if qc["passed"] is not True:
                raise RuntimeError(f"source mask failed: {branch}/{frame}")
            bindings.append(
                {
                    "frame_name": frame,
                    "source_color": _asset(color),
                    "source_mask": _asset(mask),
                    "source_mask_qc": qc,
                }
            )
        branches[branch] = {
            "view_count": len(frames),
            "selected_indices": list(record["selected_indices"]),
            "selected_frame_names": frames,
            "bindings": bindings,
            "colmap_quality_audit": audit,
            "derived_dataset": str(_derived_dataset(output, branch).resolve()),
        }
    return {
        "format": PLAN_FORMAT,
        "created_at_utc": utc_now(),
        "passed": True,
        "source_dataset": str(SOURCE_DATASET.resolve(strict=True)),
        "source_colmap_model": {
            filename: _asset(SOURCE_DATASET / "sparse/0" / filename)
            for filename in ("cameras.txt", "images.txt", "points3D.txt")
        },
        "selection": selection,
        "branches": branches,
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
        "runtime_view_selection": "lexical_even_over_exact_frozen_selected_only_dataset",
        "current_endpoint": "DINO-only -> official Native-SS30K -> Native-SLat30K -> Stock decoder",
        "reconviagen_endpoint": "strict original VGGT -> Stock SS -> Stock SLat -> Stock decoder",
        "contour_rendering_requested": False,
        "implementation": _asset(Path(__file__)),
        "algorithm_dependency": _asset(TRACKER_ROOT / "pose_point_depth_mv/server2.py"),
        "scope_guard": (
            "View selection uses only chronological order and cumulative translation of "
            "COLMAP camera centers. The two endpoints receive exactly the same frozen selected "
            "RGB/mask/pose set per branch. No contour rendering or target Mesh is used."
        ),
    }


def prepare(output: Path) -> None:
    output = output.expanduser().resolve()
    requested = _requested_plan(output)
    path = output / "experiment_plan.json"
    if path.is_file():
        existing = load_json(path)
        for key, value in requested.items():
            if key != "created_at_utc" and existing.get(key) != value:
                raise RuntimeError(f"existing experiment plan differs: field={key}")
        reused = True
    else:
        if output.exists() and any(output.iterdir()):
            raise RuntimeError(f"unbound nonempty output: {output}")
        output.mkdir(parents=True, exist_ok=True)
        atomic_json(path, requested)
        reused = False
    print(json.dumps({"passed": True, "reused": reused, "plan": str(path)}, indent=2))


def materialize(output: Path) -> None:
    output = output.expanduser().resolve(strict=True)
    plan = load_json(output / "experiment_plan.json")
    if plan.get("format") != PLAN_FORMAT or plan.get("passed") is not True:
        raise RuntimeError("experiment plan is not eligible")
    reports = []
    for branch, record in plan["branches"].items():
        dataset = _derived_dataset(output, branch)
        report_path = dataset.parent / "fixed_input_report.json"
        if report_path.is_file():
            if load_json(report_path).get("passed") is not True:
                raise RuntimeError(f"existing input report failed: {branch}")
            reports.append(str(report_path))
            continue
        if dataset.exists() and any(dataset.iterdir()):
            raise RuntimeError(f"unbound partial dataset: {dataset}")
        (dataset / "images").mkdir(parents=True, exist_ok=True)
        (dataset / "masks").mkdir(parents=True, exist_ok=True)
        (dataset / "sparse/0").mkdir(parents=True, exist_ok=True)
        bindings = []
        for frame in record["selected_frame_names"]:
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
                }
            )
        for filename in ("cameras.txt", "images.txt", "points3D.txt"):
            _copy_verified(
                SOURCE_DATASET / "sparse/0" / filename,
                dataset / "sparse/0" / filename,
            )
        exposed = sorted(path.name for path in (dataset / "images").iterdir())
        if exposed != sorted(record["selected_frame_names"]):
            raise RuntimeError(f"derived image scope differs: {branch}")
        payload = {
            "format": INPUT_FORMAT,
            "created_at_utc": utc_now(),
            "passed": True,
            "branch": branch,
            "source_dataset": str(SOURCE_DATASET.resolve(strict=True)),
            "derived_dataset": str(dataset.resolve(strict=True)),
            "frames_in_trajectory_order": list(record["selected_frame_names"]),
            "view_count": int(record["view_count"]),
            "bindings": bindings,
            "colmap_quality_audit": record["colmap_quality_audit"],
            "only_selected_images_exposed": True,
            "source_dataset_modified": False,
        }
        atomic_json(report_path, payload)
        reports.append(str(report_path))
    print(json.dumps({"passed": True, "reports": reports}, indent=2))


def finalize(output: Path) -> None:
    output = output.expanduser().resolve(strict=True)
    plan_path = output / "experiment_plan.json"
    plan = load_json(plan_path)
    if plan.get("format") != PLAN_FORMAT or plan.get("passed") is not True:
        raise RuntimeError("experiment plan is not eligible")
    results = []
    for branch, planned in plan["branches"].items():
        root = output / "branches" / branch
        paths = {
            "input": root / "00_fixed_input/fixed_input_report.json",
            "raw": root / "01_raw_cache/raw_cache_report.json",
            "runtime": root / "02_runtime_o/runtime_input_manifest.json",
            "model": root / "03_dino_only_input/model_input_manifest.json",
            "current": root / "04_current_ss30k_slat30k/inference_manifest.json",
            "recon": root / "05_reconviagen/inference_manifest.json",
        }
        payloads = {}
        for key, path in paths.items():
            if not path.is_file():
                raise FileNotFoundError(f"incomplete {branch} artifact: {path}")
            payloads[key] = load_json(path)
            if payloads[key].get("passed") is not True:
                raise RuntimeError(f"failed stage: {branch}/{key}")
        raw_row = _only_object(payloads["raw"], paths["raw"])
        runtime_row = _only_object(payloads["runtime"], paths["runtime"])
        current_row = _only_object(payloads["current"], paths["current"])
        recon_row = _only_object(payloads["recon"], paths["recon"])
        frames = list(planned["selected_frame_names"])
        if raw_row.get("selected_source_frame_names") != frames:
            raise RuntimeError(f"raw frame order differs: {branch}")
        indices = [int(value) for value in runtime_row["selected_source_view_indices"]]
        if indices != list(range(len(frames))):
            raise RuntimeError(f"runtime did not retain all frozen views: {branch}")
        if runtime_row.get("geometry_mode") != "pose_mask":
            raise RuntimeError(f"runtime geometry differs: {branch}")
        validate_runtime_o_mesh_frame_contract(current_row)
        current_mesh = Path(current_row["mesh"]).resolve(strict=True)
        recon_mesh = Path(recon_row["mesh"]).resolve(strict=True)
        deployments = plan["deployments"]
        if current_row.get("native_ss_checkpoint_sha256") != deployments["native_ss_checkpoint"]["sha256"]:
            raise RuntimeError(f"Native-SS identity differs: {branch}")
        if current_row.get("native_slat_checkpoint_sha256") != deployments["native_slat_checkpoint"]["sha256"]:
            raise RuntimeError(f"Native-SLat identity differs: {branch}")
        if int(current_row.get("native_slat_checkpoint_step", -1)) != 30000:
            raise RuntimeError(f"Native-SLat step differs: {branch}")
        if current_row.get("vggt_model_executed") is not False:
            raise RuntimeError(f"current endpoint executed VGGT: {branch}")
        if recon_row.get("method") != "reconviagen_original":
            raise RuntimeError(f"strict ReconViaGen identity differs: {branch}")
        presentation = root / "06_结果汇总"
        current_copy = presentation / "当前_SS30K_SLat30K_runtime-O.obj"
        recon_copy = presentation / "ReconViaGen原版.obj"
        _copy_verified(current_mesh, current_copy)
        _copy_verified(recon_mesh, recon_copy)
        results.append(
            {
                "branch": branch,
                "view_count": len(frames),
                "selected_frame_names": frames,
                "selected_source_indices": list(planned["selected_indices"]),
                "colmap_quality_audit": planned["colmap_quality_audit"],
                "fixed_input_report": _asset(paths["input"]),
                "runtime_input_manifest": _asset(paths["runtime"]),
                "current_inference_manifest": _asset(paths["current"]),
                "reconviagen_inference_manifest": _asset(paths["recon"]),
                "current_mesh": _asset(current_copy),
                "reconviagen_mesh": _asset(recon_copy),
            }
        )
    report = {
        "format": REPORT_FORMAT,
        "created_at_utc": utc_now(),
        "passed": True,
        "branch_count": 2,
        "results": results,
        "selection": plan["selection"],
        "experiment_plan": _asset(plan_path),
        "current_endpoint": plan["current_endpoint"],
        "reconviagen_endpoint": plan["reconviagen_endpoint"],
        "four_view_is_exact_subset_of_sixteen_view": True,
        "contour_rendering_performed": False,
        "source_colmap_recomputed": False,
        "source_dataset_modified": False,
        "interpretation": (
            "Qualitative view-count comparison on one real object. Frames are uniformly "
            "sampled by cumulative COLMAP camera-center trajectory translation; no GT Mesh "
            "or image-content signal participates in selection."
        ),
    }
    atomic_json(output / "report.json", report)
    print(json.dumps({"passed": True, "report": str(output / 'report.json')}, ensure_ascii=False, indent=2))


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "materialize", "finalize"):
        child = sub.add_parser(name)
        child.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main() -> None:
    args = make_parser().parse_args()
    if args.command == "prepare":
        prepare(args.output)
    elif args.command == "materialize":
        materialize(args.output)
    else:
        finalize(args.output)


if __name__ == "__main__":
    main()
