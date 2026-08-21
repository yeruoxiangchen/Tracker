#!/usr/bin/env python3
"""Plan, audit, and summarize the five-capture fixed-O selector matrix.

Each AR/COLMAP pose source constructs one official-compatible model-O from its
full foreground-valid pool.  Three eight-view selectors then share that exact
``T_O2W``: official-training single-seed spherical FPS, temporal uniform, and
quality-constrained multi-candidate spherical FPS.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from typing import Any

import numpy as np

from pose_point_depth_mv.omni_real_benchmark_common import (
    atomic_json,
    load_json,
    sha256_file,
)
from pose_point_depth_mv.real_capture_ss30k_slat30k_pose_mask_batch import (
    ABC_R_EVIDENCE,
    SPECS,
    SS_CHECKPOINT,
    SS_REPORT,
    SLAT_CHECKPOINT,
    STOCK_FREEZE,
    _asset,
)
from pose_point_depth_mv.trellis_mesh_coordinate_contract import (
    MESH_FRAME_CONTRACT,
    validate_runtime_o_mesh_frame_contract,
)


TRACKER_ROOT = Path(__file__).resolve().parents[1]
SOURCE_V1 = (
    TRACKER_ROOT
    / "pose_point_depth_mv/outputs2/"
    "真实采集5组_AR与COLMAPPose_ReconViaGen_vs_SS30K_SLat30K_三选帧轮廓_20260819_v1"
)
SOURCE_V2 = (
    TRACKER_ROOT
    / "pose_point_depth_mv/outputs2/"
    "真实采集5组_AR_COLMAP六分支_SS30K_SLat30K_runtimeO正确轮廓_20260819_v2"
)
DEFAULT_OUTPUT = (
    TRACKER_ROOT
    / "pose_point_depth_mv/outputs2/"
    "真实采集5组_AR_COLMAP三选帧_固定official兼容O_"
    "SS30K_SLat30K轮廓_20260819_v3"
)

PLAN_FORMAT = "pose_point_depth_mv.real_capture_fixed_official_o_selector_plan.v3"
REPORT_FORMAT = "pose_point_depth_mv.real_capture_fixed_official_o_selector_matrix.v3"

BRANCHES = (
    {
        "slug": "01_ar_training_spherical_fps8",
        "pose_source": "ar_pose",
        "selector": "training_spherical_farthest_valid_mask",
        "selector_label": "训练一致：单起点3D视角最远点采样",
    },
    {
        "slug": "02_ar_time_uniform8",
        "pose_source": "ar_pose",
        "selector": "lexical_even_valid_mask_fallback",
        "selector_label": "时间均匀8帧（仅空mask就近替换）",
    },
    {
        "slug": "03_ar_quality_spherical_fps8",
        "pose_source": "ar_pose",
        "selector": "object_spherical_farthest_valid_mask",
        "selector_label": "质量约束 + 3D视角分散",
    },
    {
        "slug": "04_colmap_training_spherical_fps8",
        "pose_source": "colmap_pose",
        "selector": "training_spherical_farthest_valid_mask",
        "selector_label": "训练一致：单起点3D视角最远点采样",
    },
    {
        "slug": "05_colmap_time_uniform8",
        "pose_source": "colmap_pose",
        "selector": "lexical_even_valid_mask_fallback",
        "selector_label": "时间均匀8帧（仅空mask就近替换）",
    },
    {
        "slug": "06_colmap_quality_spherical_fps8",
        "pose_source": "colmap_pose",
        "selector": "object_spherical_farthest_valid_mask",
        "selector_label": "质量约束 + 3D视角分散",
    },
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _raw_report(dataset_name: str, pose_source: str) -> Path:
    old_branch = (
        "01_ar_phone_spherical8" if pose_source == "ar_pose" else "04_colmap_spherical8"
    )
    return (
        SOURCE_V1
        / "objects"
        / dataset_name
        / "branches"
        / old_branch
        / "01_raw_cache/raw_cache_report.json"
    ).resolve(strict=True)


def _single_passed_object(payload: dict[str, Any], path: Path) -> dict[str, Any]:
    rows = list(payload.get("objects", []))
    if payload.get("passed") is not True or len(rows) != 1:
        raise RuntimeError(f"expected one passed object: {path}")
    return dict(rows[0])


def _requested_plan(output: Path) -> dict[str, Any]:
    source_report = load_json(SOURCE_V2 / "report.json")
    if source_report.get("passed") is not True or int(source_report.get("dataset_count", 0)) != 5:
        raise RuntimeError("source v2 five-capture report is not eligible")
    objects = []
    for spec in SPECS:
        name = str(spec["dataset_name"])
        raw = {}
        for pose_source in ("ar_pose", "colmap_pose"):
            path = _raw_report(name, pose_source)
            row = _single_passed_object(load_json(path), path)
            if int(row.get("registered_pair_count", 0)) < 8:
                raise RuntimeError(f"insufficient full candidate pool: {name}/{pose_source}")
            raw[pose_source] = {
                "report": _asset(path),
                "registered_pair_count": int(row["registered_pair_count"]),
                "source_dataset": row.get("source_dataset"),
            }
        objects.append(
            {
                "dataset_name": name,
                "raw_sources": raw,
                "branch_root": str((output / "objects" / name / "branches").resolve()),
            }
        )
    return {
        "format": PLAN_FORMAT,
        "created_at_utc": utc_now(),
        "passed": True,
        "dataset_count": len(objects),
        "branch_count_per_dataset": len(BRANCHES),
        "total_branch_count": len(objects) * len(BRANCHES),
        "selected_view_count": 8,
        "geometry_mode": "pose_mask",
        "object_frame_view_scope": "all_foreground_valid",
        "model_o_axis_convention": "official_z_up",
        "fixed_o_contract": (
            "For each dataset and pose source, all foreground-valid candidate views "
            "construct T_O2W once; the three final-eight selectors rebind only T_W2C."
        ),
        "source_v2_report": _asset(SOURCE_V2 / "report.json"),
        "deployments": {
            "native_ss_report": _asset(SS_REPORT),
            "native_ss_checkpoint": _asset(SS_CHECKPOINT),
            "native_slat_checkpoint": _asset(SLAT_CHECKPOINT),
            "abc_r_evidence": _asset(ABC_R_EVIDENCE),
            "stock_slat_freeze": _asset(STOCK_FREEZE),
        },
        "objects": objects,
        "branches": list(BRANCHES),
        "target_or_metric_consumed": False,
        "scope_guard": (
            "Real-capture qualitative selector/O-coordinate audit. No GT Mesh, target "
            "latent, metric, ICP, or post-hoc ranking is consumed."
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
                raise RuntimeError(f"existing v3 plan differs: field={key}")
        reused = True
    else:
        if output.exists() and any(output.iterdir()):
            raise RuntimeError(f"unbound nonempty v3 output: {output}")
        output.mkdir(parents=True, exist_ok=True)
        atomic_json(path, requested)
        reused = False
    print(json.dumps({"passed": True, "reused": reused, "plan": str(path)}, indent=2))


def _copy_verified(source: Path, destination: Path) -> dict[str, Any]:
    source = source.expanduser().resolve(strict=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        if sha256_file(destination) != sha256_file(source):
            raise RuntimeError(f"existing presentation Mesh differs: {destination}")
    else:
        shutil.copy2(source, destination)
    return _asset(destination)


def finalize(output: Path) -> None:
    output = output.expanduser().resolve(strict=True)
    plan_path = output / "experiment_plan.json"
    plan = load_json(plan_path)
    if plan.get("format") != PLAN_FORMAT or plan.get("passed") is not True:
        raise RuntimeError("v3 experiment plan is not eligible")

    records = []
    fixed_o_groups = []
    branch_by_slug = {row["slug"]: row for row in BRANCHES}
    for planned in plan["objects"]:
        name = str(planned["dataset_name"])
        by_pose: dict[str, list[tuple[str, np.ndarray, str]]] = {
            "ar_pose": [],
            "colmap_pose": [],
        }
        for slug, branch in branch_by_slug.items():
            root = output / "objects" / name / "branches" / slug
            paths = {
                "runtime": root / "01_runtime_o/runtime_input_manifest.json",
                "model": root / "02_dino_only_input/model_input_manifest.json",
                "current": root / "03_current_ss30k_slat30k/inference_manifest.json",
                "contour": root / "04_current_camera_contours/report.json",
            }
            payloads = {key: load_json(path) for key, path in paths.items()}
            runtime_row = _single_passed_object(payloads["runtime"], paths["runtime"])
            current_row = _single_passed_object(payloads["current"], paths["current"])
            if payloads["model"].get("passed") is not True or payloads["contour"].get("passed") is not True:
                raise RuntimeError(f"incomplete branch: {name}/{slug}")
            if runtime_row.get("object_frame_view_scope") != "all_foreground_valid":
                raise RuntimeError(f"branch did not freeze all-view O: {name}/{slug}")
            if runtime_row.get("model_o_axis_convention") != "official_z_up":
                raise RuntimeError(f"branch did not use official-compatible O: {name}/{slug}")
            with np.load(runtime_row["cache_npz"], allow_pickle=False) as cache:
                T_O2W = np.asarray(cache["T_O2W"], dtype=np.float64)
            by_pose[branch["pose_source"]].append(
                (slug, T_O2W, str(runtime_row["T_O2W_sha256"]))
            )
            result_path = Path(current_row["result"]).resolve(strict=True)
            result = load_json(result_path)
            validate_runtime_o_mesh_frame_contract(result)
            mesh_asset = _copy_verified(
                Path(current_row["mesh"]),
                root / "05_presentation/当前SS30K_SLat30K_official兼容固定O.obj",
            )
            raw_path = Path(planned["raw_sources"][branch["pose_source"]]["report"]["path"])
            raw_row = _single_passed_object(load_json(raw_path), raw_path)
            source_by_runtime = {
                str(camera["frame_name"]): str(camera.get("source_frame_name", camera["frame_name"]))
                for camera in raw_row.get("cameras", [])
            }
            records.append(
                {
                    "dataset_name": name,
                    "slug": slug,
                    "pose_source": branch["pose_source"],
                    "selector": branch["selector"],
                    "selector_label": branch["selector_label"],
                    "run_dir": str(root),
                    "selected_runtime_frame_names": list(runtime_row["selected_frame_names"]),
                    "selected_source_frame_names": [
                        source_by_runtime.get(value, value)
                        for value in runtime_row["selected_frame_names"]
                    ],
                    "view_selection": runtime_row["view_selection"],
                    "object_frame_construction": runtime_row["object_frame_construction"],
                    "T_O2W_sha256": runtime_row["T_O2W_sha256"],
                    "current_mesh": mesh_asset,
                    "contour_overview": _asset(Path(payloads["contour"]["overview"])),
                    "runtime_manifest": _asset(paths["runtime"]),
                    "model_manifest": _asset(paths["model"]),
                    "inference_manifest": _asset(paths["current"]),
                    "contour_report": _asset(paths["contour"]),
                }
            )
        for pose_source, rows in by_pose.items():
            if len(rows) != 3:
                raise RuntimeError(f"selector group incomplete: {name}/{pose_source}")
            reference = rows[0][1]
            same = all(np.array_equal(reference, row[1]) for row in rows[1:])
            same_hash = len({row[2] for row in rows}) == 1
            if not same or not same_hash:
                raise RuntimeError(f"T_O2W differs across selectors: {name}/{pose_source}")
            fixed_o_groups.append(
                {
                    "dataset_name": name,
                    "pose_source": pose_source,
                    "same_T_O2W_across_three_selectors": True,
                    "T_O2W_sha256": rows[0][2],
                    "branches": [row[0] for row in rows],
                }
            )

    report = {
        "format": REPORT_FORMAT,
        "created_at_utc": utc_now(),
        "passed": len(records) == 30 and len(fixed_o_groups) == 10,
        "dataset_count": 5,
        "branch_count": len(records),
        "fixed_o_group_count": len(fixed_o_groups),
        "all_selector_groups_share_exact_T_O2W": all(
            row["same_T_O2W_across_three_selectors"] for row in fixed_o_groups
        ),
        "object_frame_view_scope": "all_foreground_valid",
        "model_o_axis_convention": "official_z_up",
        "mesh_frame_contract": MESH_FRAME_CONTRACT,
        "plan": _asset(plan_path),
        "fixed_o_groups": fixed_o_groups,
        "records": records,
        "target_or_metric_consumed": False,
        "formal_claim_allowed": False,
        "scope_guard": plan["scope_guard"],
    }
    atomic_json(output / "report.json", report)
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "branches": report["branch_count"],
                "fixed_o_groups": report["fixed_o_group_count"],
                "report": str(output / "report.json"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def status(output: Path) -> None:
    output = output.expanduser().resolve()
    complete = {"runtime": 0, "model": 0, "inference": 0, "contour": 0}
    for spec in SPECS:
        name = str(spec["dataset_name"])
        for branch in BRANCHES:
            root = output / "objects" / name / "branches" / branch["slug"]
            checks = {
                "runtime": root / "01_runtime_o/runtime_input_manifest.json",
                "model": root / "02_dino_only_input/model_input_manifest.json",
                "inference": root / "03_current_ss30k_slat30k/inference_manifest.json",
                "contour": root / "04_current_camera_contours/report.json",
            }
            for key, path in checks.items():
                if path.is_file() and load_json(path).get("passed") is True:
                    complete[key] += 1
    print(
        json.dumps(
            {
                "expected_branches": 30,
                "completed": complete,
                "final_report_complete": (output / "report.json").is_file(),
                "output": str(output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "finalize", "status"):
        child = subparsers.add_parser(command)
        child.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    output = Path(args.output)
    if args.command == "prepare":
        prepare(output)
    elif args.command == "finalize":
        finalize(output)
    else:
        status(output)


if __name__ == "__main__":
    main()
