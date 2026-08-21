#!/usr/bin/env python3
"""Build and audit the corrected six-branch real-capture pose/view matrix.

The historical v1 run evaluated only one AR-pose view-selection policy and
exported current-endpoint Meshes with TRELLIS' presentation rotation enabled.
This v2 run is deliberately independent: it adds AR-pose time/random branches,
repairs the four already-computed branches per object by an exact hash-bound
inverse rotation, and renders every contour from a Mesh proven to be in native
runtime-O axes.  The v1 tree is read-only provenance and is never modified.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from typing import Any, Sequence

import numpy as np

from pose_point_depth_mv.dataset_tools.prepare_omni_real_video_cache import (
    parse_registered_images,
)
from pose_point_depth_mv.omni_real_benchmark_common import (
    atomic_json,
    load_json,
    sha256_file,
)
from pose_point_depth_mv.real_capture_ss30k_slat30k_pose_mask_batch import (
    ABC_R_EVIDENCE,
    DATASET_ROOT,
    SPECS,
    SS_CHECKPOINT,
    SS_REPORT,
    SLAT_CHECKPOINT,
    STOCK_FREEZE,
    _asset,
    _materialize_dataset,
    _source_frames,
)
from pose_point_depth_mv.reframe_omni_official_with_vggt_mesh import (
    reframe_manifest,
)
from pose_point_depth_mv.run_ar_offline_colmap_ab import (
    _image_path,
    _mask_path,
)
from pose_point_depth_mv.trellis_mesh_coordinate_contract import (
    MESH_FRAME_CONTRACT,
    validate_runtime_o_mesh_frame_contract,
)


TRACKER_ROOT = Path(__file__).resolve().parents[1]
OLD_OUTPUT = (
    TRACKER_ROOT
    / "pose_point_depth_mv/outputs2/"
    "真实采集5组_AR与COLMAPPose_ReconViaGen_vs_SS30K_SLat30K_三选帧轮廓_20260819_v1"
)
DEFAULT_OUTPUT = (
    TRACKER_ROOT
    / "pose_point_depth_mv/outputs2/"
    "真实采集5组_AR_COLMAP六分支_SS30K_SLat30K_runtimeO正确轮廓_20260819_v2"
)

PLAN_FORMAT = "pose_point_depth_mv.real_capture_ar_colmap_six_branch_plan.v2"
AR_SELECTION_FORMAT = "pose_point_depth_mv.real_capture_ar_pose_selection.v2"
LINEAGE_FORMAT = "pose_point_depth_mv.real_capture_corrected_branch_lineage.v2"
REPORT_FORMAT = "pose_point_depth_mv.real_capture_ar_colmap_six_branch_batch.v2"
SELECTED_VIEWS = 8
RANDOM_SEED = 20260819

MIGRATED_BRANCHES = {
    "03_ar_spherical8": "01_ar_phone_spherical8",
    "04_colmap_time_uniform8": "02_colmap_time_uniform8",
    "05_colmap_random8_seed20260819": "03_colmap_random8_seed20260819",
    "06_colmap_spherical8": "04_colmap_spherical8",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _copy_verified(source: Path, destination: Path) -> dict[str, Any]:
    source = source.expanduser().resolve(strict=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        if sha256_file(destination) != sha256_file(source):
            raise RuntimeError(f"existing copied artifact differs: {destination}")
    else:
        shutil.copy2(source, destination)
    return _asset(destination)


def _old_identity() -> tuple[Path, dict[str, Any], Path, dict[str, Any]]:
    old_plan_path = OLD_OUTPUT / "experiment_plan.json"
    old_report_path = OLD_OUTPUT / "report.json"
    old_plan = load_json(old_plan_path)
    old_report = load_json(old_report_path)
    if (
        old_plan.get("format")
        != "pose_point_depth_mv.real_capture_ar_colmap_ss30k_slat30k_plan.v1"
        or old_plan.get("passed") is not True
        or int(old_plan.get("total_branch_count", -1)) != 20
    ):
        raise RuntimeError("historical four-branch experiment plan differs")
    if (
        old_report.get("passed") is not True
        or int(old_report.get("branch_count", -1)) != 20
        or old_report.get("mesh_frame_interpretation_valid") is not False
        or old_report.get("requires_mesh_frame_v2_rerender") is not True
    ):
        raise RuntimeError("historical invalid-Mesh evidence differs")
    return old_plan_path, old_plan, old_report_path, old_report


def _old_object(old_plan: dict[str, Any], dataset_name: str) -> dict[str, Any]:
    rows = [
        dict(row)
        for row in old_plan.get("objects", [])
        if str(row.get("dataset_name")) == str(dataset_name)
    ]
    if len(rows) != 1:
        raise RuntimeError(f"historical object identity is not unique: {dataset_name}")
    return rows[0]


def _requested_plan(output: Path) -> dict[str, Any]:
    old_plan_path, old_plan, old_report_path, _old_report = _old_identity()
    objects = []
    for spec in SPECS:
        name = str(spec["dataset_name"])
        old = _old_object(old_plan, name)
        dataset = Path(old["dataset"]).resolve(strict=True)
        capture_path, names = _source_frames(dataset)
        if names != list(old["source_frame_names"]):
            raise RuntimeError(f"source timeline changed: {name}")
        objects.append(
            {
                "dataset_name": name,
                "dataset": str(dataset),
                "source_frame_names": names,
                "capture_report": _asset(capture_path),
                "frame_metadata": _asset(dataset / "frame_metadata.jsonl"),
                "phone_pose_meta": _asset(dataset / "sparse/0/phone_pose_meta.json"),
                "phone_sparse_files": {
                    filename: _asset(dataset / "sparse/0" / filename)
                    for filename in ("cameras.txt", "images.txt", "points3D.txt")
                },
                "historical_colmap_report": _asset(Path(old["colmap_report"])),
                "ar_selection_report": str(
                    (output / "objects" / name / "00_ar_pose/selection_report.json").resolve()
                ),
                "branch_root": str((output / "objects" / name / "branches").resolve()),
                "historical_branch_root": str(Path(old["branch_root"]).resolve(strict=True)),
            }
        )
    return {
        "format": PLAN_FORMAT,
        "created_at_utc": utc_now(),
        "passed": True,
        "dataset_count": len(objects),
        "branch_count_per_dataset": 6,
        "total_branch_count": len(objects) * 6,
        "selected_view_count": SELECTED_VIEWS,
        "random_seed": RANDOM_SEED,
        "geometry_mode": "pose_mask",
        "historical_v1_plan": _asset(old_plan_path),
        "historical_v1_report": _asset(old_report_path),
        "historical_v1_tree_modified": False,
        "deployments": {
            "native_ss_report": _asset(SS_REPORT),
            "native_ss_checkpoint": _asset(SS_CHECKPOINT),
            "native_ss_checkpoint_step": 30000,
            "native_slat_checkpoint": _asset(SLAT_CHECKPOINT),
            "native_slat_checkpoint_step": 30000,
            "abc_r_evidence": _asset(ABC_R_EVIDENCE),
            "stock_slat_freeze": _asset(STOCK_FREEZE),
        },
        "objects": objects,
        "branch_contract": [
            {
                "slug": "01_ar_time_uniform8",
                "pose_source": "captured_phone_unity_ar_pose",
                "selection": "timeline_uniform8",
                "execution": "fresh",
            },
            {
                "slug": "02_ar_random8_seed20260819",
                "pose_source": "captured_phone_unity_ar_pose",
                "selection": "fixed_seed_random8",
                "execution": "fresh",
            },
            {
                "slug": "03_ar_spherical8",
                "pose_source": "captured_phone_unity_ar_pose",
                "selection": "object_spherical_farthest_valid_mask",
                "execution": "hash_bound_v1_mesh_axis_repair",
            },
            {
                "slug": "04_colmap_time_uniform8",
                "pose_source": "all_frame_offline_colmap_sfm_ba",
                "selection": "timeline_uniform8",
                "execution": "hash_bound_v1_mesh_axis_repair",
            },
            {
                "slug": "05_colmap_random8_seed20260819",
                "pose_source": "all_frame_offline_colmap_sfm_ba",
                "selection": "fixed_seed_random8",
                "execution": "hash_bound_v1_mesh_axis_repair",
            },
            {
                "slug": "06_colmap_spherical8",
                "pose_source": "all_frame_offline_colmap_sfm_ba",
                "selection": "object_spherical_farthest_valid_mask",
                "execution": "hash_bound_v1_mesh_axis_repair",
            },
        ],
        "mesh_frame_contract": MESH_FRAME_CONTRACT,
        "scope_guard": (
            "Six symmetric qualitative branches: AR pose and independent COLMAP pose each "
            "use timeline, fixed-random, and spherical-farthest eight-view selection. AR "
            "and COLMAP time/random branches are frame-for-frame identical so that their "
            "only experimental difference is the camera pose solution. "
            "Historical Mesh geometry is repaired only by the exact inverse of the erroneous "
            "v1 presentation rotation; no ICP, fitting, target Mesh, or metric is consumed."
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
                raise RuntimeError(f"existing v2 plan differs: field={key}")
        reused = True
    else:
        if output.exists() and any(output.iterdir()):
            raise RuntimeError(f"unbound nonempty v2 output: {output}")
        output.mkdir(parents=True, exist_ok=True)
        atomic_json(path, requested)
        reused = False
    print(json.dumps({"passed": True, "reused": reused, "plan": str(path)}, indent=2))


def _planned_row(output: Path, dataset_name: str) -> tuple[dict[str, Any], dict[str, Any]]:
    plan = load_json(output / "experiment_plan.json")
    if plan.get("format") != PLAN_FORMAT or plan.get("passed") is not True:
        raise RuntimeError("v2 experiment plan is not eligible")
    rows = [row for row in plan["objects"] if row["dataset_name"] == dataset_name]
    if len(rows) != 1:
        raise RuntimeError(f"dataset is not uniquely planned: {dataset_name}")
    return plan, dict(rows[0])


def prepare_ar(output: Path, dataset_name: str) -> None:
    output = output.expanduser().resolve(strict=True)
    _plan, row = _planned_row(output, dataset_name)
    dataset = Path(row["dataset"]).resolve(strict=True)
    _capture, timeline = _source_frames(dataset)
    registered_rows = parse_registered_images(dataset / "sparse/0/images.txt")
    registered_names = {str(item["name"]) for item in registered_rows}
    eligible = [
        name
        for name in timeline
        if name in registered_names
        and _image_path(dataset, name).is_file()
        and _mask_path(dataset, name).is_file()
    ]
    if eligible != timeline:
        raise RuntimeError(
            f"AR pose timeline is not fully registered/mask-backed: {dataset_name} "
            f"eligible={len(eligible)} timeline={len(timeline)}"
        )
    # Matched-view contract: do not independently select AR frames.  Consume
    # the exact frozen RGB/mask identities already used by the corresponding
    # COLMAP time/random branches, and change only the camera pose source.
    colmap_path = Path(row["historical_colmap_report"]["path"])
    colmap = load_json(colmap_path)
    if colmap.get("passed") is not True:
        raise RuntimeError(f"historical COLMAP selection is incomplete: {colmap_path}")
    time_frames = [str(value) for value in colmap["selections"]["time_uniform8"]["selected_frame_names"]]
    random_frames = [str(value) for value in colmap["selections"]["random8"]["selected_frame_names"]]
    if len(time_frames) != SELECTED_VIEWS or len(random_frames) != SELECTED_VIEWS:
        raise RuntimeError("matched COLMAP selection cardinality differs")
    if not set(time_frames).issubset(set(eligible)) or not set(random_frames).issubset(set(eligible)):
        raise RuntimeError("COLMAP time/random selection lacks an original AR camera")
    indices = [eligible.index(name) for name in time_frames]
    random_indices = [eligible.index(name) for name in random_frames]
    seed = int(colmap["selections"]["random8"]["seed"])
    draw_names = [str(value) for value in colmap["selections"]["random8"]["draw_order_frame_names"]]
    draw = [eligible.index(name) for name in draw_names]

    root = output / "objects" / dataset_name / "00_ar_pose"
    prepared = root / "prepared_datasets"
    time_dataset = prepared / "time_uniform8"
    random_dataset = prepared / f"random8_seed{RANDOM_SEED}"
    for frames, destination in (
        (time_frames, time_dataset),
        (random_frames, random_dataset),
    ):
        _materialize_dataset(dataset, frames, dataset / "sparse/0", destination)
        _copy_verified(
            dataset / "sparse/0/phone_pose_meta.json",
            destination / "sparse/0/phone_pose_meta.json",
        )

    payload = {
        "format": AR_SELECTION_FORMAT,
        "created_at_utc": utc_now(),
        "passed": True,
        "dataset_name": dataset_name,
        "source_dataset": str(dataset),
        "source_capture_report": row["capture_report"],
        "source_frame_metadata": row["frame_metadata"],
        "source_phone_pose_meta": row["phone_pose_meta"],
        "source_phone_sparse_files": row["phone_sparse_files"],
        "matched_colmap_selection_report": _asset(colmap_path),
        "eligible_frame_count": len(eligible),
        "eligible_frame_names_in_timeline_order": eligible,
        "selections": {
            "time_uniform8": {
                "policy": "exact_frame_match_to_frozen_colmap_time_uniform8",
                "selected_indices": indices,
                "selected_frame_names": time_frames,
                "dataset": str(time_dataset.resolve()),
                "matched_colmap_selected_frame_names": time_frames,
            },
            "random8": {
                "policy": "exact_frame_match_to_frozen_colmap_fixed_random8",
                "seed": seed,
                "base_seed": RANDOM_SEED,
                "draw_order_indices": draw,
                "draw_order_frame_names": draw_names,
                "selected_indices_in_timeline_order": random_indices,
                "selected_frame_names": random_frames,
                "dataset": str(random_dataset.resolve()),
                "matched_colmap_selected_frame_names": random_frames,
            },
        },
        "camera_source": "original captured phone/Unity AR cameras",
        "camera_gauge": "phone_y_up",
        "gravity_up_W": [0.0, 1.0, 0.0],
        "scope_guard": (
            "RGB/mask frame identities exactly match the frozen COLMAP time/random branches. "
            "Only intrinsics/extrinsics change to the byte-identical original phone cameras; "
            "phone axis metadata remains frozen."
        ),
    }
    path = Path(row["ar_selection_report"])
    if path.is_file():
        existing = load_json(path)
        for key, value in payload.items():
            if key != "created_at_utc" and existing.get(key) != value:
                raise RuntimeError(f"existing AR selection differs: field={key}")
    else:
        atomic_json(path, payload)
    print(
        json.dumps(
            {
                "passed": True,
                "dataset": dataset_name,
                "time_uniform8": time_frames,
                "random8": random_frames,
                "report": str(path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def migrate_legacy(output: Path, dataset_name: str, slug: str) -> None:
    output = output.expanduser().resolve(strict=True)
    _plan, row = _planned_row(output, dataset_name)
    if slug not in MIGRATED_BRANCHES:
        raise ValueError(f"branch is not migration-backed: {slug}")
    source_slug = MIGRATED_BRANCHES[slug]
    source_root = Path(row["historical_branch_root"]) / source_slug
    source_manifest = source_root / "04_current_ss30k_slat30k/inference_manifest.json"
    source = load_json(source_manifest)
    if source.get("passed") is not True or len(source.get("objects", [])) != 1:
        raise RuntimeError(f"historical current inference is incomplete: {source_manifest}")
    source_mesh = Path(source["objects"][0]["mesh"]).resolve(strict=True)
    expected_mesh_sha = sha256_file(source_mesh)

    root = Path(row["branch_root"]) / slug
    corrected = root / "04_current_ss30k_slat30k"
    corrected_manifest = corrected / "inference_manifest.json"
    if corrected_manifest.is_file():
        manifest = load_json(corrected_manifest)
        if manifest.get("legacy_source_manifest_sha256") != sha256_file(source_manifest):
            raise RuntimeError(f"existing corrected branch source differs: {root}")
        if len(manifest.get("objects", [])) != 1:
            raise RuntimeError(f"existing corrected branch matrix differs: {root}")
        validate_runtime_o_mesh_frame_contract(manifest)
        validate_runtime_o_mesh_frame_contract(manifest["objects"][0])
    else:
        root.mkdir(parents=True, exist_ok=True)
        reframe_manifest(
            source_manifest_path=source_manifest,
            output_dir=corrected,
            expected_source_mesh_sha256=expected_mesh_sha,
        )

    lineage = {
        "format": LINEAGE_FORMAT,
        "created_at_utc": utc_now(),
        "passed": True,
        "dataset_name": dataset_name,
        "slug": slug,
        "historical_slug": source_slug,
        "historical_branch": str(source_root.resolve(strict=True)),
        "historical_runtime_manifest": _asset(
            source_root / "02_runtime_o/runtime_input_manifest.json"
        ),
        "historical_model_input_manifest": _asset(
            source_root / "03_dino_only_input/model_input_manifest.json"
        ),
        "historical_current_manifest": _asset(source_manifest),
        "historical_reconviagen_manifest": _asset(
            source_root / "05_reconviagen/inference_manifest.json"
        ),
        "historical_mesh_sha256": expected_mesh_sha,
        "corrected_current_manifest": _asset(corrected_manifest),
        "correction_report": _asset(corrected / "coordinate_correction_report.json"),
        "repair": "inverse of v1 (x,y,z)->(x,z,-y); topology and shape unchanged",
        "target_or_metric_consumed": False,
    }
    lineage_path = root / "legacy_v1_reuse_and_axis_repair.json"
    if lineage_path.is_file():
        existing = load_json(lineage_path)
        for key, value in lineage.items():
            if key != "created_at_utc" and existing.get(key) != value:
                raise RuntimeError(f"existing migration lineage differs: field={key}")
    else:
        atomic_json(lineage_path, lineage)
    print(
        json.dumps(
            {
                "passed": True,
                "dataset": dataset_name,
                "slug": slug,
                "corrected_manifest": str(corrected_manifest),
                "lineage": str(lineage_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _branch_specs(row: dict[str, Any]) -> list[dict[str, Any]]:
    ar = load_json(Path(row["ar_selection_report"]))
    colmap = load_json(Path(row["historical_colmap_report"]["path"]))
    if ar.get("format") != AR_SELECTION_FORMAT or ar.get("passed") is not True:
        raise RuntimeError("AR selection report is incomplete")
    if colmap.get("passed") is not True:
        raise RuntimeError("historical COLMAP selection report is incomplete")
    return [
        {
            "slug": "01_ar_time_uniform8",
            "dataset": ar["selections"]["time_uniform8"]["dataset"],
            "pose_source": "captured_phone_unity_ar_pose",
            "runtime_policy": "lexical_even",
            "expected_selected": ar["selections"]["time_uniform8"]["selected_frame_names"],
            "eligible_pool": ar["eligible_frame_names_in_timeline_order"],
            "migrated": False,
        },
        {
            "slug": "02_ar_random8_seed20260819",
            "dataset": ar["selections"]["random8"]["dataset"],
            "pose_source": "captured_phone_unity_ar_pose",
            "runtime_policy": "lexical_even",
            "expected_selected": ar["selections"]["random8"]["selected_frame_names"],
            "eligible_pool": ar["eligible_frame_names_in_timeline_order"],
            "migrated": False,
        },
        {
            "slug": "03_ar_spherical8",
            "dataset": row["dataset"],
            "pose_source": "captured_phone_unity_ar_pose",
            "runtime_policy": "object_spherical_farthest_valid_mask",
            "expected_selected": None,
            "eligible_pool": row["source_frame_names"],
            "migrated": True,
        },
        {
            "slug": "04_colmap_time_uniform8",
            "dataset": colmap["selections"]["time_uniform8"]["dataset"],
            "pose_source": "all_frame_offline_colmap_sfm_ba",
            "runtime_policy": "lexical_even",
            "expected_selected": colmap["selections"]["time_uniform8"]["selected_frame_names"],
            "eligible_pool": colmap["registered_frame_names_in_timeline_order"],
            "migrated": True,
        },
        {
            "slug": "05_colmap_random8_seed20260819",
            "dataset": colmap["selections"]["random8"]["dataset"],
            "pose_source": "all_frame_offline_colmap_sfm_ba",
            "runtime_policy": "lexical_even",
            "expected_selected": colmap["selections"]["random8"]["selected_frame_names"],
            "eligible_pool": colmap["registered_frame_names_in_timeline_order"],
            "migrated": True,
        },
        {
            "slug": "06_colmap_spherical8",
            "dataset": colmap["selections"]["spherical8"]["dataset"],
            "pose_source": "all_frame_offline_colmap_sfm_ba",
            "runtime_policy": "object_spherical_farthest_valid_mask",
            "expected_selected": None,
            "eligible_pool": colmap["registered_frame_names_in_timeline_order"],
            "migrated": True,
        },
    ]


def _stage_paths(row: dict[str, Any], branch: dict[str, Any]) -> dict[str, Path]:
    root = Path(row["branch_root"]) / branch["slug"]
    if branch["migrated"]:
        lineage_path = root / "legacy_v1_reuse_and_axis_repair.json"
        lineage = load_json(lineage_path)
        if lineage.get("format") != LINEAGE_FORMAT or lineage.get("passed") is not True:
            raise RuntimeError(f"migration lineage is incomplete: {lineage_path}")
        source = Path(lineage["historical_branch"])
        raw = source / "01_raw_cache/raw_cache_report.json"
        runtime = Path(lineage["historical_runtime_manifest"]["path"])
        model = Path(lineage["historical_model_input_manifest"]["path"])
        recon = Path(lineage["historical_reconviagen_manifest"]["path"])
    else:
        lineage_path = Path()
        raw = root / "01_raw_cache/raw_cache_report.json"
        runtime = root / "02_runtime_o/runtime_input_manifest.json"
        model = root / "03_dino_only_input/model_input_manifest.json"
        recon = root / "05_reconviagen/inference_manifest.json"
    return {
        "root": root,
        "raw": raw,
        "runtime": runtime,
        "model": model,
        "current": root / "04_current_ss30k_slat30k/inference_manifest.json",
        "recon": recon,
        "contour": root / "06_current_camera_contours/report.json",
        "lineage": lineage_path,
    }


def _selected_source_frames(raw_row: dict[str, Any], runtime_row: dict[str, Any]) -> list[str]:
    with np.load(Path(raw_row["cache_npz"]), allow_pickle=False) as payload:
        if "source_frame_name" not in payload.files:
            raise RuntimeError("raw cache lacks physical source-frame identity")
        names = [str(value) for value in payload["source_frame_name"].tolist()]
    indices = [int(value) for value in runtime_row["selected_source_view_indices"]]
    if len(indices) != SELECTED_VIEWS or any(index < 0 or index >= len(names) for index in indices):
        raise RuntimeError("runtime selected source-view indices differ")
    return [names[index] for index in indices]


def finalize(output: Path) -> None:
    output = output.expanduser().resolve(strict=True)
    plan_path = output / "experiment_plan.json"
    plan = load_json(plan_path)
    if plan.get("format") != PLAN_FORMAT or plan.get("passed") is not True:
        raise RuntimeError("v2 plan did not pass")
    records: list[dict[str, Any]] = []
    for row in plan["objects"]:
        colmap = load_json(Path(row["historical_colmap_report"]["path"]))
        for branch in _branch_specs(row):
            paths = _stage_paths(row, branch)
            for name in ("raw", "runtime", "model", "current", "recon", "contour"):
                if not paths[name].is_file():
                    raise FileNotFoundError(paths[name])
            raw = load_json(paths["raw"])
            runtime = load_json(paths["runtime"])
            model = load_json(paths["model"])
            current = load_json(paths["current"])
            recon = load_json(paths["recon"])
            contour = load_json(paths["contour"])
            if not all(item.get("passed") is True for item in (raw, runtime, model, current, recon, contour)):
                raise RuntimeError(f"one stage did not pass: {paths['root']}")
            if len(raw.get("objects", [])) != 1 or len(runtime.get("objects", [])) != 1:
                raise RuntimeError(f"raw/runtime object matrix differs: {paths['root']}")
            raw_row = raw["objects"][0]
            runtime_row = runtime["objects"][0]
            if Path(raw_row["source_dataset"]).resolve() != Path(branch["dataset"]).resolve():
                raise RuntimeError(f"branch dataset binding differs: {paths['root']}")
            sparse = Path(raw_row["authoritative_colmap_dir"]).resolve(strict=True)
            if branch["pose_source"].startswith("captured_phone"):
                if sha256_file(sparse / "phone_pose_meta.json") != row["phone_pose_meta"]["sha256"]:
                    raise RuntimeError(f"phone pose metadata differs: {paths['root']}")
                for filename, asset in row["phone_sparse_files"].items():
                    if sha256_file(sparse / filename) != asset["sha256"]:
                        raise RuntimeError(f"phone sparse camera model differs: {paths['root']}/{filename}")
            else:
                for filename, asset in colmap["text_model_files"].items():
                    if sha256_file(sparse / filename) != asset["sha256"]:
                        raise RuntimeError(f"COLMAP sparse model differs: {paths['root']}/{filename}")
            selected = _selected_source_frames(raw_row, runtime_row)
            if len(selected) != SELECTED_VIEWS or len(set(selected)) != SELECTED_VIEWS:
                raise RuntimeError(f"selected frame matrix differs: {paths['root']}")
            if not set(selected).issubset(set(branch["eligible_pool"])):
                raise RuntimeError(f"selected frame left eligible pool: {paths['root']}")
            if branch["expected_selected"] is not None and selected != branch["expected_selected"]:
                raise RuntimeError(f"frozen time/random selection differs: {paths['root']}")
            policy = str((runtime_row.get("view_selection") or {}).get("policy"))
            if branch["runtime_policy"] == "lexical_even":
                expected_policy = "lexical_frame_order_evenly_spaced_including_endpoints"
            else:
                expected_policy = "pose_mask_segmented_object_spherical_farthest_valid_mask"
            if policy != expected_policy:
                raise RuntimeError(f"runtime view policy differs: {paths['root']} {policy}")
            if model.get("vggt_model_loaded") is True or model.get("vggt_model_executed") is True:
                raise RuntimeError(f"DINO-only input unexpectedly executed VGGT: {paths['root']}")
            if len(current.get("objects", [])) != 1 or len(recon.get("objects", [])) != 1:
                raise RuntimeError(f"inference object matrix differs: {paths['root']}")
            current_row = current["objects"][0]
            recon_row = recon["objects"][0]
            # The inference manifest is only an object index.  Mesh-frame
            # provenance is stored on each object/result record, so validate
            # the single bound record below rather than requiring the outer
            # manifest to duplicate per-Mesh coordinate metadata.
            validate_runtime_o_mesh_frame_contract(current_row)
            current_mesh = Path(current_row["mesh"]).resolve(strict=True)
            recon_mesh = Path(recon_row["mesh"]).resolve(strict=True)
            if current_row.get("native_ss_checkpoint_sha256") != plan["deployments"]["native_ss_checkpoint"]["sha256"]:
                raise RuntimeError(f"SS30K identity differs: {paths['root']}")
            if current_row.get("native_slat_checkpoint_sha256") != plan["deployments"]["native_slat_checkpoint"]["sha256"]:
                raise RuntimeError(f"SLat30K identity differs: {paths['root']}")
            if int(current_row.get("native_slat_checkpoint_step", -1)) != 30000:
                raise RuntimeError(f"SLat checkpoint step differs: {paths['root']}")
            if current_row.get("vggt_model_executed") is not False:
                raise RuntimeError(f"current endpoint unexpectedly executed VGGT: {paths['root']}")
            if recon_row.get("method") != "reconviagen_original":
                raise RuntimeError(f"reference endpoint differs: {paths['root']}")
            if contour.get("mesh_o_sha256") != sha256_file(current_mesh):
                raise RuntimeError(f"contour/current Mesh hash differs: {paths['root']}")
            if contour.get("runtime_input_manifest_sha256") != sha256_file(paths["runtime"]):
                raise RuntimeError(f"contour/runtime hash differs: {paths['root']}")
            if contour.get("mesh_frame_contract") != MESH_FRAME_CONTRACT:
                raise RuntimeError(f"contour did not use corrected Mesh axes: {paths['root']}")
            presentation = paths["root"] / "07_presentation_meshes"
            current_copy = _copy_verified(
                current_mesh,
                presentation / "当前SS30K_SLat30K_runtime-O正确坐标.obj",
            )
            recon_copy = _copy_verified(
                recon_mesh,
                presentation / "ReconViaGen原版_reference-O.obj",
            )
            records.append(
                {
                    "dataset_name": row["dataset_name"],
                    "branch": branch,
                    "run_dir": str(paths["root"].resolve()),
                    "selected_source_frame_names": selected,
                    "runtime_selection": runtime_row.get("view_selection"),
                    "current_mesh": current_copy,
                    "reconviagen_mesh": recon_copy,
                    "current_inference_manifest": _asset(paths["current"]),
                    "reconviagen_inference_manifest": _asset(paths["recon"]),
                    "corrected_contour_report": _asset(paths["contour"]),
                    "corrected_contour_overview": _asset(Path(contour["overview"])),
                    "legacy_axis_repair": bool(branch["migrated"]),
                    "legacy_lineage": _asset(paths["lineage"]) if branch["migrated"] else None,
                }
            )
    expected = int(plan["total_branch_count"])
    report = {
        "format": REPORT_FORMAT,
        "created_at_utc": utc_now(),
        "passed": len(records) == expected,
        "dataset_count": int(plan["dataset_count"]),
        "branch_count": len(records),
        "expected_branch_count": expected,
        "complete_symmetric_matrix": len(records) == 30,
        "ar_pose_branch_count": sum(
            1 for record in records if record["branch"]["pose_source"].startswith("captured_phone")
        ),
        "colmap_pose_branch_count": sum(
            1 for record in records if record["branch"]["pose_source"].startswith("all_frame")
        ),
        "plan": _asset(plan_path),
        "deployments": plan["deployments"],
        "current_endpoint": "DINO-only -> official Native-SS30K -> Native-SLat30K -> Stock Mesh decoder",
        "reference_endpoint": "strict original ReconViaGen",
        "mesh_frame_contract": MESH_FRAME_CONTRACT,
        "mesh_frame_interpretation_valid": True,
        "all_current_meshes_native_runtime_o": True,
        "historical_v1_tree_modified": False,
        "repaired_historical_branch_count": sum(record["legacy_axis_repair"] for record in records),
        "fresh_ar_time_random_branch_count": sum(not record["legacy_axis_repair"] for record in records),
        "records": records,
        "target_or_metric_consumed": False,
        "formal_claim_allowed": False,
        "scope_guard": (
            "Qualitative real captures without ground-truth object pose or target Mesh. "
            "Every cyan contour uses a native runtime-O Mesh and the exact physical "
            "Mesh_O -> T_O2W -> T_W2C -> K_raw camera chain; no post-hoc fitting is used."
        ),
    }
    if not report["passed"]:
        raise RuntimeError("corrected six-branch matrix is incomplete")
    report_path = output / "report.json"
    if report_path.is_file():
        existing = load_json(report_path)
        for key, value in report.items():
            if key != "created_at_utc" and existing.get(key) != value:
                raise RuntimeError(f"existing v2 final report differs: field={key}")
    else:
        atomic_json(report_path, report)
    print(
        json.dumps(
            {"passed": True, "branches": len(records), "report": str(report_path)},
            ensure_ascii=False,
            indent=2,
        )
    )


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p = sub.add_parser("prepare-ar")
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--dataset-name", required=True)
    p = sub.add_parser("migrate-legacy")
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--dataset-name", required=True)
    p.add_argument("--slug", choices=tuple(MIGRATED_BRANCHES), required=True)
    p = sub.add_parser("finalize")
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main() -> None:
    args = make_parser().parse_args()
    if args.command == "prepare":
        prepare(args.output)
    elif args.command == "prepare-ar":
        prepare_ar(args.output, args.dataset_name)
    elif args.command == "migrate-legacy":
        migrate_legacy(args.output, args.dataset_name, args.slug)
    elif args.command == "finalize":
        finalize(args.output)
    else:  # pragma: no cover
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
