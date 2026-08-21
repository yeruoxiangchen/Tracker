#!/usr/bin/env python3
"""Freeze, prepare, and audit the five real-capture AR/COLMAP experiments.

Each source capture is evaluated with its original phone/Unity camera poses and
with a genuinely independent all-frame offline-COLMAP reconstruction.  The
COLMAP model is built exactly once per capture.  Three eight-frame input sets
are then derived from registered, mask-backed frames: timeline-uniform,
fixed-seed random, and the repository's object-centred spherical-farthest
policy.  No target Mesh, old reconstruction output, or post-hoc alignment is
consumed.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import shutil
from typing import Any, Sequence

import numpy as np

from pose_point_depth_mv.dataset_tools.prepare_omni_real_runtime_inputs import (
    evenly_spaced_frame_indices,
)
from pose_point_depth_mv.dataset_tools.prepare_omni_real_video_cache import (
    parse_registered_images,
)
from pose_point_depth_mv.omni_real_benchmark_common import (
    atomic_json,
    load_json,
    sha256_file,
)
from pose_point_depth_mv.run_ar_offline_colmap_ab import (
    _image_path,
    _mask_path,
    materialize_frozen_dataset,
    run_colmap,
)
from pose_point_depth_mv.trellis_mesh_coordinate_contract import (
    validate_runtime_o_mesh_frame_contract,
)


TRACKER_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = TRACKER_ROOT / "pose_point_depth_mv/outputs/可视AR/datasets"
OLD_RECON_ROOT = TRACKER_ROOT / "pose_point_depth_mv/outputs/可视AR/reconstructions"
DEFAULT_OUTPUT = (
    TRACKER_ROOT
    / "pose_point_depth_mv/outputs2/"
    "真实采集5组_AR与COLMAPPose_ReconViaGen_vs_SS30K_SLat30K_三选帧轮廓_20260819_v1"
)
DEFAULT_COLMAP = Path("/home/zjr/anaconda3/envs/foundpose/bin/colmap")

SS_REPORT = Path(
    "/data/zjr/proobjaverse_official_30k_heldout_dev64_ss30k_slat30k_20260818_v1/"
    "ss30k_dev64_aggregate/report.json"
)
SS_CHECKPOINT = Path(
    "/data/zjr/proobjaverse_official_30k_checkpoint_archives/"
    "ProObjaverse_30K_noVGGT_SS_SLat_numbered_checkpoints_20260818_v1/"
    "ss/checkpoints/step_030000.pt"
)
SLAT_CHECKPOINT = Path(
    "/data/zjr/proobjaverse_official_30k_checkpoint_archives/"
    "ProObjaverse_30K_noVGGT_SS_SLat_numbered_checkpoints_20260818_v1/"
    "slat/checkpoints/step_030000.pt"
)
ABC_R_EVIDENCE = Path(
    "/data/zjr/proobjaverse_official_30k_heldout_dev64_ss30k_slat30k_20260818_v1/"
    "abc_r_dev64_aggregate/report.json"
)
STOCK_FREEZE = Path(
    "/data/zjr/native_slat_genrecon_v2_mixed1k_20260802_v1/stock_slat_freeze_v2.json"
)

PLAN_FORMAT = "pose_point_depth_mv.real_capture_ar_colmap_ss30k_slat30k_plan.v1"
COLMAP_FORMAT = "pose_point_depth_mv.real_capture_offline_colmap_selection.v1"
REPORT_FORMAT = "pose_point_depth_mv.real_capture_ar_colmap_ss30k_slat30k_batch.v1"
RANDOM_SEED = 20260819
SELECTED_VIEWS = 8

SPECS = (
    {
        "dataset_name": "20260816_035545_862_axisuv_v5",
        "source_reconstruction": (
            "real_official_slat_step25000_phone_20260816_035545_862_axisuv_v5_"
            "seed42_diagnostic_v1"
        ),
    },
    {
        "dataset_name": "20260812_171117_303",
        "source_reconstruction": (
            "real_official_slat_step25000_retest_20260812_171117_303_"
            "seed42_spherical_v1"
        ),
    },
    {
        "dataset_name": "20260816_040547_970_axisuv_v5",
        "source_reconstruction": (
            "real_official_slat_step25000_phone_20260816_040547_970_axisuv_v5_"
            "seed42_diagnostic_v1"
        ),
    },
    {"dataset_name": "20260811_064454_154", "source_reconstruction": "20260811_064454_154"},
    {"dataset_name": "20260811_090511_346", "source_reconstruction": "20260811_090511_346"},
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _asset(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve(strict=True)
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def _source_frames(dataset: Path) -> tuple[Path, list[str]]:
    report_path = dataset / "capture_report.json"
    report = load_json(report_path)
    if report.get("format") != "pose_point_depth_mv.ar_object_capture.v2" or report.get("passed") is not True:
        raise RuntimeError(f"source capture contract differs: {report_path}")
    names = [str(value) for value in report.get("selected_frame_names", [])]
    if len(names) < SELECTED_VIEWS or len(names) != len(set(names)):
        raise RuntimeError(f"invalid source frame identity: {report_path}")
    for name in names:
        _image_path(dataset, name)
        _mask_path(dataset, name)
    return report_path, names


def _dataset_record(spec: dict[str, str], output: Path) -> dict[str, Any]:
    dataset = (DATASET_ROOT / spec["dataset_name"]).resolve(strict=True)
    capture, names = _source_frames(dataset)
    metadata = dataset / "frame_metadata.jsonl"
    phone_meta = dataset / "sparse/0/phone_pose_meta.json"
    if not metadata.is_file() or not phone_meta.is_file():
        raise FileNotFoundError(f"phone/Unity pose provenance is incomplete: {dataset}")
    old = (OLD_RECON_ROOT / spec["source_reconstruction"]).resolve(strict=True)
    object_root = output / "objects" / spec["dataset_name"]
    return {
        **spec,
        "dataset": str(dataset),
        "source_old_reconstruction": str(old),
        "source_old_reconstruction_consumed": False,
        "source_frame_count": len(names),
        "source_frame_names": names,
        "capture_report": _asset(capture),
        "frame_metadata": _asset(metadata),
        "phone_pose_meta": _asset(phone_meta),
        "colmap_report": str((object_root / "00_offline_colmap/selection_report.json").resolve()),
        "branch_root": str((object_root / "branches").resolve()),
    }


def _requested_plan(output: Path) -> dict[str, Any]:
    deployments = {
        "native_ss_report": _asset(SS_REPORT),
        "native_ss_checkpoint": _asset(SS_CHECKPOINT),
        "native_ss_checkpoint_step": 30000,
        "native_slat_checkpoint": _asset(SLAT_CHECKPOINT),
        "native_slat_checkpoint_step": 30000,
        "abc_r_evidence": _asset(ABC_R_EVIDENCE),
        "stock_slat_freeze": _asset(STOCK_FREEZE),
    }
    return {
        "format": PLAN_FORMAT,
        "created_at_utc": utc_now(),
        "passed": True,
        "dataset_count": len(SPECS),
        "branch_count_per_dataset": 4,
        "total_branch_count": len(SPECS) * 4,
        "selected_view_count": SELECTED_VIEWS,
        "random_seed": RANDOM_SEED,
        "geometry_mode": "pose_mask",
        "deployments": deployments,
        "objects": [_dataset_record(dict(spec), output) for spec in SPECS],
        "branch_contract": [
            {
                "slug": "01_ar_phone_spherical8",
                "camera_source": "captured phone/Unity AR pose",
                "selection": "object_spherical_farthest_valid_mask",
                "gravity_up_W": [0.0, 1.0, 0.0],
            },
            {
                "slug": "02_colmap_time_uniform8",
                "camera_source": "all-frame offline COLMAP SfM/BA",
                "selection": "timeline uniform among registered mask-backed frames",
                "gravity_up_W": None,
            },
            {
                "slug": "03_colmap_random8_seed20260819",
                "camera_source": "all-frame offline COLMAP SfM/BA",
                "selection": "fixed-seed random among registered mask-backed frames",
                "gravity_up_W": None,
            },
            {
                "slug": "04_colmap_spherical8",
                "camera_source": "all-frame offline COLMAP SfM/BA",
                "selection": "object_spherical_farthest_valid_mask",
                "gravity_up_W": None,
            },
        ],
        "scope_guard": (
            "Original captures and segmentation masks are consumed. Existing reconstruction "
            "outputs are provenance labels only. Offline COLMAP uses every saved RGB frame "
            "for SfM/BA; only registered mask-backed frames enter frozen eight-view selection."
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
                raise RuntimeError(f"existing plan differs: field={key}")
        print(json.dumps({"passed": True, "reused": True, "plan": str(plan_path)}, indent=2))
        return
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"unbound nonempty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(plan_path, requested)
    print(json.dumps({"passed": True, "reused": False, "plan": str(plan_path)}, indent=2))


def _planned_row(output: Path, dataset_name: str) -> dict[str, Any]:
    plan = load_json(output / "experiment_plan.json")
    if plan.get("format") != PLAN_FORMAT or plan.get("passed") is not True:
        raise RuntimeError("experiment plan is not eligible")
    rows = [row for row in plan["objects"] if row["dataset_name"] == dataset_name]
    if len(rows) != 1:
        raise RuntimeError(f"dataset is not uniquely planned: {dataset_name}")
    return rows[0]


def _materialize_dataset(
    source_dataset: Path,
    frames: Sequence[str],
    text_model: Path,
    destination: Path,
) -> None:
    report_path = destination / "fixed_dataset_report.json"
    if not report_path.is_file():
        materialize_frozen_dataset(
            source_dataset=source_dataset,
            fixed_frames=frames,
            text_model=text_model,
            destination=destination,
        )
    report = load_json(report_path)
    if report.get("passed") is not True or report.get("fixed_frames") != list(frames):
        raise RuntimeError(f"derived COLMAP dataset differs: {destination}")
    if Path(report["colmap_text_model"]).resolve() != text_model.resolve():
        raise RuntimeError(f"derived COLMAP model binding differs: {destination}")
    exposed_images = sorted(path.name for path in (destination / "images").iterdir() if path.is_file())
    exposed_masks = sorted(path.name for path in (destination / "masks").iterdir() if path.is_file())
    if exposed_images != sorted(frames) or exposed_masks != sorted(f"{Path(name).stem}.png" for name in frames):
        raise RuntimeError(f"derived RGB/mask scope differs: {destination}")
    for filename in ("cameras.txt", "images.txt", "points3D.txt"):
        if sha256_file(destination / "sparse/0" / filename) != sha256_file(text_model / filename):
            raise RuntimeError(f"derived sparse model differs: {destination}/{filename}")


def prepare_colmap(
    output: Path,
    dataset_name: str,
    gpu: str,
    colmap_bin: Path,
    resume: bool,
) -> None:
    output = output.expanduser().resolve(strict=True)
    planned = _planned_row(output, dataset_name)
    dataset = Path(planned["dataset"]).resolve(strict=True)
    _capture_path, all_frames = _source_frames(dataset)
    object_root = output / "objects" / dataset_name
    colmap_root = object_root / "00_offline_colmap"
    workspace = colmap_root / "workspace"
    text_model, model_selection = run_colmap(
        source_dataset=dataset,
        workspace=workspace,
        all_frames=all_frames,
        fixed_frames=(),
        colmap_bin=colmap_bin.expanduser().resolve(strict=True),
        gpu=str(gpu),
        use_foreground_masks=False,
        resume=bool(resume),
    )
    registered_rows = parse_registered_images(text_model / "images.txt")
    registered_set = {str(row["name"]) for row in registered_rows}
    unknown = sorted(registered_set - set(all_frames))
    if unknown:
        raise RuntimeError(f"COLMAP registered unknown input names: {unknown}")
    registered = [name for name in all_frames if name in registered_set]
    if len(registered) < SELECTED_VIEWS:
        raise RuntimeError(
            f"COLMAP registered fewer than {SELECTED_VIEWS} mask-backed frames: {len(registered)}"
        )
    for name in registered:
        _mask_path(dataset, name)

    time_indices = evenly_spaced_frame_indices(registered, SELECTED_VIEWS).tolist()
    time_frames = [registered[int(index)] for index in time_indices]
    spec_index = next(index for index, row in enumerate(SPECS) if row["dataset_name"] == dataset_name)
    random_seed = RANDOM_SEED + spec_index * 1009
    draw = random.Random(random_seed).sample(range(len(registered)), SELECTED_VIEWS)
    random_draw_frames = [registered[index] for index in draw]
    random_indices = sorted(draw)
    random_frames = [registered[index] for index in random_indices]

    prepared = colmap_root / "prepared_datasets"
    time_dataset = prepared / "time_uniform8"
    random_dataset = prepared / f"random8_seed{RANDOM_SEED}"
    spherical_pool = prepared / "spherical_pool_all_registered"
    _materialize_dataset(dataset, time_frames, text_model, time_dataset)
    _materialize_dataset(dataset, random_frames, text_model, random_dataset)
    _materialize_dataset(dataset, registered, text_model, spherical_pool)

    frame_bindings = [
        {
            "frame_name": name,
            "image": _asset(_image_path(dataset, name)),
            "mask": _asset(_mask_path(dataset, name)),
        }
        for name in registered
    ]
    payload = {
        "format": COLMAP_FORMAT,
        "created_at_utc": utc_now(),
        "passed": True,
        "dataset_name": dataset_name,
        "source_dataset": str(dataset),
        "source_capture_report": planned["capture_report"],
        "colmap_executable": _asset(colmap_bin),
        "feature_domain": "full original RGB; segmentation masks are not SIFT masks",
        "mask_role": "runtime-O canonicalization and foreground-valid view selection only",
        "all_source_frame_count": len(all_frames),
        "registered_mask_backed_frame_count": len(registered),
        "registered_frame_names_in_timeline_order": registered,
        "registered_frame_bindings": frame_bindings,
        "colmap_model_selection": model_selection,
        "text_model": str(text_model.resolve()),
        "text_model_files": {
            name: _asset(text_model / name)
            for name in ("cameras.txt", "images.txt", "points3D.txt")
        },
        "selections": {
            "time_uniform8": {
                "policy": "timeline_evenly_spaced_registered_mask_backed_including_endpoints",
                "selected_indices": time_indices,
                "selected_frame_names": time_frames,
                "dataset": str(time_dataset.resolve()),
            },
            "random8": {
                "policy": "python_fixed_seed_uniform_sample_without_replacement",
                "registered_pool_count": len(registered),
                "seed": random_seed,
                "base_seed": RANDOM_SEED,
                "draw_order_frame_names": random_draw_frames,
                "selected_indices_in_timeline_order": random_indices,
                "selected_frame_names": random_frames,
                "dataset": str(random_dataset.resolve()),
            },
            "spherical8": {
                "policy": "runtime object_spherical_farthest_valid_mask",
                "selection_deferred_to_runtime": True,
                "registered_pool_frame_names": registered,
                "dataset": str(spherical_pool.resolve()),
            },
        },
        "camera_gauge": (
            "arbitrary offline COLMAP world gauge; no phone/Unity gravity or camera pose is injected"
        ),
        "scope_guard": (
            "All source RGB frames are eligible for SfM/BA. Time/random selections are "
            "frozen only after registration. The spherical branch exposes every registered "
            "mask-backed frame to the authoritative runtime selector."
        ),
    }
    report_path = Path(planned["colmap_report"])
    if report_path.is_file():
        existing = load_json(report_path)
        for key, value in payload.items():
            if key != "created_at_utc" and existing.get(key) != value:
                raise RuntimeError(f"existing COLMAP selection report differs: field={key}")
    else:
        atomic_json(report_path, payload)
    print(
        json.dumps(
            {
                "passed": True,
                "dataset": dataset_name,
                "registered": len(registered),
                "time_uniform8": time_frames,
                "random8": random_frames,
                "report": str(report_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def check_runtime(manifest_path: Path, allow_diagnostic: bool) -> None:
    path = manifest_path.expanduser().resolve(strict=True)
    manifest = load_json(path)
    rows = list(manifest.get("objects", []))
    if manifest.get("passed") is not True or len(rows) != 1:
        raise RuntimeError(f"runtime did not pass exactly one object: {path}")
    row = rows[0]
    selected = [str(value) for value in row.get("selected_frame_names", [])]
    if row.get("geometry_mode") != "pose_mask" or len(selected) != SELECTED_VIEWS:
        raise RuntimeError(f"runtime geometry/view contract differs: {path}")
    quality = dict(row.get("input_quality") or {})
    formal = quality.get("formal_input_passed") is True
    if not formal and not allow_diagnostic:
        raise RuntimeError(f"runtime failed formal input gates: {quality}")
    print(
        json.dumps(
            {
                "passed": True,
                "manifest": str(path),
                "object_key": row["object_key"],
                "selected_frame_names": selected,
                "selection": row.get("view_selection"),
                "formal_input_passed": formal,
                "diagnostic_scope": bool(not formal and allow_diagnostic),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _copy_verified(source: Path, destination: Path) -> dict[str, Any]:
    source = source.resolve(strict=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        if sha256_file(destination) != sha256_file(source):
            raise RuntimeError(f"presentation artifact differs: {destination}")
    else:
        shutil.copy2(source, destination)
    return _asset(destination)


def _branch_specs(planned: dict[str, Any], colmap: dict[str, Any]) -> list[dict[str, Any]]:
    selections = colmap["selections"]
    return [
        {
            "slug": "01_ar_phone_spherical8",
            "dataset": planned["dataset"],
            "pose_source": "captured_phone_unity_ar_pose",
            "runtime_policy": "object_spherical_farthest_valid_mask",
            "expected_selected": None,
            "registered_pool": planned["source_frame_names"],
            "gravity_up_W": [0.0, 1.0, 0.0],
        },
        {
            "slug": "02_colmap_time_uniform8",
            "dataset": selections["time_uniform8"]["dataset"],
            "pose_source": "all_frame_offline_colmap_sfm_ba",
            "runtime_policy": "lexical_even",
            "expected_selected": selections["time_uniform8"]["selected_frame_names"],
            "registered_pool": colmap["registered_frame_names_in_timeline_order"],
            "gravity_up_W": None,
        },
        {
            "slug": "03_colmap_random8_seed20260819",
            "dataset": selections["random8"]["dataset"],
            "pose_source": "all_frame_offline_colmap_sfm_ba",
            "runtime_policy": "lexical_even",
            "expected_selected": selections["random8"]["selected_frame_names"],
            "registered_pool": colmap["registered_frame_names_in_timeline_order"],
            "gravity_up_W": None,
        },
        {
            "slug": "04_colmap_spherical8",
            "dataset": selections["spherical8"]["dataset"],
            "pose_source": "all_frame_offline_colmap_sfm_ba",
            "runtime_policy": "object_spherical_farthest_valid_mask",
            "expected_selected": None,
            "registered_pool": colmap["registered_frame_names_in_timeline_order"],
            "gravity_up_W": None,
        },
    ]


def finalize(output: Path) -> None:
    output = output.expanduser().resolve(strict=True)
    plan_path = output / "experiment_plan.json"
    plan = load_json(plan_path)
    if plan.get("format") != PLAN_FORMAT or plan.get("passed") is not True:
        raise RuntimeError("experiment plan did not pass")
    records: list[dict[str, Any]] = []
    for planned in plan["objects"]:
        colmap_path = Path(planned["colmap_report"]).resolve(strict=True)
        colmap = load_json(colmap_path)
        if colmap.get("format") != COLMAP_FORMAT or colmap.get("passed") is not True:
            raise RuntimeError(f"offline COLMAP report did not pass: {colmap_path}")
        branch_root = Path(planned["branch_root"])
        for branch in _branch_specs(planned, colmap):
            root = branch_root / branch["slug"]
            raw_path = root / "01_raw_cache/raw_cache_report.json"
            runtime_path = root / "02_runtime_o/runtime_input_manifest.json"
            model_path = root / "03_dino_only_input/model_input_manifest.json"
            current_path = root / "04_current_ss30k_slat30k/inference_manifest.json"
            recon_path = root / "05_reconviagen/inference_manifest.json"
            contour_path = root / "06_current_camera_contours/report.json"
            required = (raw_path, runtime_path, model_path, current_path, recon_path, contour_path)
            for path in required:
                if not path.is_file():
                    raise FileNotFoundError(path)
            raw, runtime, model, current, recon, contour = map(load_json, required)
            if not all(row.get("passed") is True for row in (raw, runtime, model, current, recon, contour)):
                raise RuntimeError(f"one branch stage did not pass: {planned['dataset_name']} {branch['slug']}")
            if len(raw.get("objects", [])) != 1 or len(runtime.get("objects", [])) != 1:
                raise RuntimeError(f"raw/runtime object matrix differs: {root}")
            raw_row = raw["objects"][0]
            runtime_row = runtime["objects"][0]
            if Path(raw_row["source_dataset"]).resolve() != Path(branch["dataset"]).resolve():
                raise RuntimeError(f"raw dataset binding differs: {root}")
            sparse = Path(raw_row["authoritative_colmap_dir"]).resolve()
            if branch["pose_source"].startswith("all_frame_offline_colmap"):
                if sparse != (Path(branch["dataset"]) / "sparse/0").resolve():
                    raise RuntimeError(f"offline COLMAP sparse binding differs: {root}")
                for name, asset in colmap["text_model_files"].items():
                    if sha256_file(sparse / name) != asset["sha256"]:
                        raise RuntimeError(f"offline COLMAP sparse hash differs: {root}/{name}")
            else:
                if not (sparse / "phone_pose_meta.json").is_file():
                    raise RuntimeError(f"AR branch lost phone pose provenance: {root}")
            if runtime_row.get("geometry_mode") != "pose_mask" or int(runtime_row.get("selected_view_count", -1)) != SELECTED_VIEWS:
                raise RuntimeError(f"runtime pose-mask contract differs: {root}")
            selected_runtime = [str(value) for value in runtime_row.get("selected_frame_names", [])]
            selected_indices = [
                int(value) for value in runtime_row.get("selected_source_view_indices", [])
            ]
            with np.load(Path(raw_row["cache_npz"]), allow_pickle=False) as raw_cache:
                if "source_frame_name" not in raw_cache.files:
                    raise RuntimeError(f"raw cache lacks physical source-frame identity: {root}")
                source_names = [str(value) for value in raw_cache["source_frame_name"].tolist()]
            if any(index < 0 or index >= len(source_names) for index in selected_indices):
                raise RuntimeError(f"runtime source-view index is out of range: {root}")
            selected_source = [source_names[index] for index in selected_indices]
            if (
                len(selected_runtime) != SELECTED_VIEWS
                or len(set(selected_runtime)) != SELECTED_VIEWS
                or len(selected_source) != SELECTED_VIEWS
                or len(set(selected_source)) != SELECTED_VIEWS
            ):
                raise RuntimeError(f"selected frame matrix differs: {root}")
            if not set(selected_source).issubset(set(branch["registered_pool"])):
                raise RuntimeError(f"runtime selected a frame outside the registered pool: {root}")
            if branch["expected_selected"] is not None and selected_source != branch["expected_selected"]:
                raise RuntimeError(f"frozen time/random selection differs: {root}")
            selection_policy = str((runtime_row.get("view_selection") or {}).get("policy"))
            if branch["runtime_policy"] == "lexical_even":
                if selection_policy != "lexical_frame_order_evenly_spaced_including_endpoints":
                    raise RuntimeError(f"lexical selection contract differs: {root}")
            elif selection_policy != "pose_mask_segmented_object_spherical_farthest_valid_mask":
                raise RuntimeError(f"spherical selection contract differs: {root}")
            if model.get("vggt_model_loaded") is True or model.get("vggt_model_executed") is True:
                raise RuntimeError(f"DINO-only input unexpectedly used VGGT: {root}")
            if len(current.get("objects", [])) != 1 or len(recon.get("objects", [])) != 1:
                raise RuntimeError(f"inference object matrix differs: {root}")
            current_row = current["objects"][0]
            recon_row = recon["objects"][0]
            validate_runtime_o_mesh_frame_contract(current_row)
            current_mesh = Path(current_row["mesh"]).resolve(strict=True)
            recon_mesh = Path(recon_row["mesh"]).resolve(strict=True)
            if current_row.get("native_ss_checkpoint_sha256") != plan["deployments"]["native_ss_checkpoint"]["sha256"]:
                raise RuntimeError(f"SS30K identity differs: {root}")
            if current_row.get("native_slat_checkpoint_sha256") != plan["deployments"]["native_slat_checkpoint"]["sha256"]:
                raise RuntimeError(f"SLat30K identity differs: {root}")
            if current_row.get("native_slat_checkpoint_step") != 30000 or current_row.get("vggt_model_executed") is not False:
                raise RuntimeError(f"current endpoint architecture/step differs: {root}")
            if recon_row.get("method") != "reconviagen_original":
                raise RuntimeError(f"reference endpoint is not strict ReconViaGen: {root}")
            if contour.get("mesh_o_sha256") != sha256_file(current_mesh):
                raise RuntimeError(f"contour/current Mesh binding differs: {root}")
            if contour.get("runtime_input_manifest_sha256") != sha256_file(runtime_path):
                raise RuntimeError(f"contour/runtime binding differs: {root}")
            if contour.get("projection_formula") != "Mesh_O -> T_O2W -> Mesh_W -> T_W2C -> K_raw + raw distortion":
                raise RuntimeError(f"contour projection formula differs: {root}")
            presentation = root / "07_presentation_meshes"
            current_copy = _copy_verified(current_mesh, presentation / "当前SS30K_SLat30K_pose-mask_runtime-O.obj")
            recon_copy = _copy_verified(recon_mesh, presentation / "ReconViaGen原版_reference-O.obj")
            records.append(
                {
                    "dataset_name": planned["dataset_name"],
                    "branch": branch,
                    "run_dir": str(root.resolve()),
                    "object_key": runtime_row["object_key"],
                    "selected_runtime_view_names": selected_runtime,
                    "selected_source_frame_names": selected_source,
                    "runtime_selection": runtime_row.get("view_selection"),
                    "input_quality": runtime_row.get("input_quality"),
                    "raw_cache_report": _asset(raw_path),
                    "runtime_input_manifest": _asset(runtime_path),
                    "model_input_manifest": _asset(model_path),
                    "current_inference_manifest": _asset(current_path),
                    "reconviagen_inference_manifest": _asset(recon_path),
                    "contour_report": _asset(contour_path),
                    "contour_overview": _asset(Path(contour["overview"])),
                    "current_mesh": current_copy,
                    "reconviagen_mesh": recon_copy,
                }
            )
    report = {
        "format": REPORT_FORMAT,
        "created_at_utc": utc_now(),
        "passed": len(records) == int(plan["total_branch_count"]),
        "dataset_count": int(plan["dataset_count"]),
        "branch_count": len(records),
        "plan": _asset(plan_path),
        "deployments": plan["deployments"],
        "current_endpoint": "DINO-only -> official Native-SS30K -> Native-SLat30K -> Stock Mesh decoder",
        "reference_endpoint": "strict original ReconViaGen",
        "records": records,
        "target_or_metric_consumed": False,
        "formal_claim_allowed": False,
        "mesh_frame_interpretation_valid": False,
        "mesh_frame_known_issue": (
            "The completed v1 inference artifacts exported the decoder Mesh with "
            "transform_pose=True. The decoder-native Mesh already shares the runtime-O "
            "sparse-grid axes, so the additional (x,y,z)->(x,z,-y) rotation invalidates "
            "the camera-contour orientation. Model inference and intrinsic Mesh geometry "
            "remain complete; corrected contours require an identity-axis v2 export."
        ),
        "requires_mesh_frame_v2_rerender": True,
        "scope_guard": (
            "Qualitative real captures without ground-truth object pose or target Mesh. "
            "COLMAP poses are estimated from full RGB; masks define runtime-O. Cyan contours "
            "use direct physical camera reprojection without ICP or post-hoc fitting."
        ),
    }
    if not report["passed"]:
        raise RuntimeError("real-capture AR/COLMAP batch is incomplete")
    report_path = output / "report.json"
    if report_path.is_file():
        existing = load_json(report_path)
        for key, value in report.items():
            if key != "created_at_utc" and existing.get(key) != value:
                raise RuntimeError(f"existing final report differs: field={key}")
    else:
        atomic_json(report_path, report)
    print(json.dumps({"passed": True, "branches": len(records), "report": str(report_path)}, indent=2))


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p = sub.add_parser("prepare-colmap")
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--dataset-name", required=True)
    p.add_argument("--gpu", required=True)
    p.add_argument("--colmap-bin", type=Path, default=DEFAULT_COLMAP)
    p.add_argument("--resume", action="store_true")
    p = sub.add_parser("check-runtime")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--allow-diagnostic", action="store_true")
    p = sub.add_parser("finalize")
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main() -> None:
    args = make_parser().parse_args()
    if args.command == "prepare":
        prepare(args.output)
    elif args.command == "prepare-colmap":
        prepare_colmap(args.output, args.dataset_name, args.gpu, args.colmap_bin, args.resume)
    elif args.command == "check-runtime":
        check_runtime(args.manifest, bool(args.allow_diagnostic))
    elif args.command == "finalize":
        finalize(args.output)
    else:  # pragma: no cover
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
