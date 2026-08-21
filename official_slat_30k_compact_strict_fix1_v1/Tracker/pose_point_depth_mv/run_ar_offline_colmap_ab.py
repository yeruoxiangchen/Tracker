#!/usr/bin/env python3
"""Run a fixed-input ARFoundation-pose versus offline-COLMAP diagnostic.

The source RGB files, foreground masks, eight inference frames, Mixed
checkpoints, and seed are frozen.  COLMAP may use all saved source frames for
SfM/BA, but the derived inference dataset exposes exactly the frozen eight.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import subprocess
from typing import Any, Callable, Sequence

import numpy as np
from PIL import Image, ImageDraw

from ar_ss_flow.shared_object_preprocessing import canonical_json_sha256
from pose_point_depth_mv.ar_object_capture import CAPTURE_FORMAT, utc_now, write_json
from pose_point_depth_mv.ar_object_reconstruction import (
    ARReconstructionConfig,
    run_ar_object_reconstruction,
)
from pose_point_depth_mv.dataset_tools.prepare_omni_real_video_cache import (
    parse_points,
    parse_registered_images,
)
from pose_point_depth_mv.omni_real_benchmark_common import sha256_file


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_DATASET = (
    REPO_ROOT
    / "pose_point_depth_mv/outputs/可视AR/datasets/20260810_133235_817"
)
DEFAULT_AR_REPLAY_ROOT = (
    REPO_ROOT
    / "pose_point_depth_mv/outputs/可视AR/replays/"
    "20260810_133235_817_axis_viewfix_v1"
)
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT
    / "pose_point_depth_mv/outputs/可视AR/diagnostics/"
    "20260810_133235_817_ar_vs_offline_colmap_fullimage_v1"
)
DEFAULT_COLMAP_BIN = Path("/home/zjr/anaconda3/envs/foundpose/bin/colmap")
DEFAULT_PYTHON = Path("/home/zjr/anaconda3/envs/reconviagen/bin/python")
DEFAULT_FIXED_FRAMES = (
    "frame_0005.jpg",
    "frame_0013.jpg",
    "frame_0025.jpg",
    "frame_0032.jpg",
    "frame_0042.jpg",
    "frame_0058.jpg",
    "frame_0075.jpg",
    "frame_0091.jpg",
)
REPORT_FORMAT = "pose_point_depth_mv.ar_offline_colmap_ab.v1"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _mask_path(dataset: Path, frame_name: str) -> Path:
    stem = Path(frame_name).stem
    for suffix in (".png", ".jpg", ".jpeg"):
        candidate = dataset / "masks" / f"{stem}{suffix}"
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"missing mask for {frame_name}: {dataset / 'masks'}")


def _image_path(dataset: Path, frame_name: str) -> Path:
    for directory in ("images", "rgb"):
        candidate = dataset / directory / frame_name
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"missing RGB frame {frame_name}: {dataset}")


def _only_object(payload: dict[str, Any], *, source: Path) -> dict[str, Any]:
    rows = list(payload.get("objects", []))
    if payload.get("passed") is not True or len(rows) != 1:
        raise RuntimeError(f"expected one passed object in {source}, got {len(rows)}")
    return rows[0]


def load_ar_control(
    source_dataset: Path,
    ar_replay_root: Path,
    expected_fixed_frames: Sequence[str],
) -> dict[str, Any]:
    source_dataset = source_dataset.resolve()
    ar_replay_root = ar_replay_root.resolve()
    capture_report_path = source_dataset / "capture_report.json"
    capture = _read_json(capture_report_path)
    if capture.get("format") != CAPTURE_FORMAT or capture.get("passed") is not True:
        raise RuntimeError(f"source is not a passed AR capture v2 dataset: {source_dataset}")
    all_frames = [str(value) for value in capture.get("selected_frame_names", [])]
    if len(all_frames) < 8 or len(all_frames) != len(set(all_frames)):
        raise RuntimeError(f"invalid saved-frame list in {capture_report_path}")

    runtime_manifest_path = (
        ar_replay_root / "shared/02_runtime_o/runtime_input_manifest.json"
    )
    runtime_manifest = _read_json(runtime_manifest_path)
    runtime_object = _only_object(runtime_manifest, source=runtime_manifest_path)
    raw_report_path = Path(runtime_manifest["source_raw_cache_report"]).resolve()
    raw_report = _read_json(raw_report_path)
    raw_object = _only_object(raw_report, source=raw_report_path)
    raw_cache_path = Path(raw_object["cache_npz"]).resolve()
    with np.load(raw_cache_path, allow_pickle=False) as cache:
        source_names = [str(value) for value in cache["source_frame_name"].tolist()]
    selected_indices = [int(value) for value in runtime_object["selected_source_view_indices"]]
    if any(index < 0 or index >= len(source_names) for index in selected_indices):
        raise RuntimeError("AR runtime manifest has out-of-range selected frame indices")
    ar_fixed_frames = [source_names[index] for index in selected_indices]
    expected = [str(value) for value in expected_fixed_frames]
    if ar_fixed_frames != expected:
        raise RuntimeError(
            "fixed-frame contract differs from the existing AR replay: "
            f"AR={ar_fixed_frames}, requested={expected}"
        )
    if any(name not in all_frames for name in expected):
        raise RuntimeError("one or more fixed frames are absent from the source capture")

    source_binding_frames = {
        str(row["source_frame_name"]): row
        for row in raw_object.get("source_binding", {}).get("frames", [])
    }
    frozen_bindings = []
    for name in expected:
        image = _image_path(source_dataset, name)
        mask = _mask_path(source_dataset, name)
        ar_binding = source_binding_frames.get(name)
        if ar_binding is None:
            raise RuntimeError(f"AR raw-cache binding is missing fixed frame {name}")
        image_hash = sha256_file(image)
        mask_hash = sha256_file(mask)
        if image_hash != ar_binding.get("image_sha256"):
            raise RuntimeError(f"source/AR replay RGB hash mismatch for {name}")
        if mask_hash != ar_binding.get("mask_sha256"):
            raise RuntimeError(f"source/AR replay mask hash mismatch for {name}")
        frozen_bindings.append(
            {
                "frame_name": name,
                "image": str(image),
                "image_sha256": image_hash,
                "mask": str(mask),
                "mask_sha256": mask_hash,
            }
        )

    ar_inference_path = ar_replay_root / "mixed/04_no_vggt_inference/inference_manifest.json"
    ar_contact_sheet = ar_replay_root / "mixed/final/previews/mesh_views_contact_sheet.png"
    if not ar_inference_path.is_file() or not ar_contact_sheet.is_file():
        raise FileNotFoundError(
            "existing AR Mixed inference or preview is incomplete under "
            f"{ar_replay_root}"
        )
    return {
        "capture_report": str(capture_report_path.resolve()),
        "capture_report_sha256": sha256_file(capture_report_path),
        "all_frames": all_frames,
        "fixed_frames": expected,
        "frozen_bindings": frozen_bindings,
        "ar_runtime_manifest": str(runtime_manifest_path.resolve()),
        "ar_runtime_manifest_sha256": sha256_file(runtime_manifest_path),
        "ar_raw_cache": str(raw_cache_path),
        "ar_inference_manifest": str(ar_inference_path.resolve()),
        "ar_inference_manifest_sha256": sha256_file(ar_inference_path),
        "ar_contact_sheet": str(ar_contact_sheet.resolve()),
    }


def colmap_commands(
    *,
    colmap_bin: Path,
    workspace: Path,
    use_foreground_masks: bool,
) -> list[tuple[str, list[str]]]:
    database = workspace / "database.db"
    images = workspace / "images_all18"
    sparse_raw = workspace / "sparse_raw"
    feature = [
        str(colmap_bin),
        "feature_extractor",
        "--database_path",
        str(database),
        "--image_path",
        str(images),
        "--ImageReader.camera_model",
        "SIMPLE_RADIAL",
        "--ImageReader.single_camera",
        "1",
        "--SiftExtraction.use_gpu",
        "1",
        "--SiftExtraction.gpu_index",
        "0",
        "--SiftExtraction.max_num_features",
        "16384",
        "--SiftExtraction.peak_threshold",
        "0.003",
    ]
    if use_foreground_masks:
        feature.extend(["--ImageReader.mask_path", str(workspace / "masks_colmap")])
    matcher = [
        str(colmap_bin),
        "exhaustive_matcher",
        "--database_path",
        str(database),
        "--SiftMatching.use_gpu",
        "1",
        "--SiftMatching.gpu_index",
        "0",
        "--SiftMatching.guided_matching",
        "1",
        "--TwoViewGeometry.min_num_inliers",
        "10",
    ]
    mapper = [
        str(colmap_bin),
        "mapper",
        "--database_path",
        str(database),
        "--image_path",
        str(images),
        "--output_path",
        str(sparse_raw),
        "--Mapper.min_model_size",
        "8",
        "--Mapper.min_num_matches",
        "10",
        "--Mapper.init_min_num_inliers",
        "30",
        "--Mapper.abs_pose_min_num_inliers",
        "20",
        "--Mapper.tri_ignore_two_view_tracks",
        "0",
        "--Mapper.ba_refine_principal_point",
        "0",
    ]
    return [("feature_extractor", feature), ("exhaustive_matcher", matcher), ("mapper", mapper)]


def _copy_verified(source: Path, destination: Path) -> None:
    if destination.is_file():
        if sha256_file(source) != sha256_file(destination):
            raise RuntimeError(f"existing copied input differs from source: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def prepare_colmap_inputs(
    source_dataset: Path,
    workspace: Path,
    frame_names: Sequence[str],
) -> None:
    image_dir = workspace / "images_all18"
    mask_dir = workspace / "masks_colmap"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    for name in frame_names:
        image = _image_path(source_dataset, name)
        mask = _mask_path(source_dataset, name)
        _copy_verified(image, image_dir / name)
        # COLMAP appends .png to the full image filename, including .jpg.
        _copy_verified(mask, mask_dir / f"{name}.png")


def _stage_marker(workspace: Path, name: str) -> Path:
    return workspace / "stages" / f"{name}.json"


def _run_logged(
    *,
    name: str,
    command: Sequence[str],
    workspace: Path,
    environment: dict[str, str],
    resume: bool,
    complete: Callable[[], bool],
) -> None:
    marker = _stage_marker(workspace, name)
    if marker.is_file():
        payload = _read_json(marker)
        if payload.get("passed") is not True or payload.get("command") != list(command):
            raise RuntimeError(f"stale COLMAP stage marker: {marker}")
        if not complete():
            raise RuntimeError(f"COLMAP stage marker has incomplete artifacts: {marker}")
        if not resume:
            raise FileExistsError(f"COLMAP stage already exists; pass --resume: {marker}")
        print(f"[ar_colmap_ab] reuse stage={name}", flush=True)
        return
    marker.parent.mkdir(parents=True, exist_ok=True)
    log_path = workspace / "colmap.log"
    header = f"\n[{utc_now()}] stage={name} command={shlex.join(command)}\n"
    print(header.rstrip(), flush=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(header)
        log.flush()
        process = subprocess.Popen(
            list(command),
            cwd=str(REPO_ROOT),
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            log.write(line)
            log.flush()
            print(f"[ar_colmap_ab:{name}] {line}", end="", flush=True)
        return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"COLMAP stage {name} failed with rc={return_code}; log={log_path}")
    if not complete():
        raise RuntimeError(f"COLMAP stage {name} produced incomplete artifacts; log={log_path}")
    write_json(
        marker,
        {
            "created_at_utc": utc_now(),
            "name": name,
            "command": list(command),
            "log": str(log_path.resolve()),
            "passed": True,
        },
    )


def _raw_model_dirs(workspace: Path) -> list[Path]:
    root = workspace / "sparse_raw"
    return sorted(
        (path for path in root.iterdir() if path.is_dir() and path.name.isdigit()),
        key=lambda path: int(path.name),
    ) if root.is_dir() else []


def _model_binary_complete(path: Path) -> bool:
    return all((path / name).is_file() for name in ("cameras.bin", "images.bin", "points3D.bin"))


def _model_text_complete(path: Path) -> bool:
    return all((path / name).is_file() for name in ("cameras.txt", "images.txt", "points3D.txt"))


def _convert_model(
    *,
    name: str,
    colmap_bin: Path,
    source: Path,
    destination: Path,
    workspace: Path,
    environment: dict[str, str],
    resume: bool,
) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    command = [
        str(colmap_bin),
        "model_converter",
        "--input_path",
        str(source),
        "--output_path",
        str(destination),
        "--output_type",
        "TXT",
    ]
    _run_logged(
        name=name,
        command=command,
        workspace=workspace,
        environment=environment,
        resume=resume,
        complete=lambda: _model_text_complete(destination),
    )


def select_largest_model(
    *,
    colmap_bin: Path,
    workspace: Path,
    fixed_frames: Sequence[str],
    environment: dict[str, str],
    resume: bool,
) -> tuple[Path, dict[str, Any]]:
    models = _raw_model_dirs(workspace)
    if not models:
        raise RuntimeError("COLMAP mapper produced no sparse model")
    candidates = []
    for model in models:
        text_dir = workspace / "candidate_txt" / model.name
        _convert_model(
            name=f"candidate_{model.name}_to_text",
            colmap_bin=colmap_bin,
            source=model,
            destination=text_dir,
            workspace=workspace,
            environment=environment,
            resume=resume,
        )
        registered = parse_registered_images(text_dir / "images.txt")
        points = parse_points(text_dir / "points3D.txt")
        candidates.append(
            {
                "model": model,
                "text": text_dir,
                "registered_frames": [str(row["name"]) for row in registered],
                "registered_count": len(registered),
                "point_count": int(len(points["xyz"])),
            }
        )
    selected = min(
        candidates,
        key=lambda row: (-row["registered_count"], -row["point_count"], int(row["model"].name)),
    )
    missing = sorted(set(fixed_frames) - set(selected["registered_frames"]))
    if missing:
        raise RuntimeError(
            "largest COLMAP model does not register every frozen inference frame; "
            f"missing={missing}. Frames will not be silently replaced."
        )
    audit = {
        "selection": "maximum_registered_images_then_points_then_model_index",
        "selected_model": str(selected["model"].resolve()),
        "selected_registered_count": selected["registered_count"],
        "selected_point_count_before_final_ba": selected["point_count"],
        "fixed_frames_all_registered": True,
        "candidates": [
            {
                "model": str(row["model"].resolve()),
                "registered_count": row["registered_count"],
                "point_count": row["point_count"],
            }
            for row in candidates
        ],
    }
    return Path(selected["model"]), audit


def run_colmap(
    *,
    source_dataset: Path,
    workspace: Path,
    all_frames: Sequence[str],
    fixed_frames: Sequence[str],
    colmap_bin: Path,
    gpu: str,
    use_foreground_masks: bool,
    resume: bool,
) -> tuple[Path, dict[str, Any]]:
    if not colmap_bin.is_file():
        raise FileNotFoundError(f"COLMAP executable is missing: {colmap_bin}")
    prepare_colmap_inputs(source_dataset, workspace, all_frames)
    (workspace / "sparse_raw").mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    commands = colmap_commands(
        colmap_bin=colmap_bin,
        workspace=workspace,
        use_foreground_masks=use_foreground_masks,
    )
    database = workspace / "database.db"
    for name, command in commands:
        if name == "feature_extractor":
            complete = lambda: database.is_file() and database.stat().st_size > 0
        elif name == "exhaustive_matcher":
            complete = lambda: database.is_file() and database.stat().st_size > 0
        else:
            complete = lambda: bool(_raw_model_dirs(workspace))
        _run_logged(
            name=name,
            command=command,
            workspace=workspace,
            environment=environment,
            resume=resume,
            complete=complete,
        )

    selected_model, selection = select_largest_model(
        colmap_bin=colmap_bin,
        workspace=workspace,
        fixed_frames=fixed_frames,
        environment=environment,
        resume=resume,
    )
    ba_model = workspace / "sparse_ba"
    ba_model.mkdir(parents=True, exist_ok=True)
    ba_command = [
        str(colmap_bin),
        "bundle_adjuster",
        "--input_path",
        str(selected_model),
        "--output_path",
        str(ba_model),
        "--BundleAdjustment.refine_focal_length",
        "1",
        "--BundleAdjustment.refine_principal_point",
        "0",
        "--BundleAdjustment.refine_extra_params",
        "1",
    ]
    _run_logged(
        name="final_bundle_adjuster",
        command=ba_command,
        workspace=workspace,
        environment=environment,
        resume=resume,
        complete=lambda: _model_binary_complete(ba_model),
    )
    text_model = workspace / "sparse_txt"
    _convert_model(
        name="final_model_to_text",
        colmap_bin=colmap_bin,
        source=ba_model,
        destination=text_model,
        workspace=workspace,
        environment=environment,
        resume=resume,
    )
    final_registered = parse_registered_images(text_model / "images.txt")
    final_names = [str(row["name"]) for row in final_registered]
    missing = sorted(set(fixed_frames) - set(final_names))
    if missing:
        raise RuntimeError(f"final BA model lost frozen frames: {missing}")
    final_points = parse_points(text_model / "points3D.txt")
    selection.update(
        {
            "final_text_model": str(text_model.resolve()),
            "final_registered_count": len(final_registered),
            "final_point_count": int(len(final_points["xyz"])),
            "final_fixed_frames_all_registered": True,
            "feature_domain": "foreground_mask" if use_foreground_masks else "full_image",
        }
    )
    return text_model, selection


def materialize_frozen_dataset(
    *,
    source_dataset: Path,
    fixed_frames: Sequence[str],
    text_model: Path,
    destination: Path,
) -> dict[str, Any]:
    image_dir = destination / "images"
    mask_dir = destination / "masks"
    sparse_dir = destination / "sparse/0"
    for name in fixed_frames:
        _copy_verified(_image_path(source_dataset, name), image_dir / name)
        _copy_verified(_mask_path(source_dataset, name), mask_dir / f"{Path(name).stem}.png")
    for filename in ("cameras.txt", "images.txt", "points3D.txt"):
        _copy_verified(text_model / filename, sparse_dir / filename)
    report = {
        "format": "pose_point_depth_mv.ar_colmap_fixed_dataset.v1",
        "created_at_utc": utc_now(),
        "source_dataset": str(source_dataset.resolve()),
        "fixed_frame_count": len(fixed_frames),
        "fixed_frames": list(fixed_frames),
        "image_files_exposed_to_inference": sorted(path.name for path in image_dir.iterdir()),
        "mask_files_exposed_to_inference": sorted(path.name for path in mask_dir.iterdir()),
        "colmap_text_model": str(text_model.resolve()),
        "sparse": str(sparse_dir.resolve()),
        "sparse_image_records": "complete_all_frame_ba_model",
        "scope_guard": (
            "Only the frozen eight RGB/mask files are exposed to runtime preprocessing. "
            "Camera poses, intrinsics, and sparse points come from all-frame offline COLMAP."
        ),
        "passed": True,
    }
    write_json(destination / "fixed_dataset_report.json", report)
    return report


def _load_camera_cache(path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    with np.load(path, allow_pickle=False) as payload:
        names = [str(value) for value in payload["source_frame_name"].tolist()]
        intrinsics = np.asarray(payload["K"], dtype=np.float64)
        poses = np.asarray(payload["T_W2C"], dtype=np.float64)
    return {name: (intrinsics[index], poses[index]) for index, name in enumerate(names)}


def _summary(values: Sequence[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return {"count": 0}
    return {
        "count": int(len(array)),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def camera_diagnostics(
    ar_cache_path: Path,
    colmap_cache_path: Path,
    fixed_frames: Sequence[str],
) -> dict[str, Any]:
    ar = _load_camera_cache(ar_cache_path)
    colmap = _load_camera_cache(colmap_cache_path)
    if any(name not in ar or name not in colmap for name in fixed_frames):
        raise RuntimeError("camera diagnostic cache does not contain every fixed frame")
    fx_relative = []
    fy_relative = []
    principal_delta = []
    rows = []
    for name in fixed_frames:
        ar_k, _ar_pose = ar[name]
        colmap_k, _colmap_pose = colmap[name]
        fx_relative.append(float(colmap_k[0, 0] / ar_k[0, 0] - 1.0))
        fy_relative.append(float(colmap_k[1, 1] / ar_k[1, 1] - 1.0))
        principal_delta.append(float(np.linalg.norm(colmap_k[:2, 2] - ar_k[:2, 2])))
        rows.append(
            {
                "frame_name": name,
                "ar_fx_fy_cx_cy": [
                    float(ar_k[0, 0]), float(ar_k[1, 1]),
                    float(ar_k[0, 2]), float(ar_k[1, 2]),
                ],
                "colmap_fx_fy_cx_cy": [
                    float(colmap_k[0, 0]), float(colmap_k[1, 1]),
                    float(colmap_k[0, 2]), float(colmap_k[1, 2]),
                ],
            }
        )
    rotation_errors = []
    ar_baselines = []
    colmap_baselines = []
    for first, second in itertools.combinations(fixed_frames, 2):
        _ar_k1, ar_pose1 = ar[first]
        _ar_k2, ar_pose2 = ar[second]
        _col_k1, col_pose1 = colmap[first]
        _col_k2, col_pose2 = colmap[second]
        ar_relative = ar_pose2[:3, :3] @ ar_pose1[:3, :3].T
        col_relative = col_pose2[:3, :3] @ col_pose1[:3, :3].T
        delta = col_relative @ ar_relative.T
        cosine = float(np.clip((np.trace(delta) - 1.0) / 2.0, -1.0, 1.0))
        rotation_errors.append(float(math.degrees(math.acos(cosine))))
        ar_centers = [np.linalg.inv(pose)[:3, 3] for pose in (ar_pose1, ar_pose2)]
        col_centers = [np.linalg.inv(pose)[:3, 3] for pose in (col_pose1, col_pose2)]
        ar_baselines.append(float(np.linalg.norm(ar_centers[1] - ar_centers[0])))
        colmap_baselines.append(float(np.linalg.norm(col_centers[1] - col_centers[0])))
    ar_scale = max(float(np.median(ar_baselines)), 1.0e-12)
    colmap_scale = max(float(np.median(colmap_baselines)), 1.0e-12)
    baseline_shape_errors = np.abs(
        np.asarray(ar_baselines) / ar_scale - np.asarray(colmap_baselines) / colmap_scale
    )
    return {
        "intrinsics": {
            "colmap_over_ar_fx_minus_one": _summary(fx_relative),
            "colmap_over_ar_fy_minus_one": _summary(fy_relative),
            "principal_point_delta_pixels": _summary(principal_delta),
            "per_frame": rows,
        },
        "pose": {
            "pairwise_relative_rotation_error_degrees": _summary(rotation_errors),
            "scale_normalized_pairwise_baseline_absolute_error": _summary(
                baseline_shape_errors.tolist()
            ),
            "pair_count": len(rotation_errors),
            "gauge_note": (
                "Relative rotations and normalized baseline distances are invariant to "
                "the independent COLMAP world similarity gauge."
            ),
        },
    }


def make_comparison_sheet(ar_sheet: Path, colmap_sheet: Path, output: Path) -> None:
    with Image.open(ar_sheet) as handle:
        left = handle.convert("RGB")
    with Image.open(colmap_sheet) as handle:
        right = handle.convert("RGB")
    target_height = max(left.height, right.height)
    if left.height != target_height:
        left = left.resize(
            (round(left.width * target_height / left.height), target_height),
            Image.Resampling.LANCZOS,
        )
    if right.height != target_height:
        right = right.resize(
            (round(right.width * target_height / right.height), target_height),
            Image.Resampling.LANCZOS,
        )
    header = 40
    canvas = Image.new("RGB", (left.width + right.width, target_height + header), "white")
    canvas.paste(left, (0, header))
    canvas.paste(right, (left.width, header))
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 13), "ARFoundation pose/K", fill="black")
    draw.text((left.width + 12, 13), "Offline COLMAP pose/K", fill="black")
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def _colmap_raw_cache(reconstruction: dict[str, Any]) -> Path:
    raw_report_path = Path(reconstruction["stage_reports"]["raw_cache"])
    raw_object = _only_object(_read_json(raw_report_path), source=raw_report_path)
    return Path(raw_object["cache_npz"]).resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_dataset", default=str(DEFAULT_SOURCE_DATASET))
    parser.add_argument("--ar_replay_root", default=str(DEFAULT_AR_REPLAY_ROOT))
    parser.add_argument("--output_root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--colmap_bin", default=str(DEFAULT_COLMAP_BIN))
    parser.add_argument("--python", default=str(DEFAULT_PYTHON))
    parser.add_argument("--gpu", default=os.environ.get("GPU", "4"))
    parser.add_argument("--fixed_frame", action="append")
    feature_group = parser.add_mutually_exclusive_group()
    feature_group.add_argument(
        "--foreground_features",
        action="store_true",
        help="diagnostic only: restrict COLMAP SIFT to the object mask",
    )
    feature_group.add_argument(
        "--full_image_features",
        action="store_true",
        help="explicitly select the default full-RGB COLMAP feature domain",
    )
    parser.add_argument("--preview_render_frames", type=int, default=24)
    parser.add_argument("--preview_count", type=int, default=6)
    parser.add_argument("--preview_resolution", type=int, default=512)
    parser.add_argument("--prepare_only", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    source_dataset = Path(args.source_dataset).expanduser().resolve()
    ar_replay_root = Path(args.ar_replay_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    colmap_bin = Path(args.colmap_bin).expanduser().resolve()
    fixed_frames = tuple(args.fixed_frame or DEFAULT_FIXED_FRAMES)
    use_foreground_masks = bool(args.foreground_features)
    control = load_ar_control(source_dataset, ar_replay_root, fixed_frames)
    workspace = output_root / "colmap_workspace"
    commands = colmap_commands(
        colmap_bin=colmap_bin,
        workspace=workspace,
        use_foreground_masks=use_foreground_masks,
    )
    if args.dry_run:
        print(json.dumps({
            "passed": True,
            "dry_run": True,
            "source_dataset": str(source_dataset),
            "fixed_frames": list(fixed_frames),
            "commands": [{"stage": name, "command": shlex.join(command)} for name, command in commands],
            "output_root": str(output_root),
        }, indent=2, ensure_ascii=False))
        return

    report_path = output_root / "report.json"
    if report_path.is_file():
        existing = _read_json(report_path)
        if existing.get("passed") is True:
            if not args.resume:
                raise FileExistsError(f"diagnostic already completed; pass --resume: {report_path}")
            print(json.dumps(existing, indent=2, ensure_ascii=False))
            return
    if output_root.exists() and any(output_root.iterdir()) and not args.resume:
        raise FileExistsError(f"diagnostic output exists; pass --resume: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    input_binding = {
        "format": "pose_point_depth_mv.ar_offline_colmap_ab_input_binding.v1",
        "source_dataset": str(source_dataset),
        "capture_report_sha256": control["capture_report_sha256"],
        "ar_runtime_manifest_sha256": control["ar_runtime_manifest_sha256"],
        "ar_inference_manifest_sha256": control["ar_inference_manifest_sha256"],
        "all_frames": control["all_frames"],
        "fixed_frames": list(fixed_frames),
        "frozen_bindings": control["frozen_bindings"],
        "colmap": {
            "binary": str(colmap_bin),
            "binary_sha256": sha256_file(colmap_bin),
            "feature_domain": "foreground_mask" if use_foreground_masks else "full_image",
            "commands": [{"stage": name, "command": command} for name, command in commands],
        },
    }
    input_binding["sha256"] = canonical_json_sha256(input_binding)
    binding_path = output_root / "input_binding.json"
    if binding_path.is_file():
        if _read_json(binding_path) != input_binding:
            raise RuntimeError(f"diagnostic input binding changed: {binding_path}")
    else:
        write_json(binding_path, input_binding)

    text_model, colmap_audit = run_colmap(
        source_dataset=source_dataset,
        workspace=workspace,
        all_frames=control["all_frames"],
        fixed_frames=fixed_frames,
        colmap_bin=colmap_bin,
        gpu=str(args.gpu),
        use_foreground_masks=use_foreground_masks,
        resume=bool(args.resume),
    )
    frozen_dataset = output_root / "frozen_colmap_dataset_8view"
    frozen_report = materialize_frozen_dataset(
        source_dataset=source_dataset,
        fixed_frames=fixed_frames,
        text_model=text_model,
        destination=frozen_dataset,
    )
    preparation = {
        "format": REPORT_FORMAT,
        "created_at_utc": utc_now(),
        "input_binding": str(binding_path.resolve()),
        "input_binding_sha256": sha256_file(binding_path),
        "colmap": colmap_audit,
        "frozen_dataset": frozen_report,
        "reconstruction_skipped": bool(args.prepare_only),
        "passed": True,
    }
    preparation_path = output_root / "preparation_report.json"
    write_json(preparation_path, preparation)
    if args.prepare_only:
        print(json.dumps({
            "passed": True,
            "prepare_only": True,
            "fixed_frames": list(fixed_frames),
            "colmap_registered": colmap_audit["final_registered_count"],
            "colmap_points": colmap_audit["final_point_count"],
            "report": str(preparation_path),
        }, indent=2, ensure_ascii=False))
        return

    reconstruction = run_ar_object_reconstruction(
        session_id=f"{source_dataset.name}_offline_colmap_fixed8",
        dataset_dir=frozen_dataset,
        output_root=output_root / "colmap_branch",
        config=ARReconstructionConfig(
            python=Path(args.python).expanduser().resolve(),
            gpu=str(args.gpu),
            selected_view_count=8,
            view_selection_policy="lexical_even",
            gravity_up_w=None,
            source_frame_names=tuple(fixed_frames),
            preview_render_frames=int(args.preview_render_frames),
            preview_count=int(args.preview_count),
            preview_resolution=int(args.preview_resolution),
        ),
    )
    colmap_cache = _colmap_raw_cache(reconstruction)
    diagnostics = camera_diagnostics(
        Path(control["ar_raw_cache"]), colmap_cache, fixed_frames
    )
    colmap_contact = Path(reconstruction["previews"]["contact_sheet"])
    comparison_sheet = output_root / "ARPose与COLMAPPose_同输入并排预览.png"
    make_comparison_sheet(
        Path(control["ar_contact_sheet"]), colmap_contact, comparison_sheet
    )
    colmap_inference = _read_json(
        Path(reconstruction["stage_reports"]["no_vggt_ss_slat_mesh"])
    )
    ar_inference = _read_json(Path(control["ar_inference_manifest"]))
    checkpoint_match = {
        field: ar_inference.get(field + "_sha256") == colmap_inference.get(field + "_sha256")
        for field in ("native_ss_checkpoint", "native_slat_checkpoint")
    }
    if not all(checkpoint_match.values()):
        raise RuntimeError(f"AR/COLMAP branches used different checkpoints: {checkpoint_match}")
    report = {
        "format": REPORT_FORMAT,
        "created_at_utc": utc_now(),
        "formal": False,
        "object_count": 1,
        "controlled_variables": {
            "source_rgb_and_mask_sha256": control["frozen_bindings"],
            "fixed_frame_names": list(fixed_frames),
            "fixed_frame_count": len(fixed_frames),
            "model": "current Mixed no-VGGT SS+SLat",
            "checkpoint_sha256_equal": checkpoint_match,
            "seed": 42,
        },
        "changed_variables": [
            "camera intrinsics K",
            "camera extrinsics T_W2C",
            "sparse points used by runtime-O canonicalization",
        ],
        "isolation_note": (
            "The current runtime-O frontend needs sparse geometry. Therefore replacing "
            "ARFoundation pose/K with COLMAP also replaces the geometrically coupled AR "
            "point cloud with COLMAP sparse points; this is a frontend-system A/B, not a "
            "mathematically pure pose-only intervention."
        ),
        "ar_branch": {
            "runtime_manifest": control["ar_runtime_manifest"],
            "inference_manifest": control["ar_inference_manifest"],
            "contact_sheet": control["ar_contact_sheet"],
        },
        "colmap_branch": {
            "preparation_report": str(preparation_path.resolve()),
            "reconstruction_report": str(
                (Path(reconstruction["run_dir"]) / "reconstruction_report.json").resolve()
            ),
            "runtime_manifest": reconstruction["stage_reports"]["runtime_o"],
            "inference_manifest": reconstruction["stage_reports"]["no_vggt_ss_slat_mesh"],
            "contact_sheet": str(colmap_contact.resolve()),
            "meshes": reconstruction["meshes"],
        },
        "pose_intrinsics_diagnostics": diagnostics,
        "comparison_sheet": str(comparison_sheet.resolve()),
        "comparison_policy": (
            "Both preview renderers apply display-only normalization. World-frame mesh "
            "coordinates are not compared directly because COLMAP has an independent Sim(3) gauge."
        ),
        "passed": True,
    }
    write_json(report_path, report)
    print(json.dumps({
        "passed": True,
        "formal": False,
        "fixed_frames": list(fixed_frames),
        "colmap_registered": colmap_audit["final_registered_count"],
        "colmap_points": colmap_audit["final_point_count"],
        "comparison_sheet": str(comparison_sheet),
        "report": str(report_path),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
