#!/usr/bin/env python3
"""Prepare and finalize an Objectron SS30K/SLat30K O-frame matrix.

The experiment freezes one official Objectron camera sequence, three eight-view
selection policies, and two object coordinate frames for the current endpoint:

* official Objectron object pose (oracle O, diagnostic only), and
* the deployable pose+mask runtime-O estimate.

Original ReconViaGen is evaluated once per accepted selection because it does
not consume either runtime-O definition.  GPU model execution remains in the
companion shell launcher; this module owns the CPU-only identity/projection
contracts and the final cross-artifact audit.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Sequence

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import numpy as np
from PIL import Image, ImageDraw


TRACKER_ROOT = Path(__file__).resolve().parents[1]
for dependency in (TRACKER_ROOT, TRACKER_ROOT / "ReconViaGen"):
    if str(dependency) not in sys.path:
        sys.path.insert(0, str(dependency))

from ar_ss_flow.shared_object_preprocessing import canonical_json_sha256  # noqa: E402
from pose_point_depth_mv.dataset_tools.prepare_omni_real_runtime_inputs import (  # noqa: E402
    MANIFEST_FORMAT as RUNTIME_MANIFEST_FORMAT,
    MARKER_FORMAT as RUNTIME_MARKER_FORMAT,
    OBJECT_FORMAT as RUNTIME_OBJECT_FORMAT,
    build_object_runtime_input,
    pose_mask_object_spherical_farthest_frame_indices,
    pose_mask_training_spherical_farthest_frame_indices,
    runtime_input_quality_record,
)
from pose_point_depth_mv.dataset_tools.prepare_omni_real_dino_only_model_inputs import (  # noqa: E402
    FEATURE_CONTRACT,
    MANIFEST_FORMAT as MODEL_INPUT_MANIFEST_FORMAT,
    MARKER_FORMAT as MODEL_INPUT_MARKER_FORMAT,
    OBJECT_FORMAT as MODEL_INPUT_OBJECT_FORMAT,
    load_runtime_lifting_geometry,
)
from pose_point_depth_mv.dataset_tools.prepare_omni_real_video_cache import (  # noqa: E402
    OBJECT_CACHE_FORMAT,
    RAW_CACHE_FORMAT,
    sha256_file,
    write_json,
    write_npz,
)
from pose_point_depth_mv.pose_mask_object_canonicalization import (  # noqa: E402
    PoseMaskObjectFrameConfig,
)
from pose_point_depth_mv.real_object_canonicalization import (  # noqa: E402
    RuntimeObjectFrame,
    array_sha256,
    normalize_similarity_extrinsics,
    validate_proper_similarity,
)
from pose_point_depth_mv.omni_real_benchmark_common import (  # noqa: E402
    atomic_json,
    atomic_torch_save,
)


PLAN_FORMAT = "pose_point_depth_mv.objectron_ss30k_slat30k_selection_o_matrix_plan.v2"
REPORT_FORMAT = "pose_point_depth_mv.objectron_ss30k_slat30k_selection_o_matrix.v2"
GT_CENTER_PLAN_KIND = "objectron_gt_center_second_stage_guided8_v1"
GT_CENTER_PAIR_REPORT_FORMAT = (
    "pose_point_depth_mv.objectron_gt_center_second_stage_pair.v1"
)
LEGACY_PLAN_FORMAT = "pose_point_depth_mv.objectron_ss30k_slat30k_selection_o_matrix_plan.v1"
TRUE_POSE_FRONTEND_FORMAT = (
    "pose_point_depth_mv.objectron_official_object_pose_frontend.v1"
)
RANDOM_SEED = 20260819
OBJECT_CATEGORY = "objectron_camera"
OBJECT_ID = "camera_batch7_24"
OBJECT_KEY = f"{OBJECT_CATEGORY}:{OBJECT_ID}"
CLIP_SEQUENCE = "camera/batch-7/24"
OFFICIAL_OBJECT_ID = 0
EXPERIMENT_LABEL = "Objectron Camera"
ALLOW_PHONE_DIAGNOSTIC = False
GRAVITY_UP_W = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
FEATURE_RESOLUTION = 518
FOREGROUND_MARGIN = 1.10
ALPHA_THRESHOLD = 0.80
EXPECTED_OBJECT_EXTENT = 0.90
SCALE_PADDING = 1.05

DEFAULT_DATASET = (
    TRACKER_ROOT
    / "yxc/datasets/Objectron_real_pose_2clips_20260819_v1"
)
DEFAULT_OUTPUT = (
    TRACKER_ROOT
    / "pose_point_depth_mv/outputs2/"
    "ObjectronCamera_三选帧_ReconViaGen_vs_SS30K_SLat30K_双RuntimeO轮廓_20260819_v2"
)


def configure_experiment(
    *,
    clip_sequence: str,
    official_object_id: int,
    object_category: str,
    object_id: str,
    experiment_label: str,
    allow_phone_diagnostic: bool,
) -> None:
    """Configure one single-instance Objectron experiment before preparation.

    Objectron clips may contain multiple annotated instances.  The generative
    endpoint reconstructs one object, so RGB masks and oracle-O must be bound to
    one explicit official ``object_id`` instead of silently using annotation 0.
    """

    global CLIP_SEQUENCE, OFFICIAL_OBJECT_ID, OBJECT_CATEGORY, OBJECT_ID
    global OBJECT_KEY, EXPERIMENT_LABEL, ALLOW_PHONE_DIAGNOSTIC
    if not clip_sequence or clip_sequence.startswith("/") or ".." in Path(clip_sequence).parts:
        raise ValueError(f"invalid Objectron clip sequence: {clip_sequence!r}")
    if official_object_id < 0:
        raise ValueError("official Objectron object id must be nonnegative")
    if not object_category or not object_id:
        raise ValueError("object category/id must be nonempty")
    CLIP_SEQUENCE = clip_sequence.strip("/")
    OFFICIAL_OBJECT_ID = int(official_object_id)
    OBJECT_CATEGORY = object_category
    OBJECT_ID = object_id
    OBJECT_KEY = f"{OBJECT_CATEGORY}:{OBJECT_ID}"
    EXPERIMENT_LABEL = experiment_label
    ALLOW_PHONE_DIAGNOSTIC = bool(allow_phone_diagnostic)

SS30K_REPORT = Path(
    "/data/zjr/proobjaverse_official_30k_heldout_dev64_ss30k_slat30k_20260818_v1/"
    "ss30k_dev64_aggregate/report.json"
)
SS30K_CHECKPOINT = Path(
    "/data/zjr/proobjaverse_official_30k_checkpoint_archives/"
    "ProObjaverse_30K_noVGGT_SS_SLat_numbered_checkpoints_20260818_v1/"
    "ss/checkpoints/step_030000.pt"
)
SLAT30K_CHECKPOINT = Path(
    "/data/zjr/proobjaverse_official_30k_checkpoint_archives/"
    "ProObjaverse_30K_noVGGT_SS_SLat_numbered_checkpoints_20260818_v1/"
    "slat/checkpoints/step_030000.pt"
)
ABC_R_EVIDENCE = Path(
    "/data/zjr/proobjaverse_official_30k_heldout_dev64_ss30k_slat30k_20260818_v1/"
    "abc_r_dev64_aggregate/report.json"
)
STOCK_FREEZE = Path(
    "/data/zjr/native_slat_genrecon_v2_mixed1k_20260802_v1/"
    "stock_slat_freeze_v2.json"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_objectron(dataset: Path) -> dict[str, Any]:
    clip = dataset / "clips" / CLIP_SEQUENCE
    schema = dataset / "official_schema"
    sys.path.insert(0, str(schema))
    from objectron.schema import annotation_data_pb2  # pylint: disable=import-outside-toplevel

    annotation_path = clip / "annotation.pbdata"
    sequence = annotation_data_pb2.Sequence()
    sequence.ParseFromString(annotation_path.read_bytes())
    frame_paths = sorted((clip / "frames").glob("frame_*.png"))
    mask_paths = sorted((clip / "masks").glob("frame_*.png"))
    if not frame_paths or len(frame_paths) != len(mask_paths):
        raise RuntimeError("Objectron RGB/mask frame counts differ")
    expected_names = [f"frame_{index:06d}.png" for index in range(len(frame_paths))]
    if [path.name for path in frame_paths] != expected_names:
        raise RuntimeError("Objectron decoded frame names are not contiguous")
    if [path.name for path in mask_paths] != expected_names:
        raise RuntimeError("Objectron mask names do not match decoded frames")
    if len(sequence.frame_annotations) != len(frame_paths):
        raise RuntimeError("Objectron annotation and decoded frame counts differ")
    frames_manifest_path = clip / "frames_manifest.json"
    masks_manifest_path = clip / "masks_manifest.json"
    frames_manifest = _load_json(frames_manifest_path)
    masks_manifest = _load_json(masks_manifest_path)
    if frames_manifest.get("passed") is not True:
        raise RuntimeError("Objectron decoded-frame manifest did not pass")
    if masks_manifest.get("passed") is not True:
        raise RuntimeError("Objectron derived-mask manifest did not pass")
    if "selected_object_id" in masks_manifest and int(
        masks_manifest["selected_object_id"]
    ) != OFFICIAL_OBJECT_ID:
        raise RuntimeError("Objectron mask manifest is bound to a different object id")
    official_matches = [
        value for value in sequence.objects if int(value.id) == OFFICIAL_OBJECT_ID
    ]
    if len(official_matches) != 1:
        raise RuntimeError(
            f"Objectron clip must contain official object id={OFFICIAL_OBJECT_ID} "
            f"exactly once; available={[int(value.id) for value in sequence.objects]}"
        )

    # Objectron stores the ARKit view/intrinsic matrices in direct row-major
    # order.  The decoded PNG is portrait while the source camera calibration
    # is landscape-right/OpenGL (-z forward).  A swaps x/y and flips z,
    # yielding portrait OpenCV/COLMAP (+z forward) without an improper pose.
    orientation = np.asarray(
        [[0.0, 1.0, 0.0, 0.0],
         [1.0, 0.0, 0.0, 0.0],
         [0.0, 0.0, -1.0, 0.0],
         [0.0, 0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    with Image.open(frame_paths[0]) as first_image:
        width, height = first_image.size
    intrinsics = []
    poses = []
    projected_center_errors = []
    projected_center_depths = []
    official_object = official_matches[0]
    center_w = np.asarray(official_object.translation, dtype=np.float64)
    for expected_index, frame in enumerate(sequence.frame_annotations):
        if int(frame.frame_id) != expected_index:
            raise RuntimeError("Objectron annotation frame_id is not contiguous")
        camera = frame.camera
        view = np.asarray(camera.view_matrix, dtype=np.float64).reshape(4, 4)
        landscape_k = np.asarray(camera.intrinsics, dtype=np.float64).reshape(3, 3)
        pose = orientation @ view
        intrinsic = np.asarray(
            [
                [landscape_k[1, 1], 0.0, width - landscape_k[1, 2]],
                [0.0, landscape_k[0, 0], landscape_k[0, 2]],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        rotation = pose[:3, :3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=2.0e-5):
            raise RuntimeError(f"Objectron converted camera is not orthonormal: {expected_index}")
        if not np.isclose(np.linalg.det(rotation), 1.0, atol=2.0e-5):
            raise RuntimeError(f"Objectron converted camera is not proper: {expected_index}")
        point_c = (pose @ np.r_[center_w, 1.0])[:3]
        projected = intrinsic @ point_c
        projected = projected[:2] / projected[2]
        annotation_matches = [
            value
            for value in frame.annotations
            if int(value.object_id) == OFFICIAL_OBJECT_ID
        ]
        if len(annotation_matches) != 1:
            raise RuntimeError(
                f"frame={expected_index} does not contain exactly one annotation "
                f"for object id={OFFICIAL_OBJECT_ID}"
            )
        annotation = annotation_matches[0]
        gt = np.asarray(
            [
                float(annotation.keypoints[0].point_2d.x) * width,
                float(annotation.keypoints[0].point_2d.y) * height,
            ]
        )
        projected_center_errors.append(float(np.linalg.norm(projected - gt)))
        projected_center_depths.append(float(point_c[2]))
        intrinsics.append(intrinsic)
        poses.append(pose)

    errors = np.asarray(projected_center_errors, dtype=np.float64)
    depths = np.asarray(projected_center_depths, dtype=np.float64)
    if float(errors.max()) > 2.0 or float(depths.min()) <= 0.0:
        raise RuntimeError("Objectron portrait camera conversion failed projection audit")
    rotation_o2w = np.asarray(official_object.rotation, dtype=np.float64).reshape(3, 3)
    if not np.allclose(rotation_o2w.T @ rotation_o2w, np.eye(3), atol=1.0e-5):
        raise RuntimeError("Objectron official object rotation is not orthonormal")
    if not np.isclose(np.linalg.det(rotation_o2w), 1.0, atol=1.0e-5):
        raise RuntimeError("Objectron official object rotation is not proper")
    scale_m = np.asarray(official_object.scale, dtype=np.float64)
    if scale_m.shape != (3,) or np.any(scale_m <= 0.0):
        raise RuntimeError("Objectron official object scale is invalid")

    return {
        "dataset": dataset,
        "clip": clip,
        "frames_dir": clip / "frames",
        "masks_dir": clip / "masks",
        "frame_names": expected_names,
        "intrinsics": np.asarray(intrinsics, dtype=np.float64),
        "T_W2C": np.asarray(poses, dtype=np.float64),
        "object_translation_W": center_w,
        "object_rotation_O2W": rotation_o2w,
        "object_scale_m": scale_m,
        "annotation_path": annotation_path,
        "frames_manifest_path": frames_manifest_path,
        "masks_manifest_path": masks_manifest_path,
        "official_object_id": OFFICIAL_OBJECT_ID,
        "official_object_category": str(official_object.category),
        "official_sequence_object_count": len(sequence.objects),
        "camera_conversion_audit": {
            "source": "official Objectron annotation camera matrices",
            "decoded_png_resolution_wh": [width, height],
            "source_convention": "landscape-right ARKit/OpenGL -z forward",
            "runtime_convention": "portrait OpenCV/COLMAP +z forward",
            "formula": "T_W2C_portrait_cv=A@view_matrix; K axes/principal point remapped",
            "A": orientation.tolist(),
            "object_center_projection_error_pixels": {
                "min": float(errors.min()),
                "median": float(np.median(errors)),
                "p95": float(np.quantile(errors, 0.95)),
                "max": float(errors.max()),
            },
            "object_center_depth_m": {
                "min": float(depths.min()),
                "max": float(depths.max()),
            },
            "passed": True,
        },
    }


def _circular_stats(angles: np.ndarray) -> tuple[float, float, float]:
    ordered = np.sort(np.asarray(angles, dtype=np.float64) % 360.0)
    gaps = np.diff(np.r_[ordered, ordered[0] + 360.0])
    maximum = float(gaps.max())
    return maximum, float(360.0 - maximum), float(gaps.std())


def _selection_geometry(source: dict[str, Any], indices: Sequence[int]) -> dict[str, Any]:
    chosen = np.asarray(indices, dtype=np.int64)
    poses = source["T_W2C"]
    center = source["object_translation_W"]
    camera_centers = np.linalg.inv(poses)[:, :3, 3]
    offsets = camera_centers - center[None]
    radii = np.linalg.norm(offsets, axis=1)
    directions = offsets / np.maximum(radii[:, None], 1.0e-12)
    gravity = GRAVITY_UP_W / np.linalg.norm(GRAVITY_UP_W)
    basis_x = np.asarray([1.0, 0.0, 0.0])
    basis_x -= np.dot(basis_x, gravity) * gravity
    basis_x /= np.linalg.norm(basis_x)
    basis_z = np.cross(gravity, basis_x)
    basis_z /= np.linalg.norm(basis_z)
    azimuths = np.degrees(np.arctan2(offsets @ basis_x, offsets @ basis_z)) % 360.0
    elevations = np.degrees(np.arcsin(np.clip(directions @ gravity, -1.0, 1.0)))
    selected_directions = directions[chosen]
    pairwise = np.degrees(
        np.arccos(np.clip(selected_directions @ selected_directions.T, -1.0, 1.0))
    )
    np.fill_diagonal(pairwise, np.inf)
    nearest = pairwise.min(axis=1)
    maximum_gap, coverage, gap_std = _circular_stats(azimuths[chosen])
    names = source["frame_names"]
    return {
        "object_center_W": center.tolist(),
        "camera_radius_m": {
            "min": float(radii[chosen].min()),
            "median": float(np.median(radii[chosen])),
            "max": float(radii[chosen].max()),
        },
        "selected_camera_direction_W_by_frame": {
            names[int(index)]: directions[int(index)].tolist() for index in chosen
        },
        "selected_azimuth_degrees_by_frame": {
            names[int(index)]: float(azimuths[int(index)]) for index in chosen
        },
        "selected_elevation_degrees_by_frame": {
            names[int(index)]: float(elevations[int(index)]) for index in chosen
        },
        "minimum_pairwise_angular_separation_degrees": float(nearest.min()),
        "mean_nearest_angular_separation_degrees": float(nearest.mean()),
        "azimuth_coverage_degrees": coverage,
        "maximum_azimuth_gap_degrees": maximum_gap,
        "azimuth_gap_std_degrees": gap_std,
    }


def _selection_record(
    source: dict[str, Any], policy: str, indices: Sequence[int], **extra: Any
) -> dict[str, Any]:
    chosen = [int(value) for value in indices]
    names = source["frame_names"]
    return {
        "policy": policy,
        "selected_view_count": len(chosen),
        "selected_source_global_indices": chosen,
        "selected_frame_names": [names[index] for index in chosen],
        **_selection_geometry(source, chosen),
        **extra,
    }


def _write_contact_sheet(
    source: dict[str, Any], indices: Sequence[int], output: Path, title: str
) -> None:
    selected_dir = output / "selected_frames"
    masks_dir = output / "selected_masks"
    selected_dir.mkdir(parents=True, exist_ok=False)
    masks_dir.mkdir(parents=True, exist_ok=False)
    cells: list[Image.Image] = []
    names = source["frame_names"]
    for slot, index in enumerate(indices):
        name = names[int(index)]
        image_path = source["frames_dir"] / name
        mask_path = source["masks_dir"] / name
        shutil.copy2(image_path, selected_dir / f"view_{slot:02d}_{name}")
        shutil.copy2(mask_path, masks_dir / f"view_{slot:02d}_{name}")
        image = Image.open(image_path).convert("RGB")
        image.thumbnail((300, 400), Image.Resampling.LANCZOS)
        cell = Image.new("RGB", (320, 445), (22, 22, 22))
        cell.paste(image, ((320 - image.width) // 2, 34))
        draw = ImageDraw.Draw(cell)
        draw.text((8, 8), f"view {slot}: source frame {int(index)}", fill=(245, 245, 245))
        cells.append(cell)
    sheet = Image.new("RGB", (1280, 920), (12, 12, 12))
    draw = ImageDraw.Draw(sheet)
    draw.text((10, 5), title, fill=(255, 255, 255))
    for position, cell in enumerate(cells):
        x = (position % 4) * 320
        y = 30 + (position // 4) * 445
        sheet.paste(cell, (x, y))
    sheet.save(output / "selected_8_frames_contact_sheet.png")


def _raw_subset(
    source: dict[str, Any], indices: Sequence[int], output: Path
) -> tuple[Path, dict[str, Any]]:
    root = output / "raw_cache"
    object_dir = root / "objects" / OBJECT_CATEGORY / OBJECT_ID
    cache_path = object_dir / "raw_cache.npz"
    chosen = np.asarray(indices, dtype=np.int64)
    names = [source["frame_names"][int(index)] for index in chosen]
    write_npz(
        cache_path,
        frame_name=np.asarray(names),
        K=np.asarray(source["intrinsics"][chosen], dtype=np.float64),
        T_W2C=np.asarray(source["T_W2C"][chosen], dtype=np.float64),
        P_W=np.empty((0, 3), dtype=np.float64),
    )
    cameras = [
        {
            "frame_name": name,
            "model": "PINHOLE",
            "width": 1440,
            "height": 1920,
            "distortion": [],
        }
        for name in names
    ]
    row = {
        "format": OBJECT_CACHE_FORMAT,
        "category": OBJECT_CATEGORY,
        "object_id": OBJECT_ID,
        "cache_npz": str(cache_path.resolve()),
        "cache_npz_sha256": sha256_file(cache_path),
        "images_dir": str(source["frames_dir"].resolve()),
        "masks_dir": str(source["masks_dir"].resolve()),
        "frame_count": len(names),
        "registered_frame_count": len(names),
        "cameras": cameras,
        "passed": True,
    }
    report = {
        "format": RAW_CACHE_FORMAT,
        "created_at_utc": utc_now(),
        "object_count": 1,
        "category_count": 1,
        "objects": [row],
        "source": (
            f"frozen official Objectron {CLIP_SEQUENCE} object_id="
            f"{OFFICIAL_OBJECT_ID} plus derived SAM2 masks"
        ),
        "annotation": str(source["annotation_path"].resolve()),
        "annotation_sha256": sha256_file(source["annotation_path"]),
        "frames_manifest": str(source["frames_manifest_path"].resolve()),
        "frames_manifest_sha256": sha256_file(source["frames_manifest_path"]),
        "masks_manifest": str(source["masks_manifest_path"].resolve()),
        "masks_manifest_sha256": sha256_file(source["masks_manifest_path"]),
        "point_cloud_consumed": False,
        "alignment_passed": False,
        "training_ready": False,
        "scope_guard": "Real qualitative input cache; no target Mesh or metric is consumed.",
        "passed": True,
    }
    report_path = root / "raw_cache_report.json"
    write_json(report_path, report)
    return report_path, row


def _runtime_manifest(
    output: Path,
    report: dict[str, Any],
    raw_report: Path,
    build_config: dict[str, Any],
) -> Path:
    build_hash = canonical_json_sha256(build_config)
    payload = {
        "format": RUNTIME_MANIFEST_FORMAT,
        "created_at_utc": utc_now(),
        "raw_cache_report": str(raw_report.resolve()),
        "raw_cache_report_sha256": sha256_file(raw_report),
        "source_raw_cache_report": str(raw_report.resolve()),
        "source_raw_cache_report_sha256": sha256_file(raw_report),
        "build_config": build_config,
        "build_config_sha256": build_hash,
        "source_selected_object_count": 1,
        "selected_object_count": 1,
        "completed_object_count": 1,
        "reused_objects": [],
        "objects": [report],
        "quality_rejections": [],
        "eligible_raw_cache_report": None,
        "failures": [],
        "training_ready": False,
        "scope_guard": "Objectron qualitative input-only runtime-O cache.",
        "passed": True,
    }
    path = output / "runtime_input_manifest.json"
    write_json(path, payload)
    return path


def _build_pose_mask_runtime(
    source: dict[str, Any],
    strategy_root: Path,
    raw_report: Path,
    raw_row: dict[str, Any],
    selection: dict[str, Any],
) -> tuple[Path, dict[str, Any]]:
    output = strategy_root / "01_runtime_pose_mask"
    config = PoseMaskObjectFrameConfig(
        mask_threshold=0.5,
        expected_object_extent=EXPECTED_OBJECT_EXTENT,
        scale_padding=SCALE_PADDING,
    )
    config.validate()
    build_config = {
        "experiment": PLAN_FORMAT,
        "o_definition": "pose+mask estimated runtime-O",
        "selected_frame_names": selection["selected_frame_names"],
        "selected_source_global_indices": selection["selected_source_global_indices"],
        "feature_resolution": FEATURE_RESOLUTION,
        "foreground_margin": FOREGROUND_MARGIN,
        "alpha_threshold": ALPHA_THRESHOLD,
        "frame_config": asdict(config),
        "gravity_up_W": GRAVITY_UP_W.tolist(),
    }
    build_hash = canonical_json_sha256(build_config)
    report, reused = build_object_runtime_input(
        raw_row,
        output_dir=output,
        selected_view_count=8,
        feature_resolution=FEATURE_RESOLUTION,
        foreground_margin=FOREGROUND_MARGIN,
        alpha_threshold=ALPHA_THRESHOLD,
        view_selection_policy="lexical_even",
        geometry_mode="pose_mask",
        resume_partial=False,
        frame_config=config,
        build_config_sha256=build_hash,
        gravity_up_w=GRAVITY_UP_W,
    )
    if reused:
        raise RuntimeError("fresh Objectron experiment unexpectedly reused pose-mask runtime")
    quality_selection = {
        "azimuth_coverage_degrees": selection["azimuth_coverage_degrees"],
        "maximum_azimuth_gap_degrees": selection["maximum_azimuth_gap_degrees"],
    }
    quality = runtime_input_quality_record(
        frame_stats=report["runtime_frame_stats"],
        T_W2C=source["T_W2C"][selection["selected_source_global_indices"]],
        view_selection=quality_selection,
        gravity_up_w=GRAVITY_UP_W,
        geometry_mode="pose_mask",
    )
    report.update(
        {
            "experimental_view_selection": selection,
            "selected_source_global_indices": selection[
                "selected_source_global_indices"
            ],
            "input_quality": quality,
            "formal_input_passed": quality["formal_input_passed"],
            "o_definition": "pose+mask estimated runtime-O",
            "oracle_object_pose_consumed": False,
        }
    )
    report_path = Path(report["cache_npz"]).parent / "report.json"
    write_json(report_path, report)
    manifest = _runtime_manifest(output, report, raw_report, build_config)
    return manifest, report


def _build_true_pose_runtime(
    source: dict[str, Any],
    strategy_root: Path,
    raw_report: Path,
    pose_report: dict[str, Any],
    selection: dict[str, Any],
) -> tuple[Path, dict[str, Any]]:
    output = strategy_root / "02_runtime_true_object_pose"
    destination = output / "objects" / OBJECT_CATEGORY / OBJECT_ID
    destination.mkdir(parents=True, exist_ok=False)
    pose_cache = Path(pose_report["cache_npz"])
    with np.load(pose_cache, allow_pickle=False) as payload:
        arrays = {name: np.asarray(payload[name]) for name in payload.files}
    raw_cache = Path(pose_report["source_raw_cache"])
    with np.load(raw_cache, allow_pickle=False) as payload:
        selected = np.asarray(arrays["selected_source_view_index"], dtype=np.int64)
        T_W2C = np.asarray(payload["T_W2C"][selected], dtype=np.float64)
    scale_m_per_o = (
        float(np.max(source["object_scale_m"]))
        * SCALE_PADDING
        / EXPECTED_OBJECT_EXTENT
    )
    T_O2W = np.eye(4, dtype=np.float64)
    T_O2W[:3, :3] = source["object_rotation_O2W"] * scale_m_per_o
    T_O2W[:3, 3] = source["object_translation_W"]
    validate_proper_similarity(T_O2W, name="Objectron official T_O2W")
    T_W2O = np.linalg.inv(T_O2W)
    T_O2C = np.matmul(T_W2C, T_O2W[None])
    T_C2O = np.linalg.inv(T_O2C)
    T_O2C_lifting = normalize_similarity_extrinsics(T_O2C)
    empty_points = np.empty((0, 3), dtype=np.float64)
    empty_keep = np.empty((0,), dtype=bool)
    frame = RuntimeObjectFrame(
        T_O2W=T_O2W,
        T_W2O=T_W2O,
        T_O2C=T_O2C,
        T_C2O=T_C2O,
        P_O=empty_points,
        point_keep_mask=empty_keep,
        stats={
            "axes": {
                "reference_view_index": int(pose_report["reference_view_index"]),
                "source": "official Objectron object rotation",
            },
            "official_object_translation_W": source["object_translation_W"].tolist(),
            "official_object_rotation_O2W": source["object_rotation_O2W"].tolist(),
            "official_object_scale_m": source["object_scale_m"].tolist(),
            "normalization_scale_m_per_runtime_o_unit": scale_m_per_o,
            "expected_object_extent": EXPECTED_OBJECT_EXTENT,
            "scale_padding": SCALE_PADDING,
        },
        contract={
            "format": TRUE_POSE_FRONTEND_FORMAT,
            "coordinate_convention": "portrait OpenCV/COLMAP +z T_W2C",
            "object_pose_source": "official Objectron annotation object rotation+translation",
            "object_scale_source": "official Objectron metric cuboid dimensions",
            "normalization_rule": "max(object_scale_m)*scale_padding/expected_object_extent",
            "oracle_pose_diagnostic_only": True,
            "point_cloud_consumed": False,
            "mask_consumed_for_object_pose": False,
        },
    )
    arrays.update(
        {
            "T_O2C": T_O2C,
            "T_O2C_lifting": T_O2C_lifting,
            "T_C2O": T_C2O,
            "T_O2W": T_O2W,
            "T_W2O": T_W2O,
            "P_O": empty_points.astype(np.float32),
            "object_point_source_index": np.empty((0,), dtype=np.int64),
        }
    )
    cache_path = destination / "runtime_input_cache.npz"
    write_npz(cache_path, **arrays)
    pose_condition = _load_json(Path(pose_report["condition_record"]))
    condition = {
        "format": TRUE_POSE_FRONTEND_FORMAT,
        "runtime_frame": frame.record(),
        "shared_image_geometry": pose_condition["shared_image_geometry"],
        "undistortion": pose_condition["undistortion"],
        "prepared_rgb_sha256": pose_condition["prepared_rgb_sha256"],
        "prepared_mask_sha256": pose_condition["prepared_mask_sha256"],
        "K_feature_sha256": pose_condition["K_feature_sha256"],
        "T_O2C_sha256": array_sha256(T_O2C),
        "T_O2C_lifting_sha256": array_sha256(T_O2C_lifting),
        "P_O_sha256": array_sha256(empty_points),
        "point_cloud_consumed": False,
        "oracle_object_pose_consumed": True,
        "condition_scope": (
            "same RGB/mask/preprocessing as pose-mask branch; O is replaced only "
            "by frozen official Objectron object pose and metric scale"
        ),
    }
    condition["condition_sha256"] = canonical_json_sha256(condition)
    condition_path = destination / "condition_record.json"
    write_json(condition_path, condition)
    build_config = {
        "experiment": PLAN_FORMAT,
        "o_definition": "official Objectron true object pose (oracle diagnostic)",
        "selected_frame_names": selection["selected_frame_names"],
        "selected_source_global_indices": selection["selected_source_global_indices"],
        "feature_resolution": FEATURE_RESOLUTION,
        "foreground_margin": FOREGROUND_MARGIN,
        "alpha_threshold": ALPHA_THRESHOLD,
        "expected_object_extent": EXPECTED_OBJECT_EXTENT,
        "scale_padding": SCALE_PADDING,
        "gravity_up_W": GRAVITY_UP_W.tolist(),
    }
    build_hash = canonical_json_sha256(build_config)
    report = {
        "format": RUNTIME_OBJECT_FORMAT,
        "created_at_utc": utc_now(),
        "category": OBJECT_CATEGORY,
        "object_id": OBJECT_ID,
        "object_key": OBJECT_KEY,
        "source_raw_cache": str(raw_cache.resolve()),
        "source_raw_cache_sha256": sha256_file(raw_cache),
        "build_config_sha256": build_hash,
        "input_frontend_format": TRUE_POSE_FRONTEND_FORMAT,
        "geometry_mode": "official_true_object_pose_oracle",
        "point_cloud_consumed": False,
        "selected_view_count": 8,
        "selected_source_view_indices": selected.tolist(),
        "selected_source_global_indices": selection[
            "selected_source_global_indices"
        ],
        "selected_frame_names": arrays["frame_name"].astype(str).tolist(),
        "view_selection": selection,
        "experimental_view_selection": selection,
        "reference_view_index": int(pose_report["reference_view_index"]),
        "cache_npz": str(cache_path.resolve()),
        "condition_record": str(condition_path.resolve()),
        "condition_sha256": condition["condition_sha256"],
        "T_O2C_sha256": array_sha256(T_O2C),
        "T_O2C_lifting_sha256": array_sha256(T_O2C_lifting),
        "lifting_extrinsics_policy": (
            "physical T_O2C retained for audit; projectively normalized "
            "T_O2C_lifting is exported to Native v2 lifting"
        ),
        "prepared_rgb_paths": list(pose_report["prepared_rgb_paths"]),
        "prepared_mask_paths": list(pose_report["prepared_mask_paths"]),
        "runtime_frame_stats": frame.stats,
        "input_quality": pose_report["input_quality"],
        "formal_input_passed": pose_report["formal_input_passed"],
        "o_definition": "official Objectron true object pose",
        "oracle_object_pose_consumed": True,
        "forbidden_gt_fields_absent": False,
        "training_ready": False,
        "scope_guard": (
            "Oracle object-pose diagnostic only. It is paired to exactly the same "
            "selected RGB/mask and preprocessing as the pose-mask runtime-O branch."
        ),
        "passed": True,
    }
    report_path = destination / "report.json"
    write_json(report_path, report)
    write_json(
        destination / "_RUNTIME_INPUT_COMPLETE.json",
        {
            "format": RUNTIME_MARKER_FORMAT,
            "completed_at_utc": utc_now(),
            "object_key": OBJECT_KEY,
            "source_cache_sha256": sha256_file(raw_cache),
            "build_config_sha256": build_hash,
            "condition_sha256": condition["condition_sha256"],
            "T_O2C_lifting_sha256": array_sha256(T_O2C_lifting),
        },
    )
    manifest = _runtime_manifest(output, report, raw_report, build_config)
    return manifest, report


def _phone_selection(source: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    names = source["frame_names"]
    with Image.open(source["frames_dir"] / names[0]) as first_image:
        source_width, source_height = first_image.size
    selection_width = 360
    selection_height = int(round(source_height * selection_width / source_width))
    scale_x = selection_width / source_width
    scale_y = selection_height / source_height
    scale_matrix = np.asarray(
        [[scale_x, 0.0, 0.0], [0.0, scale_y, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    selection_intrinsics = np.matmul(scale_matrix[None], source["intrinsics"])
    camera_by_name = {
        name: {
            "frame_name": name,
            "model": "PINHOLE",
            "width": selection_width,
            "height": selection_height,
            "distortion": [],
        }
        for name in names
    }
    with tempfile.TemporaryDirectory(prefix="objectron_phone_masks_") as temporary:
        selection_masks = Path(temporary)
        for name in names:
            with Image.open(source["masks_dir"] / name) as handle:
                resized = handle.convert("L").resize(
                    (selection_width, selection_height), Image.Resampling.NEAREST
                )
            resized.save(selection_masks / name)
        selected, record = pose_mask_object_spherical_farthest_frame_indices(
            names,
            selection_masks,
            8,
            alpha_threshold=ALPHA_THRESHOLD,
            intrinsics=selection_intrinsics,
            T_W2C=source["T_W2C"],
            camera_by_name=camera_by_name,
            gravity_up_w=GRAVITY_UP_W,
            frame_config=PoseMaskObjectFrameConfig(
                mask_threshold=0.5,
                expected_object_extent=EXPECTED_OBJECT_EXTENT,
                scale_padding=SCALE_PADDING,
            ),
            selection_identity=f"objectron:{CLIP_SEQUENCE}:object{OFFICIAL_OBJECT_ID}",
        )
    record.update(
        {
            "requested_policy": "手机采集程序 object-spherical-farthest 8视图",
            "selection_geometry_resolution": [selection_width, selection_height],
            "source_mask_resolution": [source_width, source_height],
            "selection_geometry_resize": (
                "coordinate-equivalent nearest-neighbor mask resize with K left-multiplied "
                "by diag(scale_x,scale_y,1); final runtime-O uses original resolution"
            ),
            "selected_source_global_indices": selected.astype(int).tolist(),
            "selected_frame_names": [names[int(index)] for index in selected],
        }
    )
    return selected, record


def _gt_center_training_selection(
    source: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Select the official-training FPS subset around Objectron's GT center.

    Only the annotated object translation is exposed to the selector.  Object
    rotation and scale are deliberately excluded so this arm models the second
    capture stage where an object centre has already been estimated, rather
    than granting the deployable branch the full oracle object frame.
    """

    names = source["frame_names"]
    with Image.open(source["frames_dir"] / names[0]) as first_image:
        width, height = first_image.size
    camera_by_name = {
        name: {
            "frame_name": name,
            "model": "PINHOLE",
            "width": width,
            "height": height,
            "distortion": [],
        }
        for name in names
    }
    fps_order, record = pose_mask_training_spherical_farthest_frame_indices(
        names,
        source["masks_dir"],
        8,
        alpha_threshold=ALPHA_THRESHOLD,
        intrinsics=source["intrinsics"],
        T_W2C=source["T_W2C"],
        camera_by_name=camera_by_name,
        gravity_up_w=GRAVITY_UP_W,
        selection_identity=(
            f"objectron_gt_center:{CLIP_SEQUENCE}:object{OFFICIAL_OBJECT_ID}"
        ),
        object_center_w=source["object_translation_W"],
    )
    # The official policy defines the selected set.  Runtime execution is kept
    # in chronological source order, matching the established Objectron matrix
    # and making paired image-context audits unambiguous.
    fps_order = np.asarray(fps_order, dtype=np.int64)
    selected = np.sort(fps_order)
    geometry = _selection_geometry(source, selected)
    candidate_geometry = _selection_geometry(
        source, np.arange(len(names), dtype=np.int64)
    )
    expected_center = np.asarray(source["object_translation_W"], dtype=np.float64)
    if not np.allclose(
        np.asarray(record["object_center_W"], dtype=np.float64),
        expected_center,
        atol=1.0e-10,
        rtol=0.0,
    ):
        raise RuntimeError("GT-centre selector did not retain the Objectron centre")
    record.update(
        {
            "policy": "objectron_gt_center_training_spherical_fps8",
            "algorithm": "official_training_single_seed_spherical_fps_v1",
            "object_center_source": (
                "official_objectron_sequence_object_translation_W"
            ),
            "object_center_W": expected_center.tolist(),
            "selected_view_count": 8,
            "selected_source_view_indices": selected.astype(int).tolist(),
            "selected_source_global_indices": selected.astype(int).tolist(),
            "selected_frame_names": [names[int(index)] for index in selected],
            "raw_fps_selection_order_indices": fps_order.astype(int).tolist(),
            "raw_fps_selection_order_frame_names": [
                names[int(index)] for index in fps_order
            ],
            "execution_order": "selected frame indices sorted chronologically",
            "oracle_fields_consumed_for_selection": ["object.translation"],
            "object_rotation_consumed_for_selection": False,
            "object_scale_consumed_for_selection": False,
            "mask_role": "nonempty_foreground_validity_only",
            "quality_gate_used_for_selection": False,
            "candidate_azimuth_coverage_degrees": candidate_geometry[
                "azimuth_coverage_degrees"
            ],
            "candidate_maximum_azimuth_gap_degrees": candidate_geometry[
                "maximum_azimuth_gap_degrees"
            ],
            "selected_over_candidate_azimuth_coverage_ratio": (
                float(geometry["azimuth_coverage_degrees"])
                / max(
                    float(candidate_geometry["azimuth_coverage_degrees"]),
                    1.0e-12,
                )
            ),
            "second_stage_simulation": (
                "known object centre followed by training-consistent spherical FPS8"
            ),
            **geometry,
        }
    )
    return selected, record


def prepare(dataset: Path, output: Path) -> None:
    dataset = dataset.resolve(strict=True)
    output = output.resolve()
    if output.exists():
        raise FileExistsError(output)
    for path in (
        SS30K_REPORT,
        SS30K_CHECKPOINT,
        SLAT30K_CHECKPOINT,
        ABC_R_EVIDENCE,
        STOCK_FREEZE,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    source = _parse_objectron(dataset)
    output.mkdir(parents=True, exist_ok=False)

    frame_count = len(source["frame_names"])
    uniform = np.rint(np.linspace(0, frame_count - 1, 8)).astype(np.int64)
    random_draw = np.random.default_rng(RANDOM_SEED).choice(
        frame_count, size=8, replace=False
    )
    random_selected = np.sort(random_draw.astype(np.int64))
    phone_selected, phone_record = _phone_selection(source)
    phone_selected = np.sort(phone_selected.astype(np.int64))
    phone_record.update(
        {
            "selected_source_global_indices": phone_selected.astype(int).tolist(),
            "selected_frame_names": [
                source["frame_names"][int(index)] for index in phone_selected
            ],
            "execution_order": "selected frame indices sorted chronologically",
        }
    )
    phone_quality = phone_record["selected_segment_final_view_quality"]
    phone_passed = bool(phone_quality.get("formal_input_passed"))
    phone_active = phone_passed or ALLOW_PHONE_DIAGNOSTIC

    specifications = [
        (
            "01_time_uniform8",
            "时间轴均匀8帧",
            uniform,
            _selection_record(
                source,
                "time_axis_uniform_8_including_endpoints_v1",
                uniform,
                deterministic=True,
            ),
            True,
            None,
        ),
        (
            "02_phone_spherical8",
            "手机采集程序球面最远点8帧",
            phone_selected,
            phone_record,
            phone_active,
            None
            if phone_passed
            else (
                "手机采集正式 final-8 门未通过；以显式诊断模式继续重建。"
                if ALLOW_PHONE_DIAGNOSTIC
                else "手机采集正式 final-8 门未通过；按注册规则跳过模型和 Mesh。"
            ),
        ),
        (
            "03_random8_seed20260819",
            "固定种子完全随机8帧",
            random_selected,
            _selection_record(
                source,
                "uniform_random_without_replacement_8_v1",
                random_selected,
                random_seed=RANDOM_SEED,
                raw_random_draw_order=random_draw.astype(int).tolist(),
                execution_order="selected frame indices sorted chronologically after draw",
            ),
            True,
            None,
        ),
    ]

    rows = []
    for slug, label, selected, selection, active, skip_reason in specifications:
        strategy_root = output / slug
        selected_root = strategy_root / "00_selected_input"
        selected_root.mkdir(parents=True, exist_ok=False)
        write_json(selected_root / "selection.json", selection)
        _write_contact_sheet(source, selected, selected_root, label)
        raw_report, raw_row = _raw_subset(source, selected, selected_root)
        row: dict[str, Any] = {
            "slug": slug,
            "label_zh": label,
            "active": bool(active),
            "formal_selection_passed": bool(
                phone_passed if slug == "02_phone_spherical8" else True
            ),
            "diagnostic_override": bool(
                slug == "02_phone_spherical8" and active and not phone_passed
            ),
            "skip_reason": skip_reason,
            "selected_source_global_indices": [int(value) for value in selected],
            "selected_frame_names": [
                source["frame_names"][int(value)] for value in selected
            ],
            "selection": selection,
            "selection_report": str((selected_root / "selection.json").resolve()),
            "selection_contact_sheet": str(
                (selected_root / "selected_8_frames_contact_sheet.png").resolve()
            ),
            "raw_cache_report": str(raw_report.resolve()),
        }
        if active:
            pose_manifest, pose_report = _build_pose_mask_runtime(
                source, strategy_root, raw_report, raw_row, selection
            )
            true_manifest, true_report = _build_true_pose_runtime(
                source, strategy_root, raw_report, pose_report, selection
            )
            if pose_report["selected_frame_names"] != true_report["selected_frame_names"]:
                raise RuntimeError("paired O branches do not share exact selected views")
            if pose_report["prepared_rgb_paths"] != true_report["prepared_rgb_paths"]:
                raise RuntimeError("paired O branches do not share exact prepared RGB")
            row.update(
                {
                    "runtime_pose_mask_manifest": str(pose_manifest.resolve()),
                    "runtime_true_pose_manifest": str(true_manifest.resolve()),
                    "pose_mask_formal_input_passed": bool(
                        pose_report["formal_input_passed"]
                    ),
                    "gpu_outputs": {
                        "model_input_pose_mask": str(
                            (strategy_root / "03_model_input_pose_mask").resolve()
                        ),
                        "model_input_true_pose": str(
                            (strategy_root / "04_model_input_true_pose").resolve()
                        ),
                        "current_pose_mask": str(
                            (strategy_root / "05_current_pose_mask").resolve()
                        ),
                        "current_true_pose": str(
                            (strategy_root / "06_current_true_pose").resolve()
                        ),
                        "reconviagen_once": str(
                            (strategy_root / "07_reconviagen_once").resolve()
                        ),
                        "contours_pose_mask": str(
                            (strategy_root / "08_contours_pose_mask").resolve()
                        ),
                        "contours_true_pose": str(
                            (strategy_root / "09_contours_true_pose").resolve()
                        ),
                    },
                }
            )
        else:
            write_json(
                strategy_root / "SKIPPED.json",
                {
                    "format": PLAN_FORMAT,
                    "created_at_utc": utc_now(),
                    "strategy": slug,
                    "reason": skip_reason,
                    "selected_final8_quality": phone_quality,
                    "user_contract": "不符合手机采集正式要求时允许跳过",
                    "passed": True,
                },
            )
        rows.append(row)

    plan = {
        "format": PLAN_FORMAT,
        "created_at_utc": utc_now(),
        "passed": True,
        "dataset": str(dataset),
        "clip": str(source["clip"]),
        "clip_sequence": CLIP_SEQUENCE,
        "frame_count": frame_count,
        "object_key": OBJECT_KEY,
        "experiment_label": EXPERIMENT_LABEL,
        "official_object_id": OFFICIAL_OBJECT_ID,
        "official_object_category": source["official_object_category"],
        "official_sequence_object_count": source["official_sequence_object_count"],
        "camera_conversion_audit": source["camera_conversion_audit"],
        "derived_input_manifests": {
            "frames": str(source["frames_manifest_path"].resolve()),
            "frames_sha256": sha256_file(source["frames_manifest_path"]),
            "masks": str(source["masks_manifest_path"].resolve()),
            "masks_sha256": sha256_file(source["masks_manifest_path"]),
            "masks_are_official_gt": False,
        },
        "official_object_pose": {
            "object_id": OFFICIAL_OBJECT_ID,
            "category": source["official_object_category"],
            "translation_W_m": source["object_translation_W"].tolist(),
            "rotation_O2W": source["object_rotation_O2W"].tolist(),
            "scale_m": source["object_scale_m"].tolist(),
            "annotation": str(source["annotation_path"].resolve()),
            "annotation_sha256": sha256_file(source["annotation_path"]),
        },
        "models": {
            "current": (
                "official no-VGGT posed-DINO SS30K step30000 + "
                "SLat30K step30000 (route C)"
            ),
            "native_ss_report": str(SS30K_REPORT),
            "native_ss_report_sha256": sha256_file(SS30K_REPORT),
            "native_ss_checkpoint": str(SS30K_CHECKPOINT),
            "native_ss_checkpoint_sha256": sha256_file(SS30K_CHECKPOINT),
            "native_ss_checkpoint_step": 30000,
            "native_slat_checkpoint": str(SLAT30K_CHECKPOINT),
            "native_slat_checkpoint_sha256": sha256_file(SLAT30K_CHECKPOINT),
            "native_slat_checkpoint_step": 30000,
            "cross_deployment_abc_r_evidence": str(ABC_R_EVIDENCE),
            "cross_deployment_abc_r_evidence_sha256": sha256_file(ABC_R_EVIDENCE),
            "input_context": "DINO-only; VGGT not loaded or executed",
            "stock_slat_freeze": str(STOCK_FREEZE),
            "stock_slat_freeze_sha256": sha256_file(STOCK_FREEZE),
            "reconviagen": "Stable-X/trellis-vggt-v0-2 original endpoint",
            "seed": 42,
        },
        "o_matrix": {
            "current": [
                "official Objectron true object pose O (oracle diagnostic)",
                "pose+mask estimated runtime-O (deployable input diagnostic)",
            ],
            "reconviagen": "once per accepted selection; does not consume runtime-O",
        },
        "strategies": rows,
        "scope_guard": (
            "Qualitative real-sequence diagnostic. Objectron SAM2 masks are derived, "
            "not official GT masks. True-O consumes official object pose and is oracle-only."
        ),
    }
    plan["plan_identity"] = _canonical_sha256(
        {key: value for key, value in plan.items() if key != "created_at_utc"}
    )
    write_json(output / "experiment_plan.json", plan)
    print(
        json.dumps(
            {
                "passed": True,
                "output": str(output),
                "active_strategies": [row["slug"] for row in rows if row["active"]],
                "skipped_strategies": [row["slug"] for row in rows if not row["active"]],
                "phone_formal_input_passed": phone_passed,
                "plan": str(output / "experiment_plan.json"),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


def prepare_gt_center(dataset: Path, output: Path) -> None:
    """Prepare one centre-known, training-consistent eight-view experiment."""

    dataset = dataset.resolve(strict=True)
    output = output.resolve()
    if output.exists():
        raise FileExistsError(output)
    for path in (
        SS30K_REPORT,
        SS30K_CHECKPOINT,
        SLAT30K_CHECKPOINT,
        ABC_R_EVIDENCE,
        STOCK_FREEZE,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    source = _parse_objectron(dataset)
    selected, selection = _gt_center_training_selection(source)
    if len(selected) != 8 or len(np.unique(selected)) != 8:
        raise RuntimeError("GT-centre selection did not produce eight unique views")

    output.mkdir(parents=True, exist_ok=False)
    slug = "01_gt_center_training_spherical_fps8"
    label = "Objectron真值中心引导的训练一致球面FPS8"
    strategy_root = output / slug
    selected_root = strategy_root / "00_selected_input"
    selected_root.mkdir(parents=True, exist_ok=False)
    write_json(selected_root / "selection.json", selection)
    _write_contact_sheet(source, selected, selected_root, label)
    raw_report, raw_row = _raw_subset(source, selected, selected_root)
    pose_manifest, pose_report = _build_pose_mask_runtime(
        source, strategy_root, raw_report, raw_row, selection
    )
    true_manifest, true_report = _build_true_pose_runtime(
        source, strategy_root, raw_report, pose_report, selection
    )
    if pose_report["selected_frame_names"] != true_report["selected_frame_names"]:
        raise RuntimeError("paired O branches do not share exact selected views")
    if pose_report["prepared_rgb_paths"] != true_report["prepared_rgb_paths"]:
        raise RuntimeError("paired O branches do not share exact prepared RGB")

    row = {
        "slug": slug,
        "label_zh": label,
        "active": True,
        "selector_contract_passed": True,
        "formal_selection_passed": False,
        "diagnostic_override": True,
        "formal_selection_reason": (
            "oracle GT-centre selection is diagnostic-only; "
            f"pose_mask_runtime_formal_input_passed={bool(pose_report['formal_input_passed'])}"
        ),
        "skip_reason": None,
        "selected_source_global_indices": selected.astype(int).tolist(),
        "selected_frame_names": [
            source["frame_names"][int(value)] for value in selected
        ],
        "selection": selection,
        "selection_report": str((selected_root / "selection.json").resolve()),
        "selection_contact_sheet": str(
            (selected_root / "selected_8_frames_contact_sheet.png").resolve()
        ),
        "raw_cache_report": str(raw_report.resolve()),
        "runtime_pose_mask_manifest": str(pose_manifest.resolve()),
        "runtime_true_pose_manifest": str(true_manifest.resolve()),
        "pose_mask_formal_input_passed": bool(pose_report["formal_input_passed"]),
        "gpu_outputs": {
            "model_input_pose_mask": str(
                (strategy_root / "03_model_input_pose_mask").resolve()
            ),
            "model_input_true_pose": str(
                (strategy_root / "04_model_input_true_pose").resolve()
            ),
            "current_pose_mask": str(
                (strategy_root / "05_current_pose_mask").resolve()
            ),
            "current_true_pose": str(
                (strategy_root / "06_current_true_pose").resolve()
            ),
            "reconviagen_once": str(
                (strategy_root / "07_reconviagen_once").resolve()
            ),
            "contours_pose_mask": str(
                (strategy_root / "08_contours_pose_mask").resolve()
            ),
            "contours_true_pose": str(
                (strategy_root / "09_contours_true_pose").resolve()
            ),
        },
    }
    plan = {
        "format": PLAN_FORMAT,
        "plan_kind": GT_CENTER_PLAN_KIND,
        "created_at_utc": utc_now(),
        "passed": True,
        "dataset": str(dataset),
        "clip": str(source["clip"]),
        "clip_sequence": CLIP_SEQUENCE,
        "frame_count": len(source["frame_names"]),
        "object_key": OBJECT_KEY,
        "experiment_label": EXPERIMENT_LABEL,
        "official_object_id": OFFICIAL_OBJECT_ID,
        "official_object_category": source["official_object_category"],
        "official_sequence_object_count": source["official_sequence_object_count"],
        "camera_conversion_audit": source["camera_conversion_audit"],
        "derived_input_manifests": {
            "frames": str(source["frames_manifest_path"].resolve()),
            "frames_sha256": sha256_file(source["frames_manifest_path"]),
            "masks": str(source["masks_manifest_path"].resolve()),
            "masks_sha256": sha256_file(source["masks_manifest_path"]),
            "masks_are_official_gt": False,
        },
        "official_object_pose": {
            "object_id": OFFICIAL_OBJECT_ID,
            "category": source["official_object_category"],
            "translation_W_m": source["object_translation_W"].tolist(),
            "rotation_O2W": source["object_rotation_O2W"].tolist(),
            "scale_m": source["object_scale_m"].tolist(),
            "annotation": str(source["annotation_path"].resolve()),
            "annotation_sha256": sha256_file(source["annotation_path"]),
        },
        "selection_contract": {
            "purpose": "simulate centre-known second-stage guided capture",
            "candidate_frame_count": len(source["frame_names"]),
            "selected_view_count": 8,
            "center_source": "official Objectron object translation in W",
            "selector": "official-training single-seed spherical FPS",
            "foreground_mask_role": "validity only",
            "quality_gate_used_for_selection": False,
            "oracle_rotation_or_scale_used_for_selection": False,
            "same_selected_frames_for_all_endpoints": True,
        },
        "models": {
            "current": (
                "official no-VGGT posed-DINO SS30K step30000 + "
                "SLat30K step30000 (route C)"
            ),
            "native_ss_report": str(SS30K_REPORT),
            "native_ss_report_sha256": sha256_file(SS30K_REPORT),
            "native_ss_checkpoint": str(SS30K_CHECKPOINT),
            "native_ss_checkpoint_sha256": sha256_file(SS30K_CHECKPOINT),
            "native_ss_checkpoint_step": 30000,
            "native_slat_checkpoint": str(SLAT30K_CHECKPOINT),
            "native_slat_checkpoint_sha256": sha256_file(SLAT30K_CHECKPOINT),
            "native_slat_checkpoint_step": 30000,
            "cross_deployment_abc_r_evidence": str(ABC_R_EVIDENCE),
            "cross_deployment_abc_r_evidence_sha256": sha256_file(ABC_R_EVIDENCE),
            "input_context": "DINO-only; VGGT not loaded or executed",
            "stock_slat_freeze": str(STOCK_FREEZE),
            "stock_slat_freeze_sha256": sha256_file(STOCK_FREEZE),
            "reconviagen": "Stable-X/trellis-vggt-v0-2 original endpoint",
            "seed": 42,
        },
        "o_matrix": {
            "current": [
                (
                    "pose+mask runtime-O; Objectron GT translation is consumed "
                    "only by the selector"
                ),
                "official Objectron true object pose O (oracle upper bound)",
            ],
            "reconviagen": "same selected eight RGB views; runtime-O independent",
        },
        "strategies": [row],
        "scope_guard": (
            "Qualitative Objectron oracle-selection diagnostic. GT translation is "
            "used to choose the eight views. The pose-mask O branch does not consume "
            "GT rotation/scale; the true-pose O branch is a separate oracle upper bound."
        ),
    }
    plan["plan_identity"] = _canonical_sha256(
        {key: value for key, value in plan.items() if key != "created_at_utc"}
    )
    write_json(output / "experiment_plan.json", plan)
    print(
        json.dumps(
            {
                "passed": True,
                "plan_kind": GT_CENTER_PLAN_KIND,
                "clip_sequence": CLIP_SEQUENCE,
                "candidate_frames": len(source["frame_names"]),
                "selected_frames": row["selected_frame_names"],
                "minimum_pairwise_angular_separation_degrees": selection[
                    "minimum_pairwise_angular_separation_degrees"
                ],
                "azimuth_coverage_degrees": selection[
                    "azimuth_coverage_degrees"
                ],
                "plan": str(output / "experiment_plan.json"),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


def clone_model_input(
    source_manifest_path: Path,
    target_runtime_manifest_path: Path,
    output: Path,
) -> None:
    """Clone invariant image contexts and replace only runtime-O K/T geometry."""

    import torch

    source_manifest_path = source_manifest_path.resolve(strict=True)
    target_runtime_manifest_path = target_runtime_manifest_path.resolve(strict=True)
    output = output.resolve()
    if output.exists():
        raise FileExistsError(output)
    source_manifest = _load_json(source_manifest_path)
    target_runtime = _load_json(target_runtime_manifest_path)
    if (
        source_manifest.get("format") != MODEL_INPUT_MANIFEST_FORMAT
        or source_manifest.get("passed") is not True
        or len(source_manifest.get("objects", [])) != 1
    ):
        raise RuntimeError("source DINO-only model-input manifest did not pass")
    if (
        target_runtime.get("format") != RUNTIME_MANIFEST_FORMAT
        or target_runtime.get("passed") is not True
        or len(target_runtime.get("objects", [])) != 1
    ):
        raise RuntimeError("target runtime-O manifest did not pass")
    source_row = source_manifest["objects"][0]
    target_row = target_runtime["objects"][0]
    if source_row["object_key"] != target_row["object_key"]:
        raise RuntimeError("source/target model-input object identity differs")
    if source_row["prepared_rgb_paths"] != target_row["prepared_rgb_paths"]:
        raise RuntimeError("source/target model-input prepared RGB differs")
    if source_row["prepared_mask_paths"] != target_row["prepared_mask_paths"]:
        raise RuntimeError("source/target model-input prepared masks differ")
    source_payload_path = Path(source_row["model_input"]).resolve(strict=True)
    if sha256_file(source_payload_path) != source_row["model_input_sha256"]:
        raise RuntimeError("source model-input payload hash differs")
    payload = torch.load(source_payload_path, map_location="cpu", weights_only=False)
    if (
        payload.get("format") != MODEL_INPUT_OBJECT_FORMAT
        or payload.get("feature_contract") != FEATURE_CONTRACT
    ):
        raise RuntimeError("source model-input payload contract differs")
    target_cache = Path(target_row["cache_npz"]).resolve(strict=True)
    target_cache_sha = sha256_file(target_cache)
    intrinsics, extrinsics, points_o = load_runtime_lifting_geometry(target_cache)
    source_intrinsics = payload["intrinsics"]
    target_intrinsics = torch.from_numpy(intrinsics)
    if not torch.equal(source_intrinsics, target_intrinsics):
        raise RuntimeError("paired O runtime caches changed K_feature")
    if torch.equal(payload["extrinsics"], torch.from_numpy(extrinsics)):
        raise RuntimeError("paired O runtime caches unexpectedly have identical T")
    payload.update(
        {
            "condition_sha256": str(target_row["condition_sha256"]),
            "runtime_cache": str(target_cache),
            "runtime_cache_sha256": target_cache_sha,
            "intrinsics": target_intrinsics,
            "extrinsics": torch.from_numpy(extrinsics),
            "points_o": torch.from_numpy(points_o),
        }
    )
    destination = output / "objects" / target_row["category"] / target_row["object_id"]
    destination.mkdir(parents=True, exist_ok=False)
    payload_path = destination / "dino_only_model_input.pt"
    atomic_torch_save(payload_path, payload)
    config = {
        "pretrained": "Stable-X/trellis-vggt-v0-2",
        "runtime_build_config_sha256": str(target_runtime["build_config_sha256"]),
        "feature_contract": FEATURE_CONTRACT,
        "paired_context_clone": {
            "source_model_input_manifest": str(source_manifest_path),
            "source_model_input_manifest_sha256": sha256_file(source_manifest_path),
            "policy": "copy exact DINO/Stock contexts; replace only runtime-O K/T/P_O",
        },
    }
    config_hash = _canonical_sha256(config)
    report = {
        "format": MODEL_INPUT_OBJECT_FORMAT,
        "created_at_utc": utc_now(),
        "category": str(target_row["category"]),
        "object_id": str(target_row["object_id"]),
        "object_key": str(target_row["object_key"]),
        "pretrained": "Stable-X/trellis-vggt-v0-2",
        "runtime_input_report": str(target_cache.parent / "report.json"),
        "runtime_cache": str(target_cache),
        "runtime_cache_sha256": target_cache_sha,
        "condition_sha256": str(target_row["condition_sha256"]),
        "reference_view_index": int(target_row["reference_view_index"]),
        "prepared_rgb_paths": list(target_row["prepared_rgb_paths"]),
        "prepared_mask_paths": list(target_row["prepared_mask_paths"]),
        "model_input": str(payload_path),
        "model_input_sha256": sha256_file(payload_path),
        "feature_contract": FEATURE_CONTRACT,
        "extrinsics_source": "runtime_input_cache.T_O2C_lifting",
        "encoder_stats": dict(source_row["encoder_stats"]),
        "vggt_model_loaded": False,
        "vggt_model_executed": False,
        "paired_context_clone": config["paired_context_clone"],
        "forbidden_gt_fields_absent": False,
        "training_ready": False,
        "scope_guard": (
            "Oracle-O qualitative pair: exact DINO-only image contexts are copied "
            "from the pose-mask branch and only runtime-O geometry is replaced."
        ),
        "passed": True,
    }
    atomic_json(destination / "report.json", report)
    atomic_json(
        destination / "_DINO_ONLY_MODEL_INPUT_COMPLETE.json",
        {
            "format": MODEL_INPUT_MARKER_FORMAT,
            "runtime_cache_sha256": target_cache_sha,
            "config_sha256": config_hash,
            "model_input_sha256": report["model_input_sha256"],
            "condition_sha256": report["condition_sha256"],
            "passed": True,
        },
    )
    manifest = {
        "format": MODEL_INPUT_MANIFEST_FORMAT,
        "created_at_utc": utc_now(),
        "runtime_input_manifest": str(target_runtime_manifest_path),
        "runtime_input_manifest_sha256": sha256_file(target_runtime_manifest_path),
        "config": config,
        "config_sha256": config_hash,
        "selected_object_count": 1,
        "completed_object_count": 1,
        "reused_objects": [],
        "objects": [report],
        "failures": [],
        "training_ready": False,
        "scope_guard": "Paired exact-context model input; no target/label consumed.",
        "passed": True,
    }
    atomic_json(output / "model_input_manifest.json", manifest)
    print(
        json.dumps(
            {
                "passed": True,
                "source": str(source_manifest_path),
                "target_runtime": str(target_runtime_manifest_path),
                "output": str(output / "model_input_manifest.json"),
                "exact_image_context_clone": True,
            },
            indent=2,
        )
    )


def _tensor_tree_equal(left: Any, right: Any) -> bool:
    import torch

    if torch.is_tensor(left) and torch.is_tensor(right):
        return bool(torch.equal(left, right))
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(
            _tensor_tree_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(
            _tensor_tree_equal(a, b) for a, b in zip(left, right)
        )
    return left == right


def _copy_result(source: Path, destination: Path) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    shutil.copy2(source, destination)
    return {
        "source": str(source),
        "source_sha256": sha256_file(source),
        "presentation_copy": str(destination),
        "presentation_copy_sha256": sha256_file(destination),
        "bytes": destination.stat().st_size,
    }


def finalize(output: Path) -> None:
    import torch

    output = output.resolve(strict=True)
    plan_path = output / "experiment_plan.json"
    plan = _load_json(plan_path)
    if plan.get("format") not in {PLAN_FORMAT, LEGACY_PLAN_FORMAT} or plan.get("passed") is not True:
        raise RuntimeError("Objectron experiment plan did not pass")
    gt_center_mode = plan.get("plan_kind") == GT_CENTER_PLAN_KIND
    if (output / "report.json").exists():
        raise FileExistsError(output / "report.json")
    summaries = []
    for row in plan["strategies"]:
        if not row["active"]:
            summaries.append(
                {
                    "slug": row["slug"],
                    "label_zh": row["label_zh"],
                    "status": "SKIPPED_BY_REGISTERED_PHONE_CAPTURE_GATE",
                    "reason": row["skip_reason"],
                    "selection": row["selection"],
                }
            )
            continue
        strategy = output / row["slug"]
        products = row["gpu_outputs"]
        pose_model_manifest = Path(products["model_input_pose_mask"]) / "model_input_manifest.json"
        true_model_manifest = Path(products["model_input_true_pose"]) / "model_input_manifest.json"
        pose_current_manifest = Path(products["current_pose_mask"]) / "inference_manifest.json"
        true_current_manifest = Path(products["current_true_pose"]) / "inference_manifest.json"
        recon_manifest = Path(products["reconviagen_once"]) / "inference_manifest.json"
        pose_contour = Path(products["contours_pose_mask"]) / "report.json"
        true_contour = Path(products["contours_true_pose"]) / "report.json"
        required = [
            pose_model_manifest,
            true_model_manifest,
            pose_current_manifest,
            true_current_manifest,
            recon_manifest,
            pose_contour,
            true_contour,
        ]
        for path in required:
            if not path.is_file():
                raise FileNotFoundError(path)
            payload = _load_json(path)
            if payload.get("passed") is not True:
                raise RuntimeError(f"required GPU artifact did not pass: {path}")
        pose_model = _load_json(pose_model_manifest)
        true_model = _load_json(true_model_manifest)
        if len(pose_model["objects"]) != 1 or len(true_model["objects"]) != 1:
            raise RuntimeError("paired model-input manifests must each contain one object")
        for manifest in (pose_model, true_model):
            if (
                manifest.get("config", {}).get("feature_contract") != FEATURE_CONTRACT
                or manifest["objects"][0].get("feature_contract") != FEATURE_CONTRACT
            ):
                raise RuntimeError("paired branch is not the frozen DINO-only contract")
            if (
                manifest["objects"][0].get("vggt_model_loaded", False) is not False
                or manifest["objects"][0].get("vggt_model_executed", False) is not False
            ):
                raise RuntimeError("paired no-VGGT branch loaded or executed VGGT")
        pose_payload_path = Path(pose_model["objects"][0]["model_input"])
        true_payload_path = Path(true_model["objects"][0]["model_input"])
        pose_payload = torch.load(pose_payload_path, map_location="cpu", weights_only=False)
        true_payload = torch.load(true_payload_path, map_location="cpu", weights_only=False)
        for payload in (pose_payload, true_payload):
            if (
                payload.get("vggt_model_loaded") is not False
                or payload.get("vggt_model_executed") is not False
            ):
                raise RuntimeError("paired DINO-only payload loaded or executed VGGT")
        exact_fields = ["visual_patch_features", "stock_condition", "slat_condition"]
        exact_checks = {
            field: _tensor_tree_equal(pose_payload[field], true_payload[field])
            for field in exact_fields
        }
        if not all(exact_checks.values()):
            raise RuntimeError("paired O branches changed DINO/Stock image contexts")
        if not torch.equal(pose_payload["intrinsics"], true_payload["intrinsics"]):
            raise RuntimeError("paired O branches changed K_feature")
        if torch.equal(pose_payload["extrinsics"], true_payload["extrinsics"]):
            raise RuntimeError("paired O branches unexpectedly have identical runtime-O T")
        pose_current = _load_json(pose_current_manifest)
        true_current = _load_json(true_current_manifest)
        recon = _load_json(recon_manifest)
        if not (
            len(pose_current["objects"])
            == len(true_current["objects"])
            == len(recon["objects"])
            == 1
        ):
            raise RuntimeError("each qualitative branch must contain exactly one result")
        mesh_dir = strategy / "10_meshes"
        mesh_products = {
            "current_pose_mask_o": _copy_result(
                Path(pose_current["objects"][0]["mesh"]),
                mesh_dir / "当前SS30K_SLat30K_pose-mask估计O.obj",
            ),
            "current_true_object_pose_o": _copy_result(
                Path(true_current["objects"][0]["mesh"]),
                mesh_dir / "当前SS30K_SLat30K_Objectron真实物体位姿O.obj",
            ),
            "reconviagen_once": _copy_result(
                Path(recon["objects"][0]["mesh"]),
                mesh_dir / "ReconViaGen原版_每种选帧仅一次.obj",
            ),
        }
        summaries.append(
            {
                "slug": row["slug"],
                "label_zh": row["label_zh"],
                "status": "COMPLETE",
                "selection": row["selection"],
                "runtime_pose_mask_manifest": row["runtime_pose_mask_manifest"],
                "runtime_true_pose_manifest": row["runtime_true_pose_manifest"],
                "model_input_pair_isolation": {
                    "same_selected_frames": True,
                    "same_prepared_rgb_and_masks": True,
                    "same_K_feature": True,
                    "exact_shared_image_contexts": exact_checks,
                    "vggt_model_loaded": False,
                    "vggt_model_executed": False,
                    "runtime_O_extrinsics_differ": True,
                    "passed": True,
                },
                "current_pose_mask_manifest": str(pose_current_manifest),
                "current_true_pose_manifest": str(true_current_manifest),
                "reconviagen_manifest_once": str(recon_manifest),
                "contours_pose_mask": str(pose_contour),
                "contours_true_pose": str(true_contour),
                "meshes": mesh_products,
            }
        )
        del pose_payload, true_payload

    report = {
        "format": REPORT_FORMAT,
        "plan_kind": plan.get("plan_kind", "legacy_three_selector_matrix"),
        "created_at_utc": utc_now(),
        "passed": True,
        "experiment_plan": str(plan_path),
        "experiment_plan_sha256": sha256_file(plan_path),
        "dataset": plan["dataset"],
        "camera_conversion_audit": plan["camera_conversion_audit"],
        "models": plan["models"],
        "selection_contract": plan.get("selection_contract"),
        "strategy_count": len(summaries),
        "complete_strategy_count": sum(row["status"] == "COMPLETE" for row in summaries),
        "skipped_strategy_count": sum(row["status"].startswith("SKIPPED") for row in summaries),
        "strategies": summaries,
        "interpretation_guard": (
            (
                "This is a centre-known second-stage qualitative diagnosis on one "
                "Objectron clip. GT translation selects the eight views; the full "
                "true-O branch remains an oracle upper bound. No GT surface metric "
                "or deployable endpoint claim is made."
            )
            if gt_center_mode
            else (
                "This is a multi-policy qualitative diagnosis on one Objectron clip. "
                "Oracle true-O results are not deployable evidence; pose-mask results "
                "use derived SAM2 masks. No GT surface metric is claimed."
            )
        ),
    }
    write_json(output / "report.json", report)
    title_suffix = (
        "GT中心引导球面FPS8 × 两种 O 的 SS30K+SLat30K 重建"
        if gt_center_mode
        else "三种选帧 × 两种 O 的 SS30K+SLat30K 重建"
    )
    selection_note = (
        "选帧只读取 Objectron 官方物体中心 translation；不读取物体旋转或尺度。"
        "它对全部有效候选帧执行训练一致的单 seed 球面最远点采样，选出 8 帧。"
        "pose-mask O 分支用于隔离选帧收益；完整真实物体位姿 O 分支只作 oracle 上界。"
        if gt_center_mode
        else (
            "手机采集策略使用现有 `object_spherical_farthest_valid_mask` final-8 "
            "正式门。若片段的方位覆盖不足，它按预注册规则写为 `SKIPPED`。"
        )
    )
    readme = f"""# {plan.get('experiment_label', 'Objectron')}：{title_suffix}

完成时间：{report['created_at_utc']}

当前模型：official no-VGGT SS30K step30000 + SLat30K step30000（冻结路线 C）。
对照：原版 strict ReconViaGen。
随机策略固定种子：{RANDOM_SEED}。所有当前模型分支固定采样 seed=42。

每个通过门控的选帧策略包含：

1. 当前模型 + pose-mask 估计 runtime-O 的 Mesh；
2. 当前模型 + Objectron 官方真实物体位姿 oracle-O 的 Mesh；
3. 原版 ReconViaGen Mesh（同一组选帧只运行一次）；
4. 两个当前模型 Mesh 分别投影回原始 8 帧的青色轮廓及总览图。

两种 O 严格共享相同 8 帧、prepared RGB/mask、K_feature、DINO-only
图像 context 和 Stock context；只允许 runtime-O 外参改变。当前端点不加载、
不执行 VGGT。`report.json` 已对这些字段做 exact audit。

SS30K/SLat30K 的跨部署组合由冻结 held-out Dev64 A/B/C/R 报告与四个
worker 的 SHA256 严格绑定；该实世界输出仍是 qualitative diagnosis，
不升级为 untouched test claim。

{selection_note}

若计划中的 `diagnostic_override=true`，则该分支虽保留
`formal_selection_passed=false`，但按显式实验要求继续生成定性 Mesh；它不能被解释为
通过手机输入质量门。

注意：Objectron 的物体位姿分支使用官方标注，因此只用于诊断物体坐标系影响；
SAM2 mask 是派生产物而非 Objectron 官方 GT mask。本目录不宣称任何 GT surface 指标。
"""
    (output / "README.md").write_text(readme, encoding="utf-8")
    print(
        json.dumps(
            {
                "passed": True,
                "complete_strategies": report["complete_strategy_count"],
                "skipped_strategies": report["skipped_strategy_count"],
                "report": str(output / "report.json"),
                "readme": str(output / "README.md"),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


def finalize_gt_center_pair(
    *, output_root: Path, shoe_output: Path, camera_output: Path
) -> None:
    """Bind the independently reconstructed shoe and camera diagnostics."""

    output_root = output_root.resolve(strict=True)
    if (output_root / "report.json").exists():
        raise FileExistsError(output_root / "report.json")
    clips = []
    for label, path in (("shoe", shoe_output), ("camera", camera_output)):
        clip_root = path.resolve(strict=True)
        report_path = clip_root / "report.json"
        plan_path = clip_root / "experiment_plan.json"
        report = _load_json(report_path)
        plan = _load_json(plan_path)
        if report.get("passed") is not True or plan.get("passed") is not True:
            raise RuntimeError(f"{label} Objectron GT-centre result did not pass")
        if plan.get("plan_kind") != GT_CENTER_PLAN_KIND:
            raise RuntimeError(f"{label} plan is not a GT-centre second-stage plan")
        if len(plan.get("strategies", [])) != 1:
            raise RuntimeError(f"{label} plan must contain exactly one selector")
        selection = plan["strategies"][0]["selection"]
        if int(selection.get("selected_view_count", -1)) != 8:
            raise RuntimeError(f"{label} selector did not bind eight views")
        clips.append(
            {
                "label": label,
                "clip_sequence": plan["clip_sequence"],
                "object_key": plan["object_key"],
                "candidate_frame_count": plan["frame_count"],
                "selected_frame_names": selection["selected_frame_names"],
                "azimuth_coverage_degrees": selection[
                    "azimuth_coverage_degrees"
                ],
                "minimum_pairwise_angular_separation_degrees": selection[
                    "minimum_pairwise_angular_separation_degrees"
                ],
                "report": str(report_path),
                "report_sha256": sha256_file(report_path),
                "plan": str(plan_path),
                "plan_sha256": sha256_file(plan_path),
                "strategy": report["strategies"][0],
            }
        )
    payload = {
        "format": GT_CENTER_PAIR_REPORT_FORMAT,
        "created_at_utc": utc_now(),
        "passed": True,
        "plan_kind": GT_CENTER_PLAN_KIND,
        "clip_count": 2,
        "clips": clips,
        "scientific_scope": {
            "gt_translation_used_for_selection": True,
            "gt_rotation_or_scale_used_for_pose_mask_o": False,
            "true_pose_o_is_oracle_upper_bound": True,
            "same_eight_views_used_by_current_and_reconviagen": True,
            "formal_claim": False,
        },
    }
    write_json(output_root / "report.json", payload)
    lines = [
        "# Objectron 第二阶段：GT 中心引导球面 FPS8",
        "",
        "本目录包含 shoe 与 camera 两个片段。每个片段从全部有效候选帧中，",
        "只使用 Objectron 真值物体中心执行训练一致的单 seed 球面 FPS，选 8 帧。",
        "",
        "每组输出：",
        "",
        "1. SS30K+SLat30K + pose-mask O（GT 只用于选帧）；",
        "2. SS30K+SLat30K + Objectron 完整真值 pose O（oracle 上界）；",
        "3. 同 8 帧 strict ReconViaGen；",
        "4. 两个当前模型分支的原图轮廓回投和 OBJ。",
        "",
        "这是 oracle 选帧定性诊断，不是可部署端点或正式测试结论。",
        "",
    ]
    for clip in clips:
        lines.extend(
            [
                f"- {clip['label']}: `{clip['report']}`",
                (
                    f"  - 候选 {clip['candidate_frame_count']} 帧，方位覆盖 "
                    f"{clip['azimuth_coverage_degrees']:.2f}°，最小视角间隔 "
                    f"{clip['minimum_pairwise_angular_separation_degrees']:.2f}°"
                ),
            ]
        )
    (output_root / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "passed": True,
                "report": str(output_root / "report.json"),
                "readme": str(output_root / "README.md"),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    prepare_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    prepare_parser.add_argument("--clip-sequence", default=CLIP_SEQUENCE)
    prepare_parser.add_argument("--official-object-id", type=int, default=OFFICIAL_OBJECT_ID)
    prepare_parser.add_argument("--object-category", default=OBJECT_CATEGORY)
    prepare_parser.add_argument("--object-id", default=OBJECT_ID)
    prepare_parser.add_argument("--experiment-label", default=EXPERIMENT_LABEL)
    prepare_parser.add_argument(
        "--allow-phone-diagnostic",
        action="store_true",
        help="run the legacy phone-selected branch even when formal final-8 QC fails",
    )
    gt_center_parser = subparsers.add_parser(
        "prepare-gt-center",
        help=(
            "prepare one Objectron-GT-centre, official-training spherical-FPS8 "
            "second-stage diagnostic"
        ),
    )
    gt_center_parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    gt_center_parser.add_argument("--output", type=Path, required=True)
    gt_center_parser.add_argument("--clip-sequence", required=True)
    gt_center_parser.add_argument("--official-object-id", type=int, default=0)
    gt_center_parser.add_argument("--object-category", required=True)
    gt_center_parser.add_argument("--object-id", required=True)
    gt_center_parser.add_argument("--experiment-label", required=True)
    finalize_parser = subparsers.add_parser("finalize")
    finalize_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    clone_parser = subparsers.add_parser("clone-model-input")
    clone_parser.add_argument("--source-model-input-manifest", type=Path, required=True)
    clone_parser.add_argument("--target-runtime-manifest", type=Path, required=True)
    clone_parser.add_argument("--output", type=Path, required=True)
    pair_parser = subparsers.add_parser("finalize-gt-center-pair")
    pair_parser.add_argument("--output-root", type=Path, required=True)
    pair_parser.add_argument("--shoe-output", type=Path, required=True)
    pair_parser.add_argument("--camera-output", type=Path, required=True)
    return parser


def main() -> None:
    args = make_parser().parse_args()
    if args.command in {"prepare", "prepare-gt-center"}:
        configure_experiment(
            clip_sequence=args.clip_sequence,
            official_object_id=args.official_object_id,
            object_category=args.object_category,
            object_id=args.object_id,
            experiment_label=args.experiment_label,
            allow_phone_diagnostic=bool(
                getattr(args, "allow_phone_diagnostic", False)
            ),
        )
        if args.command == "prepare":
            prepare(args.dataset, args.output)
        else:
            prepare_gt_center(args.dataset, args.output)
    elif args.command == "finalize":
        finalize(args.output)
    elif args.command == "clone-model-input":
        clone_model_input(
            args.source_model_input_manifest,
            args.target_runtime_manifest,
            args.output,
        )
    elif args.command == "finalize-gt-center-pair":
        finalize_gt_center_pair(
            output_root=args.output_root,
            shoe_output=args.shoe_output,
            camera_output=args.camera_output,
        )
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
