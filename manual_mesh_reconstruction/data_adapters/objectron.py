#!/usr/bin/env python3
"""Objectron adapter with explicit pose-mask or official true-object O."""

from __future__ import annotations

from dataclasses import asdict
import os
from pathlib import Path
import sys
from typing import Any

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import numpy as np
from PIL import Image

from manual_mesh_reconstruction.canonicalization import (
    RuntimeObjectFrame,
    array_sha256,
    normalize_similarity_extrinsics,
    validate_proper_similarity,
)
from manual_mesh_reconstruction.common import (
    atomic_json,
    atomic_npz,
    canonical_sha256,
    load_json,
    sha256_file,
)
from manual_mesh_reconstruction.data_adapters.common import (
    CameraFrame,
    deferred_selection_request,
    materialize_raw_cache,
    safe_id,
    utc_now,
)
from manual_mesh_reconstruction.pose_mask import (
    OFFICIAL_COMPATIBLE_Z_UP_OBJECT_FRAME_POLICY,
    PoseMaskObjectFrameConfig,
)
from manual_mesh_reconstruction.runtime_o import (
    MANIFEST_FORMAT as RUNTIME_MANIFEST_FORMAT,
    MARKER_FORMAT as RUNTIME_MARKER_FORMAT,
    OBJECT_FORMAT as RUNTIME_OBJECT_FORMAT,
    build_object_runtime_input,
)


TRUE_POSE_FRONTEND_FORMAT = "manual_mesh_reconstruction.objectron_true_object_pose.v1"
EXPECTED_OBJECT_EXTENT = 0.90
SCALE_PADDING = 1.05
FEATURE_RESOLUTION = 518
FOREGROUND_MARGIN = 1.10
ALPHA_THRESHOLD = 0.80


def _resolve_clip(dataset: Path, clip_sequence: str | None) -> tuple[Path, Path, str]:
    dataset = dataset.resolve(strict=True)
    if (dataset / "annotation.pbdata").is_file():
        clip = dataset
        root = dataset
        while root.parent != root and not (root / "official_schema").is_dir():
            root = root.parent
        if not (root / "official_schema").is_dir():
            raise FileNotFoundError("Objectron official_schema was not found above clip")
        relative = str(clip.relative_to(root / "clips")) if (root / "clips") in clip.parents else clip.name
        return root, clip, relative
    if not clip_sequence:
        raise ValueError("Objectron dataset root requires --objectron-clip")
    clip = (dataset / "clips" / clip_sequence).resolve(strict=True)
    return dataset, clip, str(clip_sequence).strip("/")


def parse_objectron(
    dataset: Path, *, clip_sequence: str | None, official_object_id: int
) -> dict[str, Any]:
    root, clip, resolved_sequence = _resolve_clip(dataset, clip_sequence)
    schema = root / "official_schema"
    if str(schema) not in sys.path:
        sys.path.insert(0, str(schema))
    from objectron.schema import annotation_data_pb2  # pylint: disable=import-outside-toplevel

    annotation = clip / "annotation.pbdata"
    sequence = annotation_data_pb2.Sequence()
    sequence.ParseFromString(annotation.read_bytes())
    frame_paths = sorted((clip / "frames").glob("frame_*.png"))
    mask_paths = sorted((clip / "masks").glob("frame_*.png"))
    if not frame_paths or len(frame_paths) != len(mask_paths):
        raise RuntimeError("Objectron RGB/mask frame counts differ")
    if [path.name for path in frame_paths] != [path.name for path in mask_paths]:
        raise RuntimeError("Objectron RGB/mask frame names differ")
    if len(sequence.frame_annotations) != len(frame_paths):
        raise RuntimeError("Objectron camera annotation/frame counts differ")
    matches = [value for value in sequence.objects if int(value.id) == int(official_object_id)]
    if len(matches) != 1:
        raise RuntimeError(
            f"Objectron object id={official_object_id} is not unique; "
            f"available={[int(value.id) for value in sequence.objects]}"
        )
    official = matches[0]
    orientation = np.asarray(
        [
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    with Image.open(frame_paths[0]) as image:
        width, height = image.size
    center_w = np.asarray(official.translation, dtype=np.float64)
    intrinsics = []
    poses = []
    projection_errors = []
    depths = []
    for expected_index, frame in enumerate(sequence.frame_annotations):
        if int(frame.frame_id) != expected_index:
            raise RuntimeError("Objectron frame annotations are not contiguous")
        view = np.asarray(frame.camera.view_matrix, dtype=np.float64).reshape(4, 4)
        landscape_k = np.asarray(frame.camera.intrinsics, dtype=np.float64).reshape(3, 3)
        T = orientation @ view
        K = np.asarray(
            [
                [landscape_k[1, 1], 0.0, width - landscape_k[1, 2]],
                [0.0, landscape_k[0, 0], landscape_k[0, 2]],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        rotation = T[:3, :3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=2.0e-5) or not np.isclose(
            np.linalg.det(rotation), 1.0, atol=2.0e-5
        ):
            raise RuntimeError(f"Objectron camera conversion is improper: frame={expected_index}")
        point_c = (T @ np.r_[center_w, 1.0])[:3]
        projected = K @ point_c
        projected = projected[:2] / projected[2]
        annotation_matches = [
            value for value in frame.annotations if int(value.object_id) == int(official_object_id)
        ]
        if len(annotation_matches) != 1:
            raise RuntimeError(f"Objectron frame={expected_index} object annotation is not unique")
        keypoint = annotation_matches[0].keypoints[0].point_2d
        gt = np.asarray([float(keypoint.x) * width, float(keypoint.y) * height])
        projection_errors.append(float(np.linalg.norm(projected - gt)))
        depths.append(float(point_c[2]))
        intrinsics.append(K)
        poses.append(T)
    errors = np.asarray(projection_errors)
    depths_array = np.asarray(depths)
    if float(errors.max()) > 2.0 or float(depths_array.min()) <= 0.0:
        raise RuntimeError("Objectron camera conversion failed object-center projection audit")
    rotation_o2w = np.asarray(official.rotation, dtype=np.float64).reshape(3, 3)
    validate_proper_similarity(
        np.block(
            [
                [rotation_o2w, center_w[:, None]],
                [np.asarray([[0.0, 0.0, 0.0, 1.0]])],
            ]
        ),
        name="Objectron official rigid object pose",
    )
    scale_m = np.asarray(official.scale, dtype=np.float64)
    if scale_m.shape != (3,) or np.any(scale_m <= 0.0):
        raise RuntimeError("Objectron official object scale is invalid")
    optional_manifests = {}
    for name in ("frames_manifest.json", "masks_manifest.json"):
        path = clip / name
        if path.is_file():
            payload = load_json(path)
            if payload.get("passed") is not True:
                raise RuntimeError(f"Objectron input manifest did not pass: {path}")
            if name.startswith("masks") and "selected_object_id" in payload:
                if int(payload["selected_object_id"]) != int(official_object_id):
                    raise RuntimeError("Objectron mask manifest is for a different object id")
            optional_manifests[name] = {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
            }
    return {
        "root": root,
        "clip": clip,
        "clip_sequence": resolved_sequence,
        "frames": frame_paths,
        "masks": mask_paths,
        "K": np.asarray(intrinsics),
        "T_W2C": np.asarray(poses),
        "object_translation_W": center_w,
        "object_rotation_O2W": rotation_o2w,
        "object_scale_m": scale_m,
        "object_category": str(official.category),
        "official_object_id": int(official_object_id),
        "annotation": annotation,
        "optional_manifests": optional_manifests,
        "camera_conversion_audit": {
            "passed": True,
            "source": "official Objectron camera matrices",
            "source_convention": "landscape-right ARKit/OpenGL -z forward",
            "runtime_convention": "portrait OpenCV/COLMAP +z forward",
            "formula": "T_W2C_portrait_cv=A@view_matrix; K axes/principal point remapped",
            "A": orientation.tolist(),
            "object_center_projection_error_pixels": {
                "median": float(np.median(errors)),
                "maximum": float(errors.max()),
            },
            "object_center_depth_m": {
                "minimum": float(depths_array.min()),
                "maximum": float(depths_array.max()),
            },
        },
    }


def _runtime_manifest(
    output: Path, report: dict[str, Any], raw_report: Path, build_config: dict[str, Any]
) -> Path:
    payload = {
        "format": RUNTIME_MANIFEST_FORMAT,
        "created_at_utc": utc_now(),
        "raw_cache_report": str(raw_report.resolve()),
        "raw_cache_report_sha256": sha256_file(raw_report),
        "source_raw_cache_report": str(raw_report.resolve()),
        "source_raw_cache_report_sha256": sha256_file(raw_report),
        "build_config": build_config,
        "build_config_sha256": canonical_sha256(build_config),
        "source_selected_object_count": 1,
        "selected_object_count": 1,
        "completed_object_count": 1,
        "reused_objects": [],
        "objects": [report],
        "quality_rejections": [],
        "eligible_raw_cache_report": None,
        "failures": [],
        "training_ready": False,
        "scope_guard": "Objectron input-only runtime-O cache.",
        "passed": True,
    }
    path = output / "runtime_input_manifest.json"
    atomic_json(path, payload)
    return path


def _build_pose_mask_base(
    *,
    raw_report: Path,
    raw_row: dict[str, Any],
    output: Path,
    selected_view_count: int,
    selection_policy: str,
    random_seed: int,
) -> tuple[Path, dict[str, Any]]:
    config = PoseMaskObjectFrameConfig(
        mask_threshold=0.5,
        expected_object_extent=EXPECTED_OBJECT_EXTENT,
        scale_padding=SCALE_PADDING,
        object_frame_policy=OFFICIAL_COMPATIBLE_Z_UP_OBJECT_FRAME_POLICY,
    )
    config.validate()
    build = {
        "adapter": TRUE_POSE_FRONTEND_FORMAT,
        "o_definition": "pose+mask base used only to freeze identical image preprocessing",
        "selected_view_count": int(selected_view_count),
        "selection_policy": str(selection_policy),
        "selection_seed": int(random_seed),
        "feature_resolution": FEATURE_RESOLUTION,
        "foreground_margin": FOREGROUND_MARGIN,
        "alpha_threshold": ALPHA_THRESHOLD,
        "frame_config": asdict(config),
        "gravity_up_W": [0.0, 1.0, 0.0],
    }
    output.mkdir(parents=True, exist_ok=False)
    report, reused = build_object_runtime_input(
        raw_row,
        output_dir=output,
        selected_view_count=int(selected_view_count),
        feature_resolution=FEATURE_RESOLUTION,
        foreground_margin=FOREGROUND_MARGIN,
        alpha_threshold=ALPHA_THRESHOLD,
        view_selection_policy={
            "time_uniform": "time_uniform_valid_mask",
            "random": "random_valid_mask",
        }[str(selection_policy)],
        geometry_mode="pose_mask",
        resume_partial=False,
        frame_config=config,
        build_config_sha256=canonical_sha256(build),
        gravity_up_w=np.asarray([0.0, 1.0, 0.0]),
        selection_seed=int(random_seed),
    )
    if reused:
        raise RuntimeError("fresh Objectron pose-mask base unexpectedly reused output")
    return _runtime_manifest(output, report, raw_report, build), report


def _build_true_pose_runtime(
    *,
    source: dict[str, Any],
    raw_report: Path,
    pose_report: dict[str, Any],
    output: Path,
    selection: dict[str, Any],
) -> Path:
    category = str(pose_report["category"])
    object_id = str(pose_report["object_id"])
    object_key = str(pose_report["object_key"])
    destination = output / "objects" / category / object_id
    destination.mkdir(parents=True, exist_ok=False)
    with np.load(Path(pose_report["cache_npz"]), allow_pickle=False) as payload:
        arrays = {name: np.asarray(payload[name]) for name in payload.files}
    raw_cache = Path(pose_report["source_raw_cache"])
    with np.load(raw_cache, allow_pickle=False) as payload:
        selected = np.asarray(arrays["selected_source_view_index"], dtype=np.int64)
        T_W2C = np.asarray(payload["T_W2C"][selected], dtype=np.float64)
    scale_m_per_o = (
        float(np.max(source["object_scale_m"])) * SCALE_PADDING / EXPECTED_OBJECT_EXTENT
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
    frame = RuntimeObjectFrame(
        T_O2W=T_O2W,
        T_W2O=T_W2O,
        T_O2C=T_O2C,
        T_C2O=T_C2O,
        P_O=empty_points,
        point_keep_mask=np.empty((0,), dtype=bool),
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
            "object_pose_source": "official Objectron annotation rotation+translation",
            "object_scale_source": "official Objectron metric cuboid dimensions",
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
    cache = destination / "runtime_input_cache.npz"
    atomic_npz(cache, **arrays)
    base_condition = load_json(Path(pose_report["condition_record"]))
    condition = {
        "format": TRUE_POSE_FRONTEND_FORMAT,
        "runtime_frame": frame.record(),
        "shared_image_geometry": base_condition["shared_image_geometry"],
        "undistortion": base_condition["undistortion"],
        "prepared_rgb_sha256": base_condition["prepared_rgb_sha256"],
        "prepared_mask_sha256": base_condition["prepared_mask_sha256"],
        "K_feature_sha256": base_condition["K_feature_sha256"],
        "T_O2C_sha256": array_sha256(T_O2C),
        "T_O2C_lifting_sha256": array_sha256(T_O2C_lifting),
        "P_O_sha256": array_sha256(empty_points),
        "point_cloud_consumed": False,
        "oracle_object_pose_consumed": True,
        "condition_scope": (
            "same RGB/mask/preprocessing as pose-mask base; only O is replaced by "
            "the frozen official Objectron object pose and metric scale"
        ),
    }
    condition["condition_sha256"] = canonical_sha256(condition)
    condition_path = destination / "condition_record.json"
    atomic_json(condition_path, condition)
    build = {
        "adapter": TRUE_POSE_FRONTEND_FORMAT,
        "o_definition": "official Objectron true object pose (oracle diagnostic)",
        "selection": selection,
        "expected_object_extent": EXPECTED_OBJECT_EXTENT,
        "scale_padding": SCALE_PADDING,
    }
    build_hash = canonical_sha256(build)
    report = {
        "format": RUNTIME_OBJECT_FORMAT,
        "created_at_utc": utc_now(),
        "category": category,
        "object_id": object_id,
        "object_key": object_key,
        "source_raw_cache": str(raw_cache.resolve()),
        "source_raw_cache_sha256": sha256_file(raw_cache),
        "build_config_sha256": build_hash,
        "input_frontend_format": TRUE_POSE_FRONTEND_FORMAT,
        "geometry_mode": "official_true_object_pose_oracle",
        "point_cloud_consumed": False,
        "selected_view_count": len(selected),
        "all_input_view_count": int(pose_report["all_input_view_count"]),
        "o_frozen_before_view_selection": True,
        "pose_mask_object_frame_policy": None,
        "selected_source_view_indices": selected.astype(int).tolist(),
        "selected_frame_names": arrays["frame_name"].astype(str).tolist(),
        "view_selection": selection,
        "reference_view_index": int(pose_report["reference_view_index"]),
        "cache_npz": str(cache.resolve()),
        "condition_record": str(condition_path.resolve()),
        "condition_sha256": condition["condition_sha256"],
        "T_O2C_sha256": array_sha256(T_O2C),
        "T_O2C_lifting_sha256": array_sha256(T_O2C_lifting),
        "lifting_extrinsics_policy": (
            "physical T_O2C retained; projectively normalized T_O2C_lifting is used "
            "for Native lifting"
        ),
        "prepared_rgb_paths": list(pose_report["prepared_rgb_paths"]),
        "prepared_mask_paths": list(pose_report["prepared_mask_paths"]),
        "runtime_frame_stats": frame.stats,
        "input_quality": pose_report["input_quality"],
        "formal_input_passed": pose_report["formal_input_passed"],
        "o_definition": "official Objectron true object pose",
        "o_estimation_order": (
            "official Objectron object O is frozen independently before the "
            "requested model-view subset is selected"
        ),
        "oracle_object_pose_consumed": True,
        "forbidden_gt_fields_absent": False,
        "training_ready": False,
        "scope_guard": "Oracle object-pose diagnostic only; RGB/masks are unchanged.",
        "passed": True,
    }
    atomic_json(destination / "report.json", report)
    atomic_json(
        destination / "_RUNTIME_INPUT_COMPLETE.json",
        {
            "format": RUNTIME_MARKER_FORMAT,
            "completed_at_utc": utc_now(),
            "object_key": object_key,
            "source_cache_sha256": sha256_file(raw_cache),
            "build_config_sha256": build_hash,
            "condition_sha256": condition["condition_sha256"],
            "T_O2C_lifting_sha256": array_sha256(T_O2C_lifting),
        },
    )
    return _runtime_manifest(output, report, raw_report, build)


def adapt(
    *,
    input_path: Path,
    output_dir: Path,
    selected_view_count: int,
    selection_policy: str,
    random_seed: int,
    clip_sequence: str | None,
    official_object_id: int,
    o_mode: str,
) -> dict[str, Any]:
    source = parse_objectron(
        input_path.expanduser(),
        clip_sequence=clip_sequence,
        official_object_id=int(official_object_id),
    )
    selection = deferred_selection_request(
        len(source["frames"]),
        int(selected_view_count),
        policy=selection_policy,
        random_seed=int(random_seed),
    )
    selection.update(
        {
            "selection_domain": "official Objectron chronological frame sequence",
            "eligible_source_frame_names": [
                frame.name for frame in source["frames"]
            ],
        }
    )
    frames = [
        CameraFrame(
            source_index=int(index),
            source_name=source["frames"][int(index)].name,
            image_path=source["frames"][int(index)],
            mask_path=source["masks"][int(index)],
            K=source["K"][int(index)],
            T_W2C=source["T_W2C"][int(index)],
            camera_model="PINHOLE",
            distortion=(),
            pose_source="official_objectron_camera_pose",
        )
        for index in range(len(source["frames"]))
    ]
    clip_slug = safe_id(source["clip_sequence"].replace("/", "_"), label="Objectron clip")
    object_id = f"{clip_slug}_object{int(official_object_id)}"
    source_binding = {
        "clip_sequence": source["clip_sequence"],
        "clip": str(source["clip"].resolve()),
        "annotation": str(source["annotation"].resolve()),
        "annotation_sha256": sha256_file(source["annotation"]),
        "official_object_id": int(official_object_id),
        "official_object_category": source["object_category"],
        "official_object_translation_W": source["object_translation_W"].tolist(),
        "official_object_rotation_O2W": source["object_rotation_O2W"].tolist(),
        "official_object_scale_m": source["object_scale_m"].tolist(),
        "camera_conversion_audit": source["camera_conversion_audit"],
        "input_manifests": source["optional_manifests"],
    }
    raw, row = materialize_raw_cache(
        output_dir=output_dir,
        dataset_type="objectron",
        source_path=input_path.expanduser().resolve(strict=True),
        category="objectron",
        object_id=object_id,
        input_frames=frames,
        selection_request=selection,
        source_binding=source_binding,
        extra_report={
            "official_object_pose_available": True,
            "requested_o_mode": o_mode,
            "gravity_up_W": [0.0, 1.0, 0.0],
        },
    )
    runtime = None
    if o_mode == "true_object_pose":
        _base_manifest, pose_report = _build_pose_mask_base(
            raw_report=raw,
            raw_row=row,
            output=output_dir.parent / "runtime_pose_mask_preprocessing_base",
            selected_view_count=int(selected_view_count),
            selection_policy=str(selection_policy),
            random_seed=int(random_seed),
        )
        runtime = _build_true_pose_runtime(
            source=source,
            raw_report=raw,
            pose_report=pose_report,
            output=output_dir.parent / "runtime_true_object_pose",
            selection=dict(pose_report["view_selection"]),
        )
    return {
        "raw_cache_report": str(raw.resolve()),
        "raw_cache_report_sha256": sha256_file(raw),
        "runtime_input_manifest": None if runtime is None else str(runtime.resolve()),
        "runtime_input_manifest_sha256": None if runtime is None else sha256_file(runtime),
        "object_key": row["object_key"],
        "geometry_mode": "pose_mask" if runtime is None else "official_true_object_pose_oracle",
        "gravity_up_W": [0.0, 1.0, 0.0],
        "selection": row["view_selection"],
        "source_binding": source_binding,
        "oracle_object_pose_consumed": runtime is not None,
    }
