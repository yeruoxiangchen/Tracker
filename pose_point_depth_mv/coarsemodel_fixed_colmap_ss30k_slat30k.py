#!/usr/bin/env python3
"""Audit and finalize the fixed-view heimei/snoopy2 qualitative test.

The experiment consumes exactly the RGB frames requested by the user, the
existing all-frame COLMAP model under ``sparse/0``, and foreground masks.  It
does not rerun COLMAP when the registered model passes the preregistered
coverage/reprojection gates.  snoopy2 lacks source masks, so only its selected
eight frames are segmented into the versioned experiment tree with prompted
SAM2; the source dataset remains untouched.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
import sys
from typing import Any, Sequence

import cv2
import numpy as np
from PIL import Image

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
DATASET_ROOT = TRACKER_ROOT / "CoarseModel/datasets"
DEFAULT_OUTPUT = (
    TRACKER_ROOT
    / "pose_point_depth_mv/outputs2/"
    "CoarseModel_heimei_snoopy2_指定帧_COLMAPPose_"
    "ReconViaGen_vs_SS30K_SLat30K_runtimeO轮廓_20260819_v3"
)

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
    "/data/zjr/native_slat_genrecon_v2_mixed1k_20260802_v1/"
    "stock_slat_freeze_v2.json"
)
SAM2_CHECKPOINT = TRACKER_ROOT / "sam2/checkpoints/sam2.1_hiera_tiny.pt"
SAM2_CONFIG = TRACKER_ROOT / "sam2/sam2/configs/sam2.1/sam2.1_hiera_t.yaml"
SNOOPY_REFERENCE = DATASET_ROOT / "snoopy"

PLAN_FORMAT = "pose_point_depth_mv.coarsemodel_fixed_colmap_ss30k_slat30k_plan.v3"
MASK_FORMAT = "pose_point_depth_mv.prompted_sam2_fixed_masks.v3"
INPUT_FORMAT = "pose_point_depth_mv.coarsemodel_fixed_colmap_input.v1"
REPORT_FORMAT = "pose_point_depth_mv.coarsemodel_fixed_colmap_ss30k_slat30k.v3"

CASES: tuple[dict[str, Any], ...] = (
    {
        "name": "heimei",
        "frames": [
            "00000.jpg",
            "00021.jpg",
            "00058.jpg",
            "00070.jpg",
            "00253.jpg",
            "00237.jpg",
            "00267.jpg",
        ],
        "mask_source": "frozen_source_masks",
    },
    {
        "name": "snoopy2",
        "frames": [
            "00001.jpg",
            "00011.jpg",
            "00031.jpg",
            "00050.jpg",
            "00061.jpg",
            "00081.jpg",
            "00131.jpg",
            "00141.jpg",
        ],
        "mask_source": "prompted_sam2_selected_frames_only",
    },
)

# Two foreground prompts intentionally cover the white figure and red box.
# Four distant background prompts prevent the desk/keyboard from joining the
# union.  Coordinates are normalized in the original 1280x720 images.
SNOOPY_PROMPTS: dict[str, list[dict[str, float | int | bool]]] = {
    "00001.jpg": [(0.39, 0.23, 1), (0.39, 0.48, 1)],
    "00011.jpg": [(0.39, 0.24, 1), (0.39, 0.50, 1)],
    "00031.jpg": [(0.43, 0.35, 1), (0.43, 0.54, 1)],
    "00050.jpg": [(0.46, 0.39, 1), (0.48, 0.64, 1)],
    "00061.jpg": [(0.46, 0.22, 1), (0.48, 0.49, 1)],
    "00081.jpg": [(0.43, 0.44, 1), (0.44, 0.70, 1)],
    "00131.jpg": [(0.49, 0.39, 1), (0.49, 0.66, 1)],
    "00141.jpg": [(0.37, 0.21, 1), (0.39, 0.48, 1)],
}
SNOOPY_BOXES: dict[str, tuple[float, float, float, float]] = {
    "00001.jpg": (0.25, 0.13, 0.52, 0.67),
    "00011.jpg": (0.25, 0.14, 0.53, 0.72),
    "00031.jpg": (0.27, 0.21, 0.56, 0.70),
    "00050.jpg": (0.33, 0.17, 0.62, 0.84),
    "00061.jpg": (0.37, 0.00, 0.58, 0.66),
    "00081.jpg": (0.25, 0.27, 0.59, 0.92),
    "00131.jpg": (0.36, 0.27, 0.63, 0.85),
    "00141.jpg": (0.27, 0.06, 0.51, 0.66),
}
BACKGROUND_PROMPTS = (
    (0.08, 0.08, 0),
    (0.92, 0.08, 0),
    (0.08, 0.92, 0),
    (0.92, 0.92, 0),
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


def _copy_verified(source: Path, destination: Path) -> None:
    source = source.expanduser().resolve(strict=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        if sha256_file(source) != sha256_file(destination):
            raise RuntimeError(f"existing copied artifact differs: {destination}")
        return
    shutil.copy2(source, destination)


def _source_mask(dataset: Path, frame_name: str) -> Path:
    stem = Path(frame_name).stem
    for suffix in (".png", ".jpg", ".jpeg"):
        path = dataset / "masks" / f"{stem}{suffix}"
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError(f"missing source mask: {dataset}/masks/{stem}.*")


def _qvec_to_rotation(qvec: Sequence[float]) -> np.ndarray:
    qw, qx, qy, qz = map(float, qvec)
    return np.asarray(
        [
            [1 - 2 * qy * qy - 2 * qz * qz, 2 * qx * qy - 2 * qw * qz, 2 * qx * qz + 2 * qw * qy],
            [2 * qx * qy + 2 * qw * qz, 1 - 2 * qx * qx - 2 * qz * qz, 2 * qy * qz - 2 * qw * qx],
            [2 * qx * qz - 2 * qw * qy, 2 * qy * qz + 2 * qw * qx, 1 - 2 * qx * qx - 2 * qy * qy],
        ],
        dtype=np.float64,
    )


def _summary(values: Sequence[float]) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return {"count": 0, "median": None, "p95": None, "p99": None, "max": None}
    return {
        "count": int(len(array)),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "max": float(np.max(array)),
    }


def _colmap_audit(dataset: Path, selected: Sequence[str]) -> dict[str, Any]:
    model = dataset / "sparse/0"
    camera_lines = [
        line.split()
        for line in (model / "cameras.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    if len(camera_lines) != 1 or camera_lines[0][1] != "SIMPLE_RADIAL":
        raise RuntimeError(f"expected exactly one SIMPLE_RADIAL camera: {model}")
    camera = camera_lines[0]
    camera_id = int(camera[0])
    width, height = map(int, camera[2:4])
    focal, cx, cy, radial = map(float, camera[4:8])

    points: dict[int, np.ndarray] = {}
    for line in (model / "points3D.txt").read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#"):
            fields = line.split()
            points[int(fields[0])] = np.asarray(list(map(float, fields[1:4])), dtype=np.float64)

    lines = [
        line.strip()
        for line in (model / "images.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    if len(lines) % 2:
        raise RuntimeError(f"invalid COLMAP two-line image records: {model}/images.txt")
    registered: dict[str, dict[str, Any]] = {}
    all_errors: list[float] = []
    for offset in range(0, len(lines), 2):
        header = lines[offset].split()
        observations = lines[offset + 1].split()
        if len(observations) % 3:
            raise RuntimeError("invalid POINTS2D triples")
        qvec = list(map(float, header[1:5]))
        tvec = np.asarray(list(map(float, header[5:8])), dtype=np.float64)
        if int(header[8]) != camera_id:
            raise RuntimeError("unexpected second camera in image records")
        name = str(header[9])
        rotation = _qvec_to_rotation(qvec)
        errors: list[float] = []
        for index in range(0, len(observations), 3):
            u, v = map(float, observations[index : index + 2])
            point_id = int(observations[index + 2])
            if point_id < 0 or point_id not in points:
                continue
            point_c = rotation @ points[point_id] + tvec
            if point_c[2] <= 0:
                continue
            x, y = point_c[:2] / point_c[2]
            scale = 1.0 + radial * float(x * x + y * y)
            projected_u = focal * scale * x + cx
            projected_v = focal * scale * y + cy
            errors.append(float(math.hypot(projected_u - u, projected_v - v)))
        registered[name] = {
            "observation_count": len(errors),
            "reprojection_error_px": _summary(errors),
            "camera_center_W": (-rotation.T @ tvec).tolist(),
            "_errors": errors,
        }
        all_errors.extend(errors)

    color_files = sorted(
        path.name for path in (dataset / "color").iterdir() if path.is_file()
    )
    missing = [name for name in selected if name not in registered]
    selected_errors = [
        value
        for name in selected
        if name in registered
        for value in registered[name]["_errors"]
    ]
    selected_rows = {
        name: {key: value for key, value in registered[name].items() if key != "_errors"}
        for name in selected
        if name in registered
    }
    selected_count = sum(int(row["observation_count"]) for row in selected_rows.values())
    # A weighted exact selected summary is unnecessary for the gate because all
    # per-frame medians/p95 are retained; use their observation-weighted values
    # only as a concise diagnostic.
    frame_medians = [
        float(row["reprojection_error_px"]["median"])
        for row in selected_rows.values()
        if row["reprojection_error_px"]["median"] is not None
    ]
    frame_p95 = [
        float(row["reprojection_error_px"]["p95"])
        for row in selected_rows.values()
        if row["reprojection_error_px"]["p95"] is not None
    ]
    checks = {
        "all_color_frames_registered": len(registered) == len(color_files),
        "selected_frames_registered": not missing,
        "selected_each_has_at_least_100_observations": all(
            int(row["observation_count"]) >= 100 for row in selected_rows.values()
        ),
        "selected_each_median_reprojection_le_1p5_px": all(value <= 1.5 for value in frame_medians),
        "selected_each_p95_reprojection_le_4_px": all(value <= 4.0 for value in frame_p95),
    }
    return {
        "passed": all(checks.values()),
        "camera": {
            "camera_id": camera_id,
            "model": "SIMPLE_RADIAL",
            "width": width,
            "height": height,
            "focal": focal,
            "principal_point": [cx, cy],
            "radial": radial,
        },
        "color_frame_count": len(color_files),
        "registered_frame_count": len(registered),
        "registration_rate": len(registered) / max(1, len(color_files)),
        "point3d_count": len(points),
        "all_registered_reprojection_error_px": _summary(all_errors),
        "selected_frame_count": len(selected),
        "selected_observation_count": selected_count,
        "selected_reprojection_error_px": _summary(selected_errors),
        "selected_frame_median_reprojection_px_median": float(np.median(frame_medians)),
        "selected_frame_p95_reprojection_px_median": float(np.median(frame_p95)),
        "missing_selected_frames": missing,
        "selected_frames": selected_rows,
        "checks": checks,
    }


def _case(name: str) -> dict[str, Any]:
    rows = [dict(row) for row in CASES if row["name"] == name]
    if len(rows) != 1:
        raise ValueError(f"unknown case={name!r}")
    return rows[0]


def _input_root(output: Path, name: str) -> Path:
    return output / "objects" / name / "00_fixed_input"


def _derived_dataset(output: Path, name: str) -> Path:
    views = len(_case(name)["frames"])
    return _input_root(output, name) / f"{name}_fixed{views}"


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
    objects = []
    for spec in CASES:
        name = str(spec["name"])
        dataset = (DATASET_ROOT / name).resolve(strict=True)
        frames = list(spec["frames"])
        audit = _colmap_audit(dataset, frames)
        if audit["passed"] is not True:
            raise RuntimeError(f"source COLMAP failed registered gates: {name}: {audit['checks']}")
        bindings = []
        for frame in frames:
            color = (dataset / "color" / frame).resolve(strict=True)
            image_copy = (dataset / "images" / frame).resolve(strict=True)
            rgb_copy = (dataset / "rgb" / frame).resolve(strict=True)
            if not (sha256_file(color) == sha256_file(image_copy) == sha256_file(rgb_copy)):
                raise RuntimeError(f"color/images/rgb source copies differ: {name}/{frame}")
            row: dict[str, Any] = {
                "frame_name": frame,
                "requested_color": _asset(color),
                "images_copy_same_sha256": True,
                "rgb_copy_same_sha256": True,
            }
            if name == "heimei":
                row["source_mask"] = _asset(_source_mask(dataset, frame))
            bindings.append(row)
        objects.append(
            {
                "name": name,
                "source_dataset": str(dataset),
                "requested_frames_in_order": frames,
                "selected_view_count": len(frames),
                "mask_source": spec["mask_source"],
                "source_bindings": bindings,
                "colmap_model": {
                    filename: _asset(dataset / "sparse/0" / filename)
                    for filename in ("cameras.txt", "images.txt", "points3D.txt")
                },
                "colmap_quality_audit": audit,
                "colmap_recomputed": False,
                "derived_dataset": str(_derived_dataset(output, name).resolve()),
            }
        )
    return {
        "format": PLAN_FORMAT,
        "created_at_utc": utc_now(),
        "passed": True,
        "object_count": 2,
        "objects": objects,
        "deployments": deployments,
        "sam2": {
            "scope": "snoopy2 selected eight frames only",
            "checkpoint": _asset(SAM2_CHECKPOINT),
            "config": _asset(SAM2_CONFIG),
            "same_object_reference_dataset": str(SNOOPY_REFERENCE.resolve(strict=True)),
            "reference_role": (
                "same physical object mask/foreground-shape reference only; the two "
                "sequences are not pixel-aligned and masks are never copied as snoopy2 labels"
            ),
            "source_dataset_modified": False,
        },
        "geometry_mode": "pose_mask",
        "view_selection": "exact_user_order_all_views",
        "reconviagen_endpoint": "strict original VGGT -> Stock SS -> Stock SLat -> Stock decoder",
        "current_endpoint": "DINO-only -> official Native-SS30K -> Native-SLat30K -> Stock decoder",
        "mesh_frame_contract": MESH_FRAME_CONTRACT,
        "implementation": _asset(Path(__file__)),
        "scope_guard": (
            "Only the explicitly requested 7/8 RGB frames are exposed to either endpoint. "
            "Camera intrinsics and T_W2C come from each dataset's existing all-frame sparse/0 "
            "COLMAP model, which passed full registration and selected-frame reprojection gates. "
            "No target Mesh, ICP, old reconstruction Mesh, or camera fitting is consumed."
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


def _prompt_records(frame: str) -> list[dict[str, float | int | bool]]:
    prompts = [
        {"x": x, "y": y, "label": label, "normalized": True}
        for x, y, label in SNOOPY_PROMPTS[frame]
    ]
    prompts.extend(
        {"x": x, "y": y, "label": label, "normalized": True}
        for x, y, label in BACKGROUND_PROMPTS
    )
    return prompts


def _prompt_hit(mask: np.ndarray, prompt: dict[str, Any], radius: int = 6) -> bool:
    height, width = mask.shape
    x = int(round(float(prompt["x"]) * width))
    y = int(round(float(prompt["y"]) * height))
    x0, x1 = max(0, x - radius), min(width, x + radius + 1)
    y0, y1 = max(0, y - radius), min(height, y + radius + 1)
    return bool(np.any(mask[y0:y1, x0:x1] > 0))


def _clean_largest_mask(mask: np.ndarray) -> np.ndarray:
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(binary, 8)
    if count <= 1:
        return binary.astype(bool)
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == largest


def _foreground_histogram(image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    pixels = image_bgr[np.asarray(mask, dtype=bool)]
    if len(pixels) < 32:
        raise RuntimeError("foreground histogram mask is empty")
    hsv = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV)
    histogram = cv2.calcHist([hsv], [0, 1], None, [24, 16], [0, 180, 0, 256])
    cv2.normalize(histogram, histogram, alpha=1.0, norm_type=cv2.NORM_L1)
    return histogram


def _load_sam2_image_predictor() -> Any:
    sam2_root = TRACKER_ROOT / "sam2"
    if str(sam2_root) not in sys.path:
        sys.path.insert(0, str(sam2_root))
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    model = build_sam2(
        "configs/sam2.1/sam2.1_hiera_t.yaml",
        str(SAM2_CHECKPOINT),
        device="cuda",
        apply_postprocessing=True,
    )
    return SAM2ImagePredictor(model)


def _predict_box_prompt_mask(
    predictor: Any,
    image_bgr: np.ndarray,
    prompts: Sequence[dict[str, Any]],
    box_normalized: Sequence[float],
    reference_image_bgr: np.ndarray,
    reference_mask: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    import torch

    height, width = image_bgr.shape[:2]
    box = np.asarray(
        [
            float(box_normalized[0]) * width,
            float(box_normalized[1]) * height,
            float(box_normalized[2]) * width,
            float(box_normalized[3]) * height,
        ],
        dtype=np.float32,
    )
    points = np.asarray(
        [[float(row["x"]) * width, float(row["y"]) * height] for row in prompts],
        dtype=np.float32,
    )
    labels = np.asarray([int(row["label"]) for row in prompts], dtype=np.int32)
    predictor.set_image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    candidate_payloads: list[tuple[np.ndarray, float, str]] = []
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for mode, point_coords, point_labels in (
            ("box_and_points", points, labels),
            ("box_only", None, None),
        ):
            masks, scores, _logits = predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=box,
                multimask_output=True,
                normalize_coords=True,
            )
            for mask, score in zip(masks, scores):
                candidate_payloads.append((_clean_largest_mask(mask), float(score), mode))

    reference_histogram = _foreground_histogram(
        reference_image_bgr, reference_mask > 127
    )
    x0, y0, x1, y1 = map(int, np.round(box).tolist())
    box_mask = np.zeros((height, width), dtype=bool)
    box_mask[max(0, y0) : min(height, y1 + 1), max(0, x0) : min(width, x1 + 1)] = True
    records = []
    for index, (mask, sam_score, mode) in enumerate(candidate_payloads):
        area = int(np.count_nonzero(mask))
        area_ratio = area / float(mask.size)
        outside_ratio = float(np.count_nonzero(mask & ~box_mask)) / max(1, area)
        positive_hits = [
            _prompt_hit(mask, prompt)
            for prompt in prompts
            if int(prompt["label"]) == 1
        ]
        negative_hits = [
            _prompt_hit(mask, prompt)
            for prompt in prompts
            if int(prompt["label"]) == 0
        ]
        histogram_similarity = -1.0
        if area >= 32:
            candidate_histogram = _foreground_histogram(image_bgr, mask)
            histogram_similarity = float(
                cv2.compareHist(
                    reference_histogram,
                    candidate_histogram,
                    cv2.HISTCMP_CORREL,
                )
            )
        registered_score = (
            sam_score
            + 2.0 * histogram_similarity
            + 3.0 * sum(positive_hits)
            - 8.0 * sum(negative_hits)
            - 20.0 * outside_ratio
            - (6.0 if area_ratio < 0.005 or area_ratio > 0.40 else 0.0)
        )
        records.append(
            {
                "candidate_index": index,
                "mode": mode,
                "sam_score": sam_score,
                "histogram_similarity_to_snoopy_reference": histogram_similarity,
                "foreground_ratio": area_ratio,
                "outside_prompt_box_ratio": outside_ratio,
                "positive_hits": positive_hits,
                "negative_hits": negative_hits,
                "registered_score": registered_score,
                "mask": mask,
            }
        )
    eligible = [
        row
        for row in records
        if all(row["positive_hits"])
        and not any(row["negative_hits"])
        and row["outside_prompt_box_ratio"] <= 0.03
        and 0.005 <= row["foreground_ratio"] <= 0.40
    ]
    if not eligible:
        diagnostic = [
            {key: value for key, value in row.items() if key != "mask"}
            for row in records
        ]
        raise RuntimeError(f"no eligible box-prompted SAM2 candidate: {diagnostic}")
    selected = max(eligible, key=lambda row: float(row["registered_score"]))
    audit = {
        "prompt_box_normalized_xyxy": list(map(float, box_normalized)),
        "candidate_count": len(records),
        "eligible_candidate_count": len(eligible),
        "selected_candidate_index": int(selected["candidate_index"]),
        "selected_mode": selected["mode"],
        "selected_sam_score": float(selected["sam_score"]),
        "selected_histogram_similarity_to_snoopy_reference": float(
            selected["histogram_similarity_to_snoopy_reference"]
        ),
        "selected_outside_prompt_box_ratio": float(
            selected["outside_prompt_box_ratio"]
        ),
    }
    return np.asarray(selected["mask"], dtype=np.uint8) * 255, audit


def segment_snoopy(output: Path) -> None:
    output = output.expanduser().resolve(strict=True)
    plan = load_json(output / "experiment_plan.json")
    if plan.get("format") != PLAN_FORMAT or plan.get("passed") is not True:
        raise RuntimeError("experiment plan is not eligible")
    spec = _case("snoopy2")
    source = (DATASET_ROOT / "snoopy2/color").resolve(strict=True)
    root = _input_root(output, "snoopy2")
    masks = root / "sam2_masks"
    report_path = root / "sam2_mask_report.json"
    if report_path.is_file():
        report = load_json(report_path)
        if report.get("format") != MASK_FORMAT or report.get("passed") is not True:
            raise RuntimeError(f"existing SAM2 report differs: {report_path}")
        for row in report["masks"]:
            path = Path(row["mask"]["path"])
            if sha256_file(path) != row["mask"]["sha256"]:
                raise RuntimeError(f"existing SAM2 mask hash differs: {path}")
        print(json.dumps({"passed": True, "reused": True, "report": str(report_path)}, indent=2))
        return
    masks.mkdir(parents=True, exist_ok=True)

    predictor = _load_sam2_image_predictor()

    rows = []
    for frame in spec["frames"]:
        image_path = source / frame
        mask_path = masks / f"{Path(frame).stem}.png"
        reference_image = SNOOPY_REFERENCE / "color" / frame
        reference_mask = SNOOPY_REFERENCE / "masks" / f"{Path(frame).stem}.png"
        if not reference_image.is_file() or not reference_mask.is_file():
            raise FileNotFoundError(f"missing snoopy same-object reference: {frame}")
        prompts = _prompt_records(frame)
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        reference_bgr = cv2.imread(str(reference_image), cv2.IMREAD_COLOR)
        reference_mask_array = cv2.imread(str(reference_mask), cv2.IMREAD_GRAYSCALE)
        if image is None or reference_bgr is None or reference_mask_array is None:
            raise RuntimeError(f"failed to read target/reference image or mask: {frame}")
        mask, candidate_audit = _predict_box_prompt_mask(
            predictor,
            image,
            prompts,
            SNOOPY_BOXES[frame],
            reference_bgr,
            reference_mask_array,
        )
        if not cv2.imwrite(str(mask_path), mask):
            raise RuntimeError(f"failed to write SAM2 mask: {mask_path}")
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None or mask.shape != image.shape[:2]:
            raise RuntimeError(f"SAM2 mask/image shape differs: {frame}")
        binary = mask > 127
        component_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
            binary.astype(np.uint8), 8
        )
        component_areas = sorted(
            [int(value) for value in stats[1:, cv2.CC_STAT_AREA]], reverse=True
        )
        area = int(np.count_nonzero(binary))
        ratio = area / float(binary.size)
        positive_hits = [
            _prompt_hit(binary, prompt)
            for prompt in prompts
            if int(prompt["label"]) == 1
        ]
        negative_hits = [
            _prompt_hit(binary, prompt)
            for prompt in prompts
            if int(prompt["label"]) == 0
        ]
        checks = {
            "area_ratio_in_0p005_0p40": 0.005 <= ratio <= 0.40,
            "all_positive_prompts_inside": all(positive_hits),
            "all_negative_prompts_outside": not any(negative_hits),
            "component_count_le_4": component_count - 1 <= 4,
            "largest_component_fraction_ge_0p70": (
                bool(component_areas) and component_areas[0] / max(1, area) >= 0.70
            ),
        }
        if not all(checks.values()):
            raise RuntimeError(f"SAM2 mask QC failed: frame={frame} checks={checks}")
        rows.append(
            {
                "frame_name": frame,
                "source_image": _asset(image_path),
                "mask": _asset(mask_path),
                "same_object_reference_image": _asset(reference_image),
                "same_object_reference_mask": _asset(reference_mask),
                "reference_mask_pixel_aligned": False,
                "candidate_selection": candidate_audit,
                "prompts": prompts,
                "foreground_pixels": area,
                "foreground_ratio": ratio,
                "component_count": component_count - 1,
                "component_areas": component_areas,
                "checks": checks,
            }
        )
    payload = {
        "format": MASK_FORMAT,
        "created_at_utc": utc_now(),
        "passed": True,
        "object": "snoopy2",
        "frame_count": len(rows),
        "sam2_checkpoint": _asset(SAM2_CHECKPOINT),
        "sam2_config": _asset(SAM2_CONFIG),
        "masks": rows,
        "source_dataset_modified": False,
    }
    atomic_json(report_path, payload)
    print(json.dumps({"passed": True, "reused": False, "report": str(report_path)}, indent=2))


def materialize_inputs(output: Path) -> None:
    output = output.expanduser().resolve(strict=True)
    plan = load_json(output / "experiment_plan.json")
    if plan.get("format") != PLAN_FORMAT or plan.get("passed") is not True:
        raise RuntimeError("experiment plan is not eligible")
    planned = {str(row["name"]): dict(row) for row in plan["objects"]}
    for spec in CASES:
        name = str(spec["name"])
        source = (DATASET_ROOT / name).resolve(strict=True)
        root = _input_root(output, name)
        dataset = _derived_dataset(output, name)
        report_path = root / "fixed_input_report.json"
        if report_path.is_file():
            report = load_json(report_path)
            if report.get("format") != INPUT_FORMAT or report.get("passed") is not True:
                raise RuntimeError(f"existing fixed-input report differs: {report_path}")
            continue
        if dataset.exists() and any(dataset.iterdir()):
            raise RuntimeError(f"partial unbound fixed input exists: {dataset}")
        frames = list(spec["frames"])
        bindings = []
        for frame in frames:
            source_rgb = source / "color" / frame
            target_rgb = dataset / "images" / frame
            if name == "heimei":
                source_mask = _source_mask(source, frame)
            else:
                source_mask = root / "sam2_masks" / f"{Path(frame).stem}.png"
                if not source_mask.is_file():
                    raise FileNotFoundError(f"run segment-snoopy first: {source_mask}")
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
            _copy_verified(source / "sparse/0" / filename, dataset / "sparse/0" / filename)
        exposed_images = sorted(path.name for path in (dataset / "images").iterdir() if path.is_file())
        exposed_masks = sorted(path.name for path in (dataset / "masks").iterdir() if path.is_file())
        if exposed_images != sorted(frames):
            raise RuntimeError(f"derived image scope differs: {name}")
        if exposed_masks != sorted(f"{Path(frame).stem}.png" for frame in frames):
            raise RuntimeError(f"derived mask scope differs: {name}")
        payload = {
            "format": INPUT_FORMAT,
            "created_at_utc": utc_now(),
            "passed": True,
            "object": name,
            "source_dataset": str(source),
            "derived_dataset": str(dataset.resolve()),
            "frames_in_user_order": frames,
            "view_count": len(frames),
            "mask_source": spec["mask_source"],
            "bindings": bindings,
            "colmap_model": planned[name]["colmap_model"],
            "colmap_quality_audit": planned[name]["colmap_quality_audit"],
            "only_requested_images_exposed": True,
            "source_dataset_modified": False,
        }
        atomic_json(report_path, payload)
    print(json.dumps({"passed": True, "objects": [row["name"] for row in CASES]}, indent=2))


def _only_object(payload: dict[str, Any], path: Path) -> dict[str, Any]:
    rows = list(payload.get("objects", []))
    if payload.get("passed") is not True or len(rows) != 1:
        raise RuntimeError(f"expected one passed object: {path}")
    return dict(rows[0])


def finalize(output: Path) -> None:
    output = output.expanduser().resolve(strict=True)
    plan_path = output / "experiment_plan.json"
    plan = load_json(plan_path)
    if plan.get("format") != PLAN_FORMAT or plan.get("passed") is not True:
        raise RuntimeError("experiment plan is not eligible")
    results = []
    for spec in CASES:
        name = str(spec["name"])
        object_root = output / "objects" / name
        paths = {
            "input": object_root / "00_fixed_input/fixed_input_report.json",
            "raw": object_root / "01_raw_cache/raw_cache_report.json",
            "runtime": object_root / "02_runtime_o/runtime_input_manifest.json",
            "model": object_root / "03_dino_only_input/model_input_manifest.json",
            "current": object_root / "04_current_ss30k_slat30k/inference_manifest.json",
            "recon": object_root / "05_reconviagen/inference_manifest.json",
            "contour": object_root / "06_current_camera_contours/report.json",
        }
        for path in paths.values():
            if not path.is_file():
                raise FileNotFoundError(f"incomplete experiment artifact: {path}")
        input_report = load_json(paths["input"])
        raw = load_json(paths["raw"])
        runtime = load_json(paths["runtime"])
        model = load_json(paths["model"])
        current = load_json(paths["current"])
        recon = load_json(paths["recon"])
        contour = load_json(paths["contour"])
        if not all(
            row.get("passed") is True
            for row in (input_report, raw, runtime, model, current, recon, contour)
        ):
            raise RuntimeError(f"one or more stages did not pass: {name}")
        raw_row = _only_object(raw, paths["raw"])
        runtime_row = _only_object(runtime, paths["runtime"])
        current_row = _only_object(current, paths["current"])
        recon_row = _only_object(recon, paths["recon"])
        requested = list(spec["frames"])
        if raw_row.get("selected_source_frame_names") != requested:
            raise RuntimeError(f"raw exact-frame order differs: {name}")
        selected_indices = [int(value) for value in runtime_row["selected_source_view_indices"]]
        if selected_indices != list(range(len(requested))):
            raise RuntimeError(f"runtime did not retain every requested view in order: {name}")
        if runtime_row.get("geometry_mode") != "pose_mask":
            raise RuntimeError(f"runtime geometry differs: {name}")
        validate_runtime_o_mesh_frame_contract(current_row)
        current_mesh = Path(current_row["mesh"]).resolve(strict=True)
        recon_mesh = Path(recon_row["mesh"]).resolve(strict=True)
        if current_row.get("native_ss_checkpoint_sha256") != plan["deployments"]["native_ss_checkpoint"]["sha256"]:
            raise RuntimeError(f"current Native-SS identity differs: {name}")
        if current_row.get("native_slat_checkpoint_sha256") != plan["deployments"]["native_slat_checkpoint"]["sha256"]:
            raise RuntimeError(f"current Native-SLat identity differs: {name}")
        if int(current_row.get("native_slat_checkpoint_step", -1)) != 30000:
            raise RuntimeError(f"current Native-SLat step differs: {name}")
        if current_row.get("vggt_model_executed") is not False:
            raise RuntimeError(f"current endpoint unexpectedly executed VGGT: {name}")
        if recon_row.get("method") != "reconviagen_original":
            raise RuntimeError(f"strict ReconViaGen identity differs: {name}")
        if contour.get("mesh_o_sha256") != sha256_file(current_mesh):
            raise RuntimeError(f"contour/current Mesh binding differs: {name}")
        if contour.get("runtime_input_manifest_sha256") != sha256_file(paths["runtime"]):
            raise RuntimeError(f"contour/runtime binding differs: {name}")
        if contour.get("mesh_frame_contract") != MESH_FRAME_CONTRACT:
            raise RuntimeError(f"contour Mesh frame differs: {name}")

        presentation = object_root / "07_结果汇总"
        current_copy = presentation / "当前_SS30K_SLat30K_runtime-O.obj"
        recon_copy = presentation / "ReconViaGen原版.obj"
        _copy_verified(current_mesh, current_copy)
        _copy_verified(recon_mesh, recon_copy)
        results.append(
            {
                "object": name,
                "view_count": len(requested),
                "requested_frames_in_order": requested,
                "colmap_recomputed": False,
                "colmap_quality_audit": input_report["colmap_quality_audit"],
                "fixed_input_report": _asset(paths["input"]),
                "runtime_input_manifest": _asset(paths["runtime"]),
                "current_inference_manifest": _asset(paths["current"]),
                "reconviagen_inference_manifest": _asset(paths["recon"]),
                "current_mesh": _asset(current_copy),
                "reconviagen_mesh": _asset(recon_copy),
                "contour_report": _asset(paths["contour"]),
                "contour_overview": _asset(Path(contour["overview"])),
            }
        )
    report = {
        "format": REPORT_FORMAT,
        "created_at_utc": utc_now(),
        "passed": True,
        "object_count": len(results),
        "results": results,
        "experiment_plan": _asset(plan_path),
        "current_endpoint": plan["current_endpoint"],
        "reconviagen_endpoint": plan["reconviagen_endpoint"],
        "all_current_meshes_native_runtime_o": True,
        "all_contours_bound_to_exact_current_mesh_and_colmap_runtime": True,
        "source_colmap_recomputed": False,
        "interpretation": (
            "This is a two-object qualitative real-image test with fixed user-selected frames. "
            "Cyan contours project the exact native runtime-O current Mesh through T_O2W, "
            "the source all-frame COLMAP T_W2C, calibrated intrinsics and source distortion."
        ),
    }
    atomic_json(output / "report.json", report)
    print(json.dumps({"passed": True, "report": str(output / 'report.json')}, ensure_ascii=False, indent=2))


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "segment-snoopy", "materialize-inputs", "finalize"):
        child = sub.add_parser(name)
        child.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main() -> None:
    args = make_parser().parse_args()
    if args.command == "prepare":
        prepare(args.output)
    elif args.command == "segment-snoopy":
        segment_snoopy(args.output)
    elif args.command == "materialize-inputs":
        materialize_inputs(args.output)
    else:
        finalize(args.output)


if __name__ == "__main__":
    main()
