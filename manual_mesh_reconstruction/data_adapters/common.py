#!/usr/bin/env python3
"""Shared, fail-closed contracts for dataset adapters.

The adapters normalize heterogeneous real-data layouts into the one raw-cache
schema consumed by :mod:`manual_mesh_reconstruction.runtime_o`.  They never
run DINO, SS, SLat, a Mesh decoder, or consume a generated Mesh as input.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import re
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image

from manual_mesh_reconstruction.common import (
    atomic_json,
    atomic_npz,
    canonical_sha256,
    load_json,
    sha256_file,
)
from manual_mesh_reconstruction.raw_cache import (
    OBJECT_CACHE_FORMAT,
    RAW_CACHE_FORMAT,
)


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
NORMALIZED_SOURCE_FORMAT = (
    "manual_mesh_reconstruction.normalized_camera_source.v2_all_views_before_selection"
)
SELECTION_FORMAT = (
    "manual_mesh_reconstruction.temporal_frame_selection.v2_deferred_until_after_o"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_id(value: str, *, label: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    if not normalized:
        raise ValueError(f"invalid {label}: {value!r}")
    return normalized


def natural_key(value: str | Path) -> tuple[Any, ...]:
    text = Path(value).name if isinstance(value, Path) else str(value)
    return tuple(
        int(piece) if piece.isdigit() else piece.lower()
        for piece in re.split(r"([0-9]+)", text)
    )


def time_uniform_indices(count: int, requested: int) -> np.ndarray:
    count = int(count)
    requested = int(requested)
    if requested < 1:
        raise ValueError("selected view count must be positive")
    if count < requested:
        raise RuntimeError(
            f"time-uniform selection requires >= {requested} eligible frames; got {count}"
        )
    if requested == 1:
        return np.asarray([0], dtype=np.int64)
    chosen = np.rint(np.linspace(0, count - 1, requested)).astype(np.int64)
    if len(np.unique(chosen)) != requested:
        raise RuntimeError("time-uniform selection unexpectedly produced duplicates")
    return chosen


def select_indices(
    count: int,
    requested: int,
    *,
    policy: str,
    random_seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    if policy == "time_uniform":
        selected = time_uniform_indices(count, requested)
        record = {
            "format": SELECTION_FORMAT,
            "policy": "time_axis_uniform_including_endpoints_v1",
            "deterministic": True,
            "candidate_frame_count": int(count),
            "requested_view_count": int(requested),
            "selected_source_indices": selected.astype(int).tolist(),
        }
    elif policy == "random":
        if count < requested:
            raise RuntimeError(
                f"random selection requires >= {requested} eligible frames; got {count}"
            )
        draw = np.random.default_rng(int(random_seed)).choice(
            int(count), size=int(requested), replace=False
        )
        selected = np.sort(draw.astype(np.int64))
        record = {
            "format": SELECTION_FORMAT,
            "policy": "uniform_random_without_replacement_v1",
            "deterministic": True,
            "random_seed": int(random_seed),
            "raw_draw_order": draw.astype(int).tolist(),
            "execution_order": "selected source indices sorted chronologically",
            "candidate_frame_count": int(count),
            "requested_view_count": int(requested),
            "selected_source_indices": selected.astype(int).tolist(),
        }
    else:
        raise ValueError(f"unsupported adapter frame selection policy={policy!r}")
    return selected, record


def deferred_selection_request(
    count: int,
    requested: int,
    *,
    policy: str,
    random_seed: int,
) -> dict[str, Any]:
    """Freeze selection intent while retaining every eligible input view."""

    count = int(count)
    requested = int(requested)
    if requested < 1:
        raise ValueError("selected view count must be positive")
    if count < requested:
        raise RuntimeError(
            f"{policy} selection requires >= {requested} eligible frames; got {count}"
        )
    if policy not in {
        "time_uniform",
        "random",
        "training_spherical_farthest",
    }:
        raise ValueError(f"unsupported adapter frame selection policy={policy!r}")
    return {
        "format": SELECTION_FORMAT,
        "requested_policy": str(policy),
        "requested_view_count": requested,
        "random_seed": int(random_seed) if policy == "random" else None,
        "candidate_frame_count": count,
        "selection_deferred_to_runtime_o": True,
        "ordering_contract": "source frames remain chronological",
        "reason": (
            "official-compatible model-O is estimated from every eligible input "
            "view before the model input subset is selected"
        ),
    }


def find_image_directory(dataset: Path) -> Path:
    for name in ("color", "images", "rgb", "frames"):
        candidate = dataset / name
        if candidate.is_dir() and any(
            path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
            for path in candidate.iterdir()
        ):
            return candidate.resolve()
    raise FileNotFoundError(f"no color/images/rgb/frames directory: {dataset}")


def find_mask_directory(dataset: Path) -> Path:
    for name in ("masks", "mask", "matting"):
        candidate = dataset / name
        if candidate.is_dir() and any(
            path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
            for path in candidate.iterdir()
        ):
            return candidate.resolve()
    raise FileNotFoundError(f"no masks/mask/matting directory: {dataset}")


def indexed_media(directory: Path) -> dict[str, Path]:
    output: dict[str, Path] = {}
    for path in sorted(directory.iterdir(), key=natural_key):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        key = path.stem
        if key in output:
            raise RuntimeError(f"duplicate media stem={key!r} in {directory}")
        output[key] = path.resolve()
    if not output:
        raise RuntimeError(f"no supported media files: {directory}")
    return output


@dataclass(frozen=True)
class CameraFrame:
    source_index: int
    source_name: str
    image_path: Path
    mask_path: Path
    K: np.ndarray
    T_W2C: np.ndarray
    camera_model: str = "PINHOLE"
    distortion: tuple[float, ...] = ()
    pose_source: str = "unspecified"

    def validate(self) -> None:
        if not self.image_path.is_file() or not self.mask_path.is_file():
            raise FileNotFoundError(
                f"missing RGB/mask for frame={self.source_name}: "
                f"{self.image_path} {self.mask_path}"
            )
        K = np.asarray(self.K, dtype=np.float64)
        T = np.asarray(self.T_W2C, dtype=np.float64)
        if K.shape != (3, 3) or T.shape != (4, 4):
            raise ValueError(f"invalid K/T shape for frame={self.source_name}")
        if not np.isfinite(K).all() or not np.isfinite(T).all():
            raise ValueError(f"non-finite K/T for frame={self.source_name}")
        rotation = T[:3, :3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=2.0e-4):
            raise ValueError(f"T_W2C rotation is not orthonormal: {self.source_name}")
        if not np.isclose(np.linalg.det(rotation), 1.0, atol=2.0e-4):
            raise ValueError(f"T_W2C rotation is not proper: {self.source_name}")
        if K[0, 0] <= 0 or K[1, 1] <= 0:
            raise ValueError(f"non-positive focal length: {self.source_name}")


def _normalized_image_and_mask(
    frame: CameraFrame, image_destination: Path, mask_destination: Path
) -> tuple[int, int]:
    with Image.open(frame.image_path) as handle:
        image = handle.convert("RGB")
        width, height = image.size
        image.save(image_destination, format="PNG")
    with Image.open(frame.mask_path) as handle:
        mask = handle.convert("L")
        if mask.size != (width, height):
            mask = mask.resize((width, height), Image.Resampling.NEAREST)
        mask.save(mask_destination, format="PNG")
    return width, height


def materialize_raw_cache(
    *,
    output_dir: Path,
    dataset_type: str,
    source_path: Path,
    category: str,
    object_id: str,
    input_frames: Sequence[CameraFrame],
    selection_request: dict[str, Any],
    source_binding: dict[str, Any],
    extra_report: dict[str, Any] | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Materialize every eligible view before runtime-O and model selection.

    RGB and masks are normalized to lossless, same-name PNGs.  ``K`` therefore
    remains numerically unchanged because no resize or crop is performed.  No
    model-view selection is allowed here: runtime-O must first be estimated
    from this complete observable input domain.
    """

    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"raw-cache output already exists: {output_dir}")
    category = safe_id(category, label="category")
    object_id = safe_id(object_id, label="object id")
    object_dir = output_dir / "objects" / category / object_id
    images_dir = object_dir / "images"
    masks_dir = object_dir / "masks"
    images_dir.mkdir(parents=True, exist_ok=False)
    masks_dir.mkdir(parents=True, exist_ok=False)

    names: list[str] = []
    source_names: list[str] = []
    source_indices: list[int] = []
    intrinsics: list[np.ndarray] = []
    poses: list[np.ndarray] = []
    camera_rows: list[dict[str, Any]] = []
    frame_bindings: list[dict[str, Any]] = []
    if not input_frames:
        raise ValueError("raw-cache input frame domain is empty")
    for slot, frame in enumerate(input_frames):
        frame.validate()
        name = f"view_{slot:04d}.png"
        width, height = _normalized_image_and_mask(
            frame, images_dir / name, masks_dir / name
        )
        K = np.asarray(frame.K, dtype=np.float64)
        T = np.asarray(frame.T_W2C, dtype=np.float64)
        names.append(name)
        source_names.append(str(frame.source_name))
        source_indices.append(int(frame.source_index))
        intrinsics.append(K)
        poses.append(T)
        camera_rows.append(
            {
                "frame_name": name,
                "source_frame_name": str(frame.source_name),
                "model": str(frame.camera_model),
                "width": int(width),
                "height": int(height),
                "distortion": [float(value) for value in frame.distortion],
                "pose_source": str(frame.pose_source),
            }
        )
        frame_bindings.append(
            {
                "slot": int(slot),
                "source_index": int(frame.source_index),
                "source_name": str(frame.source_name),
                "source_image": str(frame.image_path.resolve()),
                "source_image_sha256": sha256_file(frame.image_path),
                "source_mask": str(frame.mask_path.resolve()),
                "source_mask_sha256": sha256_file(frame.mask_path),
                "normalized_image_sha256": sha256_file(images_dir / name),
                "normalized_mask_sha256": sha256_file(masks_dir / name),
            }
        )

    cache = object_dir / "raw_camera_point_cache.npz"
    atomic_npz(
        cache,
        frame_name=np.asarray(names),
        source_frame_name=np.asarray(source_names),
        source_frame_index=np.asarray(source_indices, dtype=np.int64),
        K=np.asarray(intrinsics, dtype=np.float64),
        T_W2C=np.asarray(poses, dtype=np.float64),
        P_W=np.empty((0, 3), dtype=np.float64),
    )
    selection = dict(selection_request)
    selection.update(
        {
            "selection_deferred_to_runtime_o": True,
            "eligible_source_indices": source_indices,
            "eligible_source_frame_names": source_names,
            "normalized_frame_names": names,
        }
    )
    binding = {
        "format": NORMALIZED_SOURCE_FORMAT,
        "dataset_type": str(dataset_type),
        "source_path": str(Path(source_path).resolve()),
        "source": source_binding,
        "selection": selection,
        "frames": frame_bindings,
    }
    binding["sha256"] = canonical_sha256(binding)
    row = {
        "format": OBJECT_CACHE_FORMAT,
        "adapter_format": NORMALIZED_SOURCE_FORMAT,
        "created_at_utc": utc_now(),
        "category": category,
        "object_id": object_id,
        "object_key": f"{category}:{object_id}",
        "object_root": str(object_dir.resolve()),
        "source_dataset": str(Path(source_path).resolve()),
        "images_dir": str(images_dir.resolve()),
        "masks_dir": str(masks_dir.resolve()),
        "cache_npz": str(cache.resolve()),
        "cache_npz_sha256": sha256_file(cache),
        "frame_count": len(names),
        "registered_frame_count": len(names),
        "registered_pair_count": len(names),
        "input_view_count": len(names),
        "eligible_source_frame_names": source_names,
        "eligible_source_indices": source_indices,
        "selected_source_frame_names": [],
        "selected_source_indices": [],
        "view_selection": selection,
        "sparse_point_count": 0,
        "point_cloud_consumed": False,
        "geometry_availability": "camera_pose_intrinsics_mask_only",
        "camera_models": sorted({str(frame.camera_model) for frame in input_frames}),
        "cameras": camera_rows,
        "source_binding": binding,
        "coordinate_policy": (
            "T_W2C is retained in the source camera world frame; P_W is empty; "
            "runtime-O must be derived from every eligible calibrated camera ray "
            "and mask before any model-view selection."
        ),
        "training_ready": False,
        "passed": True,
    }
    atomic_json(object_dir / "raw_cache.json", row)
    report = {
        "format": RAW_CACHE_FORMAT,
        "adapter_format": NORMALIZED_SOURCE_FORMAT,
        "created_at_utc": utc_now(),
        "dataset_type": str(dataset_type),
        "source_path": str(Path(source_path).resolve()),
        "output_dir": str(output_dir),
        "category_count": 1,
        "object_count": 1,
        "objects": [row],
        "selection": selection,
        "point_cloud_consumed": False,
        "alignment_passed": False,
        "training_ready": False,
        "scope_guard": (
            "Input-only real reconstruction adapter. It consumes RGB, masks, K and "
            "T_W2C; no generated Mesh, model output, or target geometry is consumed."
        ),
        "passed": True,
        **(extra_report or {}),
    }
    report_path = output_dir / "raw_cache_report.json"
    atomic_json(report_path, report)
    return report_path, row


def validate_reusable_adapter_report(path: Path) -> dict[str, Any]:
    from manual_mesh_reconstruction.data_adapters import ADAPTER_FORMAT

    report = load_json(path)
    if report.get("format") != ADAPTER_FORMAT or report.get("passed") is not True:
        raise RuntimeError(f"adapter report did not pass: {path}")
    selection = report.get("selection")
    if not isinstance(selection, dict) or selection.get(
        "selection_deferred_to_runtime_o"
    ) is not True:
        raise RuntimeError(
            "adapter report predates the all-view-O-before-selection contract; "
            f"use a fresh output directory: {path}"
        )
    for key, hash_key in (
        ("raw_cache_report", "raw_cache_report_sha256"),
        ("runtime_input_manifest", "runtime_input_manifest_sha256"),
    ):
        value = report.get(key)
        if value is None:
            continue
        artifact = Path(str(value)).resolve()
        if not artifact.is_file() or sha256_file(artifact) != str(report.get(hash_key)):
            raise RuntimeError(f"reusable adapter artifact changed: {artifact}")
        if key == "raw_cache_report":
            raw = load_json(artifact)
            if (
                raw.get("format") != RAW_CACHE_FORMAT
                or raw.get("adapter_format") != NORMALIZED_SOURCE_FORMAT
                or raw.get("passed") is not True
            ):
                raise RuntimeError(
                    "reusable raw cache predates the all-view adapter contract: "
                    f"{artifact}"
                )
            for row in raw.get("objects", []):
                eligible = list(row.get("eligible_source_indices", []))
                if (
                    row.get("selected_source_indices") not in ([], None)
                    or row.get("selected_source_frame_names") not in ([], None)
                    or row.get("view_selection", {}).get(
                        "selection_deferred_to_runtime_o"
                    )
                    is not True
                    or int(row.get("input_view_count", -1)) != len(eligible)
                ):
                    raise RuntimeError(
                        "reusable raw cache selected views before runtime-O: "
                        f"{artifact}"
                    )
    return report
