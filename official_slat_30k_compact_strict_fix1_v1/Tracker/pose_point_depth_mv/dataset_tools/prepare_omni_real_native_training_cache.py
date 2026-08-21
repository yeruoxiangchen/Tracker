#!/usr/bin/env python3
"""Encode aligned Omni real objects as Native v2 SS/SLat supervision.

The input branch is strictly the frozen runtime-O/model-input cache.  GT Scan
meshes enter only through the label manifest and are used to create target
latents.  The resulting pose-lifting manifest is directly consumable by
``train_native_ss_genrecon``; the SLat target root is consumed later when the
adapted Native-SS deployment materializes its support cache.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import sys
from typing import Any

os.environ.setdefault("SPCONV_ALGO", "native")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
os.environ.setdefault("TORCH_EXTENSIONS_DIR", "/tmp/torch_extensions")

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import trimesh


TRACKER_ROOT = Path(__file__).resolve().parents[2]
for dependency in (
    TRACKER_ROOT,
    TRACKER_ROOT / "Pixal3D",
    TRACKER_ROOT / "ReconViaGen",
    TRACKER_ROOT / "ReconViaGen" / "wheels" / "vggt",
):
    if str(dependency) not in sys.path:
        sys.path.insert(0, str(dependency))

from ar_ss_flow.pose_lifting import (  # noqa: E402
    LIFTING_CACHE_VERSION,
    LIFTING_METADATA_NAMES,
    schema_hash,
)
from ar_ss_flow.shared_object_preprocessing import (  # noqa: E402
    canonical_json_sha256,
)
from pixal3d_multiview.dataset_tools.repair_object_level_ss_dataset import (  # noqa: E402
    coords_from_points,
    decode_threshold_coords,
    deterministic_surface_points,
    encode_sparse_latent,
    load_decoder,
    load_encoder,
    load_meshes,
    stable_object_seed,
)
from pose_point_depth_mv.build_local_lh_slats import (  # noqa: E402
    LOCAL_LH_SLAT_VERSION,
    OFFICIAL_SLAT_ENCODER_SHA256,
)
from pose_point_depth_mv.dataset_tools.prepare_omni_real_label_cache import (  # noqa: E402
    MANIFEST_FORMAT as LABEL_MANIFEST_FORMAT,
)
from pose_point_depth_mv.dataset_tools.prepare_omni_real_model_inputs import (  # noqa: E402
    FEATURE_CONTRACT,
    MANIFEST_FORMAT as MODEL_INPUT_MANIFEST_FORMAT,
    OBJECT_FORMAT as MODEL_INPUT_OBJECT_FORMAT,
)
from pose_point_depth_mv.dataset_tools.prepare_omni_real_runtime_inputs import (  # noqa: E402
    MANIFEST_FORMAT as RUNTIME_MANIFEST_FORMAT,
)
from pose_point_depth_mv.dataset_tools.prepare_omni_real_video_cache import (  # noqa: E402
    utc_now,
)
from pose_point_depth_mv.native_3d_condition import (  # noqa: E402
    sparse_projection_geometry,
)
from pose_point_depth_mv.omni_real_benchmark_common import (  # noqa: E402
    atomic_json,
    atomic_torch_save,
    load_json,
    object_key,
    select_rows,
    sha256_file,
    to_cpu_tree,
)


TRAINING_CACHE_FORMAT = "pose_point_depth_mv.omni_real_native_training_cache.v1"
OBJECT_FORMAT = "pose_point_depth_mv.omni_real_native_training_object.v1"
MARKER_FORMAT = "pose_point_depth_mv.omni_real_native_training_marker.v1"
QUALITY_REJECTION_FORMAT = (
    "pose_point_depth_mv.omni_real_native_training_quality_rejection.v1"
)


class TargetQualityError(RuntimeError):
    """A per-object label/input compatibility failure, not a batch failure."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = str(code)
        self.details = dict(details or {})


def target_quality_rejection(
    object_key_value: str,
    phase: str,
    error: TargetQualityError,
) -> dict[str, Any]:
    return {
        "object_key": str(object_key_value),
        "phase": str(phase),
        "code": error.code,
        "error": repr(error),
        "details": error.details,
    }


def load_quality_rejection(
    path: Path,
    *,
    object_key_value: str,
    config_hash: str,
) -> dict[str, Any]:
    record = load_json(path)
    if (
        record.get("format") != QUALITY_REJECTION_FORMAT
        or record.get("object_key") != str(object_key_value)
        or record.get("config_hash") != str(config_hash)
    ):
        raise RuntimeError(f"stale target-quality rejection: {path}")
    return record


def persist_quality_rejection(
    path: Path,
    *,
    object_key_value: str,
    phase: str,
    error: TargetQualityError,
    config_hash: str,
) -> dict[str, Any]:
    record = {
        "format": QUALITY_REJECTION_FORMAT,
        "created_at_utc": utc_now(),
        "config_hash": str(config_hash),
        **target_quality_rejection(object_key_value, phase, error),
    }
    atomic_json(path, record)
    return record


def training_cache_admission(
    *,
    selected_object_count: int,
    completed_object_count: int,
    hard_failure_count: int,
    quality_rejection_count: int,
    allow_target_quality_rejections: bool,
    min_completed_objects: int,
) -> dict[str, Any]:
    selected = int(selected_object_count)
    completed = int(completed_object_count)
    hard_failures = int(hard_failure_count)
    quality_rejections = int(quality_rejection_count)
    minimum = int(min_completed_objects)
    if min(selected, completed, hard_failures, quality_rejections) < 0:
        raise ValueError("admission counts must be non-negative")
    if minimum <= 0:
        raise ValueError("min_completed_objects must be positive")
    checks = {
        "completed_nonempty": completed > 0,
        "no_hard_failures": hard_failures == 0,
        "all_selected_objects_accounted_for": (
            completed + hard_failures + quality_rejections == selected
        ),
        "target_quality_rejections_allowed": (
            quality_rejections == 0 or bool(allow_target_quality_rejections)
        ),
        "minimum_completed_objects": completed >= minimum,
    }
    return {
        "policy": {
            "allow_target_quality_rejections": bool(
                allow_target_quality_rejections
            ),
            "min_completed_objects": minimum,
            "hard_failures_are_never_admissible": True,
        },
        "checks": checks,
        "passed": all(checks.values()),
    }


def atomic_savez(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().contiguous().cpu()
    import hashlib

    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
    digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def tensor_tree_sha256(value: Any) -> str:
    import hashlib

    digest = hashlib.sha256()

    def visit(item: Any) -> None:
        if torch.is_tensor(item):
            digest.update(tensor_sha256(item).encode("ascii"))
        elif isinstance(item, dict):
            for key in sorted(item):
                digest.update(str(key).encode("utf-8"))
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)
        else:
            digest.update(repr(item).encode("utf-8"))

    visit(value)
    return digest.hexdigest()


def build_lifting_config(runtime_build_config: dict[str, Any]) -> dict[str, Any]:
    """Return the split-invariant contract seen by Native SS lifting.

    Dataset manifests and target artifacts remain SHA-bound by the surrounding
    training-cache admission report.  They must not enter this hash because the
    evaluator intentionally requires train and held-out caches to share the
    same model-visible feature contract.
    """

    geometry_contract = {
        "version": "ar_ss_flow.shared_object_preprocessing.v1",
        "resolution": int(runtime_build_config["feature_resolution"]),
        "foreground_margin": float(runtime_build_config["foreground_margin"]),
        "alpha_threshold": float(runtime_build_config["alpha_threshold"]),
        "geometry": (
            "foreground_bbox_square_crop_with_out_of_frame_padding_then_resize"
        ),
        "background": "black_after_resized_alpha_threshold",
        "rgb_resampling": "bilinear",
        "mask_resampling": "bilinear",
        "intrinsics_rule": "K_feature=source_to_feature_affine@K_source",
        "affine_pixel_convention": "u_feature=s*u_source-s*crop_left",
    }
    return {
        "pretrained": "Stable-X/trellis-vggt-v0-2",
        "vggt_pretrained": "precomputed by prepare_omni_real_model_inputs.v2",
        "image_resolution": int(runtime_build_config["feature_resolution"]),
        "geometric_preprocessing": geometry_contract,
        "geometric_preprocessing_hash": canonical_json_sha256(geometry_contract),
        "vggt_feature_index": -1,
        "min_depth_matches": 0,
        "affine_improvement_ratio": 1.0,
        "save_correct_geometry": False,
        "stock_condition_source": "precomputed Native v2 input cache",
        "lifting_feature_source": "precomputed Native v2 input cache",
        "selected_view_count": int(runtime_build_config["selected_view_count"]),
        "coordinate_frame": "input-derived runtime-O",
        "lifting_extrinsics": "T_O2C_lifting (common Sim(3) scale removed)",
        "real_training_cache_format": TRAINING_CACHE_FORMAT,
    }


def voxelize_object_points(points_o: np.ndarray, resolution: int = 64) -> np.ndarray:
    points = np.asarray(points_o, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points):
        raise ValueError("runtime object points must be non-empty [N,3]")
    if not np.isfinite(points).all():
        raise ValueError("runtime object points contain non-finite values")
    inside = np.all((points >= -0.5) & (points < 0.5), axis=1)
    if not inside.any():
        raise RuntimeError("runtime point prior has no points inside object grid")
    return coords_from_points(points[inside], int(resolution))


def load_prepared_masks(paths: list[str]) -> torch.Tensor:
    masks: list[np.ndarray] = []
    for value in paths:
        with Image.open(value) as handle:
            mask = np.asarray(handle.convert("L"), dtype=np.float32) / 255.0
        if mask.shape != (518, 518):
            raise RuntimeError(f"prepared mask is not 518x518: {value}")
        masks.append(mask)
    result = torch.from_numpy(np.stack(masks)).float()
    if not bool(torch.isfinite(result).all().item()):
        raise RuntimeError("prepared masks contain non-finite values")
    return result


def source_geometry(
    runtime: dict[str, Any], runtime_arrays: dict[str, np.ndarray]
) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor, torch.Tensor]:
    condition = load_json(runtime["condition_record"])
    recorded = str(condition.pop("condition_sha256", ""))
    if recorded != str(runtime["condition_sha256"]):
        raise RuntimeError(f"runtime condition identity changed: {object_key(runtime)}")
    if canonical_json_sha256(condition) != recorded:
        raise RuntimeError(f"runtime condition record changed: {object_key(runtime)}")
    geometry = dict(condition["shared_image_geometry"])
    contract = dict(geometry["contract"])
    affines = np.asarray(runtime_arrays["source_to_feature_affine"], dtype=np.float32)
    feature_intrinsics = np.asarray(runtime_arrays["K_feature"], dtype=np.float32)
    source_intrinsics = np.asarray(
        [row["output_K"] for row in condition["undistortion"]], dtype=np.float32
    )
    if not np.allclose(
        affines @ source_intrinsics, feature_intrinsics, rtol=1.0e-5, atol=1.0e-3
    ):
        raise RuntimeError(f"K_feature != A@K_source: {object_key(runtime)}")
    view_ids = np.asarray(runtime_arrays["selected_source_view_index"], dtype=np.int64)
    identity = {
        "shared_geometry_hash": geometry["geometry_hash"],
        "view_ids": view_ids.tolist(),
        "source_intrinsics": source_intrinsics.tolist(),
        "feature_intrinsics": feature_intrinsics.tolist(),
    }
    preprocessing = {
        "shared_geometry": contract,
        "shared_geometry_hash": geometry["geometry_hash"],
        "stock_condition": "precomputed Native v2 model-input condition",
        "lifting_features": "precomputed Native v2 VGGT+DINO patch features",
        "source_to_feature_affines": affines.tolist(),
        "crop_boxes_xyxy": geometry["crop_boxes_xyxy"],
        "foreground_retained_fractions": geometry[
            "foreground_retained_fractions"
        ],
        "intrinsics_rule": "K_feature=A@K_source",
        "sample_geometry_identity_hash": canonical_json_sha256(identity),
    }
    return (
        preprocessing,
        torch.from_numpy(source_intrinsics),
        torch.from_numpy(affines),
        torch.from_numpy(view_ids.astype(np.int32)),
    )


def mesh_target_coords(
    mesh_path: Path,
    *,
    object_uid: str,
    surface_points: int,
    seed: int,
    max_surface_outside_ratio: float,
) -> tuple[trimesh.Trimesh, np.ndarray, dict[str, Any]]:
    meshes = load_meshes(str(mesh_path))
    mesh = trimesh.util.concatenate(meshes)
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    surface_seed = stable_object_seed(int(seed), object_uid)
    points = deterministic_surface_points(
        mesh, vertices, int(surface_points), surface_seed
    )
    inside = np.all((points >= -0.5) & (points < 0.5), axis=1)
    outside_ratio = float(1.0 - inside.mean())
    if outside_ratio > float(max_surface_outside_ratio):
        raise TargetQualityError(
            "mesh_o_outside_runtime_grid",
            f"Mesh_O leaves runtime object grid: outside={outside_ratio:.6f} "
            f"> {max_surface_outside_ratio:.6f}",
            details={
                "surface_outside_ratio": outside_ratio,
                "max_surface_outside_ratio": float(max_surface_outside_ratio),
                "surface_point_count": int(len(points)),
                "surface_seed": int(surface_seed),
            },
        )
    coords = coords_from_points(points[inside], 64)
    if not len(coords):
        raise TargetQualityError(
            "mesh_o_empty_occupancy",
            "Mesh_O produced no occupied voxels",
            details={
                "surface_outside_ratio": outside_ratio,
                "surface_point_count": int(len(points)),
                "surface_seed": int(surface_seed),
            },
        )
    stats = {
        "surface_seed": int(surface_seed),
        "surface_point_count": int(len(points)),
        "surface_outside_ratio": outside_ratio,
        "mesh_voxel_count": int(len(coords)),
        "mesh_bounds_min": vertices.min(axis=0).tolist(),
        "mesh_bounds_max": vertices.max(axis=0).tolist(),
    }
    return mesh, coords, stats


def fuse_precomputed_dino(
    *,
    visual_patch_features: torch.Tensor,
    coords: np.ndarray,
    intrinsics: torch.Tensor,
    extrinsics: torch.Tensor,
    masks: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, Any]]:
    visual = visual_patch_features[..., -1024:].to(device=device, dtype=torch.float32)
    coords4 = torch.cat(
        (
            torch.zeros((len(coords), 1), dtype=torch.int32),
            torch.from_numpy(np.asarray(coords, dtype=np.int32)),
        ),
        dim=1,
    ).to(device)
    intrinsics = intrinsics.to(device=device, dtype=torch.float32)
    extrinsics = extrinsics.to(device=device, dtype=torch.float32)
    masks = masks.to(device=device, dtype=torch.float32)
    views, patches, channels = map(int, visual.shape)
    patch_side = int(round(patches**0.5))
    if patch_side * patch_side != patches or channels != 1024:
        raise RuntimeError("precomputed DINO patch layout is invalid")
    geometry = sparse_projection_geometry(
        coords=coords4,
        resolution=64,
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        grid_transform="identity",
        extrinsics_type="w2c",
        camera_forward_sign=1.0,
        image_height=518,
        image_width=518,
        patch_grid_side=patch_side,
    )
    patch_maps = visual.permute(0, 2, 1).reshape(
        views, channels, patch_side, patch_side
    )
    sampled = F.grid_sample(
        patch_maps,
        geometry["patch_grid"][:, :, None, :].float(),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )[..., 0].permute(0, 2, 1)
    valid = geometry["valid"].bool()
    sampled = sampled * valid[..., None].to(sampled.dtype)
    # TRELLIS SLat target encoding averages all views, including zero padding.
    fused = sampled.sum(dim=0) / float(views)
    sampled_masks = F.grid_sample(
        masks[:, None],
        geometry["image_grid"][:, :, None, :].float(),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )[:, 0, :, 0]
    mask_support = (sampled_masks > 0.5) & valid
    stats = {
        "view_count": views,
        "point_count": int(len(coords)),
        "visible_point_ratio": float(valid.any(dim=0).float().mean().item()),
        "mask_supported_point_ratio": float(
            mask_support.any(dim=0).float().mean().item()
        ),
        "mean_visible_views": float(valid.float().sum(dim=0).mean().item()),
        "mean_mask_support_views": float(
            mask_support.float().sum(dim=0).mean().item()
        ),
        "feature_abs_mean": float(fused.abs().mean().item()),
        "feature_abs_max": float(fused.abs().amax().item()),
    }
    if fused.shape != (len(coords), 1024) or not bool(
        torch.isfinite(fused).all().item()
    ):
        raise RuntimeError("precomputed DINO fusion returned invalid features")
    return fused, stats


def sparse_iou(left: np.ndarray, right: np.ndarray) -> float:
    left_set = {tuple(map(int, row[-3:])) for row in np.asarray(left)}
    right_set = {tuple(map(int, row[-3:])) for row in np.asarray(right)}
    union = left_set | right_set
    return float(len(left_set & right_set) / len(union)) if union else 1.0


def build_lifting_sample(
    *,
    runtime: dict[str, Any],
    model_report: dict[str, Any],
    model_payload: dict[str, Any],
    ss_latent: Path,
    uid: str,
) -> dict[str, Any]:
    runtime_path = Path(runtime["cache_npz"]).resolve()
    with np.load(runtime_path, allow_pickle=False) as handle:
        arrays = {name: np.asarray(handle[name]) for name in handle.files}
    required = {"K_feature", "T_O2C_lifting", "P_O", "source_to_feature_affine"}
    missing = sorted(required.difference(arrays))
    if missing:
        raise RuntimeError(f"runtime v2 cache lacks {missing}: {runtime_path}")
    preprocessing, source_k, affines, view_ids = source_geometry(runtime, arrays)
    masks = load_prepared_masks(list(runtime["prepared_mask_paths"]))
    views = int(masks.shape[0])
    intrinsics = torch.from_numpy(arrays["K_feature"].astype(np.float32))
    extrinsics = torch.from_numpy(arrays["T_O2C_lifting"].astype(np.float32))
    prior_coords = voxelize_object_points(arrays["P_O"])
    slat_condition = to_cpu_tree(model_payload["slat_condition"])
    sample = {
        "format": LIFTING_CACHE_VERSION,
        "uid": uid,
        "object_uid": uid,
        "visual_patch_features": model_payload["visual_patch_features"].cpu(),
        "predicted_depth": torch.zeros((views, 518, 518), dtype=torch.float16),
        "depth_confidence": torch.ones((views, 518, 518), dtype=torch.float16),
        "masks": masks.to(torch.float16),
        "intrinsics": intrinsics,
        "extrinsics": extrinsics,
        "prior_coords": torch.from_numpy(prior_coords.astype(np.int32)),
        "prior_confidence": torch.ones((len(prior_coords),), dtype=torch.float32),
        "stock_condition": model_payload["stock_condition"].cpu(),
        "slat_condition": slat_condition,
        "slat_condition_provenance": {
            "source": "prepare_omni_real_model_inputs.v2",
            "model_input": str(Path(model_report["model_input"]).resolve()),
            "model_input_sha256": str(model_report["model_input_sha256"]),
            "condition_sha256": str(runtime["condition_sha256"]),
            "condition_tree_sha256": tensor_tree_sha256(slat_condition),
        },
        "ss_latent": str(ss_latent.resolve()),
        "grid_transform": "identity",
        "extrinsics_type": "w2c",
        "camera_forward_sign": 1.0,
        "feature_image_size": [518, 518],
        "image_paths": list(runtime["prepared_rgb_paths"]),
        "mask_paths": list(runtime["prepared_mask_paths"]),
        "view_ids": view_ids,
        "source_intrinsics": source_k,
        "source_to_feature_affines": affines,
        "source_image_sizes_wh": load_json(runtime["condition_record"])[
            "shared_image_geometry"
        ]["source_sizes_wh"],
        "preprocessing": preprocessing,
        "depth_calibration": {
            "enabled": False,
            "reason": "Native v2 frustum-only path does not consume VGGT depth",
        },
        "runtime_condition_sha256": str(runtime["condition_sha256"]),
        "real_training_cache_format": TRAINING_CACHE_FORMAT,
    }
    return sample


def load_joined_rows(
    runtime_manifest: Path,
    model_manifest: Path,
    label_manifest: Path,
    selectors: list[str] | None,
) -> tuple[dict[str, Any], list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]]]:
    runtime = load_json(runtime_manifest)
    model = load_json(model_manifest)
    labels = load_json(label_manifest)
    expected = (
        (runtime, RUNTIME_MANIFEST_FORMAT, "runtime"),
        (model, MODEL_INPUT_MANIFEST_FORMAT, "model input"),
        (labels, LABEL_MANIFEST_FORMAT, "label"),
    )
    for payload, format_name, label in expected:
        if payload.get("format") != format_name or payload.get("passed") is not True:
            raise RuntimeError(f"{label} manifest did not pass or has stale format")
    runtime_digest = sha256_file(runtime_manifest)
    if str(model.get("runtime_input_manifest_sha256")) != runtime_digest:
        raise RuntimeError("model-input manifest binds a different runtime manifest")
    if str(labels.get("runtime_input_manifest_sha256")) != runtime_digest:
        raise RuntimeError("label manifest binds a different runtime manifest")
    selected = select_rows(list(runtime["objects"]), selectors)
    model_by_key = {object_key(row): row for row in model["objects"]}
    label_by_key = {object_key(row): row for row in labels["objects"]}
    joined = []
    for row in selected:
        key = object_key(row)
        if key not in model_by_key or key not in label_by_key:
            raise RuntimeError(f"incomplete runtime/model/label join: {key}")
        model_row = model_by_key[key]
        label_row = label_by_key[key]
        identities = {
            str(row["condition_sha256"]),
            str(model_row["condition_sha256"]),
            str(label_row["condition_sha256"]),
        }
        if len(identities) != 1:
            raise RuntimeError(f"condition identity differs across join: {key}")
        runtime_cache = Path(row["cache_npz"]).resolve()
        runtime_cache_sha = sha256_file(runtime_cache)
        if runtime_cache_sha != str(model_row["runtime_cache_sha256"]):
            raise RuntimeError(f"model input binds a different runtime cache: {key}")
        if runtime_cache_sha != str(label_row["runtime_cache_sha256"]):
            raise RuntimeError(f"label binds a different runtime cache: {key}")
        model_file = Path(model_row["model_input"]).resolve()
        mesh_file = Path(label_row["mesh_o"]).resolve()
        if sha256_file(model_file) != str(model_row["model_input_sha256"]):
            raise RuntimeError(f"model input artifact changed: {key}")
        if sha256_file(mesh_file) != str(label_row["mesh_o_sha256"]):
            raise RuntimeError(f"runtime-O mesh label changed: {key}")
        joined.append((row, model_row, label_row))
    return runtime, joined


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime_input_manifest", required=True)
    parser.add_argument("--model_input_manifest", required=True)
    parser.add_argument("--runtime_o_label_manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--ss_encoder_pretrained",
        default="microsoft/TRELLIS-image-large/ckpts/ss_enc_conv3d_16l8_fp16",
    )
    parser.add_argument("--ss_decoder_pretrained", default="Stable-X/trellis-vggt-v0-2")
    parser.add_argument(
        "--slat_encoder_prefix",
        default=(
            "/data/zjr/models/microsoft_TRELLIS-image-large/ckpts/"
            "slat_enc_swin8_B_64l8_fp16"
        ),
    )
    parser.add_argument(
        "--expected_slat_encoder_sha256", default=OFFICIAL_SLAT_ENCODER_SHA256
    )
    parser.add_argument("--surface_points", type=int, default=160000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_surface_outside_ratio", type=float, default=0.05)
    parser.add_argument("--min_visible_point_ratio", type=float, default=0.50)
    parser.add_argument("--min_mask_supported_point_ratio", type=float, default=0.50)
    parser.add_argument("--min_slat_ss_frame_iou", type=float, default=0.90)
    parser.add_argument("--latent_dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--object", action="append")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow_failures", action="store_true")
    parser.add_argument(
        "--allow_target_quality_rejections",
        action="store_true",
        help=(
            "Reject object-local Mesh_O/SLat compatibility failures while keeping "
            "hard construction and identity failures fatal."
        ),
    )
    parser.add_argument(
        "--min_completed_objects",
        type=int,
        help="Minimum admitted objects after target-quality rejection (default: all).",
    )
    return parser


@torch.no_grad()
def main() -> None:
    args = make_parser().parse_args()
    if int(args.surface_points) <= 0:
        raise ValueError("surface_points must be positive")
    for name in (
        "max_surface_outside_ratio",
        "min_visible_point_ratio",
        "min_mask_supported_point_ratio",
        "min_slat_ss_frame_iou",
    ):
        if not 0.0 <= float(getattr(args, name)) <= 1.0:
            raise ValueError(f"{name} must lie in [0,1]")
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("official SS/SLat target encoders require CUDA")
    torch.cuda.set_device(0 if device.index is None else int(device.index))

    runtime_path = Path(args.runtime_input_manifest).expanduser().resolve()
    model_path = Path(args.model_input_manifest).expanduser().resolve()
    label_path = Path(args.runtime_o_label_manifest).expanduser().resolve()
    runtime_manifest, joined = load_joined_rows(
        runtime_path, model_path, label_path, args.object
    )
    min_completed_objects = (
        len(joined)
        if args.min_completed_objects is None
        else int(args.min_completed_objects)
    )
    if min_completed_objects <= 0 or min_completed_objects > len(joined):
        raise ValueError(
            "min_completed_objects must be positive and no greater than the "
            f"selected object count ({len(joined)})"
        )
    output_dir = Path(args.output_dir).expanduser().resolve()
    config = {
        "format": TRAINING_CACHE_FORMAT,
        "runtime_input_manifest": str(runtime_path),
        "runtime_input_manifest_sha256": sha256_file(runtime_path),
        "model_input_manifest": str(model_path),
        "model_input_manifest_sha256": sha256_file(model_path),
        "runtime_o_label_manifest": str(label_path),
        "runtime_o_label_manifest_sha256": sha256_file(label_path),
        "ss_encoder_pretrained": str(args.ss_encoder_pretrained),
        "ss_decoder_pretrained": str(args.ss_decoder_pretrained),
        "ss_target_mode": "decoder_projected",
        "slat_encoder_prefix": str(Path(args.slat_encoder_prefix).resolve()),
        "slat_encoder_weights_sha256": sha256_file(
            Path(f"{Path(args.slat_encoder_prefix).resolve()}.safetensors")
        ),
        "surface_points": int(args.surface_points),
        "seed": int(args.seed),
        "max_surface_outside_ratio": float(args.max_surface_outside_ratio),
        "min_visible_point_ratio": float(args.min_visible_point_ratio),
        "min_mask_supported_point_ratio": float(args.min_mask_supported_point_ratio),
        "min_slat_ss_frame_iou": float(args.min_slat_ss_frame_iou),
        "coordinate_frame": "input-derived runtime-O; no per-target renormalization",
        "lifting_extrinsics": "T_O2C_lifting only",
        "slat_feature_source": "precomputed trailing 1024-D Native v2 DINO tokens",
    }
    if config["slat_encoder_weights_sha256"] != str(
        args.expected_slat_encoder_sha256
    ):
        raise RuntimeError("official SLat encoder SHA256 differs")
    config_hash = canonical_json_sha256(config)
    run_config = {"config": config, "config_hash": config_hash}
    output_existed = output_dir.exists()
    output_dir.mkdir(parents=True, exist_ok=True)
    binding_path = output_dir / "run_config.json"
    if binding_path.is_file():
        if load_json(binding_path) != run_config:
            raise RuntimeError("real Native training cache resume binding changed")
    elif output_existed and any(output_dir.iterdir()):
        raise RuntimeError("unbound real Native training cache directory")
    else:
        atomic_json(binding_path, run_config)
    if output_existed and not args.resume and any(
        path.name != "run_config.json" for path in output_dir.iterdir()
    ):
        raise FileExistsError("output has artifacts; pass --resume")

    # Phase 1: deterministic Mesh_O -> SS latent and decoder-projected support.
    ss_encoder = load_encoder(str(args.ss_encoder_pretrained), device)
    ss_decoder = load_decoder(str(args.ss_decoder_pretrained), device)
    mesh_stats_by_key: dict[str, dict[str, Any]] = {}
    failures: list[dict[str, str]] = []
    hard_failed_keys: set[str] = set()
    quality_rejections: list[dict[str, Any]] = []
    quality_rejected_keys: set[str] = set()
    for position, (runtime, _model, label) in enumerate(joined, start=1):
        key = object_key(runtime)
        uid = key.replace(":", "__")
        ss_path = output_dir / "ss_latents" / uid[:2] / f"{uid}.npz"
        rejection_path = (
            output_dir / "quality_rejections" / uid[:2] / f"{uid}.json"
        )
        try:
            if rejection_path.is_file():
                if not args.allow_target_quality_rejections:
                    raise RuntimeError(
                        f"target-quality rejection exists but is not allowed: "
                        f"{rejection_path}"
                    )
                rejection = load_quality_rejection(
                    rejection_path,
                    object_key_value=key,
                    config_hash=config_hash,
                )
                quality_rejections.append(rejection)
                quality_rejected_keys.add(key)
                continue
            if ss_path.is_file():
                with np.load(ss_path, allow_pickle=False) as existing:
                    if str(existing["config_hash"].item()) != config_hash:
                        raise RuntimeError(f"stale SS target: {ss_path}")
                    mesh_stats_by_key[key] = json.loads(
                        str(existing["mesh_stats_json"].item())
                    )
                continue
            mesh_path = Path(label["mesh_o"]).resolve()
            _mesh, mesh_coords, mesh_stats = mesh_target_coords(
                mesh_path,
                object_uid=uid,
                surface_points=int(args.surface_points),
                seed=int(args.seed),
                max_surface_outside_ratio=float(args.max_surface_outside_ratio),
            )
            z = encode_sparse_latent(
                ss_encoder, mesh_coords, 64, device, str(args.latent_dtype)
            )
            target_coords = decode_threshold_coords(ss_decoder, z, device, 64)
            atomic_savez(
                ss_path,
                z=z,
                target_coords=target_coords.astype(np.int32),
                mesh_target_coords=mesh_coords.astype(np.int32),
                source_glb=np.asarray(str(mesh_path)),
                object_uid=np.asarray(uid),
                surface_seed=np.asarray(mesh_stats["surface_seed"], dtype=np.int64),
                repair_format=np.asarray("object_level_ss_repair.v1"),
                repair_target_mode=np.asarray("decoder_projected"),
                coordinate_frame=np.asarray("runtime-O"),
                config_hash=np.asarray(config_hash),
                mesh_stats_json=np.asarray(json.dumps(mesh_stats, sort_keys=True)),
            )
            mesh_stats_by_key[key] = mesh_stats
            print(
                f"[real_native_targets:ss] {position}/{len(joined)} object={key} "
                f"mesh={len(mesh_coords)} decoded={len(target_coords)}",
                flush=True,
            )
        except TargetQualityError as error:
            if not args.allow_target_quality_rejections:
                raise
            quality_rejections.append(
                persist_quality_rejection(
                    rejection_path,
                    object_key_value=key,
                    phase="ss",
                    error=error,
                    config_hash=config_hash,
                )
            )
            quality_rejected_keys.add(key)
            print(
                f"[real_native_targets:REJECT] object={key} "
                f"code={error.code}: {error}",
                flush=True,
            )
        except Exception as error:
            failures.append({"object_key": key, "phase": "ss", "error": repr(error)})
            hard_failed_keys.add(key)
            print(f"[real_native_targets:FAIL] object={key}: {error!r}", flush=True)
            if not args.allow_failures:
                raise
    del ss_encoder, ss_decoder
    gc.collect()
    torch.cuda.empty_cache()

    # Phase 2: exact cached DINO views -> SLat encoder plus pose-lifting sample.
    import trellis.models as trellis_models
    from trellis.modules import sparse as sp

    slat_encoder = trellis_models.from_pretrained(
        str(Path(args.slat_encoder_prefix).resolve())
    ).to(device).eval()
    reports: list[dict[str, Any]] = []
    for position, (runtime, model_report, label) in enumerate(joined, start=1):
        key = object_key(runtime)
        if key in quality_rejected_keys or key in hard_failed_keys:
            continue
        uid = key.replace(":", "__")
        ss_path = output_dir / "ss_latents" / uid[:2] / f"{uid}.npz"
        slat_path = output_dir / "lh_slats" / f"shard-{uid[:2]}" / f"{uid}.npz"
        sample_path = output_dir / "samples" / uid[:2] / f"{uid}.pt"
        marker_path = output_dir / "objects" / runtime["category"] / runtime["object_id"] / "_COMPLETE.json"
        report_path = marker_path.with_name("report.json")
        rejection_path = (
            output_dir / "quality_rejections" / uid[:2] / f"{uid}.json"
        )
        try:
            if marker_path.is_file():
                marker = load_json(marker_path)
                if marker.get("format") != MARKER_FORMAT or marker.get(
                    "config_hash"
                ) != config_hash:
                    raise RuntimeError(f"stale object marker: {marker_path}")
                report = load_json(report_path)
                for name, path in (
                    ("ss", ss_path),
                    ("slat", slat_path),
                    ("lifting", sample_path),
                ):
                    if sha256_file(path) != report[f"{name}_sha256"]:
                        raise RuntimeError(f"reused {name} artifact changed: {path}")
                reports.append(report)
                continue
            model_file = Path(model_report["model_input"]).resolve()
            if sha256_file(model_file) != str(model_report["model_input_sha256"]):
                raise RuntimeError(f"model input changed: {model_file}")
            model_payload = torch.load(model_file, map_location="cpu")
            if (
                model_payload.get("format") != MODEL_INPUT_OBJECT_FORMAT
                or str(model_payload.get("object_key")) != key
            ):
                raise RuntimeError(f"invalid Native v2 model input: {model_file}")
            lifting = build_lifting_sample(
                runtime=runtime,
                model_report=model_report,
                model_payload=model_payload,
                ss_latent=ss_path,
                uid=uid,
            )
            with np.load(ss_path, allow_pickle=False) as ss_payload:
                target_coords = np.asarray(ss_payload["target_coords"], dtype=np.int32)
            fused, fusion_stats = fuse_precomputed_dino(
                visual_patch_features=lifting["visual_patch_features"],
                coords=target_coords,
                intrinsics=lifting["intrinsics"],
                extrinsics=lifting["extrinsics"],
                masks=lifting["masks"],
                device=device,
            )
            if fusion_stats["visible_point_ratio"] < float(
                args.min_visible_point_ratio
            ):
                raise TargetQualityError(
                    "insufficient_visible_slat_support",
                    f"insufficient visible SLat target support: {fusion_stats}",
                    details={"fusion_stats": fusion_stats},
                )
            if fusion_stats["mask_supported_point_ratio"] < float(
                args.min_mask_supported_point_ratio
            ):
                raise TargetQualityError(
                    "insufficient_mask_slat_support",
                    f"insufficient mask SLat target support: {fusion_stats}",
                    details={"fusion_stats": fusion_stats},
                )
            coords4 = torch.cat(
                (
                    torch.zeros((len(target_coords), 1), dtype=torch.int32),
                    torch.from_numpy(target_coords),
                ),
                dim=1,
            ).to(device)
            latent = slat_encoder(
                sp.SparseTensor(feats=fused.float(), coords=coords4),
                sample_posterior=False,
            )
            output_coords = latent.coords[:, 1:].detach().cpu().numpy().astype(np.uint8)
            output_feats = latent.feats.detach().float().cpu().numpy().astype(np.float32)
            frame_iou = sparse_iou(target_coords, output_coords)
            if frame_iou < float(args.min_slat_ss_frame_iou):
                raise TargetQualityError(
                    "slat_ss_runtime_o_frame_mismatch",
                    f"SLat/SS runtime-O frame IoU={frame_iou:.6f} < "
                    f"{args.min_slat_ss_frame_iou:.6f}",
                    details={
                        "slat_ss_frame_iou": float(frame_iou),
                        "min_slat_ss_frame_iou": float(args.min_slat_ss_frame_iou),
                    },
                )
            atomic_savez(
                slat_path,
                coords=output_coords,
                feats=output_feats,
                object_uid=np.asarray(uid),
                config_hash=np.asarray(config_hash),
                source_identity_hash=np.asarray(runtime["condition_sha256"]),
                fusion_stats_json=np.asarray(json.dumps(fusion_stats, sort_keys=True)),
            )
            atomic_torch_save(sample_path, lifting)
            report = {
                "format": OBJECT_FORMAT,
                "created_at_utc": utc_now(),
                "category": str(runtime["category"]),
                "object_id": str(runtime["object_id"]),
                "object_key": key,
                "uid": uid,
                "condition_sha256": str(runtime["condition_sha256"]),
                "mesh_o": str(Path(label["mesh_o"]).resolve()),
                "mesh_o_sha256": str(label["mesh_o_sha256"]),
                "ss_latent": str(ss_path.resolve()),
                "ss_sha256": sha256_file(ss_path),
                "slat_target": str(slat_path.resolve()),
                "slat_sha256": sha256_file(slat_path),
                "lifting_sample": str(sample_path.resolve()),
                "lifting_sha256": sha256_file(sample_path),
                "mesh_stats": mesh_stats_by_key[key],
                "fusion_stats": fusion_stats,
                "slat_ss_frame_iou": frame_iou,
                "passed": True,
            }
            marker_path.parent.mkdir(parents=True, exist_ok=True)
            atomic_json(report_path, report)
            atomic_json(
                marker_path,
                {
                    "format": MARKER_FORMAT,
                    "config_hash": config_hash,
                    "condition_sha256": str(runtime["condition_sha256"]),
                    "passed": True,
                },
            )
            reports.append(report)
            print(
                f"[real_native_targets:slat] {position}/{len(joined)} object={key} "
                f"points={len(output_coords)} frame_iou={frame_iou:.4f}",
                flush=True,
            )
            del fused, latent, model_payload, lifting
            torch.cuda.empty_cache()
        except TargetQualityError as error:
            if not args.allow_target_quality_rejections:
                raise
            quality_rejections.append(
                persist_quality_rejection(
                    rejection_path,
                    object_key_value=key,
                    phase="slat",
                    error=error,
                    config_hash=config_hash,
                )
            )
            quality_rejected_keys.add(key)
            print(
                f"[real_native_targets:REJECT] object={key} "
                f"code={error.code}: {error}",
                flush=True,
            )
        except Exception as error:
            failures.append({"object_key": key, "phase": "slat", "error": repr(error)})
            hard_failed_keys.add(key)
            print(f"[real_native_targets:FAIL] object={key}: {error!r}", flush=True)
            if not args.allow_failures:
                raise
    del slat_encoder
    gc.collect()
    torch.cuda.empty_cache()

    preprocessing = dict(runtime_manifest["build_config"])
    lifting_config = build_lifting_config(preprocessing)
    lifting_config_hash = canonical_json_sha256(lifting_config)
    report_by_key = {row["object_key"]: row for row in reports}
    samples = []
    for runtime, _model, _label in joined:
        key = object_key(runtime)
        if key not in report_by_key:
            continue
        report = report_by_key[key]
        samples.append(
            {
                "uid": report["uid"],
                "object_uid": report["uid"],
                "cache_file": str(Path(report["lifting_sample"]).relative_to(output_dir)),
                "ss_latent": report["ss_latent"],
                "view_count": int(runtime["selected_view_count"]),
                "prior_point_count": int(
                    torch.load(report["lifting_sample"], map_location="cpu")[
                        "prior_coords"
                    ].shape[0]
                ),
                "depth_calibration_enabled": False,
                "depth_match_count": 0,
                "source": "omni_real_video",
            }
        )
    lifting_manifest = {
        "format": LIFTING_CACHE_VERSION,
        "output_dir": str(output_dir),
        "source_cache_manifest": str(model_path),
        "stock_condition_source": "precomputed Native v2 runtime-O model inputs",
        "lifting_feature_source": "precomputed Native v2 runtime-O model inputs",
        "sample_count": len(samples),
        "object_count": len(samples),
        "failure_count": len(failures),
        "feature_metadata": {
            "aggregated_layer_count": 24,
            "selected_vggt_feature_index": 23,
            "patch_start_idx": int(FEATURE_CONTRACT["patch_start_idx"]),
            "patch_count": int(FEATURE_CONTRACT["patch_count"]),
            "vggt_feature_dim": 2048,
            "dino_feature_dim": 1024,
            "depth_shape": [518, 518],
        },
        "visual_feature_dim": 3072,
        "metadata_names": list(LIFTING_METADATA_NAMES),
        "metadata_schema_hash": schema_hash(),
        "config": lifting_config,
        "config_hash": lifting_config_hash,
        "samples": samples,
        "split_identity": "omni_real_runtime_o.v1",
    }
    admission_decision = training_cache_admission(
        selected_object_count=len(joined),
        completed_object_count=len(samples),
        hard_failure_count=len(failures),
        quality_rejection_count=len(quality_rejections),
        allow_target_quality_rejections=bool(args.allow_target_quality_rejections),
        min_completed_objects=min_completed_objects,
    )
    passed = bool(admission_decision["passed"])
    lifting_manifest["passed"] = passed
    lifting_manifest["training_ready"] = passed
    lifting_manifest["quality_rejection_count"] = len(quality_rejections)
    lifting_manifest["quality_rejections"] = quality_rejections
    lifting_manifest["admission"] = admission_decision
    atomic_json(output_dir / "lifting_manifest.json", lifting_manifest)

    slat_config = {
        "format": LOCAL_LH_SLAT_VERSION,
        "source_kind": "GT Mesh_O plus exact cached real multiview DINO tokens",
        "real_training_cache_config_hash": config_hash,
        "coordinate_source": "decoder-projected SS target_coords in runtime-O",
        "feature_fusion": "all-view arithmetic mean with invalid zero padding",
        "posterior": "official SLatEncoder mean (sample_posterior=False)",
    }
    slat_run_config = {
        "config": slat_config,
        "config_hash": canonical_json_sha256(slat_config),
    }
    atomic_json(output_dir / "lh_slats" / "run_config.json", slat_run_config)
    admission = {
        "format": TRAINING_CACHE_FORMAT,
        "created_at_utc": utc_now(),
        "config": config,
        "config_hash": config_hash,
        "selected_object_count": len(joined),
        "completed_object_count": len(samples),
        "objects": reports,
        "failures": failures,
        "quality_rejection_count": len(quality_rejections),
        "quality_rejections": quality_rejections,
        "admission": admission_decision,
        "lifting_manifest": str((output_dir / "lifting_manifest.json").resolve()),
        "slat_root": str((output_dir / "lh_slats").resolve()),
        "native_ss_training_ready": passed,
        "native_slat_target_ready": passed,
        "native_slat_training_ready": False,
        "native_slat_next_requirement": (
            "materialize Direct-SLAT support with the selected adapted Native-SS deployment"
        ),
        "training_ready": passed,
        "scope_guard": (
            "training_ready applies to Native-SS domain adaptation. Native-SLAT "
            "targets are ready, but its rollout support must bind the adapted SS."
        ),
        "passed": passed,
    }
    atomic_json(output_dir / "training_cache_manifest.json", admission)
    print(
        json.dumps(
            {
                "passed": passed,
                "objects": len(samples),
                "quality_rejections": len(quality_rejections),
                "hard_failures": len(failures),
                "native_ss_training_ready": passed,
                "native_slat_target_ready": passed,
                "native_slat_training_ready": False,
                "manifest": str(output_dir / "training_cache_manifest.json"),
            },
            indent=2,
        ),
        flush=True,
    )
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
