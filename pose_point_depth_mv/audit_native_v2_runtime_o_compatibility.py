#!/usr/bin/env python3
"""Audit reviewed1k Native v2 compatibility with the deployable runtime-O frame.

The audit keeps RGB/features fixed, applies a deterministic proper Sim(3) gauge
to cameras and sparse points, re-estimates runtime-O from observable inputs, and
replays the exact Native SS/SLat v2 projection and learned condition paths.  GT
targets are read only after the input frame is frozen and are never used to
estimate runtime-O.

This program deliberately separates two outcomes:

* ``passed`` means the runtime frontend is gauge-equivariant and compatible
  with the frozen parent-v2 condition path.
* ``reviewed_cache_reusable`` means the old reviewed1k target frame is also
  close enough to runtime-O to reuse its target/lifting cache unchanged.

The latter may be false while the implementation audit itself passes.  In that
case reviewed1k must be rebuilt with the runtime-O contract before mixed-domain
training.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from pose_point_depth_mv.native_slat_genrecon import project_sparse_frustum_dino
from pose_point_depth_mv.native_ss_genrecon import (
    EveryBlockConditionProjection,
    GenreconViewAggregator,
    project_frustum_dino,
)
from pose_point_depth_mv.real_object_canonicalization import (
    RuntimeObjectFrame,
    RuntimeObjectFrameConfig,
    apply_transform,
    canonicalize_runtime_object_frame,
    normalize_similarity_extrinsics,
    project_object_points,
    similarity_scale,
)


REPORT_FORMAT = "pose_point_depth_mv.native_v2_runtime_o_compatibility.v1"
DEFAULT_THRESHOLDS: dict[str, float] = {
    "frame_equivariance_max_abs": 3.0e-5,
    "object_point_equivariance_max_abs": 3.0e-5,
    "projection_roundtrip_max_px": 2.0e-4,
    "projected_feature_max_abs": 3.0e-4,
    "parent_condition_max_abs": 3.0e-4,
    "parent_flow_max_abs": 2.0e-2,
    "pose_separability_min": 1.0e-6,
    "canonical_rotation_max_deg": 5.0,
    "canonical_translation_max": 0.05,
    "canonical_scale_relative_error_max": 0.05,
    "target_occupancy_iou_min": 0.95,
}
PRODUCTION_FRAME_CONFIG = RuntimeObjectFrameConfig(min_object_points=100)
PRODUCTION_FRAME_VIEW_COUNT = 8


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def atomic_json(path: str | Path, value: Any) -> None:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, destination)


def _rotation_xyz(degrees: Iterable[float]) -> np.ndarray:
    ax, ay, az = (math.radians(float(value)) for value in degrees)
    cx, cy, cz = math.cos(ax), math.cos(ay), math.cos(az)
    sx, sy, sz = math.sin(ax), math.sin(ay), math.sin(az)
    rx = np.asarray(((1, 0, 0), (0, cx, -sx), (0, sx, cx)), dtype=np.float64)
    ry = np.asarray(((cy, 0, sy), (0, 1, 0), (-sy, 0, cy)), dtype=np.float64)
    rz = np.asarray(((cz, -sz, 0), (sz, cz, 0), (0, 0, 1)), dtype=np.float64)
    return rz @ ry @ rx


def deterministic_proper_similarity(seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    angles = rng.uniform(-50.0, 50.0, size=3)
    scale = float(rng.uniform(0.55, 2.25))
    translation = rng.uniform(-1.25, 1.25, size=3)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = scale * _rotation_xyz(angles)
    transform[:3, 3] = translation
    return transform


def transform_camera_gauge(T_W2C: np.ndarray, gauge: np.ndarray) -> np.ndarray:
    """Apply a world proper-Sim(3) while keeping each camera frame metric."""

    poses = np.asarray(T_W2C, dtype=np.float64)
    scale = similarity_scale(gauge)
    rotation = gauge[:3, :3] / scale
    translation = gauge[:3, 3]
    output = []
    for pose in poses:
        c2w = np.linalg.inv(pose)
        transformed = np.eye(4, dtype=np.float64)
        transformed[:3, :3] = rotation @ c2w[:3, :3]
        transformed[:3, 3] = scale * rotation @ c2w[:3, 3] + translation
        output.append(np.linalg.inv(transformed))
    return np.stack(output)


def grid_rotation(name: str) -> np.ndarray:
    if str(name) == "identity":
        return np.eye(3, dtype=np.float64)
    if str(name) == "pixal3d_rotation":
        return np.asarray(
            ((1.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0)),
            dtype=np.float64,
        )
    raise ValueError(f"unsupported grid transform={name!r}")


def native_coords_to_physical(
    coords: np.ndarray, *, resolution: int, grid_transform: str
) -> np.ndarray:
    values = np.asarray(coords, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] not in (3, 4):
        raise ValueError("native coordinates must be [N,3/4]")
    xyz = values[:, -3:]
    if np.any(xyz < 0) or np.any(xyz >= int(resolution)):
        raise ValueError("native coordinates are outside the requested resolution")
    canonical = (xyz + 0.5) / float(resolution) - 0.5
    return canonical @ grid_rotation(grid_transform).T


def physical_to_native_coords(points: np.ndarray, *, resolution: int) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64)
    coords = np.floor((values + 0.5) * float(resolution)).astype(np.int64)
    valid = np.all((coords >= 0) & (coords < int(resolution)), axis=1)
    if not bool(np.any(valid)):
        return np.empty((0, 3), dtype=np.int32)
    return np.unique(coords[valid].astype(np.int32), axis=0)


def coordinate_iou(left: np.ndarray, right: np.ndarray) -> float:
    left_set = {tuple(map(int, row)) for row in np.asarray(left)[:, -3:]}
    right_set = {tuple(map(int, row)) for row in np.asarray(right)[:, -3:]}
    union = left_set | right_set
    return float(len(left_set & right_set) / len(union)) if union else 1.0


def rotation_angle_degrees(rotation: np.ndarray) -> float:
    value = np.asarray(rotation, dtype=np.float64)
    cosine = float(np.clip((np.trace(value) - 1.0) * 0.5, -1.0, 1.0))
    return float(math.degrees(math.acos(cosine)))


def _to_w2c(sample: dict[str, Any]) -> np.ndarray:
    poses = np.asarray(sample["extrinsics"], dtype=np.float64)
    kind = str(sample["extrinsics_type"])
    if kind == "w2c":
        return poses
    if kind == "c2w":
        return np.linalg.inv(poses)
    raise ValueError(f"unsupported extrinsics_type={kind!r}")


def _reference_view(masks: np.ndarray) -> int:
    return max(
        range(len(masks)),
        key=lambda index: (int(np.count_nonzero(masks[index] > 0.5)), -index),
    )


def _make_projection_sample(
    source: dict[str, Any], frame: RuntimeObjectFrame
) -> dict[str, Any]:
    masks = source["masks"].detach().cpu().float()
    views, height, width = map(int, masks.shape)
    return {
        "visual_patch_features": source["visual_patch_features"].detach().cpu(),
        "predicted_depth": torch.zeros((views, height, width), dtype=torch.float32),
        "depth_confidence": torch.ones((views, height, width), dtype=torch.float32),
        "masks": masks,
        "intrinsics": source["intrinsics"].detach().cpu().float(),
        "extrinsics": torch.from_numpy(
            normalize_similarity_extrinsics(frame.T_O2C).astype(np.float32)
        ),
        "grid_transform": "identity",
        "extrinsics_type": "w2c",
        "camera_forward_sign": float(source["camera_forward_sign"]),
        "depth_calibration": {"enabled": False},
    }


def _prepare_runtime_pair(
    row: dict[str, Any],
    lifting_manifest: dict[str, Any],
    *,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    cache_path = _resolve_cache_file(lifting_manifest, row)
    sample = torch.load(cache_path, map_location="cpu")
    prior_coords = sample["prior_coords"].detach().cpu().numpy()
    points_w = native_coords_to_physical(
        prior_coords, resolution=64, grid_transform=str(sample["grid_transform"])
    )
    intrinsics = np.asarray(sample["intrinsics"], dtype=np.float64)
    poses = _to_w2c(sample)
    masks = np.asarray(sample["masks"].detach().cpu(), dtype=np.float64)
    reference = _reference_view(masks)
    baseline = canonicalize_runtime_object_frame(
        points_w,
        intrinsics,
        poses,
        masks,
        config=PRODUCTION_FRAME_CONFIG,
        reference_view_index=reference,
    )
    gauge = deterministic_proper_similarity(seed)
    transformed = canonicalize_runtime_object_frame(
        apply_transform(points_w, gauge),
        intrinsics,
        transform_camera_gauge(poses, gauge),
        masks,
        config=PRODUCTION_FRAME_CONFIG,
        reference_view_index=reference,
    )
    return sample, _make_projection_sample(sample, baseline), _make_projection_sample(
        sample, transformed
    )


def _to_device_tree(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device=device)
    if isinstance(value, dict):
        return {key: _to_device_tree(child, device) for key, child in value.items()}
    if isinstance(value, list):
        return [_to_device_tree(child, device) for child in value]
    if isinstance(value, tuple):
        return tuple(_to_device_tree(child, device) for child in value)
    return value


def _amp_dtype(name: str) -> torch.dtype | None:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    if name == "none":
        return None
    raise ValueError(f"unsupported amp dtype={name!r}")


def _clone_sparse(value: Any, sparse_module: Any) -> Any:
    return sparse_module.SparseTensor(
        feats=value.feats.clone(), coords=value.coords.clone()
    )


@torch.no_grad()
def run_parent_forward_rollout(
    *,
    row: dict[str, Any],
    lifting_manifest: dict[str, Any],
    slat_manifest: dict[str, Any],
    slat_by_uid: dict[str, dict[str, Any]],
    ss_checkpoint: dict[str, Any],
    slat_checkpoint: dict[str, Any],
    pretrained: str,
    device: torch.device,
    amp_name: str,
    seed: int,
    threshold: float,
) -> dict[str, Any]:
    """Run one exact parent SS/SLat adapted forward and decode its frozen target."""

    if device.type != "cuda":
        raise ValueError("parent forward rollout requires CUDA")
    sample, baseline_sample, transformed_sample = _prepare_runtime_pair(
        row, lifting_manifest, seed=seed
    )
    uid = str(row["uid"])
    slat_row = slat_by_uid[uid]
    amp_dtype = _amp_dtype(amp_name)
    autocast_enabled = amp_dtype is not None
    generator = torch.Generator(device=device).manual_seed(int(seed) + 700001)

    from pose_point_depth_mv.native_ss_genrecon import (
        build_native_ss_genrecon_components,
        load_trainable_state_dict as load_ss_state,
        validate_native_ss_genrecon_checkpoint,
    )

    validate_native_ss_genrecon_checkpoint(ss_checkpoint, pretrained=pretrained)
    ss_args = dict(ss_checkpoint["args"])
    _, ss_model, _, _, _ = build_native_ss_genrecon_components(
        pretrained=pretrained,
        lora_rank=int(ss_args["lora_rank"]),
        lora_alpha=int(ss_args["lora_alpha"]),
        condition_channels=int(ss_args["condition_channels"]),
        gradient_checkpointing=False,
        need_decoder=False,
        device=device,
    )
    load_ss_state(ss_model, ss_checkpoint["ema_trainable_state"])
    ss_model.eval()
    ss_x = torch.randn(
        (1, 8, 16, 16, 16), generator=generator, device=device, dtype=torch.float32
    )
    ss_t = torch.full((1,), 500.0, device=device)
    ss_condition = sample["stock_condition"].to(device=device)
    with torch.autocast(
        device_type="cuda", dtype=amp_dtype or torch.bfloat16, enabled=autocast_enabled
    ):
        ss_stock = ss_model.stock_prediction(ss_x, ss_t, ss_condition)
        ss_baseline, _ = ss_model.adapted_prediction(
            ss_x,
            ss_t,
            ss_condition,
            baseline_sample,
            stock_velocity=ss_stock,
        )
        ss_transformed, _ = ss_model.adapted_prediction(
            ss_x,
            ss_t,
            ss_condition,
            transformed_sample,
            stock_velocity=ss_stock,
        )
    ss_max_abs = _tensor_max_abs(ss_baseline, ss_transformed)
    ss_model.cpu()
    del ss_model, ss_x, ss_t, ss_condition, ss_stock, ss_baseline, ss_transformed
    gc.collect()
    torch.cuda.empty_cache()

    from pose_point_depth_mv.native_slat_genrecon_v2 import (
        build_native_slat_genrecon_v2_components,
        load_stock_slat_freeze,
        load_trainable_state_dict as load_slat_state,
        validate_native_slat_genrecon_v2_checkpoint,
    )
    from trellis.modules import sparse as sp

    stock_freeze_path = Path(str(slat_checkpoint["args"]["stock_slat_freeze"])).resolve()
    stock_freeze = load_stock_slat_freeze(stock_freeze_path)
    upstream = dict(slat_checkpoint["model_summary"]["upstream_native_ss"])
    validate_native_slat_genrecon_v2_checkpoint(
        slat_checkpoint,
        pretrained=pretrained,
        stock_slat_freeze=stock_freeze,
        upstream_native_ss=upstream,
    )
    slat_args = dict(slat_checkpoint["args"])
    _, slat_model, decoder, _, _, _ = build_native_slat_genrecon_v2_components(
        pretrained=pretrained,
        stock_slat_freeze=stock_freeze,
        upstream_native_ss=upstream,
        lora_rank=int(slat_args["lora_rank"]),
        lora_alpha=int(slat_args["lora_alpha"]),
        condition_channels=int(slat_args["condition_channels"]),
        gradient_checkpointing=False,
        need_decoder=True,
        device=device,
    )
    if decoder is None:
        raise RuntimeError("Native-SLAT decoder was not loaded")
    load_slat_state(slat_model, slat_checkpoint["ema_trainable_state"])
    slat_model.eval()
    decoder.eval()
    slat_root = Path(str(slat_manifest["output_dir"])).resolve()
    support = torch.load(
        _resolve_manifest_artifact(slat_manifest, slat_row, "support_file"),
        map_location="cpu",
    )
    condition_payload = torch.load(
        _resolve_manifest_artifact(slat_manifest, slat_row, "condition_file"),
        map_location="cpu",
    )
    coords = support["corrected_coords64"].to(device=device, dtype=torch.int32)
    feats = torch.randn(
        (len(coords), 8), generator=generator, device=device, dtype=torch.float32
    )
    slat_x = sp.SparseTensor(feats=feats, coords=coords)
    slat_t = torch.full((1,), 500.0, device=device)
    slat_condition = _to_device_tree(condition_payload["condition"]["cond"], device)
    with torch.autocast(
        device_type="cuda", dtype=amp_dtype or torch.bfloat16, enabled=autocast_enabled
    ):
        slat_stock = slat_model.stock_prediction(
            _clone_sparse(slat_x, sp), slat_t, slat_condition
        )
        slat_baseline, _ = slat_model.adapted_prediction(
            _clone_sparse(slat_x, sp),
            slat_t,
            slat_condition,
            baseline_sample,
            stock_velocity=_clone_sparse(slat_stock, sp),
        )
        slat_baseline_repeat, _ = slat_model.adapted_prediction(
            _clone_sparse(slat_x, sp),
            slat_t,
            slat_condition,
            baseline_sample,
            stock_velocity=_clone_sparse(slat_stock, sp),
        )
        slat_transformed, _ = slat_model.adapted_prediction(
            _clone_sparse(slat_x, sp),
            slat_t,
            slat_condition,
            transformed_sample,
            stock_velocity=_clone_sparse(slat_stock, sp),
        )
    slat_coords_equal = bool(torch.equal(slat_baseline.coords, slat_transformed.coords))
    slat_max_abs = _tensor_max_abs(slat_baseline.feats, slat_transformed.feats)
    slat_repeat_max_abs = _tensor_max_abs(
        slat_baseline.feats, slat_baseline_repeat.feats
    )
    slat_gauge_excess = max(0.0, slat_max_abs - slat_repeat_max_abs)

    target_path = _resolve_manifest_artifact(slat_manifest, slat_row, "target_file")
    with np.load(target_path, allow_pickle=False) as payload:
        target_coords3 = np.asarray(payload["coords"], dtype=np.int32)
        target_feats = np.asarray(payload["feats"], dtype=np.float32)
    target_coords = torch.cat(
        (
            torch.zeros((len(target_coords3), 1), dtype=torch.int32),
            torch.from_numpy(target_coords3),
        ),
        dim=1,
    ).to(device=device)
    target = sp.SparseTensor(
        feats=torch.from_numpy(target_feats).to(device=device), coords=target_coords
    )
    decoded = decoder(target)[0]
    mesh = decoded.to_trimesh(transform_pose=False)
    vertex_count = int(len(mesh.vertices))
    face_count = int(len(mesh.faces))
    decoder_passed = vertex_count > 0 and face_count > 0
    passed = (
        ss_max_abs <= float(threshold)
        and slat_coords_equal
        and slat_gauge_excess <= float(threshold)
        and decoder_passed
    )
    report = {
        "executed": True,
        "uid": uid,
        "device": str(device),
        "amp_dtype": amp_name,
        "weights": "ema",
        "same_noise_and_stock_context": True,
        "ss_parent_flow_max_abs": ss_max_abs,
        "slat_parent_flow_coords_equal": slat_coords_equal,
        "slat_parent_flow_max_abs": slat_max_abs,
        "slat_parent_flow_repeat_max_abs": slat_repeat_max_abs,
        "slat_parent_flow_gauge_excess_max_abs": slat_gauge_excess,
        "target_decoder": {
            "target": str(target_path),
            "target_sha256": sha256_file(target_path),
            "vertex_count": vertex_count,
            "face_count": face_count,
            "passed": decoder_passed,
        },
        "threshold": float(threshold),
        "passed": passed,
    }
    slat_model.cpu()
    decoder.cpu()
    del (
        slat_model,
        decoder,
        support,
        condition_payload,
        coords,
        feats,
        slat_x,
        slat_t,
        slat_condition,
        slat_stock,
        slat_baseline,
        slat_baseline_repeat,
        slat_transformed,
        target,
        decoded,
        mesh,
        sample,
        baseline_sample,
        transformed_sample,
    )
    gc.collect()
    torch.cuda.empty_cache()
    return report


def _tensor_max_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.shape != right.shape:
        return float("inf")
    return float((left.float() - right.float()).abs().max().item())


def _projection_difference(
    projected: torch.Tensor,
    valid: torch.Tensor,
    control_projected: torch.Tensor,
    control_valid: torch.Tensor,
) -> float:
    feature = float(
        (projected.float() - control_projected.float()).abs().mean().item()
    )
    visibility = float((valid != control_valid).float().mean().item())
    return feature + visibility


def _subset_columns(
    projected: torch.Tensor, valid: torch.Tensor, *, maximum: int = 96
) -> tuple[torch.Tensor, torch.Tensor]:
    supported = torch.nonzero(valid.any(dim=0), as_tuple=False).flatten()
    if not len(supported):
        supported = torch.arange(int(projected.shape[1]))
    if len(supported) > int(maximum):
        positions = torch.linspace(0, len(supported) - 1, int(maximum)).long()
        supported = supported.index_select(0, positions)
    return projected.index_select(1, supported), valid.index_select(1, supported)


def _checkpoint_condition_modules(
    checkpoint: dict[str, Any], *, weights: str = "ema"
) -> tuple[GenreconViewAggregator, EveryBlockConditionProjection]:
    state_key = "ema_trainable_state" if weights == "ema" else "model_trainable_state"
    state = checkpoint.get(state_key)
    if not isinstance(state, dict):
        raise RuntimeError(f"checkpoint lacks {state_key}")
    aggregator_state = {
        key.removeprefix("aggregator."): value.float()
        for key, value in state.items()
        if key.startswith("aggregator.")
    }
    block_state = {
        key.removeprefix("block_condition."): value.float()
        for key, value in state.items()
        if key.startswith("block_condition.")
    }
    weight_keys = sorted(
        key for key in block_state if key.startswith("projections.") and key.endswith(".weight")
    )
    if not aggregator_state or not weight_keys:
        raise RuntimeError("checkpoint condition-path state is incomplete")
    first_weight = block_state[weight_keys[0]]
    block_indices = {int(key.split(".")[1]) for key in weight_keys}
    if block_indices != set(range(len(block_indices))):
        raise RuntimeError("checkpoint block-condition indices are not contiguous")
    aggregator = GenreconViewAggregator(channels=1024).eval()
    projector = EveryBlockConditionProjection(
        condition_channels=int(first_weight.shape[1]),
        flow_channels=int(first_weight.shape[0]),
        blocks=len(block_indices),
    ).eval()
    aggregator.load_state_dict(aggregator_state, strict=True)
    projector.load_state_dict(block_state, strict=True)
    return aggregator, projector


@torch.no_grad()
def replay_parent_condition_path(
    checkpoint: dict[str, Any],
    baseline_projected: torch.Tensor,
    baseline_valid: torch.Tensor,
    transformed_projected: torch.Tensor,
    transformed_valid: torch.Tensor,
) -> dict[str, float]:
    aggregator, projector = _checkpoint_condition_modules(checkpoint)
    baseline_projected, baseline_valid = _subset_columns(
        baseline_projected, baseline_valid
    )
    transformed_projected, transformed_valid = _subset_columns(
        transformed_projected, transformed_valid
    )
    baseline_condition, _ = aggregator(baseline_projected, baseline_valid)
    transformed_condition, _ = aggregator(transformed_projected, transformed_valid)
    condition_max = _tensor_max_abs(baseline_condition, transformed_condition)
    residual_max = 0.0
    for index in range(len(projector.projections)):
        residual_max = max(
            residual_max,
            _tensor_max_abs(
                projector(index, baseline_condition),
                projector(index, transformed_condition),
            ),
        )
    del aggregator, projector
    return {
        "aggregated_condition_max_abs": condition_max,
        "every_block_residual_max_abs": residual_max,
        "max_abs": max(condition_max, residual_max),
    }


def _resolve_cache_file(manifest: dict[str, Any], row: dict[str, Any]) -> Path:
    path = Path(str(row["cache_file"]))
    if path.is_absolute():
        return path.resolve()
    return (Path(str(manifest["output_dir"])) / path).resolve()


def _resolve_manifest_artifact(
    manifest: dict[str, Any], row: dict[str, Any], key: str
) -> Path:
    path = Path(str(row[key]))
    if path.is_absolute():
        return path.resolve()
    return (Path(str(manifest["output_dir"])) / path).resolve()


def _select_rows(manifest: dict[str, Any], count: int) -> list[dict[str, Any]]:
    first: dict[str, dict[str, Any]] = {}
    for row in manifest.get("samples", []):
        object_uid = str(row.get("object_uid", row["uid"]))
        current = first.get(object_uid)
        score = (int(row.get("view_count", 0)), int(row.get("prior_point_count", 0)))
        current_score = (
            (-1, -1)
            if current is None
            else (
                int(current.get("view_count", 0)),
                int(current.get("prior_point_count", 0)),
            )
        )
        if current is None or score > current_score:
            first[object_uid] = row
    eligible = [
        row
        for row in first.values()
        if int(row.get("prior_point_count", 0))
        >= PRODUCTION_FRAME_CONFIG.min_object_points
        and int(row.get("view_count", 0)) >= PRODUCTION_FRAME_VIEW_COUNT
    ]
    eligible.sort(
        key=lambda row: (
            -int(row.get("view_count", 0)),
            -int(row.get("prior_point_count", 0)),
            str(row["uid"]),
        )
    )
    if len(eligible) < int(count):
        raise RuntimeError(
            f"only {len(eligible)} reviewed objects meet the production point gate"
        )
    # Spread the audit through the ordered eligible pool instead of taking a
    # contiguous UUID prefix or only the densest examples.
    positions = np.linspace(0, len(eligible) - 1, int(count), dtype=np.int64)
    primary_indices = [int(index) for index in positions]
    primary = [eligible[index] for index in primary_indices]
    primary_uids = {str(row["uid"]) for row in primary}
    # Remaining candidates are deterministic replacements when a primary row
    # loses too many points at the multi-view mask-support gate.
    return primary + [row for row in eligible if str(row["uid"]) not in primary_uids]


def _validate_bindings(
    *,
    lifting_path: Path,
    lifting: dict[str, Any],
    slat_path: Path,
    slat: dict[str, Any],
    ss_path: Path,
    ss_checkpoint: dict[str, Any],
    slat_checkpoint_path: Path,
    slat_checkpoint: dict[str, Any],
) -> dict[str, Any]:
    lifting_hash = sha256_file(lifting_path)
    slat_hash = sha256_file(slat_path)
    ss_hash = sha256_file(ss_path)
    slat_checkpoint_hash = sha256_file(slat_checkpoint_path)
    ss_identity = dict(ss_checkpoint.get("data_identity", {}))
    slat_identity = dict(slat_checkpoint.get("data_identity", {}))
    upstream = dict(slat_checkpoint.get("model_summary", {}).get("upstream_native_ss", {}))
    checks = {
        "ss_lifting_manifest": ss_identity.get("manifest_sha256") == lifting_hash,
        "slat_lifting_manifest": slat_identity.get("lifting_cache_manifest_sha256")
        == lifting_hash,
        "slat_source_lifting_manifest": slat.get("source_lifting_manifest_sha256")
        == lifting_hash,
        "slat_training_manifest": slat_identity.get("cache_manifest_sha256")
        == slat_hash,
        "slat_upstream_ss_checkpoint": upstream.get("checkpoint_sha256") == ss_hash,
        "ss_sample_count": int(ss_identity.get("sample_count", -1))
        == int(lifting.get("sample_count", -2)),
        "slat_sample_count": int(slat_identity.get("sample_count", -1))
        == int(slat.get("sample_count", -2)),
    }
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "lifting_manifest": str(lifting_path),
        "lifting_manifest_sha256": lifting_hash,
        "slat_manifest": str(slat_path),
        "slat_manifest_sha256": slat_hash,
        "native_ss_checkpoint": str(ss_path),
        "native_ss_checkpoint_sha256": ss_hash,
        "native_ss_step": int(ss_checkpoint.get("step", -1)),
        "native_slat_checkpoint": str(slat_checkpoint_path),
        "native_slat_checkpoint_sha256": slat_checkpoint_hash,
        "native_slat_step": int(slat_checkpoint.get("step", -1)),
    }


def _world_projection(
    points: np.ndarray, intrinsics: np.ndarray, T_W2C: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    homogeneous = np.concatenate((points, np.ones((len(points), 1))), axis=1)
    camera = np.einsum("vij,nj->vni", T_W2C, homogeneous)[..., :3]
    depth = camera[..., 2]
    pixels_h = np.einsum("vij,vnj->vni", intrinsics, camera)
    safe = np.where(np.abs(depth) > 1.0e-12, depth, 1.0)
    return pixels_h[..., :2] / safe[..., None], depth


def _audit_one(
    *,
    row: dict[str, Any],
    lifting_manifest: dict[str, Any],
    slat_manifest: dict[str, Any],
    slat_by_uid: dict[str, dict[str, Any]],
    ss_checkpoint: dict[str, Any],
    slat_checkpoint: dict[str, Any],
    seed: int,
    thresholds: dict[str, float],
) -> dict[str, Any]:
    cache_path = _resolve_cache_file(lifting_manifest, row)
    sample = torch.load(cache_path, map_location="cpu")
    uid = str(row["uid"])
    if str(sample.get("uid")) != uid:
        raise RuntimeError(f"lifting cache uid changed: {cache_path}")
    slat_row = slat_by_uid.get(uid)
    if slat_row is None:
        raise RuntimeError(f"Native-SLAT target is absent for uid={uid}")

    prior_coords = sample["prior_coords"].detach().cpu().numpy()
    points_w = native_coords_to_physical(
        prior_coords, resolution=64, grid_transform=str(sample["grid_transform"])
    )
    intrinsics = np.asarray(sample["intrinsics"], dtype=np.float64)
    poses = _to_w2c(sample)
    masks = np.asarray(sample["masks"].detach().cpu(), dtype=np.float64)
    reference = _reference_view(masks)
    config = PRODUCTION_FRAME_CONFIG
    baseline = canonicalize_runtime_object_frame(
        points_w,
        intrinsics,
        poses,
        masks,
        config=config,
        reference_view_index=reference,
    )
    gauge = deterministic_proper_similarity(seed)
    transformed_points = apply_transform(points_w, gauge)
    transformed_poses = transform_camera_gauge(poses, gauge)
    transformed = canonicalize_runtime_object_frame(
        transformed_points,
        intrinsics,
        transformed_poses,
        masks,
        config=config,
        reference_view_index=reference,
    )

    expected_frame = gauge @ baseline.T_O2W
    frame_error = float(np.max(np.abs(transformed.T_O2W - expected_frame)))
    point_error = float(np.max(np.abs(transformed.P_O - baseline.P_O)))
    sentinels = baseline.P_O[
        np.linspace(0, len(baseline.P_O) - 1, min(32, len(baseline.P_O))).astype(int)
    ]
    object_uv, _ = project_object_points(sentinels, intrinsics, baseline.T_O2C)
    world_points = apply_transform(sentinels, baseline.T_O2W)
    world_uv, _ = _world_projection(world_points, intrinsics, poses)
    projection_error = float(np.max(np.abs(object_uv - world_uv)))

    baseline_sample = _make_projection_sample(sample, baseline)
    transformed_sample = _make_projection_sample(sample, transformed)
    ss_base, ss_valid, _ = project_frustum_dino(
        baseline_sample, device=torch.device("cpu")
    )
    ss_gauge, ss_gauge_valid, _ = project_frustum_dino(
        transformed_sample, device=torch.device("cpu")
    )
    ss_wrong, ss_wrong_valid, _ = project_frustum_dino(
        baseline_sample,
        device=torch.device("cpu"),
        projection_mode="pose_cyclic1",
    )
    ss_feature_error = _tensor_max_abs(ss_base, ss_gauge)
    ss_valid_equal = bool(torch.equal(ss_valid, ss_gauge_valid))
    ss_pose_separability = _projection_difference(
        ss_base, ss_valid, ss_wrong, ss_wrong_valid
    )
    ss_parent = replay_parent_condition_path(
        ss_checkpoint, ss_base, ss_valid, ss_gauge, ss_gauge_valid
    )

    ss_target_path = Path(str(row["ss_latent"])).resolve()
    with np.load(ss_target_path, allow_pickle=False) as payload:
        target_coords = np.asarray(payload["target_coords"], dtype=np.int32)
    active32 = np.unique((target_coords // 2).astype(np.int32), axis=0)
    active32 = np.concatenate(
        (np.zeros((len(active32), 1), dtype=np.int32), active32), axis=1
    )
    active32_t = torch.from_numpy(active32)
    slat_base, slat_valid, _ = project_sparse_frustum_dino(
        baseline_sample, active32_t, device=torch.device("cpu")
    )
    slat_gauge, slat_gauge_valid, _ = project_sparse_frustum_dino(
        transformed_sample, active32_t, device=torch.device("cpu")
    )
    slat_wrong, slat_wrong_valid, _ = project_sparse_frustum_dino(
        baseline_sample,
        active32_t,
        device=torch.device("cpu"),
        projection_mode="pose_cyclic1",
    )
    slat_feature_error = _tensor_max_abs(slat_base, slat_gauge)
    slat_valid_equal = bool(torch.equal(slat_valid, slat_gauge_valid))
    slat_pose_separability = _projection_difference(
        slat_base, slat_valid, slat_wrong, slat_wrong_valid
    )
    slat_parent = replay_parent_condition_path(
        slat_checkpoint,
        slat_base,
        slat_valid,
        slat_gauge,
        slat_gauge_valid,
    )

    old_grid_to_w = np.eye(4, dtype=np.float64)
    old_grid_to_w[:3, :3] = grid_rotation(str(sample["grid_transform"]))
    old_grid_to_runtime = baseline.T_W2O @ old_grid_to_w
    scale = similarity_scale(old_grid_to_runtime)
    rotation = old_grid_to_runtime[:3, :3] / scale
    rotation_error = rotation_angle_degrees(rotation)
    translation_error = float(np.linalg.norm(old_grid_to_runtime[:3, 3]))
    scale_error = abs(scale - 1.0)
    target_physical = native_coords_to_physical(
        target_coords, resolution=64, grid_transform=str(sample["grid_transform"])
    )
    target_runtime = apply_transform(target_physical, baseline.T_W2O)
    runtime_target_coords = physical_to_native_coords(target_runtime, resolution=64)
    target_iou = coordinate_iou(target_coords, runtime_target_coords)

    slat_target_path = _resolve_manifest_artifact(
        slat_manifest, slat_row, "target_file"
    )
    slat_source_ss_path = Path(str(slat_row["ss_latent"])).resolve()
    ss_target_hash = sha256_file(ss_target_path)
    slat_target_hash = sha256_file(slat_target_path)
    slat_source_ss_hash = sha256_file(slat_source_ss_path)
    with np.load(slat_target_path, allow_pickle=False) as payload:
        slat_local_ss_coords = np.asarray(payload["local_ss_coords"], dtype=np.int32)
    target_binding_checks = {
        "ss_target_exists": ss_target_path.is_file(),
        "slat_target_hash": slat_target_hash == str(slat_row["target_file_sha256"]),
        "slat_source_ss_hash": slat_source_ss_hash
        == str(slat_row["ss_latent_sha256"]),
        "object_target_coords_equal": np.array_equal(
            slat_local_ss_coords, target_coords
        ),
    }
    gauge_checks = {
        "frame_equivariance": frame_error
        <= thresholds["frame_equivariance_max_abs"],
        "object_point_equivariance": point_error
        <= thresholds["object_point_equivariance_max_abs"],
        "projection_roundtrip": projection_error
        <= thresholds["projection_roundtrip_max_px"],
        "ss_projected_feature": ss_valid_equal
        and ss_feature_error <= thresholds["projected_feature_max_abs"],
        "slat_projected_feature": slat_valid_equal
        and slat_feature_error <= thresholds["projected_feature_max_abs"],
        "ss_parent_condition": ss_parent["max_abs"]
        <= thresholds["parent_condition_max_abs"],
        "slat_parent_condition": slat_parent["max_abs"]
        <= thresholds["parent_condition_max_abs"],
        "ss_pose_separable": ss_pose_separability
        >= thresholds["pose_separability_min"],
        "slat_pose_separable": slat_pose_separability
        >= thresholds["pose_separability_min"],
        "target_binding": all(target_binding_checks.values()),
    }
    reuse_checks = {
        "canonical_rotation": rotation_error
        <= thresholds["canonical_rotation_max_deg"],
        "canonical_translation": translation_error
        <= thresholds["canonical_translation_max"],
        "canonical_scale": scale_error
        <= thresholds["canonical_scale_relative_error_max"],
        "target_occupancy": target_iou >= thresholds["target_occupancy_iou_min"],
    }
    result = {
        "uid": uid,
        "object_uid": str(row.get("object_uid", uid)),
        "cache_file": str(cache_path),
        "cache_file_sha256": sha256_file(cache_path),
        "view_count": int(sample["visual_patch_features"].shape[0]),
        "point_count": int(len(points_w)),
        "reference_view_index": reference,
        "gauge_seed": int(seed),
        "gauge_scale": similarity_scale(gauge),
        "metrics": {
            "frame_equivariance_max_abs": frame_error,
            "object_point_equivariance_max_abs": point_error,
            "projection_roundtrip_max_px": projection_error,
            "ss_projected_feature_max_abs": ss_feature_error,
            "slat_projected_feature_max_abs": slat_feature_error,
            "ss_pose_separability": ss_pose_separability,
            "slat_pose_separability": slat_pose_separability,
            "ss_parent_condition": ss_parent,
            "slat_parent_condition": slat_parent,
            "old_grid_to_runtime_rotation_deg": rotation_error,
            "old_grid_to_runtime_translation": translation_error,
            "old_grid_to_runtime_scale": scale,
            "old_grid_to_runtime_scale_relative_error": scale_error,
            "target_occupancy_iou": target_iou,
            "runtime_target_count": int(len(runtime_target_coords)),
            "stored_target_count": int(len(target_coords)),
        },
        "target_binding": {
            "ss_target": str(ss_target_path),
            "ss_target_sha256": ss_target_hash,
            "slat_target": str(slat_target_path),
            "slat_target_sha256": slat_target_hash,
            "slat_source_ss_target": str(slat_source_ss_path),
            "slat_source_ss_target_sha256": slat_source_ss_hash,
            "checks": target_binding_checks,
        },
        "gauge_checks": gauge_checks,
        "gauge_compatibility_passed": all(gauge_checks.values()),
        "reuse_checks": reuse_checks,
        "reviewed_cache_reusable": all(reuse_checks.values()),
    }
    del sample, ss_base, ss_gauge, ss_wrong, slat_base, slat_gauge, slat_wrong
    gc.collect()
    return result


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lifting_manifest", required=True)
    parser.add_argument("--slat_manifest", required=True)
    parser.add_argument("--native_ss_checkpoint", required=True)
    parser.add_argument("--native_slat_checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--objects", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--pretrained", default="Stable-X/trellis-vggt-v0-2")
    parser.add_argument("--run_parent_forward", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp_dtype", choices=("bf16", "fp16", "none"), default="bf16")
    return parser


def main() -> None:
    args = make_parser().parse_args()
    if int(args.objects) <= 0:
        raise ValueError("--objects must be positive")
    lifting_path = Path(args.lifting_manifest).expanduser().resolve()
    slat_path = Path(args.slat_manifest).expanduser().resolve()
    ss_path = Path(args.native_ss_checkpoint).expanduser().resolve()
    slat_checkpoint_path = Path(args.native_slat_checkpoint).expanduser().resolve()
    lifting = load_json(lifting_path)
    slat = load_json(slat_path)
    ss_checkpoint = torch.load(ss_path, map_location="cpu")
    slat_checkpoint = torch.load(slat_checkpoint_path, map_location="cpu")
    bindings = _validate_bindings(
        lifting_path=lifting_path,
        lifting=lifting,
        slat_path=slat_path,
        slat=slat,
        ss_path=ss_path,
        ss_checkpoint=ss_checkpoint,
        slat_checkpoint_path=slat_checkpoint_path,
        slat_checkpoint=slat_checkpoint,
    )
    if not bindings["passed"]:
        raise RuntimeError(f"parent-v2 data/checkpoint binding failed: {bindings['checks']}")
    if (
        ss_checkpoint.get("model_summary", {}).get("pretrained") != args.pretrained
        or slat_checkpoint.get("model_summary", {}).get("pretrained") != args.pretrained
    ):
        raise RuntimeError("parent-v2 pretrained identity differs")
    rows = _select_rows(lifting, int(args.objects))
    slat_by_uid = {str(row["uid"]): row for row in slat.get("samples", [])}
    records = []
    failures = []
    admission_failures = []
    for index, row in enumerate(rows, start=1):
        if len(records) >= int(args.objects):
            break
        uid = str(row["uid"])
        print(
            f"[F0C] completed={len(records)}/{int(args.objects)} "
            f"candidate={index}/{len(rows)} uid={uid}",
            flush=True,
        )
        try:
            records.append(
                _audit_one(
                    row=row,
                    lifting_manifest=lifting,
                    slat_manifest=slat,
                    slat_by_uid=slat_by_uid,
                    ss_checkpoint=ss_checkpoint,
                    slat_checkpoint=slat_checkpoint,
                    seed=int(args.seed) + index * 1009,
                    thresholds=DEFAULT_THRESHOLDS,
                )
            )
        except Exception as error:
            target = (
                admission_failures
                if "insufficient mask-supported object points" in str(error)
                else failures
            )
            target.append({"uid": uid, "error": repr(error)})
            print(f"[F0C] REJECTED uid={uid}: {error!r}", flush=True)
    gauge_passed = (
        len(records) == int(args.objects)
        and not failures
        and all(row["gauge_compatibility_passed"] for row in records)
    )
    reusable = gauge_passed and all(row["reviewed_cache_reusable"] for row in records)
    parent_forward: dict[str, Any] = {
        "executed": False,
        "passed": False,
        "reason": "rerun with --run_parent_forward for the formal F0C gate",
    }
    if args.run_parent_forward and records:
        selected_uid = str(records[0]["uid"])
        selected_row = next(row for row in rows if str(row["uid"]) == selected_uid)
        parent_forward = run_parent_forward_rollout(
            row=selected_row,
            lifting_manifest=lifting,
            slat_manifest=slat,
            slat_by_uid=slat_by_uid,
            ss_checkpoint=ss_checkpoint,
            slat_checkpoint=slat_checkpoint,
            pretrained=str(args.pretrained),
            device=torch.device(args.device),
            amp_name=str(args.amp_dtype),
            seed=int(args.seed) + 900001,
            threshold=DEFAULT_THRESHOLDS["parent_flow_max_abs"],
        )
    formal_passed = gauge_passed and bool(parent_forward.get("passed"))
    report = {
        "format": REPORT_FORMAT,
        "audit_scope": (
            "reviewed1k observable-input runtime-O proper-Sim(3), exact Native SS/SLat "
            "v2 projection, checkpoint learned condition path, target binding and pose control"
        ),
        "pretrained": str(args.pretrained),
        "seed": int(args.seed),
        "requested_object_count": int(args.objects),
        "completed_object_count": len(records),
        "thresholds": DEFAULT_THRESHOLDS,
        "bindings": bindings,
        "objects": records,
        "failures": failures,
        "reviewed_sample_admission_failures": admission_failures,
        "production_frame_config": {
            key: getattr(PRODUCTION_FRAME_CONFIG, key)
            for key in PRODUCTION_FRAME_CONFIG.__dataclass_fields__
        },
        "production_frame_view_count": PRODUCTION_FRAME_VIEW_COUNT,
        "view_policy": (
            "estimate runtime-O from 8 views; later training may subsample the "
            "already-canonicalized condition to 2/4/8 views"
        ),
        "gauge_compatibility_passed": gauge_passed,
        "reviewed_cache_reusable": reusable,
        "reviewed_cache_decision": (
            "reuse_unchanged"
            if reusable
            else "rebuild_reviewed1k_with_runtime_o_before_mixed_training"
        ),
        "parent_forward_rollout": parent_forward,
        "formal_passed": formal_passed,
        "passed": gauge_passed,
        "training_ready": False,
        "next_gate": (
            "F0E_rebuild_reviewed1k"
            if gauge_passed and not reusable
            else "F0E_bind_reuse_report"
            if reusable
            else "fix_F0C_failures"
        ),
    }
    atomic_json(args.output, report)
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "objects": len(records),
                "failures": len(failures),
                "admission_failures": len(admission_failures),
                "reviewed_cache_reusable": reusable,
                "decision": report["reviewed_cache_decision"],
                "formal_passed": formal_passed,
                "parent_forward": parent_forward,
                "training_ready": False,
                "output": str(Path(args.output).expanduser().resolve()),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
