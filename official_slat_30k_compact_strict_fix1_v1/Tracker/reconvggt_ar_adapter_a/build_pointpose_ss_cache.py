#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt

TRACKER_ROOT = Path(__file__).resolve().parents[1]
if str(TRACKER_ROOT) not in sys.path:
    sys.path.insert(0, str(TRACKER_ROOT))

from reconvggt_ar_adapter_a.pointpose_ss_condition import (  # noqa: E402
    PHYSICAL_FEATURE_NAMES,
    feature_schema_hash,
)
from trellis_point_prior_mv.common import (  # noqa: E402
    apply_grid_transform,
    load_manifest,
    parse_indices,
    resolve_path,
)


def load_mask(path: str | Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0


def require_unique_uids(samples: list[dict[str, Any]], *, label: str) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for index, sample in enumerate(samples):
        uid = str(sample.get("uid", ""))
        if not uid:
            raise ValueError(f"{label} sample index={index} has no uid")
        if uid in output:
            raise ValueError(f"duplicate uid in {label}: {uid}")
        output[uid] = sample
    return output


def validate_prior_arrays(
    *,
    uid: str,
    prior_coords: np.ndarray,
    prior_conf: np.ndarray,
    view_ids: np.ndarray,
    frame_count: int,
) -> None:
    if prior_coords.ndim != 2 or prior_coords.shape[1] not in (3, 4):
        raise ValueError(f"uid={uid} prior_coords must be [N,3/4], got {prior_coords.shape}")
    if not np.isfinite(prior_coords).all():
        raise ValueError(f"uid={uid} prior_coords contains non-finite values")
    if prior_coords.size and not np.equal(prior_coords, np.rint(prior_coords)).all():
        raise ValueError(f"uid={uid} prior_coords contains non-integer values")
    if prior_conf.ndim != 1 or len(prior_conf) != len(prior_coords):
        raise ValueError(
            f"uid={uid} prior_conf mismatch: coords={prior_coords.shape}, conf={prior_conf.shape}"
        )
    if not np.isfinite(prior_conf).all():
        raise ValueError(f"uid={uid} prior_conf contains non-finite values")
    xyz = prior_coords[:, -3:]
    if xyz.size and ((xyz < 0).any() or (xyz > 63).any()):
        bad = xyz[((xyz < 0) | (xyz > 63)).any(axis=1)][:8]
        raise ValueError(f"uid={uid} prior coords outside [0,63]: {bad.tolist()}")
    if not np.isfinite(view_ids).all():
        raise ValueError(f"uid={uid} view_ids contains non-finite values")
    if view_ids.size and not np.equal(view_ids, np.rint(view_ids)).all():
        raise ValueError(f"uid={uid} view_ids contains non-integer values")
    if view_ids.ndim != 1 or view_ids.size == 0:
        raise ValueError(f"uid={uid} view_ids must be non-empty 1D, got {view_ids.shape}")
    if len(np.unique(view_ids)) != len(view_ids):
        raise ValueError(f"uid={uid} view_ids contains duplicates: {view_ids.tolist()}")
    if (view_ids < 0).any() or (view_ids >= frame_count).any():
        raise ValueError(f"uid={uid} view_ids out of range [0,{frame_count}): {view_ids.tolist()}")


def load_and_validate_target(latent_path: Path, *, uid: str) -> tuple[np.ndarray, np.ndarray]:
    if not latent_path.is_file():
        raise FileNotFoundError(f"uid={uid} missing latent file: {latent_path}")
    with np.load(latent_path) as data:
        required = {"z", "target_coords"}
        missing = sorted(required - set(data.files))
        if missing:
            raise ValueError(f"uid={uid} latent is missing keys {missing}: {latent_path}")
        z = np.asarray(data["z"])
        target_coords = np.asarray(data["target_coords"])
    if z.ndim == 5 and z.shape[0] == 1:
        z = z[0]
    if z.shape != (8, 16, 16, 16):
        raise ValueError(f"uid={uid} target z must be [8,16,16,16], got {z.shape}")
    if not np.isfinite(z).all():
        raise ValueError(f"uid={uid} target z contains non-finite values")
    if target_coords.ndim != 2 or target_coords.shape[1] not in (3, 4):
        raise ValueError(f"uid={uid} target_coords must be [N,3/4], got {target_coords.shape}")
    if not np.isfinite(target_coords).all():
        raise ValueError(f"uid={uid} target_coords contains non-finite values")
    xyz = target_coords[:, -3:]
    if xyz.size and ((xyz < 0).any() or (xyz > 63).any()):
        bad = xyz[((xyz < 0) | (xyz > 63)).any(axis=1)][:8]
        raise ValueError(f"uid={uid} target coords outside [0,63]: {bad.tolist()}")
    return z.astype(np.float32), xyz.astype(np.int32)


def array_stats(array: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(array, dtype=np.float64)
    return {
        "min": float(values.min()) if values.size else 0.0,
        "max": float(values.max()) if values.size else 0.0,
        "mean": float(values.mean()) if values.size else 0.0,
        "nonzero_ratio": float(np.count_nonzero(values) / values.size) if values.size else 0.0,
        "numel": int(values.size),
    }


def physical_grid_audit(grid: np.ndarray) -> dict[str, Any]:
    channels = {
        name: array_stats(grid[index]) for index, name in enumerate(PHYSICAL_FEATURE_NAMES)
    }
    prior_occ = grid[PHYSICAL_FEATURE_NAMES.index("prior_occupancy")] > 0.5
    prior_conf = grid[PHYSICAL_FEATURE_NAMES.index("prior_confidence")]
    mask_support = grid[PHYSICAL_FEATURE_NAMES.index("mask_support_fraction")]
    high_conf = prior_occ & (prior_conf >= 0.5)
    return {
        "channels": channels,
        "prior_occupied_cells": int(prior_occ.sum()),
        "visible_cell_ratio": float(
            (grid[PHYSICAL_FEATURE_NAMES.index("visible_fraction")] > 0).mean()
        ),
        "mask_supported_cell_ratio": float(
            (grid[PHYSICAL_FEATURE_NAMES.index("mask_support_fraction")] > 0).mean()
        ),
        "outside_cell_ratio": float(
            (grid[PHYSICAL_FEATURE_NAMES.index("outside_visible_ratio")] > 0.5).mean()
        ),
        "high_conf_prior_cell_count": int(high_conf.sum()),
        "high_conf_prior_mask_support_mean": float(mask_support[high_conf].mean()) if high_conf.any() else None,
        "high_conf_prior_mask_support_nonzero_ratio": float((mask_support[high_conf] > 0).mean()) if high_conf.any() else None,
    }


def grid_centers(side: int = 16) -> tuple[np.ndarray, np.ndarray]:
    axis = (np.arange(side, dtype=np.float32) + 0.5) / float(side) - 0.5
    xx, yy, zz = np.meshgrid(axis, axis, axis, indexing="ij")
    points = np.stack((xx, yy, zz), axis=-1).reshape(-1, 3)
    xyz_features = np.stack((xx * 2.0, yy * 2.0, zz * 2.0), axis=0)
    return points.astype(np.float32), xyz_features.astype(np.float32)


def projection_features(
    points: np.ndarray,
    masks: list[np.ndarray],
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    *,
    extrinsics_type: str,
    camera_forward_sign: float,
) -> dict[str, np.ndarray]:
    n = int(points.shape[0])
    num_views = max(1, len(masks))
    support = np.zeros(n, dtype=np.float32)
    visible = np.zeros(n, dtype=np.float32)
    depth_sum = np.zeros(n, dtype=np.float32)
    depth_sq_sum = np.zeros(n, dtype=np.float32)
    pts_h = np.concatenate((points.astype(np.float32), np.ones((n, 1), dtype=np.float32)), axis=1)
    for mask, intrinsic, extrinsic in zip(masks, intrinsics, extrinsics):
        w2c = np.linalg.inv(extrinsic) if extrinsics_type == "c2w" else extrinsic
        cam = (w2c @ pts_h.T).T[:, :3]
        depth = cam[:, 2] * float(camera_forward_sign)
        valid_depth = depth > 1.0e-5
        safe_depth = np.maximum(depth, 1.0e-5)
        u = intrinsic[0, 0] * (cam[:, 0] / safe_depth) + intrinsic[0, 2]
        v = intrinsic[1, 1] * (cam[:, 1] / safe_depth) + intrinsic[1, 2]
        height, width = mask.shape
        ui = np.rint(u).astype(np.int32)
        vi = np.rint(v).astype(np.int32)
        valid = valid_depth & (ui >= 0) & (ui < width) & (vi >= 0) & (vi < height)
        ids = np.nonzero(valid)[0]
        if not len(ids):
            continue
        visible[ids] += 1.0
        support[ids] += mask[vi[ids], ui[ids]]
        median_depth = float(np.median(depth[ids]))
        normalized_depth = np.clip(depth[ids] / max(median_depth, 1.0e-5), 0.0, 2.0) * 0.5
        depth_sum[ids] += normalized_depth.astype(np.float32)
        depth_sq_sum[ids] += np.square(normalized_depth).astype(np.float32)
    visible_safe = np.maximum(visible, 1.0)
    ratio = support / visible_safe
    depth_mean = depth_sum / visible_safe
    depth_var = np.maximum(depth_sq_sum / visible_safe - np.square(depth_mean), 0.0)
    return {
        "support_fraction": support / float(num_views),
        "visible_fraction": visible / float(num_views),
        "mask_hit_ratio": ratio,
        "outside_visible_ratio": (visible - support) / visible_safe,
        "visual_hull_inside": ((visible > 0) & (ratio >= 0.5)).astype(np.float32),
        "depth_mean": depth_mean,
        "depth_std": np.sqrt(depth_var),
    }


def prior_features(prior_coords: np.ndarray, prior_conf: np.ndarray, side: int = 16) -> dict[str, np.ndarray]:
    occupancy = np.zeros((side, side, side), dtype=np.float32)
    confidence = np.zeros_like(occupancy)
    count = np.zeros_like(occupancy)
    if prior_coords.size:
        xyz = prior_coords[:, -3:].astype(np.int32)
        if (xyz < 0).any() or (xyz >= side * 4).any():
            raise ValueError("prior_features received coordinates outside the 64^3 grid")
        cell = xyz // 4
        np.maximum.at(confidence, (cell[:, 0], cell[:, 1], cell[:, 2]), prior_conf.astype(np.float32))
        np.add.at(count, (cell[:, 0], cell[:, 1], cell[:, 2]), 1.0)
        occupancy[count > 0] = 1.0
    if bool(occupancy.any()):
        distance = distance_transform_edt(occupancy == 0).astype(np.float32)
        distance /= max(float(math.sqrt(3.0) * (side - 1)), 1.0)
    else:
        distance = np.ones_like(occupancy)
    max_count = max(float(count.max()), 1.0)
    log_count = np.log1p(count) / math.log1p(max_count)
    return {
        "occupancy": occupancy,
        "confidence": confidence,
        "log_count": log_count.astype(np.float32),
        "distance": np.clip(distance, 0.0, 1.0),
    }


def make_physical_grid(
    *,
    prior_coords: np.ndarray,
    prior_conf: np.ndarray,
    masks: list[np.ndarray],
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    grid_transform: str,
    extrinsics_type: str,
    camera_forward_sign: float,
) -> np.ndarray:
    points, xyz = grid_centers(16)
    projection_points = apply_grid_transform(points, grid_transform)
    projection = projection_features(
        projection_points,
        masks,
        intrinsics,
        extrinsics,
        extrinsics_type=extrinsics_type,
        camera_forward_sign=camera_forward_sign,
    )
    prior = prior_features(prior_coords, prior_conf, 16)
    channels = [
        prior["occupancy"],
        prior["confidence"],
        prior["log_count"],
        prior["distance"],
        projection["support_fraction"].reshape(16, 16, 16),
        projection["visible_fraction"].reshape(16, 16, 16),
        projection["mask_hit_ratio"].reshape(16, 16, 16),
        projection["outside_visible_ratio"].reshape(16, 16, 16),
        projection["visual_hull_inside"].reshape(16, 16, 16),
        projection["depth_mean"].reshape(16, 16, 16),
        projection["depth_std"].reshape(16, 16, 16),
        xyz[0],
        xyz[1],
        xyz[2],
    ]
    grid = np.stack(channels, axis=0).astype(np.float32)
    if grid.shape != (len(PHYSICAL_FEATURE_NAMES), 16, 16, 16):
        raise RuntimeError(f"unexpected physical grid shape: {grid.shape}")
    if not np.isfinite(grid).all():
        raise RuntimeError("physical grid contains non-finite values")
    return grid


def frame_paths(
    payload: dict[str, Any],
    sample: dict[str, Any],
    view_ids: np.ndarray,
    *,
    uid: str,
) -> tuple[list[str], list[str], np.ndarray, np.ndarray, list[np.ndarray]]:
    image_root = sample.get("image_root", payload.get("image_root"))
    mask_root = sample.get("mask_root", payload.get("mask_root"))
    frames = sample.get("frames") or []
    if not frames:
        raise ValueError(f"uid={uid} source sample has no frames")
    selected = [frames[int(index)] for index in view_ids]
    image_paths = [str(resolve_path(image_root, frame["image"])) for frame in selected]
    mask_paths = [str(resolve_path(mask_root, frame["mask"])) for frame in selected]
    intrinsics = np.stack([np.asarray(frame["intrinsic"], dtype=np.float32) for frame in selected])
    extrinsics = np.stack([np.asarray(frame["extrinsic"], dtype=np.float32) for frame in selected])
    if not (len(image_paths) == len(mask_paths) == len(intrinsics) == len(extrinsics) == len(view_ids)):
        raise ValueError(f"uid={uid} image/mask/K/T/view count mismatch")
    if intrinsics.shape[1:] != (3, 3) or extrinsics.shape[1:] != (4, 4):
        raise ValueError(f"uid={uid} invalid K/T shapes: K={intrinsics.shape} T={extrinsics.shape}")
    if not np.isfinite(intrinsics).all() or not np.isfinite(extrinsics).all():
        raise ValueError(f"uid={uid} K/T contains non-finite values")
    masks: list[np.ndarray] = []
    for index, (image_path, mask_path, intrinsic) in enumerate(zip(image_paths, mask_paths, intrinsics)):
        image_file = Path(image_path)
        mask_file = Path(mask_path)
        if not image_file.is_file() or not mask_file.is_file():
            raise FileNotFoundError(
                f"uid={uid} view={index} missing image/mask: image={image_file} mask={mask_file}"
            )
        with Image.open(image_file) as image:
            image_size = image.size
        with Image.open(mask_file) as mask_image:
            mask_size = mask_image.size
        if image_size != mask_size:
            raise ValueError(
                f"uid={uid} view={index} image/mask resolution mismatch: {image_size} vs {mask_size}"
            )
        width, height = image_size
        expected_size = payload.get("image_size")
        if expected_size is not None:
            expected = (int(expected_size), int(expected_size))
            if image_size != expected:
                raise ValueError(
                    f"uid={uid} view={index} image/mask resolution {image_size} "
                    f"does not match calibrated manifest image_size={expected}"
                )
        fx, fy = float(intrinsic[0, 0]), float(intrinsic[1, 1])
        cx, cy = float(intrinsic[0, 2]), float(intrinsic[1, 2])
        if fx <= 0 or fy <= 0:
            raise ValueError(f"uid={uid} view={index} invalid focal length: fx={fx} fy={fy}")
        if not (0.0 <= cx < width and 0.0 <= cy < height):
            raise ValueError(
                f"uid={uid} view={index} principal point outside image: c=({cx},{cy}) size={image_size}"
            )
        masks.append(load_mask(mask_file))
    return image_paths, mask_paths, intrinsics, extrinsics, masks


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute compact 16^3 point/pose evidence for SS training.")
    parser.add_argument("--source_manifest", required=True)
    parser.add_argument("--prior_manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--indices", default="all")
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--log_every", type=int, default=50)
    args = parser.parse_args()

    source_payload, source_samples = load_manifest(args.source_manifest)
    prior_payload, prior_samples = load_manifest(args.prior_manifest)
    source_by_uid = require_unique_uids(source_samples, label="source manifest")
    require_unique_uids(prior_samples, label="prior manifest")
    prior_root = prior_payload.get("prior_root", prior_payload.get("output_dir"))
    latent_root = source_payload.get("latent_root")
    indices = parse_indices(args.indices, len(prior_samples))
    if int(args.max_samples) > 0:
        indices = indices[: int(args.max_samples)]
    output_dir = Path(args.output_dir)
    cache_dir = output_dir / "physical"
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_samples: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    aggregate_channels = {
        name: {"min": float("inf"), "max": float("-inf"), "sum": 0.0, "nonzero": 0, "numel": 0}
        for name in PHYSICAL_FEATURE_NAMES
    }
    view_count_histogram: Counter[int] = Counter()

    for out_index, prior_index in enumerate(indices):
        prior_sample = prior_samples[prior_index]
        uid = str(prior_sample["uid"])
        source = source_by_uid.get(uid)
        if source is None:
            raise KeyError(f"prior uid is absent from source manifest: {uid}")
        frames = source.get("frames") or []
        prior_path = resolve_path(prior_root, prior_sample["prior_npz"])
        if not prior_path.is_file():
            raise FileNotFoundError(f"uid={uid} missing prior npz: {prior_path}")
        with np.load(prior_path) as prior_data:
            required = {"prior_coords", "prior_conf", "view_ids"}
            missing = sorted(required - set(prior_data.files))
            if missing:
                raise ValueError(f"uid={uid} prior npz is missing keys {missing}: {prior_path}")
            prior_coords = np.asarray(prior_data["prior_coords"])
            prior_conf = np.asarray(prior_data["prior_conf"], dtype=np.float32)
            view_ids = np.asarray(prior_data["view_ids"])
        validate_prior_arrays(
            uid=uid,
            prior_coords=prior_coords,
            prior_conf=prior_conf,
            view_ids=view_ids,
            frame_count=len(frames),
        )
        prior_coords = prior_coords.astype(np.int32, copy=False)
        view_ids = view_ids.astype(np.int32, copy=False)
        image_paths, mask_paths, intrinsics, extrinsics, masks = frame_paths(
            source_payload, source, view_ids, uid=uid
        )
        grid_transform = str(prior_sample.get("grid_transform", prior_payload.get("grid_transform", "pixal3d_rotation")))
        physical_grid = make_physical_grid(
            prior_coords=prior_coords,
            prior_conf=prior_conf,
            masks=masks,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            grid_transform=grid_transform,
            extrinsics_type=str(source_payload.get("extrinsics_type", "c2w")),
            camera_forward_sign=float(source_payload.get("camera_forward_sign", 1.0)),
        )
        latent_path = resolve_path(latent_root, source["ss_latent"])
        target_z, target_coords = load_and_validate_target(latent_path, uid=uid)
        source_num_voxels = int(source.get("num_voxels", len(target_coords)))
        target_count_delta = source_num_voxels - int(len(target_coords))
        grid_audit = physical_grid_audit(physical_grid)
        for name, stats in grid_audit["channels"].items():
            aggregate = aggregate_channels[name]
            aggregate["min"] = min(float(aggregate["min"]), float(stats["min"]))
            aggregate["max"] = max(float(aggregate["max"]), float(stats["max"]))
            aggregate["sum"] += float(stats["mean"]) * int(stats["numel"])
            aggregate["nonzero"] += int(np.count_nonzero(physical_grid[PHYSICAL_FEATURE_NAMES.index(name)]))
            aggregate["numel"] += int(stats["numel"])
        view_count_histogram[int(len(view_ids))] += 1
        audit_rows.append(
            {
                "uid": uid,
                "object_uid": str(source["object_uid"]),
                "view_ids": view_ids.tolist(),
                "view_count": int(len(view_ids)),
                "prior_point_count": int(prior_coords.shape[0]),
                "latent_shape": list(target_z.shape),
                "target_coord_count": int(len(target_coords)),
                "source_num_voxels": source_num_voxels,
                "source_num_voxels_delta": target_count_delta,
                **grid_audit,
            }
        )
        shard = uid[:2] if len(uid) >= 2 else "00"
        cache_path = cache_dir / shard / f"{uid}.npz"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            physical_grid=physical_grid.astype(np.float16),
            prior_coords=prior_coords,
            prior_conf=prior_conf.astype(np.float16),
            view_ids=view_ids,
        )
        out_samples.append(
            {
                "uid": uid,
                "object_uid": str(source["object_uid"]),
                "source_index": int(prior_sample.get("source_index", -1)),
                "physical_grid": str(cache_path.relative_to(output_dir)),
                "ss_latent": str(latent_path),
                "image_paths": image_paths,
                "mask_paths": mask_paths,
                "view_ids": view_ids.tolist(),
                "prior_point_count": int(prior_coords.shape[0]),
                "target_coord_count": int(len(target_coords)),
            }
        )
        if (out_index + 1) % max(1, int(args.log_every)) == 0 or out_index + 1 == len(indices):
            print(
                f"[pointpose_cache] {out_index + 1}/{len(indices)} uid={uid} "
                f"views={len(view_ids)} prior={len(prior_coords)}",
                flush=True,
            )

    manifest = {
        "format": "reconvggt.pointpose_ss_cache.v1",
        "output_dir": str(output_dir),
        "source_manifest": str(Path(args.source_manifest)),
        "prior_manifest": str(Path(args.prior_manifest)),
        "feature_names": list(PHYSICAL_FEATURE_NAMES),
        "feature_schema_hash": feature_schema_hash(),
        "feature_dim": len(PHYSICAL_FEATURE_NAMES),
        "samples": out_samples,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    aggregate_report: dict[str, Any] = {}
    for name, aggregate in aggregate_channels.items():
        numel = int(aggregate["numel"])
        aggregate_report[name] = {
            "min": float(aggregate["min"]) if numel else 0.0,
            "max": float(aggregate["max"]) if numel else 0.0,
            "mean": float(aggregate["sum"] / numel) if numel else 0.0,
            "nonzero_ratio": float(aggregate["nonzero"] / numel) if numel else 0.0,
            "numel": numel,
        }
    cache_audit = {
        "format": "reconvggt.pointpose_ss_cache.audit.v1",
        "source_manifest": str(Path(args.source_manifest)),
        "prior_manifest": str(Path(args.prior_manifest)),
        "sample_count": len(out_samples),
        "unique_uid_count": len({row["uid"] for row in out_samples}),
        "unique_object_count": len({row["object_uid"] for row in out_samples}),
        "view_count_histogram": {str(key): value for key, value in sorted(view_count_histogram.items())},
        "source_num_voxels_mismatch_count": sum(
            int(row["source_num_voxels_delta"] != 0) for row in audit_rows
        ),
        "aggregate_channels": aggregate_report,
        "rows": audit_rows,
        "hard_failures": 0,
    }
    (output_dir / "cache_audit.json").write_text(
        json.dumps(cache_audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[pointpose_cache] wrote {output_dir / 'manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
