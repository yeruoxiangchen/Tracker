#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import sys
from typing import Any

os.environ.setdefault("SPCONV_ALGO", "native")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
os.environ.setdefault("TORCH_EXTENSIONS_DIR", "/tmp/torch_extensions")

import numpy as np
from PIL import Image
import torch

TRACKER_ROOT = Path(__file__).resolve().parents[1]
RECONVIAGEN_ROOT = TRACKER_ROOT / "ReconViaGen"
VGGT_ROOT = RECONVIAGEN_ROOT / "wheels" / "vggt"
for path in (TRACKER_ROOT, RECONVIAGEN_ROOT, VGGT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from trellis.pipelines import TrellisVGGTTo3DPipeline  # noqa: E402
from vggt.models.vggt import VGGT  # noqa: E402

from ar_ss_flow.pose_lifting import (  # noqa: E402
    LIFTING_CACHE_VERSION,
    LIFTING_METADATA_NAMES,
    build_projection_geometry,
    cache_config_hash,
    calibrate_vggt_depth,
    scale_intrinsics,
    schema_hash,
)
from reconvggt_ar_adapter_a.inspect_and_sanity import normalize_image_cond  # noqa: E402
from reconvggt_ar_adapter_a.train_pointpose_ss_lora import (  # noqa: E402
    PointPoseCacheDataset,
    install_unused_model_stubs,
    rgba_images,
)


def select_object_balanced_samples(
    samples: list[dict[str, Any]],
    *,
    max_objects: int,
    sequences_per_object: int,
    seed: int,
) -> list[dict[str, Any]]:
    if int(max_objects) <= 0 or int(sequences_per_object) <= 0:
        raise ValueError("object-balanced selection counts must be positive")
    by_object: dict[str, list[dict[str, Any]]] = {}
    for row in samples:
        object_uid = str(row.get("object_uid", row.get("uid", "")))
        if not object_uid:
            raise ValueError("source cache contains an empty object UID")
        by_object.setdefault(object_uid, []).append(row)
    if len(by_object) < int(max_objects):
        raise ValueError(
            f"requested {max_objects} objects but source only has {len(by_object)}"
        )
    rng = random.Random(int(seed))
    object_uids = sorted(by_object)
    rng.shuffle(object_uids)
    selected: list[dict[str, Any]] = []
    for object_uid in object_uids[: int(max_objects)]:
        candidates = sorted(
            by_object[object_uid], key=lambda row: str(row.get("uid", ""))
        )
        rng.shuffle(candidates)
        selected.extend(candidates[: int(sequences_per_object)])
    return selected


def full_frame_masked_images(
    image_paths: list[str],
    mask_paths: list[str],
    *,
    resolution: int,
) -> tuple[list[Image.Image], np.ndarray, list[tuple[int, int]]]:
    if len(image_paths) != len(mask_paths) or not image_paths:
        raise ValueError("image/mask paths must be non-empty and aligned")
    images: list[Image.Image] = []
    masks: list[np.ndarray] = []
    source_sizes: list[tuple[int, int]] = []
    for image_path, mask_path in zip(image_paths, mask_paths):
        with Image.open(image_path) as handle:
            rgb = handle.convert("RGB")
        with Image.open(mask_path) as handle:
            mask = handle.convert("L")
        if mask.size != rgb.size:
            raise ValueError(
                f"mask/image size mismatch: {mask_path}={mask.size}, {image_path}={rgb.size}"
            )
        source_sizes.append(rgb.size)
        rgb_array = np.asarray(rgb, dtype=np.uint8)
        mask_array = np.asarray(mask, dtype=np.uint8)
        masked = rgb_array * (mask_array[..., None] >= 127).astype(np.uint8)
        full_frame = Image.fromarray(masked, mode="RGB").resize(
            (resolution, resolution), Image.Resampling.LANCZOS
        )
        resized_mask = mask.resize(
            (resolution, resolution), Image.Resampling.BILINEAR
        )
        images.append(full_frame)
        masks.append(np.asarray(resized_mask, dtype=np.float32) / 255.0)
    return images, np.stack(masks), source_sizes


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@torch.no_grad()
def extract_stock_condition(pipeline, batch: dict[str, Any]) -> torch.Tensor:
    stock_images = rgba_images(batch["image_paths"], batch["mask_paths"], pipeline)
    aggregated, image_tensor = pipeline.vggt_feat(stock_images)
    raw_image_cond = pipeline.encode_image(image_tensor)
    batch_size = int(aggregated[0].shape[0])
    views = int(aggregated[0].shape[1])
    image_cond = normalize_image_cond(raw_image_cond, batch=batch_size, views=views)
    condition = pipeline.get_ss_cond(image_cond[:, :, 5:], aggregated, num_samples=1)["cond"]
    return condition.detach().cpu()


@torch.no_grad()
def extract_lifting_features(
    pipeline,
    images: list[Image.Image],
    *,
    vggt_feature_index: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    aggregated, image_tensor = pipeline.vggt_feat(images)
    raw_image_cond = pipeline.encode_image(image_tensor)
    batch_size = int(aggregated[0].shape[0])
    views = int(aggregated[0].shape[1])
    image_cond = normalize_image_cond(raw_image_cond, batch=batch_size, views=views)
    index = int(vggt_feature_index)
    if index < 0:
        index += len(aggregated)
    if index < 0 or index >= len(aggregated):
        raise IndexError(
            f"vggt_feature_index={vggt_feature_index} outside {len(aggregated)} layers"
        )
    patch_start_idx = 5
    vggt_patch = aggregated[index][:, :, patch_start_idx:].detach()
    dino_patch = image_cond[:, :, patch_start_idx:].detach()
    if vggt_patch.shape[:3] != dino_patch.shape[:3]:
        raise RuntimeError(
            f"VGGT/DINO patch mismatch: {tuple(vggt_patch.shape)}/{tuple(dino_patch.shape)}"
        )
    with torch.cuda.amp.autocast(enabled=False):
        depth, depth_confidence = pipeline.VGGT_model.depth_head(
            aggregated,
            images=image_tensor[None],
            patch_start_idx=patch_start_idx,
        )
    if depth.ndim == 5 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.ndim != 4 or depth_confidence.ndim != 4:
        raise RuntimeError(
            f"unexpected depth shapes {tuple(depth.shape)}/{tuple(depth_confidence.shape)}"
        )
    metadata = {
        "aggregated_layer_count": len(aggregated),
        "selected_vggt_feature_index": index,
        "patch_start_idx": patch_start_idx,
        "patch_count": int(vggt_patch.shape[2]),
        "vggt_feature_dim": int(vggt_patch.shape[3]),
        "dino_feature_dim": int(dino_patch.shape[3]),
        "depth_shape": list(depth.shape[-2:]),
    }
    visual = torch.cat((vggt_patch, dino_patch), dim=-1)[0]
    return (
        visual.cpu(),
        depth[0].detach().cpu(),
        depth_confidence[0].detach().cpu(),
        metadata,
    )


def build_native_stock_pipeline(pretrained: str, device: torch.device):
    """Load ReconViaGen without replacing its native VGGT model."""

    install_unused_model_stubs()
    pipeline = TrellisVGGTTo3DPipeline.from_pretrained(pretrained)
    pipeline._device = device
    pipeline.low_vram = False
    if not hasattr(pipeline, "VGGT_model"):
        raise RuntimeError("native ReconViaGen pipeline has no VGGT_model")
    pipeline.VGGT_model.to(device).eval()
    # ``low_vram`` is deliberately disabled so the native VGGT, DINO image
    # encoder, and stock condition adapter remain resident for cache building.
    # With it disabled, the pipeline does not move these modules on demand.
    # Move every module used by ``extract_stock_condition`` explicitly; leaving
    # the DINO encoder on CPU causes its Conv2d weights to disagree with the
    # CUDA image tensor created by ``encode_image``.
    for name in ("image_cond_model", "sparse_structure_vggt_cond"):
        if name not in pipeline.models:
            raise RuntimeError(f"native ReconViaGen pipeline has no {name}")
        pipeline.models[name].to(device).eval()
    for module in (
        pipeline.VGGT_model,
        pipeline.models["image_cond_model"],
        pipeline.models["sparse_structure_vggt_cond"],
    ):
        for parameter in module.parameters():
            parameter.requires_grad = False
    pipeline.stock_condition_source = "native_unmodified_reconviagen_vggt"
    return pipeline


def build_lifting_pipeline(
    pretrained: str,
    vggt_pretrained: str,
    device: torch.device,
):
    """Load the separate VGGT depth model used only for physical lifting."""

    install_unused_model_stubs()
    pipeline = TrellisVGGTTo3DPipeline.from_pretrained(pretrained)
    pipeline._device = device
    pipeline.low_vram = False
    keep_models = {"image_cond_model", "sparse_structure_vggt_cond"}
    for name in list(pipeline.models):
        if name not in keep_models:
            del pipeline.models[name]
    for name in ("slat_flow_model", "slat_vggt_cond", "sparse_structure_flow_model"):
        if hasattr(pipeline, name):
            delattr(pipeline, name)
    if not hasattr(pipeline.VGGT_model, "depth_head"):
        del pipeline.VGGT_model
        pipeline.VGGT_model = VGGT.from_pretrained(vggt_pretrained)
    for name in ("camera_head", "point_head", "track_head"):
        if hasattr(pipeline.VGGT_model, name):
            delattr(pipeline.VGGT_model, name)
    if not hasattr(pipeline.VGGT_model, "depth_head"):
        raise RuntimeError("VGGT depth_head is required for pose lifting cache")
    pipeline.VGGT_model.to(device).eval()
    pipeline.models["image_cond_model"].to(device).eval()
    pipeline.models["sparse_structure_vggt_cond"].to(device).eval()
    for module in (
        pipeline.VGGT_model,
        pipeline.models["image_cond_model"],
        pipeline.models["sparse_structure_vggt_cond"],
    ):
        for parameter in module.parameters():
            parameter.requires_grad = False
    return pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cache reusable full-frame visual patches and auditable pose-lifting inputs."
    )
    parser.add_argument("--source_cache_manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--pretrained", default="Stable-X/trellis-vggt-v0-2")
    parser.add_argument("--vggt_pretrained", default="Stable-X/vggt-object-v0-1")
    parser.add_argument("--indices", default="all")
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--max_objects", type=int, default=0)
    parser.add_argument("--sequences_per_object", type=int, default=1)
    parser.add_argument("--object_seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image_resolution", type=int, default=518)
    parser.add_argument("--vggt_feature_index", type=int, default=-1)
    parser.add_argument("--min_depth_matches", type=int, default=8)
    parser.add_argument("--affine_improvement_ratio", type=float, default=0.90)
    parser.add_argument("--save_correct_geometry", action="store_true")
    parser.add_argument("--allow_failures", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log_every", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"output directory is not empty; use --overwrite deliberately: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    dataset = PointPoseCacheDataset(args.source_cache_manifest, indices=args.indices)
    selection = {
        "mode": "sequence_order",
        "indices": str(args.indices),
        "max_samples": int(args.max_samples),
    }
    if int(args.max_objects) > 0:
        if int(args.max_samples) > 0:
            raise ValueError("--max_objects and --max_samples cannot be combined")
        dataset.samples = select_object_balanced_samples(
            dataset.samples,
            max_objects=int(args.max_objects),
            sequences_per_object=int(args.sequences_per_object),
            seed=int(args.object_seed),
        )
        selection = {
            "mode": "object_balanced",
            "indices": str(args.indices),
            "max_objects": int(args.max_objects),
            "sequences_per_object": int(args.sequences_per_object),
            "object_seed": int(args.object_seed),
        }
    limit = len(dataset) if int(args.max_samples) <= 0 else min(len(dataset), int(args.max_samples))

    # Stock conditions must be computed before loading the replacement VGGT used
    # for depth. This keeps the baseline semantically native even when the
    # ReconViaGen VGGT has no depth head.
    stock_conditions: dict[str, torch.Tensor] = {}
    failures: list[dict[str, str]] = []
    native_pipeline = build_native_stock_pipeline(args.pretrained, device)
    for sample_index in range(limit):
        batch = dataset[sample_index]
        uid = str(batch["uid"])
        try:
            stock_conditions[uid] = extract_stock_condition(
                native_pipeline, batch
            )
            if (sample_index + 1) % max(1, int(args.log_every)) == 0:
                print(
                    f"[pose_lifting_stock] {sample_index + 1}/{limit} uid={uid}",
                    flush=True,
                )
        except Exception as error:
            failures.append({"uid": uid, "error": repr(error)})
            if not args.allow_failures:
                raise
            print(f"[pose_lifting_stock] FAILED uid={uid}: {error!r}", flush=True)
    del native_pipeline
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    pipeline = build_lifting_pipeline(args.pretrained, args.vggt_pretrained, device)
    rows: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    feature_metadata: dict[str, Any] | None = None

    for sample_index in range(limit):
        batch = dataset[sample_index]
        uid = str(batch["uid"])
        if uid not in stock_conditions:
            continue
        try:
            full_images, masks, source_sizes = full_frame_masked_images(
                batch["image_paths"],
                batch["mask_paths"],
                resolution=int(args.image_resolution),
            )
            if len(set(source_sizes)) != 1:
                raise ValueError(f"uid={uid} source image sizes differ: {source_sizes}")
            source_width, source_height = source_sizes[0]
            intrinsic = scale_intrinsics(
                batch["intrinsics"].numpy(),
                source_width=source_width,
                source_height=source_height,
                target_width=int(args.image_resolution),
                target_height=int(args.image_resolution),
            )
            extrinsic = batch["extrinsics"].numpy().astype(np.float32)
            visual, depth, depth_confidence, current_metadata = extract_lifting_features(
                pipeline,
                full_images,
                vggt_feature_index=int(args.vggt_feature_index),
            )
            if feature_metadata is None:
                feature_metadata = current_metadata
            elif feature_metadata != current_metadata:
                raise RuntimeError(
                    f"uid={uid} feature schema changed: {current_metadata} != {feature_metadata}"
                )
            stock_condition = stock_conditions[uid]
            calibration = calibrate_vggt_depth(
                predicted_depth=depth.float().numpy(),
                depth_confidence=depth_confidence.float().numpy(),
                prior_coords=batch["prior_coords"].numpy(),
                prior_confidence=batch["prior_conf"].numpy(),
                intrinsics=intrinsic,
                extrinsics=extrinsic,
                grid_transform=batch["grid_transform"],
                extrinsics_type=batch["extrinsics_type"],
                camera_forward_sign=float(batch["camera_forward_sign"]),
                min_matches=int(args.min_depth_matches),
                affine_improvement_ratio=float(args.affine_improvement_ratio),
            )
            payload: dict[str, Any] = {
                "format": LIFTING_CACHE_VERSION,
                "uid": uid,
                "object_uid": batch["object_uid"],
                "visual_patch_features": visual.to(torch.float16),
                "predicted_depth": depth.to(torch.float16),
                "depth_confidence": depth_confidence.to(torch.float16),
                "masks": torch.from_numpy(masks).to(torch.float16),
                "intrinsics": torch.from_numpy(intrinsic),
                "extrinsics": torch.from_numpy(extrinsic),
                "view_ids": batch["view_ids"].to(torch.int32),
                "prior_coords": batch["prior_coords"].to(torch.int32),
                "prior_confidence": batch["prior_conf"].to(torch.float16),
                "stock_condition": stock_condition.to(torch.float16),
                "ss_latent": str(dataset.samples[sample_index]["ss_latent"]),
                "image_paths": batch["image_paths"],
                "mask_paths": batch["mask_paths"],
                "grid_transform": batch["grid_transform"],
                "extrinsics_type": batch["extrinsics_type"],
                "camera_forward_sign": float(batch["camera_forward_sign"]),
                "source_image_size": [source_height, source_width],
                "feature_image_size": [int(args.image_resolution)] * 2,
                "preprocessing": {
                    "stock_condition": "native unmodified ReconViaGen VGGT alpha crop/recenter",
                    "lifting_features": "full-frame masked resize without crop",
                    "intrinsics_scaled_to_lifting_features": True,
                },
                "depth_calibration": calibration,
            }
            if args.save_correct_geometry:
                geometry = build_projection_geometry(
                    intrinsics=payload["intrinsics"],
                    extrinsics=payload["extrinsics"],
                    grid_transform=batch["grid_transform"],
                    extrinsics_type=batch["extrinsics_type"],
                    camera_forward_sign=float(batch["camera_forward_sign"]),
                    image_height=int(args.image_resolution),
                    image_width=int(args.image_resolution),
                    patch_grid_side=int(round(current_metadata["patch_count"] ** 0.5)),
                    volume_side=16,
                )
                payload["correct_geometry"] = {
                    # Sampling grids stay FP32: FP16 coordinates measurably change
                    # high-dimensional bilinear patch sampling near cell borders.
                    "image_grid": geometry["image_grid"].to(torch.float32).cpu(),
                    "patch_grid": geometry["patch_grid"].to(torch.float32).cpu(),
                    "camera_depth": geometry["camera_depth"].to(torch.float32).cpu(),
                    "valid": geometry["valid"].to(torch.bool).cpu(),
                }
            relative_path = Path("samples") / uid[:2] / f"{uid}.pt"
            cache_path = output_dir / relative_path
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(payload, cache_path)
            rows.append(
                {
                    "uid": uid,
                    "object_uid": batch["object_uid"],
                    "cache_file": str(relative_path),
                    "ss_latent": payload["ss_latent"],
                    "view_count": len(batch["view_ids"]),
                    "prior_point_count": len(batch["prior_coords"]),
                    "depth_calibration_enabled": bool(calibration["enabled"]),
                    "depth_match_count": int(calibration["match_count"]),
                }
            )
            audits.append(
                {
                    "uid": uid,
                    "depth_calibration": calibration,
                    "visual_patch_abs_mean": float(visual.float().abs().mean().item()),
                    "depth_min": float(depth.float().min().item()),
                    "depth_max": float(depth.float().max().item()),
                    "mask_nonzero_ratio": float((torch.from_numpy(masks) > 0.5).float().mean().item()),
                }
            )
            if (sample_index + 1) % max(1, int(args.log_every)) == 0:
                print(
                    f"[pose_lifting_cache] {sample_index + 1}/{limit} uid={uid} "
                    f"views={len(batch['view_ids'])} depth={calibration['enabled']} "
                    f"matches={calibration['match_count']}",
                    flush=True,
                )
        except Exception as error:
            failures.append({"uid": uid, "error": repr(error)})
            if not args.allow_failures:
                raise
            print(f"[pose_lifting_cache] FAILED uid={uid}: {error!r}", flush=True)

    if not rows or feature_metadata is None:
        raise RuntimeError("pose lifting cache produced no valid samples")
    config = {
        "pretrained": args.pretrained,
        "vggt_pretrained": args.vggt_pretrained,
        "image_resolution": int(args.image_resolution),
        "vggt_feature_index": int(args.vggt_feature_index),
        "min_depth_matches": int(args.min_depth_matches),
        "affine_improvement_ratio": float(args.affine_improvement_ratio),
        "save_correct_geometry": bool(args.save_correct_geometry),
        "stock_condition_source": "native_unmodified_reconviagen_vggt",
        "lifting_feature_source": "separate_vggt_depth_pipeline",
    }
    manifest = {
        "format": LIFTING_CACHE_VERSION,
        "output_dir": str(output_dir.resolve()),
        "source_cache_manifest": str(Path(args.source_cache_manifest).resolve()),
        "stock_condition_source": "native_unmodified_reconviagen_vggt",
        "lifting_feature_source": "separate_vggt_depth_pipeline",
        "selection": selection,
        "samples": rows,
        "sample_count": len(rows),
        "object_count": len(
            {str(row.get("object_uid", row["uid"])) for row in rows}
        ),
        "failure_count": len(failures),
        "feature_metadata": feature_metadata,
        "visual_feature_dim": int(
            feature_metadata["vggt_feature_dim"] + feature_metadata["dino_feature_dim"]
        ),
        "metadata_names": list(LIFTING_METADATA_NAMES),
        "metadata_schema_hash": schema_hash(),
        "config": config,
        "config_hash": cache_config_hash(config),
        "depth_calibration_enabled_count": sum(
            int(row["depth_calibration_enabled"]) for row in rows
        ),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    manifest_path = output_dir / "manifest.json"
    manifest_sha256 = sha256_file(manifest_path)
    audit_report = {
        "passed": not failures,
        "cache_manifest": str(manifest_path.resolve()),
        "cache_manifest_sha256": manifest_sha256,
        "cache_config_hash": manifest["config_hash"],
        "uid_hash": hashlib.sha256(
            json.dumps(
                sorted(str(row["uid"]) for row in rows),
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
        "object_uid_hash": hashlib.sha256(
            json.dumps(
                sorted(
                    {
                        str(row.get("object_uid", row["uid"]))
                        for row in rows
                    }
                ),
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
        "sample_count": len(rows),
        "failure_count": len(failures),
        "depth_calibration_enabled_count": manifest["depth_calibration_enabled_count"],
        "depth_calibration_fallback_count": len(rows)
        - manifest["depth_calibration_enabled_count"],
        "samples": audits,
        "failures": failures,
    }
    (output_dir / "cache_audit.json").write_text(
        json.dumps(audit_report, indent=2), encoding="utf-8"
    )
    (output_dir / "failed_samples.json").write_text(
        json.dumps(failures, indent=2), encoding="utf-8"
    )
    print(json.dumps({key: manifest[key] for key in (
        "sample_count", "object_count", "failure_count", "visual_feature_dim", "depth_calibration_enabled_count"
    )}, indent=2))


if __name__ == "__main__":
    main()
