#!/usr/bin/env python3
"""Cache the unchanged Native v2 Full image conditions for real runtime-O inputs.

This is an input-only front end shared by inference and future domain training.
It runs the released ReconViaGen VGGT/DINO/condition encoders, while preserving
the explicit runtime-O K/T geometry used by Native SS and Native-SLAT lifting.
No Scan mesh, Scan alignment, or target latent is accepted by this program.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import shutil
import sys
from typing import Any

import numpy as np
from PIL import Image
import torch


TRACKER_ROOT = Path(__file__).resolve().parents[2]
for dependency in (
    TRACKER_ROOT,
    TRACKER_ROOT / "ReconViaGen",
    TRACKER_ROOT / "ReconViaGen" / "wheels" / "vggt",
):
    if str(dependency) not in sys.path:
        sys.path.insert(0, str(dependency))

from pose_aligned_reconstruction.dataset_tools.prepare_omni_real_runtime_inputs import (  # noqa: E402
    MANIFEST_FORMAT as RUNTIME_MANIFEST_FORMAT,
)
from pose_aligned_reconstruction.dataset_tools.prepare_omni_real_video_cache import (  # noqa: E402
    utc_now,
)
from pose_aligned_reconstruction.omni_real_benchmark_common import (  # noqa: E402
    atomic_json,
    atomic_torch_save,
    canonical_sha256,
    load_json,
    object_key,
    resolve_torch_device,
    select_rows,
    sha256_file,
    to_cpu_tree,
)
from reconvggt_ar_adapter_a.inspect_and_sanity import normalize_image_cond  # noqa: E402
from pose_aligned_reconstruction.real_object_canonicalization import (  # noqa: E402
    normalize_similarity_extrinsics,
)


OBJECT_FORMAT = "pose_point_depth_mv.omni_real_native_v2_model_input.v2"
MANIFEST_FORMAT = "pose_point_depth_mv.omni_real_native_v2_model_input_manifest.v2"
MARKER_FORMAT = "pose_point_depth_mv.omni_real_native_v2_model_input_marker.v2"
FEATURE_CONTRACT = {
    "preprocessing": "prepared runtime-O 518x518 black-background RGB",
    "vggt_feature_index": -1,
    "patch_start_idx": 5,
    "patch_side": 37,
    "patch_count": 1369,
    "visual_feature_dim": 3072,
    "dino_feature_dim": 1024,
    "dino_channel_location": "trailing",
    "projection_grid_transform": "identity",
    "extrinsics_type": "w2c",
    "extrinsics_source": "runtime_input_cache.T_O2C_lifting",
    "extrinsics_scale_policy": "projectively normalized proper similarity",
    "camera_forward_sign": 1.0,
    "projection_image_size": [518, 518],
    "depth_policy": "not_consumed_by_frustum-only Native v2 projection",
}


def _load_images(paths: list[str]) -> list[Image.Image]:
    images: list[Image.Image] = []
    for value in paths:
        with Image.open(value) as handle:
            image = handle.convert("RGB")
            if image.size != (518, 518):
                raise RuntimeError(f"prepared model input is not 518x518: {value}")
            images.append(image.copy())
    return images


def load_runtime_lifting_geometry(
    runtime_cache: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load and verify the v2 runtime geometry exported to Native lifting."""

    path = Path(runtime_cache).expanduser().resolve()
    with np.load(path, allow_pickle=False) as geometry:
        required = {"K_feature", "T_O2C", "T_O2C_lifting", "P_O"}
        missing = sorted(required.difference(geometry.files))
        if missing:
            raise RuntimeError(
                f"runtime cache lacks v2 lifting geometry {missing}: {path}"
            )
        intrinsics = np.asarray(geometry["K_feature"], dtype=np.float32)
        physical = np.asarray(geometry["T_O2C"], dtype=np.float64)
        lifting = np.asarray(geometry["T_O2C_lifting"], dtype=np.float64)
        points_o = np.asarray(geometry["P_O"], dtype=np.float32)
    expected = normalize_similarity_extrinsics(physical)
    if lifting.shape != expected.shape or not np.allclose(
        lifting, expected, rtol=1.0e-9, atol=1.0e-10
    ):
        raise RuntimeError(f"runtime T_O2C_lifting contract changed: {path}")
    if intrinsics.shape != (len(lifting), 3, 3):
        raise RuntimeError(f"runtime intrinsics/lifting view count differs: {path}")
    if points_o.ndim != 2 or points_o.shape[1] != 3:
        raise RuntimeError(f"runtime object points must be [N,3]: {path}")
    if not (
        np.isfinite(intrinsics).all()
        and np.isfinite(lifting).all()
        and np.isfinite(points_o).all()
    ):
        raise RuntimeError(f"runtime lifting geometry contains non-finite data: {path}")
    return intrinsics, lifting.astype(np.float32), points_o


def _build_encoder_pipeline(pretrained: str, device: torch.device):
    from pose_aligned_reconstruction.train_direct_flow import install_unused_model_stubs
    from trellis.pipelines import TrellisVGGTTo3DPipeline

    install_unused_model_stubs()
    pipeline = TrellisVGGTTo3DPipeline.from_pretrained(pretrained)
    pipeline._device = device
    pipeline.low_vram = False
    keep = {"image_cond_model", "sparse_structure_vggt_cond", "slat_vggt_cond"}
    for name in list(pipeline.models):
        if name not in keep:
            del pipeline.models[name]
    for name in keep:
        if name not in pipeline.models:
            raise RuntimeError(f"released pipeline lacks model input encoder={name}")
        pipeline.models[name].to(device).eval()
    if not hasattr(pipeline, "VGGT_model"):
        raise RuntimeError("released pipeline lacks native VGGT_model")
    pipeline.VGGT_model.to(device).eval()
    for module in [pipeline.VGGT_model, *(pipeline.models[name] for name in keep)]:
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    return pipeline


@torch.no_grad()
def encode_object(
    pipeline,
    runtime: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    images = _load_images(list(runtime["prepared_rgb_paths"]))
    aggregated, image_tensor = pipeline.vggt_feat(images)
    raw_image_cond = pipeline.encode_image(image_tensor)
    batch = int(aggregated[0].shape[0])
    views = int(aggregated[0].shape[1])
    image_cond = normalize_image_cond(raw_image_cond, batch=batch, views=views)
    patch_start = int(FEATURE_CONTRACT["patch_start_idx"])
    vggt_patch = aggregated[-1][:, :, patch_start:]
    dino_patch = image_cond[:, :, patch_start:]
    if vggt_patch.shape[:3] != dino_patch.shape[:3]:
        raise RuntimeError("released VGGT and DINO patch layouts differ")
    visual = torch.cat((vggt_patch, dino_patch), dim=-1)[0]
    expected = (
        views,
        int(FEATURE_CONTRACT["patch_count"]),
        int(FEATURE_CONTRACT["visual_feature_dim"]),
    )
    if tuple(visual.shape) != expected:
        raise RuntimeError(f"unexpected Native v2 visual shape={tuple(visual.shape)}")
    stock_condition = pipeline.get_ss_cond(
        image_cond[:, :, patch_start:], aggregated, num_samples=1
    )["cond"]
    slat_condition = pipeline.get_slat_cond(image_cond, aggregated, num_samples=1)
    tensors = [visual, stock_condition]
    tensors.extend(slat_condition["cond"])
    tensors.extend(slat_condition["neg_cond"])
    if not all(bool(torch.isfinite(value.float()).all().item()) for value in tensors):
        raise RuntimeError("Native v2 input encoder produced non-finite tensors")
    payload = {
        "format": OBJECT_FORMAT,
        "object_key": object_key(runtime),
        "condition_sha256": str(runtime["condition_sha256"]),
        "visual_patch_features": visual.to(torch.float16).cpu(),
        "stock_condition": stock_condition.to(torch.float16).cpu(),
        "slat_condition": to_cpu_tree(slat_condition),
        "feature_contract": FEATURE_CONTRACT,
    }
    stats = {
        "view_count": views,
        "visual_shape": list(visual.shape),
        "stock_condition_shape": list(stock_condition.shape),
        "slat_positive_view_count": len(slat_condition["cond"]),
        "slat_negative_view_count": len(slat_condition["neg_cond"]),
    }
    return payload, stats


def _reusable(
    destination: Path,
    *,
    runtime_hash: str,
    config_hash: str,
) -> dict[str, Any] | None:
    marker_path = destination / "_MODEL_INPUT_COMPLETE.json"
    report_path = destination / "report.json"
    if not marker_path.is_file() or not report_path.is_file():
        return None
    marker = load_json(marker_path)
    if (
        marker.get("format") != MARKER_FORMAT
        or marker.get("runtime_cache_sha256") != runtime_hash
        or marker.get("config_sha256") != config_hash
    ):
        raise RuntimeError(f"stale Native v2 model input: {destination}")
    report = load_json(report_path)
    payload = Path(str(report.get("model_input", "")))
    if (
        report.get("format") != OBJECT_FORMAT
        or report.get("passed") is not True
        or not payload.is_file()
        or sha256_file(payload) != report.get("model_input_sha256")
    ):
        raise RuntimeError(f"invalid reusable Native v2 model input: {destination}")
    return report


def build_object(
    pipeline,
    runtime: dict[str, Any],
    *,
    output_dir: Path,
    pretrained: str,
    config_hash: str,
    resume_partial: bool = False,
) -> tuple[dict[str, Any], bool]:
    runtime_cache = Path(runtime["cache_npz"]).resolve()
    runtime_hash = sha256_file(runtime_cache)
    destination = output_dir / "objects" / runtime["category"] / runtime["object_id"]
    reused = _reusable(
        destination, runtime_hash=runtime_hash, config_hash=config_hash
    )
    if reused is not None:
        return reused, True
    if destination.exists():
        raise RuntimeError(f"partial Native v2 model input exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / f".{runtime['object_id']}.model-input-building"
    if staging.exists():
        if not resume_partial:
            raise RuntimeError(f"partial Native v2 staging exists: {staging}")
        shutil.rmtree(staging)
    staging.mkdir()

    intrinsics, extrinsics, points_o = load_runtime_lifting_geometry(runtime_cache)
    payload, stats = encode_object(pipeline, runtime)
    payload.update(
        {
            "runtime_cache": str(runtime_cache),
            "runtime_cache_sha256": runtime_hash,
            "intrinsics": torch.from_numpy(intrinsics),
            "extrinsics": torch.from_numpy(extrinsics),
            "points_o": torch.from_numpy(points_o),
            "grid_transform": "identity",
            "extrinsics_type": "w2c",
            "extrinsics_source": "T_O2C_lifting",
            "camera_forward_sign": 1.0,
            "projection_image_size": (518, 518),
        }
    )
    payload_path = staging / "native_v2_model_input.pt"
    atomic_torch_save(payload_path, payload)
    final_payload = destination / payload_path.name
    report = {
        "format": OBJECT_FORMAT,
        "created_at_utc": utc_now(),
        "category": str(runtime["category"]),
        "object_id": str(runtime["object_id"]),
        "object_key": object_key(runtime),
        "pretrained": str(pretrained),
        "runtime_input_report": str(Path(runtime_cache).parent / "report.json"),
        "runtime_cache": str(runtime_cache),
        "runtime_cache_sha256": runtime_hash,
        "condition_sha256": str(runtime["condition_sha256"]),
        "reference_view_index": int(runtime["reference_view_index"]),
        "prepared_rgb_paths": list(runtime["prepared_rgb_paths"]),
        "prepared_mask_paths": list(runtime["prepared_mask_paths"]),
        "model_input": str(final_payload),
        "model_input_sha256": sha256_file(payload_path),
        "feature_contract": FEATURE_CONTRACT,
        "extrinsics_source": "runtime_input_cache.T_O2C_lifting",
        "encoder_stats": stats,
        "forbidden_gt_fields_absent": True,
        "training_ready": False,
        "scope_guard": (
            "Input-only unchanged Native v2 Full encoder cache. No Scan mesh, "
            "Scan alignment, target latent, or evaluation metric is consumed."
        ),
        "passed": True,
    }
    atomic_json(staging / "report.json", report)
    atomic_json(
        staging / "_MODEL_INPUT_COMPLETE.json",
        {
            "format": MARKER_FORMAT,
            "runtime_cache_sha256": runtime_hash,
            "config_sha256": config_hash,
            "model_input_sha256": report["model_input_sha256"],
            "condition_sha256": report["condition_sha256"],
            "passed": True,
        },
    )
    staging.replace(destination)
    return report, False


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime_input_manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--pretrained", default="Stable-X/trellis-vggt-v0-2")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--object", action="append")
    parser.add_argument("--allow_failures", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> None:
    args = make_parser().parse_args()
    runtime_path = Path(args.runtime_input_manifest).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    runtime = load_json(runtime_path)
    if runtime.get("format") != RUNTIME_MANIFEST_FORMAT or runtime.get("passed") is not True:
        raise RuntimeError(f"runtime input manifest did not pass: {runtime_path}")
    rows = select_rows(runtime.get("objects", []), args.object)
    config = {
        "pretrained": str(args.pretrained),
        "runtime_build_config_sha256": str(runtime["build_config_sha256"]),
        "feature_contract": FEATURE_CONTRACT,
    }
    config_hash = canonical_sha256(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_torch_device(args.device)
    pipeline = _build_encoder_pipeline(str(args.pretrained), device)
    reports: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    reused: list[str] = []
    try:
        for position, row in enumerate(rows, start=1):
            key = object_key(row)
            print(f"[real_native_v2_input] {position}/{len(rows)} object={key}", flush=True)
            try:
                report, was_reused = build_object(
                    pipeline,
                    row,
                    output_dir=output_dir,
                    pretrained=str(args.pretrained),
                    config_hash=config_hash,
                    resume_partial=bool(args.resume),
                )
                reports.append(report)
                if was_reused:
                    reused.append(key)
            except Exception as error:
                failures.append({"object_key": key, "error": repr(error)})
                print(f"[real_native_v2_input] FAILED object={key}: {error!r}", flush=True)
                if not args.allow_failures:
                    raise
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        del pipeline
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    manifest = {
        "format": MANIFEST_FORMAT,
        "created_at_utc": utc_now(),
        "runtime_input_manifest": str(runtime_path),
        "runtime_input_manifest_sha256": sha256_file(runtime_path),
        "config": config,
        "config_sha256": config_hash,
        "selected_object_count": len(rows),
        "completed_object_count": len(reports),
        "reused_objects": reused,
        "objects": reports,
        "failures": failures,
        "training_ready": False,
        "scope_guard": "Model inputs only; target/label construction remains separate.",
    }
    manifest["passed"] = bool(reports) and not failures
    manifest_path = output_dir / "model_input_manifest.json"
    atomic_json(manifest_path, manifest)
    print(json.dumps({
        "passed": manifest["passed"],
        "completed_object_count": len(reports),
        "failure_count": len(failures),
        "manifest": str(manifest_path),
    }, indent=2))
    if not manifest["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
