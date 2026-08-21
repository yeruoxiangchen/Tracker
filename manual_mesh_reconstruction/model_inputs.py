#!/usr/bin/env python3
"""Encode runtime-O observations with DINO only; VGGT is never loaded or run."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import shutil
import sys
from typing import Any

from PIL import Image
import torch


TRACKER_ROOT = Path(__file__).resolve().parents[1]
for dependency in (TRACKER_ROOT, TRACKER_ROOT / "ReconViaGen"):
    if str(dependency) not in sys.path:
        sys.path.insert(0, str(dependency))

from manual_mesh_reconstruction.model_geometry import (  # noqa: E402
    load_runtime_lifting_geometry,
)
from manual_mesh_reconstruction.runtime_o import (  # noqa: E402
    MANIFEST_FORMAT as RUNTIME_MANIFEST_FORMAT,
)
from manual_mesh_reconstruction.raw_cache import (  # noqa: E402
    utc_now,
)
from manual_mesh_reconstruction.dino_condition import (  # noqa: E402
    DEFAULT_SS_CONTEXT_TOKENS,
    DINO_ONLY_CONTEXT_VERSION,
    build_dino_only_contexts,
    tensor_tree_sha256,
)
from manual_mesh_reconstruction.common import (  # noqa: E402
    atomic_json,
    atomic_torch_save,
    canonical_sha256,
    load_json,
    object_key,
    resolve_torch_device,
    select_rows,
    sha256_file,
)
from reconvggt_ar_adapter_a.inspect_and_sanity import normalize_image_cond  # noqa: E402


OBJECT_FORMAT = "pose_point_depth_mv.omni_real_dino_only_model_input.v1"
MANIFEST_FORMAT = "pose_point_depth_mv.omni_real_dino_only_model_input_manifest.v1"
MARKER_FORMAT = "pose_point_depth_mv.omni_real_dino_only_model_input_marker.v1"
COMPATIBLE_RUNTIME_MANIFEST_FORMATS = {
    RUNTIME_MANIFEST_FORMAT,
    "pose_point_depth_mv.omni_real_runtime_input_manifest.v2",
}
FEATURE_CONTRACT = {
    "version": DINO_ONLY_CONTEXT_VERSION,
    "preprocessing": "prepared runtime-O 518x518 black-background RGB",
    "patch_start_idx": 5,
    "patch_side": 37,
    "patch_count": 1369,
    "visual_feature_dim": 1024,
    "vggt_feature_dim": 0,
    "dino_feature_dim": 1024,
    "context_source": "raw_dino_only",
    "projection_grid_transform": "identity",
    "extrinsics_type": "w2c",
    "extrinsics_source": "runtime_input_cache.T_O2C_lifting",
    "camera_forward_sign": 1.0,
    "projection_image_size": [518, 518],
    "depth_policy": "zero_placeholder_not_consumed",
    "vggt_model_loaded": False,
    "vggt_model_executed": False,
}


def _load_images(paths: list[str]) -> list[Image.Image]:
    images: list[Image.Image] = []
    for value in paths:
        with Image.open(value) as handle:
            image = handle.convert("RGB")
            if image.size != (518, 518):
                raise RuntimeError(f"prepared model input is not 518x518: {value}")
            images.append(image.copy())
    if not images:
        raise ValueError("DINO-only input has no images")
    return images


def _build_dino_pipeline(dino_model: str, device: torch.device):
    from trellis.pipelines import TrellisImageTo3DPipeline

    # Construct only the image encoder.  ``from_pretrained`` would also load
    # every SS/SLat model listed in pipeline.json even though this stage needs
    # none of them.
    pipeline = TrellisImageTo3DPipeline()
    pipeline.models = {}
    pipeline._device = device
    pipeline.low_vram = False
    pipeline._init_image_cond_model(str(dino_model))
    if hasattr(pipeline, "VGGT_model"):
        raise RuntimeError("DINO-only pipeline unexpectedly constructed VGGT")
    encoder = pipeline.models.get("image_cond_model")
    if encoder is None:
        raise RuntimeError("released pipeline lacks DINO image_cond_model")
    encoder.to(device).eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    return pipeline


@torch.no_grad()
def encode_object(
    pipeline: Any, runtime: dict[str, Any], *, ss_context_tokens: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    images = _load_images(list(runtime["prepared_rgb_paths"]))
    raw = pipeline.encode_image(images)
    image_cond = normalize_image_cond(raw, batch=1, views=len(images))
    patch_start = int(FEATURE_CONTRACT["patch_start_idx"])
    dino_patch = image_cond[0, :, patch_start:]
    contexts = build_dino_only_contexts(
        dino_patch, ss_context_tokens=int(ss_context_tokens)
    )
    expected = (
        len(images),
        int(FEATURE_CONTRACT["patch_count"]),
        int(FEATURE_CONTRACT["visual_feature_dim"]),
    )
    if tuple(contexts["visual_patch_features"].shape) != expected:
        raise RuntimeError(
            "unexpected DINO-only visual shape="
            f"{tuple(contexts['visual_patch_features'].shape)}"
        )
    slat = {
        key: [value.to(torch.float16).cpu() for value in values]
        for key, values in contexts["slat_condition"].items()
    }
    payload = {
        "format": OBJECT_FORMAT,
        "object_key": object_key(runtime),
        "condition_sha256": str(runtime["condition_sha256"]),
        "visual_patch_features": contexts["visual_patch_features"]
        .to(torch.float16)
        .cpu(),
        "stock_condition": contexts["stock_condition"].to(torch.float16).cpu(),
        "slat_condition": slat,
        "slat_condition_tree_sha256": tensor_tree_sha256(slat),
        "feature_contract": FEATURE_CONTRACT,
        "context_contract": contexts["context_contract"],
        "vggt_model_loaded": False,
        "vggt_model_executed": False,
    }
    stats = {
        "view_count": len(images),
        "visual_shape": list(payload["visual_patch_features"].shape),
        "stock_condition_shape": list(payload["stock_condition"].shape),
        "slat_positive_view_count": len(slat["cond"]),
        "slat_negative_view_count": len(slat["neg_cond"]),
    }
    return payload, stats


def _reusable(
    destination: Path, *, runtime_hash: str, config_hash: str
) -> dict[str, Any] | None:
    marker_path = destination / "_DINO_ONLY_MODEL_INPUT_COMPLETE.json"
    report_path = destination / "report.json"
    if not marker_path.is_file() or not report_path.is_file():
        return None
    marker = load_json(marker_path)
    if (
        marker.get("format") != MARKER_FORMAT
        or marker.get("runtime_cache_sha256") != runtime_hash
        or marker.get("config_sha256") != config_hash
    ):
        raise RuntimeError(f"stale DINO-only model input: {destination}")
    report = load_json(report_path)
    payload = Path(str(report.get("model_input", "")))
    if (
        report.get("format") != OBJECT_FORMAT
        or report.get("passed") is not True
        or not payload.is_file()
        or sha256_file(payload) != report.get("model_input_sha256")
    ):
        raise RuntimeError(f"invalid reusable DINO-only model input: {destination}")
    return report


def build_object(
    pipeline: Any,
    runtime: dict[str, Any],
    *,
    output_dir: Path,
    pretrained: str,
    config_hash: str,
    ss_context_tokens: int,
    resume_partial: bool,
) -> tuple[dict[str, Any], bool]:
    runtime_cache = Path(runtime["cache_npz"]).resolve()
    runtime_hash = sha256_file(runtime_cache)
    destination = output_dir / "objects" / runtime["category"] / runtime["object_id"]
    reused = _reusable(destination, runtime_hash=runtime_hash, config_hash=config_hash)
    if reused is not None:
        return reused, True
    if destination.exists():
        raise RuntimeError(f"partial DINO-only model input exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / f".{runtime['object_id']}.dino-only-building"
    if staging.exists():
        if not resume_partial:
            raise RuntimeError(f"partial DINO-only staging exists: {staging}")
        shutil.rmtree(staging)
    staging.mkdir()
    intrinsics, extrinsics, points_o = load_runtime_lifting_geometry(runtime_cache)
    payload, stats = encode_object(
        pipeline, runtime, ss_context_tokens=int(ss_context_tokens)
    )
    views = len(extrinsics)
    payload.update(
        {
            "runtime_cache": str(runtime_cache),
            "runtime_cache_sha256": runtime_hash,
            "intrinsics": torch.from_numpy(intrinsics),
            "extrinsics": torch.from_numpy(extrinsics),
            "points_o": torch.from_numpy(points_o),
            "predicted_depth": torch.zeros((views, 518, 518), dtype=torch.float16),
            "depth_confidence": torch.ones((views, 518, 518), dtype=torch.float16),
            "grid_transform": "identity",
            "extrinsics_type": "w2c",
            "extrinsics_source": "T_O2C_lifting",
            "camera_forward_sign": 1.0,
            "projection_image_size": (518, 518),
        }
    )
    payload_path = staging / "dino_only_model_input.pt"
    atomic_torch_save(payload_path, payload)
    final_payload = destination / payload_path.name
    report = {
        "format": OBJECT_FORMAT,
        "created_at_utc": utc_now(),
        "category": str(runtime["category"]),
        "object_id": str(runtime["object_id"]),
        "object_key": object_key(runtime),
        "pretrained": str(pretrained),
        "runtime_cache": str(runtime_cache),
        "runtime_cache_sha256": runtime_hash,
        "condition_sha256": str(runtime["condition_sha256"]),
        "reference_view_index": int(runtime["reference_view_index"]),
        "prepared_rgb_paths": list(runtime["prepared_rgb_paths"]),
        "prepared_mask_paths": list(runtime["prepared_mask_paths"]),
        "model_input": str(final_payload),
        "model_input_sha256": sha256_file(payload_path),
        "feature_contract": FEATURE_CONTRACT,
        "encoder_stats": stats,
        "vggt_model_loaded": False,
        "vggt_model_executed": False,
        "target_or_mesh_consumed": False,
        "training_ready": False,
        "scope_guard": "DINO-only runtime-O model inputs; no VGGT or GT mesh.",
        "passed": True,
    }
    atomic_json(staging / "report.json", report)
    atomic_json(
        staging / "_DINO_ONLY_MODEL_INPUT_COMPLETE.json",
        {
            "format": MARKER_FORMAT,
            "runtime_cache_sha256": runtime_hash,
            "config_sha256": config_hash,
            "model_input_sha256": report["model_input_sha256"],
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
    parser.add_argument("--dino_model", default="dinov2_vitl14_reg")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--object", action="append")
    parser.add_argument(
        "--ss_context_tokens", type=int, default=DEFAULT_SS_CONTEXT_TOKENS
    )
    parser.add_argument("--allow_failures", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> None:
    args = make_parser().parse_args()
    if int(args.ss_context_tokens) <= 0:
        raise ValueError("ss_context_tokens must be positive")
    runtime_path = Path(args.runtime_input_manifest).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    runtime = load_json(runtime_path)
    if (
        runtime.get("format") not in COMPATIBLE_RUNTIME_MANIFEST_FORMATS
        or runtime.get("passed") is not True
    ):
        raise RuntimeError(f"runtime input manifest did not pass: {runtime_path}")
    rows = select_rows(runtime.get("objects", []), args.object)
    config = {
        "pretrained": str(args.pretrained),
        "dino_model": str(args.dino_model),
        "runtime_build_config_sha256": str(runtime["build_config_sha256"]),
        "feature_contract": FEATURE_CONTRACT,
        "ss_context_tokens": int(args.ss_context_tokens),
    }
    config_hash = canonical_sha256(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_torch_device(args.device)
    pipeline = _build_dino_pipeline(str(args.dino_model), device)
    reports: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    reused: list[str] = []
    try:
        for position, row in enumerate(rows, start=1):
            key = object_key(row)
            print(f"[real_dino_only_input] {position}/{len(rows)} object={key}", flush=True)
            try:
                report, was_reused = build_object(
                    pipeline,
                    row,
                    output_dir=output_dir,
                    pretrained=str(args.pretrained),
                    config_hash=config_hash,
                    ss_context_tokens=int(args.ss_context_tokens),
                    resume_partial=bool(args.resume),
                )
                reports.append(report)
                if was_reused:
                    reused.append(key)
            except Exception as error:
                failures.append({"object_key": key, "error": repr(error)})
                print(f"[real_dino_only_input] FAILED object={key}: {error!r}", flush=True)
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
        "vggt_model_loaded": False,
        "vggt_model_executed": False,
        "training_ready": False,
        "scope_guard": "DINO-only model inputs; target construction remains separate.",
    }
    manifest["passed"] = bool(reports) and not failures
    manifest_path = output_dir / "model_input_manifest.json"
    atomic_json(manifest_path, manifest)
    print(
        json.dumps(
            {
                "passed": manifest["passed"],
                "completed_object_count": len(reports),
                "failure_count": len(failures),
                "manifest": str(manifest_path),
            },
            indent=2,
        )
    )
    if not manifest["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
