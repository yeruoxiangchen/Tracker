#!/usr/bin/env python3
"""Run the frozen original ReconViaGen release on real benchmark inputs."""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
from PIL import Image
import torch


TRACKER_ROOT = Path(__file__).resolve().parents[1]
for dependency in (
    TRACKER_ROOT,
    TRACKER_ROOT / "ReconViaGen",
    TRACKER_ROOT / "ReconViaGen" / "wheels" / "vggt",
):
    if str(dependency) not in sys.path:
        sys.path.insert(0, str(dependency))

from manual_mesh_reconstruction.runtime_o import (  # noqa: E402
    MANIFEST_FORMAT as RUNTIME_MANIFEST_FORMAT,
)
from manual_mesh_reconstruction.raw_cache import utc_now  # noqa: E402
from pose_point_depth_mv.mesh_benchmark_metrics import mesh_structure_metrics  # noqa: E402
from manual_mesh_reconstruction.common import (  # noqa: E402
    atomic_json,
    load_json,
    object_key,
    resolve_torch_device,
    select_rows,
    sha256_file,
)


REPORT_FORMAT = "pose_point_depth_mv.omni_real_reconviagen_inference.v1"
MANIFEST_FORMAT = "pose_point_depth_mv.omni_real_reconviagen_inference_manifest.v1"
LEGACY_IMAGE_ONLY_RUNTIME_MANIFEST_FORMAT = (
    "pose_point_depth_mv.omni_real_runtime_input_manifest.v1"
)
FROZEN_V2_IMAGE_ONLY_RUNTIME_MANIFEST_FORMAT = (
    "pose_point_depth_mv.omni_real_runtime_input_manifest.v2"
)
IMAGE_ONLY_RUNTIME_MANIFEST_FORMATS = {
    LEGACY_IMAGE_ONLY_RUNTIME_MANIFEST_FORMAT,
    FROZEN_V2_IMAGE_ONLY_RUNTIME_MANIFEST_FORMAT,
    RUNTIME_MANIFEST_FORMAT,
}


def parse_csv_int(value: str) -> list[int]:
    values = [int(item.strip()) for item in str(value).split(",") if item.strip()]
    if not values or len(values) != len(set(values)):
        raise ValueError("seeds must be non-empty and unique")
    return values


def _load_images(row: dict[str, Any]) -> list[Image.Image]:
    images = []
    for value in row["prepared_rgb_paths"]:
        with Image.open(value) as handle:
            if handle.size != (518, 518):
                raise RuntimeError(f"ReconViaGen input is not 518x518: {value}")
            images.append(handle.convert("RGB").copy())
    if len(images) != int(row["selected_view_count"]):
        raise RuntimeError(f"ReconViaGen view count differs: {object_key(row)}")
    return images


def _build_pipeline(pretrained: str, device: torch.device, low_vram: bool):
    from pose_point_depth_mv.train_direct_flow import install_unused_model_stubs
    from trellis.pipelines import TrellisVGGTTo3DPipeline

    install_unused_model_stubs()
    pipeline = TrellisVGGTTo3DPipeline.from_pretrained(pretrained)
    pipeline._device = device
    pipeline.low_vram = bool(low_vram)
    if not low_vram:
        pipeline.VGGT_model.to(device).eval()
        for module in pipeline.models.values():
            module.to(device).eval()
    return pipeline


def _paths(output_dir: Path, row: dict[str, Any], seed: int) -> tuple[Path, Path]:
    root = output_dir / "meshes" / row["category"] / row["object_id"] / f"seed_{seed}"
    return root / "mesh_reference_o.obj", root / "result.json"


def _reuse(
    result_path: Path,
    mesh_path: Path,
    *,
    row: dict[str, Any],
    seed: int,
    runtime_sha256: str,
    pretrained: str,
) -> dict[str, Any] | None:
    if not result_path.is_file() or not mesh_path.is_file():
        return None
    result = load_json(result_path)
    expected = {
        "format": REPORT_FORMAT,
        "object_key": object_key(row),
        "seed": int(seed),
        "runtime_input_manifest_sha256": runtime_sha256,
        "pretrained": str(pretrained),
        "mesh_sha256": sha256_file(mesh_path),
    }
    mismatch = {
        key: (result.get(key), value)
        for key, value in expected.items()
        if result.get(key) != value
    }
    if mismatch:
        raise RuntimeError(f"stale ReconViaGen result={mismatch}")
    return result


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime_input_manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--pretrained", default="Stable-X/trellis-vggt-v0-2")
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--low_vram", action="store_true")
    parser.add_argument("--object", action="append")
    return parser


def main() -> None:
    args = make_parser().parse_args()
    runtime_path = Path(args.runtime_input_manifest).expanduser().resolve()
    runtime = load_json(runtime_path)
    if (
        runtime.get("format") not in IMAGE_ONLY_RUNTIME_MANIFEST_FORMATS
        or runtime.get("passed") is not True
    ):
        raise RuntimeError(f"runtime input manifest did not pass: {runtime_path}")
    rows = select_rows(runtime.get("objects", []), args.object)
    seeds = parse_csv_int(args.seeds)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    runtime_sha = sha256_file(runtime_path)
    device = resolve_torch_device(args.device)
    pipeline = _build_pipeline(args.pretrained, device, bool(args.low_vram))
    reports: list[dict[str, Any]] = []
    try:
        for position, row in enumerate(rows, start=1):
            runtime_cache_sha256 = str(
                row.get("source_raw_cache_sha256")
                or row.get("cache_npz_sha256")
                or ""
            )
            if not runtime_cache_sha256:
                raise RuntimeError(
                    f"ReconViaGen runtime cache identity is missing: {object_key(row)}"
                )
            images = _load_images(row)
            for seed in seeds:
                mesh_path, result_path = _paths(output_dir, row, seed)
                reused = _reuse(
                    result_path,
                    mesh_path,
                    row=row,
                    seed=seed,
                    runtime_sha256=runtime_sha,
                    pretrained=args.pretrained,
                )
                if reused is not None:
                    reports.append(reused)
                    continue
                if mesh_path.parent.exists():
                    raise RuntimeError(f"partial ReconViaGen output: {mesh_path.parent}")
                outputs, coords, _ = pipeline.run(
                    images,
                    num_samples=1,
                    seed=int(seed),
                    formats=["mesh"],
                    preprocess_image=False,
                )
                decoded = outputs["mesh"][0]
                mesh = decoded.to_trimesh(transform_pose=False)
                structure = mesh_structure_metrics(mesh)
                if not structure["mesh_success"]:
                    raise RuntimeError(f"ReconViaGen decoded empty mesh: {object_key(row)}")
                mesh_path.parent.mkdir(parents=True, exist_ok=False)
                temporary = mesh_path.with_name(f".{mesh_path.name}.tmp-{os.getpid()}")
                mesh.export(temporary, file_type="obj")
                os.replace(temporary, mesh_path)
                result = {
                    "format": REPORT_FORMAT,
                    "created_at_utc": utc_now(),
                    "method": "reconviagen_original",
                    "object_key": object_key(row),
                    "category": row["category"],
                    "object_id": row["object_id"],
                    "seed": int(seed),
                    "view_count": len(images),
                    "pretrained": str(args.pretrained),
                    "runtime_input_manifest": str(runtime_path),
                    "runtime_input_manifest_sha256": runtime_sha,
                    "runtime_cache_sha256": runtime_cache_sha256,
                    "mesh": str(mesh_path),
                    "mesh_sha256": sha256_file(mesh_path),
                    "coord_count": int(coords.shape[0]),
                    "structure": structure,
                    "output_frame": "reference-view canonical O diagnostic",
                    "explicit_runtime_pose_condition": False,
                    "target_or_metric_consumed": False,
                    "passed": True,
                }
                atomic_json(result_path, result)
                reports.append(result)
                print(
                    f"[real_reconviagen] {position}/{len(rows)} "
                    f"object={object_key(row)} seed={seed}",
                    flush=True,
                )
                del outputs, coords, decoded, mesh
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            del images
    finally:
        del pipeline
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    manifest = {
        "format": MANIFEST_FORMAT,
        "created_at_utc": utc_now(),
        "method": "reconviagen_original",
        "runtime_input_manifest": str(runtime_path),
        "runtime_input_manifest_sha256": runtime_sha,
        "pretrained": str(args.pretrained),
        "seeds": seeds,
        "object_count": len(rows),
        "record_count": len(reports),
        "objects": reports,
        "target_or_metric_consumed": False,
        "passed": len(reports) == len(rows) * len(seeds),
    }
    manifest_path = output_dir / "inference_manifest.json"
    atomic_json(manifest_path, manifest)
    print(json.dumps({
        "passed": manifest["passed"],
        "object_count": len(rows),
        "record_count": len(reports),
        "manifest": str(manifest_path),
    }, indent=2))
    if not manifest["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
