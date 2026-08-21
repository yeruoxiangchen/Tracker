#!/usr/bin/env python3
"""Run original ReconViaGen stock on the RGB views bound to a Direct cache."""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import sys
from typing import Any

from .bunny_review.common import (
    atomic_json,
    binding,
    canonical_sha256,
    code_bindings,
    sha256_file,
    validate_binding,
)
from .bunny_review.infer_reconviagen import export_mesh_atomic


FORMAT = "pose_point_depth_mv.reconviagen_stock_from_direct_cache.v1"
TRACKER_ROOT = Path(__file__).resolve().parents[1]
RECONVIAGEN_ROOT = TRACKER_ROOT / "ReconViaGen"
VGGT_ROOT = RECONVIAGEN_ROOT / "wheels" / "vggt"


def resolve_from(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def load_cache_row(manifest_path: Path, uid: str) -> tuple[dict[str, Any], Path]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    matches = [
        row for row in manifest.get("samples", []) if str(row.get("uid")) == uid
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one source-lifting row for uid={uid!r}; "
            f"found {len(matches)}"
        )
    row = matches[0]
    cache_path = resolve_from(manifest_path.parent, str(row["cache_file"]))
    if not cache_path.is_file():
        raise FileNotFoundError(cache_path)
    return row, cache_path


def mesh_stats(mesh: Any) -> dict[str, Any]:
    return {
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "bounds": mesh.bounds.tolist(),
        "extents": mesh.extents.tolist(),
        "watertight": bool(mesh.is_watertight),
    }


def validate_reusable_report(
    report_path: Path,
    *,
    expected_config: dict[str, Any],
) -> dict[str, Any]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("format") != FORMAT or report.get("complete") is not True:
        raise RuntimeError(f"incomplete or unsupported report: {report_path}")
    if report.get("run_config") != expected_config:
        raise RuntimeError(
            "existing ReconViaGen stock output has a different immutable config"
        )
    validate_binding(report["mesh_canonical"], label="mesh_canonical")
    validate_binding(report["mesh_view"], label="mesh_view")
    return report


def run(args: argparse.Namespace) -> None:
    manifest_path = args.source_lifting_manifest.resolve()
    row, cache_path = load_cache_row(manifest_path, args.uid)

    import torch

    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    if not isinstance(cache, dict):
        raise TypeError(f"Direct source cache is not a dictionary: {cache_path}")
    if str(cache.get("uid")) != args.uid:
        raise RuntimeError("Direct source cache UID differs from manifest")
    image_paths = [Path(value).resolve() for value in cache.get("image_paths", [])]
    mask_paths = [Path(value).resolve() for value in cache.get("mask_paths", [])]
    if not image_paths or len(image_paths) != len(mask_paths):
        raise RuntimeError("Direct source cache has invalid image/mask bindings")
    if int(row.get("view_count", -1)) != len(image_paths):
        raise RuntimeError("Direct source cache view count differs from manifest")
    for path in [*image_paths, *mask_paths]:
        if not path.is_file():
            raise FileNotFoundError(path)

    output_dir = args.output_dir.resolve()
    report_path = output_dir / "report.json"
    run_config = {
        "uid": args.uid,
        "object_uid": str(row["object_uid"]),
        "source_lifting_manifest": binding(manifest_path),
        "source_cache": binding(cache_path),
        "input_images": [binding(path) for path in image_paths],
        "input_masks": [binding(path) for path in mask_paths],
        "view_ids": [int(value) for value in cache["view_ids"].tolist()],
        "pretrained": str(args.pretrained),
        "seed": int(args.seed),
        "multiimage_algo": args.multiimage_algo,
        "low_vram": bool(args.low_vram),
        "sampling": {
            "ss_steps": int(args.ss_steps),
            "ss_guidance": float(args.ss_guidance),
            "ss_guidance_rescale": float(args.ss_guidance_rescale),
            "ss_rescale_t": float(args.ss_rescale_t),
            "slat_steps": int(args.slat_steps),
            "slat_guidance": float(args.slat_guidance),
            "slat_guidance_rescale": float(args.slat_guidance_rescale),
            "slat_rescale_t": float(args.slat_rescale_t),
        },
        "background_policy": "frozen GT mask written to RGBA alpha; BiRefNet skipped",
    }
    run_config["config_sha256"] = canonical_sha256(run_config)
    if report_path.is_file():
        report = validate_reusable_report(
            report_path,
            expected_config=run_config,
        )
        print(
            json.dumps(
                {
                    "status": "reused",
                    "report": str(report_path),
                    "mesh": report["mesh_canonical"]["path"],
                },
                indent=2,
            )
        )
        return
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(
            f"partial ReconViaGen stock output exists; preserve and inspect: "
            f"{output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    for path in (TRACKER_ROOT, RECONVIAGEN_ROOT, VGGT_ROOT):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    from trellis.pipelines import TrellisVGGTTo3DPipeline
    from reconvggt_ar_adapter_a.train_pointpose_ss_lora import rgba_images

    pipeline = TrellisVGGTTo3DPipeline.from_pretrained(args.pretrained)
    pipeline._device = torch.device(args.device)
    pipeline.low_vram = bool(args.low_vram)
    if not pipeline.low_vram:
        for model in pipeline.models.values():
            model.to(pipeline._device)
        pipeline.VGGT_model.to(pipeline._device)

    images = rgba_images(
        [str(path) for path in image_paths],
        [str(path) for path in mask_paths],
        pipeline,
    )
    print(
        f"[reconviagen_stock_cache] uid={args.uid} "
        f"views={len(images)} seed={args.seed}",
        flush=True,
    )
    outputs, coords, ss_noise = pipeline.run(
        image=images,
        seed=int(args.seed),
        formats=["gaussian", "mesh"],
        preprocess_image=False,
        sparse_structure_sampler_params={
            "steps": int(args.ss_steps),
            "cfg_strength": float(args.ss_guidance),
            "cfg_interval": [0.6, 1.0],
            "guidance_rescale": float(args.ss_guidance_rescale),
            "rescale_t": float(args.ss_rescale_t),
        },
        slat_sampler_params={
            "steps": int(args.slat_steps),
            "cfg_strength": float(args.slat_guidance),
            "cfg_interval": [0.6, 1.0],
            "guidance_rescale": float(args.slat_guidance_rescale),
            "rescale_t": float(args.slat_rescale_t),
        },
        mode=args.multiimage_algo,
    )
    decoded = outputs["mesh"][0]
    gaussian = outputs["gaussian"][0]
    canonical = decoded.to_trimesh(transform_pose=False)
    view_mesh = decoded.to_trimesh(transform_pose=True)
    canonical_path = output_dir / "mesh_canonical.obj"
    view_path = output_dir / "mesh_view.glb"
    export_mesh_atomic(canonical, canonical_path, file_type="obj")
    export_mesh_atomic(view_mesh, view_path, file_type="glb")

    report = {
        "format": FORMAT,
        "complete": True,
        "formal": False,
        "purpose": (
            "original ReconViaGen full-chain stock baseline using exactly the "
            "RGB views bound to the Direct source cache"
        ),
        "run_config": run_config,
        "mesh_canonical": binding(canonical_path),
        "mesh_view": binding(view_path),
        "mesh_stats": mesh_stats(canonical),
        "sparse_coordinate_count": int(coords.shape[0]),
        "ss_noise_shape": list(ss_noise.shape),
        "guardrails": [
            "No Direct-SS or Direct-SLAT checkpoint is loaded.",
            "Camera matrices in the cache are not passed to native ReconViaGen.",
            "Frozen cache masks provide alpha; BiRefNet is not evaluated.",
            "The canonical OBJ is geometry-only; texture is outside this comparison.",
        ],
        "code_bindings": code_bindings(
            {
                "runner": Path(__file__).resolve(),
                "pipeline": (
                    RECONVIAGEN_ROOT
                    / "trellis"
                    / "pipelines"
                    / "trellis_image_to_3d.py"
                ),
                "rgba_helper": (
                    TRACKER_ROOT
                    / "reconvggt_ar_adapter_a"
                    / "train_pointpose_ss_lora.py"
                ),
            }
        ),
    }
    atomic_json(report_path, report)
    del (
        pipeline,
        outputs,
        coords,
        ss_noise,
        decoded,
        gaussian,
        canonical,
        view_mesh,
        images,
    )
    gc.collect()
    torch.cuda.empty_cache()
    print(
        json.dumps(
            {
                "status": "complete",
                "report": str(report_path),
                "mesh": str(canonical_path),
                "mesh_sha256": sha256_file(canonical_path),
                "view_count": len(image_paths),
            },
            indent=2,
        )
    )


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_lifting_manifest", type=Path, required=True)
    parser.add_argument("--uid", required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--pretrained", default="Stable-X/trellis-vggt-v0-2")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--low_vram", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ss_steps", type=int, default=30)
    parser.add_argument("--ss_guidance", type=float, default=7.5)
    parser.add_argument("--ss_guidance_rescale", type=float, default=0.7)
    parser.add_argument("--ss_rescale_t", type=float, default=5.0)
    parser.add_argument("--slat_steps", type=int, default=12)
    parser.add_argument("--slat_guidance", type=float, default=7.5)
    parser.add_argument("--slat_guidance_rescale", type=float, default=0.5)
    parser.add_argument("--slat_rescale_t", type=float, default=3.0)
    parser.add_argument(
        "--multiimage_algo",
        choices=("multidiffusion", "stochastic"),
        default="multidiffusion",
    )
    return parser


def main() -> None:
    run(make_parser().parse_args())


if __name__ == "__main__":
    main()
