#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg_cache")
os.environ.setdefault("TORCH_HOME", os.path.expanduser("~/.cache/torch"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import imageio
import numpy as np
import torch
from PIL import Image


TRACKER_ROOT = Path(__file__).resolve().parents[2]
RECONVIAGEN_ROOT = TRACKER_ROOT / "ReconViaGen"
sys.path.insert(0, str(TRACKER_ROOT))
sys.path.insert(0, str(RECONVIAGEN_ROOT))

from rebuild_mesh_from_coarse_dataset import apply_mask_and_crop_with_meta, init_pipeline_from_local_cache  # noqa: E402
from run_local import get_candidate_seeds, run_limited_candidate_generation, save_generation_inputs  # noqa: E402
from trellis.utils import postprocessing_utils, render_utils  # noqa: E402


def resolve_path(root: str | None, path: str) -> str:
    if os.path.isabs(path) or root is None:
        return path
    return os.path.join(root, path)


def load_masked_multiview_inputs(
    manifest_path: str,
    image_root: str | None,
    mask_root: str | None,
    max_frames: int,
    resolution: int,
) -> tuple[list[Image.Image], list[str], list[int], dict]:
    payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    frames = payload.get("frames", payload if isinstance(payload, list) else None)
    if not frames:
        raise ValueError(f"No frames in manifest: {manifest_path}")
    if max_frames > 0:
        frames = frames[:max_frames]

    images = []
    names = []
    indices = []
    crop_metas = {}
    for idx, frame in enumerate(frames):
        image_name = frame["image"]
        mask_name = frame.get("mask")
        if mask_name is None:
            raise ValueError(f"ReconViaGen benchmark requires masks, missing for {image_name}")
        image_path = resolve_path(image_root or payload.get("image_root"), image_name)
        mask_path = resolve_path(mask_root or payload.get("mask_root"), mask_name)
        image, crop_meta = apply_mask_and_crop_with_meta(image_path, mask_path, resolution=resolution)
        images.append(image)
        names.append(image_name)
        indices.append(idx)
        crop_metas[image_name] = crop_meta

    return images, names, indices, {
        "manifest": manifest_path,
        "image_root": image_root or payload.get("image_root"),
        "mask_root": mask_root or payload.get("mask_root"),
        "resolution": resolution,
        "crop_metas_by_name": crop_metas,
    }


def parse_seed_list(seed_text: str | None) -> list[int]:
    if not seed_text:
        return get_candidate_seeds()
    return [int(x.strip()) for x in seed_text.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--image_root", default=None)
    parser.add_argument("--mask_root", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_frames", type=int, default=8)
    parser.add_argument("--resolution", type=int, default=518)
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--mesh_simplify", type=float, default=0.75)
    parser.add_argument("--skip_video", action="store_true")
    parser.add_argument("--trellis_model_path", default=None)
    parser.add_argument("--vggt_model_path", default=None)
    parser.add_argument("--birefnet_model_path", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_images, selected_names, selected_indices, input_info = load_masked_multiview_inputs(
        args.manifest,
        args.image_root,
        args.mask_root,
        args.max_frames,
        args.resolution,
    )
    candidate_seeds = parse_seed_list(args.seeds)

    pipeline = init_pipeline_from_local_cache(
        trellis_model_path=args.trellis_model_path,
        vggt_model_path=args.vggt_model_path,
        birefnet_model_path=args.birefnet_model_path,
    )
    input_manifest = save_generation_inputs(
        str(output_dir),
        selected_images,
        selected_names,
        selected_indices,
        selected_indices,
        {"source": "benchmark_multiview"},
        candidate_seeds,
    )
    best_candidate, candidate_report = run_limited_candidate_generation(pipeline, selected_images, candidate_seeds)
    outputs = best_candidate["outputs"]
    gs = outputs["gaussian"][0]
    mesh = outputs["mesh"][0]

    base = output_dir / "reconstructed_object"
    gs.save_ply(f"{base}.ply")
    glb = postprocessing_utils.to_glb(gs, mesh, simplify=args.mesh_simplify, texture_size=1024, verbose=False)
    glb.export(f"{base}.glb")
    mesh.to_trimesh(transform_pose=True).export(output_dir / "reconstructed_mesh.obj")

    if not args.skip_video:
        video_color = render_utils.render_video(gs, num_frames=72)["color"]
        video_geo = render_utils.render_video(mesh, num_frames=72)["normal"]
        video = [np.concatenate([video_color[i], video_geo[i]], axis=1) for i in range(len(video_color))]
        imageio.mimsave(f"{base}.mp4", video, fps=15)

    report = {
        "status": "done",
        "method": "reconviagen_multiview_direct",
        "input_info": input_info,
        "input_manifest": input_manifest,
        "selected_names": selected_names,
        "candidate_seeds": candidate_seeds,
        "selected_seed": int(best_candidate["seed"]),
        "selected_candidate": best_candidate["metrics"],
        "candidates": candidate_report,
        "outputs": {
            "glb": f"{base}.glb",
            "ply": f"{base}.ply",
            "obj": str(output_dir / "reconstructed_mesh.obj"),
            "mp4": f"{base}.mp4" if not args.skip_video else None,
        },
    }
    (output_dir / "reconviagen_multiview_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[reconviagen_multiview] done: {output_dir}")


if __name__ == "__main__":
    main()
