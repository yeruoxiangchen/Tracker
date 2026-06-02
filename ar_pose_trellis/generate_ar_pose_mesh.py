from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

import imageio
import numpy as np
import torch
from PIL import Image

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("TORCH_EXTENSIONS_DIR", "/tmp/torch_extensions")
os.environ["PATH"] = f"{Path(sys.executable).resolve().parent}:{os.environ.get('PATH', '')}"

TRACKER_ROOT = Path(__file__).resolve().parents[1]
RECONVIAGEN_ROOT = TRACKER_ROOT / "ReconViaGen"
VGGT_WHEEL_ROOT = RECONVIAGEN_ROOT / "wheels" / "vggt"
sys.path.insert(0, str(TRACKER_ROOT))
sys.path.insert(0, str(RECONVIAGEN_ROOT))
sys.path.insert(0, str(VGGT_WHEEL_ROOT))

from ar_pose_trellis.pipeline import TrellisARPoseTo3DPipeline
from ar_pose_trellis.visual_hull import visual_hull_logit_bias
from trellis.utils import postprocessing_utils, render_utils


def _resolve_path(root: Optional[str], path: str) -> str:
    if os.path.isabs(path) or root is None:
        return path
    return os.path.join(root, path)


def load_manifest(manifest_path: str, image_root: Optional[str], mask_root: Optional[str], max_frames: int):
    with open(manifest_path, "r") as f:
        data = json.load(f)
    frames = data.get("frames", data if isinstance(data, list) else None)
    if frames is None:
        raise ValueError("camera manifest should contain a 'frames' list")
    if max_frames > 0:
        frames = frames[:max_frames]

    top_k = data.get("intrinsic") if isinstance(data, dict) else None
    images, masks, intrinsics, extrinsics = [], [], [], []
    has_masks = False
    for frame in frames:
        image_path = _resolve_path(image_root, frame["image"])
        rgba = Image.open(image_path).convert("RGBA")
        arr = np.asarray(rgba).astype(np.float32) / 255.0
        images.append(torch.from_numpy(arr[..., :3]).permute(2, 0, 1))

        mask_path = frame.get("mask")
        if mask_path is not None:
            mask = np.asarray(Image.open(_resolve_path(mask_root, mask_path)).convert("L")).astype(np.float32) / 255.0
            masks.append(torch.from_numpy(mask[None]))
            has_masks = True
        else:
            masks.append(torch.from_numpy(arr[..., 3:4]).permute(2, 0, 1))
            has_masks = has_masks or bool((arr[..., 3] < 0.999).any())

        k = frame.get("intrinsic", top_k)
        if k is None:
            raise ValueError(f"No intrinsic for frame {frame['image']}")
        intrinsics.append(torch.tensor(k, dtype=torch.float32))
        extrinsics.append(torch.tensor(frame["extrinsic"], dtype=torch.float32))

    extr_type = data.get("extrinsics_type", "c2w") if isinstance(data, dict) else "c2w"
    return (
        torch.stack(images, dim=0),
        torch.stack(masks, dim=0) if has_masks else None,
        torch.stack(intrinsics, dim=0),
        torch.stack(extrinsics, dim=0),
        extr_type,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="microsoft/TRELLIS-image-large")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--image_root", default=None)
    parser.add_argument("--mask_root", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_frames", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ss_steps", type=int, default=12)
    parser.add_argument("--slat_steps", type=int, default=12)
    parser.add_argument("--ss_guidance_strength", type=float, default=7.5)
    parser.add_argument("--slat_guidance_strength", type=float, default=3.0)
    parser.add_argument("--ss_threshold", type=float, default=0.0)
    parser.add_argument(
        "--ss_min_coords",
        type=int,
        default=4096,
        help="If sparse occupancy thresholding produces too few coords, keep this many top-logit voxels.",
    )
    parser.add_argument("--mesh_simplify", type=float, default=0.75)
    parser.add_argument("--texture_size", type=int, default=1024)
    parser.add_argument("--skip_glb", action="store_true", help="Skip GLB export and mesh postprocessing.")
    parser.add_argument("--skip_preview", action="store_true", help="Skip preview MP4 rendering.")
    parser.add_argument("--skip_gs_preview", action="store_true", help="Skip Gaussian color preview MP4 rendering.")
    parser.add_argument("--preview_frames", type=int, default=72)
    parser.add_argument("--preview_resolution", type=int, default=320)
    parser.add_argument("--preview_fps", type=int, default=15)
    parser.add_argument("--preview_side_by_side", action="store_true", help="Also render gaussian+mesh side-by-side preview.mp4.")
    parser.add_argument("--only_sparse", action="store_true", help="Only sample sparse structure diagnostics.")
    parser.add_argument("--no_crop", action="store_true")
    parser.add_argument("--pose_only", action="store_true")
    parser.add_argument("--image_only", action="store_true")
    parser.add_argument("--cond_fp16", action="store_true", help="Run AR pose condition in fp16; required for flash-attn.")
    parser.add_argument("--extrinsics_type", choices=["manifest", "c2w", "w2c"], default="manifest")
    parser.add_argument("--camera_forward_sign", type=float, default=1.0)
    parser.add_argument(
        "--absolute_pose_condition",
        action="store_true",
        help="Use absolute input camera poses in the pose condition. Default is reference-relative poses.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--visual_hull_prior_weight", type=float, default=0.0)
    parser.add_argument("--visual_hull_mask_threshold", type=float, default=0.5)
    parser.add_argument("--visual_hull_min_visible_views", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.pose_only and args.image_only:
        raise ValueError("--pose_only and --image_only are mutually exclusive.")
    os.makedirs(args.output_dir, exist_ok=True)
    images, masks, intrinsics, extrinsics, manifest_extr_type = load_manifest(
        args.manifest, args.image_root, args.mask_root, args.max_frames
    )
    extr_type = manifest_extr_type if args.extrinsics_type == "manifest" else args.extrinsics_type
    print(f"[ARPoseGenerate] frames={images.shape[0]} extrinsics_type={extr_type}")

    pipeline = TrellisARPoseTo3DPipeline.from_pretrained(
        args.weights,
        checkpoint_path=args.checkpoint,
        device=args.device,
        use_image_features=not args.pose_only,
        use_pose_features=not args.image_only,
        cond_fp16=args.cond_fp16,
        apply_lora=True,
    )
    if args.only_sparse:
        images_pre, masks_pre, intrinsics_pre = pipeline.prepare_inputs(
            images,
            intrinsics=intrinsics,
            masks=masks,
            crop_foreground=not args.no_crop,
            no_background=True,
        )
        torch.manual_seed(args.seed)
        ss_cond = pipeline.encode_ss_condition(
            images_pre,
            intrinsics_pre,
            extrinsics.to(pipeline.device).float(),
            masks=masks_pre,
            extrinsics_are_c2w=extr_type == "c2w",
            camera_forward_sign=args.camera_forward_sign,
            reference_relative_pose=not args.absolute_pose_condition,
        )
        logit_prior = None
        logit_prior_stats = None
        if float(args.visual_hull_prior_weight) != 0.0:
            logit_prior, logit_prior_stats = visual_hull_logit_bias(
                masks_pre if masks_pre is not None else torch.ones(
                    (images_pre.shape[0], 1, images_pre.shape[-2], images_pre.shape[-1]),
                    device=images_pre.device,
                    dtype=images_pre.dtype,
                ),
                intrinsics_pre,
                extrinsics.to(pipeline.device).float(),
                extrinsics_are_c2w=extr_type == "c2w",
                resolution=pipeline.sparse_logit_resolution,
                mask_threshold=args.visual_hull_mask_threshold,
                min_visible_views=args.visual_hull_min_visible_views,
                weight=args.visual_hull_prior_weight,
            )
        coords = pipeline.sample_sparse_structure(
            ss_cond,
            num_samples=1,
            sampler_params={"steps": args.ss_steps, "cfg_strength": args.ss_guidance_strength},
            threshold=args.ss_threshold,
            min_coords=args.ss_min_coords,
            logit_prior=logit_prior,
            logit_prior_stats=logit_prior_stats,
        )
        torch.save(coords.detach().cpu(), os.path.join(args.output_dir, "coords.pt"))
        with open(os.path.join(args.output_dir, "sparse_stats.json"), "w") as f:
            json.dump(getattr(pipeline, "last_sparse_stats", {}), f, indent=2)
        print(f"[ARPoseGenerate] sparse diagnostics done: {args.output_dir}")
        return

    outputs, coords = pipeline.run(
        images,
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        masks=masks,
        seed=args.seed,
        sparse_structure_sampler_params={"steps": args.ss_steps, "cfg_strength": args.ss_guidance_strength},
        slat_sampler_params={"steps": args.slat_steps, "cfg_strength": args.slat_guidance_strength},
        formats=["gaussian", "mesh"],
        crop_foreground=not args.no_crop,
        extrinsics_are_c2w=extr_type == "c2w",
        camera_forward_sign=args.camera_forward_sign,
        reference_relative_pose=not args.absolute_pose_condition,
        sparse_threshold=args.ss_threshold,
        min_sparse_coords=args.ss_min_coords,
        visual_hull_prior_weight=args.visual_hull_prior_weight,
        visual_hull_mask_threshold=args.visual_hull_mask_threshold,
        visual_hull_min_visible_views=args.visual_hull_min_visible_views,
    )

    torch.save(coords.detach().cpu(), os.path.join(args.output_dir, "coords.pt"))
    with open(os.path.join(args.output_dir, "sparse_stats.json"), "w") as f:
        json.dump(getattr(pipeline, "last_sparse_stats", {}), f, indent=2)
    gs = outputs["gaussian"][0]
    mesh = outputs["mesh"][0]
    ply_path = os.path.join(args.output_dir, "reconstructed_object.ply")
    gs_ply_path = os.path.join(args.output_dir, "gaussian_point_cloud.ply")
    obj_path = os.path.join(args.output_dir, "reconstructed_mesh.obj")
    glb_path = os.path.join(args.output_dir, "reconstructed_object.glb")
    mp4_path = os.path.join(args.output_dir, "preview.mp4")
    mesh_mp4_path = os.path.join(args.output_dir, "mesh_preview.mp4")
    mesh_normal_mp4_path = os.path.join(args.output_dir, "mesh_normal_preview.mp4")
    gs_mp4_path = os.path.join(args.output_dir, "gaussian_preview.mp4")

    gs.save_ply(ply_path)
    gs.save_ply(gs_ply_path)
    mesh.to_trimesh(transform_pose=True).export(obj_path)
    print(f"[ARPoseGenerate] gaussian point cloud: {gs_ply_path}")
    print(f"[ARPoseGenerate] mesh obj: {obj_path}")
    if not args.skip_glb:
        glb = postprocessing_utils.to_glb(
            gs,
            mesh,
            simplify=args.mesh_simplify,
            texture_size=args.texture_size,
            verbose=False,
        )
        glb.export(glb_path)
    if not args.skip_preview:
        mesh_video = render_utils.render_video(
            mesh,
            resolution=args.preview_resolution,
            ssaa=2,
            num_frames=args.preview_frames,
            pitch=0,
            inverse_direction=True,
        )
        video_geo = mesh_video["normal"]
        video_mesh_color = mesh_video.get("color", video_geo)
        imageio.mimsave(mesh_mp4_path, video_mesh_color, fps=args.preview_fps)
        imageio.mimsave(mesh_normal_mp4_path, video_geo, fps=args.preview_fps)
        print(f"[ARPoseGenerate] mesh color preview: {mesh_mp4_path}")
        print(f"[ARPoseGenerate] mesh normal preview: {mesh_normal_mp4_path}")

        if not args.skip_gs_preview:
            video_color = render_utils.render_video(
                gs,
                resolution=args.preview_resolution,
                ssaa=1,
                num_frames=args.preview_frames,
                pitch=0,
                inverse_direction=True,
            )["color"]
            imageio.mimsave(gs_mp4_path, video_color, fps=args.preview_fps)
            print(f"[ARPoseGenerate] gaussian color preview: {gs_mp4_path}")

        if args.preview_side_by_side:
            if args.skip_gs_preview:
                video_color = render_utils.render_video(
                    gs,
                    resolution=args.preview_resolution,
                    ssaa=1,
                    num_frames=args.preview_frames,
                    pitch=0,
                    inverse_direction=True,
                )["color"]
            video = [np.concatenate([video_color[i], video_geo[i]], axis=1) for i in range(len(video_color))]
            imageio.mimsave(mp4_path, video, fps=args.preview_fps)
            print(f"[ARPoseGenerate] side-by-side preview: {mp4_path}")
    print(f"[ARPoseGenerate] done: {args.output_dir}")


if __name__ == "__main__":
    main()
