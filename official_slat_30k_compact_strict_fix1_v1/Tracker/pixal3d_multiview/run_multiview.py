from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("ATTN_BACKEND", "flash_attn")
os.environ.setdefault("FLEX_GEMM_AUTOTUNER_VERBOSE", "1")

TRACKER_ROOT = Path(__file__).resolve().parents[1]
PIXAL3D_ROOT = TRACKER_ROOT / "Pixal3D"
sys.path.insert(0, str(TRACKER_ROOT))
sys.path.insert(0, str(PIXAL3D_ROOT))

import o_voxel  # noqa: E402
from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import DinoV3ProjFeatureExtractor  # noqa: E402

from pixal3d_multiview.pipeline import Pixal3DMultiviewTo3DPipeline  # noqa: E402


MODEL_PATH = "TencentARC/Pixal3D"

IMAGE_COND_CONFIGS = {
    "ss": {
        "model_name": "camenduru/dinov3-vitl16-pretrain-lvd1689m",
        "image_size": 512,
        "grid_resolution": 16,
    },
    "shape_512": {
        "model_name": "camenduru/dinov3-vitl16-pretrain-lvd1689m",
        "image_size": 512,
        "grid_resolution": 32,
        "use_naf_upsample": True,
        "naf_target_size": 512,
    },
    "shape_1024": {
        "model_name": "camenduru/dinov3-vitl16-pretrain-lvd1689m",
        "image_size": 1024,
        "grid_resolution": 64,
        "use_naf_upsample": True,
        "naf_target_size": 512,
    },
    "tex_1024": {
        "model_name": "camenduru/dinov3-vitl16-pretrain-lvd1689m",
        "image_size": 1024,
        "grid_resolution": 64,
        "use_naf_upsample": True,
        "naf_target_size": 1024,
    },
}


def build_image_cond_model(config: dict):
    model = DinoV3ProjFeatureExtractor(**config)
    model.eval()
    return model


def init_pipeline(model_path: str, device: str, low_vram: bool) -> Pixal3DMultiviewTo3DPipeline:
    print(f"[Pixal3D-MV] loading pipeline: {model_path}")
    pipeline = Pixal3DMultiviewTo3DPipeline.from_pretrained(model_path)
    pipeline.image_cond_model_ss = build_image_cond_model(IMAGE_COND_CONFIGS["ss"])
    pipeline.image_cond_model_shape_512 = build_image_cond_model(IMAGE_COND_CONFIGS["shape_512"])
    pipeline.image_cond_model_shape_1024 = build_image_cond_model(IMAGE_COND_CONFIGS["shape_1024"])
    pipeline.image_cond_model_tex_1024 = build_image_cond_model(IMAGE_COND_CONFIGS["tex_1024"])

    if low_vram:
        for attr in ["image_cond_model_ss", "image_cond_model_shape_512", "image_cond_model_shape_1024", "image_cond_model_tex_1024"]:
            model = getattr(pipeline, attr)
            if getattr(model, "use_naf_upsample", False):
                model._load_naf()
        pipeline._device = torch.device(device)
        pipeline.low_vram = True
    else:
        pipeline.low_vram = False
        pipeline.cuda()
        pipeline.image_cond_model_ss.cuda()
        pipeline.image_cond_model_shape_512.cuda()
        pipeline.image_cond_model_shape_1024.cuda()
        pipeline.image_cond_model_tex_1024.cuda()
        for attr in ["image_cond_model_ss", "image_cond_model_shape_512", "image_cond_model_shape_1024", "image_cond_model_tex_1024"]:
            model = getattr(pipeline, attr)
            if getattr(model, "use_naf_upsample", False):
                model._load_naf()
    return pipeline


def _resolve(root: Optional[str], path: str) -> str:
    if os.path.isabs(path) or root is None:
        return path
    return os.path.join(root, path)


def _load_matrix(path: Optional[str]) -> Optional[torch.Tensor]:
    if not path:
        return None
    with open(path, "r") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data.get("matrix", data.get("transform", data))
    return torch.tensor(data, dtype=torch.float32)


def load_manifest(
    manifest_path: str,
    image_root: Optional[str],
    mask_root: Optional[str],
    max_frames: int,
    apply_mask: bool,
):
    with open(manifest_path, "r") as f:
        data = json.load(f)
    frames = data.get("frames", data if isinstance(data, list) else None)
    if frames is None:
        raise ValueError("manifest should contain a frames list")
    if max_frames > 0:
        frames = frames[:max_frames]

    top_intrinsic = data.get("intrinsic") if isinstance(data, dict) else None
    images, masks, intrinsics, extrinsics, source_sizes = [], [], [], [], []
    for frame in frames:
        image = Image.open(_resolve(image_root, frame["image"])).convert("RGB")
        width, height = image.size
        source_sizes.append((width, height))

        mask_path = frame.get("mask")
        if mask_path is not None:
            mask = Image.open(_resolve(mask_root, mask_path)).convert("L")
            if mask.size != image.size:
                mask = mask.resize(image.size, Image.NEAREST)
            mask_arr = np.asarray(mask).astype(np.float32) / 255.0
        else:
            mask_arr = np.ones((height, width), dtype=np.float32)

        if apply_mask:
            rgb = np.asarray(image).astype(np.float32) / 255.0
            rgb = rgb * mask_arr[..., None]
            image = Image.fromarray((np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8))

        intrinsic = frame.get("intrinsic", top_intrinsic)
        if intrinsic is None:
            raise ValueError(f"missing intrinsic for frame {frame['image']}")
        images.append(image)
        masks.append(torch.from_numpy(mask_arr[None]))
        intrinsics.append(torch.tensor(intrinsic, dtype=torch.float32))
        extrinsics.append(torch.tensor(frame["extrinsic"], dtype=torch.float32))

    extrinsics_type = data.get("extrinsics_type", "c2w") if isinstance(data, dict) else "c2w"
    return images, torch.stack(masks, dim=0), torch.stack(intrinsics, dim=0), torch.stack(extrinsics, dim=0), source_sizes, extrinsics_type


def export_obj(vertices: torch.Tensor, faces: torch.Tensor, output_path: str) -> None:
    vertices_np = vertices.detach().cpu().numpy()
    faces_np = faces.detach().cpu().numpy()
    with open(output_path, "w") as f:
        for v in vertices_np:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for face in faces_np:
            f.write(f"f {int(face[0]) + 1} {int(face[1]) + 1} {int(face[2]) + 1}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pixal3D multi-view image+pose+mask inference")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--image_root", default=None)
    parser.add_argument("--mask_root", default=None)
    parser.add_argument("--output", required=True, help="Output GLB path")
    parser.add_argument("--obj_output", default="", help="Optional geometry-only OBJ path")
    parser.add_argument("--stats_output", default="", help="Optional JSON stats path")
    parser.add_argument("--model_path", default=MODEL_PATH)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--low_vram", action="store_true")
    parser.add_argument("--max_frames", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resolution", type=int, default=-1, choices=[-1, 1024, 1536])
    parser.add_argument("--coords_source", choices=["network", "visual_hull"], default="network")
    parser.add_argument("--extrinsics_type", choices=["manifest", "c2w", "w2c"], default="manifest")
    parser.add_argument("--camera_forward_sign", type=float, default=1.0)
    parser.add_argument(
        "--object_to_world_json",
        default="",
        help="Optional debug override for the internal projection volume. Normally leave empty.",
    )
    parser.add_argument(
        "--world_to_object_json",
        default="",
        help="Optional debug override for the internal projection volume. Normally leave empty.",
    )
    parser.add_argument("--mask_threshold", type=float, default=0.5)
    parser.add_argument("--vh_min_visible_views", type=int, default=1)
    parser.add_argument("--vh_min_support_views", type=int, default=2)
    parser.add_argument("--vh_min_support_ratio", type=float, default=0.6)
    parser.add_argument("--vh_keep_solid", action="store_true")
    parser.add_argument("--vh_max_coords", type=int, default=12000)
    parser.add_argument("--vh_volume_resolution", type=int, default=48)
    parser.add_argument("--vh_volume_initial_extent_ratio", type=float, default=0.6)
    parser.add_argument("--vh_volume_padding", type=float, default=1.25)
    parser.add_argument("--vh_volume_min_extent", type=float, default=0.05)
    parser.add_argument("--vh_volume_refine_steps", type=int, default=2)
    parser.add_argument("--no_visibility_depth", action="store_true", help="Disable visual-hull front-depth visibility weighting.")
    parser.add_argument("--vh_visibility_resolution", type=int, default=48)
    parser.add_argument("--vh_visibility_dilation", type=int, default=3)
    parser.add_argument("--visibility_depth_tolerance", type=float, default=0.0, help="Absolute camera-depth tolerance. <=0 uses volume extent ratio.")
    parser.add_argument("--visibility_depth_tolerance_ratio", type=float, default=0.15)
    parser.add_argument("--visibility_weight_min", type=float, default=0.05)
    parser.add_argument("--empty_policy", choices=["zero", "visible", "border", "soft"], default="zero")
    parser.add_argument("--fallback_weight", type=float, default=1.0)
    parser.add_argument("--support_confidence_power", type=float, default=1.0)
    parser.add_argument("--global_fusion", choices=["concat", "mean", "first"], default="concat")
    parser.add_argument("--ss_steps", type=int, default=12)
    parser.add_argument("--shape_steps", type=int, default=12)
    parser.add_argument("--tex_steps", type=int, default=12)
    parser.add_argument("--ss_guidance_strength", type=float, default=7.5)
    parser.add_argument("--shape_guidance_strength", type=float, default=7.5)
    parser.add_argument("--tex_guidance_strength", type=float, default=1.0)
    parser.add_argument("--max_num_tokens", type=int, default=49152)
    parser.add_argument("--no_apply_mask", action="store_true")
    parser.add_argument("--no_export_rotation", action="store_true", help="Disable Pixal3D's default output orientation fix.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    images, masks, intrinsics, extrinsics, source_sizes, manifest_ext_type = load_manifest(
        args.manifest,
        args.image_root,
        args.mask_root,
        args.max_frames,
        apply_mask=not args.no_apply_mask,
    )
    extrinsics_type = manifest_ext_type if args.extrinsics_type == "manifest" else args.extrinsics_type
    pipeline_type = f"{args.resolution if args.resolution > 0 else (1024 if args.low_vram else 1536)}_cascade"
    object_to_world = _load_matrix(args.object_to_world_json)
    world_to_object = _load_matrix(args.world_to_object_json)

    pipeline = init_pipeline(args.model_path, args.device, args.low_vram)
    ss_sampler = {"steps": args.ss_steps, "guidance_strength": args.ss_guidance_strength, "guidance_rescale": 0.7, "rescale_t": 5.0}
    shape_sampler = {"steps": args.shape_steps, "guidance_strength": args.shape_guidance_strength, "guidance_rescale": 0.5, "rescale_t": 3.0}
    tex_sampler = {"steps": args.tex_steps, "guidance_strength": args.tex_guidance_strength, "guidance_rescale": 0.0, "rescale_t": 3.0}

    print(f"[Pixal3D-MV] views={len(images)} extrinsics_type={extrinsics_type} pipeline_type={pipeline_type} coords_source={args.coords_source}")
    mesh_list, (_, _, res) = pipeline.run_multiview(
        images,
        intrinsics,
        extrinsics,
        source_sizes,
        masks=masks,
        extrinsics_are_c2w=extrinsics_type == "c2w",
        camera_forward_sign=args.camera_forward_sign,
        object_to_world=object_to_world,
        world_to_object=world_to_object,
        coords_source=args.coords_source,
        seed=args.seed,
        sparse_structure_sampler_params=ss_sampler,
        shape_slat_sampler_params=shape_sampler,
        tex_slat_sampler_params=tex_sampler,
        return_latent=True,
        pipeline_type=pipeline_type,
        max_num_tokens=args.max_num_tokens,
        mask_threshold=args.mask_threshold,
        vh_min_visible_views=args.vh_min_visible_views,
        vh_min_support_views=args.vh_min_support_views,
        vh_min_support_ratio=args.vh_min_support_ratio,
        vh_surface_only=not args.vh_keep_solid,
        vh_max_coords=args.vh_max_coords,
        vh_volume_resolution=args.vh_volume_resolution,
        vh_volume_initial_extent_ratio=args.vh_volume_initial_extent_ratio,
        vh_volume_padding=args.vh_volume_padding,
        vh_volume_min_extent=args.vh_volume_min_extent,
        vh_volume_refine_steps=args.vh_volume_refine_steps,
        visibility_enabled=not args.no_visibility_depth,
        vh_visibility_resolution=args.vh_visibility_resolution,
        vh_visibility_dilation=args.vh_visibility_dilation,
        visibility_depth_tolerance=args.visibility_depth_tolerance,
        visibility_depth_tolerance_ratio=args.visibility_depth_tolerance_ratio,
        visibility_weight_min=args.visibility_weight_min,
        empty_policy=args.empty_policy,
        fallback_weight=args.fallback_weight,
        support_confidence_power=args.support_confidence_power,
        global_fusion=args.global_fusion,
    )
    mesh = mesh_list[0]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[Pixal3D-MV] exporting GLB: {output_path}")
    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices,
        faces=mesh.faces,
        attr_volume=mesh.attrs,
        coords=mesh.coords,
        attr_layout=pipeline.pbr_attr_layout,
        grid_size=res,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=1000000,
        texture_size=4096,
        remesh=True,
        remesh_band=1,
        remesh_project=0,
        use_tqdm=True,
    )
    if not args.no_export_rotation:
        rot = np.array(
            [[-1, 0, 0, 0], [0, 0, -1, 0], [0, -1, 0, 0], [0, 0, 0, 1]],
            dtype=np.float64,
        )
        glb.apply_transform(rot)
    glb.export(str(output_path), extension_webp=True)

    if args.obj_output:
        obj_path = Path(args.obj_output)
        obj_path.parent.mkdir(parents=True, exist_ok=True)
        export_obj(mesh.vertices, mesh.faces, str(obj_path))
        print(f"[Pixal3D-MV] geometry OBJ: {obj_path}")

    stats_path = Path(args.stats_output) if args.stats_output else output_path.with_suffix(".stats.json")
    with open(stats_path, "w") as f:
        json.dump(pipeline.last_multiview_stats, f, indent=2)
    print(f"[Pixal3D-MV] stats: {stats_path}")


if __name__ == "__main__":
    main()
