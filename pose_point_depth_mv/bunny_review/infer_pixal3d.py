#!/usr/bin/env python3
"""Run original single-view Pixal3D on one frozen Bunny RGBA view."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image

from .common import (
    binding,
    code_bindings,
    load_method_result,
    load_protocol,
    method_dir,
    sha256_file,
    write_method_result,
)


TRACKER_ROOT = Path(__file__).resolve().parents[2]
PIXAL3D_ROOT = TRACKER_ROOT / "Pixal3D"


def export_obj_atomic(mesh, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    mesh.export(temporary, file_type="obj")
    os.replace(temporary, destination)


def run(args: argparse.Namespace) -> None:
    protocol_path = args.protocol.resolve()
    protocol = load_protocol(protocol_path)
    destination_dir = method_dir(protocol_path, args.method_id)
    result_path = destination_dir / "result.json"
    if result_path.is_file():
        result = load_method_result(protocol_path, args.method_id)
        print(
            json.dumps(
                {
                    "status": "reused",
                    "method_id": args.method_id,
                    "mesh": result["mesh"]["path"],
                },
                indent=2,
            )
        )
        return
    if destination_dir.exists() and any(destination_dir.iterdir()):
        raise RuntimeError(
            f"partial Pixal3D output exists; preserve and inspect: {destination_dir}"
        )
    destination_dir.mkdir(parents=True, exist_ok=True)

    selected_index = int(protocol["single_view_index"])
    selected = next(
        view for view in protocol["views"] if int(view["view_index"]) == selected_index
    )
    image_path = Path(selected["rgba"]["path"])
    with Image.open(image_path) as candidate:
        if candidate.mode != "RGBA":
            raise RuntimeError(f"frozen Pixal3D input is not RGBA: {image_path}")
        alpha = np.asarray(candidate.getchannel("A"))
        if not np.any(alpha == 0) or not np.any(alpha == 255):
            raise RuntimeError("frozen RGBA must contain background and foreground")

    if str(PIXAL3D_ROOT) not in sys.path:
        sys.path.insert(0, str(PIXAL3D_ROOT))
    from pose_point_depth_mv.compare_pixal3d_singleview_smoke import (
        configure_local_naf,
    )
    from inference import distance_from_fov, init_pipeline  # type: ignore
    import torch
    import trimesh

    naf = configure_local_naf(
        args.naf_repo,
        args.naf_checkpoint,
        args.naf_source_manifest,
    )
    pipeline = init_pipeline(
        model_path=args.model_path,
        device=args.device,
        low_vram=bool(args.low_vram),
        load_rembg=False,
    )
    image = Image.open(image_path)
    preprocessed = pipeline.preprocess_image(image)
    camera_angle_x = float(args.fixed_fov)
    distance = distance_from_fov(
        camera_angle_x,
        torch.tensor([-1.0, 0.0, 0.0]),
        torch.tensor([0.0, 511.0]),
        float(args.mesh_scale),
        512,
    )["distance_from_x"]
    camera_params = {
        "camera_angle_x": camera_angle_x,
        "distance": distance,
        "mesh_scale": float(args.mesh_scale),
    }
    sampler_ss = {
        "steps": int(args.sampling_steps),
        "guidance_strength": float(args.ss_guidance),
        "guidance_rescale": float(args.ss_guidance_rescale),
        "rescale_t": float(args.ss_rescale_t),
    }
    sampler_shape = {
        "steps": int(args.sampling_steps),
        "guidance_strength": float(args.shape_guidance),
        "guidance_rescale": float(args.shape_guidance_rescale),
        "rescale_t": float(args.shape_rescale_t),
    }
    sampler_texture = {
        "steps": int(args.sampling_steps),
        "guidance_strength": float(args.texture_guidance),
        "guidance_rescale": float(args.texture_guidance_rescale),
        "rescale_t": float(args.texture_rescale_t),
    }
    seed = int(args.seed)
    torch.manual_seed(seed)
    print(
        f"[bunny_pixal3d] view={selected_index} seed={seed} "
        f"fov={camera_angle_x:.6f}",
        flush=True,
    )
    mesh_list, latent = pipeline.run(
        preprocessed,
        camera_params=camera_params,
        seed=seed,
        sparse_structure_sampler_params=sampler_ss,
        shape_slat_sampler_params=sampler_shape,
        tex_slat_sampler_params=sampler_texture,
        preprocess_image=False,
        return_latent=True,
        pipeline_type=f"{int(args.resolution)}_cascade",
        max_num_tokens=int(args.max_num_tokens),
    )
    decoded = mesh_list[0]
    vertices = (
        decoded.vertices.detach().float().cpu().numpy()
        if torch.is_tensor(decoded.vertices)
        else np.asarray(decoded.vertices)
    )
    faces = (
        decoded.faces.detach().cpu().numpy()
        if torch.is_tensor(decoded.faces)
        else np.asarray(decoded.faces)
    )
    mesh = trimesh.Trimesh(
        vertices=np.asarray(vertices, dtype=np.float64),
        faces=np.asarray(faces, dtype=np.int64),
        process=False,
    )
    official_rotation = np.array(
        [
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    mesh.apply_transform(official_rotation)
    obj_path = destination_dir / "mesh.obj"
    export_obj_atomic(mesh, obj_path)

    auxiliary: dict[str, Path] = {}
    if not args.skip_textured_glb:
        import o_voxel

        if not isinstance(latent, tuple) or len(latent) != 3:
            raise RuntimeError("unexpected Pixal3D shape/texture latent contract")
        _, _, grid_resolution = latent
        glb = o_voxel.postprocess.to_glb(
            vertices=decoded.vertices,
            faces=decoded.faces,
            attr_volume=decoded.attrs,
            coords=decoded.coords,
            attr_layout=pipeline.pbr_attr_layout,
            grid_size=grid_resolution,
            aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            decimation_target=int(args.decimation_target),
            texture_size=int(args.texture_size),
            remesh=True,
            remesh_band=1,
            remesh_project=0,
            use_tqdm=True,
        )
        glb.apply_transform(official_rotation)
        glb_path = destination_dir / "mesh_textured.glb"
        temporary_glb = glb_path.with_name(
            f".{glb_path.stem}.tmp-{os.getpid()}{glb_path.suffix}"
        )
        glb.export(temporary_glb, extension_webp=True)
        os.replace(temporary_glb, glb_path)
        auxiliary["textured_glb"] = glb_path
        del glb

    inference = {
        "kind": "pixal3d_original_single_view",
        "model_path": str(args.model_path),
        "seed": seed,
        "input_view_index": selected_index,
        "background_removal": "disabled; frozen border-connected RGBA used",
        "camera_policy": "fixed_fov because source thumbnail calibration is unavailable",
        "camera_params": camera_params,
        "preprocessed_size": list(preprocessed.size),
        "pipeline_type": f"{int(args.resolution)}_cascade",
        "sampling_steps": int(args.sampling_steps),
        "vertex_count": int(len(mesh.vertices)),
        "face_count": int(len(mesh.faces)),
        "naf": naf,
        "code_bindings": code_bindings(
            {
                "runner": Path(__file__).resolve(),
                "pixal_inference": PIXAL3D_ROOT / "inference.py",
                "pixal_pipeline": (
                    PIXAL3D_ROOT
                    / "pixal3d"
                    / "pipelines"
                    / "pixal3d_image_to_3d.py"
                ),
            }
        ),
    }
    result_path = write_method_result(
        protocol_path=protocol_path,
        method_id=args.method_id,
        display_name=args.display_name,
        mesh_path=obj_path,
        auxiliary_meshes=auxiliary,
        input_view_indices=[selected_index],
        backend=inference,
        notes=[
            "Single-view baseline; it intentionally receives only one frozen thumbnail.",
            "OBJ is the decoded geometry; optional GLB includes Pixal3D PBR postprocessing.",
        ],
    )
    del mesh, decoded, mesh_list, latent, pipeline
    gc.collect()
    torch.cuda.empty_cache()
    print(
        json.dumps(
            {
                "status": "complete",
                "result": str(result_path),
                "mesh": str(obj_path),
                "mesh_sha256": sha256_file(obj_path),
                "auxiliary": {key: str(path) for key, path in auxiliary.items()},
            },
            indent=2,
        )
    )


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--method_id", default="pixal3d")
    parser.add_argument("--display_name", default="Pixal3D (single view)")
    parser.add_argument("--model_path", default="TencentARC/Pixal3D")
    parser.add_argument("--naf_repo", type=Path, required=True)
    parser.add_argument("--naf_checkpoint", type=Path, required=True)
    parser.add_argument("--naf_source_manifest", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--low_vram", action="store_true")
    parser.add_argument("--fixed_fov", type=float, default=0.857556)
    parser.add_argument("--mesh_scale", type=float, default=1.0)
    parser.add_argument("--resolution", type=int, choices=(1024, 1536), default=1024)
    parser.add_argument("--max_num_tokens", type=int, default=49152)
    parser.add_argument("--sampling_steps", type=int, default=12)
    parser.add_argument("--ss_guidance", type=float, default=7.5)
    parser.add_argument("--ss_guidance_rescale", type=float, default=0.7)
    parser.add_argument("--ss_rescale_t", type=float, default=5.0)
    parser.add_argument("--shape_guidance", type=float, default=7.5)
    parser.add_argument("--shape_guidance_rescale", type=float, default=0.5)
    parser.add_argument("--shape_rescale_t", type=float, default=3.0)
    parser.add_argument("--texture_guidance", type=float, default=1.0)
    parser.add_argument("--texture_guidance_rescale", type=float, default=0.0)
    parser.add_argument("--texture_rescale_t", type=float, default=3.0)
    parser.add_argument("--skip_textured_glb", action="store_true")
    parser.add_argument("--decimation_target", type=int, default=300000)
    parser.add_argument("--texture_size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    run(make_parser().parse_args())


if __name__ == "__main__":
    main()
