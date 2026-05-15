#!/usr/bin/env python3

"""Standalone template generation for meshes produced by ReconViaGen.

This file is intentionally independent from ``gen_templates.py``.  It reuses
the shared renderer/config/data utilities, but it does not import or call the
old template-generation script.
"""

import argparse
import json
import logging as py_logging
import os
import sys
import types
from pathlib import Path
from typing import List, NamedTuple, Optional, Tuple, Union


CORE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CORE_DIR.parent
PathLike = Union[str, os.PathLike]


def _install_png_fallback() -> None:
    try:
        import png  # noqa: F401
        return
    except ModuleNotFoundError:
        pass

    class _Writer:
        def __init__(self, width, height, greyscale=True, bitdepth=16):
            self.width = width
            self.height = height
            self.bitdepth = bitdepth

        def write(self, file_obj, rows):
            import numpy as np
            from PIL import Image

            dtype = np.uint16 if self.bitdepth == 16 else np.uint8
            arr = np.asarray(list(rows), dtype=dtype).reshape(self.height, self.width)
            Image.fromarray(arr).save(file_obj, format="PNG")

    png_module = types.ModuleType("png")
    png_module.Writer = _Writer
    sys.modules["png"] = png_module


def _prepare_import_path() -> None:
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    for path in (str(CORE_DIR), str(PROJECT_ROOT), str(PROJECT_ROOT / "external" / "dinov2")):
        if path not in sys.path:
            sys.path.insert(0, path)
    _install_png_fallback()


_prepare_import_path()

import cv2
import numpy as np
import trimesh

import inout
from config import AppConfig
from utils import config_util, geometry, json_util, logging, misc, renderer_builder
from utils.misc import warp_depth_image, warp_image
from utils.renderer_base import RenderType
from utils.structs import AlignedBox2f, PinholePlaneCameraModel, RigidTransform


class GenTemplateAutoOpts(NamedTuple):
    version: str
    object_dataset: str
    object_lids: Optional[List[int]] = None
    num_viewspheres: int = 1
    min_num_viewpoints: int = 57
    num_inplane_rotations: int = 14
    images_per_view: int = 1
    max_num_triangles: int = 20000
    back_face_culling: bool = False
    texture_size: Tuple[int, int] = (1024, 1024)
    ssaa_factor: float = 1.0
    background_type: str = "black"
    light_type: str = "multi_directional"
    crop: bool = True
    crop_rel_pad: float = 0.2
    crop_size: Tuple[int, int] = (420, 420)
    features_patch_size: int = 14
    save_templates: bool = True
    overwrite: bool = True
    debug: bool = True


def default_template_config(dataset_name: str) -> dict:
    return {
        "gen_template_auto_opts": {
            "version": "v1",
            "object_dataset": dataset_name,
            "object_lids": [1],
            "num_viewspheres": 1,
            "min_num_viewpoints": 57,
            "num_inplane_rotations": 14,
            "images_per_view": 1,
            "max_num_triangles": 20000,
            "back_face_culling": False,
            "texture_size": [1024, 1024],
            "ssaa_factor": 1.0,
            "background_type": "black",
            "light_type": "multi_directional",
            "crop": True,
            "crop_rel_pad": 0.2,
            "crop_size": [420, 420],
            "features_patch_size": 14,
            "save_templates": True,
            "overwrite": True,
            "debug": True,
        }
    }


def write_template_config(dataset_name: str, config_path: Optional[PathLike] = None) -> Path:
    if config_path is None:
        config_path = PROJECT_ROOT / "configs" / "gen_templates" / f"{dataset_name}.json"
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(default_template_config(dataset_name), f, indent=4)
    return config_path


def normalize_mesh_scale(mesh: trimesh.Trimesh, target_max_extent: float = 200.0) -> Tuple[trimesh.Trimesh, float]:
    max_extent = float(np.max(mesh.bounding_box.extents))
    if max_extent < 1e-6:
        raise ValueError("Mesh size is too small or degenerate")
    if 0.1 <= max_extent <= 1000.0:
        return mesh, 1.0
    scale = target_max_extent / max_extent
    mesh.apply_scale(scale)
    return mesh, scale


def _make_base_camera(opts: GenTemplateAutoOpts) -> Tuple[PinholePlaneCameraModel, PinholePlaneCameraModel]:
    width = 720
    height = 1280
    focal_length = 1087.8726353191628
    px = 360.0
    py = 640.0
    max_image_side = max(width, height)
    image_side = opts.features_patch_size * int(max_image_side / opts.features_patch_size)
    camera = PinholePlaneCameraModel(
        width=image_side,
        height=image_side,
        f=(focal_length, focal_length),
        c=(px - 0.5 * (width - image_side), py - 0.5 * (height - image_side)),
    )
    render_camera = PinholePlaneCameraModel(
        width=int(camera.width * opts.ssaa_factor),
        height=int(camera.height * opts.ssaa_factor),
        f=(camera.f[0] * opts.ssaa_factor, camera.f[1] * opts.ssaa_factor),
        c=(camera.c[0] * opts.ssaa_factor, camera.c[1] * opts.ssaa_factor),
    )
    return camera, render_camera


def _sample_views(mesh: trimesh.Trimesh, opts: GenTemplateAutoOpts) -> list:
    width = 720
    height = 1280
    focal = 1087.8726353191628
    fov_y = 2 * np.arctan2(height / 2, focal)
    fov_x = 2 * np.arctan2(width / 2, focal)
    model_diag = np.linalg.norm(mesh.bounding_box.extents)
    depth_min = model_diag / (2 * np.tan(min(fov_x, fov_y) / 2)) * 1.2
    depth_max = model_diag / (2 * np.tan(min(fov_x, fov_y) / 2)) * 1.5
    depth_cell_size = (depth_max - depth_min) / float(opts.num_viewspheres)

    views_sphere = []
    for depth_cell_id in range(opts.num_viewspheres):
        radius = depth_min + (depth_cell_id + 0.5) * depth_cell_size
        views_sphere += misc.sample_views(min_n_views=opts.min_num_viewpoints, radius=radius, mode="fibonacci")[0]

    if opts.num_inplane_rotations == 1:
        return views_sphere

    views = []
    inplane_angle = 2 * np.pi / opts.num_inplane_rotations
    for view_sphere in views_sphere:
        for inplane_id in range(opts.num_inplane_rotations):
            R_inplane = geometry.rotation_matrix_numpy(inplane_angle * inplane_id, np.array([0, 0, 1]))[:3, :3]
            views.append({"R": R_inplane.dot(view_sphere["R"]), "t": R_inplane.dot(view_sphere["t"])})
    return views


def _resize_outputs(output: dict, camera_model: PinholePlaneCameraModel) -> None:
    target_size = (camera_model.width, camera_model.height)
    for key in list(output.keys()):
        interpolation = cv2.INTER_AREA if key == RenderType.COLOR else cv2.INTER_NEAREST
        output[key] = misc.resize_image(image=output[key], size=target_size, interpolation=interpolation)


def synthesize_templates(opts: GenTemplateAutoOpts) -> None:
    np.random.seed(0)
    logger = logging.get_logger(level=logging.INFO if opts.debug else logging.WARNING)
    logger.setLevel(py_logging.WARNING)
    object_lids = opts.object_lids or [1]
    print(f"[gen_template_auto] Loading mesh for dataset={opts.object_dataset}", flush=True)
    camera_model, render_camera_model = _make_base_camera(opts)
    renderer = renderer_builder.build(renderer_type=renderer_builder.RendererType.PYRENDER_RASTERIZER)

    dataset_model_dir = AppConfig.DATASETS_PATH / opts.object_dataset / "models"
    raw_model_path = dataset_model_dir / f"{opts.object_dataset}.obj"
    norm_model_path = dataset_model_dir / f"{opts.object_dataset}_norm.obj"
    if not raw_model_path.exists():
        raise FileNotFoundError(f"Model not found: {raw_model_path}")

    mesh = trimesh.load(raw_model_path, force="mesh")
    mesh, _ = normalize_mesh_scale(mesh)
    mesh.export(norm_model_path)
    views = _sample_views(mesh, opts)
    total_templates = len(views) * opts.images_per_view
    print(f"[gen_template_auto] Rendering {total_templates} templates to results/templates/{opts.version}/{opts.object_dataset}", flush=True)

    for object_lid in object_lids:
        logging.log_heading(logger, f"Object {object_lid} from {opts.object_dataset}")
        output_dir = AppConfig.OUTPUT_ROOT / "templates" / opts.version / opts.object_dataset / str(object_lid)
        if output_dir.exists() and not opts.overwrite:
            raise ValueError(f"Output directory already exists: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)
        json_util.save_json(output_dir / "config.json", opts)

        rgb_dir = output_dir / "rgb"
        depth_dir = output_dir / "depth"
        mask_dir = output_dir / "mask"
        for path in (rgb_dir, depth_dir, mask_dir):
            path.mkdir(parents=True, exist_ok=True)

        renderer.add_object_model(obj_id=object_lid, model_path=str(norm_model_path), debug=True)
        metadata = []
        template_counter = 0

        for view_id, view in enumerate(views):
            for _ in range(opts.images_per_view):
                trans_m2c = RigidTransform(R=view["R"], t=view["t"])
                R_c2m = trans_m2c.R.T
                trans_c2m = RigidTransform(R=R_c2m, t=-R_c2m.dot(trans_m2c.t))
                render_camera_c2w = PinholePlaneCameraModel(
                    width=render_camera_model.width,
                    height=render_camera_model.height,
                    f=render_camera_model.f,
                    c=render_camera_model.c,
                    T_world_from_eye=misc.get_rigid_matrix(trans_c2m),
                )

                output = renderer.render_object_model(
                    obj_id=object_lid,
                    camera_model_c2w=render_camera_c2w,
                    render_types=[RenderType.COLOR, RenderType.DEPTH, RenderType.MASK],
                    return_tensors=False,
                    debug=False,
                )
                output[RenderType.MASK] = (255 * output[RenderType.MASK]).astype(np.uint8)
                ys, xs = output[RenderType.MASK].nonzero()
                if len(xs) == 0:
                    continue
                box = np.array(misc.calc_2d_box(xs, ys))
                object_box = AlignedBox2f(left=box[0], top=box[1], right=box[2], bottom=box[3])

                if opts.crop:
                    crop_box = misc.calc_crop_box(box=object_box, make_square=True)
                    crop_camera = misc.construct_crop_camera(
                        box=crop_box,
                        camera_model_c2w=render_camera_c2w,
                        viewport_size=(int(opts.crop_size[0] * opts.ssaa_factor), int(opts.crop_size[1] * opts.ssaa_factor)),
                        viewport_rel_pad=opts.crop_rel_pad,
                    )
                    for key in list(output.keys()):
                        if key == RenderType.DEPTH:
                            output[key] = warp_depth_image(render_camera_c2w, crop_camera, output[key])
                        elif key == RenderType.COLOR:
                            interp = cv2.INTER_AREA if crop_box.width >= crop_camera.width else cv2.INTER_LINEAR
                            output[key] = warp_image(render_camera_c2w, crop_camera, output[key], interpolation=interp)
                        else:
                            output[key] = warp_image(render_camera_c2w, crop_camera, output[key], interpolation=cv2.INTER_NEAREST)
                    final_camera = crop_camera.copy()
                    scale_factor = opts.crop_size[0] / float(crop_camera.width)
                    final_camera.width = opts.crop_size[0]
                    final_camera.height = opts.crop_size[1]
                    final_camera.c = (final_camera.c[0] * scale_factor, final_camera.c[1] * scale_factor)
                    final_camera.f = (final_camera.f[0] * scale_factor, final_camera.f[1] * scale_factor)
                else:
                    final_camera = camera_model.copy()
                    final_camera.T_world_from_eye = misc.get_rigid_matrix(trans_c2m)

                if opts.ssaa_factor != 1.0:
                    _resize_outputs(output, final_camera)

                ys, xs = output[RenderType.MASK].nonzero()
                box = np.array(misc.calc_2d_box(xs, ys))
                object_box = AlignedBox2f(left=box[0], top=box[1], right=box[2], bottom=box[3])
                color = np.clip(output[RenderType.COLOR], 0, 1)
                rgb = (np.power(color, 1 / 2.2) * 255).astype(np.uint8)

                rgb_path = rgb_dir / f"template_{template_counter:04d}.png"
                depth_path = depth_dir / f"template_{template_counter:04d}.png"
                mask_path = mask_dir / f"template_{template_counter:04d}.png"
                inout.save_im(rgb_path, rgb)
                inout.save_depth(depth_path, output[RenderType.DEPTH])
                inout.save_im(mask_path, output[RenderType.MASK])

                metadata.append({
                    "dataset": opts.object_dataset,
                    "lid": object_lid,
                    "template_id": template_counter,
                    "pose": RigidTransform(R=np.eye(3), t=np.zeros((3, 1))),
                    "boxes_amodal": np.array([object_box.array_ltrb()]).tolist(),
                    "visibilities": np.array([1.0]).tolist(),
                    "cameras": final_camera.to_json(),
                    "rgb_image_path": str(rgb_path),
                    "depth_map_path": str(depth_path),
                    "binary_mask_path": str(mask_path),
                })
                template_counter += 1
                if template_counter == 1 or template_counter % 25 == 0 or template_counter == total_templates:
                    print(
                        f"\r[gen_template_auto] Rendered {template_counter}/{total_templates} templates",
                        end="",
                        flush=True,
                    )

        if template_counter:
            print("", flush=True)
        json_util.save_json(output_dir / "metadata.json", metadata)
        print(f"[gen_template_auto] Saved metadata with {len(metadata)} templates: {output_dir / 'metadata.json'}", flush=True)


def generate_templates_for_dataset(dataset_name: str, config_path: Optional[PathLike] = None) -> Path:
    config_path = write_template_config(dataset_name, config_path)
    opts = config_util.load_opts_from_json(
        path=str(config_path),
        opts_types={"gen_template_auto_opts": GenTemplateAutoOpts},
    )["gen_template_auto_opts"]
    synthesize_templates(opts)
    return config_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--config-path", default=None)
    args = parser.parse_args()
    generate_templates_for_dataset(args.dataset_name, args.config_path)


if __name__ == "__main__":
    main()
