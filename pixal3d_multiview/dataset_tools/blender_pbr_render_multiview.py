#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Matrix


PIXAL3D_ROTATION_4X4 = Matrix(
    (
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, -1.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )
)


def parse_request_path() -> Path:
    if "--" not in sys.argv:
        raise SystemExit("usage: blender -b --python blender_pbr_render_multiview.py -- request.json")
    idx = sys.argv.index("--")
    if idx + 1 >= len(sys.argv):
        raise SystemExit("missing request.json after --")
    return Path(sys.argv[idx + 1])


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    for collection in (bpy.data.meshes, bpy.data.materials, bpy.data.textures, bpy.data.images, bpy.data.lights, bpy.data.cameras):
        for item in list(collection):
            collection.remove(item)


def import_asset(path: str) -> None:
    suffix = Path(path).suffix.lower()
    if suffix in {".glb", ".gltf"}:
        bpy.ops.import_scene.gltf(filepath=path)
    elif suffix == ".obj":
        if bpy.app.version[0] >= 4:
            bpy.ops.wm.obj_import(filepath=path)
        else:
            bpy.ops.import_scene.obj(filepath=path)
    else:
        raise ValueError(f"unsupported asset type for Blender render: {suffix}")


def mesh_objects() -> list[bpy.types.Object]:
    return [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]


def remove_imported_cameras_and_lights() -> None:
    for obj in list(bpy.context.scene.objects):
        if obj.type in {"CAMERA", "LIGHT"}:
            bpy.data.objects.remove(obj, do_unlink=True)


def normalize_scene(center: list[float], scale: float, margin: float) -> None:
    transform = PIXAL3D_ROTATION_4X4 @ Matrix.Diagonal((margin / scale, margin / scale, margin / scale, 1.0)) @ Matrix.Translation(
        (-center[0], -center[1], -center[2])
    )
    for obj in mesh_objects():
        obj.matrix_world = transform @ obj.matrix_world
    bpy.context.view_layer.update()


def setup_render(engine: str, image_size: int, samples: int, world_strength: float) -> None:
    scene = bpy.context.scene
    try:
        scene.render.engine = engine
    except Exception:
        scene.render.engine = "BLENDER_EEVEE_NEXT" if bpy.app.version[0] >= 4 else "BLENDER_EEVEE"
    scene.render.resolution_x = image_size
    scene.render.resolution_y = image_size
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"

    if scene.render.engine == "CYCLES":
        scene.cycles.samples = samples
        scene.cycles.use_denoising = True
        scene.cycles.diffuse_bounces = 2
        scene.cycles.glossy_bounces = 2
    elif hasattr(scene, "eevee"):
        if hasattr(scene.eevee, "use_gtao"):
            scene.eevee.use_gtao = True
        if hasattr(scene.eevee, "gtao_distance"):
            scene.eevee.gtao_distance = 3
        if hasattr(scene.eevee, "gtao_factor"):
            scene.eevee.gtao_factor = 1.4

    if scene.world is None:
        scene.world = bpy.data.worlds.new("World")
    scene.world.color = (1.0, 1.0, 1.0)
    if scene.world.node_tree is None:
        scene.world.use_nodes = True
    bg = scene.world.node_tree.nodes.get("Background") if scene.world.node_tree else None
    if bg is not None:
        bg.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
        bg.inputs["Strength"].default_value = world_strength

    try:
        scene.view_settings.view_transform = "Standard"
    except Exception:
        pass
    try:
        scene.view_settings.look = "Medium High Contrast"
    except Exception:
        scene.view_settings.look = "None"
    scene.view_settings.exposure = 0
    scene.view_settings.gamma = 1


def add_lights(light_energy: float) -> None:
    specs = [
        ("Key", (2.5, -3.0, 3.5), light_energy, 4.0),
        ("Fill", (-3.0, 2.0, 2.5), light_energy * 0.35, 5.5),
        ("Rim", (0.0, 3.5, 4.0), light_energy * 0.25, 5.0),
    ]
    for name, location, energy, size in specs:
        light = bpy.data.lights.new(name, type="AREA")
        light.energy = energy
        light.size = size
        obj = bpy.data.objects.new(name, light)
        obj.location = location
        bpy.context.collection.objects.link(obj)


def setup_camera(image_size: int, intrinsic: list[list[float]]) -> bpy.types.Object:
    cam_data = bpy.data.cameras.new("Camera")
    cam = bpy.data.objects.new("Camera", cam_data)
    bpy.context.collection.objects.link(cam)
    bpy.context.scene.camera = cam
    focal = float(intrinsic[0][0])
    cam_data.lens_unit = "FOV"
    cam_data.angle = 2.0 * math.atan(float(image_size) / max(2.0 * focal, 1e-6))
    cam_data.clip_start = 0.01
    cam_data.clip_end = 100.0
    cx = float(intrinsic[0][2])
    cy = float(intrinsic[1][2])
    cam_data.shift_x = (cx - (image_size - 1) * 0.5) / max(image_size, 1)
    cam_data.shift_y = -((cy - (image_size - 1) * 0.5) / max(image_size, 1))
    return cam


def set_camera_pose(cam: bpy.types.Object, c2w: list[list[float]]) -> None:
    right = [c2w[0][0], c2w[1][0], c2w[2][0]]
    down = [c2w[0][1], c2w[1][1], c2w[2][1]]
    forward = [c2w[0][2], c2w[1][2], c2w[2][2]]
    loc = [c2w[0][3], c2w[1][3], c2w[2][3]]
    up = [-down[0], -down[1], -down[2]]
    back = [-forward[0], -forward[1], -forward[2]]
    cam.matrix_world = Matrix(
        (
            (right[0], up[0], back[0], loc[0]),
            (right[1], up[1], back[1], loc[1]),
            (right[2], up[2], back[2], loc[2]),
            (0.0, 0.0, 0.0, 1.0),
        )
    )


def main() -> None:
    request = json.loads(parse_request_path().read_text(encoding="utf-8"))
    output_dir = Path(request["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    clear_scene()
    setup_render(
        request.get("engine", "BLENDER_EEVEE"),
        int(request["image_size"]),
        int(request.get("samples", 48)),
        float(request.get("world_strength", 0.45)),
    )
    import_asset(request["glb_path"])
    remove_imported_cameras_and_lights()
    if not mesh_objects():
        raise RuntimeError("Blender import produced no mesh objects")
    normalize_scene(request["center"], float(request["scale"]), float(request["margin"]))
    add_lights(float(request.get("light_energy", 500.0)))
    cam = setup_camera(int(request["image_size"]), request["intrinsic"])

    for view_idx, c2w in enumerate(request["c2w"]):
        set_camera_pose(cam, c2w)
        bpy.context.scene.render.filepath = str(output_dir / f"view_{view_idx:03d}.png")
        bpy.ops.render.render(write_still=True)


if __name__ == "__main__":
    main()
