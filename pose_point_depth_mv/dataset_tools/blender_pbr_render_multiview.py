#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Matrix, Vector


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


def import_asset(path: str) -> str:
    suffix = Path(path).suffix.lower()
    if suffix in {".glb", ".gltf"}:
        bpy.ops.import_scene.gltf(filepath=path)
        return "gltf_y_up_to_blender_z_up_v1"
    elif suffix == ".obj":
        if bpy.app.version[0] >= 4:
            bpy.ops.wm.obj_import(
                filepath=path,
                forward_axis="NEGATIVE_Z",
                up_axis="Y",
            )
        else:
            bpy.ops.import_scene.obj(
                filepath=path,
                axis_forward="-Z",
                axis_up="Y",
            )
        return "obj_forward_neg_z_up_y_to_blender_v1"
    else:
        raise ValueError(f"unsupported asset type for Blender render: {suffix}")


def mesh_objects() -> list[bpy.types.Object]:
    return [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]


def remove_imported_cameras_and_lights() -> None:
    for obj in list(bpy.context.scene.objects):
        if obj.type in {"CAMERA", "LIGHT"}:
            bpy.data.objects.remove(obj, do_unlink=True)


def normalize_scene(
    center: list[float],
    scale: float,
    margin: float,
    source_path: str,
) -> None:
    scaling = Matrix.Diagonal(
        (margin / scale, margin / scale, margin / scale, 1.0)
    )
    suffix = Path(source_path).suffix.lower()
    if suffix not in {".glb", ".gltf", ".obj"}:
        raise ValueError(f"unsupported source-frame normalization: {suffix}")

    # The glTF and explicitly configured OBJ importers both convert the source
    # frame with [x, y, z] -> [x, -z, y], matching Pixal3D.  Normalize in that
    # already imported frame.  Applying PIXAL3D_ROTATION_4X4 again here
    # double-rotates OBJ assets and subtracting an unrotated center also leaves
    # them off-center.
    source_center = Vector((center[0], center[1], center[2], 1.0))
    imported_center = PIXAL3D_ROTATION_4X4 @ source_center
    transform = scaling @ Matrix.Translation(
        (-imported_center.x, -imported_center.y, -imported_center.z)
    )
    for obj in mesh_objects():
        obj.matrix_world = transform @ obj.matrix_world
    bpy.context.view_layer.update()


def normalized_scene_bounds() -> dict[str, list[float]]:
    points = []
    for obj in mesh_objects():
        transform = obj.matrix_world
        points.extend(transform @ vertex.co for vertex in obj.data.vertices)
    if not points:
        raise RuntimeError("cannot measure normalized scene bounds without vertices")
    minimum = Vector(
        (
            min(point.x for point in points),
            min(point.y for point in points),
            min(point.z for point in points),
        )
    )
    maximum = Vector(
        (
            max(point.x for point in points),
            max(point.y for point in points),
            max(point.z for point in points),
        )
    )
    return {
        "minimum": [float(value) for value in minimum],
        "maximum": [float(value) for value in maximum],
        "mesh_objects": int(len(mesh_objects())),
        "vertices": int(sum(len(obj.data.vertices) for obj in mesh_objects())),
    }


def build_bounds_metadata(
    request: dict,
    import_policy: str,
) -> dict:
    bounds = normalized_scene_bounds()
    payload = {
        "normalized_scene_bounds": bounds,
        "source_suffix": Path(request["glb_path"]).suffix.lower(),
        "source_import_policy": import_policy,
        "normalization_policy": "imported_frame_center_scale_v2",
    }
    expected = request.get("expected_normalized_scene_bounds")
    tolerance = request.get("bounds_tolerance")
    if expected is None or tolerance is None:
        return payload

    expected_minimum = [float(value) for value in expected["minimum"]]
    expected_maximum = [float(value) for value in expected["maximum"]]
    actual_minimum = [float(value) for value in bounds["minimum"]]
    actual_maximum = [float(value) for value in bounds["maximum"]]
    if not (
        len(expected_minimum)
        == len(expected_maximum)
        == len(actual_minimum)
        == len(actual_maximum)
        == 3
    ):
        raise RuntimeError("source-frame bounds must contain three coordinates")
    max_abs = max(
        abs(actual - wanted)
        for actual, wanted in zip(actual_minimum, expected_minimum)
    )
    max_abs = max(
        max_abs,
        max(
            abs(actual - wanted)
            for actual, wanted in zip(actual_maximum, expected_maximum)
        ),
    )
    payload["bounds_audit"] = {
        "max_abs": float(max_abs),
        "tolerance": float(tolerance),
        "passed": bool(max_abs <= float(tolerance)),
        "expected_minimum": expected_minimum,
        "expected_maximum": expected_maximum,
        "actual_minimum": actual_minimum,
        "actual_maximum": actual_maximum,
    }
    return payload


def setup_render(
    engine: str,
    image_size: int,
    samples: int,
    world_strength: float,
    cycles_device: str,
) -> None:
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
        requested_device = cycles_device.upper()
        if requested_device == "CPU":
            scene.cycles.device = "CPU"
        else:
            cycles_preferences = bpy.context.preferences.addons["cycles"].preferences
            cycles_preferences.compute_device_type = requested_device
            cycles_preferences.get_devices()
            matching_devices = [
                device
                for device in cycles_preferences.devices
                if device.type == requested_device
            ]
            if not matching_devices:
                available = [
                    (device.name, device.type)
                    for device in cycles_preferences.devices
                ]
                raise RuntimeError(
                    f"Cycles {requested_device} device is unavailable; "
                    f"available={available}"
                )
            for device in cycles_preferences.devices:
                device.use = device.type == requested_device
            scene.cycles.device = "GPU"
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


def apply_material_mode(mode: str) -> None:
    """Keep source appearance or replace every material with one neutral clay."""

    if mode == "source":
        return
    if mode != "clay":
        raise ValueError(f"unsupported material_mode={mode!r}")
    material = bpy.data.materials.new("Shared neutral clay")
    material.use_nodes = True
    principled = material.node_tree.nodes.get("Principled BSDF")
    if principled is not None:
        principled.inputs["Base Color"].default_value = (0.62, 0.67, 0.74, 1.0)
        principled.inputs["Roughness"].default_value = 0.78
        if "Metallic" in principled.inputs:
            principled.inputs["Metallic"].default_value = 0.0
    for obj in mesh_objects():
        obj.data.materials.clear()
        obj.data.materials.append(material)


def set_camera_intrinsic(
    cam_data: bpy.types.Camera,
    image_size: int,
    intrinsic: list[list[float]],
) -> None:
    """Apply one OpenCV 3x3 intrinsic to an existing Blender camera."""

    focal = float(intrinsic[0][0])
    cam_data.lens_unit = "FOV"
    cam_data.angle = 2.0 * math.atan(float(image_size) / max(2.0 * focal, 1e-6))
    cam_data.clip_start = 0.01
    cam_data.clip_end = 100.0
    cx = float(intrinsic[0][2])
    cy = float(intrinsic[1][2])
    cam_data.shift_x = (cx - (image_size - 1) * 0.5) / max(image_size, 1)
    cam_data.shift_y = -((cy - (image_size - 1) * 0.5) / max(image_size, 1))


def setup_camera(image_size: int, intrinsic: list[list[float]]) -> bpy.types.Object:
    cam_data = bpy.data.cameras.new("Camera")
    set_camera_intrinsic(cam_data, image_size, intrinsic)
    cam = bpy.data.objects.new("Camera", cam_data)
    bpy.context.collection.objects.link(cam)
    bpy.context.scene.camera = cam
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
    bounds_only = bool(request.get("bounds_only", False))
    output_dir = Path(request.get("output_dir", "."))
    if not bounds_only:
        output_dir.mkdir(parents=True, exist_ok=True)

    clear_scene()
    if not bounds_only:
        setup_render(
            request.get("engine", "BLENDER_EEVEE"),
            int(request["image_size"]),
            int(request.get("samples", 48)),
            float(request.get("world_strength", 0.45)),
            str(request.get("cycles_device", "CUDA")),
        )
    import_policy = import_asset(request["glb_path"])
    remove_imported_cameras_and_lights()
    if not mesh_objects():
        raise RuntimeError("Blender import produced no mesh objects")
    normalize_scene(
        request["center"],
        float(request["scale"]),
        float(request["margin"]),
        request["glb_path"],
    )
    apply_material_mode(str(request.get("material_mode", "source")))
    metadata = build_bounds_metadata(request, import_policy)
    metadata_path = request.get("metadata_path")
    if metadata_path:
        Path(metadata_path).write_text(
            json.dumps(metadata),
            encoding="utf-8",
        )
    bounds_audit = metadata.get("bounds_audit")
    if bounds_audit is not None and bounds_audit["passed"] is not True:
        raise RuntimeError(
            "source asset bounds incompatible before rendering: "
            f"max_abs={bounds_audit['max_abs']:.8f}, "
            f"tolerance={bounds_audit['tolerance']:.8f}"
        )
    if bounds_only:
        return

    add_lights(float(request.get("light_energy", 500.0)))
    intrinsic_request = request["intrinsic"]
    per_view_intrinsics = (
        isinstance(intrinsic_request, list)
        and bool(intrinsic_request)
        and isinstance(intrinsic_request[0], list)
        and bool(intrinsic_request[0])
        and isinstance(intrinsic_request[0][0], list)
    )
    first_intrinsic = intrinsic_request[0] if per_view_intrinsics else intrinsic_request
    if per_view_intrinsics and len(intrinsic_request) != len(request["c2w"]):
        raise RuntimeError(
            "per-view intrinsic count differs from camera count: "
            f"{len(intrinsic_request)} != {len(request['c2w'])}"
        )
    cam = setup_camera(int(request["image_size"]), first_intrinsic)

    for view_idx, c2w in enumerate(request["c2w"]):
        if per_view_intrinsics:
            set_camera_intrinsic(
                cam.data,
                int(request["image_size"]),
                intrinsic_request[view_idx],
            )
        set_camera_pose(cam, c2w)
        bpy.context.scene.render.filepath = str(output_dir / f"view_{view_idx:03d}.png")
        bpy.ops.render.render(write_still=True)


if __name__ == "__main__":
    main()
