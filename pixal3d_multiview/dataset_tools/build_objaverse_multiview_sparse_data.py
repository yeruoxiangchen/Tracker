#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import trimesh
from PIL import Image
from tqdm.auto import tqdm


TRACKER_ROOT = Path(__file__).resolve().parents[2]
PIXAL3D_ROOT = TRACKER_ROOT / "Pixal3D"
sys.path.insert(0, str(TRACKER_ROOT))
sys.path.insert(0, str(PIXAL3D_ROOT))


PIXAL3D_ROTATION = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float32,
)

BLENDER_RENDER_SCRIPT = Path(__file__).resolve().with_name("blender_pbr_render_multiview.py")


def load_meshes(path: str) -> list[trimesh.Trimesh]:
    obj = trimesh.load(path, force="scene", process=False)
    if isinstance(obj, trimesh.Scene):
        meshes = [m for m in obj.dump(concatenate=False) if isinstance(m, trimesh.Trimesh) and len(m.vertices) > 0]
    elif isinstance(obj, trimesh.Trimesh):
        meshes = [obj]
    else:
        raise ValueError(f"unsupported object type: {type(obj)}")
    if not meshes:
        raise ValueError("scene has no mesh geometry")
    for mesh in meshes:
        if mesh.faces is None or len(mesh.faces) == 0 or len(mesh.vertices) == 0:
            raise ValueError("empty mesh")
        mesh.remove_unreferenced_vertices()
    return meshes


def normalize_vertices(vertices: np.ndarray, margin: float) -> tuple[np.ndarray, np.ndarray, float]:
    vmin = vertices.min(axis=0)
    vmax = vertices.max(axis=0)
    center = (vmin + vmax) * 0.5
    scale = float(np.max(vmax - vmin))
    if not np.isfinite(scale) or scale <= 1e-8:
        raise ValueError("invalid mesh scale")
    normalized = (vertices - center[None]) / scale * float(margin)
    return normalized.astype(np.float32), center.astype(np.float32), scale


def pixal3d_render_space(points_latent: np.ndarray) -> np.ndarray:
    return (points_latent @ PIXAL3D_ROTATION.T).astype(np.float32)


def sample_surface_points(mesh: trimesh.Trimesh, vertices_latent: np.ndarray, num_points: int) -> np.ndarray:
    local_mesh = mesh.copy()
    local_mesh.vertices = vertices_latent
    points, _ = trimesh.sample.sample_surface(local_mesh, int(num_points))
    return points.astype(np.float32)


def coords_from_points(points: np.ndarray, resolution: int) -> np.ndarray:
    coords = np.floor((points + 0.5) * int(resolution)).astype(np.int32)
    coords = np.clip(coords, 0, int(resolution) - 1)
    return np.unique(coords, axis=0).astype(np.int32)


def _sample_texture_nearest(image: Image.Image, uv: np.ndarray) -> np.ndarray:
    tex = np.asarray(image.convert("RGB"))
    uv = np.asarray(uv, dtype=np.float32)
    u = np.mod(uv[:, 0], 1.0)
    v = 1.0 - np.mod(uv[:, 1], 1.0)
    x = np.clip(np.round(u * (tex.shape[1] - 1)).astype(np.int64), 0, tex.shape[1] - 1)
    y = np.clip(np.round(v * (tex.shape[0] - 1)).astype(np.int64), 0, tex.shape[0] - 1)
    return tex[y, x, :3].astype(np.uint8)


def _material_base_color(mesh: trimesh.Trimesh) -> np.ndarray:
    material = getattr(getattr(mesh, "visual", None), "material", None)
    base = getattr(material, "baseColorFactor", None)
    if base is None:
        return np.array([180, 180, 180], dtype=np.uint8)
    base = np.asarray(base).reshape(-1)
    if base.size < 3:
        return np.array([180, 180, 180], dtype=np.uint8)
    color = base[:3] * 255.0 if base[:3].max() <= 1.0 else base[:3]
    return np.clip(color, 0, 255).astype(np.uint8)


def vertex_colors_from_visual(mesh: trimesh.Trimesh) -> np.ndarray:
    vertex_count = len(mesh.vertices)
    visual = getattr(mesh, "visual", None)
    if visual is None:
        return np.repeat(_material_base_color(mesh)[None], vertex_count, axis=0)

    if getattr(visual, "kind", None) == "texture":
        uv = getattr(visual, "uv", None)
        material = getattr(visual, "material", None)
        image = getattr(material, "image", None)
        if image is None:
            image = getattr(material, "baseColorTexture", None)
        if uv is not None and image is not None and len(uv) == vertex_count:
            try:
                return _sample_texture_nearest(image, uv)
            except Exception:
                pass

    try:
        color_visual = visual.to_color()
        vertex_colors = np.asarray(color_visual.vertex_colors)
        if vertex_colors.ndim == 2 and vertex_colors.shape[0] == vertex_count and vertex_colors.shape[1] >= 3:
            return np.clip(vertex_colors[:, :3], 0, 255).astype(np.uint8)
    except Exception:
        pass
    return np.repeat(_material_base_color(mesh)[None], vertex_count, axis=0)


def build_render_buffers(
    meshes: list[trimesh.Trimesh],
    center: np.ndarray,
    scale: float,
    margin: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    vertices_all, faces_all, colors_all, normals_all = [], [], [], []
    offset = 0
    for mesh in meshes:
        vertices_latent = (np.asarray(mesh.vertices, dtype=np.float32) - center[None]) / float(scale) * float(margin)
        vertices_render = pixal3d_render_space(vertices_latent)
        faces = np.asarray(mesh.faces, dtype=np.int32) + offset
        colors = vertex_colors_from_visual(mesh)
        normals = np.asarray(mesh.vertex_normals, dtype=np.float32)
        if normals.shape != vertices_latent.shape:
            normals = np.repeat(np.array([[0.0, 0.0, 1.0]], dtype=np.float32), vertices_latent.shape[0], axis=0)
        normals = pixal3d_render_space(normals)
        normals = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8)
        vertices_all.append(vertices_render.astype(np.float32))
        faces_all.append(faces.astype(np.int32))
        colors_all.append(colors.astype(np.uint8))
        normals_all.append(normals.astype(np.float32))
        offset += vertices_render.shape[0]
    return (
        np.concatenate(vertices_all, axis=0),
        np.concatenate(faces_all, axis=0),
        np.concatenate(colors_all, axis=0),
        np.concatenate(normals_all, axis=0),
    )


def make_nvdiffrast_context():
    try:
        import nvdiffrast.torch as dr
    except Exception as exc:
        raise RuntimeError("mesh rendering requires nvdiffrast in the active environment") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("mesh rendering requires CUDA")
    return dr.RasterizeCudaContext()


def render_mesh_rgb(
    vertices: np.ndarray,
    faces: np.ndarray,
    vertex_colors: np.ndarray,
    vertex_normals: np.ndarray,
    c2w: np.ndarray,
    intrinsic: np.ndarray,
    image_size: int,
    rastctx,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray]:
    import nvdiffrast.torch as dr

    device = torch.device("cuda")
    verts = torch.as_tensor(vertices, dtype=torch.float32, device=device)
    faces_t = torch.as_tensor(faces.astype(np.int32), dtype=torch.int32, device=device)
    colors = torch.as_tensor(vertex_colors.astype(np.float32) / 255.0, dtype=torch.float32, device=device)
    normals = torch.as_tensor(vertex_normals.astype(np.float32), dtype=torch.float32, device=device)
    c2w_t = torch.as_tensor(c2w, dtype=torch.float32, device=device)
    w2c = torch.linalg.inv(c2w_t)
    verts_h = torch.cat([verts, torch.ones((verts.shape[0], 1), device=device, dtype=torch.float32)], dim=1)
    cam = (verts_h @ w2c.T)[:, :3]
    z = cam[:, 2].clamp(min=1e-4)
    u = intrinsic[0, 0] * (cam[:, 0] / z) + intrinsic[0, 2]
    v = intrinsic[1, 1] * (cam[:, 1] / z) + intrinsic[1, 2]
    x_ndc = 2.0 * u / max(image_size - 1, 1) - 1.0
    y_ndc = 1.0 - 2.0 * v / max(image_size - 1, 1)
    near, far = 0.01, 10.0
    z_ndc = 2.0 * (z - near) / (far - near) - 1.0
    pos_clip = torch.stack([x_ndc * z, y_ndc * z, z_ndc * z, z], dim=1)

    rast, _ = dr.rasterize(rastctx, pos_clip[None], faces_t, resolution=(image_size, image_size))
    color, _ = dr.interpolate(colors[None], rast, faces_t)
    if args.renderer == "nvdiffrast" and args.shading_mode == "normal":
        normal, _ = dr.interpolate(normals[None], rast, faces_t)
        normal = torch.nn.functional.normalize(normal, dim=-1, eps=1e-6)
        camera_light = -c2w_t[:3, 2]
        fill_light = torch.tensor([0.35, 0.55, 0.75], dtype=torch.float32, device=device)
        light_dir = torch.nn.functional.normalize(camera_light + 0.45 * fill_light, dim=0, eps=1e-6)
        diffuse = (normal * light_dir.view(1, 1, 1, 3)).sum(dim=-1, keepdim=True).clamp(min=0.0)
        shade = float(args.shading_ambient) + float(args.shading_diffuse) * diffuse
        color = (color * shade).clamp(0.0, 1.0)
    try:
        color = dr.antialias(color, rast, pos_clip[None], faces_t)
    except Exception:
        pass
    mask = (rast[..., 3:4] > 0).float()
    image = np.clip(color[0].detach().cpu().numpy() * mask[0].detach().cpu().numpy() * 255.0, 0, 255).astype(np.uint8)
    alpha = np.clip(mask[0, ..., 0].detach().cpu().numpy() * 255.0, 0, 255).astype(np.uint8)
    if int((alpha > 0).sum()) == 0:
        raise ValueError("mesh renderer produced an empty mask")
    return image, alpha


def find_blender_binary(args: argparse.Namespace) -> str:
    candidates = []
    if args.blender_path:
        candidates.append(args.blender_path)
    path_blender = shutil.which("blender")
    if path_blender:
        candidates.append(path_blender)
    home = Path.home()
    candidates.extend(
        [
            "/tmp/blender-3.0.1-linux-x64/blender",
            "/tmp/blender-4.5.1-linux-x64/blender",
            str(home / ".blender" / "blender-4.5.1-linux-x64" / "blender"),
            str(home / ".blender" / "blender-3.0.1-linux-x64" / "blender"),
        ]
    )
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate
    raise FileNotFoundError(
        "Blender executable not found. Pass --blender_path or install Blender; "
        "use --renderer nvdiffrast for the CUDA fallback renderer."
    )


def render_mesh_blender(
    glb_path: str,
    center: np.ndarray,
    scale: float,
    c2w: np.ndarray,
    intrinsic: np.ndarray,
    image_size: int,
    args: argparse.Namespace,
) -> list[tuple[np.ndarray, np.ndarray]]:
    blender = find_blender_binary(args)
    with tempfile.TemporaryDirectory(prefix="pixal3d_mv_blender_") as tmpdir:
        tmpdir_path = Path(tmpdir)
        request = {
            "glb_path": str(glb_path),
            "output_dir": str(tmpdir_path / "rgba"),
            "center": center.astype(float).tolist(),
            "scale": float(scale),
            "margin": float(args.canonical_margin),
            "c2w": c2w.astype(float).tolist(),
            "intrinsic": intrinsic.astype(float).tolist(),
            "image_size": int(image_size),
            "engine": args.blender_engine,
            "samples": int(args.blender_samples),
            "world_strength": float(args.blender_world_strength),
            "light_energy": float(args.blender_light_energy),
        }
        request_path = tmpdir_path / "request.json"
        request_path.write_text(json.dumps(request), encoding="utf-8")
        cmd = [blender, "-b", "--python", str(BLENDER_RENDER_SCRIPT), "--", str(request_path)]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL if args.blender_quiet else None)

        frames = []
        for view_idx in range(c2w.shape[0]):
            rgba_path = tmpdir_path / "rgba" / f"view_{view_idx:03d}.png"
            rgba = np.asarray(Image.open(rgba_path).convert("RGBA"))
            alpha = rgba[..., 3]
            rgb = rgba[..., :3].astype(np.float32) * (alpha[..., None].astype(np.float32) / 255.0)
            frames.append((np.clip(rgb, 0, 255).astype(np.uint8), alpha.astype(np.uint8)))
    return frames


def color_stats(values: np.ndarray) -> dict:
    arr = np.asarray(values)
    if arr.size == 0:
        return {"rgb_std_mean": 0.0, "unique_colors16": 0, "channel_spread": 0.0}
    arr = arr.reshape(-1, arr.shape[-1])[:, :3].astype(np.float32)
    means = arr.mean(axis=0)
    return {
        "rgb_mean": [float(x) for x in means.tolist()],
        "rgb_std": [float(x) for x in arr.std(axis=0).tolist()],
        "rgb_std_mean": float(arr.std(axis=0).mean()),
        "unique_colors16": int(np.unique((arr.astype(np.uint8) // 16), axis=0).shape[0]),
        "channel_spread": float(means.max() - means.min()),
    }


def shape_stats(vertices_latent: np.ndarray, target_coords: np.ndarray) -> dict:
    extents = np.ptp(vertices_latent.astype(np.float32), axis=0)
    max_extent = float(np.max(extents))
    min_extent = float(np.min(extents))
    return {
        "extent": [float(x) for x in extents.tolist()],
        "extent_ratio": float(min_extent / (max_extent + 1e-8)),
        "num_voxels": int(target_coords.shape[0]),
    }


def target_coords_to_render_points(target_coords: np.ndarray, resolution: int) -> np.ndarray:
    points_latent = (target_coords.astype(np.float32) + 0.5) / float(resolution) - 0.5
    return pixal3d_render_space(points_latent)


def projection_support_stats(
    target_coords: np.ndarray,
    c2w: np.ndarray,
    intrinsic: np.ndarray,
    alphas: list[np.ndarray],
    resolution: int,
    mask_threshold: float,
) -> dict:
    points = target_coords_to_render_points(target_coords, resolution)
    points_h = np.concatenate([points, np.ones((points.shape[0], 1), dtype=np.float32)], axis=1)
    support = np.zeros((points.shape[0],), dtype=np.int32)
    visible = np.zeros((points.shape[0],), dtype=np.int32)
    per_view_hit_ratio = []

    for view_idx, alpha in enumerate(alphas):
        w2c = np.linalg.inv(c2w[view_idx].astype(np.float32))
        cam = (points_h @ w2c.T)[:, :3]
        z = cam[:, 2]
        valid_depth = z > 1e-4
        u = intrinsic[0, 0] * (cam[:, 0] / np.clip(z, 1e-4, None)) + intrinsic[0, 2]
        v = intrinsic[1, 1] * (cam[:, 1] / np.clip(z, 1e-4, None)) + intrinsic[1, 2]
        ui = np.rint(u).astype(np.int64)
        vi = np.rint(v).astype(np.int64)
        in_image = (
            valid_depth
            & (ui >= 0)
            & (ui < alpha.shape[1])
            & (vi >= 0)
            & (vi < alpha.shape[0])
        )
        visible += in_image.astype(np.int32)
        hits = np.zeros_like(in_image)
        ids = np.where(in_image)[0]
        if ids.size:
            hits[ids] = alpha[vi[ids], ui[ids]] > int(mask_threshold * 255.0)
        support += hits.astype(np.int32)
        per_view_hit_ratio.append(float(hits[ids].mean()) if ids.size else 0.0)

    visible_mask = visible > 0
    ratio = np.zeros((points.shape[0],), dtype=np.float32)
    ratio[visible_mask] = support[visible_mask].astype(np.float32) / np.maximum(visible[visible_mask].astype(np.float32), 1.0)
    if visible_mask.any():
        ratio_visible = ratio[visible_mask]
        support_visible = support[visible_mask]
        visible_visible = visible[visible_mask]
        return {
            "num_target_points": int(points.shape[0]),
            "visible_points": int(visible_mask.sum()),
            "visible_points_ratio": float(visible_mask.mean()),
            "visible_views_mean": float(visible_visible.mean()),
            "support_views_mean": float(support_visible.mean()),
            "support_ratio_mean": float(ratio_visible.mean()),
            "support_ratio_p05": float(np.quantile(ratio_visible, 0.05)),
            "support_ratio_p10": float(np.quantile(ratio_visible, 0.10)),
            "zero_support_ratio": float((support_visible <= 0).mean()),
            "per_view_hit_ratio_mean": float(np.mean(np.asarray(per_view_hit_ratio, dtype=np.float32))),
        }
    return {
        "num_target_points": int(points.shape[0]),
        "visible_points": 0,
        "visible_points_ratio": 0.0,
        "visible_views_mean": 0.0,
        "support_views_mean": 0.0,
        "support_ratio_mean": 0.0,
        "support_ratio_p05": 0.0,
        "support_ratio_p10": 0.0,
        "zero_support_ratio": 1.0,
        "per_view_hit_ratio_mean": 0.0,
    }


def select_projection_support_views(
    c2w: np.ndarray,
    alphas: list[np.ndarray],
    projection_support_frames: int,
) -> tuple[np.ndarray, list[np.ndarray], int]:
    if projection_support_frames <= 0:
        return c2w, alphas, len(alphas)
    frame_count = min(int(projection_support_frames), len(alphas), int(c2w.shape[0]))
    if frame_count <= 0:
        raise ValueError("projection support frame count resolved to 0")
    return c2w[:frame_count], alphas[:frame_count], frame_count


def look_at_c2w(eye: np.ndarray, target: Optional[np.ndarray] = None, roll_rad: float = 0.0) -> np.ndarray:
    target = np.zeros(3, dtype=np.float32) if target is None else target.astype(np.float32)
    forward = target - eye
    forward = forward / (np.linalg.norm(forward) + 1e-8)
    world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    right = np.cross(world_up, forward)
    if np.linalg.norm(right) < 1e-6:
        world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        right = np.cross(world_up, forward)
    right = right / (np.linalg.norm(right) + 1e-8)
    down = np.cross(forward, right)
    down = down / (np.linalg.norm(down) + 1e-8)
    if abs(roll_rad) > 1e-8:
        c = math.cos(roll_rad)
        s = math.sin(roll_rad)
        right, down = c * right + s * down, -s * right + c * down
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, 0] = right
    c2w[:3, 1] = down
    c2w[:3, 2] = forward
    c2w[:3, 3] = eye
    return c2w


def make_camera_ring(num_views: int, radius: float, elev_deg: float) -> np.ndarray:
    cameras = []
    elev = math.radians(elev_deg)
    for i in range(num_views):
        az = 2.0 * math.pi * i / max(num_views, 1)
        eye = np.array(
            [
                radius * math.cos(elev) * math.sin(az),
                radius * math.sin(elev),
                radius * math.cos(elev) * math.cos(az),
            ],
            dtype=np.float32,
        )
        cameras.append(look_at_c2w(eye))
    return np.stack(cameras, axis=0)


def make_ar_random_trajectory(num_views: int, args: argparse.Namespace, rng: np.random.Generator) -> np.ndarray:
    base_radius = float(rng.uniform(args.radius_min, args.radius_max))
    az0 = float(rng.uniform(0.0, 2.0 * math.pi))
    az_span = math.radians(float(rng.uniform(args.azimuth_span_min, args.azimuth_span_max)))
    if rng.random() < 0.5:
        az_span = -az_span
    elev0 = math.radians(float(rng.uniform(args.elevation_min, args.elevation_max)))
    elev_drift = math.radians(float(rng.uniform(-args.elevation_drift, args.elevation_drift)))
    radial_drift = float(rng.uniform(-args.radius_drift, args.radius_drift))
    target_base = rng.normal(0.0, args.target_jitter, size=3).astype(np.float32)
    target_base[2] *= 0.5

    ts = np.array([0.0], dtype=np.float32) if num_views <= 1 else np.linspace(-0.5, 0.5, num_views, dtype=np.float32)
    cameras = []
    for t in ts:
        az = az0 + float(t) * az_span + math.radians(float(rng.normal(0.0, args.azimuth_jitter)))
        elev = elev0 + float(t) * elev_drift + math.radians(float(rng.normal(0.0, args.elevation_jitter)))
        radius = base_radius * (1.0 + float(t) * radial_drift + float(rng.normal(0.0, args.radius_jitter)))
        radius = float(np.clip(radius, args.radius_min, args.radius_max))
        eye = np.array(
            [
                radius * math.cos(elev) * math.sin(az),
                radius * math.sin(elev),
                radius * math.cos(elev) * math.cos(az),
            ],
            dtype=np.float32,
        )
        lateral = rng.normal(0.0, args.camera_lateral_jitter, size=3).astype(np.float32)
        lateral[2] *= 0.5
        target = target_base + rng.normal(0.0, args.lookat_jitter, size=3).astype(np.float32)
        target[2] *= 0.5
        roll = math.radians(float(rng.normal(0.0, args.roll_jitter)))
        cameras.append(look_at_c2w(eye + lateral, target, roll_rad=roll))
    return np.stack(cameras, axis=0)


def make_intrinsic(args: argparse.Namespace) -> np.ndarray:
    focal = float(args.focal_ratio) * int(args.image_size)
    return np.array(
        [
            [focal, 0.0, (args.image_size - 1) * 0.5],
            [0.0, focal, (args.image_size - 1) * 0.5],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def camera_framing_stats(
    points_render: np.ndarray,
    c2w: np.ndarray,
    intrinsic: np.ndarray,
    image_size: int,
    args: argparse.Namespace,
) -> dict:
    points_h = np.concatenate(
        [points_render.astype(np.float32), np.ones((points_render.shape[0], 1), dtype=np.float32)],
        axis=1,
    )
    in_frame_ratios = []
    strict_in_frame_ratios = []
    bbox_margins = []
    margin = float(args.camera_framing_margin_px)
    width = float(image_size)
    height = float(image_size)

    for view_idx in range(c2w.shape[0]):
        w2c = np.linalg.inv(c2w[view_idx].astype(np.float32))
        cam = (points_h @ w2c.T)[:, :3]
        z = cam[:, 2]
        valid_depth = z > 1e-4
        if not valid_depth.any():
            in_frame_ratios.append(0.0)
            strict_in_frame_ratios.append(0.0)
            bbox_margins.append(-float(image_size))
            continue

        z_safe = np.clip(z[valid_depth], 1e-4, None)
        u = intrinsic[0, 0] * (cam[valid_depth, 0] / z_safe) + intrinsic[0, 2]
        v = intrinsic[1, 1] * (cam[valid_depth, 1] / z_safe) + intrinsic[1, 2]
        in_frame = (u >= 0.0) & (u < width) & (v >= 0.0) & (v < height)
        strict = (u >= margin) & (u < width - margin) & (v >= margin) & (v < height - margin)
        in_frame_ratios.append(float(in_frame.mean()))
        strict_in_frame_ratios.append(float(strict.mean()))
        bbox_margins.append(float(min(u.min(), v.min(), width - 1.0 - u.max(), height - 1.0 - v.max())))

    in_frame_arr = np.asarray(in_frame_ratios, dtype=np.float32)
    strict_arr = np.asarray(strict_in_frame_ratios, dtype=np.float32)
    bbox_arr = np.asarray(bbox_margins, dtype=np.float32)
    complete = strict_arr >= float(args.min_complete_in_frame_ratio)
    usable = in_frame_arr >= float(args.min_usable_in_frame_ratio)
    clipped = ~complete
    return {
        "in_frame_ratio_per_view": [float(x) for x in in_frame_arr.tolist()],
        "strict_in_frame_ratio_per_view": [float(x) for x in strict_arr.tolist()],
        "bbox_margin_px_per_view": [float(x) for x in bbox_arr.tolist()],
        "in_frame_ratio_mean": float(in_frame_arr.mean()) if in_frame_arr.size else 0.0,
        "strict_in_frame_ratio_mean": float(strict_arr.mean()) if strict_arr.size else 0.0,
        "bbox_margin_px_min": float(bbox_arr.min()) if bbox_arr.size else 0.0,
        "complete_views": int(complete.sum()),
        "usable_views": int(usable.sum()),
        "clipped_views": int(clipped.sum()),
    }


def _fraction_count(num_views: int, fraction: float, mode: str) -> int:
    count = float(num_views) * float(fraction)
    if mode == "floor":
        return int(math.floor(count))
    return int(math.ceil(count))


def camera_framing_is_valid(stats: dict, num_views: int, args: argparse.Namespace) -> bool:
    min_complete = _fraction_count(num_views, args.min_complete_view_fraction, "ceil")
    min_usable = _fraction_count(num_views, args.min_usable_view_fraction, "ceil")
    max_clipped = _fraction_count(num_views, args.max_clipped_view_fraction, "floor")
    return (
        stats["complete_views"] >= min_complete
        and stats["usable_views"] >= min_usable
        and stats["clipped_views"] <= max_clipped
    )


def sample_camera_trajectory(
    points_render: np.ndarray,
    intrinsic: np.ndarray,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict]:
    if args.framing_probe_points > 0 and points_render.shape[0] > args.framing_probe_points:
        ids = rng.choice(points_render.shape[0], size=args.framing_probe_points, replace=False)
        probe_points = points_render[ids]
    else:
        probe_points = points_render

    attempts = max(1, int(args.trajectory_resample_attempts))
    best_c2w = None
    best_stats = None
    best_score = -1e9
    for attempt in range(attempts):
        if args.trajectory_mode == "ring":
            c2w = make_camera_ring(args.num_views, args.camera_radius, args.elevation_deg)
        else:
            c2w = make_ar_random_trajectory(args.num_views, args, rng)
        stats = camera_framing_stats(probe_points, c2w, intrinsic, args.image_size, args)
        stats["framing_attempt"] = int(attempt + 1)
        score = (
            2.0 * stats["complete_views"]
            + stats["usable_views"]
            - stats["clipped_views"]
            + stats["strict_in_frame_ratio_mean"]
        )
        if score > best_score:
            best_c2w = c2w
            best_stats = stats
            best_score = score
        if args.disable_camera_framing_constraint or camera_framing_is_valid(stats, args.num_views, args):
            return c2w, stats

    assert best_c2w is not None and best_stats is not None
    if args.disable_camera_framing_constraint:
        return best_c2w, best_stats

    min_complete = _fraction_count(args.num_views, args.min_complete_view_fraction, "ceil")
    min_usable = _fraction_count(args.num_views, args.min_usable_view_fraction, "ceil")
    max_clipped = _fraction_count(args.num_views, args.max_clipped_view_fraction, "floor")
    raise ValueError(
        "framing rejected by camera trajectory constraint: "
        f"best complete={best_stats['complete_views']}/{args.num_views} >= {min_complete}, "
        f"usable={best_stats['usable_views']}/{args.num_views} >= {min_usable}, "
        f"clipped={best_stats['clipped_views']}/{args.num_views} <= {max_clipped}, "
        f"bbox_margin_min={best_stats['bbox_margin_px_min']:.1f}, attempts={attempts}"
    )


def foreground_render_stats(images: list[np.ndarray], alphas: list[np.ndarray]) -> dict:
    vals = []
    fg_per_view = []
    area_per_view = []
    bbox_margin_per_view = []
    bbox_area_per_view = []
    for image, alpha in zip(images, alphas):
        mask = alpha > 0
        fg_per_view.append(int(mask.sum()))
        area_per_view.append(float(mask.mean()))
        if mask.any():
            vals.append(image[mask])
            ys, xs = np.where(mask)
            xmin, xmax = int(xs.min()), int(xs.max())
            ymin, ymax = int(ys.min()), int(ys.max())
            h, w = mask.shape
            margin = min(xmin, ymin, w - 1 - xmax, h - 1 - ymax)
            bbox_margin_per_view.append(int(margin))
            bbox_area = float((xmax - xmin + 1) * (ymax - ymin + 1) / max(h * w, 1))
            bbox_area_per_view.append(bbox_area)
        else:
            bbox_margin_per_view.append(0)
            bbox_area_per_view.append(0.0)
    if not vals:
        return {
            "fg_pixels": 0,
            "fg_pixels_per_view_min": 0,
            "fg_pixels_per_view_median": 0.0,
            "fg_pixels_per_view_max": 0,
            "fg_area_ratio_mean": 0.0,
            "bbox_margin_px_min": 0,
            "bbox_margin_px_median": 0.0,
            "bbox_margin_px_per_view": bbox_margin_per_view,
            "bbox_area_ratio_mean": 0.0,
            "bbox_area_ratio_max": 0.0,
            "rgb_std_mean": 0.0,
            "unique_colors16": 0,
            "channel_spread": 0.0,
        }
    arr = np.concatenate(vals, axis=0).astype(np.float32)
    stats = color_stats(arr)
    stats.update(
        {
            "fg_pixels": int(arr.shape[0]),
            "fg_pixels_per_view_min": int(min(fg_per_view)),
            "fg_pixels_per_view_median": float(np.median(np.asarray(fg_per_view, dtype=np.float32))),
            "fg_pixels_per_view_max": int(max(fg_per_view)),
            "fg_area_ratio_mean": float(np.mean(np.asarray(area_per_view, dtype=np.float32))),
            "bbox_margin_px_min": int(min(bbox_margin_per_view)),
            "bbox_margin_px_median": float(np.median(np.asarray(bbox_margin_per_view, dtype=np.float32))),
            "bbox_margin_px_per_view": bbox_margin_per_view,
            "bbox_area_ratio_mean": float(np.mean(np.asarray(bbox_area_per_view, dtype=np.float32))),
            "bbox_area_ratio_max": float(np.max(np.asarray(bbox_area_per_view, dtype=np.float32))),
        }
    )
    return stats


def per_frame_render_stats(alphas: list[np.ndarray]) -> list[dict]:
    stats = []
    for alpha in alphas:
        mask = alpha > 0
        if mask.any():
            ys, xs = np.where(mask)
            xmin, xmax = int(xs.min()), int(xs.max())
            ymin, ymax = int(ys.min()), int(ys.max())
            h, w = mask.shape
            margin = int(min(xmin, ymin, w - 1 - xmax, h - 1 - ymax))
            bbox_area = float((xmax - xmin + 1) * (ymax - ymin + 1) / max(h * w, 1))
        else:
            margin = 0
            bbox_area = 0.0
        stats.append(
            {
                "fg_pixels": int(mask.sum()),
                "fg_area_ratio": float(mask.mean()),
                "bbox_margin_px": int(margin),
                "bbox_area_ratio": float(bbox_area),
            }
        )
    return stats


def _effective_selection_threshold(value: float, fallback: float) -> float:
    return fallback if value < 0 else value


def frame_quality_scores(frame_stats: list[dict], args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    min_fg = int(_effective_selection_threshold(args.selection_min_fg_pixels_per_view, args.min_fg_pixels_per_view))
    min_area = float(_effective_selection_threshold(args.selection_min_fg_area_ratio, args.min_fg_area_ratio))
    min_margin = float(_effective_selection_threshold(args.selection_min_bbox_margin_px, args.min_bbox_margin_px))
    max_bbox_area = float(_effective_selection_threshold(args.selection_max_bbox_area_ratio, args.max_bbox_area_ratio))
    target_area = max(float(args.selection_target_fg_area_ratio), 1e-6)
    margin_score_px = max(float(args.selection_margin_score_px), 1e-6)

    good = []
    scores = []
    for stat in frame_stats:
        is_good = (
            stat["fg_pixels"] >= min_fg
            and stat["fg_area_ratio"] >= min_area
            and stat["bbox_margin_px"] >= min_margin
            and (max_bbox_area <= 0 or stat["bbox_area_ratio"] <= max_bbox_area)
        )
        area_score = min(stat["fg_area_ratio"] / target_area, 1.0)
        margin_score = min(max(float(stat["bbox_margin_px"]), 0.0) / margin_score_px, 1.0)
        bbox_penalty = 0.0
        if max_bbox_area > 0 and stat["bbox_area_ratio"] > max_bbox_area:
            bbox_penalty = min((stat["bbox_area_ratio"] - max_bbox_area) / max(1.0 - max_bbox_area, 1e-6), 1.0)
        score = 0.65 * area_score + 0.35 * margin_score - 0.5 * bbox_penalty
        good.append(bool(is_good))
        scores.append(float(score))
    return np.asarray(good, dtype=bool), np.asarray(scores, dtype=np.float32)


def camera_pose_distance(c2w: np.ndarray, i: int, j: int) -> float:
    centers = c2w[:, :3, 3].astype(np.float32)
    forwards = c2w[:, :3, 2].astype(np.float32)
    forwards = forwards / (np.linalg.norm(forwards, axis=1, keepdims=True) + 1e-8)
    radius_scale = float(np.median(np.linalg.norm(centers, axis=1))) + 1e-6
    angle = 1.0 - float(np.clip(np.dot(forwards[i], forwards[j]), -1.0, 1.0))
    center = min(float(np.linalg.norm(centers[i] - centers[j]) / radius_scale), 2.0)
    temporal = abs(float(i - j)) / max(float(c2w.shape[0] - 1), 1.0)
    return angle + 0.35 * center + 0.15 * temporal


def select_multiview_frames(
    images_and_masks: list[tuple[np.ndarray, np.ndarray]],
    c2w: np.ndarray,
    args: argparse.Namespace,
) -> tuple[list[int], dict]:
    candidate_count = len(images_and_masks)
    if args.selected_views <= 0 or args.frame_selection_policy == "none":
        return list(range(candidate_count)), {
            "policy": "none",
            "candidate_views": int(candidate_count),
            "selected_views": int(candidate_count),
            "good_candidate_views": int(candidate_count),
            "selected_indices": list(range(candidate_count)),
        }
    if args.selected_views > candidate_count:
        raise ValueError(
            f"frame selection rejected: selected_views={args.selected_views} > candidate_views={candidate_count}"
        )

    frame_stats = per_frame_render_stats([x[1] for x in images_and_masks])
    good_mask, quality_scores = frame_quality_scores(frame_stats, args)
    good_indices = [int(i) for i in np.where(good_mask)[0].tolist()]
    min_good = int(args.min_good_candidate_views) if args.min_good_candidate_views > 0 else int(args.selected_views)
    if len(good_indices) < min_good:
        raise ValueError(
            f"frame selection rejected: good_candidate_views={len(good_indices)} < {min_good}, "
            f"candidate_views={candidate_count}"
        )

    if args.frame_selection_policy == "first_good":
        selected = good_indices[: int(args.selected_views)]
    elif args.frame_selection_policy == "mask_pose_diverse":
        remaining = set(good_indices)
        first = max(good_indices, key=lambda idx: (float(quality_scores[idx]), -idx))
        selected = [int(first)]
        remaining.remove(first)
        while len(selected) < int(args.selected_views):
            best_idx = None
            best_score = -1e9
            for idx in sorted(remaining):
                diversity = min(camera_pose_distance(c2w, idx, sel) for sel in selected)
                score = float(quality_scores[idx]) + float(args.selection_diversity_weight) * diversity
                if score > best_score:
                    best_idx = idx
                    best_score = score
            if best_idx is None:
                break
            selected.append(int(best_idx))
            remaining.remove(best_idx)
    else:
        raise ValueError(f"unknown frame_selection_policy: {args.frame_selection_policy}")

    if len(selected) < int(args.selected_views):
        raise ValueError(
            f"frame selection rejected: selected={len(selected)} < {args.selected_views}, "
            f"candidate_views={candidate_count}, good_candidate_views={len(good_indices)}"
        )
    selected = sorted(selected)
    return selected, {
        "policy": args.frame_selection_policy,
        "candidate_views": int(candidate_count),
        "selected_views": int(len(selected)),
        "good_candidate_views": int(len(good_indices)),
        "selected_indices": [int(x) for x in selected],
        "quality_score_mean": float(np.mean(quality_scores[selected])) if selected else 0.0,
        "quality_score_min": float(np.min(quality_scores[selected])) if selected else 0.0,
        "frame_stats": frame_stats,
    }


def validate_sample_quality(
    uid: str,
    render_stats: dict,
    albedo_stats: dict,
    geom_stats: dict,
    projection_stats: dict,
    accepted_count: int,
    accepted_low_texture_count: int,
    args: argparse.Namespace,
) -> dict:
    if render_stats["fg_pixels"] < args.min_fg_pixels:
        raise ValueError(f"too few foreground pixels for {uid}: {render_stats['fg_pixels']} < {args.min_fg_pixels}")
    if render_stats["fg_pixels_per_view_min"] < args.min_fg_pixels_per_view:
        raise ValueError(
            f"too few foreground pixels in at least one view for {uid}: "
            f"{render_stats['fg_pixels_per_view_min']} < {args.min_fg_pixels_per_view}"
        )
    if render_stats["fg_area_ratio_mean"] < args.min_fg_area_ratio:
        raise ValueError(
            f"foreground area ratio too small for {uid}: "
            f"{render_stats['fg_area_ratio_mean']:.4f} < {args.min_fg_area_ratio}"
        )
    if args.max_fg_area_ratio > 0 and render_stats["fg_area_ratio_mean"] > args.max_fg_area_ratio:
        raise ValueError(
            f"foreground area ratio too large for {uid}: "
            f"{render_stats['fg_area_ratio_mean']:.4f} > {args.max_fg_area_ratio}"
        )
    if args.min_bbox_margin_px > 0:
        tight_views = sum(
            int(x) < int(args.min_bbox_margin_px)
            for x in render_stats.get("bbox_margin_px_per_view", [])
        )
        if tight_views > args.max_border_touch_views:
            raise ValueError(
                f"framing rejected for {uid}: "
                f"bbox_margin_px_min={render_stats.get('bbox_margin_px_min', 0)} "
                f"views_below_margin={tight_views} > {args.max_border_touch_views}, "
                f"min_bbox_margin_px={args.min_bbox_margin_px}"
            )
    if args.max_bbox_area_ratio > 0 and render_stats.get("bbox_area_ratio_max", 0.0) > args.max_bbox_area_ratio:
        raise ValueError(
            f"framing rejected for {uid}: "
            f"bbox_area_ratio_max={render_stats['bbox_area_ratio_max']:.4f} > {args.max_bbox_area_ratio}"
        )
    if render_stats["rgb_std_mean"] < args.min_rgb_std:
        raise ValueError(
            f"low rendered foreground rgb std for {uid}: "
            f"{render_stats['rgb_std_mean']:.2f} < {args.min_rgb_std}"
        )
    if render_stats["unique_colors16"] < args.min_unique_colors16:
        raise ValueError(
            f"too few rendered foreground colors for {uid}: "
            f"{render_stats['unique_colors16']} < {args.min_unique_colors16}"
        )
    if geom_stats["extent_ratio"] < args.min_extent_ratio:
        raise ValueError(f"thin/planar geometry for {uid}: extent_ratio={geom_stats['extent_ratio']:.4f}")
    low_projection_support = (
        projection_stats["visible_points_ratio"] < args.min_projection_visible_points_ratio
        or projection_stats["support_ratio_mean"] < args.min_projection_support_ratio_mean
        or projection_stats["zero_support_ratio"] > args.max_projection_zero_support_ratio
    )
    if args.enforce_projection_support and projection_stats["visible_points_ratio"] < args.min_projection_visible_points_ratio:
        raise ValueError(
            f"low projection visible target ratio for {uid}: "
            f"{projection_stats['visible_points_ratio']:.3f} < {args.min_projection_visible_points_ratio}"
        )
    if args.enforce_projection_support and projection_stats["support_ratio_mean"] < args.min_projection_support_ratio_mean:
        raise ValueError(
            f"low projection support ratio for {uid}: "
            f"{projection_stats['support_ratio_mean']:.3f} < {args.min_projection_support_ratio_mean}"
        )
    if args.enforce_projection_support and projection_stats["zero_support_ratio"] > args.max_projection_zero_support_ratio:
        raise ValueError(
            f"too many target coords never hit masks for {uid}: "
            f"{projection_stats['zero_support_ratio']:.3f} > {args.max_projection_zero_support_ratio}"
        )

    low_texture = (
        albedo_stats["rgb_std_mean"] < args.low_texture_rgb_std
        or albedo_stats["unique_colors16"] < args.low_texture_unique_colors16
        or albedo_stats["channel_spread"] < args.low_texture_channel_spread
    )
    flat_gray_blob = low_texture and (
        render_stats["rgb_std_mean"] < args.min_shaded_rgb_std_for_low_texture
        or render_stats["unique_colors16"] < args.min_shaded_unique_colors16_for_low_texture
        or geom_stats["num_voxels"] < args.min_voxels_for_low_texture
    )
    if flat_gray_blob:
        raise ValueError(
            f"flat low-texture sample for {uid}: "
            f"albedo_std={albedo_stats['rgb_std_mean']:.2f}, "
            f"render_std={render_stats['rgb_std_mean']:.2f}, "
            f"unique16={render_stats['unique_colors16']}, voxels={geom_stats['num_voxels']}"
        )

    if low_texture and 0 <= args.max_low_texture_ratio < 1:
        new_total = accepted_count + 1
        new_low = accepted_low_texture_count + 1
        if new_total >= args.low_texture_quota_warmup and new_low / max(new_total, 1) > args.max_low_texture_ratio:
            raise ValueError(
                f"low-texture quota exceeded for {uid}: "
                f"{new_low}/{new_total} > {args.max_low_texture_ratio}"
            )

    return {
        "low_texture": bool(low_texture),
        "flat_gray_blob": bool(flat_gray_blob),
        "low_projection_support": bool(low_projection_support),
        "renderer": args.renderer,
        "masked_rgb": True,
    }


def classify_failure(error: str) -> str:
    lowered = error.lower()
    if "frame selection rejected" in lowered:
        return "frame_selection_rejected"
    if "flat low-texture" in lowered:
        return "flat_gray_blob_rejected"
    if "low-texture quota" in lowered:
        return "low_texture_quota_rejected"
    if "rgb std" in lowered or "foreground colors" in lowered:
        return "appearance_rejected"
    if "framing rejected" in lowered:
        return "framing_rejected"
    if "foreground" in lowered or "empty mask" in lowered:
        return "foreground_rejected"
    if "thin/planar" in lowered or "occupied voxels" in lowered:
        return "geometry_rejected"
    if "projection" in lowered or "hit masks" in lowered:
        return "projection_rejected"
    if "blender" in lowered or "render" in lowered:
        return "renderer_failed"
    if "cuda" in lowered or "encoder" in lowered or "pretrained" in lowered:
        return "encoder_failed"
    return "other_failed"


def make_contact_sheet(frames: list[tuple[np.ndarray, np.ndarray]], out_path: Path) -> None:
    tiles = []
    for image, alpha in frames:
        rgba = np.concatenate([image, alpha[..., None]], axis=-1)
        tile = Image.fromarray(rgba, mode="RGBA").convert("RGB")
        bg = Image.new("RGB", tile.size, (0, 0, 0))
        bg.paste(tile)
        tiles.append(bg)
    if not tiles:
        return
    w, h = tiles[0].size
    sheet = Image.new("RGB", (w * len(tiles), h), (20, 20, 20))
    for i, tile in enumerate(tiles):
        sheet.paste(tile, (i * w, 0))
    sheet.save(out_path)


def load_encoder(args: argparse.Namespace):
    import pixal3d.models as models

    encoder = models.from_pretrained(args.encoder_pretrained).eval().to(args.device)
    for param in encoder.parameters():
        param.requires_grad_(False)
    return encoder


@torch.no_grad()
def encode_sparse_latent(
    encoder,
    coords: np.ndarray,
    resolution: int,
    device: str,
    save_dtype: str,
) -> np.ndarray:
    ss = torch.zeros(1, int(resolution), int(resolution), int(resolution), dtype=torch.float32)
    coords_t = torch.from_numpy(coords.astype(np.int64, copy=False))
    ss[0, coords_t[:, 0], coords_t[:, 1], coords_t[:, 2]] = 1.0
    z = encoder(ss[None].to(device), sample_posterior=False)
    z_np = z[0].detach().cpu().numpy()
    if save_dtype == "float16":
        z_np = z_np.astype(np.float16)
    elif save_dtype == "float32":
        z_np = z_np.astype(np.float32)
    else:
        raise ValueError(f"unsupported latent save dtype: {save_dtype}")
    return z_np


def build_sample(
    uid: str,
    glb_path: str,
    output_dir: Path,
    code_output_dir: Path,
    encoder,
    rastctx,
    args: argparse.Namespace,
    rng: np.random.Generator,
    sequence_idx: int,
    sample_ordinal: int,
    accepted_count: int,
    accepted_low_texture_count: int,
) -> dict:
    meshes = load_meshes(glb_path)
    mesh = trimesh.util.concatenate(meshes)
    vertices_latent, center, scale = normalize_vertices(np.asarray(mesh.vertices, dtype=np.float32), args.canonical_margin)
    points = sample_surface_points(mesh, vertices_latent, args.surface_points)
    target_coords = coords_from_points(points, args.voxel_resolution)
    if target_coords.shape[0] < args.min_voxels:
        raise ValueError(f"too few occupied voxels: {target_coords.shape[0]}")

    geom_stats = shape_stats(vertices_latent, target_coords)
    if geom_stats["extent_ratio"] < args.min_extent_ratio:
        raise ValueError(f"thin/planar geometry: extent_ratio={geom_stats['extent_ratio']:.4f}")

    render_vertices, render_faces, render_colors, render_normals = build_render_buffers(
        meshes,
        center,
        scale,
        args.canonical_margin,
    )
    albedo_stats = color_stats(render_colors)

    intrinsic = make_intrinsic(args)
    framing_points = pixal3d_render_space(points)
    c2w, trajectory_stats = sample_camera_trajectory(framing_points, intrinsic, args, rng)

    sample_uid = uid if args.sequences_per_object == 1 else f"{uid}_seq{sequence_idx:03d}"
    image_subdir = Path(sample_uid[:2]) / sample_uid
    image_dir = output_dir / "images" / image_subdir
    mask_dir = output_dir / "masks" / image_subdir

    if args.renderer == "blender":
        images_and_masks = render_mesh_blender(glb_path, center, scale, c2w, intrinsic, args.image_size, args)
    else:
        images_and_masks = []
        for view_idx in range(args.num_views):
            image, alpha = render_mesh_rgb(
                render_vertices,
                render_faces,
                render_colors,
                render_normals,
                c2w[view_idx],
                intrinsic,
                args.image_size,
                rastctx,
                args,
            )
            images_and_masks.append((image, alpha))

    selected_indices, frame_selection_stats = select_multiview_frames(images_and_masks, c2w, args)
    selected_images_and_masks = [images_and_masks[i] for i in selected_indices]
    selected_c2w = c2w[selected_indices]
    selected_framing_stats = camera_framing_stats(framing_points, selected_c2w, intrinsic, args.image_size, args)

    alphas = [x[1] for x in images_and_masks]
    selected_alphas = [x[1] for x in selected_images_and_masks]
    stats_all_views = foreground_render_stats([x[0] for x in images_and_masks], alphas)
    stats = foreground_render_stats([x[0] for x in selected_images_and_masks], selected_alphas)
    projection_stats_all_views = projection_support_stats(
        target_coords,
        c2w,
        intrinsic,
        alphas,
        args.voxel_resolution,
        args.mask_threshold,
    )
    support_c2w, support_alphas, projection_support_frame_count = select_projection_support_views(
        selected_c2w,
        selected_alphas,
        args.projection_support_frames,
    )
    projection_stats = projection_support_stats(
        target_coords,
        support_c2w,
        intrinsic,
        support_alphas,
        args.voxel_resolution,
        args.mask_threshold,
    )
    projection_stats["frame_count"] = int(projection_support_frame_count)
    projection_stats["frame_policy"] = "selected_all" if args.projection_support_frames <= 0 else "selected_prefix"
    projection_stats_all_views["frame_count"] = int(len(alphas))
    projection_stats_all_views["frame_policy"] = "all"
    quality_flags = validate_sample_quality(
        sample_uid,
        stats,
        albedo_stats,
        geom_stats,
        projection_stats,
        accepted_count,
        accepted_low_texture_count,
        args,
    )

    z = encode_sparse_latent(encoder, target_coords, args.voxel_resolution, args.device, args.latent_dtype)

    frames = []
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    for view_idx, source_view_idx in enumerate(selected_indices):
        image, alpha = images_and_masks[source_view_idx]
        image_rel = image_subdir / f"view_{view_idx:03d}.png"
        mask_rel = image_subdir / f"view_{view_idx:03d}.png"
        Image.fromarray(image).save(output_dir / "images" / image_rel)
        Image.fromarray(alpha).save(output_dir / "masks" / mask_rel)
        frames.append(
            {
                "image": str(image_rel),
                "mask": str(mask_rel),
                "intrinsic": intrinsic.tolist(),
                "extrinsic": c2w[source_view_idx].astype(np.float32).tolist(),
                "source_view_index": int(source_view_idx),
            }
        )

    latent_rel = Path(sample_uid[:2]) / f"{sample_uid}.npz"
    latent_path = output_dir / "ss_latents" / latent_rel
    latent_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        latent_path,
        z=z,
        target_coords=target_coords.astype(np.int32),
        uid=np.array(sample_uid),
        object_uid=np.array(uid),
        source_glb=np.array(glb_path),
        normalize_center=center.astype(np.float32),
        normalize_scale=np.array(scale, dtype=np.float32),
        pixal3d_rotation=PIXAL3D_ROTATION.astype(np.float32),
        renderer=np.array(args.renderer),
    )

    if sample_ordinal < args.vis_count:
        (code_output_dir / "vis").mkdir(parents=True, exist_ok=True)
        make_contact_sheet(selected_images_and_masks, code_output_dir / "vis" / f"{sample_uid}.jpg")

    return {
        "uid": sample_uid,
        "object_uid": uid,
        "sequence_idx": int(sequence_idx),
        "ss_latent": str(latent_rel),
        "frames": frames,
        "source_glb": glb_path,
        "num_voxels": int(target_coords.shape[0]),
        "render_stats": stats,
        "render_stats_all_views": stats_all_views,
        "camera_trajectory_stats": trajectory_stats,
        "selected_camera_trajectory_stats": selected_framing_stats,
        "frame_selection_stats": frame_selection_stats,
        "albedo_stats": albedo_stats,
        "shape_stats": geom_stats,
        "projection_stats": projection_stats,
        "projection_stats_all_views": projection_stats_all_views,
        "quality_flags": quality_flags,
    }


def parse_manifest_items(path: Path) -> list[tuple[str, str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        if "samples" in payload:
            items = []
            for item in payload["samples"]:
                uid = str(item.get("uid", item.get("id", len(items))))
                glb = item.get("glb", item.get("path", item.get("source_glb")))
                if glb is None:
                    raise ValueError(f"manifest sample {uid} has no glb/path/source_glb")
                items.append((uid, str(glb)))
            return items
        return [(str(uid), str(glb)) for uid, glb in payload.items()]
    if isinstance(payload, list):
        items = []
        for i, item in enumerate(payload):
            if isinstance(item, str):
                items.append((Path(item).stem, item))
            else:
                uid = str(item.get("uid", item.get("id", i)))
                glb = item.get("glb", item.get("path", item.get("source_glb")))
                if glb is None:
                    raise ValueError(f"manifest sample {uid} has no glb/path/source_glb")
                items.append((uid, str(glb)))
        return items
    raise ValueError(f"unsupported manifest format: {type(payload)}")


def split_train_val(samples: list[dict], val_count: int, args: argparse.Namespace) -> tuple[list[dict], list[dict], dict]:
    val_count = min(max(int(val_count), 0), len(samples))
    if val_count <= 0:
        return samples, [], {"policy": args.split_policy, "val_count": 0, "train_count": len(samples)}

    if args.split_policy == "order":
        val_samples = samples[:val_count]
        train_samples = samples[val_count:]
    elif args.split_policy == "shuffle":
        rng = np.random.default_rng(int(args.seed) + 7919)
        order = rng.permutation(len(samples)).tolist()
        val_ids = set(int(i) for i in order[:val_count])
        val_samples = [sample for i, sample in enumerate(samples) if i in val_ids]
        train_samples = [sample for i, sample in enumerate(samples) if i not in val_ids]
    elif args.split_policy == "stratified":
        rng = np.random.default_rng(int(args.seed) + 7919)
        buckets: dict[str, list[int]] = {"low_texture": [], "normal": []}
        for idx, sample in enumerate(samples):
            flags = sample.get("quality_flags", {})
            key = "low_texture" if flags.get("low_texture", False) else "normal"
            buckets[key].append(idx)

        nonempty = [key for key, ids in buckets.items() if ids]
        raw = {key: val_count * len(buckets[key]) / max(len(samples), 1) for key in nonempty}
        quotas = {key: int(math.floor(raw[key])) for key in nonempty}

        if val_count >= len(nonempty):
            for key in nonempty:
                quotas[key] = max(quotas[key], 1)
        for key in nonempty:
            quotas[key] = min(quotas[key], len(buckets[key]))

        while sum(quotas.values()) > val_count:
            key = max(nonempty, key=lambda k: (quotas[k], -raw[k]))
            quotas[key] -= 1
        while sum(quotas.values()) < val_count:
            candidates = [key for key in nonempty if quotas[key] < len(buckets[key])]
            if not candidates:
                break
            key = max(candidates, key=lambda k: (raw[k] - math.floor(raw[k]), len(buckets[k])))
            quotas[key] += 1

        val_ids: set[int] = set()
        for key in nonempty:
            ids = list(buckets[key])
            rng.shuffle(ids)
            val_ids.update(int(i) for i in ids[: quotas[key]])
        val_samples = [sample for i, sample in enumerate(samples) if i in val_ids]
        train_samples = [sample for i, sample in enumerate(samples) if i not in val_ids]
    else:
        raise ValueError(f"unknown split_policy: {args.split_policy}")

    def _count_low(items: list[dict]) -> int:
        return sum(1 for sample in items if sample.get("quality_flags", {}).get("low_texture", False))

    return train_samples, val_samples, {
        "policy": args.split_policy,
        "val_count": len(val_samples),
        "train_count": len(train_samples),
        "val_low_texture": _count_low(val_samples),
        "train_low_texture": _count_low(train_samples),
        "total_low_texture": _count_low(samples),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build pixal3d_multiview sparse training data from Objaverse GLBs.")
    parser.add_argument("--objaverse_manifest", default="/data/Objaverse/manifest_0_5000.json")
    parser.add_argument("--output_dir", default="/data/pixal3d_multiview/objaverse_sparse_mv_artraj_s2")
    parser.add_argument("--code_output_dir", default="/home/zjr/Tracker/pixal3d_multiview/outputs/data_previews/objaverse_sparse_mv_artraj_s2")
    parser.add_argument("--max_objects", type=int, default=1000)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--shuffle_objects", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sequences_per_object", type=int, default=2)
    parser.add_argument("--val_count", type=int, default=64)
    parser.add_argument("--split_policy", choices=["stratified", "shuffle", "order"], default="stratified")
    parser.add_argument("--vis_count", type=int, default=32)

    parser.add_argument("--device", default="cuda")
    parser.add_argument("--encoder_pretrained", default="microsoft/TRELLIS-image-large/ckpts/ss_enc_conv3d_16l8_fp16")
    parser.add_argument("--latent_dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--voxel_resolution", type=int, default=64)
    parser.add_argument("--surface_points", type=int, default=160000)
    parser.add_argument("--min_voxels", type=int, default=1500)
    parser.add_argument("--canonical_margin", type=float, default=0.9)

    parser.add_argument("--num_views", type=int, default=12)
    parser.add_argument(
        "--candidate_views",
        type=int,
        default=0,
        help="Alias for num_views when using long AR-like trajectories before frame selection.",
    )
    parser.add_argument("--selected_views", type=int, default=0)
    parser.add_argument(
        "--frame_selection_policy",
        choices=["none", "first_good", "mask_pose_diverse"],
        default="none",
    )
    parser.add_argument(
        "--min_good_candidate_views",
        type=int,
        default=0,
        help="Minimum number of mask/bbox-good candidate frames before selecting. 0 means selected_views.",
    )
    parser.add_argument("--selection_min_fg_pixels_per_view", type=int, default=-1)
    parser.add_argument("--selection_min_fg_area_ratio", type=float, default=-1.0)
    parser.add_argument("--selection_min_bbox_margin_px", type=float, default=-1.0)
    parser.add_argument("--selection_max_bbox_area_ratio", type=float, default=-1.0)
    parser.add_argument("--selection_target_fg_area_ratio", type=float, default=0.06)
    parser.add_argument("--selection_margin_score_px", type=float, default=64.0)
    parser.add_argument("--selection_diversity_weight", type=float, default=0.75)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--trajectory_mode", choices=["ar_random", "ring"], default="ar_random")
    parser.add_argument("--camera_radius", type=float, default=2.0)
    parser.add_argument("--elevation_deg", type=float, default=20.0)
    parser.add_argument("--radius_min", type=float, default=1.35)
    parser.add_argument("--radius_max", type=float, default=3.0)
    parser.add_argument("--azimuth_span_min", type=float, default=25.0)
    parser.add_argument("--azimuth_span_max", type=float, default=210.0)
    parser.add_argument("--azimuth_jitter", type=float, default=4.0)
    parser.add_argument("--elevation_min", type=float, default=-20.0)
    parser.add_argument("--elevation_max", type=float, default=60.0)
    parser.add_argument("--elevation_drift", type=float, default=25.0)
    parser.add_argument("--elevation_jitter", type=float, default=5.0)
    parser.add_argument("--radius_drift", type=float, default=0.35)
    parser.add_argument("--radius_jitter", type=float, default=0.08)
    parser.add_argument("--target_jitter", type=float, default=0.12)
    parser.add_argument("--lookat_jitter", type=float, default=0.08)
    parser.add_argument("--camera_lateral_jitter", type=float, default=0.06)
    parser.add_argument("--roll_jitter", type=float, default=12.0)
    parser.add_argument("--focal_ratio", type=float, default=1.25)
    parser.add_argument("--trajectory_resample_attempts", type=int, default=24)
    parser.add_argument("--disable_camera_framing_constraint", action="store_true")
    parser.add_argument("--framing_probe_points", type=int, default=8192)
    parser.add_argument("--camera_framing_margin_px", type=float, default=12.0)
    parser.add_argument("--min_complete_view_fraction", type=float, default=0.60)
    parser.add_argument("--min_usable_view_fraction", type=float, default=0.85)
    parser.add_argument("--max_clipped_view_fraction", type=float, default=0.40)
    parser.add_argument("--min_complete_in_frame_ratio", type=float, default=0.95)
    parser.add_argument("--min_usable_in_frame_ratio", type=float, default=0.45)

    parser.add_argument("--renderer", choices=["nvdiffrast", "blender"], default="nvdiffrast")
    parser.add_argument("--shading_mode", choices=["none", "normal"], default="normal")
    parser.add_argument("--shading_ambient", type=float, default=0.42)
    parser.add_argument("--shading_diffuse", type=float, default=0.68)
    parser.add_argument("--blender_path", default=None)
    parser.add_argument("--blender_engine", choices=["CYCLES", "BLENDER_EEVEE", "BLENDER_EEVEE_NEXT"], default="CYCLES")
    parser.add_argument("--blender_samples", type=int, default=48)
    parser.add_argument("--blender_world_strength", type=float, default=0.45)
    parser.add_argument("--blender_light_energy", type=float, default=500.0)
    parser.add_argument("--blender_quiet", action="store_true")

    parser.add_argument("--mask_threshold", type=float, default=0.5)
    parser.add_argument("--min_fg_pixels", type=int, default=50000)
    parser.add_argument("--min_fg_pixels_per_view", type=int, default=2048)
    parser.add_argument("--min_fg_area_ratio", type=float, default=0.02)
    parser.add_argument("--max_fg_area_ratio", type=float, default=0.88)
    parser.add_argument("--min_bbox_margin_px", type=int, default=8)
    parser.add_argument("--max_border_touch_views", type=int, default=4)
    parser.add_argument("--max_bbox_area_ratio", type=float, default=0.95)
    parser.add_argument("--min_rgb_std", type=float, default=8.0)
    parser.add_argument("--min_unique_colors16", type=int, default=16)
    parser.add_argument("--min_extent_ratio", type=float, default=0.015)
    parser.add_argument("--enforce_projection_support", action="store_true")
    parser.add_argument(
        "--projection_support_frames",
        type=int,
        default=0,
        help=(
            "Number of prefix rendered frames used for projection-support filtering. "
            "0 uses all views. Set this to the training max_frames, e.g. 8, to avoid "
            "accepting samples whose full 12 views are good but first 8 training views are weak."
        ),
    )
    parser.add_argument("--min_projection_visible_points_ratio", type=float, default=0.8)
    parser.add_argument("--min_projection_support_ratio_mean", type=float, default=0.6)
    parser.add_argument("--max_projection_zero_support_ratio", type=float, default=0.2)
    parser.add_argument("--low_texture_rgb_std", type=float, default=8.0)
    parser.add_argument("--low_texture_unique_colors16", type=int, default=16)
    parser.add_argument("--low_texture_channel_spread", type=float, default=4.0)
    parser.add_argument("--max_low_texture_ratio", type=float, default=0.22)
    parser.add_argument("--low_texture_quota_warmup", type=int, default=200)
    parser.add_argument("--min_shaded_rgb_std_for_low_texture", type=float, default=10.0)
    parser.add_argument("--min_shaded_unique_colors16_for_low_texture", type=int, default=10)
    parser.add_argument("--min_voxels_for_low_texture", type=int, default=1500)
    return parser.parse_args()


def main() -> None:
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    torch.set_grad_enabled(False)
    torch.set_float32_matmul_precision("high")
    args = parse_args()
    if args.candidate_views > 0:
        args.num_views = int(args.candidate_views)
    if args.selected_views > 0 and args.frame_selection_policy == "none":
        args.frame_selection_policy = "mask_pose_diverse"
    if args.selected_views > args.num_views:
        raise ValueError(f"selected_views={args.selected_views} cannot exceed num_views={args.num_views}")
    random.seed(args.seed)
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    code_output_dir = Path(args.code_output_dir)
    for subdir in ("images", "masks", "ss_latents"):
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)
    code_output_dir.mkdir(parents=True, exist_ok=True)

    items = parse_manifest_items(Path(args.objaverse_manifest))
    if args.shuffle_objects:
        rng = np.random.default_rng(args.seed)
        order = rng.permutation(len(items))
        items = [items[int(i)] for i in order]
    items = items[args.start_index :]
    if args.max_objects > 0:
        items = items[: args.max_objects]

    encoder = load_encoder(args)
    rastctx = make_nvdiffrast_context() if args.renderer == "nvdiffrast" else None
    samples, failures = [], []
    quality_counts = {
        "accepted_low_texture": 0,
        "accepted_flat_gray_blob": 0,
        "accepted_low_projection_support": 0,
        "renderer_nvdiffrast": 0,
        "renderer_blender": 0,
    }
    failure_counts: dict[str, int] = {}
    total_tasks = len(items) * max(1, args.sequences_per_object)
    processed_tasks = 0
    pbar = tqdm(total=total_tasks, desc="Building", unit="seq", dynamic_ncols=True)
    for rank, (uid, glb_path) in enumerate(items):
        for seq_idx in range(max(1, args.sequences_per_object)):
            processed_tasks += 1
            try:
                rng = np.random.default_rng(args.seed + rank * 1009 + seq_idx)
                sample = build_sample(
                    uid,
                    glb_path,
                    output_dir,
                    code_output_dir,
                    encoder,
                    rastctx,
                    args,
                    rng,
                    seq_idx,
                    len(samples),
                    len(samples),
                    quality_counts["accepted_low_texture"],
                )
                samples.append(sample)
                flags = sample.get("quality_flags", {})
                if flags.get("low_texture", False):
                    quality_counts["accepted_low_texture"] += 1
                if flags.get("flat_gray_blob", False):
                    quality_counts["accepted_flat_gray_blob"] += 1
                if flags.get("low_projection_support", False):
                    quality_counts["accepted_low_projection_support"] += 1
                quality_counts[f"renderer_{args.renderer}"] += 1
                pbar.set_postfix(
                    {
                        "accepted": len(samples),
                        "failed": len(failures),
                        "last": "ok",
                        "uid": sample["uid"][:10],
                        "support": f"{sample['projection_stats']['support_ratio_mean']:.3f}",
                    },
                    refresh=False,
                )
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                failure_class = classify_failure(error)
                failure_counts[failure_class] = failure_counts.get(failure_class, 0) + 1
                failures.append(
                    {
                        "uid": uid,
                        "sequence_idx": int(seq_idx),
                        "source_glb": glb_path,
                        "failure_class": failure_class,
                        "error": error,
                    }
                )
                pbar.set_postfix(
                    {
                        "accepted": len(samples),
                        "failed": len(failures),
                        "last": failure_class,
                        "uid": uid[:10],
                    },
                    refresh=False,
                )
            finally:
                pbar.update(1)
    pbar.close()

    train_samples, val_samples, split_stats = split_train_val(samples, args.val_count, args)
    common = {
        "format": "pixal3d_multiview.objaverse_sparse.v1",
        "image_root": str(output_dir / "images"),
        "mask_root": str(output_dir / "masks"),
        "latent_root": str(output_dir / "ss_latents"),
        "extrinsics_type": "c2w",
        "camera_forward_sign": 1.0,
        "coordinate_frame": "pixal3d_rotated_render_space",
        "canonical_latent_frame": "pixal3d_sparse_structure",
        "num_views": int(args.selected_views if args.selected_views > 0 else args.num_views),
        "candidate_views": args.num_views,
        "image_size": args.image_size,
        "voxel_resolution": args.voxel_resolution,
        "encoder_pretrained": args.encoder_pretrained,
        "trajectory_mode": args.trajectory_mode,
        "renderer": args.renderer,
        "images_are_masked": True,
        "quality_policy": {
            "min_fg_pixels": args.min_fg_pixels,
            "min_fg_pixels_per_view": args.min_fg_pixels_per_view,
            "min_fg_area_ratio": args.min_fg_area_ratio,
            "max_fg_area_ratio": args.max_fg_area_ratio,
            "min_bbox_margin_px": args.min_bbox_margin_px,
            "max_border_touch_views": args.max_border_touch_views,
            "max_bbox_area_ratio": args.max_bbox_area_ratio,
            "trajectory_resample_attempts": args.trajectory_resample_attempts,
            "disable_camera_framing_constraint": args.disable_camera_framing_constraint,
            "framing_probe_points": args.framing_probe_points,
            "camera_framing_margin_px": args.camera_framing_margin_px,
            "min_complete_view_fraction": args.min_complete_view_fraction,
            "min_usable_view_fraction": args.min_usable_view_fraction,
            "max_clipped_view_fraction": args.max_clipped_view_fraction,
            "min_complete_in_frame_ratio": args.min_complete_in_frame_ratio,
            "min_usable_in_frame_ratio": args.min_usable_in_frame_ratio,
            "selected_views": args.selected_views,
            "frame_selection_policy": args.frame_selection_policy,
            "min_good_candidate_views": args.min_good_candidate_views,
            "selection_min_fg_pixels_per_view": args.selection_min_fg_pixels_per_view,
            "selection_min_fg_area_ratio": args.selection_min_fg_area_ratio,
            "selection_min_bbox_margin_px": args.selection_min_bbox_margin_px,
            "selection_max_bbox_area_ratio": args.selection_max_bbox_area_ratio,
            "selection_target_fg_area_ratio": args.selection_target_fg_area_ratio,
            "selection_margin_score_px": args.selection_margin_score_px,
            "selection_diversity_weight": args.selection_diversity_weight,
            "min_rgb_std": args.min_rgb_std,
            "min_unique_colors16": args.min_unique_colors16,
            "min_extent_ratio": args.min_extent_ratio,
            "enforce_projection_support": args.enforce_projection_support,
            "projection_support_frames": args.projection_support_frames,
            "min_projection_visible_points_ratio": args.min_projection_visible_points_ratio,
            "min_projection_support_ratio_mean": args.min_projection_support_ratio_mean,
            "max_projection_zero_support_ratio": args.max_projection_zero_support_ratio,
            "low_texture_rgb_std": args.low_texture_rgb_std,
            "low_texture_unique_colors16": args.low_texture_unique_colors16,
            "low_texture_channel_spread": args.low_texture_channel_spread,
            "max_low_texture_ratio": args.max_low_texture_ratio,
            "min_shaded_rgb_std_for_low_texture": args.min_shaded_rgb_std_for_low_texture,
            "min_shaded_unique_colors16_for_low_texture": args.min_shaded_unique_colors16_for_low_texture,
            "min_voxels_for_low_texture": args.min_voxels_for_low_texture,
            "split_policy": args.split_policy,
        },
        "split_stats": split_stats,
        "quality_counts": quality_counts,
        "failure_counts": failure_counts,
        "samples": samples,
        "failures": failures,
    }
    (output_dir / "manifest.json").write_text(json.dumps(common, indent=2), encoding="utf-8")
    (output_dir / "train.json").write_text(json.dumps({**common, "samples": train_samples}, indent=2), encoding="utf-8")
    (output_dir / "val.json").write_text(json.dumps({**common, "samples": val_samples}, indent=2), encoding="utf-8")

    summary = {
        "output_dir": str(output_dir),
        "code_output_dir": str(code_output_dir),
        "num_samples": len(samples),
        "num_failures": len(failures),
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "num_views": int(args.selected_views if args.selected_views > 0 else args.num_views),
        "candidate_views": args.num_views,
        "selected_views": args.selected_views,
        "frame_selection_policy": args.frame_selection_policy,
        "image_size": args.image_size,
        "trajectory_mode": args.trajectory_mode,
        "renderer": args.renderer,
        "latent_dtype": args.latent_dtype,
        "quality_counts": quality_counts,
        "failure_counts": failure_counts,
        "split_stats": split_stats,
        "accepted_low_texture_ratio": float(quality_counts["accepted_low_texture"] / max(len(samples), 1)),
    }
    (code_output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    if not samples:
        raise SystemExit("[build] no samples generated; check failures in manifest.json")


if __name__ == "__main__":
    main()
