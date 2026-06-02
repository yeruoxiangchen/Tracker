#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image


def load_meshes(path: str) -> list[trimesh.Trimesh]:
    obj = trimesh.load(path, force="scene", process=False)
    if isinstance(obj, trimesh.Scene):
        dumped = obj.dump(concatenate=False)
        meshes = [g for g in dumped if isinstance(g, trimesh.Trimesh) and len(g.vertices) > 0]
        if not meshes:
            raise ValueError("scene has no mesh geometry")
    elif isinstance(obj, trimesh.Trimesh):
        meshes = [obj]
    else:
        raise ValueError(f"unsupported object type: {type(obj)}")
    for mesh in meshes:
        if mesh.faces is None or len(mesh.faces) == 0 or len(mesh.vertices) == 0:
            raise ValueError("empty mesh")
        mesh.remove_unreferenced_vertices()
    return meshes


def load_mesh(path: str) -> trimesh.Trimesh:
    meshes = load_meshes(path)
    mesh = trimesh.util.concatenate(meshes)
    if mesh.faces is None or len(mesh.faces) == 0 or len(mesh.vertices) == 0:
        raise ValueError("empty mesh")
    mesh.remove_unreferenced_vertices()
    return mesh


def normalize_vertices(vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    vmin = vertices.min(axis=0)
    vmax = vertices.max(axis=0)
    center = (vmin + vmax) * 0.5
    scale = float(np.max(vmax - vmin))
    if not np.isfinite(scale) or scale <= 1e-8:
        raise ValueError("invalid mesh scale")
    # Keep a small margin inside the TRELLIS 64^3 volume.
    normalized = (vertices - center) / scale * 0.9
    return normalized.astype(np.float32), center.astype(np.float32), scale


def sample_points(mesh: trimesh.Trimesh, vertices_norm: np.ndarray, num_points: int) -> tuple[np.ndarray, np.ndarray]:
    local_mesh = mesh.copy()
    local_mesh.vertices = vertices_norm
    points, face_ids = trimesh.sample.sample_surface(local_mesh, num_points)
    normals = local_mesh.face_normals[face_ids]
    return points.astype(np.float32), normals.astype(np.float32)


def normal_colors(normals: np.ndarray) -> np.ndarray:
    colors = 0.55 + 0.45 * normals
    return np.clip(colors * 255.0, 0, 255).astype(np.uint8)


def coords_from_points(points: np.ndarray, resolution: int = 64) -> np.ndarray:
    coords = np.floor((points + 0.5) * resolution).astype(np.int32)
    coords = np.clip(coords, 0, resolution - 1)
    coords = np.unique(coords, axis=0)
    batch = np.zeros((coords.shape[0], 1), dtype=np.int32)
    return np.concatenate([batch, coords.astype(np.int32)], axis=1)


def look_at_c2w(eye: np.ndarray, target: np.ndarray = None, roll_rad: float = 0.0) -> np.ndarray:
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
    """Generate a short AR-like camera trajectory around a canonical object.

    The object remains in canonical space for the target sparse coords. These
    cameras only simulate real phone capture: partial sweeps, forward/backward
    motion, target offset, and roll/pitch/yaw jitter.
    """

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

    cameras = []
    if num_views <= 1:
        ts = np.array([0.0], dtype=np.float32)
    else:
        ts = np.linspace(-0.5, 0.5, num_views, dtype=np.float32)
    for t in ts:
        smooth = float(t)
        az = az0 + smooth * az_span + math.radians(float(rng.normal(0.0, args.azimuth_jitter)))
        elev = elev0 + smooth * elev_drift + math.radians(float(rng.normal(0.0, args.elevation_jitter)))
        radius = base_radius * (1.0 + smooth * radial_drift + float(rng.normal(0.0, args.radius_jitter)))
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


def rasterize_points(
    points: np.ndarray,
    colors: np.ndarray,
    c2w: np.ndarray,
    intrinsic: np.ndarray,
    image_size: int,
    point_radius: int,
) -> tuple[np.ndarray, np.ndarray]:
    w2c = np.linalg.inv(c2w).astype(np.float32)
    pts_h = np.concatenate([points, np.ones((points.shape[0], 1), dtype=np.float32)], axis=1)
    cam = (w2c @ pts_h.T).T[:, :3]
    z = cam[:, 2]
    valid = z > 1e-4
    cam = cam[valid]
    z = z[valid]
    cols = colors[valid]
    if cam.shape[0] == 0:
        raise ValueError("no visible points")

    u = intrinsic[0, 0] * cam[:, 0] / z + intrinsic[0, 2]
    v = intrinsic[1, 1] * cam[:, 1] / z + intrinsic[1, 2]
    ui = np.round(u).astype(np.int32)
    vi = np.round(v).astype(np.int32)
    inside = (ui >= 0) & (ui < image_size) & (vi >= 0) & (vi < image_size)
    ui, vi, z, cols = ui[inside], vi[inside], z[inside], cols[inside]

    image = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    alpha = np.zeros((image_size, image_size), dtype=np.uint8)
    depth = np.full((image_size, image_size), np.inf, dtype=np.float32)
    order = np.argsort(z)[::-1]
    rr = max(point_radius, 0)
    for idx in order:
        x = ui[idx]
        y = vi[idx]
        x0 = max(0, x - rr)
        x1 = min(image_size, x + rr + 1)
        y0 = max(0, y - rr)
        y1 = min(image_size, y + rr + 1)
        patch = depth[y0:y1, x0:x1] > z[idx]
        if not patch.any():
            continue
        image[y0:y1, x0:x1][patch] = cols[idx]
        alpha[y0:y1, x0:x1][patch] = 255
        depth[y0:y1, x0:x1][patch] = z[idx]
    return image, alpha


def _sample_texture_nearest(image: Image.Image, uv: np.ndarray) -> np.ndarray:
    tex = np.asarray(image.convert("RGB"))
    if tex.ndim != 3 or tex.shape[0] == 0 or tex.shape[1] == 0:
        raise ValueError("empty texture image")
    uv = np.asarray(uv, dtype=np.float32)
    u = np.mod(uv[:, 0], 1.0)
    # Most glTF UVs use a bottom-left origin, while PIL arrays are top-left.
    v = 1.0 - np.mod(uv[:, 1], 1.0)
    x = np.clip(np.round(u * (tex.shape[1] - 1)).astype(np.int64), 0, tex.shape[1] - 1)
    y = np.clip(np.round(v * (tex.shape[0] - 1)).astype(np.int64), 0, tex.shape[0] - 1)
    return tex[y, x, :3].astype(np.uint8)


def _material_base_color(mesh: trimesh.Trimesh) -> np.ndarray:
    material = getattr(mesh.visual, "material", None)
    base = getattr(material, "baseColorFactor", None)
    if base is None:
        return np.array([180, 180, 180], dtype=np.uint8)
    base = np.asarray(base).reshape(-1)
    if base.size >= 3:
        if base[:3].max() <= 1.0:
            base = base[:3] * 255.0
        else:
            base = base[:3]
        return np.clip(base, 0, 255).astype(np.uint8)
    return np.array([180, 180, 180], dtype=np.uint8)


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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vertices_all, faces_all, colors_all = [], [], []
    offset = 0
    for mesh in meshes:
        vertices = (np.asarray(mesh.vertices, dtype=np.float32) - center[None]) / scale * 0.9
        faces = np.asarray(mesh.faces, dtype=np.int32) + offset
        colors = vertex_colors_from_visual(mesh)
        vertices_all.append(vertices.astype(np.float32))
        faces_all.append(faces.astype(np.int32))
        colors_all.append(colors.astype(np.uint8))
        offset += vertices.shape[0]
    if not vertices_all:
        raise ValueError("no renderable mesh buffers")
    return (
        np.concatenate(vertices_all, axis=0),
        np.concatenate(faces_all, axis=0),
        np.concatenate(colors_all, axis=0),
    )


def make_nvdiffrast_context():
    try:
        import torch
        import nvdiffrast.torch as dr
    except Exception as exc:
        raise RuntimeError(
            "mesh_rgb rendering requires torch and nvdiffrast. "
            "Use --render_mode point_normal only for legacy/debug data."
        ) from exc
    if not torch.cuda.is_available():
        raise RuntimeError(
            "mesh_rgb rendering requires CUDA for nvdiffrast in this script. "
            "Use --render_mode point_normal only for legacy/debug data."
        )
    return dr.RasterizeCudaContext()


def render_mesh_rgb(
    vertices: np.ndarray,
    faces: np.ndarray,
    vertex_colors: np.ndarray,
    c2w: np.ndarray,
    intrinsic: np.ndarray,
    image_size: int,
    rastctx,
) -> tuple[np.ndarray, np.ndarray]:
    import torch
    import nvdiffrast.torch as dr

    device = torch.device("cuda")
    verts = torch.as_tensor(vertices, dtype=torch.float32, device=device)
    faces_t = torch.as_tensor(faces.astype(np.int32), dtype=torch.int32, device=device)
    colors = torch.as_tensor(vertex_colors.astype(np.float32) / 255.0, dtype=torch.float32, device=device)

    c2w_t = torch.as_tensor(c2w, dtype=torch.float32, device=device)
    w2c = torch.linalg.inv(c2w_t)
    verts_h = torch.cat([verts, torch.ones((verts.shape[0], 1), dtype=torch.float32, device=device)], dim=1)
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
    try:
        color = dr.antialias(color, rast, pos_clip[None], faces_t)
    except Exception:
        pass
    mask = (rast[..., 3:4] > 0).float()
    color = color * mask
    image = np.clip(color[0].detach().cpu().numpy() * 255.0, 0, 255).astype(np.uint8)
    alpha = np.clip(mask[0, ..., 0].detach().cpu().numpy() * 255.0, 0, 255).astype(np.uint8)
    if int((alpha > 0).sum()) == 0:
        raise ValueError("mesh renderer produced an empty mask")
    return image, alpha


def make_contact_sheet(images: list[np.ndarray], alphas: list[np.ndarray], out_path: Path) -> None:
    tiles = []
    for image, alpha in zip(images, alphas):
        rgba = np.concatenate([image, alpha[..., None]], axis=-1)
        tile = Image.fromarray(rgba, mode="RGBA").convert("RGB")
        overlay = Image.new("RGB", tile.size, (0, 0, 0))
        overlay.paste(tile)
        tiles.append(overlay)
    if not tiles:
        return
    w, h = tiles[0].size
    sheet = Image.new("RGB", (w * len(tiles), h), (20, 20, 20))
    for i, tile in enumerate(tiles):
        sheet.paste(tile, (i * w, 0))
    sheet.save(out_path)


def foreground_render_stats(images: list[np.ndarray], alphas: list[np.ndarray]) -> dict:
    vals = []
    for image, alpha in zip(images, alphas):
        mask = alpha > 0
        if mask.any():
            vals.append(image[mask])
    if not vals:
        return {
            "fg_pixels": 0,
            "rgb_mean": [0.0, 0.0, 0.0],
            "rgb_std": [0.0, 0.0, 0.0],
            "rgb_std_mean": 0.0,
            "gray_std": 0.0,
            "channel_spread": 0.0,
            "unique_colors16": 0,
        }
    arr = np.concatenate(vals, axis=0).astype(np.float32)
    rgb_mean = arr.mean(axis=0)
    rgb_std = arr.std(axis=0)
    unique_colors16 = int(np.unique((arr.astype(np.uint8) // 16), axis=0).shape[0])
    return {
        "fg_pixels": int(arr.shape[0]),
        "rgb_mean": [float(x) for x in rgb_mean.tolist()],
        "rgb_std": [float(x) for x in rgb_std.tolist()],
        "rgb_std_mean": float(rgb_std.mean()),
        "gray_std": float(arr.mean(axis=1).std()),
        "channel_spread": float(np.abs(rgb_mean - rgb_mean.mean()).mean()),
        "unique_colors16": unique_colors16,
    }


def validate_render_stats(uid: str, stats: dict, args: argparse.Namespace) -> None:
    if stats["rgb_std_mean"] < args.min_rgb_std:
        raise ValueError(
            f"low foreground rgb std for {uid}: {stats['rgb_std_mean']:.2f} < {args.min_rgb_std}"
        )
    if stats["channel_spread"] < args.min_channel_spread:
        raise ValueError(
            f"low foreground channel spread for {uid}: {stats['channel_spread']:.2f} < {args.min_channel_spread}"
        )
    if stats["unique_colors16"] < args.min_unique_colors16:
        raise ValueError(
            f"too few foreground colors for {uid}: {stats['unique_colors16']} < {args.min_unique_colors16}"
        )


def build_sample(
    uid: str,
    glb_path: str,
    output_dir: Path,
    args: argparse.Namespace,
    rng: np.random.Generator,
    sequence_idx: int = 0,
) -> dict:
    meshes = load_meshes(glb_path)
    mesh = trimesh.util.concatenate(meshes)
    vertices_norm, center, scale = normalize_vertices(np.asarray(mesh.vertices, dtype=np.float32))
    points, normals = sample_points(mesh, vertices_norm, args.surface_points)
    target_coords = coords_from_points(points, resolution=args.voxel_resolution)
    if target_coords.shape[0] < args.min_voxels:
        raise ValueError(f"too few occupied voxels: {target_coords.shape[0]}")
    if args.render_mode == "mesh_rgb":
        render_vertices, render_faces, render_colors = build_render_buffers(meshes, center, scale)
        if not hasattr(args, "_rastctx"):
            args._rastctx = make_nvdiffrast_context()
    else:
        point_colors = normal_colors(normals)

    if args.trajectory_mode == "ring":
        c2w = make_camera_ring(args.num_views, args.camera_radius, args.elevation_deg)
    else:
        c2w = make_ar_random_trajectory(args.num_views, args, rng)
    focal = args.focal_ratio * args.image_size
    intrinsic = np.array(
        [
            [focal, 0.0, (args.image_size - 1) * 0.5],
            [0.0, focal, (args.image_size - 1) * 0.5],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    intrinsics = np.repeat(intrinsic[None], args.num_views, axis=0)

    images, alphas = [], []
    for view in range(args.num_views):
        if args.render_mode == "mesh_rgb":
            image, alpha = render_mesh_rgb(
                render_vertices,
                render_faces,
                render_colors,
                c2w[view],
                intrinsic,
                args.image_size,
                args._rastctx,
            )
        else:
            image, alpha = rasterize_points(
                points,
                point_colors,
                c2w[view],
                intrinsic,
                args.image_size,
                args.point_radius,
            )
        images.append(image)
        alphas.append(alpha)
    render_stats = foreground_render_stats(images, alphas)
    validate_render_stats(uid, render_stats, args)

    sample_uid = uid if args.sequences_per_object == 1 else f"{uid}_seq{sequence_idx:03d}"
    sample_dir = output_dir / "samples" / sample_uid[:2]
    sample_dir.mkdir(parents=True, exist_ok=True)
    npz_path = sample_dir / f"{sample_uid}.npz"
    np.savez_compressed(
        npz_path,
        uid=np.array(sample_uid),
        object_uid=np.array(uid),
        sequence_idx=np.array(sequence_idx, dtype=np.int32),
        source_glb=np.array(glb_path),
        images=np.stack(images, axis=0),
        alpha=np.stack(alphas, axis=0),
        intrinsics=intrinsics,
        extrinsics=c2w.astype(np.float32),
        target_coords=target_coords,
        normalize_center=center,
        normalize_scale=np.array(scale, dtype=np.float32),
        render_mode=np.array(args.render_mode),
        render_rgb_mean=np.asarray(render_stats["rgb_mean"], dtype=np.float32),
        render_rgb_std=np.asarray(render_stats["rgb_std"], dtype=np.float32),
        render_gray_std=np.array(render_stats["gray_std"], dtype=np.float32),
        render_channel_spread=np.array(render_stats["channel_spread"], dtype=np.float32),
        render_unique_colors16=np.array(render_stats["unique_colors16"], dtype=np.int32),
    )
    return {
        "uid": sample_uid,
        "object_uid": uid,
        "sequence_idx": int(sequence_idx),
        "npz": str(npz_path.relative_to(output_dir)),
        "source_glb": glb_path,
        "num_voxels": int(target_coords.shape[0]),
        "render_mode": args.render_mode,
        "render_stats": render_stats,
    }, images, alphas


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--objaverse_manifest", default="/data/Objaverse/manifest_0_5000.json")
    parser.add_argument("--output_dir", default="/data/ar_pose_trellis/objaverse_pose_smoke")
    parser.add_argument("--code_output_dir", default="/home/zjr/Tracker/ar_pose_trellis/outputs/data_previews/smoke_data_build")
    parser.add_argument("--max_objects", type=int, default=10)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sequences_per_object", type=int, default=1)
    parser.add_argument("--num_views", type=int, default=6)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--surface_points", type=int, default=120000)
    parser.add_argument("--voxel_resolution", type=int, default=64)
    parser.add_argument("--min_voxels", type=int, default=256)
    parser.add_argument("--camera_radius", type=float, default=2.0)
    parser.add_argument("--elevation_deg", type=float, default=20.0)
    parser.add_argument("--trajectory_mode", choices=["ar_random", "ring"], default="ar_random")
    parser.add_argument(
        "--render_mode",
        choices=["mesh_rgb", "point_normal"],
        default="mesh_rgb",
        help="mesh_rgb renders GLB mesh colors/materials. point_normal is the old normal-color point splat debug mode.",
    )
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
    parser.add_argument("--point_radius", type=int, default=1)
    parser.add_argument("--min_rgb_std", type=float, default=0.0, help="Skip samples with low foreground RGB variation.")
    parser.add_argument("--min_channel_spread", type=float, default=0.0, help="Skip samples whose foreground mean color is nearly gray.")
    parser.add_argument("--min_unique_colors16", type=int, default=1, help="Skip samples with too few quantized foreground colors.")
    parser.add_argument("--vis_count", type=int, default=8, help="Number of generated samples to write as contact sheets.")
    parser.add_argument("--shuffle_objects", action="store_true", help="Shuffle manifest entries before applying start/max limits.")
    parser.add_argument("--val_count", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    code_output_dir = Path(args.code_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (code_output_dir / "vis").mkdir(parents=True, exist_ok=True)

    manifest = json.loads(Path(args.objaverse_manifest).read_text(encoding="utf-8"))
    items = list(manifest.items())
    if args.shuffle_objects:
        rng = np.random.default_rng(args.seed)
        order = rng.permutation(len(items))
        items = [items[int(i)] for i in order]
    items = items[args.start_index :]
    if args.max_objects > 0:
        items = items[: args.max_objects]

    samples, failures = [], []
    for rank, (uid, glb_path) in enumerate(items):
        for seq_idx in range(max(1, args.sequences_per_object)):
            try:
                rng = np.random.default_rng(args.seed + rank * 1009 + seq_idx)
                sample, images, alphas = build_sample(uid, glb_path, output_dir, args, rng, sequence_idx=seq_idx)
                samples.append(sample)
                if len(samples) <= args.vis_count:
                    make_contact_sheet(images, alphas, code_output_dir / "vis" / f"{sample['uid']}.jpg")
                print(
                    f"[build] {rank + 1}/{len(items)} seq={seq_idx} ok {uid} "
                    f"voxels={sample['num_voxels']} "
                    f"rgb_std={sample['render_stats']['rgb_std_mean']:.2f} "
                    f"unique16={sample['render_stats']['unique_colors16']}",
                    flush=True,
                )
            except Exception as exc:
                failures.append(
                    {
                        "uid": uid,
                        "sequence_idx": int(seq_idx),
                        "source_glb": glb_path,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                print(f"[build] {rank + 1}/{len(items)} seq={seq_idx} failed {uid}: {exc}", flush=True)

    val_count = min(max(args.val_count, 0), len(samples))
    train_samples = samples[val_count:]
    val_samples = samples[:val_count]
    common = {
        "format": "ar_pose_trellis.objaverse_pose.v1",
        "image_size": args.image_size,
        "num_views": args.num_views,
        "extrinsics_type": "c2w",
        "trajectory_mode": args.trajectory_mode,
        "render_mode": args.render_mode,
        "pose_condition_frame": "reference_relative",
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
        "trajectory_mode": args.trajectory_mode,
        "render_mode": args.render_mode,
        "sequences_per_object": args.sequences_per_object,
        "min_rgb_std": args.min_rgb_std,
        "min_channel_spread": args.min_channel_spread,
        "min_unique_colors16": args.min_unique_colors16,
    }
    (code_output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    if len(samples) == 0:
        raise SystemExit("[build] no samples generated; check failures in manifest.json")


if __name__ == "__main__":
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    main()
