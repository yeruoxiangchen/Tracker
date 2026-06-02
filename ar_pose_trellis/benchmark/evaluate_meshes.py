#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFont


TRACKER_ROOT = Path(__file__).resolve().parents[2]


def load_testsets(path: str) -> list[dict]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    datasets = payload.get("datasets", payload if isinstance(payload, list) else None)
    if not datasets:
        raise ValueError(f"No datasets found in {path}")
    return datasets


def first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path and path.exists():
            return path
    return None


def latest_reconviagen_nested(root: Path) -> Path | None:
    if not root.exists():
        return None
    direct = first_existing(
        [
            root / "reconstructed_mesh.obj",
            root / "reconstructed_object.glb",
            root / "reconstructed_object.ply",
        ]
    )
    if direct is not None:
        return direct
    candidates = []
    for child in root.iterdir():
        if not child.is_dir():
            continue
        mesh = first_existing(
            [
                child / "reconstructed_mesh.obj",
                child / "reconstructed_object.glb",
                child / "reconstructed_object.ply",
            ]
        )
        if mesh is not None:
            candidates.append(mesh)
    return sorted(candidates)[-1] if candidates else None


def find_dataset_model(dataset_dir: Path, dataset_name: str) -> Path | None:
    model_dir = dataset_dir / "models"
    if not model_dir.exists():
        return None
    preferred = [
        model_dir / f"{dataset_name}_norm.obj",
        model_dir / f"{dataset_name}.obj",
    ]
    found = first_existing(preferred)
    if found is not None:
        return found
    norm_models = sorted(model_dir.glob("*_norm.obj"))
    if norm_models:
        return norm_models[0]
    models = sorted(model_dir.glob("*.obj"))
    return models[0] if models else None


def resolve_method_paths(item: dict, args: argparse.Namespace) -> dict[str, Path]:
    name = item["name"]
    dataset_dir = Path(item["dataset_dir"]).resolve() if item.get("dataset_dir") else None
    paths: dict[str, Path] = {}

    if item.get("reference_coords"):
        paths["reference"] = Path(item["reference_coords"]).resolve()
    elif item.get("reference_mesh"):
        paths["reference"] = Path(item["reference_mesh"]).resolve()
    elif dataset_dir is not None:
        ref = find_dataset_model(dataset_dir, name)
        if ref is not None:
            paths["reference"] = ref

    if args.arpose_root:
        ar_root = Path(args.arpose_root) / name
        ar_path = first_existing(
            [
                ar_root / "reconstructed_mesh.obj",
                ar_root / "reconstructed_object.glb",
                ar_root / "reconstructed_object.ply",
                ar_root / "gaussian_point_cloud.ply",
            ]
        )
        if ar_path is not None:
            paths["arpose"] = ar_path

    if args.reconviagen_root:
        recon_path = latest_reconviagen_nested(Path(args.reconviagen_root) / name)
        if recon_path is not None:
            paths["reconviagen"] = recon_path
    elif dataset_dir is not None:
        recon_path = latest_reconviagen_nested(dataset_dir / "reconviagen_output")
        if recon_path is not None:
            paths["reconviagen"] = recon_path

    return {k: v for k, v in paths.items() if v is not None and v.exists()}


def load_target_coords(path: Path) -> np.ndarray:
    loaded = np.load(path)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        try:
            coords = loaded["target_coords"]
        finally:
            loaded.close()
    else:
        coords = loaded
    coords = np.asarray(coords)
    if coords.ndim != 2 or coords.shape[1] < 3:
        raise ValueError(f"Expected target coords with shape [N,3+] in {path}, got {coords.shape}")
    xyz = coords[:, -3:].astype(np.float32)
    xyz = xyz[np.isfinite(xyz).all(axis=1)]
    if xyz.shape[0] == 0:
        raise ValueError(f"No finite target coords in {path}")
    max_coord = float(np.max(xyz))
    min_coord = float(np.min(xyz))
    if min_coord >= 0.0 and max_coord > 2.0:
        resolution = max(64.0, max_coord + 1.0)
        xyz = (xyz + 0.5) / resolution - 0.5
    return xyz.astype(np.float32)


def load_geometry(path: Path) -> tuple[np.ndarray, trimesh.Trimesh | None]:
    if path.suffix.lower() in {".npy", ".npz"}:
        return load_target_coords(path), None

    loaded = trimesh.load(str(path), force="scene", process=False)
    meshes = []
    points = []
    if isinstance(loaded, trimesh.Scene):
        for geom in loaded.geometry.values():
            if isinstance(geom, trimesh.Trimesh) and len(geom.vertices) > 0:
                if geom.faces is not None and len(geom.faces) > 0:
                    meshes.append(geom)
                else:
                    points.append(np.asarray(geom.vertices))
            elif isinstance(geom, trimesh.points.PointCloud):
                points.append(np.asarray(geom.vertices))
    elif isinstance(loaded, trimesh.Trimesh):
        if loaded.faces is not None and len(loaded.faces) > 0:
            meshes.append(loaded)
        else:
            points.append(np.asarray(loaded.vertices))
    elif isinstance(loaded, trimesh.points.PointCloud):
        points.append(np.asarray(loaded.vertices))
    else:
        raise ValueError(f"Unsupported geometry type {type(loaded)} for {path}")

    mesh = trimesh.util.concatenate(meshes) if meshes else None
    if mesh is not None:
        verts = np.asarray(mesh.vertices, dtype=np.float32)
    elif points:
        verts = np.concatenate(points, axis=0).astype(np.float32)
    else:
        raise ValueError(f"No vertices in {path}")
    verts = verts[np.isfinite(verts).all(axis=1)]
    if verts.shape[0] == 0:
        raise ValueError(f"No finite vertices in {path}")
    return verts, mesh


def normalize_points(points: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    pmin = points.min(axis=0)
    pmax = points.max(axis=0)
    center = (pmin + pmax) * 0.5
    extent = pmax - pmin
    scale = float(np.max(extent))
    if not np.isfinite(scale) or scale <= 1e-8:
        scale = 1.0
    normed = (points - center) / scale
    return normed.astype(np.float32), {
        "bbox_min": pmin.tolist(),
        "bbox_max": pmax.tolist(),
        "bbox_extent": extent.tolist(),
        "bbox_scale": scale,
        "bbox_extent_sorted": sorted([float(x) for x in extent.tolist()], reverse=True),
        "extent_balance": float(np.min(extent) / (np.max(extent) + 1e-8)),
    }


def sample_points(vertices: np.ndarray, mesh: trimesh.Trimesh | None, count: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if mesh is not None and mesh.faces is not None and len(mesh.faces) > 0:
        try:
            pts, _ = trimesh.sample.sample_surface(mesh, count)
            pts, _ = normalize_points(np.asarray(pts, dtype=np.float32))
            return pts
        except Exception:
            pass
    pts, _ = normalize_points(vertices)
    if pts.shape[0] <= count:
        return pts
    ids = rng.choice(pts.shape[0], size=count, replace=False)
    return pts[ids]


def nearest_distances(src: np.ndarray, dst: np.ndarray, chunk: int = 512) -> np.ndarray:
    out = np.empty((src.shape[0],), dtype=np.float32)
    dst = dst.astype(np.float32)
    for start in range(0, src.shape[0], chunk):
        sub = src[start : start + chunk].astype(np.float32)
        diff = sub[:, None, :] - dst[None, :, :]
        d2 = np.sum(diff * diff, axis=-1)
        out[start : start + chunk] = np.sqrt(np.min(d2, axis=1))
    return out


def compare_points(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    da = nearest_distances(a, b)
    db = nearest_distances(b, a)
    out = {
        "chamfer_l1": float(0.5 * (da.mean() + db.mean())),
        "a_to_b_mean": float(da.mean()),
        "b_to_a_mean": float(db.mean()),
        "a_to_b_p95": float(np.percentile(da, 95)),
        "b_to_a_p95": float(np.percentile(db, 95)),
    }
    for threshold in [0.01, 0.02, 0.05, 0.10]:
        precision = float((da < threshold).mean())
        recall = float((db < threshold).mean())
        denom = precision + recall
        out[f"fscore_{threshold:.2f}"] = float(2.0 * precision * recall / denom) if denom > 0 else 0.0
    return out


def geometry_stats(path: Path, sample_count: int, seed: int) -> dict[str, Any]:
    vertices, mesh = load_geometry(path)
    norm_vertices, bbox = normalize_points(vertices)
    stats: dict[str, Any] = {
        "path": str(path),
        "vertex_count": int(vertices.shape[0]),
        **bbox,
    }
    if mesh is not None:
        stats.update(
            {
                "face_count": int(len(mesh.faces)) if mesh.faces is not None else 0,
                "is_watertight": bool(mesh.is_watertight),
                "component_count": int(len(mesh.split(only_watertight=False))) if mesh.faces is not None else 0,
                "surface_area": float(mesh.area) if np.isfinite(mesh.area) else None,
                "volume": float(mesh.volume) if mesh.is_watertight and np.isfinite(mesh.volume) else None,
            }
        )
    else:
        stats.update({"face_count": 0, "is_watertight": False, "component_count": 0, "surface_area": None, "volume": None})
    stats["_sample_points"] = sample_points(vertices, mesh, sample_count, seed)
    stats["_norm_vertices"] = norm_vertices
    return stats


def draw_projection(points: np.ndarray, title: str, size: int = 240) -> Image.Image:
    image = Image.new("RGB", (size * 3, size + 28), (250, 250, 250))
    draw = ImageDraw.Draw(image)
    draw.text((8, 6), title, fill=(10, 10, 10))
    projections = [(0, 1, "xy"), (0, 2, "xz"), (1, 2, "yz")]
    colors = [(28, 92, 185), (200, 82, 20), (28, 130, 70)]
    pts = np.clip(points, -0.55, 0.55)
    for tile, (a, b, label) in enumerate(projections):
        x0 = tile * size
        draw.rectangle((x0, 28, x0 + size - 1, size + 27), outline=(210, 210, 210))
        draw.text((x0 + 6, 30), label, fill=(80, 80, 80))
        xy = pts[:, [a, b]]
        pix = ((xy + 0.55) / 1.10 * (size - 12) + 6).astype(np.int32)
        pix[:, 1] = size - 1 - pix[:, 1] + 28
        if pix.shape[0] > 4000:
            pix = pix[np.linspace(0, pix.shape[0] - 1, 4000).astype(np.int32)]
        color = colors[tile]
        for x, y in pix:
            if x0 <= x + x0 < x0 + size and 28 <= y < size + 28:
                draw.point((x0 + int(x), int(y)), fill=color)
    return image


def make_contact_sheet(dataset_name: str, method_stats: dict[str, dict], out_path: Path) -> None:
    tiles = []
    for method, stats in method_stats.items():
        points = stats["_sample_points"]
        tiles.append(draw_projection(points, f"{dataset_name} | {method}"))
    if not tiles:
        return
    w = max(tile.width for tile in tiles)
    h = sum(tile.height for tile in tiles)
    sheet = Image.new("RGB", (w, h), (255, 255, 255))
    y = 0
    for tile in tiles:
        sheet.paste(tile, (0, y))
        y += tile.height
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


def strip_private(stats: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in stats.items() if not k.startswith("_")}


def evaluate_dataset(item: dict, args: argparse.Namespace, out_dir: Path) -> dict[str, Any]:
    name = item["name"]
    method_paths = resolve_method_paths(item, args)
    method_stats = {}
    errors = {}
    for method, path in method_paths.items():
        try:
            method_stats[method] = geometry_stats(path, args.sample_points, args.seed)
        except Exception as exc:
            errors[method] = f"{type(exc).__name__}: {exc}"

    pairwise = {}
    methods = sorted(method_stats)
    for i, left in enumerate(methods):
        for right in methods[i + 1 :]:
            pairwise[f"{left}__vs__{right}"] = compare_points(
                method_stats[left]["_sample_points"],
                method_stats[right]["_sample_points"],
            )

    if method_stats:
        make_contact_sheet(name, method_stats, out_dir / name / "shape_contact_sheet.png")

    return {
        "name": name,
        "note": item.get("note"),
        "method_paths": {k: str(v) for k, v in method_paths.items()},
        "method_stats": {k: strip_private(v) for k, v in method_stats.items()},
        "pairwise": pairwise,
        "errors": errors,
        "visualization": str(out_dir / name / "shape_contact_sheet.png") if method_stats else None,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--testsets", default=str(TRACKER_ROOT / "ar_pose_trellis" / "benchmark" / "testsets.json"))
    parser.add_argument("--arpose_root", default=str(TRACKER_ROOT / "ar_pose_trellis" / "benchmark_outputs" / "arpose"))
    parser.add_argument("--reconviagen_root", default=None, help="Optional controlled ReconViaGen output root. Defaults to dataset/reconviagen_output.")
    parser.add_argument("--output_dir", default=str(TRACKER_ROOT / "ar_pose_trellis" / "benchmark_outputs" / "eval"))
    parser.add_argument("--sample_points", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    reports = [evaluate_dataset(item, args, out_dir) for item in load_testsets(args.testsets)]
    summary = {
        "testsets": args.testsets,
        "arpose_root": args.arpose_root,
        "reconviagen_root": args.reconviagen_root,
        "reports": reports,
    }
    (out_dir / "benchmark_report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[evaluate_meshes] wrote {out_dir / 'benchmark_report.json'}")
    for report in reports:
        methods = ", ".join(sorted(report["method_stats"].keys())) or "none"
        print(f"[evaluate_meshes] {report['name']}: methods={methods} vis={report['visualization']}")


if __name__ == "__main__":
    main()
