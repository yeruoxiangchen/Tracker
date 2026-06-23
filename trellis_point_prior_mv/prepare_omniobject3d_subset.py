#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

TRACKER_ROOT = Path(__file__).resolve().parents[1]
if str(TRACKER_ROOT) not in sys.path:
    sys.path.insert(0, str(TRACKER_ROOT))

from trellis_point_prior_mv.common import write_json  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
MODEL_EXTS = {".obj", ".ply", ".glb", ".gltf"}


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for row in rows for k in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def rotmat_to_qvec(rot: np.ndarray) -> list[float]:
    r = np.asarray(rot, dtype=np.float64)
    trace = float(np.trace(r))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (r[2, 1] - r[1, 2]) / s
        qy = (r[0, 2] - r[2, 0]) / s
        qz = (r[1, 0] - r[0, 1]) / s
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2.0
        qw = (r[2, 1] - r[1, 2]) / s
        qx = 0.25 * s
        qy = (r[0, 1] + r[1, 0]) / s
        qz = (r[0, 2] + r[2, 0]) / s
    elif r[1, 1] > r[2, 2]:
        s = math.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2.0
        qw = (r[0, 2] - r[2, 0]) / s
        qx = (r[0, 1] + r[1, 0]) / s
        qy = 0.25 * s
        qz = (r[1, 2] + r[2, 1]) / s
    else:
        s = math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2.0
        qw = (r[1, 0] - r[0, 1]) / s
        qx = (r[0, 2] + r[2, 0]) / s
        qy = (r[1, 2] + r[2, 1]) / s
        qz = 0.25 * s
    q = np.asarray([qw, qx, qy, qz], dtype=np.float64)
    q = q / max(float(np.linalg.norm(q)), 1e-12)
    return [float(x) for x in q.tolist()]


def list_images(path: Path) -> list[Path]:
    if not path.is_dir():
        return []
    return sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def find_first_dir(root: Path, names: tuple[str, ...]) -> Path | None:
    for name in names:
        p = root / name
        if p.is_dir() and list_images(p):
            return p
    return None


def find_image_dir(root: Path) -> Path | None:
    direct = find_first_dir(root, ("images", "rgb", "color", "colors", "views", "frames"))
    if direct is not None:
        return direct
    children = [p for p in root.iterdir() if p.is_dir()] if root.is_dir() else []
    for child in children:
        found = find_first_dir(child, ("images", "rgb", "color", "colors", "views", "frames"))
        if found is not None:
            return found
    return None


def find_mask_dir(root: Path) -> Path | None:
    bases = [root]
    if root.is_dir():
        bases.extend(p for p in root.iterdir() if p.is_dir())
    for base in bases:
        for name in ("masks", "mask", "seg", "segs", "segmentation", "alpha"):
            p = base / name
            if p.is_dir() and list_images(p):
                return p
    return None


def find_model(root: Path) -> Path | None:
    for base in (root, root / "models", root / "model", root / "mesh", root / "meshes"):
        if not base.exists():
            continue
        objs = sorted(p for p in base.iterdir() if p.is_file() and p.suffix.lower() in MODEL_EXTS)
        norm = [p for p in objs if "_norm" in p.stem.lower()]
        if norm:
            return norm[0]
        if objs:
            return objs[0]
    return None


def find_sparse(root: Path) -> Path | None:
    for rel in ("sparse/0", "colmap/sparse/0", "sparse"):
        p = root / rel
        if (p / "cameras.txt").exists() and (p / "images.txt").exists():
            return p
    return None


def find_transforms(root: Path) -> Path | None:
    for name in ("transforms.json", "transforms_train.json", "camera_transforms.json"):
        p = root / name
        if p.exists():
            return p
    for child in root.iterdir() if root.is_dir() else []:
        if child.is_dir():
            p = find_transforms(child)
            if p is not None:
                return p
    return None


def select_uniform(items: list[Path], max_count: int) -> list[Path]:
    if max_count <= 0 or len(items) <= max_count:
        return items
    ids = np.linspace(0, len(items) - 1, int(max_count))
    keep = sorted({int(round(x)) for x in ids})
    return [items[i] for i in keep][: int(max_count)]


def copy_or_make_mask(mask_dir: Path | None, image: Path, out_mask: Path, allow_full_masks: bool) -> bool:
    candidates = []
    if mask_dir is not None:
        for suffix in (".png", ".jpg", ".jpeg"):
            candidates.append(mask_dir / f"{image.stem}{suffix}")
    for src in candidates:
        if src.exists():
            out_mask.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, out_mask)
            return False
    if not allow_full_masks:
        raise FileNotFoundError(f"missing mask for {image}")
    img = Image.open(image)
    mask = Image.new("L", img.size, color=255)
    out_mask.parent.mkdir(parents=True, exist_ok=True)
    mask.save(out_mask)
    return True


def copy_model(model: Path, out_dir: Path, uid: str, convert_model_to_obj: bool) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_obj = out_dir / f"{uid}_norm.obj"
    if model.suffix.lower() == ".obj":
        shutil.copy2(model, out_obj)
        return out_obj
    if not convert_model_to_obj:
        dst = out_dir / model.name
        shutil.copy2(model, dst)
        return dst
    try:
        import trimesh
    except Exception as exc:
        raise RuntimeError(f"trimesh is required to convert {model} to obj: {exc}") from exc
    mesh = trimesh.load(model, force="mesh", process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(g for g in mesh.geometry.values()))
    mesh.export(out_obj)
    return out_obj


def write_empty_points3d(path: Path) -> None:
    path.write_text(
        "# 3D point list with one line of data per point:\n"
        "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n"
        "# Number of points: 0, mean track length: 0\n",
        encoding="utf-8",
    )


def write_colmap_from_transforms(transforms_path: Path, selected_images: list[Path], out_sparse: Path) -> bool:
    payload = load_json(transforms_path)
    frames = payload.get("frames", [])
    if not frames:
        return False
    by_name = {}
    for frame in frames:
        file_path = str(frame.get("file_path", ""))
        stem = Path(file_path).stem
        by_name[stem] = frame
    sample_image = Image.open(selected_images[0])
    width, height = sample_image.size
    fx = payload.get("fl_x")
    fy = payload.get("fl_y", fx)
    cx = payload.get("cx", width * 0.5)
    cy = payload.get("cy", height * 0.5)
    if fx is None:
        angle_x = payload.get("camera_angle_x")
        if angle_x is None:
            return False
        fx = 0.5 * width / math.tan(float(angle_x) * 0.5)
        fy = fx if fy is None else fy
    out_sparse.mkdir(parents=True, exist_ok=True)
    (out_sparse / "cameras.txt").write_text(
        "# Camera list with one line of data per camera:\n"
        "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n"
        "# Number of cameras: 1\n"
        f"1 PINHOLE {width} {height} {float(fx):.9f} {float(fy):.9f} {float(cx):.9f} {float(cy):.9f}\n",
        encoding="utf-8",
    )
    lines = [
        "# Image list with two lines of data per image:\n",
        "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, IMAGE_NAME\n",
        f"# Number of images: {len(selected_images)}, mean observations per image: 0\n",
    ]
    image_id = 1
    for image in selected_images:
        frame = by_name.get(image.stem)
        if frame is None:
            continue
        c2w = np.asarray(frame.get("transform_matrix"), dtype=np.float64).reshape(4, 4)
        w2c = np.linalg.inv(c2w)
        q = rotmat_to_qvec(w2c[:3, :3])
        t = w2c[:3, 3]
        lines.append(
            f"{image_id} {q[0]:.12f} {q[1]:.12f} {q[2]:.12f} {q[3]:.12f} "
            f"{t[0]:.12f} {t[1]:.12f} {t[2]:.12f} 1 {image.name}\n"
        )
        lines.append("\n")
        image_id += 1
    if image_id == 1:
        return False
    (out_sparse / "images.txt").write_text("".join(lines), encoding="utf-8")
    write_empty_points3d(out_sparse / "points3D.txt")
    return True


def copy_sparse(sparse: Path, out_sparse: Path) -> None:
    if out_sparse.exists():
        shutil.rmtree(out_sparse)
    out_sparse.mkdir(parents=True, exist_ok=True)
    for name in ("cameras.txt", "images.txt", "points3D.txt"):
        src = sparse / name
        if src.exists():
            shutil.copy2(src, out_sparse / name)
    if not (out_sparse / "points3D.txt").exists():
        write_empty_points3d(out_sparse / "points3D.txt")


def object_candidates(source_root: Path, scan_depth: int) -> list[Path]:
    out = []
    base_depth = len(source_root.parts)
    for path in source_root.rglob("*"):
        if not path.is_dir():
            continue
        if len(path.parts) - base_depth > int(scan_depth):
            continue
        if find_image_dir(path) is not None and find_model(path) is not None:
            out.append(path)
    # Prefer shallower unique directories.
    uniq = []
    seen = set()
    for p in sorted(out, key=lambda x: (len(x.parts), str(x))):
        if any(parent in seen for parent in p.parents):
            continue
        seen.add(p)
        uniq.append(p)
    return uniq


def build_subset(args: argparse.Namespace) -> None:
    source_root = Path(args.source_root)
    output_root = Path(args.output_root)
    demo_root = Path(args.demo_output_root) if args.demo_output_root else None
    output_root.mkdir(parents=True, exist_ok=True)
    categories = {x.strip().lower() for x in args.categories.split(",") if x.strip()}
    candidates = object_candidates(source_root, args.scan_depth)
    rows = []
    selected_dirs = []
    for obj_dir in candidates:
        if categories and not any(cat in str(obj_dir).lower() for cat in categories):
            continue
        image_dir = find_image_dir(obj_dir)
        mask_dir = find_mask_dir(obj_dir)
        model = find_model(obj_dir)
        sparse = find_sparse(obj_dir)
        transforms = find_transforms(obj_dir)
        if image_dir is None or model is None:
            continue
        if sparse is None and transforms is None and not args.allow_missing_pose:
            continue
        images = select_uniform(list_images(image_dir), int(args.max_frames))
        if len(images) < int(args.min_frames):
            continue
        uid = obj_dir.name
        out_dir = output_root / uid
        if out_dir.exists() and args.overwrite:
            shutil.rmtree(out_dir)
        out_images = out_dir / "images"
        out_masks = out_dir / "masks"
        out_models = out_dir / "models"
        out_images.mkdir(parents=True, exist_ok=True)
        full_mask_count = 0
        for image in images:
            shutil.copy2(image, out_images / image.name)
            made_full = copy_or_make_mask(mask_dir, image, out_masks / f"{image.stem}.png", bool(args.allow_full_masks))
            full_mask_count += int(made_full)
        model_out = copy_model(model, out_models, uid, bool(args.convert_model_to_obj))
        out_sparse = out_dir / "sparse" / "0"
        pose_source = "missing"
        if sparse is not None:
            copy_sparse(sparse, out_sparse)
            pose_source = "colmap_sparse"
        elif transforms is not None:
            if not write_colmap_from_transforms(transforms, images, out_sparse):
                raise ValueError(f"failed to convert transforms to COLMAP for {obj_dir}")
            pose_source = "transforms_json"
        elif args.allow_missing_pose:
            out_sparse.mkdir(parents=True, exist_ok=True)
            write_empty_points3d(out_sparse / "points3D.txt")

        if demo_root is not None and len(selected_dirs) < int(args.demo_objects):
            demo_dir = demo_root / uid
            if demo_dir.exists() and args.overwrite:
                shutil.rmtree(demo_dir)
            (demo_dir / "images").mkdir(parents=True, exist_ok=True)
            (demo_dir / "masks").mkdir(parents=True, exist_ok=True)
            (demo_dir / "models").mkdir(parents=True, exist_ok=True)
            for image in images[: int(args.demo_frames)]:
                shutil.copy2(out_images / image.name, demo_dir / "images" / image.name)
                mask = out_masks / f"{image.stem}.png"
                if mask.exists():
                    shutil.copy2(mask, demo_dir / "masks" / mask.name)
            shutil.copy2(model_out, demo_dir / "models" / model_out.name)
            (demo_dir / "README.txt").write_text(
                "Small OmniObject3D demo copy for path/schema inspection only. Full data stays under /data.\n",
                encoding="utf-8",
            )

        row = {
            "uid": uid,
            "source_dir": str(obj_dir),
            "dataset_dir": str(out_dir),
            "image_count": len(images),
            "full_mask_count": full_mask_count,
            "pose_source": pose_source,
            "model_source": str(model),
            "model_out": str(model_out),
        }
        rows.append(row)
        selected_dirs.append(out_dir)
        print(f"[omni_subset] {uid} frames={len(images)} pose={pose_source} -> {out_dir}", flush=True)
        if len(selected_dirs) >= int(args.max_objects):
            break
    if not selected_dirs:
        raise SystemExit(f"[omni_subset][ERROR] no usable objects found under {source_root}")
    manifest = {
        "format": "coarsemodel_style_dataset_list_v1",
        "source": "OmniObject3D",
        "source_root": str(source_root),
        "output_root": str(output_root),
        "datasets": [str(p) for p in selected_dirs],
        "rows": rows,
        "args": vars(args),
    }
    write_json(output_root / "dataset_manifest.json", manifest)
    write_csv(output_root / "dataset_manifest.csv", rows)
    print(f"[omni_subset] wrote {output_root / 'dataset_manifest.json'}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert a small OmniObject3D subset to CoarseModel-style RGB/mask/model/pose folders.")
    parser.add_argument("--source_root", default="/data/OmniObject3D")
    parser.add_argument("--output_root", default="/data/trellis_point_prior_mv/omniobject3d_subset")
    parser.add_argument("--demo_output_root", default="/home/zjr/Tracker/trellis_point_prior_mv/demo_assets/omniobject3d_subset")
    parser.add_argument("--max_objects", type=int, default=20)
    parser.add_argument("--max_frames", type=int, default=32)
    parser.add_argument("--min_frames", type=int, default=8)
    parser.add_argument("--scan_depth", type=int, default=4)
    parser.add_argument("--categories", default="")
    parser.add_argument("--allow_full_masks", action="store_true", help="Create full-image masks if OmniObject3D masks are unavailable. Use only for smoke.")
    parser.add_argument("--allow_missing_pose", action="store_true", help="Keep objects without sparse/0 or transforms.json. They cannot run SLAM prior eval until poses are added.")
    parser.add_argument("--convert_model_to_obj", action="store_true", default=True)
    parser.add_argument("--no_convert_model_to_obj", dest="convert_model_to_obj", action="store_false")
    parser.add_argument("--demo_objects", type=int, default=2)
    parser.add_argument("--demo_frames", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    build_subset(parse_args())


if __name__ == "__main__":
    main()
