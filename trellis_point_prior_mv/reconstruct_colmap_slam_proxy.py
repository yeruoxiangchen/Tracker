#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

TRACKER_ROOT = Path(__file__).resolve().parents[1]
if str(TRACKER_ROOT) not in sys.path:
    sys.path.insert(0, str(TRACKER_ROOT))

from trellis_point_prior_mv.common import write_json  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def image_dir_for_dataset(dataset_dir: Path) -> Path:
    for name in ("images", "rgb", "color"):
        path = dataset_dir / name
        if path.is_dir():
            return path
    raise FileNotFoundError(f"missing images/rgb/color under {dataset_dir}")


def mask_dir_for_dataset(dataset_dir: Path) -> Path | None:
    path = dataset_dir / "masks"
    return path if path.is_dir() else None


def mask_for_image(mask_dir: Path | None, image: Path) -> Path | None:
    if mask_dir is None:
        return None
    for suffix in (".png", ".jpg", ".jpeg"):
        p = mask_dir / f"{image.stem}{suffix}"
        if p.exists():
            return p
    return None


def list_images(path: Path) -> list[Path]:
    return sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def parse_first_colmap_camera(cameras_txt: Path) -> tuple[str, str] | None:
    if not cameras_txt.exists():
        return None
    with cameras_txt.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            model = parts[1].upper()
            params = [float(x) for x in parts[4:]]
            if model == "PINHOLE" and len(params) >= 4:
                fx, fy, cx, cy = params[:4]
                return "PINHOLE", f"{fx},{fy},{cx},{cy}"
            if model == "SIMPLE_PINHOLE" and len(params) >= 3:
                f, cx, cy = params[:3]
                return "SIMPLE_PINHOLE", f"{f},{cx},{cy}"
            if model == "SIMPLE_RADIAL" and len(params) >= 4:
                f, cx, cy, k = params[:4]
                return "SIMPLE_RADIAL", f"{f},{cx},{cy},{k}"
            if model == "OPENCV" and len(params) >= 8:
                fx, fy, cx, cy, k1, k2, p1, p2 = params[:8]
                return "OPENCV", f"{fx},{fy},{cx},{cy},{k1},{k2},{p1},{p2}"
    return None


def intrinsics_for_dataset(dataset_dir: Path, args: argparse.Namespace) -> tuple[str, str | None, str]:
    if args.camera_params:
        return str(args.camera_model), str(args.camera_params), "manual"
    source = str(args.intrinsics_source)
    if source in {"auto", "existing_sparse"}:
        parsed = parse_first_colmap_camera(dataset_dir / str(args.intrinsics_sparse_subdir) / "cameras.txt")
        if parsed is not None:
            model, params = parsed
            return model, params, "existing_sparse"
        if source == "existing_sparse":
            raise FileNotFoundError(
                f"requested intrinsics_source=existing_sparse but no supported camera found at "
                f"{dataset_dir / str(args.intrinsics_sparse_subdir) / 'cameras.txt'}"
            )
    if source == "none" or source == "auto":
        return str(args.camera_model), None, "colmap_estimated"
    raise ValueError(f"unsupported intrinsics_source={source}")


def select_images(images: list[Path], max_frames: int, mode: str, stride: int, seed: int) -> list[Path]:
    if max_frames <= 0 or len(images) <= max_frames:
        return images
    if mode == "uniform":
        ids = np.linspace(0, len(images) - 1, int(max_frames))
        keep = sorted({int(round(x)) for x in ids})
    elif mode == "stride":
        keep = list(range(0, len(images), max(int(stride), 1)))[: int(max_frames)]
    elif mode == "random":
        rng = np.random.default_rng(int(seed))
        keep = sorted(rng.choice(len(images), size=int(max_frames), replace=False).astype(int).tolist())
    elif mode == "random_uniform":
        rng = np.random.default_rng(int(seed))
        edges = np.linspace(0, len(images), int(max_frames) + 1)
        keep = []
        for start, end in zip(edges[:-1], edges[1:]):
            lo = int(math.floor(start))
            hi = max(lo + 1, int(math.ceil(end)))
            hi = min(hi, len(images))
            keep.append(int(rng.integers(lo, hi)))
        keep = sorted(set(keep))
        if len(keep) < int(max_frames):
            rest = [i for i in range(len(images)) if i not in keep]
            extra = rng.choice(rest, size=min(int(max_frames) - len(keep), len(rest)), replace=False).astype(int).tolist()
            keep = sorted(keep + extra)
    else:
        keep = list(range(int(max_frames)))
    return [images[i] for i in keep[: int(max_frames)]]


def run_cmd(cmd: list[str], *, cwd: Path | None = None) -> None:
    print("[colmap_slam_proxy] " + " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=str(cwd) if cwd else None)
    if proc.returncode != 0:
        raise RuntimeError(f"command failed with returncode={proc.returncode}: {' '.join(cmd)}")


def ensure_colmap(colmap_bin: str) -> str:
    found = shutil.which(colmap_bin)
    if found is None:
        raise FileNotFoundError(
            f"COLMAP executable not found: {colmap_bin}. Install COLMAP or set COLMAP_BIN=/path/to/colmap. "
            "This script intentionally estimates pose+points from RGB video instead of reusing existing sparse poses."
        )
    return found


def prepare_workspace(dataset_dir: Path, work_dir: Path, selected: list[Path], args: argparse.Namespace) -> tuple[Path, Path | None]:
    images_out = work_dir / "images"
    masks_out = work_dir / "masks"
    if work_dir.exists() and args.overwrite:
        shutil.rmtree(work_dir)
    images_out.mkdir(parents=True, exist_ok=True)
    mask_dir = mask_dir_for_dataset(dataset_dir)
    used_masks = False
    for image in selected:
        shutil.copy2(image, images_out / image.name)
        mask = mask_for_image(mask_dir, image)
        if mask is not None:
            masks_out.mkdir(parents=True, exist_ok=True)
            shutil.copy2(mask, masks_out / image.name)
            used_masks = True
    return images_out, masks_out if used_masks else None


def model_dir_candidates(work_dir: Path) -> list[Path]:
    sparse = work_dir / "sparse"
    if not sparse.exists():
        return []
    return sorted(p for p in sparse.iterdir() if p.is_dir())


def convert_best_model(colmap_bin: str, work_dir: Path, output_sparse: Path, args: argparse.Namespace) -> Path:
    models = model_dir_candidates(work_dir)
    if not models:
        raise FileNotFoundError(f"COLMAP mapper produced no sparse model under {work_dir / 'sparse'}")
    best = max(models, key=lambda p: (p / "images.bin").stat().st_size if (p / "images.bin").exists() else 0)
    if output_sparse.exists() and args.overwrite:
        shutil.rmtree(output_sparse)
    output_sparse.mkdir(parents=True, exist_ok=True)
    run_cmd([
        colmap_bin,
        "model_converter",
        "--input_path",
        str(best),
        "--output_path",
        str(output_sparse),
        "--output_type",
        "TXT",
    ])
    return best


def process_dataset(dataset_dir: Path, args: argparse.Namespace, colmap_bin: str) -> dict:
    image_dir = image_dir_for_dataset(dataset_dir)
    images = list_images(image_dir)
    selected = select_images(images, int(args.max_frames), str(args.frame_select), int(args.frame_stride), int(args.seed))
    if len(selected) < int(args.min_frames):
        raise ValueError(f"{dataset_dir}: selected frames {len(selected)} < min_frames {int(args.min_frames)}")

    work_dir = dataset_dir / args.work_subdir
    output_sparse = dataset_dir / args.output_sparse_subdir
    if output_sparse.exists() and not args.overwrite:
        raise FileExistsError(f"output sparse exists; pass --overwrite: {output_sparse}")
    images_out, masks_out = prepare_workspace(dataset_dir, work_dir, selected, args)
    database = work_dir / "database.db"
    sparse_out = work_dir / "sparse"
    sparse_out.mkdir(parents=True, exist_ok=True)
    camera_model, camera_params, intrinsics_source = intrinsics_for_dataset(dataset_dir, args)

    feature_cmd = [
        colmap_bin,
        "feature_extractor",
        "--database_path",
        str(database),
        "--image_path",
        str(images_out),
        "--ImageReader.single_camera",
        "1",
        "--ImageReader.camera_model",
        str(camera_model),
        "--SiftExtraction.max_num_features",
        str(args.max_features),
        "--SiftExtraction.use_gpu",
        str(int(bool(args.use_gpu))),
    ]
    if camera_params:
        feature_cmd.extend(["--ImageReader.camera_params", str(camera_params)])
    if bool(args.use_masks) and masks_out is not None:
        feature_cmd.extend(["--ImageReader.mask_path", str(masks_out)])
    run_cmd(feature_cmd)

    if args.matcher == "sequential":
        run_cmd([
            colmap_bin,
            "sequential_matcher",
            "--database_path",
            str(database),
            "--SiftMatching.use_gpu",
            str(int(bool(args.use_gpu))),
            "--SequentialMatching.overlap",
            str(args.sequential_overlap),
        ])
    elif args.matcher == "exhaustive":
        run_cmd([
            colmap_bin,
            "exhaustive_matcher",
            "--database_path",
            str(database),
            "--SiftMatching.use_gpu",
            str(int(bool(args.use_gpu))),
        ])
    else:
        raise ValueError(f"unsupported matcher={args.matcher}")

    mapper_cmd = [
        colmap_bin,
        "mapper",
        "--database_path",
        str(database),
        "--image_path",
        str(images_out),
        "--output_path",
        str(sparse_out),
        "--Mapper.min_num_matches",
        str(args.mapper_min_num_matches),
        "--Mapper.ba_global_max_num_iterations",
        str(args.ba_global_max_num_iterations),
    ]
    fix_intrinsics = bool(args.fix_intrinsics and intrinsics_source in {"manual", "existing_sparse"})
    if fix_intrinsics:
        mapper_cmd.extend([
            "--Mapper.ba_refine_focal_length",
            "0",
            "--Mapper.ba_refine_principal_point",
            "0",
            "--Mapper.ba_refine_extra_params",
            "0",
        ])
    run_cmd(mapper_cmd)
    best = convert_best_model(colmap_bin, work_dir, output_sparse, args)

    row = {
        "dataset": str(dataset_dir),
        "image_count": len(images),
        "selected_count": len(selected),
        "frame_select": str(args.frame_select),
        "matcher": str(args.matcher),
        "use_masks": int(bool(args.use_masks and masks_out is not None)),
        "intrinsics_source": intrinsics_source,
        "camera_model": str(camera_model),
        "camera_params": str(camera_params or ""),
        "fix_intrinsics": int(fix_intrinsics),
        "work_dir": str(work_dir),
        "best_binary_sparse": str(best),
        "output_sparse": str(output_sparse),
    }
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate COLMAP pose+points from RGB video for AR-like SLAM proxy datasets.")
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--output_sparse_subdir", default="sparse_colmap_arproxy/0")
    parser.add_argument("--work_subdir", default="colmap_arproxy_work")
    parser.add_argument("--max_frames", type=int, default=64)
    parser.add_argument("--min_frames", type=int, default=8)
    parser.add_argument("--frame_select", choices=["first", "uniform", "stride", "random", "random_uniform"], default="random_uniform")
    parser.add_argument("--frame_stride", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--matcher", choices=["sequential", "exhaustive"], default="sequential")
    parser.add_argument("--sequential_overlap", type=int, default=8)
    parser.add_argument("--camera_model", default="SIMPLE_RADIAL")
    parser.add_argument("--camera_params", default="", help="COLMAP camera_params string, e.g. fx,fy,cx,cy for PINHOLE.")
    parser.add_argument("--intrinsics_source", choices=["auto", "none", "existing_sparse"], default="auto")
    parser.add_argument("--intrinsics_sparse_subdir", default="sparse/0")
    parser.add_argument("--max_features", type=int, default=4096)
    parser.add_argument("--mapper_min_num_matches", type=int, default=15)
    parser.add_argument("--ba_global_max_num_iterations", type=int, default=20)
    parser.add_argument("--use_gpu", action="store_true")
    parser.add_argument("--use_masks", action="store_true", help="Use masks during feature extraction. Default uses full RGB for pose robustness.")
    parser.add_argument("--fix_intrinsics", action="store_true", help="Keep provided/loaded intrinsics fixed in COLMAP mapper.")
    parser.add_argument("--colmap_bin", default="colmap")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--output_report", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    colmap_bin = ensure_colmap(str(args.colmap_bin))
    rows = []
    for dataset in args.datasets:
        row = process_dataset(Path(dataset), args, colmap_bin)
        rows.append(row)
        print(f"[colmap_slam_proxy] {Path(dataset).name} selected={row['selected_count']} -> {row['output_sparse']}", flush=True)
    if args.output_report:
        out = Path(args.output_report)
        write_json(out, {"rows": rows, "args": vars(args)})
        write_csv(out.with_suffix(".csv"), rows)


if __name__ == "__main__":
    main()
