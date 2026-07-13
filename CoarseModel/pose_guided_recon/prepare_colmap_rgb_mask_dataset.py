#!/usr/bin/env python3

"""Prepare an RGB+mask dataset and optionally run COLMAP.

The output is a standard dataset directory:

  colmap_dataset/
    rgb/
    images/
    masks/
    sparse/0/{cameras.txt, images.txt, points3D.txt}

This script is intentionally independent from the existing CoarseModel and
ReconViaGen server code.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


ROOT = Path("/home/zjr/Tracker")
DEFAULT_OUTPUT_ROOT = ROOT / "CoarseModel" / "pose_guided_recon" / "outputs"
DEFAULT_COLMAP_BIN = Path("/home/zjr/anaconda3/envs/foundpose/bin/colmap")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def reset_dir(path: Path) -> Path:
    if path.exists() or path.is_symlink():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return path


def safe_name(text: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in text)[:96]


def image_files(path: Path) -> List[Path]:
    if not path.exists():
        return []
    return sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def find_first_existing(paths: Sequence[Path]) -> Optional[Path]:
    for path in paths:
        if path.exists():
            return path
    return None


def find_mask(mask_dir: Path, stem: str) -> Optional[Path]:
    for suffix in [".png", ".jpg", ".jpeg"]:
        path = mask_dir / f"{stem}{suffix}"
        if path.exists():
            return path
    return None


def link_or_copy(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    if mode == "copy":
        shutil.copy2(src, dst)
    else:
        os.symlink(src.resolve(), dst)


def mask_area(mask_path: Path) -> float:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return 0.0
    return float((mask > 0).mean())


def image_sharpness(image_path: Path) -> float:
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return 0.0
    if max(image.shape[:2]) > 720:
        image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(image, cv2.CV_64F).var())


def discover_dirs(dataset_dir: Optional[Path], rgb_dir: Optional[Path], mask_dir: Optional[Path]) -> Tuple[Path, Path]:
    if dataset_dir is None and (rgb_dir is None or mask_dir is None):
        raise ValueError("Use --dataset_dir, or provide both --rgb_dir and --mask_dir")
    if dataset_dir is not None:
        rgb_dir = rgb_dir or find_first_existing([dataset_dir / "rgb", dataset_dir / "images", dataset_dir / "color"])
        mask_dir = mask_dir or dataset_dir / "masks"
    if rgb_dir is None or mask_dir is None:
        raise ValueError("Could not resolve rgb/mask directories")
    if not rgb_dir.exists() or not mask_dir.exists():
        raise FileNotFoundError(f"Missing rgb/mask directories: {rgb_dir}, {mask_dir}")
    return rgb_dir.resolve(), mask_dir.resolve()


def select_pairs(
    rgb_dir: Path,
    mask_dir: Path,
    max_images: int,
    min_mask_area: float,
    max_mask_area: float,
) -> List[Dict[str, Any]]:
    pairs = []
    for image_path in image_files(rgb_dir):
        mask_path = find_mask(mask_dir, image_path.stem)
        if mask_path is None:
            continue
        area = mask_area(mask_path)
        if area <= 0:
            continue
        pairs.append(
            {
                "image_path": image_path,
                "mask_path": mask_path,
                "name": image_path.name,
                "stem": image_path.stem,
                "mask_area": area,
                "sharpness": image_sharpness(image_path),
            }
        )
    if not pairs:
        raise RuntimeError(f"No valid image/mask pairs found in {rgb_dir} and {mask_dir}")
    if max_images <= 0 or len(pairs) <= max_images:
        return pairs

    valid = [p for p in pairs if min_mask_area <= float(p["mask_area"]) <= max_mask_area]
    if len(valid) < max_images:
        valid = pairs
    # Preserve temporal/trajectory coverage by uniform bins, but pick the best
    # mask/sharpness candidate inside each bin.
    sharp_values = np.asarray([p["sharpness"] for p in valid], dtype=np.float64)
    sharp_lo = float(np.percentile(sharp_values, 5))
    sharp_hi = float(np.percentile(sharp_values, 95))

    def quality(item: Dict[str, Any]) -> float:
        area = float(item["mask_area"])
        area_score = max(0.0, 1.0 - abs(area - 0.12) / 0.12)
        sharp_score = (float(item["sharpness"]) - sharp_lo) / max(sharp_hi - sharp_lo, 1e-6)
        return 0.70 * area_score + 0.30 * float(np.clip(sharp_score, 0.0, 1.0))

    selected: List[Dict[str, Any]] = []
    edges = np.linspace(0, len(valid), max_images + 1, dtype=int)
    for start, end in zip(edges[:-1], edges[1:]):
        bucket = valid[start:max(end, start + 1)]
        if not bucket:
            continue
        selected.append(max(bucket, key=quality))
    seen = {p["name"] for p in selected}
    if len(selected) < max_images:
        for item in sorted(valid, key=quality, reverse=True):
            if item["name"] not in seen:
                selected.append(item)
                seen.add(item["name"])
            if len(selected) >= max_images:
                break
    return sorted(selected, key=lambda p: pairs.index(p))


def normalize_dataset(
    rgb_dir: Path,
    mask_dir: Path,
    dataset_out: Path,
    max_images: int,
    min_mask_area: float,
    max_mask_area: float,
    link_mode: str,
) -> Dict[str, Any]:
    reset_dir(dataset_out)
    for name in ["rgb", "images", "masks", "models"]:
        ensure_dir(dataset_out / name)
    selected = select_pairs(rgb_dir, mask_dir, max_images=max_images, min_mask_area=min_mask_area, max_mask_area=max_mask_area)
    rows = []
    for item in selected:
        image_path = Path(item["image_path"])
        mask_path = Path(item["mask_path"])
        link_or_copy(image_path, dataset_out / "rgb" / image_path.name, link_mode)
        link_or_copy(image_path, dataset_out / "images" / image_path.name, link_mode)
        # The pose-guided scorer expects stem.png. COLMAP 3.12 expects
        # image_name + ".png" under ImageReader.mask_path, e.g. foo.jpg.png.
        mask_out = dataset_out / "masks" / f"{image_path.stem}.png"
        colmap_mask_out = dataset_out / "masks" / f"{image_path.name}.png"
        if link_mode == "copy":
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise FileNotFoundError(mask_path)
            bin_mask = (mask > 0).astype(np.uint8) * 255
            cv2.imwrite(str(mask_out), bin_mask)
            if colmap_mask_out != mask_out:
                cv2.imwrite(str(colmap_mask_out), bin_mask)
        else:
            link_or_copy(mask_path, mask_out, link_mode)
            if colmap_mask_out != mask_out:
                link_or_copy(mask_path, colmap_mask_out, link_mode)
        rows.append(
            {
                "name": image_path.name,
                "source_image": str(image_path),
                "source_mask": str(mask_path),
                "mask_area": float(item["mask_area"]),
                "sharpness": float(item["sharpness"]),
            }
        )
    report = {
        "dataset_dir": str(dataset_out),
        "source_rgb_dir": str(rgb_dir),
        "source_mask_dir": str(mask_dir),
        "selected_count": len(rows),
        "selected_frames": [r["name"] for r in rows],
        "frames": rows,
    }
    write_json(dataset_out / "reconviagen_meta.json", report)
    return report


def run_cmd(cmd: Sequence[str], log_path: Path, cwd: Optional[Path] = None) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write(" ".join(cmd) + "\n\n")
        log.flush()
        proc = subprocess.run(cmd, cwd=str(cwd) if cwd else None, stdout=log, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed returncode={proc.returncode}; see {log_path}")


def read_registered_image_names(images_txt: Path) -> List[str]:
    if not images_txt.exists():
        return []
    names: List[str] = []
    lines = images_txt.read_text(encoding="utf-8", errors="ignore").splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line or line.startswith("#"):
            i += 1
            continue
        parts = line.split()
        if len(parts) >= 10:
            names.append(" ".join(parts[9:]))
            i += 2
        else:
            i += 1
    return names


def run_colmap_once(
    dataset_out: Path,
    colmap_bin: Path,
    use_masks: bool,
    matcher: str,
    use_gpu: int,
    single_camera: int,
    max_image_size: int,
) -> Dict[str, Any]:
    if not colmap_bin.exists():
        raise FileNotFoundError(f"COLMAP binary not found: {colmap_bin}")
    database_path = dataset_out / "database.db"
    sparse_path = ensure_dir(dataset_out / "sparse")
    if database_path.exists():
        database_path.unlink()
    for child in sparse_path.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()

    image_path = dataset_out / "images"
    masks_path = dataset_out / "masks"
    logs_dir = ensure_dir(dataset_out / "colmap_logs")

    feature_cmd = [
        str(colmap_bin),
        "feature_extractor",
        "--database_path",
        str(database_path),
        "--image_path",
        str(image_path),
        "--ImageReader.single_camera",
        str(int(single_camera)),
        "--ImageReader.camera_model",
        "SIMPLE_RADIAL",
        "--SiftExtraction.use_gpu",
        str(int(use_gpu)),
        "--SiftExtraction.max_image_size",
        str(int(max_image_size)),
    ]
    if use_masks:
        feature_cmd.extend(["--ImageReader.mask_path", str(masks_path)])
    run_cmd(feature_cmd, logs_dir / "01_feature_extractor.log", cwd=dataset_out)

    if matcher == "sequential":
        matcher_cmd = [
            str(colmap_bin),
            "sequential_matcher",
            "--database_path",
            str(database_path),
            "--SiftMatching.use_gpu",
            str(int(use_gpu)),
            "--SequentialMatching.overlap",
            "10",
            "--SequentialMatching.loop_detection",
            "1",
        ]
        matcher_log = logs_dir / "02_sequential_matcher.log"
    else:
        matcher_cmd = [
            str(colmap_bin),
            "exhaustive_matcher",
            "--database_path",
            str(database_path),
            "--SiftMatching.use_gpu",
            str(int(use_gpu)),
        ]
        matcher_log = logs_dir / "02_exhaustive_matcher.log"
    run_cmd(matcher_cmd, matcher_log, cwd=dataset_out)

    mapper_cmd = [
        str(colmap_bin),
        "mapper",
        "--database_path",
        str(database_path),
        "--image_path",
        str(image_path),
        "--output_path",
        str(sparse_path),
    ]
    run_cmd(mapper_cmd, logs_dir / "03_mapper.log", cwd=dataset_out)

    models = sorted([p for p in sparse_path.iterdir() if p.is_dir()])
    if not models:
        raise RuntimeError(f"COLMAP mapper produced no sparse model under {sparse_path}")
    model_dir = models[0]
    if model_dir.name != "0":
        target = sparse_path / "0"
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(model_dir, target)
        model_dir = target

    converter_txt_cmd = [
        str(colmap_bin),
        "model_converter",
        "--input_path",
        str(model_dir),
        "--output_path",
        str(model_dir),
        "--output_type",
        "TXT",
    ]
    run_cmd(converter_txt_cmd, logs_dir / "04_model_converter_txt.log", cwd=dataset_out)

    converter_ply_cmd = [
        str(colmap_bin),
        "model_converter",
        "--input_path",
        str(model_dir),
        "--output_path",
        str(model_dir / "model.ply"),
        "--output_type",
        "PLY",
    ]
    run_cmd(converter_ply_cmd, logs_dir / "05_model_converter_ply.log", cwd=dataset_out)

    required = [model_dir / "cameras.txt", model_dir / "images.txt"]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise RuntimeError(f"COLMAP text model missing files: {missing}")
    registered_image_names = read_registered_image_names(model_dir / "images.txt")
    return {
        "database_path": str(database_path),
        "sparse_model_dir": str(model_dir),
        "logs_dir": str(logs_dir),
        "use_masks": bool(use_masks),
        "matcher": matcher,
        "use_gpu": int(use_gpu),
        "registered_image_count": len(registered_image_names),
        "registered_image_names": registered_image_names,
    }


def run_colmap(
    dataset_out: Path,
    colmap_bin: Path,
    use_masks: bool,
    matcher: str,
    use_gpu: int,
    single_camera: int,
    max_image_size: int,
    fallback_without_masks: bool,
) -> Dict[str, Any]:
    try:
        report = run_colmap_once(
            dataset_out=dataset_out,
            colmap_bin=colmap_bin,
            use_masks=use_masks,
            matcher=matcher,
            use_gpu=use_gpu,
            single_camera=single_camera,
            max_image_size=max_image_size,
        )
        report["fallback_without_masks"] = False
        return report
    except Exception as exc:
        if not use_masks or not fallback_without_masks:
            raise
        logs_dir = ensure_dir(dataset_out / "colmap_logs")
        (logs_dir / "00_fallback_without_masks.txt").write_text(
            "Masked COLMAP failed. Retrying without ImageReader.mask_path.\n\n"
            f"Reason: {exc}\n",
            encoding="utf-8",
        )
        report = run_colmap_once(
            dataset_out=dataset_out,
            colmap_bin=colmap_bin,
            use_masks=False,
            matcher=matcher,
            use_gpu=use_gpu,
            single_camera=single_camera,
            max_image_size=max_image_size,
        )
        report["fallback_without_masks"] = True
        report["fallback_reason"] = str(exc)
        return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", default=None)
    parser.add_argument("--rgb_dir", default=None)
    parser.add_argument("--mask_dir", default=None)
    parser.add_argument("--case_name", default=None)
    parser.add_argument("--output_root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--link_mode", choices=["symlink", "copy"], default="symlink")
    parser.add_argument("--max_colmap_images", type=int, default=120)
    parser.add_argument("--min_mask_area", type=float, default=0.005)
    parser.add_argument("--max_mask_area", type=float, default=0.65)
    parser.add_argument("--run_colmap", type=int, default=1)
    parser.add_argument("--reuse_colmap", type=int, default=0)
    parser.add_argument("--colmap_bin", default=str(DEFAULT_COLMAP_BIN))
    parser.add_argument("--colmap_use_masks", type=int, default=1)
    parser.add_argument("--colmap_matcher", choices=["exhaustive", "sequential"], default="exhaustive")
    parser.add_argument("--colmap_use_gpu", type=int, default=0)
    parser.add_argument("--colmap_single_camera", type=int, default=1)
    parser.add_argument("--colmap_max_image_size", type=int, default=2000)
    parser.add_argument("--colmap_fallback_without_masks", type=int, default=1)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir).resolve() if args.dataset_dir else None
    rgb_dir = Path(args.rgb_dir).resolve() if args.rgb_dir else None
    mask_dir = Path(args.mask_dir).resolve() if args.mask_dir else None
    rgb_dir, mask_dir = discover_dirs(dataset_dir, rgb_dir, mask_dir)

    if args.case_name:
        case_name = safe_name(args.case_name)
    elif dataset_dir is not None:
        case_name = safe_name(dataset_dir.name + "_colmap_from_rgbmask")
    else:
        case_name = safe_name(rgb_dir.parent.name + "_colmap_from_rgbmask")
    output_root = Path(args.output_root).resolve() / case_name
    dataset_out = output_root / "colmap_dataset"
    sparse_txt = dataset_out / "sparse" / "0" / "images.txt"

    if int(args.reuse_colmap) and sparse_txt.exists():
        report = {
            "case_name": case_name,
            "dataset_dir": str(dataset_out),
            "source_rgb_dir": str(rgb_dir),
            "source_mask_dir": str(mask_dir),
            "reuse_colmap": True,
            "colmap": {"sparse_model_dir": str(dataset_out / "sparse" / "0")},
        }
    else:
        report = normalize_dataset(
            rgb_dir=rgb_dir,
            mask_dir=mask_dir,
            dataset_out=dataset_out,
            max_images=args.max_colmap_images,
            min_mask_area=args.min_mask_area,
            max_mask_area=args.max_mask_area,
            link_mode=args.link_mode,
        )
        report["case_name"] = case_name
        if int(args.run_colmap):
            report["colmap"] = run_colmap(
                dataset_out=dataset_out,
                colmap_bin=Path(args.colmap_bin).resolve(),
                use_masks=bool(args.colmap_use_masks),
                matcher=args.colmap_matcher,
                use_gpu=int(args.colmap_use_gpu),
                single_camera=int(args.colmap_single_camera),
                max_image_size=int(args.colmap_max_image_size),
                fallback_without_masks=bool(args.colmap_fallback_without_masks),
            )
        else:
            report["colmap"] = None
    write_json(output_root / "prepare_colmap_report.json", report)
    print(json.dumps({"dataset_dir": str(dataset_out), "report": str(output_root / "prepare_colmap_report.json")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
