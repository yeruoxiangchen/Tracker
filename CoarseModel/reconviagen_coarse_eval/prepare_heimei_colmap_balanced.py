#!/usr/bin/env python3

"""Prepare a COLMAP-backed dataset for ReconViaGen with pose-balanced frames."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw

from common import DEFAULT_OUTPUT_ROOT, copy_tree_contents, ensure_dir, link_or_copy, write_json


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
MASK_SUFFIXES = [".png", ".jpg", ".jpeg"]


def qvec_to_rotmat(qvec: Sequence[float]) -> np.ndarray:
    qw, qx, qy, qz = [float(v) for v in qvec]
    return np.array(
        [
            [1 - 2 * qy * qy - 2 * qz * qz, 2 * qx * qy - 2 * qz * qw, 2 * qx * qz + 2 * qy * qw],
            [2 * qx * qy + 2 * qz * qw, 1 - 2 * qx * qx - 2 * qz * qz, 2 * qy * qz - 2 * qx * qw],
            [2 * qx * qz - 2 * qy * qw, 2 * qy * qz + 2 * qx * qw, 1 - 2 * qx * qx - 2 * qy * qy],
        ],
        dtype=np.float64,
    )


def read_colmap_images_txt(path: Path) -> Dict[str, Dict[str, Any]]:
    records: Dict[str, Dict[str, Any]] = {}
    lines = path.read_text(encoding="utf-8").splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line or line.startswith("#"):
            i += 1
            continue
        parts = line.split()
        if len(parts) < 10:
            i += 1
            continue
        image_id = int(parts[0])
        qvec = [float(v) for v in parts[1:5]]
        tvec = np.array([float(v) for v in parts[5:8]], dtype=np.float64)
        camera_id = int(parts[8])
        name = " ".join(parts[9:])
        R_w2c = qvec_to_rotmat(qvec)
        center = -R_w2c.T @ tvec
        records[name] = {
            "image_id": image_id,
            "camera_id": camera_id,
            "qvec": qvec,
            "tvec": [float(v) for v in tvec.tolist()],
            "R_w2c": R_w2c,
            "camera_center": center,
        }
        i += 2
    return records


def image_files(path: Path) -> List[Path]:
    if not path.exists():
        return []
    return sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def find_mask(mask_dir: Path, stem: str) -> Optional[Path]:
    for suffix in MASK_SUFFIXES:
        path = mask_dir / f"{stem}{suffix}"
        if path.exists():
            return path
    return None


def mask_stats(mask_path: Optional[Path]) -> Dict[str, Any]:
    if mask_path is None:
        return {
            "mask_path": None,
            "mask_area_ratio": None,
            "border_touch_ratio": None,
            "valid_mask": False,
        }
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return {
            "mask_path": str(mask_path),
            "mask_area_ratio": None,
            "border_touch_ratio": None,
            "valid_mask": False,
        }
    fg = mask > 0
    h, w = fg.shape[:2]
    area = float(fg.mean())
    border = np.zeros_like(fg, dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    border_touch = float(np.logical_and(fg, border).sum() / max(float(fg.sum()), 1.0))
    return {
        "mask_path": str(mask_path),
        "mask_area_ratio": area,
        "border_touch_ratio": border_touch,
        "valid_mask": bool(fg.any()),
    }


def image_sharpness(path: Path) -> float:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return 0.0
    return float(cv2.Laplacian(image, cv2.CV_64F).var())


def circular_span_and_unwrapped(angles: np.ndarray) -> Tuple[float, np.ndarray, float]:
    """Return observed circular span in radians and angles unwrapped to that interval."""
    if len(angles) <= 1:
        return 0.0, np.zeros_like(angles, dtype=np.float64), 0.0
    angles = np.mod(angles, 2.0 * math.pi)
    order = np.argsort(angles)
    sorted_angles = angles[order]
    gaps = np.diff(np.concatenate([sorted_angles, sorted_angles[:1] + 2.0 * math.pi]))
    gap_idx = int(np.argmax(gaps))
    start = float(sorted_angles[(gap_idx + 1) % len(sorted_angles)])
    unwrapped = np.mod(angles - start, 2.0 * math.pi)
    span = float(2.0 * math.pi - gaps[gap_idx])
    return span, unwrapped, start


def add_pose_angles(records: List[Dict[str, Any]]) -> Dict[str, float]:
    centers = np.stack([r["camera_center"] for r in records], axis=0)
    origin = centers.mean(axis=0)
    rel = centers - origin[None]
    radius = np.linalg.norm(rel, axis=1)
    azimuth = np.arctan2(rel[:, 0], rel[:, 2])
    elevation = np.degrees(np.arctan2(rel[:, 1], np.maximum(np.linalg.norm(rel[:, [0, 2]], axis=1), 1e-8)))
    span, unwrapped, start = circular_span_and_unwrapped(azimuth)
    for i, record in enumerate(records):
        record["camera_center"] = [float(v) for v in centers[i].tolist()]
        record["azimuth_deg"] = float(math.degrees(azimuth[i]))
        record["azimuth_unwrapped_deg"] = float(math.degrees(unwrapped[i]))
        record["elevation_deg"] = float(elevation[i])
        record["radius"] = float(radius[i])
    return {
        "pose_origin": [float(v) for v in origin.tolist()],
        "azimuth_span_deg": float(math.degrees(span)),
        "azimuth_unwrap_start_deg": float(math.degrees(start)),
        "elevation_range_deg": float(elevation.max() - elevation.min()) if len(elevation) else 0.0,
        "radius_mean": float(radius.mean()) if len(radius) else 0.0,
        "radius_std": float(radius.std()) if len(radius) else 0.0,
    }


def frame_quality(record: Dict[str, Any], min_mask_area: float, max_mask_area: float, max_border_touch: float) -> float:
    area = record.get("mask_area_ratio")
    border = record.get("border_touch_ratio")
    sharp = float(record.get("sharpness") or 0.0)
    if area is None or not record.get("valid_mask"):
        return -1e6
    quality = math.log1p(max(sharp, 0.0))
    if area < min_mask_area:
        quality -= 8.0 * (min_mask_area - area) / max(min_mask_area, 1e-6)
    if area > max_mask_area:
        quality -= 8.0 * (area - max_mask_area) / max(1.0 - max_mask_area, 1e-6)
    if border is not None and border > max_border_touch:
        quality -= 4.0 * (border - max_border_touch) / max(1.0 - max_border_touch, 1e-6)
    return float(quality)


def select_balanced(records: List[Dict[str, Any]], max_frames: Optional[int]) -> List[Dict[str, Any]]:
    valid = [r for r in records if float(r.get("quality", -1e6)) > -1e5]
    if not valid:
        valid = list(records)
    if max_frames is None or max_frames <= 0 or len(valid) <= max_frames:
        return sorted(valid, key=lambda r: int(r["source_order"]))

    span = max(float(max(r["azimuth_unwrapped_deg"] for r in valid) - min(r["azimuth_unwrapped_deg"] for r in valid)), 1e-6)
    q_values = np.array([float(r.get("quality", 0.0)) for r in valid], dtype=np.float64)
    q_min = float(q_values.min())
    q_max = float(q_values.max())
    for record, q in zip(valid, q_values):
        record["_quality_norm"] = float((q - q_min) / max(q_max - q_min, 1e-8))

    lo = min(float(r["azimuth_unwrapped_deg"]) for r in valid)
    targets = np.linspace(lo, lo + span, int(max_frames), endpoint=False) + 0.5 * span / float(max_frames)
    selected: List[Dict[str, Any]] = []
    used = set()
    for target in targets:
        best = None
        best_score = float("inf")
        for record in valid:
            if record["name"] in used:
                continue
            distance = abs(float(record["azimuth_unwrapped_deg"]) - float(target)) / span
            score = distance - 0.08 * float(record.get("_quality_norm", 0.0))
            if score < best_score:
                best = record
                best_score = score
        if best is not None:
            selected.append(best)
            used.add(best["name"])

    if len(selected) < max_frames:
        remaining = [r for r in valid if r["name"] not in used]
        remaining.sort(key=lambda r: float(r.get("quality", 0.0)), reverse=True)
        selected.extend(remaining[: max_frames - len(selected)])
    return sorted(selected, key=lambda r: int(r["source_order"]))


def valid_quality_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    valid = [r for r in records if float(r.get("quality", -1e6)) > -1e5]
    return valid if valid else list(records)


def select_sorted_first(records: List[Dict[str, Any]], max_frames: Optional[int]) -> List[Dict[str, Any]]:
    selected = sorted(valid_quality_records(records), key=lambda r: int(r["source_order"]))
    if max_frames is not None and max_frames > 0:
        selected = selected[: int(max_frames)]
    return selected


def select_random(records: List[Dict[str, Any]], max_frames: Optional[int], random_seed: int) -> List[Dict[str, Any]]:
    valid = valid_quality_records(records)
    if max_frames is None or max_frames <= 0 or len(valid) <= max_frames:
        return sorted(valid, key=lambda r: int(r["source_order"]))
    rng = np.random.default_rng(int(random_seed))
    ids = rng.choice(len(valid), size=int(max_frames), replace=False)
    return sorted([valid[int(i)] for i in ids], key=lambda r: int(r["source_order"]))


def filter_arc_records(
    records: List[Dict[str, Any]],
    arc_span_deg: float,
    arc_center_fraction: float,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    valid = valid_quality_records(records)
    if not valid:
        return [], {}
    values = np.array([float(r["azimuth_unwrapped_deg"]) for r in valid], dtype=np.float64)
    lo = float(values.min())
    hi = float(values.max())
    full_span = max(hi - lo, 1e-6)
    span = float(max(1.0, min(float(arc_span_deg), full_span)))
    center = lo + float(np.clip(arc_center_fraction, 0.0, 1.0)) * full_span
    left = center - 0.5 * span
    right = center + 0.5 * span
    inside = [r for r in valid if left <= float(r["azimuth_unwrapped_deg"]) <= right]
    if not inside:
        inside = sorted(valid, key=lambda r: abs(float(r["azimuth_unwrapped_deg"]) - center))[:1]
    return inside, {
        "arc_left_deg": left,
        "arc_right_deg": right,
        "arc_center_deg": center,
        "arc_span_deg": span,
        "arc_center_fraction": float(arc_center_fraction),
        "full_available_span_deg": full_span,
    }


def filter_elevation_records(records: List[Dict[str, Any]], band: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    valid = valid_quality_records(records)
    values = np.array([float(r["elevation_deg"]) for r in valid], dtype=np.float64)
    if len(values) == 0:
        return valid, {}
    q1, q2 = np.quantile(values, [1.0 / 3.0, 2.0 / 3.0])
    if band == "low":
        selected = [r for r in valid if float(r["elevation_deg"]) <= float(q1)]
        desc = f"elevation <= {float(q1):.3f}"
    elif band == "high":
        selected = [r for r in valid if float(r["elevation_deg"]) >= float(q2)]
        desc = f"elevation >= {float(q2):.3f}"
    else:
        selected = [r for r in valid if float(q1) <= float(r["elevation_deg"]) <= float(q2)]
        desc = f"{float(q1):.3f} <= elevation <= {float(q2):.3f}"
    return selected or valid, {
        "elevation_band": band,
        "elevation_q33_deg": float(q1),
        "elevation_q66_deg": float(q2),
        "elevation_filter": desc,
    }


def select_by_trajectory(
    records: List[Dict[str, Any]],
    max_frames: Optional[int],
    trajectory_mode: str,
    arc_span_deg: float,
    arc_center_fraction: float,
    random_seed: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], List[Dict[str, Any]]]:
    mode = trajectory_mode.strip().lower()
    policy: Dict[str, Any] = {
        "type": mode,
        "max_frames": max_frames,
        "arc_span_deg": arc_span_deg,
        "arc_center_fraction": arc_center_fraction,
        "random_seed": int(random_seed),
    }

    if mode in {"balanced", "full_balanced"}:
        candidates = valid_quality_records(records)
        selected = select_balanced(candidates, max_frames)
    elif mode == "sorted_first":
        candidates = valid_quality_records(records)
        selected = select_sorted_first(candidates, max_frames)
    elif mode == "random":
        candidates = valid_quality_records(records)
        selected = select_random(candidates, max_frames, random_seed=random_seed)
    elif mode in {"arc", "arc_narrow", "arc_medium", "arc_wide"}:
        preset_span = {"arc_narrow": 70.0, "arc_medium": 140.0, "arc_wide": 220.0}.get(mode)
        candidates, details = filter_arc_records(
            records,
            arc_span_deg=float(preset_span if preset_span is not None else arc_span_deg),
            arc_center_fraction=arc_center_fraction,
        )
        policy.update(details)
        selected = select_balanced(candidates, max_frames)
    elif mode in {"elevation_low", "elevation_mid", "elevation_high"}:
        band = mode.split("_", 1)[1]
        candidates, details = filter_elevation_records(records, band=band)
        policy.update(details)
        selected = select_balanced(candidates, max_frames)
    else:
        raise ValueError(
            "Unsupported trajectory_mode="
            f"{trajectory_mode}; use balanced, sorted_first, random, arc, "
            "arc_narrow, arc_medium, arc_wide, elevation_low, elevation_mid, elevation_high"
        )

    policy["candidate_count"] = len(candidates)
    policy["selected_count"] = len(selected)
    return selected, policy, candidates


def reset_dir(path: Path) -> Path:
    ensure_dir(path)
    for child in path.iterdir():
        if child.is_symlink() or child.is_file():
            child.unlink()
        elif child.is_dir():
            shutil.rmtree(child)
    return path


def make_contact_sheet(records: List[Dict[str, Any]], output_path: Path, thumb: int = 160) -> Optional[str]:
    if not records:
        return None
    cols = min(6, len(records))
    rows = int(math.ceil(len(records) / float(cols)))
    sheet = Image.new("RGB", (cols * thumb, rows * (thumb + 22)), (30, 30, 30))
    draw = ImageDraw.Draw(sheet)
    for i, record in enumerate(records):
        image = Image.open(record["image_path"]).convert("RGB")
        image.thumbnail((thumb, thumb), Image.Resampling.BILINEAR)
        x = (i % cols) * thumb
        y = (i // cols) * (thumb + 22)
        canvas = Image.new("RGB", (thumb, thumb), (0, 0, 0))
        canvas.paste(image, ((thumb - image.width) // 2, (thumb - image.height) // 2))
        sheet.paste(canvas, (x, y))
        draw.text((x + 4, y + thumb + 3), record["name"], fill=(240, 240, 240))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)
    return str(output_path)


def prepare_heimei_colmap_balanced(
    dataset_dir: Path,
    output_root: Path,
    case_name: str,
    max_frames: Optional[int],
    link_mode: str,
    min_mask_area: float,
    max_mask_area: float,
    max_border_touch: float,
    trajectory_mode: str = "balanced",
    arc_span_deg: float = 90.0,
    arc_center_fraction: float = 0.5,
    random_seed: int = 0,
) -> Dict[str, Any]:
    dataset_dir = dataset_dir.resolve()
    sparse_dir = dataset_dir / "sparse" / "0"
    images_txt = sparse_dir / "images.txt"
    if not images_txt.exists():
        raise FileNotFoundError(images_txt)

    rgb_dir = dataset_dir / "rgb"
    if not rgb_dir.exists():
        rgb_dir = dataset_dir / "images"
    images_dir = dataset_dir / "images"
    masks_dir = dataset_dir / "masks"
    if not rgb_dir.exists() or not masks_dir.exists():
        raise FileNotFoundError(f"Expected rgb/images and masks under {dataset_dir}")

    colmap_records = read_colmap_images_txt(images_txt)
    rgb_files = image_files(rgb_dir)
    records: List[Dict[str, Any]] = []
    for order, image_path in enumerate(rgb_files):
        name = image_path.name
        colmap = colmap_records.get(name)
        if colmap is None:
            continue
        mask_path = find_mask(masks_dir, image_path.stem)
        stats = mask_stats(mask_path)
        record: Dict[str, Any] = {
            "source_order": order,
            "name": name,
            "image_path": str(image_path),
            "images_path": str(images_dir / name) if (images_dir / name).exists() else str(image_path),
            "camera_id": int(colmap["camera_id"]),
            "image_id": int(colmap["image_id"]),
            "qvec": colmap["qvec"],
            "tvec": colmap["tvec"],
            "camera_center": colmap["camera_center"],
            "sharpness": image_sharpness(image_path),
            **stats,
        }
        records.append(record)
    if not records:
        raise RuntimeError(f"No rgb frames matched COLMAP records under {dataset_dir}")

    coverage = add_pose_angles(records)
    for record in records:
        record["quality"] = frame_quality(record, min_mask_area, max_mask_area, max_border_touch)
    selected, trajectory_policy, candidate_records = select_by_trajectory(
        records=records,
        max_frames=max_frames,
        trajectory_mode=trajectory_mode,
        arc_span_deg=arc_span_deg,
        arc_center_fraction=arc_center_fraction,
        random_seed=random_seed,
    )

    workspace = ensure_dir(output_root / "workspace")
    dst_dir = ensure_dir(workspace / "datasets" / case_name)
    run_dir = ensure_dir(output_root / "runs" / case_name)
    reset_dir(dst_dir / "rgb")
    reset_dir(dst_dir / "images")
    reset_dir(dst_dir / "masks")
    ensure_dir(dst_dir / "models")

    for src_name in ["sparse"]:
        src = dataset_dir / src_name
        if src.exists():
            dst = dst_dir / src_name
            if dst.exists() or dst.is_symlink():
                if dst.is_symlink() or dst.is_file():
                    dst.unlink()
                elif link_mode == "copy":
                    shutil.rmtree(dst)
            if not dst.exists():
                link_or_copy(src, dst, mode=link_mode)

    source_models = dataset_dir / "models"
    if source_models.exists():
        copy_tree_contents(source_models, dst_dir / "models")

    for record in selected:
        src_rgb = Path(record["image_path"])
        src_img = Path(record["images_path"])
        mask_path = Path(record["mask_path"]) if record.get("mask_path") else None
        link_or_copy(src_rgb, dst_dir / "rgb" / src_rgb.name, mode=link_mode)
        link_or_copy(src_img, dst_dir / "images" / src_rgb.name, mode=link_mode)
        if mask_path and mask_path.exists():
            link_or_copy(mask_path, dst_dir / "masks" / f"{src_rgb.stem}.png", mode=link_mode)

    selected_for_json = []
    for record in selected:
        item = dict(record)
        item.pop("_quality_norm", None)
        selected_for_json.append(item)

    all_for_json = []
    for record in records:
        item = dict(record)
        item.pop("_quality_norm", None)
        all_for_json.append(item)

    source_model_mesh = dataset_dir / "models" / f"{dataset_dir.name}.obj"
    source_model_norm_mesh = dataset_dir / "models" / f"{dataset_dir.name}_norm.obj"
    if not source_model_mesh.exists() and source_models.exists():
        candidates = sorted(source_models.glob("*.obj"))
        source_model_mesh = candidates[0] if candidates else source_model_mesh

    selected_angles = [float(r["azimuth_unwrapped_deg"]) for r in selected]
    selected_elev = [float(r["elevation_deg"]) for r in selected]
    selected_mask = [float(r["mask_area_ratio"]) for r in selected if r.get("mask_area_ratio") is not None]
    candidate_angles = [float(r["azimuth_unwrapped_deg"]) for r in candidate_records]
    candidate_elev = [float(r["elevation_deg"]) for r in candidate_records]
    selected_coverage = {
        "selected_count": len(selected),
        "selected_names": [r["name"] for r in selected],
        "selected_indices": [int(r["source_order"]) for r in selected],
        "selected_azimuth_span_deg": float(max(selected_angles) - min(selected_angles)) if selected_angles else 0.0,
        "selected_elevation_range_deg": float(max(selected_elev) - min(selected_elev)) if selected_elev else 0.0,
        "selected_mask_area_mean": float(np.mean(selected_mask)) if selected_mask else None,
        "selected_mask_area_min": float(np.min(selected_mask)) if selected_mask else None,
        "selected_mask_area_max": float(np.max(selected_mask)) if selected_mask else None,
    }
    candidate_coverage = {
        "candidate_count": len(candidate_records),
        "candidate_azimuth_span_deg": float(max(candidate_angles) - min(candidate_angles)) if candidate_angles else 0.0,
        "candidate_elevation_range_deg": float(max(candidate_elev) - min(candidate_elev)) if candidate_elev else 0.0,
    }
    contact_sheet = make_contact_sheet(selected, run_dir / "selected_frames_contact_sheet.jpg")

    report: Dict[str, Any] = {
        "case_name": case_name,
        "source_dataset_dir": str(dataset_dir),
        "dataset_dir": str(dst_dir),
        "workspace": str(workspace),
        "models_dir": str(dst_dir / "models"),
        "run_dir": str(run_dir),
        "input_type": "heimei_colmap_coverage_sweep",
        "link_mode": link_mode,
        "max_frames": max_frames,
        "trajectory_mode": trajectory_mode,
        "source_frame_count": len(rgb_files),
        "colmap_matched_frame_count": len(records),
        "coverage": coverage,
        "candidate_coverage": candidate_coverage,
        "selected_coverage": selected_coverage,
        "selected_frames": [r["name"] for r in selected],
        "selected_indices": [int(r["source_order"]) for r in selected],
        "frames": selected_for_json,
        "all_frame_summary": all_for_json,
        "source_model_mesh": str(source_model_mesh) if source_model_mesh.exists() else None,
        "source_model_norm_mesh": str(source_model_norm_mesh) if source_model_norm_mesh.exists() else None,
        "contact_sheet": contact_sheet,
        "selection_policy": {
            **trajectory_policy,
            "center": "mean camera center",
            "min_mask_area": min_mask_area,
            "max_mask_area": max_mask_area,
            "max_border_touch": max_border_touch,
            "quality": "log1p(laplacian_sharpness) with mask/border penalties",
        },
    }
    write_json(dst_dir / "reconviagen_meta.json", report)
    write_json(run_dir / "prepared_sample.json", report)
    write_json(run_dir / "selected_frames.json", selected_coverage)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", default="/home/zjr/Tracker/CoarseModel/datasets/heimei")
    parser.add_argument("--output_root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--case_name", default=None)
    parser.add_argument("--max_frames", type=int, default=18, help="<=0 uses all valid COLMAP frames")
    parser.add_argument("--link_mode", choices=["symlink", "copy"], default="symlink")
    parser.add_argument("--min_mask_area", type=float, default=0.02)
    parser.add_argument("--max_mask_area", type=float, default=0.85)
    parser.add_argument("--max_border_touch", type=float, default=0.20)
    parser.add_argument(
        "--trajectory_mode",
        default="balanced",
        choices=[
            "balanced",
            "full_balanced",
            "sorted_first",
            "random",
            "arc",
            "arc_narrow",
            "arc_medium",
            "arc_wide",
            "elevation_low",
            "elevation_mid",
            "elevation_high",
        ],
    )
    parser.add_argument("--arc_span_deg", type=float, default=90.0)
    parser.add_argument("--arc_center_fraction", type=float, default=0.5)
    parser.add_argument("--random_seed", type=int, default=0)
    args = parser.parse_args()

    max_frames = None if args.max_frames <= 0 else int(args.max_frames)
    case_name = args.case_name or (
        f"heimei_colmap_{args.trajectory_mode}{max_frames}" if max_frames is not None else f"heimei_colmap_{args.trajectory_mode}_all"
    )
    report = prepare_heimei_colmap_balanced(
        dataset_dir=Path(args.dataset_dir),
        output_root=Path(args.output_root),
        case_name=case_name,
        max_frames=max_frames,
        link_mode=args.link_mode,
        min_mask_area=args.min_mask_area,
        max_mask_area=args.max_mask_area,
        max_border_touch=args.max_border_touch,
        trajectory_mode=args.trajectory_mode,
        arc_span_deg=args.arc_span_deg,
        arc_center_fraction=args.arc_center_fraction,
        random_seed=args.random_seed,
    )
    print(
        json.dumps(
            {
                "prepared_sample": str(Path(report["run_dir"]) / "prepared_sample.json"),
                "dataset_dir": report["dataset_dir"],
                "contact_sheet": report["contact_sheet"],
                "coverage": report["coverage"],
                "selected_coverage": report["selected_coverage"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
