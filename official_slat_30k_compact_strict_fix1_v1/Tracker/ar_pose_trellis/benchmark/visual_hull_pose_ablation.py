#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


TRACKER_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TRACKER_ROOT))

from ar_pose_trellis.benchmark.evaluate_sparse_pose_ablation import (  # noqa: E402
    VALID_MODES,
    compare_sparse_coords,
    load_reference_coords,
    make_manifest_variant,
)
from ar_pose_trellis.visual_hull import visual_hull_coords  # noqa: E402


def load_testsets(path: str) -> list[dict]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    datasets = payload.get("datasets", payload if isinstance(payload, list) else None)
    if not datasets:
        raise ValueError(f"No datasets found in {path}")
    return datasets


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def stable_int_seed(*parts: str, base_seed: int = 0) -> int:
    digest = hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()
    return int((base_seed + int(digest[:8], 16)) % (2**32 - 1))


def resolve_path(root: str | None, path: str) -> Path:
    p = Path(path)
    if p.is_absolute() or root is None:
        return p
    return Path(root) / p


def load_manifest_tensors(
    manifest_path: Path,
    image_root: str | None,
    mask_root: str | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
    data = load_json(manifest_path)
    frames = data.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError(f"Manifest has no frames: {manifest_path}")

    masks = []
    intrinsics = []
    extrinsics = []
    top_k = data.get("intrinsic")
    for frame in frames:
        mask_name = frame.get("mask")
        if mask_name is None:
            image_path = resolve_path(image_root, frame["image"])
            rgba = Image.open(image_path).convert("RGBA")
            mask = np.asarray(rgba)[..., 3]
        else:
            mask_path = resolve_path(mask_root, mask_name)
            mask = np.asarray(Image.open(mask_path).convert("L"))
        masks.append(torch.from_numpy(mask.astype(np.float32) / 255.0)[None])

        k = frame.get("intrinsic", top_k)
        if k is None:
            raise ValueError(f"No intrinsic for frame in {manifest_path}: {frame}")
        intrinsics.append(torch.tensor(k, dtype=torch.float32))
        extrinsics.append(torch.tensor(frame["extrinsic"], dtype=torch.float32))

    extrinsics_type = data.get("extrinsics_type", "c2w")
    return torch.stack(masks, dim=0), torch.stack(intrinsics, dim=0), torch.stack(extrinsics, dim=0), extrinsics_type


def flatten_row(row: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for key, value in row.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                out[f"{key}.{sub_key}"] = sub_value
        else:
            out[key] = value
    return out


def run_visual_hull_case(
    item: dict,
    mode: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    name = item["name"]
    source_manifest = Path(item["manifest"]).resolve()
    case_root = Path(args.output_root) / name
    manifest_path = case_root / "manifests" / f"{mode}.json"
    make_manifest_variant(
        source_manifest,
        manifest_path,
        mode=mode,
        max_frames=args.max_frames,
        seed=stable_int_seed(name, mode, base_seed=args.seed),
        noise_rot_deg=args.noise_rot_deg,
        noise_trans_std=args.noise_trans_std,
    )

    image_root = item.get("image_root")
    mask_root = item.get("mask_root")
    if image_root is None or mask_root is None:
        manifest = load_json(source_manifest)
        image_root = manifest.get("image_root")
        mask_root = manifest.get("mask_root")
    if image_root is None or mask_root is None:
        raise ValueError(f"{name} needs image_root and mask_root")

    masks, intrinsics, extrinsics, extrinsics_type = load_manifest_tensors(manifest_path, image_root, mask_root)
    coords_t, vh_stats = visual_hull_coords(
        masks,
        intrinsics,
        extrinsics,
        extrinsics_are_c2w=extrinsics_type == "c2w",
        resolution=args.resolution,
        mask_threshold=args.mask_threshold,
        min_visible_views=args.min_visible_views,
        min_support_views=args.min_support_views,
        min_support_ratio=args.min_support_ratio,
        surface_only=not args.keep_solid,
        chunk_size=args.chunk_size,
    )

    output_dir = case_root / mode
    output_dir.mkdir(parents=True, exist_ok=True)
    coords_np = coords_t.detach().cpu().numpy().astype(np.int32)
    coords_path = output_dir / "visual_hull_coords.npy"
    stats_path = output_dir / "visual_hull_stats.json"
    np.save(coords_path, coords_np)
    write_json(stats_path, vh_stats.to_dict())

    ref_path = Path(item["reference_coords"]).resolve()
    ref_coords = load_reference_coords(ref_path)
    metrics = compare_sparse_coords(
        coords_np,
        ref_coords,
        sample_points=args.sample_points,
        seed=stable_int_seed(name, mode, "metrics", base_seed=args.seed),
        fscore_thresholds=args.fscore_thresholds,
    )
    print(
        f"[visual_hull] {name}/{mode}: pred={metrics['pred_count']} "
        f"iou={metrics['voxel_iou']:.4f} chamfer={metrics['chamfer_vox']}",
        flush=True,
    )
    return {
        "name": name,
        "mode": mode,
        "coords_path": str(coords_path),
        "stats_path": str(stats_path),
        "reference_coords": str(ref_path),
        "visual_hull_stats": vh_stats.to_dict(),
        "metrics": metrics,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pure geometry pose ablation using visual hull from masks, intrinsics, and camera poses."
    )
    parser.add_argument("--testsets", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--modes", default="correct,identity,shuffle,noise")
    parser.add_argument("--max_frames", type=int, default=8)
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--mask_threshold", type=float, default=0.5)
    parser.add_argument("--min_visible_views", type=int, default=1)
    parser.add_argument("--min_support_views", type=int, default=2)
    parser.add_argument("--min_support_ratio", type=float, default=0.6)
    parser.add_argument("--keep_solid", action="store_true", help="Keep solid visual hull occupancy instead of surface voxels.")
    parser.add_argument("--noise_rot_deg", type=float, default=15.0)
    parser.add_argument("--noise_trans_std", type=float, default=0.10)
    parser.add_argument("--sample_points", type=int, default=8000)
    parser.add_argument("--fscore_thresholds", default="1,2,4")
    parser.add_argument("--chunk_size", type=int, default=32768)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--continue_on_error", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    bad_modes = [mode for mode in modes if mode not in VALID_MODES]
    if bad_modes:
        raise ValueError(f"Unsupported modes {bad_modes}; valid modes={VALID_MODES}")
    args.fscore_thresholds = [float(x) for x in args.fscore_thresholds.split(",") if x.strip()]

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    datasets = load_testsets(args.testsets)
    reports = []
    errors = []
    for item in datasets:
        for mode in modes:
            try:
                reports.append(run_visual_hull_case(item, mode, args))
            except Exception as exc:
                if not args.continue_on_error:
                    raise
                error = {"name": item.get("name"), "mode": mode, "error": f"{type(exc).__name__}: {exc}"}
                errors.append(error)
                print(f"[visual_hull] ERROR {error}", flush=True)

    payload = {
        "testsets": args.testsets,
        "modes": modes,
        "resolution": args.resolution,
        "mask_threshold": args.mask_threshold,
        "min_visible_views": args.min_visible_views,
        "min_support_views": args.min_support_views,
        "min_support_ratio": args.min_support_ratio,
        "surface_only": not args.keep_solid,
        "reports": reports,
        "errors": errors,
    }
    report_path = output_root / "visual_hull_pose_ablation_report.json"
    write_json(report_path, payload)
    csv_path = output_root / "visual_hull_pose_ablation_report.csv"
    rows = [flatten_row(row) for row in reports]
    if rows:
        fieldnames = sorted({key for row in rows for key in row.keys()})
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    print(f"[visual_hull] wrote {report_path}")
    print(f"[visual_hull] wrote {csv_path}")


if __name__ == "__main__":
    main()
