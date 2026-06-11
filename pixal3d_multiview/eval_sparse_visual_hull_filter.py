from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from tqdm.auto import tqdm

TRACKER_ROOT = Path(__file__).resolve().parents[1]
PIXAL3D_ROOT = TRACKER_ROOT / "Pixal3D"
sys.path.insert(0, str(TRACKER_ROOT))
sys.path.insert(0, str(PIXAL3D_ROOT))

from pixal3d_multiview.multiview_projection import (  # noqa: E402
    compute_visual_hull_score_grid,
    estimate_object_volume_from_visual_hull,
)
from pixal3d_multiview.sample_sparse_checkpoint import make_preview, sparse_overlap_metrics, write_ply  # noqa: E402
from pixal3d_multiview.train_sparse_multiview import MultiviewSparseManifestDataset  # noqa: E402


def parse_indices(spec: str, total: int) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            out.extend(range(int(start_s), int(end_s) + 1))
        else:
            out.append(int(part))
    bad = [idx for idx in out if idx < 0 or idx >= total]
    if bad:
        raise IndexError(f"indices out of range for dataset size={total}: {bad}")
    return out


def filter_coords_by_visual_hull(
    coords: np.ndarray,
    batch: dict,
    *,
    resolution: int,
    mask_threshold: float,
    min_visible_views: int,
    min_support_views: int,
    min_support_ratio: float,
    camera_forward_sign: float,
    device: torch.device,
) -> tuple[np.ndarray, dict]:
    if coords.size == 0:
        return coords, {"input_coords": 0, "kept_coords": 0, "keep_ratio": 0.0}

    masks = batch["masks"].to(device=device, dtype=torch.float32)
    intrinsics = batch["intrinsics"].to(device=device, dtype=torch.float32)
    extrinsics = batch["extrinsics"].to(device=device, dtype=torch.float32)
    extrinsics_are_c2w = str(batch["extrinsics_type"]).lower() == "c2w"
    volume = estimate_object_volume_from_visual_hull(
        masks,
        intrinsics,
        extrinsics,
        extrinsics_are_c2w=extrinsics_are_c2w,
        camera_forward_sign=camera_forward_sign,
        mask_threshold=mask_threshold,
        resolution=48,
        min_visible_views=min_visible_views,
        min_support_views=min_support_views,
        min_support_ratio=min_support_ratio,
        initial_extent_ratio=0.6,
        padding=1.25,
        min_extent=0.05,
        refine_steps=2,
    )
    score, support, visible = compute_visual_hull_score_grid(
        masks,
        intrinsics,
        extrinsics,
        extrinsics_are_c2w=extrinsics_are_c2w,
        camera_forward_sign=camera_forward_sign,
        object_to_world=volume.object_to_world,
        resolution=resolution,
        mask_threshold=mask_threshold,
        min_visible_views=min_visible_views,
    )
    valid = (visible >= int(min_visible_views)) & (support >= int(min_support_views)) & (score >= float(min_support_ratio))
    xyz = coords[:, -3:].astype(np.int64)
    xyz = np.clip(xyz, 0, resolution - 1)
    keep = valid[xyz[:, 0], xyz[:, 1], xyz[:, 2]].detach().cpu().numpy().astype(bool)
    filtered = coords[keep]
    stats = {
        "input_coords": int(coords.shape[0]),
        "kept_coords": int(filtered.shape[0]),
        "keep_ratio": float(filtered.shape[0] / max(coords.shape[0], 1)),
        "volume": volume.to_dict(),
        "vh_score_mean": float(score.float().mean().item()),
        "vh_support_mean": float(support.float().mean().item()),
        "vh_visible_mean": float(visible.float().mean().item()),
    }
    return filtered, stats


def make_comparison_grid(rows: list[dict], output_dir: Path) -> None:
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    except Exception:
        font = ImageFont.load_default()
    thumb_w, thumb_h = 220, 150
    label_h = 38
    margin = 10
    cols = 3
    canvas = Image.new("RGB", (cols * thumb_w + (cols + 1) * margin, len(rows) * (thumb_h + label_h) + margin), (245, 245, 245))
    draw = ImageDraw.Draw(canvas)
    for row_i, row in enumerate(rows):
        for col, key in enumerate(("raw_preview", "filtered_preview", "target_preview")):
            path = Path(row[key])
            img = Image.open(path).convert("RGB")
            img.thumbnail((thumb_w, thumb_h), Image.LANCZOS)
            x = margin + col * (thumb_w + margin)
            y = margin + row_i * (thumb_h + label_h)
            draw.rectangle([x, y, x + thumb_w, y + thumb_h + label_h - 4], fill=(255, 255, 255), outline=(210, 210, 210))
            if col == 0:
                title = f"raw idx={row['sample_index']} IoU {row['raw_iou']:.3f}"
            elif col == 1:
                title = f"vh-filter IoU {row['filtered_iou']:.3f}"
            else:
                title = "target"
            draw.text((x + 6, y + 6), title, fill=(30, 30, 30), font=font)
            if col == 1:
                draw.text((x + 6, y + 22), f"keep {row['keep_ratio']:.2f}", fill=(80, 80, 80), font=font)
            canvas.paste(img, (x + (thumb_w - img.width) // 2, y + label_h + (thumb_h - img.height) // 2))
    canvas.save(output_dir / "visual_hull_filter_grid.png")


def write_csv(path: Path, rows: list[dict]) -> None:
    keys = [
        "sample_index",
        "uid",
        "raw_pred_unique",
        "filtered_pred_unique",
        "target_unique",
        "raw_iou",
        "filtered_iou",
        "raw_target_recall",
        "filtered_target_recall",
        "raw_pred_precision",
        "filtered_pred_precision",
        "keep_ratio",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in keys})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter sampled sparse coords by visual-hull support and visualize raw vs filtered.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--sample_root", required=True, help="Root containing checkpoint_idxXXXX/sparse_sample.npz outputs.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--indices", default="0,1,5,10,20,30,50,80,100")
    parser.add_argument("--max_frames", type=int, default=8)
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--camera_forward_sign", type=float, default=1.0)
    parser.add_argument("--mask_threshold", type=float, default=0.5)
    parser.add_argument("--min_visible_views", type=int, default=1)
    parser.add_argument("--min_support_views", type=int, default=2)
    parser.add_argument("--min_support_ratio", type=float, default=0.6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_root = Path(args.sample_root)
    device = torch.device(args.device)
    dataset = MultiviewSparseManifestDataset(args.manifest, max_frames=args.max_frames, apply_mask=True)
    indices = parse_indices(args.indices, len(dataset))
    rows = []
    for sample_index in tqdm(indices, desc="Filtering", unit="sample", dynamic_ncols=True):
        batch = dataset[sample_index]
        npz_path = sample_root / f"checkpoint_idx{sample_index:04d}" / "sparse_sample.npz"
        with np.load(npz_path) as data:
            pred = data["pred_coords"].astype(np.int32)
            target = data["target_coords"].astype(np.int32)
        filtered, vh_stats = filter_coords_by_visual_hull(
            pred,
            batch,
            resolution=args.resolution,
            mask_threshold=args.mask_threshold,
            min_visible_views=args.min_visible_views,
            min_support_views=args.min_support_views,
            min_support_ratio=args.min_support_ratio,
            camera_forward_sign=args.camera_forward_sign,
            device=device,
        )
        raw_metrics = sparse_overlap_metrics(pred, target)
        filtered_metrics = sparse_overlap_metrics(filtered, target)
        sample_dir = output_dir / f"idx{sample_index:04d}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(sample_dir / "sparse_visual_hull_filter.npz", pred_raw=pred, pred_filtered=filtered, target_coords=target)
        write_ply(sample_dir / "pred_raw_sparse_coords.ply", pred, resolution=args.resolution, color=(240, 220, 80))
        write_ply(sample_dir / "pred_filtered_sparse_coords.ply", filtered, resolution=args.resolution, color=(80, 220, 120))
        write_ply(sample_dir / "target_sparse_coords.ply", target, resolution=args.resolution, color=(80, 200, 255))
        make_preview(pred, target, args.resolution, sample_dir / "raw_preview.png")
        make_preview(filtered, target, args.resolution, sample_dir / "filtered_preview.png")
        make_preview(target, target, args.resolution, sample_dir / "target_preview.png")
        row = {
            "sample_index": int(sample_index),
            "uid": batch["uid"],
            "raw_pred_unique": raw_metrics["pred_unique"],
            "filtered_pred_unique": filtered_metrics["pred_unique"],
            "target_unique": raw_metrics["target_unique"],
            "raw_iou": raw_metrics["iou"],
            "filtered_iou": filtered_metrics["iou"],
            "raw_target_recall": raw_metrics["target_recall"],
            "filtered_target_recall": filtered_metrics["target_recall"],
            "raw_pred_precision": raw_metrics["pred_precision"],
            "filtered_pred_precision": filtered_metrics["pred_precision"],
            "keep_ratio": vh_stats["keep_ratio"],
            "raw_preview": str(sample_dir / "raw_preview.png"),
            "filtered_preview": str(sample_dir / "filtered_preview.png"),
            "target_preview": str(sample_dir / "target_preview.png"),
            "vh_stats": vh_stats,
        }
        rows.append(row)
        (sample_dir / "summary.json").write_text(json.dumps(row, indent=2, ensure_ascii=False), encoding="utf-8")
    result = {
        "manifest": args.manifest,
        "sample_root": str(sample_root),
        "indices": indices,
        "filter": {
            "min_visible_views": args.min_visible_views,
            "min_support_views": args.min_support_views,
            "min_support_ratio": args.min_support_ratio,
        },
        "rows": rows,
    }
    (output_dir / "summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(output_dir / "metrics.csv", rows)
    make_comparison_grid(rows, output_dir)
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
