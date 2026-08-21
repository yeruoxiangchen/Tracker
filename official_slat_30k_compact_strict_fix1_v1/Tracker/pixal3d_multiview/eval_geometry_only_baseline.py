from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

TRACKER_ROOT = Path(__file__).resolve().parents[1]
PIXAL3D_ROOT = TRACKER_ROOT / "Pixal3D"
sys.path.insert(0, str(TRACKER_ROOT))
sys.path.insert(0, str(PIXAL3D_ROOT))

from pixal3d_multiview.multiview_projection import (  # noqa: E402
    compute_visual_hull_score_grid,
    estimate_object_volume_from_visual_hull,
    occupancy_to_surface_coords,
)
from pixal3d_multiview.sample_sparse_checkpoint import (  # noqa: E402
    load_target_coords,
    make_preview,
    sparse_overlap_metrics,
    write_ply,
)
from pixal3d_multiview.train_sparse_multiview import (  # noqa: E402
    MultiviewSparseManifestDataset,
    POSE_MODES,
    apply_pose_mode,
)


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
    if not out:
        out = [0]
    bad = [idx for idx in out if idx < 0 or idx >= total]
    if bad:
        raise IndexError(f"indices out of range for dataset size={total}: {bad}")
    return out


def parse_pose_modes(spec: str) -> list[str]:
    modes = [part.strip() for part in spec.split(",") if part.strip()]
    valid = set(POSE_MODES)
    bad = [mode for mode in modes if mode not in valid]
    if bad:
        raise ValueError(f"Unknown pose modes: {bad}")
    return modes or ["correct"]


def parse_pred_modes(spec: str) -> list[str]:
    modes = [part.strip() for part in spec.split(",") if part.strip()]
    valid = {"vh_surface", "vh_volume", "topk_score", "topk_surface_score"}
    bad = [mode for mode in modes if mode not in valid]
    if bad:
        raise ValueError(f"Unknown prediction modes: {bad}")
    return modes or ["vh_surface"]


def flat_to_coords(flat: torch.Tensor, resolution: int) -> torch.Tensor:
    flat = flat.long()
    x = flat // (resolution * resolution)
    y = (flat // resolution) % resolution
    z = flat % resolution
    return torch.stack([x, y, z], dim=1).int()


def coords_to_np(coords: torch.Tensor) -> np.ndarray:
    return coords.detach().cpu().int().numpy()


def topk_coords(score: torch.Tensor, support: torch.Tensor, visible: torch.Tensor, k: int, *, candidates: torch.Tensor | None = None) -> torch.Tensor:
    resolution = int(score.shape[0])
    score_flat = score.reshape(-1).float()
    support_flat = support.reshape(-1).float()
    visible_flat = visible.reshape(-1).float()
    view_norm = visible_flat.max().clamp_min(1.0)
    rank = score_flat + 0.05 * support_flat / view_norm + 0.01 * visible_flat / view_norm

    if candidates is not None and candidates.numel() > 0:
        cand = candidates.long()
        cand_flat = cand[:, 0] * resolution * resolution + cand[:, 1] * resolution + cand[:, 2]
        rank_values = rank[cand_flat]
        take = min(int(k), int(cand_flat.numel()))
        if take <= 0:
            return torch.zeros((0, 3), device=score.device, dtype=torch.int32)
        top = torch.topk(rank_values, k=take, largest=True).indices
        return cand[top].int()

    valid = visible_flat > 0
    valid_ids = torch.where(valid)[0]
    if valid_ids.numel() == 0 or k <= 0:
        return torch.zeros((0, 3), device=score.device, dtype=torch.int32)
    take = min(int(k), int(valid_ids.numel()))
    top = torch.topk(rank[valid_ids], k=take, largest=True).indices
    return flat_to_coords(valid_ids[top], resolution)


def summarize(rows: list[dict]) -> list[dict]:
    keys = ["iou", "target_recall", "pred_precision", "pred_unique", "target_unique"]
    groups: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        groups.setdefault((row["pose_mode"], row["pred_mode"]), []).append(row)
    summary_rows = []
    for (pose_mode, pred_mode), selected in sorted(groups.items()):
        out = {"pose_mode": pose_mode, "pred_mode": pred_mode, "count": len(selected)}
        for key in keys:
            values = np.asarray([float(row[key]) for row in selected], dtype=np.float64)
            out[f"{key}_mean"] = float(values.mean()) if values.size else None
            out[f"{key}_median"] = float(np.median(values)) if values.size else None
        summary_rows.append(out)
    return summary_rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in keys})


def write_summary_markdown(path: Path, result: dict) -> None:
    lines = [
        "# Geometry-only Sparse Baseline",
        "",
        f"时间：`{result['timestamp_utc']}`",
        f"manifest：`{result['manifest']}`",
        f"indices：`{result['indices']}`",
        f"pose_modes：`{result['pose_modes']}`",
        f"pred_modes：`{result['pred_modes']}`",
        "",
        "## 汇总",
        "",
        "| pose | pred mode | IoU | recall | precision | pred unique | target unique |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in result["summary"]:
        lines.append(
            "| {pose_mode} | {pred_mode} | {iou_mean:.4f} | {target_recall_mean:.4f} | "
            "{pred_precision_mean:.4f} | {pred_unique_mean:.1f} | {target_unique_mean:.1f} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## 如何阅读",
            "",
            "- `vh_surface`：mask + pose 形成 visual hull 后取 surface coords。",
            "- `vh_volume`：visual hull 内部所有 occupied coords。",
            "- `topk_score`：按 geometry support score 选出与 target 数量相近的 grid coords。",
            "- `topk_surface_score`：只在 visual hull surface 候选上按 score 选 top-k。",
            "- 如果 `correct` 不能稳定高于 `shuffle/identity`，说明几何信号本身或 object volume 估计还不足以约束 sparse stage。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Geometry-only sparse baseline from masks and camera poses.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--indices", default="0,1,5,10,20,30,50,80,100")
    parser.add_argument("--pose_modes", default="correct,shuffle,reverse,noise,large_noise,identity")
    parser.add_argument("--pred_modes", default="vh_surface,vh_volume,topk_score,topk_surface_score")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_frames", type=int, default=8)
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--camera_forward_sign", type=float, default=1.0)
    parser.add_argument("--mask_threshold", type=float, default=0.5)
    parser.add_argument("--no_apply_mask", action="store_true")
    parser.add_argument("--vh_min_visible_views", type=int, default=1)
    parser.add_argument("--vh_min_support_views", type=int, default=2)
    parser.add_argument("--vh_min_support_ratio", type=float, default=0.6)
    parser.add_argument("--vh_volume_resolution", type=int, default=48)
    parser.add_argument("--vh_volume_initial_extent_ratio", type=float, default=0.6)
    parser.add_argument("--vh_volume_padding", type=float, default=1.25)
    parser.add_argument("--vh_volume_min_extent", type=float, default=0.05)
    parser.add_argument("--vh_volume_refine_steps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--save_previews", action="store_true")
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = MultiviewSparseManifestDataset(
        args.manifest,
        max_frames=args.max_frames,
        apply_mask=not args.no_apply_mask,
    )
    indices = parse_indices(args.indices, len(dataset))
    pose_modes = parse_pose_modes(args.pose_modes)
    pred_modes = parse_pred_modes(args.pred_modes)

    rows: list[dict] = []
    for sample_index in tqdm(indices, desc="Geometry-only", unit="sample", dynamic_ncols=True):
        batch = dataset[sample_index]
        target_coords = load_target_coords(batch["latent_path"])
        target_unique = sparse_overlap_metrics(np.zeros((0, 3), dtype=np.int32), target_coords)["target_unique"]

        for pose_mode in pose_modes:
            cond_batch = apply_pose_mode(batch, pose_mode, args.seed + sample_index * 104729)
            masks = cond_batch["masks"].to(device=device, dtype=torch.float32)
            intrinsics = cond_batch["intrinsics"].to(device=device, dtype=torch.float32)
            extrinsics = cond_batch["extrinsics"].to(device=device, dtype=torch.float32)
            extrinsics_are_c2w = str(cond_batch["extrinsics_type"]).lower() == "c2w"

            volume = estimate_object_volume_from_visual_hull(
                masks,
                intrinsics,
                extrinsics,
                extrinsics_are_c2w=extrinsics_are_c2w,
                camera_forward_sign=args.camera_forward_sign,
                mask_threshold=args.mask_threshold,
                resolution=args.vh_volume_resolution,
                min_visible_views=args.vh_min_visible_views,
                min_support_views=args.vh_min_support_views,
                min_support_ratio=args.vh_min_support_ratio,
                initial_extent_ratio=args.vh_volume_initial_extent_ratio,
                padding=args.vh_volume_padding,
                min_extent=args.vh_volume_min_extent,
                refine_steps=args.vh_volume_refine_steps,
            )
            score, support, visible = compute_visual_hull_score_grid(
                masks,
                intrinsics,
                extrinsics,
                extrinsics_are_c2w=extrinsics_are_c2w,
                camera_forward_sign=args.camera_forward_sign,
                object_to_world=volume.object_to_world,
                resolution=args.resolution,
                mask_threshold=args.mask_threshold,
                min_visible_views=args.vh_min_visible_views,
            )
            occupied = (
                (visible >= int(args.vh_min_visible_views))
                & (support >= int(args.vh_min_support_views))
                & (score >= float(args.vh_min_support_ratio))
            )
            volume_coords = torch.nonzero(occupied, as_tuple=False).int()
            surface_coords = occupancy_to_surface_coords(occupied)

            predictions: dict[str, torch.Tensor] = {
                "vh_surface": surface_coords,
                "vh_volume": volume_coords,
                "topk_score": topk_coords(score, support, visible, int(target_unique)),
                "topk_surface_score": topk_coords(score, support, visible, int(target_unique), candidates=surface_coords),
            }

            for pred_mode in pred_modes:
                pred_coords = coords_to_np(predictions[pred_mode])
                metrics = sparse_overlap_metrics(pred_coords, target_coords)
                sample_dir = output_dir / "samples" / f"{pred_mode}_{pose_mode}_idx{sample_index:04d}"
                row = {
                    "sample_index": int(sample_index),
                    "uid": batch["uid"],
                    "pose_mode": pose_mode,
                    "pose_permutation": cond_batch.get("pose_permutation"),
                    "pred_mode": pred_mode,
                    "output_dir": str(sample_dir),
                    "volume_fallback": bool(volume.fallback),
                    "volume_extent_world": float(volume.extent_world),
                    "volume_occupied_ratio": float(volume.occupied_ratio),
                    "vh_score_mean": float(score.mean().item()),
                    "vh_support_mean": float(support.float().mean().item()),
                    "vh_visible_mean": float(visible.float().mean().item()),
                    **metrics,
                }
                rows.append(row)
                if args.save_previews:
                    sample_dir.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(sample_dir / "geometry_sparse.npz", pred_coords=pred_coords, target_coords=target_coords)
                    write_ply(sample_dir / "pred_sparse_coords.ply", pred_coords, resolution=args.resolution, color=(240, 220, 80))
                    write_ply(sample_dir / "target_sparse_coords.ply", target_coords, resolution=args.resolution, color=(80, 200, 255))
                    make_preview(pred_coords, target_coords, resolution=args.resolution, path=sample_dir / "sparse_preview.png")
                    (sample_dir / "summary.json").write_text(json.dumps(row, indent=2, ensure_ascii=False), encoding="utf-8")

    summary_rows = summarize(rows)
    result = {
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "manifest": args.manifest,
        "indices": indices,
        "pose_modes": pose_modes,
        "pred_modes": pred_modes,
        "resolution": args.resolution,
        "max_frames": args.max_frames,
        "vh_min_visible_views": args.vh_min_visible_views,
        "vh_min_support_views": args.vh_min_support_views,
        "vh_min_support_ratio": args.vh_min_support_ratio,
        "summary": summary_rows,
        "rows": rows,
    }
    (output_dir / "summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(output_dir / "metrics.csv", rows)
    write_csv(output_dir / "summary.csv", summary_rows)
    write_summary_markdown(output_dir / "summary.md", result)
    print(json.dumps({"output_dir": str(output_dir), "summary": summary_rows}, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
