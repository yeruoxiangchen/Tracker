from __future__ import annotations

import argparse
import csv
import json
import random
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

from pixal3d_multiview.pose_consistency_head import load_pose_consistency_head  # noqa: E402
from pixal3d_multiview.train_pose_consistency_head import (  # noqa: E402
    POSE_CONSISTENCY_MODES,
    make_cross_sample_batch,
    parse_modes,
    score_batch,
    setup_condition_builder,
)
from pixal3d_multiview.train_sparse_multiview import (  # noqa: E402
    IMAGE_COND_CONFIG,
    MultiviewSparseManifestDataset,
    apply_pose_mode,
    build_image_cond_model,
)


def parse_indices(spec: str, total: int) -> list[int]:
    out: list[int] = []
    for part in str(spec).split(","):
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


def deterministic_cross_sample(dataset, sample_index: int, seed: int) -> dict:
    if len(dataset) <= 1:
        return dataset[sample_index]
    other = (int(sample_index) * 1103515245 + int(seed) * 12345 + 17) % len(dataset)
    if other == sample_index:
        other = (other + 1) % len(dataset)
    return dataset[int(other)]


def write_csv(path: Path, rows: list[dict], keys: list[str] | None = None) -> None:
    if not rows:
        return
    if keys is None:
        keys = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in keys})


def summarize(rows: list[dict], modes: list[str]) -> list[dict]:
    out = []
    for mode in modes:
        selected = [row for row in rows if row["pose_mode"] == mode]
        if not selected:
            continue
        score = np.asarray([row["score"] for row in selected], dtype=np.float64)
        keep = np.asarray([row["keep_ratio"] for row in selected], dtype=np.float64)
        out.append(
            {
                "pose_mode": mode,
                "count": len(selected),
                "score_mean": float(score.mean()),
                "score_median": float(np.median(score)),
                "score_min": float(score.min()),
                "score_max": float(score.max()),
                "keep_mean": float(keep.mean()),
                "keep_median": float(np.median(keep)),
            }
        )
    return out


def pairwise(rows: list[dict], modes: list[str], reference: str) -> list[dict]:
    by_mode = {
        mode: {int(row["sample_index"]): row for row in rows if row["pose_mode"] == mode}
        for mode in modes
    }
    ref = by_mode.get(reference, {})
    out = []
    for mode in modes:
        if mode == reference:
            continue
        other = by_mode.get(mode, {})
        indices = sorted(set(ref) & set(other))
        if not indices:
            continue
        for metric in ("score", "keep_ratio"):
            deltas = np.asarray([float(ref[idx][metric]) - float(other[idx][metric]) for idx in indices], dtype=np.float64)
            out.append(
                {
                    "reference_pose": reference,
                    "wrong_pose": mode,
                    "metric": metric,
                    "count": int(deltas.size),
                    "mean_delta": float(deltas.mean()),
                    "median_delta": float(np.median(deltas)),
                    "reference_wins": int((deltas > 0).sum()),
                    "reference_win_rate": float((deltas > 0).mean()),
                }
            )
    return out


def rank_summary(rows: list[dict], modes: list[str], reference: str) -> list[dict]:
    by_mode = {
        mode: {int(row["sample_index"]): row for row in rows if row["pose_mode"] == mode}
        for mode in modes
    }
    common = sorted(set.intersection(*(set(values) for values in by_mode.values()))) if by_mode else []
    out = []
    for metric in ("score", "keep_ratio"):
        ranks = []
        top1 = 0
        for idx in common:
            values = {mode: float(by_mode[mode][idx][metric]) for mode in modes}
            ref_value = values[reference]
            rank = 1 + sum(value > ref_value for mode, value in values.items() if mode != reference)
            ranks.append(rank)
            top1 += int(rank == 1)
        if ranks:
            arr = np.asarray(ranks, dtype=np.float64)
            out.append(
                {
                    "metric": metric,
                    "count": len(ranks),
                    "reference_pose": reference,
                    "reference_top1": top1,
                    "reference_top1_rate": float(top1 / len(ranks)),
                    "reference_rank_mean": float(arr.mean()),
                    "reference_rank_median": float(np.median(arr)),
                }
            )
    return out


def write_markdown(path: Path, result: dict, summary_rows: list[dict], pair_rows: list[dict], rank_rows: list[dict]) -> None:
    lines = [
        "# Pose Consistency Head Evaluation",
        "",
        f"time: `{result['timestamp_utc']}`",
        f"manifest: `{result['manifest']}`",
        f"head: `{result['head_checkpoint']}`",
        f"indices: `{result['indices']}`",
        f"pose_modes: `{', '.join(result['pose_modes'])}`",
        "",
        "## Score Summary",
        "",
        "| pose | count | score mean | score median | keep mean |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['pose_mode']} | {row['count']} | {row['score_mean']:.4f} | "
            f"{row['score_median']:.4f} | {row['keep_mean']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Correct-vs-Wrong",
            "",
            "| wrong pose | metric | mean delta | median delta | correct wins | win rate |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in pair_rows:
        lines.append(
            f"| {row['wrong_pose']} | {row['metric']} | {row['mean_delta']:.4f} | "
            f"{row['median_delta']:.4f} | {row['reference_wins']}/{row['count']} | {row['reference_win_rate']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Correct Rank",
            "",
            "| metric | top1 | top1 rate | rank mean | rank median |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in rank_rows:
        lines.append(
            f"| {row['metric']} | {row['reference_top1']}/{row['count']} | "
            f"{row['reference_top1_rate']:.3f} | {row['reference_rank_mean']:.3f} | "
            f"{row['reference_rank_median']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Reading",
            "",
            "- `score` 是 head 输出的 sample-level image-pose consistency score，越高表示当前 image-pose 配对越可信。",
            "- `keep_ratio` 是 gate/visible-pair 诊断值；pairwise 模式下来自 visible pair score 的全局加权均值。",
            "- pairwise head 会先输出 `pair_match_logits[V,V,N]`，再按 `pair_weight_mode` 聚合成 centered `logits[V,N]` 供 sparse view gate 使用。",
            "- `cross_sample` 只用于 head 评测/训练，不进入 sparse sweep 的单样本 pose corruption。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a pose consistency head.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--head_checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--image_cond_model", default=IMAGE_COND_CONFIG["model_name"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_frames", type=int, default=8)
    parser.add_argument("--indices", default="0-63")
    parser.add_argument("--pose_modes", default="correct,cyclic_shift1,cyclic_shift2,reverse,noise,large_noise,cross_sample,identity")
    parser.add_argument("--reference_pose", default="correct")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--ss_grid_resolution", type=int, default=16)
    parser.add_argument("--camera_forward_sign", type=float, default=1.0)
    parser.add_argument("--mask_threshold", type=float, default=0.5)
    parser.add_argument("--no_apply_mask", action="store_true")
    parser.add_argument("--no_auto_volume", action="store_true")
    parser.add_argument("--vh_min_visible_views", type=int, default=1)
    parser.add_argument("--vh_min_support_views", type=int, default=2)
    parser.add_argument("--vh_min_support_ratio", type=float, default=0.6)
    parser.add_argument("--vh_volume_resolution", type=int, default=48)
    parser.add_argument("--vh_volume_initial_extent_ratio", type=float, default=0.6)
    parser.add_argument("--vh_volume_padding", type=float, default=1.25)
    parser.add_argument("--vh_volume_min_extent", type=float, default=0.05)
    parser.add_argument("--vh_volume_refine_steps", type=int, default=2)
    parser.add_argument("--no_visibility_depth", action="store_true")
    parser.add_argument("--vh_visibility_resolution", type=int, default=48)
    parser.add_argument("--vh_visibility_dilation", type=int, default=3)
    parser.add_argument("--visibility_depth_tolerance", type=float, default=0.0)
    parser.add_argument("--visibility_depth_tolerance_ratio", type=float, default=0.15)
    parser.add_argument("--visibility_weight_min", type=float, default=0.05)
    parser.add_argument("--empty_policy", choices=["zero", "visible", "border", "soft"], default="zero")
    parser.add_argument("--fallback_weight", type=float, default=1.0)
    parser.add_argument("--support_confidence_power", type=float, default=1.0)
    parser.add_argument("--global_fusion", choices=["concat", "mean", "first"], default="concat")
    parser.add_argument("--geometry_feature_mode", choices=["none", "add", "replace"], default="none")
    parser.add_argument("--geometry_feature_scale", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pose_modes = parse_modes(args.pose_modes, allow_correct=True)
    if args.reference_pose not in pose_modes:
        raise ValueError(f"--reference_pose {args.reference_pose!r} should be included in --pose_modes")

    dataset = MultiviewSparseManifestDataset(args.manifest, max_frames=args.max_frames, apply_mask=not args.no_apply_mask)
    indices = parse_indices(args.indices, len(dataset))
    image_cond_model = build_image_cond_model(device, args.ss_grid_resolution, args.image_cond_model)
    feature_dim = int(getattr(image_cond_model, "embed_dim", image_cond_model.model.config.hidden_size))
    head = load_pose_consistency_head(args.head_checkpoint, feature_dim=feature_dim, device=device).eval()
    condition_builder = setup_condition_builder(args, image_cond_model, head, device)

    rows: list[dict] = []
    with torch.no_grad():
        for sample_index in tqdm(indices, desc="Pose consistency eval", unit="sample", dynamic_ncols=True):
            batch = dataset[sample_index]
            for pose_mode in pose_modes:
                if pose_mode == "correct":
                    cond_batch = batch
                elif pose_mode == "cross_sample":
                    other = deterministic_cross_sample(dataset, sample_index, args.seed)
                    cond_batch = make_cross_sample_batch(batch, other)
                else:
                    cond_batch = apply_pose_mode(batch, pose_mode, args.seed + sample_index * 104729)
                score, keep, stats = score_batch(condition_builder, image_cond_model, cond_batch, args, device)
                rows.append(
                    {
                        "sample_index": int(sample_index),
                        "uid": batch["uid"],
                        "pose_mode": pose_mode,
                        "score": float(score.detach().cpu().item()),
                        "keep_ratio": float(keep.detach().cpu().item()),
                        "gate_mean": stats.get("gate_mean"),
                        "gate_min": stats.get("gate_min"),
                        "gate_max": stats.get("gate_max"),
                        "valid_ratio": stats.get("valid_ratio"),
                        "score_mode": stats.get("score_mode"),
                        "pair_weight_mode": stats.get("pair_weight_mode"),
                        "pair_weight_threshold": stats.get("pair_weight_threshold"),
                        "pair_valid_ratio": stats.get("pair_valid_ratio"),
                        "pair_count_mean": stats.get("pair_count_mean"),
                        "pair_weight_mean": stats.get("pair_weight_mean"),
                        "view_weight_mean": stats.get("view_weight_mean"),
                        "view_weight_nonzero_ratio": stats.get("view_weight_nonzero_ratio"),
                        "pair_supported_voxel_ratio": stats.get("pair_supported_voxel_ratio"),
                        "pair_sample_score": stats.get("pair_sample_score"),
                        "pair_keep_ratio": stats.get("pair_keep_ratio"),
                        "pair_logit_mean": stats.get("pair_logit_mean"),
                        "pair_logit_min": stats.get("pair_logit_min"),
                        "pair_logit_max": stats.get("pair_logit_max"),
                        "view_prior_abs_mean": stats.get("view_prior_abs_mean"),
                        "pose_permutation": cond_batch.get("pose_permutation"),
                        "cross_sample_uid": cond_batch.get("cross_sample_uid"),
                    }
                )
    summary_rows = summarize(rows, pose_modes)
    pair_rows = pairwise(rows, pose_modes, args.reference_pose)
    rank_rows = rank_summary(rows, pose_modes, args.reference_pose)
    result = {
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "manifest": args.manifest,
        "head_checkpoint": args.head_checkpoint,
        "indices": indices,
        "pose_modes": pose_modes,
        "reference_pose": args.reference_pose,
        "summary": summary_rows,
        "pairwise": pair_rows,
        "rank_summary": rank_rows,
    }
    (output_dir / "score_summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(output_dir / "all_scores.csv", rows)
    write_csv(output_dir / "score_summary.csv", summary_rows)
    write_csv(output_dir / "score_pairwise.csv", pair_rows)
    write_csv(output_dir / "score_rank_summary.csv", rank_rows)
    write_markdown(output_dir / "score_report.md", result, summary_rows, pair_rows, rank_rows)
    print(json.dumps({"output_dir": str(output_dir), "summary": summary_rows, "rank": rank_rows}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
