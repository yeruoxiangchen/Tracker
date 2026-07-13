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

from pixal3d import models as pixal3d_models  # noqa: E402
from pixal3d_multiview.eval_fixed_train_loss import load_checkpoint_weights  # noqa: E402
from pixal3d_multiview.eval_sparse_sampling_batch import (  # noqa: E402
    parse_indices,
    sample_coords,
    sparse_overlap_metrics,
    write_csv,
)
from pixal3d_multiview.sample_sparse_checkpoint import load_target_coords, make_preview, write_ply  # noqa: E402
from pixal3d_multiview.sparse_condition import SparseMultiviewConditionBuilder  # noqa: E402
from pixal3d_multiview.pose_consistency_head import load_pose_consistency_head  # noqa: E402
from pixal3d_multiview.train_sparse_multiview import (  # noqa: E402
    MultiviewSparseManifestDataset,
    POSE_MODES,
    apply_pose_mode,
    build_image_cond_model,
    build_geometry_adapter,
    build_view_aggregator,
    load_sparse_flow_model,
    make_multiview_condition,
)


def parse_checkpoint_spec(spec: str) -> list[Path]:
    paths: list[Path] = []
    for part in spec.split(","):
        part = part.strip()
        if part:
            paths.append(Path(part).expanduser())
    if not paths:
        raise ValueError("--checkpoints must contain at least one path")
    return paths


def parse_pose_modes(spec: str) -> list[str]:
    poses = [part.strip() for part in spec.split(",") if part.strip()]
    valid = set(POSE_MODES)
    bad = [pose for pose in poses if pose not in valid]
    if bad:
        raise ValueError(f"Unknown pose modes: {bad}")
    if not poses:
        raise ValueError("--pose_modes must contain at least one pose")
    return poses


def checkpoint_tag(path: Path) -> str:
    return path.stem.replace(".", "_")


SUMMARY_METRICS = ("iou", "target_recall", "pred_precision", "pred_unique", "target_unique", "intersection")
RANK_METRICS = ("iou", "target_recall", "pred_precision")


def _metric_values(rows: list[dict], metric: str) -> np.ndarray:
    values = [row.get(metric) for row in rows if row.get(metric) is not None]
    return np.asarray(values, dtype=np.float64)


def metric_summary(rows: list[dict], metric: str) -> dict:
    values = _metric_values(rows, metric)
    if values.size == 0:
        return {}
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def summarize_pose_rows(rows: list[dict]) -> dict:
    out = {"count": len(rows)}
    for metric in SUMMARY_METRICS:
        summary = metric_summary(rows, metric)
        if summary:
            out[metric] = summary
    return out


def add_summary_columns(row: dict, summary: dict) -> dict:
    out = dict(row)
    out["count"] = summary.get("count", 0)
    for metric in SUMMARY_METRICS:
        values = summary.get(metric, {})
        for stat in ("mean", "median", "min", "max"):
            out[f"{metric}_{stat}"] = values.get(stat)
    return out


def write_aggregate_csv(path: Path, summaries: list[dict]) -> None:
    keys = [
        "checkpoint_tag",
        "checkpoint",
        "checkpoint_step",
        "checkpoint_epoch",
        "pose_mode",
        "count",
        "iou_mean",
        "iou_median",
        "target_recall_mean",
        "target_recall_median",
        "pred_precision_mean",
        "pred_precision_median",
        "pred_unique_mean",
        "pred_unique_median",
        "target_unique_mean",
        "target_unique_median",
        "intersection_mean",
        "intersection_median",
        "view_aggregator",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in summaries:
            writer.writerow({key: row.get(key) for key in keys})


def write_pose_metrics_csv(path: Path, rows: list[dict]) -> None:
    keys = [
        "checkpoint_tag",
        "checkpoint",
        "checkpoint_step",
        "checkpoint_epoch",
        "pose_mode",
        "sample_index",
        "uid",
        "pred_unique",
        "target_unique",
        "intersection",
        "iou",
        "target_recall",
        "pred_precision",
        "output_dir",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in keys})


def group_rows(rows: list[dict], *, checkpoint_tag_value: str, pose_mode: str) -> dict[int, dict]:
    return {
        int(row["sample_index"]): row
        for row in rows
        if row.get("checkpoint_tag") == checkpoint_tag_value and row.get("pose_mode") == pose_mode
    }


def build_pairwise_rows(rows: list[dict], checkpoint_tags: list[str], pose_modes: list[str], reference_pose: str) -> list[dict]:
    out: list[dict] = []
    for tag in checkpoint_tags:
        ref_by_index = group_rows(rows, checkpoint_tag_value=tag, pose_mode=reference_pose)
        for pose_mode in pose_modes:
            if pose_mode == reference_pose:
                continue
            other_by_index = group_rows(rows, checkpoint_tag_value=tag, pose_mode=pose_mode)
            indices = sorted(set(ref_by_index) & set(other_by_index))
            for metric in ("iou", "target_recall", "pred_precision", "pred_unique", "intersection"):
                deltas = np.asarray(
                    [float(ref_by_index[idx][metric]) - float(other_by_index[idx][metric]) for idx in indices],
                    dtype=np.float64,
                )
                if deltas.size == 0:
                    continue
                out.append(
                    {
                        "checkpoint_tag": tag,
                        "reference_pose": reference_pose,
                        "wrong_pose": pose_mode,
                        "metric": metric,
                        "count": int(deltas.size),
                        "mean_delta": float(deltas.mean()),
                        "median_delta": float(np.median(deltas)),
                        "min_delta": float(deltas.min()),
                        "max_delta": float(deltas.max()),
                        "reference_wins": int((deltas > 0).sum()),
                        "reference_win_rate": float((deltas > 0).mean()),
                        "reference_ties": int((deltas == 0).sum()),
                    }
                )
    return out


def write_pairwise_csv(path: Path, rows: list[dict]) -> None:
    keys = [
        "checkpoint_tag",
        "reference_pose",
        "wrong_pose",
        "metric",
        "count",
        "mean_delta",
        "median_delta",
        "min_delta",
        "max_delta",
        "reference_wins",
        "reference_win_rate",
        "reference_ties",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in keys})


def build_rank_rows(rows: list[dict], checkpoint_tags: list[str], pose_modes: list[str], reference_pose: str) -> tuple[list[dict], list[dict]]:
    per_sample: list[dict] = []
    summary: list[dict] = []
    for tag in checkpoint_tags:
        by_pose = {
            pose: group_rows(rows, checkpoint_tag_value=tag, pose_mode=pose)
            for pose in pose_modes
        }
        common_indices = sorted(set.intersection(*(set(indexed) for indexed in by_pose.values()))) if by_pose else []
        for metric in RANK_METRICS:
            ranks: list[int] = []
            margins: list[float] = []
            top1 = 0
            for sample_index in common_indices:
                values = {
                    pose: float(by_pose[pose][sample_index][metric])
                    for pose in pose_modes
                }
                ref_value = values[reference_pose]
                better = [pose for pose, value in values.items() if value > ref_value]
                rank = len(better) + 1
                best_pose = max(values, key=values.get)
                best_value = values[best_pose]
                margin_to_best = ref_value - best_value
                ranks.append(rank)
                margins.append(margin_to_best)
                top1 += int(rank == 1)
                per_sample.append(
                    {
                        "checkpoint_tag": tag,
                        "sample_index": sample_index,
                        "metric": metric,
                        "reference_pose": reference_pose,
                        "reference_value": ref_value,
                        "best_pose": best_pose,
                        "best_value": best_value,
                        "reference_rank": rank,
                        "margin_to_best": margin_to_best,
                    }
                )
            if ranks:
                summary.append(
                    {
                        "checkpoint_tag": tag,
                        "metric": metric,
                        "count": len(ranks),
                        "reference_pose": reference_pose,
                        "reference_top1": top1,
                        "reference_top1_rate": float(top1 / len(ranks)),
                        "reference_rank_mean": float(np.mean(ranks)),
                        "reference_rank_median": float(np.median(ranks)),
                        "margin_to_best_mean": float(np.mean(margins)),
                        "margin_to_best_median": float(np.median(margins)),
                    }
                )
    return per_sample, summary


def write_rank_csv(path: Path, rows: list[dict], *, summary: bool = False) -> None:
    if summary:
        keys = [
            "checkpoint_tag",
            "metric",
            "count",
            "reference_pose",
            "reference_top1",
            "reference_top1_rate",
            "reference_rank_mean",
            "reference_rank_median",
            "margin_to_best_mean",
            "margin_to_best_median",
        ]
    else:
        keys = [
            "checkpoint_tag",
            "sample_index",
            "metric",
            "reference_pose",
            "reference_value",
            "best_pose",
            "best_value",
            "reference_rank",
            "margin_to_best",
        ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in keys})


def write_markdown_report(path: Path, aggregate: dict, pairwise_rows: list[dict], rank_summary: list[dict]) -> None:
    lines = [
        "# Sparse Pose Corruption Sweep",
        "",
        f"time: `{aggregate['timestamp_utc']}`",
        f"manifest: `{aggregate['manifest']}`",
        f"checkpoints: `{', '.join(aggregate['checkpoints'])}`",
        f"indices: `{aggregate['indices'][0]}...{aggregate['indices'][-1]}` count=`{len(aggregate['indices'])}`",
        f"pose_modes: `{', '.join(aggregate['pose_modes'])}`",
        f"steps: `{aggregate['steps']}`",
        (
            f"condition: `empty_policy={aggregate['empty_policy']}`, "
            f"`global_fusion={aggregate['global_fusion']}`, "
            f"`view_aggregator={aggregate['view_aggregator']}`, "
            f"`geometry_adapter={aggregate.get('geometry_adapter', 'none')}`, "
            f"`pose_consistency_head={aggregate.get('pose_consistency_head') or 'none'}`, "
            f"`pose_consistency_alpha={aggregate.get('pose_consistency_alpha', 1.0)}`"
        ),
        "",
        "## Pose Summary",
        "",
        "| pose | IoU mean | IoU median | recall mean | precision mean | pred unique mean |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate["summaries"]:
        lines.append(
            "| {pose} | {iou:.6f} | {iou_med:.6f} | {recall:.6f} | {precision:.6f} | {pred:.1f} |".format(
                pose=row["pose_mode"],
                iou=row.get("iou_mean") or 0.0,
                iou_med=row.get("iou_median") or 0.0,
                recall=row.get("target_recall_mean") or 0.0,
                precision=row.get("pred_precision_mean") or 0.0,
                pred=row.get("pred_unique_mean") or 0.0,
            )
        )
    lines.extend(
        [
            "",
            "## Correct-vs-Wrong Pairwise",
            "",
            "| wrong pose | metric | mean delta | median delta | correct wins | win rate |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in pairwise_rows:
        if row["metric"] not in {"iou", "target_recall", "pred_precision"}:
            continue
        lines.append(
            "| {wrong} | {metric} | {mean:.6f} | {median:.6f} | {wins}/{count} | {rate:.3f} |".format(
                wrong=row["wrong_pose"],
                metric=row["metric"],
                mean=row["mean_delta"],
                median=row["median_delta"],
                wins=row["reference_wins"],
                count=row["count"],
                rate=row["reference_win_rate"],
            )
        )
    lines.extend(
        [
            "",
            "## Correct Rank",
            "",
            "| metric | top1 | top1 rate | rank mean | rank median | margin-to-best mean |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rank_summary:
        lines.append(
            "| {metric} | {top1}/{count} | {rate:.3f} | {rank_mean:.3f} | {rank_median:.3f} | {margin:.6f} |".format(
                metric=row["metric"],
                top1=row["reference_top1"],
                count=row["count"],
                rate=row["reference_top1_rate"],
                rank_mean=row["reference_rank_mean"],
                rank_median=row["reference_rank_median"],
                margin=row["margin_to_best_mean"],
            )
        )
    lines.extend(
        [
            "",
            "## Reading",
            "",
            "- `reverse` keeps the same camera set but reverses temporal/view order; it is a mild-to-medium correspondence corruption.",
            "- `noise` and `large_noise` perturb camera rotation/translation and are stronger pose-corruption checks.",
            "- `identity` removes the AR trajectory structure and is the strongest convention/pose sanity check.",
            "- The checkpoint should not be selected by `correct` metrics alone. Prefer checkpoints where `correct` ranks top-1 often and has positive median delta against strong wrong poses.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate multiple sparse checkpoints with pose-mode sparse sampling.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoints", required=True, help="Comma-separated checkpoint paths.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--indices", default="0-63")
    parser.add_argument("--pose_modes", default="correct,reverse,noise,large_noise,identity")
    parser.add_argument("--reference_pose", choices=list(POSE_MODES), default="correct")
    parser.add_argument("--model_path", default="TencentARC/Pixal3D")
    parser.add_argument("--sparse_flow_model", default="TencentARC/Pixal3D/ckpts/ss_flow_img_dit_1_3B_64_bf16")
    parser.add_argument("--sparse_decoder", default="microsoft/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16")
    parser.add_argument("--image_cond_model", default="/home/zjr/Tracker/models/dinov3-vitl16-pretrain-lvd1689m")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_frames", type=int, default=8)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--rescale_t", type=float, default=1.0)
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
    parser.add_argument("--empty_policy", choices=["zero", "visible", "border", "soft"], default="soft")
    parser.add_argument("--fallback_weight", type=float, default=1.0)
    parser.add_argument("--support_confidence_power", type=float, default=1.0)
    parser.add_argument("--global_fusion", choices=["concat", "mean", "first"], default="mean")
    parser.add_argument("--geometry_feature_mode", choices=["none", "add", "replace"], default="none")
    parser.add_argument("--geometry_feature_scale", type=float, default=1.0)
    parser.add_argument("--view_aggregator", choices=["none", "gated"], default="none")
    parser.add_argument("--view_aggregator_geom_dim", type=int, default=11)
    parser.add_argument("--view_aggregator_reduced_dim", type=int, default=128)
    parser.add_argument("--view_aggregator_hidden_dim", type=int, default=256)
    parser.add_argument("--view_aggregator_dropout", type=float, default=0.0)
    parser.add_argument("--view_aggregator_residual_scale", type=float, default=1.0)
    parser.add_argument(
        "--view_aggregator_geom_mode",
        choices=["full", "no_xyz", "uv_depth_only", "support_only"],
        default="full",
    )
    parser.add_argument("--geometry_adapter", choices=["none", "mlp"], default="none")
    parser.add_argument("--geometry_adapter_dim", type=int, default=0)
    parser.add_argument("--geometry_adapter_hidden_dim", type=int, default=256)
    parser.add_argument("--geometry_adapter_dropout", type=float, default=0.0)
    parser.add_argument("--geometry_adapter_residual_scale", type=float, default=1.0)
    parser.add_argument(
        "--pose_consistency_head",
        default="",
        help="Optional pose-consistency head checkpoint. If set, its logits are added to view-gated aggregator logits.",
    )
    parser.add_argument(
        "--pose_consistency_alpha",
        type=float,
        default=1.0,
        help="Scale for adding pose-consistency logits to view-gated aggregator logits.",
    )
    parser.add_argument("--ablation_name", default="")
    parser.add_argument("--save_previews", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoints = parse_checkpoint_spec(args.checkpoints)
    pose_modes = parse_pose_modes(args.pose_modes)
    if args.reference_pose not in pose_modes:
        raise ValueError(f"--reference_pose {args.reference_pose!r} must be included in --pose_modes")

    dataset = MultiviewSparseManifestDataset(
        args.manifest,
        max_frames=args.max_frames,
        apply_mask=not args.no_apply_mask,
    )
    indices = parse_indices(args.indices, len(dataset))

    denoiser = load_sparse_flow_model(args.model_path, args.sparse_flow_model, device).eval()
    image_cond_model = build_image_cond_model(device, args.ss_grid_resolution, args.image_cond_model)
    decoder = pixal3d_models.from_pretrained(args.sparse_decoder).to(device).eval()
    condition_builder = SparseMultiviewConditionBuilder(device=device, low_vram=False)
    view_aggregator = build_view_aggregator(args, image_cond_model, device)
    if view_aggregator is not None:
        view_aggregator.eval()
    condition_builder.view_aggregator = view_aggregator
    if args.pose_consistency_head:
        feature_dim = int(getattr(image_cond_model, "embed_dim", image_cond_model.model.config.hidden_size))
        condition_builder.pose_consistency_head = load_pose_consistency_head(
            args.pose_consistency_head,
            feature_dim=feature_dim,
            device=device,
        ).eval()
        condition_builder.pose_consistency_alpha = float(args.pose_consistency_alpha)
    geometry_adapter = build_geometry_adapter(args, image_cond_model, device)
    if geometry_adapter is not None:
        geometry_adapter.eval()
    condition_builder.geometry_adapter = geometry_adapter

    summaries: list[dict] = []
    all_rows: list[dict] = []
    for checkpoint_path in checkpoints:
        if not checkpoint_path.exists():
            print(f"[skip] missing checkpoint: {checkpoint_path}", flush=True)
            continue
        tag = checkpoint_tag(checkpoint_path)
        checkpoint_info = load_checkpoint_weights(denoiser, str(checkpoint_path), view_aggregator, geometry_adapter)
        denoiser.eval()
        if view_aggregator is not None:
            view_aggregator.eval()
        if geometry_adapter is not None:
            geometry_adapter.eval()
        for pose_mode in pose_modes:
            rows: list[dict] = []
            desc = f"{tag}/{pose_mode}"
            for sample_index in tqdm(indices, desc=desc, unit="sample", dynamic_ncols=True):
                batch = dataset[sample_index]
                cond_batch = apply_pose_mode(batch, pose_mode, args.seed + sample_index * 104729)
                with torch.no_grad():
                    cond = make_multiview_condition(condition_builder, image_cond_model, cond_batch, args, device)
                coords = sample_coords(
                    denoiser,
                    decoder,
                    cond,
                    seed=args.seed + sample_index * 1009,
                    steps=args.steps,
                    rescale_t=args.rescale_t,
                    device=device,
                )
                target_coords = load_target_coords(batch["latent_path"])
                metrics = sparse_overlap_metrics(coords, target_coords)
                sample_dir = output_dir / "samples" / f"{tag}_{pose_mode}_idx{sample_index:04d}"
                row = {
                    "model": "checkpoint",
                    "checkpoint_tag": tag,
                    "checkpoint": str(checkpoint_path),
                    "checkpoint_step": checkpoint_info.get("checkpoint_step"),
                    "checkpoint_epoch": checkpoint_info.get("checkpoint_epoch"),
                    "sample_index": int(sample_index),
                    "uid": batch["uid"],
                    "output_dir": str(sample_dir),
                    "pose_mode": pose_mode,
                    "pose_permutation": cond_batch.get("pose_permutation"),
                    **metrics,
                }
                rows.append(row)
                all_rows.append(row)
                if args.save_previews:
                    sample_dir.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(sample_dir / "sparse_sample.npz", pred_coords=coords, target_coords=target_coords)
                    write_ply(sample_dir / "pred_sparse_coords.ply", coords, resolution=64, color=(240, 220, 80))
                    write_ply(sample_dir / "target_sparse_coords.ply", target_coords, resolution=64, color=(80, 200, 255))
                    make_preview(coords, target_coords, resolution=64, path=sample_dir / "sparse_preview.png")
                    (sample_dir / "summary.json").write_text(json.dumps(row, indent=2, ensure_ascii=False), encoding="utf-8")

            summary = summarize_pose_rows(rows)
            summary_row = {
                "checkpoint_tag": tag,
                "checkpoint": str(checkpoint_path),
                "checkpoint_step": checkpoint_info.get("checkpoint_step"),
                "checkpoint_epoch": checkpoint_info.get("checkpoint_epoch"),
                "pose_mode": pose_mode,
                "summary": summary,
                "rows": rows,
            }
            compact = add_summary_columns(
                {
                    "checkpoint_tag": tag,
                    "checkpoint": str(checkpoint_path),
                    "checkpoint_step": checkpoint_info.get("checkpoint_step"),
                    "checkpoint_epoch": checkpoint_info.get("checkpoint_epoch"),
                    "pose_mode": pose_mode,
                    "view_aggregator": args.view_aggregator,
                    "view_aggregator_geom_mode": args.view_aggregator_geom_mode,
                    "geometry_adapter": args.geometry_adapter,
                },
                summary,
            )
            summaries.append(compact)
            mode_dir = output_dir / "sparse_sampling" / f"{tag}_{pose_mode}"
            mode_dir.mkdir(parents=True, exist_ok=True)
            (mode_dir / "summary.json").write_text(json.dumps(summary_row, indent=2, ensure_ascii=False), encoding="utf-8")
            write_csv(mode_dir / "metrics.csv", rows)
            print(json.dumps(compact, indent=2, ensure_ascii=False), flush=True)
            torch.cuda.empty_cache()

    aggregate = {
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "ablation_name": args.ablation_name,
        "manifest": args.manifest,
        "checkpoints": [str(path) for path in checkpoints],
        "indices": indices,
        "pose_modes": pose_modes,
        "reference_pose": args.reference_pose,
        "steps": args.steps,
        "empty_policy": args.empty_policy,
        "global_fusion": args.global_fusion,
        "geometry_feature_mode": args.geometry_feature_mode,
        "geometry_feature_scale": args.geometry_feature_scale,
        "view_aggregator": args.view_aggregator,
        "view_aggregator_geom_mode": args.view_aggregator_geom_mode,
        "geometry_adapter": args.geometry_adapter,
        "pose_consistency_head": args.pose_consistency_head,
        "pose_consistency_alpha": args.pose_consistency_alpha,
        "summaries": summaries,
    }
    checkpoint_tags = [checkpoint_tag(path) for path in checkpoints if path.exists()]
    pairwise_rows = build_pairwise_rows(all_rows, checkpoint_tags, pose_modes, args.reference_pose)
    rank_rows, rank_summary = build_rank_rows(all_rows, checkpoint_tags, pose_modes, args.reference_pose)
    aggregate["pairwise_summary"] = pairwise_rows
    aggregate["rank_summary"] = rank_summary
    (output_dir / "sweep_summary.json").write_text(json.dumps(aggregate, indent=2, ensure_ascii=False), encoding="utf-8")
    write_aggregate_csv(output_dir / "sweep_summary.csv", summaries)
    write_pose_metrics_csv(output_dir / "all_metrics.csv", all_rows)
    write_pairwise_csv(output_dir / "pose_pairwise.csv", pairwise_rows)
    write_rank_csv(output_dir / "pose_rank_per_sample.csv", rank_rows)
    write_rank_csv(output_dir / "pose_rank_summary.csv", rank_summary, summary=True)
    write_markdown_report(output_dir / "sweep_report.md", aggregate, pairwise_rows, rank_summary)


if __name__ == "__main__":
    main()
