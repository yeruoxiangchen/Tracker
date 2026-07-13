from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
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


def parse_pose_modes(spec: str) -> list[str]:
    modes: list[str] = []
    seen: set[str] = set()
    for part in str(spec).split(","):
        mode = part.strip().lower()
        if not mode:
            continue
        if mode not in POSE_CONSISTENCY_MODES:
            raise ValueError(f"Unknown pose mode {mode!r}; valid={POSE_CONSISTENCY_MODES}")
        if mode not in seen:
            modes.append(mode)
            seen.add(mode)
    if not modes:
        raise ValueError("--candidate_pose_modes must contain at least one mode")
    return modes


def deterministic_cross_sample(dataset, sample_index: int, seed: int) -> dict:
    if len(dataset) <= 1:
        return dataset[sample_index]
    other = (int(sample_index) * 1103515245 + int(seed) * 12345 + 17) % len(dataset)
    if other == sample_index:
        other = (other + 1) % len(dataset)
    return dataset[int(other)]


def make_candidate_batch(dataset, batch: dict, sample_index: int, mode: str, seed: int) -> dict:
    if mode == "correct":
        out = dict(batch)
        out["pose_mode"] = "correct"
        out["pose_permutation"] = None
        out["cross_sample_uid"] = None
        return out
    if mode == "cross_sample":
        other = deterministic_cross_sample(dataset, sample_index, seed)
        return make_cross_sample_batch(batch, other)
    return apply_pose_mode(batch, mode, seed)


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


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def summarize_scores(rows: list[dict], pose_modes: list[str]) -> list[dict]:
    out: list[dict] = []
    for mode in pose_modes:
        selected = [row for row in rows if row["pose_mode"] == mode]
        if not selected:
            continue
        scores = np.asarray([float(row["score"]) for row in selected], dtype=np.float64)
        keeps = np.asarray([float(row["keep_ratio"]) for row in selected], dtype=np.float64)
        out.append(
            {
                "pose_mode": mode,
                "count": len(selected),
                "score_mean": float(scores.mean()),
                "score_median": float(np.median(scores)),
                "score_min": float(scores.min()),
                "score_max": float(scores.max()),
                "keep_mean": float(keeps.mean()),
                "keep_median": float(np.median(keeps)),
            }
        )
    return out


def rank_candidates(
    rows: list[dict],
    *,
    pose_modes: list[str],
    reference_pose: str,
    input_pose: str,
    metric: str,
    score_threshold: float | None,
    margin_threshold: float,
) -> tuple[list[dict], list[dict], dict]:
    by_sample: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        by_sample[int(row["sample_index"])].append(row)

    ranked_rows: list[dict] = []
    decision_rows: list[dict] = []
    json_rows: list[dict] = []
    action_counts: Counter[str] = Counter()
    top_pose_counts: Counter[str] = Counter()
    reference_ranks: list[int] = []
    input_ranks: list[int] = []
    top_margins: list[float] = []
    input_margins: list[float] = []
    reference_top1 = 0
    input_top1 = 0
    sanity_pass_count = 0

    for sample_index in sorted(by_sample):
        candidates = by_sample[sample_index]
        by_pose = {row["pose_mode"]: row for row in candidates}
        missing = [pose for pose in pose_modes if pose not in by_pose]
        if missing:
            raise RuntimeError(f"sample_index={sample_index} missing candidate modes: {missing}")
        if reference_pose not in by_pose:
            raise RuntimeError(f"sample_index={sample_index} missing reference_pose={reference_pose}")
        if input_pose not in by_pose:
            raise RuntimeError(f"sample_index={sample_index} missing input_pose={input_pose}")

        ordered = sorted(candidates, key=lambda row: float(row[metric]), reverse=True)
        top = ordered[0]
        second = ordered[1] if len(ordered) > 1 else None
        top_score = float(top[metric])
        second_score = float(second[metric]) if second is not None else float("-inf")
        top_margin = top_score - second_score if second is not None else float("inf")

        for rank, row in enumerate(ordered, start=1):
            ranked = dict(row)
            ranked["rank"] = rank
            ranked["selection_metric"] = metric
            ranked["selected"] = int(rank == 1)
            ranked["score_gap_to_top"] = top_score - float(row[metric])
            ranked_rows.append(ranked)

        reference_row = by_pose[reference_pose]
        input_row = by_pose[input_pose]
        reference_value = float(reference_row[metric])
        input_value = float(input_row[metric])
        reference_rank = 1 + sum(float(row[metric]) > reference_value for row in candidates if row is not reference_row)
        input_rank = 1 + sum(float(row[metric]) > input_value for row in candidates if row is not input_row)
        best_other_input = max((row for row in candidates if row["pose_mode"] != input_pose), key=lambda row: float(row[metric]), default=None)
        input_margin_to_best_other = (
            input_value - float(best_other_input[metric])
            if best_other_input is not None
            else float("inf")
        )
        pass_input_top1 = input_rank == 1
        input_pass_score = True if score_threshold is None else input_value >= float(score_threshold)
        input_pass_margin = input_margin_to_best_other >= float(margin_threshold)
        top_pass_score = True if score_threshold is None else top_score >= float(score_threshold)
        top_pass_margin = top_margin >= float(margin_threshold)
        sanity_pass = bool(pass_input_top1 and input_pass_score and input_pass_margin)
        if pass_input_top1:
            if not input_pass_score:
                action = "reject_low_score"
            elif not input_pass_margin:
                action = "reject_ambiguous_margin"
            else:
                action = "accept_input_pose"
        elif top_pass_score and top_pass_margin:
            action = "rerank_to_top"
        elif not top_pass_score:
            action = "reject_low_score"
        else:
            action = "reject_ambiguous_margin"

        top_pose = str(top["pose_mode"])
        top_pose_counts[top_pose] += 1
        action_counts[action] += 1
        reference_ranks.append(reference_rank)
        input_ranks.append(input_rank)
        top_margins.append(top_margin)
        input_margins.append(input_margin_to_best_other)
        reference_top1 += int(reference_rank == 1)
        input_top1 += int(input_rank == 1)
        sanity_pass_count += int(sanity_pass)

        decision = {
            "sample_index": sample_index,
            "uid": top["uid"],
            "selection_metric": metric,
            "candidate_count": len(candidates),
            "top_pose": top_pose,
            "top_score": top_score,
            "second_pose": second["pose_mode"] if second is not None else None,
            "second_score": second_score if second is not None else None,
            "top_margin": top_margin,
            "reference_pose": reference_pose,
            "reference_score": reference_value,
            "reference_rank": reference_rank,
            "reference_is_top1": int(reference_rank == 1),
            "input_pose": input_pose,
            "input_score": input_value,
            "input_rank": input_rank,
            "input_is_top1": int(input_rank == 1),
            "best_non_input_pose": best_other_input["pose_mode"] if best_other_input is not None else None,
            "best_non_input_score": float(best_other_input[metric]) if best_other_input is not None else None,
            "input_margin_to_best_other": input_margin_to_best_other,
            "score_threshold": score_threshold,
            "margin_threshold": margin_threshold,
            "input_pass_score_threshold": int(input_pass_score),
            "input_pass_margin_threshold": int(input_pass_margin),
            "top_pass_score_threshold": int(top_pass_score),
            "top_pass_margin_threshold": int(top_pass_margin),
            "sanity_pass": int(sanity_pass),
            "action": action,
        }
        decision_rows.append(decision)
        json_rows.append(
            {
                **decision,
                "candidates": [
                    {
                        "rank": rank,
                        "pose_mode": row["pose_mode"],
                        "score": float(row["score"]),
                        "keep_ratio": float(row["keep_ratio"]),
                    }
                    for rank, row in enumerate(ordered, start=1)
                ],
            }
        )

    count = len(decision_rows)
    summary = {
        "count": count,
        "selection_metric": metric,
        "reference_pose": reference_pose,
        "input_pose": input_pose,
        "score_threshold": score_threshold,
        "margin_threshold": float(margin_threshold),
        "reference_top1": reference_top1,
        "reference_top1_rate": float(reference_top1 / count) if count else 0.0,
        "input_top1": input_top1,
        "input_top1_rate": float(input_top1 / count) if count else 0.0,
        "sanity_pass": sanity_pass_count,
        "sanity_pass_rate": float(sanity_pass_count / count) if count else 0.0,
        "reference_rank_mean": float(np.mean(reference_ranks)) if reference_ranks else 0.0,
        "reference_rank_median": float(np.median(reference_ranks)) if reference_ranks else 0.0,
        "input_rank_mean": float(np.mean(input_ranks)) if input_ranks else 0.0,
        "input_rank_median": float(np.median(input_ranks)) if input_ranks else 0.0,
        "top_margin_mean": float(np.mean(top_margins)) if top_margins else 0.0,
        "top_margin_median": float(np.median(top_margins)) if top_margins else 0.0,
        "input_margin_mean": float(np.mean(input_margins)) if input_margins else 0.0,
        "input_margin_median": float(np.median(input_margins)) if input_margins else 0.0,
        "top_pose_counts": dict(top_pose_counts),
        "action_counts": dict(action_counts),
        "json_rows": json_rows,
    }
    return ranked_rows, decision_rows, summary


def _format_counts(counts: dict[str, int], total: int) -> list[str]:
    lines = []
    for key, value in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        rate = float(value / total) if total else 0.0
        lines.append(f"| {key} | {value} | {rate:.3f} |")
    return lines


def write_markdown(path: Path, result: dict, score_summary: list[dict], rerank_summary: dict) -> None:
    count = int(rerank_summary.get("count", 0))
    lines = [
        "# Pose Head Reranker / Sanity Check",
        "",
        f"time: `{result['timestamp_utc']}`",
        f"manifest: `{result['manifest']}`",
        f"head: `{result['head_checkpoint']}`",
        f"indices: `{result['indices']}`",
        f"candidate_pose_modes: `{', '.join(result['candidate_pose_modes'])}`",
        f"selection_metric: `{rerank_summary['selection_metric']}`",
        f"reference_pose: `{rerank_summary['reference_pose']}`",
        f"input_pose: `{rerank_summary['input_pose']}`",
        f"score_threshold: `{rerank_summary['score_threshold']}`",
        f"margin_threshold: `{rerank_summary['margin_threshold']}`",
        "",
        "## Candidate Score Summary",
        "",
        "| pose | count | score mean | score median | keep mean |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in score_summary:
        lines.append(
            f"| {row['pose_mode']} | {row['count']} | {row['score_mean']:.4f} | "
            f"{row['score_median']:.4f} | {row['keep_mean']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Rerank Summary",
            "",
            "| metric | value |",
            "|---|---:|",
            f"| count | {count} |",
            f"| reference top1 | {rerank_summary['reference_top1']}/{count} |",
            f"| reference top1 rate | {rerank_summary['reference_top1_rate']:.3f} |",
            f"| reference rank mean | {rerank_summary['reference_rank_mean']:.3f} |",
            f"| reference rank median | {rerank_summary['reference_rank_median']:.3f} |",
            f"| input top1 | {rerank_summary['input_top1']}/{count} |",
            f"| input top1 rate | {rerank_summary['input_top1_rate']:.3f} |",
            f"| sanity pass | {rerank_summary['sanity_pass']}/{count} |",
            f"| sanity pass rate | {rerank_summary['sanity_pass_rate']:.3f} |",
            f"| top margin mean | {rerank_summary['top_margin_mean']:.4f} |",
            f"| top margin median | {rerank_summary['top_margin_median']:.4f} |",
            f"| input margin mean | {rerank_summary['input_margin_mean']:.4f} |",
            f"| input margin median | {rerank_summary['input_margin_median']:.4f} |",
            "",
            "## Top Pose Distribution",
            "",
            "| top pose | count | rate |",
            "|---|---:|---:|",
        ]
    )
    lines.extend(_format_counts(rerank_summary.get("top_pose_counts", {}), count))
    lines.extend(
        [
            "",
            "## Sanity Action Distribution",
            "",
            "| action | count | rate |",
            "|---|---:|---:|",
        ]
    )
    lines.extend(_format_counts(rerank_summary.get("action_counts", {}), count))
    lines.extend(
        [
            "",
            "## Reading",
            "",
            "- `top_pose` is the candidate with the largest pose-head score.",
            "- `sanity_pass` keeps the current/input pose only when it is top-1, passes `score_threshold`, and beats every other candidate by `margin_threshold`.",
            "- `rerank_to_top` means the head prefers another candidate with enough margin; this is the pose-hypothesis reranker path.",
            "- `reject_low_score` or `reject_ambiguous_margin` should be treated as a deferred/unsafe pose before mesh refinement.",
            "- `identity` is useful as a stress negative, but the current head has historically over-scored identity; do not include it in production reranking unless that is the exact sanity check being tested.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Use a trained pose consistency head as a pose reranker and sanity checker.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--head_checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--image_cond_model", default=IMAGE_COND_CONFIG["model_name"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_frames", type=int, default=8)
    parser.add_argument("--indices", default="0-63")
    parser.add_argument(
        "--candidate_pose_modes",
        default="correct,cyclic_shift1,cyclic_shift2,reverse,noise,large_noise",
        help="Candidate hypotheses to score and rerank. Add identity/cross_sample only for stress diagnostics.",
    )
    parser.add_argument("--reference_pose", default="correct", help="Offline label used to measure reranker top-1/rank.")
    parser.add_argument("--input_pose", default="correct", help="Current pose to sanity-check; usually the incoming estimated pose.")
    parser.add_argument("--selection_metric", choices=["score", "keep_ratio"], default="score")
    parser.add_argument("--score_threshold", type=float, default=None)
    parser.add_argument("--margin_threshold", type=float, default=0.05)
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

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    candidate_pose_modes = parse_pose_modes(args.candidate_pose_modes)
    if args.reference_pose not in candidate_pose_modes:
        raise ValueError(f"--reference_pose {args.reference_pose!r} must be included in --candidate_pose_modes")
    if args.input_pose not in candidate_pose_modes:
        raise ValueError(f"--input_pose {args.input_pose!r} must be included in --candidate_pose_modes")

    dataset = MultiviewSparseManifestDataset(args.manifest, max_frames=args.max_frames, apply_mask=not args.no_apply_mask)
    indices = parse_indices(args.indices, len(dataset))
    image_cond_model = build_image_cond_model(device, args.ss_grid_resolution, args.image_cond_model)
    feature_dim = int(getattr(image_cond_model, "embed_dim", image_cond_model.model.config.hidden_size))
    head = load_pose_consistency_head(args.head_checkpoint, feature_dim=feature_dim, device=device).eval()
    condition_builder = setup_condition_builder(args, image_cond_model, head, device)

    rows: list[dict] = []
    with torch.no_grad():
        for sample_index in tqdm(indices, desc="Pose reranker", unit="sample", dynamic_ncols=True):
            batch = dataset[sample_index]
            for mode in candidate_pose_modes:
                seed = int(args.seed + sample_index * 104729)
                cond_batch = make_candidate_batch(dataset, batch, sample_index, mode, seed)
                score, keep, stats = score_batch(condition_builder, image_cond_model, cond_batch, args, device)
                rows.append(
                    {
                        "sample_index": int(sample_index),
                        "uid": batch["uid"],
                        "pose_mode": mode,
                        "score": float(score.detach().cpu().item()),
                        "keep_ratio": float(keep.detach().cpu().item()),
                        "score_mode": stats.get("score_mode"),
                        "pair_weight_mode": stats.get("pair_weight_mode"),
                        "pair_valid_ratio": stats.get("pair_valid_ratio"),
                        "pair_count_mean": stats.get("pair_count_mean"),
                        "pair_weight_mean": stats.get("pair_weight_mean"),
                        "view_weight_mean": stats.get("view_weight_mean"),
                        "view_weight_nonzero_ratio": stats.get("view_weight_nonzero_ratio"),
                        "pair_supported_voxel_ratio": stats.get("pair_supported_voxel_ratio"),
                        "view_prior_abs_mean": stats.get("view_prior_abs_mean"),
                        "pose_permutation": cond_batch.get("pose_permutation"),
                        "cross_sample_uid": cond_batch.get("cross_sample_uid"),
                    }
                )

    score_summary = summarize_scores(rows, candidate_pose_modes)
    ranked_rows, decision_rows, rerank_summary = rank_candidates(
        rows,
        pose_modes=candidate_pose_modes,
        reference_pose=args.reference_pose,
        input_pose=args.input_pose,
        metric=args.selection_metric,
        score_threshold=args.score_threshold,
        margin_threshold=args.margin_threshold,
    )
    json_rows = rerank_summary.pop("json_rows")
    result = {
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "manifest": args.manifest,
        "head_checkpoint": args.head_checkpoint,
        "indices": indices,
        "candidate_pose_modes": candidate_pose_modes,
        "args": vars(args),
        "score_summary": score_summary,
        "rerank_summary": rerank_summary,
    }

    (output_dir / "rerank_summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    write_jsonl(output_dir / "rerank_selection.jsonl", json_rows)
    write_csv(output_dir / "candidate_scores.csv", rows)
    write_csv(output_dir / "candidate_scores_ranked.csv", ranked_rows)
    write_csv(output_dir / "rerank_decisions.csv", decision_rows)
    write_csv(output_dir / "score_summary.csv", score_summary)
    write_markdown(output_dir / "rerank_report.md", result, score_summary, rerank_summary)
    print(json.dumps({"output_dir": str(output_dir), "rerank_summary": rerank_summary}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
