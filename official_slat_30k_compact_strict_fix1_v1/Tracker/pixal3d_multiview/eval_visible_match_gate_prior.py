from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

TRACKER_ROOT = Path(__file__).resolve().parents[1]
PIXAL3D_ROOT = TRACKER_ROOT / "Pixal3D"
sys.path.insert(0, str(TRACKER_ROOT))
sys.path.insert(0, str(PIXAL3D_ROOT))

from pixal3d_multiview.projection_alignment_head import load_projection_alignment_head  # noqa: E402
from pixal3d_multiview.train_pose_consistency_head import parse_modes  # noqa: E402
from pixal3d_multiview.train_projection_alignment_head import (  # noqa: E402
    extract_alignment_pack,
    make_negative_batch,
    match_visible_view_mask,
    weighted_mean,
)
from pixal3d_multiview.train_sparse_multiview import (  # noqa: E402
    IMAGE_COND_CONFIG,
    MultiviewSparseManifestDataset,
    build_image_cond_model,
)
from pixal3d_multiview.eval_condition_view_consistency import parse_indices  # noqa: E402


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in keys})


def parse_float_list(text: str) -> list[float]:
    return [float(item.strip()) for item in str(text).split(",") if item.strip()]


def resolve_manifest_path(path: str) -> str:
    manifest = Path(path)
    if manifest.is_dir():
        for name in ("val.json", "manifest.json", "train.json"):
            candidate = manifest / name
            if candidate.exists():
                return str(candidate)
        raise FileNotFoundError(
            f"--manifest points to a directory but none of val.json/manifest.json/train.json exists: {manifest}"
        )
    return str(manifest)


def calibrate_match_logits(
    match_logits: torch.Tensor,
    support_valid: torch.Tensor,
    visible: torch.Tensor,
    centering: str,
    temperature: float,
) -> torch.Tensor:
    logits = match_logits.float()
    centering = str(centering).lower()
    if centering != "none":
        if centering == "support":
            center_mask = support_valid
        elif centering == "visible":
            center_mask = visible
        else:
            raise ValueError(f"Unknown match logit centering: {centering}")
        weights = center_mask.float()
        mean = (logits * weights).sum(dim=0) / weights.sum(dim=0).clamp_min(1.0)
        logits = logits - mean[None]
    return logits / max(float(temperature), 1e-6)


def target_weight_from_soft(target_soft: torch.Tensor, args: argparse.Namespace) -> torch.Tensor:
    weight = torch.where(
        target_soft.float() >= float(args.match_target_soft_threshold),
        target_soft.float(),
        torch.zeros_like(target_soft.float()),
    )
    if weight.sum() <= 0:
        weight = target_soft.float().clamp_min(0.0)
    return weight


def softmax_attention(logits: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    masked = logits.float().masked_fill(~valid, -1.0e4)
    attn = torch.softmax(masked, dim=0) * valid.float()
    return attn / attn.sum(dim=0, keepdim=True).clamp_min(1e-6)


def gate_prior_score(
    out: dict[str, torch.Tensor],
    support: torch.Tensor,
    geom: torch.Tensor,
    target_soft: torch.Tensor,
    alpha: float,
    temperature: float,
    args: argparse.Namespace,
) -> dict[str, float]:
    match_logits = out["match_logits"].float()
    support = support.float().clamp_min(0.0)
    support_valid = support > 0
    visible = match_visible_view_mask(support, geom, args)
    prior_component = calibrate_match_logits(
        match_logits,
        support_valid,
        visible,
        args.match_logit_centering,
        temperature,
    )
    view_count = visible.float().sum(dim=0)
    enough_views = view_count >= int(args.match_min_views)
    target_weight = target_weight_from_soft(target_soft, args)

    base_logits = torch.log(support.clamp_min(1e-6))
    prior_logits = base_logits + float(alpha) * prior_component
    attn = softmax_attention(prior_logits, support_valid)
    visible_attn = attn * visible.float()
    visible_mass = visible_attn.sum(dim=0)

    per_voxel = (match_logits * visible_attn).sum(dim=0) / visible_mass.clamp_min(1e-6)
    per_voxel_calibrated = (prior_component * visible_attn).sum(dim=0) / visible_mass.clamp_min(1e-6)
    usable = enough_views & (visible_mass > 0)
    per_voxel = torch.where(usable, per_voxel, torch.full_like(per_voxel, float(args.match_missing_score)))
    per_voxel_calibrated = torch.where(
        usable,
        per_voxel_calibrated,
        torch.full_like(per_voxel_calibrated, float(args.match_missing_score)),
    )
    score = weighted_mean(per_voxel, target_weight)
    calibrated_score = weighted_mean(per_voxel_calibrated, target_weight)

    entropy = -(attn.clamp_min(1e-8) * attn.clamp_min(1e-8).log()).sum(dim=0)
    support_any = support_valid.any(dim=0)
    target_supported_weight = target_weight * support_any.float()
    target_usable_weight = target_weight * usable.float()
    visible_mass_mean = weighted_mean(visible_mass, target_weight)
    entropy_mean = weighted_mean(entropy, target_supported_weight) if target_supported_weight.sum() > 0 else entropy.new_tensor(float("nan"))
    usable_ratio = weighted_mean(usable.float(), target_weight)
    logit_mean = weighted_mean(per_voxel, target_usable_weight) if target_usable_weight.sum() > 0 else score.new_tensor(float("nan"))

    return {
        "score": float(score.detach().cpu().item()),
        "calibrated_score": float(calibrated_score.detach().cpu().item()),
        "target_visible_usable_ratio": float(usable_ratio.detach().cpu().item()),
        "target_visible_mass_mean": float(visible_mass_mean.detach().cpu().item()),
        "target_attention_entropy_mean": float(entropy_mean.detach().cpu().item()),
        "target_match_logit_mean": float(logit_mean.detach().cpu().item()),
    }


def summarize_rows(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, float, float, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                str(row["centering"]),
                float(row["temperature"]),
                float(row["alpha"]),
                str(row["wrong_mode"]),
            )
        ].append(row)
    summary = []
    for (centering, temperature, alpha, mode), group in sorted(grouped.items(), key=lambda item: item[0]):
        deltas = [float(row["delta"]) for row in group]
        wins = [int(row["correct_win"]) for row in group]
        calibrated_deltas = [float(row["calibrated_delta"]) for row in group]
        calibrated_wins = [int(row["calibrated_correct_win"]) for row in group]
        summary.append(
            {
                "centering": centering,
                "temperature": temperature,
                "alpha": alpha,
                "wrong_mode": mode,
                "count": len(group),
                "delta_mean": float(np.mean(deltas)) if deltas else None,
                "delta_median": float(np.median(deltas)) if deltas else None,
                "correct_wins": int(sum(wins)),
                "correct_win_rate": float(np.mean(wins)) if wins else None,
                "calibrated_delta_mean": float(np.mean(calibrated_deltas)) if calibrated_deltas else None,
                "calibrated_correct_wins": int(sum(calibrated_wins)),
                "calibrated_correct_win_rate": float(np.mean(calibrated_wins)) if calibrated_wins else None,
                "correct_score_mean": float(np.mean([row["correct_score"] for row in group])),
                "wrong_score_mean": float(np.mean([row["wrong_score"] for row in group])),
                "correct_calibrated_score_mean": float(np.mean([row["correct_calibrated_score"] for row in group])),
                "wrong_calibrated_score_mean": float(np.mean([row["wrong_calibrated_score"] for row in group])),
                "correct_visible_mass_mean": float(np.mean([row["correct_visible_mass"] for row in group])),
                "wrong_visible_mass_mean": float(np.mean([row["wrong_visible_mass"] for row in group])),
                "correct_entropy_mean": float(np.mean([row["correct_entropy"] for row in group])),
                "wrong_entropy_mean": float(np.mean([row["wrong_entropy"] for row in group])),
            }
        )
    return summary


def write_report(path: Path, result: dict) -> None:
    lines = [
        "# Visible Match Gate-Logit Prior Eval",
        "",
        f"time: `{result['timestamp_utc']}`",
        f"manifest: `{result['manifest']}`",
        f"checkpoint: `{result['checkpoint']}`",
        f"indices: `{result['indices'][0]}...{result['indices'][-1]}` count=`{len(result['indices'])}`",
        f"alphas: `{', '.join(str(x) for x in result['alphas'])}`",
        f"temperatures: `{', '.join(str(x) for x in result['temperatures'])}`",
        f"centering: `{result['match_logit_centering']}`",
        "",
        "## What This Tests",
        "",
        "This script does not sample sparse coords. It tests whether the learned `match_logits[v,n]` can be used as a prior before view-gated aggregation:",
        "",
        "`gate_logits_new = log(support_weights) + alpha * match_logits`",
        "",
        "A useful prior should make correct pose score higher than reverse/cyclic/cross-sample pose, especially at moderate alpha.",
        "",
        "## Summary",
        "",
        "| centering | temp | alpha | wrong pose | raw delta | raw win | calibrated delta | calibrated win |",
        "|---|---:|---:|---|---:|---:|---:|---:|",
    ]
    for row in result["summary"]:
        lines.append(
            "| {centering} | {temp:.3g} | {alpha:.2f} | {mode} | {delta} | {wins}/{count} | {cdelta} | {cwins}/{count} |".format(
                centering=row["centering"],
                temp=float(row["temperature"]),
                alpha=float(row["alpha"]),
                mode=row["wrong_mode"],
                delta="" if row.get("delta_mean") is None else f"{row['delta_mean']:.4f}",
                wins=row.get("correct_wins", 0),
                count=row.get("count", 0),
                cdelta="" if row.get("calibrated_delta_mean") is None else f"{row['calibrated_delta_mean']:.4f}",
                cwins=row.get("calibrated_correct_wins", 0),
            )
        )
    lines.extend(
        [
            "",
            "## How To Read",
            "",
            "- `alpha=0` is the support-only baseline.",
            "- If `alpha>0` improves reverse/cyclic deltas and win rates, `match_logits` is useful as a gate prior.",
            "- If `alpha>0` only improves noise/identity, it does not solve the main AR pose ambiguity.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate visible match logits as a view-gate logit prior.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--image_cond_model", default=IMAGE_COND_CONFIG["model_name"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--indices", default="0-127")
    parser.add_argument("--pose_modes", default="reverse,cyclic_shift1,cyclic_shift2,cross_sample,identity,noise,large_noise")
    parser.add_argument("--alphas", default="0,0.25,0.5,1.0")
    parser.add_argument("--temperatures", default="1.0")
    parser.add_argument("--match_logit_centering", choices=["none", "support", "visible"], default="none")
    parser.add_argument("--max_frames", type=int, default=8)
    parser.add_argument("--ss_grid_resolution", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target_neighbor1", type=float, default=0.5)
    parser.add_argument("--target_neighbor2", type=float, default=0.25)
    parser.add_argument("--camera_forward_sign", type=float, default=1.0)
    parser.add_argument("--mask_threshold", type=float, default=0.5)
    parser.add_argument("--vh_min_visible_views", type=int, default=1)
    parser.add_argument("--vh_min_support_views", type=int, default=2)
    parser.add_argument("--vh_min_support_ratio", type=float, default=0.6)
    parser.add_argument("--vh_volume_resolution", type=int, default=48)
    parser.add_argument("--vh_volume_initial_extent_ratio", type=float, default=0.6)
    parser.add_argument("--vh_volume_padding", type=float, default=1.25)
    parser.add_argument("--vh_volume_min_extent", type=float, default=0.05)
    parser.add_argument("--vh_volume_refine_steps", type=int, default=2)
    parser.add_argument("--vh_visibility_resolution", type=int, default=48)
    parser.add_argument("--vh_visibility_dilation", type=int, default=3)
    parser.add_argument("--visibility_depth_tolerance", type=float, default=0.0)
    parser.add_argument("--visibility_depth_tolerance_ratio", type=float, default=0.15)
    parser.add_argument("--visibility_weight_min", type=float, default=0.05)
    parser.add_argument("--match_target_soft_threshold", type=float, default=0.999)
    parser.add_argument("--match_min_views", type=int, default=3)
    parser.add_argument("--match_visible_surface_only", type=int, default=1)
    parser.add_argument("--match_visibility_threshold", type=float, default=0.3)
    parser.add_argument("--match_mask_value_threshold", type=float, default=0.5)
    parser.add_argument("--match_mask_hit_threshold", type=float, default=0.5)
    parser.add_argument("--match_min_support_weight", type=float, default=0.05)
    parser.add_argument("--match_require_valid_depth", type=int, default=1)
    parser.add_argument("--match_missing_score", type=float, default=-1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.manifest = resolve_manifest_path(args.manifest)
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = MultiviewSparseManifestDataset(args.manifest, max_frames=args.max_frames, apply_mask=True)
    indices = parse_indices(args.indices, len(dataset))
    pose_modes = parse_modes(args.pose_modes, allow_correct=False)
    alphas = parse_float_list(args.alphas)
    temperatures = parse_float_list(args.temperatures)

    image_cond_model = build_image_cond_model(device, args.ss_grid_resolution, args.image_cond_model)
    feature_dim = int(getattr(image_cond_model, "embed_dim", image_cond_model.model.config.hidden_size))
    head = load_projection_alignment_head(args.checkpoint, feature_dim=feature_dim, device=device).eval()

    rows = []
    with torch.no_grad():
        for idx in tqdm(indices, desc="Gate prior eval", unit="sample", dynamic_ncols=True):
            batch = dataset[idx]
            correct_pack = extract_alignment_pack(batch, image_cond_model, args, device)
            correct_out = head(correct_pack["sampled"], correct_pack["support"], correct_pack["geom"])
            for mode_i, mode in enumerate(pose_modes):
                seed = int(args.seed + idx * 104729 + mode_i * 9176 + 11)
                wrong_batch = make_negative_batch(dataset, batch, mode, seed)
                wrong_pack = extract_alignment_pack(
                    wrong_batch,
                    image_cond_model,
                    args,
                    device,
                    patch_features=correct_pack["patch_features"],
                )
                wrong_out = head(wrong_pack["sampled"], wrong_pack["support"], wrong_pack["geom"])
                for temperature in temperatures:
                    for alpha in alphas:
                        correct_stats = gate_prior_score(
                            correct_out,
                            correct_pack["support"],
                            correct_pack["geom"],
                            correct_pack["target_soft"],
                            alpha,
                            temperature,
                            args,
                        )
                        wrong_stats = gate_prior_score(
                            wrong_out,
                            wrong_pack["support"],
                            wrong_pack["geom"],
                            correct_pack["target_soft"],
                            alpha,
                            temperature,
                            args,
                        )
                        delta = correct_stats["score"] - wrong_stats["score"]
                        calibrated_delta = correct_stats["calibrated_score"] - wrong_stats["calibrated_score"]
                        rows.append(
                            {
                                "index": idx,
                                "uid": batch["uid"],
                                "wrong_mode": mode,
                                "centering": args.match_logit_centering,
                                "temperature": float(temperature),
                                "alpha": float(alpha),
                                "correct_score": correct_stats["score"],
                                "wrong_score": wrong_stats["score"],
                                "delta": float(delta),
                                "correct_win": int(delta > 0),
                                "correct_calibrated_score": correct_stats["calibrated_score"],
                                "wrong_calibrated_score": wrong_stats["calibrated_score"],
                                "calibrated_delta": float(calibrated_delta),
                                "calibrated_correct_win": int(calibrated_delta > 0),
                                "correct_visible_mass": correct_stats["target_visible_mass_mean"],
                                "wrong_visible_mass": wrong_stats["target_visible_mass_mean"],
                                "correct_entropy": correct_stats["target_attention_entropy_mean"],
                                "wrong_entropy": wrong_stats["target_attention_entropy_mean"],
                                "correct_usable_ratio": correct_stats["target_visible_usable_ratio"],
                                "wrong_usable_ratio": wrong_stats["target_visible_usable_ratio"],
                                "correct_match_logit": correct_stats["target_match_logit_mean"],
                                "wrong_match_logit": wrong_stats["target_match_logit_mean"],
                            }
                        )

    summary = summarize_rows(rows)
    result = {
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "manifest": args.manifest,
        "checkpoint": args.checkpoint,
        "indices": indices,
        "pose_modes": pose_modes,
        "alphas": alphas,
        "temperatures": temperatures,
        "match_logit_centering": args.match_logit_centering,
        "summary": summary,
        "args": vars(args),
    }
    (output_dir / "summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(output_dir / "gate_prior_rows.csv", rows)
    write_csv(output_dir / "gate_prior_summary.csv", summary)
    write_report(output_dir / "report.md", result)
    print(json.dumps({"output_dir": str(output_dir), "summary": summary}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
