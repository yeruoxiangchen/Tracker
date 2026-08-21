#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import mean, median
import sys
from typing import Any

import torch
import torch.nn.functional as F

TRACKER_ROOT = Path(__file__).resolve().parents[1]
RECONVIAGEN_ROOT = TRACKER_ROOT / "ReconViaGen"
VGGT_ROOT = RECONVIAGEN_ROOT / "wheels" / "vggt"
for path in (TRACKER_ROOT, RECONVIAGEN_ROOT, VGGT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ar_ss_flow.local_pose_lifting_flow import PoseLiftingCacheDataset, volume_from_sample
from ar_ss_flow.pose_lifting import LIFTING_VOLUME_VERSION
from ar_ss_flow.train_local_pose_lifting_ss_flow import (
    CORRUPTION_MODES,
    HARD_CORRUPTION_MODES,
    build_model,
    validate_checkpoint,
)
from reconvggt_ar_adapter_a.pointpose_ss_condition import load_partial_state


def parse_csv_ints(text: str) -> list[int]:
    values = [int(item.strip()) for item in str(text).split(",") if item.strip()]
    if not values:
        raise ValueError("integer list must be non-empty")
    return values


def parse_csv_floats(text: str) -> list[float]:
    values = [float(item.strip()) for item in str(text).split(",") if item.strip()]
    if not values or any(value <= 0.0 or value >= 1.0 for value in values):
        raise ValueError("t values must be non-empty and inside (0,1)")
    return values


def masked_mse(left: torch.Tensor, right: torch.Tensor, mask: torch.Tensor) -> float:
    expanded = mask.expand_as(left)
    if not bool(expanded.any().item()):
        return 0.0
    return float((left.float() - right.float()).square()[expanded].mean().item())


def masked_max_abs(left: torch.Tensor, right: torch.Tensor, mask: torch.Tensor) -> float:
    expanded = mask.expand_as(left)
    if not bool(expanded.any().item()):
        return 0.0
    return float((left.float() - right.float()).abs()[expanded].max().item())


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": mean(values),
        "median": median(values),
        "min": min(values),
        "max": max(values),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fixed-noise teacher-forced evaluation for local pose-lifting SS adapter."
    )
    parser.add_argument("--cache_manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--pretrained", default="Stable-X/trellis-vggt-v0-2")
    parser.add_argument("--indices", default="all")
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--t_values", default="0.1,0.3,0.5,0.7,0.9")
    parser.add_argument("--adapter_hidden_dim", type=int, default=96)
    parser.add_argument("--physical_scale", type=float, default=1.0)
    parser.add_argument("--amp_dtype", choices=("fp16", "bf16", "none"), default="bf16")
    parser.add_argument("--min_object_win_rate", type=float, default=0.65)
    parser.add_argument("--min_positive_t", type=int, default=4)
    parser.add_argument("--max_corrupt_abs_gain", type=float, default=0.005)
    parser.add_argument("--min_correct_stock_object_win_rate", type=float, default=0.65)
    parser.add_argument("--min_hard_positive_t", type=int, default=4)
    parser.add_argument("--min_perturb_object_win_rate", type=float, default=0.55)
    parser.add_argument("--min_perturb_positive_t", type=int, default=4)
    parser.add_argument("--max_perturb_correct_gap", type=float, default=0.005)
    parser.add_argument("--max_neutral_velocity_abs", type=float, default=0.0)
    parser.add_argument("--fail_on_error", action="store_true")
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    dataset = PoseLiftingCacheDataset(args.cache_manifest, indices=args.indices)
    cache_manifest_payload = json.loads(
        Path(args.cache_manifest).read_text(encoding="utf-8")
    )
    count = len(dataset) if args.max_samples <= 0 else min(len(dataset), args.max_samples)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    saved_amp = checkpoint.get("args", {}).get("amp_dtype")
    if saved_amp in {"fp16", "bf16", "none"}:
        args.amp_dtype = str(saved_amp)
    validate_checkpoint(checkpoint, args)
    flow_sampler, model, model_summary = build_model(
        args, device, dataset.visual_feature_dim
    )
    load_partial_state(
        model,
        checkpoint["model_trainable_state"],
        require_all_trainable=True,
    )
    model.eval()
    seeds = parse_csv_ints(args.seeds)
    t_values = parse_csv_floats(args.t_values)
    use_amp = args.amp_dtype != "none"
    amp_dtype = torch.float16 if args.amp_dtype == "fp16" else torch.bfloat16
    records: list[dict[str, Any]] = []
    null_max_abs = 0.0
    for sample_index in range(count):
        sample = dataset[sample_index]
        volumes: dict[str, tuple[torch.Tensor, torch.Tensor, dict[str, Any]]] = {
            mode: volume_from_sample(sample, device=device, mode=mode)
            for mode in ("correct", *CORRUPTION_MODES)
        }
        target = sample["target"].unsqueeze(0).to(device=device, dtype=torch.float32)
        condition = sample["stock_condition"].to(device=device)
        for seed in seeds:
            for t_value in t_values:
                combined_seed = (
                    int(seed) * 1000003 + sample_index * 1009 + int(round(t_value * 1000))
                )
                torch.manual_seed(combined_seed)
                torch.cuda.manual_seed_all(combined_seed)
                endpoint = torch.randn_like(target)
                x_t, gt_velocity = flow_sampler._get_model_gt(target, t_value, endpoint)
                t_tensor = torch.full(
                    (1,), 1000.0 * t_value, device=device, dtype=torch.float32
                )
                with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
                    stock = model.stock_prediction(x_t, t_tensor, condition)
                    predictions: dict[str, torch.Tensor] = {}
                    stats: dict[str, dict[str, torch.Tensor]] = {}
                    for mode, (volume, metadata, _) in volumes.items():
                        predictions[mode], stats[mode] = model.adapt_from_stock(
                            x_t,
                            t_tensor,
                            stock,
                            volume,
                            metadata,
                            scale=float(args.physical_scale),
                        )
                    disabled, _ = model.adapt_from_stock(
                        x_t,
                        t_tensor,
                        stock,
                        volumes["correct"][0],
                        volumes["correct"][1],
                        physical_present=False,
                    )
                    null, _ = model.adapt_from_stock(
                        x_t,
                        t_tensor,
                        stock,
                        torch.zeros_like(volumes["correct"][0]),
                        torch.zeros_like(volumes["correct"][1]),
                    )
                null_max_abs = max(
                    null_max_abs,
                    float((disabled - stock).abs().max().item()),
                    float((null - stock).abs().max().item()),
                )
                stock_loss = float(F.mse_loss(stock.float(), gt_velocity.float()).item())
                mode_losses = {
                    mode: float(
                        F.mse_loss(prediction.float(), gt_velocity.float()).item()
                    )
                    for mode, prediction in predictions.items()
                }
                correct_gain = (stock_loss - mode_losses["correct"]) / max(stock_loss, 1.0e-8)
                correct_support = volumes["correct"][1][:, 0:1] > 0
                neutral = ~correct_support
                row: dict[str, Any] = {
                    "uid": sample["uid"],
                    "object_uid": sample["object_uid"],
                    "seed": seed,
                    "t": t_value,
                    "stock_flow_mse": stock_loss,
                    "correct_flow_mse": mode_losses["correct"],
                    "correct_relative_gain_vs_stock": correct_gain,
                    "correct_support_flow_mse": masked_mse(
                        predictions["correct"], gt_velocity, correct_support
                    ),
                    "stock_support_flow_mse": masked_mse(
                        stock, gt_velocity, correct_support
                    ),
                    "neutral_velocity_mse_vs_stock": masked_mse(
                        predictions["correct"], stock, neutral
                    ),
                    "neutral_velocity_max_abs_vs_stock": masked_max_abs(
                        predictions["correct"], stock, neutral
                    ),
                    "correct_delta_rms": float(stats["correct"]["delta_rms"].item()),
                }
                for mode in CORRUPTION_MODES:
                    corrupt_gain = (stock_loss - mode_losses[mode]) / max(stock_loss, 1.0e-8)
                    row[f"{mode}_flow_mse"] = mode_losses[mode]
                    row[f"{mode}_relative_gain_vs_stock"] = corrupt_gain
                    row[f"correct_relative_gain_vs_{mode}"] = (
                        mode_losses[mode] - mode_losses["correct"]
                    ) / max(stock_loss, 1.0e-8)
                records.append(row)
        print(
            f"[pose_lifting_eval] {sample_index + 1}/{count} uid={sample['uid']}",
            flush=True,
        )

    correct_gains = [row["correct_relative_gain_vs_stock"] for row in records]
    comparison_summary: dict[str, Any] = {}
    for mode in CORRUPTION_MODES:
        differences = [row[f"correct_relative_gain_vs_{mode}"] for row in records]
        corrupt_gains = [row[f"{mode}_relative_gain_vs_stock"] for row in records]
        comparison_summary[mode] = {
            "correct_gain_difference": summarize(differences),
            "correct_win_rate": mean(float(value > 0.0) for value in differences),
            "corrupt_gain_vs_stock": summarize(corrupt_gains),
            "mean_abs_corrupt_gain_vs_stock": mean(abs(value) for value in corrupt_gains),
        }
    object_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        object_rows[str(row["object_uid"])].append(row)
    object_correct_gains = [
        mean(item["correct_relative_gain_vs_stock"] for item in rows)
        for rows in object_rows.values()
    ]
    correct_stock_object_win_rate = mean(
        float(value > 0.0) for value in object_correct_gains
    )
    object_mode_gains = {
        mode: [
            mean(item[f"{mode}_relative_gain_vs_stock"] for item in rows)
            for rows in object_rows.values()
        ]
        for mode in CORRUPTION_MODES
    }
    object_win_rates = {
        mode: mean(
            float(
                mean(item[f"correct_relative_gain_vs_{mode}"] for item in rows) > 0.0
            )
            for rows in object_rows.values()
        )
        for mode in CORRUPTION_MODES
    }
    t_summary: dict[str, Any] = {}
    comparison_t_summary: dict[str, Any] = {}
    positive_t = 0
    for t_value in t_values:
        selected = [
            row["correct_relative_gain_vs_stock"]
            for row in records
            if abs(float(row["t"]) - t_value) < 1.0e-9
        ]
        t_summary[str(t_value)] = summarize(selected)
        positive_t += int(t_summary[str(t_value)]["mean"] > 0.0)
        comparison_t_summary[str(t_value)] = {
            mode: summarize(
                [
                    row[f"correct_relative_gain_vs_{mode}"]
                    for row in records
                    if abs(float(row["t"]) - t_value) < 1.0e-9
                ]
            )
            for mode in CORRUPTION_MODES
        }
    hard_positive_t = {
        mode: sum(
            int(comparison_t_summary[str(t_value)][mode]["mean"] > 0.0)
            for t_value in t_values
        )
        for mode in HARD_CORRUPTION_MODES
    }
    perturb_positive_t = sum(
        int(
            summarize(
                [
                    row["pose_perturb_relative_gain_vs_stock"]
                    for row in records
                    if abs(float(row["t"]) - t_value) < 1.0e-9
                ]
            )["mean"]
            > 0.0
        )
        for t_value in t_values
    )
    perturb_correct_gap = comparison_summary["pose_perturb"][
        "correct_gain_difference"
    ]["mean"]
    neutral_max_abs = max(
        row["neutral_velocity_max_abs_vs_stock"] for row in records
    )
    checks = {
        "physical_off_and_null_bit_exact_stock": null_max_abs == 0.0,
        "correct_mean_gain_positive": mean(correct_gains) > 0.0,
        "correct_median_gain_positive": median(correct_gains) > 0.0,
        "object_balanced_correct_gain_positive": mean(object_correct_gains) > 0.0,
        "correct_stock_object_win_rate": correct_stock_object_win_rate
        >= float(args.min_correct_stock_object_win_rate),
        "positive_t_count": positive_t >= int(args.min_positive_t),
        "correct_beats_hard_corruptions": all(
            comparison_summary[mode]["correct_gain_difference"]["mean"] > 0.0
            and object_win_rates[mode] >= float(args.min_object_win_rate)
            for mode in HARD_CORRUPTION_MODES
        ),
        "hard_corruptions_near_stock_without_sign_cancellation": all(
            comparison_summary[mode]["mean_abs_corrupt_gain_vs_stock"]
            <= float(args.max_corrupt_abs_gain)
            for mode in HARD_CORRUPTION_MODES
        ),
        "hard_corruption_direction_per_t": all(
            hard_positive_t[mode] >= int(args.min_hard_positive_t)
            for mode in HARD_CORRUPTION_MODES
        ),
        "pose_perturb_improves_over_stock": (
            comparison_summary["pose_perturb"]["corrupt_gain_vs_stock"]["mean"] > 0.0
            and mean(float(value > 0.0) for value in object_mode_gains["pose_perturb"])
            >= float(args.min_perturb_object_win_rate)
            and perturb_positive_t >= int(args.min_perturb_positive_t)
        ),
        "pose_perturb_close_to_correct": perturb_correct_gap
        <= float(args.max_perturb_correct_gap),
        "neutral_velocity_preserved": neutral_max_abs
        <= float(args.max_neutral_velocity_abs),
    }
    training_args = checkpoint.get("args", {})
    loss_keys = (
        "flow_weight",
        "corrupt_stock_weight",
        "correct_gain_weight",
        "correct_gain_margin",
        "correct_corrupt_rank_weight",
        "correct_corrupt_margin",
        "delta_norm_weight",
        "perturb_flow_weight",
        "perturb_consistency_weight",
        "perturb_gain_weight",
        "perturb_gain_margin",
    )
    protocol = {
        "cache_config_hash": cache_manifest_payload.get("config_hash"),
        "indices": str(args.indices),
        "max_samples": int(args.max_samples),
        "sample_uids": [str(row["uid"]) for row in dataset.rows[:count]],
        "checkpoint_step": int(checkpoint.get("step", 0)),
        "noise_seeds": seeds,
        "t_values": t_values,
        "corruption_modes": list(CORRUPTION_MODES),
        "corruption_roles": {
            "pose_perturb": "robust_perturbation_to_target",
            "pose_shuffle": "hard_invalid_to_stock",
            "depth_corrupt": "hard_invalid_to_stock",
        },
        "corruption_config": {
            "lifting_volume_version": LIFTING_VOLUME_VERSION,
            "pose_perturb_rotation_degrees": 3.0,
            "pose_perturb_translation": 0.02,
            "pose_shuffle": "deterministic_view_roll_by_one",
            "depth_corruption_scale": 1.15,
            "depth_corruption_spatial": "roll_h_div_6_w_div_5_then_horizontal_flip",
        },
        "adapter_hidden_dim": int(args.adapter_hidden_dim),
        "physical_scale": float(args.physical_scale),
        "training_loss_weights": {
            key: training_args.get(key) for key in loss_keys
        },
        "model_version": checkpoint.get("format"),
    }
    report = {
        "passed": all(checks.values()),
        "checks": checks,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_step": int(checkpoint.get("step", 0)),
        "train_seed": int(checkpoint.get("args", {}).get("seed", -1)),
        "args": vars(args),
        "model": model_summary,
        "protocol": protocol,
        "sample_count": count,
        "object_count": len(object_rows),
        "record_count": len(records),
        "null_max_abs_diff": null_max_abs,
        "correct_gain_vs_stock": summarize(correct_gains),
        "object_balanced_correct_gain_vs_stock": summarize(object_correct_gains),
        "correct_stock_object_win_rate": correct_stock_object_win_rate,
        "object_win_rates_correct_vs_corruption": object_win_rates,
        "positive_t_count": positive_t,
        "t_summary": t_summary,
        "comparison_t_summary": comparison_t_summary,
        "hard_positive_t_count": hard_positive_t,
        "pose_perturb_positive_t_count": perturb_positive_t,
        "comparisons": comparison_summary,
        "neutral_velocity_mse_vs_stock": summarize(
            [row["neutral_velocity_mse_vs_stock"] for row in records]
        ),
        "neutral_velocity_max_abs_vs_stock": neutral_max_abs,
        "records": records,
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    markdown = [
        "# Local Pose Lifting Teacher-forced Evaluation",
        "",
        f"- passed: `{report['passed']}`",
        f"- samples / objects / records: `{count} / {len(object_rows)} / {len(records)}`",
        f"- correct gain mean / median: `{mean(correct_gains):.6g} / {median(correct_gains):.6g}`",
        f"- positive t: `{positive_t}/{len(t_values)}`",
        f"- physical-off/null max abs diff: `{null_max_abs:.6g}`",
        f"- correct-vs-stock object win rate: `{correct_stock_object_win_rate:.6g}`",
        f"- neutral velocity max abs: `{neutral_max_abs:.6g}`",
        "",
        "| corruption | correct-minus-corrupt mean | object win rate | gain vs stock | mean abs gain vs stock |",
        "|---|---:|---:|---:|---:|",
    ]
    for mode in CORRUPTION_MODES:
        item = comparison_summary[mode]
        markdown.append(
            f"| {mode} | {item['correct_gain_difference']['mean']:.6g} | "
            f"{object_win_rates[mode]:.6g} | "
            f"{item['corrupt_gain_vs_stock']['mean']:.6g} | "
            f"{item['mean_abs_corrupt_gain_vs_stock']:.6g} |"
        )
    (output_dir / "report.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "checks": checks}, indent=2))
    if args.fail_on_error and not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
