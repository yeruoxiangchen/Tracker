#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import sys
from typing import Any

os.environ.setdefault("SPCONV_ALGO", "native")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
os.environ.setdefault("TORCH_EXTENSIONS_DIR", "/tmp/torch_extensions")

import numpy as np
import torch
import torch.nn.functional as F

TRACKER_ROOT = Path(__file__).resolve().parents[1]
RECONVIAGEN_ROOT = TRACKER_ROOT / "ReconViaGen"
VGGT_ROOT = RECONVIAGEN_ROOT / "wheels" / "vggt"
for path in (TRACKER_ROOT, RECONVIAGEN_ROOT, VGGT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from reconvggt_ar_adapter_a.pointpose_ss_condition import load_partial_state  # noqa: E402
from reconvggt_ar_adapter_a.sparse_anchor_flow import (  # noqa: E402
    build_sparse_anchor_masks,
    shift_sparse_prior,
)
from reconvggt_ar_adapter_a.train_pointpose_ss_lora import (  # noqa: E402
    PointPoseCacheDataset,
)
from reconvggt_ar_adapter_a.train_sparse_anchor_ss_flow import (  # noqa: E402
    build_pipeline_and_model,
    build_stock_condition,
    parse_shifts,
    stock_equivalence_audit,
    validate_checkpoint,
)


def parse_ints(text: str) -> list[int]:
    values = [int(item.strip()) for item in str(text).split(",") if item.strip()]
    if not values or len(set(values)) != len(values):
        raise ValueError(f"expected unique integer list, got {text!r}")
    return values


def parse_floats(text: str) -> list[float]:
    values = [float(item.strip()) for item in str(text).split(",") if item.strip()]
    if not values or len(set(values)) != len(values) or any(not 0 < value < 1 for value in values):
        raise ValueError(f"expected unique t values in (0,1), got {text!r}")
    return values


def distribution(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise ValueError("distribution requires at least one value")
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p25": float(np.quantile(array, 0.25)),
        "p75": float(np.quantile(array, 0.75)),
        "min": float(array.min()),
        "max": float(array.max()),
        "positive_rate": float((array > 0).mean()),
        "negative_rate": float((array < 0).mean()),
    }


def selected_mean(values: torch.Tensor, mask: torch.Tensor) -> float:
    selected = values[mask.expand_as(values)]
    if selected.numel() == 0:
        raise RuntimeError("evaluation mask is empty")
    return float(selected.float().mean().item())


def source_metrics(
    logits: torch.Tensor,
    velocity: torch.Tensor,
    stock_velocity: torch.Tensor,
    gt_velocity: torch.Tensor,
    masks: dict[str, torch.Tensor],
) -> dict[str, float]:
    probability = torch.sigmoid(logits.float())
    positive = masks["positive64"]
    negative = masks["negative64"]
    neutral = masks["neutral16"]
    return {
        "flow_mse": float(F.mse_loss(velocity.float(), gt_velocity.float()).item()),
        "positive_probability": selected_mean(probability, positive),
        "positive_recall": selected_mean((probability >= 0.5).float(), positive),
        "outside_probability": selected_mean(probability, negative),
        "outside_fpr": selected_mean((probability >= 0.5).float(), negative),
        "neutral_velocity_mse": selected_mean(
            (velocity.float() - stock_velocity.float()).square(), neutral
        ),
        "velocity_delta_rms": float(
            (velocity.float() - stock_velocity.float()).square().mean().sqrt().item()
        ),
    }


def summarize(rows: list[dict[str, Any]], t_values: list[float]) -> dict[str, Any]:
    comparison_keys = (
        "stock_minus_correct_flow",
        "correct_minus_stock_positive_probability",
        "correct_minus_corrupted_positive_probability",
        "stock_minus_correct_outside_probability",
    )
    row_comparisons = {
        key: distribution([float(row["comparisons"][key]) for row in rows])
        for key in comparison_keys
    }
    objects = sorted({str(row["object_uid"]) for row in rows})
    by_object_values = {key: [] for key in comparison_keys}
    for object_uid in objects:
        selected = [row for row in rows if str(row["object_uid"]) == object_uid]
        for key in comparison_keys:
            by_object_values[key].append(
                float(np.mean([float(row["comparisons"][key]) for row in selected]))
            )
    by_t: dict[str, Any] = {}
    for t_value in t_values:
        selected = [row for row in rows if abs(float(row["t"]) - t_value) < 1.0e-8]
        by_t[f"{t_value:.3f}"] = {
            key: distribution([float(row["comparisons"][key]) for row in selected])
            for key in comparison_keys
        }
    return {
        "row_count": len(rows),
        "object_count": len(objects),
        "row_comparisons": row_comparisons,
        "by_object": {
            key: distribution(values) for key, values in by_object_values.items()
        },
        "by_t": by_t,
        "correct_neutral_velocity_mse": distribution(
            [float(row["sources"]["correct"]["neutral_velocity_mse"]) for row in rows]
        ),
        "correct_velocity_delta_rms": distribution(
            [float(row["sources"]["correct"]["velocity_delta_rms"]) for row in rows]
        ),
        "correct_positive_probability": distribution(
            [float(row["sources"]["correct"]["positive_probability"]) for row in rows]
        ),
        "correct_outside_probability": distribution(
            [float(row["sources"]["correct"]["outside_probability"]) for row in rows]
        ),
    }


def make_decision(
    summary: dict[str, Any],
    stock_equivalence: dict[str, Any],
    *,
    min_object_win_rate: float,
    max_outside_degradation: float,
    max_neutral_velocity_mse: float,
) -> dict[str, Any]:
    stock_gain = summary["by_object"]["correct_minus_stock_positive_probability"]
    corrupt_gain = summary["by_object"]["correct_minus_corrupted_positive_probability"]
    flow_gain = summary["by_object"]["stock_minus_correct_flow"]
    outside_gain = summary["by_object"]["stock_minus_correct_outside_probability"]
    positive_t_stock = sum(
        float(row["correct_minus_stock_positive_probability"]["mean"]) > 0
        for row in summary["by_t"].values()
    )
    positive_t_corrupt = sum(
        float(row["correct_minus_corrupted_positive_probability"]["mean"]) > 0
        for row in summary["by_t"].values()
    )
    required_t = max(1, len(summary["by_t"]) - 1)
    stock_exact = (
        float(stock_equivalence["disabled_max_abs_diff"]) == 0.0
        and float(stock_equivalence["null_max_abs_diff"]) == 0.0
    )
    checks = {
        "stock_and_null_exact": stock_exact,
        "positive_correct_beats_stock_mean": float(stock_gain["mean"]) > 0,
        "positive_correct_beats_stock_median": float(stock_gain["median"]) > 0,
        "positive_correct_beats_stock_object_win": (
            float(stock_gain["positive_rate"]) >= float(min_object_win_rate)
        ),
        "positive_correct_beats_corrupted_mean": float(corrupt_gain["mean"]) > 0,
        "positive_correct_beats_corrupted_median": float(corrupt_gain["median"]) > 0,
        "positive_correct_beats_corrupted_object_win": (
            float(corrupt_gain["positive_rate"]) >= float(min_object_win_rate)
        ),
        "positive_t_correct_vs_stock": positive_t_stock >= required_t,
        "positive_t_correct_vs_corrupted": positive_t_corrupt >= required_t,
        "outside_not_degraded": (
            float(outside_gain["mean"]) >= -float(max_outside_degradation)
        ),
        "neutral_preserved": (
            float(summary["correct_neutral_velocity_mse"]["mean"])
            <= float(max_neutral_velocity_mse)
        ),
        "global_flow_not_degraded": float(flow_gain["mean"]) >= 0,
    }
    return {
        "checks": checks,
        "positive_t_correct_vs_stock": int(positive_t_stock),
        "positive_t_correct_vs_corrupted": int(positive_t_corrupt),
        "required_positive_t": int(required_t),
        "min_object_win_rate": float(min_object_win_rate),
        "max_outside_degradation": float(max_outside_degradation),
        "max_neutral_velocity_mse": float(max_neutral_velocity_mse),
        "passed": all(checks.values()),
    }


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    summary = report["summary"]
    decision = report["decision"]
    lines = [
        "# SS16 Sparse-anchor Teacher-forced Evaluation",
        "",
        f"- checkpoint: `{report['checkpoint']}`",
        f"- rows / objects: `{summary['row_count']} / {summary['object_count']}`",
        f"- decision: `{'PASS' if decision['passed'] else 'FAIL'}`",
        f"- stock equivalence: `{report['stock_equivalence']}`",
        "",
        "## Object-balanced Local Gains",
        "",
        "Positive values are improvements from the correct sparse anchor.",
        "",
        "| comparison | mean | median | object win |",
        "| --- | ---: | ---: | ---: |",
    ]
    for key in (
        "correct_minus_stock_positive_probability",
        "correct_minus_corrupted_positive_probability",
        "stock_minus_correct_outside_probability",
        "stock_minus_correct_flow",
    ):
        row = summary["by_object"][key]
        lines.append(
            f"| {key} | {row['mean']:.8f} | {row['median']:.8f} | "
            f"{row['positive_rate']:.2%} |"
        )
    lines.extend(
        [
            "",
            "## Absolute Diagnostics",
            "",
            f"- correct positive probability: `{summary['correct_positive_probability']['mean']:.8f}`",
            f"- correct outside probability: `{summary['correct_outside_probability']['mean']:.8f}`",
            f"- neutral velocity MSE: `{summary['correct_neutral_velocity_mse']['mean']:.10f}`",
            f"- velocity delta RMS: `{summary['correct_velocity_delta_rms']['mean']:.8f}`",
            "",
            "## Per-t Local Gains",
            "",
            "| t | correct-stock positive | correct-corrupted positive | stock-correct outside |",
            "| ---: | ---: | ---: | ---: |",
        ]
    )
    for t_key, row in summary["by_t"].items():
        lines.append(
            f"| {t_key} | "
            f"{row['correct_minus_stock_positive_probability']['mean']:.8f} | "
            f"{row['correct_minus_corrupted_positive_probability']['mean']:.8f} | "
            f"{row['stock_minus_correct_outside_probability']['mean']:.8f} |"
        )
    lines.extend(["", "## Checks", ""])
    for key, passed in decision["checks"].items():
        lines.append(f"- {key}: `{'PASS' if passed else 'FAIL'}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fixed-noise multi-t local evaluation for SS16 sparse-anchor adapter."
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
    parser.add_argument("--corruption_shifts", default="1,0,0;-1,0,0;0,1,0;0,-1,0;0,0,1;0,0,-1")
    parser.add_argument("--anchor_hidden_dim", type=int, default=96)
    parser.add_argument("--physical_scale", type=float, default=1.0)
    parser.add_argument("--prior_confidence_min", type=float, default=0.25)
    parser.add_argument("--prior_mask_support_min", type=float, default=0.0)
    parser.add_argument("--anchor_radius_16", type=int, default=1)
    parser.add_argument("--outside_visible_min", type=float, default=0.5)
    parser.add_argument("--outside_ratio_min", type=float, default=0.9)
    parser.add_argument("--negative_surface_margin_64", type=int, default=1)
    parser.add_argument("--min_object_win_rate", type=float, default=0.65)
    parser.add_argument("--max_outside_degradation", type=float, default=0.001)
    parser.add_argument("--max_neutral_velocity_mse", type=float, default=1.0e-5)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    saved_args = checkpoint.get("args", {})
    args.amp_dtype = str(saved_args.get("amp_dtype", "bf16"))
    args.amp_init_scale = float(saved_args.get("amp_init_scale", 8192.0))
    validate_checkpoint(checkpoint, args)
    seeds = parse_ints(args.seeds)
    t_values = parse_floats(args.t_values)
    shifts = parse_shifts(args.corruption_shifts)
    random.seed(seeds[0])
    np.random.seed(seeds[0])
    torch.manual_seed(seeds[0])
    device = torch.device(args.device)
    pipeline, model, decoder, model_summary = build_pipeline_and_model(args, device)
    load_info = load_partial_state(
        model,
        checkpoint["model_trainable_state"],
        require_all_trainable=True,
    )
    model.adapter.eval()
    model.stock_flow.eval()
    dataset = PointPoseCacheDataset(args.cache_manifest, indices=args.indices)
    count = len(dataset) if int(args.max_samples) <= 0 else min(len(dataset), int(args.max_samples))
    stock_equivalence = stock_equivalence_audit(
        pipeline, model, dataset[0], device, expect_zero_init=False
    )
    decoder_dtype = next(decoder.parameters()).dtype
    rows: list[dict[str, Any]] = []

    with torch.no_grad():
        for index in range(count):
            sample = dataset[index]
            physical = sample["physical_grid"].unsqueeze(0).to(device=device, dtype=torch.float32)
            target = sample["target"].unsqueeze(0).to(device=device, dtype=torch.float32)
            target_coords = sample["target_coords"].to(device=device)
            masks = build_sparse_anchor_masks(
                physical,
                target_coords,
                prior_confidence_min=float(args.prior_confidence_min),
                prior_mask_support_min=float(args.prior_mask_support_min),
                anchor_radius_16=int(args.anchor_radius_16),
                outside_visible_min=float(args.outside_visible_min),
                outside_ratio_min=float(args.outside_ratio_min),
                negative_surface_margin_64=int(args.negative_surface_margin_64),
            )
            cond = build_stock_condition(pipeline, sample, device)
            for seed_position, seed in enumerate(seeds):
                generator = torch.Generator(device=device).manual_seed(
                    int(seed) * 1000003 + index
                )
                noise = torch.randn(target.shape, device=device, dtype=target.dtype, generator=generator)
                shift = shifts[(seed_position + index) % len(shifts)]
                corrupted = shift_sparse_prior(physical, shift)
                for t_value in t_values:
                    x_t, gt_velocity = pipeline.sparse_structure_sampler._get_model_gt(
                        target, t_value, noise
                    )
                    t_tensor = torch.full(
                        (1,), 1000.0 * t_value, device=device, dtype=torch.float32
                    )
                    stock = model.stock_prediction(x_t, t_tensor, cond)
                    correct, _ = model.adapt_from_stock(
                        x_t,
                        t_tensor,
                        stock,
                        physical,
                        physical_scale=float(args.physical_scale),
                    )
                    corrupted_prediction, _ = model.adapt_from_stock(
                        x_t,
                        t_tensor,
                        stock,
                        corrupted,
                        physical_scale=float(args.physical_scale),
                    )
                    logits: dict[str, torch.Tensor] = {}
                    for source, velocity in (
                        ("stock", stock),
                        ("correct", correct),
                        ("corrupted", corrupted_prediction),
                    ):
                        x0 = pipeline.sparse_structure_sampler._pred_to_xstart(
                            x_t, t_value, velocity
                        )
                        logits[source] = decoder(x0.to(dtype=decoder_dtype)).float()
                    sources = {
                        "stock": source_metrics(logits["stock"], stock, stock, gt_velocity, masks),
                        "correct": source_metrics(logits["correct"], correct, stock, gt_velocity, masks),
                        "corrupted": source_metrics(
                            logits["corrupted"], corrupted_prediction, stock, gt_velocity, masks
                        ),
                    }
                    comparisons = {
                        "stock_minus_correct_flow": (
                            sources["stock"]["flow_mse"] - sources["correct"]["flow_mse"]
                        ),
                        "correct_minus_stock_positive_probability": (
                            sources["correct"]["positive_probability"]
                            - sources["stock"]["positive_probability"]
                        ),
                        "correct_minus_corrupted_positive_probability": (
                            sources["correct"]["positive_probability"]
                            - sources["corrupted"]["positive_probability"]
                        ),
                        "stock_minus_correct_outside_probability": (
                            sources["stock"]["outside_probability"]
                            - sources["correct"]["outside_probability"]
                        ),
                    }
                    rows.append(
                        {
                            "uid": str(sample["uid"]),
                            "object_uid": str(sample["object_uid"]),
                            "seed": int(seed),
                            "t": float(t_value),
                            "corruption_shift": list(shift),
                            "positive64_count": int(masks["positive64"].sum().item()),
                            "negative64_count": int(masks["negative64"].sum().item()),
                            "sources": sources,
                            "comparisons": comparisons,
                        }
                    )

    summary = summarize(rows, t_values)
    decision = make_decision(
        summary,
        stock_equivalence,
        min_object_win_rate=float(args.min_object_win_rate),
        max_outside_degradation=float(args.max_outside_degradation),
        max_neutral_velocity_mse=float(args.max_neutral_velocity_mse),
    )
    report = {
        "format": "reconvggt.sparse_anchor_ss_flow_eval.v1",
        "args": vars(args),
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": int(checkpoint.get("step", -1)),
        "checkpoint_train_seed": int(saved_args.get("seed", -1)),
        "load_info": load_info,
        "model": model_summary,
        "stock_equivalence": stock_equivalence,
        "summary": summary,
        "decision": decision,
        "rows": rows,
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_markdown(output_dir / "report.md", report)
    print(json.dumps({"passed": decision["passed"], "checks": decision["checks"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()

