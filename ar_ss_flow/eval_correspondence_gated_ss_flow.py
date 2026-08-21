#!/usr/bin/env python3
from __future__ import annotations

import sys
import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import mean, median
from typing import Any

import torch
import torch.nn.functional as F

# Support both `python -m ar_ss_flow.<module>` and direct script execution.
TRACKER_ROOT = Path(__file__).resolve().parents[1]
if str(TRACKER_ROOT) not in sys.path:
    sys.path.insert(0, str(TRACKER_ROOT))

from ar_ss_flow.correspondence_gated_flow import CORRESPONDENCE_GATED_FLOW_VERSION
from ar_ss_flow.correspondence_lifting import (
    CORRESPONDENCE_NEGATIVE_MODES,
    correspondence_volume_from_sample,
    load_correspondence_checkpoint,
    parse_csv,
)
from ar_ss_flow.local_pose_lifting_flow import PoseLiftingCacheDataset
from ar_ss_flow.train_correspondence_gated_ss_flow import (
    FLOW_NEGATIVE_MODES,
    build_model,
    find_cross_sample,
)
from reconvggt_ar_adapter_a.pointpose_ss_condition import load_partial_state


def parse_int_csv(text: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in text.split(",") if item.strip())
    if not values:
        raise ValueError("integer CSV must be non-empty")
    return values


def parse_float_csv(text: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in text.split(",") if item.strip())
    if not values or any(value <= 0.0 or value >= 1.0 for value in values):
        raise ValueError("t values must lie in (0,1)")
    return values


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
        description="Fixed-noise paired evaluation for correspondence-gated SS residual."
    )
    parser.add_argument("--cache_manifest", required=True)
    parser.add_argument("--correspondence_checkpoint", required=True)
    parser.add_argument("--flow_checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--pretrained", default="Stable-X/trellis-vggt-v0-2")
    parser.add_argument("--indices", default="0-15")
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--t_values", default="0.1,0.3,0.5,0.7,0.9")
    parser.add_argument("--negative_modes", default="pose_cyclic1,pose_cyclic2,pose_reverse,cross_sample")
    parser.add_argument("--adapter_hidden_dim", type=int, default=96)
    parser.add_argument("--physical_scale", type=float, default=1.0)
    parser.add_argument("--confidence_threshold", type=float, default=0.55)
    parser.add_argument("--neighborhood_radius", type=int, default=1)
    parser.add_argument("--min_source_views", type=int, default=2)
    parser.add_argument("--amp_dtype", choices=("bf16", "fp16", "none"), default="bf16")
    parser.add_argument("--min_correct_stock_object_win_rate", type=float, default=0.65)
    parser.add_argument("--min_correct_wrong_object_win_rate", type=float, default=0.65)
    parser.add_argument("--min_positive_t", type=int, default=4)
    parser.add_argument("--max_wrong_abs_gain", type=float, default=0.005)
    parser.add_argument("--min_confidence_object_win_rate", type=float, default=0.65)
    parser.add_argument("--fail_on_error", action="store_true")
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    dataset = PoseLiftingCacheDataset(args.cache_manifest, indices=args.indices)
    count = len(dataset) if args.max_samples <= 0 else min(len(dataset), args.max_samples)
    modes = parse_csv(args.negative_modes)
    invalid = [mode for mode in modes if mode not in FLOW_NEGATIVE_MODES]
    if invalid:
        raise ValueError(f"invalid negative modes={invalid}")
    corr_model, corr_checkpoint = load_correspondence_checkpoint(
        args.correspondence_checkpoint,
        device=device,
        visual_channels=dataset.visual_feature_dim,
    )
    corr_model.eval()
    flow_checkpoint = torch.load(args.flow_checkpoint, map_location="cpu")
    if flow_checkpoint.get("format") != CORRESPONDENCE_GATED_FLOW_VERSION:
        raise ValueError(f"unexpected flow checkpoint format={flow_checkpoint.get('format')!r}")
    saved = flow_checkpoint.get("args", {})
    for key in (
        "pretrained",
        "adapter_hidden_dim",
        "confidence_threshold",
        "neighborhood_radius",
        "min_source_views",
    ):
        if key in saved:
            setattr(args, key, saved[key])
    saved_amp = saved.get("amp_dtype")
    if saved_amp in {"bf16", "fp16", "none"}:
        args.amp_dtype = saved_amp
    flow_sampler, model, model_summary = build_model(
        args, device, dataset.visual_feature_dim
    )
    load_partial_state(
        model,
        flow_checkpoint["model_trainable_state"],
        require_all_trainable=True,
    )
    model.eval()
    seeds = parse_int_csv(args.seeds)
    t_values = parse_float_csv(args.t_values)
    use_amp = args.amp_dtype != "none"
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16

    volume_cache: dict[tuple[int, str], tuple[torch.Tensor, torch.Tensor, dict[str, Any]]] = {}
    cross_cache: dict[int, torch.Tensor] = {}
    records: list[dict[str, Any]] = []
    null_max_abs = 0.0
    disabled_max_abs = 0.0
    for sample_index in range(count):
        sample = dataset[sample_index]
        correct_volume = correspondence_volume_from_sample(
            sample,
            device=device,
            model=corr_model,
            mode="correct",
            neighborhood_radius=int(args.neighborhood_radius),
            min_source_views=int(args.min_source_views),
            confidence_floor=float(args.confidence_threshold),
        )
        volume_cache[(sample_index, "correct")] = correct_volume
        for mode in modes:
            visual_override = None
            wrong_mode = mode
            if mode == "cross_sample":
                cross = find_cross_sample(dataset, sample_index, sample)
                visual_override = cross["visual_patch_features"]
                wrong_mode = "correct"
            volume_cache[(sample_index, mode)] = correspondence_volume_from_sample(
                sample,
                device=device,
                model=corr_model,
                mode=wrong_mode,
                visual_patch_features_override=visual_override,
                neighborhood_radius=int(args.neighborhood_radius),
                min_source_views=int(args.min_source_views),
                confidence_floor=float(args.confidence_threshold),
            )
        target = sample["target"].unsqueeze(0).to(device=device, dtype=torch.float32)
        condition = sample["stock_condition"].to(device=device)
        for seed in seeds:
            for t_value in t_values:
                combined_seed = seed * 1000003 + sample_index * 1009 + int(round(t_value * 1000))
                generator = torch.Generator(device=device).manual_seed(combined_seed)
                endpoint = torch.randn(target.shape, generator=generator, device=device)
                x_t, gt_velocity = flow_sampler._get_model_gt(target, t_value, endpoint)
                t_tensor = torch.full(
                    (1,), 1000.0 * t_value, device=device, dtype=torch.float32
                )
                with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
                    stock = model.stock_prediction(x_t, t_tensor, condition)
                    correct_prediction, correct_stats = model.adapt_from_stock(
                        x_t,
                        t_tensor,
                        stock,
                        correct_volume[0],
                        correct_volume[1],
                        scale=float(args.physical_scale),
                    )
                    disabled, _ = model.adapt_from_stock(
                        x_t,
                        t_tensor,
                        stock,
                        correct_volume[0],
                        correct_volume[1],
                        physical_present=False,
                    )
                    null, _ = model.adapt_from_stock(
                        x_t,
                        t_tensor,
                        stock,
                        torch.zeros_like(correct_volume[0]),
                        torch.zeros_like(correct_volume[1]),
                    )
                    predictions: dict[str, tuple[torch.Tensor, dict[str, torch.Tensor]]] = {}
                    for mode in modes:
                        volume = volume_cache[(sample_index, mode)]
                        predictions[mode] = model.adapt_from_stock(
                            x_t,
                            t_tensor,
                            stock,
                            volume[0],
                            volume[1],
                            scale=float(args.physical_scale),
                        )
                null_max_abs = max(null_max_abs, float((null - stock).abs().max().item()))
                disabled_max_abs = max(
                    disabled_max_abs, float((disabled - stock).abs().max().item())
                )
                stock_loss = float(F.mse_loss(stock.float(), gt_velocity.float()).item())
                correct_loss = float(
                    F.mse_loss(correct_prediction.float(), gt_velocity.float()).item()
                )
                for mode, (prediction, wrong_stats) in predictions.items():
                    wrong_loss = float(
                        F.mse_loss(prediction.float(), gt_velocity.float()).item()
                    )
                    wrong_stock = float(
                        F.mse_loss(prediction.float(), stock.float()).item()
                        / max(float(stock.float().square().mean().item()), 1.0e-6)
                    )
                    wrong_volume_stats = volume_cache[(sample_index, mode)][2]
                    records.append(
                        {
                            "sample_index": sample_index,
                            "uid": str(sample["uid"]),
                            "object_uid": str(sample.get("object_uid", sample["uid"])),
                            "seed": seed,
                            "t": t_value,
                            "mode": mode,
                            "stock_loss": stock_loss,
                            "correct_loss": correct_loss,
                            "wrong_loss": wrong_loss,
                            "correct_gain": stock_loss - correct_loss,
                            "correct_vs_wrong": wrong_loss - correct_loss,
                            "wrong_abs_gain": stock_loss - wrong_loss,
                            "wrong_stock_loss": wrong_stock,
                            "correct_delta_rms": float(correct_stats["delta_rms"].float().item()),
                            "wrong_delta_rms": float(wrong_stats["delta_rms"].float().item()),
                            "correct_gate_mean": float(correct_stats["gate_mean"].float().item()),
                            "wrong_gate_mean": float(wrong_stats["gate_mean"].float().item()),
                            "correct_corr_confidence": float(
                                correct_volume[2]["mean_correspondence_confidence"]
                            ),
                            "wrong_corr_confidence": float(
                                wrong_volume_stats["mean_correspondence_confidence"]
                            ),
                        }
                    )

    object_buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        object_buckets[(row["object_uid"], row["mode"])].append(row)
    object_rows: list[dict[str, Any]] = []
    for (object_uid, mode), rows in sorted(object_buckets.items()):
        object_rows.append(
            {
                "object_uid": object_uid,
                "mode": mode,
                "correct_gain": mean([float(row["correct_gain"]) for row in rows]),
                "correct_vs_wrong": mean(
                    [float(row["correct_vs_wrong"]) for row in rows]
                ),
                "wrong_abs_gain": mean([float(row["wrong_abs_gain"]) for row in rows]),
                "confidence_gap": mean(
                    [
                        float(row["correct_corr_confidence"])
                        - float(row["wrong_corr_confidence"])
                        for row in rows
                    ]
                ),
            }
        )
    summary: dict[str, Any] = {}
    checks: dict[str, Any] = {}
    for mode in modes:
        rows = [row for row in object_rows if row["mode"] == mode]
        correct_gains = [float(row["correct_gain"]) for row in rows]
        correct_wrong = [float(row["correct_vs_wrong"]) for row in rows]
        wrong_abs = [float(row["wrong_abs_gain"]) for row in rows]
        confidence_gap = [float(row["confidence_gap"]) for row in rows]
        t_direction = {}
        for t_value in t_values:
            t_rows = [
                row for row in records if row["mode"] == mode and row["t"] == t_value
            ]
            t_direction[str(t_value)] = mean(
                [float(row["correct_vs_wrong"]) for row in t_rows]
            ) if t_rows else 0.0
        positive_t = sum(value > 0.0 for value in t_direction.values())
        row_summary = {
            "object_count": len(rows),
            "correct_gain": summarize(correct_gains),
            "correct_stock_object_win_rate": (
                sum(value > 0.0 for value in correct_gains) / len(correct_gains)
                if correct_gains
                else 0.0
            ),
            "correct_vs_wrong": summarize(correct_wrong),
            "correct_wrong_object_win_rate": (
                sum(value > 0.0 for value in correct_wrong) / len(correct_wrong)
                if correct_wrong
                else 0.0
            ),
            "wrong_abs_gain": summarize(wrong_abs),
            "confidence_gap": summarize(confidence_gap),
            "confidence_object_win_rate": (
                sum(value > 0.0 for value in confidence_gap) / len(confidence_gap)
                if confidence_gap
                else 0.0
            ),
            "t_direction": t_direction,
            "positive_t": positive_t,
        }
        summary[mode] = row_summary
        checks[mode] = {
            "passed": (
                row_summary["correct_stock_object_win_rate"]
                >= float(args.min_correct_stock_object_win_rate)
                and row_summary["correct_wrong_object_win_rate"]
                >= float(args.min_correct_wrong_object_win_rate)
                and row_summary["correct_gain"]["mean"] > 0.0
                and row_summary["correct_gain"]["median"] > 0.0
                and row_summary["correct_vs_wrong"]["mean"] > 0.0
                and row_summary["correct_vs_wrong"]["median"] > 0.0
                and positive_t >= int(args.min_positive_t)
                and abs(row_summary["wrong_abs_gain"]["mean"])
                <= float(args.max_wrong_abs_gain)
                and row_summary["confidence_object_win_rate"]
                >= float(args.min_confidence_object_win_rate)
            ),
            "metrics": row_summary,
        }
    passed = (
        null_max_abs == 0.0
        and disabled_max_abs == 0.0
        and all(row["passed"] for row in checks.values())
    )
    report = {
        "stage": "C2 correspondence-gated SS fixed-noise evaluation",
        "passed": passed,
        "args": vars(args),
        "correspondence_checkpoint_step": int(corr_checkpoint.get("step", 0)),
        "flow_checkpoint_step": int(flow_checkpoint.get("step", 0)),
        "model_summary": model_summary,
        "null_max_abs_diff": null_max_abs,
        "physical_off_max_abs_diff": disabled_max_abs,
        "summary": summary,
        "checks": checks,
        "records": records,
        "object_rows": object_rows,
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)
    if args.fail_on_error and not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
