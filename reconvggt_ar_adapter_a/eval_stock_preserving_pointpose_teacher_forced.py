#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
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
from reconvggt_ar_adapter_a.experiment_gates import strict_decision_exit_code  # noqa: E402
from reconvggt_ar_adapter_a.stock_preserving_pointpose_bridge import (  # noqa: E402
    make_null_physical_grid,
)
from reconvggt_ar_adapter_a.pointpose_patch_features import (  # noqa: E402
    make_null_projected_patch_features,
)
from reconvggt_ar_adapter_a.train_pointpose_ss_lora import (  # noqa: E402
    PointPoseCacheDataset,
    encode_frozen_features,
    rgba_images,
)
from reconvggt_ar_adapter_a.train_stock_preserving_pointpose_bridge import (  # noqa: E402
    HardNegativeMiner,
    build_bridge_condition_inputs,
    build_models,
    paired_tensor_diagnostics,
)


SOURCES = ("stock", "correct", "shuffled", "disabled")


def parse_float_list(text: str) -> list[float]:
    values = [float(item.strip()) for item in str(text).split(",") if item.strip()]
    if not values or len(set(values)) != len(values):
        raise ValueError(f"expected unique float values, got {text!r}")
    if any(value <= 0 or value >= 1 for value in values):
        raise ValueError(f"teacher-forced t values must be in (0,1), got {values}")
    return values


def parse_int_list(text: str) -> list[int]:
    values = [int(item.strip()) for item in str(text).split(",") if item.strip()]
    if not values or len(set(values)) != len(values):
        raise ValueError(f"expected unique integer values, got {text!r}")
    return values


def distribution(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
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


def summarize_rows(rows: list[dict[str, Any]], t_values: list[float]) -> dict[str, Any]:
    losses = {
        source: distribution([float(row["losses"][source]) for row in rows])
        for source in SOURCES
    }
    comparisons = {
        "stock_minus_correct": distribution(
            [float(row["losses"]["stock"] - row["losses"]["correct"]) for row in rows]
        ),
        "shuffled_minus_correct": distribution(
            [float(row["losses"]["shuffled"] - row["losses"]["correct"]) for row in rows]
        ),
        "disabled_minus_correct": distribution(
            [float(row["losses"]["disabled"] - row["losses"]["correct"]) for row in rows]
        ),
    }
    by_t: dict[str, Any] = {}
    for value in t_values:
        selected = [row for row in rows if abs(float(row["t"]) - value) < 1.0e-8]
        by_t[f"{value:.3f}"] = {
            "stock_minus_correct": distribution(
                [float(row["losses"]["stock"] - row["losses"]["correct"]) for row in selected]
            ),
            "shuffled_minus_correct": distribution(
                [float(row["losses"]["shuffled"] - row["losses"]["correct"]) for row in selected]
            ),
        }
    objects = sorted({str(row["object_uid"]) for row in rows})
    object_deltas: dict[str, list[float]] = {
        "stock_minus_correct": [],
        "shuffled_minus_correct": [],
    }
    for object_uid in objects:
        selected = [row for row in rows if str(row["object_uid"]) == object_uid]
        object_deltas["stock_minus_correct"].append(
            float(np.mean([row["losses"]["stock"] - row["losses"]["correct"] for row in selected]))
        )
        object_deltas["shuffled_minus_correct"].append(
            float(
                np.mean(
                    [row["losses"]["shuffled"] - row["losses"]["correct"] for row in selected]
                )
            )
        )
    by_object = {key: distribution(values) for key, values in object_deltas.items()}
    return {
        "row_count": len(rows),
        "object_count": len(objects),
        "losses": losses,
        "comparisons": comparisons,
        "by_t": by_t,
        "by_object": by_object,
    }


def decision(
    summary: dict[str, Any],
    condition_summary: dict[str, Any],
    *,
    min_specificity_ratio: float,
    min_alignment_gap: float,
    min_shuffled_object_win_rate: float,
) -> dict[str, Any]:
    stock = summary["by_object"]["stock_minus_correct"]
    shuffled = summary["by_object"]["shuffled_minus_correct"]
    by_t = summary["by_t"]
    stock_positive_t = sum(
        float(item["stock_minus_correct"]["mean"]) > 0 for item in by_t.values()
    )
    shuffled_positive_t = sum(
        float(item["shuffled_minus_correct"]["mean"]) > 0 for item in by_t.values()
    )
    stock_gain = float(stock["mean"])
    shuffled_gain = float(shuffled["mean"])
    signed_specificity_ratio = shuffled_gain / max(abs(stock_gain), 1.0e-12)
    specificity_ratio = max(0.0, shuffled_gain) / max(abs(stock_gain), 1.0e-12)
    alignment_gap = float(condition_summary["alignment_probability_gap"]["mean"])
    checks = {
        "correct_beats_stock_mean": float(stock["mean"]) > 0,
        "correct_beats_stock_median": float(stock["median"]) > 0,
        "correct_beats_stock_object_win_rate": float(stock["positive_rate"]) > 0.55,
        "correct_beats_shuffled_mean": float(shuffled["mean"]) > 0,
        "correct_beats_shuffled_median": float(shuffled["median"]) > 0,
        "correct_beats_shuffled_object_win_rate": (
            float(shuffled["positive_rate"]) >= float(min_shuffled_object_win_rate)
        ),
        "stock_positive_t_count": stock_positive_t >= max(1, len(by_t) - 1),
        "shuffled_positive_t_count": shuffled_positive_t >= max(1, len(by_t) - 1),
        "physical_specificity_ratio": specificity_ratio >= float(min_specificity_ratio),
        "alignment_probability_gap": alignment_gap >= float(min_alignment_gap),
    }
    return {
        "checks": checks,
        "stock_positive_t_count": int(stock_positive_t),
        "shuffled_positive_t_count": int(shuffled_positive_t),
        "required_positive_t_count": max(1, len(by_t) - 1),
        "physical_specificity_ratio": float(specificity_ratio),
        "signed_physical_specificity_ratio": float(signed_specificity_ratio),
        "required_physical_specificity_ratio": float(min_specificity_ratio),
        "alignment_probability_gap": float(alignment_gap),
        "required_alignment_probability_gap": float(min_alignment_gap),
        "required_shuffled_object_win_rate": float(min_shuffled_object_win_rate),
        "passed": all(checks.values()),
    }


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    summary = report["summary"]
    decision_row = report["decision"]
    lines = [
        "# Stock-preserving PointPose Teacher-forced Evaluation",
        "",
        f"- checkpoint: `{report['checkpoint']}`",
        f"- rows / objects: `{summary['row_count']} / {summary['object_count']}`",
        f"- decision: `{'PASS' if decision_row['passed'] else 'FAIL'}`",
        f"- hard stock equivalence: `{report['stock_equivalence']}`",
        f"- physical specificity ratio: `{decision_row['physical_specificity_ratio']:.6f}`",
        f"- signed physical specificity ratio: "
        f"`{decision_row['signed_physical_specificity_ratio']:.6f}`",
        f"- alignment probability gap: `{decision_row['alignment_probability_gap']:.6f}`",
        f"- correct-shuffled condition RMS: "
        f"`{report['condition_summary']['correct_shuffled_condition_rms']['mean']:.8f}`",
        "",
        "## Object-balanced improvements",
        "",
        "Positive values mean the correct physical condition has lower flow MSE.",
        "",
        "| comparison | mean | median | object win rate |",
        "| --- | ---: | ---: | ---: |",
    ]
    for key in ("stock_minus_correct", "shuffled_minus_correct"):
        row = summary["by_object"][key]
        lines.append(
            f"| {key} | {row['mean']:.8f} | {row['median']:.8f} | {row['positive_rate']:.2%} |"
        )
    lines.extend(
        [
            "",
            "## Per-t improvements",
            "",
            "| t | stock-correct mean | shuffled-correct mean |",
            "| ---: | ---: | ---: |",
        ]
    )
    for t_key, row in summary["by_t"].items():
        lines.append(
            f"| {t_key} | {row['stock_minus_correct']['mean']:.8f} | "
            f"{row['shuffled_minus_correct']['mean']:.8f} |"
        )
    stage_names = [
        f"stage_{index}"
        for index in report["model"].get("bridge", {}).get("fusion_stages", [])
    ]
    if stage_names:
        fusion_position = report["model"].get("bridge", {}).get(
            "fusion_position", "selected bridge stage"
        )
        lines.extend(
            [
                "",
                "## Per-stage physical diagnostics",
                "",
                f"Fusion position: `{fusion_position}`.",
                "",
                "| stage | correct gate | shuffled gate | gate gap | "
                "correct delta/hidden | shuffled delta/hidden |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        condition = report["condition_summary"]
        for stage_name in stage_names:
            correct_gate = condition[f"{stage_name}_correct_alignment_probability"]["mean"]
            shuffled_gate = condition[f"{stage_name}_shuffled_alignment_probability"]["mean"]
            gap = condition[f"{stage_name}_alignment_probability_gap"]["mean"]
            correct_ratio = condition[f"{stage_name}_correct_delta_to_hidden_ratio"]["mean"]
            shuffled_ratio = condition[f"{stage_name}_shuffled_delta_to_hidden_ratio"]["mean"]
            lines.append(
                f"| {stage_name} | {correct_gate:.6f} | {shuffled_gate:.6f} | "
                f"{gap:.6f} | {correct_ratio:.8f} | {shuffled_ratio:.8f} |"
            )
        first_pair_key = f"{stage_names[0]}_attended_correct_shuffled_rms"
        if first_pair_key in condition:
            lines.extend(
                [
                    "",
                    "### Content interaction diagnostics",
                    "",
                    "| stage | attended correct-shuffled RMS | attended cosine | "
                    "context delta correct-shuffled RMS | context delta cosine |",
                    "| --- | ---: | ---: | ---: | ---: |",
                ]
            )
            for stage_name in stage_names:
                lines.append(
                    f"| {stage_name} | "
                    f"{condition[f'{stage_name}_attended_correct_shuffled_rms']['mean']:.8f} | "
                    f"{condition[f'{stage_name}_attended_correct_shuffled_cosine']['mean']:.6f} | "
                    f"{condition[f'{stage_name}_context_delta_correct_shuffled_rms']['mean']:.8f} | "
                    f"{condition[f'{stage_name}_context_delta_correct_shuffled_cosine']['mean']:.6f} |"
                )
    lines.extend(["", "## Decision checks", ""])
    for key, passed in decision_row["checks"].items():
        lines.append(f"- {key}: `{'PASS' if passed else 'FAIL'}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-t paired teacher-forced evaluation for J0/J1a stock-preserving fusion."
    )
    parser.add_argument("--cache_manifest", required=True)
    parser.add_argument("--checkpoint", default="none")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--pretrained", default="Stable-X/trellis-vggt-v0-2")
    parser.add_argument("--indices", default="all")
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--t_values", default="0.1,0.3,0.5,0.7,0.9")
    parser.add_argument("--physical_scale", type=float, default=1.0)
    parser.add_argument("--physical_hidden_dim", type=int, default=256)
    parser.add_argument("--physical_heads", type=int, default=8)
    parser.add_argument(
        "--bridge_fusion_mode",
        choices=[
            "last1_cross_attention",
            "multistage_local16",
            "content_visual8",
            "pose_guided_patch",
        ],
        default="last1_cross_attention",
    )
    parser.add_argument("--bridge_last_blocks", type=int, choices=[1, 2], default=1)
    parser.add_argument("--fusion_stages", default="0,1,2,3")
    parser.add_argument("--local_fusion_hidden_dim", type=int, default=128)
    parser.add_argument("--content_fusion_dim", type=int, default=128)
    parser.add_argument("--content_fusion_heads", type=int, default=4)
    parser.add_argument("--min_specificity_ratio", type=float, default=0.05)
    parser.add_argument("--min_alignment_gap", type=float, default=0.01)
    parser.add_argument("--min_shuffled_object_win_rate", type=float, default=0.65)
    parser.add_argument(
        "--fail_on_decision",
        action="store_true",
        help="Exit with status 2 after writing reports when the specificity decision fails.",
    )
    args = parser.parse_args()
    args.gradient_checkpointing = False
    if args.min_specificity_ratio < 0 or args.min_alignment_gap < 0:
        raise ValueError("specificity and alignment thresholds must be non-negative")
    if not 0 <= args.min_shuffled_object_win_rate <= 1:
        raise ValueError("min_shuffled_object_win_rate must be in [0,1]")

    seeds = parse_int_list(args.seeds)
    t_values = parse_float_list(args.t_values)
    random.seed(seeds[0])
    np.random.seed(seeds[0])
    torch.manual_seed(seeds[0])
    device = torch.device(args.device)
    pipeline, model, model_summary = build_models(args, device)
    checkpoint_path = None if str(args.checkpoint).lower() == "none" else Path(args.checkpoint)
    checkpoint_step = 0
    checkpoint_train_seed = None
    load_info = None
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        saved_args = checkpoint.get("args", {})
        checkpoint_train_seed = int(saved_args.get("seed", -1))
        expected = {
            "pretrained": str(args.pretrained),
            "physical_hidden_dim": int(args.physical_hidden_dim),
            "physical_heads": int(args.physical_heads),
            "bridge_last_blocks": int(args.bridge_last_blocks),
            "bridge_fusion_mode": str(args.bridge_fusion_mode),
            "fusion_stages": str(args.fusion_stages),
            "local_fusion_hidden_dim": int(args.local_fusion_hidden_dim),
            "content_fusion_dim": int(args.content_fusion_dim),
            "content_fusion_heads": int(args.content_fusion_heads),
        }
        legacy_defaults = {
            "bridge_fusion_mode": "last1_cross_attention",
            "fusion_stages": "0,1,2,3",
            "local_fusion_hidden_dim": 128,
            "content_fusion_dim": 128,
            "content_fusion_heads": 4,
        }
        mismatch = {
            key: {
                "checkpoint": saved_args.get(key, legacy_defaults.get(key)),
                "current": value,
            }
            for key, value in expected.items()
            if saved_args.get(key, legacy_defaults.get(key)) != value
        }
        if str(args.bridge_fusion_mode) in {
            "multistage_local16",
            "content_visual8",
            "pose_guided_patch",
        }:
            saved_bridge_version = (
                checkpoint.get("model_summary", {})
                .get("bridge", {})
                .get("version")
            )
            current_bridge_version = model.bridge_fusion.metadata()["version"]
            if saved_bridge_version != current_bridge_version:
                mismatch["bridge_version"] = {
                    "checkpoint": saved_bridge_version,
                    "current": current_bridge_version,
                }
        if mismatch:
            raise RuntimeError(f"teacher-forced checkpoint mismatch: {mismatch}")
        load_info = load_partial_state(
            model,
            checkpoint["model_trainable_state"],
            require_all_trainable=True,
        )
        checkpoint_step = int(checkpoint.get("step", -1))
    model.eval()

    dataset = PointPoseCacheDataset(args.cache_manifest, indices=args.indices)
    negative_miner = HardNegativeMiner(dataset)
    count = len(dataset) if int(args.max_samples) <= 0 else min(len(dataset), int(args.max_samples))
    rows: list[dict[str, Any]] = []
    stock_equivalence = {
        "condition_disabled_max_abs_diff": 0.0,
        "velocity_disabled_max_abs_diff": 0.0,
        "stock_vs_split_condition_max_abs_diff": 0.0,
        "condition_exact": True,
        "velocity_exact": True,
        "flow_lora_enabled": False,
        "null_present_checked": bool(getattr(model.bridge_fusion, "paired_training", False)),
        "null_evidence_preserves_xyz": bool(
            getattr(model.bridge_fusion, "paired_training", False)
            and not getattr(model.bridge_fusion, "projected_patch_fusion", False)
        ),
        "null_pose_ray_uv_preserved": bool(
            getattr(model.bridge_fusion, "projected_patch_fusion", False)
        ),
        "null_present_condition_max_abs_diff": 0.0,
        "null_present_condition_exact": True,
        "null_present_velocity_max_abs_diff": 0.0,
        "null_present_velocity_exact": True,
    }
    condition_stats: list[dict[str, float]] = []

    for index in range(count):
        sample = dataset[index]
        negative_index = negative_miner[index]
        negative = dataset[negative_index]
        images = rgba_images(sample["image_paths"], sample["mask_paths"], pipeline)
        aggregated, image_cond = encode_frozen_features(pipeline, images)
        physical, negative_physical, _ = build_bridge_condition_inputs(
            model,
            sample,
            negative,
            aggregated,
            device,
        )
        target = sample["target"].unsqueeze(0).to(device=device, dtype=torch.float32)
        with torch.no_grad():
            cond_stock = model.bridge_fusion.stock_condition(aggregated, image_cond)
            shared_kwargs = {"cond_stock": cond_stock} if bool(
                getattr(model.bridge_fusion, "paired_training", False)
            ) else {}
            correct_paths = model.bridge_fusion.condition_paths(
                aggregated,
                image_cond,
                physical,
                physical_scale=float(args.physical_scale),
                **shared_kwargs,
            )
            shuffled_paths = model.bridge_fusion.condition_paths(
                aggregated,
                image_cond,
                negative_physical,
                physical_scale=float(args.physical_scale),
                **shared_kwargs,
            )
            cond_disabled = model.bridge_fusion.condition(
                aggregated,
                image_cond,
                None,
                physical_present=False,
            )
            split_diff = float(
                (correct_paths.cond_stock.float() - cond_stock.float()).abs().max().item()
            )
            stock_equivalence["stock_vs_split_condition_max_abs_diff"] = max(
                float(stock_equivalence["stock_vs_split_condition_max_abs_diff"]), split_diff
            )
            correct_shuffled_delta = (
                correct_paths.cond_fused.float() - shuffled_paths.cond_fused.float()
            )
            if bool(getattr(model.bridge_fusion, "paired_training", False)):
                null_physical = (
                    make_null_projected_patch_features(physical)
                    if bool(
                        getattr(model.bridge_fusion, "projected_patch_fusion", False)
                    )
                    else make_null_physical_grid(physical)
                )
                null_paths = model.bridge_fusion.condition_paths(
                    aggregated,
                    image_cond,
                    null_physical,
                    physical_scale=float(args.physical_scale),
                    cond_stock=cond_stock,
                )
                null_present_delta = null_paths.cond_fused.float() - cond_stock.float()
                stock_equivalence["null_present_condition_max_abs_diff"] = max(
                    float(stock_equivalence["null_present_condition_max_abs_diff"]),
                    float(null_present_delta.abs().amax().item()),
                )
                stock_equivalence["null_present_condition_exact"] = bool(
                    stock_equivalence["null_present_condition_exact"]
                    and torch.equal(null_paths.cond_fused, cond_stock)
                )
            else:
                null_present_delta = torch.zeros_like(cond_stock.float())
            correct_probability = torch.sigmoid(correct_paths.alignment_logit).mean()
            shuffled_probability = torch.sigmoid(shuffled_paths.alignment_logit).mean()
            condition_row = {
                    "correct_delta_rms": float(correct_paths.stats["condition_delta_rms"].item()),
                    "correct_delta_ratio": float(
                        correct_paths.stats["condition_delta_to_stock_ratio"].item()
                    ),
                    "correct_alignment_probability": float(
                        correct_probability.item()
                    ),
                    "shuffled_alignment_probability": float(
                        shuffled_probability.item()
                    ),
                    "alignment_probability_gap": float(
                        (correct_probability - shuffled_probability).item()
                    ),
                    "correct_shuffled_condition_rms": float(
                        torch.sqrt(correct_shuffled_delta.square().mean()).item()
                    ),
                    "correct_shuffled_condition_abs_max": float(
                        correct_shuffled_delta.abs().amax().item()
                    ),
                    "null_present_condition_rms": float(
                        torch.sqrt(null_present_delta.square().mean()).item()
                    ),
                    "null_present_condition_abs_max": float(
                        null_present_delta.abs().amax().item()
                    ),
                }
            for stage_name, correct_stage in (correct_paths.stage_stats or {}).items():
                shuffled_stage = (shuffled_paths.stage_stats or {})[stage_name]
                correct_gate = correct_stage["alignment_probability_mean"]
                shuffled_gate = shuffled_stage["alignment_probability_mean"]
                condition_row.update(
                    {
                        f"{stage_name}_correct_alignment_probability": float(
                            correct_gate.item()
                        ),
                        f"{stage_name}_shuffled_alignment_probability": float(
                            shuffled_gate.item()
                        ),
                        f"{stage_name}_alignment_probability_gap": float(
                            (correct_gate - shuffled_gate).item()
                        ),
                        f"{stage_name}_correct_delta_rms": float(
                            correct_stage["effective_delta_rms"].item()
                        ),
                        f"{stage_name}_shuffled_delta_rms": float(
                            shuffled_stage["effective_delta_rms"].item()
                        ),
                        f"{stage_name}_correct_delta_to_hidden_ratio": float(
                            correct_stage["effective_delta_to_hidden_ratio"].item()
                        ),
                        f"{stage_name}_shuffled_delta_to_hidden_ratio": float(
                            shuffled_stage["effective_delta_to_hidden_ratio"].item()
                        ),
                    }
                )
                correct_tensors = (correct_paths.stage_tensors or {}).get(stage_name)
                shuffled_tensors = (shuffled_paths.stage_tensors or {}).get(stage_name)
                if correct_tensors is not None and shuffled_tensors is not None:
                    attended_pair = paired_tensor_diagnostics(
                        correct_tensors["attended_centered"],
                        shuffled_tensors["attended_centered"],
                    )
                    context_pair = paired_tensor_diagnostics(
                        correct_tensors["context_delta_effective"],
                        shuffled_tensors["context_delta_effective"],
                    )
                    condition_row.update(
                        {
                            f"{stage_name}_attended_correct_shuffled_rms": float(
                                attended_pair["difference_rms"].item()
                            ),
                            f"{stage_name}_attended_correct_shuffled_cosine": float(
                                attended_pair["cosine"].item()
                            ),
                            f"{stage_name}_context_delta_correct_shuffled_rms": float(
                                context_pair["difference_rms"].item()
                            ),
                            f"{stage_name}_context_delta_correct_shuffled_cosine": float(
                                context_pair["cosine"].item()
                            ),
                        }
                    )
            condition_stats.append(condition_row)
        condition_diff = float((cond_disabled.float() - cond_stock.float()).abs().max().item())
        stock_equivalence["condition_disabled_max_abs_diff"] = max(
            float(stock_equivalence["condition_disabled_max_abs_diff"]), condition_diff
        )
        stock_equivalence["condition_exact"] = bool(
            stock_equivalence["condition_exact"] and torch.equal(cond_disabled, cond_stock)
        )

        for seed in seeds:
            for t_index, t_value in enumerate(t_values):
                generator = torch.Generator(device=device).manual_seed(
                    int(seed) + index * 1009 + t_index * 1000003
                )
                noise = torch.randn(target.shape, generator=generator, device=device, dtype=target.dtype)
                x_t, gt_velocity = pipeline.sparse_structure_sampler._get_model_gt(
                    target, float(t_value), noise
                )
                t_tensor = torch.full(
                    (1,), 1000.0 * float(t_value), device=device, dtype=torch.float32
                )
                with torch.no_grad():
                    prediction_stock = model.flow(x_t, t_tensor, cond_stock)
                    prediction_correct = model.flow(x_t, t_tensor, correct_paths.cond_fused)
                    prediction_shuffled = model.flow(x_t, t_tensor, shuffled_paths.cond_fused)
                    prediction_disabled = model.flow(x_t, t_tensor, cond_disabled)
                    prediction_null = (
                        model.flow(x_t, t_tensor, null_paths.cond_fused)
                        if bool(getattr(model.bridge_fusion, "paired_training", False))
                        else prediction_stock
                    )
                velocity_diff = float(
                    (prediction_disabled.float() - prediction_stock.float()).abs().max().item()
                )
                stock_equivalence["velocity_disabled_max_abs_diff"] = max(
                    float(stock_equivalence["velocity_disabled_max_abs_diff"]), velocity_diff
                )
                stock_equivalence["velocity_exact"] = bool(
                    stock_equivalence["velocity_exact"]
                    and torch.equal(prediction_disabled, prediction_stock)
                )
                null_velocity_diff = float(
                    (prediction_null.float() - prediction_stock.float()).abs().max().item()
                )
                stock_equivalence["null_present_velocity_max_abs_diff"] = max(
                    float(stock_equivalence["null_present_velocity_max_abs_diff"]),
                    null_velocity_diff,
                )
                stock_equivalence["null_present_velocity_exact"] = bool(
                    stock_equivalence["null_present_velocity_exact"]
                    and torch.equal(prediction_null, prediction_stock)
                )
                predictions = {
                    "stock": prediction_stock,
                    "correct": prediction_correct,
                    "shuffled": prediction_shuffled,
                    "disabled": prediction_disabled,
                }
                losses = {
                    source: float(
                        F.mse_loss(prediction.float(), gt_velocity.float()).item()
                    )
                    for source, prediction in predictions.items()
                }
                rows.append(
                    {
                        "index": index,
                        "uid": sample["uid"],
                        "object_uid": sample["object_uid"],
                        "negative_uid": negative["uid"],
                        "seed": int(seed),
                        "t": float(t_value),
                        "losses": losses,
                    }
                )
        print(
            f"[j1a_teacher] {index + 1}/{count} uid={sample['uid']} "
            f"gate_correct={condition_stats[-1]['correct_alignment_probability']:.4f} "
            f"gate_shuffled={condition_stats[-1]['shuffled_alignment_probability']:.4f}",
            flush=True,
        )

    summary = summarize_rows(rows, t_values)
    condition_summary = {
        key: distribution([row[key] for row in condition_stats])
        for key in condition_stats[0]
    }
    decision_row = decision(
        summary,
        condition_summary,
        min_specificity_ratio=float(args.min_specificity_ratio),
        min_alignment_gap=float(args.min_alignment_gap),
        min_shuffled_object_win_rate=float(args.min_shuffled_object_win_rate),
    )
    report = {
        "format": "reconvggt.stock_preserving_pointpose_teacher_forced.v1",
        "args": vars(args),
        "checkpoint": "none" if checkpoint_path is None else str(checkpoint_path),
        "checkpoint_step": int(checkpoint_step),
        "checkpoint_train_seed": checkpoint_train_seed,
        "load_info": load_info,
        "model": model_summary,
        "stock_equivalence": stock_equivalence,
        "condition_summary": condition_summary,
        "summary": summary,
        "decision": decision_row,
        "rows": rows,
    }
    if not (
        stock_equivalence["condition_exact"]
        and stock_equivalence["velocity_exact"]
        and stock_equivalence["condition_disabled_max_abs_diff"] == 0.0
        and stock_equivalence["velocity_disabled_max_abs_diff"] == 0.0
        and not stock_equivalence["flow_lora_enabled"]
        and (
            not stock_equivalence["null_present_checked"]
            or (
                stock_equivalence["null_present_condition_exact"]
                and stock_equivalence["null_present_condition_max_abs_diff"] == 0.0
                and stock_equivalence["null_present_velocity_exact"]
                and stock_equivalence["null_present_velocity_max_abs_diff"] == 0.0
            )
        )
    ):
        raise RuntimeError(f"stock-preserving teacher-forced audit failed: {stock_equivalence}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_markdown(output_dir / "report.md", report)
    print(
        json.dumps(
            {
                "stock_equivalence": stock_equivalence,
                "condition_summary": condition_summary,
                "by_object": summary["by_object"],
                "decision": decision_row,
            },
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )
    if args.fail_on_decision:
        exit_code = strict_decision_exit_code(bool(decision_row["passed"]))
        if exit_code:
            raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
