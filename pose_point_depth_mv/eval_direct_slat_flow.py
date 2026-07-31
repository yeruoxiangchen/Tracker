#!/usr/bin/env python3
"""Teacher-forced stock/LoRA/adapter/full evaluation for direct SLAT Flow."""

from __future__ import annotations

import argparse
from collections import defaultdict
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

from pose_point_depth_mv.direct_slat_data import DirectSLatCacheDataset  # noqa: E402
from pose_point_depth_mv.direct_slat_flow import (  # noqa: E402
    DIRECT_SLAT_FLOW_VERSION,
    DIRECT_SLAT_TRAINING_SEMANTICS_V3,
    DIRECT_SLAT_TRAINING_SEMANTICS_V4,
    DIRECT_SLAT_TRAINING_SEMANTICS_V5,
    SLAT_SUPPORT_INTERVAL_CFG_ACTIVE,
    assert_disjoint_object_splits,
    build_direct_slat_components,
    canonical_json_sha256,
    cfg_interval_is_active,
    combine_sparse_cfg,
    deterministic_wrong_support_index,
    load_strict_trainable_state,
    resolve_slat_guided_delta_policy,
    resolve_slat_delta_bound_mode,
    resolve_slat_support_interval_policy,
    resolve_slat_residual_combination_policy,
    slat_target_cache_identity,
    support_generator_identity,
)
from pose_point_depth_mv.train_direct_slat_flow import (  # noqa: E402
    normalized_target,
    to_device_tree,
)
from trellis.modules import sparse as sp  # noqa: E402


def parse_csv(value: str, cast) -> list[Any]:
    result = [cast(item.strip()) for item in str(value).split(",") if item.strip()]
    if not result:
        raise ValueError("CSV argument is empty")
    return result


def bootstrap_mean_ci(
    values: list[float], *, samples: int, seed: int
) -> list[float]:
    if not values:
        return [float("nan"), float("nan")]
    array = np.asarray(values, dtype=np.float64)
    generator = np.random.default_rng(seed)
    means = np.empty(int(samples), dtype=np.float64)
    for index in range(int(samples)):
        means[index] = generator.choice(array, size=len(array), replace=True).mean()
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def summarize(values: list[float], *, bootstrap_samples: int, seed: int) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "mean": float("nan"),
            "median": float("nan"),
            "positive_rate": float("nan"),
            "bootstrap_mean_95_ci": [float("nan"), float("nan")],
        }
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "positive_rate": float(np.mean(array > 0)),
        "bootstrap_mean_95_ci": bootstrap_mean_ci(
            values, samples=bootstrap_samples, seed=seed
        ),
    }


def select_evaluation_indices(
    rows: list[dict[str, Any]],
    *,
    max_samples: int = 0,
    max_objects: int = 0,
) -> list[int]:
    """Select rows directly or select complete objects in first-seen order."""

    sample_limit = int(max_samples)
    object_limit = int(max_objects)
    if sample_limit < 0 or object_limit < 0:
        raise ValueError("max_samples and max_objects must be non-negative")
    if sample_limit > 0 and object_limit > 0:
        raise ValueError("max_samples and max_objects are mutually exclusive")
    if sample_limit > 0:
        return list(range(min(sample_limit, len(rows))))
    if object_limit <= 0:
        return list(range(len(rows)))

    selected_object_uids: list[str] = []
    selected_object_set: set[str] = set()
    for row in rows:
        object_uid = str(row.get("object_uid", ""))
        if not object_uid:
            raise ValueError("evaluation row is missing object_uid")
        if object_uid in selected_object_set:
            continue
        selected_object_uids.append(object_uid)
        selected_object_set.add(object_uid)
        if len(selected_object_uids) == object_limit:
            break
    if len(selected_object_uids) < object_limit:
        raise ValueError(
            f"requested max_objects={object_limit}, but only "
            f"{len(selected_object_uids)} unique objects are available"
        )
    return [
        index
        for index, row in enumerate(rows)
        if str(row["object_uid"]) in selected_object_set
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache_manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--pretrained", default="Stable-X/trellis-vggt-v0-2")
    parser.add_argument("--indices", default="all")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="Evaluate at most this many cache rows (legacy row-based selection).",
    )
    selection.add_argument(
        "--max_objects",
        type=int,
        default=0,
        help=(
            "Evaluate the first N unique object_uids and include every cache row "
            "for each selected object."
        ),
    )
    parser.add_argument("--noise_seeds", default="42,43,44")
    parser.add_argument("--t_values", default="0.1,0.3,0.5,0.7,0.9")
    parser.add_argument("--amp_dtype", choices=("bf16", "fp16", "none"), default="bf16")
    parser.add_argument("--support_scale", type=float, default=None)
    parser.add_argument(
        "--slat_delta_scale",
        type=float,
        default=None,
        help="Override checkpoint Direct-SLAT residual scale.",
    )
    parser.add_argument(
        "--slat_delta_rms_ratio_cap",
        type=float,
        default=None,
        help=(
            "Override checkpoint per-batch RMS ratio cap; a negative value "
            "explicitly disables clipping."
        ),
    )
    parser.add_argument(
        "--max_slat_points",
        type=int,
        default=0,
        help="0 reuses the training checkpoint value (normally 40960).",
    )
    parser.add_argument("--bootstrap_samples", type=int, default=5000)
    parser.add_argument("--train_cache_manifest", default="")
    parser.add_argument("--verify_cache_hashes", action="store_true")
    parser.add_argument(
        "--decision_profile", choices=("report_only", "strict"), default="report_only"
    )
    parser.add_argument("--min_object_win_rate", type=float, default=0.55)
    parser.add_argument("--require_mechanism_gate", action="store_true")
    parser.add_argument("--min_correct_wrong_win_rate", type=float, default=0.60)
    parser.add_argument(
        "--correct_wrong_margin",
        type=float,
        default=None,
        help="Override checkpoint correct-over-wrong support loss margin.",
    )
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    noise_seeds = parse_csv(args.noise_seeds, int)
    t_values = parse_csv(args.t_values, float)
    if any(value <= 0 or value >= 1 for value in t_values):
        raise ValueError("teacher-forced t values must lie strictly inside (0,1)")
    if int(args.bootstrap_samples) <= 0:
        raise ValueError("bootstrap_samples must be positive")
    device = torch.device("cuda")
    torch.cuda.set_device(0)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    dataset = DirectSLatCacheDataset(
        args.cache_manifest,
        indices=args.indices,
        verify_hashes=bool(args.verify_cache_hashes),
    )
    selected_indices = select_evaluation_indices(
        dataset.rows,
        max_samples=int(args.max_samples),
        max_objects=int(args.max_objects),
    )
    if not selected_indices:
        raise ValueError("evaluation subset is empty")
    split_audit = None
    if args.train_cache_manifest:
        train_payload = json.loads(
            Path(args.train_cache_manifest).read_text(encoding="utf-8")
        )
        split_audit = assert_disjoint_object_splits(
            train_payload["samples"], dataset.rows
        )
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    if checkpoint.get("format") != DIRECT_SLAT_FLOW_VERSION:
        raise ValueError(f"unexpected checkpoint format={checkpoint.get('format')!r}")
    saved_args = checkpoint["args"]
    rollout_aligned = saved_args.get("training_semantics") in {
        DIRECT_SLAT_TRAINING_SEMANTICS_V3,
        DIRECT_SLAT_TRAINING_SEMANTICS_V4,
        DIRECT_SLAT_TRAINING_SEMANTICS_V5,
    }
    train_cfg_strength = float(saved_args.get("train_cfg_strength", 1.0))
    train_cfg_interval = tuple(
        float(value)
        for value in saved_args.get("train_cfg_interval", (0.5, 1.0))
    )
    rollout_guided_delta_policy = resolve_slat_guided_delta_policy(saved_args)
    slat_delta_bound_mode = resolve_slat_delta_bound_mode(saved_args)
    support_interval_policy = resolve_slat_support_interval_policy(saved_args)
    slat_residual_combination_policy = (
        resolve_slat_residual_combination_policy(saved_args)
    )
    slat_lora_delta_scale = float(
        saved_args.get("slat_lora_delta_scale", 1.0)
    )
    slat_lora_delta_rms_ratio_cap = float(
        saved_args.get("slat_lora_delta_rms_ratio_cap", -1.0)
    )
    slat_support_delta_scale = float(
        saved_args.get("slat_support_delta_scale", 1.0)
    )
    slat_support_delta_rms_ratio_cap = float(
        saved_args.get("slat_support_delta_rms_ratio_cap", -1.0)
    )
    if str(saved_args["pretrained"]) != str(args.pretrained):
        raise RuntimeError("checkpoint pretrained binding differs from evaluation")
    eval_support_generator = support_generator_identity(dataset.config)
    if checkpoint.get("model_summary", {}).get("support_generator") != eval_support_generator:
        raise RuntimeError(
            "evaluation cache was generated by a different frozen SS support protocol"
        )
    sampler, model, _, normalization, model_summary = build_direct_slat_components(
        pretrained=args.pretrained,
        adapter_hidden_dim=int(saved_args["adapter_hidden_dim"]),
        lora_rank=int(saved_args["lora_rank"]),
        lora_alpha=int(saved_args["lora_alpha"]),
        gradient_checkpointing=False,
        device=device,
    )
    load_strict_trainable_state(model, checkpoint["model_trainable_state"])
    model.eval()
    runtime_normalization = {
        key: [float(item) for item in value] for key, value in normalization.items()
    }
    if canonical_json_sha256(runtime_normalization) != dataset.slat_normalization_hash:
        raise RuntimeError("runtime SLAT normalization differs from evaluation cache")
    mean = torch.tensor(runtime_normalization["mean"], device=device)[None]
    std = torch.tensor(runtime_normalization["std"], device=device)[None]
    support_scale = (
        float(saved_args.get("support_scale", 1.0))
        if args.support_scale is None
        else float(args.support_scale)
    )
    slat_delta_scale = (
        float(saved_args.get("slat_delta_scale", 1.0))
        if args.slat_delta_scale is None
        else float(args.slat_delta_scale)
    )
    saved_ratio_cap = float(saved_args.get("slat_delta_rms_ratio_cap", -1.0))
    requested_ratio_cap = (
        saved_ratio_cap
        if args.slat_delta_rms_ratio_cap is None
        else float(args.slat_delta_rms_ratio_cap)
    )
    if slat_delta_scale < 0:
        raise ValueError("slat_delta_scale must be non-negative")
    slat_delta_rms_ratio_cap = (
        None if requested_ratio_cap < 0 else requested_ratio_cap
    )
    correct_wrong_margin = (
        float(saved_args.get("wrong_support_margin", 0.0))
        if args.correct_wrong_margin is None
        else float(args.correct_wrong_margin)
    )
    if correct_wrong_margin < 0:
        raise ValueError("correct_wrong_margin must be non-negative")
    if not 0.0 <= float(args.min_correct_wrong_win_rate) <= 1.0:
        raise ValueError("min_correct_wrong_win_rate must be within [0, 1]")
    max_slat_points = (
        int(saved_args.get("max_slat_points", 40960))
        if int(args.max_slat_points) <= 0
        else int(args.max_slat_points)
    )
    use_amp = args.amp_dtype != "none"
    amp_dtype = torch.float16 if args.amp_dtype == "fp16" else torch.bfloat16
    records: list[dict[str, Any]] = []
    for order, index in enumerate(selected_indices, start=1):
        sample = dataset[index]
        target = normalized_target(
            sample,
            mean=mean,
            std=std,
            device=device,
            max_points=max_slat_points,
            selection_seed=72022026 + int(index) * 1013,
        )
        condition = to_device_tree(sample["condition"]["cond"], device)
        negative_condition = (
            to_device_tree(sample["condition"]["neg_cond"], device)
            if rollout_aligned
            else None
        )
        support = (
            sample["corrected_ss"].to(device=device),
            sample["occupancy_logits64"].to(device=device),
            sample["physical_tokens16"].to(device=device),
        )
        wrong_sample = None
        wrong_support = None
        if args.require_mechanism_gate:
            wrong_index = deterministic_wrong_support_index(
                dataset.rows,
                correct_object_uid=str(sample["object_uid"]),
                support_seed=int(sample["support_seed"]),
                selection_seed=72022026 + int(index) * 1013,
            )
            wrong_sample = dataset[wrong_index]
            wrong_support = (
                wrong_sample["corrected_ss"].to(device=device),
                wrong_sample["occupancy_logits64"].to(device=device),
                wrong_sample["physical_tokens16"].to(device=device),
            )
        for noise_seed in noise_seeds:
            generator = torch.Generator(device=device).manual_seed(
                int(noise_seed) * 3000017 + int(index) * 3011
            )
            noise = sp.SparseTensor(
                feats=torch.randn(
                    target.feats.shape,
                    generator=generator,
                    device=device,
                    dtype=target.feats.dtype,
                ),
                coords=target.coords,
            )
            for t_value in t_values:
                x_t, gt_velocity = sampler._get_model_gt(target, t_value, noise)
                t_tensor = torch.full(
                    (1,), 1000.0 * t_value, device=device, dtype=torch.float32
                )
                with torch.autocast(
                    device_type="cuda", dtype=amp_dtype, enabled=use_amp
                ):
                    stock_positive = model.stock_prediction(
                        x_t, t_tensor, condition
                    )
                    cfg_active = (
                        rollout_aligned
                        and cfg_interval_is_active(t_value, train_cfg_interval)
                    )
                    stock_negative = (
                        model.stock_prediction(
                            x_t, t_tensor, negative_condition
                        )
                        if cfg_active
                        else None
                    )
                    support_active = (
                        True
                        if support_interval_policy
                        != SLAT_SUPPORT_INTERVAL_CFG_ACTIVE
                        else cfg_active
                    )
                    if rollout_aligned:
                        full, stock, stats = (
                            model.post_cfg_conditioned_prediction(
                                x_t,
                                t_tensor,
                                condition,
                                negative_condition,
                                corrected_ss=support[0],
                                occupancy_logits64=support[1],
                                physical_tokens16=support[2],
                                stock_positive_velocity=stock_positive,
                                stock_negative_velocity=stock_negative,
                                cfg_strength=train_cfg_strength,
                                cfg_active=cfg_active,
                                support_scale=support_scale,
                                slat_delta_scale=slat_delta_scale,
                                slat_delta_rms_ratio_cap=slat_delta_rms_ratio_cap,
                                slat_delta_bound_mode=slat_delta_bound_mode,
                                slat_residual_combination_policy=(
                                    slat_residual_combination_policy
                                ),
                                slat_lora_delta_scale=slat_lora_delta_scale,
                                slat_lora_delta_rms_ratio_cap=(
                                    slat_lora_delta_rms_ratio_cap
                                ),
                                slat_support_delta_scale=(
                                    slat_support_delta_scale
                                ),
                                slat_support_delta_rms_ratio_cap=(
                                    slat_support_delta_rms_ratio_cap
                                ),
                                support_active=support_active,
                            )
                        )
                    else:
                        stock = stock_positive
                        full, stats = model.conditioned_prediction(
                            x_t,
                            t_tensor,
                            condition,
                            corrected_ss=support[0],
                            occupancy_logits64=support[1],
                            physical_tokens16=support[2],
                            stock_velocity=stock,
                            support_scale=support_scale,
                            slat_delta_scale=slat_delta_scale,
                            slat_delta_rms_ratio_cap=slat_delta_rms_ratio_cap,
                            slat_delta_bound_mode=slat_delta_bound_mode,
                        )
                    lora_positive = model.lora_only_prediction(
                        x_t, t_tensor, condition
                    )
                    adapter_positive = model.adapter_only_prediction(
                        x_t,
                        t_tensor,
                        condition,
                        corrected_ss=support[0],
                        occupancy_logits64=support[1],
                        physical_tokens16=support[2],
                        support_scale=support_scale,
                    )
                    if stock_negative is not None:
                        lora_only = combine_sparse_cfg(
                            lora_positive,
                            stock_negative,
                            cfg_strength=train_cfg_strength,
                        )
                        adapter_only = combine_sparse_cfg(
                            adapter_positive,
                            stock_negative,
                            cfg_strength=train_cfg_strength,
                        )
                    else:
                        lora_only = lora_positive
                        adapter_only = adapter_positive
                    wrong_prediction = None
                    if wrong_support is not None:
                        if rollout_aligned:
                            wrong_prediction, wrong_stock, _ = (
                                model.post_cfg_conditioned_prediction(
                                    x_t,
                                    t_tensor,
                                    condition,
                                    negative_condition,
                                    corrected_ss=wrong_support[0],
                                    occupancy_logits64=wrong_support[1],
                                    physical_tokens16=wrong_support[2],
                                    stock_positive_velocity=stock_positive,
                                    stock_negative_velocity=stock_negative,
                                    cfg_strength=train_cfg_strength,
                                    cfg_active=cfg_active,
                                    support_scale=support_scale,
                                    slat_delta_scale=slat_delta_scale,
                                    slat_delta_rms_ratio_cap=slat_delta_rms_ratio_cap,
                                    slat_delta_bound_mode=slat_delta_bound_mode,
                                    slat_residual_combination_policy=(
                                        slat_residual_combination_policy
                                    ),
                                    slat_lora_delta_scale=(
                                        slat_lora_delta_scale
                                    ),
                                    slat_lora_delta_rms_ratio_cap=(
                                        slat_lora_delta_rms_ratio_cap
                                    ),
                                    slat_support_delta_scale=(
                                        slat_support_delta_scale
                                    ),
                                    slat_support_delta_rms_ratio_cap=(
                                        slat_support_delta_rms_ratio_cap
                                    ),
                                    support_active=support_active,
                                )
                            )
                            if not torch.equal(wrong_stock.feats, stock.feats):
                                raise RuntimeError(
                                    "correct/wrong evaluation stock references differ"
                                )
                        else:
                            wrong_prediction, _ = model.conditioned_prediction(
                                x_t,
                                t_tensor,
                                condition,
                                corrected_ss=wrong_support[0],
                                occupancy_logits64=wrong_support[1],
                                physical_tokens16=wrong_support[2],
                                stock_velocity=stock,
                                support_scale=support_scale,
                                slat_delta_scale=slat_delta_scale,
                                slat_delta_rms_ratio_cap=slat_delta_rms_ratio_cap,
                                slat_delta_bound_mode=slat_delta_bound_mode,
                            )
                losses = {
                    name: float(
                        F.mse_loss(
                            prediction.feats.float(), gt_velocity.feats.float()
                        ).item()
                    )
                    for name, prediction in (
                        ("stock", stock),
                        ("full", full),
                        ("lora_only", lora_only),
                        ("adapter_only", adapter_only),
                    )
                }
                if wrong_prediction is not None:
                    losses["wrong_support"] = float(
                        F.mse_loss(
                            wrong_prediction.feats.float(),
                            gt_velocity.feats.float(),
                        ).item()
                    )
                correct_over_wrong_advantage = (
                    losses["wrong_support"] - losses["full"]
                    if "wrong_support" in losses
                    else None
                )
                correct_support_stock_mse = float(
                    F.mse_loss(
                        full.feats.float(),
                        stock.feats.float(),
                    ).item()
                )
                wrong_support_stock_mse = (
                    float(
                        F.mse_loss(
                            wrong_prediction.feats.float(),
                            stock.feats.float(),
                        ).item()
                    )
                    if wrong_prediction is not None
                    else None
                )
                wrong_support_stock_reversion_advantage = (
                    correct_support_stock_mse - wrong_support_stock_mse
                    if wrong_support_stock_mse is not None
                    else None
                )
                records.append(
                    {
                        "sample_index": index,
                        "uid": str(sample["uid"]),
                        "object_uid": str(sample["object_uid"]),
                        "views": int(sample["view_count"]),
                        "support_seed": int(sample["support_seed"]),
                        "wrong_support_uid": (
                            str(wrong_sample["uid"])
                            if wrong_sample is not None
                            else ""
                        ),
                        "wrong_support_object_uid": (
                            str(wrong_sample["object_uid"])
                            if wrong_sample is not None
                            else ""
                        ),
                        "noise_seed": int(noise_seed),
                        "t": float(t_value),
                        "losses": losses,
                        "full_gain_vs_stock": losses["stock"] - losses["full"],
                        "lora_only_gain_vs_stock": losses["stock"] - losses["lora_only"],
                        "adapter_only_gain_vs_stock": losses["stock"] - losses["adapter_only"],
                        "full_gain_vs_lora_only": losses["lora_only"] - losses["full"],
                        "full_gain_vs_adapter_only": losses["adapter_only"] - losses["full"],
                        "correct_over_wrong_support_advantage": (
                            correct_over_wrong_advantage
                        ),
                        "correct_over_wrong_margin_pass": (
                            correct_over_wrong_advantage >= correct_wrong_margin
                            if correct_over_wrong_advantage is not None
                            else None
                        ),
                        "correct_support_stock_mse": correct_support_stock_mse,
                        "wrong_support_stock_mse": wrong_support_stock_mse,
                        "wrong_support_stock_reversion_advantage": (
                            wrong_support_stock_reversion_advantage
                        ),
                        "cfg_active": bool(cfg_active),
                        "applied_cfg_strength": (
                            train_cfg_strength if cfg_active else 1.0
                        ),
                        "support_token_rms": float(stats["support_token_rms"].item()),
                        "flow_delta_rms": float(stats["flow_delta_rms"].item()),
                        "stock_velocity_rms": float(
                            stats["stock_velocity_rms"].item()
                        ),
                        "raw_flow_delta_rms": float(
                            stats["raw_flow_delta_rms"].item()
                        ),
                        "effective_flow_delta_rms": float(
                            stats["effective_flow_delta_rms"].item()
                        ),
                        "delta_clip_scale": float(
                            stats["delta_clip_scale"].item()
                        ),
                        "delta_clip_activated": bool(
                            stats["delta_clip_activated"].item() > 0.5
                        ),
                        "raw_flow_delta_abs_max": float(
                            stats["raw_flow_delta_abs_max"].item()
                        ),
                        "effective_flow_delta_abs_max": float(
                            stats["effective_flow_delta_abs_max"].item()
                        ),
                    }
                )
        print(
            f"[direct_slat_eval] {order}/{len(selected_indices)} {sample['uid']} "
            f"support_seed={sample['support_seed']}",
            flush=True,
        )
        torch.cuda.empty_cache()

    by_object: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        by_object[str(row["object_uid"])].append(row)
    metric_names = (
        "full_gain_vs_stock",
        "lora_only_gain_vs_stock",
        "adapter_only_gain_vs_stock",
        "full_gain_vs_lora_only",
        "full_gain_vs_adapter_only",
    )
    mechanism_metric_name = (
        "wrong_support_stock_reversion_advantage"
        if rollout_aligned
        else "correct_over_wrong_support_advantage"
    )
    if args.require_mechanism_gate:
        metric_names = (*metric_names, mechanism_metric_name)
    object_rows = []
    for object_uid, values in sorted(by_object.items()):
        object_rows.append(
            {
                "object_uid": object_uid,
                **{
                    name: float(np.mean([float(row[name]) for row in values]))
                    for name in metric_names
                },
                "record_count": len(values),
            }
        )
    summary = {
        name: summarize(
            [float(row[name]) for row in object_rows],
            bootstrap_samples=int(args.bootstrap_samples),
            seed=42000 + position,
        )
        for position, name in enumerate(metric_names)
    }
    primary = summary["full_gain_vs_stock"]
    checks = {
        "mean_positive": primary["mean"] > 0,
        "median_positive": primary["median"] > 0,
        "object_win_at_least_threshold": primary["positive_rate"]
        >= float(args.min_object_win_rate),
        "bootstrap_ci_lower_positive": primary["bootstrap_mean_95_ci"][0] > 0,
        "full_beats_lora_only_mean": summary["full_gain_vs_lora_only"]["mean"] > 0,
        "full_beats_adapter_only_mean": summary["full_gain_vs_adapter_only"]["mean"] > 0,
    }
    mechanism_pass = True
    if args.require_mechanism_gate:
        mechanism = summary[mechanism_metric_name]
        prefix = (
            "wrong_support_stock_reversion"
            if rollout_aligned
            else "correct_support"
        )
        checks[f"{prefix}_mean_advantage_meets_margin"] = (
            mechanism["mean"] >= correct_wrong_margin
        )
        checks[f"{prefix}_median_advantage_positive"] = mechanism["median"] > 0
        checks[f"{prefix}_win_rate"] = (
            mechanism["positive_rate"] >= float(args.min_correct_wrong_win_rate)
        )
        checks[f"{prefix}_ci_lower_positive"] = (
            mechanism["bootstrap_mean_95_ci"][0] > 0
        )
        mechanism_pass = all(
            checks[f"{prefix}_{suffix}"]
            for suffix in (
                "mean_advantage_meets_margin",
                "median_advantage_positive",
                "win_rate",
            )
        )
    core_pass = all(
        checks[name]
        for name in (
            "mean_positive",
            "median_positive",
            "object_win_at_least_threshold",
        )
    )
    strong_pass = core_pass and checks["bootstrap_ci_lower_positive"]
    if args.require_mechanism_gate:
        core_pass = core_pass and mechanism_pass
        strong_pass = (
            strong_pass
            and mechanism_pass
            and checks[f"{prefix}_ci_lower_positive"]
        )
    report = {
        "format": DIRECT_SLAT_FLOW_VERSION,
        "evaluation": "teacher-forced GT SLAT coordinates/features",
        "teacher_prediction_policy": (
            "deployed_post_cfg_velocity"
            if rollout_aligned
            else "positive_condition_velocity"
        ),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_step": int(checkpoint.get("step", -1)),
        "cache_identity": slat_target_cache_identity(
            args.cache_manifest,
            rows=[dataset.rows[index] for index in selected_indices],
        ),
        "evaluation_selection": {
            "mode": (
                "complete_objects"
                if int(args.max_objects) > 0
                else "cache_rows" if int(args.max_samples) > 0 else "all"
            ),
            "requested_max_samples": int(args.max_samples),
            "requested_max_objects": int(args.max_objects),
            "selected_sample_rows": len(selected_indices),
            "selected_object_count": len(object_rows),
        },
        "train_eval_split_audit": split_audit,
        "sample_rows": len(selected_indices),
        "object_count": len(object_rows),
        "noise_seeds": noise_seeds,
        "t_values": t_values,
        "support_scale": support_scale,
        "slat_delta_policy": {
            "scale": slat_delta_scale,
            "rms_ratio_cap": slat_delta_rms_ratio_cap,
            "per_sparse_batch": True,
            "rollout_guided_delta_policy": str(
                rollout_guided_delta_policy
            ),
            "bound_mode": slat_delta_bound_mode,
            "support_interval_policy": support_interval_policy,
            "residual_combination_policy": (
                slat_residual_combination_policy
            ),
            "lora_delta_scale": slat_lora_delta_scale,
            "lora_delta_rms_ratio_cap": slat_lora_delta_rms_ratio_cap,
            "support_delta_scale": slat_support_delta_scale,
            "support_delta_rms_ratio_cap": (
                slat_support_delta_rms_ratio_cap
            ),
            "train_cfg_strength": train_cfg_strength,
            "train_cfg_interval": list(train_cfg_interval),
        },
        "mechanism_gate": {
            "required": bool(args.require_mechanism_gate),
            "correct_wrong_margin": correct_wrong_margin,
            "min_correct_wrong_win_rate": float(args.min_correct_wrong_win_rate),
            "passed": bool(mechanism_pass),
            "metric": mechanism_metric_name,
            "selection": (
                "deterministic object-disjoint support; same support seed preferred"
            ),
        },
        "max_slat_points": max_slat_points,
        "formal_weighting": "records averaged per object before summary/bootstrap",
        "summary": summary,
        "checks": checks,
        "core_pass": core_pass,
        "strong_pass": strong_pass,
        "object_rows": object_rows,
        "records": records,
        "scope_guard": (
            "teacher-forced evidence selects checkpoints; corrected-coordinate rollout "
            "and same-noise Mesh comparison remain required before a science claim"
        ),
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    lines = [
        "Direct SLAT teacher-forced evaluation",
        "=====================================",
        f"objects: {len(object_rows)}",
        f"checkpoint step: {report['checkpoint_step']}",
        f"full gain vs stock mean: {primary['mean']:+.8f}",
        f"full gain vs stock median: {primary['median']:+.8f}",
        f"object win rate: {primary['positive_rate']:.6f}",
        f"bootstrap mean 95% CI: {primary['bootstrap_mean_95_ci']}",
        f"CORE PASS: {core_pass}",
        f"STRONG PASS: {strong_pass}",
    ]
    if args.require_mechanism_gate:
        mechanism = summary[mechanism_metric_name]
        lines.extend(
            [
                f"{mechanism_metric_name} mean: {mechanism['mean']:+.8f}",
                f"{mechanism_metric_name} median: {mechanism['median']:+.8f}",
                f"{mechanism_metric_name} win rate: {mechanism['positive_rate']:.6f}",
                f"MECHANISM PASS: {mechanism_pass}",
            ]
        )
    lines.extend(["", report["scope_guard"]])
    (output_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)
    if args.decision_profile == "strict" and not core_pass:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
