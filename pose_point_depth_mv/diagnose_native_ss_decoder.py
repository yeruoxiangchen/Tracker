#!/usr/bin/env python3
"""Development-only Native SS residual-scale and frozen-decoder diagnosis."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ar_ss_flow.local_pose_lifting_flow import PoseLiftingCacheDataset
from pose_point_depth_mv.eval_direct_flow import (
    bootstrap_mean_ci,
    overlap_metrics,
    positive_rate,
    summarize,
)
from pose_point_depth_mv.native_3d_condition import (
    NATIVE_SS_FLOW_VERSION,
    NativeStockSSFlow,
    PostCFGNativeSSRolloutFlow,
    build_native_ss_components,
    load_trainable_state_dict,
    native_ss_deployment_from_checkpoint,
    normalize_native_cfg_interval,
    sha256_file,
    validate_lifting_feature_metadata,
    validate_native_checkpoint,
)
from pose_point_depth_mv.native_ss_occupancy import (
    coords_from_logits,
    logit_quantiles,
    target_occupancy_grid,
)


REPORT_VERSION = "pose_point_depth_mv.native_ss_decoder_scale_diagnostic.v1"
METRIC_NAMES = ("iou_gain", "precision_gain", "recall_gain", "latent_mse_gain")


def parse_csv(value: str, cast) -> list[Any]:
    result = [cast(item.strip()) for item in str(value).split(",") if item.strip()]
    if not result or len(result) != len(set(result)):
        raise ValueError("CSV values must be non-empty and unique")
    return result


def object_indices(rows: list[dict[str, Any]], max_objects: int) -> list[int]:
    first: dict[str, int] = {}
    for index, row in enumerate(rows):
        first.setdefault(str(row.get("object_uid", row["uid"])), index)
    selected = [index for _, index in sorted(first.items())]
    return selected[: int(max_objects)] if int(max_objects) > 0 else selected


def summarize_object_records(
    rows: list[dict[str, Any]],
    *,
    bootstrap_samples: int,
    seed_offset: int,
) -> dict[str, Any]:
    by_object: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_object[str(row["object_uid"])].append(row)
    object_rows = [
        {
            "object_uid": object_uid,
            **{
                name: float(np.mean([float(item[name]) for item in values]))
                for name in METRIC_NAMES
            },
            "full_minus_stock_count": float(
                np.mean([float(item["full_minus_stock_count"]) for item in values])
            ),
            "full_stock_count_ratio": float(
                np.mean([float(item["full_stock_count_ratio"]) for item in values])
            ),
        }
        for object_uid, values in sorted(by_object.items())
    ]
    summary = {}
    for position, name in enumerate(METRIC_NAMES):
        values = [float(row[name]) for row in object_rows]
        summary[name] = {
            **summarize(values),
            "positive_rate": positive_rate(values),
            "bootstrap_mean_95_ci": bootstrap_mean_ci(
                values,
                samples=int(bootstrap_samples),
                seed=int(seed_offset) + position,
            ),
        }
    count_summary = {
        "full_minus_stock_count": summarize(
            [float(row["full_minus_stock_count"]) for row in object_rows]
        ),
        "full_stock_count_ratio": summarize(
            [float(row["full_stock_count_ratio"]) for row in object_rows]
        ),
    }
    return {
        "object_count": len(object_rows),
        "object_rows": object_rows,
        "summary": summary,
        "occupancy_count_summary": count_summary,
    }


def aggregate_logit_quantiles(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for branch in ("stock", "full"):
        result[branch] = {}
        groups = rows[0]["logit_quantiles"][branch]
        for group, quantiles in groups.items():
            result[branch][group] = {
                quantile: float(
                    np.mean(
                        [
                            float(row["logit_quantiles"][branch][group][quantile])
                            for row in rows
                        ]
                    )
                )
                for quantile in quantiles
            }
    return result


def threshold_key(value: float) -> str:
    return f"{float(value):+.6g}"


def validate_requested_deployment(
    args: argparse.Namespace, deployment: dict[str, Any]
) -> None:
    requested = {
        "steps": int(args.steps),
        "cfg_strength": float(args.cfg_strength),
        "cfg_interval": normalize_native_cfg_interval(
            float(item.strip())
            for item in str(args.cfg_interval).split(",")
            if item.strip()
        ),
        "guidance_rescale": float(args.guidance_rescale),
        "rescale_t": float(args.rescale_t),
        "condition_scale": float(args.condition_scale),
    }
    mismatch = {
        key: (requested[key], deployment[key])
        for key in requested
        if requested[key] != deployment[key]
    }
    if mismatch:
        raise ValueError(
            f"diagnostic protocol differs from checkpoint deployment={mismatch}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache_manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--pretrained", default="Stable-X/trellis-vggt-v0-2")
    parser.add_argument("--indices", default="all")
    parser.add_argument("--max_objects", type=int, default=16)
    parser.add_argument("--joint_seeds", default="42,43,44")
    parser.add_argument("--scale_multipliers", default="0.25,0.5,0.75,1.0")
    parser.add_argument(
        "--common_thresholds", default="-0.5,-0.25,-0.1,0,0.1,0.25,0.5"
    )
    parser.add_argument("--logit_quantiles", default="0.01,0.1,0.5,0.9,0.99")
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--cfg_strength", type=float, default=5.0)
    parser.add_argument("--cfg_interval", default="0.5,1.0")
    parser.add_argument("--guidance_rescale", type=float, default=0.0)
    parser.add_argument("--rescale_t", type=float, default=3.0)
    parser.add_argument("--condition_scale", type=float, default=1.0)
    parser.add_argument(
        "--amp_dtype", choices=("bf16", "fp16", "none"), default="bf16"
    )
    parser.add_argument("--bootstrap_samples", type=int, default=5000)
    parser.add_argument("--min_iou_gain_mean", type=float, default=0.0)
    parser.add_argument("--min_iou_win_rate", type=float, default=0.5)
    parser.add_argument("--min_recall_gain_mean", type=float, default=0.0)
    parser.add_argument("--min_latent_mse_gain_mean", type=float, default=0.0)
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    seeds = parse_csv(args.joint_seeds, int)
    scales = parse_csv(args.scale_multipliers, float)
    thresholds = parse_csv(args.common_thresholds, float)
    quantiles = parse_csv(args.logit_quantiles, float)
    if (
        any(value <= 0.0 for value in scales)
        or 0.0 not in thresholds
        or int(args.max_objects) <= 0
        or int(args.bootstrap_samples) <= 0
    ):
        raise ValueError(
            "scales/max_objects/bootstrap must be positive and thresholds include 0"
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    dataset = PoseLiftingCacheDataset(args.cache_manifest, indices=args.indices)
    selected = object_indices(dataset.rows, int(args.max_objects))
    if not selected:
        raise RuntimeError("Native SS decoder diagnostic selected no objects")

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    validate_native_checkpoint(
        checkpoint, expected_format=NATIVE_SS_FLOW_VERSION, pretrained=args.pretrained
    )
    deployment = native_ss_deployment_from_checkpoint(checkpoint)
    validate_requested_deployment(args, deployment)
    validate_lifting_feature_metadata(
        visual_feature_dim=dataset.visual_feature_dim,
        feature_metadata=dataset.feature_metadata,
        feature_source=str(checkpoint["args"]["feature_source"]),
    )

    device = torch.device("cuda")
    torch.cuda.set_device(0)
    sampler, model, decoder, _, defaults = build_native_ss_components(
        pretrained=args.pretrained,
        hidden_dim=int(checkpoint["args"]["hidden_dim"]),
        feature_source=str(checkpoint["args"]["feature_source"]),
        gradient_checkpointing=False,
        need_decoder=True,
        device=device,
    )
    if decoder is None:
        raise RuntimeError("decoder diagnostic requires frozen occupancy decoder")
    load_trainable_state_dict(model, checkpoint["model_trainable_state"])
    model.eval()
    decoder.eval()
    decoder_dtype = next(decoder.parameters()).dtype

    stock_params = dict(defaults)
    stock_params.update(
        {
            "steps": int(deployment["steps"]),
            "cfg_strength": float(deployment["cfg_strength"]),
            "cfg_interval": tuple(deployment["cfg_interval"]),
            "guidance_rescale": float(deployment["guidance_rescale"]),
            "rescale_t": float(deployment["rescale_t"]),
        }
    )
    full_params = dict(stock_params)
    full_params["cfg_strength"] = float(deployment["external_full_cfg_strength"])
    use_amp = args.amp_dtype != "none"
    amp_dtype = torch.float16 if args.amp_dtype == "fp16" else torch.bfloat16
    records_by_scale: dict[float, list[dict[str, Any]]] = {
        float(scale): [] for scale in scales
    }
    disabled_equivalence = None

    for object_position, index in enumerate(selected):
        sample = dataset[index]
        condition = sample["stock_condition"].to(device=device)
        negative = torch.zeros_like(condition)
        target_coords = sample["target_coords"].numpy()
        occupancy_target = target_occupancy_grid(
            sample["target_coords"], device=device
        )
        target_latent = sample["target"].to(device)[None]
        for seed in seeds:
            generator = torch.Generator(device=device).manual_seed(
                int(seed) * 1000003 + object_position * 1009
            )
            initial = torch.randn(
                (1, 8, 16, 16, 16), generator=generator, device=device
            )
            with torch.autocast(
                device_type="cuda", dtype=amp_dtype, enabled=use_amp
            ):
                stock_latent = sampler.sample(
                    NativeStockSSFlow(model),
                    initial.clone(),
                    cond=condition,
                    neg_cond=negative,
                    **stock_params,
                    verbose=False,
                ).samples
                stock_logits = decoder(
                    stock_latent.to(dtype=decoder_dtype)
                ).float()

            if disabled_equivalence is None:
                disabled_flow = PostCFGNativeSSRolloutFlow(
                    model,
                    condition,
                    negative,
                    sample,
                    cfg_strength=float(deployment["cfg_strength"]),
                    cfg_interval=deployment["cfg_interval"],
                    condition_scale=0.0,
                    delta_scale=float(deployment["delta_scale"]),
                    delta_rms_ratio_cap=float(deployment["delta_rms_ratio_cap"]),
                )
                with torch.autocast(
                    device_type="cuda", dtype=amp_dtype, enabled=use_amp
                ):
                    disabled_latent = sampler.sample(
                        disabled_flow,
                        initial.clone(),
                        cond=condition,
                        neg_cond=negative,
                        **full_params,
                        verbose=False,
                    ).samples
                disabled_equivalence = {
                    "uid": str(sample["uid"]),
                    "seed": int(seed),
                    "latent_max_abs": float(
                        (disabled_latent.float() - stock_latent.float())
                        .abs()
                        .amax()
                        .item()
                    ),
                    "wrapper": disabled_flow.stats_summary(),
                }
                disabled_equivalence["passed"] = bool(
                    disabled_equivalence["latent_max_abs"] == 0.0
                    and disabled_flow.calls == int(deployment["steps"])
                    and disabled_flow.active_calls
                    == len(deployment["active_timesteps"])
                )
                if not disabled_equivalence["passed"]:
                    raise RuntimeError(
                        "disabled diagnostic wrapper differs from Stock: "
                        f"{disabled_equivalence}"
                    )
                del disabled_latent, disabled_flow

            stock_threshold = {
                threshold_key(value): overlap_metrics(
                    coords_from_logits(stock_logits, threshold=value),
                    target_coords,
                )
                for value in thresholds
            }
            stock_quantiles = logit_quantiles(
                stock_logits, occupancy_target, quantiles
            )
            stock_latent_mse = torch.mean(
                (stock_latent.float() - target_latent.float()).square()
            ).item()

            for scale in scales:
                effective_scale = float(deployment["delta_scale"]) * float(scale)
                full_flow = PostCFGNativeSSRolloutFlow(
                    model,
                    condition,
                    negative,
                    sample,
                    cfg_strength=float(deployment["cfg_strength"]),
                    cfg_interval=deployment["cfg_interval"],
                    condition_scale=float(deployment["condition_scale"]),
                    delta_scale=effective_scale,
                    delta_rms_ratio_cap=float(deployment["delta_rms_ratio_cap"]),
                )
                with torch.autocast(
                    device_type="cuda", dtype=amp_dtype, enabled=use_amp
                ):
                    full_latent = sampler.sample(
                        full_flow,
                        initial.clone(),
                        cond=condition,
                        neg_cond=negative,
                        **full_params,
                        verbose=False,
                    ).samples
                    full_logits = decoder(
                        full_latent.to(dtype=decoder_dtype)
                    ).float()
                if (
                    full_flow.calls != int(deployment["steps"])
                    or full_flow.active_calls != len(deployment["active_timesteps"])
                ):
                    raise RuntimeError(
                        "diagnostic post-CFG wrapper call schedule differs"
                    )
                full_threshold = {
                    threshold_key(value): overlap_metrics(
                        coords_from_logits(full_logits, threshold=value),
                        target_coords,
                    )
                    for value in thresholds
                }
                zero_key = threshold_key(0.0)
                stock_zero = stock_threshold[zero_key]
                full_zero = full_threshold[zero_key]
                stock_count = int(stock_zero["coord_count"])
                full_count = int(full_zero["coord_count"])
                if stock_count <= 0:
                    raise RuntimeError("Stock decoder diagnostic produced no occupancy")
                record = {
                    "uid": str(sample["uid"]),
                    "object_uid": str(
                        sample.get("object_uid", sample["uid"])
                    ),
                    "seed": int(seed),
                    "scale_multiplier": float(scale),
                    "effective_delta_scale": effective_scale,
                    "same_initial_noise": True,
                    "target_count": int(len(target_coords)),
                    "stock_count": stock_count,
                    "full_count": full_count,
                    "full_minus_stock_count": full_count - stock_count,
                    "full_stock_count_ratio": float(full_count / stock_count),
                    "iou_gain": float(full_zero["iou"] - stock_zero["iou"]),
                    "precision_gain": float(
                        full_zero["precision"] - stock_zero["precision"]
                    ),
                    "recall_gain": float(
                        full_zero["recall"] - stock_zero["recall"]
                    ),
                    "latent_mse_gain": float(
                        stock_latent_mse
                        - torch.mean(
                            (full_latent.float() - target_latent.float()).square()
                        ).item()
                    ),
                    "threshold_sweep": {
                        key: {
                            "stock": stock_threshold[key],
                            "full": full_threshold[key],
                            "iou_gain": float(
                                full_threshold[key]["iou"]
                                - stock_threshold[key]["iou"]
                            ),
                            "precision_gain": float(
                                full_threshold[key]["precision"]
                                - stock_threshold[key]["precision"]
                            ),
                            "recall_gain": float(
                                full_threshold[key]["recall"]
                                - stock_threshold[key]["recall"]
                            ),
                        }
                        for key in stock_threshold
                    },
                    "logit_quantiles": {
                        "stock": stock_quantiles,
                        "full": logit_quantiles(
                            full_logits, occupancy_target, quantiles
                        ),
                    },
                    "guided_stats_summary": full_flow.stats_summary(),
                }
                records_by_scale[float(scale)].append(record)
                del full_latent, full_logits, full_flow
            print(
                f"[native_ss_decoder_diag] "
                f"{object_position + 1}/{len(selected)} seed={seed} "
                f"uid={sample['uid']} scales={scales}",
                flush=True,
            )
            del stock_latent, stock_logits, initial
            torch.cuda.empty_cache()
        del target_latent, occupancy_target

    scale_reports = {}
    for scale_position, scale in enumerate(scales):
        rows = records_by_scale[float(scale)]
        aggregated = summarize_object_records(
            rows,
            bootstrap_samples=int(args.bootstrap_samples),
            seed_offset=92000 + scale_position * 100,
        )
        threshold_report = {}
        for threshold in thresholds:
            key = threshold_key(threshold)
            threshold_rows = [
                {
                    **row,
                    "iou_gain": row["threshold_sweep"][key]["iou_gain"],
                    "precision_gain": row["threshold_sweep"][key][
                        "precision_gain"
                    ],
                    "recall_gain": row["threshold_sweep"][key]["recall_gain"],
                }
                for row in rows
            ]
            threshold_report[key] = summarize_object_records(
                threshold_rows,
                bootstrap_samples=int(args.bootstrap_samples),
                seed_offset=93000 + scale_position * 1000,
            )["summary"]
        summary = aggregated["summary"]
        checks = {
            "iou_gain_mean": summary["iou_gain"]["mean"]
            > float(args.min_iou_gain_mean),
            "iou_object_win_rate": summary["iou_gain"]["positive_rate"]
            >= float(args.min_iou_win_rate),
            "recall_gain_mean": summary["recall_gain"]["mean"]
            >= float(args.min_recall_gain_mean),
            "latent_mse_gain_mean": summary["latent_mse_gain"]["mean"]
            >= float(args.min_latent_mse_gain_mean),
            "disabled_wrapper_stock_equivalence": bool(
                disabled_equivalence and disabled_equivalence["passed"]
            ),
        }
        scale_reports[str(scale)] = {
            "scale_multiplier": float(scale),
            "effective_delta_scale": float(deployment["delta_scale"])
            * float(scale),
            **aggregated,
            "common_threshold_summary": threshold_report,
            "mean_logit_quantiles": aggregate_logit_quantiles(rows),
            "checks_at_native_threshold_zero": checks,
            "development_candidate": all(checks.values()),
            "records": rows,
        }

    candidates = [
        float(key)
        for key, value in scale_reports.items()
        if value["development_candidate"]
    ]
    recommended = (
        max(
            candidates,
            key=lambda value: scale_reports[str(value)]["summary"]["iou_gain"][
                "mean"
            ],
        )
        if candidates
        else None
    )
    report = {
        "format": REPORT_VERSION,
        "status": "complete",
        "formal_pass": False,
        "development_only": True,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "checkpoint_step": int(checkpoint.get("step", -1)),
        "cache_manifest": str(Path(args.cache_manifest).resolve()),
        "cache_manifest_sha256": sha256_file(args.cache_manifest),
        "object_count": len(selected),
        "joint_seeds": seeds,
        "scale_multipliers": scales,
        "common_thresholds": thresholds,
        "checkpoint_deployment": {
            **deployment,
            "cfg_interval": list(deployment["cfg_interval"]),
            "active_timesteps": list(deployment["active_timesteps"]),
        },
        "disabled_wrapper_stock_equivalence": disabled_equivalence,
        "scale_reports": scale_reports,
        "development_candidate_scales": candidates,
        "recommended_scale_multiplier": recommended,
        "scope_guard": (
            "Development diagnosis only. A selected scale requires a fresh "
            "64-object same-noise threshold-zero confirmation and cannot unlock "
            "SS1K, SLAT, or Mesh by itself."
        ),
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "Native SS frozen-decoder residual-scale diagnosis",
        "=" * 51,
        f"objects: {len(selected)}",
        f"seeds: {seeds}",
    ]
    for scale in scales:
        item = scale_reports[str(scale)]
        summary = item["summary"]
        lines.append(
            f"scale={scale:.3f} "
            f"iou={summary['iou_gain']['mean']:+.8f} "
            f"recall={summary['recall_gain']['mean']:+.8f} "
            f"latent={summary['latent_mse_gain']['mean']:+.8f} "
            f"win={summary['iou_gain']['positive_rate']:.6f} "
            f"candidate={item['development_candidate']}"
        )
    lines.extend(
        (
            f"recommended development scale: {recommended}",
            "formal PASS: False",
            "",
            report["scope_guard"],
        )
    )
    (output_dir / "summary.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
