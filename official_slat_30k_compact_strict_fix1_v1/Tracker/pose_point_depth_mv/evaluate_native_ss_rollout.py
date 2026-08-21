#!/usr/bin/env python3
"""Same-noise native-schedule SS rollout and occupancy comparison."""

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
    decode_coords,
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


REPORT_VERSION = "pose_point_depth_mv.native_ss_rollout_eval.v2"


def parse_csv(value: str, cast) -> list[Any]:
    result = [cast(item.strip()) for item in str(value).split(",") if item.strip()]
    if not result or len(result) != len(set(result)):
        raise ValueError("CSV values must be non-empty and unique")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache_manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--pretrained", default="Stable-X/trellis-vggt-v0-2")
    parser.add_argument("--indices", default="all")
    parser.add_argument("--max_objects", type=int, default=64)
    parser.add_argument("--joint_seeds", default="42,43,44")
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--cfg_strength", type=float, default=5.0)
    parser.add_argument("--cfg_interval", default="0.5,1.0")
    parser.add_argument("--guidance_rescale", type=float, default=0.0)
    parser.add_argument("--rescale_t", type=float, default=3.0)
    parser.add_argument("--condition_scale", type=float, default=1.0)
    parser.add_argument("--amp_dtype", choices=("bf16", "fp16", "none"), default="bf16")
    parser.add_argument("--bootstrap_samples", type=int, default=5000)
    parser.add_argument("--min_iou_gain_mean", type=float, default=0.0)
    parser.add_argument("--min_iou_win_rate", type=float, default=0.5)
    parser.add_argument("--min_recall_gain_mean", type=float, default=0.0)
    parser.add_argument("--min_latent_mse_gain_mean", type=float, default=0.0)
    return parser.parse_args()


def object_indices(rows: list[dict[str, Any]], max_objects: int) -> list[int]:
    first: dict[str, int] = {}
    for index, row in enumerate(rows):
        first.setdefault(str(row.get("object_uid", row["uid"])), index)
    selected = [index for _, index in sorted(first.items())]
    return selected[: int(max_objects)] if int(max_objects) > 0 else selected


@torch.no_grad()
def main() -> None:
    args = parse_args()
    seeds = parse_csv(args.joint_seeds, int)
    if int(args.steps) <= 0 or int(args.bootstrap_samples) <= 0:
        raise ValueError("steps/bootstrap_samples must be positive")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    dataset = PoseLiftingCacheDataset(args.cache_manifest, indices=args.indices)
    selected = object_indices(dataset.rows, int(args.max_objects))
    if not selected:
        raise RuntimeError("SS rollout selected no objects")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    validate_native_checkpoint(
        checkpoint, expected_format=NATIVE_SS_FLOW_VERSION, pretrained=args.pretrained
    )
    deployment = native_ss_deployment_from_checkpoint(checkpoint)
    requested_interval = normalize_native_cfg_interval(
        float(item.strip())
        for item in str(args.cfg_interval).split(",")
        if item.strip()
    )
    requested = {
        "steps": int(args.steps),
        "cfg_strength": float(args.cfg_strength),
        "cfg_interval": requested_interval,
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
            f"rollout protocol differs from checkpoint-bound deployment={mismatch}"
        )
    validate_lifting_feature_metadata(
        visual_feature_dim=dataset.visual_feature_dim,
        feature_metadata=dataset.feature_metadata,
        feature_source=str(checkpoint["args"]["feature_source"]),
    )
    device = torch.device("cuda")
    torch.cuda.set_device(0)
    sampler, model, decoder, model_summary, defaults = build_native_ss_components(
        pretrained=args.pretrained,
        hidden_dim=int(checkpoint["args"]["hidden_dim"]),
        feature_source=str(checkpoint["args"]["feature_source"]),
        gradient_checkpointing=False,
        need_decoder=True,
        device=device,
    )
    if decoder is None:
        raise RuntimeError("native SS rollout requires frozen occupancy decoder")
    load_trainable_state_dict(model, checkpoint["model_trainable_state"])
    model.eval()
    decoder.eval()
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
    records = []
    disabled_equivalence: dict[str, Any] | None = None
    for object_position, index in enumerate(selected):
        sample = dataset[index]
        condition = sample["stock_condition"].to(device=device)
        negative = torch.zeros_like(condition)
        target_coords = sample["target_coords"].numpy()
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
                        "disabled post-CFG wrapper differs from Stock rollout: "
                        f"{disabled_equivalence}"
                    )
                del disabled_latent, disabled_flow
            full_flow = PostCFGNativeSSRolloutFlow(
                model,
                condition,
                negative,
                sample,
                cfg_strength=float(deployment["cfg_strength"]),
                cfg_interval=deployment["cfg_interval"],
                condition_scale=float(deployment["condition_scale"]),
                delta_scale=float(deployment["delta_scale"]),
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
            if (
                full_flow.calls != int(deployment["steps"])
                or full_flow.active_calls != len(deployment["active_timesteps"])
            ):
                raise RuntimeError(
                    "post-CFG wrapper call schedule differs from frozen deployment"
                )
            stock_coords = decode_coords(decoder, stock_latent)
            full_coords = decode_coords(decoder, full_latent)
            stock_metrics = overlap_metrics(stock_coords, target_coords)
            full_metrics = overlap_metrics(full_coords, target_coords)
            stock_count = int(len(stock_coords))
            full_count = int(len(full_coords))
            if stock_count <= 0:
                raise RuntimeError("Stock SS rollout decoded no occupied voxels")
            target_latent = sample["target"].to(device)[None]
            records.append(
                {
                    "uid": str(sample["uid"]),
                    "object_uid": str(sample.get("object_uid", sample["uid"])),
                    "seed": int(seed),
                    "same_initial_noise": True,
                    "stock": stock_metrics,
                    "full": full_metrics,
                    "target_count": int(len(target_coords)),
                    "stock_count": stock_count,
                    "full_count": full_count,
                    "full_minus_stock_count": full_count - stock_count,
                    "full_stock_count_ratio": (
                        float(full_count / stock_count) if stock_count else None
                    ),
                    "iou_gain": float(full_metrics["iou"] - stock_metrics["iou"]),
                    "precision_gain": float(
                        full_metrics["precision"] - stock_metrics["precision"]
                    ),
                    "recall_gain": float(full_metrics["recall"] - stock_metrics["recall"]),
                    "latent_mse_gain": float(
                        torch.mean(
                            (stock_latent.float() - target_latent).square()
                        ).item()
                        - torch.mean(
                            (full_latent.float() - target_latent).square()
                        ).item()
                    ),
                    "guided_stats_summary": full_flow.stats_summary(),
                    "guided_timestep_stats": full_flow.guided_stats,
                }
            )
            print(
                f"[native_ss_rollout] {object_position + 1}/{len(selected)} "
                f"seed={seed} uid={sample['uid']}",
                flush=True,
            )
            del stock_latent, full_latent, target_latent, full_flow, initial
            torch.cuda.empty_cache()
    by_object: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        by_object[str(row["object_uid"])].append(row)
    metric_names = ("iou_gain", "precision_gain", "recall_gain", "latent_mse_gain")
    object_rows = [
        {
            "object_uid": object_uid,
            **{
                name: float(np.mean([float(row[name]) for row in values]))
                for name in metric_names
            },
            "full_minus_stock_count": float(
                np.mean([float(row["full_minus_stock_count"]) for row in values])
            ),
            "full_stock_count_ratio": float(
                np.mean(
                    [
                        float(row["full_stock_count_ratio"])
                        for row in values
                        if row["full_stock_count_ratio"] is not None
                    ]
                )
            ),
        }
        for object_uid, values in sorted(by_object.items())
    ]
    summaries = {}
    for position, name in enumerate(metric_names):
        values = [float(row[name]) for row in object_rows]
        summaries[name] = {
            **summarize(values),
            "positive_rate": positive_rate(values),
            "bootstrap_mean_95_ci": bootstrap_mean_ci(
                values,
                samples=int(args.bootstrap_samples),
                seed=81000 + position,
            ),
        }
    checks = {
        "iou_gain_mean": summaries["iou_gain"]["mean"]
        > float(args.min_iou_gain_mean),
        "iou_object_win_rate": summaries["iou_gain"]["positive_rate"]
        >= float(args.min_iou_win_rate),
        "recall_gain_mean": summaries["recall_gain"]["mean"]
        >= float(args.min_recall_gain_mean),
        "latent_mse_gain_mean": summaries["latent_mse_gain"]["mean"]
        >= float(args.min_latent_mse_gain_mean),
        "disabled_wrapper_stock_equivalence": bool(
            disabled_equivalence and disabled_equivalence.get("passed")
        ),
    }
    occupancy_count_summary = {
        "full_minus_stock_count": summarize(
            [float(row["full_minus_stock_count"]) for row in object_rows]
        ),
        "full_stock_count_ratio": summarize(
            [float(row["full_stock_count_ratio"]) for row in object_rows]
        ),
    }
    report = {
        "format": REPORT_VERSION,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "checkpoint_step": int(checkpoint.get("step", -1)),
        "cache_manifest": str(Path(args.cache_manifest).resolve()),
        "cache_manifest_sha256": sha256_file(args.cache_manifest),
        "object_count": len(object_rows),
        "record_count": len(records),
        "joint_seeds": seeds,
        "checkpoint_deployment": {
            **deployment,
            "cfg_interval": list(deployment["cfg_interval"]),
            "active_timesteps": list(deployment["active_timesteps"]),
        },
        "stock_sampling": stock_params,
        "full_external_sampling": full_params,
        "disabled_wrapper_stock_equivalence": disabled_equivalence,
        "summary": summaries,
        "occupancy_count_summary": occupancy_count_summary,
        "checks": checks,
        "passed": all(checks.values()),
        "object_rows": object_rows,
        "records": records,
        "scope_guard": "SS rollout/occupancy gate; downstream SLAT and Mesh remain required",
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "Native SS same-noise rollout/occupancy evaluation",
        "=" * 52,
        f"objects: {len(object_rows)}",
        f"seeds: {seeds}",
    ]
    for name in metric_names:
        row = summaries[name]
        lines.append(
            f"{name}: mean={row['mean']:+.8f} median={row['median']:+.8f} "
            f"win={row['positive_rate']:.6f} CI={row['bootstrap_mean_95_ci']}"
        )
    lines.extend((f"PASS: {report['passed']}", "", report["scope_guard"]))
    lines.insert(
        -2,
        "full_stock_count_ratio: "
        f"mean={occupancy_count_summary['full_stock_count_ratio']['mean']:.8f}",
    )
    lines.insert(-2, f"checks: {checks}")
    (output_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)
    raise SystemExit(0 if report["passed"] else 2)


if __name__ == "__main__":
    main()
