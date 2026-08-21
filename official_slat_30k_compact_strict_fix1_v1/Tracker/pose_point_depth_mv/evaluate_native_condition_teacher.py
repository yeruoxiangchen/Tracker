#!/usr/bin/env python3
"""Teacher-forced checkpoint screening for native SS or SLAT conditioning."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from ar_ss_flow.local_pose_lifting_flow import PoseLiftingCacheDataset
from pose_point_depth_mv.direct_slat_flow import canonical_json_sha256
from pose_point_depth_mv.native_3d_condition import (
    NATIVE_SLAT_FLOW_VERSION,
    NATIVE_SS_FLOW_VERSION,
    NativeConditionSLatDataset,
    build_native_slat_components,
    build_native_ss_components,
    load_trainable_state_dict,
    native_ss_cfg_is_active,
    native_ss_deployment_from_checkpoint,
    sha256_file,
    validate_lifting_feature_metadata,
    validate_native_checkpoint,
)
from pose_point_depth_mv.train_direct_slat_flow import normalized_target, to_device_tree
from trellis.modules import sparse as sp


REPORT_VERSION = "pose_point_depth_mv.native_condition_teacher_eval.v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("ss", "slat"), required=True)
    parser.add_argument("--cache_manifest", required=True)
    parser.add_argument("--lifting_cache_manifest", default="")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--pretrained", default="Stable-X/trellis-vggt-v0-2")
    parser.add_argument("--indices", default="all")
    parser.add_argument("--max_objects", type=int, default=0)
    parser.add_argument("--t_values", default="0.2,0.5,0.8")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--condition_scale", type=float, default=1.0)
    parser.add_argument("--control_mode", default="pose_cyclic1")
    parser.add_argument("--max_slat_points", type=int, default=40960)
    parser.add_argument("--bootstrap_samples", type=int, default=5000)
    parser.add_argument("--min_gain_mean", type=float, default=0.0)
    parser.add_argument("--min_object_win_rate", type=float, default=0.5)
    parser.add_argument("--require_control_advantage", action="store_true")
    parser.add_argument("--amp_dtype", choices=("bf16", "fp16", "none"), default="bf16")
    return parser.parse_args()


def parse_t_values(value: str) -> tuple[float, ...]:
    result = tuple(float(item.strip()) for item in str(value).split(",") if item.strip())
    if not result or any(not 0 < item < 1 for item in result):
        raise ValueError("t_values must be finite values strictly within (0,1)")
    return result


def bootstrap_mean_ci(values: list[float], *, samples: int, seed: int) -> list[float]:
    if not values:
        return [float("nan"), float("nan")]
    array = np.asarray(values, dtype=np.float64)
    generator = np.random.default_rng(int(seed))
    means = np.empty(int(samples), dtype=np.float64)
    for start in range(0, int(samples), 512):
        count = min(512, int(samples) - start)
        positions = generator.integers(0, len(array), size=(count, len(array)))
        means[start : start + count] = array[positions].mean(axis=1)
    return [float(item) for item in np.quantile(means, (0.025, 0.975))]


def summarize(values: list[float], *, bootstrap_samples: int, seed: int) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "min": float(array.min()),
        "max": float(array.max()),
        "positive_rate": float((array > 0).mean()),
        "bootstrap_mean_95_ci": bootstrap_mean_ci(
            values, samples=bootstrap_samples, seed=seed
        ),
    }


def select_object_indices(rows: list[dict[str, Any]], max_objects: int) -> list[int]:
    by_object: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        object_uid = str(row.get("object_uid", row.get("uid", "")))
        by_object[object_uid].append(index)
    selected = [indices[0] for _, indices in sorted(by_object.items())]
    return selected[: int(max_objects)] if int(max_objects) > 0 else selected


@torch.no_grad()
def evaluate_ss(
    *,
    dataset: PoseLiftingCacheDataset,
    model: Any,
    sampler: Any,
    indices: list[int],
    t_values: tuple[float, ...],
    args: argparse.Namespace,
    device: torch.device,
    deployment: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    use_amp = args.amp_dtype != "none"
    amp_dtype = torch.float16 if args.amp_dtype == "fp16" else torch.bfloat16
    for position, index in enumerate(indices):
        sample = dataset[index]
        target = sample["target"].unsqueeze(0).to(device=device, dtype=torch.float32)
        condition = sample["stock_condition"].to(device=device)
        negative = torch.zeros_like(condition)
        generator = torch.Generator(device=device).manual_seed(
            int(args.seed) * 1000003 + position * 2017
        )
        noise = torch.randn(target.shape, generator=generator, device=device)
        step_rows = []
        for t_value_float in t_values:
            t_value = torch.tensor(t_value_float, device=device)
            x_t, gt_velocity = sampler._get_model_gt(target, t_value, noise)
            t = torch.full((1,), 1000.0 * t_value, device=device)
            cfg_active = native_ss_cfg_is_active(
                t_value_float, deployment["cfg_interval"]
            )
            stock_positive = model.stock_prediction(x_t, t, condition)
            stock_negative = (
                model.stock_prediction(x_t, t, negative) if cfg_active else None
            )
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                full, stock, full_stats = model.post_cfg_conditioned_prediction(
                    x_t,
                    t,
                    condition,
                    negative,
                    sample,
                    stock_positive_velocity=stock_positive,
                    stock_negative_velocity=stock_negative,
                    cfg_strength=float(deployment["cfg_strength"]),
                    cfg_active=cfg_active,
                    support_active=cfg_active,
                    condition_scale=float(deployment["condition_scale"]),
                    delta_scale=float(deployment["delta_scale"]),
                    delta_rms_ratio_cap=float(deployment["delta_rms_ratio_cap"]),
                )
                control, control_stock, _ = model.post_cfg_conditioned_prediction(
                    x_t,
                    t,
                    condition,
                    negative,
                    sample,
                    stock_positive_velocity=stock_positive,
                    stock_negative_velocity=stock_negative,
                    cfg_strength=float(deployment["cfg_strength"]),
                    cfg_active=cfg_active,
                    support_active=cfg_active,
                    condition_scale=float(deployment["condition_scale"]),
                    projection_mode=str(args.control_mode),
                    delta_scale=float(deployment["delta_scale"]),
                    delta_rms_ratio_cap=float(deployment["delta_rms_ratio_cap"]),
                )
            if not torch.equal(stock, control_stock):
                raise RuntimeError("teacher control changed the frozen Stock reference")
            stock_loss = F.mse_loss(stock.float(), gt_velocity.float())
            full_loss = F.mse_loss(full.float(), gt_velocity.float())
            control_loss = F.mse_loss(control.float(), gt_velocity.float())
            step_rows.append(
                {
                    "t": t_value_float,
                    "stock_loss": float(stock_loss.item()),
                    "full_loss": float(full_loss.item()),
                    "control_loss": float(control_loss.item()),
                    "gain_vs_stock": float((stock_loss - full_loss).item()),
                    "correct_over_control": float((control_loss - full_loss).item()),
                    "cfg_active": bool(cfg_active),
                    "raw_flow_delta_rms": float(
                        full_stats["raw_flow_delta_rms"].item()
                    ),
                    "effective_flow_delta_rms": float(
                        full_stats["effective_flow_delta_rms"].item()
                    ),
                    "effective_flow_delta_ratio": float(
                        full_stats["effective_flow_delta_ratio"].item()
                    ),
                    "delta_clip_scale": float(
                        full_stats["delta_clip_scale"].item()
                    ),
                }
            )
        rows.append(
            {
                "uid": str(sample["uid"]),
                "object_uid": str(sample.get("object_uid", sample["uid"])),
                "steps": step_rows,
                "gain_vs_stock": float(np.mean([row["gain_vs_stock"] for row in step_rows])),
                "correct_over_control": float(
                    np.mean([row["correct_over_control"] for row in step_rows])
                ),
            }
        )
        print(f"[native_teacher:ss] {position + 1}/{len(indices)} {sample['uid']}", flush=True)
    return rows


@torch.no_grad()
def evaluate_slat(
    *,
    dataset: NativeConditionSLatDataset,
    model: Any,
    sampler: Any,
    normalization: dict[str, Any],
    indices: list[int],
    t_values: tuple[float, ...],
    args: argparse.Namespace,
    device: torch.device,
) -> list[dict[str, Any]]:
    mean = torch.tensor(normalization["mean"], device=device)[None]
    std = torch.tensor(normalization["std"], device=device)[None]
    rows = []
    use_amp = args.amp_dtype != "none"
    amp_dtype = torch.float16 if args.amp_dtype == "fp16" else torch.bfloat16
    for position, index in enumerate(indices):
        sample = dataset[index]
        target = normalized_target(
            sample,
            mean=mean,
            std=std,
            device=device,
            max_points=int(args.max_slat_points),
            selection_seed=int(args.seed) * 1000003 + position * 2017,
        )
        condition = to_device_tree(sample["condition"]["cond"], device)
        generator = torch.Generator(device=device).manual_seed(
            int(args.seed) * 2000003 + position * 2017
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
        step_rows = []
        for t_value_float in t_values:
            t_value = torch.tensor(t_value_float, device=device)
            x_t, gt_velocity = sampler._get_model_gt(target, t_value, noise)
            t = torch.full((1,), 1000.0 * t_value, device=device)
            stock = model.stock_prediction(x_t, t, condition)
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                full, _ = model.conditioned_prediction(
                    x_t,
                    t,
                    condition,
                    sample["lifting_sample"],
                    stock_velocity=stock,
                    condition_scale=float(args.condition_scale),
                )
                control, _ = model.conditioned_prediction(
                    x_t,
                    t,
                    condition,
                    sample["lifting_sample"],
                    stock_velocity=stock,
                    condition_scale=float(args.condition_scale),
                    projection_mode=str(args.control_mode),
                )
            stock_loss = F.mse_loss(stock.feats.float(), gt_velocity.feats.float())
            full_loss = F.mse_loss(full.feats.float(), gt_velocity.feats.float())
            control_loss = F.mse_loss(control.feats.float(), gt_velocity.feats.float())
            step_rows.append(
                {
                    "t": t_value_float,
                    "stock_loss": float(stock_loss.item()),
                    "full_loss": float(full_loss.item()),
                    "control_loss": float(control_loss.item()),
                    "gain_vs_stock": float((stock_loss - full_loss).item()),
                    "correct_over_control": float((control_loss - full_loss).item()),
                }
            )
        rows.append(
            {
                "uid": str(sample["uid"]),
                "object_uid": str(sample["object_uid"]),
                "support_seed": int(sample["support_seed"]),
                "steps": step_rows,
                "gain_vs_stock": float(np.mean([row["gain_vs_stock"] for row in step_rows])),
                "correct_over_control": float(
                    np.mean([row["correct_over_control"] for row in step_rows])
                ),
            }
        )
        print(f"[native_teacher:slat] {position + 1}/{len(indices)} {sample['uid']}", flush=True)
    return rows


def main() -> None:
    args = parse_args()
    t_values = parse_t_values(args.t_values)
    if int(args.bootstrap_samples) <= 0:
        raise ValueError("bootstrap_samples must be positive")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    device = torch.device("cuda")
    torch.cuda.set_device(0)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    ss_deployment: dict[str, Any] = {}
    if args.stage == "ss":
        validate_native_checkpoint(
            checkpoint, expected_format=NATIVE_SS_FLOW_VERSION, pretrained=args.pretrained
        )
        dataset: Any = PoseLiftingCacheDataset(args.cache_manifest, indices=args.indices)
        validate_lifting_feature_metadata(
            visual_feature_dim=dataset.visual_feature_dim,
            feature_metadata=dataset.feature_metadata,
            feature_source=str(checkpoint["args"]["feature_source"]),
        )
        ss_deployment = native_ss_deployment_from_checkpoint(checkpoint)
        if float(args.condition_scale) != float(ss_deployment["condition_scale"]):
            raise ValueError(
                "SS teacher condition_scale must match the checkpoint-bound deployment"
            )
        sampler, model, _, model_summary, _ = build_native_ss_components(
            pretrained=args.pretrained,
            hidden_dim=int(checkpoint["args"]["hidden_dim"]),
            feature_source=str(checkpoint["args"]["feature_source"]),
            gradient_checkpointing=False,
            need_decoder=False,
            device=device,
        )
        load_trainable_state_dict(model, checkpoint["model_trainable_state"])
        model.eval()
        indices = select_object_indices(dataset.rows, int(args.max_objects))
        rows = evaluate_ss(
            dataset=dataset,
            model=model,
            sampler=sampler,
            indices=indices,
            t_values=t_values,
            args=args,
            device=device,
            deployment=ss_deployment,
        )
        normalization_hash = ""
    else:
        if not args.lifting_cache_manifest:
            raise ValueError("SLAT evaluation requires --lifting_cache_manifest")
        validate_native_checkpoint(
            checkpoint,
            expected_format=NATIVE_SLAT_FLOW_VERSION,
            pretrained=args.pretrained,
        )
        dataset = NativeConditionSLatDataset(
            args.cache_manifest,
            args.lifting_cache_manifest,
            indices=args.indices,
        )
        validate_lifting_feature_metadata(
            visual_feature_dim=dataset.lifting.visual_feature_dim,
            feature_metadata=dataset.lifting.feature_metadata,
            feature_source=str(checkpoint["args"]["feature_source"]),
        )
        sampler, model, _, normalization, model_summary = build_native_slat_components(
            pretrained=args.pretrained,
            hidden_dim=int(checkpoint["args"]["hidden_dim"]),
            feature_source=str(checkpoint["args"]["feature_source"]),
            gradient_checkpointing=False,
            device=device,
        )
        runtime_normalization = {
            key: [float(item) for item in value] for key, value in normalization.items()
        }
        normalization_hash = canonical_json_sha256(runtime_normalization)
        if normalization_hash != dataset.slat_normalization_hash:
            raise RuntimeError("runtime SLAT normalization differs from evaluation cache")
        load_trainable_state_dict(model, checkpoint["model_trainable_state"])
        model.eval()
        indices = select_object_indices(dataset.rows, int(args.max_objects))
        rows = evaluate_slat(
            dataset=dataset,
            model=model,
            sampler=sampler,
            normalization=runtime_normalization,
            indices=indices,
            t_values=t_values,
            args=args,
            device=device,
        )
    if not rows:
        raise RuntimeError("teacher evaluation selected no objects")
    gain_summary = summarize(
        [float(row["gain_vs_stock"]) for row in rows],
        bootstrap_samples=int(args.bootstrap_samples),
        seed=int(args.seed) + 701,
    )
    control_summary = summarize(
        [float(row["correct_over_control"]) for row in rows],
        bootstrap_samples=int(args.bootstrap_samples),
        seed=int(args.seed) + 709,
    )
    checks = {
        "gain_mean": gain_summary["mean"] >= float(args.min_gain_mean),
        "object_win_rate": gain_summary["positive_rate"]
        >= float(args.min_object_win_rate),
        "control_advantage": (
            control_summary["mean"] > 0 and control_summary["positive_rate"] >= 0.5
        )
        if args.require_control_advantage
        else True,
    }
    report = {
        "format": REPORT_VERSION,
        "stage": str(args.stage),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "checkpoint_step": int(checkpoint.get("step", -1)),
        "cache_manifest": str(Path(args.cache_manifest).resolve()),
        "cache_manifest_sha256": sha256_file(args.cache_manifest),
        "lifting_cache_manifest": (
            str(Path(args.lifting_cache_manifest).resolve())
            if args.lifting_cache_manifest
            else ""
        ),
        "object_count": len(rows),
        "t_values": list(t_values),
        "ss_deployment": (
            {
                **ss_deployment,
                "cfg_interval": list(ss_deployment["cfg_interval"]),
                "active_timesteps": list(ss_deployment["active_timesteps"]),
            }
            if args.stage == "ss"
            else {}
        ),
        "gain_vs_stock": gain_summary,
        "correct_over_control": control_summary,
        "normalization_hash": normalization_hash,
        "checks": checks,
        "passed": all(checks.values()),
        "rows": rows,
        "scope_guard": (
            "teacher-forced evidence selects checkpoints only; native-schedule "
            "rollout and same-noise Mesh remain required"
        ),
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    summary_lines = [
        f"Native {args.stage.upper()} teacher-forced evaluation",
        "=" * 44,
        f"objects: {len(rows)}",
        f"checkpoint step: {report['checkpoint_step']}",
        (
            "gain vs stock: "
            f"mean={gain_summary['mean']:+.8f} median={gain_summary['median']:+.8f} "
            f"win={gain_summary['positive_rate']:.6f} "
            f"CI={gain_summary['bootstrap_mean_95_ci']}"
        ),
        (
            "correct over control: "
            f"mean={control_summary['mean']:+.8f} "
            f"median={control_summary['median']:+.8f} "
            f"win={control_summary['positive_rate']:.6f}"
        ),
        f"PASS: {report['passed']}",
        "",
        report["scope_guard"],
    ]
    (output_dir / "summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print("\n".join(summary_lines), flush=True)
    raise SystemExit(0 if report["passed"] else 2)


if __name__ == "__main__":
    main()
