#!/usr/bin/env python3
"""Calibrate or evaluate Direct-SS/GenReCon Native SS with same-noise rollout."""

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
from pose_point_depth_mv.native_ss_genrecon import (
    NATIVE_SS_GENRECON_CALIBRATION,
    NATIVE_SS_GENRECON_EVAL,
    NativeSSCalibratedCFGFlow,
    NativeSSStockFlow,
    build_native_ss_genrecon_components,
    canonical_json_sha256,
    load_trainable_state_dict,
    require_disjoint_object_uids,
    select_object_indices,
    sha256_file,
    validate_genrecon_cache_contract,
    validate_native_ss_genrecon_checkpoint,
)


METRICS = ("iou_gain", "precision_gain", "recall_gain", "latent_mse_gain")


def parse_csv(value: str, cast) -> list[Any]:
    result = [cast(item.strip()) for item in str(value).split(",") if item.strip()]
    if not result or len(result) != len(set(result)):
        raise ValueError("CSV values must be non-empty and unique")
    return result


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("calibrate", "evaluate"), required=True)
    parser.add_argument("--cache_manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--calibration", default="")
    parser.add_argument("--pretrained", default="Stable-X/trellis-vggt-v0-2")
    parser.add_argument("--indices", default="all")
    parser.add_argument("--object_start", type=int, default=0)
    parser.add_argument("--object_end", type=int, default=0)
    parser.add_argument("--joint_seeds", default="42,43,44")
    parser.add_argument("--candidate_cfg_strengths", default="1,3,5")
    parser.add_argument("--weights", choices=("ema", "raw"), default="ema")
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--cfg_interval", default="0.5,1.0")
    parser.add_argument("--guidance_rescale", type=float, default=0.0)
    parser.add_argument("--rescale_t", type=float, default=3.0)
    parser.add_argument("--amp_dtype", choices=("bf16", "fp16", "none"), default="bf16")
    parser.add_argument("--bootstrap_samples", type=int, default=5000)
    parser.add_argument("--min_iou_gain_mean", type=float, default=0.0)
    parser.add_argument("--min_iou_win_rate", type=float, default=0.5)
    parser.add_argument("--min_recall_gain_mean", type=float, default=0.0)
    parser.add_argument("--min_latent_mse_gain_mean", type=float, default=0.0)
    parser.add_argument("--min_count_ratio", type=float, default=0.85)
    parser.add_argument("--max_count_ratio", type=float, default=1.20)
    parser.add_argument("--min_pose_control_iou_advantage", type=float, default=0.0)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if int(args.steps) <= 0 or int(args.bootstrap_samples) <= 0:
        raise ValueError("steps/bootstrap_samples must be positive")
    if int(args.object_start) < 0 or (
        int(args.object_end) > 0 and int(args.object_end) <= int(args.object_start)
    ):
        raise ValueError("invalid object slice")
    interval = parse_csv(args.cfg_interval, float)
    if len(interval) != 2 or not 0.0 <= interval[0] <= interval[1] <= 1.0:
        raise ValueError("cfg_interval must be two ordered values in [0,1]")
    if not 0.0 < float(args.min_count_ratio) <= float(args.max_count_ratio):
        raise ValueError("invalid count-ratio interval")
    if args.mode == "evaluate" and not args.calibration:
        raise ValueError("evaluate mode requires --calibration")


def sampling_params(
    defaults: dict[str, Any], args: argparse.Namespace, cfg_strength: float
) -> dict[str, Any]:
    params = dict(defaults)
    params.update(
        {
            "steps": int(args.steps),
            "cfg_strength": float(cfg_strength),
            "cfg_interval": tuple(parse_csv(args.cfg_interval, float)),
            "guidance_rescale": float(args.guidance_rescale),
            "rescale_t": float(args.rescale_t),
        }
    )
    return params


def aggregate_records(
    records: list[dict[str, Any]], *, bootstrap_samples: int, seed: int
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    by_object: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        by_object[str(row["object_uid"])].append(row)
    object_rows = []
    for object_uid, values in sorted(by_object.items()):
        count_ratios = [
            float(item["full_stock_count_ratio"])
            for item in values
            if item["full_stock_count_ratio"] is not None
            and np.isfinite(float(item["full_stock_count_ratio"]))
        ]
        row = {
            "object_uid": object_uid,
            **{
                name: float(np.mean([float(item[name]) for item in values]))
                for name in METRICS
            },
            "full_stock_count_ratio": (
                float(np.mean(count_ratios)) if count_ratios else None
            ),
            "full_stock_count_ratio_defined_seed_count": len(count_ratios),
            "stock_empty_seed_count": sum(
                int(item["stock_count"] <= 0) for item in values
            ),
            "full_minus_stock_count": float(
                np.mean([float(item["full_minus_stock_count"]) for item in values])
            ),
        }
        object_rows.append(row)
    summaries = {}
    for position, name in enumerate(METRICS):
        values = [float(row[name]) for row in object_rows]
        summaries[name] = {
            **summarize(values),
            "positive_rate": positive_rate(values),
            "bootstrap_mean_95_ci": bootstrap_mean_ci(
                values,
                samples=int(bootstrap_samples),
                seed=int(seed) + position,
            ),
        }
    object_count_ratios = [
        float(row["full_stock_count_ratio"])
        for row in object_rows
        if row["full_stock_count_ratio"] is not None
        and np.isfinite(float(row["full_stock_count_ratio"]))
    ]
    stock_empty_record_count = sum(int(row["stock_count"] <= 0) for row in records)
    count_summary = {
        "full_stock_count_ratio": summarize(object_count_ratios),
        "full_stock_count_ratio_defined_object_count": len(object_count_ratios),
        "full_stock_count_ratio_total_object_count": len(object_rows),
        "stock_empty_record_count": stock_empty_record_count,
        "stock_empty_record_rate": float(stock_empty_record_count / max(len(records), 1)),
        "full_minus_stock_count": summarize(
            [float(row["full_minus_stock_count"]) for row in object_rows]
        ),
    }
    return object_rows, summaries, count_summary


@torch.no_grad()
def run_candidate(
    *,
    dataset: PoseLiftingCacheDataset,
    selected: list[int],
    seeds: list[int],
    model,
    decoder,
    model_sampler,
    sampler_defaults: dict[str, Any],
    args: argparse.Namespace,
    cfg_strength: float,
    projection_mode: str,
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
    audit_disabled: bool,
    baseline_cache: dict[tuple[int, int, float], dict[str, Any]],
) -> dict[str, Any]:
    params = sampling_params(sampler_defaults, args, cfg_strength)
    records = []
    disabled_audit = None
    for object_position, index in enumerate(selected):
        sample = dataset[index]
        positive = sample["stock_condition"].to(device=device)
        negative = torch.zeros_like(positive)
        target_coords = sample["target_coords"].numpy()
        target_latent = sample["target"].to(device=device)[None]
        for seed in seeds:
            cache_key = (int(index), int(seed), float(cfg_strength))
            generator = torch.Generator(device=device).manual_seed(
                int(seed) * 1000003 + int(index) * 1009
            )
            initial = torch.randn(
                (1, 8, 16, 16, 16), generator=generator, device=device
            )
            cached = baseline_cache.get(cache_key)
            if cached is None:
                with torch.autocast(
                    device_type="cuda", dtype=amp_dtype, enabled=use_amp
                ):
                    stock_latent = model_sampler.sample(
                        NativeSSStockFlow(model),
                        initial.clone(),
                        cond=positive,
                        neg_cond=negative,
                        **params,
                        verbose=False,
                    ).samples
                stock_coords = decode_coords(decoder, stock_latent)
                stock_metrics = overlap_metrics(stock_coords, target_coords)
                stock_count = int(len(stock_coords))
                if stock_count <= 0:
                    print(
                        f"[native_ss_{args.mode}:empty_stock] mode={projection_mode} "
                        f"cfg={cfg_strength:g} weights={args.weights} "
                        f"seed={seed} uid={sample['uid']}",
                        flush=True,
                    )
                baseline_cache[cache_key] = {
                    "latent": stock_latent.detach().cpu(),
                    "metrics": stock_metrics,
                    "count": stock_count,
                }
            else:
                stock_latent = cached["latent"].to(device=device)
                stock_metrics = dict(cached["metrics"])
                stock_count = int(cached["count"])
            if audit_disabled and disabled_audit is None:
                disabled_flow = NativeSSCalibratedCFGFlow(
                    model,
                    positive,
                    sample,
                    enabled=False,
                )
                with torch.autocast(
                    device_type="cuda", dtype=amp_dtype, enabled=use_amp
                ):
                    disabled_latent = model_sampler.sample(
                        disabled_flow,
                        initial.clone(),
                        cond=positive,
                        neg_cond=negative,
                        **params,
                        verbose=False,
                    ).samples
                disabled_audit = {
                    "uid": str(sample["uid"]),
                    "seed": int(seed),
                    "latent_max_abs": float(
                        (disabled_latent.float() - stock_latent.float())
                        .abs()
                        .amax()
                        .item()
                    ),
                    "wrapper": disabled_flow.summary(),
                }
                disabled_audit["passed"] = disabled_audit["latent_max_abs"] == 0.0
                if not disabled_audit["passed"]:
                    raise RuntimeError(f"disabled Native SS differs from Stock: {disabled_audit}")
                del disabled_latent, disabled_flow
            full_flow = NativeSSCalibratedCFGFlow(
                model,
                positive,
                sample,
                enabled=True,
                projection_mode=str(projection_mode),
            )
            with torch.autocast(
                device_type="cuda", dtype=amp_dtype, enabled=use_amp
            ):
                full_latent = model_sampler.sample(
                    full_flow,
                    initial.clone(),
                    cond=positive,
                    neg_cond=negative,
                    **params,
                    verbose=False,
                ).samples
            if float(cfg_strength) != 1.0 and (
                full_flow.positive_calls == 0 or full_flow.negative_calls == 0
            ):
                raise RuntimeError("standard CFG did not call both adapted branches")
            full_coords = decode_coords(decoder, full_latent)
            full_metrics = overlap_metrics(full_coords, target_coords)
            full_count = int(len(full_coords))
            records.append(
                {
                    "uid": str(sample["uid"]),
                    "object_uid": str(sample.get("object_uid", sample["uid"])),
                    "seed": int(seed),
                    "projection_mode": str(projection_mode),
                    "same_initial_noise": True,
                    "stock": stock_metrics,
                    "full": full_metrics,
                    "stock_count": stock_count,
                    "full_count": full_count,
                    "full_minus_stock_count": full_count - stock_count,
                    "full_stock_count_ratio": (
                        float(full_count / stock_count) if stock_count > 0 else None
                    ),
                    "iou_gain": float(full_metrics["iou"] - stock_metrics["iou"]),
                    "precision_gain": float(
                        full_metrics["precision"] - stock_metrics["precision"]
                    ),
                    "recall_gain": float(full_metrics["recall"] - stock_metrics["recall"]),
                    "latent_mse_gain": float(
                        torch.mean((stock_latent.float() - target_latent).square()).item()
                        - torch.mean((full_latent.float() - target_latent).square()).item()
                    ),
                    "wrapper": full_flow.summary(),
                }
            )
            print(
                f"[native_ss_{args.mode}] mode={projection_mode} "
                f"cfg={cfg_strength:g} weights={args.weights} "
                f"{object_position + 1}/{len(selected)} seed={seed} uid={sample['uid']}",
                flush=True,
            )
            del initial, stock_latent, full_latent, full_flow
            torch.cuda.empty_cache()
    object_rows, summaries, count_summary = aggregate_records(
        records,
        bootstrap_samples=int(args.bootstrap_samples),
        seed=91000 + int(round(float(cfg_strength) * 100)),
    )
    return {
        "cfg_strength": float(cfg_strength),
        "condition_scale_policy": "learned_projection_only",
        "projection_mode": str(projection_mode),
        "sampling": params,
        "object_count": len(object_rows),
        "record_count": len(records),
        "summary": summaries,
        "count_summary": count_summary,
        "stock_empty_record_count": int(count_summary["stock_empty_record_count"]),
        "stock_empty_record_rate": float(count_summary["stock_empty_record_rate"]),
        "disabled_stock_equivalence": disabled_audit,
        "stock_baseline_cache_entries": len(baseline_cache),
        "object_rows": object_rows,
        "records": records,
    }


def candidate_checks(candidate: dict[str, Any], args: argparse.Namespace) -> dict[str, bool]:
    summary = candidate["summary"]
    count_ratio = candidate["count_summary"]["full_stock_count_ratio"]["mean"]
    return {
        "iou_gain_mean": summary["iou_gain"]["mean"] >= float(args.min_iou_gain_mean),
        "recall_gain_mean": summary["recall_gain"]["mean"]
        >= float(args.min_recall_gain_mean),
        "latent_mse_gain_mean": summary["latent_mse_gain"]["mean"]
        >= float(args.min_latent_mse_gain_mean),
        "count_ratio_lower": count_ratio >= float(args.min_count_ratio),
        "count_ratio_upper": count_ratio <= float(args.max_count_ratio),
        "stock_baseline_nonempty": int(candidate.get("stock_empty_record_count", 0))
        == 0,
    }


def load_runtime(args: argparse.Namespace, device: torch.device):
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    validate_native_ss_genrecon_checkpoint(checkpoint, pretrained=args.pretrained)
    saved = checkpoint["args"]
    model_sampler, model, decoder, model_summary, defaults = (
        build_native_ss_genrecon_components(
            pretrained=args.pretrained,
            lora_rank=int(saved["lora_rank"]),
            lora_alpha=int(saved["lora_alpha"]),
            condition_channels=int(saved["condition_channels"]),
            gradient_checkpointing=False,
            need_decoder=True,
            device=device,
        )
    )
    if decoder is None:
        raise RuntimeError("Native SS calibration/evaluation requires the frozen decoder")
    state_key = (
        "ema_trainable_state" if str(args.weights) == "ema" else "model_trainable_state"
    )
    load_trainable_state_dict(model, checkpoint[state_key])
    model.eval()
    decoder.eval()
    return checkpoint, model_sampler, model, decoder, model_summary, defaults


def calibration_protocol(
    args: argparse.Namespace,
    dataset: PoseLiftingCacheDataset,
    selected: list[int],
    checkpoint: dict[str, Any],
) -> dict[str, Any]:
    object_uids = sorted(
        {str(dataset.rows[index].get("object_uid", dataset.rows[index]["uid"])) for index in selected}
    )
    return {
        "cache_manifest": str(Path(args.cache_manifest).resolve()),
        "cache_manifest_sha256": sha256_file(args.cache_manifest),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "checkpoint_step": int(checkpoint["step"]),
        "object_start": int(args.object_start),
        "object_end": int(args.object_end),
        "object_uids": object_uids,
        "object_uid_hash": canonical_json_sha256(object_uids),
        "joint_seeds": parse_csv(args.joint_seeds, int),
        "steps": int(args.steps),
        "cfg_interval": parse_csv(args.cfg_interval, float),
        "guidance_rescale": float(args.guidance_rescale),
        "rescale_t": float(args.rescale_t),
        "amp_dtype": str(args.amp_dtype),
        "weights": str(args.weights),
        "condition_scale_policy": "learned_projection_only",
        "post_cfg_cap": False,
    }


def main() -> None:
    args = make_parser().parse_args()
    validate_args(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    dataset = PoseLiftingCacheDataset(args.cache_manifest, indices=args.indices)
    selected = select_object_indices(
        dataset.rows, start=int(args.object_start), end=int(args.object_end)
    )
    seeds = parse_csv(args.joint_seeds, int)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    checkpoint, model_sampler, model, decoder, model_summary, defaults = load_runtime(
        args, device
    )
    training_identity = checkpoint.get("data_identity", {})
    training_object_uids = training_identity.get("object_uids")
    if not isinstance(training_object_uids, list):
        raise RuntimeError(
            "Native SS checkpoint lacks training object UIDs required for held-out evaluation"
        )
    require_disjoint_object_uids(
        (
            str(dataset.rows[index].get("object_uid", dataset.rows[index]["uid"]))
            for index in selected
        ),
        training_object_uids,
    )
    cache_contract = validate_genrecon_cache_contract(
        dataset,
        training_config_hash=str(training_identity.get("config_hash", "")),
    )
    protocol = calibration_protocol(args, dataset, selected, checkpoint)
    protocol["cache_feature_contract"] = cache_contract
    use_amp = args.amp_dtype != "none"
    amp_dtype = torch.float16 if args.amp_dtype == "fp16" else torch.bfloat16

    if args.mode == "calibrate":
        cfg_values = parse_csv(args.candidate_cfg_strengths, float)
        if any(not np.isfinite(value) or value <= 0 for value in cfg_values):
            raise ValueError("CFG calibration candidates must be finite and positive")
        candidates = []
        baseline_cache: dict[tuple[int, int, float], dict[str, Any]] = {}
        for cfg_strength in cfg_values:
            candidate = run_candidate(
                dataset=dataset,
                selected=selected,
                seeds=seeds,
                model=model,
                decoder=decoder,
                model_sampler=model_sampler,
                sampler_defaults=defaults,
                args=args,
                cfg_strength=cfg_strength,
                projection_mode="correct",
                device=device,
                use_amp=use_amp,
                amp_dtype=amp_dtype,
                audit_disabled=not candidates,
                baseline_cache=baseline_cache,
            )
            candidate["checks"] = candidate_checks(candidate, args)
            candidate["eligible"] = all(candidate["checks"].values())
            candidates.append(candidate)
        eligible = [candidate for candidate in candidates if candidate["eligible"]]
        eligible.sort(
            key=lambda candidate: (
                float(candidate["summary"]["iou_gain"]["mean"]),
                float(candidate["summary"]["recall_gain"]["mean"]),
                float(candidate["summary"]["latent_mse_gain"]["mean"]),
                -abs(
                    float(candidate["count_summary"]["full_stock_count_ratio"]["mean"])
                    - 1.0
                ),
                -float(candidate["cfg_strength"]),
            ),
            reverse=True,
        )
        selected_candidate = eligible[0] if eligible else None
        report = {
            "format": NATIVE_SS_GENRECON_CALIBRATION,
            "passed": selected_candidate is not None,
            "protocol": protocol,
            "candidate_cfg_strengths": cfg_values,
            "selection_rule": (
                "eligible on IoU/recall/latent/count gates; lexicographic max "
                "IoU, recall, latent, count closeness, then lower CFG"
            ),
            "selected": (
                None
                if selected_candidate is None
                else {
                    "cfg_strength": selected_candidate["cfg_strength"],
                    "condition_scale_policy": "learned_projection_only",
                    "summary": selected_candidate["summary"],
                    "count_summary": selected_candidate["count_summary"],
                    "checks": selected_candidate["checks"],
                }
            ),
            "candidates": candidates,
            "model_summary": model_summary,
        }
        (output_dir / "calibration.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        (output_dir / "report.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        print(
            json.dumps(
                {"passed": report["passed"], "selected": report["selected"]},
                ensure_ascii=False,
            ),
            flush=True,
        )
        raise SystemExit(0 if report["passed"] else 2)

    calibration = json.loads(Path(args.calibration).read_text(encoding="utf-8"))
    if calibration.get("format") != NATIVE_SS_GENRECON_CALIBRATION or calibration.get("passed") is not True:
        raise ValueError("evaluation requires a passed Native SS calibration")
    calibrated_protocol = calibration["protocol"]
    bindings = {
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "cache_manifest_sha256": sha256_file(args.cache_manifest),
        "steps": int(args.steps),
        "cfg_interval": parse_csv(args.cfg_interval, float),
        "guidance_rescale": float(args.guidance_rescale),
        "rescale_t": float(args.rescale_t),
        "joint_seeds": parse_csv(args.joint_seeds, int),
        "amp_dtype": str(args.amp_dtype),
        "weights": str(args.weights),
        "condition_scale_policy": "learned_projection_only",
        "post_cfg_cap": False,
    }
    mismatch = {
        key: (calibrated_protocol.get(key), value)
        for key, value in bindings.items()
        if calibrated_protocol.get(key) != value
    }
    if mismatch:
        raise ValueError(f"evaluation differs from calibration protocol={mismatch}")
    overlap = sorted(set(protocol["object_uids"]).intersection(calibrated_protocol["object_uids"]))
    if overlap:
        raise RuntimeError(f"calibration/evaluation objects overlap: {overlap[:8]}")
    selected_parameters = calibration["selected"]
    baseline_cache: dict[tuple[int, int, float], dict[str, Any]] = {}
    correct = run_candidate(
        dataset=dataset,
        selected=selected,
        seeds=seeds,
        model=model,
        decoder=decoder,
        model_sampler=model_sampler,
        sampler_defaults=defaults,
        args=args,
        cfg_strength=float(selected_parameters["cfg_strength"]),
        projection_mode="correct",
        device=device,
        use_amp=use_amp,
        amp_dtype=amp_dtype,
        audit_disabled=True,
        baseline_cache=baseline_cache,
    )
    control = run_candidate(
        dataset=dataset,
        selected=selected,
        seeds=seeds,
        model=model,
        decoder=decoder,
        model_sampler=model_sampler,
        sampler_defaults=defaults,
        args=args,
        cfg_strength=float(selected_parameters["cfg_strength"]),
        projection_mode="pose_cyclic1",
        device=device,
        use_amp=use_amp,
        amp_dtype=amp_dtype,
        audit_disabled=False,
        baseline_cache=baseline_cache,
    )
    control_by_key = {
        (row["object_uid"], row["seed"]): row for row in control["records"]
    }
    pose_advantages = []
    for row in correct["records"]:
        other = control_by_key[(row["object_uid"], row["seed"])]
        pose_advantages.append(float(row["full"]["iou"] - other["full"]["iou"]))
    pose_by_object: dict[str, list[float]] = defaultdict(list)
    for row, advantage in zip(correct["records"], pose_advantages):
        pose_by_object[str(row["object_uid"])].append(advantage)
    pose_values = [float(np.mean(values)) for _, values in sorted(pose_by_object.items())]
    pose_summary = {
        **summarize(pose_values),
        "positive_rate": positive_rate(pose_values),
        "bootstrap_mean_95_ci": bootstrap_mean_ci(
            pose_values, samples=int(args.bootstrap_samples), seed=99001
        ),
    }
    checks = {
        **candidate_checks(correct, args),
        "iou_object_win_rate": correct["summary"]["iou_gain"]["positive_rate"]
        >= float(args.min_iou_win_rate),
        "pose_control_iou_advantage": pose_summary["mean"]
        > float(args.min_pose_control_iou_advantage),
        "disabled_stock_equivalence": bool(
            correct["disabled_stock_equivalence"]
            and correct["disabled_stock_equivalence"]["passed"]
        ),
    }
    report = {
        "format": NATIVE_SS_GENRECON_EVAL,
        "passed": all(checks.values()),
        "protocol": protocol,
        "calibration": str(Path(args.calibration).resolve()),
        "calibration_sha256": sha256_file(args.calibration),
        "calibrated_parameters": {
            "cfg_strength": float(selected_parameters["cfg_strength"]),
            "condition_scale_policy": "learned_projection_only",
            "post_cfg_cap": False,
        },
        "correct": correct,
        "pose_cyclic_control": control,
        "correct_over_pose_control_iou": pose_summary,
        "checks": checks,
        "scope_guard": "Native SS only; SLAT and Mesh remain locked until this gate passes",
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    summary_lines = [
        "Native SS calibrated same-noise evaluation",
        "=" * 42,
        f"objects: {correct['object_count']}",
        f"seeds: {seeds}",
        f"weights: {args.weights}",
        f"cfg_strength: {selected_parameters['cfg_strength']}",
        "condition_scale_policy: learned_projection_only",
        "post_cfg_cap: false",
    ]
    for name in METRICS:
        row = correct["summary"][name]
        summary_lines.append(
            f"{name}: mean={row['mean']:+.8f} median={row['median']:+.8f} "
            f"win={row['positive_rate']:.6f} CI={row['bootstrap_mean_95_ci']}"
        )
    summary_lines.extend(
        (
            f"correct_over_pose_control_iou: {pose_summary}",
            f"checks: {checks}",
            f"PASS: {report['passed']}",
            report["scope_guard"],
        )
    )
    (output_dir / "summary.txt").write_text(
        "\n".join(summary_lines) + "\n", encoding="utf-8"
    )
    print("\n".join(summary_lines), flush=True)
    raise SystemExit(0 if report["passed"] else 2)


if __name__ == "__main__":
    main()
