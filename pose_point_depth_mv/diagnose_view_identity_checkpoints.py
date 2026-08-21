#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("SPCONV_ALGO", "native")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
os.environ.setdefault("TORCH_EXTENSIONS_DIR", "/tmp/torch_extensions")

import torch
import torch.nn.functional as F

from ar_ss_flow.local_pose_lifting_flow import PoseLiftingCacheDataset
from pose_point_depth_mv.eval_local_target_probe import (
    deterministic_noise_seed,
    object_balanced,
    parse_csv_floats,
    parse_csv_ints,
    summarize,
)
from pose_point_depth_mv.train_local_target_probe import (
    build_frozen_stock_flow,
    masked_mse,
)
from pose_point_depth_mv.view_identity_lifting import (
    VIEW_IDENTITY_ABLATION_MODES,
    VIEW_IDENTITY_CHECKPOINT_VERSION,
    VIEW_IDENTITY_CONTROL_NAMES,
    ViewIdentityPoseDepthProbe,
    build_view_identity_evidence,
    load_view_identity_probe_state,
    make_null_view_identity_evidence,
    protocol_hash,
)


LOSS_CONFIG_KEYS = (
    "local_target_weight",
    "flow_weight",
    "corrupt_zero_weight",
    "gain_weight",
    "gain_margin",
    "rank_weight",
    "rank_margin",
    "delta_norm_weight",
)


def parse_csv_strings(value: str) -> list[str]:
    result = [item.strip() for item in value.split(",") if item.strip()]
    if not result:
        raise ValueError("at least one string value is required")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Shared-stock checkpoint trajectory and controlled ablations for "
            "view-identity pose-guided lifting."
        )
    )
    parser.add_argument("--cache_manifest", required=True)
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--checkpoint_steps", default="25,50,75,100")
    parser.add_argument("--ablation_steps", default="100")
    parser.add_argument(
        "--ablation_modes", default=",".join(VIEW_IDENTITY_ABLATION_MODES)
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--pretrained", default="Stable-X/trellis-vggt-v0-2")
    parser.add_argument("--train_indices", default="0-15")
    parser.add_argument("--fresh_indices", default="16-63")
    parser.add_argument("--max_samples_per_split", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seeds", default="48,49,50")
    parser.add_argument("--t_values", default="0.1,0.3,0.5,0.7,0.9")
    parser.add_argument("--physical_scale", type=float, default=1.0)
    parser.add_argument("--bootstrap_samples", type=int, default=10000)
    parser.add_argument("--min_object_win_rate", type=float, default=0.65)
    parser.add_argument("--min_positive_t_count", type=int, default=4)
    parser.add_argument("--reference_train_report")
    parser.add_argument("--reference_fresh_report")
    parser.add_argument("--max_reference_metric_diff", type=float, default=1.0e-7)
    return parser.parse_args()


def sha256_strings(values: list[str]) -> str:
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def checkpoint_schema(checkpoint: dict[str, Any]) -> dict[str, Any]:
    args = checkpoint.get("args", {})
    summary = checkpoint.get("model_summary", {})
    return {
        "pretrained": args.get("pretrained"),
        "amp_dtype": args.get("amp_dtype"),
        "hidden_dim": args.get("hidden_dim"),
        "pair_dim": args.get("pair_dim"),
        "min_views": args.get("min_views"),
        "physical_scale": args.get("physical_scale"),
        "training_seed": args.get("seed"),
        "loss_config": {key: args.get(key) for key in LOSS_CONFIG_KEYS},
        "cache_config_hash": summary.get("cache_config_hash"),
        "train_object_uids": sorted(
            str(value) for value in summary.get("train_object_uids", ())
        ),
        "control_names": list(summary.get("control_names", ())),
        "probe": summary.get("probe"),
        "flow": summary.get("flow"),
    }


def load_probes(
    args: argparse.Namespace,
    *,
    visual_channels: int,
    cache_hash: str,
    device: torch.device,
) -> tuple[dict[int, ViewIdentityPoseDepthProbe], dict[str, Any]]:
    steps = parse_csv_ints(args.checkpoint_steps)
    schemas: list[dict[str, Any]] = []
    probes: dict[int, ViewIdentityPoseDepthProbe] = {}
    for step in steps:
        path = Path(args.checkpoint_dir) / f"step_{step:06d}.pt"
        if not path.is_file():
            raise FileNotFoundError(path)
        checkpoint = torch.load(path, map_location="cpu")
        if checkpoint.get("format") != VIEW_IDENTITY_CHECKPOINT_VERSION:
            raise ValueError(f"unexpected checkpoint format: {path}")
        if int(checkpoint.get("step", -1)) != int(step):
            raise ValueError(f"checkpoint step mismatch: {path}")
        schema = checkpoint_schema(checkpoint)
        if schema["pretrained"] != args.pretrained:
            raise RuntimeError(f"pretrained mismatch at step={step}")
        if float(schema["physical_scale"]) != float(args.physical_scale):
            raise RuntimeError(f"physical scale mismatch at step={step}")
        if str(schema["cache_config_hash"]) != str(cache_hash):
            raise RuntimeError(f"checkpoint/cache hash mismatch at step={step}")
        if tuple(schema["control_names"]) != VIEW_IDENTITY_CONTROL_NAMES:
            raise RuntimeError(f"control schema mismatch at step={step}")
        schemas.append(schema)
        probe = ViewIdentityPoseDepthProbe(
            visual_channels=int(visual_channels),
            hidden_dim=int(schema["hidden_dim"]),
            pair_dim=int(schema["pair_dim"]),
            min_views=int(schema["min_views"]),
        ).to(device).eval()
        load_view_identity_probe_state(probe, checkpoint["model_trainable_state"])
        probes[int(step)] = probe
    if any(schema != schemas[0] for schema in schemas[1:]):
        raise RuntimeError("checkpoint trajectory protocol/schema mismatch")
    return probes, schemas[0]


def decision_for(
    result: dict[str, Any], args: argparse.Namespace
) -> dict[str, Any]:
    correct = result["correct_vs_stock"]
    controls = result["controls"]
    positive_t_count = sum(
        float(row["correct_relative_gain"]["mean"]) > 0.0
        for row in result["per_t"].values()
    )
    checks = {
        "correct_mean_positive": float(correct["object"]["mean"]) > 0.0,
        "correct_median_positive": float(correct["object"]["median"]) > 0.0,
        "correct_object_win_rate": float(correct["object_win_rate"])
        >= float(args.min_object_win_rate),
        "correct_bootstrap_ci_positive": float(
            correct["object_bootstrap_95_ci"][0]
        )
        > 0.0,
        "correct_beats_every_control_mean": all(
            float(row["correct_advantage_flow"]["object"]["mean"]) > 0.0
            for row in controls.values()
        ),
        "correct_beats_every_control_median": all(
            float(row["correct_advantage_flow"]["object"]["median"]) > 0.0
            for row in controls.values()
        ),
        "control_object_win_rates": all(
            float(row["correct_advantage_flow"]["object_win_rate"])
            >= float(args.min_object_win_rate)
            for row in controls.values()
        ),
        "control_bootstrap_cis_positive": all(
            float(row["correct_advantage_flow"]["object_bootstrap_95_ci"][0])
            > 0.0
            for row in controls.values()
        ),
        "positive_t_count": positive_t_count >= int(args.min_positive_t_count),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "positive_t_count": positive_t_count,
    }


def summarize_records(
    records: list[dict[str, Any]],
    *,
    t_values: list[float],
    bootstrap_samples: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    correct = object_balanced(
        records, "correct_relative_gain", bootstrap_samples=bootstrap_samples
    )
    local = object_balanced(
        records, "correct_local_target_gain", bootstrap_samples=bootstrap_samples
    )
    controls = {
        mode: {
            "relative_gain_vs_stock": object_balanced(
                records,
                f"{mode}_relative_gain",
                bootstrap_samples=bootstrap_samples,
            ),
            "correct_advantage_flow": object_balanced(
                records,
                f"correct_vs_{mode}",
                bootstrap_samples=bootstrap_samples,
            ),
            "correct_advantage_local": object_balanced(
                records,
                f"correct_local_vs_{mode}",
                bootstrap_samples=bootstrap_samples,
            ),
            "delta_rms": object_balanced(
                records, f"{mode}_delta_rms", bootstrap_samples=bootstrap_samples
            ),
            "pair_consensus": object_balanced(
                records,
                f"{mode}_pair_consensus",
                bootstrap_samples=bootstrap_samples,
            ),
        }
        for mode in VIEW_IDENTITY_CONTROL_NAMES
    }
    per_t = {
        f"{t_value:.3f}": {
            "correct_relative_gain": summarize(
                [
                    float(row["correct_relative_gain"])
                    for row in records
                    if row["t"] == float(t_value)
                ]
            ),
            "correct_vs_controls": {
                mode: summarize(
                    [
                        float(row[f"correct_vs_{mode}"])
                        for row in records
                        if row["t"] == float(t_value)
                    ]
                )
                for mode in VIEW_IDENTITY_CONTROL_NAMES
            },
        }
        for t_value in t_values
    }
    result = {
        "record_count": len(records),
        "object_count": len({str(row["object_uid"]) for row in records}),
        "correct_vs_stock": correct,
        "correct_local_vs_zero": local,
        "correct_delta_rms": object_balanced(
            records, "correct_delta_rms", bootstrap_samples=bootstrap_samples
        ),
        "correct_pair_consensus": object_balanced(
            records, "correct_pair_consensus", bootstrap_samples=bootstrap_samples
        ),
        "controls": controls,
        "per_t": per_t,
    }
    result["decision"] = decision_for(result, args)
    return result


def aligned_rows(
    rows: list[dict[str, Any]],
) -> dict[tuple[str, int, float], dict[str, Any]]:
    result: dict[tuple[str, int, float], dict[str, Any]] = {}
    for row in rows:
        key = (str(row["uid"]), int(row["noise_seed"]), float(row["t"]))
        if key in result:
            raise RuntimeError(f"duplicate diagnostic record key: {key}")
        result[key] = row
    return result


def attribution_report(
    full_rows: list[dict[str, Any]],
    ablated_rows: list[dict[str, Any]],
    *,
    bootstrap_samples: int,
) -> dict[str, Any]:
    full = aligned_rows(full_rows)
    ablated = aligned_rows(ablated_rows)
    if set(full) != set(ablated):
        raise RuntimeError("ablation record keys differ from full protocol")
    rows = []
    for key in sorted(full):
        left = full[key]
        right = ablated[key]
        row = {
            "uid": left["uid"],
            "object_uid": left["object_uid"],
            "noise_seed": left["noise_seed"],
            "t": left["t"],
            "full_minus_ablation_correct_gain": float(
                left["correct_relative_gain"] - right["correct_relative_gain"]
            ),
            "full_minus_ablation_delta_rms": float(
                left["correct_delta_rms"] - right["correct_delta_rms"]
            ),
        }
        for mode in VIEW_IDENTITY_CONTROL_NAMES:
            row[f"full_minus_ablation_specificity_{mode}"] = float(
                left[f"correct_vs_{mode}"] - right[f"correct_vs_{mode}"]
            )
        rows.append(row)
    return {
        "correct_gain": object_balanced(
            rows,
            "full_minus_ablation_correct_gain",
            bootstrap_samples=bootstrap_samples,
        ),
        "delta_rms": object_balanced(
            rows,
            "full_minus_ablation_delta_rms",
            bootstrap_samples=bootstrap_samples,
        ),
        "specificity": {
            mode: object_balanced(
                rows,
                f"full_minus_ablation_specificity_{mode}",
                bootstrap_samples=bootstrap_samples,
            )
            for mode in VIEW_IDENTITY_CONTROL_NAMES
        },
    }


def reference_metrics(result: dict[str, Any]) -> dict[str, float]:
    metrics = {
        "correct_mean": float(result["correct_vs_stock"]["object"]["mean"]),
        "correct_median": float(result["correct_vs_stock"]["object"]["median"]),
        "correct_win": float(result["correct_vs_stock"]["object_win_rate"]),
    }
    for mode, row in result["controls"].items():
        metrics[f"{mode}_adv_mean"] = float(
            row["correct_advantage_flow"]["object"]["mean"]
        )
        metrics[f"{mode}_adv_median"] = float(
            row["correct_advantage_flow"]["object"]["median"]
        )
        metrics[f"{mode}_adv_win"] = float(
            row["correct_advantage_flow"]["object_win_rate"]
        )
    return metrics


def compare_reference(
    result: dict[str, Any], path: str | None, tolerance: float
) -> dict[str, Any] | None:
    if path is None:
        return None
    reference_path = Path(path)
    if not reference_path.is_file():
        raise FileNotFoundError(reference_path)
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    current_metrics = reference_metrics(result)
    reference_values = reference_metrics(reference)
    differences = {
        key: abs(current_metrics[key] - reference_values[key])
        for key in current_metrics
    }
    maximum = max(differences.values(), default=0.0)
    return {
        "reference": str(reference_path.resolve()),
        "max_abs_metric_diff": maximum,
        "tolerance": float(tolerance),
        "passed": maximum <= float(tolerance),
        "differences": differences,
    }


def write_trajectory_csv(report: dict[str, Any], path: Path) -> None:
    fields = (
        "split",
        "step",
        "correct_mean",
        "correct_median",
        "correct_object_win_rate",
        "correct_ci_low",
        "correct_ci_high",
        "minimum_control_advantage_mean",
        "positive_t_count",
        "passed",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for split, split_rows in report["results"].items():
            for step, modes in split_rows.items():
                row = modes["full"]
                correct = row["correct_vs_stock"]
                writer.writerow(
                    {
                        "split": split,
                        "step": step,
                        "correct_mean": correct["object"]["mean"],
                        "correct_median": correct["object"]["median"],
                        "correct_object_win_rate": correct["object_win_rate"],
                        "correct_ci_low": correct["object_bootstrap_95_ci"][0],
                        "correct_ci_high": correct["object_bootstrap_95_ci"][1],
                        "minimum_control_advantage_mean": min(
                            float(value["correct_advantage_flow"]["object"]["mean"])
                            for value in row["controls"].values()
                        ),
                        "positive_t_count": row["decision"]["positive_t_count"],
                        "passed": row["decision"]["passed"],
                    }
                )


def write_ablation_csv(report: dict[str, Any], path: Path) -> None:
    fields = (
        "split",
        "step",
        "ablation",
        "correct_mean",
        "correct_median",
        "correct_object_win_rate",
        "delta_rms_mean",
        "full_minus_ablation_correct_gain_mean",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for split, step_rows in report["attribution"].items():
            for step, mode_rows in step_rows.items():
                for mode, attribution in mode_rows.items():
                    result = report["results"][split][step][mode]
                    writer.writerow(
                        {
                            "split": split,
                            "step": step,
                            "ablation": mode,
                            "correct_mean": result["correct_vs_stock"]["object"][
                                "mean"
                            ],
                            "correct_median": result["correct_vs_stock"]["object"][
                                "median"
                            ],
                            "correct_object_win_rate": result["correct_vs_stock"][
                                "object_win_rate"
                            ],
                            "delta_rms_mean": result["correct_delta_rms"]["object"][
                                "mean"
                            ],
                            "full_minus_ablation_correct_gain_mean": attribution[
                                "correct_gain"
                            ]["object"]["mean"],
                        }
                    )


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# View-identity V1 Checkpoint Trajectory and Ablations",
        "",
        f"- Overall paired checkpoint found: **{report['passed']}**",
        f"- Checkpoints: `{report['protocol']['checkpoint_steps']}`",
        f"- Ablation steps: `{report['protocol']['ablation_steps']}`",
        f"- Ablations: `{report['protocol']['ablation_modes']}`",
        "- Primary protocol: fixed correct view weight and support gate.",
        "",
        "## Full trajectory",
        "",
        "| Split | Step | Mean | Median | Win | CI low | Min control adv | +t | PASS |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for split, split_rows in report["results"].items():
        for step, modes in split_rows.items():
            row = modes["full"]
            correct = row["correct_vs_stock"]
            lines.append(
                "| {split} | {step} | {mean:.6g} | {median:.6g} | {win:.3f} | "
                "{ci:.6g} | {control:.6g} | {positive_t} | {passed} |".format(
                    split=split,
                    step=step,
                    mean=float(correct["object"]["mean"]),
                    median=float(correct["object"]["median"]),
                    win=float(correct["object_win_rate"]),
                    ci=float(correct["object_bootstrap_95_ci"][0]),
                    control=min(
                        float(value["correct_advantage_flow"]["object"]["mean"])
                        for value in row["controls"].values()
                    ),
                    positive_t=row["decision"]["positive_t_count"],
                    passed=row["decision"]["passed"],
                )
            )
    lines.extend(["", "## Ablations", ""])
    for split, step_rows in report["attribution"].items():
        lines.extend([f"### {split}", ""])
        for step, mode_rows in step_rows.items():
            lines.extend(
                [
                    f"Step `{step}`",
                    "",
                    "| Mode | Correct mean | Win | Delta RMS | Full-mode gain |",
                    "|---|---:|---:|---:|---:|",
                ]
            )
            for mode, attribution in mode_rows.items():
                result = report["results"][split][step][mode]
                lines.append(
                    "| {mode} | {gain:.6g} | {win:.3f} | {delta:.6g} | "
                    "{attribution:.6g} |".format(
                        mode=mode,
                        gain=float(result["correct_vs_stock"]["object"]["mean"]),
                        win=float(result["correct_vs_stock"]["object_win_rate"]),
                        delta=float(result["correct_delta_rms"]["object"]["mean"]),
                        attribution=float(
                            attribution["correct_gain"]["object"]["mean"]
                        ),
                    )
                )
            lines.append("")
    lines.extend(
        [
            "## Decision",
            "",
            (
                "At least one checkpoint passes train and fresh strict gates."
                if report["passed"]
                else "No checkpoint passes both train and fresh strict gates; stop V1."
            ),
            "",
            "Scientific failure is recorded in the report and does not change the "
            "process exit code.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@torch.no_grad()
def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("view-identity checkpoint diagnostic requires CUDA")

    steps = parse_csv_ints(args.checkpoint_steps)
    ablation_steps = set(parse_csv_ints(args.ablation_steps))
    if not ablation_steps.issubset(set(steps)):
        raise ValueError("ablation steps must be included in checkpoint steps")
    ablation_modes = parse_csv_strings(args.ablation_modes)
    if "full" not in ablation_modes:
        raise ValueError("ablation modes must include full")
    if any(mode not in VIEW_IDENTITY_ABLATION_MODES for mode in ablation_modes):
        raise ValueError(
            f"ablation modes must be drawn from {VIEW_IDENTITY_ABLATION_MODES}"
        )
    seeds = parse_csv_ints(args.seeds)
    t_values = parse_csv_floats(args.t_values)

    datasets = {
        "train16": PoseLiftingCacheDataset(
            args.cache_manifest, indices=args.train_indices
        ),
        "fresh48": PoseLiftingCacheDataset(
            args.cache_manifest, indices=args.fresh_indices
        ),
    }
    cache_hashes = {dataset.config_hash for dataset in datasets.values()}
    if len(cache_hashes) != 1:
        raise RuntimeError("train/fresh cache hashes differ")
    cache_hash = next(iter(cache_hashes))
    visual_channels = next(iter(datasets.values())).visual_feature_dim
    probes, schema = load_probes(
        args,
        visual_channels=visual_channels,
        cache_hash=cache_hash,
        device=device,
    )
    train_objects = set(schema["train_object_uids"])
    split_uids: dict[str, list[str]] = {}
    split_objects: dict[str, set[str]] = {}
    for split, dataset in datasets.items():
        count = len(dataset)
        if args.max_samples_per_split > 0:
            count = min(count, int(args.max_samples_per_split))
        rows = dataset.rows[:count]
        split_uids[split] = [str(row["uid"]) for row in rows]
        split_objects[split] = {
            str(row.get("object_uid", row["uid"])) for row in rows
        }
    if not split_objects["train16"].issubset(train_objects):
        raise RuntimeError("train16 diagnostic contains non-training objects")
    if (
        int(args.max_samples_per_split) <= 0
        and split_objects["train16"] != train_objects
    ):
        raise RuntimeError("full train16 objects differ from checkpoint training objects")
    overlap = sorted(split_objects["fresh48"] & train_objects)
    if overlap:
        raise RuntimeError(f"fresh48 object leakage: {overlap}")

    sampler, stock_flow, flow_schema = build_frozen_stock_flow(
        args.pretrained, device
    )
    amp_name = str(schema["amp_dtype"])
    use_amp = amp_name != "none"
    amp_dtype = torch.float16 if amp_name == "fp16" else torch.bfloat16
    records: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    zero_audit = {
        "physical_off_max_abs": 0.0,
        "null_evidence_max_abs": 0.0,
        "neutral_max_abs": 0.0,
    }

    for split, dataset in datasets.items():
        count = len(split_uids[split])
        for position in range(count):
            sample = dataset[position]
            uid = str(sample["uid"])
            object_uid = str(sample.get("object_uid", uid))
            target = sample["target"].unsqueeze(0).to(
                device=device, dtype=torch.float32
            )
            condition = sample["stock_condition"].to(device=device)
            correct_evidence = build_view_identity_evidence(
                sample, device=device, mode="correct"
            )
            control_evidence = {
                mode: build_view_identity_evidence(sample, device=device, mode=mode)
                for mode in VIEW_IDENTITY_CONTROL_NAMES
            }
            fixed_weight = correct_evidence["view_weight"].float()
            fixed_support = probes[steps[-1]].support_gate(
                correct_evidence, view_weight_override=fixed_weight
            )
            active_mask = fixed_support.reshape(1, 1, 16, 16, 16)
            if not bool(active_mask.any().item()):
                raise RuntimeError(f"uid={uid} has empty fixed support")

            states: list[dict[str, Any]] = []
            for noise_seed in seeds:
                for t_value in t_values:
                    generator = torch.Generator(device=device).manual_seed(
                        deterministic_noise_seed(uid, noise_seed, t_value)
                    )
                    noise = torch.randn(
                        target.shape,
                        generator=generator,
                        device=device,
                        dtype=target.dtype,
                    )
                    x_t, gt_velocity = sampler._get_model_gt(
                        target, t_value, noise
                    )
                    t_tensor = torch.full(
                        (1,),
                        1000.0 * t_value,
                        device=device,
                        dtype=torch.float32,
                    )
                    with torch.cuda.amp.autocast(
                        enabled=use_amp, dtype=amp_dtype
                    ):
                        stock = stock_flow(x_t, t_tensor, condition)
                    target_residual = gt_velocity.float() - stock.float()
                    target_energy = masked_mse(
                        target_residual,
                        torch.zeros_like(target_residual),
                        active_mask,
                    ).clamp_min(1.0e-6)
                    states.append(
                        {
                            "noise_seed": int(noise_seed),
                            "t": float(t_value),
                            "x_t": x_t,
                            "gt_velocity": gt_velocity,
                            "t_tensor": t_tensor,
                            "stock": stock,
                            "stock_loss": F.mse_loss(
                                stock.float(), gt_velocity.float()
                            ).clamp_min(1.0e-6),
                            "target_residual": target_residual,
                            "target_energy": target_energy,
                        }
                    )

            for step in steps:
                probe = probes[step]
                modes = ablation_modes if step in ablation_steps else ["full"]
                for ablation_mode in modes:
                    key = (split, int(step), ablation_mode)
                    output_rows = records.setdefault(key, [])
                    with torch.cuda.amp.autocast(
                        enabled=use_amp, dtype=amp_dtype
                    ):
                        prepared_correct = probe.prepare_evidence(
                            correct_evidence,
                            view_weight_override=fixed_weight,
                            ablation_mode=ablation_mode,
                        )
                    sample_rows: list[dict[str, Any]] = []
                    for state in states:
                        with torch.cuda.amp.autocast(
                            enabled=use_amp, dtype=amp_dtype
                        ):
                            delta, stats = probe.forward_prepared(
                                state["x_t"],
                                state["stock"],
                                state["t_tensor"],
                                prepared_correct,
                                scale=float(args.physical_scale),
                                support_gate_override=fixed_support,
                            )
                        correct_loss = F.mse_loss(
                            (state["stock"] + delta).float(),
                            state["gt_velocity"].float(),
                        )
                        local_loss = masked_mse(
                            delta, state["target_residual"], active_mask
                        ) / state["target_energy"]
                        row = {
                            "uid": uid,
                            "object_uid": object_uid,
                            "noise_seed": state["noise_seed"],
                            "t": state["t"],
                            "stock_flow_loss": float(state["stock_loss"].item()),
                            "correct_flow_loss": float(correct_loss.item()),
                            "correct_relative_gain": float(
                                (
                                    (state["stock_loss"] - correct_loss)
                                    / state["stock_loss"]
                                ).item()
                            ),
                            "correct_local_target_gain": float(
                                (1.0 - local_loss).item()
                            ),
                            "correct_delta_rms": float(stats["delta_rms"].item()),
                            "correct_pair_consensus": float(
                                stats["pair_consensus"].item()
                            ),
                            "active_ratio": float(fixed_support.mean().item()),
                        }
                        sample_rows.append(row)
                        zero_audit["neutral_max_abs"] = max(
                            zero_audit["neutral_max_abs"],
                            float(stats["neutral_abs_max"].item()),
                        )

                    for control_name, evidence in control_evidence.items():
                        with torch.cuda.amp.autocast(
                            enabled=use_amp, dtype=amp_dtype
                        ):
                            prepared_control = probe.prepare_evidence(
                                evidence,
                                view_weight_override=fixed_weight,
                                ablation_mode=ablation_mode,
                            )
                        for index, state in enumerate(states):
                            with torch.cuda.amp.autocast(
                                enabled=use_amp, dtype=amp_dtype
                            ):
                                delta, stats = probe.forward_prepared(
                                    state["x_t"],
                                    state["stock"],
                                    state["t_tensor"],
                                    prepared_control,
                                    scale=float(args.physical_scale),
                                    support_gate_override=fixed_support,
                                )
                            flow_loss = F.mse_loss(
                                (state["stock"] + delta).float(),
                                state["gt_velocity"].float(),
                            )
                            local_loss = masked_mse(
                                delta, state["target_residual"], active_mask
                            ) / state["target_energy"]
                            row = sample_rows[index]
                            row[f"{control_name}_relative_gain"] = float(
                                (
                                    (state["stock_loss"] - flow_loss)
                                    / state["stock_loss"]
                                ).item()
                            )
                            row[f"correct_vs_{control_name}"] = float(
                                (
                                    (flow_loss - row["correct_flow_loss"])
                                    / state["stock_loss"]
                                ).item()
                            )
                            correct_local_loss = 1.0 - float(
                                row["correct_local_target_gain"]
                            )
                            row[f"correct_local_vs_{control_name}"] = float(
                                local_loss.item() - correct_local_loss
                            )
                            row[f"{control_name}_delta_rms"] = float(
                                stats["delta_rms"].item()
                            )
                            row[f"{control_name}_pair_consensus"] = float(
                                stats["pair_consensus"].item()
                            )
                        del prepared_control
                    output_rows.extend(sample_rows)

                    if (
                        split == "train16"
                        and position == 0
                        and step == steps[-1]
                        and ablation_mode == "full"
                    ):
                        state = states[0]
                        off_delta, _ = probe.forward_prepared(
                            state["x_t"],
                            state["stock"],
                            state["t_tensor"],
                            prepared_correct,
                            physical_present=False,
                            support_gate_override=fixed_support,
                        )
                        null = make_null_view_identity_evidence(correct_evidence)
                        with torch.cuda.amp.autocast(
                            enabled=use_amp, dtype=amp_dtype
                        ):
                            null_prepared = probe.prepare_evidence(null)
                            null_delta, _ = probe.forward_prepared(
                                state["x_t"],
                                state["stock"],
                                state["t_tensor"],
                                null_prepared,
                            )
                        zero_audit["physical_off_max_abs"] = float(
                            off_delta.float().abs().max().item()
                        )
                        zero_audit["null_evidence_max_abs"] = float(
                            null_delta.float().abs().max().item()
                        )
                        del null_prepared
                    del prepared_correct
            print(
                f"[view_identity_diagnostic] split={split} "
                f"{position + 1}/{count} uid={uid}",
                flush=True,
            )

    results: dict[str, Any] = {}
    for split in datasets:
        results[split] = {}
        for step in steps:
            modes = ablation_modes if step in ablation_steps else ["full"]
            results[split][str(step)] = {
                mode: summarize_records(
                    records[(split, step, mode)],
                    t_values=t_values,
                    bootstrap_samples=int(args.bootstrap_samples),
                    args=args,
                )
                for mode in modes
            }

    attribution: dict[str, Any] = {}
    for split in datasets:
        attribution[split] = {}
        for step in sorted(ablation_steps):
            full_rows = records[(split, step, "full")]
            attribution[split][str(step)] = {
                mode: attribution_report(
                    full_rows,
                    records[(split, step, mode)],
                    bootstrap_samples=int(args.bootstrap_samples),
                )
                for mode in ablation_modes
            }

    primary_pass = {
        split: {
            str(step): bool(results[split][str(step)]["full"]["decision"]["passed"])
            for step in steps
        }
        for split in results
    }
    paired_pass = {
        str(step): all(primary_pass[split][str(step)] for split in results)
        for step in steps
    }
    references = {
        "train16": compare_reference(
            results["train16"][str(steps[-1])]["full"],
            args.reference_train_report,
            args.max_reference_metric_diff,
        ),
        "fresh48": compare_reference(
            results["fresh48"][str(steps[-1])]["full"],
            args.reference_fresh_report,
            args.max_reference_metric_diff,
        ),
    }
    reference_pass = all(
        row is None or bool(row["passed"]) for row in references.values()
    )
    protocol = {
        "version": "pose_point_depth_mv.view_identity_diagnostic.v1",
        "cache_manifest": str(Path(args.cache_manifest).resolve()),
        "cache_config_hash": cache_hash,
        "checkpoint_dir": str(Path(args.checkpoint_dir).resolve()),
        "checkpoint_steps": steps,
        "ablation_steps": sorted(ablation_steps),
        "ablation_modes": ablation_modes,
        "checkpoint_schema": schema,
        "noise_seeds": seeds,
        "t_values": t_values,
        "physical_scale": float(args.physical_scale),
        "fixed_correct_view_weight": True,
        "fixed_correct_support_gate": True,
        "flow": flow_schema,
        "split_uids": split_uids,
        "split_uid_hashes": {
            split: sha256_strings(uids) for split, uids in split_uids.items()
        },
    }
    protocol["protocol_hash"] = protocol_hash(protocol)
    report = {
        "stage": "View-identity V1 checkpoint trajectory and ablations",
        "passed": any(paired_pass.values()) and reference_pass,
        "protocol": protocol,
        "stock_preservation": zero_audit,
        "reference_reproduction": references,
        "reference_reproduction_passed": reference_pass,
        "primary_pass_by_split_step": primary_pass,
        "paired_train_fresh_pass_by_step": paired_pass,
        "results": results,
        "attribution": attribution,
    }
    (output_dir / "diagnostic_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    write_trajectory_csv(report, output_dir / "checkpoint_trajectory.csv")
    write_ablation_csv(report, output_dir / "ablation_summary.csv")
    write_markdown(report, output_dir / "report.md")
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "paired_train_fresh_pass_by_step": paired_pass,
                "reference_reproduction_passed": reference_pass,
                "stock_preservation": zero_audit,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
