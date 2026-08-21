#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
from statistics import mean, median
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
for path in (
    TRACKER_ROOT,
    TRACKER_ROOT / "ReconViaGen",
    TRACKER_ROOT / "ReconViaGen" / "wheels" / "vggt",
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from pose_point_depth_mv.eval_local_target_probe import (  # noqa: E402
    bootstrap_ci,
    deterministic_noise_seed,
    parse_csv_floats,
    parse_csv_ints,
    positive_rate,
    summarize,
)
from pose_point_depth_mv.local_target_probe import (  # noqa: E402
    EVIDENCE_ABLATIONS,
    PPDLocalTargetProbe,
    PPDProbeEvidenceDataset,
    PROBE_CORRUPTIONS,
    ablate_evidence,
    load_probe_state,
    make_null_evidence,
)
from pose_point_depth_mv.train_local_target_probe import (  # noqa: E402
    CHECKPOINT_VERSION,
    build_frozen_stock_flow,
    masked_mse,
)


MASK_PROTOCOLS = ("native", "fixed_correct")
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
    output = [item.strip() for item in value.split(",") if item.strip()]
    if not output:
        raise ValueError("at least one value is required")
    return output


def sha256_strings(values: list[str]) -> str:
    payload = "\n".join(values).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def masked_loss_or_none(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> float | None:
    if not bool(mask.bool().any().item()):
        return None
    energy = masked_mse(target, torch.zeros_like(target), mask).clamp_min(1.0e-6)
    return float((masked_mse(prediction, target, mask) / energy).item())


@dataclass
class MetricAccumulator:
    values: list[float] = field(default_factory=list)
    object_values: dict[str, list[float]] = field(
        default_factory=lambda: defaultdict(list)
    )
    t_values: dict[str, list[float]] = field(
        default_factory=lambda: defaultdict(list)
    )

    def add(self, value: float | None, *, object_uid: str, t_value: float) -> None:
        if value is None or not np.isfinite(float(value)):
            return
        number = float(value)
        self.values.append(number)
        self.object_values[str(object_uid)].append(number)
        self.t_values[f"{float(t_value):.3f}"].append(number)

    def report(self, *, bootstrap_samples: int) -> dict[str, Any]:
        object_means = [mean(values) for values in self.object_values.values()]
        return {
            "record": summarize(self.values),
            "record_win_rate": positive_rate(self.values),
            "object": summarize(object_means),
            "object_win_rate": positive_rate(object_means),
            "object_bootstrap_95_ci": bootstrap_ci(
                object_means,
                seed=20260715,
                samples=int(bootstrap_samples),
            ),
            "per_t": {
                key: summarize(values) for key, values in sorted(self.t_values.items())
            },
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Shared-stock-flow checkpoint trajectory and evidence-ablation "
            "diagnostic for existing PPD-3A probes."
        )
    )
    parser.add_argument("--probe_cache_manifest", required=True)
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--checkpoint_steps", default="25,50,75,100")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--pretrained", default="Stable-X/trellis-vggt-v0-2")
    parser.add_argument("--train_indices", default="0-15")
    parser.add_argument("--fresh_indices", default="16-63")
    parser.add_argument("--max_samples_per_split", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seeds", default="48,49,50")
    parser.add_argument("--t_values", default="0.1,0.3,0.5,0.7,0.9")
    parser.add_argument("--physical_scale", type=float, default=1.0)
    parser.add_argument("--mask_protocols", default=",".join(MASK_PROTOCOLS))
    parser.add_argument(
        "--ablation_modes", default=",".join(EVIDENCE_ABLATIONS)
    )
    parser.add_argument("--bootstrap_samples", type=int, default=10000)
    parser.add_argument("--min_correct_object_win_rate", type=float, default=0.65)
    parser.add_argument("--min_corruption_object_win_rate", type=float, default=0.60)
    parser.add_argument("--min_positive_t_count", type=int, default=4)
    return parser.parse_args()


def load_checkpoints(
    args: argparse.Namespace,
    device: torch.device,
    cache_hash: str,
) -> tuple[dict[int, PPDLocalTargetProbe], dict[str, Any]]:
    steps = parse_csv_ints(args.checkpoint_steps)
    checkpoints: dict[int, dict[str, Any]] = {}
    schemas: list[dict[str, Any]] = []
    for step in steps:
        path = Path(args.checkpoint_dir) / f"step_{step:06d}.pt"
        if not path.is_file():
            raise FileNotFoundError(path)
        checkpoint = torch.load(path, map_location="cpu")
        if checkpoint.get("format") != CHECKPOINT_VERSION:
            raise ValueError(f"unexpected checkpoint format: {path}")
        if int(checkpoint.get("step", -1)) != int(step):
            raise ValueError(f"checkpoint step mismatch: {path}")
        saved_args = checkpoint.get("args", {})
        trained_hash = str(
            checkpoint.get("model_summary", {}).get(
                "evidence_cache_config_hash", ""
            )
        )
        if trained_hash != cache_hash:
            raise RuntimeError(
                f"checkpoint/cache hash mismatch at step={step}: "
                f"{trained_hash} != {cache_hash}"
            )
        if saved_args.get("pretrained") != args.pretrained:
            raise RuntimeError(f"pretrained mismatch at step={step}")
        schema = {
            "rank": int(saved_args["rank"]),
            "amp_dtype": str(saved_args.get("amp_dtype", "bf16")),
            "pretrained": str(saved_args["pretrained"]),
            "cache_hash": trained_hash,
            "probe": checkpoint.get("model_summary", {}).get("probe"),
            "corruptions": checkpoint.get("model_summary", {}).get(
                "corruption_modes"
            ),
            "loss_config": {
                key: saved_args.get(key) for key in LOSS_CONFIG_KEYS
            },
        }
        schemas.append(schema)
        checkpoints[int(step)] = checkpoint
    if any(schema != schemas[0] for schema in schemas[1:]):
        raise RuntimeError("checkpoint sweep protocol/schema mismatch")

    probes: dict[int, PPDLocalTargetProbe] = {}
    for step, checkpoint in checkpoints.items():
        probe = PPDLocalTargetProbe(rank=int(schemas[0]["rank"])).to(device).eval()
        load_probe_state(probe, checkpoint["model_trainable_state"])
        probes[step] = probe
    return probes, schemas[0]


def mask_regions(
    correct: torch.Tensor, corrupt: torch.Tensor
) -> dict[str, torch.Tensor]:
    correct_bool = correct > 0.5
    corrupt_bool = corrupt > 0.5
    return {
        "shared": (correct_bool & corrupt_bool).float(),
        "correct_only": (correct_bool & ~corrupt_bool).float(),
        "corrupt_only": (~correct_bool & corrupt_bool).float(),
        "union": (correct_bool | corrupt_bool).float(),
    }


def append_metric(
    accumulators: dict[tuple[str, int, str, str, str], MetricAccumulator],
    *,
    split: str,
    step: int,
    protocol: str,
    ablation: str,
    metric: str,
    value: float | None,
    object_uid: str,
    t_value: float,
) -> None:
    key = (split, int(step), protocol, ablation, metric)
    accumulators.setdefault(key, MetricAccumulator()).add(
        value, object_uid=object_uid, t_value=t_value
    )


def summarize_mask_overlap(
    rows: dict[tuple[str, str, str], list[float]]
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for (split, corruption, metric), values in sorted(rows.items()):
        output.setdefault(split, {}).setdefault(corruption, {})[metric] = summarize(
            values
        )
    return output


def decision_for(
    summary: dict[str, Any], args: argparse.Namespace
) -> dict[str, Any]:
    correct = summary["metrics"]["correct/relative_gain_vs_stock"]
    corruptions = summary["corruptions"]
    positive_t = sum(
        float(row["mean"]) > 0.0 for row in correct["per_t"].values()
    )
    checks = {
        "correct_mean_positive": float(correct["object"]["mean"]) > 0.0,
        "correct_median_positive": float(correct["object"]["median"]) > 0.0,
        "correct_object_win_rate": float(correct["object_win_rate"])
        >= float(args.min_correct_object_win_rate),
        "correct_positive_t_count": positive_t >= int(args.min_positive_t_count),
        "correct_beats_every_corruption_mean": all(
            float(row["correct_advantage_flow"]["object"]["mean"]) > 0.0
            for row in corruptions.values()
        ),
        "correct_beats_every_corruption_median": all(
            float(row["correct_advantage_flow"]["object"]["median"]) > 0.0
            for row in corruptions.values()
        ),
        "corruption_object_win_rates": all(
            float(row["correct_advantage_flow"]["object_win_rate"])
            >= float(args.min_corruption_object_win_rate)
            for row in corruptions.values()
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "positive_t_count": positive_t,
    }


def write_trajectory_csv(report: dict[str, Any], path: Path) -> None:
    fields = [
        "split",
        "step",
        "mask_protocol",
        "ablation",
        "correct_mean",
        "correct_median",
        "correct_object_win_rate",
        "correct_ci_low",
        "correct_ci_high",
        "decision_passed",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for split, split_rows in report["results"].items():
            for step, step_rows in split_rows.items():
                for protocol, protocol_rows in step_rows.items():
                    for ablation, row in protocol_rows.items():
                        correct = row["metrics"]["correct/relative_gain_vs_stock"]
                        ci = correct["object_bootstrap_95_ci"]
                        writer.writerow(
                            {
                                "split": split,
                                "step": step,
                                "mask_protocol": protocol,
                                "ablation": ablation,
                                "correct_mean": correct["object"]["mean"],
                                "correct_median": correct["object"]["median"],
                                "correct_object_win_rate": correct["object_win_rate"],
                                "correct_ci_low": ci[0],
                                "correct_ci_high": ci[1],
                                "decision_passed": row["decision"]["passed"],
                            }
                        )


def write_ablation_csv(report: dict[str, Any], path: Path) -> None:
    fields = [
        "split",
        "step",
        "mask_protocol",
        "comparison",
        "object_mean",
        "object_median",
        "object_win_rate",
        "ci_low",
        "ci_high",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for split, split_rows in report["attribution"].items():
            for step, step_rows in split_rows.items():
                for protocol, comparisons in step_rows.items():
                    for name, row in comparisons.items():
                        writer.writerow(
                            {
                                "split": split,
                                "step": step,
                                "mask_protocol": protocol,
                                "comparison": name,
                                "object_mean": row["object"]["mean"],
                                "object_median": row["object"]["median"],
                                "object_win_rate": row["object_win_rate"],
                                "ci_low": row["object_bootstrap_95_ci"][0],
                                "ci_high": row["object_bootstrap_95_ci"][1],
                            }
                        )


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# PPD-3A Checkpoint Sweep and Input Ablation",
        "",
        f"- Overall decision: **{'PASS' if report['passed'] else 'FAIL'}**",
        f"- Checkpoints: `{report['protocol']['checkpoint_steps']}`",
        f"- Mask protocols: `{report['protocol']['mask_protocols']}`",
        f"- Ablations: `{report['protocol']['ablation_modes']}`",
        "- Primary scientific protocol: `fixed_correct + full`",
        "",
        "## Primary trajectory",
        "",
        "| Split | Step | Correct mean | Correct median | Win | Pose cyc adv | Depth cyc adv | Point cross adv | PASS |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for split, split_rows in report["results"].items():
        for step, step_rows in split_rows.items():
            row = step_rows["fixed_correct"]["full"]
            correct = row["metrics"]["correct/relative_gain_vs_stock"]
            corruptions = row["corruptions"]
            lines.append(
                "| {split} | {step} | {mean:.6g} | {median:.6g} | {win:.3f} | "
                "{pose:.6g} | {depth:.6g} | {point:.6g} | {passed} |".format(
                    split=split,
                    step=step,
                    mean=float(correct["object"]["mean"]),
                    median=float(correct["object"]["median"]),
                    win=float(correct["object_win_rate"]),
                    pose=float(
                        corruptions["pose_cyclic1"]["correct_advantage_flow"][
                            "object"
                        ]["mean"]
                    ),
                    depth=float(
                        corruptions["depth_view_cyclic1"][
                            "correct_advantage_flow"
                        ]["object"]["mean"]
                    ),
                    point=float(
                        corruptions["point_cross_object"][
                            "correct_advantage_flow"
                        ]["object"]["mean"]
                    ),
                    passed=row["decision"]["passed"],
                )
            )
    lines.extend(
        [
            "",
            "## Attribution",
            "",
            "The paired attribution tables are in `ablation_summary.csv`. Positive "
            "values mean the first input has a larger correct-vs-stock gain.",
            "",
            "## Mask overlap",
            "",
            "```json",
            json.dumps(report["mask_overlap"], indent=2),
            "```",
            "",
            "Scientific FAIL is recorded in this report and does not change the process "
            "exit code.",
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
        raise ValueError("PPD checkpoint diagnostics require CUDA")

    mask_protocols = parse_csv_strings(args.mask_protocols)
    if any(item not in MASK_PROTOCOLS for item in mask_protocols):
        raise ValueError(f"mask protocols must be drawn from {MASK_PROTOCOLS}")
    ablations = parse_csv_strings(args.ablation_modes)
    if any(item not in EVIDENCE_ABLATIONS for item in ablations):
        raise ValueError(f"ablation modes must be drawn from {EVIDENCE_ABLATIONS}")
    if "fixed_correct" not in mask_protocols or "full" not in ablations:
        raise ValueError("scientific diagnostic requires fixed_correct and full")

    datasets = {
        "train16": PPDProbeEvidenceDataset(
            args.probe_cache_manifest, indices=args.train_indices, eligible_only=True
        ),
        "fresh": PPDProbeEvidenceDataset(
            args.probe_cache_manifest, indices=args.fresh_indices, eligible_only=True
        ),
    }
    cache_hashes = {dataset.config_hash for dataset in datasets.values()}
    if len(cache_hashes) != 1:
        raise RuntimeError("train/fresh cache hashes differ")
    cache_hash = next(iter(cache_hashes))
    probes, checkpoint_schema = load_checkpoints(args, device, cache_hash)
    steps = sorted(probes)
    seeds = parse_csv_ints(args.seeds)
    t_values = parse_csv_floats(args.t_values)
    amp_name = str(checkpoint_schema["amp_dtype"])
    use_amp = amp_name != "none"
    amp_dtype = torch.float16 if amp_name == "fp16" else torch.bfloat16
    sampler, stock_flow, flow_schema = build_frozen_stock_flow(
        args.pretrained, device
    )

    metrics: dict[tuple[str, int, str, str, str], MetricAccumulator] = {}
    attribution: dict[tuple[str, int, str, str], MetricAccumulator] = {}
    mask_rows: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    zero_audit = {
        "physical_off_max_abs": 0.0,
        "null_evidence_max_abs": 0.0,
        "fixed_mask_neutral_max_abs": 0.0,
    }
    split_uids: dict[str, list[str]] = {}

    for split, dataset in datasets.items():
        count = len(dataset)
        if args.max_samples_per_split > 0:
            count = min(count, int(args.max_samples_per_split))
        split_uids[split] = [str(dataset.rows[index]["uid"]) for index in range(count)]
        for position in range(count):
            sample = dataset[position]
            uid = str(sample["uid"])
            object_uid = str(sample.get("object_uid", uid))
            target = sample["target"].unsqueeze(0).to(
                device=device, dtype=torch.float32
            )
            condition = sample["stock_condition"].to(device=device)
            correct_evidence = sample["ppd_correct_features"].unsqueeze(0).to(
                device=device, dtype=torch.float32
            )
            corrupt_evidence = {
                name: value.unsqueeze(0).to(device=device, dtype=torch.float32)
                for name, value in sample["ppd_corrupt_features"].items()
            }
            correct_mask = PPDLocalTargetProbe.active_mask(correct_evidence).to(
                device=device
            )
            corrupt_masks = {
                name: PPDLocalTargetProbe.active_mask(evidence).to(device=device)
                for name, evidence in corrupt_evidence.items()
            }
            regions = {
                name: mask_regions(correct_mask, mask)
                for name, mask in corrupt_masks.items()
            }
            for name, corrupt_mask in corrupt_masks.items():
                shared = regions[name]["shared"]
                union = regions[name]["union"]
                union_count = float(union.sum().item())
                overlap = {
                    "correct_active_ratio": float(correct_mask.mean().item()),
                    "corrupt_active_ratio": float(corrupt_mask.mean().item()),
                    "shared_active_ratio": float(shared.mean().item()),
                    "correct_only_ratio": float(
                        regions[name]["correct_only"].mean().item()
                    ),
                    "corrupt_only_ratio": float(
                        regions[name]["corrupt_only"].mean().item()
                    ),
                    "active_mask_iou": (
                        float(shared.sum().item()) / union_count
                        if union_count > 0.0
                        else 1.0
                    ),
                }
                for metric, value in overlap.items():
                    mask_rows[(split, name, metric)].append(value)

            ablated_correct = {
                mode: ablate_evidence(
                    correct_evidence,
                    mode,
                    reference_active_mask=correct_mask,
                )
                for mode in ablations
            }
            ablated_corrupt = {
                name: {
                    mode: ablate_evidence(
                        evidence,
                        mode,
                        reference_active_mask=correct_mask,
                    )
                    for mode in ablations
                }
                for name, evidence in corrupt_evidence.items()
            }

            for noise_seed in seeds:
                for t_value in t_values:
                    generator = torch.Generator(device=device).manual_seed(
                        deterministic_noise_seed(uid, noise_seed, t_value)
                    )
                    endpoint = torch.randn(
                        target.shape,
                        generator=generator,
                        device=device,
                        dtype=target.dtype,
                    )
                    x_t, gt_velocity = sampler._get_model_gt(
                        target, t_value, endpoint
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
                    stock_loss = F.mse_loss(stock.float(), gt_velocity.float()).clamp_min(
                        1.0e-6
                    )
                    target_residual = gt_velocity.float() - stock.float()

                    for step, probe in probes.items():
                        correct_gains: dict[tuple[str, str], float] = {}
                        for protocol in mask_protocols:
                            override = correct_mask if protocol == "fixed_correct" else None
                            for ablation in ablations:
                                with torch.cuda.amp.autocast(
                                    enabled=use_amp, dtype=amp_dtype
                                ):
                                    correct_delta, correct_stats = probe(
                                        x_t,
                                        stock,
                                        t_tensor,
                                        ablated_correct[ablation],
                                        scale=float(args.physical_scale),
                                        active_mask_override=override,
                                    )
                                correct_flow_loss = F.mse_loss(
                                    (stock + correct_delta).float(),
                                    gt_velocity.float(),
                                )
                                correct_gain = float(
                                    ((stock_loss - correct_flow_loss) / stock_loss).item()
                                )
                                correct_gains[(protocol, ablation)] = correct_gain
                                correct_local = masked_loss_or_none(
                                    correct_delta, target_residual, correct_mask
                                )
                                base = dict(
                                    split=split,
                                    step=step,
                                    protocol=protocol,
                                    ablation=ablation,
                                    object_uid=object_uid,
                                    t_value=t_value,
                                )
                                append_metric(
                                    metrics,
                                    metric="correct/relative_gain_vs_stock",
                                    value=correct_gain,
                                    **base,
                                )
                                append_metric(
                                    metrics,
                                    metric="correct/local_target_loss",
                                    value=correct_local,
                                    **base,
                                )
                                append_metric(
                                    metrics,
                                    metric="correct/local_target_gain",
                                    value=(
                                        None
                                        if correct_local is None
                                        else 1.0 - correct_local
                                    ),
                                    **base,
                                )
                                for metric in (
                                    "delta_rms",
                                    "delta_abs_max",
                                    "active_ratio",
                                    "neutral_abs_max",
                                ):
                                    append_metric(
                                        metrics,
                                        metric=f"correct/{metric}",
                                        value=float(correct_stats[metric].float().item()),
                                        **base,
                                    )

                                if (
                                    step == steps[-1]
                                    and protocol == "fixed_correct"
                                    and ablation == "full"
                                ):
                                    off_delta, _ = probe(
                                        x_t,
                                        stock,
                                        t_tensor,
                                        correct_evidence,
                                        physical_present=False,
                                    )
                                    null_delta, _ = probe(
                                        x_t,
                                        stock,
                                        t_tensor,
                                        make_null_evidence(correct_evidence),
                                    )
                                    zero_audit["physical_off_max_abs"] = max(
                                        zero_audit["physical_off_max_abs"],
                                        float(off_delta.float().abs().max().item()),
                                    )
                                    zero_audit["null_evidence_max_abs"] = max(
                                        zero_audit["null_evidence_max_abs"],
                                        float(null_delta.float().abs().max().item()),
                                    )
                                    zero_audit["fixed_mask_neutral_max_abs"] = max(
                                        zero_audit["fixed_mask_neutral_max_abs"],
                                        float(
                                            correct_stats["neutral_abs_max"].float().item()
                                        ),
                                    )

                                for name, evidence in ablated_corrupt.items():
                                    with torch.cuda.amp.autocast(
                                        enabled=use_amp, dtype=amp_dtype
                                    ):
                                        delta, stats = probe(
                                            x_t,
                                            stock,
                                            t_tensor,
                                            evidence[ablation],
                                            scale=float(args.physical_scale),
                                            active_mask_override=override,
                                        )
                                    flow_loss = F.mse_loss(
                                        (stock + delta).float(), gt_velocity.float()
                                    )
                                    corrupt_gain = float(
                                        ((stock_loss - flow_loss) / stock_loss).item()
                                    )
                                    advantage = float(
                                        ((flow_loss - correct_flow_loss) / stock_loss).item()
                                    )
                                    prefix = f"corruption/{name}"
                                    append_metric(
                                        metrics,
                                        metric=f"{prefix}/relative_gain_vs_stock",
                                        value=corrupt_gain,
                                        **base,
                                    )
                                    append_metric(
                                        metrics,
                                        metric=f"{prefix}/correct_advantage_flow",
                                        value=advantage,
                                        **base,
                                    )
                                    for metric in (
                                        "delta_rms",
                                        "delta_abs_max",
                                        "active_ratio",
                                        "neutral_abs_max",
                                    ):
                                        append_metric(
                                            metrics,
                                            metric=f"{prefix}/{metric}",
                                            value=float(stats[metric].float().item()),
                                            **base,
                                        )
                                    append_metric(
                                        metrics,
                                        metric=f"{prefix}/native_active_ratio",
                                        value=float(corrupt_masks[name].mean().item()),
                                        **base,
                                    )
                                    corrupt_correct_mask_loss = masked_loss_or_none(
                                        delta, target_residual, correct_mask
                                    )
                                    append_metric(
                                        metrics,
                                        metric=(
                                            f"{prefix}/local_target_loss_"
                                            "fixed_correct"
                                        ),
                                        value=corrupt_correct_mask_loss,
                                        **base,
                                    )
                                    for region_name, region_mask in {
                                        "fixed_correct": correct_mask,
                                        **regions[name],
                                    }.items():
                                        correct_region_loss = masked_loss_or_none(
                                            correct_delta, target_residual, region_mask
                                        )
                                        corrupt_region_loss = masked_loss_or_none(
                                            delta, target_residual, region_mask
                                        )
                                        local_advantage = (
                                            None
                                            if correct_region_loss is None
                                            or corrupt_region_loss is None
                                            else corrupt_region_loss
                                            - correct_region_loss
                                        )
                                        append_metric(
                                            metrics,
                                            metric=(
                                                f"{prefix}/correct_advantage_local_"
                                                f"{region_name}"
                                            ),
                                            value=local_advantage,
                                            **base,
                                        )

                        for protocol in mask_protocols:
                            pairs = {
                                "full_minus_active_mask_only": (
                                    "full",
                                    "active_mask_only",
                                ),
                                "point_only_minus_active_mask_only": (
                                    "point_only",
                                    "active_mask_only",
                                ),
                                "pose_depth_only_minus_active_mask_only": (
                                    "pose_depth_only",
                                    "active_mask_only",
                                ),
                                "full_minus_point_only": ("full", "point_only"),
                                "full_minus_pose_depth_only": (
                                    "full",
                                    "pose_depth_only",
                                ),
                            }
                            for name, (left, right) in pairs.items():
                                if (protocol, left) not in correct_gains or (
                                    protocol,
                                    right,
                                ) not in correct_gains:
                                    continue
                                key = (split, step, protocol, name)
                                attribution.setdefault(key, MetricAccumulator()).add(
                                    correct_gains[(protocol, left)]
                                    - correct_gains[(protocol, right)],
                                    object_uid=object_uid,
                                    t_value=t_value,
                                )
            print(
                f"[ppd3a_diagnostic] split={split} {position + 1}/{count} uid={uid}",
                flush=True,
            )

    results: dict[str, Any] = {}
    for split in datasets:
        results[split] = {}
        for step in steps:
            results[split][str(step)] = {}
            for protocol in mask_protocols:
                results[split][str(step)][protocol] = {}
                for ablation in ablations:
                    prefix = (split, step, protocol, ablation)
                    metric_reports = {
                        key[-1]: accumulator.report(
                            bootstrap_samples=args.bootstrap_samples
                        )
                        for key, accumulator in metrics.items()
                        if key[:4] == prefix
                    }
                    corruptions = {}
                    for name in PROBE_CORRUPTIONS:
                        marker = f"corruption/{name}/"
                        corruptions[name] = {
                            metric[len(marker) :]: value
                            for metric, value in metric_reports.items()
                            if metric.startswith(marker)
                        }
                    row = {
                        "metrics": {
                            metric: value
                            for metric, value in metric_reports.items()
                            if not metric.startswith("corruption/")
                        },
                        "corruptions": corruptions,
                    }
                    row["decision"] = decision_for(row, args)
                    results[split][str(step)][protocol][ablation] = row

    attribution_report: dict[str, Any] = {}
    for (split, step, protocol, name), accumulator in attribution.items():
        attribution_report.setdefault(split, {}).setdefault(
            str(step), {}
        ).setdefault(protocol, {})[name] = accumulator.report(
            bootstrap_samples=args.bootstrap_samples
        )

    primary_pass = {
        split: {
            str(step): results[split][str(step)]["fixed_correct"]["full"][
                "decision"
            ]["passed"]
            for step in steps
        }
        for split in results
    }
    paired_steps = {
        str(step): all(primary_pass[split][str(step)] for split in results)
        for step in steps
    }
    report = {
        "stage": "PPD-3A checkpoint sweep with controlled masks and ablations",
        "passed": any(paired_steps.values()),
        "protocol": {
            "probe_cache_manifest": str(Path(args.probe_cache_manifest).resolve()),
            "cache_config_hash": cache_hash,
            "checkpoint_dir": str(Path(args.checkpoint_dir).resolve()),
            "checkpoint_steps": steps,
            "checkpoint_schema": checkpoint_schema,
            "runtime_probe_schema": probes[steps[-1]].metadata(),
            "noise_seeds": seeds,
            "t_values": t_values,
            "physical_scale": float(args.physical_scale),
            "mask_protocols": mask_protocols,
            "primary_mask_protocol": "fixed_correct",
            "ablation_modes": ablations,
            "primary_ablation": "full",
            "corruptions": list(PROBE_CORRUPTIONS),
            "flow": flow_schema,
            "split_uids": split_uids,
            "split_uid_hashes": {
                split: sha256_strings(uids) for split, uids in split_uids.items()
            },
        },
        "stock_preservation": zero_audit,
        "mask_overlap": summarize_mask_overlap(mask_rows),
        "primary_pass_by_split_step": primary_pass,
        "paired_train_fresh_pass_by_step": paired_steps,
        "results": results,
        "attribution": attribution_report,
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
                "primary_pass_by_split_step": primary_pass,
                "paired_train_fresh_pass_by_step": paired_steps,
                "stock_preservation": zero_audit,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
