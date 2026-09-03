#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import mean, median
import sys
from typing import Any

import numpy as np
from scipy.ndimage import label
import torch
import torch.nn.functional as F


TRACKER_ROOT = Path(__file__).resolve().parents[1]
RECONVIAGEN_ROOT = TRACKER_ROOT / "ReconViaGen"
VGGT_ROOT = RECONVIAGEN_ROOT / "wheels" / "vggt"
for path in (TRACKER_ROOT, RECONVIAGEN_ROOT, VGGT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ar_ss_flow.local_pose_lifting_flow import PoseLiftingCacheDataset  # noqa: E402
from pose_aligned_reconstruction.direct_flow import (  # noqa: E402
    DIRECT_CORRUPTION_MODES,
    DIRECT_FLOW_VERSION,
    NativeStockFlow,
    PositivePhysicalRolloutFlow,
    lifting_cache_identity,
    load_frozen_correspondence_head,
    make_direct_evidence_bundle,
    null_evidence_like,
    parse_csv,
    validate_n3_checkpoint,
)
from pose_aligned_reconstruction.train_direct_flow import (  # noqa: E402
    build_direct_components,
)
from reconvggt_ar_adapter_a.pointpose_ss_condition import (  # noqa: E402
    load_partial_state,
)


def parse_int_csv(text: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in str(text).split(",") if item.strip())
    if not values:
        raise ValueError("integer CSV must be non-empty")
    return values


def parse_float_csv(text: str) -> tuple[float, ...]:
    values = tuple(
        float(item.strip()) for item in str(text).split(",") if item.strip()
    )
    if not values or any(not 0.0 < value < 1.0 for value in values):
        raise ValueError("t values must lie in (0,1)")
    return values


def summarize(values: list[float]) -> dict[str, float | int]:
    finite = [float(value) for value in values if np.isfinite(value)]
    if not finite:
        return {"count": 0, "mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
    return {
        "count": len(finite),
        "mean": mean(finite),
        "median": median(finite),
        "min": min(finite),
        "max": max(finite),
    }


def positive_rate(values: list[float]) -> float:
    return float(mean(value > 0.0 for value in values)) if values else 0.0


def bootstrap_mean_ci(
    values: list[float],
    *,
    samples: int,
    seed: int,
) -> list[float]:
    finite = np.asarray(
        [float(value) for value in values if np.isfinite(value)],
        dtype=np.float64,
    )
    if finite.size == 0 or int(samples) <= 0:
        return [0.0, 0.0]
    if finite.size == 1:
        value = float(finite[0])
        return [value, value]
    generator = np.random.default_rng(int(seed))
    means = np.empty(int(samples), dtype=np.float64)
    for index in range(int(samples)):
        draw = generator.choice(finite, size=finite.size, replace=True)
        means[index] = draw.mean()
    return [float(value) for value in np.quantile(means, [0.025, 0.975])]


def summarize_formal(
    values: list[float],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    output = summarize(values)
    output["bootstrap_mean_95_ci"] = bootstrap_mean_ci(
        values,
        samples=bootstrap_samples,
        seed=bootstrap_seed,
    )
    return output


def coord_set(coords: np.ndarray) -> set[tuple[int, int, int]]:
    return {tuple(int(value) for value in row[-3:]) for row in coords}


def overlap_metrics(
    predicted: np.ndarray, target: np.ndarray
) -> dict[str, float | int]:
    pred = coord_set(predicted)
    gt = coord_set(target)
    intersection = len(pred & gt)
    union = len(pred | gt)
    return {
        "coord_count": len(pred),
        "target_coord_count": len(gt),
        "iou": intersection / max(union, 1),
        "precision": intersection / max(len(pred), 1),
        "recall": intersection / max(len(gt), 1),
        "coord_count_ratio": len(pred) / max(len(gt), 1),
    }


def component_metrics(coords: np.ndarray) -> dict[str, float | int]:
    occupancy = np.zeros((64, 64, 64), dtype=np.uint8)
    if coords.size:
        xyz = coords[:, -3:].astype(np.int64)
        valid = np.all((xyz >= 0) & (xyz < 64), axis=1)
        xyz = xyz[valid]
        occupancy[xyz[:, 0], xyz[:, 1], xyz[:, 2]] = 1
    components, count = label(occupancy)
    total = int(occupancy.sum())
    if count <= 0 or total <= 0:
        return {"component_count": 0, "largest_component_ratio": 0.0}
    sizes = np.bincount(components.reshape(-1))[1:]
    return {
        "component_count": int(count),
        "largest_component_ratio": float(sizes.max() / total),
    }


@torch.no_grad()
def decode_coords(decoder: torch.nn.Module, latent: torch.Tensor) -> np.ndarray:
    dtype = next(decoder.parameters()).dtype
    logits = decoder(latent.to(dtype=dtype)).float()
    return (
        torch.argwhere(logits > 0)[:, [0, 2, 3, 4]]
        .cpu()
        .numpy()
        .astype(np.int32)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Object-balanced teacher-forced and rollout evaluation for the "
            "large-data direct physical SS Flow."
        )
    )
    parser.add_argument("--cache_manifest", required=True)
    parser.add_argument("--flow_checkpoint", required=True)
    parser.add_argument("--correspondence_checkpoint", required=True)
    parser.add_argument("--n3_report", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--pretrained", default="Stable-X/trellis-vggt-v0-2")
    parser.add_argument("--indices", default="all")
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--t_values", default="0.1,0.3,0.5,0.7,0.9")
    parser.add_argument(
        "--corruption_modes",
        default="pose_cyclic1,depth_view_cyclic1,visual_view_cyclic1",
    )
    parser.add_argument("--physical_scale", type=float, default=None)
    parser.add_argument(
        "--amp_dtype", choices=("bf16", "fp16", "none"), default="bf16"
    )
    parser.add_argument("--allow_object_overlap", action="store_true")
    parser.add_argument("--decode_teacher", action="store_true")
    parser.add_argument("--benchmark_min_object_win", type=float, default=0.55)
    parser.add_argument("--benchmark_min_positive_t", type=int, default=3)
    parser.add_argument("--mechanism_min_object_win", type=float, default=0.55)
    parser.add_argument("--mechanism_min_positive_t", type=int, default=4)
    parser.add_argument("--max_corrupt_mean_abs_gain", type=float, default=0.01)
    parser.add_argument("--bootstrap_samples", type=int, default=2000)
    parser.add_argument(
        "--decision_profile",
        choices=("report_only", "benchmark_relaxed", "mechanism_strict"),
        default="report_only",
    )
    parser.add_argument("--fail_on_science", action="store_true")
    parser.add_argument("--rollout_steps", type=int, default=0)
    parser.add_argument("--rollout_max_samples", type=int, default=4)
    parser.add_argument("--rollout_seeds", default="42")
    parser.add_argument("--rollout_controls", default="pose_cyclic1,depth_view_cyclic1")
    parser.add_argument("--benchmark_require_rollout", action="store_true")
    parser.add_argument("--max_rollout_iou_degradation", type=float, default=0.01)
    parser.add_argument(
        "--max_rollout_precision_degradation", type=float, default=0.01
    )
    parser.add_argument(
        "--max_rollout_largest_component_degradation",
        type=float,
        default=0.02,
    )
    parser.add_argument("--cfg_strength", type=float, default=7.5)
    parser.add_argument("--guidance_rescale", type=float, default=0.5)
    parser.add_argument("--rescale_t", type=float, default=3.0)
    return parser.parse_args()


def validate_eval_split(
    checkpoint: dict[str, Any],
    eval_dataset: PoseLiftingCacheDataset,
    *,
    allow_overlap: bool,
) -> dict[str, Any]:
    train_manifest = checkpoint.get("args", {}).get("cache_manifest")
    if not train_manifest:
        raise ValueError("Flow checkpoint is missing training cache manifest")
    train_dataset = PoseLiftingCacheDataset(
        train_manifest,
        indices=str(checkpoint.get("args", {}).get("indices", "all")),
    )
    saved_identity = checkpoint.get("model_summary", {}).get("data_identity")
    if not isinstance(saved_identity, dict):
        raise ValueError("Flow checkpoint is missing bound training data identity")
    runtime_train_identity = lifting_cache_identity(
        train_manifest,
        rows=train_dataset.rows,
    )
    for name in (
        "manifest_sha256",
        "cache_schema_hash",
        "uid_hash",
        "object_uid_hash",
        "source_cache_manifest_sha256",
    ):
        if runtime_train_identity.get(name) != saved_identity.get(name):
            raise RuntimeError(
                f"training cache identity changed after checkpoint: field={name}"
            )
    eval_identity = lifting_cache_identity(
        eval_dataset.manifest_path,
        rows=eval_dataset.rows,
    )
    if (
        eval_identity["cache_schema_hash"]
        != saved_identity["cache_schema_hash"]
    ):
        raise RuntimeError("training/evaluation lifting cache schemas differ")
    train_objects = {
        str(row.get("object_uid", row["uid"])) for row in train_dataset.rows
    }
    eval_objects = {
        str(row.get("object_uid", row["uid"])) for row in eval_dataset.rows
    }
    overlap = sorted(train_objects & eval_objects)
    if overlap and not allow_overlap:
        raise RuntimeError(
            f"evaluation is not object-disjoint; overlap count={len(overlap)}"
        )
    return {
        "training_manifest": str(Path(train_manifest).resolve()),
        "training_object_count": len(train_objects),
        "evaluation_object_count": len(eval_objects),
        "object_overlap_count": len(overlap),
        "object_disjoint": not overlap,
        "overlap_allowed": bool(allow_overlap),
        "cache_schema_match": True,
        "training_data_identity": runtime_train_identity,
        "evaluation_data_identity": eval_identity,
    }


def object_balanced_rows(
    records: list[dict[str, Any]],
    *,
    include_t: bool = False,
) -> list[dict[str, Any]]:
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        key: tuple[Any, ...] = (
            str(row["object_uid"]),
            str(row["branch"]),
        )
        if include_t:
            key = (*key, float(row["t"]))
        buckets[key].append(row)
    output = []
    for key, rows in sorted(buckets.items()):
        item = {
            "object_uid": key[0],
            "branch": key[1],
            "gain_vs_stock": mean(float(row["gain_vs_stock"]) for row in rows),
            "correct_advantage": mean(
                float(row["correct_advantage"]) for row in rows
            ),
        }
        if include_t:
            item["t"] = float(key[2])
        output.append(item)
    return output


def aggregate_teacher(
    records: list[dict[str, Any]],
    *,
    branches: tuple[str, ...],
    t_values: tuple[float, ...],
    bootstrap_samples: int,
) -> tuple[dict[str, Any], dict[str, Any], int, dict[str, Any]]:
    object_rows = object_balanced_rows(records)
    branch_summary = {}
    for branch_index, branch in enumerate(branches):
        rows = [row for row in object_rows if row["branch"] == branch]
        gains = [float(row["gain_vs_stock"]) for row in rows]
        advantages = [float(row["correct_advantage"]) for row in rows]
        branch_summary[branch] = {
            "gain_vs_stock": summarize_formal(
                gains,
                bootstrap_samples=bootstrap_samples,
                bootstrap_seed=42000 + branch_index,
            ),
            "stock_object_win_rate": positive_rate(gains),
            "correct_advantage": summarize_formal(
                advantages,
                bootstrap_samples=bootstrap_samples,
                bootstrap_seed=43000 + branch_index,
            ),
            "correct_object_win_rate": positive_rate(advantages),
        }
    object_t_rows = object_balanced_rows(records, include_t=True)
    t_summary = {}
    positive_t = 0
    for t_index, t_value in enumerate(t_values):
        values = [
            float(row["gain_vs_stock"])
            for row in object_t_rows
            if row["branch"] == "correct" and row["t"] == t_value
        ]
        t_summary[str(t_value)] = summarize_formal(
            values,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=44000 + t_index,
        )
        positive_t += int(float(t_summary[str(t_value)]["mean"]) > 0.0)
    sequence_weighted = {}
    for branch in branches:
        rows = [row for row in records if row["branch"] == branch]
        sequence_weighted[branch] = {
            "gain_vs_stock": summarize(
                [float(row["gain_vs_stock"]) for row in rows]
            ),
            "correct_advantage": summarize(
                [float(row["correct_advantage"]) for row in rows]
            ),
        }
    return branch_summary, t_summary, positive_t, sequence_weighted


def paired_rollout_delta_values(
    rows: list[dict[str, Any]],
    *,
    branch: str,
    metric: str,
) -> tuple[list[float], list[float]]:
    stock_rows = [row for row in rows if row["branch"] == "stock"]
    stock_keys = [(str(row["uid"]), int(row["seed"])) for row in stock_rows]
    if len(stock_keys) != len(set(stock_keys)):
        raise RuntimeError("rollout stock rows contain duplicate (uid, seed) keys")
    stock_by_key = dict(zip(stock_keys, stock_rows))
    branch_rows = [row for row in rows if row["branch"] == branch]
    branch_keys = [(str(row["uid"]), int(row["seed"])) for row in branch_rows]
    if len(branch_keys) != len(set(branch_keys)):
        raise RuntimeError(
            f"rollout branch={branch} contains duplicate (uid, seed) keys"
        )
    by_object: dict[str, list[float]] = defaultdict(list)
    raw_values = []
    for row, key in zip(branch_rows, branch_keys):
        if key not in stock_by_key:
            raise RuntimeError(f"rollout branch is missing stock pair: {key}")
        value = float(row[metric]) - float(stock_by_key[key][metric])
        raw_values.append(value)
        by_object[str(row["object_uid"])].append(value)
    return [mean(items) for items in by_object.values()], raw_values


@torch.no_grad()
def run_rollout(
    *,
    args: argparse.Namespace,
    dataset: PoseLiftingCacheDataset,
    count: int,
    model,
    decoder,
    sampler,
    sampler_params: dict[str, Any],
    correspondence_head,
    correspondence_runtime: dict[str, Any],
    device: torch.device,
    physical_scale: float,
    use_amp: bool,
    amp_dtype: torch.dtype,
) -> dict[str, Any] | None:
    if int(args.rollout_steps) <= 0:
        return None
    if decoder is None:
        raise RuntimeError("rollout requires the frozen SS decoder")
    controls = parse_csv(args.rollout_controls)
    invalid = [mode for mode in controls if mode not in DIRECT_CORRUPTION_MODES]
    if invalid:
        raise ValueError(f"invalid rollout controls={invalid}")
    params = dict(sampler_params)
    params.update(
        {
            "steps": int(args.rollout_steps),
            "cfg_strength": float(args.cfg_strength),
            "guidance_rescale": float(args.guidance_rescale),
            "rescale_t": float(args.rescale_t),
        }
    )
    rows = []
    rollout_count = min(count, max(int(args.rollout_max_samples), 0))
    seeds = parse_int_csv(args.rollout_seeds)
    for sample_index in range(rollout_count):
        sample = dataset[sample_index]
        evidence = make_direct_evidence_bundle(
            sample,
            modes=controls,
            device=device,
            correspondence_head=correspondence_head,
            correspondence_runtime=correspondence_runtime,
        )
        condition = sample["stock_condition"].to(device=device)
        negative_condition = torch.zeros_like(condition)
        target_coords = sample["target_coords"].numpy().astype(np.int32)
        for seed in seeds:
            generator = torch.Generator(device=device).manual_seed(
                int(seed) * 1000003 + sample_index * 1009
            )
            noise = torch.randn(
                (1, 8, 16, 16, 16), generator=generator, device=device
            )
            with torch.autocast(
                device_type="cuda", dtype=amp_dtype, enabled=use_amp
            ):
                stock_latent = sampler.sample(
                    NativeStockFlow(model),
                    noise.clone(),
                    cond=condition,
                    neg_cond=negative_condition,
                    **params,
                    verbose=False,
                ).samples
            branch_latents = {"stock": stock_latent}
            call_audits = {}
            for branch in ("correct", *controls):
                wrapper = PositivePhysicalRolloutFlow(
                    model,
                    condition,
                    evidence[branch][:4],
                    physical_scale=physical_scale,
                )
                with torch.autocast(
                    device_type="cuda", dtype=amp_dtype, enabled=use_amp
                ):
                    branch_latents[branch] = sampler.sample(
                        wrapper,
                        noise.clone(),
                        cond=condition,
                        neg_cond=negative_condition,
                        **params,
                        verbose=False,
                    ).samples
                if wrapper.positive_calls <= 0:
                    raise RuntimeError(f"rollout branch={branch} never used physical path")
                call_audits[branch] = {
                    "positive_calls": wrapper.positive_calls,
                    "negative_calls": wrapper.negative_calls,
                }
            for branch, latent in branch_latents.items():
                coords = decode_coords(decoder, latent)
                rows.append(
                    {
                        "uid": str(sample["uid"]),
                        "object_uid": str(sample.get("object_uid", sample["uid"])),
                        "views": int(evidence["correct"][4]["views"]),
                        "seed": int(seed),
                        "branch": branch,
                        **overlap_metrics(coords, target_coords),
                        **component_metrics(coords),
                        "cfg_calls": call_audits.get(branch),
                    }
                )
        print(f"[direct_flow_rollout] {sample_index + 1}/{rollout_count}", flush=True)
    branches = ("stock", "correct", *controls)
    metrics = (
        "iou",
        "precision",
        "recall",
        "coord_count_ratio",
        "component_count",
        "largest_component_ratio",
    )
    object_branch_rows: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        object_branch_rows[(str(row["object_uid"]), str(row["branch"]))].append(row)
    object_values: dict[str, dict[str, list[float]]] = {
        branch: {metric: [] for metric in metrics} for branch in branches
    }
    for (_, branch), branch_rows in object_branch_rows.items():
        for metric in metrics:
            object_values[branch][metric].append(
                mean(float(row[metric]) for row in branch_rows)
            )
    summary = {
        branch: {
            metric: summarize_formal(
                object_values[branch][metric],
                bootstrap_samples=int(args.bootstrap_samples),
                bootstrap_seed=51000 + branch_index * 100 + metric_index,
            )
            for metric_index, metric in enumerate(metrics)
        }
        for branch_index, branch in enumerate(branches)
    }
    sequence_weighted_summary = {
        branch: {
            metric: summarize(
                [float(row[metric]) for row in rows if row["branch"] == branch]
            )
            for metric in metrics
        }
        for branch in branches
    }
    delta = {}
    sequence_weighted_delta = {}
    for branch_index, branch in enumerate(branches[1:]):
        delta[branch] = {}
        sequence_weighted_delta[branch] = {}
        for metric_index, metric in enumerate((
            "iou",
            "precision",
            "recall",
            "largest_component_ratio",
        )):
            values, raw_values = paired_rollout_delta_values(
                rows,
                branch=branch,
                metric=metric,
            )
            delta[branch][metric] = {
                **summarize_formal(
                    values,
                    bootstrap_samples=int(args.bootstrap_samples),
                    bootstrap_seed=52000 + branch_index * 100 + metric_index,
                ),
                "object_win_rate": positive_rate(values),
            }
            sequence_weighted_delta[branch][metric] = {
                **summarize(raw_values),
                "pair_win_rate": positive_rate(raw_values),
            }
    return {
        "sample_count": rollout_count,
        "seeds": list(seeds),
        "steps": int(args.rollout_steps),
        "formal_weighting": "uid-seed paired then averaged per object",
        "summary": summary,
        "delta_vs_stock": delta,
        "sequence_weighted_summary": sequence_weighted_summary,
        "sequence_weighted_delta_vs_stock": sequence_weighted_delta,
        "records": rows,
    }


def report_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Direct physical SS Flow evaluation",
        "",
        f"- selected profile: `{report['selected_decision_profile']}`",
        f"- selected passed: `{report['selected_passed']}`",
        f"- benchmark relaxed: `{report['benchmark_relaxed']['passed']}`",
        f"- mechanism strict: `{report['mechanism_strict']['passed']}`",
        f"- samples / objects: `{report['sample_count']}` / `{report['object_count']}`",
        f"- object disjoint: `{report['split_audit']['object_disjoint']}`",
        "",
        "## Teacher-forced",
        "",
        "| branch | gain vs stock mean | median | stock win | correct advantage mean | correct win |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for branch, row in report["branch_summary"].items():
        lines.append(
            "| {branch} | {gain:.6g} | {median:.6g} | {stock_win:.3f} | "
            "{adv:.6g} | {correct_win:.3f} |".format(
                branch=branch,
                gain=row["gain_vs_stock"]["mean"],
                median=row["gain_vs_stock"]["median"],
                stock_win=row["stock_object_win_rate"],
                adv=row["correct_advantage"]["mean"],
                correct_win=row["correct_object_win_rate"],
            )
        )
    lines.extend(
        [
            "",
            "`benchmark_relaxed` is the baseline comparison. `mechanism_strict` "
            "additionally requires the correct pose/depth/visual binding to beat all corruptions.",
        ]
    )
    return "\n".join(lines) + "\n"


@torch.no_grad()
def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("direct Flow evaluation requires CUDA")
    torch.cuda.set_device(0 if device.index is None else int(device.index))
    seeds = parse_int_csv(args.seeds)
    t_values = parse_float_csv(args.t_values)
    controls = parse_csv(args.corruption_modes)
    invalid = [mode for mode in controls if mode not in DIRECT_CORRUPTION_MODES]
    if invalid:
        raise ValueError(f"invalid controls={invalid}")
    if int(args.bootstrap_samples) < 0:
        raise ValueError("bootstrap_samples must be non-negative")
    degradation_limits = (
        float(args.max_rollout_iou_degradation),
        float(args.max_rollout_precision_degradation),
        float(args.max_rollout_largest_component_degradation),
    )
    if any(value < 0.0 for value in degradation_limits):
        raise ValueError("rollout degradation limits must be non-negative")
    if args.decision_profile == "mechanism_strict" and set(controls) != set(
        DIRECT_CORRUPTION_MODES
    ):
        raise ValueError(
            "mechanism_strict requires pose, depth, and visual corruption controls"
        )

    checkpoint_path = Path(args.flow_checkpoint).resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if checkpoint.get("format") != DIRECT_FLOW_VERSION:
        raise ValueError(f"unexpected direct Flow checkpoint={checkpoint.get('format')!r}")
    saved_args = checkpoint["args"]
    if str(saved_args.get("pretrained")) != str(args.pretrained):
        raise ValueError("evaluation pretrained model differs from Flow checkpoint")
    for field in ("lora_rank", "lora_alpha", "physical_hidden_dim"):
        if field not in saved_args:
            raise ValueError(f"checkpoint is missing architecture field={field}")

    dataset = PoseLiftingCacheDataset(args.cache_manifest, indices=args.indices)
    count = (
        len(dataset)
        if int(args.max_samples) <= 0
        else min(len(dataset), int(args.max_samples))
    )
    if count <= 0:
        raise ValueError("evaluation subset is empty")
    split_audit = validate_eval_split(
        checkpoint, dataset, allow_overlap=bool(args.allow_object_overlap)
    )
    n3_audit = validate_n3_checkpoint(
        args.n3_report, args.correspondence_checkpoint
    )
    correspondence_head, _, correspondence_runtime = load_frozen_correspondence_head(
        args.correspondence_checkpoint,
        device=device,
        visual_channels=dataset.visual_feature_dim,
    )
    if correspondence_runtime["checkpoint_sha256"] != n3_audit["checkpoint_sha256"]:
        raise RuntimeError("runtime C0 checkpoint is not the N3-bound checkpoint")
    saved_correspondence = checkpoint.get("model_summary", {}).get(
        "correspondence", {}
    )
    if (
        correspondence_runtime["checkpoint_sha256"]
        != saved_correspondence.get("checkpoint_sha256")
    ):
        raise RuntimeError(
            "evaluation correspondence checkpoint differs from Flow training"
        )
    need_decoder = bool(args.decode_teacher) or int(args.rollout_steps) > 0
    sampler, model, decoder, model_summary, sampler_params = build_direct_components(
        pretrained=args.pretrained,
        visual_channels=dataset.visual_feature_dim,
        physical_hidden_dim=int(saved_args["physical_hidden_dim"]),
        lora_rank=int(saved_args["lora_rank"]),
        lora_alpha=int(saved_args["lora_alpha"]),
        gradient_checkpointing=False,
        need_decoder=need_decoder,
        device=device,
    )
    saved_encoder = checkpoint.get("model_summary", {}).get("physical_encoder")
    if model_summary.get("physical_encoder") != saved_encoder:
        raise RuntimeError(
            "evaluation physical encoder metadata differs from Flow checkpoint"
        )
    load_partial_state(
        model,
        checkpoint["model_trainable_state"],
        require_all_trainable=True,
    )
    model.eval()
    use_amp = args.amp_dtype != "none"
    amp_dtype = torch.float16 if args.amp_dtype == "fp16" else torch.bfloat16
    physical_scale = (
        float(saved_args.get("physical_scale", 1.0))
        if args.physical_scale is None
        else float(args.physical_scale)
    )

    records = []
    occupancy_records = []
    off_max_abs = 0.0
    null_max_abs = 0.0
    for sample_index in range(count):
        sample = dataset[sample_index]
        evidence = make_direct_evidence_bundle(
            sample,
            modes=controls,
            device=device,
            correspondence_head=correspondence_head,
            correspondence_runtime=correspondence_runtime,
        )
        target = sample["target"].unsqueeze(0).to(device=device, dtype=torch.float32)
        condition = sample["stock_condition"].to(device=device)
        target_coords = sample["target_coords"].numpy().astype(np.int32)
        for seed in seeds:
            for t_value in t_values:
                generator = torch.Generator(device=device).manual_seed(
                    int(seed) * 1000003
                    + sample_index * 1009
                    + int(round(t_value * 1000.0))
                )
                noise = torch.randn(target.shape, generator=generator, device=device)
                x_t, gt_velocity = sampler._get_model_gt(target, t_value, noise)
                t_tensor = torch.full(
                    (1,), 1000.0 * t_value, device=device, dtype=torch.float32
                )
                stock = model.stock_prediction(x_t, t_tensor, condition)
                predictions = {}
                with torch.autocast(
                    device_type="cuda", dtype=amp_dtype, enabled=use_amp
                ):
                    for branch in ("correct", *controls):
                        predictions[branch], _ = model.conditioned_prediction(
                            x_t,
                            t_tensor,
                            condition,
                            *evidence[branch][:4],
                            stock_velocity=stock,
                            physical_scale=physical_scale,
                        )
                if seed == seeds[0] and t_value == t_values[0]:
                    disabled, _ = model.conditioned_prediction(
                        x_t,
                        t_tensor,
                        condition,
                        *evidence["correct"][:4],
                        stock_velocity=stock,
                        physical_present=False,
                    )
                    null, _ = model.conditioned_prediction(
                        x_t,
                        t_tensor,
                        condition,
                        *null_evidence_like(evidence["correct"]),
                        stock_velocity=stock,
                    )
                    off_max_abs = max(
                        off_max_abs, float((disabled - stock).abs().max().item())
                    )
                    null_max_abs = max(
                        null_max_abs, float((null - stock).abs().max().item())
                    )
                stock_loss = float(
                    F.mse_loss(stock.float(), gt_velocity.float()).item()
                )
                branch_losses = {
                    branch: float(
                        F.mse_loss(prediction.float(), gt_velocity.float()).item()
                    )
                    for branch, prediction in predictions.items()
                }
                correct_loss = branch_losses["correct"]
                for branch, branch_loss in branch_losses.items():
                    records.append(
                        {
                            "sample_index": sample_index,
                            "uid": str(sample["uid"]),
                            "object_uid": str(sample.get("object_uid", sample["uid"])),
                            "views": int(evidence["correct"][4]["views"]),
                            "seed": int(seed),
                            "t": float(t_value),
                            "branch": branch,
                            "stock_loss": stock_loss,
                            "branch_loss": branch_loss,
                            "gain_vs_stock": (stock_loss - branch_loss)
                            / max(stock_loss, 1.0e-8),
                            "correct_advantage": (branch_loss - correct_loss)
                            / max(stock_loss, 1.0e-8),
                        }
                    )
                if args.decode_teacher and decoder is not None:
                    for branch, velocity in (
                        ("stock", stock),
                        ("correct", predictions["correct"]),
                    ):
                        latent = sampler._pred_to_xstart(x_t, t_value, velocity)
                        coords = decode_coords(decoder, latent)
                        occupancy_records.append(
                            {
                                "uid": str(sample["uid"]),
                                "object_uid": str(
                                    sample.get("object_uid", sample["uid"])
                                ),
                                "seed": int(seed),
                                "t": float(t_value),
                                "branch": branch,
                                **overlap_metrics(coords, target_coords),
                                **component_metrics(coords),
                            }
                        )
        print(f"[direct_flow_eval] {sample_index + 1}/{count}", flush=True)
        torch.cuda.empty_cache()

    branches = ("correct", *controls)
    branch_summary, t_summary, positive_t, teacher_sequence_weighted = aggregate_teacher(
        records,
        branches=branches,
        t_values=t_values,
        bootstrap_samples=int(args.bootstrap_samples),
    )
    correct = branch_summary["correct"]
    benchmark_checks = {
        "stock_exact_off_and_null": off_max_abs == 0.0 and null_max_abs == 0.0,
        "correct_mean_gain_positive": float(correct["gain_vs_stock"]["mean"]) > 0.0,
        "correct_median_gain_positive": float(correct["gain_vs_stock"]["median"]) > 0.0,
        "correct_object_win": float(correct["stock_object_win_rate"])
        >= float(args.benchmark_min_object_win),
        "positive_t_count": positive_t >= int(args.benchmark_min_positive_t),
    }
    mechanism_checks = dict(benchmark_checks)
    mechanism_checks["strict_positive_t_count"] = positive_t >= int(
        args.mechanism_min_positive_t
    )
    for branch in controls:
        row = branch_summary[branch]
        mechanism_checks[f"correct_mean_vs_{branch}"] = float(
            row["correct_advantage"]["mean"]
        ) > 0.0
        mechanism_checks[f"correct_median_vs_{branch}"] = float(
            row["correct_advantage"]["median"]
        ) > 0.0
        mechanism_checks[f"correct_win_vs_{branch}"] = float(
            row["correct_object_win_rate"]
        ) >= float(args.mechanism_min_object_win)
        mechanism_checks[f"{branch}_near_stock"] = abs(
            float(row["gain_vs_stock"]["mean"])
        ) <= float(args.max_corrupt_mean_abs_gain)

    occupancy_summary = None
    if occupancy_records:
        occupancy_summary = {
            "formal_weighting": "averaged per object",
            "object_balanced": {},
            "sequence_weighted": {},
        }
        for branch in ("stock", "correct"):
            rows = [row for row in occupancy_records if row["branch"] == branch]
            by_object: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in rows:
                by_object[str(row["object_uid"])].append(row)
            occupancy_summary["object_balanced"][branch] = {
                metric: summarize_formal(
                    [
                        mean(float(item[metric]) for item in object_rows)
                        for object_rows in by_object.values()
                    ],
                    bootstrap_samples=int(args.bootstrap_samples),
                    bootstrap_seed=61000 + metric_index,
                )
                for metric_index, metric in enumerate(
                    (
                        "iou",
                        "precision",
                        "recall",
                        "coord_count_ratio",
                        "component_count",
                        "largest_component_ratio",
                    )
                )
            }
            occupancy_summary["sequence_weighted"][branch] = {
                metric: summarize([float(row[metric]) for row in rows])
                for metric in (
                    "iou",
                    "precision",
                    "recall",
                    "coord_count_ratio",
                    "component_count",
                    "largest_component_ratio",
                )
            }
        stock_occupancy_rows = [
            row for row in occupancy_records if row["branch"] == "stock"
        ]
        stock_occupancy_by_key = {
            (str(row["uid"]), int(row["seed"]), float(row["t"])): row
            for row in stock_occupancy_rows
        }
        if len(stock_occupancy_by_key) != len(stock_occupancy_rows):
            raise RuntimeError("teacher occupancy has duplicate stock pairing keys")
        occupancy_summary["correct_delta_vs_stock"] = {}
        for metric_index, metric in enumerate(
            ("iou", "precision", "recall", "largest_component_ratio")
        ):
            by_object_delta: dict[str, list[float]] = defaultdict(list)
            for row in occupancy_records:
                if row["branch"] != "correct":
                    continue
                key = (str(row["uid"]), int(row["seed"]), float(row["t"]))
                if key not in stock_occupancy_by_key:
                    raise RuntimeError(
                        f"teacher occupancy is missing stock pair: {key}"
                    )
                by_object_delta[str(row["object_uid"])].append(
                    float(row[metric])
                    - float(stock_occupancy_by_key[key][metric])
                )
            values = [mean(items) for items in by_object_delta.values()]
            occupancy_summary["correct_delta_vs_stock"][metric] = {
                **summarize_formal(
                    values,
                    bootstrap_samples=int(args.bootstrap_samples),
                    bootstrap_seed=62000 + metric_index,
                ),
                "object_win_rate": positive_rate(values),
            }

    rollout = run_rollout(
        args=args,
        dataset=dataset,
        count=count,
        model=model,
        decoder=decoder,
        sampler=sampler,
        sampler_params=sampler_params,
        correspondence_head=correspondence_head,
        correspondence_runtime=correspondence_runtime,
        device=device,
        physical_scale=physical_scale,
        use_amp=use_amp,
        amp_dtype=amp_dtype,
    )
    if args.benchmark_require_rollout:
        benchmark_checks["rollout_available"] = rollout is not None
        if rollout is not None:
            correct_rollout = rollout["delta_vs_stock"]["correct"]
            benchmark_checks["rollout_iou_non_degrading"] = float(
                correct_rollout["iou"]["mean"]
            ) >= -float(args.max_rollout_iou_degradation)
            benchmark_checks["rollout_precision_non_degrading"] = float(
                correct_rollout["precision"]["mean"]
            ) >= -float(args.max_rollout_precision_degradation)
            benchmark_checks[
                "rollout_largest_component_non_degrading"
            ] = float(
                correct_rollout["largest_component_ratio"]["mean"]
            ) >= -float(args.max_rollout_largest_component_degradation)
        else:
            benchmark_checks["rollout_iou_non_degrading"] = False
            benchmark_checks["rollout_precision_non_degrading"] = False
            benchmark_checks[
                "rollout_largest_component_non_degrading"
            ] = False
    benchmark_passed = all(benchmark_checks.values())
    mechanism_checks.update(
        {
            name: value
            for name, value in benchmark_checks.items()
            if name not in mechanism_checks
        }
    )
    mechanism_passed = all(mechanism_checks.values())
    selected_passed = {
        "report_only": True,
        "benchmark_relaxed": benchmark_passed,
        "mechanism_strict": mechanism_passed,
    }[args.decision_profile]
    report = {
        "stage": "large-data direct view-identity physical SS Flow evaluation",
        "format": DIRECT_FLOW_VERSION,
        "selected_decision_profile": args.decision_profile,
        "selected_passed": selected_passed,
        "benchmark_relaxed": {
            "passed": benchmark_passed,
            "checks": benchmark_checks,
            "claim": "Correct physical input improves frozen ReconViaGen teacher-forced Flow.",
        },
        "mechanism_strict": {
            "passed": mechanism_passed,
            "checks": mechanism_checks,
            "claim": "Improvement depends on the correct pose/depth/visual binding.",
        },
        "args": vars(args),
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": int(checkpoint.get("step", 0)),
        "model_summary": checkpoint.get("model_summary", {}),
        "runtime_model_summary": model_summary,
        "n3_audit": n3_audit,
        "split_audit": split_audit,
        "sample_count": count,
        "object_count": len(
            {
                str(row.get("object_uid", row["uid"]))
                for row in dataset.rows[:count]
            }
        ),
        "physical_scale": physical_scale,
        "seeds": list(seeds),
        "t_values": list(t_values),
        "positive_t_count": positive_t,
        "physical_off_max_abs": off_max_abs,
        "null_evidence_max_abs": null_max_abs,
        "branch_summary": branch_summary,
        "formal_teacher_weighting": "paired record gains averaged per object",
        "teacher_sequence_weighted": teacher_sequence_weighted,
        "t_summary": t_summary,
        "teacher_occupancy": occupancy_summary,
        "rollout": rollout,
        "records": records,
        "occupancy_records": occupancy_records,
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "report.md").write_text(
        report_markdown(report), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    if args.fail_on_science and not selected_passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
