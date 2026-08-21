#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
import gc
import json
from pathlib import Path
from statistics import mean, median
import sys
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import label

TRACKER_ROOT = Path(__file__).resolve().parents[1]
RECONVIAGEN_ROOT = TRACKER_ROOT / "ReconViaGen"
VGGT_ROOT = RECONVIAGEN_ROOT / "wheels" / "vggt"
for path in (TRACKER_ROOT, RECONVIAGEN_ROOT, VGGT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from trellis.pipelines import TrellisVGGTTo3DPipeline  # noqa: E402

from ar_ss_flow.correspondence_lifting import (  # noqa: E402
    correspondence_pair_volume_from_sample,
    load_correspondence_checkpoint,
    pair_feature_dim,
    parse_csv,
)
from ar_ss_flow.local_pose_lifting_flow import PoseLiftingCacheDataset  # noqa: E402
from ar_ss_flow.pair_feature_ss_flow import (  # noqa: E402
    PAIR_FEATURE_SS_FLOW_VERSION,
    PositiveConditionRolloutFlow,
)
from ar_ss_flow.train_pair_feature_ss_flow import (  # noqa: E402
    PAIR_NEGATIVE_MODES,
    build_model,
    find_cross_sample,
)
from reconvggt_ar_adapter_a.pointpose_ss_condition import load_partial_state  # noqa: E402
from reconvggt_ar_adapter_a.train_pointpose_ss_lora import (  # noqa: E402
    install_unused_model_stubs,
)


CONTROL_BRANCHES = ("constant_pair", "spatial_permuted_pair")


def parse_int_csv(text: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in text.split(",") if item.strip())
    if not values:
        raise ValueError("integer CSV must be non-empty")
    return values


def parse_float_csv(text: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in text.split(",") if item.strip())
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


def constant_pair_control(
    pair_features: torch.Tensor, pair_valid: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    valid = pair_valid.float()
    weighted = pair_features.float() * valid
    denominator = valid.sum(dim=(1, 3, 4, 5), keepdim=False).clamp_min(1.0)
    feature_mean = weighted.sum(dim=(1, 3, 4, 5)) / denominator
    output = feature_mean[:, None, :, None, None, None].expand_as(pair_features)
    return (output * valid).to(dtype=pair_features.dtype), pair_valid


def spatial_permuted_pair_control(
    pair_features: torch.Tensor,
    pair_valid: torch.Tensor,
    *,
    shift: tuple[int, int, int] = (3, 5, 7),
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.roll(pair_features, shifts=shift, dims=(-3, -2, -1)),
        torch.roll(pair_valid, shifts=shift, dims=(-3, -2, -1)),
    )


def coord_set(coords: np.ndarray) -> set[tuple[int, int, int]]:
    return {tuple(int(value) for value in row[-3:]) for row in coords}


def overlap_metrics(predicted: np.ndarray, target: np.ndarray) -> dict[str, float | int]:
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
    return torch.argwhere(logits > 0)[:, [0, 2, 3, 4]].cpu().numpy().astype(np.int32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Teacher-forced and optional rollout evaluation for C3 pair-feature SS residual."
        )
    )
    parser.add_argument("--cache_manifest", required=True)
    parser.add_argument("--correspondence_checkpoint", required=True)
    parser.add_argument("--flow_checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--pretrained", default="Stable-X/trellis-vggt-v0-2")
    parser.add_argument("--indices", default="16-63")
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--t_values", default="0.55,0.65,0.75,0.85,0.95")
    parser.add_argument(
        "--inactive_t_probe",
        type=float,
        default=0.4,
        help="One low-noise time that must remain bit-exact stock.",
    )
    parser.add_argument(
        "--negative_modes",
        default="pose_cyclic1,pose_cyclic2,pose_reverse,cross_sample",
    )
    parser.add_argument("--physical_scale", type=float, default=None)
    parser.add_argument("--amp_dtype", choices=("bf16", "fp16", "none"), default="bf16")
    parser.add_argument("--benchmark_min_object_win", type=float, default=0.55)
    parser.add_argument("--benchmark_min_positive_t", type=int, default=3)
    parser.add_argument("--mechanism_min_object_win", type=float, default=0.65)
    parser.add_argument("--mechanism_min_control_win", type=float, default=0.55)
    parser.add_argument("--mechanism_min_positive_t", type=int, default=4)
    parser.add_argument("--max_corrupt_abs_gain", type=float, default=0.005)
    parser.add_argument(
        "--decision_profile",
        choices=("report_only", "benchmark_relaxed", "mechanism_strict"),
        default="report_only",
    )
    parser.add_argument("--fail_on_error", action="store_true")
    parser.add_argument("--rollout_steps", type=int, default=0)
    parser.add_argument("--rollout_max_samples", type=int, default=4)
    parser.add_argument("--rollout_seeds", default="42")
    parser.add_argument("--rollout_min_object_win", type=float, default=0.55)
    parser.add_argument("--rollout_max_precision_drop", type=float, default=1.0e-4)
    parser.add_argument(
        "--rollout_max_largest_component_drop", type=float, default=0.01
    )
    parser.add_argument("--cfg_strength", type=float, default=7.5)
    parser.add_argument("--guidance_rescale", type=float, default=0.5)
    parser.add_argument("--rescale_t", type=float, default=3.0)
    return parser.parse_args()


def make_evidence(
    dataset: PoseLiftingCacheDataset,
    sample_index: int,
    sample: dict[str, Any],
    mode: str,
    *,
    device: torch.device,
    correspondence_model,
    neighborhood_radius: int,
    min_source_views: int,
    confidence_floor: float,
):
    visual_override = None
    evidence_mode = mode
    if mode == "cross_sample":
        cross = find_cross_sample(dataset, sample_index, sample)
        visual_override = cross["visual_patch_features"]
        evidence_mode = "correct"
    return correspondence_pair_volume_from_sample(
        sample,
        device=device,
        model=correspondence_model,
        mode=evidence_mode,
        visual_patch_features_override=visual_override,
        neighborhood_radius=neighborhood_radius,
        min_source_views=min_source_views,
        confidence_floor=confidence_floor,
    )


def object_balanced_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        buckets[(str(row["object_uid"]), str(row["branch"]))].append(row)
    output = []
    for (uid, branch), rows in sorted(buckets.items()):
        output.append(
            {
                "object_uid": uid,
                "branch": branch,
                "gain_vs_stock": mean(float(row["gain_vs_stock"]) for row in rows),
                "correct_gain_vs_branch": mean(
                    float(row.get("correct_gain_vs_branch", 0.0)) for row in rows
                ),
            }
        )
    return output


@torch.no_grad()
def run_rollout(
    *,
    args: argparse.Namespace,
    dataset: PoseLiftingCacheDataset,
    count: int,
    device: torch.device,
    model,
    correspondence_model,
    modes: tuple[str, ...],
    use_amp: bool,
    amp_dtype: torch.dtype,
    physical_scale: float,
) -> dict[str, Any] | None:
    if int(args.rollout_steps) <= 0:
        return None
    install_unused_model_stubs()
    pipeline = TrellisVGGTTo3DPipeline.from_pretrained(args.pretrained)
    sampler = pipeline.sparse_structure_sampler
    sampler_params = dict(pipeline.sparse_structure_sampler_params)
    sampler_params.update(
        {
            "steps": int(args.rollout_steps),
            "cfg_strength": float(args.cfg_strength),
            "guidance_rescale": float(args.guidance_rescale),
            "rescale_t": float(args.rescale_t),
        }
    )
    decoder = pipeline.models["sparse_structure_decoder"].to(device).eval()
    for parameter in decoder.parameters():
        parameter.requires_grad = False
    del pipeline
    gc.collect()

    rollout_count = min(count, max(int(args.rollout_max_samples), 0))
    seeds = parse_int_csv(args.rollout_seeds)
    rows: list[dict[str, Any]] = []
    branch_names = ("stock", "correct", *CONTROL_BRANCHES, *modes)
    for sample_index in range(rollout_count):
        sample = dataset[sample_index]
        condition = sample["stock_condition"].to(device=device)
        negative_condition = torch.zeros_like(condition)
        target_coords = sample["target_coords"].numpy().astype(np.int32)
        correct = make_evidence(
            dataset,
            sample_index,
            sample,
            "correct",
            device=device,
            correspondence_model=correspondence_model,
            neighborhood_radius=int(args.neighborhood_radius),
            min_source_views=int(args.min_source_views),
            confidence_floor=float(args.confidence_floor),
        )
        constant_pairs = constant_pair_control(correct[2], correct[3])
        permuted_pairs = spatial_permuted_pair_control(correct[2], correct[3])
        branch_evidence = {
            "correct": (correct[0], correct[1], correct[2], correct[3]),
            "constant_pair": (correct[0], correct[1], *constant_pairs),
            "spatial_permuted_pair": (correct[0], correct[1], *permuted_pairs),
        }
        for mode in modes:
            item = make_evidence(
                dataset,
                sample_index,
                sample,
                mode,
                device=device,
                correspondence_model=correspondence_model,
                neighborhood_radius=int(args.neighborhood_radius),
                min_source_views=int(args.min_source_views),
                confidence_floor=float(args.confidence_floor),
            )
            branch_evidence[mode] = (item[0], item[1], item[2], item[3])
        for seed in seeds:
            generator = torch.Generator(device=device).manual_seed(
                int(seed) * 1000003 + sample_index * 1009
            )
            noise = torch.randn((1, 8, 16, 16, 16), generator=generator, device=device)
            with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
                stock_latent = sampler.sample(
                    model.stock_flow,
                    noise.clone(),
                    cond=condition,
                    neg_cond=negative_condition,
                    **sampler_params,
                    verbose=False,
                ).samples
            latents = {"stock": stock_latent}
            call_audits = {}
            for branch, evidence in branch_evidence.items():
                wrapper = PositiveConditionRolloutFlow(
                    model,
                    condition,
                    evidence,
                    scale=physical_scale,
                )
                with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
                    latents[branch] = sampler.sample(
                        wrapper,
                        noise.clone(),
                        cond=condition,
                        neg_cond=negative_condition,
                        **sampler_params,
                        verbose=False,
                    ).samples
                call_audits[branch] = {
                    "positive_calls": wrapper.positive_calls,
                    "negative_calls": wrapper.negative_calls,
                }
                if wrapper.positive_calls <= 0:
                    raise RuntimeError(f"rollout branch {branch} did not adapt positive CFG")
            for branch in branch_names:
                coords = decode_coords(decoder, latents[branch])
                rows.append(
                    {
                        "sample_index": sample_index,
                        "uid": str(sample["uid"]),
                        "object_uid": str(sample.get("object_uid", sample["uid"])),
                        "seed": int(seed),
                        "branch": branch,
                        **overlap_metrics(coords, target_coords),
                        **component_metrics(coords),
                        "cfg_calls": call_audits.get(branch),
                    }
                )
        print(f"[pair_feature_rollout] {sample_index + 1}/{rollout_count}", flush=True)

    summaries = {}
    for branch in branch_names:
        branch_rows = [row for row in rows if row["branch"] == branch]
        summaries[branch] = {
            metric: summarize([float(row[metric]) for row in branch_rows])
            for metric in (
                "iou",
                "precision",
                "recall",
                "coord_count_ratio",
                "component_count",
                "largest_component_ratio",
            )
        }
    stock_by_key = {
        (row["object_uid"], row["seed"]): row
        for row in rows
        if row["branch"] == "stock"
    }
    delta = {}
    object_balanced_delta = {}
    for branch in branch_names:
        if branch == "stock":
            continue
        branch_rows = [row for row in rows if row["branch"] == branch]
        delta[branch] = {
            metric: summarize(
                [
                    float(row[metric])
                    - float(stock_by_key[(row["object_uid"], row["seed"])][metric])
                    for row in branch_rows
                ]
            )
            for metric in ("iou", "precision", "recall", "largest_component_ratio")
        }
        object_balanced_delta[branch] = {}
        for metric in ("iou", "precision", "recall", "largest_component_ratio"):
            buckets: dict[str, list[float]] = defaultdict(list)
            for row in branch_rows:
                stock_row = stock_by_key[(row["object_uid"], row["seed"])]
                buckets[str(row["object_uid"])].append(
                    float(row[metric]) - float(stock_row[metric])
                )
            object_values = [mean(values) for values in buckets.values()]
            object_balanced_delta[branch][metric] = {
                **summarize(object_values),
                "object_win_rate": positive_rate(object_values),
            }
    return {
        "sample_count": rollout_count,
        "seeds": list(seeds),
        "sampler_params": sampler_params,
        "summaries": summaries,
        "delta_vs_stock": delta,
        "object_balanced_delta_vs_stock": object_balanced_delta,
        "records": rows,
    }


@torch.no_grad()
def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    dataset = PoseLiftingCacheDataset(args.cache_manifest, indices=args.indices)
    count = len(dataset) if args.max_samples <= 0 else min(len(dataset), args.max_samples)
    modes = parse_csv(args.negative_modes)
    invalid = [mode for mode in modes if mode not in PAIR_NEGATIVE_MODES]
    if invalid:
        raise ValueError(f"invalid negative modes={invalid}")
    correspondence_model, correspondence_checkpoint = load_correspondence_checkpoint(
        args.correspondence_checkpoint,
        device=device,
        visual_channels=dataset.visual_feature_dim,
    )
    correspondence_model.eval()
    flow_checkpoint = torch.load(args.flow_checkpoint, map_location="cpu")
    if flow_checkpoint.get("format") != PAIR_FEATURE_SS_FLOW_VERSION:
        raise ValueError(f"unexpected C3 checkpoint={flow_checkpoint.get('format')!r}")
    saved = flow_checkpoint.get("args", {})
    for key in (
        "pretrained",
        "adapter_hidden_dim",
        "residual_t_min",
        "residual_t_ramp",
        "confidence_floor",
        "neighborhood_radius",
        "min_source_views",
        "amp_dtype",
    ):
        if key in saved:
            setattr(args, key, saved[key])
    physical_scale = (
        float(saved.get("physical_scale", 0.5))
        if args.physical_scale is None
        else float(args.physical_scale)
    )
    pair_dim = pair_feature_dim(correspondence_model.pairwise_dim)
    sampler, model, model_summary = build_model(
        args, device, dataset.visual_feature_dim, pair_dim
    )
    load_partial_state(
        model,
        flow_checkpoint["model_trainable_state"],
        require_all_trainable=True,
    )
    model.eval()
    seeds = parse_int_csv(args.seeds)
    t_values = parse_float_csv(args.t_values)
    if any(value <= float(args.residual_t_min) for value in t_values):
        raise ValueError(
            "all decision t_values must be above the checkpoint residual_t_min; "
            "use --inactive_t_probe for the exact-stock low-t audit"
        )
    if not 0.0 < float(args.inactive_t_probe) <= float(args.residual_t_min):
        raise ValueError(
            "inactive_t_probe must lie in (0, checkpoint residual_t_min]"
        )
    if int(args.benchmark_min_positive_t) > len(t_values):
        raise ValueError("benchmark_min_positive_t exceeds active t count")
    if int(args.mechanism_min_positive_t) > len(t_values):
        raise ValueError("mechanism_min_positive_t exceeds active t count")
    use_amp = args.amp_dtype != "none"
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16

    records: list[dict[str, Any]] = []
    null_max_abs = 0.0
    disabled_max_abs = 0.0
    inactive_t_max_abs = 0.0
    for sample_index in range(count):
        sample = dataset[sample_index]
        correct = make_evidence(
            dataset,
            sample_index,
            sample,
            "correct",
            device=device,
            correspondence_model=correspondence_model,
            neighborhood_radius=int(args.neighborhood_radius),
            min_source_views=int(args.min_source_views),
            confidence_floor=float(args.confidence_floor),
        )
        actual_evidence = {
            mode: make_evidence(
                dataset,
                sample_index,
                sample,
                mode,
                device=device,
                correspondence_model=correspondence_model,
                neighborhood_radius=int(args.neighborhood_radius),
                min_source_views=int(args.min_source_views),
                confidence_floor=float(args.confidence_floor),
            )
            for mode in modes
        }
        constant_pairs = constant_pair_control(correct[2], correct[3])
        permuted_pairs = spatial_permuted_pair_control(correct[2], correct[3])
        target = sample["target"].unsqueeze(0).to(device=device, dtype=torch.float32)
        condition = sample["stock_condition"].to(device=device)
        if sample_index == 0:
            probe_generator = torch.Generator(device=device).manual_seed(40015)
            probe_endpoint = torch.randn(
                target.shape, generator=probe_generator, device=device
            )
            probe_x_t, _ = sampler._get_model_gt(
                target, float(args.inactive_t_probe), probe_endpoint
            )
            probe_t = torch.full(
                (1,),
                1000.0 * float(args.inactive_t_probe),
                device=device,
                dtype=torch.float32,
            )
            with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
                probe_stock = model.stock_prediction(probe_x_t, probe_t, condition)
                probe_prediction, _ = model.adapt_from_stock(
                    probe_x_t,
                    probe_t,
                    probe_stock,
                    correct[0],
                    correct[1],
                    correct[2],
                    correct[3],
                    scale=physical_scale,
                )
            inactive_t_max_abs = float(
                (probe_prediction - probe_stock).abs().max().item()
            )
        for seed in seeds:
            for t_value in t_values:
                combined_seed = (
                    int(seed) * 1000003
                    + sample_index * 1009
                    + int(round(t_value * 1000))
                )
                generator = torch.Generator(device=device).manual_seed(combined_seed)
                endpoint = torch.randn(target.shape, generator=generator, device=device)
                x_t, gt_velocity = sampler._get_model_gt(target, t_value, endpoint)
                t_tensor = torch.full(
                    (1,), 1000.0 * t_value, device=device, dtype=torch.float32
                )
                with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
                    stock = model.stock_prediction(x_t, t_tensor, condition)
                    correct_prediction, correct_stats = model.adapt_from_stock(
                        x_t,
                        t_tensor,
                        stock,
                        correct[0],
                        correct[1],
                        correct[2],
                        correct[3],
                        scale=physical_scale,
                    )
                    constant_prediction, _ = model.adapt_from_stock(
                        x_t,
                        t_tensor,
                        stock,
                        correct[0],
                        correct[1],
                        constant_pairs[0],
                        constant_pairs[1],
                        scale=physical_scale,
                    )
                    permuted_prediction, _ = model.adapt_from_stock(
                        x_t,
                        t_tensor,
                        stock,
                        correct[0],
                        correct[1],
                        permuted_pairs[0],
                        permuted_pairs[1],
                        scale=physical_scale,
                    )
                    disabled, _ = model.adapt_from_stock(
                        x_t,
                        t_tensor,
                        stock,
                        correct[0],
                        correct[1],
                        correct[2],
                        correct[3],
                        physical_present=False,
                    )
                    null, _ = model.adapt_from_stock(
                        x_t,
                        t_tensor,
                        stock,
                        torch.zeros_like(correct[0]),
                        torch.zeros_like(correct[1]),
                        torch.zeros_like(correct[2]),
                        torch.zeros_like(correct[3]),
                    )
                    actual_predictions = {}
                    for mode in modes:
                        item = actual_evidence[mode]
                        actual_predictions[mode] = model.adapt_from_stock(
                            x_t,
                            t_tensor,
                            stock,
                            item[0],
                            item[1],
                            item[2],
                            item[3],
                            scale=physical_scale,
                        )
                null_max_abs = max(null_max_abs, float((null - stock).abs().max().item()))
                disabled_max_abs = max(
                    disabled_max_abs, float((disabled - stock).abs().max().item())
                )
                stock_loss = float(F.mse_loss(stock.float(), gt_velocity.float()).item())
                branch_predictions = {
                    "correct": correct_prediction,
                    "constant_pair": constant_prediction,
                    "spatial_permuted_pair": permuted_prediction,
                    **{mode: value[0] for mode, value in actual_predictions.items()},
                }
                branch_losses = {
                    branch: float(F.mse_loss(value.float(), gt_velocity.float()).item())
                    for branch, value in branch_predictions.items()
                }
                correct_loss = branch_losses["correct"]
                for branch, branch_loss in branch_losses.items():
                    records.append(
                        {
                            "sample_index": sample_index,
                            "uid": str(sample["uid"]),
                            "object_uid": str(sample.get("object_uid", sample["uid"])),
                            "seed": int(seed),
                            "t": float(t_value),
                            "branch": branch,
                            "stock_loss": stock_loss,
                            "branch_loss": branch_loss,
                            "gain_vs_stock": (stock_loss - branch_loss)
                            / max(stock_loss, 1.0e-8),
                            "correct_gain_vs_branch": (branch_loss - correct_loss)
                            / max(stock_loss, 1.0e-8),
                            "correct_delta_rms": float(
                                correct_stats["delta_rms"].float().item()
                            ),
                        }
                    )
        print(f"[pair_feature_eval] {sample_index + 1}/{count}", flush=True)
        del correct, actual_evidence
        torch.cuda.empty_cache()

    object_rows = object_balanced_rows(records)
    branches = ("correct", *CONTROL_BRANCHES, *modes)
    branch_summary = {}
    for branch in branches:
        rows = [row for row in object_rows if row["branch"] == branch]
        gains = [float(row["gain_vs_stock"]) for row in rows]
        advantages = [float(row["correct_gain_vs_branch"]) for row in rows]
        branch_summary[branch] = {
            "gain_vs_stock": summarize(gains),
            "stock_object_win_rate": positive_rate(gains),
            "correct_gain_vs_branch": summarize(advantages),
            "correct_object_win_vs_branch": positive_rate(advantages),
        }

    t_summary = {}
    positive_t = 0
    for t_value in t_values:
        values = [
            float(row["gain_vs_stock"])
            for row in records
            if row["branch"] == "correct" and row["t"] == t_value
        ]
        t_summary[str(t_value)] = summarize(values)
        positive_t += int(float(t_summary[str(t_value)]["mean"]) > 0.0)

    correct = branch_summary["correct"]
    benchmark_checks = {
        "stock_exact_off": disabled_max_abs == 0.0 and null_max_abs == 0.0,
        "inactive_t_exact_stock": inactive_t_max_abs == 0.0,
        "correct_mean_gain_positive": float(correct["gain_vs_stock"]["mean"]) > 0.0,
        "correct_median_gain_positive": float(correct["gain_vs_stock"]["median"]) > 0.0,
        "correct_stock_object_win": float(correct["stock_object_win_rate"])
        >= float(args.benchmark_min_object_win),
        "positive_t": positive_t >= int(args.benchmark_min_positive_t),
    }
    benchmark_passed = all(benchmark_checks.values())

    mechanism_checks: dict[str, bool] = dict(benchmark_checks)
    mechanism_checks["strict_positive_t"] = positive_t >= int(
        args.mechanism_min_positive_t
    )
    for branch in (*CONTROL_BRANCHES, *modes):
        row = branch_summary[branch]
        threshold = (
            float(args.mechanism_min_control_win)
            if branch in CONTROL_BRANCHES
            else float(args.mechanism_min_object_win)
        )
        mechanism_checks[f"correct_mean_vs_{branch}"] = float(
            row["correct_gain_vs_branch"]["mean"]
        ) > 0.0
        mechanism_checks[f"correct_median_vs_{branch}"] = float(
            row["correct_gain_vs_branch"]["median"]
        ) > 0.0
        mechanism_checks[f"correct_win_vs_{branch}"] = float(
            row["correct_object_win_vs_branch"]
        ) >= threshold
        if branch in modes:
            mechanism_checks[f"{branch}_near_stock"] = abs(
                float(row["gain_vs_stock"]["mean"])
            ) <= float(args.max_corrupt_abs_gain)
    mechanism_passed = all(mechanism_checks.values())

    rollout = run_rollout(
        args=args,
        dataset=dataset,
        count=count,
        device=device,
        model=model,
        correspondence_model=correspondence_model,
        modes=modes,
        use_amp=use_amp,
        amp_dtype=amp_dtype,
        physical_scale=physical_scale,
    )
    rollout_benchmark = None
    if rollout is not None:
        correct_rollout = rollout["object_balanced_delta_vs_stock"]["correct"]
        rollout_checks = {
            "correct_iou_mean_positive": float(correct_rollout["iou"]["mean"])
            > 0.0,
            "correct_iou_median_positive": float(
                correct_rollout["iou"]["median"]
            )
            > 0.0,
            "correct_iou_object_win": float(
                correct_rollout["iou"]["object_win_rate"]
            )
            >= float(args.rollout_min_object_win),
            "precision_not_degraded": float(correct_rollout["precision"]["mean"])
            >= -float(args.rollout_max_precision_drop),
            "largest_component_not_degraded": float(
                correct_rollout["largest_component_ratio"]["mean"]
            )
            >= -float(args.rollout_max_largest_component_drop),
        }
        rollout_benchmark = {
            "passed": all(rollout_checks.values()),
            "checks": rollout_checks,
            "teacher_forced_and_rollout_passed": benchmark_passed
            and all(rollout_checks.values()),
        }
    selected_passed = {
        "report_only": True,
        "benchmark_relaxed": benchmark_passed,
        "mechanism_strict": mechanism_passed,
    }[args.decision_profile]
    report = {
        "stage": "C3 actual-corruption pair-feature SS evaluation",
        "format": PAIR_FEATURE_SS_FLOW_VERSION,
        "selected_decision_profile": args.decision_profile,
        "selected_passed": selected_passed,
        "benchmark_relaxed": {
            "passed": benchmark_passed,
            "checks": benchmark_checks,
            "claim": (
                "Teacher-forced flow objective improves over frozen ReconViaGen "
                "stock on this protocol. Final generation claims require the "
                "separate rollout benchmark."
            ),
        },
        "mechanism_strict": {
            "passed": mechanism_passed,
            "checks": mechanism_checks,
            "claim": "Supports pose/correspondence-specific causal attribution.",
        },
        "args": vars(args),
        "physical_scale": physical_scale,
        "sample_count": count,
        "sample_uids": [str(row["uid"]) for row in dataset.rows[:count]],
        "noise_seeds": list(seeds),
        "t_values": list(t_values),
        "positive_t": positive_t,
        "active_t_count": len(t_values),
        "inactive_t_probe": float(args.inactive_t_probe),
        "inactive_t_max_abs_diff": inactive_t_max_abs,
        "correspondence_checkpoint_step": int(correspondence_checkpoint.get("step", 0)),
        "flow_checkpoint_step": int(flow_checkpoint.get("step", 0)),
        "model_summary": model_summary,
        "null_max_abs_diff": null_max_abs,
        "physical_off_max_abs_diff": disabled_max_abs,
        "branch_summary": branch_summary,
        "t_summary": t_summary,
        "rollout": rollout,
        "rollout_benchmark": rollout_benchmark,
        "object_rows": object_rows,
        "records": records,
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)
    if args.fail_on_error and not selected_passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
