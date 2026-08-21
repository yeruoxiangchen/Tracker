#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
from collections import defaultdict
import json
import os
from pathlib import Path
import random
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
for path in (TRACKER_ROOT, TRACKER_ROOT / "ReconViaGen", TRACKER_ROOT / "ReconViaGen" / "wheels" / "vggt"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ar_ss_flow.local_pose_lifting_flow import (  # noqa: E402
    PoseLiftingCacheDataset,
    volume_from_sample,
)
from ar_ss_flow.train_local_pose_lifting_ss_flow import build_model  # noqa: E402
from pose_point_depth_mv import PACKAGE_VERSION  # noqa: E402
from pose_point_depth_mv.geometry import (  # noqa: E402
    build_evidence,
    match_gated_delta_rms,
    mean_match_gate,
    prepare_frozen_crossfit_calibration,
)


C2_CHECKPOINT_VERSION = "ar_ss_flow.object_gated_pose_lifting_c2.v1"
CORRUPTIONS = {
    "pose_cyclic1_local": {"pose_mode": "pose_cyclic1"},
    "pose_reverse_local": {"pose_mode": "pose_reverse"},
    "depth_view_cyclic1_local": {"depth_mode": "depth_view_cyclic1"},
    "depth_spatial_local": {"depth_mode": "depth_spatial"},
    "point_reflect_local": {"point_mode": "point_reflect"},
    "point_cross_object_local": {"point_mode": "point_cross_object"},
}
BRANCHES = (
    "residual_off",
    "uniform_base",
    "matched_uniform",
    "correct_local",
    *CORRUPTIONS.keys(),
    "spatial_shuffle_local",
)
NEGATIVE_BRANCHES = (*CORRUPTIONS.keys(), "spatial_shuffle_local")


def parse_csv_ints(value: str) -> list[int]:
    result = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not result:
        raise ValueError("at least one seed is required")
    return result


def parse_csv_floats(value: str) -> list[float]:
    result = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not result or any(not 0.0 < item < 1.0 for item in result):
        raise ValueError("t values must lie in (0,1)")
    return result


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
    finite = [float(value) for value in values if np.isfinite(value)]
    return float(mean(value > 0.0 for value in finite)) if finite else 0.0


def different_object_sample(dataset, index: int, count: int) -> dict[str, Any]:
    source = dataset[index]
    source_uid = str(source.get("object_uid", source["uid"]))
    for offset in range(1, count):
        candidate = dataset[(index + offset) % count]
        if str(candidate.get("object_uid", candidate["uid"])) != source_uid:
            return candidate
    raise RuntimeError("point_cross_object requires at least two distinct objects")


def bootstrap_ci(values: list[float], *, seed: int, samples: int = 10000) -> list[float]:
    if not values:
        return [0.0, 0.0]
    rng = random.Random(int(seed))
    draws: list[float] = []
    for _ in range(int(samples)):
        draws.append(mean(values[rng.randrange(len(values))] for _ in values))
    draws.sort()
    return [draws[int(0.025 * len(draws))], draws[int(0.975 * len(draws))]]


def load_adapter_state(model: torch.nn.Module, state: dict[str, torch.Tensor]) -> None:
    current = model.state_dict()
    unexpected = sorted(set(state).difference(current))
    if unexpected:
        raise KeyError(f"checkpoint has unexpected parameters={unexpected[:5]}")
    missing = sorted(
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and name not in state
    )
    if missing:
        raise KeyError(f"checkpoint missing trainable parameters={missing[:5]}")
    model.load_state_dict(state, strict=False)


def time_gate(t: float, start: float, ramp: float) -> float:
    if ramp <= 0.0:
        return float(t >= start)
    value = (float(t) - float(start)) / float(ramp)
    return float(np.clip(value, 0.0, 1.0))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Standalone C3 audit: apply explicit local pose-point-depth gates to an "
            "already-trained C2 residual without modifying ar_ss_flow."
        )
    )
    parser.add_argument("--cache_manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--pretrained", default="Stable-X/trellis-vggt-v0-2")
    parser.add_argument("--indices", default="16-63")
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seeds", default="48,49,50")
    parser.add_argument("--t_values", default="0.5,0.7,0.9")
    parser.add_argument("--adapter_hidden_dim", type=int, default=96)
    parser.add_argument("--base_residual_scale", type=float, default=0.5)
    parser.add_argument("--object_cap", type=float, default=1.0)
    parser.add_argument("--time_gate_start", type=float, default=0.5)
    parser.add_argument("--time_gate_ramp", type=float, default=0.0)
    parser.add_argument("--amp_dtype", choices=("fp16", "bf16", "none"), default="bf16")
    parser.add_argument("--minimum_depth_tolerance", type=float, default=0.02)
    parser.add_argument("--maximum_depth_tolerance", type=float, default=0.15)
    parser.add_argument("--surface_threshold", type=float, default=0.30)
    parser.add_argument("--free_threshold", type=float, default=0.30)
    parser.add_argument("--minimum_surface_views", type=int, default=2)
    parser.add_argument("--minimum_free_views", type=int, default=2)
    parser.add_argument("--prior_radius_voxels", type=float, default=1.5)
    parser.add_argument("--gate_floor", type=float, default=0.25)
    parser.add_argument("--min_depth_matches", type=int, default=8)
    parser.add_argument("--min_heldout_matches", type=int, default=8)
    parser.add_argument("--affine_improvement_ratio", type=float, default=0.90)
    parser.add_argument("--calibration_fit_fraction", type=float, default=0.5)
    parser.add_argument("--calibration_split_seed", type=int, default=20260715)
    parser.add_argument("--minimum_points_per_split", type=int, default=4)
    parser.add_argument("--calibration_mask_threshold", type=float, default=0.5)
    parser.add_argument("--calibration_zbuffer_cell_size", type=int, default=14)
    parser.add_argument("--allow_affine_calibration", action="store_true")
    parser.add_argument("--maximum_heldout_median_residual", type=float, default=0.25)
    parser.add_argument("--maximum_heldout_p90_residual", type=float, default=0.60)
    parser.add_argument("--quality_reference_residual", type=float, default=0.25)
    parser.add_argument("--min_eligible_ratio", type=float, default=0.70)
    parser.add_argument("--max_inactive_gate_abs", type=float, default=0.0)
    parser.add_argument("--max_energy_match_scale", type=float, default=10.0)
    parser.add_argument("--hard_max_energy_match_scale", type=float, default=1000.0)
    parser.add_argument("--max_energy_match_relative_error", type=float, default=1.0e-5)
    parser.add_argument("--bootstrap_samples", type=int, default=10000)
    parser.add_argument("--min_object_win_vs_matched", type=float, default=0.55)
    parser.add_argument("--min_object_win_vs_corrupt", type=float, default=0.60)
    parser.add_argument("--min_correct_object_gain", type=float, default=-0.001)
    parser.add_argument("--allow_failures", action="store_true")
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    if not 0.0 <= float(args.object_cap) <= 1.0:
        raise ValueError("object_cap must be in [0,1]")
    if float(args.base_residual_scale) < 0.0:
        raise ValueError("base_residual_scale must be nonnegative")
    if float(args.max_energy_match_scale) <= 0.0:
        raise ValueError("max_energy_match_scale must be positive")
    if float(args.hard_max_energy_match_scale) < float(
        args.max_energy_match_scale
    ):
        raise ValueError(
            "hard_max_energy_match_scale must be >= max_energy_match_scale"
        )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("Flow causal audit requires CUDA")
    dataset = PoseLiftingCacheDataset(args.cache_manifest, indices=args.indices)
    count = len(dataset) if args.max_samples <= 0 else min(len(dataset), args.max_samples)
    evidence_common = {
        "device": device,
        "volume_side": 16,
        "minimum_depth_tolerance": float(args.minimum_depth_tolerance),
        "maximum_depth_tolerance": float(args.maximum_depth_tolerance),
        "surface_threshold": float(args.surface_threshold),
        "free_threshold": float(args.free_threshold),
        "minimum_surface_views": int(args.minimum_surface_views),
        "minimum_free_views": int(args.minimum_free_views),
        "prior_radius_voxels": float(args.prior_radius_voxels),
        "gate_floor": float(args.gate_floor),
        "min_depth_matches": int(args.min_depth_matches),
        "affine_improvement_ratio": float(args.affine_improvement_ratio),
        "calibration_mask_threshold": float(args.calibration_mask_threshold),
        "calibration_zbuffer_cell_size": int(
            args.calibration_zbuffer_cell_size
        ),
        "force_scale_only": not bool(args.allow_affine_calibration),
        "recalibrate_each_hypothesis": False,
    }
    gates_by_index: dict[int, dict[str, torch.Tensor]] = {}
    gate_stats: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    eligible: list[int] = []
    for index in range(count):
        sample = dataset[index]
        cross = different_object_sample(dataset, index, len(dataset))
        uid = str(sample["uid"])
        try:
            protocol = prepare_frozen_crossfit_calibration(
                sample,
                device=device,
                fit_fraction=float(args.calibration_fit_fraction),
                split_seed=int(args.calibration_split_seed),
                minimum_points_per_split=int(args.minimum_points_per_split),
                min_depth_matches=int(args.min_depth_matches),
                min_heldout_matches=int(args.min_heldout_matches),
                affine_improvement_ratio=float(args.affine_improvement_ratio),
                mask_threshold=float(args.calibration_mask_threshold),
                zbuffer_cell_size=int(args.calibration_zbuffer_cell_size),
                force_scale_only=not bool(args.allow_affine_calibration),
                maximum_heldout_median_residual=float(
                    args.maximum_heldout_median_residual
                ),
                maximum_heldout_p90_residual=float(
                    args.maximum_heldout_p90_residual
                ),
                quality_reference_residual=float(args.quality_reference_residual),
            )
            evidence_protocol = {
                "calibration_override": protocol["calibration"],
                "input_prior_coords": protocol["fit_coords"],
                "input_prior_confidence": protocol["fit_confidence"],
                "evaluation_prior_coords": protocol["heldout_coords"],
                "evaluation_prior_confidence": protocol["heldout_confidence"],
            }
            correct = build_evidence(
                sample, **evidence_common, **evidence_protocol
            )
            correct_gate = correct.local_gate.to(device=device, dtype=torch.float32)
            if not correct.stats["depth_calibration_enabled"]:
                skipped.append(
                    {
                        "index": index,
                        "uid": uid,
                        "object_uid": str(sample.get("object_uid", uid)),
                        "error": "held-out depth calibration quality rejected",
                    }
                )
                continue
            if not bool((correct_gate > 0.0).any().item()):
                skipped.append(
                    {
                        "index": index,
                        "uid": uid,
                        "object_uid": str(sample.get("object_uid", uid)),
                        "error": "no trusted positive surface voxel",
                    }
                )
                continue
            gate_rows: dict[str, torch.Tensor] = {
                "uniform_base": torch.ones_like(correct_gate),
                "matched_uniform": torch.full_like(
                    correct_gate, float(correct_gate.mean().item())
                ),
                "correct_local": correct_gate,
            }
            mode_stats: dict[str, Any] = {"correct_local": correct.stats}
            for branch, overrides in CORRUPTIONS.items():
                result = build_evidence(
                    sample,
                    cross_object_sample=cross,
                    **evidence_common,
                    **evidence_protocol,
                    **overrides,
                )
                gate_rows[branch] = mean_match_gate(
                    result.local_gate.to(device=device), correct_gate
                )
                mode_stats[branch] = result.stats
            shuffled = torch.roll(correct_gate, shifts=(3, 5, 7), dims=(-3, -2, -1))
            gate_rows["spatial_shuffle_local"] = shuffled
            gate_rows["residual_off"] = torch.zeros_like(correct_gate)
            gates_by_index[index] = gate_rows
            gate_stats.append(
                {
                    "index": index,
                    "uid": uid,
                    "object_uid": str(sample.get("object_uid", uid)),
                    "correct_gate_mean": float(correct_gate.mean().item()),
                    "correct_gate_min": float(correct_gate.min().item()),
                    "correct_gate_max": float(correct_gate.max().item()),
                    "calibration": protocol["calibration"],
                    "calibration_fit_indices": protocol["fit_indices"].cpu().tolist(),
                    "calibration_heldout_indices": protocol[
                        "heldout_indices"
                    ].cpu().tolist(),
                    "cross_object_uid": str(
                        cross.get("object_uid", cross["uid"])
                    ),
                    "correct": correct.stats,
                    "modes": mode_stats,
                }
            )
            eligible.append(index)
            print(
                f"[ppd_flow_gate] {index + 1}/{count} uid={uid} "
                f"mean_gate={correct_gate.mean().item():.4f}",
                flush=True,
            )
        except Exception as error:  # noqa: BLE001
            skipped.append(
                {
                    "index": index,
                    "uid": uid,
                    "object_uid": str(sample.get("object_uid", uid)),
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            if not args.allow_failures:
                raise
    if not eligible:
        raise RuntimeError("no objects have valid pose-point-depth local gates")

    gc.collect()
    torch.cuda.empty_cache()

    # Load the large frozen SS Flow only after all local gates have been built.
    # This keeps the geometry stage isolated and avoids holding model memory
    # during repeated pose/depth/point calibration.
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    if checkpoint.get("format") != C2_CHECKPOINT_VERSION:
        raise ValueError(f"unexpected checkpoint format={checkpoint.get('format')!r}")
    saved_args = checkpoint.get("args", {})
    args.adapter_hidden_dim = int(saved_args.get("adapter_hidden_dim", args.adapter_hidden_dim))
    args.amp_dtype = str(saved_args.get("amp_dtype", args.amp_dtype))
    flow_sampler, model, model_summary = build_model(args, device, dataset.visual_feature_dim)
    load_adapter_state(model, checkpoint["model_trainable_state"])
    model.eval()

    seeds = parse_csv_ints(args.seeds)
    t_values = parse_csv_floats(args.t_values)
    use_amp = args.amp_dtype != "none"
    amp_dtype = torch.float16 if args.amp_dtype == "fp16" else torch.bfloat16
    records: list[dict[str, Any]] = []
    off_max_abs = 0.0
    max_energy_match_relative_error = 0.0
    max_observed_energy_match_scale = 0.0
    energy_match_scale_exceedance_count = 0
    energy_match_scale_exceedance_examples: list[dict[str, Any]] = []
    energy_match_unattainable_count = 0
    energy_match_unattainable_examples: list[dict[str, Any]] = []

    for position, index in enumerate(eligible):
        sample = dataset[index]
        uid = str(sample["uid"])
        object_uid = str(sample.get("object_uid", uid))
        volume, metadata, volume_stats = volume_from_sample(
            sample, device=device, mode="correct"
        )
        target = sample["target"].unsqueeze(0).to(device=device, dtype=torch.float32)
        condition = sample["stock_condition"].to(device=device)
        local_gates = gates_by_index[index]
        for noise_seed in seeds:
            for t_value in t_values:
                combined_seed = (
                    int(noise_seed) * 1000003
                    + int(index) * 1009
                    + int(round(t_value * 1000))
                )
                torch.manual_seed(combined_seed)
                torch.cuda.manual_seed_all(combined_seed)
                endpoint = torch.randn_like(target)
                x_t, gt_velocity = flow_sampler._get_model_gt(target, t_value, endpoint)
                t_tensor = torch.full(
                    (1,), 1000.0 * t_value, device=device, dtype=torch.float32
                )
                with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
                    stock = model.stock_prediction(x_t, t_tensor, condition)
                    raw_delta, adapter_stats = model.adapter(
                        x_t,
                        stock,
                        t_tensor,
                        volume,
                        metadata,
                        scale=1.0,
                        physical_present=True,
                    )
                time_value = time_gate(
                    t_value, float(args.time_gate_start), float(args.time_gate_ramp)
                )
                global_scale = (
                    float(args.base_residual_scale)
                    * float(args.object_cap)
                    * float(time_value)
                )
                predictions: dict[str, torch.Tensor] = {}
                applied: dict[str, torch.Tensor] = {}
                energy_match_scale: dict[str, float] = {}
                raw_delta_float = raw_delta.float()
                reference_pre_global = raw_delta_float * local_gates[
                    "correct_local"
                ].reshape(1, 1, 16, 16, 16)
                reference_rms = float(
                    reference_pre_global.square().mean().sqrt().item()
                )
                for branch in BRANCHES:
                    gate = local_gates[branch].reshape(1, 1, 16, 16, 16)
                    candidate = raw_delta_float * gate
                    if branch in ("matched_uniform", *NEGATIVE_BRANCHES):
                        candidate_rms_before = float(
                            candidate.square().mean().sqrt().item()
                        )
                        (
                            candidate,
                            effective_gate,
                            matched_scale,
                            energy_attainable,
                            relative_error,
                        ) = match_gated_delta_rms(
                            raw_delta_float,
                            gate,
                            reference_pre_global,
                            maximum_scale=float(
                                args.hard_max_energy_match_scale
                            ),
                            relative_tolerance=float(
                                args.max_energy_match_relative_error
                            ),
                        )
                        max_observed_energy_match_scale = max(
                            max_observed_energy_match_scale,
                            float(matched_scale),
                        )
                        if float(matched_scale) > float(
                            args.max_energy_match_scale
                        ):
                            energy_match_scale_exceedance_count += 1
                            if len(energy_match_scale_exceedance_examples) < 100:
                                energy_match_scale_exceedance_examples.append(
                                    {
                                        "uid": uid,
                                        "object_uid": object_uid,
                                        "index": int(index),
                                        "noise_seed": int(noise_seed),
                                        "t": float(t_value),
                                        "branch": branch,
                                        "scale": float(matched_scale),
                                        "reference_rms": reference_rms,
                                        "candidate_rms_before": (
                                            candidate_rms_before
                                        ),
                                    }
                                )
                        if not energy_attainable:
                            energy_match_unattainable_count += 1
                            if len(energy_match_unattainable_examples) < 100:
                                energy_match_unattainable_examples.append(
                                    {
                                        "uid": uid,
                                        "object_uid": object_uid,
                                        "index": int(index),
                                        "noise_seed": int(noise_seed),
                                        "t": float(t_value),
                                        "branch": branch,
                                        "scale": float(matched_scale),
                                        "reference_rms": reference_rms,
                                        "candidate_rms_before": (
                                            candidate_rms_before
                                        ),
                                        "relative_error": float(relative_error),
                                        "effective_gate_max": float(
                                            effective_gate.max().item()
                                        ),
                                    }
                                )
                    else:
                        matched_scale = 1.0
                        relative_error = 0.0
                    candidate_rms = float(candidate.square().mean().sqrt().item())
                    if branch in ("matched_uniform", *NEGATIVE_BRANCHES):
                        max_energy_match_relative_error = max(
                            max_energy_match_relative_error, relative_error
                        )
                    energy_match_scale[branch] = float(matched_scale)
                    delta = candidate * global_scale
                    applied[branch] = delta
                    predictions[branch] = stock.float() + delta
                off_max_abs = max(
                    off_max_abs,
                    float((predictions["residual_off"] - stock.float()).abs().max().item()),
                )
                stock_loss = float(F.mse_loss(stock.float(), gt_velocity.float()).item())
                branch_loss = {
                    branch: float(F.mse_loss(value, gt_velocity.float()).item())
                    for branch, value in predictions.items()
                }
                row: dict[str, Any] = {
                    "uid": uid,
                    "object_uid": object_uid,
                    "noise_seed": int(noise_seed),
                    "t": float(t_value),
                    "time_gate": float(time_value),
                    "stock_flow_mse": stock_loss,
                    "raw_delta_rms": float(
                        adapter_stats["delta_rms"].float().item()
                    ),
                    "supported_voxel_ratio": float(
                        volume_stats["supported_voxel_ratio"]
                    ),
                    "correct_local_gate_mean": float(
                        local_gates["correct_local"].mean().item()
                    ),
                    "correct_local_pre_global_delta_rms": reference_rms,
                }
                for branch in BRANCHES:
                    gain = (stock_loss - branch_loss[branch]) / max(stock_loss, 1.0e-8)
                    row[f"{branch}_flow_mse"] = branch_loss[branch]
                    row[f"{branch}_relative_gain_vs_stock"] = gain
                    row[f"{branch}_delta_rms"] = float(
                        applied[branch].square().mean().sqrt().item()
                    )
                    row[f"{branch}_energy_match_scale"] = energy_match_scale[branch]
                for control in ("uniform_base", "matched_uniform", *NEGATIVE_BRANCHES):
                    row[f"correct_local_gain_vs_{control}"] = (
                        branch_loss[control] - branch_loss["correct_local"]
                    ) / max(stock_loss, 1.0e-8)
                records.append(row)
        print(
            f"[ppd_flow] {position + 1}/{len(eligible)} uid={uid}", flush=True
        )

    object_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        object_rows[str(row["object_uid"])].append(row)

    branch_summary: dict[str, Any] = {}
    for branch in BRANCHES:
        values = [row[f"{branch}_relative_gain_vs_stock"] for row in records]
        object_values = [
            mean(row[f"{branch}_relative_gain_vs_stock"] for row in rows)
            for rows in object_rows.values()
        ]
        branch_summary[branch] = {
            "record_gain": summarize(values),
            "object_balanced_gain": summarize(object_values),
            "object_positive_rate": positive_rate(object_values),
            "object_bootstrap_95_ci": bootstrap_ci(
                object_values,
                seed=20260714,
                samples=int(args.bootstrap_samples),
            ),
            "mean_delta_rms": mean(row[f"{branch}_delta_rms"] for row in records),
            "energy_match_scale": summarize(
                [row[f"{branch}_energy_match_scale"] for row in records]
            ),
        }

    comparisons: dict[str, Any] = {}
    for control in ("uniform_base", "matched_uniform", *NEGATIVE_BRANCHES):
        key = f"correct_local_gain_vs_{control}"
        values = [row[key] for row in records]
        object_values = [mean(row[key] for row in rows) for rows in object_rows.values()]
        comparisons[control] = {
            "record_difference": summarize(values),
            "record_win_rate": positive_rate(values),
            "object_difference": summarize(object_values),
            "object_win_rate": positive_rate(object_values),
            "object_bootstrap_95_ci": bootstrap_ci(
                object_values,
                seed=20260715,
                samples=int(args.bootstrap_samples),
            ),
        }

    t_summary: dict[str, Any] = {}
    positive_t = 0
    for t_value in t_values:
        values = [
            row["correct_local_relative_gain_vs_stock"]
            for row in records
            if abs(float(row["t"]) - t_value) < 1.0e-9
        ]
        t_summary[str(t_value)] = summarize(values)
        positive_t += int(float(t_summary[str(t_value)]["mean"]) > 0.0)

    correct_object = branch_summary["correct_local"]["object_balanced_gain"]
    eligible_ratio = len(eligible) / max(count, 1)
    max_mask_zero_gate = max(
        float(mode["mask_zero_gate_mean"])
        for row in gate_stats
        for mode in row["modes"].values()
    )
    max_neutral_gate = max(
        float(mode["neutral_gate_mean"])
        for row in gate_stats
        for mode in row["modes"].values()
    )
    max_negative_gate = max(
        float(mode["negative_gate_mean"])
        for row in gate_stats
        for mode in row["modes"].values()
    )
    checks = {
        "residual_off_bit_exact_stock": off_max_abs == 0.0,
        "eligible_geometry_ratio": eligible_ratio >= float(args.min_eligible_ratio),
        "mask_zero_gate_is_zero": max_mask_zero_gate
        <= float(args.max_inactive_gate_abs),
        "neutral_gate_is_zero": max_neutral_gate
        <= float(args.max_inactive_gate_abs),
        "negative_gate_is_zero": max_negative_gate
        <= float(args.max_inactive_gate_abs),
        "matched_controls_equal_applied_delta_rms": (
            max_energy_match_relative_error
            <= float(args.max_energy_match_relative_error)
        ),
        "all_matched_control_energies_attainable_without_amplification": (
            energy_match_unattainable_count == 0
        ),
        "correct_local_mean_gain_positive": float(correct_object["mean"]) > 0.0,
        "correct_local_median_gain_positive": float(correct_object["median"]) > 0.0,
        "correct_local_tail_bounded": float(correct_object["min"])
        >= float(args.min_correct_object_gain),
        "correct_local_beats_matched_uniform": (
            float(comparisons["matched_uniform"]["object_difference"]["mean"]) > 0.0
            and float(comparisons["matched_uniform"]["object_win_rate"])
            >= float(args.min_object_win_vs_matched)
        ),
        "correct_local_beats_mean_matched_corruptions": all(
            float(comparisons[branch]["object_difference"]["mean"]) > 0.0
            and float(comparisons[branch]["object_win_rate"])
            >= float(args.min_object_win_vs_corrupt)
            for branch in NEGATIVE_BRANCHES
        ),
        "all_evaluated_t_positive": positive_t == len(t_values),
    }
    passed = all(checks.values())
    report = {
        "stage": "standalone local pose-point-depth gated SS residual causal audit",
        "format": PACKAGE_VERSION,
        "passed": passed,
        "args": vars(args),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_step": int(checkpoint.get("step", 0)),
        "model_summary": model_summary,
        "eligible_object_count": len(object_rows),
        "eligible_sample_ratio": eligible_ratio,
        "record_count": len(records),
        "skipped": skipped,
        "residual_off_max_abs_diff": off_max_abs,
        "max_energy_match_relative_error": max_energy_match_relative_error,
        "max_observed_energy_match_scale": max_observed_energy_match_scale,
        "energy_match_scale_exceedance_count": (
            energy_match_scale_exceedance_count
        ),
        "energy_match_scale_exceedance_examples": (
            energy_match_scale_exceedance_examples
        ),
        "energy_match_unattainable_count": energy_match_unattainable_count,
        "energy_match_unattainable_examples": (
            energy_match_unattainable_examples
        ),
        "max_mask_zero_gate_mean": max_mask_zero_gate,
        "max_neutral_gate_mean": max_neutral_gate,
        "max_negative_gate_mean": max_negative_gate,
        "protocol": {
            "calibration": "correct-only frozen cross-fit",
            "point_protocol": "fit points as input; disjoint held-out points for scoring",
            "evaluation_points": "held-out correct-object points",
            "cross_object_must_differ": True,
            "control_normalization": (
                "mean-matched gates followed by applied-residual RMS matching"
            ),
            "residual_source": (
                "fixed previously-trained C2 correct-volume residual; this is "
                "a gate-location causal audit, not depth-conditioned training"
            ),
        },
        "positive_t_count": positive_t,
        "gate_stats": gate_stats,
        "t_summary": t_summary,
        "branch_summary": branch_summary,
        "comparisons": comparisons,
        "checks": checks,
        "records": records,
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    lines = [
        "# Local Pose-Point-Depth Flow Audit",
        "",
        f"- passed: `{passed}`",
        f"- objects / records: `{len(object_rows)} / {len(records)}`",
        f"- residual-off max abs: `{off_max_abs:.8g}`",
        f"- eligible sample ratio: `{eligible_ratio:.4f}`",
        f"- max energy-match relative error: "
        f"`{max_energy_match_relative_error:.3g}`",
        f"- max observed energy-match scale: "
        f"`{max_observed_energy_match_scale:.6g}`",
        f"- energy-match scale exceedances: "
        f"`{energy_match_scale_exceedance_count}`",
        f"- unattainable energy matches: `{energy_match_unattainable_count}`",
        f"- max mask-zero / neutral / negative gate: "
        f"`{max_mask_zero_gate:.3g} / {max_neutral_gate:.3g} / {max_negative_gate:.3g}`",
        f"- positive t: `{positive_t}/{len(t_values)}`",
        "",
        "| branch | object gain mean | median | positive | bootstrap 95% CI | delta RMS |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for branch in BRANCHES:
        item = branch_summary[branch]
        ci = item["object_bootstrap_95_ci"]
        lines.append(
            f"| {branch} | {item['object_balanced_gain']['mean']:+.6g} | "
            f"{item['object_balanced_gain']['median']:+.6g} | "
            f"{item['object_positive_rate']:.4f} | "
            f"[{ci[0]:+.6g}, {ci[1]:+.6g}] | {item['mean_delta_rms']:.6g} |"
        )
    lines.extend(
        [
            "",
            "| control | correct-minus-control mean | median | object win | bootstrap 95% CI |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for control, item in comparisons.items():
        ci = item["object_bootstrap_95_ci"]
        lines.append(
            f"| {control} | {item['object_difference']['mean']:+.6g} | "
            f"{item['object_difference']['median']:+.6g} | "
            f"{item['object_win_rate']:.4f} | [{ci[0]:+.6g}, {ci[1]:+.6g}] |"
        )
    lines.extend(["", "## Checks", ""])
    lines.extend(f"- {key}: `{value}`" for key, value in checks.items())
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("=" * 100)
    print(report["stage"])
    print("passed:", passed)
    print("objects:", len(object_rows), "records:", len(records))
    print("off_max_abs:", off_max_abs)
    print("eligible_ratio:", eligible_ratio)
    print("max_energy_match_relative_error:", max_energy_match_relative_error)
    print("max_observed_energy_match_scale:", max_observed_energy_match_scale)
    print(
        "energy_match_scale_exceedance_count:",
        energy_match_scale_exceedance_count,
    )
    print("energy_match_unattainable_count:", energy_match_unattainable_count)
    for branch in BRANCHES:
        item = branch_summary[branch]
        print(
            branch,
            "gain=", f"{item['object_balanced_gain']['mean']:+.8f}",
            "median=", f"{item['object_balanced_gain']['median']:+.8f}",
            "positive=", f"{item['object_positive_rate']:.4f}",
            "delta_rms=", f"{item['mean_delta_rms']:.8f}",
        )
    for control, item in comparisons.items():
        print(
            "correct_vs", control,
            "gap=", f"{item['object_difference']['mean']:+.8f}",
            "win=", f"{item['object_win_rate']:.4f}",
        )
    print("checks:", checks)
    print("report:", output_dir / "report.json")
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
