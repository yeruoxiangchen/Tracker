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
import torch

TRACKER_ROOT = Path(__file__).resolve().parents[1]
if str(TRACKER_ROOT) not in sys.path:
    sys.path.insert(0, str(TRACKER_ROOT))

from ar_ss_flow.pairwise_local_confidence import (
    parse_methods,
    transform_confidence_batched,
)

MODES = ("pose_cyclic1", "pose_cyclic2", "pose_reverse")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train-select and fresh-validate raw/local visual-only pairwise confidence. "
            "The aggregation method and thresholds are selected only on train objects."
        )
    )
    parser.add_argument("--train_report", required=True)
    parser.add_argument("--train_volumes", required=True)
    parser.add_argument("--fresh_report", required=True)
    parser.add_argument("--fresh_volumes", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--methods", default="raw,local_mean,local_topk")
    parser.add_argument("--local_radius", type=int, default=1)
    parser.add_argument("--local_topk", type=int, default=8)
    parser.add_argument("--transform_batch_size", type=int, default=32)
    parser.add_argument("--max_voxels_per_mode", type=int, default=200000)
    parser.add_argument("--min_objects", type=int, default=16)
    parser.add_argument("--min_object_win_rate", type=float, default=0.80)
    parser.add_argument("--min_auc", type=float, default=0.70)
    parser.add_argument("--min_gate_coverage", type=float, default=0.20)
    parser.add_argument("--max_gate_coverage", type=float, default=0.80)
    parser.add_argument("--min_gate_selectivity", type=float, default=0.10)
    parser.add_argument("--min_top30_relative_uplift", type=float, default=0.25)
    parser.add_argument("--min_top30_absolute_uplift", type=float, default=0.002)
    parser.add_argument("--max_mode_threshold_spread", type=float, default=0.15)
    parser.add_argument("--min_train_selection_gain", type=float, default=0.01)
    parser.add_argument("--min_fresh_mean_auc_gain", type=float, default=0.03)
    parser.add_argument("--max_fresh_mode_auc_drop", type=float, default=0.005)
    parser.add_argument("--fail_on_error", action="store_true")
    return parser.parse_args()


def load_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def load_npz(path: str | Path) -> dict[str, np.ndarray]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as payload:
        return {key: payload[key] for key in payload.files}


def _rankdata_average(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        rank = 0.5 * (start + end - 1) + 1.0
        ranks[order[start:end]] = rank
        start = end
    return ranks


def binary_auc(positive: np.ndarray, negative: np.ndarray) -> float:
    positive = np.asarray(positive, dtype=np.float64)
    negative = np.asarray(negative, dtype=np.float64)
    positive = positive[np.isfinite(positive)]
    negative = negative[np.isfinite(negative)]
    if positive.size == 0 or negative.size == 0:
        return 0.5
    values = np.concatenate((positive, negative))
    ranks = _rankdata_average(values)
    positive_rank_sum = float(ranks[: positive.size].sum())
    return (
        positive_rank_sum - positive.size * (positive.size + 1) / 2.0
    ) / float(positive.size * negative.size)


def choose_threshold(
    positive: np.ndarray,
    negative: np.ndarray,
    *,
    min_coverage: float,
    max_coverage: float,
) -> dict[str, float]:
    positive = np.asarray(positive, dtype=np.float64)
    negative = np.asarray(negative, dtype=np.float64)
    positive = positive[np.isfinite(positive)]
    negative = negative[np.isfinite(negative)]
    if positive.size == 0 or negative.size == 0:
        raise ValueError("threshold selection requires non-empty positive and negative arrays")
    candidates = np.unique(
        np.quantile(
            np.concatenate((positive, negative)),
            np.linspace(0.0, 1.0, 1001),
        )
    )
    best: dict[str, float] | None = None
    for threshold in candidates:
        pos_coverage = float(np.mean(positive >= threshold))
        if not (float(min_coverage) <= pos_coverage <= float(max_coverage)):
            continue
        neg_coverage = float(np.mean(negative >= threshold))
        youden = pos_coverage - neg_coverage
        balanced_accuracy = 0.5 * (pos_coverage + (1.0 - neg_coverage))
        row = {
            "threshold": float(threshold),
            "train_correct_coverage": pos_coverage,
            "train_wrong_coverage": neg_coverage,
            "train_youden": youden,
            "train_balanced_accuracy": balanced_accuracy,
        }
        if best is None or (
            row["train_youden"], row["train_balanced_accuracy"], -row["threshold"]
        ) > (
            best["train_youden"],
            best["train_balanced_accuracy"],
            -best["threshold"],
        ):
            best = row
    if best is None:
        threshold = float(np.quantile(positive, 1.0 - float(max_coverage)))
        pos_coverage = float(np.mean(positive >= threshold))
        neg_coverage = float(np.mean(negative >= threshold))
        best = {
            "threshold": threshold,
            "train_correct_coverage": pos_coverage,
            "train_wrong_coverage": neg_coverage,
            "train_youden": pos_coverage - neg_coverage,
            "train_balanced_accuracy": 0.5 * (
                pos_coverage + (1.0 - neg_coverage)
            ),
        }
    return best


def top30_uplift(confidence: np.ndarray, advantage: np.ndarray) -> dict[str, float]:
    confidence = np.asarray(confidence, dtype=np.float64)
    advantage = np.asarray(advantage, dtype=np.float64)
    valid = np.isfinite(confidence) & np.isfinite(advantage)
    confidence = confidence[valid]
    advantage = advantage[valid]
    if confidence.size == 0:
        return {
            "overall_mean": 0.0,
            "top30_mean": 0.0,
            "absolute_uplift": 0.0,
            "relative_uplift": 0.0,
        }
    threshold = float(np.quantile(confidence, 0.70))
    top = advantage[confidence >= threshold]
    overall = float(np.mean(advantage))
    top_mean = float(np.mean(top)) if top.size else 0.0
    absolute = top_mean - overall
    relative = absolute / max(abs(overall), 1.0e-6)
    return {
        "confidence_q70": threshold,
        "overall_mean": overall,
        "top30_mean": top_mean,
        "absolute_uplift": absolute,
        "relative_uplift": relative,
    }


def deterministic_subsample(arrays: tuple[np.ndarray, ...], max_count: int) -> tuple[np.ndarray, ...]:
    if not arrays:
        return arrays
    size = int(arrays[0].size)
    if any(int(array.size) != size for array in arrays):
        raise ValueError("subsample arrays must have equal size")
    if int(max_count) <= 0 or size <= int(max_count):
        return arrays
    positions = np.linspace(0, size - 1, num=int(max_count)).round().astype(np.int64)
    return tuple(array[positions] for array in arrays)


def transform_records(
    arrays: dict[str, np.ndarray],
    *,
    prefix: str,
    method: str,
    radius: int,
    topk: int,
    batch_size: int,
) -> dict[str, np.ndarray]:
    support_key = (
        "deployment_common_support"
        if prefix == "deployment"
        else "probe_common_source_support"
    )
    support = torch.from_numpy(arrays[support_key].astype(np.float32))
    result: dict[str, np.ndarray] = {}
    for branch in ("correct", "wrong", "shuffle"):
        values = torch.from_numpy(
            arrays[f"{prefix}_{branch}_confidence"].astype(np.float32)
        )
        transformed = transform_confidence_batched(
            values,
            support,
            method=method,
            radius=int(radius),
            topk=int(topk),
            batch_size=int(batch_size),
        )
        result[branch] = transformed.cpu().numpy().astype(np.float32)
    return result


def summarize_values(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": mean(values),
        "median": median(values),
        "min": min(values),
        "max": max(values),
    }


def object_metrics(
    arrays: dict[str, np.ndarray],
    transformed_deployment: dict[str, np.ndarray],
    transformed_probe: dict[str, np.ndarray],
    *,
    mode_index: int,
) -> dict[str, Any]:
    dep_rows: dict[int, list[tuple[float, float]]] = defaultdict(list)
    dep_mask = arrays["deployment_mode_index"] == int(mode_index)
    for row_index in np.nonzero(dep_mask)[0]:
        valid = arrays["deployment_common_support"][row_index].astype(bool)
        if not np.any(valid):
            continue
        object_index = int(arrays["deployment_object_index"][row_index])
        correct = float(np.mean(transformed_deployment["correct"][row_index][valid]))
        wrong = float(np.mean(transformed_deployment["wrong"][row_index][valid]))
        shuffle = float(np.mean(transformed_deployment["shuffle"][row_index][valid]))
        dep_rows[object_index].append((correct - wrong, correct - shuffle))

    dep_correct_wrong: list[float] = []
    dep_correct_shuffle: list[float] = []
    for rows in dep_rows.values():
        dep_correct_wrong.append(mean([row[0] for row in rows]))
        dep_correct_shuffle.append(mean([row[1] for row in rows]))

    probe_rows: dict[int, list[tuple[float, float, float, float]]] = defaultdict(list)
    probe_mask = arrays["probe_mode_index"] == int(mode_index)
    for row_index in np.nonzero(probe_mask)[0]:
        valid = arrays["probe_valid_mask"][row_index].astype(bool)
        if not np.any(valid):
            continue
        object_index = int(arrays["probe_object_index"][row_index])
        correct = transformed_probe["correct"][row_index][valid]
        wrong = transformed_probe["wrong"][row_index][valid]
        shuffle = transformed_probe["shuffle"][row_index][valid]
        reproj = arrays["probe_reprojection_advantage"][row_index][valid]
        shuffle_reproj = arrays["probe_shuffle_reprojection_advantage"][row_index][valid]
        probe_rows[object_index].append(
            (
                float(np.mean(correct - wrong)),
                float(np.mean(correct - shuffle)),
                float(np.mean(reproj)),
                float(np.mean(shuffle_reproj)),
            )
        )

    probe_correct_wrong: list[float] = []
    probe_correct_shuffle: list[float] = []
    probe_reproj: list[float] = []
    probe_shuffle_reproj: list[float] = []
    for rows in probe_rows.values():
        probe_correct_wrong.append(mean([row[0] for row in rows]))
        probe_correct_shuffle.append(mean([row[1] for row in rows]))
        probe_reproj.append(mean([row[2] for row in rows]))
        probe_shuffle_reproj.append(mean([row[3] for row in rows]))

    return {
        "deployment_object_count": len(dep_correct_wrong),
        "deployment_correct_minus_wrong": summarize_values(dep_correct_wrong),
        "deployment_correct_greater_wrong_rate": (
            sum(value > 0.0 for value in dep_correct_wrong) / len(dep_correct_wrong)
            if dep_correct_wrong
            else 0.0
        ),
        "deployment_correct_minus_shuffle": summarize_values(dep_correct_shuffle),
        "deployment_correct_greater_shuffle_rate": (
            sum(value > 0.0 for value in dep_correct_shuffle)
            / len(dep_correct_shuffle)
            if dep_correct_shuffle
            else 0.0
        ),
        "probe_object_count": len(probe_correct_wrong),
        "probe_correct_minus_wrong": summarize_values(probe_correct_wrong),
        "probe_correct_greater_wrong_rate": (
            sum(value > 0.0 for value in probe_correct_wrong)
            / len(probe_correct_wrong)
            if probe_correct_wrong
            else 0.0
        ),
        "probe_correct_minus_shuffle": summarize_values(probe_correct_shuffle),
        "probe_correct_greater_shuffle_rate": (
            sum(value > 0.0 for value in probe_correct_shuffle)
            / len(probe_correct_shuffle)
            if probe_correct_shuffle
            else 0.0
        ),
        "probe_reprojection_advantage": summarize_values(probe_reproj),
        "probe_reprojection_win_rate": (
            sum(value > 0.0 for value in probe_reproj) / len(probe_reproj)
            if probe_reproj
            else 0.0
        ),
        "probe_shuffle_reprojection_advantage": summarize_values(
            probe_shuffle_reproj
        ),
    }


def collect_mode_voxels(
    arrays: dict[str, np.ndarray],
    transformed_probe: dict[str, np.ndarray],
    *,
    mode_index: int,
    max_count: int,
) -> dict[str, np.ndarray]:
    row_mask = arrays["probe_mode_index"] == int(mode_index)
    valid = arrays["probe_valid_mask"][row_mask].astype(bool)
    correct = transformed_probe["correct"][row_mask][valid]
    wrong = transformed_probe["wrong"][row_mask][valid]
    shuffle = transformed_probe["shuffle"][row_mask][valid]
    advantage = arrays["probe_reprojection_advantage"][row_mask][valid]
    shuffle_advantage = arrays["probe_shuffle_reprojection_advantage"][row_mask][valid]
    correct, wrong, shuffle, advantage, shuffle_advantage = deterministic_subsample(
        (correct, wrong, shuffle, advantage, shuffle_advantage),
        int(max_count),
    )
    return {
        "correct": correct.astype(np.float32),
        "wrong": wrong.astype(np.float32),
        "shuffle": shuffle.astype(np.float32),
        "advantage": advantage.astype(np.float32),
        "shuffle_advantage": shuffle_advantage.astype(np.float32),
    }


def analyze_method(
    report: dict[str, Any],
    arrays: dict[str, np.ndarray],
    *,
    method: str,
    radius: int,
    topk: int,
    batch_size: int,
    max_voxels_per_mode: int,
    min_gate_coverage: float,
    max_gate_coverage: float,
) -> dict[str, Any]:
    mode_map = {str(key): int(value) for key, value in report["mode_to_index"].items()}
    transformed_deployment = transform_records(
        arrays,
        prefix="deployment",
        method=method,
        radius=radius,
        topk=topk,
        batch_size=batch_size,
    )
    transformed_probe = transform_records(
        arrays,
        prefix="probe",
        method=method,
        radius=radius,
        topk=topk,
        batch_size=batch_size,
    )

    per_mode: dict[str, Any] = {}
    all_positive: list[np.ndarray] = []
    all_negative: list[np.ndarray] = []
    for mode in MODES:
        if mode not in mode_map:
            raise KeyError(f"missing mode={mode}")
        samples = collect_mode_voxels(
            arrays,
            transformed_probe,
            mode_index=mode_map[mode],
            max_count=max_voxels_per_mode,
        )
        threshold = choose_threshold(
            samples["correct"],
            samples["wrong"],
            min_coverage=min_gate_coverage,
            max_coverage=max_gate_coverage,
        )
        auc_wrong = binary_auc(samples["correct"], samples["wrong"])
        auc_shuffle = binary_auc(samples["correct"], samples["shuffle"])
        threshold["auc_correct_vs_wrong"] = auc_wrong
        threshold["auc_correct_vs_shuffle"] = auc_shuffle
        per_mode[mode] = {
            "threshold": threshold,
            "object_metrics": object_metrics(
                arrays,
                transformed_deployment,
                transformed_probe,
                mode_index=mode_map[mode],
            ),
            "voxel_samples": samples,
        }
        all_positive.append(samples["correct"])
        all_negative.append(samples["wrong"])

    pooled_positive = np.concatenate(all_positive)
    pooled_negative = np.concatenate(all_negative)
    pooled_threshold = choose_threshold(
        pooled_positive,
        pooled_negative,
        min_coverage=min_gate_coverage,
        max_coverage=max_gate_coverage,
    )
    tau_low = float(pooled_threshold["threshold"])
    tau_high = max(float(np.quantile(pooled_positive, 0.90)), tau_low + 1.0e-3)
    thresholds = [float(per_mode[mode]["threshold"]["threshold"]) for mode in MODES]
    train_selection_score = mean(
        [
            min(
                float(per_mode[mode]["threshold"]["auc_correct_vs_wrong"]),
                float(per_mode[mode]["threshold"]["auc_correct_vs_shuffle"]),
            )
            for mode in MODES
        ]
    )
    return {
        "method": method,
        "tau_low": tau_low,
        "tau_high": tau_high,
        "pooled_threshold": pooled_threshold,
        "per_mode_threshold_spread": max(thresholds) - min(thresholds),
        "selection_score": train_selection_score,
        "per_mode": per_mode,
    }


def strip_samples(method_row: dict[str, Any]) -> dict[str, Any]:
    result = {key: value for key, value in method_row.items() if key != "per_mode"}
    result["per_mode"] = {}
    for mode, row in method_row["per_mode"].items():
        result["per_mode"][mode] = {
            "threshold": row["threshold"],
            "object_metrics": row["object_metrics"],
        }
    return result


def fresh_metrics_for_method(
    train_method: dict[str, Any],
    fresh_report: dict[str, Any],
    fresh_arrays: dict[str, np.ndarray],
    *,
    method: str,
    radius: int,
    topk: int,
    batch_size: int,
    max_voxels_per_mode: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    mode_map = {
        str(key): int(value) for key, value in fresh_report["mode_to_index"].items()
    }
    transformed_deployment = transform_records(
        fresh_arrays,
        prefix="deployment",
        method=method,
        radius=radius,
        topk=topk,
        batch_size=batch_size,
    )
    transformed_probe = transform_records(
        fresh_arrays,
        prefix="probe",
        method=method,
        radius=radius,
        topk=topk,
        batch_size=batch_size,
    )
    tau_low = float(train_method["tau_low"])
    per_mode: dict[str, Any] = {}
    for mode in MODES:
        samples = collect_mode_voxels(
            fresh_arrays,
            transformed_probe,
            mode_index=mode_map[mode],
            max_count=max_voxels_per_mode,
        )
        correct = samples["correct"]
        wrong = samples["wrong"]
        shuffle = samples["shuffle"]
        object_row = object_metrics(
            fresh_arrays,
            transformed_deployment,
            transformed_probe,
            mode_index=mode_map[mode],
        )
        correct_coverage = float(np.mean(correct >= tau_low)) if correct.size else 0.0
        wrong_coverage = float(np.mean(wrong >= tau_low)) if wrong.size else 0.0
        shuffle_coverage = float(np.mean(shuffle >= tau_low)) if shuffle.size else 0.0
        selectivity = correct_coverage - wrong_coverage
        shuffle_selectivity = correct_coverage - shuffle_coverage
        auc_wrong = binary_auc(correct, wrong)
        auc_shuffle = binary_auc(correct, shuffle)
        uplift = top30_uplift(correct, samples["advantage"])
        criteria = {
            "enough_objects": (
                int(object_row["deployment_object_count"]) >= int(args.min_objects)
            ),
            "deployment_object_win_rate": (
                float(object_row["deployment_correct_greater_wrong_rate"])
                >= float(args.min_object_win_rate)
            ),
            "deployment_shuffle_object_win_rate": (
                float(object_row["deployment_correct_greater_shuffle_rate"])
                >= float(args.min_object_win_rate)
            ),
            "fresh_auc_correct_vs_wrong": auc_wrong >= float(args.min_auc),
            "fresh_auc_correct_vs_shuffle": auc_shuffle >= float(args.min_auc),
            "gate_coverage": (
                float(args.min_gate_coverage)
                <= correct_coverage
                <= float(args.max_gate_coverage)
            ),
            "gate_selectivity": selectivity >= float(args.min_gate_selectivity),
            "shuffle_selectivity": (
                shuffle_selectivity >= float(args.min_gate_selectivity)
            ),
            "top30_reprojection_uplift": (
                float(uplift["relative_uplift"])
                >= float(args.min_top30_relative_uplift)
                or float(uplift["absolute_uplift"])
                >= float(args.min_top30_absolute_uplift)
            ),
        }
        per_mode[mode] = {
            "passed": all(criteria.values()),
            "criteria": criteria,
            "object_metrics": object_row,
            "voxel_metrics": {
                "voxel_sample_count": int(correct.size),
                "auc_correct_vs_wrong": auc_wrong,
                "auc_correct_vs_shuffle": auc_shuffle,
                "correct_confidence": {
                    "mean": float(np.mean(correct)),
                    "median": float(np.median(correct)),
                },
                "wrong_confidence": {
                    "mean": float(np.mean(wrong)),
                    "median": float(np.median(wrong)),
                },
                "shuffle_confidence": {
                    "mean": float(np.mean(shuffle)),
                    "median": float(np.median(shuffle)),
                },
                "correct_gate_coverage": correct_coverage,
                "wrong_gate_coverage": wrong_coverage,
                "shuffle_gate_coverage": shuffle_coverage,
                "correct_minus_wrong_gate_coverage": selectivity,
                "correct_minus_shuffle_gate_coverage": shuffle_selectivity,
                "top30_reprojection_uplift": uplift,
            },
        }
    return {
        "method": method,
        "tau_low": float(train_method["tau_low"]),
        "tau_high": float(train_method["tau_high"]),
        "per_mode": per_mode,
    }


def select_method(train_methods: dict[str, dict[str, Any]]) -> str:
    # Selection uses train objects only. Equal scores prefer the simpler method.
    complexity = {"raw": 0, "local_mean": 1, "local_topk": 2}
    return max(
        train_methods,
        key=lambda method: (
            float(train_methods[method]["selection_score"]),
            -complexity.get(method, 99),
        ),
    )


def main() -> None:
    args = parse_args()
    methods = parse_methods(args.methods)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)

    train_report = load_json(args.train_report)
    fresh_report = load_json(args.fresh_report)
    train_arrays = load_npz(args.train_volumes)
    fresh_arrays = load_npz(args.fresh_volumes)
    if train_report["mode_to_index"] != fresh_report["mode_to_index"]:
        raise ValueError("train/fresh mode maps differ")
    if int(train_report["protocol"]["volume_side"]) != int(
        fresh_report["protocol"]["volume_side"]
    ):
        raise ValueError("train/fresh volume sides differ")

    train_methods: dict[str, dict[str, Any]] = {}
    for method in methods:
        train_methods[method] = analyze_method(
            train_report,
            train_arrays,
            method=method,
            radius=int(args.local_radius),
            topk=int(args.local_topk),
            batch_size=int(args.transform_batch_size),
            max_voxels_per_mode=int(args.max_voxels_per_mode),
            min_gate_coverage=float(args.min_gate_coverage),
            max_gate_coverage=float(args.max_gate_coverage),
        )

    selected_method = select_method(train_methods)
    fresh_methods: dict[str, dict[str, Any]] = {}
    for method in methods:
        fresh_methods[method] = fresh_metrics_for_method(
            train_methods[method],
            fresh_report,
            fresh_arrays,
            method=method,
            radius=int(args.local_radius),
            topk=int(args.local_topk),
            batch_size=int(args.transform_batch_size),
            max_voxels_per_mode=int(args.max_voxels_per_mode),
            args=args,
        )

    if "raw" not in train_methods:
        raise ValueError("raw must be included as the fixed baseline")
    selected_train = train_methods[selected_method]
    selected_fresh = fresh_methods[selected_method]
    raw_train = train_methods["raw"]
    raw_fresh = fresh_methods["raw"]

    train_gain = float(selected_train["selection_score"]) - float(
        raw_train["selection_score"]
    )
    fresh_wrong_gains = [
        float(selected_fresh["per_mode"][mode]["voxel_metrics"]["auc_correct_vs_wrong"])
        - float(raw_fresh["per_mode"][mode]["voxel_metrics"]["auc_correct_vs_wrong"])
        for mode in MODES
    ]
    fresh_shuffle_gains = [
        float(selected_fresh["per_mode"][mode]["voxel_metrics"]["auc_correct_vs_shuffle"])
        - float(raw_fresh["per_mode"][mode]["voxel_metrics"]["auc_correct_vs_shuffle"])
        for mode in MODES
    ]
    fresh_mean_auc_gain = mean(fresh_wrong_gains)

    global_checks = {
        "selected_method_is_local": selected_method != "raw",
        "train_selection_gain": train_gain >= float(args.min_train_selection_gain),
        "fresh_mean_auc_gain_vs_raw": (
            fresh_mean_auc_gain >= float(args.min_fresh_mean_auc_gain)
        ),
        "no_fresh_wrong_auc_mode_drop": (
            min(fresh_wrong_gains) >= -float(args.max_fresh_mode_auc_drop)
        ),
        "no_fresh_shuffle_auc_mode_drop": (
            min(fresh_shuffle_gains) >= -float(args.max_fresh_mode_auc_drop)
        ),
        "per_mode_threshold_spread": (
            float(selected_train["per_mode_threshold_spread"])
            <= float(args.max_mode_threshold_spread)
        ),
        "all_selected_method_modes_passed": all(
            bool(selected_fresh["per_mode"][mode]["passed"]) for mode in MODES
        ),
    }
    passed = all(global_checks.values())

    calibration = {
        "format": "ar_ss_flow.visual_only_pairwise_local_gate_calibration.v2",
        "checkpoint": fresh_report["protocol"]["checkpoint"],
        "checkpoint_step": fresh_report["protocol"]["checkpoint_step"],
        "confidence_source": "visual_only_pairwise_confidence",
        "geometry_pair_scale": 0.0,
        "aggregation_method": selected_method,
        "local_radius": int(args.local_radius),
        "local_kernel_size": 2 * int(args.local_radius) + 1,
        "local_topk": int(args.local_topk),
        "support_policy": "common_binary_source_support_shared_by_all_branches",
        "tau_low": float(selected_train["tau_low"]),
        "tau_high": float(selected_train["tau_high"]),
        "gate_formula": "clamp((local_confidence-tau_low)/(tau_high-tau_low),0,1)",
        "selected_on_indices": train_report["protocol"]["indices"],
        "validated_on_indices": fresh_report["protocol"]["indices"],
    }

    report = {
        "stage": "C1.5-v2 local visual-only pairwise deployment calibration",
        "passed": passed,
        "args": vars(args),
        "selected_method": selected_method,
        "calibration": calibration,
        "train_method_selection": {
            method: strip_samples(row) for method, row in train_methods.items()
        },
        "fresh_method_comparison": fresh_methods,
        "selected_vs_raw": {
            "train_selection_score_gain": train_gain,
            "fresh_auc_correct_vs_wrong_gain_by_mode": dict(
                zip(MODES, fresh_wrong_gains)
            ),
            "fresh_auc_correct_vs_shuffle_gain_by_mode": dict(
                zip(MODES, fresh_shuffle_gains)
            ),
            "fresh_mean_auc_correct_vs_wrong_gain": fresh_mean_auc_gain,
        },
        "global_checks": global_checks,
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    (output_dir / "calibration.json").write_text(
        json.dumps(calibration, indent=2), encoding="utf-8"
    )

    print("=" * 100)
    print(report["stage"])
    for method in methods:
        print(
            f"train method={method} "
            f"selection_score={train_methods[method]['selection_score']:.4f} "
            f"tau_low={train_methods[method]['tau_low']:.6f} "
            f"tau_high={train_methods[method]['tau_high']:.6f}"
        )
    print("selected_method:", selected_method)
    print(f"train_selection_gain_vs_raw={train_gain:+.4f}")
    print(f"fresh_mean_wrong_auc_gain_vs_raw={fresh_mean_auc_gain:+.4f}")
    for mode in MODES:
        selected_row = selected_fresh["per_mode"][mode]
        raw_row = raw_fresh["per_mode"][mode]
        sv = selected_row["voxel_metrics"]
        rv = raw_row["voxel_metrics"]
        obj = selected_row["object_metrics"]
        print(
            f"{mode}: passed={selected_row['passed']} "
            f"obj_win={obj['deployment_correct_greater_wrong_rate']:.4f} "
            f"auc={sv['auc_correct_vs_wrong']:.4f} "
            f"raw_auc={rv['auc_correct_vs_wrong']:.4f} "
            f"gain={sv['auc_correct_vs_wrong'] - rv['auc_correct_vs_wrong']:+.4f} "
            f"shuffle_auc={sv['auc_correct_vs_shuffle']:.4f} "
            f"coverage={sv['correct_gate_coverage']:.4f} "
            f"selectivity={sv['correct_minus_wrong_gate_coverage']:+.4f} "
            f"top30={sv['top30_reprojection_uplift']['relative_uplift']:+.4f}"
        )
    print("global_checks:", global_checks)
    print("passed:", passed)
    print("report:", output_dir / "report.json")
    print("calibration:", output_dir / "calibration.json")

    if args.fail_on_error and not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
