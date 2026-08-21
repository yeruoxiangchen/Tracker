#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np

MODES = ("pose_cyclic1", "pose_cyclic2", "pose_reverse")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Choose a visual-only pairwise confidence threshold on training objects "
            "and evaluate the frozen threshold on fresh objects."
        )
    )
    parser.add_argument("--train_report", required=True)
    parser.add_argument("--train_voxels", required=True)
    parser.add_argument("--fresh_report", required=True)
    parser.add_argument("--fresh_voxels", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--min_objects", type=int, default=16)
    parser.add_argument("--min_object_win_rate", type=float, default=0.80)
    parser.add_argument("--min_auc", type=float, default=0.70)
    parser.add_argument("--min_gate_coverage", type=float, default=0.20)
    parser.add_argument("--max_gate_coverage", type=float, default=0.80)
    parser.add_argument("--min_gate_selectivity", type=float, default=0.10)
    parser.add_argument("--min_top30_relative_uplift", type=float, default=0.25)
    parser.add_argument("--min_top30_absolute_uplift", type=float, default=0.002)
    parser.add_argument("--max_mode_threshold_spread", type=float, default=0.15)
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
    with np.load(path) as payload:
        return {name: payload[name] for name in payload.files}


def binary_auc(positive: np.ndarray, negative: np.ndarray) -> float:
    positive = np.asarray(positive, dtype=np.float64).reshape(-1)
    negative = np.asarray(negative, dtype=np.float64).reshape(-1)
    positive = positive[np.isfinite(positive)]
    negative = negative[np.isfinite(negative)]
    if positive.size == 0 or negative.size == 0:
        return 0.5
    values = np.concatenate([positive, negative])
    labels = np.concatenate(
        [np.ones(positive.size, dtype=np.int8), np.zeros(negative.size, dtype=np.int8)]
    )
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    cursor = 0
    while cursor < values.size:
        end = cursor + 1
        while end < values.size and sorted_values[end] == sorted_values[cursor]:
            end += 1
        average_rank = 0.5 * ((cursor + 1) + end)
        ranks[order[cursor:end]] = average_rank
        cursor = end
    rank_sum_positive = ranks[labels == 1].sum()
    auc = (
        rank_sum_positive - positive.size * (positive.size + 1) / 2.0
    ) / (positive.size * negative.size)
    return float(auc)


def candidate_thresholds(positive: np.ndarray, negative: np.ndarray) -> np.ndarray:
    values = np.concatenate([positive, negative]).astype(np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.asarray([0.5], dtype=np.float64)
    quantiles = np.linspace(0.02, 0.98, 97)
    candidates = np.quantile(values, quantiles)
    return np.unique(candidates)


def choose_threshold(
    positive: np.ndarray,
    negative: np.ndarray,
    *,
    min_coverage: float,
    max_coverage: float,
) -> dict[str, float]:
    positive = np.asarray(positive, dtype=np.float64)
    negative = np.asarray(negative, dtype=np.float64)
    best: tuple[float, float, float, float, float] | None = None
    fallback: tuple[float, float, float, float, float] | None = None
    for threshold in candidate_thresholds(positive, negative):
        tpr = float(np.mean(positive >= threshold))
        fpr = float(np.mean(negative >= threshold))
        youden = tpr - fpr
        balanced = 0.5 * (tpr + (1.0 - fpr))
        row = (youden, balanced, -abs(tpr - 0.5), float(threshold), tpr)
        if fallback is None or row > fallback:
            fallback = row
        if min_coverage <= tpr <= max_coverage:
            if best is None or row > best:
                best = row
    chosen = best if best is not None else fallback
    if chosen is None:
        return {
            "threshold": 0.5,
            "train_correct_coverage": 0.0,
            "train_wrong_coverage": 0.0,
            "train_youden": 0.0,
            "train_balanced_accuracy": 0.5,
        }
    threshold = float(chosen[3])
    tpr = float(np.mean(positive >= threshold))
    fpr = float(np.mean(negative >= threshold))
    return {
        "threshold": threshold,
        "train_correct_coverage": tpr,
        "train_wrong_coverage": fpr,
        "train_youden": tpr - fpr,
        "train_balanced_accuracy": 0.5 * (tpr + (1.0 - fpr)),
    }


def top30_uplift(confidence: np.ndarray, advantage: np.ndarray) -> dict[str, float | bool]:
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


def object_metrics(report: dict[str, Any], mode: str) -> dict[str, float]:
    deployment = report["deployment_summary"][mode]
    probe = report["heldout_probe_summary"][mode]
    return {
        "deployment_object_count": float(deployment["object_count"]),
        "deployment_correct_greater_wrong_rate": float(
            deployment["correct_greater_wrong_object_rate"]
        ),
        "deployment_correct_greater_shuffle_rate": float(
            deployment["correct_greater_shuffle_object_rate"]
        ),
        "deployment_correct_minus_wrong_mean": float(
            deployment["correct_minus_wrong_confidence"]["mean"]
        ),
        "deployment_correct_minus_shuffle_mean": float(
            deployment["correct_minus_shuffle_confidence"]["mean"]
        ),
        "probe_object_count": float(probe["object_count"]),
        "probe_correct_greater_wrong_rate": float(
            probe["correct_greater_wrong_object_rate"]
        ),
        "probe_correct_greater_shuffle_rate": float(
            probe["correct_greater_shuffle_object_rate"]
        ),
        "probe_reprojection_win_rate": float(probe["reprojection_object_win_rate"]),
        "probe_reprojection_mean": float(probe["reprojection_advantage"]["mean"]),
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)

    train_report = load_json(args.train_report)
    fresh_report = load_json(args.fresh_report)
    train = load_npz(args.train_voxels)
    fresh = load_npz(args.fresh_voxels)

    train_mode_map = {str(k): int(v) for k, v in train_report["mode_to_index"].items()}
    fresh_mode_map = {str(k): int(v) for k, v in fresh_report["mode_to_index"].items()}
    if train_mode_map != fresh_mode_map:
        raise ValueError("train/fresh mode index maps differ")

    all_train_positive: list[np.ndarray] = []
    all_train_negative: list[np.ndarray] = []
    per_mode_threshold: dict[str, Any] = {}
    for mode in MODES:
        index = train_mode_map[mode]
        mask = train["mode_index"] == index
        positive = train["correct_confidence"][mask]
        negative = train["wrong_confidence"][mask]
        selected = choose_threshold(
            positive,
            negative,
            min_coverage=float(args.min_gate_coverage),
            max_coverage=float(args.max_gate_coverage),
        )
        selected["train_auc_correct_vs_wrong"] = binary_auc(positive, negative)
        selected["train_auc_correct_vs_shuffle"] = binary_auc(
            positive, train["shuffle_confidence"][mask]
        )
        per_mode_threshold[mode] = selected
        all_train_positive.append(positive)
        all_train_negative.append(negative)

    pooled_threshold = choose_threshold(
        np.concatenate(all_train_positive),
        np.concatenate(all_train_negative),
        min_coverage=float(args.min_gate_coverage),
        max_coverage=float(args.max_gate_coverage),
    )
    tau_low = float(pooled_threshold["threshold"])
    pooled_positive = np.concatenate(all_train_positive)
    tau_high = float(np.quantile(pooled_positive, 0.90))
    tau_high = max(tau_high, tau_low + 1.0e-3)

    threshold_values = [float(row["threshold"]) for row in per_mode_threshold.values()]
    threshold_spread = max(threshold_values) - min(threshold_values)

    per_mode: dict[str, Any] = {}
    mode_passes: list[bool] = []
    for mode in MODES:
        index = fresh_mode_map[mode]
        mask = fresh["mode_index"] == index
        correct = fresh["correct_confidence"][mask]
        wrong = fresh["wrong_confidence"][mask]
        shuffle = fresh["shuffle_confidence"][mask]
        advantage = fresh["reprojection_advantage"][mask]
        object_row = object_metrics(fresh_report, mode)

        correct_coverage = float(np.mean(correct >= tau_low)) if correct.size else 0.0
        wrong_coverage = float(np.mean(wrong >= tau_low)) if wrong.size else 0.0
        shuffle_coverage = float(np.mean(shuffle >= tau_low)) if shuffle.size else 0.0
        selectivity = correct_coverage - wrong_coverage
        shuffle_selectivity = correct_coverage - shuffle_coverage
        uplift = top30_uplift(correct, advantage)

        criteria = {
            "enough_objects": (
                int(object_row["deployment_object_count"]) >= int(args.min_objects)
            ),
            "deployment_object_win_rate": (
                object_row["deployment_correct_greater_wrong_rate"]
                >= float(args.min_object_win_rate)
            ),
            "deployment_shuffle_object_win_rate": (
                object_row["deployment_correct_greater_shuffle_rate"]
                >= float(args.min_object_win_rate)
            ),
            "fresh_auc_correct_vs_wrong": (
                binary_auc(correct, wrong) >= float(args.min_auc)
            ),
            "fresh_auc_correct_vs_shuffle": (
                binary_auc(correct, shuffle) >= float(args.min_auc)
            ),
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
        mode_passed = all(criteria.values())
        mode_passes.append(mode_passed)
        per_mode[mode] = {
            "passed": mode_passed,
            "criteria": criteria,
            "object_metrics": object_row,
            "voxel_metrics": {
                "voxel_count": int(correct.size),
                "auc_correct_vs_wrong": binary_auc(correct, wrong),
                "auc_correct_vs_shuffle": binary_auc(correct, shuffle),
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

    global_checks = {
        "per_mode_threshold_spread": (
            threshold_spread <= float(args.max_mode_threshold_spread)
        ),
        "all_modes_passed": all(mode_passes),
    }
    passed = all(global_checks.values())

    calibration = {
        "format": "ar_ss_flow.visual_only_pairwise_gate_calibration.v1",
        "checkpoint": fresh_report["protocol"]["checkpoint"],
        "checkpoint_step": fresh_report["protocol"]["checkpoint_step"],
        "confidence_source": "visual_only_pairwise_confidence",
        "geometry_pair_scale": 0.0,
        "tau_low": tau_low,
        "tau_high": tau_high,
        "gate_formula": "clamp((confidence-tau_low)/(tau_high-tau_low),0,1)",
        "selected_on_indices": train_report["protocol"]["indices"],
        "validated_on_indices": fresh_report["protocol"]["indices"],
    }

    report = {
        "stage": "C1.5 visual-only pairwise deployment calibration",
        "passed": passed,
        "args": vars(args),
        "calibration": calibration,
        "pooled_train_threshold": pooled_threshold,
        "per_mode_train_threshold": per_mode_threshold,
        "per_mode_threshold_spread": threshold_spread,
        "global_checks": global_checks,
        "fresh_per_mode": per_mode,
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    (output_dir / "calibration.json").write_text(
        json.dumps(calibration, indent=2), encoding="utf-8"
    )

    print("=" * 100)
    print(report["stage"])
    print(f"tau_low={tau_low:.6f} tau_high={tau_high:.6f}")
    print(f"per_mode_threshold_spread={threshold_spread:.6f}")
    for mode in MODES:
        row = per_mode[mode]
        voxel = row["voxel_metrics"]
        obj = row["object_metrics"]
        print(
            f"{mode}: passed={row['passed']} "
            f"obj_win={obj['deployment_correct_greater_wrong_rate']:.4f} "
            f"auc={voxel['auc_correct_vs_wrong']:.4f} "
            f"shuffle_auc={voxel['auc_correct_vs_shuffle']:.4f} "
            f"coverage={voxel['correct_gate_coverage']:.4f} "
            f"selectivity={voxel['correct_minus_wrong_gate_coverage']:+.4f} "
            f"top30_uplift={voxel['top30_reprojection_uplift']['relative_uplift']:+.4f}"
        )
    print("global_checks:", global_checks)
    print("passed:", passed)
    print("report:", output_dir / "report.json")
    print("calibration:", output_dir / "calibration.json")

    if args.fail_on_error and not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
