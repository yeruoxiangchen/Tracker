#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np

TRACKER_ROOT = Path(__file__).resolve().parents[1]
if str(TRACKER_ROOT) not in sys.path:
    sys.path.insert(0, str(TRACKER_ROOT))

from ar_ss_flow.pairwise_percentile_gate import (
    SelectorMeans,
    aggregate_selector_means,
    evaluate_object_selectors,
    parse_fractions,
)

MODES = ("pose_cyclic1", "pose_cyclic2", "pose_reverse")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select a per-volume object-internal top-quantile gate using raw visual-only "
            "pairwise confidence on train objects, then evaluate the selected fraction "
            "without adjustment on fresh objects. Equal-coverage random, wrong-confidence, "
            "and visual-shuffle-confidence selectors are explicit controls."
        )
    )
    parser.add_argument("--train_report", required=True)
    parser.add_argument("--train_volumes", required=True)
    parser.add_argument("--fresh_report", required=True)
    parser.add_argument("--fresh_volumes", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--fractions", default="0.20,0.30,0.40")
    parser.add_argument("--random_trials", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min_objects", type=int, default=8)
    parser.add_argument("--min_deployment_object_win_rate", type=float, default=0.80)
    parser.add_argument("--min_selected_advantage", type=float, default=0.0)
    parser.add_argument("--min_relative_uplift_vs_overall", type=float, default=0.25)
    parser.add_argument("--min_absolute_uplift_vs_overall", type=float, default=0.002)
    parser.add_argument("--min_gain_vs_random", type=float, default=0.001)
    parser.add_argument("--min_object_win_vs_random", type=float, default=0.65)
    parser.add_argument("--min_gain_vs_wrong_selector", type=float, default=0.0)
    parser.add_argument("--min_object_win_vs_wrong_selector", type=float, default=0.60)
    parser.add_argument("--min_shuffle_gain_vs_random", type=float, default=0.001)
    parser.add_argument("--min_shuffle_object_win_vs_random", type=float, default=0.65)
    parser.add_argument("--min_gain_vs_shuffle_selector", type=float, default=0.0)
    parser.add_argument("--min_object_win_vs_shuffle_selector", type=float, default=0.60)
    parser.add_argument("--min_train_selection_score", type=float, default=0.0)
    parser.add_argument("--fail_on_error", action="store_true")
    return parser.parse_args()


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_npz(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def mean(values: Iterable[float]) -> float:
    values = [float(value) for value in values]
    return float(np.mean(values)) if values else 0.0


def rate(values: Iterable[bool]) -> float:
    values = [bool(value) for value in values]
    return float(sum(values) / len(values)) if values else 0.0


def summarize_values(values: Iterable[float]) -> dict[str, float | int]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return {"count": 0, "mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def validate_schema(arrays: dict[str, np.ndarray]) -> None:
    required = {
        "deployment_mode_index",
        "deployment_object_index",
        "deployment_correct_confidence",
        "deployment_wrong_confidence",
        "deployment_shuffle_confidence",
        "deployment_common_support",
        "probe_mode_index",
        "probe_object_index",
        "probe_correct_confidence",
        "probe_wrong_confidence",
        "probe_shuffle_confidence",
        "probe_valid_mask",
        "probe_reprojection_advantage",
        "probe_shuffle_reprojection_advantage",
    }
    missing = sorted(required.difference(arrays))
    if missing:
        raise KeyError(f"volume archive is missing arrays={missing}")

    dep_rows = int(arrays["deployment_mode_index"].shape[0])
    probe_rows = int(arrays["probe_mode_index"].shape[0])
    for name in (
        "deployment_object_index",
        "deployment_correct_confidence",
        "deployment_wrong_confidence",
        "deployment_shuffle_confidence",
        "deployment_common_support",
    ):
        if int(arrays[name].shape[0]) != dep_rows:
            raise ValueError(f"deployment row count mismatch for {name}")
    for name in (
        "probe_object_index",
        "probe_correct_confidence",
        "probe_wrong_confidence",
        "probe_shuffle_confidence",
        "probe_valid_mask",
        "probe_reprojection_advantage",
        "probe_shuffle_reprojection_advantage",
    ):
        if int(arrays[name].shape[0]) != probe_rows:
            raise ValueError(f"probe row count mismatch for {name}")


def validate_report(report: dict[str, Any], arrays: dict[str, np.ndarray]) -> None:
    protocol = report.get("protocol", {})
    if not bool(protocol.get("visual_only_pairwise", False)):
        raise ValueError("report is not marked visual_only_pairwise")
    if not bool(protocol.get("geometry_pair_scale_forced_zero", False)):
        raise ValueError("report does not confirm geometry_pair_scale_forced_zero")
    if not bool(protocol.get("full_volume_records_saved", False)):
        raise ValueError("report does not contain full-volume records")
    if int(report.get("deployment_record_count", -1)) != int(
        arrays["deployment_mode_index"].shape[0]
    ):
        raise ValueError("deployment record count differs from report")
    if int(report.get("probe_record_count", -1)) != int(
        arrays["probe_mode_index"].shape[0]
    ):
        raise ValueError("probe record count differs from report")


def deployment_object_metrics(
    arrays: dict[str, np.ndarray],
    *,
    mode_index: int,
) -> dict[str, Any]:
    row_ids = np.flatnonzero(arrays["deployment_mode_index"] == int(mode_index))
    grouped: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for row in row_ids:
        support = arrays["deployment_common_support"][row].astype(bool).reshape(-1)
        if not np.any(support):
            continue
        c = arrays["deployment_correct_confidence"][row].reshape(-1)[support]
        w = arrays["deployment_wrong_confidence"][row].reshape(-1)[support]
        s = arrays["deployment_shuffle_confidence"][row].reshape(-1)[support]
        grouped[int(arrays["deployment_object_index"][row])].append(
            (float(np.mean(c - w)), float(np.mean(c - s)))
        )
    correct_wrong = [mean(item[0] for item in rows) for rows in grouped.values()]
    correct_shuffle = [mean(item[1] for item in rows) for rows in grouped.values()]
    return {
        "object_count": len(grouped),
        "correct_minus_wrong": summarize_values(correct_wrong),
        "correct_greater_wrong_rate": rate(value > 0.0 for value in correct_wrong),
        "correct_minus_shuffle": summarize_values(correct_shuffle),
        "correct_greater_shuffle_rate": rate(value > 0.0 for value in correct_shuffle),
    }


def selector_row_to_dict(row: SelectorMeans) -> dict[str, float | int]:
    return {
        "overall_wrong": row.overall_wrong,
        "correct_selected_wrong": row.correct_selected_wrong,
        "wrong_selected_wrong": row.wrong_selected_wrong,
        "shuffle_selected_wrong": row.shuffle_selected_wrong,
        "random_selected_wrong": row.random_selected_wrong,
        "overall_shuffle": row.overall_shuffle,
        "correct_selected_shuffle": row.correct_selected_shuffle,
        "wrong_selected_shuffle": row.wrong_selected_shuffle,
        "shuffle_selected_shuffle": row.shuffle_selected_shuffle,
        "random_selected_shuffle": row.random_selected_shuffle,
        "selected_count": row.selected_count,
        "valid_count": row.valid_count,
    }


def evaluate_mode(
    arrays: dict[str, np.ndarray],
    *,
    mode_index: int,
    fraction: float,
    random_trials: int,
    seed: int,
) -> dict[str, Any]:
    """Select top fraction independently in every held-out volume.

    Each row is one held-out target volume. Row metrics are first averaged per
    object, then objects are treated as independent evaluation units.
    """
    row_ids = np.flatnonzero(arrays["probe_mode_index"] == int(mode_index))
    grouped: dict[int, list[SelectorMeans]] = defaultdict(list)
    for row in row_ids:
        valid = arrays["probe_valid_mask"][row].astype(bool).reshape(-1)
        if not np.any(valid):
            continue
        object_index = int(arrays["probe_object_index"][row])
        grouped[object_index].append(
            evaluate_object_selectors(
                correct_confidence=arrays["probe_correct_confidence"][row],
                wrong_confidence=arrays["probe_wrong_confidence"][row],
                shuffle_confidence=arrays["probe_shuffle_confidence"][row],
                reprojection_advantage=arrays["probe_reprojection_advantage"][row],
                shuffle_reprojection_advantage=arrays[
                    "probe_shuffle_reprojection_advantage"
                ][row],
                valid=valid,
                fraction=float(fraction),
                random_trials=int(random_trials),
                seed=(
                    int(seed)
                    + int(mode_index) * 1000003
                    + int(object_index) * 9176
                    + int(row) * 313
                ),
            )
        )
    objects = {
        object_index: aggregate_selector_means(rows)
        for object_index, rows in grouped.items()
        if rows
    }

    def values(name: str) -> list[float]:
        return [float(getattr(row, name)) for row in objects.values()]

    overall_w = values("overall_wrong")
    correct_w = values("correct_selected_wrong")
    wrong_w = values("wrong_selected_wrong")
    shuffle_w = values("shuffle_selected_wrong")
    random_w = values("random_selected_wrong")
    overall_s = values("overall_shuffle")
    correct_s = values("correct_selected_shuffle")
    wrong_s = values("wrong_selected_shuffle")
    shuffle_s = values("shuffle_selected_shuffle")
    random_s = values("random_selected_shuffle")

    gain_overall_w = [a - b for a, b in zip(correct_w, overall_w)]
    gain_random_w = [a - b for a, b in zip(correct_w, random_w)]
    gain_wrong_selector = [a - b for a, b in zip(correct_w, wrong_w)]
    gain_shuffle_selector_on_wrong = [a - b for a, b in zip(correct_w, shuffle_w)]
    gain_overall_s = [a - b for a, b in zip(correct_s, overall_s)]
    gain_random_s = [a - b for a, b in zip(correct_s, random_s)]
    gain_shuffle_selector = [a - b for a, b in zip(correct_s, shuffle_s)]
    gain_wrong_selector_on_shuffle = [a - b for a, b in zip(correct_s, wrong_s)]

    mean_overall_w = mean(overall_w)
    mean_correct_w = mean(correct_w)
    mean_overall_s = mean(overall_s)
    mean_correct_s = mean(correct_s)
    relative_wrong = (mean_correct_w - mean_overall_w) / max(abs(mean_overall_w), 1.0e-6)
    relative_shuffle = (mean_correct_s - mean_overall_s) / max(abs(mean_overall_s), 1.0e-6)

    result = {
        "object_count": len(objects),
        "fraction": float(fraction),
        "selection_scope": "per_heldout_volume_then_object_mean",
        "selected_voxel_count": summarize_values(
            int(row.selected_count) for row in objects.values()
        ),
        "valid_voxel_count": summarize_values(
            int(row.valid_count) for row in objects.values()
        ),
        "wrong_pose_label": {
            "overall_advantage": summarize_values(overall_w),
            "correct_selector_advantage": summarize_values(correct_w),
            "wrong_selector_advantage": summarize_values(wrong_w),
            "shuffle_selector_advantage": summarize_values(shuffle_w),
            "random_selector_advantage": summarize_values(random_w),
            "correct_minus_overall": summarize_values(gain_overall_w),
            "correct_minus_random": summarize_values(gain_random_w),
            "correct_minus_wrong_selector": summarize_values(gain_wrong_selector),
            "correct_minus_shuffle_selector": summarize_values(
                gain_shuffle_selector_on_wrong
            ),
            "correct_greater_random_object_rate": rate(
                value > 0.0 for value in gain_random_w
            ),
            "correct_greater_wrong_selector_object_rate": rate(
                value > 0.0 for value in gain_wrong_selector
            ),
            "correct_greater_shuffle_selector_object_rate": rate(
                value > 0.0 for value in gain_shuffle_selector_on_wrong
            ),
            "relative_uplift_vs_overall": float(relative_wrong),
        },
        "shuffle_label": {
            "overall_advantage": summarize_values(overall_s),
            "correct_selector_advantage": summarize_values(correct_s),
            "wrong_selector_advantage": summarize_values(wrong_s),
            "shuffle_selector_advantage": summarize_values(shuffle_s),
            "random_selector_advantage": summarize_values(random_s),
            "correct_minus_overall": summarize_values(gain_overall_s),
            "correct_minus_random": summarize_values(gain_random_s),
            "correct_minus_shuffle_selector": summarize_values(gain_shuffle_selector),
            "correct_minus_wrong_selector": summarize_values(
                gain_wrong_selector_on_shuffle
            ),
            "correct_greater_random_object_rate": rate(
                value > 0.0 for value in gain_random_s
            ),
            "correct_greater_shuffle_selector_object_rate": rate(
                value > 0.0 for value in gain_shuffle_selector
            ),
            "correct_greater_wrong_selector_object_rate": rate(
                value > 0.0 for value in gain_wrong_selector_on_shuffle
            ),
            "relative_uplift_vs_overall": float(relative_shuffle),
        },
        "per_object": {
            str(object_index): selector_row_to_dict(row)
            for object_index, row in objects.items()
        },
    }
    result["selection_score"] = float(
        result["wrong_pose_label"]["correct_minus_random"]["mean"]
        + result["wrong_pose_label"]["correct_minus_wrong_selector"]["mean"]
        + result["shuffle_label"]["correct_minus_random"]["mean"]
        + result["shuffle_label"]["correct_minus_shuffle_selector"]["mean"]
    )
    return result


def evaluate_fraction(
    report: dict[str, Any],
    arrays: dict[str, np.ndarray],
    *,
    fraction: float,
    random_trials: int,
    seed: int,
) -> dict[str, Any]:
    mode_map = {str(key): int(value) for key, value in report["mode_to_index"].items()}
    per_mode = {
        mode: evaluate_mode(
            arrays,
            mode_index=mode_map[mode],
            fraction=float(fraction),
            random_trials=int(random_trials),
            seed=int(seed),
        )
        for mode in MODES
    }
    scores = [float(per_mode[mode]["selection_score"]) for mode in MODES]
    return {
        "fraction": float(fraction),
        "per_mode": per_mode,
        "minimum_mode_selection_score": min(scores),
        "mean_mode_selection_score": mean(scores),
        # Minimum mode dominates, mean is only a weak tie-break reward.
        "selection_score": min(scores) + 0.25 * mean(scores),
    }


def add_fresh_criteria(
    fraction_result: dict[str, Any],
    deployment: dict[str, dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], bool]:
    all_passed = True
    for mode in MODES:
        row = fraction_result["per_mode"][mode]
        dep = deployment[mode]
        wrong = row["wrong_pose_label"]
        shuffle = row["shuffle_label"]
        absolute_uplift = float(wrong["correct_minus_overall"]["mean"])
        relative_uplift = float(wrong["relative_uplift_vs_overall"])
        criteria = {
            "enough_objects": int(row["object_count"]) >= int(args.min_objects),
            "deployment_correct_greater_wrong": (
                float(dep["correct_greater_wrong_rate"])
                >= float(args.min_deployment_object_win_rate)
            ),
            "deployment_correct_greater_shuffle": (
                float(dep["correct_greater_shuffle_rate"])
                >= float(args.min_deployment_object_win_rate)
            ),
            "selected_wrong_advantage_positive": (
                float(wrong["correct_selector_advantage"]["mean"])
                > float(args.min_selected_advantage)
            ),
            "uplift_vs_overall": (
                relative_uplift >= float(args.min_relative_uplift_vs_overall)
                or absolute_uplift >= float(args.min_absolute_uplift_vs_overall)
            ),
            "gain_vs_equal_coverage_random": (
                float(wrong["correct_minus_random"]["mean"])
                >= float(args.min_gain_vs_random)
            ),
            "object_win_vs_equal_coverage_random": (
                float(wrong["correct_greater_random_object_rate"])
                >= float(args.min_object_win_vs_random)
            ),
            "gain_vs_wrong_confidence_selector": (
                float(wrong["correct_minus_wrong_selector"]["mean"])
                > float(args.min_gain_vs_wrong_selector)
            ),
            "object_win_vs_wrong_confidence_selector": (
                float(wrong["correct_greater_wrong_selector_object_rate"])
                >= float(args.min_object_win_vs_wrong_selector)
            ),
            "shuffle_gain_vs_equal_coverage_random": (
                float(shuffle["correct_minus_random"]["mean"])
                >= float(args.min_shuffle_gain_vs_random)
            ),
            "shuffle_object_win_vs_equal_coverage_random": (
                float(shuffle["correct_greater_random_object_rate"])
                >= float(args.min_shuffle_object_win_vs_random)
            ),
            "visual_shuffle_degrades_selector": (
                float(shuffle["correct_minus_shuffle_selector"]["mean"])
                > float(args.min_gain_vs_shuffle_selector)
            ),
            "object_win_vs_shuffle_confidence_selector": (
                float(shuffle["correct_greater_shuffle_selector_object_rate"])
                >= float(args.min_object_win_vs_shuffle_selector)
            ),
        }
        row["deployment_metrics"] = dep
        row["criteria"] = criteria
        row["passed"] = all(criteria.values())
        all_passed = all_passed and bool(row["passed"])
    return fraction_result, all_passed


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)

    train_report = load_json(args.train_report)
    fresh_report = load_json(args.fresh_report)
    train = load_npz(args.train_volumes)
    fresh = load_npz(args.fresh_volumes)
    validate_schema(train)
    validate_schema(fresh)
    validate_report(train_report, train)
    validate_report(fresh_report, fresh)

    train_map = {str(k): int(v) for k, v in train_report["mode_to_index"].items()}
    fresh_map = {str(k): int(v) for k, v in fresh_report["mode_to_index"].items()}
    if train_map != fresh_map:
        raise ValueError("train/fresh mode maps differ")
    for mode in MODES:
        if mode not in train_map:
            raise KeyError(f"missing required mode={mode}")

    train_protocol = train_report["protocol"]
    fresh_protocol = fresh_report["protocol"]
    for key in ("source_view_count", "probe_total_view_count", "volume_side"):
        if train_protocol.get(key) != fresh_protocol.get(key):
            raise ValueError(f"train/fresh protocol mismatch for {key}")

    fractions = parse_fractions(args.fractions)
    train_comparison: dict[str, Any] = {}
    for fraction in fractions:
        result = evaluate_fraction(
            train_report,
            train,
            fraction=fraction,
            random_trials=int(args.random_trials),
            seed=int(args.seed),
        )
        train_comparison[f"{fraction:.6f}"] = result

    selected_train = max(
        train_comparison.values(),
        key=lambda row: (
            float(row["selection_score"]),
            -abs(float(row["fraction"]) - 0.30),
        ),
    )
    selected_fraction = float(selected_train["fraction"])

    fresh_selected = evaluate_fraction(
        fresh_report,
        fresh,
        fraction=selected_fraction,
        random_trials=int(args.random_trials),
        seed=int(args.seed) + 7000001,
    )
    fresh_deployment = {
        mode: deployment_object_metrics(fresh, mode_index=fresh_map[mode])
        for mode in MODES
    }
    fresh_selected, modes_passed = add_fresh_criteria(
        fresh_selected, fresh_deployment, args
    )

    global_checks = {
        "train_selection_score": (
            float(selected_train["selection_score"])
            > float(args.min_train_selection_score)
        ),
        "all_fresh_modes_passed": bool(modes_passed),
    }
    passed = all(global_checks.values())

    calibration = {
        "format": "ar_ss_flow.visual_only_pairwise_percentile_gate.v3",
        "checkpoint": fresh_protocol["checkpoint"],
        "checkpoint_step": fresh_protocol["checkpoint_step"],
        "confidence_source": "raw_visual_only_pairwise_confidence",
        "geometry_pair_scale": 0.0,
        "selection_scope": "per_volume_valid_voxels",
        "top_fraction": selected_fraction,
        "hard_gate_formula": "gate=1[percentile_rank(confidence)>=1-top_fraction]",
        "soft_gate_formula": (
            "gate=clamp((percentile_rank(confidence)-(1-top_fraction))/top_fraction,0,1)"
        ),
        "absolute_confidence_threshold": None,
        "selected_on_indices": train_protocol["indices"],
        "validated_on_indices": fresh_protocol["indices"],
    }

    report = {
        "stage": "C1.5-v3 raw percentile visual-only pairwise gate calibration",
        "passed": passed,
        "args": vars(args),
        "calibration": calibration,
        "train_fraction_comparison": train_comparison,
        "selected_fraction": selected_fraction,
        "selected_train_result": selected_train,
        "fresh_selected_result": fresh_selected,
        "global_checks": global_checks,
        "interpretation": {
            "classifier_auc_required": False,
            "selection_is_per_volume": True,
            "equal_coverage_controls": [
                "random_selector",
                "wrong_confidence_selector",
                "visual_shuffle_confidence_selector",
            ],
            "gate_is_relative_rank_not_calibrated_probability": True,
        },
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    (output_dir / "calibration.json").write_text(
        json.dumps(calibration, indent=2), encoding="utf-8"
    )

    print("=" * 100)
    print(report["stage"])
    for key in sorted(train_comparison, key=float):
        row = train_comparison[key]
        print(
            f"train fraction={row['fraction']:.2f} "
            f"min_mode_score={row['minimum_mode_selection_score']:+.6f} "
            f"mean_mode_score={row['mean_mode_selection_score']:+.6f} "
            f"selection_score={row['selection_score']:+.6f}"
        )
    print(f"selected_fraction={selected_fraction:.2f}")
    for mode in MODES:
        row = fresh_selected["per_mode"][mode]
        wrong = row["wrong_pose_label"]
        shuffle = row["shuffle_label"]
        print(
            f"{mode}: passed={row['passed']} objects={row['object_count']} "
            f"selected_adv={wrong['correct_selector_advantage']['mean']:+.6f} "
            f"uplift={wrong['correct_minus_overall']['mean']:+.6f} "
            f"rel={wrong['relative_uplift_vs_overall']:+.4f} "
            f"vs_random={wrong['correct_minus_random']['mean']:+.6f} "
            f"win_random={wrong['correct_greater_random_object_rate']:.4f} "
            f"vs_wrong_selector={wrong['correct_minus_wrong_selector']['mean']:+.6f} "
            f"win_wrong={wrong['correct_greater_wrong_selector_object_rate']:.4f} "
            f"shuffle_vs_random={shuffle['correct_minus_random']['mean']:+.6f} "
            f"shuffle_vs_shuffle_selector={shuffle['correct_minus_shuffle_selector']['mean']:+.6f}"
        )
        failed = [name for name, ok in row["criteria"].items() if not ok]
        if failed:
            print("  failed_criteria:", failed)
    print("global_checks:", global_checks)
    print("passed:", passed)
    print("report:", output_dir / "report.json")
    print("calibration:", output_dir / "calibration.json")

    if args.fail_on_error and not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
