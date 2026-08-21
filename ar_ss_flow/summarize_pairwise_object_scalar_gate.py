#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    from ar_ss_flow.pairwise_object_scalar_gate import (
        Statistic,
        binary_auc,
        calibrate_thresholds,
        object_score,
        parse_statistics,
        rank_correlation,
        scalar_gate,
        statistic_from_dict,
    )
except ModuleNotFoundError:
    from pairwise_object_scalar_gate import (
        Statistic,
        binary_auc,
        calibrate_thresholds,
        object_score,
        parse_statistics,
        rank_correlation,
        scalar_gate,
        statistic_from_dict,
    )


MODES = ("pose_cyclic1", "pose_cyclic2", "pose_reverse")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "C1.6 object-level scalar gate calibration. Select one robust full-volume "
            "confidence statistic on train objects, calibrate a scalar gate, and freeze "
            "both statistic and thresholds on fresh objects."
        )
    )
    parser.add_argument("--train_report", required=True)
    parser.add_argument("--train_volumes", required=True)
    parser.add_argument("--fresh_report", required=True)
    parser.add_argument("--fresh_volumes", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--statistics",
        default="mean,median,trimmed_mean_10,top20_mean",
    )
    parser.add_argument("--frozen_statistic_from", default="")
    parser.add_argument("--min_valid_voxels", type=int, default=64)
    parser.add_argument("--min_train_correct_coverage", type=float, default=0.50)
    parser.add_argument("--max_train_correct_coverage", type=float, default=0.95)
    parser.add_argument("--tau_high_quantile", type=float, default=0.90)
    parser.add_argument("--min_objects", type=int, default=8)
    parser.add_argument("--min_train_selection_score", type=float, default=0.80)
    parser.add_argument("--min_score_auc", type=float, default=0.85)
    parser.add_argument("--min_score_win_rate", type=float, default=0.85)
    parser.add_argument("--min_gate_win_rate", type=float, default=0.80)
    parser.add_argument("--min_gate_gap", type=float, default=0.10)
    parser.add_argument("--min_correct_gate_mean", type=float, default=0.20)
    parser.add_argument("--max_mode_threshold_spread", type=float, default=0.03)
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


def validate_inputs(report: dict[str, Any], arrays: dict[str, np.ndarray]) -> None:
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
        raise KeyError(f"volume archive missing arrays={missing}")
    protocol = report.get("protocol", {})
    if not bool(protocol.get("visual_only_pairwise", False)):
        raise ValueError("report is not visual-only pairwise")
    if not bool(protocol.get("geometry_pair_scale_forced_zero", False)):
        raise ValueError("geometry pair scale was not forced to zero")
    if not bool(protocol.get("full_volume_records_saved", False)):
        raise ValueError("report does not contain full-volume records")


def mode_index_map(report: dict[str, Any]) -> dict[str, int]:
    mapping = report.get("mode_to_index", {})
    missing = [mode for mode in MODES if mode not in mapping]
    if missing:
        raise KeyError(f"report mode_to_index missing={missing}")
    return {mode: int(mapping[mode]) for mode in MODES}


def summarize(values: list[float]) -> dict[str, float | int]:
    data = np.asarray([value for value in values if np.isfinite(value)], dtype=np.float64)
    if data.size == 0:
        return {
            "count": 0,
            "mean": 0.0,
            "median": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
        }
    return {
        "count": int(data.size),
        "mean": float(np.mean(data)),
        "median": float(np.median(data)),
        "std": float(np.std(data)),
        "min": float(np.min(data)),
        "max": float(np.max(data)),
    }


def positive_rate(values: list[float]) -> float:
    data = np.asarray([value for value in values if np.isfinite(value)], dtype=np.float64)
    return float(np.mean(data > 0.0)) if data.size else 0.0


def average_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    return {
        key: float(np.mean([float(row[key]) for row in rows]))
        for key in rows[0]
    }


def deployment_objects(
    arrays: dict[str, np.ndarray],
    *,
    mode_index: int,
    statistic: Statistic,
    min_valid_voxels: int,
) -> list[dict[str, float | int]]:
    grouped: dict[int, list[dict[str, float]]] = defaultdict(list)
    row_ids = np.flatnonzero(arrays["deployment_mode_index"] == int(mode_index))
    for row_id in row_ids:
        valid = arrays["deployment_common_support"][row_id].astype(bool).reshape(-1)
        row = {
            "correct_score": object_score(
                arrays["deployment_correct_confidence"][row_id],
                valid,
                statistic,
                min_valid_voxels=min_valid_voxels,
            ),
            "wrong_score": object_score(
                arrays["deployment_wrong_confidence"][row_id],
                valid,
                statistic,
                min_valid_voxels=min_valid_voxels,
            ),
            "shuffle_score": object_score(
                arrays["deployment_shuffle_confidence"][row_id],
                valid,
                statistic,
                min_valid_voxels=min_valid_voxels,
            ),
            "valid_voxel_count": float(np.sum(valid)),
        }
        if all(np.isfinite(value) for value in row.values()):
            grouped[int(arrays["deployment_object_index"][row_id])].append(row)
    result: list[dict[str, float | int]] = []
    for object_index, rows in sorted(grouped.items()):
        result.append({"object_index": int(object_index), **average_rows(rows)})
    return result


def deployment_metrics(objects: list[dict[str, float | int]]) -> dict[str, Any]:
    correct = np.asarray([float(row["correct_score"]) for row in objects], dtype=np.float64)
    wrong = np.asarray([float(row["wrong_score"]) for row in objects], dtype=np.float64)
    shuffle = np.asarray([float(row["shuffle_score"]) for row in objects], dtype=np.float64)
    return {
        "object_count": len(objects),
        "auc_correct_vs_wrong": binary_auc(correct, wrong),
        "auc_correct_vs_shuffle": binary_auc(correct, shuffle),
        "correct_greater_wrong_rate": float(np.mean(correct > wrong)) if correct.size else 0.0,
        "correct_greater_shuffle_rate": float(np.mean(correct > shuffle)) if correct.size else 0.0,
        "correct_minus_wrong": summarize((correct - wrong).tolist()),
        "correct_minus_shuffle": summarize((correct - shuffle).tolist()),
        "correct_score": summarize(correct.tolist()),
        "wrong_score": summarize(wrong.tolist()),
        "shuffle_score": summarize(shuffle.tolist()),
        "valid_voxel_count": summarize(
            [float(row["valid_voxel_count"]) for row in objects]
        ),
    }


def statistic_evaluation(
    report: dict[str, Any],
    arrays: dict[str, np.ndarray],
    statistic: Statistic,
    *,
    min_valid_voxels: int,
) -> dict[str, Any]:
    mapping = mode_index_map(report)
    per_mode: dict[str, Any] = {}
    scores: list[float] = []
    for mode in MODES:
        objects = deployment_objects(
            arrays,
            mode_index=mapping[mode],
            statistic=statistic,
            min_valid_voxels=min_valid_voxels,
        )
        metrics = deployment_metrics(objects)
        components = [
            float(metrics["auc_correct_vs_wrong"]),
            float(metrics["auc_correct_vs_shuffle"]),
            float(metrics["correct_greater_wrong_rate"]),
            float(metrics["correct_greater_shuffle_rate"]),
        ]
        mode_score = float(0.75 * min(components) + 0.25 * np.mean(components))
        per_mode[mode] = {
            "metrics": metrics,
            "selection_components": components,
            "selection_score": mode_score,
        }
        scores.append(mode_score)
    minimum = float(min(scores))
    average = float(np.mean(scores))
    return {
        "statistic": statistic.to_dict(),
        "per_mode": per_mode,
        "minimum_mode_selection_score": minimum,
        "mean_mode_selection_score": average,
        "selection_score": float(0.75 * minimum + 0.25 * average),
    }


def pooled_train_scores(
    report: dict[str, Any],
    arrays: dict[str, np.ndarray],
    statistic: Statistic,
    *,
    min_valid_voxels: int,
) -> dict[str, np.ndarray]:
    mapping = mode_index_map(report)
    correct: list[float] = []
    wrong: list[float] = []
    shuffle: list[float] = []
    mode_threshold_rows: dict[str, dict[str, np.ndarray]] = {}
    for mode in MODES:
        objects = deployment_objects(
            arrays,
            mode_index=mapping[mode],
            statistic=statistic,
            min_valid_voxels=min_valid_voxels,
        )
        c = np.asarray([float(row["correct_score"]) for row in objects], dtype=np.float64)
        w = np.asarray([float(row["wrong_score"]) for row in objects], dtype=np.float64)
        s = np.asarray([float(row["shuffle_score"]) for row in objects], dtype=np.float64)
        correct.extend(c.tolist())
        wrong.extend(w.tolist())
        shuffle.extend(s.tolist())
        mode_threshold_rows[mode] = {"correct": c, "negative": np.concatenate([w, s])}
    return {
        "correct": np.asarray(correct, dtype=np.float64),
        "wrong": np.asarray(wrong, dtype=np.float64),
        "shuffle": np.asarray(shuffle, dtype=np.float64),
        "negative": np.asarray(wrong + shuffle, dtype=np.float64),
        "per_mode": mode_threshold_rows,
    }


def probe_metrics(
    report: dict[str, Any],
    arrays: dict[str, np.ndarray],
    statistic: Statistic,
    *,
    mode_index: int,
    min_valid_voxels: int,
    tau_low: float,
    tau_high: float,
) -> dict[str, Any]:
    grouped: dict[int, list[dict[str, float]]] = defaultdict(list)
    row_ids = np.flatnonzero(arrays["probe_mode_index"] == int(mode_index))
    for row_id in row_ids:
        valid = arrays["probe_valid_mask"][row_id].astype(bool).reshape(-1)
        if int(np.sum(valid)) < int(min_valid_voxels):
            continue
        c = object_score(
            arrays["probe_correct_confidence"][row_id], valid, statistic,
            min_valid_voxels=min_valid_voxels,
        )
        w = object_score(
            arrays["probe_wrong_confidence"][row_id], valid, statistic,
            min_valid_voxels=min_valid_voxels,
        )
        s = object_score(
            arrays["probe_shuffle_confidence"][row_id], valid, statistic,
            min_valid_voxels=min_valid_voxels,
        )
        wrong_adv = float(np.mean(
            np.asarray(arrays["probe_reprojection_advantage"][row_id], dtype=np.float64).reshape(-1)[valid]
        ))
        shuffle_adv = float(np.mean(
            np.asarray(arrays["probe_shuffle_reprojection_advantage"][row_id], dtype=np.float64).reshape(-1)[valid]
        ))
        row = {
            "correct_score": c,
            "wrong_score": w,
            "shuffle_score": s,
            "correct_gate": scalar_gate(c, tau_low, tau_high),
            "wrong_gate": scalar_gate(w, tau_low, tau_high),
            "shuffle_gate": scalar_gate(s, tau_low, tau_high),
            "wrong_reprojection_advantage": wrong_adv,
            "shuffle_reprojection_advantage": shuffle_adv,
        }
        if all(np.isfinite(value) for value in row.values()):
            grouped[int(arrays["probe_object_index"][row_id])].append(row)
    objects = [average_rows(rows) for rows in grouped.values() if rows]
    if not objects:
        return {"object_count": 0}
    values = lambda key: np.asarray([row[key] for row in objects], dtype=np.float64)
    score_gap_wrong = values("correct_score") - values("wrong_score")
    score_gap_shuffle = values("correct_score") - values("shuffle_score")
    return {
        "object_count": len(objects),
        "score_win_wrong_rate": positive_rate(score_gap_wrong.tolist()),
        "score_win_shuffle_rate": positive_rate(score_gap_shuffle.tolist()),
        "gate_win_wrong_rate": positive_rate(
            (values("correct_gate") - values("wrong_gate")).tolist()
        ),
        "gate_win_shuffle_rate": positive_rate(
            (values("correct_gate") - values("shuffle_gate")).tolist()
        ),
        "score_gap_wrong_vs_reprojection_rank_correlation": rank_correlation(
            score_gap_wrong, values("wrong_reprojection_advantage")
        ),
        "score_gap_shuffle_vs_reprojection_rank_correlation": rank_correlation(
            score_gap_shuffle, values("shuffle_reprojection_advantage")
        ),
        "wrong_reprojection_advantage": summarize(
            values("wrong_reprojection_advantage").tolist()
        ),
        "shuffle_reprojection_advantage": summarize(
            values("shuffle_reprojection_advantage").tolist()
        ),
    }


def fresh_mode_evaluation(
    report: dict[str, Any],
    arrays: dict[str, np.ndarray],
    statistic: Statistic,
    *,
    mode: str,
    min_valid_voxels: int,
    tau_low: float,
    tau_high: float,
    args: argparse.Namespace,
) -> dict[str, Any]:
    mapping = mode_index_map(report)
    objects = deployment_objects(
        arrays,
        mode_index=mapping[mode],
        statistic=statistic,
        min_valid_voxels=min_valid_voxels,
    )
    metrics = deployment_metrics(objects)
    correct = np.asarray([float(row["correct_score"]) for row in objects], dtype=np.float64)
    wrong = np.asarray([float(row["wrong_score"]) for row in objects], dtype=np.float64)
    shuffle = np.asarray([float(row["shuffle_score"]) for row in objects], dtype=np.float64)
    correct_gate = scalar_gate(correct, tau_low, tau_high)
    wrong_gate = scalar_gate(wrong, tau_low, tau_high)
    shuffle_gate = scalar_gate(shuffle, tau_low, tau_high)
    metrics["gate"] = {
        "correct": summarize(correct_gate.tolist()),
        "wrong": summarize(wrong_gate.tolist()),
        "shuffle": summarize(shuffle_gate.tolist()),
        "correct_minus_wrong": summarize((correct_gate - wrong_gate).tolist()),
        "correct_minus_shuffle": summarize((correct_gate - shuffle_gate).tolist()),
        "correct_greater_wrong_rate": positive_rate((correct_gate - wrong_gate).tolist()),
        "correct_greater_shuffle_rate": positive_rate((correct_gate - shuffle_gate).tolist()),
    }
    probe = probe_metrics(
        report,
        arrays,
        statistic,
        mode_index=mapping[mode],
        min_valid_voxels=min_valid_voxels,
        tau_low=tau_low,
        tau_high=tau_high,
    )
    criteria = {
        "minimum_objects": int(metrics["object_count"]) >= int(args.min_objects),
        "score_auc_correct_vs_wrong": float(metrics["auc_correct_vs_wrong"]) >= float(args.min_score_auc),
        "score_auc_correct_vs_shuffle": float(metrics["auc_correct_vs_shuffle"]) >= float(args.min_score_auc),
        "score_win_correct_vs_wrong": float(metrics["correct_greater_wrong_rate"]) >= float(args.min_score_win_rate),
        "score_win_correct_vs_shuffle": float(metrics["correct_greater_shuffle_rate"]) >= float(args.min_score_win_rate),
        "gate_win_correct_vs_wrong": float(metrics["gate"]["correct_greater_wrong_rate"]) >= float(args.min_gate_win_rate),
        "gate_win_correct_vs_shuffle": float(metrics["gate"]["correct_greater_shuffle_rate"]) >= float(args.min_gate_win_rate),
        "gate_gap_correct_vs_wrong": float(metrics["gate"]["correct_minus_wrong"]["mean"]) >= float(args.min_gate_gap),
        "gate_gap_correct_vs_shuffle": float(metrics["gate"]["correct_minus_shuffle"]["mean"]) >= float(args.min_gate_gap),
        "correct_gate_mean": float(metrics["gate"]["correct"]["mean"]) >= float(args.min_correct_gate_mean),
    }
    return {
        "passed": all(criteria.values()),
        "criteria": criteria,
        "deployment": metrics,
        "heldout_probe_diagnostic": probe,
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    train_report = load_json(args.train_report)
    fresh_report = load_json(args.fresh_report)
    train_arrays = load_npz(args.train_volumes)
    fresh_arrays = load_npz(args.fresh_volumes)
    validate_inputs(train_report, train_arrays)
    validate_inputs(fresh_report, fresh_arrays)
    if mode_index_map(train_report) != mode_index_map(fresh_report):
        raise ValueError("train/fresh mode maps differ")

    candidates = parse_statistics(args.statistics)
    train_results = {
        statistic.name: statistic_evaluation(
            train_report,
            train_arrays,
            statistic,
            min_valid_voxels=int(args.min_valid_voxels),
        )
        for statistic in candidates
    }

    if args.frozen_statistic_from:
        reference = load_json(args.frozen_statistic_from)
        config = reference.get("statistic", reference.get("calibration", {}).get("statistic"))
        if not isinstance(config, dict):
            raise KeyError("frozen statistic file does not contain statistic config")
        selected = statistic_from_dict(config)
        if selected.name not in train_results:
            raise KeyError(f"frozen statistic {selected.name} is not in current candidates")
        selection_mode = "statistic_frozen_from_reference"
    else:
        selected = max(
            enumerate(candidates),
            key=lambda item: (
                float(train_results[item[1].name]["selection_score"]),
                -int(item[0]),
            ),
        )[1]
        selection_mode = "selected_on_train_objects"

    pooled = pooled_train_scores(
        train_report,
        train_arrays,
        selected,
        min_valid_voxels=int(args.min_valid_voxels),
    )
    threshold = calibrate_thresholds(
        pooled["correct"],
        pooled["negative"],
        min_correct_coverage=float(args.min_train_correct_coverage),
        max_correct_coverage=float(args.max_train_correct_coverage),
        tau_high_quantile=float(args.tau_high_quantile),
    )
    tau_low = float(threshold["tau_low"])
    tau_high = float(threshold["tau_high"])

    per_mode_train_threshold: dict[str, Any] = {}
    for mode in MODES:
        row = pooled["per_mode"][mode]
        per_mode_train_threshold[mode] = calibrate_thresholds(
            row["correct"],
            row["negative"],
            min_correct_coverage=float(args.min_train_correct_coverage),
            max_correct_coverage=float(args.max_train_correct_coverage),
            tau_high_quantile=float(args.tau_high_quantile),
        )
    mode_tau_low = [float(row["tau_low"]) for row in per_mode_train_threshold.values()]
    threshold_spread = float(max(mode_tau_low) - min(mode_tau_low))

    fresh_per_mode = {
        mode: fresh_mode_evaluation(
            fresh_report,
            fresh_arrays,
            selected,
            mode=mode,
            min_valid_voxels=int(args.min_valid_voxels),
            tau_low=tau_low,
            tau_high=tau_high,
            args=args,
        )
        for mode in MODES
    }
    selected_train_score = float(train_results[selected.name]["selection_score"])
    global_checks = {
        "train_selection_score": selected_train_score >= float(args.min_train_selection_score),
        "mode_tau_low_spread": threshold_spread <= float(args.max_mode_threshold_spread),
        "all_fresh_modes_passed": all(row["passed"] for row in fresh_per_mode.values()),
    }
    passed = all(global_checks.values())

    calibration = {
        "format": "ar_ss_flow.visual_only_pairwise_object_scalar_gate.v1",
        "checkpoint": fresh_report["protocol"].get("checkpoint"),
        "checkpoint_step": fresh_report["protocol"].get("checkpoint_step"),
        "confidence_source": "raw_visual_only_pairwise_confidence",
        "geometry_pair_scale": 0.0,
        "statistic": selected.to_dict(),
        "minimum_valid_voxels": int(args.min_valid_voxels),
        "tau_low": tau_low,
        "tau_high": tau_high,
        "gate_formula": "g_object=clip((object_score-tau_low)/(tau_high-tau_low),0,1)",
        "residual_formula": "cond_final=cond_base+g_object*delta_cond",
        "stock_condition_untouched": True,
        "selected_on_indices": train_report["protocol"].get("indices"),
        "validated_on_indices": fresh_report["protocol"].get("indices"),
    }
    report = {
        "stage": "C1.6 visual-only pairwise object-level scalar gate calibration",
        "passed": passed,
        "args": vars(args),
        "selection_mode": selection_mode,
        "selected_statistic": selected.to_dict(),
        "selected_train_score": selected_train_score,
        "train_candidate_comparison": train_results,
        "pooled_train_threshold": threshold,
        "per_mode_train_threshold": per_mode_train_threshold,
        "mode_tau_low_spread": threshold_spread,
        "fresh_per_mode": fresh_per_mode,
        "global_checks": global_checks,
        "calibration": calibration,
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "calibration.json").write_text(
        json.dumps(calibration, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("=" * 100)
    print(report["stage"])
    for statistic in candidates:
        row = train_results[statistic.name]
        print(
            f"train statistic={statistic.name} "
            f"min_mode={row['minimum_mode_selection_score']:.4f} "
            f"mean_mode={row['mean_mode_selection_score']:.4f} "
            f"selection_score={row['selection_score']:.4f}"
        )
    print("selection_mode:", selection_mode)
    print("selected_statistic:", selected.to_dict())
    print(f"tau_low={tau_low:.6f} tau_high={tau_high:.6f} spread={threshold_spread:.6f}")
    for mode in MODES:
        row = fresh_per_mode[mode]
        dep = row["deployment"]
        gate = dep["gate"]
        print(
            f"{mode}: passed={row['passed']} objects={dep['object_count']} "
            f"auc_wrong={dep['auc_correct_vs_wrong']:.4f} "
            f"auc_shuffle={dep['auc_correct_vs_shuffle']:.4f} "
            f"score_win_wrong={dep['correct_greater_wrong_rate']:.4f} "
            f"score_win_shuffle={dep['correct_greater_shuffle_rate']:.4f} "
            f"gate_gap_wrong={gate['correct_minus_wrong']['mean']:+.4f} "
            f"gate_gap_shuffle={gate['correct_minus_shuffle']['mean']:+.4f} "
            f"correct_gate={gate['correct']['mean']:.4f}"
        )
        failed = [name for name, value in row["criteria"].items() if not value]
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
