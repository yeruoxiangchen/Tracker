#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np

from ar_ss_flow.pairwise_multiscale_region_gate import (
    Candidate,
    binary_auc,
    build_candidates,
    parse_int_csv,
    percentile_region_gate,
    rank_correlation,
    reduce_regions,
    region_label_means,
    weighted_region_mean,
)

MODES = ("pose_cyclic1", "pose_cyclic2", "pose_reverse")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "C1.6 multi-scale region audit. Select the finest reliable spatial "
            "confidence scale on train objects and freeze it on fresh objects."
        )
    )
    parser.add_argument("--train_report", required=True)
    parser.add_argument("--train_volumes", required=True)
    parser.add_argument("--fresh_report", required=True)
    parser.add_argument("--fresh_volumes", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--divisions", default="1,2,4")
    parser.add_argument("--shrinkage_kappas", default="32,64")
    parser.add_argument("--trim_fraction", type=float, default=0.10)
    parser.add_argument("--min_region_voxels", type=int, default=8)
    parser.add_argument("--frozen_candidate_from", default="")
    parser.add_argument("--min_objects", type=int, default=8)
    parser.add_argument("--min_train_selection_score", type=float, default=0.001)
    parser.add_argument("--min_deployment_object_win_rate", type=float, default=0.80)
    parser.add_argument("--min_deployment_region_win_fraction", type=float, default=0.55)
    parser.add_argument("--min_gain_vs_object", type=float, default=0.001)
    parser.add_argument("--min_object_win_vs_object", type=float, default=0.60)
    parser.add_argument("--min_gain_vs_wrong_gate", type=float, default=0.0)
    parser.add_argument("--min_object_win_vs_wrong_gate", type=float, default=0.60)
    parser.add_argument("--min_gain_vs_shuffle_gate", type=float, default=0.0)
    parser.add_argument("--min_object_win_vs_shuffle_gate", type=float, default=0.60)
    parser.add_argument("--min_spatial_rank_correlation", type=float, default=0.05)
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


def summarize(values: list[float]) -> dict[str, float | int]:
    finite = np.asarray([x for x in values if np.isfinite(x)], dtype=np.float64)
    if finite.size == 0:
        return {"count": 0, "mean": 0.0, "median": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "count": int(finite.size),
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "std": float(np.std(finite)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
    }


def positive_rate(values: list[float]) -> float:
    finite = [x for x in values if np.isfinite(x)]
    return float(np.mean([x > 0.0 for x in finite])) if finite else 0.0


def average_dict_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    return {
        key: float(np.mean([float(row[key]) for row in rows]))
        for key in rows[0]
    }


def mode_index_map(report: dict[str, Any]) -> dict[str, int]:
    mapping = report.get("mode_to_index", {})
    missing = [mode for mode in MODES if mode not in mapping]
    if missing:
        raise KeyError(f"report mode_to_index missing={missing}")
    return {mode: int(mapping[mode]) for mode in MODES}


def candidate_region_values(
    confidence: np.ndarray,
    valid: np.ndarray,
    *,
    side: int,
    candidate: Candidate,
    trim_fraction: float,
    min_region_voxels: int,
):
    region = reduce_regions(
        confidence,
        valid,
        volume_side=side,
        divisions=candidate.divisions,
        trim_fraction=trim_fraction,
        min_region_voxels=min_region_voxels,
    )
    return region


def candidate_gate(region, candidate: Candidate, common: np.ndarray) -> np.ndarray:
    gate = percentile_region_gate(region.scores, common)
    if candidate.shrinkage_kappa is None or not np.any(common):
        return gate
    counts = region.counts.astype(np.float64)
    weight = counts[common]
    denominator = float(np.sum(weight))
    parent = float(np.sum(gate[common] * weight) / denominator) if denominator > 0.0 else 1.0
    alpha = counts / (counts + float(candidate.shrinkage_kappa))
    out = gate.copy()
    out[common] = alpha[common] * gate[common] + (1.0 - alpha[common]) * parent
    return out


def evaluate_deployment_mode(
    arrays: dict[str, np.ndarray],
    *,
    mode_index: int,
    side: int,
    candidate: Candidate,
    trim_fraction: float,
    min_region_voxels: int,
) -> dict[str, Any]:
    grouped: dict[int, list[dict[str, float]]] = defaultdict(list)
    row_ids = np.flatnonzero(arrays["deployment_mode_index"] == int(mode_index))
    for row in row_ids:
        valid = arrays["deployment_common_support"][row].astype(bool).reshape(-1)
        c = candidate_region_values(
            arrays["deployment_correct_confidence"][row], valid,
            side=side, candidate=candidate, trim_fraction=trim_fraction,
            min_region_voxels=min_region_voxels,
        )
        w = candidate_region_values(
            arrays["deployment_wrong_confidence"][row], valid,
            side=side, candidate=candidate, trim_fraction=trim_fraction,
            min_region_voxels=min_region_voxels,
        )
        s = candidate_region_values(
            arrays["deployment_shuffle_confidence"][row], valid,
            side=side, candidate=candidate, trim_fraction=trim_fraction,
            min_region_voxels=min_region_voxels,
        )
        common = c.valid & w.valid & s.valid
        if not np.any(common):
            continue
        counts = np.minimum(np.minimum(c.counts, w.counts), s.counts).astype(np.float64)
        denom = float(np.sum(counts[common]))
        if denom <= 0.0:
            continue
        c_obj = float(np.sum(c.scores[common] * counts[common]) / denom)
        w_obj = float(np.sum(w.scores[common] * counts[common]) / denom)
        s_obj = float(np.sum(s.scores[common] * counts[common]) / denom)
        grouped[int(arrays["deployment_object_index"][row])].append(
            {
                "correct_minus_wrong_object": c_obj - w_obj,
                "correct_minus_shuffle_object": c_obj - s_obj,
                "region_win_wrong_fraction": float(np.mean(c.scores[common] > w.scores[common])),
                "region_win_shuffle_fraction": float(np.mean(c.scores[common] > s.scores[common])),
                "valid_region_count": float(np.sum(common)),
            }
        )
    objects = [average_dict_rows(rows) for rows in grouped.values() if rows]
    cw = [row["correct_minus_wrong_object"] for row in objects]
    cs = [row["correct_minus_shuffle_object"] for row in objects]
    rw = [row["region_win_wrong_fraction"] for row in objects]
    rs = [row["region_win_shuffle_fraction"] for row in objects]
    return {
        "object_count": len(objects),
        "correct_minus_wrong_object": summarize(cw),
        "correct_greater_wrong_object_rate": positive_rate(cw),
        "correct_minus_shuffle_object": summarize(cs),
        "correct_greater_shuffle_object_rate": positive_rate(cs),
        "region_correct_greater_wrong_fraction": summarize(rw),
        "region_correct_greater_shuffle_fraction": summarize(rs),
        "valid_region_count": summarize([row["valid_region_count"] for row in objects]),
    }


def evaluate_probe_volume(
    arrays: dict[str, np.ndarray],
    *,
    row: int,
    side: int,
    candidate: Candidate,
    trim_fraction: float,
    min_region_voxels: int,
) -> dict[str, float] | None:
    valid_voxels = arrays["probe_valid_mask"][row].astype(bool).reshape(-1)
    c = candidate_region_values(
        arrays["probe_correct_confidence"][row], valid_voxels,
        side=side, candidate=candidate, trim_fraction=trim_fraction,
        min_region_voxels=min_region_voxels,
    )
    w = candidate_region_values(
        arrays["probe_wrong_confidence"][row], valid_voxels,
        side=side, candidate=candidate, trim_fraction=trim_fraction,
        min_region_voxels=min_region_voxels,
    )
    s = candidate_region_values(
        arrays["probe_shuffle_confidence"][row], valid_voxels,
        side=side, candidate=candidate, trim_fraction=trim_fraction,
        min_region_voxels=min_region_voxels,
    )
    label_w, label_counts_w, valid_label_w = region_label_means(
        arrays["probe_reprojection_advantage"][row], valid_voxels,
        volume_side=side, divisions=candidate.divisions,
        min_region_voxels=min_region_voxels,
    )
    label_s, label_counts_s, valid_label_s = region_label_means(
        arrays["probe_shuffle_reprojection_advantage"][row], valid_voxels,
        volume_side=side, divisions=candidate.divisions,
        min_region_voxels=min_region_voxels,
    )
    common_w = c.valid & w.valid & s.valid & valid_label_w
    common_s = c.valid & w.valid & s.valid & valid_label_s
    if not np.any(common_w) or not np.any(common_s):
        return None
    gate_c_w = candidate_gate(c, candidate, common_w)
    gate_w_w = candidate_gate(w, candidate, common_w)
    gate_s_w = candidate_gate(s, candidate, common_w)
    gate_c_s = candidate_gate(c, candidate, common_s)
    gate_w_s = candidate_gate(w, candidate, common_s)
    gate_s_s = candidate_gate(s, candidate, common_s)
    ones_w = common_w.astype(np.float64)
    ones_s = common_s.astype(np.float64)

    overall_w = weighted_region_mean(label_w, ones_w, label_counts_w, common_w)
    correct_w = weighted_region_mean(label_w, gate_c_w, label_counts_w, common_w)
    wrong_w = weighted_region_mean(label_w, gate_w_w, label_counts_w, common_w)
    shuffle_w = weighted_region_mean(label_w, gate_s_w, label_counts_w, common_w)
    overall_s = weighted_region_mean(label_s, ones_s, label_counts_s, common_s)
    correct_s = weighted_region_mean(label_s, gate_c_s, label_counts_s, common_s)
    wrong_s = weighted_region_mean(label_s, gate_w_s, label_counts_s, common_s)
    shuffle_s = weighted_region_mean(label_s, gate_s_s, label_counts_s, common_s)

    common_pose = c.valid & w.valid & s.valid
    return {
        "wrong_overall": overall_w,
        "wrong_correct_gate": correct_w,
        "wrong_wrong_gate": wrong_w,
        "wrong_shuffle_gate": shuffle_w,
        "wrong_gain_vs_object": correct_w - overall_w,
        "wrong_gain_vs_wrong_gate": correct_w - wrong_w,
        "wrong_gain_vs_shuffle_gate": correct_w - shuffle_w,
        "wrong_spatial_rank_correlation": rank_correlation(c.scores, label_w, common_w),
        "shuffle_overall": overall_s,
        "shuffle_correct_gate": correct_s,
        "shuffle_wrong_gate": wrong_s,
        "shuffle_shuffle_gate": shuffle_s,
        "shuffle_gain_vs_object": correct_s - overall_s,
        "shuffle_gain_vs_wrong_gate": correct_s - wrong_s,
        "shuffle_gain_vs_shuffle_gate": correct_s - shuffle_s,
        "shuffle_spatial_rank_correlation": rank_correlation(c.scores, label_s, common_s),
        "region_correct_minus_wrong_confidence": float(np.mean(c.scores[common_pose] - w.scores[common_pose])),
        "region_correct_minus_shuffle_confidence": float(np.mean(c.scores[common_pose] - s.scores[common_pose])),
        "region_correct_greater_wrong_fraction": float(np.mean(c.scores[common_pose] > w.scores[common_pose])),
        "region_correct_greater_shuffle_fraction": float(np.mean(c.scores[common_pose] > s.scores[common_pose])),
        "valid_region_count": float(np.sum(common_pose)),
    }


def evaluate_probe_mode(
    arrays: dict[str, np.ndarray],
    *,
    mode_index: int,
    side: int,
    candidate: Candidate,
    trim_fraction: float,
    min_region_voxels: int,
) -> dict[str, Any]:
    grouped: dict[int, list[dict[str, float]]] = defaultdict(list)
    row_ids = np.flatnonzero(arrays["probe_mode_index"] == int(mode_index))
    for row in row_ids:
        result = evaluate_probe_volume(
            arrays, row=int(row), side=side, candidate=candidate,
            trim_fraction=trim_fraction, min_region_voxels=min_region_voxels,
        )
        if result is not None:
            grouped[int(arrays["probe_object_index"][row])].append(result)
    objects = [average_dict_rows(rows) for rows in grouped.values() if rows]

    def vals(key: str) -> list[float]:
        return [row[key] for row in objects]

    keys = [
        "wrong_overall", "wrong_correct_gate", "wrong_wrong_gate", "wrong_shuffle_gate",
        "wrong_gain_vs_object", "wrong_gain_vs_wrong_gate", "wrong_gain_vs_shuffle_gate",
        "wrong_spatial_rank_correlation", "shuffle_overall", "shuffle_correct_gate",
        "shuffle_wrong_gate", "shuffle_shuffle_gate", "shuffle_gain_vs_object",
        "shuffle_gain_vs_wrong_gate", "shuffle_gain_vs_shuffle_gate",
        "shuffle_spatial_rank_correlation", "region_correct_minus_wrong_confidence",
        "region_correct_minus_shuffle_confidence", "region_correct_greater_wrong_fraction",
        "region_correct_greater_shuffle_fraction", "valid_region_count",
    ]
    result: dict[str, Any] = {"object_count": len(objects)}
    for key in keys:
        result[key] = summarize(vals(key))
    for key in (
        "wrong_gain_vs_object", "wrong_gain_vs_wrong_gate", "wrong_gain_vs_shuffle_gate",
        "shuffle_gain_vs_object", "shuffle_gain_vs_wrong_gate", "shuffle_gain_vs_shuffle_gate",
    ):
        result[f"{key}_object_win_rate"] = positive_rate(vals(key))
    return result


def evaluate_candidate(
    report: dict[str, Any],
    arrays: dict[str, np.ndarray],
    *,
    candidate: Candidate,
    trim_fraction: float,
    min_region_voxels: int,
) -> dict[str, Any]:
    side = int(report["protocol"]["volume_side"])
    mapping = mode_index_map(report)
    per_mode: dict[str, Any] = {}
    mode_scores: list[float] = []
    for mode in MODES:
        deployment = evaluate_deployment_mode(
            arrays, mode_index=mapping[mode], side=side, candidate=candidate,
            trim_fraction=trim_fraction, min_region_voxels=min_region_voxels,
        )
        probe = evaluate_probe_mode(
            arrays, mode_index=mapping[mode], side=side, candidate=candidate,
            trim_fraction=trim_fraction, min_region_voxels=min_region_voxels,
        )
        components = [
            float(probe["wrong_gain_vs_object"]["mean"]),
            float(probe["wrong_gain_vs_wrong_gate"]["mean"]),
            float(probe["shuffle_gain_vs_object"]["mean"]),
            float(probe["shuffle_gain_vs_shuffle_gate"]["mean"]),
        ]
        mode_score = float(min(components) + 0.25 * np.mean(components))
        if candidate.divisions == 1:
            mode_score = 0.0
        mode_scores.append(mode_score)
        per_mode[mode] = {
            "deployment": deployment,
            "probe": probe,
            "selection_components": components,
            "selection_score": mode_score,
        }
    minimum = float(min(mode_scores))
    average = float(np.mean(mode_scores))
    return {
        "candidate": {
            "name": candidate.name,
            "divisions": candidate.divisions,
            "region_count": candidate.divisions ** 3,
            "shrinkage_kappa": candidate.shrinkage_kappa,
        },
        "per_mode": per_mode,
        "minimum_mode_selection_score": minimum,
        "mean_mode_selection_score": average,
        "selection_score": float(minimum + 0.25 * average),
    }


def candidate_from_calibration(path: str | Path) -> Candidate:
    data = load_json(path)
    config = data.get("candidate", data.get("calibration", {}).get("candidate"))
    if not isinstance(config, dict):
        raise KeyError("frozen candidate file does not contain candidate config")
    kappa = config.get("shrinkage_kappa")
    return Candidate(
        name=str(config["name"]),
        divisions=int(config["divisions"]),
        shrinkage_kappa=None if kappa is None else float(kappa),
    )


def fresh_mode_judgment(row: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    dep = row["deployment"]
    probe = row["probe"]
    criteria = {
        "enough_objects": int(probe["object_count"]) >= int(args.min_objects),
        "deployment_object_win_wrong": float(dep["correct_greater_wrong_object_rate"]) >= float(args.min_deployment_object_win_rate),
        "deployment_object_win_shuffle": float(dep["correct_greater_shuffle_object_rate"]) >= float(args.min_deployment_object_win_rate),
        "deployment_region_win_wrong": float(dep["region_correct_greater_wrong_fraction"]["mean"]) >= float(args.min_deployment_region_win_fraction),
        "deployment_region_win_shuffle": float(dep["region_correct_greater_shuffle_fraction"]["mean"]) >= float(args.min_deployment_region_win_fraction),
        "wrong_gain_vs_object": float(probe["wrong_gain_vs_object"]["mean"]) >= float(args.min_gain_vs_object),
        "wrong_object_win_vs_object": float(probe["wrong_gain_vs_object_object_win_rate"]) >= float(args.min_object_win_vs_object),
        "wrong_gain_vs_wrong_gate": float(probe["wrong_gain_vs_wrong_gate"]["mean"]) > float(args.min_gain_vs_wrong_gate),
        "wrong_object_win_vs_wrong_gate": float(probe["wrong_gain_vs_wrong_gate_object_win_rate"]) >= float(args.min_object_win_vs_wrong_gate),
        "shuffle_gain_vs_object": float(probe["shuffle_gain_vs_object"]["mean"]) >= float(args.min_gain_vs_object),
        "shuffle_object_win_vs_object": float(probe["shuffle_gain_vs_object_object_win_rate"]) >= float(args.min_object_win_vs_object),
        "shuffle_gain_vs_shuffle_gate": float(probe["shuffle_gain_vs_shuffle_gate"]["mean"]) > float(args.min_gain_vs_shuffle_gate),
        "shuffle_object_win_vs_shuffle_gate": float(probe["shuffle_gain_vs_shuffle_gate_object_win_rate"]) >= float(args.min_object_win_vs_shuffle_gate),
        "wrong_spatial_rank_correlation": float(probe["wrong_spatial_rank_correlation"]["mean"]) >= float(args.min_spatial_rank_correlation),
        "shuffle_spatial_rank_correlation": float(probe["shuffle_spatial_rank_correlation"]["mean"]) >= float(args.min_spatial_rank_correlation),
    }
    return {"passed": all(criteria.values()), "criteria": criteria}


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    divisions = parse_int_csv(args.divisions)
    kappas = parse_int_csv(args.shrinkage_kappas)
    candidates = build_candidates(divisions, kappas)

    train_report = load_json(args.train_report)
    fresh_report = load_json(args.fresh_report)
    train_arrays = load_npz(args.train_volumes)
    fresh_arrays = load_npz(args.fresh_volumes)
    validate_inputs(train_report, train_arrays)
    validate_inputs(fresh_report, fresh_arrays)
    if int(train_report["protocol"]["volume_side"]) != int(fresh_report["protocol"]["volume_side"]):
        raise ValueError("train/fresh volume_side mismatch")

    train_results = {
        candidate.name: evaluate_candidate(
            train_report, train_arrays, candidate=candidate,
            trim_fraction=float(args.trim_fraction),
            min_region_voxels=int(args.min_region_voxels),
        )
        for candidate in candidates
    }

    if args.frozen_candidate_from:
        selected = candidate_from_calibration(args.frozen_candidate_from)
        selection_mode = "frozen_from_reference"
        if selected.name not in train_results:
            raise KeyError(
                f"frozen candidate {selected.name} is not among current candidates={list(train_results)}"
            )
    else:
        selected = max(
            enumerate(candidates),
            key=lambda item: (
                float(train_results[item[1].name]["selection_score"]),
                -int(item[1].divisions),
                -int(item[0]),
            ),
        )[1]
        selection_mode = "selected_on_train_objects"

    fresh_selected = evaluate_candidate(
        fresh_report, fresh_arrays, candidate=selected,
        trim_fraction=float(args.trim_fraction),
        min_region_voxels=int(args.min_region_voxels),
    )
    per_mode_judgment: dict[str, Any] = {}
    for mode in MODES:
        judgment = fresh_mode_judgment(fresh_selected["per_mode"][mode], args)
        fresh_selected["per_mode"][mode].update(judgment)
        per_mode_judgment[mode] = judgment

    selected_train_score = float(train_results[selected.name]["selection_score"])
    global_checks = {
        "selected_candidate_is_region": int(selected.divisions) > 1,
        "train_selection_score": selected_train_score >= float(args.min_train_selection_score),
        "all_fresh_modes_passed": all(x["passed"] for x in per_mode_judgment.values()),
    }
    passed = all(global_checks.values())
    candidate_config = {
        "name": selected.name,
        "divisions": int(selected.divisions),
        "region_count": int(selected.divisions ** 3),
        "shrinkage_kappa": selected.shrinkage_kappa,
    }
    calibration = {
        "format": "ar_ss_flow.visual_only_pairwise_multiscale_region_gate.v4",
        "checkpoint": fresh_report["protocol"].get("checkpoint"),
        "checkpoint_step": fresh_report["protocol"].get("checkpoint_step"),
        "confidence_source": "raw_visual_only_pairwise_confidence",
        "geometry_pair_scale": 0.0,
        "candidate": candidate_config,
        "volume_side": int(fresh_report["protocol"]["volume_side"]),
        "region_statistic": "trimmed_mean",
        "trim_fraction": float(args.trim_fraction),
        "minimum_valid_voxels_per_region": int(args.min_region_voxels),
        "spatial_gate_formula": "rank_gate=percentile_rank(region_trimmed_mean_confidence); shrink_gate=alpha*rank_gate+(1-alpha)*matched_object_broadcast, alpha=N_eff/(N_eff+kappa)",
        "future_c2_composition": "g_region_final=clip(g_object*relative_region_gate/weighted_mean(relative_region_gate),0,1)",
        "matched_object_baseline": "broadcast constant gate; equivalent to unweighted valid-region mean",
        "residual_injection_formula": "cond_final=cond_base+g_region*delta_cond",
        "selected_on_indices": "0-15" if not args.frozen_candidate_from else "frozen reference candidate",
        "validated_on_indices": "16-63",
    }
    report = {
        "stage": "C1.6 multi-scale visual-only pairwise region gate calibration",
        "passed": passed,
        "args": vars(args),
        "selection_mode": selection_mode,
        "selected_candidate": candidate_config,
        "selected_train_score": selected_train_score,
        "train_candidate_comparison": train_results,
        "fresh_selected_result": fresh_selected,
        "global_checks": global_checks,
        "calibration": calibration,
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output_dir / "calibration.json").write_text(json.dumps(calibration, indent=2), encoding="utf-8")

    print("=" * 100)
    print(report["stage"])
    for candidate in candidates:
        row = train_results[candidate.name]
        print(
            f"train candidate={candidate.name} regions={candidate.divisions**3} "
            f"min_mode={row['minimum_mode_selection_score']:+.6f} "
            f"mean_mode={row['mean_mode_selection_score']:+.6f} "
            f"selection_score={row['selection_score']:+.6f}"
        )
    print("selection_mode:", selection_mode)
    print("selected_candidate:", selected.name)
    for mode in MODES:
        row = fresh_selected["per_mode"][mode]
        dep = row["deployment"]
        probe = row["probe"]
        print(
            f"{mode}: passed={row['passed']} objects={probe['object_count']} "
            f"dep_obj_win={dep['correct_greater_wrong_object_rate']:.4f} "
            f"dep_region_win={dep['region_correct_greater_wrong_fraction']['mean']:.4f} "
            f"wrong_vs_object={probe['wrong_gain_vs_object']['mean']:+.6f} "
            f"win={probe['wrong_gain_vs_object_object_win_rate']:.4f} "
            f"wrong_vs_wrong_gate={probe['wrong_gain_vs_wrong_gate']['mean']:+.6f} "
            f"win={probe['wrong_gain_vs_wrong_gate_object_win_rate']:.4f} "
            f"shuffle_vs_object={probe['shuffle_gain_vs_object']['mean']:+.6f} "
            f"shuffle_vs_shuffle_gate={probe['shuffle_gain_vs_shuffle_gate']['mean']:+.6f} "
            f"corr={probe['wrong_spatial_rank_correlation']['mean']:+.4f}"
        )
        failed = [key for key, value in row["criteria"].items() if not value]
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
