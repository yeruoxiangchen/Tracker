#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    from ar_ss_flow.pairwise_object_selfref_gate import (
        SelfReferenceConfig,
        binary_auc,
        build_configs,
        calibrate_thresholds,
        config_from_dict,
        contrastive_object_score,
        parse_reference_reducers,
        parse_statistics,
        scalar_gate,
        sigmoid_gate,
    )
except ModuleNotFoundError:
    from pairwise_object_selfref_gate import (
        SelfReferenceConfig,
        binary_auc,
        build_configs,
        calibrate_thresholds,
        config_from_dict,
        contrastive_object_score,
        parse_reference_reducers,
        parse_statistics,
        scalar_gate,
        sigmoid_gate,
    )


POSE_MODES = ("pose_cyclic1", "pose_cyclic2", "pose_reverse")
EXPECTED_HYPOTHESES = ("correct", *POSE_MODES, "visual_shuffle")
EXPECTED_VARIANTS = ("correct", *POSE_MODES)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select and calibrate a self-referenced object scalar: observed confidence "
            "minus a robust reducer of three pose-perturbed confidence scores."
        )
    )
    parser.add_argument("--train_report", required=True)
    parser.add_argument("--train_samples", required=True)
    parser.add_argument("--fresh_report", required=True)
    parser.add_argument("--fresh_samples", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--statistics", default="mean,median,trimmed_mean_10,top20_mean")
    parser.add_argument("--reference_reducers", default="median,mean,max")
    parser.add_argument("--frozen_config_from", default="")
    parser.add_argument("--min_valid_voxels", type=int, default=64)
    parser.add_argument("--min_train_correct_coverage", type=float, default=0.80)
    parser.add_argument("--max_train_correct_coverage", type=float, default=0.98)
    parser.add_argument("--target_train_correct_coverage", type=float, default=0.90)
    parser.add_argument("--tau_high_quantile", type=float, default=0.90)
    parser.add_argument("--min_objects", type=int, default=8)
    parser.add_argument("--min_train_selection_score", type=float, default=0.80)
    parser.add_argument("--min_score_auc", type=float, default=0.85)
    parser.add_argument("--min_score_win_rate", type=float, default=0.85)
    parser.add_argument("--min_gate_win_rate", type=float, default=0.80)
    parser.add_argument("--min_gate_gap", type=float, default=0.10)
    parser.add_argument("--min_correct_gate_mean", type=float, default=0.20)
    parser.add_argument(
    "--target_train_correct_gate",
    type=float,
    default=0.75,
)
    parser.add_argument("--max_mode_tau_low_spread", type=float, default=0.03)
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
        "selfref_object_index",
        "selfref_confidence",
        "selfref_common_support",
        "selfref_valid_voxel_count",
    }
    missing = sorted(required.difference(arrays))
    if missing:
        raise KeyError(f"self-reference archive missing arrays={missing}")
    protocol = report.get("protocol", {})
    if tuple(protocol.get("hypotheses", ())) != EXPECTED_HYPOTHESES:
        raise ValueError(f"unexpected hypotheses={protocol.get('hypotheses')}")
    if tuple(protocol.get("variants", ())) != EXPECTED_VARIANTS:
        raise ValueError(f"unexpected variants={protocol.get('variants')}")
    if not bool(protocol.get("visual_only_pairwise", False)):
        raise ValueError("report is not visual-only pairwise")
    if not bool(protocol.get("geometry_pair_scale_forced_zero", False)):
        raise ValueError("geometry pair scale was not forced to zero")
    if not bool(protocol.get("complete_reperturbation_for_each_hypothesis", False)):
        raise ValueError("report does not contain complete self-reference perturbations")
    confidence = arrays["selfref_confidence"]
    support = arrays["selfref_common_support"]
    if confidence.ndim != 4:
        raise ValueError(f"expected confidence [R,H,V,N], got={confidence.shape}")
    if support.ndim != 3:
        raise ValueError(f"expected support [R,H,N], got={support.shape}")
    if confidence.shape[:2] != support.shape[:2] or confidence.shape[3] != support.shape[2]:
        raise ValueError(f"confidence/support shape mismatch: {confidence.shape} vs {support.shape}")
    if confidence.shape[1] != len(EXPECTED_HYPOTHESES):
        raise ValueError("hypothesis count mismatch")
    if confidence.shape[2] != len(EXPECTED_VARIANTS):
        raise ValueError("variant count mismatch")


def summarize(values: np.ndarray | list[float]) -> dict[str, float | int]:
    data = np.asarray(values, dtype=np.float64).reshape(-1)
    data = data[np.isfinite(data)]
    if data.size == 0:
        return {"count": 0, "mean": 0.0, "median": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "count": int(data.size),
        "mean": float(np.mean(data)),
        "median": float(np.median(data)),
        "std": float(np.std(data)),
        "min": float(np.min(data)),
        "max": float(np.max(data)),
    }


def positive_rate(values: np.ndarray | list[float]) -> float:
    data = np.asarray(values, dtype=np.float64).reshape(-1)
    data = data[np.isfinite(data)]
    return float(np.mean(data > 0.0)) if data.size else 0.0

def calibrate_sigmoid_temperature(
    score_map: dict[str, np.ndarray],
    *,
    target_correct_gate: float,
) -> dict[str, Any]:
    """
    Select one train-only temperature.

    The median positive correct score is mapped approximately to the
    requested correct gate value. A robust MAD floor prevents an
    excessively small temperature and numerical saturation.
    """
    target = float(target_correct_gate)
    if not 0.5 < target < 1.0:
        raise ValueError(
            "target_correct_gate must be between 0.5 and 1.0"
        )

    correct = np.asarray(
        score_map["correct"],
        dtype=np.float64,
    ).reshape(-1)
    correct = correct[np.isfinite(correct)]
    if correct.size == 0:
        raise ValueError("correct train scores are empty")

    all_scores = np.concatenate(
        [
            np.asarray(score_map[name], dtype=np.float64).reshape(-1)
            for name in EXPECTED_HYPOTHESES
        ]
    )
    all_scores = all_scores[np.isfinite(all_scores)]
    if all_scores.size == 0:
        raise ValueError("train scores are empty")

    positive_correct = correct[correct > 0.0]

    if positive_correct.size:
        anchor = float(np.median(positive_correct))
    else:
        anchor = float(np.median(np.abs(correct)))

    center = float(np.median(all_scores))
    mad = float(np.median(np.abs(all_scores - center)))

    logit_target = float(np.log(target / (1.0 - target)))

    temperature_from_anchor = (
        anchor / logit_target
        if anchor > 0.0
        else 0.0
    )

    # Prevent a tiny T from turning sigmoid into another hard threshold.
    robust_floor = 0.5 * mad

    temperature = max(
        temperature_from_anchor,
        robust_floor,
        1.0e-6,
    )

    gate_map = {
        name: np.asarray(
            sigmoid_gate(score_map[name], temperature),
            dtype=np.float64,
        )
        for name in EXPECTED_HYPOTHESES
    }

    per_mode: dict[str, Any] = {}
    for mode in POSE_MODES:
        per_mode[mode] = {
            "correct_minus_wrong_gate": summarize(
                gate_map["correct"] - gate_map[mode]
            ),
            "correct_minus_shuffle_gate": summarize(
                gate_map["correct"] -
                gate_map["visual_shuffle"]
            ),
            "correct_greater_wrong_gate_rate": positive_rate(
                gate_map["correct"] - gate_map[mode]
            ),
            "correct_greater_shuffle_gate_rate": positive_rate(
                gate_map["correct"] -
                gate_map["visual_shuffle"]
            ),
        }

    return {
        "temperature": float(temperature),
        "target_correct_gate": target,
        "positive_correct_anchor": float(anchor),
        "all_score_center": center,
        "all_score_mad": mad,
        "temperature_from_anchor": float(
            temperature_from_anchor
        ),
        "robust_temperature_floor": float(robust_floor),
        "train_correct_gate": summarize(
            gate_map["correct"]
        ),
        "train_per_mode": per_mode,
    }

def score_objects(
    arrays: dict[str, np.ndarray],
    config: SelfReferenceConfig,
    *,
    min_valid_voxels: int,
) -> list[dict[str, Any]]:
    confidence = arrays["selfref_confidence"]
    support = arrays["selfref_common_support"]
    object_indices = arrays["selfref_object_index"]
    rows: list[dict[str, Any]] = []
    for record in range(confidence.shape[0]):
        hypothesis_rows: dict[str, Any] = {}
        valid = True
        for hypothesis_index, hypothesis in enumerate(EXPECTED_HYPOTHESES):
            result = contrastive_object_score(
                confidence[record, hypothesis_index],
                support[record, hypothesis_index],
                config,
                min_valid_voxels=min_valid_voxels,
            )
            if not np.isfinite(float(result["score"])):
                valid = False
                break
            hypothesis_rows[hypothesis] = {
                "score": float(result["score"]),
                "observed_score": float(result["observed_score"]),
                "reference_score": float(result["reference_score"]),
                "valid_voxel_count": int(result["valid_voxel_count"]),
            }
        if valid:
            rows.append(
                {
                    "object_index": int(object_indices[record]),
                    "hypotheses": hypothesis_rows,
                }
            )
    return rows


def vectors(rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    return {
        hypothesis: np.asarray(
            [float(row["hypotheses"][hypothesis]["score"]) for row in rows],
            dtype=np.float64,
        )
        for hypothesis in EXPECTED_HYPOTHESES
    }


def mode_metrics(scores: dict[str, np.ndarray], mode: str) -> dict[str, Any]:
    correct = scores["correct"]
    wrong = scores[mode]
    shuffle = scores["visual_shuffle"]
    return {
        "object_count": int(correct.size),
        "auc_correct_vs_wrong": binary_auc(correct, wrong),
        "auc_correct_vs_shuffle": binary_auc(correct, shuffle),
        "correct_greater_wrong_rate": positive_rate(correct - wrong),
        "correct_greater_shuffle_rate": positive_rate(correct - shuffle),
        "correct_minus_wrong": summarize(correct - wrong),
        "correct_minus_shuffle": summarize(correct - shuffle),
        "correct_score": summarize(correct),
        "wrong_score": summarize(wrong),
        "shuffle_score": summarize(shuffle),
    }


def evaluate_config(
    arrays: dict[str, np.ndarray],
    config: SelfReferenceConfig,
    *,
    min_valid_voxels: int,
) -> dict[str, Any]:
    rows = score_objects(arrays, config, min_valid_voxels=min_valid_voxels)
    score_map = vectors(rows)
    per_mode: dict[str, Any] = {}
    mode_scores: list[float] = []
    for mode in POSE_MODES:
        metrics = mode_metrics(score_map, mode)
        components = [
            float(metrics["auc_correct_vs_wrong"]),
            float(metrics["auc_correct_vs_shuffle"]),
            float(metrics["correct_greater_wrong_rate"]),
            float(metrics["correct_greater_shuffle_rate"]),
        ]
        selection_score = float(0.75 * min(components) + 0.25 * np.mean(components))
        per_mode[mode] = {
            "metrics": metrics,
            "selection_components": components,
            "selection_score": selection_score,
        }
        mode_scores.append(selection_score)
    minimum = float(min(mode_scores))
    average = float(np.mean(mode_scores))
    return {
        "config": config.to_dict(),
        "object_count": len(rows),
        "per_mode": per_mode,
        "minimum_mode_selection_score": minimum,
        "mean_mode_selection_score": average,
        "selection_score": float(0.75 * minimum + 0.25 * average),
    }


def calibrate_for_scores(
    score_map: dict[str, np.ndarray], args: argparse.Namespace
) -> tuple[dict[str, Any], dict[str, Any], float]:
    correct = score_map["correct"]
    all_negative = np.concatenate(
        [score_map[mode] for mode in POSE_MODES] + [score_map["visual_shuffle"]]
    )
    pooled = calibrate_thresholds(
        correct,
        all_negative,
        min_correct_coverage=float(args.min_train_correct_coverage),
        max_correct_coverage=float(args.max_train_correct_coverage),
        tau_high_quantile=float(args.tau_high_quantile),
        target_correct_coverage=float(args.target_train_correct_coverage),
    )
    per_mode: dict[str, Any] = {}
    for mode in POSE_MODES:
        negative = np.concatenate([score_map[mode], score_map["visual_shuffle"]])
        per_mode[mode] = calibrate_thresholds(
            correct,
            negative,
            min_correct_coverage=float(args.min_train_correct_coverage),
            max_correct_coverage=float(args.max_train_correct_coverage),
            tau_high_quantile=float(args.tau_high_quantile),
            target_correct_coverage=float(args.target_train_correct_coverage),
        )
    tau_low_values = [float(row["tau_low"]) for row in per_mode.values()]
    return pooled, per_mode, float(max(tau_low_values) - min(tau_low_values))


def fresh_mode_evaluation(
    score_map: dict[str, np.ndarray],
    mode: str,
    *,
    temperature: float,
    args: argparse.Namespace,
) -> dict[str, Any]:
    metrics = mode_metrics(score_map, mode)

    correct = score_map["correct"]
    wrong = score_map[mode]
    shuffle = score_map["visual_shuffle"]

    correct_gate = np.asarray(
        sigmoid_gate(correct, temperature),
        dtype=np.float64,
    )
    wrong_gate = np.asarray(
        sigmoid_gate(wrong, temperature),
        dtype=np.float64,
    )
    shuffle_gate = np.asarray(
        sigmoid_gate(shuffle, temperature),
        dtype=np.float64,
    )

    metrics["gate"] = {
        "correct": summarize(correct_gate),
        "wrong": summarize(wrong_gate),
        "shuffle": summarize(shuffle_gate),
        "correct_minus_wrong": summarize(
            correct_gate - wrong_gate
        ),
        "correct_minus_shuffle": summarize(
            correct_gate - shuffle_gate
        ),
        "correct_greater_wrong_rate": positive_rate(
            correct_gate - wrong_gate
        ),
        "correct_greater_shuffle_rate": positive_rate(
            correct_gate - shuffle_gate
        ),
    }

    criteria = {
        "minimum_objects":
            int(metrics["object_count"]) >= int(args.min_objects),

        "score_auc_correct_vs_wrong":
            float(metrics["auc_correct_vs_wrong"])
            >= float(args.min_score_auc),

        "score_auc_correct_vs_shuffle":
            float(metrics["auc_correct_vs_shuffle"])
            >= float(args.min_score_auc),

        "score_win_correct_vs_wrong":
            float(metrics["correct_greater_wrong_rate"])
            >= float(args.min_score_win_rate),

        "score_win_correct_vs_shuffle":
            float(metrics["correct_greater_shuffle_rate"])
            >= float(args.min_score_win_rate),

        "gate_win_correct_vs_wrong":
            float(
                metrics["gate"]["correct_greater_wrong_rate"]
            ) >= float(args.min_gate_win_rate),

        "gate_win_correct_vs_shuffle":
            float(
                metrics["gate"]["correct_greater_shuffle_rate"]
            ) >= float(args.min_gate_win_rate),

        "gate_gap_correct_vs_wrong":
            float(
                metrics["gate"]["correct_minus_wrong"]["mean"]
            ) >= float(args.min_gate_gap),

        "gate_gap_correct_vs_shuffle":
            float(
                metrics["gate"]["correct_minus_shuffle"]["mean"]
            ) >= float(args.min_gate_gap),

        "correct_gate_mean":
            float(metrics["gate"]["correct"]["mean"])
            >= float(args.min_correct_gate_mean),
    }

    return {
        "passed": all(criteria.values()),
        "criteria": criteria,
        "deployment": metrics,
    }

def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    train_report = load_json(args.train_report)
    fresh_report = load_json(args.fresh_report)
    train_arrays = load_npz(args.train_samples)
    fresh_arrays = load_npz(args.fresh_samples)
    validate_inputs(train_report, train_arrays)
    validate_inputs(fresh_report, fresh_arrays)
    if train_report["protocol"]["variants"] != fresh_report["protocol"]["variants"]:
        raise ValueError("train/fresh variant protocols differ")

    configs = build_configs(
        parse_statistics(args.statistics),
        parse_reference_reducers(args.reference_reducers),
    )
    train_results = {
        config.name: evaluate_config(
            train_arrays,
            config,
            min_valid_voxels=int(args.min_valid_voxels),
        )
        for config in configs
    }

    if args.frozen_config_from:
        reference = load_json(args.frozen_config_from)
        payload = reference.get("config", reference.get("calibration", {}).get("config"))
        if not isinstance(payload, dict):
            raise KeyError("frozen config file does not contain config")
        selected = config_from_dict(payload)
        if selected.name not in train_results:
            raise KeyError(f"frozen config {selected.name} is not in current candidates")
        selection_mode = "config_frozen_from_reference"
    else:
        selected = max(
            enumerate(configs),
            key=lambda item: (
                float(train_results[item[1].name]["selection_score"]),
                -int(item[0]),
            ),
        )[1]
        selection_mode = "selected_on_train_objects"

    train_rows = score_objects(
        train_arrays,
        selected,
        min_valid_voxels=int(args.min_valid_voxels),
    )
    train_score_map = vectors(train_rows)
    temperature_calibration = calibrate_sigmoid_temperature(
    train_score_map,
    target_correct_gate=float(
        args.target_train_correct_gate
        ),
    )
    temperature = float(
        temperature_calibration["temperature"]
    )

    fresh_rows = score_objects(
        fresh_arrays,
        selected,
        min_valid_voxels=int(args.min_valid_voxels),
    )
    fresh_score_map = vectors(fresh_rows)
    fresh_per_mode = {
        mode: fresh_mode_evaluation(
            fresh_score_map,
            mode,
            temperature=temperature,
            args=args,
        )
        for mode in POSE_MODES
    }

    selected_train_score = float(train_results[selected.name]["selection_score"])
    global_checks = {
    "train_selection_score":
        selected_train_score >=
        float(args.min_train_selection_score),

    "temperature_finite_positive":
        bool(
            np.isfinite(temperature)
            and temperature > 0.0
        ),

    "all_fresh_modes_passed":
        all(
            row["passed"]
            for row in fresh_per_mode.values()
        ),
}
    passed = all(global_checks.values())

    calibration = {
        "format": "ar_ss_flow.visual_only_pairwise_object_selfref_gate.v2",
        "checkpoint": fresh_report["protocol"].get("checkpoint"),
        "checkpoint_step": fresh_report["protocol"].get("checkpoint_step"),
        "confidence_source": "raw_visual_only_pairwise_confidence",
        "geometry_pair_scale": 0.0,
        "config": selected.to_dict(),
        "minimum_valid_voxels": int(args.min_valid_voxels),
        "hypotheses_used_for_training": list(EXPECTED_HYPOTHESES),
        "runtime_variants": list(EXPECTED_VARIANTS),
        "gate_type": "sigmoid_selfref",
        "temperature": temperature,
        "target_train_correct_gate":
            float(args.target_train_correct_gate),
        "temperature_calibration":
            temperature_calibration,
        "score_formula": "score=stat(observed)-reduce(stat(pose_cyclic1),stat(pose_cyclic2),stat(pose_reverse))",
        "gate_formula":"g_object=sigmoid(self_reference_score/temperature)",
        "residual_formula": "cond_final=cond_base+g_object*delta_cond",
        "stock_condition_untouched": True,
        "selected_on_indices": train_report["protocol"].get("indices"),
        "validated_on_indices": fresh_report["protocol"].get("indices"),
    }
    report = {
        "stage": "C1.6 self-referenced visual-only pairwise object scalar gate calibration",
        "passed": passed,
        "args": vars(args),
        "selection_mode": selection_mode,
        "selected_config": selected.to_dict(),
        "selected_train_score": selected_train_score,
        "train_candidate_comparison": train_results,
        "train_temperature_calibration":
    temperature_calibration,
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
    for config in configs:
        row = train_results[config.name]
        print(
            f"train config={config.name} min_mode={row['minimum_mode_selection_score']:.4f} "
            f"mean_mode={row['mean_mode_selection_score']:.4f} "
            f"selection_score={row['selection_score']:.4f}"
        )
    print("selection_mode:", selection_mode)
    print("selected_config:", selected.to_dict())
    print(
        f"temperature={temperature:.8f} "
        f"target_train_correct_gate="
        f"{args.target_train_correct_gate:.4f}"
    )
    for mode in POSE_MODES:
        row = fresh_per_mode[mode]
        dep = row["deployment"]
        gate = dep["gate"]
        print(
            f"{mode}: passed={row['passed']} objects={dep['object_count']} "
            f"auc_wrong={dep['auc_correct_vs_wrong']:.4f} "
            f"auc_shuffle={dep['auc_correct_vs_shuffle']:.4f} "
            f"score_win_wrong={dep['correct_greater_wrong_rate']:.4f} "
            f"score_win_shuffle={dep['correct_greater_shuffle_rate']:.4f} "
            f"gate_win_wrong={gate['correct_greater_wrong_rate']:.4f} "
            f"gate_win_shuffle={gate['correct_greater_shuffle_rate']:.4f} "
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
