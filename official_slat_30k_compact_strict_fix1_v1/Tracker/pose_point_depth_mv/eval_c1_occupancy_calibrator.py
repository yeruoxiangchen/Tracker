#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any

import torch

from pose_point_depth_mv.c1_occupancy import (
    C1_CALIBRATOR_CHECKPOINT_VERSION,
    C1MapTargetDataset,
    MonotoneOccupancyCalibrator,
    average_precision,
    balanced_binary_loss,
    c1_policy_scores,
    comparison_summary,
    file_sha256,
    load_json,
    permute_within_active,
    policy_metrics,
    summarize,
)


C1_CALIBRATOR_EVAL_VERSION = "pose_point_depth_mv.c1_nested_calibrator_eval.v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a C1.1 monotone occupancy calibrator with causal controls."
    )
    parser.add_argument("--c0_report", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--permutation_repeats", type=int, default=16)
    parser.add_argument("--bootstrap_samples", type=int, default=10000)
    parser.add_argument("--min_object_win_rate", type=float, default=0.65)
    parser.add_argument("--fail_on_decision", action="store_true")
    return parser.parse_args()


def branch_metrics(logits: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    probabilities = torch.sigmoid(logits.float())
    active = torch.ones_like(target, dtype=torch.bool)
    ranking = policy_metrics(probabilities, target, active)
    return {
        "balanced_bce": float(balanced_binary_loss(logits, target).item()),
        "average_precision": average_precision(logits, target),
        "weighted_target_rate": float(ranking["weighted_target_rate"]),
        "top_10_target_rate": float(ranking["top_10_target_rate"]),
        "probability_mean": float(probabilities.mean().item()),
    }


def average_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    return {key: mean(row[key] for row in rows) for key in rows[0]}


def compare_records(
    records: list[dict[str, Any]], control: str, *, bootstrap_samples: int
) -> dict[str, Any]:
    loss_gain = [
        float(row["branches"][control]["balanced_bce"])
        - float(row["branches"]["M2_correct"]["balanced_bce"])
        for row in records
    ]
    ap_gain = [
        float(row["branches"]["M2_correct"]["average_precision"])
        - float(row["branches"][control]["average_precision"])
        for row in records
    ]
    target_rate_gain = [
        float(row["branches"]["M2_correct"]["weighted_target_rate"])
        - float(row["branches"][control]["weighted_target_rate"])
        for row in records
    ]
    return {
        "balanced_bce_gain": comparison_summary(
            loss_gain, bootstrap_samples=bootstrap_samples
        ),
        "average_precision_gain": comparison_summary(
            ap_gain, bootstrap_samples=bootstrap_samples
        ),
        "weighted_target_rate_gain": comparison_summary(
            target_rate_gain, bootstrap_samples=bootstrap_samples
        ),
    }


def compare_to_hardest_corruption(
    records: list[dict[str, Any]],
    controls: list[str],
    *,
    bootstrap_samples: int,
) -> dict[str, Any]:
    if not controls:
        raise ValueError("C1.1 requires real corruption-derived controls")
    loss_gains: list[float] = []
    ap_gains: list[float] = []
    target_rate_gains: list[float] = []
    strongest_counts = {name: 0 for name in controls}
    for row in records:
        branches = row["branches"]
        strongest = min(controls, key=lambda name: branches[name]["balanced_bce"])
        strongest_counts[strongest] += 1
        loss_gains.append(
            float(branches[strongest]["balanced_bce"])
            - float(branches["M2_correct"]["balanced_bce"])
        )
        strongest_ap = max(controls, key=lambda name: branches[name]["average_precision"])
        ap_gains.append(
            float(branches["M2_correct"]["average_precision"])
            - float(branches[strongest_ap]["average_precision"])
        )
        strongest_rate = max(
            controls, key=lambda name: branches[name]["weighted_target_rate"]
        )
        target_rate_gains.append(
            float(branches["M2_correct"]["weighted_target_rate"])
            - float(branches[strongest_rate]["weighted_target_rate"])
        )
    return {
        "balanced_bce_gain": comparison_summary(
            loss_gains, bootstrap_samples=bootstrap_samples
        ),
        "average_precision_gain": comparison_summary(
            ap_gains, bootstrap_samples=bootstrap_samples
        ),
        "weighted_target_rate_gain": comparison_summary(
            target_rate_gains, bootstrap_samples=bootstrap_samples
        ),
        "controls": controls,
        "strongest_by_bce_counts": strongest_counts,
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# C1.1 Monotone Occupancy Calibrator Evaluation",
        "",
        f"- Split: `{report['split_name']}`",
        f"- Decision: **{'PASS' if report['passed'] else 'FAIL'}**",
        f"- Objects: `{report['object_count']}`",
        f"- Weight policy: `{report['weight_policy']}`",
        f"- Target mode: `{report['target_mode']}`",
        "- Flow/decoder loaded: `false`",
        "",
        "## Checks",
        "",
    ]
    lines.extend(f"- `{key}`: `{value}`" for key, value in report["checks"].items())
    lines.extend(["", "## Comparisons", "", "```json"])
    lines.append(json.dumps(report["comparisons"], indent=2))
    lines.extend(["```", "", "## View Groups", "", "```json"])
    lines.append(json.dumps(report["view_groups"], indent=2))
    lines.append("```")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if int(args.permutation_repeats) <= 0:
        raise ValueError("--permutation_repeats must be positive")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(args.checkpoint).resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if checkpoint.get("format") != C1_CALIBRATOR_CHECKPOINT_VERSION:
        raise ValueError("unexpected C1.1 checkpoint format")
    if file_sha256(checkpoint["source_c1_summary"]) != checkpoint.get(
        "source_c1_summary_sha256"
    ):
        raise ValueError("C1.0 summary changed after C1.1 training")
    c0_path = Path(args.c0_report).resolve()
    dataset = C1MapTargetDataset(c0_path)
    if int(dataset.report["training_seed"]) != int(
        load_json(checkpoint["source_c0_report"])["training_seed"]
    ):
        raise ValueError("C1.1 evaluation seed differs from training map seed")
    if int(checkpoint["training_seed"]) != int(dataset.report["training_seed"]):
        raise ValueError("C1.1 calibrator seed differs from C0 training seed")
    if str(dataset.report["checkpoint_sha256"]) != str(
        checkpoint["source_c0_checkpoint_sha256"]
    ):
        raise ValueError("C1.1 evaluation C0 checkpoint differs from training")

    expected_models = {
        "M0_bias": (False, False),
        "M1_reliability": (False, True),
        "M2_weight_reliability": (True, True),
    }
    models: dict[str, MonotoneOccupancyCalibrator] = {}
    for name, (include_score, include_reliability) in expected_models.items():
        metadata = checkpoint["model_metadata"].get(name, {})
        if (
            bool(metadata.get("include_score")) != include_score
            or bool(metadata.get("include_reliability")) != include_reliability
        ):
            raise ValueError(f"C1.1 nested model metadata mismatch for {name}")
        model = MonotoneOccupancyCalibrator(
            include_score=include_score,
            include_reliability=include_reliability,
        )
        model.load_state_dict(checkpoint["model_states"][name], strict=True)
        model.eval()
        models[name] = model
    weight_policy = str(checkpoint["weight_policy"])
    target_mode = str(checkpoint["target_mode"])
    records: list[dict[str, Any]] = []
    corruption_prefix = f"corruption_{weight_policy}_"
    with torch.no_grad():
        for index in range(len(dataset)):
            item = dataset[index]
            payload = item["map"]
            active = payload["active_mask"].bool()
            scores = c1_policy_scores(payload)
            if weight_policy not in scores:
                raise ValueError(f"map lacks checkpoint policy={weight_policy}")
            score = scores[weight_policy]
            reliability = payload["audit_maps"]["raw_reliability"].float()
            target = item["targets"][target_mode].float()
            selected_target = target[active]
            selected_score = score[active]
            selected_reliability = reliability[active]
            correct = branch_metrics(
                models["M2_weight_reliability"](
                    selected_score, selected_reliability
                ),
                selected_target,
            )
            reliability_only = branch_metrics(
                models["M1_reliability"](
                    torch.zeros_like(selected_score), selected_reliability
                ),
                selected_target,
            )
            bias_only = branch_metrics(
                models["M0_bias"](
                    torch.zeros_like(selected_score),
                    torch.zeros_like(selected_reliability),
                ),
                selected_target,
            )
            permuted_rows = []
            for repeat in range(int(args.permutation_repeats)):
                permuted = permute_within_active(
                    score,
                    active,
                    uid=item["uid"],
                    repeat=repeat,
                )
                permuted_rows.append(
                    branch_metrics(
                        models["M2_weight_reliability"](
                            permuted[active], selected_reliability
                        ),
                        selected_target,
                    )
                )
            branches = {
                "M2_correct": correct,
                "M1_reliability": reliability_only,
                "M0_bias": bias_only,
                "M2_spatial_permuted": average_rows(permuted_rows),
            }
            corruption_branches = sorted(
                name for name in scores if name.startswith(corruption_prefix)
            )
            if not corruption_branches:
                raise ValueError(
                    f"map lacks real corruption branches for policy={weight_policy}"
                )
            for policy_name in corruption_branches:
                short_name = f"M2_corruption_{policy_name[len(corruption_prefix):]}"
                branches[short_name] = branch_metrics(
                    models["M2_weight_reliability"](
                        scores[policy_name][active], selected_reliability
                    ),
                    selected_target,
                )
            records.append(
                {
                    "uid": item["uid"],
                    "object_uid": item["object_uid"],
                    "views": item["views"],
                    "active_count": int(active.sum().item()),
                    "target_positive_count": int(selected_target.sum().item()),
                    "branches": branches,
                }
            )
            print(
                f"[c1_calibrator_eval] {index + 1}/{len(dataset)} "
                f"uid={item['uid']} views={item['views']} "
                f"gain_rel={reliability_only['balanced_bce'] - correct['balanced_bce']:.6g} "
                f"gain_perm={branches['M2_spatial_permuted']['balanced_bce'] - correct['balanced_bce']:.6g}",
                flush=True,
            )

    standard_controls = (
        "M0_bias",
        "M1_reliability",
        "M2_spatial_permuted",
    )
    corruption_controls = sorted(
        name for name in records[0]["branches"] if name.startswith("M2_corruption_")
    )
    comparisons = {
        control: compare_records(
            records, control, bootstrap_samples=int(args.bootstrap_samples)
        )
        for control in (*standard_controls, *corruption_controls)
    }
    comparisons["M2_hardest_corruption"] = compare_to_hardest_corruption(
        records,
        corruption_controls,
        bootstrap_samples=int(args.bootstrap_samples),
    )
    checks: dict[str, bool] = {
        "checkpoint_complete": int(checkpoint["step"]) > 0,
        "same_c0_checkpoint_across_train_and_eval": True,
        "three_independent_nested_models_loaded": set(models) == set(expected_models),
        "target_not_used_as_input": True,
        "flow_not_loaded": True,
        "decoder_not_loaded": True,
    }
    for control in (
        "M1_reliability",
        "M2_spatial_permuted",
        "M2_hardest_corruption",
    ):
        loss = comparisons[control]["balanced_bce_gain"]
        checks[f"beats_{control}_loss_mean"] = loss["object"]["mean"] > 0.0
        checks[f"beats_{control}_loss_median"] = loss["object"]["median"] > 0.0
        checks[f"beats_{control}_loss_win_rate"] = (
            loss["object_win_rate"] >= float(args.min_object_win_rate)
        )
        checks[f"beats_{control}_loss_ci"] = loss["object_bootstrap_95_ci"][0] > 0.0
    for control in corruption_controls:
        checks[f"{control}_loss_mean_nonnegative"] = (
            comparisons[control]["balanced_bce_gain"]["object"]["mean"] >= 0.0
        )

    # M2-vs-M0 is auxiliary, not a substitute for the formal nested M2-vs-M1 test.
    checks["beats_M0_bias_loss_mean"] = (
        comparisons["M0_bias"]["balanced_bce_gain"]["object"]["mean"] > 0.0
    )

    view_groups: dict[str, Any] = {}
    for views in sorted({int(row["views"]) for row in records}):
        group = [row for row in records if int(row["views"]) == views]
        view_groups[str(views)] = {
            "object_count": len(group),
            "vs_M1_reliability": compare_records(
                group,
                "M1_reliability",
                bootstrap_samples=int(args.bootstrap_samples),
            ),
            "vs_M2_spatial_permuted": compare_records(
                group,
                "M2_spatial_permuted",
                bootstrap_samples=int(args.bootstrap_samples),
            ),
            "vs_M2_hardest_corruption": compare_to_hardest_corruption(
                group,
                corruption_controls,
                bootstrap_samples=int(args.bootstrap_samples),
            ),
        }
    if dataset.report["split_name"] in {"fresh48", "holdout"}:
        two_view = view_groups.get("2")
        checks["two_view_present"] = two_view is not None
        if two_view is not None:
            checks["two_view_M2_vs_M1_nonnegative"] = (
                two_view["vs_M1_reliability"]["balanced_bce_gain"]["object"]["mean"]
                >= 0.0
            )
            checks["two_view_spatial_nonnegative"] = (
                two_view["vs_M2_spatial_permuted"]["balanced_bce_gain"]["object"]["mean"]
                >= 0.0
            )
            checks["two_view_corruption_nonnegative"] = (
                two_view["vs_M2_hardest_corruption"]["balanced_bce_gain"]["object"]["mean"]
                >= 0.0
            )
    branch_aggregate = {
        branch: {
            field: summarize(row["branches"][branch][field] for row in records)
            for field in records[0]["branches"][branch]
        }
        for branch in records[0]["branches"]
    }
    passed = all(checks.values())
    decision_protocol = {
        "primary_metric": "object-balanced BCE gain",
        "formal_nested_control": "independently-trained M1_reliability",
        "auxiliary_nested_control": "independently-trained M0_bias",
        "spatial_control": "M2 with weight permuted inside fixed active support",
        "corruption_control": (
            "M2 with real pose/depth/visual corruption-derived weight, fixed "
            "active support, and unchanged raw reliability"
        ),
        "average_precision": "tie-aware threshold-grouped; diagnostic only",
        "fresh_holdout_two_view_mean_must_be_nonnegative": True,
    }
    report = {
        "format": C1_CALIBRATOR_EVAL_VERSION,
        "stage": "C1.1 monotone occupancy calibrator causal evaluation",
        "passed": passed,
        "split_name": dataset.report["split_name"],
        "training_seed": int(dataset.report["training_seed"]),
        "object_count": len(records),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "checkpoint_step": int(checkpoint["step"]),
        "weight_policy": weight_policy,
        "target_mode": target_mode,
        "permutation_repeats": int(args.permutation_repeats),
        "source_c0_report": str(dataset.report_path),
        "source_c0_checkpoint_sha256": dataset.report["checkpoint_sha256"],
        "source_c1_summary": checkpoint["source_c1_summary"],
        "source_c1_summary_sha256": checkpoint["source_c1_summary_sha256"],
        "model_metadata": checkpoint["model_metadata"],
        "parameter_values": checkpoint["parameter_values"],
        "shared_training_protocol": checkpoint["shared_training_protocol"],
        "decision_protocol": decision_protocol,
        "flow_loaded": False,
        "flow_lora_enabled": False,
        "decoder_loaded": False,
        "target_used_as_input": False,
        "checks": checks,
        "branches": branch_aggregate,
        "comparisons": comparisons,
        "view_groups": view_groups,
        "records": records,
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    write_markdown(report, output_dir / "report.md")
    print(json.dumps({"passed": passed, "checks": checks}, indent=2))
    if args.fail_on_decision and not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
