#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pose_point_depth_mv.c1_occupancy import (
    C1_ENRICHMENT_SUMMARY_VERSION,
    load_json,
    summarize,
)
from pose_point_depth_mv.eval_c1_occupancy_calibrator import (
    C1_CALIBRATOR_EVAL_VERSION,
)


C1_CALIBRATOR_SUMMARY_VERSION = "pose_point_depth_mv.c1_nested_calibrator_summary.v2"
SPLITS = ("train16", "fresh48", "holdout")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize C1.1 monotone occupancy probes across seeds/splits."
    )
    parser.add_argument("--c1_summary", required=True)
    parser.add_argument("--report_dirs", nargs="+", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--fail_on_decision", action="store_true")
    return parser.parse_args()


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# C1.1 Multi-seed Occupancy Calibrator Summary",
        "",
        f"- Decision: **{'PASS' if report['passed'] else 'FAIL'}**",
        f"- Seeds: `{report['seeds']}`",
        f"- Weight policy: `{report['weight_policy']}`",
        f"- Target mode: `{report['target_mode']}`",
        "- Flow LoRA: `disabled`",
        "",
        "## Checks",
        "",
    ]
    lines.extend(f"- `{key}`: `{value}`" for key, value in report["checks"].items())
    lines.extend(["", "## Aggregate", "", "```json"])
    lines.append(json.dumps(report["aggregate"], indent=2))
    lines.append("```")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    c1_summary_path = Path(args.c1_summary).resolve()
    c1_summary = load_json(c1_summary_path)
    if c1_summary.get("format") != C1_ENRICHMENT_SUMMARY_VERSION:
        raise ValueError("unexpected C1.0 summary format")
    if c1_summary.get("passed") is not True:
        raise ValueError("C1.1 summary is blocked by failed C1.0")
    paths = [Path(path) / "report.json" for path in args.report_dirs]
    reports = [load_json(path) for path in paths]
    if any(report.get("format") != C1_CALIBRATOR_EVAL_VERSION for report in reports):
        raise ValueError("unexpected C1.1 evaluation report format")
    indexed: dict[tuple[int, str], dict[str, Any]] = {}
    for report in reports:
        key = (int(report["training_seed"]), str(report["split_name"]))
        if key in indexed:
            raise ValueError(f"duplicate C1.1 report: {key}")
        indexed[key] = report
    seeds = sorted(int(seed) for seed in c1_summary["seeds"])
    expected = {(seed, split) for seed in seeds for split in SPLITS}
    policies = {str(report["weight_policy"]) for report in reports}
    targets = {str(report["target_mode"]) for report in reports}
    summary_paths = {Path(report["source_c1_summary"]).resolve() for report in reports}
    checkpoint_steps = {int(report["checkpoint_step"]) for report in reports}
    model_metadata = [report.get("model_metadata") for report in reports]
    decision_protocols = [report.get("decision_protocol") for report in reports]
    corruption_sets = [
        sorted(
            name
            for name in report.get("comparisons", {})
            if name.startswith("M2_corruption_")
        )
        for report in reports
    ]
    normalized_training_protocols: list[dict[str, Any]] = []
    training_seed_bindings_ok = True
    for report in reports:
        protocol = dict(report.get("shared_training_protocol", {}))
        training_seed_bindings_ok = training_seed_bindings_ok and (
            int(protocol.pop("same_initialization_seed", -1))
            == int(report["training_seed"])
        )
        normalized_training_protocols.append(protocol)

    same_checkpoint_per_seed = True
    disjoint_per_seed = True
    for seed in seeds:
        seed_reports = [indexed.get((seed, split)) for split in SPLITS]
        if any(report is None for report in seed_reports):
            same_checkpoint_per_seed = False
            disjoint_per_seed = False
            continue
        checkpoints = {
            (report["checkpoint"], report["checkpoint_sha256"])
            for report in seed_reports
            if report is not None
        }
        same_checkpoint_per_seed = same_checkpoint_per_seed and len(checkpoints) == 1
        uid_sets = [
            {str(row["uid"]) for row in report["records"]}
            for report in seed_reports
            if report is not None
        ]
        disjoint_per_seed = disjoint_per_seed and all(
            uid_sets[left].isdisjoint(uid_sets[right])
            for left in range(len(uid_sets))
            for right in range(left + 1, len(uid_sets))
        )

    aggregate: dict[str, Any] = {}
    for split in SPLITS:
        aggregate[split] = {}
        for control in (
            "M0_bias",
            "M1_reliability",
            "M2_spatial_permuted",
            "M2_hardest_corruption",
        ):
            aggregate[split][control] = {}
            for metric in (
                "balanced_bce_gain",
                "average_precision_gain",
                "weighted_target_rate_gain",
            ):
                values = [
                    float(indexed[(seed, split)]["comparisons"][control][metric]["object"]["mean"])
                    for seed in seeds
                    if (seed, split) in indexed
                ]
                aggregate[split][control][metric] = summarize(values)

    checks = {
        "c1_0_summary_passed": c1_summary.get("passed") is True,
        "complete_seed_split_grid": set(indexed) == expected,
        "single_weight_policy": len(policies) == 1,
        "policy_matches_c1_0_admission": policies == {c1_summary["admitted_policy"]},
        "single_target_mode": len(targets) == 1,
        "single_checkpoint_step": len(checkpoint_steps) == 1,
        "nested_model_architecture_consistent": bool(model_metadata)
        and all(row == model_metadata[0] for row in model_metadata[1:]),
        "decision_protocol_consistent": bool(decision_protocols)
        and all(row == decision_protocols[0] for row in decision_protocols[1:]),
        "corruption_control_set_consistent": bool(corruption_sets)
        and all(row == corruption_sets[0] for row in corruption_sets[1:]),
        "shared_training_protocol_consistent": bool(normalized_training_protocols)
        and all(
            row == normalized_training_protocols[0]
            for row in normalized_training_protocols[1:]
        ),
        "initialization_seed_bound_to_training_seed": training_seed_bindings_ok,
        "all_reports_bound_to_same_c1_0_summary": summary_paths == {c1_summary_path},
        "same_calibrator_checkpoint_per_seed": same_checkpoint_per_seed,
        "splits_object_disjoint_per_seed": disjoint_per_seed,
        "all_train_fresh_holdout_reports_pass": all(
            report.get("passed") is True for report in reports
        ),
        "flow_lora_disabled": all(
            report.get("flow_lora_enabled") is False for report in reports
        ),
    }
    passed = all(checks.values())
    report = {
        "format": C1_CALIBRATOR_SUMMARY_VERSION,
        "stage": "C1.1 multi-seed monotone occupancy probe admission",
        "passed": passed,
        "seeds": seeds,
        "splits": list(SPLITS),
        "weight_policy": next(iter(policies)) if len(policies) == 1 else None,
        "target_mode": next(iter(targets)) if len(targets) == 1 else None,
        "source_c1_summary": str(c1_summary_path),
        "source_reports": [str(path.resolve()) for path in paths],
        "checks": checks,
        "aggregate": aggregate,
        "flow_lora_enabled": False,
        "allowed_next_stage": (
            "C1.2 frozen-base local decoder-logit causal adapter"
            if passed
            else "stop before any decoder or Flow modification"
        ),
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
