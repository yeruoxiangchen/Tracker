#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pose_point_depth_mv.c1_direct_occupancy import (
    C1_DIRECT_OCCUPANCY_CHECKPOINT_VERSION,
    C1_DIRECT_OCCUPANCY_EVAL_VERSION,
    C1_DIRECT_OCCUPANCY_SUMMARY_VERSION,
)
from pose_point_depth_mv.c1_matched_budget import (
    C1_MATCHED_BUDGET_SUMMARY_VERSION,
    CORRUPTION_POLICY_NAMES,
)
from pose_point_depth_mv.c1_occupancy import load_json, summarize


SPLITS = ("fresh48", "holdout")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize three-seed C1.1b direct occupancy evaluations."
    )
    parser.add_argument("--c1_0b_summary", required=True)
    parser.add_argument("--report_dirs", nargs="+", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--fail_on_integrity", action="store_true")
    return parser.parse_args()


def _target_signature(report: dict[str, Any]) -> dict[str, str]:
    return {
        str(row["uid"]): str(
            row["target_mapping_audit"]["exact_target_mask_sha256"]
        )
        for row in report["records"]
    }


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# C1.1b Multi-seed Direct Occupancy Summary",
        "",
        f"- Decision: **{'PASS' if report['passed'] else 'FAIL'}**",
        f"- Allowed next stage: `{report['allowed_next_stage']}`",
        "- Flow LoRA: `disabled`",
        "",
        "## Checks",
        "",
    ]
    lines.extend(
        f"- `{name}`: `{value}`" for name, value in report["checks"].items()
    )
    lines.extend(["", "## Aggregate", "", "```json"])
    lines.append(json.dumps(report["aggregate"], indent=2))
    lines.append("```")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    summary_path = Path(args.c1_0b_summary).resolve()
    c1_0b = load_json(summary_path)
    if c1_0b.get("format") != C1_MATCHED_BUDGET_SUMMARY_VERSION:
        raise ValueError("unexpected C1.0b summary format")
    if c1_0b.get("passed") is not True:
        raise ValueError("C1.1b summary is blocked by C1.0b")
    paths = [Path(value).resolve() / "report.json" for value in args.report_dirs]
    reports = [load_json(path) for path in paths]
    if any(
        report.get("format") != C1_DIRECT_OCCUPANCY_EVAL_VERSION
        for report in reports
    ):
        raise ValueError("unexpected C1.1b eval report format")
    indexed: dict[tuple[int, str], dict[str, Any]] = {}
    for report in reports:
        key = (int(report["training_seed"]), str(report["split_name"]))
        if key in indexed:
            raise ValueError(f"duplicate C1.1b eval report {key}")
        indexed[key] = report
    seeds = sorted(int(value) for value in c1_0b["seeds"])
    expected = {(seed, split) for seed in seeds for split in SPLITS}

    checkpoint_binding = True
    train_reports: dict[int, dict[str, Any]] = {}
    for seed in seeds:
        seed_reports = [indexed.get((seed, split)) for split in SPLITS]
        if any(report is None for report in seed_reports):
            checkpoint_binding = False
            continue
        checkpoint_paths = {report["source_checkpoint"] for report in seed_reports}
        checkpoint_hashes = {report["source_checkpoint_sha256"] for report in seed_reports}
        checkpoint_binding = checkpoint_binding and len(checkpoint_paths) == 1 and len(
            checkpoint_hashes
        ) == 1
        checkpoint_path = Path(next(iter(checkpoint_paths)))
        train_path = checkpoint_path.parents[1] / "train_report.json"
        if not train_path.is_file():
            checkpoint_binding = False
            continue
        train = load_json(train_path)
        train_reports[seed] = train
        checkpoint_binding = checkpoint_binding and bool(
            train.get("format") == C1_DIRECT_OCCUPANCY_CHECKPOINT_VERSION
            and train.get("passed") is True
            and int(train.get("training_seed", -1)) == seed
            and Path(train.get("checkpoint", "")).resolve() == checkpoint_path.resolve()
        )

    protocol_signatures = {
        (
            report["target_mode"],
            report["policy"],
            report["feature_metadata"]["version"],
            report["feature_metadata"]["spatial_tolerance"],
            report["feature_metadata"]["base_feature_definition"],
            report["feature_metadata"]["correspondence_feature_definition"],
            report["feature_metadata"]["input_dim"],
            json.dumps(report["decision_thresholds"], sort_keys=True),
        )
        for report in reports
    }
    same_protocol = len(protocol_signatures) == 1
    same_summary = all(
        Path(report["source_c1_0b_summary"]).resolve() == summary_path
        for report in reports
    )
    split_targets = {
        split: [
            _target_signature(indexed[(seed, split)])
            for seed in seeds
            if (seed, split) in indexed
        ]
        for split in SPLITS
    }
    targets_equal = all(
        len(rows) == len(seeds) and all(row == rows[0] for row in rows[1:])
        for rows in split_targets.values()
    )
    split_uids = {
        split: set(rows[0]) if rows else set()
        for split, rows in split_targets.items()
    }
    splits_disjoint = split_uids["fresh48"].isdisjoint(split_uids["holdout"])

    formal_names = (
        "M2_vs_M1_balanced_bce",
        *(
            f"M2_correct_vs_{branch}_balanced_bce"
            for branch in CORRUPTION_POLICY_NAMES
        ),
    )
    all_formal_pass = all(
        report["checks"].get(name) is True
        for report in reports
        for name in formal_names
    )
    all_two_view_pass = all(
        report["checks"].get("two_view_present") is True
        and all(
            report["checks"].get(f"two_view_{name}_nonnegative") is True
            for name in formal_names
        )
        for report in reports
    )
    all_integrity = all(report.get("integrity_passed") is True for report in reports)

    aggregate: dict[str, Any] = {}
    for split in SPLITS:
        aggregate[split] = {}
        for name in formal_names:
            aggregate[split][name] = {
                "per_seed_object_mean": summarize(
                    indexed[(seed, split)]["comparisons"][name]["object"]["mean"]
                    for seed in seeds
                ),
                "per_seed_object_win_rate": summarize(
                    indexed[(seed, split)]["comparisons"][name]["object_win_rate"]
                    for seed in seeds
                ),
                "per_seed_ci_low": summarize(
                    indexed[(seed, split)]["comparisons"][name][
                        "object_bootstrap_95_ci"
                    ][0]
                    for seed in seeds
                ),
            }

    checks = {
        "c1_0b_admitted": c1_0b.get("passed") is True,
        "expected_three_seed_fresh_holdout_grid": set(indexed) == expected,
        "same_c1_0b_summary": same_summary,
        "same_seed_splits_use_identical_probe_checkpoint": checkpoint_binding,
        "all_train_reports_passed": len(train_reports) == len(seeds),
        "protocol_consistent": same_protocol,
        "same_split_target_masks_identical_across_seeds": targets_equal,
        "fresh_holdout_object_disjoint": splits_disjoint,
        "all_eval_integrity_checks_pass": all_integrity,
        "m2_beats_m1_and_all_corruptions_in_every_seed_split": all_formal_pass,
        "two_view_protection_in_every_seed_split": all_two_view_pass,
        "flow_lora_disabled": True,
    }
    passed = all(checks.values())
    report = {
        "format": C1_DIRECT_OCCUPANCY_SUMMARY_VERSION,
        "stage": "C1.1b three-seed object-disjoint direct occupancy decision",
        "passed": passed,
        "checks": checks,
        "seeds": seeds,
        "splits": list(SPLITS),
        "source_c1_0b_summary": str(summary_path),
        "source_reports": [str(path) for path in paths],
        "aggregate": aggregate,
        "allowed_next_stage": (
            "C1.2 frozen-base causal occupancy adapter"
            if passed
            else "stop before C1.2; correspondence remains audit/quality feature"
        ),
        "flow_lora_enabled": False,
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    _write_markdown(report, output_dir / "report.md")
    print(
        json.dumps(
            {
                "passed": passed,
                "checks": checks,
                "allowed_next_stage": report["allowed_next_stage"],
            },
            indent=2,
        ),
        flush=True,
    )
    if args.fail_on_integrity and not all(
        checks[name]
        for name in (
            "expected_three_seed_fresh_holdout_grid",
            "same_c1_0b_summary",
            "same_seed_splits_use_identical_probe_checkpoint",
            "all_train_reports_passed",
            "protocol_consistent",
            "same_split_target_masks_identical_across_seeds",
            "fresh_holdout_object_disjoint",
            "all_eval_integrity_checks_pass",
        )
    ):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
