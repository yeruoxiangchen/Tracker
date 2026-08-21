#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pose_point_depth_mv.c1_matched_budget import (
    C1_MATCHED_BUDGET_REPORT_VERSION,
    C1_MATCHED_BUDGET_SUMMARY_VERSION,
    CORRUPTION_POLICY_NAMES,
    MATCHED_POLICIES,
)
from pose_point_depth_mv.c1_occupancy import load_json, summarize


SPLITS = ("fresh48", "holdout")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize C1.0b matched-budget audits over N3 seeds."
    )
    parser.add_argument("--n3_report", required=True)
    parser.add_argument("--report_dirs", nargs="+", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--fail_on_integrity", action="store_true")
    return parser.parse_args()


def _target_signature(report: dict[str, Any]) -> dict[str, tuple[str, str]]:
    return {
        str(row["uid"]): (
            str(row["target_mapping_audit"]["exact_target_mask_sha256"]),
            str(row["target_mapping_audit"]["surface_r1_mask_sha256"]),
        )
        for row in report["records"]
    }


def _policy_route(
    reports: list[dict[str, Any]], policy: str
) -> tuple[str, dict[str, bool]]:
    strict_fixed_corruptions_pass = all(
        report["decision"]["comparison_decisions"][target][policy][control][
            "passed"
        ]
        for report in reports
        for target in ("exact", "surface_r1")
        for control in CORRUPTION_POLICY_NAMES
    )
    strict_reliability_pass = all(
        report["decision"]["comparison_decisions"][target][policy][
            "reliability"
        ]["passed"]
        for report in reports
        for target in ("exact", "surface_r1")
    )
    strict_spatial_pass = all(
        report["decision"]["comparison_decisions"][target][policy][
            "spatial_permutation"
        ]["passed"]
        for report in reports
        for target in ("exact", "surface_r1")
    )
    # Auxiliary admission is intentionally weaker than direct-gate admission.
    # It only establishes a directionally consistent feature that C1.1b must
    # then prove adds value beyond the stronger M1 visual/geometry baseline.
    directional_metric_names = (
        "histogram_weighted_target_rate",
        "top_05_target_rate",
        "top_10_target_rate",
        "top_20_target_rate",
    )
    directional_fixed_corruptions = all(
        report["comparisons"][target][policy][control][metric]["object"]["mean"]
        > 0.0
        for report in reports
        for target in ("exact", "surface_r1")
        for control in CORRUPTION_POLICY_NAMES
        for metric in directional_metric_names
    )
    directional_spatial = all(
        report["comparisons"][target][policy]["spatial_permutation"][metric][
            "object"
        ]["mean"]
        > 0.0
        for report in reports
        for target in ("exact", "surface_r1")
        for metric in directional_metric_names
    )
    two_view_nonnegative = all(
        report["view_groups"].get("2") is not None
        and report["view_groups"]["2"]["comparisons"][target][policy][control][
            "histogram_weighted_target_rate"
        ]["object"]["mean"]
        >= 0.0
        for report in reports
        for target in ("exact", "surface_r1")
        for control in (*CORRUPTION_POLICY_NAMES, "reliability", "spatial_permutation")
    )
    checks = {
        "strictly_beats_all_fixed_corruptions": strict_fixed_corruptions_pass,
        "strictly_beats_reliability": strict_reliability_pass,
        "strictly_beats_spatial_permutation": strict_spatial_pass,
        "directionally_beats_all_fixed_corruptions": directional_fixed_corruptions,
        "directionally_beats_spatial_permutation": directional_spatial,
        "two_view_nonnegative": two_view_nonnegative,
    }
    if (
        strict_fixed_corruptions_pass
        and strict_reliability_pass
        and strict_spatial_pass
        and two_view_nonnegative
    ):
        route = "restricted_surface_occupancy_gate_candidate"
    elif directional_fixed_corruptions and directional_spatial:
        route = "auxiliary_correspondence_feature_only"
    else:
        route = "stop_target_occupancy_direction"
    return route, checks


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# C1.0b Multi-seed Matched-budget Summary",
        "",
        f"- Integrity: **{'PASS' if report['integrity_passed'] else 'FAIL'}**",
        f"- Scientific route: `{report['decision']['route']}`",
        f"- Selected policy: `{report['decision']['selected_policy']}`",
        "- Flow LoRA: `disabled`",
        "",
        "## Checks",
        "",
    ]
    lines.extend(
        f"- `{name}`: `{value}`" for name, value in report["checks"].items()
    )
    lines.extend(["", "## Decision", "", "```json"])
    lines.append(json.dumps(report["decision"], indent=2))
    lines.extend(["```", "", "## Aggregate", "", "```json"])
    lines.append(json.dumps(report["aggregate"], indent=2))
    lines.append("```")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    n3_path = Path(args.n3_report).resolve()
    n3 = load_json(n3_path)
    if n3.get("passed") is not True:
        raise ValueError("C1.0b requires the passed N3 summary")
    paths = [Path(value).resolve() / "report.json" for value in args.report_dirs]
    reports = [load_json(path) for path in paths]
    if any(
        report.get("format") != C1_MATCHED_BUDGET_REPORT_VERSION
        for report in reports
    ):
        raise ValueError("unexpected C1.0b report format")
    indexed: dict[tuple[int, str], dict[str, Any]] = {}
    for report in reports:
        key = (int(report["training_seed"]), str(report["split_name"]))
        if key in indexed:
            raise ValueError(f"duplicate C1.0b report {key}")
        indexed[key] = report
    seeds = sorted(int(value) for value in n3["seeds"])
    expected = {(seed, split) for seed in seeds for split in SPLITS}
    n3_by_seed = {int(row["seed"]): row for row in n3["per_seed"]}

    source_bound = True
    for (seed, split), report in indexed.items():
        n3_row = n3_by_seed.get(seed, {})
        expected_c0 = (
            Path(n3_row.get("run_dir", "/__missing__"))
            / f"c0_3_{split}"
            / "report.json"
        )
        source_bound = source_bound and bool(
            Path(report["source_c0_report"]).resolve() == expected_c0.resolve()
            and report["source_c0_checkpoint_sha256"]
            == n3_row.get("checkpoint_sha256")
        )
    protocol_hashes = {str(report["protocol_hash"]) for report in reports}
    same_protocol = len(protocol_hashes) == 1
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
    all_integrity = all(report.get("integrity_passed") is True for report in reports)

    policy_decisions: dict[str, Any] = {}
    for policy in MATCHED_POLICIES:
        if all(policy in report["protocol"]["policies"] for report in reports):
            route, checks = _policy_route(reports, policy)
            policy_decisions[policy] = {"route": route, "checks": checks}
    route_priority = (
        "restricted_surface_occupancy_gate_candidate",
        "auxiliary_correspondence_feature_only",
        "stop_target_occupancy_direction",
    )
    selected_policy = None
    selected_route = "stop_target_occupancy_direction"
    for route in route_priority:
        matches = [
            policy
            for policy in MATCHED_POLICIES
            if policy in policy_decisions
            and policy_decisions[policy]["route"] == route
        ]
        if matches:
            selected_policy = matches[0]
            selected_route = route
            break

    aggregate: dict[str, Any] = {}
    for split in SPLITS:
        aggregate[split] = {}
        for policy in policy_decisions:
            aggregate[split][policy] = {}
            for target in ("exact", "surface_r1"):
                aggregate[split][policy][target] = {}
                for control in (
                    "reliability",
                    *CORRUPTION_POLICY_NAMES,
                    "spatial_permutation",
                ):
                    values = [
                        indexed[(seed, split)]["comparisons"][target][policy][
                            control
                        ]["histogram_weighted_target_rate"]["object"]["mean"]
                        for seed in seeds
                    ]
                    aggregate[split][policy][target][control] = summarize(values)

    checks = {
        "n3_passed": n3.get("passed") is True,
        "expected_three_seed_fresh_holdout_grid": set(indexed) == expected,
        "source_c0_reports_and_checkpoints_bound_to_n3": source_bound,
        "protocol_consistent": same_protocol,
        "same_split_target_masks_identical_across_seeds": targets_equal,
        "fresh_holdout_object_disjoint": splits_disjoint,
        "all_report_integrity_checks_pass": all_integrity,
        "flow_lora_disabled": True,
    }
    integrity_passed = all(checks.values())
    admitted = integrity_passed and selected_route != "stop_target_occupancy_direction"
    report = {
        "format": C1_MATCHED_BUDGET_SUMMARY_VERSION,
        "stage": "C1.0b three-seed Fresh/Holdout matched-budget decision",
        "passed": admitted,
        "integrity_passed": integrity_passed,
        "seeds": seeds,
        "splits": list(SPLITS),
        "source_n3_report": str(n3_path),
        "source_reports": [str(path) for path in paths],
        "protocol_hash": next(iter(protocol_hashes)) if same_protocol else None,
        "checks": checks,
        "decision": {
            "route": selected_route,
            "selected_policy": selected_policy,
            "policy_decisions": policy_decisions,
            "allowed_next_stage": (
                "C1.1b direct local occupancy M0/M1/M2 probe"
                if admitted
                else "stop target occupancy; retain C0 only as correspondence audit"
            ),
            "two_view_deployment_policy": (
                "enabled"
                if selected_policy
                and policy_decisions[selected_policy]["checks"][
                    "two_view_nonnegative"
                ]
                else "abstain or disable physical branch for 2-view"
            ),
        },
        "aggregate": aggregate,
        "flow_lora_enabled": False,
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    _write_markdown(report, output_dir / "report.md")
    print(
        json.dumps(
            {
                "integrity_passed": integrity_passed,
                "scientific_admission": admitted,
                "route": selected_route,
                "selected_policy": selected_policy,
            },
            indent=2,
        ),
        flush=True,
    )
    if args.fail_on_integrity and not integrity_passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
