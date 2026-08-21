#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pose_point_depth_mv.c1_occupancy import (
    C1_ENRICHMENT_REPORT_VERSION,
    C1_ENRICHMENT_SUMMARY_VERSION,
    PRIMARY_POLICIES,
    load_json,
    protocol_signature,
    summarize,
)


SPLITS = ("train16", "fresh48", "holdout")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize preregistered C1.0 reports across N3 seeds/splits."
    )
    parser.add_argument("--n3_report", required=True)
    parser.add_argument("--report_dirs", nargs="+", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--fail_on_decision", action="store_true")
    return parser.parse_args()


def report_index(reports: list[dict[str, Any]]) -> dict[tuple[int, str], dict[str, Any]]:
    output: dict[tuple[int, str], dict[str, Any]] = {}
    for report in reports:
        key = (int(report["training_seed"]), str(report["split_name"]))
        if key in output:
            raise ValueError(f"duplicate C1.0 report for {key}")
        output[key] = report
    return output


def uid_target_signature(report: dict[str, Any]) -> dict[str, tuple[int, int, str, str]]:
    return {
        str(row["uid"]): (
            int(row["targets"]["exact"]["target_count"]),
            int(row["targets"]["surface_r1"]["target_count"]),
            str(row["target_mapping_audit"]["exact_target_mask_sha256"]),
            str(row["target_mapping_audit"]["surface_r1_mask_sha256"]),
        )
        for row in report["records"]
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# C1.0 Multi-seed Target Enrichment Summary",
        "",
        f"- Decision: **{'PASS' if report['passed'] else 'FAIL'}**",
        f"- Seeds: `{report['seeds']}`",
        f"- Admitted policy: `{report['admitted_policy']}`",
        "- Flow LoRA: `disabled`",
        "",
        "## Checks",
        "",
    ]
    lines.extend(f"- `{key}`: `{value}`" for key, value in report["checks"].items())
    lines.extend(["", "## Per Seed", "", "```json"])
    lines.append(json.dumps(report["per_seed"], indent=2))
    lines.extend(["```", "", "## Aggregate", "", "```json"])
    lines.append(json.dumps(report["aggregate"], indent=2))
    lines.append("```")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    n3_path = Path(args.n3_report).resolve()
    n3 = load_json(n3_path)
    if n3.get("passed") is not True:
        raise ValueError("C1.0 summary requires a passed N3 report")
    report_paths = [Path(path) / "report.json" for path in args.report_dirs]
    reports = [load_json(path) for path in report_paths]
    if any(report.get("format") != C1_ENRICHMENT_REPORT_VERSION for report in reports):
        raise ValueError("unexpected C1.0 report format")
    indexed = report_index(reports)
    seeds = sorted(int(seed) for seed in n3["seeds"])
    expected_keys = {(seed, split) for seed in seeds for split in SPLITS}
    actual_keys = set(indexed)

    n3_by_seed = {int(row["seed"]): row for row in n3["per_seed"]}
    source_bindings_ok = True
    for (seed, split), report in indexed.items():
        n3_row = n3_by_seed.get(seed, {})
        expected_c0 = Path(n3_row.get("run_dir", "/__missing__")) / f"c0_3_{split}" / "report.json"
        source_bindings_ok = source_bindings_ok and (
            Path(report["source_c0_report"]).resolve() == expected_c0.resolve()
            and str(report["source_c0_checkpoint_sha256"])
            == str(n3_row.get("checkpoint_sha256", ""))
        )

    signatures = [protocol_signature(report) for report in reports]
    protocol_consistent = bool(signatures) and all(
        signature == signatures[0] for signature in signatures[1:]
    )
    split_uid_signatures: dict[str, list[dict[str, tuple[int, int]]]] = {
        split: [uid_target_signature(indexed[(seed, split)]) for seed in seeds]
        for split in SPLITS
        if all((seed, split) in indexed for seed in seeds)
    }
    same_split_targets_across_seeds = all(
        rows and all(row == rows[0] for row in rows[1:])
        for rows in split_uid_signatures.values()
    )
    split_uid_sets = {
        split: set(next(iter(rows)).keys()) if rows else set()
        for split, rows in split_uid_signatures.items()
    }
    split_disjoint = all(
        split_uid_sets[left].isdisjoint(split_uid_sets[right])
        for index, left in enumerate(SPLITS)
        for right in SPLITS[index + 1 :]
        if left in split_uid_sets and right in split_uid_sets
    )

    common_policies: list[str] = []
    for policy in ("hard_admitted", "continuous"):
        if all(
            indexed[(seed, split)]["policy_decisions"].get(policy, {}).get("passed")
            is True
            for seed in seeds
            for split in SPLITS
            if (seed, split) in indexed
        ):
            common_policies.append(policy)
    admitted_policy = common_policies[0] if common_policies else None

    per_seed: list[dict[str, Any]] = []
    for seed in seeds:
        seed_row: dict[str, Any] = {"seed": seed, "splits": {}}
        for split in SPLITS:
            report = indexed.get((seed, split))
            if report is None:
                continue
            formal = report["formal_target_mode"]
            split_row: dict[str, Any] = {"passed": bool(report["passed"])}
            for policy in PRIMARY_POLICIES:
                comparison = report["comparisons"][formal][policy]
                split_row[policy] = {
                    name: {
                        "mean": float(value["object"]["mean"]),
                        "median": float(value["object"]["median"]),
                        "win_rate": float(value["object_win_rate"]),
                        "ci_low": float(value["object_bootstrap_95_ci"][0]),
                    }
                    for name, value in comparison.items()
                }
            seed_row["splits"][split] = split_row
        per_seed.append(seed_row)

    aggregate: dict[str, Any] = {}
    for split in SPLITS:
        aggregate[split] = {}
        for policy in PRIMARY_POLICIES:
            aggregate[split][policy] = {}
            for control in (
                "vs_active_only",
                "vs_reliability_only",
                "vs_spatial_permutation",
                "vs_hardest_corruption",
            ):
                values = [
                    float(
                        indexed[(seed, split)]["comparisons"][
                            indexed[(seed, split)]["formal_target_mode"]
                        ][policy][control]["object"]["mean"]
                    )
                    for seed in seeds
                    if (seed, split) in indexed
                ]
                aggregate[split][policy][control] = summarize(values)

    checks = {
        "n3_passed": n3.get("passed") is True,
        "expected_seed_split_grid": actual_keys == expected_keys,
        "source_c0_reports_and_checkpoints_bound_to_n3": source_bindings_ok,
        "protocol_consistent": protocol_consistent,
        "same_split_target_labels_identical_across_seeds": same_split_targets_across_seeds,
        "all_target_mapping_roundtrips_pass": all(
            row["target_mapping_audit"]["passed"] is True
            for report in reports
            for row in report["records"]
        ),
        "train_fresh_holdout_object_disjoint": split_disjoint,
        "all_integrity_checks_passed": all(
            all(bool(value) for value in report["checks"].values())
            for report in reports
        ),
        "common_policy_passes_all_train_fresh_and_holdout": admitted_policy is not None,
        "two_view_protection_in_all_fresh_and_holdout": admitted_policy is not None
        and all(
            indexed[(seed, split)]["policy_decisions"][admitted_policy]["checks"].get(
                "two_view_vs_reliability_nonnegative"
            )
            is True
            and indexed[(seed, split)]["policy_decisions"][admitted_policy]["checks"].get(
                "two_view_vs_permutation_nonnegative"
            )
            is True
            and indexed[(seed, split)]["policy_decisions"][admitted_policy]["checks"].get(
                "two_view_vs_corruption_nonnegative"
            )
            is True
            for seed in seeds
            for split in ("fresh48", "holdout")
        ),
        "flow_lora_disabled": True,
    }
    passed = all(checks.values())
    summary = {
        "format": C1_ENRICHMENT_SUMMARY_VERSION,
        "stage": "C1.0 multi-seed target occupancy enrichment admission",
        "passed": passed,
        "seeds": seeds,
        "splits": list(SPLITS),
        "admitted_policy": admitted_policy,
        "common_passing_policies": common_policies,
        "source_n3_report": str(n3_path),
        "source_reports": [str(path.resolve()) for path in report_paths],
        "protocol_signature": signatures[0] if signatures else None,
        "checks": checks,
        "per_seed": per_seed,
        "aggregate": aggregate,
        "flow_lora_enabled": False,
        "allowed_next_stage": (
            "C1.1 monotone local occupancy calibration probe"
            if passed
            else "stop before C1.1 and inspect target alignment"
        ),
    }
    (output_dir / "report.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    write_markdown(summary, output_dir / "report.md")
    print(json.dumps({"passed": passed, "checks": checks, "admitted_policy": admitted_policy}, indent=2))
    if args.fail_on_decision and not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
