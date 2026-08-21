#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, median
from typing import Any

from pose_point_depth_mv.correspondence_head import VOXEL_SELFCAL_VERSION
from pose_point_depth_mv.eval_local_target_probe import object_balanced


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Strict multi-seed summary for C0.1 voxel self-calibration."
    )
    parser.add_argument("--run_dirs", nargs="+", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--train_subdir", default="voxel_selfcal_train16_t0")
    parser.add_argument("--fresh_subdir", default="voxel_selfcal_fresh48_t0")
    parser.add_argument("--expected_seeds", default="42,43,44")
    parser.add_argument("--fail_on_decision", action="store_true")
    return parser.parse_args()


def load_report(path: Path, split: str) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("format") != VOXEL_SELFCAL_VERSION:
        raise ValueError(f"unexpected report format: {path}")
    if report.get("split_name") != split:
        raise ValueError(f"report split mismatch: {path}")
    gate = report.get("gate_protocol", {})
    if gate.get("scope") != "per_voxel" or gate.get(
        "uses_object_score_as_gate"
    ) is not False:
        raise ValueError(f"report does not use a voxel-only gate: {path}")
    return report


def protocol_signature(report: dict[str, Any]) -> dict[str, Any]:
    args = report["args"]
    threshold_names = (
        "threshold",
        "min_voxel_positive_ratio",
        "min_per_object_positive_ratio",
        "min_object_local_pass_rate",
        "min_heldout_gate_positive_ratio",
        "min_spatial_control_object_win_rate",
        "min_spatial_control_gate_positive_ratio",
        "min_spatial_std",
        "max_permutation_diff",
    )
    return {
        "format": report["format"],
        "cache_config_hash": report["cache_config_hash"],
        "protocol_hash": report["protocol_hash"],
        "checkpoint_step": report["checkpoint_step"],
        "checkpoint_format": report["checkpoint_format"],
        "head_version": report["head_version"],
        "head_metadata": report["head_metadata"],
        "evidence_version": report["evidence_version"],
        "evidence_schema_hash": report["evidence_schema_hash"],
        "model_architecture_hash": report["model_architecture_hash"],
        "training_controls": report["training_controls"],
        "heldout_control_names": report["heldout_control_names"],
        "gate_protocol": report["gate_protocol"],
        "hard_admitted_soft_weight_protocol": report.get(
            "hard_admitted_soft_weight_protocol"
        ),
        "continuous_soft_weight_protocol": report.get(
            "continuous_soft_weight_protocol"
        ),
        "spatial_control": {
            "name": report["spatial_control"]["name"],
            "definition": report["spatial_control"]["definition"],
        },
        "thresholds": {name: args[name] for name in threshold_names},
    }


def object_uids(report: dict[str, Any]) -> list[str]:
    return sorted(str(row["object_uid"]) for row in report["records"])


def object_metric(report: dict[str, Any], *keys: str) -> dict[str, Any]:
    value: Any = report
    for key in keys:
        value = value[key]
    return value["object"]


def recompute_raw_metrics(report: dict[str, Any]) -> dict[str, Any]:
    """Rebuild all deterministic C0.1 aggregates directly from records."""

    records = report["records"]
    samples = int(report["args"]["bootstrap_samples"])
    primary_fields = (
        "hard_margin_mean",
        "hard_margin_median",
        "hard_voxel_positive_ratio",
        "hard_margin_std",
        "hard_margin_iqr",
        "hard_margin_normalized_std",
        "gate_fraction_active",
        "gate_component_count",
        "gate_largest_component_fraction",
        "gate_boundary_fraction",
    )
    primary = {
        field: object_balanced(records, field, bootstrap_samples=samples)
        for field in primary_fields
    }
    local_object_pass_rate = (
        mean(
            float(row["hard_voxel_positive_ratio"])
            >= float(report["args"]["min_per_object_positive_ratio"])
            for row in records
        )
        if records
        else 0.0
    )
    heldout = {
        mode: {
            "margin": object_balanced(
                records, f"{mode}_margin_mean", bootstrap_samples=samples
            ),
            "voxel_positive_ratio": object_balanced(
                records, f"{mode}_voxel_positive_ratio", bootstrap_samples=samples
            ),
            "gate_positive_ratio": object_balanced(
                records, f"{mode}_gate_positive_ratio", bootstrap_samples=samples
            ),
        }
        for mode in report["heldout_control_names"]
    }
    spatial = {
        "margin": object_balanced(
            records, "spatial_margin_mean", bootstrap_samples=samples
        ),
        "voxel_positive_ratio": object_balanced(
            records, "spatial_voxel_positive_ratio", bootstrap_samples=samples
        ),
        "gate_positive_ratio": object_balanced(
            records, "spatial_gate_positive_ratio", bootstrap_samples=samples
        ),
    }
    hardest = {
        mode: object_balanced(
            records, f"hardest_{mode}_ratio", bootstrap_samples=samples
        )
        for mode in report["training_controls"]
    }
    return {
        "primary": primary,
        "local_object_pass_rate": float(local_object_pass_rate),
        "heldout_controls": heldout,
        "spatial_control": spatial,
        "hardest_training_control": hardest,
    }


def equal_without_bootstrap(expected: Any, actual: Any) -> bool:
    if isinstance(expected, dict) and isinstance(actual, dict):
        keys = set(expected) | set(actual)
        return all(
            key == "object_bootstrap_95_ci"
            or (
                key in expected
                and key in actual
                and equal_without_bootstrap(expected[key], actual[key])
            )
            for key in keys
        )
    if isinstance(expected, list) and isinstance(actual, list):
        return len(expected) == len(actual) and all(
            equal_without_bootstrap(left, right)
            for left, right in zip(expected, actual)
        )
    if isinstance(expected, (float, int)) and isinstance(actual, (float, int)):
        return math.isclose(float(expected), float(actual), rel_tol=1.0e-8, abs_tol=1.0e-8)
    return expected == actual


def raw_metrics_match_report(report: dict[str, Any], metrics: dict[str, Any]) -> bool:
    primary = report["primary"]
    primary_match = all(
        equal_without_bootstrap(metrics["primary"][source], primary[target])
        for source, target in (
            ("hard_margin_mean", "hard_margin_mean"),
            ("hard_margin_median", "hard_margin_median"),
            ("hard_voxel_positive_ratio", "voxel_positive_ratio"),
            ("hard_margin_std", "spatial_std"),
            ("hard_margin_iqr", "spatial_iqr"),
            ("hard_margin_normalized_std", "normalized_spatial_std"),
            ("gate_fraction_active", "gate_fraction_active"),
            ("gate_component_count", "gate_component_count"),
            ("gate_largest_component_fraction", "gate_largest_component_fraction"),
            ("gate_boundary_fraction", "gate_boundary_fraction"),
        )
    )
    return (
        primary_match
        and math.isclose(
            metrics["local_object_pass_rate"],
            float(primary["local_object_pass_rate"]),
            rel_tol=1.0e-8,
            abs_tol=1.0e-8,
        )
        and equal_without_bootstrap(
            metrics["heldout_controls"], report["heldout_controls"]
        )
        and equal_without_bootstrap(
            metrics["spatial_control"],
            {
                key: report["spatial_control"][key]
                for key in metrics["spatial_control"]
            },
        )
        and equal_without_bootstrap(
            metrics["hardest_training_control"], report["hardest_training_control"]
        )
    )


def recompute_report_decision(report: dict[str, Any], metrics: dict[str, Any]) -> bool:
    """Re-evaluate primary gates instead of trusting a stored PASS flag."""

    args = report["args"]
    primary = metrics["primary"]
    hard_mean = primary["hard_margin_mean"]
    hard_median = primary["hard_margin_median"]
    voxel_ratio = primary["hard_voxel_positive_ratio"]
    spatial_std = primary["hard_margin_std"]
    checks = [
        not report["failures"],
        int(report["sample_count"]) == len(report["records"]),
        int(report["object_count"]) == len(report["records"]),
        float(report["voxel_permutation_max_abs_diff"])
        <= float(args["max_permutation_diff"]),
        float(hard_mean["object"]["mean"]) > 0.0,
        float(hard_median["object"]["median"]) > 0.0,
        float(hard_mean["object_bootstrap_95_ci"][0]) > 0.0,
        float(voxel_ratio["object"]["mean"])
        >= float(args["min_voxel_positive_ratio"]),
        float(metrics["local_object_pass_rate"])
        >= float(args["min_object_local_pass_rate"]),
        float(spatial_std["object"]["mean"]) >= float(args["min_spatial_std"]),
    ]
    for value in metrics["heldout_controls"].values():
        margin = value["margin"]
        heldout_voxel_ratio = value["voxel_positive_ratio"]
        gate_ratio = value["gate_positive_ratio"]
        checks.extend(
            (
                float(margin["object"]["mean"]) > 0.0,
                float(margin["object"]["median"]) > 0.0,
                float(margin["object_bootstrap_95_ci"][0]) > 0.0,
                float(heldout_voxel_ratio["object"]["mean"])
                >= float(args["min_voxel_positive_ratio"]),
                float(gate_ratio["object"]["mean"])
                >= float(args["min_heldout_gate_positive_ratio"]),
            )
        )
    spatial = metrics["spatial_control"]
    checks.extend(
        (
            float(spatial["margin"]["object"]["mean"]) > 0.0,
            float(spatial["margin"]["object"]["median"]) > 0.0,
            float(spatial["margin"]["object_bootstrap_95_ci"][0]) > 0.0,
            float(spatial["margin"]["object_win_rate"])
            >= float(args["min_spatial_control_object_win_rate"]),
            float(spatial["voxel_positive_ratio"]["object"]["mean"])
            >= float(args["min_voxel_positive_ratio"]),
            float(spatial["gate_positive_ratio"]["object"]["mean"])
            >= float(args["min_spatial_control_gate_positive_ratio"]),
        )
    )
    return all(checks)


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# C0.1 Voxel Self-calibration Multi-seed Summary",
        "",
        f"- Decision: **{'PASS' if report['passed'] else 'FAIL'}**",
        f"- Seeds: `{report['seeds']}`",
        "- Gate scope: `per_voxel`",
        "- Object/sample scores are reporting metrics only and are never broadcast as a gate.",
        "",
        "## Checks",
        "",
    ]
    lines.extend(f"- `{key}`: `{value}`" for key, value in report["checks"].items())
    lines.extend(
        [
            "",
            "## Per-seed Results",
            "",
            "| Seed | Train | Fresh | Fresh hard mean | Fresh voxel positive | Fresh local pass |",
            "| ---: | :---: | :---: | ---: | ---: | ---: |",
        ]
    )
    for row in report["per_seed"]:
        lines.append(
            "| {seed} | {train} | {fresh} | {hard:.6f} | {voxel:.4f} | {local:.4f} |".format(
                seed=row["seed"],
                train="PASS" if row["train_passed"] else "FAIL",
                fresh="PASS" if row["fresh_passed"] else "FAIL",
                hard=row["fresh_hard_margin_mean"],
                voxel=row["fresh_voxel_positive_ratio"],
                local=row["fresh_local_object_pass_rate"],
            )
        )
    lines.extend(
        [
            "",
            "## Aggregate",
            "",
            "```json",
            json.dumps(report["aggregate"], indent=2),
            "```",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    expected_seeds = sorted(
        int(value.strip()) for value in args.expected_seeds.split(",") if value.strip()
    )
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for run_dir_value in args.run_dirs:
        run_dir = Path(run_dir_value)
        train = load_report(run_dir / args.train_subdir / "report.json", "train16")
        fresh = load_report(run_dir / args.fresh_subdir / "report.json", "fresh48")
        if int(train["training_seed"]) != int(fresh["training_seed"]):
            raise ValueError(f"train/fresh seed mismatch: {run_dir}")
        pairs.append((train, fresh))

    seeds = sorted(int(train["training_seed"]) for train, _ in pairs)
    signatures = [protocol_signature(report) for pair in pairs for report in pair]
    first_signature = signatures[0] if signatures else None
    protocol_consistent = bool(first_signature) and all(
        signature == first_signature for signature in signatures[1:]
    )
    train_uid_sets = [object_uids(train) for train, _ in pairs]
    fresh_uid_sets = [object_uids(fresh) for _, fresh in pairs]
    train_uids_consistent = bool(train_uid_sets) and all(
        values == train_uid_sets[0] for values in train_uid_sets[1:]
    )
    fresh_uids_consistent = bool(fresh_uid_sets) and all(
        values == fresh_uid_sets[0] for values in fresh_uid_sets[1:]
    )
    split_disjoint = all(
        not (set(train_uids) & set(fresh_uids))
        for train_uids, fresh_uids in zip(train_uid_sets, fresh_uid_sets)
    )
    raw_metrics = {
        id(report): recompute_raw_metrics(report)
        for pair in pairs
        for report in pair
    }

    per_seed: list[dict[str, Any]] = []
    for train, fresh in sorted(pairs, key=lambda pair: int(pair[0]["training_seed"])):
        hard = object_metric(fresh, "primary", "hard_margin_mean")
        voxel = object_metric(fresh, "primary", "voxel_positive_ratio")
        per_seed.append(
            {
                "seed": int(train["training_seed"]),
                "train_passed": bool(train["passed"]),
                "fresh_passed": bool(fresh["passed"]),
                "fresh_hard_margin_mean": float(hard["mean"]),
                "fresh_hard_margin_median": float(hard["median"]),
                "fresh_voxel_positive_ratio": float(voxel["mean"]),
                "fresh_local_object_pass_rate": float(
                    fresh["primary"]["local_object_pass_rate"]
                ),
                "fresh_heldout": {
                    mode: {
                        "margin_mean": float(value["margin"]["object"]["mean"]),
                        "voxel_positive_ratio": float(
                            value["voxel_positive_ratio"]["object"]["mean"]
                        ),
                        "gate_positive_ratio": float(
                            value["gate_positive_ratio"]["object"]["mean"]
                        ),
                    }
                    for mode, value in fresh["heldout_controls"].items()
                },
            }
        )

    checks = {
        "expected_seed_set": seeds == expected_seeds,
        "seeds_unique": len(seeds) == len(set(seeds)),
        "protocol_consistent": protocol_consistent,
        "train_object_uids_consistent": train_uids_consistent,
        "fresh_object_uids_consistent": fresh_uids_consistent,
        "train_fresh_object_disjoint": split_disjoint,
        "all_train_reports_pass": bool(pairs)
        and all(bool(train["passed"]) for train, _ in pairs),
        "all_fresh_reports_pass": bool(pairs)
        and all(bool(fresh["passed"]) for _, fresh in pairs),
        "report_decisions_match_checks": bool(pairs)
        and all(
            bool(report["passed"])
            == all(bool(value) for value in report["checks"].values())
            for pair in pairs
            for report in pair
        ),
        "report_counts_match_records": bool(pairs)
        and all(
            int(report["sample_count"]) == len(report["records"])
            and int(report["object_count"])
            == len({str(row["object_uid"]) for row in report["records"]})
            for pair in pairs
            for report in pair
        ),
        "all_report_metrics_recomputed_pass": bool(pairs)
        and all(
            recompute_report_decision(report, raw_metrics[id(report)])
            for pair in pairs
            for report in pair
        ),
        "raw_record_metrics_match_report": bool(pairs)
        and all(
            raw_metrics_match_report(report, raw_metrics[id(report)])
            for pair in pairs
            for report in pair
        ),
        "all_reports_use_voxel_only_gate": bool(pairs)
        and all(
            report["gate_protocol"]["uses_object_score_as_gate"] is False
            for pair in pairs
            for report in pair
        ),
    }
    aggregate_keys = (
        "fresh_hard_margin_mean",
        "fresh_hard_margin_median",
        "fresh_voxel_positive_ratio",
        "fresh_local_object_pass_rate",
    )
    aggregate = {
        key: {
            "mean_across_seeds": mean(float(row[key]) for row in per_seed),
            "median_across_seeds": median(float(row[key]) for row in per_seed),
            "min_across_seeds": min(float(row[key]) for row in per_seed),
            "max_across_seeds": max(float(row[key]) for row in per_seed),
        }
        for key in aggregate_keys
    } if per_seed else {}
    report = {
        "format": "pose_point_depth_mv.voxel_selfcal_multiseed.v2",
        "stage": "C0.1 strict voxel self-calibration multi-seed gate",
        "passed": all(checks.values()),
        "seeds": seeds,
        "expected_seeds": expected_seeds,
        "run_dirs": [str(Path(value).resolve()) for value in args.run_dirs],
        "protocol_signature": first_signature,
        "checks": checks,
        "per_seed": per_seed,
        "aggregate": aggregate,
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    write_markdown(report, output_dir / "report.md")
    print(json.dumps(report, indent=2))
    if args.fail_on_decision and not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
