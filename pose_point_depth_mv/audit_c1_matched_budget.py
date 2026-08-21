#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from pose_point_depth_mv.c1_matched_budget import (
    C1_MATCHED_BUDGET_REPORT_VERSION,
    C1_MATCHED_BUDGET_VERSION,
    CORRUPTION_POLICY_NAMES,
    MATCHED_CONTROLS,
    MATCHED_POLICIES,
    budget_key,
    matched_budget_metrics,
    matched_candidate_weights,
    parse_budget_fractions,
)
from pose_point_depth_mv.c1_occupancy import (
    C1MapTargetDataset,
    SEMANTIC_NAMES,
    TARGET_MODES,
    stable_seed,
    summarize,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "C1.0b no-training matched-budget occupancy audit. Every control "
            "uses the exact correct-policy weight histogram and fixed support."
        )
    )
    parser.add_argument("--c0_report", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--budget_fractions", default="0.05,0.10,0.20")
    parser.add_argument("--policies", default="hard_admitted,continuous")
    parser.add_argument("--bootstrap_samples", type=int, default=10000)
    parser.add_argument("--min_object_win_rate", type=float, default=0.65)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--fail_on_integrity", action="store_true")
    return parser.parse_args()


def _metric_value(row: dict[str, Any], path: tuple[str, ...]) -> float:
    value: Any = row
    for key in path:
        value = value[key]
    return float(value)


def _comparison(
    records: list[dict[str, Any]],
    *,
    policy: str,
    target_mode: str,
    control: str,
    path: tuple[str, ...],
    bootstrap_samples: int,
) -> dict[str, Any]:
    differences = [
        _metric_value(
            row["targets"][target_mode][policy]["candidates"]["correct"], path
        )
        - _metric_value(
            row["targets"][target_mode][policy]["candidates"][control], path
        )
        for row in records
    ]
    values = np.asarray(differences, dtype=np.float64)
    rng = np.random.default_rng(
        stable_seed(
            C1_MATCHED_BUDGET_VERSION,
            policy,
            target_mode,
            control,
            "/".join(path),
            len(records),
        )
    )
    sampled = values[
        rng.integers(
            0,
            len(values),
            size=(int(bootstrap_samples), len(values)),
            endpoint=False,
        )
    ].mean(axis=1)
    result = {
        "object": summarize(differences),
        "object_win_rate": float(np.mean(values > 0.0)),
        "object_bootstrap_95_ci": [
            float(np.quantile(sampled, 0.025)),
            float(np.quantile(sampled, 0.975)),
        ],
    }
    result["metric_path"] = list(path)
    return result


def _comparison_passed(row: dict[str, Any], min_win_rate: float) -> bool:
    return bool(
        row["object"]["mean"] > 0.0
        and row["object"]["median"] > 0.0
        and row["object_win_rate"] >= float(min_win_rate)
        and row["object_bootstrap_95_ci"][0] > 0.0
    )


def _semantic_aggregate(
    records: list[dict[str, Any]],
    *,
    target_mode: str,
    policy: str,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for semantic in SEMANTIC_NAMES.values():
        output[semantic] = {}
        for candidate in ("correct", *MATCHED_CONTROLS):
            rows = [
                row["targets"][target_mode][policy]["candidates"][candidate][
                    "semantics"
                ][semantic]
                for row in records
            ]
            output[semantic][candidate] = {
                field: summarize(value[field] for value in rows)
                for field in (
                    "voxel_count",
                    "score_mass_fraction",
                    "target_rate",
                    "weighted_target_rate",
                )
            }
    return output


def _build_comparisons(
    records: list[dict[str, Any]],
    *,
    policies: tuple[str, ...],
    fractions: tuple[float, ...],
    bootstrap_samples: int,
    min_win_rate: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    comparisons: dict[str, Any] = {}
    decisions: dict[str, Any] = {}
    for target_mode in TARGET_MODES:
        comparisons[target_mode] = {}
        decisions[target_mode] = {}
        for policy in policies:
            comparisons[target_mode][policy] = {}
            decisions[target_mode][policy] = {}
            for control in MATCHED_CONTROLS:
                metrics: dict[str, Any] = {
                    "histogram_weighted_target_rate": _comparison(
                        records,
                        policy=policy,
                        target_mode=target_mode,
                        control=control,
                        path=("weighted_target_rate",),
                        bootstrap_samples=bootstrap_samples,
                    ),
                    "histogram_weighted_target_coverage": _comparison(
                        records,
                        policy=policy,
                        target_mode=target_mode,
                        control=control,
                        path=("weighted_target_coverage",),
                        bootstrap_samples=bootstrap_samples,
                    ),
                }
                for fraction in fractions:
                    key = budget_key(fraction)
                    metrics[f"{key}_target_rate"] = _comparison(
                        records,
                        policy=policy,
                        target_mode=target_mode,
                        control=control,
                        path=("budgets", key, "target_rate"),
                        bootstrap_samples=bootstrap_samples,
                    )
                    metrics[f"{key}_target_coverage"] = _comparison(
                        records,
                        policy=policy,
                        target_mode=target_mode,
                        control=control,
                        path=("budgets", key, "target_coverage"),
                        bootstrap_samples=bootstrap_samples,
                    )
                comparisons[target_mode][policy][control] = metrics
                primary_names = (
                    "histogram_weighted_target_rate",
                    *(f"{budget_key(fraction)}_target_rate" for fraction in fractions),
                )
                checks = {
                    name: _comparison_passed(metrics[name], min_win_rate)
                    for name in primary_names
                }
                decisions[target_mode][policy][control] = {
                    "passed": all(checks.values()),
                    "checks": checks,
                }
    return comparisons, decisions


def _route_for_policy(decisions: dict[str, Any], policy: str) -> str:
    corruption_controls = tuple(CORRUPTION_POLICY_NAMES)
    corruptions_pass = all(
        decisions[target_mode][policy][control]["passed"]
        for target_mode in TARGET_MODES
        for control in corruption_controls
    )
    reliability_pass = all(
        decisions[target_mode][policy]["reliability"]["passed"]
        for target_mode in TARGET_MODES
    )
    spatial_pass = all(
        decisions[target_mode][policy]["spatial_permutation"]["passed"]
        for target_mode in TARGET_MODES
    )
    if corruptions_pass and reliability_pass and spatial_pass:
        return "restricted_surface_occupancy_gate_candidate"
    if corruptions_pass and spatial_pass:
        return "auxiliary_correspondence_feature_only"
    return "stop_target_occupancy_direction"


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# C1.0b Matched-budget Occupancy Audit",
        "",
        f"- Split: `{report['split_name']}`",
        f"- Seed: `{report['training_seed']}`",
        f"- Integrity: **{'PASS' if report['integrity_passed'] else 'FAIL'}**",
        f"- Route: `{report['decision']['preferred_route']}`",
        "- Training: `none`",
        "- Flow/decoder: `not loaded`",
        "",
        "## Checks",
        "",
    ]
    lines.extend(
        f"- `{name}`: `{value}`" for name, value in report["checks"].items()
    )
    lines.extend(["", "## Policy Routes", "", "```json"])
    lines.append(json.dumps(report["decision"], indent=2))
    lines.extend(["```", "", "## Comparisons", "", "```json"])
    lines.append(json.dumps(report["comparisons"], indent=2))
    lines.extend(["```", "", "## View Groups", "", "```json"])
    lines.append(json.dumps(report["view_groups"], indent=2))
    lines.extend(["```", "", "## Semantic Strata", "", "```json"])
    lines.append(json.dumps(report["semantic_strata"], indent=2))
    lines.append("```")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    fractions = parse_budget_fractions(args.budget_fractions)
    policies = tuple(
        item.strip() for item in args.policies.split(",") if item.strip()
    )
    if not policies or any(policy not in MATCHED_POLICIES for policy in policies):
        raise ValueError(f"invalid C1.0b policies: {policies}")
    if not 0.0 <= float(args.min_object_win_rate) <= 1.0:
        raise ValueError("min object win rate must be in [0,1]")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    dataset = C1MapTargetDataset(args.c0_report)

    records: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    count = len(dataset) if int(args.max_samples) <= 0 else min(
        len(dataset), int(args.max_samples)
    )
    for index in range(count):
        uid = str(dataset.records[index].get("uid", index))
        try:
            item = dataset[index]
            payload = item["map"]
            active = payload["active_mask"].bool()
            semantic_label = payload["audit_maps"]["depth_semantic_label"].long()
            target_rows: dict[str, Any] = {
                mode: {} for mode in TARGET_MODES
            }
            invariant_rows: dict[str, Any] = {}
            for policy in policies:
                weights, invariants = matched_candidate_weights(
                    payload, policy=policy, uid=item["uid"]
                )
                invariant_rows[policy] = invariants
                for target_mode in TARGET_MODES:
                    target_rows[target_mode][policy] = {
                        "candidates": {
                            name: matched_budget_metrics(
                                weight,
                                item["targets"][target_mode],
                                active,
                                semantic_label,
                                uid=item["uid"],
                                name=f"{policy}:{name}",
                                fractions=fractions,
                            )
                            for name, weight in weights.items()
                        }
                    }
            invariant_passed = all(
                row["histogram_equal"]
                and row["support_equal"]
                and row["inactive_zero"]
                and row["mass_abs_diff"]
                <= 1.0e-6 * max(1.0, float(invariant_rows[policy]["reference_mass"]))
                for policy, policy_row in invariant_rows.items()
                for row in policy_row["candidates"].values()
            )
            records.append(
                {
                    "uid": item["uid"],
                    "object_uid": item["object_uid"],
                    "views": item["views"],
                    "target_mapping_audit": item["target_mapping_audit"],
                    "matched_budget_invariants": invariant_rows,
                    "matched_budget_invariants_passed": invariant_passed,
                    "targets": target_rows,
                }
            )
            exact = target_rows["exact"][policies[0]]["candidates"]
            print(
                f"[c1_0b] {index + 1}/{count} uid={item['uid']} "
                f"views={item['views']} "
                f"correct={exact['correct']['weighted_target_rate']:.6f} "
                f"reliability={exact['reliability']['weighted_target_rate']:.6f}",
                flush=True,
            )
        except Exception as error:  # Preserve all sample failures in the report.
            failures.append({"uid": uid, "error": repr(error)})
            print(f"[c1_0b] FAILED uid={uid}: {error!r}", flush=True)
    if not records:
        raise RuntimeError("C1.0b produced no valid records")

    comparisons, comparison_decisions = _build_comparisons(
        records,
        policies=policies,
        fractions=fractions,
        bootstrap_samples=int(args.bootstrap_samples),
        min_win_rate=float(args.min_object_win_rate),
    )
    policy_routes = {
        policy: _route_for_policy(comparison_decisions, policy)
        for policy in policies
    }
    route_priority = (
        "restricted_surface_occupancy_gate_candidate",
        "auxiliary_correspondence_feature_only",
        "stop_target_occupancy_direction",
    )
    preferred_route = next(
        route
        for route in route_priority
        if route in set(policy_routes.values())
    )

    view_groups: dict[str, Any] = {}
    for views in sorted({int(row["views"]) for row in records}):
        group = [row for row in records if int(row["views"]) == views]
        group_comparisons, _ = _build_comparisons(
            group,
            policies=policies,
            fractions=fractions,
            bootstrap_samples=int(args.bootstrap_samples),
            min_win_rate=float(args.min_object_win_rate),
        )
        view_groups[str(views)] = {
            "object_count": len(group),
            "comparisons": group_comparisons,
        }
    semantic_strata = {
        target_mode: {
            policy: _semantic_aggregate(
                records, target_mode=target_mode, policy=policy
            )
            for policy in policies
        }
        for target_mode in TARGET_MODES
    }

    checks = {
        "source_c0_passed": dataset.report.get("passed") is True,
        "source_c0_gaussian3": dataset.report.get(
            "evaluation_spatial_tolerance"
        )
        == "gaussian3",
        "all_samples_loaded": not failures and len(records) == count,
        "all_histograms_support_and_mass_matched": all(
            row["matched_budget_invariants_passed"] for row in records
        ),
        "all_target_mapping_roundtrips_pass": all(
            row["target_mapping_audit"]["passed"] for row in records
        ),
        "targets_never_enter_ranking": True,
        "flow_not_loaded": True,
        "decoder_not_loaded": True,
    }
    integrity_passed = all(checks.values())
    protocol = {
        "version": C1_MATCHED_BUDGET_VERSION,
        "policies": list(policies),
        "controls": list(MATCHED_CONTROLS),
        "corruption_sources": CORRUPTION_POLICY_NAMES,
        "target_modes": list(TARGET_MODES),
        "budget_fractions": list(fractions),
        "histogram_matching": (
            "sort exact correct-policy active weights and assign that identical "
            "histogram by each target-independent candidate ranking"
        ),
        "tie_break": "deterministic UID/name hash; target-independent",
        "decision_metric": (
            "weighted target rate plus top-K target rate; positive mean/median, "
            "object win threshold, and positive object bootstrap CI"
        ),
        "min_object_win_rate": float(args.min_object_win_rate),
    }
    protocol_hash = hashlib.sha256(
        json.dumps(protocol, sort_keys=True).encode("utf-8")
    ).hexdigest()
    report = {
        "format": C1_MATCHED_BUDGET_REPORT_VERSION,
        "stage": "C1.0b no-training matched-budget occupancy audit",
        "passed": integrity_passed
        and preferred_route != "stop_target_occupancy_direction",
        "integrity_passed": integrity_passed,
        "split_name": dataset.report["split_name"],
        "training_seed": int(dataset.report["training_seed"]),
        "object_count": len(records),
        "source_c0_report": str(dataset.report_path),
        "source_c0_checkpoint": dataset.report["checkpoint"],
        "source_c0_checkpoint_sha256": dataset.report["checkpoint_sha256"],
        "source_cache_manifest": dataset.report["cache_manifest"],
        "source_cache_config_hash": dataset.report["cache_config_hash"],
        "protocol": protocol,
        "protocol_hash": protocol_hash,
        "checks": checks,
        "decision": {
            "preferred_route": preferred_route,
            "policy_routes": policy_routes,
            "comparison_decisions": comparison_decisions,
        },
        "comparisons": comparisons,
        "view_groups": view_groups,
        "semantic_strata": semantic_strata,
        "failures": failures,
        "records": records,
        "flow_loaded": False,
        "flow_lora_enabled": False,
        "decoder_loaded": False,
        "target_used_as_input": False,
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    _write_markdown(report, output_dir / "report.md")
    print(
        json.dumps(
            {
                "integrity_passed": integrity_passed,
                "scientific_passed": report["passed"],
                "split": report["split_name"],
                "seed": report["training_seed"],
                "route": preferred_route,
            },
            indent=2,
        ),
        flush=True,
    )
    if args.fail_on_integrity and not integrity_passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
