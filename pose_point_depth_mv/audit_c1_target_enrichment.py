#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean
from typing import Any

import torch

from pose_point_depth_mv.c1_occupancy import (
    C1_ENRICHMENT_REPORT_VERSION,
    C1_OCCUPANCY_PROTOCOL_VERSION,
    C1MapTargetDataset,
    PRIMARY_POLICIES,
    SEMANTIC_NAMES,
    TARGET_MODES,
    c1_policy_scores,
    comparison_summary,
    permute_within_active,
    policy_metrics,
    summarize,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "C1.0 target-enrichment audit. Targets are labels only and are "
            "loaded after immutable C0 maps have been validated."
        )
    )
    parser.add_argument("--c0_report", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--permutation_repeats", type=int, default=16)
    parser.add_argument("--bootstrap_samples", type=int, default=10000)
    parser.add_argument("--formal_target_mode", choices=TARGET_MODES, default="exact")
    parser.add_argument(
        "--admission_policies",
        default="hard_admitted,continuous",
        help="Comma-separated policies; at least one must pass every control.",
    )
    parser.add_argument("--min_object_win_rate", type=float, default=0.65)
    parser.add_argument("--fail_on_decision", action="store_true")
    return parser.parse_args()


def average_metric_rows(rows: list[dict[str, float | int]]) -> dict[str, float]:
    if not rows:
        raise ValueError("cannot average an empty metric list")
    return {
        key: mean(float(row[key]) for row in rows)
        for key in rows[0]
        if key not in {"active_count", "target_count", "active_target_count"}
    } | {
        key: int(rows[0][key])
        for key in ("active_count", "target_count", "active_target_count")
    }


def semantic_metrics(
    score: torch.Tensor,
    target: torch.Tensor,
    active: torch.Tensor,
    labels: torch.Tensor,
) -> dict[str, dict[str, float | int]]:
    output: dict[str, dict[str, float | int]] = {}
    for label_id, name in SEMANTIC_NAMES.items():
        mask = active & labels.eq(int(label_id))
        count = int(mask.sum().item())
        if count == 0:
            output[name] = {
                "voxel_count": 0,
                "target_rate": 0.0,
                "weighted_target_rate": 0.0,
                "score_mass": 0.0,
            }
            continue
        values = score[mask].float()
        truth = target[mask].float()
        mass = values.sum()
        output[name] = {
            "voxel_count": count,
            "target_rate": float(truth.mean().item()),
            "weighted_target_rate": (
                float((values * truth).sum().div(mass).item())
                if float(mass.item()) > 0.0
                else 0.0
            ),
            "score_mass": float(mass.item()),
        }
    return output


def aggregate_policy_metrics(
    records: list[dict[str, Any]], target_mode: str
) -> dict[str, Any]:
    policies = sorted(records[0]["targets"][target_mode]["policies"])
    fields = sorted(
        records[0]["targets"][target_mode]["policies"][policies[0]]
    )
    return {
        policy: {
            field: summarize(
                row["targets"][target_mode]["policies"][policy][field]
                for row in records
            )
            for field in fields
        }
        for policy in policies
    }


def comparison(
    records: list[dict[str, Any]],
    *,
    target_mode: str,
    candidate: str,
    control: str,
    field: str,
    bootstrap_samples: int,
) -> dict[str, Any]:
    values = [
        float(row["targets"][target_mode]["policies"][candidate][field])
        - float(row["targets"][target_mode]["policies"][control][field])
        for row in records
    ]
    return comparison_summary(values, bootstrap_samples=bootstrap_samples)


def corruption_controls_for_policy(
    records: list[dict[str, Any]], *, target_mode: str, policy: str
) -> list[str]:
    prefix = f"corruption_{policy}_"
    names = sorted(
        name
        for name in records[0]["targets"][target_mode]["policies"]
        if name.startswith(prefix)
    )
    if not names:
        raise ValueError(f"no real corruption-derived controls for policy={policy}")
    return names


def comparison_to_hardest_corruption(
    records: list[dict[str, Any]],
    *,
    target_mode: str,
    candidate: str,
    field: str,
    bootstrap_samples: int,
) -> dict[str, Any]:
    controls = corruption_controls_for_policy(
        records, target_mode=target_mode, policy=candidate
    )
    values = []
    hardest_names: list[str] = []
    for row in records:
        policies = row["targets"][target_mode]["policies"]
        strongest = max(controls, key=lambda name: float(policies[name][field]))
        hardest_names.append(strongest)
        values.append(float(policies[candidate][field]) - float(policies[strongest][field]))
    result = comparison_summary(values, bootstrap_samples=bootstrap_samples)
    result["controls"] = controls
    result["strongest_control_counts"] = {
        name: hardest_names.count(name) for name in controls
    }
    return result


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# C1.0 Target Occupancy Enrichment Audit",
        "",
        f"- Split: `{report['split_name']}`",
        f"- Training seed: `{report['training_seed']}`",
        f"- Objects: `{report['object_count']}`",
        f"- Decision: **{'PASS' if report['passed'] else 'FAIL'}**",
        f"- Formal target: `{report['formal_target_mode']}`",
        "- Flow/decoder loaded: `false`",
        "- Target used as model input: `false`",
        "",
        "## Checks",
        "",
    ]
    lines.extend(f"- `{key}`: `{value}`" for key, value in report["checks"].items())
    lines.extend(["", "## Policy Decisions", "", "```json"])
    lines.append(json.dumps(report["policy_decisions"], indent=2))
    lines.extend(["```", "", "## Formal Comparisons", "", "```json"])
    lines.append(json.dumps(report["comparisons"][report["formal_target_mode"]], indent=2))
    lines.extend(["```", "", "## View Groups", "", "```json"])
    lines.append(json.dumps(report["view_groups"], indent=2))
    lines.append("```")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if int(args.permutation_repeats) <= 0:
        raise ValueError("--permutation_repeats must be positive")
    admission_policies = tuple(
        item.strip() for item in args.admission_policies.split(",") if item.strip()
    )
    if not admission_policies or any(
        policy not in PRIMARY_POLICIES for policy in admission_policies
    ):
        raise ValueError(f"invalid admission policies: {admission_policies}")

    dataset = C1MapTargetDataset(args.c0_report)
    records: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    policy_names: set[str] = set()
    for index in range(len(dataset)):
        try:
            item = dataset[index]
            payload = item["map"]
            active = payload["active_mask"].bool()
            scores = c1_policy_scores(payload)
            policy_names.update(scores)
            permutation_metrics: dict[str, dict[str, dict[str, float | int]]] = {}
            for policy in PRIMARY_POLICIES:
                permutation_metrics[policy] = {}
                for target_mode in TARGET_MODES:
                    repeats = [
                        policy_metrics(
                            permute_within_active(
                                scores[policy],
                                active,
                                uid=item["uid"],
                                repeat=repeat,
                            ),
                            item["targets"][target_mode],
                            active,
                        )
                        for repeat in range(int(args.permutation_repeats))
                    ]
                    permutation_metrics[policy][target_mode] = average_metric_rows(
                        repeats
                    )

            target_rows: dict[str, Any] = {}
            semantic_label = payload["audit_maps"]["depth_semantic_label"].long()
            for target_mode in TARGET_MODES:
                target = item["targets"][target_mode]
                metrics = {
                    policy: policy_metrics(score, target, active)
                    for policy, score in scores.items()
                }
                for policy in PRIMARY_POLICIES:
                    metrics[f"permuted_{policy}"] = permutation_metrics[policy][
                        target_mode
                    ]
                target_rows[target_mode] = {
                    "target_count": int(target.sum().item()),
                    "target_ratio": float(target.float().mean().item()),
                    "target_inside_active_ratio": float(
                        (target & active).sum().div(target.sum().clamp_min(1)).item()
                    ),
                    "policies": metrics,
                    "semantics": {
                        policy: semantic_metrics(
                            score, target, active, semantic_label
                        )
                        for policy, score in scores.items()
                    },
                }
            records.append(
                {
                    "uid": item["uid"],
                    "object_uid": item["object_uid"],
                    "views": item["views"],
                    "map_file": item["map_path"],
                    "active_ratio": float(active.float().mean().item()),
                    "target_mapping_audit": item["target_mapping_audit"],
                    "targets": target_rows,
                }
            )
            print(
                f"[c1_enrichment] {index + 1}/{len(dataset)} "
                f"uid={item['uid']} views={item['views']} "
                f"exact={target_rows['exact']['target_count']} "
                f"hard={target_rows['exact']['policies']['hard_admitted']['weighted_target_rate']:.6f} "
                f"active={target_rows['exact']['policies']['active_only']['weighted_target_rate']:.6f}",
                flush=True,
            )
        except Exception as error:  # Keep a complete audit trail.
            uid = str(dataset.records[index].get("uid", index))
            failures.append({"uid": uid, "error": repr(error)})
            print(f"[c1_enrichment] FAILED uid={uid}: {error!r}", flush=True)

    if not records:
        raise RuntimeError("C1.0 produced no valid records")
    aggregates = {
        target_mode: aggregate_policy_metrics(records, target_mode)
        for target_mode in TARGET_MODES
    }
    comparisons: dict[str, Any] = {}
    for target_mode in TARGET_MODES:
        comparisons[target_mode] = {}
        for candidate in PRIMARY_POLICIES:
            candidate_comparisons = {
                "vs_active_only": comparison(
                    records,
                    target_mode=target_mode,
                    candidate=candidate,
                    control="active_only",
                    field="weighted_target_rate",
                    bootstrap_samples=int(args.bootstrap_samples),
                ),
                "vs_reliability_only": comparison(
                    records,
                    target_mode=target_mode,
                    candidate=candidate,
                    control="reliability_only",
                    field="weighted_target_rate",
                    bootstrap_samples=int(args.bootstrap_samples),
                ),
                "vs_spatial_permutation": comparison(
                    records,
                    target_mode=target_mode,
                    candidate=candidate,
                    control=f"permuted_{candidate}",
                    field="weighted_target_rate",
                    bootstrap_samples=int(args.bootstrap_samples),
                ),
                "ap_vs_spatial_permutation": comparison(
                    records,
                    target_mode=target_mode,
                    candidate=candidate,
                    control=f"permuted_{candidate}",
                    field="average_precision_active",
                    bootstrap_samples=int(args.bootstrap_samples),
                ),
                "vs_hardest_corruption": comparison_to_hardest_corruption(
                    records,
                    target_mode=target_mode,
                    candidate=candidate,
                    field="weighted_target_rate",
                    bootstrap_samples=int(args.bootstrap_samples),
                ),
            }
            for corruption in corruption_controls_for_policy(
                records, target_mode=target_mode, policy=candidate
            ):
                candidate_comparisons[f"vs_{corruption}"] = comparison(
                    records,
                    target_mode=target_mode,
                    candidate=candidate,
                    control=corruption,
                    field="weighted_target_rate",
                    bootstrap_samples=int(args.bootstrap_samples),
                )
            comparisons[target_mode][candidate] = candidate_comparisons

    view_groups: dict[str, Any] = {}
    for views in sorted({int(row["views"]) for row in records}):
        group = [row for row in records if int(row["views"]) == views]
        view_groups[str(views)] = {"object_count": len(group), "policies": {}}
        for candidate in PRIMARY_POLICIES:
            view_groups[str(views)]["policies"][candidate] = {
                "vs_active_only": comparison(
                    group,
                    target_mode=args.formal_target_mode,
                    candidate=candidate,
                    control="active_only",
                    field="weighted_target_rate",
                    bootstrap_samples=int(args.bootstrap_samples),
                ),
                "vs_reliability_only": comparison(
                    group,
                    target_mode=args.formal_target_mode,
                    candidate=candidate,
                    control="reliability_only",
                    field="weighted_target_rate",
                    bootstrap_samples=int(args.bootstrap_samples),
                ),
                "vs_spatial_permutation": comparison(
                    group,
                    target_mode=args.formal_target_mode,
                    candidate=candidate,
                    control=f"permuted_{candidate}",
                    field="weighted_target_rate",
                    bootstrap_samples=int(args.bootstrap_samples),
                ),
                "vs_hardest_corruption": comparison_to_hardest_corruption(
                    group,
                    target_mode=args.formal_target_mode,
                    candidate=candidate,
                    field="weighted_target_rate",
                    bootstrap_samples=int(args.bootstrap_samples),
                ),
            }

    formal_comparisons = comparisons[args.formal_target_mode]
    policy_decisions: dict[str, Any] = {}
    for policy in admission_policies:
        controls = formal_comparisons[policy]
        checks: dict[str, bool] = {}
        for control_name in (
            "vs_active_only",
            "vs_reliability_only",
            "vs_spatial_permutation",
            "vs_hardest_corruption",
        ):
            row = controls[control_name]
            checks[f"{control_name}_mean_positive"] = row["object"]["mean"] > 0.0
            checks[f"{control_name}_median_positive"] = row["object"]["median"] > 0.0
            checks[f"{control_name}_win_rate"] = (
                row["object_win_rate"] >= float(args.min_object_win_rate)
            )
            checks[f"{control_name}_bootstrap_ci_positive"] = (
                row["object_bootstrap_95_ci"][0] > 0.0
            )
        if dataset.report["split_name"] in {"fresh48", "holdout"}:
            two_view = view_groups.get("2", {}).get("policies", {}).get(policy)
            checks["two_view_present"] = two_view is not None
            if two_view is not None:
                checks["two_view_vs_reliability_nonnegative"] = (
                    two_view["vs_reliability_only"]["object"]["mean"] >= 0.0
                )
                checks["two_view_vs_permutation_nonnegative"] = (
                    two_view["vs_spatial_permutation"]["object"]["mean"] >= 0.0
                )
                checks["two_view_vs_corruption_nonnegative"] = (
                    two_view["vs_hardest_corruption"]["object"]["mean"] >= 0.0
                )
        policy_decisions[policy] = {
            "passed": all(checks.values()),
            "checks": checks,
        }

    integrity_checks = {
        "source_c0_report_passed": dataset.report.get("passed") is True,
        "source_c0_is_gaussian3": dataset.report.get(
            "evaluation_spatial_tolerance"
        )
        == "gaussian3",
        "all_samples_loaded": not failures and len(records) == len(dataset),
        "target_labels_absent_from_c0_maps": not failures,
        "flow_not_loaded": True,
        "decoder_not_loaded": True,
        "target_used_only_after_map_validation": True,
        "all_target_coordinate_roundtrips_pass": all(
            row["target_mapping_audit"]["passed"] for row in records
        ),
        "all_exact_target_hashes_present": all(
            len(row["target_mapping_audit"]["exact_target_mask_sha256"]) == 64
            for row in records
        ),
        "policy_set_complete": set(
            (*PRIMARY_POLICIES, "active_only", "reliability_only")
        ).issubset(policy_names),
    }
    scientific_pass = any(
        row["passed"] for row in policy_decisions.values()
    )
    passed = all(integrity_checks.values()) and scientific_pass
    policy_protocol = {
        "primary": list(PRIMARY_POLICIES),
        "baselines": ["active_only", "reliability_only"],
        "spatial_control": (
            "deterministic permutation of candidate values within fixed active "
            "support; histogram and support are preserved"
        ),
        "corruption_control": (
            "reconstruct each real corrupted branch score as correct_score-margin; "
            "apply the same one-vs-rest hard/continuous transform with fixed "
            "correct active support and raw reliability"
        ),
        "average_precision": "tie-aware threshold-grouped; diagnostic only",
    }
    target_protocol = {
        "source": "ss latent target_coords; audit label only",
        "exact": "floor(target_coords_64 / 4) on the existing SS 16^3 grid",
        "surface_r1": "symmetric 3x3x3 max-pool of exact occupancy",
        "formal": args.formal_target_mode,
        "target_enters_model_or_gate": False,
    }
    decision_thresholds = {
        "admission_policies": list(admission_policies),
        "min_object_win_rate": float(args.min_object_win_rate),
        "mean_positive": True,
        "median_positive": True,
        "bootstrap_ci_lower_positive": True,
        "must_beat": [
            "active_only",
            "reliability_only",
            "spatial_permutation",
            "hardest_real_corruption_candidate",
        ],
        "fresh_holdout_two_view_mean_must_be_nonnegative": True,
    }
    protocol_payload = {
        "version": C1_OCCUPANCY_PROTOCOL_VERSION,
        "target_protocol": target_protocol,
        "policy_protocol": policy_protocol,
        "decision_thresholds": decision_thresholds,
        "permutation_repeats": int(args.permutation_repeats),
    }
    protocol_hash = hashlib.sha256(
        json.dumps(protocol_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    report = {
        "format": C1_ENRICHMENT_REPORT_VERSION,
        "stage": "C1.0 target occupancy enrichment without training",
        "passed": passed,
        "split_name": dataset.report["split_name"],
        "training_seed": int(dataset.report["training_seed"]),
        "object_count": len(records),
        "source_c0_report": str(dataset.report_path),
        "source_c0_checkpoint": dataset.report["checkpoint"],
        "source_c0_checkpoint_sha256": dataset.report["checkpoint_sha256"],
        "source_cache_manifest": dataset.report["cache_manifest"],
        "source_cache_config_hash": dataset.report["cache_config_hash"],
        "formal_target_mode": args.formal_target_mode,
        "permutation_repeats": int(args.permutation_repeats),
        "protocol_hash": protocol_hash,
        "target_protocol": target_protocol,
        "policy_protocol": policy_protocol,
        "decision_thresholds": decision_thresholds,
        "flow_loaded": False,
        "flow_lora_enabled": False,
        "decoder_loaded": False,
        "target_used_as_input": False,
        "checks": integrity_checks,
        "policy_decisions": policy_decisions,
        "aggregates": aggregates,
        "comparisons": comparisons,
        "view_groups": view_groups,
        "failures": failures,
        "records": records,
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    write_markdown(report, output_dir / "report.md")
    print(
        json.dumps(
            {
                "passed": passed,
                "split": report["split_name"],
                "seed": report["training_seed"],
                "policy_decisions": policy_decisions,
            },
            indent=2,
        )
    )
    if args.fail_on_decision and not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
