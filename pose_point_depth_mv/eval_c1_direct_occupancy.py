#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch

from pose_point_depth_mv.c1_matched_budget import CORRUPTION_POLICY_NAMES
from pose_point_depth_mv.c1_direct_occupancy import (
    C1_DIRECT_OCCUPANCY_CHECKPOINT_VERSION,
    C1_DIRECT_OCCUPANCY_EVAL_VERSION,
    extract_direct_occupancy_objects,
    initialize_nested_models,
    model_logits,
    normalize_features,
    occupancy_metrics,
)
from pose_point_depth_mv.c1_occupancy import (
    C1MapTargetDataset,
    comparison_summary,
    file_sha256,
    load_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate C1.1b M0/M1/M2 with matched causal corruptions."
    )
    parser.add_argument("--c0_report", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--max_score_diff", type=float, default=1.0e-3)
    parser.add_argument("--bootstrap_samples", type=int, default=10000)
    parser.add_argument("--min_object_win_rate", type=float, default=0.65)
    parser.add_argument("--fail_on_integrity", action="store_true")
    return parser.parse_args()


def _comparison(
    records: list[dict[str, Any]],
    field: str,
    *,
    bootstrap_samples: int,
) -> dict[str, Any]:
    return comparison_summary(
        [float(row["comparisons"][field]) for row in records],
        bootstrap_samples=int(bootstrap_samples),
    )


def _passed(row: dict[str, Any], min_win_rate: float) -> bool:
    return bool(
        row["object"]["mean"] > 0.0
        and row["object"]["median"] > 0.0
        and row["object_win_rate"] >= float(min_win_rate)
        and row["object_bootstrap_95_ci"][0] > 0.0
    )


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# C1.1b Direct Local Occupancy Evaluation",
        "",
        f"- Split: `{report['split_name']}`",
        f"- Seed: `{report['training_seed']}`",
        f"- Decision: **{'PASS' if report['passed'] else 'FAIL'}**",
        "- Flow/decoder: `not loaded`",
        "",
        "## Checks",
        "",
    ]
    lines.extend(
        f"- `{name}`: `{value}`" for name, value in report["checks"].items()
    )
    lines.extend(["", "## Comparisons", "", "```json"])
    lines.append(json.dumps(report["comparisons"], indent=2))
    lines.extend(["```", "", "## View Groups", "", "```json"])
    lines.append(json.dumps(report["view_groups"], indent=2))
    lines.append("```")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@torch.no_grad()
def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    checkpoint_path = Path(args.checkpoint).resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if checkpoint.get("format") != C1_DIRECT_OCCUPANCY_CHECKPOINT_VERSION:
        raise ValueError("unexpected C1.1b checkpoint format")
    dataset = C1MapTargetDataset(args.c0_report)
    if int(dataset.report["training_seed"]) != int(checkpoint["training_seed"]):
        raise ValueError("C1.1b eval C0 seed differs from probe checkpoint")
    if dataset.report["checkpoint_sha256"] != checkpoint["source_c0_checkpoint_sha256"]:
        raise ValueError("C1.1b eval C0 checkpoint differs from training N3 seed")
    summary_path = Path(checkpoint["source_c1_0b_summary"]).resolve()
    if file_sha256(summary_path) != checkpoint["source_c1_0b_summary_sha256"]:
        raise ValueError("C1.0b summary changed after C1.1b training")
    summary = load_json(summary_path)
    n3 = load_json(summary["source_n3_report"])
    n3_rows = [
        row
        for row in n3["per_seed"]
        if int(row["seed"]) == int(checkpoint["training_seed"])
    ]
    if len(n3_rows) != 1:
        raise ValueError("C1.1b N3 seed binding is not unique")
    expected_report = (
        Path(n3_rows[0]["run_dir"])
        / f"c0_3_{dataset.report['split_name']}"
        / "report.json"
    )
    if dataset.report_path != expected_report.resolve():
        raise ValueError("C1.1b eval report is not the N3-bound split")

    objects, feature_metadata = extract_direct_occupancy_objects(
        dataset,
        policy=str(checkpoint["policy"]),
        target_mode=str(checkpoint["target_mode"]),
        device=device,
        max_samples=int(args.max_samples),
        max_score_diff=float(args.max_score_diff),
    )
    config = checkpoint["model_config"]
    models = initialize_nested_models(
        input_dim=int(config["input_dim"]),
        hidden_dim=int(config["hidden_dim"]),
        seed=int(checkpoint["training_seed"]),
    )
    if set(models) != set(checkpoint["model_states"]):
        raise ValueError("C1.1b checkpoint model set mismatch")
    for name, model in models.items():
        model.load_state_dict(checkpoint["model_states"][name], strict=True)
        model.to(device).eval()
    normalization = {
        name: value.to(device) for name, value in checkpoint["normalization"].items()
    }

    records: list[dict[str, Any]] = []
    for index, row in enumerate(objects):
        active = row["active"]
        target = row["target"][active].to(device)
        reliability = row["reliability"][active].to(device)
        branch_metrics: dict[str, Any] = {}
        for branch, features in row["candidates"].items():
            base, correspondence = normalize_features(
                features["base"][active].to(device),
                features["correspondence"][active].to(device),
                normalization,
            )
            logits = model_logits(
                models,
                reliability=reliability,
                base=base,
                correspondence=correspondence,
            )
            branch_metrics[branch] = {
                name: occupancy_metrics(value, target)
                for name, value in logits.items()
            }

        correct = branch_metrics["correct"]
        comparisons = {
            "M2_vs_M0_balanced_bce": (
                correct["M0_reliability"]["balanced_bce"]
                - correct["M2_plus_correspondence"]["balanced_bce"]
            ),
            "M2_vs_M1_balanced_bce": (
                correct["M1_view_geometry"]["balanced_bce"]
                - correct["M2_plus_correspondence"]["balanced_bce"]
            ),
        }
        for branch in row["candidates"]:
            if branch == "correct":
                continue
            comparisons[f"M2_correct_vs_{branch}_balanced_bce"] = (
                branch_metrics[branch]["M2_plus_correspondence"]["balanced_bce"]
                - correct["M2_plus_correspondence"]["balanced_bce"]
            )
            comparisons[f"M1_correct_vs_{branch}_balanced_bce"] = (
                branch_metrics[branch]["M1_view_geometry"]["balanced_bce"]
                - correct["M1_view_geometry"]["balanced_bce"]
            )
        records.append(
            {
                "uid": row["uid"],
                "object_uid": row["object_uid"],
                "views": row["views"],
                "metrics": branch_metrics,
                "comparisons": comparisons,
                "target_mapping_audit": row["target_mapping_audit"],
            }
        )
        print(
            f"[c1_1b_eval] {index + 1}/{len(objects)} uid={row['uid']} "
            f"views={row['views']} m2_vs_m1={comparisons['M2_vs_M1_balanced_bce']:.6e}",
            flush=True,
        )

    comparison_names = tuple(records[0]["comparisons"])
    comparisons = {
        name: _comparison(
            records, name, bootstrap_samples=int(args.bootstrap_samples)
        )
        for name in comparison_names
    }
    formal_names = (
        "M2_vs_M1_balanced_bce",
        *(
            f"M2_correct_vs_{branch}_balanced_bce"
            for branch in CORRUPTION_POLICY_NAMES
        ),
    )
    formal_checks = {
        name: _passed(comparisons[name], float(args.min_object_win_rate))
        for name in formal_names
    }
    view_groups: dict[str, Any] = {}
    for views in sorted({int(row["views"]) for row in records}):
        group = [row for row in records if int(row["views"]) == views]
        view_groups[str(views)] = {
            "object_count": len(group),
            "comparisons": {
                name: _comparison(
                    group, name, bootstrap_samples=int(args.bootstrap_samples)
                )
                for name in comparison_names
            },
        }
    two_view_checks: dict[str, bool] = {}
    if dataset.report["split_name"] in {"fresh48", "holdout"}:
        two_view = view_groups.get("2")
        two_view_checks["two_view_present"] = two_view is not None
        if two_view is not None:
            for name in formal_names:
                two_view_checks[f"two_view_{name}_nonnegative"] = (
                    two_view["comparisons"][name]["object"]["mean"] >= 0.0
                )

    integrity_checks = {
        "source_checkpoint_format_valid": True,
        "source_c0_report_bound_to_n3": True,
        "source_c1_0b_summary_immutable": True,
        "feature_protocol_matches_training": (
            feature_metadata["input_dim"] == checkpoint["feature_metadata"]["input_dim"]
            and feature_metadata["spatial_tolerance"]
            == checkpoint["feature_metadata"]["spatial_tolerance"]
        ),
        "all_target_mapping_roundtrips_pass": all(
            row["target_mapping_audit"]["passed"] for row in records
        ),
        "all_outputs_finite": all(
            math.isfinite(float(value))
            for row in records
            for branch in row["metrics"].values()
            for model in branch.values()
            for value in model.values()
        ),
        "flow_not_loaded": True,
        "decoder_not_loaded": True,
        "target_not_used_as_input": True,
    }
    checks = {**integrity_checks, **formal_checks, **two_view_checks}
    integrity_passed = all(integrity_checks.values())
    scientific_passed = all(formal_checks.values()) and all(two_view_checks.values())
    report = {
        "format": C1_DIRECT_OCCUPANCY_EVAL_VERSION,
        "stage": "C1.1b direct local occupancy nested causal evaluation",
        "passed": integrity_passed and scientific_passed,
        "integrity_passed": integrity_passed,
        "scientific_passed": scientific_passed,
        "split_name": dataset.report["split_name"],
        "training_seed": int(checkpoint["training_seed"]),
        "object_count": len(records),
        "target_mode": checkpoint["target_mode"],
        "policy": checkpoint["policy"],
        "source_checkpoint": str(checkpoint_path),
        "source_checkpoint_sha256": file_sha256(checkpoint_path),
        "source_c0_report": str(dataset.report_path),
        "source_c0_checkpoint_sha256": dataset.report["checkpoint_sha256"],
        "source_c1_0b_summary": str(summary_path),
        "training_protocol_hash": checkpoint["training_protocol_hash"],
        "feature_metadata": feature_metadata,
        "decision_thresholds": {
            "min_object_win_rate": float(args.min_object_win_rate),
            "mean_positive": True,
            "median_positive": True,
            "bootstrap_ci_lower_positive": True,
            "fresh_holdout_two_view_mean_nonnegative": True,
        },
        "checks": checks,
        "comparisons": comparisons,
        "view_groups": view_groups,
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
                "passed": report["passed"],
                "integrity_passed": integrity_passed,
                "scientific_passed": scientific_passed,
                "formal_checks": formal_checks,
                "two_view_checks": two_view_checks,
            },
            indent=2,
        ),
        flush=True,
    )
    if args.fail_on_integrity and not integrity_passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
