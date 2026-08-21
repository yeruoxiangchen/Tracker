#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, median
from typing import Any

from pose_point_depth_mv.summarize_voxel_selfcal_multiseed import (
    load_report,
    object_uids,
    protocol_signature,
    raw_metrics_match_report,
    recompute_raw_metrics,
    recompute_report_decision,
)


NEIGHBORHOOD_MULTISEED_VERSION = (
    "pose_point_depth_mv.neighborhood_voxel_selfcal_multiseed.v2"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Strict C0.3 multi-seed gate over train, fresh, and untouched "
            "object-disjoint holdout reports."
        )
    )
    parser.add_argument("--run_dirs", nargs="+", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--train_subdir", default="c0_3_train16")
    parser.add_argument("--fresh_subdir", default="c0_3_fresh48")
    parser.add_argument("--holdout_subdir", default="c0_3_holdout")
    parser.add_argument("--expected_seeds", default="42,43,44")
    parser.add_argument("--fail_on_decision", action="store_true")
    return parser.parse_args()


def split_metrics(report: dict[str, Any]) -> dict[str, float | bool]:
    primary = report["primary"]
    return {
        "passed": bool(report["passed"]),
        "hard_margin_mean": float(
            primary["hard_margin_mean"]["object"]["mean"]
        ),
        "hard_margin_ci_low": float(
            primary["hard_margin_mean"]["object_bootstrap_95_ci"][0]
        ),
        "voxel_positive_ratio": float(
            primary["voxel_positive_ratio"]["object"]["mean"]
        ),
        "local_object_pass_rate": float(primary["local_object_pass_rate"]),
        "hard_admitted_soft_weight": float(
            primary["hard_admitted_soft_weight"]["object"]["mean"]
        ),
        "continuous_soft_weight": float(
            primary["continuous_soft_weight"]["object"]["mean"]
        ),
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# C0.3 Neighborhood-aware Multi-seed and Holdout Gate",
        "",
        f"- Decision: **{'PASS' if report['passed'] else 'FAIL'}**",
        f"- Seeds: `{report['seeds']}`",
        "- Required protocol: matched `gaussian3`, symmetric fixed-correct support",
        "- Flow LoRA: `disabled`",
        "",
        "## Per-seed",
        "",
        "| Seed | Train | Fresh | Holdout | Fresh voxel | Holdout voxel |",
        "| ---: | :---: | :---: | :---: | ---: | ---: |",
    ]
    for row in report["per_seed"]:
        lines.append(
            "| {seed} | {train} | {fresh} | {holdout} | {fresh_voxel:.4f} | "
            "{holdout_voxel:.4f} |".format(
                seed=row["seed"],
                train="PASS" if row["train"]["passed"] else "FAIL",
                fresh="PASS" if row["fresh"]["passed"] else "FAIL",
                holdout="PASS" if row["holdout"]["passed"] else "FAIL",
                fresh_voxel=row["fresh"]["voxel_positive_ratio"],
                holdout_voxel=row["holdout"]["voxel_positive_ratio"],
            )
        )
    lines.extend(["", "## Checks", ""])
    lines.extend(f"- `{key}`: `{value}`" for key, value in report["checks"].items())
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
        int(value.strip())
        for value in str(args.expected_seeds).split(",")
        if value.strip()
    )

    bundles: list[dict[str, Any]] = []
    for value in args.run_dirs:
        run_dir = Path(value)
        train = load_report(run_dir / args.train_subdir / "report.json", "train16")
        fresh = load_report(run_dir / args.fresh_subdir / "report.json", "fresh48")
        holdout = load_report(
            run_dir / args.holdout_subdir / "report.json", "holdout"
        )
        seeds = {
            int(train["training_seed"]),
            int(fresh["training_seed"]),
            int(holdout["training_seed"]),
        }
        if len(seeds) != 1:
            raise ValueError(f"split seed mismatch: {run_dir}")
        checkpoint_paths = {
            str(report.get("checkpoint", ""))
            for report in (train, fresh, holdout)
        }
        checkpoint_hashes = {
            str(report.get("checkpoint_sha256", ""))
            for report in (train, fresh, holdout)
        }
        bundles.append(
            {
                "run_dir": str(run_dir.resolve()),
                "seed": seeds.pop(),
                "train": train,
                "fresh": fresh,
                "holdout": holdout,
                "checkpoint_paths": checkpoint_paths,
                "checkpoint_hashes": checkpoint_hashes,
            }
        )

    bundles.sort(key=lambda row: int(row["seed"]))
    seeds = [int(row["seed"]) for row in bundles]
    reports = [
        row[split]
        for row in bundles
        for split in ("train", "fresh", "holdout")
    ]
    signatures = [protocol_signature(report) for report in reports]
    protocol_consistent = bool(signatures) and all(
        value == signatures[0] for value in signatures[1:]
    )
    uid_sets = {
        split: [object_uids(row[split]) for row in bundles]
        for split in ("train", "fresh", "holdout")
    }
    uid_consistent = {
        split: bool(values) and all(value == values[0] for value in values[1:])
        for split, values in uid_sets.items()
    }
    split_disjoint = all(
        not (set(row["train_uids"]) & set(row["fresh_uids"]))
        and not (set(row["train_uids"]) & set(row["holdout_uids"]))
        and not (set(row["fresh_uids"]) & set(row["holdout_uids"]))
        for row in (
            {
                "train_uids": uid_sets["train"][index],
                "fresh_uids": uid_sets["fresh"][index],
                "holdout_uids": uid_sets["holdout"][index],
            }
            for index in range(len(bundles))
        )
    )
    raw_metrics = {id(report): recompute_raw_metrics(report) for report in reports}
    matched_neighborhood = bool(reports) and all(
        report.get("evaluation_spatial_tolerance") == "gaussian3"
        and report.get("training_spatial_tolerance") == "gaussian3"
        and report.get("spatial_tolerance_matches_training") is True
        and report.get("gate_protocol", {}).get("evaluation_matches_training") is True
        for report in reports
    )
    weight_protocols_consistent = bool(reports) and all(
        report.get("hard_admitted_soft_weight_protocol")
        == reports[0].get("hard_admitted_soft_weight_protocol")
        and report.get("continuous_soft_weight_protocol")
        == reports[0].get("continuous_soft_weight_protocol")
        and report.get("hard_admitted_soft_weight_protocol", {}).get(
            "flow_lora_enabled"
        )
        is False
        and report.get("hard_admitted_soft_weight_protocol", {}).get(
            "formal_n3_gate"
        )
        is True
        and report.get("continuous_soft_weight_protocol", {}).get(
            "c1_ablation_only"
        )
        is True
        for report in reports
    )
    same_checkpoint_paths = bool(bundles) and all(
        len(row["checkpoint_paths"]) == 1 and "" not in row["checkpoint_paths"]
        for row in bundles
    )
    same_checkpoint_hashes = bool(bundles) and all(
        len(row["checkpoint_hashes"]) == 1 and "" not in row["checkpoint_hashes"]
        for row in bundles
    )

    checks = {
        "expected_seed_set": seeds == expected_seeds,
        "seeds_unique": len(seeds) == len(set(seeds)),
        "protocol_consistent": protocol_consistent,
        "train_uids_consistent": uid_consistent["train"],
        "fresh_uids_consistent": uid_consistent["fresh"],
        "holdout_uids_consistent": uid_consistent["holdout"],
        "all_splits_object_disjoint": split_disjoint,
        "matched_gaussian3_training_and_evaluation": matched_neighborhood,
        "weight_protocols_consistent": weight_protocols_consistent,
        "same_seed_splits_share_checkpoint_path": same_checkpoint_paths,
        "same_seed_splits_share_checkpoint_sha256": same_checkpoint_hashes,
        "model_head_and_evidence_identity_consistent": protocol_consistent,
        "all_reports_pass": bool(reports)
        and all(bool(report["passed"]) for report in reports),
        "all_report_metrics_recomputed_pass": bool(reports)
        and all(
            recompute_report_decision(report, raw_metrics[id(report)])
            for report in reports
        ),
        "raw_record_metrics_match_reports": bool(reports)
        and all(
            raw_metrics_match_report(report, raw_metrics[id(report)])
            for report in reports
        ),
        "all_maps_requested": bool(reports)
        and all(bool(report["args"].get("save_maps")) for report in reports),
    }

    per_seed = [
        {
            "seed": int(row["seed"]),
            "run_dir": row["run_dir"],
            "checkpoint": next(iter(row["checkpoint_paths"])),
            "checkpoint_sha256": next(iter(row["checkpoint_hashes"])),
            "train": split_metrics(row["train"]),
            "fresh": split_metrics(row["fresh"]),
            "holdout": split_metrics(row["holdout"]),
        }
        for row in bundles
    ]
    aggregate: dict[str, Any] = {}
    for split in ("train", "fresh", "holdout"):
        aggregate[split] = {}
        for field in (
            "hard_margin_mean",
            "hard_margin_ci_low",
            "voxel_positive_ratio",
            "local_object_pass_rate",
            "hard_admitted_soft_weight",
            "continuous_soft_weight",
        ):
            values = [float(row[split][field]) for row in per_seed]
            aggregate[split][field] = {
                "mean_across_seeds": mean(values),
                "median_across_seeds": median(values),
                "min_across_seeds": min(values),
                "max_across_seeds": max(values),
            }

    report = {
        "format": NEIGHBORHOOD_MULTISEED_VERSION,
        "stage": "N3 C0.3 multi-seed plus untouched holdout gate",
        "passed": all(checks.values()),
        "seeds": seeds,
        "expected_seeds": expected_seeds,
        "run_dirs": [row["run_dir"] for row in bundles],
        "protocol_signature": signatures[0] if signatures else None,
        "flow_lora_enabled": False,
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
