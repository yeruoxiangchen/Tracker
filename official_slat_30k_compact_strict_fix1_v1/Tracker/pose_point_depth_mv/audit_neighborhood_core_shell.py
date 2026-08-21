#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any

import torch

from pose_point_depth_mv.correspondence_head import VOXEL_SELFCAL_VERSION
from pose_point_depth_mv.eval_local_target_probe import object_balanced
from pose_point_depth_mv.summarize_voxel_selfcal_multiseed import (
    raw_metrics_match_report,
    recompute_raw_metrics,
    recompute_report_decision,
)


CORE_SHELL_AUDIT_VERSION = "pose_point_depth_mv.neighborhood_core_shell.v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Decompose a symmetric neighborhood C0 result into the original "
            "exact-support core and newly admitted neighbor shell."
        )
    )
    parser.add_argument("--exact_dir", required=True)
    parser.add_argument("--neighborhood_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--bootstrap_samples", type=int, default=10000)
    parser.add_argument("--min_core_positive_ratio", type=float, default=0.60)
    parser.add_argument("--min_shell_positive_ratio", type=float, default=0.50)
    parser.add_argument("--min_core_object_pass_rate", type=float, default=0.65)
    parser.add_argument("--min_shell_object_pass_rate", type=float, default=0.65)
    parser.add_argument("--fail_on_decision", action="store_true")
    return parser.parse_args()


def partition_support(
    exact_active: torch.Tensor,
    neighborhood_active: torch.Tensor,
) -> dict[str, torch.Tensor]:
    exact = exact_active.detach().bool()
    neighborhood = neighborhood_active.detach().bool()
    if exact.shape != neighborhood.shape:
        raise ValueError("exact and neighborhood support shapes differ")
    return {
        "core": exact & neighborhood,
        "shell": neighborhood & ~exact,
        "lost": exact & ~neighborhood,
        "union": exact | neighborhood,
    }


def selected_stats(values: torch.Tensor, mask: torch.Tensor) -> dict[str, float | int]:
    selected = values.detach().float()[mask.detach().bool()]
    if selected.numel() == 0:
        return {
            "count": 0,
            "mean": 0.0,
            "median": 0.0,
            "positive_ratio": 0.0,
        }
    return {
        "count": int(selected.numel()),
        "mean": float(selected.mean().item()),
        "median": float(selected.median().item()),
        "positive_ratio": float(selected.gt(0).float().mean().item()),
    }


def load_report(directory: Path) -> dict[str, Any]:
    path = directory / "report.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("format") != VOXEL_SELFCAL_VERSION:
        raise ValueError(f"unexpected voxel report format: {path}")
    return report


def load_maps(directory: Path, report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    maps_dir = directory / "voxel_maps"
    if not maps_dir.is_dir():
        raise FileNotFoundError(f"voxel maps are missing: {maps_dir}")
    output: dict[str, dict[str, Any]] = {}
    for row in report["records"]:
        uid = str(row["uid"])
        path = maps_dir / f"{uid}.pt"
        payload = torch.load(path, map_location="cpu")
        if payload.get("format") != VOXEL_SELFCAL_VERSION:
            raise ValueError(f"unexpected voxel map format: {path}")
        if str(payload.get("uid")) != uid:
            raise ValueError(f"voxel map UID mismatch: {path}")
        output[uid] = payload
    return output


def aggregate_metric(
    records: list[dict[str, Any]],
    field: str,
    *,
    bootstrap_samples: int,
) -> dict[str, Any]:
    return object_balanced(
        records,
        field,
        bootstrap_samples=int(bootstrap_samples),
    )


def write_markdown(report: dict[str, Any], path: Path) -> None:
    primary = report["primary"]
    lines = [
        "# N0 Neighborhood Core/Shell Matched-support Audit",
        "",
        f"- Decision: **{'PASS' if report['passed'] else 'FAIL'}**",
        f"- Exact report: `{report['exact_report']}`",
        f"- Neighborhood report: `{report['neighborhood_report']}`",
        f"- Objects: `{report['object_count']}`",
        "",
        "## Primary",
        "",
        "| Region | Margin mean | Positive ratio | Object pass rate |",
        "| --- | ---: | ---: | ---: |",
    ]
    for label, prefix in (
        ("Exact core", "exact_core"),
        ("Neighborhood core", "neighborhood_core"),
        ("Neighborhood shell", "neighborhood_shell"),
        ("Neighborhood full", "neighborhood_full"),
    ):
        lines.append(
            "| {label} | {margin:.6f} | {positive:.4f} | {passed:.4f} |".format(
                label=label,
                margin=primary[f"{prefix}_margin_mean"]["object"]["mean"],
                positive=primary[f"{prefix}_positive_ratio"]["object"]["mean"],
                passed=primary[f"{prefix}_object_pass_rate"],
            )
        )
    lines.extend(
        [
            "",
            "## Support",
            "",
            "```json",
            json.dumps(report["support"], indent=2),
            "```",
            "",
            "## Checks",
            "",
        ]
    )
    lines.extend(f"- `{name}`: `{value}`" for name, value in report["checks"].items())
    lines.extend(
        [
            "",
            "## Controls by Region",
            "",
            "```json",
            json.dumps(report["controls"], indent=2),
            "```",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    exact_dir = Path(args.exact_dir)
    neighborhood_dir = Path(args.neighborhood_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)

    exact_report = load_report(exact_dir)
    neighborhood_report = load_report(neighborhood_dir)
    neighborhood_raw_metrics = recompute_raw_metrics(neighborhood_report)
    neighborhood_recomputed_passed = recompute_report_decision(
        neighborhood_report, neighborhood_raw_metrics
    )
    neighborhood_metrics_match = raw_metrics_match_report(
        neighborhood_report, neighborhood_raw_metrics
    )
    exact_mode = exact_report.get("gate_protocol", {}).get("spatial_tolerance")
    neighborhood_mode = neighborhood_report.get("gate_protocol", {}).get(
        "spatial_tolerance"
    )
    if exact_mode != "exact":
        raise ValueError(f"expected exact baseline, got {exact_mode!r}")
    if neighborhood_mode != "gaussian3":
        raise ValueError(
            f"expected gaussian3 neighborhood report, got {neighborhood_mode!r}"
        )
    protocol_fields = (
        "split_name",
        "training_seed",
        "checkpoint",
        "checkpoint_step",
        "cache_config_hash",
        "training_controls",
        "heldout_control_names",
    )
    for field in protocol_fields:
        if exact_report.get(field) != neighborhood_report.get(field):
            raise ValueError(f"exact/neighborhood protocol mismatch: {field}")

    exact_maps = load_maps(exact_dir, exact_report)
    neighborhood_maps = load_maps(neighborhood_dir, neighborhood_report)
    if set(exact_maps) != set(neighborhood_maps):
        raise ValueError("exact/neighborhood map UID sets differ")

    records: list[dict[str, Any]] = []
    control_names = tuple(
        list(exact_report["training_controls"])
        + list(exact_report["heldout_control_names"])
    )
    for uid in sorted(exact_maps):
        exact = exact_maps[uid]
        neighborhood = neighborhood_maps[uid]
        if str(exact["object_uid"]) != str(neighborhood["object_uid"]):
            raise ValueError(f"object UID mismatch for {uid}")
        parts = partition_support(exact["active_mask"], neighborhood["active_mask"])
        core = parts["core"]
        shell = parts["shell"]
        lost = parts["lost"]
        exact_count = int(exact["active_mask"].bool().sum().item())
        neighborhood_count = int(neighborhood["active_mask"].bool().sum().item())
        if exact_count <= 0 or neighborhood_count <= 0:
            raise ValueError(f"empty active support for {uid}")

        exact_core = selected_stats(exact["hard_margin"], core)
        neighborhood_core = selected_stats(neighborhood["hard_margin"], core)
        neighborhood_shell = selected_stats(neighborhood["hard_margin"], shell)
        neighborhood_full = selected_stats(
            neighborhood["hard_margin"], neighborhood["active_mask"]
        )
        row: dict[str, Any] = {
            "uid": uid,
            "object_uid": str(exact["object_uid"]),
            "views": int(exact["views"]),
            "exact_active_count": exact_count,
            "neighborhood_active_count": neighborhood_count,
            "core_count": int(core.sum().item()),
            "shell_count": int(shell.sum().item()),
            "lost_count": int(lost.sum().item()),
            "support_expansion_fraction_exact": float(shell.sum().item())
            / float(exact_count),
            "exact_core_margin_mean": float(exact_core["mean"]),
            "exact_core_positive_ratio": float(exact_core["positive_ratio"]),
            "neighborhood_core_margin_mean": float(neighborhood_core["mean"]),
            "neighborhood_core_positive_ratio": float(
                neighborhood_core["positive_ratio"]
            ),
            "neighborhood_shell_margin_mean": float(neighborhood_shell["mean"]),
            "neighborhood_shell_positive_ratio": float(
                neighborhood_shell["positive_ratio"]
            ),
            "neighborhood_full_margin_mean": float(neighborhood_full["mean"]),
            "neighborhood_full_positive_ratio": float(
                neighborhood_full["positive_ratio"]
            ),
            "core_margin_delta_mean": float(
                neighborhood_core["mean"] - exact_core["mean"]
            ),
        }
        for mode in control_names:
            source_key = (
                "training_control_margins"
                if mode in neighborhood["training_control_margins"]
                else "heldout_margins"
            )
            margin = neighborhood[source_key][mode]
            for region_name, region_mask in (("core", core), ("shell", shell)):
                stats = selected_stats(margin, region_mask)
                row[f"{mode}_{region_name}_margin_mean"] = float(stats["mean"])
                row[f"{mode}_{region_name}_positive_ratio"] = float(
                    stats["positive_ratio"]
                )
        records.append(row)

    samples = int(args.bootstrap_samples)
    primary: dict[str, Any] = {}
    for prefix in (
        "exact_core",
        "neighborhood_core",
        "neighborhood_shell",
        "neighborhood_full",
    ):
        primary[f"{prefix}_margin_mean"] = aggregate_metric(
            records, f"{prefix}_margin_mean", bootstrap_samples=samples
        )
        primary[f"{prefix}_positive_ratio"] = aggregate_metric(
            records, f"{prefix}_positive_ratio", bootstrap_samples=samples
        )
    thresholds = {
        "exact_core": float(args.min_core_positive_ratio),
        "neighborhood_core": float(args.min_core_positive_ratio),
        "neighborhood_shell": float(args.min_shell_positive_ratio),
        "neighborhood_full": float(args.min_core_positive_ratio),
    }
    for prefix, threshold in thresholds.items():
        primary[f"{prefix}_object_pass_rate"] = mean(
            float(row[f"{prefix}_positive_ratio"]) >= threshold for row in records
        )

    controls: dict[str, Any] = {}
    for mode in control_names:
        controls[mode] = {}
        for region in ("core", "shell"):
            controls[mode][region] = {
                "margin": aggregate_metric(
                    records,
                    f"{mode}_{region}_margin_mean",
                    bootstrap_samples=samples,
                ),
                "positive_ratio": aggregate_metric(
                    records,
                    f"{mode}_{region}_positive_ratio",
                    bootstrap_samples=samples,
                ),
            }

    support = {
        "exact_active_count": aggregate_metric(
            records, "exact_active_count", bootstrap_samples=samples
        ),
        "neighborhood_active_count": aggregate_metric(
            records, "neighborhood_active_count", bootstrap_samples=samples
        ),
        "core_count": aggregate_metric(records, "core_count", bootstrap_samples=samples),
        "shell_count": aggregate_metric(
            records, "shell_count", bootstrap_samples=samples
        ),
        "lost_count": aggregate_metric(records, "lost_count", bootstrap_samples=samples),
        "support_expansion_fraction_exact": aggregate_metric(
            records, "support_expansion_fraction_exact", bootstrap_samples=samples
        ),
    }
    checks = {
        "neighborhood_formal_report_passed": neighborhood_report.get("passed")
        is True,
        "neighborhood_formal_checks_all_true": bool(
            neighborhood_report.get("checks")
        )
        and all(bool(value) for value in neighborhood_report["checks"].values()),
        "neighborhood_recomputed_decision_passed": neighborhood_recomputed_passed,
        "neighborhood_raw_metrics_match_report": neighborhood_metrics_match,
        "same_object_count": len(records) == int(exact_report["object_count"]),
        "no_exact_support_lost": all(int(row["lost_count"]) == 0 for row in records),
        "every_object_has_core": all(int(row["core_count"]) > 0 for row in records),
        "every_object_has_shell": all(int(row["shell_count"]) > 0 for row in records),
        "neighborhood_core_margin_positive": float(
            primary["neighborhood_core_margin_mean"]["object"]["mean"]
        )
        > 0.0,
        "neighborhood_shell_margin_positive": float(
            primary["neighborhood_shell_margin_mean"]["object"]["mean"]
        )
        > 0.0,
        "neighborhood_core_positive_ratio": float(
            primary["neighborhood_core_positive_ratio"]["object"]["mean"]
        )
        >= float(args.min_core_positive_ratio),
        "neighborhood_shell_positive_ratio": float(
            primary["neighborhood_shell_positive_ratio"]["object"]["mean"]
        )
        >= float(args.min_shell_positive_ratio),
        "neighborhood_core_object_pass_rate": float(
            primary["neighborhood_core_object_pass_rate"]
        )
        >= float(args.min_core_object_pass_rate),
        "neighborhood_shell_object_pass_rate": float(
            primary["neighborhood_shell_object_pass_rate"]
        )
        >= float(args.min_shell_object_pass_rate),
    }
    report = {
        "format": CORE_SHELL_AUDIT_VERSION,
        "stage": "N0 exact-core versus neighborhood-shell matched-support audit",
        "passed": all(checks.values()),
        "exact_report": str((exact_dir / "report.json").resolve()),
        "neighborhood_report": str((neighborhood_dir / "report.json").resolve()),
        "checkpoint": exact_report["checkpoint"],
        "checkpoint_step": int(exact_report["checkpoint_step"]),
        "training_seed": int(exact_report["training_seed"]),
        "split_name": exact_report["split_name"],
        "object_count": len(records),
        "thresholds": {
            "min_core_positive_ratio": float(args.min_core_positive_ratio),
            "min_shell_positive_ratio": float(args.min_shell_positive_ratio),
            "min_core_object_pass_rate": float(args.min_core_object_pass_rate),
            "min_shell_object_pass_rate": float(args.min_shell_object_pass_rate),
        },
        "support": support,
        "primary": primary,
        "controls": controls,
        "neighborhood_formal_audit": {
            "stored_passed": neighborhood_report.get("passed") is True,
            "recomputed_passed": neighborhood_recomputed_passed,
            "raw_metrics_match_report": neighborhood_metrics_match,
        },
        "checks": checks,
        "records": records,
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    write_markdown(report, output_dir / "report.md")
    print(json.dumps({"passed": report["passed"], "checks": checks}, indent=2))
    if args.fail_on_decision and not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
