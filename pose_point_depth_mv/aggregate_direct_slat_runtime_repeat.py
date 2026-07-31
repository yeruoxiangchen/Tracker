#!/usr/bin/env python3
"""Aggregate same-process and independent-process Direct-SLAT repeats."""

from __future__ import annotations

import argparse
from collections import defaultdict
from itertools import combinations
import json
from pathlib import Path
from typing import Any

import torch

from pose_point_depth_mv.direct_slat_blind import (
    atomic_json,
    bind_file,
    canonical_sha256,
    sha256_file,
    validate_binding_tree,
)
from pose_point_depth_mv.direct_slat_runtime_repeat import (
    AGGREGATE_REPORT_FORMAT,
    PROCESS_REPORT_FORMAT,
    evaluate_runtime,
    record_metric_diff,
    repeat_policy_from_criteria,
    sparse_payload_diff,
    summarize_comparisons,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--runtime_order", default="N0,N1,N2,N3")
    parser.add_argument("--expected_processes", type=int, default=3)
    parser.add_argument("--same_process_repeats", type=int, default=5)
    parser.add_argument("--max_latent_rms_p95", type=float, required=True)
    parser.add_argument("--max_latent_max_abs_p95", type=float, required=True)
    parser.add_argument("--max_chamfer_p95", type=float, required=True)
    parser.add_argument("--max_fscore_p95", type=float, required=True)
    parser.add_argument("--max_lcr_p95", type=float, required=True)
    parser.add_argument("--max_boundary_count_p95", type=float, required=True)
    parser.add_argument("--max_boundary_length_p95", type=float, required=True)
    parser.add_argument("--max_nonmanifold_count_p95", type=float, required=True)
    parser.add_argument("--max_component_count_p95", type=float, required=True)
    parser.add_argument("--catastrophic_latent_rms_max", type=float, required=True)
    parser.add_argument(
        "--catastrophic_latent_max_abs", type=float, required=True
    )
    parser.add_argument("--catastrophic_chamfer_max", type=float, required=True)
    parser.add_argument("--catastrophic_fscore_max", type=float, required=True)
    parser.add_argument("--catastrophic_lcr_max", type=float, required=True)
    parser.add_argument(
        "--catastrophic_boundary_count_max", type=float, required=True
    )
    parser.add_argument(
        "--catastrophic_boundary_length_max", type=float, required=True
    )
    parser.add_argument(
        "--catastrophic_nonmanifold_count_max", type=float, required=True
    )
    parser.add_argument(
        "--catastrophic_component_count_max", type=float, required=True
    )
    parser.add_argument("--max_watertight_flip_rate", type=float, required=True)
    parser.add_argument("--max_zero_boundary_flip_rate", type=float, required=True)
    parser.add_argument(
        "--max_nonmanifold_free_flip_rate", type=float, required=True
    )
    return parser.parse_args()


def load_process_report(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    body = dict(report)
    saved = str(body.pop("report_sha256", ""))
    if (
        report.get("format") != PROCESS_REPORT_FORMAT
        or canonical_sha256(body) != saved
    ):
        raise RuntimeError(f"invalid runtime process report: {path}")
    validate_binding_tree(report["code_bindings"], f"{path}.code_bindings")
    report["_report_path"] = path
    report["_root"] = path.parent
    return report


def record_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return str(row["stage"]), str(row["pair_id"]), str(row["side"])


def validate_record_pair(left: dict[str, Any], right: dict[str, Any]) -> None:
    fields = (
        "stage",
        "pair_id",
        "side",
        "branch",
        "object_uid",
        "uid",
        "seed",
        "master_seed",
        "initial_noise_sha256",
    )
    if any(left.get(name) != right.get(name) for name in fields):
        raise RuntimeError("repeat records do not describe the same frozen case")
    if left["stage"] == "decoder_only" and left["latent"]["sha256"] != right[
        "latent"
    ]["sha256"]:
        raise RuntimeError("decoder-only repeats did not use one exact saved latent")


def load_bound_payload(root: Path, row: dict[str, Any]) -> dict[str, torch.Tensor]:
    relative = Path(str(row["latent"]["path"]))
    if relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError("SLAT diagnostic latent path is not safe and relative")
    path = root / relative
    if sha256_file(path) != str(row["latent"]["sha256"]):
        raise RuntimeError(f"diagnostic latent changed: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or set(payload) != {"coords", "feats"}:
        raise RuntimeError(f"invalid diagnostic latent payload: {path}")
    return payload


def comparison_row(
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    left_root: Path,
    right_root: Path,
    scope: str,
    left_process: int,
    right_process: int,
) -> dict[str, Any]:
    validate_record_pair(left, right)
    metric_diff, topology_changed = record_metric_diff(left, right)
    coords_exact = True
    latent_diff = None
    if left["stage"] == "slat":
        latent_diff = sparse_payload_diff(
            load_bound_payload(left_root, left),
            load_bound_payload(right_root, right),
        )
        coords_exact = bool(latent_diff["coords_exact"])
        metric_diff.update(
            {
                "latent_feature_rms": float(latent_diff["latent_feature_rms"]),
                "latent_feature_max_abs": float(
                    latent_diff["latent_feature_max_abs"]
                ),
            }
        )
    return {
        "scope": scope,
        "stage": str(left["stage"]),
        "pair_id": str(left["pair_id"]),
        "side": str(left["side"]),
        "branch": str(left["branch"]),
        "object_uid": str(left["object_uid"]),
        "seed": int(left["seed"]),
        "left_process_index": int(left_process),
        "right_process_index": int(right_process),
        "left_run_index": int(left["run_index"]),
        "right_run_index": int(right["run_index"]),
        "metric_abs_diff": metric_diff,
        "topology_changed": topology_changed,
        "coords_exact": coords_exact,
        "latent_diff": latent_diff,
        "hard_integrity_passed": bool(left["hard_integrity"]["passed"])
        and bool(right["hard_integrity"]["passed"]),
    }


def build_runtime_comparisons(
    reports: list[dict[str, Any]],
    *,
    expected_processes: int,
    same_process_repeats: int,
) -> list[dict[str, Any]]:
    process_indices = [int(report["process_index"]) for report in reports]
    if (
        len(reports) != expected_processes
        or len(process_indices) != len(set(process_indices))
    ):
        raise RuntimeError("runtime has missing or duplicate independent processes")
    reference_selection = reports[0]["case_selection"]
    reference_protocol = reports[0]["protocol"]["protocol_sha256"]
    reference_runtime = reports[0]["runtime"]
    if any(
        report["case_selection"] != reference_selection
        or report["protocol"]["protocol_sha256"] != reference_protocol
        or report["runtime"] != reference_runtime
        for report in reports[1:]
    ):
        raise RuntimeError("runtime process identities or case selections differ")
    repeat_processes = [
        report
        for report in reports
        if int(report["slat_repeats"]) == same_process_repeats
        and int(report["decoder_repeats"]) == same_process_repeats
    ]
    if len(repeat_processes) != 1:
        raise RuntimeError("runtime must contain exactly one multi-repeat process")
    if any(
        int(report["slat_repeats"]) not in (1, same_process_repeats)
        or int(report["decoder_repeats"]) not in (1, same_process_repeats)
        for report in reports
    ):
        raise RuntimeError("unexpected per-process repeat count")

    output: list[dict[str, Any]] = []
    for report in reports:
        grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in report["records"]:
            grouped[record_key(row)].append(row)
        for (stage, _, _), rows in grouped.items():
            rows.sort(key=lambda row: int(row["run_index"]))
            if len(rows) <= 1:
                continue
            scope = f"{stage}_same_process"
            for left, right in combinations(rows, 2):
                output.append(
                    comparison_row(
                        left,
                        right,
                        left_root=report["_root"],
                        right_root=report["_root"],
                        scope=scope,
                        left_process=int(report["process_index"]),
                        right_process=int(report["process_index"]),
                    )
                )

    cross_process: dict[tuple[str, str, str], list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for report in reports:
        for row in report["records"]:
            if int(row["run_index"]) == 0:
                cross_process[record_key(row)].append((report, row))
    for (stage, _, _), values in cross_process.items():
        if len(values) != expected_processes:
            raise RuntimeError("independent-process case coverage differs")
        scope = f"{stage}_independent_process"
        for (left_report, left), (right_report, right) in combinations(values, 2):
            output.append(
                comparison_row(
                    left,
                    right,
                    left_root=left_report["_root"],
                    right_root=right_report["_root"],
                    scope=scope,
                    left_process=int(left_report["process_index"]),
                    right_process=int(right_report["process_index"]),
                )
            )
    return output


def criteria_from_args(args: argparse.Namespace) -> dict[str, Any]:
    regular = {
        "latent_feature_rms": args.max_latent_rms_p95,
        "latent_feature_max_abs": args.max_latent_max_abs_p95,
        "chamfer_l1_abs": args.max_chamfer_p95,
        "fscore_0p02_abs": args.max_fscore_p95,
        "largest_component_ratio_abs": args.max_lcr_p95,
        "boundary_edge_count_abs": args.max_boundary_count_p95,
        "boundary_total_length_abs": args.max_boundary_length_p95,
        "nonmanifold_edge_count_abs": args.max_nonmanifold_count_p95,
        "component_count_abs": args.max_component_count_p95,
    }
    catastrophic = {
        "latent_feature_rms": args.catastrophic_latent_rms_max,
        "latent_feature_max_abs": args.catastrophic_latent_max_abs,
        "chamfer_l1_abs": args.catastrophic_chamfer_max,
        "fscore_0p02_abs": args.catastrophic_fscore_max,
        "largest_component_ratio_abs": args.catastrophic_lcr_max,
        "boundary_edge_count_abs": args.catastrophic_boundary_count_max,
        "boundary_total_length_abs": args.catastrophic_boundary_length_max,
        "nonmanifold_edge_count_abs": args.catastrophic_nonmanifold_count_max,
        "component_count_abs": args.catastrophic_component_count_max,
    }
    flips = {
        "is_watertight": args.max_watertight_flip_rate,
        "zero_boundary": args.max_zero_boundary_flip_rate,
        "nonmanifold_free": args.max_nonmanifold_free_flip_rate,
    }
    values = [*regular.values(), *catastrophic.values(), *flips.values()]
    if any(float(value) < 0.0 for value in values):
        raise ValueError("runtime criteria must be nonnegative")
    return {
        "regular_p95_max": {key: float(value) for key, value in regular.items()},
        "catastrophic_max": {
            key: float(value) for key, value in catastrophic.items()
        },
        "topology_flip_rate_max": {
            key: float(value) for key, value in flips.items()
        },
        "surface_signal_reference": {
            "s6_chamfer_mean_improvement": 0.00142657,
            "s6_fscore_mean_delta": 0.00611121,
            "note": (
                "engineering limits are frozen before v2_fix2 and are evaluated "
                "separately for surface, latent, and topology repeatability"
            ),
        },
    }


def main() -> None:
    args = parse_args()
    if args.expected_processes < 2 or args.same_process_repeats < 3:
        raise ValueError("require at least two processes and three same-process repeats")
    input_root = Path(args.input_root).resolve()
    report_paths = sorted(input_root.glob("*/process_*/report.json"))
    if not report_paths:
        raise FileNotFoundError(f"no process reports under {input_root}")
    reports = [load_process_report(path) for path in report_paths]
    runtime_order = [
        value.strip() for value in args.runtime_order.split(",") if value.strip()
    ]
    if not runtime_order or len(runtime_order) != len(set(runtime_order)):
        raise ValueError("runtime_order must be non-empty and unique")
    by_runtime: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for report in reports:
        by_runtime[str(report["runtime"]["runtime_id"])].append(report)
    expected_runtimes = set(runtime_order)
    actual_runtimes = set(by_runtime)
    if actual_runtimes != expected_runtimes:
        missing = sorted(expected_runtimes - actual_runtimes)
        unexpected = sorted(actual_runtimes - expected_runtimes)
        partial = {
            runtime_id: sorted(
                str(path.relative_to(input_root))
                for path in (input_root / runtime_id).glob("process_*/run_identity.json")
                if not (path.parent / "report.json").is_file()
            )
            for runtime_id in missing
        }
        raise RuntimeError(
            "runtime report matrix is incomplete or unexpected: "
            f"missing={missing}, unexpected={unexpected}, partial={partial}, "
            f"expected_order={runtime_order}"
        )
    criteria = criteria_from_args(args)
    runtime_results: dict[str, Any] = {}
    all_comparisons: dict[str, list[dict[str, Any]]] = {}
    for runtime_id in runtime_order:
        runtime_reports = sorted(
            by_runtime[runtime_id], key=lambda row: int(row["process_index"])
        )
        comparisons = build_runtime_comparisons(
            runtime_reports,
            expected_processes=int(args.expected_processes),
            same_process_repeats=int(args.same_process_repeats),
        )
        decision = evaluate_runtime(comparisons, criteria)
        if not all(report.get("complete") is True for report in runtime_reports):
            decision["checks"]["all_process_reports_complete"] = False
            decision["passed"] = False
        else:
            decision["checks"]["all_process_reports_complete"] = True
        runtime_results[runtime_id] = {
            "runtime": runtime_reports[0]["runtime"],
            "process_count": len(runtime_reports),
            "same_process_repeats": int(args.same_process_repeats),
            "summary": summarize_comparisons(comparisons),
            "decision": decision,
        }
        all_comparisons[runtime_id] = comparisons

    selected_id = next(
        (
            runtime_id
            for runtime_id in runtime_order
            if runtime_results[runtime_id]["decision"]["passed"]
        ),
        None,
    )
    selected = None
    repeat_policy = None
    if selected_id is not None:
        selected_runtime = dict(runtime_results[selected_id]["runtime"])
        selected = {
            "runtime_id": selected_id,
            "selection_rule": (
                "first passing runtime in the frozen order "
                + ",".join(runtime_order)
            ),
            "runtime": selected_runtime,
            "same_process_repeat_count": int(args.same_process_repeats),
            "independent_process_count": int(args.expected_processes),
        }
        repeat_policy = repeat_policy_from_criteria(criteria)
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    report = {
        "format": AGGREGATE_REPORT_FORMAT,
        "complete": True,
        "formal": False,
        "science_decision_emitted": False,
        "passed": selected is not None,
        "runtime_order": runtime_order,
        "criteria": criteria,
        "selected_runtime": selected,
        "repeat_policy": repeat_policy,
        "runtime_results": runtime_results,
        "comparison_rows": all_comparisons,
        "input_reports": [bind_file(path) for path in report_paths],
        "scope_guard": (
            "seen-object engineering runtime selection only; no unseen holdout was "
            "read and no stock/full advantage was decided"
        ),
    }
    report["report_sha256"] = canonical_sha256(report)
    atomic_json(output_dir / "report.json", report)
    lines = [
        "Direct-SLAT runtime repeatability matrix",
        "========================================",
    ]
    for runtime_id in runtime_order:
        decision = runtime_results[runtime_id]["decision"]
        lines.append(
            f"{runtime_id}: PASS={decision['passed']} "
            f"comparisons={decision['comparison_count']}"
        )
        lines.append(f"  p95={decision['worst_branch_same_process_p95']}")
        lines.append(f"  max={decision['global_max']}")
        lines.append(
            "  topology_flip_rate="
            f"{decision['worst_scope_branch_topology_flip_rate']}"
        )
    lines.extend(
        (
            "",
            f"selected runtime: {selected_id or 'NONE'}",
            report["scope_guard"],
        )
    )
    (output_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)
    if selected is None:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
