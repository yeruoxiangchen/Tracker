#!/usr/bin/env python3
"""Compare Stock-init Objaverse2K step800 with frozen development references."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from pose_point_depth_mv.eval_direct_slat_flow import summarize
from pose_point_depth_mv.evaluate_objaverse2k_no_vggt_slat import (
    STRUCTURE_METRICS,
    SURFACE_METRICS,
    absolute_branch_differences,
    atomic_json,
    canonical_json_sha256,
    paired_improvement,
    parse_csv,
    stable_seed,
    validate_worker_reports,
)
from pose_point_depth_mv.evaluate_objaverse2k_no_vggt_slat_fourway import (
    validate_worker_reports as validate_fourway_worker_reports,
)


REPORT_FORMAT = "pose_point_depth_mv.objaverse2k_stockinit_step800_comparison.v1"
ALL_METRICS = (*SURFACE_METRICS, *STRUCTURE_METRICS)
MODEL_KEYS = (
    "m8_step800",
    "stockinit_objaverse2k800",
    "current_m8init_objaverse2k800",
    "current_m8init_objaverse2k2000",
)
COMPARISONS = (
    ("stockinit_objaverse2k800", "stock", "stockinit800_vs_stock"),
    ("m8_step800", "stock", "m8_step800_vs_stock"),
    (
        "current_m8init_objaverse2k800",
        "stock",
        "current_m8init_objaverse2k800_vs_stock",
    ),
    (
        "current_m8init_objaverse2k2000",
        "stock",
        "current_m8init_objaverse2k2000_vs_stock",
    ),
    ("stockinit_objaverse2k800", "m8_step800", "stockinit800_vs_m8_step800"),
    (
        "stockinit_objaverse2k800",
        "current_m8init_objaverse2k800",
        "stockinit800_vs_current_m8init800",
    ),
    (
        "stockinit_objaverse2k800",
        "current_m8init_objaverse2k2000",
        "stockinit800_vs_current_m8init2000",
    ),
    (
        "current_m8init_objaverse2k800",
        "m8_step800",
        "current_m8init800_vs_m8_step800",
    ),
)
STOCK_P95_TOLERANCES = {
    "chamfer_l1": 1.0e-3,
    "chamfer_l2": 2.0e-4,
    "fscore_0p01": 1.0e-2,
    "fscore_0p02": 1.0e-2,
    "fscore_0p05": 1.0e-2,
    "normal_consistency": 1.5e-2,
    "largest_component_ratio": 5.0e-2,
    "component_count": 20.0,
}
STOCK_MAX_TOLERANCES = {
    **STOCK_P95_TOLERANCES,
    # Independent bf16 sparse/decoder runs are not byte deterministic.  A
    # highly tessellated dev object has a reproducible long tail in these two
    # metrics, while the across-record P95 remains well below the strict gate.
    "chamfer_l2": 3.0e-4,
    "normal_consistency": 5.0e-2,
}


def _paired_identity(record: dict[str, Any]) -> dict[str, Any]:
    identity = dict(record["identity"])
    identity.pop("model_label", None)
    identity.pop("checkpoint_sha256", None)
    return identity


def _validate_protocols(
    report_groups: dict[str, list[dict[str, Any]]],
    record_groups: dict[str, dict[tuple[str, str, int], dict[str, Any]]],
) -> None:
    reference_key = "stockinit_objaverse2k800"
    reference_records = record_groups[reference_key]
    invariant_fields = (
        "cache_manifest_sha256",
        "lifting_cache_manifest_sha256",
        "native_ss_report_sha256",
        "stock_slat_freeze_sha256",
        "sampling",
        "joint_seeds",
        "noise_protocol",
        "noise_seed",
        "surface_samples",
        "expected_objects",
    )
    reference_config = report_groups[reference_key][0]["run_config"]
    for model_key in MODEL_KEYS:
        if set(record_groups[model_key]) != set(reference_records):
            raise RuntimeError(f"{model_key} evaluation matrix differs")
        config = report_groups[model_key][0]["run_config"]
        mismatch = {
            field: (reference_config.get(field), config.get(field))
            for field in invariant_fields
            if reference_config.get(field) != config.get(field)
        }
        if mismatch:
            raise RuntimeError(f"{model_key} evaluation protocol differs: {mismatch}")
        for key, record in record_groups[model_key].items():
            if _paired_identity(record) != _paired_identity(reference_records[key]):
                raise RuntimeError(f"{model_key} paired identity differs: {key}")


def aggregate_comparison(
    *,
    report_groups: dict[str, list[dict[str, Any]]],
    record_groups: dict[str, dict[tuple[str, str, int], dict[str, Any]]],
    bootstrap_samples: int,
) -> dict[str, Any]:
    _validate_protocols(report_groups, record_groups)
    reference_records = record_groups["stockinit_objaverse2k800"]
    stock_reproduction_rows = []
    seed_rows = []
    by_object: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for key in sorted(reference_records):
        records = {name: record_groups[name][key] for name in MODEL_KEYS}
        canonical_stock = records["stockinit_objaverse2k800"]["branches"]["stock"]
        branches = {
            "stock": canonical_stock,
            "m8_step800": records["m8_step800"]["branches"]["full"],
            "stockinit_objaverse2k800": records["stockinit_objaverse2k800"]
            ["branches"]["correct"],
            "current_m8init_objaverse2k800": records[
                "current_m8init_objaverse2k800"
            ]["branches"]["full"],
            "current_m8init_objaverse2k2000": records[
                "current_m8init_objaverse2k2000"
            ]["branches"]["full"],
        }
        stock_differences = {}
        for model_key, record in records.items():
            differences = absolute_branch_differences(
                record["branches"]["stock"], canonical_stock
            )
            failed = {
                metric: {
                    "difference": differences[metric],
                    "max_tolerance": tolerance,
                }
                for metric, tolerance in STOCK_MAX_TOLERANCES.items()
                if differences[metric] > tolerance
            }
            if failed:
                raise RuntimeError(
                    f"{model_key} Stock reproduction exceeds tolerance: {key} {failed}"
                )
            stock_differences[model_key] = differences
        stock_reproduction_rows.append(
            {
                "object_uid": key[0],
                "uid": key[1],
                "support_seed": key[2],
                "absolute_differences_vs_stockinit_run": stock_differences,
            }
        )
        comparisons = {}
        for left, right, label in COMPARISONS:
            # A model-vs-Stock delta must use the Stock rollout generated in
            # the same process/run.  Cross-run Stock values are checked below
            # for numerical reproduction, but are not used as its baseline.
            right_branch = (
                records[left]["branches"]["stock"]
                if right == "stock"
                else branches[right]
            )
            comparisons[label] = paired_improvement(branches[left], right_branch)
        row = {
            "object_uid": key[0],
            "uid": key[1],
            "support_seed": key[2],
            "master_noise_seed": records["stockinit_objaverse2k800"]["identity"]
            ["master_noise_seed"],
            "branches": branches,
            "comparisons": comparisons,
        }
        seed_rows.append(row)
        by_object[key[0]].append(row)

    object_rows = []
    for object_uid, rows in sorted(by_object.items()):
        object_rows.append(
            {
                "object_uid": object_uid,
                "seed_count": len(rows),
                "comparisons": {
                    label: {
                        metric: float(
                            np.mean(
                                [row["comparisons"][label][metric] for row in rows]
                            )
                        )
                        for metric in ALL_METRICS
                    }
                    for _, _, label in COMPARISONS
                },
            }
        )
    expected_seed_count = len(
        report_groups["stockinit_objaverse2k800"][0]["run_config"]["joint_seeds"]
    )
    if any(row["seed_count"] != expected_seed_count for row in object_rows):
        raise RuntimeError("object seed coverage is incomplete")
    stock_reproduction_summary = {}
    for model_key in MODEL_KEYS:
        stock_reproduction_summary[model_key] = {}
        for metric in ALL_METRICS:
            values = np.asarray(
                [
                    row["absolute_differences_vs_stockinit_run"][model_key][metric]
                    for row in stock_reproduction_rows
                ],
                dtype=np.float64,
            )
            p95 = float(np.quantile(values, 0.95))
            max_index = int(np.argmax(values))
            worst_row = stock_reproduction_rows[max_index]
            if p95 > STOCK_P95_TOLERANCES[metric]:
                raise RuntimeError(
                    f"{model_key} Stock reproduction P95 exceeds tolerance: "
                    f"{metric} p95={p95} tolerance={STOCK_P95_TOLERANCES[metric]}"
                )
            stock_reproduction_summary[model_key][metric] = {
                "mean": float(np.mean(values)),
                "p95": p95,
                "max": float(values[max_index]),
                "worst_identity": {
                    "object_uid": worst_row["object_uid"],
                    "uid": worst_row["uid"],
                    "support_seed": worst_row["support_seed"],
                },
            }
    summary = {
        label: {
            "left": left,
            "right": right,
            "positive_means_left_better": True,
            "metrics": {
                metric: summarize(
                    [row["comparisons"][label][metric] for row in object_rows],
                    bootstrap_samples=int(bootstrap_samples),
                    seed=stable_seed("stockinit_step800_comparison", label, metric),
                )
                for metric in ALL_METRICS
            },
        }
        for left, right, label in COMPARISONS
    }
    checkpoints = {
        model_key: {
            "path": report_groups[model_key][0]["run_config"]["checkpoint"],
            "sha256": report_groups[model_key][0]["run_config"][
                "checkpoint_sha256"
            ],
            "step": report_groups[model_key][0]["run_config"]["checkpoint_step"],
            "weights": report_groups[model_key][0]["run_config"]["weights"],
            "evaluation_workers": len(report_groups[model_key]),
        }
        for model_key in MODEL_KEYS
    }
    return {
        "format": REPORT_FORMAT,
        "passed": True,
        "formal": False,
        "training_overlap": False,
        "object_count": len(object_rows),
        "seed_count_per_object": expected_seed_count,
        "record_count": len(seed_rows),
        "branches": ["stock", *MODEL_KEYS],
        "same_native_ss_coordinates": True,
        "same_initial_noise": True,
        "checkpoints": checkpoints,
        "training_topology": {
            "stockinit_objaverse2k800": "8 GPU x grad_accum 1 = global batch 8",
            "m8_step800": "historical M8 training topology",
            "current_m8init_objaverse2k800": (
                "8 GPU x grad_accum 1 = global batch 8"
            ),
            "current_m8init_objaverse2k2000": (
                "8 GPU x grad_accum 1 = global batch 8"
            ),
        },
        "initialization_interpretation": {
            "stockinit_objaverse2k800": (
                "fresh Stock-equivalent zero adapter; no init checkpoint"
            ),
            "m8_step800": (
                "historical M8 trajectory at its own step800; not a Stock-init run"
            ),
            "current_m8init_objaverse2k800": (
                "M8 step2000 EMA warm-start plus 800 Objaverse2K updates"
            ),
            "current_m8init_objaverse2k2000": (
                "M8 step2000 EMA warm-start plus 2000 Objaverse2K updates"
            ),
        },
        "stock_reproduction": {
            "passed": True,
            "p95_tolerances": STOCK_P95_TOLERANCES,
            "max_tolerances": STOCK_MAX_TOLERANCES,
            "summary_by_model": stock_reproduction_summary,
            "rows": stock_reproduction_rows,
            "interpretation": (
                "Independent bf16 CUDA sparse/decoder Stock runs are not byte "
                "deterministic. Every record must pass the maximum gate and the "
                "48-record P95 must pass the stricter distributional gate."
            ),
        },
        "stock_comparison_pairing": (
            "Every model-vs-Stock delta uses the Stock rollout from that model's "
            "own matched worker record."
        ),
        "summary": summary,
        "object_rows": object_rows,
        "records": seed_rows,
        "scope_guard": (
            "This frozen dev16 slice is object-disjoint from Objaverse2K SLat "
            "training but is used for development and checkpoint selection. The "
            "Stock-init and current M8-init Objaverse2K trajectories have matched "
            "eight-GPU topology, global batch, seed, schedule, data, and Objaverse2K "
            "update counts at step800; their intended difference is initialization "
            "history. This is not formal holdout evidence."
        ),
    }


def load_group(
    paths: str, *, model_label: str
) -> tuple[list[dict[str, Any]], dict[tuple[str, str, int], dict[str, Any]]]:
    values = parse_csv(paths)
    return validate_worker_reports(
        values, model_label=model_label, expected_workers=len(values)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m8_step800_reports", required=True)
    parser.add_argument("--stockinit_reports", required=True)
    parser.add_argument("--current800_reports", required=True)
    parser.add_argument("--current2000_reports", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--expected_objects", type=int, default=16)
    parser.add_argument("--bootstrap_samples", type=int, default=10000)
    args = parser.parse_args()

    report_groups: dict[str, list[dict[str, Any]]] = {}
    record_groups: dict[str, dict[tuple[str, str, int], dict[str, Any]]] = {}
    inputs = (
        ("m8_step800", args.m8_step800_reports, "m8_step800"),
        ("current_m8init_objaverse2k800", args.current800_reports, "objaverse2k"),
        ("current_m8init_objaverse2k2000", args.current2000_reports, "objaverse2k"),
    )
    for key, paths, label in inputs:
        report_groups[key], record_groups[key] = load_group(paths, model_label=label)
    stockinit_paths = parse_csv(args.stockinit_reports)
    report_groups["stockinit_objaverse2k800"], record_groups[
        "stockinit_objaverse2k800"
    ] = validate_fourway_worker_reports(
        stockinit_paths, expected_workers=len(stockinit_paths)
    )
    report = aggregate_comparison(
        report_groups=report_groups,
        record_groups=record_groups,
        bootstrap_samples=int(args.bootstrap_samples),
    )
    if report["object_count"] != int(args.expected_objects):
        raise RuntimeError("comparison object count differs from requested dev slice")
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    body = dict(report)
    report["report_sha256"] = canonical_json_sha256(body)
    atomic_json(output_dir / "report.json", report)

    lines = [
        "Objaverse2K Stock-init step800 dev16 comparison",
        "=" * 59,
        "passed: true",
        "formal: false",
        "training overlap: false",
        f"objects: {report['object_count']} seeds/object: {report['seed_count_per_object']}",
        "all deltas: positive means the left branch is better",
        "",
    ]
    for _, _, label in COMPARISONS:
        comparison = report["summary"][label]
        metrics = comparison["metrics"]
        chamfer = metrics["chamfer_l1"]
        fscore = metrics["fscore_0p02"]
        normal = metrics["normal_consistency"]
        lines.append(
            f"{label}: chamfer={chamfer['mean']:+.8f} "
            f"median={chamfer['median']:+.8f} "
            f"win={chamfer['positive_rate']:.4f} "
            f"CI={chamfer['bootstrap_mean_95_ci']}; "
            f"f@0.02={fscore['mean']:+.8f}; normal={normal['mean']:+.8f}"
        )
    lines.extend(("", report["scope_guard"]))
    (output_dir / "summary.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
