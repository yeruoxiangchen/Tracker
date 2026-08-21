#!/usr/bin/env python3
"""Evaluate corrected-SS stock/full SLAT and Pixal3D on the same smoke cases.

This is a read-only follow-up to ``compare_pixal3d_singleview_smoke.py``.
It deliberately writes to a new output directory because the original smoke
protocol binds the original evaluator source by SHA-256.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

from pose_point_depth_mv.compare_pixal3d_singleview_smoke import (
    REPORT_FORMAT as PIXAL_REPORT_FORMAT,
    atomic_json,
    atomic_text,
    binding,
    load_mesh,
    sha256_file,
    similarity_icp,
    surface_metrics,
    validate_protocol,
)


FORMAT = "pose_point_depth_mv.pixal3d_multiview_branch_matrix.v1"
METHODS = ("corrected_ss_native_slat", "current_full", "pixal3d_singleview")
PRIMARY_METRICS = (
    "chamfer_l1",
    "fscore_0p02",
    "normal_consistency",
    "precision_0p02",
    "recall_0p02",
)


def numeric_summary(values: list[float]) -> dict[str, float]:
    import numpy as np

    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise ValueError("cannot summarize an empty collection")
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def resolve_stock_mesh(full_mesh: str | Path) -> Path:
    full = Path(full_mesh).resolve()
    if full.name != "mesh_canonical.obj" or full.parent.name != "full":
        raise ValueError(f"unexpected current-full mesh layout: {full}")
    stock = full.parent.parent / "stock" / full.name
    if not stock.is_file():
        raise FileNotFoundError(stock)
    return stock


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("branch matrix has no records")
    methods = {
        method: {
            metric: numeric_summary(
                [float(row["methods"][method]["surface"][metric]) for row in records]
            )
            for metric in PRIMARY_METRICS
        }
        for method in METHODS
    }

    comparisons: dict[str, Any] = {}
    comparison_pairs = (
        (
            "full_minus_stock",
            "current_full",
            "corrected_ss_native_slat",
        ),
        (
            "full_minus_pixal3d",
            "current_full",
            "pixal3d_singleview",
        ),
        (
            "stock_minus_pixal3d",
            "corrected_ss_native_slat",
            "pixal3d_singleview",
        ),
    )
    for name, lhs, rhs in comparison_pairs:
        # Positive deltas always mean lhs is better.
        chamfer = [
            float(row["methods"][rhs]["surface"]["chamfer_l1"])
            - float(row["methods"][lhs]["surface"]["chamfer_l1"])
            for row in records
        ]
        fscore = [
            float(row["methods"][lhs]["surface"]["fscore_0p02"])
            - float(row["methods"][rhs]["surface"]["fscore_0p02"])
            for row in records
        ]
        normal = [
            float(row["methods"][lhs]["surface"]["normal_consistency"])
            - float(row["methods"][rhs]["surface"]["normal_consistency"])
            for row in records
        ]
        comparisons[name] = {
            "lhs": lhs,
            "rhs": rhs,
            "positive_means_lhs_better": True,
            "chamfer_improvement": numeric_summary(chamfer),
            "fscore_0p02_delta": numeric_summary(fscore),
            "normal_consistency_delta": numeric_summary(normal),
            "chamfer_win_rate": float(sum(value > 0.0 for value in chamfer) / len(chamfer)),
            "fscore_win_rate": float(sum(value > 0.0 for value in fscore) / len(fscore)),
            "normal_win_rate": float(sum(value > 0.0 for value in normal) / len(normal)),
        }

    full_stock_case_deltas = []
    for row in records:
        stock = row["methods"]["corrected_ss_native_slat"]["surface"]
        full = row["methods"]["current_full"]["surface"]
        full_stock_case_deltas.append(
            {
                "case_id": row["case_id"],
                "chamfer_improvement": float(
                    stock["chamfer_l1"] - full["chamfer_l1"]
                ),
                "fscore_0p02_delta": float(
                    full["fscore_0p02"] - stock["fscore_0p02"]
                ),
            }
        )
    dominant = min(full_stock_case_deltas, key=lambda row: row["chamfer_improvement"])
    remaining = [row for row in records if row["case_id"] != dominant["case_id"]]
    sensitivity = {
        "criterion": "most negative full-vs-stock chamfer improvement",
        "excluded_case": dominant,
        "remaining_case_count": len(remaining),
        "remaining_summary": (
            summarize_records_without_sensitivity(remaining) if remaining else None
        ),
    }
    return {
        "case_count": len(records),
        "methods": methods,
        "comparisons": comparisons,
        "full_stock_sensitivity": sensitivity,
    }


def summarize_records_without_sensitivity(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    stock_chamfer = [
        float(row["methods"]["corrected_ss_native_slat"]["surface"]["chamfer_l1"])
        for row in records
    ]
    full_chamfer = [
        float(row["methods"]["current_full"]["surface"]["chamfer_l1"])
        for row in records
    ]
    stock_fscore = [
        float(row["methods"]["corrected_ss_native_slat"]["surface"]["fscore_0p02"])
        for row in records
    ]
    full_fscore = [
        float(row["methods"]["current_full"]["surface"]["fscore_0p02"])
        for row in records
    ]
    return {
        "corrected_ss_native_slat": {
            "chamfer_l1": numeric_summary(stock_chamfer),
            "fscore_0p02": numeric_summary(stock_fscore),
        },
        "current_full": {
            "chamfer_l1": numeric_summary(full_chamfer),
            "fscore_0p02": numeric_summary(full_fscore),
        },
        "full_minus_stock": {
            "chamfer_improvement": numeric_summary(
                [stock - full for stock, full in zip(stock_chamfer, full_chamfer)]
            ),
            "fscore_0p02_delta": numeric_summary(
                [full - stock for stock, full in zip(stock_fscore, full_fscore)]
            ),
        },
    }


def load_pixal_report(path: Path, protocol: dict[str, Any]) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format") != PIXAL_REPORT_FORMAT:
        raise ValueError(f"unexpected Pixal report format={payload.get('format')!r}")
    if payload.get("protocol_sha256") != protocol["protocol_sha256"]:
        raise RuntimeError("Pixal report and protocol canonical SHA differ")
    records = payload.get("records")
    if not isinstance(records, list) or len(records) != len(protocol["cases"]):
        raise RuntimeError("Pixal report case coverage differs from protocol")
    return payload


def validate_current_report(
    protocol: dict[str, Any],
) -> dict[tuple[str, int], dict[str, Any]]:
    current_binding = protocol["bindings"]["current_report"]
    current_path = Path(current_binding["path"])
    if sha256_file(current_path) != current_binding["sha256"]:
        raise RuntimeError("frozen current report changed")
    report = json.loads(current_path.read_text(encoding="utf-8"))
    if report.get("same_coordinates") != "both branches use frozen corrected-SS coords":
        raise RuntimeError("current report is not the corrected-coordinate comparison")
    if report.get("same_noise") != "coordinate-keyed SLAT initial noise is bit-identical":
        raise RuntimeError("current report is not a same-SLAT-noise comparison")
    output: dict[tuple[str, int], dict[str, Any]] = {}
    for row in report.get("records", []):
        key = (str(row["pair_id"]), int(row["joint_seed"]))
        if key in output:
            raise RuntimeError(f"duplicate current report record: {key}")
        if row.get("same_initial_noise") is not True:
            raise RuntimeError(f"current record lacks same-noise flag: {key}")
        output[key] = row
    return output


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--pixal_report", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--candidate_samples", type=int, default=1000)
    parser.add_argument("--alignment_samples", type=int, default=4000)
    parser.add_argument("--candidate_iterations", type=int, default=8)
    parser.add_argument("--final_iterations", type=int, default=30)
    parser.add_argument("--surface_samples", type=int, default=20000)
    parser.add_argument(
        "--reuse_frozen_full_pixal_metrics",
        action="store_true",
        help=(
            "reuse the already SHA-bound current-full/Pixal metrics and only "
            "align/evaluate corrected-SS native-SLAT stock"
        ),
    )
    return parser


def main() -> None:
    args = make_parser().parse_args()
    if min(
        args.candidate_samples,
        args.alignment_samples,
        args.candidate_iterations,
        args.final_iterations,
        args.surface_samples,
    ) <= 0:
        raise ValueError("sample and iteration counts must be positive")
    protocol_path = args.protocol.resolve()
    pixal_report_path = args.pixal_report.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite branch-matrix output: {output_dir}"
        )
    output_dir.mkdir(parents=True)

    protocol = validate_protocol(protocol_path)
    pixal_report = load_pixal_report(pixal_report_path, protocol)
    current_rows = validate_current_report(protocol)
    pixal_by_case = {
        str(row["case_id"]): row for row in pixal_report["records"]
    }
    records: list[dict[str, Any]] = []
    for position, case in enumerate(protocol["cases"], start=1):
        case_id = str(case["case_id"])
        pixal_row = pixal_by_case.get(case_id)
        if pixal_row is None:
            raise RuntimeError(f"Pixal report lacks case={case_id}")
        current_key = (str(case["pair_id"]), int(case["current_seed"]))
        current_row = current_rows.get(current_key)
        if current_row is None:
            raise RuntimeError(f"current report lacks record={current_key}")
        if (
            str(current_row["uid"]) != str(case["uid"])
            or str(current_row["object_uid"]) != str(case["object_uid"])
            or int(current_row["dataset_index"]) != int(case["dataset_index"])
        ):
            raise RuntimeError(f"current case identity mismatch: {case_id}")

        full_path = Path(case["current_mesh"]["path"]).resolve()
        stock_path = resolve_stock_mesh(full_path)
        pixal_path = Path(pixal_row["pixal3d"]["mesh"]["path"]).resolve()
        for label, path, expected in (
            ("full", full_path, case["current_mesh"]["sha256"]),
            ("pixal3d", pixal_path, pixal_row["pixal3d"]["mesh"]["sha256"]),
        ):
            if sha256_file(path) != expected:
                raise RuntimeError(f"{case_id} {label} mesh binding changed")
        target = load_mesh(case["target_mesh"]["path"])
        source_paths = {
            "corrected_ss_native_slat": stock_path,
            "current_full": full_path,
            "pixal3d_singleview": pixal_path,
        }
        methods: dict[str, Any] = {}
        # Use the exact original full seed for stock/full and the original
        # Pixal offset for reconciliation with the frozen two-method report.
        alignment_seeds = {
            "corrected_ss_native_slat": args.seed + position * 100,
            "current_full": args.seed + position * 100,
            "pixal3d_singleview": args.seed + position * 100 + 10,
        }
        for method in METHODS:
            source_path = source_paths[method]
            if args.reuse_frozen_full_pixal_metrics and method != METHODS[0]:
                old_key = (
                    "current" if method == "current_full" else "pixal3d"
                )
                old = pixal_row[old_key]
                methods[method] = {
                    "mesh": binding(source_path),
                    "alignment": old["alignment"],
                    "alignment_seed": int(alignment_seeds[method]),
                    "surface": old["surface"],
                    "reused_from_frozen_pixal_report": True,
                }
                if "aligned_mesh" in old:
                    methods[method]["aligned_mesh"] = old["aligned_mesh"]
                continue
            aligned, alignment = similarity_icp(
                load_mesh(source_path),
                target,
                seed=int(alignment_seeds[method]),
                candidate_samples=int(args.candidate_samples),
                final_samples=int(args.alignment_samples),
                candidate_iterations=int(args.candidate_iterations),
                final_iterations=int(args.final_iterations),
            )
            surface = surface_metrics(
                aligned,
                target,
                count=int(args.surface_samples),
                seed=int(args.seed) + position,
                thresholds=(0.01, 0.02, 0.05),
            )
            aligned_path = output_dir / "aligned" / case_id / f"{method}.obj"
            aligned_path.parent.mkdir(parents=True, exist_ok=True)
            aligned.export(aligned_path)
            methods[method] = {
                "mesh": binding(source_path),
                "aligned_mesh": binding(aligned_path),
                "alignment": alignment,
                "alignment_seed": int(alignment_seeds[method]),
                "surface": surface,
                "reused_from_frozen_pixal_report": False,
            }
        for method, old_key in (
            ("current_full", "current"),
            ("pixal3d_singleview", "pixal3d"),
        ):
            for metric in PRIMARY_METRICS:
                old = float(pixal_row[old_key]["surface"][metric])
                new = float(methods[method]["surface"][metric])
                if abs(old - new) > 1.0e-12:
                    raise RuntimeError(
                        f"frozen metric reconciliation failed: "
                        f"{case_id}.{method}.{metric}: {new} != {old}"
                    )
        records.append(
            {
                "case_id": case_id,
                "uid": case["uid"],
                "object_uid": case["object_uid"],
                "view_count": int(case["view_count"]),
                "pair_id": case["pair_id"],
                "joint_seed": int(case["current_seed"]),
                "dataset_index": int(case["dataset_index"]),
                "same_corrected_ss_coordinates_stock_vs_full": True,
                "same_initial_slat_noise_stock_vs_full": True,
                "methods": methods,
                "target_mesh": case["target_mesh"],
            }
        )
        print(
            f"[branch_matrix] {position}/{len(protocol['cases'])} {case_id}",
            flush=True,
        )

    report = {
        "format": FORMAT,
        "formal": False,
        "passed": True,
        "protocol": binding(protocol_path),
        "protocol_sha256": protocol["protocol_sha256"],
        "pixal_report": binding(pixal_report_path),
        "source_current_report": protocol["bindings"]["current_report"],
        "code": binding(Path(__file__)),
        "evaluation": {
            "seed": int(args.seed),
            "candidate_samples": int(args.candidate_samples),
            "alignment_samples": int(args.alignment_samples),
            "candidate_iterations": int(args.candidate_iterations),
            "final_iterations": int(args.final_iterations),
            "surface_samples": int(args.surface_samples),
            "reuse_frozen_full_pixal_metrics": bool(
                args.reuse_frozen_full_pixal_metrics
            ),
            "alignment": (
                "24 proper cube rotations plus proper isotropic-similarity ICP; "
                "stock/full share alignment initialization; original Pixal offset retained"
            ),
        },
        "summary": summarize_records(records),
        "records": records,
        "interpretation": {
            "stock_to_full": (
                "isolates Direct-SLAT support injection because corrected SS coordinates "
                "and coordinate-keyed SLAT initial noise are identical"
            ),
            "stock_or_full_to_pixal3d": (
                "cross-system exploratory comparison; different input budgets and "
                "training corpora remain confounders"
            ),
        },
        "guardrail": (
            "Exploratory n=6 viewed cases. This matrix diagnoses where the existing "
            "Full path loses quality; it is not a formal model-ranking gate."
        ),
    }
    report_path = output_dir / "report.json"
    atomic_json(report_path, report)

    csv_path = output_dir / "metrics.csv"
    temporary = csv_path.with_name(f".{csv_path.name}.tmp-{os.getpid()}")
    fields = ["case_id", "view_count"]
    for method in METHODS:
        fields.extend(f"{method}_{metric}" for metric in PRIMARY_METRICS)
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in records:
            output: dict[str, Any] = {
                "case_id": row["case_id"],
                "view_count": row["view_count"],
            }
            for method in METHODS:
                for metric in PRIMARY_METRICS:
                    output[f"{method}_{metric}"] = row["methods"][method]["surface"][
                        metric
                    ]
            writer.writerow(output)
    os.replace(temporary, csv_path)

    summary = report["summary"]
    lines = [
        "Pixal3D / corrected-SS native-SLAT / current Full branch matrix",
        "================================================================",
        "FORMAL: false",
        f"cases: {summary['case_count']}",
        "",
    ]
    for method in METHODS:
        row = summary["methods"][method]
        lines.append(
            f"{method}: chamfer={row['chamfer_l1']['mean']:.9f}, "
            f"fscore@0.02={row['fscore_0p02']['mean']:.9f}, "
            f"normal={row['normal_consistency']['mean']:.9f}, "
            f"precision={row['precision_0p02']['mean']:.9f}, "
            f"recall={row['recall_0p02']['mean']:.9f}"
        )
    lines.extend(
        [
            "",
            "Positive comparison values mean the lhs method is better.",
            json.dumps(summary["comparisons"], ensure_ascii=False, indent=2),
            "",
            "Leave-one-out sensitivity:",
            json.dumps(summary["full_stock_sensitivity"], ensure_ascii=False, indent=2),
            "",
            report["guardrail"],
        ]
    )
    atomic_text(output_dir / "summary.txt", "\n".join(lines) + "\n")
    print("\n".join(lines[:9]), flush=True)
    print(f"report: {report_path}", flush=True)


if __name__ == "__main__":
    main()
