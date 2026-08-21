#!/usr/bin/env python3
"""Summarize the matched Stock-context x posed-DINO SLat experiment.

The all-view and first-view arms are trained independently from the same
contracted parent.  Each source four-way report contributes its ``lora_only``
(posed-DINO off) and ``correct`` (posed-DINO on) branches.  All paired signs are
normalized so positive means the named left side is better.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pose_point_depth_mv.direct_slat_flow import canonical_json_sha256
from pose_point_depth_mv.eval_direct_slat_flow import summarize
from pose_point_depth_mv.evaluate_mixed_no_vggt_slat_fourway import (
    STRUCTURE_METRICS,
    SURFACE_METRICS,
    paired_improvements,
)


FORMAT = "pose_point_depth_mv.mixed_slat_stock_context_2x2.v1"
METRICS = (*SURFACE_METRICS, *STRUCTURE_METRICS)
GROUPS = ("synthetic", "real", "mixed_macro_1to1")


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all_report", required=True)
    parser.add_argument("--first_report", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--bootstrap_samples", type=int, default=10000)
    return parser


def _load(path: str | Path) -> tuple[Path, dict[str, Any]]:
    resolved = Path(path).expanduser().resolve()
    report = json.loads(resolved.read_text(encoding="utf-8"))
    saved_hash = report.get("report_sha256")
    body = dict(report)
    body.pop("report_sha256", None)
    if saved_hash != canonical_json_sha256(body):
        raise RuntimeError(f"source report hash differs: {resolved}")
    if report.get("passed") is not True:
        raise RuntimeError(f"source report did not pass: {resolved}")
    if report.get("formal") is not False or report.get("training_overlap") is not True:
        raise RuntimeError("2x2 mechanism reports must remain non-formal/training-overlap")
    return resolved, report


def report_context_mode(report: dict[str, Any]) -> str:
    mode = str(
        report.get(
            "stock_context_views",
            report.get("run_config", {}).get("stock_context_views", "all"),
        )
    )
    if mode not in ("all", "first"):
        raise ValueError(f"invalid report Stock context mode={mode!r}")
    return mode


def _records(report: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    by_object: dict[tuple[str, str], dict[str, Any]] = {}
    for row in report.get("records", []):
        key = (str(row["domain"]), str(row["object_uid"]))
        branches = by_object.setdefault(key, {})
        branch = str(row["branch"])
        if branch in branches:
            raise RuntimeError(f"duplicate report branch={key}:{branch}")
        branches[branch] = row
    required = {"lora_only", "correct"}
    invalid = {key: sorted(required - set(rows)) for key, rows in by_object.items() if not required <= set(rows)}
    if invalid:
        raise RuntimeError(f"source report lacks required branches: {invalid}")
    return by_object


def _assert_matched(all_report: dict[str, Any], first_report: dict[str, Any]) -> None:
    keys = (
        "object_count",
        "objects_per_domain",
        "same_native_ss_coordinates",
        "same_initial_noise",
        "sampling",
    )
    mismatch = {
        key: (all_report.get(key), first_report.get(key))
        for key in keys
        if all_report.get(key) != first_report.get(key)
    }
    all_run = all_report.get("run_config", {})
    first_run = first_report.get("run_config", {})
    for key in (
        "checkpoint_step",
        "cache_identity",
        "selection",
        "noise_seed",
        "surface_samples",
    ):
        if all_run.get(key) != first_run.get(key):
            mismatch[f"run_config.{key}"] = (all_run.get(key), first_run.get(key))
    if mismatch:
        raise RuntimeError(f"2x2 source reports are not matched: {mismatch}")


def _groups(keys: list[tuple[str, str]]) -> dict[str, list[tuple[str, str]]]:
    return {
        "synthetic": [key for key in keys if key[0] == "synthetic"],
        "real": [key for key in keys if key[0] == "real"],
        "mixed_macro_1to1": list(keys),
    }


def _summaries(
    values: dict[str, list[float]], *, bootstrap_samples: int, seed_base: int
) -> dict[str, Any]:
    return {
        metric: summarize(
            rows,
            bootstrap_samples=int(bootstrap_samples),
            seed=int(seed_base) + metric_index * 101,
        )
        for metric_index, (metric, rows) in enumerate(values.items())
    }


def summarize_2x2(
    all_report: dict[str, Any],
    first_report: dict[str, Any],
    *,
    bootstrap_samples: int,
) -> dict[str, Any]:
    _assert_matched(all_report, first_report)
    all_rows = _records(all_report)
    first_rows = _records(first_report)
    if set(all_rows) != set(first_rows):
        raise RuntimeError("2x2 source reports selected different objects")

    result: dict[str, Any] = {}
    keys = sorted(all_rows)
    for group_index, (group, group_keys) in enumerate(_groups(keys).items()):
        if not group_keys:
            raise RuntimeError(f"2x2 report group is empty: {group}")
        values = {
            label: {metric: [] for metric in METRICS}
            for label in (
                "posed_increment_all",
                "posed_increment_first",
                "interaction_first_minus_all",
                "full_first_minus_all",
                "lora_only_first_minus_all",
            )
        }
        for key in group_keys:
            all_increment = paired_improvements(
                all_rows[key]["correct"], all_rows[key]["lora_only"]
            )
            first_increment = paired_improvements(
                first_rows[key]["correct"], first_rows[key]["lora_only"]
            )
            full_comparison = paired_improvements(
                first_rows[key]["correct"], all_rows[key]["correct"]
            )
            lora_comparison = paired_improvements(
                first_rows[key]["lora_only"], all_rows[key]["lora_only"]
            )
            for metric in METRICS:
                values["posed_increment_all"][metric].append(all_increment[metric])
                values["posed_increment_first"][metric].append(first_increment[metric])
                values["interaction_first_minus_all"][metric].append(
                    first_increment[metric] - all_increment[metric]
                )
                values["full_first_minus_all"][metric].append(full_comparison[metric])
                values["lora_only_first_minus_all"][metric].append(
                    lora_comparison[metric]
                )
        result[group] = {
            label: {
                "positive_meaning": (
                    "first-view Stock increases the marginal posed-DINO gain"
                    if label == "interaction_first_minus_all"
                    else "named left side is better"
                ),
                "metrics": _summaries(
                    metric_values,
                    bootstrap_samples=bootstrap_samples,
                    seed_base=20260811 + group_index * 10007 + label_index * 1009,
                ),
            }
            for label_index, (label, metric_values) in enumerate(values.items())
        }
    return result


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def main() -> None:
    args = make_parser().parse_args()
    if int(args.bootstrap_samples) <= 0:
        raise ValueError("bootstrap_samples must be positive")
    all_path, all_report = _load(args.all_report)
    first_path, first_report = _load(args.first_report)
    if report_context_mode(all_report) != "all":
        raise RuntimeError("--all_report is not the all-view Stock-context arm")
    if report_context_mode(first_report) != "first":
        raise RuntimeError("--first_report is not the first-view Stock-context arm")
    summary = summarize_2x2(
        all_report,
        first_report,
        bootstrap_samples=int(args.bootstrap_samples),
    )
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    report = {
        "format": FORMAT,
        "passed": True,
        "formal": False,
        "training_overlap": True,
        "checkpoint_step": int(all_report["run_config"]["checkpoint_step"]),
        "object_count": int(all_report["object_count"]),
        "source_reports": {
            "all": {"path": str(all_path), "sha256": all_report["report_sha256"]},
            "first": {
                "path": str(first_path),
                "sha256": first_report["report_sha256"],
            },
        },
        "controlled_variable": (
            "Stock SLat non-spatial cross-attention receives all selected views versus "
            "only the first selected view; posed-DINO receives all views in both arms"
        ),
        "summary": summary,
        "scope_guard": (
            "Both source reports consume the mixed SLat training cache. This is a matched "
            "mechanism/development experiment, not held-out generalization evidence."
        ),
    }
    body = dict(report)
    report["report_sha256"] = canonical_json_sha256(body)
    _atomic_json(output_dir / "report.json", report)

    lines = [
        "Mixed no-VGGT SLat Stock-context x posed-DINO 2x2",
        "=" * 58,
        "passed: true",
        "formal: false",
        "training overlap: true",
        f"matched checkpoint step: {report['checkpoint_step']}",
        f"objects: {report['object_count']}",
        "positive interaction = first-view Stock increased posed-DINO marginal gain",
        "positive full_first_minus_all = first-view Full is absolutely better",
        "",
    ]
    for group in GROUPS:
        lines.append(f"[{group}]")
        for label in (
            "posed_increment_all",
            "posed_increment_first",
            "interaction_first_minus_all",
            "full_first_minus_all",
            "lora_only_first_minus_all",
        ):
            metrics = summary[group][label]["metrics"]
            lines.append(
                f"{label}: chamfer={metrics['chamfer_l1']['mean']:+.8f} "
                f"f@0.02={metrics['fscore_0p02']['mean']:+.8f} "
                f"normal={metrics['normal_consistency']['mean']:+.8f}"
            )
        lines.append("")
    lines.append(report["scope_guard"])
    text = "\n".join(lines) + "\n"
    (output_dir / "summary.txt").write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
