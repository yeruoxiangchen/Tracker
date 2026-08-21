#!/usr/bin/env python3
"""Freeze the label-qualified subset of an Omni real training split.

The runtime and adjudicated-alignment manifests intentionally retain all
source candidates.  Training, however, needs two passed manifests with exactly
the same object set.  This tool drops only rows that failed an already-frozen
input/alignment gate and emits paired manifests without recomputing geometry.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pose_point_depth_mv.dataset_tools.adjudicate_omni_real_mesh_alignment import (
    MANIFEST_FORMAT as ALIGNMENT_FORMAT,
)
from pose_point_depth_mv.dataset_tools.prepare_omni_real_runtime_inputs import (
    MANIFEST_FORMAT as RUNTIME_FORMAT,
)
from pose_point_depth_mv.dataset_tools.prepare_omni_real_video_cache import utc_now
from pose_point_depth_mv.omni_real_benchmark_common import atomic_json, load_json, sha256_file


REPORT_FORMAT = "pose_point_depth_mv.omni_real_native_training_subset.v1"


def object_key(row: dict[str, Any]) -> str:
    return f"{row['category']}:{row['object_id']}"


def _unique(rows: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    result = {object_key(row): row for row in rows}
    if len(result) != len(rows):
        raise RuntimeError(f"{label} contains duplicate object identities")
    return result


def freeze_subset(
    runtime_path: Path,
    alignment_path: Path,
    output_dir: Path,
    *,
    expected_source_objects: int,
    min_objects: int,
) -> dict[str, Any]:
    runtime_path = runtime_path.expanduser().resolve()
    alignment_path = alignment_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    runtime = load_json(runtime_path)
    alignment = load_json(alignment_path)
    if runtime.get("format") != RUNTIME_FORMAT:
        raise RuntimeError("unexpected runtime-O manifest format")
    if alignment.get("format") != ALIGNMENT_FORMAT:
        raise RuntimeError("unexpected adjudicated-alignment manifest format")
    runtime_rows = _unique(list(runtime.get("objects", [])), "runtime manifest")
    alignment_rows = _unique(list(alignment.get("objects", [])), "alignment manifest")
    if len(runtime_rows) != int(expected_source_objects):
        raise RuntimeError(
            f"runtime source count={len(runtime_rows)} != {expected_source_objects}"
        )
    if set(runtime_rows) != set(alignment_rows):
        raise RuntimeError("runtime/alignment source object sets differ")
    if runtime.get("failures") or alignment.get("failures"):
        raise RuntimeError("source manifests contain construction failures")
    if runtime.get("raw_cache_report_sha256") != alignment.get(
        "raw_cache_report_sha256"
    ):
        raise RuntimeError("runtime/alignment raw-cache bindings differ")

    accepted = sorted(
        key
        for key in runtime_rows
        if runtime_rows[key].get("passed") is True
        and alignment_rows[key].get("automatic_passed") is True
    )
    excluded = sorted(set(runtime_rows).difference(accepted))
    if len(accepted) < int(min_objects):
        raise RuntimeError(
            f"qualified training objects={len(accepted)} < minimum={min_objects}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    runtime_output = output_dir / "runtime_input_manifest.json"
    alignment_output = output_dir / "alignment_adjudicated.json"
    report_output = output_dir / "selection_report.json"
    source_bindings = {
        "runtime_manifest": str(runtime_path),
        "runtime_manifest_sha256": sha256_file(runtime_path),
        "alignment_manifest": str(alignment_path),
        "alignment_manifest_sha256": sha256_file(alignment_path),
    }
    if report_output.is_file():
        report = load_json(report_output)
        if report.get("format") != REPORT_FORMAT or any(
            report.get(key) != value for key, value in source_bindings.items()
        ):
            raise RuntimeError("existing training-subset binding differs")
        for key, path in (
            ("runtime_output_sha256", runtime_output),
            ("alignment_output_sha256", alignment_output),
        ):
            if not path.is_file() or report.get(key) != sha256_file(path):
                raise RuntimeError(f"frozen training-subset artifact changed: {path}")
        return report
    if any(output_dir.iterdir()):
        raise RuntimeError("partial unbound training-subset output exists")

    created = utc_now()
    runtime_subset = {
        **runtime,
        "created_at_utc": created,
        "source_runtime_manifest": str(runtime_path),
        "source_runtime_manifest_sha256": source_bindings[
            "runtime_manifest_sha256"
        ],
        "selected_object_count": len(accepted),
        "completed_object_count": len(accepted),
        "reused_objects": [],
        "objects": [runtime_rows[key] for key in accepted],
        "failures": [],
        "excluded_object_keys": excluded,
        "training_ready": False,
        "scope_guard": (
            "Label-qualified runtime-O subset only; model inputs and targets are not "
            "materialized here."
        ),
        "passed": True,
    }
    alignment_subset = {
        **alignment,
        "created_at_utc": created,
        "source_adjudicated_manifest": str(alignment_path),
        "source_adjudicated_manifest_sha256": source_bindings[
            "alignment_manifest_sha256"
        ],
        "selected_object_count": len(accepted),
        "completed_object_count": len(accepted),
        "source_automatic_pass_count": sum(
            int(alignment_rows[key].get("source_automatic_passed", False))
            for key in accepted
        ),
        "automatic_pass_count": len(accepted),
        "objects": [alignment_rows[key] for key in accepted],
        "failures": [],
        "excluded_object_keys": excluded,
        "training_ready": False,
        "scope_guard": (
            "Frozen robust-alignment training subset; no transform or quality policy "
            "was recomputed."
        ),
        "passed": True,
    }
    atomic_json(runtime_output, runtime_subset)
    atomic_json(alignment_output, alignment_subset)
    report = {
        "format": REPORT_FORMAT,
        "created_at_utc": created,
        **source_bindings,
        "expected_source_object_count": int(expected_source_objects),
        "minimum_training_object_count": int(min_objects),
        "qualified_object_count": len(accepted),
        "excluded_object_count": len(excluded),
        "excluded_object_keys": excluded,
        "runtime_output": str(runtime_output),
        "runtime_output_sha256": sha256_file(runtime_output),
        "alignment_output": str(alignment_output),
        "alignment_output_sha256": sha256_file(alignment_output),
        "object_set_equal": True,
        "geometry_recomputed": False,
        "quality_policy_recomputed": False,
        "training_ready": False,
        "passed": True,
    }
    atomic_json(report_output, report)
    return report


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime_input_manifest", required=True)
    parser.add_argument("--alignment_manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--expected_source_objects", type=int, default=500)
    parser.add_argument("--min_objects", type=int, default=450)
    return parser


def main() -> None:
    args = make_parser().parse_args()
    if int(args.expected_source_objects) <= 0 or int(args.min_objects) <= 0:
        raise ValueError("object-count gates must be positive")
    if int(args.min_objects) > int(args.expected_source_objects):
        raise ValueError("min_objects exceeds expected_source_objects")
    report = freeze_subset(
        Path(args.runtime_input_manifest),
        Path(args.alignment_manifest),
        Path(args.output_dir),
        expected_source_objects=int(args.expected_source_objects),
        min_objects=int(args.min_objects),
    )
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "qualified_object_count": report["qualified_object_count"],
                "excluded_object_count": report["excluded_object_count"],
                "output": str(Path(args.output_dir).expanduser().resolve()),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
