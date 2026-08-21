#!/usr/bin/env python3
"""Dora-Bench-300 binding for the proven Omni exact-model-O evaluator.

The tensor preparation, SS30K/SLat30K inference contract, GT transformation,
surface sampling, CD and F-score math are intentionally shared.  This module
only installs Dora-specific artifact formats and user-facing summary labels.
"""

from __future__ import annotations

import contextlib
import io
from pathlib import Path

from manual_mesh_reconstruction.common import atomic_json, load_json
from pose_point_depth_mv.dataset_tools.build_reconviagen_dorabench300_benchmark import (
    FINAL_FORMAT,
)
from pose_point_depth_mv import evaluate_omni200_ss30k_slat30k as base


base.BENCHMARK_MANIFEST_FORMAT = FINAL_FORMAT
base.PREPARE_FORMAT = "reconviagen.dorabench_dora300_exact_model_o_runtime.v1"
base.METRIC_OBJECT_FORMAT = "reconviagen.dorabench_dora300_ss30k_slat30k_metric.v1"
base.METRIC_WORKER_FORMAT = (
    "reconviagen.dorabench_dora300_ss30k_slat30k_metric_worker.v1"
)
base.METRIC_AGGREGATE_FORMAT = (
    "reconviagen.dorabench_dora300_ss30k_slat30k_metric_aggregate.v1"
)

_base_aggregate = base.cmd_aggregate


def dora_aggregate(args) -> None:
    # Suppress only the inherited Omni label; all validation/errors propagate.
    with contextlib.redirect_stdout(io.StringIO()):
        _base_aggregate(args)
    report_path = Path(args.output_dir).expanduser().resolve() / "report.json"
    report = load_json(report_path)
    report["benchmark"] = "Dora-Bench-300 registered reproduction"
    report["complexity_level_count"] = report.pop("category_count")
    report["by_complexity_level"] = report.pop("by_category")
    report["scope_guard"] = (
        "SS30K+SLat30K only. ReconViaGen original is neither loaded nor run. "
        "The frozen TRELLIS trajectory contains 40 cameras and model inputs are "
        "exactly views 0,9,19,29. Metrics use 100k surface samples and F-score radius 0.1."
    )
    atomic_json(report_path, report)
    summary = [
        "Dora-Bench-300 SS30K+SLat30K evaluation",
        "========================================",
        f"objects: {report['object_count']} complexity levels: {report['complexity_level_count']}",
        (
            "Chamfer Distance: "
            f"mean={report['chamfer_distance']['mean']:.8f} "
            f"median={report['chamfer_distance']['median']:.8f}"
        ),
        (
            "F-score@0.1: "
            f"mean={report['fscore']['mean']:.8f} "
            f"median={report['fscore']['median']:.8f}"
        ),
        "ReconViaGen original loaded/run: no",
        f"report: {report_path}",
    ]
    (report_path.parent / "summary.txt").write_text(
        "\n".join(summary) + "\n", encoding="utf-8"
    )
    print("\n".join(summary))


base.cmd_aggregate = dora_aggregate


def main() -> None:
    base.main()


if __name__ == "__main__":
    main()
