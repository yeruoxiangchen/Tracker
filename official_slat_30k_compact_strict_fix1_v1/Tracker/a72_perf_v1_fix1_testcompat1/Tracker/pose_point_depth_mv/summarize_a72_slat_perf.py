#!/usr/bin/env python3
"""Summarize optimizer-boundary wall times emitted by the A72 perf runtime."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
from typing import Any


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        raise ValueError("percentile requires values")
    ordered = sorted(values)
    position = (len(ordered) - 1) * float(fraction)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize(
    report: dict[str, Any],
    *,
    start_step: int,
    end_step: int,
    discard_first: int,
) -> dict[str, Any]:
    rows = [
        row
        for row in report.get("history", [])
        if isinstance(row, dict)
        and int(start_step) < int(row.get("step", -1)) <= int(end_step)
        and isinstance(row.get("optimizer_step_wall_seconds"), (int, float))
    ]
    rows = rows[int(discard_first) :]
    values = [float(row["optimizer_step_wall_seconds"]) for row in rows]
    if not values or not all(math.isfinite(value) and value > 0 for value in values):
        raise ValueError("report has no finite positive timed optimizer steps")
    optimization = dict(report.get("model_summary", {}).get("optimization", {}))
    global_batch = int(optimization.get("global_effective_batch", 0))
    if global_batch <= 0:
        raise ValueError("report lacks a positive global effective batch")
    total_seconds = sum(values)
    return {
        "format": "pose_point_depth_mv.a72_slat_perf_summary.v1",
        "passed": True,
        "source_step_interval": [int(start_step), int(end_step)],
        "discarded_initial_timed_steps": int(discard_first),
        "timed_optimizer_steps": len(values),
        "first_timed_step": int(rows[0]["step"]),
        "last_timed_step": int(rows[-1]["step"]),
        "global_effective_batch": global_batch,
        "total_timed_seconds": total_seconds,
        "seconds_per_optimizer_step": {
            "mean": statistics.fmean(values),
            "median": statistics.median(values),
            "p10": percentile(values, 0.10),
            "p90": percentile(values, 0.90),
            "min": min(values),
            "max": max(values),
        },
        "optimizer_steps_per_hour": len(values) * 3600.0 / total_seconds,
        "global_samples_per_second": len(values) * global_batch / total_seconds,
        "global_samples_per_hour": len(values) * global_batch * 3600.0 / total_seconds,
        "runtime_performance": report.get("model_summary", {}).get(
            "runtime_performance"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--start_step", type=int, required=True)
    parser.add_argument("--end_step", type=int, required=True)
    parser.add_argument("--discard_first", type=int, default=2)
    args = parser.parse_args()
    if args.end_step <= args.start_step:
        raise ValueError("end_step must exceed start_step")
    if args.discard_first < 0:
        raise ValueError("discard_first must be non-negative")
    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    result = summarize(
        report,
        start_step=args.start_step,
        end_step=args.end_step,
        discard_first=args.discard_first,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
