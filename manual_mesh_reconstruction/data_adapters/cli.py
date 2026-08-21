#!/usr/bin/env python3
"""Unified dataset-path to raw-cache/runtime-O adapter CLI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from manual_mesh_reconstruction.common import atomic_json
from manual_mesh_reconstruction.data_adapters import ADAPTER_FORMAT
from manual_mesh_reconstruction.data_adapters.common import (
    utc_now,
    validate_reusable_adapter_report,
)


def detect_type(path: Path, *, clip_sequence: str | None) -> str:
    path = path.resolve(strict=True)
    if (path / "annotation.pbdata").is_file() or (
        clip_sequence and (path / "clips" / clip_sequence / "annotation.pbdata").is_file()
    ):
        return "objectron"
    if (
        (path / "poses.txt").is_file()
        or path.name in {"runtime", "data", "masks"}
        or (path / "reconstruction_report.json").is_file()
        or (path / "01_raw_cache/raw_cache_report.json").is_file()
    ):
        return "phone"
    return "colmap"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument(
        "--dataset-type",
        choices=("auto", "phone", "colmap", "objectron"),
        default="auto",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--selected-view-count", type=int, default=8)
    parser.add_argument(
        "--frame-selection",
        choices=("time_uniform", "random"),
        default="time_uniform",
    )
    parser.add_argument("--random-seed", type=int, default=20260819)
    parser.add_argument("--session-id", default="")
    parser.add_argument(
        "--colmap-mode",
        choices=("auto", "reuse", "rebuild"),
        default="auto",
        help="auto reuses a complete model and otherwise rebuilds; reuse/rebuild are strict",
    )
    parser.add_argument("--colmap-sparse", default="")
    parser.add_argument("--colmap-bin", default="colmap")
    parser.add_argument(
        "--colmap-matcher",
        choices=("sequential", "exhaustive"),
        default="sequential",
    )
    parser.add_argument("--colmap-use-foreground-masks", action="store_true")
    parser.add_argument("--colmap-cpu", action="store_true")
    parser.add_argument("--objectron-clip", default="")
    parser.add_argument("--objectron-object-id", type=int, default=0)
    parser.add_argument(
        "--objectron-o",
        choices=("pose_mask", "true_object_pose"),
        default="pose_mask",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    dataset = Path(args.dataset_path).expanduser().resolve(strict=True)
    output = Path(args.output_dir).expanduser().resolve()
    report_path = output / "adapter_report.json"
    if report_path.is_file():
        if not args.resume:
            raise FileExistsError(f"adapter output exists; pass --resume: {report_path}")
        report = validate_reusable_adapter_report(report_path)
        print(json.dumps({"reused": True, **report}, indent=2, ensure_ascii=False))
        return report
    dataset_type = (
        detect_type(dataset, clip_sequence=args.objectron_clip or None)
        if args.dataset_type == "auto"
        else str(args.dataset_type)
    )
    raw_report = output / "raw_cache/raw_cache_report.json"
    true_runtime = output / "runtime_true_object_pose/runtime_input_manifest.json"
    if args.dry_run:
        plan = {
            "format": ADAPTER_FORMAT,
            "dry_run": True,
            "dataset_type": dataset_type,
            "dataset_path": str(dataset),
            "output_dir": str(output),
            "selected_view_count": int(args.selected_view_count),
            "frame_selection": str(args.frame_selection),
            "colmap_mode": str(args.colmap_mode) if dataset_type == "colmap" else None,
            "objectron_o": str(args.objectron_o) if dataset_type == "objectron" else None,
            "expected_raw_cache_report": str(raw_report),
            "expected_runtime_input_manifest": (
                str(true_runtime)
                if dataset_type == "objectron" and args.objectron_o == "true_object_pose"
                else None
            ),
            "passed": True,
        }
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        return plan
    if output.exists() and not args.resume:
        raise FileExistsError(f"adapter output exists: {output}")
    output.mkdir(parents=True, exist_ok=True)
    if dataset_type == "phone":
        from manual_mesh_reconstruction.data_adapters.phone import adapt

        result = adapt(
            input_path=dataset,
            output_dir=output / "raw_cache",
            selected_view_count=int(args.selected_view_count),
            selection_policy=str(args.frame_selection),
            random_seed=int(args.random_seed),
            session_id=args.session_id or None,
        )
    elif dataset_type == "colmap":
        from manual_mesh_reconstruction.data_adapters.colmap import adapt

        result = adapt(
            input_path=dataset,
            output_dir=output / "raw_cache",
            selected_view_count=int(args.selected_view_count),
            selection_policy=str(args.frame_selection),
            random_seed=int(args.random_seed),
            colmap_mode=str(args.colmap_mode),
            colmap_sparse=(
                Path(args.colmap_sparse).expanduser().resolve(strict=True)
                if args.colmap_sparse
                else None
            ),
            colmap_bin=str(args.colmap_bin),
            matcher=str(args.colmap_matcher),
            use_foreground_masks=bool(args.colmap_use_foreground_masks),
            use_gpu=not bool(args.colmap_cpu),
            resume=bool(args.resume),
        )
    elif dataset_type == "objectron":
        from manual_mesh_reconstruction.data_adapters.objectron import adapt

        result = adapt(
            input_path=dataset,
            output_dir=output / "raw_cache",
            selected_view_count=int(args.selected_view_count),
            selection_policy=str(args.frame_selection),
            random_seed=int(args.random_seed),
            clip_sequence=args.objectron_clip or None,
            official_object_id=int(args.objectron_object_id),
            o_mode=str(args.objectron_o),
        )
    else:
        raise AssertionError(dataset_type)
    report = {
        "format": ADAPTER_FORMAT,
        "created_at_utc": utc_now(),
        "passed": True,
        "dataset_type": dataset_type,
        "dataset_path": str(dataset),
        "output_dir": str(output),
        "selected_view_count": int(args.selected_view_count),
        "frame_selection": str(args.frame_selection),
        "random_seed": int(args.random_seed),
        "colmap_mode": str(args.colmap_mode) if dataset_type == "colmap" else None,
        "objectron_o": str(args.objectron_o) if dataset_type == "objectron" else None,
        **result,
    }
    atomic_json(report_path, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()

