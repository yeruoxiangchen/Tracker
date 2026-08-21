#!/usr/bin/env python3
"""Read-only progress summary for the resumable Mixed-10k build."""

from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
from typing import Any


EXTRACT_REPORT = "_EXTRACT_REPORT.json"
COMPLETE_MARKER = "_WORKER_COMPLETE.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_inputs", required=True)
    parser.add_argument("--omni_extract_root", required=True)
    parser.add_argument("--source_freeze", required=True)
    parser.add_argument("--render_root", required=True)
    parser.add_argument("--final_root", required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file() or path.stat().st_size <= 0:
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def remote_categories(path: Path) -> list[str]:
    if not path.is_file():
        return []
    result = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if not value:
            continue
        name = PurePosixPath(value).name
        if name.endswith(".tar.gz"):
            result.append(name[: -len(".tar.gz")])
    return sorted(result)


def extracted_categories(root: Path) -> tuple[list[str], int, int]:
    categories = root / "categories"
    names: list[str] = []
    accepted = 0
    rejected = 0
    if not categories.is_dir():
        return names, accepted, rejected
    for report_path in sorted(categories.glob(f"*/{EXTRACT_REPORT}")):
        report = load_json(report_path)
        if report is None:
            continue
        names.append(report_path.parent.name)
        accepted += int(report.get("object_count", 0))
        rejected += int(report.get("rejected_object_count", 0))
    return names, accepted, rejected


def render_progress(root: Path, source: str, expected: int) -> dict[str, Any]:
    source_root = root / source
    markers = (
        sorted(source_root.glob(f"shard_*/{COMPLETE_MARKER}"))
        if source_root.is_dir()
        else []
    )
    manifests = (
        sorted(source_root.glob("shard_*/manifest.json"))
        if source_root.is_dir()
        else []
    )
    incomplete = (
        sorted(source_root.glob("shard_*.incomplete_*"))
        if source_root.is_dir()
        else []
    )
    return {
        "expected_shards": expected,
        "complete_markers": len(markers),
        "manifests": len(manifests),
        "preserved_incomplete_directories": len(incomplete),
    }


def main() -> None:
    args = parse_args()
    source_inputs = Path(args.source_inputs).expanduser().resolve()
    extract_root = Path(args.omni_extract_root).expanduser().resolve()
    source_freeze = Path(args.source_freeze).expanduser().resolve()
    render_root = Path(args.render_root).expanduser().resolve()
    final_root = Path(args.final_root).expanduser().resolve()

    expected = remote_categories(source_inputs / "omni_remote_paths.txt")
    extracted, accepted, rejected = extracted_categories(extract_root)
    missing = sorted(set(expected) - set(extracted))
    unexpected = sorted(set(extracted) - set(expected))

    source_report = load_json(source_freeze / "source_report.json")
    plan = load_json(source_freeze / "build_plan.json")
    final_report = load_json(final_root / "report.json")

    objaverse_shards = (
        int(plan["objaverse"]["shard_count"])
        if plan is not None and "objaverse" in plan
        else 16
    )
    omni_shards = (
        int(plan["omni"]["shard_count"])
        if plan is not None and "omni" in plan
        else 8
    )
    summary = {
        "source_inputs": {
            "complete": (source_inputs / "input_report.json").is_file(),
            "expected_omni_categories": len(expected),
        },
        "omni_extraction": {
            "complete_categories": len(extracted),
            "accepted_objects_from_reports": accepted,
            "rejected_objects_from_reports": rejected,
            "missing_categories": missing,
            "unexpected_categories": unexpected,
            "preserved_staging_directories": len(
                list((extract_root / ".staging").glob("*"))
            )
            if (extract_root / ".staging").is_dir()
            else 0,
        },
        "source_freeze": {
            "complete": source_report is not None and plan is not None,
            "passed": None if source_report is None else source_report.get("passed"),
            "objaverse_objects": (
                None
                if source_report is None
                else source_report.get("objaverse", {}).get("available_count")
            ),
            "omni_objects": (
                None
                if source_report is None
                else source_report.get("omni", {}).get("object_count")
            ),
            "omni_rejected_objects": (
                None
                if source_report is None
                else source_report.get("omni", {}).get("rejected_object_count")
            ),
        },
        "render": {
            "objaverse": render_progress(
                render_root,
                "objaverse",
                objaverse_shards,
            ),
            "omni": render_progress(render_root, "omni", omni_shards),
        },
        "final_dataset": {
            "complete": final_report is not None,
            "passed": None if final_report is None else final_report.get("passed"),
            "object_count": (
                None if final_report is None else final_report.get("object_count")
            ),
        },
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
