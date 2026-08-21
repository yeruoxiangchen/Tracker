#!/usr/bin/env python3
"""Copy four-way review OBJ files into compact, case-grouped directories."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from .bunny_review.common import atomic_json, binding, sha256_file


METHOD_NAMES = {
    "reference": "01_GT.obj",
    "native_full": "02_Native_Full.obj",
    "stock": "03_Stock.obj",
    "pixal3d_official_final": "04_Pixal3D.obj",
}


def copy_verified(source: Path, destination: Path) -> dict[str, object]:
    source = source.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    if sha256_file(source) != sha256_file(destination):
        raise RuntimeError(f"copied file hash changed: {source} -> {destination}")
    return binding(destination)


def run(args: argparse.Namespace) -> None:
    review_root = args.review_root.resolve()
    output_dir = args.output_dir.resolve()
    reports = sorted(review_root.glob("*/report.json"))
    if len(reports) != int(args.expected_cases):
        raise RuntimeError(
            f"expected {args.expected_cases} review reports, found {len(reports)}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = []
    for position, report_path in enumerate(reports, start=1):
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("complete") is not True:
            raise RuntimeError(f"incomplete review: {report_path}")
        comparison = report.get("comparisons", {}).get(args.review_mode)
        if not isinstance(comparison, dict):
            raise RuntimeError(
                f"review lacks mode={args.review_mode}: {report_path}"
            )
        config = report["run_config"]
        uid = str(config["uid"])
        source = str(config["source"])
        view_count = int(config["view_count"])
        short_uid = str(config["object_uid"]).replace("objaverse_", "")[:12]
        case_name = f"{position:02d}_{source}_v{view_count:02d}_{short_uid}"
        case_dir = output_dir / case_name
        method_rows = {
            str(row["method_id"]): row for row in comparison["methods"]
        }
        if set(method_rows) != set(METHOD_NAMES):
            raise RuntimeError(
                f"unexpected method set in {report_path}: {set(method_rows)}"
            )
        copied_methods = {}
        for method_id, filename in METHOD_NAMES.items():
            copied_methods[method_id] = copy_verified(
                Path(method_rows[method_id]["mesh"]["path"]),
                case_dir / filename,
            )
        sheet = copy_verified(
            Path(comparison["normal_contact_sheet"]["path"]),
            case_dir / "00_四路简图.png",
        )
        cases.append(
            {
                "case_position": position,
                "case_name": case_name,
                "uid": uid,
                "source": source,
                "view_count": view_count,
                "review_report": binding(report_path),
                "contact_sheet": sheet,
                "methods": copied_methods,
            }
        )
    result = {
        "format": "pose_point_depth_mv.fourway_review_assets.v1",
        "complete": True,
        "review_mode": args.review_mode,
        "case_count": len(cases),
        "obj_count": sum(len(case["methods"]) for case in cases),
        "cases": cases,
    }
    atomic_json(output_dir / "资产清单.json", result)
    print(
        json.dumps(
            {
                "passed": True,
                "output_dir": str(output_dir),
                "case_count": result["case_count"],
                "obj_count": result["obj_count"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--review_mode", default="rigid_pose_gt")
    parser.add_argument("--expected_cases", type=int, default=4)
    return parser


if __name__ == "__main__":
    run(make_parser().parse_args())
