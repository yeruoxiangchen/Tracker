#!/usr/bin/env python3

"""Summarize heimei coverage-sweep prepared/recon/mesh-eval reports."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from common import DEFAULT_OUTPUT_ROOT, ensure_dir, write_json


def read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else None


def get_nested(data: Optional[Dict[str, Any]], *keys: str) -> Any:
    cur: Any = data
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def row_for_case(output_root: Path, case_name: str) -> Dict[str, Any]:
    prepared = read_json(output_root / "runs" / case_name / "prepared_sample.json")
    recon = read_json(output_root / "runs" / case_name / "recon_generation_report.json")
    mesh = read_json(output_root / "mesh_quality" / case_name / "mesh_quality_report.json")

    rebuild_summary = get_nested(recon, "rebuild_summary") or {}
    surface = get_nested(mesh, "surface_metrics") or {}
    pred_basic = get_nested(mesh, "pred_basic") or {}

    return {
        "case_name": case_name,
        "trajectory_mode": get_nested(prepared, "trajectory_mode"),
        "max_frames": get_nested(prepared, "max_frames"),
        "candidate_count": get_nested(prepared, "candidate_coverage", "candidate_count"),
        "candidate_azimuth_span_deg": get_nested(prepared, "candidate_coverage", "candidate_azimuth_span_deg"),
        "selected_count": get_nested(prepared, "selected_coverage", "selected_count"),
        "selected_azimuth_span_deg": get_nested(prepared, "selected_coverage", "selected_azimuth_span_deg"),
        "selected_elevation_range_deg": get_nested(prepared, "selected_coverage", "selected_elevation_range_deg"),
        "selected_mask_area_mean": get_nested(prepared, "selected_coverage", "selected_mask_area_mean"),
        "selected_names": ",".join(get_nested(prepared, "selected_frames") or []),
        "contact_sheet": get_nested(prepared, "contact_sheet"),
        "recon_mode": get_nested(recon, "mode"),
        "recon_output_dir": get_nested(recon, "recon_output_dir"),
        "mesh_path": get_nested(recon, "mesh_path"),
        "selected_seed": rebuild_summary.get("selected_seed") if isinstance(rebuild_summary, dict) else None,
        "recon_candidate_count": rebuild_summary.get("candidate_count") if isinstance(rebuild_summary, dict) else None,
        "pred_vertices": pred_basic.get("vertices") if isinstance(pred_basic, dict) else None,
        "pred_faces": pred_basic.get("faces") if isinstance(pred_basic, dict) else None,
        "pred_bbox_diag": pred_basic.get("bbox_diag") if isinstance(pred_basic, dict) else None,
        "chamfer_l1": surface.get("chamfer_l1") if isinstance(surface, dict) else None,
        "pred_to_gt_mean": surface.get("pred_to_gt_mean") if isinstance(surface, dict) else None,
        "gt_to_pred_mean": surface.get("gt_to_pred_mean") if isinstance(surface, dict) else None,
        "fscore_0p05": surface.get("fscore_0p05") if isinstance(surface, dict) else None,
        "fscore_0p10": surface.get("fscore_0p10") if isinstance(surface, dict) else None,
        "prepared_report": str(output_root / "runs" / case_name / "prepared_sample.json"),
        "recon_report": str(output_root / "runs" / case_name / "recon_generation_report.json")
        if (output_root / "runs" / case_name / "recon_generation_report.json").exists()
        else None,
        "mesh_quality_report": str(output_root / "mesh_quality" / case_name / "mesh_quality_report.json")
        if (output_root / "mesh_quality" / case_name / "mesh_quality_report.json").exists()
        else None,
    }


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def parse_case_names(text: str) -> List[str]:
    names = [item.strip() for item in text.split(",") if item.strip()]
    if not names:
        raise ValueError("--case_names cannot be empty")
    return names


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--sweep_name", required=True)
    parser.add_argument("--case_names", required=True, help="Comma-separated case names")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    case_names = parse_case_names(args.case_names)
    rows = [row_for_case(output_root, case_name) for case_name in case_names]
    out_dir = ensure_dir(output_root / "coverage_sweeps" / args.sweep_name)
    write_json(out_dir / "summary.json", {"sweep_name": args.sweep_name, "cases": rows})
    write_csv(out_dir / "summary.csv", rows)
    print(
        json.dumps(
            {
                "summary_json": str(out_dir / "summary.json"),
                "summary_csv": str(out_dir / "summary.csv"),
                "case_count": len(rows),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
