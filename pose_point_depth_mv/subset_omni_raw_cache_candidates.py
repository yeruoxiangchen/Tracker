#!/usr/bin/env python3
"""Freeze one Omni raw-cache object to an explicit candidate-frame subset.

This is a diagnostic input derivative: camera/image rows are restricted to the
named candidates, while the already reconstructed COLMAP sparse cloud is kept
unchanged.  Downstream point-mask runtime-O may therefore select eight views
from exactly this candidate set without feeding every candidate to the model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


FORMAT = "pose_point_depth_mv.omni_real_video_raw_cache.v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--raw_cache_report", type=Path, required=True)
    parser.add_argument("--candidate_report", type=Path, required=True)
    parser.add_argument("--object", required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()

    source_report_path = args.raw_cache_report.expanduser().resolve()
    candidate_report_path = args.candidate_report.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    report_path = output_dir / "raw_cache_report.json"
    if report_path.is_file():
        existing = json.loads(report_path.read_text(encoding="utf-8"))
        if existing.get("passed") is True:
            print(json.dumps(existing, indent=2, ensure_ascii=False))
            return
    if output_dir.exists():
        raise RuntimeError(f"partial immutable output exists: {output_dir}")

    source_report = json.loads(source_report_path.read_text(encoding="utf-8"))
    candidate_report = json.loads(candidate_report_path.read_text(encoding="utf-8"))
    candidates = [str(value) for value in candidate_report["source_frame_names"]]
    if len(candidates) != 64 or len(set(candidates)) != 64:
        raise RuntimeError("candidate report must bind exactly 64 unique frames")
    category, object_id = args.object.split(":", 1)
    rows = [
        row for row in source_report["objects"]
        if str(row["category"]) == category and str(row["object_id"]) == object_id
    ]
    if len(rows) != 1:
        raise RuntimeError(f"object selector matched {len(rows)} rows")
    source_row = rows[0]
    source_cache = Path(source_row["cache_npz"]).resolve()

    output_dir.mkdir(parents=True)
    cache_path = output_dir / "objects" / category / object_id / "raw_camera_point_cache.npz"
    cache_path.parent.mkdir(parents=True)
    with np.load(source_cache, allow_pickle=False) as source:
        source_names = [str(value) for value in source["frame_name"].tolist()]
        missing = [name for name in candidates if name not in source_names]
        if missing:
            raise RuntimeError(f"candidate frames missing from raw cache: {missing[:3]}")
        indices = np.asarray([source_names.index(name) for name in candidates], dtype=np.int64)
        payload: dict[str, np.ndarray] = {}
        view_fields = {"frame_name", "image_id", "camera_id", "K", "T_W2C"}
        for key in source.files:
            value = np.asarray(source[key])
            payload[key] = value[indices] if key in view_fields else value
        np.savez_compressed(cache_path, **payload)

    camera_by_name = {
        str(row["frame_name"]): row for row in source_row.get("cameras", [])
    }
    derived_row = dict(source_row)
    derived_row.update(
        {
            "object_root": str(cache_path.parent.resolve()),
            "cache_npz": str(cache_path.resolve()),
            "registered_pair_count": 64,
            "cameras": [camera_by_name[name] for name in candidates],
            "candidate_subset": {
                "policy": "frozen_uniform64_from_registered_plant_views",
                "source_frame_names": candidates,
                "source_raw_cache": str(source_cache),
                "source_raw_cache_sha256": _sha256(source_cache),
                "candidate_report": str(candidate_report_path),
                "candidate_report_sha256": _sha256(candidate_report_path),
                "sparse_cloud_policy": (
                    "reuse frozen COLMAP P_W; only RGB/mask/pose candidate rows are "
                    "restricted to 64 before downstream 64-to-8 selection"
                ),
            },
        }
    )
    derived_report = {
        "format": source_report["format"],
        "created_from": str(source_report_path),
        "source_report_sha256": _sha256(source_report_path),
        "output_dir": str(output_dir),
        "category_count": 1,
        "object_count": 1,
        "objects": [derived_row],
        "authoritative_colmap": source_report.get("authoritative_colmap"),
        "excluded_inputs": source_report.get("excluded_inputs"),
        "alignment_passed": source_report.get("alignment_passed", True),
        "training_ready": False,
        "scope_guard": (
            "Development Plant point-mask candidate ablation. Exactly 64 RGB/mask/pose "
            "rows are visible to view selection; the model receives only the selected 8."
        ),
        "passed": True,
    }
    _write_json(report_path, derived_report)
    print(json.dumps(derived_report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
