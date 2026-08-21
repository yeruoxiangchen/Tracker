#!/usr/bin/env python3
"""Copy final SS30K+SLat30K qualitative outputs into the isolated package."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any


FORMAT = "manual_mesh_reconstruction.existing_output_migration.v1"
DEFAULT_SOURCE = Path(__file__).resolve().parents[1] / "pose_point_depth_mv/outputs2"
DEFAULT_DESTINATION = Path(__file__).resolve().parent / "output/已有SS30K_SLat30K结果"

CANONICAL = (
    "OmniPlant012冻结8视图_ReconViaGen_vs_SS30K_SLat30K_相机轮廓_20260819_v1",
    "ObjectronCamera_三选帧_ReconViaGen_vs_SS30K_SLat30K_双RuntimeO轮廓_20260819_v2",
    "ObjectronShoe_obj0_三选帧_ReconViaGen_vs_SS30K_SLat30K_双RuntimeO轮廓_20260819_v1",
    "真实采集5组_AR_COLMAP六分支_SS30K_SLat30K_runtimeO正确轮廓_20260819_v2",
    "CoarseModel_heimei_snoopy2_指定帧_COLMAPPose_ReconViaGen_vs_SS30K_SLat30K_runtimeO轮廓_20260819_v3",
    "CoarseModel_snoopy_指定8帧_COLMAPPose_ReconViaGen_vs_SS30K_SLat30K_runtimeO轮廓_20260819_v1",
    "CoarseModel_snoopy_轨迹均匀4帧16帧_ReconViaGen_vs_SS30K_SLat30K_20260819_v1",
    "ProObjaverse30K_Dev64_CminusR优势最明显Top4_GT_ReconViaGen_SS30K_SLat30K_输入Mask_20260819_v1",
)

EXCLUDED = {
    "ObjectronCamera_三选帧_ReconViaGen_vs_SS30K_SLat30K_双RuntimeO轮廓_20260819_v2_cpu_prepare_partial_20260819T0337Z": "partial preparation tree",
    "真实采集5组_AR与COLMAPPose_ReconViaGen_vs_SS30K_SLat30K_三选帧轮廓_20260819_v1": "superseded by physical runtime-O contour v2",
    "CoarseModel_heimei_snoopy2_指定帧_COLMAPPose_ReconViaGen_vs_SS30K_SLat30K_runtimeO轮廓_20260819_v1": "failed/incomplete predecessor",
    "CoarseModel_heimei_snoopy2_指定帧_COLMAPPose_ReconViaGen_vs_SS30K_SLat30K_runtimeO轮廓_20260819_v2": "superseded by final v3",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _control(path: Path) -> tuple[Path, dict[str, Any]]:
    candidates = [path / "report.json", path / "summary.json"]
    existing = [candidate for candidate in candidates if candidate.is_file()]
    if len(existing) != 1:
        raise RuntimeError(f"canonical output has no unique control report: {path}")
    payload = json.loads(existing[0].read_text(encoding="utf-8"))
    if payload.get("passed") is not True:
        raise RuntimeError(f"canonical output did not pass: {existing[0]}")
    return existing[0], payload


def _tree_record(root: Path) -> dict[str, Any]:
    files = sorted(path for path in root.rglob("*") if path.is_file())
    digest = hashlib.sha256()
    total = 0
    for path in files:
        relative = path.relative_to(root).as_posix()
        size = path.stat().st_size
        file_hash = sha256_file(path)
        digest.update(f"{relative}\0{size}\0{file_hash}\n".encode("utf-8"))
        total += size
    return {
        "regular_file_count": len(files),
        "total_bytes": total,
        "tree_sha256": digest.hexdigest(),
    }


def run(source: Path, destination: Path) -> dict[str, Any]:
    source = source.expanduser().resolve(strict=True)
    destination = destination.expanduser().resolve()
    if destination.exists():
        raise FileExistsError(destination)
    destination.mkdir(parents=True)
    records = []
    for name in CANONICAL:
        source_dir = source / name
        control, payload = _control(source_dir)
        target_dir = destination / name
        print(f"[migrate] {name}", flush=True)
        shutil.copytree(source_dir, target_dir, copy_function=shutil.copy2)
        source_tree = _tree_record(source_dir)
        target_tree = _tree_record(target_dir)
        if source_tree != target_tree:
            raise RuntimeError(f"copied output tree differs: {name}")
        target_control = target_dir / control.relative_to(source_dir)
        records.append(
            {
                "name": name,
                "source": str(source_dir),
                "destination": str(target_dir),
                "control_report": str(target_control),
                "control_report_sha256": sha256_file(target_control),
                "control_format": payload.get("format"),
                **target_tree,
            }
        )
    report = {
        "format": FORMAT,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "passed": True,
        "source_root": str(source),
        "destination_root": str(destination),
        "copied_directory_count": len(records),
        "regular_file_count": sum(row["regular_file_count"] for row in records),
        "total_bytes": sum(row["total_bytes"] for row in records),
        "directories": records,
        "excluded_noncanonical": EXCLUDED,
        "selection_rule": (
            "final PASS outputs using no-VGGT SS30K+SLat30K; partial, superseded "
            "coordinate-frame outputs, 2K, and with-VGGT outputs are excluded"
        ),
    }
    report_path = destination / "MIGRATION_MANIFEST.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"passed": True, "report": str(report_path)}, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=str(DEFAULT_SOURCE))
    parser.add_argument("--destination", default=str(DEFAULT_DESTINATION))
    args = parser.parse_args()
    run(Path(args.source), Path(args.destination))


if __name__ == "__main__":
    main()

