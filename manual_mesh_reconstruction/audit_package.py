#!/usr/bin/env python3
"""Freeze source provenance and package/output identity for the manual tool."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any


PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parent
MANIFEST = PACKAGE / "PACKAGE_MANIFEST.json"

COPIED_IMPLEMENTATIONS = {
    "runtime_o.py": "pose_point_depth_mv/dataset_tools/prepare_omni_real_runtime_inputs.py",
    "model_inputs.py": "pose_point_depth_mv/dataset_tools/prepare_omni_real_dino_only_model_inputs.py",
    "model_geometry.py": "pose_point_depth_mv/dataset_tools/prepare_omni_real_model_inputs.py",
    "current_model.py": "pose_point_depth_mv/infer_real_proobjaverse_official_ss_slat.py",
    "reconviagen.py": "pose_point_depth_mv/infer_omni_real_reconviagen.py",
    "mesh_coordinates.py": "pose_point_depth_mv/trellis_mesh_coordinate_contract.py",
    "contours.py": "pose_point_depth_mv/render_runtime_o_mesh_camera_contours.py",
    "render_mesh.py": "pose_point_depth_mv/render_ar_object_mesh_previews.py",
    "canonicalization.py": "pose_point_depth_mv/real_object_canonicalization.py",
    "pose_mask.py": "pose_point_depth_mv/pose_mask_object_canonicalization.py",
    "common.py": "pose_point_depth_mv/omni_real_benchmark_common.py",
    "raw_cache.py": "pose_point_depth_mv/dataset_tools/prepare_omni_real_video_cache.py",
    "dino_condition.py": "pose_point_depth_mv/dino_only_condition.py",
}

SHARED_MODEL_BACKENDS = (
    "pose_point_depth_mv/native_ss_genrecon_no_vggt.py",
    "pose_point_depth_mv/native_slat_genrecon_no_vggt.py",
    "pose_point_depth_mv/native_slat_genrecon_v2.py",
    "pose_point_depth_mv/proobjaverse_official_ss.py",
    "pose_point_depth_mv/evaluate_proobjaverse_official_native_ss_stock_slat.py",
    "ReconViaGen/trellis/pipelines/trellis_image_to_3d.py",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def binding(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
    ).strip()


def main() -> None:
    source_files = sorted(
        path
        for path in PACKAGE.rglob("*")
        if path.is_file()
        and "output" not in path.relative_to(PACKAGE).parts
        and "__pycache__" not in path.parts
        and path != MANIFEST
    )
    migration = PACKAGE / "output/已有SS30K_SLat30K结果/MIGRATION_MANIFEST.json"
    output_identity = (
        PACKAGE / "output/已有SS30K_SLat30K结果/MODEL_IDENTITY_AUDIT.json"
    )
    migration_payload = json.loads(migration.read_text(encoding="utf-8"))
    if migration_payload.get("passed") is not True:
        raise RuntimeError("existing-output migration did not pass")
    copied = {}
    for local_name, source_name in COPIED_IMPLEMENTATIONS.items():
        copied[local_name] = {
            "local": binding(PACKAGE / local_name),
            "source_at_extraction": binding(ROOT / source_name),
            "relationship": "copied then imports/entrypoint adapted inside isolated package",
        }
    shared = {name: binding(ROOT / name) for name in SHARED_MODEL_BACKENDS}
    report = {
        "format": "manual_mesh_reconstruction.package_manifest.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "passed": True,
        "package_root": str(PACKAGE),
        "git": {
            "branch": _git("branch", "--show-current"),
            "head": _git("rev-parse", "HEAD"),
            "working_tree_dirty": bool(_git("status", "--short", "--untracked-files=no")),
        },
        "source_regular_file_count": len(source_files),
        "source_total_bytes": sum(path.stat().st_size for path in source_files),
        "source_files": {
            path.relative_to(PACKAGE).as_posix(): binding(path)
            for path in source_files
        },
        "copied_implementation_provenance": copied,
        "shared_validated_model_backends": shared,
        "shared_backend_policy": (
            "manual orchestration/inference entrypoints are isolated here; validated "
            "network definitions and ReconViaGen runtime remain read-only dependencies "
            "to prevent architecture drift"
        ),
        "existing_outputs": {
            "migration_manifest": binding(migration),
            "model_identity_audit": binding(output_identity),
            "copied_directory_count": migration_payload["copied_directory_count"],
            "regular_file_count": migration_payload["regular_file_count"],
            "total_bytes": migration_payload["total_bytes"],
        },
    }
    MANIFEST.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "passed": True,
                "source_regular_file_count": len(source_files),
                "existing_output_count": migration_payload["copied_directory_count"],
                "manifest": str(MANIFEST),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
