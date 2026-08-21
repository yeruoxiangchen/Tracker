#!/usr/bin/env python3
"""Repair erroneous v1 real-input Mesh axes without rerunning models.

The affected v1 exporter called ``to_trimesh(transform_pose=True)`` even
though the decoder-native vertices already shared runtime-O sparse-grid axes.
This one-way, hash-bound migration applies the exact inverse rotation and
writes a new output tree.  It never modifies the v1 directory.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from pose_point_depth_mv.mesh_benchmark_metrics import mesh_structure_metrics
from pose_point_depth_mv.omni_real_benchmark_common import (
    atomic_json,
    load_json,
    sha256_file,
)
from pose_point_depth_mv.trellis_mesh_coordinate_contract import (
    LEGACY_V1_TO_RUNTIME_O,
    mesh_frame_contract_fields,
    reframe_legacy_decoder_mesh,
    validate_erroneous_v1_mesh_frame_contract,
    validate_runtime_o_mesh_frame_contract,
)


CORRECTION_FORMAT = "pose_point_depth_mv.omni_with_vggt_mesh_axis_correction.v2"

# The erroneous exporter was shared by the historical with-VGGT qualitative
# endpoint and the later official no-VGGT SS30K+SLat30K real-input endpoint.
# Keep this repair deliberately narrow: only these exact manifest/result pairs
# are eligible, and the original format identity is preserved in the repaired
# artifact.
SUPPORTED_FORMAT_PAIRS = {
    "pose_point_depth_mv.omni_real_official_with_vggt_inference_manifest.v1":
        "pose_point_depth_mv.omni_real_official_with_vggt_inference.v1",
    "pose_point_depth_mv.real_proobjaverse_official_ss_slat_inference_manifest.v1":
        "pose_point_depth_mv.real_proobjaverse_official_ss_slat_inference.v1",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_mesh(path: Path) -> trimesh.Trimesh:
    value = trimesh.load(path, force="mesh", process=False)
    if not isinstance(value, trimesh.Trimesh):
        raise RuntimeError(f"legacy Mesh is not a single Trimesh: {path}")
    if len(value.vertices) <= 0 or len(value.faces) <= 0:
        raise RuntimeError(f"legacy Mesh is empty: {path}")
    if not bool(np.isfinite(np.asarray(value.vertices)).all()):
        raise RuntimeError(f"legacy Mesh has non-finite vertices: {path}")
    return value


def reframe_manifest(
    *,
    source_manifest_path: Path,
    output_dir: Path,
    expected_source_mesh_sha256: str,
) -> dict[str, Any]:
    source_manifest_path = source_manifest_path.expanduser().resolve(strict=True)
    source = load_json(source_manifest_path)
    result_format = SUPPORTED_FORMAT_PAIRS.get(str(source.get("format")))
    if (
        result_format is None
        or source.get("passed") is not True
        or source.get("output_frame") != "runtime-O"
    ):
        raise RuntimeError("legacy real-input inference manifest identity differs")
    rows = list(source.get("objects", []))
    if len(rows) != 1:
        raise RuntimeError("axis correction requires exactly one object/seed")
    legacy_row = rows[0]
    validate_erroneous_v1_mesh_frame_contract(legacy_row)

    source_mesh = Path(legacy_row["mesh"]).expanduser().resolve(strict=True)
    source_mesh_sha = sha256_file(source_mesh)
    if (
        source_mesh_sha != str(legacy_row.get("mesh_sha256"))
        or source_mesh_sha != str(expected_source_mesh_sha256)
    ):
        raise RuntimeError("legacy source Mesh SHA256 differs")
    source_result = source_mesh.parent / "result.json"
    source_result_payload = load_json(source_result)
    if source_result_payload != legacy_row:
        raise RuntimeError("legacy result.json and manifest row differ")
    if source_result_payload.get("format") != result_format:
        raise RuntimeError("legacy result/manifest format pair differs")

    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    destination_mesh = (
        output_dir
        / "meshes"
        / str(legacy_row["category"])
        / str(legacy_row["object_id"])
        / f"seed_{int(legacy_row['seed'])}"
        / "mesh_o.obj"
    )
    destination_result = destination_mesh.parent / "result.json"
    destination_mesh.parent.mkdir(parents=True, exist_ok=False)

    legacy_mesh = _load_mesh(source_mesh)
    corrected_mesh = reframe_legacy_decoder_mesh(legacy_mesh)
    if len(corrected_mesh.vertices) != len(legacy_mesh.vertices) or len(
        corrected_mesh.faces
    ) != len(legacy_mesh.faces):
        raise RuntimeError("axis correction changed Mesh topology")
    expected_vertices = trimesh.transform_points(
        np.asarray(legacy_mesh.vertices, dtype=np.float64), LEGACY_V1_TO_RUNTIME_O
    )
    if not np.array_equal(
        np.asarray(corrected_mesh.vertices, dtype=np.float64), expected_vertices
    ):
        raise RuntimeError("axis correction numerical result differs")

    temporary = destination_mesh.with_name(
        f".{destination_mesh.name}.tmp-{os.getpid()}"
    )
    corrected_mesh.export(temporary, file_type="obj")
    os.replace(temporary, destination_mesh)
    corrected_structure = mesh_structure_metrics(corrected_mesh)
    if corrected_structure.get("mesh_success") is not True:
        raise RuntimeError("corrected runtime-O Mesh structure failed")

    frame_fields = mesh_frame_contract_fields(
        export_policy=(
            "hash-bound inverse of erroneous v1 transform_pose=True export; "
            "numerically equivalent to decoder.to_trimesh(transform_pose=False)"
        )
    )
    result = dict(legacy_row)
    result.update(
        {
            "format": result_format,
            "created_at_utc": utc_now(),
            "mesh": str(destination_mesh),
            "mesh_sha256": sha256_file(destination_mesh),
            "result": str(destination_result),
            "structure": corrected_structure,
            "output_frame": "runtime-O",
            "legacy_posthoc_reframe": True,
            "legacy_v1_erroneous_axis_transform_repaired": True,
            "legacy_source_result": str(source_result),
            "legacy_source_result_sha256": sha256_file(source_result),
            "legacy_source_mesh": str(source_mesh),
            "legacy_source_mesh_sha256": source_mesh_sha,
            **frame_fields,
        }
    )
    validate_runtime_o_mesh_frame_contract(result)
    atomic_json(destination_result, result)

    manifest = dict(source)
    manifest.update(
        {
            "created_at_utc": utc_now(),
            "objects": [result],
            "output_frame": "runtime-O",
            "legacy_posthoc_reframe": True,
            "legacy_source_manifest": str(source_manifest_path),
            "legacy_source_manifest_sha256": sha256_file(source_manifest_path),
            **frame_fields,
        }
    )
    validate_runtime_o_mesh_frame_contract(manifest)
    manifest_path = output_dir / "inference_manifest.json"
    atomic_json(manifest_path, manifest)
    correction = {
        "format": CORRECTION_FORMAT,
        "created_at_utc": utc_now(),
        "passed": True,
        "source_manifest": str(source_manifest_path),
        "source_manifest_sha256": sha256_file(source_manifest_path),
        "source_mesh": str(source_mesh),
        "source_mesh_sha256": source_mesh_sha,
        "corrected_manifest": str(manifest_path),
        "corrected_manifest_sha256": sha256_file(manifest_path),
        "corrected_mesh": str(destination_mesh),
        "corrected_mesh_sha256": sha256_file(destination_mesh),
        "vertex_count": int(len(corrected_mesh.vertices)),
        "face_count": int(len(corrected_mesh.faces)),
        "source_bbox_extent": np.asarray(legacy_mesh.extents, dtype=float).tolist(),
        "corrected_bbox_extent": np.asarray(
            corrected_mesh.extents, dtype=float
        ).tolist(),
        **frame_fields,
    }
    validate_runtime_o_mesh_frame_contract(
        {"output_frame": "runtime-O", **correction}
    )
    atomic_json(output_dir / "coordinate_correction_report.json", correction)
    return correction


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--expected_source_mesh_sha256", required=True)
    return parser


def main() -> None:
    args = make_parser().parse_args()
    result = reframe_manifest(
        source_manifest_path=Path(args.source_manifest),
        output_dir=Path(args.output_dir),
        expected_source_mesh_sha256=str(args.expected_source_mesh_sha256),
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
