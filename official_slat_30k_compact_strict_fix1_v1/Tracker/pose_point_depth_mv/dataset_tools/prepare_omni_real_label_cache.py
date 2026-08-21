#!/usr/bin/env python3
"""Bind GT Scan meshes to input-derived runtime-O for training/evaluation labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
from typing import Any

import numpy as np

from pose_point_depth_mv.dataset_tools.align_omni_real_mesh_to_colmap import (
    MANIFEST_FORMAT as ALIGNMENT_MANIFEST_FORMAT,
    transform_obj_geometry,
)
from pose_point_depth_mv.dataset_tools.adjudicate_omni_real_mesh_alignment import (
    MANIFEST_FORMAT as ADJUDICATED_ALIGNMENT_MANIFEST_FORMAT,
)
from pose_point_depth_mv.dataset_tools.prepare_omni_real_runtime_inputs import (
    MANIFEST_FORMAT as RUNTIME_MANIFEST_FORMAT,
)
from pose_point_depth_mv.dataset_tools.prepare_omni_real_video_cache import (
    sha256_file,
    utc_now,
    write_json,
    write_npz,
)
from pose_point_depth_mv.real_object_canonicalization import canonical_json_sha256
from pose_point_depth_mv.real_object_label_binding import (
    REAL_OBJECT_LABEL_BINDING_VERSION,
    bind_scan_to_runtime_object,
)


OBJECT_FORMAT = "pose_point_depth_mv.omni_real_runtime_o_label_object.v1"
MANIFEST_FORMAT = "pose_point_depth_mv.omni_real_runtime_o_label_manifest.v1"
MARKER_FORMAT = "pose_point_depth_mv.omni_real_runtime_o_label_marker.v1"


def _object_key(row: dict[str, Any]) -> str:
    return f"{row['category']}:{row['object_id']}"


def _load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _verify_condition_identity(runtime: dict[str, Any]) -> str:
    condition_path = Path(runtime["condition_record"]).resolve()
    condition = _load_json(condition_path)
    recorded = str(condition.pop("condition_sha256", ""))
    expected = str(runtime["condition_sha256"])
    if not recorded or recorded != expected:
        raise RuntimeError(f"runtime condition identity mismatch: {condition_path}")
    if canonical_json_sha256(condition) != expected:
        raise RuntimeError(f"runtime condition record changed: {condition_path}")
    return expected


def _load_reusable(
    destination: Path,
    *,
    runtime_sha256: str,
    alignment_sha256: str,
    alignment_quality_passed: bool,
    alignment_quality_warning_included: bool,
) -> dict[str, Any] | None:
    marker_path = destination / "_LABEL_COMPLETE.json"
    report_path = destination / "report.json"
    if not marker_path.is_file() or not report_path.is_file():
        return None
    marker = _load_json(marker_path)
    if (
        marker.get("format") != MARKER_FORMAT
        or marker.get("runtime_cache_sha256") != runtime_sha256
        or marker.get("alignment_cache_sha256") != alignment_sha256
    ):
        raise RuntimeError(f"stale runtime-O label output: {destination}")
    report = _load_json(report_path)
    if report.get("format") != OBJECT_FORMAT or report.get("passed") is not True:
        raise RuntimeError(f"invalid reusable runtime-O label: {report_path}")
    if (
        report.get("alignment_quality_passed") is not alignment_quality_passed
        or report.get("alignment_quality_warning_included")
        is not alignment_quality_warning_included
    ):
        raise RuntimeError(f"reused label quality contract changed: {report_path}")
    for key in ("mesh_o", "label_cache"):
        path = Path(str(report.get(key, "")))
        if not path.is_file():
            raise RuntimeError(f"runtime-O label artifact is missing: {path}")
        digest_key = f"{key}_sha256"
        if report.get(digest_key) != sha256_file(path):
            raise RuntimeError(f"runtime-O label artifact binding changed: {path}")
    return report


def build_object_label(
    runtime: dict[str, Any],
    alignment: dict[str, Any],
    *,
    output_dir: Path,
    resume_partial: bool = False,
    include_alignment_quality_warning: bool = False,
) -> tuple[dict[str, Any], bool]:
    key = _object_key(runtime)
    if key != _object_key(alignment):
        raise RuntimeError(f"runtime/alignment object mismatch: {key}")
    if runtime.get("passed") is not True:
        raise RuntimeError(f"runtime input did not pass: {key}")
    alignment_quality_passed = alignment.get("automatic_passed") is True
    alignment_quality_warning_included = (
        not alignment_quality_passed and include_alignment_quality_warning
    )
    if not alignment_quality_passed and not alignment_quality_warning_included:
        raise RuntimeError(f"GT alignment did not pass: {key}")
    if alignment_quality_warning_included and (
        not isinstance(alignment.get("alignment_quality_policy"), dict)
        or not isinstance(alignment.get("alignment_quality_checks"), dict)
    ):
        raise RuntimeError(f"GT alignment warning lacks adjudication evidence: {key}")
    runtime_cache = Path(runtime["cache_npz"]).resolve()
    alignment_cache = Path(alignment["cache_npz"]).resolve()
    runtime_hash = sha256_file(runtime_cache)
    alignment_hash = sha256_file(alignment_cache)
    if runtime.get("source_raw_cache_sha256") != alignment.get("raw_cache_sha256"):
        raise RuntimeError(f"runtime/alignment raw-cache binding differs: {key}")
    condition_hash = _verify_condition_identity(runtime)

    destination = output_dir / "objects" / runtime["category"] / runtime["object_id"]
    reusable = _load_reusable(
        destination,
        runtime_sha256=runtime_hash,
        alignment_sha256=alignment_hash,
        alignment_quality_passed=alignment_quality_passed,
        alignment_quality_warning_included=alignment_quality_warning_included,
    )
    if reusable is not None:
        if reusable.get("condition_sha256") != condition_hash:
            raise RuntimeError(f"reused label condition identity changed: {key}")
        return reusable, True
    if destination.exists():
        raise RuntimeError(f"partial runtime-O label output exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / f".{runtime['object_id']}.label-building"
    if staging.exists():
        if not resume_partial:
            raise RuntimeError(f"partial runtime-O label staging exists: {staging}")
        shutil.rmtree(staging)
    staging.mkdir()

    with np.load(runtime_cache, allow_pickle=False) as payload:
        T_O2W = np.asarray(payload["T_O2W"], dtype=np.float64)
        T_W2O = np.asarray(payload["T_W2O"], dtype=np.float64)
    with np.load(alignment_cache, allow_pickle=False) as payload:
        T_Scan2W = np.asarray(payload["T_Scan_to_COLMAP_W"], dtype=np.float64)
    T_Scan2O = bind_scan_to_runtime_object(T_Scan2W, T_O2W=T_O2W)
    expected = T_W2O @ T_Scan2W
    if not np.allclose(T_Scan2O, expected, rtol=1.0e-10, atol=1.0e-10):
        raise RuntimeError(f"Scan-to-O composition mismatch: {key}")
    roundtrip_error = float(np.max(np.abs(T_O2W @ T_W2O - np.eye(4))))
    if roundtrip_error > 1.0e-8:
        raise RuntimeError(f"runtime-O inverse roundtrip failed: {key}")

    mesh_name = "Scan_in_runtime_O.obj"
    cache_name = "label_binding.npz"
    mesh_stats = transform_obj_geometry(
        Path(alignment["scan_obj"]),
        staging / mesh_name,
        T_Scan2O,
        strip_materials=True,
    )
    write_npz(
        staging / cache_name,
        T_Scan2W=T_Scan2W,
        T_O2W=T_O2W,
        T_W2O=T_W2O,
        T_Scan2O=T_Scan2O,
    )
    report = {
        "format": OBJECT_FORMAT,
        "created_at_utc": utc_now(),
        "category": str(runtime["category"]),
        "object_id": str(runtime["object_id"]),
        "object_key": key,
        "label_binding_version": REAL_OBJECT_LABEL_BINDING_VERSION,
        "runtime_input_report": str(Path(runtime["cache_npz"]).parent / "report.json"),
        "runtime_cache": str(runtime_cache),
        "runtime_cache_sha256": runtime_hash,
        "alignment_report": str(Path(alignment["cache_npz"]).parent / "report.json"),
        "alignment_cache": str(alignment_cache),
        "alignment_cache_sha256": alignment_hash,
        "alignment_quality_passed": alignment_quality_passed,
        "alignment_quality_warning_included": alignment_quality_warning_included,
        "alignment_quality_policy": alignment.get("alignment_quality_policy"),
        "alignment_quality_checks": alignment.get("alignment_quality_checks"),
        "alignment_quality_diagnostics": {
            "median_normalized": alignment.get("median_normalized"),
            "inlier_rate_3pct": alignment.get("inlier_rate_3pct"),
            "p90_normalized": alignment.get(
                "p90_normalized_diagnostic", alignment.get("p90_normalized")
            ),
        },
        "scan_obj": str(Path(alignment["scan_obj"]).resolve()),
        "scan_obj_sha256": sha256_file(Path(alignment["scan_obj"])),
        "mesh_o": str((destination / mesh_name).resolve()),
        "mesh_o_sha256": sha256_file(staging / mesh_name),
        "label_cache": str((destination / cache_name).resolve()),
        "label_cache_sha256": sha256_file(staging / cache_name),
        "mesh_stats": mesh_stats,
        "runtime_inverse_roundtrip_max_abs": roundtrip_error,
        "condition_sha256": condition_hash,
        "condition_identity_unchanged_by_gt_binding": True,
        "gt_fields_exported_to_model_condition": False,
        "training_ready": False,
        "scope_guard": (
            "Label-only Scan-to-runtime-O binding. Scan/T_Scan2W/T_Scan2O and "
            "Mesh_O are forbidden from inference conditions. Target latent "
            "encoding and model compatibility gates remain separate. A false "
            "alignment_quality_passed value is retained as an explicit evaluation "
            "warning and never changes the fitted transform."
        ),
        "passed": True,
    }
    write_json(staging / "report.json", report)
    write_json(
        staging / "_LABEL_COMPLETE.json",
        {
            "format": MARKER_FORMAT,
            "completed_at_utc": utc_now(),
            "object_key": key,
            "runtime_cache_sha256": runtime_hash,
            "alignment_cache_sha256": alignment_hash,
            "alignment_quality_passed": alignment_quality_passed,
            "alignment_quality_warning_included": alignment_quality_warning_included,
            "condition_sha256": condition_hash,
            "passed": True,
        },
    )
    staging.replace(destination)
    return report, False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--runtime_input_manifest", required=True)
    parser.add_argument("--alignment_manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--allow_failures", action="store_true")
    parser.add_argument(
        "--include_alignment_quality_warnings",
        action="store_true",
        help=(
            "Keep completed adjudicated alignments that miss the frozen quality "
            "threshold. Their transforms remain unchanged and warning metadata is "
            "propagated into the label/evaluation manifests."
        ),
    )
    parser.add_argument("--max_alignment_quality_warnings", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    runtime_path = Path(args.runtime_input_manifest).expanduser().resolve()
    alignment_path = Path(args.alignment_manifest).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    runtime = _load_json(runtime_path)
    alignment = _load_json(alignment_path)
    if runtime.get("format") != RUNTIME_MANIFEST_FORMAT or runtime.get("passed") is not True:
        raise RuntimeError(f"runtime input manifest did not pass: {runtime_path}")
    alignment_format = alignment.get("format")
    if alignment_format not in {
        ALIGNMENT_MANIFEST_FORMAT,
        ADJUDICATED_ALIGNMENT_MANIFEST_FORMAT,
    }:
        raise RuntimeError(f"alignment manifest did not pass: {alignment_path}")
    alignment_rows = list(alignment.get("objects", []))
    quality_warning_rows = [
        row for row in alignment_rows if row.get("automatic_passed") is not True
    ]
    include_quality_warnings = bool(args.include_alignment_quality_warnings)
    if alignment.get("passed") is not True:
        complete_adjudicated = (
            alignment_format == ADJUDICATED_ALIGNMENT_MANIFEST_FORMAT
            and int(alignment.get("selected_object_count", -1)) == len(alignment_rows)
            and int(alignment.get("completed_object_count", -1)) == len(alignment_rows)
            and not alignment.get("failures")
            and int(alignment.get("automatic_pass_count", -1))
            == len(alignment_rows) - len(quality_warning_rows)
        )
        if not include_quality_warnings or not complete_adjudicated:
            raise RuntimeError(f"alignment manifest did not pass: {alignment_path}")
    if include_quality_warnings:
        if int(args.max_alignment_quality_warnings) < 0:
            raise ValueError("max_alignment_quality_warnings must be non-negative")
        if len(quality_warning_rows) > int(args.max_alignment_quality_warnings):
            raise RuntimeError(
                "alignment quality warning count exceeds explicit bound: "
                f"{len(quality_warning_rows)} > {args.max_alignment_quality_warnings}"
            )
    runtime_by_key = {_object_key(row): row for row in runtime.get("objects", [])}
    alignment_by_key = {_object_key(row): row for row in alignment.get("objects", [])}
    if set(runtime_by_key) != set(alignment_by_key):
        raise RuntimeError("runtime/alignment object sets differ")
    output_dir.mkdir(parents=True, exist_ok=True)
    reports: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    reused: list[str] = []
    for index, key in enumerate(sorted(runtime_by_key), start=1):
        print(f"[runtime_o_label] {index}/{len(runtime_by_key)} object={key}", flush=True)
        try:
            report, was_reused = build_object_label(
                runtime_by_key[key],
                alignment_by_key[key],
                output_dir=output_dir,
                resume_partial=bool(args.resume),
                include_alignment_quality_warning=include_quality_warnings,
            )
            reports.append(report)
            if was_reused:
                reused.append(key)
        except Exception as error:
            failures.append({"object_key": key, "error": repr(error)})
            print(f"[runtime_o_label] FAILED object={key}: {error!r}", flush=True)
            if not args.allow_failures:
                raise
    manifest = {
        "format": MANIFEST_FORMAT,
        "created_at_utc": utc_now(),
        "runtime_input_manifest": str(runtime_path),
        "runtime_input_manifest_sha256": sha256_file(runtime_path),
        "alignment_manifest": str(alignment_path),
        "alignment_manifest_sha256": sha256_file(alignment_path),
        "selected_object_count": len(runtime_by_key),
        "completed_object_count": len(reports),
        "reused_objects": reused,
        "objects": reports,
        "failures": failures,
        "alignment_quality_all_passed": not quality_warning_rows,
        "alignment_quality_pass_count": len(runtime_by_key) - len(quality_warning_rows),
        "alignment_quality_warning_count": len(quality_warning_rows),
        "alignment_quality_warning_object_keys": sorted(
            _object_key(row) for row in quality_warning_rows
        ),
        "alignment_quality_warnings_included": bool(
            include_quality_warnings and quality_warning_rows
        ),
        "alignment_quality_warning_policy": (
            "complete adjudicated object coverage is retained without transform "
            "refit, object deletion, replacement, or inference-condition changes; "
            "the primary report uses all objects and reports a reliable-only "
            "sensitivity subgroup"
            if include_quality_warnings and quality_warning_rows
            else None
        ),
        "condition_identity_unchanged_by_gt_binding": all(
            row["condition_identity_unchanged_by_gt_binding"] for row in reports
        ),
        "training_ready": False,
        "scope_guard": (
            "Batch label-only Mesh_O cache. No target latent or model input is "
            "created here; inference remains independent of every GT field."
        ),
    }
    manifest["passed"] = (
        bool(reports)
        and not failures
        and len(reports) == len(runtime_by_key)
        and manifest["condition_identity_unchanged_by_gt_binding"]
    )
    manifest_path = output_dir / "runtime_o_label_manifest.json"
    write_json(manifest_path, manifest)
    print(
        json.dumps(
            {
                "passed": manifest["passed"],
                "completed_object_count": len(reports),
                "failure_count": len(failures),
                "alignment_quality_warning_count": len(quality_warning_rows),
                "training_ready": False,
                "manifest": str(manifest_path),
            },
            indent=2,
        )
    )
    if not manifest["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
