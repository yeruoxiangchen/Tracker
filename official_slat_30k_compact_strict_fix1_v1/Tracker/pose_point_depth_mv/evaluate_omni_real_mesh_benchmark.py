#!/usr/bin/env python3
"""Direct runtime-O Mesh metrics for Native v2 Full and frozen external bases."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from pose_point_depth_mv.dataset_tools.prepare_omni_real_label_cache import (
    MANIFEST_FORMAT as LABEL_MANIFEST_FORMAT,
)
from pose_point_depth_mv.dataset_tools.prepare_omni_real_video_cache import utc_now
from pose_point_depth_mv.mesh_benchmark_metrics import (
    mesh_structure_metrics,
    surface_metrics,
)
from pose_point_depth_mv.omni_real_benchmark_common import (
    atomic_json,
    canonical_sha256,
    index_objects,
    load_json,
    object_key,
    sha256_file,
    validate_bound_file,
)


REPORT_FORMAT = "pose_point_depth_mv.omni_real_mesh_benchmark.v1"
RUNTIME_MANIFEST_FORMAT = "pose_point_depth_mv.omni_real_runtime_input_manifest.v2"
LEGACY_IMAGE_ONLY_RUNTIME_MANIFEST_FORMAT = (
    "pose_point_depth_mv.omni_real_runtime_input_manifest.v1"
)
METHOD_FORMATS = {
    "native_v2_full": "pose_point_depth_mv.omni_real_native_v2_inference_manifest.v1",
    "reconviagen_original": (
        "pose_point_depth_mv.omni_real_reconviagen_inference_manifest.v1"
    ),
    "pixal3d_official": (
        "pose_point_depth_mv.omni_real_pixal3d_inference_manifest.v1"
    ),
}
RECORD_METHODS = {
    "native_v2_full": "native_v2_full",
    "reconviagen_original": "reconviagen_original",
    "pixal3d_official": "pixal3d_official_single_reference_view",
}
SURFACE_FIELDS = (
    "chamfer_l1",
    "chamfer_l2",
    "fscore_0p01",
    "fscore_0p02",
    "normal_consistency",
)
STRUCTURE_FIELDS = (
    "largest_component_ratio",
    "component_count",
)
EXTERNAL_VISIBLE_BUILD_FIELDS = (
    "selected_view_count",
    "view_selection",
    "reference_view",
    "feature_resolution",
    "foreground_margin",
    "alpha_threshold",
)


def load_mesh(path: str | Path) -> trimesh.Trimesh:
    loaded = trimesh.load(Path(path), force="scene", process=False)
    if isinstance(loaded, trimesh.Scene):
        pieces = [
            value
            for value in loaded.dump(concatenate=False)
            if isinstance(value, trimesh.Trimesh)
            and len(value.vertices)
            and len(value.faces)
        ]
        if not pieces:
            raise RuntimeError(f"mesh scene contains no triangles: {path}")
        return trimesh.util.concatenate(pieces)
    if not isinstance(loaded, trimesh.Trimesh):
        raise TypeError(f"unsupported mesh type={type(loaded)}: {path}")
    if not len(loaded.vertices) or not len(loaded.faces):
        raise RuntimeError(f"mesh is empty: {path}")
    return loaded


def summarize(values: list[float]) -> dict[str, Any]:
    finite = np.asarray(
        [float(value) for value in values if np.isfinite(float(value))],
        dtype=np.float64,
    )
    if not len(finite):
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "count": int(len(finite)),
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
    }


def _records_by_pair(manifest: dict[str, Any], *, method: str) -> dict[tuple[str, int], dict[str, Any]]:
    output: dict[tuple[str, int], dict[str, Any]] = {}
    for row in manifest.get("objects", []):
        if row.get("passed") is not True or row.get("method") != RECORD_METHODS[method]:
            raise RuntimeError(f"invalid {method} inference record")
        pair = (str(row["object_key"]), int(row["seed"]))
        if pair in output:
            raise RuntimeError(f"duplicate {method} pair={pair}")
        output[pair] = row
    if not output:
        raise RuntimeError(f"{method} inference manifest contains no records")
    object_keys = {key for key, _ in output}
    declared_seeds = [int(value) for value in manifest.get("seeds", [])]
    if not declared_seeds or len(declared_seeds) != len(set(declared_seeds)):
        raise RuntimeError(f"{method} declares invalid seeds")
    expected = {
        (key, seed) for key in object_keys for seed in declared_seeds
    }
    if set(output) != expected:
        raise RuntimeError(f"{method} does not contain a complete object/seed product")
    if (
        int(manifest.get("object_count", -1)) != len(object_keys)
        or int(manifest.get("record_count", -1)) != len(output)
    ):
        raise RuntimeError(f"{method} manifest counts differ from its records")
    return output


def require_identical_pair_coverage(
    pair_rows: dict[str, dict[tuple[str, int], dict[str, Any]]],
    *,
    label_keys: set[str],
) -> set[tuple[str, int]]:
    if set(pair_rows) != set(METHOD_FORMATS):
        raise RuntimeError("the benchmark method set differs from the frozen protocol")
    expected = set(pair_rows["native_v2_full"])
    if any(set(rows) != expected for rows in pair_rows.values()):
        raise RuntimeError("the three methods do not cover identical object/seed pairs")
    if {key for key, _ in expected} != set(label_keys):
        raise RuntimeError("inference and runtime-O GT object sets differ")
    return expected


def _existing_file_sha256(path: str | Path, *, label: str) -> str:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise RuntimeError(f"{label} is missing: {resolved}")
    return sha256_file(resolved)


def _external_visible_runtime_signature(runtime: dict[str, Any]) -> dict[str, Any]:
    """Bind the runtime fields and files visible to frozen image baselines."""

    if (
        runtime.get("format")
        not in {RUNTIME_MANIFEST_FORMAT, LEGACY_IMAGE_ONLY_RUNTIME_MANIFEST_FORMAT}
        or runtime.get("passed") is not True
    ):
        raise RuntimeError("external runtime input manifest did not pass")
    raw_report = validate_bound_file(
        runtime["raw_cache_report"],
        runtime["raw_cache_report_sha256"],
        label="external runtime raw-cache report",
    )
    build_config = runtime.get("build_config")
    if not isinstance(build_config, dict):
        raise RuntimeError("external runtime lacks build_config")
    missing_build = [
        field for field in EXTERNAL_VISIBLE_BUILD_FIELDS if field not in build_config
    ]
    if missing_build:
        raise RuntimeError(
            f"external runtime lacks visible build fields={missing_build}"
        )
    rows = index_objects(runtime.get("objects", []), label="external runtime")
    if (
        int(runtime.get("selected_object_count", -1)) != len(rows)
        or int(runtime.get("completed_object_count", -1)) != len(rows)
        or runtime.get("failures")
    ):
        raise RuntimeError("external runtime object counts/failures are invalid")

    object_signatures = []
    rgb_file_count = 0
    mask_file_count = 0
    for key in sorted(rows):
        row = rows[key]
        if (
            row.get("passed") is not True
            or row.get("forbidden_gt_fields_absent") is not True
            or row.get("training_ready") is not False
        ):
            raise RuntimeError(f"external runtime object contract failed: {key}")
        view_count = int(row["selected_view_count"])
        selected_indices = [int(value) for value in row["selected_source_view_indices"]]
        frame_names = [str(value) for value in row["selected_frame_names"]]
        rgb_paths = [str(value) for value in row["prepared_rgb_paths"]]
        mask_paths = [str(value) for value in row["prepared_mask_paths"]]
        if any(
            len(values) != view_count
            for values in (selected_indices, frame_names, rgb_paths, mask_paths)
        ):
            raise RuntimeError(f"external runtime view counts differ: {key}")
        reference_index = int(row["reference_view_index"])
        if reference_index < 0 or reference_index >= view_count:
            raise RuntimeError(f"external runtime reference view is invalid: {key}")
        raw_cache = validate_bound_file(
            row["source_raw_cache"],
            row["source_raw_cache_sha256"],
            label=f"external runtime source cache {key}",
        )
        rgb_hashes = [
            _existing_file_sha256(path, label=f"RGB {key}")
            for path in rgb_paths
        ]
        mask_hashes = [
            _existing_file_sha256(path, label=f"mask {key}")
            for path in mask_paths
        ]
        rgb_file_count += len(rgb_hashes)
        mask_file_count += len(mask_hashes)
        object_signatures.append(
            {
                "object_key": key,
                "source_raw_cache": str(raw_cache),
                "source_raw_cache_sha256": str(row["source_raw_cache_sha256"]),
                "selected_view_count": view_count,
                "selected_source_view_indices": selected_indices,
                "selected_frame_names": frame_names,
                "reference_view_index": reference_index,
                "prepared_rgb_sha256": rgb_hashes,
                "prepared_mask_sha256": mask_hashes,
            }
        )
    return {
        "contract": "external_model_visible_runtime_inputs.v1",
        "raw_cache_report": str(raw_report),
        "raw_cache_report_sha256": str(runtime["raw_cache_report_sha256"]),
        "build_config": {
            field: build_config[field] for field in EXTERNAL_VISIBLE_BUILD_FIELDS
        },
        "object_count": len(rows),
        "prepared_rgb_file_count": rgb_file_count,
        "prepared_mask_file_count": mask_file_count,
        "objects": object_signatures,
    }


def validate_method_runtime_binding(
    method: str,
    manifest: dict[str, Any],
    *,
    reference_runtime_path: Path,
    reference_runtime: dict[str, Any],
    reference_runtime_sha256: str,
    signature_cache: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Require exact Native binding or audited image-only v1/v2 equivalence."""

    declared_hash = str(manifest.get("runtime_input_manifest_sha256", ""))
    bound_path = validate_bound_file(
        manifest.get("runtime_input_manifest", ""),
        declared_hash,
        label=f"{method} runtime input manifest",
    )
    base = {
        "method_runtime_manifest": str(bound_path),
        "method_runtime_manifest_sha256": declared_hash,
        "reference_runtime_manifest": str(reference_runtime_path),
        "reference_runtime_manifest_sha256": str(reference_runtime_sha256),
    }
    if declared_hash == str(reference_runtime_sha256):
        return {**base, "binding_mode": "exact_manifest_sha256", "passed": True}
    if method == "native_v2_full":
        raise RuntimeError(
            "native_v2_full must bind the exact runtime-v2 manifest used by labels"
        )
    if method not in {"reconviagen_original", "pixal3d_official"}:
        raise RuntimeError(f"unsupported runtime compatibility method={method}")
    bound_runtime = load_json(bound_path)
    if bound_runtime.get("format") != LEGACY_IMAGE_ONLY_RUNTIME_MANIFEST_FORMAT:
        raise RuntimeError(
            f"{method} non-exact runtime must be the frozen image-only v1 format"
        )
    if reference_runtime.get("format") != RUNTIME_MANIFEST_FORMAT:
        raise RuntimeError("benchmark labels must bind the runtime-v2 manifest")

    cache = signature_cache if signature_cache is not None else {}
    reference_key = str(reference_runtime_sha256)
    bound_key = declared_hash
    if reference_key not in cache:
        cache[reference_key] = _external_visible_runtime_signature(reference_runtime)
    if bound_key not in cache:
        cache[bound_key] = _external_visible_runtime_signature(bound_runtime)
    reference_signature = cache[reference_key]
    bound_signature = cache[bound_key]
    reference_signature_hash = canonical_sha256(reference_signature)
    bound_signature_hash = canonical_sha256(bound_signature)
    if bound_signature != reference_signature:
        raise RuntimeError(
            f"{method} runtime-v1 is not external-input-equivalent to runtime-v2"
        )
    return {
        **base,
        "binding_mode": "audited_external_visible_v1_v2_equivalence",
        "legacy_runtime_format": bound_runtime["format"],
        "reference_runtime_format": reference_runtime["format"],
        "external_visible_input_signature_sha256": bound_signature_hash,
        "reference_visible_input_signature_sha256": reference_signature_hash,
        "raw_cache_report_sha256": bound_signature["raw_cache_report_sha256"],
        "object_count": bound_signature["object_count"],
        "prepared_rgb_file_count": bound_signature["prepared_rgb_file_count"],
        "prepared_mask_file_count": bound_signature["prepared_mask_file_count"],
        "verified_fields": [
            "raw_cache_report/path/hash",
            "object_set",
            "source_raw_cache/path/hash",
            "external_visible_build_config",
            "selected_view_indices/frame_names",
            "reference_view_index",
            "ordered_prepared_rgb_file_hashes",
            "ordered_prepared_mask_file_hashes",
        ],
        "excluded_lifting_only_fields": [
            "condition_sha256",
            "condition_record",
            "runtime_input_cache",
            "T_O2C_lifting",
        ],
        "passed": True,
    }


def _method_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    output = {
        field: summarize([row[field] for row in records])
        for field in (*SURFACE_FIELDS, *STRUCTURE_FIELDS)
    }
    output["mesh_success_rate"] = float(
        np.mean([float(row["mesh_success"]) for row in records])
    )
    output["record_count"] = len(records)
    return output


def _paired_delta(
    records: list[dict[str, Any]], *, left: str, right: str
) -> dict[str, Any]:
    left_rows = {
        (row["object_key"], row["seed"]): row
        for row in records
        if row["method"] == left
    }
    right_rows = {
        (row["object_key"], row["seed"]): row
        for row in records
        if row["method"] == right
    }
    if set(left_rows) != set(right_rows):
        raise RuntimeError(f"paired method coverage differs: {left}/{right}")
    fields = {}
    for field in SURFACE_FIELDS:
        # Positive always means the left method is better.
        sign = -1.0 if field.startswith("chamfer") else 1.0
        values = [
            sign * (float(left_rows[pair][field]) - float(right_rows[pair][field]))
            for pair in sorted(left_rows)
        ]
        fields[f"{field}_left_improvement"] = {
            **summarize(values),
            "positive_rate": float(np.mean(np.asarray(values) > 0.0)),
        }
    return {"left": left, "right": right, "metrics": fields}


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label_manifest", required=True)
    parser.add_argument("--native_manifest", required=True)
    parser.add_argument("--reconviagen_manifest", required=True)
    parser.add_argument("--pixal3d_manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--surface_samples", type=int, default=20000)
    return parser


def main() -> None:
    args = make_parser().parse_args()
    if int(args.surface_samples) <= 0:
        raise ValueError("surface_samples must be positive")
    label_path = Path(args.label_manifest).expanduser().resolve()
    labels = load_json(label_path)
    if labels.get("format") != LABEL_MANIFEST_FORMAT or labels.get("passed") is not True:
        raise RuntimeError(f"runtime-O label manifest did not pass: {label_path}")
    label_by_key = index_objects(labels.get("objects", []), label="runtime-O labels")
    runtime_hash = str(labels["runtime_input_manifest_sha256"])
    reference_runtime_path = validate_bound_file(
        labels["runtime_input_manifest"],
        runtime_hash,
        label="runtime-O label input manifest",
    )
    reference_runtime = load_json(reference_runtime_path)
    if (
        reference_runtime.get("format") != RUNTIME_MANIFEST_FORMAT
        or reference_runtime.get("passed") is not True
    ):
        raise RuntimeError(
            f"runtime-O labels do not bind a passed runtime-v2 manifest: "
            f"{reference_runtime_path}"
        )
    method_paths = {
        "native_v2_full": Path(args.native_manifest).expanduser().resolve(),
        "reconviagen_original": Path(args.reconviagen_manifest).expanduser().resolve(),
        "pixal3d_official": Path(args.pixal3d_manifest).expanduser().resolve(),
    }
    pair_rows: dict[str, dict[tuple[str, int], dict[str, Any]]] = {}
    runtime_bindings: dict[str, dict[str, Any]] = {}
    signature_cache: dict[str, dict[str, Any]] = {}
    for method, path in method_paths.items():
        manifest = load_json(path)
        if (
            manifest.get("format") != METHOD_FORMATS[method]
            or manifest.get("passed") is not True
            or manifest.get("target_or_metric_consumed") is not False
        ):
            raise RuntimeError(f"{method} manifest/protocol binding did not pass: {path}")
        runtime_bindings[method] = validate_method_runtime_binding(
            method,
            manifest,
            reference_runtime_path=reference_runtime_path,
            reference_runtime=reference_runtime,
            reference_runtime_sha256=runtime_hash,
            signature_cache=signature_cache,
        )
        pair_rows[method] = _records_by_pair(manifest, method=method)
    expected_pairs = require_identical_pair_coverage(
        pair_rows, label_keys=set(label_by_key)
    )

    target_meshes: dict[str, trimesh.Trimesh] = {}
    target_bindings: dict[str, dict[str, str]] = {}
    for key, row in label_by_key.items():
        path = validate_bound_file(
            row["mesh_o"], row["mesh_o_sha256"], label=f"Mesh_O {key}"
        )
        target_meshes[key] = load_mesh(path)
        target_bindings[key] = {"path": str(path), "sha256": row["mesh_o_sha256"]}

    records: list[dict[str, Any]] = []
    for pair_position, pair in enumerate(sorted(expected_pairs)):
        key, seed = pair
        target = target_meshes[key]
        surface_seed = int(seed) * 1009 + int(pair_position) * 9173
        for method in METHOD_FORMATS:
            source = pair_rows[method][pair]
            mesh_path = validate_bound_file(
                source["mesh"], source["mesh_sha256"], label=f"{method} {pair}"
            )
            mesh = load_mesh(mesh_path)
            structure = mesh_structure_metrics(mesh)
            surface = surface_metrics(
                mesh,
                target,
                count=int(args.surface_samples),
                seed=surface_seed,
                thresholds=(0.01, 0.02),
            )
            records.append(
                {
                    "method": method,
                    "object_key": key,
                    "seed": int(seed),
                    "mesh": str(mesh_path),
                    "mesh_sha256": source["mesh_sha256"],
                    "target_mesh": target_bindings[key],
                    "surface_seed": surface_seed,
                    "mesh_success": bool(structure["mesh_success"]),
                    **{field: float(surface[field]) for field in SURFACE_FIELDS},
                    **{field: float(structure[field]) for field in STRUCTURE_FIELDS},
                }
            )
        print(
            f"[real_mesh_benchmark] {pair_position + 1}/{len(expected_pairs)} "
            f"object={key} seed={seed}",
            flush=True,
        )

    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        by_method[row["method"]].append(row)
    summary = {method: _method_summary(by_method[method]) for method in METHOD_FORMATS}
    comparisons = [
        _paired_delta(records, left="native_v2_full", right="reconviagen_original"),
        _paired_delta(records, left="native_v2_full", right="pixal3d_official"),
    ]
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "format": REPORT_FORMAT,
        "created_at_utc": utc_now(),
        "formal": False,
        "protocol_scope": "reusable benchmark32 development set; holdout64 untouched",
        "coordinate_policy": (
            "direct runtime-O/reference-view coordinates; no per-method GT ICP, "
            "scale fit, translation fit, or reflection"
        ),
        "label_manifest": str(label_path),
        "label_manifest_sha256": sha256_file(label_path),
        "runtime_input_manifest_sha256": runtime_hash,
        "runtime_protocol_bindings": runtime_bindings,
        "method_manifests": {
            method: {"path": str(path), "sha256": sha256_file(path)}
            for method, path in method_paths.items()
        },
        "object_count": len(label_by_key),
        "seed_count": len({seed for _, seed in expected_pairs}),
        "pair_count": len(expected_pairs),
        "surface_samples": int(args.surface_samples),
        "surface_thresholds": [0.01, 0.02],
        "summary": summary,
        "paired_comparisons": comparisons,
        "records": records,
        "holdout64_consumed": False,
        "passed": all(
            values["mesh_success_rate"] == 1.0
            and values["record_count"] == len(expected_pairs)
            for values in summary.values()
        ),
    }
    report_path = output_dir / "report.json"
    atomic_json(report_path, report)
    lines = [
        "Omni real benchmark32 direct runtime-O Mesh evaluation",
        "=" * 59,
        f"objects: {report['object_count']} pairs: {report['pair_count']}",
        "No GT ICP/scale/translation/reflection is applied to any method.",
        "",
    ]
    for method, values in summary.items():
        lines.extend(
            [
                method,
                f"  Chamfer-L1 mean/median: {values['chamfer_l1']['mean']:.8f} / {values['chamfer_l1']['median']:.8f}",
                f"  F-score@0.01 mean: {values['fscore_0p01']['mean']:.8f}",
                f"  F-score@0.02 mean: {values['fscore_0p02']['mean']:.8f}",
                f"  normal consistency mean: {values['normal_consistency']['mean']:.8f}",
                f"  mesh success: {values['mesh_success_rate']:.6f}",
                f"  largest component ratio mean: {values['largest_component_ratio']['mean']:.8f}",
                "",
            ]
        )
    (output_dir / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(json.dumps({"passed": report["passed"], "report": str(report_path)}, indent=2))
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
