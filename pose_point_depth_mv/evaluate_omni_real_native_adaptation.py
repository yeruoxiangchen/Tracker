#!/usr/bin/env python3
"""Evaluate adapted Native v2 Full against its parent and frozen image bases."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Any

import numpy as np

from pose_point_depth_mv.dataset_tools.freeze_omni_real_raw_split import (
    SOURCE_INVENTORY_FORMAT,
    SPLIT_ROWS_FORMAT,
)
from pose_point_depth_mv.dataset_tools.prepare_omni_real_label_cache import (
    MANIFEST_FORMAT as LABEL_MANIFEST_FORMAT,
)
from pose_point_depth_mv.dataset_tools.prepare_omni_real_runtime_inputs import (
    MANIFEST_FORMAT as RUNTIME_MANIFEST_FORMAT,
)
from pose_point_depth_mv.dataset_tools.prepare_omni_real_video_cache import (
    RAW_CACHE_FORMAT,
    utc_now,
)
from pose_point_depth_mv.evaluate_omni_real_mesh_benchmark import (
    METHOD_FORMATS,
    STRUCTURE_FIELDS,
    SURFACE_FIELDS,
    load_mesh,
    summarize,
    validate_method_runtime_binding,
)
from pose_point_depth_mv.mesh_benchmark_metrics import (
    mesh_structure_metrics,
    surface_metrics,
)
from pose_point_depth_mv.omni_real_benchmark_common import (
    atomic_json,
    index_objects,
    load_json,
    object_key,
    sha256_file,
    validate_bound_file,
)


REPORT_FORMAT = "pose_point_depth_mv.omni_real_native_adaptation_benchmark.v1"
METHOD_SPECS = {
    "adapted_native_v2_full": ("native_v2_full", "native_v2_full"),
    "parent_native_v2_full": ("native_v2_full", "native_v2_full"),
    "reconviagen_original": ("reconviagen_original", "reconviagen_original"),
    "pixal3d_official": (
        "pixal3d_official",
        "pixal3d_official_single_reference_view",
    ),
}
PROTOCOL_SCOPES = ("development_benchmark32", "formal_holdout64")


def _records_by_pair(
    manifest: dict[str, Any], *, expected_method: str, label: str
) -> dict[tuple[str, int], dict[str, Any]]:
    rows: dict[tuple[str, int], dict[str, Any]] = {}
    for row in manifest.get("objects", []):
        if row.get("passed") is not True or row.get("method") != expected_method:
            raise RuntimeError(f"invalid {label} inference record")
        pair = (str(row["object_key"]), int(row["seed"]))
        if pair in rows:
            raise RuntimeError(f"duplicate {label} object/seed pair={pair}")
        rows[pair] = row
    seeds = [int(value) for value in manifest.get("seeds", [])]
    keys = {key for key, _ in rows}
    expected = {(key, seed) for key in keys for seed in seeds}
    if not rows or not seeds or len(seeds) != len(set(seeds)) or set(rows) != expected:
        raise RuntimeError(f"{label} lacks a complete object/seed product")
    if (
        int(manifest.get("object_count", -1)) != len(keys)
        or int(manifest.get("record_count", -1)) != len(rows)
    ):
        raise RuntimeError(f"{label} manifest counts differ from records")
    return rows


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    output = {
        field: summarize([float(row[field]) for row in rows])
        for field in (*SURFACE_FIELDS, *STRUCTURE_FIELDS)
    }
    output["mesh_success_rate"] = float(
        np.mean([float(row["mesh_success"]) for row in rows])
    )
    output["record_count"] = len(rows)
    return output


def _paired_delta(
    rows: list[dict[str, Any]], *, left: str, right: str
) -> dict[str, Any]:
    left_rows = {
        (str(row["object_key"]), int(row["seed"])): row
        for row in rows
        if row["method"] == left
    }
    right_rows = {
        (str(row["object_key"]), int(row["seed"])): row
        for row in rows
        if row["method"] == right
    }
    if set(left_rows) != set(right_rows):
        raise RuntimeError(f"paired coverage differs: {left}/{right}")
    metrics = {}
    for field in SURFACE_FIELDS:
        sign = -1.0 if field.startswith("chamfer") else 1.0
        values = [
            sign * (float(left_rows[pair][field]) - float(right_rows[pair][field]))
            for pair in sorted(left_rows)
        ]
        metrics[f"{field}_left_improvement"] = {
            **summarize(values),
            "positive_rate": float(np.mean(np.asarray(values) > 0.0)),
        }
    return {"left": left, "right": right, "metrics": metrics}


def adaptation_decision(comparison: dict[str, Any]) -> dict[str, Any]:
    metrics = comparison["metrics"]
    chamfer = metrics["chamfer_l1_left_improvement"]
    secondary = {
        "fscore_0p02_mean_nonnegative": (
            float(metrics["fscore_0p02_left_improvement"]["mean"]) >= 0.0
        ),
        "normal_consistency_mean_nonnegative": (
            float(metrics["normal_consistency_left_improvement"]["mean"]) >= 0.0
        ),
    }
    primary = {
        "chamfer_l1_mean_positive": float(chamfer["mean"]) > 0.0,
        "chamfer_l1_median_positive": float(chamfer["median"]) > 0.0,
        "chamfer_l1_object_win_rate_at_least_half": (
            float(chamfer["positive_rate"]) >= 0.5
        ),
    }
    return {
        "priority": (
            "Chamfer-L1 mean/median/object-win first; F-score@0.02 and normal "
            "consistency are secondary diagnostics"
        ),
        "thresholds": {
            "chamfer_l1_mean_improvement": 0.0,
            "chamfer_l1_median_improvement": 0.0,
            "chamfer_l1_min_object_win_rate": 0.5,
        },
        "primary_checks": primary,
        "secondary_checks": secondary,
        "primary_passed": all(primary.values()),
        "secondary_all_nonnegative": all(secondary.values()),
    }


def _formal_holdout_binding(
    *,
    split_path: Path,
    label_keys: set[str],
    runtime_path: Path,
    runtime: dict[str, Any],
    expected_objects: int,
) -> dict[str, Any]:
    split_path = split_path.expanduser().resolve()
    split = load_json(split_path)
    if (
        split.get("format") != SPLIT_ROWS_FORMAT
        or split.get("split") != "holdout"
        or split.get("training_ready") is not False
    ):
        raise RuntimeError("formal split is not the frozen holdout rows contract")
    split_rows = list(split.get("objects", []))
    split_keys = {object_key(row) for row in split_rows}
    if (
        len(split_rows) != expected_objects
        or int(split.get("object_count", -1)) != expected_objects
        or split_keys != label_keys
    ):
        raise RuntimeError("formal holdout object set differs from runtime-O labels")

    raw_path = validate_bound_file(
        runtime["raw_cache_report"],
        runtime["raw_cache_report_sha256"],
        label="formal holdout raw cache report",
    )
    raw = load_json(raw_path)
    if (
        raw.get("format") != RAW_CACHE_FORMAT
        or raw.get("passed") is not True
        or int(raw.get("object_count", -1)) != expected_objects
    ):
        raise RuntimeError("formal holdout raw cache contract did not pass")
    inventory_path = validate_bound_file(
        raw["inventory"], raw["inventory_sha256"], label="formal holdout inventory"
    )
    inventory = load_json(inventory_path)
    if (
        inventory.get("format") != SOURCE_INVENTORY_FORMAT
        or inventory.get("passed") is not True
        or int(inventory.get("video_object_count", -1)) != expected_objects
        or str(Path(inventory.get("source_split", "")).resolve()) != str(split_path)
        or inventory.get("source_split_sha256") != sha256_file(split_path)
        or inventory.get("split") != "holdout"
    ):
        raise RuntimeError("formal raw-cache inventory is not bound to holdout64")
    inventory_keys = {object_key(row) for row in inventory.get("objects", [])}
    if inventory_keys != split_keys:
        raise RuntimeError("formal inventory object set differs from holdout64")
    return {
        "frozen_split_manifest": str(split_path),
        "frozen_split_manifest_sha256": sha256_file(split_path),
        "runtime_input_manifest": str(runtime_path),
        "runtime_input_manifest_sha256": sha256_file(runtime_path),
        "raw_cache_report": str(raw_path),
        "raw_cache_report_sha256": sha256_file(raw_path),
        "extraction_inventory": str(inventory_path),
        "extraction_inventory_sha256": sha256_file(inventory_path),
        "object_count": expected_objects,
        "passed": True,
    }


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label_manifest", required=True)
    parser.add_argument("--adapted_native_manifest", required=True)
    parser.add_argument("--parent_native_manifest", required=True)
    parser.add_argument("--reconviagen_manifest", required=True)
    parser.add_argument("--pixal3d_manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--protocol_scope", choices=PROTOCOL_SCOPES, required=True)
    parser.add_argument("--frozen_split_manifest", default="")
    parser.add_argument("--expected_objects", type=int, required=True)
    parser.add_argument("--surface_samples", type=int, default=20000)
    return parser


def main() -> None:
    args = make_parser().parse_args()
    expected_objects = int(args.expected_objects)
    if expected_objects <= 0 or int(args.surface_samples) <= 0:
        raise ValueError("expected_objects and surface_samples must be positive")
    if args.protocol_scope == "development_benchmark32":
        if expected_objects != 32 or args.frozen_split_manifest:
            raise ValueError("development benchmark requires 32 objects and no split")
    elif expected_objects != 64 or not args.frozen_split_manifest:
        raise ValueError("formal holdout requires 64 objects and its frozen split")

    label_path = Path(args.label_manifest).expanduser().resolve()
    labels = load_json(label_path)
    if labels.get("format") != LABEL_MANIFEST_FORMAT or labels.get("passed") is not True:
        raise RuntimeError(f"runtime-O label manifest did not pass: {label_path}")
    label_by_key = index_objects(labels.get("objects", []), label="runtime-O labels")
    if len(label_by_key) != expected_objects:
        raise RuntimeError("runtime-O label count differs from frozen protocol")
    runtime_hash = str(labels["runtime_input_manifest_sha256"])
    runtime_path = validate_bound_file(
        labels["runtime_input_manifest"], runtime_hash, label="runtime-O label input"
    )
    runtime = load_json(runtime_path)
    if runtime.get("format") != RUNTIME_MANIFEST_FORMAT or runtime.get("passed") is not True:
        raise RuntimeError("runtime-O label input is not a passed v2 manifest")

    method_paths = {
        "adapted_native_v2_full": Path(args.adapted_native_manifest).resolve(),
        "parent_native_v2_full": Path(args.parent_native_manifest).resolve(),
        "reconviagen_original": Path(args.reconviagen_manifest).resolve(),
        "pixal3d_official": Path(args.pixal3d_manifest).resolve(),
    }
    pair_rows: dict[str, dict[tuple[str, int], dict[str, Any]]] = {}
    runtime_bindings = {}
    signature_cache: dict[str, dict[str, Any]] = {}
    for label, path in method_paths.items():
        base_method, record_method = METHOD_SPECS[label]
        manifest = load_json(path)
        if (
            manifest.get("format") != METHOD_FORMATS[base_method]
            or manifest.get("passed") is not True
            or manifest.get("target_or_metric_consumed") is not False
        ):
            raise RuntimeError(f"{label} inference contract did not pass: {path}")
        runtime_bindings[label] = validate_method_runtime_binding(
            base_method,
            manifest,
            reference_runtime_path=runtime_path,
            reference_runtime=runtime,
            reference_runtime_sha256=runtime_hash,
            signature_cache=signature_cache,
        )
        pair_rows[label] = _records_by_pair(
            manifest, expected_method=record_method, label=label
        )
    expected_pairs = set(pair_rows["adapted_native_v2_full"])
    if any(set(rows) != expected_pairs for rows in pair_rows.values()):
        raise RuntimeError("four methods do not cover identical object/seed pairs")
    if {key for key, _ in expected_pairs} != set(label_by_key):
        raise RuntimeError("inference and runtime-O label object sets differ")
    if len({seed for _, seed in expected_pairs}) != 1:
        raise RuntimeError("real adaptation protocol requires exactly one fixed seed")

    formal_binding = None
    if args.protocol_scope == "formal_holdout64":
        formal_binding = _formal_holdout_binding(
            split_path=Path(args.frozen_split_manifest),
            label_keys=set(label_by_key),
            runtime_path=runtime_path,
            runtime=runtime,
            expected_objects=expected_objects,
        )

    targets = {}
    target_bindings = {}
    for key, row in label_by_key.items():
        path = validate_bound_file(row["mesh_o"], row["mesh_o_sha256"], label=f"Mesh_O {key}")
        targets[key] = load_mesh(path)
        target_bindings[key] = {"path": str(path), "sha256": row["mesh_o_sha256"]}

    records: list[dict[str, Any]] = []
    for position, pair in enumerate(sorted(expected_pairs)):
        key, seed = pair
        surface_seed = int(seed) * 1009 + int(position) * 9173
        for method in METHOD_SPECS:
            source = pair_rows[method][pair]
            mesh_path = validate_bound_file(
                source["mesh"], source["mesh_sha256"], label=f"{method} {pair}"
            )
            mesh = load_mesh(mesh_path)
            structure = mesh_structure_metrics(mesh)
            surface = surface_metrics(
                mesh,
                targets[key],
                count=int(args.surface_samples),
                seed=surface_seed,
                thresholds=(0.01, 0.02),
            )
            records.append(
                {
                    "method": method,
                    "object_key": key,
                    "seed": seed,
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
            f"[real_adaptation_benchmark] {position + 1}/{len(expected_pairs)} "
            f"object={key} seed={seed}",
            flush=True,
        )

    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        by_method[row["method"]].append(row)
    summaries = {method: _summary(by_method[method]) for method in METHOD_SPECS}
    comparisons = [
        _paired_delta(records, left="adapted_native_v2_full", right=right)
        for right in (
            "parent_native_v2_full",
            "reconviagen_original",
            "pixal3d_official",
        )
    ]
    decision = adaptation_decision(comparisons[0])
    protocol_passed = all(
        values["mesh_success_rate"] == 1.0
        and values["record_count"] == len(expected_pairs)
        for values in summaries.values()
    )
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "format": REPORT_FORMAT,
        "created_at_utc": utc_now(),
        "formal": args.protocol_scope == "formal_holdout64",
        "protocol_scope": args.protocol_scope,
        "coordinate_policy": (
            "direct runtime-O/reference-view coordinates; no per-method GT ICP, "
            "scale fit, translation fit, or reflection"
        ),
        "label_manifest": str(label_path),
        "label_manifest_sha256": sha256_file(label_path),
        "runtime_input_manifest_sha256": runtime_hash,
        "formal_holdout_binding": formal_binding,
        "runtime_protocol_bindings": runtime_bindings,
        "method_manifests": {
            method: {"path": str(path), "sha256": sha256_file(path)}
            for method, path in method_paths.items()
        },
        "object_count": len(label_by_key),
        "seed_count": 1,
        "pair_count": len(expected_pairs),
        "surface_samples": int(args.surface_samples),
        "surface_thresholds": [0.01, 0.02],
        "summary": summaries,
        "paired_comparisons": comparisons,
        "adaptation_decision": decision,
        "holdout64_consumed": args.protocol_scope == "formal_holdout64",
        "passed": protocol_passed,
    }
    report_path = output_dir / "report.json"
    atomic_json(report_path, report)
    lines = [
        "Omni real Native v2 adaptation direct runtime-O Mesh evaluation",
        "=" * 66,
        f"scope: {args.protocol_scope}",
        f"objects: {report['object_count']} pairs: {report['pair_count']}",
        "No GT ICP/scale/translation/reflection is applied to any method.",
        "",
    ]
    for method, values in summaries.items():
        lines.extend(
            [
                method,
                f"  Chamfer-L1 mean/median: {values['chamfer_l1']['mean']:.8f} / {values['chamfer_l1']['median']:.8f}",
                f"  F-score@0.02 mean: {values['fscore_0p02']['mean']:.8f}",
                f"  normal consistency mean: {values['normal_consistency']['mean']:.8f}",
                f"  mesh success: {values['mesh_success_rate']:.6f}",
                "",
            ]
        )
    lines.append(f"adaptation primary passed: {decision['primary_passed']}")
    lines.append(f"secondary all nonnegative: {decision['secondary_all_nonnegative']}")
    (output_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(json.dumps({"passed": protocol_passed, "report": str(report_path)}, indent=2))
    if not protocol_passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
