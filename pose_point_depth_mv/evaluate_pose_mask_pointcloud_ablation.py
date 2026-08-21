#!/usr/bin/env python3
"""Paired Benchmark32-half evaluation of point+mask O versus pose+mask-only O."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from pose_point_depth_mv.dataset_tools.prepare_omni_real_label_cache import (
    MANIFEST_FORMAT as LABEL_MANIFEST_FORMAT,
)
from pose_point_depth_mv.dataset_tools.prepare_omni_real_runtime_inputs import (
    MANIFEST_FORMAT as RUNTIME_MANIFEST_FORMAT,
)
from pose_point_depth_mv.dataset_tools.prepare_omni_real_video_cache import utc_now
from pose_point_depth_mv.evaluate_omni_real_mesh_benchmark import (
    STRUCTURE_FIELDS,
    SURFACE_FIELDS,
    load_mesh,
    summarize,
)
from pose_point_depth_mv.mesh_benchmark_metrics import (
    mesh_structure_metrics,
    surface_metrics,
)
from pose_point_depth_mv.omni_real_benchmark_common import (
    atomic_json,
    index_objects,
    load_json,
    sha256_file,
    validate_bound_file,
)
from pose_point_depth_mv.rebase_pose_mask_inference_to_reference_o import (
    MANIFEST_FORMAT as REBASED_MANIFEST_FORMAT,
)


REPORT_FORMAT = "pose_point_depth_mv.pose_mask_pointcloud_ablation_benchmark.v1"
POSE_MASK_RUNTIME_VARIANT = "pose_point_depth_mv.omni_real_pose_mask_runtime_ablation.v1"
NO_VGGT_INFERENCE_MANIFEST_FORMAT = (
    "pose_point_depth_mv.omni_real_native_no_vggt_mixed_inference_manifest.v1"
)
METHODS = ("point_mask", "pose_mask")
DINO_MODEL_INPUT_MANIFEST_FORMAT = (
    "pose_point_depth_mv.omni_real_dino_only_model_input_manifest.v1"
)


def paired_metric_improvements(
    records: list[dict[str, Any]], *, left: str = "pose_mask", right: str = "point_mask"
) -> dict[str, Any]:
    """Summarize paired deltas; positive always means ``left`` is better."""

    left_rows = {
        (str(row["object_key"]), int(row["seed"])): row
        for row in records
        if row["method"] == left
    }
    right_rows = {
        (str(row["object_key"]), int(row["seed"])): row
        for row in records
        if row["method"] == right
    }
    if not left_rows or set(left_rows) != set(right_rows):
        raise RuntimeError(f"paired coverage differs: {left}/{right}")
    metrics: dict[str, Any] = {}
    for field in SURFACE_FIELDS:
        sign = -1.0 if field.startswith("chamfer") else 1.0
        values = [
            sign * (float(left_rows[pair][field]) - float(right_rows[pair][field]))
            for pair in sorted(left_rows)
        ]
        metrics[f"{field}_left_improvement"] = {
            **summarize(values),
            "positive_rate": float(np.mean(np.asarray(values) > 0.0)),
            "nonnegative_rate": float(np.mean(np.asarray(values) >= 0.0)),
        }
    return {
        "left": left,
        "right": right,
        "positive_definition": "positive means pose+mask is better than point+mask",
        "metrics": metrics,
    }


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


def _pair_records(
    rows: list[dict[str, Any]], *, expected_method: str, label: str
) -> dict[tuple[str, int], dict[str, Any]]:
    output: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        if row.get("passed") is not True or row.get("method") != expected_method:
            raise RuntimeError(f"invalid {label} inference record")
        pair = (str(row["object_key"]), int(row["seed"]))
        if pair in output:
            raise RuntimeError(f"duplicate {label} pair={pair}")
        output[pair] = row
    if not output:
        raise RuntimeError(f"{label} has no records")
    return output


def _surface_seed(key: str, seed: int) -> int:
    digest = hashlib.sha256(
        f"pose-mask-pointcloud-ablation-v1|{key}|{int(seed)}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:4], byteorder="big", signed=False)


def _assert_no_gt_keys(value: Any, *, location: str) -> None:
    forbidden = {
        "scan_obj",
        "mesh_o",
        "label_cache",
        "alignment_cache",
        "target_ss",
        "target_slat",
        "gt_mesh_bounds",
    }
    if isinstance(value, dict):
        overlap = forbidden.intersection(str(key).lower() for key in value)
        if overlap:
            raise RuntimeError(f"GT keys leaked into {location}: {sorted(overlap)}")
        for key, item in value.items():
            # The explicit forbidden-input declaration names forbidden fields but
            # does not carry their values and is part of the audit contract.
            if str(key) != "forbidden_inputs":
                _assert_no_gt_keys(item, location=location)
    elif isinstance(value, list):
        for item in value:
            _assert_no_gt_keys(item, location=location)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label_manifest", required=True)
    parser.add_argument("--reference_runtime_manifest", required=True)
    parser.add_argument("--pose_mask_runtime_manifest", required=True)
    parser.add_argument("--baseline_manifest", required=True)
    parser.add_argument("--pose_mask_rebased_manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--surface_samples", type=int, default=20000)
    parser.add_argument("--expected_objects", type=int, default=16)
    return parser


def main() -> None:
    args = make_parser().parse_args()
    if int(args.surface_samples) <= 0 or int(args.expected_objects) <= 0:
        raise ValueError("surface_samples and expected_objects must be positive")
    label_path = Path(args.label_manifest).expanduser().resolve()
    reference_path = Path(args.reference_runtime_manifest).expanduser().resolve()
    pose_runtime_path = Path(args.pose_mask_runtime_manifest).expanduser().resolve()
    baseline_path = Path(args.baseline_manifest).expanduser().resolve()
    rebased_path = Path(args.pose_mask_rebased_manifest).expanduser().resolve()
    labels = load_json(label_path)
    reference = load_json(reference_path)
    pose_runtime = load_json(pose_runtime_path)
    baseline = load_json(baseline_path)
    rebased = load_json(rebased_path)

    if labels.get("format") != LABEL_MANIFEST_FORMAT or labels.get("passed") is not True:
        raise RuntimeError(f"label manifest did not pass: {label_path}")
    if reference.get("format") != RUNTIME_MANIFEST_FORMAT or reference.get("passed") is not True:
        raise RuntimeError(f"reference runtime did not pass: {reference_path}")
    reference_hash = sha256_file(reference_path)
    if (
        Path(str(labels["runtime_input_manifest"])).resolve() != reference_path
        or str(labels["runtime_input_manifest_sha256"]) != reference_hash
    ):
        raise RuntimeError("labels are not bound to the requested reference runtime")
    if (
        pose_runtime.get("format") != RUNTIME_MANIFEST_FORMAT
        or pose_runtime.get("manifest_variant") != POSE_MASK_RUNTIME_VARIANT
        or pose_runtime.get("passed") is not True
        or pose_runtime.get("point_cloud_consumed") is not False
        or Path(str(pose_runtime["reference_runtime_manifest"])).resolve()
        != reference_path
        or str(pose_runtime["reference_runtime_manifest_sha256"]) != reference_hash
    ):
        raise RuntimeError("pose+mask runtime protocol binding failed")
    if (
        baseline.get("format") != NO_VGGT_INFERENCE_MANIFEST_FORMAT
        or baseline.get("passed") is not True
        or baseline.get("method") != "native_no_vggt_mixed"
        or baseline.get("target_or_metric_consumed") is not False
        or Path(str(baseline["runtime_input_manifest"])).resolve() != reference_path
        or str(baseline["runtime_input_manifest_sha256"]) != reference_hash
    ):
        raise RuntimeError("point+mask baseline protocol binding failed")
    if (
        int(baseline.get("object_count", -1)) != int(args.expected_objects)
        or int(baseline.get("record_count", -1)) != int(args.expected_objects)
    ):
        raise RuntimeError("point+mask baseline is not the paired half-set replay")
    if (
        rebased.get("format") != REBASED_MANIFEST_FORMAT
        or rebased.get("passed") is not True
        or rebased.get("method") != "native_no_vggt_pose_mask_rebased"
        or rebased.get("target_or_metric_consumed") is not False
        or rebased.get("point_cloud_consumed") is not False
        or Path(str(rebased["reference_runtime_manifest"])).resolve() != reference_path
        or str(rebased["reference_runtime_manifest_sha256"]) != reference_hash
        or Path(str(rebased["pose_mask_runtime_manifest"])).resolve()
        != pose_runtime_path
        or str(rebased["pose_mask_runtime_manifest_sha256"])
        != sha256_file(pose_runtime_path)
    ):
        raise RuntimeError("pose+mask rebased protocol binding failed")
    if (
        int(rebased.get("object_count", -1)) != int(args.expected_objects)
        or int(rebased.get("record_count", -1)) != int(args.expected_objects)
    ):
        raise RuntimeError("pose+mask rebased output is not the paired half-set")

    baseline_model_path = validate_bound_file(
        baseline["model_input_manifest"],
        baseline["model_input_manifest_sha256"],
        label="point+mask baseline model-input manifest",
    )
    pose_model_path = validate_bound_file(
        rebased["model_input_manifest"],
        rebased["model_input_manifest_sha256"],
        label="pose+mask model-input manifest",
    )
    baseline_model = load_json(baseline_model_path)
    pose_model = load_json(pose_model_path)
    for name, model in (("point+mask", baseline_model), ("pose+mask", pose_model)):
        if (
            model.get("format") != DINO_MODEL_INPUT_MANIFEST_FORMAT
            or model.get("passed") is not True
            or model.get("vggt_model_loaded") is not False
            or model.get("vggt_model_executed") is not False
        ):
            raise RuntimeError(f"{name} DINO-only model-input contract failed")
    for field in ("pretrained", "dino_model", "ss_context_tokens"):
        if baseline_model.get("config", {}).get(field) != pose_model.get("config", {}).get(
            field
        ):
            raise RuntimeError(f"A/B DINO configuration differs: {field}")
    if (
        Path(str(baseline_model["runtime_input_manifest"])).resolve() != reference_path
        or str(baseline_model["runtime_input_manifest_sha256"]) != reference_hash
        or Path(str(pose_model["runtime_input_manifest"])).resolve()
        != pose_runtime_path
        or str(pose_model["runtime_input_manifest_sha256"])
        != sha256_file(pose_runtime_path)
    ):
        raise RuntimeError("A/B DINO-only runtime bindings failed")

    reference_rows = index_objects(reference.get("objects", []), label="reference runtime")
    pose_rows = index_objects(pose_runtime.get("objects", []), label="pose+mask runtime")
    label_rows = index_objects(labels.get("objects", []), label="Benchmark32 labels")
    split_path = validate_bound_file(
        pose_runtime["ablation_split"],
        pose_runtime["ablation_split_sha256"],
        label="pose+mask ablation split",
    )
    split = load_json(split_path)
    selected_keys = [str(value) for value in split["selected_object_keys"]]
    if (
        len(selected_keys) != int(args.expected_objects)
        or len(selected_keys) != len(set(selected_keys))
        or set(selected_keys) != set(pose_rows)
        or not set(selected_keys).issubset(reference_rows)
        or not set(selected_keys).issubset(label_rows)
    ):
        raise RuntimeError("frozen 16-object split coverage failed")
    baseline_model_rows = index_objects(
        baseline_model.get("objects", []), label="point+mask model inputs"
    )
    pose_model_rows = index_objects(
        pose_model.get("objects", []), label="pose+mask model inputs"
    )
    if set(baseline_model_rows) != set(selected_keys) or set(pose_model_rows) != set(
        selected_keys
    ):
        raise RuntimeError("A/B model-input object sets differ from the frozen split")
    for key in selected_keys:
        point_input = baseline_model_rows[key]
        pose_input = pose_model_rows[key]
        if (
            point_input["prepared_rgb_paths"] != pose_input["prepared_rgb_paths"]
            or point_input["prepared_mask_paths"] != pose_input["prepared_mask_paths"]
            or point_input["feature_contract"] != pose_input["feature_contract"]
        ):
            raise RuntimeError(f"A/B visible DINO inputs differ: {key}")
    for key, row in pose_rows.items():
        condition_path = validate_bound_file(
            row["condition_record"],
            sha256_file(row["condition_record"]),
            label=f"pose+mask condition {key}",
        )
        condition = load_json(condition_path)
        if (
            row.get("point_cloud_consumed") is not False
            or row.get("point_cloud_fields_read") != []
            or row.get("forbidden_gt_fields_absent") is not True
            or not all(row.get("external_visible_equivalence", {}).values())
            or condition.get("point_cloud_consumed") is not False
        ):
            raise RuntimeError(f"pose+mask no-point contract failed: {key}")
        _assert_no_gt_keys(condition, location=f"pose+mask condition {key}")
        label_runtime = validate_bound_file(
            label_rows[key]["runtime_cache"],
            label_rows[key]["runtime_cache_sha256"],
            label=f"label-bound reference runtime cache {key}",
        )
        if label_runtime != Path(reference_rows[key]["cache_npz"]).resolve():
            raise RuntimeError(f"label/reference runtime cache differs: {key}")

    baseline_pairs_all = _pair_records(
        baseline.get("objects", []),
        expected_method="native_no_vggt_mixed",
        label="point+mask baseline",
    )
    pose_pairs = _pair_records(
        rebased.get("objects", []),
        expected_method="native_no_vggt_pose_mask_rebased",
        label="pose+mask rebased",
    )
    expected_pairs = set(pose_pairs)
    if (
        len(expected_pairs) != int(args.expected_objects)
        or {key for key, _ in expected_pairs} != set(selected_keys)
        or {seed for _, seed in expected_pairs} != {42}
        or not expected_pairs.issubset(baseline_pairs_all)
    ):
        raise RuntimeError("paired object/seed coverage is not frozen 16 x seed42")
    baseline_pairs = {pair: baseline_pairs_all[pair] for pair in expected_pairs}
    if [int(value) for value in baseline.get("seeds", [])] != [42] or [
        int(value) for value in rebased.get("seeds", [])
    ] != [42]:
        raise RuntimeError("A/B inference seeds are not exactly [42]")

    invariant_fields = (
        "native_ss_checkpoint_sha256",
        "native_ss_weights",
        "native_slat_checkpoint_sha256",
        "native_slat_weights",
        "stock_slat_freeze_sha256",
        "sampling_sha256",
        "post_cfg_cap",
    )
    for pair in sorted(expected_pairs):
        left = baseline_pairs[pair]
        right = pose_pairs[pair]
        changed = [field for field in invariant_fields if left[field] != right[field]]
        if changed:
            raise RuntimeError(f"A/B model or sampler changed for {pair}: {changed}")
        key, _ = pair
        if (
            left["model_input_sha256"] != baseline_model_rows[key]["model_input_sha256"]
            or right["model_input_sha256"] != pose_model_rows[key]["model_input_sha256"]
        ):
            raise RuntimeError(f"A/B inference model-input binding changed: {pair}")

    records: list[dict[str, Any]] = []
    target_bindings: dict[str, dict[str, str]] = {}
    for position, pair in enumerate(sorted(expected_pairs), start=1):
        key, seed = pair
        label_row = label_rows[key]
        target_path = validate_bound_file(
            label_row["mesh_o"], label_row["mesh_o_sha256"], label=f"Mesh_O {key}"
        )
        target = load_mesh(target_path)
        target_bindings[key] = {
            "path": str(target_path),
            "sha256": str(label_row["mesh_o_sha256"]),
        }
        sample_seed = _surface_seed(key, seed)
        for method, source in (
            ("point_mask", baseline_pairs[pair]),
            ("pose_mask", pose_pairs[pair]),
        ):
            mesh_path = validate_bound_file(
                source["mesh"], source["mesh_sha256"], label=f"{method} {pair}"
            )
            mesh = load_mesh(mesh_path)
            structure = mesh_structure_metrics(mesh)
            surface = surface_metrics(
                mesh,
                target,
                count=int(args.surface_samples),
                seed=sample_seed,
                thresholds=(0.01, 0.02),
            )
            records.append(
                {
                    "method": method,
                    "object_key": key,
                    "seed": seed,
                    "mesh": str(mesh_path),
                    "mesh_sha256": str(source["mesh_sha256"]),
                    "target_mesh": target_bindings[key],
                    "surface_seed": sample_seed,
                    "mesh_success": bool(structure["mesh_success"]),
                    **{field: float(surface[field]) for field in SURFACE_FIELDS},
                    **{field: float(structure[field]) for field in STRUCTURE_FIELDS},
                }
            )
        print(
            f"[pose_mask_ablation] {position}/{len(expected_pairs)} "
            f"object={key} seed={seed}",
            flush=True,
        )

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[str(row["method"])].append(row)
    comparison = paired_metric_improvements(records)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "format": REPORT_FORMAT,
        "created_at_utc": utc_now(),
        "passed": True,
        "formal": False,
        "protocol_scope": "Benchmark32 deterministic half-set development ablation",
        "question": "effect of removing point cloud from runtime-O construction",
        "method_definition": {
            "point_mask": "existing M9 point+mask runtime-O baseline",
            "pose_mask": (
                "same no-VGGT model with pose+mask-only O, then observable "
                "O_posemask->W->O_reference rebase"
            ),
        },
        "coordinate_policy": (
            "both meshes evaluated in the original Benchmark32 reference O; "
            "no GT ICP, scale fit, translation fit, reflection, or per-method alignment"
        ),
        "frozen_invariants": list(invariant_fields),
        "noise_contract": (
            "both inference manifests contain the same 16 object keys; shared select_rows "
            "sorts them identically, so seed42 and position-derived SS/SLat noise are equal"
        ),
        "surface_samples": int(args.surface_samples),
        "surface_seed_policy": "sha256(protocol|object_key|inference_seed), shared by A/B",
        "object_count": len(selected_keys),
        "record_count": len(records),
        "selected_object_keys": selected_keys,
        "bindings": {
            "label_manifest": {"path": str(label_path), "sha256": sha256_file(label_path)},
            "reference_runtime_manifest": {
                "path": str(reference_path),
                "sha256": reference_hash,
            },
            "pose_mask_runtime_manifest": {
                "path": str(pose_runtime_path),
                "sha256": sha256_file(pose_runtime_path),
            },
            "baseline_manifest": {
                "path": str(baseline_path),
                "sha256": sha256_file(baseline_path),
            },
            "pose_mask_rebased_manifest": {
                "path": str(rebased_path),
                "sha256": sha256_file(rebased_path),
            },
            "ablation_split": {"path": str(split_path), "sha256": sha256_file(split_path)},
        },
        "summary": {method: _summary(grouped[method]) for method in METHODS},
        "paired_comparison": comparison,
        "records": records,
        "interpretation_guard": (
            "Protocol pass does not imply pose+mask wins. Read paired metric means, "
            "confidence is limited to this development subset, and holdout64 is untouched."
        ),
    }
    report_path = output_dir / "report.json"
    atomic_json(report_path, report)
    print(
        json.dumps(
            {
                "passed": True,
                "formal": False,
                "objects": len(selected_keys),
                "paired_comparison": comparison,
                "report": str(report_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
