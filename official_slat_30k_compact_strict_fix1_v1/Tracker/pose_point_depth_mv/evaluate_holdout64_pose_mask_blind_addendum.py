#!/usr/bin/env python3
"""One-shot six-way formal evaluation for the blind Pose+Mask Holdout64 addendum."""

from __future__ import annotations

import argparse
from collections import defaultdict
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
    METHOD_FORMATS,
    STRUCTURE_FIELDS,
    SURFACE_FIELDS,
    load_mesh,
    summarize,
    validate_method_runtime_binding,
)
from pose_point_depth_mv.evaluate_omni_real_native_adaptation import (
    _formal_holdout_binding,
)
from pose_point_depth_mv.evaluate_omni_real_no_vggt_final import (
    NO_VGGT_INFERENCE_MANIFEST_FORMAT,
    _validate_no_vggt_runtime_binding,
)
from pose_point_depth_mv.freeze_holdout64_pose_mask_blind_protocol import (
    validate_protocol_contract,
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
from pose_point_depth_mv.rebase_pose_mask_inference_to_reference_o import (
    MANIFEST_FORMAT as POSE_MASK_REBASED_MANIFEST_FORMAT,
)


REPORT_FORMAT = "pose_point_depth_mv.holdout64_pose_mask_blind_addendum.v1"
EXPECTED_OBJECTS = 64
EXPECTED_SEED = 42
METHOD_SPECS = {
    "point_mask": {
        "format": NO_VGGT_INFERENCE_MANIFEST_FORMAT,
        "record_method": "native_no_vggt_mixed",
        "runtime_method": "native_no_vggt",
    },
    "pose_mask": {
        "format": POSE_MASK_REBASED_MANIFEST_FORMAT,
        "record_method": "native_no_vggt_pose_mask_rebased",
        "runtime_method": "pose_mask_rebased",
    },
    "real_adapted_native_v2_full": {
        "format": METHOD_FORMATS["native_v2_full"],
        "record_method": "native_v2_full",
        "runtime_method": "native_v2_full",
    },
    "synthetic_parent_native_v2_full": {
        "format": METHOD_FORMATS["native_v2_full"],
        "record_method": "native_v2_full",
        "runtime_method": "native_v2_full",
    },
    "reconviagen_original": {
        "format": METHOD_FORMATS["reconviagen_original"],
        "record_method": "reconviagen_original",
        "runtime_method": "reconviagen_original",
    },
    "pixal3d_official": {
        "format": METHOD_FORMATS["pixal3d_official"],
        "record_method": "pixal3d_official_single_reference_view",
        "runtime_method": "pixal3d_official",
    },
}


def records_by_pair(
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
    keys = {key for key, _ in rows}
    if (
        manifest.get("seeds") != [EXPECTED_SEED]
        or len(keys) != EXPECTED_OBJECTS
        or len(rows) != EXPECTED_OBJECTS
        or int(manifest.get("object_count", -1)) != EXPECTED_OBJECTS
        or int(manifest.get("record_count", -1)) != EXPECTED_OBJECTS
    ):
        raise RuntimeError(f"{label} does not provide exact 64xseed42 coverage")
    return rows


def paired_method_comparison(
    rows: list[dict[str, Any]], *, left: str, right: str
) -> dict[str, Any]:
    by_method = {
        method: {
            (str(row["object_key"]), int(row["seed"])): row
            for row in rows
            if row["method"] == method
        }
        for method in (left, right)
    }
    if set(by_method[left]) != set(by_method[right]):
        raise RuntimeError(f"paired coverage differs: {left}/{right}")
    metrics: dict[str, Any] = {}
    for field in SURFACE_FIELDS:
        sign = -1.0 if field.startswith("chamfer") else 1.0
        values = np.asarray(
            [
                sign
                * (
                    float(by_method[left][pair][field])
                    - float(by_method[right][pair][field])
                )
                for pair in sorted(by_method[left])
            ],
            dtype=np.float64,
        )
        metrics[f"{field}_left_improvement"] = {
            **summarize(values.tolist()),
            "positive_rate": float(np.mean(values > 0.0)),
            "nonnegative_rate": float(np.mean(values >= 0.0)),
            "left_win_count": int(np.sum(values > 0.0)),
            "tie_count": int(np.sum(values == 0.0)),
            "right_win_count": int(np.sum(values < 0.0)),
        }
    return {
        "left": left,
        "right": right,
        "positive_definition": "positive means Pose+Mask is better",
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


def _pose_runtime_binding(
    manifest: dict[str, Any], *, reference_path: Path, reference_sha256: str
) -> dict[str, Any]:
    if (
        manifest.get("formal") is not True
        or manifest.get("protocol_scope") != "formal_holdout64_blind_addendum"
        or manifest.get("point_cloud_consumed") is not False
        or manifest.get("target_or_metric_consumed") is not False
        or manifest.get("formal_holdout_binding", {}).get("passed") is not True
    ):
        raise RuntimeError("Pose+Mask rebased manifest lacks its formal blind contract")
    bound_reference = validate_bound_file(
        manifest.get("reference_runtime_manifest", ""),
        str(manifest.get("reference_runtime_manifest_sha256", "")),
        label="Pose+Mask reference runtime",
    )
    if bound_reference != reference_path or sha256_file(bound_reference) != reference_sha256:
        raise RuntimeError("Pose+Mask output is not in the frozen reference runtime-O")
    pose_runtime_path = validate_bound_file(
        manifest.get("pose_mask_runtime_manifest", ""),
        str(manifest.get("pose_mask_runtime_manifest_sha256", "")),
        label="Pose+Mask runtime",
    )
    pose_runtime = load_json(pose_runtime_path)
    if (
        pose_runtime.get("formal") is not True
        or pose_runtime.get("point_cloud_consumed") is not False
        or pose_runtime.get("gt_consumed") is not False
        or pose_runtime.get("old_mesh_consumed") is not False
        or pose_runtime.get("metric_or_ranking_consumed") is not False
    ):
        raise RuntimeError("Pose+Mask runtime consumed a forbidden formal input")
    return {
        "binding_mode": "observable O_posemask-to-W-to-O_reference",
        "reference_runtime_manifest": str(reference_path),
        "reference_runtime_manifest_sha256": reference_sha256,
        "pose_mask_runtime_manifest": str(pose_runtime_path),
        "pose_mask_runtime_manifest_sha256": sha256_file(pose_runtime_path),
        "point_cloud_consumed": False,
        "gt_fit_applied": False,
        "passed": True,
    }


def _validate_point_pose_sampling(
    point_rows: dict[tuple[str, int], dict[str, Any]],
    pose_rows: dict[tuple[str, int], dict[str, Any]],
    *,
    expected_ss_sha256: str,
    expected_slat_sha256: str,
    expected_stock_freeze_sha256: str,
) -> dict[str, Any]:
    fields = (
        "native_ss_checkpoint_sha256",
        "native_ss_weights",
        "native_slat_checkpoint_sha256",
        "native_slat_weights",
        "stock_slat_freeze_sha256",
        "sampling_sha256",
        "post_cfg_cap",
        "condition_scale_policy",
    )
    expected_sampling = {
        "steps": 25,
        "cfg_strength": 5.0,
        "cfg_interval": [0.5, 1.0],
        "guidance_rescale": 0.0,
        "rescale_t": 3.0,
    }
    def field_value(row: dict[str, Any], field: str) -> Any:
        if field == "condition_scale_policy":
            return row.get(field, row.get("wrapper", {}).get(field))
        return row.get(field)

    for pair in sorted(point_rows):
        mismatch = {
            field: (
                field_value(point_rows[pair], field),
                field_value(pose_rows[pair], field),
            )
            for field in fields
            if field_value(point_rows[pair], field)
            != field_value(pose_rows[pair], field)
        }
        if mismatch:
            raise RuntimeError(f"Point/Pose sampling differs for {pair}: {mismatch}")
        for label, row in (("Point+Mask", point_rows[pair]), ("Pose+Mask", pose_rows[pair])):
            actual_sampling = dict(row.get("sampling", {}))
            for name, expected in expected_sampling.items():
                actual = actual_sampling.get(name)
                if name == "cfg_interval" and isinstance(actual, tuple):
                    actual = list(actual)
                if actual != expected:
                    raise RuntimeError(
                        f"{label} frozen SLat sampling changed for {pair}: "
                        f"{name}={actual}"
                    )
            if (
                row.get("native_ss_checkpoint_sha256") != expected_ss_sha256
                or row.get("native_slat_checkpoint_sha256") != expected_slat_sha256
                or row.get("stock_slat_freeze_sha256")
                != expected_stock_freeze_sha256
                or row.get("native_ss_weights") != "ema"
                or row.get("native_slat_weights") != "ema"
                or row.get("post_cfg_cap") is not False
                or field_value(row, "condition_scale_policy")
                != "learned_projection_only"
            ):
                raise RuntimeError(f"{label} deployment contract changed for {pair}")
    return {
        "matched_fields": list(fields),
        "pair_count": len(point_rows),
        "same_object_position_required": True,
        "expected_sampling": expected_sampling,
        "expected_ss_checkpoint_sha256": expected_ss_sha256,
        "expected_slat_checkpoint_sha256": expected_slat_sha256,
        "expected_stock_slat_freeze_sha256": expected_stock_freeze_sha256,
        "weights": "ema",
        "passed": True,
    }


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blind_protocol_contract", required=True)
    parser.add_argument("--frozen_split_manifest", required=True)
    parser.add_argument("--label_manifest", required=True)
    parser.add_argument("--reference_runtime_manifest", required=True)
    parser.add_argument("--point_mask_manifest", required=True)
    parser.add_argument("--pose_mask_rebased_manifest", required=True)
    parser.add_argument("--real_full_manifest", required=True)
    parser.add_argument("--synthetic_full_manifest", required=True)
    parser.add_argument("--reconviagen_manifest", required=True)
    parser.add_argument("--pixal3d_manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--surface_samples", type=int, default=20000)
    return parser


def main() -> None:
    args = make_parser().parse_args()
    if int(args.surface_samples) != 20000:
        raise ValueError("formal blind addendum freezes surface_samples=20000")
    protocol_path = Path(args.blind_protocol_contract).expanduser().resolve()
    protocol = validate_protocol_contract(protocol_path)
    output_dir = Path(args.output_dir).expanduser().resolve()
    report_path = output_dir / "report.json"
    if report_path.exists():
        raise RuntimeError("one-shot blind report already exists; refusing to recompute")

    reference_path = Path(args.reference_runtime_manifest).expanduser().resolve()
    reference = load_json(reference_path)
    if reference.get("format") != RUNTIME_MANIFEST_FORMAT or reference.get("passed") is not True:
        raise RuntimeError("reference runtime manifest did not pass")
    reference_sha256 = sha256_file(reference_path)
    reference_order = [object_key(row) for row in reference.get("objects", [])]
    if len(reference_order) != EXPECTED_OBJECTS or len(set(reference_order)) != EXPECTED_OBJECTS:
        raise RuntimeError("reference runtime is not an ordered Holdout64")

    label_path = Path(args.label_manifest).expanduser().resolve()
    labels = load_json(label_path)
    if labels.get("format") != LABEL_MANIFEST_FORMAT or labels.get("passed") is not True:
        raise RuntimeError("runtime-O labels did not pass")
    label_by_key = index_objects(labels.get("objects", []), label="runtime-O labels")
    if set(label_by_key) != set(reference_order):
        raise RuntimeError("runtime-O labels and reference runtime differ")
    if (
        Path(str(labels.get("runtime_input_manifest", ""))).resolve() != reference_path
        or labels.get("runtime_input_manifest_sha256") != reference_sha256
    ):
        raise RuntimeError("labels are not bound to the frozen reference runtime")

    method_paths = {
        "point_mask": Path(args.point_mask_manifest).resolve(),
        "pose_mask": Path(args.pose_mask_rebased_manifest).resolve(),
        "real_adapted_native_v2_full": Path(args.real_full_manifest).resolve(),
        "synthetic_parent_native_v2_full": Path(args.synthetic_full_manifest).resolve(),
        "reconviagen_original": Path(args.reconviagen_manifest).resolve(),
        "pixal3d_official": Path(args.pixal3d_manifest).resolve(),
    }
    frozen_required = protocol["required_existing_inference_manifests"]
    for method, path in method_paths.items():
        if method != "pose_mask" and path != Path(frozen_required[method]).resolve():
            raise RuntimeError(f"{method} path differs from frozen blind protocol")

    pair_rows: dict[str, dict[tuple[str, int], dict[str, Any]]] = {}
    runtime_bindings: dict[str, dict[str, Any]] = {}
    manifests: dict[str, dict[str, Any]] = {}
    signature_cache: dict[str, dict[str, Any]] = {}
    for method, path in method_paths.items():
        manifest = load_json(path)
        spec = METHOD_SPECS[method]
        if manifest.get("format") != spec["format"] or manifest.get("passed") is not True:
            raise RuntimeError(f"{method} inference manifest did not pass")
        if manifest.get("target_or_metric_consumed") is not False:
            raise RuntimeError(f"{method} inference consumed target or metric")
        pair_rows[method] = records_by_pair(
            manifest, expected_method=str(spec["record_method"]), label=method
        )
        ordered_keys = [str(row["object_key"]) for row in manifest.get("objects", [])]
        if ordered_keys != reference_order:
            raise RuntimeError(f"{method} object order differs from M11C")
        if method == "point_mask":
            runtime_bindings[method] = _validate_no_vggt_runtime_binding(
                manifest,
                reference_runtime_path=reference_path,
                reference_runtime_sha256=reference_sha256,
            )
        elif method == "pose_mask":
            runtime_bindings[method] = _pose_runtime_binding(
                manifest,
                reference_path=reference_path,
                reference_sha256=reference_sha256,
            )
        else:
            runtime_bindings[method] = validate_method_runtime_binding(
                str(spec["runtime_method"]),
                manifest,
                reference_runtime_path=reference_path,
                reference_runtime=reference,
                reference_runtime_sha256=reference_sha256,
                signature_cache=signature_cache,
            )
        manifests[method] = manifest

    expected_pairs = {(key, EXPECTED_SEED) for key in reference_order}
    if any(set(rows) != expected_pairs for rows in pair_rows.values()):
        raise RuntimeError("six methods do not cover the same 64xseed42 pairs")
    sampling_binding = _validate_point_pose_sampling(
        pair_rows["point_mask"],
        pair_rows["pose_mask"],
        expected_ss_sha256=protocol["frozen_inputs"]
        ["mixed_no_vggt_ss_ema_checkpoint"]["sha256"],
        expected_slat_sha256=protocol["frozen_inputs"]
        ["mixed_no_vggt_slat_ema_checkpoint"]["sha256"],
        expected_stock_freeze_sha256=protocol["frozen_inputs"]
        ["stock_slat_freeze"]["sha256"],
    )
    formal_binding = _formal_holdout_binding(
        split_path=Path(args.frozen_split_manifest),
        label_keys=set(reference_order),
        runtime_path=reference_path,
        runtime=reference,
        expected_objects=EXPECTED_OBJECTS,
    )

    quality_by_key = {
        key: row.get("alignment_quality_passed", True) is True
        for key, row in label_by_key.items()
    }
    warning_keys = sorted(key for key, passed in quality_by_key.items() if not passed)
    if warning_keys and (
        labels.get("alignment_quality_warnings_included") is not True
        or sorted(labels.get("alignment_quality_warning_object_keys", []))
        != warning_keys
    ):
        raise RuntimeError("label quality warning disclosure is incomplete")

    targets = {
        key: load_mesh(
            validate_bound_file(
                row["mesh_o"], row["mesh_o_sha256"], label=f"Mesh_O {key}"
            )
        )
        for key, row in label_by_key.items()
    }
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
                    "alignment_quality_tier": (
                        "reliable" if quality_by_key[key] else "low_confidence"
                    ),
                    "surface_seed": surface_seed,
                    "mesh_success": bool(structure["mesh_success"]),
                    **{field: float(surface[field]) for field in SURFACE_FIELDS},
                    **{field: float(structure[field]) for field in STRUCTURE_FIELDS},
                }
            )
        print(f"[blind_joint_eval] {position + 1}/{EXPECTED_OBJECTS}", flush=True)

    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        by_method[str(row["method"])].append(row)
    summaries = {method: _summary(by_method[method]) for method in METHOD_SPECS}
    comparisons = {
        method: paired_method_comparison(records, left="pose_mask", right=method)
        for method in METHOD_SPECS
        if method != "pose_mask"
    }
    subgroup_reports: dict[str, Any] = {}
    for group, keys in {
        "reliable": sorted(key for key, passed in quality_by_key.items() if passed),
        "low_confidence": warning_keys,
    }.items():
        selected = [row for row in records if row["object_key"] in set(keys)]
        subgroup_reports[group] = {
            "object_count": len(keys),
            "object_keys": keys,
            "summary": (
                {
                    method: _summary([row for row in selected if row["method"] == method])
                    for method in METHOD_SPECS
                }
                if keys
                else {}
            ),
            "pose_mask_paired_comparisons": (
                {
                    method: paired_method_comparison(
                        selected, left="pose_mask", right=method
                    )
                    for method in METHOD_SPECS
                    if method != "pose_mask"
                }
                if keys
                else {}
            ),
        }
    protocol_passed = (
        len(records) == EXPECTED_OBJECTS * len(METHOD_SPECS)
        and all(summary["mesh_success_rate"] == 1.0 for summary in summaries.values())
        and formal_binding["passed"] is True
        and sampling_binding["passed"] is True
    )
    report = {
        "format": REPORT_FORMAT,
        "created_at_utc": utc_now(),
        "formal": True,
        "protocol_scope": "formal_holdout64_blind_addendum",
        "blind_protocol_contract": {
            "path": str(protocol_path),
            "sha256": sha256_file(protocol_path),
            "payload_sha256": protocol["payload_sha256"],
        },
        "coordinate_policy": (
            "Point and external methods remain in M11C runtime-O; Pose+Mask uses only "
            "O_posemask->W->O_reference. No GT ICP, scale, translation, or reflection fit."
        ),
        "label_manifest": str(label_path),
        "label_manifest_sha256": sha256_file(label_path),
        "reference_runtime_manifest": str(reference_path),
        "reference_runtime_manifest_sha256": reference_sha256,
        "formal_holdout_binding": formal_binding,
        "runtime_protocol_bindings": runtime_bindings,
        "point_pose_sampling_binding": sampling_binding,
        "method_manifests": {
            method: {"path": str(path), "sha256": sha256_file(path)}
            for method, path in method_paths.items()
        },
        "object_count": EXPECTED_OBJECTS,
        "seed_count": 1,
        "seeds": [EXPECTED_SEED],
        "record_count": len(records),
        "surface_samples": int(args.surface_samples),
        "surface_thresholds": [0.01, 0.02],
        "surface_seed_policy": "seed*1009 + sorted_pair_position*9173",
        "primary_population": "all64",
        "label_quality_protocol": {
            "reliable_object_count": EXPECTED_OBJECTS - len(warning_keys),
            "low_confidence_object_count": len(warning_keys),
            "low_confidence_object_keys": warning_keys,
            "low_confidence_objects_retained": bool(warning_keys),
            "selection_replacement_or_reweighting": False,
        },
        "summary": summaries,
        "pose_mask_paired_comparisons": comparisons,
        "label_quality_subgroups": subgroup_reports,
        "records": records,
        "passed_semantics": "protocol and mesh completeness only; not method victory",
        "unblinding_required": True,
        "passed": protocol_passed,
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    atomic_json(report_path, report)
    print(
        {
            "passed": protocol_passed,
            "formal": True,
            "object_count": EXPECTED_OBJECTS,
            "record_count": len(records),
            "metrics_printed": False,
            "report": str(report_path),
        }
    )
    if not protocol_passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
