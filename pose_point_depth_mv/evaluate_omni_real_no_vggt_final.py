#!/usr/bin/env python3
"""Evaluate final mixed no-VGGT against Full parents and frozen image bases."""

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


REPORT_FORMAT = "pose_point_depth_mv.omni_real_no_vggt_final_benchmark.v1"
DINO_MODEL_INPUT_MANIFEST_FORMAT = (
    "pose_point_depth_mv.omni_real_dino_only_model_input_manifest.v1"
)
NO_VGGT_INFERENCE_MANIFEST_FORMAT = (
    "pose_point_depth_mv.omni_real_native_no_vggt_mixed_inference_manifest.v1"
)
PROTOCOL_SCOPES = ("development_benchmark32", "formal_holdout64")
METHOD_SPECS = {
    "final_native_no_vggt": {
        "format": NO_VGGT_INFERENCE_MANIFEST_FORMAT,
        "record_method": "native_no_vggt_mixed",
        "runtime_method": "native_no_vggt",
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


def _validate_no_vggt_runtime_binding(
    manifest: dict[str, Any],
    *,
    reference_runtime_path: Path,
    reference_runtime_sha256: str,
) -> dict[str, Any]:
    if manifest.get("method") != "native_no_vggt_mixed":
        raise RuntimeError("final no-VGGT manifest method differs")
    if (
        manifest.get("vggt_model_loaded") is not False
        or manifest.get("vggt_model_executed") is not False
    ):
        raise RuntimeError("final no-VGGT inference did not freeze VGGT off")
    runtime_path = validate_bound_file(
        manifest.get("runtime_input_manifest", ""),
        str(manifest.get("runtime_input_manifest_sha256", "")),
        label="final no-VGGT runtime input",
    )
    if (
        runtime_path != reference_runtime_path
        or str(manifest.get("runtime_input_manifest_sha256"))
        != str(reference_runtime_sha256)
    ):
        raise RuntimeError("final no-VGGT must bind the exact runtime-O label input")
    model_path = validate_bound_file(
        manifest.get("model_input_manifest", ""),
        str(manifest.get("model_input_manifest_sha256", "")),
        label="final no-VGGT DINO model input",
    )
    model = load_json(model_path)
    if (
        model.get("format") != DINO_MODEL_INPUT_MANIFEST_FORMAT
        or model.get("passed") is not True
        or model.get("vggt_model_loaded") is not False
        or model.get("vggt_model_executed") is not False
        or model.get("runtime_input_manifest_sha256") != reference_runtime_sha256
        or Path(str(model.get("runtime_input_manifest", ""))).resolve()
        != reference_runtime_path
    ):
        raise RuntimeError("final no-VGGT DINO-only model input contract failed")
    model_rows = index_objects(model.get("objects", []), label="DINO-only model input")
    inference_keys = {str(row["object_key"]) for row in manifest.get("objects", [])}
    if set(model_rows) != inference_keys:
        raise RuntimeError("no-VGGT inference and DINO model-input object sets differ")
    for key, row in model_rows.items():
        if (
            row.get("passed") is not True
            or row.get("vggt_model_loaded") is not False
            or row.get("vggt_model_executed") is not False
            or row.get("target_or_mesh_consumed") is not False
        ):
            raise RuntimeError(f"DINO-only model input record failed: {key}")
    for row in manifest.get("objects", []):
        if (
            row.get("vggt_model_loaded") is not False
            or row.get("vggt_model_executed") is not False
        ):
            raise RuntimeError("a no-VGGT inference result executed VGGT")
    for stage in ("ss", "slat"):
        contract = manifest.get(f"{stage}_migration_contract")
        if (
            not isinstance(contract, dict)
            or contract.get("selected_weights") != "ema"
            or contract.get("optimizer_inherited") is not False
            or contract.get("step_inherited") is not False
        ):
            raise RuntimeError(f"final no-VGGT {stage} migration lineage failed")
    return {
        "binding_mode": "exact_runtime_and_dino_only_manifest_sha256",
        "runtime_input_manifest": str(runtime_path),
        "runtime_input_manifest_sha256": str(reference_runtime_sha256),
        "model_input_manifest": str(model_path),
        "model_input_manifest_sha256": sha256_file(model_path),
        "object_count": len(model_rows),
        "vggt_model_loaded": False,
        "vggt_model_executed": False,
        "migration_contracts_verified": ["ss", "slat"],
        "passed": True,
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


def _quality_subgroup_report(
    records: list[dict[str, Any]],
    *,
    quality_by_key: dict[str, bool],
) -> dict[str, Any]:
    groups = {
        "reliable": sorted(key for key, passed in quality_by_key.items() if passed),
        "low_confidence": sorted(
            key for key, passed in quality_by_key.items() if not passed
        ),
    }
    output: dict[str, Any] = {}
    for name, keys in groups.items():
        key_set = set(keys)
        group_rows = [row for row in records if str(row["object_key"]) in key_set]
        if not keys:
            output[name] = {
                "object_count": 0,
                "object_keys": [],
                "summary": {},
                "paired_comparisons": [],
            }
            continue
        by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in group_rows:
            by_method[str(row["method"])].append(row)
        output[name] = {
            "object_count": len(keys),
            "object_keys": keys,
            "summary": {
                method: _summary(by_method[method]) for method in METHOD_SPECS
            },
            "paired_comparisons": [
                _paired_delta(
                    group_rows, left="final_native_no_vggt", right=right
                )
                for right in METHOD_SPECS
                if right != "final_native_no_vggt"
            ],
        }
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


def _safe_ratio(left: float, right: float) -> float:
    if not np.isfinite(left) or not np.isfinite(right) or left < 0.0 or right < 0.0:
        raise ValueError(f"invalid metric ratio inputs: {left}/{right}")
    if right == 0.0:
        return 1.0 if left == 0.0 else float("inf")
    return float(left / right)


def no_vggt_decision(
    summaries: dict[str, dict[str, Any]], comparison: dict[str, Any]
) -> dict[str, Any]:
    final = summaries["final_native_no_vggt"]
    full = summaries["real_adapted_native_v2_full"]
    chamfer = comparison["metrics"]["chamfer_l1_left_improvement"]
    ratios = {
        "chamfer_l1_mean": _safe_ratio(
            float(final["chamfer_l1"]["mean"]), float(full["chamfer_l1"]["mean"])
        ),
        "chamfer_l1_median": _safe_ratio(
            float(final["chamfer_l1"]["median"]),
            float(full["chamfer_l1"]["median"]),
        ),
        "fscore_0p02_mean": _safe_ratio(
            float(final["fscore_0p02"]["mean"]),
            float(full["fscore_0p02"]["mean"]),
        ),
        "normal_consistency_mean": _safe_ratio(
            float(final["normal_consistency"]["mean"]),
            float(full["normal_consistency"]["mean"]),
        ),
    }
    superiority = {
        "chamfer_l1_mean_better": float(chamfer["mean"]) > 0.0,
        "chamfer_l1_median_better": float(chamfer["median"]) > 0.0,
        "chamfer_l1_object_win_rate_at_least_half": (
            float(chamfer["positive_rate"]) >= 0.5
        ),
    }
    primary_non_regression = {
        "chamfer_l1_mean_ratio_at_most_1p05": ratios["chamfer_l1_mean"] <= 1.05,
        "chamfer_l1_median_ratio_at_most_1p05": (
            ratios["chamfer_l1_median"] <= 1.05
        ),
        "chamfer_l1_object_win_rate_at_least_0p40": (
            float(chamfer["positive_rate"]) >= 0.40
        ),
    }
    secondary = {
        "fscore_0p02_retention_at_least_0p95": ratios["fscore_0p02_mean"] >= 0.95,
        "normal_consistency_retention_at_least_0p98": (
            ratios["normal_consistency_mean"] >= 0.98
        ),
    }
    return {
        "primary_comparator": "real_adapted_native_v2_full",
        "priority": (
            "Chamfer-L1 mean/median/object-win first; F-score@0.02 and normal "
            "consistency retention are secondary"
        ),
        "frozen_non_regression_thresholds": {
            "max_chamfer_l1_mean_ratio": 1.05,
            "max_chamfer_l1_median_ratio": 1.05,
            "min_chamfer_l1_object_win_rate": 0.40,
            "min_fscore_0p02_mean_retention": 0.95,
            "min_normal_consistency_mean_retention": 0.98,
        },
        "metric_ratios_final_over_real_full": ratios,
        "superiority_checks": superiority,
        "superiority_passed": all(superiority.values()),
        "primary_non_regression_checks": primary_non_regression,
        "primary_non_regression_passed": all(primary_non_regression.values()),
        "secondary_retention_checks": secondary,
        "secondary_retention_passed": all(secondary.values()),
        "holdout_unlock_passed": all(primary_non_regression.values()),
    }


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label_manifest", required=True)
    parser.add_argument("--no_vggt_manifest", required=True)
    parser.add_argument("--real_full_manifest", required=True)
    parser.add_argument("--synthetic_full_manifest", required=True)
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
    quality_by_key = {
        key: row.get("alignment_quality_passed", True) is True
        for key, row in label_by_key.items()
    }
    warning_keys = sorted(key for key, passed in quality_by_key.items() if not passed)
    if warning_keys:
        if (
            labels.get("alignment_quality_warnings_included") is not True
            or int(labels.get("alignment_quality_warning_count", -1))
            != len(warning_keys)
            or sorted(labels.get("alignment_quality_warning_object_keys", []))
            != warning_keys
        ):
            raise RuntimeError("label quality warning contract is incomplete")
    runtime_hash = str(labels["runtime_input_manifest_sha256"])
    runtime_path = validate_bound_file(
        labels["runtime_input_manifest"], runtime_hash, label="runtime-O label input"
    )
    runtime = load_json(runtime_path)
    if runtime.get("format") != RUNTIME_MANIFEST_FORMAT or runtime.get("passed") is not True:
        raise RuntimeError("runtime-O label input is not a passed v2 manifest")

    method_paths = {
        "final_native_no_vggt": Path(args.no_vggt_manifest).resolve(),
        "real_adapted_native_v2_full": Path(args.real_full_manifest).resolve(),
        "synthetic_parent_native_v2_full": Path(args.synthetic_full_manifest).resolve(),
        "reconviagen_original": Path(args.reconviagen_manifest).resolve(),
        "pixal3d_official": Path(args.pixal3d_manifest).resolve(),
    }
    pair_rows: dict[str, dict[tuple[str, int], dict[str, Any]]] = {}
    runtime_bindings: dict[str, dict[str, Any]] = {}
    signature_cache: dict[str, dict[str, Any]] = {}
    for label, path in method_paths.items():
        spec = METHOD_SPECS[label]
        manifest = load_json(path)
        if (
            manifest.get("format") != spec["format"]
            or manifest.get("passed") is not True
            or manifest.get("target_or_metric_consumed") is not False
        ):
            raise RuntimeError(f"{label} inference contract did not pass: {path}")
        if label == "final_native_no_vggt":
            runtime_bindings[label] = _validate_no_vggt_runtime_binding(
                manifest,
                reference_runtime_path=runtime_path,
                reference_runtime_sha256=runtime_hash,
            )
        else:
            runtime_bindings[label] = validate_method_runtime_binding(
                str(spec["runtime_method"]),
                manifest,
                reference_runtime_path=runtime_path,
                reference_runtime=runtime,
                reference_runtime_sha256=runtime_hash,
                signature_cache=signature_cache,
            )
        pair_rows[label] = _records_by_pair(
            manifest, expected_method=str(spec["record_method"]), label=label
        )
    expected_pairs = set(pair_rows["final_native_no_vggt"])
    if any(set(rows) != expected_pairs for rows in pair_rows.values()):
        raise RuntimeError("five methods do not cover identical object/seed pairs")
    if {key for key, _ in expected_pairs} != set(label_by_key):
        raise RuntimeError("inference and runtime-O label object sets differ")
    if len({seed for _, seed in expected_pairs}) != 1:
        raise RuntimeError("final no-VGGT protocol requires exactly one fixed seed")

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
                    "alignment_quality_tier": (
                        "reliable" if quality_by_key[key] else "low_confidence"
                    ),
                    "surface_seed": surface_seed,
                    "mesh_success": bool(structure["mesh_success"]),
                    **{field: float(surface[field]) for field in SURFACE_FIELDS},
                    **{field: float(structure[field]) for field in STRUCTURE_FIELDS},
                }
            )
        print(
            f"[real_no_vggt_final] {position + 1}/{len(expected_pairs)} "
            f"object={key} seed={seed}",
            flush=True,
        )

    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        by_method[row["method"]].append(row)
    summaries = {method: _summary(by_method[method]) for method in METHOD_SPECS}
    comparisons = [
        _paired_delta(records, left="final_native_no_vggt", right=right)
        for right in METHOD_SPECS
        if right != "final_native_no_vggt"
    ]
    quality_subgroups = _quality_subgroup_report(
        records, quality_by_key=quality_by_key
    )
    decision = no_vggt_decision(summaries, comparisons[0])
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
        "label_quality_protocol": {
            "primary_population": "all_objects",
            "alignment_quality_all_passed": not warning_keys,
            "reliable_object_count": len(label_by_key) - len(warning_keys),
            "low_confidence_object_count": len(warning_keys),
            "low_confidence_object_keys": warning_keys,
            "low_confidence_objects_retained": bool(warning_keys),
            "selection_or_replacement_after_holdout_consumption": False,
            "transform_refit_for_low_confidence_objects": False,
            "interpretation": (
                "All-object metrics remain primary. Reliable-only and "
                "low-confidence subgroups are sensitivity disclosures; they do "
                "not delete, replace, or reweight the frozen holdout."
            ),
        },
        "summary": summaries,
        "label_quality_subgroups": quality_subgroups,
        "records": records,
        "paired_comparisons": comparisons,
        "no_vggt_decision": decision,
        "holdout64_consumed": args.protocol_scope == "formal_holdout64",
        "passed": protocol_passed,
    }
    report_path = output_dir / "report.json"
    atomic_json(report_path, report)
    lines = [
        "Omni real final no-VGGT direct runtime-O Mesh evaluation",
        "=" * 63,
        f"scope: {args.protocol_scope}",
        f"objects: {report['object_count']} pairs: {report['pair_count']}",
        (
            "label quality: "
            f"reliable={len(label_by_key) - len(warning_keys)} "
            f"low-confidence={len(warning_keys)}; All-object metrics are primary"
        ),
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
    lines.extend(
        [
            (
                "low-confidence objects: "
                + (", ".join(warning_keys) if warning_keys else "none")
            ),
            f"no-VGGT superiority passed: {decision['superiority_passed']}",
            f"primary non-regression passed: {decision['primary_non_regression_passed']}",
            f"secondary retention passed: {decision['secondary_retention_passed']}",
            f"holdout unlock passed: {decision['holdout_unlock_passed']}",
        ]
    )
    (output_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print({"passed": protocol_passed, "report": str(report_path)})
    if not protocol_passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
