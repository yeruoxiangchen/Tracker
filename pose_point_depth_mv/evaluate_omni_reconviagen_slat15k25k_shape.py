#!/usr/bin/env python3
"""Compare ReconViaGen and official-SS SLat-15k/25k on Omni Holdout64.

The report is deliberately shape-only and retrospective.  It contains two
tracks:

1. ``normalized_fixed_contract`` applies the released ReconViaGen decoder-axis
   rotation and independently centers/scales every Mesh by its own AABB.
2. ``proper_sim3_shape_only`` additionally fits each prediction to the same GT
   with a proper isotropic Sim(3).  Reflection and anisotropic scale are
   forbidden, so this track cannot support pose, metric-scale, or AR-placement
   claims.

Both official candidates must consume the exact Point+Mask -> runtime-O model
input manifest and must share the same official Native-SS deployment.  Only
their frozen SLat checkpoints (step 15000 versus 25000) may differ.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from pose_point_depth_mv.compare_pixal3d_singleview_smoke import (
    load_mesh,
    similarity_icp,
    surface_metrics,
)
from pose_point_depth_mv.dataset_tools.prepare_omni_real_label_cache import (
    MANIFEST_FORMAT as LABEL_MANIFEST_FORMAT,
)
from pose_point_depth_mv.evaluate_holdout64_current_vs_reconviagen_shape import (
    RECONVIAGEN_DECODER_TO_REFERENCE,
    normalize_mesh_bbox,
)
from pose_point_depth_mv.evaluate_stock2_full2_reconviagen_sim3_shape import (
    audit_proper_sim3,
)
from pose_point_depth_mv.omni_real_benchmark_common import (
    atomic_json,
    canonical_sha256,
    index_objects,
    load_json,
    sha256_file,
    validate_bound_file,
)


REPORT_FORMAT = "pose_point_depth_mv.omni_recon_slat15k25k_shape.v1"
CONFIG_FORMAT = "pose_point_depth_mv.omni_recon_slat15k25k_shape_config.v1"
RECORD_FORMAT = "pose_point_depth_mv.omni_recon_slat15k25k_shape_record.v1"
RECON_MANIFEST_FORMAT = (
    "pose_point_depth_mv.omni_real_reconviagen_inference_manifest.v1"
)
OFFICIAL_MANIFEST_FORMAT = (
    "pose_point_depth_mv.real_proobjaverse_official_ss_slat_inference_manifest.v1"
)
RUNTIME_MANIFEST_FORMAT = "pose_point_depth_mv.omni_real_runtime_input_manifest.v2"
DINO_INPUT_MANIFEST_FORMAT = (
    "pose_point_depth_mv.omni_real_dino_only_model_input_manifest.v1"
)
RECON_RECORD_METHOD = "reconviagen_original"
OFFICIAL_RECORD_METHOD = "proobjaverse_official_native_ss_trained_slat"
METHODS = (
    "reconviagen_original",
    "official_ss_slat15000",
    "official_ss_slat25000",
)
COMPARISONS = (
    (
        "slat15000_vs_reconviagen",
        "official_ss_slat15000",
        "reconviagen_original",
    ),
    (
        "slat25000_vs_reconviagen",
        "official_ss_slat25000",
        "reconviagen_original",
    ),
    (
        "slat25000_vs_slat15000",
        "official_ss_slat25000",
        "official_ss_slat15000",
    ),
)
TRACKS = ("normalized_fixed_contract", "proper_sim3_shape_only")
METRICS = (
    "pred_to_gt_mean",
    "gt_to_pred_mean",
    "chamfer_l1",
    "chamfer_l2",
    "fscore_0p01",
    "fscore_0p02",
    "fscore_0p05",
    "normal_consistency",
)
LOWER_IS_BETTER = {
    "pred_to_gt_mean",
    "gt_to_pred_mean",
    "chamfer_l1",
    "chamfer_l2",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def file_binding(path: str | Path, *, expected_sha256: str = "") -> dict[str, str]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    digest = sha256_file(resolved)
    if expected_sha256 and digest != str(expected_sha256):
        raise RuntimeError(f"file SHA256 changed: {resolved}")
    return {"path": str(resolved), "sha256": digest}


def _validate_recon_manifest(
    manifest: dict[str, Any], *, expected_objects: int, seed: int
) -> dict[str, dict[str, Any]]:
    expected = {
        "format": RECON_MANIFEST_FORMAT,
        "method": RECON_RECORD_METHOD,
        "seeds": [int(seed)],
        "object_count": int(expected_objects),
        "record_count": int(expected_objects),
        "passed": True,
    }
    mismatch = {
        key: (manifest.get(key), value)
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if mismatch:
        raise RuntimeError(f"ReconViaGen manifest contract differs: {mismatch}")
    rows: dict[str, dict[str, Any]] = {}
    for row in manifest.get("objects", []):
        key = str(row.get("object_key", ""))
        if (
            not key
            or key in rows
            or row.get("method") != RECON_RECORD_METHOD
            or int(row.get("seed", -1)) != int(seed)
            or row.get("passed") is not True
            or row.get("explicit_runtime_pose_condition") is not False
        ):
            raise RuntimeError(f"invalid ReconViaGen record: {key!r}")
        rows[key] = row
    if len(rows) != int(expected_objects):
        raise RuntimeError("ReconViaGen object coverage differs")
    return rows


def _validate_official_manifest(
    manifest: dict[str, Any],
    *,
    expected_objects: int,
    seed: int,
    expected_step: int,
) -> dict[str, dict[str, Any]]:
    expected = {
        "format": OFFICIAL_MANIFEST_FORMAT,
        "method": OFFICIAL_RECORD_METHOD,
        "seeds": [int(seed)],
        "object_count": int(expected_objects),
        "record_count": int(expected_objects),
        "output_frame": "runtime-O",
        "vggt_model_loaded": False,
        "vggt_model_executed": False,
        "passed": True,
    }
    mismatch = {
        key: (manifest.get(key), value)
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if mismatch:
        raise RuntimeError(
            f"official SLat step{expected_step} manifest contract differs: {mismatch}"
        )
    deployment = dict(manifest.get("native_slat_deployment") or {})
    if (
        int(deployment.get("checkpoint_step", -1)) != int(expected_step)
        or deployment.get("weights") != "ema"
    ):
        raise RuntimeError(f"official SLat step{expected_step} identity differs")
    file_binding(
        deployment.get("checkpoint", ""),
        expected_sha256=str(deployment.get("checkpoint_sha256", "")),
    )
    ss = dict(manifest.get("native_ss_deployment") or {})
    expected_ss = {
        "checkpoint_step": 2000,
        "weights": "ema",
        "cfg_strength": 5.0,
        "steps": 25,
        "cfg_interval": [0.5, 1.0],
        "guidance_rescale": 0.0,
        "rescale_t": 3.0,
        "amp_dtype": "bf16",
        "false_checks": [],
    }
    ss_mismatch = {
        key: (ss.get(key), value)
        for key, value in expected_ss.items()
        if ss.get(key) != value
    }
    if ss_mismatch:
        raise RuntimeError(f"official Native-SS deployment differs: {ss_mismatch}")
    file_binding(
        ss.get("report", ""), expected_sha256=str(ss.get("report_sha256", ""))
    )
    file_binding(
        ss.get("checkpoint", ""),
        expected_sha256=str(ss.get("checkpoint_sha256", "")),
    )
    bridge = dict(manifest.get("cross_deployment_bridge") or {})
    if bridge.get("passed") is not True:
        raise RuntimeError("official SLat cross-deployment artifact binding did not pass")
    file_binding(bridge.get("path", ""), expected_sha256=str(bridge.get("sha256", "")))
    stock = file_binding(
        manifest.get("stock_slat_freeze", ""),
        expected_sha256=str(manifest.get("stock_slat_freeze_sha256", "")),
    )
    if not stock["sha256"]:
        raise RuntimeError("official SLat stock freeze binding is empty")

    rows: dict[str, dict[str, Any]] = {}
    for row in manifest.get("objects", []):
        key = str(row.get("object_key", ""))
        if (
            not key
            or key in rows
            or row.get("method") != OFFICIAL_RECORD_METHOD
            or int(row.get("seed", -1)) != int(seed)
            or int(row.get("native_slat_checkpoint_step", -1)) != int(expected_step)
            or row.get("native_slat_weights") != "ema"
            or row.get("output_frame") != "runtime-O"
            or row.get("vggt_model_loaded") is not False
            or row.get("vggt_model_executed") is not False
            or row.get("target_or_metric_consumed") is not False
            or row.get("formal_claim_allowed") is not False
            or row.get("passed") is not True
        ):
            raise RuntimeError(
                f"invalid official SLat step{expected_step} record: {key!r}"
            )
        rows[key] = row
    if len(rows) != int(expected_objects):
        raise RuntimeError(f"official SLat step{expected_step} coverage differs")
    return rows


def _runtime_binding(
    manifest: dict[str, Any], *, expected_sha256: str, label: str
) -> dict[str, str]:
    declared = str(manifest.get("runtime_input_manifest_sha256", ""))
    path = validate_bound_file(
        manifest.get("runtime_input_manifest", ""), declared, label=f"{label} runtime"
    )
    if declared != str(expected_sha256):
        raise RuntimeError(f"{label} does not consume the frozen GT runtime-O input")
    return {"path": str(path), "sha256": declared}


def _official_artifact_identity(manifest: dict[str, Any]) -> dict[str, Any]:
    ss = dict(manifest.get("native_ss_deployment") or {})
    return {
        "model_input_manifest_sha256": str(
            manifest.get("model_input_manifest_sha256", "")
        ),
        "native_ss_report_sha256": str(ss.get("report_sha256", "")),
        "native_ss_checkpoint_sha256": str(ss.get("checkpoint_sha256", "")),
        "native_ss_checkpoint_step": int(ss.get("checkpoint_step", -1)),
        "native_ss_weights": str(ss.get("weights", "")),
        "native_ss_cfg_strength": float(ss.get("cfg_strength", float("nan"))),
        "native_ss_steps": int(ss.get("steps", -1)),
        "native_ss_cfg_interval": list(ss.get("cfg_interval") or []),
        "native_ss_guidance_rescale": float(
            ss.get("guidance_rescale", float("nan"))
        ),
        "native_ss_rescale_t": float(ss.get("rescale_t", float("nan"))),
        "native_ss_amp_dtype": str(ss.get("amp_dtype", "")),
        "stock_slat_freeze_sha256": str(
            manifest.get("stock_slat_freeze_sha256", "")
        ),
    }


def numeric_summary(
    values: Iterable[float], *, bootstrap_samples: int, seed: int
) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=np.float64)
    if not len(array) or not np.isfinite(array).all():
        raise ValueError("summary requires finite values")
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, len(array), size=(int(bootstrap_samples), len(array)))
    means = array[indices].mean(axis=1)
    return {
        "count": int(len(array)),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "min": float(array.min()),
        "max": float(array.max()),
        "positive_rate": float(np.mean(array > 0.0)),
        "nonnegative_rate": float(np.mean(array >= 0.0)),
        "bootstrap_mean_95_ci": [
            float(np.quantile(means, 0.025)),
            float(np.quantile(means, 0.975)),
        ],
    }


def summarize_track(
    records: list[dict[str, Any]],
    *,
    track: str,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    methods: dict[str, Any] = {}
    for method_position, method in enumerate(METHODS):
        methods[method] = {
            metric: numeric_summary(
                [float(row["methods"][method][track][metric]) for row in records],
                bootstrap_samples=int(bootstrap_samples),
                seed=int(seed) + method_position * 1000 + metric_position,
            )
            for metric_position, metric in enumerate(METRICS)
        }

    comparisons: dict[str, Any] = {}
    for comparison_position, (name, candidate, baseline) in enumerate(COMPARISONS):
        metric_rows: dict[str, Any] = {}
        for metric_position, metric in enumerate(METRICS):
            values = []
            per_object = {}
            for row in records:
                candidate_value = float(row["methods"][candidate][track][metric])
                baseline_value = float(row["methods"][baseline][track][metric])
                delta = (
                    baseline_value - candidate_value
                    if metric in LOWER_IS_BETTER
                    else candidate_value - baseline_value
                )
                values.append(delta)
                per_object[str(row["object_key"])] = float(delta)
            summary = numeric_summary(
                values,
                bootstrap_samples=int(bootstrap_samples),
                seed=int(seed)
                + 10000
                + comparison_position * 1000
                + metric_position,
            )
            summary.update(
                {
                    "candidate_win_count": int(sum(value > 0.0 for value in values)),
                    "tie_count": int(sum(value == 0.0 for value in values)),
                    "baseline_win_count": int(sum(value < 0.0 for value in values)),
                    "per_object": per_object,
                }
            )
            metric_rows[metric] = summary
        comparisons[name] = {
            "candidate": candidate,
            "baseline": baseline,
            "positive_definition": "positive means candidate is better",
            "metrics": metric_rows,
        }
    return {"methods": methods, "comparisons": comparisons}


def summary_text(report: dict[str, Any]) -> str:
    lines = [
        "Omni Holdout64: ReconViaGen vs official SS + SLat15k/25k",
        "=========================================================",
        f"formal: false  post_hoc: true  objects: {report['object_count']}",
        "positive paired values mean the named candidate is better",
    ]
    for track in report["tracks"]:
        lines.extend(("", f"[{track}]"))
        summary = report["summary"]["tracks"][track]
        for method in METHODS:
            metrics = summary["methods"][method]
            lines.append(
                f"{method}: L1={metrics['chamfer_l1']['mean']:.8f} "
                f"F02={metrics['fscore_0p02']['mean']:.8f} "
                f"Normal={metrics['normal_consistency']['mean']:.8f}"
            )
        for name, comparison in summary["comparisons"].items():
            metrics = comparison["metrics"]
            lines.append(
                f"{name}: L1={metrics['chamfer_l1']['mean']:+.8f} "
                f"win={metrics['chamfer_l1']['positive_rate']:.4f} "
                f"CI={metrics['chamfer_l1']['bootstrap_mean_95_ci']} "
                f"F02={metrics['fscore_0p02']['mean']:+.8f} "
                f"Normal={metrics['normal_consistency']['mean']:+.8f}"
            )
    if "proper_sim3_shape_only" in report["tracks"]:
        interpretation = (
            "Primary interpretation: proper_sim3_shape_only removes global proper",
            "rotation, translation, and isotropic scale. It cannot support world-pose,",
            "metric-scale, or AR-placement claims. Holdout64 is already consumed, so",
            "this comparison remains retrospective and formal=false.",
        )
    else:
        interpretation = (
            "Fast diagnostic only: proper Sim(3) was skipped; this report contains",
            "independent AABB normalization and the fixed decoder-axis contract only.",
            "Holdout64 is already consumed, so it remains retrospective/formal=false.",
        )
    lines.extend(("", *interpretation))
    return "\n".join(lines) + "\n"


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label_manifest", required=True)
    parser.add_argument("--reconviagen_manifest", required=True)
    parser.add_argument("--slat15000_manifest", required=True)
    parser.add_argument("--slat25000_manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--expected_objects", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--candidate_samples", type=int, default=2000)
    parser.add_argument("--alignment_samples", type=int, default=10000)
    parser.add_argument("--candidate_iterations", type=int, default=12)
    parser.add_argument("--final_iterations", type=int, default=50)
    parser.add_argument("--surface_samples", type=int, default=20000)
    parser.add_argument("--bootstrap_samples", type=int, default=10000)
    parser.add_argument("--metric_seed", type=int, default=20260816)
    parser.add_argument("--skip_proper_sim3", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> None:
    args = make_parser().parse_args()
    positive = (
        args.expected_objects,
        args.candidate_samples,
        args.alignment_samples,
        args.candidate_iterations,
        args.final_iterations,
        args.surface_samples,
        args.bootstrap_samples,
    )
    if any(int(value) <= 0 for value in positive):
        raise ValueError("all count and iteration arguments must be positive")

    label_path = Path(args.label_manifest).expanduser().resolve()
    recon_path = Path(args.reconviagen_manifest).expanduser().resolve()
    slat15_path = Path(args.slat15000_manifest).expanduser().resolve()
    slat25_path = Path(args.slat25000_manifest).expanduser().resolve()
    labels = load_json(label_path)
    recon_manifest = load_json(recon_path)
    slat15_manifest = load_json(slat15_path)
    slat25_manifest = load_json(slat25_path)
    if labels.get("format") != LABEL_MANIFEST_FORMAT or labels.get("passed") is not True:
        raise RuntimeError("Omni runtime-O GT label manifest did not pass")
    label_rows = index_objects(labels.get("objects", []), label="Omni labels")
    if (
        len(label_rows) != int(args.expected_objects)
        or int(labels.get("selected_object_count", -1)) != int(args.expected_objects)
        or int(labels.get("completed_object_count", -1)) != int(args.expected_objects)
    ):
        raise RuntimeError("Omni GT label coverage differs")
    recon_rows = _validate_recon_manifest(
        recon_manifest, expected_objects=int(args.expected_objects), seed=int(args.seed)
    )
    slat15_rows = _validate_official_manifest(
        slat15_manifest,
        expected_objects=int(args.expected_objects),
        seed=int(args.seed),
        expected_step=15000,
    )
    slat25_rows = _validate_official_manifest(
        slat25_manifest,
        expected_objects=int(args.expected_objects),
        seed=int(args.seed),
        expected_step=25000,
    )
    if not (
        set(label_rows) == set(recon_rows) == set(slat15_rows) == set(slat25_rows)
    ):
        raise RuntimeError("GT/ReconViaGen/SLat15k/SLat25k object sets differ")

    reference_runtime_sha256 = str(labels.get("runtime_input_manifest_sha256", ""))
    reference_runtime = validate_bound_file(
        labels.get("runtime_input_manifest", ""),
        reference_runtime_sha256,
        label="GT reference runtime-O",
    )
    runtime_payload = load_json(reference_runtime)
    runtime_rows = index_objects(
        runtime_payload.get("objects", []), label="Point+Mask runtime-O"
    )
    build_config = dict(runtime_payload.get("build_config") or {})
    frame_config = dict(build_config.get("frame_config") or {})
    minimum_points = int(frame_config.get("min_object_points", -1))
    if (
        runtime_payload.get("format") != RUNTIME_MANIFEST_FORMAT
        or runtime_payload.get("passed") is not True
        or build_config.get("input_frontend_format")
        != "pose_point_depth_mv.real_input_frontend.v2"
        or minimum_points <= 0
        or len(runtime_rows) != int(args.expected_objects)
        or set(runtime_rows) != set(label_rows)
    ):
        raise RuntimeError("Point+Mask -> runtime-O input contract differs")
    for key, row in runtime_rows.items():
        support = dict(row.get("runtime_frame_stats", {}).get("support") or {})
        if int(support.get("mask_supported_point_count", -1)) < minimum_points:
            raise RuntimeError(f"Point+Mask support is insufficient: {key}")
    runtime_bindings = {
        "gt": {"path": str(reference_runtime), "sha256": reference_runtime_sha256},
        "reconviagen_original": _runtime_binding(
            recon_manifest,
            expected_sha256=reference_runtime_sha256,
            label="ReconViaGen",
        ),
        "official_ss_slat15000": _runtime_binding(
            slat15_manifest,
            expected_sha256=reference_runtime_sha256,
            label="official SLat15k",
        ),
        "official_ss_slat25000": _runtime_binding(
            slat25_manifest,
            expected_sha256=reference_runtime_sha256,
            label="official SLat25k",
        ),
    }
    identity15 = _official_artifact_identity(slat15_manifest)
    identity25 = _official_artifact_identity(slat25_manifest)
    if identity15 != identity25:
        raise RuntimeError(
            "SLat15k/25k differ outside the SLat checkpoint: "
            f"15k={identity15}, 25k={identity25}"
        )
    model_input = file_binding(
        slat15_manifest.get("model_input_manifest", ""),
        expected_sha256=identity15["model_input_manifest_sha256"],
    )
    if str(slat25_manifest.get("model_input_manifest")) != model_input["path"]:
        raise RuntimeError("SLat15k/25k do not consume the same DINO-only inputs")
    model_input_payload = load_json(model_input["path"])
    dino_rows = index_objects(
        model_input_payload.get("objects", []), label="DINO-only model inputs"
    )
    if (
        model_input_payload.get("format") != DINO_INPUT_MANIFEST_FORMAT
        or model_input_payload.get("passed") is not True
        or model_input_payload.get("vggt_model_loaded") is not False
        or model_input_payload.get("vggt_model_executed") is not False
        or str(model_input_payload.get("runtime_input_manifest_sha256", ""))
        != reference_runtime_sha256
        or len(dino_rows) != int(args.expected_objects)
        or set(dino_rows) != set(label_rows)
        or any(row.get("target_or_mesh_consumed") is not False for row in dino_rows.values())
    ):
        raise RuntimeError("DINO-only Point+Mask model-input contract differs")

    tracks = ["normalized_fixed_contract"]
    if not args.skip_proper_sim3:
        tracks.append("proper_sim3_shape_only")
    thresholds = (0.01, 0.02, 0.05)
    config = {
        "format": CONFIG_FORMAT,
        "formal": False,
        "post_hoc": True,
        "holdout64_consumed": True,
        "protocol_scope": "post_hoc_consumed_omni_holdout64_shape_only",
        "label_manifest": file_binding(label_path),
        "reconviagen_manifest": file_binding(recon_path),
        "slat15000_manifest": file_binding(slat15_path),
        "slat25000_manifest": file_binding(slat25_path),
        "runtime_bindings": runtime_bindings,
        "shared_official_identity": identity15,
        "model_input_manifest": model_input,
        "point_mask_to_runtime_o_required": True,
        "object_count": int(args.expected_objects),
        "seed": int(args.seed),
        "methods": list(METHODS),
        "tracks": tracks,
        "normalized_fixed_contract": {
            "all_meshes_center": "own axis-aligned bounding-box midpoint",
            "all_meshes_scale": "divide by own longest AABB extent",
            "reconviagen_fixed_decoder_axis": "(x,y,z) -> (x,z,-y)",
            "gt_fit_or_optimization": False,
        },
        "proper_sim3_shape_only": {
            "enabled": not bool(args.skip_proper_sim3),
            "policy": "GT-assisted proper isotropic Sim(3), independently per method",
            "initializations": "24 proper cube rotations followed by similarity ICP",
            "reflection": False,
            "anisotropic_scale": False,
            "candidate_samples": int(args.candidate_samples),
            "alignment_samples": int(args.alignment_samples),
            "candidate_iterations": int(args.candidate_iterations),
            "final_iterations": int(args.final_iterations),
        },
        "evaluation": {
            "surface_samples": int(args.surface_samples),
            "thresholds_in_unit_longest_extent": list(thresholds),
            "metric_seed": int(args.metric_seed),
            "bootstrap_samples": int(args.bootstrap_samples),
            "paired_resampling_unit": "object",
        },
        "interpretation_scope": (
            "geometry shape only; no world pose, metric scale, or AR placement claim"
        ),
        "implementation": file_binding(Path(__file__).resolve()),
    }
    config_sha256 = canonical_sha256(config)
    config["config_sha256"] = config_sha256

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "run_config.json"
    if config_path.is_file():
        if load_json(config_path) != config:
            raise RuntimeError(f"existing shape config differs: {config_path}")
    elif any(output_dir.iterdir()):
        raise RuntimeError(f"unbound output directory is not empty: {output_dir}")
    else:
        atomic_json(config_path, config)

    report_path = output_dir / "report.json"
    if report_path.is_file():
        report = load_json(report_path)
        if (
            report.get("format") != REPORT_FORMAT
            or report.get("config_sha256") != config_sha256
            or report.get("passed") is not True
        ):
            raise RuntimeError(f"existing report contract differs: {report_path}")
        print(summary_text(report), end="")
        print(json.dumps({"passed": True, "reused": True, "report": str(report_path)}))
        return

    records: list[dict[str, Any]] = []
    for position, key in enumerate(sorted(label_rows)):
        label_row = label_rows[key]
        bindings = {
            "reconviagen_original": file_binding(
                recon_rows[key]["mesh"],
                expected_sha256=str(recon_rows[key]["mesh_sha256"]),
            ),
            "official_ss_slat15000": file_binding(
                slat15_rows[key]["mesh"],
                expected_sha256=str(slat15_rows[key]["mesh_sha256"]),
            ),
            "official_ss_slat25000": file_binding(
                slat25_rows[key]["mesh"],
                expected_sha256=str(slat25_rows[key]["mesh_sha256"]),
            ),
        }
        target_binding = file_binding(
            label_row["mesh_o"], expected_sha256=str(label_row["mesh_o_sha256"])
        )
        identity = {
            "format": RECORD_FORMAT,
            "config_sha256": config_sha256,
            "object_key": key,
            "target_mesh_sha256": target_binding["sha256"],
            "method_mesh_sha256": {
                method: bindings[method]["sha256"] for method in METHODS
            },
        }
        record_path = (
            output_dir
            / "records"
            / str(label_row["category"])
            / f"{label_row['object_id']}.json"
        )
        if record_path.is_file():
            if not args.resume:
                raise RuntimeError(f"record exists; pass --resume: {record_path}")
            record = load_json(record_path)
            mismatch = {
                field: (record.get(field), value)
                for field, value in identity.items()
                if record.get(field) != value
            }
            if mismatch or record.get("passed") is not True:
                raise RuntimeError(f"stale shape record={mismatch}: {record_path}")
            records.append(record)
            print(
                f"[omni_shape:reuse] {position + 1}/{len(label_rows)} object={key}",
                flush=True,
            )
            continue

        target_raw = load_mesh(target_binding["path"])
        target, target_normalization = normalize_mesh_bbox(target_raw)
        alignment_seed = int(args.metric_seed) + position * 100003
        metric_seed = int(args.metric_seed) + 50_000_000 + position * 100003
        method_rows: dict[str, Any] = {}
        for method in METHODS:
            mesh = load_mesh(bindings[method]["path"])
            if method == "reconviagen_original":
                mesh.apply_transform(RECONVIAGEN_DECODER_TO_REFERENCE)
            normalized, normalization = normalize_mesh_bbox(mesh)
            normalized_metrics = surface_metrics(
                normalized,
                target,
                count=int(args.surface_samples),
                seed=metric_seed,
                thresholds=thresholds,
            )
            method_row = {
                "mesh": bindings[method],
                "normalization": normalization,
                "normalized_fixed_contract": {
                    metric: float(normalized_metrics[metric]) for metric in METRICS
                },
            }
            if not args.skip_proper_sim3:
                aligned, alignment = similarity_icp(
                    normalized,
                    target,
                    seed=alignment_seed,
                    candidate_samples=int(args.candidate_samples),
                    final_samples=int(args.alignment_samples),
                    candidate_iterations=int(args.candidate_iterations),
                    final_iterations=int(args.final_iterations),
                )
                method_row["alignment"] = audit_proper_sim3(alignment)
                shape_metrics = surface_metrics(
                    aligned,
                    target,
                    count=int(args.surface_samples),
                    seed=metric_seed,
                    thresholds=thresholds,
                )
                method_row["proper_sim3_shape_only"] = {
                    metric: float(shape_metrics[metric]) for metric in METRICS
                }
                del aligned
            method_rows[method] = method_row
            del mesh, normalized

        record = {
            **identity,
            "created_at_utc": utc_now(),
            "category": str(label_row["category"]),
            "object_id": str(label_row["object_id"]),
            "target_mesh": target_binding,
            "target_normalization": target_normalization,
            "alignment_seed": alignment_seed,
            "metric_seed": metric_seed,
            "methods": method_rows,
            "passed": True,
        }
        atomic_json(record_path, record)
        records.append(record)
        print(
            f"[omni_shape] {position + 1}/{len(label_rows)} object={key}",
            flush=True,
        )
        del target_raw, target

    track_summaries = {
        track: summarize_track(
            records,
            track=track,
            bootstrap_samples=int(args.bootstrap_samples),
            seed=int(args.metric_seed) + track_position * 100000,
        )
        for track_position, track in enumerate(tracks)
    }
    alignments: dict[str, Any] = {}
    if not args.skip_proper_sim3:
        for method_position, method in enumerate(METHODS):
            alignments[method] = {
                field: numeric_summary(
                    [float(row["methods"][method]["alignment"][field]) for row in records],
                    bootstrap_samples=int(args.bootstrap_samples),
                    seed=int(args.metric_seed) + 900000 + method_position * 100 + field_position,
                )
                for field_position, field in enumerate(
                    ("isotropic_scale", "rotation_angle_deg", "translation_norm", "cost")
                )
            }

    report = {
        "format": REPORT_FORMAT,
        "created_at_utc": utc_now(),
        "passed": len(records) == int(args.expected_objects),
        "formal": False,
        "post_hoc": True,
        "holdout64_consumed": True,
        "protocol_scope": "post_hoc_consumed_omni_holdout64_shape_only",
        "interpretation_scope": config["interpretation_scope"],
        "config": str(config_path),
        "config_sha256": config_sha256,
        "object_count": len(records),
        "record_count": len(records) * len(METHODS),
        "tracks": tracks,
        "summary": {"tracks": track_summaries, "alignments": alignments},
        "records": records,
    }
    atomic_json(report_path, report)
    text = summary_text(report)
    atomic_text(output_dir / "summary.txt", text)
    print(text, end="")
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "formal": report["formal"],
                "objects": report["object_count"],
                "records": report["record_count"],
                "report": str(report_path),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
