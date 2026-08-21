#!/usr/bin/env python3
"""Unified canonical geometry scoring for Native/Stock/Pixal3D/GenRecon.

The protocol may be the four-case smoke or the exact T3 final32 matched set.
Metrics are recomputed here with one sampler and one coordinate policy;
historical reports from different evaluators are never mixed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from pose_point_depth_mv.compare_pixal3d_singleview_smoke import (
    atomic_json,
    atomic_text,
    binding,
    canonical_sha256,
    load_mesh,
    pixal3d_mesh_path,
    pixal3d_result_path,
    sha256_file,
    surface_metrics,
    validate_official_inference_result,
)
from pose_point_depth_mv.evaluate_direct_slat_pixal3d_utility import (
    case_canonical_transform,
    numeric_summary,
    validate_transform,
)
from pose_point_depth_mv.export_direct_flow_mesh_pairs import mesh_structure_metrics
from pose_point_depth_mv.infer_official_genrecon_objects import (
    CAMERA_CONVERSION,
    FORMAT as GENRECON_RESULT_FORMAT,
)
from pose_point_depth_mv.prepare_native_ss_matched_baselines import (
    SELECTION_MODE as MATCHED_SELECTION_MODE,
    validate_matched_protocol,
)
from pose_point_depth_mv.prepare_native_ss_pixal3d_review import validate_native_protocol


FORMAT = "pose_point_depth_mv.native_ss_pixal_genrecon_geometry.v1"
METHODS = ("native_full", "stock", "pixal3d_official", "genrecon_official")
METRICS = (
    "chamfer_l1",
    "pred_to_gt_mean",
    "gt_to_pred_mean",
    "fscore_0p01",
    "fscore_0p02",
    "fscore_0p05",
    "normal_consistency",
    "precision_0p02",
    "recall_0p02",
    "largest_component_ratio",
)
LOWER_IS_BETTER = {"chamfer_l1", "pred_to_gt_mean", "gt_to_pred_mean"}


def validate_benchmark_protocol(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    mode = raw.get("comparison", {}).get("selection_mode")
    if mode == MATCHED_SELECTION_MODE:
        return validate_matched_protocol(path)
    return validate_native_protocol(path)


def _load_genrecon_result(
    root: Path,
    protocol: dict[str, Any],
    case: dict[str, Any],
) -> tuple[Path, dict[str, Any]]:
    case_root = root / str(case["case_id"])
    mesh_path = case_root / "mesh_canonical.obj"
    result_path = case_root / "result.json"
    if not mesh_path.is_file() or not result_path.is_file():
        raise FileNotFoundError(
            f"missing official GenRecon output for case={case['case_id']}: {case_root}"
        )
    result = json.loads(result_path.read_text(encoding="utf-8"))
    expected_hash = str(result.get("result_sha256", ""))
    body = dict(result)
    body.pop("result_sha256", None)
    if not expected_hash or canonical_sha256(body) != expected_hash:
        raise RuntimeError(f"GenRecon result SHA mismatch: {result_path}")
    if result.get("format") != GENRECON_RESULT_FORMAT:
        raise RuntimeError(f"unexpected GenRecon result format: {result_path}")
    if result.get("complete") is not True:
        raise RuntimeError(f"incomplete GenRecon result: {result_path}")
    if result.get("protocol_sha256") != protocol["protocol_sha256"]:
        raise RuntimeError(f"GenRecon protocol mismatch: {result_path}")
    if result.get("case_id") != case["case_id"]:
        raise RuntimeError(f"GenRecon case mismatch: {result_path}")
    if result.get("view_count") != case["view_count"]:
        raise RuntimeError(f"GenRecon view-count mismatch: {result_path}")
    if result.get("camera_conversion") != CAMERA_CONVERSION:
        raise RuntimeError(f"GenRecon camera conversion changed: {result_path}")
    if result.get("mesh", {}).get("sha256") != sha256_file(mesh_path):
        raise RuntimeError(f"GenRecon mesh SHA mismatch: {mesh_path}")
    return mesh_path, result


def _method_meshes(
    *,
    protocol_path: Path,
    protocol: dict[str, Any],
    case: dict[str, Any],
    pixal_transform: dict[str, Any],
    genrecon_root: Path,
) -> tuple[Any, dict[str, Any], dict[str, dict[str, Any]], dict[str, Any]]:
    target_path = Path(case["target_mesh"]["path"]).resolve()
    target = load_mesh(target_path)
    paths = {
        "native_full": Path(case["native_full_mesh"]["path"]).resolve(),
        "stock": Path(case["stock_mesh"]["path"]).resolve(),
        "pixal3d_official": pixal3d_mesh_path(protocol_path, case["case_id"]),
    }
    pixal_result_path = pixal3d_result_path(protocol_path, case["case_id"])
    if not paths["pixal3d_official"].is_file() or not pixal_result_path.is_file():
        raise FileNotFoundError(f"missing official Pixal3D output: {case['case_id']}")
    pixal_result = json.loads(pixal_result_path.read_text(encoding="utf-8"))
    validate_official_inference_result(
        pixal_result,
        protocol=protocol,
        case=case,
        mesh_path=paths["pixal3d_official"],
    )
    genrecon_path, genrecon_result = _load_genrecon_result(
        genrecon_root, protocol, case
    )
    paths["genrecon_official"] = genrecon_path

    transforms = {
        # T3 exported transform_pose=False and scored these meshes directly in
        # the canonical latent frame.  The axis transform used by the human
        # turntable renderer is display-only and is forbidden here.
        "native_full": np.eye(4, dtype=np.float64),
        "stock": np.eye(4, dtype=np.float64),
        "pixal3d_official": case_canonical_transform(case, pixal_transform),
        "genrecon_official": np.eye(4, dtype=np.float64),
    }
    meshes = {}
    method_bindings = {}
    for method in METHODS:
        mesh = load_mesh(paths[method])
        mesh.apply_transform(transforms[method])
        meshes[method] = mesh
        method_bindings[method] = {
            "mesh": binding(paths[method]),
            "source_to_reference": transforms[method].tolist(),
        }
    method_bindings["pixal3d_official"]["result"] = binding(pixal_result_path)
    method_bindings["genrecon_official"]["result"] = binding(
        genrecon_root / case["case_id"] / "result.json"
    )
    runtime = {
        "pixal3d": pixal_result,
        "genrecon": genrecon_result,
    }
    return target, meshes, method_bindings, runtime


def _summaries(
    records: list[dict[str, Any]],
    *,
    bootstrap_samples: int,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    absolute: dict[str, Any] = {}
    for method_index, method in enumerate(METHODS):
        absolute[method] = {}
        for metric_index, metric in enumerate(METRICS):
            values = [float(row["methods"][method][metric]) for row in records]
            absolute[method][metric] = numeric_summary(
                values,
                bootstrap_samples=bootstrap_samples,
                seed=seed + 1000 * method_index + metric_index,
            )

    comparisons: dict[str, Any] = {}
    for baseline_index, baseline in enumerate(METHODS[1:]):
        key = f"native_full_minus_{baseline}"
        comparisons[key] = {}
        for metric_index, metric in enumerate(METRICS):
            if metric in LOWER_IS_BETTER:
                values = [
                    float(row["methods"][baseline][metric])
                    - float(row["methods"]["native_full"][metric])
                    for row in records
                ]
            else:
                values = [
                    float(row["methods"]["native_full"][metric])
                    - float(row["methods"][baseline][metric])
                    for row in records
                ]
            comparisons[key][metric] = numeric_summary(
                values,
                bootstrap_samples=bootstrap_samples,
                seed=seed + 10000 + baseline_index * 1000 + metric_index,
            )
        comparisons[key]["positive_means_native_better"] = True
    return absolute, comparisons


def _summary_text(report: dict[str, Any]) -> str:
    lines = [
        "Native Full / Stock / official Pixal3D / official GenRecon geometry benchmark",
        "============================================================================",
        f"formal: {str(report['formal']).lower()}",
        f"objects: {report['object_count']}",
        f"surface samples per mesh: {report['surface_samples']}",
        "coordinate policy: canonical pose; no per-method ICP/scale/reflection",
        "",
        "Absolute means:",
    ]
    for method in METHODS:
        metrics = report["summary"]["methods"][method]
        lines.append(
            f"{method}: chamfer={metrics['chamfer_l1']['mean']:.8f} "
            f"f@0.02={metrics['fscore_0p02']['mean']:.8f} "
            f"normal={metrics['normal_consistency']['mean']:.8f} "
            f"lcr={metrics['largest_component_ratio']['mean']:.8f}"
        )
    lines.extend(["", "Native Full paired utility (positive = Native better):"])
    for key, values in report["summary"]["comparisons"].items():
        lines.append(
            f"{key}: chamfer={values['chamfer_l1']['mean']:+.8f} "
            f"f@0.02={values['fscore_0p02']['mean']:+.8f} "
            f"normal={values['normal_consistency']['mean']:+.8f} "
            f"lcr={values['largest_component_ratio']['mean']:+.8f}"
        )
    lines.extend(
        [
            "",
            "Input budgets:",
            "- Native Full / Stock: all frozen posed views (2/4/8).",
            "- official GenRecon: all frozen posed views (2/4/8), one unit chunk.",
            "- official Pixal3D: one frozen largest-mask view.",
            "",
            "Scope guard: retrospective matched exploratory evaluation. Official GenRecon is a",
            "scene-scale model evaluated cross-domain on centred single objects.",
            "Use a fresh untouched source-balanced holdout for a formal claim.",
        ]
    )
    return "\n".join(lines) + "\n"


def command_evaluate(args: argparse.Namespace) -> None:
    protocol_path = args.protocol.resolve()
    protocol = validate_benchmark_protocol(protocol_path)
    protocol["_protocol_path"] = str(protocol_path)
    pixal_transform = validate_transform(args.pixal_transform.resolve())
    genrecon_root = args.genrecon_output_root.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite immutable score output: {output_dir}")
    output_dir.mkdir(parents=True)

    records = []
    runtime_bindings = []
    for case_index, case in enumerate(protocol["cases"]):
        print(f"[unified_geometry] {case_index + 1}/{len(protocol['cases'])} {case['case_id']}")
        target, meshes, method_bindings, runtime = _method_meshes(
            protocol_path=protocol_path,
            protocol=protocol,
            case=case,
            pixal_transform=pixal_transform,
            genrecon_root=genrecon_root,
        )
        methods = {}
        if "t3_split_phase_position" in case:
            surface_seed = (
                int(case["current_seed"]) * 1009
                + int(case["t3_split_phase_position"]) * 9173
            )
            surface_seed_policy = "exact T3 seed*1009+final_phase_position*9173"
        else:
            surface_seed = int(args.seed) + case_index * 100
            surface_seed_policy = "benchmark_seed+case_index*100"
        for method in METHODS:
            surface = surface_metrics(
                meshes[method],
                target,
                count=int(args.surface_samples),
                # Reuse identical deterministic surface variates for every
                # method in a case; method ordering must not perturb metrics.
                seed=surface_seed,
                thresholds=(0.01, 0.02, 0.05),
            )
            structure = mesh_structure_metrics(meshes[method])
            methods[method] = {**surface, **structure}
            if methods[method]["mesh_success"] is not True:
                raise RuntimeError(f"mesh failed structure audit: {case['case_id']} {method}")
        t3_replay = None
        if "t3_split_phase_position" in case:
            differences = {}
            for method, branch in (("native_full", "native"), ("stock", "stock")):
                frozen = case["t3_mesh_surface"][branch]
                for metric in (
                    "chamfer_l1",
                    "pred_to_gt_mean",
                    "gt_to_pred_mean",
                    "fscore_0p02",
                    "normal_consistency",
                ):
                    differences[f"{method}.{metric}"] = abs(
                        float(methods[method][metric]) - float(frozen[metric])
                    )
            maximum = max(differences.values(), default=0.0)
            if maximum > 1.0e-4:
                raise RuntimeError(
                    f"T3 canonical metric replay changed for {case['case_id']}: {maximum}"
                )
            t3_replay = {
                "passed": True,
                "max_abs": maximum,
                "tolerance": 1.0e-4,
                "differences": differences,
            }
        records.append(
            {
                "case_id": case["case_id"],
                "uid": case["uid"],
                "source": case["source"],
                "view_count": int(case["view_count"]),
                "selected_view_id": case["selected_view_id"],
                "surface_seed": surface_seed,
                "surface_seed_policy": surface_seed_policy,
                "t3_metric_replay": t3_replay,
                "target_mesh": case["target_mesh"],
                "method_bindings": method_bindings,
                "methods": methods,
            }
        )
        runtime_bindings.append(
            {
                "case_id": case["case_id"],
                "pixal3d_result": runtime["pixal3d"]["mesh_sha256"],
                "genrecon_result_sha256": runtime["genrecon"]["result_sha256"],
            }
        )

    methods, comparisons = _summaries(
        records,
        bootstrap_samples=int(args.bootstrap_samples),
        seed=int(args.seed),
    )
    report = {
        "format": FORMAT,
        "passed": True,
        "formal": False,
        "scope_guard": (
            "retrospective matched geometry evaluation on a Stock/Full-used set; "
            "not an untouched formal SOTA claim"
        ),
        "cross_domain_guard": (
            "official GenRecon is scene-scale and is evaluated here on centred "
            "single-object inputs"
        ),
        "protocol": binding(protocol_path),
        "protocol_sha256": protocol["protocol_sha256"],
        "pixal_transform": binding(args.pixal_transform.resolve()),
        "genrecon_output_root": str(genrecon_root),
        "object_count": len(records),
        "surface_samples": int(args.surface_samples),
        "bootstrap_samples": int(args.bootstrap_samples),
        "thresholds": [0.01, 0.02, 0.05],
        "matched_t3_surface_seed_replay": all(
            row["surface_seed_policy"]
            == "exact T3 seed*1009+final_phase_position*9173"
            for row in records
        ),
        "t3_metric_replay_max_abs": max(
            (
                float(row["t3_metric_replay"]["max_abs"])
                for row in records
                if row["t3_metric_replay"] is not None
            ),
            default=None,
        ),
        "coordinate_policy": {
            "primary": "canonical pose without GT fitting",
            "native_full": "identity; T3 transform_pose=False canonical latent frame",
            "stock": "identity; T3 transform_pose=False canonical latent frame",
            "pixal3d_official": "frozen score-independent official-final transform v2",
            "genrecon_official": "identity chunk0 canonical coordinates",
            "forbidden": ["per-method ICP", "per-method scaling", "reflection"],
        },
        "input_budget": {
            "native_full": "all frozen posed views",
            "stock": "all frozen posed views",
            "genrecon_official": "all frozen posed views",
            "pixal3d_official": "one frozen largest-mask view",
        },
        "methods": list(METHODS),
        "records": records,
        "runtime_bindings": runtime_bindings,
        "summary": {"methods": methods, "comparisons": comparisons},
        "code": binding(Path(__file__).resolve()),
    }
    report["report_sha256"] = canonical_sha256(report)
    atomic_json(output_dir / "report.json", report)
    atomic_text(output_dir / "summary.txt", _summary_text(report))
    print(_summary_text(report), end="")
    print(f"report: {output_dir / 'report.json'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--pixal_transform", type=Path, required=True)
    parser.add_argument("--genrecon_output_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--surface_samples", type=int, default=20000)
    parser.add_argument("--bootstrap_samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260802)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.surface_samples <= 0 or args.bootstrap_samples <= 0:
        raise ValueError("sample counts must be positive")
    command_evaluate(args)


if __name__ == "__main__":
    main()
