#!/usr/bin/env python3
"""Repair a Dora300 worker after a registered active-point decoder refusal.

The repair is deliberately failure-aware.  It never prunes support or raises
the decoder safety limit.  Successful suffix objects retain their original
worker-local master-noise positions; the oversized object remains visible as
one model-output failure and surface metrics are reported over valid meshes.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shutil
from typing import Any

import numpy as np

from manual_mesh_reconstruction.common import atomic_json, load_json, sha256_file
from manual_mesh_reconstruction.current_model import (
    MANIFEST_FORMAT as CURRENT_INFERENCE_FORMAT,
    REPORT_FORMAT as CURRENT_RESULT_FORMAT,
)
from manual_mesh_reconstruction.model_inputs import (
    MANIFEST_FORMAT as MODEL_INPUT_MANIFEST_FORMAT,
)
from pose_point_depth_mv.evaluate_proobjaverse_official_native_ss_stock_slat import (
    MAX_SAFE_SLAT_DECODER_INPUT_POINTS,
)
from pose_point_depth_mv import evaluate_dorabench300_ss30k_slat30k as dora


PLAN_FORMAT = "reconviagen.dorabench_dora300_active_point_repair_plan.v1"
FAILURE_FORMAT = "reconviagen.dorabench_dora300_model_output_failure.v1"
AGGREGATE_FORMAT = (
    "reconviagen.dorabench_dora300_ss30k_slat30k_"
    "metric_aggregate_failure_aware.v1"
)


def _key(row: dict[str, Any]) -> str:
    return f"{row['category']}:{row['object_id']}"


def _expected_rows(
    benchmark_path: Path,
    model_manifest_path: Path,
    *,
    worker_index: int,
    num_workers: int,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    benchmark = dora.base._benchmark(benchmark_path, expected_objects=300)
    model_manifest = load_json(model_manifest_path)
    if (
        model_manifest.get("format") != MODEL_INPUT_MANIFEST_FORMAT
        or model_manifest.get("passed") is not True
    ):
        raise RuntimeError("worker model-input manifest did not pass")
    expected_benchmark = [
        row
        for index, row in enumerate(benchmark["objects"])
        if index % int(num_workers) == int(worker_index)
    ]
    model_rows = list(model_manifest.get("objects") or [])
    expected_keys = [f"{row['category']}:{row['uid']}" for row in expected_benchmark]
    model_keys = [_key(row) for row in model_rows]
    if len(model_keys) != len(expected_keys) or set(model_keys) != set(expected_keys):
        raise RuntimeError("worker model-input membership differs from frozen Dora partition")
    return benchmark, model_manifest, model_rows


def _scan(
    model_rows: list[dict[str, Any]],
    inference_root: Path,
    *,
    seed: int,
    limit: int,
) -> dict[str, Any]:
    completed: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for position, row in enumerate(model_rows):
        category = str(row["category"])
        uid = str(row["object_id"])
        object_key = _key(row)
        result_path = (
            inference_root / "meshes" / category / uid / f"seed_{seed}" / "result.json"
        )
        mesh_path = result_path.with_name("mesh_o.obj")
        coord_path = inference_root / "ss_coords" / category / uid / f"seed_{seed}.npz"
        coord_report_path = coord_path.with_suffix(".json")
        if not coord_path.is_file() or not coord_report_path.is_file():
            raise RuntimeError(f"missing frozen Native-SS coordinate artifact: {object_key}")
        coord_report = load_json(coord_report_path)
        if (
            coord_report.get("passed") is not True
            or coord_report.get("object_key") != object_key
            or int(coord_report.get("seed", -1)) != int(seed)
            or coord_report.get("coords_sha256") != sha256_file(coord_path)
            or coord_report.get("model_input_sha256") != row["model_input_sha256"]
        ):
            raise RuntimeError(f"Native-SS coordinate binding differs: {object_key}")
        with np.load(coord_path, allow_pickle=False) as payload:
            coords = np.asarray(payload["coords"])
        coord_count = int(len(coords))
        if coord_count != int(coord_report.get("coord_count", -1)):
            raise RuntimeError(f"Native-SS coordinate count differs: {object_key}")
        common = {
            "position": int(position),
            "object_key": object_key,
            "category": category,
            "uid": uid,
            "seed": int(seed),
            "coord_count": coord_count,
            "coords": str(coord_path.resolve()),
            "coords_sha256": sha256_file(coord_path),
            "coord_report": str(coord_report_path.resolve()),
            "coord_report_sha256": sha256_file(coord_report_path),
            "model_input_sha256": row["model_input_sha256"],
        }
        if result_path.is_file() or mesh_path.is_file():
            if not result_path.is_file() or not mesh_path.is_file():
                raise RuntimeError(f"partial Mesh result: {object_key}")
            result = load_json(result_path)
            expected_master_seed = int(seed) * 2_000_003 + int(position) * 2_017 + 7_919
            checks = {
                "format": CURRENT_RESULT_FORMAT,
                "passed": True,
                "object_key": object_key,
                "object_id": uid,
                "seed": int(seed),
                "model_input_sha256": row["model_input_sha256"],
                "mesh_sha256": sha256_file(mesh_path),
                "master_noise_seed": expected_master_seed,
            }
            mismatch = {
                name: (result.get(name), value)
                for name, value in checks.items()
                if result.get(name) != value
            }
            if mismatch:
                raise RuntimeError(f"completed Mesh binding differs={object_key}: {mismatch}")
            completed.append(
                {
                    **common,
                    "result": str(result_path.resolve()),
                    "result_sha256": sha256_file(result_path),
                    "mesh": str(mesh_path.resolve()),
                    "mesh_sha256": sha256_file(mesh_path),
                }
            )
        elif coord_count > int(limit):
            failures.append(
                {
                    **common,
                    "stage": "native_slat_mesh_decode_preflight",
                    "error": {
                        "type": "RuntimeError",
                        "kind": "slat_decoder_active_point_limit_exceeded",
                        "message": (
                            "SLat decoder input exceeds safe active-point limit: "
                            f"points={coord_count} limit={int(limit)}"
                        ),
                    },
                    "safe_active_point_limit": int(limit),
                    "retryable_on_current_runtime": False,
                }
            )
        else:
            pending.append(common)
    return {"completed": completed, "pending": pending, "failures": failures}


def cmd_plan(args: argparse.Namespace) -> None:
    benchmark_path = Path(args.benchmark_manifest).expanduser().resolve(strict=True)
    model_path = Path(args.model_input_manifest).expanduser().resolve(strict=True)
    inference_root = Path(args.inference_root).expanduser().resolve(strict=True)
    _, _, rows = _expected_rows(
        benchmark_path,
        model_path,
        worker_index=args.worker_index,
        num_workers=args.num_workers,
    )
    scan = _scan(rows, inference_root, seed=args.seed, limit=args.limit)
    if len(scan["failures"]) != 1:
        raise RuntimeError(
            f"expected exactly one registered active-point failure; got {len(scan['failures'])}"
        )
    pending_positions = [int(row["position"]) for row in scan["pending"]]
    if pending_positions and pending_positions != list(
        range(pending_positions[0], pending_positions[0] + len(pending_positions))
    ):
        raise RuntimeError("repair candidates are not one contiguous suffix")
    if pending_positions and pending_positions[0] != int(scan["failures"][0]["position"]) + 1:
        raise RuntimeError("repair suffix does not begin immediately after the failure")
    plan = {
        "format": PLAN_FORMAT,
        "created_at_utc": dora.base.utc_now(),
        "passed": True,
        "benchmark_manifest": str(benchmark_path),
        "benchmark_manifest_sha256": sha256_file(benchmark_path),
        "model_input_manifest": str(model_path),
        "model_input_manifest_sha256": sha256_file(model_path),
        "inference_root": str(inference_root),
        "worker_index": int(args.worker_index),
        "num_workers": int(args.num_workers),
        "requested_object_count": len(rows),
        "completed_object_count": len(scan["completed"]),
        "pending_object_count": len(scan["pending"]),
        "pending_objects": scan["pending"],
        "master_position_offset": (
            int(scan["pending"][0]["position"]) if scan["pending"] else None
        ),
        "registered_failure_count": len(scan["failures"]),
        "registered_failures": scan["failures"],
        "safety_policy": "limit remains frozen; support is not pruned",
    }
    output = Path(args.output).expanduser().resolve()
    atomic_json(output, plan)
    print(json.dumps(plan, indent=2, ensure_ascii=False))


def cmd_finalize_worker(args: argparse.Namespace) -> None:
    benchmark_path = Path(args.benchmark_manifest).expanduser().resolve(strict=True)
    model_path = Path(args.model_input_manifest).expanduser().resolve(strict=True)
    inference_root = Path(args.inference_root).expanduser().resolve(strict=True)
    template_path = Path(args.template_manifest).expanduser().resolve(strict=True)
    output_path = Path(args.output).expanduser().resolve()
    _, model_manifest, rows = _expected_rows(
        benchmark_path,
        model_path,
        worker_index=args.worker_index,
        num_workers=args.num_workers,
    )
    scan = _scan(rows, inference_root, seed=args.seed, limit=args.limit)
    if scan["pending"] or len(scan["failures"]) != 1:
        raise RuntimeError(
            f"repair is incomplete: pending={len(scan['pending'])} "
            f"failures={len(scan['failures'])}"
        )
    template = load_json(template_path)
    if template.get("format") != CURRENT_INFERENCE_FORMAT or template.get("passed") is not True:
        raise RuntimeError("current-model template manifest did not pass")
    results = []
    for item in scan["completed"]:
        result = load_json(Path(item["result"]))
        for name in (
            "native_ss_report_sha256",
            "native_ss_checkpoint_sha256",
            "native_ss_weights",
            "native_slat_checkpoint_sha256",
            "native_slat_weights",
            "stock_slat_freeze_sha256",
            "cross_deployment_bridge_sha256",
        ):
            expected = template["objects"][0][name]
            if result.get(name) != expected:
                raise RuntimeError(f"deployment binding differs={item['object_key']}:{name}")
        results.append(result)
    failure = copy.deepcopy(scan["failures"][0])
    failure["format"] = FAILURE_FORMAT
    failure["passed"] = False
    failure["native_ss_report_sha256"] = template["native_ss_deployment"]["report_sha256"]
    failure["native_ss_checkpoint_sha256"] = template["native_ss_deployment"][
        "checkpoint_sha256"
    ]
    failure["native_slat_checkpoint_sha256"] = template["native_slat_deployment"][
        "checkpoint_sha256"
    ]
    failure["stock_slat_freeze_sha256"] = template["stock_slat_freeze_sha256"]
    failure_path = output_path.parent / "recorded_model_output_failures" / (
        f"{failure['uid']}_seed_{int(args.seed)}.json"
    )
    atomic_json(failure_path, failure)
    failure_ref = {
        **failure,
        "failure_report": str(failure_path),
        "failure_report_sha256": sha256_file(failure_path),
    }
    if output_path.is_file():
        prior = load_json(output_path)
        if prior.get("complete_with_registered_failures") is not True:
            archive = output_path.parent / "repair_worker01_active_point_v1" / (
                "inference_manifest_suffix_repair.json"
            )
            archive.parent.mkdir(parents=True, exist_ok=True)
            if not archive.exists():
                shutil.copy2(output_path, archive)
    combined = copy.deepcopy(template)
    combined.update(
        {
            "created_at_utc": dora.base.utc_now(),
            "model_input_manifest": str(model_path),
            "model_input_manifest_sha256": sha256_file(model_path),
            "runtime_input_manifest": model_manifest["runtime_input_manifest"],
            "runtime_input_manifest_sha256": model_manifest[
                "runtime_input_manifest_sha256"
            ],
            "object_count": len(results),
            "record_count": len(results),
            "objects": results,
            "requested_object_count": len(rows),
            "valid_mesh_object_count": len(results),
            "registered_model_output_failure_count": 1,
            "registered_model_output_failures": [failure_ref],
            "complete_with_registered_failures": True,
            "master_position_offset": 0,
            "master_noise_position_contract": (
                "original frozen worker-local position; repaired suffix uses exact offset"
            ),
            "passed": True,
            "scope_guard": (
                "Worker partition is complete as 42 valid Meshes plus one explicitly "
                "registered 97,218-point decoder-safety refusal. No support pruning, "
                "safety-limit increase, target access, or noise-position change."
            ),
        }
    )
    atomic_json(output_path, combined)
    print(
        json.dumps(
            {
                "passed": True,
                "requested": len(rows),
                "valid_meshes": len(results),
                "registered_failures": 1,
                "manifest": str(output_path),
            },
            indent=2,
        )
    )


def cmd_aggregate(args: argparse.Namespace) -> None:
    benchmark_path = Path(args.benchmark_manifest).expanduser().resolve(strict=True)
    benchmark = dora.base._benchmark(benchmark_path, expected_objects=args.expected_objects)
    workers_root = Path(args.workers_root).expanduser().resolve(strict=True)
    metric_paths = sorted(workers_root.glob("worker_*/03_metrics/metrics_report.json"))
    if len(metric_paths) != int(args.expected_workers):
        raise RuntimeError(
            f"metric worker report count differs: {len(metric_paths)} != {args.expected_workers}"
        )
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    inference_refs = []
    for worker_index in range(int(args.expected_workers)):
        worker = workers_root / f"worker_{worker_index:02d}"
        inference_path = worker / "02_current_ss30k_slat30k" / "inference_manifest.json"
        metric_path = worker / "03_metrics" / "metrics_report.json"
        inference = load_json(inference_path)
        metric = load_json(metric_path)
        if inference.get("format") != CURRENT_INFERENCE_FORMAT or inference.get("passed") is not True:
            raise RuntimeError(f"worker inference manifest did not pass: {inference_path}")
        if metric.get("format") != dora.base.METRIC_WORKER_FORMAT or metric.get("passed") is not True:
            raise RuntimeError(f"worker metric report did not pass: {metric_path}")
        if metric.get("benchmark_manifest_sha256") != sha256_file(benchmark_path):
            raise RuntimeError(f"worker benchmark identity differs: {metric_path}")
        if metric.get("inference_manifest_sha256") != sha256_file(inference_path):
            raise RuntimeError(f"worker inference identity differs: {metric_path}")
        rows.extend(metric["objects"])
        failures.extend(inference.get("registered_model_output_failures") or [])
        inference_refs.append(
            {"path": str(inference_path), "sha256": sha256_file(inference_path)}
        )
    success_uids = [str(row["uid"]) for row in rows]
    failure_uids = [str(row["uid"]) for row in failures]
    expected_uids = {str(row["uid"]) for row in benchmark["objects"]}
    if len(success_uids) != len(set(success_uids)):
        raise RuntimeError("duplicate successful metric UIDs")
    if len(failure_uids) != len(set(failure_uids)):
        raise RuntimeError("duplicate registered failure UIDs")
    if set(success_uids) & set(failure_uids):
        raise RuntimeError("successful and failed UID sets overlap")
    if set(success_uids) | set(failure_uids) != expected_uids:
        raise RuntimeError("success plus registered-failure sets do not cover Dora300")
    if len(failures) != int(args.expected_failures):
        raise RuntimeError(
            f"registered failure count differs: {len(failures)} != {args.expected_failures}"
        )
    by_level: dict[str, dict[str, Any]] = {}
    for level in sorted({str(row["category"]) for row in rows}):
        selected = [row for row in rows if str(row["category"]) == level]
        by_level[level] = {
            "surface_metric_object_count": len(selected),
            "chamfer_distance": dora.base._summary(
                [float(row["metrics"]["chamfer_distance"]) for row in selected]
            ),
            "fscore": dora.base._summary(
                [float(row["metrics"]["fscore"]) for row in selected]
            ),
            "registered_model_output_failure_count": sum(
                str(item["category"]) == level for item in failures
            ),
        }
    report = {
        "format": AGGREGATE_FORMAT,
        "created_at_utc": dora.base.utc_now(),
        "passed": True,
        "runtime_integrity_passed": True,
        "all_model_outputs_passed": len(failures) == 0,
        "method": "no-VGGT SS30K step30000 + no-VGGT SLat30K step30000",
        "benchmark": "Dora-Bench-300 registered reproduction",
        "benchmark_manifest": str(benchmark_path),
        "benchmark_manifest_sha256": sha256_file(benchmark_path),
        "protocol_sha256": benchmark["protocol_sha256"],
        "requested_object_count": int(args.expected_objects),
        "valid_mesh_object_count": len(rows),
        "mesh_success_rate": float(len(rows) / int(args.expected_objects)),
        "registered_model_output_failure_count": len(failures),
        "registered_model_output_failures": failures,
        "surface_metric_object_count": len(rows),
        "surface_metrics_exclude_registered_failures": True,
        "surface_points_per_mesh": int(args.surface_points),
        "fscore_radius": float(args.fscore_radius),
        "chamfer_distance": dora.base._summary(
            [float(row["metrics"]["chamfer_distance"]) for row in rows]
        ),
        "fscore": dora.base._summary(
            [float(row["metrics"]["fscore"]) for row in rows]
        ),
        "by_complexity_level": by_level,
        "objects": rows,
        "metric_worker_reports": [
            {"path": str(path), "sha256": sha256_file(path)} for path in metric_paths
        ],
        "inference_manifests": inference_refs,
        "reconviagen_model_loaded_or_run": False,
        "scope_guard": (
            "Dora300 is fully accounted for as valid Meshes plus explicit model-output "
            "failures. CD/F-score summarize valid decoded surfaces only; the Mesh success "
            "rate and failed-object record must be reported beside them."
        ),
    }
    output_dir = Path(args.output_dir).expanduser().resolve()
    atomic_json(output_dir / "report.json", report)
    summary = [
        "Dora-Bench-300 SS30K+SLat30K failure-aware evaluation",
        "======================================================",
        f"requested objects: {int(args.expected_objects)}",
        f"valid Meshes: {len(rows)}/{int(args.expected_objects)} ({report['mesh_success_rate']:.6f})",
        f"registered model-output failures: {len(failures)}",
        f"surface metric objects: {len(rows)}",
        (
            "Chamfer Distance (valid surfaces): "
            f"mean={report['chamfer_distance']['mean']:.8f} "
            f"median={report['chamfer_distance']['median']:.8f}"
        ),
        (
            "F-score@0.1 (valid surfaces): "
            f"mean={report['fscore']['mean']:.8f} "
            f"median={report['fscore']['median']:.8f}"
        ),
        "ReconViaGen original loaded/run: no",
        f"report: {output_dir / 'report.json'}",
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print("\n".join(summary))


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--benchmark_manifest", required=True)
    common.add_argument("--model_input_manifest", required=True)
    common.add_argument("--inference_root", required=True)
    common.add_argument("--worker_index", type=int, default=1)
    common.add_argument("--num_workers", type=int, default=7)
    common.add_argument("--seed", type=int, default=42)
    common.add_argument("--limit", type=int, default=MAX_SAFE_SLAT_DECODER_INPUT_POINTS)
    plan = sub.add_parser("plan", parents=[common])
    plan.add_argument("--output", required=True)
    plan.set_defaults(func=cmd_plan)
    finalize = sub.add_parser("finalize-worker", parents=[common])
    finalize.add_argument("--template_manifest", required=True)
    finalize.add_argument("--output", required=True)
    finalize.set_defaults(func=cmd_finalize_worker)
    aggregate = sub.add_parser("aggregate")
    aggregate.add_argument("--benchmark_manifest", required=True)
    aggregate.add_argument("--workers_root", required=True)
    aggregate.add_argument("--output_dir", required=True)
    aggregate.add_argument("--expected_workers", type=int, default=7)
    aggregate.add_argument("--expected_objects", type=int, default=300)
    aggregate.add_argument("--expected_failures", type=int, default=1)
    aggregate.add_argument("--surface_points", type=int, default=100000)
    aggregate.add_argument("--fscore_radius", type=float, default=0.1)
    aggregate.set_defaults(func=cmd_aggregate)
    return parser


def main() -> None:
    args = make_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
