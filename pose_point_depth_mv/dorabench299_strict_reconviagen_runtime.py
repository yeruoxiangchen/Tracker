#!/usr/bin/env python3
"""Freeze, audit, repair-plan and aggregate the Dora299 strict runtime."""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
from typing import Any

from manual_mesh_reconstruction.common import (
    atomic_json,
    canonical_sha256,
    load_json,
    sha256_file,
)
from pose_point_depth_mv.dataset_tools.build_reconviagen_dorabench300_benchmark import (
    FINAL_FORMAT,
)
from pose_point_depth_mv.evaluate_dorabench299_strict_reconviagen import (
    INFERENCE_FORMAT,
    SUBSET_FORMAT,
)


RESULT_FORMAT = "pose_point_depth_mv.omni_real_reconviagen_inference.v1"
RUNTIME_FORMAT = "pose_point_depth_mv.omni_real_runtime_input_manifest.v3"
CURRENT_FORMAT = (
    "reconviagen.dorabench_dora300_ss30k_slat30k_metric_aggregate_failure_aware.v1"
)
EXCLUDED_UID = "dora_39170f9710c47fb395de"
EXCLUDED_KEY = f"Level4:{EXCLUDED_UID}"


def _parent_inputs(
    benchmark_path: Path, runtime_path: Path, current_path: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    benchmark = load_json(benchmark_path)
    runtime = load_json(runtime_path)
    current = load_json(current_path)
    identity_payload = dict(benchmark)
    saved_identity = str(identity_payload.pop("manifest_identity", ""))
    if (
        benchmark.get("format") != FINAL_FORMAT
        or int(benchmark.get("object_count", -1)) != 300
        or len(benchmark.get("objects") or []) != 300
        or not saved_identity
        or canonical_sha256(identity_payload) != saved_identity
    ):
        raise RuntimeError("frozen Dora300 benchmark identity differs")
    benchmark_sha = sha256_file(benchmark_path)
    benchmark_keys = [
        f"{row['category']}:{row['uid']}" for row in benchmark["objects"]
    ]
    runtime_keys = [str(row["object_key"]) for row in runtime.get("objects") or []]
    if (
        runtime.get("format") != RUNTIME_FORMAT
        or runtime.get("passed") is not True
        or runtime.get("benchmark_manifest_sha256") != benchmark_sha
        or int(runtime.get("selected_object_count", -1)) != 300
        or int(runtime.get("completed_object_count", -1)) != 300
        or len(runtime_keys) != 300
        or set(runtime_keys) != set(benchmark_keys)
        or any(int(row.get("selected_view_count", -1)) != 4 for row in runtime["objects"])
    ):
        raise RuntimeError("frozen Dora300 runtime identity differs")
    failures = list(current.get("registered_model_output_failures") or [])
    current_uids = [str(row["uid"]) for row in current.get("objects") or []]
    expected_uids = {
        str(row["uid"])
        for row in benchmark["objects"]
        if str(row["uid"]) != EXCLUDED_UID
    }
    if (
        current.get("format") != CURRENT_FORMAT
        or current.get("passed") is not True
        or current.get("benchmark_manifest_sha256") != benchmark_sha
        or int(current.get("requested_object_count", -1)) != 300
        or int(current.get("surface_metric_object_count", -1)) != 299
        or int(current.get("registered_model_output_failure_count", -1)) != 1
        or len(failures) != 1
        or str(failures[0].get("uid")) != EXCLUDED_UID
        or str(failures[0].get("object_key")) != EXCLUDED_KEY
        or str(failures[0].get("stage"))
        != "native_slat_mesh_decode_preflight"
        or len(current_uids) != 299
        or len(set(current_uids)) != 299
        or set(current_uids) != expected_uids
    ):
        raise RuntimeError("registered current-model 299/failure identity differs")
    return benchmark, runtime, current


def prepare(args: argparse.Namespace) -> None:
    benchmark_path = Path(args.benchmark_manifest).expanduser().resolve(strict=True)
    runtime_path = Path(args.runtime_input_manifest).expanduser().resolve(strict=True)
    current_path = Path(args.current_report).expanduser().resolve(strict=True)
    benchmark, _, current = _parent_inputs(benchmark_path, runtime_path, current_path)
    rows = [
        row for row in benchmark["objects"] if str(row["uid"]) != EXCLUDED_UID
    ]
    failure = current["registered_model_output_failures"][0]
    payload = {
        "format": SUBSET_FORMAT,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "passed": True,
        "selection_policy": (
            "frozen Dora300 minus the one preregistered SS30K+SLat30K "
            "active-point-limit model-output failure"
        ),
        "object_count": 299,
        "input_views_per_object": 4,
        "selected_input_view_indices": [0, 9, 19, 29],
        "parent_benchmark_manifest": {
            "path": str(benchmark_path),
            "sha256": sha256_file(benchmark_path),
            "manifest_identity": benchmark["manifest_identity"],
        },
        "runtime_input_manifest": {
            "path": str(runtime_path),
            "sha256": sha256_file(runtime_path),
        },
        "current_report": {
            "path": str(current_path),
            "sha256": sha256_file(current_path),
        },
        "excluded_current_failure": {
            "object_key": EXCLUDED_KEY,
            "uid": EXCLUDED_UID,
            "category": "Level4",
            "stage": failure["stage"],
            "coord_count": int(failure["coord_count"]),
            "error": failure["error"],
        },
        "objects": rows,
    }
    payload["subset_identity"] = canonical_sha256(payload)
    destination = Path(args.output).expanduser().resolve()
    if destination.exists():
        observed = load_json(destination)
        if observed != payload:
            # created_at is intentionally frozen by the first successful write.
            check = dict(observed)
            identity = str(check.pop("subset_identity", ""))
            if (
                observed.get("format") != SUBSET_FORMAT
                or canonical_sha256(check) != identity
                or observed.get("objects") != rows
                or observed.get("parent_benchmark_manifest")
                != payload["parent_benchmark_manifest"]
                or observed.get("runtime_input_manifest")
                != payload["runtime_input_manifest"]
                or observed.get("current_report") != payload["current_report"]
                or observed.get("excluded_current_failure")
                != payload["excluded_current_failure"]
            ):
                raise RuntimeError(f"existing subset differs: {destination}")
            payload = observed
    else:
        atomic_json(destination, payload)
    print(
        {
            "passed": True,
            "objects": 299,
            "excluded": EXCLUDED_KEY,
            "subset_identity": payload["subset_identity"],
            "output": str(destination),
        }
    )


def _valid_results(
    output_root: Path,
    subset: dict[str, Any],
    runtime_path: Path,
    seed: int,
) -> dict[str, tuple[dict[str, Any], Path]]:
    expected = {f"{row['category']}:{row['uid']}" for row in subset["objects"]}
    runtime_sha = sha256_file(runtime_path)
    found: dict[str, tuple[dict[str, Any], Path]] = {}
    for path in sorted(output_root.glob(f"**/seed_{seed}/result.json")):
        row = load_json(path)
        if row.get("format") != RESULT_FORMAT or row.get("passed") is not True:
            continue
        key = str(row.get("object_key", ""))
        if key not in expected or int(row.get("seed", -1)) != seed:
            continue
        if row.get("runtime_input_manifest_sha256") != runtime_sha:
            raise RuntimeError(f"strict result runtime identity differs: {path}")
        mesh = Path(str(row["mesh"])).resolve(strict=True)
        if sha256_file(mesh) != str(row["mesh_sha256"]):
            raise RuntimeError(f"strict result Mesh hash differs: {mesh}")
        if key in found:
            raise RuntimeError(
                f"duplicate strict result for {key}: {found[key][1]} and {path}"
            )
        found[key] = (row, path)
    return found


def plan(args: argparse.Namespace) -> None:
    subset_path = Path(args.subset_manifest).expanduser().resolve(strict=True)
    subset = load_json(subset_path)
    output_root = Path(args.output_root).expanduser().resolve(strict=True)
    runtime_path = Path(args.runtime_input_manifest).expanduser().resolve(strict=True)
    found = _valid_results(output_root, subset, runtime_path, int(args.seed))
    expected = [f"{row['category']}:{row['uid']}" for row in subset["objects"]]
    missing = [key for key in expected if key not in found]
    plan_root = Path(args.plan_root).expanduser().resolve()
    plan_root.mkdir(parents=True, exist_ok=True)
    for worker in range(int(args.worker_count)):
        values = missing[worker :: int(args.worker_count)]
        (plan_root / f"{args.stage}_worker{worker}.txt").write_text(
            "\n".join(values) + ("\n" if values else ""), encoding="utf-8"
        )
    atomic_json(
        plan_root / f"{args.stage}_summary.json",
        {
            "stage": str(args.stage),
            "complete": len(found),
            "missing": len(missing),
            "missing_keys": missing,
            "worker_count": int(args.worker_count),
        },
    )
    print({"stage": args.stage, "complete": len(found), "missing": len(missing)})


def aggregate(args: argparse.Namespace) -> None:
    subset_path = Path(args.subset_manifest).expanduser().resolve(strict=True)
    subset = load_json(subset_path)
    output_root = Path(args.output_root).expanduser().resolve(strict=True)
    runtime_path = Path(args.runtime_input_manifest).expanduser().resolve(strict=True)
    found = _valid_results(output_root, subset, runtime_path, int(args.seed))
    expected = [f"{row['category']}:{row['uid']}" for row in subset["objects"]]
    missing = [key for key in expected if key not in found]
    if missing or len(found) != 299:
        raise RuntimeError(
            f"strict Dora299 coverage differs: complete={len(found)} missing={missing}"
        )
    objects = [found[key][0] for key in expected]
    bindings = [
        {"path": str(found[key][1]), "sha256": sha256_file(found[key][1])}
        for key in expected
    ]
    payload = {
        "format": INFERENCE_FORMAT,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "passed": True,
        "method": (
            "strict original ReconViaGen release: VGGT -> Stock SS -> "
            "Stock SLat -> Stock Mesh decoder"
        ),
        "pretrained": str(args.pretrained),
        "seed": int(args.seed),
        "object_count": 299,
        "record_count": 299,
        "input_views_per_object": 4,
        "subset_manifest": str(subset_path),
        "subset_manifest_sha256": sha256_file(subset_path),
        "subset_identity": subset["subset_identity"],
        "runtime_input_manifest": {
            "path": str(runtime_path),
            "sha256": sha256_file(runtime_path),
        },
        "atomic_result_bindings": bindings,
        "objects": objects,
        "metric_or_target_consumed_during_inference": False,
        "output_frame": "decoder-native; transform_pose=False",
    }
    destination = Path(args.output).expanduser().resolve()
    atomic_json(destination, payload)
    print({"passed": True, "objects": 299, "output": str(destination)})


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--benchmark_manifest", required=True)
    p.add_argument("--runtime_input_manifest", required=True)
    p.add_argument("--current_report", required=True)
    p.add_argument("--output", required=True)
    p.set_defaults(func=prepare)
    p = sub.add_parser("plan")
    p.add_argument("--subset_manifest", required=True)
    p.add_argument("--runtime_input_manifest", required=True)
    p.add_argument("--output_root", required=True)
    p.add_argument("--plan_root", required=True)
    p.add_argument("--stage", required=True)
    p.add_argument("--worker_count", type=int, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=plan)
    p = sub.add_parser("aggregate")
    p.add_argument("--subset_manifest", required=True)
    p.add_argument("--runtime_input_manifest", required=True)
    p.add_argument("--output_root", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--pretrained", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=aggregate)
    return parser


def main() -> None:
    args = make_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
