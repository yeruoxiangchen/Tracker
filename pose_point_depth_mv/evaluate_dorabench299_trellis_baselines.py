#!/usr/bin/env python3
"""Run and score frozen TRELLIS-S/TRELLIS-M on Dora-Bench 299.

The two baselines share the official ``microsoft/TRELLIS-image-large``
checkpoint and the exact four frozen Dora views (0, 9, 19, 29).  TRELLIS-S
uses the official stochastic multi-image sampler and TRELLIS-M uses the
official multidiffusion sampler.  Inference never reads a target Mesh.
"""

from __future__ import annotations

import argparse
import datetime as dt
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
from PIL import Image
import torch


TRACKER_ROOT = Path(__file__).resolve().parents[1]
for dependency in (TRACKER_ROOT, TRACKER_ROOT / "ReconViaGen"):
    if str(dependency) not in sys.path:
        sys.path.insert(0, str(dependency))

from manual_mesh_reconstruction.common import (  # noqa: E402
    atomic_json,
    load_json,
    sha256_file,
)
from pose_point_depth_mv.evaluate_dorabench299_strict_reconviagen import (  # noqa: E402
    _metrics,
    _subset,
)
from pose_point_depth_mv.evaluate_omni200_ss30k_slat30k import (  # noqa: E402
    _gt_mesh,
    _one_mesh,
    _summary,
)
from pose_point_depth_mv.mesh_benchmark_metrics import (  # noqa: E402
    mesh_structure_metrics,
)


RESULT_FORMAT = "reconviagen.dorabench_dora299_trellis_baseline_result.v1"
INFERENCE_FORMAT = "reconviagen.dorabench_dora299_trellis_baseline_inference.v1"
METRIC_OBJECT_FORMAT = "reconviagen.dorabench_dora299_trellis_baseline_metric.v1"
METRIC_WORKER_FORMAT = "reconviagen.dorabench_dora299_trellis_baseline_metric_worker.v1"
METRIC_AGGREGATE_FORMAT = "reconviagen.dorabench_dora299_trellis_baseline_metric_aggregate.v1"
BASELINES = {
    "trellis_s": "stochastic",
    "trellis_m": "multidiffusion",
}
EXPECTED_VIEWS = [0, 9, 19, 29]


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _sampling(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "sparse_structure": {
            "steps": int(args.ss_steps),
            "cfg_strength": float(args.ss_cfg),
        },
        "structured_latent": {
            "steps": int(args.slat_steps),
            "cfg_strength": float(args.slat_cfg),
        },
        "seed": int(args.seed),
    }


def _model_snapshot(pretrained: str, revision: str) -> Path:
    from huggingface_hub import snapshot_download

    snapshot = Path(
        snapshot_download(
            repo_id=str(pretrained),
            revision=str(revision),
            local_files_only=True,
        )
    ).resolve(strict=True)
    pipeline_json = snapshot / "pipeline.json"
    if not pipeline_json.is_file():
        raise RuntimeError(f"TRELLIS pipeline snapshot is incomplete: {snapshot}")
    return snapshot


def _selected_rows(subset: dict[str, Any], requested: list[str] | None) -> list[dict[str, Any]]:
    rows = list(subset["objects"])
    if not requested:
        return rows
    requested_set = set(requested)
    if len(requested_set) != len(requested):
        raise ValueError("--object values must be unique")
    selected = [row for row in rows if f"{row['category']}:{row['uid']}" in requested_set]
    observed = {f"{row['category']}:{row['uid']}" for row in selected}
    missing = requested_set - observed
    if missing:
        raise RuntimeError(f"requested Dora objects are absent from frozen subset: {sorted(missing)}")
    return selected


def _images(row: dict[str, Any]) -> tuple[list[Image.Image], list[dict[str, str]]]:
    if list(row.get("selected_input_view_indices") or []) != EXPECTED_VIEWS:
        raise RuntimeError(f"Dora view identity differs: {row['uid']}")
    paths = [Path(value).resolve(strict=True) for value in row.get("rgba_images") or []]
    if len(paths) != 4:
        raise RuntimeError(f"Dora input view count differs: {row['uid']}")
    images: list[Image.Image] = []
    bindings: list[dict[str, str]] = []
    for path in paths:
        with Image.open(path) as handle:
            image = handle.convert("RGBA").copy()
        alpha = np.asarray(image)[..., 3]
        if alpha.max() != 255 or alpha.min() == 255:
            raise RuntimeError(f"Dora RGBA alpha contract differs: {path}")
        images.append(image)
        bindings.append({"path": str(path), "sha256": sha256_file(path)})
    return images, bindings


def _result_paths(root: Path, row: dict[str, Any], seed: int) -> tuple[Path, Path]:
    destination = root / "objects" / str(row["category"]) / str(row["uid"]) / f"seed_{seed}"
    return destination / "mesh_model_o.obj", destination / "result.json"


def _valid_result(
    path: Path,
    mesh_path: Path,
    *,
    row: dict[str, Any],
    args: argparse.Namespace,
    subset_sha: str,
    pipeline_sha: str,
) -> dict[str, Any] | None:
    if not path.is_file() or not mesh_path.is_file():
        return None
    payload = load_json(path)
    expected = {
        "format": RESULT_FORMAT,
        "passed": True,
        "baseline": str(args.baseline),
        "multi_image_mode": BASELINES[str(args.baseline)],
        "object_key": f"{row['category']}:{row['uid']}",
        "seed": int(args.seed),
        "subset_manifest_sha256": subset_sha,
        "pretrained": str(args.pretrained),
        "model_revision": str(args.model_revision),
        "pipeline_json_sha256": pipeline_sha,
        "sampling": _sampling(args),
        "mesh_sha256": sha256_file(mesh_path),
    }
    mismatch = {key: (payload.get(key), value) for key, value in expected.items() if payload.get(key) != value}
    if mismatch:
        raise RuntimeError(f"stale TRELLIS baseline result {path}: {mismatch}")
    return payload


def inference_worker(args: argparse.Namespace) -> None:
    if args.baseline not in BASELINES:
        raise ValueError(f"unsupported baseline: {args.baseline}")
    subset_path = Path(args.subset_manifest).expanduser().resolve(strict=True)
    subset = _subset(subset_path)
    rows = _selected_rows(subset, args.object)
    output = Path(args.output_root).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    subset_sha = sha256_file(subset_path)
    snapshot = _model_snapshot(args.pretrained, args.model_revision)
    pipeline_json = snapshot / "pipeline.json"
    pipeline_sha = sha256_file(pipeline_json)
    from trellis.pipelines import TrellisImageTo3DPipeline

    pipeline = TrellisImageTo3DPipeline.from_pretrained(str(snapshot))
    device = torch.device(args.device)
    pipeline.to(device)
    mode = BASELINES[str(args.baseline)]
    reports: list[dict[str, Any]] = []
    try:
        for position, row in enumerate(rows, 1):
            mesh_path, result_path = _result_paths(output, row, int(args.seed))
            reused = _valid_result(
                result_path,
                mesh_path,
                row=row,
                args=args,
                subset_sha=subset_sha,
                pipeline_sha=pipeline_sha,
            )
            if reused is not None:
                reports.append(reused)
                print(f"[{args.baseline}] {position}/{len(rows)} reused object={row['category']}:{row['uid']}", flush=True)
                continue
            if mesh_path.parent.exists():
                raise RuntimeError(f"partial TRELLIS baseline output: {mesh_path.parent}")
            images, image_bindings = _images(row)
            outputs = pipeline.run_multi_image(
                images,
                num_samples=1,
                seed=int(args.seed),
                sparse_structure_sampler_params={
                    "steps": int(args.ss_steps),
                    "cfg_strength": float(args.ss_cfg),
                },
                slat_sampler_params={
                    "steps": int(args.slat_steps),
                    "cfg_strength": float(args.slat_cfg),
                },
                formats=["mesh"],
                preprocess_image=True,
                mode=mode,
            )
            decoded = outputs["mesh"][0]
            mesh = decoded.to_trimesh(transform_pose=False)
            structure = mesh_structure_metrics(mesh)
            if not structure["mesh_success"]:
                raise RuntimeError(f"TRELLIS decoded empty Mesh: {row['category']}:{row['uid']}")
            mesh_path.parent.mkdir(parents=True, exist_ok=False)
            temporary = mesh_path.with_name(f".{mesh_path.name}.tmp-{os.getpid()}")
            mesh.export(temporary, file_type="obj")
            os.replace(temporary, mesh_path)
            result = {
                "format": RESULT_FORMAT,
                "created_at_utc": utc_now(),
                "passed": True,
                "baseline": str(args.baseline),
                "multi_image_mode": mode,
                "object_key": f"{row['category']}:{row['uid']}",
                "category": str(row["category"]),
                "object_id": str(row["uid"]),
                "seed": int(args.seed),
                "input_view_indices": EXPECTED_VIEWS,
                "input_images": image_bindings,
                "native_trellis_preprocess": True,
                "subset_manifest": str(subset_path),
                "subset_manifest_sha256": subset_sha,
                "subset_identity": subset["subset_identity"],
                "pretrained": str(args.pretrained),
                "model_revision": str(args.model_revision),
                "pipeline_json": str(pipeline_json),
                "pipeline_json_sha256": pipeline_sha,
                "sampling": _sampling(args),
                "mesh": str(mesh_path),
                "mesh_sha256": sha256_file(mesh_path),
                "structure": structure,
                "output_frame": "decoder-native sparse-latent/model-O; transform_pose=False",
                "camera_pose_consumed": False,
                "target_or_metric_consumed": False,
            }
            atomic_json(result_path, result)
            reports.append(result)
            print(f"[{args.baseline}] {position}/{len(rows)} object={row['category']}:{row['uid']} seed={args.seed}", flush=True)
            del images, outputs, decoded, mesh
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
    finally:
        del pipeline
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    print(json.dumps({"passed": len(reports) == len(rows), "baseline": args.baseline, "objects": len(reports)}, indent=2))


def _scan_results(args: argparse.Namespace, subset: dict[str, Any]) -> dict[str, tuple[dict[str, Any], Path]]:
    root = Path(args.output_root).expanduser().resolve()
    expected = {f"{row['category']}:{row['uid']}" for row in subset["objects"]}
    found: dict[str, tuple[dict[str, Any], Path]] = {}
    for path in sorted(root.glob(f"**/seed_{int(args.seed)}/result.json")):
        payload = load_json(path)
        if payload.get("format") != RESULT_FORMAT or payload.get("passed") is not True:
            continue
        if payload.get("baseline") != args.baseline or payload.get("multi_image_mode") != BASELINES[args.baseline]:
            continue
        key = str(payload.get("object_key", ""))
        if key not in expected:
            continue
        mesh = Path(str(payload["mesh"])).resolve(strict=True)
        if sha256_file(mesh) != str(payload.get("mesh_sha256")):
            raise RuntimeError(f"TRELLIS Mesh hash differs: {mesh}")
        if key in found:
            raise RuntimeError(f"duplicate TRELLIS baseline output for {key}")
        found[key] = (payload, path)
    return found


def plan(args: argparse.Namespace) -> None:
    subset = _subset(Path(args.subset_manifest).expanduser().resolve(strict=True))
    found = _scan_results(args, subset)
    expected = [f"{row['category']}:{row['uid']}" for row in subset["objects"]]
    missing = [key for key in expected if key not in found]
    root = Path(args.plan_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    for worker in range(int(args.worker_count)):
        values = missing[worker :: int(args.worker_count)]
        (root / f"{args.stage}_worker{worker}.txt").write_text(
            "\n".join(values) + ("\n" if values else ""), encoding="utf-8"
        )
    atomic_json(root / f"{args.stage}_summary.json", {
        "baseline": args.baseline,
        "stage": args.stage,
        "complete": len(found),
        "missing": len(missing),
        "missing_keys": missing,
    })
    print({"baseline": args.baseline, "complete": len(found), "missing": len(missing)})


def aggregate_inference(args: argparse.Namespace) -> None:
    subset_path = Path(args.subset_manifest).expanduser().resolve(strict=True)
    subset = _subset(subset_path)
    found = _scan_results(args, subset)
    expected = [f"{row['category']}:{row['uid']}" for row in subset["objects"]]
    missing = [key for key in expected if key not in found]
    if missing or len(found) != 299:
        raise RuntimeError(f"TRELLIS {args.baseline} coverage differs: complete={len(found)} missing={missing}")
    objects = [found[key][0] for key in expected]
    output = Path(args.output).expanduser().resolve()
    payload = {
        "format": INFERENCE_FORMAT,
        "created_at_utc": utc_now(),
        "passed": True,
        "baseline": args.baseline,
        "multi_image_mode": BASELINES[args.baseline],
        "method": (
            "TRELLIS-S official stochastic four-image conditioning"
            if args.baseline == "trellis_s"
            else "TRELLIS-M official multidiffusion four-image conditioning"
        ),
        "pretrained": args.pretrained,
        "model_revision": args.model_revision,
        "sampling": _sampling(args),
        "seed": int(args.seed),
        "object_count": 299,
        "record_count": 299,
        "input_views_per_object": 4,
        "selected_input_view_indices": EXPECTED_VIEWS,
        "subset_manifest": str(subset_path),
        "subset_manifest_sha256": sha256_file(subset_path),
        "subset_identity": subset["subset_identity"],
        "objects": objects,
        "metric_or_target_consumed_during_inference": False,
        "output_frame": "decoder-native sparse-latent/model-O; transform_pose=False",
    }
    atomic_json(output, payload)
    print({"passed": True, "baseline": args.baseline, "objects": 299, "output": str(output)})


def _inference(path: Path, subset: dict[str, Any], baseline: str, seed: int) -> dict[str, Any]:
    payload = load_json(path)
    rows = list(payload.get("objects") or [])
    expected = [f"{row['category']}:{row['uid']}" for row in subset["objects"]]
    observed = [str(row.get("object_key")) for row in rows]
    if (
        payload.get("format") != INFERENCE_FORMAT
        or payload.get("passed") is not True
        or payload.get("baseline") != baseline
        or payload.get("multi_image_mode") != BASELINES[baseline]
        or payload.get("subset_identity") != subset["subset_identity"]
        or int(payload.get("seed", -1)) != int(seed)
        or observed != expected
    ):
        raise RuntimeError(f"TRELLIS inference identity differs: {path}")
    return payload


def metric_worker(args: argparse.Namespace) -> None:
    subset_path = Path(args.subset_manifest).expanduser().resolve(strict=True)
    inference_path = Path(args.inference_aggregate).expanduser().resolve(strict=True)
    subset = _subset(subset_path)
    inference = _inference(inference_path, subset, args.baseline, int(args.seed))
    source_by_uid = {str(row["uid"]): row for row in subset["objects"]}
    records = [row for index, row in enumerate(inference["objects"]) if index % int(args.worker_count) == int(args.worker_index)]
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    subset_sha = sha256_file(subset_path)
    inference_sha = sha256_file(inference_path)
    results = []
    for position, record in enumerate(records, 1):
        uid = str(record["object_id"])
        source = source_by_uid[uid]
        mesh_path = Path(record["mesh"]).resolve(strict=True)
        destination = output / "objects" / uid / "metric.json"
        if destination.is_file():
            cached = load_json(destination)
            if (
                cached.get("format") == METRIC_OBJECT_FORMAT
                and cached.get("passed") is True
                and cached.get("baseline") == args.baseline
                and cached.get("predicted_mesh_sha256") == sha256_file(mesh_path)
                and cached.get("subset_manifest_sha256") == subset_sha
                and cached.get("inference_aggregate_sha256") == inference_sha
            ):
                results.append(cached)
                continue
            raise RuntimeError(f"stale TRELLIS metric artifact: {destination}")
        predicted = _one_mesh(mesh_path)
        target = _gt_mesh(source)
        uid_seed = int.from_bytes(hashlib.sha256(uid.encode("utf-8")).digest()[:8], "big")
        metric_seed = (int(args.seed) * 1_000_003 + uid_seed) % (2**63 - 1)
        metrics, normalization = _metrics(
            predicted,
            target,
            count=int(args.surface_points),
            seed=metric_seed,
            radius=float(args.fscore_radius),
        )
        result = {
            "format": METRIC_OBJECT_FORMAT,
            "passed": True,
            "baseline": args.baseline,
            "uid": uid,
            "category": str(source["category"]),
            "seed": int(args.seed),
            "subset_manifest_sha256": subset_sha,
            "inference_aggregate_sha256": inference_sha,
            "predicted_mesh": str(mesh_path),
            "predicted_mesh_sha256": sha256_file(mesh_path),
            "gt_mesh": str(Path(source["source_mesh"]).resolve(strict=True)),
            "gt_scan_tree_sha256": source["source_scan_tree_sha256"],
            "surface_points": int(args.surface_points),
            "fscore_radius": float(args.fscore_radius),
            "metric_seed": metric_seed,
            "normalization": normalization,
            "metrics": metrics,
            "alignment": "decoder-native/model-O identity; independent AABB normalization; no GT fit/ICP",
        }
        atomic_json(destination, result)
        results.append(result)
        print(f"[{args.baseline}:metric] {position}/{len(records)} uid={uid} CD={metrics['chamfer_distance']:.8f} F={metrics['fscore']:.8f}", flush=True)
    atomic_json(output / "metrics_report.json", {
        "format": METRIC_WORKER_FORMAT,
        "passed": len(results) == len(records),
        "baseline": args.baseline,
        "worker_index": int(args.worker_index),
        "worker_count": int(args.worker_count),
        "object_count": len(results),
        "subset_manifest_sha256": subset_sha,
        "inference_aggregate_sha256": inference_sha,
        "objects": results,
    })


def aggregate_metrics(args: argparse.Namespace) -> None:
    subset_path = Path(args.subset_manifest).expanduser().resolve(strict=True)
    inference_path = Path(args.inference_aggregate).expanduser().resolve(strict=True)
    current_path = Path(args.current_report).expanduser().resolve(strict=True)
    subset = _subset(subset_path)
    _inference(inference_path, subset, args.baseline, int(args.seed))
    subset_sha = sha256_file(subset_path)
    inference_sha = sha256_file(inference_path)
    reports = sorted(Path(args.workers_root).expanduser().resolve(strict=True).glob("worker_*/metrics_report.json"))
    if len(reports) != int(args.expected_workers):
        raise RuntimeError(f"TRELLIS metric worker count differs: {len(reports)}")
    rows: list[dict[str, Any]] = []
    for path in reports:
        report = load_json(path)
        if (
            report.get("format") != METRIC_WORKER_FORMAT
            or report.get("passed") is not True
            or report.get("baseline") != args.baseline
            or report.get("subset_manifest_sha256") != subset_sha
            or report.get("inference_aggregate_sha256") != inference_sha
        ):
            raise RuntimeError(f"TRELLIS metric worker identity differs: {path}")
        rows.extend(report["objects"])
    expected = {str(row["uid"]) for row in subset["objects"]}
    observed = [str(row["uid"]) for row in rows]
    if len(rows) != 299 or len(set(observed)) != 299 or set(observed) != expected:
        raise RuntimeError("TRELLIS metric workers do not exactly cover Dora299")
    rows.sort(key=lambda row: str(row["uid"]))
    by_level = {}
    for level in sorted({str(row["category"]) for row in rows}):
        selected = [row for row in rows if row["category"] == level]
        by_level[level] = {
            "object_count": len(selected),
            "chamfer_distance": _summary([float(row["metrics"]["chamfer_distance"]) for row in selected]),
            "fscore": _summary([float(row["metrics"]["fscore"]) for row in selected]),
        }
    current = load_json(current_path)
    if (
        current.get("format") != "reconviagen.dorabench_dora300_ss30k_slat30k_metric_aggregate_failure_aware.v1"
        or current.get("passed") is not True
        or int(current.get("surface_metric_object_count", -1)) != 299
        or current.get("benchmark_manifest_sha256") != subset["parent_benchmark_manifest"]["sha256"]
    ):
        raise RuntimeError("bound current Dora299 report identity differs")
    current_by_uid = {str(row["uid"]): row for row in current["objects"]}
    if set(current_by_uid) != expected:
        raise RuntimeError("current metric object set differs from Dora299")
    baseline_by_uid = {str(row["uid"]): row for row in rows}
    cd_gain = [float(baseline_by_uid[uid]["metrics"]["chamfer_distance"]) - float(current_by_uid[uid]["metrics"]["chamfer_distance"]) for uid in sorted(expected)]
    f_gain = [float(current_by_uid[uid]["metrics"]["fscore"]) - float(baseline_by_uid[uid]["metrics"]["fscore"]) for uid in sorted(expected)]
    report = {
        "format": METRIC_AGGREGATE_FORMAT,
        "passed": True,
        "baseline": args.baseline,
        "multi_image_mode": BASELINES[args.baseline],
        "subset_manifest": str(subset_path),
        "subset_manifest_sha256": subset_sha,
        "subset_identity": subset["subset_identity"],
        "inference_aggregate": str(inference_path),
        "inference_aggregate_sha256": inference_sha,
        "object_count": 299,
        "surface_points_per_mesh": int(args.surface_points),
        "fscore_radius": float(args.fscore_radius),
        "chamfer_distance": _summary([float(row["metrics"]["chamfer_distance"]) for row in rows]),
        "fscore": _summary([float(row["metrics"]["fscore"]) for row in rows]),
        "by_complexity_level": by_level,
        "objects": rows,
        "paired_current_ss30k_slat30k": {
            "report": str(current_path),
            "report_sha256": sha256_file(current_path),
            "chamfer_improvement_current_over_baseline": {**_summary(cd_gain), "positive_rate": float(np.mean(np.asarray(cd_gain) > 0.0))},
            "fscore_delta_current_minus_baseline": {**_summary(f_gain), "positive_rate": float(np.mean(np.asarray(f_gain) > 0.0))},
        },
        "coordinate_contract": {
            "prediction": "decoder-native sparse-latent/model-O; transform_pose=False",
            "decoder_to_model_o": np.eye(4).tolist(),
            "normalization": "independent AABB center/max-extent to [-1,1]",
            "gt_fitted": False,
            "forbidden": ["rotation fit", "ICP", "reflection", "best-of-transform"],
        },
        "metric_stage_loaded_model": False,
    }
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "report.json", report)
    lines = [
        f"Dora-Bench current-valid 299 {args.baseline.upper()} evaluation",
        "========================================================",
        "objects: 299",
        f"Chamfer Distance: mean={report['chamfer_distance']['mean']:.8f} median={report['chamfer_distance']['median']:.8f}",
        f"F-score@0.1: mean={report['fscore']['mean']:.8f} median={report['fscore']['median']:.8f}",
        f"Current-over-baseline CD improvement: mean={report['paired_current_ss30k_slat30k']['chamfer_improvement_current_over_baseline']['mean']:+.8f}",
        f"Current-minus-baseline F-score: mean={report['paired_current_ss30k_slat30k']['fscore_delta_current_minus_baseline']['mean']:+.8f}",
        "Alignment: decoder-native/model-O identity; no GT fit/ICP",
        f"report: {output / 'report.json'}",
    ]
    (output / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--subset_manifest", required=True)
    parser.add_argument("--baseline", choices=sorted(BASELINES), required=True)
    parser.add_argument("--pretrained", default="microsoft/TRELLIS-image-large")
    parser.add_argument("--model_revision", default="25e0d31ffbebe4b5a97464dd851910efc3002d96")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ss_steps", type=int, default=30)
    parser.add_argument("--ss_cfg", type=float, default=7.5)
    parser.add_argument("--slat_steps", type=int, default=12)
    parser.add_argument("--slat_cfg", type=float, default=3.0)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("inference-worker")
    _common(p)
    p.add_argument("--output_root", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--object", action="append")
    p.set_defaults(func=inference_worker)
    p = sub.add_parser("plan")
    _common(p)
    p.add_argument("--output_root", required=True)
    p.add_argument("--plan_root", required=True)
    p.add_argument("--stage", required=True)
    p.add_argument("--worker_count", type=int, required=True)
    p.set_defaults(func=plan)
    p = sub.add_parser("aggregate-inference")
    _common(p)
    p.add_argument("--output_root", required=True)
    p.add_argument("--output", required=True)
    p.set_defaults(func=aggregate_inference)
    p = sub.add_parser("metric-worker")
    p.add_argument("--subset_manifest", required=True)
    p.add_argument("--inference_aggregate", required=True)
    p.add_argument("--baseline", choices=sorted(BASELINES), required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--worker_index", type=int, required=True)
    p.add_argument("--worker_count", type=int, required=True)
    p.add_argument("--surface_points", type=int, default=100000)
    p.add_argument("--fscore_radius", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=metric_worker)
    p = sub.add_parser("aggregate-metrics")
    p.add_argument("--subset_manifest", required=True)
    p.add_argument("--inference_aggregate", required=True)
    p.add_argument("--current_report", required=True)
    p.add_argument("--workers_root", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--baseline", choices=sorted(BASELINES), required=True)
    p.add_argument("--expected_workers", type=int, required=True)
    p.add_argument("--surface_points", type=int, default=100000)
    p.add_argument("--fscore_radius", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=aggregate_metrics)
    return parser


def main() -> None:
    args = make_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
