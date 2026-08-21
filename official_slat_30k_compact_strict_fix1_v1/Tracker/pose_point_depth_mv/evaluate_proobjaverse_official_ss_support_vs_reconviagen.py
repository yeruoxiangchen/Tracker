#!/usr/bin/env python3
"""Compare official Native-SS and strict ReconViaGen Stock-SS supports.

This evaluator intentionally stops before SLat.  The GPU worker replays only

    frozen RGBA views -> VGGT -> Stock SS -> decoded 64^3 coordinates

and saves the coordinates.  The CPU aggregate then compares those coordinates
and the already frozen official Native-SS coordinates against the same official
GT-SLat coordinate support.

The two methods use their frozen deployment interfaces and sampling recipes.
Consequently this is an external support-quality comparison, not a same-noise
network-only ablation.  Object metrics are first averaged over seeds 42/43/44
before bootstrap confidence intervals are computed.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("SPCONV_ALGO", "native")

import numpy as np
import torch

from pose_point_depth_mv.direct_slat_data import DirectSLatCacheDataset
from pose_point_depth_mv.eval_direct_flow import component_metrics
from pose_point_depth_mv.evaluate_native_ss_stock_slat_mesh import summary_with_ci
from pose_point_depth_mv.evaluate_proobjaverse_official_native_ss_stock_slat import (
    END_TO_END_WORKER_FORMAT,
    pair_id,
)
from pose_point_depth_mv import evaluate_proobjaverse_official_reconviagen as strict_rg
from pose_point_depth_mv.proobjaverse_official_slat_protocol import (
    atomic_json,
    canonical_sha256,
    load_json,
    sha256_file,
)


WORKER_FORMAT = (
    "pose_point_depth_mv.proobjaverse_official_strict_reconviagen_ss_support_worker.v1"
)
REPORT_FORMAT = (
    "pose_point_depth_mv.proobjaverse_official_native_ss_vs_strict_reconviagen_ss_support.v1"
)
SEEDS = (42, 43, 44)
GRID_RESOLUTION = 64
STRICT_SS_EXECUTION = "tracker_reconviagen_pipeline_run_ss_only.v1"
QUALITY_METRICS = (
    "iou",
    "precision",
    "recall",
    "f1",
    "coord_count_ratio",
    "count_abs_log_error",
    "component_count",
    "component_count_abs_error",
    "largest_component_ratio",
    "largest_component_ratio_abs_error",
)
IMPROVEMENT_METRICS = (
    "iou",
    "precision",
    "recall",
    "f1",
    "count_abs_log_error",
    "component_count_abs_error",
    "largest_component_ratio_abs_error",
)


def parse_csv(value: str, cast: Any = str) -> list[Any]:
    values = [cast(item.strip()) for item in str(value).split(",") if item.strip()]
    if not values or len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("CSV values must be non-empty and unique")
    return values


def _verify_internal_hash(payload: dict[str, Any], path: Path) -> None:
    saved = payload.get("report_sha256")
    if saved is None:
        raise RuntimeError(f"report lacks internal SHA256: {path}")
    body = dict(payload)
    body.pop("report_sha256")
    if canonical_sha256(body) != str(saved):
        raise RuntimeError(f"report internal SHA256 differs: {path}")


def _target_contract(
    slat_manifest: str | Path,
    strict_contract: dict[str, Any] | None = None,
) -> tuple[DirectSLatCacheDataset, dict[str, Any]]:
    dataset = DirectSLatCacheDataset(slat_manifest, indices="all")
    target = dict(dataset.config.get("target_source", {}))
    required = {
        "split": "dev",
        "support_policy": "official_gt_slat_coordinates",
        "coordinate_resolution": GRID_RESOLUTION,
    }
    mismatch = {
        key: (target.get(key), expected)
        for key, expected in required.items()
        if target.get(key) != expected
    }
    if mismatch:
        raise RuntimeError(f"SLat manifest is not official Dev64 GT support: {mismatch}")
    protocol = str(target.get("protocol_sha256", ""))
    if not protocol:
        raise RuntimeError("official target support protocol hash is missing")
    if strict_contract is not None and protocol != str(strict_contract["protocol_sha256"]):
        raise RuntimeError("strict ReconViaGen and GT-support protocol hashes differ")
    if len(dataset.rows) != 64:
        raise RuntimeError(f"official Dev support must contain 64 objects, got {len(dataset.rows)}")
    uids = [str(row["object_uid"]) for row in dataset.rows]
    if len(set(uids)) != 64:
        raise RuntimeError("official Dev support contains duplicate object identities")
    if strict_contract is not None:
        strict_uids = [str(row["uid"]) for row in strict_contract["rows"]]
        if uids != strict_uids:
            raise RuntimeError("strict ReconViaGen and GT-support UID ordering differs")
    return dataset, target


def _target_path(dataset: DirectSLatCacheDataset, row: dict[str, Any]) -> Path:
    path = Path(row["target_file"])
    return path if path.is_absolute() else dataset.root / path


def _load_target_coords(
    dataset: DirectSLatCacheDataset, row: dict[str, Any]
) -> tuple[np.ndarray, Path]:
    path = _target_path(dataset, row).resolve(strict=True)
    if sha256_file(path) != str(row["target_file_sha256"]):
        raise RuntimeError(f"official target support changed: {path}")
    with np.load(path) as payload:
        coords3 = np.ascontiguousarray(payload["coords"], dtype=np.int32)
    coords = np.concatenate(
        [np.zeros((len(coords3), 1), dtype=np.int32), coords3], axis=1
    )
    _coordinate_contract(coords, label=f"target:{row['object_uid']}")
    return coords, path


def _coordinate_contract(coords: np.ndarray, *, label: str) -> dict[str, Any]:
    array = np.asarray(coords)
    shape_ok = array.ndim == 2 and array.shape[1] == 4
    integer_ok = np.issubdtype(array.dtype, np.integer)
    if not shape_ok or not integer_ok:
        raise RuntimeError(
            f"{label} coordinates must be integer Nx4, got shape={array.shape} dtype={array.dtype}"
        )
    batch_ok = bool(len(array) == 0 or np.all(array[:, 0] == 0))
    range_ok = bool(
        len(array) == 0
        or np.all((array[:, 1:] >= 0) & (array[:, 1:] < GRID_RESOLUTION))
    )
    unique_count = int(len(np.unique(array, axis=0)))
    unique_ok = unique_count == len(array)
    result = {
        "shape_nx4": shape_ok,
        "integer_dtype": integer_ok,
        "single_zero_batch": batch_ok,
        "coordinate_range_0_63": range_ok,
        "unique_coordinates": unique_ok,
        "nonempty": bool(len(array) > 0),
        "coord_count": int(len(array)),
        "unique_coord_count": unique_count,
        "resolution": GRID_RESOLUTION,
        "passed": bool(batch_ok and range_ok and unique_ok and len(array) > 0),
    }
    if not result["passed"]:
        raise RuntimeError(f"{label} coordinate contract failed: {result}")
    return result


def _coord_set(coords: np.ndarray) -> set[tuple[int, int, int]]:
    return {tuple(int(value) for value in row[-3:]) for row in np.asarray(coords)}


def support_quality(predicted: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    pred = _coord_set(predicted)
    gt = _coord_set(target)
    intersection = len(pred & gt)
    union = len(pred | gt)
    precision = intersection / max(len(pred), 1)
    recall = intersection / max(len(gt), 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    pred_components = component_metrics(np.asarray(predicted, dtype=np.int32))
    target_components = component_metrics(np.asarray(target, dtype=np.int32))
    ratio = len(pred) / max(len(gt), 1)
    return {
        "coord_count": int(len(pred)),
        "target_coord_count": int(len(gt)),
        "intersection_count": int(intersection),
        "union_count": int(union),
        "false_positive_count": int(len(pred) - intersection),
        "false_negative_count": int(len(gt) - intersection),
        "iou": float(intersection / max(union, 1)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "coord_count_ratio": float(ratio),
        "count_abs_log_error": float(abs(np.log(max(ratio, 1e-12)))),
        "component_count": int(pred_components["component_count"]),
        "target_component_count": int(target_components["component_count"]),
        "component_count_abs_error": float(
            abs(
                int(pred_components["component_count"])
                - int(target_components["component_count"])
            )
        ),
        "largest_component_ratio": float(
            pred_components["largest_component_ratio"]
        ),
        "target_largest_component_ratio": float(
            target_components["largest_component_ratio"]
        ),
        "largest_component_ratio_abs_error": float(
            abs(
                float(pred_components["largest_component_ratio"])
                - float(target_components["largest_component_ratio"])
            )
        ),
    }


def _pair_overlap(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    lhs = _coord_set(left)
    rhs = _coord_set(right)
    intersection = len(lhs & rhs)
    union = len(lhs | rhs)
    return {
        "intersection_count": int(intersection),
        "union_count": int(union),
        "iou": float(intersection / max(union, 1)),
        "native_to_strict_count_ratio": float(len(lhs) / max(len(rhs), 1)),
    }


def _strict_sampling(args: argparse.Namespace) -> dict[str, Any]:
    namespace = argparse.Namespace(
        ss_steps=int(args.ss_steps),
        ss_guidance=float(args.ss_guidance),
        ss_guidance_rescale=float(args.ss_guidance_rescale),
        ss_rescale_t=float(args.ss_rescale_t),
        slat_steps=12,
        slat_guidance=7.5,
        slat_guidance_rescale=0.5,
        slat_rescale_t=3.0,
        multiimage_algo="multidiffusion",
    )
    return strict_rg._sampling(namespace)["sparse_structure"]


def _load_historical_strict_records(
    reports_csv: str,
) -> tuple[list[dict[str, Any]], dict[tuple[str, int], dict[str, Any]]]:
    paths = parse_csv(reports_csv, str)
    bindings, records = strict_rg._load_recon_reports(paths)
    return bindings, records


@torch.no_grad()
def _sample_strict_stock_support(
    pipeline: Any,
    images: list[Any],
    *,
    seed: int,
    sampler_params: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    # Call the original Tracker/ReconViaGen pipeline directly.  Its explicit
    # early-return flag stops after decoded Stock-SS coordinates, while the
    # entire prefix (including CPU noise generation followed by .to(device))
    # remains byte-for-byte shared with the full pipeline.
    outputs, coords, noise = pipeline.run(
        image=images,
        seed=int(seed),
        formats=["mesh"],
        preprocess_image=False,
        sparse_structure_sampler_params=sampler_params,
        slat_sampler_params={},
        mode="multidiffusion",
        return_ss_support_only=True,
    )
    if outputs is not None:
        raise RuntimeError("ReconViaGen SS-only branch unexpectedly decoded SLat")
    noise_sha256 = hashlib.sha256(
        noise.detach().float().cpu().numpy().tobytes()
    ).hexdigest()
    coords = (
        coords.int()
        .cpu()
        .numpy()
        .astype(np.int32)
    )
    audit = {
        "seed": int(seed),
        "ss_noise_shape": list(noise.shape),
        "ss_noise_sha256_float32": noise_sha256,
        "reconviagen_original_run_ss_only": True,
        "vggt_model_executed": True,
        "stock_ss_flow_executed": True,
        "slat_executed": False,
        "mesh_decoder_executed": False,
    }
    del noise
    return coords, audit


def _npz_atomic(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp-{os.getpid()}.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def _record_path(output: Path, uid: str, seed: int) -> Path:
    return output / "records" / uid / f"seed_{int(seed)}.json"


def _coords_path(output: Path, uid: str, seed: int) -> Path:
    return output / "coords" / f"{pair_id(uid, seed)}.npz"


def _load_reusable(
    path: Path, expected: dict[str, Any], *, resume: bool
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    if not resume:
        raise FileExistsError(path)
    row = load_json(path)
    mismatch = {
        key: (row.get(key), value)
        for key, value in expected.items()
        if row.get(key) != value
    }
    coords_path = Path(str(row.get("coords_npz", "")))
    if not coords_path.is_file():
        mismatch["coords_npz"] = (str(coords_path), "existing file")
    elif sha256_file(coords_path) != str(row.get("coords_npz_sha256", "")):
        mismatch["coords_npz_sha256"] = (
            row.get("coords_npz_sha256"),
            sha256_file(coords_path),
        )
    if row.get("passed") is not True:
        mismatch["passed"] = (row.get("passed"), True)
    if mismatch:
        raise RuntimeError(f"stale strict SS support record: {path}: {mismatch}")
    return row


def _worker_contract(args: argparse.Namespace) -> tuple[
    dict[str, Any], DirectSLatCacheDataset, dict[str, Any], list[dict[str, Any]], dict[tuple[str, int], dict[str, Any]]
]:
    strict_contract = strict_rg._load_contract(args)
    dataset, target_contract = _target_contract(args.slat_manifest, strict_contract)
    historical_bindings, historical = _load_historical_strict_records(
        args.strict_recon_reports
    )
    expected_pairs = {
        (str(row["uid"]), seed)
        for row in strict_contract["rows"]
        for seed in SEEDS
    }
    if set(historical) != expected_pairs:
        raise RuntimeError("historical strict ReconViaGen reports do not cover Dev64x3")
    return strict_contract, dataset, target_contract, historical_bindings, historical


def run_worker(args: argparse.Namespace) -> None:
    seeds = tuple(parse_csv(args.seeds, int))
    if seeds != SEEDS:
        raise RuntimeError("support comparison requires seeds 42,43,44")
    if int(args.num_workers) <= 0 or not 0 <= int(args.worker_index) < int(args.num_workers):
        raise ValueError("worker_index must be in [0,num_workers)")
    (
        strict_contract,
        dataset,
        target_contract,
        historical_bindings,
        historical,
    ) = _worker_contract(args)
    start = int(args.object_start)
    end = 64 if int(args.object_end) <= 0 else int(args.object_end)
    if start < 0 or end <= start or end > 64:
        raise ValueError(f"invalid object slice [{start}:{end}]")
    selected = [
        row
        for row in strict_contract["rows"]
        if start <= int(row["index"]) < end
        and int(row["index"]) % int(args.num_workers) == int(args.worker_index)
    ]
    if not selected:
        raise RuntimeError("strict SS worker shard contains no objects")
    sampler = _strict_sampling(args)
    for key, row in historical.items():
        if row.get("sampling", {}).get("sparse_structure") != sampler:
            raise RuntimeError(f"strict historical/current SS sampler differs: {key}")
    dry = {
        "passed": True,
        "dry_run": True,
        "worker_index": int(args.worker_index),
        "num_workers": int(args.num_workers),
        "object_indices": [int(row["index"]) for row in selected],
        "object_count": len(selected),
        "record_count": len(selected) * len(SEEDS),
        "pipeline": "VGGT -> Stock SS -> decoded support; STOP before SLat",
        "official_protocol_sha256": str(target_contract["protocol_sha256"]),
    }
    if args.dry_run:
        print(json.dumps(dry, indent=2, ensure_ascii=False))
        return

    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    pipeline = strict_rg._build_pipeline(
        args.pretrained, device, bool(args.low_vram)
    )
    target_by_uid = {
        str(row["object_uid"]): row for row in dataset.rows
    }
    records: list[dict[str, Any]] = []
    try:
        for position, row in enumerate(selected, start=1):
            uid = str(row["uid"])
            target_row = target_by_uid[uid]
            target_coords, target_path = _load_target_coords(dataset, target_row)
            images, input_sha, render_audit = strict_rg._selected_images(row, pipeline)
            if images is None or len(images) != 8:
                raise RuntimeError(f"uid={uid} does not provide eight strict RGBA views")
            for seed in SEEDS:
                coords_path = _coords_path(output, uid, seed).resolve()
                expected = {
                    "format": WORKER_FORMAT,
                    "object_index": int(row["index"]),
                    "object_uid": uid,
                    "seed": int(seed),
                    "strict_ss_execution": STRICT_SS_EXECUTION,
                    "selected_input_sha256": input_sha,
                    "official_protocol_sha256": str(target_contract["protocol_sha256"]),
                    "strict_ss_sampler": sampler,
                    "target_file_sha256": str(target_row["target_file_sha256"]),
                    "pretrained": str(args.pretrained),
                }
                record_path = _record_path(output, uid, seed)
                reused = _load_reusable(
                    record_path, expected, resume=bool(args.resume)
                )
                if reused is not None:
                    records.append(reused)
                    continue
                coords, runtime = _sample_strict_stock_support(
                    pipeline,
                    images,
                    seed=seed,
                    sampler_params=sampler,
                )
                contract = _coordinate_contract(
                    coords, label=f"strict_stock_ss:{uid}:{seed}"
                )
                quality = support_quality(coords, target_coords)
                historical_row = historical[(uid, seed)]
                historical_count = (
                    int(historical_row["coord_count"])
                    if historical_row.get("passed") is True
                    else None
                )
                replay_match = (
                    None
                    if historical_count is None
                    else int(len(coords)) == historical_count
                )
                if replay_match is False:
                    raise RuntimeError(
                        f"strict Stock SS coordinate count replay differs uid={uid} "
                        f"seed={seed}: {len(coords)} != {historical_count}"
                    )
                _npz_atomic(coords_path, strict_stock=coords.astype(np.int32))
                record = {
                    **expected,
                    "created_at_utc": datetime.now(timezone.utc).isoformat(),
                    "render_tar": str(row["render_tar"]),
                    "render_archive_audit": render_audit,
                    "target_file": str(target_path),
                    "coords_npz": str(coords_path),
                    "coords_npz_sha256": sha256_file(coords_path),
                    "coordinate_contract": contract,
                    "quality_vs_official_gt_support": quality,
                    "historical_strict_full_record_passed": bool(
                        historical_row.get("passed") is True
                    ),
                    "historical_strict_full_coord_count": historical_count,
                    "historical_coord_count_replay_match": replay_match,
                    "runtime": runtime,
                    "passed": True,
                }
                atomic_json(record_path, record)
                records.append(record)
                print(
                    f"[strict_rg_ss_support] {position}/{len(selected)} "
                    f"uid={uid} seed={seed} count={len(coords)} replay={replay_match}",
                    flush=True,
                )
                del coords
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            del images, target_coords
    finally:
        del pipeline
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    records.sort(key=lambda row: (int(row["object_index"]), int(row["seed"])))
    expected_count = len(selected) * len(SEEDS)
    replay_checked = [
        row for row in records if row["historical_coord_count_replay_match"] is not None
    ]
    report = {
        "format": WORKER_FORMAT,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "complete": len(records) == expected_count,
        "passed": bool(
            len(records) == expected_count
            and all(row.get("passed") is True for row in records)
            and all(row["historical_coord_count_replay_match"] is True for row in replay_checked)
        ),
        "worker_index": int(args.worker_index),
        "num_workers": int(args.num_workers),
        "object_start": start,
        "object_end": end,
        "object_indices": [int(row["index"]) for row in selected],
        "object_uids": [str(row["uid"]) for row in selected],
        "object_count": len(selected),
        "record_count": len(records),
        "seeds": list(SEEDS),
        "official_protocol_sha256": str(target_contract["protocol_sha256"]),
        "slat_manifest": str(Path(args.slat_manifest).resolve()),
        "slat_manifest_sha256": sha256_file(args.slat_manifest),
        "dev_split_sha256": strict_contract["split_sha256"],
        "cache_report_sha256": strict_contract["cache_report_sha256"],
        "pretrained": str(args.pretrained),
        "strict_ss_execution": STRICT_SS_EXECUTION,
        "strict_ss_sampler": sampler,
        "pipeline": "frozen RGBA views -> VGGT -> Stock SS -> decoded 64^3 support",
        "vggt_model_executed": True,
        "stock_ss_executed": True,
        "slat_executed": False,
        "mesh_decoder_executed": False,
        "historical_coord_count_replay": {
            "checked_records": len(replay_checked),
            "matched_records": sum(
                row["historical_coord_count_replay_match"] is True
                for row in replay_checked
            ),
        },
        "historical_strict_reports": historical_bindings,
        "records": records,
    }
    report["report_sha256"] = canonical_sha256(report)
    atomic_json(output / "report.json", report)
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "objects": report["object_count"],
                "records": report["record_count"],
                "replay": report["historical_coord_count_replay"],
                "report": str(output / "report.json"),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    if not report["passed"]:
        raise SystemExit(2)


def _load_strict_support_reports(
    paths: list[str], *, expected_protocol: str
) -> tuple[dict[tuple[str, int], dict[str, Any]], list[dict[str, Any]]]:
    records: dict[tuple[str, int], dict[str, Any]] = {}
    bindings = []
    workers = set()
    shared = None
    for value in paths:
        path = Path(value).expanduser().resolve(strict=True)
        report = load_json(path)
        _verify_internal_hash(report, path)
        if (
            report.get("format") != WORKER_FORMAT
            or report.get("complete") is not True
            or report.get("passed") is not True
        ):
            raise RuntimeError(f"invalid strict SS support worker report: {path}")
        worker = (int(report["worker_index"]), int(report["num_workers"]))
        if worker in workers:
            raise RuntimeError(f"duplicate strict SS worker: {worker}")
        workers.add(worker)
        identity = {
            key: report[key]
            for key in (
                "object_start",
                "object_end",
                "seeds",
                "official_protocol_sha256",
                "slat_manifest_sha256",
                "dev_split_sha256",
                "cache_report_sha256",
                "pretrained",
                "strict_ss_execution",
                "strict_ss_sampler",
                "pipeline",
                "vggt_model_executed",
                "stock_ss_executed",
                "slat_executed",
                "mesh_decoder_executed",
            )
        }
        if shared is None:
            shared = identity
        elif shared != identity:
            raise RuntimeError("strict SS support worker identities differ")
        if str(report["official_protocol_sha256"]) != expected_protocol:
            raise RuntimeError("strict SS worker protocol differs from GT support")
        for row in report["records"]:
            key = (str(row["object_uid"]), int(row["seed"]))
            if key in records:
                raise RuntimeError(f"duplicate strict SS support record: {key}")
            coords_path = Path(row["coords_npz"]).resolve(strict=True)
            if sha256_file(coords_path) != str(row["coords_npz_sha256"]):
                raise RuntimeError(f"strict SS coordinate file changed: {coords_path}")
            records[key] = {**row, "_coords_path": str(coords_path)}
        bindings.append({"path": str(path), "sha256": sha256_file(path), "worker": worker})
    worker_count = next(iter(workers))[1] if workers else 0
    if workers != {(index, worker_count) for index in range(worker_count)}:
        raise RuntimeError("strict SS support worker set is incomplete")
    return records, sorted(bindings, key=lambda row: row["worker"])


def _load_native_support_reports(
    paths: list[str], *, expected_protocol: str
) -> tuple[dict[tuple[str, int], dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    records: dict[tuple[str, int], dict[str, Any]] = {}
    bindings = []
    shared = None
    for value in paths:
        path = Path(value).expanduser().resolve(strict=True)
        report = load_json(path)
        _verify_internal_hash(report, path)
        if report.get("format") != END_TO_END_WORKER_FORMAT or report.get("complete") is not True:
            raise RuntimeError(f"invalid official Native-SS worker report: {path}")
        identity = dict(report.get("run_identity", {}))
        required = {
            "official_protocol_sha256": expected_protocol,
            "joint_seeds": list(SEEDS),
            "weights": "ema",
        }
        mismatch = {
            key: (identity.get(key), expected)
            for key, expected in required.items()
            if identity.get(key) != expected
        }
        if mismatch:
            raise RuntimeError(f"Native-SS worker identity differs: {mismatch}")
        comparable = {
            key: value
            for key, value in identity.items()
            if key not in {"object_start", "object_end", "object_uids"}
        }
        if shared is None:
            shared = comparable
        elif shared != comparable:
            raise RuntimeError("Native-SS worker identities differ")
        coord_root = path.parent / "ss_coords"
        for row in report.get("ss_records", []):
            key = (str(row["object_uid"]), int(row["seed"]))
            if key in records:
                raise RuntimeError(f"duplicate Native-SS support record: {key}")
            coords_path = (coord_root / f"{pair_id(*key)}.npz").resolve(strict=True)
            if sha256_file(coords_path) != str(row["coords_npz_sha256"]):
                raise RuntimeError(f"Native-SS coordinate file changed: {coords_path}")
            records[key] = {**row, "_coords_path": str(coords_path)}
        bindings.append({"path": str(path), "sha256": sha256_file(path)})
    if shared is None:
        raise RuntimeError("no Native-SS worker reports")
    return records, bindings, shared


def _load_coords(path: str | Path, key: str) -> np.ndarray:
    with np.load(Path(path)) as payload:
        coords = np.ascontiguousarray(payload[key], dtype=np.int32)
    _coordinate_contract(coords, label=f"{key}:{path}")
    return coords


def _summarize_absolute(
    rows: dict[str, dict[str, float]], *, samples: int, seed: int
) -> dict[str, Any]:
    return {
        metric: summary_with_ci(
            [float(row[metric]) for row in rows.values()],
            samples=samples,
            seed=seed + position,
        )
        for position, metric in enumerate(QUALITY_METRICS)
    }


def _metric_improvement(metric: str, native: float, strict: float) -> float:
    if metric.endswith("_abs_error") or metric == "count_abs_log_error":
        return float(strict - native)
    return float(native - strict)


def run_aggregate(args: argparse.Namespace) -> None:
    strict_contract = strict_rg._load_contract(args)
    dataset, target_contract = _target_contract(args.slat_manifest, strict_contract)
    protocol = str(target_contract["protocol_sha256"])
    strict_records, strict_bindings = _load_strict_support_reports(
        parse_csv(args.strict_support_reports, str), expected_protocol=protocol
    )
    native_records, native_bindings, native_identity = _load_native_support_reports(
        parse_csv(args.native_reports, str), expected_protocol=protocol
    )
    start = int(args.object_start)
    end = 64 if int(args.object_end) <= 0 else int(args.object_end)
    if start < 0 or end <= start or end > 64:
        raise ValueError(f"invalid aggregate object slice [{start}:{end}]")
    selected_rows = dataset.rows[start:end]
    selected_uids = [str(row["object_uid"]) for row in selected_rows]
    expected = {(uid, seed) for uid in selected_uids for seed in SEEDS}
    if set(strict_records) != expected:
        raise RuntimeError(
            f"strict SS support coverage differs: missing={len(expected-set(strict_records))} "
            f"extra={len(set(strict_records)-expected)}"
        )
    if set(native_records) != expected:
        raise RuntimeError(
            f"Native-SS support coverage differs: missing={len(expected-set(native_records))} "
            f"extra={len(set(native_records)-expected)}"
        )

    target_by_uid = {str(row["object_uid"]): row for row in selected_rows}
    seed_rows = []
    for uid in selected_uids:
        target, _ = _load_target_coords(dataset, target_by_uid[uid])
        for seed in SEEDS:
            key = (uid, seed)
            strict_coords = _load_coords(strict_records[key]["_coords_path"], "strict_stock")
            native_coords = _load_coords(native_records[key]["_coords_path"], "native")
            strict_quality = support_quality(strict_coords, target)
            native_quality = support_quality(native_coords, target)
            seed_rows.append(
                {
                    "object_uid": uid,
                    "seed": seed,
                    "strict_reconviagen_stock_ss": strict_quality,
                    "official_native_ss": native_quality,
                    "native_minus_strict_improvement": {
                        metric: _metric_improvement(
                            metric,
                            float(native_quality[metric]),
                            float(strict_quality[metric]),
                        )
                        for metric in IMPROVEMENT_METRICS
                    },
                    "native_strict_support_overlap": _pair_overlap(
                        native_coords, strict_coords
                    ),
                }
            )

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in seed_rows:
        grouped[str(row["object_uid"])].append(row)
    object_rows = []
    strict_means: dict[str, dict[str, float]] = {}
    native_means: dict[str, dict[str, float]] = {}
    for uid in selected_uids:
        rows = sorted(grouped[uid], key=lambda row: int(row["seed"]))
        if [int(row["seed"]) for row in rows] != list(SEEDS):
            raise RuntimeError(f"object {uid} does not contain three fixed seeds")
        strict_mean = {
            metric: float(
                np.mean(
                    [float(row["strict_reconviagen_stock_ss"][metric]) for row in rows]
                )
            )
            for metric in QUALITY_METRICS
        }
        native_mean = {
            metric: float(
                np.mean([float(row["official_native_ss"][metric]) for row in rows])
            )
            for metric in QUALITY_METRICS
        }
        improvements = {
            metric: _metric_improvement(
                metric, native_mean[metric], strict_mean[metric]
            )
            for metric in IMPROVEMENT_METRICS
        }
        overlap = {
            name: float(
                np.mean([float(row["native_strict_support_overlap"][name]) for row in rows])
            )
            for name in ("iou", "native_to_strict_count_ratio")
        }
        strict_means[uid] = strict_mean
        native_means[uid] = native_mean
        object_rows.append(
            {
                "object_uid": uid,
                "strict_reconviagen_stock_ss": strict_mean,
                "official_native_ss": native_mean,
                "native_minus_strict_improvement": improvements,
                "native_strict_support_overlap": overlap,
            }
        )

    samples = int(args.bootstrap_samples)
    delta_summary = {
        metric: summary_with_ci(
            [
                float(row["native_minus_strict_improvement"][metric])
                for row in object_rows
            ],
            samples=samples,
            seed=2026081600 + position,
        )
        for position, metric in enumerate(IMPROVEMENT_METRICS)
    }
    overlap_summary = {
        metric: summary_with_ci(
            [float(row["native_strict_support_overlap"][metric]) for row in object_rows],
            samples=samples,
            seed=2026081700 + position,
        )
        for position, metric in enumerate(("iou", "native_to_strict_count_ratio"))
    }
    iou = delta_summary["iou"]
    precision = delta_summary["precision"]
    recall = delta_summary["recall"]
    runtime_passed = bool(
        len(seed_rows) == len(selected_uids) * len(SEEDS)
        and all(
            row["coordinate_contract"]["passed"] is True
            for row in strict_records.values()
        )
        and all(row.get("passed") is True for row in native_records.values())
    )
    checks = {
        "runtime_integrity": runtime_passed,
        "exact_object_seed_coverage": len(seed_rows)
        == len(selected_uids) * len(SEEDS),
        "iou_mean_positive": float(iou["mean"]) > 0.0,
        "iou_median_positive": float(iou["median"]) > 0.0,
        "iou_object_win_rate_gt_half": float(iou["positive_rate"]) > 0.5,
        "iou_bootstrap_ci_lower_positive": float(iou["bootstrap_mean_95_ci"][0]) > 0.0,
        "precision_mean_nonnegative": float(precision["mean"]) >= 0.0,
        "recall_mean_nonnegative": float(recall["mean"]) >= 0.0,
    }
    advantage = bool(all(checks.values()))
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "format": REPORT_FORMAT,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "passed": runtime_passed,
        "formal": False,
        "object_start": start,
        "object_end": end,
        "object_count": len(selected_uids),
        "record_count": len(seed_rows),
        "seeds": list(SEEDS),
        "official_protocol_sha256": protocol,
        "strict_reconviagen_stock_ss_absolute": _summarize_absolute(
            strict_means, samples=samples, seed=2026081800
        ),
        "official_native_ss_absolute": _summarize_absolute(
            native_means, samples=samples, seed=2026081900
        ),
        "native_minus_strict_improvement": delta_summary,
        "native_strict_support_distribution_overlap": overlap_summary,
        "decision": {
            "native_ss_support_advantage_established": advantage,
            "checks": checks,
            "interpretation": (
                "Official posed Native-SS establishes a held-out support-quality advantage over strict ReconViaGen VGGT Stock-SS."
                if advantage
                else "At least one held-out support-quality gate has not passed; inspect precision/recall/count/connectivity before any SLat compatibility test."
            ),
        },
        "comparability": {
            "same_objects": True,
            "same_selected_eight_source_views": True,
            "same_official_gt_slat_coordinate_support": True,
            "same_grid_resolution": GRID_RESOLUTION,
            "same_seed_labels": True,
            "same_initial_noise_tensor": False,
            "strict_interface": "frozen RGBA -> VGGT -> Stock SS",
            "native_interface": "frozen posed-DINO evidence -> official Native-SS step2000 EMA",
            "component_attribution": (
                "external SS endpoint comparison; it includes the intended VGGT-vs-known-pose input-interface difference and must not be described as a same-condition architecture-only ablation"
            ),
            "slat_executed": False,
            "mesh_decoder_executed": False,
        },
        "strict_support_worker_reports": strict_bindings,
        "native_support_source_reports": native_bindings,
        "native_support_identity": native_identity,
        "slat_manifest": str(Path(args.slat_manifest).resolve()),
        "slat_manifest_sha256": sha256_file(args.slat_manifest),
        "per_object": object_rows,
        "per_seed": seed_rows,
        "scope_guard": (
            "Official ProObjaverse held-out Dev[16:64) support-only development comparison. No SLat or Mesh decoder is executed. The result decides SS support quality, not Stock-SLat compatibility and not final Mesh superiority."
        ),
    }
    report["report_sha256"] = canonical_sha256(report)
    atomic_json(output / "report.json", report)
    lines = [
        "Official Native-SS vs strict ReconViaGen Stock-SS: support only",
        "=" * 70,
        f"objects: {len(selected_uids)} records: {len(seed_rows)} seeds: {list(SEEDS)}",
        "strict: frozen RGBA -> VGGT -> Stock SS",
        "native: frozen posed-DINO -> official Native-SS step2000 EMA",
        "SLat executed: no; Mesh decoder executed: no",
        f"IoU improvement: {delta_summary['iou']}",
        f"precision improvement: {delta_summary['precision']}",
        f"recall improvement: {delta_summary['recall']}",
        f"F1 improvement: {delta_summary['f1']}",
        f"count log-error improvement: {delta_summary['count_abs_log_error']}",
        f"component-count error improvement: {delta_summary['component_count_abs_error']}",
        f"LCR error improvement: {delta_summary['largest_component_ratio_abs_error']}",
        f"native/strict distribution overlap: {overlap_summary}",
        f"checks: {checks}",
        f"native_ss_support_advantage_established: {advantage}",
        report["scope_guard"],
        f"report: {output / 'report.json'}",
    ]
    (output / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    raise SystemExit(0 if advantage else 3)


def add_common_contract_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dev_split", required=True)
    parser.add_argument("--cache_report", required=True)
    parser.add_argument("--target_report", required=True)
    parser.add_argument("--target_mesh_root", required=True)
    parser.add_argument("--paired_target_cache_roots", default="")
    parser.add_argument("--slat_manifest", required=True)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    worker = commands.add_parser("worker")
    add_common_contract_paths(worker)
    worker.add_argument("--strict_recon_reports", required=True)
    worker.add_argument("--output_dir", required=True)
    worker.add_argument("--pretrained", default="Stable-X/trellis-vggt-v0-2")
    worker.add_argument("--seeds", default="42,43,44")
    worker.add_argument("--device", default="cuda:0")
    worker.add_argument("--low_vram", action="store_true")
    worker.add_argument("--ss_steps", type=int, default=30)
    worker.add_argument("--ss_guidance", type=float, default=7.5)
    worker.add_argument("--ss_guidance_rescale", type=float, default=0.7)
    worker.add_argument("--ss_rescale_t", type=float, default=5.0)
    worker.add_argument("--object_start", type=int, default=16)
    worker.add_argument("--object_end", type=int, default=64)
    worker.add_argument("--worker_index", type=int, default=0)
    worker.add_argument("--num_workers", type=int, default=1)
    worker.add_argument("--resume", action="store_true")
    worker.add_argument("--dry_run", action="store_true")

    aggregate = commands.add_parser("aggregate")
    add_common_contract_paths(aggregate)
    aggregate.add_argument("--strict_support_reports", required=True)
    aggregate.add_argument("--native_reports", required=True)
    aggregate.add_argument("--output_dir", required=True)
    aggregate.add_argument("--object_start", type=int, default=16)
    aggregate.add_argument("--object_end", type=int, default=64)
    aggregate.add_argument("--bootstrap_samples", type=int, default=5000)
    return parser


def main() -> None:
    args = make_parser().parse_args()
    if args.command == "worker":
        run_worker(args)
    else:
        run_aggregate(args)


if __name__ == "__main__":
    main()
