#!/usr/bin/env python3
"""Evaluate the original ReconViaGen full pipeline on official ProObjaverse Dev64.

The worker executes VGGT -> stock SS -> stock SLat -> stock Mesh decoder from
``Stable-X/trellis-vggt-v0-2``.  It computes metrics in memory and deliberately
does not save predicted meshes or rendered previews.  The aggregate command
also supports a paired comparison with the existing official Native-SS +
trained-SLat endpoint reports.
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
from typing import Any, Iterable

import numpy as np
from PIL import Image
import torch
import trimesh

from pose_point_depth_mv.evaluate_objaverse16_no_vggt import load_mesh
from pose_point_depth_mv.infer_objaverse16_reconviagen import _build_pipeline
from pose_point_depth_mv.mesh_benchmark_metrics import (
    mesh_structure_metrics,
    surface_metrics,
)
from pose_point_depth_mv.prepare_proobjaverse_official_slat_dino_cache import (
    _load_views_with_audit,
)
from pose_point_depth_mv.proobjaverse_official_slat_protocol import (
    SPLIT_FORMAT,
    atomic_json,
    canonical_sha256,
    load_json,
    sha256_file,
)


RECORD_FORMAT = "pose_point_depth_mv.proobjaverse_official_reconviagen_metric.v1"
WORKER_FORMAT = "pose_point_depth_mv.proobjaverse_official_reconviagen_worker.v1"
REPORT_FORMAT = "pose_point_depth_mv.proobjaverse_official_reconviagen_vs_current.v1"
CURRENT_WORKER_FORMAT = (
    "pose_point_depth_mv.proobjaverse_official_native_ss_slat_end_to_end_worker.v1"
)
RECON_METHOD = "reconviagen_original_vggt_stock_ss_stock_slat"
CURRENT_METHOD = "official_native_ss_step2000_slat_step25000"
LOWER_IS_BETTER = {
    "pred_to_gt_mean",
    "gt_to_pred_mean",
    "chamfer_l1",
    "chamfer_l2",
}
TERMINAL_RECON_FAILURE_KINDS = frozenset(
    {
        "spconv_int32_range_exceeded",
        "flexicubes_topology_invalid",
        "decoded_mesh_invalid",
    }
)


def parse_csv(value: str, cast: Any = str) -> list[Any]:
    result = [cast(item.strip()) for item in str(value).split(",") if item.strip()]
    if not result:
        raise argparse.ArgumentTypeError("CSV value must be non-empty")
    return result


def summarize(values: Iterable[float]) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=np.float64)
    if not len(array) or not np.isfinite(array).all():
        raise ValueError("metric distribution is empty or non-finite")
    return {
        "count": int(len(array)),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "min": float(array.min()),
        "max": float(array.max()),
        "positive_rate": float(np.mean(array > 0.0)),
        "nonnegative_rate": float(np.mean(array >= 0.0)),
    }


def bootstrap_mean_ci(values: list[float], *, samples: int, seed: int) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        raise ValueError("cannot bootstrap an empty distribution")
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, len(array), size=(int(samples), len(array)))
    means = array[indices].mean(axis=1)
    return [float(value) for value in np.quantile(means, [0.025, 0.975])]


def _verify_internal_hash(payload: dict[str, Any], *, path: Path) -> None:
    if "report_sha256" not in payload:
        return
    body = dict(payload)
    saved = str(body.pop("report_sha256"))
    if canonical_sha256(body) != saved:
        raise RuntimeError(f"internal report SHA256 differs: {path}")


def _load_contract(args: argparse.Namespace) -> dict[str, Any]:
    split_path = Path(args.dev_split).expanduser().resolve(strict=True)
    cache_report_path = Path(args.cache_report).expanduser().resolve(strict=True)
    target_report_path = Path(args.target_report).expanduser().resolve(strict=True)
    target_root = Path(args.target_mesh_root).expanduser().resolve(strict=True)
    target_cache_roots = [
        Path(value).expanduser().resolve(strict=True)
        for value in parse_csv(
            getattr(args, "paired_target_cache_roots", ""), str
        )
    ] if getattr(args, "paired_target_cache_roots", "") else []
    split = load_json(split_path)
    cache_report = load_json(cache_report_path)
    target_report = load_json(target_report_path)
    _verify_internal_hash(cache_report, path=cache_report_path)
    _verify_internal_hash(target_report, path=target_report_path)

    if (
        split.get("format") != SPLIT_FORMAT
        or split.get("name") != "dev"
        or int(split.get("count", -1)) != 64
        or len(split.get("rows", [])) != 64
    ):
        raise RuntimeError("dev_split is not the frozen official Dev64 split")
    required_cache = {
        "passed": True,
        "split": "dev",
        "object_count": 64,
        "selected_views": 8,
        "native_ss_executed": False,
        "vggt_executed": False,
    }
    mismatch = {
        key: (cache_report.get(key), expected)
        for key, expected in required_cache.items()
        if cache_report.get(key) != expected
    }
    if mismatch:
        raise RuntimeError(f"official Dev64 cache report differs: {mismatch}")
    if len(cache_report.get("records", [])) != 64:
        raise RuntimeError("official Dev64 cache report does not contain 64 records")
    target_config = dict(target_report.get("run_config", {}))
    if (
        target_report.get("passed") is not True
        or target_config.get("official_split") != "dev"
        or target_config.get("target")
        != "frozen Stock decoder applied to official SLat label"
        or target_config.get("official_protocol_sha256")
        != cache_report.get("protocol_sha256")
    ):
        raise RuntimeError("target report is not the passed official Dev64 target run")

    split_rows = list(split["rows"])
    cache_rows = list(cache_report["records"])
    split_uids = [str(row["uid"]) for row in split_rows]
    cache_uids = [str(row["uid"]) for row in cache_rows]
    target_uids = [str(value) for value in target_config.get("object_uids", [])]
    if split_uids != cache_uids or split_uids != target_uids:
        raise RuntimeError("split/cache/target Dev64 UID ordering differs")
    if len(split_uids) != len(set(split_uids)):
        raise RuntimeError("official Dev64 contains duplicate UIDs")

    rows = []
    for index, (split_row, cache_row) in enumerate(zip(split_rows, cache_rows)):
        uid = str(split_row["uid"])
        render_tar = Path(split_row["render_tar"]).expanduser().resolve(strict=True)
        if render_tar.stat().st_size != int(split_row["render_size"]):
            raise RuntimeError(f"official render tar size changed: {render_tar}")
        selected_view_ids = [int(value) for value in cache_row["selected_view_ids"]]
        if len(selected_view_ids) != 8 or len(set(selected_view_ids)) != 8:
            raise RuntimeError(f"uid={uid} does not bind eight unique selected views")
        target_mesh = target_root / uid / "decoded_official_target.obj"
        target_source = "official_target_obj_from_frozen_stock_decoder"
        if index >= 16 and target_cache_roots:
            matches = [root / f"{uid}.npz" for root in target_cache_roots]
            matches = [path for path in matches if path.is_file()]
            if len(matches) != 1:
                raise RuntimeError(
                    f"uid={uid} expected exactly one paired target cache; "
                    f"found={matches}"
                )
            target_mesh = matches[0]
            target_source = "exact_target_npz_used_by_current_endpoint_worker"
        if not target_mesh.is_file():
            raise FileNotFoundError(target_mesh)
        rows.append(
            {
                "index": index,
                "uid": uid,
                "render_tar": render_tar,
                "selected_view_ids": selected_view_ids,
                "target_mesh": target_mesh,
                "target_source": target_source,
            }
        )
    return {
        "rows": rows,
        "split_path": split_path,
        "split_sha256": sha256_file(split_path),
        "cache_report_path": cache_report_path,
        "cache_report_sha256": sha256_file(cache_report_path),
        "target_report_path": target_report_path,
        "target_report_sha256": sha256_file(target_report_path),
        "protocol_sha256": str(cache_report["protocol_sha256"]),
        "paired_target_cache_roots": [str(path) for path in target_cache_roots],
    }


def _sampling(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "sparse_structure": {
            "steps": int(args.ss_steps),
            "cfg_strength": float(args.ss_guidance),
            "cfg_interval": [0.6, 1.0],
            "guidance_rescale": float(args.ss_guidance_rescale),
            "rescale_t": float(args.ss_rescale_t),
        },
        "slat": {
            "steps": int(args.slat_steps),
            "cfg_strength": float(args.slat_guidance),
            "cfg_interval": [0.6, 1.0],
            "guidance_rescale": float(args.slat_guidance_rescale),
            "rescale_t": float(args.slat_rescale_t),
        },
        "multiimage_algo": str(args.multiimage_algo),
    }


def _selected_images(
    row: dict[str, Any], pipeline: Any | None
) -> tuple[list[Image.Image] | None, str, dict[str, Any]]:
    views, audit = _load_views_with_audit(row["render_tar"], row["uid"])
    by_id = {int(view["id"]): view for view in views}
    if len(by_id) != len(views):
        raise RuntimeError(f"duplicate official view IDs: {row['render_tar']}")
    missing = [value for value in row["selected_view_ids"] if value not in by_id]
    if missing:
        raise RuntimeError(f"selected official views missing uid={row['uid']}: {missing}")
    identities = []
    images = [] if pipeline is not None else None
    for view_id in row["selected_view_ids"]:
        rgba = np.ascontiguousarray(by_id[view_id]["rgba"], dtype=np.uint8)
        identities.append(
            {
                "view_id": int(view_id),
                "shape": list(rgba.shape),
                "rgba_sha256": hashlib.sha256(rgba.tobytes()).hexdigest(),
            }
        )
        if images is not None:
            image = Image.fromarray(rgba, mode="RGBA")
            images.append(pipeline.preprocess_image(image).convert("RGB"))
    return images, canonical_sha256(identities), audit


def _load_target_mesh(path: Path) -> trimesh.Trimesh:
    if path.suffix == ".npz":
        with np.load(path) as payload:
            mesh = trimesh.Trimesh(
                vertices=np.asarray(payload["vertices"]),
                faces=np.asarray(payload["faces"]),
                process=False,
            )
        if not len(mesh.vertices) or not len(mesh.faces):
            raise ValueError(f"cached official target Mesh is empty: {path}")
        return mesh
    return load_mesh(path)


def _record_path(output: Path, uid: str, seed: int) -> Path:
    return output / "records" / uid / f"seed_{int(seed)}.json"


def _spconv_algo() -> str:
    return str(os.environ.get("SPCONV_ALGO", "auto"))


def _classify_terminal_recon_failure(error: Exception) -> dict[str, Any] | None:
    """Classify deterministic model-output/decoder failures only.

    Unknown exceptions, CUDA OOM, driver failures, and implementation errors are
    deliberately not recordable: those must still abort the worker instead of
    being silently converted into scientific model failures.
    """

    if not isinstance(error, RuntimeError):
        return None
    message = str(error)
    if "your data exceed int32 range" in message:
        kind = "spconv_int32_range_exceeded"
        stage = "stock_mesh_decoder_sparse_convolution"
    elif message.startswith("FlexiCubes topology index is inconsistent:"):
        kind = "flexicubes_topology_invalid"
        stage = "stock_mesh_decoder_flexicubes"
    elif message.startswith("ReconViaGen decoded invalid Mesh"):
        kind = "decoded_mesh_invalid"
        stage = "stock_mesh_decoder_output_validation"
    else:
        return None
    return {
        "type": type(error).__name__,
        "kind": kind,
        "message": message,
        "stage": stage,
        "retryable": False,
        "spconv_algo": _spconv_algo(),
    }


def _is_terminal_recon_failure_record(record: dict[str, Any]) -> bool:
    error = record.get("error")
    return bool(
        record.get("passed") is False
        and isinstance(error, dict)
        and error.get("kind") in TERMINAL_RECON_FAILURE_KINDS
        and error.get("retryable") is False
        and str(error.get("spconv_algo")) == str(record.get("spconv_algo"))
    )


def _complete_object_uids(
    records: dict[tuple[str, int], dict[str, Any]], objects: list[str]
) -> list[str]:
    return [
        uid
        for uid in objects
        if all(
            (uid, seed) in records and records[(uid, seed)].get("passed") is True
            for seed in (42, 43, 44)
        )
    ]


def _record_identity(
    *,
    row: dict[str, Any],
    seed: int,
    contract: dict[str, Any],
    render_tar_sha256: str,
    selected_input_sha256: str,
    target_mesh_sha256: str,
    pretrained: str,
    sampling: dict[str, Any],
    surface_samples: int,
) -> dict[str, Any]:
    return {
        "format": RECORD_FORMAT,
        "method": RECON_METHOD,
        "object_index": int(row["index"]),
        "uid": str(row["uid"]),
        "object_uid": str(row["uid"]),
        "seed": int(seed),
        "selected_view_ids": list(row["selected_view_ids"]),
        "selected_input_sha256": selected_input_sha256,
        "render_tar_sha256": render_tar_sha256,
        "target_mesh_sha256": target_mesh_sha256,
        "target_source": str(row["target_source"]),
        "dev_split_sha256": contract["split_sha256"],
        "cache_report_sha256": contract["cache_report_sha256"],
        "target_report_sha256": contract["target_report_sha256"],
        "official_protocol_sha256": contract["protocol_sha256"],
        "pretrained": str(pretrained),
        "sampling_sha256": canonical_sha256(sampling),
        "surface_samples": int(surface_samples),
        "fscore_thresholds": [0.01, 0.02, 0.05],
        "metric_seed": int(seed) * 1009 + int(row["index"]) * 9173,
    }


def _load_reusable(path: Path, expected: dict[str, Any], *, resume: bool) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    if not resume:
        raise FileExistsError(path)
    record = load_json(path)
    mismatch = {
        key: (record.get(key), value)
        for key, value in expected.items()
        if record.get(key) != value
    }
    # All records produced before this guard were created by the frozen launcher,
    # which explicitly used SPCONV_ALGO=native.  Treat a missing legacy binding as
    # native, but never reuse it under another convolution backend.
    saved_spconv_algo = str(record.get("spconv_algo", "native"))
    if saved_spconv_algo != _spconv_algo():
        mismatch["spconv_algo"] = (saved_spconv_algo, _spconv_algo())
    if record.get("passed") is not True and not _is_terminal_recon_failure_record(record):
        mismatch["terminal_failure"] = (record.get("error"), "approved terminal failure")
    if mismatch:
        raise RuntimeError(f"stale ReconViaGen metric record={mismatch}: {path}")
    return record


def run_worker(args: argparse.Namespace) -> None:
    if int(args.num_workers) <= 0 or not 0 <= int(args.worker_index) < int(args.num_workers):
        raise ValueError("worker_index must be in [0, num_workers)")
    seeds = parse_csv(args.seeds, int)
    if seeds != [42, 43, 44]:
        raise RuntimeError("official comparison requires seeds 42,43,44")
    contract = _load_contract(args)
    selected = [
        row
        for row in contract["rows"]
        if int(row["index"]) % int(args.num_workers) == int(args.worker_index)
    ]
    if not selected:
        raise RuntimeError("worker shard contains no Dev64 objects")
    if args.dry_run:
        print(
            json.dumps(
                {
                    "passed": True,
                    "dry_run": True,
                    "objects": len(selected),
                    "records": len(selected) * len(seeds),
                    "object_indices": [row["index"] for row in selected],
                    "predicted_meshes_saved": False,
                    "render_previews_saved": False,
                },
                indent=2,
            )
        )
        return

    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    sampling = _sampling(args)
    device = torch.device(args.device)
    pipeline = None
    records: list[dict[str, Any]] = []
    try:
        for local_position, row in enumerate(selected, start=1):
            render_tar_sha = sha256_file(row["render_tar"])
            _, selected_input_sha, view_audit = _selected_images(row, None)
            target_mesh_sha = sha256_file(row["target_mesh"])
            expected_by_seed = {
                seed: _record_identity(
                    row=row,
                    seed=seed,
                    contract=contract,
                    render_tar_sha256=render_tar_sha,
                    selected_input_sha256=selected_input_sha,
                    target_mesh_sha256=target_mesh_sha,
                    pretrained=args.pretrained,
                    sampling=sampling,
                    surface_samples=int(args.surface_samples),
                )
                for seed in seeds
            }
            pending = []
            for seed in seeds:
                record_path = _record_path(output, row["uid"], seed)
                reused = _load_reusable(
                    record_path, expected_by_seed[seed], resume=bool(args.resume)
                )
                if reused is None:
                    pending.append(seed)
                else:
                    records.append(reused)
            if not pending:
                print(
                    f"[official_reconviagen:reuse] {local_position}/{len(selected)} "
                    f"uid={row['uid']}",
                    flush=True,
                )
                continue
            if pipeline is None:
                pipeline = _build_pipeline(args.pretrained, device, bool(args.low_vram))
            images, checked_input_sha, checked_audit = _selected_images(row, pipeline)
            if checked_input_sha != selected_input_sha or checked_audit != view_audit:
                raise RuntimeError("official selected RGBA inputs changed during worker run")
            if images is None or len(images) != 8:
                raise RuntimeError("ReconViaGen did not receive exactly eight images")
            target_mesh = _load_target_mesh(row["target_mesh"])
            target_structure = mesh_structure_metrics(target_mesh)
            if not target_structure["mesh_success"]:
                raise RuntimeError(f"invalid official target Mesh uid={row['uid']}")
            for seed in pending:
                print(
                    f"[official_reconviagen] {local_position}/{len(selected)} "
                    f"uid={row['uid']} seed={seed} views=8",
                    flush=True,
                )
                outputs = coords = ss_noise = decoded = predicted_mesh = None
                record_common = {
                    **expected_by_seed[seed],
                    "created_at_utc": datetime.now(timezone.utc).isoformat(),
                    "render_tar": str(row["render_tar"]),
                    "target_mesh": str(row["target_mesh"]),
                    "render_archive_audit": view_audit,
                    "sampling": sampling,
                    "target_structure": target_structure,
                    "input_contract": (
                        "same eight frozen official RGBA views; ReconViaGen alpha-mask "
                        "1.10 foreground recenter/518 preprocessing"
                    ),
                    "pipeline_contract": (
                        "VGGT + stock sparse-structure flow + stock SLat flow + "
                        "stock Mesh decoder"
                    ),
                    "spconv_algo": _spconv_algo(),
                    "vggt_model_executed": True,
                    "explicit_camera_pose_consumed": False,
                    "target_or_metric_consumed_during_inference": False,
                    "output_frame": "latent decoder canonical; transform_pose=False",
                    "predicted_mesh_saved": False,
                    "render_preview_saved": False,
                }
                try:
                    if pipeline is None:
                        pipeline = _build_pipeline(
                            args.pretrained, device, bool(args.low_vram)
                        )
                    outputs, coords, ss_noise = pipeline.run(
                        image=images,
                        seed=int(seed),
                        formats=["mesh"],
                        preprocess_image=False,
                        sparse_structure_sampler_params=sampling["sparse_structure"],
                        slat_sampler_params=sampling["slat"],
                        mode=sampling["multiimage_algo"],
                    )
                    decoded = outputs["mesh"][0]
                    predicted_mesh = decoded.to_trimesh(transform_pose=False)
                    structure = mesh_structure_metrics(predicted_mesh)
                    if not structure["mesh_success"]:
                        raise RuntimeError(
                            f"ReconViaGen decoded invalid Mesh uid={row['uid']} "
                            f"seed={seed}"
                        )
                    surface = surface_metrics(
                        predicted_mesh,
                        target_mesh,
                        count=int(args.surface_samples),
                        seed=expected_by_seed[seed]["metric_seed"],
                        thresholds=(0.01, 0.02, 0.05),
                    )
                    record = {
                        **record_common,
                        "surface": surface,
                        "structure": structure,
                        "coord_count": int(coords.shape[0]),
                        "ss_noise_shape": list(ss_noise.shape),
                        "passed": True,
                    }
                except Exception as error:
                    failure = _classify_terminal_recon_failure(error)
                    if failure is None:
                        raise
                    record = {
                        **record_common,
                        "surface": None,
                        "structure": {"mesh_success": False},
                        "error": failure,
                        "passed": False,
                    }
                    print(
                        f"[official_reconviagen:recorded_failure] "
                        f"{local_position}/{len(selected)} uid={row['uid']} "
                        f"seed={seed} kind={failure['kind']}",
                        flush=True,
                    )
                    # Rebuild after a recorded decoder failure.  This preserves
                    # later seeds while ensuring large sparse workspaces and any
                    # cached indice pairs cannot leak into the next inference.
                    old_pipeline = pipeline
                    pipeline = None
                    del old_pipeline
                atomic_json(_record_path(output, row["uid"], seed), record)
                records.append(record)
                del outputs, coords, ss_noise, decoded, predicted_mesh
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            del images, target_mesh
            gc.collect()
    finally:
        if pipeline is not None:
            del pipeline
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    expected_count = len(selected) * len(seeds)
    records.sort(key=lambda row: (int(row["object_index"]), int(row["seed"])))
    successful_records = [row for row in records if row.get("passed") is True]
    failed_records = [row for row in records if row.get("passed") is False]
    complete = len(records) == expected_count
    report = {
        "format": WORKER_FORMAT,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "complete": complete,
        "passed": complete and not failed_records,
        "formal": False,
        "method": RECON_METHOD,
        "worker_index": int(args.worker_index),
        "num_workers": int(args.num_workers),
        "object_indices": [int(row["index"]) for row in selected],
        "object_uids": [str(row["uid"]) for row in selected],
        "object_count": len(selected),
        "record_count": len(records),
        "successful_record_count": len(successful_records),
        "failed_record_count": len(failed_records),
        "mesh_success_rate": (
            float(len(successful_records) / expected_count) if expected_count else 0.0
        ),
        "failed_records": [
            {
                "object_index": int(row["object_index"]),
                "object_uid": str(row["object_uid"]),
                "seed": int(row["seed"]),
                "error": row["error"],
            }
            for row in failed_records
        ],
        "seeds": seeds,
        "dev_split": str(contract["split_path"]),
        "dev_split_sha256": contract["split_sha256"],
        "cache_report": str(contract["cache_report_path"]),
        "cache_report_sha256": contract["cache_report_sha256"],
        "target_report": str(contract["target_report_path"]),
        "target_report_sha256": contract["target_report_sha256"],
        "paired_target_cache_roots": contract["paired_target_cache_roots"],
        "official_protocol_sha256": contract["protocol_sha256"],
        "pretrained": str(args.pretrained),
        "spconv_algo": _spconv_algo(),
        "sampling": sampling,
        "sampling_sha256": canonical_sha256(sampling),
        "surface_samples": int(args.surface_samples),
        "predicted_meshes_saved": False,
        "render_previews_saved": False,
        "records": records,
    }
    report["report_sha256"] = canonical_sha256(report)
    report_path = output / "report.json"
    atomic_json(report_path, report)
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "complete": report["complete"],
                "objects": report["object_count"],
                "records": report["record_count"],
                "successful_records": report["successful_record_count"],
                "failed_records": report["failed_record_count"],
                "report": str(report_path),
            },
            indent=2,
        )
    )
    if not report["complete"]:
        raise SystemExit(2)


def _load_recon_reports(paths: list[str]) -> tuple[list[dict[str, Any]], dict[tuple[str, int], dict[str, Any]]]:
    reports = []
    records: dict[tuple[str, int], dict[str, Any]] = {}
    workers = set()
    shared = None
    for value in paths:
        path = Path(value).expanduser().resolve(strict=True)
        report = load_json(path)
        _verify_internal_hash(report, path=path)
        legacy_complete = bool(
            report.get("passed") is True
            and int(report.get("record_count", -1))
            == int(report.get("object_count", -1)) * len(report.get("seeds", []))
        )
        if (
            report.get("format") != WORKER_FORMAT
            or not (report.get("complete") is True or legacy_complete)
        ):
            raise RuntimeError(f"invalid ReconViaGen worker report: {path}")
        worker = (int(report["worker_index"]), int(report["num_workers"]))
        if worker in workers:
            raise RuntimeError(f"duplicate ReconViaGen worker={worker}")
        workers.add(worker)
        identity = {
            key: report[key]
            for key in (
                "method",
                "seeds",
                "dev_split_sha256",
                "cache_report_sha256",
                "target_report_sha256",
                "paired_target_cache_roots",
                "official_protocol_sha256",
                "pretrained",
                "sampling",
                "sampling_sha256",
                "surface_samples",
                "predicted_meshes_saved",
                "render_previews_saved",
            )
        }
        identity["spconv_algo"] = str(report.get("spconv_algo", "native"))
        if shared is None:
            shared = identity
        elif identity != shared:
            raise RuntimeError("ReconViaGen worker identities differ")
        for row in report["records"]:
            key = (str(row["object_uid"]), int(row["seed"]))
            if key in records:
                raise RuntimeError(f"duplicate ReconViaGen record={key}")
            row_spconv_algo = str(row.get("spconv_algo", "native"))
            if row_spconv_algo != identity["spconv_algo"]:
                raise RuntimeError(
                    f"ReconViaGen record/report spconv backend differs={key}"
                )
            if row.get("passed") is not True and not _is_terminal_recon_failure_record(row):
                raise RuntimeError(f"unapproved failed ReconViaGen record={key}")
            records[key] = row
        successful_count = sum(
            row.get("passed") is True for row in report["records"]
        )
        reports.append(
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "worker_index": worker[0],
                "num_workers": worker[1],
                "spconv_algo": identity["spconv_algo"],
                "complete": True,
                "all_meshes_valid": successful_count == len(report["records"]),
                "successful_record_count": successful_count,
                "failed_record_count": len(report["records"]) - successful_count,
            }
        )
    if not workers:
        raise RuntimeError("no ReconViaGen worker reports")
    worker_count = next(iter(workers))[1]
    if workers != {(index, worker_count) for index in range(worker_count)}:
        raise RuntimeError("ReconViaGen worker set is incomplete")
    if len(records) != 64 * 3:
        raise RuntimeError("ReconViaGen Dev64 matrix is incomplete")
    return sorted(reports, key=lambda row: row["worker_index"]), records


def _load_current_reports(
    paths: list[str], *, expected_step: int, expected_sha256: str
) -> tuple[list[dict[str, Any]], dict[tuple[str, int], dict[str, Any]], list[dict[str, Any]]]:
    reports = []
    records: dict[tuple[str, int], dict[str, Any]] = {}
    failed = []
    observed_uids = []
    shared = None
    for value in paths:
        path = Path(value).expanduser().resolve(strict=True)
        report = load_json(path)
        _verify_internal_hash(report, path=path)
        if report.get("format") != CURRENT_WORKER_FORMAT or report.get("complete") is not True:
            raise RuntimeError(f"invalid current endpoint worker report: {path}")
        identity = dict(report["run_identity"])
        required = {
            "expected_trained_slat_step": int(expected_step),
            "trained_slat_checkpoint_sha256": str(expected_sha256),
            "joint_seeds": [42, 43, 44],
            "weights": "ema",
            "surface_samples": 20000,
            "save_meshes": False,
        }
        mismatch = {
            key: (identity.get(key), expected)
            for key, expected in required.items()
            if identity.get(key) != expected
        }
        if mismatch:
            raise RuntimeError(f"current endpoint identity differs: {mismatch}")
        comparable = {
            key: value
            for key, value in identity.items()
            if key not in {"object_start", "object_end", "object_uids"}
        }
        if shared is None:
            shared = comparable
        elif shared != comparable:
            raise RuntimeError("current endpoint worker identities differ")
        observed_uids.extend(str(uid) for uid in identity["object_uids"])
        for row in report["mesh_branch_records"]:
            if row.get("branch") != "native_trained":
                continue
            key = (str(row["object_uid"]), int(row["seed"]))
            if key in records:
                raise RuntimeError(f"duplicate current endpoint record={key}")
            if row.get("passed") is True:
                records[key] = row
            else:
                failed.append(
                    {
                        "object_uid": key[0],
                        "seed": key[1],
                        "error": row.get("error"),
                    }
                )
        reports.append({"path": str(path), "sha256": sha256_file(path)})
    if len(observed_uids) != 48 or len(set(observed_uids)) != 48:
        raise RuntimeError("current endpoint reports do not cover exactly Dev[16:64)")
    return reports, records, failed


def _object_means(
    records: dict[tuple[str, int], dict[str, Any]], objects: list[str]
) -> dict[str, dict[str, float]]:
    output = {}
    for uid in objects:
        rows = [records[(uid, seed)] for seed in (42, 43, 44)]
        names = sorted(rows[0]["surface"])
        output[uid] = {
            **{
                name: float(np.mean([float(row["surface"][name]) for row in rows]))
                for name in names
            },
            "largest_component_ratio": float(
                np.mean(
                    [float(row["structure"]["largest_component_ratio"]) for row in rows]
                )
            ),
        }
    return output


def _absolute_summary(means: dict[str, dict[str, float]]) -> dict[str, Any]:
    names = sorted(next(iter(means.values())))
    return {
        name: summarize(row[name] for row in means.values()) for name in names
    }


def _paired_summary(
    current: dict[str, dict[str, float]],
    recon: dict[str, dict[str, float]],
    *,
    bootstrap_samples: int,
) -> dict[str, Any]:
    if set(current) != set(recon) or not current:
        raise RuntimeError("paired complete-object sets differ or are empty")
    improvements: dict[str, list[float]] = defaultdict(list)
    per_object = {}
    for uid in sorted(current):
        per_object[uid] = {}
        for metric in sorted(current[uid]):
            delta = (
                recon[uid][metric] - current[uid][metric]
                if metric in LOWER_IS_BETTER
                else current[uid][metric] - recon[uid][metric]
            )
            improvements[metric].append(float(delta))
            per_object[uid][metric] = float(delta)
    metric_summary = {}
    for position, (metric, values) in enumerate(sorted(improvements.items())):
        metric_summary[f"{metric}_current_improvement"] = {
            **summarize(values),
            "object_bootstrap_mean_95_ci": bootstrap_mean_ci(
                values,
                samples=int(bootstrap_samples),
                seed=20260816 + position,
            ),
        }
    return {
        "left": CURRENT_METHOD,
        "right": RECON_METHOD,
        "positive_definition": "positive means the current endpoint is better",
        "unit_of_analysis": (
            "complete held-out Dev48 objects; each object metric is the mean of "
            "seeds 42/43/44"
        ),
        "object_count": len(current),
        "metric_deltas": metric_summary,
        "per_object_improvements": per_object,
    }


def run_aggregate(args: argparse.Namespace) -> None:
    contract = _load_contract(args)
    recon_bindings, recon_records = _load_recon_reports(
        parse_csv(args.recon_reports, str)
    )
    expected = {
        (str(row["uid"]), seed)
        for row in contract["rows"]
        for seed in (42, 43, 44)
    }
    if set(recon_records) != expected:
        raise RuntimeError("ReconViaGen records do not exactly cover Dev64 x three seeds")
    all_uids = [str(row["uid"]) for row in contract["rows"]]
    heldout_uids = all_uids[16:64]
    recon_valid_records = {
        key: row for key, row in recon_records.items() if row.get("passed") is True
    }
    recon_failures = [
        {
            "object_index": int(row["object_index"]),
            "object_uid": str(row["object_uid"]),
            "seed": int(row["seed"]),
            "error": row["error"],
        }
        for row in recon_records.values()
        if row.get("passed") is False
    ]
    recon_failures.sort(
        key=lambda row: (row["object_index"], row["seed"])
    )
    recon_all_complete_uids = _complete_object_uids(recon_records, all_uids)
    recon_heldout_complete_uids = _complete_object_uids(
        recon_records, heldout_uids
    )
    recon_all_incomplete_uids = [
        uid for uid in all_uids if uid not in set(recon_all_complete_uids)
    ]
    recon_heldout_incomplete_uids = [
        uid for uid in heldout_uids if uid not in set(recon_heldout_complete_uids)
    ]
    heldout_uid_set = set(heldout_uids)
    recon_heldout_failures = [
        row for row in recon_failures if row["object_uid"] in heldout_uid_set
    ]
    recon_heldout_successful_record_count = 48 * 3 - len(
        recon_heldout_failures
    )
    if not recon_all_complete_uids or not recon_heldout_complete_uids:
        raise RuntimeError("no complete ReconViaGen objects remain for surface metrics")
    recon_all_means = _object_means(
        recon_valid_records, recon_all_complete_uids
    )
    recon_heldout_means = _object_means(
        recon_valid_records, recon_heldout_complete_uids
    )

    current_bindings, current_records, current_failures = _load_current_reports(
        parse_csv(args.current_reports, str),
        expected_step=int(args.expected_current_step),
        expected_sha256=str(args.expected_current_sha256),
    )
    for key, current_row in current_records.items():
        if key not in recon_records:
            raise RuntimeError(f"current endpoint pair is outside strict Dev64={key}")
        if recon_records[key]["target_structure"] != current_row["target_structure"]:
            raise RuntimeError(
                f"strict/current target Mesh structure differs for pair={key}"
            )
    current_complete_uids = _complete_object_uids(current_records, heldout_uids)
    complete_uids = [
        uid
        for uid in heldout_uids
        if uid in set(current_complete_uids)
        and uid in set(recon_heldout_complete_uids)
    ]
    current_incomplete_uids = [
        uid for uid in heldout_uids if uid not in set(current_complete_uids)
    ]
    excluded_uids = [
        uid for uid in heldout_uids if uid not in set(complete_uids)
    ]
    if not complete_uids:
        raise RuntimeError("no complete current/Reconstruction object pairs")
    current_complete_means = _object_means(current_records, complete_uids)
    recon_complete_means = _object_means(recon_valid_records, complete_uids)
    paired = _paired_summary(
        current_complete_means,
        recon_complete_means,
        bootstrap_samples=int(args.bootstrap_samples),
    )

    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "format": REPORT_FORMAT,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "passed": True,
        "runtime_integrity_passed": True,
        "formal": False,
        "post_selection_development_diagnostic": True,
        "official_protocol_sha256": contract["protocol_sha256"],
        "strict_reconviagen_dev64": {
            "object_count": 64,
            "seed_count": 3,
            "record_count": 192,
            "successful_record_count": len(recon_valid_records),
            "failed_record_count": len(recon_failures),
            "mesh_success_rate": float(len(recon_valid_records) / 192),
            "all_meshes_valid": not recon_failures,
            "complete_surface_object_count": len(recon_all_complete_uids),
            "complete_surface_object_uids": recon_all_complete_uids,
            "incomplete_surface_object_uids": recon_all_incomplete_uids,
            "summary_of_complete_object_seed_means": _absolute_summary(
                recon_all_means
            ),
        },
        "strict_reconviagen_heldout_dev48": {
            "object_start": 16,
            "object_end": 64,
            "object_count": 48,
            "record_count": 144,
            "successful_record_count": recon_heldout_successful_record_count,
            "failed_record_count": len(recon_heldout_failures),
            "mesh_success_rate": float(
                recon_heldout_successful_record_count / (48 * 3)
            ),
            "all_meshes_valid": not recon_heldout_failures,
            "complete_surface_object_count": len(
                recon_heldout_complete_uids
            ),
            "complete_surface_object_uids": recon_heldout_complete_uids,
            "incomplete_surface_object_uids": recon_heldout_incomplete_uids,
            "summary_of_complete_object_seed_means": _absolute_summary(
                recon_heldout_means
            ),
        },
        "paired_current_vs_reconviagen_heldout_dev48": paired,
        "paired_complete_object_count": len(complete_uids),
        "paired_complete_object_uids": complete_uids,
        "excluded_incomplete_object_uids": excluded_uids,
        "reconviagen_incomplete_object_uids": recon_heldout_incomplete_uids,
        "current_incomplete_object_uids": current_incomplete_uids,
        "reconviagen_failed_seed_records": recon_failures,
        "current_failed_seed_records": current_failures,
        "reconviagen_worker_reports": recon_bindings,
        "current_endpoint_worker_reports": current_bindings,
        "surface_samples": 20000,
        "seeds": [42, 43, 44],
        "method_contracts": {
            RECON_METHOD: {
                "pipeline": "VGGT + stock SS + stock SLat + stock Mesh decoder",
                "input": "the same eight frozen official RGBA views",
                "explicit_camera_pose_consumed": False,
                "vggt_model_executed": True,
                "spconv_algo": str(
                    next(iter({row["spconv_algo"] for row in recon_bindings}), "native")
                ),
            },
            CURRENT_METHOD: {
                "pipeline": (
                    "official Native-SS step2000 EMA + official Native-SLat "
                    f"step{int(args.expected_current_step)} EMA"
                ),
                "input": "the same eight frozen official posed-DINO views",
                "explicit_camera_pose_consumed": True,
                "vggt_model_executed": False,
            },
        },
        "coordinate_contract": {
            "prediction": "stock Mesh decoder canonical; transform_pose=False",
            "target": "official SLat decoded by the same stock Mesh decoder; transform_pose=False",
            "alignment": "none; no ICP, scale fit, or GT-conditioned pose fit",
        },
        "storage_contract": {
            "predicted_meshes_saved": False,
            "render_previews_saved": False,
            "outputs": "JSON metric records and reports only",
        },
        "scope_guard": (
            "ReconViaGen runtime outcomes cover all Dev64 x three seeds; surface "
            "metrics use only objects with all three valid decoded meshes. The "
            "primary paired current-vs-ReconViaGen comparison uses Dev[16:64), "
            "because Dev[0:16) was used for Native-SS CFG calibration. Any object "
            "lacking all three valid outputs from either endpoint is explicitly "
            "excluded rather than hidden. Decoder failures remain visible in the "
            "record-level mesh-success rate and failed-record list."
        ),
    }
    report["report_sha256"] = canonical_sha256(report)
    report_path = output / "report.json"
    atomic_json(report_path, report)
    chamfer = paired["metric_deltas"]["chamfer_l1_current_improvement"]
    fscore = paired["metric_deltas"]["fscore_0p02_current_improvement"]
    normal = paired["metric_deltas"]["normal_consistency_current_improvement"]
    lcr = paired["metric_deltas"]["largest_component_ratio_current_improvement"]
    lines = [
        "Official ProObjaverse Dev64: strict ReconViaGen vs current endpoint",
        "=" * 68,
        "strict ReconViaGen: VGGT -> Stock SS -> Stock SLat -> Stock Mesh decoder",
        "saved meshes/previews: no",
        "ReconViaGen runtime matrix: 64 objects / 192 records",
        f"ReconViaGen valid meshes: {len(recon_valid_records)}/192 "
        f"({len(recon_valid_records) / 192:.6f})",
        f"ReconViaGen complete surface objects: {len(recon_all_complete_uids)}/64",
        f"ReconViaGen failed seed records: {recon_failures}",
        f"paired held-out complete objects: {len(complete_uids)}/48",
        f"excluded incomplete objects: {excluded_uids}",
        f"current Chamfer-L1 improvement: {chamfer}",
        f"current F-score@0.02 delta: {fscore}",
        f"current normal-consistency delta: {normal}",
        f"current largest-component-ratio delta: {lcr}",
        report["scope_guard"],
        f"report: {report_path}",
    ]
    (output / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def add_common_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dev_split", required=True)
    parser.add_argument("--cache_report", required=True)
    parser.add_argument("--target_report", required=True)
    parser.add_argument("--target_mesh_root", required=True)
    parser.add_argument(
        "--paired_target_cache_roots",
        default="",
        help=(
            "optional comma-separated target_mesh_cache directories from the "
            "current endpoint workers; when supplied, Dev[16:64) uses those exact "
            "target triangles for paired metrics"
        ),
    )


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    worker = commands.add_parser("worker", help="run one sharded ReconViaGen worker")
    add_common_paths(worker)
    worker.add_argument("--output_dir", required=True)
    worker.add_argument("--pretrained", default="Stable-X/trellis-vggt-v0-2")
    worker.add_argument("--seeds", default="42,43,44")
    worker.add_argument("--device", default="cuda")
    worker.add_argument("--low_vram", action="store_true")
    worker.add_argument("--ss_steps", type=int, default=30)
    worker.add_argument("--ss_guidance", type=float, default=7.5)
    worker.add_argument("--ss_guidance_rescale", type=float, default=0.7)
    worker.add_argument("--ss_rescale_t", type=float, default=5.0)
    worker.add_argument("--slat_steps", type=int, default=12)
    worker.add_argument("--slat_guidance", type=float, default=7.5)
    worker.add_argument("--slat_guidance_rescale", type=float, default=0.5)
    worker.add_argument("--slat_rescale_t", type=float, default=3.0)
    worker.add_argument(
        "--multiimage_algo",
        choices=("multidiffusion", "stochastic"),
        default="multidiffusion",
    )
    worker.add_argument("--surface_samples", type=int, default=20000)
    worker.add_argument("--worker_index", type=int, default=0)
    worker.add_argument("--num_workers", type=int, default=1)
    worker.add_argument("--resume", action="store_true")
    worker.add_argument("--dry_run", action="store_true")

    aggregate = commands.add_parser("aggregate", help="aggregate and pair with current endpoint")
    add_common_paths(aggregate)
    aggregate.add_argument("--recon_reports", required=True)
    aggregate.add_argument("--current_reports", required=True)
    aggregate.add_argument("--expected_current_step", type=int, default=25000)
    aggregate.add_argument("--expected_current_sha256", required=True)
    aggregate.add_argument("--bootstrap_samples", type=int, default=5000)
    aggregate.add_argument("--output_dir", required=True)
    return parser


def main() -> None:
    args = make_parser().parse_args()
    if args.command == "worker":
        run_worker(args)
    else:
        run_aggregate(args)


if __name__ == "__main__":
    main()
