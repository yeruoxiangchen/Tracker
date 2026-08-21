#!/usr/bin/env python3
"""Evaluate with-VGGT endpoint C against frozen strict ReconViaGen on Dev48.

This module is deliberately candidate-only.  It reuses the content-addressed
VSS predicted-support coordinates from a completed endpoint run, evaluates the
requested trained V-SLat checkpoint once, and scores the resulting Mesh against
the exact target NPZ recorded by the frozen strict ReconViaGen workers.

The aggregate subcommand then pairs:

R: strict ReconViaGen (VGGT -> Stock SS -> Stock SLat)
C: VSS predicted support -> trained with-VGGT V-SLat

at object/seed level.  Positive deltas always mean C is better.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from official_ss_with_vggt_perf_v1.ss_slat_endpoint import (
    WORKER_FORMAT as SOURCE_ENDPOINT_WORKER_FORMAT,
    WithVGGTSSSLatEndpointDataset,
    activate_ss_cache_manifest,
    build_trained_slat_pipeline,
    endpoint_contract,
    official_target_contract,
)
from pose_point_depth_mv.aggregate_proobjaverse_official_ss_slat_vs_reconviagen import (
    paired_route_summary,
)
from pose_point_depth_mv.evaluate_proobjaverse_official_native_ss_stock_slat import (
    MAX_SAFE_SLAT_DECODER_INPUT_POINTS,
    _cuda_context_poisoning_mesh_decode_error,
    _load_frozen_target_mesh_bindings,
    _materialize_frozen_target_mesh,
    _recordable_mesh_decode_error,
    _sample_trained_slat,
    pair_id,
    parse_csv,
)
from pose_point_depth_mv.evaluate_proobjaverse_official_reconviagen import (
    LOWER_IS_BETTER,
    RECON_METHOD,
    _absolute_summary,
    _load_contract,
    _load_recon_reports,
    _object_means,
    _verify_internal_hash,
    add_common_paths,
)
from pose_point_depth_mv.export_direct_flow_mesh_pairs import (
    canonical_coords,
    mesh_structure_metrics,
    sparse_noise_from_master,
    surface_metrics,
)
from pose_point_depth_mv.native_slat_genrecon import load_stock_slat_freeze
from pose_point_depth_mv.native_ss_genrecon import sha256_file
from pose_point_depth_mv.proobjaverse_official_slat_protocol import (
    atomic_json,
    canonical_sha256,
    load_json,
)
from pose_point_depth_mv.train_direct_slat_flow import to_device_tree


WORKER_FORMAT = (
    "official_ss_with_vggt_perf_v1.c_vs_strict_reconviagen.v1.worker"
)
REPORT_FORMAT = (
    "official_ss_with_vggt_perf_v1.c_vs_strict_reconviagen.v1.aggregate"
)
ROUTE_C = "with_vggt_vss2000_v_slat"
SEEDS = (42, 43, 44)


def _write_hashed_json(path: Path, payload: dict[str, Any]) -> None:
    value = deepcopy(payload)
    value.pop("report_sha256", None)
    value["report_sha256"] = canonical_sha256(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _expected_pairs(uids: list[str], seeds: list[int]) -> set[tuple[str, int]]:
    return {(uid, int(seed)) for uid in uids for seed in seeds}


def _load_source_supports(
    report_csv: str,
    *,
    selected_uids: list[str],
    seeds: list[int],
    cache_manifest: str,
    lifting_cache_manifest: str,
    ss_cache_manifest: str,
) -> tuple[dict[tuple[str, int], dict[str, Any]], list[dict[str, Any]]]:
    """Load and hash-validate VSS predicted support from completed C workers."""

    expected = _expected_pairs(selected_uids, seeds)
    selected = set(selected_uids)
    supports: dict[tuple[str, int], dict[str, Any]] = {}
    bindings: list[dict[str, Any]] = []
    expected_endpoint = endpoint_contract()
    expected_hashes = {
        "cache_manifest_sha256": sha256_file(cache_manifest),
        "lifting_cache_manifest_sha256": sha256_file(lifting_cache_manifest),
        "ss_cache_manifest_sha256": sha256_file(ss_cache_manifest),
    }
    for value in parse_csv(report_csv, str):
        path = Path(value).expanduser().resolve(strict=True)
        payload = load_json(path)
        _verify_internal_hash(payload, path=path)
        if (
            payload.get("format") != SOURCE_ENDPOINT_WORKER_FORMAT
            or payload.get("complete") is not True
            or payload.get("with_vggt_endpoint_contract") != expected_endpoint
        ):
            raise RuntimeError(f"source endpoint worker identity differs: {path}")
        identity = payload.get("run_identity")
        if not isinstance(identity, dict):
            raise RuntimeError(f"source endpoint lacks run identity: {path}")
        mismatch = {
            key: {"observed": identity.get(key), "expected": expected_value}
            for key, expected_value in expected_hashes.items()
            if identity.get(key) != expected_value
        }
        if (
            mismatch
            or [int(value) for value in identity.get("joint_seeds", [])] != seeds
            or identity.get("slat_support_input") != "predicted_only"
            or identity.get("gt_support_used_as_slat_input") is not False
        ):
            raise RuntimeError(
                f"source endpoint support contract differs: path={path} "
                f"mismatch={mismatch}"
            )
        source_root = path.parent
        local_count = 0
        for row in payload.get("ss_records", []):
            uid = str(row.get("object_uid", ""))
            seed = int(row.get("seed", -1))
            key = (uid, seed)
            if uid not in selected:
                continue
            if key not in expected or key in supports or row.get("passed") is not True:
                raise RuntimeError(f"source support record differs: {key}")
            current = pair_id(uid, seed)
            audit_path = source_root / "ss_coords" / f"{current}.json"
            npz_path = source_root / "ss_coords" / f"{current}.npz"
            audit = load_json(audit_path)
            if (
                str(audit.get("object_uid", "")) != uid
                or int(audit.get("seed", -1)) != seed
                or audit.get("passed") is not True
                or str(audit.get("coords_npz_sha256", "")) != sha256_file(npz_path)
            ):
                raise RuntimeError(f"source support artifact differs: {npz_path}")
            with np.load(npz_path) as arrays:
                if "native" not in arrays or np.asarray(arrays["native"]).ndim != 2:
                    raise RuntimeError(f"source support lacks native coordinates: {npz_path}")
                native_count = int(len(arrays["native"]))
            if native_count <= 0:
                raise RuntimeError(f"source native support is empty: {npz_path}")
            supports[key] = {
                "path": str(npz_path),
                "sha256": sha256_file(npz_path),
                "source_audit": str(audit_path),
                "source_audit_sha256": sha256_file(audit_path),
                "native_coord_count": native_count,
            }
            local_count += 1
        if local_count:
            bindings.append(
                {
                    "path": str(path),
                    "sha256": sha256_file(path),
                    "selected_support_pair_count": local_count,
                }
            )
    if set(supports) != expected:
        missing = sorted(expected - set(supports))[:8]
        raise RuntimeError(f"source endpoint support coverage differs: missing={missing}")
    if not bindings:
        raise RuntimeError("no source endpoint worker contributed selected supports")
    return supports, bindings


def _load_existing_pair(
    path: Path,
    *,
    uid: str,
    seed: int,
    target_sha256: str,
    checkpoint_sha256: str,
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    row = load_json(path)
    if (
        row.get("format") != f"{WORKER_FORMAT}.pair"
        or str(row.get("object_uid", "")) != uid
        or int(row.get("seed", -1)) != int(seed)
        or str(row.get("target_mesh_sha256", "")) != target_sha256
        or str(row.get("trained_slat_checkpoint_sha256", ""))
        != checkpoint_sha256
        or not isinstance(row.get("passed"), bool)
    ):
        raise RuntimeError(f"resumed C pair identity differs: {path}")
    body = dict(row)
    saved = str(body.pop("report_sha256", ""))
    if not saved or canonical_sha256(body) != saved:
        raise RuntimeError(f"resumed C pair hash differs: {path}")
    return row


@torch.no_grad()
def run_worker(args: argparse.Namespace) -> None:
    seeds = parse_csv(args.joint_seeds, int)
    if tuple(seeds) != SEEDS:
        raise ValueError(f"strict comparison seeds must be {SEEDS}")
    if int(args.surface_samples) != 20000:
        raise ValueError("strict comparison requires surface_samples=20000")
    activate_ss_cache_manifest(args.ss_cache_manifest)
    dataset = WithVGGTSSSLatEndpointDataset(
        args.cache_manifest,
        args.lifting_cache_manifest,
        ss_cache_manifest=args.ss_cache_manifest,
    )
    contract = official_target_contract(dataset)
    total = len(dataset)
    start = int(args.object_start)
    end = total if int(args.object_end) <= 0 else int(args.object_end)
    if start < 0 or end <= start or end > total:
        raise ValueError(f"invalid object slice [{start}:{end}] for {total} objects")
    indices = list(range(start, end))
    selected_uids = [str(dataset.rows[index]["object_uid"]) for index in indices]
    supports, support_reports = _load_source_supports(
        args.source_endpoint_reports,
        selected_uids=selected_uids,
        seeds=seeds,
        cache_manifest=args.cache_manifest,
        lifting_cache_manifest=args.lifting_cache_manifest,
        ss_cache_manifest=args.ss_cache_manifest,
    )
    frozen_targets, target_identity = _load_frozen_target_mesh_bindings(
        args.strict_recon_reports,
        selected_uids=selected_uids,
        seeds=seeds,
        protocol_sha256=str(contract["protocol_sha256"]),
    )
    if target_identity["policy"] != "exact_npz_from_frozen_strict_reconviagen_reports":
        raise RuntimeError("strict C-R worker did not bind frozen R targets")

    output = Path(args.output_dir).expanduser().resolve()
    checkpoint_path = Path(args.trained_slat_checkpoint).expanduser().resolve(strict=True)
    checkpoint_sha256 = sha256_file(checkpoint_path)
    identity = {
        "format": WORKER_FORMAT,
        "cache_manifest": str(Path(args.cache_manifest).expanduser().resolve(strict=True)),
        "cache_manifest_sha256": sha256_file(args.cache_manifest),
        "lifting_cache_manifest": str(
            Path(args.lifting_cache_manifest).expanduser().resolve(strict=True)
        ),
        "lifting_cache_manifest_sha256": sha256_file(args.lifting_cache_manifest),
        "ss_cache_manifest": str(
            Path(args.ss_cache_manifest).expanduser().resolve(strict=True)
        ),
        "ss_cache_manifest_sha256": sha256_file(args.ss_cache_manifest),
        "source_endpoint_reports": support_reports,
        "strict_recon_reports": target_identity["strict_recon_reports"],
        "target_mesh_policy": target_identity["policy"],
        "frozen_target_binding_sha256": target_identity["binding_sha256"],
        "trained_slat_checkpoint": str(checkpoint_path),
        "trained_slat_checkpoint_sha256": checkpoint_sha256,
        "expected_trained_slat_step": int(args.expected_trained_slat_step),
        "trained_slat_weights": "ema",
        "stock_slat_freeze": str(
            Path(args.stock_slat_freeze).expanduser().resolve(strict=True)
        ),
        "stock_slat_freeze_sha256": sha256_file(args.stock_slat_freeze),
        "official_protocol_sha256": str(contract["protocol_sha256"]),
        "object_start": start,
        "object_end": end,
        "object_uids": selected_uids,
        "joint_seeds": seeds,
        "surface_samples": int(args.surface_samples),
        "amp_dtype": str(args.amp_dtype),
        "source_support_reused": True,
        "vss_inference_reexecuted": False,
        "strict_reconviagen_inference_reexecuted": False,
        "evaluated_branch": "C_native_trained_only",
        "same_metric_seed_as_strict_reconviagen": True,
    }
    identity_path = output / "run_identity.json"
    if output.exists() and not args.resume:
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    if identity_path.is_file():
        if load_json(identity_path) != identity:
            raise RuntimeError("resume arguments differ from existing C worker identity")
    else:
        atomic_json(identity_path, identity)
    report_path = output / "report.json"
    if report_path.is_file():
        payload = load_json(report_path)
        body = dict(payload)
        saved = str(body.pop("report_sha256", ""))
        if payload.get("format") != WORKER_FORMAT or canonical_sha256(body) != saved:
            raise RuntimeError("existing C worker report differs")
        print(json.dumps({"reused": True, "passed": payload["passed"]}, indent=2))
        return

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    amp_enabled = args.amp_dtype != "none"
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    stock_freeze = load_stock_slat_freeze(args.stock_slat_freeze)
    runtime = build_trained_slat_pipeline(
        checkpoint_path=checkpoint_path,
        weights="ema",
        pretrained=str(args.pretrained),
        stock_freeze=stock_freeze,
        dataset=dataset,
        expected_step=int(args.expected_trained_slat_step),
        device=device,
        evaluation_object_uids=selected_uids,
        allow_target_protocol_mismatch=False,
        expected_training_membership="all_disjoint",
    )
    if runtime["checkpoint_sha256"] != checkpoint_sha256:
        raise RuntimeError("trained C checkpoint changed during runtime construction")
    mesh_decoder = runtime["decoder"]
    channels = int(runtime["model"].flow_core.in_channels)
    records: list[dict[str, Any]] = []
    pair_root = output / "pairs"
    target_root = output / "target_mesh_cache"

    for position, index in enumerate(indices):
        uid = str(dataset.rows[index]["object_uid"])
        target_cache = target_root / f"{uid}.npz"
        target_mesh, target_sha256 = _materialize_frozen_target_mesh(
            frozen_targets[uid], target_cache
        )
        target_structure = mesh_structure_metrics(target_mesh)
        if target_structure != frozen_targets[uid]["structure"]:
            raise RuntimeError(f"strict R target structure changed: {uid}")
        sample = dataset[index]
        condition = to_device_tree(sample["condition"], device)
        lifting_sample = sample["lifting_sample"]
        for seed in seeds:
            current = pair_id(uid, seed)
            pair_path = pair_root / current / "record.json"
            existing = _load_existing_pair(
                pair_path,
                uid=uid,
                seed=seed,
                target_sha256=target_sha256,
                checkpoint_sha256=checkpoint_sha256,
            )
            if existing is not None:
                records.append(existing)
                print(
                    f"[strict_c:reuse] {position + 1}/{len(indices)} "
                    f"seed={seed} uid={uid}",
                    flush=True,
                )
                continue
            support_binding = supports[(uid, seed)]
            support_path = Path(support_binding["path"])
            if sha256_file(support_path) != support_binding["sha256"]:
                raise RuntimeError(f"source VSS support changed: {support_path}")
            with np.load(support_path) as arrays:
                coords = canonical_coords(arrays["native"], resolution=64)
            generator = torch.Generator(device=device).manual_seed(
                int(seed) * 2000003 + int(index) * 2017 + 7919
            )
            master = torch.randn(
                (64, 64, 64, channels),
                generator=generator,
                device=device,
                dtype=torch.float32,
            )
            initial = sparse_noise_from_master(coords, master, device=device)
            metric_seed = int(seed) * 1009 + int(index) * 9173
            row: dict[str, Any] = {
                "format": f"{WORKER_FORMAT}.pair",
                "object_uid": uid,
                "object_index": int(index),
                "seed": int(seed),
                "branch": "native_trained",
                "pair_id": current,
                "passed": False,
                "source_support_npz": str(support_path),
                "source_support_npz_sha256": support_binding["sha256"],
                "source_native_coord_count": int(support_binding["native_coord_count"]),
                "target_mesh_sha256": target_sha256,
                "target_mesh_policy": target_identity["policy"],
                "target_structure": target_structure,
                "trained_slat_checkpoint_sha256": checkpoint_sha256,
                "metric_seed": metric_seed,
                "surface_samples": int(args.surface_samples),
            }
            latent = decoded = mesh = None
            poison = False
            try:
                latent, flow_summary = _sample_trained_slat(
                    runtime=runtime,
                    initial=initial,
                    condition=condition,
                    lifting_sample=lifting_sample,
                    adapted=True,
                    amp_enabled=amp_enabled,
                    amp_dtype=amp_dtype,
                )
                active_points = int(latent.feats.shape[0])
                row["slat_active_point_count"] = active_points
                row["slat_active_point_limit"] = int(
                    MAX_SAFE_SLAT_DECODER_INPUT_POINTS
                )
                if active_points > MAX_SAFE_SLAT_DECODER_INPUT_POINTS:
                    raise RuntimeError(
                        "SLat decoder input exceeds safe active-point limit: "
                        f"points={active_points} "
                        f"limit={MAX_SAFE_SLAT_DECODER_INPUT_POINTS}"
                    )
                try:
                    decoded = mesh_decoder(latent)[0]
                except RuntimeError as error:
                    if _cuda_context_poisoning_mesh_decode_error(error):
                        poison = True
                        raise RuntimeError(
                            "SLat decoder CUDA topology failure: "
                            f"branch=native_trained uid={uid} seed={seed}: {error}"
                        ) from error
                    raise
                mesh = decoded.to_trimesh(transform_pose=False)
                structure = mesh_structure_metrics(mesh)
                if not structure["mesh_success"]:
                    raise RuntimeError(f"decoded native_trained Mesh is invalid: {uid}")
                row.update(
                    {
                        "passed": True,
                        "structure": structure,
                        "surface": surface_metrics(
                            mesh,
                            target_mesh,
                            count=int(args.surface_samples),
                            seed=metric_seed,
                            thresholds=(0.01, 0.02, 0.05),
                        ),
                        "flow_summary": flow_summary,
                    }
                )
            except Exception as error:
                if not _recordable_mesh_decode_error(error):
                    raise
                row["error"] = {
                    "type": type(error).__name__,
                    "message": str(error),
                    "stage": "native_trained_slat_mesh_decode",
                }
            finally:
                del latent, decoded, mesh, initial, master
            _write_hashed_json(pair_path, row)
            records.append(load_json(pair_path))
            print(
                f"[strict_c:{'ok' if row['passed'] else 'failed'}] "
                f"{position + 1}/{len(indices)} seed={seed} uid={uid}",
                flush=True,
            )
            if poison:
                raise SystemExit(75)
            torch.cuda.empty_cache()
        del condition, lifting_sample, sample, target_mesh
        gc.collect()
        torch.cuda.empty_cache()

    expected = _expected_pairs(selected_uids, seeds)
    observed = {(str(row["object_uid"]), int(row["seed"])) for row in records}
    if observed != expected or len(records) != len(expected):
        raise RuntimeError("completed C worker pair matrix differs")
    report = {
        "format": WORKER_FORMAT,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "complete": True,
        "passed": all(row.get("passed") is True for row in records),
        "formal": False,
        "object_count": len(selected_uids),
        "record_count": len(records),
        "successful_record_count": sum(row.get("passed") is True for row in records),
        "failed_record_count": sum(row.get("passed") is not True for row in records),
        "run_identity": identity,
        "checkpoint_evaluation_membership": runtime[
            "checkpoint_evaluation_membership"
        ],
        "records": records,
        "scope_guard": (
            "held-out Dev48 strict-target development comparison; VSS support and "
            "strict ReconViaGen outputs are reused, and only C is re-evaluated"
        ),
    }
    _write_hashed_json(report_path, report)
    print(
        json.dumps(
            {
                "report": str(report_path),
                "passed": report["passed"],
                "objects": report["object_count"],
                "records": report["record_count"],
            },
            indent=2,
        )
    )


def _runtime_summary(
    records: dict[tuple[str, int], dict[str, Any]], uids: list[str]
) -> dict[str, Any]:
    expected = _expected_pairs(uids, list(SEEDS))
    successful = {key for key, row in records.items() if row.get("passed") is True}
    complete = [
        uid
        for uid in uids
        if all((uid, seed) in successful for seed in SEEDS)
    ]
    return {
        "requested_object_count": len(uids),
        "requested_record_count": len(expected),
        "successful_record_count": len(successful),
        "failed_record_count": len(expected - successful),
        "record_success_rate": len(successful) / len(expected),
        "complete_surface_object_count": len(complete),
        "complete_surface_object_uids": complete,
        "failed_records": [
            {
                "object_uid": uid,
                "seed": seed,
                "error": records[(uid, seed)].get("error"),
            }
            for uid, seed in sorted(expected - successful)
        ],
    }


def run_aggregate(args: argparse.Namespace) -> None:
    contract = _load_contract(args)
    all_dev_uids = [str(row["uid"]) for row in contract["rows"]]
    start = int(args.object_start)
    end = int(args.object_end)
    if not 0 <= start < end <= len(all_dev_uids):
        raise ValueError("invalid aggregate object range")
    uids = all_dev_uids[start:end]
    expected = _expected_pairs(uids, list(SEEDS))

    recon_bindings, recon_all = _load_recon_reports(
        parse_csv(args.strict_recon_reports, str)
    )
    if set(recon_all) != _expected_pairs(all_dev_uids, list(SEEDS)):
        raise RuntimeError("strict ReconViaGen reports do not cover Dev64 x seeds")
    recon = {key: row for key, row in recon_all.items() if key in expected}

    candidate: dict[tuple[str, int], dict[str, Any]] = {}
    candidate_bindings: list[dict[str, Any]] = []
    shared_checkpoint: dict[str, Any] | None = None
    for value in parse_csv(args.candidate_reports, str):
        path = Path(value).expanduser().resolve(strict=True)
        payload = load_json(path)
        body = dict(payload)
        saved = str(body.pop("report_sha256", ""))
        if (
            payload.get("format") != WORKER_FORMAT
            or payload.get("complete") is not True
            or not saved
            or canonical_sha256(body) != saved
        ):
            raise RuntimeError(f"invalid candidate C worker report: {path}")
        identity = payload.get("run_identity")
        if not isinstance(identity, dict):
            raise RuntimeError(f"candidate C worker lacks identity: {path}")
        if (
            identity.get("target_mesh_policy")
            != "exact_npz_from_frozen_strict_reconviagen_reports"
            or identity.get("official_protocol_sha256")
            != contract["protocol_sha256"]
            or identity.get("joint_seeds") != list(SEEDS)
            or identity.get("surface_samples") != 20000
            or identity.get("same_metric_seed_as_strict_reconviagen") is not True
            or identity.get("evaluated_branch") != "C_native_trained_only"
        ):
            raise RuntimeError(f"candidate C worker strict identity differs: {path}")
        checkpoint = {
            "path": identity["trained_slat_checkpoint"],
            "sha256": identity["trained_slat_checkpoint_sha256"],
            "step": identity["expected_trained_slat_step"],
            "weights": identity["trained_slat_weights"],
        }
        if shared_checkpoint is None:
            shared_checkpoint = checkpoint
        elif checkpoint != shared_checkpoint:
            raise RuntimeError("candidate C checkpoint bindings differ")
        for row in payload.get("records", []):
            key = (str(row.get("object_uid", "")), int(row.get("seed", -1)))
            if key not in expected or key in candidate:
                raise RuntimeError(f"candidate C pair coverage differs: {key}")
            if (
                str(row.get("target_mesh_sha256", ""))
                != str(recon[key].get("target_mesh_sha256", ""))
                or row.get("target_structure") != recon[key].get("target_structure")
                or int(row.get("metric_seed", -1))
                != int(recon[key].get("metric_seed", -2))
                or int(row.get("surface_samples", -1)) != 20000
            ):
                raise RuntimeError(f"candidate/strict R metric target differs: {key}")
            candidate[key] = row
        candidate_bindings.append(
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "object_start": identity["object_start"],
                "object_end": identity["object_end"],
                "frozen_target_binding_sha256": identity[
                    "frozen_target_binding_sha256"
                ],
                "runtime_passed": payload.get("passed") is True,
            }
        )
    if set(candidate) != expected:
        raise RuntimeError("candidate C reports do not cover Dev48 x seeds")
    if shared_checkpoint is None:
        raise RuntimeError("no candidate C reports were loaded")
    checkpoint_path = Path(shared_checkpoint["path"]).expanduser().resolve(strict=True)
    if sha256_file(checkpoint_path) != shared_checkpoint["sha256"]:
        raise RuntimeError("live candidate C checkpoint SHA256 differs")

    runtime_r = _runtime_summary(recon, uids)
    runtime_c = _runtime_summary(candidate, uids)
    complete_r = set(runtime_r["complete_surface_object_uids"])
    complete_c = set(runtime_c["complete_surface_object_uids"])
    common_uids = [uid for uid in uids if uid in complete_r and uid in complete_c]
    if not common_uids:
        raise RuntimeError("no common complete C/R objects remain")
    means_r = _object_means(recon, common_uids)
    means_c = _object_means(candidate, common_uids)
    comparison = paired_route_summary(
        means_c,
        means_r,
        candidate_name=(
            f"with_vggt_vss2000_v_slat_step{int(shared_checkpoint['step'])}"
        ),
        baseline_name=RECON_METHOD,
        bootstrap_samples=int(args.bootstrap_samples),
        bootstrap_seed=20260816,
        comparison_kind="C_minus_R_complete_endpoint_comparison",
        clean_component_isolation=False,
        caveat=(
            "C-R changes the input interface, SS and SLat together. Both routes "
            "are scored against byte-identical frozen target NPZs with identical "
            "surface sampling seeds; the comparison cannot be attributed to one "
            "component."
        ),
        unit_scope="heldout_development",
    )
    report = {
        "format": REPORT_FORMAT,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "passed": True,
        "runtime_integrity_passed": True,
        "formal": False,
        "post_selection_development_diagnostic": True,
        "split": {
            "name": "official_proobjaverse_dev",
            "object_start": start,
            "object_end": end,
            "requested_object_count": len(uids),
            "seeds": list(SEEDS),
        },
        "official_protocol_sha256": contract["protocol_sha256"],
        "route_runtime": {RECON_METHOD: runtime_r, ROUTE_C: runtime_c},
        "common_complete_object_count": len(common_uids),
        "common_complete_object_uids": common_uids,
        "excluded_from_paired_surface_metrics": [
            uid for uid in uids if uid not in set(common_uids)
        ],
        "route_absolute_summary_on_common_objects": {
            RECON_METHOD: _absolute_summary(means_r),
            ROUTE_C: _absolute_summary(means_c),
        },
        "comparison": comparison,
        "comparability_guard": {
            "same_official_object_seed_pairs": True,
            "same_frozen_input_views": True,
            "same_target_mesh_structure": True,
            "same_target_mesh_sha256": True,
            "same_surface_samples": True,
            "same_metric_seed": True,
            "same_fscore_thresholds": [0.01, 0.02, 0.05],
            "strict_reconviagen_inference_reused": True,
            "vss_predicted_support_reused": True,
            "only_c_slat_and_mesh_recomputed": True,
            "c_minus_r_is_complete_endpoint_not_component_attribution": True,
        },
        "candidate_checkpoint": shared_checkpoint,
        "strict_reconviagen_worker_reports": recon_bindings,
        "candidate_worker_reports": sorted(
            candidate_bindings, key=lambda row: row["object_start"]
        ),
        "source_contract": {
            "dev_split": str(contract["split_path"]),
            "dev_split_sha256": contract["split_sha256"],
            "cache_report": str(contract["cache_report_path"]),
            "cache_report_sha256": contract["cache_report_sha256"],
        },
        "scope_guard": (
            "Object-disjoint held-out Dev48 post-selection development comparison, "
            "not a final untouched test claim. Surface summaries use only objects "
            "with all three valid seeds in both C and strict R; every registered "
            "model-output failure remains visible."
        ),
    }
    report["report_sha256"] = canonical_sha256(report)
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / "report.json", report)

    core = comparison["metric_deltas"]
    lines = [
        "With-VGGT endpoint C vs strict ReconViaGen R on frozen-target Dev48",
        "====================================================================",
        "R = VGGT -> Stock SS -> Stock SLat -> Stock Mesh decoder",
        (
            "C = with-VGGT VSS step2000 predicted support -> "
            f"with-VGGT V-SLat step{int(shared_checkpoint['step'])} -> Stock Mesh decoder"
        ),
        f"common complete objects: {len(common_uids)}/{len(uids)}",
        f"excluded objects: {report['excluded_from_paired_surface_metrics']}",
        "positive deltas mean C is better",
    ]
    labels = (
        ("chamfer_l1", "Chamfer-L1 improvement"),
        ("fscore_0p02", "F-score@0.02 delta"),
        ("normal_consistency", "Normal delta"),
        ("largest_component_ratio", "LCR delta"),
    )
    for key, label in labels:
        row = core[key]
        lines.append(
            f"{label}: mean={row['mean']:+.8f} median={row['median']:+.8f} "
            f"win={row['positive_rate']:.6f} "
            f"CI={row['object_bootstrap_mean_95_ci']}"
        )
    lines.extend([report["scope_guard"], f"report: {output / 'report.json'}"])
    (output / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    worker = subparsers.add_parser("worker")
    worker.add_argument("--cache_manifest", required=True)
    worker.add_argument("--lifting_cache_manifest", required=True)
    worker.add_argument("--ss_cache_manifest", required=True)
    worker.add_argument("--source_endpoint_reports", required=True)
    worker.add_argument("--strict_recon_reports", required=True)
    worker.add_argument("--trained_slat_checkpoint", required=True)
    worker.add_argument("--expected_trained_slat_step", type=int, required=True)
    worker.add_argument("--stock_slat_freeze", required=True)
    worker.add_argument("--output_dir", required=True)
    worker.add_argument("--pretrained", default="Stable-X/trellis-vggt-v0-2")
    worker.add_argument("--joint_seeds", default="42,43,44")
    worker.add_argument("--object_start", type=int, required=True)
    worker.add_argument("--object_end", type=int, required=True)
    worker.add_argument("--surface_samples", type=int, default=20000)
    worker.add_argument("--amp_dtype", choices=("bf16", "fp16", "none"), default="bf16")
    worker.add_argument("--resume", action="store_true")

    aggregate = subparsers.add_parser("aggregate")
    add_common_paths(aggregate)
    aggregate.add_argument("--strict_recon_reports", required=True)
    aggregate.add_argument("--candidate_reports", required=True)
    aggregate.add_argument("--object_start", type=int, default=16)
    aggregate.add_argument("--object_end", type=int, default=64)
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
