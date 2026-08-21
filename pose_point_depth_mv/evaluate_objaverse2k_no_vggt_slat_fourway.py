#!/usr/bin/env python3
"""Matched four-way attribution test for an Objaverse2K SLat checkpoint.

Each object/seed quartet shares frozen Native-SS coordinates, initial noise,
Stock image context, sampler and decoder:

* ``stock`` disables both LoRA and the posed-DINO residual;
* ``lora_only`` keeps LoRA but zeros the complete every-block 3D residual;
* ``pose_cyclic1`` keeps LoRA/posed-DINO but cyclically mismatches camera poses;
* ``correct`` keeps LoRA/posed-DINO with the correct image/pose binding.

The selected objects are disjoint from Objaverse2K SLat training, but remain a
development set used for mechanism diagnosis and checkpoint selection.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import nullcontext
import gc
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from pose_point_depth_mv.evaluate_mixed_no_vggt_slat_fourway import LoRAOnlyCFGFlow
from pose_point_depth_mv.evaluate_objaverse2k_no_vggt_slat import (
    STRUCTURE_METRICS,
    SURFACE_METRICS,
    atomic_json,
    canonical_json_sha256,
    decode_mesh_fp32,
    load_json,
    paired_improvement,
    parse_csv,
    parse_int_csv,
    sampling_contract,
    select_worker_matrix,
    stable_seed,
    upstream_binding,
)
from pose_point_depth_mv.eval_direct_slat_flow import summarize
from pose_point_depth_mv.export_direct_flow_mesh_pairs import (
    canonical_coords,
    load_canonical_gt,
    mesh_structure_metrics,
    sparse_noise_from_master,
    surface_metrics,
)
from pose_point_depth_mv.native_3d_condition import NativeConditionSLatDataset
from pose_point_depth_mv.native_slat_genrecon_no_vggt import (
    NATIVE_SLAT_NO_VGGT_VERSION,
    build_native_slat_no_vggt_components,
    validate_native_slat_no_vggt_checkpoint,
)
from pose_point_depth_mv.native_slat_genrecon_v2 import (
    NativeSLatCalibratedCFGFlow,
    NativeSLatStockFlow,
    load_stock_slat_freeze,
    load_trainable_state_dict,
)
from pose_point_depth_mv.no_vggt_ss_evidence import load_no_vggt_ss_evidence
from pose_point_depth_mv.objaverse2k_slat_pipeline import (
    resolve_native_objaverse_normalization_bindings,
)
from pose_point_depth_mv.omni_real_benchmark_common import sha256_file
from pose_point_depth_mv.train_direct_slat_flow import to_device_tree
from trellis.modules import sparse as sp


WORKER_REPORT_FORMAT = "pose_point_depth_mv.objaverse2k_slat_fourway_worker.v1"
AGGREGATE_REPORT_FORMAT = "pose_point_depth_mv.objaverse2k_slat_fourway.v1"
BRANCHES = ("stock", "lora_only", "pose_cyclic1", "correct")
COMPARISONS = (
    ("correct", "stock", "total_trained_slat"),
    ("correct", "lora_only", "posed_dino_increment"),
    ("correct", "pose_cyclic1", "correct_pose_specificity"),
    ("lora_only", "stock", "generic_lora_increment"),
)
ALL_METRICS = (*SURFACE_METRICS, *STRUCTURE_METRICS)


def record_identity(
    *,
    checkpoint_sha256: str,
    object_uid: str,
    uid: str,
    support_seed: int,
    master_noise_seed: int,
    metric_seed: int,
    cache_manifest_sha256: str,
) -> dict[str, Any]:
    return {
        "checkpoint_sha256": checkpoint_sha256,
        "object_uid": object_uid,
        "uid": uid,
        "support_seed": int(support_seed),
        "master_noise_seed": int(master_noise_seed),
        "metric_seed": int(metric_seed),
        "cache_manifest_sha256": cache_manifest_sha256,
    }


def _validate_branch_matrix(record: dict[str, Any]) -> None:
    branches = record.get("branches", {})
    if set(branches) != set(BRANCHES):
        raise RuntimeError(
            f"incomplete four-way record={record.get('identity', {})}: "
            f"{sorted(branches)}"
        )
    if record.get("same_native_ss_coordinates") is not True:
        raise RuntimeError("four-way record does not freeze Native-SS coordinates")
    if record.get("same_initial_noise") is not True:
        raise RuntimeError("four-way record does not freeze initial noise")
    lora_wrapper = branches["lora_only"].get("wrapper", {})
    if (
        lora_wrapper.get("mode") != "lora_only"
        or lora_wrapper.get("posed_dino_input") is not False
        or lora_wrapper.get("every_block_condition_output")
        != "exact_zero_including_projection_bias"
    ):
        raise RuntimeError("LoRA-only branch did not exactly suppress posed-DINO")
    for branch in ("lora_only", "pose_cyclic1", "correct"):
        wrapper = branches[branch].get("wrapper", {})
        if int(wrapper.get("positive_calls", 0)) <= 0 or int(
            wrapper.get("negative_calls", 0)
        ) <= 0:
            raise RuntimeError(f"{branch} did not execute both CFG branches")
    if branches["pose_cyclic1"]["wrapper"].get("projection_mode") != "pose_cyclic1":
        raise RuntimeError("cyclic branch lacks the cyclic-pose binding")
    if branches["correct"]["wrapper"].get("projection_mode") != "correct":
        raise RuntimeError("correct branch lacks the correct-pose binding")


@torch.no_grad()
def command_worker(args: argparse.Namespace) -> None:
    seeds = parse_int_csv(args.joint_seeds)
    cache_path = Path(args.cache_manifest).expanduser().resolve()
    lifting_path = Path(args.lifting_cache_manifest).expanduser().resolve()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    ss_report_path = Path(args.native_ss_report).expanduser().resolve()
    freeze_path = Path(args.stock_slat_freeze).expanduser().resolve()
    for path in (cache_path, lifting_path, checkpoint_path, ss_report_path, freeze_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    dataset = NativeConditionSLatDataset(cache_path, lifting_path, indices="all")
    if dataset.config.get("condition_arch") != "native_ss_genrecon_v2":
        raise RuntimeError("four-way evaluation requires a no-VGGT Native-SS cache")
    selected, object_start, object_end = select_worker_matrix(
        dataset.rows,
        seeds=seeds,
        worker_index=int(args.worker_index),
        num_workers=int(args.num_workers),
        expected_objects=int(args.expected_objects),
    )
    normalization_bindings = resolve_native_objaverse_normalization_bindings(
        cache_path,
        load_json(cache_path),
        [dataset.rows[seed_map[seeds[0]]] for _, _, seed_map in selected],
    )
    _, ss_binding_all = load_no_vggt_ss_evidence(ss_report_path)
    if dataset.config.get("native_ss_deployment") != ss_binding_all:
        raise RuntimeError("cache/evaluation Native-SS deployments differ")
    ss_binding = upstream_binding(ss_binding_all)
    stock_freeze = load_stock_slat_freeze(freeze_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if checkpoint.get("format") != NATIVE_SLAT_NO_VGGT_VERSION:
        raise RuntimeError("four-way evaluation requires a trained no-VGGT checkpoint")
    validate_native_slat_no_vggt_checkpoint(
        checkpoint,
        pretrained=args.pretrained,
        stock_slat_freeze=stock_freeze,
        upstream_native_ss=ss_binding,
        allow_v2_parent=False,
    )
    if checkpoint.get("args", {}).get("architecture") != "v2":
        raise RuntimeError("Objaverse2K four-way evaluation freezes architecture v2")
    if checkpoint.get("args", {}).get("stock_context_views", "all") != "all":
        raise RuntimeError("Objaverse2K four-way evaluation requires all-view Stock context")

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    sampler, model, decoder, model_summary, defaults, normalization = (
        build_native_slat_no_vggt_components(
            pretrained=args.pretrained,
            stock_slat_freeze=stock_freeze,
            upstream_native_ss=ss_binding,
            lora_rank=int(checkpoint["args"]["lora_rank"]),
            lora_alpha=int(checkpoint["args"]["lora_alpha"]),
            condition_channels=int(checkpoint["args"]["condition_channels"]),
            gradient_checkpointing=False,
            need_decoder=True,
            device=device,
        )
    )
    if decoder is None:
        raise RuntimeError("mesh decoder was not constructed")
    state_key = "ema_trainable_state" if args.weights == "ema" else "model_trainable_state"
    load_trainable_state_dict(model, checkpoint[state_key])
    model.eval()
    decoder.eval()
    runtime_normalization = {
        key: [float(value) for value in values] for key, values in normalization.items()
    }
    if canonical_json_sha256(runtime_normalization) != dataset.slat_normalization_hash:
        raise RuntimeError("runtime SLat normalization differs from cache")
    mean = torch.tensor(runtime_normalization["mean"], device=device)[None]
    std = torch.tensor(runtime_normalization["std"], device=device)[None]
    params = sampling_contract(defaults)
    cache_sha = sha256_file(cache_path)
    checkpoint_sha = sha256_file(checkpoint_path)
    run_config = {
        "format": WORKER_REPORT_FORMAT,
        "formal": False,
        "training_overlap": False,
        "scope": "object-disjoint Objaverse2K development mechanism diagnostic",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_step": int(checkpoint["step"]),
        "weights": str(args.weights),
        "cache_manifest": str(cache_path),
        "cache_manifest_sha256": cache_sha,
        "lifting_cache_manifest": str(lifting_path),
        "lifting_cache_manifest_sha256": sha256_file(lifting_path),
        "native_ss_report": str(ss_report_path),
        "native_ss_report_sha256": sha256_file(ss_report_path),
        "native_ss": ss_binding,
        "stock_slat_freeze": str(freeze_path),
        "stock_slat_freeze_sha256": sha256_file(freeze_path),
        "sampling": {
            key: params[key]
            for key in (
                "steps",
                "cfg_strength",
                "cfg_interval",
                "guidance_rescale",
                "rescale_t",
            )
        },
        "branches": list(BRANCHES),
        "joint_seeds": seeds,
        "noise_protocol": "sha256(object_uid,uid,support_seed,noise_seed)_master64x64x64x8.v1",
        "noise_seed": int(args.noise_seed),
        "surface_samples": int(args.surface_samples),
        "worker_index": int(args.worker_index),
        "num_workers": int(args.num_workers),
        "expected_objects": int(args.expected_objects),
        "object_start": object_start,
        "object_end": object_end,
        "selected": [
            {"object_uid": object_uid, "uid": uid} for object_uid, uid, _ in selected
        ],
    }
    output_dir = Path(args.output_dir).expanduser().resolve()
    config_path = output_dir / "run_config.json"
    if output_dir.exists():
        if not args.resume:
            raise FileExistsError(output_dir)
        if not config_path.is_file() or load_json(config_path) != run_config:
            raise RuntimeError("four-way resume protocol differs")
    else:
        output_dir.mkdir(parents=True)
        atomic_json(config_path, run_config)

    amp_enabled = args.amp_dtype != "none"
    amp_dtype = torch.float16 if args.amp_dtype == "fp16" else torch.bfloat16
    records: list[dict[str, Any]] = []
    for position, (object_uid, uid, seed_map) in enumerate(selected, start=object_start):
        for seed in seeds:
            master_seed = stable_seed(
                "objaverse2k_slat_noise_v1",
                int(args.noise_seed),
                object_uid,
                uid,
                seed,
            )
            metric_seed = stable_seed(
                "objaverse2k_slat_metric_v1", object_uid, uid, seed
            )
            identity = record_identity(
                checkpoint_sha256=checkpoint_sha,
                object_uid=object_uid,
                uid=uid,
                support_seed=seed,
                master_noise_seed=master_seed,
                metric_seed=metric_seed,
                cache_manifest_sha256=cache_sha,
            )
            result_path = output_dir / "records" / object_uid / f"seed_{seed}.json"
            if result_path.is_file():
                existing = load_json(result_path)
                if existing.get("identity") != identity:
                    raise RuntimeError(f"resumed record identity differs: {result_path}")
                _validate_branch_matrix(existing)
                for branch in BRANCHES:
                    branch_row = existing["branches"][branch]
                    if sha256_file(branch_row["mesh"]) != branch_row["mesh_sha256"]:
                        raise RuntimeError(f"resumed mesh changed: {branch_row['mesh']}")
                records.append(existing)
                print(
                    f"[objaverse2k_slat_fourway] reuse {position + 1}/"
                    f"{args.expected_objects} {object_uid} seed={seed}",
                    flush=True,
                )
                continue

            sample = dataset[seed_map[seed]]
            coords_np = canonical_coords(
                sample["corrected_coords64"].numpy(), resolution=64
            )
            generator = torch.Generator(device=device).manual_seed(master_seed)
            master = torch.randn(
                (64, 64, 64, 8),
                generator=generator,
                device=device,
                dtype=torch.float32,
            )
            initial = sparse_noise_from_master(coords_np, master, device=device)
            condition = to_device_tree(sample["condition"], device)
            ss_latent_path = Path(str(sample["ss_latent"])).resolve()
            target_mesh, target_metadata = load_canonical_gt(
                sample,
                canonical_margin_binding=normalization_bindings.get(str(ss_latent_path)),
            )
            branches: dict[str, Any] = {}
            for branch in BRANCHES:
                noise = sp.SparseTensor(
                    feats=initial.feats.clone(), coords=initial.coords.clone()
                )
                wrapper: dict[str, Any] | None
                if branch == "stock":
                    flow: nn.Module = NativeSLatStockFlow(model)
                    wrapper = {
                        "mode": "stock",
                        "lora": False,
                        "posed_dino_residual": False,
                        "projection_mode": "none",
                    }
                elif branch == "lora_only":
                    flow = LoRAOnlyCFGFlow(model, condition["cond"])
                    wrapper = None
                else:
                    projection_mode = (
                        "correct" if branch == "correct" else "pose_cyclic1"
                    )
                    flow = NativeSLatCalibratedCFGFlow(
                        model,
                        condition["cond"],
                        sample["lifting_sample"],
                        enabled=True,
                        projection_mode=projection_mode,
                    )
                    wrapper = None
                autocast = (
                    torch.autocast(device_type="cuda", dtype=amp_dtype)
                    if amp_enabled
                    else nullcontext()
                )
                with autocast:
                    latent = sampler.sample(
                        flow, noise, **condition, **params, verbose=False
                    ).samples
                if not torch.equal(latent.coords, initial.coords):
                    raise RuntimeError(f"{branch} changed frozen Native-SS coordinates")
                if branch != "stock":
                    wrapper = flow.summary()  # type: ignore[attr-defined]
                    wrapper["projection_mode"] = (
                        "none" if branch == "lora_only" else branch
                    )
                decoded = decode_mesh_fp32(decoder, latent, std=std, mean=mean)
                mesh = decoded.to_trimesh(transform_pose=False)
                structure = mesh_structure_metrics(mesh)
                if not structure["mesh_success"]:
                    raise RuntimeError(f"empty/non-finite mesh: {object_uid} {branch}")
                mesh_path = (
                    output_dir
                    / "meshes"
                    / object_uid
                    / f"seed_{seed}"
                    / branch
                    / "mesh.obj"
                )
                mesh_path.parent.mkdir(parents=True, exist_ok=True)
                temporary = mesh_path.with_name(".mesh.tmp.obj")
                mesh.export(temporary, file_type="obj")
                temporary.replace(mesh_path)
                surface = surface_metrics(
                    mesh,
                    target_mesh,
                    count=int(args.surface_samples),
                    seed=metric_seed,
                    thresholds=(0.01, 0.02, 0.05),
                )
                branches[branch] = {
                    "mesh": str(mesh_path),
                    "mesh_sha256": sha256_file(mesh_path),
                    "surface": surface,
                    "structure": structure,
                    "wrapper": wrapper,
                }
                del noise, flow, latent, decoded, mesh
                torch.cuda.empty_cache()
            record = {
                "identity": identity,
                "object_position": position,
                "same_native_ss_coordinates": True,
                "same_initial_noise": True,
                "coord_count": int(initial.coords.shape[0]),
                "target": target_metadata,
                "branches": branches,
            }
            _validate_branch_matrix(record)
            atomic_json(result_path, record)
            records.append(record)
            print(
                f"[objaverse2k_slat_fourway] {position + 1}/"
                f"{args.expected_objects} {object_uid} seed={seed}",
                flush=True,
            )
            del sample, master, initial, condition, target_mesh, branches
            gc.collect()
            torch.cuda.empty_cache()

    report = {
        "format": WORKER_REPORT_FORMAT,
        "passed": len(records) == len(selected) * len(seeds),
        "formal": False,
        "training_overlap": False,
        "worker_index": int(args.worker_index),
        "num_workers": int(args.num_workers),
        "object_start": object_start,
        "object_end": object_end,
        "object_count": len(selected),
        "record_count": len(records),
        "run_config": run_config,
        "model_summary": model_summary,
        "records": records,
    }
    body = dict(report)
    report["report_sha256"] = canonical_json_sha256(body)
    atomic_json(output_dir / "report.json", report)
    model.cpu()
    decoder.cpu()
    gc.collect()
    torch.cuda.empty_cache()


def validate_worker_reports(
    paths: list[str], *, expected_workers: int
) -> tuple[list[dict[str, Any]], dict[tuple[str, str, int], dict[str, Any]]]:
    if len(paths) != int(expected_workers):
        raise RuntimeError("four-way report count differs from expected workers")
    reports = [load_json(Path(path).expanduser().resolve()) for path in paths]
    reports.sort(key=lambda report: int(report.get("worker_index", -1)))
    if [int(report.get("worker_index", -1)) for report in reports] != list(
        range(int(expected_workers))
    ):
        raise RuntimeError("four-way worker indices are incomplete")
    records: dict[tuple[str, str, int], dict[str, Any]] = {}
    ranges = []
    protocol_reference: dict[str, Any] | None = None
    for report in reports:
        body = dict(report)
        claimed = body.pop("report_sha256", "")
        if (
            report.get("format") != WORKER_REPORT_FORMAT
            or report.get("passed") is not True
            or report.get("formal") is not False
            or report.get("training_overlap") is not False
            or canonical_json_sha256(body) != claimed
        ):
            raise RuntimeError("invalid four-way worker report")
        run_config = dict(report.get("run_config", {}))
        for field in ("worker_index", "object_start", "object_end", "selected"):
            run_config.pop(field, None)
        if protocol_reference is None:
            protocol_reference = run_config
        elif run_config != protocol_reference:
            raise RuntimeError("four-way worker protocols differ")
        if (
            int(report.get("object_count", -1))
            != int(report["object_end"]) - int(report["object_start"])
            or int(report.get("record_count", -1))
            != int(report["object_count"])
            * len(report["run_config"].get("joint_seeds", []))
        ):
            raise RuntimeError("four-way worker counts differ")
        ranges.append((int(report["object_start"]), int(report["object_end"])))
        for record in report["records"]:
            _validate_branch_matrix(record)
            identity = record["identity"]
            key = (
                str(identity["object_uid"]),
                str(identity["uid"]),
                int(identity["support_seed"]),
            )
            if key in records:
                raise RuntimeError(f"duplicate four-way record: {key}")
            records[key] = record
    expected_objects = int(reports[0]["run_config"]["expected_objects"])
    expected_ranges = [
        (
            expected_objects * worker // int(expected_workers),
            expected_objects * (worker + 1) // int(expected_workers),
        )
        for worker in range(int(expected_workers))
    ]
    if ranges != expected_ranges:
        raise RuntimeError("four-way worker ranges do not cover dev exactly")
    return reports, records


def aggregate_reports(
    *,
    reports: list[dict[str, Any]],
    records: dict[tuple[str, str, int], dict[str, Any]],
    bootstrap_samples: int,
) -> dict[str, Any]:
    if not reports or not records:
        raise RuntimeError("four-way aggregate requires non-empty reports/records")
    by_object: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seed_rows = []
    for key in sorted(records):
        record = records[key]
        _validate_branch_matrix(record)
        comparisons = {
            label: paired_improvement(
                record["branches"][left], record["branches"][right]
            )
            for left, right, label in COMPARISONS
        }
        row = {
            "object_uid": key[0],
            "uid": key[1],
            "support_seed": key[2],
            "master_noise_seed": record["identity"]["master_noise_seed"],
            "branches": record["branches"],
            "comparisons": comparisons,
        }
        seed_rows.append(row)
        by_object[key[0]].append(row)

    object_rows = []
    for object_uid, rows in sorted(by_object.items()):
        object_rows.append(
            {
                "object_uid": object_uid,
                "seed_count": len(rows),
                "absolute": {
                    branch: {
                        metric: float(
                            np.mean(
                                [
                                    row["branches"][branch][
                                        "surface" if metric in SURFACE_METRICS else "structure"
                                    ][metric]
                                    for row in rows
                                ]
                            )
                        )
                        for metric in ALL_METRICS
                    }
                    for branch in BRANCHES
                },
                "comparisons": {
                    label: {
                        metric: float(
                            np.mean(
                                [row["comparisons"][label][metric] for row in rows]
                            )
                        )
                        for metric in ALL_METRICS
                    }
                    for _, _, label in COMPARISONS
                },
            }
        )

    absolute = {
        branch: {
            metric: summarize(
                [row["absolute"][branch][metric] for row in object_rows],
                bootstrap_samples=int(bootstrap_samples),
                seed=stable_seed("objaverse2k_fourway_absolute", branch, metric),
            )
            for metric in ALL_METRICS
        }
        for branch in BRANCHES
    }
    comparisons = {
        label: {
            "left": left,
            "right": right,
            "positive_means_left_better": True,
            "metrics": {
                metric: summarize(
                    [row["comparisons"][label][metric] for row in object_rows],
                    bootstrap_samples=int(bootstrap_samples),
                    seed=stable_seed("objaverse2k_fourway_comparison", label, metric),
                )
                for metric in ALL_METRICS
            },
        }
        for left, right, label in COMPARISONS
    }
    config = reports[0]["run_config"]
    expected_seeds = len(config["joint_seeds"])
    if any(row["seed_count"] != expected_seeds for row in object_rows):
        raise RuntimeError("four-way object seed coverage is incomplete")
    return {
        "format": AGGREGATE_REPORT_FORMAT,
        "passed": True,
        "formal": False,
        "training_overlap": False,
        "scope": "object-disjoint Objaverse2K development mechanism diagnostic",
        "object_count": len(object_rows),
        "seed_count_per_object": expected_seeds,
        "record_count": len(seed_rows),
        "branch_rollout_count": len(seed_rows) * len(BRANCHES),
        "branches": list(BRANCHES),
        "same_native_ss_coordinates": True,
        "same_initial_noise": True,
        "checkpoint": {
            "path": config["checkpoint"],
            "sha256": config["checkpoint_sha256"],
            "step": config["checkpoint_step"],
            "weights": config["weights"],
        },
        "protocol": {
            key: config[key]
            for key in (
                "cache_manifest_sha256",
                "lifting_cache_manifest_sha256",
                "native_ss_report_sha256",
                "stock_slat_freeze_sha256",
                "sampling",
                "joint_seeds",
                "noise_protocol",
                "noise_seed",
                "surface_samples",
                "expected_objects",
                "num_workers",
            )
        },
        "absolute": absolute,
        "comparisons": comparisons,
        "object_rows": object_rows,
        "records": seed_rows,
        "scope_guard": (
            f"This {len(object_rows)}-object development slice is object-disjoint from "
            "Objaverse2K SLat training, but is used for mechanism diagnosis and "
            "checkpoint selection. It is not the frozen Objaverse16 final diagnostic."
        ),
    }


def command_aggregate(args: argparse.Namespace) -> None:
    report_paths = parse_csv(args.worker_reports)
    reports, records = validate_worker_reports(
        report_paths, expected_workers=int(args.expected_workers)
    )
    report = aggregate_reports(
        reports=reports,
        records=records,
        bootstrap_samples=int(args.bootstrap_samples),
    )
    if report["object_count"] != int(args.expected_objects):
        raise RuntimeError("aggregate object count differs from requested dev slice")
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    body = dict(report)
    report["report_sha256"] = canonical_json_sha256(body)
    atomic_json(output_dir / "report.json", report)

    lines = [
        f"Objaverse2K SLat four-way dev{report['object_count']} attribution",
        "=" * 58,
        "passed: true",
        "formal: false",
        "training overlap: false",
        f"checkpoint step: {report['checkpoint']['step']} weights: {report['checkpoint']['weights']}",
        f"objects: {report['object_count']} seeds/object: {report['seed_count_per_object']}",
        f"matched quartets: {report['record_count']} branch rollouts: {report['branch_rollout_count']}",
        "branches: " + ", ".join(BRANCHES),
        "same Native-SS coordinates/noise inside every quartet: true",
        "all reported deltas: positive means the left branch is better",
        "",
    ]
    for _, _, label in COMPARISONS:
        comparison = report["comparisons"][label]
        metrics = comparison["metrics"]
        chamfer = metrics["chamfer_l1"]
        fscore = metrics["fscore_0p02"]
        normal = metrics["normal_consistency"]
        lines.append(
            f"{label} ({comparison['left']} - {comparison['right']}): "
            f"chamfer={chamfer['mean']:+.8f} median={chamfer['median']:+.8f} "
            f"win={chamfer['positive_rate']:.4f} "
            f"CI={chamfer['bootstrap_mean_95_ci']}; "
            f"f@0.02={fscore['mean']:+.8f}; normal={normal['mean']:+.8f}"
        )
    lines.extend(("", report["scope_guard"]))
    (output_dir / "summary.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print("\n".join(lines), flush=True)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    worker = subparsers.add_parser("worker")
    worker.add_argument("--cache_manifest", required=True)
    worker.add_argument("--lifting_cache_manifest", required=True)
    worker.add_argument("--checkpoint", required=True)
    worker.add_argument("--native_ss_report", required=True)
    worker.add_argument("--stock_slat_freeze", required=True)
    worker.add_argument("--output_dir", required=True)
    worker.add_argument("--pretrained", default="Stable-X/trellis-vggt-v0-2")
    worker.add_argument("--weights", choices=("ema", "raw"), default="ema")
    worker.add_argument("--joint_seeds", default="42,43,44")
    worker.add_argument("--noise_seed", type=int, default=20260811)
    worker.add_argument("--worker_index", type=int, required=True)
    worker.add_argument("--num_workers", type=int, default=8)
    worker.add_argument("--expected_objects", type=int, default=16)
    worker.add_argument("--surface_samples", type=int, default=20000)
    worker.add_argument("--amp_dtype", choices=("bf16", "fp16", "none"), default="bf16")
    worker.add_argument("--resume", action="store_true")
    worker.set_defaults(handler=command_worker)

    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--worker_reports", required=True)
    aggregate.add_argument("--output_dir", required=True)
    aggregate.add_argument("--expected_workers", type=int, default=8)
    aggregate.add_argument("--expected_objects", type=int, default=16)
    aggregate.add_argument("--bootstrap_samples", type=int, default=10000)
    aggregate.set_defaults(handler=command_aggregate)
    return parser


def main() -> None:
    args = make_parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
