#!/usr/bin/env python3
"""Four-GPU matched Stock/M8/Objaverse2K SLat development evaluation."""

from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import nullcontext
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
import torch
from torch import nn

from pose_point_depth_mv.eval_direct_slat_flow import summarize
from pose_point_depth_mv.export_direct_flow_mesh_pairs import (
    canonical_coords,
    load_canonical_gt,
    mesh_structure_metrics,
    sparse_noise_from_master,
    surface_metrics,
)
from pose_point_depth_mv.export_native_slat_mesh_pairs import select_matrix
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
from pose_point_depth_mv.omni_real_benchmark_common import sha256_file
from pose_point_depth_mv.objaverse2k_slat_pipeline import (
    resolve_native_objaverse_normalization_bindings,
)
from pose_point_depth_mv.render_direct_slat_fourway import (
    LATENT_DECODER_TO_REFERENCE,
)
from pose_point_depth_mv.train_direct_slat_flow import to_device_tree
from trellis.modules import sparse as sp


WORKER_REPORT_FORMAT = "pose_point_depth_mv.objaverse2k_no_vggt_slat_worker.v1"
AGGREGATE_REPORT_FORMAT = "pose_point_depth_mv.objaverse2k_no_vggt_slat_comparison.v1"
MODEL_LABELS = (
    "m8",
    "objaverse2k",
    "m8_step800",
    "stockinit_objaverse2k",
)
BRANCHES = ("stock", "full")
SURFACE_METRICS = (
    "chamfer_l1",
    "chamfer_l2",
    "fscore_0p01",
    "fscore_0p02",
    "fscore_0p05",
    "normal_consistency",
)
STRUCTURE_METRICS = ("largest_component_ratio", "component_count")
STOCK_REPRODUCTION_TOLERANCES = {
    "chamfer_l1": 1.0e-3,
    "chamfer_l2": 2.0e-4,
    "fscore_0p01": 1.0e-2,
    "fscore_0p02": 1.0e-2,
    "fscore_0p05": 1.0e-2,
    "normal_consistency": 1.0e-2,
}


def parse_csv(value: str) -> list[str]:
    result = [item.strip() for item in str(value).split(",") if item.strip()]
    if not result or len(result) != len(set(result)):
        raise ValueError("CSV values must be non-empty and unique")
    return result


def parse_int_csv(value: str) -> list[int]:
    result = [int(item) for item in parse_csv(value)]
    if len(result) != len(set(result)):
        raise ValueError("integer CSV values must be unique")
    return result


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def upstream_binding(value: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "report",
        "report_sha256",
        "checkpoint",
        "checkpoint_sha256",
        "checkpoint_step",
        "weights",
        "cfg_strength",
        "steps",
        "cfg_interval",
        "guidance_rescale",
        "rescale_t",
    )
    binding = {key: value[key] for key in keys}
    if value.get("exploratory") is True:
        binding.update(
            {
                key: value[key]
                for key in (
                    "formal",
                    "exploratory",
                    "diagnostic_scope",
                    "failed_quality_checks",
                    "source_calibration",
                    "source_calibration_sha256",
                    "scope_guard",
                )
            }
        )
    return binding


def stable_seed(namespace: str, *values: Any) -> int:
    text = "\0".join([namespace, *(str(value) for value in values)])
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:4], "big")


def decode_mesh_fp32(
    decoder: nn.Module,
    latent: Any,
    *,
    std: torch.Tensor,
    mean: torch.Tensor,
) -> Any:
    with torch.autocast(device_type=latent.device.type, enabled=False):
        return decoder((latent * std + mean).float())[0]


def sampling_contract(defaults: dict[str, Any]) -> dict[str, Any]:
    params = dict(defaults)
    params.setdefault("guidance_rescale", 0.0)
    contract = {
        "steps": int(params.get("steps", -1)),
        "cfg_strength": float(params.get("cfg_strength", -1)),
        "cfg_interval": [float(value) for value in params.get("cfg_interval", ())],
        "guidance_rescale": float(params.get("guidance_rescale", -1)),
        "rescale_t": float(params.get("rescale_t", -1)),
    }
    expected = {
        "steps": 25,
        "cfg_strength": 5.0,
        "cfg_interval": [0.5, 1.0],
        "guidance_rescale": 0.0,
        "rescale_t": 3.0,
    }
    if contract != expected:
        raise RuntimeError(f"Stock SLat sampler defaults changed: {contract}")
    return {**params, **contract}


def select_worker_matrix(
    rows: list[dict[str, Any]],
    *,
    seeds: list[int],
    worker_index: int,
    num_workers: int,
    expected_objects: int,
    object_selection_seed: int | None = None,
) -> tuple[list[tuple[str, str, dict[int, int]]], int, int]:
    if object_selection_seed is None:
        selected = select_matrix(
            rows,
            joint_seeds=seeds,
            max_objects=int(expected_objects),
            object_offset=0,
            require_complete=True,
        )
    else:
        eligible = select_matrix(
            rows,
            joint_seeds=seeds,
            max_objects=0,
            object_offset=0,
            require_complete=False,
        )
        eligible.sort(
            key=lambda row: hashlib.sha256(
                f"objaverse2k_train_eval\0{int(object_selection_seed)}\0{row[0]}\0{row[1]}".encode()
            ).hexdigest()
        )
        selected = eligible[: int(expected_objects)]
        if len(selected) != int(expected_objects):
            raise RuntimeError(
                f"incomplete hashed object selection: {len(selected)} != {expected_objects}"
            )
    if int(num_workers) <= 0 or not 0 <= int(worker_index) < int(num_workers):
        raise ValueError("worker_index must lie in [0,num_workers)")
    start = len(selected) * int(worker_index) // int(num_workers)
    end = len(selected) * (int(worker_index) + 1) // int(num_workers)
    return selected[start:end], start, end


def record_identity(
    *,
    model_label: str,
    checkpoint_sha256: str,
    object_uid: str,
    uid: str,
    support_seed: int,
    master_noise_seed: int,
    metric_seed: int,
    cache_manifest_sha256: str,
) -> dict[str, Any]:
    return {
        "model_label": model_label,
        "checkpoint_sha256": checkpoint_sha256,
        "object_uid": object_uid,
        "uid": uid,
        "support_seed": int(support_seed),
        "master_noise_seed": int(master_noise_seed),
        "metric_seed": int(metric_seed),
        "cache_manifest_sha256": cache_manifest_sha256,
    }


def command_worker(args: argparse.Namespace) -> None:
    seeds = parse_int_csv(args.joint_seeds)
    if str(args.model_label) not in MODEL_LABELS:
        raise ValueError(f"unsupported model label={args.model_label!r}")
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
        raise RuntimeError("evaluation requires a no-VGGT Native-SS cache")
    selected, object_start, object_end = select_worker_matrix(
        dataset.rows,
        seeds=seeds,
        worker_index=int(args.worker_index),
        num_workers=int(args.num_workers),
        expected_objects=int(args.expected_objects),
        object_selection_seed=args.object_selection_seed,
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
        raise RuntimeError("development evaluation requires a trained no-VGGT checkpoint")
    validate_native_slat_no_vggt_checkpoint(
        checkpoint,
        pretrained=args.pretrained,
        stock_slat_freeze=stock_freeze,
        upstream_native_ss=ss_binding,
        allow_v2_parent=False,
    )
    if checkpoint.get("args", {}).get("architecture") != "v2":
        raise RuntimeError("Objaverse2K comparison freezes Native-SLat architecture v2")
    if checkpoint.get("args", {}).get("stock_context_views", "all") != "all":
        raise RuntimeError("Objaverse2K comparison requires all-view Stock context")
    if args.training_overlap:
        identity = dict(checkpoint.get("data_identity", {}))
        if (
            Path(str(identity.get("cache_manifest", ""))).resolve() != cache_path
            or identity.get("cache_manifest_sha256") != sha256_file(cache_path)
        ):
            raise RuntimeError("training-overlap evaluation cache is not checkpoint training data")
        training_objects = set(map(str, identity.get("object_uids", [])))
        selected_objects = {object_uid for object_uid, _, _ in selected}
        if not selected_objects or not selected_objects.issubset(training_objects):
            raise RuntimeError("training-overlap selection contains a non-training object")

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
        "scope": "object-disjoint Objaverse2K development checkpoint selection",
        "model_label": str(args.model_label),
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
            {"object_uid": object_uid, "uid": uid}
            for object_uid, uid, _ in selected
        ],
    }
    if args.fixed_axis_evaluation:
        run_config["coordinate_evaluation"] = {
            "mapping": "(x,y,z) -> (x,z,-y)",
            "decoder_to_source_axis_transform": LATENT_DECODER_TO_REFERENCE.tolist(),
            "applied_identically_to_stock_and_full": True,
            "alignment": "fixed proper axis transform only; no ICP/scale/GT fit",
        }
    if args.training_overlap:
        train_scope = (
            f"Objaverse2K train{int(args.expected_objects)} Stock-vs-Full "
            "fitting diagnostic"
        )
        run_config.update(
            {
                "scope": train_scope,
                "training_overlap": True,
                "training_object_disjoint": False,
                "source_mesh_disjoint": False,
                "object_selection_seed": int(args.object_selection_seed),
                "object_selection_rule": (
                    "SHA256(objaverse2k_train_eval, seed, object_uid, uid), "
                    "then first expected_objects"
                ),
            }
        )
    output_dir = Path(args.output_dir).expanduser().resolve()
    config_path = output_dir / "run_config.json"
    if output_dir.exists():
        if not args.resume:
            raise FileExistsError(output_dir)
        if not config_path.is_file() or load_json(config_path) != run_config:
            raise RuntimeError("evaluation resume protocol differs")
    else:
        output_dir.mkdir(parents=True)
        atomic_json(config_path, run_config)

    amp_enabled = args.amp_dtype != "none"
    amp_dtype = torch.float16 if args.amp_dtype == "fp16" else torch.bfloat16
    records = []
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
                model_label=str(args.model_label),
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
                for branch in BRANCHES:
                    mesh_path = Path(existing["branches"][branch]["mesh"])
                    if sha256_file(mesh_path) != existing["branches"][branch]["mesh_sha256"]:
                        raise RuntimeError(f"resumed mesh changed: {mesh_path}")
                records.append(existing)
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
                canonical_margin_binding=normalization_bindings.get(
                    str(ss_latent_path)
                ),
            )
            branches = {}
            for branch in BRANCHES:
                noise = sp.SparseTensor(
                    feats=initial.feats.clone(), coords=initial.coords.clone()
                )
                wrapper = None
                if branch == "stock":
                    flow: nn.Module = NativeSLatStockFlow(model)
                else:
                    flow = NativeSLatCalibratedCFGFlow(
                        model,
                        condition["cond"],
                        sample["lifting_sample"],
                        enabled=True,
                    )
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
                if branch == "full":
                    wrapper = flow.summary()  # type: ignore[attr-defined]
                    if wrapper["positive_calls"] <= 0 or wrapper["negative_calls"] <= 0:
                        raise RuntimeError("Full SLat did not execute both CFG branches")
                decoded = decode_mesh_fp32(decoder, latent, std=std, mean=mean)
                mesh = decoded.to_trimesh(transform_pose=False)
                if args.fixed_axis_evaluation:
                    mesh.apply_transform(LATENT_DECODER_TO_REFERENCE)
                structure = mesh_structure_metrics(mesh)
                if not structure["mesh_success"]:
                    raise RuntimeError(f"empty/non-finite mesh: {object_uid} {branch}")
                mesh_path = output_dir / "meshes" / object_uid / f"seed_{seed}" / branch / "mesh.obj"
                mesh_path.parent.mkdir(parents=True, exist_ok=True)
                temporary = mesh_path.with_name(".mesh.tmp.obj")
                mesh.export(temporary, file_type="obj")
                os.replace(temporary, mesh_path)
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
            atomic_json(result_path, record)
            records.append(record)
            print(
                f"[objaverse2k_slat_eval] {args.model_label} "
                f"{position + 1}/{args.expected_objects} {object_uid} seed={seed}",
                flush=True,
            )
            del sample, master, initial, condition, target_mesh, branches
            gc.collect()
            torch.cuda.empty_cache()
    report = {
        "format": WORKER_REPORT_FORMAT,
        "passed": len(records) == len(selected) * len(seeds),
        "formal": False,
        "model_label": str(args.model_label),
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
    if args.training_overlap:
        report.update(
            {
                "training_overlap": True,
                "training_object_disjoint": False,
                "source_mesh_disjoint": False,
                "scope": (
                    f"Objaverse2K train{int(args.expected_objects)} Stock-vs-Full "
                    "fitting diagnostic"
                ),
            }
        )
    body = dict(report)
    report["report_sha256"] = canonical_json_sha256(body)
    atomic_json(output_dir / "report.json", report)
    model.cpu()
    decoder.cpu()
    gc.collect()
    torch.cuda.empty_cache()


def paired_improvement(left: dict[str, Any], right: dict[str, Any]) -> dict[str, float]:
    return {
        "chamfer_l1": float(right["surface"]["chamfer_l1"] - left["surface"]["chamfer_l1"]),
        "chamfer_l2": float(right["surface"]["chamfer_l2"] - left["surface"]["chamfer_l2"]),
        "fscore_0p01": float(left["surface"]["fscore_0p01"] - right["surface"]["fscore_0p01"]),
        "fscore_0p02": float(left["surface"]["fscore_0p02"] - right["surface"]["fscore_0p02"]),
        "fscore_0p05": float(left["surface"]["fscore_0p05"] - right["surface"]["fscore_0p05"]),
        "normal_consistency": float(
            left["surface"]["normal_consistency"] - right["surface"]["normal_consistency"]
        ),
        "largest_component_ratio": float(
            left["structure"]["largest_component_ratio"]
            - right["structure"]["largest_component_ratio"]
        ),
        "component_count": float(
            right["structure"]["component_count"] - left["structure"]["component_count"]
        ),
    }


def absolute_branch_differences(
    left: dict[str, Any], right: dict[str, Any]
) -> dict[str, float]:
    return {
        metric: abs(float(left["surface"][metric]) - float(right["surface"][metric]))
        for metric in SURFACE_METRICS
    } | {
        metric: abs(
            float(left["structure"][metric]) - float(right["structure"][metric])
        )
        for metric in STRUCTURE_METRICS
    }


def validate_worker_reports(
    paths: list[str], *, model_label: str, expected_workers: int
) -> tuple[list[dict[str, Any]], dict[tuple[str, str, int], dict[str, Any]]]:
    if len(paths) != int(expected_workers):
        raise RuntimeError(f"{model_label} report count differs from expected workers")
    reports = [load_json(Path(path).expanduser().resolve()) for path in paths]
    reports.sort(key=lambda report: int(report.get("worker_index", -1)))
    if [int(report.get("worker_index", -1)) for report in reports] != list(
        range(int(expected_workers))
    ):
        raise RuntimeError(f"{model_label} worker indices are incomplete")
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
            or report.get("model_label") != model_label
            or canonical_json_sha256(body) != claimed
        ):
            raise RuntimeError(f"invalid {model_label} worker report")
        run_config = dict(report.get("run_config", {}))
        for field in ("worker_index", "object_start", "object_end", "selected"):
            run_config.pop(field, None)
        if protocol_reference is None:
            protocol_reference = run_config
        elif run_config != protocol_reference:
            raise RuntimeError(f"{model_label} worker protocols differ")
        if (
            int(report.get("object_count", -1))
            != int(report["object_end"]) - int(report["object_start"])
            or int(report.get("record_count", -1))
            != int(report["object_count"])
            * len(report["run_config"].get("joint_seeds", []))
        ):
            raise RuntimeError(f"{model_label} worker counts differ")
        ranges.append((int(report["object_start"]), int(report["object_end"])))
        for row in report["records"]:
            identity = row["identity"]
            key = (
                str(identity["object_uid"]),
                str(identity["uid"]),
                int(identity["support_seed"]),
            )
            if key in records:
                raise RuntimeError(f"duplicate {model_label} record: {key}")
            records[key] = row
    if ranges != [
        (
            int(reports[0]["run_config"]["expected_objects"]) * worker // int(expected_workers),
            int(reports[0]["run_config"]["expected_objects"]) * (worker + 1) // int(expected_workers),
        )
        for worker in range(int(expected_workers))
    ]:
        raise RuntimeError(f"{model_label} worker ranges do not cover dev exactly")
    return reports, records


def aggregate_reports(
    *,
    m8_reports: list[dict[str, Any]],
    m8_records: dict[tuple[str, str, int], dict[str, Any]],
    candidate_reports: list[dict[str, Any]],
    candidate_records: dict[tuple[str, str, int], dict[str, Any]],
    bootstrap_samples: int,
    stock_reproduction_tolerances: dict[str, float] | None = None,
) -> dict[str, Any]:
    effective_stock_tolerances = dict(STOCK_REPRODUCTION_TOLERANCES)
    if stock_reproduction_tolerances is not None:
        unknown = set(stock_reproduction_tolerances) - set(effective_stock_tolerances)
        if unknown:
            raise ValueError(f"unknown Stock reproduction tolerances: {sorted(unknown)}")
        for metric, tolerance in stock_reproduction_tolerances.items():
            if not math.isfinite(float(tolerance)) or float(tolerance) < 0.0:
                raise ValueError(f"invalid Stock reproduction tolerance: {metric}={tolerance}")
            effective_stock_tolerances[metric] = float(tolerance)
    if set(m8_records) != set(candidate_records):
        raise RuntimeError("M8/candidate evaluation matrices differ")
    invariant_fields = (
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
    left_config = m8_reports[0]["run_config"]
    right_config = candidate_reports[0]["run_config"]
    mismatch = {
        field: (left_config.get(field), right_config.get(field))
        for field in invariant_fields
        if left_config.get(field) != right_config.get(field)
    }
    if mismatch:
        raise RuntimeError(f"M8/candidate evaluation protocols differ: {mismatch}")
    paired_rows = []
    by_object: dict[str, list[dict[str, Any]]] = defaultdict(list)
    stock_reproduction_rows = []
    for key in sorted(m8_records):
        m8 = m8_records[key]
        candidate = candidate_records[key]
        left_identity = dict(m8["identity"])
        right_identity = dict(candidate["identity"])
        for field in (
            "model_label",
            "checkpoint_sha256",
        ):
            left_identity.pop(field, None)
            right_identity.pop(field, None)
        if left_identity != right_identity:
            raise RuntimeError(f"paired identity differs: {key}")
        m8_stock = m8["branches"]["stock"]
        candidate_stock = candidate["branches"]["stock"]
        stock_differences = absolute_branch_differences(m8_stock, candidate_stock)
        failed_reproduction = {
            metric: {
                "difference": stock_differences[metric],
                "tolerance": tolerance,
            }
            for metric, tolerance in effective_stock_tolerances.items()
            if stock_differences[metric] > tolerance
        }
        if failed_reproduction:
            raise RuntimeError(
                f"Stock numerical reproduction exceeds tolerance: {key} "
                f"{failed_reproduction}"
            )
        stock_reproduction_rows.append(
            {
                "object_uid": key[0],
                "uid": key[1],
                "support_seed": key[2],
                "mesh_exact_match": (
                    m8_stock["mesh_sha256"] == candidate_stock["mesh_sha256"]
                ),
                "m8_stock_mesh_sha256": m8_stock["mesh_sha256"],
                "objaverse2k_stock_mesh_sha256": candidate_stock["mesh_sha256"],
                "absolute_differences": stock_differences,
            }
        )
        row = {
            "object_uid": key[0],
            "uid": key[1],
            "support_seed": key[2],
            "master_noise_seed": left_identity["master_noise_seed"],
            "stock_mesh_sha256": m8_stock["mesh_sha256"],
            "objaverse2k_stock_mesh_sha256": candidate_stock["mesh_sha256"],
            "branches": {
                "stock": m8_stock,
                "objaverse2k_stock": candidate_stock,
                "m8": m8["branches"]["full"],
                "objaverse2k": candidate["branches"]["full"],
            },
        }
        row["comparisons"] = {
            "m8_vs_stock": paired_improvement(row["branches"]["m8"], row["branches"]["stock"]),
            "objaverse2k_vs_stock": paired_improvement(
                row["branches"]["objaverse2k"],
                row["branches"]["objaverse2k_stock"],
            ),
            "objaverse2k_vs_m8": paired_improvement(
                row["branches"]["objaverse2k"], row["branches"]["m8"]
            ),
        }
        paired_rows.append(row)
        by_object[key[0]].append(row)
    object_rows = []
    for object_uid, rows in sorted(by_object.items()):
        object_rows.append(
            {
                "object_uid": object_uid,
                "seed_count": len(rows),
                "comparisons": {
                    comparison: {
                        metric: float(
                            np.mean([row["comparisons"][comparison][metric] for row in rows])
                        )
                        for metric in (*SURFACE_METRICS, *STRUCTURE_METRICS)
                    }
                    for comparison in (
                        "m8_vs_stock",
                        "objaverse2k_vs_stock",
                        "objaverse2k_vs_m8",
                    )
                },
            }
        )
    summary = {
        comparison: {
            metric: summarize(
                [row["comparisons"][comparison][metric] for row in object_rows],
                bootstrap_samples=int(bootstrap_samples),
                seed=stable_seed("objaverse2k_summary", comparison, metric),
            )
            for metric in (*SURFACE_METRICS, *STRUCTURE_METRICS)
        }
        for comparison in (
            "m8_vs_stock",
            "objaverse2k_vs_stock",
            "objaverse2k_vs_m8",
        )
    }
    stock_reproduction = {
        "passed": True,
        "mesh_exact_match_count": sum(
            bool(row["mesh_exact_match"]) for row in stock_reproduction_rows
        ),
        "record_count": len(stock_reproduction_rows),
        "tolerances": effective_stock_tolerances,
        "tolerance_overrides": {
            metric: tolerance
            for metric, tolerance in effective_stock_tolerances.items()
            if tolerance != STOCK_REPRODUCTION_TOLERANCES[metric]
        },
        "absolute_difference": {
            metric: {
                "mean": float(
                    np.mean(
                        [
                            row["absolute_differences"][metric]
                            for row in stock_reproduction_rows
                        ]
                    )
                ),
                "max": float(
                    np.max(
                        [
                            row["absolute_differences"][metric]
                            for row in stock_reproduction_rows
                        ]
                    )
                ),
            }
            for metric in (*SURFACE_METRICS, *STRUCTURE_METRICS)
        },
        "rows": stock_reproduction_rows,
        "interpretation": (
            "The two frozen Stock branches use identical coordinates and initial noise, "
            "but independent CUDA sparse/decoder runs are not byte deterministic. Core "
            "surface metrics must remain within the frozen numerical tolerances."
        ),
    }
    return {
        "format": AGGREGATE_REPORT_FORMAT,
        "passed": True,
        "formal": False,
        "training_overlap": False,
        "scope": "object-disjoint Objaverse2K dev64 checkpoint selection",
        "object_count": len(object_rows),
        "record_count": len(paired_rows),
        "branches": ["stock", "objaverse2k_stock", "m8", "objaverse2k"],
        "same_native_ss_coordinates": True,
        "same_initial_noise": True,
        "stock_mesh_exact_match_across_checkpoint_runs": (
            stock_reproduction["mesh_exact_match_count"]
            == stock_reproduction["record_count"]
        ),
        "stock_numerical_reproduction": stock_reproduction,
        "protocol": {
            field: left_config[field] for field in invariant_fields
        },
        "checkpoints": {
            "m8": {
                "path": left_config["checkpoint"],
                "sha256": left_config["checkpoint_sha256"],
                "step": left_config["checkpoint_step"],
                "weights": left_config["weights"],
            },
            "objaverse2k": {
                "path": right_config["checkpoint"],
                "sha256": right_config["checkpoint_sha256"],
                "step": right_config["checkpoint_step"],
                "weights": right_config["weights"],
            },
        },
        "summary": summary,
        "object_rows": object_rows,
        "records": paired_rows,
        "scope_guard": (
            f"This {len(object_rows)}-object development slice is object-disjoint from "
            "Objaverse2K SLat training but is used for checkpoint selection. It is not "
            "the frozen Objaverse16 final diagnostic."
        ),
    }


def command_aggregate(args: argparse.Namespace) -> None:
    m8_paths = parse_csv(args.m8_reports)
    candidate_paths = parse_csv(args.objaverse2k_reports)
    m8_reports, m8_records = validate_worker_reports(
        m8_paths, model_label="m8", expected_workers=int(args.expected_workers)
    )
    candidate_reports, candidate_records = validate_worker_reports(
        candidate_paths,
        model_label="objaverse2k",
        expected_workers=int(args.expected_workers),
    )
    tolerance_overrides = {}
    if args.stock_normal_tolerance is not None:
        tolerance_overrides["normal_consistency"] = float(
            args.stock_normal_tolerance
        )
    report = aggregate_reports(
        m8_reports=m8_reports,
        m8_records=m8_records,
        candidate_reports=candidate_reports,
        candidate_records=candidate_records,
        bootstrap_samples=int(args.bootstrap_samples),
        stock_reproduction_tolerances=tolerance_overrides or None,
    )
    if report["object_count"] != int(args.expected_objects):
        raise RuntimeError("aggregate object count differs from the requested development slice")
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    body = dict(report)
    report["report_sha256"] = canonical_json_sha256(body)
    atomic_json(output_dir / "report.json", report)
    lines = [
        f"Objaverse2K no-VGGT SLat matched dev{report['object_count']} comparison",
        "=" * 57,
        "passed: true",
        "formal: false",
        "training overlap: false",
        f"objects: {report['object_count']} records: {report['record_count']}",
        "branches: stock(m8 run), stock(objaverse2k run), m8, objaverse2k",
        "same Native-SS coordinates/noise: true",
        "Stock numerical reproduction within tolerance: true",
        (
            "Stock exact OBJ matches: "
            f"{report['stock_numerical_reproduction']['mesh_exact_match_count']}/"
            f"{report['stock_numerical_reproduction']['record_count']}"
        ),
        (
            "Stock tolerance overrides: "
            f"{report['stock_numerical_reproduction']['tolerance_overrides']}"
        ),
        "",
    ]
    for comparison in ("m8_vs_stock", "objaverse2k_vs_stock", "objaverse2k_vs_m8"):
        metrics = report["summary"][comparison]
        chamfer = metrics["chamfer_l1"]
        fscore = metrics["fscore_0p02"]
        normal = metrics["normal_consistency"]
        lines.append(
            f"{comparison}: chamfer={chamfer['mean']:+.8f} "
            f"median={chamfer['median']:+.8f} win={chamfer['positive_rate']:.4f}; "
            f"f@0.02={fscore['mean']:+.8f}; normal={normal['mean']:+.8f}"
        )
    lines.extend(("", report["scope_guard"]))
    (output_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    worker = subparsers.add_parser("worker")
    worker.add_argument("--cache_manifest", required=True)
    worker.add_argument("--lifting_cache_manifest", required=True)
    worker.add_argument("--checkpoint", required=True)
    worker.add_argument("--model_label", choices=MODEL_LABELS, required=True)
    worker.add_argument("--native_ss_report", required=True)
    worker.add_argument("--stock_slat_freeze", required=True)
    worker.add_argument("--output_dir", required=True)
    worker.add_argument("--pretrained", default="Stable-X/trellis-vggt-v0-2")
    worker.add_argument("--weights", choices=("ema", "raw"), default="ema")
    worker.add_argument("--joint_seeds", default="42,43,44")
    worker.add_argument("--noise_seed", type=int, default=20260811)
    worker.add_argument("--worker_index", type=int, required=True)
    worker.add_argument("--num_workers", type=int, default=2)
    worker.add_argument("--expected_objects", type=int, default=64)
    worker.add_argument("--training_overlap", action="store_true")
    worker.add_argument("--object_selection_seed", type=int)
    worker.add_argument("--fixed_axis_evaluation", action="store_true")
    worker.add_argument("--surface_samples", type=int, default=20000)
    worker.add_argument("--amp_dtype", choices=("bf16", "fp16", "none"), default="bf16")
    worker.add_argument("--resume", action="store_true")
    worker.set_defaults(handler=command_worker)

    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--m8_reports", required=True)
    aggregate.add_argument("--objaverse2k_reports", required=True)
    aggregate.add_argument("--output_dir", required=True)
    aggregate.add_argument("--expected_workers", type=int, default=2)
    aggregate.add_argument("--expected_objects", type=int, default=64)
    aggregate.add_argument("--bootstrap_samples", type=int, default=10000)
    aggregate.add_argument("--stock_normal_tolerance", type=float)
    aggregate.set_defaults(handler=command_aggregate)
    return parser


def main() -> None:
    args = make_parser().parse_args()
    if getattr(args, "training_overlap", False) and args.object_selection_seed is None:
        raise ValueError("--training_overlap requires --object_selection_seed")
    args.handler(args)


if __name__ == "__main__":
    main()
