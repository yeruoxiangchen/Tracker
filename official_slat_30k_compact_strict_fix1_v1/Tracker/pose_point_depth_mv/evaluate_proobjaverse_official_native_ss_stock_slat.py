#!/usr/bin/env python3
"""Evaluate Native SS on official ProObjaverse support and frozen Stock SLat.

The official package provides SLat labels, not a separately released SS latent.
Consequently this evaluator deliberately avoids a fabricated SS-latent MSE.  It
compares Stock SS and Native SS in two observable spaces instead:

1. decoded 64^3 occupancy against the coordinates of the official SLat label;
2. Mesh geometry after both predicted supports pass through the same frozen
   Stock SLat Flow, sampler, image condition, decoder, and coordinate-keyed
   initial noise.

The worker command is resumable and shardable.  The aggregate command is CPU
only and makes the development decision about whether SS retraining is needed.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import gc
import hashlib
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("SPCONV_ALGO", "native")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
os.environ.setdefault("TORCH_EXTENSIONS_DIR", "/tmp/torch_extensions")

import numpy as np
import torch
import trimesh

from pose_point_depth_mv.dino_only_condition import (
    validate_dino_only_lifting_contract,
)
from pose_point_depth_mv.eval_direct_flow import decode_coords, overlap_metrics
from pose_point_depth_mv.evaluate_native_ss_genrecon import sampling_params
from pose_point_depth_mv.evaluate_native_ss_stock_slat_mesh import (
    aggregate_transfer_records,
    make_sampling_namespace,
    summary_with_ci,
    write_json,
)
from pose_point_depth_mv.export_direct_flow_mesh_pairs import (
    canonical_coords,
    mesh_structure_metrics,
    shared_noise_audit,
    sparse_noise_from_master,
    surface_metrics,
)
from pose_point_depth_mv.native_3d_condition import NativeConditionSLatDataset
from pose_point_depth_mv.native_slat_genrecon import (
    NativeSLatCalibratedCFGFlow,
    NativeSLatStockFlow,
    load_stock_slat_freeze,
    validate_runtime_stock_slat,
)
from pose_point_depth_mv.native_slat_genrecon_no_vggt import (
    build_native_slat_no_vggt_components,
    validate_native_slat_no_vggt_checkpoint,
)
from pose_point_depth_mv.native_slat_genrecon_v2 import (
    load_trainable_state_dict as load_slat_trainable_state_dict,
)
from pose_point_depth_mv.native_ss_genrecon import (
    NativeSSCalibratedCFGFlow,
    NativeSSStockFlow,
    load_trainable_state_dict,
    require_disjoint_object_uids,
    sha256_file,
)
from pose_point_depth_mv.native_ss_genrecon_no_vggt import (
    build_native_ss_no_vggt_components,
    validate_native_ss_no_vggt_checkpoint,
)
from pose_point_depth_mv.proobjaverse_official_ss import (
    load_official_native_ss_deployment,
)
from pose_point_depth_mv.proobjaverse_official_slat_protocol import canonical_sha256
from pose_point_depth_mv.train_direct_flow import install_unused_model_stubs
from pose_point_depth_mv.train_direct_slat_flow import to_device_tree
from trellis.modules import sparse as sp


WORKER_FORMAT = (
    "pose_point_depth_mv.proobjaverse_official_native_ss_stock_slat_worker.v1"
)
REPORT_FORMAT = (
    "pose_point_depth_mv.proobjaverse_official_native_ss_stock_slat_eval.v1"
)
END_TO_END_WORKER_FORMAT = (
    "pose_point_depth_mv.proobjaverse_official_native_ss_slat_end_to_end_worker.v1"
)
END_TO_END_REPORT_FORMAT = (
    "pose_point_depth_mv.proobjaverse_official_native_ss_slat_end_to_end_eval.v1"
)
OCCUPANCY_METRICS = ("iou_gain", "precision_gain", "recall_gain")
# The frozen decoder expands each input SLat point to as many as 64 active
# features at its high-resolution stage.  spconv 2.2/2.3 uses an int32 GEMM
# byte-count guard, so fp16/bf16 with 192 channels must stay below roughly
# 87k input points.  Keep a deterministic safety margin and treat larger
# generated supports as undecodable model outputs.
MAX_SAFE_SLAT_DECODER_INPUT_POINTS = 80_000


def parse_csv(value: str, cast) -> list[Any]:
    result = [cast(item.strip()) for item in str(value).split(",") if item.strip()]
    if not result or len(result) != len(set(result)):
        raise ValueError("CSV values must be non-empty and unique")
    return result


def pair_id(object_uid: str, seed: int) -> str:
    digest = hashlib.sha256(f"{object_uid}|{int(seed)}".encode("utf-8")).hexdigest()
    return digest[:24]


def _official_target_contract(dataset: NativeConditionSLatDataset) -> dict[str, Any]:
    target = dict(dataset.config.get("target_source", {}))
    if target.get("support_policy") != "official_gt_slat_coordinates":
        raise RuntimeError("cache is not the official GT-SLat-support protocol")
    if str(target.get("split", "")) != "dev":
        raise RuntimeError("this evaluator requires the frozen official dev split")
    if int(target.get("coordinate_resolution", -1)) != 64:
        raise RuntimeError("official target coordinates are not on the 64^3 grid")
    if not str(target.get("protocol_sha256", "")):
        raise RuntimeError("official target protocol hash is missing")
    validate_dino_only_lifting_contract(dataset.lifting)
    return target


def _validate_ss_evidence_domain(
    payload: dict[str, Any],
    *,
    target_contract: dict[str, Any],
    pretrained: str,
) -> None:
    domain = payload.get("official_ss_domain_contract")
    if not isinstance(domain, dict):
        raise RuntimeError("Native SS evidence lacks the official domain contract")
    if (
        str(domain.get("official_slat_protocol_sha256", ""))
        != str(target_contract.get("protocol_sha256", ""))
    ):
        raise RuntimeError("Native SS evidence/SLat cache official protocols differ")
    if str(domain.get("ss_decoder_pretrained", "")) != str(pretrained):
        raise RuntimeError("Native SS evidence uses a different Stock decoder")


def _load_ss_runtime(
    args: argparse.Namespace,
    dataset: NativeConditionSLatDataset,
    selected_indices: list[int],
    target_contract: dict[str, Any],
    device: torch.device,
):
    ss_report, binding = load_official_native_ss_deployment(args.native_ss_report)
    if ss_report.get("passed") is not True:
        raise RuntimeError("Native SS deployment report did not pass")
    _validate_ss_evidence_domain(
        ss_report,
        target_contract=target_contract,
        pretrained=str(args.pretrained),
    )
    evidence_uids = {str(value) for value in ss_report["object_uids"]}
    selected_uids = {str(dataset.rows[index]["object_uid"]) for index in selected_indices}
    if not selected_uids or not selected_uids.issubset(evidence_uids):
        raise RuntimeError(
            "Mesh-transfer objects are not covered by the Native SS held-out report"
        )
    if str(args.weights) != str(binding["weights"]):
        raise RuntimeError("worker weights differ from the frozen Native SS report")
    if str(args.amp_dtype) != str(binding["amp_dtype"]):
        raise RuntimeError("worker AMP dtype differs from the frozen Native SS report")
    checkpoint = torch.load(binding["checkpoint"], map_location="cpu")
    validate_native_ss_no_vggt_checkpoint(
        checkpoint, pretrained=args.pretrained, allow_v2_parent=False
    )
    if int(checkpoint["step"]) != int(binding["checkpoint_step"]):
        raise RuntimeError("Native SS checkpoint step differs from deployment report")
    if sha256_file(binding["checkpoint"]) != str(binding["checkpoint_sha256"]):
        raise RuntimeError("Native SS checkpoint hash differs from deployment report")
    training_uids = checkpoint.get("data_identity", {}).get("object_uids")
    if not isinstance(training_uids, list):
        raise RuntimeError("Native SS checkpoint lacks training object identities")
    require_disjoint_object_uids(
        (str(dataset.rows[index]["object_uid"]) for index in selected_indices),
        training_uids,
    )
    saved = checkpoint["args"]
    sampler, model, decoder, summary, defaults = build_native_ss_no_vggt_components(
        pretrained=args.pretrained,
        lora_rank=int(saved["lora_rank"]),
        lora_alpha=int(saved["lora_alpha"]),
        condition_channels=int(saved["condition_channels"]),
        gradient_checkpointing=False,
        need_decoder=True,
        device=device,
    )
    if decoder is None:
        raise RuntimeError("Native SS evaluation requires the frozen SS decoder")
    state_key = "ema_trainable_state" if args.weights == "ema" else "model_trainable_state"
    load_trainable_state_dict(model, checkpoint[state_key])
    model.eval()
    decoder.eval()
    params = sampling_params(
        defaults,
        make_sampling_namespace(binding),
        float(binding["cfg_strength"]),
    )
    return binding, checkpoint, sampler, model, decoder, summary, params


def _build_stock_slat_pipeline(
    *, pretrained: str, stock_freeze: dict[str, Any], device: torch.device
):
    from trellis import pipelines

    install_unused_model_stubs()
    pipeline = pipelines.TrellisImageTo3DPipeline.from_pretrained(pretrained)
    pipeline._device = device
    pipeline.low_vram = False
    flow = pipeline.models["slat_flow_model"].to(device).eval()
    decoder = pipeline.models["slat_decoder_mesh"].to(device).eval()
    for module in (flow, decoder):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    validate_runtime_stock_slat(
        stock_freeze,
        pretrained=pretrained,
        flow=flow,
        decoder=decoder,
        sampler_params=dict(pipeline.slat_sampler_params),
        normalization=dict(pipeline.slat_normalization),
    )
    return pipeline, flow, decoder


def _sample_stock_slat(
    *,
    pipeline,
    flow,
    initial: sp.SparseTensor,
    condition: dict[str, Any],
    params: dict[str, Any],
    mean: torch.Tensor,
    std: torch.Tensor,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
) -> sp.SparseTensor:
    noise = sp.SparseTensor(
        feats=initial.feats.clone(), coords=initial.coords.clone()
    )
    with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_enabled):
        latent = pipeline.slat_sampler.sample(
            flow, noise, **condition, **params, verbose=False
        ).samples
    return latent * std + mean


def _build_trained_slat_pipeline(
    *,
    checkpoint_path: str | Path,
    weights: str,
    pretrained: str,
    stock_freeze: dict[str, Any],
    dataset: NativeConditionSLatDataset,
    expected_step: int,
    device: torch.device,
):
    path = Path(checkpoint_path).expanduser().resolve(strict=True)
    checkpoint = torch.load(path, map_location="cpu")
    if int(checkpoint.get("step", -1)) != int(expected_step):
        raise RuntimeError(
            f"trained SLat checkpoint step differs: {checkpoint.get('step')} != "
            f"{expected_step}"
        )
    saved_upstream = checkpoint.get("data_identity", {}).get("native_ss")
    summary_upstream = checkpoint.get("model_summary", {}).get("upstream_native_ss")
    if not isinstance(saved_upstream, dict) or saved_upstream != summary_upstream:
        raise RuntimeError("trained SLat checkpoint Native-SS identity is inconsistent")
    saved_protocol = (
        checkpoint.get("data_identity", {})
        .get("target_decoder_audit", {})
        .get("protocol_sha256")
    )
    current_protocol = dataset.config.get("target_source", {}).get("protocol_sha256")
    if not saved_protocol or saved_protocol != current_protocol:
        raise RuntimeError("trained SLat checkpoint official target protocol differs")
    validate_native_slat_no_vggt_checkpoint(
        checkpoint,
        pretrained=pretrained,
        stock_slat_freeze=stock_freeze,
        upstream_native_ss=saved_upstream,
        allow_v2_parent=False,
    )
    saved = checkpoint["args"]
    sampler, model, decoder, summary, defaults, normalization = (
        build_native_slat_no_vggt_components(
            pretrained=pretrained,
            stock_slat_freeze=stock_freeze,
            upstream_native_ss=saved_upstream,
            lora_rank=int(saved["lora_rank"]),
            lora_alpha=int(saved["lora_alpha"]),
            condition_channels=int(saved["condition_channels"]),
            gradient_checkpointing=False,
            need_decoder=True,
            device=device,
        )
    )
    if decoder is None:
        raise RuntimeError("trained SLat evaluation requires the Stock Mesh decoder")
    state_key = "ema_trainable_state" if weights == "ema" else "model_trainable_state"
    load_slat_trainable_state_dict(model, checkpoint[state_key])
    model.eval()
    decoder.eval()
    runtime_normalization = {
        key: [float(value) for value in values]
        for key, values in normalization.items()
    }
    if canonical_sha256(runtime_normalization) != dataset.slat_normalization_hash:
        raise RuntimeError("trained SLat runtime/cache normalization differs")
    params = dict(defaults)
    params.setdefault("guidance_rescale", 0.0)
    result = {
        "checkpoint_path": str(path),
        "checkpoint_sha256": sha256_file(path),
        "checkpoint_step": int(checkpoint["step"]),
        "weights": str(weights),
        "sampler": sampler,
        "model": model,
        "decoder": decoder,
        "summary": summary,
        "params": params,
        "mean": torch.tensor(runtime_normalization["mean"], device=device)[None],
        "std": torch.tensor(runtime_normalization["std"], device=device)[None],
    }
    del checkpoint
    return result


def _sample_trained_slat(
    *,
    runtime: dict[str, Any],
    initial: sp.SparseTensor,
    condition: dict[str, Any],
    lifting_sample: dict[str, Any],
    adapted: bool,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
) -> tuple[sp.SparseTensor, dict[str, Any]]:
    noise = sp.SparseTensor(
        feats=initial.feats.clone(), coords=initial.coords.clone()
    )
    if adapted:
        flow = NativeSLatCalibratedCFGFlow(
            runtime["model"],
            condition["cond"],
            lifting_sample,
            enabled=True,
        )
    else:
        flow = NativeSLatStockFlow(runtime["model"])
    with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_enabled):
        latent = runtime["sampler"].sample(
            flow,
            noise,
            **condition,
            **runtime["params"],
            verbose=False,
        ).samples
    summary = flow.summary() if adapted else {"adapted": False}
    return latent * runtime["std"] + runtime["mean"], summary


def _ss_record(
    *,
    object_uid: str,
    seed: int,
    stock_coords: np.ndarray,
    native_coords: np.ndarray,
    target_coords: np.ndarray,
    wrapper: dict[str, Any],
) -> dict[str, Any]:
    stock = overlap_metrics(stock_coords, target_coords)
    native = overlap_metrics(native_coords, target_coords)
    return {
        "object_uid": str(object_uid),
        "seed": int(seed),
        "same_initial_noise": True,
        "stock": stock,
        "native": native,
        "stock_count": int(len(stock_coords)),
        "native_count": int(len(native_coords)),
        "native_stock_count_ratio": (
            float(len(native_coords) / len(stock_coords)) if len(stock_coords) else None
        ),
        "iou_gain": float(native["iou"] - stock["iou"]),
        "precision_gain": float(native["precision"] - stock["precision"]),
        "recall_gain": float(native["recall"] - stock["recall"]),
        "native_wrapper": wrapper,
        "passed": bool(len(stock_coords) and len(native_coords)),
    }


def _recordable_mesh_decode_error(error: Exception) -> bool:
    """Return whether a decoder failure is a model output, not a runtime fault.

    CUDA OOM and unrelated implementation errors must still abort the worker so
    that resource failures are never counted as scientific model failures.
    """

    message = str(error)
    return isinstance(error, RuntimeError) and (
        message.startswith("FlexiCubes topology index is inconsistent:")
        or message.startswith("SLat decoder input exceeds safe active-point limit:")
        or (message.startswith("decoded ") and " Mesh is invalid:" in message)
    )


def _aggregate_occupancy(
    records: list[dict[str, Any]], *, bootstrap_samples: int
) -> dict[str, Any]:
    by_object: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        by_object[str(row["object_uid"])].append(row)
    object_rows = []
    for object_uid, rows in sorted(by_object.items()):
        ratios = [
            float(row["native_stock_count_ratio"])
            for row in rows
            if row["native_stock_count_ratio"] is not None
            and np.isfinite(float(row["native_stock_count_ratio"]))
        ]
        object_rows.append(
            {
                "object_uid": object_uid,
                **{
                    name: float(np.mean([float(row[name]) for row in rows]))
                    for name in OCCUPANCY_METRICS
                },
                "native_stock_count_ratio": (
                    float(np.mean(ratios)) if ratios else None
                ),
            }
        )
    summary = {
        name: summary_with_ci(
            [float(row[name]) for row in object_rows],
            samples=int(bootstrap_samples),
            seed=20260814 + position,
        )
        for position, name in enumerate(OCCUPANCY_METRICS)
    }
    ratios = [
        float(row["native_stock_count_ratio"])
        for row in object_rows
        if row["native_stock_count_ratio"] is not None
    ]
    count_ratio = summary_with_ci(
        ratios, samples=int(bootstrap_samples), seed=20260901
    )
    iou = summary["iou_gain"]
    recall = summary["recall_gain"]
    checks = {
        "all_supports_nonempty": all(row.get("passed") is True for row in records),
        "iou_mean_positive": iou["mean"] is not None and float(iou["mean"]) > 0.0,
        "iou_median_positive": iou["median"] is not None
        and float(iou["median"]) > 0.0,
        "iou_object_win_rate_ge_0p55": iou["positive_rate"] is not None
        and float(iou["positive_rate"]) >= 0.55,
        "iou_ci_lower_positive": iou["bootstrap_mean_95_ci"] is not None
        and float(iou["bootstrap_mean_95_ci"][0]) > 0.0,
        "recall_mean_nonnegative": recall["mean"] is not None
        and float(recall["mean"]) >= 0.0,
        "count_ratio_in_0p85_1p20": count_ratio["mean"] is not None
        and 0.85 <= float(count_ratio["mean"]) <= 1.20,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "summary": summary,
        "count_ratio": count_ratio,
        "object_rows": object_rows,
    }


def _select_transfer_branches(
    records: list[dict[str, Any]], *, baseline: str, candidate: str
) -> list[dict[str, Any]]:
    """Project named end-to-end branches into the strict stock/native aggregator."""

    if baseline == candidate:
        raise ValueError("transfer baseline and candidate must differ")
    selected: list[dict[str, Any]] = []
    for row in records:
        branch = str(row.get("branch", ""))
        if branch not in {baseline, candidate}:
            continue
        projected = dict(row)
        projected["source_branch"] = branch
        projected["branch"] = "stock" if branch == baseline else "native"
        selected.append(projected)
    return selected


def _worker_parser(subparsers) -> None:
    parser = subparsers.add_parser("worker", help="run one GPU object shard")
    parser.add_argument("--cache_manifest", required=True)
    parser.add_argument("--lifting_cache_manifest", required=True)
    parser.add_argument("--native_ss_report", required=True)
    parser.add_argument("--stock_slat_freeze", required=True)
    parser.add_argument("--trained_slat_checkpoint", default="")
    parser.add_argument(
        "--trained_slat_weights", choices=("ema", "raw"), default="ema"
    )
    parser.add_argument("--expected_trained_slat_step", type=int, default=8000)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--pretrained", default="Stable-X/trellis-vggt-v0-2")
    parser.add_argument("--weights", choices=("ema", "raw"), default="ema")
    parser.add_argument("--joint_seeds", default="42,43,44")
    parser.add_argument("--object_start", type=int, default=0)
    parser.add_argument("--object_end", type=int, default=0)
    parser.add_argument("--surface_samples", type=int, default=20000)
    parser.add_argument("--amp_dtype", choices=("bf16", "fp16", "none"), default="bf16")
    parser.add_argument("--save_meshes", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--restart_after_recorded_failure", action="store_true")


def _aggregate_parser(subparsers) -> None:
    parser = subparsers.add_parser("aggregate", help="aggregate completed shards")
    parser.add_argument("--cache_manifest", required=True)
    parser.add_argument("--lifting_cache_manifest", required=True)
    parser.add_argument("--shard_reports", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--joint_seeds", default="42,43,44")
    parser.add_argument("--expected_objects", type=int, default=64)
    parser.add_argument("--object_start", type=int, default=0)
    parser.add_argument("--object_end", type=int, default=0)
    parser.add_argument("--bootstrap_samples", type=int, default=5000)
    parser.add_argument("--chamfer_win_rate_min", type=float, default=0.55)
    parser.add_argument("--largest_component_delta_min", type=float, default=-0.02)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    _worker_parser(subparsers)
    _aggregate_parser(subparsers)
    return parser


@torch.no_grad()
def run_worker(args: argparse.Namespace) -> None:
    seeds = parse_csv(args.joint_seeds, int)
    if int(args.surface_samples) <= 0:
        raise ValueError("surface_samples must be positive")
    dataset = NativeConditionSLatDataset(
        args.cache_manifest, args.lifting_cache_manifest, indices="all"
    )
    target_contract = _official_target_contract(dataset)
    total = len(dataset)
    start = int(args.object_start)
    end = total if int(args.object_end) <= 0 else int(args.object_end)
    if start < 0 or end <= start or end > total:
        raise ValueError(f"invalid object slice [{start}:{end}] for {total} objects")
    selected = list(range(start, end))
    output = Path(args.output_dir).expanduser().resolve()
    identity = {
        "format": (
            END_TO_END_WORKER_FORMAT if args.trained_slat_checkpoint else WORKER_FORMAT
        ),
        "cache_manifest": str(Path(args.cache_manifest).resolve()),
        "cache_manifest_sha256": sha256_file(args.cache_manifest),
        "lifting_cache_manifest": str(Path(args.lifting_cache_manifest).resolve()),
        "lifting_cache_manifest_sha256": sha256_file(args.lifting_cache_manifest),
        "native_ss_report": str(Path(args.native_ss_report).resolve()),
        "native_ss_report_sha256": sha256_file(args.native_ss_report),
        "stock_slat_freeze": str(Path(args.stock_slat_freeze).resolve()),
        "stock_slat_freeze_sha256": sha256_file(args.stock_slat_freeze),
        "trained_slat_checkpoint": (
            str(Path(args.trained_slat_checkpoint).expanduser().resolve(strict=True))
            if args.trained_slat_checkpoint
            else ""
        ),
        "trained_slat_checkpoint_sha256": (
            sha256_file(args.trained_slat_checkpoint)
            if args.trained_slat_checkpoint
            else ""
        ),
        "trained_slat_weights": str(args.trained_slat_weights),
        "expected_trained_slat_step": int(args.expected_trained_slat_step),
        "official_protocol_sha256": str(target_contract["protocol_sha256"]),
        "object_start": start,
        "object_end": end,
        "object_uids": [str(dataset.rows[index]["object_uid"]) for index in selected],
        "joint_seeds": seeds,
        "weights": str(args.weights),
        "amp_dtype": str(args.amp_dtype),
        "surface_samples": int(args.surface_samples),
        "save_meshes": bool(args.save_meshes),
        "same_ss_initial_noise": True,
        "same_slat_condition": True,
        "same_coordinate_keyed_slat_noise": True,
        "stock_slat_pair_native_ss_only_difference": True,
        "end_to_end_pair_changes_native_ss_and_slat": bool(
            args.trained_slat_checkpoint
        ),
        "paired_branches": (
            ["stock", "native", "native_trained"]
            if args.trained_slat_checkpoint
            else ["stock", "native"]
        ),
    }
    identity_path = output / "run_identity.json"
    if output.exists() and not args.resume:
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    if identity_path.is_file():
        existing = json.loads(identity_path.read_text(encoding="utf-8"))
        if existing != identity:
            raise RuntimeError("resume arguments differ from existing worker identity")
    else:
        write_json(identity_path, identity)
    report_path = output / "report.json"
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        print(json.dumps({"reused": True, "passed": report["passed"]}, indent=2))
        return

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    amp_enabled = args.amp_dtype != "none"
    amp_dtype = torch.float16 if args.amp_dtype == "fp16" else torch.bfloat16
    (
        ss_binding,
        checkpoint,
        ss_sampler,
        ss_model,
        ss_decoder,
        ss_summary,
        ss_params,
    ) = _load_ss_runtime(args, dataset, selected, target_contract, device)
    coord_root = output / "ss_coords"
    ss_records: list[dict[str, Any]] = []
    for local_position, index in enumerate(selected):
        sample = dataset[index]
        lifting = sample["lifting_sample"]
        object_uid = str(sample["object_uid"])
        target_coords = sample["target_coords"].cpu().numpy()
        positive = lifting["stock_condition"].to(device=device)
        negative = torch.zeros_like(positive)
        for seed in seeds:
            current = pair_id(object_uid, seed)
            npz_path = coord_root / f"{current}.npz"
            audit_path = coord_root / f"{current}.json"
            if npz_path.is_file() and audit_path.is_file():
                ss_records.append(json.loads(audit_path.read_text(encoding="utf-8")))
                continue
            generator = torch.Generator(device=device).manual_seed(
                int(seed) * 1000003 + int(index) * 1009
            )
            initial = torch.randn(
                (1, 8, 16, 16, 16), generator=generator, device=device
            )
            with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_enabled):
                stock_latent = ss_sampler.sample(
                    NativeSSStockFlow(ss_model),
                    initial.clone(),
                    cond=positive,
                    neg_cond=negative,
                    **ss_params,
                    verbose=False,
                ).samples
                native_flow = NativeSSCalibratedCFGFlow(
                    ss_model,
                    positive,
                    lifting,
                    enabled=True,
                    projection_mode="correct",
                )
                native_latent = ss_sampler.sample(
                    native_flow,
                    initial.clone(),
                    cond=positive,
                    neg_cond=negative,
                    **ss_params,
                    verbose=False,
                ).samples
            if float(ss_binding["cfg_strength"]) != 1.0 and (
                native_flow.positive_calls == 0 or native_flow.negative_calls == 0
            ):
                raise RuntimeError("Native SS CFG did not execute both branches")
            stock_coords = decode_coords(ss_decoder, stock_latent)
            native_coords = decode_coords(ss_decoder, native_latent)
            row = _ss_record(
                object_uid=object_uid,
                seed=seed,
                stock_coords=stock_coords,
                native_coords=native_coords,
                target_coords=target_coords,
                wrapper=native_flow.summary(),
            )
            coord_root.mkdir(parents=True, exist_ok=True)
            with npz_path.open("wb") as handle:
                np.savez_compressed(
                    handle,
                    stock=stock_coords.astype(np.int32),
                    native=native_coords.astype(np.int32),
                )
            row["coords_npz_sha256"] = sha256_file(npz_path)
            write_json(audit_path, row)
            ss_records.append(row)
            print(
                f"[official_ss:ss] {local_position + 1}/{len(selected)} "
                f"seed={seed} uid={object_uid}",
                flush=True,
            )
            del initial, stock_latent, native_latent, native_flow
            torch.cuda.empty_cache()
        del positive, negative
    ss_model.cpu()
    ss_decoder.cpu()
    del ss_model, ss_decoder, ss_sampler, checkpoint
    gc.collect()
    torch.cuda.empty_cache()

    stock_freeze = load_stock_slat_freeze(args.stock_slat_freeze)
    trained_runtime = None
    trained_slat_binding = None
    if args.trained_slat_checkpoint:
        trained_runtime = _build_trained_slat_pipeline(
            checkpoint_path=args.trained_slat_checkpoint,
            weights=str(args.trained_slat_weights),
            pretrained=args.pretrained,
            stock_freeze=stock_freeze,
            dataset=dataset,
            expected_step=int(args.expected_trained_slat_step),
            device=device,
        )
        pipeline = slat_flow = None
        mesh_decoder = trained_runtime["decoder"]
        slat_params = dict(trained_runtime["params"])
        mean = trained_runtime["mean"]
        std = trained_runtime["std"]
        slat_channels = int(trained_runtime["model"].flow_core.in_channels)
        trained_slat_binding = {
            "checkpoint": trained_runtime["checkpoint_path"],
            "checkpoint_sha256": trained_runtime["checkpoint_sha256"],
            "checkpoint_step": trained_runtime["checkpoint_step"],
            "weights": trained_runtime["weights"],
            "model_summary": trained_runtime["summary"],
        }
    else:
        pipeline, slat_flow, mesh_decoder = _build_stock_slat_pipeline(
            pretrained=args.pretrained, stock_freeze=stock_freeze, device=device
        )
        slat_params = dict(stock_freeze["slat_sampler_params"])
        mean = torch.tensor(
            stock_freeze["slat_normalization"]["mean"], device=device
        )[None]
        std = torch.tensor(
            stock_freeze["slat_normalization"]["std"], device=device
        )[None]
        slat_channels = int(slat_flow.in_channels)
    records: list[dict[str, Any]] = []
    expected_branches = (
        ("stock", "native", "native_trained")
        if trained_runtime is not None
        else ("stock", "native")
    )
    pair_root = output / "mesh_pairs"
    target_root = output / "target_mesh_cache"
    for local_position, index in enumerate(selected):
        object_uid = str(dataset.rows[index]["object_uid"])
        existing_records = []
        object_complete = True
        for seed in seeds:
            existing_path = pair_root / pair_id(object_uid, seed) / "pair_record.json"
            if not existing_path.is_file():
                object_complete = False
                break
            existing = json.loads(existing_path.read_text(encoding="utf-8"))
            if (
                str(existing.get("object_uid")) != object_uid
                or int(existing.get("seed", -1)) != int(seed)
                or {str(row.get("branch")) for row in existing.get("branches", [])}
                != set(expected_branches)
            ):
                raise RuntimeError(f"resumed Mesh pair identity differs: {existing_path}")
            existing_records.extend(existing["branches"])
        if object_complete:
            records.extend(existing_records)
            print(
                f"[official_ss:mesh_reuse_object] {local_position + 1}/"
                f"{len(selected)} uid={object_uid}",
                flush=True,
            )
            continue

        sample = dataset[index]
        target_coords = sample["target_coords"].to(device=device, dtype=torch.int32)
        target_feats = sample["target_feats"].to(
            device=device, dtype=next(mesh_decoder.parameters()).dtype
        )
        target_cache = target_root / f"{object_uid}.npz"
        if target_cache.is_file():
            with np.load(target_cache) as payload:
                target_mesh = trimesh.Trimesh(
                    vertices=np.asarray(payload["vertices"]),
                    faces=np.asarray(payload["faces"]),
                    process=False,
                )
            target_latent = None
        else:
            target_latent = sp.SparseTensor(feats=target_feats, coords=target_coords)
            target_mesh = mesh_decoder(target_latent)[0].to_trimesh(
                transform_pose=False
            )
            target_root.mkdir(parents=True, exist_ok=True)
            temporary = target_cache.with_name(f".{target_cache.name}.tmp")
            with temporary.open("wb") as handle:
                np.savez_compressed(
                    handle,
                    vertices=np.asarray(target_mesh.vertices),
                    faces=np.asarray(target_mesh.faces),
                )
            os.replace(temporary, target_cache)
        target_structure = mesh_structure_metrics(target_mesh)
        if not target_structure["mesh_success"]:
            raise RuntimeError(f"official decoded target Mesh is invalid: {object_uid}")
        condition = to_device_tree(sample["condition"], device)
        for seed in seeds:
            current = pair_id(object_uid, seed)
            record_path = pair_root / current / "pair_record.json"
            if record_path.is_file():
                records.extend(
                    json.loads(record_path.read_text(encoding="utf-8"))["branches"]
                )
                continue
            npz_path = coord_root / f"{current}.npz"
            with np.load(npz_path) as payload:
                coords = {
                    branch: canonical_coords(payload[branch], resolution=64)
                    for branch in ("stock", "native")
                }
            generator = torch.Generator(device=device).manual_seed(
                int(seed) * 2000003 + int(index) * 2017 + 7919
            )
            master = torch.randn(
                (64, 64, 64, slat_channels),
                generator=generator,
                device=device,
                dtype=torch.float32,
            )
            initial = {
                branch: sparse_noise_from_master(coords[branch], master, device=device)
                for branch in ("stock", "native")
            }
            noise_audit = shared_noise_audit(
                initial["stock"].coords,
                initial["stock"].feats,
                initial["native"].coords,
                initial["native"].feats,
            )
            if not noise_audit["common_coord_noise_bit_exact"]:
                raise RuntimeError("Stock/Native common-coordinate SLat noise differs")
            offset = int(current[-1], 16) % len(expected_branches)
            order = expected_branches[offset:] + expected_branches[:offset]
            branch_rows: dict[str, dict[str, Any]] = {}
            restart_after_pair = False
            for branch in order:
                row = {
                    "pair_id": current,
                    "branch": branch,
                    "object_uid": object_uid,
                    "seed": int(seed),
                    "target_structure": target_structure,
                    "passed": False,
                }
                latent = decoded = mesh = None
                try:
                    support = "stock" if branch == "stock" else "native"
                    if trained_runtime is None:
                        latent = _sample_stock_slat(
                            pipeline=pipeline,
                            flow=slat_flow,
                            initial=initial[support],
                            condition=condition,
                            params=slat_params,
                            mean=mean,
                            std=std,
                            amp_enabled=amp_enabled,
                            amp_dtype=amp_dtype,
                        )
                        flow_summary = {"adapted": False}
                    else:
                        latent, flow_summary = _sample_trained_slat(
                            runtime=trained_runtime,
                            initial=initial[support],
                            condition=condition,
                            lifting_sample=sample["lifting_sample"],
                            adapted=branch == "native_trained",
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
                    decoded = mesh_decoder(latent)[0]
                    mesh = decoded.to_trimesh(transform_pose=False)
                    structure = mesh_structure_metrics(mesh)
                    if not structure["mesh_success"]:
                        raise RuntimeError(
                            f"decoded {branch} Mesh is invalid: {object_uid}"
                        )
                    surface = surface_metrics(
                        mesh,
                        target_mesh,
                        count=int(args.surface_samples),
                        seed=int(seed) * 1009 + int(index) * 9173,
                        thresholds=(0.01, 0.02, 0.05),
                    )
                    row.update(
                        {
                            "passed": True,
                            "structure": structure,
                            "surface": surface,
                            "flow_summary": flow_summary,
                        }
                    )
                    if args.save_meshes:
                        mesh_path = pair_root / current / branch / "mesh.obj"
                        mesh_path.parent.mkdir(parents=True, exist_ok=True)
                        mesh.export(mesh_path)
                        row["mesh"] = str(mesh_path)
                        row["mesh_sha256"] = sha256_file(mesh_path)
                except Exception as error:
                    if not _recordable_mesh_decode_error(error):
                        raise
                    row["error"] = {
                        "type": type(error).__name__,
                        "message": str(error),
                        "stage": f"{branch}_slat_mesh_decode",
                    }
                    restart_after_pair = True
                    print(
                        f"[official_ss:mesh_failed] {local_position + 1}/"
                        f"{len(selected)} seed={seed} branch={branch} "
                        f"uid={object_uid}: {error}",
                        flush=True,
                    )
                branch_rows[branch] = row
                del latent, decoded, mesh
                torch.cuda.empty_cache()
            pair_record = {
                "pair_id": current,
                "object_uid": object_uid,
                "seed": int(seed),
                "execution_order": list(order),
                "noise_audit": noise_audit,
                "native_support_same_initial_tensor_by_construction": bool(
                    trained_runtime is not None
                ),
                "branches": [branch_rows[branch] for branch in expected_branches],
                "passed": all(row["passed"] for row in branch_rows.values()),
            }
            write_json(record_path, pair_record)
            records.extend(pair_record["branches"])
            print(
                f"[official_ss:mesh] {local_position + 1}/{len(selected)} "
                f"seed={seed} uid={object_uid}",
                flush=True,
            )
            del master, initial, branch_rows
            torch.cuda.empty_cache()
            if restart_after_pair and args.restart_after_recorded_failure:
                print(
                    "[official_ss:restart_required] recorded a topology failure; "
                    "restart the CUDA process before the next pair",
                    flush=True,
                )
                raise SystemExit(75)
        del target_latent, target_mesh, condition
    if trained_runtime is None:
        slat_flow.cpu()
        mesh_decoder.cpu()
        del slat_flow, mesh_decoder, pipeline
    else:
        trained_runtime["model"].cpu()
        trained_runtime["decoder"].cpu()
        del trained_runtime
    gc.collect()
    torch.cuda.empty_cache()

    expected_pairs = len(selected) * len(seeds)
    runtime_passed = bool(
        len(ss_records) == expected_pairs
        and all(row.get("passed") is True for row in ss_records)
        and len(records) == len(expected_branches) * expected_pairs
        and all(row.get("passed") is True for row in records)
    )
    report = {
        "format": identity["format"],
        "complete": True,
        "passed": runtime_passed,
        "formal": False,
        "run_identity": identity,
        "native_ss_binding": ss_binding,
        "native_ss_model_summary": ss_summary,
        "trained_slat": trained_slat_binding,
        "object_count": len(selected),
        "record_count": expected_pairs,
        "ss_records": ss_records,
        "mesh_branch_records": records,
        "scope_guard": (
            "official held-out Dev shard runtime only; aggregate the exact frozen "
            "object slice before making a development bridge decision"
        ),
    }
    report["report_sha256"] = canonical_sha256(report)
    write_json(report_path, report)
    print(
        json.dumps(
            {
                "passed": runtime_passed,
                "objects": len(selected),
                "pairs": expected_pairs,
                "report": str(report_path),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    if not runtime_passed:
        raise SystemExit(2)


def run_aggregate(args: argparse.Namespace) -> None:
    seeds = parse_csv(args.joint_seeds, int)
    reports = [Path(path).expanduser().resolve() for path in parse_csv(args.shard_reports, str)]
    dataset = NativeConditionSLatDataset(
        args.cache_manifest, args.lifting_cache_manifest, indices="all"
    )
    target_contract = _official_target_contract(dataset)
    expected_objects = int(args.expected_objects)
    object_start = int(args.object_start)
    object_end = len(dataset) if int(args.object_end) <= 0 else int(args.object_end)
    if (
        object_start < 0
        or object_end <= object_start
        or object_end > len(dataset)
        or object_end - object_start != expected_objects
    ):
        raise RuntimeError(
            f"invalid aggregate object slice [{object_start}:{object_end}] for "
            f"cache={len(dataset)} expected={expected_objects}"
        )
    expected_rows = dataset.rows[object_start:object_end]
    expected_uids = {str(row["object_uid"]) for row in expected_rows}
    if len(expected_uids) != expected_objects:
        raise RuntimeError("official Dev cache does not contain one row per object")
    ss_records: list[dict[str, Any]] = []
    mesh_records: list[dict[str, Any]] = []
    observed_uids: list[str] = []
    shard_bindings = []
    shared_identity: dict[str, Any] | None = None
    worker_format: str | None = None
    workers_passed = True
    for path in reports:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("format") not in {WORKER_FORMAT, END_TO_END_WORKER_FORMAT} or payload.get("complete") is not True:
            raise RuntimeError(f"invalid/incomplete worker report: {path}")
        if worker_format is None:
            worker_format = str(payload["format"])
        elif str(payload["format"]) != worker_format:
            raise RuntimeError("worker report formats differ")
        workers_passed = workers_passed and payload.get("passed") is True
        body = dict(payload)
        saved_hash = str(body.pop("report_sha256", ""))
        if not saved_hash or canonical_sha256(body) != saved_hash:
            raise RuntimeError(f"worker report hash mismatch: {path}")
        identity = dict(payload["run_identity"])
        comparable = {
            key: value
            for key, value in identity.items()
            if key not in {"object_start", "object_end", "object_uids"}
        }
        if shared_identity is None:
            shared_identity = comparable
        elif comparable != shared_identity:
            raise RuntimeError("worker run identities differ")
        if identity["joint_seeds"] != seeds:
            raise RuntimeError("worker seeds differ from aggregate seeds")
        observed_uids.extend(str(value) for value in identity["object_uids"])
        ss_records.extend(payload["ss_records"])
        mesh_records.extend(payload["mesh_branch_records"])
        shard_bindings.append(
            {"path": str(path), "sha256": sha256_file(path), "report_sha256": saved_hash}
        )
    duplicate_uids = len(observed_uids) != len(set(observed_uids))
    coverage_ok = (
        not duplicate_uids
        and len(observed_uids) == expected_objects
        and set(observed_uids) == expected_uids
    )
    expected_pairs = expected_objects * len(seeds)
    pair_keys = {(str(row["object_uid"]), int(row["seed"])) for row in ss_records}
    ss_integrity = bool(
        coverage_ok
        and len(ss_records) == expected_pairs
        and len(pair_keys) == expected_pairs
        and all(row.get("passed") is True for row in ss_records)
    )
    occupancy = _aggregate_occupancy(
        ss_records, bootstrap_samples=int(args.bootstrap_samples)
    )
    metadata = {uid: {"source": "official_proobjaverse", "view_count": 8} for uid in expected_uids}
    stock_slat_transfer = aggregate_transfer_records(
        _select_transfer_branches(
            mesh_records, baseline="stock", candidate="native"
        ),
        expected_pairs=expected_pairs,
        seeds=seeds,
        bootstrap_samples=int(args.bootstrap_samples),
        metadata_by_object=metadata,
        chamfer_win_rate_min=float(args.chamfer_win_rate_min),
        lcr_delta_min=float(args.largest_component_delta_min),
    )
    trained_enabled = bool(
        shared_identity and shared_identity.get("trained_slat_checkpoint")
    )
    trained_end_to_end_transfer = None
    trained_increment = None
    if trained_enabled:
        trained_end_to_end_transfer = aggregate_transfer_records(
            _select_transfer_branches(
                mesh_records, baseline="stock", candidate="native_trained"
            ),
            expected_pairs=expected_pairs,
            seeds=seeds,
            bootstrap_samples=int(args.bootstrap_samples),
            metadata_by_object=metadata,
            chamfer_win_rate_min=float(args.chamfer_win_rate_min),
            lcr_delta_min=float(args.largest_component_delta_min),
        )
        trained_increment = aggregate_transfer_records(
            _select_transfer_branches(
                mesh_records, baseline="native", candidate="native_trained"
            ),
            expected_pairs=expected_pairs,
            seeds=seeds,
            bootstrap_samples=int(args.bootstrap_samples),
            metadata_by_object=metadata,
            chamfer_win_rate_min=float(args.chamfer_win_rate_min),
            lcr_delta_min=float(args.largest_component_delta_min),
        )
    primary_transfers = [stock_slat_transfer]
    if trained_end_to_end_transfer is not None:
        primary_transfers.append(trained_end_to_end_transfer)
    runtime_passed = bool(
        workers_passed
        and ss_integrity
        and all(
            transfer["checks"]["expected_record_count"]
            and transfer["checks"]["expected_pair_count"]
            and transfer["checks"]["no_invalid_pairs"]
            for transfer in primary_transfers
        )
    )
    stock_bridge_passed = bool(
        runtime_passed and occupancy["passed"] and stock_slat_transfer["passed"]
    )
    trained_end_to_end_passed = bool(
        stock_bridge_passed
        and trained_end_to_end_transfer is not None
        and trained_end_to_end_transfer["passed"]
    )
    science_passed = (
        trained_end_to_end_passed if trained_enabled else stock_bridge_passed
    )
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "format": (
            END_TO_END_REPORT_FORMAT
            if worker_format == END_TO_END_WORKER_FORMAT
            else REPORT_FORMAT
        ),
        "passed": runtime_passed,
        "formal": False,
        "official_protocol_sha256": str(target_contract["protocol_sha256"]),
        "object_count": expected_objects,
        "record_count": expected_pairs,
        "object_start": object_start,
        "object_end": object_end,
        "joint_seeds": seeds,
        "worker_reports": shard_bindings,
        "integrity": {
            "object_coverage_exact": coverage_ok,
            "ss_records_exact": ss_integrity,
            "mesh_pairs_exact": bool(
                all(
                    transfer["checks"]["expected_record_count"]
                    and transfer["checks"]["expected_pair_count"]
                    and transfer["checks"]["no_invalid_pairs"]
                    for transfer in primary_transfers
                )
            ),
        },
        "occupancy": occupancy,
        "stock_slat_mesh_transfer": stock_slat_transfer,
        "trained_slat_end_to_end_transfer": trained_end_to_end_transfer,
        "trained_slat_increment_on_native_support": trained_increment,
        "decision": {
            "native_ss_stock_slat_bridge_passed": stock_bridge_passed,
            "native_ss_trained_slat_end_to_end_passed": (
                trained_end_to_end_passed if trained_enabled else None
            ),
            "required": {
                "runtime_integrity": runtime_passed,
                "official_occupancy_advantage": occupancy["passed"],
                "frozen_stock_slat_mesh_advantage": stock_slat_transfer["passed"],
                "trained_slat_end_to_end_advantage": (
                    trained_end_to_end_transfer["passed"]
                    if trained_end_to_end_transfer is not None
                    else None
                ),
            },
            "interpretation": (
                "The retrained Native SS passes both the frozen-Stock-SLat bridge and the trained-SLat end-to-end development gate."
                if science_passed and trained_enabled
                else (
                    "The retrained Native SS passes the frozen-Stock-SLat development bridge."
                    if science_passed
                    else "At least one required official development bridge has not passed."
                )
            ),
        },
        "scope_guard": (
            "official ProObjaverse held-out development test; it decides whether the "
            "two predicted-support bridges pass, but it is not a final untouched claim"
        ),
    }
    report["report_sha256"] = canonical_sha256(report)
    write_json(output / "report.json", report)
    iou = occupancy["summary"]["iou_gain"]
    stock_chamfer = stock_slat_transfer["summary"]["chamfer_l1_improvement"]
    trained_chamfer = (
        None
        if trained_end_to_end_transfer is None
        else trained_end_to_end_transfer["summary"]["chamfer_l1_improvement"]
    )
    lines = [
        "Official held-out Dev Native SS -> Stock/trained SLat test",
        "=" * 57,
        f"objects: {expected_objects} records: {expected_pairs} seeds: {seeds}",
        f"runtime_passed: {runtime_passed}",
        f"occupancy_iou_gain: {iou}",
        f"new_ss_stock_slat_chamfer_improvement: {stock_chamfer}",
        f"new_ss_trained_slat_chamfer_improvement: {trained_chamfer}",
        f"occupancy_checks: {occupancy['checks']}",
        f"new_ss_stock_slat_checks: {stock_slat_transfer['checks']}",
        f"new_ss_trained_slat_checks: {None if trained_end_to_end_transfer is None else trained_end_to_end_transfer['checks']}",
        f"science_passed: {science_passed}",
        report["scope_guard"],
    ]
    (output / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    raise SystemExit(0 if science_passed else 3)


def main() -> None:
    args = make_parser().parse_args()
    if args.command == "worker":
        run_worker(args)
    else:
        run_aggregate(args)


if __name__ == "__main__":
    main()
