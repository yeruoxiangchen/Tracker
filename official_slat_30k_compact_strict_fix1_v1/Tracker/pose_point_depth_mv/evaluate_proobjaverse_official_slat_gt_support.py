#!/usr/bin/env python3
"""Same-noise Mesh test against decoded official SLat targets on GT support."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from pose_point_depth_mv.eval_direct_slat_flow import summarize
from pose_point_depth_mv.export_direct_flow_mesh_pairs import (
    mesh_structure_metrics,
    sparse_noise_from_master,
    surface_metrics,
)
from pose_point_depth_mv.native_3d_condition import NativeConditionSLatDataset
from pose_point_depth_mv.native_slat_condition_only import (
    NativeSLatConditionOnlyCFGFlow,
    NativeSLatConditionOnlyStockFlow,
    load_trainable_state_dict as load_condition_only_state,
)
from pose_point_depth_mv.native_slat_condition_only_objective_v2 import (
    build_native_slat_condition_only_objective_v2_components,
    validate_native_slat_condition_only_objective_v2_checkpoint,
)
from pose_point_depth_mv.native_slat_genrecon import (
    NativeSLatCalibratedCFGFlow,
    NativeSLatStockFlow,
    load_stock_slat_freeze,
)
from pose_point_depth_mv.native_slat_genrecon_no_vggt import (
    build_native_slat_no_vggt_components,
    validate_native_slat_no_vggt_checkpoint,
)
from pose_point_depth_mv.native_slat_genrecon_v2 import (
    load_trainable_state_dict as load_condition_lora_state,
)
from pose_point_depth_mv.no_vggt_ss_evidence import load_no_vggt_ss_evidence
from pose_point_depth_mv.proobjaverse_official_slat_protocol import (
    canonical_sha256,
    sha256_file,
)
from pose_point_depth_mv.train_direct_slat_flow import to_device_tree
from pose_point_depth_mv.train_native_slat_genrecon import upstream_binding
from trellis.modules import sparse as sp


REPORT_FORMAT = "pose_point_depth_mv.proobjaverse_official_slat_gt_support_mesh.v1"


def parse_csv(value: str) -> list[int]:
    result = [int(item) for item in str(value).split(",") if item.strip()]
    if not result or len(result) != len(set(result)):
        raise ValueError("joint_seeds must be non-empty and unique")
    return result


def _upstream_binding(value: dict[str, Any]) -> dict[str, Any]:
    binding = upstream_binding(value)
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
                if key in value
            }
        )
    return binding


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("condition_only", "condition_lora"), required=True)
    parser.add_argument("--cache_manifest", required=True)
    parser.add_argument("--lifting_cache_manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--native_ss_report", required=True)
    parser.add_argument("--stock_slat_freeze", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--pretrained", default="Stable-X/trellis-vggt-v0-2")
    parser.add_argument("--weights", choices=("ema", "raw"), default="ema")
    parser.add_argument("--joint_seeds", default="42")
    parser.add_argument("--max_objects", type=int, default=16)
    parser.add_argument("--object_start", type=int, default=0)
    parser.add_argument("--object_end", type=int, default=0)
    parser.add_argument("--surface_samples", type=int, default=20000)
    parser.add_argument("--bootstrap_samples", type=int, default=5000)
    parser.add_argument("--amp_dtype", choices=("bf16", "fp16", "none"), default="bf16")
    return parser


@torch.no_grad()
def main() -> None:
    args = make_parser().parse_args()
    seeds = parse_csv(args.joint_seeds)
    if int(args.max_objects) <= 0 or int(args.surface_samples) <= 0:
        raise ValueError("max_objects/surface_samples must be positive")
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    dataset = NativeConditionSLatDataset(
        args.cache_manifest, args.lifting_cache_manifest, indices="all"
    )
    if dataset.config.get("target_source", {}).get("support_policy") != "official_gt_slat_coordinates":
        raise RuntimeError("evaluation cache is not the official GT-support protocol")
    target_source = dict(dataset.config.get("target_source", {}))
    official_split = str(target_source.get("split", ""))
    if official_split not in {
        "train",
        "dev",
        "decoder_audit",
        "predicted_support_bridge",
    }:
        raise RuntimeError(f"official evaluation split is invalid: {official_split!r}")
    all_object_indices: list[int] = []
    seen: set[str] = set()
    for index, row in enumerate(dataset.rows):
        uid = str(row["object_uid"])
        if uid not in seen:
            seen.add(uid)
            all_object_indices.append(index)
        if len(all_object_indices) == int(args.max_objects):
            break
    if not all_object_indices:
        raise RuntimeError("official evaluation selection is empty")
    object_start = int(args.object_start)
    object_end = (
        len(all_object_indices) if int(args.object_end) <= 0 else int(args.object_end)
    )
    if (
        object_start < 0
        or object_end <= object_start
        or object_end > len(all_object_indices)
    ):
        raise ValueError(
            f"invalid object slice [{object_start}:{object_end}] for "
            f"{len(all_object_indices)} selected objects"
        )
    selected_objects = list(
        enumerate(all_object_indices[object_start:object_end], start=object_start)
    )
    object_indices = [index for _, index in selected_objects]

    _, ss_binding_all = load_no_vggt_ss_evidence(args.native_ss_report)
    if dataset.config.get("native_ss_deployment") != ss_binding_all:
        raise RuntimeError("official cache/evaluation Native-SS bridge binding differs")
    ss_binding = _upstream_binding(ss_binding_all)
    stock_freeze = load_stock_slat_freeze(args.stock_slat_freeze)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    if args.arm == "condition_only":
        validate_native_slat_condition_only_objective_v2_checkpoint(
            checkpoint,
            pretrained=args.pretrained,
            stock_slat_freeze=stock_freeze,
            upstream_native_ss=ss_binding,
        )
        sampler, model, decoder, model_summary, defaults, normalization = (
            build_native_slat_condition_only_objective_v2_components(
                pretrained=args.pretrained,
                stock_slat_freeze=stock_freeze,
                upstream_native_ss=ss_binding,
                condition_channels=int(checkpoint["args"]["condition_channels"]),
                gradient_checkpointing=False,
                need_decoder=True,
                device=device,
            )
        )
        load_state = load_condition_only_state
        stock_wrapper = NativeSLatConditionOnlyStockFlow
        full_wrapper = NativeSLatConditionOnlyCFGFlow
    else:
        validate_native_slat_no_vggt_checkpoint(
            checkpoint,
            pretrained=args.pretrained,
            stock_slat_freeze=stock_freeze,
            upstream_native_ss=ss_binding,
            allow_v2_parent=False,
        )
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
        load_state = load_condition_lora_state
        stock_wrapper = NativeSLatStockFlow
        full_wrapper = NativeSLatCalibratedCFGFlow
    if decoder is None:
        raise RuntimeError("official GT-support Mesh evaluation requires the Stock decoder")
    state_key = "ema_trainable_state" if args.weights == "ema" else "model_trainable_state"
    load_state(model, checkpoint[state_key])
    model.eval()
    decoder.eval()
    runtime_normalization = {
        key: [float(value) for value in values] for key, values in normalization.items()
    }
    if canonical_sha256(runtime_normalization) != dataset.slat_normalization_hash:
        raise RuntimeError("official cache/runtime SLat normalization differs")
    mean = torch.tensor(runtime_normalization["mean"], device=device)[None]
    std = torch.tensor(runtime_normalization["std"], device=device)[None]
    params = dict(defaults)
    params.setdefault("guidance_rescale", 0.0)
    amp_enabled = args.amp_dtype != "none"
    amp_dtype = torch.float16 if args.amp_dtype == "fp16" else torch.bfloat16
    run_config = {
        "format": REPORT_FORMAT,
        "arm": args.arm,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "checkpoint_step": int(checkpoint["step"]),
        "weights": args.weights,
        "baseline": "frozen_stock_slat_on_same_official_gt_support",
        "target": "frozen Stock decoder applied to official SLat label",
        "official_protocol_sha256": str(target_source["protocol_sha256"]),
        "official_split": official_split,
        "training_overlap": official_split == "train",
        "same_coordinates": True,
        "same_initial_noise": True,
        "native_ss_executed": False,
        "native_ss_role": "future predicted-support bridge binding only",
        "sampling": params,
        "seeds": seeds,
        "object_start": object_start,
        "object_end": object_end,
        "expected_object_count_before_sharding": len(all_object_indices),
        "object_uids": [str(dataset.rows[index]["object_uid"]) for index in object_indices],
    }
    (output / "run_config.json").write_text(
        json.dumps(run_config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    records: list[dict[str, Any]] = []
    for object_position, dataset_index in selected_objects:
        sample = dataset[dataset_index]
        coords = sample["target_coords"].to(device=device, dtype=torch.int32)
        target_feats = sample["target_feats"].to(
            device=device, dtype=next(decoder.parameters()).dtype
        )
        target_latent = sp.SparseTensor(feats=target_feats, coords=coords)
        target_mesh = decoder(target_latent)[0].to_trimesh(transform_pose=False)
        target_structure = mesh_structure_metrics(target_mesh)
        if not target_structure["mesh_success"]:
            raise RuntimeError(f"official target Mesh is invalid: {sample['object_uid']}")
        target_dir = output / "targets" / str(sample["object_uid"])
        target_dir.mkdir(parents=True, exist_ok=True)
        target_mesh.export(target_dir / "decoded_official_target.obj")
        condition = to_device_tree(sample["condition"], device)
        coords_np = coords.detach().cpu().numpy()
        for seed in seeds:
            generator = torch.Generator(device=device).manual_seed(
                int(seed) * 2000003 + object_position * 2017 + 7919
            )
            master = torch.randn(
                (64, 64, 64, 8), generator=generator, device=device, dtype=torch.float32
            )
            initial = sparse_noise_from_master(coords_np, master, device=device)
            outputs = {}
            wrapper_summary = {}
            for branch in ("stock", "full"):
                noise = sp.SparseTensor(
                    feats=initial.feats.clone(), coords=initial.coords.clone()
                )
                flow = (
                    stock_wrapper(model)
                    if branch == "stock"
                    else full_wrapper(
                        model,
                        condition["cond"],
                        sample["lifting_sample"],
                        enabled=True,
                    )
                )
                with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_enabled):
                    outputs[branch] = sampler.sample(
                        flow, noise, **condition, **params, verbose=False
                    ).samples
                if branch == "full":
                    wrapper_summary = flow.summary()
            branches = {}
            pair_id = f"obj_{object_position:04d}_seed_{seed}"
            for branch, latent in outputs.items():
                mesh = decoder(latent * std + mean)[0].to_trimesh(transform_pose=False)
                structure = mesh_structure_metrics(mesh)
                if not structure["mesh_success"]:
                    raise RuntimeError(f"invalid sampled Mesh: {pair_id}/{branch}")
                branch_dir = output / "mesh_pairs" / pair_id / branch
                branch_dir.mkdir(parents=True, exist_ok=True)
                mesh.export(branch_dir / "mesh.obj")
                surface = surface_metrics(
                    mesh,
                    target_mesh,
                    count=int(args.surface_samples),
                    seed=int(seed) * 1009 + object_position * 9173,
                    thresholds=(0.01, 0.02, 0.05),
                )
                branches[branch] = {"structure": structure, "surface": surface}
            records.append(
                {
                    "pair_id": pair_id,
                    "object_uid": str(sample["object_uid"]),
                    "seed": seed,
                    "coord_count": int(coords.shape[0]),
                    "wrapper": wrapper_summary,
                    "target_structure": target_structure,
                    "branches": branches,
                    "chamfer_l1_improvement": float(
                        branches["stock"]["surface"]["chamfer_l1"]
                        - branches["full"]["surface"]["chamfer_l1"]
                    ),
                    "fscore_0p02_delta": float(
                        branches["full"]["surface"]["fscore_0p02"]
                        - branches["stock"]["surface"]["fscore_0p02"]
                    ),
                    "normal_consistency_delta": float(
                        branches["full"]["surface"]["normal_consistency"]
                        - branches["stock"]["surface"]["normal_consistency"]
                    ),
                    "largest_component_ratio_delta": float(
                        branches["full"]["structure"]["largest_component_ratio"]
                        - branches["stock"]["structure"]["largest_component_ratio"]
                    ),
                }
            )
            print(f"[official_gt_support_mesh] {args.arm} {pair_id}", flush=True)
            del master, initial, outputs, branches
            torch.cuda.empty_cache()
        del sample, target_latent, target_mesh, condition
    metrics = (
        "chamfer_l1_improvement",
        "fscore_0p02_delta",
        "normal_consistency_delta",
        "largest_component_ratio_delta",
    )
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[row["object_uid"]].append(row)
    object_rows = [
        {
            "object_uid": uid,
            **{
                name: float(np.mean([float(row[name]) for row in rows]))
                for name in metrics
            },
        }
        for uid, rows in sorted(grouped.items())
    ]
    summary = {
        name: summarize(
            [float(row[name]) for row in object_rows],
            bootstrap_samples=int(args.bootstrap_samples),
            seed=20260813 + index,
        )
        for index, name in enumerate(metrics)
    }
    report = {
        "format": REPORT_FORMAT,
        "passed": len(records) == len(object_indices) * len(seeds),
        "formal": False,
        "run_config": run_config,
        "model_summary": model_summary,
        "object_count": len(object_rows),
        "record_count": len(records),
        "summary": summary,
        "object_rows": object_rows,
        "records": records,
        "scope_guard": (
            "official-target GT-support training-overlap fit diagnosis only; no "
            "generalization claim and no Native-SS predicted-support claim"
            if official_split == "train"
            else "official-target GT-support development generalization diagnosis "
            "only; no final claim and no Native-SS predicted-support claim"
        ),
    }
    report["report_sha256"] = canonical_sha256(report)
    (output / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"passed": report["passed"], "arm": args.arm, "objects": len(object_rows), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
