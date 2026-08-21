#!/usr/bin/env python3
"""Same-coordinate/same-noise Mesh evaluation for Native-SLAT v3 view fusion."""

from __future__ import annotations

import argparse
from collections import defaultdict
import gc
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from pose_point_depth_mv.eval_direct_slat_flow import summarize
from pose_point_depth_mv.evaluate_native_ss_stock_slat_mesh import load_ss_evidence
from pose_point_depth_mv.export_direct_flow_mesh_pairs import (
    canonical_coords,
    load_canonical_gt,
    mesh_structure_metrics,
    sparse_noise_from_master,
    surface_metrics,
)
from pose_point_depth_mv.export_native_slat_mesh_pairs import select_matrix
from pose_point_depth_mv.native_3d_condition import NativeConditionSLatDataset
from pose_point_depth_mv.native_slat_genrecon import (
    NATIVE_SLAT_BASELINE,
    NATIVE_SLAT_GENRECON_VERSION,
    NativeSLatCalibratedCFGFlow,
    NativeSLatStockFlow,
    build_native_slat_genrecon_components,
    canonical_json_sha256,
    load_stock_slat_freeze,
    load_trainable_state_dict,
    sha256_file,
    validate_native_slat_genrecon_checkpoint,
)
from pose_point_depth_mv.train_direct_slat_flow import to_device_tree
from trellis.modules import sparse as sp


REPORT_VERSION = "pose_point_depth_mv.native_slat_genrecon_mesh.v3"


def parse_csv(value: str, cast) -> list[Any]:
    result = [cast(item.strip()) for item in str(value).split(",") if item.strip()]
    if not result or len(result) != len(set(result)):
        raise ValueError("CSV values must be non-empty and unique")
    return result


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
    return {key: value[key] for key in keys}


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache_manifest", required=True)
    parser.add_argument("--lifting_cache_manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--native_ss_report", required=True)
    parser.add_argument("--stock_slat_freeze", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--pretrained", default="Stable-X/trellis-vggt-v0-2")
    parser.add_argument("--weights", choices=("ema", "raw"), default="ema")
    parser.add_argument("--joint_seeds", default="42,43,44")
    parser.add_argument("--max_objects", type=int, default=32)
    parser.add_argument("--object_offset", type=int, default=0)
    parser.add_argument("--surface_samples", type=int, default=20000)
    parser.add_argument("--bootstrap_samples", type=int, default=5000)
    parser.add_argument("--amp_dtype", choices=("bf16", "fp16", "none"), default="bf16")
    parser.add_argument("--require_complete_matrix", action="store_true")
    return parser


@torch.no_grad()
def main() -> None:
    args = make_parser().parse_args()
    seeds = parse_csv(args.joint_seeds, int)
    if int(args.max_objects) <= 0 or int(args.surface_samples) <= 0:
        raise ValueError("max_objects/surface_samples must be positive")
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    dataset = NativeConditionSLatDataset(
        args.cache_manifest, args.lifting_cache_manifest, indices="all"
    )
    if dataset.config.get("condition_arch") != "native_ss_genrecon_v2":
        raise RuntimeError("Native-SLAT v3 Mesh evaluation requires the v2 cache contract")
    selections = select_matrix(
        dataset.rows,
        joint_seeds=seeds,
        max_objects=int(args.max_objects),
        object_offset=int(args.object_offset),
        require_complete=bool(args.require_complete_matrix),
    )
    _, ss_binding_all = load_ss_evidence(args.native_ss_report)
    if dataset.config.get("native_ss_deployment") != ss_binding_all:
        raise RuntimeError("Mesh cache and evaluation bind different Native SS deployments")
    ss_binding = upstream_binding(ss_binding_all)
    stock_freeze = load_stock_slat_freeze(args.stock_slat_freeze)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    validate_native_slat_genrecon_checkpoint(
        checkpoint,
        pretrained=args.pretrained,
        stock_slat_freeze=stock_freeze,
        upstream_native_ss=ss_binding,
    )
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    sampler, model, decoder, summary, defaults, normalization = (
        build_native_slat_genrecon_components(
            pretrained=args.pretrained,
            stock_slat_freeze=stock_freeze,
            upstream_native_ss=ss_binding,
            lora_rank=int(checkpoint["args"]["lora_rank"]),
            lora_alpha=int(checkpoint["args"]["lora_alpha"]),
            condition_channels=int(checkpoint["args"]["condition_channels"]),
            view_fusion_hidden_dim=int(checkpoint["args"]["view_fusion_hidden_dim"]),
            geometry_logit_scale_init=float(
                checkpoint["args"]["geometry_logit_scale_init"]
            ),
            gradient_checkpointing=False,
            need_decoder=True,
            device=device,
        )
    )
    if decoder is None:
        raise RuntimeError("Native-SLAT Mesh evaluation requires frozen decoder")
    state_key = "ema_trainable_state" if args.weights == "ema" else "model_trainable_state"
    load_trainable_state_dict(model, checkpoint[state_key])
    model.eval()
    decoder.eval()
    runtime_normalization = {
        key: [float(value) for value in values] for key, values in normalization.items()
    }
    if canonical_json_sha256(runtime_normalization) != dataset.slat_normalization_hash:
        raise RuntimeError("runtime SLAT normalization differs from cache")
    mean = torch.tensor(runtime_normalization["mean"], device=device)[None]
    std = torch.tensor(runtime_normalization["std"], device=device)[None]
    params = dict(defaults)
    params.setdefault("guidance_rescale", 0.0)
    if {
        "steps": int(params.get("steps", -1)),
        "cfg_strength": float(params.get("cfg_strength", -1)),
        "cfg_interval": list(params.get("cfg_interval", ())),
        "guidance_rescale": float(params.get("guidance_rescale", -1)),
        "rescale_t": float(params.get("rescale_t", -1)),
    } != {
        "steps": 25,
        "cfg_strength": 5.0,
        "cfg_interval": [0.5, 1.0],
        "guidance_rescale": 0.0,
        "rescale_t": 3.0,
    }:
        raise RuntimeError(f"Stock SLAT sampler defaults changed: {params}")
    run_config = {
        "format": REPORT_VERSION,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "checkpoint_step": int(checkpoint["step"]),
        "weights": args.weights,
        "baseline": NATIVE_SLAT_BASELINE,
        "same_native_ss_coordinates": True,
        "same_initial_noise": True,
        "native_ss": ss_binding,
        "stock_slat_freeze_sha256": stock_freeze["freeze_sha256"],
        "sampling": params,
        "seeds": seeds,
        "selected": [
            {"object_uid": object_uid, "uid": uid}
            for object_uid, uid, _ in selections
        ],
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    amp_enabled = args.amp_dtype != "none"
    amp_dtype = torch.float16 if args.amp_dtype == "fp16" else torch.bfloat16
    records = []
    for object_position, (object_uid, uid, seed_map) in enumerate(selections):
        for seed in seeds:
            sample = dataset[seed_map[seed]]
            coords_np = canonical_coords(
                sample["corrected_coords64"].numpy(), resolution=64
            )
            master_seed = int(seed) * 2000003 + object_position * 2017 + 7919
            generator = torch.Generator(device=device).manual_seed(master_seed)
            master = torch.randn(
                (64, 64, 64, 8),
                generator=generator,
                device=device,
                dtype=torch.float32,
            )
            initial = sparse_noise_from_master(coords_np, master, device=device)
            condition = to_device_tree(sample["condition"], device)
            outputs = {}
            wrapper_summary = {}
            for branch in ("stock", "full"):
                noise = sp.SparseTensor(
                    feats=initial.feats.clone(), coords=initial.coords.clone()
                )
                if branch == "stock":
                    flow = NativeSLatStockFlow(model)
                else:
                    flow = NativeSLatCalibratedCFGFlow(
                        model,
                        condition["cond"],
                        sample["lifting_sample"],
                        enabled=True,
                    )
                with torch.autocast(
                    device_type="cuda", dtype=amp_dtype, enabled=amp_enabled
                ):
                    outputs[branch] = sampler.sample(
                        flow,
                        noise,
                        **condition,
                        **params,
                        verbose=False,
                    ).samples
                if branch == "full":
                    wrapper_summary = flow.summary()
                    if flow.positive_calls <= 0 or flow.negative_calls <= 0:
                        raise RuntimeError("Native-SLAT standard CFG missed a branch")
            if not torch.equal(outputs["stock"].coords, outputs["full"].coords):
                raise RuntimeError("Stock/Full changed Native SS sparse coordinates")
            target_mesh, target_metadata = load_canonical_gt(sample)
            branches = {}
            pair_id = f"obj_{object_position:04d}_seed_{seed}"
            for branch, latent in outputs.items():
                decoded = decoder(latent * std + mean)[0]
                mesh = decoded.to_trimesh(transform_pose=False)
                structure = mesh_structure_metrics(mesh)
                if not structure["mesh_success"]:
                    raise RuntimeError(f"empty/non-finite Mesh: {pair_id} {branch}")
                branch_dir = output_dir / "mesh_pairs" / pair_id / branch
                branch_dir.mkdir(parents=True, exist_ok=True)
                mesh.export(branch_dir / "mesh_canonical.obj")
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
                    "object_uid": object_uid,
                    "uid": uid,
                    "seed": seed,
                    "same_native_ss_coordinates": True,
                    "same_initial_noise": True,
                    "coord_count": int(initial.coords.shape[0]),
                    "target": target_metadata,
                    "wrapper": wrapper_summary,
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
            print(f"[native_slat_v3_mesh] {pair_id} {uid}", flush=True)
            del sample, master, initial, condition, outputs, target_mesh, branches
            torch.cuda.empty_cache()
    metric_names = (
        "chamfer_l1_improvement",
        "fscore_0p02_delta",
        "normal_consistency_delta",
        "largest_component_ratio_delta",
    )
    by_object: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        by_object[str(row["object_uid"])].append(row)
    object_rows = [
        {
            "object_uid": object_uid,
            **{
                name: float(np.mean([float(row[name]) for row in rows]))
                for name in metric_names
            },
        }
        for object_uid, rows in sorted(by_object.items())
    ]
    metric_summary = {
        name: summarize(
            [float(row[name]) for row in object_rows],
            bootstrap_samples=int(args.bootstrap_samples),
            seed=20260802 + index,
        )
        for index, name in enumerate(metric_names)
    }
    report = {
        "format": REPORT_VERSION,
        "passed": len(records) == len(selections) * len(seeds),
        "formal": False,
        "run_config": run_config,
        "model_summary": summary,
        "object_count": len(object_rows),
        "record_count": len(records),
        "summary": metric_summary,
        "object_rows": object_rows,
        "records": records,
        "scope_guard": (
            "development comparison of Native-SLAT v3 learned view fusion against the frozen "
            "Native-SS+Stock-SLAT baseline; freeze a later untouched protocol only "
            "after architecture/checkpoint selection is complete"
        ),
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    lines = [
        "Native-SLAT v3 same-Native-SS-coordinate Mesh evaluation",
        "=" * 61,
        f"baseline: {NATIVE_SLAT_BASELINE}",
        f"objects: {len(object_rows)} records: {len(records)} weights: {args.weights}",
    ]
    for name in metric_names:
        row = metric_summary[name]
        lines.append(
            f"{name}: mean={row['mean']:+.8f} median={row['median']:+.8f} "
            f"win={row['positive_rate']:.6f} CI={row['bootstrap_mean_95_ci']}"
        )
    lines.extend(("", report["scope_guard"]))
    (output_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)
    model.cpu()
    decoder.cpu()
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
