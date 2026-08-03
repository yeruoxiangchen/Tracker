#!/usr/bin/env python3
"""Same-coordinate/same-noise Mesh evaluation for native SLAT conditioning."""

from __future__ import annotations

import argparse
from collections import defaultdict
import gc
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from pose_point_depth_mv.direct_slat_flow import canonical_json_sha256
from pose_point_depth_mv.eval_direct_slat_flow import summarize
from pose_point_depth_mv.export_direct_flow_mesh_pairs import (
    canonical_coords,
    load_canonical_gt,
    mesh_structure_metrics,
    sparse_from_payload,
    sparse_noise_from_master,
    sparse_payload,
    surface_metrics,
)
from pose_point_depth_mv.native_3d_condition import (
    NATIVE_SLAT_FLOW_VERSION,
    NativeConditionSLatDataset,
    NativeStockSLatFlow,
    PositiveNativeSLatRolloutFlow,
    build_native_slat_components,
    load_trainable_state_dict,
    sha256_file,
    validate_lifting_feature_metadata,
    validate_native_checkpoint,
)
from pose_point_depth_mv.train_direct_slat_flow import to_device_tree
from trellis.modules import sparse as sp


REPORT_VERSION = "pose_point_depth_mv.native_slat_mesh_exploratory.v1"


def parse_csv(value: str, cast) -> list[Any]:
    result = [cast(item.strip()) for item in str(value).split(",") if item.strip()]
    if not result or len(result) != len(set(result)):
        raise ValueError("CSV values must be non-empty and unique")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache_manifest", required=True)
    parser.add_argument("--lifting_cache_manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--pretrained", default="Stable-X/trellis-vggt-v0-2")
    parser.add_argument("--indices", default="all")
    parser.add_argument("--joint_seeds", default="42,43,44")
    parser.add_argument("--max_objects", type=int, default=32)
    parser.add_argument("--object_offset", type=int, default=0)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--cfg_strength", type=float, default=5.0)
    parser.add_argument("--cfg_interval", default="0.5,1.0")
    parser.add_argument("--rescale_t", type=float, default=3.0)
    parser.add_argument("--condition_scale", type=float, default=1.0)
    parser.add_argument("--surface_samples", type=int, default=20000)
    parser.add_argument("--bootstrap_samples", type=int, default=5000)
    parser.add_argument("--amp_dtype", choices=("bf16", "fp16", "none"), default="bf16")
    parser.add_argument("--require_complete_matrix", action="store_true")
    return parser.parse_args()


def atomic_torch_save(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def select_matrix(
    rows: list[dict[str, Any]],
    *,
    joint_seeds: list[int],
    max_objects: int,
    object_offset: int,
    require_complete: bool,
) -> list[tuple[str, str, dict[int, int]]]:
    by_object: dict[str, dict[str, dict[int, int]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for index, row in enumerate(rows):
        object_uid = str(row["object_uid"])
        uid = str(row["uid"])
        seed = int(row["support_seed"])
        if seed in by_object[object_uid][uid]:
            raise RuntimeError("duplicate object/uid/support-seed row")
        by_object[object_uid][uid][seed] = index
    eligible: list[tuple[str, str, dict[int, int]]] = []
    required = set(joint_seeds)
    for object_uid, uid_rows in sorted(by_object.items()):
        candidates = [
            (uid, seed_map)
            for uid, seed_map in sorted(uid_rows.items())
            if required.issubset(seed_map)
        ]
        if candidates:
            uid, seed_map = candidates[0]
            eligible.append((object_uid, uid, seed_map))
    selected = eligible[int(object_offset) :]
    if int(max_objects) > 0:
        selected = selected[: int(max_objects)]
    if require_complete and (
        int(max_objects) <= 0
        or len(selected) != int(max_objects)
        or any(set(seed_map) < required for _, _, seed_map in selected)
    ):
        raise RuntimeError(
            f"incomplete requested Mesh matrix: objects={len(selected)} "
            f"expected={max_objects} seeds={joint_seeds}"
        )
    if not selected:
        raise RuntimeError("Mesh evaluation selected no complete object/seed matrix")
    return selected


@torch.no_grad()
def main() -> None:
    args = parse_args()
    joint_seeds = parse_csv(args.joint_seeds, int)
    cfg_interval = parse_csv(args.cfg_interval, float)
    if len(cfg_interval) != 2 or not 0 <= cfg_interval[0] <= cfg_interval[1] <= 1:
        raise ValueError("cfg_interval must satisfy 0 <= lo <= hi <= 1")
    if int(args.steps) <= 0 or int(args.surface_samples) <= 0:
        raise ValueError("steps/surface_samples must be positive")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    dataset = NativeConditionSLatDataset(
        args.cache_manifest,
        args.lifting_cache_manifest,
        indices=args.indices,
    )
    selections = select_matrix(
        dataset.rows,
        joint_seeds=joint_seeds,
        max_objects=int(args.max_objects),
        object_offset=int(args.object_offset),
        require_complete=bool(args.require_complete_matrix),
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    validate_native_checkpoint(
        checkpoint,
        expected_format=NATIVE_SLAT_FLOW_VERSION,
        pretrained=args.pretrained,
    )
    validate_lifting_feature_metadata(
        visual_feature_dim=dataset.lifting.visual_feature_dim,
        feature_metadata=dataset.lifting.feature_metadata,
        feature_source=str(checkpoint["args"]["feature_source"]),
    )
    device = torch.device("cuda")
    torch.cuda.set_device(0)
    built = build_native_slat_components(
        pretrained=args.pretrained,
        hidden_dim=int(checkpoint["args"]["hidden_dim"]),
        feature_source=str(checkpoint["args"]["feature_source"]),
        gradient_checkpointing=False,
        device=device,
        retain_pipeline=True,
    )
    sampler, model, sampler_defaults, normalization, model_summary, pipeline = built
    runtime_normalization = {
        key: [float(item) for item in value] for key, value in normalization.items()
    }
    if canonical_json_sha256(runtime_normalization) != dataset.slat_normalization_hash:
        raise RuntimeError("runtime SLAT normalization differs from Mesh cache")
    load_trainable_state_dict(model, checkpoint["model_trainable_state"])
    model.eval()
    params = dict(sampler_defaults)
    params.update(
        {
            "steps": int(args.steps),
            "cfg_strength": float(args.cfg_strength),
            "cfg_interval": cfg_interval,
            "rescale_t": float(args.rescale_t),
        }
    )
    run_config = {
        "format": REPORT_VERSION,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "checkpoint_step": int(checkpoint.get("step", -1)),
        "cache_manifest": str(Path(args.cache_manifest).resolve()),
        "cache_manifest_sha256": sha256_file(args.cache_manifest),
        "lifting_cache_manifest": str(Path(args.lifting_cache_manifest).resolve()),
        "lifting_cache_manifest_sha256": sha256_file(args.lifting_cache_manifest),
        "joint_seeds": joint_seeds,
        "object_offset": int(args.object_offset),
        "selected": [
            {"object_uid": object_uid, "uid": uid}
            for object_uid, uid, _ in selections
        ],
        "sampling": params,
        "condition_scale": float(args.condition_scale),
        "same_coordinates": True,
        "same_initial_noise": True,
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2) + "\n", encoding="utf-8"
    )
    use_amp = args.amp_dtype != "none"
    amp_dtype = torch.float16 if args.amp_dtype == "fp16" else torch.bfloat16
    latent_root = output_dir / "slat"
    pair_rows = []
    mean = torch.tensor(runtime_normalization["mean"], device=device)[None]
    std = torch.tensor(runtime_normalization["std"], device=device)[None]
    for object_position, (object_uid, uid, seed_map) in enumerate(selections):
        for joint_seed in joint_seeds:
            index = seed_map[joint_seed]
            sample = dataset[index]
            pair_id = f"obj_{object_position:04d}_seed_{joint_seed}"
            pair_dir = latent_root / pair_id
            branch_paths = {
                branch: pair_dir / f"{branch}.pt" for branch in ("stock", "full")
            }
            master_seed = int(joint_seed) * 2000003 + object_position * 2017 + 7919
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
            outputs: dict[str, Any] = {}
            flow_stats: dict[str, Any] = {}
            for branch in ("stock", "full"):
                noise = sp.SparseTensor(
                    feats=initial.feats.clone(), coords=initial.coords.clone()
                )
                if branch == "stock":
                    flow = NativeStockSLatFlow(model)
                else:
                    flow = PositiveNativeSLatRolloutFlow(
                        model,
                        condition["cond"],
                        sample["lifting_sample"],
                        condition_scale=float(args.condition_scale),
                    )
                with torch.autocast(
                    device_type="cuda", dtype=amp_dtype, enabled=use_amp
                ):
                    sampled = sampler.sample(
                        flow,
                        noise,
                        **condition,
                        **params,
                        verbose=False,
                    ).samples
                if branch == "full":
                    if flow.positive_calls <= 0:
                        raise RuntimeError("native full rollout never used positive branch")
                    flow_stats = flow.stats_summary()
                outputs[branch] = sampled * std + mean
            if not torch.equal(outputs["stock"].coords, outputs["full"].coords):
                raise RuntimeError("stock/full SLAT rollout coordinates diverged")
            for branch, value in outputs.items():
                atomic_torch_save(sparse_payload(value), branch_paths[branch])
            pair_rows.append(
                {
                    "pair_id": pair_id,
                    "object_position": object_position,
                    "object_uid": object_uid,
                    "uid": uid,
                    "joint_seed": joint_seed,
                    "dataset_index": index,
                    "coord_count": int(initial.coords.shape[0]),
                    "slat_master_seed": master_seed,
                    "same_coordinates": True,
                    "same_initial_noise": True,
                    "flow_stats": flow_stats,
                }
            )
            print(f"[native_slat_mesh:sample] {pair_id} {uid}", flush=True)
            del sample, master, initial, condition, outputs
            torch.cuda.empty_cache()

    model.cpu()
    del model
    gc.collect()
    torch.cuda.empty_cache()
    mesh_decoder = pipeline.models["slat_decoder_mesh"].to(device).eval()
    records = []
    for pair in pair_rows:
        sample = dataset[int(pair["dataset_index"])]
        target_mesh, target_metadata = load_canonical_gt(sample)
        pair_dir = output_dir / "mesh_pairs" / str(pair["pair_id"])
        branches = {}
        for branch in ("stock", "full"):
            payload = torch.load(
                latent_root / str(pair["pair_id"]) / f"{branch}.pt",
                map_location="cpu",
            )
            decoded = mesh_decoder(sparse_from_payload(payload, device))[0]
            canonical_mesh = decoded.to_trimesh(transform_pose=False)
            view_mesh = decoded.to_trimesh(transform_pose=True)
            structure = mesh_structure_metrics(canonical_mesh)
            if not structure["mesh_success"]:
                raise RuntimeError(f"empty/non-finite decoded Mesh: {pair['pair_id']} {branch}")
            branch_dir = pair_dir / branch
            branch_dir.mkdir(parents=True, exist_ok=True)
            canonical_mesh.export(branch_dir / "mesh_canonical.obj")
            view_mesh.export(branch_dir / "mesh_view.glb")
            metric_seed = int(pair["joint_seed"]) * 1009 + int(pair["object_position"]) * 9173
            surface = surface_metrics(
                canonical_mesh,
                target_mesh,
                count=int(args.surface_samples),
                seed=metric_seed,
                thresholds=(0.01, 0.02, 0.05),
            )
            branches[branch] = {"structure": structure, "surface": surface}
        comparison = {
            "chamfer_l1_improvement": (
                branches["stock"]["surface"]["chamfer_l1"]
                - branches["full"]["surface"]["chamfer_l1"]
            ),
            "fscore_0p02_delta": (
                branches["full"]["surface"]["fscore_0p02"]
                - branches["stock"]["surface"]["fscore_0p02"]
            ),
            "normal_consistency_delta": (
                branches["full"]["surface"]["normal_consistency"]
                - branches["stock"]["surface"]["normal_consistency"]
            ),
            "largest_component_ratio_delta": (
                branches["full"]["structure"]["largest_component_ratio"]
                - branches["stock"]["structure"]["largest_component_ratio"]
            ),
        }
        records.append(
            {
                **pair,
                "target": target_metadata,
                "branches": branches,
                **comparison,
            }
        )
        print(f"[native_slat_mesh:decode] {pair['pair_id']}", flush=True)
        del sample, target_mesh, branches
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
                name: float(np.mean([float(row[name]) for row in values]))
                for name in metric_names
            },
        }
        for object_uid, values in sorted(by_object.items())
    ]
    summary = {
        name: summarize(
            [float(row[name]) for row in object_rows],
            bootstrap_samples=int(args.bootstrap_samples),
            seed=73000 + position,
        )
        for position, name in enumerate(metric_names)
    }
    report = {
        "format": REPORT_VERSION,
        "formal": False,
        "checkpoint": run_config["checkpoint"],
        "checkpoint_step": run_config["checkpoint_step"],
        "object_count": len(object_rows),
        "record_count": len(records),
        "joint_seeds": joint_seeds,
        "same_coordinates": True,
        "same_initial_noise": True,
        "summary": summary,
        "object_rows": object_rows,
        "records": records,
        "scope_guard": (
            "exploratory G4 stock-vs-native-condition diagnosis only; use a fresh "
            "frozen blind protocol before a confirmatory science claim"
        ),
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "Native SLAT same-coordinate same-noise Mesh evaluation",
        "=" * 57,
        f"objects: {len(object_rows)}",
        f"records: {len(records)}",
        f"seeds: {joint_seeds}",
    ]
    for name in metric_names:
        row = summary[name]
        lines.append(
            f"{name}: mean={row['mean']:+.8f} median={row['median']:+.8f} "
            f"win={row['positive_rate']:.6f} CI={row['bootstrap_mean_95_ci']}"
        )
    lines.extend(("", report["scope_guard"]))
    (output_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
