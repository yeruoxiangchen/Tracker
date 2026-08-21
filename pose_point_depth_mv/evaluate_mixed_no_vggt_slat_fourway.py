#!/usr/bin/env python3
"""Four-way SLat mechanism diagnostic on the frozen 1:1 mixed cache.

The four branches share the exact same predicted Native-SS support, initial
SLat noise, sampler and frozen Mesh decoder:

* ``stock``: released Stock SLat (LoRA and posed-DINO residual both off),
* ``lora_only``: trained LoRA on, every-block posed-DINO residual exactly zero,
* ``pose_cyclic1``: trained LoRA/residual on, camera poses cyclically mismatched,
* ``correct``: trained LoRA/residual on with the correct image/pose binding.

The current mixed manifest is the M8 training cache.  Results from this file
are therefore deliberately marked training-overlap/mechanism-only, even when
all execution checks pass.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager
import gc
import json
from pathlib import Path
import re
from typing import Any, Iterator

import numpy as np
import torch
from torch import nn

from pose_point_depth_mv.direct_slat_flow import canonical_json_sha256
from pose_point_depth_mv.eval_direct_slat_flow import summarize
from pose_point_depth_mv.export_direct_flow_mesh_pairs import (
    canonical_coords,
    load_canonical_gt,
    mesh_structure_metrics,
    sparse_noise_from_master,
    surface_metrics,
)
from pose_point_depth_mv.mixed_no_vggt_data import (
    MixedNativeConditionSLatDataset,
    REQUIRED_DOMAINS,
)
from pose_point_depth_mv.native_slat_genrecon import (
    NativeSLatCalibratedCFGFlow,
    NativeSLatStockFlow,
    _same_condition_identity,
)
from pose_point_depth_mv.native_slat_genrecon_no_vggt import (
    NATIVE_SLAT_NO_VGGT_VERSION,
    build_native_slat_no_vggt_components,
    validate_native_slat_no_vggt_checkpoint,
)
from pose_point_depth_mv.native_slat_genrecon_v2 import (
    load_stock_slat_freeze,
    load_trainable_state_dict,
)
from pose_point_depth_mv.omni_real_benchmark_common import sha256_file
from pose_point_depth_mv.real_full_no_vggt_migration import (
    load_migration_contract,
    migration_summary,
    validate_destination_migration,
)
from pose_point_depth_mv.train_direct_slat_flow import to_device_tree
from pose_point_depth_mv.train_native_slat_genrecon import (
    checkpoint_stock_context_views,
    select_stock_context_views,
)
from trellis.modules import sparse as sp


REPORT_FORMAT = "pose_point_depth_mv.mixed_no_vggt_slat_fourway.v1"
BRANCHES = ("stock", "lora_only", "pose_cyclic1", "correct")
COMPARISONS = (
    ("correct", "stock", "total_trained_slat"),
    ("correct", "lora_only", "posed_dino_increment"),
    ("correct", "pose_cyclic1", "correct_pose_specificity"),
    ("lora_only", "stock", "generic_lora_increment"),
)
SURFACE_METRICS = (
    "chamfer_l1",
    "chamfer_l2",
    "fscore_0p01",
    "fscore_0p02",
    "fscore_0p05",
    "normal_consistency",
)
STRUCTURE_METRICS = ("largest_component_ratio", "component_count")


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache_manifest", required=True)
    parser.add_argument("--lifting_cache_manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--migration_contract", required=True)
    parser.add_argument("--stock_slat_freeze", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--pretrained", default="Stable-X/trellis-vggt-v0-2")
    parser.add_argument("--weights", choices=("ema", "raw"), default="ema")
    parser.add_argument("--objects_per_domain", type=int, default=16)
    parser.add_argument("--selection_seed", type=int, default=20260811)
    parser.add_argument("--noise_seed", type=int, default=42)
    parser.add_argument("--surface_samples", type=int, default=20000)
    parser.add_argument("--bootstrap_samples", type=int, default=10000)
    parser.add_argument("--amp_dtype", choices=("bf16", "fp16", "none"), default="bf16")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--verify_cache_hashes", action="store_true")
    return parser


def _upstream_binding(value: dict[str, Any]) -> dict[str, Any]:
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


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _safe_name(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._")
    return result[:96] or "object"


def decode_mesh_fp32(
    decoder: nn.Module,
    latent: Any,
    *,
    std: torch.Tensor,
    mean: torch.Tensor,
) -> Any:
    """Run Cube2Mesh in FP32, outside the SLat sampling autocast region.

    Cube2Mesh uses FP32 scatter accumulation buffers.  Allowing autocast to
    produce BF16/FP16 vertex attributes makes ``torch.scatter_reduce`` reject
    the mixed dtypes before a Mesh can be extracted.
    """

    with torch.autocast(device_type=latent.device.type, enabled=False):
        normalized = (latent * std + mean).float()
        return decoder(normalized)[0]


def select_balanced_objects(
    rows: list[dict[str, Any]], *, objects_per_domain: int, seed: int
) -> list[dict[str, Any]]:
    """Select equal object counts, preferring the largest-view row per object."""

    if int(objects_per_domain) <= 0:
        raise ValueError("objects_per_domain must be positive")
    grouped: dict[str, dict[str, list[tuple[int, dict[str, Any]]]]] = {
        domain: defaultdict(list) for domain in REQUIRED_DOMAINS
    }
    for index, row in enumerate(rows):
        domain = str(row.get("_mixed_domain", ""))
        object_uid = str(row.get("object_uid", ""))
        if domain not in grouped or not object_uid:
            raise ValueError("mixed row lacks frozen domain/object identity")
        grouped[domain][object_uid].append((index, row))

    chosen_by_domain: dict[str, list[dict[str, Any]]] = {}
    for domain_index, domain in enumerate(REQUIRED_DOMAINS):
        object_uids = sorted(grouped[domain])
        if len(object_uids) < int(objects_per_domain):
            raise RuntimeError(
                f"{domain} has only {len(object_uids)} objects; "
                f"requested {objects_per_domain}"
            )
        generator = np.random.default_rng(int(seed) + domain_index * 1_000_003)
        order = generator.permutation(len(object_uids))[: int(objects_per_domain)]
        selected: list[dict[str, Any]] = []
        for object_position in order.tolist():
            object_uid = object_uids[int(object_position)]
            # A deterministic largest-view sequence removes sequence-count bias
            # while keeping every object represented once.
            candidates = sorted(
                grouped[domain][object_uid],
                key=lambda item: (
                    -int(item[1].get("view_count", 0)),
                    str(item[1].get("uid", "")),
                    int(item[1].get("support_seed", 0)),
                ),
            )
            index, row = candidates[0]
            selected.append(
                {
                    "dataset_index": int(index),
                    "domain": domain,
                    "object_uid": object_uid,
                    "uid": str(row["uid"]),
                    "support_seed": int(row["support_seed"]),
                    "view_count": int(row.get("view_count", 0)),
                }
            )
        chosen_by_domain[domain] = selected

    # Interleaving makes a partial/resumed run domain-balanced at every prefix.
    return [
        chosen_by_domain[domain][position]
        for position in range(int(objects_per_domain))
        for domain in REQUIRED_DOMAINS
    ]


@contextmanager
def zero_every_block_condition(model: nn.Module) -> Iterator[None]:
    """Zero the complete projected residual, including every learned bias."""

    block_condition = getattr(model, "block_condition", None)
    if not isinstance(block_condition, nn.Module):
        raise TypeError("SLat model lacks an every-block condition module")

    def zero_output(_module: nn.Module, _inputs: Any, output: torch.Tensor):
        return torch.zeros_like(output)

    handle = block_condition.register_forward_hook(zero_output)
    try:
        yield
    finally:
        handle.remove()


class LoRAOnlyCFGFlow(nn.Module):
    """Keep trained LoRA active while exactly removing all 3D residuals."""

    def __init__(self, model: nn.Module, positive_condition: Any) -> None:
        super().__init__()
        self.model = model
        self.positive_condition = positive_condition
        self.positive_calls = 0
        self.negative_calls = 0
        self.delta_rms: list[float] = []

    def forward(self, x: Any, t: torch.Tensor, condition: Any) -> Any:
        positive = _same_condition_identity(condition, self.positive_condition)
        self.positive_calls += int(positive)
        self.negative_calls += int(not positive)
        # sample=None alone is insufficient: learned Linear biases would remain.
        with zero_every_block_condition(self.model):
            prediction, stats = self.model.adapted_prediction(
                x, t, condition, None, projection_mode="correct"
            )
        self.delta_rms.append(float(stats["flow_delta_rms"].detach().item()))
        return prediction

    def summary(self) -> dict[str, Any]:
        return {
            "mode": "lora_only",
            "positive_calls": self.positive_calls,
            "negative_calls": self.negative_calls,
            "posed_dino_input": False,
            "every_block_condition_output": "exact_zero_including_projection_bias",
            "mean_flow_delta_rms": (
                float(np.mean(self.delta_rms)) if self.delta_rms else 0.0
            ),
        }


def paired_improvements(
    left: dict[str, Any], right: dict[str, Any]
) -> dict[str, float]:
    """Return positive-is-left-better deltas for all registered metrics."""

    result = {
        "chamfer_l1": float(
            right["surface"]["chamfer_l1"] - left["surface"]["chamfer_l1"]
        ),
        "chamfer_l2": float(
            right["surface"]["chamfer_l2"] - left["surface"]["chamfer_l2"]
        ),
        "fscore_0p01": float(
            left["surface"]["fscore_0p01"] - right["surface"]["fscore_0p01"]
        ),
        "fscore_0p02": float(
            left["surface"]["fscore_0p02"] - right["surface"]["fscore_0p02"]
        ),
        "fscore_0p05": float(
            left["surface"]["fscore_0p05"] - right["surface"]["fscore_0p05"]
        ),
        "normal_consistency": float(
            left["surface"]["normal_consistency"]
            - right["surface"]["normal_consistency"]
        ),
        "largest_component_ratio": float(
            left["structure"]["largest_component_ratio"]
            - right["structure"]["largest_component_ratio"]
        ),
        "component_count": float(
            right["structure"]["component_count"]
            - left["structure"]["component_count"]
        ),
    }
    return result


def _group_names() -> tuple[str, ...]:
    return (*REQUIRED_DOMAINS, "mixed_macro_1to1")


def summarize_records(
    records: list[dict[str, Any]], *, bootstrap_samples: int
) -> dict[str, Any]:
    by_object: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    domains: dict[str, str] = {}
    for row in records:
        object_uid = str(row["object_uid"])
        branch = str(row["branch"])
        if branch in by_object[object_uid]:
            raise RuntimeError(f"duplicate object/branch result={object_uid}:{branch}")
        by_object[object_uid][branch] = row
        domains[object_uid] = str(row["domain"])
    incomplete = {
        object_uid: sorted(set(BRANCHES) - set(branches))
        for object_uid, branches in by_object.items()
        if set(branches) != set(BRANCHES)
    }
    if incomplete:
        raise RuntimeError(f"incomplete four-way matrix={incomplete}")

    groups = {
        domain: sorted(uid for uid, value in domains.items() if value == domain)
        for domain in REQUIRED_DOMAINS
    }
    groups["mixed_macro_1to1"] = sorted(by_object)
    absolute: dict[str, Any] = {}
    comparisons: dict[str, Any] = {}
    for group_index, group in enumerate(_group_names()):
        object_uids = groups[group]
        absolute[group] = {}
        for branch_index, branch in enumerate(BRANCHES):
            branch_rows = [by_object[uid][branch] for uid in object_uids]
            absolute[group][branch] = {
                metric: summarize(
                    [float(row["surface"][metric]) for row in branch_rows],
                    bootstrap_samples=int(bootstrap_samples),
                    seed=20260811 + group_index * 101 + branch_index * 17 + metric_index,
                )
                for metric_index, metric in enumerate(SURFACE_METRICS)
            }
            absolute[group][branch].update(
                {
                    metric: summarize(
                        [float(row["structure"][metric]) for row in branch_rows],
                        bootstrap_samples=int(bootstrap_samples),
                        seed=20261811
                        + group_index * 101
                        + branch_index * 17
                        + metric_index,
                    )
                    for metric_index, metric in enumerate(STRUCTURE_METRICS)
                }
            )
        comparisons[group] = {}
        for comparison_index, (left, right, label) in enumerate(COMPARISONS):
            deltas = [
                paired_improvements(by_object[uid][left], by_object[uid][right])
                for uid in object_uids
            ]
            comparisons[group][label] = {
                "left": left,
                "right": right,
                "positive_means_left_better": True,
                "metrics": {
                    metric: summarize(
                        [float(row[metric]) for row in deltas],
                        bootstrap_samples=int(bootstrap_samples),
                        seed=20262811
                        + group_index * 1009
                        + comparison_index * 97
                        + metric_index,
                    )
                    for metric_index, metric in enumerate(
                        (*SURFACE_METRICS, *STRUCTURE_METRICS)
                    )
                },
            }
    return {"absolute": absolute, "comparisons": comparisons}


def _reuse_branch_result(
    path: Path, *, expected: dict[str, Any], mesh_path: Path
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    row = json.loads(path.read_text(encoding="utf-8"))
    mismatch = {
        key: (row.get(key), value)
        for key, value in expected.items()
        if row.get(key) != value
    }
    if mismatch:
        raise RuntimeError(f"existing branch result identity differs: {mismatch}")
    if not mesh_path.is_file() or sha256_file(mesh_path) != row.get("mesh_sha256"):
        raise RuntimeError(f"existing branch Mesh is missing/changed: {mesh_path}")
    return row


@torch.no_grad()
def main() -> None:
    args = make_parser().parse_args()
    if int(args.surface_samples) <= 0 or int(args.bootstrap_samples) <= 0:
        raise ValueError("surface_samples/bootstrap_samples must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("four-way rollout requires CUDA")

    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists() and not args.resume:
        raise FileExistsError(f"output exists; pass --resume after inspection: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = MixedNativeConditionSLatDataset(
        args.cache_manifest,
        args.lifting_cache_manifest,
        indices="all",
        verify_hashes=bool(args.verify_cache_hashes),
    )
    selected = select_balanced_objects(
        dataset.rows,
        objects_per_domain=int(args.objects_per_domain),
        seed=int(args.selection_seed),
    )
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if checkpoint.get("format") != NATIVE_SLAT_NO_VGGT_VERSION:
        raise ValueError("four-way diagnostic requires the trained mixed no-VGGT SLat")
    contract = load_migration_contract(args.migration_contract, stage="slat")
    validate_destination_migration(checkpoint, contract)
    stock_context_views = checkpoint_stock_context_views(checkpoint)
    summary_context = (
        checkpoint.get("model_summary", {})
        .get("stock_cross_attention_context", {})
        .get("views")
    )
    if summary_context is not None and str(summary_context) != stock_context_views:
        raise RuntimeError("checkpoint args/model summary Stock context policy differs")
    upstream = dict(checkpoint.get("model_summary", {}).get("upstream_native_ss", {}))
    dataset_upstream = _upstream_binding(dict(dataset.config["native_ss_deployment"]))
    if dataset_upstream != upstream:
        raise RuntimeError("mixed cache and SLat checkpoint bind different Native SS")
    stock_freeze = load_stock_slat_freeze(args.stock_slat_freeze)
    validate_native_slat_no_vggt_checkpoint(
        checkpoint,
        pretrained=args.pretrained,
        stock_slat_freeze=stock_freeze,
        upstream_native_ss=upstream,
        allow_v2_parent=False,
    )

    checkpoint_sha = sha256_file(checkpoint_path)
    run_config = {
        "format": REPORT_FORMAT,
        "formal": False,
        "training_overlap": True,
        "scope": "M8 mixed-cache mechanism diagnostic; not generalization evidence",
        "branches": list(BRANCHES),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_step": int(checkpoint["step"]),
        "weights": args.weights,
        "stock_context_views": stock_context_views,
        "posed_dino_3d_views": "all cached views",
        "migration_contract": migration_summary(contract),
        "stock_slat_freeze": str(Path(args.stock_slat_freeze).resolve()),
        "stock_slat_freeze_sha256": stock_freeze["freeze_sha256"],
        "cache_identity": dict(dataset.identity),
        "selection": {
            "policy": "seeded_equal_domain_objects_largest_view_row",
            "objects_per_domain": int(args.objects_per_domain),
            "seed": int(args.selection_seed),
            "rows": selected,
        },
        "noise_seed": int(args.noise_seed),
        "surface_samples": int(args.surface_samples),
        "amp_dtype": args.amp_dtype,
    }
    run_config_path = output_dir / "run_config.json"
    if run_config_path.exists():
        existing = json.loads(run_config_path.read_text(encoding="utf-8"))
        if canonical_json_sha256(existing) != canonical_json_sha256(run_config):
            raise RuntimeError("--resume run_config differs from existing output")
    else:
        _atomic_json(run_config_path, run_config)
    run_config_sha = canonical_json_sha256(run_config)

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    sampler, model, decoder, model_summary, defaults, normalization = (
        build_native_slat_no_vggt_components(
            pretrained=args.pretrained,
            stock_slat_freeze=stock_freeze,
            upstream_native_ss=upstream,
            lora_rank=int(checkpoint["args"]["lora_rank"]),
            lora_alpha=int(checkpoint["args"]["lora_alpha"]),
            condition_channels=int(checkpoint["args"]["condition_channels"]),
            gradient_checkpointing=False,
            need_decoder=True,
            device=device,
        )
    )
    if decoder is None:
        raise RuntimeError("frozen SLat Mesh decoder is unavailable")
    state_key = "ema_trainable_state" if args.weights == "ema" else "model_trainable_state"
    load_trainable_state_dict(model, checkpoint[state_key])
    model.eval()
    decoder.eval()

    runtime_normalization = {
        key: [float(value) for value in values] for key, values in normalization.items()
    }
    if canonical_json_sha256(runtime_normalization) != dataset.slat_normalization_hash:
        raise RuntimeError("runtime SLat normalization differs from mixed cache")
    mean = torch.tensor(runtime_normalization["mean"], device=device)[None]
    std = torch.tensor(runtime_normalization["std"], device=device)[None]
    params = dict(defaults)
    params.setdefault("guidance_rescale", 0.0)
    sampling_contract = {
        "steps": int(params.get("steps", -1)),
        "cfg_strength": float(params.get("cfg_strength", -1)),
        "cfg_interval": list(params.get("cfg_interval", ())),
        "guidance_rescale": float(params.get("guidance_rescale", -1)),
        "rescale_t": float(params.get("rescale_t", -1)),
    }
    expected_sampling = {
        "steps": 25,
        "cfg_strength": 5.0,
        "cfg_interval": [0.5, 1.0],
        "guidance_rescale": 0.0,
        "rescale_t": 3.0,
    }
    if sampling_contract != expected_sampling:
        raise RuntimeError(f"Stock SLat sampling contract changed: {sampling_contract}")

    amp_enabled = args.amp_dtype != "none"
    amp_dtype = torch.float16 if args.amp_dtype == "fp16" else torch.bfloat16
    records: list[dict[str, Any]] = []
    for object_position, selection in enumerate(selected):
        sample = dataset[int(selection["dataset_index"])]
        object_uid = str(selection["object_uid"])
        uid = str(selection["uid"])
        domain = str(selection["domain"])
        coords_np = canonical_coords(
            sample["corrected_coords64"].numpy(), resolution=64
        )
        master_seed = (
            int(args.noise_seed) * 2_000_003 + object_position * 2017 + 7919
        )
        generator = torch.Generator(device=device).manual_seed(master_seed)
        master = torch.randn(
            (64, 64, 64, 8), generator=generator, device=device, dtype=torch.float32
        )
        initial = sparse_noise_from_master(coords_np, master, device=device)
        condition_all = to_device_tree(sample["condition"], device)
        condition = {
            "cond": select_stock_context_views(
                condition_all["cond"], stock_context_views
            ),
            "neg_cond": select_stock_context_views(
                condition_all["neg_cond"], stock_context_views
            ),
        }
        target_mesh, target_metadata = load_canonical_gt(sample)
        object_dir = (
            output_dir
            / "objects"
            / f"{object_position:03d}_{domain}_{_safe_name(object_uid)}"
        )
        for branch in BRANCHES:
            branch_dir = object_dir / branch
            result_path = branch_dir / "result.json"
            mesh_path = branch_dir / "mesh_canonical.obj"
            expected_identity = {
                "format": REPORT_FORMAT,
                "run_config_sha256": run_config_sha,
                "branch": branch,
                "domain": domain,
                "object_uid": object_uid,
                "uid": uid,
                "support_seed": int(selection["support_seed"]),
                "checkpoint_sha256": checkpoint_sha,
                "stock_context_views": stock_context_views,
            }
            reused = _reuse_branch_result(
                result_path, expected=expected_identity, mesh_path=mesh_path
            )
            if reused is not None:
                records.append(reused)
                print(
                    f"[mixed_slat_fourway] reuse {object_position + 1}/{len(selected)} "
                    f"{domain} {object_uid} {branch}",
                    flush=True,
                )
                continue

            noise = sp.SparseTensor(
                feats=initial.feats.clone(), coords=initial.coords.clone()
            )
            if branch == "stock":
                flow: nn.Module = NativeSLatStockFlow(model)
                wrapper = {
                    "mode": "stock",
                    "lora": False,
                    "posed_dino_residual": False,
                }
            elif branch == "lora_only":
                flow = LoRAOnlyCFGFlow(model, condition["cond"])
                wrapper = None
            else:
                projection_mode = "correct" if branch == "correct" else "pose_cyclic1"
                flow = NativeSLatCalibratedCFGFlow(
                    model,
                    condition["cond"],
                    sample["lifting_sample"],
                    enabled=True,
                    projection_mode=projection_mode,
                )
                wrapper = None
            with torch.autocast(
                device_type="cuda", dtype=amp_dtype, enabled=amp_enabled
            ):
                latent = sampler.sample(
                    flow, noise, **condition, **params, verbose=False
                ).samples
            decoded = decode_mesh_fp32(decoder, latent, std=std, mean=mean)
            if not torch.equal(latent.coords, initial.coords):
                raise RuntimeError(f"{branch} changed the frozen Native-SS support")
            if branch != "stock":
                wrapper = flow.summary()  # type: ignore[attr-defined]
                if int(wrapper["positive_calls"]) <= 0 or int(wrapper["negative_calls"]) <= 0:
                    raise RuntimeError(f"{branch} did not execute both CFG branches")
                wrapper["projection_mode"] = (
                    "none" if branch == "lora_only" else branch
                )
            mesh = decoded.to_trimesh(transform_pose=False)
            structure = mesh_structure_metrics(mesh)
            if not structure["mesh_success"]:
                raise RuntimeError(f"empty/non-finite Mesh: {object_uid}:{branch}")
            branch_dir.mkdir(parents=True, exist_ok=True)
            temporary_mesh = branch_dir / ".mesh_canonical.tmp.obj"
            mesh.export(temporary_mesh, file_type="obj")
            temporary_mesh.replace(mesh_path)
            metric_seed = int(args.noise_seed) * 1009 + object_position * 9173
            surface = surface_metrics(
                mesh,
                target_mesh,
                count=int(args.surface_samples),
                seed=metric_seed,
                thresholds=(0.01, 0.02, 0.05),
            )
            row = {
                **expected_identity,
                "view_count": int(selection["view_count"]),
                "stock_context_view_count": len(condition["cond"]),
                "posed_dino_view_count": int(selection["view_count"]),
                "same_native_ss_coordinates": True,
                "same_initial_noise": True,
                "master_noise_seed": master_seed,
                "coord_count": int(initial.coords.shape[0]),
                "target": target_metadata,
                "mesh": str(mesh_path),
                "mesh_sha256": sha256_file(mesh_path),
                "wrapper": wrapper,
                "surface": surface,
                "structure": structure,
            }
            _atomic_json(result_path, row)
            records.append(row)
            print(
                f"[mixed_slat_fourway] {object_position + 1}/{len(selected)} "
                f"{domain} {object_uid} {branch}",
                flush=True,
            )
            del noise, flow, latent, decoded, mesh
            torch.cuda.empty_cache()
        del sample, master, initial, condition_all, condition, target_mesh
        gc.collect()
        torch.cuda.empty_cache()

    summary = summarize_records(
        records, bootstrap_samples=int(args.bootstrap_samples)
    )
    expected_records = int(args.objects_per_domain) * len(REQUIRED_DOMAINS) * len(BRANCHES)
    report = {
        "format": REPORT_FORMAT,
        "passed": len(records) == expected_records,
        "formal": False,
        "training_overlap": True,
        "object_count": int(args.objects_per_domain) * len(REQUIRED_DOMAINS),
        "objects_per_domain": int(args.objects_per_domain),
        "record_count": len(records),
        "branches": list(BRANCHES),
        "stock_context_views": stock_context_views,
        "posed_dino_3d_views": "all cached views",
        "same_native_ss_coordinates": True,
        "same_initial_noise": True,
        "run_config": run_config,
        "model_summary": model_summary,
        "sampling": sampling_contract,
        "summary": summary,
        "records": records,
        "scope_guard": (
            "The selected synthetic868+real376 cache is the M8 training cache. "
            "This report diagnoses whether the frozen checkpoint reacts specifically "
            "to correct posed-DINO evidence; it cannot establish held-out generalization "
            "or a publishable 5K scaling gain."
        ),
    }
    body = dict(report)
    report["report_sha256"] = canonical_json_sha256(body)
    _atomic_json(output_dir / "report.json", report)

    lines = [
        "Mixed no-VGGT SLat four-way mechanism diagnostic",
        "=" * 56,
        f"passed: {report['passed']}",
        "formal: false",
        "training overlap: true",
        f"objects: {report['object_count']} "
        f"({args.objects_per_domain} synthetic + {args.objects_per_domain} real)",
        "branches: " + ", ".join(BRANCHES),
        f"Stock cross-attention views: {stock_context_views}",
        "posed-DINO 3D views: all cached views",
        "",
    ]
    for group in _group_names():
        lines.append(f"[{group}]")
        for _, _, label in COMPARISONS:
            metrics = summary["comparisons"][group][label]["metrics"]
            chamfer = metrics["chamfer_l1"]
            fscore = metrics["fscore_0p02"]
            normal = metrics["normal_consistency"]
            lines.append(
                f"{label}: chamfer={chamfer['mean']:+.8f} "
                f"median={chamfer['median']:+.8f} win={chamfer['positive_rate']:.4f}; "
                f"f@0.02={fscore['mean']:+.8f}; normal={normal['mean']:+.8f}"
            )
        lines.append("")
    lines.append(report["scope_guard"])
    (output_dir / "summary.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print("\n".join(lines), flush=True)
    model.cpu()
    decoder.cpu()
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
