#!/usr/bin/env python3
"""Exploratory same-coordinate/same-noise Mesh test for direct SLAT Flow."""

from __future__ import annotations

import argparse
from collections import defaultdict
import gc
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

os.environ.setdefault("SPCONV_ALGO", "native")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
os.environ.setdefault("TORCH_EXTENSIONS_DIR", "/tmp/torch_extensions")

import numpy as np
import torch


TRACKER_ROOT = Path(__file__).resolve().parents[1]
RECONVIAGEN_ROOT = TRACKER_ROOT / "ReconViaGen"
VGGT_ROOT = RECONVIAGEN_ROOT / "wheels" / "vggt"
for path in (TRACKER_ROOT, RECONVIAGEN_ROOT, VGGT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from pose_point_depth_mv.direct_slat_data import (  # noqa: E402
    DirectSLatCacheDataset,
    sha256_file,
)
from pose_point_depth_mv.direct_slat_flow import (  # noqa: E402
    DIRECT_SLAT_FLOW_VERSION,
    DIRECT_SLAT_TRAINING_SEMANTICS_V3,
    DIRECT_SLAT_TRAINING_SEMANTICS_V4,
    DIRECT_SLAT_TRAINING_SEMANTICS_V5,
    NativeStockSLATFlow,
    PostCFGSupportSLATRolloutFlow,
    PositiveSupportSLATRolloutFlow,
    SLAT_GUIDED_DELTA_POLICIES,
    SLAT_GUIDED_DELTA_POLICY_LEGACY,
    SLAT_GUIDED_DELTA_POLICY_V2,
    SLAT_DELTA_BOUND_MODES,
    SLAT_SUPPORT_INTERVAL_POLICIES,
    SLAT_ROLLOUT_COMPONENTS,
    SLAT_ROLLOUT_COMPONENT_FULL,
    SLAT_RESIDUAL_COMBINATION_POLICIES,
    assert_disjoint_object_splits,
    build_direct_slat_components,
    canonical_json_sha256,
    load_strict_trainable_state,
    resolve_slat_guided_delta_policy,
    resolve_slat_delta_bound_mode,
    resolve_slat_support_interval_policy,
    resolve_slat_residual_combination_policy,
    support_generator_identity,
)
from pose_point_depth_mv.direct_slat_matched_mesh_blind import (  # noqa: E402
    REPORT_FORMAT as MATCHED_BLIND_REPORT_FORMAT,
    bind_file as bind_blind_file,
    canonical_sha256 as blind_canonical_sha256,
    formal_decision,
    load_protocol as load_matched_blind_protocol,
    summarize_seed_directions,
)
from pose_point_depth_mv.eval_direct_slat_flow import summarize  # noqa: E402
from pose_point_depth_mv.export_direct_flow_mesh_pairs import (  # noqa: E402
    canonical_coords,
    load_canonical_gt,
    mesh_structure_metrics,
    sparse_from_payload,
    sparse_noise_from_master,
    sparse_payload,
    surface_metrics,
)
from pose_point_depth_mv.train_direct_slat_flow import to_device_tree  # noqa: E402
from pose_point_depth_mv.freeze_matched_view_protocol import (  # noqa: E402
    stable_identity_seed,
)
from trellis.modules import sparse as sp  # noqa: E402


def parse_csv(value: str, cast) -> list[Any]:
    result = [cast(item.strip()) for item in str(value).split(",") if item.strip()]
    if not result or len(result) != len(set(result)):
        raise ValueError("CSV values must be non-empty and unique")
    return result


def atomic_torch_save(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def atomic_json_save(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache_manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--train_cache_manifest", default="")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--pretrained", default="Stable-X/trellis-vggt-v0-2")
    parser.add_argument(
        "--frozen_protocol",
        default="",
        help=(
            "Optional formal matched-coordinate blind protocol. When supplied, "
            "the protocol owns object selection, seeds, sampler and statistics."
        ),
    )
    parser.add_argument("--joint_seeds", default="42,43,44")
    parser.add_argument(
        "--uids",
        default="",
        help=(
            "Optional comma-separated exact sequence UIDs. This is intended "
            "for frozen seen-set sentinel/matrix diagnostics."
        ),
    )
    parser.add_argument("--max_objects", type=int, default=0)
    parser.add_argument(
        "--object_offset",
        type=int,
        default=0,
        help=(
            "Skip this many object_uid-sorted eligible objects before applying "
            "--max_objects. Use a non-zero offset only for a preregistered "
            "fresh holdout."
        ),
    )
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--cfg_strength", type=float, default=5.0)
    parser.add_argument("--cfg_interval", default="0.5,1.0")
    parser.add_argument("--rescale_t", type=float, default=3.0)
    parser.add_argument("--support_scale", type=float, default=None)
    parser.add_argument(
        "--slat_delta_scale",
        type=float,
        default=None,
        help="Override checkpoint Direct-SLAT residual scale.",
    )
    parser.add_argument(
        "--slat_delta_rms_ratio_cap",
        type=float,
        default=None,
        help=(
            "Override checkpoint per-batch RMS ratio cap; a negative value "
            "explicitly disables clipping."
        ),
    )
    parser.add_argument(
        "--slat_guided_delta_policy",
        choices=SLAT_GUIDED_DELTA_POLICIES,
        default=None,
        help=(
            "Override checkpoint rollout scope. post_cfg_v2 bounds the final "
            "guided Full-stock velocity; legacy checkpoints default to "
            "positive_branch_v1."
        ),
    )
    parser.add_argument(
        "--slat_delta_bound_mode",
        choices=SLAT_DELTA_BOUND_MODES,
        default=None,
    )
    parser.add_argument(
        "--support_interval_policy",
        choices=SLAT_SUPPORT_INTERVAL_POLICIES,
        default=None,
    )
    parser.add_argument(
        "--slat_residual_combination_policy",
        choices=SLAT_RESIDUAL_COMBINATION_POLICIES,
        default=None,
    )
    parser.add_argument("--slat_lora_delta_scale", type=float, default=None)
    parser.add_argument(
        "--slat_lora_delta_rms_ratio_cap", type=float, default=None
    )
    parser.add_argument("--slat_support_delta_scale", type=float, default=None)
    parser.add_argument(
        "--slat_support_delta_rms_ratio_cap", type=float, default=None
    )
    parser.add_argument("--surface_samples", type=int, default=20000)
    parser.add_argument("--bootstrap_samples", type=int, default=5000)
    parser.add_argument("--amp_dtype", choices=("bf16", "fp16", "none"), default="bf16")
    parser.add_argument(
        "--noise_identity_mode",
        choices=("object_position_v1", "object_uid_v1"),
        default="object_position_v1",
        help=(
            "object_position_v1 preserves legacy outputs; object_uid_v1 keeps "
            "SLAT noise fixed when object order/subset/view count changes"
        ),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--comparison_branches",
        default="stock,full",
        help=(
            "Exploratory Mesh branches. Legal values are stock, lora_only, "
            "adapter_only and full. Formal frozen protocols require stock,full."
        ),
    )
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    comparison_branches = parse_csv(args.comparison_branches, str)
    legal_branches = {"stock", *SLAT_ROLLOUT_COMPONENTS}
    if (
        any(branch not in legal_branches for branch in comparison_branches)
        or comparison_branches[0] != "stock"
        or SLAT_ROLLOUT_COMPONENT_FULL not in comparison_branches
    ):
        raise ValueError(
            "comparison_branches must start with stock, contain full, and use "
            f"only {sorted(legal_branches)}"
        )
    frozen_protocol = None
    if args.frozen_protocol:
        frozen_protocol = load_matched_blind_protocol(args.frozen_protocol)
        bindings = dict(frozen_protocol["bindings"])
        expected_paths = {
            "cache_manifest": Path(args.cache_manifest).resolve(),
            "train_cache_manifest": Path(args.train_cache_manifest).resolve(),
            "checkpoint": Path(args.checkpoint).resolve(),
        }
        for key, expected_path in expected_paths.items():
            if not str(expected_path) or expected_path != Path(
                str(bindings[key]["path"])
            ).resolve():
                raise RuntimeError(f"formal protocol {key} path differs")
        if comparison_branches != ["stock", SLAT_ROLLOUT_COMPONENT_FULL]:
            raise ValueError("formal protocol permits only stock,full branches")
        sampling = dict(frozen_protocol["sampling"])
        args.joint_seeds = ",".join(
            str(value) for value in sampling["joint_seeds"]
        )
        args.uids = ",".join(
            str(row["uid"]) for row in frozen_protocol["selected"]
        )
        args.max_objects = 0
        args.object_offset = 0
        args.steps = int(sampling["steps"])
        args.cfg_strength = float(sampling["cfg_strength"])
        args.cfg_interval = ",".join(
            str(value) for value in sampling["cfg_interval"]
        )
        args.rescale_t = float(sampling["rescale_t"])
        args.surface_samples = int(sampling["surface_samples"])
        args.bootstrap_samples = int(sampling["bootstrap_samples"])
        args.amp_dtype = str(sampling["amp_dtype"])
        args.noise_identity_mode = str(sampling["noise_identity_mode"])
        args.support_scale = float(frozen_protocol["candidate"]["support_scale"])
        delta_policy = dict(sampling["slat_delta_policy"])
        args.slat_delta_scale = float(delta_policy["scale"])
        args.slat_delta_rms_ratio_cap = float(delta_policy["rms_ratio_cap"])
        args.slat_guided_delta_policy = str(delta_policy["guided_delta_policy"])
        args.slat_delta_bound_mode = str(delta_policy["bound_mode"])
        args.support_interval_policy = str(
            delta_policy["support_interval_policy"]
        )
        args.slat_residual_combination_policy = str(
            delta_policy.get(
                "residual_combination_policy",
                "joint_total_v1",
            )
        )
        for name in (
            "slat_lora_delta_scale",
            "slat_lora_delta_rms_ratio_cap",
            "slat_support_delta_scale",
            "slat_support_delta_rms_ratio_cap",
        ):
            protocol_name = name.removeprefix("slat_")
            value = delta_policy.get(protocol_name)
            setattr(args, name, None if value is None else float(value))
    joint_seeds = parse_csv(args.joint_seeds, int)
    requested_uids = (
        set(parse_csv(args.uids, str)) if str(args.uids).strip() else set()
    )
    cfg_interval = parse_csv(args.cfg_interval, float)
    if len(cfg_interval) != 2:
        raise ValueError("cfg_interval requires exactly two values")
    if min(int(args.steps), int(args.surface_samples), int(args.bootstrap_samples)) <= 0:
        raise ValueError("steps/sample counts must be positive")
    output_dir = Path(args.output_dir)
    output_existed = output_dir.exists()
    if output_existed and not args.resume:
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = DirectSLatCacheDataset(args.cache_manifest)
    split_audit = None
    if args.train_cache_manifest:
        train_payload = json.loads(
            Path(args.train_cache_manifest).read_text(encoding="utf-8")
        )
        split_audit = assert_disjoint_object_splits(
            train_payload["samples"], dataset.rows
        )
    by_object_uid: dict[str, dict[str, dict[int, int]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for index, row in enumerate(dataset.rows):
        by_object_uid[str(row["object_uid"])][str(row["uid"])][
            int(row["support_seed"])
        ] = index
    selections: list[tuple[str, str, dict[int, int]]] = []
    for object_uid in sorted(by_object_uid):
        eligible = [
            (uid, seed_map)
            for uid, seed_map in sorted(by_object_uid[object_uid].items())
            if all(seed in seed_map for seed in joint_seeds)
            and (not requested_uids or uid in requested_uids)
        ]
        if not eligible and not requested_uids:
            raise RuntimeError(
                f"object={object_uid} has no sequence covering joint seeds={joint_seeds}"
            )
        if not eligible:
            continue
        uid, seed_map = eligible[0]
        selections.append((object_uid, uid, seed_map))
    selected_uids = {uid for _, uid, _ in selections}
    if requested_uids and selected_uids != requested_uids:
        raise RuntimeError(
            "requested Mesh UIDs are missing or lack all joint seeds: "
            f"{sorted(requested_uids - selected_uids)}"
        )
    if int(args.max_objects) > 0:
        if int(args.object_offset) < 0:
            raise ValueError("object_offset must be non-negative")
        if requested_uids and int(args.object_offset) != 0:
            raise ValueError("object_offset cannot be combined with exact --uids")
        start = int(args.object_offset)
        stop = (
            start + int(args.max_objects)
            if int(args.max_objects) > 0
            else None
        )
        selections = selections[start:stop]
    elif int(args.object_offset) != 0:
        raise ValueError("object_offset requires a positive --max_objects")
    if not selections:
        raise ValueError("Mesh selection is empty")
    if frozen_protocol is not None:
        expected_selection = {
            (str(row["object_uid"]), str(row["uid"])): {
                int(seed): int(index)
                for seed, index in dict(row["cache_indices"]).items()
            }
            for row in frozen_protocol["selected"]
        }
        actual_pairs = {
            (str(object_uid), str(uid)) for object_uid, uid, _ in selections
        }
        if actual_pairs != set(expected_selection):
            raise RuntimeError("formal Mesh runtime selection differs from protocol")
        for object_uid, uid, seed_map in selections:
            if {
                int(seed): int(index) for seed, index in seed_map.items()
                if int(seed) in joint_seeds
            } != expected_selection[(str(object_uid), str(uid))]:
                raise RuntimeError(
                    "formal Mesh cache indices differ from frozen protocol"
                )

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    if checkpoint.get("format") != DIRECT_SLAT_FLOW_VERSION:
        raise ValueError(f"unexpected checkpoint format={checkpoint.get('format')!r}")
    saved_args = checkpoint["args"]
    if frozen_protocol is not None and int(checkpoint.get("step", -1)) != int(
        frozen_protocol["candidate"]["checkpoint_step"]
    ):
        raise RuntimeError("formal Mesh checkpoint step differs from protocol")
    if str(saved_args.get("pretrained")) != str(args.pretrained):
        raise RuntimeError("checkpoint pretrained binding differs from Mesh runtime")
    if str(dataset.config.get("pretrained")) != str(args.pretrained):
        raise RuntimeError("cache native condition binding differs from Mesh runtime")
    cache_support_generator = support_generator_identity(dataset.config)
    if checkpoint.get("model_summary", {}).get("support_generator") != cache_support_generator:
        raise RuntimeError("checkpoint and Mesh cache use different frozen SS support")
    support_scale = (
        float(saved_args.get("support_scale", 1.0))
        if args.support_scale is None
        else float(args.support_scale)
    )
    slat_delta_scale = (
        float(saved_args.get("slat_delta_scale", 1.0))
        if args.slat_delta_scale is None
        else float(args.slat_delta_scale)
    )
    requested_ratio_cap = (
        float(saved_args.get("slat_delta_rms_ratio_cap", -1.0))
        if args.slat_delta_rms_ratio_cap is None
        else float(args.slat_delta_rms_ratio_cap)
    )
    if slat_delta_scale < 0:
        raise ValueError("slat_delta_scale must be non-negative")
    slat_delta_rms_ratio_cap = (
        None if requested_ratio_cap < 0 else requested_ratio_cap
    )
    slat_guided_delta_policy = resolve_slat_guided_delta_policy(
        saved_args,
        args.slat_guided_delta_policy,
    )
    slat_delta_bound_mode = resolve_slat_delta_bound_mode(
        saved_args,
        args.slat_delta_bound_mode,
    )
    support_interval_policy = resolve_slat_support_interval_policy(
        saved_args,
        args.support_interval_policy,
    )
    slat_residual_combination_policy = (
        resolve_slat_residual_combination_policy(
            saved_args,
            args.slat_residual_combination_policy,
        )
    )

    def resolve_branch_float(name: str, default: float) -> float:
        saved_value = float(saved_args.get(name, default))
        requested = getattr(args, name)
        resolved = saved_value if requested is None else float(requested)
        if (
            saved_args.get("training_semantics")
            == DIRECT_SLAT_TRAINING_SEMANTICS_V5
            and not math.isclose(
                resolved,
                saved_value,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
        ):
            raise ValueError(
                f"versioned Direct-SLAT V5 {name} is immutable: "
                f"saved={saved_value}, requested={resolved}"
            )
        return resolved

    slat_lora_delta_scale = resolve_branch_float(
        "slat_lora_delta_scale", 1.0
    )
    slat_lora_delta_rms_ratio_cap = resolve_branch_float(
        "slat_lora_delta_rms_ratio_cap", -1.0
    )
    slat_support_delta_scale = resolve_branch_float(
        "slat_support_delta_scale", 1.0
    )
    slat_support_delta_rms_ratio_cap = resolve_branch_float(
        "slat_support_delta_rms_ratio_cap", -1.0
    )
    if (
        comparison_branches != ["stock", SLAT_ROLLOUT_COMPONENT_FULL]
        and slat_guided_delta_policy != SLAT_GUIDED_DELTA_POLICY_V2
    ):
        raise ValueError(
            "LoRA-only/adapter-only rollout diagnostics require post_cfg_v2"
        )
    if (
        saved_args.get("training_semantics")
        in {
            DIRECT_SLAT_TRAINING_SEMANTICS_V3,
            DIRECT_SLAT_TRAINING_SEMANTICS_V4,
            DIRECT_SLAT_TRAINING_SEMANTICS_V5,
        }
    ):
        trained_cfg_strength = float(saved_args.get("train_cfg_strength", 1.0))
        trained_cfg_interval = [
            float(value)
            for value in saved_args.get("train_cfg_interval", ())
        ]
        requested_cfg_interval = [float(value) for value in cfg_interval]
        if not math.isclose(
            float(args.cfg_strength),
            trained_cfg_strength,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ValueError(
                "rollout_aligned_v3 CFG strength is immutable: "
                f"trained={trained_cfg_strength}, requested={args.cfg_strength}"
            )
        if requested_cfg_interval != trained_cfg_interval:
            raise ValueError(
                "rollout_aligned_v3 CFG interval is immutable: "
                f"trained={trained_cfg_interval}, requested={requested_cfg_interval}"
            )
    run_config = {
        "format": "pose_point_depth_mv.direct_slat_mesh_run.v3",
        "cache_manifest": str(Path(args.cache_manifest).resolve()),
        "cache_manifest_sha256": sha256_file(args.cache_manifest),
        "cache_config_hash": dataset.config_hash,
        "train_cache_manifest": (
            str(Path(args.train_cache_manifest).resolve())
            if args.train_cache_manifest
            else ""
        ),
        "train_cache_manifest_sha256": (
            sha256_file(args.train_cache_manifest)
            if args.train_cache_manifest
            else ""
        ),
        "train_eval_split_audit": split_audit,
        "support_generator": cache_support_generator,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "checkpoint_step": int(checkpoint.get("step", -1)),
        "pretrained": args.pretrained,
        "joint_seeds": joint_seeds,
        "comparison_branches": comparison_branches,
        "requested_uids": sorted(requested_uids),
        "object_offset": int(args.object_offset),
        "selected": [
            {"object_uid": object_uid, "uid": uid}
            for object_uid, uid, _ in selections
        ],
        "steps": int(args.steps),
        "cfg_strength": float(args.cfg_strength),
        "cfg_interval": [float(value) for value in cfg_interval],
        "rescale_t": float(args.rescale_t),
        "support_scale": support_scale,
        "slat_delta_policy": {
            "scale": slat_delta_scale,
            "rms_ratio_cap": slat_delta_rms_ratio_cap,
            "per_sparse_batch": True,
            "guided_delta_policy": slat_guided_delta_policy,
            "bound_mode": slat_delta_bound_mode,
            "support_interval_policy": support_interval_policy,
            "residual_combination_policy": (
                slat_residual_combination_policy
            ),
            "lora_delta_scale": slat_lora_delta_scale,
            "lora_delta_rms_ratio_cap": (
                slat_lora_delta_rms_ratio_cap
            ),
            "support_delta_scale": slat_support_delta_scale,
            "support_delta_rms_ratio_cap": (
                slat_support_delta_rms_ratio_cap
            ),
            "positive_cfg_only": (
                slat_guided_delta_policy == SLAT_GUIDED_DELTA_POLICY_LEGACY
            ),
            "post_cfg_full_vs_stock": (
                slat_guided_delta_policy == SLAT_GUIDED_DELTA_POLICY_V2
            ),
        },
        "surface_samples": int(args.surface_samples),
        "amp_dtype": args.amp_dtype,
    }
    if frozen_protocol is not None:
        run_config["frozen_protocol"] = {
            **bind_blind_file(args.frozen_protocol),
            "protocol_sha256": frozen_protocol["protocol_sha256"],
        }
    if args.noise_identity_mode != "object_position_v1":
        run_config["noise_identity_mode"] = args.noise_identity_mode
    run_config["config_hash"] = canonical_json_sha256(run_config)
    run_config_path = output_dir / "run_config.json"
    if run_config_path.is_file():
        existing = json.loads(run_config_path.read_text(encoding="utf-8"))
        if existing != run_config:
            raise RuntimeError("Mesh resume arguments/checkpoint/cache binding changed")
    else:
        if output_existed and any(output_dir.iterdir()):
            raise RuntimeError("refusing to resume an unbound Mesh output directory")
        run_config_path.write_text(json.dumps(run_config, indent=2), encoding="utf-8")
    device = torch.device("cuda")
    torch.cuda.set_device(0)
    built = build_direct_slat_components(
        pretrained=args.pretrained,
        adapter_hidden_dim=int(saved_args["adapter_hidden_dim"]),
        lora_rank=int(saved_args["lora_rank"]),
        lora_alpha=int(saved_args["lora_alpha"]),
        gradient_checkpointing=False,
        device=device,
        retain_pipeline=True,
    )
    sampler, model, sampler_defaults, normalization, _, pipeline = built
    runtime_normalization = {
        key: [float(item) for item in value] for key, value in normalization.items()
    }
    if canonical_json_sha256(runtime_normalization) != dataset.slat_normalization_hash:
        raise RuntimeError("runtime SLAT normalization differs from Mesh cache")
    load_strict_trainable_state(model, checkpoint["model_trainable_state"])
    model.eval()
    slat_params = dict(sampler_defaults)
    slat_params.update(
        {
            "steps": int(args.steps),
            "cfg_strength": float(args.cfg_strength),
            "cfg_interval": [float(value) for value in cfg_interval],
            "rescale_t": float(args.rescale_t),
        }
    )
    if (
        slat_guided_delta_policy == SLAT_GUIDED_DELTA_POLICY_V2
        and float(slat_params.get("guidance_rescale", 0.0)) != 0.0
    ):
        raise RuntimeError(
            "post_cfg_v2 requires guidance_rescale=0 to match native CFG composition"
        )
    use_amp = args.amp_dtype != "none"
    amp_dtype = torch.float16 if args.amp_dtype == "fp16" else torch.bfloat16
    latent_root = output_dir / "slat"
    pair_metadata: list[dict[str, Any]] = []
    for object_position, (object_uid, uid, seed_map) in enumerate(selections):
        for joint_seed in joint_seeds:
            index = seed_map[joint_seed]
            sample = dataset[index]
            pair_id = f"obj_{object_position:04d}_seed_{joint_seed}"
            pair_dir = latent_root / pair_id
            branch_paths = {
                branch: pair_dir / f"{branch}.pt"
                for branch in comparison_branches
            }
            flow_stats_paths = {
                branch: pair_dir / f"{branch}_flow_stats.json"
                for branch in comparison_branches
                if branch != "stock"
            }
            master_seed = (
                int(joint_seed) * 2000003 + object_position * 2017 + 7919
                if args.noise_identity_mode == "object_position_v1"
                else stable_identity_seed(
                    object_uid=object_uid,
                    joint_seed=int(joint_seed),
                    stage="slat",
                )
            )
            if all(path.is_file() for path in branch_paths.values()) and all(
                path.is_file() for path in flow_stats_paths.values()
            ):
                flow_stats = {
                    branch: json.loads(path.read_text(encoding="utf-8"))
                    for branch, path in flow_stats_paths.items()
                }
                pair_row = {
                    "pair_id": pair_id,
                    "object_position": object_position,
                    "object_uid": object_uid,
                    "uid": uid,
                    "joint_seed": joint_seed,
                    "dataset_index": index,
                    "same_initial_noise": True,
                    "flow_stats": flow_stats,
                    "full_flow_stats": flow_stats[
                        SLAT_ROLLOUT_COMPONENT_FULL
                    ],
                }
                if args.noise_identity_mode != "object_position_v1":
                    pair_row.update(
                        {
                            "slat_master_seed": int(master_seed),
                            "noise_identity_mode": args.noise_identity_mode,
                        }
                    )
                pair_metadata.append(pair_row)
                continue
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
            support = (
                sample["corrected_ss"].to(device=device),
                sample["occupancy_logits64"].to(device=device),
                sample["physical_tokens16"].to(device=device),
            )
            outputs: dict[str, Any] = {}
            flow_stats: dict[str, dict[str, Any]] = {}
            for branch in comparison_branches:
                noise = sp.SparseTensor(
                    feats=initial.feats.clone(), coords=initial.coords.clone()
                )
                branch_slat_params = dict(slat_params)
                if branch == "stock":
                    flow = NativeStockSLATFlow(model)
                elif slat_guided_delta_policy == SLAT_GUIDED_DELTA_POLICY_V2:
                    flow = PostCFGSupportSLATRolloutFlow(
                        model,
                        condition["cond"],
                        condition["neg_cond"],
                        support,
                        cfg_strength=float(slat_params["cfg_strength"]),
                        cfg_interval=slat_params["cfg_interval"],
                        support_scale=support_scale,
                        slat_delta_scale=slat_delta_scale,
                        slat_delta_rms_ratio_cap=slat_delta_rms_ratio_cap,
                        slat_delta_bound_mode=slat_delta_bound_mode,
                        slat_residual_combination_policy=(
                            slat_residual_combination_policy
                        ),
                        slat_lora_delta_scale=slat_lora_delta_scale,
                        slat_lora_delta_rms_ratio_cap=(
                            slat_lora_delta_rms_ratio_cap
                        ),
                        slat_support_delta_scale=slat_support_delta_scale,
                        slat_support_delta_rms_ratio_cap=(
                            slat_support_delta_rms_ratio_cap
                        ),
                        support_interval_policy=support_interval_policy,
                        rollout_component=branch,
                    )
                    # The wrapper performs positive/negative CFG internally and
                    # returns an already-guided velocity to the native sampler.
                    branch_slat_params["cfg_strength"] = 1.0
                else:
                    flow = PositiveSupportSLATRolloutFlow(
                        model,
                        condition["cond"],
                        support,
                        support_scale=support_scale,
                        slat_delta_scale=slat_delta_scale,
                        slat_delta_rms_ratio_cap=slat_delta_rms_ratio_cap,
                    )
                with torch.autocast(
                    device_type="cuda", dtype=amp_dtype, enabled=use_amp
                ):
                    sampled = sampler.sample(
                        flow,
                        noise,
                        **condition,
                        **branch_slat_params,
                        verbose=False,
                    ).samples
                if branch != "stock" and flow.positive_calls <= 0:
                    raise RuntimeError(
                        f"{branch} SLAT rollout never used its positive path"
                    )
                if branch != "stock":
                    flow_stats[branch] = flow.stats_summary()
                mean = torch.tensor(normalization["mean"], device=device)[None]
                std = torch.tensor(normalization["std"], device=device)[None]
                outputs[branch] = sampled * std + mean
            for branch in comparison_branches[1:]:
                if not torch.equal(
                    outputs["stock"].coords,
                    outputs[branch].coords,
                ):
                    raise RuntimeError(
                        f"stock/{branch} SLAT rollout coordinates diverged"
                    )
            for branch, output in outputs.items():
                atomic_torch_save(
                    sparse_payload(output),
                    branch_paths[branch],
                )
            if set(flow_stats) != set(comparison_branches[1:]):
                raise RuntimeError(
                    "non-stock SLAT rollout produced incomplete bounded-delta stats"
                )
            for branch, stats in flow_stats.items():
                atomic_json_save(stats, flow_stats_paths[branch])
            pair_row = {
                "pair_id": pair_id,
                "object_position": object_position,
                "object_uid": object_uid,
                "uid": uid,
                "joint_seed": joint_seed,
                "dataset_index": index,
                "coord_count": int(initial.coords.shape[0]),
                "same_initial_noise": True,
                "flow_stats": flow_stats,
                "full_flow_stats": flow_stats[
                    SLAT_ROLLOUT_COMPONENT_FULL
                ],
            }
            if args.noise_identity_mode != "object_position_v1":
                pair_row.update(
                    {
                        "slat_master_seed": int(master_seed),
                        "noise_identity_mode": args.noise_identity_mode,
                    }
                )
            pair_metadata.append(pair_row)
            print(f"[direct_slat_mesh:sample] {pair_id} {uid}", flush=True)
            del sample, master, initial, condition, support, outputs
            torch.cuda.empty_cache()

    model.cpu()
    del model
    gc.collect()
    torch.cuda.empty_cache()
    mesh_decoder = pipeline.models["slat_decoder_mesh"].to(device).eval()
    records: list[dict[str, Any]] = []
    for pair in pair_metadata:
        sample = dataset[int(pair["dataset_index"])]
        target_mesh, target_metadata = load_canonical_gt(sample)
        pair_dir = output_dir / "mesh_pairs" / str(pair["pair_id"])
        pair_dir.mkdir(parents=True, exist_ok=True)
        branch_rows = {}
        for branch in comparison_branches:
            payload = torch.load(
                latent_root / str(pair["pair_id"]) / f"{branch}.pt",
                map_location="cpu",
            )
            decoded = mesh_decoder(sparse_from_payload(payload, device))[0]
            canonical_mesh = decoded.to_trimesh(transform_pose=False)
            view_mesh = decoded.to_trimesh(transform_pose=True)
            structure = mesh_structure_metrics(canonical_mesh)
            if not structure["mesh_success"]:
                raise RuntimeError(f"empty/non-finite decoded mesh: {pair['pair_id']} {branch}")
            branch_dir = pair_dir / branch
            branch_dir.mkdir(exist_ok=True)
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
            branch_rows[branch] = {
                "structure": structure,
                "surface": surface,
            }
        branch_comparisons = {}
        for branch in comparison_branches[1:]:
            branch_comparisons[branch] = {
                "chamfer_l1_improvement": (
                    branch_rows["stock"]["surface"]["chamfer_l1"]
                    - branch_rows[branch]["surface"]["chamfer_l1"]
                ),
                "fscore_0p02_delta": (
                    branch_rows[branch]["surface"]["fscore_0p02"]
                    - branch_rows["stock"]["surface"]["fscore_0p02"]
                ),
                "normal_consistency_delta": (
                    branch_rows[branch]["surface"]["normal_consistency"]
                    - branch_rows["stock"]["surface"]["normal_consistency"]
                ),
                "largest_component_ratio_delta": (
                    branch_rows[branch]["structure"]["largest_component_ratio"]
                    - branch_rows["stock"]["structure"]["largest_component_ratio"]
                ),
            }
        primary = branch_comparisons[SLAT_ROLLOUT_COMPONENT_FULL]
        row = {
            **pair,
            "target": target_metadata,
            "branches": branch_rows,
            "branch_comparisons": branch_comparisons,
            **primary,
        }
        records.append(row)
        print(f"[direct_slat_mesh:decode] {pair['pair_id']}", flush=True)
        del sample, target_mesh, branch_rows
        torch.cuda.empty_cache()
    metric_names = (
        "chamfer_l1_improvement",
        "fscore_0p02_delta",
        "normal_consistency_delta",
        "largest_component_ratio_delta",
    )
    branch_object_rows: dict[str, list[dict[str, Any]]] = {}
    branch_summaries: dict[str, dict[str, Any]] = {}
    for branch_position, branch in enumerate(comparison_branches[1:]):
        object_values: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in records:
            object_values[str(row["object_uid"])].append(
                dict(row["branch_comparisons"][branch])
            )
        rows = [
            {
                "object_uid": object_uid,
                **{
                    name: float(
                        np.mean([float(row[name]) for row in values])
                    )
                    for name in metric_names
                },
            }
            for object_uid, values in sorted(object_values.items())
        ]
        branch_object_rows[branch] = rows
        branch_summaries[branch] = {
            name: summarize(
                [float(row[name]) for row in rows],
                bootstrap_samples=int(args.bootstrap_samples),
                seed=72000 + branch_position * 100 + metric_position,
            )
            for metric_position, name in enumerate(metric_names)
        }
    object_rows = branch_object_rows[SLAT_ROLLOUT_COMPONENT_FULL]
    summary = branch_summaries[SLAT_ROLLOUT_COMPONENT_FULL]
    is_formal = frozen_protocol is not None
    if is_formal:
        expected_objects = int(
            frozen_protocol["selection"]["expected_objects"]
        )
        expected_keys = {
            (str(row["object_uid"]), int(seed))
            for row in frozen_protocol["selected"]
            for seed in frozen_protocol["sampling"]["joint_seeds"]
        }
        actual_keys = {
            (str(row["object_uid"]), int(row["joint_seed"]))
            for row in records
        }
        if (
            len(object_rows) != expected_objects
            or len(records) != len(expected_keys)
            or actual_keys != expected_keys
        ):
            raise RuntimeError(
                "formal Mesh records do not cover every frozen object/seed once"
            )
    report = {
        "format": (
            MATCHED_BLIND_REPORT_FORMAT
            if is_formal
            else "pose_point_depth_mv.direct_slat_mesh_exploratory.v2"
        ),
        "formal": is_formal,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_step": int(checkpoint.get("step", -1)),
        "object_count": len(object_rows),
        "joint_seeds": joint_seeds,
        "comparison_branches": comparison_branches,
        "same_coordinates": "both branches use frozen corrected-SS coords",
        "same_noise": "coordinate-keyed SLAT initial noise is bit-identical",
        "slat_delta_policy": run_config["slat_delta_policy"],
        "train_eval_split_audit": split_audit,
        "summary": summary,
        "object_rows": object_rows,
        "branch_summaries": branch_summaries,
        "branch_object_rows": branch_object_rows,
        "records": records,
        "scope_guard": (
            frozen_protocol["scope_guard"]
            if is_formal
            else (
                "exploratory checkpoint/component diagnosis only; freeze a "
                "fresh blind protocol before any confirmatory Mesh claim"
            )
        ),
    }
    if is_formal:
        by_seed_summary = summarize_seed_directions(records, metric_names)
        decision = formal_decision(
            summary,
            by_seed_summary,
            dict(frozen_protocol["statistics"]["thresholds"]),
        )
        report.update(
            {
                "frozen_protocol": {
                    **bind_blind_file(args.frozen_protocol),
                    "protocol_sha256": frozen_protocol["protocol_sha256"],
                },
                "by_seed_summary": by_seed_summary,
                "decision": decision,
            }
        )
        report["report_sha256"] = blind_canonical_sha256(report)
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    lines = [
        (
            "Direct SLAT formal matched-coordinate same-noise Mesh evaluation"
            if is_formal
            else "Direct SLAT exploratory same-noise Mesh evaluation"
        ),
        "=" * 66,
        f"objects: {len(object_rows)}",
        f"seeds: {joint_seeds}",
    ]
    for name in metric_names:
        row = summary[name]
        lines.append(
            f"{name}: mean={row['mean']:+.8f} median={row['median']:+.8f} "
            f"win={row['positive_rate']:.6f} CI={row['bootstrap_mean_95_ci']}"
        )
    if is_formal:
        lines.extend(
            [
                "",
                f"PRIMARY PASS: {report['decision']['primary_pass']}",
                f"SECONDARY PASS: {report['decision']['secondary_pass']}",
                f"FORMAL PASS: {report['decision']['formal_pass']}",
                (
                    "positive seed fraction: "
                    f"{report['decision']['positive_seed_fraction']:.6f}"
                ),
            ]
        )
    elif len(comparison_branches) > 2:
        lines.extend(["", "Component branch means versus stock:"])
        for branch in comparison_branches[1:]:
            values = branch_summaries[branch]
            lines.append(
                f"{branch}: chamfer={values['chamfer_l1_improvement']['mean']:+.8f} "
                f"fscore={values['fscore_0p02_delta']['mean']:+.8f} "
                f"normal={values['normal_consistency_delta']['mean']:+.8f} "
                "lcr="
                f"{values['largest_component_ratio_delta']['mean']:+.8f}"
            )
    lines.extend(["", report["scope_guard"]])
    (output_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
