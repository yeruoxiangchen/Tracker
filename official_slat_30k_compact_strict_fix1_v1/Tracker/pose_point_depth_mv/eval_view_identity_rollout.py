#!/usr/bin/env python3
from __future__ import annotations

import argparse
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

from ar_ss_flow.local_pose_lifting_flow import PoseLiftingCacheDataset
from pose_point_depth_mv.eval_local_target_probe import parse_csv_ints, summarize
from pose_point_depth_mv.eval_point_anchor_rollout_v2 import (
    check_positive,
    compare_branches,
    coord_set,
    coords_from_logits,
    evaluate_coords,
    guided_stock_velocity,
    load_models,
    parse_interval,
    rollout_stock_loop,
    timestep_pairs,
)
from pose_point_depth_mv.view_identity_lifting import (
    VIEW_IDENTITY_CHECKPOINT_VERSION,
    VIEW_IDENTITY_CONTROL_NAMES,
    ViewIdentityPoseDepthProbe,
    build_view_identity_evidence,
    load_view_identity_probe_state,
)


ROLLOUT_VERSION = "pose_point_depth_mv.view_identity_rollout.v1"
BRANCH_NAMES = ("stock", "correct", *VIEW_IDENTITY_CONTROL_NAMES)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fixed-noise rollout for view-identity pose/depth lifting."
    )
    parser.add_argument("--cache_manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--pretrained", default="Stable-X/trellis-vggt-v0-2")
    parser.add_argument("--indices", default="16-63")
    parser.add_argument("--split_name", choices=("train16", "fresh48"), required=True)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seeds", default="48,49,50")
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--cfg_strength", type=float, default=5.0)
    parser.add_argument("--cfg_interval", default="0.5,1.0")
    parser.add_argument("--rescale_t", type=float, default=3.0)
    parser.add_argument("--guidance_rescale", type=float, default=0.0)
    parser.add_argument("--physical_scale", type=float, default=1.0)
    parser.add_argument("--bootstrap_samples", type=int, default=10000)
    parser.add_argument("--min_object_win_rate", type=float, default=0.65)
    parser.add_argument("--min_positive_seed_count", type=int, default=2)
    parser.add_argument("--max_global_iou_mean_degradation", type=float, default=0.001)
    parser.add_argument("--max_global_precision_mean_degradation", type=float, default=0.005)
    parser.add_argument("--max_outside_iou_mean_degradation", type=float, default=0.001)
    parser.add_argument("--max_component_count_mean_increase", type=float, default=5.0)
    parser.add_argument(
        "--max_largest_component_ratio_mean_degradation", type=float, default=0.02
    )
    parser.add_argument("--fail_on_decision", action="store_true")
    return parser.parse_args()


def rollout_noise_seed(uid: str, seed: int) -> int:
    digest = hashlib.sha256(
        f"{uid}:{int(seed)}:view-identity-rollout-v1".encode()
    ).digest()
    return int.from_bytes(digest[:8], "little") % (2**63 - 1)


@torch.no_grad()
def rollout_view_branches(
    sampler: Any,
    flow: torch.nn.Module,
    probe: ViewIdentityPoseDepthProbe,
    noise: torch.Tensor,
    condition: torch.Tensor,
    negative_condition: torch.Tensor,
    evidences: list[dict[str, Any]],
    fixed_view_weight: torch.Tensor,
    fixed_support: torch.Tensor,
    *,
    steps: int,
    cfg_strength: float,
    cfg_interval: tuple[float, float],
    rescale_t: float,
    guidance_rescale: float,
    physical_scale: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    branch_count = len(evidences)
    if branch_count != 1 + len(VIEW_IDENTITY_CONTROL_NAMES):
        raise ValueError("view rollout requires correct plus every control")
    sample = noise.expand(branch_count, *noise.shape[1:]).clone()
    cond = condition.expand(branch_count, *condition.shape[1:]).contiguous()
    neg_cond = negative_condition.expand(
        branch_count, *negative_condition.shape[1:]
    ).contiguous()
    delta_rms_values: list[float] = []
    delta_abs_max = 0.0
    neutral_abs_max = 0.0
    for t, t_previous in timestep_pairs(steps, rescale_t):
        positive, _, guided = guided_stock_velocity(
            sampler,
            flow,
            sample,
            t,
            cond,
            neg_cond,
            cfg_strength=cfg_strength,
            cfg_interval=cfg_interval,
            guidance_rescale=guidance_rescale,
        )
        deltas: list[torch.Tensor] = []
        for branch_index, evidence in enumerate(evidences):
            t_tensor = torch.full(
                (1,), 1000.0 * float(t), device=sample.device, dtype=torch.float32
            )
            delta, stats = probe(
                sample[branch_index : branch_index + 1],
                positive[branch_index : branch_index + 1],
                t_tensor,
                evidence,
                scale=float(physical_scale),
                view_weight_override=fixed_view_weight,
                support_gate_override=fixed_support,
            )
            deltas.append(delta)
            delta_rms_values.append(float(stats["delta_rms"].float().item()))
            delta_abs_max = max(
                delta_abs_max, float(stats["delta_abs_max"].float().item())
            )
            neutral_abs_max = max(
                neutral_abs_max, float(stats["neutral_abs_max"].float().item())
            )
        delta_batch = torch.cat(deltas, dim=0)
        sample = sample - (t - t_previous) * (guided + delta_batch)
    return sample, {
        "delta_rms_step_branch_mean": float(np.mean(delta_rms_values)),
        "delta_rms_step_branch_max": float(np.max(delta_rms_values)),
        "delta_abs_max": delta_abs_max,
        "direct_neutral_delta_abs_max": neutral_abs_max,
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# View-identity Pose-guided Lifting Fixed-noise Rollout",
        "",
        f"- Decision: **{'PASS' if report['passed'] else 'FAIL'}**",
        f"- Split: `{report['split_name']}`",
        f"- Objects / records: `{report['object_count']} / {report['record_count']}`",
        f"- Training seed: `{report['training_seed']}`",
        "",
        "## Checks",
        "",
    ]
    lines.extend(
        f"- `{name}`: `{value}`" for name, value in report["decision"]["checks"].items()
    )
    lines.extend(
        [
            "",
            "## Stock Equivalence",
            "",
            "```json",
            json.dumps(report["stock_rollout_equivalence"], indent=2),
            "```",
            "",
            "## Correct vs Stock",
            "",
            "```json",
            json.dumps(report["comparisons"]["correct_vs_stock"], indent=2),
            "```",
            "",
            "## Correct vs Controls",
            "",
            "```json",
            json.dumps(report["comparisons"]["correct_vs_controls"], indent=2),
            "```",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@torch.no_grad()
def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("view-identity rollout requires CUDA")
    seeds = parse_csv_ints(args.seeds)
    cfg_interval = parse_interval(args.cfg_interval)

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    if checkpoint.get("format") != VIEW_IDENTITY_CHECKPOINT_VERSION:
        raise ValueError("unexpected view-identity checkpoint format")
    saved_args = checkpoint.get("args", {})
    if saved_args.get("pretrained") != args.pretrained:
        raise RuntimeError("rollout pretrained configuration mismatch")
    if float(saved_args.get("physical_scale", float("nan"))) != float(
        args.physical_scale
    ):
        raise RuntimeError("rollout physical_scale differs from training")

    dataset = PoseLiftingCacheDataset(args.cache_manifest, indices=args.indices)
    model_summary = checkpoint.get("model_summary", {})
    if str(model_summary.get("cache_config_hash")) != dataset.config_hash:
        raise RuntimeError("rollout checkpoint/cache hash mismatch")
    train_objects = set(str(item) for item in model_summary.get("train_object_uids", ()))
    eval_objects = {str(row.get("object_uid", row["uid"])) for row in dataset.rows}
    overlap = sorted(train_objects & eval_objects)
    if args.split_name == "fresh48" and overlap:
        raise RuntimeError(f"fresh rollout leaks train objects: {overlap}")
    if args.split_name == "train16" and eval_objects != train_objects:
        raise RuntimeError("train16 rollout object set differs from checkpoint")

    sampler, flow, decoder, defaults, flow_schema = load_models(
        args.pretrained, device
    )
    expected_defaults = {
        "steps": int(args.steps),
        "cfg_strength": float(args.cfg_strength),
        "cfg_interval": list(cfg_interval),
        "rescale_t": float(args.rescale_t),
    }
    for key, expected in expected_defaults.items():
        if defaults.get(key) != expected:
            raise RuntimeError(
                f"rollout {key}={expected!r} differs from native default="
                f"{defaults.get(key)!r}"
            )
    if float(args.guidance_rescale) != 0.0:
        raise ValueError("strict native rollout requires guidance_rescale=0")

    probe = ViewIdentityPoseDepthProbe(
        visual_channels=dataset.visual_feature_dim,
        hidden_dim=int(saved_args["hidden_dim"]),
        pair_dim=int(saved_args["pair_dim"]),
        min_views=int(saved_args["min_views"]),
    ).to(device).eval()
    load_view_identity_probe_state(probe, checkpoint["model_trainable_state"])
    count = len(dataset) if args.max_samples <= 0 else min(
        len(dataset), int(args.max_samples)
    )
    rows: list[dict[str, Any]] = []
    stock_audit: dict[str, Any] | None = None
    direct_neutral_max = 0.0

    for index in range(count):
        sample = dataset[index]
        uid = str(sample["uid"])
        object_uid = str(sample.get("object_uid", uid))
        condition = sample["stock_condition"].to(device=device)
        negative_condition = torch.zeros_like(condition)
        correct_evidence = build_view_identity_evidence(
            sample, device=device, mode="correct"
        )
        controls = [
            build_view_identity_evidence(sample, device=device, mode=mode)
            for mode in VIEW_IDENTITY_CONTROL_NAMES
        ]
        evidences = [correct_evidence, *controls]
        fixed_view_weight = correct_evidence["view_weight"].float()
        fixed_support = probe.support_gate(
            correct_evidence, view_weight_override=fixed_view_weight
        )
        active16 = fixed_support.reshape(16, 16, 16).cpu().numpy().astype(np.bool_)
        active64 = np.repeat(np.repeat(np.repeat(active16, 4, 0), 4, 1), 4, 2)
        support_cells = active16
        target_coords = sample["target_coords"].numpy().astype(np.int32)

        for seed in seeds:
            generator = torch.Generator(device=device).manual_seed(
                rollout_noise_seed(uid, seed)
            )
            noise = torch.randn(
                (1, int(flow.in_channels), 16, 16, 16),
                generator=generator,
                device=device,
                dtype=torch.float32,
            )
            native_stock = sampler.sample(
                flow,
                noise.clone(),
                cond=condition,
                neg_cond=negative_condition,
                steps=int(args.steps),
                cfg_strength=float(args.cfg_strength),
                cfg_interval=cfg_interval,
                rescale_t=float(args.rescale_t),
                guidance_rescale=float(args.guidance_rescale),
                verbose=False,
            ).samples
            custom_stock = rollout_stock_loop(
                sampler,
                flow,
                noise,
                condition,
                negative_condition,
                steps=int(args.steps),
                cfg_strength=float(args.cfg_strength),
                cfg_interval=cfg_interval,
                rescale_t=float(args.rescale_t),
                guidance_rescale=float(args.guidance_rescale),
            )
            if stock_audit is None:
                stock_audit = {
                    "latent_max_abs_diff": float(
                        (native_stock.float() - custom_stock.float()).abs().max().item()
                    ),
                    "latent_equal": bool(torch.equal(native_stock, custom_stock)),
                }

            modified, rollout_stats = rollout_view_branches(
                sampler,
                flow,
                probe,
                noise,
                condition,
                negative_condition,
                evidences,
                fixed_view_weight,
                fixed_support,
                steps=int(args.steps),
                cfg_strength=float(args.cfg_strength),
                cfg_interval=cfg_interval,
                rescale_t=float(args.rescale_t),
                guidance_rescale=float(args.guidance_rescale),
                physical_scale=float(args.physical_scale),
            )
            direct_neutral_max = max(
                direct_neutral_max,
                float(rollout_stats["direct_neutral_delta_abs_max"]),
            )
            logits = decoder(
                torch.cat((native_stock, modified), dim=0).to(
                    dtype=next(decoder.parameters()).dtype
                )
            ).float()[:, 0]
            branch_results: dict[str, Any] = {}
            for branch_index, branch in enumerate(BRANCH_NAMES):
                decoded = coords_from_logits(logits[branch_index], len(target_coords))
                branch_results[branch] = {
                    name: evaluate_coords(
                        coords, target_coords, active64, support_cells
                    )
                    for name, coords in decoded.items()
                }
            if stock_audit is not None and "threshold_coord_equal" not in stock_audit:
                audit_logits = decoder(
                    torch.cat((native_stock, custom_stock), dim=0).to(
                        dtype=next(decoder.parameters()).dtype
                    )
                ).float()[:, 0]
                native_coords = coords_from_logits(audit_logits[0], len(target_coords))
                custom_coords = coords_from_logits(audit_logits[1], len(target_coords))
                stock_audit.update(
                    {
                        "threshold_coord_equal": coord_set(native_coords["threshold_0"])
                        == coord_set(custom_coords["threshold_0"]),
                        "topk_coord_equal": coord_set(
                            native_coords["topk_target_oracle_count"]
                        ) == coord_set(custom_coords["topk_target_oracle_count"]),
                    }
                )
            rows.append(
                {
                    "uid": uid,
                    "object_uid": object_uid,
                    "noise_seed": int(seed),
                    "active_ratio": float(active16.mean()),
                    "support_cell_count": int(support_cells.sum()),
                    "rollout_stats": rollout_stats,
                    "branches": branch_results,
                }
            )
            print(
                f"[view_identity_rollout] {index + 1}/{count} seed={seed} uid={uid} "
                f"stock_local={branch_results['stock']['threshold_0']['local']['iou']:.6f} "
                f"correct_local={branch_results['correct']['threshold_0']['local']['iou']:.6f}",
                flush=True,
            )

    if stock_audit is None:
        raise RuntimeError("rollout produced no records")
    correct_stock = compare_branches(
        rows, "correct", "stock", bootstrap_samples=int(args.bootstrap_samples)
    )
    correct_controls = {
        mode: compare_branches(
            rows, "correct", mode, bootstrap_samples=int(args.bootstrap_samples)
        )
        for mode in VIEW_IDENTITY_CONTROL_NAMES
    }
    seed_local = {
        str(seed): summarize(
            [
                float(row["branches"]["correct"]["threshold_0"]["local"]["iou"])
                - float(row["branches"]["stock"]["threshold_0"]["local"]["iou"])
                for row in rows
                if int(row["noise_seed"]) == int(seed)
            ]
        )
        for seed in seeds
    }
    positive_seed_count = sum(float(row["mean"]) > 0.0 for row in seed_local.values())
    checks = {
        "native_stock_rollout_bit_exact": bool(stock_audit["latent_equal"])
        and stock_audit["latent_max_abs_diff"] == 0.0
        and bool(stock_audit.get("threshold_coord_equal"))
        and bool(stock_audit.get("topk_coord_equal")),
        "direct_non_support_delta_exact_zero": direct_neutral_max == 0.0,
        "object_disjoint_if_fresh": args.split_name != "fresh48" or not overlap,
        "correct_local_iou_beats_stock": check_positive(
            correct_stock["local_iou"], args.min_object_win_rate
        ),
        "correct_support_hit_beats_stock": check_positive(
            correct_stock["correct_anchor_cell_hit_rate"], args.min_object_win_rate
        ),
        "correct_local_iou_beats_every_control": all(
            check_positive(row["local_iou"], args.min_object_win_rate)
            for row in correct_controls.values()
        ),
        "positive_noise_seed_count": positive_seed_count
        >= int(args.min_positive_seed_count),
        "global_iou_preserved": float(correct_stock["global_iou"]["object"]["mean"])
        >= -float(args.max_global_iou_mean_degradation),
        "global_precision_preserved": float(
            correct_stock["global_precision"]["object"]["mean"]
        ) >= -float(args.max_global_precision_mean_degradation),
        "outside_iou_preserved": float(correct_stock["outside_iou"]["object"]["mean"])
        >= -float(args.max_outside_iou_mean_degradation),
        "component_count_bounded": float(
            correct_stock["component_count"]["object"]["mean"]
        ) <= float(args.max_component_count_mean_increase),
        "largest_component_ratio_preserved": float(
            correct_stock["largest_component_ratio"]["object"]["mean"]
        ) >= -float(args.max_largest_component_ratio_mean_degradation),
    }
    report = {
        "format": ROLLOUT_VERSION,
        "stage": "View-identity pose-guided lifting fixed-noise SS rollout",
        "passed": all(checks.values()),
        "args": vars(args),
        "split_name": args.split_name,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_step": int(checkpoint.get("step", -1)),
        "training_seed": int(saved_args.get("seed", -1)),
        "cache_config_hash": dataset.config_hash,
        "probe": probe.metadata(),
        "flow": flow_schema,
        "native_sampler_defaults": defaults,
        "integration": {
            "mode": "post_cfg_delta_from_positive_stock",
            "view_identity_preserved_until_same_voxel_attention": True,
            "fixed_correct_view_weight": True,
            "fixed_correct_support_gate": True,
            "flow_lora": False,
        },
        "sample_count": count,
        "object_count": len({row["object_uid"] for row in rows}),
        "record_count": len(rows),
        "noise_seeds": seeds,
        "eval_object_uid_hash": hashlib.sha256(
            "\n".join(sorted(eval_objects)).encode()
        ).hexdigest(),
        "stock_rollout_equivalence": stock_audit,
        "direct_neutral_delta_abs_max": direct_neutral_max,
        "comparisons": {
            "correct_vs_stock": correct_stock,
            "correct_vs_controls": correct_controls,
        },
        "per_noise_seed_correct_local_iou_delta": seed_local,
        "decision": {
            "checks": checks,
            "positive_noise_seed_count": positive_seed_count,
            "required_positive_noise_seed_count": int(args.min_positive_seed_count),
        },
        "records": rows,
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    write_markdown(report, output_dir / "report.md")
    print(json.dumps({
        "passed": report["passed"],
        "checks": checks,
        "correct_vs_stock_local_iou": correct_stock["local_iou"],
        "positive_noise_seed_count": positive_seed_count,
    }, indent=2), flush=True)
    if args.fail_on_decision and not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
