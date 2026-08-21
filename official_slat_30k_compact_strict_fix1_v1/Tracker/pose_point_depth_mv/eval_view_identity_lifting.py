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

import torch
import torch.nn.functional as F

from ar_ss_flow.local_pose_lifting_flow import PoseLiftingCacheDataset
from pose_point_depth_mv.eval_local_target_probe import (
    deterministic_noise_seed,
    object_balanced,
    parse_csv_floats,
    parse_csv_ints,
    summarize,
)
from pose_point_depth_mv.train_local_target_probe import (
    build_frozen_stock_flow,
    masked_mse,
)
from pose_point_depth_mv.view_identity_lifting import (
    VIEW_IDENTITY_CHECKPOINT_VERSION,
    VIEW_IDENTITY_CONTROL_NAMES,
    ViewIdentityPoseDepthProbe,
    build_view_identity_evidence,
    load_view_identity_probe_state,
    make_null_view_identity_evidence,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Strict teacher-forced evaluation of view-identity lifting."
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
    parser.add_argument("--t_values", default="0.1,0.3,0.5,0.7,0.9")
    parser.add_argument("--physical_scale", type=float, default=1.0)
    parser.add_argument("--bootstrap_samples", type=int, default=10000)
    parser.add_argument("--min_object_win_rate", type=float, default=0.65)
    parser.add_argument("--min_positive_t_count", type=int, default=4)
    parser.add_argument("--fail_on_decision", action="store_true")
    return parser.parse_args()


def support_iou(left: torch.Tensor, right: torch.Tensor) -> float:
    left_mask = left.gt(1.0e-6)
    right_mask = right.gt(1.0e-6)
    intersection = (left_mask & right_mask).sum().float()
    union = (left_mask | right_mask).sum().float()
    return float((intersection / union.clamp_min(1.0)).item())


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# View-identity Pose-guided SS Probe Evaluation",
        "",
        f"- Split: `{report['split_name']}`",
        f"- Decision: **{'PASS' if report['passed'] else 'FAIL'}**",
        f"- Checkpoint step: `{report['checkpoint_step']}`",
        f"- Objects: `{report['object_count']}`",
        f"- Records: `{report['record_count']}`",
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
            "## Correct vs Stock",
            "",
            "```json",
            json.dumps(report["correct_vs_stock"], indent=2),
            "```",
            "",
            "## Fixed-mask Controls",
            "",
            "```json",
            json.dumps(report["controls"], indent=2),
            "```",
            "",
            "## Per-t",
            "",
            "```json",
            json.dumps(report["per_t"], indent=2),
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
        raise ValueError("view-identity Flow evaluation requires CUDA")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    if checkpoint.get("format") != VIEW_IDENTITY_CHECKPOINT_VERSION:
        raise ValueError("unexpected view-identity checkpoint format")
    saved_args = checkpoint.get("args", {})
    if saved_args.get("pretrained") != args.pretrained:
        raise RuntimeError("view-identity pretrained configuration mismatch")
    if float(saved_args.get("physical_scale", float("nan"))) != float(
        args.physical_scale
    ):
        raise RuntimeError("view-identity physical_scale differs from training")

    dataset = PoseLiftingCacheDataset(args.cache_manifest, indices=args.indices)
    model_summary = checkpoint.get("model_summary", {})
    if str(model_summary.get("cache_config_hash")) != dataset.config_hash:
        raise RuntimeError("view-identity checkpoint/cache hash mismatch")
    train_objects = set(str(value) for value in model_summary.get("train_object_uids", ()))
    eval_objects = {str(row.get("object_uid", row["uid"])) for row in dataset.rows}
    overlap = sorted(train_objects & eval_objects)
    if args.split_name == "fresh48" and overlap:
        raise RuntimeError(f"fresh view-identity evaluation leaks train objects: {overlap}")
    if args.split_name == "train16" and eval_objects != train_objects:
        raise RuntimeError("train16 object set differs from checkpoint")

    count = len(dataset) if args.max_samples <= 0 else min(
        len(dataset), int(args.max_samples)
    )
    sampler, stock_flow, flow_schema = build_frozen_stock_flow(
        args.pretrained, device
    )
    probe = ViewIdentityPoseDepthProbe(
        visual_channels=dataset.visual_feature_dim,
        hidden_dim=int(saved_args["hidden_dim"]),
        pair_dim=int(saved_args["pair_dim"]),
        min_views=int(saved_args["min_views"]),
    ).to(device).eval()
    load_view_identity_probe_state(probe, checkpoint["model_trainable_state"])
    amp_name = str(saved_args.get("amp_dtype", "bf16"))
    use_amp = amp_name != "none"
    amp_dtype = torch.float16 if amp_name == "fp16" else torch.bfloat16
    seeds = parse_csv_ints(args.seeds)
    t_values = parse_csv_floats(args.t_values)
    records: list[dict[str, Any]] = []
    zero_audit = {
        "physical_off_max_abs": 0.0,
        "null_evidence_max_abs": 0.0,
        "neutral_max_abs": 0.0,
    }
    support_reports: list[dict[str, Any]] = []

    for position in range(count):
        sample = dataset[position]
        uid = str(sample["uid"])
        object_uid = str(sample.get("object_uid", uid))
        target = sample["target"].unsqueeze(0).to(device=device, dtype=torch.float32)
        condition = sample["stock_condition"].to(device=device)
        correct_evidence = build_view_identity_evidence(
            sample, device=device, mode="correct"
        )
        controls = {
            mode: build_view_identity_evidence(sample, device=device, mode=mode)
            for mode in VIEW_IDENTITY_CONTROL_NAMES
        }
        fixed_weight = correct_evidence["view_weight"].float()
        fixed_support = probe.support_gate(
            correct_evidence, view_weight_override=fixed_weight
        )
        active_mask = fixed_support.reshape(1, 1, 16, 16, 16)
        if not bool(active_mask.any().item()):
            raise RuntimeError(f"uid={uid} has empty multi-view support")
        support_reports.append(
            {
                "uid": uid,
                "correct_support_ratio": float(fixed_support.mean().item()),
                "controls": {
                    mode: {
                        "native_support_ratio": float(
                            probe.support_gate(evidence).mean().item()
                        ),
                        "native_view_weight_iou": support_iou(
                            fixed_weight, evidence["view_weight"]
                        ),
                    }
                    for mode, evidence in controls.items()
                },
            }
        )

        for noise_seed in seeds:
            for t_value in t_values:
                generator = torch.Generator(device=device).manual_seed(
                    deterministic_noise_seed(uid, noise_seed, t_value)
                )
                noise = torch.randn(
                    target.shape,
                    generator=generator,
                    device=device,
                    dtype=target.dtype,
                )
                x_t, gt_velocity = sampler._get_model_gt(target, t_value, noise)
                t_tensor = torch.full(
                    (1,), 1000.0 * t_value, device=device, dtype=torch.float32
                )
                with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
                    stock = stock_flow(x_t, t_tensor, condition)
                    correct_delta, correct_stats = probe(
                        x_t,
                        stock,
                        t_tensor,
                        correct_evidence,
                        scale=float(args.physical_scale),
                        view_weight_override=fixed_weight,
                        support_gate_override=fixed_support,
                    )
                    control_outputs = {
                        mode: probe(
                            x_t,
                            stock,
                            t_tensor,
                            evidence,
                            scale=float(args.physical_scale),
                            view_weight_override=fixed_weight,
                            support_gate_override=fixed_support,
                        )
                        for mode, evidence in controls.items()
                    }
                    off_delta, _ = probe(
                        x_t,
                        stock,
                        t_tensor,
                        correct_evidence,
                        physical_present=False,
                    )
                    null_delta, _ = probe(
                        x_t,
                        stock,
                        t_tensor,
                        make_null_view_identity_evidence(correct_evidence),
                    )
                zero_audit["physical_off_max_abs"] = max(
                    zero_audit["physical_off_max_abs"],
                    float(off_delta.float().abs().max().item()),
                )
                zero_audit["null_evidence_max_abs"] = max(
                    zero_audit["null_evidence_max_abs"],
                    float(null_delta.float().abs().max().item()),
                )
                zero_audit["neutral_max_abs"] = max(
                    zero_audit["neutral_max_abs"],
                    float(correct_stats["neutral_abs_max"].float().item()),
                )
                stock_loss = F.mse_loss(
                    stock.float(), gt_velocity.float()
                ).clamp_min(1.0e-6)
                correct_loss = F.mse_loss(
                    (stock + correct_delta).float(), gt_velocity.float()
                )
                target_residual = gt_velocity.float() - stock.float()
                target_energy = masked_mse(
                    target_residual,
                    torch.zeros_like(target_residual),
                    active_mask,
                ).clamp_min(1.0e-6)
                correct_local_loss = masked_mse(
                    correct_delta, target_residual, active_mask
                ) / target_energy
                row: dict[str, Any] = {
                    "uid": uid,
                    "object_uid": object_uid,
                    "noise_seed": int(noise_seed),
                    "t": float(t_value),
                    "stock_flow_loss": float(stock_loss.item()),
                    "correct_flow_loss": float(correct_loss.item()),
                    "correct_relative_gain": float(
                        ((stock_loss - correct_loss) / stock_loss).item()
                    ),
                    "correct_local_target_gain": float(
                        (1.0 - correct_local_loss).item()
                    ),
                    "correct_delta_rms": float(correct_stats["delta_rms"].item()),
                    "correct_pair_consensus": float(
                        correct_stats["pair_consensus"].item()
                    ),
                    "active_ratio": float(fixed_support.mean().item()),
                }
                for mode, (delta, stats) in control_outputs.items():
                    flow_loss = F.mse_loss(
                        (stock + delta).float(), gt_velocity.float()
                    )
                    local_loss = masked_mse(
                        delta, target_residual, active_mask
                    ) / target_energy
                    row[f"{mode}_relative_gain"] = float(
                        ((stock_loss - flow_loss) / stock_loss).item()
                    )
                    row[f"correct_vs_{mode}"] = float(
                        ((flow_loss - correct_loss) / stock_loss).item()
                    )
                    row[f"correct_local_vs_{mode}"] = float(
                        (local_loss - correct_local_loss).item()
                    )
                    row[f"{mode}_delta_rms"] = float(stats["delta_rms"].item())
                    row[f"{mode}_pair_consensus"] = float(
                        stats["pair_consensus"].item()
                    )
                records.append(row)
        print(f"[view_identity_eval] {position + 1}/{count} uid={uid}", flush=True)

    correct_vs_stock = object_balanced(
        records,
        "correct_relative_gain",
        bootstrap_samples=args.bootstrap_samples,
    )
    correct_local = object_balanced(
        records,
        "correct_local_target_gain",
        bootstrap_samples=args.bootstrap_samples,
    )
    controls_report = {
        mode: {
            "relative_gain_vs_stock": object_balanced(
                records,
                f"{mode}_relative_gain",
                bootstrap_samples=args.bootstrap_samples,
            ),
            "correct_advantage_flow": object_balanced(
                records,
                f"correct_vs_{mode}",
                bootstrap_samples=args.bootstrap_samples,
            ),
            "correct_advantage_local": object_balanced(
                records,
                f"correct_local_vs_{mode}",
                bootstrap_samples=args.bootstrap_samples,
            ),
            "delta_rms": object_balanced(
                records,
                f"{mode}_delta_rms",
                bootstrap_samples=args.bootstrap_samples,
            ),
            "pair_consensus": object_balanced(
                records,
                f"{mode}_pair_consensus",
                bootstrap_samples=args.bootstrap_samples,
            ),
        }
        for mode in VIEW_IDENTITY_CONTROL_NAMES
    }
    per_t = {
        f"{t_value:.3f}": {
            "correct_relative_gain": summarize(
                [
                    float(row["correct_relative_gain"])
                    for row in records
                    if row["t"] == float(t_value)
                ]
            ),
            "correct_vs_controls": {
                mode: summarize(
                    [
                        float(row[f"correct_vs_{mode}"])
                        for row in records
                        if row["t"] == float(t_value)
                    ]
                )
                for mode in VIEW_IDENTITY_CONTROL_NAMES
            },
        }
        for t_value in t_values
    }
    positive_t_count = sum(
        float(row["correct_relative_gain"]["mean"]) > 0.0
        for row in per_t.values()
    )
    checks = {
        "physical_off_exact_stock": zero_audit["physical_off_max_abs"] == 0.0,
        "null_evidence_exact_stock": zero_audit["null_evidence_max_abs"] == 0.0,
        "neutral_exact_stock": zero_audit["neutral_max_abs"] == 0.0,
        "object_disjoint_if_fresh": args.split_name != "fresh48" or not overlap,
        "correct_mean_positive": float(correct_vs_stock["object"]["mean"]) > 0.0,
        "correct_median_positive": float(correct_vs_stock["object"]["median"]) > 0.0,
        "correct_object_win_rate": float(correct_vs_stock["object_win_rate"])
        >= float(args.min_object_win_rate),
        "correct_bootstrap_ci_positive": float(
            correct_vs_stock["object_bootstrap_95_ci"][0]
        ) > 0.0,
        "correct_beats_every_control_mean": all(
            float(controls_report[mode]["correct_advantage_flow"]["object"]["mean"])
            > 0.0
            for mode in VIEW_IDENTITY_CONTROL_NAMES
        ),
        "correct_beats_every_control_median": all(
            float(controls_report[mode]["correct_advantage_flow"]["object"]["median"])
            > 0.0
            for mode in VIEW_IDENTITY_CONTROL_NAMES
        ),
        "control_object_win_rates": all(
            float(controls_report[mode]["correct_advantage_flow"]["object_win_rate"])
            >= float(args.min_object_win_rate)
            for mode in VIEW_IDENTITY_CONTROL_NAMES
        ),
        "control_bootstrap_cis_positive": all(
            float(
                controls_report[mode]["correct_advantage_flow"]
                ["object_bootstrap_95_ci"][0]
            ) > 0.0
            for mode in VIEW_IDENTITY_CONTROL_NAMES
        ),
        "positive_t_count": positive_t_count >= int(args.min_positive_t_count),
    }
    report = {
        "stage": "View-identity pose-guided lifting strict teacher-forced evaluation",
        "passed": all(checks.values()),
        "args": vars(args),
        "split_name": args.split_name,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_step": int(checkpoint.get("step", -1)),
        "training_seed": int(saved_args.get("seed", -1)),
        "training_config": {
            key: saved_args.get(key)
            for key in (
                "hidden_dim",
                "pair_dim",
                "min_views",
                "physical_scale",
                "local_target_weight",
                "flow_weight",
                "corrupt_zero_weight",
                "gain_weight",
                "gain_margin",
                "rank_weight",
                "rank_margin",
                "delta_norm_weight",
            )
        },
        "cache_config_hash": dataset.config_hash,
        "probe": probe.metadata(),
        "flow": flow_schema,
        "sample_count": count,
        "object_count": len({row["object_uid"] for row in records}),
        "record_count": len(records),
        "eval_object_uid_hash": hashlib.sha256(
            "\n".join(sorted(eval_objects)).encode("utf-8")
        ).hexdigest(),
        "noise_seeds": seeds,
        "t_values": t_values,
        "physical_scale": float(args.physical_scale),
        "mask_protocol": "fixed_correct_view_weight_and_support_gate",
        "stock_preservation": zero_audit,
        "support_diagnostics": support_reports,
        "correct_vs_stock": correct_vs_stock,
        "correct_local_vs_zero": correct_local,
        "controls": controls_report,
        "per_t": per_t,
        "decision": {
            "checks": checks,
            "positive_t_count": positive_t_count,
            "required_positive_t_count": int(args.min_positive_t_count),
        },
        "records": records,
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    write_markdown(report, output_dir / "report.md")
    print(json.dumps({
        "passed": report["passed"],
        "checks": checks,
        "correct_vs_stock": correct_vs_stock,
        "positive_t_count": positive_t_count,
    }, indent=2), flush=True)
    if args.fail_on_decision and not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
