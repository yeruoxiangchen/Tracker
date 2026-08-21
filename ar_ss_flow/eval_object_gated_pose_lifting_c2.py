#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
from statistics import mean, median
import sys
from typing import Any

os.environ.setdefault("SPCONV_ALGO", "native")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
os.environ.setdefault("TORCH_EXTENSIONS_DIR", "/tmp/torch_extensions")

import numpy as np
import torch
import torch.nn.functional as F

TRACKER_ROOT = Path(__file__).resolve().parents[1]
RECONVIAGEN_ROOT = TRACKER_ROOT / "ReconViaGen"
VGGT_ROOT = RECONVIAGEN_ROOT / "wheels" / "vggt"
for path in (TRACKER_ROOT, RECONVIAGEN_ROOT, VGGT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ar_ss_flow.local_pose_lifting_flow import (  # noqa: E402
    PoseLiftingCacheDataset,
    volume_from_sample,
)
from ar_ss_flow.object_gate_c2 import (  # noqa: E402
    HYPOTHESES,
    SelfReferenceObjectGateTable,
    apply_object_gate_exact,
    deterministic_permuted_gates,
)
from ar_ss_flow.train_local_pose_lifting_ss_flow import build_model  # noqa: E402
from ar_ss_flow.train_object_gated_pose_lifting_c2 import (  # noqa: E402
    C2_CHECKPOINT_VERSION,
)


BRANCHES = (
    "residual_off",
    "ungated",
    "matched_constant",
    "permuted_correct",
    "correct_gate",
    "pose_cyclic1_gate",
    "pose_cyclic2_gate",
    "pose_reverse_gate",
    "visual_shuffle_gate",
)
NEGATIVE_BRANCHES = (
    "pose_cyclic1_gate",
    "pose_cyclic2_gate",
    "pose_reverse_gate",
    "visual_shuffle_gate",
)


def parse_csv_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_csv_floats(value: str) -> list[float]:
    result = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not result or any(not 0.0 < item < 1.0 for item in result):
        raise ValueError("t_values must lie in (0,1)")
    return result


def summarize(values: list[float]) -> dict[str, float | int]:
    finite = [float(value) for value in values if np.isfinite(value)]
    if not finite:
        return {"count": 0, "mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
    return {
        "count": len(finite),
        "mean": mean(finite),
        "median": median(finite),
        "min": min(finite),
        "max": max(finite),
    }


def positive_rate(values: list[float]) -> float:
    finite = [float(value) for value in values if np.isfinite(value)]
    return float(mean(value > 0.0 for value in finite)) if finite else 0.0


def load_adapter_state(model: torch.nn.Module, state: dict[str, torch.Tensor]) -> None:
    current = model.state_dict()
    unexpected = sorted(set(state).difference(current))
    if unexpected:
        raise KeyError(f"checkpoint has unexpected parameters={unexpected[:5]}")
    missing = sorted(
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and name not in state
    )
    if missing:
        raise KeyError(f"checkpoint missing trainable parameters={missing[:5]}")
    model.load_state_dict(state, strict=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fixed-noise C2 causal audit for self-referenced object-gated SS residual."
    )
    parser.add_argument("--cache_manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--gate_report", required=True)
    parser.add_argument("--gate_samples", required=True)
    parser.add_argument("--gate_calibration", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--pretrained", default="Stable-X/trellis-vggt-v0-2")
    parser.add_argument("--indices", default="16-63")
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--t_values", default="0.1,0.3,0.5,0.7,0.9")
    parser.add_argument("--adapter_hidden_dim", type=int, default=96)
    parser.add_argument("--residual_scale", type=float, default=1.0)
    parser.add_argument("--amp_dtype", choices=("fp16", "bf16", "none"), default="bf16")
    parser.add_argument("--permutation_seed", type=int, default=20260714)
    parser.add_argument("--min_positive_t", type=int, default=4)
    parser.add_argument("--min_object_win_vs_constant", type=float, default=0.55)
    parser.add_argument("--min_object_win_vs_permuted", type=float, default=0.55)
    parser.add_argument("--min_object_win_vs_negative", type=float, default=0.60)
    parser.add_argument("--min_gain_vs_constant", type=float, default=0.0)
    parser.add_argument("--min_gain_vs_permuted", type=float, default=0.0)
    parser.add_argument("--min_gain_vs_negative", type=float, default=0.0)
    parser.add_argument("--fail_on_error", action="store_true")
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    dataset = PoseLiftingCacheDataset(args.cache_manifest, indices=args.indices)
    count = len(dataset) if args.max_samples <= 0 else min(len(dataset), args.max_samples)
    gate_table = SelfReferenceObjectGateTable.load(
        report_path=args.gate_report,
        samples_path=args.gate_samples,
        calibration_path=args.gate_calibration,
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    if checkpoint.get("format") != C2_CHECKPOINT_VERSION:
        raise ValueError(f"unexpected C2 checkpoint format={checkpoint.get('format')}")
    saved_args = checkpoint.get("args", {})
    args.adapter_hidden_dim = int(saved_args.get("adapter_hidden_dim", args.adapter_hidden_dim))
    args.amp_dtype = str(saved_args.get("amp_dtype", args.amp_dtype))
    flow_sampler, model, model_summary = build_model(
        args, device, dataset.visual_feature_dim
    )
    load_adapter_state(model, checkpoint["model_trainable_state"])
    model.eval()

    eligible: list[tuple[int, str]] = []
    skipped: list[dict[str, Any]] = []
    for index in range(count):
        sample = dataset[index]
        uid = str(sample.get("object_uid", sample["uid"]))
        if uid not in gate_table:
            skipped.append({"index": index, "object_uid": uid, "reason": "missing_gate"})
        else:
            eligible.append((index, uid))
    if not eligible:
        raise RuntimeError("no fresh objects have complete gate records")

    correct_gate_by_uid = {uid: gate_table.gate(uid, "correct") for _, uid in eligible}
    matched_constant = float(np.mean(list(correct_gate_by_uid.values())))
    permuted = deterministic_permuted_gates(
        correct_gate_by_uid,
        correct_gate_by_uid,
        seed=int(args.permutation_seed),
    )
    seeds = parse_csv_ints(args.seeds)
    t_values = parse_csv_floats(args.t_values)
    use_amp = args.amp_dtype != "none"
    amp_dtype = torch.float16 if args.amp_dtype == "fp16" else torch.bfloat16
    records: list[dict[str, Any]] = []
    off_max_abs = 0.0

    for position, (sample_index, uid) in enumerate(eligible):
        sample = dataset[sample_index]
        volume, metadata, volume_stats = volume_from_sample(
            sample, device=device, mode="correct"
        )
        target = sample["target"].unsqueeze(0).to(device=device, dtype=torch.float32)
        condition = sample["stock_condition"].to(device=device)
        branch_gates = {
            "residual_off": 0.0,
            "ungated": 1.0,
            "matched_constant": matched_constant,
            "permuted_correct": permuted[uid],
            "correct_gate": gate_table.gate(uid, "correct"),
            "pose_cyclic1_gate": gate_table.gate(uid, "pose_cyclic1"),
            "pose_cyclic2_gate": gate_table.gate(uid, "pose_cyclic2"),
            "pose_reverse_gate": gate_table.gate(uid, "pose_reverse"),
            "visual_shuffle_gate": gate_table.gate(uid, "visual_shuffle"),
        }
        for noise_seed in seeds:
            for t_value in t_values:
                combined_seed = (
                    int(noise_seed) * 1000003
                    + sample_index * 1009
                    + int(round(t_value * 1000))
                )
                torch.manual_seed(combined_seed)
                torch.cuda.manual_seed_all(combined_seed)
                endpoint = torch.randn_like(target)
                x_t, gt_velocity = flow_sampler._get_model_gt(target, t_value, endpoint)
                t_tensor = torch.full(
                    (1,), 1000.0 * t_value, device=device, dtype=torch.float32
                )
                with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
                    stock = model.stock_prediction(x_t, t_tensor, condition)
                    raw_delta, adapter_stats = model.adapter(
                        x_t,
                        stock,
                        t_tensor,
                        volume,
                        metadata,
                        scale=float(args.residual_scale),
                        physical_present=True,
                    )
                    predictions: dict[str, torch.Tensor] = {}
                    applied: dict[str, torch.Tensor] = {}
                    for branch, gate in branch_gates.items():
                        predictions[branch], applied[branch] = apply_object_gate_exact(
                            stock, raw_delta, gate
                        )
                off_max_abs = max(
                    off_max_abs,
                    float((predictions["residual_off"] - stock).abs().max().item()),
                )
                stock_loss = float(F.mse_loss(stock.float(), gt_velocity.float()).item())
                branch_loss = {
                    branch: float(F.mse_loss(value.float(), gt_velocity.float()).item())
                    for branch, value in predictions.items()
                }
                row: dict[str, Any] = {
                    "uid": str(sample["uid"]),
                    "object_uid": uid,
                    "noise_seed": int(noise_seed),
                    "t": float(t_value),
                    "stock_flow_mse": stock_loss,
                    "raw_delta_rms": float(adapter_stats["delta_rms"].float().item()),
                    "supported_voxel_ratio": float(volume_stats["supported_voxel_ratio"]),
                    "gates": branch_gates,
                }
                for branch in BRANCHES:
                    gain = (stock_loss - branch_loss[branch]) / max(stock_loss, 1.0e-8)
                    row[f"{branch}_flow_mse"] = branch_loss[branch]
                    row[f"{branch}_relative_gain_vs_stock"] = gain
                    row[f"{branch}_delta_rms"] = float(
                        applied[branch].float().square().mean().sqrt().item()
                    )
                for control in (
                    "matched_constant",
                    "permuted_correct",
                    "ungated",
                    *NEGATIVE_BRANCHES,
                ):
                    row[f"correct_gate_gain_vs_{control}"] = (
                        branch_loss[control] - branch_loss["correct_gate"]
                    ) / max(stock_loss, 1.0e-8)
                records.append(row)
        print(
            f"[object_gate_c2_eval] {position + 1}/{len(eligible)} uid={uid}",
            flush=True,
        )

    object_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        object_rows[str(row["object_uid"])].append(row)

    branch_summary: dict[str, Any] = {}
    for branch in BRANCHES:
        values = [row[f"{branch}_relative_gain_vs_stock"] for row in records]
        object_values = [
            mean(row[f"{branch}_relative_gain_vs_stock"] for row in rows)
            for rows in object_rows.values()
        ]
        branch_summary[branch] = {
            "record_gain": summarize(values),
            "object_balanced_gain": summarize(object_values),
            "object_positive_rate": positive_rate(object_values),
            "mean_delta_rms": mean(row[f"{branch}_delta_rms"] for row in records),
        }

    comparisons: dict[str, Any] = {}
    controls = (
        "matched_constant",
        "permuted_correct",
        "ungated",
        *NEGATIVE_BRANCHES,
    )
    for control in controls:
        key = f"correct_gate_gain_vs_{control}"
        values = [row[key] for row in records]
        object_values = [mean(row[key] for row in rows) for rows in object_rows.values()]
        comparisons[control] = {
            "record_difference": summarize(values),
            "record_win_rate": positive_rate(values),
            "object_difference": summarize(object_values),
            "object_win_rate": positive_rate(object_values),
        }

    t_summary: dict[str, Any] = {}
    positive_t = 0
    for t_value in t_values:
        values = [
            row["correct_gate_relative_gain_vs_stock"]
            for row in records
            if abs(float(row["t"]) - t_value) < 1.0e-9
        ]
        t_summary[str(t_value)] = summarize(values)
        positive_t += int(float(t_summary[str(t_value)]["mean"]) > 0.0)

    correct_object = branch_summary["correct_gate"]["object_balanced_gain"]
    checks = {
        "residual_off_bit_exact_stock": off_max_abs == 0.0,
        "correct_gate_mean_gain_positive": float(correct_object["mean"]) > 0.0,
        "correct_gate_median_gain_positive": float(correct_object["median"]) > 0.0,
        "correct_gate_positive_t": positive_t >= int(args.min_positive_t),
        "correct_beats_matched_constant": (
            float(comparisons["matched_constant"]["object_difference"]["mean"])
            > float(args.min_gain_vs_constant)
            and float(comparisons["matched_constant"]["object_win_rate"])
            >= float(args.min_object_win_vs_constant)
        ),
        "correct_beats_permuted_gate": (
            float(comparisons["permuted_correct"]["object_difference"]["mean"])
            > float(args.min_gain_vs_permuted)
            and float(comparisons["permuted_correct"]["object_win_rate"])
            >= float(args.min_object_win_vs_permuted)
        ),
        "correct_beats_wrong_and_shuffle_gates": all(
            float(comparisons[branch]["object_difference"]["mean"])
            > float(args.min_gain_vs_negative)
            and float(comparisons[branch]["object_win_rate"])
            >= float(args.min_object_win_vs_negative)
            for branch in NEGATIVE_BRANCHES
        ),
    }
    passed = all(checks.values())
    report = {
        "stage": "C2 self-referenced object-gated local SS residual causal audit",
        "passed": passed,
        "args": vars(args),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_step": int(checkpoint.get("step", 0)),
        "train_gate_summary": checkpoint.get("gate_summary", {}),
        "fresh_gate_summary": gate_table.summary(),
        "fresh_matched_constant_gate": matched_constant,
        "eligible_object_count": len(eligible),
        "record_count": len(records),
        "skipped": skipped,
        "residual_off_max_abs_diff": off_max_abs,
        "positive_t_count": positive_t,
        "t_summary": t_summary,
        "branch_summary": branch_summary,
        "comparisons": comparisons,
        "checks": checks,
        "records": records,
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    markdown = [
        "# C2 Object-gated Local SS Residual Audit",
        "",
        f"- passed: `{passed}`",
        f"- eligible objects / records: `{len(eligible)} / {len(records)}`",
        f"- residual-off max abs diff: `{off_max_abs:.8g}`",
        f"- fresh matched constant gate: `{matched_constant:.6f}`",
        f"- positive t: `{positive_t}/{len(t_values)}`",
        "",
        "| branch | object gain mean | object gain median | object positive rate | mean delta RMS |",
        "|---|---:|---:|---:|---:|",
    ]
    for branch in BRANCHES:
        item = branch_summary[branch]
        markdown.append(
            f"| {branch} | {item['object_balanced_gain']['mean']:+.6g} | "
            f"{item['object_balanced_gain']['median']:+.6g} | "
            f"{item['object_positive_rate']:.4f} | {item['mean_delta_rms']:.6g} |"
        )
    markdown.extend(
        [
            "",
            "| control | correct-minus-control mean | object win rate |",
            "|---|---:|---:|",
        ]
    )
    for control in controls:
        item = comparisons[control]
        markdown.append(
            f"| {control} | {item['object_difference']['mean']:+.6g} | "
            f"{item['object_win_rate']:.4f} |"
        )
    markdown.extend(["", "## Checks", ""])
    markdown.extend(f"- {name}: `{value}`" for name, value in checks.items())
    (output_dir / "report.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")

    print("=" * 100)
    print(report["stage"])
    print("eligible_objects:", len(eligible), "records:", len(records))
    print("fresh_matched_constant_gate:", matched_constant)
    for branch in BRANCHES:
        item = branch_summary[branch]
        print(
            branch,
            "gain=", f"{item['object_balanced_gain']['mean']:+.6f}",
            "positive=", f"{item['object_positive_rate']:.4f}",
            "delta_rms=", f"{item['mean_delta_rms']:.6f}",
        )
    for control in controls:
        item = comparisons[control]
        print(
            "correct_vs", control,
            "gain=", f"{item['object_difference']['mean']:+.6f}",
            "object_win=", f"{item['object_win_rate']:.4f}",
        )
    print("checks:", checks)
    print("passed:", passed)
    print("report:", output_dir / "report.json")
    if args.fail_on_error and not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
