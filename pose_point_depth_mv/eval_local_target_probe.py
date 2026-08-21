#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import random
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
for path in (
    TRACKER_ROOT,
    TRACKER_ROOT / "ReconViaGen",
    TRACKER_ROOT / "ReconViaGen" / "wheels" / "vggt",
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from pose_point_depth_mv.local_target_probe import (  # noqa: E402
    PPDLocalTargetProbe,
    PPDProbeEvidenceDataset,
    PROBE_CORRUPTIONS,
    load_probe_state,
    make_null_evidence,
)
from pose_point_depth_mv.train_local_target_probe import (  # noqa: E402
    CHECKPOINT_VERSION,
    build_frozen_stock_flow,
    masked_mse,
)


def parse_csv_ints(value: str) -> list[int]:
    output = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not output:
        raise ValueError("at least one noise seed is required")
    return output


def parse_csv_floats(value: str) -> list[float]:
    output = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not output or any(not 0.0 < item < 1.0 for item in output):
        raise ValueError("t values must lie in (0,1)")
    return output


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


def bootstrap_ci(
    values: list[float], *, seed: int, samples: int
) -> list[float]:
    if not values:
        return [0.0, 0.0]
    rng = random.Random(int(seed))
    draws = [
        mean(values[rng.randrange(len(values))] for _ in values)
        for _ in range(int(samples))
    ]
    draws.sort()
    return [draws[int(0.025 * len(draws))], draws[int(0.975 * len(draws))]]


def object_balanced(
    records: list[dict[str, Any]], field: str, *, bootstrap_samples: int
) -> dict[str, Any]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in records:
        grouped[str(row["object_uid"])].append(float(row[field]))
    values = [mean(items) for items in grouped.values()]
    return {
        "record": summarize([float(row[field]) for row in records]),
        "record_win_rate": positive_rate([float(row[field]) for row in records]),
        "object": summarize(values),
        "object_win_rate": positive_rate(values),
        "object_bootstrap_95_ci": bootstrap_ci(
            values, seed=20260715, samples=int(bootstrap_samples)
        ),
    }


def deterministic_noise_seed(uid: str, seed: int, t_value: float) -> int:
    text = f"{uid}:{int(seed)}:{float(t_value):.6f}"
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**63 - 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Paired teacher-forced PPD-3A evaluation of a local target "
            "residual learnability probe."
        )
    )
    parser.add_argument("--probe_cache_manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--pretrained", default="Stable-X/trellis-vggt-v0-2")
    parser.add_argument("--indices", default="16-63")
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seeds", default="48,49,50")
    parser.add_argument("--t_values", default="0.1,0.3,0.5,0.7,0.9")
    parser.add_argument("--physical_scale", type=float, default=1.0)
    parser.add_argument("--bootstrap_samples", type=int, default=10000)
    parser.add_argument("--min_correct_object_win_rate", type=float, default=0.65)
    parser.add_argument("--min_corruption_object_win_rate", type=float, default=0.60)
    parser.add_argument("--min_positive_t_count", type=int, default=4)
    parser.add_argument("--max_neutral_abs", type=float, default=0.0)
    parser.add_argument("--fail_on_decision", action="store_true")
    return parser.parse_args()


def write_markdown(report: dict[str, Any], path: Path) -> None:
    decision = report["decision"]
    lines = [
        "# PPD-3A Local Target Probe Evaluation",
        "",
        f"- Decision: **{'PASS' if report['passed'] else 'FAIL'}**",
        f"- Objects: `{report['object_count']}`",
        f"- Records: `{report['record_count']}`",
        f"- Checkpoint step: `{report['checkpoint_step']}`",
        f"- Physical scale: `{report['physical_scale']}`",
        "",
        "## Checks",
        "",
    ]
    lines.extend(f"- `{name}`: `{value}`" for name, value in decision["checks"].items())
    lines.extend(
        [
            "",
            "## Correct vs Stock",
            "",
            "```json",
            json.dumps(report["comparisons"]["correct_vs_stock"], indent=2),
            "```",
            "",
            "## Correct vs Corruptions",
            "",
            "```json",
            json.dumps(report["comparisons"]["correct_vs_corruptions"], indent=2),
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
        raise ValueError("PPD-3A Flow evaluation requires CUDA")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    if checkpoint.get("format") != CHECKPOINT_VERSION:
        raise ValueError(f"unexpected checkpoint format={checkpoint.get('format')!r}")
    saved_args = checkpoint.get("args", {})
    mismatch = {
        key: {"checkpoint": saved_args.get(key), "current": getattr(args, key)}
        for key in ("pretrained",)
        if saved_args.get(key) != getattr(args, key)
    }
    if mismatch:
        raise RuntimeError(f"checkpoint configuration mismatch: {mismatch}")
    rank = int(saved_args["rank"])
    amp_dtype_name = str(saved_args.get("amp_dtype", "bf16"))
    use_amp = amp_dtype_name != "none"
    amp_dtype = torch.float16 if amp_dtype_name == "fp16" else torch.bfloat16
    dataset = PPDProbeEvidenceDataset(
        args.probe_cache_manifest, indices=args.indices, eligible_only=True
    )
    trained_cache_hash = str(
        checkpoint.get("model_summary", {}).get("evidence_cache_config_hash", "")
    )
    if trained_cache_hash != dataset.config_hash:
        raise RuntimeError(
            "evaluation evidence cache does not match training cache: "
            f"{dataset.config_hash} != {trained_cache_hash}"
        )
    count = len(dataset) if args.max_samples <= 0 else min(
        len(dataset), int(args.max_samples)
    )
    sampler, stock_flow, flow_schema = build_frozen_stock_flow(
        args.pretrained, device
    )
    probe = PPDLocalTargetProbe(rank=rank).to(device).eval()
    load_probe_state(probe, checkpoint["model_trainable_state"])
    seeds = parse_csv_ints(args.seeds)
    t_values = parse_csv_floats(args.t_values)
    records: list[dict[str, Any]] = []
    zero_off_max_abs = 0.0
    null_max_abs = 0.0
    neutral_max_abs = 0.0

    for position in range(count):
        sample = dataset[position]
        uid = str(sample["uid"])
        object_uid = str(sample.get("object_uid", uid))
        target = sample["target"].unsqueeze(0).to(device=device, dtype=torch.float32)
        condition = sample["stock_condition"].to(device=device)
        correct_evidence = sample["ppd_correct_features"].unsqueeze(0).to(
            device=device, dtype=torch.float32
        )
        corrupt_evidence = {
            name: value.unsqueeze(0).to(device=device, dtype=torch.float32)
            for name, value in sample["ppd_corrupt_features"].items()
        }
        correct_mask = probe.active_mask(correct_evidence).to(device=device)
        for noise_seed in seeds:
            for t_value in t_values:
                generator = torch.Generator(device=device).manual_seed(
                    deterministic_noise_seed(uid, noise_seed, t_value)
                )
                endpoint = torch.randn(
                    target.shape,
                    generator=generator,
                    device=device,
                    dtype=target.dtype,
                )
                x_t, gt_velocity = sampler._get_model_gt(
                    target, t_value, endpoint
                )
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
                    )
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
                        make_null_evidence(correct_evidence),
                    )
                    corruption_deltas = {
                        name: probe(
                            x_t,
                            stock,
                            t_tensor,
                            evidence,
                            scale=float(args.physical_scale),
                        )[0]
                        for name, evidence in corrupt_evidence.items()
                    }
                zero_off_max_abs = max(
                    zero_off_max_abs, float(off_delta.float().abs().max().item())
                )
                null_max_abs = max(
                    null_max_abs, float(null_delta.float().abs().max().item())
                )
                neutral_max_abs = max(
                    neutral_max_abs,
                    float(correct_stats["neutral_abs_max"].float().item()),
                )
                target_residual = gt_velocity.float() - stock.float()
                target_energy = masked_mse(
                    target_residual,
                    torch.zeros_like(target_residual),
                    correct_mask,
                ).clamp_min(1.0e-6)
                stock_loss = F.mse_loss(stock.float(), gt_velocity.float())
                correct_prediction = stock + correct_delta
                correct_loss = F.mse_loss(
                    correct_prediction.float(), gt_velocity.float()
                )
                correct_gain = (stock_loss - correct_loss) / stock_loss.clamp_min(1.0e-6)
                local_correct_loss = masked_mse(
                    correct_delta, target_residual, correct_mask
                ) / target_energy
                row: dict[str, Any] = {
                    "source_index": int(sample["ppd_source_index"]),
                    "uid": uid,
                    "object_uid": object_uid,
                    "noise_seed": int(noise_seed),
                    "t": float(t_value),
                    "stock_flow_loss": float(stock_loss.item()),
                    "correct_flow_loss": float(correct_loss.item()),
                    "correct_relative_gain": float(correct_gain.item()),
                    "correct_local_target_loss": float(local_correct_loss.item()),
                    "correct_local_target_gain": float(1.0 - local_correct_loss.item()),
                    "correct_delta_rms": float(
                        correct_delta.float().square().mean().sqrt().item()
                    ),
                    "correct_active_ratio": float(correct_mask.float().mean().item()),
                }
                for name, delta in corruption_deltas.items():
                    prediction = stock + delta
                    flow_loss = F.mse_loss(prediction.float(), gt_velocity.float())
                    local_loss = masked_mse(delta, target_residual, correct_mask) / target_energy
                    row[f"{name}_flow_loss"] = float(flow_loss.item())
                    row[f"correct_vs_{name}"] = float(
                        ((flow_loss - correct_loss) / stock_loss.clamp_min(1.0e-6)).item()
                    )
                    row[f"correct_local_vs_{name}"] = float(
                        (local_loss - local_correct_loss).item()
                    )
                records.append(row)
        print(
            f"[ppd3a_eval] {position + 1}/{count} uid={uid}", flush=True
        )

    correct_vs_stock = object_balanced(
        records, "correct_relative_gain", bootstrap_samples=args.bootstrap_samples
    )
    local_correct = object_balanced(
        records,
        "correct_local_target_gain",
        bootstrap_samples=args.bootstrap_samples,
    )
    corrupt_comparisons = {
        name: {
            "flow": object_balanced(
                records,
                f"correct_vs_{name}",
                bootstrap_samples=args.bootstrap_samples,
            ),
            "local_target": object_balanced(
                records,
                f"correct_local_vs_{name}",
                bootstrap_samples=args.bootstrap_samples,
            ),
        }
        for name in PROBE_CORRUPTIONS
    }
    per_t: dict[str, Any] = {}
    for t_value in t_values:
        subset = [row for row in records if row["t"] == float(t_value)]
        per_t[f"{t_value:.3f}"] = {
            "correct_relative_gain": summarize(
                [float(row["correct_relative_gain"]) for row in subset]
            ),
            "correct_vs_corruptions": {
                name: summarize(
                    [float(row[f"correct_vs_{name}"]) for row in subset]
                )
                for name in PROBE_CORRUPTIONS
            },
        }
    positive_t_count = sum(
        float(row["correct_relative_gain"]["mean"]) > 0.0
        for row in per_t.values()
    )
    checks = {
        "physical_off_bit_exact_stock": zero_off_max_abs == 0.0,
        "null_evidence_bit_exact_stock": null_max_abs == 0.0,
        "neutral_exact_zero": neutral_max_abs <= float(args.max_neutral_abs),
        "correct_mean_gain_positive": float(correct_vs_stock["object"]["mean"]) > 0.0,
        "correct_median_gain_positive": float(correct_vs_stock["object"]["median"]) > 0.0,
        "correct_object_win_rate": float(correct_vs_stock["object_win_rate"])
        >= float(args.min_correct_object_win_rate),
        "local_target_mean_gain_positive": float(local_correct["object"]["mean"]) > 0.0,
        "local_target_median_gain_positive": float(local_correct["object"]["median"])
        > 0.0,
        "correct_beats_every_corruption_mean": all(
            float(row["flow"]["object"]["mean"]) > 0.0
            for row in corrupt_comparisons.values()
        ),
        "correct_beats_every_corruption_median": all(
            float(row["flow"]["object"]["median"]) > 0.0
            for row in corrupt_comparisons.values()
        ),
        "corruption_object_win_rates": all(
            float(row["flow"]["object_win_rate"])
            >= float(args.min_corruption_object_win_rate)
            for row in corrupt_comparisons.values()
        ),
        "local_target_beats_every_corruption_mean": all(
            float(row["local_target"]["object"]["mean"]) > 0.0
            for row in corrupt_comparisons.values()
        ),
        "local_target_beats_every_corruption_median": all(
            float(row["local_target"]["object"]["median"]) > 0.0
            for row in corrupt_comparisons.values()
        ),
        "positive_t_count": positive_t_count >= int(args.min_positive_t_count),
    }
    report = {
        "stage": "PPD-3A local target residual paired teacher-forced evaluation",
        "passed": all(checks.values()),
        "args": vars(args),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_step": int(checkpoint.get("step", -1)),
        "probe": probe.metadata(),
        "flow": flow_schema,
        "physical_scale": float(args.physical_scale),
        "sample_count": count,
        "object_count": len({str(row["object_uid"]) for row in records}),
        "record_count": len(records),
        "noise_seeds": seeds,
        "t_values": t_values,
        "stock_preservation": {
            "physical_off_max_abs": zero_off_max_abs,
            "null_evidence_max_abs": null_max_abs,
            "neutral_max_abs": neutral_max_abs,
        },
        "comparisons": {
            "correct_vs_stock": correct_vs_stock,
            "local_correct_vs_zero": local_correct,
            "correct_vs_corruptions": corrupt_comparisons,
        },
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
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "checks": checks,
                "correct_vs_stock": correct_vs_stock,
                "positive_t_count": positive_t_count,
            },
            indent=2,
        ),
        flush=True,
    )
    if args.fail_on_decision and not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
