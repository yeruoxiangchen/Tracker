#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path
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

from reconvggt_ar_adapter_a.train_pointpose_ss_lora import (  # noqa: E402
    PointPoseCacheDataset,
    encode_frozen_features,
    rgba_images,
)
from reconvggt_ar_adapter_a.train_stock_preserving_pointpose_bridge import (  # noqa: E402
    build_models,
)


def _distribution_stats(values: torch.Tensor) -> dict[str, float | int]:
    flat = values.detach().float().reshape(-1)
    return {
        "numel": int(flat.numel()),
        "max": float(flat.max().item()),
        "mean": float(flat.mean().item()),
        "rmse": float(torch.sqrt(flat.square().mean()).item()),
    }


def _index_to_xyz(index: int, side: int) -> tuple[int, int, int]:
    x = int(index) // (int(side) ** 2)
    remainder = int(index) % (int(side) ** 2)
    y = remainder // int(side)
    z = remainder % int(side)
    return x, y, z


def _local_energy_ratio(
    energy: torch.Tensor,
    center: tuple[int, int, int],
    radius: int,
) -> float:
    x, y, z = center
    side = int(energy.shape[0])
    local = energy[
        max(0, x - radius) : min(side, x + radius + 1),
        max(0, y - radius) : min(side, y + radius + 1),
        max(0, z - radius) : min(side, z + radius + 1),
    ]
    return float((local.sum() / energy.sum().clamp_min(1.0e-20)).item())


def _sentinel_localization(
    *,
    flow: torch.nn.Module,
    x_t: torch.Tensor,
    t_tensor: torch.Tensor,
    cond_stock: torch.Tensor,
    baseline_velocity: torch.Tensor,
    token_index: int,
    channel_direction: torch.Tensor,
    perturb_scale: float,
    expected_min_percentile: float,
    max_argmax_distance: float,
    repeatability_rmse: float,
    min_signal_to_repeat_ratio: float,
) -> dict[str, Any]:
    side = int(flow.resolution)
    expected_xyz = _index_to_xyz(int(token_index), side)
    cond_rms = torch.sqrt(cond_stock.float().square().mean().clamp_min(1.0e-12))
    perturb = channel_direction * cond_rms * float(perturb_scale)
    cond_perturbed = cond_stock.clone()
    cond_perturbed[:, int(token_index), :] += perturb.to(cond_perturbed.dtype)
    with torch.no_grad():
        velocity = flow(x_t, t_tensor, cond_perturbed)
    absolute = (velocity.float() - baseline_velocity.float()).abs()
    spatial_rms = torch.sqrt(
        (velocity.float() - baseline_velocity.float()).square().mean(dim=1)
    )[0]
    energy = spatial_rms.square()
    flat = spatial_rms.reshape(-1)
    argmax_index = int(flat.argmax().item())
    argmax_xyz = _index_to_xyz(argmax_index, side)
    expected_value = float(spatial_rms[expected_xyz].item())
    expected_percentile = float((flat <= expected_value).float().mean().item())
    argmax_distance = float(
        math.sqrt(sum((a - b) ** 2 for a, b in zip(argmax_xyz, expected_xyz)))
    )
    coordinates = torch.stack(
        torch.meshgrid(
            torch.arange(side, device=energy.device, dtype=torch.float32),
            torch.arange(side, device=energy.device, dtype=torch.float32),
            torch.arange(side, device=energy.device, dtype=torch.float32),
            indexing="ij",
        ),
        dim=-1,
    )
    center_of_mass = (
        (coordinates * energy[..., None]).sum(dim=(0, 1, 2))
        / energy.sum().clamp_min(1.0e-20)
    )
    center_of_mass_xyz = [float(value) for value in center_of_mass.tolist()]
    center_of_mass_distance = float(
        math.sqrt(
            sum((value - expected) ** 2 for value, expected in zip(center_of_mass_xyz, expected_xyz))
        )
    )
    response_rmse = float(torch.sqrt(absolute.square().mean()).item())
    signal_to_repeat_ratio = response_rmse / max(float(repeatability_rmse), 1.0e-20)
    passed = bool(
        response_rmse > 0.0
        and signal_to_repeat_ratio >= float(min_signal_to_repeat_ratio)
        and expected_percentile >= float(expected_min_percentile)
        and argmax_distance <= float(max_argmax_distance)
    )
    return {
        "token_index": int(token_index),
        "expected_xyz": list(expected_xyz),
        "argmax_xyz": list(argmax_xyz),
        "argmax_distance": argmax_distance,
        "expected_response": expected_value,
        "expected_response_percentile": expected_percentile,
        "center_of_mass_xyz": center_of_mass_xyz,
        "center_of_mass_distance": center_of_mass_distance,
        "radius1_energy_ratio": _local_energy_ratio(energy, expected_xyz, 1),
        "radius2_energy_ratio": _local_energy_ratio(energy, expected_xyz, 2),
        "velocity_abs_diff": _distribution_stats(absolute),
        "signal_to_repeatability_rmse_ratio": signal_to_repeat_ratio,
        "passed": passed,
    }


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    permutation = report["condition_permutation"]
    lines = [
        "# ReconViaGen Bridge Condition Spatial-Mapping Audit",
        "",
        "This audit tests the 4096-token index-alignment hypothesis against the frozen SS Flow.",
        "Permutation invariance alone only tests row order; sentinel localization tests whether "
        "individual condition-token content has the claimed 16^3 locality.",
        "",
        f"- strict local16 gate: `{'PASS' if report['passed'] else 'FAIL'}`",
        f"- bridge condition shape: `{report['condition_shape']}`",
        f"- flow latent shape: `{report['flow_shape']}`",
        f"- random-permutation max abs diff: `{permutation['velocity_abs_diff']['max']:.8g}`",
        f"- same-input repeat RMSE: `{report['repeatability']['velocity_abs_diff']['rmse']:.8g}`",
        f"- permutation/median-sentinel RMSE ratio: "
        f"`{permutation['rmse_to_median_sentinel_ratio']:.8g}`",
        f"- row-order sensitive: `{permutation['row_order_sensitive']}`",
        "",
        "## Sentinel localization",
        "",
        "| token | expected xyz | argmax xyz | distance | expected percentile | r=1 energy | pass |",
        "| ---: | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for row in report["sentinels"]:
        lines.append(
            f"| {row['token_index']} | {row['expected_xyz']} | {row['argmax_xyz']} | "
            f"{row['argmax_distance']:.4f} | {row['expected_response_percentile']:.4f} | "
            f"{row['radius1_energy_ratio']:.6f} | {'PASS' if row['passed'] else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            "## Judgment",
            "",
            report["judgment"],
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit whether ReconViaGen's 4096 bridge-condition tokens have local 16^3 semantics."
    )
    parser.add_argument("--cache_manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--pretrained", default="Stable-X/trellis-vggt-v0-2")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--t", type=float, default=0.5)
    parser.add_argument("--perturb_scale", type=float, default=0.05)
    parser.add_argument("--sentinel_indices", default="0,1,16,256")
    parser.add_argument("--permutation_min_max_abs_diff", type=float, default=1.0e-5)
    parser.add_argument("--permutation_min_response_ratio", type=float, default=0.1)
    parser.add_argument("--min_signal_to_repeat_ratio", type=float, default=5.0)
    parser.add_argument("--sentinel_min_expected_percentile", type=float, default=0.95)
    parser.add_argument("--sentinel_max_argmax_distance", type=float, default=2.0)
    parser.add_argument(
        "--flow_precision",
        choices=["fp32", "checkpoint"],
        default="fp32",
        help="Use FP32 for the semantic audit to suppress low-precision reduction-order noise.",
    )
    parser.add_argument("--fail_on_audit", action="store_true")
    args = parser.parse_args()
    if not 0 < float(args.t) < 1:
        raise ValueError("t must be in (0,1)")
    if float(args.perturb_scale) <= 0:
        raise ValueError("perturb_scale must be positive")
    if args.flow_precision == "fp32" and os.environ.get(
        "ATTN_BACKEND", "flash_attn"
    ) == "flash_attn":
        raise RuntimeError(
            "flow_precision=fp32 requires ATTN_BACKEND=sdpa; "
            "FlashAttention only supports FP16/BF16"
        )

    sentinels = [
        int(item.strip()) for item in str(args.sentinel_indices).split(",") if item.strip()
    ]
    if sentinels != [0, 1, 16, 256]:
        raise ValueError("the strict audit requires sentinel_indices=0,1,16,256")

    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    device = torch.device(args.device)
    build_args = argparse.Namespace(
        pretrained=str(args.pretrained),
        gradient_checkpointing=False,
        bridge_fusion_mode="last1_cross_attention",
        bridge_last_blocks=1,
        physical_hidden_dim=32,
        physical_heads=8,
        fusion_stages="0,1,2",
        local_fusion_hidden_dim=32,
    )
    pipeline, model, _ = build_models(build_args, device)
    model.eval()
    dataset = PointPoseCacheDataset(args.cache_manifest, indices=str(args.index))
    sample = dataset[0]
    images = rgba_images(sample["image_paths"], sample["mask_paths"], pipeline)
    aggregated, image_cond = encode_frozen_features(pipeline, images)
    with torch.no_grad():
        cond_stock = model.bridge_fusion.stock_condition(aggregated, image_cond)

    flow = model.flow.eval()
    if int(cond_stock.shape[1]) != 16**3:
        raise RuntimeError(f"expected 4096 condition tokens, got {tuple(cond_stock.shape)}")
    if int(flow.resolution) != 16:
        raise RuntimeError(f"expected SS Flow resolution 16, got {flow.resolution}")
    checkpoint_flow_dtype = str(getattr(flow, "dtype", "unknown"))
    if args.flow_precision == "fp32":
        flow.convert_to_fp32()
    generator = torch.Generator(device=device).manual_seed(int(args.seed) + 991)
    x_t = torch.randn(
        1,
        int(flow.in_channels),
        int(flow.resolution),
        int(flow.resolution),
        int(flow.resolution),
        generator=generator,
        device=device,
        dtype=torch.float32,
    )
    t_tensor = torch.full((1,), 1000.0 * float(args.t), device=device, dtype=torch.float32)
    with torch.no_grad():
        baseline_velocity = flow(x_t, t_tensor, cond_stock)
        repeated_velocity = flow(x_t, t_tensor, cond_stock)
    repeatability_abs = (repeated_velocity.float() - baseline_velocity.float()).abs()
    repeatability_stats = _distribution_stats(repeatability_abs)

    permutation_generator = torch.Generator(device="cpu").manual_seed(int(args.seed) + 1777)
    permutation = torch.randperm(int(cond_stock.shape[1]), generator=permutation_generator).to(
        cond_stock.device
    )
    with torch.no_grad():
        permuted_velocity = flow(x_t, t_tensor, cond_stock[:, permutation])
    permutation_abs = (permuted_velocity.float() - baseline_velocity.float()).abs()

    direction_generator = torch.Generator(device=device).manual_seed(int(args.seed) + 31337)
    channel_direction = torch.randn(
        int(cond_stock.shape[-1]),
        generator=direction_generator,
        device=device,
        dtype=torch.float32,
    )
    channel_direction = channel_direction / torch.sqrt(
        channel_direction.square().mean().clamp_min(1.0e-12)
    )
    sentinel_rows = [
        _sentinel_localization(
            flow=flow,
            x_t=x_t,
            t_tensor=t_tensor,
            cond_stock=cond_stock,
            baseline_velocity=baseline_velocity,
            token_index=index,
            channel_direction=channel_direction,
            perturb_scale=float(args.perturb_scale),
            expected_min_percentile=float(args.sentinel_min_expected_percentile),
            max_argmax_distance=float(args.sentinel_max_argmax_distance),
            repeatability_rmse=float(repeatability_stats["rmse"]),
            min_signal_to_repeat_ratio=float(args.min_signal_to_repeat_ratio),
        )
        for index in sentinels
    ]
    sentinel_rmse = np.asarray(
        [float(row["velocity_abs_diff"]["rmse"]) for row in sentinel_rows],
        dtype=np.float64,
    )
    median_sentinel_rmse = float(np.median(sentinel_rmse))
    permutation_stats = _distribution_stats(permutation_abs)
    response_ratio = float(
        float(permutation_stats["rmse"]) / max(median_sentinel_rmse, 1.0e-20)
    )
    permutation_to_repeat_ratio = float(
        float(permutation_stats["rmse"])
        / max(float(repeatability_stats["rmse"]), 1.0e-20)
    )
    row_order_sensitive = bool(
        float(permutation_stats["max"]) >= float(args.permutation_min_max_abs_diff)
        and response_ratio >= float(args.permutation_min_response_ratio)
        and permutation_to_repeat_ratio >= float(args.min_signal_to_repeat_ratio)
    )
    sentinel_localization_passed = all(bool(row["passed"]) for row in sentinel_rows)
    passed = bool(row_order_sensitive and sentinel_localization_passed)
    judgment = (
        "PASS: both condition row order and the four sentinel perturbations support the proposed "
        "index-aligned 16^3 local-fusion hypothesis."
        if passed
        else "FAIL: 4096-token cardinality does not establish the proposed index-aligned 16^3 "
        "semantics. Do not run multistage_local16 overfit training; use content-based physical-to-visual "
        "cross-attention or another mapping that does not assume condition-row/voxel identity."
    )
    report = {
        "format": "reconvggt.reconbridge_condition_spatial_mapping.v1",
        "args": vars(args),
        "sample": {
            "index": int(args.index),
            "uid": str(sample["uid"]),
            "object_uid": str(sample["object_uid"]),
        },
        "condition_shape": list(cond_stock.shape),
        "flow_shape": list(baseline_velocity.shape),
        "precision": {
            "checkpoint_flow_dtype": checkpoint_flow_dtype,
            "audit_flow_dtype": str(getattr(flow, "dtype", "unknown")),
        },
        "repeatability": {
            "velocity_abs_diff": repeatability_stats,
            "exact": bool(torch.equal(repeated_velocity, baseline_velocity)),
        },
        "condition_permutation": {
            "velocity_abs_diff": permutation_stats,
            "median_sentinel_response_rmse": median_sentinel_rmse,
            "rmse_to_median_sentinel_ratio": response_ratio,
            "rmse_to_repeatability_ratio": permutation_to_repeat_ratio,
            "row_order_sensitive": row_order_sensitive,
            "note": (
                "Cross-attention can be permutation-invariant to KV row order even when token content "
                "contains identity information; the sentinel localization result is therefore reported "
                "separately and is the direct locality test."
            ),
        },
        "sentinels": sentinel_rows,
        "sentinel_localization_passed": sentinel_localization_passed,
        "passed": passed,
        "judgment": judgment,
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _write_markdown(output_dir / "report.md", report)
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    if args.fail_on_audit and not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
